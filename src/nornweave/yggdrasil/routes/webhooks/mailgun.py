"""Mailgun webhook handler."""

import json
import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from starlette.datastructures import FormData, UploadFile

from nornweave.adapters.mailgun import MailgunAdapter, MailgunWebhookError
from nornweave.core.config import Settings, get_settings
from nornweave.core.interfaces import (
    InboundAttachment,
    StorageInterface,
)
from nornweave.models.attachment import AttachmentDisposition
from nornweave.verdandi.ingest import ingest_message
from nornweave.yggdrasil.dependencies import get_storage

router = APIRouter()
logger = logging.getLogger(__name__)


async def _extract_attachments(form_data: FormData) -> list[InboundAttachment]:
    """Read attachment-N file parts from Mailgun's multipart payload.

    Mailgun posts attachment-count plus attachment-1..N file fields, and a
    content-id-map JSON object ({"<cid>": "attachment-N"}) for inline parts.
    """
    content_id_by_field: dict[str, str] = {}
    raw_map = form_data.get("content-id-map")
    if isinstance(raw_map, str) and raw_map:
        try:
            content_id_by_field = {
                field_name: str(cid).strip("<>") for cid, field_name in json.loads(raw_map).items()
            }
        except json.JSONDecodeError, AttributeError:
            logger.warning("Ignoring malformed content-id-map: %s", raw_map)

    try:
        count = int(str(form_data.get("attachment-count") or 0))
    except ValueError:
        count = 0

    attachments: list[InboundAttachment] = []
    for i in range(1, count + 1):
        field_name = f"attachment-{i}"
        upload = form_data.get(field_name)
        if not isinstance(upload, UploadFile):
            continue
        content = await upload.read()
        content_id = content_id_by_field.get(field_name)
        attachments.append(
            InboundAttachment(
                filename=upload.filename or field_name,
                content_type=upload.content_type or "application/octet-stream",
                content=content,
                size_bytes=len(content),
                disposition=(
                    AttachmentDisposition.INLINE if content_id else AttachmentDisposition.ATTACHMENT
                ),
                content_id=content_id,
            )
        )
    return attachments


def _get_mailgun_adapter(settings: Settings) -> MailgunAdapter:
    """Create MailgunAdapter with webhook verification config."""
    return MailgunAdapter(
        api_key=settings.mailgun_api_key,
        domain=settings.mailgun_domain,
        webhook_signing_key=settings.webhook_secret,
    )


@router.post("/mailgun", status_code=status.HTTP_200_OK)
async def mailgun_webhook(
    request: Request,
    storage: StorageInterface = Depends(get_storage),
    settings: Settings = Depends(get_settings),
) -> dict[str, str]:
    """Handle inbound email webhook from Mailgun.

    Mailgun sends inbound emails as multipart/form-data.
    This handler:
    1. Parses the webhook payload
    2. Delegates to the shared ingestion pipeline
    """
    # Parse form data from Mailgun
    form_data = await request.form()
    payload = dict(form_data.items())

    logger.info("Received Mailgun webhook for recipient: %s", payload.get("recipient"))
    logger.debug("Mailgun payload keys: %s", list(payload.keys()))

    # Enforce webhook signature verification before parsing/ingestion.
    if not settings.webhook_secret:
        logger.error("WEBHOOK_SECRET is not configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Mailgun webhook verification is not configured",
        )

    adapter = _get_mailgun_adapter(settings)
    try:
        adapter.verify_webhook_signature(payload)
    except MailgunWebhookError as e:
        logger.warning("Mailgun webhook signature verification failed: %s", e)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=str(e),
        ) from e

    # Parse the webhook payload using the Mailgun adapter
    try:
        inbound = adapter.parse_inbound_webhook(payload)
    except Exception as e:
        logger.error("Failed to parse Mailgun webhook: %s", e)
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Failed to parse webhook payload: {e}",
        ) from e

    # Attachment file parts need async reads, so they are extracted here
    # rather than in the adapter's sync parse.
    inbound.attachments = await _extract_attachments(form_data)

    # Delegate to shared ingestion pipeline
    result = await ingest_message(inbound, storage, settings)

    return {
        "status": result.status,
        "message_id": result.message_id,
        "thread_id": result.thread_id,
    }
