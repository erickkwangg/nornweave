"""Message endpoints."""

import base64
import binascii
import logging
import uuid
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel, Field

from nornweave.core.config import Settings, get_settings
from nornweave.core.domain_filter import DomainFilter
from nornweave.core.interfaces import (
    EmailProvider,
    InboundMessage,
    StorageInterface,
)
from nornweave.core.storage import AttachmentMetadata, create_attachment_storage
from nornweave.models.attachment import AttachmentUpload, SendAttachment
from nornweave.models.message import Message, MessageDirection
from nornweave.models.thread import Thread
from nornweave.skuld.rate_limiter import GlobalRateLimiter  # noqa: TC001 - needed at runtime
from nornweave.verdandi.ingest import ingest_message
from nornweave.verdandi.summarize import generate_thread_summary
from nornweave.yggdrasil.dependencies import get_email_provider, get_rate_limiter, get_storage

logger = logging.getLogger(__name__)

router = APIRouter()


class MessageResponse(BaseModel):
    """Response model for a message with all email metadata fields."""

    id: str
    thread_id: str
    inbox_id: str
    direction: str
    provider_message_id: str | None
    subject: str | None = None
    from_address: str | None = None
    to_addresses: list[str] = Field(default_factory=list)
    cc_addresses: list[str] | None = None
    bcc_addresses: list[str] | None = None
    reply_to_addresses: list[str] | None = None
    text: str | None = None
    html: str | None = None
    content_clean: str = ""
    timestamp: datetime | None = None
    labels: list[str] = Field(default_factory=list)
    preview: str | None = None
    size: int = 0
    in_reply_to: str | None = None
    references: list[str] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime | None = None


class MessageListResponse(BaseModel):
    """Response model for message list with pagination support."""

    items: list[MessageResponse]
    count: int
    total: int


class SendMessageRequest(BaseModel):
    """Request to send an outbound message."""

    inbox_id: str
    to: list[str] = Field(..., min_length=1)
    subject: str = Field(..., min_length=1)
    body: str = Field(..., description="Markdown body content")
    reply_to_thread_id: str | None = None
    cc: list[str] | None = Field(None, description="CC recipients")
    bcc: list[str] | None = Field(None, description="BCC recipients")
    reply_to: str | None = Field(None, description="Reply-To address")
    html_body: str | None = Field(
        None, description="Pre-rendered HTML body (otherwise body is converted from Markdown)"
    )
    attachments: list[AttachmentUpload] | None = Field(
        None, description="Optional list of attachments to send"
    )


class SendMessageResponse(BaseModel):
    """Response after sending a message."""

    id: str
    thread_id: str
    provider_message_id: str | None
    status: str
    error: str | None = None


def _rfc_msgid(raw: str | None) -> str | None:
    """Best-effort RFC 5322 Message-ID for threading headers.

    Stored provider ids from inbound mail carry angle brackets already; some
    provider APIs return bare ids (Resend returns a UUID that is not a wire
    Message-ID at all, so ids without an @ are unusable and dropped).
    """
    if not raw:
        return None
    if raw.startswith("<") and raw.endswith(">"):
        return raw
    if "@" in raw:
        return f"<{raw}>"
    return None


def _build_reply_headers(thread_messages: list[Message]) -> tuple[str | None, list[str] | None]:
    """Derive In-Reply-To and References from the thread's latest addressable message."""
    for msg in reversed(thread_messages):
        parent_id = _rfc_msgid(msg.provider_message_id)
        if parent_id is None:
            continue
        references: list[str] = []
        for ref in msg.references or []:
            ref_id = _rfc_msgid(ref)
            if ref_id and ref_id not in references:
                references.append(ref_id)
        if parent_id not in references:
            references.append(parent_id)
        return parent_id, references
    return None, None


def _message_to_response(msg: Message) -> MessageResponse:
    """Convert Message model to response with all email metadata fields."""
    return MessageResponse(
        id=msg.id,
        thread_id=msg.thread_id,
        inbox_id=msg.inbox_id,
        direction=msg.direction.value,
        provider_message_id=msg.provider_message_id,
        subject=msg.subject,
        from_address=msg.from_address,
        to_addresses=msg.to if msg.to else [],
        cc_addresses=msg.cc,
        bcc_addresses=msg.bcc,
        reply_to_addresses=msg.reply_to,
        text=msg.text,
        html=msg.html,
        content_clean=msg.content_clean or "",
        timestamp=msg.timestamp,
        labels=msg.labels if msg.labels else [],
        preview=msg.preview,
        size=msg.size,
        in_reply_to=msg.in_reply_to,
        references=msg.references,
        metadata=msg.metadata if msg.metadata else {},
        created_at=msg.created_at,
    )


@router.get("/messages", response_model=MessageListResponse)
async def list_messages(
    inbox_id: str | None = None,
    thread_id: str | None = None,
    q: str | None = None,
    limit: int = 50,
    offset: int = 0,
    storage: StorageInterface = Depends(get_storage),
) -> MessageListResponse:
    """
    List and search messages with flexible filters.

    At least one of inbox_id or thread_id must be provided.
    Optional text search (q) searches across subject, body, sender, and attachment filenames.
    """
    # Validate at least one filter is provided
    if inbox_id is None and thread_id is None:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="At least one filter (inbox_id or thread_id) is required",
        )

    # Verify inbox exists if provided
    if inbox_id:
        inbox = await storage.get_inbox(inbox_id)
        if inbox is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Inbox {inbox_id} not found",
            )

    # Verify thread exists if provided
    if thread_id:
        thread = await storage.get_thread(thread_id)
        if thread is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {thread_id} not found",
            )

    # Use advanced search method with all filters
    messages, total = await storage.search_messages_advanced(
        inbox_id=inbox_id,
        thread_id=thread_id,
        query=q,
        limit=limit,
        offset=offset,
    )

    return MessageListResponse(
        items=[_message_to_response(m) for m in messages],
        count=len(messages),
        total=total,
    )


@router.get("/messages/{message_id}", response_model=MessageResponse)
async def get_message(
    message_id: str,
    storage: StorageInterface = Depends(get_storage),
) -> MessageResponse:
    """Get a message by ID."""
    message = await storage.get_message(message_id)
    if message is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Message {message_id} not found",
        )
    return _message_to_response(message)


@router.post("/messages", response_model=SendMessageResponse, status_code=status.HTTP_201_CREATED)
async def send_message(
    payload: SendMessageRequest,
    storage: StorageInterface = Depends(get_storage),
    email_provider: EmailProvider = Depends(get_email_provider),
    settings: Settings = Depends(get_settings),
    rate_limiter: GlobalRateLimiter = Depends(get_rate_limiter),
) -> SendMessageResponse:
    """Send an outbound message.

    If reply_to_thread_id is provided, the message is added to that thread.
    Otherwise, a new thread is created.

    Supports optional attachments which are stored and sent with the email.
    """
    # Get inbox
    inbox = await storage.get_inbox(payload.inbox_id)
    if inbox is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Inbox {payload.inbox_id} not found",
        )

    # Outbound domain filtering (allow/blocklist) across all recipient fields
    outbound_filter = DomainFilter(
        allowlist=settings.outbound_domain_allowlist,
        blocklist=settings.outbound_domain_blocklist,
        direction="outbound",
    )
    blocked_domains: list[str] = []
    for recipient in [*payload.to, *(payload.cc or []), *(payload.bcc or [])]:
        if not outbound_filter.check(recipient):
            _, _, domain = recipient.rpartition("@")
            blocked_domains.append(domain or recipient)
    if blocked_domains:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=(
                f"Recipient domain(s) blocked by outbound policy: "
                f"{', '.join(sorted(set(blocked_domains)))}"
            ),
        )

    # Global send rate limiting (check before provider call)
    rl_result = rate_limiter.check()
    if not rl_result.allowed:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=rl_result.detail,
            headers={"Retry-After": str(rate_limiter.retry_after_header(rl_result))},
        )

    # Get or create thread; for replies, derive RFC threading headers from the
    # thread history so recipients' mail clients thread the reply correctly.
    thread_id: str
    reply_in_reply_to: str | None = None
    reply_references: list[str] | None = None
    if payload.reply_to_thread_id:
        thread = await storage.get_thread(payload.reply_to_thread_id)
        if thread is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Thread {payload.reply_to_thread_id} not found",
            )
        thread_id = thread.id
        thread_messages = await storage.list_messages_for_thread(thread_id)
        reply_in_reply_to, reply_references = _build_reply_headers(thread_messages)
    else:
        # Create a new thread
        new_thread = Thread(
            thread_id=str(uuid.uuid4()),
            inbox_id=payload.inbox_id,
            subject=payload.subject,
            timestamp=datetime.now(UTC),
            participant_hash=None,  # Will be set when we have participants
        )
        created_thread = await storage.create_thread(new_thread)
        thread_id = created_thread.id

    # Generate message ID early so we can link attachments
    message_id = str(uuid.uuid4())

    # Process attachments if provided
    attachment_records: list[dict[str, Any]] = []
    provider_attachments: list[SendAttachment] = []

    if payload.attachments:
        # Create storage backend
        storage_backend = create_attachment_storage(settings)

        for i, attachment in enumerate(payload.attachments):
            # Validate and decode base64 content
            try:
                content_bytes = base64.b64decode(attachment.content_base64)
            except (binascii.Error, ValueError) as e:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Invalid base64 content in attachment {i}: {e}",
                )

            if len(content_bytes) == 0:
                raise HTTPException(
                    status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                    detail=f"Attachment {i} has empty content",
                )

            # Generate attachment ID and store
            attachment_id = str(uuid.uuid4())

            metadata = AttachmentMetadata(
                attachment_id=attachment_id,
                message_id=message_id,
                filename=attachment.filename,
                content_type=attachment.content_type,
                content_disposition=attachment.disposition.value,
                content_id=attachment.content_id,
            )

            # Store in configured backend
            storage_result = await storage_backend.store(
                attachment_id=attachment_id,
                content=content_bytes,
                metadata=metadata,
            )

            # Prepare attachment record for database
            attachment_records.append(
                {
                    "attachment_id": attachment_id,
                    "filename": attachment.filename,
                    "content_type": attachment.content_type,
                    "size_bytes": storage_result.size_bytes,
                    "disposition": attachment.disposition.value,
                    "content_id": attachment.content_id,
                    "storage_path": storage_result.storage_key,
                    "storage_backend": storage_result.backend,
                    "content_hash": storage_result.content_hash,
                    # For database backend, also store content
                    "content": content_bytes if storage_result.backend == "database" else None,
                }
            )

            # Prepare attachment for email provider
            provider_attachments.append(
                SendAttachment(
                    filename=attachment.filename,
                    content_type=attachment.content_type,
                    content_disposition=attachment.disposition,
                    content_id=attachment.content_id,
                    content=attachment.content_base64,  # Keep as base64 for provider
                )
            )

    # Send email via provider
    # On failure the message is still recorded (status="failed") so the
    # caller can see it and retry, instead of silently losing the attempt.
    provider_message_id: str | None = None
    send_error: str | None = None
    try:
        provider_message_id = await email_provider.send_email(
            to=payload.to,
            subject=payload.subject,
            body=payload.body,
            from_address=inbox.email_address,
            reply_to=payload.reply_to,
            in_reply_to=reply_in_reply_to,
            references=reply_references,
            cc=payload.cc,
            bcc=payload.bcc,
            html_body=payload.html_body,
            attachments=provider_attachments if provider_attachments else None,
        )
    except Exception:
        logger.exception("Failed to send email via %s provider", settings.email_provider)
        send_error = (
            f"Email provider ({settings.email_provider}) failed to send the message. "
            "The message has been recorded but was not delivered. "
            "Check server logs for details."
        )

    # Record successful send in rate limiter (only when provider returned an id)
    if provider_message_id is not None:
        rate_limiter.record()

    # Create message record
    message = Message(
        message_id=message_id,
        thread_id=thread_id,
        inbox_id=payload.inbox_id,
        provider_message_id=provider_message_id,
        direction=MessageDirection.OUTBOUND,
        text=payload.body,
        extracted_text=payload.body,  # Already markdown
        html=payload.html_body,
        cc=payload.cc,
        bcc=payload.bcc,
        reply_to=[payload.reply_to] if payload.reply_to else None,
        in_reply_to=reply_in_reply_to,
        references=reply_references,
        headers={
            "to": ",".join(payload.to),  # Join list into comma-separated string
            "subject": payload.subject,
        },
        created_at=datetime.now(UTC),
    )
    created_message = await storage.create_message(message)

    # Create attachment records linked to the message
    for rec in attachment_records:
        await storage.create_attachment(
            message_id=created_message.id,
            filename=rec["filename"],
            content_type=rec["content_type"],
            size_bytes=rec["size_bytes"],
            disposition=rec["disposition"],
            content_id=rec["content_id"],
            storage_path=rec["storage_path"],
            storage_backend=rec["storage_backend"],
            content_hash=rec["content_hash"],
            content=rec["content"],
        )

    # Update thread's last_message_at
    thread = await storage.get_thread(thread_id)
    if thread:
        thread.last_message_at = created_message.created_at
        await storage.update_thread(thread)

    # Fire-and-forget thread summarization
    await generate_thread_summary(storage, thread_id)

    # Demo mode loopback: deliver a copy to each recipient that is a demo inbox
    if settings.email_provider == "demo" and provider_message_id:
        for recipient in [*payload.to, *(payload.cc or [])]:
            recipient_inbox = await storage.get_inbox_by_email(recipient)
            if recipient_inbox is None:
                continue
            inbound = InboundMessage(
                from_address=inbox.email_address,
                to_address=recipient,
                subject=payload.subject,
                body_plain=payload.body,
                message_id=provider_message_id,
                in_reply_to=reply_in_reply_to,
                references=reply_references or [],
                timestamp=datetime.now(UTC),
            )
            await ingest_message(inbound, storage, settings)

    if send_error:
        send_status = "failed"
    elif provider_message_id:
        send_status = "sent"
    else:
        send_status = "pending"

    return SendMessageResponse(
        id=created_message.id,
        thread_id=thread_id,
        provider_message_id=provider_message_id,
        status=send_status,
        error=send_error,
    )
