"""Unit tests for Mailgun webhook route verification behavior."""

import hashlib
import hmac
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nornweave.core.config import get_settings
from nornweave.yggdrasil.dependencies import get_storage
from nornweave.yggdrasil.routes.webhooks import mailgun


def _mailgun_signature(signing_key: str, timestamp: int, token: str) -> str:
    signed_payload = f"{timestamp}{token}".encode()
    return hmac.new(signing_key.encode("utf-8"), signed_payload, hashlib.sha256).hexdigest()


def _make_app(signing_key: str) -> FastAPI:
    app = FastAPI()
    app.include_router(mailgun.router, prefix="/webhooks")
    app.dependency_overrides[get_storage] = lambda: AsyncMock()
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        mailgun_api_key="key-test",
        mailgun_domain="mail.example.com",
        webhook_secret=signing_key,
    )
    return app


def _mailgun_form(
    *, signing_key: str, token: str = "token-123", signature: str | None = None
) -> dict[str, str]:
    timestamp = int(time.time())
    signed = signature or _mailgun_signature(signing_key, timestamp, token)
    return {
        "sender": "alice@gmail.com",
        "from": "Alice Smith <alice@gmail.com>",
        "recipient": "support@example.com",
        "subject": "Test subject",
        "body-plain": "Hello",
        "timestamp": str(timestamp),
        "token": token,
        "signature": signed,
    }


@pytest.mark.unit
class TestMailgunWebhookRouteVerification:
    """Ensure webhook verification is enforced before parsing and ingestion."""

    def test_rejects_invalid_signature_before_parse_and_ingest(self) -> None:
        app = _make_app("mailgun-signing-key")
        client = TestClient(app)
        form_data = _mailgun_form(signing_key="mailgun-signing-key", signature="invalid-signature")

        with (
            patch(
                "nornweave.yggdrasil.routes.webhooks.mailgun.MailgunAdapter.parse_inbound_webhook"
            ) as mock_parse,
            patch(
                "nornweave.yggdrasil.routes.webhooks.mailgun.ingest_message",
                new_callable=AsyncMock,
            ) as mock_ingest,
        ):
            response = client.post("/webhooks/mailgun", data=form_data)

        assert response.status_code == 401
        assert response.json()["detail"] == "Signature verification failed"
        mock_parse.assert_not_called()
        mock_ingest.assert_not_awaited()

    def test_rejects_when_signing_key_not_configured(self) -> None:
        app = _make_app("")
        client = TestClient(app)
        form_data = _mailgun_form(signing_key="fallback-key")

        with (
            patch(
                "nornweave.yggdrasil.routes.webhooks.mailgun.MailgunAdapter.parse_inbound_webhook"
            ) as mock_parse,
            patch(
                "nornweave.yggdrasil.routes.webhooks.mailgun.ingest_message",
                new_callable=AsyncMock,
            ) as mock_ingest,
        ):
            response = client.post("/webhooks/mailgun", data=form_data)

        assert response.status_code == 503
        assert response.json()["detail"] == "Mailgun webhook verification is not configured"
        mock_parse.assert_not_called()
        mock_ingest.assert_not_awaited()

    def test_accepts_valid_signature_and_processes_webhook(self) -> None:
        app = _make_app("mailgun-signing-key")
        client = TestClient(app)
        form_data = _mailgun_form(signing_key="mailgun-signing-key")

        with patch(
            "nornweave.yggdrasil.routes.webhooks.mailgun.ingest_message",
            new_callable=AsyncMock,
        ) as mock_ingest:
            mock_ingest.return_value = SimpleNamespace(
                status="received",
                message_id="msg-123",
                thread_id="th-456",
            )

            response = client.post("/webhooks/mailgun", data=form_data)

        assert response.status_code == 200
        assert response.json() == {
            "status": "received",
            "message_id": "msg-123",
            "thread_id": "th-456",
        }
        mock_ingest.assert_awaited_once()


@pytest.mark.unit
class TestMailgunWebhookAttachments:
    """Attachment file parts must be extracted and passed to ingestion."""

    def test_attachments_extracted_from_multipart(self) -> None:
        app = _make_app("mailgun-signing-key")
        client = TestClient(app)
        form_data = _mailgun_form(signing_key="mailgun-signing-key")
        form_data["attachment-count"] = "2"
        form_data["content-id-map"] = '{"<logo-cid>": "attachment-2"}'

        with patch(
            "nornweave.yggdrasil.routes.webhooks.mailgun.ingest_message",
            new_callable=AsyncMock,
        ) as mock_ingest:
            mock_ingest.return_value = SimpleNamespace(
                status="received", message_id="msg-1", thread_id="th-1"
            )
            response = client.post(
                "/webhooks/mailgun",
                data=form_data,
                files=[
                    ("attachment-1", ("report.pdf", b"%PDF-1.4 fake", "application/pdf")),
                    ("attachment-2", ("logo.png", b"\x89PNG fake", "image/png")),
                ],
            )

        assert response.status_code == 200
        inbound = mock_ingest.await_args.args[0]
        assert len(inbound.attachments) == 2

        pdf = inbound.attachments[0]
        assert pdf.filename == "report.pdf"
        assert pdf.content_type == "application/pdf"
        assert pdf.content == b"%PDF-1.4 fake"
        assert pdf.size_bytes == len(b"%PDF-1.4 fake")
        assert pdf.disposition.value == "attachment"
        assert pdf.content_id is None

        logo = inbound.attachments[1]
        assert logo.filename == "logo.png"
        assert logo.content_id == "logo-cid"
        assert logo.disposition.value == "inline"

    def test_no_attachments_yields_empty_list(self) -> None:
        app = _make_app("mailgun-signing-key")
        client = TestClient(app)
        form_data = _mailgun_form(signing_key="mailgun-signing-key")

        with patch(
            "nornweave.yggdrasil.routes.webhooks.mailgun.ingest_message",
            new_callable=AsyncMock,
        ) as mock_ingest:
            mock_ingest.return_value = SimpleNamespace(
                status="received", message_id="msg-1", thread_id="th-1"
            )
            response = client.post("/webhooks/mailgun", data=form_data)

        assert response.status_code == 200
        inbound = mock_ingest.await_args.args[0]
        assert inbound.attachments == []

    def test_malformed_count_and_map_are_tolerated(self) -> None:
        app = _make_app("mailgun-signing-key")
        client = TestClient(app)
        form_data = _mailgun_form(signing_key="mailgun-signing-key")
        form_data["attachment-count"] = "not-a-number"
        form_data["content-id-map"] = "{broken json"

        with patch(
            "nornweave.yggdrasil.routes.webhooks.mailgun.ingest_message",
            new_callable=AsyncMock,
        ) as mock_ingest:
            mock_ingest.return_value = SimpleNamespace(
                status="received", message_id="msg-1", thread_id="th-1"
            )
            response = client.post("/webhooks/mailgun", data=form_data)

        assert response.status_code == 200
        assert mock_ingest.await_args.args[0].attachments == []
