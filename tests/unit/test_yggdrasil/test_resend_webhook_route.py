"""Unit tests for Resend webhook route verification behavior."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from nornweave.core.config import get_settings
from nornweave.yggdrasil.dependencies import get_storage
from nornweave.yggdrasil.routes.webhooks import resend


def _make_app(webhook_secret: str | None) -> FastAPI:
    app = FastAPI()
    app.include_router(resend.router, prefix="/webhooks")
    app.dependency_overrides[get_storage] = lambda: AsyncMock()
    app.dependency_overrides[get_settings] = lambda: SimpleNamespace(
        resend_api_key="re_test_key",
        resend_webhook_secret=webhook_secret,
    )
    return app


def _inbound_payload() -> dict[str, object]:
    return {
        "type": "email.received",
        "data": {
            "email_id": "em_123",
            "from": "alice@gmail.com",
            "to": ["support@example.com"],
            "subject": "Test subject",
        },
    }


@pytest.mark.unit
class TestResendWebhookRouteVerification:
    """Signature verification must fail closed, matching the Mailgun route."""

    def test_rejects_when_secret_not_configured(self) -> None:
        app = _make_app(webhook_secret=None)
        client = TestClient(app)

        response = client.post("/webhooks/resend", json=_inbound_payload())

        assert response.status_code == 503
        assert "not configured" in response.json()["detail"]

    def test_rejects_invalid_signature(self) -> None:
        app = _make_app(webhook_secret="whsec_" + "a" * 32)
        client = TestClient(app)

        response = client.post(
            "/webhooks/resend",
            json=_inbound_payload(),
            headers={
                "svix-id": "msg_123",
                "svix-timestamp": "1700000000",
                "svix-signature": "v1,invalid",
            },
        )

        assert response.status_code == 401

    def test_rejects_missing_signature_headers(self) -> None:
        app = _make_app(webhook_secret="whsec_" + "a" * 32)
        client = TestClient(app)

        response = client.post("/webhooks/resend", json=_inbound_payload())

        assert response.status_code == 401
