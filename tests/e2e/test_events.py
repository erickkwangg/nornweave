"""E2E tests for the events layer: emission through real send/receive flows."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING

from nornweave.core.config import Settings
from nornweave.core.interfaces import InboundMessage
from nornweave.verdandi.ingest import ingest_message

if TYPE_CHECKING:
    from httpx import AsyncClient

    from nornweave.urdr.adapters.sqlite import SQLiteAdapter


async def _create_inbox(e2e_client: AsyncClient, username: str) -> str:
    response = await e2e_client.post(
        "/v1/inboxes",
        json={"name": username.capitalize(), "email_username": username},
    )
    assert response.status_code in (200, 201)
    return str(response.json()["id"])


class TestEventEmission:
    """Events appear when mail is sent and received."""

    async def test_send_emits_message_sent(self, e2e_client: AsyncClient) -> None:
        inbox_id = await _create_inbox(e2e_client, "events-sender")

        response = await e2e_client.post(
            "/v1/messages",
            json={
                "inbox_id": inbox_id,
                "to": ["human@example.com"],
                "subject": "Event check",
                "body": "Hello",
            },
        )
        assert response.status_code in (200, 201)

        events = (await e2e_client.get(f"/v1/events?inbox_id={inbox_id}")).json()
        types = [item["type"] for item in events["items"]]
        assert "message.sent" in types

    async def test_ingest_emits_message_received(
        self,
        e2e_client: AsyncClient,
        e2e_storage: SQLiteAdapter,
    ) -> None:
        receiver_id = await _create_inbox(e2e_client, "events-b")

        settings = Settings(
            environment="test",
            db_driver="sqlite",
            email_provider="mailgun",
            email_domain="test.nornweave.local",
            api_key="test-api-key",
        )
        inbound = InboundMessage(
            from_address="human@example.com",
            to_address="events-b@test.nornweave.local",
            subject="Loopback",
            body_plain="Ping",
            message_id="<events-e2e-1@example.com>",
            timestamp=datetime.now(UTC),
        )
        result = await ingest_message(inbound, e2e_storage, settings)
        assert result.status == "received"

        events = (await e2e_client.get(f"/v1/events?inbox_id={receiver_id}")).json()
        types = [item["type"] for item in events["items"]]
        assert "message.received" in types
        received = next(item for item in events["items"] if item["type"] == "message.received")
        assert received["payload"]["subject"] == "Loopback"
        assert received["thread_id"]
        assert received["message_id"]

    async def test_event_type_filter(self, e2e_client: AsyncClient) -> None:
        inbox_id = await _create_inbox(e2e_client, "events-filter")

        await e2e_client.post(
            "/v1/messages",
            json={
                "inbox_id": inbox_id,
                "to": ["human@example.com"],
                "subject": "Filter check",
                "body": "Hello",
            },
        )

        sent_only = (
            await e2e_client.get(f"/v1/events?inbox_id={inbox_id}&event_type=message.sent")
        ).json()
        assert sent_only["count"] >= 1
        assert all(item["type"] == "message.sent" for item in sent_only["items"])

        received_only = (
            await e2e_client.get(f"/v1/events?inbox_id={inbox_id}&event_type=message.received")
        ).json()
        assert received_only["count"] == 0
