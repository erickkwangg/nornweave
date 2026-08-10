"""Unit tests for event emission and the SSE broker."""

from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from typing import Any

from nornweave.models.event import Event, EventType
from nornweave.verdandi.events import EventBroker, emit_event


def _event(event_id: str = "evt-1") -> Event:
    return Event(
        id=event_id,
        type=EventType.MESSAGE_RECEIVED,
        created_at=datetime.now(UTC),
        inbox_id="inbox-1",
        thread_id="thread-1",
        message_id="msg-1",
        payload={"subject": "hello"},
    )


class StubStorage:
    """Storage stub that assigns ids like the real adapter."""

    def __init__(self) -> None:
        self.created: list[Event] = []

    async def create_event(self, event: Event) -> Event:
        stored = event.model_copy(update={"id": event.id or "generated-id"})
        self.created.append(stored)
        return stored


class TestEventBroker:
    """Tests for the in-process pub/sub broker."""

    def test_publish_reaches_all_subscribers(self) -> None:
        broker = EventBroker()
        first = broker.subscribe()
        second = broker.subscribe()

        broker.publish(_event())

        assert first.get_nowait().id == "evt-1"
        assert second.get_nowait().id == "evt-1"

    def test_unsubscribe_stops_delivery(self) -> None:
        broker = EventBroker()
        queue = broker.subscribe()
        broker.unsubscribe(queue)

        broker.publish(_event())

        assert queue.empty()
        assert broker.subscriber_count == 0

    def test_full_queue_drops_instead_of_raising(self) -> None:
        broker = EventBroker()
        queue = broker.subscribe()
        for i in range(queue.maxsize):
            broker.publish(_event(f"evt-{i}"))

        broker.publish(_event("evt-overflow"))

        assert queue.qsize() == queue.maxsize


class TestEmitEvent:
    """Tests for emit_event persistence and fan-out."""

    async def test_persists_and_publishes(self, monkeypatch: Any) -> None:
        import nornweave.verdandi.events as events_module

        broker = EventBroker()
        monkeypatch.setattr(events_module, "broker", broker)
        queue = broker.subscribe()
        storage = StubStorage()

        stored = await emit_event(
            storage,  # type: ignore[arg-type]
            event_type=EventType.MESSAGE_SENT,
            inbox_id="inbox-1",
            thread_id="thread-1",
            message_id="msg-1",
            payload={"subject": "hi"},
        )

        assert stored.id == "generated-id"
        assert storage.created[0].type == EventType.MESSAGE_SENT
        assert queue.get_nowait().id == "generated-id"

    async def test_storage_failure_propagates(self) -> None:
        class FailingStorage:
            async def create_event(self, _event: Event) -> Event:
                raise RuntimeError("db down")

        try:
            await emit_event(
                FailingStorage(),  # type: ignore[arg-type]
                event_type=EventType.MESSAGE_SENT,
            )
            raise AssertionError("expected RuntimeError")
        except RuntimeError:
            pass

    async def test_slow_subscriber_does_not_block_emission(self, monkeypatch: Any) -> None:
        import nornweave.verdandi.events as events_module

        broker = EventBroker()
        monkeypatch.setattr(events_module, "broker", broker)
        queue = broker.subscribe()
        storage = StubStorage()
        for _ in range(queue.maxsize):
            broker.publish(_event())

        stored = await asyncio.wait_for(
            emit_event(storage, event_type=EventType.MESSAGE_RECEIVED),  # type: ignore[arg-type]
            timeout=1,
        )

        assert stored.id == "generated-id"
