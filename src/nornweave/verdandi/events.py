"""Event emission: persist to storage and fan out to live subscribers.

Single owner of event creation. Everything that makes something happen
(ingest, sending) calls emit_event; consumers read events back via
GET /v1/events or the SSE stream, which subscribes to the in-process broker.
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING, Any

from nornweave.models.event import Event, EventType

if TYPE_CHECKING:
    from nornweave.core.interfaces import StorageInterface

logger = logging.getLogger(__name__)


class EventBroker:
    """In-process pub/sub for server-sent events.

    Subscribers get a bounded queue; slow consumers drop events rather than
    blocking emission. Single-process only, which matches the uvicorn
    deployment; a multi-process deployment needs an external broker.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Event]] = set()

    def subscribe(self) -> asyncio.Queue[Event]:
        queue: asyncio.Queue[Event] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Event]) -> None:
        self._subscribers.discard(queue)

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)

    def publish(self, event: Event) -> None:
        for queue in list(self._subscribers):
            try:
                queue.put_nowait(event)
            except asyncio.QueueFull:
                logger.warning("SSE subscriber queue full, dropping event %s", event.id)


broker = EventBroker()


async def emit_event(
    storage: StorageInterface,
    *,
    event_type: EventType,
    inbox_id: str | None = None,
    thread_id: str | None = None,
    message_id: str | None = None,
    payload: dict[str, Any] | None = None,
) -> Event:
    """Persist an event and notify live subscribers.

    Emission must never break the operation that triggered it; storage
    errors are logged and re-raised only if persistence itself fails,
    while broker fan-out is best effort.
    """
    event = Event(
        id="",
        type=event_type,
        inbox_id=inbox_id,
        thread_id=thread_id,
        message_id=message_id,
        payload=payload or {},
    )
    stored = await storage.create_event(event)
    broker.publish(stored)
    logger.debug("Emitted event %s (%s)", stored.id, stored.type)
    return stored
