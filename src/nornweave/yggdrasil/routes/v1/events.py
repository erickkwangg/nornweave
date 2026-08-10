"""Event endpoints: polling list and server-sent event stream."""

from __future__ import annotations

import asyncio
from datetime import datetime  # noqa: TC003 - needed at runtime for Pydantic
from typing import TYPE_CHECKING, Any

from fastapi import APIRouter, Depends, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from nornweave.core.interfaces import StorageInterface  # noqa: TC001 - needed at runtime
from nornweave.models.event import EventType  # noqa: TC001 - needed at runtime for Pydantic
from nornweave.verdandi.events import broker
from nornweave.yggdrasil.dependencies import get_storage

if TYPE_CHECKING:
    from nornweave.models.event import Event

router = APIRouter()

HEARTBEAT_SECONDS = 15


class EventResponse(BaseModel):
    """A single event."""

    id: str
    type: EventType
    created_at: datetime
    inbox_id: str | None
    thread_id: str | None
    message_id: str | None
    payload: dict[str, Any]


class EventListResponse(BaseModel):
    """Response model for the event list."""

    items: list[EventResponse]
    count: int


def _to_response(event: Event) -> EventResponse:
    return EventResponse(
        id=event.id,
        type=event.type,
        created_at=event.created_at,
        inbox_id=event.inbox_id,
        thread_id=event.thread_id,
        message_id=event.message_id,
        payload=event.payload,
    )


@router.get("/events", response_model=EventListResponse)
async def list_events(
    event_type: EventType | None = None,
    inbox_id: str | None = None,
    thread_id: str | None = None,
    limit: int = 50,
    offset: int = 0,
    storage: StorageInterface = Depends(get_storage),
) -> EventListResponse:
    """List events, newest first, optionally filtered."""
    events = await storage.list_events(
        event_type=event_type,
        inbox_id=inbox_id,
        thread_id=thread_id,
        limit=limit,
        offset=offset,
    )
    items = [_to_response(event) for event in events]
    return EventListResponse(items=items, count=len(items))


@router.get("/events/stream")
async def stream_events(
    request: Request,
    inbox_id: str | None = None,
    thread_id: str | None = None,
) -> StreamingResponse:
    """Server-sent event stream of live events.

    Emits one SSE message per event as it happens, with heartbeat comments
    while idle. Filters apply per subscriber; an agent can watch one inbox
    or one thread and wake on `message.received`.
    """
    queue = broker.subscribe()

    async def generate() -> Any:
        try:
            yield ": connected\n\n"
            while True:
                if await request.is_disconnected():
                    break
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=HEARTBEAT_SECONDS)
                except TimeoutError:
                    yield ": heartbeat\n\n"
                    continue
                if inbox_id is not None and event.inbox_id != inbox_id:
                    continue
                if thread_id is not None and event.thread_id != thread_id:
                    continue
                data = _to_response(event).model_dump_json()
                yield f"event: {event.type.value}\ndata: {data}\n\n"
        finally:
            broker.unsubscribe(queue)

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
