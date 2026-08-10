"""Unit tests for the wait_for_reply MCP tool."""

from __future__ import annotations

import types
from typing import TYPE_CHECKING, Any, cast

import httpx
import pytest

from nornweave.muninn.tools import wait_for_reply

if TYPE_CHECKING:
    from nornweave.huginn.client import NornWeaveClient


def _msg(role: str, content: str, author: str = "human@example.com") -> dict[str, Any]:
    return {
        "role": role,
        "author": author,
        "content": content,
        "timestamp": "2026-08-10T12:00:00Z",
    }


def _thread(*messages: dict[str, Any]) -> dict[str, Any]:
    return {"id": "t1", "subject": "Hello", "summary": None, "messages": list(messages)}


def _http_error(status_code: int) -> httpx.HTTPStatusError:
    request = httpx.Request("GET", "http://test/v1/threads/t1")
    response = httpx.Response(status_code, request=request)
    return httpx.HTTPStatusError(f"HTTP {status_code}", request=request, response=response)


class StubClient:
    """Scripted stand-in for NornWeaveClient.

    get_thread returns the scripted responses in order, repeating the last one
    once the script is exhausted. Exceptions in the script are raised instead.
    """

    def __init__(self, *responses: dict[str, Any] | Exception) -> None:
        self._responses = list(responses)
        self.calls = 0

    async def get_thread(self, thread_id: str) -> dict[str, Any]:
        assert thread_id == "t1"
        self.calls += 1
        response = self._responses.pop(0) if len(self._responses) > 1 else self._responses[0]
        if isinstance(response, Exception):
            raise response
        return response


def _client(stub: StubClient) -> NornWeaveClient:
    return cast("NornWeaveClient", stub)


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    """Make polling sleeps instant and record their durations."""
    recorded: list[float] = []

    async def _sleep(seconds: float) -> None:
        recorded.append(seconds)

    monkeypatch.setattr(
        "nornweave.muninn.tools.asyncio",
        types.SimpleNamespace(sleep=_sleep),
    )
    return recorded


class TestWaitForReply:
    """Tests for wait_for_reply polling behavior."""

    async def test_returns_reply_when_inbound_arrives(self, sleeps: list[float]) -> None:
        """A new inbound (role=user) message is returned as the reply."""
        initial = _thread(_msg("assistant", "Ping from agent", author="agent@example.com"))
        stub = StubClient(initial, _thread(*initial["messages"], _msg("user", "Pong from human")))

        result = await wait_for_reply(_client(stub), "t1", timeout_seconds=30, poll_interval=5)

        assert result["received"] is True
        assert result["message"]["author"] == "human@example.com"
        assert result["message"]["content"] == "Pong from human"
        assert result["message"]["timestamp"] == "2026-08-10T12:00:00Z"
        assert sleeps == [5]

    async def test_ignores_own_outbound_messages(self, sleeps: list[float]) -> None:
        """Outbound (role=assistant) messages sent while waiting are not replies."""
        first = _msg("user", "Original question")
        outbound = _msg("assistant", "Agent follow-up", author="agent@example.com")
        reply = _msg("user", "Actual reply")
        stub = StubClient(
            _thread(first),
            _thread(first, outbound),
            _thread(first, outbound, reply),
        )

        result = await wait_for_reply(_client(stub), "t1", timeout_seconds=30, poll_interval=5)

        assert result["received"] is True
        assert result["message"]["content"] == "Actual reply"
        assert stub.calls == 3
        assert sleeps == [5, 5]

    async def test_returns_reply_even_when_followed_by_outbound(self, sleeps: list[float]) -> None:
        """The inbound reply wins even if an outbound message landed after it."""
        first = _msg("user", "Original question")
        reply = _msg("user", "Actual reply")
        outbound = _msg("assistant", "Auto-ack", author="agent@example.com")
        stub = StubClient(_thread(first), _thread(first, reply, outbound))

        result = await wait_for_reply(_client(stub), "t1", timeout_seconds=30, poll_interval=5)

        assert result["received"] is True
        assert result["message"]["content"] == "Actual reply"
        assert sleeps == [5]

    async def test_times_out_when_only_outbound_activity(self, sleeps: list[float]) -> None:
        """Outbound-only activity runs the full timeout instead of returning."""
        first = _msg("user", "Original question")
        outbound = _msg("assistant", "Agent follow-up", author="agent@example.com")
        stub = StubClient(_thread(first), _thread(first, outbound))

        result = await wait_for_reply(_client(stub), "t1", timeout_seconds=10, poll_interval=5)

        assert result == {"received": False, "timeout": True, "waited_seconds": 10}
        assert sleeps == [5, 5]

    async def test_times_out_when_no_new_messages(self, sleeps: list[float]) -> None:
        """No activity at all returns the timeout indicator."""
        stub = StubClient(_thread(_msg("user", "Original question")))

        result = await wait_for_reply(_client(stub), "t1", timeout_seconds=10, poll_interval=5)

        assert result == {"received": False, "timeout": True, "waited_seconds": 10}
        assert stub.calls == 3
        assert sleeps == [5, 5]

    async def test_thread_not_found_raises(self, sleeps: list[float]) -> None:
        """A 404 on the initial fetch raises instead of polling."""
        stub = StubClient(_http_error(404))

        with pytest.raises(Exception, match="Thread 't1' not found"):
            await wait_for_reply(_client(stub), "t1", timeout_seconds=10, poll_interval=5)

        assert stub.calls == 1
        assert sleeps == []

    async def test_initial_server_error_raises(self, sleeps: list[float]) -> None:
        """A non-404 error on the initial fetch raises with the status code."""
        stub = StubClient(_http_error(500))

        with pytest.raises(Exception, match="Failed to access thread: 500"):
            await wait_for_reply(_client(stub), "t1", timeout_seconds=10, poll_interval=5)

        assert sleeps == []

    async def test_transient_poll_errors_are_ignored(self, sleeps: list[float]) -> None:
        """A failing poll is skipped and polling continues to the reply."""
        initial = _thread(_msg("user", "Original question"))
        stub = StubClient(
            initial,
            _http_error(500),
            _thread(*initial["messages"], _msg("user", "Actual reply")),
        )

        result = await wait_for_reply(_client(stub), "t1", timeout_seconds=30, poll_interval=5)

        assert result["received"] is True
        assert result["message"]["content"] == "Actual reply"
        assert stub.calls == 3
        assert sleeps == [5, 5]

    async def test_zero_timeout_returns_immediately(self, sleeps: list[float]) -> None:
        """timeout_seconds=0 checks the thread once and never polls."""
        stub = StubClient(_thread(_msg("user", "Original question")))

        result = await wait_for_reply(_client(stub), "t1", timeout_seconds=0, poll_interval=5)

        assert result == {"received": False, "timeout": True, "waited_seconds": 0}
        assert stub.calls == 1
        assert sleeps == []

    async def test_poll_interval_clamped_to_one(self, sleeps: list[float]) -> None:
        """poll_interval=0 cannot spin forever; it is clamped to 1 second."""
        stub = StubClient(_thread(_msg("user", "Original question")))

        result = await wait_for_reply(_client(stub), "t1", timeout_seconds=2, poll_interval=0)

        assert result == {"received": False, "timeout": True, "waited_seconds": 2}
        assert sleeps == [1, 1]
