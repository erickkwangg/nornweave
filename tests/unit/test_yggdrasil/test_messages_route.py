"""Unit tests for messages route."""

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from nornweave.models.message import Message, MessageDirection
from nornweave.yggdrasil.routes.v1.messages import (
    MessageListResponse,
    MessageResponse,
    _build_reply_headers,
    _message_to_response,
    _rfc_msgid,
)


def _make_message(
    message_id: str,
    provider_message_id: str | None,
    references: list[str] | None = None,
) -> Message:
    return Message(
        message_id=message_id,
        thread_id="th_1",
        inbox_id="ibx_1",
        direction=MessageDirection.INBOUND,
        provider_message_id=provider_message_id,
        references=references,
    )


class TestRfcMsgid:
    """Tests for _rfc_msgid normalization."""

    def test_bracketed_id_passes_through(self) -> None:
        assert _rfc_msgid("<abc@mail.example.com>") == "<abc@mail.example.com>"

    def test_bare_id_with_at_gets_brackets(self) -> None:
        assert _rfc_msgid("abc@mail.example.com") == "<abc@mail.example.com>"

    def test_provider_api_id_without_at_is_dropped(self) -> None:
        # e.g. Resend returns a UUID that is not a wire Message-ID
        assert _rfc_msgid("49a3999c-0ce1-4ea6-ab68-afcd6dc2e794") is None

    def test_none_and_empty(self) -> None:
        assert _rfc_msgid(None) is None
        assert _rfc_msgid("") is None


class TestBuildReplyHeaders:
    """Tests for _build_reply_headers derivation from thread history."""

    def test_empty_thread_returns_none(self) -> None:
        assert _build_reply_headers([]) == (None, None)

    def test_single_parent_message(self) -> None:
        parent = _make_message("m1", "<orig@example.com>")
        in_reply_to, references = _build_reply_headers([parent])
        assert in_reply_to == "<orig@example.com>"
        assert references == ["<orig@example.com>"]

    def test_uses_latest_message_and_extends_chain(self) -> None:
        first = _make_message("m1", "<orig@example.com>")
        reply = _make_message("m2", "<reply@example.com>", references=["<orig@example.com>"])
        in_reply_to, references = _build_reply_headers([first, reply])
        assert in_reply_to == "<reply@example.com>"
        assert references == ["<orig@example.com>", "<reply@example.com>"]

    def test_skips_messages_without_usable_id(self) -> None:
        first = _make_message("m1", "<orig@example.com>")
        failed = _make_message("m2", None)
        api_id_only = _make_message("m3", "49a3999c-0ce1-4ea6-ab68-afcd6dc2e794")
        in_reply_to, references = _build_reply_headers([first, failed, api_id_only])
        assert in_reply_to == "<orig@example.com>"
        assert references == ["<orig@example.com>"]

    def test_deduplicates_references(self) -> None:
        msg = _make_message(
            "m1", "<a@example.com>", references=["<a@example.com>", "<b@example.com>"]
        )
        in_reply_to, references = _build_reply_headers([msg])
        assert in_reply_to == "<a@example.com>"
        assert references == ["<a@example.com>", "<b@example.com>"]


class TestMessageResponse:
    """Tests for MessageResponse model."""

    def test_message_response_minimal_fields(self) -> None:
        """Test MessageResponse with minimal required fields."""
        response = MessageResponse(
            id="msg_123",
            thread_id="th_456",
            inbox_id="ibx_789",
            direction="inbound",
            provider_message_id=None,
        )

        assert response.id == "msg_123"
        assert response.thread_id == "th_456"
        assert response.inbox_id == "ibx_789"
        assert response.direction == "inbound"
        assert response.provider_message_id is None
        # Default values
        assert response.to_addresses == []
        assert response.labels == []
        assert response.size == 0
        assert response.content_clean == ""
        assert response.metadata == {}

    def test_message_response_all_fields(self) -> None:
        """Test MessageResponse with all fields populated."""
        now = datetime.now(UTC)
        response = MessageResponse(
            id="msg_123",
            thread_id="th_456",
            inbox_id="ibx_789",
            direction="inbound",
            provider_message_id="<abc@mail.example.com>",
            subject="Test Subject",
            from_address="sender@example.com",
            to_addresses=["recipient@example.com"],
            cc_addresses=["cc@example.com"],
            bcc_addresses=["bcc@example.com"],
            reply_to_addresses=["reply@example.com"],
            text="Plain text body",
            html="<p>HTML body</p>",
            content_clean="Plain text body",
            timestamp=now,
            labels=["important", "inbox"],
            preview="Plain text...",
            size=1234,
            in_reply_to="<parent@mail.example.com>",
            references=["<ref1@mail.example.com>", "<ref2@mail.example.com>"],
            metadata={"custom": "header"},
            created_at=now,
        )

        assert response.subject == "Test Subject"
        assert response.from_address == "sender@example.com"
        assert response.to_addresses == ["recipient@example.com"]
        assert response.cc_addresses == ["cc@example.com"]
        assert response.bcc_addresses == ["bcc@example.com"]
        assert response.reply_to_addresses == ["reply@example.com"]
        assert response.text == "Plain text body"
        assert response.html == "<p>HTML body</p>"
        assert response.content_clean == "Plain text body"
        assert response.timestamp == now
        assert response.labels == ["important", "inbox"]
        assert response.preview == "Plain text..."
        assert response.size == 1234
        assert response.in_reply_to == "<parent@mail.example.com>"
        assert response.references == ["<ref1@mail.example.com>", "<ref2@mail.example.com>"]
        assert response.metadata == {"custom": "header"}
        assert response.created_at == now

    def test_message_response_null_optional_fields(self) -> None:
        """Test MessageResponse with null optional fields."""
        response = MessageResponse(
            id="msg_123",
            thread_id="th_456",
            inbox_id="ibx_789",
            direction="outbound",
            provider_message_id=None,
            subject=None,
            from_address=None,
            cc_addresses=None,
            bcc_addresses=None,
            reply_to_addresses=None,
            text=None,
            html=None,
            timestamp=None,
            preview=None,
            in_reply_to=None,
            references=None,
            created_at=None,
        )

        assert response.subject is None
        assert response.from_address is None
        assert response.cc_addresses is None
        assert response.text is None
        assert response.html is None


class TestMessageToResponse:
    """Tests for _message_to_response conversion function."""

    def test_converts_all_fields(self) -> None:
        """Test that _message_to_response maps all fields correctly."""
        now = datetime.now(UTC)
        message = Message(
            message_id="msg_123",
            thread_id="th_456",
            inbox_id="ibx_789",
            direction=MessageDirection.INBOUND,
            provider_message_id="<abc@mail.example.com>",
            subject="Test Subject",
            from_address="sender@example.com",
            to=["recipient@example.com"],
            cc=["cc@example.com"],
            bcc=["bcc@example.com"],
            reply_to=["reply@example.com"],
            text="Plain text body",
            html="<p>HTML body</p>",
            extracted_text="Plain text body",
            timestamp=now,
            labels=["important"],
            preview="Plain text...",
            size=1234,
            in_reply_to="<parent@mail.example.com>",
            references=["<ref1@mail.example.com>"],
            headers={"X-Custom": "value"},
            created_at=now,
        )

        response = _message_to_response(message)

        assert response.id == "msg_123"
        assert response.thread_id == "th_456"
        assert response.inbox_id == "ibx_789"
        assert response.direction == "inbound"
        assert response.provider_message_id == "<abc@mail.example.com>"
        assert response.subject == "Test Subject"
        assert response.from_address == "sender@example.com"
        assert response.to_addresses == ["recipient@example.com"]
        assert response.cc_addresses == ["cc@example.com"]
        assert response.bcc_addresses == ["bcc@example.com"]
        assert response.reply_to_addresses == ["reply@example.com"]
        assert response.text == "Plain text body"
        assert response.html == "<p>HTML body</p>"
        assert response.content_clean == "Plain text body"
        assert response.timestamp == now
        assert response.labels == ["important"]
        assert response.preview == "Plain text..."
        assert response.size == 1234
        assert response.in_reply_to == "<parent@mail.example.com>"
        assert response.references == ["<ref1@mail.example.com>"]
        assert response.metadata == {"X-Custom": "value"}
        assert response.created_at == now

    def test_handles_empty_lists(self) -> None:
        """Test that empty lists are handled correctly."""
        message = Message(
            message_id="msg_123",
            thread_id="th_456",
            inbox_id="ibx_789",
            direction=MessageDirection.OUTBOUND,
            to=[],
            labels=[],
        )

        response = _message_to_response(message)

        assert response.to_addresses == []
        assert response.labels == []

    def test_handles_none_values(self) -> None:
        """Test that None values are handled correctly."""
        message = Message(
            message_id="msg_123",
            thread_id="th_456",
            inbox_id="ibx_789",
            direction=MessageDirection.INBOUND,
            subject=None,
            from_address=None,
            cc=None,
            bcc=None,
            reply_to=None,
            text=None,
            html=None,
            extracted_text=None,
            timestamp=None,
            headers=None,
            created_at=None,
        )

        response = _message_to_response(message)

        assert response.subject is None
        assert response.from_address is None
        assert response.cc_addresses is None
        assert response.bcc_addresses is None
        assert response.reply_to_addresses is None
        assert response.text is None
        assert response.html is None
        assert response.content_clean == ""
        assert response.timestamp is None
        assert response.metadata == {}
        assert response.created_at is None


class TestMessageListResponse:
    """Tests for MessageListResponse model."""

    def test_list_response_with_items(self) -> None:
        """Test MessageListResponse with items."""
        response = MessageListResponse(
            items=[
                MessageResponse(
                    id="msg_1",
                    thread_id="th_1",
                    inbox_id="ibx_1",
                    direction="inbound",
                    provider_message_id=None,
                ),
                MessageResponse(
                    id="msg_2",
                    thread_id="th_1",
                    inbox_id="ibx_1",
                    direction="outbound",
                    provider_message_id=None,
                ),
            ],
            count=2,
            total=10,
        )

        assert len(response.items) == 2
        assert response.count == 2
        assert response.total == 10

    def test_list_response_empty(self) -> None:
        """Test MessageListResponse with no items."""
        response = MessageListResponse(
            items=[],
            count=0,
            total=0,
        )

        assert response.items == []
        assert response.count == 0
        assert response.total == 0

    def test_list_response_requires_total(self) -> None:
        """Test that MessageListResponse requires total field."""
        with pytest.raises(ValidationError):
            MessageListResponse(
                items=[],
                count=0,
                # missing total
            )
