from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, ForeignKey, String, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from agentic_api.database import Base
from agentic_api.database.session import (
    session_add_one,
    session_delete,
    session_get_all,
    session_get_one,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Response(Base):
    """A single Responses API response, used as a continuation checkpoint.

    `history_item_ids` is an ordered JSON array of Item IDs representing the full
    history at the point this response was completed. It is always populated, regardless
    of whether `conversation_id` is set. It is the single authoritative checkpoint used
    for rehydration on all paths (conversation and standalone).

    `previous_response_id` is a self-referencing FK for lineage inspection; it is not
    walked at rehydration time (the history_item_ids checkpoint is used instead).
    """

    __tablename__ = "responses"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    conversation_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("conversations.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        default=None,
    )
    previous_response_id: Mapped[str | None] = mapped_column(
        String,
        ForeignKey("responses.id", ondelete="SET NULL"),
        nullable=True,
        index=True,
        default=None,
    )
    # Ordered Item ID checkpoint. Always populated on both conversation and standalone paths.
    history_item_ids: Mapped[list[str] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
        default=None,
    )
    # Effective tool config and other metadata needed to rehydrate the next turn.
    # Stores: model, effective_tools, effective_tool_choice, effective_instructions.
    response_metadata: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )

    # Self-referencing relationship for lineage traversal (not used in hot rehydration path).
    previous: Mapped[Response | None] = relationship(
        "Response",
        foreign_keys=[previous_response_id],
        remote_side="Response.id",
        lazy="raise",
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@session_add_one
async def create_response(
    *,
    id: str,
    conversation_id: str | None = None,
    previous_response_id: str | None = None,
    history_item_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Response:
    """Insert a new Response row. Raises IntegrityError if the ID already exists."""
    now = _utcnow()
    return Response(
        id=id,
        conversation_id=conversation_id,
        previous_response_id=previous_response_id,
        history_item_ids=history_item_ids,
        response_metadata=metadata,
        created_at=now,
        updated_at=now,
    )


@session_get_one
async def get_response(*, id: str):
    """Fetch a single Response by primary key. Returns None if not found."""
    return select(Response).where(Response.id == id)


@session_get_all
async def get_responses_by_conversation(*, conversation_id: str):
    """Fetch all Response rows belonging to a Conversation, ordered by creation time."""
    return (
        select(Response)
        .where(Response.conversation_id == conversation_id)
        .order_by(Response.created_at)
    )


@session_delete
async def delete_response(*, id: str):
    """Delete a Response by ID. Silently does nothing if not found."""
    return select(Response).where(Response.id == id)
