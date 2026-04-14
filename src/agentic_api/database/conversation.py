from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from sqlalchemy import DateTime, String, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.types import JSON

from agentic_api.database import Base
from agentic_api.database.session import (
    run_in_session,
    session_add_one,
    session_delete,
    session_get_all,
    session_get_one,
)


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Conversation(Base):
    """An ordered collection of Items representing a full conversation thread.

    Used by the Conversation API path. When a Response belongs to a Conversation,
    `Response.history_item_ids` is null — the Conversation's `item_ids` is the
    authoritative ordered history source.

    `item_ids` grows as turns are appended; it is the ordered list of all Item IDs
    in the conversation so far.

    `metadata_` is an open JSON object for caller-supplied context (e.g. title,
    external IDs, tags). Not interpreted by the store.
    """

    __tablename__ = "conversations"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    item_ids: Mapped[list[str]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
        default=list,
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
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

    # Responses that belong to this conversation.
    responses: Mapped[list[Any]] = relationship(
        "Response",
        foreign_keys="Response.conversation_id",
        back_populates=None,
        lazy="raise",
        order_by="Response.created_at",
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@session_add_one
async def create_conversation(
    *,
    id: str,
    item_ids: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> Conversation:
    """Insert a new Conversation row. Raises IntegrityError if the ID already exists."""
    now = _utcnow()
    return Conversation(
        id=id,
        item_ids=item_ids or [],
        metadata_=metadata,
        created_at=now,
        updated_at=now,
    )


@session_get_one
async def get_conversation(*, id: str):
    """Fetch a single Conversation by primary key. Returns None if not found."""
    return select(Conversation).where(Conversation.id == id)


@session_get_all
async def get_conversations(*, ids: list[str]):
    """Bulk-fetch Conversations by ID. Returns a list in unspecified order."""
    return select(Conversation).where(Conversation.id.in_(ids))


@run_in_session
async def update_conversation_item_ids(
    session: AsyncSession,
    *,
    id: str,
    item_ids: list[str],
) -> Conversation | None:
    """Append item_ids to a Conversation row and update updated_at.

    Returns the updated Conversation, or None if not found.
    """
    result = await session.execute(select(Conversation).where(Conversation.id == id))
    conversation = result.scalar_one_or_none()
    if conversation is None:
        return None
    conversation.item_ids = item_ids
    conversation.updated_at = _utcnow()
    await session.flush()
    return conversation


@session_delete
async def delete_conversation(*, id: str):
    """Delete a Conversation by ID. Silently does nothing if not found.

    Note: Response rows with this conversation_id will have their conversation_id
    set to NULL (ondelete=SET NULL on the FK).
    """
    return select(Conversation).where(Conversation.id == id)
