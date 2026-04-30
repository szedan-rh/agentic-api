"""SQLAlchemy ORM model and CRUD for the ``vector_stores`` metadata table."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import DateTime, String, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.types import JSON

from agentic_api.database import Base
from agentic_api.database.session import (
    run_in_session,
    session_add_one,
    session_delete,
    session_get_all,
    session_get_one,
)
from agentic_api.utils.common import utcnow


class VectorStoreRow(Base):
    __tablename__ = "vector_stores"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    name: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="in_progress")
    file_counts: Mapped[dict[str, Any]] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=False,
    )
    metadata_: Mapped[dict[str, Any] | None] = mapped_column(
        "metadata",
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
        default=None,
    )
    embedding_model: Mapped[str] = mapped_column(String, nullable=False)
    embedding_dimension: Mapped[int] = mapped_column(nullable=False)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@session_add_one
async def create_vector_store(
    *,
    id: str,
    name: str,
    embedding_model: str,
    embedding_dimension: int,
    metadata: dict[str, Any] | None = None,
) -> VectorStoreRow:
    now = utcnow()
    return VectorStoreRow(
        id=id,
        name=name,
        status="in_progress",
        file_counts={"in_progress": 0, "completed": 0, "failed": 0, "total": 0},
        metadata_=metadata,
        embedding_model=embedding_model,
        embedding_dimension=embedding_dimension,
        created_at=now,
        updated_at=now,
    )


@session_get_one
async def get_vector_store(*, id: str):
    return select(VectorStoreRow).where(VectorStoreRow.id == id)


@session_get_all
async def list_vector_stores():
    return select(VectorStoreRow).order_by(VectorStoreRow.created_at.desc())


@session_delete
async def delete_vector_store(*, id: str):
    return select(VectorStoreRow).where(VectorStoreRow.id == id)


@run_in_session
async def update_vector_store(
    session,
    *,
    id: str,
    name: str | None = None,
    status: str | None = None,
    file_counts: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = ...,  # type: ignore[assignment]
    embedding_dimension: int | None = None,
) -> VectorStoreRow | None:
    result = await session.execute(
        select(VectorStoreRow).where(VectorStoreRow.id == id)
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None

    if name is not None:
        row.name = name
    if status is not None:
        row.status = status
    if file_counts is not None:
        row.file_counts = file_counts
    if metadata is not ...:
        row.metadata_ = metadata
    if embedding_dimension is not None:
        row.embedding_dimension = embedding_dimension
    row.updated_at = utcnow()

    await session.flush()
    return row
