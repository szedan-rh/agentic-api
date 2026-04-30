"""SQLAlchemy ORM model and CRUD for the ``vector_store_files`` table."""

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


class VectorStoreFileRow(Base):
    __tablename__ = "vector_store_files"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    vector_store_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="in_progress")
    chunking_strategy: Mapped[dict[str, Any] | None] = mapped_column(
        JSON().with_variant(JSONB, "postgresql"),
        nullable=True,
        default=None,
    )
    chunk_count: Mapped[int] = mapped_column(nullable=False, default=0)
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
async def create_vector_store_file(
    *,
    id: str,
    vector_store_id: str,
    filename: str,
    chunking_strategy: dict[str, Any] | None = None,
) -> VectorStoreFileRow:
    now = utcnow()
    return VectorStoreFileRow(
        id=id,
        vector_store_id=vector_store_id,
        filename=filename,
        status="in_progress",
        chunking_strategy=chunking_strategy,
        chunk_count=0,
        created_at=now,
        updated_at=now,
    )


@session_get_one
async def get_vector_store_file(*, id: str, vector_store_id: str):
    return select(VectorStoreFileRow).where(
        VectorStoreFileRow.id == id,
        VectorStoreFileRow.vector_store_id == vector_store_id,
    )


@session_get_all
async def list_vector_store_files(*, vector_store_id: str):
    return (
        select(VectorStoreFileRow)
        .where(VectorStoreFileRow.vector_store_id == vector_store_id)
        .order_by(VectorStoreFileRow.created_at.desc())
    )


@session_delete
async def delete_vector_store_file(*, id: str, vector_store_id: str):
    return select(VectorStoreFileRow).where(
        VectorStoreFileRow.id == id,
        VectorStoreFileRow.vector_store_id == vector_store_id,
    )


@run_in_session
async def update_vector_store_file_status(
    session,
    *,
    id: str,
    vector_store_id: str,
    status: str,
    chunk_count: int | None = None,
) -> VectorStoreFileRow | None:
    result = await session.execute(
        select(VectorStoreFileRow).where(
            VectorStoreFileRow.id == id,
            VectorStoreFileRow.vector_store_id == vector_store_id,
        )
    )
    row = result.scalar_one_or_none()
    if row is None:
        return None

    row.status = status
    if chunk_count is not None:
        row.chunk_count = chunk_count
    row.updated_at = utcnow()

    await session.flush()
    return row
