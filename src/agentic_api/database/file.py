from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Integer, LargeBinary, String, select
from sqlalchemy.orm import Mapped, mapped_column

from agentic_api.database import Base
from agentic_api.database.session import (
    session_add_one,
    session_delete,
    session_get_all,
    session_get_one,
)
from agentic_api.utils.common import utcnow


class File(Base):
    __tablename__ = "files"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    filename: Mapped[str] = mapped_column(String, nullable=False)
    purpose: Mapped[str] = mapped_column(String, nullable=False)
    bytes_: Mapped[int] = mapped_column("bytes", Integer, nullable=False)
    content: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    status: Mapped[str] = mapped_column(String, nullable=False, default="uploaded")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        nullable=False,
        index=True,
    )
    expires_at: Mapped[datetime | None] = mapped_column(
        DateTime(timezone=True),
        nullable=True,
        default=None,
    )


# ---------------------------------------------------------------------------
# CRUD
# ---------------------------------------------------------------------------


@session_add_one
async def create_file(
    *,
    id: str,
    filename: str,
    purpose: str,
    bytes_: int,
    content: bytes,
    status: str = "uploaded",
) -> File:
    """Insert a new File row. Raises IntegrityError if the ID already exists."""
    return File(
        id=id,
        filename=filename,
        purpose=purpose,
        bytes_=bytes_,
        content=content,
        status=status,
        created_at=utcnow(),
    )


@session_get_one
async def get_file(*, id: str):
    """Fetch a single File by primary key. Returns None if not found."""
    return select(File).where(File.id == id)


@session_get_all
async def list_files(*, limit: int = 20):
    """Fetch files ordered by created_at descending, with a limit."""
    return select(File).order_by(File.created_at.desc()).limit(limit)


@session_delete
async def delete_file(*, id: str):
    """Delete a File by ID. Silently does nothing if not found."""
    return select(File).where(File.id == id)
