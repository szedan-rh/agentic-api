from __future__ import annotations

import os
import re

from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_api.database.file import (
    File,
    create_file,
    delete_file,
    get_file,
    list_files,
)
from agentic_api.database.session import configure_session_factory
from agentic_api.types.files import FileDeleteResponse, FileListResponse, FileObject
from agentic_api.utils.common import uuid7_str


def _row_to_file_object(row: File) -> FileObject:
    return FileObject(
        id=row.id,
        filename=row.filename,
        purpose=row.purpose,  # type: ignore[arg-type]
        bytes=row.bytes_,
        status=row.status,  # type: ignore[arg-type]
        created_at=int(row.created_at.timestamp()),
        expires_at=int(row.expires_at.timestamp()) if row.expires_at else None,
    )


class FileStore:
    def __init__(self, *, engine: AsyncEngine) -> None:
        configure_session_factory(engine)

    async def upload(
        self, *, filename: str, content: bytes, purpose: str
    ) -> FileObject:
        safe_name = self.sanitize_filename(filename)
        file_id = uuid7_str("file-")
        row: File = await create_file(
            id=file_id,
            filename=safe_name,
            purpose=purpose,
            bytes_=len(content),
            content=content,
        )
        return _row_to_file_object(row)

    async def get(self, *, file_id: str) -> FileObject | None:
        row: File | None = await get_file(id=file_id)
        if row is None:
            return None
        return _row_to_file_object(row)

    async def list(
        self,
        *,
        limit: int = 20,
        order: str = "desc",
        after: str | None = None,
        purpose: str | None = None,
    ) -> FileListResponse:
        rows: list[File] = await list_files(limit=limit)

        if purpose is not None:
            rows = [r for r in rows if r.purpose == purpose]

        if after is not None:
            found = False
            filtered: list[File] = []
            for r in rows:
                if found:
                    filtered.append(r)
                if r.id == after:
                    found = True
            rows = filtered

        if order == "asc":
            rows = list(reversed(rows))

        return FileListResponse(
            data=[_row_to_file_object(r) for r in rows],
            has_more=False,
        )

    async def delete(self, *, file_id: str) -> FileDeleteResponse | None:
        row: File | None = await get_file(id=file_id)
        if row is None:
            return None
        await delete_file(id=file_id)
        return FileDeleteResponse(id=file_id)

    async def get_content(self, *, file_id: str) -> bytes | None:
        row: File | None = await get_file(id=file_id)
        if row is None:
            return None
        return row.content

    @staticmethod
    def sanitize_filename(filename: str) -> str:
        name = os.path.basename(filename)
        name = name.replace("\x00", "")
        name = re.sub(r"\.{2,}", ".", name)
        name = name.lstrip(".")
        name = name.strip()
        if not name:
            return "download"
        return name
