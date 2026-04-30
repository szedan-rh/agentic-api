"""Pydantic models for the Files API."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class FileObject(BaseModel):
    id: str
    object: Literal["file"] = "file"
    filename: str
    purpose: Literal["assistants", "batch"]
    bytes: int
    status: Literal["uploaded", "processed", "error"] = "uploaded"
    created_at: int
    expires_at: int | None = None


class FileListResponse(BaseModel):
    object: Literal["list"] = "list"
    data: list[FileObject]
    has_more: bool = False


class FileDeleteResponse(BaseModel):
    id: str
    object: Literal["file"] = "file"
    deleted: bool = True
