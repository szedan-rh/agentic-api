"""Pydantic models for the Vector Stores API.

Covers resource shapes, request/response types, and Responses API integration
types (FileSearchTool, FileSearchToolCall) for future use.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Shared components
# ---------------------------------------------------------------------------


class FileCounts(BaseModel):
    in_progress: int = 0
    completed: int = 0
    failed: int = 0
    total: int = 0


class ChunkingStrategy(BaseModel):
    type: Literal["static"] = "static"
    max_chunk_size_tokens: int = 800
    chunk_overlap_tokens: int = 400


class RankingOptions(BaseModel):
    ranker: Literal["rrf", "weighted"] = "rrf"
    score_threshold: float = 0.0
    rrf_k: int = 60
    vector_weight: float = 0.7
    keyword_weight: float = 0.3


# ---------------------------------------------------------------------------
# Vector Store resource
# ---------------------------------------------------------------------------


class VectorStore(BaseModel):
    id: str
    object: Literal["vector_store"] = "vector_store"
    name: str
    status: Literal["in_progress", "completed", "expired"] = "in_progress"
    file_counts: FileCounts = Field(default_factory=FileCounts)
    metadata_: dict[str, Any] | None = Field(default=None, alias="metadata")
    embedding_model: str
    embedding_dimension: int
    created_at: int
    updated_at: int

    model_config = {"populate_by_name": True}


class VectorStoreFile(BaseModel):
    id: str
    object: Literal["vector_store.file"] = "vector_store.file"
    vector_store_id: str
    status: Literal["in_progress", "completed", "failed"] = "in_progress"
    chunking_strategy: ChunkingStrategy | None = None
    chunk_count: int = 0
    created_at: int


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------


class CreateVectorStoreRequest(BaseModel):
    name: str
    embedding_model: str
    embedding_dimension: int | None = None
    metadata_: dict[str, Any] | None = Field(default=None, alias="metadata")

    model_config = {"populate_by_name": True}


class AttachFileRequest(BaseModel):
    file_id: str
    chunking_strategy: ChunkingStrategy | None = None


class SearchVectorStoreRequest(BaseModel):
    query: str
    max_num_results: int = Field(default=20, ge=1, le=50)
    search_mode: Literal["vector", "keyword", "hybrid"] = "hybrid"
    ranking_options: RankingOptions | None = None
    filters: dict[str, Any] | None = None


# ---------------------------------------------------------------------------
# Search results
# ---------------------------------------------------------------------------


class SearchResult(BaseModel):
    file_id: str
    filename: str
    score: float
    text: str
    chunk_index: int
    attributes: dict[str, Any] | None = None


class VectorStoreSearchResponse(BaseModel):
    object: Literal["vector_store.search_results"] = "vector_store.search_results"
    data: list[SearchResult]


# ---------------------------------------------------------------------------
# Delete responses
# ---------------------------------------------------------------------------


class VectorStoreDeleteResponse(BaseModel):
    id: str
    object: Literal["vector_store"] = "vector_store"
    deleted: bool = True


class VectorStoreFileDeleteResponse(BaseModel):
    id: str
    object: Literal["vector_store.file.deleted"] = "vector_store.file.deleted"
    deleted: bool = True


# ---------------------------------------------------------------------------
# Responses API integration types
# ---------------------------------------------------------------------------


class FileSearchTool(BaseModel):
    type: Literal["file_search"] = "file_search"
    vector_store_ids: list[str]
    max_num_results: int = 20
    ranking_options: RankingOptions | None = None


class FileSearchToolCall(BaseModel):
    type: Literal["file_search_call"] = "file_search_call"
    id: str
    status: Literal["in_progress", "completed"] = "completed"
    queries: list[str]
    results: list[SearchResult] | None = None
