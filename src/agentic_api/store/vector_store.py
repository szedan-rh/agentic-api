"""VectorStoreManager — coordinates metadata, chunking, embedding, and search."""

from __future__ import annotations

import logging
from typing import Any

from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_api.database import vector_search
from agentic_api.database.vector_search import ChunkRow
from agentic_api.database.vector_store import (
    create_vector_store,
    delete_vector_store,
    get_vector_store,
    list_vector_stores,
    update_vector_store,
    VectorStoreRow,
)
from agentic_api.database.vector_store_file import (
    create_vector_store_file,
    delete_vector_store_file,
    get_vector_store_file,
    list_vector_store_files,
    update_vector_store_file_status,
    VectorStoreFileRow,
)
from agentic_api.store.chunker import chunk_text
from agentic_api.store.embedding_client import EmbeddingClient
from agentic_api.store.file_store import FileStore
from agentic_api.store.ranker import (
    ScoredChunk,
    reciprocal_rank_fusion,
    weighted_rerank,
)
from agentic_api.types.vector_stores import (
    ChunkingStrategy,
    FileCounts,
    RankingOptions,
    SearchResult,
    VectorStore,
    VectorStoreFile,
    VectorStoreSearchResponse,
)
from agentic_api.utils.common import uuid7_str

logger = logging.getLogger(__name__)


def _row_to_vector_store(row: VectorStoreRow) -> VectorStore:
    fc = row.file_counts or {}
    return VectorStore(
        id=row.id,
        name=row.name,
        status=row.status,
        file_counts=FileCounts(**fc),
        metadata=row.metadata_,
        embedding_model=row.embedding_model,
        embedding_dimension=row.embedding_dimension,
        created_at=int(row.created_at.timestamp()),
        updated_at=int(row.updated_at.timestamp()),
    )


def _row_to_vector_store_file(row: VectorStoreFileRow) -> VectorStoreFile:
    cs = ChunkingStrategy(**row.chunking_strategy) if row.chunking_strategy else None
    return VectorStoreFile(
        id=row.id,
        vector_store_id=row.vector_store_id,
        status=row.status,
        chunking_strategy=cs,
        chunk_count=row.chunk_count,
        created_at=int(row.created_at.timestamp()),
    )


class VectorStoreManager:
    def __init__(
        self,
        *,
        engine: AsyncEngine,
        file_store: FileStore,
        embedding_client: EmbeddingClient,
        db_path: str,
    ) -> None:
        self._engine = engine
        self._file_store = file_store
        self._embedding_client = embedding_client
        self._db_path = db_path

    # ------------------------------------------------------------------
    # Store CRUD
    # ------------------------------------------------------------------

    async def create_store(
        self,
        *,
        name: str,
        embedding_model: str,
        embedding_dimension: int | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> VectorStore:
        dimension = embedding_dimension or 0
        store_id = uuid7_str(prefix="vs_")
        row = await create_vector_store(
            id=store_id,
            name=name,
            embedding_model=embedding_model,
            embedding_dimension=dimension,
            metadata=metadata,
        )
        if dimension > 0:
            await vector_search.create_store_tables(self._db_path, store_id, dimension)
        return _row_to_vector_store(row)

    async def get_store(self, store_id: str) -> VectorStore | None:
        row = await get_vector_store(id=store_id)
        return _row_to_vector_store(row) if row else None

    async def list_stores(self) -> list[VectorStore]:
        rows = await list_vector_stores()
        return [_row_to_vector_store(r) for r in rows]

    async def delete_store(self, store_id: str) -> bool:
        row = await get_vector_store(id=store_id)
        if row is None:
            return False
        await vector_search.drop_store_tables(self._db_path, store_id)
        await delete_vector_store(id=store_id)
        return True

    # ------------------------------------------------------------------
    # File attachment
    # ------------------------------------------------------------------

    async def attach_file(
        self,
        *,
        store_id: str,
        file_id: str,
        chunking_strategy: ChunkingStrategy | None = None,
    ) -> VectorStoreFile:
        store_row = await get_vector_store(id=store_id)
        if store_row is None:
            raise ValueError(f"Vector store '{store_id}' not found.")

        content_bytes = await self._file_store.get_content(file_id=file_id)
        if content_bytes is None:
            raise ValueError(f"File '{file_id}' not found.")

        file_obj = await self._file_store.get(file_id=file_id)
        filename = file_obj.filename if file_obj else file_id

        cs = chunking_strategy or ChunkingStrategy()
        vsf_id = uuid7_str(prefix="vsf_")

        vsf_row = await create_vector_store_file(
            id=vsf_id,
            vector_store_id=store_id,
            filename=filename,
            chunking_strategy=cs.model_dump(),
        )

        try:
            text_content = content_bytes.decode("utf-8", errors="replace")
            chunks = chunk_text(
                text_content,
                max_chunk_size_tokens=cs.max_chunk_size_tokens,
                chunk_overlap_tokens=cs.chunk_overlap_tokens,
            )

            if not chunks:
                await update_vector_store_file_status(
                    id=vsf_id,
                    vector_store_id=store_id,
                    status="completed",
                    chunk_count=0,
                )
                return _row_to_vector_store_file(vsf_row)

            chunk_texts = [c.text for c in chunks]
            embeddings = await self._embedding_client.embed(chunk_texts)

            dimension = len(embeddings[0])
            if store_row.embedding_dimension == 0:
                await update_vector_store(id=store_id, embedding_dimension=dimension)
                await vector_search.create_store_tables(
                    self._db_path, store_id, dimension
                )

            chunk_rows = [
                ChunkRow(
                    chunk_id=uuid7_str(prefix="chk_"),
                    file_id=file_id,
                    chunk_index=c.chunk_index,
                    content=c.text,
                    embedding=emb,
                )
                for c, emb in zip(chunks, embeddings)
            ]
            await vector_search.insert_chunks(self._db_path, store_id, chunk_rows)

            await update_vector_store_file_status(
                id=vsf_id,
                vector_store_id=store_id,
                status="completed",
                chunk_count=len(chunk_rows),
            )

            await self._update_file_counts(store_id)

        except Exception:
            logger.exception("Failed to index file %s into store %s", file_id, store_id)
            await update_vector_store_file_status(
                id=vsf_id, vector_store_id=store_id, status="failed"
            )
            await self._update_file_counts(store_id)
            raise

        updated_row = await get_vector_store_file(id=vsf_id, vector_store_id=store_id)
        return _row_to_vector_store_file(updated_row or vsf_row)

    async def get_file(self, store_id: str, file_id: str) -> VectorStoreFile | None:
        row = await get_vector_store_file(id=file_id, vector_store_id=store_id)
        return _row_to_vector_store_file(row) if row else None

    async def list_files(self, store_id: str) -> list[VectorStoreFile]:
        rows = await list_vector_store_files(vector_store_id=store_id)
        return [_row_to_vector_store_file(r) for r in rows]

    async def delete_file(self, store_id: str, file_id: str) -> bool:
        row = await get_vector_store_file(id=file_id, vector_store_id=store_id)
        if row is None:
            return False
        await vector_search.delete_file_chunks(self._db_path, store_id, file_id)
        await delete_vector_store_file(id=file_id, vector_store_id=store_id)
        await self._update_file_counts(store_id)
        return True

    # ------------------------------------------------------------------
    # Search
    # ------------------------------------------------------------------

    async def search(
        self,
        *,
        store_id: str,
        query: str,
        max_num_results: int = 20,
        search_mode: str = "hybrid",
        ranking_options: RankingOptions | None = None,
        filters: dict[str, Any] | None = None,
    ) -> VectorStoreSearchResponse:
        store_row = await get_vector_store(id=store_id)
        if store_row is None:
            raise ValueError(f"Vector store '{store_id}' not found.")

        opts = ranking_options or RankingOptions()
        file_ids: list[str] | None = None
        if filters and "file_id" in filters:
            val = filters["file_id"]
            file_ids = val if isinstance(val, list) else [val]

        results: list[ScoredChunk] = []

        if search_mode in ("vector", "hybrid"):
            query_embedding = await self._embedding_client.embed_query(query)
            vec_results = await vector_search.vector_search(
                self._db_path,
                store_id,
                query_embedding,
                k=max_num_results,
                file_ids=file_ids,
            )
            if search_mode == "vector":
                results = vec_results

        if search_mode in ("keyword", "hybrid"):
            kw_results = await vector_search.keyword_search(
                self._db_path, store_id, query, k=max_num_results, file_ids=file_ids
            )
            if search_mode == "keyword":
                results = kw_results

        if search_mode == "hybrid":
            if opts.ranker == "rrf":
                results = reciprocal_rank_fusion(
                    [vec_results, kw_results], k=opts.rrf_k, max_results=max_num_results
                )
            else:
                results = weighted_rerank(
                    vec_results,
                    kw_results,
                    vector_weight=opts.vector_weight,
                    keyword_weight=opts.keyword_weight,
                    max_results=max_num_results,
                )

        if opts.score_threshold > 0:
            results = [r for r in results if r.score >= opts.score_threshold]

        vsf_rows = await list_vector_store_files(vector_store_id=store_id)
        file_id_to_filename: dict[str, str] = {}
        for vsf in vsf_rows:
            file_id_to_filename[vsf.id] = vsf.filename

        search_results = [
            SearchResult(
                file_id=r.file_id,
                filename=file_id_to_filename.get(r.file_id, r.file_id),
                score=r.score,
                text=r.text,
                chunk_index=r.chunk_index,
            )
            for r in results
        ]

        return VectorStoreSearchResponse(data=search_results)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _update_file_counts(self, store_id: str) -> None:
        rows = await list_vector_store_files(vector_store_id=store_id)
        counts = {"in_progress": 0, "completed": 0, "failed": 0, "total": len(rows)}
        for r in rows:
            if r.status in counts:
                counts[r.status] += 1
        await update_vector_store(id=store_id, file_counts=counts)

        if counts["in_progress"] == 0 and counts["total"] > 0:
            await update_vector_store(id=store_id, status="completed")
