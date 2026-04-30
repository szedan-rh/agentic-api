"""Raw sqlite3 + sqlite-vec + FTS5 access for vector store search.

This module intentionally bypasses SQLAlchemy and uses the raw sqlite3 driver
with ``asyncio.to_thread()`` so that the vec0 virtual-table extension and FTS5
can be loaded and queried directly.
"""

from __future__ import annotations

import asyncio
import json
import sqlite3
import struct
from dataclasses import dataclass, field
from typing import Any

import sqlite_vec

from agentic_api.store.ranker import ScoredChunk


def serialize_vector(vector: list[float]) -> bytes:
    """Pack a float list into a compact binary blob for vec0 storage."""
    return struct.pack(f"{len(vector)}f", *vector)


def _get_connection(db_path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    return conn


# ---------------------------------------------------------------------------
# Table lifecycle
# ---------------------------------------------------------------------------


def _create_store_tables_sync(db_path: str, store_id: str, dimension: int) -> None:
    conn = _get_connection(db_path)
    try:
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vs_{store_id}_vec USING vec0("
            f"  chunk_id TEXT PRIMARY KEY,"
            f"  file_id TEXT,"
            f"  chunk_index INTEGER,"
            f"  embedding float[{dimension}]"
            f");"
        )
        conn.execute(
            f"CREATE VIRTUAL TABLE IF NOT EXISTS vs_{store_id}_fts USING fts5("
            f"  chunk_id, content"
            f");"
        )
        conn.execute(
            f"CREATE TABLE IF NOT EXISTS vs_{store_id}_chunks ("
            f"  chunk_id TEXT PRIMARY KEY,"
            f"  file_id TEXT NOT NULL,"
            f"  chunk_index INTEGER NOT NULL,"
            f"  content TEXT NOT NULL,"
            f"  metadata TEXT"
            f");"
        )
        conn.commit()
    finally:
        conn.close()


async def create_store_tables(db_path: str, store_id: str, dimension: int) -> None:
    await asyncio.to_thread(_create_store_tables_sync, db_path, store_id, dimension)


def _drop_store_tables_sync(db_path: str, store_id: str) -> None:
    conn = _get_connection(db_path)
    try:
        conn.execute(f"DROP TABLE IF EXISTS vs_{store_id}_vec;")
        conn.execute(f"DROP TABLE IF EXISTS vs_{store_id}_fts;")
        conn.execute(f"DROP TABLE IF EXISTS vs_{store_id}_chunks;")
        conn.commit()
    finally:
        conn.close()


async def drop_store_tables(db_path: str, store_id: str) -> None:
    await asyncio.to_thread(_drop_store_tables_sync, db_path, store_id)


# ---------------------------------------------------------------------------
# Chunk data
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ChunkRow:
    chunk_id: str
    file_id: str
    chunk_index: int
    content: str
    embedding: list[float]
    metadata: dict[str, Any] | None = field(default=None)


# ---------------------------------------------------------------------------
# Insert / delete
# ---------------------------------------------------------------------------


def _insert_chunks_sync(db_path: str, store_id: str, chunks: list[ChunkRow]) -> None:
    conn = _get_connection(db_path)
    try:
        for chunk in chunks:
            conn.execute(
                f"INSERT INTO vs_{store_id}_vec (chunk_id, file_id, chunk_index, embedding) VALUES (?, ?, ?, ?);",
                (
                    chunk.chunk_id,
                    chunk.file_id,
                    chunk.chunk_index,
                    serialize_vector(chunk.embedding),
                ),
            )
            conn.execute(
                f"INSERT INTO vs_{store_id}_fts (chunk_id, content) VALUES (?, ?);",
                (chunk.chunk_id, chunk.content),
            )
            meta_json = json.dumps(chunk.metadata) if chunk.metadata else None
            conn.execute(
                f"INSERT INTO vs_{store_id}_chunks (chunk_id, file_id, chunk_index, content, metadata)"
                f" VALUES (?, ?, ?, ?, ?);",
                (
                    chunk.chunk_id,
                    chunk.file_id,
                    chunk.chunk_index,
                    chunk.content,
                    meta_json,
                ),
            )
        conn.commit()
    finally:
        conn.close()


async def insert_chunks(db_path: str, store_id: str, chunks: list[ChunkRow]) -> None:
    await asyncio.to_thread(_insert_chunks_sync, db_path, store_id, chunks)


def _delete_file_chunks_sync(db_path: str, store_id: str, file_id: str) -> None:
    conn = _get_connection(db_path)
    try:
        chunk_ids = [
            row[0]
            for row in conn.execute(
                f"SELECT chunk_id FROM vs_{store_id}_chunks WHERE file_id = ?;",
                (file_id,),
            ).fetchall()
        ]
        if not chunk_ids:
            return

        placeholders = ",".join("?" for _ in chunk_ids)
        conn.execute(
            f"DELETE FROM vs_{store_id}_vec WHERE chunk_id IN ({placeholders});",
            chunk_ids,
        )
        conn.execute(
            f"DELETE FROM vs_{store_id}_fts WHERE chunk_id IN ({placeholders});",
            chunk_ids,
        )
        conn.execute(
            f"DELETE FROM vs_{store_id}_chunks WHERE file_id = ?;",
            (file_id,),
        )
        conn.commit()
    finally:
        conn.close()


async def delete_file_chunks(db_path: str, store_id: str, file_id: str) -> None:
    await asyncio.to_thread(_delete_file_chunks_sync, db_path, store_id, file_id)


# ---------------------------------------------------------------------------
# Search
# ---------------------------------------------------------------------------


def _vector_search_sync(
    db_path: str,
    store_id: str,
    embedding: list[float],
    k: int = 20,
    file_ids: list[str] | None = None,
) -> list[ScoredChunk]:
    conn = _get_connection(db_path)
    try:
        if file_ids:
            placeholders = ",".join("?" for _ in file_ids)
            query = (
                f"SELECT v.chunk_id, v.file_id, v.chunk_index, v.distance, c.content"
                f" FROM vs_{store_id}_vec v"
                f" JOIN vs_{store_id}_chunks c ON v.chunk_id = c.chunk_id"
                f" WHERE v.embedding MATCH ? AND k = ? AND v.file_id IN ({placeholders})"
                f" ORDER BY v.distance;"
            )
            params: list[Any] = [serialize_vector(embedding), k, *file_ids]
        else:
            query = (
                f"SELECT v.chunk_id, v.file_id, v.chunk_index, v.distance, c.content"
                f" FROM vs_{store_id}_vec v"
                f" JOIN vs_{store_id}_chunks c ON v.chunk_id = c.chunk_id"
                f" WHERE v.embedding MATCH ? AND k = ?"
                f" ORDER BY v.distance;"
            )
            params = [serialize_vector(embedding), k]

        rows = conn.execute(query, params).fetchall()
        return [
            ScoredChunk(
                chunk_id=row[0],
                file_id=row[1],
                chunk_index=row[2],
                score=1.0 / (1.0 + row[3]),
                text=row[4],
            )
            for row in rows
        ]
    finally:
        conn.close()


async def vector_search(
    db_path: str,
    store_id: str,
    embedding: list[float],
    k: int = 20,
    file_ids: list[str] | None = None,
) -> list[ScoredChunk]:
    return await asyncio.to_thread(
        _vector_search_sync, db_path, store_id, embedding, k, file_ids
    )


def _keyword_search_sync(
    db_path: str,
    store_id: str,
    query: str,
    k: int = 20,
    file_ids: list[str] | None = None,
) -> list[ScoredChunk]:
    conn = _get_connection(db_path)
    try:
        sql = (
            f"SELECT f.chunk_id, c.file_id, c.chunk_index, bm25(vs_{store_id}_fts) AS score, c.content"
            f" FROM vs_{store_id}_fts f"
            f" JOIN vs_{store_id}_chunks c ON f.chunk_id = c.chunk_id"
            f" WHERE f.vs_{store_id}_fts MATCH ?"
            f" ORDER BY score"
            f" LIMIT ?;"
        )
        rows = conn.execute(sql, (query, k)).fetchall()

        results = [
            ScoredChunk(
                chunk_id=row[0],
                file_id=row[1],
                chunk_index=row[2],
                score=-row[3],
                text=row[4],
            )
            for row in rows
        ]

        if file_ids:
            allowed = set(file_ids)
            results = [r for r in results if r.file_id in allowed]

        return results
    finally:
        conn.close()


async def keyword_search(
    db_path: str,
    store_id: str,
    query: str,
    k: int = 20,
    file_ids: list[str] | None = None,
) -> list[ScoredChunk]:
    return await asyncio.to_thread(
        _keyword_search_sync, db_path, store_id, query, k, file_ids
    )
