from __future__ import annotations

import asyncio
import os

import logging

from sqlalchemy.ext.asyncio import AsyncEngine

from agentic_api.database import Base
from agentic_api.database import conversation, item, response  # noqa: F401
from agentic_api.database.db_engine import postgres_advisory_lock

logger = logging.getLogger(__name__)

POSTGRES_SCHEMA_LOCK_NAME = "agentic-api:responses_state_schema_v2"


class SchemaManager:
    """Owns DDL lifecycle for the three-table schema (Item, Response, Conversation).

    Responsibilities:
    - Run Base.metadata.create_all() exactly once per process.
    - Guard against concurrent coroutines racing at cold start (asyncio.Lock).
    - Respect AA_DB_SCHEMA_READY env var set by the supervisor so worker
      processes skip DDL entirely.

    Intended usage: instantiate once at startup, call ensure_ready() before
    accepting requests.
    """

    def __init__(self, engine: AsyncEngine) -> None:
        self._engine = engine
        self._lock = asyncio.Lock()
        self._ready = False

    def _is_marked_ready(self) -> bool:
        """True when the supervisor already ran DDL."""
        value = os.environ.get("AA_DB_SCHEMA_READY", "").strip().lower()
        return value in {"1", "true", "t", "yes", "y", "on"}

    async def ensure_ready(
        self, *, gateway_workers: int = 1, db_dialect: str = "sqlite"
    ) -> None:
        """Create tables if they don't exist. Idempotent and coroutine-safe.

        After the first successful call this is a no-op for the process lifetime.
        """
        if self._ready:
            return
        async with self._lock:
            if self._ready:
                return
            if self._is_marked_ready():
                logger.debug("[schema] DDL skipped — marked ready by supervisor.")
                self._ready = True
                return
            if db_dialect == "sqlite" and gateway_workers > 1:
                raise RuntimeError(
                    "SQLite schema initialization is not multi-worker safe when started directly. "
                    "Use `agentic-api serve` (recommended) or run with AA_WORKERS=1."
                )
            logger.debug("[schema] Running DDL (create_all)...")
            async with self._engine.begin() as conn:
                if db_dialect == "postgresql":
                    async with postgres_advisory_lock(
                        conn, name=POSTGRES_SCHEMA_LOCK_NAME
                    ):
                        await conn.run_sync(Base.metadata.create_all)
                else:
                    await conn.run_sync(Base.metadata.create_all)
            self._ready = True
            logger.info("[schema] DB schema ready.")
