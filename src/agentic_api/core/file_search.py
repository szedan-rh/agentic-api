"""FileSearchExecutor — executes file_search tool calls against VectorStoreManager."""

from __future__ import annotations

import json

from agentic_api.store.vector_store import VectorStoreManager
from agentic_api.types.vector_stores import FileSearchTool, RankingOptions


class FileSearchExecutor:
    def __init__(
        self,
        *,
        vector_store_manager: VectorStoreManager,
        tool_config: FileSearchTool,
    ) -> None:
        self._manager = vector_store_manager
        self._config = tool_config

    async def execute(self, query: str) -> str:
        all_results = []
        for store_id in self._config.vector_store_ids:
            response = await self._manager.search(
                store_id=store_id,
                query=query,
                max_num_results=self._config.max_num_results,
                search_mode="hybrid",
                ranking_options=self._config.ranking_options or RankingOptions(),
            )
            all_results.extend(response.data)

        all_results.sort(key=lambda r: r.score, reverse=True)
        all_results = all_results[: self._config.max_num_results]

        formatted = []
        for r in all_results:
            formatted.append(
                {
                    "file_id": r.file_id,
                    "filename": r.filename,
                    "score": round(r.score, 4),
                    "text": r.text,
                }
            )
        return json.dumps(formatted, separators=(",", ":"))
