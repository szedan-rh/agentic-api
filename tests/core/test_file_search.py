import json
from unittest.mock import AsyncMock

import pytest

from agentic_api.core.file_search import FileSearchExecutor
from agentic_api.types.vector_stores import (
    FileSearchTool,
    SearchResult,
    VectorStoreSearchResponse,
)


@pytest.fixture
def mock_manager():
    manager = AsyncMock()
    manager.search = AsyncMock(
        return_value=VectorStoreSearchResponse(
            data=[
                SearchResult(
                    file_id="file-abc",
                    filename="test.txt",
                    score=0.95,
                    text="relevant content here",
                    chunk_index=0,
                ),
                SearchResult(
                    file_id="file-abc",
                    filename="test.txt",
                    score=0.80,
                    text="also relevant",
                    chunk_index=1,
                ),
            ]
        )
    )
    return manager


@pytest.mark.anyio
async def test_execute_returns_json(mock_manager):
    config = FileSearchTool(vector_store_ids=["vs_123"], max_num_results=10)
    executor = FileSearchExecutor(vector_store_manager=mock_manager, tool_config=config)

    result = await executor.execute("test query")
    parsed = json.loads(result)
    assert len(parsed) == 2
    assert parsed[0]["file_id"] == "file-abc"
    assert parsed[0]["score"] == 0.95


@pytest.mark.anyio
async def test_execute_searches_all_stores(mock_manager):
    config = FileSearchTool(vector_store_ids=["vs_1", "vs_2"], max_num_results=20)
    executor = FileSearchExecutor(vector_store_manager=mock_manager, tool_config=config)

    await executor.execute("query")
    assert mock_manager.search.call_count == 2


@pytest.mark.anyio
async def test_execute_respects_max_results(mock_manager):
    config = FileSearchTool(vector_store_ids=["vs_1"], max_num_results=1)
    executor = FileSearchExecutor(vector_store_manager=mock_manager, tool_config=config)

    result = await executor.execute("query")
    parsed = json.loads(result)
    assert len(parsed) == 1
