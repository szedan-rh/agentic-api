import pytest

from agentic_api.database.vector_search import (
    ChunkRow,
    create_store_tables,
    drop_store_tables,
    insert_chunks,
    keyword_search,
    vector_search,
    delete_file_chunks,
)


DIMENSION = 4
STORE_ID = "test_store"


@pytest.fixture
async def vec_db(tmp_path):
    db_path = str(tmp_path / "test_vec.db")
    await create_store_tables(db_path, STORE_ID, DIMENSION)
    yield db_path
    await drop_store_tables(db_path, STORE_ID)


def _make_chunk(
    chunk_id: str, file_id: str, idx: int, text: str, emb: list[float]
) -> ChunkRow:
    return ChunkRow(
        chunk_id=chunk_id, file_id=file_id, chunk_index=idx, content=text, embedding=emb
    )


@pytest.mark.anyio
async def test_insert_and_vector_search(vec_db):
    chunks = [
        _make_chunk("c1", "f1", 0, "hello world", [1.0, 0.0, 0.0, 0.0]),
        _make_chunk("c2", "f1", 1, "goodbye world", [0.0, 1.0, 0.0, 0.0]),
        _make_chunk("c3", "f2", 0, "different file", [0.0, 0.0, 1.0, 0.0]),
    ]
    await insert_chunks(vec_db, STORE_ID, chunks)

    results = await vector_search(vec_db, STORE_ID, [1.0, 0.0, 0.0, 0.0], k=3)
    assert len(results) == 3
    assert results[0].chunk_id == "c1"
    assert results[0].score > 0


@pytest.mark.anyio
async def test_vector_search_with_file_filter(vec_db):
    chunks = [
        _make_chunk("c1", "f1", 0, "text a", [1.0, 0.0, 0.0, 0.0]),
        _make_chunk("c2", "f2", 0, "text b", [0.9, 0.1, 0.0, 0.0]),
    ]
    await insert_chunks(vec_db, STORE_ID, chunks)

    results = await vector_search(
        vec_db, STORE_ID, [1.0, 0.0, 0.0, 0.0], k=10, file_ids=["f2"]
    )
    assert len(results) == 1
    assert results[0].file_id == "f2"


@pytest.mark.anyio
async def test_keyword_search(vec_db):
    chunks = [
        _make_chunk("c1", "f1", 0, "machine learning is great", [0.0, 0.0, 0.0, 0.0]),
        _make_chunk("c2", "f1", 1, "deep learning networks", [0.0, 0.0, 0.0, 0.0]),
        _make_chunk("c3", "f1", 2, "unrelated content here", [0.0, 0.0, 0.0, 0.0]),
    ]
    await insert_chunks(vec_db, STORE_ID, chunks)

    results = await keyword_search(vec_db, STORE_ID, "learning", k=10)
    assert len(results) >= 2
    texts = [r.text for r in results]
    assert any("learning" in t for t in texts)


@pytest.mark.anyio
async def test_delete_file_chunks_removes_all(vec_db):
    chunks = [
        _make_chunk("c1", "f1", 0, "keep this", [1.0, 0.0, 0.0, 0.0]),
        _make_chunk("c2", "f_del", 0, "delete this", [0.0, 1.0, 0.0, 0.0]),
        _make_chunk("c3", "f_del", 1, "also delete", [0.0, 0.0, 1.0, 0.0]),
    ]
    await insert_chunks(vec_db, STORE_ID, chunks)

    await delete_file_chunks(vec_db, STORE_ID, "f_del")

    results = await vector_search(vec_db, STORE_ID, [0.0, 1.0, 0.0, 0.0], k=10)
    file_ids = {r.file_id for r in results}
    assert "f_del" not in file_ids
    assert "f1" in file_ids
