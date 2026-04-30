from agentic_api.store.chunker import chunk_text


def test_empty_input():
    assert chunk_text("") == []
    assert chunk_text("   ") == []


def test_single_chunk():
    result = chunk_text(
        "Hello world", max_chunk_size_tokens=100, chunk_overlap_tokens=0
    )
    assert len(result) == 1
    assert result[0].chunk_index == 0
    assert "Hello world" in result[0].text


def test_multiple_chunks_with_overlap():
    long_text = " ".join(f"word{i}" for i in range(500))
    result = chunk_text(long_text, max_chunk_size_tokens=50, chunk_overlap_tokens=10)
    assert len(result) > 1
    for i, chunk in enumerate(result):
        assert chunk.chunk_index == i
        assert chunk.token_count <= 50


def test_chunk_indices_are_sequential():
    text = " ".join(f"token{i}" for i in range(200))
    result = chunk_text(text, max_chunk_size_tokens=30, chunk_overlap_tokens=10)
    indices = [c.chunk_index for c in result]
    assert indices == list(range(len(result)))


def test_no_overlap():
    text = "a " * 100
    result = chunk_text(text, max_chunk_size_tokens=20, chunk_overlap_tokens=0)
    assert len(result) >= 5
    for c in result:
        assert c.token_count <= 20
