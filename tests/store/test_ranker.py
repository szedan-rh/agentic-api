from agentic_api.store.ranker import (
    ScoredChunk,
    reciprocal_rank_fusion,
    weighted_rerank,
)


def _chunk(
    cid: str, score: float, file_id: str = "f1", chunk_index: int = 0
) -> ScoredChunk:
    return ScoredChunk(
        chunk_id=cid,
        file_id=file_id,
        score=score,
        text=f"text-{cid}",
        chunk_index=chunk_index,
    )


def test_rrf_single_list():
    results = reciprocal_rank_fusion(
        [[_chunk("a", 0.9), _chunk("b", 0.8), _chunk("c", 0.7)]],
        k=60,
        max_results=10,
    )
    assert len(results) == 3
    assert results[0].chunk_id == "a"
    assert results[1].chunk_id == "b"
    assert results[2].chunk_id == "c"


def test_rrf_two_lists_with_overlap():
    list_a = [_chunk("a", 0.9), _chunk("b", 0.8)]
    list_b = [_chunk("b", 0.95), _chunk("c", 0.7)]
    results = reciprocal_rank_fusion([list_a, list_b], k=60, max_results=10)
    ids = [r.chunk_id for r in results]
    assert "b" in ids
    assert ids.index("b") == 0


def test_rrf_max_results():
    chunks = [_chunk(f"c{i}", 1.0 - i * 0.1) for i in range(10)]
    results = reciprocal_rank_fusion([chunks], k=60, max_results=3)
    assert len(results) == 3


def test_weighted_rerank_basic():
    vec = [_chunk("a", 0.9), _chunk("b", 0.5)]
    kw = [_chunk("b", 0.8), _chunk("c", 0.6)]
    results = weighted_rerank(
        vec, kw, vector_weight=0.7, keyword_weight=0.3, max_results=10
    )
    ids = [r.chunk_id for r in results]
    assert "a" in ids
    assert "b" in ids
    assert "c" in ids


def test_weighted_rerank_max_results():
    vec = [_chunk(f"v{i}", 1.0 - i * 0.1) for i in range(5)]
    kw = [_chunk(f"k{i}", 1.0 - i * 0.1) for i in range(5)]
    results = weighted_rerank(vec, kw, max_results=3)
    assert len(results) == 3


def test_weighted_rerank_empty_inputs():
    results = weighted_rerank([], [], max_results=10)
    assert results == []
