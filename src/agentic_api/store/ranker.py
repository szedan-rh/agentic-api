"""Result fusion utilities for hybrid search."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class ScoredChunk:
    chunk_id: str
    file_id: str
    score: float
    text: str
    chunk_index: int


def reciprocal_rank_fusion(
    result_lists: list[list[ScoredChunk]],
    *,
    k: int = 60,
    max_results: int = 20,
) -> list[ScoredChunk]:
    """Merge multiple ranked lists using Reciprocal Rank Fusion.

    RRF score for each chunk = sum(1 / (k + rank)) across all lists where the
    chunk appears.  Ranks are 1-based.
    """
    scores: dict[str, float] = defaultdict(float)
    best: dict[str, ScoredChunk] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            scores[chunk.chunk_id] += 1.0 / (k + rank)
            if chunk.chunk_id not in best or chunk.score > best[chunk.chunk_id].score:
                best[chunk.chunk_id] = chunk

    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return [
        ScoredChunk(
            chunk_id=cid,
            file_id=best[cid].file_id,
            score=rrf_score,
            text=best[cid].text,
            chunk_index=best[cid].chunk_index,
        )
        for cid, rrf_score in ranked[:max_results]
    ]


def _min_max_normalize(chunks: list[ScoredChunk]) -> dict[str, float]:
    """Normalize scores to [0, 1] via min-max scaling."""
    if not chunks:
        return {}
    scores = [c.score for c in chunks]
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi != lo else 1.0
    return {c.chunk_id: (c.score - lo) / span for c in chunks}


def weighted_rerank(
    vector_results: list[ScoredChunk],
    keyword_results: list[ScoredChunk],
    *,
    vector_weight: float = 0.7,
    keyword_weight: float = 0.3,
    max_results: int = 20,
) -> list[ScoredChunk]:
    """Combine vector and keyword results using weighted score fusion."""
    vec_norm = _min_max_normalize(vector_results)
    kw_norm = _min_max_normalize(keyword_results)

    all_ids = set(vec_norm) | set(kw_norm)
    best: dict[str, ScoredChunk] = {}
    for c in [*vector_results, *keyword_results]:
        if c.chunk_id not in best:
            best[c.chunk_id] = c

    combined: list[ScoredChunk] = []
    for cid in all_ids:
        score = vector_weight * vec_norm.get(cid, 0.0) + keyword_weight * kw_norm.get(
            cid, 0.0
        )
        ref = best[cid]
        combined.append(
            ScoredChunk(
                chunk_id=cid,
                file_id=ref.file_id,
                score=score,
                text=ref.text,
                chunk_index=ref.chunk_index,
            )
        )

    combined.sort(key=lambda c: c.score, reverse=True)
    return combined[:max_results]
