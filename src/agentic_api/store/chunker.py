"""Static token-based text chunking using tiktoken."""

from __future__ import annotations

from dataclasses import dataclass

import tiktoken


@dataclass(frozen=True, slots=True)
class Chunk:
    text: str
    chunk_index: int
    token_count: int


def chunk_text(
    text: str,
    *,
    max_chunk_size_tokens: int = 800,
    chunk_overlap_tokens: int = 400,
    encoding_name: str = "cl100k_base",
) -> list[Chunk]:
    """Split *text* into overlapping token-window chunks.

    Returns an empty list for empty/whitespace-only input.  Text shorter than
    one full window produces a single chunk.
    """
    if not text or not text.strip():
        return []

    enc = tiktoken.get_encoding(encoding_name)
    tokens = enc.encode(text)
    total = len(tokens)

    if total == 0:
        return []

    step = max(1, max_chunk_size_tokens - chunk_overlap_tokens)
    chunks: list[Chunk] = []
    idx = 0
    start = 0

    while start < total:
        end = min(start + max_chunk_size_tokens, total)
        window = tokens[start:end]
        chunks.append(
            Chunk(
                text=enc.decode(window),
                chunk_index=idx,
                token_count=len(window),
            )
        )
        idx += 1
        start += step

    return chunks
