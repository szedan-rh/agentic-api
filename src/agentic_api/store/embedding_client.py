"""Async httpx client for the OpenAI-compatible embeddings API."""

from __future__ import annotations

import httpx


class EmbeddingClient:
    """Thin async wrapper around ``POST /v1/embeddings``.

    Batches large inputs and caches the detected embedding dimension.
    """

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None = None,
        model: str,
        batch_size: int = 128,
    ) -> None:
        headers: dict[str, str] = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._client = httpx.AsyncClient(
            base_url=base_url,
            headers=headers,
            timeout=httpx.Timeout(60.0, connect=10.0),
        )
        self._model = model
        self._batch_size = batch_size
        self._dimension: int | None = None

    async def embed(self, texts: list[str]) -> list[list[float]]:
        """Embed a list of texts, batching as needed. Returns one vector per input text."""
        all_embeddings: list[list[float]] = []

        for start in range(0, len(texts), self._batch_size):
            batch = texts[start : start + self._batch_size]
            resp = await self._client.post(
                "/v1/embeddings",
                json={"model": self._model, "input": batch},
            )
            resp.raise_for_status()
            data = resp.json()

            sorted_items = sorted(data["data"], key=lambda d: d["index"])
            embeddings = [item["embedding"] for item in sorted_items]
            all_embeddings.extend(embeddings)

            if self._dimension is None and embeddings:
                self._dimension = len(embeddings[0])

        return all_embeddings

    async def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        results = await self.embed([text])
        return results[0]

    @property
    def dimension(self) -> int | None:
        """Return the embedding dimension detected from the last call, or None."""
        return self._dimension

    async def aclose(self) -> None:
        """Close the underlying httpx client."""
        await self._client.aclose()
