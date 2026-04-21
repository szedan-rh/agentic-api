"""Cassette replay utilities for agentic-api tests.

Cassettes record upstream LLM HTTP request/response pairs in YAML format
matching the vllm upstream_responses cassette format.  Tests inject a
CassetteReplayer into the mock upstream ASGI app so each request consumes the
next cassette in order — no real LLM required.
"""

from __future__ import annotations

import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True, slots=True)
class CassetteRequest:
    method: str
    path: str
    query_params: dict[str, Any]
    body: dict[str, Any]


@dataclass(frozen=True, slots=True)
class CassetteResponse:
    status_code: int
    headers: dict[str, str]
    body: dict[str, Any] | None
    sse: list[str] | None

    @property
    def is_stream(self) -> bool:
        return self.sse is not None


@dataclass(frozen=True, slots=True)
class Cassette:
    filename: str
    request: CassetteRequest
    response: CassetteResponse


def _parse_cassette(raw: dict, fallback_filename: str) -> Cassette:
    request = raw.get("request") or {}
    response = raw.get("response") or {}
    return Cassette(
        filename=str(raw.get("filename") or fallback_filename),
        request=CassetteRequest(
            method=str((request.get("method") or "POST")).upper(),
            path=str(request.get("path") or ""),
            query_params=dict(request.get("query_params") or {}),
            body=dict((request.get("body") or {})),
        ),
        response=CassetteResponse(
            status_code=int(response.get("status_code") or 200),
            headers={
                str(k): str(v) for k, v in (response.get("headers") or {}).items()
            },
            body=response.get("body"),
            sse=response.get("sse"),
        ),
    )


def load_cassette_yaml(path: Path) -> Cassette:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    return _parse_cassette(raw, path.stem)


def load_multi_turn_cassette_yaml(path: Path) -> list[Cassette]:
    """Load a multi-turn cassette file (has a top-level 'turns:' list).

    Each turn is a single cassette record. Returns cassettes in order.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    turns = raw.get("turns")
    if turns is None:
        # Fallback: treat as a single-cassette file.
        return [_parse_cassette(raw, path.stem)]
    stem = path.stem
    return [_parse_cassette(turn, f"{stem}-t{i + 1}") for i, turn in enumerate(turns)]


class CassetteReplayError(RuntimeError):
    pass


class CassetteQueue:
    """Ordered, stateful cassette queue for deterministic replay.

    Each incoming request consumes exactly one cassette in order.
    """

    def __init__(self, *, name: str, cassettes: list[Cassette]) -> None:
        if not cassettes:
            raise ValueError("Queue must contain at least one cassette.")
        self.name = name
        self._cassettes = cassettes
        self._cursor = 0
        self._lock = threading.Lock()

    def reset(self) -> None:
        with self._lock:
            self._cursor = 0

    def consume_next(self) -> Cassette:
        with self._lock:
            if self._cursor >= len(self._cassettes):
                raise CassetteReplayError(
                    f"Queue '{self.name}' exhausted (len={len(self._cassettes)})."
                )
            cassette = self._cassettes[self._cursor]
            self._cursor += 1
            return cassette


class CassetteReplayer:
    """Replay cassettes deterministically for a single test scenario."""

    def __init__(self, *, cassettes: list[Cassette]) -> None:
        self._queue = CassetteQueue(name="default", cassettes=cassettes)

    def reset(self) -> None:
        self._queue.reset()

    def next_response(self, *, stream: bool) -> CassetteResponse:
        cassette = self._queue.consume_next()
        # pydantic_ai always streams upstream; don't enforce stream flag matching.
        if cassette.response.is_stream and cassette.response.sse is None:
            raise CassetteReplayError(
                f"Cassette '{cassette.filename}' has no sse data."
            )
        if not cassette.response.is_stream and cassette.response.body is None:
            raise CassetteReplayError(
                f"Cassette '{cassette.filename}' has no body data."
            )
        return cassette.response


async def stream_sse_chunks(chunks: list[str]) -> AsyncIterator[bytes]:
    for chunk in chunks:
        yield chunk.encode("utf-8")


def _synthesize_sse_from_body(body: dict) -> list[str]:
    """Convert a non-streaming JSON response body into minimal SSE events.

    pydantic_ai always calls request_stream on the upstream, so even cassettes
    recorded with stream=false must be served as SSE.  We emit the minimum set
    of events that pydantic_ai's OpenAIResponsesModel needs to produce a result:
    response.created → (per text part) response.output_text.delta → response.completed.
    """
    import json as _json

    seq = 0
    events: list[str] = []

    def _evt(type_: str, data: dict) -> None:
        nonlocal seq
        data["sequence_number"] = seq
        seq += 1
        events.append(f"event: {type_}\n")
        events.append(f"data: {_json.dumps(data, separators=(',', ':'))}\n\n")

    created_response = dict(body)
    created_response["status"] = "in_progress"
    created_response["output"] = []
    _evt("response.created", {"type": "response.created", "response": created_response})

    for output_index, item in enumerate(body.get("output", [])):
        item_id = item.get("id", f"msg_{output_index}")
        for content_index, part in enumerate(item.get("content", [])):
            if part.get("type") == "output_text":
                text = part.get("text", "")
                _evt(
                    "response.output_text.delta",
                    {
                        "type": "response.output_text.delta",
                        "content_index": content_index,
                        "delta": text,
                        "item_id": item_id,
                        "output_index": output_index,
                    },
                )

    _evt("response.completed", {"type": "response.completed", "response": body})
    events.append("data: [DONE]\n\n")
    return events


def make_replayer(*filenames: str, cassette_dir: Path) -> CassetteReplayer:
    cassettes = [load_cassette_yaml(cassette_dir / f) for f in filenames]
    return CassetteReplayer(cassettes=cassettes)


def make_replayer_from_multi_turn(
    filename: str, *, cassette_dir: Path
) -> CassetteReplayer:
    """Load all turns from a single multi-turn YAML file into one replayer.

    Turns whose path is not /v1/responses are skipped — they record upstream
    calls (e.g. POST /v1/conversations) that agentic-api handles internally.
    """
    cassettes = [
        c
        for c in load_multi_turn_cassette_yaml(cassette_dir / filename)
        if c.request.path == "/v1/responses"
    ]
    return CassetteReplayer(cassettes=cassettes)


def cassettes_dir() -> Path:
    return Path(__file__).parent.parent / "cassettes"


def build_cassette_upstream(replayer_holder: list[CassetteReplayer | None]):
    """Build a Starlette upstream stub that serves responses from a CassetteReplayer.

    The replayer is held in a mutable list so tests can swap it per-test:
        replayer_holder[0] = make_replayer("cassette1.yaml", "cassette2.yaml", cassette_dir=...)
    """

    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, Response, StreamingResponse
    from starlette.routing import Route

    async def models(request: Request) -> Response:
        return JSONResponse(
            {"object": "list", "data": [{"id": "test-model", "object": "model"}]}
        )

    async def responses(request: Request) -> Response:
        replayer = replayer_holder[0]
        if replayer is None:
            return JSONResponse({"error": "No cassette replayer set"}, status_code=500)

        body = await request.json()
        stream = bool(body.get("stream"))

        cassette_resp = replayer.next_response(stream=stream)
        headers = cassette_resp.headers.copy()

        if cassette_resp.is_stream:
            sse_chunks = cassette_resp.sse or []
        else:
            # pydantic_ai always uses request_stream internally, so we must serve
            # SSE even for cassettes recorded with stream=false.
            sse_chunks = _synthesize_sse_from_body(cassette_resp.body or {})

        return StreamingResponse(
            stream_sse_chunks(sse_chunks),
            status_code=cassette_resp.status_code,
            headers={
                "content-type": headers.get(
                    "content-type", "text/event-stream; charset=utf-8"
                )
            },
            media_type="text/event-stream",
        )

    return Starlette(
        routes=[
            Route("/v1/models", models, methods=["GET"]),
            Route("/v1/responses", responses, methods=["GET", "POST"]),
        ]
    )
