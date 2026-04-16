"""End-to-end tests for the agentic-api gateway with a stubbed LLM."""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Any
from unittest.mock import patch

import httpx
import pytest
from asgi_lifespan import LifespanManager

from pydantic_ai import (
    AgentRunResultEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
)

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.entrypoints.app import create_app
from tests.utils import make_agent_run_result


async def _fake_stream_events(text: str) -> AsyncIterator[Any]:
    part = TextPart(content=text)
    yield PartStartEvent(index=0, part=part)
    yield PartEndEvent(index=0, part=part)
    yield AgentRunResultEvent(result=make_agent_run_result())


def make_agent_stub(text: str):
    async def _stub(*args: Any, **kwargs: Any) -> AsyncIterator[Any]:
        async for event in _fake_stream_events(text):
            yield event

    return _stub


def _build_e2e_runtime_config() -> RuntimeConfig:
    return RuntimeConfig(
        llm_api_base="http://fake-upstream",
        openai_api_key="test-key",
        gateway_host="0.0.0.0",
        gateway_port=9000,
        gateway_workers=1,
        upstream_ready_timeout_s=5.0,
        upstream_ready_interval_s=0.1,
        db_url="sqlite+aiosqlite:///:memory:",
        response_store_enabled=True,
    )


@asynccontextmanager
async def _e2e_client() -> AsyncIterator[httpx.AsyncClient]:
    runtime_config = _build_e2e_runtime_config()
    app = create_app(runtime_config)
    async with LifespanManager(app):
        transport = httpx.ASGITransport(app=app)
        async with httpx.AsyncClient(
            transport=transport, base_url="http://gateway"
        ) as client:
            yield client


def _parse_sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                events.append(json.loads(line[5:]))
            except json.JSONDecodeError:
                pass
    return events


def _sse_completed_response(text: str) -> dict | None:
    for evt in _parse_sse_events(text):
        if evt.get("type") == "response.completed":
            return evt.get("response")
    return None


def _sse_output_text(text: str) -> str:
    parts = []
    for evt in _parse_sse_events(text):
        if evt.get("type") == "response.output_text.delta":
            parts.append(evt.get("delta", ""))
    return "".join(parts)


def _json_output_text(response_body: dict) -> str:
    parts = []
    for item in response_body.get("output", []):
        for c in item.get("content", []):
            if c.get("type") == "output_text":
                parts.append(c.get("text", ""))
    return " ".join(parts)


@pytest.mark.anyio
async def test_single_turn_non_streaming() -> None:
    async with _e2e_client() as client:
        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("ALPHA")):
            resp = await client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "Reply with exactly one word: ALPHA",
                    "stream": False,
                    "response_store_enabled": True,
                },
            )

    assert resp.status_code == 200
    body = resp.json()
    assert body["id"].startswith("resp_")
    assert body["status"] == "completed"
    assert "ALPHA" in _json_output_text(body)


@pytest.mark.anyio
async def test_single_turn_streaming() -> None:
    async with _e2e_client() as client:
        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("BETA")):
            async with client.stream(
                "POST",
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "Reply with exactly one word: BETA",
                    "stream": True,
                    "response_store_enabled": True,
                },
            ) as resp:
                chunks = []
                async for chunk in resp.aiter_text():
                    chunks.append(chunk)

    sse_text = "".join(chunks)
    assert resp.status_code == 200
    assert "data: [DONE]" in sse_text

    completed = _sse_completed_response(sse_text)
    assert completed is not None
    assert completed["id"].startswith("resp_")
    assert completed["status"] == "completed"
    assert "BETA" in _sse_output_text(sse_text)


@pytest.mark.anyio
async def test_multi_turn_non_streaming() -> None:
    async with _e2e_client() as client:
        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("ALPHA")):
            resp1 = await client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "Reply with exactly one word: ALPHA",
                    "stream": False,
                    "response_store_enabled": True,
                },
            )
        assert resp1.status_code == 200
        resp1_id = resp1.json()["id"]

        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("ALPHA")):
            resp2 = await client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "What was the single word I asked you to say in my first message?",
                    "previous_response_id": resp1_id,
                    "stream": False,
                    "response_store_enabled": True,
                },
            )
        assert resp2.status_code == 200
        resp2_body = resp2.json()
        assert resp2_body["id"].startswith("resp_")
        assert resp2_body["id"] != resp1_id
        assert resp2_body["status"] == "completed"
        assert resp2_body.get("previous_response_id") == resp1_id


@pytest.mark.anyio
async def test_multi_turn_streaming() -> None:
    async with _e2e_client() as client:
        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("BETA")):
            async with client.stream(
                "POST",
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "Reply with exactly one word: BETA",
                    "stream": True,
                    "response_store_enabled": True,
                },
            ) as resp:
                sse1 = "".join([chunk async for chunk in resp.aiter_text()])

        completed1 = _sse_completed_response(sse1)
        assert completed1 is not None
        stream1_id = completed1["id"]

        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("BETA")):
            async with client.stream(
                "POST",
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "What was the single word I asked you to say in my first message?",
                    "previous_response_id": stream1_id,
                    "stream": True,
                    "response_store_enabled": True,
                },
            ) as resp:
                sse2 = "".join([chunk async for chunk in resp.aiter_text()])

        assert "data: [DONE]" in sse2
        completed2 = _sse_completed_response(sse2)
        assert completed2 is not None
        assert completed2["id"] != stream1_id


@pytest.mark.anyio
async def test_three_turn_chain() -> None:
    async with _e2e_client() as client:
        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("ALPHA")):
            r1 = await client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "Reply with exactly one word: ALPHA",
                    "stream": False,
                    "response_store_enabled": True,
                },
            )
        assert r1.status_code == 200
        r1_id = r1.json()["id"]

        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("ALPHA")):
            r2 = await client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "Now also say BETA",
                    "previous_response_id": r1_id,
                    "stream": False,
                    "response_store_enabled": True,
                },
            )
        assert r2.status_code == 200
        r2_body = r2.json()
        r2_id = r2_body["id"]
        assert r2_body["previous_response_id"] == r1_id

        with patch(
            "pydantic_ai.Agent.run_stream_events", new=make_agent_stub("ALPHA BETA")
        ):
            r3 = await client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "List every word I asked you to say, one per line.",
                    "previous_response_id": r2_id,
                    "stream": False,
                    "response_store_enabled": True,
                },
            )
        assert r3.status_code == 200
        r3_body = r3.json()
        assert r3_body["status"] == "completed"
        assert r3_body["previous_response_id"] == r2_id


@pytest.mark.anyio
async def test_store_false_not_persisted() -> None:
    async with _e2e_client() as client:
        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("ALPHA")):
            resp = await client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "hello",
                    "stream": False,
                    "response_store_enabled": False,
                },
            )
        assert resp.status_code == 200
        body = resp.json()
        resp_id = body["id"]
        assert body["status"] == "completed"

        with patch("pydantic_ai.Agent.run_stream_events", new=make_agent_stub("hi")):
            follow_up = await client.post(
                "/v1/responses",
                json={
                    "model": "test-model",
                    "input": "follow up",
                    "previous_response_id": resp_id,
                    "stream": False,
                    "response_store_enabled": False,
                },
            )
        assert follow_up.status_code == 400
        assert follow_up.json()["error"]["code"] == "previous_response_not_found"
