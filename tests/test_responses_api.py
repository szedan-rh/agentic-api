"""Cassette-based end-to-end tests for the Responses API (previous_response_id continuity).

Each test loads one multi-turn cassette from tests/cassettes/text_only/responses/
and drives the gateway through the full scenario without a live model.

Cassettes available:
  resp-single-gpt-4o-nonstreaming.yaml    1 turn, non-streaming
  resp-single-gpt-4o-streaming.yaml       1 turn, streaming
  resp-two-turn-gpt-4o-nonstreaming.yaml  2 turns, non-streaming, previous_response_id chaining
  resp-two-turn-gpt-4o-streaming.yaml     2 turns, streaming, previous_response_id chaining
  resp-no-store-gpt-4o-nonstreaming.yaml  store=false — follow-up must return 400
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import httpx
import pytest

from tests.utils.gateway import (
    json_output_text,
    load_turns,
    post_response,
    sse_completed_response,
    sse_output_text,
    stream_response,
)

_CASSETTE_DIR = Path(__file__).parent / "cassettes" / "text_only" / "responses"


# ── case 1: single-turn non-streaming ─────────────────────────────────────────


@pytest.mark.anyio
async def test_single_turn_nonstreaming(
    responses_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = responses_gateway_client
    use_cassette("resp-single-gpt-4o-nonstreaming.yaml")
    (t1,) = load_turns(_CASSETTE_DIR / "resp-single-gpt-4o-nonstreaming.yaml")

    body = await post_response(
        client,
        model="test-model",
        input=t1.input,
        stream=False,
        response_store_enabled=True,
    )
    assert body["id"].startswith("resp_")
    assert body["status"] == "completed"
    assert json_output_text(body) == t1.output


# ── case 2: single-turn streaming ─────────────────────────────────────────────


@pytest.mark.anyio
async def test_single_turn_streaming(
    responses_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = responses_gateway_client
    use_cassette("resp-single-gpt-4o-streaming.yaml")
    (t1,) = load_turns(_CASSETTE_DIR / "resp-single-gpt-4o-streaming.yaml")

    sse = await stream_response(
        client,
        model="test-model",
        input=t1.input,
        stream=True,
        response_store_enabled=True,
    )
    assert "data: [DONE]" in sse
    completed = sse_completed_response(sse)
    assert completed is not None
    assert completed["id"].startswith("resp_")
    assert completed["status"] == "completed"
    assert sse_output_text(sse) == t1.output


# ── case 3: two-turn non-streaming via previous_response_id ───────────────────


@pytest.mark.anyio
async def test_two_turn_nonstreaming_previous_response_id(
    responses_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = responses_gateway_client
    use_cassette("resp-two-turn-gpt-4o-nonstreaming.yaml")
    t1, t2 = load_turns(_CASSETTE_DIR / "resp-two-turn-gpt-4o-nonstreaming.yaml")

    b1 = await post_response(
        client,
        model="test-model",
        input=t1.input,
        stream=False,
        response_store_enabled=True,
    )
    assert b1["id"].startswith("resp_")
    assert b1["status"] == "completed"
    assert json_output_text(b1) == t1.output

    b2 = await post_response(
        client,
        model="test-model",
        input=t2.input,
        stream=False,
        response_store_enabled=True,
        previous_response_id=b1["id"],
    )
    assert b2["id"].startswith("resp_")
    assert b2["id"] != b1["id"]
    assert b2["status"] == "completed"
    assert b2["previous_response_id"] == b1["id"]
    assert json_output_text(b2) == t2.output


# ── case 4: two-turn streaming via previous_response_id ───────────────────────


@pytest.mark.anyio
async def test_two_turn_streaming_previous_response_id(
    responses_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = responses_gateway_client
    use_cassette("resp-two-turn-gpt-4o-streaming.yaml")
    t1, t2 = load_turns(_CASSETTE_DIR / "resp-two-turn-gpt-4o-streaming.yaml")

    sse1 = await stream_response(
        client,
        model="test-model",
        input=t1.input,
        stream=True,
        response_store_enabled=True,
    )
    assert "data: [DONE]" in sse1
    c1 = sse_completed_response(sse1)
    assert c1 is not None
    assert c1["id"].startswith("resp_")
    assert c1["status"] == "completed"
    assert sse_output_text(sse1) == t1.output

    sse2 = await stream_response(
        client,
        model="test-model",
        input=t2.input,
        stream=True,
        response_store_enabled=True,
        previous_response_id=c1["id"],
    )
    assert "data: [DONE]" in sse2
    c2 = sse_completed_response(sse2)
    assert c2 is not None
    assert c2["id"] != c1["id"]
    assert c2["status"] == "completed"
    assert sse_output_text(sse2) == t2.output


# ── case 5: response_store_enabled=false — not reusable as previous_response_id


@pytest.mark.anyio
async def test_store_disabled_not_reusable(
    responses_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = responses_gateway_client
    use_cassette("resp-no-store-gpt-4o-nonstreaming.yaml")
    t1 = load_turns(_CASSETTE_DIR / "resp-no-store-gpt-4o-nonstreaming.yaml")[0]

    ns = await post_response(
        client,
        model="test-model",
        input=t1.input,
        stream=False,
        response_store_enabled=False,
    )
    assert ns["status"] == "completed"

    # Follow-up using that id — gateway returns 400 without hitting upstream
    follow = await client.post(
        "/v1/responses",
        json={
            "model": "test-model",
            "input": "follow up",
            "previous_response_id": ns["id"],
            "stream": False,
            "response_store_enabled": False,
        },
    )
    assert follow.status_code == 400
    assert follow.json()["error"]["code"] == "previous_response_not_found"
