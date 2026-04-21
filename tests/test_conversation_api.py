"""Cassette-based end-to-end tests for the conversation API.

Each test loads one multi-turn cassette from tests/cassettes/text_only/conversation/
and drives the gateway through the recorded scenario without a live model.

The cassettes were recorded against OpenAI directly (via record_cassette.py).
Turns with path=/v1/conversations are skipped by the replayer — agentic-api
manages conversations internally via its own DB store, not by calling the upstream.

agentic-api differences from OpenAI request body:
  - Uses response_store_enabled / conversation_store_enabled (bool) instead of store (bool)
  - Uses conversation_id instead of conversation (str)
  - previous_response_id is the same

Cassettes available:
  conv-two-turn-gpt-4o-nonstreaming.yaml        2 turns, non-streaming
  conv-two-turn-gpt-4o-streaming.yaml           2 turns, streaming
  conv-isolation-gpt-4o-nonstreaming.yaml       2 independent conversations, 3 turns each
  conv-multi-turn-single-branch-gpt-4o-nonstreaming.yaml  3-turn chain + 1 branch turn off turn 1
  conv-multi-branch-multi-turn-gpt-4o-nonstreaming.yaml   5-turn chain with 2 inline branches
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

_CASSETTE_DIR = Path(__file__).parent / "cassettes" / "text_only" / "conversation"

_CONV_BASE = {
    "model": "test-model",
    "stream": False,
    "response_store_enabled": True,
    "conversation_store_enabled": True,
}


def _turns(filename: str):
    return load_turns(_CASSETTE_DIR / filename)


# ── test 6: 2-turn non-streaming via conversation_id ─────────────────────────
# Cassette: conv-two-turn-gpt-4o-nonstreaming.yaml
# Turn 1: "Remember the word CHERRY. Just say: OK"
# Turn 2: "What word did I ask you to remember?"


@pytest.mark.anyio
async def test_two_turn_nonstreaming(
    conversation_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = conversation_gateway_client
    use_cassette("conv-two-turn-gpt-4o-nonstreaming.yaml")
    t1, t2 = _turns("conv-two-turn-gpt-4o-nonstreaming.yaml")

    # Turn 1: no conversation_id — gateway creates one and returns it
    b1 = await post_response(client, **_CONV_BASE, input=t1.input)
    assert b1["id"].startswith("resp_")
    assert b1["status"] == "completed"
    assert b1["conversation_id"] is not None
    assert json_output_text(b1) == t1.output

    # Turn 2: pass conversation_id — gateway rehydrates history
    b2 = await post_response(
        client, **_CONV_BASE, input=t2.input, conversation_id=b1["conversation_id"]
    )
    assert b2["id"].startswith("resp_")
    assert b2["id"] != b1["id"]
    assert b2["status"] == "completed"
    assert json_output_text(b2) == t2.output


# ── test 7: 2-turn streaming via conversation_id ──────────────────────────────
# Cassette: conv-two-turn-gpt-4o-streaming.yaml
# Turn 1: "Remember the word MANGO. Just say: OK"
# Turn 2: "What word did I ask you to remember?"


@pytest.mark.anyio
async def test_two_turn_streaming(
    conversation_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = conversation_gateway_client
    use_cassette("conv-two-turn-gpt-4o-streaming.yaml")
    t1, t2 = _turns("conv-two-turn-gpt-4o-streaming.yaml")

    _stream_base = {**_CONV_BASE, "stream": True}

    # Turn 1: no conversation_id — extract conv_id from response.completed event
    sse1 = await stream_response(client, **_stream_base, input=t1.input)
    c1 = sse_completed_response(sse1)
    assert c1 is not None
    assert c1["id"].startswith("resp_")
    assert c1["status"] == "completed"
    assert c1["conversation_id"] is not None
    assert sse_output_text(sse1) == t1.output

    # Turn 2: pass conversation_id
    sse2 = await stream_response(
        client, **_stream_base, input=t2.input, conversation_id=c1["conversation_id"]
    )
    c2 = sse_completed_response(sse2)
    assert c2 is not None
    assert c2["id"] != c1["id"]
    assert c2["status"] == "completed"
    assert sse_output_text(sse2) == t2.output


# ── test 8: conversation isolation — 2 independent conversations ──────────────
# Cassette: conv-isolation-gpt-4o-nonstreaming.yaml
# Conv A turns 1-3, then Conv B turns 1-3 (order matches record_cassette.py run_isolation)


@pytest.mark.anyio
async def test_conversation_isolation(
    conversation_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = conversation_gateway_client
    use_cassette("conv-isolation-gpt-4o-nonstreaming.yaml")
    ta1, ta2, ta3, tb1, tb2, tb3 = _turns("conv-isolation-gpt-4o-nonstreaming.yaml")

    # Conv A
    ba1 = await post_response(client, **_CONV_BASE, input=ta1.input)
    conv_a = ba1["conversation_id"]
    assert conv_a is not None
    assert json_output_text(ba1) == ta1.output

    ba2 = await post_response(
        client, **_CONV_BASE, input=ta2.input, conversation_id=conv_a
    )
    assert json_output_text(ba2) == ta2.output

    ba3 = await post_response(
        client, **_CONV_BASE, input=ta3.input, conversation_id=conv_a
    )
    assert json_output_text(ba3) == ta3.output

    # Conv B — must get a different conversation_id
    bb1 = await post_response(client, **_CONV_BASE, input=tb1.input)
    conv_b = bb1["conversation_id"]
    assert conv_b is not None
    assert conv_b != conv_a
    assert json_output_text(bb1) == tb1.output

    bb2 = await post_response(
        client, **_CONV_BASE, input=tb2.input, conversation_id=conv_b
    )
    assert json_output_text(bb2) == tb2.output

    bb3 = await post_response(
        client, **_CONV_BASE, input=tb3.input, conversation_id=conv_b
    )
    assert json_output_text(bb3) == tb3.output


# ── test 9: 3-turn conv chain then branch off turn 1 ─────────────────────────
# Cassette: conv-branch-gpt-4o-nonstreaming.yaml
# Turn 1: "What is 2+2?" → 4  |  Turn 2: +1 → 5  |  Turn 3: +2 → 7
# Branch (off turn 1, answer=4): +1 → 5  (turn 2/3 context is absent)


@pytest.mark.anyio
async def test_branch_off_turn_1(
    conversation_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = conversation_gateway_client
    use_cassette("conv-multi-turn-single-branch-gpt-4o-nonstreaming.yaml")
    t1, t2, t3, t4 = _turns("conv-multi-turn-single-branch-gpt-4o-nonstreaming.yaml")

    b1 = await post_response(client, **_CONV_BASE, input=t1.input)
    assert b1["status"] == "completed"
    assert json_output_text(b1) == t1.output
    conv_id, r1_id = b1["conversation_id"], b1["id"]

    b2 = await post_response(
        client, **_CONV_BASE, input=t2.input, conversation_id=conv_id
    )
    assert json_output_text(b2) == t2.output

    b3 = await post_response(
        client, **_CONV_BASE, input=t3.input, conversation_id=conv_id
    )
    assert json_output_text(b3) == t3.output

    # Branch off turn 1 via previous_response_id — only turn 1 context visible
    b4 = await post_response(
        client, **_CONV_BASE, input=t4.input, previous_response_id=r1_id
    )
    assert b4["status"] == "completed"
    assert json_output_text(b4) == t4.output


# ── test 10: 5-turn conv with 2 inline branches ───────────────────────────────
# Cassette: conv-branch-turn-number-gpt-4o-nonstreaming.yaml
# Turn 1: "What is 2+2?" → 4
# Turn 2 (from turn 1): "Add 2" → 6
# Turn 3 (branch1 from turn 1): "Add 1" → 5
# Turn 4 (branch1 continues from turn 3): "Add 3" → 8
# Turn 5 (branch2 from turn 2): "Add 4" → 10


@pytest.mark.anyio
async def test_multi_branch(
    conversation_gateway_client: tuple[httpx.AsyncClient, Callable[[str], None]],
) -> None:
    client, use_cassette = conversation_gateway_client
    use_cassette("conv-multi-branch-multi-turn-gpt-4o-nonstreaming.yaml")
    t1, t2, t3, t4, t5 = _turns("conv-multi-branch-multi-turn-gpt-4o-nonstreaming.yaml")

    b1 = await post_response(client, **_CONV_BASE, input=t1.input)
    assert b1["status"] == "completed"
    assert json_output_text(b1) == t1.output
    conv_id, r1_id = b1["conversation_id"], b1["id"]

    b2 = await post_response(
        client, **_CONV_BASE, input=t2.input, conversation_id=conv_id
    )
    assert json_output_text(b2) == t2.output
    r2_id = b2["id"]

    # Branch1 — from turn 1
    b3 = await post_response(
        client, **_CONV_BASE, input=t3.input, previous_response_id=r1_id
    )
    assert b3["status"] == "completed"
    assert json_output_text(b3) == t3.output

    b4 = await post_response(
        client, **_CONV_BASE, input=t4.input, previous_response_id=b3["id"]
    )
    assert b4["status"] == "completed"
    assert json_output_text(b4) == t4.output

    # Branch2 — from turn 2
    b5 = await post_response(
        client, **_CONV_BASE, input=t5.input, previous_response_id=r2_id
    )
    assert b5["status"] == "completed"
    assert json_output_text(b5) == t5.output
