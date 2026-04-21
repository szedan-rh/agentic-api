"""Shared helpers for cassette-based end-to-end tests."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import yaml


class Turn:
    """One /v1/responses turn loaded from a cassette."""

    def __init__(self, input: str, output: str) -> None:
        self.input = input
        self.output = output


def load_turns(cassette_path: Path) -> list[Turn]:
    """Return input/output pairs for every /v1/responses turn in a cassette, in order.

    Turns with path != /v1/responses are skipped (e.g. /v1/conversations).
    For streaming turns the output is reconstructed from response.output_text.delta events.
    """
    data = yaml.safe_load(cassette_path.read_text(encoding="utf-8"))
    turns: list[Turn] = []
    for turn in data.get("turns", []):
        req = turn.get("request", {})
        if req.get("path") != "/v1/responses":
            continue
        inp = req.get("body", {}).get("input", "")
        resp = turn.get("response", {})
        if sse := resp.get("sse"):
            output = ""
            for line in sse:
                line = line.strip()
                if line.startswith("data:") and line[5:].strip() not in ("", "[DONE]"):
                    try:
                        evt = json.loads(line[5:].strip())
                        if evt.get("type") == "response.output_text.delta":
                            output += evt.get("delta", "")
                    except json.JSONDecodeError:
                        pass
        else:
            body = resp.get("body") or {}
            try:
                output = body["output"][0]["content"][0]["text"]
            except (KeyError, IndexError, TypeError):
                output = ""
        turns.append(Turn(input=inp, output=output))
    return turns


def parse_sse_events(text: str) -> list[dict]:
    events = []
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("data:") and line[5:].strip() != "[DONE]":
            try:
                events.append(json.loads(line[5:]))
            except json.JSONDecodeError:
                pass
    return events


def sse_completed_response(text: str) -> dict | None:
    for evt in parse_sse_events(text):
        if evt.get("type") == "response.completed":
            return evt.get("response")
    return None


def sse_output_text(text: str) -> str:
    return "".join(
        evt.get("delta", "")
        for evt in parse_sse_events(text)
        if evt.get("type") == "response.output_text.delta"
    )


def json_output_text(body: dict) -> str:
    return " ".join(
        c.get("text", "")
        for item in body.get("output", [])
        for c in item.get("content", [])
        if c.get("type") == "output_text"
    )


async def post_response(client: httpx.AsyncClient, **fields: Any) -> dict:
    """POST /v1/responses, assert 200, return parsed body."""
    r = await client.post("/v1/responses", json=fields)
    assert r.status_code == 200, r.text
    return r.json()


async def stream_response(client: httpx.AsyncClient, **fields: Any) -> str:
    """POST /v1/responses with stream=True, assert 200, return full SSE text."""
    async with client.stream("POST", "/v1/responses", json=fields) as resp:
        text = "".join([c async for c in resp.aiter_text()])
    assert resp.status_code == 200, text
    return text
