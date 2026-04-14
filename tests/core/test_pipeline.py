from __future__ import annotations

from typing import Any
from unittest.mock import patch

import pytest
from pydantic_ai import (
    AgentRunResultEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    ToolCallPart,
)

from agentic_api.core.pipeline import Pipeline
from agentic_api.types.responses import (
    AgentRunSettings,
    FunctionToolCall,
    OutputMessage,
    ResponsesRequest,
    ResponsesResponse,
)
from agentic_api.utils.failures import FailureCounters
from tests.utils import make_agent, make_agent_run_result


def _make_pipeline() -> Pipeline:
    request = ResponsesRequest(model="test-model", input="hello")
    response = ResponsesResponse.create_from_response_request(request)
    return Pipeline.build(response=response)


def _make_run_settings() -> AgentRunSettings:
    return AgentRunSettings(
        message_history=[],
        instructions=None,
        toolsets=[],
        usage_limits=None,
    )


def _stub_stream(*parts: tuple[str, str]):
    """Build a run_stream_events stub that yields text parts.

    Each entry in parts is (item_key_unused, text).
    """

    async def _stub(*args: Any, **kwargs: Any):
        for _, text in parts:
            part = TextPart(content=text)
            yield PartStartEvent(index=0, part=part)
            yield PartEndEvent(index=0, part=part)
        yield AgentRunResultEvent(result=make_agent_run_result())

    return _stub


async def _collect(pipeline: Pipeline, agent) -> list[Any]:
    run_settings = _make_run_settings()
    counters = FailureCounters()
    events = []
    list(pipeline.composer.start())
    async with pipeline.run_agent(agent, run_settings, counters) as (stream, _messages):
        async for event in stream:
            events.append(event)
    return events


@pytest.mark.anyio
async def test_pipeline_emits_completed_event_for_text_output() -> None:
    pipeline = _make_pipeline()
    agent = make_agent()

    with patch(
        "pydantic_ai.Agent.run_stream_events", new=_stub_stream(("k0", "hello world"))
    ):
        events = await _collect(pipeline, agent)

    types = [e.type for e in events]
    assert "response.output_item.added" in types
    assert "response.output_text.delta" in types
    assert "response.output_item.done" in types
    assert "response.completed" in types


@pytest.mark.anyio
async def test_pipeline_response_status_completed_after_run() -> None:
    pipeline = _make_pipeline()
    agent = make_agent()

    with patch("pydantic_ai.Agent.run_stream_events", new=_stub_stream(("k0", "hi"))):
        await _collect(pipeline, agent)

    assert pipeline.composer.response.status == "completed"


@pytest.mark.anyio
async def test_pipeline_output_contains_text_content() -> None:
    pipeline = _make_pipeline()
    agent = make_agent()

    with patch(
        "pydantic_ai.Agent.run_stream_events", new=_stub_stream(("k0", "hello world"))
    ):
        await _collect(pipeline, agent)

    output = pipeline.composer.response.output
    assert len(output) == 1
    assert isinstance(output[0], OutputMessage)
    assert output[0].content[0].text == "hello world"


@pytest.mark.anyio
async def test_pipeline_emits_function_call_events() -> None:
    pipeline = _make_pipeline()
    agent = make_agent()

    async def _tool_stub(*args: Any, **kwargs: Any):
        part = ToolCallPart(
            tool_name="search", args='{"q":"test"}', tool_call_id="call_1"
        )
        yield PartStartEvent(index=0, part=part)
        yield PartEndEvent(index=0, part=part)
        yield AgentRunResultEvent(result=make_agent_run_result())

    with patch("pydantic_ai.Agent.run_stream_events", new=_tool_stub):
        events = await _collect(pipeline, agent)

    types = [e.type for e in events]
    assert "response.output_item.added" in types
    assert "response.function_call_arguments.done" in types
    assert "response.output_item.done" in types

    output = pipeline.composer.response.output
    assert len(output) == 1
    assert isinstance(output[0], FunctionToolCall)
    assert output[0].name == "search"
    assert output[0].arguments == '{"q":"test"}'


@pytest.mark.anyio
async def test_pipeline_handles_empty_stream() -> None:
    pipeline = _make_pipeline()
    agent = make_agent()

    async def _empty_stub(*args: Any, **kwargs: Any):
        yield AgentRunResultEvent(result=make_agent_run_result())

    with patch("pydantic_ai.Agent.run_stream_events", new=_empty_stub):
        events = await _collect(pipeline, agent)

    types = [e.type for e in events]
    assert "response.completed" in types
    assert pipeline.composer.response.output == []


@pytest.mark.anyio
async def test_pipeline_sequence_numbers_monotonically_increasing() -> None:
    pipeline = _make_pipeline()
    agent = make_agent()

    all_events = list(pipeline.composer.start())
    with patch("pydantic_ai.Agent.run_stream_events", new=_stub_stream(("k0", "hi"))):
        async with pipeline.run_agent(
            agent, _make_run_settings(), FailureCounters()
        ) as (stream, _):
            async for event in stream:
                all_events.append(event)

    seq = [e.sequence_number for e in all_events]
    assert seq == sorted(seq)
    assert len(seq) == len(set(seq))
