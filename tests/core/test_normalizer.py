from __future__ import annotations

from pydantic_ai import (
    AgentRunResultEvent,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

from agentic_api.core.normalized_events import (
    FunctionCallArgumentsDelta,
    FunctionCallDone,
    FunctionCallStarted,
    MessageDelta,
    MessageDone,
    MessageStarted,
    ReasoningDelta,
    ReasoningDone,
    ReasoningStarted,
    UsageFinal,
)
from agentic_api.core.normalizer import PydanticAINormalizer

from tests.utils import make_agent_run_result


def normalizer() -> PydanticAINormalizer:
    return PydanticAINormalizer()


def test_text_part_start_emits_message_started() -> None:
    n = normalizer()
    events = list(n.on_event(PartStartEvent(index=0, part=TextPart(content=""))))
    assert len(events) == 1
    assert isinstance(events[0], MessageStarted)
    assert events[0].item_key == "part:0"


def test_text_part_start_with_content_emits_started_and_delta() -> None:
    n = normalizer()
    events = list(n.on_event(PartStartEvent(index=0, part=TextPart(content="hello"))))
    assert len(events) == 2
    assert isinstance(events[0], MessageStarted)
    assert isinstance(events[1], MessageDelta)
    assert events[1].delta == "hello"


def test_text_part_delta_emits_message_delta() -> None:
    n = normalizer()
    n.on_event(PartStartEvent(index=0, part=TextPart(content="")))
    events = list(
        n.on_event(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="world")))
    )
    assert len(events) == 1
    assert isinstance(events[0], MessageDelta)
    assert events[0].delta == "world"


def test_text_part_delta_empty_content_emits_nothing() -> None:
    n = normalizer()
    n.on_event(PartStartEvent(index=0, part=TextPart(content="")))
    events = list(
        n.on_event(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="")))
    )
    assert events == []


def test_text_part_end_emits_message_done() -> None:
    n = normalizer()
    part = TextPart(content="final text")
    n.on_event(PartStartEvent(index=0, part=part))
    events = list(n.on_event(PartEndEvent(index=0, part=part)))
    assert len(events) == 1
    assert isinstance(events[0], MessageDone)
    assert events[0].text == "final text"
    assert events[0].item_key == "part:0"


def test_thinking_part_start_emits_reasoning_started() -> None:
    n = normalizer()
    events = list(n.on_event(PartStartEvent(index=0, part=ThinkingPart(content=""))))
    assert len(events) == 1
    assert isinstance(events[0], ReasoningStarted)


def test_thinking_part_start_with_content_emits_started_and_delta() -> None:
    n = normalizer()
    events = list(
        n.on_event(PartStartEvent(index=0, part=ThinkingPart(content="let me think")))
    )
    assert len(events) == 2
    assert isinstance(events[0], ReasoningStarted)
    assert isinstance(events[1], ReasoningDelta)
    assert events[1].delta == "let me think"


def test_thinking_part_delta_emits_reasoning_delta() -> None:
    n = normalizer()
    n.on_event(PartStartEvent(index=0, part=ThinkingPart(content="")))
    events = list(
        n.on_event(
            PartDeltaEvent(
                index=0, delta=ThinkingPartDelta(content_delta="more thought")
            )
        )
    )
    assert len(events) == 1
    assert isinstance(events[0], ReasoningDelta)
    assert events[0].delta == "more thought"


def test_thinking_part_end_emits_reasoning_done() -> None:
    n = normalizer()
    part = ThinkingPart(content="conclusion")
    n.on_event(PartStartEvent(index=0, part=part))
    events = list(n.on_event(PartEndEvent(index=0, part=part)))
    assert len(events) == 1
    assert isinstance(events[0], ReasoningDone)
    assert events[0].text == "conclusion"


def test_tool_call_part_start_emits_function_call_started() -> None:
    n = normalizer()
    part = ToolCallPart(tool_name="my_tool", args=None, tool_call_id="call_abc")
    events = list(n.on_event(PartStartEvent(index=0, part=part)))
    assert len(events) == 1
    assert isinstance(events[0], FunctionCallStarted)
    assert events[0].name == "my_tool"
    assert events[0].call_id == "call_abc"


def test_tool_call_part_start_with_args_emits_started_and_delta() -> None:
    n = normalizer()
    part = ToolCallPart(tool_name="my_tool", args='{"x":1}', tool_call_id="call_abc")
    events = list(n.on_event(PartStartEvent(index=0, part=part)))
    assert len(events) == 2
    assert isinstance(events[0], FunctionCallStarted)
    assert isinstance(events[1], FunctionCallArgumentsDelta)
    assert events[1].delta == '{"x":1}'


def test_tool_call_part_delta_emits_arguments_delta() -> None:
    n = normalizer()
    part = ToolCallPart(tool_name="my_tool", args=None, tool_call_id="call_abc")
    n.on_event(PartStartEvent(index=0, part=part))
    events = list(
        n.on_event(
            PartDeltaEvent(index=0, delta=ToolCallPartDelta(args_delta='{"y":2}'))
        )
    )
    assert len(events) == 1
    assert isinstance(events[0], FunctionCallArgumentsDelta)
    assert events[0].delta == '{"y":2}'


def test_tool_call_part_end_emits_function_call_done() -> None:
    n = normalizer()
    part = ToolCallPart(tool_name="my_tool", args='{"z":3}', tool_call_id="call_abc")
    n.on_event(PartStartEvent(index=0, part=part))
    events = list(n.on_event(PartEndEvent(index=0, part=part)))
    assert len(events) == 1
    assert isinstance(events[0], FunctionCallDone)
    assert events[0].arguments_json == '{"z":3}'


def test_agent_run_result_event_emits_usage_final() -> None:
    n = normalizer()
    events = list(
        n.on_event(
            AgentRunResultEvent(
                result=make_agent_run_result(input_tokens=20, output_tokens=8)
            )
        )
    )
    assert len(events) == 1
    assert isinstance(events[0], UsageFinal)
    assert events[0].input_tokens == 20
    assert events[0].output_tokens == 8
    assert events[0].incomplete_reason is None


def test_unknown_event_emits_nothing() -> None:
    n = normalizer()
    assert list(n.on_event("not_an_event")) == []


def test_delta_for_unknown_index_emits_nothing() -> None:
    n = normalizer()
    # No PartStartEvent for index 99 — delta should be silently dropped.
    events = list(
        n.on_event(PartDeltaEvent(index=99, delta=TextPartDelta(content_delta="x")))
    )
    assert events == []


def test_multiple_parts_get_distinct_item_keys() -> None:
    n = normalizer()
    n.on_event(PartStartEvent(index=0, part=TextPart(content="")))
    n.on_event(PartStartEvent(index=1, part=ThinkingPart(content="")))

    delta0 = list(
        n.on_event(PartDeltaEvent(index=0, delta=TextPartDelta(content_delta="a")))
    )
    delta1 = list(
        n.on_event(PartDeltaEvent(index=1, delta=ThinkingPartDelta(content_delta="b")))
    )

    assert delta0[0].item_key == "part:0"
    assert delta1[0].item_key == "part:1"
    assert delta0[0].item_key != delta1[0].item_key
