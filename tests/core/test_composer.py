from __future__ import annotations

import pytest

from agentic_api.core.composer import ResponseComposer
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
from agentic_api.types.responses import (
    ContentPartEvent,
    ErrorEvent,
    FunctionToolCall,
    OutputItemEvent,
    OutputMessage,
    ResponsesRequest,
    ResponsesResponse,
)


def _make_composer() -> ResponseComposer:
    request = ResponsesRequest(model="test-model", input="hello")
    response = ResponsesResponse.create_from_response_request(request)
    return ResponseComposer(response=response)


def _started_composer() -> ResponseComposer:
    c = _make_composer()
    c.start()
    return c


def _usage(
    input_tokens: int = 10,
    output_tokens: int = 5,
    incomplete_reason: str | None = None,
) -> UsageFinal:
    return UsageFinal(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        total_tokens=input_tokens + output_tokens,
        cache_read_tokens=0,
        cache_write_tokens=0,
        reasoning_tokens=0,
        incomplete_reason=incomplete_reason,
    )


def test_start_emits_created_and_in_progress() -> None:
    c = _make_composer()
    events = list(c.start())
    assert len(events) == 2
    assert events[0].type == "response.created"
    assert events[1].type == "response.in_progress"
    assert c.response.status == "in_progress"


def test_feed_before_start_raises() -> None:
    c = _make_composer()
    with pytest.raises(RuntimeError):
        list(c.feed(MessageStarted(item_key="k")))


def test_message_started_emits_output_item_added_and_content_part_added() -> None:
    c = _started_composer()
    events = list(c.feed(MessageStarted(item_key="k0")))
    types = [e.type for e in events]
    assert types == ["response.output_item.added", "response.content_part.added"]
    assert isinstance(events[0], OutputItemEvent)
    assert isinstance(events[1], ContentPartEvent)


def test_message_done_emits_text_done_content_part_done_output_item_done() -> None:
    c = _started_composer()
    list(c.feed(MessageStarted(item_key="k0")))
    list(c.feed(MessageDelta(item_key="k0", delta="hello")))
    events = list(c.feed(MessageDone(item_key="k0", text="hello")))
    types = [e.type for e in events]
    assert types == [
        "response.output_text.done",
        "response.content_part.done",
        "response.output_item.done",
    ]
    output_item_done = events[2]
    assert isinstance(output_item_done, OutputItemEvent)
    assert isinstance(output_item_done.item, OutputMessage)
    assert output_item_done.item.content[0].text == "hello"
    assert output_item_done.item.status == "completed"


def test_message_output_index_and_item_id_stable() -> None:
    c = _started_composer()
    out = []
    out.extend(c.feed(MessageStarted(item_key="m1")))
    out.extend(c.feed(MessageDelta(item_key="m1", delta="Hello")))
    out.extend(c.feed(MessageDelta(item_key="m1", delta=", world")))
    out.extend(c.feed(MessageDone(item_key="m1", text="Hello, world")))
    out.extend(c.feed(_usage()))

    added = [
        e
        for e in out
        if e.type == "response.output_item.added" and e.item.type == "message"
    ]
    assert len(added) == 1
    out_index = added[0].output_index
    item_id = added[0].item.id

    deltas = [e for e in out if e.type == "response.output_text.delta"]
    assert deltas
    assert {d.output_index for d in deltas} == {out_index}
    assert {d.item_id for d in deltas} == {item_id}

    done = [e for e in out if e.type == "response.output_text.done"]
    assert len(done) == 1
    assert done[0].output_index == out_index
    assert done[0].item_id == item_id


def test_function_call_started_emits_output_item_added() -> None:
    c = _started_composer()
    events = list(
        c.feed(
            FunctionCallStarted(
                item_key="fc0",
                call_id="call_1",
                name="my_tool",
                initial_arguments_json="",
            )
        )
    )
    assert len(events) == 1
    assert events[0].type == "response.output_item.added"
    assert isinstance(events[0].item, FunctionToolCall)
    assert events[0].item.name == "my_tool"


def test_function_call_done_emits_arguments_done_and_output_item_done() -> None:
    c = _started_composer()
    list(
        c.feed(
            FunctionCallStarted(
                item_key="fc0",
                call_id="call_1",
                name="my_tool",
                initial_arguments_json="",
            )
        )
    )
    events = list(c.feed(FunctionCallDone(item_key="fc0", arguments_json='{"x":1}')))
    types = [e.type for e in events]
    assert types == [
        "response.function_call_arguments.done",
        "response.output_item.done",
    ]
    output_item_done = events[1]
    assert isinstance(output_item_done.item, FunctionToolCall)
    assert output_item_done.item.arguments == '{"x":1}'
    assert output_item_done.item.status == "completed"


def test_function_call_arguments_deltas_attributed_to_function_item() -> None:
    c = _started_composer()
    out = []
    out.extend(
        c.feed(
            FunctionCallStarted(
                item_key="fc1",
                call_id="call_123",
                name="get_weather",
                initial_arguments_json="",
            )
        )
    )
    out.extend(c.feed(MessageStarted(item_key="m1")))
    out.extend(c.feed(MessageDelta(item_key="m1", delta="(ignored)")))
    out.extend(c.feed(FunctionCallArgumentsDelta(item_key="fc1", delta="{")))
    out.extend(c.feed(FunctionCallArgumentsDelta(item_key="fc1", delta='"x":1')))
    out.extend(c.feed(FunctionCallArgumentsDelta(item_key="fc1", delta="}")))
    out.extend(c.feed(FunctionCallDone(item_key="fc1", arguments_json='{"x":1}')))
    out.extend(c.feed(_usage()))

    fc_added = [
        e
        for e in out
        if e.type == "response.output_item.added" and e.item.type == "function_call"
    ]
    assert len(fc_added) == 1
    fc_item_id = fc_added[0].item.id
    fc_out_index = fc_added[0].output_index

    arg_deltas = [e for e in out if e.type == "response.function_call_arguments.delta"]
    assert arg_deltas
    assert {d.item_id for d in arg_deltas} == {fc_item_id}
    assert {d.output_index for d in arg_deltas} == {fc_out_index}


def test_reasoning_events_emit_nothing() -> None:
    c = _started_composer()
    assert list(c.feed(ReasoningStarted(item_key="r0"))) == []
    assert list(c.feed(ReasoningDelta(item_key="r0", delta="thinking..."))) == []
    assert list(c.feed(ReasoningDone(item_key="r0", text="thought"))) == []


def test_usage_final_emits_response_completed() -> None:
    c = _started_composer()
    list(c.feed(MessageStarted(item_key="k0")))
    list(c.feed(MessageDone(item_key="k0", text="hi")))
    events = list(c.feed(_usage()))
    assert len(events) == 1
    assert events[0].type == "response.completed"
    assert c.response.status == "completed"
    assert c.response.usage is not None
    assert c.response.usage.input_tokens == 10
    assert c.response.usage.output_tokens == 5


def test_completed_response_omits_reasoning_item_when_no_thinking_part() -> None:
    c = _started_composer()
    out = []
    out.extend(c.feed(MessageStarted(item_key="m1")))
    out.extend(c.feed(MessageDelta(item_key="m1", delta="Hello")))
    out.extend(c.feed(MessageDone(item_key="m1", text="Hello")))
    out.extend(c.feed(_usage()))

    completed = [e for e in out if e.type == "response.completed"]
    assert len(completed) == 1
    assert [o.type for o in completed[0].response.output] == ["message"]


def test_incomplete_response_sets_incomplete_details_for_max_output_tokens() -> None:
    c = _started_composer()
    out = []
    out.extend(c.feed(MessageStarted(item_key="m1")))
    out.extend(c.feed(MessageDone(item_key="m1", text="Partial")))
    out.extend(c.feed(_usage(incomplete_reason="max_output_tokens")))

    assert not [e for e in out if e.type == "response.completed"]
    incomplete = [e for e in out if e.type == "response.incomplete"]
    assert len(incomplete) == 1
    resp = incomplete[0].response
    assert resp.status == "incomplete"
    assert resp.incomplete_details is not None
    assert resp.incomplete_details.reason == "max_output_tokens"


def test_make_error_events_emits_error_and_failed() -> None:
    c = _started_composer()
    events = list(
        c.make_error_events(
            code="upstream_error", message="something went wrong", param="model"
        )
    )
    types = [e.type for e in events]
    assert types == ["response.error", "response.failed"]
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "upstream_error"
    assert events[0].message == "something went wrong"
    assert events[0].param == "model"
    assert c.response.status == "failed"


def test_sequence_numbers_are_monotonically_increasing() -> None:
    c = _make_composer()
    all_events = []
    all_events.extend(c.start())
    all_events.extend(c.feed(MessageStarted(item_key="k0")))
    all_events.extend(c.feed(MessageDelta(item_key="k0", delta="hi")))
    all_events.extend(c.feed(MessageDone(item_key="k0", text="hi")))
    all_events.extend(c.feed(_usage()))

    seq_numbers = [e.sequence_number for e in all_events]
    assert seq_numbers == sorted(seq_numbers)
    assert len(seq_numbers) == len(set(seq_numbers))
