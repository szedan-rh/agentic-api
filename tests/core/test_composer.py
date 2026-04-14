from __future__ import annotations

from agentic_api.core.composer import ResponseComposer
from agentic_api.core.models import (
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
    ResponseEvent,
    ResponsesRequest,
    ResponsesResponse,
    TextDeltaEvent,
)


def _make_composer() -> ResponseComposer:
    request = ResponsesRequest(model="test-model", input="hello")
    response = ResponsesResponse.create_from_response_request(request)
    return ResponseComposer(response=response)


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


def test_start_is_idempotent() -> None:
    c = _make_composer()
    c.start()
    events = list(c.start())
    assert events == []


def test_feed_before_start_raises() -> None:
    c = _make_composer()
    import pytest

    with pytest.raises(RuntimeError):
        list(c.feed(MessageStarted(item_key="k")))


def test_message_started_emits_output_item_added_and_content_part_added() -> None:
    c = _make_composer()
    c.start()
    events = list(c.feed(MessageStarted(item_key="k0")))
    types = [e.type for e in events]
    assert types == ["response.output_item.added", "response.content_part.added"]
    assert isinstance(events[0], OutputItemEvent)
    assert isinstance(events[1], ContentPartEvent)


def test_message_delta_emits_text_delta() -> None:
    c = _make_composer()
    c.start()
    list(c.feed(MessageStarted(item_key="k0")))
    events = list(c.feed(MessageDelta(item_key="k0", delta="hello")))
    assert len(events) == 1
    assert isinstance(events[0], TextDeltaEvent)
    assert events[0].type == "response.output_text.delta"
    assert events[0].delta == "hello"


def test_message_done_emits_text_done_content_part_done_output_item_done() -> None:
    c = _make_composer()
    c.start()
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


def test_function_call_started_emits_output_item_added() -> None:
    c = _make_composer()
    c.start()
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


def test_function_call_arguments_delta_emits_delta_event() -> None:
    c = _make_composer()
    c.start()
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
    events = list(c.feed(FunctionCallArgumentsDelta(item_key="fc0", delta='{"x":')))
    assert len(events) == 1
    assert events[0].type == "response.function_call_arguments.delta"
    assert events[0].delta == '{"x":'


def test_function_call_done_emits_arguments_done_and_output_item_done() -> None:
    c = _make_composer()
    c.start()
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


def test_reasoning_events_emit_nothing() -> None:
    c = _make_composer()
    c.start()
    assert list(c.feed(ReasoningStarted(item_key="r0"))) == []
    assert list(c.feed(ReasoningDelta(item_key="r0", delta="thinking..."))) == []
    assert list(c.feed(ReasoningDone(item_key="r0", text="thought"))) == []


def test_usage_final_emits_response_completed() -> None:
    c = _make_composer()
    c.start()
    list(c.feed(MessageStarted(item_key="k0")))
    list(c.feed(MessageDone(item_key="k0", text="hi")))
    events = list(c.feed(_usage()))
    assert len(events) == 1
    assert events[0].type == "response.completed"
    assert isinstance(events[0], ResponseEvent)
    assert c.response.status == "completed"
    assert c.response.usage is not None
    assert c.response.usage.input_tokens == 10
    assert c.response.usage.output_tokens == 5


def test_usage_final_with_incomplete_reason_emits_response_incomplete() -> None:
    c = _make_composer()
    c.start()
    events = list(c.feed(_usage(incomplete_reason="max_output_tokens")))
    assert len(events) == 1
    assert events[0].type == "response.incomplete"
    assert c.response.status == "incomplete"
    assert c.response.incomplete_details.reason == "max_output_tokens"


def test_completed_response_output_contains_all_items() -> None:
    c = _make_composer()
    c.start()
    list(c.feed(MessageStarted(item_key="k0")))
    list(c.feed(MessageDone(item_key="k0", text="first")))
    list(c.feed(MessageStarted(item_key="k1")))
    list(c.feed(MessageDone(item_key="k1", text="second")))
    list(c.feed(_usage()))
    assert len(c.response.output) == 2


def test_make_error_events_emits_error_and_failed() -> None:
    c = _make_composer()
    c.start()
    events = list(
        c.make_error_events(code="upstream_error", message="something went wrong")
    )
    types = [e.type for e in events]
    assert types == ["response.error", "response.failed"]
    assert isinstance(events[0], ErrorEvent)
    assert events[0].code == "upstream_error"
    assert events[0].message == "something went wrong"
    assert c.response.status == "failed"
    assert c.response.error == {
        "code": "upstream_error",
        "message": "something went wrong",
    }


def test_make_error_events_with_param() -> None:
    c = _make_composer()
    c.start()
    events = list(
        c.make_error_events(code="bad_input", message="invalid field", param="model")
    )
    assert isinstance(events[0], ErrorEvent)
    assert events[0].param == "model"


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
