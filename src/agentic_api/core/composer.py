from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from time import time

from agentic_api.core.models import (
    FunctionCallArgumentsDelta,
    FunctionCallDone,
    FunctionCallStarted,
    MessageDelta,
    MessageDone,
    MessageStarted,
    NormalizedEvent,
    ReasoningDelta,
    ReasoningDone,
    ReasoningStarted,
    UsageFinal,
)
from agentic_api.types.responses import (
    ContentPartEvent,
    ErrorEvent,
    FunctionToolCall,
    IncompleteDetails,
    InputTokenDetails,
    OutputItemEvent,
    OutputMessage,
    OutputTextContent,
    OutputTokenDetails,
    ResponseEvent,
    ResponseUsage,
    ResponsesResponse,
    StreamEvent,
    TextDeltaEvent,
)
from agentic_api.utils.common import uuid7_str


@dataclass(slots=True)
class _ItemState:
    item_id: str
    output_index: int
    kind: str
    text: str = ""
    reasoning: str = ""
    function_name: str | None = None
    function_call_id: str | None = None
    function_args_json: str = ""


class ResponseComposer:
    """Compose a Responses SSE/JSON contract from NormalizedEvents.

    Handles: text messages, reasoning, and user-defined function tool calls.
    """

    def __init__(
        self, *, response: ResponsesResponse, include: set[str] | None = None
    ) -> None:
        self._response = response
        self._started = False
        self._sequence_number = 0
        self._next_output_index = 0
        self._items: dict[str, _ItemState] = {}
        self._output_items: list[OutputMessage | FunctionToolCall] = []

    @property
    def response(self) -> ResponsesResponse:
        return self._response

    def feed(self, event: NormalizedEvent) -> Iterable[StreamEvent]:
        if not self._started:
            raise RuntimeError("ResponseComposer.start() must be called before feed().")

        if isinstance(event, MessageStarted):
            yield from self._start_message(event)
        elif isinstance(event, MessageDelta):
            yield from self._message_delta(event)
        elif isinstance(event, MessageDone):
            yield from self._message_done(event)
        elif isinstance(event, ReasoningStarted):
            pass  # Reasoning items are not emitted as output events in this implementation
        elif isinstance(event, ReasoningDelta):
            pass
        elif isinstance(event, ReasoningDone):
            pass
        elif isinstance(event, FunctionCallStarted):
            yield from self._start_function_call(event)
        elif isinstance(event, FunctionCallArgumentsDelta):
            yield from self._function_args_delta(event)
        elif isinstance(event, FunctionCallDone):
            yield from self._function_done(event)
        elif isinstance(event, UsageFinal):
            yield from self._complete_response(event)

    def start(self) -> Iterable[StreamEvent]:
        if self._started:
            return []
        self._started = True
        self._response.status = "in_progress"
        return [
            ResponseEvent(
                type="response.created",
                sequence_number=self.alloc_sequence_number(),
                response=self._response,
            ),
            ResponseEvent(
                type="response.in_progress",
                sequence_number=self.alloc_sequence_number(),
                response=self._response,
            ),
        ]

    def alloc_sequence_number(self) -> int:
        return self._incr_seq()

    def _incr_seq(self) -> int:
        current = self._sequence_number
        self._sequence_number += 1
        return current

    def _alloc_output_index(self) -> int:
        current = self._next_output_index
        self._next_output_index += 1
        return current

    def _start_message(self, event: MessageStarted) -> Iterable[StreamEvent]:
        item_id = uuid7_str("msg_")
        out_index = self._alloc_output_index()
        state = _ItemState(item_id=item_id, output_index=out_index, kind="message")
        self._items[event.item_key] = state

        yield OutputItemEvent(
            type="response.output_item.added",
            sequence_number=self._incr_seq(),
            output_index=out_index,
            item=OutputMessage(content=[], id=item_id, status="in_progress"),
        )
        yield ContentPartEvent(
            type="response.content_part.added",
            item_id=item_id,
            sequence_number=self._incr_seq(),
            output_index=out_index,
            content_index=0,
            part=OutputTextContent(text=""),
        )

    def _message_delta(self, event: MessageDelta) -> Iterable[StreamEvent]:
        state = self._items[event.item_key]
        state.text += event.delta
        yield TextDeltaEvent(
            type="response.output_text.delta",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            content_index=0,
            delta=event.delta,
            logprobs=[],
        )

    def _message_done(self, event: MessageDone) -> Iterable[StreamEvent]:
        state = self._items[event.item_key]
        state.text = event.text
        yield TextDeltaEvent(
            type="response.output_text.done",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            content_index=0,
            text=event.text,
            logprobs=[],
        )
        yield ContentPartEvent(
            type="response.content_part.done",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            content_index=0,
            part=OutputTextContent(text=event.text),
        )
        item = OutputMessage(
            content=[OutputTextContent(text=event.text)],
            id=state.item_id,
            status="completed",
        )
        yield OutputItemEvent(
            type="response.output_item.done",
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)

    def _start_function_call(self, event: FunctionCallStarted) -> Iterable[StreamEvent]:
        item_id = uuid7_str("fc_")
        out_index = self._alloc_output_index()
        state = _ItemState(
            item_id=item_id,
            output_index=out_index,
            kind="function_call",
            function_name=event.name,
            function_call_id=event.call_id,
            function_args_json=event.initial_arguments_json,
        )
        self._items[event.item_key] = state

        yield OutputItemEvent(
            type="response.output_item.added",
            sequence_number=self._incr_seq(),
            output_index=out_index,
            item=FunctionToolCall(
                arguments=event.initial_arguments_json,
                call_id=event.call_id,
                name=event.name,
                id=item_id,
                status="in_progress",
            ),
        )

    def _function_args_delta(
        self, event: FunctionCallArgumentsDelta
    ) -> Iterable[StreamEvent]:
        state = self._items[event.item_key]
        state.function_args_json += event.delta
        yield TextDeltaEvent(
            type="response.function_call_arguments.delta",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            delta=event.delta,
        )

    def _function_done(self, event: FunctionCallDone) -> Iterable[StreamEvent]:
        state = self._items[event.item_key]
        state.function_args_json = event.arguments_json
        yield TextDeltaEvent(
            type="response.function_call_arguments.done",
            item_id=state.item_id,
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            arguments=event.arguments_json,
        )
        item = FunctionToolCall(
            arguments=event.arguments_json,
            call_id=state.function_call_id or "",
            name=state.function_name or "",
            id=state.item_id,
            status="completed",
        )
        yield OutputItemEvent(
            type="response.output_item.done",
            sequence_number=self._incr_seq(),
            output_index=state.output_index,
            item=item,
        )
        self._output_items.append(item)

    def _complete_response(self, event: UsageFinal) -> Iterable[StreamEvent]:
        self._response.usage = ResponseUsage(
            input_tokens=event.input_tokens,
            input_tokens_details=InputTokenDetails(
                cached_tokens=event.cache_read_tokens
            ),
            output_tokens=event.output_tokens,
            output_tokens_details=OutputTokenDetails(
                reasoning_tokens=event.reasoning_tokens
            ),
            total_tokens=event.total_tokens,
        )
        self._response.output = list(self._output_items)
        if event.incomplete_reason is not None:
            self._response.status = "incomplete"
            self._response.incomplete_details = IncompleteDetails(
                reason=event.incomplete_reason
            )
            self._response.completed_at = None  # type: ignore[assignment]
            yield ResponseEvent(
                type="response.incomplete",
                sequence_number=self._incr_seq(),
                response=self._response,
            )
            return

        self._response.status = "completed"
        self._response.incomplete_details = None
        self._response.created_at = int(time())
        yield ResponseEvent(
            type="response.completed",
            sequence_number=self._incr_seq(),
            response=self._response,
        )

    def make_error_events(
        self, *, code: str, message: str, param: str | None = None
    ) -> Iterable[StreamEvent]:
        """Emit an error event + response.failed lifecycle event."""
        self._response.error = {"code": code, "message": message}
        self._response.status = "failed"
        yield ErrorEvent(
            type="response.error",
            sequence_number=self.alloc_sequence_number(),
            code=code,
            message=message,
            param=param,
        )
        yield ResponseEvent(
            type="response.failed",
            sequence_number=self.alloc_sequence_number(),
            response=self._response,
        )
