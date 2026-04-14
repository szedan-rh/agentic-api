from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

from pydantic_ai import (
    AgentRunResultEvent,
    FunctionToolCallEvent,
    FunctionToolResultEvent,
    ModelMessage,
    ModelResponse,
    PartDeltaEvent,
    PartEndEvent,
    PartStartEvent,
    RunUsage,
    TextPart,
    TextPartDelta,
    ThinkingPart,
    ThinkingPartDelta,
    ToolCallPart,
    ToolCallPartDelta,
)

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


class PydanticAINormalizer:
    """Convert pydantic_ai stream events into internal NormalizedEvents.

    Handles: text messages, reasoning (thinking), and user-defined function tool calls.
    Does not handle MCP, code interpreter, or web search.
    """

    def __init__(self) -> None:
        self._index_to_item_key: dict[int, str] = {}
        self._tool_call_id_to_item_key: dict[str, str] = {}
        self._item_kind: dict[str, str] = {}

    def on_event(self, event: Any) -> Iterable[NormalizedEvent]:
        if isinstance(event, PartStartEvent):
            return list(self._on_part_start(event))
        if isinstance(event, PartDeltaEvent):
            return list(self._on_part_delta(event))
        if isinstance(event, PartEndEvent):
            return list(self._on_part_end(event))
        if isinstance(event, FunctionToolCallEvent):
            return []
        if isinstance(event, FunctionToolResultEvent):
            return []
        if isinstance(event, AgentRunResultEvent):
            return [self._usage_final(event)]
        return []

    def _on_part_start(self, event: PartStartEvent) -> Iterable[NormalizedEvent]:
        part = event.part
        item_key = f"part:{event.index}"
        self._index_to_item_key[event.index] = item_key

        if isinstance(part, TextPart):
            self._item_kind[item_key] = "message"
            yield MessageStarted(item_key=item_key)
            if part.content:
                yield MessageDelta(item_key=item_key, delta=part.content)
            return

        if isinstance(part, ThinkingPart):
            self._item_kind[item_key] = "reasoning"
            yield ReasoningStarted(item_key=item_key)
            if part.content:
                yield ReasoningDelta(item_key=item_key, delta=part.content)
            return

        if isinstance(part, ToolCallPart):
            self._tool_call_id_to_item_key[part.tool_call_id] = item_key
            self._item_kind[item_key] = "function_call"
            yield FunctionCallStarted(
                item_key=item_key,
                call_id=part.tool_call_id,
                name=part.tool_name,
                initial_arguments_json="",
            )
            if part.args_as_json_str():
                yield FunctionCallArgumentsDelta(
                    item_key=item_key, delta=part.args_as_json_str()
                )

    def _on_part_delta(self, event: PartDeltaEvent) -> Iterable[NormalizedEvent]:
        item_key = self._index_to_item_key.get(event.index)
        if item_key is None:
            return

        delta = event.delta
        if isinstance(delta, TextPartDelta):
            if delta.content_delta:
                yield MessageDelta(item_key=item_key, delta=delta.content_delta)
            return

        if isinstance(delta, ThinkingPartDelta):
            if delta.content_delta:
                yield ReasoningDelta(item_key=item_key, delta=delta.content_delta)
            return

        if isinstance(delta, ToolCallPartDelta):
            tool_item_key = (
                self._tool_call_id_to_item_key.get(delta.tool_call_id, item_key)
                if delta.tool_call_id is not None
                else item_key
            )
            if delta.args_delta is None:
                return
            yield FunctionCallArgumentsDelta(
                item_key=tool_item_key,
                delta=delta.args_delta
                if isinstance(delta.args_delta, str)
                else json.dumps(
                    delta.args_delta, separators=(",", ":"), ensure_ascii=False
                ),
            )

    def _on_part_end(self, event: PartEndEvent) -> Iterable[NormalizedEvent]:
        part = event.part
        item_key = self._index_to_item_key.get(event.index, f"part:{event.index}")

        if isinstance(part, TextPart):
            yield MessageDone(item_key=item_key, text=part.content)
            return

        if isinstance(part, ThinkingPart):
            yield ReasoningDone(item_key=item_key, text=part.content)
            return

        if isinstance(part, ToolCallPart):
            tool_item_key = self._tool_call_id_to_item_key.get(
                part.tool_call_id, item_key
            )
            yield FunctionCallDone(
                item_key=tool_item_key,
                arguments_json=part.args_as_json_str(),
            )

    @staticmethod
    def _usage_final(event: AgentRunResultEvent) -> UsageFinal:
        run_usage: RunUsage = event.result.usage()
        incomplete_reason = _incomplete_reason_from_model_messages(
            event.result.all_messages()
        )
        return UsageFinal(
            input_tokens=run_usage.input_tokens,
            output_tokens=run_usage.output_tokens,
            total_tokens=run_usage.total_tokens,
            cache_read_tokens=run_usage.cache_read_tokens,
            cache_write_tokens=run_usage.cache_write_tokens,
            reasoning_tokens=run_usage.details.get("reasoning_tokens", 0),
            incomplete_reason=incomplete_reason,
        )


def _incomplete_reason_from_model_messages(
    messages: list[ModelMessage],
) -> str | None:
    for message in reversed(messages):
        if not isinstance(message, ModelResponse):
            continue
        finish_reason = message.finish_reason
        if finish_reason == "length":
            return "max_output_tokens"
        if finish_reason == "content_filter":
            return "content_filter"
        return None
    return None
