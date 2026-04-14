from __future__ import annotations

from pydantic_ai.messages import (
    ModelMessage,
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agentic_api.types.responses import (
    FunctionToolCall,
    FunctionToolResultMessage,
    InputItem,
    InputMessage,
    OutputItem,
    OutputMessage,
)


class RequestInputTranslator:
    """Translates rehydrated InputItem/OutputItem lists into pydantic_ai ModelMessages.

    Mirrors the role of PydanticAINormalizer on the output side:
      normalizer  — pydantic_ai events  → NormalizedEvent  (output)
      translator  — InputItem/OutputItem → ModelMessage[]  (input)
    """

    _instance: RequestInputTranslator | None = None

    def __new__(cls) -> RequestInputTranslator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def translate(self, items: list[InputItem | OutputItem]) -> list[ModelMessage]:
        messages: list[ModelMessage] = []
        for item in items:
            message = self._translate_item(item)
            if message is not None:
                messages.append(message)
        return messages

    def _translate_item(self, item: InputItem | OutputItem) -> ModelMessage | None:
        if isinstance(item, InputMessage):
            return self._translate_input_message(item)
        if isinstance(item, FunctionToolResultMessage):
            return self._translate_tool_result(item)
        if isinstance(item, OutputMessage):
            return self._translate_output_message(item)
        if isinstance(item, FunctionToolCall):
            return self._translate_function_tool_call(item)
        return None

    def _translate_input_message(self, item: InputMessage) -> ModelMessage:
        text = self._content_to_str(item.content)
        if item.role in ("system", "developer"):
            return ModelRequest(parts=[SystemPromptPart(content=text)])
        if item.role == "assistant":
            return ModelResponse(parts=[TextPart(content=text)])
        return ModelRequest(parts=[UserPromptPart(content=text)])

    def _translate_tool_result(self, item: FunctionToolResultMessage) -> ModelMessage:
        return ModelRequest(
            parts=[
                ToolReturnPart(
                    tool_name="", content=item.output, tool_call_id=item.call_id
                )
            ]
        )

    def _translate_output_message(self, item: OutputMessage) -> ModelMessage:
        text = "".join(c.text for c in item.content)
        return ModelResponse(parts=[TextPart(content=text)])

    def _translate_function_tool_call(self, item: FunctionToolCall) -> ModelMessage:
        return ModelResponse(
            parts=[
                ToolCallPart(
                    tool_name=item.name,
                    args=item.arguments,
                    tool_call_id=item.call_id,
                )
            ]
        )

    @staticmethod
    def _content_to_str(content: str | list) -> str:
        if isinstance(content, str):
            return content
        return "".join(c.text for c in content if hasattr(c, "text"))
