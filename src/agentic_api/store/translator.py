from __future__ import annotations

from pydantic import TypeAdapter

from agentic_api.types.responses import (
    InputItem,
    ResponsesTool,
    ToolChoice,
)

_input_item_adapter = TypeAdapter(list[InputItem])
_tools_adapter = TypeAdapter(list[ResponsesTool])
_tool_choice_adapter = TypeAdapter(ToolChoice)


class StoreInputTranslator:
    """Translates raw request input into store-ready types.

    Mirrors the role of RequestInputTranslator on the engine side:
      RequestInputTranslator  — list[InputItem | OutputItem] → list[ModelMessage]  (engine)
      StoreInputTranslator    — str | list[InputItem]        → list[InputItem]      (store)
    """

    _instance: StoreInputTranslator | None = None

    def __new__(cls) -> StoreInputTranslator:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def normalize_input(self, value: str | list[InputItem]) -> list[InputItem]:
        if isinstance(value, str):
            return _input_item_adapter.validate_python(
                [{"type": "message", "role": "user", "content": value}]
            )
        return value

    def resolve_tools(
        self,
        *,
        request_tools: list[ResponsesTool] | None,
        stored_tools: list[ResponsesTool] | None,
        tools_explicitly_set: bool,
    ) -> list[ResponsesTool] | None:
        effective = request_tools if tools_explicitly_set else stored_tools
        return (
            _tools_adapter.validate_python(effective) if effective is not None else None
        )

    def resolve_tool_choice(
        self,
        *,
        request_tool_choice: ToolChoice,
        stored_tool_choice: ToolChoice,
        tool_choice_explicitly_set: bool,
    ) -> ToolChoice:
        effective = (
            request_tool_choice if tool_choice_explicitly_set else stored_tool_choice
        )
        return _tool_choice_adapter.validate_python(effective)
