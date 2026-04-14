"""Unit tests for StoreInputTranslator."""

from __future__ import annotations

from agentic_api.store.translator import StoreInputTranslator
from agentic_api.types.responses import (
    AutoToolChoice,
    FunctionTool,
    FunctionToolChoice,
    InputMessage,
    NoneToolChoice,
    RequiredToolChoice,
)


def translator() -> StoreInputTranslator:
    StoreInputTranslator._instance = None
    return StoreInputTranslator()


def _tool(name: str) -> FunctionTool:
    return FunctionTool(name=name, description=f"tool {name}")


def test_normalize_input_str_wraps_in_user_message() -> None:
    t = translator()
    result = t.normalize_input("hello")
    assert len(result) == 1
    assert isinstance(result[0], InputMessage)
    assert result[0].role == "user"
    assert result[0].content == "hello"


def test_normalize_input_list_returned_unchanged() -> None:
    t = translator()
    items = [InputMessage(role="user", content="hi")]
    result = t.normalize_input(items)
    assert result is items


def test_normalize_input_empty_list_returned_unchanged() -> None:
    t = translator()
    result = t.normalize_input([])
    assert result == []


def test_resolve_tools_returns_request_tools_when_explicitly_set() -> None:
    t = translator()
    request_tools = [_tool("search")]
    stored_tools = [_tool("calc")]
    result = t.resolve_tools(
        request_tools=request_tools,
        stored_tools=stored_tools,
        tools_explicitly_set=True,
    )
    assert result is not None
    assert result[0].name == "search"


def test_resolve_tools_returns_stored_tools_when_not_explicitly_set() -> None:
    t = translator()
    request_tools = [_tool("search")]
    stored_tools = [_tool("calc")]
    result = t.resolve_tools(
        request_tools=request_tools,
        stored_tools=stored_tools,
        tools_explicitly_set=False,
    )
    assert result is not None
    assert result[0].name == "calc"


def test_resolve_tools_returns_none_when_effective_is_none() -> None:
    t = translator()
    result = t.resolve_tools(
        request_tools=None,
        stored_tools=None,
        tools_explicitly_set=False,
    )
    assert result is None


def test_resolve_tools_request_none_falls_back_to_stored() -> None:
    t = translator()
    stored_tools = [_tool("calc")]
    result = t.resolve_tools(
        request_tools=None,
        stored_tools=stored_tools,
        tools_explicitly_set=False,
    )
    assert result is not None
    assert result[0].name == "calc"


def test_resolve_tools_stored_none_explicitly_set_returns_none() -> None:
    t = translator()
    result = t.resolve_tools(
        request_tools=None,
        stored_tools=[_tool("calc")],
        tools_explicitly_set=True,
    )
    assert result is None


def test_resolve_tool_choice_returns_request_when_explicitly_set() -> None:
    t = translator()
    result = t.resolve_tool_choice(
        request_tool_choice=RequiredToolChoice(),
        stored_tool_choice=AutoToolChoice(),
        tool_choice_explicitly_set=True,
    )
    assert isinstance(result, RequiredToolChoice)


def test_resolve_tool_choice_returns_stored_when_not_explicitly_set() -> None:
    t = translator()
    result = t.resolve_tool_choice(
        request_tool_choice=RequiredToolChoice(),
        stored_tool_choice=NoneToolChoice(),
        tool_choice_explicitly_set=False,
    )
    assert isinstance(result, NoneToolChoice)


def test_resolve_tool_choice_function_choice_preserved() -> None:
    t = translator()
    stored = FunctionToolChoice(name="search")
    result = t.resolve_tool_choice(
        request_tool_choice=AutoToolChoice(),
        stored_tool_choice=stored,
        tool_choice_explicitly_set=False,
    )
    assert isinstance(result, FunctionToolChoice)
    assert result.name == "search"


def test_store_translator_is_singleton() -> None:
    StoreInputTranslator._instance = None
    t1 = StoreInputTranslator()
    t2 = StoreInputTranslator()
    assert t1 is t2
