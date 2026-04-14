from __future__ import annotations

from pydantic_ai.messages import (
    ModelRequest,
    ModelResponse,
    SystemPromptPart,
    TextPart,
    ToolCallPart,
    ToolReturnPart,
    UserPromptPart,
)

from agentic_api.core.translator import RequestInputTranslator
from agentic_api.types.responses import (
    FunctionToolCall,
    FunctionToolResultMessage,
    InputMessage,
    InputTextContent,
    OutputMessage,
    OutputTextContent,
)


def translator() -> RequestInputTranslator:
    # Reset singleton so each test gets a clean instance.
    RequestInputTranslator._instance = None
    return RequestInputTranslator()


def test_user_message_becomes_model_request_with_user_prompt_part() -> None:
    t = translator()
    msg = InputMessage(role="user", content="hello")
    result = t.translate([msg])
    assert len(result) == 1
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[0].parts[0], UserPromptPart)
    assert result[0].parts[0].content == "hello"


def test_system_message_becomes_model_request_with_system_prompt_part() -> None:
    t = translator()
    msg = InputMessage(role="system", content="be helpful")
    result = t.translate([msg])
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[0].parts[0], SystemPromptPart)
    assert result[0].parts[0].content == "be helpful"


def test_developer_message_becomes_system_prompt_part() -> None:
    t = translator()
    msg = InputMessage(role="developer", content="system instructions")
    result = t.translate([msg])
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[0].parts[0], SystemPromptPart)


def test_assistant_input_message_becomes_model_response_with_text_part() -> None:
    t = translator()
    msg = InputMessage(role="assistant", content="I can help")
    result = t.translate([msg])
    assert isinstance(result[0], ModelResponse)
    assert isinstance(result[0].parts[0], TextPart)
    assert result[0].parts[0].content == "I can help"


def test_input_message_with_content_list_joins_text() -> None:
    t = translator()
    msg = InputMessage(
        role="user",
        content=[InputTextContent(text="hello "), InputTextContent(text="world")],
    )
    result = t.translate([msg])
    assert result[0].parts[0].content == "hello world"


def test_tool_result_becomes_model_request_with_tool_return_part() -> None:
    t = translator()
    msg = FunctionToolResultMessage(call_id="call_1", output='{"result":42}')
    result = t.translate([msg])
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[0].parts[0], ToolReturnPart)
    assert result[0].parts[0].tool_call_id == "call_1"
    assert result[0].parts[0].content == '{"result":42}'


def test_output_message_becomes_model_response_with_text_part() -> None:
    t = translator()
    msg = OutputMessage(id="msg_1", content=[OutputTextContent(text="the answer")])
    result = t.translate([msg])
    assert isinstance(result[0], ModelResponse)
    assert isinstance(result[0].parts[0], TextPart)
    assert result[0].parts[0].content == "the answer"


def test_output_message_with_multiple_content_parts_joins_text() -> None:
    t = translator()
    msg = OutputMessage(
        id="msg_1",
        content=[OutputTextContent(text="foo"), OutputTextContent(text="bar")],
    )
    result = t.translate([msg])
    assert result[0].parts[0].content == "foobar"


def test_function_tool_call_becomes_model_response_with_tool_call_part() -> None:
    t = translator()
    msg = FunctionToolCall(
        id="fc_1", call_id="call_1", name="search", arguments='{"q":"test"}'
    )
    result = t.translate([msg])
    assert isinstance(result[0], ModelResponse)
    assert isinstance(result[0].parts[0], ToolCallPart)
    assert result[0].parts[0].tool_name == "search"
    assert result[0].parts[0].tool_call_id == "call_1"


# ---------------------------------------------------------------------------
# Mixed sequence — order preserved
# ---------------------------------------------------------------------------


def test_translate_preserves_order_of_mixed_items() -> None:
    t = translator()
    items = [
        InputMessage(role="user", content="turn 1"),
        OutputMessage(id="msg_1", content=[OutputTextContent(text="reply 1")]),
        InputMessage(role="user", content="turn 2"),
        OutputMessage(id="msg_2", content=[OutputTextContent(text="reply 2")]),
    ]
    result = t.translate(items)
    assert len(result) == 4
    assert isinstance(result[0], ModelRequest)
    assert isinstance(result[1], ModelResponse)
    assert isinstance(result[2], ModelRequest)
    assert isinstance(result[3], ModelResponse)


def test_empty_list_returns_empty() -> None:
    t = translator()
    assert t.translate([]) == []


def test_translator_is_singleton() -> None:
    RequestInputTranslator._instance = None
    t1 = RequestInputTranslator()
    t2 = RequestInputTranslator()
    assert t1 is t2
