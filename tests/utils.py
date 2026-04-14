from __future__ import annotations

from pydantic_ai import Agent, AgentRunResult, RunUsage
from pydantic_ai._agent_graph import GraphAgentState
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentic_api.types.responses import (
    InputMessage,
    OutputMessage,
    OutputTextContent,
    ResponsesRequest,
    ResponsesResponse,
    ResponseUsage,
)


def make_request(
    *,
    model: str = "test-model",
    input: str = "hello",
    previous_response_id: str | None = None,
    store: bool = True,
) -> ResponsesRequest:
    return ResponsesRequest(
        model=model,
        input=input,
        previous_response_id=previous_response_id,
        store=store,
    )


def make_response(
    *,
    response_id: str = "resp_001",
    model: str = "test-model",
    text: str = "hi there",
    previous_response_id: str | None = None,
) -> ResponsesResponse:
    return ResponsesResponse(
        id=response_id,
        model=model,
        status="completed",
        output=[
            OutputMessage(
                id="msg_001",
                content=[OutputTextContent(text=text)],
            )
        ],
        usage=ResponseUsage(input_tokens=10, output_tokens=5, total_tokens=15),
        previous_response_id=previous_response_id,
    )


def make_user_msg(text: str) -> InputMessage:
    return InputMessage(role="user", content=text)


def make_assistant_msg(text: str) -> OutputMessage:
    return OutputMessage(id="msg_x", content=[OutputTextContent(text=text)])


def make_agent_run_result(
    input_tokens: int = 10,
    output_tokens: int = 5,
) -> AgentRunResult:
    state = GraphAgentState()
    state.usage = RunUsage(
        requests=1,
        input_tokens=input_tokens,
        output_tokens=output_tokens,
    )
    result = AgentRunResult.__new__(AgentRunResult)
    object.__setattr__(result, "_new_message_index", 0)
    object.__setattr__(result, "_output_tool_name", None)
    object.__setattr__(result, "_state", state)
    object.__setattr__(result, "_traceparent_value", None)
    object.__setattr__(result, "output", "")
    return result


def make_agent(model: str = "test-model") -> Agent:
    """Build an Agent using OpenAIResponsesModel with a dummy key — safe to construct without env vars."""
    return Agent(
        OpenAIResponsesModel(
            model_name=model,
            provider=OpenAIProvider(
                api_key="test-key",
                base_url="http://fake-upstream/v1",
            ),
        )
    )


def make_response_metadata(model: str = "test-model") -> dict:
    return {
        "model": model,
        "previous_response_id": None,
        "effective_tools": None,
        "effective_tool_choice": {"type": "auto"},
        "effective_instructions": None,
    }
