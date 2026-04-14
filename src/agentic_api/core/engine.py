from __future__ import annotations

from typing import AsyncGenerator

from pydantic_ai import Agent, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.core.pipeline import Pipeline
from agentic_api.core.sse import stream_responses_sse
from agentic_api.core.translator import RequestInputTranslator
from agentic_api.store.conversation import ConversationStore, StoredConversation
from agentic_api.store.response import ResponseMetadata, ResponseStore
from agentic_api.store.translator import StoreInputTranslator
from agentic_api.types.responses import (
    AgentRunSettings,
    InputItem,
    OutputItem,
    ResponsesRequest,
    ResponsesResponse,
    StreamEvent,
)
from agentic_api.utils.exceptions import BadInputError
from agentic_api.utils.failures import FailureCounters


def _build_openai_provider(runtime_config: RuntimeConfig) -> OpenAIProvider:
    base = runtime_config.llm_api_base.rstrip("/")
    if not base.endswith("/v1"):
        base = f"{base}/v1"
    return OpenAIProvider(
        api_key=runtime_config.openai_api_key or "",
        base_url=base,
    )


class Engine:
    """Orchestrate one Responses API request using the upstream vLLM via pydantic_ai.

    Input:  ResponsesRequest (with optional previous_response_id for multi-turn)
    Output: ResponsesResponse (non-stream) or AsyncGenerator[str] (SSE stream)

    The engine is stateless across requests — instantiate once per request.
    """

    def __init__(
        self,
        body: ResponsesRequest,
        *,
        response_store: ResponseStore,
        conversation_store: ConversationStore | None,
        runtime_config: RuntimeConfig,
    ) -> None:
        self._body = body
        self._response_store = response_store
        self._conversation_store = conversation_store
        self._runtime_config = runtime_config
        self._translator = RequestInputTranslator()
        self._store_translator = StoreInputTranslator()
        self._agent = Agent(
            OpenAIResponsesModel(
                model_name=body.model,
                provider=_build_openai_provider(runtime_config),
            ),
            model_settings=body.as_openai_chat_settings(),
        )

    async def run(self) -> AsyncGenerator[str, None] | ResponsesResponse:
        if self._body.stream:
            return self._run_stream()
        return await self._run()

    async def _run(self) -> ResponsesResponse:
        (
            hydrated_body,
            conversation,
            run_settings,
            pipeline,
        ) = await self._prepare_request()
        response: ResponsesResponse | None = None
        async for chunk in self._iter_events(run_settings, pipeline, stream=False):
            if chunk.type in {"response.completed", "response.incomplete"}:
                response = pipeline.composer.response
                await self._persist(
                    hydrated_body=hydrated_body,
                    response=response,
                    conversation=conversation,
                )
        if response is None:
            raise BadInputError("No response generated from Engine.")
        return response

    async def _run_stream(self) -> AsyncGenerator[str, None]:
        async for frame in stream_responses_sse(self._tap_stream()):
            yield frame

    async def _tap_stream(
        self,
    ) -> AsyncGenerator[StreamEvent, None]:
        (
            hydrated_body,
            conversation,
            run_settings,
            pipeline,
        ) = await self._prepare_request()
        async for event in self._iter_events(run_settings, pipeline, stream=True):
            if event.type in {"response.completed", "response.incomplete"}:
                await self._persist(
                    hydrated_body=hydrated_body,
                    response=pipeline.composer.response,
                    conversation=conversation,
                )
            yield event

    def _build_run_settings(self, request: ResponsesRequest) -> AgentRunSettings:
        items = request.input if isinstance(request.input, list) else []
        return AgentRunSettings(
            message_history=self._translator.translate(items),
            instructions=request.instructions,
            toolsets=[],
            usage_limits=None,
        )

    async def _resolve_conversation(self) -> StoredConversation | None:
        """Load the Conversation row when conversation_id is present on the request.

        Returns None for pure Responses API requests (no conversation_id).
        When conversation_id is provided, delegates to get_or_create which loads the
        existing row or creates a new one if the ID is not yet known to the store.
        """
        if self._conversation_store is None or self._body.conversation_id is None:
            return None
        return await self._conversation_store.get_or_create(
            conversation_id=self._body.conversation_id
        )

    async def _rehydrate(
        self, *, conversation: StoredConversation | None
    ) -> ResponsesRequest:
        """Resolve the hydrated input for this request.

        Branching:
        - No previous_response_id: normalise input only.
        - conversation set: Conversation API — use ConversationStore.rehydrate().
        - otherwise: Responses API — use ResponseStore.rehydrate().
        """
        new_input = self._store_translator.normalize_input(self._body.input)
        if not self._body.previous_response_id:
            return self._body.model_copy(update={"input": new_input})

        if conversation is not None:
            # Conversation API path: Conversation.item_ids is the authoritative checkpoint.
            history_items: list[
                InputItem | OutputItem
            ] = await self._conversation_store.rehydrate(  # type: ignore[union-attr]
                conversation_id=conversation.conversation_id
            )
            return self._body.model_copy(
                update={
                    "previous_response_id": None,
                    "input": [*history_items, *new_input],
                }
            )

        # Responses API path: load stored response and use history_item_ids checkpoint.
        stored = await self._response_store.get_or_raise(
            response_id=self._body.previous_response_id
        )
        history_items = await self._response_store.rehydrate(stored=stored)

        fields_set = self._body.model_fields_set
        return self._body.model_copy(
            update={
                "previous_response_id": None,
                "input": [*history_items, *new_input],
                "tools": self._store_translator.resolve_tools(
                    request_tools=self._body.tools,
                    stored_tools=stored.metadata.effective_tools,
                    tools_explicitly_set="tools" in fields_set,
                ),
                "tool_choice": self._store_translator.resolve_tool_choice(
                    request_tool_choice=self._body.tool_choice,
                    stored_tool_choice=stored.metadata.effective_tool_choice,
                    tool_choice_explicitly_set="tool_choice" in fields_set,
                ),
            }
        )

    async def _persist(
        self,
        *,
        hydrated_body: ResponsesRequest,
        response: ResponsesResponse,
        conversation: StoredConversation | None,
    ) -> None:
        """Persist the completed turn to the appropriate store.

        Branching:
        - conversation set: Conversation API — call ConversationStore.put_turn().
        - otherwise: Responses API — call ResponseStore.put_completed().
        """
        if response.status not in {"completed", "incomplete"}:
            return
        if not response.id:
            return
        if not self._body.store:
            return

        if conversation is not None:
            hydrated_input = self._store_translator.normalize_input(hydrated_body.input)
            new_items: list[InputItem | OutputItem] = [
                *hydrated_input,
                *response.output,
            ]
            metadata = ResponseMetadata(
                model=response.model,
                previous_response_id=response.previous_response_id,
                effective_tools=hydrated_body.tools,
                effective_tool_choice=hydrated_body.tool_choice,
                effective_instructions=hydrated_body.instructions,
            )
            await self._conversation_store.put_turn(  # type: ignore[union-attr]
                conversation_id=conversation.conversation_id,
                response_id=response.id,
                previous_response_id=response.previous_response_id,
                new_items=new_items,
                response_metadata=metadata.model_dump(mode="json"),
            )
        else:
            await self._response_store.put_completed(
                request=self._body,
                hydrated_request=hydrated_body,
                response=response,
            )

    async def _prepare_request(
        self,
    ) -> tuple[ResponsesRequest, StoredConversation | None, AgentRunSettings, Pipeline]:
        """Resolve conversation, rehydrate history, and build a fresh Pipeline for this request."""
        response = ResponsesResponse.create_from_response_request(self._body)
        conversation = await self._resolve_conversation()
        hydrated_body = await self._rehydrate(conversation=conversation)
        run_settings = self._build_run_settings(hydrated_body)
        pipeline = Pipeline.build(response=response)
        return hydrated_body, conversation, run_settings, pipeline

    async def _iter_events(
        self,
        run_settings: AgentRunSettings,
        pipeline: Pipeline,
        *,
        stream: bool,
    ) -> AsyncGenerator[StreamEvent, None]:
        failure_counters = FailureCounters()
        phase = "stream" if stream else "non_stream"

        for chunk in pipeline.composer.start():
            yield chunk

        async with pipeline.run_agent(self._agent, run_settings, failure_counters) as (
            events,
            messages,
        ):
            try:
                async for out in events:
                    yield out
            except (ModelHTTPError, UnexpectedModelBehavior) as e:
                details = pipeline.log_failure(
                    phase=phase,
                    e=e,
                    messages=messages,
                    counters=failure_counters,
                    log_model_messages=self._runtime_config.log_model_messages,
                )
                if not stream:
                    raise
                for err_event in pipeline.composer.make_error_events(
                    code=details.code,
                    message=details.message,
                    param=details.param,
                ):
                    yield err_event
