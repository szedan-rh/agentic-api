from __future__ import annotations

from typing import AsyncGenerator

from pydantic_ai import Agent, ModelHTTPError, UnexpectedModelBehavior
from pydantic_ai.models.openai import OpenAIResponsesModel
from pydantic_ai.providers.openai import OpenAIProvider

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.core.pipeline import Pipeline
from agentic_api.core.sse import stream_responses_sse
from agentic_api.core.translator import RequestInputTranslator
from agentic_api.store.rehydration import ResponseStore
from agentic_api.types.responses import (
    AgentRunSettings,
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
        store: ResponseStore,
        runtime_config: RuntimeConfig,
    ) -> None:
        self._body = body
        self._store = store
        self._runtime_config = runtime_config
        self._translator = RequestInputTranslator()
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
        hydrated_body, run_settings, pipeline = await self._prepare_request()
        response: ResponsesResponse | None = None
        async for chunk in self._iter_events(run_settings, pipeline, stream=False):
            if chunk.type in {"response.completed", "response.incomplete"}:
                response = pipeline.composer.response
                await self._store.put_completed(
                    request=self._body,
                    hydrated_request=hydrated_body,
                    response=response,
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
        hydrated_body, run_settings, pipeline = await self._prepare_request()
        async for event in self._iter_events(run_settings, pipeline, stream=True):
            if event.type in {"response.completed", "response.incomplete"}:
                await self._store.put_completed(
                    request=self._body,
                    hydrated_request=hydrated_body,
                    response=pipeline.composer.response,
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

    async def _prepare_request(
        self,
    ) -> tuple[ResponsesRequest, AgentRunSettings, Pipeline]:
        """Rehydrate conversation history and build a fresh Pipeline for this request."""
        response = ResponsesResponse.create_from_response_request(self._body)
        hydrated_body = await self._store.rehydrate_request(request=self._body)
        run_settings = self._build_run_settings(hydrated_body)
        pipeline = Pipeline.build(response=response)
        return hydrated_body, run_settings, pipeline

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
