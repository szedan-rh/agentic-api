from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Self, Any, AsyncGenerator

from pydantic_ai import (
    Agent,
    DeferredToolRequests,
    ModelHTTPError,
    UnexpectedModelBehavior,
    capture_run_messages,
    ModelMessage,
)

from agentic_api.core.composer import ResponseComposer
from agentic_api.utils.failures import (
    FailureCounters,
    FailureDetails,
    classify_failure_log_level,
    extract_failure_details,
    log_failure_summary,
)
from agentic_api.core.normalizer import PydanticAINormalizer
from agentic_api.types.responses import AgentRunSettings, ResponsesResponse, StreamEvent


_ComposedEvent = StreamEvent


class Pipeline:
    """Per-request processing pipeline: owns one Composer and one Normalizer instance.

    Instantiate via `Pipeline.build()` once per request. The pipeline holds all mutable
    event-processing state for that request and must not be reused across requests.
    """

    def __init__(
        self,
        *,
        composer: ResponseComposer,
        normalizer: PydanticAINormalizer,
    ) -> None:
        self.composer = composer
        self.normalizer = normalizer

    @classmethod
    def build(cls, *, response: ResponsesResponse) -> Self:
        composer = ResponseComposer(response=response)
        normalizer = PydanticAINormalizer()
        return cls(composer=composer, normalizer=normalizer)

    @asynccontextmanager
    async def run_agent(
        self,
        agent: Agent,
        run_settings: AgentRunSettings,
        failure_counters: FailureCounters,
    ) -> AsyncGenerator[
        tuple[AsyncGenerator[_ComposedEvent, None], list[ModelMessage]], None
    ]:
        """Context manager that captures messages and yields (events, messages).

        Usage::

            async with pipeline.run_agent(agent, run_settings, counters) as (events, messages):
                try:
                    async for out in events:
                        yield out
                except (ModelHTTPError, UnexpectedModelBehavior) as e:
                    details = pipeline.log_failure(..., messages=messages, ...)
        """
        with capture_run_messages() as messages:
            yield self._iter_composed(agent, run_settings, failure_counters), messages

    async def _iter_composed(
        self,
        agent: Agent,
        run_settings: AgentRunSettings,
        failure_counters: FailureCounters,
    ) -> AsyncGenerator[_ComposedEvent, None]:
        for chunk in self.composer.start():
            yield chunk
        async for event in agent.run_stream_events(
            output_type=[agent.output_type, DeferredToolRequests],
            message_history=run_settings["message_history"],
            instructions=run_settings["instructions"],
            toolsets=run_settings["toolsets"],
            usage_limits=run_settings["usage_limits"],
        ):
            for normalized in self.normalizer.on_event(event):
                failure_counters.observe(normalized)
                for composed in self.composer.feed(normalized):
                    yield composed

    def log_failure(
        self,
        *,
        phase: str,
        e: ModelHTTPError | UnexpectedModelBehavior,
        messages: list[Any],
        counters: FailureCounters,
        log_model_messages: bool,
    ) -> FailureDetails:
        """Extract, classify, and log a failure. Returns FailureDetails for the caller."""
        details = extract_failure_details(e)
        log_failure_summary(
            response_id=self.composer.response.id,
            failure_phase=phase,
            error_class=details.error_class,
            log_level=classify_failure_log_level(
                error_class=details.error_class,
                upstream_status_code=details.upstream_status_code,
            ),
            upstream_status_code=details.upstream_status_code,
            error_message=details.message,
            messages=messages,
            counters=counters,
            upstream_error_raw=details.upstream_error_raw,
            log_model_messages=log_model_messages,
        )
        return details
