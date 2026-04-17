from __future__ import annotations

from typing import Literal

from pydantic import BaseModel


class MessageStarted(BaseModel):
    item_key: str


class MessageDelta(BaseModel):
    item_key: str
    delta: str


class MessageDone(BaseModel):
    item_key: str
    text: str


class ReasoningStarted(BaseModel):
    item_key: str


class ReasoningDelta(BaseModel):
    item_key: str
    delta: str


class ReasoningDone(BaseModel):
    item_key: str
    text: str


class FunctionCallStarted(BaseModel):
    item_key: str
    call_id: str
    name: str
    initial_arguments_json: str


class FunctionCallArgumentsDelta(BaseModel):
    item_key: str
    delta: str


class FunctionCallDone(BaseModel):
    item_key: str
    arguments_json: str


class UsageFinal(BaseModel):
    input_tokens: int
    output_tokens: int
    total_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    reasoning_tokens: int
    incomplete_reason: Literal["max_output_tokens", "content_filter"] | None = None


NormalizedEvent = (
    MessageStarted
    | MessageDelta
    | MessageDone
    | ReasoningStarted
    | ReasoningDelta
    | ReasoningDone
    | FunctionCallStarted
    | FunctionCallArgumentsDelta
    | FunctionCallDone
    | UsageFinal
)
