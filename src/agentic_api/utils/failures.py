import json
import logging
from dataclasses import dataclass
from typing import Any

from pydantic_ai import ModelHTTPError, UnexpectedModelBehavior

from agentic_api.core.models import FunctionCallStarted

logger = logging.getLogger(__name__)


_FAILURE_ERROR_MESSAGE_MAX_CHARS = 512
_FAILURE_UPSTREAM_RAW_MAX_CHARS = 2048
_FAILURE_DEBUG_MESSAGE_MAX_CHARS = 8192


def _truncate_prefix(value: str, limit: int) -> str:
    return value[:limit]


@dataclass(frozen=True, slots=True)
class FailureDetails:
    error_class: str
    code: str
    message: str
    param: str
    upstream_status_code: int | None
    upstream_error_raw: str | None


@dataclass(slots=True)
class FailureCounters:
    tool_call_parts_seen: int = 0

    def observe(self, normalized_event: object) -> None:
        if isinstance(normalized_event, FunctionCallStarted):
            self.tool_call_parts_seen += 1


def _extract_error_body(raw: Any) -> dict[str, Any]:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (str, bytes, bytearray)):
        try:
            parsed = json.loads(raw)
        except Exception:
            return {}
        if isinstance(parsed, dict):
            return parsed
    return {}


def _upstream_error_raw(raw: Any) -> str | None:
    if raw is None:
        return None
    return _truncate_prefix(str(raw), _FAILURE_UPSTREAM_RAW_MAX_CHARS)


def _extract_openai_error_fields(
    err_body: dict[str, Any] | None,
    *,
    fallback_message: str,
) -> tuple[str, str, str]:
    body = err_body or {}
    err = body.get("error") if isinstance(body.get("error"), dict) else body
    if not isinstance(err, dict):
        return "", fallback_message, ""
    code_raw = err.get("code")
    code = "" if code_raw is None else str(code_raw)
    message_raw = err.get("message")
    message = (
        str(message_raw)
        if isinstance(message_raw, str) and message_raw
        else fallback_message
    )
    param_raw = err.get("param")
    param = "" if param_raw is None else str(param_raw)
    return code, message, param


def extract_failure_details(
    e: ModelHTTPError | UnexpectedModelBehavior,
) -> FailureDetails:
    if isinstance(e, ModelHTTPError):
        err_body = _extract_error_body(e.body)
        code, message, param = _extract_openai_error_fields(
            err_body, fallback_message=str(e)
        )
        return FailureDetails(
            error_class=e.__class__.__name__,
            code=code,
            message=message,
            param=param,
            upstream_status_code=e.status_code,
            upstream_error_raw=_upstream_error_raw(e.body),
        )

    err_body = _extract_error_body(e.body)
    code, message, param = _extract_openai_error_fields(
        err_body, fallback_message=e.message
    )
    return FailureDetails(
        error_class=e.__class__.__name__,
        code=code,
        message=message,
        param=param,
        upstream_status_code=None,
        upstream_error_raw=_upstream_error_raw(e.body),
    )


def classify_failure_log_level(
    *, error_class: str, upstream_status_code: int | None
) -> str:
    if error_class == ModelHTTPError.__name__ and upstream_status_code is not None:
        if 400 <= upstream_status_code < 500:
            return "warning"
    return "error"


def log_failure_summary(
    *,
    response_id: str | None,
    failure_phase: str,
    error_class: str,
    log_level: str,
    upstream_status_code: int | None,
    error_message: str,
    messages: list[Any] | Any,
    counters: FailureCounters,
    upstream_error_raw: str | None,
    log_model_messages: bool,
) -> None:
    summary: dict[str, Any] = {
        "request_id": response_id,
        "failure_phase": failure_phase,
        "error_class": error_class,
        "log_level": log_level,
        "upstream_status_code": upstream_status_code,
        "error_message": _truncate_prefix(
            error_message, _FAILURE_ERROR_MESSAGE_MAX_CHARS
        ),
        "total_messages": len(messages) if hasattr(messages, "__len__") else 0,
        "tool_call_parts_seen": counters.tool_call_parts_seen,
    }
    if upstream_error_raw:
        summary["upstream_error_raw"] = upstream_error_raw

    log_fn = logger.warning if log_level == "warning" else logger.error
    log_fn("Engine failure summary: %s", summary)

    if not log_model_messages:
        return

    if not isinstance(messages, list):
        log_fn(
            "Engine captured messages debug dump unavailable for type: %s",
            type(messages).__name__,
        )
        return

    for i, entry in enumerate(messages):
        log_fn(
            "Engine captured_messages[%s]: %s",
            i,
            _truncate_prefix(repr(entry), _FAILURE_DEBUG_MESSAGE_MAX_CHARS),
        )
