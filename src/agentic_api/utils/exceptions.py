from __future__ import annotations


class AgenticAPIError(Exception):
    """Base exception for agentic-api errors."""


class BadInputError(AgenticAPIError):
    """The request input is invalid."""


class ResponsesAPIError(AgenticAPIError):
    """OpenAI Responses-style structured API error."""

    def __init__(
        self,
        message: str,
        *,
        status_code: int,
        error_type: str = "invalid_request_error",
        param: str | None = None,
        code: str | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = int(status_code)
        self.error_type = str(error_type)
        self.param = param
        self.code = code
