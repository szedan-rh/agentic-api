from typing import Literal

from pydantic import BaseModel, Field, computed_field, field_validator
from sqlalchemy.engine import make_url


class RuntimeConfig(BaseModel):
    llm_api_base: str = Field(
        description="Base URL of the upstream vLLM server (e.g. http://127.0.0.1:8000)."
    )
    openai_api_key: str | None = Field(
        description="API key forwarded to the upstream server."
    )

    gateway_host: str = Field(description="Host address the gateway binds to.")
    gateway_port: int = Field(description="Port the gateway listens on.")
    gateway_workers: int = Field(description="Number of uvicorn worker processes.")

    upstream_ready_timeout_s: float = Field(
        description="Seconds to wait for the upstream to become ready before aborting."
    )
    upstream_ready_interval_s: float = Field(
        description="Polling interval in seconds when waiting for the upstream to become ready."
    )

    db_url: str = Field(
        default="sqlite+aiosqlite:///./agentic_api.db",
        description="SQLAlchemy async database URL.",
    )

    @field_validator("db_url")
    @classmethod
    def validate_db_url(cls, v: str) -> str:
        url = make_url(v)
        if not url.get_dialect().is_async:
            raise ValueError(
                f"db_url must use an async driver (e.g. sqlite+aiosqlite://, postgresql+asyncpg://), got: {v!r}"
            )
        return v

    @computed_field
    @property
    def db_dialect(self) -> Literal["sqlite", "postgresql"]:
        return make_url(self.db_url).get_dialect().name

    response_store_enabled: bool = Field(
        default=True,
        description="Enable response persistence for multi-turn rehydration.",
    )

    log_model_messages: bool = Field(
        default=False,
        description="Log full pydantic_ai model messages on engine failure (verbose).",
    )

    embedding_api_base: str | None = Field(
        default=None,
        description="Base URL for the embedding API (defaults to llm_api_base).",
    )
    embedding_model: str | None = Field(
        default=None,
        description="Model name for embedding requests.",
    )
    embedding_api_key: str | None = Field(
        default=None,
        description="API key for the embedding endpoint (defaults to openai_api_key).",
    )
    vector_store_db_path: str = Field(
        default="./vector_store.db",
        description="Path to the sqlite-vec database for vector store tables.",
    )
