import argparse
import sys

from agentic_api.config.runtime import RuntimeConfig
from agentic_api.entrypoints.serve import run


def _normalize_base_url(url: str) -> str:
    """Strip trailing /v1 or /v1/ so callers can pass either form."""
    stripped = url.rstrip("/")
    if stripped.endswith("/v1"):
        stripped = stripped[:-3]
    return stripped.rstrip("/")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="agentic-API",
        description="Start the agentic-API gateway in front of a running vLLM server.",
    )
    parser.add_argument(
        "--llm-api-base",
        required=True,
        help="Base URL of the upstream vLLM server (e.g. http://127.0.0.1:8000). /v1 suffix is optional.",
    )
    parser.add_argument(
        "--openai-api-key", default=None, help="API key forwarded to upstream."
    )
    parser.add_argument("--gateway-host", default="0.0.0.0")
    parser.add_argument("--gateway-port", type=int, default=9000)
    parser.add_argument("--gateway-workers", type=int, default=1)
    parser.add_argument(
        "--upstream-ready-timeout",
        type=float,
        default=600.0,
        dest="upstream_ready_timeout_s",
    )
    parser.add_argument(
        "--upstream-ready-interval",
        type=float,
        default=2.0,
        dest="upstream_ready_interval_s",
    )
    parser.add_argument(
        "--db-url",
        default="sqlite+aiosqlite:///./agentic_api.db",
        dest="db_url",
        help="SQLAlchemy async database URL (e.g. sqlite+aiosqlite:///./agentic_api.db).",
    )
    parser.add_argument(
        "--embedding-api-base",
        default=None,
        dest="embedding_api_base",
        help="Base URL for the embedding API (defaults to --llm-api-base).",
    )
    parser.add_argument(
        "--embedding-model",
        default=None,
        dest="embedding_model",
        help="Model name for embedding requests.",
    )
    parser.add_argument(
        "--embedding-api-key",
        default=None,
        dest="embedding_api_key",
        help="API key for the embedding endpoint (defaults to --openai-api-key).",
    )
    parser.add_argument(
        "--vector-store-db-path",
        default="./vector_store.db",
        dest="vector_store_db_path",
        help="Path to the sqlite-vec database for vector store tables.",
    )

    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    runtime_config = RuntimeConfig(
        llm_api_base=_normalize_base_url(args.llm_api_base),
        openai_api_key=args.openai_api_key,
        gateway_host=args.gateway_host,
        gateway_port=args.gateway_port,
        gateway_workers=args.gateway_workers,
        upstream_ready_timeout_s=args.upstream_ready_timeout_s,
        upstream_ready_interval_s=args.upstream_ready_interval_s,
        db_url=args.db_url,
        embedding_api_base=args.embedding_api_base,
        embedding_model=args.embedding_model,
        embedding_api_key=args.embedding_api_key,
        vector_store_db_path=args.vector_store_db_path,
    )
    run(runtime_config)


if __name__ == "__main__":
    main()
