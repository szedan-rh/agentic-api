# agentic-api
Stateful API logic for agentic applications using vLLM

A Rust-based gateway that adds stateful, agentic capabilities on top of
[vLLM](https://github.com/vllm-project/vllm)'s high-throughput inference engine.
Built on [Praxis](https://github.com/praxis-proxy/praxis), a composable filter-based
proxy framework, so each concern (state hydration, tool dispatch, agentic looping) is
an independent filter wired together via YAML configuration.

Design decisions are tracked in the ADRs under `docs/adr/`.

## Architecture

```
Client -> [Agentic API (Praxis filters)] -> [vLLM Core]
                      |
              [State Store]
           (Files, Vector Stores,
           Search, Conversations)
```

Filters in the pipeline:

| Filter | Role |
|--------|------|
| `state_hydration` | Hydrates conversation state via `previous_response_id` |
| `agentic_loop` | Detects tool calls and re-enters the inference loop |
| `tool_dispatch` | Executes tool calls (MCP, code interpreter, file search) |
| `responses_proxy` | Routes requests to vLLM's `/v1/responses` endpoint |

## Repository layout

- `src/filters/` — Praxis filter implementations
- `config/agentic-api.yaml` — Default filter pipeline configuration
- `docs/` — Documentation and ADRs

## Build

```bash
cargo build
```

## Run

```bash
cargo run -- -c config/agentic-api.yaml
```

## Test

```bash
cargo test
```

## Lint and format

```bash
cargo clippy --all-targets -- -D warnings
cargo fmt -- --check
```

## Documentation

```bash
uv venv
uv pip install -r docs/requirements.txt
uv run mkdocs serve
```
