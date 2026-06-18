# ADR-04 - Cached rendered token IDs for vLLM Responses conversations

> **Status:** Draft
> **Date:** 2026-06-18
> **Related:** [ADR-01 - Core Architecture](ADR-01_core.md), [ADR-02 - Response Store](ADR-02_response_store.md), [ADR-03 - Layered Crate Architecture](ADR-03_gateway_integration.md)

---

## Intention

Evaluate and prototype whether `agentic-api` should persist rendered Harmony token IDs for completed Responses conversation turns, so future turns can avoid re-rendering the entire cumulative conversation.

The target workload is an agentic loop with many short turns, tool calls, and growing context, similar to a Codex-style 20-turn conversation.

---

## Context

vLLM serves an OpenAI-compatible `/v1/responses` endpoint. For `openai/gpt-oss-*` models, vLLM renders the full conversation into Harmony token IDs on each turn before prefill and decode.

The proposed optimization separates two possible wins:

1. **Render/tokenize CPU savings.** Avoid re-rendering and re-tokenizing stable conversation history.
2. **Prefix-cache stability.** Automatic prefix caching (APC) only helps when prefix token IDs match exactly. Persisting exact IDs can avoid subtle template or boundary drift.

The optimization is only correct if token spans are appended at Harmony message boundaries. We must not splice inside message text because `tokenize(A + B)` is not guaranteed to equal `tokenize(A) + tokenize(B)`.

---

## Prototype decision

Implement a Phase 2 prototype behind a flag.

### vLLM fork

Branch: `franciscojavierarceo/vllm.git`, `responses-tokens-endpoint`

Commits used for this ADR:

- `f146d224d feat: add response token ids for harmony responses`
- `1f5115c9 feat: support responses prompt cache replay`

The fork extends Responses with vLLM-specific token fields:

- Request: `prompt_token_ids`
- Request: `prompt_cache_ref`
- Request: `append_token_ids`
- Response: `prompt_token_ids`
- Response: `output_token_ids`

When `prompt_token_ids` is provided on a Harmony Responses request, vLLM uses those IDs directly for
generation instead of calling `render_for_completion(messages)`.

When `prompt_cache_ref` and `append_token_ids` are provided, vLLM resolves the referenced in-memory
prefix token span and appends only the marginal token IDs before generation. This avoids resending the
full prompt-token array over JSON. The handle registry is process-local in the prototype: a vLLM restart
clears the registered handles, so the first request after restart must seed the prefix through normal
rendering or `prompt_token_ids`.

Prototype limitation: vLLM still constructs the request's Harmony message list
before choosing the explicit token path. With a minimal `input` payload, that reconstruction is small;
with a full cumulative `input` payload, request JSON parsing and message reconstruction still cost
time even when token rendering is skipped.

### agentic-api

`agentic-api` now has a `token_cache_enabled` feature path that:

- requests token-debug output from vLLM by setting `enable_response_messages`
- persists a `CachedTokenSpan` on the response row
- stores compatibility metadata: model, tokenizer, renderer, template fingerprint, effective instructions hash, effective tools hash
- rehydrates the latest cached span for a conversation when the flag is enabled

The prototype deliberately marks captured spans with `ends_at_boundary = false`. That makes the data usable for measurement and diagnostics, but prevents accidental live replay until boundary validation exists.

---

## Performance model

For turn `i`:

```text
C_i = cumulative prompt tokens
d_i = marginal new prompt tokens
g_i = generated tokens

TTFT(i) ~= R(C_i) + Prefill(i) + fixed_overhead
TurnLatency(i) ~= TTFT(i) + decode_per_tok * g_i
```

With a correct rendered-token cache:

```text
R(C_i) -> R(d_i)
Gain A ~= R_per_tok * C_{i-1}
```

If re-rendered token IDs drift and break APC:

```text
Gain B ~= prefill_per_tok * extra_reprefilled_tokens
```

Gain B is conditional. If APC already sees byte-identical prompt IDs, Gain B is zero.

Expected regimes:

- **Decode-dominated:** short context and long outputs; this cache should not matter much.
- **APC-stable, TTFT-dominated:** long context, short outputs, many agentic turns; only render/tokenize remains as a possible win.
- **APC-unstable:** if re-rendering drifts, exact cached IDs can recover a large prefill win.

---

## Benchmark Method

This ADR records summarized benchmark results only. The prototype harness and raw CSV artifacts are
intentionally left out of this ADR-only change so the decision can be reviewed independently from
implementation code and diagnostic artifacts. The PR includes one summarized SVG graph for the
12-repetition paired prefix-handle comparison.

Server:

- URL: `http://10.0.0.99:8000`
- Model: `openai/gpt-oss-20b`
- Initial server configuration reported `max_model_len`: `16384`
- Long-window server configuration reported `max_model_len`: `131072`
- APC: enabled, verified by response `cached_tokens` and Prometheus prefix-cache metrics

Method:

1. Generate synthetic cumulative Responses inputs for turn indexes 1, 5, 10, 15, and 20.
2. Capture `prompt_token_ids` twice with `enable_response_messages=true`.
3. Verify the two captured prompt token sequences are identical.
4. Measure streaming TTFT and total latency for:
   - `full_stream`: normal Responses request, vLLM renders Harmony prompt
   - `token_replay_stream`: same request, plus captured `prompt_token_ids`
   - `token_replay_minimal_stream`: captured `prompt_token_ids`, but only the current user input in
     the request `input` payload
   - `prompt_cache_ref_stream`: full cumulative `input`, plus server-side prefix handle and marginal
     `append_token_ids`
   - `prompt_cache_ref_minimal_stream`: minimal current input, plus server-side prefix handle and
     marginal `append_token_ids`
5. Use 3 measured repetitions per profile after one warmup for the initial sweep.
6. Use 8 measured repetitions with shuffled mode order for targeted long-context runs.
7. Use 6 measured repetitions for the `prompt_cache_ref` long-window measurement.
8. Use 12 measured paired repetitions for the dense confidence-interval measurement.

This benchmark measures the vLLM primitive: full render/tokenize vs prompt-token replay. It does not
claim that agentic-api can already construct the next full prompt token IDs; that still requires
marginal render support or a vLLM-side incremental renderer.

Three measurement caveats matter:

1. The harness talks directly to vLLM. `agentic-api` storage and database queries are not in this
   latency path.
2. The initial replay requests still sent the full cumulative `input` payload and vLLM still ran
   `_construct_input_messages_with_harmony`. That measured only the incremental value of bypassing
   `render_for_completion(messages)` after full reconstruction had already happened.
3. The `token_replay_minimal_stream` mode measures a closer target architecture: vLLM still
   receives full prompt token IDs for generation, but it no longer receives or reconstructs the full
   conversation as JSON messages.
4. The `prompt_cache_ref_*` modes seed vLLM's in-memory handle registry once per profile, then exclude
   the seed request from p50 comparisons. They measure the intended steady-state shape where the client
   sends a prefix handle plus a small appended token suffix.

---

## Results

### ID stability

The same conversation rendered twice produced identical `prompt_token_ids` for all measured profiles.

Verdict: **Gain B was not observed on this server.** APC already hit almost the entire prompt during repeated requests.

### Standard synthetic 20-turn profile

`words_per_message=120`, `max_output_tokens=16`.

| Turn | Prompt tokens | Cached tokens | Full TTFT p50 ms | Replay TTFT p50 ms | Replay delta ms | Full total p50 ms | Replay total p50 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 211 | 208 | 124.4 | 127.2 | +2.7 | 396.0 | 401.1 |
| 5 | 1,427 | 1,424 | 129.9 | 147.8 | +17.9 | 394.0 | 424.2 |
| 10 | 2,947 | 2,944 | 170.1 | 181.3 | +11.2 | 435.7 | 451.5 |
| 15 | 4,467 | 4,464 | 150.2 | 177.1 | +26.9 | 430.6 | 452.7 |
| 20 | 5,987 | 5,984 | 181.4 | 187.1 | +5.7 | 458.4 | 459.7 |

### Large context sweep

`words_per_message=300`, `max_output_tokens=8`.

| Turn | Prompt tokens | Cached tokens | Full TTFT p50 ms | Replay TTFT p50 ms | Replay delta ms | Full total p50 ms | Replay total p50 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 6,652 | 6,640 | 175.4 | 176.4 | +1.0 | 269.4 | 271.4 |
| 15 | 10,122 | 10,112 | 186.2 | 208.2 | +22.1 | 281.3 | 298.5 |
| 20 | 13,592 | 13,584 | 214.1 | 212.3 | -1.8 | 309.8 | 313.4 |


### Minimal-input token replay

The initial replay mode was confounded because it sent both the full cumulative `input` and the full
`prompt_token_ids` array. That made the request larger than baseline and still forced vLLM to rebuild
the full Harmony message list.

The minimal-input replay mode sends only the current input message plus the captured full prompt token IDs.
This is closer to the intended production path where `agentic-api` owns the cached prefix.

Synthetic near-cap profile, `words_per_message=300`, `max_output_tokens=8`, 8 shuffled repetitions:

| Prompt tokens | Mode | Request bytes | TTFT p50 ms | Total p50 ms | TTFT vs full |
|---:|---|---:|---:|---:|---:|
| 13,592 | `full_stream` | 95,899 | 203.3 | 298.2 | baseline |
| 13,592 | `token_replay_stream` | 168,960 | 211.2 | 313.8 | +7.9 |
| 13,592 | `token_replay_minimal_stream` | 75,650 | 184.5 | 283.1 | -18.8 |

Codex-session fixture, local JSONL transcript capped by the current server length, 8 shuffled
repetitions:

| Fixture messages | Prompt tokens | Full TTFT p50 ms | Minimal replay TTFT p50 ms | TTFT delta | Full total p50 ms | Minimal total p50 ms |
|---:|---:|---:|---:|---:|---:|---:|
| 42 | 9,975 | 207.6 | 180.9 | -26.7 | 292.0 | 283.0 |
| 48 | 12,077 | 217.0 | 186.3 | -30.7 | 312.2 | 281.9 |
| 49 | 12,153 | 199.7 | 178.4 | -21.3 | 294.3 | 276.4 |

### Long-window token replay

With `max_model_len=131072`, the Codex-session fixture reaches 114k prompt tokens. APC remains hot:
cached tokens are almost equal to prompt tokens in every streaming request.

`max_output_tokens=4`, 8 shuffled repetitions per point:

| Prompt tokens | Full request bytes | Minimal replay bytes | Full-input replay bytes | Full TTFT p50 ms | Minimal replay TTFT p50 ms | Minimal delta | Full-input replay TTFT p50 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 33,194 | 137,581 | 161,116 | 298,450 | 255.2 | 247.8 | -7.4 | 255.8 |
| 64,551 | 266,513 | 310,773 | 577,039 | 354.0 | 287.4 | -66.6 | 355.8 |
| 102,879 | 437,682 | 498,360 | 935,795 | 401.2 | 381.6 | -19.6 | 397.8 |
| 114,457 | 482,436 | 554,364 | 1,036,553 | 437.0 | 372.0 | -65.0 | 418.5 |


### Prefix-handle replay

The `prompt_cache_ref + append_token_ids` path measures the intended server-side prefix-handle shape
with a 512-token marginal suffix. Each profile seeds the process-local vLLM prefix registry once, then
measures steady-state streaming requests. The seed request is included in the raw CSV but excluded
from the p50 rows below.

`max_output_tokens=4`, 6 shuffled repetitions per point:

| Prompt tokens | Full bytes | Handle minimal bytes | Token replay minimal bytes | Full TTFT p50 ms | Handle minimal TTFT p50 ms | Handle delta | Token replay minimal TTFT p50 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 33,194 | 137,581 | 3,188 | 161,116 | 279.1 | 189.1 | -90.0 | 243.1 |
| 85,181 | 360,909 | 3,220 | 411,154 | 397.1 | 241.5 | -155.6 | 341.1 |
| 109,098 | 460,806 | 3,185 | 528,411 | 457.6 | 272.4 | -185.1 | 389.8 |
| 114,457 | 482,436 | 3,289 | 554,364 | 504.7 | 266.4 | -238.3 | 390.8 |


This transport shape keeps the upstream request small. The minimal handle request body was about
3.2 KB at all four long-context points, while the full text request ranged from about 138 KB to 482 KB
and the full prompt-token replay request ranged from about 161 KB to 554 KB. The TTFT delta therefore
grew with context length instead of being eaten by JSON token-array overhead.

Initial growth-rate fit over the four p50 points:

| Series | Fitted slope | Interpretation |
|---|---:|---|
| Full cumulative context TTFT | about 25.9 ms / 10k prompt tokens | Cost of continuing to send and render the full conversation |
| Prefix-handle minimal TTFT | about 10.2 ms / 10k prompt tokens | Remaining scheduler, marginal prefill, network, and decode-start overhead |
| Saved TTFT delta | about 15.8 ms / 10k prompt tokens | Incremental benefit from the prefix-handle shape as context grows |

The fitted saved-delta line has `R^2 = 0.898`, so treat the exact slope as directional rather than
universal. The adjacent growth rates were about 12.6 ms / 10k tokens from 33k to 85k and about
12.3 ms / 10k tokens from 85k to 109k. The final 109k to 114k segment was much steeper because the
full cumulative p50 landed high, so it is useful as a positive long-context signal but not enough by
itself to extrapolate. The durable takeaway is that the cached-prefix handle reduced the observed
context-growth slope by roughly 60% in this run.

Dense confidence-interval measurement: same Codex fixture, 12 measured paired repetitions per point,
2 warmups, shuffled mode order. Rows below compare the two production-relevant shapes:
`full_stream` and `prompt_cache_ref_minimal_stream`. `CI95` is `1.96 * standard_error`.

| Prompt tokens | n | Full mean +/- CI95 ms | Handle mean +/- CI95 ms | Paired saved mean +/- CI95 ms |
|---:|---:|---:|---:|---:|
| 749 | 12 | 148.0 +/- 4.5 | 149.6 +/- 7.0 | -1.6 +/- 8.5 |
| 4,639 | 12 | 176.5 +/- 8.0 | 157.0 +/- 4.9 | 19.5 +/- 8.1 |
| 10,402 | 12 | 189.0 +/- 12.2 | 160.6 +/- 11.6 | 28.4 +/- 7.8 |
| 23,847 | 12 | 258.3 +/- 13.3 | 174.9 +/- 8.6 | 83.3 +/- 13.2 |
| 29,986 | 12 | 258.0 +/- 14.6 | 172.3 +/- 6.5 | 85.7 +/- 12.8 |
| 35,819 | 12 | 281.2 +/- 15.7 | 201.5 +/- 9.4 | 79.7 +/- 18.6 |
| 51,014 | 12 | 319.6 +/- 21.9 | 218.8 +/- 7.6 | 100.8 +/- 16.5 |
| 75,902 | 12 | 389.2 +/- 25.1 | 242.7 +/- 7.2 | 146.5 +/- 22.0 |
| 102,879 | 12 | 478.5 +/- 30.0 | 283.4 +/- 12.9 | 195.1 +/- 26.6 |
| 114,457 | 12 | 581.2 +/- 45.7 | 307.7 +/- 25.5 | 273.6 +/- 42.1 |

### TTFT by conversation size

![Line chart showing that cached prefix handles reduce time to first token compared with sending the full conversation as prompt size grows](results/response_token_cache_prompt_cache_ref_long_ttft_stdev_2026-06-18.svg)

The dense measurement shows the regime boundary more clearly. At about 750 prompt tokens the handle path is
not faster. At 4.6k and 10.4k tokens the win is modest but positive. From about 24k prompt tokens
onward, the confidence intervals separate clearly enough to treat the benefit as real for this
fixture. A linear fit over paired mean saved TTFT gives about 20.4 ms / 10k prompt tokens with
`R^2 = 0.953`. Fitting the two TTFT series separately gives about 34.0 ms / 10k tokens for full
cumulative context and about 13.6 ms / 10k tokens for minimal prefix-handle replay.

The interpretation is not only that minimal replay wins at long context. JSON-encoded token IDs are
larger than the original text request for this fixture. External replay over HTTP can therefore give
back part of the CPU win as transport and JSON parsing overhead. The stronger production design is a
server-side cached-prefix handle plus marginal token IDs, with full-render fallback when the
process-local handle is absent.

### Observations

1. APC is already highly effective for repeated prompts. Cached tokens were nearly equal to prompt tokens in streaming measurements.
2. Prompt-token replay with the full cumulative `input` payload did not produce a meaningful win and
   often made the request larger than baseline.
3. Minimal-input prompt-token replay did show a measurable TTFT win at 10k-13.6k prompt tokens,
   roughly 19-31 ms in the targeted runs.
4. With `max_model_len=131072`, minimal-input replay shows a larger but still noisy long-context win:
   about 7 ms at 33k, 67 ms at 64k, 20 ms at 103k, and 65 ms at 114k prompt tokens.
5. Prefix-handle minimal replay showed a larger long-context TTFT win. The dense confidence-interval
   measurement saw no reliable win at about 750 prompt tokens, modest wins at 4.6k and 10.4k, and clearer
   wins from about 24k onward. The fitted paired-mean benefit grew by about 20.4 ms per additional
   10k prompt tokens in that run.
6. Returning full token ID arrays over JSON is expensive. Non-streaming capture latencies ranged from about 428 ms at 211 prompt tokens to about 796 ms at 13.6k prompt tokens. This reinforces that token capture is diagnostic/prototype-grade, not a production hot path by itself.
7. At 33k-114k prompt tokens, token-ID replay request bodies were larger than the text-only full-render
   request bodies. This is a concrete reason to avoid a production design that sends the whole prompt
   token array over JSON every turn.
8. The observed decode tail was roughly 12-17 ms/token in these short-output runs, depending on output length and profile.

### Storage note

The vLLM benchmark bypasses `agentic-api`, so database latency cannot explain the vLLM-direct replay result.
As a quick order-of-magnitude check, a local SQLite Criterion run on the same branch produced:

```bash
BENCH_CONCURRENCY=1 cargo bench -p agentic-core --bench benches rehydrate \
  -- --sample-size=10 --warm-up-time=1 --measurement-time=2
```

| Benchmark | Local result |
|---|---:|
| `conversation_rehydrate` | p50 about 0.91 ms |
| `response_rehydrate` | p50 about 12.16 ms |
| `concurrent_conversation_rehydrate/1` | p50 about 37.7 us |
| `concurrent_response_rehydrate/1` | p50 about 72.4 us |
| `rehydrate_only/prev_response_depth/{1..5}` | p50 about 110-113 us |

These numbers are not production Postgres numbers and do not cover remote database latency. They do
suggest that storage lookup is a separate measurement axis, not the source of the vLLM-direct replay
result. A production report should add Postgres measurements for latest token-span lookup, span
persist, and full conversation rehydrate under realistic row sizes.

---

## Measured constants

These are preliminary and specific to the DGX server measured on 2026-06-18.

| Constant | Measured status | Value / interpretation |
|---|---|---|
| `R_per_tok` | Not cleanly measurable as pure render/tokenize | Full-input replay was confounded by larger JSON requests. Minimal prompt-token replay suggested the combined full-history transport/reconstruction/render residual was roughly 1.4-2.6 microseconds/token at 10k-13.6k tokens. Prefix-handle replay is a cleaner steady-state proxy: the dense confidence-interval measurement saved roughly -2 ms at 749 prompt tokens, 20-28 ms at 4.6k-10.4k, and 83-274 ms over 24k-114k prompt tokens. A linear fit over paired mean saved TTFT puts the growth rate at about 2.04 microseconds/token for this Codex fixture. |
| `prefill_per_tok` with APC on | Partially measured | Residual TTFT slope was roughly 6-10 microseconds/token across APC-hot streaming requests, but this includes scheduler/network overhead and is not pure GPU prefill. |
| `prefill_per_tok` with APC off | Not measured | Requires restarting vLLM with prefix caching disabled or isolating uncached prompts. |
| `decode_per_tok` | Approximate | Around 12-17 ms/token in the short-output profiles. |
| ID stability | Measured | Stable for all benchmarked conversations. |

---

## Verdict

There is a real speedup in the intended long-context agentic regime, provided the implementation uses a
server-side prefix handle plus a small marginal token suffix. The vLLM fork measurement exercises that
transport shape directly.

The vLLM primitive works: Responses can surface token IDs, accept exact prompt token IDs for Harmony
generation, and accept an in-memory `prompt_cache_ref + append_token_ids` replay request. The
agentic-api prototype can persist and rehydrate captured spans. On the measured server:

- APC already hits for essentially all repeated prompt tokens.
- Re-rendered Harmony prompt IDs were stable.
- Full-input prompt-token replay did not improve TTFT or total latency.
- Minimal-input prompt-token replay did improve TTFT in targeted near-cap and Codex-session fixture
  runs, including 33k-114k prompt-token runs with a 131k context window.
- Minimal-input prefix-handle replay improved TTFT much more clearly in the dense measurement: the benefit
  was negligible at about 750 prompt tokens, modest at 4.6k-10.4k, and about 83-274 ms over
  24k-114k prompt-token runs, with request bodies staying around 3.1-3.3 KB.

That means Gain B is zero for this deterministic server: APC already hits. Gain A is
real and becomes meaningful once replay avoids both full rendering and full-history
message reconstruction/transport.

The remaining concern is production state management. If `agentic-api` sends full prompt IDs over HTTP
every turn, the token array can be larger than the text prompt. The better primitive is:

- vLLM maintains a prefix-token registry and the client sends only a validated prefix handle plus the
  marginal new token span.

The measured fork primitive is still narrower than the final architecture: the registry is in-memory,
process-local, and must be reseeded after restart. A production cache path should add handle lifetime,
eviction, multi-worker routing semantics, and strict Harmony-boundary proof. The likely value of this
work is concentrated in:

- diagnostics for exact rendered prompt IDs
- correctness tests for determinism
- a foundation for a future incremental render path
- protection if future templates, reloads, or model-specific renderers introduce token drift
- long-context agentic loops where cumulative request construction and render/tokenize CPU become a
  visible residual after APC removes most GPU prefill

---

## Prototype replay contract

The prototype includes a typed representation of the production primitive, without wiring it into
the live executor yet.

`CachedTokenSpan::build_replay_plan` accepts a compatible `CacheKey` and a fresh full render of the
current prompt. It returns:

- `prompt_cache_ref`: a stable prefix reference containing a deterministic token-ID hash, prefix token
  count, and model/tokenizer/renderer/template/effective-instruction fingerprints.
- `append_token_ids`: only the marginal token suffix after the cached prefix.

The replay plan intentionally refuses unsafe cases:

- model, tokenizer, renderer, template, instruction, or tool fingerprints differ
- cached span does not start and end at a proven Harmony boundary
- persisted token count does not match the token ID array
- cached IDs are not a strict prefix of a fresh full render
- no marginal token suffix exists

`RequestPayload::to_upstream_request_with_replay_plan` serializes this as the intended future upstream
shape:

```json
{
  "model": "openai/gpt-oss-20b",
  "input": "...marginal input only...",
  "stream": true,
  "prompt_cache_ref": {
    "handle": "vllm_prefix_...",
    "prefix_hash": "sha256:...",
    "prefix_token_count": 64551,
    "model": "openai/gpt-oss-20b",
    "tokenizer_fingerprint": "...",
    "renderer": "harmony",
    "renderer_version": "...",
    "template_fingerprint": "...",
    "effective_instructions_hash": "...",
    "effective_tools_hash": "..."
  },
  "append_token_ids": [200006, 882, 200008]
}
```

This shape is implemented in the `responses-tokens-endpoint` fork, but it is not an upstream vLLM API.
The prototype stores prefix handles in memory on one server process. The executor should keep live
replay behind a flag until agentic-api can prove Harmony boundaries, validate strict prefixes, and
handle vLLM restart/cache-miss fallback.

---

## Scaled deployment with llm-d

The database-backed token cache and vLLM's KV prefix cache are different layers.

`agentic-api` can durably store rendered prompt token IDs, prefix hashes, and compatibility metadata.
That data is useful for deterministic reconstruction, strict-prefix validation, replay-plan building,
and reseeding after cache miss or restart. It does not by itself make KV tensors resident on every
vLLM pod.

In an llm-d deployment, fleet-level KV reuse is handled by the router and model-server cache layer:

- The llm-d Router's EPP parses OpenAI traffic, runs data producers, scores candidate endpoints, and
  picks the target pod through a filter-score-pick scheduling pipeline.
- Approximate prefix-cache routing keeps an in-memory history of recently routed prompt blocks. This
  is lightweight, but it is an EPP-local assumption and can diverge from the model server's actual
  cache state.
- Precise prefix-cache routing uses a `token-producer`, `precise-prefix-cache-producer`, and
  `prefix-cache-scorer`. The producer tokenizes prompts through a vLLM render endpoint, subscribes to
  per-pod KV events over ZMQ, and maintains a block-key index of which pods hold which prefix blocks.
- The precise scorer scores the longest consecutive cached prefix per candidate pod, then composes
  that signal with queue-depth, KV-utilization, and no-hit-LRU scorers. The guide configuration gives
  the prefix-cache scorer a higher weight than the load scorers, while still balancing against hot
  pods.
- In active-active EPP mode, each router replica subscribes to every model-server pod's KV-event
  stream via pod discovery, so each replica converges to the same view. Approximate local-prefix
  routing is not sufficient for active-active correctness because each EPP replica would otherwise
  learn a different partial history.
- Tiered prefix-cache deployments use vLLM `OffloadingConnector`, the llm-d filesystem backend, or
  LMCache-compatible connectors to extend cache capacity into CPU memory or shared storage. That helps
  idle agentic sessions resume without full prefill recomputation and can allow new pods to reuse
  cache from shared storage.

The production integration should therefore treat `prompt_cache_ref` as an endpoint-scoped execution
hint, not as a durable global database pointer. A safe reference needs at least:

```text
model
tokenizer_fingerprint
renderer_version
template_fingerprint
prefix_hash
prefix_token_count
block_size
serving_endpoint_identity or cache_epoch
```

The `serving_endpoint_identity` or `cache_epoch` part must be considered volatile. If the request is
routed to a different pod, the pod restarts, or the EPP observes `AllBlocksCleared`, the handle may be
invalid even though the database token span is still correct.

### Router-visible prefix identity

The current prototype request is efficient for one process because it sends only:

```text
prompt_cache_ref + append_token_ids
```

In llm-d, the router must also see enough prefix identity to route the request to the right pod before
vLLM evaluates the handle. There are three viable integration shapes:

1. **Router re-tokenizes/render-checks the request.**
   This matches llm-d precise prefix routing today, but only works if the EPP can reconstruct the same
   full Responses/Harmony prompt that vLLM will execute. For this ADR, that means vLLM needs a
   Responses render endpoint or the EPP token-producer must understand the agentic-api replay shape.

2. **agentic-api sends full prompt token IDs to the router.**
   This is simple and precise, but the benchmark shows that JSON token arrays are often larger than
   the original text request. It is useful for diagnostics and reseeding, not the desired steady-state
   hot path.

3. **agentic-api sends a compact prefix hint.**
   The preferred shape is a router-visible prefix identity derived from the cached token IDs, such as
   a prefix hash, token count, block size, and eventually a block-hash chain compatible with llm-d's
   precise KV index. The EPP can score pod locality from that compact hint, then forward only the
   endpoint-scoped `prompt_cache_ref + append_token_ids` to the selected vLLM pod.

The third shape keeps the request small and lets llm-d own fleet-level scheduling. It requires one new
contract between `agentic-api`, the EPP, and vLLM: the prefix hash or block-hash chain must be derived
from the same token stream and block size that vLLM uses for KV events. For the precise-prefix guide,
the vLLM `--block-size` and the router `tokenProcessorConfig.blockSize` must match; our replay plan
metadata should carry that value.

### Scaled request flow

The target scaled flow is:

1. `agentic-api` loads the latest validated `CachedTokenSpan` from the conversation store.
2. It builds a replay plan with a deterministic prefix hash, token count, block size, and marginal
   `append_token_ids`.
3. It sends the request through the llm-d Router with a trusted, router-visible prefix hint. End users
   should not be able to forge this hint.
4. The EPP scores candidate pods using precise prefix-cache routing when available, or approximate
   prefix/session affinity as a lower-precision fallback.
5. The selected vLLM pod validates or resolves the endpoint-scoped `prompt_cache_ref`, appends the
   marginal token IDs, and generates.
6. If the handle misses, the pod restarts, the prefix hash is absent from the EPP index, or the cache
   key is incompatible, `agentic-api` falls back to full render or full prompt-token seeding and then
   refreshes the stored token span.

### Metrics to add for scaled validation

The scaled benchmark should report both application-cache and llm-d routing signals:

| Layer | Metric |
|---|---|
| `agentic-api` | token-span lookup time, replay-plan build time, strict-prefix pass/fail, fallback reason |
| llm-d EPP | prefix-index hit ratio, matched prefix blocks, selected pod, scheduler latency, scorer weights |
| vLLM | prompt cached tokens, KV-event publication, handle hit/miss, cold-prefill fallback |
| KV offload | HBM/CPU/storage tier hit ratio, load time, offload bytes, eviction rate |

This keeps the production claim honest. A database token cache can remove render/tokenize work and make
the prompt identity deterministic. Preserving the prefill win at scale requires llm-d routing and/or
tiered KV offload so the selected serving endpoint actually has, can load, or can cheaply reconstruct
the matching KV blocks.

---

## Required production work before replay

1. Add Harmony boundary detection for captured spans and only mark `ends_at_boundary=true` when proven.
2. Add strict-prefix validation against a fresh full render before any replay path is enabled.
3. Decide where marginal rendering lives:
   - in agentic-api via a compatible Harmony renderer/tokenizer, or
   - in vLLM via an incremental Responses render API, or
   - in vLLM by accepting cached prefix IDs plus marginal messages.
4. Make the replay prefix router-visible in scaled deployments without sending the full token array.
   The preferred contract is a compact prefix hash or block-hash chain that llm-d can score against its
   KV index.
5. Scope `prompt_cache_ref` to the selected serving endpoint or cache epoch; never treat a process-local
   handle as durable database state.
6. Ensure vLLM KV events are emitted for the Responses replay path and that llm-d's token/block
   identity matches the replay-plan token stream and block size.
7. Avoid returning full `prompt_token_ids` on every production turn. Prefer the `prompt_cache_ref`
   handle path plus a marginal span, with a full-render fallback on cache miss.
8. Re-run benchmarks with:
   - APC disabled
   - cold prefixes
   - real agentic traffic profiles
   - longer contexts if the served model length allows it
   - server-side render/tokenize timing instrumentation
   - agentic-api storage microbenchmarks for latest-span lookup and span persistence
   - an agentic-api end-to-end path that sends `prompt_cache_ref + append_token_ids`
   - llm-d precise-prefix routing, active-active EPP, and tiered KV offload enabled
   - wrong-pod, pod-restart, `AllBlocksCleared`, and shared-storage reload scenarios

---

## Decision status

Keep ADR-04 in **Draft**.

The implementation is acceptable as a measurement and persistence prototype behind a flag. The measured
prefix-handle result justifies implementing the guarded agentic-api live replay path next.
