# Documentation-coverage audit — what shipped vs what the docs cover

**Claim:** `CLAIM-DOCS-COVERAGE` (coordination.md).
**Base:** `origin/main` @ `f887c34c1b10ad2d51125099cef55ef024b7732b` (1115 commits).
**Scope:** READ-ONLY analysis of code + git history vs `README.md` and
[docs/BENCHMARKS.md](../../docs/BENCHMARKS.md). No code changed. The only doc
edits made in the closing commit are outright-falsehood repairs (class B);
the README restructuring is a follow-up increment.

Gap classes, per the audit request:

- **A** IMPLEMENTED BUT UNDOCUMENTED — shipped and user-visible, absent from the README.
- **B** STALE / CONTRADICTORY DOC — the doc asserts something false about current state.
- **C** DOCUMENTED BUT UNUSABLE — status documented, no usage path / no CLI flag.
- **D** DOCUMENTED BUT NOT IMPLEMENTED / OVERSTATED — the README claims more than the code does.
- **E** UNDISCOVERABLE KNOBS — env vars / defaults that change behavior but appear nowhere in the docs.

---

## 1. Class D first — the truthfulness bugs

Two D findings. The first is correctness-relevant and is the most serious
finding in this audit.

### D1 (LOUD) — "KV offload to CPU / disk ... scheduler and worker sides wired" is FALSE for the disk/CPU tier

README Features row `KV offload to CPU / disk` says the connector seam has
"scheduler and worker sides wired", and the Serving-and-API-notes bullet says
"The worker-side GPU store/load is wired". Both are true **only for the LMCache
`lm://` connector**. For the CPU/disk `OffloadingConnector` the worker side does
not exist:

- `src/vllm/v1/worker/gpu/runner.cpp:1206-1209` — `GPUModelRunner::ConnectorLoadExternalKv()`
  opens with `dynamic_cast<LMCacheConnector*>(kv_connector_)` and returns
  immediately otherwise: `// a base/non-worker connector: nothing to load`.
- `src/vllm/v1/worker/gpu/runner.cpp:1267-1270` — `ConnectorStorePromptKv()` does the same.
- `src/vllm/v1/kv_offload/kv_connector.cpp:285-290` — `OffloadingConnector::build_connector_meta()`
  emits `ConnectorLoadJob`s. A repo-wide grep finds `ConnectorLoadJob` **only**
  in `kv_connector.*` and the lmcache files: nothing consumes the disk
  connector's jobs.
- `src/vllm/v1/kv_offload/kv_connector.cpp:305` — the source itself says
  `(the key set only; byte movement is the worker's)`.
- `include/vllm/v1/kv_offload/kv_connector.h:179-190` — the worker-side hooks
  (`register_kv_caches`, `start_load_kv`, `wait_for_layer_load`,
  `save_kv_layer`, `wait_for_save`) are no-op defaults; `OffloadingConnector`
  overrides none of them.
- The disk tier's byte home is a host `PrimaryByteView`
  (`include/vllm/v1/kv_offload/tiering_manager.h:129-132`); there is no
  GPU-page to CPU-tier copy anywhere.

**Why it matters beyond documentation.** The scheduler half DOES shortcut
prefill for externally-matched blocks (BENCHMARKS records "32/48 prefill tokens
saved, 2/3 blocks promoted from disk" for `KV-OFFLOAD` W4), and
`entrypoints/model_loader.cpp:109-140` will build and wire `OffloadingConnector`
for any caller that sets `EngineParams::kv_transfer_config` with
`kv_connector = "OffloadingConnector"`, with no device guard. Selecting it on a
GPU model would skip prefill for KV that is never written into the GPU blocks,
i.e. silently wrong output. The README currently reads as if that path were
complete.

**Fix.** (a) One README cell + one Serving-notes sentence: scope the
worker-side wiring claim to the LMCache connector and mark the CPU/disk
connector scheduler-side-only / not GPU-safe. (b) Separately (code, not this
increment): either implement the worker half for `OffloadingConnector` or make
`BuildKvConnector` refuse it on a non-CPU device. Recommend (b)-refuse first, it
is a five-line guard.

### D2 (soft) — "runs on CUDA, CPU, Metal, and Vulkan" reads as four working backends

README intro line 8 ("runs on CUDA, CPU, Metal, and Vulkan from one source
tree") and the Why-vllm.cpp bullet ("One source tree, many backends. CUDA, CPU,
Metal, and Vulkan from the same code") both list Vulkan as a run target. The
Acceleration table two screens down says the truth: Vulkan is a "Skeleton: 8 ops
plus the fusion catalogue ... **No model runs yet**", matching
`.agents/backend-matrix.md` (`BACKEND-VULKAN`, skeleton) and BENCHMARKS:1720.
Metal is likewise two models, 18 of 75 ops native. The lead sentence is not
false about the *source tree*, but a newcomer reads it as "four backends run
models".

**Fix.** One-line: "runs on CUDA and CPU, with Metal and Vulkan backends in
bring-up" (or append "(Metal partial, Vulkan skeleton)"). Pure README edit.

### Class-D checks that came back CLEAN

Stated explicitly so the absence of more D findings is not mistaken for
absence of checking:

| README claim checked | Verified against | Verdict |
|---|---|---|
| Supported-models table, all 13 rows | `.agents/model-matrix.md` architecture-support checklist (15 `✅` + 1 `🚧` engaged rows) | every listed row is backed; no model claimed above its row state |
| Phi-3 "15/16, not a clean pass" and OLMo-3 "oracle-blocked" | matrix `🚧 Phi3ForCausalLM`, `Olmo2/Olmo3` row | honestly stated, no over-claim |
| Quantization table (NVFP4 W4A4/W4A16, CT-NVFP4A16, GGUF 8 encodings, FP8 W8A8, MXFP4/MXFP8 planned) | `.agents/quantization-matrix.md` (`QUANT-NVFP4-MO-W4A16` DONE, `QUANT-NVFP4-CT-W4A4` DONE, `QUANT-NVFP4-CT-W4A16` ACTIVE, `QUANT-FP8-MO-STATIC` DONE, GGUF F32/F16/Q4_0/Q8_0/Q3_K/Q4_K/Q5_K/Q6_K PARTIAL, MXFP8/MXFP4 INVENTORIED) | exact match, including the "generic FP8 modes and FP8 KV remain open" caveat |
| Acceleration table (sm_121a runtime, sm_120a build-only, sm_90a build/portable-only, Metal partial, Vulkan skeleton, XPU HW-blocked, ROCm/ANE roadmap) | `.agents/backend-matrix.md` header + rows | exact match |
| C ABI "`VLLM_ABI_VERSION 5`, 19 exported symbols" | `include/vllm.h:57` (`#define VLLM_ABI_VERSION 5`), 19 `VLLM_API` declarations | exact |
| OpenAI endpoint table (5 routes) | `src/vllm/entrypoints/openai/api_server.cpp:311,315,319,323,327` | exact, no phantom endpoint |
| "Reasoning parsing, 7 parsers" | `src/vllm/entrypoints/openai/reasoning_parsers/` = basic, deepseek_r1, minimax_m2, mistral, olmo3, step3, think_auto | exact |
| `vllm-cli` flag table (9 flags) | `examples/cli/main.cpp` | exact, no invented flag |
| CMake options table | `CMakeLists.txt` `option()` / `set(... CACHE ...)` | every listed option exists with the stated default (one option is MISSING from the table, see A6, but nothing listed is fabricated) |
| "no `/metrics` endpoint yet", "Speculative decoding is not user-visible yet", "Multimodal / LoRA / multi-GPU not supported" | `SERVE-METRICS` INVENTORIED, `SPEC-MTP` GATING, `LORA-*`/`PAR-*` INVENTORIED | honest |

One count could not be exactly reproduced and is flagged low-priority rather
than as a D: the Features row says **"39 dialects"**, while
`src/vllm/entrypoints/openai/tool_parsers/abstract.cpp:94-252` has 36 factory
branches over 40 accepted names (aliases `llama4_json`, `qwen3_xml`, `mimo`,
`glm45`). The claim is not inflated relative to the code (36 families / 40
names), but the counting basis is not stated. See A9.

---

## 2. Prioritized gap table

Ordered by user impact. `→` marks the concrete fix.

### Class B — stale / contradictory doc (the worst class; ALL instances)

| # | Gap | Evidence | Impact | Fix |
|---|---|---|---|---|
| B1 | **`docs/BENCHMARKS.md:1722`: "External KV / LMCache \| NOT IMPLEMENTED / NOT BENCHMARKED \| Connector ABI and two-engine store/retrieve remain roadmap inventory"** — flatly contradicts the six LMCache rows W1-W5 in the SAME table (lines 1686-1691) and the shipped code | `87a7c15`, `88dcba7`, `26e6609`, `26a0fc0`, `94a706b`, `99a43c6`; `src/vllm/v1/kv_offload/lmcache/` | a reader of the open-gates table concludes the feature does not exist | → replace the row with the current disposition (built + output-invariance CLOSED, binding throughput number still PENDING). **Fixed in the closing commit.** |
| B2 | **README:74 "Gemma 1 / 2 / 4 remain spiked-only"** contradicts README:31-32 in the SAME document, which record Gemma-2 (48/48 near-tie-band) and Gemma-1 (STRICT 48/48) as Correctness-complete, and contradicts `.agents/model-matrix.md` (`GemmaForCausalLM` ✅, `Gemma2ForCausalLM` ✅) | landed by `a60b88b`; the Features table was updated, the Supported-models prose was not | the README calls its own shipped models unimplemented | → rewrite that sentence to "Gemma 1 / 2 / 3 are landed; Gemma 4 remains blocked". **Fixed in the closing commit.** |
| B3 | **README:287 "Tool calling uses a Hermes-style / Qwen3 parser subset; the Qwen3-Coder XML forced-reasoning template is not fully implemented"** contradicts README:45 ("39 dialects, streaming, every vLLM tool parser at the pin except the three Rust/Harmony-backed ones") and the code, which has `qwen3_coder` / `qwen3_xml` / `mimo` | `src/vllm/entrypoints/openai/tool_parsers/qwen3_coder.cpp`; `abstract.cpp:174`; waves `b846f2a`, `da93382`, `954a3c5`, `cbcc612`, `df8909b` | understates the largest single feature the project shipped | → delete/replace the bullet. **Fixed in the closing commit.** |
| B4 | README:74 "`Olmo3ForCausalLM` rides the same class ...; the Olmo-3 interleaved sliding-window path is a **follow-on**" — it landed in `4e9f1d2` and has its own row in both README tables | `include/vllm/model_executor/models/olmo2.h`, `tests/.../test_olmo3_paged_engine.cpp` | mild; contradicted by the same README's own tables | → one-word edit ("landed, oracle-blocked"). Fixed in the closing commit. |
| B5 | `.agents/engine-matrix.md:171` `SERVE-C-ABI` still says "**17 exported symbols**"; the ABI has been 19 since `c44c1f8`/`eb9d129` (ABI v4/v5 added `tool_parser`/`reasoning_parser` and the chat entry points) | `include/vllm.h` = 19 `VLLM_API` symbols; README:231 correctly says 19 | internal record only, but it is the row a porter reads | → matrix-row edit. **FIXED 2026-07-24 by `CLAIM-DOCS-T2-FIXES`** (now reads 19 with the ABI-v4/v5 provenance) |

### Class C — documented but unusable (no usage path)

| # | Gap | Evidence | Impact | Fix |
|---|---|---|---|---|
| C1 | **LMCache / KV-transfer has NO CLI flag and NO env selector.** `--kv-transfer-config` exists nowhere in the tree. The only way to enable any connector is programmatic: `EngineParams::kv_transfer_config` | `include/vllm/entrypoints/model_loader.h:76`; `src/vllm/entrypoints/model_loader.cpp:109-140`; grep `kv-transfer-config` = 0 hits in `examples/`, `src/` | the whole W1-W5 LMCache campaign is unreachable from `server` and `vllm-cli` | → **LANDED 2026-07-24 (`CLAIM-DOCS-T2-FIXES`)**: `--kv-transfer-config '<json>'` on the server (`vllm::ParseKVTransferConfigJson`, vLLM's flag name + JSON shape, malformed/unknown-key/unknown-role/unknown-connector all loud at startup) plus [docs/KV-OFFLOAD.md](../../docs/KV-OFFLOAD.md). |
| C2 | **The OpenAI server hardcodes the tool parser to `"hermes"` and disables reasoning parsing**, with no `--tool-call-parser` / `--reasoning-parser` flag (vLLM has both) | `examples/server/main.cpp:288-289`: `OpenAIServingChat chat(..., "hermes", /*reasoning_parser_name=*/"", ...)` | 39 tool dialects + 7 reasoning parsers ship but only 1 dialect and 0 reasoning parsers are reachable over HTTP. Serving any Qwen3-Coder / DeepSeek-R1 / Mistral / GLM model through the server silently uses the wrong parser | → **LANDED 2026-07-24 (`CLAIM-DOCS-T2-FIXES`)**: `--tool-call-parser` / `--reasoning-parser`. NOTE a deliberate deviation from this recommendation — the flags do NOT default to auto-detection, because the marker table would then silently change the parser of every existing deployment (26 tool rows, 2 reasoning rows would fire). They default to the previous hardcode (`hermes` / disabled) and take `auto` to opt into the C ABI's detection. |
| C3 | **`vllm-cli` cannot reach structured output, stop strings, penalties, min-p, min-tokens or ignore-eos** although `vllm_sampling_params` exposes all of them (ABI v2) | `include/vllm.h:139-180`; `examples/cli/main.cpp` has only model/prompt/max-tokens/temperature/top-p/top-k/seed/stream | the flagship "Structured output: Supported (subset)" feature has no CLI demo | → add `--json-schema`, `--gbnf`, `--regex`, `--stop`, `--min-p`, `--repetition-penalty`, `--min-tokens`, `--ignore-eos`. Small. |
| C4 | **KV offload to CPU/disk is "Built, opt-in"** but, beyond C1, has no user-facing story at all: no doc for the `kv_connector_extra_config` keys the disk tier reads (`root_dir`, byte budget, `offload_block_tokens`, `offload_prompt_only`) | `src/vllm/v1/kv_offload/kv_connector.cpp:93-170` (`extra_int`/`extra_bool` readers) | the feature is undocumented-by-construction even for a programmatic caller | → **LANDED 2026-07-24**: [docs/KV-OFFLOAD.md](../../docs/KV-OFFLOAD.md), including the disk tier's `root_dir` / `num_cpu_blocks` / `offload_block_tokens` / `offload_prompt_only` keys and the refusal semantics. |

### Class A — implemented but undocumented

| # | Gap | Evidence | Impact | Fix |
|---|---|---|---|---|
| A1 | **The OpenAI server accepts ~25 request fields the README never mentions**: `n`, `top_k`, `min_p`, `repetition_penalty`, `presence_penalty`, `frequency_penalty`, `seed`, `stop`, `stop_token_ids`, `stream_options` (`include_usage`, `continuous_usage_stats`), `logprobs`, `top_logprobs`, `prompt_logprobs`, `echo`, `min_tokens`, `ignore_eos`, `include_stop_str_in_output`, `skip_special_tokens`, `spaces_between_special_tokens`, `priority`, `response_format`, `tools`, `tool_choice`, `max_completion_tokens` | `include/vllm/entrypoints/openai/protocol.h:181-360`; parsed at `src/vllm/entrypoints/openai/protocol.cpp:224-355` | users assume only the 4 fields shown in the curl example work | → one "Supported request fields" table in the OpenAI-server section (with the parsed-but-deferred ones, e.g. `echo`, `parallel_tool_calls`, marked) |
| A2 | **Three server flags are undocumented**: `--cuda-profile-graph-replays`, `--cuda-profile-graph-batch`, `--benchmark-shutdown-fifo` | `examples/server/main.cpp` | maintainer knobs leaking into the production binary | → one README row each, or mark them "diagnostic" |
| A3 | **Three `vllm-bench` flags are undocumented**: `--seed`, `--temperature`, `--output-token-ids` (README names 8 of the 11) | `examples/bench/main.cpp` | reproduction recipes cannot be written from the README | → extend the one-line bench sentence |
| A4 | **Three example binaries are undocumented**: `dump_container`, `dequant_nvfp4`, `quant-gemm-bench` all build by default; README names only `vllm-cli`, `server`, `vllm-bench`, `tokenize` | `examples/CMakeLists.txt:1,9,30` | minor | → extend the binaries sentence, or note they are dev tools |
| A5 | **`priority` request field + `--scheduling-policy priority`** — the flag is documented, but not that requests must then carry a `priority` field for it to do anything | `include/vllm/entrypoints/openai/protocol.h:216,337`; `ENG-PRIORITY-SCHED` | the documented flag looks like a no-op | → one clause on the existing flag row |
| A6 | **CMake option `VLLM_CPP_FLASH_ATTN` (default `ON`)** is absent from the README's CMake-options table (15 rows, this is the 16th) | `CMakeLists.txt:981` | users cannot turn off the vendored FA2 build | → one table row |
| A7 | **Chat over the C ABI / server supports GGUF-embedded chat templates** (`tokenizer.chat_template` GGUF metadata) — mentioned once in the dense C-ABI paragraph, absent from the Tokenizers/GGUF rows | `f46db5f` (vendored minja), `aaed7ec` | discoverability | → one clause |
| A8 | **Prefix-cache hit-rate statistics are readable** (`LLMEngine::prefix_cache_metrics()`, `Scheduler`, `EngineCore`), landed in `a2112b2`; README says only "Hit-rate statistics are counted per vLLM's own counters" and "there is no `/metrics` endpoint yet" — it never says how a library consumer reads them | `src/vllm/v1/metrics/stats.*`; BENCHMARKS:1700 | library consumers cannot find the API | → one clause in the Consuming-as-a-library section |
| A9 | **"39 dialects" has no stated basis** (36 parser families / 40 accepted names) | `src/vllm/entrypoints/openai/tool_parsers/abstract.cpp:94-252` | low; risks reading as a rounded marketing number | → state the basis ("36 parser families, 40 accepted names") |

### Class E — undiscoverable knobs

The headline number: **153 distinct `VT_*` / `VLLM_*` environment variables are
read from `src/` and `include/`. Exactly ONE (`VT_GGUF_KEEP_QUANT`) appears in
the README.** 58 appear somewhere in `docs/BENCHMARKS.md`, but as forensic
A/B evidence inside prose, not as a reference a user can act on.

Most are kernel-selection knobs that legitimately belong in the record rather
than the README. The subset below changes user-visible behavior and has no
other surface:

| # | Env var | What it does | Class |
|---|---|---|---|
| E1 | `VT_LMCACHE_HOST` / `VT_LMCACHE_PORT` | **the only non-programmatic way to point the LMCache client at a server** (`src/vllm/v1/kv_offload/lmcache/remote_client.cpp:25`) | the de-facto config surface for C1, undocumented |
| E2 | `VLLM_CPP_CPU_THREADS` | overrides the CPU threadpool width (`src/vt/cpu/cpu_threadpool.cpp:50`) | the single most useful CPU-deployment knob; no CLI flag |
| E3 | `VLLM_PREFIX_CACHING_HASH_SEED` | block-hash seed; `=random` makes hashes non-deterministic across processes and takes any persisted KV cache to a 0% hit rate (`src/vllm/v1/core/kv_cache_utils.cpp:351`). Mirrors vLLM's `PYTHONHASHSEED` | correctness-of-caching knob |
| E4 | `VT_ASYNC_RUNNER`, `VT_ASYNC_SCHED` | **default ON**; the rollback for async/overlap scheduling (`src/vllm/v1/worker/gpu/runner.cpp:80`, `src/vllm/config/scheduler.cpp:13`) | the documented first-line workaround for any scheduling bug is undiscoverable |
| E5 | `VT_GGUF_KEEP_F16` | **default ON since L7** (`1a50ebe`); README documents its sibling `VT_GGUF_KEEP_QUANT` but not this one | RSS/perf behavior flip |
| E6 | `VT_CPU_REF` | forces the portable reference path (dequantize-everything oracle) | the standard "is this a kernel bug?" bisect switch |
| E7 | `VLLM_CPP_CUDAGRAPH`, `VT_DEVICE_KV_CACHE`, `VT_GPU_SAMPLE`, `VT_GDN_PACKED_DECODE`, `VT_CONV_REG`, `VT_FA2_*` | default-ON CUDA fast paths with per-feature rollbacks | every one is a supported bisect lever with no user-facing name |
| E8 | `VT_VULKAN_DEVICE` | forces the Vulkan physical device (`src/vt/vulkan/vulkan_context.cpp:160`) | required on multi-GPU hosts |
| E9 | `VLLM_CPP_HTTP_FIXED_POOL` | switches the server HTTP worker pool between capacity-fixed and legacy-dynamic (`examples/server/main.cpp:292`) | serving behavior, set in the example binary itself |

**Fix.** A single new `docs/ENVIRONMENT.md` reference with three sections
(deployment knobs, rollback/bisect switches, maintainer/diagnostic), generated
from the grep and kept honest by a small CI check that every `getenv` name in
`src/`+`include/` is either listed or explicitly on a "kernel-internal"
allowlist. That converts a 153-variable blind spot into one reviewable file.

---

## 3. Doc-internal consistency checks

| Check | Result |
|---|---|
| README Supported-models table vs `.agents/model-matrix.md` checklist | **MISMATCH**: the matrix has 15 `✅` engaged architectures; the README table has 13 rows and **omits Gemma-1, Gemma-2 and Gemma-3 entirely** (they exist only in the Features table and in the prose that calls two of them "spiked-only"). See B2. |
| README Acceleration table vs `.agents/backend-matrix.md` | consistent |
| README Quantization table vs `.agents/quantization-matrix.md` | consistent |
| README C ABI version vs `include/vllm.h` | consistent (v5, 19 symbols) |
| README endpoint table vs `api_server.cpp` | consistent (5 routes) |
| `.agents/engine-matrix.md` `SERVE-C-ABI` symbol count vs `include/vllm.h` | **MISMATCH** (17 vs 19). See B5. |
| README CMake table vs `CMakeLists.txt` | 15/16 options; `VLLM_CPP_FLASH_ATTN` missing. See A6. |

---

## 4. Recommended fix plan, ordered by impact

**Tier 1 — falsehoods (one-line/one-cell README + BENCHMARKS edits).**
Done in this claim's closing commit: B1, B2, B3, B4, and the D1 scoping
sentence. These are the only edits that can be made without restructuring.

**Tier 2 — small code wiring (each independently shippable, each ~20-40 lines).**
1. C2 `--tool-call-parser` / `--reasoning-parser` on the server (highest
   user impact per line: unlocks 39 dialects + 7 reasoning parsers over HTTP).
   **LANDED 2026-07-24, `CLAIM-DOCS-T2-FIXES`.**
2. C1 `--kv-transfer-config` on the server, mirroring vLLM's own CLI.
   **LANDED 2026-07-24, `CLAIM-DOCS-T2-FIXES`.**
3. D1 device guard: refuse `OffloadingConnector` on a non-CPU device in
   `BuildKvConnector` until its worker half exists (safety, not ergonomics).
   **LANDED 2026-07-24, `CLAIM-DOCS-T2-FIXES` — and it refuses on EVERY device,
   not only GPU. See the correction below.**
4. C3 sampling/structured-output flags on `vllm-cli`. **STILL OPEN.**

### T2 fix-wave outcome (2026-07-24, `CLAIM-DOCS-T2-FIXES`)

Landed: Tier-2 items 1-3, plus `docs/KV-OFFLOAD.md` (Tier 4), the README flag
rows and the three missing Gemma model rows (Tier 3 subset), and B5
(`SERVE-C-ABI` 17 → 19). Not landed: Tier-2 item 4, the Tier-3 README structure
pass, `docs/ENVIRONMENT.md`.

**Correction to this audit's D1 fix recommendation.** The audit proposed
refusing the disk connector "on a non-CPU device". That is too narrow, and the
implementation deliberately does not follow it: the connector's worker half does
not exist on ANY device — the bytes are never copied into a KV page whether that
page is CUDA or host memory — so a CPU-admitting guard would have shipped the
same silent-corruption bug on the CPU backend. The guard therefore refuses the
connector everywhere the engine would wire it. The scheduler-level 32/48
prefill-shortcut e2e is unaffected: it drives the connector directly through
`KVConnectorFactory`, never through `BuildKvConnector`, and still passes.

**Second correction: the D1 hazard was latent, not live-and-silent.** The audit
described selecting `OffloadingConnector` on a GPU model as reaching wrong
output. In fact `BuildKvConnector` never populated `KVConnectorContext::
page_size_bytes` or `::identity`, so `OffloadingConnector::CreateFromConfig`
already threw first — with an unrelated precondition message
("page_size_bytes must be > 0") that named neither the real problem nor a
remedy. The hazard was one plumbing patch away from becoming real, and the
misleading error is itself a defect. The guard now runs BEFORE construction and
reports the actual problem, so this stops depending on an accident.

**Shape of the guard.** Not a `dynamic_cast` ladder: `KVConnector` gained
`supports_worker_transfer_on(vt::DeviceType)` (base default false, since the
base's worker hooks are no-ops), the registry gained a matching static
`KVConnectorWorkerTransferFn` registered beside the builder
(`REGISTER_KV_CONNECTOR_WITH_WORKER`) so capability can be queried by NAME
before anything is constructed, and `EnsureWorkerTransferSupported` is the one
refusal point. Admitting a future worker half is one override plus one
registration argument; no call site changes.

**What remains honestly unimplemented.** The disk connector's worker half. This
wave converts a silent-wrong-output path into a loud refusal; it does not make
the disk tier usable from a serving engine, and nothing here should be read as
claiming otherwise.

**Tier 3 — README additions (one increment, needs the structure pass).**
A1 request-field table, A2/A3/A4/A5/A6/A7/A8/A9 rows and clauses, plus the
missing Gemma rows in the Supported-models table.

**Tier 4 — new docs pages.**
1. `docs/ENVIRONMENT.md` — the env-var reference (E1-E9 + the classified tail),
   with the CI completeness check.
2. `docs/KV-OFFLOAD.md` — the KV-offload / LMCache usage guide that C1 and C4
   both need: how to run an `lm://` server, the `KVTransferConfig` fields, the
   `kv_connector_extra_config` keys, and the identity-refusal semantics.

**Tier 5 — record repair.** B5 (`SERVE-C-ABI` 17 → 19) by the next agent that
owns that row.

---

## 5. Audit coverage statement

**Checked exhaustively (every instance enumerated, machine-swept):**

- Every `--flag` string literal in `examples/` (all 7 example binaries).
- Every `VT_*` / `VLLM_*` environment name read from `src/` and `include/`
  (153 distinct), and each one's presence/absence in README and BENCHMARKS.
- Every `VLLM_API` symbol and `VLLM_ABI_VERSION` in `include/vllm.h`.
- Every HTTP route registered in `api_server.cpp`.
- Every field of `CompletionRequest` / `ChatCompletionRequest` in
  `include/vllm/entrypoints/openai/protocol.h`.
- Every `option()` / `set(... CACHE ...)` in `CMakeLists.txt`, and every
  `add_executable` in `examples/CMakeLists.txt`.
- Every tool-parser factory branch and every reasoning-parser source file.
- Every row of `.agents/model-matrix.md`'s architecture-support checklist,
  `.agents/quantization-matrix.md`, and `.agents/backend-matrix.md`'s
  platform/CUDA-target rows, against the matching README table.
- The full `## Current checkpoint` open-gates table in `docs/BENCHMARKS.md`
  (rows 1686-1722), row by row, for stale dispositions.

**Sampled, not exhaustive:**

- `git log` (1115 commits): the full subject list was swept and scope-histogrammed,
  and every `feat(` commit in the user-visible scopes (`capi`, `openai`,
  `entrypoints`, `serve`, `model`, `kv`, `gguf`, `quant`, `backend`,
  `tokenizer`, `parsers`, `structured-output`, `cuda-arch`) was read; `perf(`
  and `fix(` commits were sampled, not read individually.
- `.agents/engine-matrix.md`'s 87 rows were read as a state list; only the
  `DONE`/`ACTIVE`/`PARTIAL` rows relevant to a README claim were opened.
- Kernel-level env vars (the ~120 `VT_GDN_*` / `VT_FP4_*` / `VT_MOE_*` tail)
  were enumerated but individually classified only by name and read site, not
  by reading each kernel.
- `docs/BENCHMARKS.md` outside the checkpoint table was searched by pattern
  (`NOT IMPLEMENTED`, `not implemented`, `roadmap inventory`, `not supported`)
  rather than read in full; the file is 3250 lines.

**Not checked:** runtime behavior. This audit read source and records; nothing
was built or executed, so "implemented" here means "present and reachable in
source", grounded on the matrices' own gate evidence for correctness claims.
## Owed

- **`ISSUE-GH-242` — `docs/FEATURES.md` drifted from the registry.** The
  architecture counts read 30 against 35 in the registry, and
  multimodal-over-HTTP stays marked unshipped though W1-W3 landed. That is this
  audit's class B against a surface it covers.
- **`ISSUE-GH-652` — `.agents/model-matrix.md`'s five prose counters drifted
  when LTX-2.5 landed.** The model reached the rows and the CI-enforced rollup
  and none of the sentences that count models moved with it; record-coverage
  drift is the same class B shape as the FEATURES row above.
