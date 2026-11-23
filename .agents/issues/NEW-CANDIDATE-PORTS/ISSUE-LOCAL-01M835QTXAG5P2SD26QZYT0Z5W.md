ID: ISSUE-LOCAL-01M835QTXAG5P2SD26QZYT0Z5W
Title: Wire olala greedy decode through a production seam (registry forward or thin example)
Row: NEW-CANDIDATE-PORTS
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-23
Updated: 2026-11-23
Closed: 2026-11-23

## Problem

Reachability check for benchmark axis p150-olala-bf16 (spec .agents/specs/p150-benchmark-axes.md @ row/QWEN38-DFLASH2-STAGE-C-bench-plan, worktree /tmp/vllm-bench-plan) fails at the W2e head 048c6b22d. OlalaGenerateHost (src/vllm/model_executor/models/olala_ops.cpp:811) is reached only from tests/vllm/models/test_olala_ops.cpp:767,784,948. The registered production forward, ForwardOlalaForCausalLM in src/vllm/model_executor/models/olala_registry.cpp:76-85, is still the W2b refuse-by-name stub (VT_CHECK(false, OlalaForwardRefusalOps())), and no olala example, CLI, or server path exists. Per AGENTS.md section Nothing lands dead and the spec's identical-workload rules, a benchmark number measured through the test harness cannot be published, so the axis stays PENDING regardless of the token gates being EXACT (bf16, q6_k, q8_0-safe2 per docs/bench-evidence/new-candidate-ports-olala-w2e-20261122/). Wiring unit owed: port the ForwardOlalaForCausalLM body onto OlalaGenerateHost (or a block-forward equivalent, reusing the W2e arms and the KV spec already registered in MakeOlalaForCausalLMKVCache), or add a thin olala example like examples/kimi_linear_gen as the sanctioned seam, so decode is reachable from ModelRegistry::Forward, the loader, or a registered CLI path on default config. Only then can axis A1/A2/A8 numbers for p150-olala-bf16 be produced and published with the llama-cpp-olala @ bd20ca9b denominator.

## Resolution

2026-11-23, closed by the wiring commit on `row/NEW-CANDIDATE-PORTS-olala-bench`
(worktree /tmp/olala-bench-axis). Shape (a), the registry forward: the paged
fold NemotronH's A2-P took, chosen because every engine entry point
(`LoadedEngine::FromModelDir`, the vllm.h ABI, the server, the examples)
reaches decode only through `ModelRegistry::Forward` with runner-allocated
caches — a thin example (shape b) could not exist without this forward, and a
host-block example would not be a thin client of include/vllm.h.

- `ForwardOlalaForCausalLM` (olala_registry.cpp) now routes every step with
  runner caches to `OlalaPagedForward` (olala_paged.cpp, new) and refuses by
  name without them. The M layers' Mamba3-MIMO state scatters through the four
  f32 `GdnStateCache::states` rows; each V layer's sliding-window page carries
  a second head plane with the token's RAW pre-shift k/v, so the shift
  recurrence (modeling_olala.py:619) lives entirely in the paged cache. Pages
  and states are f32, annotated in MakeOlalaForCausalLMKVCache, so the engine
  path is bit-identical to the W2d EXACT host gate.
- The block-table page maps arrive through `MultiKvCacheIndex`
  (consumes_multi_kv topology); the recurrent slot through the GDN metadata's
  remapped state index. Refuses by name: batched decode, CUDA queue,
  sequences past `sliding_window`.
- Reachability evidence: `tests/parity/test_olala_paged_engine.cpp` drives
  `FromModelDir(kCPU)` on the pinned bf16 GGUF through the full engine stack
  (LLMEngine → GPUModelRunner::execute_model → ModelRegistry::Forward) and
  requires the 6 generated ids EXACT against both the llama-cpp-olala
  @ bd20ca9b bf16 golden and the in-process `OlalaGenerateHost` — 18/18
  assertions green in Release and bounds-checked Debug (both builds green:
  test_olala_ops 10/10, test_olala_gguf_weights 7/7). The token gates are
  unchanged (the shared per-token tails were extracted verbatim from the
  generate host). Axis p150-olala-bf16 can leave PENDING.
