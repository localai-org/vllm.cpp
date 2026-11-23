# olala bf16 CPU-baseline A/B — identical-workload greedy battery, 2026-11-23

Axis `p150-olala-bf16` (spec `.agents/specs/p150-benchmark-axes.md`, axes
A1-A3+A8). This run is the **host-ops baseline leg**: both arms on CPU. The
P150 device/traced cells of the axis remain PENDING (no traced device path for
olala exists yet).

## Identical workload

- Model: `/mnt/models/olala-v1-mini/Olala-v1.0-mini-7A1B-bf16.gguf`
  (OVHaiLLM/Olala-v1.0-mini-7A1B bf16 GGUF), sha256
  `b266da8e292fb502b161ef0af413242f40c3d5736ebbcb5153fbf9c16bef5747`
  (`model-sha256.txt`). The SAME artifact both arms.
- Prompt: token ids `[151644, 872, 198, 15588, 220, 151645]`. Our side passes
  the ids directly; the fork side tokenizes
  `'<|im_start|>user\n hi <|im_end|>'` with special-token parsing and no BOS —
  ids verified identical (`llama-tokenize` and the golden harness stderr
  `prompt tokens (6): 151644 872 198 15588 220 151645`).
- Greedy (argmax, temp 0), 64 generated tokens, batch 1, no BOS, context 512.

## Arms

| | vllm.cpp | llama.cpp fork (OVHai) |
|---|---|---|
| Revision | `row/NEW-CANDIDATE-PORTS-olala-bench` @ `6ce35000c` (+ `examples/olala_bench/main.cpp`) | `bd20ca9b` |
| Build | Release, `VLLM_CPP_TENSTORRENT=OFF`, ccache; `/tmp/bench-build` | `/tmp/olala-ref/build` (ggml `0.25.3`) |
| Path | `LoadedEngine::FromModelDir(kCPU)` → LLMEngine → `ModelRegistry::Forward` → `OlalaPagedForward` (production) | scratch `golden-bench.cpp` (this dir): raw ids, greedy argmax, 1 token/batch decode |
| Threads | engine default (not printed) | ggml default (not printed) |

Host: Ampere-1a, 128 threads, Linux, idle-normalized load 1.3-4.2 of 128
(<0.03 per-core) at both legs' measurement times; sequential runs.
Timestamps: vllm.cpp leg 10:46:00-10:55:31+01:00; fork legs 11:01:43-11:03:26+01:00.

## Numbers (3 legs, median + spread)

| Metric | vllm.cpp | llama.cpp fork | Ratio (fork/ours) |
|---|---|---|---|
| Prompt-eval (6-token prefill) | 0.4579 tok/s (0.4530-0.4580) | 2.615 tok/s (2.613-2.616) | 5.71x |
| Decode (64 tokens) | 0.3990 tok/s (0.3980-0.3991) | 2.158 tok/s (2.158-2.160) | 5.41x |

Notes: our prompt-eval is wall-time of the `max_tokens=1` generate call (full
prefill + one decode step); decode t/s = 63 / (total − prefill). The fork's
prompt-eval includes its first graph build (2294 ms / 6 tokens).

## Greedy-stream divergence

The first 58 generated tokens (indices 0-57) are IDENTICAL to the fork's
greedy stream, including the full 12-id W2d golden prefix. Divergence at index
58: fork emits `576` where we emit `2055`, both plausible continuations of the
same context. This is floating-point path drift in greedy argmax, not a
correctness regression: the binding token gate (test_olala_paged_engine.cpp,
18/18 EXACT) covers the compared prefix.

## Files

- `vllm-olala-bench-stdout.log` — our 3 legs, per-leg timings + full 64-id stream
- `gb-{1,2,3}.err` — fork harness stderr (prompt ids, TIMING lines, model metadata)
- `golden-bench.cpp` — the fork-side timed harness (scratch, not part of vllm.cpp)
- `llama-simple-contaminated-19tok.err` — the rejected `llama-simple` leg
  (its non-flags polluted the prompt to 19 tokens; superseded by golden-bench)
- `model-sha256.txt`

## Next traceable hypotheses (the 5.4x decode gap — no ceiling declared)

1. Scalar f32 GEMV/GEMM in `OlalaPagedForward` host ops vs ggml's vectorized
   blocked kernels (NEON/SVE/FMA dispatch) — profile per-op share first.
2. Per-token op-dispatch overhead: our engine re-plans the chain per decode
   step; measure fixed per-step cost vs compute (the 6-token prefill at
   13.1 s ≈ 0.458 tok/s suggests a large constant per-call cost).
3. No bf16-native vectorization on the host path (widen-to-f32 loads) —
   compare with the BFDOT/BFMLA verdicts recorded for kolibri1.
4. ggml uses a threadpool across the GEMMs; confirm whether our host path
   parallelizes the same GEMV shapes (thread count unlogged on both sides —
   log it on the re-take).
