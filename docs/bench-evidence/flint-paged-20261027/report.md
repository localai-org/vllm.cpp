# Flint CPU paged engine — the first throughput numbers

Row: `MODEL-TEXT-FLINT` waves W-a/W-b/W-c/W-d (the paged-engine row)
Spec: `.agents/specs/flint-paged.md`
Issue: ISSUE-LOCAL-01M5W2QQBJ32HRB29G1KVQJS77
Branch: `row/FLINT-PAGED`, base `9cda5a603` (spec commit), measured at `92a217307`
Date: 2026-10-27

## Recipe

- Host: the shared aarch64 dev box (128 threads, 255 GB), 32 threads via
  `taskset -c 0-31`, `nice -n 5`, otherwise idle; the P150 belongs to another
  session and was not touched (CPU-only row).
- Build: `cmake -B build -G Ninja -DCMAKE_BUILD_TYPE=Release && ninja -j 32`
  at `92a217307`.
- Container: `/mnt/models/flint-GGUF/flint-Q4_K_M.gguf`,
  sha256 `20c89707f762af73f0840f7b2c2911193617e60cac2ae30a96cb2cda7c15927e`
  (the container the token gates were built against).
- Harness: `build/examples/vllm-bench` through the FULL engine stack
  (`LoadedEngine::FromModelDir` on the .gguf → InputProcessor → Scheduler →
  the paged forward arm → Sampler → OutputProcessor), asynchronous scheduling
  on (`max_concurrent_batches=2`), raw prompts (no chat template), greedy
  (temperature 0), ignore-eos, KV dtype auto. The KV pool auto-fitted the
  container's 262144-token context to 4064 tokens on the default 256-block
  pool (2 KV cache groups: full-attention + sliding-window).
- Raw reports: `c1-decode.txt`, `prefill512.txt`, `c4-serving.txt` beside
  this file, one invocation per leg.

## Results

| leg | shape | prefill (tok/s, in/TTFT) | decode (tok/s) | TTFT (ms) | TPOT (ms) |
|---|---|---:|---:|---:|---:|
| single stream | 1×8 in / 32 out, c=1 | 1.75 | **1.26** (per stream; 1.07 aggregate) | 5149 | 797 |
| prefill | 1×512 in / 1 out, c=1 | **7.44** | — | 68950 | — |
| small serving | 8×128 in / 32 out, c=4 | 1.84 | 0.97 aggregate, 0.49 per stream | 69620 | 2032 |

Request throughput on the serving leg: 0.03 req/s.

## Disposition

These are the FIRST CPU throughput numbers for Flint anywhere in this
repository, recorded per the row direction ("the FIRST Flint CPU throughput
numbers"). They are a **host-specific baseline, not a parity claim**: no vLLM
run of flint exists on any fleet device (vLLM registers no flint
architecture), and the llama-cpp-flint oracle serves single-sequence only, so
there is no like-for-like denominator on this shape. The correctness record
for the same container is the row's token gates, not this page.

Known shape of the gap to keep open (per AGENTS.md, no ceiling declared):

- decode TPOT 797 ms at batch 1 with 32 threads is dominated by the MoE
  expert paths running through the generic per-expert gather/SiLU/scatter
  CPU route; `layers::MlpGateUpMethodBase` / `vt::MergedGemmGroup` fusions
  and a grouped-GEMM expert path are the next traceable lever.
- prefill at 512 tokens (7.44 tok/s) runs the paged attention over the
  full window; the sliding-window layers only mask beyond 513, so the
  long-context prefill cost is real work, not a window bug.
- the c=4 leg's per-stream decode (0.49 tok/s) shows the batched path
  scaling WORSE than batch 1, which points at memory-bandwidth contention
  in the expert gather/scatter rather than at scheduling — the engine step
  count (28 steps for the W-c gate's three continuous requests) says the
  scheduler batches correctly.
