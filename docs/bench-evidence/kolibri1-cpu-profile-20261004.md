# Kolibri-1 CPU forward profile (MODEL-TEXT-kolibri-1)

Date: 2026-10-06 (note slug dated 20261004 per task row assignment; all
measurements in this file were taken 2026-10-06).
Host: aarch64, 128 cores, 255 GB RAM, Linux. Worktree `/tmp/vllm-kolibri`,
branch `row/kolibri-cpu`, base 96fa752d6 ("route the residual add+RMSNorm
glue through vt::FusedChain").

## Method

`perf stat` attaches fine (one run, see the 128-thread pathology below), but
`perf record` cannot map its ring buffer in this sandbox: the kernel
`perf_event_mlock_kb` limit is 576 KB and the default per-CPU ring across
128 CPUs exceeds it at the minimum `-m 1`, so every `perf record` invocation
fails with "Permission error mapping pages" before the run starts. The
ranking below therefore comes from env-gated stage instrumentation in the
forward itself (`VT_KOLIBRI1_PROFILE=1`,
`src/vllm/model_executor/models/kolibri1_forward.cpp`, `prof::Scope`
accumulators; zero cost when unset). The gate workload is deterministic, so
cumulative stage seconds at equal forward counts are directly comparable
between two binaries; forward #k performs identical work in both.

Workload: `build/tests/test_kolibri1_w3` — the real checkpoint
(`/mnt/models/Aleph-Alpha/Kolibri-1`, fp8-block 128x128), 8 golden prompts,
greedy decode + teacher-forced fingerprint passes; the gate passed
900/900 assertions in the profiled baseline run.

## Thread-count pathology found first (the serial claim quantified)

The shipped default is `VLLM_CPP_CPU_THREADS = hardware_concurrency()` =
128. At the default, the W3 gate ran **3h43m wall** (`/usr/bin/time -v`),
10,686% CPU, **3.02e9 involuntary context switches** — the ggml-port pool
spins (poll=50) with 128 workers on a forward whose parallel grain is one
fp8 dequant call or one tiny GEMM, so the box thrashes. The "~10 of 128
cores, essentially serial" prior observation is the spin package, not the
compute. All numbers below use `VLLM_CPP_CPU_THREADS=8`, which the pool
contract makes bit-identical to any other count.

## Ranked table (loop → % of wall)

Baseline (threaded-pool GEMMs as shipped, dequant SERIAL), 8 threads,
cumulative after 30 forwards — total_forward 1981.42 s (66.0 s/forward):

| stage | scope | s | % of total_forward |
|---|---|---|---|
| routed+shared expert MLP container (gather/scatter loop, per-expert dequant, SwiGLU glue) | `moe_glue` | 1430.61 | 72.2% |
| attention + lm_head + router GEMMs (threaded MatmulBT) | `linear_gemm` | 36.82 | 1.9% |
| lm_head GEMV | `lm_head` | 3.52 | 0.18% |
| RoPE (sliding layers) | `attn_rope` | 0.86 | 0.04% |
| KV write + paged attention | `attn_core` | 0.71 | 0.04% |
| RMSNorms / FusedChain | `norms` | 0.37 | 0.02% |
| router top-k selection | `moe_topk` | 0.09 | <0.01% |
| router logits D2H | `moe_d2h` | 0.01 | <0.01% |
| (unattributed: allocations, residency, embedding, combine) | | ~508 | ~26% |

Verified expectation, with one correction: routed-expert work dominates
(≈72%), but the mass inside `moe_glue` is the **fp8-block dequant**, not the
GEMVs (which are already threaded through the pool and cost 1.9%). The
dequant-at-load disposition re-dequants every used weight PER CALL: ~7
experts/layer × 3 matrices × 50 layers ≈ 1.4 G elements per decode step,
serial scalar `ldexp`-based conversion at ~28 M elem/s. The expected
ranking "experts ≫ dequant ≈ norms ≈ attention" is wrong on the middle
terms: dequant ≫ everything else inside the expert container, and the
norms are noise (0.02%).

## Phase 2 — thread the ranked-#1 loop (landed)

`DequantFp8Block` now partitions its OUTPUT rows through the one pool
(`host_parallel::ForOutputRows`, the `#1664` adapter; row = unit of work,
never a K-chunk). Elementwise output partitioning is bit-identical to the
serial loop BY CONSTRUCTION (pool determinism contract), so the 2.5-nat
near-tie band is not even engaged.

Why per-expert loop threading was NOT used: the expert loop's GEMMs already
dispatch through the same pool; wrapping the loop in a second pool kick
would nest `Threadpool::Run` inside a pool worker (ggml-port pools are not
reentrant — deadlock). Row-parallel dequant removes the same #1 stage
without nesting.

After (8 threads, cumulative at 30 forwards) — total_forward **312.99 s**
vs 1981.42 s baseline at the same forward count: **6.33×** on the forward
pass. The full gate: **3h43m (128-thread default, baseline binary) and an
extrapolated ≈3.6h at 8 threads on the baseline binary → 34:14 wall with
threading on** (`/usr/bin/time -v`; forward time is ~96% of the gate wall,
and the baseline binary ran 6.33× slower at the same forward index, so the
whole-gate ratio ≈ 6.2×).

CAVEAT on the post-change per-stage split: the scopes nest (dequant runs
inside the `moe_glue` container AND inside the attention linears), so the
listed stage seconds double-count inner stages and their sum exceeds
`total_forward`; `total_forward` itself is a direct entry/exit measurement
and is exact. At the 130-forward checkpoint the threaded dequant accounts
for 1539 s of the 1763 s total — it remains the dominant stage and the next
lever (a bit-exact NEON fp8 decode — the scalar `ldexp` conversion, not the
memory traffic, is the cost — or a bounded dequant cache is owed; Phase 3
NEON was not reached in this session and stays owed on the row).

## Numerics evidence

- `test_kolibri1_w2` (tiny hermetic model): new case "threaded pool is
  bit-identical to the serial pool" — the same full-depth fixture forward
  under a 1-thread pool and an 8-thread pool, every logit bitwise equal.
  7/7 cases, 1608/1608 assertions.
- `test_kolibri1_w3` (real checkpoint, 900/900, threaded build, 8 threads):
  argmax chain **141/145 compared positions match the golden greedy decode,
  4 flips, ALL 4 near-tie, 0 hard** — byte-for-byte the landed serial
  result (same flips: gaps 0.0093/0.262/0.290/0.382, all inside the 2.5
  band). Threading moved no token, no fingerprint.

## Gates (post-change)

- W3 `test_kolibri1_w3`: 900/900 assertions, argmax chain 141/145 with the
  same 4 near-tie adjudications as the landed gate (identical to serial).
- W1 `test_kolibri1`: 27/27 cases, 186/186 assertions.
- `scripts/check-agent-record.py`: exit 0.

## Phase 3 — bit-exact NEON fp8 decode (landed 2026-10-06, same worktree,
branch `row/kolibri-perf-rebase`, HEAD 432b98b54 + this change)

The kernel is `src/vllm/model_executor/models/kolibri1_fp8_dequant.h`: the
scalar `vt::F8E4M3ToF32 x scale_inv` composition is rewritten as an exact
closed form on the f32 bit pattern (`(e+120)<<23 | m<<20` for normals,
`cvt(m)*2^-9` for subnormals, `+qNaN` with the sign suppressed for
0x7F/0xFF) and vectorized 16 bytes per `vld1q_u8` step, then the f32 lanes
are multiplied by the (block-constant) scale and stored bf16 with a
bit-identical port of `vt::F32ToBF16` (round-to-nearest-even incl. the
carry into the exponent, NaN truncate-to-quiet). Chunks never cross a
`block_k` boundary, so the scale is one `vdupq_n_f32` per chunk. The
non-aarch64 build keeps the scalar loop (the x86 CI path compiles it with
`-U__aarch64__` checked).

Bitwise identity is gated exhaustively, not sampled:
`tests/vllm/models/test_kolibri1_dequant.cpp` compares raw uint16 bf16
patterns — ALL 256 e4m3fn bytes x 6 scales (1.0, 2^-14 subnormal-range,
3.0e34, -1.5, 0.0, 2^-9) through the full row driver; a ragged-block run
(5x45, block 2x16, odd-K tail) against the scalar reference; and pool-style
partial row partitions. A deliberate sign-bit mutation of the kernel turns
the suite red (mutation check), restored to green byte-for-byte.

Same-host A/B, both binaries built from this tree (scalar = stash of this
change), `VLLM_CPP_CPU_THREADS=8`, `VT_KOLIBRI1_PROFILE=1`, the full W3
gate, `/usr/bin/time -v`:

| axis | scalar (HEAD) | NEON | multiple |
|---|---|---|---|
| `dequant_fp8_block` at 150 forwards | 1714.31 s | 248.28 s | **6.91x** |
| `total_forward` at 150 forwards | 1972.85 s | 513.43 s | 3.84x |
| whole-gate wall | 33:59.36 | 9:05.61 | 3.74x |

The conversion-bound reading was right: one NEON multiply-add-free
shuffle-free pass removes ~1466 s of 150-forward wall; the dequant stage
falls from 87% of `total_forward` to 48%, and `linear_gemm` (222.6 s) is
now the co-dominant stage — the next lever is a bounded dequant cache (the
re-dequant-per-call disposition), not more decode micro-optimization.

## Gates (post-NEON)

- `test_kolibri1_dequant`: 3/3 cases, 8/8 assertions (bitwise, exhaustive
  over the e4m3fn byte domain).
- W3 `test_kolibri1_w3` (NEON build): 900/900 assertions, argmax chain
  141/145 with the SAME 4 near-tie flips and the SAME fingerprints as the
  scalar binary (worst topk logit diff 2.18646 both sides) — the NEON path
  moved no bit that reaches the gate.
- W2 `test_kolibri1_w2`: 7/7 cases, 1608/1608 assertions. W1
  `test_kolibri1`: 27/27 cases, 186/186 assertions.
- `scripts/check-agent-record.py`: exit 0.
