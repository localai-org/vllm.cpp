# FLINT-NEON-Q4K-DOT — the NEON `vec_dot_q4_K_q8_K` for the keep-quant routed phase

Row: `FLINT-NEON-Q4K-DOT` · spec: `.agents/specs/flint-neon-q4k-dot.md` ·
issue: `ISSUE-LOCAL-01M697KEDFRNMRWDZ240NQAE0D` · date 2026-10-31 ·
worktree `/tmp/flint-neon-q4k`, branch `row/FLINT-NEON-Q4K-DOT`, kernel commit
`273bdb9d1`. Host: shared aarch64 dev box, 128 CPUs, governor `performance`
(as found). Builds on CPUs 64-127 (`-j 64`), bench legs on 32-63/32-47.

## What landed

`src/vt/cpu/cpu_quant_dot_q4k_neon.cpp` — the Q4_K x Q8_K keep-quant inner dot
on NEON DotProd (`vdotq_s32`), in the house `cpu_quant_dot_sdot.cpp` shape:
one TU compiled `-march=armv8.2-a+dotprod+fp16`, runtime-gated on the shared
HWCAP dotprod probe, selected by `VT_CPU_Q4K_DOT=auto|portable|neon`
(default `auto`), dispatched from `BlockVecDot(DType::kQ4_K)` — the exact seam
the serving GEMM reaches. `VT_CPU_Q4K_DOT=portable` is the same-binary A/B
rollback.

## Bit-identity methodology and results (the row's core gate)

The contract: the NEON kernel must be BIT-IDENTICAL to the scalar reference on
EVERY input. The reference is `QuantQ4KPortableVecDot()` — the true scalar
kernel, immune to the env switch — because on a DotProd host the traits
dispatch IS the NEON tier and comparing through it would compare the kernel
with itself. The comparison is an f32 bit compare (memcmp of the result), no
tolerance.

Arithmetic summary (how the NEON reduction matches the scalar exactly):

- The scalar `aux32[l]` accumulates a STRIDED partition: within each 32-value
  sub-block, values `l + 8c` for `c = 0..3`, scaled per sub-block. A first
  draft that let `vdotq_s32` lanes group 4 CONSECUTIVE values moved the same
  products into different float accumulators and shifted the final sum by
  1-2 ulp — the suite caught it.
- Fix: both operands are permuted with an 8x4 transpose (`kPerm32[t] =
  t/4 + 8*(t%4)`) so each vdotq lane sums exactly one strided group. The
  nibble side reads the layout the scalar fill actually uses (sub-block `2j`
  = LOW nibbles of q4 bytes `[32j, 32j+32)`, `2j+1` = HIGH nibbles of the
  same bytes — a second suite-caught bug), so value `k` sits at container
  index `k` and the same table permutes both sides.
- The two 16-byte halves stay in SEPARATE accumulators (`acc0` lanes =
  `aux32[0..3]`, `acc1` lanes = `aux32[4..7]`, both over ALL 8 sub-blocks):
  merging them lane-wise turned the scalar's single
  `sums[l] += d * (float)aux32[l]` into two float mul-adds — 1-2 ulp again,
  the suite caught that too.
- The mins bias stays a scalar 16-term integer sum (`sumf -= dmin *
  (float)sumi`), and the final `sumf += sums[l]` fold runs l = 0..7
  sequentially after the block loop, both verbatim from quants.c:645.
  Integer adds are exact, so the lane partitioning cannot move a bit; the
  global `-ffp-contract=off` keeps the float stage non-fused.

Suite: `tests/vt/test_neon_q4k_dot.cpp` — **16130/16130 assertions PASS**:

- 4096 single-super-block randomized cases (4 seeds x 1024) + 1024
  multi-super-block cases (1..4 blocks, exercising the cross-block float
  accumulation order);
- the adversarial grid: d/dmin in {0, f16 denormal bits 1..4, min normal, 1,
  max finite} x nibble modes {all 0x0, all 0xF, alternating, cycling} x q8
  modes {0, +127, -128, alternating extremes, small cycle} x scale bytes
  {0x00, 0xFF};
- per-sub-block scale extremes: all 4096 per-byte 00/FF masks of the 12-byte
  scale field, on saturated and zero nibbles;
- a traits-dispatch end-to-end leg (the kernel reached through
  `QuantTraits(kQ4_K).vec_dot`, the serving entry, vs the scalar reference).

Negative results are part of the evidence: three draft shapes FAILED the
suite (consecutive lanes; 16-byte sub-block layout; merged halves) before the
landed shape passed everything. The integer dot alone does not buy
bit-identity; the float stage's accumulation partition does.

## Serving-token gate (must NOT move — it is the bit-identity check)

`test_flint_gguf` keep-quant paged gate: **53/60 greedy tokens exact, 7
near-tie, 0 hard, worst gap 2.48648** — byte-identical to the standing record
(log: `q4k-gguf.log`). Tokens did not move.

## Other gates

| gate | result |
|---|---|
| `test_neon_q4k_dot` (new) | PASS (16130 asserts) |
| `test_ops_quant_dot` | PASS |
| `test_flint_paged` | PASS |
| `test_flint_paged_engine` | PASS (11 asserts; run from the source dir — the goldens path is CWD-relative) |
| `test_ops_paged_attn_dtype` | PASS |
| `check-agent-record`, commit style `--range`, trailers `--range --filled`, preflight | PASS (preflight `check-env-doc` initially failed on the new `VT_CPU_Q4K_DOT`; documented in `docs/ENVIRONMENT.md`) |

MiMoV2 sink gate not run: the kernel region it pins (paged attention) is
untouched; this row only adds a quant-dot TU and its dispatch arm.

## Harness payoff (FLINT-KQ-CPU-PERF rot-guarded harness, same binary)

`tools/bench/flint_kq_moe_class_ab.cpp` profile-split leg, 3-rep medians,
batch 1, Q4_K_M keep-quant, one gate tower = 512 rows
(logs: `harness-kq-moe-class-ab-{portable,neon}.log`):

| leg | `VT_CPU_Q4K_DOT=portable` | `=neon` | delta |
|---|---:|---:|---:|
| raw vec_dot, one gate tower (512 rows) | 0.234 ms | 0.111 ms | **2.11x faster** |
| projected dot cost, x3 towers x6 experts | 4.2 ms/token-layer | 2.0 ms/token-layer | -52% |
| routed-expert phase (a), batch 1 | 100.114 ms | 96.553 ms | -3.6% |

Share re-confirmation: on THIS build the isolated scalar dot measures as
~4.2% of the harness's routed-phase leg (4.2 of 100.1 ms), not the ~55% the
FLINT-SLIDING-ATTN attribution reported for the in-situ matched-pool routed
phase. The harness's class-(a) leg carries per-expert loop/gather/scatter
overhead the attribution excluded, so the two instruments do not measure the
same denominator; the dot itself halves either way, and the serving A/B below
is the decision instrument.

## Serving (keep-quant paged arm, tg200, 3-rep medians)

First sweep (`kq-t*.log`, idle gate load < 16) was CONTENDED — the box ran
1-min load 13-50 from other work during the legs — and its medians
(16t: 5.92, 32t: 5.79 tok/s) are declared NOISY, not evidence.

Second sweep (`ab/kq-t*-*.log`): same-binary A/B, strict idle gate (1-min
load < 6 before every leg — although the box still carried load 11-18 from
other tenants during the legs, so the ABSOLUTE numbers sit below the
7.01/6.83 FLINT-SLIDING-ATTN records), governor `performance`, portable vs
NEON arms interleaved rep-by-rep:

| threads | pool | portable tok/s (3 reps) | NEON tok/s (3 reps) | median delta |
|---:|---:|---:|---:|---:|
| 16 | 16 (matched) | 5.57 / 5.56 / 5.58 → **5.57** | 5.93 / 5.93 / 5.91 → **5.93** | **+6.5%** |
| 32 | 32 (matched) | 5.60 / 7.07 / 5.71 → **5.71** | 5.74 / 5.82 / 5.74 → **5.74** | +0.5% (one 7.07 portable outlier bounds the contention noise) |

The 16t spread is tight (±0.4% within each arm) and the lift is consistent
with the isolated 2.1x dot speedup over the dot's ~4-8% share of the step;
the 32t arm is noise-bound on this shared box (the outlier lands ON the
portable side, i.e. against the direction of the claimed win).

## Disposition

The NEON dot is bit-identical (16130 assertions, tokens unmoved) and ~2.1x
faster isolated; serving +6.5% at 16t on the interleaved same-binary A/B,
32t unresolved at this noise floor. The routed phase is ~33% of serving
decode, so the remaining gap is dominated by the non-routed phases
(TT-NONROUTED-ATTRIBUTION). No ceiling declared.
