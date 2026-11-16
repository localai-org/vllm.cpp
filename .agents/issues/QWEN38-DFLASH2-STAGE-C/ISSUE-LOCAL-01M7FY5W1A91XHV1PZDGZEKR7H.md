ID: ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H
Title: The drafter-ON verify forward degenerates upstream of the GDN spec kernels: the conv-spec x stream collapses at GDN layer 8 with NaNs — the emitted stream is [220, 0, 0, ...] (the W4 token-identity STOP)
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: -

## Problem

With kGdnSpecDecode (id 12) and kGreedyRejectionSample (id 107) both ported, the w4-on-probe leg (Q4_K_M KV 64, drafter-ON, direct arm, bfp8, eager) runs the loop to completion for the first time — exit 0, 24 tokens, acceptance telemetry live (0.75) — and the emitted token stream is [220, 0, 0, ...] vs the recorded greedy stream [220, 16, 23066, ...] (d49c53d2...): the W4 greedy token-identity gate FAILS (the spec's STOP condition). Probe evidence (VT_TT_GDN_SPEC_DEBUG, w4-diag-spec.stderr.log): (1) the conv-spec kernel's x stream — the spec tokens' qkvz projection — has NaN at the FIRST call (step 0, GDN layer 0: x_asum=nan) and collapses to a frozen ~43.3553 sum at GDN layer 8+ in EVERY step (steps 0-3), while the state-window reads stay real (T_asum ~6-15k) — so the degeneracy is UPSTREAM of both spec kernels (the projection GEMM or the residual it reads); (2) the GDN spec kernel's gathered state is real (s_asum ~0.6-9.5k) and its received activations are EXACTLY zero (q_asum=0, v_asum=0) for the collapsed layers — the zero outs are a downstream consequence, not the cause; (3) the acceptance telemetry (24/32 proposed accepted, num_sampled live through finalize) plus the all-zero ids fit BOTH argmaxes degenerating to token 0 — the classic NaN-argmax signature (NaN comparisons false -> best stays 0) — i.e. the verify's block logits AND the draft's block logits are NaN-poisoned. The GDN spec kernel and the sampler are exonerated: their focused gates are bit-exact vs the CPU oracle, the metadata (nat0/init0/slots) is sane every call, and the first NaN precedes any GdnSpecDecode call. Owed: bisect the NaN injection point in the verify forward's spec rows (the embed of the draft's proposals -> the qkvz projection GEMM -> the conv-spec x), with the bfp8 weight-staging lever as the prime suspect (the draft trunk stages BFP8_B; the no-bfp8 A/B is capacity-blocked at KV 32 on this base — the draft's bf16 trunk OOMs at the same 178257920 B ffn_down allocation).

## Resolution

-

## W6-W13 bisect (2026-11-15 evening — the follow-on session's record)

The W4 "collapses at GDN layer 8" framing is WRONG in one respect and right in
a sharper one: the collapse is NOT op-fixed — its onset layer MOVES with the
staging arm (w7 device-res: l=10; w8/w9 forced-host-residual: l=4, the first
full-attn layer; w12 host-free-off: l=1). The poison is POOL-STATE dependent.
What stays true: the poison is live in the FIRST spec step (T=9 — the only
non-tile-multiple geometry in the workload), and the verify's logits collapse
to argmax-0 (the all-zero accepted stream).

New evidence (all in docs/bench-evidence/qwen38-dflash2-stage-c-w3-20261115/,
README.md's W6-W13 section carries the full record):
- `[DRAFT_LOGITS]` (w13): propose #1 CLEAN (asum 4.94e6, 0 NaN); proposes #2-3
  read 100% NaN. The drafter is a VICTIM (it reads the verify's poisoned aux
  taps), not the poisoner.
- `[SPEC-OUT]`/`[RNG-IN]` (w11): the z gate reads asum 3.62e40 + 109 Infs at
  the spec step's 8th GDN layer (l=9) from CLEAN h (22227); from l=10 the gate
  reads ALL-ZEROS for the rest of the run. `[QKVZ-OUT]` mixed reads garbage
  (8.67e40+) from call 51 (l=4) while `[QKVZ-IN]` h reads clean — in BOTH
  staging arms. The poison sits between the qkvz GEMM's clean input and its
  garbage-read output.
- `[CONV-SPEC-STAGE]` (w7b): the conv-spec fill's cached zeros base reads
  ZEROS at fill time while the fill's output reads recycled garbage (identical
  bytes asum=1.34712e40 + 36 NaNs at every first-GDN-after-FA call). tt-metal
  indexed_fill = out-of-place clone + row writes (indexed_fill.hpp:11-15), so
  the output garbage is NOT the base's content.
- `[TT-SLOT-ALIAS]` census scan (w10): ZERO duplicate device addresses across
  all slots — no double-hand.

Op-level exonerations (standalone tests, GREEN at the engine geometry,
47/47 assertions): `kCausalConv1dSpecUpdate` sweep F (C=10240, widened
state_len=11, T=9, slots=9, nat=1, NaN-seeded out: max_abs 2.98e-08,
nonfinite=0) and sweep G (shifted-address second call: the call-1 buffers
held alive so call 2 lands elsewhere — max_abs 2.98e-08). The stale-address
replay hypothesis is dead for this op. `GdnSpecDecodeKernel` checksums clean
at every T=9 call. EnsureDevice2D REFUSES row-strided views outright
(residency.cpp:930) — a new test pins that contract.

## W20 (2026-11-16): the trusted dual-read — the alias theory FALSIFIED

The decisive instrument ran (README.md's W20 section carries the full
record). Verdicts:

- **The stale-download/serve-alias hypothesis is FALSIFIED.** The read-only
  trusted dual-read at the qkvz COMMIT side reads the same garbage the
  serve reads (asum 3.75443e+40, 113 Infs at the first spec step's l=9,
  call 55). The matmul's own output reads garbage PRE-commit
  ([ZGEMM-PRECOMMIT]) with BOTH operands dual-read clean and stable at the
  same site ([MMBT9]: h asum 22227.6, the staged weight byte-identical
  across steps).
- **The block-recycle alias is falsified by intervention.** A 3-deep output
  keep-alive in MatmulBTKernel (the current output's block can never be a
  recent recycle) MOVED the poison (onset later, bigger: ~9e42, ~2500 Infs)
  instead of removing it. The recycled-block content is not the mechanism.
  The keep-alive stays (harmless; onset later; first token 220→23066) but
  the stream stays degenerate (e171d6b1).
- **Three isolated reproductions at the exact engine geometry are ALL
  GREEN** (tests/vt/test_tenstorrent_backend.cpp, w20-bfp8-{m9,warm,unsynced}
  .log): the bfp8-staged weight at M=9/M=32 vs the f32 oracle (max_rel
  0.75 percent), 200 warm program-cache-hit calls, 50 unsynced-staging
  lifecycle reps. The W3 scope claim that the target's GDN weights never
  route through EnsureMatmulWeightDevice is WRONG (they do; bfp8 degrades
  them since W3) but the bfp8 matmul itself is numerically fine at the spec
  geometry.
- The dead agent's qkvz commit-side TrustDump calls were unreachable dead
  code (after `return out;` in both ProjectGdnQkvzRaw arms) — why waves
  w14-w19 captured no qkvz commit. Fixed.

## Next instrument

**SHARPENED: the recurrence's core output is ~ZERO at the spec calls**
(asum 120 → 1.2 → 0.9, exactly 4 of 9 rows zero, later NaN) with CLEAN
q/k/v (the pc_q/pc_v trusted dumps have no zero rows pre-collapse). The
recurrence's UNPROBED inputs are the next probes: the INITIAL-STATE gather
(column num_accepted-1 — the prefill's committed state as the ENGINE holds
it) and the BA outputs (araw/braw → g/beta; the sba_* dumps cover only the
prefill). The core ~0 explains the degenerate stream on its own; the z
GEMM's 1e40 read at c55 is a second anomaly (it collapses h at l=10).

Beyond that, the poison needs the engine's INTERLEAVED op mix: the conv-spec fill's
indexed_fill (the cached-zeros-base/clone anomaly stands), the GDN
recurrence's gathers/scatters, and the paged attention, all sharing the
tt-metal program cache and the DRAM allocator with the matmuls. Strongest
hypothesis: a tt-metal-pin program-cache/runtime-arg staleness under
interleaved programs (the kernel writing a previous call's buffer while the
current output reads uninitialized recycled bytes — fits the keep-alive
intervention moving the poison, the identical-bytes periodicity, and the
staging-arm-dependent onset). Next step: the interleaved-op focused case
(matmul + indexed_fill + gather through the SHARED program cache in one
harness) — the first reproduction that varies the op MIX, not the op — plus
the state-gather/BA trusted reads at the spec calls.
