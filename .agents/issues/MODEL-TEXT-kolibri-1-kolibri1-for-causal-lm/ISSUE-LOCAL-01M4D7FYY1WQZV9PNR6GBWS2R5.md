ID: ISSUE-LOCAL-01M4D7FYY1WQZV9PNR6GBWS2R5
Title: kolibri-1 CPU: moe_glue is the largest remaining decode stage — profile it and land ONE bit-exact sub-lever
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: 2026-10-08

## Problem

With the default NEON GEMM tier plus the opt-in 16 GiB fp8 dequant cache, kolibri-1 CPU production decode is ~3.1 tok/s (2.52x the NEON-on cache-off baseline). The VT_KOLIBRI1_PROFILE stage table on the decode bench shows moe_glue as the biggest remaining stage (it nests the routed experts' glue: sigmoid-of-unbiased-logits routing weights, top-k selection bookkeeping, per-expert gather/scatter, expert output combine, shared-expert path). Evidence: docs/bench-evidence/kolibri1-combined-neon-cache-20261008.md (mine/row/kolibri-combined) and docs/bench-evidence/kolibri1-dequant-cache-reland-20261007.md. This unit: profile moe_glue's sub-operations on main (cache ON 16 GiB and OFF) on the decode bench and W3, pick ONE sub-lever justified by the profile numbers, and land it under the row's bit-exactness contract (order-preserving NEON across independent outputs or layout changes that preserve per-element arithmetic; anything that cannot be bit-exact by construction is skipped and recorded). Gates: W3 900/900 with chain 141/145 + fingerprints 2.18646/26763 at 16 GiB cache, decode_bench anchor 109726 with chain md5-identical across cache off/8/16 GiB, unit suites green, red-first raw-uint16 bit-identity tests over the real expert geometries (2560 hidden, 384 experts, 6-of-384, shared expert, ragged/tail) plus a mutation check, same-host A/B on the decode bench (cache on 16 GiB primary, cache off reported), 8 threads, quiet window, 2+ runs each. Evidence record: docs/bench-evidence/kolibri1-moe-glue-<lever>-20261008.md.

## Resolution

Landed 2026-10-08 on branch `row/kolibri-glue` (worktree
/tmp/vllm-kolibri-glue, base origin/main 5d2f5d31a). Full record:
docs/bench-evidence/kolibri1-moe-glue-neon-combine-20261008.md.

PROFILE (VT_KOLIBRI1_PROFILE=1, 8 threads, decode bench + W3, cache OFF and
ON 16 GiB, pristine main binary + a scratch fine-grained instrumentation
for the sub-op split): at cache ON the decode step is 317.5 ms; moe_glue is
81.5 ms of it (25.7%; 15.53 s of the 35.97 s whole-run stage totals — the
biggest stage), split into nested expert GEMMs 71.5 ms (87.7% of moe_glue),
nested expert dequant-cache work 3.5 ms, nested expert silu 2.5 ms, and a
MoE-scoped glue-exclusive 4.5 ms whose largest bit-exact-able item is the
expert-output combine at 2.0 ms/fwd. W3 (cache ON): moe_glue 60.4% of the
forward, dequant 43.4%, linear_gemm 48.1%; the W3 gate shape thrashes the
16 GiB cache (60% miss rate) so the W3 wall moves only -3.8%.

LEVER (one): order-preserving NEON for the expert-output combine
(`MoeCombineKernel`, src/vt/cpu/cpu_ops.cpp) — vectorized across the
independent output columns, each lane keeping the scalar j-order reduction,
products vmulq+vaddq (never vfmaq), the bf16 widening/store bit-identical
to BF16ToF32/F32ToBF16. Bit-exact by construction; the skipped candidates
and their reasons are recorded in the evidence doc (expert GEMM stream
pattern: shared vt kernel, K-axis-locked; expert cold decodes: cache
component, budget-bound; expert silu: std::exp not bit-exact-vectorizable;
scan/loop bookkeeping: smaller than the combine; dead expert_out zero:
negligible).

RED-FIRST + GATES: tests/vllm/models/test_kolibri1_moe_glue.cpp (7 cases,
21 assertions, raw-uint16 bit-identity over the real expert geometries:
2560-wide hidden, 6-of-384 route over the 384-expert domain, shared-expert
path, ragged t=1/7/128, h/k tails, exact-tie stores, overflow/default-NaN,
cross-thread identity). Mutations all RED and restored byte-for-byte:
scalar j-reorder (3 cases red), NEON vfmaq fusion (5 cases red), NEON
j-reorder (3 cases red). W3 gate at 16 GiB cache: 900/900, chain 141/145
(4 near-tie, 0 hard), fingerprints 2.18646/26763 — identical to baseline.
decode_bench anchor 109726 + chain md5 255e04ab... identical at cache
off/8/16 GiB. Unit suites green (test_kolibri1 186/186, w2 1608/1608,
dequant 10/10, dequant_cache 52/52, moe_glue 21/21) and the shared MoE
pins (81/81, 69/69, 219/219). check-agent-record rc=0 (one stale anchor
repaired: kernel-matrix KERNEL-ATTN-DFLASH-BLOCK cpu_ops.cpp:3541 -> 3619,
shifted by this change's insertion).

A/B (decode bench, 8 threads, quiet window, 2+ runs per config): decode
tok/s OFF 1.256 -> 1.257 (+0.08%), 8 GiB 3.028 -> 3.052 (+0.79%), 16 GiB
(primary) 3.129 -> 3.170 (+1.31%); decode wall -0.06% / -0.75% / -1.24%.
All deltas non-negative and inside the run-to-run noise (baseline paired
spread 0.9-2.2%); the attribution-derived expectation is ~0.5% (combine
2.0 ms of the 317.5 ms step, ~4x kernel speedup). Not a negative result,
so the lever LANDS, recorded with its noise context — a small real win,
not a measurable one; the publishable production figure is unchanged
(~3.1 tok/s at 16 GiB).

OWED FOLLOW-UPS recorded with measured shares (evidence doc §Outcome):
shared GEMM M=1 GEMV weight-stream pattern (22.5% expert + 14% attn/other
+ 4.7% lm_head of the decode step; vt-layer, K-axis-locked); attn_core at
t=1 (28%); the unscoped 23% forward remainder (needs its own scopes);
the dequant cache's cold-miss page faults on fresh 2.6 MB entries
(3.5 ms/fwd at decode, 7.4 s of the prefill); a CSR routing-bucket build
(2.0 ms/fwd of bookkeeping, bit-exact); the dead expert_out zero skip.

Window deviation (documented in the evidence doc): the strict quiet window
was held closed for the first 8 legs by a foreign session's stalled
single-core device test (test_kolibri1_tt_b2i, 79.4 GB static, >1 h); those
legs ran under a narrow waiver (available > 110 GB AND zero foreign model
procs except tt_b2i; ~63 GB headroom for the worst leg; a watchdog would
have killed any leg within 5 s of another foreign model proc appearing — it
never fired). tt_b2i exited ~11:23; every leg from prof-w3-off onward ran
under the STRICT window.
-
