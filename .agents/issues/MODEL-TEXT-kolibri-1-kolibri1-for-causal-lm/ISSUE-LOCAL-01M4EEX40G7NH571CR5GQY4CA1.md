ID: ISSUE-LOCAL-01M4EEX40G7NH571CR5GQY4CA1
Title: kolibri1 CPU: name and land the next decode performance lever from the stage profile
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: enhancement
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: 2026-10-08

## Problem

Production decode is ~3.1 tok/s (8 threads, 16 GiB dequant cache) after two landed levers (NEON GEMM tier 2.07x, fp8 dequant cache 2.52x). The recorded stage profile (docs/bench-evidence/kolibri1-combined-neon-cache-20261008.md) still shows moe_glue ~0.25 s (residual per-expert dequant work), linear_gemm ~0.25-0.30 s, and unchanged attn_core/lm_head per forward. No measurement-first attribution of the remaining wall exists on the current tree, so the next lever is unnamed. Profile-first scope: reproduce the baseline, attribute the decode wall per stage at 4/8/16 threads, name the top lever from measurement, land it red-first with bit-exactness pinned (fingerprints 2.18646/26763, anchor 109726, budget invariance), or deliver a measured proposal if the lever is a policy choice (cache default/budget).

## Resolution

- 2026-10-08 (branch `row/kolibri-perf-next`): resolved as a measured
  proposal, no product code. Baseline reproduced (3.17-3.21 decode tok/s,
  8 threads, 16 GiB; anchor 109726, chain md5 8de1463a, counters
  byte-identical). Remaining wall attributed: linear_gemm ~50% of forward at
  the order-preserving ALU roof, cold-miss whole-matrix decodes ~26%,
  attention ~17%, lm_head 4%; the recorded moe_glue "residual" reading is
  corrected (the scope nests the routed experts' GEMM/dequant). Falsified by
  measurement: fp8-direct GEMV (bit-exact, 8x slower — the kernel is
  ALU-bound, not bandwidth-bound); no bit-exact GEMM inner-loop lever exists
  under the row's contract. The one open lever is the dequant-cache
  production budget (developer decision): 8 GiB 2.99-3.13 tok/s, 16 GiB
  3.17-3.21, 32 GiB 3.36-3.37 (evictions 18321/13264/5043), OFF 1.23-1.26;
  W3 16 GiB gate green on this tree (900/900, ARGMAX 141/145 = 4 near-tie,
  fingerprints 2.18646/26763, counters byte-identical, 92.7 GiB RSS).
  Evidence: docs/bench-evidence/kolibri1-perf-next-lever-20261008.md.
