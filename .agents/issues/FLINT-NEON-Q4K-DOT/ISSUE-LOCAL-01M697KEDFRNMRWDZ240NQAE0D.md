ID: ISSUE-LOCAL-01M697KEDFRNMRWDZ240NQAE0D
Title: The paged kernel's Q4_K->Q8_K integer vec_dot runs a scalar generic port at m=1 decode -- ~55% of the routed phase
Row: FLINT-NEON-Q4K-DOT
State: CLOSED
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: 2026-10-31

## Problem

FLINT-SLIDING-ATTN's evidence: with pool matched and attention fixed, ~55% of the matched-pool routed-expert phase is SCALAR dots -- the Q4_K->Q8_K quantize-once-integer-dot class runs the generic scalar vec_dot at m=1 decode while the attention kernel got its NEON lane. Deliverable per spec: a NEON vec_dot_q4_K_q8_K BIT-IDENTICAL to the scalar reference (integer dot has no reordering -- divergence = wrong port, STOP), wired behind the house gating pattern, payoff measured on the FLINT-KQ-CPU-PERF harness + the keep-quant serving arm (token gate must NOT move -- bit-identity is itself the check). Spec: .agents/specs/flint-neon-q4k-dot.md

## Resolution

- 2026-10-31: NEON vec_dot_q4_K_q8_K landed (commit 273bdb9d1), BIT-IDENTICAL
  to the scalar reference on 16130 suite assertions; keep-quant serving token
  gate unmoved (53/60, 0 hard, worst 2.48648). Isolated dot 2.11x faster;
  serving +6.5% @16t on the same-binary A/B, 32t noise-bound on a shared box.
  Evidence: docs/bench-evidence/flint-neon-q4k-dot-20261031/ (the ~55% share
  claim does not reproduce on the micro harness — ~4.2% measured; instrument
  denominators differ, recorded in the README).

