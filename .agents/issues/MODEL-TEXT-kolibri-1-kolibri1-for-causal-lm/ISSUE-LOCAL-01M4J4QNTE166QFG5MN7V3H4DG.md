ID: ISSUE-LOCAL-01M4J4QNTE166QFG5MN7V3H4DG
Title: CPU paged-attention NEON lane returns wrong tokens under thread contention
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-10
Updated: 2026-10-10
Closed: 2026-10-10

## Problem

The default-on NEON lane (0c49a679e, PR #3433) serves a DEGENERATE chain (prefill anchor 33382 instead of 101807, then token 33382 forever) on the kolibri1 decode bench whenever the host is under concurrent load (load avg ~18 from a parallel build). Same binary, same env: NEON=0 on the same loaded host returns the golden anchor; NEON=1 with cache and NUMA off reproduces the garbage. On a quiet host (load ~1.7) the identical binary passed W3 900/900 and the bench at 02:48. So the lane's parallel dispatch has a load-dependent race: thread scheduling changes the outcome. The scalar path under the same load and threads is correct. The default flip must be rolled back until the race is fixed; the =0 rollback works.

## Resolution

2026-10-10: FALSIFIED by the tree — there is no race. The 33382 first-token flip is the KNOWN, adjudicated near-tie flip recorded in docs/bench-evidence/kolibri1-neon-paged-attn-20261008.md since PR #3425: deterministic at every thread count (reproduced single-threaded at t1), invariant across allocator poisoning, identical on the pre-flip binary; the NEON replay agrees with scalar to 1 bf16 ulp and the scalar top1-top2 gap at the flip position is 0.197 nats — inside the row's 2.5-nat band. The load correlation was coincidence. The real record gap: PR #3433 flipped the default without the anchor adjudication the #3425 disposition names as a precondition; the rollback (#3435) restores opt-in. What is owed is NOT a race fix but the gate-semantics decision: a ratified gap instrument for the decode-bench anchor (like W3's) before any default flip.
