ID: ISSUE-LOCAL-01M4GQ82X99D6JNBWDPX7ATJPY
Title: kolibri1 CPU decode: in-process NUMA interleaving for weight arenas
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-10
Closed: 2026-10-10

## Problem

External numactl --interleave=all + 32 threads lifts kolibri1 CPU decode from 3.17-3.21 to 4.69 tok/s (+47%) (docs/bench-evidence/kolibri1-perf-next-lever-20261008.md; PR #3423 repair section). The default first-touch allocation keeps weight and dequant-cache pages on the loading node, so 32 threads across 4 NUMA nodes hit remote memory. The lever must land in-process (no external wrapper) as an allocation policy on the weight arenas, gated behind VT_KOLIBRI1_NUMA_INTERLEAVE for A/B, with numerics unchanged (anchor 109726, chain md5 8de1463a on every leg) and a fresh post-NUMA decode-phase attribution naming the next lever.

## Resolution

2026-10-10: landed on main via PR #3431 (1bdb9c1cc..bb4abf97b, on main as eb92c585f/0290598f5). Measured: same-binary A/B, 32 threads + 16 GiB dequant cache, three interleaved pairs under one mutex hold — decode 4.192 -> 4.973 tok/s (+18.6%), prefill 7.96 -> 6.84 s (-14%), beating the external numactl figure (4.69); golden chain 101807,109726 on all six legs; W3 900/900, ARGMAX 141/145 (4 near-tie, 0 hard). Mutation review: policy-set-and-restore, multi-node residency, and the VT_KOLIBRI1_NUMA_INTERLEAVE=0 off switch each red-first proven; the kolibri1_weights.cpp guard site is pinned by the model-gated loader case (bb4abf97b), the dequant-cache site by the cache case. Evidence: docs/bench-evidence/kolibri1-numa-interleave-20261009.md.
