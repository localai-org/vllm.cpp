ID: ISSUE-LOCAL-01M4GQ82X99D6JNBWDPX7ATJPY
Title: kolibri1 CPU decode: in-process NUMA interleaving for weight arenas
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: -

## Problem

External numactl --interleave=all + 32 threads lifts kolibri1 CPU decode from 3.17-3.21 to 4.69 tok/s (+47%) (docs/bench-evidence/kolibri1-perf-next-lever-20261008.md; PR #3423 repair section). The default first-touch allocation keeps weight and dequant-cache pages on the loading node, so 32 threads across 4 NUMA nodes hit remote memory. The lever must land in-process (no external wrapper) as an allocation policy on the weight arenas, gated behind VT_KOLIBRI1_NUMA_INTERLEAVE for A/B, with numerics unchanged (anchor 109726, chain md5 8de1463a on every leg) and a fresh post-NUMA decode-phase attribution naming the next lever.

## Resolution

-
