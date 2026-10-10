ID: ISSUE-LOCAL-01M4GQ82X99D6JNBWDPX7ATJPY
Title: kolibri1 CPU decode: in-process NUMA interleaving for weight arenas
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-10
Closed: -

## Problem

External numactl --interleave=all + 32 threads lifts kolibri1 CPU decode from 3.17-3.21 to 4.69 tok/s (+47%) (docs/bench-evidence/kolibri1-perf-next-lever-20261008.md; PR #3423 repair section). The default first-touch allocation keeps weight and dequant-cache pages on the loading node, so 32 threads across 4 NUMA nodes hit remote memory. The lever must land in-process (no external wrapper) as an allocation policy on the weight arenas, gated behind VT_KOLIBRI1_NUMA_INTERLEAVE for A/B, with numerics unchanged (anchor 109726, chain md5 8de1463a on every leg) and a fresh post-NUMA decode-phase attribution naming the next lever.

## Resolution

2026-10-10: implemented and measured on row/kolibri-numa-interleave (implementation commit 1bdb9c1cc). In-process MPOL_INTERLEAVE over LoadKolibri1Weights and the dequant-cache miss decode (kolibri1_numa.h, raw syscalls, VT_KOLIBRI1_NUMA_INTERLEAVE=0 opt-out, default ON). A/B 3 reps x t8/t32 x off/on: t32 3.963/4.076/4.105 off vs 4.638/4.635/4.585 on tok/s; t8 3.168/3.149/3.143 vs 3.213/3.190/3.229. ALL 12 legs exit 0 with chain md5 8de1463a / anchor 109726; profiled-leg whole-run counters byte-identical to the record (hits=71157 misses=18717 evictions=13264 decode_calls=89874). W3 at the final config (t32 + interleave on + 16 GiB) PASS: 900/900, ARGMAX 141/145 (4 near-tie, 0 hard), fingerprints 2.18646/26763. Kolibri battery green (test_kolibri1, _dequant, _dequant_cache, _moe_glue, mimov2_w1, _w2) plus new test_kolibri1_numa 9/9. Post-NUMA decode attribution (R60-R10, 50 pure decode forwards): linear_gemm 46 pct (97.0 ms/step, was 59 pct/178), attn_core 39 pct (82.0 ms/step, UNCHANGED), moe_glue 20 pct nested, decode step 210.6 ms (was ~315) - the next wall is the attention core. Evidence: docs/bench-evidence/kolibri1-perf-next-lever-20261008.md section 4. The issue CLOSES when the PR merges.
