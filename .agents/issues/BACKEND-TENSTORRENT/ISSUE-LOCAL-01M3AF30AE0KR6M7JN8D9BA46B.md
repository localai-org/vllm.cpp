ID: ISSUE-LOCAL-01M3AF30AE0KR6M7JN8D9BA46B
Title: batch>1 GDN prefill asserts at load on current tt-metal (BH 480/192 vs 110 cores)
Row: BACKEND-TENSTORRENT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-24
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

On tt-metal 81f3bbf3b405 the 27B APEX decode asserts at LOAD for any concurrency above 1: chunk_gdn_phased_program_factory.cpp:137 requires BH <= ncores and the GDN prefill shape wants 480/192 against 110 cores. Concurrency 16 and 4 both died before serving a request (DRAM-ledger session, docs/bench-evidence/tt-27b-dram-ledger-20260924.md); only c=1 runs. This makes M>1 batching unreachable on the GDN decode lane at this revision and blocks the +43% M=2 batching result from reproducing. The assert is either a tt-metal program-factory grid-shape regression or a caller-side shape choice that needs the multi-core decomposition the older revision had.

## Resolution

2026-10-21: FALSIFIED BY THE TREE. `d482c6585` ("fix(BACKEND-TENSTORRENT):
decompose GDN batches past the tt-metal core bound", 2026-09-25) names this
issue as the defect it fixes: on tt-metal `81f3bbf3b405` the chunk_gdn phased
program factory requires `batch * value_heads <= compute cores`, so any 27B
APEX shape above concurrency 1 (BH 480/192 against the P150's 110 cores)
asserted at program build during LOAD.

The fix queries the device grid at runtime and decomposes into
`sub_b = max(1, cores / hv)` sub-calls, so this tree carries the split at
`src/vt/tenstorrent/tenstorrent_gdn.cpp:288-319` (prefill) and `:823-848`
(chunked decode). The device gate in the same commit records that the load
assert is gone: c=4 with 16 prompts LOADS and serves, and c=1 stays
byte-identical to the pre-fix binary.

The distinct device limit the fix made visible -- the per-program L1
circular-buffer overflow at the BH' sub-shape -- is a separate open record,
`ISSUE-LOCAL-01M3BMSXHHE6HCAFFFDZQ3QS60`, and is not this issue's condition.

