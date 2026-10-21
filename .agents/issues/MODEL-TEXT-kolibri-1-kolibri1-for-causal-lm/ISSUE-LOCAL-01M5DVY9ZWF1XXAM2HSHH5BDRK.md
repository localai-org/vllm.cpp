ID: ISSUE-LOCAL-01M5DVY9ZWF1XXAM2HSHH5BDRK
Title: the default Kolibri-1 dequant-cache budget is sampled at the wrong moment and can land below the measured knee
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-21
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

The default budget of the Kolibri-1 CPU fp8 dequant cache (src/vllm/model_executor/models/kolibri1_dequant_cache.h) is planned on the FIRST USE of the process-wide cache, which happens DURING the 74 GB checkpoint load: the page cache is full by then, so /proc/meminfo MemAvailable reads 66 GiB where the host's idle reading is ~145 GiB. The policy clamp(MemAvailable / 8, 1 GiB, 32 GiB) then yields 8.25 GiB, and a budget that low buys nothing: measured on node 3 (aarch64, 3.4 GHz, 32 threads, empty-prompt 32-token decode, 2026-10-21) the default leg showed budget_mib=8454 from budget_source_mib=67638, hits=9097, misses=18349, evictions=16068, a 33% hit rate and 3.84 tok/s, against 6.38 tok/s for an explicit 16384 MiB budget and 3.62 tok/s with the cache off. The defect has two parts with one root cause: the reading is taken at a moment our own load has depressed, and the policy can land below the 16 GiB knee the measurement puts at the point where the cache stops thrashing. Fix: sample the host's available memory at PROCESS START (a namespace-scope initialiser, before main and before any model load) and use the MAXIMUM of that reading and the reading at plan time, and floor the budget at the measured 16 GiB knee when the host's honest reading is at or above a stated host threshold, keeping the 1 GiB floor, the 32 GiB cap, the env contract (0 disables, N sets exactly N MiB, an unparseable value falls back to the default policy) and the byte-for-byte uncached fallback.

## Resolution

2026-10-21: the default budget is planned from the HONEST available-memory reading, the maximum of the PROCESS-START reading (a namespace-scope initialiser, taken before main and before the 74 GB checkpoint load) and the reading at plan time, and it is floored at the measured 16 GiB knee when the honest reading is at or above the 64 GiB host threshold; below the threshold the divisor result stands with the 1 GiB floor, so the default is never disabled. The env contract is unchanged (0 disables, N sets exactly N MiB, an unparseable value falls back to the default policy) and the uncached fallback is byte-for-byte. Red-first: against the pre-repair policy the new cases fail at runtime (DefaultBudgetBytesFor(66 GiB) = 8.25 GiB, below the knee). Measured after the repair (2026-10-21, node 3, aarch64, 3.4 GHz, 32 threads, empty-prompt 32-token decode, VT_KOLIBRI1_PROFILE=1, the file mutex held): the DEFAULT leg planned budget_mib=27157 from budget_start_mib=217262 and budget_plan_mib=143920, ended at hits=72190 misses=17756 evictions=7993 (an 80% hit rate) and delivered 7.44833, 7.39141 and 7.46827 tok/s over three runs, against 7.05777, 7.05438 and 6.98659 tok/s for an explicit 16384 MiB budget in the same window and 3.62428, 3.78799 and 3.54865 tok/s with the cache off. The 72 GiB gap between the plan-time and the process-start reading IS the checkpoint load. Gates: test_kolibri1_dequant_cache 13/13 (132200 assertions), test_kolibri1 27/27, test_kolibri1_w2 7/7, test_kolibri1_numa 3/3, test_kolibri1_dequant 3/3, test_kolibri1_moe_glue 9/9, test_kolibri1_bfdot_gemv 5/5, test_kolibri1_decode_bench SUCCESS.
