ID: ISSUE-LOCAL-01M5DMH6KQJSP9F8PFCQAAS4WH
Title: the Kolibri-1 CPU dequant cache defaults to OFF, so the shipped configuration is the slow one
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: enhancement
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-20
Updated: 2026-10-21
Closed: 2026-10-21

## Problem

The Kolibri-1 CPU forward re-dequantises every used FP8 weight on every forward unless the decoded weights are kept resident in the bounded LRU dequant cache (src/vllm/model_executor/models/kolibri1_dequant_cache.h). The cache is OFF by default: VT_KOLIBRI1_DEQUANT_CACHE_MB unset means budget 0, and the shipped configuration therefore pays the per-forward dequant on every decode step. The configuration is worth MORE THAN 2x on CPU decode: 4.02 tok/s with the cache off against 9.07 tok/s with a 16 GiB budget (2026-10-20, node 3, aarch64, 3.4 GHz, 32 threads, empty-prompt 32-token decode; prefill p512 43.33 tok/s with the cache off). Fix: make the cache the DEFAULT for CPU serving, with a budget derived from the host's own available memory (clamp(MemAvailable / 8, 1 GiB, 32 GiB)) so a small machine is not committed to a large allocation, keep VT_KOLIBRI1_DEQUANT_CACHE_MB=0 as the off switch and an explicit value as the exact budget, and make a failed allocation degrade to the uncached path instead of aborting the forward. The cached bytes must stay byte-identical to the uncached ones: the cache is a caching change, never an arithmetic one.

## Resolution

2026-10-20: the CPU dequant cache is the DEFAULT, with the budget derived from the host's available memory (clamp(MemAvailable / 8, 1 GiB, 32 GiB)); VT_KOLIBRI1_DEQUANT_CACHE_MB=0 still disables it and an explicit value still sets the budget exactly. An allocation the cache cannot satisfy degrades the process to the uncached path instead of failing the forward. The cached bytes are byte-identical to the production uncached dequant, cold and hot. Gates: test_kolibri1_dequant_cache 11/11 (2127 assertions), test_kolibri1_w2 7/7, test_kolibri1 27/27, test_kolibri1_numa 3/3, test_kolibri1_dequant 3/3, test_kolibri1_moe_glue 9/9, test_kolibri1_bfdot_gemv 5/5; the W3 token gate and the n-gram spec gate are red on pristine origin/main and unchanged by this change (W3: 134/138, 4 near-tie flips, 0 hard, worst topk 3.40201 before and after). The decode and prefill benchmark rows for the default configuration are owed to the node-3 window; the commands are in the pull request body.
