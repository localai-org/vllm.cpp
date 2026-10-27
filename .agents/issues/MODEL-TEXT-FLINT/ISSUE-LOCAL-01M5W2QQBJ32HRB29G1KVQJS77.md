ID: ISSUE-LOCAL-01M5W2QQBJ32HRB29G1KVQJS77
Title: Flint CPU paged-engine serving arm (MODEL-TEXT-FLINT)
Row: MODEL-TEXT-FLINT
State: OPEN
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-26
Updated: 2026-10-26
Closed: -

## Problem

Flint serves on CPU only through the single-sequence reference forward: the model never reaches the LLMEngine/paged path, so there is no continuous batching and no honest CPU speed number. W-b (2026-10-26/27, worktree /tmp/flint-paged): the paged arm landed through ModelRegistry::Forward (multi-request batching, per-head sinks on every layer, sliding-window paged attention, hybrid KV groups) and the synthetic gate now GATES the house 2.5-nat near-tie band (request 0 replays the reference chain token-for-token). The gate caught and fixed three real defects: the in-place QK RMSNorm, the shared cpu_paged_attn sink seed (ISSUE-LOCAL-01M5YMB0XZ8723N0688YHMKP3N), and a double-added residual stream in the arm. Owed: the W-c real-weights paged gate, the W-d bench, the W-e docs.

## Resolution

-
