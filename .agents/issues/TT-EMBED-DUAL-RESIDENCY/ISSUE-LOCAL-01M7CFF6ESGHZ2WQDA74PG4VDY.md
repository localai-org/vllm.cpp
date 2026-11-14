ID: ISSUE-LOCAL-01M7CFF6ESGHZ2WQDA74PG4VDY
Title: TT embed dual residency: the tied embed table lives twice on device (9.95 GiB row words + 2.44 GiB bf16 twin) — eliminate the duplicate and unlock the captured arm
Row: TT-EMBED-DUAL-RESIDENCY
State: OPEN
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-14
Updated: 2026-11-14
Closed: -

## Problem

TT-STAGING-FOOTPRINT's W3 recorded the captured-arm serving gap (~80 MB trace-region short at KV 8, fragmentation-bound) and named the lever: the tied embed table is resident twice — 9.95 GiB keep-quant row words + 2.44 GiB bf16 twin. The dual copy comes from the tied keep-quant non-share fall-through in LoadEmbedAndHead (qwen3_5_gguf_weights.cpp:883 share gate -> :897 OwnMatmulWeight, deliberate non-share :869-880); device side: head row words tenstorrent_keepquant.cpp:107 (pre-pass tenstorrent_capture.cpp:473) beside the gather bf16 twin tenstorrent_residency.cpp:30 (call tenstorrent_ops.cpp:335). Removing the twin frees ~2.4 GB: the captured-arm gap clears with margin, KV headroom grows, and UD-Q8_K_XL viability gets re-checked on a board that is no longer full. Owed: W1 consumer census (which consumers read the bf16 twin vs row words, file:line; the share-gate reason from git history before overturning it), W2 elimination per the census's smallest correct design, W3 THE GATE: captured-arm greedy stream byte-identical to the recorded eager stream b36bc61a... on UD-Q6_K_XL (divergence = STOP) + TPOT comparison + KV headroom + UD-Q8_K_XL viability re-check. Evidence: docs/bench-evidence/tt-staging-footprint-20261113/ + docs/bench-evidence/qwen38-q6k-vs-q4km-20261113/. Spec: .agents/specs/tt-embed-dual-residency.md

## Resolution

-
