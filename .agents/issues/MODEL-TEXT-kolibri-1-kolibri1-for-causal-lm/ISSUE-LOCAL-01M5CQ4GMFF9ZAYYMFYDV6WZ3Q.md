ID: ISSUE-LOCAL-01M5CQ4GMFF9ZAYYMFYDV6WZ3Q
Title: primary top-100 rows are stored one position behind the generated chain (off-by-one in the writer)
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-20
Updated: 2026-10-20
Closed: 2026-10-20

## Problem

scripts/gen-kolibri1-primary-topk-rows.py:66 computes pos = len(p_ids) - 1 + k, but vLLM prompt_logprobs[i] is the predictive distribution over full[i], so the distribution predicting gen[k] lives at pos = len(p_ids) + k. Every emitted row (gen_index k) therefore holds the previous position's distribution. The skip 'if not any(tid == gid)' also tests membership in the wrong distribution. A peer measured generated[k] == rows[k+1].top1 at 195/195 positions, confirming the one-step lag. Affected consumers: tests/vllm/models/kolibri1_primary_topk_rows.json (landed via #3463) and the margin instruments that consumed it (kolibri1-threeway-analysis.py outputs, near-tie/hard-flip margins for the W3 prompts). #3482's red line used a consistent-but-lagged alignment, so ordering survives but the absolute label does not. Fix: pos = len(p_ids) + k; after the fix gen[k] is by construction the argmax of that distribution under greedy, so the silent skip becomes a loud CHECK that gen[k] is top-1.

## Resolution

2026-10-20: writer fixed (pos = len(p_ids) + k, skip replaced by a top-1 CHECK) in row/PRIMARY-ROWS-OFFBYONE; rows re-emitted from the primary oracle at 256/256 positions with zero skips and generated[k] == rows[k].top1 at 256/256; corrected three-way and margin numbers recorded in docs/bench-evidence/kolibri1-primary-rows-offbyone-20261020/report.md.
