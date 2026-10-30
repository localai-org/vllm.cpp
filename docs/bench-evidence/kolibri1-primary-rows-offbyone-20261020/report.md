# Kolibri-1 primary top-k rows off-by-one: writer fix, re-emission, impact (2026-10-20)

Issue: ISSUE-LOCAL-01M5CQ4GMFF9ZAYYMFYDV6WZ3Q
(row MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm). Branch
`row/PRIMARY-ROWS-OFFBYONE` (fresh from `origin/main` `61d333e52`).

## Bug

`scripts/gen-kolibri1-primary-topk-rows.py` scored the teacher-forced chain at
`pos = len(p_ids) - 1 + k`. vLLM's `prompt_logprobs[i]` is the predictive
distribution OVER `full[i]`, so the distribution that PREDICTS `gen[k]` sits at
`len(p_ids) + k`. Every emitted row held the previous position's distribution,
and the skip (`if not any(tid == gid)`) tested membership in that wrong
distribution. The peer measured `generated[k] == rows[k+1].top1` at 195/195,
which is the one-step lag read directly off the file.

## Fix

`pos = len(p_ids) + k`, the stale index comment corrected, and the silent skip
replaced by two loud failures: a missing distribution and a top-1 CHECK — under
greedy decode `gen[k]` must be the argmax of the distribution that predicted it,
so the old skip path can only now mean a diverged chain, and the generator
aborts instead of emitting a shifted row.

## Re-emission

Environment: the same primary-oracle environment as the original run —
`/tmp/opencode/aaiv` venv with the source build `/tmp/opencode/vllmsrc`
(vLLM 0.31.0, `VLLM_TARGET_DEVICE=cpu`, `aleph_alpha_inference` plugin),
`tensor_parallel_size=4` mp on this host, model
`/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean`, greedy, `max_tokens=32`,
`prompt_logprobs=100`. Output overwrote
`tests/vllm/models/kolibri1_primary_topk_rows.json`.

Validation: 256/256 generated positions emitted, zero skips, and the top-1
CHECK passed on every position (`generated[k] == rows[k].top1` at 256/256). A
second independent run reproduced the file byte-for-byte. The old file scored
202/256 with only 63/202 rows whose top-1 was the emitted token.

## Old vs new (the lagged numbers were cross-position comparisons)

`scripts/kolibri1-threeway-analysis.py` over the four captured dump dirs
(`/tmp/threeway/{prod,draft,before_f32,after_bf16dot}`), same 202 aligned dump
positions (the dumps pre-date the fix; the corrected fixture covers all 256
positions):

| mode | top-1 | gap mean (OLD) | gap mean (NEW) | cos (OLD) | cos (NEW) | KL (OLD) | KL (NEW) |
|---|---|---|---|---|---|---|---|
| prod | 98.0% | 5.806 | **-0.028** | 0.9814 | **0.9991** | 7.070 | **0.017** |
| draft | 97.0% | 5.836 | **0.002** | 0.9812 | **0.9989** | 6.971 | **0.020** |
| before_f32 | 97.0% | 5.842 | **0.008** | 0.9812 | **0.9989** | 6.994 | **0.021** |
| after_bf16dot | 98.0% | 5.854 | **0.020** | 0.9804 | **0.9981** | 7.018 | **0.051** |

Rank percentages (top-1/top-5/top-100) are unchanged in every mode: the lagged
alignment was consistent-but-shifted, so ordering conclusions survive, while
every absolute magnitude collapses from ~6 nats of phantom gap to hundredths of
a nat. Our production bf16-dot default's real top-1 agreement with the primary
over aligned positions is 98.0% (196/202 dump positions), and its real logprob
gap to the primary is -0.03 nats (ours marginally ABOVE the primary on mean).

Margin instruments (top1_top2 / top6_top7, all 8 prompts):

| metric | OLD (202 rows) | NEW (256 rows) |
|---|---|---|
| top1_top2 median | 3.8913 | 3.7989 |
| top1_top2 p10 | 1.0849 | 0.4812 |
| top1_top2 min | 0.0022 | 0.0022 |
| top6_top7 median | 0.1855 | 0.1901 |
| top6_top7 min | 0.0028 | 0.0018 |
| top6_top7 rows < 0.1 nats | 58/202 | 69/256 |

Every same-index margin changed (0/202 identical): each old row carried the
previous position's distribution, so no old margin label is trustworthy. The
NEW file has 54 more scored positions, and the knife-edge tail is larger than
the lagged file showed (69 positions with top6_top7 < 0.1 nats vs 58).

## Impact on prior analyses

- `kolibri1-threeway-20261018` (#3469): the mode ORDERING stands (prod
  closest, bf16-dot default next, f32 tiers behind — and the corrected table
  keeps prod and after_bf16dot at the top with the same 98%/97% split). The
  absolute gap/cos/KL numbers in that report are superseded by the table above;
  they measured our position k against the primary's position k-1.
- The bf16-dot adoption gate (`kolibri1-bf16dot-adoption-20261019`, #3482 red
  line): STANDS. The red line was top-1 agreement not dropping, computed on a
  consistent-but-lagged alignment; the corrected alignment gives the same
  97.0% -> 98.0% BEFORE -> AFTER move. The absolute gap/KL columns in that
  report should be re-read as the NEW values above (after_bf16dot real gap is
  +0.020 nats, KL 0.051 — still the loosest mode, same conclusion).
- Band margins for the W3 prompts: the near-tie/hard-flip classification in
  `tests/vllm/models/test_kolibri1_w3.cpp` uses OUR logit gaps, not this file,
  so that gate is untouched. Instruments that quoted primary margins (the
  knife-edge census above) shift to the NEW column.

No test reads the rows file's content (`grep -rl primary_topk_rows tests/`:
only the path macro and manifest provenance in
`test_kolibri1_threeway_capture.cpp`), so no recorded anchor changes in this
change.

## llama.cpp goldens

`tests/vllm/models/kolibri1_llamacpp_topk_rows.json` landed with this change:
8 prompts x 32 positions of per-position token-ids + top-20 logprobs from
llama.cpp commit `62318cd57` (Kolibri-1 support), GGUF sha256
`76ddb05e458cce1a023d42cd4f13dc8e8152fbb6501c1f680918fc7d2a0fcefd`, BF16,
greedy. Validated 256/256 rows with top-1 == emitted token.
