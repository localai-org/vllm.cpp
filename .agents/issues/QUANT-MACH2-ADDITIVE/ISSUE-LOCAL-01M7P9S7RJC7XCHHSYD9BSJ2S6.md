ID: ISSUE-LOCAL-01M7P9S7RJC7XCHHSYD9BSJ2S6
Title: q4logit-diff prints no rel_proj or rel_sumabs, though the spec, the docstring, the runtime trailer and --kl all promise them
Row: QUANT-MACH2-ADDITIVE
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-17
Updated: 2026-11-17
Closed: 2026-11-17

## Problem

scripts/q4logit-diff.py defines rel_sumabs() and rel_proj() and calls neither. Its per-step table ends at `appeared`, so a reader is told four times to trust a column that does not exist: the docstring's WHAT THIS TOOL PRINTS INSTEAD list, the preamble of every run ('the sign-sensitive rel_proj are what this capture can support'), the trailer ('rel_proj is sign-SENSITIVE over the whole buffer'), and KL_REFUSAL ('Until then use top_dmax ... and rel_proj'). .agents/specs/quant-mach2-additive.md line 383 already specifies the 'top_dmax/rel_proj columns', so the implementation is what lags the committed spec. Found while adding a test that pins docs/USAGE.md to the tool's own output: the section repeats the promise, which is how the pin surfaced it. Repair: print both whole-buffer comparisons per step, red-first.

## Resolution

2026-11-18: repaired in the same flow, red-first. The per-step table now ends with rel_sumabs and rel_proj, computed by the two functions that had no caller, and it spells top1_flip and max_rank_disp in full instead of flip/rank so the table, the docstring and docs/USAGE.md use one vocabulary. The summary gained STEPS WITH rel_proj != 0 and LARGEST rel_proj. Red on the parent commit: tests/scripts/test_q4exp_logits_diff.py ran 32 cases with 4 failures and 1 error, including 'Lists differ: [dropped, appeared] != [rel_sumabs, rel_proj]' and the columns case naming ['max_rank_disp', 'rel_proj', 'rel_sumabs', 'top1_flip'] as columns the tool never prints. Green after the change: 32/32. Mutations: rel_proj forced to return 0.0 reds the two rel_ cases; --kl returning 1 reds the documented-exit-code case. Durable proof that the surface cannot silently lose the column again is the PublishedUsageSection control in the same file.
