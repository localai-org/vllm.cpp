ID: ISSUE-GH-1601
Title: **`SPEC-DRAFTER-CHAIN` cites llama.cpp by bare line number with no revision beside it.** Two sites landed by W1 at `31cefe631`: `include/vllm/config/speculative.h:34` cites `common/arg.cpp:3754-3763`, and `.agents/specs/drafter-chain.md` `## Upstream chain` cites that plus `common/speculative.cpp:2164-2186`. `.agents/oracles/llama-cpp.md` records `pin = 10bf611e533d81f739128304991c5e133c6aebd8`, `pin_label = b10451`, `pinned_on = 2026-08-16`, `gateable = no`, and neither citation names it. llama.cpp moves several times a day, so those ranges will name different code within weeks with nothing in the tree able to notice: `scripts/check-symbol-anchors.py` states outright that it cannot verify a LINE citation, the same gap that let `SPEC-DSPARK-QWEN3-ROUTING` carry a wrongly shifted anchor under a helper's name until a repair wave caught it by hand. **The exposure is bounded and stated**: W1 cites NO llama.cpp gate input — its rules are this engine's own document-shape decisions — so this is design context going stale, not a measurement resting on a moving target, which is why it is filed rather than fixed inside a wave scoped to a config field. Minimum repair: append `@ 10bf611e` to both citations and prefer a symbol over a line range where one exists (`common_speculative_n_max` in the same paragraph already needs nothing). Listed under `## Owed` in [drafter-chain.md](../../specs/drafter-chain.md). Found by a fresh review of `row/SPEC-DRAFTER-CHAIN-W1`
Row: SPEC-DRAFTER-CHAIN
State: UNKNOWN
Kind: gap
GitHub: 1601
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:566`

### Frozen archive evidence

> | [#1601](https://github.com/mudler/vllm.cpp/issues/1601) | `SPEC-DRAFTER-CHAIN` | **`SPEC-DRAFTER-CHAIN` cites llama.cpp by bare line number with no revision beside it.** Two sites landed by W1 at `31cefe631`: `include/vllm/config/speculative.h:34` cites `common/arg.cpp:3754-3763`, and `.agents/specs/drafter-chain.md` `## Upstream chain` cites that plus `common/speculative.cpp:2164-2186`. `.agents/oracles/llama-cpp.md` records `pin = 10bf611e533d81f739128304991c5e133c6aebd8`, `pin_label = b10451`, `pinned_on = 2026-08-16`, `gateable = no`, and neither citation names it. llama.cpp moves several times a day, so those ranges will name different code within weeks with nothing in the tree able to notice: `scripts/check-symbol-anchors.py` states outright that it cannot verify a LINE citation, the same gap that let `SPEC-DSPARK-QWEN3-ROUTING` carry a wrongly shifted anchor under a helper's name until a repair wave caught it by hand. **The exposure is bounded and stated**: W1 cites NO llama.cpp gate input — its rules are this engine's own document-shape decisions — so this is design context going stale, not a measurement resting on a moving target, which is why it is filed rather than fixed inside a wave scoped to a config field. Minimum repair: append `@ 10bf611e` to both citations and prefer a symbol over a line range where one exists (`common_speculative_n_max` in the same paragraph already needs nothing). Listed under `## Owed` in [drafter-chain.md](../../specs/drafter-chain.md). Found by a fresh review of `row/SPEC-DRAFTER-CHAIN-W1` | gap |

## Resolution

-
