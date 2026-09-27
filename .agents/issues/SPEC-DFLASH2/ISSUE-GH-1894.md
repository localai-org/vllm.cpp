ID: ISSUE-GH-1894
Title: **The DFlash2 runner fixture drafts a CONSTANT — `19 19 19` at all eight steps — so every drafted-token comparison through it is a tautology against a numerics change.** Found by #1890's mutation pass rather than by reading, with a one-off probe that printed what `DraftedBlocks` returns. Three mutations of the draft-block attention, each a genuinely WRONG attention (mask polarity forced causal; the paged K/V write neutralised; the read handed the store's `seq_lens` instead of the extended bound), left the comparison GREEN while the byte-for-byte op gate `test_qwen3_dflash_block_route` red on all of them — which is how the degeneracy was located. **Not only a W11 concern**: the landed `dflash2 runner (W8): the paged lane and the materialized lane draft identically` case compares the same constants, so its stated guarantee is not measured by it either. NOT degenerate under every perturbation — the D9-scalars case does observe a difference — so the fix is weights whose per-position argmax actually separates, applied without disturbing the W3/W4/W9/W10 cases that read the same fixture. NOT FIXED IN FLOW: changing the fixture's weights moves five landed cases at once and needs its own red-before evidence, a different unit of work from #1890. Owned by row `SPEC-DFLASH2` and listed under `## Owed` in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md)
Row: SPEC-DFLASH2
State: UNKNOWN
Kind: bug
GitHub: 1894
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:706`

### Frozen archive evidence

> | [#1894](https://github.com/mudler/vllm.cpp/issues/1894) | `SPEC-DFLASH2` | **The DFlash2 runner fixture drafts a CONSTANT — `19 19 19` at all eight steps — so every drafted-token comparison through it is a tautology against a numerics change.** Found by #1890's mutation pass rather than by reading, with a one-off probe that printed what `DraftedBlocks` returns. Three mutations of the draft-block attention, each a genuinely WRONG attention (mask polarity forced causal; the paged K/V write neutralised; the read handed the store's `seq_lens` instead of the extended bound), left the comparison GREEN while the byte-for-byte op gate `test_qwen3_dflash_block_route` red on all of them — which is how the degeneracy was located. **Not only a W11 concern**: the landed `dflash2 runner (W8): the paged lane and the materialized lane draft identically` case compares the same constants, so its stated guarantee is not measured by it either. NOT degenerate under every perturbation — the D9-scalars case does observe a difference — so the fix is weights whose per-position argmax actually separates, applied without disturbing the W3/W4/W9/W10 cases that read the same fixture. NOT FIXED IN FLOW: changing the fixture's weights moves five landed cases at once and needs its own red-before evidence, a different unit of work from #1890. Owned by row `SPEC-DFLASH2` and listed under `## Owed` in [dflash2-draft-block-fa2.md](../../specs/dflash2-draft-block-fa2.md) | bug |

## Resolution

-
