ID: ISSUE-GH-1887
Title: **Three citations of `include/vt/ops.h:3304-3306` in [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md) were stale and asserted the OPPOSITE of the sentence they supported.** Found by the fresh review of [#1871](https://github.com/mudler/vllm.cpp/pull/1871) while that change was adding a FOURTH citation of the same anchor. At `def85d285` those lines sit inside the `AttentionRelPos` doc comment and read "Reductions are strictly sequential per output element => thread-count independent and byte-reproducible", while the three sites (§4 numerics, §10.2, and §11.3 -- the ratified criterion section the whole pixel lane rests on) cite them for the claim that `vt::Attention` and `vt::AttentionDenseFlash` "differ only in association". Correct at `90e8c3c85`/`ff8f72807`, stale by `c4ba829a3`. **Correct anchors, each verified unique by phrase**: `ops.h:3315-3316` (`Fast` NOT bit-identical to `Attention`, different head_dim partial-sum grouping), `ops.h:3328-3329` (`Flash` order UNCHANGED from `Fast`, bit-identical), `ops.h:3381-3382` (`Fa2` NOT bit-identical to `Fast`/`Flash`, `mma.sync` reassociates QK^T and PV). **A second defect the same reading exposed**: the anchor was being used to claim `flash` and `fa2` share ONE reassociated order, and they do not -- they are TWO DIFFERENT reassociations, so the supportable claim is a shared CLASS, which is what §12.6 measures. FIXED IN FLOW in #1871: all three pre-existing citations repaired and the fourth corrected before publication. Owned by row `LTX25-DIT-ATTN-FLASH`
Row: LTX25-DIT-ATTN-FLASH
State: UNKNOWN
Kind: bug
GitHub: 1887
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:716`

### Frozen archive evidence

> | [#1887](https://github.com/mudler/vllm.cpp/issues/1887) | `LTX25-DIT-ATTN-FLASH` | **Three citations of `include/vt/ops.h:3304-3306` in [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md) were stale and asserted the OPPOSITE of the sentence they supported.** Found by the fresh review of [#1871](https://github.com/mudler/vllm.cpp/pull/1871) while that change was adding a FOURTH citation of the same anchor. At `def85d285` those lines sit inside the `AttentionRelPos` doc comment and read "Reductions are strictly sequential per output element => thread-count independent and byte-reproducible", while the three sites (§4 numerics, §10.2, and §11.3 -- the ratified criterion section the whole pixel lane rests on) cite them for the claim that `vt::Attention` and `vt::AttentionDenseFlash` "differ only in association". Correct at `90e8c3c85`/`ff8f72807`, stale by `c4ba829a3`. **Correct anchors, each verified unique by phrase**: `ops.h:3315-3316` (`Fast` NOT bit-identical to `Attention`, different head_dim partial-sum grouping), `ops.h:3328-3329` (`Flash` order UNCHANGED from `Fast`, bit-identical), `ops.h:3381-3382` (`Fa2` NOT bit-identical to `Fast`/`Flash`, `mma.sync` reassociates QK^T and PV). **A second defect the same reading exposed**: the anchor was being used to claim `flash` and `fa2` share ONE reassociated order, and they do not -- they are TWO DIFFERENT reassociations, so the supportable claim is a shared CLASS, which is what §12.6 measures. FIXED IN FLOW in #1871: all three pre-existing citations repaired and the fourth corrected before publication. Owned by row `LTX25-DIT-ATTN-FLASH` | bug |

## Resolution

-
