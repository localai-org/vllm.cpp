ID: ISSUE-GH-1853
Title: **`PENDING` on a `dgx:gpu0` lease: the arithmetic-perturbation reference render that would make [#1743](https://github.com/mudler/vllm.cpp/issues/1743)'s criterion RELATIVE.** §11 of [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md) relocates the pixel verdict onto **correspondence** and **incoherence**, which discriminate a degraded render from a separated trajectory. It deliberately does NOT answer the other half of #1743: is the swap's divergence no worse than this pipeline's own divergence under an arithmetic perturbation of comparable size. That needs one further arm - the **naive** path at `768x448/49f`, seed `20260820`, on §10.7's pinned binary and checkpoints, with a bounded `+/-1` bf16 ULP dither injected at the DiT attention output at the `8.6e-05` to `3.7e-04` per-element flip rate §10.2 derives - after which `D(flash, naive) <= D(dither, naive)` is a bound with NO chosen constant. **No lease was authorised for #1743, so this is PENDING and not skipped.** The cross-build `baseline-20260820` vs `naive` figure (mean |delta| **9.452407**, LARGER than the swap's **6.414156**) is NOT that control and is not used as one: the binary lineage differs, so every other commit between `a50c57d69` and `3e2961ef0` sits inside it, which §10.8 already records. NOT FIXED IN FLOW: it needs a GPU lease this work does not have. Owned by row `LTX25-DIT-ATTN-FLASH` and listed under `## Owed`
Row: LTX25-DIT-ATTN-FLASH
State: UNKNOWN
Kind: bug
GitHub: 1853
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:698`

### Frozen archive evidence

> | [#1853](https://github.com/mudler/vllm.cpp/issues/1853) | `LTX25-DIT-ATTN-FLASH` | **`PENDING` on a `dgx:gpu0` lease: the arithmetic-perturbation reference render that would make [#1743](https://github.com/mudler/vllm.cpp/issues/1743)'s criterion RELATIVE.** §11 of [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md) relocates the pixel verdict onto **correspondence** and **incoherence**, which discriminate a degraded render from a separated trajectory. It deliberately does NOT answer the other half of #1743: is the swap's divergence no worse than this pipeline's own divergence under an arithmetic perturbation of comparable size. That needs one further arm - the **naive** path at `768x448/49f`, seed `20260820`, on §10.7's pinned binary and checkpoints, with a bounded `+/-1` bf16 ULP dither injected at the DiT attention output at the `8.6e-05` to `3.7e-04` per-element flip rate §10.2 derives - after which `D(flash, naive) <= D(dither, naive)` is a bound with NO chosen constant. **No lease was authorised for #1743, so this is PENDING and not skipped.** The cross-build `baseline-20260820` vs `naive` figure (mean \|delta\| **9.452407**, LARGER than the swap's **6.414156**) is NOT that control and is not used as one: the binary lineage differs, so every other commit between `a50c57d69` and `3e2961ef0` sits inside it, which §10.8 already records. NOT FIXED IN FLOW: it needs a GPU lease this work does not have. Owned by row `LTX25-DIT-ATTN-FLASH` and listed under `## Owed` | bug |

## Resolution

-
