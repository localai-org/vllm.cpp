ID: ISSUE-GH-1878
Title: Packed GDN decode diverges from the rollback arm within 950 greedy tokens on the bf16 35B (first difference token 33 request 2, token 213 request 3, reproduced exactly across pairs) with a CLEAN same-arm null -- a deterministic kernel-numerics difference (FLA cubin bf16 vs split F32 pair), the 27B near-tie class, recorded as a measured property of the lever pending a quality-at-depth or oracle-continuation disposition. Found by the GDN-MOE-PACKED-BA speed A/B; listed under `## Owed` in [gdn-moe-packed-ba.md](../../specs/gdn-moe-packed-ba.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1878
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:704`

### Frozen archive evidence

> | [#1878](https://github.com/mudler/vllm.cpp/issues/1878) | — | Packed GDN decode diverges from the rollback arm within 950 greedy tokens on the bf16 35B (first difference token 33 request 2, token 213 request 3, reproduced exactly across pairs) with a CLEAN same-arm null -- a deterministic kernel-numerics difference (FLA cubin bf16 vs split F32 pair), the 27B near-tie class, recorded as a measured property of the lever pending a quality-at-depth or oracle-continuation disposition. Found by the GDN-MOE-PACKED-BA speed A/B; listed under `## Owed` in [gdn-moe-packed-ba.md](../../specs/gdn-moe-packed-ba.md) | bug |

## Resolution

-
