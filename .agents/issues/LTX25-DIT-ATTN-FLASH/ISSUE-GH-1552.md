ID: ISSUE-GH-1552
Title: **A caller that never opts into a fast attention op is never told, so sweep every remaining `vt::Attention` call site.** The generalisation of [#1549](https://github.com/mudler/vllm.cpp/issues/1549), filed separately because the mechanism is not LTX-specific. `kAttention` is frozen on the naive kernel so text decode stays byte-identical (`src/vt/cuda/cuda_ops.cu:3120-3122`) -- correct, and untouched here. The consequence is that `kAttentionDenseFast` / `kAttentionDenseFlash` / `kAttentionDenseFa2` are separate ops each caller must name, with no shape routing and no fallback notice, so a caller that never opts in gets correct output at up to ~500x the cost and NOTHING detects it: the goldens pass because the output is right, no refusal fires because the op is registered, and `GetOpProviderStats` counts the naive selection as a success because it is one. The only symptom is a wall clock the model may have no gate for. OWES: (1) every non-decode `vt::Attention` call site enumerated with its head_dim, sequence length and dense/non-causal eligibility; (2) per eligible site, either a routing change with its own reachability proof and numerics gate or a recorded reason to stay; (3) a decision between leaving it caller-opt-in, warning once on a large-token `kAttention` selection on CUDA, or shape-routing `kAttention` itself -- only the third removes the failure mode and only the third risks the byte-identity guarantee, so it needs its own spec. Same class as AGENTS.md "Nothing lands dead", inverted: there a capability lands unreached, here a FASTER capability lands unreached and the slow one is correct enough that nobody looks. NOT fixed in flow. Owner: row `LTX25-DIT-ATTN-FLASH`, under `## Owed` in [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md)
Row: LTX25-DIT-ATTN-FLASH
State: UNKNOWN
Kind: bug
GitHub: 1552
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:560`

### Frozen archive evidence

> | [#1552](https://github.com/mudler/vllm.cpp/issues/1552) | `LTX25-DIT-ATTN-FLASH` | **A caller that never opts into a fast attention op is never told, so sweep every remaining `vt::Attention` call site.** The generalisation of [#1549](https://github.com/mudler/vllm.cpp/issues/1549), filed separately because the mechanism is not LTX-specific. `kAttention` is frozen on the naive kernel so text decode stays byte-identical (`src/vt/cuda/cuda_ops.cu:3120-3122`) -- correct, and untouched here. The consequence is that `kAttentionDenseFast` / `kAttentionDenseFlash` / `kAttentionDenseFa2` are separate ops each caller must name, with no shape routing and no fallback notice, so a caller that never opts in gets correct output at up to ~500x the cost and NOTHING detects it: the goldens pass because the output is right, no refusal fires because the op is registered, and `GetOpProviderStats` counts the naive selection as a success because it is one. The only symptom is a wall clock the model may have no gate for. OWES: (1) every non-decode `vt::Attention` call site enumerated with its head_dim, sequence length and dense/non-causal eligibility; (2) per eligible site, either a routing change with its own reachability proof and numerics gate or a recorded reason to stay; (3) a decision between leaving it caller-opt-in, warning once on a large-token `kAttention` selection on CUDA, or shape-routing `kAttention` itself -- only the third removes the failure mode and only the third risks the byte-identity guarantee, so it needs its own spec. Same class as AGENTS.md "Nothing lands dead", inverted: there a capability lands unreached, here a FASTER capability lands unreached and the slow one is correct enough that nobody looks. NOT fixed in flow. Owner: row `LTX25-DIT-ATTN-FLASH`, under `## Owed` in [`ltx25-dit-attn-flash.md`](../../specs/ltx25-dit-attn-flash.md) | bug |

## Resolution

-
