ID: ISSUE-GH-1573
Title: **`AttentionDenseFlash`'s repaired head_dim bound is proven by arithmetic and not by a launch.** The pure-host bound (`AttentionDenseFlashMaxHeadDim`, 192 bf16 / 96 f32) is unit-tested and mutated on a CPU box, but nothing there executes `LaunchAttentionDenseFlash`, so the CUDA case asserting that head_dim 256 in f32 REFUSES and names `vt::AttentionDenseFast` — and that head_dim 96, exactly on the cap, still runs — emits a loud PENDING message and returns. Its reachability mutation (drop the `VT_CHECK` on the bound, require the case to go RED) needs the same device. NOT fixed in the flow that filed it: `dgx:gpu0` was held by the developer for the whole row and AGENTS.md forbids reaching a fleet device outside a lease, so the honest report is PENDING on a named resource rather than a skip wearing a pass. Owed under `## Owed` in [attention-rung-visibility.md](../../specs/attention-rung-visibility.md) (risk R1)
Row: -
State: UNKNOWN
Kind: gap
GitHub: 1573
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:551`

### Frozen archive evidence

> | [#1573](https://github.com/mudler/vllm.cpp/issues/1573) | — | **`AttentionDenseFlash`'s repaired head_dim bound is proven by arithmetic and not by a launch.** The pure-host bound (`AttentionDenseFlashMaxHeadDim`, 192 bf16 / 96 f32) is unit-tested and mutated on a CPU box, but nothing there executes `LaunchAttentionDenseFlash`, so the CUDA case asserting that head_dim 256 in f32 REFUSES and names `vt::AttentionDenseFast` — and that head_dim 96, exactly on the cap, still runs — emits a loud PENDING message and returns. Its reachability mutation (drop the `VT_CHECK` on the bound, require the case to go RED) needs the same device. NOT fixed in the flow that filed it: `dgx:gpu0` was held by the developer for the whole row and AGENTS.md forbids reaching a fleet device outside a lease, so the honest report is PENDING on a named resource rather than a skip wearing a pass. Owed under `## Owed` in [attention-rung-visibility.md](../../specs/attention-rung-visibility.md) (risk R1) | gap |

## Resolution

-
