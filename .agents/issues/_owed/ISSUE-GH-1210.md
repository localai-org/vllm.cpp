ID: ISSUE-GH-1210
Title: The two-stage LoRA rebind cost that `src/vllm/multimodal/ltx2_video.cpp:2843-2851` records as "UNMEASURED on real weights", with a later perf row owning the number, is now measured. A two-stage recipe loads FUSED, phase 0 asks `Ltx2PhaseLoraScope::kNoAdapters` and `Ltx2RebindDitLoras` un-fuses, phase 1 asks `kAllAdapters` and re-fuses — so the load-time fusion is **provably wasted**, undone before any denoise step runs, and the DiT is left fused so the next render pays the same two again. At [#1202](https://github.com/mudler/vllm.cpp/issues/1202)'s measured ~0.53 GFLOP/s each pass is hours, and a two-stage full-model render pays three of them before the first step. TWO independent fixes, not one change: making the fusion fast (#1202) shrinks the constant but leaves the wasted round trip; separately, `Ltx2PipelineRecipe::phases` is available before `Load` runs, so the load could honour phase 0's scope and skip the fuse/un-fuse entirely. The terminal fused state is chosen rather than forced and is worth revisiting in the same change. Affects `a2vid_two_stage` and `ti2vid_two_stage`; `one_stage` pays nothing. NOT CLAIMED: the wall-clock figures are a rate measured over a 10.4-minute window and extrapolated, not a completed pass — no two-stage full-model render has completed, so the end-to-end number stays open; the rate, thread count and stack attribution are measured. Owed by [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) `## Owed`
Row: -
State: UNKNOWN
Kind: perf
GitHub: 1210
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:389`

### Frozen archive evidence

> | [#1210](https://github.com/mudler/vllm.cpp/issues/1210) | — | The two-stage LoRA rebind cost that `src/vllm/multimodal/ltx2_video.cpp:2843-2851` records as "UNMEASURED on real weights", with a later perf row owning the number, is now measured. A two-stage recipe loads FUSED, phase 0 asks `Ltx2PhaseLoraScope::kNoAdapters` and `Ltx2RebindDitLoras` un-fuses, phase 1 asks `kAllAdapters` and re-fuses — so the load-time fusion is **provably wasted**, undone before any denoise step runs, and the DiT is left fused so the next render pays the same two again. At [#1202](https://github.com/mudler/vllm.cpp/issues/1202)'s measured ~0.53 GFLOP/s each pass is hours, and a two-stage full-model render pays three of them before the first step. TWO independent fixes, not one change: making the fusion fast (#1202) shrinks the constant but leaves the wasted round trip; separately, `Ltx2PipelineRecipe::phases` is available before `Load` runs, so the load could honour phase 0's scope and skip the fuse/un-fuse entirely. The terminal fused state is chosen rather than forced and is worth revisiting in the same change. Affects `a2vid_two_stage` and `ti2vid_two_stage`; `one_stage` pays nothing. NOT CLAIMED: the wall-clock figures are a rate measured over a 10.4-minute window and extrapolated, not a completed pass — no two-stage full-model render has completed, so the end-to-end number stays open; the rate, thread count and stack attribution are measured. Owed by [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) `## Owed` | perf |

## Resolution

-
