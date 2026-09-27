ID: ISSUE-GH-930
Title: LTX-2.5's phase loop is fixed at one `Ltx2VideoTokenCount(vshape, 1)`, and that single limitation blocks THREE conditioning arms, not the two #930 was written against: reference video, the LAST-frame keyframe, and generated keyframe slots ([#920](https://github.com/mudler/vllm.cpp/issues/920)). Row `LTX25-TOKEN-APPEND` (spec [`ltx25-token-append.md`](../../specs/ltx25-token-append.md)) ports the two missing halves of the append — `extend_keyframes_mask` (`mask_utils.py:74-105`), which upstream's own docstring says EVERY appending item must call, and `clear_conditioning` (`tools.py:88-117`), which trims back to the target count and restores an ALL-ONES mask rather than the conditioned one — and lifts the last-frame keyframe as the demonstration. The attention mask is NOT the gap and no field is added for it: both ported video items pass a literal `attention_mask=None` (`keyframe_cond.py:68-76`, `reference_video_cond.py:88-96`) and the only route to a non-None mask is `ConditioningItemAttentionStrengthWrapper`, applied solely at `iclora_utils.py:169`. The sigma schedule must keep reading the TARGET count — `math.prod(latent.shape[2:])` (`schedulers.py:32 @ fd4ded7fa`) is the unpatchified target and cannot see an append — so the engine's `Ltx2SigmaSchedule(steps, video.tokens)` call, which sits AFTER the conditioning block (`src/vllm/multimodal/ltx2_video.cpp:1719 @ bc6433d1b`), re-shifts the whole schedule the moment anything appends. Reference video and generated slots stay refused: at `bc6433d1b` the reference refusal's LoRA-metadata cause is still true because PR [#938](https://github.com/mudler/vllm.cpp/pull/938) is open and unmerged
Row: LTX25-TOKEN-APPEND
State: UNKNOWN
Kind: feature
GitHub: 930
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:260`

### Frozen archive evidence

> | [#930](https://github.com/mudler/vllm.cpp/issues/930) | `LTX25-TOKEN-APPEND` | LTX-2.5's phase loop is fixed at one `Ltx2VideoTokenCount(vshape, 1)`, and that single limitation blocks THREE conditioning arms, not the two #930 was written against: reference video, the LAST-frame keyframe, and generated keyframe slots ([#920](https://github.com/mudler/vllm.cpp/issues/920)). Row `LTX25-TOKEN-APPEND` (spec [`ltx25-token-append.md`](../../specs/ltx25-token-append.md)) ports the two missing halves of the append — `extend_keyframes_mask` (`mask_utils.py:74-105`), which upstream's own docstring says EVERY appending item must call, and `clear_conditioning` (`tools.py:88-117`), which trims back to the target count and restores an ALL-ONES mask rather than the conditioned one — and lifts the last-frame keyframe as the demonstration. The attention mask is NOT the gap and no field is added for it: both ported video items pass a literal `attention_mask=None` (`keyframe_cond.py:68-76`, `reference_video_cond.py:88-96`) and the only route to a non-None mask is `ConditioningItemAttentionStrengthWrapper`, applied solely at `iclora_utils.py:169`. The sigma schedule must keep reading the TARGET count — `math.prod(latent.shape[2:])` (`schedulers.py:32 @ fd4ded7fa`) is the unpatchified target and cannot see an append — so the engine's `Ltx2SigmaSchedule(steps, video.tokens)` call, which sits AFTER the conditioning block (`src/vllm/multimodal/ltx2_video.cpp:1719 @ bc6433d1b`), re-shifts the whole schedule the moment anything appends. Reference video and generated slots stay refused: at `bc6433d1b` the reference refusal's LoRA-metadata cause is still true because PR [#938](https://github.com/mudler/vllm.cpp/pull/938) is open and unmerged | feature |

## Resolution

-
