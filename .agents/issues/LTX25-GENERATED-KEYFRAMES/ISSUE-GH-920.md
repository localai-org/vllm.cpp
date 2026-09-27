ID: ISSUE-GH-920
Title: Generated keyframe slots — upstream `VideoGeneratedKeyframeSlots` (`conditioning/types/keyframe_slots.py:27-174` @ `fd4ded7f`), reached as `--num-generated-keyframes` (`ltx-pipelines/utils/args.py:833-844`) and documented at `ltx-pipelines/docs/conditioning.md:29-61` — have no REQUEST surface in this tree, so asking for them falls through the per-generation extras check and gets `unknown per-generation extra`, a message that says the family does not define the key and sends the reader looking for a typo. This is the ONLY upstream conditioning item that passes `marked=True` to `extend_keyframes_mask` (`:121` against `keyframe_cond.py:84-86`), so it is the only user-facing feature that puts #658's trained bias on a token other than the target's own first latent frame; `KeyframeInterpolationPipeline` does NOT (no `generated_keyframe` reference in its 362 lines, and absent from the feature's own applies-to list at `conditioning.md:47-51`). ONE blocker, the `GeneratedKeyframeLayout` READBACK: the slots are the OUTPUT, so they must be located by the layout the append recorded rather than assumed to trail, extracted before the extra tokens are trimmed, and each decoded as a STANDALONE one-frame clip (`ltx_core/types.py:269-272`). The token-append machinery is NOT a second blocker, and this row states that rather than restating the refusal it replaces: `c7cb59fbb` (#930) landed the seam and serves the LAST-frame supplied arm through it. Owed to THIS row rather than to #930 is a production caller for the `marked=true` branch of `Ltx2ExtendKeyframesMask` (`include/vllm/model_executor/models/ltx2_conditioning.h:175` @ `e5351776c`), whose only driver today is the unit case at `tests/vllm/models/test_ltx2_vae.cpp:2494` @ `e5351776c`. Spec [`ltx25-generated-keyframes.md`](../../specs/ltx25-generated-keyframes.md), campaign #644
Row: LTX25-GENERATED-KEYFRAMES
State: UNKNOWN
Kind: bug
GitHub: 920
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:246`

### Frozen archive evidence

> | [#920](https://github.com/mudler/vllm.cpp/issues/920) | `LTX25-GENERATED-KEYFRAMES` | Generated keyframe slots — upstream `VideoGeneratedKeyframeSlots` (`conditioning/types/keyframe_slots.py:27-174` @ `fd4ded7f`), reached as `--num-generated-keyframes` (`ltx-pipelines/utils/args.py:833-844`) and documented at `ltx-pipelines/docs/conditioning.md:29-61` — have no REQUEST surface in this tree, so asking for them falls through the per-generation extras check and gets `unknown per-generation extra`, a message that says the family does not define the key and sends the reader looking for a typo. This is the ONLY upstream conditioning item that passes `marked=True` to `extend_keyframes_mask` (`:121` against `keyframe_cond.py:84-86`), so it is the only user-facing feature that puts #658's trained bias on a token other than the target's own first latent frame; `KeyframeInterpolationPipeline` does NOT (no `generated_keyframe` reference in its 362 lines, and absent from the feature's own applies-to list at `conditioning.md:47-51`). ONE blocker, the `GeneratedKeyframeLayout` READBACK: the slots are the OUTPUT, so they must be located by the layout the append recorded rather than assumed to trail, extracted before the extra tokens are trimmed, and each decoded as a STANDALONE one-frame clip (`ltx_core/types.py:269-272`). The token-append machinery is NOT a second blocker, and this row states that rather than restating the refusal it replaces: `c7cb59fbb` (#930) landed the seam and serves the LAST-frame supplied arm through it. Owed to THIS row rather than to #930 is a production caller for the `marked=true` branch of `Ltx2ExtendKeyframesMask` (`include/vllm/model_executor/models/ltx2_conditioning.h:175` @ `e5351776c`), whose only driver today is the unit case at `tests/vllm/models/test_ltx2_vae.cpp:2494` @ `e5351776c`. Spec [`ltx25-generated-keyframes.md`](../../specs/ltx25-generated-keyframes.md), campaign #644 | bug |

## Resolution

-
