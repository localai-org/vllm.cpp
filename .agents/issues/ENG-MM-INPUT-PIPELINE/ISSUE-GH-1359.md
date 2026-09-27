ID: ISSUE-GH-1359
Title: Muse Glimmer's perception encoder is held in HOST F32 where the checkpoint ships bf16: `MuseGlimmerVisionWeights`, `MuseGlimmerVisionAdapterWeights` and `MuseGlimmerVisionTower::projection` are all `std::vector<float>` (`include/vllm/model_executor/models/muse_glimmer_vision.h:106-118`), so the 809 vision tensors that are 3.580 GiB on disk cost **7.161 GiB resident** — twice the checkpoint's own — with no annotation naming a reason, against AGENTS.md's "Inherit vLLM defaults". Measured from the two shard headers of `/mnt/nas_share/checkpoints/muse-glimmer-30b` (1436 tensors, 55.463 GiB total; the encoder is 6.45% of it). Nobody noticed for the reason that paragraph names: a token gate cannot detect a dtype that is too wide, so `test_muse_glimmer_vision` and the wiring gate are both correct and both blind. Qwen3-VL's tower has the same shape (`qwen3_vl_vision.h:76-82`), and `qwen3_vl_vision.h:106-114` already records the host-f32 form dominating encode time without recording that it is also a polarity departure. NOT fixed in flow: narrowing a tower's storage dtype changes numerics on every path that reads it, so it needs its own spec, a bf16-vs-f32 comparison against the reference and its own gate — the surprising-fix path, not an in-flow repair — and it is orthogonal to #607 L3, which removes the tower entirely at zero limits rather than narrowing it. Recorded in [`multimodal-track.md`](../../specs/multimodal-track.md) §1.5 L3 beside the RSS threshold, which is stated against 7.161 GiB for this reason. Owned by `ENG-MM-INPUT-PIPELINE`; listed under `## Owed`
Row: ENG-MM-INPUT-PIPELINE
State: UNKNOWN
Kind: bug
GitHub: 1359
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:649`

### Frozen archive evidence

> | [#1359](https://github.com/mudler/vllm.cpp/issues/1359) | `ENG-MM-INPUT-PIPELINE` | Muse Glimmer's perception encoder is held in HOST F32 where the checkpoint ships bf16: `MuseGlimmerVisionWeights`, `MuseGlimmerVisionAdapterWeights` and `MuseGlimmerVisionTower::projection` are all `std::vector<float>` (`include/vllm/model_executor/models/muse_glimmer_vision.h:106-118`), so the 809 vision tensors that are 3.580 GiB on disk cost **7.161 GiB resident** — twice the checkpoint's own — with no annotation naming a reason, against AGENTS.md's "Inherit vLLM defaults". Measured from the two shard headers of `/mnt/nas_share/checkpoints/muse-glimmer-30b` (1436 tensors, 55.463 GiB total; the encoder is 6.45% of it). Nobody noticed for the reason that paragraph names: a token gate cannot detect a dtype that is too wide, so `test_muse_glimmer_vision` and the wiring gate are both correct and both blind. Qwen3-VL's tower has the same shape (`qwen3_vl_vision.h:76-82`), and `qwen3_vl_vision.h:106-114` already records the host-f32 form dominating encode time without recording that it is also a polarity departure. NOT fixed in flow: narrowing a tower's storage dtype changes numerics on every path that reads it, so it needs its own spec, a bf16-vs-f32 comparison against the reference and its own gate — the surprising-fix path, not an in-flow repair — and it is orthogonal to #607 L3, which removes the tower entirely at zero limits rather than narrowing it. Recorded in [`multimodal-track.md`](../../specs/multimodal-track.md) §1.5 L3 beside the RSS threshold, which is stated against 7.161 GiB for this reason. Owned by `ENG-MM-INPUT-PIPELINE`; listed under `## Owed` | bug |

## Resolution

-
