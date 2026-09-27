ID: ISSUE-GH-1094
Title: `HDRICLoraPipeline` (`hdr_ic_lora.py:229` @ `fd4ded7f`) is absent with no refusal and no `Ltx2UnportedPipelineFeature` marker. It is the only upstream pipeline returning LINEAR HDR rather than display-referred pixels, so the gap is colour science and not only scheduling. Five citations exist outside `.agents/` and none is an implementation: comments at `ltx2_lora.h:168-169`, `ltx2_pipeline.h:588`, `test_ltx2_lora.cpp:481`, `test_ltx2_video.cpp:542`, plus one string literal inside a `Fail(...)` argument at `ltx2_lora.cpp:246`; `git grep -i HDRICLora` returns zero hits tree-wide. Blocked on (a) a LogC3 / ACEScct decode tail: upstream `ltx-core/hdr.py:37-43` (ARRI EI-800 constants), `:53-65` (compress/decompress), `:82-84` (`HDRTransfer`), `:140-172` (`to_linear`, `to_hdr_linear`), applied at `hdr_ic_lora.py:624` and selected from the adapter's own safetensors metadata (`:178-209`). `git grep -i "logc3|HDRTransfer|to_hdr_linear|acescct"` returns zero product-code hits here, with `git grep -i yuv` as the control (live ffmpeg argv at `minimax_h3_mux.cpp:80`), and the exclusion is already deliberate: [`ltx25-retire-dead-arms.md`](../../specs/ltx25-retire-dead-arms.md):167 classifies scene-linear HDR colour as "no - colour science". (b) TWO artifacts not on the NAS: `--hdr-lora` (`required=True`, `:833`) from repo `Lightricks/LTX-2.3-22b-IC-LoRA-HDR`, which upstream names only by repo, and `--text-embeddings` (`:834`), since this pipeline loads no text encoder at all (`:275-281`). (c) Per-phase stage-2 tiling and IC-LoRA toggles (`:102-104`, consumed `:485-500`), which our fixed two-phase recipe shape cannot express
Row: ROAD-V1-LTX25
State: UNKNOWN
Kind: feature
GitHub: 1094
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:322`

### Frozen archive evidence

> | [#1094](https://github.com/mudler/vllm.cpp/issues/1094) | `ROAD-V1-LTX25` | `HDRICLoraPipeline` (`hdr_ic_lora.py:229` @ `fd4ded7f`) is absent with no refusal and no `Ltx2UnportedPipelineFeature` marker. It is the only upstream pipeline returning LINEAR HDR rather than display-referred pixels, so the gap is colour science and not only scheduling. Five citations exist outside `.agents/` and none is an implementation: comments at `ltx2_lora.h:168-169`, `ltx2_pipeline.h:588`, `test_ltx2_lora.cpp:481`, `test_ltx2_video.cpp:542`, plus one string literal inside a `Fail(...)` argument at `ltx2_lora.cpp:246`; `git grep -i HDRICLora` returns zero hits tree-wide. Blocked on (a) a LogC3 / ACEScct decode tail: upstream `ltx-core/hdr.py:37-43` (ARRI EI-800 constants), `:53-65` (compress/decompress), `:82-84` (`HDRTransfer`), `:140-172` (`to_linear`, `to_hdr_linear`), applied at `hdr_ic_lora.py:624` and selected from the adapter's own safetensors metadata (`:178-209`). `git grep -i "logc3\|HDRTransfer\|to_hdr_linear\|acescct"` returns zero product-code hits here, with `git grep -i yuv` as the control (live ffmpeg argv at `minimax_h3_mux.cpp:80`), and the exclusion is already deliberate: [`ltx25-retire-dead-arms.md`](../../specs/ltx25-retire-dead-arms.md):167 classifies scene-linear HDR colour as "no - colour science". (b) TWO artifacts not on the NAS: `--hdr-lora` (`required=True`, `:833`) from repo `Lightricks/LTX-2.3-22b-IC-LoRA-HDR`, which upstream names only by repo, and `--text-embeddings` (`:834`), since this pipeline loads no text encoder at all (`:275-281`). (c) Per-phase stage-2 tiling and IC-LoRA toggles (`:102-104`, consumed `:485-500`), which our fixed two-phase recipe shape cannot express | feature |

## Resolution

-
