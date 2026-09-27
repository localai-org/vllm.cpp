ID: ISSUE-GH-1195
Title: The transcription-only server path has no seam either half of an `--offload-config` document could reach, so [#1135](https://github.com/mudler/vllm.cpp/issues/1135) REFUSED the flag there instead of wiring it. `ParakeetTranscriber::FromDir` (`src/vllm/multimodal/parakeet_transcription.cpp:49`) builds no `EngineParams` and calls no `LoadedEngine::FromModelDir`; it reads its weights through `LoadParakeetForCTC` / `LoadParakeetTransducer`, so that path has no `SetWeightResidencyConfig` call, no `CreateWeightOffloader` call, no GGUF mapping and no expert slot store — no field of either half has a reader on it. Wiring it means first giving the transcription stack a loader seam that consults the process-global offloader and the residency config, which is larger than #1135 and belongs to whoever adds one. WHAT LANDED INSTEAD: the server parses `--offload-config` ONCE, ahead of the architecture branch, so a typo is refused at startup on every path, and a NON-EMPTY document on the transcription-only path aborts at startup naming the missing seam rather than being dropped in silence. Listed under `## Owed` in [`weight-residency-config.md`](../../specs/weight-residency-config.md)
Row: ARCH-ONE-SURFACE
State: UNKNOWN
Kind: gap
GitHub: 1195
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:375`

### Frozen archive evidence

> | [#1195](https://github.com/mudler/vllm.cpp/issues/1195) | `ARCH-ONE-SURFACE` | The transcription-only server path has no seam either half of an `--offload-config` document could reach, so [#1135](https://github.com/mudler/vllm.cpp/issues/1135) REFUSED the flag there instead of wiring it. `ParakeetTranscriber::FromDir` (`src/vllm/multimodal/parakeet_transcription.cpp:49`) builds no `EngineParams` and calls no `LoadedEngine::FromModelDir`; it reads its weights through `LoadParakeetForCTC` / `LoadParakeetTransducer`, so that path has no `SetWeightResidencyConfig` call, no `CreateWeightOffloader` call, no GGUF mapping and no expert slot store — no field of either half has a reader on it. Wiring it means first giving the transcription stack a loader seam that consults the process-global offloader and the residency config, which is larger than #1135 and belongs to whoever adds one. WHAT LANDED INSTEAD: the server parses `--offload-config` ONCE, ahead of the architecture branch, so a typo is refused at startup on every path, and a NON-EMPTY document on the transcription-only path aborts at startup naming the missing seam rather than being dropped in silence. Listed under `## Owed` in [`weight-residency-config.md`](../../specs/weight-residency-config.md) | gap |

## Resolution

-
