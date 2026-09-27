ID: ISSUE-GH-1135
Title: `--offload-config` does not reach three entry points, and two of them are server-side. It is parsed once in `server_main.cpp`, AFTER the architecture resolution, so the server's POOLING/embedding path (the `if (pooling_model)` block) and its transcription-only path build their `EngineParams` without it, and `vllm-cli` has no such flag at all. Not specific to the `vllm_cpp` residency extension: the MIRRORED `uva`/`prefetch` half is dropped on the same two paths and has been since before that key existed, so the new key inherits a pre-existing gap rather than introducing one. An embedding or transcription server started with `--offload-config` therefore places weights as though the flag were absent and says nothing; for the residency half that is the 370 GiB case, where the difference is whether the process fits in host RAM at all. Fixing the server half means moving the offload parse ahead of the architecture branch, which is `ENG-WEIGHT-OFFLOAD`'s surface as much as this row's; `vllm-cli` having no flag is a deliberate scope line for [#1110](https://github.com/mudler/vllm.cpp/issues/1110) rather than a defect, recorded together so a reader need not rediscover which of the three is which. FILED because the gap was listed under `## Owed` against [#1122](https://github.com/mudler/vllm.cpp/issues/1122), the review issue [#1119](https://github.com/mudler/vllm.cpp/pull/1119) closes, so on landing it would have had no open issue (#1133 L8). Documented in `docs/USAGE.md` beside the config form. Listed under `## Owed` in [`weight-residency-config.md`](../../specs/weight-residency-config.md)
Row: ENG-RESIDENCY-CONFIG
State: UNKNOWN
Kind: bug
GitHub: 1135
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:341`

### Frozen archive evidence

> | [#1135](https://github.com/mudler/vllm.cpp/issues/1135) | `ENG-RESIDENCY-CONFIG` | `--offload-config` does not reach three entry points, and two of them are server-side. It is parsed once in `server_main.cpp`, AFTER the architecture resolution, so the server's POOLING/embedding path (the `if (pooling_model)` block) and its transcription-only path build their `EngineParams` without it, and `vllm-cli` has no such flag at all. Not specific to the `vllm_cpp` residency extension: the MIRRORED `uva`/`prefetch` half is dropped on the same two paths and has been since before that key existed, so the new key inherits a pre-existing gap rather than introducing one. An embedding or transcription server started with `--offload-config` therefore places weights as though the flag were absent and says nothing; for the residency half that is the 370 GiB case, where the difference is whether the process fits in host RAM at all. Fixing the server half means moving the offload parse ahead of the architecture branch, which is `ENG-WEIGHT-OFFLOAD`'s surface as much as this row's; `vllm-cli` having no flag is a deliberate scope line for [#1110](https://github.com/mudler/vllm.cpp/issues/1110) rather than a defect, recorded together so a reader need not rediscover which of the three is which. FILED because the gap was listed under `## Owed` against [#1122](https://github.com/mudler/vllm.cpp/issues/1122), the review issue [#1119](https://github.com/mudler/vllm.cpp/pull/1119) closes, so on landing it would have had no open issue (#1133 L8). Documented in `docs/USAGE.md` beside the config form. Listed under `## Owed` in [`weight-residency-config.md`](../../specs/weight-residency-config.md) | bug |

## Resolution

-
