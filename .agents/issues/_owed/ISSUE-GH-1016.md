ID: ISSUE-GH-1016
Title: The LTX-2.5 loaders **never call `MaybeReleaseSourcePages`**: 0 hits in `src/vllm/model_executor/models/ltx2_loader.cpp` and `src/vllm/multimodal/ltx2_video.cpp @ 332aed738`. Positive control: the symbol appears in **15** other files under `src/vllm @ 332aed738` (`muse_glimmer_weights.cpp`, `phi_weights.cpp`, `gemma4_weights.cpp`, `nemotron_h_weights.cpp`, `kimi_linear_weights.cpp`, `qwen3_5_dense_weights.cpp`, `olmo2_weights.cpp` among them), so the symbol and path set are right and the calls genuinely are not there. It belongs in the device staging loop `ltx2_loader.cpp:738-756 @ 332aed738`, which walks one tensor at a time and `Synchronize`s at `:749`, so each source range is provably dead per iteration. On GB10 the staged copy and the file pages share one 119 GiB pool. Unlike [#1015](https://github.com/mudler/vllm.cpp/issues/1015) this is NOT excluded as a contributor to [#1014](https://github.com/mudler/vllm.cpp/issues/1014), because `MemAvailable` discounts reclaimable pages and the `Anonymous`-vs-`Rss_File` split has never been measured here. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1016
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:292`

### Frozen archive evidence

> | [#1016](https://github.com/mudler/vllm.cpp/issues/1016) | — | The LTX-2.5 loaders **never call `MaybeReleaseSourcePages`**: 0 hits in `src/vllm/model_executor/models/ltx2_loader.cpp` and `src/vllm/multimodal/ltx2_video.cpp @ 332aed738`. Positive control: the symbol appears in **15** other files under `src/vllm @ 332aed738` (`muse_glimmer_weights.cpp`, `phi_weights.cpp`, `gemma4_weights.cpp`, `nemotron_h_weights.cpp`, `kimi_linear_weights.cpp`, `qwen3_5_dense_weights.cpp`, `olmo2_weights.cpp` among them), so the symbol and path set are right and the calls genuinely are not there. It belongs in the device staging loop `ltx2_loader.cpp:738-756 @ 332aed738`, which walks one tensor at a time and `Synchronize`s at `:749`, so each source range is provably dead per iteration. On GB10 the staged copy and the file pages share one 119 GiB pool. Unlike [#1015](https://github.com/mudler/vllm.cpp/issues/1015) this is NOT excluded as a contributor to [#1014](https://github.com/mudler/vllm.cpp/issues/1014), because `MemAvailable` discounts reclaimable pages and the `Anonymous`-vs-`Rss_File` split has never been measured here. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | bug |

## Resolution

-
