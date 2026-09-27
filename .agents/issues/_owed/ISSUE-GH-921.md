ID: ISSUE-GH-921
Title: The res_2s DENOISING LOOP (`ltx-pipelines utils/samplers.py:206-447` @ `fd4ded7f2`) is unported, so `TI2VidTwoStagesHQPipeline` cannot be served. What exists is one substep's SDE arithmetic: `Ltx2Res2sSdeCoeff`/`Ltx2Res2sStep` (`ltx2_pipeline.cpp:307-360 @ 5a0ffe9e3`, the two functions in full) mirror `Res2sDiffusionStep` (`diffusion_steps.py:118-190`) and are gated. Absent are the `phi`/`get_res2s_coefficients` exponential integrator (`utils/res2s.py:4-62`), the SECOND transformer evaluation per step at `sub_sigma = sqrt(sigma * sigma_next)` (`samplers.py:315` and `samplers.py:380-386`, spelt out because a bare `:NN` after a res2s.py citation reads as res2s.py) against our once-per-step loop (`ltx2_video.cpp:1735 @ 5a0ffe9e3`, with its single forward at `ltx2_video.cpp:1813-1817 @ 5a0ffe9e3`), the bong anchor refinement (`samplers.py:357-364`), and any `Ltx2StepperKind` enumerator to select it. The sampler IS the HQ variant, so this arm must refuse by name rather than substitute Euler and render something plausible that is quietly not HQ. Listed under `## Owed` in [`ltx25-resolution-envelope.md`](../../specs/ltx25-resolution-envelope.md)
Row: -
State: UNKNOWN
Kind: feature
GitHub: 921
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:248`

### Frozen archive evidence

> | [#921](https://github.com/mudler/vllm.cpp/issues/921) | — | The res_2s DENOISING LOOP (`ltx-pipelines utils/samplers.py:206-447` @ `fd4ded7f2`) is unported, so `TI2VidTwoStagesHQPipeline` cannot be served. What exists is one substep's SDE arithmetic: `Ltx2Res2sSdeCoeff`/`Ltx2Res2sStep` (`ltx2_pipeline.cpp:307-360 @ 5a0ffe9e3`, the two functions in full) mirror `Res2sDiffusionStep` (`diffusion_steps.py:118-190`) and are gated. Absent are the `phi`/`get_res2s_coefficients` exponential integrator (`utils/res2s.py:4-62`), the SECOND transformer evaluation per step at `sub_sigma = sqrt(sigma * sigma_next)` (`samplers.py:315` and `samplers.py:380-386`, spelt out because a bare `:NN` after a res2s.py citation reads as res2s.py) against our once-per-step loop (`ltx2_video.cpp:1735 @ 5a0ffe9e3`, with its single forward at `ltx2_video.cpp:1813-1817 @ 5a0ffe9e3`), the bong anchor refinement (`samplers.py:357-364`), and any `Ltx2StepperKind` enumerator to select it. The sampler IS the HQ variant, so this arm must refuse by name rather than substitute Euler and render something plausible that is quietly not HQ. Listed under `## Owed` in [`ltx25-resolution-envelope.md`](../../specs/ltx25-resolution-envelope.md) | feature |

## Resolution

-
