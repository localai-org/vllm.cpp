ID: ISSUE-GH-1039
Title: LTX-2.5 T2A combined its guidance passes in VELOCITY space. Upstream hands the denoiser an `X0Model` (`ltx-pipelines/utils/blocks.py:480-482 @ fd4ded7f`) whose `forward` returns `to_denoised(latent, v, timesteps)` (`ltx-core/model/transformer/model.py:601-604`, `to_denoised` at `ltx-core/utils.py:39-52`), so `_guided_denoise` combines DENOISED tensors — `all_v, all_a = transformer(...)` at `utils/denoisers.py:188` and `audio_guider.calculate(...)` at `:203`. `Ltx2T2aGenerate` combined raw DiT velocities and applied `ToDenoised` once afterwards. `MultiModalGuider.calculate`'s LINEAR terms are invariant under `x0 = latent - sigma*v`, so the two forms agree exactly while `rescale_scale == 0`; the RESCALE branch (`guiders.py:268-271`) is not, because upstream's `std(x0_cond)/std(x0_pred)` scales the whole x0 to `f*(latent - sigma*v)` where scaling the velocity gives `latent - sigma*f*v` — a difference of `(f - 1) * latent`, non-zero wherever the latent is, which on this path is everywhere. `rescale_scale = 0.7` is the SHIPPED T2A default (`utils/constants.py:63`, `utils/args.py:1101-1106`), so every default render took the divergent branch. No gate saw it: the forward counts, perturbed blocks, latent absmax, waveform length, channel count and sample rate are identical between the two forms. FOUND by review of PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032) and FIXED in flow on the same branch, before it landed. The VIDEO arm is unaffected and the reason is recorded rather than assumed: `Ltx2MultiModalGuidance` has exactly ONE production caller and the joint driver runs a single unguided forward per step. Spec [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md)
Row: LTX25-T2A-ONE-STAGE
State: UNKNOWN
Kind: bug
GitHub: 1039
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:303`

### Frozen archive evidence

> | [#1039](https://github.com/mudler/vllm.cpp/issues/1039) | `LTX25-T2A-ONE-STAGE` | LTX-2.5 T2A combined its guidance passes in VELOCITY space. Upstream hands the denoiser an `X0Model` (`ltx-pipelines/utils/blocks.py:480-482 @ fd4ded7f`) whose `forward` returns `to_denoised(latent, v, timesteps)` (`ltx-core/model/transformer/model.py:601-604`, `to_denoised` at `ltx-core/utils.py:39-52`), so `_guided_denoise` combines DENOISED tensors — `all_v, all_a = transformer(...)` at `utils/denoisers.py:188` and `audio_guider.calculate(...)` at `:203`. `Ltx2T2aGenerate` combined raw DiT velocities and applied `ToDenoised` once afterwards. `MultiModalGuider.calculate`'s LINEAR terms are invariant under `x0 = latent - sigma*v`, so the two forms agree exactly while `rescale_scale == 0`; the RESCALE branch (`guiders.py:268-271`) is not, because upstream's `std(x0_cond)/std(x0_pred)` scales the whole x0 to `f*(latent - sigma*v)` where scaling the velocity gives `latent - sigma*f*v` — a difference of `(f - 1) * latent`, non-zero wherever the latent is, which on this path is everywhere. `rescale_scale = 0.7` is the SHIPPED T2A default (`utils/constants.py:63`, `utils/args.py:1101-1106`), so every default render took the divergent branch. No gate saw it: the forward counts, perturbed blocks, latent absmax, waveform length, channel count and sample rate are identical between the two forms. FOUND by review of PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032) and FIXED in flow on the same branch, before it landed. The VIDEO arm is unaffected and the reason is recorded rather than assumed: `Ltx2MultiModalGuidance` has exactly ONE production caller and the joint driver runs a single unguided forward per step. Spec [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md) | bug |

## Resolution

-
