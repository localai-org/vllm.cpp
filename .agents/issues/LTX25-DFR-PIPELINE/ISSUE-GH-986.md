ID: ISSUE-GH-986
Title: `DFRPipeline` (`dfr_pipeline.py` + `dfr_layout.py` @ `fd4ded7f`) has no representation in this tree — `git grep -i dfr` over `src include tests examples docs` returns ZERO product hits, against 87 `ltx2` hits in `ltx2.cpp` alone as the positive control ([#604](https://github.com/mudler/vllm.cpp/issues/604)). It matters beyond its own feature because it is the **ONLY** upstream consumer of the temporal x2 latent upsampler this project already ships: [`ltx25-temporal-upsampler.md`](../../specs/ltx25-temporal-upsampler.md) section 7 records the operator as ported, loader-parsed and gated but *"not reachable from any shipped pipeline"*, and `docs/FEATURES.md` carries that as `Temporal x2 ups gated, UNDRIVEN`. No issue tracked that state — `undriven` returned zero hits across open and closed issues. DFR also needs the generated-keyframe-slot READBACK that [#920](https://github.com/mudler/vllm.cpp/issues/920) refused by name and left owed after it CLOSED, so the debt had a spec bullet and no open issue; it needs the LAYOUT and the EXTRACTION but **not** the standalone single-frame decode, because DFR never decodes its slots — it hands them to the spatial upsampler (`dfr_pipeline.py:348`) and feeds them back as `initial_keyframes` (`:364`). Spec [`ltx25-dfr-pipeline.md`](../../specs/ltx25-dfr-pipeline.md). Campaign [#644](https://github.com/mudler/vllm.cpp/issues/644)
Row: LTX25-DFR-PIPELINE
State: UNKNOWN
Kind: feature
GitHub: 986
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:272`

### Frozen archive evidence

> | [#986](https://github.com/mudler/vllm.cpp/issues/986) | `LTX25-DFR-PIPELINE` | `DFRPipeline` (`dfr_pipeline.py` + `dfr_layout.py` @ `fd4ded7f`) has no representation in this tree — `git grep -i dfr` over `src include tests examples docs` returns ZERO product hits, against 87 `ltx2` hits in `ltx2.cpp` alone as the positive control ([#604](https://github.com/mudler/vllm.cpp/issues/604)). It matters beyond its own feature because it is the **ONLY** upstream consumer of the temporal x2 latent upsampler this project already ships: [`ltx25-temporal-upsampler.md`](../../specs/ltx25-temporal-upsampler.md) section 7 records the operator as ported, loader-parsed and gated but *"not reachable from any shipped pipeline"*, and `docs/FEATURES.md` carries that as `Temporal x2 ups gated, UNDRIVEN`. No issue tracked that state — `undriven` returned zero hits across open and closed issues. DFR also needs the generated-keyframe-slot READBACK that [#920](https://github.com/mudler/vllm.cpp/issues/920) refused by name and left owed after it CLOSED, so the debt had a spec bullet and no open issue; it needs the LAYOUT and the EXTRACTION but **not** the standalone single-frame decode, because DFR never decodes its slots — it hands them to the spatial upsampler (`dfr_pipeline.py:348`) and feeds them back as `initial_keyframes` (`:364`). Spec [`ltx25-dfr-pipeline.md`](../../specs/ltx25-dfr-pipeline.md). Campaign [#644](https://github.com/mudler/vllm.cpp/issues/644) | feature |

## Resolution

-
