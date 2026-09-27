ID: ISSUE-GH-1011
Title: `memory_efficient_decode.py` is **ON BY DEFAULT** upstream (`memory_efficient: bool = True`, `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1059 @ fd4ded7f2`, sole caller `:1090-1095`) and unported here — but it is **NOT** the unattributed ~59 GiB, and [`ltx25-resolution-envelope.md`](../../specs/ltx25-resolution-envelope.md):353 calling it "the obvious" candidate is refuted: summing every intermediate the conv decoder ever produces at 448x256/25f in f32 with **no frees at all** gives 9.649 GiB (COMPUTED), 6x short. Still owed for byte traffic and memory format — workspace buffers (`memory_efficient_decode.py:108-114 @ fd4ded7f2`) replacing a per-conv `repeat`+`concatenate` (`model/video_vae/convolution.py:306-311`), in-place chunked Conv3d (`:122-204`), free-before-conv (`:234-248`), and NDHWC (`:617-627`). The shipped LTX-2.5 conv VAE is non-causal, so all four apply. Lever 4. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md)
Row: -
State: UNKNOWN
Kind: feature
GitHub: 1011
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:288`

### Frozen archive evidence

> | [#1011](https://github.com/mudler/vllm.cpp/issues/1011) | — | `memory_efficient_decode.py` is **ON BY DEFAULT** upstream (`memory_efficient: bool = True`, `packages/ltx-pipelines/src/ltx_pipelines/utils/blocks.py:1059 @ fd4ded7f2`, sole caller `:1090-1095`) and unported here — but it is **NOT** the unattributed ~59 GiB, and [`ltx25-resolution-envelope.md`](../../specs/ltx25-resolution-envelope.md):353 calling it "the obvious" candidate is refuted: summing every intermediate the conv decoder ever produces at 448x256/25f in f32 with **no frees at all** gives 9.649 GiB (COMPUTED), 6x short. Still owed for byte traffic and memory format — workspace buffers (`memory_efficient_decode.py:108-114 @ fd4ded7f2`) replacing a per-conv `repeat`+`concatenate` (`model/video_vae/convolution.py:306-311`), in-place chunked Conv3d (`:122-204`), free-before-conv (`:234-248`), and NDHWC (`:617-627`). The shipped LTX-2.5 conv VAE is non-causal, so all four apply. Lever 4. Listed under `## Owed` in [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) | feature |

## Resolution

-
