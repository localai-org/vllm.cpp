ID: ISSUE-GH-2036
Title: `DenseAlignFor` (`qwen3_5.cpp:2825-2849`) allocates five blocks and calls `d.b.Synchronize(d.q)` at `:2846` on an `M` miss, and `EnsureCtmp` (`cuda_marlin_dense.cu:74-89`) grows with `cudaMallocAsync` at `:85`, both with no `cudaStreamIsCapturing` refusal — unlike the six sibling shape-keyed caches that have one. Latent today (the cold step visits the same key), found while fixing #2029; owed under `## Owed` in [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md)
Row: ENG-CUDAGRAPH-BREAK
State: UNKNOWN
Kind: bug
GitHub: 2036
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:775`

### Frozen archive evidence

> | [#2036](https://github.com/mudler/vllm.cpp/issues/2036) | `ENG-CUDAGRAPH-BREAK` | `DenseAlignFor` (`qwen3_5.cpp:2825-2849`) allocates five blocks and calls `d.b.Synchronize(d.q)` at `:2846` on an `M` miss, and `EnsureCtmp` (`cuda_marlin_dense.cu:74-89`) grows with `cudaMallocAsync` at `:85`, both with no `cudaStreamIsCapturing` refusal — unlike the six sibling shape-keyed caches that have one. Latent today (the cold step visits the same key), found while fixing #2029; owed under `## Owed` in [cudagraph-pregrow-nonspec.md](../../specs/cudagraph-pregrow-nonspec.md) | bug |

## Resolution

-
