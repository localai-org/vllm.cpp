ID: ISSUE-GH-501
Title: `AlphaVecBf16TakesTwoLaunch` bounded a COUNT of ulp mismatches instead of their MAGNITUDE, and was RED on its first CUDA run at ~26% — the double-rounding population the bf16-D lever produces by construction. Replaced by a max-ulp bound (`<= 1`, and `<= 0` at a pow2 alpha), measured 0/1-ulp only over 2.17M words on GB10, spec [`perf-fp8-alpha-fold.md`](../../specs/perf-fp8-alpha-fold.md) §The bf16-vs-f32 divergence is DOUBLE ROUNDING
Row: PERF-27B-LMHEAD-FP4
State: UNKNOWN
Kind: bug
GitHub: 501
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:119`

### Frozen archive evidence

> | [#501](https://github.com/mudler/vllm.cpp/issues/501) | `PERF-27B-LMHEAD-FP4` | `AlphaVecBf16TakesTwoLaunch` bounded a COUNT of ulp mismatches instead of their MAGNITUDE, and was RED on its first CUDA run at ~26% — the double-rounding population the bf16-D lever produces by construction. Replaced by a max-ulp bound (`<= 1`, and `<= 0` at a pow2 alpha), measured 0/1-ulp only over 2.17M words on GB10, spec [`perf-fp8-alpha-fold.md`](../../specs/perf-fp8-alpha-fold.md) §The bf16-vs-f32 divergence is DOUBLE ROUNDING | bug |

## Resolution

-
