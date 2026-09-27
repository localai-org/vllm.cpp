ID: ISSUE-GH-1050
Title: The guider rescale's `std` comment states a consequence that cannot exist. `src/vllm/model_executor/models/ltx2_pipeline.cpp:505-506 @ c1fe35592`, repeated at `include/vllm/model_executor/models/ltx2_pipeline.h:319-322`, says torch's `std` is the UNBIASED (N-1) estimator by default and "the biased one would be a small, everywhere, resolution-dependent gain error that no shape or finiteness check can see". `factor_raw` is `unbiased_std(cond) / unbiased_std(pred)`, two `std`s over the SAME `count`, so the divisor cancels exactly: `sqrt(ss_c/(n-1))/sqrt(ss_p/(n-1)) == sqrt(ss_c/ss_p) == sqrt(ss_c/n)/sqrt(ss_p/n)`. There is no gain error, small or otherwise, and nothing about it is resolution-dependent; the two forms differ only by f32 rounding in the divide. Worth a record rather than a silent correction because the comment tells the next reader a gate is needed there and it is not: the fresh review of [#1039](https://github.com/mudler/vllm.cpp/issues/1039) mutated the estimator to the biased form and it survived — correctly, because it is an IDENTITY — and a survivor at that site otherwise reads as a blind instrument and costs another investigation. The CODE is right as written and should stay `unbiased_std`, because the name is what mirrors torch even where the ratio does not care; the COMMENT is the defect. Pre-existing from `cefacd2d0` (#641). Found repairing that review on PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032) and out of scope there under its explicit exclusions. Listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md)
Row: -
State: UNKNOWN
Kind: bug
GitHub: 1050
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:306`

### Frozen archive evidence

> | [#1050](https://github.com/mudler/vllm.cpp/issues/1050) | — | The guider rescale's `std` comment states a consequence that cannot exist. `src/vllm/model_executor/models/ltx2_pipeline.cpp:505-506 @ c1fe35592`, repeated at `include/vllm/model_executor/models/ltx2_pipeline.h:319-322`, says torch's `std` is the UNBIASED (N-1) estimator by default and "the biased one would be a small, everywhere, resolution-dependent gain error that no shape or finiteness check can see". `factor_raw` is `unbiased_std(cond) / unbiased_std(pred)`, two `std`s over the SAME `count`, so the divisor cancels exactly: `sqrt(ss_c/(n-1))/sqrt(ss_p/(n-1)) == sqrt(ss_c/ss_p) == sqrt(ss_c/n)/sqrt(ss_p/n)`. There is no gain error, small or otherwise, and nothing about it is resolution-dependent; the two forms differ only by f32 rounding in the divide. Worth a record rather than a silent correction because the comment tells the next reader a gate is needed there and it is not: the fresh review of [#1039](https://github.com/mudler/vllm.cpp/issues/1039) mutated the estimator to the biased form and it survived — correctly, because it is an IDENTITY — and a survivor at that site otherwise reads as a blind instrument and costs another investigation. The CODE is right as written and should stay `unbiased_std`, because the name is what mirrors torch even where the ratio does not care; the COMMENT is the defect. Pre-existing from `cefacd2d0` (#641). Found repairing that review on PR [#1032](https://github.com/mudler/vllm.cpp/pull/1032) and out of scope there under its explicit exclusions. Listed under `## Owed` in [`ltx25-t2a-one-stage.md`](../../specs/ltx25-t2a-one-stage.md) | bug |

## Resolution

-
