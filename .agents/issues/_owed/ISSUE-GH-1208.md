ID: ISSUE-GH-1208
Title: `Linear` in the LTX-2.5 text tower (`src/vllm/model_executor/models/ltx2_text_encoder.cpp:60-71`) is a scalar single-threaded triple loop with a `double` accumulator and both operands widened per multiply, so it cannot reach the f32 FMA path even if the compiler vectorised it. On the full model's caption projection (in_features on the order of 1.9e5) cost is `rows * out_features * in_features` and it dominates the pre-generation wall: measured on `dgx`, resident memory went flat at t≈150 s and stayed **byte-identical for 1073 s** with stacks in the text feature extractor. Because `ltx2-gen` prints nothing between load and completion ([#1010](https://github.com/mudler/vllm.cpp/issues/1010) owns that), this presents to a user as a hang rather than as slow arithmetic, and it is why an earlier probe's "reached Generate" claim was retracted — the trace does not support it. TWO separate defects: the execution strategy, which belongs on the `vt::` GEMM seam; and the `double` accumulator, which is NOT a mirror — `torch.nn.functional.linear` on f32 inputs accumulates in f32, and the comment directly above this function cites `F.linear` as its reference, so the widening diverges from the oracle it names, cannot be bit-compared against upstream, and hides reduction-order differences an f32 accumulator would expose. Per the dtype-polarity rule an f64 accumulator kept deliberately needs a one-line reason beside it. Every LTX-2.5 pipeline kind goes through the text tower, so this is on the critical path of all of them including `one_stage`. Owed by [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) `## Owed`
Row: -
State: UNKNOWN
Kind: perf
GitHub: 1208
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:388`

### Frozen archive evidence

> | [#1208](https://github.com/mudler/vllm.cpp/issues/1208) | — | `Linear` in the LTX-2.5 text tower (`src/vllm/model_executor/models/ltx2_text_encoder.cpp:60-71`) is a scalar single-threaded triple loop with a `double` accumulator and both operands widened per multiply, so it cannot reach the f32 FMA path even if the compiler vectorised it. On the full model's caption projection (in_features on the order of 1.9e5) cost is `rows * out_features * in_features` and it dominates the pre-generation wall: measured on `dgx`, resident memory went flat at t≈150 s and stayed **byte-identical for 1073 s** with stacks in the text feature extractor. Because `ltx2-gen` prints nothing between load and completion ([#1010](https://github.com/mudler/vllm.cpp/issues/1010) owns that), this presents to a user as a hang rather than as slow arithmetic, and it is why an earlier probe's "reached Generate" claim was retracted — the trace does not support it. TWO separate defects: the execution strategy, which belongs on the `vt::` GEMM seam; and the `double` accumulator, which is NOT a mirror — `torch.nn.functional.linear` on f32 inputs accumulates in f32, and the comment directly above this function cites `F.linear` as its reference, so the widening diverges from the oracle it names, cannot be bit-compared against upstream, and hides reduction-order differences an f32 accumulator would expose. Per the dtype-polarity rule an f64 accumulator kept deliberately needs a one-line reason beside it. Every LTX-2.5 pipeline kind goes through the text tower, so this is on the critical path of all of them including `one_stage`. Owed by [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) `## Owed` | perf |

## Resolution

-
