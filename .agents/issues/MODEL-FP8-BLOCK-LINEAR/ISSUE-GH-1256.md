ID: ISSUE-GH-1256
Title: `main` went RED at `09597106e` (#1189 M3): `LoadQwen3_5DenseGdn` builds its `TensorExists` as `[] (const std::string&) { return true; }` (`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:679`), a stub that answers YES for every name. That was harmless while the only reader of `has` was a dtype probe that went on to `get` the tensor and would throw on a name that was not there. M3's config/tensor cross-check is a different kind of reader: `IsFp8BlockProjection` asks `has(proj + ".weight_scale_inv")` and never fetches it, so on a checkpoint with no block-wise scale the stub invents one, the `!block.block_quant` guard sees tensors and config disagree, and the load is refused with a message about a tensor that does not exist. `test_qwen27_dense_forward` and `test_qwen27_dense_forward_glue_fuse_off` have been red on `main` since that commit; the scheduled CI baseline had not run past it, so the red was invisible in `main-baseline.py`. FIXED in #1189 M4's flow by asking the resolver instead of asserting: `TensorResolver` throws on a missing tensor, so a `try`/`catch` probe is the honest answer and the only one available at this seam. 9/9 cases and 583 assertions green after, 8/9 and 563 before, so the repair is measured rather than assumed. The general lesson is the one this index keeps relearning: a predicate that cannot say NO is not an instrument, and it reads as a passing probe right up until someone asks it a question whose answer matters
Row: MODEL-FP8-BLOCK-LINEAR
State: UNKNOWN
Kind: bug
GitHub: 1256
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:406`

### Frozen archive evidence

> | [#1256](https://github.com/mudler/vllm.cpp/issues/1256) | `MODEL-FP8-BLOCK-LINEAR` | `main` went RED at `09597106e` (#1189 M3): `LoadQwen3_5DenseGdn` builds its `TensorExists` as `[] (const std::string&) { return true; }` (`src/vllm/model_executor/models/qwen3_5_dense_weights.cpp:679`), a stub that answers YES for every name. That was harmless while the only reader of `has` was a dtype probe that went on to `get` the tensor and would throw on a name that was not there. M3's config/tensor cross-check is a different kind of reader: `IsFp8BlockProjection` asks `has(proj + ".weight_scale_inv")` and never fetches it, so on a checkpoint with no block-wise scale the stub invents one, the `!block.block_quant` guard sees tensors and config disagree, and the load is refused with a message about a tensor that does not exist. `test_qwen27_dense_forward` and `test_qwen27_dense_forward_glue_fuse_off` have been red on `main` since that commit; the scheduled CI baseline had not run past it, so the red was invisible in `main-baseline.py`. FIXED in #1189 M4's flow by asking the resolver instead of asserting: `TensorResolver` throws on a missing tensor, so a `try`/`catch` probe is the honest answer and the only one available at this seam. 9/9 cases and 583 assertions green after, 8/9 and 563 before, so the repair is measured rather than assumed. The general lesson is the one this index keeps relearning: a predicate that cannot say NO is not an instrument, and it reads as a passing probe right up until someone asks it a question whose answer matters | bug |

## Resolution

-
