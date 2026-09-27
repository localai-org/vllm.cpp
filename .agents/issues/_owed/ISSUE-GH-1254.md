ID: ISSUE-GH-1254
Title: Once [#1202](https://github.com/mudler/vllm.cpp/issues/1202) routed the LoRA delta product through `vt::Matmul`, the majority of what a fused tensor costs is the loop that adds the delta BACK into the weight: `src/vllm/model_executor/models/ltx2_lora.cpp::Ltx2FuseLoraIntoTensor`'s bf16 branch, one thread and three out-of-line conversions per element. Measured with a same-binary A/B on the production fuser at the shipped geometry (`4096 x 450 x 4096`, 20-core Zen 5, Release, `-ffp-contract=off`, median of 5): the whole fuse is 0.1242 s, and the same call at `rank = 1` — same output size, negligible GEMM — is **0.0733 s, so the aggregator zero-fill plus the add-back loop is 59% of it** and the GEMM is ~0.046 s. Before #1202 that loop was 0.5% of the call and correctly ignored. `vt::Add` already carries the exact contract the bf16 branch needs ("computed in f32, rounded on store; `out` may alias `a`"; `include/vt/ops.h`), so the fix is the same shape #1202 took, gated the same way. The **f32 branch is NOT a match and must not be folded in with it**: it rounds the sum through bf16 before an f32 store to mirror `deltas.add_(weight)` on a bf16 aggregator followed by `.to(dtype=weight.dtype)` (`fuse_loras.py:67-68`), and `vt::Add` with an f32 output would skip that rounding and be silently more precise than the oracle — which `test_ltx2_lora`'s "the f32 target branch rounds through the bf16 accumulator" case exists to catch. Whether that branch keeps a recorded exception or the seam grows a form for it belongs in the row's spec. Owed by [`ltx25-lora-fuse-seam.md`](../../specs/ltx25-lora-fuse-seam.md) `## Owed`
Row: -
State: UNKNOWN
Kind: perf
GitHub: 1254
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:405`

### Frozen archive evidence

> | [#1254](https://github.com/mudler/vllm.cpp/issues/1254) | — | Once [#1202](https://github.com/mudler/vllm.cpp/issues/1202) routed the LoRA delta product through `vt::Matmul`, the majority of what a fused tensor costs is the loop that adds the delta BACK into the weight: `src/vllm/model_executor/models/ltx2_lora.cpp::Ltx2FuseLoraIntoTensor`'s bf16 branch, one thread and three out-of-line conversions per element. Measured with a same-binary A/B on the production fuser at the shipped geometry (`4096 x 450 x 4096`, 20-core Zen 5, Release, `-ffp-contract=off`, median of 5): the whole fuse is 0.1242 s, and the same call at `rank = 1` — same output size, negligible GEMM — is **0.0733 s, so the aggregator zero-fill plus the add-back loop is 59% of it** and the GEMM is ~0.046 s. Before #1202 that loop was 0.5% of the call and correctly ignored. `vt::Add` already carries the exact contract the bf16 branch needs ("computed in f32, rounded on store; `out` may alias `a`"; `include/vt/ops.h`), so the fix is the same shape #1202 took, gated the same way. The **f32 branch is NOT a match and must not be folded in with it**: it rounds the sum through bf16 before an f32 store to mirror `deltas.add_(weight)` on a bf16 aggregator followed by `.to(dtype=weight.dtype)` (`fuse_loras.py:67-68`), and `vt::Add` with an f32 output would skip that rounding and be silently more precise than the oracle — which `test_ltx2_lora`'s "the f32 target branch rounds through the bf16 accumulator" case exists to catch. Whether that branch keeps a recorded exception or the seam grows a form for it belongs in the row's spec. Owed by [`ltx25-lora-fuse-seam.md`](../../specs/ltx25-lora-fuse-seam.md) `## Owed` | perf |

## Resolution

-
