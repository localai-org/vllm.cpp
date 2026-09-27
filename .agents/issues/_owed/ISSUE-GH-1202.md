ID: ISSUE-GH-1202
Title: `Ltx2FuseLoraIntoTensor` (`src/vllm/model_executor/models/ltx2_lora.cpp:321-334`) computes the `(B * strength) @ A` LoRA product with a scalar single-threaded triple loop: one thread, no blocking, no SIMD, a non-inlined `vt::BF16ToF32` per multiply, and an inner operand `pair->a[k * cols + i]` striding by `cols` so every load in the innermost loop is its own cache line. Measured on `dgx` (GB10, 20 cores) loading the full/dev transformer (21,004,025,600 params) with the shipped 8.9 GB distilled adapter: three `gdb` stacks all reading `vt::BF16ToF32` <- `Ltx2FuseLoraIntoTensor` <- `Ltx2LoadDitFromSafetensors` <- `Ltx2VideoEngine::Load`, one thread at 99.9% of one core with 19 idle, and an f32 working set growing 9.432 -> 10.235 GiB over 300-629 s = **2.3% of one pass in 10.4 minutes**, cross-checked against the sum of `out*in*rank` over the 1660 targeted modules = 8.53e12 MAC, consistent with ~0.53 GFLOP/s. The operation is a rank-`r` GEMM and belongs on the `vt::` GEMM seam like every other projection in the tree; the arithmetic is already a correct mirror of `fuse_loras.py:103-116` (`B * strength` rounds to bf16 BEFORE the product, f32 accumulation, bf16 store) and only the execution strategy is wrong, so a replacement has a bit-exact oracle rather than a tolerance. Blocks every LoRA-bearing pipeline kind on the full model; `one_stage` is unaffected because upstream marks it `Full` with no adapter, which is why it is the only full-model arm that currently reaches generation. Owed by [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) `## Owed`, whose §5 already frames "why the decode is single-threaded and on the host"
Row: -
State: UNKNOWN
Kind: perf
GitHub: 1202
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:387`

### Frozen archive evidence

> | [#1202](https://github.com/mudler/vllm.cpp/issues/1202) | — | `Ltx2FuseLoraIntoTensor` (`src/vllm/model_executor/models/ltx2_lora.cpp:321-334`) computes the `(B * strength) @ A` LoRA product with a scalar single-threaded triple loop: one thread, no blocking, no SIMD, a non-inlined `vt::BF16ToF32` per multiply, and an inner operand `pair->a[k * cols + i]` striding by `cols` so every load in the innermost loop is its own cache line. Measured on `dgx` (GB10, 20 cores) loading the full/dev transformer (21,004,025,600 params) with the shipped 8.9 GB distilled adapter: three `gdb` stacks all reading `vt::BF16ToF32` <- `Ltx2FuseLoraIntoTensor` <- `Ltx2LoadDitFromSafetensors` <- `Ltx2VideoEngine::Load`, one thread at 99.9% of one core with 19 idle, and an f32 working set growing 9.432 -> 10.235 GiB over 300-629 s = **2.3% of one pass in 10.4 minutes**, cross-checked against the sum of `out*in*rank` over the 1660 targeted modules = 8.53e12 MAC, consistent with ~0.53 GFLOP/s. The operation is a rank-`r` GEMM and belongs on the `vt::` GEMM seam like every other projection in the tree; the arithmetic is already a correct mirror of `fuse_loras.py:103-116` (`B * strength` rounds to bf16 BEFORE the product, f32 accumulation, bf16 store) and only the execution strategy is wrong, so a replacement has a bit-exact oracle rather than a tolerance. Blocks every LoRA-bearing pipeline kind on the full model; `one_stage` is unaffected because upstream marks it `Full` with no adapter, which is why it is the only full-model arm that currently reaches generation. Owed by [`ltx25-decode-speed.md`](../../specs/ltx25-decode-speed.md) `## Owed`, whose §5 already frames "why the decode is single-threaded and on the host" | perf |

## Resolution

-
