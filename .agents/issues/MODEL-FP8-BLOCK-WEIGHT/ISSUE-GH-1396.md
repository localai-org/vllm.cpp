ID: ISSUE-GH-1396
Title: Eight comments and one spec `## Owed` asserted that the block-wise FP8 forward wiring was still owed, after it landed. All were last written by M3 (`09597106e`, MODEL-FP8-BLOCK-WEIGHT) and none was revisited when M4 (`281b4bc76`, the linear method and the dense forward that reads the weight), M5 (`489a9a4c0`, the mainloop-scaled CUTLASS kernel) and M6 (`836c13c35`, the merged `gate_up` and QKV) landed. The tree disagreed with them in two directions: `qwen3_5.cpp` calls the block-scaled GEMMs at TEN sites -- the attention `o_proj`; q/k/v once on the split path, where one `project` lambda serves q, k and v alike, and once more on the merged one; the GDN `in_proj`'s `qkv` and `z` halves; the GDN `out_proj` in each of its three block arms (`GdnBlock`, `GdnBlockPagedMixedSpec`, `GdnBlockPaged`); and the dense MLP's merged `gate_up` and its `down_proj` -- which is eight `MatmulFp8BlockScaledD`, one `MatmulFp8BlockMergedD` and one `Fp8BlockGateUpSwiGLUD`, so `NOTHING CONSUMES THIS YET` was false; and `RefuseUnrunnableQwen3_5DenseFp8Block` had already been narrowed by M4 to refuse a DEVICE with no block-scaled GEMM rather than the weight, so `Deleted by M5` and `Milestone M5 removes this` were false in the other direction -- M5 narrowed the refusal to a CUDA arch outside `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a) rather than deleting it. Both available readings of the stale text are costly: re-implement a delivered milestone, or refuse to use a working arm. FIXED IN FLOW, comment and spec text only, no behaviour change. The change deliberately does NOT narrow the real debt, which is unchanged and recorded in [`vt-matmul-fp8-block-cuda.md`](../../specs/vt-matmul-fp8-block-cuda.md) `## Owed`: the CUDA kernel has never executed on hardware and there is no token gate against `Qwen/Qwen3.8-27B-FP8`. No gate can hold this class -- no checker here compares a comment against the code it annotates -- so the reviewer's check is those ten call sites and the body of `ModelRegistry::Prepare`
Row: MODEL-FP8-BLOCK-WEIGHT
State: UNKNOWN
Kind: bug
GitHub: 1396
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:488`

### Frozen archive evidence

> | [#1396](https://github.com/mudler/vllm.cpp/issues/1396) | `MODEL-FP8-BLOCK-WEIGHT` | Eight comments and one spec `## Owed` asserted that the block-wise FP8 forward wiring was still owed, after it landed. All were last written by M3 (`09597106e`, MODEL-FP8-BLOCK-WEIGHT) and none was revisited when M4 (`281b4bc76`, the linear method and the dense forward that reads the weight), M5 (`489a9a4c0`, the mainloop-scaled CUTLASS kernel) and M6 (`836c13c35`, the merged `gate_up` and QKV) landed. The tree disagreed with them in two directions: `qwen3_5.cpp` calls the block-scaled GEMMs at TEN sites -- the attention `o_proj`; q/k/v once on the split path, where one `project` lambda serves q, k and v alike, and once more on the merged one; the GDN `in_proj`'s `qkv` and `z` halves; the GDN `out_proj` in each of its three block arms (`GdnBlock`, `GdnBlockPagedMixedSpec`, `GdnBlockPaged`); and the dense MLP's merged `gate_up` and its `down_proj` -- which is eight `MatmulFp8BlockScaledD`, one `MatmulFp8BlockMergedD` and one `Fp8BlockGateUpSwiGLUD`, so `NOTHING CONSUMES THIS YET` was false; and `RefuseUnrunnableQwen3_5DenseFp8Block` had already been narrowed by M4 to refuse a DEVICE with no block-scaled GEMM rather than the weight, so `Deleted by M5` and `Milestone M5 removes this` were false in the other direction -- M5 narrowed the refusal to a CUDA arch outside `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a) rather than deleting it. Both available readings of the stale text are costly: re-implement a delivered milestone, or refuse to use a working arm. FIXED IN FLOW, comment and spec text only, no behaviour change. The change deliberately does NOT narrow the real debt, which is unchanged and recorded in [`vt-matmul-fp8-block-cuda.md`](../../specs/vt-matmul-fp8-block-cuda.md) `## Owed`: the CUDA kernel has never executed on hardware and there is no token gate against `Qwen/Qwen3.8-27B-FP8`. No gate can hold this class -- no checker here compares a comment against the code it annotates -- so the reviewer's check is those ten call sites and the body of `ModelRegistry::Prepare` | bug |

## Resolution

-
