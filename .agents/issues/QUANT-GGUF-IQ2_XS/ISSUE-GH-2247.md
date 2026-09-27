ID: ISSUE-GH-2247
Title: **Keep-quant `vec_dot` for IQ2_XS and IQ4_XS: 325.58 GiB, and the difference between the staged GLM-5.3-Flash artifact fitting `dgx:gpu0` and overflowing it 3.6x.** [#2245](https://github.com/mudler/vllm.cpp/pull/2245) gave both types a row DECODER, which is what moved the loader past `unknown ggml type id 17`. A decode-only type has no `vec_dot`, so `HasQuantDotKernel` is false and every GEMM weight of that type expands to bf16 at load. Measured from the artifact's own headers, all four shards and all 1412 tensors: **101.24 GiB on disk, 597.46 GiB as bf16**, an expansion of 5.9x, of which IQ2_XS alone is 53.33 -> 369.00 GiB and IQ4_XS 3.59 -> 13.50 GiB. Resident TODAY **426.72 GiB** against the ~119.63 GiB the box has, so it does not fit; with these two kernels **101.14 GiB**, which fits with 18.49 GiB of headroom. Every other encoding in the file already keeps its quantization, IQ3_XXS (`VecDotIQ3_XXSQ8_K`) included, so these two are the entire gap. Two rows in `src/vt/cpu/cpu_quant_dot.cpp` beside the fifteen already there, ported from the pinned llama.cpp `b10451` and gated BYTE-FOR-BYTE against the oracle's own kernel on real artifact bytes, because a `vec_dot` defect shows up as numeric drift and not as a crash. Owning rows `QUANT-GGUF-IQ2_XS` and `QUANT-GGUF-IQ4_XS` in [`quantization-matrix.md`](../../quantization-matrix.md), both carrying it as `C` = `-`; also recorded as O18 in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md)
Row: QUANT-GGUF-IQ2_XS
State: UNKNOWN
Kind: feature
GitHub: 2247
Mirror: MISSING
Availability: METADATA_ONLY
Created: UNKNOWN
Updated: UNKNOWN
Closed: UNKNOWN

## Problem

Archive: `.agents/completed/issue-index.md:873`

### Frozen archive evidence

> | [#2247](https://github.com/mudler/vllm.cpp/issues/2247) | `QUANT-GGUF-IQ2_XS` | **Keep-quant `vec_dot` for IQ2_XS and IQ4_XS: 325.58 GiB, and the difference between the staged GLM-5.3-Flash artifact fitting `dgx:gpu0` and overflowing it 3.6x.** [#2245](https://github.com/mudler/vllm.cpp/pull/2245) gave both types a row DECODER, which is what moved the loader past `unknown ggml type id 17`. A decode-only type has no `vec_dot`, so `HasQuantDotKernel` is false and every GEMM weight of that type expands to bf16 at load. Measured from the artifact's own headers, all four shards and all 1412 tensors: **101.24 GiB on disk, 597.46 GiB as bf16**, an expansion of 5.9x, of which IQ2_XS alone is 53.33 -> 369.00 GiB and IQ4_XS 3.59 -> 13.50 GiB. Resident TODAY **426.72 GiB** against the ~119.63 GiB the box has, so it does not fit; with these two kernels **101.14 GiB**, which fits with 18.49 GiB of headroom. Every other encoding in the file already keeps its quantization, IQ3_XXS (`VecDotIQ3_XXSQ8_K`) included, so these two are the entire gap. Two rows in `src/vt/cpu/cpu_quant_dot.cpp` beside the fifteen already there, ported from the pinned llama.cpp `b10451` and gated BYTE-FOR-BYTE against the oracle's own kernel on real artifact bytes, because a `vec_dot` defect shows up as numeric drift and not as a crash. Owning rows `QUANT-GGUF-IQ2_XS` and `QUANT-GGUF-IQ4_XS` in [`quantization-matrix.md`](../../quantization-matrix.md), both carrying it as `C` = `-`; also recorded as O18 in [`specs/glm5-next-flash.md`](../../specs/glm5-next-flash.md) | feature |

## Resolution

-
