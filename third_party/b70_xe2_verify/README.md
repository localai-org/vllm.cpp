# B70 Xe2 shared-KV verification donor

This optional Torch-free Q2–Q5 FP8 verification policy derives from
`vllm-project/vllm-xpu-kernels` commit
`6d92b1bfbf32767ecda8e819613eb151e70030ad` plus the production
`shared-kv-verification.patch` in the pinned local Python deployment.
The root Apache-2.0 license is in `LICENSE`; copied collective and kernel
headers retain their BSD-3-Clause notices.

The source is built with `VLLM_CPP_XPU_XE2_PREFILL=ON` or
`VLLM_CPP_XPU_XE2_GDN=ON`, the pinned
SYCL-TLA checkout `87f6850680a580654b9ea2c80dbc01aeb36ad231` and
oneAPI 2026.1.1. Packed Q2–Q5 verification is off by default at runtime. Set
`VT_XPU_XE2_VERIFY=1` for the eligible automatic route or
`VT_XPU_ATTENTION=verify` for an explicit diagnostic; unsupported shapes
fall back to the existing attention implementation.

`paged_decode.hpp` is narrowed to the one required policy and no longer
contains the Torch dispatcher. `src/vt/xpu/xpu_attention_verify_xe2.cpp`
adapts VT's head-contiguous FP8 K/V pages and Q layout to the donor's
packed Q layout. No Python or Torch code is loaded for C++ inference.

The explicit verifier now admits page1600/1664 and planar or head-interleaved
rows. C1/Q4 at active length4100 is byte-exact against the executed pinned
page1600 original fixture, including NaN page-tail poison and output aliasing.
Packed metadata is private `{0,1}`, while the public logical `{0,4}` remains
unchanged. Inactive V tail operands are zeroed before DPAS for causal
verification too. The original icpx math model is scoped to the verifier
translation unit. Derived original Q2/Q3/Q5 and Q4 page-boundary/32K
operator inputs match across planar/interleaved/padded layouts. Focused
unsupported-form tests and C1 raw graph replay qualify shared scratch, fresh
metadata validation and retirement. The native C4/Q4 path uses one batched donor invocation and private
physical offsets `{0,1,2,3,4}`. It requires the existing uniform host query
offsets; the existing GPU metadata check proves their agreement with device
offsets before writes. Distinct request pages/lengths, permutation, poisoned
padding, aliases and strided output copies match the executed original. Missing
or ragged host offsets retain the generic route; the speculative classification
hint alone does not admit a packed batch. Raw C4/C1 graphs qualify shared scratch,
fresh metadata, request permutation/input mutation and retirement across queues.
The dense model graph policy now tracks route settings, every KV binding/layout
and partition/page boundaries. Requested C4 verification uses a page-end grid
bound, without raising request/context limits. Actual model-owner retirement
and emitted-cycle qualification remain pending. These are isolated operator
proofs; automatic verification stays off by default.

`src/vt/xpu/xpu_attention_decode_xe2.cpp` uses a separate short C1 policy:
FP16 Q/output, Hq24/Hkv4/D256, unit E4M3 scales, page1600/1664 and
max sequence length1–960. The automatic B70 route preserves FP16
unnormalized probabilities before XMX P*V and uses one split. It consumes
the caller's Q/output directly and masks invalid V tail operands before
DPAS; no Q packing, scale upload or host readback is needed. The original
icpx math model is scoped to this translation unit, as it is for the packed
verifier after its independent same-input proof.

The `flash_attention_v2/collective/{fmha_fusion,copy_block_slm}.hpp` and
`cutlass/util/packed_stride.hpp` helpers are unmodified copies from the
same SYCL-TLA commit87f6850680a580654b9ea2c80dbc01aeb36ad231, extracted
from the already available local Git objects. They retain their BSD-3-Clause
notices. Unused GEMM adapter/reference utility includes were removed from
the native launcher; the full utility/tool dependency tree is unnecessary.
