# B70 Xe2 native GDN experiment

The native SYCL GDN kernel body and `gdn_attn_utils.h` are adapted from
`vllm-project/vllm-xpu-kernels` release branch `release/0.1.15.4`, commit
`ddf336d86e3c8602888572a3502f951abd51df12` (Apache-2.0; the unchanged donor
license is in `LICENSE`). `gemm.hpp`
retains its Intel BSD-3-Clause notice. The donor branch pins Intel SYCL-TLA
`87f6850680a580654b9ea2c80dbc01aeb36ad231`, which is also the local
Xe2 attention dependency. This source is a version-matched reference branch;
its exact correspondence to the installed Python wheel has not been proven.

`chunk_gated_delta_rule_kernels_xe2.hpp` omits the donor's PyTorch dispatcher
and its Torch include. The C++ wrapper is `src/vt/xpu/xpu_gdn_xe2.cpp`.
Its preparation stage reads VT's already normalized Q/K and computed G/beta:
G/beta are transposed from token-major to head-major, G is summed per chunk,
and Q receives the required `1/sqrt(128)` factor. The five native SYCL stages
then run on VT's own queue and FP32 state. No Python, PyTorch, or Triton
runtime is used by this route. A separate bounded 160 MiB workspace keeps
the existing 16/32 MiB GDN fallback untouched.

The prototype selects single-sequence P4096, P8192 or P16384 with
F16/Hv48/Hk16/D128 and requires `VT_XPU_GDN_NATIVE=1`. Longer inputs are
processed as sequential 4K macro segments with the same bounded scratch and
the FP32 state carried forward. Other inputs use the existing path. The 4K
and 8K model results are recorded under `docs/bench-evidence`; wider default
qualification remains open.
