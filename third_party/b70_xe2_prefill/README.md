# B70 Xe2 FP8 prefill donor

This optional, Torch-free policy is copied from `vllm-project/vllm-xpu-kernels`
commit `6d92b1bfbf32767ecda8e819613eb151e70030ad`. The root Apache-2.0
license is in `LICENSE`; the copied collective and kernel headers retain their
BSD-3-Clause notices. The Python reference binary used for the 4096-token
operator replay was `vllm_xpu_kernels/_vllm_fa2_C.abi3.so` from package
`0.1.15.4`, SHA-256
`28c36760256c4065e1eabd5ba6ffd3fa181073dafe372b35a3e2952e86d262fe`.
The exact source-to-binary correspondence of that DSO is not independently
established.

The policy requires Intel SYCL-TLA commit
`87f6850680a580654b9ea2c80dbc01aeb36ad231` and the pinned oneAPI 2026.1.1
compiler. It is built only with `VLLM_CPP_XPU_XE2_PREFILL=ON` and an explicit
`VLLM_CPP_SYCL_TLA_DIR`. The build option defaults to ON for XPU builds that
provide that checkout. Eligible FP8 prefill inputs select Xe2 by default;
`VT_XPU_XE2_PREFILL=0` disables it. Unsupported inputs use the existing
attention path. The qualified B70 configuration accepts 64- and 1600-token
KV pages.

Copied source files relative to `csrc/xpu/attn/xe_2/` and original SHA-256:

| File | Original SHA-256 |
| --- | --- |
| `chunk_prefill.hpp` | `b21a4d4cdd490dcbfb892a9cf30ba260dfe9ea32e8d5c0104b62703b674a0563` |
| `collective/chunk_prefill_mainloop.hpp` | `65987b9d0416012a73517253174f57cb1a14b8ecac12a14b29c2f46abb30ba28` |
| `collective/chunk_prefill_epilogue.hpp` | `edd78bf05bc40b51a72ed82e79ecc12f2652e114a41730f85486e90bd3c2f171` |
| `collective/chunk_prefill_scheduler.hpp` | `6b7b94844ea5fc29f6bbd36c4fe0ac638c0d30e090129da203196b0b81ed981a` |
| `kernel/chunk_prefill_kernel.hpp` | `420a8276cf1843aab3e21fb664412ff7c649786cfca9e3d4b405d30c51676a7c` |

`chunk_prefill.hpp` excludes the upstream Torch dispatcher, omits its
zero-byte default-device workspace allocation, and returns the actual SYCL
launch event to VT. The mainloop interprets null
K/V scale pointers as unit scales for the guarded VT policy. Five explicit
prefetch calls in the selected prefill mainloop can be restored with
`VLLM_CPP_XPU_XE2_PREFILL_PREFETCH=ON`; all block loads remain active.
