ID: ISSUE-LOCAL-01M3T5GBR5NS2B778YRKWK0ZG3
Title: EXL3 prefill on ROCm: recon arm's hipblasGemmEx picks a scalar kernel (HSS) — wire hipBLASLt
Row: BACKEND-ROCM
State: CLOSED
Kind: perf
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-01
Updated: 2026-10-01
Closed: 2026-10-01

## Problem

Prefill (m>32) routes to Exl3ReconstructGemmKernelRocm, whose hipblasGemmEx HIPBLAS_COMPUTE_32F resolves to Cijk...HSS at 16.4ms/call on gfx1101 (~2 TFLOP/s, 6% of gfx1101 fp16 peak). rocprofv3 on vllm-cpp:rocm-gfx1101-exl3 serving Qwen3.5-9B-EXL3-4.00bpw, 2x2022-token prompts: recon GEMM 34.0%, GdnScanK 32.3%, PagedAttnOnline 22.4%.

## Resolution

Landed on branch pp-exl3-recon-lt (commit b690dea96): hipBLASLt + timed-algo pick inside Exl3ReconstructGemmKernelRocm, 10.8x on the GEMM, 1.46x end-to-end on a 2022-token prompt, VT_EXL3_RECON_NO_LT=1 keeps the old arm. Green on gfx1101: test_exl3_rocm (33 asserts), test_exl3_rocm_gemv (60 asserts).
