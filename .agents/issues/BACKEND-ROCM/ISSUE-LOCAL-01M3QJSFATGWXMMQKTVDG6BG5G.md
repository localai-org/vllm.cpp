ID: ISSUE-LOCAL-01M3QJSFATGWXMMQKTVDG6BG5G
Title: ROCm EXL3 decode is 97% Exl3GemmK scalar-transcription kernel — port GEMV arm
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: 2026-09-30

## Problem

On gfx1101 serving Qwen3.5-9B-EXL3-4.00bpw, rocprofv3 attributes 11392ms of ~11700ms kernel time to vt::rocm::Exl3GemmK (~712ms/token, ~1.7 tok/s). Byte-exact CPU transcription: 16/256 live threads at m=1, double barriers per k-tile, no prefetch. Upstream's m<=8 exl3_gemv arm (exl3_gemv_kernel.cuh; ROCm-proven in exllamav3-rocm) unregistered for kROCM; kExl3ReconstructGemm CUDA-only.

## Resolution

GEMV and reconstruct arms landed on gfx1101 (row/BACKEND-ROCM, spec rocm-exl3-gemv): 8 instantiated arms at rel RMS ~7e-4 vs bound 6.0e-3, byte-exact fallback preserved, mode-1 dispatch verified, serving c=1/4/8 at 3.30/12.27/21.93 tok/s vs 1.75/5.90/8.91. Residual: Exl3GemmK still 90% of kernel time on this checkpoint (6bpw lm_head uninstantiated, mid-m/GDN shapes decline) — recorded in spec Outcome.
