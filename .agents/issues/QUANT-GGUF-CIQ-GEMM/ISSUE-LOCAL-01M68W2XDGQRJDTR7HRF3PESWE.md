ID: ISSUE-LOCAL-01M68W2XDGQRJDTR7HRF3PESWE
Title: x86 CPU quantized GEMM runs the portable vec_dot tier only
Row: QUANT-GGUF-CIQ-GEMM
State: CLOSED
Kind: performance
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-31
Updated: 2026-10-31
Closed: 2026-10-31

## Problem

The Flare GSQ IQ3_M serving profile spends most decode time in the portable IQ3_S/Q4_K/Q6_K vec_dot kernels. The CPU has AVX2/FMA and llama.cpp b10451 is ~3.5x faster end to end on the same model, but BlockVecDot never selects an x86 SIMD tier. Add runtime-gated AVX2 kernels for the three dtypes that dominate this model and gate them against the f64 reference.

## Resolution

Fixed 2026-10-31. `src/vt/cpu/cpu_quant_dot_avx2.cpp` ports the AVX2 q4_K/q6_K/IQ3_S `vec_dot` arms of llama.cpp b10451 `arch/x86/quants.c:2018`, `:2406` and `:3364`; `BlockVecDot` selects them at runtime through `DetectX86IsaCaps().avx2` with `VT_CPU_QUANT_AVX2=auto|portable|avx2` and the portable tier as the fallback. Same-binary Flare GSQ A/B: decode 16.68 s -> 6.30 s (2.65x), 32-token text byte-identical. `test_ops_quant_dot` is 35 cases / 285,691 assertions green under auto, portable and avx2; mutations of the IQ3_S sign handling and of the `2*ls+1` weight both redden the IQ3_S numeric checks. Installed as `/mnt/openwebui/AIKAR/Lumen-3.8-Flare/bin/lumenengine-server` (sha256 `2f3b4c5d023cad0e225f5096f6fcfefcec6535dc6faf9e40c96bc267d00da0fa`) and serving the `gsq-rco-cpu` profile on port 11435; `verify.py --expect-cache on` PASS. Evidence: `.agents/evidence/quant-g5-avx2/` and `evidence/gsq-lumen-g5-20261031/` in the deployment package. llama.cpp b10451 remains ~1.5x faster (3.91/4.24 s); the residual is recorded in the spec's G5 result section.
