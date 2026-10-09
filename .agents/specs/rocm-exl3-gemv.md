# ROCm EXL3 decode GEMV arm (BACKEND-ROCM)

Row: `BACKEND-ROCM`. Issue: `.agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3QJSFATGWXMMQKTVDG6BG5G.md`.
Follows `.agents/specs/backend-rocm-exl3.md` (the byte-exact CPU transcription this spec keeps as
the fallback arm) and `.agents/specs/quant-exl3-perf.md` (the CUDA GEMV this mirrors).

## Problem

`vt::rocm::Exl3GemmK` (`src/vt/rocm/rocm_exl3.hip`) is a byte-exact transcription of the CPU
reference. Measured on gfx1101 (RX 7700 XT, ROCm 7.15, `local/rocm-base:10.0.0` image) serving
`Qwen3.5-9B-EXL3-4.00bpw`:

- rocprofv3 kernel trace, one 16-token decode: `Exl3GemmK` = 11,392 ms of ~11,700 ms total
  kernel time (97%), 3,216 calls avg 3.5 ms → ~712 ms/token, ~1.7 tok/s end to end.
- The 16×16-tile workgroup layout: one warp-load per k-tile serialized behind 2×`__syncthreads`,
  16 of 256 threads hold live accumulators at m=1, no prefetch, no matrix cores. Cost scales
  O(k·n) scalar instruction latency, not the ~7.5 ms/token bandwidth floor (~4.1 GB trellis).
- Upstream exllamav3 (the registered secondary oracle for EXL3) serves small-m through a
  dedicated GEMV: `exl3_gemv_kernel.cuh` — k-split warps, register prefetch ring on
  `__ldcs`-style streaming loads, two-word bit windows resolved by in-warp shuffles, one MMA
  pair per 16×16 tile, cross-warp f32 reduction. `cuda_exl3.cu` already ports it for the CUDA
  arm inside `Exl3GemmKernelCuda`'s try-launch. `kROCM` has none of it: no GEMV, and
  `kExl3ReconstructGemm` (the M>144 cuBLAS path) is CUDA-only too.

## Design

### Slice A — the m≤8 GEMV arm (this spec's core)

New file `src/vt/rocm/rocm_exl3_gemv.hip`, donor `exl3_gemv_kernel.cuh` @ exllamav3
2398c05635fbbad01a0a51dce63c85c6c8a8450e, as ROCm-ported and verified in
`exllamav3-rocm` @ 4a19d7a (which serves ~11 tok/s on this GPU family).

Faithful port of the CUDA arm's geometry, with three stated adaptations:

1. **No cooperative launch.** The upstream kernel wraps Hadamard stages and the GEMM body in
   `grid.sync()` inside one cooperative launch. `Exl3GemmKernelRocm` already runs three separate
   launches (`LaunchHad` in → GEMM → `LaunchHad` out) on one stream, which gives the same
   ordering without cooperative-launch constraints, `locks` buffer, or occupancy capping to
   co-residency. The GEMV body is launched between those two `LaunchHad` calls, identical
   stream ordering to upstream.
2. **Accumulate path selectable: emulated `m16n8k16` vs WMMA.** exllamav3-rocm carries both:
   `mma_m16n8k16_f16_emu` (shuffle-gather + packed `__hfma2`, default; reproduces the CUDA arm's
   fp16-fragment accumulation exactly, so the CUDA arm's 6.0e-3 relative-RMS bound
   (`test_exl3_gemv.cpp:454-456`, spec `quant-exl3-perf` tier 3c) transfers to this arm), and
   `wmma_m16n16k16` (`__builtin_amdgcn_wmma_f32_16x16x16_f16_w32`) flagged there with a known NaN
   issue on m≥3 paths. Default = emulated; `VT_ROCM_EXL3_WMMA=1` selects WMMA for evaluation only.
3. **Streaming loads**: `__ldcs` → `__builtin_nontemporal_load` (the exllamav3-rocm shim — on AMD
   it emits `.slc` evict-first loads). `__shfl_sync` is native wave32 on gfx11.

Instantiation coverage mirrors upstream's GEMV envelope: bits 3 and 4 over codebooks 0, 1, 2 —
the full set `dq8_regs_*bits<cb>` supports (upstream instantiates 2/3/4 bpw; vllm.cpp's CUDA arm
carries (3,1)(3,2)(4,2); this arm widens to upstream's own cb list so stock cb-0 checkpoints —
this checkpoint — take the fast path). mmode 0 (m=1) and 1 (2≤m≤8), CFG 0/1, c_fp32 both.

Dispatch in `Exl3GemmKernelRocm`, mirroring `Exl3GemmKernelCuda`: GEMV try-launch first for
`m ≤ 8` + instantiated `(bits, cb)` + `Exl3GemvHardEligible`; else `Exl3GemmK`. `VT_EXL3_GEMV=0`
disables (same env the CUDA arm reads via `Exl3GemvMode()`); `args.force_gemv` honored.

### Slice B — `kExl3ReconstructGemm` on ROCm (prefill, M>144)

New file `src/vt/rocm/rocm_exl3_recon.hip`: a straight trellis→f16 decode kernel
(one codeword per thread, coalesced output — the bandwidth-bound work the fused arm avoids by
fusion) + hipBLASLt fp16 GEMM (`rocm_matmul_hipblaslt.hip`'s existing planner) + the two
`LaunchHad` stages. Registers `OpId::kExl3ReconstructGemm` for `kROCM`, which makes the existing
`M > 144` dispatch in `dense_attn_block.h::Exl3MatmulD` reachable. Per-stream persistent scratch
mirroring the CUDA arm's `Exl3PersistentReconScratch` discipline (hipMallocAsync, grow-only,
capture-safety rules). Instantiated for the same (bits, cb) set as the CUDA arm.

## Gates

1. **Unit** — `tests/vt/test_exl3_rocm_gemv.cpp` (new, this spec): ROCm GEMV vs the CPU arm at
   the CUDA GEMV arm's own bound — relative RMS ≤ 6.0e-3 (tier 3c; fp16-fragment accumulation),
   over every instantiated (bits, cb), m ∈ {1, 2, 8}, both cfgs, plus the gate-identifies-wrong-arm
   sibling check (sibling arm must exceed 100× the bound). Skips-with-notice on no ROCm device,
   asserts registration first.
2. **No regression** — `test_exl3_rocm` byte-exact suite still passes: its GEMM cases run with
   `VT_EXL3_GEMV=0` (the transcription arm keeps its byte gate; the env is asserted to select it).
3. **Dispatch** — `test_exl3_matmul_dispatch`-style case: m≤8 selects GEMV, m>8 falls through,
   `VT_EXL3_GEMV=0` falls through, `force_gemv` forces.
4. **E2E** — image rebuilt; server on :8420; greedy 32-token continuation on a fixed prompt is
   token-identical or differs by ≤1 position vs the pre-change binary (fp16-accum arm parity),
   plus a 128-token coherence check; rocprofv3 re-trace shows `Exl3GemmK` below 10% of kernel
   time; benchmark legs c=1/4/8 re-run with the same corpus and reported against the 1.75/5.90/8.91
   tok/s baseline.
5. **Slice B gate** — reconstruct arm vs CPU arm at 1.0e-3 rel-RMS (tier 3, f32 accumulate GEMM),
   `m ∈ {145, 1024}`, instantiated (bits, cb) set; dispatch covered by the m>144 case.

## Stop conditions

- GEMV kernel cannot reach ≤ 6.0e-3 rel-RMS on instantiated arms → do not dispatch it;
  `kExl3Gemm` keeps the transcription and this spec records the measured numbers.
- `__shfl_sync`-based decode wrong on gfx1101 (e.g., wave64 scheduling) → gate on
  `__builtin_amdgcn_ds_bpermute` fallback or hold the arm.

## Outcome

Landed 2026-09-30 on row/BACKEND-ROCM at base ee622a899.

Gates 1, 2, 3, 5: PASS on gfx1101. test_exl3_rocm_gemv — all eight instantiated
arms at rel RMS 6.5e-4..7.2e-4 vs the f64 reference (bound 6.0e-3), sibling
arms >100x, force_gemv=0 byte-equal to the CPU arm, unforced mode-1 dispatch
reaches the GEMV. test_exl3_rocm — byte gate preserved under force_gemv=0.
test_exl3_rocm_recon — nine arms at m=256 plus m=145 and m=1024 cases at rel
RMS ~3.1e-4 (bound 1.0e-3). The only defect the tier-3c gate caught: the
4-bit funnel window was built with operands swapped
(`__funnelshift_r(a, b, 20)` for the donor's `fshift(b, a, 20)`); every 4-bit
arm measured ~0.89 relative RMS until it was fixed to `fshift(b, a, 20)`.

Gate 4: PARTIAL. Serving Qwen3.5-9B-EXL3-4.00bpw on :8420 measured 3.30 /
12.27 / 21.93 tok/s aggregate at c=1/4/8 against the recorded 1.75 / 5.90 /
8.91 baseline (1.9x / 2.1x / 2.5x) — but this required fixing the container:
the compose service named `render` in group_add while the host render GID is
992, so /dev/kfd was unreachable, the engine had silently fallen back to CPU,
and the stored baseline was CPU-side. Token correctness held (greedy decode
coherent across 128 tokens). rocprofv3 on the fixed stack shows Exl3GemvK
dispatching in production decode (896 calls, ~0.14s) but Exl3GemmK still at
90.35% of kernel time (2.11s over 712 calls) — the <10% projection did NOT
hold on this checkpoint. Attribution: the checkpoint is mixed-bitwidth
(lm_head at 6bpw is uninstantiated → always Exl3GemmK, n=248320) and the
decode-batch + GDN/mid-m shapes decline the arm; the residual is a new
candidate gap, not a defect in the landed arm.

Rejected: WMMA stays evaluation-only behind VT_ROCM_EXL3_WMMA (spec default);
the fused M>=1024 reconstruct arm was not ported — the unfused chain covers
all m with no measurable loss at this row's serving scale.

## Residual gap (2026-09-30, row/BACKEND-ROCM-exl3-gfx11-gemv)

Round-2 kernel work landed the 4 bpw m=1 wide-load arm (Exl3GemvM1K4,
load-position ownership — no per-tile shuffle) and halved the 6 bpw lm_head
(dot arm: __hadd decode, paired-codeword decode, double-buffered LDS staging,
A-fragment ring prefetch; 99 → 300 GB/s on the 4096×256000 shape). End-to-end
decode on the same checkpoint measures ~35 tok/s warm (vs 21.9 on the round-1
build); a rocprofv3 kernel-trace on that older build attributed per-token GPU
time as: body GEMV 11.9 ms, DotK<6> lm_head 7.4 ms, GdnScanK 2.3 ms,
HadK 1.5 ms, plus ~5 ms of smaller attention/GDN/cast kernels.

Still open for the 50 tok/s target: DotK<6> stalls at ~300 GB/s against a
~176-instr/tile-pair VALU floor (~2.1 ms) — the remaining gap is the
ld_tile ring's vmcnt(0) waits serialising each pair-iteration (~45%);
pre-fill m>8 shapes route to the byte-exact Exl3GemmK transcription at
~14 GB/s (1.5 K-token prompts take ~9 s); the warp-per-row GDN scan
(VT_GDN_SCAN_COOP=1) shaves ~1 ms/token but stays opt-in until the NMSE
policy for its reordered reduction is resolved; HadK is fused upstream of
every projection and contributes ~1.5 ms/token of fixed transform cost.
