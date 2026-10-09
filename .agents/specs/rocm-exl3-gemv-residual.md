<!--
REVISED during implementation: R3 changed from the reconstruct arm to a new
fused decode+dot kernel (`rocm_exl3_dot.hip`, Exl3DotK). The trace showed the
recon lm_head chain costing ~40 ms/token (decode to fp16 scratch + hipBLAS is
2.5x the trellis traffic); the dot arm reads the trellis once, needs no
scratch, and is capture-safe under VLLM_CPP_ROCM_STATIC_GRAPH. R2's chunked
variant now launches dot chunks rather than GEMV chunks (same 8-row slicing).
-->
# ROCm EXL3 residual: the three shapes that still take the scalar arm (BACKEND-ROCM)


Row: `BACKEND-ROCM`. Issue:
`.agents/issues/BACKEND-ROCM/ISSUE-LOCAL-01M3RM2TM98FY569AZ6CEASBD3.md`.
Follows `.agents/specs/rocm-exl3-gemv.md` (the landed m≤8 GEMV + m>144
reconstruct arms) and `.agents/specs/backend-rocm-exl3.md` (the byte-exact
CPU transcription that remains the fallback arm).

## Problem

rocprofv3 kernel trace of `vllm-qwen35-9b-exl3` (Qwen3.5-9B-EXL3-4.00bpw,
gfx1101, ROCm 10.0, 96-token greedy decode + 21-token prefill):

| Kernel | Dispatches | Total | Share |
|---|---|---|---|
| `Exl3GemmK` (scalar transcription) | 8,656 | 26,190 ms | 91.4% |
| `Exl3GemvK<4,1,0,1>` | 9,120 | 1,500 ms | 5.2% |
| `Exl3GemvK<4,1,0,0>` | 1,520 | 45 ms | 0.2% |
| everything else | ~106k | ~910 ms | 3.2% |

Attributing `Exl3GemmK` by grid geometry (grid_x/256 = n/16 blocks):

1. **n=4096, m≤16, 87 calls/token, 2.40 ms each — 70% of decode time.**
   These are the 4-bpw projections that the upstream `Exl3GemvSelectConfig`
   heuristic *declines*: for (k=4096, n=4096, cb=1) the selector falls through
   every branch and returns -1 (`exl3_policy.cpp:153-181`). Upstream declines
   because its m≤8 alternative is a tensor-core GEMM; ours is the byte-exact
   scalar transcription at ~30 GB/s effective, so a decline on ROCm is a ~25×
   slowdown, not a tuning decision.
2. **n=248320 (lm_head), 6 bpw, 1 call/token, 53.8 ms.** `Exl3GemvHardEligible`
   rejects `bits > 4` — upstream instantiates no 6-bpw GEMV arm at all. The
   reconstruct arm (`ReconstructDecodeSliceK`) decodes 1..8 bpw through the
   scalar `Exl3DecodeCodeword`, so the n-sliced reconstruct+hipBLAS path covers
   this weight with machinery that already exists and is gated.
3. **9 ≤ m ≤ 144 (prefill), 930 ms for a 21-token prompt.** Between the GEMV
   envelope and the `dense_attn_block.h` reconstruct threshold, every
   projection runs the scalar arm.

## Design

Three changes inside `Exl3GemmKernelRocm` (`src/vt/rocm/rocm_exl3.hip`) and its
GEMV try-launch (`src/vt/rocm/rocm_exl3_gemv.hip`). The byte-exact transcription
stays reachable under `VT_EXL3_GEMV=0`, which becomes the documented "scalar
arm for every shape" switch on this backend.

### R1 — never decline a hard-eligible m≤8 call on ROCm

`Exl3GemvHardEligible` stays upstream-verbatim (shared with the CUDA arm).
`Exl3GemvSelectConfig` stays upstream-verbatim too — the ROCm try-launch
already calls it; the change is that a -1 is no longer a decline. When the
selector returns -1 and the call is hard-eligible with an instantiated
(bits, cb), fall back to the mode-2 forced rule `cfg = size_n <= 8192 ? 0 : 1`
(exl3_gemv.cu:66-67 — upstream's own forced-cfg arithmetic). On gfx11 the
alternative is not "a different fast arm" but the transcription, so decline
must mean "not eligible," not "the Ampere heuristic has no opinion."

### R2 — mid-m (9 ≤ m ≤ 144): chunked GEMV, else the reconstruct arm

For `8 < m ≤ 32` with an instantiated (bits, cb) arm: loop the GEMV launch
over 8-row chunks of `a_had`/`raw` — each chunk is upstream's own m≤8 arm with
`size_m = min(8, m - base)`, `mmode = 1`. B is re-read per chunk (upstream's
own trade-off for mmode 1), so the cap keeps re-reads ≤ 4× the weight and
bounded ALU growth.

For `32 < m ≤ 144`, or any mid-m shape the GEMV arm cannot take
(bits > 4, un-instantiated cb): call `Exl3ReconstructGemmKernelRocm`
(`rocm_exl3_recon.hip`) directly with an empty `w_scratch` — the persistent
per-stream scratch discipline — and return, letting that arm run its own
Hadamard stages. The M > 144 caller dispatch is unchanged.

### R3 — lm_head and other bits > 4 shapes: the reconstruct arm at every m

`bits ∉ [2,4]` or `(bits, cb)` uninstantiated ⇒ the reconstruct arm inside
`Exl3GemmKernelRocm` for all m (decode included), replacing the scalar arm.
The 32768-column slice bound already caps scratch at `k*32768*2` bytes
(256 MiB at k=4096), so the n=248320 lm_head runs as ~8 decode+hipBLAS slices.
This is interim: a fused 6-bpw GEMV would avoid materializing the fp16 weight
(2.5× the traffic) and is named as owed if the profile still shows lm_head
dominant.

### Ordering and guards

`VT_EXL3_GEMV=0` (`Exl3GemvMode()==0`) bypasses ALL three fast paths — R2/R3
read the same env so the byte-exact transcription remains reachable for the
whole shape space. `force_gemv > 0` semantics unchanged. `HadK` launch order
unchanged for shapes that stay on the fused path (R1/R2 keep the existing
three-launch structure; R3 and the recon branch of R2 delegate to the recon
arm before the first `LaunchHad`).

## Tests

Extend `tests/vt/test_exl3_rocm_gemv.cpp` and the recon suite rather than
inventing new bounds:

1. **R1 dispatch**: the recorded declined shape (m∈{1,8}, k=4096, n=4096,
   bits=4, cb=1) must now produce a GEMV launch — assert via a counter hook or
   by timing-independence; simplest is `force_gemv`-equivalent rel-RMS
   correctness on that exact shape (bound 6.0e-3 as landed).
2. **R2**: m ∈ {9, 21, 32, 33, 100, 144} on 4-bpw cb-1 — chunked-GEMV cases
   at 6.0e-3 rel-RMS vs the CPU arm; recon-routed cases at 1.0e-3.
3. **R3**: 6-bpw cb-1, m ∈ {1, 8, 21} — rel-RMS ≤ 1.0e-3 vs the CPU arm (the
   recon arm's own bound).
4. **Byte-exact regression**: `VT_EXL3_GEMV=0` still byte-equals the CPU arm on
   every shape above, including bits=6 and mid-m.
5. **E2E**: image rebuild, c=1 greedy 128-token coherence + tok/s recorded
   against the 3.30 t/s baseline; rocprofv3 re-trace shows `Exl3GemmK`
   attribution collapsed.

## Gates

Unit gates via ctest on the gfx1101 host build (`-DVLLM_CPP_HIP=ON
-DCMAKE_HIP_ARCHITECTURES=gfx1101`), the same recipe the parent spec used.
Serving gate on the `vllm-qwen35-9b-exl3` compose stack.

## Stop conditions

- Chunked-GEMV rel-RMS exceeds the arm's bound at any m → drop R2, route all
  mid-m to recon, record the numbers.
- Recon for lm_head breaks greedy parity beyond the ≤1-position tolerance the
  parent spec set → revert R3 only, keep R1/R2, and the 6-bpw fused GEMV
  becomes mandatory rather than owed.

## Outcome

Landed on `row/BACKEND-ROCM-exl3-gfx11-gemv` at the commit carrying this file.
R3 was REVISED mid-implementation: the lm_head chain showed the reconstruct
arm costing ~40 ms/token (decode to fp16 scratch + hipBLAS reads the decoded
matrix twice — 2.5x the trellis traffic), so a NEW fused decode+dot kernel
(`src/vt/rocm/rocm_exl3_dot.hip`, `Exl3DotK<BITS>`) replaced it. The kernel
reads each trellis word once per warp, accumulates in f32, and is capture-safe
(no hipMalloc inside the launch path, plain kernel launch only).

The trace-driven design changes from the first draft:
- Span extraction collapsed to ONE `v_fshr`/`v_alignbit` funnel shift per
  codeword — `(a<<32|b)>>s` is the same shape the SASS emits for the merged
  two-word read, and `e0` (the lane's span bit residue) is compile-time for
  bits==4 (==20) and bits==6 (==22 or 6 on lane parity). BITS is a template
  parameter for exactly this reason; every other width falls back to the
  runtime-bits arm.
- B fetches went COALESCED: lane l loads tile word l (bits==4: 32 words =
  one warp-load; bits==6: lanes 0-15 hold a second), and the lane's span is
  assembled by in-warp shuffle. The first draft's per-lane 4-word span load
  re-read every shared word four or five times — measured at ~55 GB/s on the
  lm_head, the coalesced version reaches ~70 GB/s (the decode arithmetic,
  not the memory path, is the bound).
- A-loads pack two adjacent fp16s per u32 (RowMajorIndex places each lane's
  k-rows at r0+{0,1} and r0+{8,9} — both adjacent pairs), halving the LSU
  work in the m-loop.
- Prefetch is an explicit two-deep ring of named registers, NOT pf[kt&3]:
  a dynamic index into a register array lowers to local-memory spill on this
  backend (measured: the dynamic-index variant ran 17 t/s vs 20.4 t/s).

Serving measurement on the Qwen3.5-9B-EXL3-4.00bpw compose stack: decode went
3.30 -> 20.4 tok/s (warm), a 6.2x gain over the container's earlier baseline
and still ~2.4x under the 50 tok/s target. The residual is the emulated-MMA
GEMV arm on the 4-bpw layer GEMMs (about 25 of 47 kernel-ms per token), which
both the dot arm and the WMMA variant lose to (10.1 and 11.7 tok/s
respectively when forced to serve m<=8). ROCm static-graph capture is enabled
(VLLM_CPP_ROCM_STATIC_GRAPH=1 + VLLM_CPP_CUDAGRAPH=1) and engaged — the
remaining gap is kernel time, not launch overhead. The host GPU measures
~352 GB/s copy bandwidth (not the spec sheet's 540), which puts the
bandwidth-only ceiling at ~80 tok/s and 50 tok/s within reach of a kernel
that holds ~250 GB/s effective.

Round 2 (this branch, `acf3e96ad`/`8b8ebdbaf`): the dot arm was rebuilt once
more — one 16-column tile per 256-thread block with eight-way in-block
k-split (matching the GEMV arm's warp-level k-parallelism), plus a
`amd_mixed_dot` (v_dot2_f32_f16, a VOPD dual-issue candidate on gfx11) inner
loop on the fp16-exact decoded weights. The emulated-MMA GEMV arm gained an
MMODE==0 specialization that drops the provably-zero rows-8..15 gather and
its half of the hfma2 chain. Serving still measures ~21.2 tok/s; dot-first
re-measured 10.4-13.4 tok/s even k-balanced, so the GEMV arm stays first for
m<=8. PF=8 prefetch rings REGRESSED (20.0 tok/s — VGPR pressure beats the
latency cover) and are not carried. `VT_EXL3_GEMV_CFG=1` was observed to
stall generation entirely under the serving driver (GPU idle, requests
queued) while the unit test passes under the same env — recorded as a
cfg-1-specific defect, not diagnosed further in this round.

Verified: `ctest -R test_exl3_rocm` 3/3 pass on gfx1101 (all eight
instantiated arms at rel RMS ~7e-4 vs f64; the chunked-dot arm at ~2e-5;
bits==6 m=1..21 at ~1e-5). The test's per-shape divergence CHECK was relaxed
from `> ref.size()/4` to `> 0` — the dot arm accumulates in f32 and is
byte-NEARER the transcription than the fp16-fragment GEMV, so a single
differing output (not half of them) is what proves a fast arm ran.
