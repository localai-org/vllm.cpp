# matmul trace-record class split — the ~144 KB/launch is OUR program COUNT, not tt-metal's page packing (2026-10-01)

Discriminating experiment for the last unexplained trace-record class
([tt-matmul-record-attribution-20260930.md](tt-matmul-record-attribution-20260930.md)
§4). On the post-fusion tree (`row/tt-matmul-class-split`, build2 @ pin
`6449cf13f7b`, fused keepquant kernel in), a warmed single-launch `MatmulBT`
capture at the 27B hidden GEMM shape closes a region at **147,456 B** —
14x the stock `ttnn::matmul` record at the identical shape (10,240 B for
one launch). The logging-build dispatch census attributes effectively ALL
of it.

## 1. Reproducer

Focused doctest `kTENSTORRENT matmul region record class split
(VT_TT_MMCLASS=1)` in `tests/vt/test_tenstorrent_backend.cpp` (env-driven
shape: `VT_TT_MMCLASS_ROWS/_K/_N/_LAUNCHES`). One eager warm call, then one
`GraphCaptureScope` region of N `MatmulBT` launches; the region close reads
the `SetGraphRegionBytesProbe` staging census. Run under
`flock /home/lu_zero/gpu.lock`, `luwen reset` + 15 s, pin-first
`LD_LIBRARY_PATH` (`build_release_script/lib64`, `ttnn`,
`libexec/tt-metalium`), no other `TT_METAL*` env. Logs `/tmp/mmsplit/`.

## 2. Reproduced region closes (delta bisect)

| shape [M,K]x[K,N] | launches | region close | per-launch marginal |
|---|---|---|---|
| [64,5120]x[5120,5120] | 1 | 147,456 B | — |
| [64,5120]x[5120,5120] | 8 | 1,155,072 B | **~143,945 B** |
| [1,5120]x[5120,5120] | 1 | 188,416 B | — |
| [1,5120]x[5120,17408] (27B gate/up) | 1 | 540,672 B | — |
| [64,5120]x[5120,1024] (down-proj) | 1 | 19,456 B | — |
| [64,256]x[256,256] | 1 | 14,336 B | — |

The record is per-launch linear at ~144 KB and grows with N: the 27B
gate/up GEMM alone records 540,672 B per launch. (`logging1.full.log` run
ends `leg_exit=139` — the pre-existing teardown segfault recorded in the
issue; the census line printed before it.)

## 3. The class split (logging build, TT_METAL_LOGGER_LEVEL=TRACE
TT_METAL_LOGGER_TYPES=Dispatch, `~/Sources/tt/tt-metal-pin/build_logging`)

The [64,5120]x[5120,5120] 1-launch capture log
(`/tmp/mmsplit/logging1.full.log`) assembles **20 distinct tt-metal
programs** per MatmulBT launch, each assembled twice (warm eager + captured)
— Program IDs 2..40, one command sequence each. Their
`Command Sequence Summary` TOTALs:

4928, 13440, 640, 3904, 15680, 10624, 3904, 4288, 13440, 640, 3904, 6528,
6848, 3712, 2752, 13568, 13568, 6400, 13568, 4928 — **sum 147,264 B**.

**147,264 of the 147,456 B region close is exactly the 20 per-program
command sequences**; the 192 B residual is the per-trace header floor.
Every individual program sits at tt-metal's KB floor (640-15,680 B; stock
full-grid matmul records 8,484 B). The same log refutes the remaining
candidates:

- **(a) non-identical per-core pages — REFUTED.** The packed relay works:
  `Common RTA (MCAST via kernel_group)` collapses the common args
  (dispatch.cpp:1158), `Kernel Binary (MCAST)` streams each binary once per
  core range (dispatch.cpp:1997). The largest single in-program class is
  per-core **Unique RTA (UNICAST)** (dispatch.cpp:1221, 44 B/core x 110
  cores ≈ 5,696 B on the 15,680 B program) — real but inside the stock KB
  band, and stock matmul pays the same shape of term.
- **(b) prefetch-ring overflow fallback — REFUTED.** Each program's
  assembled sequence is a one-shot fetch of 15,872 B max ("One-shot mode:
  true, Fetch size: 15872"), 65x under the 1,024 KB ring fit; no
  `add_prefetch_relay_paged` fallback appears.
- **(c) captured device-side movement — REFUTED.** No inline payload
  commands in the capture window; the only `write_interleaved_buffer_to_device`
  traffic is the pre-capture warm staging, and the capture-scope upload
  guards held (no staging writes during capture).

## 4. Verdict

**Dominant class: (d) — many kernels per launch. OUR program construction.**
One `MatmulBT` launch on our chain enqueues ~20 tt-metal programs (the
keep-quant decode/typecast/layout prep plus the grouped matmul chain); each
is individually at tt-metal's floor, so the per-launch record is
20 x ~7.4 KB ≈ 144 KB where stock matmul pays one program at 8.5 KB.
Secondary term: the per-program sequences scale with N (14 KB at N=256 →
541 KB per launch at the 27B gate/up N=17408) through per-core tile-count
growth in RTA/launch-message payloads — same locus, same fix.

## 5. Wave-2 fix proposal

Collapse the per-launch program chain to 1-2 fused full-grid programs, the
exact lever the decode fusion already proved (region 1: 2,965,504 → 32,768
B, `tt-decode-fusion-20260930.md`): one custom kernel in the
`kKeepQuantDecodeFusedKernelSrc` house style — in-kernel decode+GEMM tiling,
ONE `SetCommonRuntimeArgs` vector, uniform self-cycled CBs, warm-first
cache — replacing the decode→typecast→to_layout→matmul chain assembled in
`src/vt/tenstorrent/tenstorrent_keepquant.cpp:1279` (the chunk/program
emission) and the `MatmulBT` dispatch in `src/vt/tenstorrent/
tenstorrent_ops.cpp`. Keep the chain as the named
`VT_TT_KEEPQUANT_FUSED=0`-style fallback. Expected win: 147,456 → ~8-16 KB
per launch (≈10-18x) at the 27B hidden GEMM, and ~540 KB → ~10-20 KB at the
gate/up shape; on the 27B whole-graph capture (~357 non-eltwise launches)
that removes ~50 MB of trace demand and, with the per-launch floor at the
stock band, re-opens the fit-wall re-measure (c1/c2 legs).

## 6. Records

Not upstream-filed: tt-metal's per-program record path is at its floor and
the packed relay works; the 20-program count is ours. Harness committed in
the same change (`tests/vt/test_tenstorrent_backend.cpp`, the VT_TT_MMCLASS
doctest); issue
`.agents/issues/BACKEND-TENSTORRENT-QWEN35/ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ`
updated 2026-10-01. Raw logs: `/tmp/mmsplit/{leg4,bisect1,logging1.full}.log`.
