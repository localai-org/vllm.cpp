# matmul trace-record attribution — the ~8 MB/launch class is OURS (2026-09-30)

Final unexplained class of the 27B whole-graph trace wall. After the fused
keepquant kernel collapsed the eltwise chain (region 2,965,504 → 32,768 B,
90x, `tt-decode-fusion-20260930.md`), the remaining mass is the
matmul/attention-class launches: a captured region with one `MatmulBT` close
measured 3,088,384 B (pre-fusion pin), and the 27B's remaining ~357
non-eltwise launches need ~8 MB/launch to reach the recorded ~2.9 GB
whole-graph wall. Question: is the MB-scale recorded stream tt-metal's
(matmul record structure) or ours (program construction)?

## 1. Reproducer

`repro_matmul_trace.cpp` in this directory. Standalone ttnn harness against
the pin install tree (`vllm-cpp-pin/20260923-adv` @ `6449cf13f7b`,
`~/Sources/tt/tt-metal-pin/build_release_script`), mirroring
[tt-trace-config-page-repro-20260929.md](../tt-trace-config-page-repro-20260929/repro_trace_config_pages.cpp):
warm the op with the program cache on, capture 1 launch and 8 launches in
traces, read `MeshDevice::get_trace_buffers_size()`, divide out the fixed
1,024 B empty-trace floor. Build:

```sh
PIN=~/Sources/tt/tt-metal-pin/build_release_script
SPD=~/Sources/tt/tt-metal-pin/.cpmcache/spdlog/b1c2586bb5c35a7929362e87f62433eb68206873/include
clang++ -O1 -std=c++20 repro_matmul_trace.cpp -o /tmp/mmrepro \
  -isystem $PIN/include -isystem $PIN/libexec/tt-metalium/ttnn \
  -isystem $PIN/include/tracy -isystem $PIN/libexec/tt-metalium/tt_metal/hostdevcommon/api \
  -isystem $PIN/libexec/tt-metalium -isystem $PIN/include/metalium-thirdparty -isystem $SPD \
  -L$PIN/lib64 -L$PIN/ttnn -Wl,-rpath,$PIN/lib64 -Wl,-rpath,$PIN/ttnn \
  -l:_ttnncpp.so -ltt_metal -ltt_stl -ltt-umd -lfmt -lspdlog
```

Run: `flock /home/lu_zero/gpu.lock`, luwen `reset` + 15 s,
`TT_METAL_HOME`/`TT_METAL_RUNTIME_ROOT` at the pin source tree,
`LD_LIBRARY_PATH` pin-first (`lib64`, `ttnn`, `libexec/tt-metalium`).
Log `/tmp/mmrepro-run4.log`, `EXIT=0` (first attempt, run2, hit "Root
Directory is not set" — the pin reads `TT_METAL_RUNTIME_ROOT`, not
`TT_METAL_HOME`).

## 2. Measured — bytes/launch

| op | shape | grid | empty | 1 launch | 8 launches | bytes/launch (delta) |
|---|---|---|---|---|---|---|
| ttnn::matmul bf16 | [64,5120]×[5120,5120] (27B hidden GEMM) | 11×10 = 110 cores, full device | 1,024 | 10,240 | 69,632 | **8,484** |
| ttnn::matmul bf16 | [64,256]×[256,256] (small) | 110 cores | 1,024 | 4,096 | 28,672 | 3,510 |
| ttnn::matmul bf16 | [64,5120]×[5120,1024] (down-proj) | 110 cores | 1,024 | 5,120 | 34,816 | 4,242 |

Stock ttnn::matmul at real 27B shapes records **~8.5 KB per launch** on the
full device grid — 1,000× below the ~8 MB/launch the 27B wall needs. Grid
extent does not scale the record (consistent with the 2026-09-29 refutation;
the packed relay collapses full-grid programs).

## 3. Control — the old inline-H2D mechanism is dead on this pin

The 2026-09-28 audit
([tt-trace-record-audit-20260928.md](../tt-trace-record-audit-20260928.md))
attributed region 1's 3,088,384 B to a captured inline H2D weight payload
(2,048 B headers + 3,086,336 B payload). Re-tested on this pin: a
`ttnn::Tensor::from_vector` of ~1.54 M bf16 elements issued INSIDE a capture
**FATALS and aborts** — `TT_FATAL ... "Writes are not supported during trace
capture"` at `fd_mesh_command_queue.cpp:830` (`!trace_id_.has_value()`), log
`/tmp/mmrepro-run3.log`, EXIT=134. The pin rejects ALL mesh writes during
capture, so no payload can be recorded inline anymore. The old mechanism is
falsified on `6449cf13f7b`; whatever produces ~8 MB/launch in our backend
today is not a host→device write under capture.

## 4. Class split

Reached at the header/payload level from the previous row's audit plus this
pin's behavior; the per-chunk issue_queue census on the logging build was not
re-run this leg (budget):

- empty trace (fixed per-trace cost): 1,024 B — unchanged.
- headers + RTA + CB/DFB snapshot per stock program launch: the 1,024–8,484 B
  band above; the packed relay keeps a full-grid, big-shape matmul in it.
- captured inline H2D payload: **structurally zero** — writes FATAL on this
  pin (§3).
- unexplained: our `MatmulBT`/attention regions at 3.09 MB per region close
  (pre-fusion measure). The delta over stock (~3 MB − ~8 KB) is neither
  headers, nor stock RTA/CB/DFB snapshot cost, nor H2D payloads. Remaining
  candidates, in order: (a) our program's per-core config/RTA pages are NOT
  byte-identical across the grid so packing fails and per-page relay embeds
  them (the fused small program packed, but a large-matmul program with
  per-core variance may not); (b) our program exceeds the 1,024 KB prefetch
  ring-buffer fit and falls back to `add_prefetch_relay_paged` over
  program-kernel pages recorded per launch; (c) per-launch captured
  data-movement inside our custom programs (device-side copies recorded as
  commands).

## 5. Verdict

**OURS.** Stock tt-metal on this pin records a full-grid 27B-shape matmul at
8,484 B/launch and refuses capture-scope writes outright. The ~8 MB/launch
class cannot originate in tt-metal's matmul record path; it is specific to how
vllm.cpp constructs the MatmulBT/attention programs (or what our capture path
records around them). Fix locus: our program construction in
`src/vt/tenstorrent/` — next leg is the logging-build or issue_queue census on
one current (post-fusion-pin) MatmulBT region to discriminate (a)/(b)/(c) in
§4, targeting the same 27B whole-graph capture used by the retention legs.

Not upstream-filed: no tt-metal defect exists to file — the stock record path
is already at the KB floor and enforces the no-writes invariant.
