# trace config-page repro: the full-grid × per-launch scaling hypothesis is REFUTED (2026-09-29)

Follows [tt-keepquant-rta-fix-20260929.md](tt-keepquant-rta-fix-20260929.md),
which left the 27B whole-graph trace at 2,925,109,248 B (1,037 launches,
~2.82 MB/launch post per-core-RTA fix) with the remaining dominant class
named "per-launch CB/DFB config pages", suspected to scale as config-page
count × grid. This note records the standalone reproducer built to test that
scaling on stock ops — and the refutation.

## 1. The reproducer

`repro_trace_config_pages.cpp` in this directory. Standalone C++ harness
against the pin's libs (no vllm.cpp), ~85 lines. One trivial op
(`ttnn::multiply`, bf16, TILE), captured inside a trace at two grid extents:

- case 1: [32,32] interleaved (1-core extent),
- case 2: [32×110, 64] HEIGHT_SHARDED on L1 across the FULL 11×10 = 110-core
  compute grid of the P150 (shard grid verified in-process: `memory layout=2
  buffer=1 shard grid cores=110`),
- case 3: 8 launches of case 2 inside one trace.

Trace bytes read from `MeshDevice::get_trace_buffers_size()` after
`end_trace_capture` and before `release_trace` (live total,
`tt_metal/distributed/mesh_trace.cpp:62` adds `padded_size` on commit and
`trace_buffer.cpp:24` subtracts on release, so the reading is exactly the one
trace's padded size; `MeshTraceBuffer::desc->total_trace_size` is not
reachable through any installed header).

Program cache enabled; each variant warmed before capture (capture refuses
new binaries, `mesh_workload.cpp:201-205`). Run recipe:
`flock /home/lu_zero/gpu.lock`, `reset` + 15 s, `TT_METAL_HOME` /
`LD_LIBRARY_PATH` at `~/Sources/tt/tt-metal-pin/build_release_script`
(pin `6449cf13f7b` ≈ upstream `98134127a7b` + local series), aarch64
clang/gnu-16. Log: `/tmp/ttrace-repro3.log`, `EXIT=0`.

## 2. Measured

| op | grid extent | trace bytes |
|---|---|---|
| ttnn::multiply [32,32] bf16, 1 launch | 1 core (interleaved) | 17,408 |
| ttnn::multiply [3520,64] bf16, 1 launch | 110 cores (full-grid sharded) | 17,408 |
| ttnn::multiply [3520,64] bf16, 8 launches | 110 cores | 139,264 (17,408/launch) |

**No grid scaling.** The full-grid program records byte-identical trace size
to the 1-core program, and per-launch cost is grid-independent: the 8-launch
trace is exactly 8 × 17,408. Stock tt-metal records a full-grid program's
command sequence at the RmsNorm-class KB floor (17.4 KB padded, i.e. ~2–16 KB
unpadded — the same floor the attribution doc measured for our RmsNorm
record).

## 3. Verdict

The task's stop condition fired: a full-grid trivial op does NOT reproduce
the MB-per-launch class, so the generic "per-launch config pages × grid"
framing is wrong as an upstream ask. tt-metal already records a stock
multi-core program compactly — consistent with the packed relay
(`add_prefetch_relay_paged_packed`, `dispatch.cpp:2079-2116`) collapsing
identical per-core config pages. **No issue filed.**

The 27B residual (~2.82 MB × 1,037 launches) must be specific to our
keepquant program's config structure, not to grid extent per se. The
attribution doc's own dispatch-log census already points there: the captured
window contains ~250 one-shot "Writing Program Command Sequence" fetches per
launch. Candidate discriminators, in order: (a) number of distinct kernel
config pages per launch (the fused chain instantiates more kernels than one
ttnn op), (b) number of distinct CB config pages (many circular buffers vs
the op's two), (c) failure to hit the packed relay because pages are not
byte-identical across cores (per-core-varying content), (d) program size
exceeding the packed-path threshold and falling back to per-page relay.

## 4. Re-scope

The upstream ask is dead in its current shape. The next leg is local: bisect
keepquant's per-launch recorded bytes against a synthetic program that varies
(a)-(d) one at a time on this same harness (a compile-time N-kernel/N-CB
program, same 110-core grid). If a synthetic with our config-page count
reproduces MB/launch while an equal-grid 1-kernel/2-CB program stays at KB,
the finding is a config-page-count × pages-not-packed gap worth either an
upstream report (with the synthetic) or a local program-shape fix. Until
then this row's remaining trace-fit wall keeps its recorded bound.

Reproducer: `repro_trace_config_pages.cpp`; build and run recipe in §1.
