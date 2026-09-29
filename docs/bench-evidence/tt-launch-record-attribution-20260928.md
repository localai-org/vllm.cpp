# tt launch-record attribution — what the ~3.04 MB per captured command is (2026-09-28)

**UPDATE 2026-09-29 (advanced pin, logging-enabled discriminator).** The pin
advanced from `9161e8fdb27+4` (head `d20b8e27f29`) to upstream-live
`98134127a7b` + the same local series rebased (pin head `6449cf13f7b`, branch
`vllm-cpp-pin/20260923-adv`), with `TT_METAL_ENABLE_LOGGING=ON` in a dedicated
`build_logging` build dir (env `TT_METAL_LOGGER_LEVEL=TRACE`,
`TT_METAL_LOGGER_TYPES=Dispatch`, names per
`build_logging/include/tt-logger/tt-logger.hpp:98,182`). Upstream did NOT
change the per-core trace record between the pins (576+ commits:
`create_trace_node`/`issue_queue_reserve` in
`tt_metal/impl/program/dispatch.cpp` untouched; the only nearby changes are
sub-device setup caching `262a365421f` and trace-allocation-tracker fixes
`c05eff45369`, neither of which touches the per-core record). The focused leg
on the new pin reads `region 0 = 2,048 B; region 1 = 2,965,504 B`
(`/tmp/tregion-newpin.log`, case 1/1 passed, 1,032/1,032 assertions) — the
pin move shaved ~4% off the record; the mechanism stands. The logger
discriminator closes the open question: the capture window is exactly 254
`Writing Program Command Sequence` one-shot fetches summing 2,961,024 B
(= region 1 + region 0 + trace header pages, `/tmp/burst.log` extraction of
`/tmp/tregion-newpin.log`), and the window contains **11,040 per-core Unique
RTA (UNICAST) writes** at 40–48 B payload each plus 110 full-grid CB/DFB
config pages — i.e. the record is OUR keepquant program's per-core
`SetRuntimeArgs` stream
(`src/vt/tenstorrent/tenstorrent_keepquant.cpp:2108-2130`: 12 words per core,
of which 10 are shape-global constants and only `r0`/`rc` vary, and those two
are `c*tcols` and its clamp — computable in-kernel from the core coordinate).
Verdict flips to **OURS** (see §3 below for the tt-metal-side framing this
update refines): replacing the per-core `SetRuntimeArgs` with in-kernel
`r0/rc` derivation + `SetCommonRuntimeArgs`-only launches drops the record to
the RmsNorm-class floor (~2–16 KB per captured launch, ~200×), which closes
the whole 27B decode-trace DRAM-fit site.

---

Worktree `row/tt-27b-region-capture-spec` at `5f7ac1fa7` (probes were
uncommitted scratch, reverted before the commit). Fixes the open candidate from
[tt-capture-upload-guard-20260928.md](tt-capture-upload-guard-20260928.md):
the 27B whole-graph trace demand is byte-identical 3,153,969,152 B ≈ 1,037 ×
~3.04 MB, capture-scope uploads are eliminated (zero fires), and the leading
hypothesis was per-launch kernel-binary relay for programs that miss the
1,024 KB prefetch ringbuffer. Device evidence: focused legs under
`flock /home/lu_zero/gpu.lock` on thalia (P150), the region-handoff doctest
(`tests/vt/test_tenstorrent_backend.cpp:11376`, reproduces census `region 0 =
2,048 B; region 1 = 3,088,384 B`), run under `gdb -batch` with breakpoints on
the tt-metal pin's (`~/Sources/tt/tt-metal`, rev `d20b8e27f29`, the exact tree
the release libs were built from) dispatch internals. Logs: `/tmp/tregion2.log`
(census reproduction), `/tmp/tt-pcs-census3.log` (per-sequence census),
`/tmp/tt-cw2.log` / `tt-iq3.log` (per-chunk command-queue census),
`/tmp/tt-iq4.log` (chunk backtraces), `/tmp/tt-wb3.log` / `tt-iqc.log`
(capture-window-gated probes).

## 1. The binary-relay hypothesis is REFUTED at source

- The prefetcher "ringbuffer" is 1,024 KB
  (`tt_metal/impl/dispatch/util/dispatch_settings.cpp:72`, shrunk to 67 KB only
  when two CQs share a dispatch engine, `:113`); the fit decision is
  `max_program_kernels_sizeB <= ringbuffer_size()` per MeshWorkload
  (`tt_metal/distributed/fd_mesh_command_queue.cpp:453`,
  `tt_metal/distributed/mesh_workload.cpp:272-275`).
- Either way, the recorded command stream carries the binary BY REFERENCE, not
  by value: `add_prefetch_relay_paged` sub-commands pointing at the program's
  DRAM `kernels_buffer` pages when the program does not fit
  (`tt_metal/impl/program/dispatch.cpp:1942-1961`, multicast
  `add_prefetch_relay_paged_packed` `:2079-2116`), or
  `add_prefetch_relay_ringbuffer` when it does (`:1961-1971`). The relay reads
  device DRAM at replay; the bytes never enter the host-recorded `bypass_data`
  (`fd_mesh_command_queue.cpp:1660-1665`). Data point in-tree: region 0's
  complete RmsNorm record is 2,048 B — no room for any compiled binary, so
  binaries ride by reference for cached programs of any size.
- tt-metal additionally REFUSES first-time binary loads during capture by name:
  `MeshWorkloadImpl::load_binaries` TT_FATALs "Cannot load new binaries during
  trace capture … Warm up before capturing a trace"
  (`tt_metal/distributed/mesh_workload.cpp:201-205`). Our warm pass covers
  this; the focused legs never hit it.
- OUR kernel sizes: the compiled keepquant device kernel (both cache variants,
  `~/.cache/tt-metal-cache/192705464604149581/kernels/Kernel_Source_Code/
  {6195984949586344520,16729977526042552009}/brisc/brisc.elf`) measures
  `.text` 51,648 B + `.data` 3,372 B (`size(1)`) — **18× under the 1,024 KB
  ringbuffer threshold**. (The unstripped ELF artifact is 1,124,984 B, but that
  size is `.rela`/symbol tables, not the transfer payload class the relay
  commands reference.) The "one of our keep-quant binaries exceeds the
  ringbuffer so the full binary is relayed into the trace per launch"
  mechanism does not exist on this pin.

## 2. Device attribution of region 1's 3,088,384 B

Method notes: the release libs have `TT_METAL_ENABLE_LOGGING=OFF`
(`build*/CMakeCache.txt`), so `TT_METAL_LOGGER_LEVEL=TRACE` yields zero
`tt::LogDispatch` output (verified: the logger leg produced only the doctest
lines) — the dispatch-log route needs a pin rebuild and was not taken.
Instead the legs breakpinned the pin's dispatch internals directly:

- **The record is pure command-stream chunks, not data writes.** Gating
  `SystemMemoryManager::issue_queue_reserve` on our `tt_capture_active()` flag
  (`/tmp/tt-iqc.log`): the capture window contains **284 reserve chunks and
  nothing else**; gating `buffer_dispatch::write_to_device_buffer` the same way
  (`/tmp/tt-wb3.log`): **zero real in-capture buffer-data writes** (the only 2
  hits register-size 0). So no H2D payload — ours or tt-metal's — is recorded
  inline in the MatmulBT region, consistent with the zero `[TT-UP]`/`[TT-KQ]`
  census of the guard leg.
- **The 3,088,384 B is those 284 chunks.** `issue_queue_reserve` is exactly how
  the bypass buffer grows in trace recording
  (`tt_metal/impl/dispatch/system_memory_manager.cpp:519-524`); contiguous runs
  of the leg's chunk list sum to exactly 3,088,384 B (and to the 2,048 B of
  region 0 as a single chunk).
- **The class is the program's own recorded launch stream, scaled per core.**
  Region 1 records ONE `MatmulBTQuantGrouped` launch whose program spans the
  FULL worker grid (one `CreateKernelFromString` over
  `CoreRange{0,0}-{grid.x-1,grid.y-1}`, full-grid CBs, per-core
  `SetRuntimeArgs` — `src/vt/tenstorrent/tenstorrent_keepquant.cpp:2017-2136`),
  while region 0's RmsNorm program (ttnn, few cores) records 2,048 B total. The
  pin's per-program record path writes per-sequence chunks through
  `write_data_to_cq` (`tt_metal/impl/program/dispatch.cpp:3443-3512`), and the
  trace node snapshots per-core-range RTA/CB/DFB config payloads
  (`create_trace_node`, `dispatch.cpp:3520-3651`) — the MB scale rides with the
  keepquant program's per-core configuration/RTA footprint (284 chunks ≈ 2 ×
  ~142 grid cores at ~10.9 KB average), not with headers, relay payloads, or
  any tensor upload.

## 3. Verdict

| Class | Bytes (region 1) |
|---|---|
| Command headers + configs for a small program (region 0 floor) | ≤ 2,048 B |
| Inline H2D payload (audit model C) | **0** — zero in-capture buffer writes |
| Kernel-binary relay payload | **0** — relay is by reference; kernel is 18× under the ringbuffer threshold anyway |
| The keepquant program's own recorded per-core launch stream | **~3,086,336 B** (284 chunks, ~10.9 KB each) |

**Fix locus: tt-metal-side.** The record scales with per-core dispatch writes
for a full-grid custom-kernel program; the shrink levers are (a) tt-metal
recording per-core config/RTA pages once per program (or by reference) instead
of per launch, or (b) tt-metal's prefetcher-cache path covering the custom
kernel's config stream. Expected win if the record drops to the RmsNorm-class
floor: region record 3,088,384 → ~2-16 KB per command, whole-graph
3,153,969,152 → tens of MB — the entire 27B decode-trace DRAM-fit site closes.

**Our-side partial lever (not a fix):** shrinking the keepquant grid (fewer,
fatter cores) scales the record roughly linearly with core count; halving the
grid halves a 3.15 GB demand — still an OOM, so it only bounds the mechanism,
it does not close the site. Refuse to ship a "fix" that only re-shapes the
program.

## Open discriminator (one step)

Which per-sequence class dominates the 284 chunks (per-core config-buffer page
vs per-core RTA write vs packed-binary sub-commands) needs one leg against a
logging-enabled pin build (`TT_METAL_ENABLE_LOGGING=ON` +
`TT_METAL_LOGGER_LEVEL=TRACE`, env names verified at
`build_Release/include/tt-logger/tt-logger.hpp:98,182`) or an equivalent
`issue_queue_reserve` caller-tag patch. The class attribution above (per-core
launch stream, tt-metal-side) does not depend on it.

## What falsified what

- "program does not fit the 1,024 KB ringbuffer → full binary relayed through
  the trace per launch": **falsified** — relay is by reference in both arms
  (`dispatch.cpp:1942-1971`), and the keepquant kernel is 55 KB against a
  1,024 KB threshold.
- "config/RTA page writes scaling with tensor geometry": **refined** — the
  scaling is with the program's CORE-GRID footprint (per-core dispatch
  writes), not tensor geometry; the payload is the recorded command stream
  itself.
- Audit model C (inline H2D payload): **falsified on-device** — zero
  in-capture buffer writes despite the exact 3,088,384 B close.
