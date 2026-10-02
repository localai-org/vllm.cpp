# tt trace-record audit — what the 3.15 GB staging demand actually is (2026-09-28)

Audit of the 27B decode trace-record composition. Evidence:
`/tmp/leg-region-c1-fix.log` (the 64-region c1 leg, 2026-09-28 14:52),
`docs/bench-evidence/tt-region-capture-20260928.md` at `180befff5` (the 2-region
focused census), the whole-graph numbers in
`.agents/issues/BACKEND-TENSTORRENT-QWEN35/ISSUE-LOCAL-01M3JXEFQKSZP23PP2HWY9G0VQ.md:83,129-134`
(file committed at `ec4e8a824`; the issue directory currently carries
`ISSUE-LOCAL-01M3918KQ580Z3NHVRNXVF15FZ.md`), and the tt-metal pin
`~/Sources/tt/tt-metal-pin` (read-only). No device legs were run; source and
existing logs suffice.

## 1. The arithmetic model comparison

Raw numbers:

- Whole graph (issue :83,:129-132): `end_trace_capture` asks for one
  **3,153,969,152 B** staging buffer; the captured decode graph records
  **1,037 tt-metal commands** → mean **3,041,286 B/command**.
- Focused 2-region census (region-capture evidence, "The region handoff"):
  region 0 (one `RmsNorm` command) closes at **2,048 B**; region 1 (one
  `MatmulBT` command) closes at **3,088,384 B**.
- 64-region leg: 8 segments closed sequentially, the 8th close dies in
  `populate_mesh_buffer` (`mesh_trace.cpp:118-125`) at allocation high-water
  4,229,506,816 B. The per-segment close sizes were not yet logged in that leg
  (the per-segment staging-delta census landed in the test harness at
  `180befff5`, after the leg), so the leg constrains the total, not the shape.

Three models fitted:

| Model | Prediction | Verdict |
|---|---|---|
| A. per-command uniform | every region ≈ 3.04 MB/command | **refuted by region 0**: one full RmsNorm close = 2,048 B, 1,487× below the mean |
| B. fixed-per-trace/region F + small c | from the focused test F + c ≤ 3,090,432 B; 64 regions ≤ 198 MB. To reach 3.15 GB needs F ≈ 49.3 MB/region — 16× the largest region close ever measured | **refuted at measured magnitudes** |
| C. per-command heterogeneous: ~2 KB dispatch headers + inline H2D payload wherever a capture-scope upload fires | region 0 = headers only (✓ 2,048 B); region 1 = headers + ~3.086 MB payload; 1,037 × 3,088,384 B = 3.20 GB ≈ the whole-graph 3.15 GB within 1.5% | **supported, high confidence** |

Region 1's 3,088,384 B decomposes exactly as 2,048 B of command headers plus a
3,086,336 B payload — a byte count consistent with one bf16 tensor of
~1.54 M elements written host→device during capture (3,088,384 = 2 × 1,544,192;
bf16 payloads are recorded byte-exact, see §2). The whole-graph mean
(3,041,286 B) sits 1.5% below that single-command measurement, i.e. the 3.15 GB
is ~99.9% per-command inline payload, not fixed overhead. If the 64-region
fixed-per-trace hypothesis were right, the whole-graph number would have to be
~64 × 3.09 MB ≈ 198 MB — the leg's 4.23 GB high-water at segment 8 already
exceeds 21 regions × 49 MB, and the focused test measured a complete region
close (staging level before/after) at 3.09 MB, not 49 MB.

**The data supports per-command-linear with a ~3 MB dominant term. Confidence:
high** on the model choice (two independent single-command measurements bracket
the whole-graph mean); the exact payload identity per command in the 27B graph
is unmeasured (§4's attribution leg closes that).

## 2. What tt-metal puts in the trace buffer per command

Read path, pin `~/Sources/tt/tt-metal-pin`:

- The trace buffer's content is the host-recorded dispatch command stream:
  `tt_metal/distributed/mesh_trace.cpp:56-59` sizes it from
  `MeshTraceDescriptor::total_trace_size`, and `:155-172` writes
  `mesh_trace_data.data` (uint32 command words) into the DRAM buffer. The
  fatal the leg hit is the top-down DRAM overlap check at
  `mesh_trace.cpp:108-152` (message at `:118-125`).
- Recording is sysmem bypass mode: `FDMeshCommandQueue::record_begin` sets
  `set_bypass_mode(true)` at `tt_metal/distributed/fd_mesh_command_queue.cpp:1350`;
  everything the queue issues during capture is appended to
  `bypass_data` and reduced into `ordered_trace_data` at
  `fd_mesh_command_queue.cpp:1660-1665` (per-device `max_trace_size`).
- Per program launch in bypass: `fd_mesh_command_queue.cpp:449-462` creates a
  trace node; `program_dispatch::create_trace_node`
  (`tt_metal/impl/program/dispatch.cpp:3520`) snapshots RTAs, circular-buffer
  configs, dataflow-buffer configs, and cross-node config pages — KBs.
- **Kernel binaries are NOT embedded per command.** The binary command
  sequence emits relay commands that reference the program's persistent
  storage: `add_prefetch_relay_paged` pointing at the program's
  `kernels_buffer` DRAM pages when uncached
  (`tt_metal/impl/program/dispatch.cpp:1942-1961`), or
  `add_prefetch_relay_ringbuffer` pointing at the prefetcher ring buffer when
  the program fits the cache (`:1961-1971`). The cache is the 1,024 KB
  prefetch ringbuffer (`tt_metal/impl/dispatch/util/dispatch_settings.cpp:72`;
  shrunk to 67 KB only when two CQs share one dispatch engine, `:113`), and the
  fit decision is `max_program_kernels_sizeB <= prefetcher_cache_sizeB` at
  `fd_mesh_command_queue.cpp:453`. **Data confirms this**: region 0's complete
  2,048 B close cannot carry any compiled binary, so binaries ride by
  reference.
- What IS copied inline: any H2D write issued during capture. Bypass mode
  records the write command with its payload so the replay re-writes the same
  bytes. This is the only MB-scale per-command mechanism in the record path.
- Fixed per-trace cost: `record_begin`/`record_end` wrappers, the
  `exec_buf_end` epilogue (`fd_mesh_command_queue.cpp:1661`), completion and
  event-reset bookkeeping (`:1691-1705`) — bytes to KB, consistent with region
  0's 2,048 B.

So tt-metal's minimal command costs ~2 KB (headers + go signal + snapshot
configs); the fixed per-trace cost is the same order. **A 3 MB command is not
tt-metal overhead — it is a captured inline write.**

## 3. Our capture path — staging findings

- `EnsureDevice2D` (`src/vt/tenstorrent/tenstorrent_residency.cpp:337`)
  lazily uploads via `from_vector` whenever a tensor is not yet device-current.
  If the first touch happens inside a capture scope, the whole tensor is
  recorded inline into the trace. There is no refusal or debug print on this
  route (the `tt_capture_active()` guards there only divert reshape paths).
- Broadcast `AddKernel` builds a replicated `[rows, d]` host tensor and
  `from_vector`s it unconditionally — with an explicit
  "[TT-UP] AddKernel from_vector WRITE during capture" debug line
  (`src/vt/tenstorrent/tenstorrent_ops.cpp:113-133`). A rows×d×4 B payload is
  inlined per captured Add.
- `WarmDecodeIds` refreshes the ids device buffer with a `copy_to_device`
  per step, including the capture step
  (`src/vt/tenstorrent/tenstorrent_capture.cpp:314-317`) — small (n × 4 B).
- Six `EnsureHostBytes DURING CAPTURE` fired in the c1 leg
  (log `/tmp/leg-region-c1-fix.log:1006-1011`; print at
  `tenstorrent_residency.cpp:1114`). These are blocking device→host readbacks
  inside capture — a capture-scope sync hazard, but they add no trace bytes.
- Region-1 composition explained: the focused test's `MatmulBT` command
  recorded 2,048 B of headers plus one ~3.086 MB inline H2D payload —
  byte-count consistent with the matmul's bf16 weight (~1.54 M elems) being
  uploaded during capture via the lazy staging route instead of warmed before
  it.

## 4. Conclusion and the lever

**Per-command dominates; fewer/bigger traces is the wrong lever.**
Quantified best cases:

- If the ~3 MB/command term is an inline capture-scope upload, eliminating it
  drops every command to region 0's measured floor: 1,037 × ~2 KB ≈
  **2.1 MB total** — three orders of magnitude under the 3.15 GB wall, and the
  whole-graph capture fits 2.2 GB trivially. This is the actionable lever:
  attribute then hoist the upload (warm every tensor before
  `TraceBeginCapture`; refuse/flag any `from_vector`/`EnsureDevice2D` upload
  with `tt_capture_active()`, the way `tenstorrent_ops.cpp:130` already
  prints for AddKernel's broadcast route).
- The fewer-traces alternative (capture the biggest 8 layers whole) does not
  survive the arithmetic: even taking the measured per-command mean at face
  value, 8 layer-regions ≈ 130 commands × 3.04 MB ≈ 395 MB — it only buys a
  ~5.6× demand cut by dropping 87% of the commands, and it is dominated by the
  upload fix, which buys ~1,500×.

Discriminating leg (cheap, no new mechanism): rerun the focused 2-region case
with `VT_TT_TRACE_DEBUG=1` plus a print added on `EnsureDevice2D`'s upload
route; region 1's payload attribution line either names the weight upload or
falsifies §3's reading. The committed per-segment staging-delta census
(`180befff5`) gives the per-region denominators for the same leg.

No tt-metal record-size ask is indicated by this data: the record path is
already minimal for warmed programs (region 0 = 2,048 B proves it).
