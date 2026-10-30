# Spike: expert streaming from disk (`ENG-EXPERT-STREAM`)

Run MoE models with less resident GPU/unified memory by keeping routed-expert
weights on fast storage (NVMe) and paging them in on demand, keyed on router
output, under a byte-budgeted resident cache. User-directed spike
(2026-07-10); design grounded in a full source scan of **antirez/ds4**
(DwarfStar, github.com/antirez/ds4) plus measured dgx.casa NVMe numbers and
the actual 35B-A3B checkpoint shapes. Related mirror-floor inventory row:
`ENG-WEIGHT-OFFLOAD` (vLLM UVA `cpu_offload_gb`, stays a separate future
mirror port).

Verdict up front (from the bandwidth math in §3, all inputs measured):

- **Low concurrency (c1-c4): VIABLE as a capacity feature, conditional on the
  bank-only loader and fixed-slot Marlin design below.** At a 50%
  expert-resident fraction the 35B-A3B needs ≤ 270 MiB of expert reads per
  token worst-case; the dgx NVMe sustains 5.0-5.3 GB/s with a small read
  pool, an I/O-only bound of ≥ 17.7 tok/s before locality. It can free ~8.4 GiB
  (f=0.5) to ~12.7 GiB (f=0.25) only if streaming mode never materializes the
  current per-expert host `OwnedTensor` copies or the full device Marlin
  resident; keeping either would erase the capacity win on unified-memory GB10.
- **High concurrency (the MVP gate operating point): NOT SERVED.** At
  conc≥32 every step touches ~64-98% of all 256 experts per layer
  (1-(1-8/256)^B), so per-step I/O approaches (1-f) x 16.9 GiB regardless of
  reordering — orders of magnitude below the gated ~2.8k tok/s. The feature
  must refuse/warn at high concurrency, never silently degrade the gate.
- **tmpfs//tmp is NOT a disk tier on GB10**: unified memory means tmpfs pages
  ARE the same LPDDR5X the GPU uses (and dgx's `/tmp` is ext4 on the same
  NVMe anyway — verified `df -hT /tmp` 2026-07-10). On discrete-GPU hosts a
  host-RAM tier IS meaningful (PCIe ~25-60 GB/s, ~5-10x NVMe) — kept as a
  pluggable backing-store leaf (W7), the expert-granular analogue of vLLM's
  blanket `cpu_offload_gb`.

## Scope

| Field | Content |
|---|---|
| Row IDs | `ENG-EXPERT-STREAM` (this spike; work-breakdown leaves W0-W8 below). Mirror-floor context row: `ENG-WEIGHT-OFFLOAD` (INVENTORIED, not claimed here) |
| In | Routed-MoE expert weights only (gate/up/down per expert per layer) for MoE models, first target Qwen3.6-35B-A3B NVFP4 (safetensors): versioned NVMe Marlin-layout expert bank, bank-only loader, fixed-capacity contiguous Marlin slot arrays, logical-expert→slot remap after router D2H, async O_DIRECT pread/copy pool, hotness-decayed-LFU + LRU-tiebreak eviction with in-flight/selected protection, chunked slot sweeps for long prefill + decode-cache seeding, optional hotlist preload, `--simulate-used-memory`-style honest measurement mode |
| Out (this row) | Dense weights, shared experts, router, norms, KV cache (all stay resident); GGUF expert streaming (a follow-up leaf, but NOT for the reason first recorded — see the GGUF reconciliation note below: the slicer LANDED 2026-07-22 and only the residency policy remains); host-RAM tier for discrete GPUs (W7, after gate); vLLM UVA `cpu_offload_gb` mirror (own row `ENG-WEIGHT-OFFLOAD`); expert-parallel EPLB (`PAR-EP-EPLB`) |
| Supported modes | `off` (default, unchanged engine); `nvme` (expert bank file + device cache). Budget accepted as expert count or `NGB` (ds4 CLI semantics `--ssd-streaming-cache-experts 32GB`, ds4_ssd.c:46-70); auto budget = fraction of free device memory minus non-streamed needs (ds4_ssd.c:80-106) |
| Dispatch behavior | Streaming engages only when enabled AND the model is MoE. It branches before `BuildMoeMarlinResident`, never builds the full `[E,...]` resident, copies router IDs device→host and synchronizes once per MoE layer in phase 1, maps logical IDs to fixed cache slots, then runs the unchanged dense-stride Marlin kernel over slot IDs. Engine WARNS and refuses (or auto-disables per config) when max concurrency exceeds the regime bound (§3, default warn at conc>4, hard cap configurable); dense models reject the flag |
| Regimes served | c1-c4 capacity/single-user; models larger than device memory. Explicitly NOT the high-concurrency throughput gate |

## ds4 anatomy (what we are porting from)

**What ds4 is.** DwarfStar: antirez's self-contained C inference engine (no
ggml link; kernels/quants adapted from llama.cpp under MIT, README.md:46-57)
for exactly two models: DeepSeek V4 Flash (43 layers, 256 routed experts
top-6 + 1 shared, 3 hash-routed layers, ds4.c:180-196) and DeepSeek V4 PRO
(61 layers, 384 experts top-6, ds4.c:214-232), custom GGUFs only, backends
Metal (primary), CUDA (incl. DGX Spark/GB10) and ROCm/Strix Halo
(README.md:14-17). SSD streaming is its capacity mode: "non-routed model
weights stay resident, while routed MoE experts are kept in an in-memory
cache and loaded from the GGUF file on cache misses" (README.md:180-216).
README calls it Metal-only but the code enables CUDA/Linux and ROCm too
(ds4.c:79-90 `ds4_backend_supports_ssd_streaming`; commit bbd069d "Add CUDA
and ROCm SSD streaming").

**Design, precisely (file:line in the ds4 tree @ 80ebbc3):**

- **Weight layout / reads**: single GGUF, experts live in per-layer 3D
  tensors; a per-expert slice is `tensor_bytes / n_expert` at
  `abs_offset + expert_id * per_expert_bytes` (ds4.c:14048-14060) —
  per-expert bytes must be uniform within a layer
  (`weights_streaming_layer_experts_uniform`, ds4.c:3338-3356; mixed
  per-LAYER quants allowed, e.g. Q4 "boosted" last layers among IQ2, commit
  81f35e7). Reads are plain `pread(2)` against the model fd
  (ds4_metal.m:7776-7820); macOS gets `F_RDADVISE` page-cache readahead
  hints (ds4_metal.m:7661-7686); the CUDA path has an O_DIRECT variant with
  aligned staging (ds4_cuda.cu:1055-1104) then `cudaMemcpyAsync` on an
  upload stream (ds4_cuda.cu:1262).
- **Async I/O**: a persistent pthread pool, default 9 threads, cap 18
  (`DS4_METAL_STREAMING_EXPERT_PREAD_THREADS`, ds4_metal.m:7754-7773,
  7836-7848); one task per missing tensor — up to 6 experts x 3 tensors = 18
  tasks per layer (ds4_metal.m:7699-7723 `ds4_gpu_stream_expert_pending_load`).
- **When selection is known vs when weights are needed (the prefetch
  window)**: selection is only known after the layer's router. ds4 splits
  the GPU command buffer at each MoE layer, reads back the 6 selected ids
  (or computes the router ON CPU from the read-back pre-FFN norm
  activations, `metal_graph_decode_cpu_router`, ds4.c:14170-14247), then
  `ds4_gpu_stream_expert_cache_begin_selected_load` starts pool preads for
  the missing experts (ds4_metal.m:10618-10800) while the GPU continues; a
  masked address-table kernel computes the RESIDENT experts' contribution
  first and a deferred pass computes the MISSING ones after
  `pending_load_finish` waits on the pool (split path,
  ds4_metal.m:23392-23433; timing counters resident_submit/missing_wait,
  ds4.c:2608). Exception: DeepSeek's 3 hash layers route by token id alone
  (`ffn_gate_tid2eid`, ds4.c:7307-7320, 14020-14076), so their loads can
  begin before any compute — a true lookahead ds4 exploits; Qwen3.6 has no
  hash layers, so our window is strictly within-layer.
- **Caching/eviction**: per-(layer,expert) entry table `[61][384]` backed by
  slab-allocated buffers (256 slabs, ds4_metal.m:370-458); mlocked with
  graceful budget-capping when mlock fails (ds4_metal.m:8108-8205, commit
  7a77a28; README.md:246-256). Eviction is NOT plain LRU: victim = lowest
  **route-hotness** with LRU tiebreak, where hotness += 1 for every
  *selected* expert (hit or miss) and is halved every 16 decode tokens
  (`DS4_METAL_STREAM_EXPERT_HOTNESS_DECAY_TOKENS`, ds4_metal.m:376,
  8678-8776, 9666-9830). Recorded rationale: hit-count LFU "penalizes
  experts that are repeatedly selected but evicted before a second hit"
  (ds4_metal.m:9679-9683). Entries referenced by in-flight GPU command
  buffers (sequence-tracked, ds4_metal.m:834-905) or currently selected are
  eviction-protected. CUDA backend uses simple LRU ticks + device slabs,
  default budget 512 experts (ds4_cuda.cu:38, 156-198, 1967, 2011-2098).
- **Budget**: `--ssd-streaming-cache-experts N|NGB` (GiB converted via
  per-expert bytes, ds4_ssd.c:46-78); auto plan = 80% of the recommended
  working set minus non-routed bytes (ds4_ssd.c:80-106).
- **Prefill is a different regime and ds4 knows it**: long prefill touches
  ~all experts, so streaming prefill switches to **full-layer sequential
  streaming** above a per-quant token cutoff with a lookahead thread
  pipelining the next layer's expert tensors (ds4.c:11696-11760, commit
  57b8a4c), and afterwards **seeds the decode cache from the last ≤64
  prefill tokens' router selections** (`DS4_STREAMING_PREFILL_CACHE_SEED_MAX_TOKENS`,
  ds4.c:10310, 13812-13824, 19657).
- **Hot experts**: an expert-locality profiler records per-layer selection
  histograms + weights, simulates latest-N-unique caches for caps 1..384 and
  measures adjacent-token overlap/Jaccard (ds4.c:750-830); its output ships
  in-source as default hotlists (13,320 (layer,expert) pairs across PRO +
  Flash, ds4_streaming_hotlist.inc:1-3) used to pre-warm the cache at
  startup, capped at 4096 experts in auto mode (ds4.c:13949-14016);
  `--ssd-streaming-cold` disables preload for measurement.
- **Honest measurement tooling**: `--simulate-used-memory` mlocks N GiB
  before load so page-cache effects can't fake streaming results
  (ds4_ssd.c:108-181).
- **Measured numbers ds4 documents**: sparse. README.md:246-249 (M5 Max
  128 GB, PRO q2): the automatic budget (~59 GB expert cache) beat manual
  64-75 GB settings on a short streaming decode bench; no absolute tok/s or
  hit rates are published — the profiler exists to measure them per
  deployment. Non-streamed reference points: PRO q2 9.56 tok/s decode @32k
  (M3 Ultra 512 GB), Flash q2 13.75 tok/s decode on DGX Spark GB10
  (README.md:162-175). Hit/miss/pread-bytes/pread-ms are runtime log counters
  (ds4.c:2738-2828), not published results.

**Portable to us**: the resident/streamed split (non-routed resident, routed
experts tiered); byte-or-count budget + auto plan; hotness-decayed-LFU
eviction with in-flight protection; the pread pool; a synchronous phase-1
router-readback/load boundary; chunked prefill streaming + last-K decode seed;
locality profiler + hotlist preload; the simulate-used-memory honesty tool;
CLI semantics. ds4's resident-first/missing-deferred overlap is a later
candidate, not a mechanical port: our selected GB10 Marlin kernel uses dense
expert strides and has no address table or pair-mask input.
**ds4-specific, not portable**: Metal no-copy mmap buffer wrapping +
`F_RDADVISE`; hash-layer lookahead (DeepSeek `tid2eid`; Qwen3.6 has none);
top-6 bitmasks (we are top-8). GGUF 3D-tensor slicing arithmetic was listed
here as non-portable on the grounds that safetensors, where each expert is a
separate tensor, is simpler for us. That is now WRONG in both directions and is
corrected below.

## Upstream chain

Pinned vLLM `e24d1b24` (verified `git log -1` 2026-07-10). The three senses
of "streaming", settled with citations — one-line verdicts:

1. **LOAD-TIME weight streaming (fast startup, full residency after): PRESENT
   in-pin — NOT this feature.** `LoadFormats` registry incl. `runai_streamer`
   (`vllm/model_executor/model_loader/__init__.py:33-49`, loader map :50-66);
   `RunaiModelStreamerLoader`
   (`vllm/model_executor/model_loader/runai_streamer_loader.py:21`;
   iterator `weight_utils.py:987`); safetensors mmap/lazy strategy
   (`vllm/config/load.py:63-84`), `sharded_state_loader.py`, fastsafetensors
   (`weight_utils.py:1024`, `default_loader.py:267-268`). Mirror obligation
   already tracked by `LOAD-SAFETENSORS`/`LOAD-LONGTAIL`.
2. **INFERENCE-TIME CPU weight offload (`cpu_offload_gb`): PRESENT in-pin,
   blanket per-parameter, v1-supported, NOT expert-router-aware.**
   `UVAOffloadConfig.cpu_offload_gb` (`vllm/config/offload.py:23`); UVA
   offloader `_maybe_offload_to_cpu` per-parameter pinned+zero-copy
   (`vllm/model_executor/offloader/uva.py:64,80-108`); creation
   (`offloader/base.py:126-162`), installed in the v1 worker
   (`vllm/v1/worker/gpu_model_runner.py:445,913` — no v1 guard/raise).
   `cpu_offload_params` allows opt-in name-segment targeting of expert
   tensors (`config/offload.py:34-44`) but offload is static — no router
   keying, no per-token paging decisions. A second backend, layer-group
   `PrefetchOffloader` (`offloader/prefetch.py`; config
   `config/offload.py:47-76`), is CPU-only (`prefetch.py:557-560`). Mirror
   obligation recorded as `ENG-WEIGHT-OFFLOAD` (INVENTORIED; own future
   spike `specs/weight-offload-uva.md`).
3. **INFERENCE-TIME disk/SSD-tier weight paging (expert/layer granular):
   ABSENT in-pin.** Searched `vllm/model_executor/layers/fused_moe/`,
   `vllm/model_executor/offloader/`, `vllm/config/offload.py`,
   `model_loader/` for offload/nvme/io_uring/disk/ktransformers/expert-cache:
   nothing pages weights from disk during decode. `eplb` is expert
   REBALANCING across EP ranks, not offload
   (`vllm/distributed/eplb/eplb_state.py:3-27`); `ep_weight_filter.py:3-8`
   skips non-local experts at LOAD time only; the only disk tier in-pin is
   for KV cache (`vllm/v1/kv_offload/tiering/fs/manager.py:76`), covered by
   `KV-OFFLOAD`.

**Consequence**: `ENG-EXPERT-STREAM` is a **surpass-track** feature (roadmap
"mirror as the floor, surpass beyond it"). Upstream-sync safety: the feature
is (a) strictly additive and default-off, (b) implemented behind our loader +
MoE dispatch seams (no upstream-mirrored file is restructured), (c) its
config namespace is ours (`expert_streaming.*`), kept disjoint from vLLM's
`offload_config` so a future mirror port of `ENG-WEIGHT-OFFLOAD`/sense-2
lands without collision, and (d) recorded here + in the ledger at
implementation time as a surpass divergence, so the recurring sync cycle
(`upstream-sync.md`) diffs cleanly against the pin. If upstream later grows
expert-granular offload, that becomes the mirror target and this row gets
re-based on it (same rule as GGUF vs llama.cpp).

Design-reference chain (non-vLLM, cited per the ground-every-impl rule):
antirez/ds4 @ 80ebbc3 — files/lines inventoried in §ds4-anatomy above. Ports
from ds4 carry `// Ported from: antirez/ds4 <file>:<line> @80ebbc3` headers
exactly like vLLM-derived files.

**Runtime trace plan (required because dispatch is dynamic).** Before W1, run
`nsys profile` on the non-streamed 35B c1 workload and record the actual router,
`moe_align`, Marlin w13/w2, repack, allocation, and synchronization kernels plus
their wall time/launch order. W0 also runs an instrumented router-only probe that
copies `dtid` after `MoeRouterTopK` and proves the device→host event boundary.
At W3, trace streaming-off and phase-1 streaming-on in one same-binary series:
the OFF trace must remain structurally identical; the ON trace must show one
intentional D2H/event wait per MoE layer, bank reads/copies before slot use, and
no full-resident repack/allocation. At W4, trace each prefill slot sweep and
verify every routed pair is computed exactly once. Kernel names and steady-state
times, not source inference, are the acceptance evidence.

## Our baseline

- **Checkpoint (measured from the real safetensors header, dgx 2026-07-10)**:
  `nvidia/Qwen3.6-35B-A3B-NVFP4`, 40 MoE layers x 256 experts, top-8 + 1
  shared (`config.json`: num_experts=256, num_experts_per_tok=8,
  moe_intermediate_size=512, hidden 2048). Per expert per layer: gate
  U8[512,1024] 524,288 B + FP8 scales [512,128] 65,536 B + f32 scale2;
  up identical; down U8[2048,256] 524,288 B + scales 65,536 B + f32 scale2
  → **1,769,484 B ≈ 1.688 MiB**. Routed-expert total is **16.88 GiB of the
  ~22 GiB checkpoint (~77%)**. Each projection is a separate safetensors
  entry, but our reader exposes only an mmap pointer/length through `StTensor`
  (`safetensors_reader.h:13-47`); it does not expose the owning shard path, fd,
  data-section base, or absolute offset for later pread. The bank builder must
  consume those spans while shards are alive or extend the reader metadata.
- **Our MoE execution path (the seams)**: `LoadNvfp4Raw` copies every packed
  weight and scale span into host `OwnedTensor` storage
  (`qwen3_5_weights.cpp:196-223,276-298`), and the model retains those copies.
  CUDA then uploads per-expert originals, repacks ALL E experts once into one
  contiguous Marlin layout — fused w13 `[E, H/16, 4I]`, w2
  `[E, I/16, 2H]`, contiguous scale arrays and `[E]` global scales —
  synchronizes, and frees only the temporary DEVICE originals
  (`qwen3_5.cpp:2392-2549`). Decode runs grouped
  `moe_wna16_marlin_gemm` over `moe_align` outputs
  (router `MoeRouterTopKKernel` `src/vt/cuda/cuda_moe.cu:47-127`;
  `MoeAlignKernel`/`MarlinMoeAlignBlockSize`
  `src/vt/cuda/cuda_marlin_repack.cu:203-293`). The Marlin kernel reads
  `expert_ids_ptr[block]`, indexes `global_scale[expert_id]`, and computes
  `B_expert_off = expert_id * prob_n * prob_k / (pack_factor*4)`
  (`marlin_template.h:543-550`): runtime storage is dense-stride, not a pointer
  table. Router IDs stay on device (`cuda_moe.cu:54-128`).
- **Honest gaps**: (1) nothing in the engine can drop or re-load expert
  weights after load — residency is all-or-nothing; (2) retaining the current
  host `OwnedTensor` expert vectors would consume the bytes streaming is meant
  to free on GB10 even if the device cache were smaller; (3) a naive checkpoint
  pread would still require the Marlin repack/scale-processing kernels per miss,
  including layer-wide combined scale factors; (4) `MoeMarlinResident` is a
  process-static full-E allocation keyed by `MoeBlockWeights*` and must be
  bypassed, not partially populated; (5) router IDs require an explicit D2H
  synchronization before host cache decisions; (6) memory accounting
  (`src/vllm/entrypoints/model_loader.cpp:117-129`) assumes static weights;
  (7) we have no expert-locality data for Qwen3.6 routing; (8) c1 decode tok/s
  on dgx is not yet recorded in the ledger
  (the online-serving campaign `SERVE-GATE-ONLINE` is in flight) — the W6
  campaign must measure the non-streamed c1 baseline first.

### The honest bandwidth/locality math (decides viability)

Measured storage (dgx.casa `/dev/nvme0n1`, ext4 root, 2026-07-10, this
spike): sequential O_DIRECT 16 MiB blocks **5.4 GB/s**; random expert-sized
(1,769,472 B) O_DIRECT preads: **2.76 GB/s @1 thread (0.64 ms/read), 5.02
GB/s @4, 5.03 GB/s @9, 5.29 GB/s @16** — a small pool saturates the disk at
~5 GB/s; commands recorded in state.md. (`/tmp` is ext4 on this same NVMe —
tmpfs would be unified LPDDR5X, i.e. the memory we are trying to save.)

Per-token worst case (batch 1, zero hits): 8 experts x 40 layers x 1.688 MiB
= **540 MiB/token** → I/O-only ceiling 5.0 GB/s / 540 MiB = **8.8 tok/s**.
With hit rate h the bound scales as 1/(1-h):

| resident fraction f (bytes) | uniform-routing h≈f | I/O-only tok/s bound | memory freed vs full residency |
|---|---|---|---|
| 0.25 (4.2 GiB) | 0.25 | 11.8 | 12.7 GiB |
| 0.50 (8.4 GiB) | 0.50 | 17.7 | 8.4 GiB |
| 0.75 (12.7 GiB) | 0.75 | 35.4 | 4.2 GiB |
| 0.90 (15.2 GiB) | 0.90 | 88.5 | 1.7 GiB |

Uniform routing is the conservative cache-hit model; routing skew + hotness
caching + consecutive-token locality may push h above f, but W0/W5 must measure
that rather than assume it. The table is an I/O-only upper bound: c64 TPOT does
not identify c1 compute time, and phase 1 deliberately adds 40 router-readback
waits per token. The **≥12 tok/s at f=0.5** target remains a gate derived from
the 17.7 tok/s I/O ceiling, not a predicted result; W0 records the non-streamed
c1 floor before implementation and no overlap credit is assumed until traced.

High-batch regime (why the gate is out of scope): B independent top-8 draws
touch ~256x(1-(31/32)^B) experts per layer per step — B=16: ~102 (40%),
B=32: ~163 (64%), B=64: ~225 (88%), B=128: ~250 (98%). Per-step miss bytes
approach (1-f) x 16.88 GiB; at f=0.5, B=64 that is ~6.3 GiB/step ≈ 1.26 s
of I/O per step ≈ 51 tok/s aggregate — vs ~2.8k tok/s gated non-streamed
(state.md:1275). No reordering changes the bytes. Streaming therefore serves
capacity, not throughput; the engine must say so at configuration time.

The 27B gate model is dense — this row does not apply to it. The larger
strategic reach: MoE checkpoints BIGGER than GB10's 119 GB unified memory
(DeepSeek-class, GLM-4.x-class at 2-4 bit) become runnable at all — pinned
vLLM has no answer on this hardware (sense-3 absent), which is the
surpass-track headline.

## Port map

New code goes under our seams; no upstream-mirrored file is restructured.
"Ported from" = design source to cite in headers (ds4 @ 80ebbc3).

| Source (design) | Local target | Notes |
|---|---|---|
| ds4_ssd.c:46-106 (budget parse + auto plan), ds4_ssd.c:108-181 (simulate-used-memory) | `src/vllm/model_executor/expert_stream/budget.{h,cpp}`; probe tool `tools/expert_stream/simulate_used_memory.cpp` | count-or-`NGB` parsing, auto plan from free device memory minus non-streamed needs; honesty tool for W6 |
| ds4_metal.m:370-458, 8678-8776, 9653-9830 (entry table, slabs, hotness decay, victim scan, protection); ds4_cuda.cu:156-198, 1967, 2011-2098 (device slabs, LRU) | `src/vllm/model_executor/expert_stream/expert_cache.{h,cpp}` + `src/vt/cuda/cuda_expert_cache.cu` | CPU metadata maps `(layer, logical_expert) -> slot`; CUDA owns fixed contiguous arrays for C Marlin slots (`w13`, `w2`, both processed scales, both global scales), because the kernel addresses `base + slot*stride`. Never allocate unrelated per-entry device pointers. Hotness-decayed LFU (decay every 16 decode tokens), LRU tiebreak, in-flight + selected protection |
| ds4_metal.m:7699-7995 (pread task/pool), ds4_cuda.cu:1055-1104 (O_DIRECT staging), 1262 (async upload) | `src/vllm/model_executor/expert_stream/pread_pool.{h,cpp}` | persistent pool (default 9, cap 18 threads), one fixed-size bank-entry task per missing expert; O_DIRECT into aligned pinned staging, then explicit copy-stream upload into the chosen contiguous slot. GB10 is unified memory but current cache arrays are `cudaMalloc`; copy cost/overlap is measured, never assumed away |
| ds4_metal.m:10618-10800 (begin_selected_load), 23392-23433 (resident/missing split) | streaming branch beside `MoeBlockFusedMarlinCuda` in `src/vllm/model_executor/models/qwen3_5.cpp`, before the current full-E `MoeMarlinResidentFor` build | Phase 1: run router; async D2H `dtid` + event synchronize; update hotness/dedupe logical IDs; load/evict until all selected experts are resident; rewrite a device copy of top-k IDs from logical IDs to slot IDs; run `MarlinMoeAlignBlockSize(..., num_experts=C)` and unchanged Marlin GEMMs over C-slot tensors. OFF path is byte- and trace-identical. Resident-first/missing-deferred overlap requires new pair partition/accumulation kernels and is out until phase-1 profiling justifies a separately spiked leaf |
| ds4.c:11696-11760 (full-layer streaming prefill + lookahead thread), 13812-13824 + 19657 (decode cache seed from last ≤64 prefill tokens) | same streaming branch, prefill mode | full-layer residency is impossible when C<E. Sweep the bank in chunks of at most C logical experts: filter routed pairs for the chunk, load contiguous slots, run aligned Marlin, scatter/accumulate each pair exactly once, then advance; pipeline next chunk reads only after correctness. Seed decode cache from the last-K prompt selections |
| ds4.c:750-1207 (expert locality profiler + hotlist write), ds4.c:13949-14016 (preload), ds4_streaming_hotlist.inc | `tools/expert_stream/expert_profiler` (+ optional hotlist preload in expert_cache) | measures per-layer histograms, latest-N cache simulation, adjacency; produces hotlist for preload and the h(f) curve for gates |
| (ours, no ds4 counterpart) | `tools/expert_stream/build_expert_bank.cpp`; source-metadata extension in `safetensors_reader.{h,cpp}`; streaming loader branch near `qwen3_5_weights.cpp:196-223,276-298,326-339` | one-time offline/first-load repack of all experts to a **Marlin-layout bank**. Builder consumes mmap spans while each owning shard lives (or explicit path/absolute-offset metadata), computes layer-wide w13/w2 combined scale factors, and writes fixed-size per-expert records plus manifest. Runtime streaming mode loads only manifest/non-expert weights: it must not populate `expert_{gate,up,down}_fp4` host bytes or call `BuildMoeMarlinResident`. Bank key covers checkpoint content, architecture, shapes, fused-vs-split w13 mode, Marlin layout ABI and target arch; derived bank is never redistributed |
| vLLM config style (`vllm/config/offload.py:15-44` as the shape reference) | `include/vllm/config/expert_stream.h` + plumb through `src/vllm/entrypoints/model_loader.cpp:117-129` memory accounting | our namespace `expert_streaming.*` (see upstream-sync safety); memory accounting learns "streamed expert bytes not resident" |

## Tests to port

Upstream vLLM has NO tests for inference-time disk expert paging (feature
absent in-pin — §Upstream chain sense 3); per test-porting.md the executable
spec here is (a) our own token-exactness/e2e gates in the established
harness patterns and (b) upstream tests for the ADJACENT mirror row recorded
for its own future spike, not this one.

| Test | Source / pattern | Local tier / target | Status plan |
|---|---|---|---|
| Streaming-vs-resident token exactness (same checkpoint, same seed, 16/16 greedy; f ∈ {0.25, 0.5, 1.0-ε}) | pattern: `tests/parity/test_qwen36_paged_engine.cpp:140` gate | T-parity, new `tests/parity/test_qwen36_expert_stream.cpp` (checkpoint-gated, dgx) | with W3 |
| Expert cache unit semantics: budget parse (count/`NGB`), auto plan subtraction, hotness decay halving, victim = min-hotness/LRU-tiebreak, in-flight + selected protection, logical→slot mapping, slot reuse, C<E bounds | ds4 semantics at ds4_ssd.c:46-106 and ds4_metal.m:8678-8776, 9653-9830 (cited in test header) | T-unit doctest, new `tests/vllm/test_expert_cache.cpp` (CPU-only, mock storage) | with W1 |
| pread pool: task fan-out, O_DIRECT alignment, failure propagation, shutdown | ds4_metal.m:7776-7995 | T-unit, `tests/vllm/test_expert_pread_pool.cpp` (tmp files, CPU-only CI) | with W2 |
| Expert bank builder: shard ownership/offset bounds, manifest-key rejection, layer-wide scale factors, and per-expert byte identity vs the existing full-resident Marlin repack | ours, current reference `qwen3_5.cpp:2392-2549` | T-unit + real-checkpoint golden compare, `tests/vllm/test_expert_bank.cpp` | with W2 |
| Slot-remap kernel: random logical top-k IDs + mappings become valid `[0,C)` IDs; `moe_align` + w13/w2 outputs equal an E-resident reference; unmapped ID hard-fails before GEMM | current Marlin dense-stride contract `marlin_template.h:543-550` | T-unit CPU reference + CUDA op parity | with W3 |
| Prefill chunk sweeps + decode seed: every routed pair appears exactly once across chunks; accumulated output equals full-resident path; cache state after prefill equals last-K selections | ds4.c:11696-11760, 13812 adapted to dense-slot Marlin | T-unit with mock router + checkpoint-gated e2e assert | with W4 |
| Regime guard: conc>bound warns/refuses; dense model rejects flag | ours (product rule from §3) | T-unit config + `tests/vllm/entrypoints/openai/test_conformance.cpp` addition | with W3 |
| `ENG-WEIGHT-OFFLOAD` mirror tests — `tests/basic_correctness/test_cpu_offload.py:11` (UVA/pin-memory matrix), `tests/quantization/test_cpu_offload.py:18,32,48,64` | upstream | NOT this row — inventoried for `specs/weight-offload-uva.md` | recorded only |

## Gates

All GPU work under the coordination GPU-lock rules; benchmark arms on an
idle box, ≥2-3 reps, exact commands into the ledger.

| Gate | Requirement | Exact command sketch |
|---|---|---|
| G1 token exactness (precondition, never traded) | Streaming on (f=0.5 and f=0.25) is 16/16 greedy token-exact vs the SAME build with streaming off, 35B gate corpus; and the streaming-off build stays 16/16 vs the vLLM oracle as today | `flock /tmp/gpu -c './tests/parity/test_qwen36_expert_stream --resident-frac 0.5'` (dgx) |
| G2 measured memory reduction | At f=0.5: peak whole-system used memory AND CUDA allocation high-water are ≥7.5 GiB below non-streamed. Prove the streaming loader retained no routed-expert host `OwnedTensor` bytes and allocated no full-E `MoeMarlinResident`. On GB10, `nvidia-smi` alone is insufficient: record `/proc`/`free`, process RSS/PSS, `cudaMemGetInfo`, and page-cache baseline; use O_DIRECT plus `simulate_used_memory` so file cache cannot fake the win | A/B same binary, fresh process per arm: `--expert-streaming off` vs `cache=8.4GB` |
| G3 tok/s floor at stated resident fraction | c1 decode (1024-token prefill, 128 decode, greedy) on dgx NVMe at f=0.5: **≥ 12 tok/s**, with the full measured curve published for f ∈ {0.25, 0.5, 0.75, 1.0-ε} incl. hit rates (I/O math §3 supports 17.7 uniform; 12 leaves compute+sync margin — a miss on 12 means the overlap machinery, not the concept, failed) | bench harness c1 arm x3 reps + cache hit-rate counters |
| G4 prefill regression bound | 8k-token prefill with streaming ≤1.5x non-streamed at f=0.5. Trace chunked slot sweeps: each routed pair exactly once, no full-layer resident allocation, and next-chunk overlap only if separately measured correct | same harness, prefill-heavy arm + nsys |
| G5 regime/off-path honesty | conc>4 warns/refuses; streaming mode disables CUDA graphs explicitly in phase 1; with feature off, 35B and 27B large-concurrency throughput/TTFT/TPOT/peak memory and kernel trace are unchanged within reproduced noise against fresh vLLM denominators | conformance + same-binary OFF-vs-parent A/B, both gate models, 3 reps |
| G6 record closure | Matrix row anchors + ledger row + README capacity-mode note + state.md, per DoD | `python3 scripts/check-agent-record.py` |

## Dependencies

| Dependency | Why | State |
|---|---|---|
| dgx.casa NVMe + 35B NVFP4 checkpoint | the only MoE gate model; bank build needs ~17 GiB free disk (346 GB free, headroom rule OK) | available (measured this spike) |
| Marlin fused-w13 MoE path (`VT_MOE_FUSED_W13`, `qwen3_5.cpp:2392-2652`) | bank records and fixed slots must match its fused layout, layer-wide scale factors, dense expert stride and global-scale indexing | merged, gated; exact contract now cited |
| Safetensors source metadata + bank-only loader representation | current `StTensor` lacks public owning path/fd/absolute offset, and `LoadNvfp4Raw` always copies bytes | W2 owns the narrow metadata/representation change; hard dependency before paging |
| Router D2H + logical-to-slot op | current `dtid` is device-only and Marlin consumes dense expert IDs | W3; phase 1 intentionally non-graphed |
| `SERVE-GATE-ONLINE` c1 baseline numbers | G3 needs the honest non-streamed c1 denominator | in flight (`CLAIM-SERVE-GATE-1`); W6 can measure its own baseline if still open |
| `QUANT-GGUF-KEEPQ-LOADER` (`specs/gguf-keep-quant-loader.md`) | **SATISFIED, not pending.** L2+L3 landed `429e19d6a` 2026-07-22 and shipped the per-expert row slicer this spec was waiting for; still not needed for safetensors/W1-W6 | L1/L2/L3/L5/L6 landed |
| No new third-party deps | plain pread/O_DIRECT + pthreads + CUDA runtime; io_uring explicitly NOT required (measured 5 GB/s with 4 threads) | - |
| Licenses | ds4 is MIT with llama.cpp/GGML attribution retained; we port design + cite, keeping our header discipline | OK |

## Work breakdown

Claim-sized, non-overlapping leaves. W0 is evidence before implementation;
W1 is CPU-only; W2 has CPU manifest tests plus one GPU golden; W3-W7 are the
critical runtime/gating path.

| Leaf | Scope (files above) | Depends on | Gate slice |
|---|---|---|---|
| W0 trace + c1 baseline | nsys current 35B router/align/Marlin/repack; router D2H boundary probe; fresh non-streamed c1 throughput/latency/memory; bank-layout byte accounting | - | committed evidence; no code state claim |
| W1 cache policy + budget | CPU `expert_cache` metadata and `budget`: logical→slot mapping, LFU/LRU/protection, no device allocation; unit tests | W0 | CPU tests green |
| W2 bank format/builder + reader + pread pool | safetensors source metadata, layer-wide scale-factor/full-resident byte golden, versioned bank manifest, bank-only weight representation, O_DIRECT pool and aligned staging. Does not change live dispatch | W0 | CPU tests + real-bank golden; build/startup/RSS checkpoint vs parent |
| W3 phase-1 decode dispatch | fixed contiguous C-slot CUDA arrays; router D2H/event wait; ensure-resident; logical→slot rewrite; align with C; unchanged Marlin GEMMs; config/memory accounting/regime guard; bypass full-E resident | W1,W2 | G1 + G2 + G5; nsys contract; own performance checkpoint |
| W4 prefill chunk sweeps + decode seed | routed-pair filter/scatter accumulation over ≤C experts per pass; last-K seed; only then measured next-chunk I/O overlap | W3 | G4 + exact pair coverage |
| W5 locality profiler + hotlist preload | h(f)/adjacency report for 35B corpus, optional preload | W3 (parallel with W4) | published curve; no gate rebasing |
| W6 optional resident/missing overlap spike | only if W3 trace shows wait dominance: inventory pair partition/accumulation kernel and graph implications before code | W3 evidence | separate accepted spike required |
| W7 per-regime campaign + closure | c1 curve, prefill, whole-system memory, OFF-path both-model vLLM A/B, ledger/README/matrix/state | W3-W5 | G3 + G6, all benchmark-protocol axes |
| W8 post-gate host-RAM tier | bank-file vs pinned-host backing store on discrete GPU; distinct from vLLM `cpu_offload_gb` mirror | W3 | separate hardware mini-gate |

## GGUF reconciliation (2026-08-14, issue #824)

**This spec was written 2026-07-10 and put GGUF expert streaming out of scope
because it "needs per-expert slicing of 3D tensors" and had to wait for
`QUANT-GGUF-KEEPQ-LOADER`. That slicer landed 2026-07-22, twelve days later,
under exactly that row, and is now the production decode path.** The record was
never reconciled, so until this note the spec directed a reader to wait for
something that already existed.

What exists, with anchors re-derived at HEAD:

- `OwnGgufQuantBlocks(tensor, n, k, row_offset, ...)`
  (`src/vllm/model_executor/models/qwen3_5_gguf_weights.cpp:57`) slices a stacked
  GGUF quant tensor by ROW OFFSET, deriving the byte offset from
  `vt::RowSizeBytes(dt, k)` and bounds-checking it against `tensor.nbytes`. Landed
  `429e19d6a` (L2+L3), refined `f6be46eda` (L6) and `d967733d2` (NVFP4).
- `GemmRowSlice(be, w, x, T, N, K, row_off)`
  (`src/vllm/model_executor/models/deepseek_v4.cpp:468`) runs a keep-quant GEMM
  against a row-slice of a stacked `[E*out, K]` weight, on DEVICE when the weight
  is block-quant.
- It is already called PER EXPERT at runtime: `deepseek_v4.cpp:1004,1005,2487,2489,2493`
  with `row_off = e * mi`, and `laguna.cpp:1225` with `id * moe_I`.
- The layout contract is stated on the role itself: `gguf_keep_quant.cpp:27`,
  `kStackedExpertWeight` is `[E, out, in]` and "each expert slice is whole rows of
  the same K".

**The consequence is a scheduling inversion, not just a tidy-up.** This spec
designs a safetensors -> Marlin BANK (versioned bank file, manifest, repack)
because that path needs one: safetensors experts are separate tensors in a
non-Marlin layout. On the GGUF path the mmap'd file ALREADY IS the bank — block
quantized, per-expert rows contiguous, offsets computable, and the GEMM already
takes a base pointer plus a row offset. No bank build, no manifest, no repack,
and no second on-disk copy of the expert bytes (the ~17 GiB cost this spec
accepts under Risks for the safetensors lane).

So the GGUF lane is plausibly the CHEAPER of the two, and it is the one that
matters commercially: every Qwen3.8-class checkpoint that exists today is GGUF
(`unsloth/Qwen3.8-2.4T-A95B-GGUF`), so nothing in that lane runs at all until
GGUF is served.

**What genuinely remains for GGUF is the residency policy, not the slicer.**
Today every caller passes `row_offset=0` and makes the WHOLE stacked
`[E*out, K]` tensor resident; streaming needs a bounded cache that materializes
only the touched expert slices and evicts under a byte budget. That is W1's
cache policy applied to a different backing store, which is what the original
"same cache, different slicer" note was reaching for — it was right about the
cache and wrong about needing a new slicer.

This note does NOT re-scope W1-W7, claim the GGUF leaf, or promise a date. It
corrects the record so the next person to cost that leaf starts from what is
true. Re-verify these anchors before implementing: they were re-derived at HEAD
on 2026-08-14 and this spec has already been wrong about them once.

## Target checkpoint census and the quant encodings it needs (2026-08-15, issue #912)

The GGUF reconciliation note above establishes that the GGUF lane needs no bank
and only a residency policy. It did not ask a prior question: **can this tree
decode the bytes of the checkpoint the row exists to run?** Measuring that
changed the target.

Every published GGUF of `unsloth/Qwen3.8-2.4T-A95B-GGUF` at revision
`567d3e6ac26c5474b18311e619c04350fb9a5556` was censused by parsing tensor
headers directly (HTTP range requests, no full download). Coverage is total, not
sampled: **1702 tensor records parsed against 1702 declared in
`split.tensors.count`.**

`UD-Q1_0` (370 GiB), the smallest quant and the one first chosen for this row,
is **not decodable by any admissible oracle** and was rejected:

- Its expert tensors carry ggml type **66**, holding ~97 % of all parameters.
- Type 66 exists in neither the pinned llama.cpp (`237ad9b96`, whose enum ends
  at `GGML_TYPE_Q1_0 = 41`) nor upstream `ggml-org/llama.cpp` master
  (`ad1de39e0`, 2026-08-15, ends at `GGML_TYPE_Q2_0 = 42`, `COUNT = 43`).
- It is defined only on the fork branch `unslothai/llama.cpp @ iq1-narrow` as
  `GGML_TYPE_IQ1_XXXS = 66, // 1.1875 bpw, 256-entry grid`. The bits-per-weight
  derived independently here from GGUF offset deltas is **1.1875**, matching that
  declaration exactly, which is what confirms the identification.

**That refusal was overruled by developer direction on 15 August 2026, and the
fork is now pinned.** The refusal rested on a policy premise, no upstream
definition and no pinned oracle, rather than on a technical one. The fork is
public and pinnable, so the developer directed anchoring the encoding to it, and
that resolves the objection at its root: the encoding now HAS a recorded
upstream and a fixed revision to cite.

`unslothai/llama.cpp` is therefore admitted to the oracle table as
`llama-cpp-unsloth`, pinned at `36fe8e1cc` on branch `iq1-narrow` and scoped to
the sub-IQ1_S encodings alone. It records `gateable = no`, because the port was
grounded in the fork's SOURCE rather than in a running comparison, and issue
#933 owes the build-and-run measurement. See
[`.agents/oracles/llama-cpp-unsloth.md`](../oracles/llama-cpp-unsloth.md).

Both checkpoints are therefore targets, and the order is UD-Q1_0 first per
developer direction, then UD-IQ1_S. The two are structurally IDENTICAL, censused
the same way over all shards, 1702 records against 1702 declared in each: same
276 expert tensors, same 96.92 %, same six other encodings with the same counts.
Only the expert encoding differs, ggml 66 against ggml 19. So the same streaming
lane serves both, and each needed exactly one new encoding.

The census is the scope statement for both targets, `UD-Q1_0` (370 GiB) and
`UD-IQ1_S` (473 GiB). Support status is read from the CODE rather than from
comments (`BlockDTypeFromGgmlTypeId` in `src/vt/dtype.cpp`, `BlockVecDot` in
`src/vt/cpu/cpu_quant_dot.cpp`, and the `DType` enum itself). That distinction
earned itself: a stale header comment listing six encodings led this section to
record Q2_K as missing, and reading the dispatch showed it has been served all
along.

Both censuses, side by side. The `UD-Q1_0` row set is identical apart from the
expert encoding:

| Encoding | ggml id | Tensors | Gparams | % params | Status |
|---|---|---|---|---|---|
| IQ1_XXXS (`UD-Q1_0` experts) | 66 | 276 | 2370.8 | **96.92** | added, fork-anchored |
| IQ1_S (`UD-IQ1_S` experts) | 19 | 276 | 2370.8 | **96.92** | added |
| Q5_K | 13 | 420 | 34.0 | 1.39 | served (`VecDotQ5_KQ8_K`) |
| Q2_K | 10 | 3 | 25.8 | 1.05 | served (`VecDotQ2_KQ8_K`) |
| Q6_K | 14 | 162 | 10.8 | 0.44 | served (`VecDotQ6_KQ8_K`) |
| Q4_K | 12 | 2 | 4.1 | 0.17 | served (`VecDotQ4_KQ8_K`) |
| Q8_0 | 8 | 1 | 0.1 | 0.01 | served |
| F32 | 0 | 838 | 0.5 | 0.02 | served (not keep-quant) |

**Each checkpoint was missing exactly ONE encoding, and it was 96.92 % of the
model.** In both, the expert tensors are precisely the 92 non-MTP layers times
three expert matrices (`ffn_down_exps`, `ffn_gate_exps`, `ffn_up_exps`), and the
Q2_K trio is block 92, the `nextn` MTP layer. So one encoding per checkpoint
makes a complete 92-layer forward pass loadable, and nothing else gates the
first benchmark.

IQ1_S is an ordinary upstream k-quant at the existing pin, and this tree already
carries the whole grid-table pattern it needs, built for DeepSeek-V4 under
`.agents/specs/gguf-iquant-dsv4.md`: `kIQ2_XXS`, `kIQ3_XXS`, `kIQ2_S` and
`kMXFP4` each have a keep-quant `vec_dot` against the Q8_K activation encoding,
CPU grids in `src/vt/cpu/cpu_quant_iq_tables.h` and device grids in
`src/vt/cuda/cuda_quant_iq_tables.cuh`. IQ1_S adds one more row on those rails.

Upstream anchors, re-derived at the pin `237ad9b96` (re-verify before
implementing, per the reconciliation note's own warning):

| Piece | Anchor |
|---|---|
| `block_iq1_s` (`ggml_half d; uint8_t qs[32]; uint16_t qh[8]`) | `ggml/src/ggml-common.h:414-419` |
| block size assert, 50 bytes per 256 elements = 1.5625 bpw | `ggml/src/ggml-common.h:420` |
| `ggml_vec_dot_iq1_s_q8_K_generic` | `ggml/src/ggml-cpu/quants.c:1099` |
| `iq1s_grid`, uint64, `NGRID_IQ1S` = 2048 | `ggml/src/ggml-common.h:1124` |
| `iq1s_grid_gpu`, uint32, 2048 | `ggml/src/ggml-common.h:1639` |

### The IQ1_XXXS decode was checked against REAL checkpoint bytes

Synthetic random blocks sweep bit patterns, but they cannot catch a
misunderstanding of the FILE: a wrong field order, a wrong stride, or a codebook
that turns real weights into legal-looking noise. So the decode was also run on
the downloaded `UD-Q1_0` shards, 15 August 2026.

Two independent things were compared on the SAME bytes: this tree's
`DequantIQ1_XXXS`, and a separate transcription of the fork's
`dequantize_row_iq1_xxxs` reading the fork's grid directly out of its own tree.

| Tensor | Layer | K | Weights | Result |
|---|---|---|---|---|
| `ffn_gate_exps` | 0 | 8192 | 524288 | bit-identical |
| `ffn_up_exps` | 0, 23 | 8192 | 262144 | bit-identical |
| `ffn_down_exps` | 0, 23 | 2048 | 65536 | bit-identical |

**1179648 real weights, max absolute difference 0.0.** The decoded values also
look like weights rather than noise: mean -4.2e-7, sd 8.6e-4, symmetric tails,
24 discrete levels, and no non-finite value. The file layout resolves exactly as
the port assumes, `row_bytes = K/256*38`, and the implied tensor size of
1275068416 bytes matches 512 experts times 2048x8192 at 1.1875 bpw.

Be precise about what this does and does not establish. It removes transcription
error from OUR C++, which is the failure this port was most exposed to, and it
proves the reader addresses the real file correctly. It does NOT make the fork
oracle gateable: both sides are transcriptions of the same source, so a defect in
the FORK would be reproduced identically by both. Only building and running the
fork closes that, which is what #933 owes.

### The decode is now sealed against the ORACLES, not against itself (#1023)

An independent review of #946 found that this seal covered the CODEBOOKS and
nothing else. `kIq1sGrid` and `kIq1xxxsGrid` each carry an FNV-1a digest and a
lane census, and corrupting a grid entry does fail. Every other decode parameter
was pinned only by self-consistency, because `ReferenceDotF64` and the G3
`MatmulBTQuant` NMSE reference both decode the weight with
`vt::cpu::BlockToFloat`, the function under test. They are independent in the
SUMMATION and nowhere else. Three defects were injected, each applied and
compiled, and the whole gate stayed green with an unchanged assertion count:

| Mutation | Before #1023 | After |
|---|---|---|
| `kIq1sDelta` `0.125F` to `0.25F`, which hits BOTH encodings | UNCAUGHT | 3 cases, 2049 assertions fail |
| IQ1_S delta sign inverted in dequant AND vec_dot | UNCAUGHT | 1 case, 1024 assertions fail |
| IQ1_S scale from `qh` bits 13-15 instead of 12-14, both paths | UNCAUGHT | 1 case, 960 assertions fail |

What closes it is a committed golden-vector fixture,
`tests/vt/iq1_golden_vectors.h`, whose EXPECTED values are produced by the
oracles themselves rather than by this tree:
`ggml_get_type_traits(type)->to_float`, called in a build of
`ggml-org/llama.cpp @ 237ad9b96` for IQ1_S and IQ3_XXS, and of
`unslothai/llama.cpp @ 36fe8e1cc7f2b3b8c92fdda0ab07600141921786` for IQ1_XXXS.
The IQ1 inputs are real `blk.0.ffn_gate_exps.weight` bytes, four 256-element
blocks from shard 2 of each split, so this commits a reproducible slice of the
1179648-weight run above. Agreement is BIT-EXACT on all 1024 values per
encoding, and `kIq1sDelta` is additionally sealed by value against upstream
`IQ1S_DELTA` (`ggml-common.h:1121`).

This is stronger than the 15 August run, and it is worth saying how. That run
compared our C++ against a hand transcription of the fork, so BOTH sides were
transcriptions and a defect in the FORK would have been reproduced identically
by each. These vectors are decoded by the fork's own compiled code. That still
does not make the fork gateable in the sense #933 owes, which is running the
MODEL, but the fork's decoder is no longer transcribed at all.

The same review found the NMSE ceiling widened past the point where it
discriminates. It was `2e-3`, described as about 4x the residual. Re-measured 16
August 2026 over all 12 shapes per type, with the ceiling forced to `1e-12` so
doctest prints every captured value:

| Type | Unmutated max | With `kIq1sDelta = 0.25` |
|---|---|---|
| `iq1_s` | 5.240e-4 (m=4, n=1) | 6.967e-4 |
| `iq1_xxxs` | 3.109e-4 (m=1, n=1) | 1.420e-4 |

So `2e-3` passed that defect and `6e-4`, the value now set, fails it. The
`iq1_xxxs` column is the more useful half: the defect moves that statistic the
WRONG WAY, so NO ceiling catches it there. An NMSE against a dequant-f32
reference cannot seal a decode parameter at all when both sides decode through
the same function. The ceiling bounds quantization error, which is its own job;
the goldens carry the decode.

Two further findings, repaired in the same flow. `DequantGgufRowToF32` listed no
`case 19` and no `case 66`, so the expansion path threw `unsupported ggml type`
for the two encodings the target checkpoints are 96.92 % made of, and
`RouteGgufTensor` routes a tensor there whenever `VT_CPU_REF` is set, keep-quant
is off, K is ragged, or the role is not verbatim: a refusal to load on the
reference lane. `case 18` (IQ3_XXS, the DeepSeek-V4 UD-IQ2_XXS `ffn_down`
encoding) carried the same omission and is fixed with it, gated on its own
oracle-produced golden. And both checkpoint-census cases claimed TOTAL coverage
of 1702 tensor records while enumerating six of the seven encodings and summing
to 864; they now carry F32's 838 tensors and assert that the buckets sum.

Why this is in this spec rather than its own row: the encoding is the load
path's half of the same capability. A streamer that can address an expert slice
it cannot decode moves bytes for nothing, so the row's own gate cannot be met
without it. It is nonetheless useful beyond this row, since any low-bit GGUF
gains it, which is the standing k-quant obligation in `AGENTS.md` rather than a
detour for one model.

This note does not re-scope W1-W7 or claim a date. It adds the leaves below and
records why the target checkpoint changed.

| Leaf | Scope | Depends on | Gate slice |
|---|---|---|---|
| W9 IQ1_S decode | `DType::kIQ1_S`, geometry `{256, 50, 19}`, `BlockDTypeFromGgmlTypeId` row, CPU `iq1s_grid` + `VecDotIQ1_SQ8_K` traits row, dequant path | - | CPU tests green against upstream-derived vectors; `KeepQuantDType(19)` true |
| W10 IQ1_S device decode | `iq1s_grid_gpu` + CUDA `vec_dot`, so streamed slices dot on device. NOT optional polish: `cuda_quant_dot.cu:1531` maps an unknown weight dtype to `return false`, which is a SILENT CPU fallback. Tokens would still be correct, so no token gate can see it, and a 2.4 T model would simply run at CPU speed while looking healthy | W9 | CUDA tests; parity with the CPU arm; the fallback must be observable rather than inferred |
| W11 checkpoint load | UD-Q1_0 first, then UD-IQ1_S, loads end to end on dgx.casa, refusing any encoding it cannot honour by name. BLOCKED on W4: measured 16 August 2026, the load needs memory of the order of the whole file (4 GiB per 110 s, linear, no plateau) and cannot complete in 119 GiB | W9, W10, W12, W4 | model loads; token output captured |
| W4a allocation attribution | **DONE 16 August 2026.** The allocator is the GDN V-head reorder forcing `attn_qkv` and `ssm_out` to `kTransformedWeight`, which expands them to bf16: about 50 GiB across 93 layers. Experts were never the problem; they borrow with 0 copies | - | measured, see the census section |
| W4b transformed-weight residency | Stop the V-head reorder from costing ~50 GiB. Either apply the reorder at RUNTIME so the stored blocks stay borrowed, or confine it inside block boundaries so a block-typed weight survives it. Without this, streaming the experts still does not fit | W4a | the two projections stay block-resident, measured on the same load |
| W12 IQ1_XXXS decode | `DType::kIQ1_XXXS`, geometry `{256, 38, 66}`, the 256-entry fork codebook, `VecDotIQ1_XXXSQ8_K`, dequant, traits row and the reader's `case 66`. Grounded in the pinned fork oracle, cited per site | - | CPU tests green; `KeepQuantDType(66)` true; grid digest sealed |
| W13 IQ1_XXXS device decode | the CUDA arm, for the same silent-fallback reason as W10 | W12 | CUDA tests; parity with the CPU arm |

## Loading a 370 GiB split GGUF: the recipe, and what it cost to find

Recorded 16 August 2026 from two dry runs against the partially downloaded
`UD-Q1_0`, because each failure was cheap to hit and expensive to guess.

**The model path must be a `.gguf` FILE, not a directory.** `--model <dir>`
sends the server down the HuggingFace branch, which fatals on a missing
`config.json` before it ever looks for GGUFs (`model_loader.cpp:1287` takes the
GGUF branch only for `fs::is_regular_file(dir) && extension == ".gguf"`). For a
split model, point at shard 1; the reader finds its siblings by the
`-NNNNN-of-MMMMM.gguf` naming and cross-checks `split.count`.

**The container needs `--gpus all` even for `--help`.** `libcuda.so.1` is
injected by the NVIDIA Container Toolkit, so a CUDA-linked binary cannot load
without it. This is not about using the GPU; a build or a usage message fails
the same way.

**Paths on dgx.casa after the 14 August reimage:** the NAS is mounted at
`/usr/local/nas_share`, not `/mnt/nas_share`. There is no host CUDA toolkit, so
builds run inside `vllmcpp-build:gb10`, and `docker` needs `sudo`.

Working invocation:

```sh
docker run --gpus all -v /usr/local/nas_share:/nas:ro -e VT_GGUF_PREFAULT=0 \
  vllmcpp-build:gb10 ./build-srv/examples/vllm-server \
  --model /nas/checkpoints/<model>/<model>-00001-of-000NN.gguf \
  --device cpu --max-num-seqs 1 --max-model-len 512
```

`VT_GGUF_PREFAULT=0` is the load-bearing flag for a model larger than memory.
Keep-quant weights are BORROWED from the mmap rather than copied
(`p.mmap_residency = EnvOnOr("VT_GGUF_MMAP", p.keep_quant)`,
`gguf_keep_quant.cpp:208`, then the `OwnedBytes::Borrow` branch), so the tower
costs address space rather than anonymous memory. The load-time prefault would
undo that by touching every page, which is right for a model that FITS and wrong
for one that does not.

`--max-num-seqs 1` follows this spec's own regime bound. At high concurrency
every step touches most experts, so the working set stops being a working set.

**What the second dry run proved, before the checkpoint was complete.** Pointed
at shard 1 with 9 of 10 shards present, the loader merged the split, parsed the
metadata, resolved `qwen35moe` with no architecture refusal, did NOT refuse
IQ1_XXXS, and walked tensors by name to `blk.87.ffn_up_exps.weight` before
stopping on a span that exceeded the incomplete shard 9. Everything above the
byte read therefore works: architecture resolution, split merging, tensor naming
and the new encoding. What remains untested is what happens AFTER the last
tensor is read.

**One operational note that is not about this tree.** A long-lived HuggingFace
connection can decay to under 1 MiB/s while a fresh connection to the same file
runs at 20 MiB/s and the NAS writes at 97 MiB/s. Both endpoints measure healthy,
so a stalled fetch is indistinguishable from a slow one without a rate check.
Killing the curl lets the fetch script's `-C -` resume open a fresh connection.

## The 370 GiB load was MEASURED, and the page cache does not do this for free

16 August 2026, dgx.casa, 119 GiB of unified memory, `UD-Q1_0` complete and
sha256-verified, run from LOCAL NVMe so the network filesystem is not a variable.

**This section exists to record a refuted hypothesis, because it was mine and it
was load-bearing.** Reading the loader, the keep-quant path BORROWS weights from
the mmap (`p.mmap_residency = EnvOnOr("VT_GGUF_MMAP", p.keep_quant)`,
`gguf_keep_quant.cpp:208`, then the `OwnedBytes::Borrow` branch). From that it
seemed to follow that a model larger than memory would simply be demand-paged by
the kernel, and that the streaming lane was an optimisation rather than a
requirement. That is not what happens.

### What was measured

Routing is CORRECT. A probe that asks the policy directly, rather than inferring
it, reports `keep_quant = 1`, `mmap_residency = 1`, `quant_repack = 0`, and:

| Tensor | ggml | Routed to |
|---|---|---|
| `blk.0.ffn_gate_exps.weight` | 66 | `keep_quant` |
| `blk.0.ffn_down_exps.weight` | 66 | `keep_quant` |
| `blk.0.attn_qkv.weight` | 13 | `keep_quant` |
| `blk.0.ssm_out.weight` | 14 | `keep_quant` |
| `output.weight` | 12 | `keep_quant` |
| `token_embd.weight` | 12 | `expand_bf16` (2.03 G params, about 4 GiB) |

And yet the process accumulates ANONYMOUS memory linearly with load progress:

| t | anon | shared_clean | avail |
|---|---|---|---|
| 300 s | 21 GiB | 0 | 93 GiB |
| 600 s | 32 GiB | 0 | 82 GiB |
| 1360 s | 37 GiB | 0 | 77 GiB |
| 1800 s | 53 GiB | 0 | 60 GiB |

**4 GiB per 110 s, dead linear for 30 minutes, no inflection.** Extrapolated,
the load needs memory of the order of the whole 370 GiB file and cannot complete
in 119 GiB. The run was stopped at 53 GiB rather than allowed to reach the OOM
guard, because on GB10 an out-of-memory kill takes the MACHINE down and not just
the process.

`shared_clean` stayed at 0 for the entire run. No file pages were ever resident
in the process, which is what a borrow with `VT_GGUF_PREFAULT=0` should look
like, and is also why RSS alone cannot answer this question: RSS counts resident
file pages, so it conflates the two hypotheses. Anonymous versus file-backed is
the measurement that separates them.

### What this establishes, and what it does not

Established: **the model does not load in 119 GiB today, and expert streaming is
REQUIRED rather than an optimisation.** The row's original premise stands
unchanged; the "the kernel already does this" shortcut does not exist.

**Narrowed 16 August 2026, same run.** The anonymous memory is ONE merged VMA,
not many small ones: at t+300 s the process held a single 12.9 GiB anonymous
region, fully resident, alongside a 209 MiB heap and nothing else above 64 MiB.
Linux merges adjacent anonymous VMAs with identical flags, so that signature is
a run of large sequential allocations coalescing, which is what
`AllocAligned64` (`src/vt/cpu/cpu_backend.cpp:15`) produces one weight at a
time. It is NOT an arena (the CPU backend has none), and it is NOT the KV cache
(`num_blocks` auto-resolves to 256, about 4 GiB).

So the weights are being copied into CPU device memory even though
`qwen3_5.cpp:976` aliases them when `GetPlatform(d.q.device.type).is_cpu()`, and
`--device cpu` demonstrably reaches the loader (`server_main.cpp:1034` sets
`engine_params.device` before the `FromModelDir` at :1103). Those two facts still
disagree, and THAT is the remaining question: instrument the branch and find out
whether `is_cpu()` is false at that point, or whether a second copy happens
elsewhere.

**ANSWERED 16 August 2026 by instrumenting the load. The allocator is the
GatedDeltaNet V-head reorder, and it has nothing to do with the experts.**

Counters placed at each candidate, then run on both hosts, eliminated every
other explanation and found the real one:

| Candidate | Verdict |
|---|---|
| keep-quant copy branch (`OwnGgufQuantBlocks`) | **0 copies**, 281 borrows, 147.6 GiB borrowed |
| q8_0 repack branch | only accepts `kQ8_0`; this checkpoint has ONE such tensor |
| model materialisation (`qwen3_5.cpp:976`) | instrument NEVER fired during load |
| vt CPU and CUDA allocators | never reached the 2 GiB print threshold |
| `OwnedBytes` copy | shallow; a borrow shares its owner |
| **expand-to-bf16 (`DqBf16`)** | **69 calls by layer 9**, on `attn_qkv` and `ssm_out` |

`in_proj` (`attn_qkv`) and `out_proj` (`ssm_out`) become `kTransformedWeight`
whenever the V-head reorder is active (`num_v != num_k`, `qwen3_5_gguf_weights
.cpp:1020`). A transformed weight is LAYOUT-REWRITTEN at load, so it "can never
keep its blocks" and expands to bf16. For this model that is 8192x16384 plus
16384x8192 per layer, about 268 MiB each after expansion, so roughly 536 MiB per
layer across 93 layers, or **about 50 GiB** — which matches the measured 53 GiB
almost exactly.

The two hosts disagreed for a reason worth keeping: on x86 without the reorder
path taken the same load holds **anon = 0 GiB across 281 borrows**, and on GB10
it climbs. The reorder is what separates them, not CUDA, not the filesystem, and
not the expert encoding.

**The consequence changes the plan.** Expert streaming alone does NOT make this
model load. The experts already borrow correctly and cost no anonymous memory;
the blocker is ~50 GiB of DENSE weights inflating from about 5.5 bits to 16.
Streaming the experts and leaving this in place still exceeds the box. Whoever
takes this needs BOTH: the streaming lane for the ~330 GiB of experts, and a
transformed-weight path that does not expand, by doing the reorder at runtime or
by confining it inside block boundaries.

Instrument note, because it nearly cost the diagnosis: the first loader counter
printed on `(n % 300) == 1`, so it fired once and read as "1 borrow" when there
were 281. A counter that samples cannot answer "how many". The second flaw was
placing that counter AFTER the repack early-return, which would have hidden
every repack copy had there been any.

NOT established previously, now closed. The remaining unknown is not the
allocation site. The evidence bounds the OUTCOME
(linear anonymous growth proportional to load progress) without identifying
which call allocates. Routing says borrow, the process says anonymous, and those
two facts are not yet reconciled. The likely candidates, in order, are that the
borrowed `OwnedTensor` is copied again when the model materialises its own
weight buffers, or that a per-expert path re-materialises what the stacked path
would have borrowed. Whoever takes W4 should settle that FIRST, because if it is
the former then wiring `ExpertStreamer` under a loader that copies afterwards
changes nothing.

### What the load DID prove

Everything above the byte read works on the real checkpoint. The loader merged
all 10 split shards, parsed the metadata, resolved `qwen35moe` with no
architecture refusal, did NOT refuse IQ1_XXXS, and walked tensors by name deep
into the stack (`blk.87.ffn_up_exps.weight` in the incomplete-checkpoint dry
run). Architecture resolution, split merging, tensor naming and both new
encodings are not the blocker. Capacity is.

## First run: Qwen3.8-2.4T-A95B loads and generates on one DGX Spark

16 August 2026, dgx.casa, GB10, 119 GiB unified memory, `UD-Q1_0` (370 GiB, 10
shards, sha256-verified) on local NVMe, CPU device, `--max-num-seqs 1
--max-model-len 512`, `VT_GGUF_PREFAULT=0`, server entry point.

**It loads, it serves, and it is correct.**

```
prompt: "Q: What is the capital of France? A:"
output: " Paris. Q: What"
```

| Axis | Measured |
|---|---|
| load to serving | 13 min |
| resident anonymous memory | **62 GiB** of 119 GiB |
| TTFT (token 1) | **3318 s** |
| steady decode | **66.7 s/token** (66.5, 66.9, 66.8) |
| decode rate | **0.015 tok/s** |
| expert read bandwidth during decode | ~100 MB/s (measured 50 MiB/s at one sample, 914 % CPU) |

### Why a 370 GiB model fits in 119 GiB

Because the experts are never copied. They are BORROWED from the mmap and
demand-paged by the kernel, costing zero anonymous memory. What is resident is
the dense remainder, and it matches the prediction from the checkpoint's own
tensor table:

| Component | Predicted |
|---|---|
| `attn_qkv` expanded to bf16 | 21.56 GiB |
| `ssm_out` expanded to bf16 | 17.25 GiB |
| `token_embd` + F32 norms | 5.81 GiB |
| predicted total | 44.6 GiB |
| measured, with KV and runtime | **62 GiB** |

This closes the question the earlier measurement left open, and corrects the
conclusion drawn from it. The load was previously stopped at 53 GiB on the
extrapolation that it was heading for 370 GiB. It was not; it plateaus. The
extrapolation was wrong, the data was not, and the difference was letting it
finish.

### What the speed says, precisely

0.015 tok/s is not a streaming result. It is the result of having NO streaming
lane: every token needs roughly 6.7 GB of expert bytes (10 experts x 3 matrices
x 93 layers), and the kernel serves them as 4 KiB demand faults in router order.
That yields about 100 MB/s against an NVMe that sustains ~5 GB/s, so **the gap
is about 50x and it is entirely access pattern.**

The steady-state number is the useful one. 66.5, 66.9 and 66.8 seconds for three
consecutive tokens is not noise; it is the same working set being re-faulted
every step, which is exactly the behaviour a resident expert cache removes.

TTFT of 3318 s is prefill plus the cold set and should not be quoted as a decode
number.

### What this changes for the row

Streaming is NOT what makes this model fit. Borrowing already does. Streaming is
what makes it FAST, and the size of that prize is now measured rather than
assumed: 50x of pure I/O access pattern, before any cache-hit benefit. The c1
target of 3 to 6 tok/s needs roughly a 200 to 400x improvement, so it needs both
the explicit batched reads AND the hit rate the hotness cache is designed for.

Everything the lane needs is already in tree and unwired: `ExpertSlotCache`
(W1), `GgufExpertSpanOf` (W2) and `ExpertStreamer` (W3). W4 now has a measured
baseline to beat and a working end-to-end vehicle to beat it on.

## W4 measured: correct, 4.5x on TTFT, and NO steady-state gain (and why)

16 August 2026, dgx.casa, same vehicle and config as the first run, streaming
enabled with 8000 slots (`[expert-stream] ON slots=8000 slot_bytes=2490368
resident=18.55 GiB`).

| Axis | Baseline (no streaming) | Streaming ON |
|---|---|---|
| output | `" Paris. Q: What"` | `" Paris. Q:"` (**identical tokens**) |
| TTFT | 3318.1 s | **733.4 s** (4.5x) |
| steady decode | 66.5 / 66.9 / 66.8 s | 68.7 / 67.7 s (**unchanged**) |

**Correctness holds**: the slot path is byte-faithful, which is the precondition
for caring about the rest.

**The steady-state result is a negative one, and the cause is this
implementation rather than the design.** `EnsureSpan` copies from `base +
offset`, a pointer INTO the mmap, so filling a slot still takes the page fault
it was meant to avoid. What was actually changed is fault ORDER (random to
sequential) plus an extra 2.49 MiB memcpy, and that combination measures the
same ~103 MB/s the mmap path already did.

The spec said this on the first page and the implementation did not follow it:
"Reads are plain `pread(2)` against the model fd", with an O_DIRECT variant and
an aligned staging buffer. ds4 preads from the FILE DESCRIPTOR. Copying from the
mapping inherits the exact fault path the lane exists to bypass.

**A second bound is independent of that fix.** Each token needs 2790 slices
(10 experts x 3 matrices x 93 layers) at 2.49 MiB, so **6.9 GB per token**,
against a 19.9 GiB cache: under three tokens of working set, with top-10-of-512
routing making consecutive tokens rarely reuse an expert. Even perfect
sequential I/O leaves the hit rate to be won separately, by a larger budget, the
hotlist preload (W5) or lookahead prefetch.

So the ordering for the next attempt is now measured rather than guessed:

1. **pread from the fd**, not memcpy from the mapping. This is the one that
   makes the I/O rate move at all, and until it lands no cache size matters.
2. **Overlap** the read with compute, which is what the async pool in the
   original design is for; a synchronous fill serialises I/O behind every layer.
3. **Hit rate** last, because it multiplies a bandwidth that is currently wrong.

TTFT improving 4.5x while decode did not is consistent with all of this: prefill
touches a wide expert set once, where sequential order and slot reuse both help,
while decode re-reads a fresh 6.9 GB every step.

## One predicate, three switches, and only one of them grew (issue #1029)

`#967` taught `IsCudaKeepQuantSupported` (`src/vt/cuda/cuda_quant_dot.cu`) to
return true for `kIQ1_S` and `kIQ1_XXXS`. Three dispatch switches consume that
predicate and `#967` extended one of them, the dense `MatmulBTQuantKernelCuda`.

The grouped GEMM used the same predicate to SKIP its CPU fallback and then
dispatched through a `switch (w)` that had no case for either dtype and no
`default:`. It quantized the activation, launched nothing, and returned;
`CheckCuda(cudaGetLastError())` reported success, because a launch that never
happened cannot fail; and the output tensor kept whatever it already held. An
independent review measured this on GB10 through a poisoned output buffer: both
IQ1 encodings left `-12345` in place, at NMSE `4.58e6` and `9.96e6` against the
CPU oracle, while the `iq2_s` control passed. Before `#967` these dtypes took
the CPU fallback and emitted correct tokens slowly, so `#967` converted
correct-but-slow into silently wrong, which inverts its stated purpose.

That path is the DEFAULT routed-expert path of the target checkpoint:
`qwen3_5_gguf_weights.cpp` accepts any `KeepQuantDType`, which now includes ggml
19 and 66, and `qwen3_5.cpp` `KqGrouped` reaches `vt::MatmulBTQuantGrouped` with
`VT_QWEN35_GROUPED_MOE` on by default. The two encodings are 96.92 % of the
model. The fused `MoeGateUpSwiGLUGroupedCuda` seam had the same hole, where the
consequence was worse in kind: its guard previously threw a NAMED refusal, and
`#967` turned that named refusal into silence.

The repair is three things, and the third is the one that matters most.

1. Both grouped switches gained the `kIQ1_S` and `kIQ1_XXXS` arms. No new kernel
   code was needed: `QuantDotGemmGroupedKernel` and
   `QuantDotGemmGroupedFusedSwiGLUKernel` are generic in `W` and depend only on
   `DotSuperblock<W>` and `FinalFactor<W>`, both already specialized for these
   two dtypes by the dense path.
2. All three switches gained a `default:` that THROWS and names the dtype. This
   is the general repair rather than the specific one: past the
   `IsCudaKeepQuantSupported` gate there is no CPU fallback left, so any future
   missing case is a silent no-op unless something refuses out loud.
3. The grouped seams are now gated. They had NO test: `grep -rl
   MatmulBTQuantGrouped tests/` found two files and neither mentioned `kCUDA`,
   which is exactly why F1 and F2 landed green. `tests/vt/test_cuda_quant_dot.cpp`
   now drives both grouped seams over the same case table the dense gate uses,
   including the two new dtypes, against the CPU grouped golden, THROUGH A
   POISONED OUTPUT BUFFER.

The poison is not decoration and it is not redundant with the value comparison.
Mutating `MatmulBTQuantKernel` to write nothing makes the golden AND the
independent reconstruction both stay at the poison value, so the byte comparison
passes with zero failures and only the poison assertion fires: 55 failed
assertions, all of them `poisoned == 0`, and `memcmp` failures zero. Every value
gate in that file would have read an unwritten buffer as a merely inaccurate
result.

Chosen against `-Wswitch` for CUDA (`cmake/CompilerWarnings.cmake`), which the
issue proposed. CUDA gets `-Werror=all-warnings` and no `-Wall`, so `-Wswitch`
never runs, but adding it would not have caught this defect either: `-Wswitch`
is silent whenever a `default:` label exists, and a `default:` is exactly what
the repair adds. The flag that would fire is `-Wswitch-enum`, which warns on
every enum switch in the tree that omits any enumerator even with a default, and
that is a tree-wide change with no measurement behind it here, on a lane this
box cannot compile. Recorded rather than done.

**Owed on this repair.** The CUDA arms of the new gate have NOT run on a device.
`dgx.casa` was unreachable for the whole of this work (`No route to host`, ping
100 % loss), this box has no CUDA toolkit and no NVIDIA device, and there is no
second CUDA host. What ran is the CPU arm, which is real: it drives the same
poisoned-buffer instrument through the CPU grouped golden on every host, and the
mutations above are its evidence. What is still owed is the GB10 run of the three
CUDA cases, including the confirmation that removing either new `case` turns the
grouped gate red. Issue #1029 stays open until that runs.

Also sealed here: `cuda_quant_iq_tables.cuh` claimed from the day it landed that
"a runtime test memcmps these tables against the CPU host tables". No such test
existed. The CPU tests digest the HOST symbols, and nothing read
`vt::cuda::d_iq1s_grid` at all, so a device transcription slip was visible only
when a weight sample happened to address the drifted entry. Replaying the CUDA
gate's own `std::mt19937(0x5EED)` stream, 266 of the 2048 `d_iq1s_grid` entries
(13.0 %) are never addressed, which is why drifting entry 0 is caught and
drifting entry 3 stays green at 150032/150032 assertions. That figure came from
the review and was re-derived here rather than quoted: replaying the stream over
the dense gate's widest weight (16 rows times 8 super-blocks, 128 blocks, 4096
grid draws) gives 1782 distinct entries and 266 never addressed.

The new grouped gate widens the same weight to 64 rows (E=4, N=16), which is 512
blocks and 16384 draws, and that reaches all 2048 entries. Recorded because it is
true, not because it closes anything. It is coverage by accident of shape rather
than by contract, one shape change away from shrinking again, and it says nothing
at all about the other seven tables. The seal now exists
(`src/vt/cuda/cuda_iq_table_seal.h` plus the gate case), it covers all eight
device codebooks byte for byte, and both false comments were corrected.

## Readahead attempt: landed, UNMEASURED, blocked on host contention

16 August 2026. W4's decode result showed that filling a slot by memcpy from the
mapping keeps the fault path: sequential order, but still 608 four-KiB traps per
2.49 MiB slice. The spec's answer is `pread(2)` from the fd, which needs an fd
accessor on `ReadOnlyFileMapping` and a path from the model's `OwnedTensor` back
to its file offset, because `OwnedBytes::owner` is a type-erased
`shared_ptr<const void>`. That is a real plumbing change.

`madvise(MADV_WILLNEED)` attacks the same root cause in three lines: it hands
the whole slice to the kernel's readahead in ONE call instead of trapping page
by page, which is the same lever `PrefaultBorrowedSpan` already uses at load,
applied per slice at decode. It is advisory and read-only, so it cannot change a
byte.

**It is not measured, and this note exists so nobody assumes otherwise.** Three
consecutive attempts on dgx.casa were killed by the kernel OOM at 48.6 GiB anon,
because another session's `ltx2-gen` holds 32.6 GiB and this model needs about
62 GiB on a 119 GiB box. Further attempts were stopped rather than retried: the
OOM killer picks the largest consumer, and repeatedly loading a 62 GiB model
next to someone else's 33 GiB job risks taking THEIR work down.

So the readahead lever is landed behind the default-off streaming flag with its
reasoning stated, and its number is owed. Re-run when dgx is quiet. If it does
NOT move decode, that is the evidence that the fault path itself must be
bypassed and the `pread` plumbing is necessary rather than merely preferable.

One supporting piece IS tested. `ExpertSlotCache::Contains` is a pure residency
probe so the prefetch can ask "will this be a fill?" without scoring the entry.
If asking scored, the probe would decide the eviction order it was only meant to
observe, and the hotness policy would be measuring itself. The test probes one
entry fifty times and asserts the OTHER one is still the survivor.

An operational note that cost a diagnosis: the first attempt ran with
`docker --rm`, so when it died the logs died with it and the failure was
unreadable. A container that erases its own evidence on failure is not a usable
instrument for a run that might fail.

## The wiring review (#912 F1-F11): what was measured on a dead cache

16 August 2026. An independent review of the W4 wiring returned FAIL on eleven
findings. Four of them were the same defect at different depths, and the first
one invalidates the decode number recorded two sections above.

**F1. `Qwen35ExpertStream::EndStep()` had no caller.** `grep -rn EndStep src
include` found definitions only, and deleting the definition still compiled.
`ExpertSlotCache::Acquire` marks every entry it serves `protected_this_step`,
and ONLY `EndStep` clears that mark, so protection was permanent: once the cache
filled, `ColdestEvictable` found nothing evictable and returned -1, `Acquire`
returned slot -1, `Slice` returned nullptr, and `KqExpertSlice` fell back to the
mmap path. The step clock never advanced, so the hotness decay, the LFU score,
the LRU tiebreak and eviction never ran in production at all. A reviewer probe
against unmodified sources: 8 slots, 40 distinct slices asked, **8 served and 32
REFUSED**, 0 evictions, 0 steps; with `EndStep()` called, 40 served, 0 refused,
32 evictions, 10 steps.

**THE ARITHMETIC MATCHES THE PUBLISHED RUN EXACTLY, and that is the point.** The
run above used 8000 slots against 2790 slices per token, so the cache had
8000/2790 = **2.87 tokens** of capacity before every slot was permanently
protected. Prefill is ONE forward and therefore one step, and its working set fit
inside that budget, which is why TTFT really did improve 4.5x. Decode is one
forward per token, and from partway through token 3 onward every slice was
refused and served from the mapping instead. **The "steady decode: unchanged"
row was measured on a lane that had switched itself off**, and it is VOID.

The cause the section above assigned to that result -- that `EnsureSpan` memcpy's
from the mapping and so inherits the fault path -- is a true statement about the
code and was **not established by that measurement**. It remains a plausible
bound and it is now unmeasured rather than measured. Both it and the readahead
lever are owed a re-run on a live cache.

Nothing in the run could have shown this, and that is the second finding worth
recording. The process printed `[expert-stream] ON ...` once at startup and
nothing afterwards, so a cache that died in token 3 looked exactly like one that
worked for 200 tokens. There is now one stderr line carrying `steps`, `hits`,
`misses`, `evictions`, `fills`, `bytes`, `exhausted` and `advised`; `steps == 0`
and `exhausted > 0` are precisely the F1 signature and both are wrong at a
glance.

**F2. A failed fill left the key resident over a half-written slot.**
`EnsureFile` has to acquire before it preads, because the read needs a
destination. A throw then unwound past a cache entry that claimed residency over
`done` correct bytes followed by the tail of whatever the slot held before, and
nothing downstream reads an exception as data: the next acquisition of that key
was an ordinary HIT, a hit moves no bytes by contract, and the GEMM multiplied
half of one expert spliced onto half of another. The commit that added the pread
claimed the opposite ("a read that hits EOF THROWS rather than leaving a
partially filled slot"). `ExpertSlotCache::Invalidate` now undoes the
acquisition and returns the slot to the budget.

**F3. Nothing reached any of it from a gate.** `Qwen35ExpertStream`,
`KqExpertSlice`, `VT_MOE_EXPERT_STREAM`, `SourceOfSpan` and `fd()` appeared
nowhere under `tests/`; replacing the production call site with `nullptr` and
forcing streaming unconditionally ON both left the full gate green. Every test
this row had constructed the cache, the store and the streamer by hand, which
proves the classes work and never proves anything reaches them.

**And the gate that closed F3 immediately found something nobody was looking
for, which is the strongest argument in this section for writing it.** Its
byte-identity case failed on all 160 logits while each arm was internally
deterministic, so the streamed and unstreamed arms genuinely disagreed.
`Qwen35ExpertStream` is a process-lifetime singleton and keyed its cache on the
tower's host buffer ADDRESS. Its comment stated the premise and drew the wrong
conclusion: a tower's base pointer IS stable for the model's life, but the CACHE
is not scoped to one model's life. Free a model, load another, and the allocator
hands the new towers addresses the old ones held, so the new model's expert
resolved to an entry filled from a different checkpoint -- as a HIT, silently.
Instrumenting `KqExpertSlice` to memcmp each slot against the slice it claims to
be: **24 towers occupied 21 distinct addresses, and 20 of 222 slices returned
another tower's bytes**. Filed as [#1066](https://github.com/mudler/vllm.cpp/issues/1066)
and fixed by `OwnedTensor::TowerUid()`, a process-unique counter that cannot
collide because it never goes backwards.

**F4. `GgufFile::SourceOfSpan` had zero coverage** and it is the one place in
the chain that decides WHICH FILE a weight is read from. Deleting the sibling
walk (every span resolves to shard 0's descriptor) and forcing `out.offset = 0`
both survived the full gate. Both are the "wrong shard at a plausible offset"
failure: the pread succeeds, returns exactly the bytes asked for, and the model
multiplies another tensor.

**F5. The `MADV_WILLNEED` hint was inert.** madvise(2) returns EINVAL on an
address that is not page-aligned, GGUF tensor data is aligned to
`general.alignment` (default 32, `gguf_reader.cpp:401`), and the return value was
discarded. Confirmed by mutation: reverting to the unaligned address drops the
accepted-call count to zero. It is now aligned down, extended, and counted. **No
speedup is claimed** -- this makes the call well formed, nothing more.

**F7.** The singleton used `static T* inst = nullptr; if (inst == nullptr) inst
= new T(...)`, which is not the magic-static idiom fourteen lines above and
double-constructs an ~18 GiB store under two concurrent first calls. Separately,
gate/up and down slices differ in size whenever a UD quant keeps `down_proj` at
higher precision, and the store was sized from whichever slice arrived first, so
a UD checkpoint refused its own first down slice mid-decode. `ExpertMlpKq` now
declares max(gate, up, down) before taking any of them.

**F8-F11.** `Contains` was byte-identical to the pre-existing `IsResident`, and
its doc comment had been spliced in front of `capacity_exhausted()`, orphaning
that one; the duplicate is gone. The size-before-acquire ordering is now pinned
on a FULL cache, where it is observable -- the existing 2-slot case would have
taken a free slot and evicted nothing either way. `ReleaseHost` and
`AdoptDeviceBytesAsHost` cleared `mmap_src` but not `mmap_fd`, leaving a
descriptor that outlived its own subject. The `expert >= 0` ternary had an
unreachable arm: all three callers pass a loop index.

The step boundary now lives in `ForwardLayers`, which every MoE entry point
funnels through exactly once per forward, as an RAII guard so a step that ends by
throwing still ends. `qwen3_moe.cpp` composes the same MoE block from another
translation unit and marks its own step for the same reason.

## The observability review (#1091): the instrument could not report its own defect

17 August 2026. A fresh review of the F1-F11 repair above returned FAIL on six
findings. None of them was a red test. Every one was a gap in what the gate could
see, which is the same class as the defect the repair had just fixed.

**The statistics line could not print the number the docs told an operator to
read.** `ReportStats` had exactly one caller, `EndStep`, and it returned early on
`steps == 0`. So a run whose step boundary is never reached — F1, the reason the
line exists — printed nothing at all. Measured on one binary with
`VT_MOE_EXPERT_STREAM_STATS_EVERY=1`: healthy, 8 lines; F1 reinjected, 0 lines
and only the startup banner. `docs/ENVIRONMENT.md` and `docs/USAGE.md` both told
the operator to read `steps == 0` off a line that could not exist, and the
**absence** was the signature.

A second consequence found while acting on it: `stats_every_` defaults to 16, so
a short healthy run prints nothing either. A benchmark that reads absence as
failure therefore reports VOID on a working lane, which is what happened to the
streaming benchmark and is why it had to be restarted.

The repair is one final line, printed from the store's own destructor, once per
process, crossing both early returns. Not a second teardown hook registered when
streaming is REQUESTED, which was the first shape tried: on a CPU-only host that
hook's only unique job — a run that asks for streaming and never builds a store —
is not reachable by any test, because `Reserve` and `Get` are called from the same
call chain. It would have been an untestable branch added to fix an
untestable-branch problem. What replaces it is a protocol the docs now state:
the `[expert-stream] ON ...` banner says a store was built, the final line says
what it did, and each combination of present/absent means one thing. The docs
carry four rather than three: an in-process flush through the exposed seam takes
the once-flag, so a gate that calls it leaves the teardown line absent, and only
a gate can produce that shape (#1106).

**`CHECK(s.advised > 0)` could not fail for the defect it named.** Reinjecting the
pre-fix unaligned `madvise` address exits 0 in 40 of 40 runs. The measured reason
is that `> 0` over 48 calls is satisfied whenever heap layout happens to
page-align a single slice, and one did: `advised=1` against `fills=48`. The
assertion is now `advised == fills`, which is the true healthy invariant on this
arm — madvise is issued on the mapping-copy path only, and only when the key is
not already resident, which is exactly the condition under which `EnsureSpan`
goes on to fill, with `exhausted == 0` asserted beside it as the premise. Verified
stable over 50 consecutive runs before being asserted, rather than after.

**"Every MoE entry point funnels through here exactly once per forward" was
false.** Four more forwards reach `ExpertMlpKq -> KqExpertSlice`:
`Qwen3_5Model::ForwardDense`, `Qwen3_5MTPModel::Forward`,
`Qwen3_5MTPModel::ForwardPaged` and `Qwen3_5ReplayLayer`. One of them,
`Qwen3_5MTPModel::ForwardPaged`, is the production spec-decode DRAFT forward, so
a draft's acquisitions stayed `protected_this_step` across the following target
forward — F1 at draft scale. This paragraph said "the MTP pair" until #1106
finding 2 measured it; the other three are parity-only entry points, and that is
recorded below and under `## Owed` as #1108.

ONE FORWARD IS ONE STEP, and that is the call the draft forced. Folding a draft
into the target's step would pin the draft's slots across a second forward for no
benefit, so each draft gets its own step and a spec-decode iteration advances the
clock once per draft plus once for the target. The opposite mistake is the one
adding guards invites — a guard nested inside another ends the step twice, which
decays every resident entry an extra tick for a step that never happened — so the
guard now REFUSES to nest, stated as a precondition in the same idiom
`MatmulF32Slice` uses for `expert >= 0` rather than handled. That refusal was
STATED here and pinned by nothing, which the next review measured; see #1106
below. `RunMoeBlock` stays
deliberately unguarded: it is one block, not a forward, and qwen3_moe.cpp owns
the boundary for the model that composes it. That exemption is what makes the
`steps == 0` case above constructible without breaking anything.

**Three more, smaller.** `EnsureFile` — the arm every real GGUF-mmap checkpoint
takes — was reached by no test, so the `file_offset + offset` composition was
unverified; it is now driven from a temp file at a deliberately awkward offset
(4109 bytes: past a page, not on a page, not on a 34-byte Q8_0 block), and the
arm is PROVEN rather than assumed by `advised` staying flat while `fills` grows,
which is the one number that separates a pread from an `EnsureSpan`.
`OwnedTensor::TowerUid`'s comment promised an identity for "this tensor's CURRENT
bytes" while the code keys on `bytes.data()`; the comment now states where the
guarantee stops, and a borrowed-buffer case pins both halves, because #1066 was
that same overclaim on that same field. `SetForceFallback` has no production
caller and was incrementing the operator-facing `exhausted_`, so a gate asking for
the unstreamed arm told an operator to raise a budget that was never the reason
(measured: `exhausted=42` from the switch alone); it has its own counter now, kept
off the stderr line because in a production process it is always zero.

Thirteen mutations, thirteen caught, each recorded with a non-empty
`git diff --stat`, a zero compile status and a non-zero doctest case count. The
first attempt at two of them was INVALID rather than passing — one did not build
(`-Werror` on an unused variable), and two reported the CHILD process's doctest
summary because a failing case dumps the child's output into the parent's log,
so the first `test cases:` match in the file belonged to the child.

## The review of that repair (#1106): three claims outran the code

17 August 2026. A fresh review of the pull request above returned FAIL. The six
functional repairs are correct and all thirteen mutation claims reproduce
independently. What failed is what was said about them, and three of the four
findings are the class that pull request was fixing.

**The comment asserted a mechanism that does not exist.**
`qwen3_5_internal.h` said the final line is reached "at process teardown: a
static registered the first time streaming is requested, plus the store's own
destructor, whichever runs first". There is no such static — the same pull
request says in its own body that the hook was deliberately not built, and the
grep for `atexit` returns nothing. It also claimed "exactly one line per
process, even on a run with zero steps" without the two qualifiers `docs/USAGE.md`
carries: a store must have been BUILT, and the process must RUN its static
destructors. This is #1091 finding 5 — a comment promising more than the code —
reintroduced one file away in the change that fixes it, which is the strongest
argument on record that the class is a habit rather than an accident.
`~Qwen35ExpertStream` is now named as the only production path to the LINE, with
both qualifiers, and the note says what calling the exposed seam costs: it takes
the once-flag, so it suppresses the teardown line for the rest of the process.
The header says that `ExpertStreamFlushStats` itself has ZERO production callers
and that the destructor does not route through it, because the first repair of
this finding headed that comment "`~Qwen35ExpertStream` IS THE ONLY PRODUCTION
CALLER" — true of the line, and read as a call that the destructor deliberately
does not make.

**"Nothing lands dead" was claimed for four step guards and holds for one.**
Only `Qwen3_5MTPModel::ForwardPaged` has a production caller
(`runner.cpp:2183` -> `spec_decode/mtp/speculator.cpp:107,262`). The other three
sit in parity-only entry points: `Qwen3_5MTPModel::Forward` is reached only
through `ForwardLogitsHost`, itself a "standalone parity convenience" with no
caller outside `tests/`; `Qwen3_5Model::ForwardDense` is the parity reference by
its own header; `Qwen3_5ReplayLayer` is per-layer parity replay. A call site
inside a test is not reach. Nothing is deleted — the guards are correct where
they sit and become live the moment any of those entry points gains a production
caller, and adding the guard later WITH the caller is exactly how this row lost
its step boundary the first time. What changes is the record: they are named as a
staged slice that lands unreached, in the commit body, in the pull request body
and under `## Owed` below, tracked as #1108.

**The nesting refusal was asserted everywhere and pinned nowhere.** The source,
the spec and the pull request body all stated that the guard refuses to nest.
Deleting its `VT_CHECK` left both focused binaries fully green — 6/6 and 4/4 —
and it appeared in none of the thirteen mutations. It is unreachable through
production code by construction: every forward that takes expert slices is a
complete forward that no other one contains, so no legitimate call graph nests
one. `detail::ExpertStreamStepScope` exists for that and nothing else, and it
forwards to the guard's own `Begin`/`End` rather than restating the flag, so a
gate holding it measures the production boundary. The case asserts the refusal
twice: a second scope throws, AND a real `ForwardDense` entered while the scope
is held throws too — the second is what proves the two share a boundary rather
than agreeing by coincidence, and a mutation that gives the scope a parallel flag
kills only that pair.

The refusal is deliberately NOT gated on `Qwen35ExpertStreamRequested()`. "One
forward is one step" is a property of the call graph, not of the streaming lane,
so a nest is a defect whether or not a store exists. Arming it only under
streaming — the rare configuration — would let the default path establish a nest
that nobody sees until someone turns streaming on, which is this row's recurring
shape. The cost is that a nest reds every Qwen3.5 forward and not merely the
streamed ones, and that is the intended polarity.

**The MSVC repair was incomplete.** `::setenv` sat at namespace scope in both new
gates with no `_WIN32` guard. It is POSIX; MSVC's CRT has only `_putenv_s`, the
targets are added unconditionally, and `build-windows-release.ps1` configures
`VLLM_CPP_BUILD_TESTS=ON` — so the translation units did not compile there at
all, and the claim that the step-clock cases are "built everywhere" was false.
Both now use `vllm_test::SetEnv` from `support/test_env.h`, which has been the
one place that branch lives since #603. CI could not report it because the
Windows lanes fail earlier, inside the product library, on #1068; a lane that
never reaches a test translation unit cannot fail in one. The static checker that
could have is blind to it twice over — it scans only the shipped-server sources
and knows neither `setenv` nor `unsetenv` — filed as #1107 against
`ENG-RELEASE-WINDOWS` and not fixed here, because changing a checker's semantics
needs its own spec and red-before evidence.

Three mutations on the added guarantee, three caught, each with a changed sha256,
a zero compile status and a non-zero doctest case count: deleting the `VT_CHECK`
(7 cases, 1 failed, all six assertions of the new case red, and `Steps()` reading
3 where 1 is correct — the double-count the guard exists to stop); dropping
`Open() = false` from `End` (7 cases, 4 failed); and giving the scope its own
parallel flag (7 cases, 1 failed, exactly the two assertions that pin the shared
boundary). The Windows repair is NOT mutation-proven: no MSVC is reachable from
this host, and the checker that would have caught it statically is the subject of
#1107.

## `--device cuda` loads for 26 minutes and then dies: the allocation, named (#1123)

17 August 2026. The same checkpoint that serves on `--device cpu` reaches a
serving state on `--device cuda` on the same box and the same binary, and then
dies on the first request with `vt cuda: cudaMalloc: out of memory` inside the
EngineCore busy loop. The load succeeded, so the failure is the forward.

The log line cannot say which allocation failed or how big it was:
`CudaBackend::Alloc` is `Check(cudaMalloc(&p, bytes), "cudaMalloc")`
(`src/vt/cuda/cuda_backend.cu:77-81`) and `Check` composes
`"vt cuda: " + what + ": " + cudaGetErrorString(err)`
(`cuda_backend.cu:48-52`), where `what` is the literal `"cudaMalloc"`. `bytes`
is in scope and discarded. So the size was established from the code and the
checkpoint rather than from the message.

### The allocation

`ResidentWeight` (`src/vllm/model_executor/models/qwen3_5.cpp:963-1025`) aliases
the host bytes when `GetPlatform(...).is_cpu()` and otherwise **uploads the whole
tensor**: `const size_t nb = w.bytes.size(); void* p = d.b.Alloc(nb);`
(`qwen3_5.cpp:1010-1011`). For a routed-expert weight, `w` is the STACKED
`[E*N,K]` keep-quant tower, so `nb` is one tower — every expert of one matrix of
one layer, in one contiguous `cudaMalloc`.

**Both switch positions of the keep-quant MoE path reach that same line**, which
is why no knob avoids it. Every `f:N` below is a CALL SITE — the line inside `f`
that invokes the next hop — never a definition line, so the chain can be walked
one `sed` at a time:

| Configuration | Path | Reaches |
|---|---|---|
| default (`VT_QWEN35_GROUPED_MOE` unset ⇒ on) | `MoeBlock:6615,6616,6620` → `KqGrouped:5694` | `ResidentWeight(d, w_kq)` |
| `VT_MOE_EXPERT_STREAM=1` (which DISABLES grouping, `:5670-5676`) | `ExpertMlpKq:5651,5652` → `MatmulF32Slice:5611` → `KqExpertSlice:5595` | `KqResidentSlice:5114` → `ResidentWeight` |

`KqExpertSlice`'s slot arm is guarded by `is_cpu()` (`qwen3_5.cpp:5578`), so on a
device platform it falls through before the store is even constructed. That is
the `## Owed` line "Streaming serves the CPU-resident borrowed tower only"
observed from the other end: the lane that makes the model fit has no device arm,
and the device path therefore asks `cudaMalloc` for every expert byte.

Ruled out by reading, not by assumption. `src/vt/cuda/cuda_moe.cu` and
`cuda_glue.cu` contain no allocation at all. `BuildMoeMarlinResident`
(`qwen3_5.cpp:6010-6215`; its `E ×` per-expert allocations are `:6049-6064`, plus
two repack temporaries at `:6094-6095`) does allocate unpooled, and it
is NOT on this path: `MoeBlock` takes the fp4/Marlin arm at `:6555`, guarded by
`const bool fp4 = !w.expert_gate_fp4.empty()` at `:6548`, and a GGUF keep-quant
load populates `expert_*_kq`, not `expert_*_fp4`.

### The size, measured from the checkpoint

Both GGUF tensor tables of `unsloth/Qwen3.8-2.4T-A95B-GGUF` at revision
`567d3e6ac26c5474b18311e619c04350fb9a5556` were re-censused independently of the
earlier census in this spec, by HTTP range request over all ten shards, with no
tensor data downloaded: **1702 tensor records parsed against the 1702 declared in
`split.tensors.count`**, and the two numbers agreeing is the coverage claim.
Shard 1 carries the 58 metadata keys and ZERO tensors; shards 2-10 carry the
table.

| Encoding | ggml | Tensors | Bytes | % bytes |
|---|---|---|---|---|
| IQ1_XXXS (routed experts) | 66 | 276 | 351,918,882,816 | 88.59 |
| Q5_K | 13 | 420 | 23,391,633,408 | 5.89 |
| Q6_K | 14 | 162 | 8,876,851,200 | 2.23 |
| Q2_K (the `nextn` MTP block's 3 towers) | 10 | 3 | 8,455,716,864 | 2.13 |
| Q4_K | 12 | 2 | 2,288,517,120 | 0.58 |
| F32 | 0 | 838 | 2,171,133,440 | 0.55 |
| Q8_0 | 8 | 1 | 142,606,336 | 0.04 |
| total | | 1702 | 397,245,341,184 | 369.96 GiB |

`expert_count = 512`, `embedding_length = 8192`,
`expert_feed_forward_length = 2048`, `expert_used_count = 10`,
`block_count = 93`. So one tower is:

| Tower | Bytes | | Count |
|---|---|---|---|
| IQ1_XXXS `ffn_{gate,up,down}_exps`, dims `[8192,2048,512]` / `[2048,8192,512]` | **1,275,068,416** | 1.1875 GiB | 276 |
| Q2_K `ffn_{gate,up,down}_exps` (block 92, the MTP layer) | **2,818,572,288** | 2.6250 GiB | 3 |
| all `*_exps` | **360,374,599,680** | **335.62 GiB** | 279 |

`1,275,068,416 / 512 = 2,490,368` bytes per expert slice, which is exactly the
`slot_bytes=2490368` the W4 banner printed, so the arithmetic here and the
running lane agree on the same weight.

**So the answer to "which allocation and how big" is: `qwen3_5.cpp:1011`,
1,275,068,416 bytes at a time (2,818,572,288 for three of them), 279 times,
335.62 GiB in total.**

### Why it fails, and why the loader does not

The budget is measurable, and `nvidia-smi` is the wrong instrument for it: on
this box it answers `[N/A], [N/A], [N/A]` for `memory.total,memory.free,
memory.used`, and the `rc` fleet label likewise records `vram=[N/A]M`.
`cudaMemGetInfo` answers honestly. Measured on `dgx:gpu0` under an `rc` hold,
through `libcudart.so.13` in the `vllmcpp-build:gb10` image:

```
cudaGetDeviceCount rc = 0 count = 1
cudaMemGetInfo     rc = 0
free  = 122059919360 (113.677 GiB)
total = 128452956160 (119.631 GiB)
attr Integrated           rc=0 value=1
attr UnifiedAddressing    rc=0 value=1
```

`total` is EXACTLY `/proc/meminfo MemTotal` (125442340 kB) times 1024, and equals
the fleet's own `mem_total_bytes=128452956160`. **One unified pool, correctly
reported by the CUDA runtime and not by `nvidia-smi`.**

Against that pool, 335.62 GiB of tower staging is 2.8x the whole machine, and
the total device-resident weight demand (towers plus the dense remainder, which
this spec measured at 62 GiB resident on the CPU arm) is over 3x. The load
survives because a borrowed tower costs ZERO anonymous bytes — that is the
finding in "Why a 370 GiB model fits in 119 GiB" above. Staging converts each
borrow into a real allocation, so the pool is exhausted after roughly
`(119.6 - 62) / 1.1875 ≈ 48` towers, i.e. partway through layer 16 of 93, on the
first forward. Hence: 26 minutes to READY, then death mid-stream.

### The refusal: what it keys on, and what it deliberately does not

Loading for 26 minutes and dying mid-stream is the worst of the three available
behaviours, and AGENTS.md already says which one is right: refuse an
unimplemented arm at load with a message that names the missing part. This
change lands that refusal and nothing else. It does NOT build the device-slot
arm; that stays owed below.

The predicate is keyed on the MEASURED condition, never on "CUDA + GGUF" and
never on an architecture name, because a GGUF that genuinely fits the pool must
still load:

```
refuse  ⇔  needs_weight_staging()  ∧  budget_known  ∧  staged_lower_bound > budget
```

Three properties are deliberate.

**A PER-TENSOR lower bound, and it is wrong in both directions.** Per tensor the
bound is `min(gguf_bytes, elems × model_dtype_bytes)`. A weight the loader keeps
quantized is staged verbatim, which is `gguf_bytes`; a weight it expands is staged
at the model dtype, which is `elems × 2` for bf16. Taking the minimum makes each
term a true lower bound on THAT tensor's staged size.

It does not follow that the sum is a lower bound on the load, and the first
version of this section claimed it did — "so the refusal can never over-refuse".
The review that caught it supplied the counter-example, which is present on every
default load:

- **Over-count.** A tensor counted and never staged is a positive error. The MTP /
  `nextn` block is attached only when a speculator is configured, so on a default
  load block 92 of the target checkpoint — 20 tensors, 8,940,488,704 bytes, 8.33
  GiB, **2.2506 %** — is counted and not staged. A budget in
  `[what a default load stages, what this counts)` refuses a weight set that fits.
- **Under-count.** The bound omits KV cache, activations, scratch pools and the
  CUDA context, so a checkpoint at 0.95x of the pool still passes here and still
  dies later.

The two errors are on DIFFERENT quantities and do not cancel, so "the under-count
dominates" is not an argument that the refusal is safe — it is an argument about a
number the refusal never compares. `gguf_device_fit.h` said this correctly from the
day it landed; the spec and the commit body did not, which is why the wording here
is now the header's. `test_gguf_device_fit` carries a counted-but-unstaged fixture
so the over-count direction is executable rather than described, and both remainders
are owed: the over-count to [#1136](https://github.com/mudler/vllm.cpp/issues/1136),
because closing it means teaching the bound which tensors THIS load will stage,
which is load policy and not a property of the file; the under-count to the startup
memory profile `KV-WARMUP-PROFILE` owns. A headroom fraction invented here would be
the guess this bound exists to avoid, in either direction.

**`total`, not `free`.** `free` at load time carries the page cache and whatever
else the box is doing, so it makes the refusal a function of contention. `total`
is a device property.

**Unknown is not a verdict.** `ResidencyPolicy::device_memory_total_bytes` is 0
on every platform that does not probe one, and 0 means UNKNOWN. A caller that
cannot learn the budget declines to decide, so no non-CUDA device and no CUDA
build without the probe changes behaviour. This is the polarity
`gemma4_moe.cpp:506` chose for the opposite reason (it refuses the device
allocation on unknown, because a hung `hipMalloc` is worse than a host
fallback); here the risk runs the other way, since refusing a load on an unknown
budget would break every device whose budget nothing reports.

**Which platforms this actually covers**, because the first version of this
section and `docs/USAGE.md` both got it wrong in the same way. The two predicates
coincide on exactly one platform: `needs_weight_staging()` is true only on
`CudaPlatform` (`src/vllm/platforms/cuda.cpp:71`) and a budget is probed only
there. So **every** NVIDIA GPU this build runs on gets both the probe and the
refusal — a discrete card is `CudaPlatform` too, not a separate case. ROCm,
Vulkan and Metal answer `needs_weight_staging() == false` (ROCm says so
explicitly, `src/vllm/platforms/rocm.cpp:74`), which means they read the mapping
where it lies and have no staging allocation to fail: the refusal is not "owed"
to them, it is inapplicable. What IS owed on ROCm is the separate
`Backend::DeviceMemoryInfo` capability (#1126).

The probe is added to `CudaPlatform`, which already includes `<cuda_runtime.h>`
and already probes device attributes at registration, and NOT to
`Backend::DeviceMemoryInfo`. That seam's comment claimed "ROCm/CUDA override with
hipMemGetInfo/cudaMemGetInfo" and only ROCm does
(`src/vt/rocm/rocm_backend.hip:358-365`). The comment is **corrected in this
change**, in the two places that carried it: `include/vt/backend.h:78-93` on the
seam, and `gemma4_moe.cpp:440-448` on the only call site — the second copy was
found by this round's audit and is why the first correction alone would have left
the claim in the tree. Overriding the seam would also silently wake `Gemma4MoE`'s
device-expert LRU, whose `MakeRoom` refuses on CUDA today precisely because the
query is absent (`gemma4_moe.cpp:506`). Waking another model's residency policy is
a behaviour change with its own measurement, so it is filed rather than done.

### Tests, and how the device branch is reached on a CPU-only host

`needs_weight_staging()` is true on exactly one platform in this tree
(`src/vllm/platforms/cuda.cpp:71`), so the branch is unreachable from the real
loader on a host with no CUDA device — the untestable-device-branch shape this
row has hit repeatedly. It is reached here by registering a FAKE staging
platform in the CUDA lookup slot, which is the instrument
`tests/vllm/entrypoints/test_device_selection.cpp` already established for
exactly this reason, in its own executable so the global registry cannot leak
into other suites.

| Case | Instrument |
|---|---|
| the arithmetic, both directions and the boundary | `GgufStagedWeightFootprint` / `CheckDeviceWeightFit` over a table: `>` refuses, `==` and `<` do not, unknown budget does not, a non-staging platform does not, an F32 tensor is counted at bf16 and a quantized one at its GGUF size |
| the bound's OVER-count direction | a fixture carrying a `blk.N.nextn.*` tensor a default load never stages: the footprint counts it, and both ends of the resulting over-refusal window are asserted, so the direction cannot be claimed away again |
| the AUTO arm names the device the load will RUN on | a fake staging platform whose backend's `CreateQueue()` can be made to throw, driven through `FromModelDir` twice at the same budget: it refuses when the queue can be created, and refuses NOTHING when it cannot, because that load runs on CPU |
| the CUDA residency policy assembles the probed budget | `CudaResidencyPolicy` in `platforms/interface.h`, unit-tested on every host, so the assignment is no longer reachable only in a CUDA build |
| the refusal is REACHED from the loader | `LoadedEngine::FromModelDir` on a synthetic `qwen35moe` GGUF with the fake staging platform registered and a small `VT_DEVICE_WEIGHT_BUDGET_BYTES`: the thrown message is the fit refusal |
| a fitting GGUF still loads | the SAME call with a generous budget: the throw is a LATER, different one (the synthetic file has no tokenizer), which is what proves the check let it through rather than that it never ran |
| the CPU arm is untouched | the same file with `device=cpu` never refuses, whatever the budget |

## The user-facing recipe and the checkpoint pin ([#1194](https://github.com/mudler/vllm.cpp/issues/1194))

`docs/USAGE.md` named `Qwen3.8-2.4T-A95B UD-Q1_0` three times and pinned it zero
times. AGENTS.md binds the pin to that file: file name, size, repo AND revision,
and a sha256 for a quantized artifact. The streaming section there is a MECHANISM
reference. It never said which file `--model` takes, what the load costs, what
decode costs, or where the ceiling is, while the four sibling per-model recipes
in the same file all do.

Landed as a fifth sibling recipe, `Qwen3.8-2.4T-A95B UD-Q1_0: 370 GiB served from
a 119 GiB box`, which LINKS the mechanism section rather than restating it, so
each fact keeps one home. Three facts are repeated on purpose and the section
says which ones and why: which device to use, the expert bytes a token reads, and
the two streaming decode figures. Both places quote `ENG-EXPERT-STREAM-DEVICE`
W0e, so the re-measure that row owes has to change both.

The pin is settled against the ARTIFACT rather than against a document or the
HuggingFace tree API, whose `lfs.oid` is fabricated for a gated repo: TEN shards
`UD-Q1_0/Qwen3.8-2.4T-A95B-UD-Q1_0-000{01..10}-of-00010.gguf` summing to exactly
397,256,393,248 B (369.97 GiB), with shard 1 declaring `split.count = 10` and
`split.tensors.count = 1702` in its own metadata and carrying no tensors. That
also answers [#1420](https://github.com/mudler/vllm.cpp/issues/1420), whose two
copy-paste commands named `-of-00008`, a file that does not exist at that
revision.

The section publishes no new measurement. Every figure in it was already
recorded, it carries no ratio between runs taken on different source trees, and
it carries no `--device cuda` speed number, because that arm's token gate fails.
The W0e and W0f figures come from the harness
`benchmarks/expert_stream_device_w0e.cpp` and not from the `vllm-server` command
the section publishes, and the section discloses that difference beside the
prompt and environment differences it already disclosed. The 16 August 2026
streaming-off run is the one figure that did come from `vllm-server` ([#1447](https://github.com/mudler/vllm.cpp/issues/1447)).

## Owed

Carried debt for this row. Each item names why it is not closed here.

| Owed | Why it is open |
|---|---|
| ~~**Re-measure decode on a LIVE cache.** The `docs/BENCHMARKS.md` decode figure for this row was taken with the step clock dead from token 3 onward and is void.~~ **CLOSED by `ENG-EXPERT-STREAM-DEVICE` W0e**, 2026-08-18 (`c805ccbb3`, [#1414](https://github.com/mudler/vllm.cpp/pull/1414)): streaming-ON decode on a live cache is **11.05 s/token** steady at 4000 slots, rep 2's median over steps 4 to 32 with rep 1 at 11.22, and the decode-phase `exhausted` delta is 0 in the same run. Recorded in [`../benchmark-record.md`](../benchmark-record.md) under `ENG-EXPERT-STREAM-DEVICE W0e`. | Kept as a line rather than deleted, because the reason the figure was void is what shaped the replacement. `Qwen35ExpertStream::EndStep()` had no caller, so `protected_this_step` never cleared, `Acquire` returned -1 once the cache filled and every slice from partway through token 3 came from the mmap, which IS the baseline. That is why W0e gates `exhausted` on its DECODE-PHASE DELTA rather than on a total, and why it reports two reps. The host blocker this entry named is spent: the measurement ran on `dgx:gpu0` inside one `rc` lease. The TTFT row of the same 16 August run was never void and is unchanged. |
| **The `pread` path has never run on the model.** `EnsureFile` now has a CPU-local gate that drives it through the production seam from a temp file and proves the `file_offset + offset` composition (#1091 finding 4), so it is no longer UNREACHED. It is still unmeasured on a real checkpoint. | Same host. Three earlier attempts were OOM-killed at 48.6 GiB anon beside another session's 32.6 GiB job. |
| **A run that REQUESTS streaming and never builds a store prints no statistics line.** The `[expert-stream] ON ...` banner is absent in that case too, so no-banner means "nothing reached the lane" and banner-without-line means "the process died"; the docs state all four shapes. | A teardown hook that could report it is not reachable from any test on a CPU-only host, because `Reserve` and `Get` sit in one call chain and a device platform is what separates them. Landing it would have been an untestable branch added to fix an untestable-branch problem. Needs `dgx.casa` (see #1091). There is NO such hook in the tree: `~Qwen35ExpertStream` is the only production path to the final line, and it prints it directly rather than through `ExpertStreamFlushStats`, which has no production caller at all. The header now says both, rather than describing the hook that was rejected (#1106). |
| **Three of the four step guards land UNREACHED.** `Qwen3_5MTPModel::Forward`, `Qwen3_5Model::ForwardDense` and `Qwen3_5ReplayLayer` are parity-only entry points with no caller outside `tests/`, so their `Qwen35ExpertStreamStep` guard is reached by no production path. Only `Qwen3_5MTPModel::ForwardPaged` is (`runner.cpp:2183` -> `spec_decode/mtp/speculator.cpp:107,262`), and even that caller is "UNREACHABLE unless a speculator is configured" (`runner.cpp:2120`) — so a DEFAULT-configuration run reaches none of the four, which is a weaker statement than "one of four is reached" and is recorded here rather than rounded up. Owning row `ENG-EXPERT-STREAM`; tracked as [#1108](https://github.com/mudler/vllm.cpp/issues/1108). | Nothing is deleted, because the guards are correct where they sit and cost nothing, and the alternative — add the guard later, together with the caller — is precisely how this row lost its step boundary in the first place. It closes when one of those entry points gains a production caller, or when they are retired as parity references. Neither is scheduled and neither should be forced by the record. |
| **`check-windows-portability.py` cannot see this class.** It scans only the sources reachable from the shipped server target, so no test translation unit at all, and `setenv`/`unsetenv` are in none of its patterns. Tracked as [#1107](https://github.com/mudler/vllm.cpp/issues/1107) against `ENG-RELEASE-WINDOWS`. | Changing a checker's semantics needs its own spec, a red-before test and green-after evidence, which is a different unit of work from repairing two test files. Widening the scan to `tests/` also has to separate a guarded POSIX call from an unguarded one across a large surface, and that wants measurement rather than a guess. |
| **The Windows repair is not mutation-proven.** Both gates now use `vllm_test::SetEnv`, and nothing here executed an MSVC compile of them. | No MSVC is reachable from this host, and the Windows CI lanes fail earlier in the product library on #1068, so they cannot report a test translation unit either way. The static checker that could have is #1107. |
| **The `MADV_WILLNEED` readahead is unmeasured.** It is now well formed and counted; whether it moves decode is unknown. | Same host and the same OOM contention. No speedup is claimed anywhere for it. |
| **Windows has no streaming.** `EnsureFile` throws `"EnsureFile needs pread"` on `_WIN32`, and `SourceOfSpan` returns `fd = -1` there, so the lane falls back to the mapping copy. | No `pread(2)`; needs an `OVERLAPPED`/`ReadFile` arm. Refused by name rather than silently degraded. |
| **The CUDA arms of [#1029](https://github.com/mudler/vllm.cpp/issues/1029)'s grouped gate have not run on a device.** | Recorded in that issue, which stays open for it. Unchanged by this repair. |
| **Streaming serves the CPU-resident borrowed tower only.** A staged device weight takes `KqResidentSlice`, so there is no device-slot arm. | Deliberate for phase 1 (copying a device-resident weight through host slots moves MORE bytes); W7 owns the pluggable backing store. |
| **The grouped keep-quant MoE path and streaming are mutually exclusive.** `VT_MOE_EXPERT_STREAM=1` disables grouping and says so once on stderr. | Grouping stages the whole tower, which is what streaming exists to avoid. Making them compose needs a slot-aware grouped GEMM, which is its own row. |
| **`--device cuda` still cannot SERVE a larger-than-pool GGUF; it only refuses by name now.** The device-slot arm is the missing capability: a `DeviceExpertSlotStore` behind `ExpertSlotStore`, a read accessor on that interface (`KqExpertSlice` reads `HostExpertSlotStore::Slot()`, the CONCRETE class, so the seam cannot be swapped today), a device filler that is not `pread`-into-host (`ExpertSlotStore::SlotForWrite` is handed straight to `::pread`, `expert_streamer.cpp:76-94`), and lifting the `is_cpu()` guard at `qwen3_5.cpp:5578`. Sized by the measurement above: 2790 slices per token at 2,490,368 bytes is 6.95 GB per token against a 119.631 GiB pool that already holds the dense remainder. Tracked as [#1124](https://github.com/mudler/vllm.cpp/issues/1124). | It is a campaign, not a fix: W7 (the pluggable backing store) is its declared owner in the work breakdown, and the CPU arm's own I/O rate is still unmeasured on a live cache two rows above. Building a device lane on top of a host lane whose bandwidth number is void would be optimising against a number nobody has. |
| **The fit bound omits everything that is not a weight.** KV cache, activations, the scratch pools and the CUDA context are not counted, so a checkpoint at 0.95x of the pool passes the refusal and still dies on the first forward. | A headroom fraction invented here would be exactly the guess the per-tensor bound exists to avoid. The number wants the startup memory profile that `KV-WARMUP-PROFILE` owns (`INVENTORIED`; upstream's is `GPUWorker.determine_available_memory`, `vllm/v1/worker/gpu_worker.py:451-495`, around `profile_run`, `vllm/v1/worker/gpu/model_runner.py:682`), which is a different row. Those two anchors are stated here from the pinned tree because that row's own three anchors are stale at the current pin, and `gguf_device_fit.h` had copied two of them — filed as [#1139](https://github.com/mudler/vllm.cpp/issues/1139), owned by `KV-WARMUP-PROFILE`, blocked here only by the `engine-matrix.md` record lock #1119 holds. |
| **The fit bound also counts too MUCH, and that direction can refuse a load that fits.** A tensor present in the file and not staged by THIS load is a positive over-count. On a default load that is the MTP / `nextn` block: 8,940,488,704 bytes, 8.33 GiB, 2.2506 % of the target checkpoint. A budget in that window refuses a weight set that would have fitted. | Not closed here. Closing it means the bound taking a per-tensor staging POLICY as input, which is the caller's knowledge and not the file's, and the exclusion's own failure mode is an under-count to nothing — which restores the 26-minute-then-OOM this row exists to remove, on a device nobody here has to measure it on. So the direction is stated in `gguf_device_fit.h`, pinned executably by `test_gguf_device_fit`, exposed to operators in `docs/USAGE.md`, and tracked as [#1136](https://github.com/mudler/vllm.cpp/issues/1136). `VT_DEVICE_WEIGHT_BUDGET_BYTES` is the way out of the window in the meantime. |
| **`Backend::DeviceMemoryInfo` has no CUDA override, and waking it is not the one-line port of the ROCm one that #1126 describes: on CUDA it would wake a THROW.** Only ROCm implements it (`src/vt/rocm/rocm_backend.hip:358-365`), so `Gemma4MoE`'s device-expert LRU refuses on every CUDA device (`gemma4_moe.cpp:506`). Where a per-expert FP8 checkpoint is present, the expert's BF16 bytes are re-copied HOST->DEVICE on every use instead (`ExpertGeGLUHost`, `gemma4_moe.cpp:49-74`, reached at `:1515-1521`; the H2D is `:59-60` and it drains the queue per expert at `:73`), silently and for the life of the process. That H2D cost is CONDITIONAL, not present-tense: the whole device LRU is `ex.is_fp8`-gated (`:991`, `:1506`), and by point (1) below no such Gemma-4 checkpoint is pinned anywhere, so on CUDA today the LRU-MISS fallback is never even asked for. (A BF16 Gemma-4 checkpoint reaches `ExpertGeGLUHost` at `:1525` too, but through the host-weight branch the LRU never governs, so it is not a cost of the missing probe.) The name misleads: it computes on the DEVICE from host-resident weights, so what the dead LRU would cost is bandwidth and a per-expert `Synchronize`, not a wrong answer. | The CAPABILITY is [#1126](https://github.com/mudler/vllm.cpp/issues/1126) and is still not built. Note that the MERGED `#1126` index row cites `rocm_backend.hip:338-345`, which a later commit moved to `:358-365`. That row is append-only and is deliberately left alone: editing a merged row makes a union merge DUPLICATE it rather than merge it, which is a worse outcome than one stale number. This spec row carries the current anchor and is authoritative for it. The false COMMENT was corrected by this row in both places that carried it, `include/vt/backend.h:78-93` and `gemma4_moe.cpp:440-448`; both anchors were re-verified exact against `fd64c76ee`, as were `rocm_backend.hip:358-365`, `gemma4_moe.cpp:506`, `platforms/cuda.cpp:71` and `platforms/rocm.cpp:74`. SCOPE, because the previous headline read as coverage it did not have. The first review repair re-audited every anchor cited by the FOUR `## Owed` rows in this cluster — this one, the #1126 step-3 row, the #1197 row and the #1205 row — against the repaired tree: 53 examined, 53 exact, 0 stale. It audited those four rows and nothing else. The GitHub ISSUE BODIES were never in the audited set, and #1205's body was in fact 11 lines stale at `7beada17c` for exactly that reason. The second review repair widened the set: it re-derived every anchor in those four rows AND in the #1197 and #1205 issue bodies AND in the #1205 index row against the final tree — 71 examined, 71 exact, 0 stale after repair. Anchors from `gemma4_moe.cpp:549` onward moved by 22 lines in that repair, because the arm-existence guard it added sits at `:571`. **Four things were established while re-reading it for #1126, and each one raises the price of the override.** (1) *There is nothing to run it on.* The LRU needs `ex.is_fp8`, which only `LoadMoeFp8PerExpert` sets (`gemma4_weights.cpp:210-215`) from a per-expert `F8_E4M3` export; no such Gemma-4 checkpoint is pinned anywhere in `docs/USAGE.md` — all 32 lines there matching `gemma` case-insensitively were swept, and every checkpoint among them is the LTX-2.5 text tower (`gemma4-12b-with-proj*.safetensors`), not a per-expert FP8 MoE decoder, so the woken path cannot be exercised, here or elsewhere, until one is. (2) *The device-resident arm has no CUDA implementation to route into. It has a throw.* `ExpertGeGLUDeviceAccum` (`gemma4_moe.cpp:76-93`) READS as generic — `vt::MatmulBT`, `GeluAndMul`, `vt::MatmulBTAlphaBeta` — and an earlier draft of this row concluded from that reading that it "would run". It does not. `vt::MatmulBTAlphaBeta` (`src/vt/fused_ops.cpp:111-157`, dispatching at `:117`) is guarded on `#if defined(VLLM_CPP_HIP)` AND `q.device.type == kROCM`; its only implementation in the tree is `rocm::MatmulBTAlphaBetaRocm` (`src/vt/rocm/rocm_matmul_hipblaslt.hip:516`, declared `include/vt/rocm/rocm_matmul_batch.h:28`), and every other device falls through to a refusal. So the chain the override WOULD wake is: `EnsureGemma4Fp8ExpertOnDevice` (`gemma4_moe.cpp:548-608`) -> `lru.MakeRoom` (`:587`) succeeding as soon as `FreeBytes` can answer -> `true` at `:597` -> the call site at `:1508` -> `ExpertGeGLUDeviceAccum` at `:1509` -> `vt::MatmulBTAlphaBeta` at `:90` -> THROW, mid-decode. The `try`/`catch (...)` at `:585-607` wraps only the UPLOAD; the compute at `:1509` sits outside it, so the exception would propagate out of the decode step rather than degrading to the host fallback. **That chain is now cut at its first link.** `EnsureGemma4Fp8ExpertOnDevice` refuses at `:571` when `vt::HasMatmulBTAlphaBeta(d.q)` is false, BEFORE the upload, so the caller takes the `else` at `:1515-1521` — `EnsureGemma4Fp8ExpertCached` plus `ExpertGeGLUHost` — and the step answers instead of throwing. The refusal at `:90` stays as the backstop. It is latent today only because the other route into that function, `same_dev` (`:752-753`), needs `ex.gate_up_dev`, which is assigned nowhere but `src/vt/rocm/rocm_gemma4_experts.hip:207,226` — so the resident arm is UNREACHABLE off ROCm rather than safe. This is the actual blocker under #1126, it was recorded nowhere, and it is a stronger argument than the other three: filed as [#1205](https://github.com/mudler/vllm.cpp/issues/1205), and the refusal itself is now gated by `tests/vt/test_gemma4_rocm_fp8_seams.cpp`, which is what a CUDA implementation will have to satisfy. **Two corrections to the earlier draft's supporting claims, both of which overstated the case.** *The HIP-only list was one symbol too long.* Three of the four are genuinely HIP-only stubs inside `gemma4_moe.cpp`'s ONLY `#ifndef VLLM_CPP_HIP` block (`gemma4_moe.cpp:1596-1650`): `RunGemma4FusedTopkExpertGeGLU` (`:1621`), `PeerCopyGemma4Fp8ExpertSlice` (`:1629`) and `RunGemma4Fp8TopKOnExpertDevice` (`:1633`). `ExpertGeGLUDeviceBatched` (`:240`) is NOT: it sits OUTSIDE that block, in an anonymous namespace, with no HIP implementation and no header declaration, and is unconditionally `return false` under its own lab note (`:237-239` — gather+strided produced wrong tokens at ~23 t/s, pointer-batch ~0.8 t/s, serial/fused-gelu kept at ~34 t/s). It is disabled EVERYWHERE, ROCm included, so naming it beside the three inflated the ROCm/CUDA asymmetry. *The token-neutrality argument had the wrong mechanism.* The conclusion stands — the swap would not be token-neutral — but not because "the two arms sum the top-k experts in a different order". They do not: both run inside the SAME `for (int i = 0; i < top_k; ++i)` at `:1453`, so the summation order is identical. The real difference is ROUNDING and where the routing weight is applied. The resident arm folds `ww` into the GEMM `alpha` and accumulates in the epilogue with `beta` (`:1456`, `:1464`, `:90`), so the weight multiplies in the GEMM's own accumulator. The fallback writes the UNWEIGHTED product to a BF16 buffer (`:67`, `:70`) and applies `ww` afterwards with separate BF16 kernels — `vt::MulScalar` at `:1546` on the first expert, `MulScalar` plus `vt::Add` at `:1548-1549` on the rest. Two extra BF16 roundings per expert, with the weight applied post-rounding. (3) *The headroom test does not mean the same thing on the CUDA device this project gates on.* `MakeRoom` admits iff `free_b >= need + 1.5 GiB` (`:514`), a constant tuned on discrete dual R9700s where free VRAM is a quantity distinct from host RAM. On a GB10 it is not. This row already measured that `cudaMemGetInfo`'s `total` there is EXACTLY `/proc/meminfo MemTotal` times 1024, which is why it reached for that instrument where `nvidia-smi` answers `[N/A]`; its `free` is therefore reported over the same unified pool, and the host BF16 expert cache the device upload exists to relieve (`ex.cached_gu`/`cached_dn` via `EnsureGemma4Fp8ExpertCached`, bounded by the host LRU at `gemma4_moe.cpp:352`) is drawn from that pool too. So the admission test would double-count, on a box whose unified-memory OOM takes the host down with it. The `free` half is an inference from the measured `total`, not a second measurement, and it wants confirming on the device before any override lands. (4) *The gap is isolated, not a pattern.* Comparing overrides one by one across `src/vt/cuda/cuda_backend.cu` and `src/vt/rocm/rocm_backend.hip` at `fd64c76ee`, `DeviceMemoryInfo` is the ONLY optional `vt::Backend` seam ROCm answers and CUDA does not. |
| **#1126's own closing plan, step 3, must be NARROWED before it is done: the load-time fit check may read the seam's `total`, and never its `free`.** The issue proposes that "the #1123 fit check can then read the budget from the backend seam on every platform that reports one". As written that invites the live half, which would be a defect — but the seam returns BOTH halves (`bool DeviceMemoryInfo(size_t* free_bytes, size_t* total_bytes)`, `include/vt/backend.h:94`), and only one of them is illegitimate here. | The tree holds two answers to "how much device memory", and they answer different questions. `vt::Backend::DeviceMemoryInfo(free, total)` is a LIVE probe that moves with contention; it is authoritative for a RUNTIME admission decision — can this allocation succeed right now — and for nothing else. `vllm::platforms::ResidencyPolicy::device_memory_total_bytes` is a TOTAL probed once at platform registration with `0 == UNKNOWN`; it is authoritative for a LOAD-TIME budget verdict, which has to be reproducible and independent of whatever else the box is doing. Sourcing the load-time verdict from `free` would make the same checkpoint load or be refused depending on the page cache, which is precisely the property **`total`, not `free`** above was chosen to avoid. Reading `total` through the seam is not that. It is contention-independent, it is the same quantity `ResidencyPolicy` already carries, and a seam that answers it on every platform is a defensible place to source it from. What step 3 must additionally preserve is the PROBE-ONCE semantics: `device_memory_total_bytes` is probed at platform registration (`include/vllm/platforms/interface.h:70-72`), and a per-load live call would reintroduce the contention dependence by the back door even reading only `total`, because a load-time verdict has to be reproducible from the record rather than from the moment. So the defensible statement, and the one this row asserts: **step 3 may read only `total`, never `free`, and must keep the value probed once at platform registration.** An earlier draft of this row said "never step 3" outright; that overstated it and would have blocked a legitimate simplification, so it is corrected here rather than quietly narrowed. Both seams already carry the division in prose (`include/vt/backend.h:90-93` and `include/vllm/platforms/interface.h:61-69`); it is restated here because #1126 is the record a reader of that issue will act on, and as filed it points the other way. What #1126 owes is its steps 1 and 2 together — the override AND the Gemma4 measurement, with [#1205](https://github.com/mudler/vllm.cpp/issues/1205) ahead of both — never step 1 alone, and step 3 only in the narrowed form above. |
| **The device-expert LRU's slot cap makes its own eviction opt-in inert.** `MakeRoom` tests `slots.size() >= kMaxSlots` (`gemma4_moe.cpp:498`) BEFORE the eviction loop (`:499-500`), and `EvictOne` (`:457`, the device LRU's — a host-cache namesake sits at `:275`) is the only thing that SHRINKS `slots`. The one other statement that touches its size, `slots.clear()` in `DevExpertLru::Note` (`:522`), is a device-index RESET rather than an eviction: it drops bookkeeping when `dev != d.q.device.index` and frees nothing, and it is unreachable in a single-device process. It is named here so the next reader does not conclude the #1197 sweep missed it. So once 24 slots are resident `VT_GEMMA4_EXPERT_EVICT=1` never runs again and the cache degrades permanently to fill-only. It binds only when `24 * expert_bytes < BudgetBytes()`, so it is condition-dependent and silent either way. Tracked as [#1197](https://github.com/mudler/vllm.cpp/issues/1197). | Filed, not fixed, and for the same reason as the row above rather than for effort: the one-line repair wakes more `hipFree` under load, which the surrounding comments say has been observed as a permanent `kfd_wait` hang with the GPU idle and no decode tokens. The current ordering may well be deliberate belt-and-braces. Deciding that needs the dual-RDNA4 box `.agents/specs/gemma4-rocm-fp8-moe.md` describes; this host has neither a ROCm nor a CUDA device. It closes when the cap moves after the eviction loop and a run stays hang-free, or when the comment says the cap is by design — one of the two, not silence. |
| **`vt::MatmulBTAlphaBeta` is ROCm-only and has no CUDA implementation at all, which is what #1126 step 1 is actually blocked on.** `src/vt/fused_ops.cpp:117` dispatches to `rocm::MatmulBTAlphaBetaRocm` (`src/vt/rocm/rocm_matmul_hipblaslt.hip:516`) under `#if defined(VLLM_CPP_HIP)` and `q.device.type == kROCM` — `src/vt/fused_ops.cpp:111-112` is the signature, not the dispatch — and every other device falls through to the refusal at `src/vt/fused_ops.cpp:152`. There is no CUDA, Vulkan, Metal or CPU arm. The full chain from the missing `DeviceMemoryInfo` override to that refusal is traced in the first row above. Tracked as [#1205](https://github.com/mudler/vllm.cpp/issues/1205). | The REFUSAL is fixed in flow, because a bare `std::runtime_error` reading "ROCm-only in this build" does not satisfy the standing rule that an unimplemented arm refuses with a message NAMING the missing part: a caller who hits it on CUDA cannot tell a missing kernel from a missing build flag. It now names the device that asked, names the one arm that exists, and names the issue (`:152`), and a kROCM queue — which reaches the same line in a build configured without `-DVLLM_CPP_HIP` — gets a DIFFERENT message naming the absent build flag (`:138`), because for that caller the kernel exists and telling them to write one would send them to fix the wrong thing. `tests/vt/test_gemma4_rocm_fp8_seams.cpp` gates both messages on a posed CUDA queue, on `kCPU`/`kVULKAN`/`kMETAL`, and on kROCM — mutation-proven by restoring the old message (RED), by deleting the refusal outright (RED), and by deleting the kROCM branch so that case falls to the generic message (RED). **Say plainly what that message change does and does not pin: a contract in a unit test, not observable behaviour.** The throw is unreachable off ROCm in any shipped configuration, so no production run can print either string today; what the test fixes is what a CUDA implementation has to satisfy when someone writes one. **The reachable half of this row is the GUARD.** `EnsureGemma4Fp8ExpertOnDevice` refuses at `gemma4_moe.cpp:571` when `vt::HasMatmulBTAlphaBeta(d.q)` is false, BEFORE the upload rather than after it, which converts the mid-decode exception traced above into the host fallback that was already sitting in the `else` at `:1515-1521`: slower, two extra BF16 roundings per expert, and correct. The predicate (`include/vt/fused_ops.h`, defined `src/vt/fused_ops.cpp:102-109`) is the same condition the dispatch at `:117` uses rather than a second copy of it, so the two cannot drift and writing the CUDA kernel wakes the device arm with no edit at the call site. It is gated by `tests/vllm/models/test_gemma4_moe_device_arm_guard.cpp`, which enters through `vllm::RunGemma4Moe` — the production layer entry `src/vllm/model_executor/models/gemma4.cpp:634` calls — and decorates the registered CPU backend so `DeviceMemoryInfo` ANSWERS, which is the post-#1126 state and the only state in which the guard binds at all. Deleting the guard makes that test RED with the exact `no implementation for device 'cpu'` throw; forcing `HasMatmulBTAlphaBeta` to `true` makes it RED too. A test that constructed the `Dev` or the LRU by hand would have stayed green under both. The IMPLEMENTATION stays owed and is what [#1205](https://github.com/mudler/vllm.cpp/issues/1205) tracks. It is not written here: a `beta`-accumulating BT GEMM on cuBLASLt is a kernel with its own correctness gate, the `DeviceMemoryInfo` row's point (1) above says there is no checkpoint to exercise it on, and this host has neither a ROCm nor a CUDA device to measure either arm. |
| **`model_loader.cpp` is cited by absolute line number from 109 sites in 45 files, and this row's change moved them.** Measured between `e7d0a1f7c` and the repaired head: 203 moved line references over 109 citing sites, 10 unmoved. The file is ~1640 lines and almost every engine and model row edits it, so any edit near its top invalidates citations in files the editing change never opens. | Not swept here, deliberately, and the reason is not effort: several of the 109 were ALREADY stale (`model-matrix.md:197` cites `:184-223` as the "live loader"; line 184 at `e7d0a1f7c` is `static const bool once = [] {`), and rewriting all of them from the current tree would launder pre-existing debt into a clean-looking record. What IS fixed here is the two anchors this change authored itself, checked against the final tree. Tracked as [#1143](https://github.com/mudler/vllm.cpp/issues/1143), which lists the three candidate fixes; it needs a row of its own and is parked here because this row is what measured it. |
| **The budget knob is an environment variable, not a config key.** `VT_DEVICE_WEIGHT_BUDGET_BYTES`. | CLOSED. It waited for `ENG-RESIDENCY-CONFIG` ([#1110](https://github.com/mudler/vllm.cpp/issues/1110), PR #1119) to land the `vllm_cpp` namespace inside `--offload-config`, because landing a second, competing config surface while that one was unmerged would have created the conflict both changes then had to resolve. `ENG-RESIDENCY-CONFIG` W2 then added `vllm_cpp.device_fit.weight_budget_bytes`, and `DeviceWeightBudgetBytes` now resolves environment variable > config > device probe. `0` still suppresses the refusal from either input. [#1127](https://github.com/mudler/vllm.cpp/issues/1127); the key is specified in [`weight-residency-config.md`](weight-residency-config.md). |
| ~~**`EnsureGemma4Fp8NativeOnDevice` has the same missing-arm shape and no guard, and it is the DEFAULT arm.**~~ **CLOSED**, 2026-09-03, by [`gemma4-fp8-native-arm-guard.md`](gemma4-fp8-native-arm-guard.md) / [#2623](https://github.com/mudler/vllm.cpp/issues/2623) (this row's own tracking issue #1218 no longer resolves on the forge, which is why the successor carries a new number). The twin now refuses on all THREE ops it commits to, gated by `tests/vllm/models/test_gemma4_moe_fp8_native_arm_guard.cpp`. **This row's REASONING was right and its PREMISE was dead.** It called the twin latent because "its `MakeRoom` also needs `Backend::DeviceMemoryInfo`" and CUDA had no override; `src/vt/cuda/cuda_backend.cu:93` has had one since PERF-QWEN35-STAGE-WEIGHTS, so every admission term passed by default and the arm was LIVE, not latent -- the first consumer threw out of a decode step. The false comment this row's premise rested on is corrected at `gemma4_moe.cpp` `DevExpertLru::FreeBytes` in the same change. | **The refusal of the one-line repair was correct and is now measured.** This cell argued that "reusing `HasMatmulBTAlphaBeta` there would be a guard naming the wrong arm" and that an honest guard needs a predicate per op. Mutation M7 proves it: with the guard deleted the layer throws `vt::MatmulBTFp8Channel`, and with two of the three call sites additionally disabled it throws `vt::DequantFp8ChannelBf16` -- two different ops in one test, NEITHER of them the one the sibling's predicate names. `vt::HasMatmulBTFp8Channel` and `vt::HasDequantFp8ChannelBf16` were added for it, each dispatched on by the op it describes so the pair cannot drift. What is NOT proven, and is carried forward as owed in the new spec: the three terms are one expression today, so deleting any single term is green and no mutation here separates them. |
| **A production-entered gate for the guard exists; a production-entered gate for the REFUSAL MESSAGE does not, and cannot be built here.** `test_gemma4_moe_device_arm_guard.cpp` drives `vllm::RunGemma4Moe`, so the guard is measured as a capability. The message itself is only reachable when the guard is absent, which is precisely what that test forbids, so the message's own gate is a unit contract on a posed `vt::Queue`. | This is a property of the fix, not a gap in the test. A refusal that a correct program never reaches has no production path by construction; the alternative would be to leave the hazard unguarded so the string could be observed. Naming it here so no later reader reads the seams suite as a reachability proof. Closed when a CUDA `MatmulBTAlphaBeta` lands under [#1205](https://github.com/mudler/vllm.cpp/issues/1205) and the message stops being the answer at all. |
| **Router-lookahead prefetch for offloaded experts.** `ISSUE-GH-834`: W3 copies router identifiers device to host and waits once per MoE layer, a synchronous hop a lookahead prefetch would hide. No shipped arm owns it. | The issue predates the cutover and carries `Row: -`. The hop is measured, not speculative, and hiding it needs a second in-flight router read the current step boundary does not have — a design increment, not a repair. |
| **GPUDirect Storage (GDS / cuFile) for weight reads.** `ISSUE-GH-835`: W2's `O_DIRECT` pool bypasses the page cache and still stages every expert through host memory; GDS would land it directly in VRAM. | The issue predates the cutover and carries `Row: -`. The value differs by host and neither of the two existing GDS mentions covers weights, so the row must measure both paths before it claims a number; that measurement is the entry condition. |

## Risks/decisions

| Risk / decision | Call |
|---|---|
| Product call: is a below-gate-throughput capacity mode in scope? | YES per user direction (this spike was user-directed); it is surpass-track, default-off, and G5 protects the gate paths. Only genuine product call in here — everything behavioral follows ds4's proven design + our math |
| Per-layer sync adds ~40 waits/token | Phase 1 accepts the structural cost but does not assume margin; W0 measures c1 and W3 has its own checkpoint. Any resident/missing overlap is W6 and requires a fresh spike, not an in-place optimization promise |
| Uniform-routing h≈f may undershoot G3 | W5 measures h(f), but G3 stays fixed at f=0.5/12 tok/s. A miss is an open gap; changing the fraction is a new recorded gate, never silent rebasing |
| Expert bank = second copy of expert bytes on disk (~17 GiB) | Accepted: one-time build, keyed+versioned, evictable file; alternative (per-miss repack kernel) taxes every miss forever |
| GGUF checkpoints (APEX 35B, and every Qwen3.8-class checkpoint that exists) not covered by W1-W6 | Still out of scope for W1-W6, but the stated blocker is GONE: the slicer landed 2026-07-22. What remains is the residency policy, and the GGUF lane needs NO bank at all (see the reconciliation note) |
| Target checkpoint carries an encoding no UPSTREAM oracle defines (`UD-Q1_0`, ggml type 66) | First refused, then admitted on developer direction (15 August 2026) by pinning the fork as `llama-cpp-unsloth` at `36fe8e1cc`. The objection was policy, not feasibility, and a recorded pin answers it. The cost is honest and recorded rather than hidden: the pin is a BRANCH, which can be rebased under its own name, so the ported grid carries a digest seal, and `gateable = no` with issue #933 owing the build-and-run measurement. The oracle is scoped to the sub-IQ1_S encodings only and never outranks vLLM or upstream llama.cpp |
| IQ1_S at 1.5625 bpw may cost accuracy versus the 1.1875 bpw quant originally chosen | Accepted, and it moves the other way too: IQ1_S is the HIGHER-fidelity encoding of the two. The cost is 473 GiB rather than 370 GiB on disk, which the 3.0 TB free on dgx.casa absorbs. Quality is not asserted here; the row's gate is token output against an oracle, and any accuracy claim needs its own measurement |
| One encoding is 96.92 % of the target model | This is why W9 blocks W11 and why no partial-decode fallback is offered. A model that expands its experts to bf16 to avoid IQ1_S would need multiple TB of memory, so "unsupported encoding" here means refusal by name, never a silent widening |
| tmpfs "tier" temptation on GB10 | Rejected with reasons (§Scope verdict): tmpfs is the same unified memory; documented so it is not re-proposed |
| CUDA graphs vs data-dependent miss handling | Phase 1 explicitly disables graphs. Current Marlin has no address table; graph compatibility would require a separately spiked kernel/dispatch change after W3 profiling |
| Full-layer prefill with C<E | A single unmodified Marlin launch cannot address experts absent from slots. W4 uses exact chunk filters + scatter accumulation and proves every routed pair once; it may not allocate a hidden E-sized buffer |
| Host-copy trap on GB10 | Streaming loader must never materialize the 16.88 GiB expert `OwnedTensor` vectors. G2 inspects representation and whole-system memory, not device allocation alone |
