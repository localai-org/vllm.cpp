# Kolibri-1 — Tenstorrent (P150) port spec (`MODEL-TEXT-kolibri-1-tenstorrent`)

Row: `MODEL-TEXT-kolibri-1-tenstorrent`

Owning row: `MODEL-TEXT-kolibri-1-tenstorrent`

## Scope

Port `Kolibri1ForCausalLM` to the Tenstorrent (Blackhole P150) backend in
waves. **Wave A (this spec, device-free):** the TT weight-staging plan —
config + registry reuse from the landed CPU row (`MODEL-TEXT-kolibri-1`,
`.agents/specs/kolibri-1-cpu.md`), the FP8-block on-device dtype decision,
the per-tensor staging plan with byte accounting against the P150 budget,
and GGUF refused by name. Nothing here touches a device; every gate runs on
CPU against synthetic weights and the real checkpoint manifest
(`tests/vllm/models/kolibri1_manifest.inc`).

Not in wave A (owed, see `## Owed`): the TT forward (attention, per-head
qk-norm, sandwich norms, sigmoid-logit-add MoE dispatch, sampling), the
mesh/expert-parallel staging of the full model, the sliding-window attention
kernels, and the gateability measurement for the aleph-alpha-inference
oracle (GPU lease required; `.agents/oracles/aleph-alpha-inference.md`
records `gateable = no`).

## Upstream anchors

- Architecture: the aleph-alpha-inference plugin `kolibri1.py` (pinned in
  `.agents/oracles/aleph-alpha-inference.md`), mirrored by the CPU row.
- TT backend patterns: `src/vt/tenstorrent/tenstorrent_keepquant.cpp`
  (residency + blocked-weight machinery), the mimo_v2 registry TU pattern
  (`src/vllm/model_executor/models/mimo_v2_registry.cpp`).
- vLLM defines the model behavior; the TT device behavior is the tt-metal
  pin tree (local checkout of the pinned revision).

## FP8 on P150 — investigation result (wave A decision)

**Native FP8_E4M3 staging, not dequant-at-load.** Evidence from the tt-metal
pin tree:

- `tt_metal/api/tt-metalium/tt_backend_api_types.hpp:35` — `Fp8_e4m3 = 26`
  is a first-class `DataFormat`; `:96,141` fold it into the 8-bit-float
  handling.
- Blackhole LLK support:
  `tt_metal/hw/ckernels/blackhole/metal/llk_api/data_format_derive.h:81`
  (L1 format derivation handles `Fp8_e4m3`),
  `llk_pack_tile_api.h:75` and `llk_pack_custom_api.h:38` ("8-bit datums
  (Int8, UInt8, Fp8_e4m3, Lf8) do not require the Blackhole tilize
  workaround"), `hw/inc/internal/tt-1xx/blackhole/tensix_types.h:230`
  (`Fp8_e4m3 = 26`, encoded as Lf8 + a 5th bit).
- ttnn tensor plumbing: `ttnn/core/tensor/tensor.cpp:393` (element size 1),
  `ttnn/core/tensor/tensor_impl.cpp:267-275` (host-side FP8 → f32 widening
  path exists).

Known constraints recorded for the compute wave (not blockers for staging):

- FP8_E4M3 is ROW_MAJOR-only at tensor creation
  (`ttnn/core/tensor/py_to_tt_tensor.cpp:58-59`, "FP8_E4M3 is RM-only");
  tilized consumption happens inside the compute kernels (wave B).
- `ttnn/core/tensor/flatbuffer/tensor_spec_flatbuffer.cpp:63` — FP8_E4M3
  cannot be serialized to flatbuffer (no trace-capture serialization of the
  staged weights; warmup must re-stage per process).
- `ttnn/core/tensor/tensor_impl.cpp:481-483` — no `extract_shard` for
  FP8_E4M3 (readback of a staged shard needs a dtype pivot).

Dequant-at-load is REJECTED as the default: it doubles the staged bytes
(70.3 GiB → 140.6 GiB for the routed experts alone, against a 32 GiB chip)
and the P150 consumes fp8 natively. The scale grids stay host-side in wave A
and stage f32 with the packed operand when the compute wave needs them.

## The 384-expert staging memory math

Per routed expert per layer (fp8 bytes): `gate [512,2560]` + `up [512,2560]`
+ `down [2560,512]` = 3×1,310,720 = **3,932,160 B**; scale grids f32
(4×20 + 4×20 + 20×4)×4 = **960 B**. Total ≈ 3.75 MiB/expert.

| Component | Per layer | ×50 layers |
|---|---|---|
| 384 routed experts | 384 × 3,933,120 = 1,510,318,080 B (1.407 GiB) | **70.33 GiB** |
| shared expert | 3,932,160 B + 960 B | 187.9 MiB |
| attention (q 6144×2560, k/v 512×2560, o 2560×6144, fp8) | 34,078,720 B | 1.587 GiB |
| router gate (bf16 [384,2560]) | 1,966,080 B | 93.75 MiB |
| norms | ~40 KiB | ~2 MiB |
| embed + untied lm_head (bf16 [128000,2560] each) | — | 1.221 GiB |
| **Total** | | **≈ 73.6 GiB** |

This is the largest expert count ever staged on this backend (the previous
TT rows stage ≤ 64-expert MoEs). **A single P150 (32 GiB) cannot hold the
fp8 model.** Wave A therefore ships the planner that computes this
accounting tensor-by-tensor and REFUSES a single-device full-model stage by
name, with the byte deficit in the message. The full model needs mesh
sharding (e.g. 4 chips × 32 GiB at ~18.4 GiB/chip of routed experts under
expert-parallel, attention/embed replicated or TP-sharded) — that staging
layout is owed to the device wave and is NOT decided here.

## Design

Wave A adds ONE new model-side TU pair, mirroring the "new file per
backend/architecture" rule:

- `src/vllm/model_executor/models/kolibri1_tt.h/.cpp` —
  `Kolibri1TTStagingPlan`: per-tensor staging decisions over the landed
  `Kolibri1Weights`. Every fp8-block projection stages **native FP8_E4M3**
  (row-major, bytes verbatim from `Fp8BlockWeight::packed`); the router
  gate, norms, embed, and lm_head stage **BF16** (they are bf16 in the
  checkpoint and are not block-quantized). The plan asserts the hybrid
  geometry it was built from (two KV groups: 10 full RNoPE / 40 SWA window
  513, uniform head_dim 128) and refuses a checkpoint whose layer pattern
  differs. Options carry the device budget (default 32 GiB P150) and the
  mesh chip count (0 = single device). Full-model single-device plans are
  refused by name with the deficit.
- Registration stays the single landed `REGISTER_VLLM_MODEL` in
  `kolibri1_registry.cpp` (the registry is device-agnostic; the backend
  dispatch happens in the forward, which is a later wave). The TT TU is
  compiled into the same target and exercised by its own test.
- GGUF is refused by name (the landed CPU-registry refusal already covers
  every backend; the TT TU re-states it in its loader-facing entry so the
  refusal names the TT row).

## Tests (red-first)

- `tests/vllm/models/test_kolibri1_tt.cpp`, registered in
  `tests/CMakeLists.txt` under `if(VLLM_CPP_TENSTORRENT)` beside
  `test_tenstorrent_backend` (plain C++, no ttnn headers — the staging plan
  is host-side policy). Cases: plan dtype decisions (fp8 native per
  projection, bf16 router/norms/embed/head), byte accounting against the
  manifest numbers above, the single-device full-model refusal naming the
  deficit, mesh shard accounting, geometry assertion, GGUF refusal naming
  the row, and every `F8_E4M3` manifest tensor resolving to an FP8_E4M3
  staging entry (116,303 real checkpoint tensors, manifest-driven).
- `test_kolibri1` (CPU, 27/186 cases in this build's split) must stay green.

## Gates

1. New TT tests green (CPU-only build: they are not compiled; TT build: green
   without a card where the cases are device-independent by construction).
2. `test_kolibri1` green (CPU unchanged).
3. Record battery: `check-agent-record.py` exit 0;
   `tests/scripts/test_agent_record.py` green.
4. TT compile check: configure + build with `-DVLLM_CPP_TENSTORRENT=ON`
   against the local tt-metal pin tree; the new TU must compile.

## Evidence plan

Wave A records: this spec, the pin file:line evidence above, the byte math,
and the test outputs. Device measurements are NOT wave-A evidence.

## Stop conditions

- The tt-metal pin cannot compile the TU under `-DVLLM_CPP_TENSTORRENT=ON`
  → stop, report the build failure, leave the row `ACTIVE` with the failure
  recorded.
- The manifest numbers drift from the real index → regenerate the manifest
  from the pinned checkpoint first (separate commit), never hand-edit.

## Single-P150 expert streaming — design (developer-directed 2026-10-06)

The full model cannot reside on one P150 (§ the 384-expert staging memory
math: 70.33 GiB routed experts against 32 GiB). The directed capacity
feature is expert streaming: resident non-expert components, stream routed
experts per token keyed on router output.

Decisions recorded with the developer (2026-10-06):

- **Backing tier: host RAM first, NVMe as a pluggable leaf.** The P150 is a
  discrete card, so the host-RAM tier is meaningful (PCIe 25–60 GB/s), unlike
  the GB10 unified-memory case the parent spike
  (`expert-streaming.md` §3) had to reject. Decode streams
  50 layers × 6 experts × 3.93 MiB ≈ **1.18 GiB/token**; at 25–60 GB/s the
  raw stream bound is **20–47 ms/token**. The NVMe tier (~5 GB/s,
  ~236 ms/token) is the backing-store leaf, not the first target.
- **Device hot set: yes.** A hotness-decayed-LFU cache (the
  `ENG-EXPERT-STREAM` mechanism) sized inside the device budget. Budget:
  32 GiB − ~3.08 GiB resident (attention 1.587, embed+head 1.221, router
  0.094, shared expert 0.188, norms ~0.002, GiB) − KV cache − hot set.
- **Concurrency refusal is inherited.** At conc≥32 the per-step touched
  fraction of 384 experts approaches 1 and per-step I/O approaches
  (1−f) × 70 GiB regardless of reordering (`expert-streaming.md` verdict).
  The streaming mode must refuse or warn by name at high concurrency, never
  silently degrade.

Platform mapping:

- The **mechanism** (bank loader, hotness cache, pread pool,
  logical-expert→slot remap) is platform-neutral and is reused, not
  rewritten.
- The **device destination is TT-native**: a fixed-capacity pool of
  FP8_E4M3 expert slot buffers staged per the wave-A dtype decision
  (`packed` bytes verbatim, row-major; scale grids f32 host-side). The
  landed `DeviceExpertSlotStore` (`expert-stream-device-slots.md`) is
  CUDA-Marlin-shaped and lands unreached; it is a pattern reference, not a
  dependency.
- **Trace capture cannot cover the MoE region**: FP8_E4M3 does not
  serialize to flatbuffer (§ FP8 constraints) and streamed slot content
  changes per token. The graph reset-on-content-change mechanism landed for
  GDN state churn (issue
  `ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W`, #3404) is the direct precedent:
  expert-slot swaps join the same reset lane, and the resident
  (attention/norm/router/shared) region stays capturable.

Wave shape for this design (each wave owns its spec section before device
work):

- **B1 (host-side, device-free)**: streaming staging planner — host-tier
  byte budget, resident-fraction accounting, per-token miss bound with the
  hot set, concurrency refusal. Gates stay CPU-only like wave A.

### B1 scope (committed before implementation)

`Kolibri1TTStagingPlan` grows a streaming mode beside the existing
full-residency plans. Device-free; every gate stays CPU-only.

In:

- `Kolibri1TTStreamingPlan` (same TU pair, `kolibri1_tt.h/.cpp`): inputs
  are the device budget (default 32 GiB), the host-tier byte budget, the
  hot-set byte budget, and the concurrency operating point.
- Resident accounting: non-expert components (§ byte math: attention
  1.587 GiB, shared expert 0.188 GiB, router 0.094 GiB, norms ~2 MiB,
  embed + untied head 1.221 GiB) always resident; the planner refuses by
  name if they alone exceed the device budget.
- Hot-set sizing: the planner takes the hot-set budget as an expert count
  derived from the device residual (device budget − resident − KV reserve);
  each hot expert costs 3,933,120 B (fp8 + scales, § byte math).
- Host-tier accounting: all 384 × 50 experts must fit the host budget
  (70.33 GiB fp8 + scale grids); the planner refuses by name with the
  deficit when it does not. The NVMe tier is a named-but-unimplemented
  leaf: a plan that cannot fit RAM refuses with a message that names the
  NVMe leaf as owed, and never silently spills.
- Per-token miss bound: decode streams at most
  `layers × topk × per-expert bytes` = 1.18 GiB/token; the plan records the
  bound and the RAM-tier bandwidth envelope (25–60 GB/s ⇒ 20–47 ms/token)
  as derived numbers, not measurements.
- Concurrency refusal: at a concurrency whose expected touched-expert
  fraction per layer exceeds a declared threshold, the plan refuses (or
  warns, when the caller asks for best-effort) naming the
  (1−f) × 70.33 GiB per-step I/O consequence — the
  `expert-streaming.md` high-concurrency verdict mirrored.

Out (owed to B2/B3): any device allocation, the slot store, routing
readback, the NVMe filler, overlap. B1 is host-side policy only.

Tests (red-first, `tests/vllm/models/test_kolibri1_tt.cpp`): resident
accounting against the § byte-math table; hot-set derivation from the
device residual; host-budget refusal naming the deficit; the NVMe leaf
naming; the per-token stream bound 50 × 6 × 3,933,120 B; the concurrency
refusal at a touched fraction above threshold and the best-effort warning
below refusal. `test_kolibri1` (CPU) stays green.

Gates: the wave-A gate list unchanged (CPU-only build skips the TU; TT
build compiles it; record battery green).

- **B2**: TT device forward for resident components + streaming MoE
  dispatch through the TT slot store.
- **B3**: overlap (fetch layer *i*'s experts while layer *i−1* computes),
  hot-set seeding, measured TPOT vs the resident-deficit refusal of wave A.

### B2 scope (committed before implementation)

B2 owns two things: the **TT device forward for the resident components**
and the **TT-native expert slot store** that the streaming dispatch
fills. The device forward is device-bound work; the slot-store policy and
the dispatch plan are device-free and split off first, because the B1
lesson holds — host-side policy gates on CPU, and only the kernels need
the card.

**Device forward (B2 device-bound arm, resident components only).** The
TT forward runs what wave B1 proved resident (§ byte-math table):

- Attention: the hybrid geometry as staged — 40 sliding-window layers at
  window 513 and 10 full-attention layers with full RNoPE (no rope tables
  on the full group; § wave-A design). Per-head q/k RMS norms
  (`q_norm`/`k_norm`, bf16 `[head_dim]` per head) applied inside the
  attention path.
- Sandwich norms: the four per-layer norms the CPU row landed
  (`input_layernorm`, `post_attn_norm`, `post_attention_layernorm`,
  `post_ffn_norm`), staged bf16.
- MoE dispatch: the router runs resident (bf16 `[384,2560]` gate, f32
  `e_score_correction_bias`), sigmoid scoring with the bias added to the
  logits (sigmoid-logit-add), top-6-of-384 selection. Routed-expert
  compute consumes slots from the slot store below; the shared expert is
  resident and computed directly.
- Sampling: on-device sampling through the landed decode seam
  (`ModelRegistry::Forward`, `dense_attn::AttnBlock`) where the TT
  backend already provides it.

The device forward NEVER references a routed expert that is not resident
in a slot. A miss is a fetch (B3 overlaps it), never a host fallback.

**TT-native expert slot store (device tier of the streaming design).**
A fixed-capacity pool of FP8_E4M3 slot buffers on device, staged per the
wave-A dtype decision: `Fp8BlockWeight::packed` bytes verbatim,
row-major (FP8 is RM-only at tensor creation,
`ttnn/core/tensor/py_to_tt_tensor.cpp:58-59`); the f32 scale grids stay
host-side beside the slot table and stage with the slot when the compute
kernel needs them (§ FP8 constraints — no `extract_shard`, no
flatbuffer serialization for FP8). Each slot holds one expert, one
layer: 3,933,120 B (fp8 + scale grids, § byte math). Capacity comes from
the streaming plan's `hot_experts` (device residual / expert bytes), not
from a second accounting. A logical-expert→slot remap table
(host-side) translates router output ids to slot indices. Slot eviction
is a hotness-decayed-LFU policy hook (the `ENG-EXPERT-STREAM` mechanism);
B2 defines the policy interface and the refusal, not the replacement
heuristic tuning.

Slot swaps join the **graph reset-on-content-change lane** — the GDN
slot-churn fix (`qwen3_5.cpp` TT-GDN-SLOT-CHURN,
ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W, #3404) is the direct precedent:
captured graphs bind slot buffer content, so a slot whose logical expert
changed must flip a reset predicate; an identical re-selection must not.
The predicate is the slot-table fingerprint (slot → logical-expert
mapping, layer-scoped), not the bytes of the weights.

**Split into device-free sub-waves:**

- **B2a (device-free, CPU-only gates, this wave):**
  - `Kolibri1TTExpertSlotPolicy`: capacity = the streaming plan's hot
    expert count; the slot table (logical expert id → slot, per layer);
    LRU/hotness eviction hooks (policy only — no device allocation, no
    ttnn includes); a refuse-by-name when the declared hot set exceeds
    the device residual budget.
  - The MoE dispatch plan: given a router output (6-of-384 ids per
    layer), produce the per-layer slot-fetch list, the resident/missed
    split, and the per-token stream byte bound (misses × 3,933,120 B)
    consistent with the B1 `per_token_stream_bytes` ceiling.
  - The reset-lane predicate: a slot-table fingerprint whose change
    flips graph-reset exactly like the GDN churn fix; an identical
    re-selection leaves it stable.
  - Tests gate B2a: `tests/vllm/models/test_kolibri1_tt.cpp` cases for
    the slot policy (capacity from the streaming plan, refusal naming
    the deficit, eviction hook bookkeeping), the dispatch plan
    (fetch list, resident/missed split, byte bound), and the reset
    predicate (change flips, identical re-selection does not).
    `test_kolibri1` (CPU) stays green. Red-first in the manifest-driven
    style of the existing file.
- **B2b (device-bound, needs the P150 card, own spec addendum before any
  device work):** the actual TT kernels — attention SWA 513 + full
  RNoPE, per-head qk-norm, sandwich norms, the sigmoid-logit-add MoE
  dispatch consuming slots, sampling. Gates: device gates under the GPU
  lease, plus the parity trace against the pinned oracle. No B2b code
  lands before that addendum is committed.

Out (owed to B2b/B3): any device allocation or kernel, slot fetch
executors, overlap, hot-set seeding. B2a touches no device.

## Now

`ACTIVE` — wave A (registration/config-reuse, staging plan, loader wiring,
manifest) in review; the device forward is the next wave and owns its own
spec section before any P150 work. The single-P150 expert streaming design
above is recorded; B1 owes its spec section before implementation.

## Git integration

One pull request for wave A (spec + implementation together), branched from
`origin/main`, no push in this session (the operator queues the row).

## Owed

- The TT forward wave: device attention (SWA 513 + full RNoPE), per-head
  qk-norm, sandwich norms, sigmoid-logit-add MoE dispatch with 6-of-384
  routing on device, sampling.
- Mesh / expert-parallel staging of the full 73.6 GiB fp8 model.
- Single-P150 expert streaming: B1 planner spec + implementation (design
  section above), then B2/B3. The NVMe backing tier is a pluggable leaf
  behind the RAM tier.
- Tile-layout consumption of the staged FP8_E4M3 operands (tilize inside the
  compute kernels; FP8 is RM-only at tensor creation).
- Trace/warmup design that tolerates the FP8 flatbuffer serialization gap.
- The aleph-alpha-inference oracle gateability measurement (GPU lease).
