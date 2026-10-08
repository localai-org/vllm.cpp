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

### B2 scope — B2b addendum (committed before any device work)

B2b is the device-bound arm of B2 named in the B2 scope section above: the
actual TT kernels. It consumes the landed B2a scaffolding unchanged —
`Kolibri1TTExpertSlotPolicy` (`kolibri1_tt.h:203`, the slot table, LRU
eviction hook, `Fingerprint`/`ContentChangedSince` reset predicate),
`Kolibri1TTDispatchPlan` (`kolibri1_tt.h:272`, the resident/fetch split and
the per-step stream byte bound), and the B1 `Kolibri1TTStreamingPlan`
(`kolibri1_tt.h:137`, `per_token_stream_bytes`, the concurrency refusal) —
and stages resident components per the wave-A dtype decision. No B2b code
lands before this addendum, and no device work starts before the P150 card
window (§ Gates).

**Wave split inside B2b — two slices, ordered:**

- **B2b-i — dense-resident device forward, no streaming.** The resident
  non-expert set (§ byte-math table: attention 1.587 GiB, embed + untied
  head 1.221 GiB, router 0.094 GiB, shared expert 0.188 GiB, norms ~2 MiB;
  ≈ 3.09 GiB) runs entirely on device with the routed-expert tier absent:
  - Attention: the hybrid geometry as staged — 40 sliding-window layers at
    window 513 and 10 full-attention layers with full RNoPE (no rope tables
    on the full group), two-group KV per the wave-A design; per-head q/k
    RMS norms applied inside the attention path; the four sandwich norms
    (`input_layernorm`, `post_attn_norm`, `post_attention_layernorm`,
    `post_ffn_norm`) per the landed CPU row
    (`.agents/specs/kolibri-1-cpu.md`).
  - Router: bf16 `[384,2560]` gate, f32 `e_score_correction_bias`,
    sigmoid-logit-add scoring, top-6-of-384 — the CPU row's contract, with
    the f32 softmax/sigmoid compute path inherited from it. In slice i the
    router output is used only for the shared expert + a nameable
    unimplemented-routed-expert refusal; the routed path arrives in slice
    ii.
  - Embed + untied lm_head staged bf16, on-device sampling through the
    landed decode seam (`ModelRegistry::Forward`, `dense_attn::AttnBlock`)
    where the TT backend provides it.
  - No streaming: slice i stages resident components only and carries no
    slot pool. Completion condition: one greedy decode of a golden prompt
    on the card.
- **B2b-ii — streaming MoE.** The FP8_E4M3 slot buffers consume the B2a
  slot policy verbatim (capacity = the streaming plan's `hot_experts`;
  staged `Fp8BlockWeight::packed` bytes verbatim, row-major). Router
  readback → host remap (the logical-expert→slot table) → fetch list →
  fetch executor fills the slots → `Touch()`. Slot swaps join the
  reset-on-content-change lane: `ContentChangedSince`
  (`kolibri1_tt.h:233`) drives the graph reset exactly as the GDN churn
  fix (`qwen3_5.cpp` TT-GDN-SLOT-CHURN,
  ISSUE-LOCAL-01M433M0TNT8FWC6SMT4R3700W, #3404). The B1 per-token stream
  bound is asserted at runtime: a step whose `stream_bytes` exceeds
  `per_token_stream_bytes` fails loudly, never silently degrades. The
  concurrency refusal/warning from B1 is inherited unchanged.

**FP8/trace constraints (carried from § FP8 on P150, not re-derived):**
FP8_E4M3 is RM-only at tensor creation
(`ttnn/core/tensor/py_to_tt_tensor.cpp:58-59`); no flatbuffer serialization
(`ttnn/core/tensor/flatbuffer/tensor_spec_flatbuffer.cpp:63`), so warmup
re-stages per process; no `extract_shard` for readback
(`ttnn/core/tensor/tensor_impl.cpp:481-483`), so the B2b-ii router readback
and any slot-content verification go through a dtype pivot or a host-side
shadow of the staged bytes. Scale grids stay host-side and stage with the
slot (§ B2 scope).

**Tests (red-first):**

- Device-free (run on CPU, gate before any card time): the B2a planner
  contracts are unchanged and their existing cases must stay green — B2b
  adds no policy code. Any B2b host-side helper (the fetch executor's
  host half, the readback dtype pivot) gets its own manifest-driven cases
  in `tests/vllm/models/test_kolibri1_tt.cpp` before the device run.
- Device-bound (P150 lease, `rc run`/`rc hold`):
  - Token gate vs the CPU golden chains — the W3 methodology
    (`tests/vllm/models/test_kolibri1_w3.cpp` against
    `tests/vllm/models/kolibri1_goldens.json`,
    `.agents/specs/kolibri-1-cpu.md` § Now): 141/145 argmax positions with
    the 4 known flips adjudicated inside the 2.5-nat band, and **0 hard
    flips allowed** — any flip outside the band is a gate failure, not a
    re-adjudication. Slice i runs the gate with the routed-expert refusal
    active only if the goldens cannot be replayed without it; the
    full-model gate runs after slice ii.
  - The production bench anchor (the TT precedent: the recorded anchor
    recipe, process per leg, `BENCH_EXIT`/TPOT per
    `.agents/specs/tenstorrent-decode-fusion.md` § evidence) runs only
    after the token gate passes.

**Explicit gate ordering:** no device measurement — throughput, latency,
or memory — is B2b evidence until the dense-resident slice (B2b-i)
completes a greedy decode on the card. Numbers recorded before that
milestone are build or smoke output, not evidence.

**Evidence plan.** Per slice, record: the build recipe against the tt-metal
pin tree, the lease identity and window, the staged byte totals as measured
on device vs the § byte-math plan, the token-gate result (exact counts and
the per-flip nat gaps), and the bench anchor when it runs — in a
`docs/bench-evidence/kolibri1-tt-<slice>-<date>.md` file. The
aleph-alpha-inference oracle gateability attempt (`.agents/oracles/
aleph-alpha-inference.md`, `gateable = no`) rides opportunistically in a
B2b device window when the lease allows: run the pinned plugin on the GPU
host, record the result in the oracle file either way. It is a rider, not a
gate.

**Stop conditions:**

- The pin tree cannot compile the B2b device TU under
  `-DVLLM_CPP_TENSTORRENT=ON` → stop, record the build failure, leave the
  row `ACTIVE`.
- The stale `_ttnncpp.so` link blocker (the `chunk_gated_delta_rule`
  `use_mcast` symbol mismatch, `.agents/issues/BACKEND-TENSTORRENT/
  ISSUE-LOCAL-01M2NSDATJQ1YNW1PA9ZBMAAM5.md` § stale-lib64): a verified
  fresh `lib64/_ttnncpp.so` (ninja rebuild of `ttnn tt_metal` + copy) is a
  **prerequisite for any TT test binary** and is owed before the first B2b
  device run — recorded under `## Owed` below, not assumed done.
- No card window: B2b cannot start. The row stays `ACTIVE` with the B2b
  spec landed and implementation owed.

## Now

`ACTIVE` — waves A and B1 landed; B2a landed (slot policy, dispatch plan,
reset predicate in `kolibri1_tt.h/.cpp`). B2b-i first slice landed
(2026-10-08): the TT build compiles and links against the pin source via the
fresh `/tmp/pin-build` lib64; the resident non-expert slice (3,311,163,520 B
= 3.084 GiB, 753 tensors) stages on the P150 with byte-exact readback
verification of all 1103 operands, and one device op (embedding bit-exact +
layer-0 q_proj within the stated envelope) is verified against the CPU row —
evidence `docs/bench-evidence/kolibri1-tt-b2i-smoke-20261008.md`. The device
leg is green on BOTH builds: the pin build against the fresh `/tmp/pin-build`
libs (smoke `SMOKE_RC=0`, 36,880/36,880 assertions) and the non-pin build +
gdn stub. The earlier pin device-op hangs were the pin tree's STALE in-tree
`build_Release/lib64`, not the pin source or the KMD/fw pair — no pin bump is
owed (§ Owed). Slice completion (one greedy decode on the card), B2b-ii, and
the token gate / bench anchor remain owed.

## Git integration

One pull request for wave A (spec + implementation together), branched from
`origin/main`, no push in this session (the operator queues the row).

## Owed

- B2b implementation (the § B2 scope — B2b addendum): the B2b-i bring-up
  slice landed 2026-10-08 (build/link, resident staging, one verified device
  op — see `## Now`); the dense-resident device forward's completion
  condition (one greedy decode on the card), then the streaming MoE
  (B2b-ii), remain owed.
- The `_ttnncpp.so` pin rebuild (verified-fresh `lib64/_ttnncpp.so`, ninja
  `ttnn tt_metal` + copy) — a named prerequisite for every TT test binary
  before the first B2b device run (the stale-lib64 blocker; the issue
  reference lives in the addendum's stop conditions). DONE for the pin-side
  build 2026-10-08: the fresh rebuild lives at `/tmp/pin-build` (symbol
  verified with `nm -D`) and the pin vllm.cpp build links clean against it;
  the durable copy into the pin tree's own `build_Release/lib64` remains
  owed. The non-pin tree's lib64 is equally stale; its build links through
  a no-op stub object for the one GDN symbol (not exercised by this row's
  gates).
- Stale pin lib64 (NOT a pin bump): the pin tree's in-tree
  `build_Release/lib64` (2026-09-25) predates the pin source (6449cf13f7b,
  2026-09-29) and hangs/crashes device ops (dmesg ARC timeout → AER →
  recovery failed → PCI rescan). A fresh out-of-source rebuild at
  `/tmp/pin-build` links AND runs green on this KMD 2.10.1-pre / fw 19.7.1
  box (health trio rc=0; B2b-i smoke `SMOKE_RC=0`, 2026-10-08 — evidence
  `docs/bench-evidence/kolibri1-tt-b2i-smoke-20261008.md` §2). The durable
  copy of the fresh lib64 into the pin tree's own `build_Release/lib64`
  remains owed — an operator decision pending.
- Mesh / expert-parallel staging of the full 73.6 GiB fp8 model.
- Single-P150 expert streaming: B1 planner spec + implementation (design
  section above), then B2/B3. The NVMe backing tier is a pluggable leaf
  behind the RAM tier.
- Tile-layout consumption of the staged FP8_E4M3 operands (tilize inside the
  compute kernels; FP8 is RM-only at tensor creation).
- Trace/warmup design that tolerates the FP8 flatbuffer serialization gap.
- The aleph-alpha-inference oracle gateability measurement (GPU lease).
