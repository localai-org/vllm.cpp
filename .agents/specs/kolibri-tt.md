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
| attention (q 6144×2560, k/v 512×2560, o 2560×6144, fp8) | 34,078,720 B | 1.625 GiB |
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

## Now

`ACTIVE` — wave A (registration/config-reuse, staging plan, loader wiring,
manifest) in review; the device forward is the next wave and owns its own
spec section before any P150 work.

## Git integration

One pull request for wave A (spec + implementation together), branched from
`origin/main`, no push in this session (the operator queues the row).

## Owed

- The TT forward wave: device attention (SWA 513 + full RNoPE), per-head
  qk-norm, sandwich norms, sigmoid-logit-add MoE dispatch with 6-of-384
  routing on device, sampling.
- Mesh / expert-parallel staging of the full 73.6 GiB fp8 model.
- Tile-layout consumption of the staged FP8_E4M3 operands (tilize inside the
  compute kernels; FP8 is RM-only at tensor creation).
- Trace/warmup design that tolerates the FP8 flatbuffer serialization gap.
- The aleph-alpha-inference oracle gateability measurement (GPU lease).
