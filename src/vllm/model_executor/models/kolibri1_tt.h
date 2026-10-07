// Kolibri-1 — Tenstorrent (Blackhole P150) weight-staging plan (wave A).
//
// (MODEL-TEXT-kolibri-1-tenstorrent, spec .agents/specs/kolibri-tt.md,
// issue ISSUE-LOCAL-01M49K2SN7E3EH60K2FK7T88JX.) The registry entry stays
// the single landed REGISTER_VLLM_MODEL in kolibri1_registry.cpp — the
// registry is device-agnostic and the TT forward dispatch is a later wave.
// This TU owns the DEVICE-FREE half of the TT port: the per-tensor staging
// decisions over the landed Kolibri1Weights, the byte accounting against a
// P150's 32 GiB, and the refusals the plan must raise before any device is
// touched.
//
// FP8 DECISION (spec §FP8): the fp8-block projections stage NATIVE
// FP8_E4M3 — bytes verbatim from Fp8BlockWeight::packed, row-major (the
// tt-metal pin constrains FP8_E4M3 to RM at tensor creation,
// ttnn/core/tensor/py_to_tt_tensor.cpp:58-59) — and the f32 scale grids
// stay beside them. Blackhole LLK support: data_format_derive.h:81,
// llk_pack_tile_api.h:75, tensix_types.h:230. Dequant-at-load is rejected:
// it doubles the staged bytes and the P150 consumes fp8 natively.
//
// THE 384-EXPERT WALL (spec §math): the routed experts alone are
// 50 × 384 × 3,933,120 B ≈ 70.33 GiB and the whole model ≈ 73.6 GiB —
// a single 32 GiB P150 cannot hold it. PlanKolibri1TTStaging computes the
// accounting per tensor and REFUSES a single-device full-model plan by
// name with the byte deficit; the mesh/expert-parallel layout is owed to
// the device wave and is deliberately not decided here.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/kolibri1.h"
#include "vllm/model_executor/models/kolibri1_weights.h"

namespace vllm {

// Device dtype of one staged tensor.
enum class Kolibri1TTDType {
  kFp8E4M3,  // native: fp8 bytes verbatim, f32 scale grid beside
  kBf16,     // router gate, norms, embed, lm_head (bf16 in the checkpoint)
  kF32,      // the router's e_score_correction_bias (widened at load)
};

// One staged tensor with its device dtype and on-device byte cost
// (weights only; scale grids are accounted separately in the plan totals).
struct Kolibri1TTTensor {
  std::string name;
  Kolibri1TTDType dtype;
  int64_t rows = 0;
  int64_t cols = 0;
  int64_t bytes = 0;  // device bytes for the packed operand
  int64_t scale_bytes = 0;  // f32 scale grid bytes (0 for bf16 tensors)
};

struct Kolibri1TTStagingPlan {
  Kolibri1Params params;
  std::vector<Kolibri1TTTensor> tensors;

  int64_t total_bytes = 0;        // packed operands + scale grids
  int64_t fp8_bytes = 0;          // subset staged FP8_E4M3
  int64_t bf16_bytes = 0;         // subset staged BF16
  int64_t tensor_count = 0;

  // The hybrid geometry this plan asserts: 10 full-attention (RNoPE) and
  // 40 sliding-window layers, window 513, uniform head_dim 128.
  int64_t full_layers = 0;
  int64_t swa_layers = 0;
};

struct Kolibri1TTStagingOptions {
  // P150 DRAM per chip. Overridable so tests can exercise the refusal.
  int64_t device_budget_bytes = int64_t(32) << 30;
  // 0 = single device. >0 = mesh chip count; the plan divides NOTHING
  // automatically — wave A only verifies the plan FITS the mesh total and
  // records per-chip accounting as owed (the sharding layout is the device
  // wave's decision).
  int64_t mesh_chips = 0;
};

// Plans the TT staging of fully-loaded kolibri1 weights: every fp8-block
// projection → FP8_E4M3 native (bytes from Fp8BlockWeight::packed), every
// bf16 module → BF16. Asserts the checkpoint's hybrid geometry (two KV
// groups, full-rotary head_dim 128) and refuses:
//  - a weights tree with an empty or non-fp8 expert (an incomplete load),
//  - a single-device plan whose total exceeds the device budget (the
//    refusal names the byte deficit and the owed mesh row),
//  - a mesh plan that exceeds mesh_chips × device_budget_bytes.
// Throws std::runtime_error on every refusal.
Kolibri1TTStagingPlan PlanKolibri1TTStaging(
    const Kolibri1Weights& weights,
    const Kolibri1TTStagingOptions& options = {});

// ---- Wave B1: single-P150 expert streaming (host-side policy only) ----
//
// The full routed-expert tier (50 × 384 × 3,933,120 B ≈ 70.33 GiB) cannot
// reside on one P150, so decode STREAMS experts from a host-RAM tier keyed
// on router output (spec §"Single-P150 expert streaming — design"). This
// plan records the byte policy of that mode and refuses infeasible
// operating points before any device or tier is touched. Device-free: no
// allocation, no slot store, no ttnn — those are B2/B3 (spec §"Out").

// The checkpoint geometry the streaming policy accounts over. Defaults are
// the shipped kolibri1 checkpoint's byte math (spec § byte-math table);
// every field is overridable so tests can exercise the refusals.
struct Kolibri1TTStreamingShape {
  int64_t layers = 50;
  int64_t experts = 384;
  int64_t topk = 6;
  // One routed expert, one layer: fp8 bytes + f32 scale grids
  // (3×1,310,720 + (4×20 + 4×20 + 20×4)×4 = 3,933,120).
  int64_t expert_bytes = 3933120;
  int64_t attention_bytes = 1703936000;      // 50 × 34,078,720, always resident
  int64_t shared_expert_bytes = 196656000;   // 50 × 3,933,120, always resident
  int64_t router_bytes = 98304000;           // 50 × bf16 [384,2560]
  int64_t norm_bytes = 2 << 20;              // ~2 MiB
  int64_t embed_head_bytes = 1310720000;     // bf16 [128000,2560] × 2
};

struct Kolibri1TTStreamingOptions {
  // P150 DRAM per chip.
  int64_t device_budget_bytes = int64_t(32) << 30;
  // Host-RAM tier capacity for the full routed-expert tier. The NVMe tier
  // is a pluggable leaf B2/B3 owe; a host refusal NAMES it, never spills.
  int64_t host_budget_bytes = int64_t(96) << 30;
  // KV cache bytes reserved on device before the hot set is sized.
  int64_t kv_reserve_bytes = 0;
  // Decode concurrency operating point (concurrent tokens per step).
  int64_t concurrency = 1;
  // Refuse (or warn, best-effort) when the expected per-layer touched
  // fraction of the expert space exceeds this.
  double touched_fraction_threshold = 0.25;
  // true = warn in the plan instead of refusing at high concurrency.
  bool best_effort = false;
};

struct Kolibri1TTStreamingPlan {
  // Resident non-expert components (attention, shared expert, router,
  // norms, embed + untied head).
  int64_t resident_bytes = 0;
  // device budget − resident − KV reserve.
  int64_t device_residual_bytes = 0;
  // Hot set derived from the residual: floor(residual / expert_bytes).
  int64_t hot_experts = 0;
  int64_t hot_set_bytes = 0;
  // The full routed-expert tier the host tier must hold.
  int64_t host_required_bytes = 0;
  // layers × topk × expert_bytes: the per-token miss-stream bound.
  int64_t per_token_stream_bytes = 0;
  // At 60 GB/s and 25 GB/s (derived numbers, not measurements).
  double stream_ms_low = 0.0;
  double stream_ms_high = 0.0;
  // Expected distinct-expert fraction touched per layer:
  // 1 − (1 − topk/experts)^concurrency.
  double touched_fraction_per_layer = 0.0;
  // Non-empty only in best-effort mode above the threshold.
  std::string concurrency_warning;
};

// Computes the streaming byte policy and refuses:
//  - resident components alone exceeding the device budget (names the
//    deficit),
//  - the full routed-expert tier exceeding the host budget (names the
//    deficit AND the NVMe leaf as owed — never a silent spill),
//  - a concurrency whose touched fraction exceeds the declared threshold
//    (names the (1−f) × routed-bytes per-step I/O consequence), unless
//    best_effort asks for a warning instead.
// Throws std::runtime_error on every refusal.
Kolibri1TTStreamingPlan PlanKolibri1TTStreaming(
    const Kolibri1TTStreamingShape& shape,
    const Kolibri1TTStreamingOptions& options = {});

}  // namespace vllm
