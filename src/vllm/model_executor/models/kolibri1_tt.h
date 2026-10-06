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

}  // namespace vllm
