// Kolibri-1 — Tenstorrent staging plan (wave A). See kolibri1_tt.h for the
// design, the FP8_E4M3 decision and its tt-metal-pin evidence, and the
// 384-expert byte math this accounting enforces.
#include "vllm/model_executor/models/kolibri1_tt.h"

#include <stdexcept>
#include <string>

namespace vllm {

namespace {

int64_t ShapeElems(const OwnedTensor& t) {
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
  return n;
}

int64_t Bf16Bytes(const OwnedTensor& t) {
  // bf16 modules only; anything else here means the loader stored an
  // unexpected dtype on a module the TT plan stages as BF16.
  if (t.dtype != vt::DType::kBF16) {
    throw std::runtime_error(
        "kolibri1-tt: bf16 staging module carries dtype " +
        std::to_string(static_cast<int>(t.dtype)) + " — the loader must "
        "store bf16 for router gate, norms, embed and lm_head");
  }
  return ShapeElems(t) * 2;
}

Kolibri1TTTensor PlanFp8(const std::string& name,
                         const Fp8BlockWeight& w) {
  Kolibri1TTTensor t;
  t.name = name;
  t.dtype = Kolibri1TTDType::kFp8E4M3;
  t.rows = w.n;
  t.cols = w.k;
  t.bytes = w.n * w.k;  // one fp8-e4m3 byte per element, verbatim
  t.scale_bytes = w.scale.rank == 2 ? w.scale.shape[0] * w.scale.shape[1] * 4
                                    : 0;
  if (w.scale.dtype != vt::DType::kF32) {
    throw std::runtime_error(
        "kolibri1-tt: fp8-block projection '" + name +
        "' carries a non-f32 scale grid — the loader must widen "
        "weight_scale_inv to f32");
  }
  if (t.scale_bytes == 0) {
    throw std::runtime_error(
        "kolibri1-tt: fp8-block projection '" + name +
        "' carries no f32 scale grid — an fp8 operand without its "
        "weight_scale_inv cannot be consumed on device");
  }
  return t;
}

void Add(Kolibri1TTStagingPlan& plan, Kolibri1TTTensor t) {
  plan.total_bytes += t.bytes + t.scale_bytes;
  if (t.dtype == Kolibri1TTDType::kFp8E4M3) {
    plan.fp8_bytes += t.bytes + t.scale_bytes;
  } else {
    plan.bf16_bytes += t.bytes;
  }
  ++plan.tensor_count;
  plan.tensors.push_back(std::move(t));
}

void PlanProjection(Kolibri1TTStagingPlan& plan, const std::string& name,
                    const Kolibri1Projection& p) {
  if (p.IsFp8Block()) {
    Add(plan, PlanFp8(name, p.fp8_block));
    return;
  }
  if (p.IsBf16()) {
    Kolibri1TTTensor t;
    t.name = name;
    t.dtype = Kolibri1TTDType::kBf16;
    t.rows = p.bf16.rank == 2 ? p.bf16.shape[0] : 0;
    t.cols = p.bf16.rank == 2 ? p.bf16.shape[1] : 0;
    t.bytes = Bf16Bytes(p.bf16);
    Add(plan, std::move(t));
    return;
  }
  throw std::runtime_error(
      "kolibri1-tt: projection '" + name +
      "' is empty — the TT plan plans a FULLY-loaded Kolibri1Weights tree");
}

}  // namespace

Kolibri1TTStagingPlan PlanKolibri1TTStaging(const Kolibri1Weights& weights,
                                            const Kolibri1TTStagingOptions&
                                                options) {
  const Kolibri1Params& p = weights.params;
  Kolibri1TTStagingPlan plan;
  plan.params = p;

  // Geometry assertion: the two-group hybrid the CPU row landed. RNoPE on
  // the full layers means NO rope tables stage for them at all — the TT
  // plan exploits that (rope tables, when the compute wave stages any, are
  // sliding-group-only).
  for (int64_t i = 0; i < static_cast<int64_t>(p.layer_types.size()); ++i) {
    if (p.IsSlidingLayer(i)) {
      ++plan.swa_layers;
    } else {
      ++plan.full_layers;
    }
  }
  if (plan.full_layers + plan.swa_layers != p.num_hidden_layers ||
      plan.full_layers == 0 || plan.swa_layers == 0) {
    throw std::runtime_error(
        "kolibri1-tt: hybrid geometry must have both a full-attention (RNoPE) "
        "and a sliding-window group; got " +
        std::to_string(plan.full_layers) + " full / " +
        std::to_string(plan.swa_layers) + " swa of " +
        std::to_string(p.num_hidden_layers) + " layers");
  }
  if (p.use_sliding_window && p.sliding_window <= 0 && plan.swa_layers > 0) {
    throw std::runtime_error(
        "kolibri1-tt: sliding-window layers require a positive "
        "sliding_window — got " + std::to_string(p.sliding_window) +
        " (the checkpoint pins 513; the device attention wave designs for "
        "that geometry)");
  }

  Add(plan, [&] {
    Kolibri1TTTensor t;
    t.name = "model.embed_tokens.weight";
    t.dtype = Kolibri1TTDType::kBf16;
    t.rows = weights.embed_tokens.shape[0];
    t.cols = weights.embed_tokens.shape[1];
    t.bytes = Bf16Bytes(weights.embed_tokens);
    return t;
  }());
  Add(plan, [&] {
    Kolibri1TTTensor t;
    t.name = "lm_head.weight";
    t.dtype = Kolibri1TTDType::kBf16;
    t.rows = weights.lm_head.shape[0];
    t.cols = weights.lm_head.shape[1];
    t.bytes = Bf16Bytes(weights.lm_head);
    return t;
  }());
  Add(plan, [&] {
    Kolibri1TTTensor t;
    t.name = "model.norm.weight";
    t.dtype = Kolibri1TTDType::kBf16;
    t.bytes = Bf16Bytes(weights.final_norm);
    return t;
  }());

  for (size_t l = 0; l < weights.layers.size(); ++l) {
    const Kolibri1LayerWeights& lw = weights.layers[l];
    const std::string layer =
        "model.layers." + std::to_string(l) + ".";
    for (const OwnedTensor* norm :
         {&lw.input_layernorm, &lw.post_attn_norm,
          &lw.post_attention_layernorm, &lw.post_ffn_norm}) {
      Kolibri1TTTensor t;
      t.name = layer + "norm";
      t.dtype = Kolibri1TTDType::kBf16;
      t.bytes = Bf16Bytes(*norm);
      Add(plan, std::move(t));
    }
    PlanProjection(plan, layer + "self_attn.q_proj", lw.attn.q_proj);
    PlanProjection(plan, layer + "self_attn.k_proj", lw.attn.k_proj);
    PlanProjection(plan, layer + "self_attn.v_proj", lw.attn.v_proj);
    PlanProjection(plan, layer + "self_attn.o_proj", lw.attn.o_proj);
    // Per-head q/k norms (bf16, [head_dim] per head).
    for (const OwnedTensor* qk : {&lw.attn.q_norm, &lw.attn.k_norm}) {
      Kolibri1TTTensor t;
      t.name = layer + "qk_norm";
      t.dtype = Kolibri1TTDType::kBf16;
      t.bytes = Bf16Bytes(*qk);
      Add(plan, std::move(t));
    }
    // Router gate: bf16, the modules_to_not_convert module. An fp8 router
    // is refused by the loader already; here it simply cannot be bf16-less.
    Kolibri1TTTensor rt;
    rt.name = layer + "mlp.gate.weight";
    rt.dtype = Kolibri1TTDType::kBf16;
    rt.bytes = Bf16Bytes(lw.moe.router_gate);
    Add(plan, std::move(rt));
    // The router bias: f32 (losslessly widened at load).
    if (lw.moe.e_score_correction_bias.dtype != vt::DType::kF32) {
      throw std::runtime_error(
          "kolibri1-tt: e_score_correction_bias must be f32 (the loader "
          "widens it losslessly)");
    }
    Kolibri1TTTensor bt;
    bt.name = layer + "moe.router.expert_bias";
    bt.dtype = Kolibri1TTDType::kF32;
    bt.bytes = ShapeElems(lw.moe.e_score_correction_bias) * 4;
    Add(plan, std::move(bt));
    PlanProjection(plan, layer + "mlp.shared_experts.gate_proj",
                   lw.moe.shared_experts.gate_proj);
    PlanProjection(plan, layer + "mlp.shared_experts.up_proj",
                   lw.moe.shared_experts.up_proj);
    PlanProjection(plan, layer + "mlp.shared_experts.down_proj",
                   lw.moe.shared_experts.down_proj);
    for (size_t e = 0; e < lw.moe.experts.size(); ++e) {
      const std::string expert =
          layer + "mlp.experts." + std::to_string(e) + ".";
      PlanProjection(plan, expert + "gate_proj", lw.moe.experts[e].gate_proj);
      PlanProjection(plan, expert + "up_proj", lw.moe.experts[e].up_proj);
      PlanProjection(plan, expert + "down_proj", lw.moe.experts[e].down_proj);
    }
  }

  const int64_t budget =
      options.mesh_chips > 0
          ? options.mesh_chips * options.device_budget_bytes
          : options.device_budget_bytes;
  if (plan.total_bytes > budget) {
    const int64_t deficit = plan.total_bytes - budget;
    throw std::runtime_error(
        "kolibri1-tt: staging plan needs " + std::to_string(plan.total_bytes) +
        " device bytes but the " +
        (options.mesh_chips > 0
             ? std::to_string(options.mesh_chips) + "-chip mesh"
             : "single P150") + " budget is " + std::to_string(budget) +
        " — deficit " + std::to_string(deficit) + " bytes. The 384-expert "
        "routed MoE alone is ~70.3 GiB at fp8; the mesh / expert-parallel "
        "staging layout is OWED to the device wave "
        "(row MODEL-TEXT-kolibri-1-tenstorrent, spec "
        ".agents/specs/kolibri-tt.md ## Owed), and wave A refuses to stage "
        "a plan that cannot fit.");
  }
  return plan;
}

}  // namespace vllm
