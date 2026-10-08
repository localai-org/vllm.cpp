// Kolibri-1 — Tenstorrent staging plan (wave A). See kolibri1_tt.h for the
// design, the FP8_E4M3 decision and its tt-metal-pin evidence, and the
// 384-expert byte math this accounting enforces.
#include "vllm/model_executor/models/kolibri1_tt.h"

#include <cmath>
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
  // The host source bytes (device-free bookkeeping; the B2b-i device
  // staging consumes exactly these).
  t.host = w.packed.bytes.data();
  t.scale_host = w.scale.bytes.data();
  t.scale_rows = w.scale.rank == 2 ? w.scale.shape[0] : 0;
  t.scale_cols = w.scale.rank == 2 ? w.scale.shape[1] : 0;
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
    t.host = p.bf16.bytes.data();
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

// ---- Wave B1: single-P150 expert streaming (host-side policy only) ----

Kolibri1TTStreamingPlan PlanKolibri1TTStreaming(
    const Kolibri1TTStreamingShape& shape,
    const Kolibri1TTStreamingOptions& options) {
  const std::string row = "(row MODEL-TEXT-kolibri-1-tenstorrent, spec "
                          ".agents/specs/kolibri-tt.md)";
  Kolibri1TTStreamingPlan plan;

  // Resident accounting: the non-expert components are ALWAYS resident.
  plan.resident_bytes = shape.attention_bytes + shape.shared_expert_bytes +
                        shape.router_bytes + shape.norm_bytes +
                        shape.embed_head_bytes;
  if (plan.resident_bytes > options.device_budget_bytes) {
    const int64_t deficit = plan.resident_bytes - options.device_budget_bytes;
    throw std::runtime_error(
        "kolibri1-tt: streaming resident components (attention + shared "
        "expert + router + norms + embed/head) need " +
        std::to_string(plan.resident_bytes) + " bytes but the device budget "
        "is " + std::to_string(options.device_budget_bytes) + " — deficit " +
        std::to_string(deficit) + " bytes. Expert streaming cannot start: "
        "these components have no host tier. " + row);
  }

  // Hot-set sizing from the device residual.
  plan.device_residual_bytes = options.device_budget_bytes -
                               plan.resident_bytes -
                               options.kv_reserve_bytes;
  plan.hot_experts = shape.expert_bytes > 0
                         ? plan.device_residual_bytes / shape.expert_bytes
                         : 0;
  plan.hot_set_bytes = plan.hot_experts * shape.expert_bytes;

  // Host-tier accounting: the FULL routed-expert tier must fit RAM. The
  // NVMe backing tier is a named-but-unimplemented leaf (B2/B3); a plan
  // that cannot fit RAM refuses and names it — never a silent spill.
  plan.host_required_bytes = shape.layers * shape.experts * shape.expert_bytes;
  if (plan.host_required_bytes > options.host_budget_bytes) {
    const int64_t deficit =
        plan.host_required_bytes - options.host_budget_bytes;
    throw std::runtime_error(
        "kolibri1-tt: streaming host tier needs " +
        std::to_string(plan.host_required_bytes) +
        " bytes for all " + std::to_string(shape.layers) + " × " +
        std::to_string(shape.experts) + " routed experts but the host budget "
        "is " + std::to_string(options.host_budget_bytes) + " — deficit " +
        std::to_string(deficit) + " bytes. The NVMe backing tier is owed "
        "but not implemented (pluggable leaf behind the RAM tier, " + row +
        " ## Owed); this plan refuses instead of silently spilling.");
  }

  // Per-token miss-stream bound and the RAM-bandwidth envelope (derived
  // numbers, not measurements).
  plan.per_token_stream_bytes = shape.layers * shape.topk * shape.expert_bytes;
  constexpr double kGb = 1e9;
  plan.stream_ms_low =
      double(plan.per_token_stream_bytes) / (60.0 * kGb) * 1000.0;   // 60 GB/s
  plan.stream_ms_high =
      double(plan.per_token_stream_bytes) / (25.0 * kGb) * 1000.0;   // 25 GB/s

  // Concurrency policy: the expert-streaming.md high-concurrency verdict,
  // mirrored. Expected distinct experts touched per layer by `concurrency`
  // tokens each drawing topk of `experts`:
  // 1 − (1 − topk/experts)^concurrency.
  const double p = double(shape.topk) / double(shape.experts);
  const double f = 1.0 - std::pow(1.0 - p, double(options.concurrency));
  plan.touched_fraction_per_layer = f;
  const int64_t per_step_io =
      int64_t((1.0 - f) * double(plan.host_required_bytes));
  const std::string verdict =
      "kolibri1-tt: concurrency " + std::to_string(options.concurrency) +
      " touches an expected fraction " + std::to_string(f) +
      " of the expert space per layer (threshold " +
      std::to_string(options.touched_fraction_threshold) +
      "); per-step stream I/O approaches (1−f) × routed tier = " +
      std::to_string(per_step_io) +
      " bytes regardless of reordering. Reduce concurrency or raise the "
      "hot set. " + row;
  if (f > options.touched_fraction_threshold) {
    if (options.best_effort) {
      plan.concurrency_warning = verdict;
    } else {
      throw std::runtime_error(verdict);
    }
  }
  return plan;
}

// ---- Wave B2a: expert slot store policy, MoE dispatch, reset lane ----

Kolibri1TTExpertSlotPolicy PlanKolibri1TTExpertSlotPolicy(
      const Kolibri1TTSlotPolicyOptions& options) {
    const std::string row = "(row MODEL-TEXT-kolibri-1-tenstorrent, spec "
                            ".agents/specs/kolibri-tt.md)";
    const int64_t max_hot =
        options.expert_bytes > 0
            ? options.device_residual_bytes / options.expert_bytes
            : 0;
    if (options.requested_hot_experts > max_hot) {
      const int64_t requested_bytes = options.requested_hot_experts *
                                      options.expert_bytes;
      const int64_t deficit = requested_bytes - options.device_residual_bytes;
      throw std::runtime_error(
          "kolibri1-tt: slot store hot set of " +
          std::to_string(options.requested_hot_experts) + " experts (" +
          std::to_string(requested_bytes) +
          " bytes) exceeds the device residual budget of " +
          std::to_string(options.device_residual_bytes) + " bytes — deficit " +
          std::to_string(deficit) +
          " bytes. The hot set is capped by device budget − resident − KV "
          "reserve; refuse rather than over-allocate the slot pool. " + row);
    }
    Kolibri1TTExpertSlotPolicy policy;
    policy.layers = options.layers;
    policy.experts = options.experts;
    policy.expert_bytes = options.expert_bytes;
    policy.capacity = options.requested_hot_experts;
    policy.slot_pool_bytes = policy.capacity * options.expert_bytes;
    policy.table_.assign(static_cast<size_t>(options.layers),
                         std::vector<int64_t>(
                             static_cast<size_t>(policy.capacity), -1));
    policy.stamp_.assign(static_cast<size_t>(options.layers),
                         std::vector<uint64_t>(
                             static_cast<size_t>(policy.capacity), 0));
    return policy;
}


int64_t Kolibri1TTExpertSlotPolicy::Touch(int64_t layer, int64_t expert) {
    auto& tab = table_[static_cast<size_t>(layer)];
    auto& st = stamp_[static_cast<size_t>(layer)];
    // Resident already: refresh recency, keep the slot.
    for (size_t s = 0; s < tab.size(); ++s) {
      if (tab[s] == expert) {
        st[s] = ++clock_;
        return static_cast<int64_t>(s);
      }
    }
    // Free slot first (they are allocated low-first and reused on evict).
    for (size_t s = 0; s < tab.size(); ++s) {
      if (tab[s] == -1) {
        tab[s] = expert;
        st[s] = ++clock_;
        return static_cast<int64_t>(s);
      }
    }
    // Overflow: evict the least-recently-used resident.
    size_t victim = 0;
    for (size_t s = 1; s < st.size(); ++s) {
      if (st[s] < st[victim]) victim = s;
    }
    const int64_t evicted = tab[victim];
    const int64_t slot = static_cast<int64_t>(victim);
    tab[victim] = expert;
    st[victim] = ++clock_;
    if (hook_) hook_(layer, evicted, slot);
    return slot;
}

int64_t Kolibri1TTExpertSlotPolicy::SlotFor(int64_t layer,
                                              int64_t expert) const {
    const auto& tab = table_[static_cast<size_t>(layer)];
    for (size_t s = 0; s < tab.size(); ++s) {
      if (tab[s] == expert) return static_cast<int64_t>(s);
    }
    return -1;
}

int64_t Kolibri1TTExpertSlotPolicy::ResidentCount(int64_t layer) const {
    int64_t n = 0;
    for (const int64_t e : table_[static_cast<size_t>(layer)]) {
      if (e != -1) ++n;
    }
    return n;
}

int64_t Kolibri1TTExpertSlotPolicy::SuggestSlotFor(int64_t layer) const {
    const auto& tab = table_[static_cast<size_t>(layer)];
    const auto& st = stamp_[static_cast<size_t>(layer)];
    for (size_t s = 0; s < tab.size(); ++s) {
      if (tab[s] == -1) return static_cast<int64_t>(s);
    }
    size_t victim = 0;
    for (size_t s = 1; s < st.size(); ++s) {
      if (st[s] < st[victim]) victim = s;
    }
    return tab.empty() ? -1 : static_cast<int64_t>(victim);
}

const std::vector<int64_t>& Kolibri1TTExpertSlotPolicy::TableFor(
      int64_t layer) const {
    return table_[static_cast<size_t>(layer)];
}

uint64_t Kolibri1TTExpertSlotPolicy::Fingerprint() const {
    // FNV-1a over (layer, slot, expert) for every occupied cell, in
    // layer/slot order — order-independent w.r.t. touch history,
    // sensitive to every slot→logical-expert change reachable through
    // Touch() on any layer (a pure slot permutation is not reachable:
    // Touch only reassigns the LRU slot on overflow, and identical
    // re-selection keeps the table).
    uint64_t h = 1469598103934665603ull;
    auto mix = [&h](uint64_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    for (size_t l = 0; l < table_.size(); ++l) {
      for (size_t s = 0; s < table_[l].size(); ++s) {
        if (table_[l][s] == -1) continue;
        mix(static_cast<uint64_t>(l));
        mix(static_cast<uint64_t>(s));
        mix(static_cast<uint64_t>(table_[l][s]));
      }
    }
    return h;
}

namespace {

// Fills one layer's dispatch: distinct misses, resident split, target
// slots for the fetch. Read-only over the tables.
Kolibri1TTLayerDispatch DispatchLayer(const Kolibri1TTExpertSlotPolicy& policy,
                                        int64_t layer,
                                        const std::vector<int64_t>& ids) {
    Kolibri1TTLayerDispatch d;
    d.layer = layer;
    std::vector<char> seen(static_cast<size_t>(policy.experts), 0);
    for (const int64_t e : ids) {
      if (e < 0 || e >= policy.experts) {
        throw std::runtime_error(
            "kolibri1-tt: router output expert id " + std::to_string(e) +
            " out of range for " + std::to_string(policy.experts) +
            " experts on layer " + std::to_string(layer));
      }
      if (seen[static_cast<size_t>(e)]) continue;  // repeats collapse
      seen[static_cast<size_t>(e)] = 1;
      const int64_t slot = policy.SlotFor(layer, e);
      if (slot >= 0) {
        d.resident_experts.push_back(e);
        d.resident_slots.push_back(slot);
      } else {
        d.missed_experts.push_back(e);
        d.fetch_slots.push_back(policy.SuggestSlotFor(layer));
      }
    }
    return d;
}

}  // namespace

Kolibri1TTDispatchPlan PlanKolibri1TTMoEDispatch(
      const Kolibri1TTExpertSlotPolicy& policy,
      const std::vector<std::pair<int64_t, std::vector<int64_t>>>&
          layer_requests) {
    Kolibri1TTDispatchPlan plan;
    int64_t misses = 0;
    for (const auto& [layer, ids] : layer_requests) {
      plan.layers.push_back(DispatchLayer(policy, layer, ids));
      misses += static_cast<int64_t>(plan.layers.back().missed_experts.size());
    }
    // Each miss streams one expert, one layer (fp8 + scale grids).
    plan.stream_bytes = misses * policy.expert_bytes;
    return plan;
}

Kolibri1TTDispatchPlan PlanKolibri1TTMoEDispatchAllLayers(
      const Kolibri1TTExpertSlotPolicy& policy, int64_t layers,
      const std::vector<int64_t>& topk_ids) {
    std::vector<std::pair<int64_t, std::vector<int64_t>>> requests;
    requests.reserve(static_cast<size_t>(layers));
    for (int64_t l = 0; l < layers; ++l) requests.emplace_back(l, topk_ids);
    return PlanKolibri1TTMoEDispatch(policy, requests);
}

// ---- Wave B2b-i: the dense-resident device slice ----
//
// The resident slice is every non-expert component the B1 streaming plan
// keeps on device (spec §"B2 scope — B2b addendum", slice i). The walk
// mirrors the wave-A plan's per-tensor decisions over exactly that slice
// and never touches lw.moe.experts — the routed tier is absent in B2b-i
// (streamed in B2b-ii), so the plan also runs over a weights tree whose
// experts were never loaded.

namespace {

// The resident component a planned tensor belongs to (the per-component
// accounting the B1 streaming shape commits to).
enum class ResidentComponent {
    kAttention,
    kSharedExpert,
    kRouter,
    kNorm,
    kEmbedHead,
};

void AddResident(Kolibri1TTResidentStagingPlan& plan, Kolibri1TTTensor t,
                   ResidentComponent component) {
    const int64_t total = t.bytes + t.scale_bytes;
    plan.total_bytes += total;
    ++plan.tensor_count;
    plan.tensors.push_back(std::move(t));
    const Kolibri1TTTensor& st = plan.tensors.back();

    plan.fp8_bytes += st.dtype == Kolibri1TTDType::kFp8E4M3 ? st.bytes : 0;
    plan.scale_bytes += st.scale_bytes;
    plan.bf16_bytes += st.dtype == Kolibri1TTDType::kBf16 ? st.bytes : 0;
    plan.f32_bytes += st.dtype == Kolibri1TTDType::kF32 ? st.bytes : 0;

    Kolibri1TTResidentComponentBytes& c = plan.components;
    switch (component) {
      case ResidentComponent::kAttention:
        c.attention_fp8_bytes += st.bytes;
        c.attention_scale_bytes += st.scale_bytes;
        break;
      case ResidentComponent::kSharedExpert:
        c.shared_expert_fp8_bytes += st.bytes;
        c.shared_expert_scale_bytes += st.scale_bytes;
        break;
      case ResidentComponent::kRouter:
        if (st.dtype == Kolibri1TTDType::kF32) {
          c.router_bias_bytes += st.bytes;
        } else {
          c.router_gate_bytes += st.bytes;
        }
        break;
      case ResidentComponent::kNorm:
        c.norm_bytes += st.bytes;
        break;
      case ResidentComponent::kEmbedHead:
        c.embed_head_bytes += st.bytes;
        break;
    }
    c.total_bytes = c.attention_fp8_bytes + c.attention_scale_bytes +
                    c.shared_expert_fp8_bytes + c.shared_expert_scale_bytes +
                    c.router_gate_bytes + c.router_bias_bytes + c.norm_bytes +
                    c.embed_head_bytes;
}

// One bf16 module (norm / router gate / embed / head): rows x cols with
// cols == 1 for the rank-1 norms, host bytes recorded.
Kolibri1TTTensor PlanBf16Module(const std::string& name,
                                  const OwnedTensor& t) {
    Kolibri1TTTensor out;
    out.name = name;
    out.dtype = Kolibri1TTDType::kBf16;
    out.rows = t.rank >= 1 ? t.shape[0] : 0;
    out.cols = t.rank == 2 ? t.shape[1] : 1;
    out.bytes = Bf16Bytes(t);
    out.host = t.bytes.data();
    return out;
}

// One fp8-block resident projection: the checkpoint's fp8 arm. A
// non-fp8 projection here means the loader stored something the resident
// slice's dtype decision does not cover — refuse by name rather than
// plan a projection the device staging cannot fill.
Kolibri1TTTensor PlanFp8Resident(const std::string& name,
                                   const Kolibri1Projection& p) {
    if (!p.IsFp8Block()) {
      throw std::runtime_error(
          "kolibri1-tt: resident projection '" + name +
          "' is not an fp8-block weight — the B2b-i resident slice stages "
          "attention and shared-expert projections native FP8_E4M3 (spec "
          ".agents/specs/kolibri-tt.md §FP8); a bf16 module here is refused "
          "rather than silently re-armed");
    }
    return PlanFp8(name, p.fp8_block);
}

}  // namespace

Kolibri1TTResidentStagingPlan PlanKolibri1TTResidentStaging(
      const Kolibri1Weights& weights, const Kolibri1TTStagingOptions& options) {
    const Kolibri1Params& p = weights.params;
    Kolibri1TTResidentStagingPlan plan;
    plan.params = p;

    // The mesh / expert-parallel staging layout is owed (spec ## Owed): the
    // resident slice plans a single P150, exactly the device the B2b-i
    // bring-up brings up.
    if (options.mesh_chips != 0) {
      throw std::runtime_error(
          "kolibri1-tt: the B2b-i resident slice plans a single P150; "
          "mesh_chips=" + std::to_string(options.mesh_chips) +
          " is refused — the mesh / expert-parallel staging layout is OWED "
          "(row MODEL-TEXT-kolibri-1-tenstorrent, spec "
          ".agents/specs/kolibri-tt.md ## Owed)");
    }

    // Geometry assertion: identical to the wave-A plan (the two-group hybrid
    // the CPU row landed; RNoPE on the full group).
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

    // Embed + untied lm_head + final norm (bf16).
    AddResident(plan,
                PlanBf16Module("model.embed_tokens.weight",
                               weights.embed_tokens),
                ResidentComponent::kEmbedHead);
    AddResident(plan, PlanBf16Module("lm_head.weight", weights.lm_head),
                ResidentComponent::kEmbedHead);
    AddResident(plan, PlanBf16Module("model.norm.weight", weights.final_norm),
                ResidentComponent::kNorm);

    for (size_t l = 0; l < weights.layers.size(); ++l) {
      const Kolibri1LayerWeights& lw = weights.layers[l];
      const std::string layer = "model.layers." + std::to_string(l) + ".";
      // The four sandwich norms (bf16).
      for (const OwnedTensor* norm :
           {&lw.input_layernorm, &lw.post_attn_norm,
            &lw.post_attention_layernorm, &lw.post_ffn_norm}) {
        AddResident(plan, PlanBf16Module(layer + "norm", *norm),
                    ResidentComponent::kNorm);
      }
      // Attention projections (fp8-block → FP8_E4M3 native) + per-head q/k
      // norms (bf16).
      AddResident(plan, PlanFp8Resident(layer + "self_attn.q_proj",
                                       lw.attn.q_proj),
                  ResidentComponent::kAttention);
      AddResident(plan, PlanFp8Resident(layer + "self_attn.k_proj",
                                       lw.attn.k_proj),
                  ResidentComponent::kAttention);
      AddResident(plan, PlanFp8Resident(layer + "self_attn.v_proj",
                                       lw.attn.v_proj),
                  ResidentComponent::kAttention);
      AddResident(plan, PlanFp8Resident(layer + "self_attn.o_proj",
                                       lw.attn.o_proj),
                  ResidentComponent::kAttention);
      for (const OwnedTensor* qk : {&lw.attn.q_norm, &lw.attn.k_norm}) {
        AddResident(plan, PlanBf16Module(layer + "qk_norm", *qk),
                    ResidentComponent::kNorm);
      }
      // Router gate (bf16) + e_score_correction_bias (f32, widened at load).
      AddResident(plan,
                  PlanBf16Module(layer + "mlp.gate.weight",
                                 lw.moe.router_gate),
                  ResidentComponent::kRouter);
      if (lw.moe.e_score_correction_bias.dtype != vt::DType::kF32) {
        throw std::runtime_error(
            "kolibri1-tt: e_score_correction_bias must be f32 (the loader "
            "widens it losslessly)");
      }
      {
        Kolibri1TTTensor bt;
        bt.name = layer + "moe.router.expert_bias";
        bt.dtype = Kolibri1TTDType::kF32;
        bt.rows = ShapeElems(lw.moe.e_score_correction_bias);
        bt.cols = 1;
        bt.bytes = bt.rows * 4;
        bt.host = lw.moe.e_score_correction_bias.bytes.data();
        AddResident(plan, std::move(bt), ResidentComponent::kRouter);
      }
      // The shared expert (fp8-block → FP8_E4M3 native). The ROUTED experts
      // are deliberately absent: B2b-i carries no expert tier.
      AddResident(plan,
                  PlanFp8Resident(layer + "mlp.shared_experts.gate_proj",
                                  lw.moe.shared_experts.gate_proj),
                  ResidentComponent::kSharedExpert);
      AddResident(plan,
                  PlanFp8Resident(layer + "mlp.shared_experts.up_proj",
                                  lw.moe.shared_experts.up_proj),
                  ResidentComponent::kSharedExpert);
      AddResident(plan,
                  PlanFp8Resident(layer + "mlp.shared_experts.down_proj",
                                  lw.moe.shared_experts.down_proj),
                  ResidentComponent::kSharedExpert);
    }

    if (plan.total_bytes != plan.components.total_bytes) {
      throw std::runtime_error(
          "kolibri1-tt: resident component accounting drifted from the plan "
          "total — an internal accounting defect, not a checkpoint property");
    }
    if (plan.total_bytes > options.device_budget_bytes) {
      const int64_t deficit = plan.total_bytes - options.device_budget_bytes;
      throw std::runtime_error(
          "kolibri1-tt: the B2b-i resident slice (attention + shared expert + "
          "router + norms + embed/head; the routed experts stream in B2b-ii) "
          "needs " + std::to_string(plan.total_bytes) +
          " device bytes but the single P150 budget is " +
          std::to_string(options.device_budget_bytes) + " — deficit " +
          std::to_string(deficit) +
          " bytes. These components have no host tier, so expert streaming "
          "cannot start (row MODEL-TEXT-kolibri-1-tenstorrent, spec "
          ".agents/specs/kolibri-tt.md ### B2 scope — B2b addendum, slice i).");
    }
    return plan;
}

}  // namespace vllm
