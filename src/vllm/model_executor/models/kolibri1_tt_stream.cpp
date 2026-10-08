// Kolibri-1 — Tenstorrent B2b-ii streaming MoE, the host half
// (MODEL-TEXT-kolibri-1-tenstorrent, spec .agents/specs/kolibri-tt.md
// ### B2 scope — B2b addendum, slice ii; issue
// ISSUE-LOCAL-01M4ER0E9HHM95YZYJB7T5FECN). See kolibri1_tt_stream.h.
#include "vllm/model_executor/models/kolibri1_tt_stream.h"

#include <algorithm>

#include "vt/dtype.h"  // VT_CHECK

namespace vllm {
namespace {

const std::string kRow = "(row MODEL-TEXT-kolibri-1-tenstorrent, spec "
                         ".agents/specs/kolibri-tt.md ### B2 scope — B2b "
                         "addendum, slice ii)";

}  // namespace

Kolibri1TTExpertSlotPayload Kolibri1TTRoutedExpertPayload(
    const Kolibri1Weights& weights, int64_t layer, int64_t expert) {
  VT_CHECK(layer >= 0 &&
               layer < static_cast<int64_t>(weights.layers.size()),
           "kolibri1-tt B2b-ii: layer " + std::to_string(layer) +
               " out of range " + kRow);
  const Kolibri1LayerWeights& lw =
      weights.layers[static_cast<size_t>(layer)];
  VT_CHECK(expert >= 0 &&
               expert < static_cast<int64_t>(lw.moe.experts.size()),
           "kolibri1-tt B2b-ii: expert " + std::to_string(expert) +
               " out of range on layer " + std::to_string(layer) + " " + kRow);
  const Kolibri1ExpertWeights& ew =
      lw.moe.experts[static_cast<size_t>(expert)];
  Kolibri1TTExpertSlotPayload out;
  // Concatenated gate / up / down, each verbatim row-major.
  const Kolibri1Projection* projs[3] = {&ew.gate_proj, &ew.up_proj,
                                        &ew.down_proj};
  for (const Kolibri1Projection* p : projs) {
    VT_CHECK(p->IsFp8Block(),
             "kolibri1-tt B2b-ii: routed expert " + std::to_string(expert) +
                 " on layer " + std::to_string(layer) +
                 " carries a non-fp8-block projection — the streaming slot "
                 "payload is the Fp8BlockWeight::packed bytes verbatim " +
                 kRow);
    out.packed_bytes += static_cast<int64_t>(p->fp8_block.packed.bytes.size());
    out.scale_bytes += static_cast<int64_t>(p->fp8_block.scale.bytes.size());
  }
  // The concatenated views need one contiguous range per side; the
  // loaded weights hold each projection separately, so the fetch stages
  // per projection and the payload view reports the FIRST projection's
  // base with the TOTAL byte count — the fetch executor stages
  // projection-by-projection (StageKolibri1TTSlot below) and never
  // relies on cross-projection contiguity.
  out.packed_host = ew.gate_proj.fp8_block.packed.bytes.data();
  out.scale_host =
      reinterpret_cast<const float*>(ew.gate_proj.fp8_block.scale.bytes.data());
  return out;
}

Kolibri1TTSlotPoolPlan PlanKolibri1TTSlotPool(
    const Kolibri1Weights& weights, const Kolibri1TTStreamingPlan& streaming) {
  Kolibri1TTSlotPoolPlan plan;
  plan.layers = static_cast<int64_t>(weights.layers.size());
  const int64_t experts =
      plan.layers > 0
          ? static_cast<int64_t>(weights.layers[0].moe.experts.size())
          : 0;
  plan.layers = 0;
  for (const Kolibri1LayerWeights& lw : weights.layers) {
    VT_CHECK(static_cast<int64_t>(lw.moe.experts.size()) == experts,
             "kolibri1-tt B2b-ii: every layer must carry the same routed "
             "expert count " + kRow);
    if (!lw.moe.experts.empty()) ++plan.layers;
  }
  VT_CHECK(plan.layers > 0 && experts > 0,
           "kolibri1-tt B2b-ii: no routed experts in the loaded weights " +
               kRow);

  // The per-expert byte math from the FIRST loaded (layer, expert).
  int64_t packed = 0, scales = 0;
  {
    const Kolibri1ExpertWeights& ew = weights.layers[0].moe.experts[0];
    const Kolibri1Projection* projs[3] = {&ew.gate_proj, &ew.up_proj,
                                          &ew.down_proj};
    for (const Kolibri1Projection* p : projs) {
      packed += static_cast<int64_t>(p->fp8_block.packed.bytes.size());
      scales += static_cast<int64_t>(p->fp8_block.scale.bytes.size());
    }
  }
  plan.packed_bytes_per_slot = packed;
  plan.scale_bytes_per_slot = scales;
  const int64_t expert_bytes = packed + scales;

  // Cross-check the B1 streaming plan's routed-tier math against the
  // loaded weights: host_required_bytes == layers x experts x expert_bytes.
  VT_CHECK(streaming.host_required_bytes ==
               plan.layers * experts * expert_bytes,
           "kolibri1-tt B2b-ii: the B1 streaming plan's host tier (" +
               std::to_string(streaming.host_required_bytes) +
               " bytes) does not equal the loaded routed tier (" +
               std::to_string(plan.layers * experts * expert_bytes) +
               " bytes) — the slot pool plans over the bytes it will "
               "actually stage " + kRow);

  // The per-layer residual share and the per-layer capacity: the B1
  // hot_experts divided across the layers, capped by the share. The
  // policy itself is built UNCHANGED through PlanKolibri1TTExpertSlotPolicy.
  VT_CHECK(plan.layers > 0 && streaming.device_residual_bytes > 0,
           "kolibri1-tt B2b-ii: no device residual for the slot pool " + kRow);
  const int64_t share = streaming.device_residual_bytes / plan.layers;
  const int64_t cap_by_share = expert_bytes > 0 ? share / expert_bytes : 0;
  VT_CHECK(cap_by_share >= 1,
           "kolibri1-tt B2b-ii: the device residual (" +
               std::to_string(streaming.device_residual_bytes) +
               " bytes) cannot afford ONE slot per layer (" +
               std::to_string(plan.layers) + " layers x " +
               std::to_string(expert_bytes) +
               " bytes per slot) — refusing rather than planning an "
               "unallocatable pool " + kRow);
  plan.capacity_per_layer = std::min(streaming.hot_experts, cap_by_share);
  plan.capacity_per_layer = std::max(plan.capacity_per_layer, int64_t{1});

  Kolibri1TTSlotPolicyOptions opts;
  opts.layers = plan.layers;
  opts.experts = experts;
  opts.expert_bytes = expert_bytes;
  opts.device_residual_bytes = share;
  opts.requested_hot_experts = plan.capacity_per_layer;
  plan.policy = PlanKolibri1TTExpertSlotPolicy(opts);

  plan.per_layer_pool_bytes = plan.policy.slot_pool_bytes;
  plan.total_pool_bytes = plan.per_layer_pool_bytes * plan.layers;
  plan.total_scale_bytes = plan.scale_bytes_per_slot * plan.capacity_per_layer *
                           plan.layers;
  VT_CHECK(plan.total_pool_bytes <= streaming.device_residual_bytes,
           "kolibri1-tt B2b-ii: the planned slot pool (" +
               std::to_string(plan.total_pool_bytes) +
               " bytes) exceeds the device residual (" +
               std::to_string(streaming.device_residual_bytes) + " bytes) " +
               kRow);
  return plan;
}

Kolibri1TTSlotFetchList BuildKolibri1TTSlotFetchList(
    const Kolibri1TTExpertSlotPolicy& policy,
    const Kolibri1TTLayerDispatch& dispatch,
    const Kolibri1Weights& weights,
    const Kolibri1TTSlotPoolPlan& pool,
    int64_t per_token_stream_bytes) {
  VT_CHECK(dispatch.missed_experts.size() == dispatch.fetch_slots.size(),
           "kolibri1-tt B2b-ii: dispatch plan's missed/fetch lists diverge " +
               kRow);
  // THE LOUD REFUSAL: a fetch whose stream bytes exceed the B1
  // per-token ceiling never silently degrades. The layer's stream bytes
  // are the DISTINCT misses x expert_bytes (the dispatch plan's own
  // accounting, recomputed here so the enforcement is local).
  const int64_t stream = static_cast<int64_t>(dispatch.missed_experts.size()) *
                         policy.expert_bytes;
  VT_CHECK(stream <= per_token_stream_bytes,
           "kolibri1-tt B2b-ii: layer " + std::to_string(dispatch.layer) +
               " fetch needs " + std::to_string(stream) +
               " stream bytes but the B1 per-token bound is " +
               std::to_string(per_token_stream_bytes) +
               " bytes — refusing LOUDLY instead of degrading " + kRow);

  Kolibri1TTSlotFetchList list;
  list.stream_bytes = stream;
  list.jobs.reserve(dispatch.missed_experts.size());
  for (size_t i = 0; i < dispatch.missed_experts.size(); ++i) {
    const int64_t expert = dispatch.missed_experts[i];
    const int64_t slot = dispatch.fetch_slots[i];
    Kolibri1TTSlotFetchJob job;
    job.layer = dispatch.layer;
    job.expert = expert;
    job.slot = slot;
    const Kolibri1ExpertWeights& ew =
        weights.layers[static_cast<size_t>(dispatch.layer)]
            .moe.experts[static_cast<size_t>(expert)];
    const Kolibri1Projection* projs[3] = {&ew.gate_proj, &ew.up_proj,
                                          &ew.down_proj};
    // The fetch stages per projection; the job carries the three host
    // bases implicitly through the weights and the device offset of the
    // slot base. The stage loop (device half) walks gate/up/down in
    // order, matching the payload layout (concatenated, row-major,
    // verbatim).
    VT_CHECK(projs[0]->IsFp8Block() && projs[1]->IsFp8Block() &&
                 projs[2]->IsFp8Block(),
             "kolibri1-tt B2b-ii: routed expert " + std::to_string(expert) +
                 " on layer " + std::to_string(dispatch.layer) +
                 " is not fully fp8-block " + kRow);
    job.packed_host = projs[0]->fp8_block.packed.bytes.data();
    job.scale_host = reinterpret_cast<const float*>(
        projs[0]->fp8_block.scale.bytes.data());
    job.packed_bytes = pool.packed_bytes_per_slot;
    job.scale_bytes = pool.scale_bytes_per_slot;
    job.device_offset = slot * pool.packed_bytes_per_slot;
    VT_CHECK(slot >= 0 && slot < pool.capacity_per_layer,
             "kolibri1-tt B2b-ii: fetch slot " + std::to_string(slot) +
                 " out of the per-layer capacity " + kRow);
    list.jobs.push_back(std::move(job));
  }
  (void)policy;
  return list;
}

void Kolibri1TTStreamBoundGuard::Charge(int64_t layer, int64_t bytes) {
  VT_CHECK(charged_ + bytes <= bound_,
           "kolibri1-tt B2b-ii: step stream bound exceeded at layer " +
               std::to_string(layer) + " — " +
               std::to_string(charged_ + bytes) + " charged against the B1 "
               "per-token bound of " + std::to_string(bound_) +
               " bytes. Refusing LOUDLY instead of degrading " + kRow);
  charged_ += bytes;
}

Kolibri1TTSlotShadow::Kolibri1TTSlotShadow(
    const Kolibri1TTExpertSlotPolicy& policy, int64_t packed_bytes_per_slot)
    : policy_(&policy), packed_bytes_per_slot_(packed_bytes_per_slot) {
  bytes_.assign(static_cast<size_t>(policy.layers), {});
}

Kolibri1TTSlotShadow::~Kolibri1TTSlotShadow() {
  const_cast<Kolibri1TTExpertSlotPolicy*>(policy_)->SetEvictHook(nullptr);
}

void Kolibri1TTSlotShadow::AttachEvictHook() {
  Kolibri1TTExpertSlotPolicy* p =
      const_cast<Kolibri1TTExpertSlotPolicy*>(policy_);
  p->SetEvictHook([this](int64_t layer, int64_t expert, int64_t slot) {
    (void)expert;
    Clear(layer, slot);
  });
}

void Kolibri1TTSlotShadow::Clear(int64_t layer, int64_t slot) {
  auto& per_layer = bytes_[static_cast<size_t>(layer)];
  if (static_cast<int64_t>(per_layer.size()) > slot &&
      !per_layer[static_cast<size_t>(slot)].empty()) {
    per_layer[static_cast<size_t>(slot)].clear();
    --filled_;
  }
}

void Kolibri1TTSlotShadow::Record(int64_t layer, int64_t slot,
                                  const uint8_t* packed, int64_t bytes) {
  VT_CHECK(bytes == packed_bytes_per_slot_,
           "kolibri1-tt B2b-ii: shadow record of " + std::to_string(bytes) +
               " bytes does not match the slot payload size " + kRow);
  auto& per_layer = bytes_[static_cast<size_t>(layer)];
  if (static_cast<int64_t>(per_layer.size()) <= slot) {
    per_layer.resize(static_cast<size_t>(slot) + 1);
  }
  auto& cell = per_layer[static_cast<size_t>(slot)];
  if (cell.empty()) ++filled_;
  cell.assign(packed, packed + bytes);
}

int64_t Kolibri1TTSlotShadow::VerifyReadback(int64_t layer, int64_t slot,
                                             const uint8_t* packed,
                                             int64_t bytes) const {
  VT_CHECK(Has(layer, slot),
           "kolibri1-tt B2b-ii: readback verification of an unfilled slot " +
               kRow);
  VT_CHECK(bytes == packed_bytes_per_slot_,
           "kolibri1-tt B2b-ii: readback of " + std::to_string(bytes) +
               " bytes does not match the slot payload size " + kRow);
  const auto& cell =
      bytes_[static_cast<size_t>(layer)][static_cast<size_t>(slot)];
  VT_CHECK(std::equal(cell.begin(), cell.end(), packed),
           "kolibri1-tt B2b-ii: DEVICE READBACK of layer " +
               std::to_string(layer) + " slot " + std::to_string(slot) +
               " does NOT match the host shadow byte-for-byte — the staged "
               "slot content diverged; refusing instead of computing over "
               "corrupt bytes " + kRow);
  return bytes;
}

bool Kolibri1TTSlotShadow::Has(int64_t layer, int64_t slot) const {
  const auto& per_layer = bytes_[static_cast<size_t>(layer)];
  return static_cast<int64_t>(per_layer.size()) > slot &&
         !per_layer[static_cast<size_t>(slot)].empty();
}

}  // namespace vllm
