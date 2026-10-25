// Flint on Tenstorrent (Blackhole P150) — W0 device forward.
//
// (MODEL-TEXT-FLINT device arm, spec .agents/specs/flint-tenstorrent.md,
// issue ISSUE-LOCAL-01M5RVS7Q3EFVHGC6YBVF02A9Q.)
//
// WHAT RUNS ON DEVICE, and why the rest does not (the spec names both lists):
//
//   DEVICE: the embedding lookup, every RMSNorm, and EVERY linear projection
//   — q/k/v/o, the dense MLP (layers 0-1), the router, the shared expert, the
//   routed experts, and lm_head — through vt::MatmulBT over bf16-resident
//   weights. q/k/v, the normed streams, the router and the lm_head request
//   f32 OUTPUTS: the CPU reference keeps the head stream and the router
//   logits f32 (flint.cpp's "the score path" comments), and the device arm
//   mirrors that wideness rather than silently narrowing it (AGENTS.md
//   §"Inherit vLLM defaults" — a dtype that is too wide is at least visible;
//   one that is too narrow corrupts).
//
//   HOST (each owed to the next wave, named in the spec ## Owed): the
//   per-head q/k RMSNorm, the partial-RoPE pattern, the sinks softmax with
//   the sliding window, sigmoid top-k routing with renormalisation, the MoE
//   combine, and the residual adds. The host legs re-USE the CPU reference's
//   own math verbatim so the device-vs-CPU distance isolates the device
//   matmuls, not a re-derived host formula.
//
// RESIDENCY. The routed-expert tier is 47 MoE layers x 256 experts x
// (3 x 2048x512 + 512x2048) bf16 ≈ 47.1 GiB against a 32 GiB P150 — the
// kolibri 384-expert wall (kolibri1_tt.h). The selected experts (top_k per
// token, plus the resident shared expert) STREAM from the host bytes per
// layer. W0 re-uploads every layer: correctness vehicle, memoisation owed.
//
// THE CONTEXT LIFETIME. The context borrows the backend and queue it was
// built on; the caller keeps both alive for the context's lifetime (the
// engine's prepare queue, or the test's local queue).
#include "vllm/model_executor/models/flint_tt.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/ops.h"
#include "vt/tenstorrent/tenstorrent_device.h"

namespace vllm {

namespace {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::MakeTensor;

constexpr float kNegInf = -std::numeric_limits<float>::infinity();

// The same two bf16 helpers the CPU reference uses (flint.cpp) — copied, not
// shared, because flint.cpp's live in an anonymous namespace by design.
float Bf16(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint16_t ToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t lsb = (u >> 16) & 1;
  u += 0x7fff + lsb;
  return static_cast<uint16_t>(u >> 16);
}

const uint16_t* Bf16Data(const OwnedTensor& t) {
  return reinterpret_cast<const uint16_t*>(t.bytes.data());
}

int64_t Bf16Numel(const OwnedTensor& t) {
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
  return n;
}

bool DumpOn() { return std::getenv("FLINT_TT_W0_DUMP") != nullptr; }

// NeoX-style (rotate_half) RoPE over full head_dim — the CPU reference's own
// RopeNeoxHalfSplit body (flint.cpp:292), a host leg at W0.
void RopeNeoxHalfSplit(float* t, int64_t rows, int64_t heads, int64_t dh,
                       const std::vector<int32_t>& positions, double theta) {
  const int64_t half = dh / 2;
  for (int64_t r = 0; r < rows; ++r) {
    const double pos = static_cast<double>(positions[static_cast<size_t>(r)]);
    for (int64_t h = 0; h < heads; ++h) {
      float* head = t + (r * heads + h) * dh;
      for (int64_t d2 = 0; d2 < half; ++d2) {
        const double ang =
            pos / std::pow(theta, static_cast<double>(2 * d2) / double(dh));
        const double c = std::cos(ang), s = std::sin(ang);
        const float x1 = head[d2], x2 = head[d2 + half];
        head[d2] = static_cast<float>(x1 * c - x2 * s);
        head[d2 + half] = static_cast<float>(x1 * s + x2 * c);
      }
    }
  }
}

// The sinks attention, the CPU reference's own body (flint.cpp:408-445):
// f32 scores, the per-head sink folded into the softmax denominator, the
// sliding window reaching back sliding_window-1 positions, output rounded
// to bf16. Host leg at W0.
void SinksAttention(const float* q, const float* k, const float* v,
                    const float* sinks, int64_t t, int64_t hq, int64_t hkv,
                    int64_t dh, bool is_sliding, int64_t sliding_window,
                    float scale, float* attn_out) {
  const int64_t gqa = hq / hkv;
  for (int64_t i = 0; i < t; ++i) {
    const int64_t first =
        is_sliding ? std::max<int64_t>(0, i - (sliding_window - 1)) : 0;
    for (int64_t hh = 0; hh < hq; ++hh) {
      const float* qh = &q[(i * hq + hh) * dh];
      const int64_t kvh = hh / gqa;
      float m = kNegInf;
      std::vector<float> s(static_cast<size_t>(i - first + 1));
      for (int64_t j = first; j <= i; ++j) {
        const float* kh = k + (j * hkv + kvh) * dh;
        float dot = 0.0f;
        for (int64_t d2 = 0; d2 < dh; ++d2) dot += qh[d2] * kh[d2];
        s[static_cast<size_t>(j - first)] = dot * scale;
        m = std::max(m, s[static_cast<size_t>(j - first)]);
      }
      const float sink = sinks[hh];
      double denom = std::exp(static_cast<double>(sink - m));
      for (float sv : s) denom += std::exp(static_cast<double>(sv - m));
      float* out = &attn_out[(i * hq + hh) * dh];
      for (int64_t d2 = 0; d2 < dh; ++d2) out[d2] = 0.0f;
      for (int64_t j = first; j <= i; ++j) {
        const float wgt = static_cast<float>(
            std::exp(static_cast<double>(
                s[static_cast<size_t>(j - first)] - m)) /
            denom);
        const float* vh = v + (j * hkv + kvh) * dh;
        for (int64_t d2 = 0; d2 < dh; ++d2) out[d2] += wgt * vh[d2];
      }
    }
  }
  for (int64_t i = 0; i < t * hq * dh; ++i)
    attn_out[i] = Bf16(ToBf16(attn_out[i]));
}

std::vector<float> DownloadF32(Dev d, DBuf& buf) {
  std::vector<float> host(buf.bytes() / sizeof(float));
  buf.Download(d, host.data());
  return host;
}

// Downloads a bf16 buffer widened to f32 (the host legs read bf16 device
// outputs at their exact bf16 values).
std::vector<float> DownloadBf16AsF32(Dev d, DBuf& buf) {
  std::vector<uint16_t> bits(buf.bytes() / sizeof(uint16_t));
  buf.Download(d, bits.data());
  std::vector<float> host(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) host[i] = Bf16(bits[i]);
  return host;
}

// Widens a bf16 OwnedTensor into a host f32 copy (the host attention leg's
// norm/sink scalars), once at build.
std::vector<float> Widen(const OwnedTensor& t) {
  const uint16_t* src = Bf16Data(t);
  std::vector<float> out(static_cast<size_t>(Bf16Numel(t)));
  for (size_t i = 0; i < out.size(); ++i) out[i] = Bf16(src[i]);
  return out;
}

}  // namespace

// ---- the staging plan (device-free) -----------------------------------------

FlintTTStagingPlan PlanFlintTTStaging(const FlintWeights& w,
                                      const FlintTTStagingOptions& options) {
  const FlintParams& p = w.params;
  if (p.num_hidden_layers == 0 || p.hidden_size == 0)
    throw std::runtime_error(
        "flint TT staging: empty weight tree (load first, plan second)");

  FlintTTStagingPlan plan;
  int64_t resident = 0;
  resident += Bf16Numel(w.embed_tokens) * 2;
  resident += Bf16Numel(w.lm_head) * 2;
  resident += Bf16Numel(w.final_norm) * 2;
  int64_t expert_tier = 0;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const FlintLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    if (p.IsSlidingLayer(l)) ++plan.sliding_layers;
    resident += Bf16Numel(lw.input_layernorm) * 2;
    resident += Bf16Numel(lw.post_attention_layernorm) * 2;
    resident += Bf16Numel(lw.attn.q_proj.w) * 2;
    resident += Bf16Numel(lw.attn.k_proj.w) * 2;
    resident += Bf16Numel(lw.attn.v_proj.w) * 2;
    resident += Bf16Numel(lw.attn.o_proj.w) * 2;
    if (p.IsDenseLayer(l)) {
      ++plan.dense_layers;
      resident += Bf16Numel(lw.dense_mlp.gate_proj.w) * 2;
      resident += Bf16Numel(lw.dense_mlp.up_proj.w) * 2;
      resident += Bf16Numel(lw.dense_mlp.down_proj.w) * 2;
    } else {
      ++plan.moe_layers;
      resident += Bf16Numel(lw.moe.router_gate) * 2;
      resident += Bf16Numel(lw.moe.shared_expert.gate_proj.w) * 2;
      resident += Bf16Numel(lw.moe.shared_expert.up_proj.w) * 2;
      resident += Bf16Numel(lw.moe.shared_expert.down_proj.w) * 2;
      for (const FlintExpertWeights& ex : lw.moe.experts)
        expert_tier += (Bf16Numel(ex.gate_proj.w) + Bf16Numel(ex.up_proj.w) +
                        Bf16Numel(ex.down_proj.w)) *
                       2;
    }
  }
  plan.resident_bytes = resident;
  plan.expert_tier_bytes = expert_tier;
  const int64_t inter = p.moe_intermediate_size;
  const int64_t h = p.hidden_size;
  plan.streamed_bytes_per_layer =
      p.num_experts_per_tok * 2 * (2 * h * inter + inter * h);
  plan.tensor_count =
      3 + static_cast<int64_t>(p.num_hidden_layers) * 6 +
      plan.moe_layers * 4;  // +router, +3 shared-expert projections
  if (resident > options.device_budget_bytes) {
    throw std::runtime_error(
        "flint TT staging: the resident slice (" +
        std::to_string(resident) +
        " B) exceeds the device budget (" +
        std::to_string(options.device_budget_bytes) + " B); the byte deficit is " +
        std::to_string(resident - options.device_budget_bytes) +
        " B; the resident slice is embed + head + per-layer attention "
        "projections, norms, dense MLPs, shared experts and routers. The "
        "routed-expert tier (" +
        std::to_string(expert_tier) +
        " B) STREAMS and never resides (the kolibri 384-expert wall).");
  }
  return plan;
}

// ---- the device context ------------------------------------------------------

namespace {

// One layer's resident device weights + the host copies the W0 host
// attention leg consumes (the bf16 norm/sink scalars widened once at build).
struct TtLayerWeights {
  DBuf ln1{}, ln2{}, qw{}, kw{}, vw{}, ow{};
  DBuf gate{}, up{}, down{};                     // dense layers
  DBuf router{}, sh_gate{}, sh_up{}, sh_down{};  // MoE layers
  int64_t dense_inter = 0, shared_inter = 0;
  bool has_rope = false, is_dense = false, is_sliding = false;
  std::vector<float> q_norm, k_norm, sinks;  // widened host copies
};

// One bf16 upload from raw host bytes, owned by a DBuf (its constructor
// copies, its destructor returns the pool block).
DBuf StageBf16(Dev d, const void* host_bytes, int64_t rows, int64_t cols) {
  return DBuf(d, vt::DType::kBF16, {rows, cols}, host_bytes);
}

}  // namespace

struct FlintTTDeviceContext::Impl {
  vt::Backend& b;
  vt::Queue& q;
  Dev d;
  int64_t resident = 0;
  DBuf embed{}, lm_head{}, final_norm{};
  std::vector<TtLayerWeights> layers;

  Impl(vt::Backend& backend, vt::Queue& queue)
      : b(backend), q(queue), d{backend, queue} {}
};

FlintTTDeviceContext::FlintTTDeviceContext() = default;
FlintTTDeviceContext::~FlintTTDeviceContext() = default;
FlintTTDeviceContext::FlintTTDeviceContext(FlintTTDeviceContext&&) noexcept =
    default;
FlintTTDeviceContext& FlintTTDeviceContext::operator=(
    FlintTTDeviceContext&&) noexcept = default;
int64_t FlintTTDeviceContext::resident_bytes() const {
  return impl_ ? impl_->resident : 0;
}

std::unique_ptr<FlintTTDeviceContext> BuildFlintTTDeviceContext(
    vt::Backend& backend, vt::Queue& queue, const FlintWeights& weights) {
  const FlintParams& p = weights.params;
  if (p.num_hidden_layers == 0)
    throw std::runtime_error(
        "flint TT build: empty weight tree (load first, build second)");

  auto ctx = std::make_unique<FlintTTDeviceContext>();
  ctx->impl_ = std::make_unique<FlintTTDeviceContext::Impl>(backend, queue);
  FlintTTDeviceContext::Impl& c = *ctx->impl_;
  const Dev d = c.d;

  c.embed = StageBf16(d, Bf16Data(weights.embed_tokens), p.vocab_size,
                      p.hidden_size);
  c.lm_head = StageBf16(d, Bf16Data(weights.lm_head), p.hidden_size,
                        p.vocab_size);
  // The RMSNorm weights are RANK-1 [H] (vt::RmsNorm's contract).
  c.final_norm = DBuf(d, vt::DType::kBF16, {p.hidden_size},
                      Bf16Data(weights.final_norm));
  c.resident += Bf16Numel(weights.embed_tokens) * 2;
  c.resident += Bf16Numel(weights.lm_head) * 2;
  c.resident += Bf16Numel(weights.final_norm) * 2;

  c.layers.reserve(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const FlintLayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    TtLayerWeights tl;
    tl.has_rope = p.LayerHasRope(l);
    tl.is_dense = p.IsDenseLayer(l);
    tl.is_sliding = p.IsSlidingLayer(l);
    tl.ln1 = DBuf(d, vt::DType::kBF16, {p.hidden_size},
                  Bf16Data(lw.input_layernorm));
    tl.ln2 = DBuf(d, vt::DType::kBF16, {p.hidden_size},
                  Bf16Data(lw.post_attention_layernorm));
    tl.qw = StageBf16(d, Bf16Data(lw.attn.q_proj.w), lw.attn.q_proj.w.shape[0],
                      lw.attn.q_proj.w.shape[1]);
    tl.kw = StageBf16(d, Bf16Data(lw.attn.k_proj.w), lw.attn.k_proj.w.shape[0],
                      lw.attn.k_proj.w.shape[1]);
    tl.vw = StageBf16(d, Bf16Data(lw.attn.v_proj.w), lw.attn.v_proj.w.shape[0],
                      lw.attn.v_proj.w.shape[1]);
    tl.ow = StageBf16(d, Bf16Data(lw.attn.o_proj.w), lw.attn.o_proj.w.shape[0],
                      lw.attn.o_proj.w.shape[1]);
    c.resident += Bf16Numel(lw.input_layernorm) * 2 +
                  Bf16Numel(lw.post_attention_layernorm) * 2 +
                  Bf16Numel(lw.attn.q_proj.w) * 2 +
                  Bf16Numel(lw.attn.k_proj.w) * 2 +
                  Bf16Numel(lw.attn.v_proj.w) * 2 +
                  Bf16Numel(lw.attn.o_proj.w) * 2;
    if (tl.is_dense) {
      tl.dense_inter = lw.dense_mlp.gate_proj.w.shape[1];
      tl.gate = StageBf16(d, Bf16Data(lw.dense_mlp.gate_proj.w),
                          lw.dense_mlp.gate_proj.w.shape[0], tl.dense_inter);
      tl.up = StageBf16(d, Bf16Data(lw.dense_mlp.up_proj.w),
                        lw.dense_mlp.up_proj.w.shape[0], tl.dense_inter);
      tl.down = StageBf16(d, Bf16Data(lw.dense_mlp.down_proj.w),
                          lw.dense_mlp.down_proj.w.shape[0], p.hidden_size);
      c.resident += (Bf16Numel(lw.dense_mlp.gate_proj.w) +
                     Bf16Numel(lw.dense_mlp.up_proj.w) +
                     Bf16Numel(lw.dense_mlp.down_proj.w)) * 2;
    } else {
      tl.router = StageBf16(d, Bf16Data(lw.moe.router_gate), p.hidden_size,
                            p.num_experts);
      tl.shared_inter = lw.moe.shared_expert.gate_proj.w.shape[1];
      tl.sh_gate = StageBf16(d, Bf16Data(lw.moe.shared_expert.gate_proj.w),
                             lw.moe.shared_expert.gate_proj.w.shape[0],
                             tl.shared_inter);
      tl.sh_up = StageBf16(d, Bf16Data(lw.moe.shared_expert.up_proj.w),
                           lw.moe.shared_expert.up_proj.w.shape[0],
                           tl.shared_inter);
      tl.sh_down = StageBf16(d, Bf16Data(lw.moe.shared_expert.down_proj.w),
                             lw.moe.shared_expert.down_proj.w.shape[0],
                             p.hidden_size);
      c.resident += (Bf16Numel(lw.moe.router_gate) +
                     Bf16Numel(lw.moe.shared_expert.gate_proj.w) +
                     Bf16Numel(lw.moe.shared_expert.up_proj.w) +
                     Bf16Numel(lw.moe.shared_expert.down_proj.w)) * 2;
    }
    // The host attention leg's widened scalars.
    tl.q_norm = Widen(lw.attn.q_norm);
    tl.k_norm = Widen(lw.attn.k_norm);
    tl.sinks = Widen(lw.attn.sinks);
    c.layers.push_back(std::move(tl));
  }

  // The plan must agree with what was actually staged: the same resident
  // byte count, inside the budget.
  const FlintTTStagingPlan plan = PlanFlintTTStaging(weights);
  if (plan.resident_bytes != c.resident) {
    throw std::runtime_error(
        "flint TT build: staged resident bytes (" + std::to_string(c.resident) +
        ") disagree with the plan (" + std::to_string(plan.resident_bytes) +
        "); the plan and the build must account the same slice");
  }
  return ctx;
}

// ---- the W0 device forward ----------------------------------------------------

int64_t g_flint_tt_forward_calls = 0;

int64_t FlintTTForwardCalls() { return g_flint_tt_forward_calls; }

std::vector<float> ForwardFlintTT(const std::vector<int32_t>& token_ids,
                                  const std::vector<int32_t>& positions,
                                  const FlintWeights& w,
                                  FlintTTDeviceContext& ctx) {
  ++g_flint_tt_forward_calls;
  if (!ctx.impl_)
    throw std::runtime_error(
        "flint TT forward: no device context (build it at prepare on the "
        "Tenstorrent queue)");
  const FlintParams& p = w.params;
  const int64_t t = static_cast<int64_t>(token_ids.size());
  if (t <= 0)
    throw std::runtime_error("flint TT forward: empty token_ids");
  if (positions.size() != token_ids.size())
    throw std::runtime_error(
        "flint TT forward: positions length must equal token_ids length");
  for (int64_t i = 1; i < t; ++i)
    if (positions[static_cast<size_t>(i)] !=
        positions[static_cast<size_t>(i - 1)] + 1)
      throw std::runtime_error(
          "flint TT forward: the W0 arm serves ONE contiguous sequence "
          "(the CPU reference arm's same contract)");

  const FlintTTDeviceContext::Impl& c = *ctx.impl_;
  const Dev d = c.d;
  const int64_t h = p.hidden_size;
  const int64_t hq = p.num_attention_heads;
  const int64_t hkv = p.num_key_value_heads;
  const int64_t dh = p.head_dim;
  const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

  // ---- embedding (device) → host residual stream, bf16-valued f32 ----
  const uint16_t* emb = Bf16Data(w.embed_tokens);
  std::vector<float> x(static_cast<size_t>(t) * h);
  for (int64_t i = 0; i < t; ++i) {
    const int64_t tok = token_ids[static_cast<size_t>(i)];
    if (tok < 0 || tok >= p.vocab_size)
      throw std::runtime_error("flint TT forward: token id out of range");
    for (int64_t d2 = 0; d2 < h; ++d2)
      x[static_cast<size_t>(i) * h + d2] = Bf16(emb[tok * h + d2]);
  }
  if (DumpOn()) {
    double s = 0.0;
    for (float v : x) s += v;
    std::fprintf(stderr, "[FLINT-TT] embed sum=%.6f\n", s);
  }

  DBuf x_dev(d, vt::DType::kBF16, {t, h});
  std::vector<uint16_t> x_bits;
  std::vector<float> q, k, v, attn_out(static_cast<size_t>(t) * hq * dh), ao,
      ff, normed2_host;

  for (int64_t layer = 0; layer < p.num_hidden_layers; ++layer) {
    const FlintLayerWeights& lw = w.layers[static_cast<size_t>(layer)];
    const TtLayerWeights& tl = c.layers[static_cast<size_t>(layer)];

    x_bits.resize(static_cast<size_t>(t) * h);
    for (int64_t i = 0; i < t * h; ++i)
      x_bits[static_cast<size_t>(i)] = ToBf16(x[static_cast<size_t>(i)]);
    d.b.Copy(d.q, x_dev.ptr(), x_bits.data(),
             static_cast<size_t>(t) * h * 2);
    d.b.Synchronize(d.q);

    // ---- attention: h = attn(ln1(x)) + x ----
    DBuf normed_dev(d, vt::DType::kBF16, {t, h});
    vt::RmsNorm(d.q, normed_dev.t(), x_dev.t(), tl.ln1.t(),
                vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), false});
    DBuf q_dev(d, vt::DType::kBF16, {t, hq * dh});
    DBuf k_dev(d, vt::DType::kBF16, {t, hkv * dh});
    DBuf v_dev(d, vt::DType::kBF16, {t, hkv * dh});
    vt::Matmul(d.q, q_dev.t(), normed_dev.t(), tl.qw.t());
    vt::Matmul(d.q, k_dev.t(), normed_dev.t(), tl.kw.t());
    vt::Matmul(d.q, v_dev.t(), normed_dev.t(), tl.vw.t());
    q = DownloadBf16AsF32(d, q_dev);
    k = DownloadBf16AsF32(d, k_dev);
    v = DownloadBf16AsF32(d, v_dev);

    // Host leg: per-head q/k RMSNorm BEFORE RoPE (the CPU reference's own
    // head_norm body, flint.cpp:381-392).
    {
      auto head_norm = [&](float* t4, int64_t rows,
                           const std::vector<float>& nw) {
        for (int64_t r = 0; r < rows; ++r) {
          float* head = t4 + r * dh;
          double ss = 0.0;
          for (int64_t d2 = 0; d2 < dh; ++d2) ss += double(head[d2]) * head[d2];
          const float inv = static_cast<float>(
              1.0 / std::sqrt(ss / double(dh) + p.rms_norm_eps));
          for (int64_t d2 = 0; d2 < dh; ++d2) head[d2] *= inv * nw[static_cast<size_t>(d2)];
        }
      };
      head_norm(q.data(), t * hq, tl.q_norm);
      head_norm(k.data(), t * hkv, tl.k_norm);
    }
    if (tl.has_rope)
      RopeNeoxHalfSplit(q.data(), t, hq, dh, positions, p.rope_theta);
    SinksAttention(q.data(), k.data(), v.data(), tl.sinks.data(), t, hq, hkv,
                   dh, tl.is_sliding, p.sliding_window, scale,
                   attn_out.data());

    // o_proj on device: attn_out is bf16 (the CPU reference rounds it too),
    // output f32.
    std::vector<uint16_t> ao_bits(static_cast<size_t>(t) * hq * dh);
    for (int64_t i = 0; i < t * hq * dh; ++i)
      ao_bits[static_cast<size_t>(i)] = ToBf16(attn_out[static_cast<size_t>(i)]);
    DBuf ao_dev(d, vt::DType::kBF16, {t, hq * dh}, ao_bits.data());
    DBuf aoout_dev(d, vt::DType::kBF16, {t, h});
    vt::Matmul(d.q, aoout_dev.t(), ao_dev.t(), tl.ow.t());
    ao = DownloadBf16AsF32(d, aoout_dev);
    for (int64_t i = 0; i < t * h; ++i)
      x[static_cast<size_t>(i)] =
          Bf16(ToBf16(x[static_cast<size_t>(i)] + ao[static_cast<size_t>(i)]));
    if (DumpOn() && layer == 0) {
      double s = 0.0;
      for (float v2 : x) s += v2;
      std::fprintf(stderr, "[FLINT-TT] layer 0 post-attn x sum=%.6f\n", s);
    }

    // Re-upload the residual stream: ln2 normalises the POST-attention x.
    for (int64_t i = 0; i < t * h; ++i)
      x_bits[static_cast<size_t>(i)] = ToBf16(x[static_cast<size_t>(i)]);
    d.b.Copy(d.q, x_dev.ptr(), x_bits.data(), static_cast<size_t>(t) * h * 2);
    d.b.Synchronize(d.q);

    // ---- MLP / MoE: out = mlp(ln2(h)) + h ----
    DBuf normed2_dev(d, vt::DType::kBF16, {t, h});
    vt::RmsNorm(d.q, normed2_dev.t(), x_dev.t(), tl.ln2.t(),
                vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), false});
    ff.assign(static_cast<size_t>(t) * h, 0.0f);
    if (tl.is_dense) {
      DBuf g(d, vt::DType::kBF16, {t, tl.dense_inter});
      DBuf u(d, vt::DType::kBF16, {t, tl.dense_inter});
      vt::Matmul(d.q, g.t(), normed2_dev.t(), tl.gate.t());
      vt::Matmul(d.q, u.t(), normed2_dev.t(), tl.up.t());
      DBuf a(d, vt::DType::kBF16, {t, tl.dense_inter});
      vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
      DBuf ff_dev(d, vt::DType::kBF16, {t, h});
      vt::Matmul(d.q, ff_dev.t(), a.t(), tl.down.t());
      ff = DownloadBf16AsF32(d, ff_dev);
    } else {
      // Router on device, f32 out (the score path's wideness).
      DBuf rt_dev(d, vt::DType::kF32, {t, p.num_experts});
      vt::Matmul(d.q, rt_dev.t(), normed2_dev.t(), tl.router.t());
      const std::vector<float> rlogits = DownloadF32(d, rt_dev);
      normed2_host = DownloadBf16AsF32(d, normed2_dev);

      // Per token: sigmoid → top-k → renormalise (the CPU reference's own
      // routing body, flint.cpp:491-523), then STREAM the selected experts.
      const int64_t top_k = p.num_experts_per_tok;
      const int64_t inter = p.moe_intermediate_size;
      for (int64_t i = 0; i < t; ++i) {
        std::vector<std::pair<float, int64_t>> score(
            static_cast<size_t>(p.num_experts));
        for (int64_t n = 0; n < p.num_experts; ++n) {
          const float lg =
              rlogits[static_cast<size_t>(i) * p.num_experts + n];
          score[static_cast<size_t>(n)] = {1.0f / (1.0f + std::exp(-lg)), n};
        }
        std::partial_sort(score.begin(), score.begin() + top_k, score.end(),
                          [](const auto& a, const auto& b) {
                            return a.first > b.first;
                          });
        double sum = 0.0;
        for (int64_t kk = 0; kk < top_k; ++kk)
          sum += score[static_cast<size_t>(kk)].first;

        // The token's normed row, staged bf16 (the CPU reference's expert_in
        // is its normed row AT ITS BF16-ROUNDED values, so this staging is
        // value-exact).
        std::vector<uint16_t> row_bits(static_cast<size_t>(h));
        for (int64_t d2 = 0; d2 < h; ++d2)
          row_bits[static_cast<size_t>(d2)] = ToBf16(
              normed2_host[static_cast<size_t>(i) * h + d2]);
        DBuf tok_in(d, vt::DType::kBF16, {1, h}, row_bits.data());
        std::vector<float> combined(static_cast<size_t>(h), 0.0f);
        for (int64_t kk = 0; kk < top_k; ++kk) {
          const float wgt = static_cast<float>(
              score[static_cast<size_t>(kk)].first / sum);
          const FlintExpertWeights& ex =
              lw.moe.experts[static_cast<size_t>(
                  score[static_cast<size_t>(kk)].second)];
          DBuf gw = StageBf16(d, Bf16Data(ex.gate_proj.w), h, inter);
          DBuf uw = StageBf16(d, Bf16Data(ex.up_proj.w), h, inter);
          DBuf dw = StageBf16(d, Bf16Data(ex.down_proj.w), inter, h);
          DBuf g(d, vt::DType::kBF16, {1, inter});
          DBuf u(d, vt::DType::kBF16, {1, inter});
          vt::Matmul(d.q, g.t(), tok_in.t(), gw.t());
          vt::Matmul(d.q, u.t(), tok_in.t(), uw.t());
          DBuf a(d, vt::DType::kBF16, {1, inter});
          vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
          DBuf down(d, vt::DType::kBF16, {1, h});
          vt::Matmul(d.q, down.t(), a.t(), dw.t());
          const std::vector<float> dhost = DownloadBf16AsF32(d, down);
          for (int64_t d2 = 0; d2 < h; ++d2)
            combined[static_cast<size_t>(d2)] += wgt * dhost[static_cast<size_t>(d2)];
        }
        // The UNGATED shared expert (device-resident), always added.
        {
          DBuf sg(d, vt::DType::kBF16, {1, tl.shared_inter});
          DBuf su(d, vt::DType::kBF16, {1, tl.shared_inter});
          vt::Matmul(d.q, sg.t(), tok_in.t(), tl.sh_gate.t());
          vt::Matmul(d.q, su.t(), tok_in.t(), tl.sh_up.t());
          DBuf sa(d, vt::DType::kBF16, {1, tl.shared_inter});
          vt::MoeSiluMul(d.q, sa.t(), sg.t(), su.t());
          DBuf sdown(d, vt::DType::kBF16, {1, h});
          vt::Matmul(d.q, sdown.t(), sa.t(), tl.sh_down.t());
          const std::vector<float> shost = DownloadBf16AsF32(d, sdown);
          for (int64_t d2 = 0; d2 < h; ++d2)
            combined[static_cast<size_t>(d2)] += shost[static_cast<size_t>(d2)];
        }
        for (int64_t d2 = 0; d2 < h; ++d2)
          ff[static_cast<size_t>(i) * h + d2] =
              Bf16(ToBf16(combined[static_cast<size_t>(d2)]));
      }
    }
    for (int64_t i = 0; i < t * h; ++i)
      x[static_cast<size_t>(i)] =
          Bf16(ToBf16(x[static_cast<size_t>(i)] + ff[static_cast<size_t>(i)]));
    if (DumpOn() && layer == 0) {
      double s = 0.0;
      for (float v2 : x) s += v2;
      std::fprintf(stderr, "[FLINT-TT] layer 0 post-mlp x sum=%.6f\n", s);
    }
  }

  // ---- final norm (device) + untied lm_head (device, f32 out) ----
  x_bits.resize(static_cast<size_t>(t) * h);
  for (int64_t i = 0; i < t * h; ++i)
    x_bits[static_cast<size_t>(i)] = ToBf16(x[static_cast<size_t>(i)]);
  d.b.Copy(d.q, x_dev.ptr(), x_bits.data(), static_cast<size_t>(t) * h * 2);
  d.b.Synchronize(d.q);
  DBuf fnorm_dev(d, vt::DType::kBF16, {t, h});
  vt::RmsNorm(d.q, fnorm_dev.t(), x_dev.t(), c.final_norm.t(),
              vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), false});
  DBuf logits_dev(d, vt::DType::kF32, {t, p.vocab_size});
  vt::Matmul(d.q, logits_dev.t(), fnorm_dev.t(), c.lm_head.t());
  std::vector<float> logits = DownloadF32(d, logits_dev);
  if (DumpOn()) {
    double s = 0.0;
    for (int64_t n = 0; n < p.vocab_size; ++n) s += logits[static_cast<size_t>(n)];
    std::fprintf(stderr, "[FLINT-TT] logits tok0 sum=%.6f\n", s);
  }
  return logits;
}

}  // namespace vllm
