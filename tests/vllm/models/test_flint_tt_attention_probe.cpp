// FLINT-P150-ATTENTION stage 1 (spec .agents/specs/flint-p150-attention.md,
// issue ISSUE-LOCAL-01M687MYRKDS21SBESBG012KN7): the SINGLE-LAYER device
// attention probe.
//
// One Flint attention layer's core — per-head QK RMSNorm (head_dim), optional
// NeoX RoPE, the per-head learnable sink, and the 513-token sliding window —
// runs on the Tenstorrent device through the shared seam ops
// (vt::RmsNorm + vt::RopeNeox + ReshapeAndCache + vt::PagedAttention with the
// row's attn_sink seeding) and is compared against the HOST SinksAttention
// shadow (the flint_tt.cpp:126 body, replicated here on identical bf16 inputs).
//
// THE CAMPAIGN SCARS SHAPE THE CHECKS (spec "Known hazards"):
//   * SDPA-computed-only-2-heads  -> EVERY head's output slice is compared
//     individually, not just the pooled output.
//   * PA-correct-but-output-wrong -> the model-level check rides on top: the
//     full-model keep-quant band gate (stage 2) must pass before "works".
//
// THE BAND IS DERIVED, NOT INVENTED (spec "Band discipline"): the W2 substrate
// record (docs/bench-evidence/flint-tt-w2-20261025/report.md §4) measures an
// operand-dtype-independent ~2.64e-2 rel_rms PER SINGLE PRODUCT on this pin.
// An attention output element is a softmax-weighted sum of bf16-exact V rows
// whose weights carry one score-dot's worth of that noise (128 products, then
// the softmax pass, then the reduction): two noise factors of the same class,
// times a 1.5 margin. Band = 3 x 2.64e-2 ~= 8.0e-2 per-head rel_rms. A head
// above 5x the band is NOT substrate noise and fails as structural.
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <random>
#include <vector>

#include <doctest/doctest.h>

#include "vllm/model_executor/models/dense_attn_block.h"
#include "vt/ops.h"

namespace {

using vllm::dense_attn::DBuf;
using vllm::dense_attn::Dev;
using vllm::dense_attn::KvSlice;
using vllm::dense_attn::Reshape;

constexpr double kSingleProduct = 2.64e-2;  // the W2 record's measured figure
constexpr double kPerHeadBand = 3.0 * kSingleProduct;  // 7.9e-2
constexpr double kStructuralBand = 5.0 * kPerHeadBand;

constexpr int64_t kHq = 32, kHkv = 4, kDh = 128;

uint16_t ToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t lsb = (u >> 16) & 1;
  u += 0x7fff + lsb;
  return static_cast<uint16_t>(u >> 16);
}

float FromBf16(uint16_t bits) {
  uint32_t u = static_cast<uint32_t>(bits) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

float Bf16Round(float f) { return FromBf16(ToBf16(f)); }

// ---- the HOST SHADOW (the flint_tt.cpp CPU-reference bodies) -----------------

void HeadNorm(float* t4, int64_t rows, const std::vector<float>& nw,
              double eps) {
  for (int64_t r = 0; r < rows; ++r) {
    float* head = t4 + r * kDh;
    double ss = 0.0;
    for (int64_t e = 0; e < kDh; ++e) ss += double(head[e]) * head[e];
    const float inv =
        static_cast<float>(1.0 / std::sqrt(ss / double(kDh) + eps));
    for (int64_t e = 0; e < kDh; ++e) head[e] *= inv * nw[static_cast<size_t>(e)];
  }
}

void RopeNeoxHalfSplit(float* t4, int64_t rows, int64_t heads,
                       const std::vector<int32_t>& positions, double theta) {
  const int64_t half = kDh / 2;
  for (int64_t r = 0; r < rows; ++r) {
    const double pos = static_cast<double>(positions[static_cast<size_t>(r)]);
    for (int64_t h = 0; h < heads; ++h) {
      float* head = t4 + (r * heads + h) * kDh;
      for (int64_t d2 = 0; d2 < half; ++d2) {
        const double ang =
            pos / std::pow(theta, static_cast<double>(2 * d2) / double(kDh));
        const double c = std::cos(ang), s = std::sin(ang);
        const float x1 = head[d2], x2 = head[d2 + half];
        head[d2] = static_cast<float>(x1 * c - x2 * s);
        head[d2 + half] = static_cast<float>(x1 * s + x2 * c);
      }
    }
  }
}

void SinksAttention(const float* q, const float* k, const float* v,
                    const float* sinks, int64_t t, bool is_sliding,
                    int64_t sliding_window, float scale, float* attn_out) {
  constexpr float kNegInf = -std::numeric_limits<float>::infinity();
  const int64_t gqa = kHq / kHkv;
  for (int64_t i = 0; i < t; ++i) {
    const int64_t first =
        is_sliding ? std::max<int64_t>(0, i - (sliding_window - 1)) : 0;
    for (int64_t hh = 0; hh < kHq; ++hh) {
      const float* qh = &q[(i * kHq + hh) * kDh];
      const int64_t kvh = hh / gqa;
      float m = kNegInf;
      std::vector<float> s(static_cast<size_t>(i - first + 1));
      for (int64_t j = first; j <= i; ++j) {
        const float* kh = k + (j * kHkv + kvh) * kDh;
        float dot = 0.0f;
        for (int64_t d2 = 0; d2 < kDh; ++d2) dot += qh[d2] * kh[d2];
        s[static_cast<size_t>(j - first)] = dot * scale;
        m = std::max(m, s[static_cast<size_t>(j - first)]);
      }
      const float sink = sinks[hh];
      double denom = std::exp(static_cast<double>(sink - m));
      for (float sv : s) denom += std::exp(static_cast<double>(sv - m));
      float* out = &attn_out[(i * kHq + hh) * kDh];
      for (int64_t d2 = 0; d2 < kDh; ++d2) out[d2] = 0.0f;
      for (int64_t j = first; j <= i; ++j) {
        const float wgt = static_cast<float>(
            std::exp(static_cast<double>(s[static_cast<size_t>(j - first)] - m)) /
            denom);
        const float* vh = v + (j * kHkv + kvh) * kDh;
        for (int64_t d2 = 0; d2 < kDh; ++d2) out[d2] += wgt * vh[d2];
      }
    }
  }
  for (int64_t i = 0; i < t * kHq * kDh; ++i)
    attn_out[i] = FromBf16(ToBf16(attn_out[i]));
}

// ---- the device layer --------------------------------------------------------

// The stage-1 device layer: per-head QK RMSNorm (vt::RmsNorm over [T*H, Dh]),
// optional vt::RopeNeox, the KV-cache write, and vt::PagedAttention with the
// sliding window and the per-head sink. Returns the bf16 [T, Hq*Dh] output.
std::vector<float> DeviceLayer(Dev& d, int64_t t, bool is_sliding,
                               int64_t sliding_window, bool has_rope,
                               double theta, double eps,
                               const std::vector<uint16_t>& q_bits,
                               const std::vector<uint16_t>& k_bits,
                               const std::vector<uint16_t>& v_bits,
                               const std::vector<float>& q_norm,
                               const std::vector<float>& k_norm,
                               const std::vector<float>& sinks,
                               const std::vector<int32_t>& positions) {
  const int64_t bs = 32;
  const int64_t nblk = (t + bs - 1) / bs;
  vllm::PagedKvCache kv{};
  kv.dtype = vt::DType::kBF16;
  kv.num_blocks = nblk;
  kv.block_size = bs;
  kv.num_kv_heads = kHkv;
  kv.head_size = kDh;

  DBuf q(d, vt::DType::kBF16, {t, kHq * kDh}, q_bits.data());
  DBuf k(d, vt::DType::kBF16, {t, kHkv * kDh}, k_bits.data());
  DBuf v(d, vt::DType::kBF16, {t, kHkv * kDh}, v_bits.data());
  // The norms are staged at their BF16 model values (the host shadow consumes
  // the widened floats; the device consumes the same rounded bf16 bytes).
  std::vector<uint16_t> qn_bits(kDh), kn_bits(kDh);
  for (int64_t e = 0; e < kDh; ++e) {
    qn_bits[static_cast<size_t>(e)] = ToBf16(q_norm[static_cast<size_t>(e)]);
    kn_bits[static_cast<size_t>(e)] = ToBf16(k_norm[static_cast<size_t>(e)]);
  }
  DBuf qw(d, vt::DType::kBF16, {kDh}, qn_bits.data());
  DBuf kw(d, vt::DType::kBF16, {kDh}, kn_bits.data());
  DBuf sinks_dev(d, vt::DType::kF32, {kHq}, sinks.data());
  DBuf pos_dev(d, vt::DType::kI32, {t}, positions.data());
  std::vector<int64_t> slots(static_cast<size_t>(t));
  for (int64_t i = 0; i < t; ++i) slots[static_cast<size_t>(i)] = i;
  DBuf slot_dev(d, vt::DType::kI64, {t}, slots.data());
  std::vector<int32_t> btab(static_cast<size_t>(nblk));
  for (int64_t b = 0; b < nblk; ++b) btab[static_cast<size_t>(b)] = static_cast<int32_t>(b);
  DBuf bt_dev(d, vt::DType::kI32, {1, nblk}, btab.data());
  const int32_t slens = static_cast<int32_t>(t);
  DBuf sl_dev(d, vt::DType::kI32, {1}, &slens);
  const int32_t qsl[2] = {0, static_cast<int32_t>(t)};
  DBuf qsl_dev(d, vt::DType::kI32, {2}, qsl);

  // The cache page: one DBuf carries both regions (K first, then V), the
  // dense_attn::KvSlice layout.
  const int64_t page_elems = nblk * bs * kHkv * kDh;
  DBuf cache(d, vt::DType::kBF16, {2 * page_elems});
  kv.data = cache.ptr();
  vt::Tensor k_cache = KvSlice(kv, d.q.device, 0);
  vt::Tensor v_cache = KvSlice(kv, d.q.device, 1);

  // Per-head QK RMSNorm BEFORE RoPE (the W1 device-norm arm's shape).
  vt::Tensor qh = Reshape(q.t(), {t * kHq, kDh});
  vt::Tensor kh = Reshape(k.t(), {t * kHkv, kDh});
  DBuf qn(d, vt::DType::kBF16, {t * kHq, kDh});
  DBuf kn(d, vt::DType::kBF16, {t * kHkv, kDh});
  vt::RmsNorm(d.q, qn.t(), qh, qw.t(), vt::RmsNormArgs{static_cast<float>(eps), false});
  vt::RmsNorm(d.q, kn.t(), kh, kw.t(), vt::RmsNormArgs{static_cast<float>(eps), false});

  vt::Tensor q3 = Reshape(qn.t(), {t, kHq, kDh});
  vt::Tensor k3 = Reshape(kn.t(), {t, kHkv, kDh});
  vt::Tensor v3 = Reshape(v.t(), {t, kHkv, kDh});
  if (has_rope) {
    vt::RopeArgs ra{};
    ra.base = static_cast<float>(theta);
    ra.rotary_dim = static_cast<int>(kDh);
    ra.is_neox_style = true;
    vt::RopeNeox(d.q, q3, k3, pos_dev.t(), ra);
  }

  vllm::dense_attn::WriteKvCache(d.q, kv, k3, v3, k_cache, v_cache,
                                 slot_dev.t());

  DBuf attn(d, vt::DType::kBF16, {t, kHq, kDh});
  vt::Tensor sinks_t = sinks_dev.t();  // the args carry a borrowed pointer
  vt::PagedAttentionArgs pa{};
  pa.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kDh)));
  pa.causal = true;
  if (is_sliding)
    pa.window_size = vt::AttentionWindow{static_cast<int32_t>(sliding_window - 1), 0};
  pa.attn_sink = &sinks_t;
  vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, bt_dev.t(),
                     sl_dev.t(), qsl_dev.t(), pa);

  std::vector<uint16_t> out_bits(static_cast<size_t>(t * kHq * kDh));
  attn.Download(d, out_bits.data());
  std::vector<float> out(static_cast<size_t>(t * kHq * kDh));
  for (size_t i = 0; i < out.size(); ++i) out[i] = FromBf16(out_bits[i]);
  return out;
}

struct ProbeFixtures {
  std::vector<uint16_t> q_bits, k_bits, v_bits;
  std::vector<float> q_norm, k_norm, sinks, q_wide, k_wide, v_wide;
  std::vector<int32_t> positions;
};

ProbeFixtures MakeFixtures(int64_t t, std::mt19937& rng) {
  std::normal_distribution<float> nd(0.0f, 0.5f);
  ProbeFixtures f;
  f.q_bits.resize(static_cast<size_t>(t * kHq * kDh));
  f.k_bits.resize(static_cast<size_t>(t * kHkv * kDh));
  f.v_bits.resize(static_cast<size_t>(t * kHkv * kDh));
  for (auto& b : f.q_bits) b = ToBf16(nd(rng));
  for (auto& b : f.k_bits) b = ToBf16(nd(rng));
  for (auto& b : f.v_bits) b = ToBf16(nd(rng));
  std::normal_distribution<float> nw(1.0f, 0.1f), sn(0.0f, 2.0f);
  f.q_norm.resize(static_cast<size_t>(kDh));
  f.k_norm.resize(static_cast<size_t>(kDh));
  for (auto& w : f.q_norm) w = Bf16Round(nw(rng));
  for (auto& w : f.k_norm) w = Bf16Round(nw(rng));
  f.sinks.resize(static_cast<size_t>(kHq));
  for (auto& s : f.sinks) s = sn(rng);
  auto widen = [](const std::vector<uint16_t>& bits) {
    std::vector<float> out(bits.size());
    for (size_t i = 0; i < bits.size(); ++i) out[i] = FromBf16(bits[i]);
    return out;
  };
  f.q_wide = widen(f.q_bits);
  f.k_wide = widen(f.k_bits);
  f.v_wide = widen(f.v_bits);
  f.positions.resize(static_cast<size_t>(t));
  for (int64_t i = 0; i < t; ++i) f.positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  return f;
}

}  // namespace

// ---- the probe ---------------------------------------------------------------

// One window mode: the device layer vs the host shadow, PER-HEAD slices.
// Both required modes run (the mode flip is where prior attention ports
// broke): sliding-with-RoPE at t=600 (past the 513 window edge, so the clip
// is exercised) and full/no-RoPE.
void RunMode(Dev& d, int64_t t, bool is_sliding, bool has_rope,
             std::mt19937& rng) {
  const double eps = 1e-5;
  const double theta = 10000.0;
  const int64_t window = 513;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(kDh)));
  const ProbeFixtures f = MakeFixtures(t, rng);

  // The host shadow: head-norm -> (rope) -> sinks attention, on the same
  // bf16-widened inputs.
  std::vector<float> q = f.q_wide, k = f.k_wide, v = f.v_wide;
  HeadNorm(q.data(), t * kHq, f.q_norm, eps);
  HeadNorm(k.data(), t * kHkv, f.k_norm, eps);
  if (has_rope) {
    RopeNeoxHalfSplit(q.data(), t, kHq, f.positions, theta);
    RopeNeoxHalfSplit(k.data(), t, kHkv, f.positions, theta);
  }
  std::vector<float> shadow(static_cast<size_t>(t * kHq * kDh));
  SinksAttention(q.data(), k.data(), v.data(), f.sinks.data(), t, is_sliding,
                 window, scale, shadow.data());

  const std::vector<float> dev =
      DeviceLayer(d, t, is_sliding, window, has_rope, theta, eps, f.q_bits,
                  f.k_bits, f.v_bits, f.q_norm, f.k_norm, f.sinks,
                  f.positions);

  // THE SCAR CHECK: every head's slice, individually. A kernel that computes
  // only some heads (tt-sdpa-only-2-heads) or mixes GQA groups
  // (tt-pa-correct-but-output-wrong) fails here even when pooled statistics
  // look fine.
  double worst_rel = 0.0, sum_rel = 0.0;
  int64_t worst_head = -1;
  std::vector<int64_t> structural;
  for (int64_t h = 0; h < kHq; ++h) {
    double se = 0.0, sr = 0.0, max_abs = 0.0;
    for (int64_t i = 0; i < t; ++i)
      for (int64_t e = 0; e < kDh; ++e) {
        const size_t idx = static_cast<size_t>((i * kHq + h) * kDh + e);
        const double diff = static_cast<double>(dev[idx]) - shadow[idx];
        se += diff * diff;
        sr += static_cast<double>(shadow[idx]) * shadow[idx];
        max_abs = std::max(max_abs, std::fabs(diff));
      }
    const double n = static_cast<double>(t * kDh);
    const double rel = std::sqrt(se / n) / std::sqrt(sr / n);
    MESSAGE("head " << h << ": rel_rms=" << rel << " max_abs=" << max_abs);
    sum_rel += rel;
    if (rel > worst_rel) {
      worst_rel = rel;
      worst_head = h;
    }
    if (rel > kStructuralBand) structural.push_back(h);
  }
  const double mean_rel = sum_rel / static_cast<double>(kHq);
  MESSAGE((is_sliding ? "SLIDING" : "FULL")
          << " t=" << t << ": worst head " << worst_head << " rel_rms="
          << worst_rel << ", mean " << mean_rel << " (band "
          << kPerHeadBand << ")");
  // Structural divergence is a defect finding, never a band relaxation: a wrong
  // kernel cannot hide inside substrate noise.
  REQUIRE(structural.empty());
  // The band gate itself: EVERY head holds the W2-derived band.
  CHECK_LE(worst_rel, kPerHeadBand);
}

TEST_CASE("flint tt attention probe: sliding layer with RoPE vs the host shadow") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(1501);
  RunMode(d, 600, /*is_sliding=*/true, /*has_rope=*/true, rng);
}

TEST_CASE("flint tt attention probe: full layer without RoPE vs the host shadow") {
  vt::Backend* tb = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  if (tb == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Queue q = tb->CreateQueue();
  Dev d{*tb, q};
  std::mt19937 rng(1502);
  RunMode(d, 96, /*is_sliding=*/false, /*has_rope=*/false, rng);
}
