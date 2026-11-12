// SCRATCH staging-order repro (row FLINT-TT-DENSE-MLP-STAGING, spec
// .agents/specs/flint-tt-dense-mlp-staging.md): the minimal two-tensor
// staging test the spec's stop condition names.
//
// The row's model-level measurement localized the W0 band's allocation-order
// sensitivity to ONE op: the dense DOWN projection's output differs when the
// o_proj operand blocks are released before the MLP section draws (the
// oproj-scoped leg: ff rel_rms 1.8059 vs 1.10e-2 landed), while its INPUT
// bytes (the silu output, verified by download) are bit-identical in both
// legs. This probe reduces that to the smallest sequence that can show it:
//
//   ONE bf16 [M,K]x[K,N] matmul, identical activation and weight bytes in
//   every arm, differing ONLY in the pool-block history of the OUT DBuf:
//
//     arm held:     a same-size block is committed and KEPT while out draws
//     arm recycled: that block is committed and RETURNED first, so out
//                   receives the recycled block (the pool's LIFO free list)
//
// If the two outs differ, the defect reproduces OUTSIDE flint_tt.cpp — the
// spec's substrate stop condition — and the repro below is the record.
// If they agree bit-for-bit, the in-model defect needs more of the forward's
// context, and the probe says that too.

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <vector>

#include <doctest/doctest.h>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/ops.h"

namespace {

using vllm::dense_attn::DBuf;
using vllm::dense_attn::Dev;

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

constexpr int64_t kM = 5, kK = 6144, kN = 2048;  // the L0 dense down shape

// One matmul through the same ops the flint TT forward uses (vt::Matmul on a
// bf16 [K,N] Matmul-B weight), out downloaded as bf16-widened f32.
std::vector<float> DownMatmul(Dev d, DBuf& act, const DBuf& weight) {
  DBuf out(d, vt::DType::kBF16, {kM, kN});
  vt::Matmul(d.q, out.t(), act.t(), weight.t());
  std::vector<uint16_t> bits(static_cast<size_t>(kM * kN));
  out.Download(d, bits.data());
  std::vector<float> host(bits.size());
  for (size_t i = 0; i < bits.size(); ++i) host[i] = FromBf16(bits[i]);
  return host;
}

}  // namespace

TEST_CASE("SCRATCH staging-order repro: the down projection vs the out "
          "block's pool history") {
  if (vt::TryGetBackend(vt::DeviceType::kTENSTORRENT) == nullptr) {
    MESSAGE("no Tenstorrent device; skipped");
    return;
  }
  vt::Backend& ttb = *vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue ttq = ttb.CreateQueue();
  Dev d{ttb, ttq};

  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 0.5f);
  std::vector<uint16_t> act_bits(static_cast<size_t>(kM * kK));
  std::vector<float> act_f(static_cast<size_t>(kM * kK));
  for (size_t i = 0; i < act_bits.size(); ++i) {
    act_bits[i] = ToBf16(nd(rng));
    act_f[i] = FromBf16(act_bits[i]);
  }
  std::vector<uint16_t> w_bits(static_cast<size_t>(kK * kN));
  std::vector<double> ref(static_cast<size_t>(kM * kN), 0.0);
  for (int64_t k = 0; k < kK; ++k)
    for (int64_t n = 0; n < kN; ++n)
      w_bits[static_cast<size_t>(k * kN + n)] = ToBf16(nd(rng) * 0.02f);
  for (int64_t m = 0; m < kM; ++m)
    for (int64_t k = 0; k < kK; ++k) {
      const double a = static_cast<double>(FromBf16(
          act_bits[static_cast<size_t>(m * kK + k)]));
      if (a == 0.0) continue;
      for (int64_t n = 0; n < kN; ++n)
        ref[static_cast<size_t>(m * kN + n)] +=
            a * static_cast<double>(FromBf16(
                    w_bits[static_cast<size_t>(k * kN + n)]));
    }

  // ARM HELD: the same-size scratch block is committed (a matmul result) and
  // KEPT while the down matmul draws its out block — the committed
  // (layer-scope o_proj) pool sequence.
  std::vector<float> held;
  {
    DBuf act(d, vt::DType::kBF16, {kM, kK}, act_bits.data());
    DBuf weight(d, vt::DType::kBF16, {kK, kN}, w_bits.data());
    DBuf oproj_out(d, vt::DType::kBF16, {kM, kN});
    vt::Matmul(d.q, oproj_out.t(), act.t(), weight.t());  // commits a shadow
    held = DownMatmul(d, act, weight);
    // oproj_out still alive here — the layer-scope lifetime.
  }

  // ARM RECYCLED: the same-size block is committed and RETURNED first, so
  // the down matmul's out DBuf receives the RECYCLED block — the
  // oproj-scoped pool sequence.
  std::vector<float> recycled;
  {
    DBuf act(d, vt::DType::kBF16, {kM, kK}, act_bits.data());
    DBuf weight(d, vt::DType::kBF16, {kK, kN}, w_bits.data());
    {
      DBuf oproj_out(d, vt::DType::kBF16, {kM, kN});
      vt::Matmul(d.q, oproj_out.t(), act.t(), weight.t());
    }  // the block returns HERE, before the down matmul draws
    recycled = DownMatmul(d, act, weight);
  }

  double max_abs = 0.0;
  int64_t differing = 0;
  for (size_t i = 0; i < held.size(); ++i) {
    if (held[i] != recycled[i]) ++differing;
    max_abs = std::max(max_abs, std::fabs(double(held[i]) - double(recycled[i])));
  }
  double se = 0.0, sr = 0.0;
  for (size_t i = 0; i < ref.size(); ++i) {
    const double dh = held[i] - ref[i];
    se += dh * dh;
    sr += ref[i] * ref[i];
  }
  const double rel_rms_held =
      std::sqrt(se / double(ref.size())) / std::sqrt(sr / double(ref.size()));
  std::fprintf(stderr,
               "[STAGING-REPRO] held vs recycled: differing=%lld/%zu "
               "max_abs=%.4e; held-vs-hostdouble rel_rms=%.4e\n",
               static_cast<long long>(differing), held.size(), max_abs,
               rel_rms_held);
  CHECK(differing == 0);
}
