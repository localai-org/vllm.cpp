// Row NEW-CANDIDATE-PORTS W2c-1 — unit gates for the olala host ops
// (olala_ops.{h,cpp}). The expected values come from an INDEPENDENT f64
// transcription of the author's reference formulas (modeling_olala.py @
// f781f124, cited per case), written inline here in double precision so the
// test never shares code with the implementation. Tiny synthetic weights; the
// pinned-artifact end-to-end gate is the later W2d wave.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <cstdint>
#include <cstring>
#include <vector>

#include "vllm/model_executor/models/olala.h"
#include "vllm/model_executor/models/olala_ops.h"

namespace {

vllm::OwnedTensor F32Tensor(std::vector<int64_t> shape, std::vector<float> vals) {
  vllm::OwnedTensor t;
  t.dtype = vt::DType::kF32;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    t.shape[i] = shape[i];
    n *= shape[i];
  }
  t.bytes.resize(static_cast<size_t>(n) * sizeof(float));
  std::memcpy(t.bytes.data(), vals.data(), vals.size() * sizeof(float));
  return t;
}

vllm::OwnedTensor Bf16Tensor(std::vector<int64_t> shape, const std::vector<float>& vals) {
  vllm::OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (size_t i = 0; i < shape.size(); ++i) {
    t.shape[i] = shape[i];
    n *= shape[i];
  }
  t.bytes.resize(static_cast<size_t>(n) * 2);
  auto* out = reinterpret_cast<uint16_t*>(t.bytes.data());
  for (size_t i = 0; i < vals.size(); ++i) {
    uint32_t bits;
    float f = vals[i];
    std::memcpy(&bits, &f, 4);
    out[i] = static_cast<uint16_t>(bits >> 16);  // truncate = bf16 round-toward-zero
  }
  return t;
}

double Bf16Bits(uint16_t h) {  // reinterpret bf16 bits as f32 (impl's Bf16ToF32)
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  float r;
  std::memcpy(&r, &bits, 4);
  return r;
}

// Independent f64 reference for OlalaGeodesicNormHost (modeling_olala.py:1596-1617).
std::vector<double> RefGeodesic(const std::vector<double>& x,
                                const std::vector<double>& g, double scale,
                                double bias, int64_t layer_idx) {
  const double pi4 = 3.14159265358979323846 / 4;  // torch.pi/4
  size_t C = x.size();
  double xn2 = 1e-12, xg = 0, R = 0;
  for (size_t c = 0; c < C; ++c) {
    xn2 += x[c] * x[c];
    xg += x[c] * g[c];
    R += x[c] * x[c];
  }
  R = std::sqrt(std::max(R, 1e-12));
  double proj = xg / xn2;
  std::vector<double> grad(C);
  double tn = 0;
  for (size_t c = 0; c < C; ++c) {
    grad[c] = g[c] - proj * x[c];
    tn += grad[c] * grad[c];
  }
  tn = std::sqrt(tn);
  double safe_tn = std::max(tn, 1e-8);
  double theta = std::min(tn / std::max(R, 1e-6), pi4);
  theta = std::min((theta * scale + bias) / (layer_idx + 1), pi4);
  std::vector<double> out(C);
  for (size_t c = 0; c < C; ++c)
    out[c] = x[c] * std::cos(theta) + (grad[c] / safe_tn) * R * std::sin(theta);
  return out;
}

// Independent f64 reference for OlalaMoeHost (modeling_olala.py:1448-1552).
std::vector<double> RefMoe(const std::vector<double>& x, int64_t hidden,
                           const vllm::OlalaLayerWeights& lw,
                           const vllm::OlalaParams& p) {
  const int64_t E = p.experts, K = p.top_k, L = p.latent, F = p.expert_ff;
  auto W2 = [&](const vllm::OwnedTensor& w, int64_t j, int64_t i) {
    if (w.dtype == vt::DType::kF32)
      return double(reinterpret_cast<const float*>(w.bytes.data())[j * w.shape[1] + i]);
    return Bf16Bits(reinterpret_cast<const uint16_t*>(w.bytes.data())[j * w.shape[1] + i]);
  };
  auto W3 = [&](const vllm::OwnedTensor& w, int64_t e, int64_t j, int64_t i) {
    int64_t in = w.shape[2];
    int64_t off = (e * w.shape[1] + j) * in + i;
    if (w.dtype == vt::DType::kF32)
      return double(reinterpret_cast<const float*>(w.bytes.data())[off]);
    return Bf16Bits(reinterpret_cast<const uint16_t*>(w.bytes.data())[off]);
  };
  const float* gate = reinterpret_cast<const float*>(lw.router_weight.bytes.data());
  const float* eb = reinterpret_cast<const float*>(lw.router_bias.bytes.data());
  std::vector<double> sc(E);
  for (int64_t e = 0; e < E; ++e) {
    double s = eb[e];
    for (int64_t i = 0; i < hidden; ++i) s += x[i] * gate[e * hidden + i];
    sc[e] = 1.0 / (1.0 + std::exp(-s));
  }
  std::vector<int64_t> idx(E);
  for (int64_t e = 0; e < E; ++e) idx[e] = e;
  std::stable_sort(idx.begin(), idx.end(),
                   [&](int64_t a, int64_t b) { return sc[a] > sc[b]; });
  idx.resize(K);
  double psum = 0;
  for (int64_t k = 0; k < K; ++k) psum += sc[idx[k]];
  std::vector<double> prob(K);
  for (int64_t k = 0; k < K; ++k) prob[k] = sc[idx[k]] / (psum + 1e-20) * p.routed_scale;
  std::vector<double> lat(L, 0), outl(L, 0);
  for (int64_t l = 0; l < L; ++l) {
    double s = 0;
    for (int64_t i = 0; i < hidden; ++i) s += W2(lw.ffn_latent_down, l, i) * x[i];
    lat[l] = s;
  }
  for (int64_t k = 0; k < K; ++k) {
    int64_t e = idx[k];
    std::vector<double> h(F);
    for (int64_t f = 0; f < F; ++f) {
      double s = 0;
      for (int64_t l = 0; l < L; ++l) s += W3(lw.ffn_up_exps, e, f, l) * lat[l];
      double a = std::max(s, 0.0);
      h[f] = a * a;
    }
    for (int64_t l = 0; l < L; ++l) {
      double s = 0;
      for (int64_t f = 0; f < F; ++f) s += W3(lw.ffn_down_exps, e, l, f) * h[f];
      outl[l] += s * prob[k];
    }
  }
  std::vector<double> out(hidden);
  for (int64_t i = 0; i < hidden; ++i) {
    double s = 0;
    for (int64_t l = 0; l < L; ++l) s += W2(lw.ffn_latent_up, i, l) * outl[l];
    out[i] = s;
  }
  std::vector<double> hs(p.shared_ff);
  for (int64_t f = 0; f < p.shared_ff; ++f) {
    double s = 0;
    for (int64_t i = 0; i < hidden; ++i) s += W2(lw.ffn_up_shexp, f, i) * x[i];
    double a = std::max(s, 0.0);
    hs[f] = a * a;
  }
  for (int64_t i = 0; i < hidden; ++i) {
    double s = 0;
    for (int64_t f = 0; f < p.shared_ff; ++f) s += W2(lw.ffn_down_shexp, i, f) * hs[f];
    out[i] += s;
  }
  return out;
}

}  // namespace

TEST_CASE("olala W2c-1: geodesic norm matches the reference formula") {
  const int64_t R = 3, C = 5;
  std::vector<float> x = {1, -2, 0.5f, 3, -1, 0.25f, 2, -0.5f, 1.5f, 0,
                          -1, 1, 0.75f, -2, 2};
  std::vector<float> g = {0.5f, 1, -1, 0.25f, 2, -0.5f, 0.5f, 1.5f, -1, 1,
                          2, -2, 0.5f, 1, -0.25f};
  const float scale = 1.3f, bias = 0.2f;
  const int64_t layer = 7;
  std::vector<float> out(R * C);
  vllm::OlalaGeodesicNormHost(out.data(), x.data(), g.data(), R, C, scale, bias,
                              layer);
  std::vector<double> xd(x.begin(), x.end()), gd(g.begin(), g.end());
  for (int64_t r = 0; r < R; ++r) {
    std::vector<double> xr(xd.begin() + r * C, xd.begin() + (r + 1) * C);
    std::vector<double> gr(gd.begin() + r * C, gd.begin() + (r + 1) * C);
    auto ref = RefGeodesic(xr, gr, scale, bias, layer);
    for (int64_t c = 0; c < C; ++c) {
      CAPTURE(r); CAPTURE(c);
      CHECK(std::fabs(out[r * C + c] - ref[c]) < 1e-5);
    }
  }
}

TEST_CASE("olala W2c-1: latent MoE matches the reference formula (bf16 experts, f32 router)") {
  vllm::OlalaParams p;
  p.hidden = 6;
  p.experts = 4;
  p.top_k = 2;
  p.latent = 3;
  p.expert_ff = 5;
  p.shared_ff = 4;
  p.routed_scale = 2.45;
  vllm::OlalaLayerWeights lw;
  lw.router_weight = F32Tensor({4, 6}, {0.1f, -0.2f, 0.3f, 0.05f, -0.1f, 0.2f,
                                         0.4f, 0.1f, -0.3f, 0.2f, 0.0f, -0.1f,
                                         -0.2f, 0.5f, 0.1f, -0.4f, 0.3f, 0.1f,
                                         0.0f, -0.1f, 0.2f, 0.4f, -0.2f, 0.3f});
  lw.router_bias = F32Tensor({4}, {0.05f, -0.1f, 0.2f, 0.0f});
  lw.ffn_latent_down = Bf16Tensor({3, 6}, {0.2f, -0.1f, 0.3f, 0.0f, 0.1f, -0.2f,
                                            0.1f, 0.2f, -0.3f, 0.4f, 0.0f, 0.1f,
                                            -0.2f, 0.0f, 0.1f, 0.3f, -0.1f, 0.2f});
  lw.ffn_latent_up = Bf16Tensor({6, 3}, {0.1f, 0.2f, -0.1f, 0.3f, 0.0f, -0.2f,
                                          0.0f, 0.1f, 0.2f, -0.1f, 0.3f, 0.1f,
                                          0.2f, -0.2f, 0.1f, 0.0f, 0.1f, -0.3f});
  // experts: [E=4, out=F=5, in=L=3] up, [E=4, out=L=3, in=F=5] down
  std::vector<float> up, down;
  for (int e = 0; e < 4; ++e)
    for (int f = 0; f < 5; ++f)
      for (int l = 0; l < 3; ++l) up.push_back(0.1f * ((e + f + l) % 5) - 0.2f);
  for (int e = 0; e < 4; ++e)
    for (int l = 0; l < 3; ++l)
      for (int f = 0; f < 5; ++f) down.push_back(0.07f * ((e + l + f) % 4) - 0.1f);
  lw.ffn_up_exps = Bf16Tensor({4, 5, 3}, up);
  lw.ffn_down_exps = Bf16Tensor({4, 3, 5}, down);
  lw.ffn_up_shexp = Bf16Tensor({4, 6}, {0.15f, -0.05f, 0.2f, 0.1f, -0.1f, 0.05f,
                                        0.0f, 0.25f, -0.15f, 0.1f, 0.2f, -0.1f,
                                        0.05f, -0.2f, 0.1f, 0.15f, 0.0f, -0.05f,
                                        0.1f, 0.1f, -0.25f, 0.2f, 0.05f, -0.15f});
  lw.ffn_down_shexp = Bf16Tensor({6, 4}, {0.1f, 0.0f, -0.1f, 0.2f, 0.05f, -0.15f,
                                          0.2f, 0.1f, 0.0f, -0.05f, 0.15f, 0.1f,
                                          -0.1f, 0.2f, 0.05f, -0.2f, 0.0f, 0.1f,
                                          0.05f, -0.1f, 0.15f, 0.2f, -0.05f, 0.0f});

  const int64_t T = 2, H = 6;
  std::vector<float> x = {0.5f, -1, 0.25f, 2, -0.5f, 1, 1.5f, 0.5f, -0.75f, 0, 2, -1};
  std::vector<float> out(T * H, 0.0f);
  vllm::OlalaMoeHost(out.data(), x.data(), T, H, lw, p);
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> xt(x.begin() + t * H, x.begin() + (t + 1) * H);
    auto ref = RefMoe(xt, H, lw, p);
    for (int64_t i = 0; i < H; ++i) {
      CAPTURE(t); CAPTURE(i);
      CHECK(std::fabs(out[t * H + i] - ref[i]) < 2e-4);
    }
  }
}
