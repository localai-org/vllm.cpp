// vllm.cpp ORIGINAL — the `olala` host ops (W2c-1). See olala_ops.h for the
// reference citations. All compute is f32 on host bytes; the weights arrive as
// OwnedTensor (bf16 matmuls, f32 scalars) from LoadOlalaGgufWeights.
#include "vllm/model_executor/models/olala_ops.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace vllm {
namespace {

float Bf16ToF32(uint16_t h) {
  uint32_t bits = static_cast<uint32_t>(h) << 16;
  float f;
  std::memcpy(&f, &bits, sizeof f);
  return f;
}

const float* F32Bytes(const OwnedTensor& t, const char* what) {
  if (t.dtype != vt::DType::kF32) {
    throw std::runtime_error(std::string("olala op: ") + what +
                             " must be F32");
  }
  return reinterpret_cast<const float*>(t.bytes.data());
}

// A [out, in] row-major weight as f32 on the fly: out[j*in + i].
float WAt(const OwnedTensor& w, int64_t j, int64_t i, const char* what) {
  if (w.dtype == vt::DType::kF32) {
    return reinterpret_cast<const float*>(w.bytes.data())[j * w.shape[1] + i];
  }
  if (w.dtype == vt::DType::kBF16) {
    return Bf16ToF32(
        reinterpret_cast<const uint16_t*>(w.bytes.data())[j * w.shape[1] + i]);
  }
  throw std::runtime_error(std::string("olala op: ") + what +
                           " must be F32 or BF16");
}

// Rank-3 expert weight [experts, out, in]: element (e, j, i).
float W3At(const OwnedTensor& w, int64_t e, int64_t j, int64_t i,
           const char* what) {
  if (w.rank != 3) {
    throw std::runtime_error(std::string("olala op: ") + what +
                             " must be rank 3");
  }
  const int64_t in = w.shape[2];
  const int64_t off = (e * w.shape[1] + j) * in + i;
  if (w.dtype == vt::DType::kF32) {
    return reinterpret_cast<const float*>(w.bytes.data())[off];
  }
  return Bf16ToF32(reinterpret_cast<const uint16_t*>(w.bytes.data())[off]);
}

}  // namespace

void OlalaGeodesicNormHost(float* out, const float* x, const float* g,
                           int64_t rows, int64_t cols, float scale,
                           float bias, int64_t layer_idx) {
  constexpr float kClamp = 0.7853981633974483f;  // pi/4 (reference self.clamp = torch.pi/4)
  for (int64_t r = 0; r < rows; ++r) {
    const float* xr = x + r * cols;
    const float* gr = g + r * cols;
    float* orow = out + r * cols;
    double xn2 = 0.0, xg = 0.0, R = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      xn2 += double(xr[c]) * xr[c];
      xg += double(xr[c]) * gr[c];
      R += double(xr[c]) * xr[c];
    }
    xn2 = std::max(xn2, 1e-12);
    R = std::sqrt(std::max(R, 1e-12));
    const double proj = xg / xn2;
    std::vector<float> grad(cols);
    double tn = 0.0;
    for (int64_t c = 0; c < cols; ++c) {
      grad[c] = float(double(gr[c]) - proj * xr[c]);
      tn += double(grad[c]) * grad[c];
    }
    tn = std::sqrt(tn);
    const double safe_tn = std::max(tn, 1e-8);
    double theta = std::min(tn / std::max(R, 1e-6), double(kClamp));
    theta = std::min((theta * scale + bias) / (layer_idx + 1), double(kClamp));
    const double ct = std::cos(theta), st = std::sin(theta);
    for (int64_t c = 0; c < cols; ++c) {
      orow[c] = float(double(xr[c]) * ct + (grad[c] / safe_tn) * R * st);
    }
  }
}

void OlalaMoeHost(float* out, const float* x, int64_t tokens, int64_t hidden,
                  const OlalaLayerWeights& lw, const OlalaParams& p) {
  const int64_t E = p.experts, K = p.top_k, L = p.latent, F = p.expert_ff;
  if (lw.router_weight.rank != 2 || lw.router_weight.shape[0] != E ||
      lw.router_weight.shape[1] != hidden) {
    throw std::runtime_error("olala moe: router weight shape mismatch");
  }
  const float* gate = F32Bytes(lw.router_weight, "router_weight");
  const float* ebias = F32Bytes(lw.router_bias, "router_bias");
  std::vector<float> logits(E), scores(E), latent(L), out_lat(L, 0.0f);
  std::vector<int64_t> idx(K);
  std::vector<float> probs(K);
  for (int64_t t = 0; t < tokens; ++t) {
    const float* xt = x + t * hidden;
    for (int64_t e = 0; e < E; ++e) {
      double s = ebias[e];
      for (int64_t i = 0; i < hidden; ++i) s += double(xt[i]) * gate[e * hidden + i];
      logits[e] = float(s);
      scores[e] = 1.0f / (1.0f + std::exp(-float(s)));
    }
    // top-K by score (stable: lowest index wins ties — torch.topk behavior on
    // equal values is the lower index first for sorted=True; the reference
    // uses torch.topk(scores, k) which returns the FIRST k in sorted order).
    for (int64_t k = 0; k < K; ++k) {
      int64_t best = -1;
      for (int64_t e = 0; e < E; ++e) {
        if (std::find(idx.begin(), idx.begin() + k, e) != idx.begin() + k)
          continue;
        if (best < 0 || scores[e] > scores[best]) best = e;
      }
      idx[k] = best;
    }
    double psum = 0.0;
    for (int64_t k = 0; k < K; ++k) psum += scores[idx[k]];
    for (int64_t k = 0; k < K; ++k)
      probs[k] = float(double(scores[idx[k]]) / (psum + 1e-20) * p.routed_scale);
    // latent down: [L, hidden] @ x
    std::fill(latent.begin(), latent.end(), 0.0f);
    for (int64_t l = 0; l < L; ++l) {
      double s = 0.0;
      for (int64_t i = 0; i < hidden; ++i)
        s += WAt(lw.ffn_latent_down, l, i, "latent_down") * xt[i];
      latent[l] = float(s);
    }
    std::fill(out_lat.begin(), out_lat.end(), 0.0f);
    std::vector<float> h(F);
    for (int64_t k = 0; k < K; ++k) {
      const int64_t e = idx[k];
      for (int64_t f = 0; f < F; ++f) {
        double s = 0.0;
        for (int64_t l = 0; l < L; ++l)
          s += W3At(lw.ffn_up_exps, e, f, l, "up_exps") * latent[l];
        const double a = s > 0.0 ? s : 0.0;  // relu(x)^2 (:1487)
        h[f] = float(a * a);
      }
      for (int64_t l = 0; l < L; ++l) {
        double s = 0.0;
        for (int64_t f = 0; f < F; ++f)
          s += W3At(lw.ffn_down_exps, e, l, f, "down_exps") * h[f];
        out_lat[l] += float(s) * probs[k];  // fp32 accumulate (:1467)
      }
    }
    // latent up: [hidden, L] @ out_lat, plus shared MLP (relu^2, :1416)
    float* ot = out + t * hidden;
    for (int64_t i = 0; i < hidden; ++i) {
      double s = 0.0;
      for (int64_t l = 0; l < L; ++l)
        s += WAt(lw.ffn_latent_up, i, l, "latent_up") * out_lat[l];
      ot[i] = float(s);
    }
    std::vector<float> hs(p.shared_ff);
    for (int64_t f = 0; f < p.shared_ff; ++f) {
      double s = 0.0;
      for (int64_t i = 0; i < hidden; ++i)
        s += WAt(lw.ffn_up_shexp, f, i, "up_shexp") * xt[i];
      const double a = s > 0.0 ? s : 0.0;
      hs[f] = float(a * a);
    }
    for (int64_t i = 0; i < hidden; ++i) {
      double s = 0.0;
      for (int64_t f = 0; f < p.shared_ff; ++f)
        s += WAt(lw.ffn_down_shexp, i, f, "down_shexp") * hs[f];
      ot[i] += float(s);  // out = shared + routed (:1552)
    }
  }
}

}  // namespace vllm
