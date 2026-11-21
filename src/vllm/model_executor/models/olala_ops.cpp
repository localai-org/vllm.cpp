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

namespace vllm {
namespace {

// Plain RMSNorm over the last dim (zero_centered_gamma weight is zeros on this
// artifact, so (1+w)=1 — modeling_olala.py:249-258), f32.
void RmsNormHead(float* y, const float* x, const float* w, int64_t n,
                 double eps) {
  double s = 0.0;
  for (int64_t i = 0; i < n; ++i) s += double(x[i]) * x[i];
  const double inv = 1.0 / std::sqrt(s / n + eps);
  for (int64_t i = 0; i < n; ++i) y[i] = float(double(x[i]) * inv * (1.0 + w[i]));
}

}  // namespace

void OlalaVAttentionHost(float* out, const float* x, const int64_t* positions,
                         int64_t len, const OlalaLayerWeights& lw,
                         const OlalaParams& p) {
  const int64_t H = p.heads, KV = p.kv_heads, D = p.head_dim, R = p.tpa_rank;
  const int64_t SIG = H - KV;                 // 36 signal heads (:515)
  const int64_t SNR = SIG / KV;               // 3 (:524)
  const int64_t GQA = H / KV;                 // 4 repeat (:464)
  const double scale = 1.0 / double(D);       // use_completed_p (:727)
  const double softcap = p.softcap;
  const int64_t wsize = p.window > 0 ? p.window : len;  // wsize (:647)

  // Projections (:578-589): q [L,H,D]; A_k [L,KV,R], B_k [L,R,D] -> k =
  // (A_k @ B_k)/R [L,KV,D]; same for v.
  std::vector<float> q(len * H * D), ak(len * KV * R), bk(len * R * D),
      av(len * KV * R), bv(len * R * D);
  for (int64_t t = 0; t < len; ++t) {
    const float* xt = x + t * p.hidden;
    for (int64_t h = 0; h < H; ++h) {
      double s = 0.0;
      for (int64_t i = 0; i < p.hidden; ++i)
        s += WAt(lw.attn_q, h * D, i, "attn_q") * xt[i];
      q[(t * H + h) * D + 0] = 0.0f;  // placeholder to keep layout; fill below
      for (int64_t d = 0; d < D; ++d) {
        double sd = 0.0;
        for (int64_t i = 0; i < p.hidden; ++i)
          sd += WAt(lw.attn_q, h * D + d, i, "attn_q") * xt[i];
        q[(t * H + h) * D + d] = float(sd);
      }
      (void)s;
    }
    for (int64_t h = 0; h < KV; ++h)
      for (int64_t r = 0; r < R; ++r) {
        double s = 0.0;
        for (int64_t i = 0; i < p.hidden; ++i)
          s += WAt(lw.attn_wa_k, h * R + r, i, "wa_k") * xt[i];
        ak[(t * KV + h) * R + r] = float(s);
      }
    for (int64_t r = 0; r < R; ++r)
      for (int64_t d = 0; d < D; ++d) {
        double s = 0.0;
        for (int64_t i = 0; i < p.hidden; ++i)
          s += WAt(lw.attn_wb_k, r * D + d, i, "wb_k") * xt[i];
        bk[(t * R + r) * D + d] = float(s);
      }
    for (int64_t h = 0; h < KV; ++h)
      for (int64_t r = 0; r < R; ++r) {
        double s = 0.0;
        for (int64_t i = 0; i < p.hidden; ++i)
          s += WAt(lw.attn_wa_v, h * R + r, i, "wa_v") * xt[i];
        av[(t * KV + h) * R + r] = float(s);
      }
    for (int64_t r = 0; r < R; ++r)
      for (int64_t d = 0; d < D; ++d) {
        double s = 0.0;
        for (int64_t i = 0; i < p.hidden; ++i)
          s += WAt(lw.attn_wb_v, r * D + d, i, "wb_v") * xt[i];
        bv[(t * R + r) * D + d] = float(s);
      }
  }
  std::vector<float> k(len * KV * D), v(len * KV * D);
  for (int64_t t = 0; t < len; ++t)
    for (int64_t h = 0; h < KV; ++h)
      for (int64_t d = 0; d < D; ++d) {
        double sk = 0.0, sv = 0.0;
        for (int64_t r = 0; r < R; ++r) {
          sk += double(ak[(t * KV + h) * R + r]) * bk[(t * R + r) * D + d];
          sv += double(av[(t * KV + h) * R + r]) * bv[(t * R + r) * D + d];
        }
        k[(t * KV + h) * D + d] = float(sk / R);
        v[(t * KV + h) * D + d] = float(sv / R);
      }

  // Token shift (:595-640): alpha = sigmoid(shift_proj(x)); k = a*k_prev +
  // (1-a)*k with k_prev zeroed at doc starts (pos==0).
  for (int64_t t = 0; t < len; ++t) {
    const float* xt = x + t * p.hidden;
    const bool doc_start = positions[t] == 0;
    for (int64_t h = 0; h < KV; ++h) {
      double sa = 0.0, sb = 0.0;
      for (int64_t i = 0; i < p.hidden; ++i) {
        sa += WAt(lw.attn_shift_k, h, i, "shift_k") * xt[i];
        sb += WAt(lw.attn_shift_v, h, i, "shift_v") * xt[i];
      }
      // doc starts zero the previous contribution: alpha masked to 0 (:636-637)
      const float akv = doc_start ? 0.0f : float(1.0 / (1.0 + std::exp(-sa)));
      const float avv = doc_start ? 0.0f : float(1.0 / (1.0 + std::exp(-sb)));
      const int64_t tp = doc_start ? -1 : t - 1;
      for (int64_t d = 0; d < D; ++d) {
        const float kp = tp < 0 ? 0.0f : k[(tp * KV + h) * D + d];
        const float vp = tp < 0 ? 0.0f : v[(tp * KV + h) * D + d];
        k[(t * KV + h) * D + d] = akv * kp + (1.0f - akv) * k[(t * KV + h) * D + d];
        v[(t * KV + h) * D + d] = avv * vp + (1.0f - avv) * v[(t * KV + h) * D + d];
      }
    }
  }

  // QK-norm (plain RMSNorm, eps=1e-6) per head (:643-645, :249-258).
  const float* qn = F32Bytes(lw.attn_q_norm, "q_norm");
  const float* kn = F32Bytes(lw.attn_k_norm, "k_norm");
  for (int64_t t = 0; t < len; ++t)
    for (int64_t h = 0; h < H; ++h)
      RmsNormHead(&q[(t * H + h) * D], &q[(t * H + h) * D], qn, D, 1e-6);
  for (int64_t t = 0; t < len; ++t)
    for (int64_t h = 0; h < KV; ++h)
      RmsNormHead(&k[(t * KV + h) * D], &k[(t * KV + h) * D], kn, D, 1e-6);

  // Scalable softmax: q *= scaler[h] * log(min(pos+1, wsize)) (:661-668).
  const float* scaler = F32Bytes(lw.attn_softmax_scaler, "scaler");
  for (int64_t t = 0; t < len; ++t) {
    const double lp = std::log(std::min(double(positions[t]) + 1.0, double(wsize)));
    for (int64_t h = 0; h < H; ++h) {
      const double f = scaler[h] * lp;
      for (int64_t d = 0; d < D; ++d) q[(t * H + h) * D + d] = float(q[(t * H + h) * D + d] * f);
    }
  }

  // Attention (:467-495) + differential merge (:734-740).
  const float* lam = nullptr;
  std::vector<float> lamb(len * KV);
  for (int64_t t = 0; t < len; ++t) {
    const float* xt = x + t * p.hidden;
    for (int64_t h = 0; h < KV; ++h) {
      double s = 0.0;
      for (int64_t i = 0; i < p.hidden; ++i)
        s += WAt(lw.attn_lambda, h, i, "lambda") * xt[i];
      lamb[t * KV + h] = float(1.0 / (1.0 + std::exp(-s)));
    }
  }
  (void)lam;
  std::vector<float> att(len * H * D, 0.0f);
  std::vector<double> wbuf(len);
  for (int64_t t = 0; t < len; ++t) {
    for (int64_t h = 0; h < H; ++h) {
      const int64_t kvh = h / GQA;  // repeat_interleave (:464)
      double wsum = 0.0;
      for (int64_t s = 0; s <= t; ++s) {  // causal; window below
        if (t - s > wsize) continue;      // j >= i - w_left (:489)
        double sc = 0.0;
        for (int64_t d = 0; d < D; ++d)
          sc += double(q[(t * H + h) * D + d]) * k[(s * KV + kvh) * D + d];
        sc *= scale;
        if (softcap > 0.0) sc = softcap * std::tanh(sc / softcap);  // :469-470
        wbuf[s] = sc;
        wsum += std::exp(sc);
      }
      // softmax over allowed s (f32 accumulation like flash-attn :459,493)
      double inv = 1.0 / wsum;
      for (int64_t s = 0; s <= t; ++s) {
        if (t - s > wsize) continue;
        const double pr = std::exp(wbuf[s]) * inv;
        for (int64_t d = 0; d < D; ++d)
          att[(t * H + h) * D + d] += float(pr * v[(s * KV + kvh) * D + d]);
      }
    }
  }
  // Differential merge: view [L, KV, SNR+1, D]; sig = [..., :SNR], noi =
  // [..., SNR:SNR+1]; out = sig - sigmoid(lambda) * noi -> [L, SIG, D].
  for (int64_t t = 0; t < len; ++t) {
    for (int64_t h = 0; h < KV; ++h) {
      const double lv = lamb[t * KV + h];
      for (int64_t s = 0; s < SNR; ++s)
        for (int64_t d = 0; d < D; ++d) {
          const float a = att[(t * H + h * (SNR + 1) + s) * D + d];
          const float n = att[(t * H + h * (SNR + 1) + SNR) * D + d];
          out[(t * SIG + h * SNR + s) * D + d] = float(a - lv * n);
        }
    }
  }
}

}  // namespace vllm
