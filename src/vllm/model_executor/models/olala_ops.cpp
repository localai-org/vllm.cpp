// vllm.cpp ORIGINAL — the `olala` host ops (W2c-1). See olala_ops.h for the
// reference citations. All compute is f32 on host bytes; the weights arrive as
// OwnedTensor (bf16 matmuls, f32 scalars) from LoadOlalaGgufWeights.
#include "vllm/model_executor/models/olala_ops.h"

namespace vllm {
constexpr double kOlalaPi = 3.14159265358979323846;  // the math constant PI is not portable to MSVC under that name
}  // namespace vllm

#include <algorithm>
#include <cstdio>
#include <cstdlib>
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

namespace vllm {

void OlalaMambaStepHost(float* out, OlalaMambaState& st, const float* u,
                        const OlalaLayerWeights& lw, const OlalaParams& p) {
  const bool dbg = std::getenv("OLALA_STEP_DBG") != nullptr;
  if (dbg) fprintf(stderr, "step enter: H=%lld P=%lld N=%lld R=%lld A=%lld hid=%lld\n",
                   (long long)(2*p.hidden/p.headdim), (long long)p.headdim,
                   (long long)p.d_state, (long long)p.mimo_dim,
                   (long long)p.rope_angles, (long long)p.hidden);
  const int64_t H = 2 * p.hidden / p.headdim;  // nheads = d_inner/headdim (:956)
  const int64_t P = p.headdim, N = p.d_state, R = p.mimo_dim, A = p.rope_angles;
  const int64_t hid = p.hidden;
  const double A_floor = 1e-4;                 // :949
  // in_proj split per head: z[P], x[P], dt, A, trap (:1287-1291; 2P+3 per head)
  std::vector<float> z(H * P), x(H * P), dt(H), Av(H), trap(H);
  for (int64_t h = 0; h < H; ++h) {
    for (int64_t d = 0; d < P; ++d) {
      double sz = 0, sx = 0;
      for (int64_t i = 0; i < hid; ++i) {
        const double ui = u[i];
        sz += WAt(lw.ssm_in, h * (2 * P + 3) + d, i, "ssm_in") * ui;
        sx += WAt(lw.ssm_in, h * (2 * P + 3) + P + d, i, "ssm_in") * ui;
      }
      z[h * P + d] = float(sz);
      x[h * P + d] = float(sx);
    }
    double sdt = 0, sA = 0, str = 0;
    for (int64_t i = 0; i < hid; ++i) {
      const double ui = u[i];
      sdt += WAt(lw.ssm_in, h * (2 * P + 3) + 2 * P, i, "ssm_in") * ui;
      sA += WAt(lw.ssm_in, h * (2 * P + 3) + 2 * P + 1, i, "ssm_in") * ui;
      str += WAt(lw.ssm_in, h * (2 * P + 3) + 2 * P + 2, i, "ssm_in") * ui;
    }
    dt[h] = float(sdt);
    Av[h] = float(sA);
    trap[h] = float(str);
  }
  if (dbg) fprintf(stderr, "stage: in_proj done\n");
  // in_proj_dyn: B [R*N], C [R*N], angles [A] (:1299-1301; ngroups=1)
  std::vector<float> Bv(R * N), Cv(R * N), ang(A);
  for (int64_t j = 0; j < R * N; ++j) {
    double sb = 0, sc = 0;
    for (int64_t i = 0; i < hid; ++i) {
      sb += WAt(lw.ssm_in_dyn, j, i, "ssm_in_dyn") * u[i];
      sc += WAt(lw.ssm_in_dyn, R * N + j, i, "ssm_in_dyn") * u[i];
    }
    Bv[j] = float(sb);
    Cv[j] = float(sc);
  }
  for (int64_t a = 0; a < A; ++a) {
    double s = 0;
    for (int64_t i = 0; i < hid; ++i)
      s += WAt(lw.ssm_in_dyn, 2 * R * N + a, i, "ssm_in_dyn") * u[i];
    ang[a] = float(s);
  }
  // _preprocess (:1248-1263): DT, _A clamp, trap sigmoid, B/C norm over N, expand
  const float* bn = F32Bytes(lw.ssm_b_norm, "b_norm");
  const float* cn = F32Bytes(lw.ssm_c_norm, "c_norm");
  const float* dbias = F32Bytes(lw.ssm_dt_bias, "dt_bias");
  std::vector<float> DT(H), alpha(H), beta(H), gamma(H), Bn(R * H * N), Cn(R * H * N);
  for (int64_t h = 0; h < H; ++h) {
    const double dtv = std::log1p(std::exp(double(dt[h]) + dbias[h]));  // softplus
    DT[h] = float(dtv);
    double a = -std::log1p(std::exp(double(Av[h])));                    // -softplus(A)
    a = std::min(a, -A_floor);                                          // clamp max=-A_floor
    const double adt = a * dtv;
    alpha[h] = float(std::exp(adt));
    const double tr = 1.0 / (1.0 + std::exp(-double(trap[h])));         // sigmoid
    beta[h] = float((1.0 - tr) * dtv * std::exp(adt));
    gamma[h] = float(tr * dtv);
    (void)adt;
  }
  // B/C norm over d_state then expand G->H (ngroups=1, so r index direct)
  for (int64_t r = 0; r < R; ++r) {
    for (int64_t h = 0; h < H; ++h) {
      // rmsnorm over N of Bv[r*N ..], weight bn (zero-centered -> 1+w=1 on this ckpt)
      double s = 0;
      for (int64_t n = 0; n < N; ++n) s += double(Bv[r * N + n]) * Bv[r * N + n];
      const double inv = 1.0 / std::sqrt(s / N + 1e-6);
      for (int64_t n = 0; n < N; ++n) {
        const double w = 1.0 + bn[n];
        Bn[(r * H + h) * N + n] = float(double(Bv[r * N + n]) * inv * w);
      }
    }
    double s = 0;
    for (int64_t n = 0; n < N; ++n) s += double(Cv[r * N + n]) * Cv[r * N + n];
    const double inv = 1.0 / std::sqrt(s / N + 1e-6);
    for (int64_t h = 0; h < H; ++h)
      for (int64_t n = 0; n < N; ++n) {
        const double w = 1.0 + cn[n];
        Cn[(r * H + h) * N + n] = float(double(Cv[r * N + n]) * inv * w);
      }
  }
  if (dbg) fprintf(stderr, "stage: dyn done\n");
  // rotary step (:861-868): angle += tanh(ang)*DT*pi; rotate q=C+C_bias, k=B+B_bias
  // pairs (i, i+N/2) for i < A (:764-770). biases are (h, r, n) -> (r, h, n).
  const float* cb = F32Bytes(lw.ssm_c_bias, "c_bias");
  const float* kb = F32Bytes(lw.ssm_b_bias, "b_bias");
  std::vector<float> Q(R * H * N), K(R * H * N);
  for (int64_t h = 0; h < H; ++h) {
    std::vector<double> na(A);
    for (int64_t a = 0; a < A; ++a) {
      const double prev = st.angle[h * A + a];
      na[a] = prev + std::tanh(double(ang[a])) * DT[h] * kOlalaPi;
      st.angle[h * A + a] = float(std::fmod(na[a], 2 * kOlalaPi));
    }
    std::vector<double> cs(A), sn(A);
    for (int64_t a = 0; a < A; ++a) { cs[a] = std::cos(na[a]); sn[a] = std::sin(na[a]); }
    for (int64_t r = 0; r < R; ++r) {
      for (int64_t n = 0; n < N; ++n) {
        double qv = Cn[(r * H + h) * N + n] + cb[(h * R + r) * N + n];
        double kv = Bn[(r * H + h) * N + n] + kb[(h * R + r) * N + n];
        if (n < A) {
          const double q2 = Cn[(r * H + h) * N + (n + N / 2)] + cb[(h * R + r) * N + (n + N / 2)];
          const double k2 = Bn[(r * H + h) * N + (n + N / 2)] + kb[(h * R + r) * N + (n + N / 2)];
          Q[(r * H + h) * N + n] = float(qv * cs[n] - q2 * sn[n]);
          Q[(r * H + h) * N + (n + N / 2)] = float(qv * sn[n] + q2 * cs[n]);
          K[(r * H + h) * N + n] = float(kv * cs[n] - k2 * sn[n]);
          K[(r * H + h) * N + (n + N / 2)] = float(kv * sn[n] + k2 * cs[n]);
        } else {
          Q[(r * H + h) * N + n] = float(qv);
          K[(r * H + h) * N + n] = float(kv);
        }
      }
    }
  }
  if (dbg) fprintf(stderr, "stage: rotary done\n");
  // mimo projections (h, r, p) -> (r, h, p) (:1328-1330)
  auto proj = [&](const OwnedTensor& w, int64_t r, int64_t h, int64_t pp) {
    return WAt(w, h * R + r, pp, "mimo");  // [H, R, P] read as [h*R+r, p]
  };
  // step (:871-889)
  const float* Dv = F32Bytes(lw.ssm_d, "D");
  std::vector<float> newssm(H * P * N, 0.0f);
  for (int64_t h = 0; h < H; ++h) {
    for (int64_t pp = 0; pp < P; ++pp) {
      for (int64_t n = 0; n < N; ++n) {
        double s = st.ssm[(h * P + pp) * N + n] * alpha[h];
        for (int64_t r = 0; r < R; ++r) {
          const double xv = double(x[h * P + pp]) * proj(lw.ssm_mimo_x, r, h, pp);
          s += xv * gamma[h] * K[(r * H + h) * N + n];
          const double xs = st.v_state[h * P + pp] * proj(lw.ssm_mimo_x, r, h, pp);
          s += xs * beta[h] * st.k_state[(r * H + h) * N + n];
        }
        newssm[(h * P + pp) * N + n] = float(s);
      }
    }
  }
  if (dbg) fprintf(stderr, "stage: mimo proj done, ssm=%zu\n", newssm.size());
  // o = einsum(new, C) + xv * D per (r,h,p), gated by silu(z*zproj), then
  // y = sum_r o_r * outproj_r (:886-888).
  for (int64_t h = 0; h < H; ++h)
    for (int64_t pp = 0; pp < P; ++pp) {
      double yv = 0;
      for (int64_t r = 0; r < R; ++r) {
        double accr = 0;
        for (int64_t n = 0; n < N; ++n)
          accr += newssm[(h * P + pp) * N + n] * Q[(r * H + h) * N + n];
        const double xvr = double(x[h * P + pp]) * proj(lw.ssm_mimo_x, r, h, pp);
        const double orr = accr + xvr * Dv[h];
        const double zproj = proj(lw.ssm_mimo_z, r, h, pp);
        const double gate = double(z[h * P + pp]) * zproj;
        const double silu = gate / (1.0 + std::exp(-gate));
        yv += orr * silu * proj(lw.ssm_mimo_o, r, h, pp);
      }
      out[h * P + pp] = float(yv);
    }
  if (dbg) fprintf(stderr, "stage: out combine done\n");
  st.ssm = newssm;
  // k_state = K (B), v_state = x
  for (int64_t r = 0; r < R; ++r)
    for (int64_t h = 0; h < H; ++h)
      for (int64_t n = 0; n < N; ++n)
        st.k_state[(r * H + h) * N + n] = K[(r * H + h) * N + n];
  for (int64_t h = 0; h < H; ++h)
    for (int64_t pp = 0; pp < P; ++pp) st.v_state[h * P + pp] = x[h * P + pp];
}

}  // namespace vllm

namespace vllm {

void OlalaVAttentionStepHost(float* attn_k_cache, float* attn_v_cache,
                             float* raw_k_last, float* raw_v_last,
                             const float* x, int64_t pos,
                             const OlalaLayerWeights& lw, const OlalaParams& p,
                             float* out) {
  // Decode arm (L==1, modeling_olala.py:578-645, :643-673, :720-740).
  // attn_k_cache/attn_v_cache: [pos+1, KV*D] post-shift post-norm k/v; rows
  // 0..pos-1 are read, row pos is WRITTEN by this step. raw_k_last/
  // raw_v_last: [KV*D] in/out — the RAW pre-shift k/v of the previous token
  // (the reference caches them at :619, BEFORE qk-norm, precisely so the
  // shift can blend with them next step). x: [hidden]. out: [SIG*D = 4608]
  // mixer output (pre-mixer_proj).
  const int64_t H = p.heads, KV = p.kv_heads, D = p.head_dim, R = p.tpa_rank;
  const int64_t SIG = H - KV, SNR = SIG / KV, GQA = H / KV;
  const double scale = 1.0 / double(D);  // use_completed_p (:727)
  const int64_t wsize = p.window;

  // 1. projections: q [H,D]; k_raw/v_raw [KV,D] = (A @ B)/rank (:578-589)
  std::vector<float> q(H * D), kraw(KV * D), vraw(KV * D), ak(KV * R),
      av(KV * R), bk(R * D), bv(R * D);
  for (int64_t h = 0; h < H; ++h)
    for (int64_t d = 0; d < D; ++d) {
      double s = 0.0;
      for (int64_t i = 0; i < p.hidden; ++i)
        s += WAt(lw.attn_q, h * D + d, i, "attn_q") * x[i];
      q[h * D + d] = float(s);
    }
  for (int64_t h = 0; h < KV; ++h)
    for (int64_t r = 0; r < R; ++r) {
      double sa = 0.0, sv = 0.0;
      for (int64_t i = 0; i < p.hidden; ++i) {
        sa += WAt(lw.attn_wa_k, h * R + r, i, "wa_k") * x[i];
        sv += WAt(lw.attn_wa_v, h * R + r, i, "wa_v") * x[i];
      }
      ak[h * R + r] = float(sa);
      av[h * R + r] = float(sv);
    }
  for (int64_t r = 0; r < R; ++r)
    for (int64_t d = 0; d < D; ++d) {
      double sb = 0.0, sbv = 0.0;
      for (int64_t i = 0; i < p.hidden; ++i) {
        sb += WAt(lw.attn_wb_k, r * D + d, i, "wb_k") * x[i];
        sbv += WAt(lw.attn_wb_v, r * D + d, i, "wb_v") * x[i];
      }
      bk[r * D + d] = float(sb);
      bv[r * D + d] = float(sbv);
    }
  for (int64_t h = 0; h < KV; ++h)
    for (int64_t d = 0; d < D; ++d) {
      double sk = 0.0, sv = 0.0;
      for (int64_t r = 0; r < R; ++r) {
        sk += double(ak[h * R + r]) * bk[r * D + d];
        sv += double(av[h * R + r]) * bv[r * D + d];
      }
      kraw[h * D + d] = float(sk / R);
      vraw[h * D + d] = float(sv / R);
    }

  // 2. token shift: k = alpha*k_prev + (1-alpha)*k_raw, alpha =
  //    sigmoid(shift_proj(x)) (:596-611). k_prev = the cached RAW k/v.
  std::vector<float> k(KV * D), v(KV * D);
  for (int64_t h = 0; h < KV; ++h) {
    double sa = 0.0, sb = 0.0;
    for (int64_t i = 0; i < p.hidden; ++i) {
      sa += WAt(lw.attn_shift_k, h, i, "shift_k") * x[i];
      sb += WAt(lw.attn_shift_v, h, i, "shift_v") * x[i];
    }
    const float a_k = float(1.0 / (1.0 + std::exp(-sa)));
    const float a_v = float(1.0 / (1.0 + std::exp(-sb)));
    for (int64_t d = 0; d < D; ++d) {
      k[h * D + d] = a_k * raw_k_last[h * D + d] +
                     (1.0f - a_k) * kraw[h * D + d];
      v[h * D + d] = a_v * raw_v_last[h * D + d] +
                     (1.0f - a_v) * vraw[h * D + d];
    }
  }

  // 3. cache the RAW k/v for the NEXT token's shift (:619) — before qk-norm.
  for (int64_t i = 0; i < KV * D; ++i) {
    raw_k_last[i] = kraw[i];
    raw_v_last[i] = vraw[i];
  }

  // 4. QK-norm on q and the shifted k (:643-645; zero-centered weight is zero).
  const float* qn = F32Bytes(lw.attn_q_norm, "q_norm");
  const float* kn = F32Bytes(lw.attn_k_norm, "k_norm");
  for (int64_t h = 0; h < H; ++h) RmsNormHead(&q[h * D], &q[h * D], qn, D, 1e-6);
  for (int64_t h = 0; h < KV; ++h) RmsNormHead(&k[h * D], &k[h * D], kn, D, 1e-6);

  // 5. scalable softmax: q *= scaler[h] * log(min(pos+1, wsize)) (:661-668).
  const float* scaler = F32Bytes(lw.attn_softmax_scaler, "scaler");
  const double lp = std::log(std::min(double(pos) + 1.0, double(wsize)));
  for (int64_t h = 0; h < H; ++h) {
    const double f = scaler[h] * lp;
    for (int64_t d = 0; d < D; ++d) q[h * D + d] = float(q[h * D + d] * f);
  }

  // 6. lambda (differential noise gate) (:738).
  std::vector<float> lamb(KV);
  for (int64_t h = 0; h < KV; ++h) {
    double s = 0.0;
    for (int64_t i = 0; i < p.hidden; ++i)
      s += WAt(lw.attn_lambda, h, i, "lambda") * x[i];
    lamb[h] = float(1.0 / (1.0 + std::exp(-s)));
  }

  // 7. attention: causal, left window wsize, scale 1/D, softcap tanh,
  //    GQA repeat_interleave, fp32 accumulation (:459-495). The cache holds
  //    positions 0..pos-1; position pos is the current k/v.
  std::vector<float> att(H * D, 0.0f);
  std::vector<double> w(pos + 1);
  for (int64_t h = 0; h < H; ++h) {
    const int64_t kvh = h / GQA;  // repeat_interleave (:464)
    double wsum = 0.0;
    for (int64_t s = 0; s <= pos; ++s) {
      if (pos - s > wsize) continue;  // j >= i - w_left (:489)
      double sc = 0.0;
      for (int64_t d = 0; d < D; ++d) {
        const float kv_d =
            s == pos ? k[kvh * D + d] : attn_k_cache[s * KV * D + kvh * D + d];
        sc += double(q[h * D + d]) * kv_d;
      }
      sc *= scale;
      if (p.softcap > 0.0) sc = p.softcap * std::tanh(sc / p.softcap);  // :469
      w[s] = sc;
      wsum += std::exp(sc);
    }
    const double inv = 1.0 / wsum;
    std::vector<double> o(D, 0.0);
    for (int64_t s = 0; s <= pos; ++s) {
      if (pos - s > wsize) continue;
      const double pr = std::exp(w[s]) * inv;
      for (int64_t d = 0; d < D; ++d) {
        const float vv =
            s == pos ? v[kvh * D + d] : attn_v_cache[s * KV * D + kvh * D + d];
        o[d] += pr * vv;
      }
    }
    for (int64_t d = 0; d < D; ++d) att[h * D + d] = float(o[d]);
  }

  // 8. differential merge: head view [KV, SNR+1, D]; sig = slots 0..SNR-1,
  //    noi = slot SNR; out[kv*SNR+s] = sig - sigmoid(lambda)*noi (:734-740).
  for (int64_t kv = 0; kv < KV; ++kv) {
    const double lv = lamb[kv];
    for (int64_t d = 0; d < D; ++d)
      for (int64_t s = 0; s < SNR; ++s) {
        const int64_t sig_h = kv * (SNR + 1) + s;
        const int64_t noi_h = kv * (SNR + 1) + SNR;
        out[(kv * SNR + s) * D + d] =
            float(att[sig_h * D + d] - lv * att[noi_h * D + d]);
      }
  }

  // 9. commit this token's post-shift post-norm k/v into cache row pos (the
  //    caller sized the cache to (pos+1)*KV*D) and re-cache the RAW k/v for
  //    the NEXT token's shift (:619, before qk-norm).
  for (int64_t i = 0; i < KV * D; ++i) {
    attn_k_cache[pos * KV * D + i] = k[i];
    attn_v_cache[pos * KV * D + i] = v[i];
    raw_k_last[i] = kraw[i];
    raw_v_last[i] = vraw[i];
  }
}

void OlalaGenerateHost(const OlalaWeights& w, const OlalaParams& p,
                       std::vector<int64_t>& token_ids, int64_t n_gen) {
  const int64_t L = p.layers, H = p.heads, KV = p.kv_heads, D = p.head_dim;
  const int64_t SIG = H - KV;
  if (static_cast<int64_t>(p.layer_types.size()) != L)
    throw std::runtime_error("olala gen: layer_types length mismatch");
  if (static_cast<int64_t>(w.layers.size()) != L)
    throw std::runtime_error("olala gen: layer weight count mismatch");

  // per-layer caches: M layers carry the Mamba3 state, V layers the attention
  // KV cache + the raw k/v shift cache.
  struct MState { OlalaMambaState s; };
  std::vector<OlalaMambaState> mstates;
  std::vector<std::vector<float>> vk_cache, vv_cache;  // [pos, KV*D]
  std::vector<std::vector<float>> rk_last, rv_last;    // [KV*D]
  std::vector<int64_t> v_pos;                          // cache length per V layer
  for (int64_t l = 0; l < L; ++l) {
    if (p.layer_types[static_cast<size_t>(l)] == 'M') {
      OlalaMambaState s;
      s.angle.assign(2 * p.hidden / p.headdim * p.rope_angles, 0.0f);
      s.ssm.assign(2 * p.hidden / p.headdim * p.headdim * p.d_state, 0.0f);
      s.k_state.assign(p.mimo_dim * (2 * p.hidden / p.headdim) * p.d_state, 0.0f);
      s.v_state.assign(2 * p.hidden / p.headdim * p.headdim, 0.0f);
      mstates.push_back(std::move(s));
    } else {
      vk_cache.emplace_back();
      vv_cache.emplace_back();
      rk_last.emplace_back(KV * D, 0.0f);
      rv_last.emplace_back(KV * D, 0.0f);
      v_pos.push_back(0);
    }
  }

  auto embed_at = [&](int64_t tok, float* x) {
    for (int64_t i = 0; i < p.hidden; ++i) {
      const size_t off = (static_cast<size_t>(tok) * p.hidden + i) * 2;
      const uint32_t bits = static_cast<uint32_t>(w.embed.bytes.data()[off]) |
                            (static_cast<uint32_t>(w.embed.bytes.data()[off + 1]) << 8);
      x[i] = Bf16ToF32(static_cast<uint16_t>(bits));
    }
    // normalize_embeddings_ngpt=True (modeling_olala.py:2020-2021)
    double s = 0.0;
    for (int64_t i = 0; i < p.hidden; ++i) s += double(x[i]) * x[i];
    const double inv = 1.0 / std::sqrt(std::max(s, 1e-12));
    for (int64_t i = 0; i < p.hidden; ++i) x[i] = float(double(x[i]) * inv);
  };

  const int64_t SIG_ALL = p.heads - p.kv_heads;  // signal heads
  const int64_t total = static_cast<int64_t>(token_ids.size()) + n_gen;
  std::vector<float> x(p.hidden), ymix;
  int64_t midx = 0, vidx = 0;  // M/V layer ordinals
  std::vector<float> logits(p.vocab);
  for (int64_t t = 0; t < total; ++t) {
    embed_at(token_ids[static_cast<size_t>(t)], x.data());
    midx = 0;
    vidx = 0;
    for (int64_t l = 0; l < L; ++l) {
      const OlalaLayerWeights& lw = w.layers[static_cast<size_t>(l)];
      const bool is_m = p.layer_types[static_cast<size_t>(l)] == 'M';
      // MIXER (resize the scratch to this layer's mixer output dim:
      // M = 2*hidden, V = signal_heads*head_dim)
      ymix.assign(static_cast<size_t>(is_m ? 2 * p.hidden : SIG_ALL * D), 0.0f);
      if (is_m) {
        OlalaMambaStepHost(ymix.data(), mstates[static_cast<size_t>(midx)],
                           x.data(), lw, p);
        ++midx;
      } else {
        // grow the V cache by one row, then run the step against it
        vk_cache[static_cast<size_t>(vidx)].resize(
            static_cast<size_t>((t + 1) * KV * D), 0.0f);
        vv_cache[static_cast<size_t>(vidx)].resize(
            static_cast<size_t>((t + 1) * KV * D), 0.0f);
        OlalaVAttentionStepHost(
            vk_cache[static_cast<size_t>(vidx)].data(),
            vv_cache[static_cast<size_t>(vidx)].data(),
            rk_last[static_cast<size_t>(vidx)].data(),
            rv_last[static_cast<size_t>(vidx)].data(),
            x.data(), t, lw, p, ymix.data());
        ++vidx;
      }
      // mixer_proj: attn_output.weight [hidden, mixer_out] @ y
      const int64_t mixer_out = is_m ? 2 * p.hidden : SIG * D;
      std::vector<float> proj(p.hidden, 0.0f);
      for (int64_t i = 0; i < p.hidden; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < mixer_out; ++j)
          s += WAt(lw.attn_output, i, j, "attn_output") * ymix[static_cast<size_t>(j)];
        proj[static_cast<size_t>(i)] = float(s);
      }
      // geodesic_mixer
      std::vector<float> xg(p.hidden);
      OlalaGeodesicNormHost(xg.data(), x.data(), proj.data(), 1, p.hidden,
                            F32Bytes(lw.geo_mixer_scale, "geo_mixer_scale")[0],
                            F32Bytes(lw.geo_mixer_bias, "geo_mixer_bias")[0],
                            l);
      for (int64_t i = 0; i < p.hidden; ++i) x[i] = xg[i];
      // MLP (latent MoE)
      std::vector<float> mo(p.hidden, 0.0f);
      OlalaMoeHost(mo.data(), x.data(), 1, p.hidden, lw, p);
      // geodesic_mlp
      std::vector<float> xg2(p.hidden);
      OlalaGeodesicNormHost(xg2.data(), x.data(), mo.data(), 1, p.hidden,
                            F32Bytes(lw.geo_mlp_scale, "geo_mlp_scale")[0],
                            F32Bytes(lw.geo_mlp_bias, "geo_mlp_bias")[0],
                            l);
      for (int64_t i = 0; i < p.hidden; ++i) x[i] = xg2[i];
    }
    if (t >= static_cast<int64_t>(token_ids.size()) - 1) {
      // lm_head + argmax (final_norm=False)
      int64_t best = 0;
      double bestv = -1e300;
      for (int64_t vv = 0; vv < p.vocab; ++vv) {
        double s = 0.0;
        const size_t off = (static_cast<size_t>(vv) * p.hidden) * 2;
        for (int64_t i = 0; i < p.hidden; ++i) {
          const uint32_t bits = static_cast<uint32_t>(w.head.bytes.data()[off + i * 2]) |
                                (static_cast<uint32_t>(w.head.bytes.data()[off + i * 2 + 1]) << 8);
          s += double(Bf16ToF32(static_cast<uint16_t>(bits))) * x[i];
        }
        if (s > bestv) { bestv = s; best = vv; }
      }
      // append the argmax after every step from the last prompt position
      // through total-2, so each generated token is consumed by the next
      // step and token_ids grows by exactly n_gen.
      if (t >= static_cast<int64_t>(token_ids.size()) - 1 &&
          t < total - 1) {
        token_ids.push_back(best);
      }
    }
  }
  // drop the prompt back out? caller keeps prompt + generated; token_ids grew
  // by n_gen exactly.
  (void)logits;
}

}  // namespace vllm
