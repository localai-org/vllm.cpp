// Gemma-4 USM-Conformer AUDIO tower forward — MODEL-GEMMA4 G3.
//
// Ported 1:1 from transformers/models/gemma4/modeling_gemma4.py @ 5.13.1 (see the
// header for the class:line map). Pure host FLOAT32 (correctness-first, gated
// f32-vs-f32 against the dumped oracle tower reference); the device-resident bf16
// forward + the audio->text e2e merge are the named residuals, mirroring the
// G2-impl vision tower-in-isolation cadence.
#include "vllm/model_executor/models/gemma4_audio.h"

#include <algorithm>
#include <cmath>
#include <vector>

namespace vllm::multimodal {
namespace {

inline float Clampf(float v, float lo, float hi) { return std::min(std::max(v, lo), hi); }
inline float Siluf(float x) { return x / (1.0f + std::exp(-x)); }
inline float Sigmoidf(float x) { return 1.0f / (1.0f + std::exp(-x)); }
inline float Softplusf(float x) {
  // numerically-stable log(1+exp(x))
  return x > 20.0f ? x : std::log1p(std::exp(x));
}

// out[M,N] = x[M,K] @ W[N,K]^T (+ bias[N] if given). Row-major, f64 accumulate.
std::vector<float> Linear(const std::vector<float>& x, int64_t M, int64_t K,
                          const std::vector<float>& w, int64_t N, const std::vector<float>* bias) {
  std::vector<float> out(static_cast<size_t>(M) * N);
  for (int64_t m = 0; m < M; ++m) {
    const float* xr = &x[static_cast<size_t>(m) * K];
    float* orow = &out[static_cast<size_t>(m) * N];
    for (int64_t n = 0; n < N; ++n) {
      const float* wr = &w[static_cast<size_t>(n) * K];
      double acc = bias != nullptr ? static_cast<double>((*bias)[static_cast<size_t>(n)]) : 0.0;
      for (int64_t kk = 0; kk < K; ++kk) acc += static_cast<double>(xr[kk]) * wr[kk];
      orow[n] = static_cast<float>(acc);
    }
  }
  return out;
}

// Gemma4ClippableLinear: clamp(x,in) -> Linear(bias-free) -> clamp(out).
std::vector<float> ClipLin(const std::vector<float>& x, int64_t M, int64_t K, const ClipLinear& cl,
                           int64_t N) {
  std::vector<float> cx(x.size());
  for (size_t i = 0; i < x.size(); ++i) cx[i] = Clampf(x[i], cl.clip.in_min, cl.clip.in_max);
  std::vector<float> out = Linear(cx, M, K, cl.w, N, nullptr);
  for (auto& v : out) v = Clampf(v, cl.clip.out_min, cl.clip.out_max);
  return out;
}

// Gemma4RMSNorm: x.float(), mean(x^2)+eps, *pow(-0.5), *weight (or none).
void RmsNorm(std::vector<float>& x, int64_t M, int64_t H, const std::vector<float>* w, float eps) {
  for (int64_t m = 0; m < M; ++m) {
    float* r = &x[static_cast<size_t>(m) * H];
    double ss = 0.0;
    for (int64_t h = 0; h < H; ++h) ss += static_cast<double>(r[h]) * r[h];
    const double inv = std::pow(ss / static_cast<double>(H) + static_cast<double>(eps), -0.5);
    for (int64_t h = 0; h < H; ++h) {
      float v = static_cast<float>(static_cast<double>(r[h]) * inv);
      if (w != nullptr) v *= (*w)[static_cast<size_t>(h)];
      r[h] = v;
    }
  }
}

// nn.LayerNorm(C, eps, elementwise_affine=True, bias=False) over the last dim.
void LayerNormNoBias(std::vector<float>& x, int64_t M, int64_t C, const std::vector<float>& w,
                     float eps) {
  for (int64_t m = 0; m < M; ++m) {
    float* r = &x[static_cast<size_t>(m) * C];
    double mean = 0.0;
    for (int64_t c = 0; c < C; ++c) mean += r[c];
    mean /= static_cast<double>(C);
    double var = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      const double d = static_cast<double>(r[c]) - mean;
      var += d * d;
    }
    var /= static_cast<double>(C);
    const double inv = 1.0 / std::sqrt(var + static_cast<double>(eps));
    for (int64_t c = 0; c < C; ++c)
      r[c] = static_cast<float>((static_cast<double>(r[c]) - mean) * inv) * w[static_cast<size_t>(c)];
  }
}

// One Conv2d(in_ch->out_ch, k3, stride2, padding1, bias-free) with an optional
// per-time-row valid mask (masked rows zeroed BEFORE the conv, matching upstream).
// Input  x  : [in_ch, Hin, Win] row-major (channel-major)
// Weight cw : [out_ch, in_ch, 3, 3]
// Output    : [out_ch, Hout, Wout], Hout=(Hin+2-3)/2+1, Wout=(Win+2-3)/2+1.
std::vector<float> Conv2dK3S2P1(const std::vector<float>& x, int64_t in_ch, int64_t Hin, int64_t Win,
                                const std::vector<float>& cw, int64_t out_ch, int64_t& Hout,
                                int64_t& Wout, const std::vector<int32_t>* row_mask) {
  Hout = (Hin + 2 - 3) / 2 + 1;
  Wout = (Win + 2 - 3) / 2 + 1;
  std::vector<float> out(static_cast<size_t>(out_ch) * Hout * Wout, 0.0f);
  const int64_t pad = 1, stride = 2;
  for (int64_t oc = 0; oc < out_ch; ++oc) {
    for (int64_t oh = 0; oh < Hout; ++oh) {
      for (int64_t ow = 0; ow < Wout; ++ow) {
        double acc = 0.0;
        for (int64_t ic = 0; ic < in_ch; ++ic) {
          for (int64_t kh = 0; kh < 3; ++kh) {
            const int64_t ih = oh * stride - pad + kh;
            if (ih < 0 || ih >= Hin) continue;
            const float mrow =
                (row_mask != nullptr && (*row_mask)[static_cast<size_t>(ih)] == 0) ? 0.0f : 1.0f;
            if (mrow == 0.0f) continue;
            for (int64_t kw = 0; kw < 3; ++kw) {
              const int64_t iw = ow * stride - pad + kw;
              if (iw < 0 || iw >= Win) continue;
              const double xv = static_cast<double>(
                  x[((static_cast<size_t>(ic) * Hin + ih) * Win) + iw]);
              const double wv = static_cast<double>(
                  cw[(((static_cast<size_t>(oc) * in_ch + ic) * 3 + kh) * 3) + kw]);
              acc += xv * wv;
            }
          }
        }
        out[((static_cast<size_t>(oc) * Hout + oh) * Wout) + ow] = static_cast<float>(acc);
      }
    }
  }
  return out;
}

}  // namespace

std::vector<float> Gemma4AudioForward(const std::vector<float>& input_features, int64_t T,
                                      const std::vector<int32_t>& feature_mask,
                                      const Gemma4AudioWeights& w, const Gemma4AudioConfig& cfg,
                                      Gemma4AudioCapture* cap,
                                      const std::vector<float>* embed_projection) {
  const int64_t H = cfg.hidden_size;
  const int64_t nh = cfg.num_heads;
  const int64_t hd = cfg.head_dim;
  const int64_t F = cfg.feature_size;
  const float eps = cfg.rms_norm_eps;
  // HF a08ace4 modeling_gemma4.py:198,423-424,535-537: these block
  // norms use Gemma4RMSNorm's default, unlike frontend/light-conv/connector.
  constexpr float block_eps = 1e-6f;

  // ---- mask over the T mel frames (1=valid). Absent => all valid. ------------
  std::vector<int32_t> mask0(static_cast<size_t>(T), 1);
  if (!feature_mask.empty())
    for (int64_t t = 0; t < T && t < static_cast<int64_t>(feature_mask.size()); ++t)
      mask0[static_cast<size_t>(t)] = feature_mask[static_cast<size_t>(t)];

  // ================= Subsample conv projection ================================
  // input_features [T,F] -> unsqueeze(1) -> [1(ch),T,F]. layer0 conv (1->128).
  int64_t H0 = 0, W0 = 0;
  std::vector<float> c0 =
      Conv2dK3S2P1(input_features, /*in_ch=*/1, T, F, w.sub0_conv, cfg.sub_ch0, H0, W0, &mask0);
  // norm over the CHANNEL dim: permute to [H0,W0,ch] then LayerNorm(ch), ReLU.
  {
    std::vector<float> perm(static_cast<size_t>(H0) * W0 * cfg.sub_ch0);
    for (int64_t c = 0; c < cfg.sub_ch0; ++c)
      for (int64_t h = 0; h < H0; ++h)
        for (int64_t wv = 0; wv < W0; ++wv)
          perm[((static_cast<size_t>(h) * W0 + wv) * cfg.sub_ch0) + c] =
              c0[((static_cast<size_t>(c) * H0 + h) * W0) + wv];
    LayerNormNoBias(perm, H0 * W0, cfg.sub_ch0, w.sub0_norm, eps);
    for (auto& v : perm) v = std::max(v, 0.0f);  // ReLU
    // permute back to [ch,H0,W0]
    for (int64_t c = 0; c < cfg.sub_ch0; ++c)
      for (int64_t h = 0; h < H0; ++h)
        for (int64_t wv = 0; wv < W0; ++wv)
          c0[((static_cast<size_t>(c) * H0 + h) * W0) + wv] =
              perm[((static_cast<size_t>(h) * W0 + wv) * cfg.sub_ch0) + c];
  }
  // downsample the mask: mask[::2] over the time (H) axis.
  std::vector<int32_t> mask1(static_cast<size_t>(H0), 1);
  for (int64_t h = 0; h < H0; ++h) mask1[static_cast<size_t>(h)] = mask0[static_cast<size_t>(h) * 2];

  // layer1 conv (128->32).
  int64_t H1 = 0, W1 = 0;
  std::vector<float> c1 =
      Conv2dK3S2P1(c0, cfg.sub_ch0, H0, W0, w.sub1_conv, cfg.sub_ch1, H1, W1, &mask1);
  {
    std::vector<float> perm(static_cast<size_t>(H1) * W1 * cfg.sub_ch1);
    for (int64_t c = 0; c < cfg.sub_ch1; ++c)
      for (int64_t h = 0; h < H1; ++h)
        for (int64_t wv = 0; wv < W1; ++wv)
          perm[((static_cast<size_t>(h) * W1 + wv) * cfg.sub_ch1) + c] =
              c1[((static_cast<size_t>(c) * H1 + h) * W1) + wv];
    LayerNormNoBias(perm, H1 * W1, cfg.sub_ch1, w.sub1_norm, eps);
    for (auto& v : perm) v = std::max(v, 0.0f);
    for (int64_t c = 0; c < cfg.sub_ch1; ++c)
      for (int64_t h = 0; h < H1; ++h)
        for (int64_t wv = 0; wv < W1; ++wv)
          c1[((static_cast<size_t>(c) * H1 + h) * W1) + wv] =
              perm[((static_cast<size_t>(h) * W1 + wv) * cfg.sub_ch1) + c];
  }

  // reshape: permute(0,2,3,1)=[S,W1,ch] then reshape [S, W1*ch]. S=H1.
  const int64_t S = H1;
  const int64_t proj_in = W1 * cfg.sub_ch1;  // (F/4)*32 = 1024
  std::vector<float> flat(static_cast<size_t>(S) * proj_in);
  for (int64_t s = 0; s < S; ++s)
    for (int64_t wv = 0; wv < W1; ++wv)
      for (int64_t c = 0; c < cfg.sub_ch1; ++c)
        flat[((static_cast<size_t>(s) * W1 + wv) * cfg.sub_ch1) + c] =
            c1[((static_cast<size_t>(c) * H1 + s) * W1) + wv];
  std::vector<float> hidden = Linear(flat, S, proj_in, w.input_proj, H, nullptr);  // [S,H]
  if (cap != nullptr) cap->subsample_out = hidden;

  // ================= Relative positional encoding =============================
  // inv_timescales(H/2), position_ids=arange(ctx//2,-1,-1) [P], pos=[sin|cos].
  const int64_t ctx = cfg.context_size();
  const int64_t P = ctx / 2 + 1;
  const int64_t nts = H / 2;
  const double log_inc =
      std::log(10000.0 / 1.0) /
      std::max<double>(static_cast<double>(nts - 1), 1.0);
  std::vector<double> inv_ts(static_cast<size_t>(nts));
  for (int64_t i = 0; i < nts; ++i) inv_ts[static_cast<size_t>(i)] = std::exp(i * -log_inc);
  std::vector<float> posemb(static_cast<size_t>(P) * H);
  for (int64_t p = 0; p < P; ++p) {
    const double pid = static_cast<double>(ctx / 2 - p);  // arange(ctx//2,-1,-1)
    float* row = &posemb[static_cast<size_t>(p) * H];
    for (int64_t i = 0; i < nts; ++i) {
      const double a = pid * inv_ts[static_cast<size_t>(i)];
      row[i] = static_cast<float>(std::sin(a));
      row[nts + i] = static_cast<float>(std::cos(a));
    }
  }
  if (cap != nullptr) cap->position_embeddings = posemb;

  // ================= Conformer layers ========================================
  const int64_t chunk = cfg.chunk_size;
  const int64_t past = cfg.max_past_horizon();  // 12
  const float softcap = cfg.attention_logit_cap;
  const float invalid = cfg.attention_invalid_logits_value;
  const float grad_clip = static_cast<float>(std::min(cfg.gradient_clipping, 3.4e38));
  const double q_scale = (std::pow(static_cast<double>(hd), -0.5)) / std::log(2.0);
  const double k_scale = std::log(1.0 + std::exp(1.0)) / std::log(2.0);
  const int64_t num_blocks = (S + chunk - 1) / chunk;

  // relative bias: rel_key = relative_k_proj(posemb) -> [P, nh, hd], per-layer.
  std::vector<float> rel_key;

  auto ff_forward = [&](std::vector<float>& x, const Gemma4AudioFFWeights& fw) {
    std::vector<float> residual = x;
    for (auto& v : x) v = Clampf(v, -grad_clip, grad_clip);
    RmsNorm(x, S, H, &fw.pre_ln, block_eps);
    std::vector<float> h1 = ClipLin(x, S, H, fw.ffw1, cfg.ffn_dim());
    for (auto& v : h1) v = Siluf(v);
    std::vector<float> h2 = ClipLin(h1, S, cfg.ffn_dim(), fw.ffw2, H);
    for (auto& v : h2) v = Clampf(v, -grad_clip, grad_clip);
    RmsNorm(h2, S, H, &fw.post_ln, block_eps);
    for (size_t i = 0; i < x.size(); ++i)
      x[i] = h2[i] * cfg.residual_weight + residual[i];
  };

  for (int64_t li = 0; li < cfg.num_layers; ++li) {
    const Gemma4AudioLayerWeights& lw = w.layers[static_cast<size_t>(li)];

    // ff1 (half-step)
    ff_forward(hidden, lw.ff1);

    // ---- attention block: residual + norm_pre_attn ----
    std::vector<float> residual = hidden;
    std::vector<float> x = hidden;
    for (auto& v : x) v = Clampf(v, -grad_clip, grad_clip);
    RmsNorm(x, S, H, &lw.norm_pre_attn, block_eps);

    // q,k,v projections [S,H], view [S,nh,hd].
    std::vector<float> qv = ClipLin(x, S, H, lw.attn.q_proj, H);
    std::vector<float> kv = ClipLin(x, S, H, lw.attn.k_proj, H);
    std::vector<float> vv = ClipLin(x, S, H, lw.attn.v_proj, H);
    // q *= q_scale * softplus(per_dim_scale)[d]; k *= k_scale.
    std::vector<float> pds(static_cast<size_t>(hd));
    for (int64_t d = 0; d < hd; ++d)
      pds[static_cast<size_t>(d)] =
          static_cast<float>(q_scale * Softplusf(lw.attn.per_dim_scale[static_cast<size_t>(d)]));
    for (int64_t s = 0; s < S; ++s)
      for (int64_t h = 0; h < nh; ++h)
        for (int64_t d = 0; d < hd; ++d) {
          qv[((static_cast<size_t>(s) * nh + h) * hd) + d] *= pds[static_cast<size_t>(d)];
          kv[((static_cast<size_t>(s) * nh + h) * hd) + d] *= static_cast<float>(k_scale);
        }
    // rel_key for THIS layer: relative_k_proj(posemb) -> [P, nh, hd].
    rel_key = Linear(posemb, P, H, lw.attn.relative_k_proj, H, nullptr);

    // attn output [S,H].
    std::vector<float> attn_out(static_cast<size_t>(S) * H, 0.0f);
    std::vector<float> aw(static_cast<size_t>(ctx));
    for (int64_t b = 0; b < num_blocks; ++b) {
      for (int64_t c = 0; c < chunk; ++c) {
        const int64_t qi = b * chunk + c;
        if (qi >= S) continue;
        for (int64_t h = 0; h < nh; ++h) {
          const float* qrow = &qv[((static_cast<size_t>(qi) * nh + h) * hd)];
          // matrix_ac + matrix_bd + softcap + mask over ctx slots.
          for (int64_t j = 0; j < ctx; ++j) {
            const int64_t kj = b * chunk + j - past;  // key global idx
            // matrix_ac
            double ac = 0.0;
            if (kj >= 0 && kj < S) {
              const float* krow = &kv[((static_cast<size_t>(kj) * nh + h) * hd)];
              for (int64_t d = 0; d < hd; ++d)
                ac += static_cast<double>(qrow[d]) * krow[d];
            }
            // matrix_bd via _rel_shift: out(c,j) = M[r,jj], idx=c*ctx+j,
            // r=idx/(ctx+1), jj=idx%(ctx+1); valid iff jj<P and r<chunk.
            double bd = 0.0;
            const int64_t idx = c * ctx + j;
            const int64_t r = idx / (ctx + 1);
            const int64_t jj = idx % (ctx + 1);
            if (jj < P && r < chunk) {
              const int64_t qr = b * chunk + r;  // query for shifted row
              if (qr < S) {
                const float* qrr = &qv[((static_cast<size_t>(qr) * nh + h) * hd)];
                const float* rk = &rel_key[((static_cast<size_t>(jj) * nh + h) * hd)];
                for (int64_t d = 0; d < hd; ++d)
                  bd += static_cast<double>(qrr[d]) * rk[d];
              }
            }
            float v = static_cast<float>(ac + bd);
            v = std::tanh(v / softcap) * softcap;
            // sliding-window + padding mask (sliding_window_mask_function
            // ((context_left-1, context_right)) = (past=12, 0)): valid iff kj
            // in [0,S) and dist = qi-kj in [0, past) — i.e. kj in [qi-11, qi].
            const int64_t dist = qi - kj;
            const bool valid = (kj >= 0 && kj < S && dist >= 0 && dist < past);
            aw[static_cast<size_t>(j)] = valid ? v : invalid;
          }
          // softmax over ctx (f32)
          float mx = aw[0];
          for (int64_t j = 1; j < ctx; ++j) mx = std::max(mx, aw[static_cast<size_t>(j)]);
          double sum = 0.0;
          for (int64_t j = 0; j < ctx; ++j) {
            const double e = std::exp(static_cast<double>(aw[static_cast<size_t>(j)] - mx));
            aw[static_cast<size_t>(j)] = static_cast<float>(e);
            sum += e;
          }
          const double invs = 1.0 / sum;
          // attn_out[qi,h] = sum_j softmax_j * v[kj,h]
          float* orow = &attn_out[((static_cast<size_t>(qi) * nh + h) * hd)];
          for (int64_t j = 0; j < ctx; ++j) {
            const int64_t kj = b * chunk + j - past;
            if (kj < 0 || kj >= S) continue;
            const float p = static_cast<float>(aw[static_cast<size_t>(j)] * invs);
            const float* vr = &vv[((static_cast<size_t>(kj) * nh + h) * hd)];
            for (int64_t d = 0; d < hd; ++d) orow[d] += p * vr[d];
          }
        }
      }
    }
    // post proj (clippable) + clamp + norm_post_attn + residual add.
    std::vector<float> attn = ClipLin(attn_out, S, H, lw.attn.post, H);
    for (auto& v : attn) v = Clampf(v, -grad_clip, grad_clip);
    RmsNorm(attn, S, H, &lw.norm_post_attn, block_eps);
    for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = attn[i] + residual[i];

    // ---- light conv module ----
    {
      std::vector<float> res = hidden;
      std::vector<float> ln = hidden;
      RmsNorm(ln, S, H, &lw.lconv.pre_ln, eps);
      std::vector<float> gs = ClipLin(ln, S, H, lw.lconv.linear_start, 2 * H);  // [S,2H]
      // GLU: a * sigmoid(b), split along last dim.
      std::vector<float> g(static_cast<size_t>(S) * H);
      for (int64_t s = 0; s < S; ++s)
        for (int64_t d = 0; d < H; ++d) {
          const float a = gs[static_cast<size_t>(s) * 2 * H + d];
          const float bg = gs[static_cast<size_t>(s) * 2 * H + H + d];
          g[static_cast<size_t>(s) * H + d] = a * Sigmoidf(bg);
        }
      // depthwise causal conv1d over time: left_pad=(K-1)+1-1=K-1, per-channel.
      const int64_t K = cfg.conv_kernel_size;
      const int64_t left_pad = K - 1;
      std::vector<float> conv(static_cast<size_t>(S) * H, 0.0f);
      for (int64_t d = 0; d < H; ++d) {
        const float* kern = &lw.lconv.depthwise[static_cast<size_t>(d) * K];  // [1,K]->[K]
        for (int64_t s = 0; s < S; ++s) {
          double acc = 0.0;
          for (int64_t kk = 0; kk < K; ++kk) {
            const int64_t ti = s - left_pad + kk;  // causal
            if (ti < 0 || ti >= S) continue;
            acc += static_cast<double>(g[static_cast<size_t>(ti) * H + d]) * kern[kk];
          }
          conv[static_cast<size_t>(s) * H + d] = static_cast<float>(acc);
        }
      }
      for (auto& v : conv) v = Clampf(v, -grad_clip, grad_clip);
      RmsNorm(conv, S, H, &lw.lconv.conv_norm, eps);
      for (auto& v : conv) v = Siluf(v);
      std::vector<float> le = ClipLin(conv, S, H, lw.lconv.linear_end, H);
      for (size_t i = 0; i < hidden.size(); ++i) hidden[i] = le[i] + res[i];
    }

    // ff2 (half-step)
    ff_forward(hidden, lw.ff2);

    // norm_out (clamp + RMSNorm)
    for (auto& v : hidden) v = Clampf(v, -grad_clip, grad_clip);
    RmsNorm(hidden, S, H, &lw.norm_out, block_eps);

    if (cap != nullptr) {
      if (li == 0) cap->block0 = hidden;
      if (li == cfg.num_layers / 2) cap->block_mid = hidden;
      if (li == cfg.num_layers - 1) cap->block_last = hidden;
    }
  }

  // ================= output_proj (Linear WITH bias, 1024->1536) ===============
  std::vector<float> out_proj =
      Linear(hidden, S, H, w.output_proj_w, cfg.output_proj_dims, &w.output_proj_b);
  if (cap != nullptr) cap->output_proj = out_proj;

  // ================= embed_audio projector: RMSNorm(no-weight) -> Linear ======
  std::vector<float> normed = out_proj;
  RmsNorm(normed, S, cfg.output_proj_dims, nullptr, eps);
  std::vector<float> projected =
      Linear(normed, S, cfg.output_proj_dims, embed_projection ? *embed_projection : w.embed_proj, cfg.text_hidden_size, nullptr);
  return projected;
}

}  // namespace vllm::multimodal
