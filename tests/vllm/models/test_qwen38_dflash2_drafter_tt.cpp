// vllm.cpp original; no upstream mirror. QWEN38-DFLASH2-STAGE-C W2 — the DFlash2
// drafter's PER-ITERATION FORWARD, component-validated on the Tenstorrent P150
// against the bundle's reference math.
//
// WHAT THIS GATES. The bundle's DFlash2 drafter (extra_code/qwen38/dflash2.py,
// DFlash2Draft.propose_hidden / DFlash2Drafter.draft; repo
// mando2222/qwen3.8-27b-dflash2-vision-p300x2-q4kv @ 28add9a1) defines one
// fixed-shape per-iteration forward: an [anchor, MASK x 7] 8-row query block
// attending over context K/V projected from the target's residual taps
// (hctx = rms(fc(taps)); _hctx/_ctx_kv, dflash2.py:532-541). This test runs OUR
// chain — Qwen3DFlashModel::CombineAuxFeatures (the fc tap combine) ->
// AppendContextKVDevice (the per-row tap->K/V projection, k_norm + RoPE, into
// the D11 device-resident store) -> ForwardBlockLogitsWithDeviceKV (the 8-row
// block forward over the persistent context, through the paged seam) — on the TENSTORRENT device over the REAL pinned drafter
// checkpoint (incoai/Qwen3.8-27B-DFlash2 @ dedf8df6, model.safetensors sha256
// 67fc76d6..., W1 evidence docs/bench-evidence/qwen38-dflash2-stage-c-w1-20261114/),
// and compares the post-final-norm block hidden against an INDEPENDENT fp64
// host reference that mirrors the bundle's documented math line by line
// (dflash2.py:159-388: the fc/hnorm combine, per-layer input/post RMSNorms,
// per-head q/k norms, NeoX rotate-half RoPE theta 1e7, GQA 32q/8kv, the gated
// dynamic conv over the attention_conv/mlp_conv kernel_projection + base_kernel
// pairs, SwiGLU MLP, final norm). The artifact's five layers are all
// bidirectional sliding (is_causal=false -> causal=false, dflash2.py:64-67) with
// window 2048, and context+block = 24 rows << 2048, so the window never binds:
// every key is visible to every query (the _attention_mask rule, dflash2.py:295-303).
//
// THE BAR, AND WHY IT IS SHAPED THE WAY IT IS. The bundle ships no numeric
// fixtures (its own bring-up gate, tests/dflash2_ttnn_bringup.py, is cited by
// the module docstring at dflash2.py:5 but is not in the bundle), so the
// reference here is the bundle's MATH, computed independently — not a dumped
// tensor, and not our own device path. The band is NOT invented either: the
// bundle's own precedent gates a bf16 device chain at "PCC > 0.98"
// (model.py:1132, the prefill sub-chunk bar), and this pin's Tenstorrent matmul
// path carries a measured tf32 accumulation loss (rel_rms 2.62e-2 per GEMM;
// test_flint_tt_w2_micro.cpp, P150, M=8 K=N=2560) that the fp64 reference does
// not share. The gate is therefore PCC >= 0.98 on EVERY draft row — the
// bundle's own band — with the measured PCC and rel_rms PRINTED for the
// record. The second case is the RED proof: with one fc output row's sign
// flipped, the same gate must collapse, so a wrong drafter cannot pass silently.
//
// Synthetic, deterministic embed_tokens + lm_head (the draft checkpoint shares
// them with the target, and the target is out of W2 scope): the gate compares
// the drafter's HIDDEN states, which depend on the drafter weights only; the
// block forward's logits are computed through the head and discarded. The
// tables are filled from an index hash so both arms (ours and the reference,
// which reads the same rows) see identical values.
//
// Skips loudly without the pinned checkpoint or a Tenstorrent device; it never
// runs on another backend and never reports a silent pass.

#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/model_executor/models/qwen3_dflash.h"
#include "vt/backend.h"
// The TT residency helper comes from the PUBLIC, ttnn-free header (it carries
// a non-TT no-op stub, so this always-compiled test links in every configure) —
// forward-declaring it here is what broke the CPU-only lanes
// (ISSUE-LOCAL-01M7W2ZKR8HRBYJF6HCPP46AGH). The internal header that also
// declares it pulls ttnn headers and stays out of this TU.
#include "vt/tenstorrent/tenstorrent_device.h"

namespace fs = std::filesystem;
using namespace vllm;

namespace {

// ---- the artifact geometry (the checkpoint's own config.json) ----
constexpr int64_t kH = 5120;
constexpr int64_t kNH = 32;
constexpr int64_t kNKV = 8;
constexpr int64_t kHD = 128;
constexpr int64_t kLayers = 5;
constexpr int64_t kIV = 17408;
constexpr int64_t kTaps = 5;
constexpr int64_t kVocab = 248320;
constexpr int32_t kMaskId = 248070;
constexpr int32_t kAnchorId = 123456;
constexpr double kRopeTheta = 1e7;
constexpr int64_t kCtx = 16;   // context rows (taps)
constexpr int64_t kBlock = 8;  // [anchor, MASK x 7]
constexpr double kPccFloor = 0.98;  // the bundle's own band (model.py:1132)

std::string WeightsDir() {
  const char* env = std::getenv("VT_QWEN38_DFLASH2_DRAFTER");
  return (env != nullptr && *env != '\0') ? std::string(env)
                                          : std::string("/mnt/models/qwen38-dflash2-drafter");
}

bool HasTT() { return vt::TryGetBackend(vt::DeviceType::kTENSTORRENT) != nullptr; }

// ---- bf16 <-> f32 ----
float Bf16BitsToFloat(uint16_t b) {
  uint32_t u = static_cast<uint32_t>(b) << 16;
  float f;
  std::memcpy(&f, &u, 4);
  return f;
}

uint16_t FloatToBf16(float f) {
  uint32_t u;
  std::memcpy(&u, &f, 4);
  const uint32_t lsb = (u >> 16) & 1;
  u += 0x7fff + lsb;
  return static_cast<uint16_t>(u >> 16);
}

std::vector<float> Bf16Round(const std::vector<float>& v) {
  std::vector<float> out(v.size());
  for (size_t i = 0; i < v.size(); ++i) out[i] = Bf16BitsToFloat(FloatToBf16(v[i]));
  return out;
}

// Ties-to-even bf16 add (the CPU kernel's `round(b + d)`).
float Bf16AddEven(float a, float b) {
  return Bf16BitsToFloat(FloatToBf16(a + b));
}
// Ties-away bf16 add, emulating a round-half-up f32->bf16 conversion, for the
// parity diagnosis only.
float Bf16AddAway(float a, float b) {
  uint32_t u;
  const float s = a + b;
  std::memcpy(&u, &s, 4);
  const bool neg = (u >> 31) != 0;
  uint32_t m = neg ? (u & 0x7fffffff) : u;
  const uint32_t lsb = (m >> 16) & 1;
  m = m + 0x7fff + (lsb ^ 1);  // force ties up
  const uint32_t bits = (neg ? (1u << 31) : 0u) | (m >> 16 << 16 ? (m & 0xffff0000u) : (m & 0xffff0000u));
  (void)bits;
  float f;
  const uint32_t out = (neg ? (1u << 31) : 0u) | ((m >> 16) << 16);
  std::memcpy(&f, &out, 4);
  return f;
}

// ---- read a named BF16 tensor out of the artifact checkpoint ----
std::vector<float> ReadTensor(const std::vector<SafetensorsFile>& shards,
                              const std::string& name, std::vector<int64_t>* shape) {
  for (const SafetensorsFile& s : shards) {
    for (const std::string& n : s.Names()) {
      if (n != name) continue;
      const StTensor& t = s.Get(name);
      REQUIRE_MESSAGE(t.dtype == "BF16", "expected BF16 tensor " << name << ", got " << t.dtype);
      if (shape != nullptr) *shape = t.shape;
      std::vector<float> out(t.nbytes / 2);
      for (size_t i = 0; i < out.size(); ++i) {
        uint16_t b;
        std::memcpy(&b, t.data + 2 * i, 2);
        out[i] = Bf16BitsToFloat(b);
      }
      return out;
    }
  }
  FAIL("tensor not found in drafter checkpoint: " << name);
  return {};
}

// ---- tiny fp64 host reference (mirrors dflash2.py; NEVER our device path) ----
using Mat = std::vector<double>;

// The real chain materializes bf16 after every step (the bundle's ttnn chain and
// our DBuf chain both do), so the reference rounds each stored intermediate to
// bf16 — without this the reference is not a valid band anchor wherever the
// conv's k = base + coefficient cancellation makes the output pure rounding
// noise (the [0,406] probe: b0=1, d0=-1.0035 -> k=-0.0035, a 0.4% bf16 rounding
// of the coefficient is a 100% rounding of k).
double RoundB(double v) {
  return static_cast<double>(Bf16BitsToFloat(FloatToBf16(static_cast<float>(v))));
}
Mat RoundBAll(Mat m) {
  for (double& v : m) v = RoundB(v);
  return m;
}

Mat FromF32(const std::vector<float>& v) {
  Mat m(v.size());
  for (size_t i = 0; i < v.size(); ++i) m[i] = static_cast<double>(v[i]);
  return m;
}

// y[T,N] = x[T,K] @ W^T, where W is the checkpoint's [N,K] row-major tensor
// (bundle _lin stores `weight.T.contiguous()` so ttnn.linear(x, w) = x @ W.T;
// the checkpoint tensor is [out, in]).
Mat MulWT(const Mat& x, int64_t T, int64_t K, const Mat& w, int64_t N) {
  Mat y(static_cast<size_t>(T * N), 0.0);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t n = 0; n < N; ++n) {
      const double* wr = &w[static_cast<size_t>(n * K)];
      const double* xr = &x[static_cast<size_t>(t * K)];
      double acc = 0.0;
      for (int64_t k = 0; k < K; ++k) acc += xr[k] * wr[k];
      y[static_cast<size_t>(t * N + n)] = acc;
    }
  return y;
}

// RMSNorm rows of x[T,C] with weight w[C], epsilon 1e-6 (dflash2.py:225-227)
Mat Rms(const Mat& x, int64_t T, int64_t C, const Mat& w) {
  Mat y = x;
  for (int64_t t = 0; t < T; ++t) {
    double ss = 0.0;
    for (int64_t c = 0; c < C; ++c) {
      const double v = x[static_cast<size_t>(t * C + c)];
      ss += v * v;
    }
    const double inv = 1.0 / std::sqrt(ss / static_cast<double>(C) + 1e-6);
    for (int64_t c = 0; c < C; ++c)
      y[static_cast<size_t>(t * C + c)] =
          x[static_cast<size_t>(t * C + c)] * inv * w[static_cast<size_t>(c)];
  }
  return y;
}

// NeoX rotate-half RoPE tables for absolute positions [0, T) (dflash2.py:100-106)
void RopeTables(int64_t T, Mat* cos, Mat* sin) {
  cos->assign(static_cast<size_t>(T * kHD), 0.0);
  sin->assign(static_cast<size_t>(T * kHD), 0.0);
  for (int64_t p = 0; p < T; ++p)
    for (int64_t i = 0; i < kHD / 2; ++i) {
      const double inv =
          1.0 / std::pow(kRopeTheta, static_cast<double>(2 * i) / static_cast<double>(kHD));
      const double f = static_cast<double>(p) * inv;
      const double c = std::cos(f), s = std::sin(f);
      (*cos)[static_cast<size_t>(p * kHD + i)] = c;
      (*cos)[static_cast<size_t>(p * kHD + i + kHD / 2)] = c;
      (*sin)[static_cast<size_t>(p * kHD + i)] = s;
      (*sin)[static_cast<size_t>(p * kHD + i + kHD / 2)] = s;
    }
}

// x[T,D]*cos + rot_half(x)[T,D]*sin at absolute positions row_offset.. (dflash2.py:256-266;
// rot_half = concat(-x[d:], x[:d]))
Mat RopeApply(const Mat& x, int64_t T, int64_t D, const Mat& cos, const Mat& sin,
              int64_t row_offset) {
  Mat y(static_cast<size_t>(T * D), 0.0);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t d = 0; d < D; ++d) {
      const double v = x[static_cast<size_t>(t * D + d)];
      const double rot = (d < D / 2) ? -x[static_cast<size_t>(t * D + d + D / 2)]
                                     : x[static_cast<size_t>(t * D + d - D / 2)];
      y[static_cast<size_t>(t * D + d)] =
          v * cos[static_cast<size_t>((row_offset + t) * D + d)] +
          rot * sin[static_cast<size_t>((row_offset + t) * D + d)];
    }
  return y;
}

// The gated dynamic conv (dflash2.py:267-274): values[T,H] shaped by the
// projection kp[T,1280]; selection `sel` reads kp columns [sel*640, sel*640+640)
// split into two 320-group halves, each expanded to H through the group
// expansion E[320,H] (group g fills columns [g*16,(g+1)*16)), plus the
// base_kernel row; out = c0 * values + c1 * shift(values), shift = one row back
// (row 0 -> 0, the _shift matrix dflash2.py:93-98).
Mat Gdc(const Mat& values, int64_t T, const Mat& kp, const Mat& b0, const Mat& b1,
        int64_t sel) {
  const int64_t off = sel * 640;
  Mat y(static_cast<size_t>(T * kH), 0.0);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t j = 0; j < kH; ++j) {
      const int64_t g = j / 16;
      // the op's per-step bf16 rounding: k = round_bf16(round_bf16(b) +
      // round_bf16(d)), term and accumulation rounded (cpu_ops.cpp
      // DFlashGroupedConvKernel). EACH OPERAND is bf16 before the add — the
      // conv probe proved the distinction: b=1, d=-1.0035 gives bf16(d)=-1.0
      // so the op's k is EXACTLY 0, while round(1-1.0035) = -0.0035 (a 100%
      // error of k wherever b+d cancels).
      const double k0 =
          RoundB(RoundB(b0[static_cast<size_t>(j)]) +
                 RoundB(kp[static_cast<size_t>(t * 1280 + off + g)]));
      const double k1 =
          RoundB(RoundB(b1[static_cast<size_t>(j)]) +
                 RoundB(kp[static_cast<size_t>(t * 1280 + off + 320 + g)]));
      const double v = values[static_cast<size_t>(t * kH + j)];
      const double shifted = (t > 0) ? values[static_cast<size_t>((t - 1) * kH + j)] : 0.0;
      const double acc0 = RoundB(k0 * v);
      const int64_t pos = t % 8;  // the power-of-two block masking
      double acc = acc0;
      if (pos >= 1) acc = RoundB(acc0 + RoundB(k1 * shifted));
      y[static_cast<size_t>(t * kH + j)] = acc;
    }
  return y;
}

// Per-head RMSNorm over head_dim: x[T, heads*HD] with weight w[HD]
Mat HeadRms(const Mat& x, int64_t T, int64_t heads, const Mat& w) {
  Mat y = x;
  for (int64_t t = 0; t < T; ++t)
    for (int64_t h = 0; h < heads; ++h) {
      double ss = 0.0;
      for (int64_t d = 0; d < kHD; ++d) {
        const double v = x[static_cast<size_t>((t * heads + h) * kHD + d)];
        ss += v * v;
      }
      const double inv = 1.0 / std::sqrt(ss / static_cast<double>(kHD) + 1e-6);
      for (int64_t d = 0; d < kHD; ++d)
        y[static_cast<size_t>((t * heads + h) * kHD + d)] =
            x[static_cast<size_t>((t * heads + h) * kHD + d)] * inv * w[static_cast<size_t>(d)];
    }
  return y;
}

struct RefLayer {
  Mat in_ln, post_ln, qn, kn, q, k, v, o;
  Mat ac_kp, ac_b00, ac_b01, ac_b10, ac_b11;
  Mat gate, up, down, mc_kp, mc_b00, mc_b01, mc_b10, mc_b11;
};

struct RefWeights {
  Mat fc;     // [H, H*kTaps]
  Mat hnorm;  // [H]
  Mat fnorm;  // [H]
  std::vector<RefLayer> layers;
};

// RefLayerForward call counter for the W2_REF_INT_DUMP bisect
int g_ref_layer_calls = 0;

// One reference layer over the block rows, attending the whole [context; block]
// window: mirrors dflash2.py _layer/_attn/_mlp (:275-352).
Mat RefLayerForward(const RefLayer& lw, const Mat& hctx, const Mat& hb, const Mat& cos,
                    const Mat& sin, int64_t pos0, int64_t ctx_rows = kCtx,
                    Mat* pre_residual_down = nullptr) {
  const int64_t C = ctx_rows, B = kBlock, T = C + B;
  static const bool no_conv = std::getenv("W2_REF_NO_CONV") != nullptr;
  static const int dump_idx = [] { return std::getenv("W2_REF_INT_DUMP") ? 0 : -1; }();
  const bool dbg = dump_idx >= 0 && 6 == g_ref_layer_calls++;
  Mat hn = Rms(hb, B, kH, lw.in_ln);
  hn = RoundBAll(hn);
  Mat kp = MulWT(hn, B, kH, lw.ac_kp, 1280);  // Gdc rounds each conv operand itself
  Mat attn_in = no_conv ? hn : Gdc(hn, B, kp, lw.ac_b00, lw.ac_b01, 0);
  auto dumpi = [&](const char* tag, const Mat& m) {
    if (!dbg) return;
    std::FILE* fp = std::fopen((std::string("/tmp/w2-int-") + tag + ".bin").c_str(), "wb");
    for (double v : m) {
      const float f = static_cast<float>(v);
      std::fwrite(&f, 4, 1, fp);
    }
    std::fclose(fp);
  };
  dumpi("attn_in", attn_in);

  Mat q = RoundBAll(MulWT(attn_in, B, kH, lw.q, kNH * kHD));
  Mat kcat(static_cast<size_t>(T * kNKV * kHD)), vcat(static_cast<size_t>(T * kNKV * kHD));
  {
    Mat kc = RoundBAll(MulWT(hctx, C, kH, lw.k, kNKV * kHD));
    Mat kb = RoundBAll(MulWT(attn_in, B, kH, lw.k, kNKV * kHD));
    Mat vc = RoundBAll(MulWT(hctx, C, kH, lw.v, kNKV * kHD));
    Mat vb = RoundBAll(MulWT(attn_in, B, kH, lw.v, kNKV * kHD));
    for (int64_t t = 0; t < T; ++t) {
      const Mat& ks = (t < C) ? kc : kb;  // kc empty when C == 0
      const Mat& vs = (t < C) ? vc : vb;
      const int64_t src = (t < C) ? t : t - C;
      std::memcpy(&kcat[static_cast<size_t>(t * kNKV * kHD)],
                  &ks[static_cast<size_t>(src * kNKV * kHD)],
                  sizeof(double) * static_cast<size_t>(kNKV * kHD));
      std::memcpy(&vcat[static_cast<size_t>(t * kNKV * kHD)],
                  &vs[static_cast<size_t>(src * kNKV * kHD)],
                  sizeof(double) * static_cast<size_t>(kNKV * kHD));
    }
  }
  // RoPE applies PER HEAD on the [T, heads*HD] layout (dflash2.py broadcasts the
  // [1,T,HD] tables over the head axis).
  auto head_rope = [&](const Mat& x, int64_t rows, int64_t heads, int64_t row_offset) {
    Mat y(static_cast<size_t>(rows * heads * kHD), 0.0);
    for (int64_t hh = 0; hh < heads; ++hh) {
      Mat slice(static_cast<size_t>(rows * kHD), 0.0);
      for (int64_t t = 0; t < rows; ++t)
        std::memcpy(&slice[static_cast<size_t>(t * kHD)],
                    &x[static_cast<size_t>((t * heads + hh) * kHD)],
                    sizeof(double) * static_cast<size_t>(kHD));
      Mat roped = RopeApply(slice, rows, kHD, cos, sin, row_offset);
      for (int64_t t = 0; t < rows; ++t)
        std::memcpy(&y[static_cast<size_t>((t * heads + hh) * kHD)],
                    &roped[static_cast<size_t>(t * kHD)],
                    sizeof(double) * static_cast<size_t>(kHD));
    }
    return y;
  };
  Mat qh = RoundBAll(head_rope(RoundBAll(HeadRms(q, B, kNH, lw.qn)), B, kNH, pos0));
  if (dbg) {
    {
      const Mat qq = RoundBAll(MulWT(attn_in, B, kH, lw.q, kNH * kHD));
      std::FILE* fp = std::fopen("/tmp/w2-int-qpre.bin", "wb");
      for (double v : qq) { const float f = static_cast<float>(v); std::fwrite(&f, 4, 1, fp); }
      std::fclose(fp);
      const Mat hh2 = RoundBAll(HeadRms(qq, B, kNH, lw.qn));
      fp = std::fopen("/tmp/w2-int-qhn.bin", "wb");
      for (double v : hh2) { const float f = static_cast<float>(v); std::fwrite(&f, 4, 1, fp); }
      std::fclose(fp);
    }
    std::FILE* fp = std::fopen("/tmp/w2-int-qh.bin", "wb");
    for (double v : qh) { const float f = static_cast<float>(v); std::fwrite(&f, 4, 1, fp); }
    std::fclose(fp);
  }
  Mat kh = RoundBAll(head_rope(RoundBAll(HeadRms(kcat, T, kNKV, lw.kn)), T, kNKV, 0));
  if (dbg) {
    std::FILE* fp = std::fopen("/tmp/w2-int-kh.bin", "wb");
    for (double v : kh) { const float f = static_cast<float>(v); std::fwrite(&f, 4, 1, fp); }
    std::fclose(fp);
  }

  // GQA attention, bidirectional full window, scale 1/sqrt(HD)
  // [B, kNH*kHD] EXACTLY — the o_proj below reads this CONTIGUOUS at row
  // stride kNH*kHD; an earlier draft allocated [B, kH] (5120) and the stride
  // mismatch corrupted rows 1..7 of every reference layer (row 0 matched,
  // rows 1+ collapsed — the exact signature the W2 gate reported).
  Mat o2d(static_cast<size_t>(B * kNH * kHD), 0.0);
  const double scale = 1.0 / std::sqrt(static_cast<double>(kHD));
  std::vector<double> scores(static_cast<size_t>(T));
  for (int64_t t = 0; t < B; ++t)
    for (int64_t h = 0; h < kNH; ++h) {
      const int64_t kvh = h / (kNH / kNKV);
      double mx = -1e300;
      for (int64_t s = 0; s < T; ++s) {
        double acc = 0.0;
        for (int64_t d = 0; d < kHD; ++d)
          acc += qh[static_cast<size_t>((t * kNH + h) * kHD + d)] *
                 kh[static_cast<size_t>((s * kNKV + kvh) * kHD + d)];
        scores[static_cast<size_t>(s)] = acc * scale;
        mx = std::max(mx, scores[static_cast<size_t>(s)]);
      }
      double denom = 0.0;
      for (int64_t s = 0; s < T; ++s) {
        scores[static_cast<size_t>(s)] = std::exp(scores[static_cast<size_t>(s)] - mx);
        denom += scores[static_cast<size_t>(s)];
      }
      for (int64_t d = 0; d < kHD; ++d) {
        double acc = 0.0;
        for (int64_t s = 0; s < T; ++s)
          acc += scores[static_cast<size_t>(s)] *
                 vcat[static_cast<size_t>((s * kNKV + kvh) * kHD + d)];
        o2d[static_cast<size_t>(t * (kNH * kHD) + h * kHD + d)] = acc / denom;
      }
    }
  Mat a = RoundBAll(MulWT(RoundBAll(o2d), B, kNH * kHD, lw.o, kH));  // o_proj [kH, kNH*kHD]
  dumpi("a", a);
  dumpi("o2d", o2d);
  static const bool no_attn = std::getenv("W2_REF_NO_ATTN") != nullptr;
  Mat acv = (no_conv || no_attn) ? Mat(static_cast<size_t>(B * kH), 0.0)
                                 : Gdc(a, B, kp, lw.ac_b10, lw.ac_b11, 1);
  Mat h1(static_cast<size_t>(B * kH));
  for (size_t i = 0; i < h1.size(); ++i) h1[i] = hb[i] + acv[i];

  Mat h2 = RoundBAll(Rms(h1, B, kH, lw.post_ln));
  Mat kpm = MulWT(h2, B, kH, lw.mc_kp, 1280);  // Gdc rounds each conv operand itself
  Mat mlp_in = no_conv ? h2 : Gdc(h2, B, kpm, lw.mc_b00, lw.mc_b01, 0);
  Mat g = RoundBAll(MulWT(mlp_in, B, kH, lw.gate, kIV));
  Mat u = RoundBAll(MulWT(mlp_in, B, kH, lw.up, kIV));
  Mat act(static_cast<size_t>(B * kIV));
  for (size_t i = 0; i < act.size(); ++i) {
    const double sv = 1.0 / (1.0 + std::exp(-g[i]));
    act[i] = RoundB(g[i] * sv * u[i]);  // silu(gate) rounds bf16, mul rounds bf16
  }
  Mat m = RoundBAll(MulWT(act, B, kIV, lw.down, kH));
  dumpi("act", act);
  dumpi("mlp_in", mlp_in);
  static const bool no_mlp = std::getenv("W2_REF_NO_MLP") != nullptr;
  Mat mcv = (no_conv || no_mlp) ? Mat(static_cast<size_t>(B * kH), 0.0)
                                : Gdc(m, B, kpm, lw.mc_b10, lw.mc_b11, 1);
  mcv = RoundBAll(mcv);
  dumpi("mcv", mcv);
  dumpi("m_pre", m);
  dumpi("h2", h2);
  if (pre_residual_down != nullptr) *pre_residual_down = mcv;
  Mat out(static_cast<size_t>(B * kH));
  for (size_t i = 0; i < out.size(); ++i) out[i] = h1[i] + mcv[i];
  return out;
}

// The full reference forward: taps [C, H*kTaps] + block embed rows [B, H] ->
// post-final-norm block hidden [B, H] (dflash2.py:367-388 propose_hidden).
Mat RefForward(const RefWeights& rw, const Mat& taps, const Mat& block_rows) {
  Mat hctx = MulWT(taps, kCtx, kH * kTaps, rw.fc, kH);
  hctx = Rms(hctx, kCtx, kH, rw.hnorm);
  Mat cos, sin;
  RopeTables(kCtx + kBlock, &cos, &sin);
  Mat h = block_rows;
  for (size_t l = 0; l < rw.layers.size(); ++l)
    h = RefLayerForward(rw.layers[l], hctx, h, cos, sin, kCtx);
  return Rms(h, kBlock, kH, rw.fnorm);
}

struct PccStats {
  double pcc = 0.0;
  double rel_rms = 0.0;
};

PccStats Compare(const std::vector<float>& got, const Mat& ref, size_t begin, size_t count) {
  double sg = 0.0, sr = 0.0, sgg = 0.0, srr = 0.0, sgr = 0.0, sde = 0.0, sdr = 0.0;
  for (size_t i = 0; i < count; ++i) {
    const double g = static_cast<double>(got[begin + i]);
    const double r = ref[begin + i];
    sg += g; sr += r; sgg += g * g; srr += r * r; sgr += g * r;
    const double d = g - r;
    sde += d * d; sdr += r * r;
  }
  const double nd = static_cast<double>(count);
  PccStats s;
  const double cov = sgr - sg * sr / nd;
  const double vg = sgg - sg * sg / nd;
  const double vr = srr - sr * sr / nd;
  s.pcc = cov / std::sqrt(std::max(vg * vr, 1e-300));
  s.rel_rms = std::sqrt(sde / nd) / std::sqrt(std::max(sdr / nd, 1e-300));
  return s;
}

// Deterministic synthetic bf16 [rows, kH] table (index hash; see file header)
OwnedTensor MakeSynthTable(int64_t rows, int64_t cols) {
  OwnedTensor t;
  t.dtype = vt::DType::kBF16;
  t.rank = 2;
  t.nk = false;
  t.shape[0] = rows;
  t.shape[1] = cols;
  t.bytes.resize(static_cast<size_t>(rows * cols) * 2);
  for (int64_t r = 0; r < rows; ++r)
    for (int64_t c = 0; c < cols; ++c) {
      uint64_t h = (static_cast<uint64_t>(r) * 0x9E3779B97F4A7C15ull) ^
                   (static_cast<uint64_t>(c) * 0xC2B2AE3D27D4EB4Full);
      h ^= h >> 29; h *= 0xBF58476D1CE4E5B9ull; h ^= h >> 32;
      const float v = (static_cast<float>(h & 0xFFFF) / 32768.0f - 1.0f) * 0.02f;
      const uint16_t b = FloatToBf16(v);
      std::memcpy(t.bytes.data() + static_cast<size_t>((r * cols + c) * 2), &b, 2);
    }
  return t;
}

struct RefLayerSource {
  const char* qn;
  const char* kn;
  const char* q;
  const char* k;
  const char* v;
  const char* o;
  const char* ac_kp;
  const char* ac_bk;
  const char* in_ln;
  const char* post_ln;
  const char* gate;
  const char* up;
  const char* down;
  const char* mc_kp;
  const char* mc_bk;
};

RefWeights LoadRefWeights(const std::vector<SafetensorsFile>& shards) {
  RefWeights rw;
  std::vector<int64_t> shape;
  rw.fc = FromF32(ReadTensor(shards, "fc.weight", &shape));
  REQUIRE_EQ(shape[0], kH);
  REQUIRE_EQ(shape[1], kH * kTaps);
  rw.hnorm = FromF32(ReadTensor(shards, "hidden_norm.weight", nullptr));
  rw.fnorm = FromF32(ReadTensor(shards, "norm.weight", nullptr));
  const std::vector<RefLayerSource> names = {
      {"layers.0.self_attn.q_norm.weight", "layers.0.self_attn.k_norm.weight",
       "layers.0.self_attn.q_proj.weight", "layers.0.self_attn.k_proj.weight",
       "layers.0.self_attn.v_proj.weight", "layers.0.self_attn.o_proj.weight",
       "layers.0.attention_conv.kernel_projection.weight",
       "layers.0.attention_conv.base_kernel", "layers.0.input_layernorm.weight",
       "layers.0.post_attention_layernorm.weight", "layers.0.mlp.gate_proj.weight",
       "layers.0.mlp.up_proj.weight", "layers.0.mlp.down_proj.weight",
       "layers.0.mlp_conv.kernel_projection.weight", "layers.0.mlp_conv.base_kernel"},
      {"layers.1.self_attn.q_norm.weight", "layers.1.self_attn.k_norm.weight",
       "layers.1.self_attn.q_proj.weight", "layers.1.self_attn.k_proj.weight",
       "layers.1.self_attn.v_proj.weight", "layers.1.self_attn.o_proj.weight",
       "layers.1.attention_conv.kernel_projection.weight",
       "layers.1.attention_conv.base_kernel", "layers.1.input_layernorm.weight",
       "layers.1.post_attention_layernorm.weight", "layers.1.mlp.gate_proj.weight",
       "layers.1.mlp.up_proj.weight", "layers.1.mlp.down_proj.weight",
       "layers.1.mlp_conv.kernel_projection.weight", "layers.1.mlp_conv.base_kernel"},
      {"layers.2.self_attn.q_norm.weight", "layers.2.self_attn.k_norm.weight",
       "layers.2.self_attn.q_proj.weight", "layers.2.self_attn.k_proj.weight",
       "layers.2.self_attn.v_proj.weight", "layers.2.self_attn.o_proj.weight",
       "layers.2.attention_conv.kernel_projection.weight",
       "layers.2.attention_conv.base_kernel", "layers.2.input_layernorm.weight",
       "layers.2.post_attention_layernorm.weight", "layers.2.mlp.gate_proj.weight",
       "layers.2.mlp.up_proj.weight", "layers.2.mlp.down_proj.weight",
       "layers.2.mlp_conv.kernel_projection.weight", "layers.2.mlp_conv.base_kernel"},
      {"layers.3.self_attn.q_norm.weight", "layers.3.self_attn.k_norm.weight",
       "layers.3.self_attn.q_proj.weight", "layers.3.self_attn.k_proj.weight",
       "layers.3.self_attn.v_proj.weight", "layers.3.self_attn.o_proj.weight",
       "layers.3.attention_conv.kernel_projection.weight",
       "layers.3.attention_conv.base_kernel", "layers.3.input_layernorm.weight",
       "layers.3.post_attention_layernorm.weight", "layers.3.mlp.gate_proj.weight",
       "layers.3.mlp.up_proj.weight", "layers.3.mlp.down_proj.weight",
       "layers.3.mlp_conv.kernel_projection.weight", "layers.3.mlp_conv.base_kernel"},
      {"layers.4.self_attn.q_norm.weight", "layers.4.self_attn.k_norm.weight",
       "layers.4.self_attn.q_proj.weight", "layers.4.self_attn.k_proj.weight",
       "layers.4.self_attn.v_proj.weight", "layers.4.self_attn.o_proj.weight",
       "layers.4.attention_conv.kernel_projection.weight",
       "layers.4.attention_conv.base_kernel", "layers.4.input_layernorm.weight",
       "layers.4.post_attention_layernorm.weight", "layers.4.mlp.gate_proj.weight",
       "layers.4.mlp.up_proj.weight", "layers.4.mlp.down_proj.weight",
       "layers.4.mlp_conv.kernel_projection.weight", "layers.4.mlp_conv.base_kernel"},
  };
  for (const RefLayerSource& src : names) {
    RefLayer lw;
    lw.in_ln = FromF32(ReadTensor(shards, src.in_ln, nullptr));
    lw.post_ln = FromF32(ReadTensor(shards, src.post_ln, nullptr));
    lw.qn = FromF32(ReadTensor(shards, src.qn, nullptr));
    lw.kn = FromF32(ReadTensor(shards, src.kn, nullptr));
    lw.q = FromF32(ReadTensor(shards, src.q, &shape));
    REQUIRE_EQ(shape[0], kNH * kHD);
    REQUIRE_EQ(shape[1], kH);
    lw.k = FromF32(ReadTensor(shards, src.k, &shape));
    REQUIRE_EQ(shape[0], kNKV * kHD);
    REQUIRE_EQ(shape[1], kH);
    lw.v = FromF32(ReadTensor(shards, src.v, nullptr));
    lw.o = FromF32(ReadTensor(shards, src.o, &shape));
    REQUIRE_EQ(shape[0], kH);
    REQUIRE_EQ(shape[1], kNH * kHD);
    lw.ac_kp = FromF32(ReadTensor(shards, src.ac_kp, &shape));
    REQUIRE_EQ(shape[0], 1280);
    REQUIRE_EQ(shape[1], kH);
    lw.mc_kp = FromF32(ReadTensor(shards, src.mc_kp, nullptr));
    lw.gate = FromF32(ReadTensor(shards, src.gate, &shape));
    REQUIRE_EQ(shape[0], kIV);
    REQUIRE_EQ(shape[1], kH);
    lw.up = FromF32(ReadTensor(shards, src.up, nullptr));
    lw.down = FromF32(ReadTensor(shards, src.down, &shape));
    REQUIRE_EQ(shape[0], kH);
    REQUIRE_EQ(shape[1], kIV);
    // base_kernel [2,2,H]: rows (b00, b01 / b10, b11), dflash2.py:240-246
    const Mat bk = FromF32(ReadTensor(shards, src.ac_bk, nullptr));
    lw.ac_b00 = Mat(bk.begin(), bk.begin() + kH);
    lw.ac_b01 = Mat(bk.begin() + kH, bk.begin() + 2 * kH);
    lw.ac_b10 = Mat(bk.begin() + 2 * kH, bk.begin() + 3 * kH);
    lw.ac_b11 = Mat(bk.begin() + 3 * kH, bk.begin() + 4 * kH);
    const Mat mbk = FromF32(ReadTensor(shards, src.mc_bk, nullptr));
    lw.mc_b00 = Mat(mbk.begin(), mbk.begin() + kH);
    lw.mc_b01 = Mat(mbk.begin() + kH, mbk.begin() + 2 * kH);
    lw.mc_b10 = Mat(mbk.begin() + 2 * kH, mbk.begin() + 3 * kH);
    lw.mc_b11 = Mat(mbk.begin() + 3 * kH, mbk.begin() + 4 * kH);
    rw.layers.push_back(std::move(lw));
  }
  return rw;
}

// The shared fixture: the production loader reads the artifact; the reference
// reads the same checkpoint independently. taps are bf16-rounded f32 so both
// arms consume identical VALUES.
struct Fixture {
  HfConfig config;
  Qwen3DFlashWeights weights;
  std::vector<SafetensorsFile> shards;
  RefWeights ref;
  std::vector<float> taps;  // [kCtx, kH*kTaps]
  Mat block_rows;           // [kBlock, kH] fp64 (from the synthetic tables)
  vt::Queue* cpu_queue = nullptr;
};

// The CPU arm's fc combine buffer (host bf16-rounded matmul; no device)
std::vector<float> fc_out_storage;

// bf16 byte buffer -> uint16 view (weight-identity probes)
template <class Bytes>
std::vector<uint16_t> ToU16(const Bytes& bytes) {
  std::vector<uint16_t> v(bytes.size() / 2);
  std::memcpy(v.data(), bytes.data(), v.size() * 2);
  return v;
}

// base_kernel [2, taps, H] -> flat [taps*H] bf16 slice for one side
std::vector<uint16_t> BaseSlice(const OwnedTensor& t, int64_t side) {
  const int64_t h = t.shape[2];
  const auto* u = static_cast<const uint16_t*>(static_cast<const void*>(t.bytes.data()));
  return std::vector<uint16_t>(u + side * 2 * h, u + (side + 1) * 2 * h);
}

// The fixture WITHOUT the device: the CPU arm of the chain runs against it.
Fixture MakeFixtureCpuOnly() {
  Fixture f;
  const std::string dir = WeightsDir();
  REQUIRE_MESSAGE(fs::exists(dir + "/config.json"),
                  "the pinned drafter checkpoint is absent at " << dir
                  << " (set VT_QWEN38_DFLASH2_DRAFTER) — SKIP, not pass");
  std::ifstream cf(dir + "/config.json");
  nlohmann::json cj;
  cf >> cj;
  f.config = MakeQwen3DFlashDraftConfig(cj);
  f.shards.push_back(SafetensorsFile::Open(dir + "/model.safetensors"));
  f.weights = LoadQwen3DFlash(f.shards, f.config, kTaps, kMaskId);
  // The production loader seeds this from the resolved k (model_loader.cpp:1140,
  // draft->weights.conv_block_size = k + 1): the grouped convolution masks its
  // taps by `row index mod conv_block_size`, so W2's single 8-row block needs 8.
  f.weights.conv_block_size = kBlock;
  // W2 compares HIDDEN states; the shared head/embed stand in synthetically.
  f.weights.embed_tokens = MakeSynthTable(kVocab, kH);
  f.weights.lm_head = MakeSynthTable(kVocab, kH);
  f.weights.draft_vocab_size = kVocab;
  f.ref = LoadRefWeights(f.shards);

  // Deterministic taps: 5 residual taps concatenated, bf16-rounded.
  std::mt19937 rng(0);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  std::vector<float> raw(static_cast<size_t>(kCtx * kH * kTaps));
  for (float& v : raw) v = nd(rng);
  f.taps = Bf16Round(raw);

  // Block rows from the SAME synthetic tables: anchor + 7 mask rows.
  OwnedTensor emb = MakeSynthTable(kVocab, kH);
  auto row_of = [&](int32_t id) {
    Mat r(static_cast<size_t>(kH));
    for (int64_t c = 0; c < kH; ++c) {
      uint16_t b;
      std::memcpy(&b,
                  emb.bytes.data() +
                      static_cast<size_t>((static_cast<int64_t>(id) * kH + c) * 2), 2);
      r[static_cast<size_t>(c)] = static_cast<double>(Bf16BitsToFloat(b));
    }
    return r;
  };
  const Mat anchor = row_of(kAnchorId);
  const Mat mask = row_of(kMaskId);
  f.block_rows.resize(static_cast<size_t>(kBlock * kH));
  for (int64_t b = 0; b < kBlock; ++b) {
    const Mat& src = (b == 0) ? anchor : mask;
    std::memcpy(&f.block_rows[static_cast<size_t>(b * kH)], src.data(),
                sizeof(double) * static_cast<size_t>(kH));
  }
  return f;
}

Fixture MakeFixture() {
  Fixture f = MakeFixtureCpuOnly();
  REQUIRE_MESSAGE(HasTT(), "no Tenstorrent device on this box — SKIP, not pass");
  static vt::Backend& cpu = vt::GetBackend(vt::DeviceType::kCPU);
  static vt::Queue cq = cpu.CreateQueue();
  f.cpu_queue = &cq;
  return f;
}

// OUR chain, on the TT device, at the per-iteration shape: the taps combine,
// the DEVICE-RESIDENT context KV append (the D11 store the production runner
// holds across verify steps), then the 8-row block forward. The paged-seam
// route (ReshapeAndCache + PagedAttention, both TT-implemented) is the route
// this takes; the bespoke DFlashBlockAttention op has no TT kernel and the
// production path does not reach it on this route.
std::vector<float> RunOurForward(const Fixture& f, vt::Queue& q, Qwen3DFlashWeights& w,
                                 DflashDeviceKVStore* store) {
  auto fc_out = Qwen3DFlashModel::CombineAuxFeatures(f.taps, kCtx, w, f.config, q);
  REQUIRE(static_cast<int64_t>(fc_out.size()) == kCtx * kH);
  std::vector<int32_t> ctx_pos(static_cast<size_t>(kCtx));
  for (int64_t i = 0; i < kCtx; ++i) ctx_pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  if (Qwen3DFlashModel::DeviceKVNumCtx(*store) == 0)
    Qwen3DFlashModel::AppendContextKVDevice(*store, fc_out, ctx_pos, w, f.config, q);
  REQUIRE(Qwen3DFlashModel::DeviceKVNumCtx(*store) == kCtx);

  std::vector<int32_t> ids(static_cast<size_t>(kBlock), kMaskId);
  ids[0] = kAnchorId;
  std::vector<int32_t> bpos(static_cast<size_t>(kBlock));
  std::vector<int32_t> cu = {0, static_cast<int32_t>(kBlock)};
  std::vector<int32_t> ctx_cu = {0, static_cast<int32_t>(kCtx)};
  for (int64_t i = 0; i < kBlock; ++i)
    bpos[static_cast<size_t>(i)] = static_cast<int32_t>(kCtx + i);
  // device_out keeps the single-request PAGED branch (final_out would
  // disqualify it); the hidden downloads once for the comparison.
  Qwen3DFlashModel::DflashBlockDeviceOut dev_out;
  auto logits = Qwen3DFlashModel::ForwardBlockLogitsWithDeviceKV(
      {store}, ctx_cu, ids, bpos, cu, w, f.config, q, nullptr, nullptr, &dev_out);
  REQUIRE(logits.empty());
  REQUIRE(dev_out.hidden.data != nullptr);
  REQUIRE_EQ(dev_out.hidden.rank, 2);
  REQUIRE_EQ(dev_out.hidden.shape[0], kBlock);
  REQUIRE_EQ(dev_out.hidden.shape[1], kH);
  std::vector<float> final_out(static_cast<size_t>(kBlock * kH));
  if (q.device.type == vt::DeviceType::kTENSTORRENT) {
    // The result DBuf is device-authoritative with a TILE device shadow; a raw
    // byte Copy reads the padded layout. EnsureHost refreshes the host master
    // through the slot's own download path.
    vt::tenstorrent::EnsureHost(dev_out.hidden);
    const uint8_t* hb = static_cast<const uint8_t*>(dev_out.hidden.data);
    for (size_t i = 0; i < final_out.size(); ++i) {
      uint16_t b;
      std::memcpy(&b, hb + i * 2, 2);
      final_out[i] = Bf16BitsToFloat(b);
    }
  } else {
    vt::Backend& bb = vt::GetBackend(q.device.type);
    bb.Copy(q, final_out.data(), dev_out.hidden.data,
            static_cast<size_t>(kBlock * kH) * 2);
    bb.Synchronize(q);
  }
  return final_out;
}

}  // namespace

// The W2 component gate: OUR TT chain vs the bundle reference math, at the
// fixed per-iteration shape, on the REAL pinned drafter weights.
TEST_CASE("qwen38 dflash2 drafter TT: the 8-row block forward over tap context "
          "matches the bundle reference (PCC band)") {
  if (!HasTT()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  const std::string dir = WeightsDir();
  if (!fs::exists(dir + "/config.json")) {
    MESSAGE("SKIPPED: pinned drafter checkpoint not found at " << dir
            << " (set VT_QWEN38_DFLASH2_DRAFTER)");
    return;
  }
  Fixture f = MakeFixture();

  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  std::fprintf(stderr, "[GATE] ref begin\n");
  auto ref = RefForward(f.ref, FromF32(f.taps), f.block_rows);
  std::fprintf(stderr, "[GATE] ref end\n");

  // STAGE GATES, so a divergence names its stage instead of a final-hidden PCC.
  // Stage 1: the fc tap combine (device matmul) vs the reference hctx.
  auto fc_out = Qwen3DFlashModel::CombineAuxFeatures(f.taps, kCtx, f.weights, f.config, q);
  {
  // CombineAuxFeatures returns the PRE-norm fc output (the chain applies
  // hidden_norm later, qwen3_dflash.cpp:367) — compare against the RAW MulWT,
  // not the normed hctx (the pre-fix harness compared pre- vs post-norm and
  // reported a fictitious rel_rms 33 scale mismatch).
  const Mat hctx = MulWT(FromF32(f.taps), kCtx, kH * kTaps, f.ref.fc, kH);
  const PccStats st = Compare(fc_out, hctx, 0, static_cast<size_t>(kCtx * kH));
  // row-wise PCC, so a per-row divergence names its row
  for (int64_t r = 0; r < kCtx; ++r) {
    const PccStats sr = Compare(fc_out, hctx, static_cast<size_t>(r * kH), static_cast<size_t>(kH));
      if (sr.pcc < kPccFloor)
        MESSAGE("stage fc row " << r << ": PCC " << sr.pcc << ", rel_rms " << sr.rel_rms);
    }
    double maxa = 0, maxb = 0;
    size_t outliers = 0;
    size_t first_outlier = 0;
    for (size_t i = 0; i < fc_out.size(); ++i) {
      const double a = std::fabs(static_cast<double>(fc_out[i]));
      const double b = std::fabs(hctx[i]);
      maxa = std::max(maxa, a);
      maxb = std::max(maxb, b);
      if (a > 10.0 * (b + 1e-6)) {
        if (outliers == 0) first_outlier = i;
        ++outliers;
      }
    }
    MESSAGE("stage fc: PCC " << st.pcc << ", rel_rms " << st.rel_rms
            << ", max|ours| " << maxa << ", max|ref| " << maxb
            << ", outliers " << outliers << ", first at " << first_outlier
            << " (row " << first_outlier / kH << " col " << first_outlier % kH << ")"
            << " ours " << fc_out[first_outlier] << " ref " << hctx[first_outlier]);
    REQUIRE_MESSAGE(st.pcc >= kPccFloor, "stage fc diverged: PCC " << st.pcc);
  }
  // Stage 2: the context K/V projection (per layer 0) from the host store.
  {
    Qwen3DFlashModel::PrecomputedContextKV ckv;
    std::vector<int32_t> ctx_pos(static_cast<size_t>(kCtx));
    for (int64_t i = 0; i < kCtx; ++i)
      ctx_pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    Qwen3DFlashModel::AppendContextKVHost(ckv, fc_out, ctx_pos, f.weights, f.config, q);
    REQUIRE_EQ(ckv.num_ctx, kCtx);
    const int64_t kdim = kNKV * kHD;
    const RefLayer& lw0 = f.ref.layers[0];
    Mat hctx = MulWT(FromF32(f.taps), kCtx, kH * kTaps, f.ref.fc, kH);
    hctx = Rms(hctx, kCtx, kH, f.ref.hnorm);
    Mat kc = MulWT(hctx, kCtx, kH, lw0.k, kNKV * kHD);
    Mat vc = MulWT(hctx, kCtx, kH, lw0.v, kNKV * kHD);
    // k_norm + RoPE per head at absolute positions (the reference's own path)
    Mat cos, sin;
    RopeTables(kCtx + kBlock, &cos, &sin);
    Mat kcat(static_cast<size_t>(kCtx * kNKV * kHD));
    std::memcpy(kcat.data(), kc.data(), sizeof(double) * kcat.size());
    // reuse the reference's head rope helper through RefLayerForward's idiom
    Mat kh(static_cast<size_t>(kCtx * kNKV * kHD), 0.0);
    for (int64_t hh = 0; hh < kNKV; ++hh) {
      Mat slice(static_cast<size_t>(kCtx * kHD), 0.0);
      for (int64_t t = 0; t < kCtx; ++t)
        std::memcpy(&slice[static_cast<size_t>(t * kHD)],
                    &kcat[static_cast<size_t>((t * kNKV + hh) * kHD)],
                    sizeof(double) * static_cast<size_t>(kHD));
      Mat hr = HeadRms(slice, kCtx, 1, lw0.kn);
      Mat roped = RopeApply(hr, kCtx, kHD, cos, sin, 0);
      for (int64_t t = 0; t < kCtx; ++t)
        std::memcpy(&kh[static_cast<size_t>((t * kNKV + hh) * kHD)],
                    &roped[static_cast<size_t>(t * kHD)],
                    sizeof(double) * static_cast<size_t>(kHD));
    }
    // v (no norm, no rope) isolates the projection from the k-norm/rope side
    {
      std::vector<float> vhf(static_cast<size_t>(kCtx * kdim));
      for (size_t i = 0; i < vhf.size(); ++i)
        vhf[i] = Bf16BitsToFloat(ckv.v[0][i]);
      Mat vc2(static_cast<size_t>(kCtx * kdim));
      for (int64_t t = 0; t < kCtx; ++t)
        std::memcpy(&vc2[static_cast<size_t>(t * kdim)],
                    &vc[static_cast<size_t>(t * kdim)], sizeof(double) * static_cast<size_t>(kdim));
      const PccStats stv = Compare(vhf, vc2, 0, static_cast<size_t>(kCtx * kdim));
      // fusion-order hypothesis: device v against the reference RAW K
      Mat kr(static_cast<size_t>(kCtx * kdim));
      std::memcpy(kr.data(), kc.data(), sizeof(double) * kr.size());
      const PccStats stvk = Compare(vhf, kr, 0, static_cast<size_t>(kCtx * kdim));
      Mat qr(static_cast<size_t>(kCtx * (kNH * kHD)));
      {
        Mat qraw = MulWT(hctx, kCtx, kH, lw0.q, kNH * kHD);
        qr = qraw;
      }
      // device v vs reference q (first kCtx rows' heads squashed is nonsense; skip)
      MESSAGE("stage ctxV l0: PCC " << stv.pcc << ", rel_rms " << stv.rel_rms
              << " | deviceV vs refK-raw PCC " << stvk.pcc);
      MESSAGE("ctxV ours[0..2] " << vhf[0] << " " << vhf[1] << " " << vhf[2]
              << " | ref " << vc2[0] << " " << vc2[1] << " " << vc2[2]);
    }
    for (int l = 0; l < static_cast<int>(ckv.k.size()); ++l) {
      // compare the DEVICE store's projection only where the reference has the
      // layer's weights: layer 0 here, the rest only for finiteness
      std::vector<float> khf(static_cast<size_t>(kCtx * kdim));
      for (size_t i = 0; i < khf.size(); ++i)
        khf[i] = Bf16BitsToFloat(ckv.k[static_cast<size_t>(l)][i]);
      if (l == 0) {
        const PccStats st = Compare(khf, kh, 0, static_cast<size_t>(kCtx * kdim));
        const PccStats st_raw = Compare(khf, kc, 0, static_cast<size_t>(kCtx * kdim));
        Mat kn_(static_cast<size_t>(kCtx * kNKV * kHD), 0.0);
        for (int64_t hh = 0; hh < kNKV; ++hh) {
          Mat slice(static_cast<size_t>(kCtx * kHD), 0.0);
          for (int64_t t = 0; t < kCtx; ++t)
            std::memcpy(&slice[static_cast<size_t>(t * kHD)],
                        &kcat[static_cast<size_t>((t * kNKV + hh) * kHD)],
                        sizeof(double) * static_cast<size_t>(kHD));
          Mat hr = HeadRms(slice, kCtx, 1, lw0.kn);
          for (int64_t t = 0; t < kCtx; ++t)
            std::memcpy(&kn_[static_cast<size_t>((t * kNKV + hh) * kHD)],
                        &hr[static_cast<size_t>(t * kHD)],
                        sizeof(double) * static_cast<size_t>(kHD));
        }
        const PccStats st_norope = Compare(khf, kn_, 0, static_cast<size_t>(kCtx * kdim));
        // heads-outer layout hypothesis: [Hkv, num_ctx, Dh]
        Mat kho(static_cast<size_t>(kCtx * kNKV * kHD), 0.0);
        for (int64_t hh = 0; hh < kNKV; ++hh)
          for (int64_t t = 0; t < kCtx; ++t)
            std::memcpy(&kho[static_cast<size_t>(((hh * kCtx) + t) * kHD)],
                        &kh[static_cast<size_t>((t * kNKV + hh) * kHD)],
                        sizeof(double) * static_cast<size_t>(kHD));
        const PccStats st_ho = Compare(khf, kho, 0, static_cast<size_t>(kCtx * kdim));
        MESSAGE("stage ctxK l0 vs normed+rope: PCC " << st.pcc << ", rel_rms " << st.rel_rms
                << " | vs raw: PCC " << st_raw.pcc << " | vs normed-norope: PCC "
                << st_norope.pcc << " | vs heads-outer: PCC " << st_ho.pcc);
        const bool ok_ctxk = (st.pcc >= kPccFloor) || (st_ho.pcc >= kPccFloor);
        REQUIRE_MESSAGE(ok_ctxk, "stage ctxK diverged: PCC " << st.pcc
                        << " / heads-outer " << st_ho.pcc);
      }
    }
  }

  auto store = Qwen3DFlashModel::MakeDeviceKVStore(f.config, q, 256);
  auto ours = RunOurForward(f, q, f.weights, store.get());

  double worst_pcc = 1.0, worst_rel = 0.0;
  for (int64_t b = 0; b < kBlock; ++b) {
    const PccStats s = Compare(ours, ref, static_cast<size_t>(b * kH), static_cast<size_t>(kH));
    MESSAGE("row " << b << " (" << (b == 0 ? "anchor" : "mask") << "): PCC "
                   << s.pcc << ", rel_rms " << s.rel_rms);
    worst_pcc = std::min(worst_pcc, s.pcc);
    worst_rel = std::max(worst_rel, s.rel_rms);
  }
  const PccStats all = Compare(ours, ref, 0, static_cast<size_t>(kBlock * kH));
  MESSAGE("ALL rows: PCC " << all.pcc << ", rel_rms " << all.rel_rms
                           << " (floor " << kPccFloor << ", the bundle's band, model.py:1132)");
  CHECK_MESSAGE(worst_pcc >= kPccFloor,
                "the drafter forward diverged from the bundle reference: worst-row PCC "
                << worst_pcc << " < " << kPccFloor);
  CHECK_MESSAGE(all.rel_rms <= 0.10,
                "the drafter forward's rel_rms envelope blew past 0.10: " << all.rel_rms);
}

// The TT conv kernel vs the CPU reference. The CPU op's contract is bit-exact
// scalar math (cpu_ops.cpp rounds every step), and the TT composition applies
// the SAME one-round-per-step structure — but the pinned substrate's f32->bf16
// conversion does not round ties to even (measured: the exact tie 170.5 ulp
// rounds UP at i=8, the exact tie 207.5 ulp rounds DOWN at the worst case), so
// k = bf16(b + d) can land one k-ulp away wherever b + d is a tie, and a k≈0
// cancellation can flip a whole tap term (cpu 0 vs tt 0.03125). The gate is
// therefore the op-magnitude envelope below, not equality. Odd T exercises the
// pos masking and the power-of-two `&` arm together.
TEST_CASE("qwen38 dflash2 drafter TT: DFlashGroupedConv matches CPU within the "
          "tie-rounding envelope") {
  if (!HasTT()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  constexpr int64_t T = 17, H = 5120, GROUPS = 320, GSIZE = 16, TAPS = 2, SIDES = 2;
  std::mt19937 rng(7);
  std::uniform_int_distribution<int> ud(-200, 200);
  std::vector<uint16_t> xb(static_cast<size_t>(T * H)), cb(static_cast<size_t>(T * SIDES * TAPS * GROUPS)),
      bb_(static_cast<size_t>(SIDES * TAPS * H));
  auto fill = [&](std::vector<uint16_t>& v) {
    for (auto& e : v) e = FloatToBf16(static_cast<float>(ud(rng)) / 100.0f);
  };
  fill(xb); fill(cb); fill(bb_);
  vt::DFlashGroupedConvArgs a;
  a.block_size = 8;
  a.taps = TAPS;
  a.num_groups = GROUPS;
  a.group_size = GSIZE;

  auto run = [&](vt::DeviceType dt, std::vector<uint16_t>* out) {
    vt::Backend& be = vt::GetBackend(dt);
    vt::Queue q = be.CreateQueue();
    auto mk = [&](std::vector<uint16_t>& v, std::initializer_list<int64_t> shape) {
      return vt::Tensor::Contiguous(v.data(), vt::DType::kBF16,
                                    vt::Device{dt, 0}, shape);
    };
    vt::Tensor tx = mk(xb, {T, H});
    vt::Tensor tc = mk(cb, {T, SIDES, TAPS, GROUPS});
    vt::Tensor tb = mk(bb_, {SIDES, TAPS, H});
    out->assign(static_cast<size_t>(T * H), 0);
    vt::Tensor to = mk(*out, {T, H});
    vt::DFlashGroupedConv(q, to, tx, tc, tb, a);
    be.Synchronize(q);
    be.DestroyQueue(q);
  };
  std::vector<uint16_t> cpu_out, tt_out;
  run(vt::DeviceType::kCPU, &cpu_out);
  run(vt::DeviceType::kTENSTORRENT, &tt_out);
  REQUIRE_EQ(cpu_out.size(), tt_out.size());
  size_t diff = 0, first = 0;
  for (size_t i = 0; i < cpu_out.size(); ++i)
    if (cpu_out[i] != tt_out[i]) {
      if (diff == 0) first = i;
      ++diff;
    }
  size_t big = 0;
  double maxreldiff = 0.0;
  std::vector<size_t> per_row(static_cast<size_t>(T), 0);
  for (size_t i = 0; i < cpu_out.size(); ++i) {
    if (cpu_out[i] != tt_out[i]) {
      ++per_row[i / H];
      const float cf = Bf16BitsToFloat(cpu_out[i]), tf = Bf16BitsToFloat(tt_out[i]);
      const double rel = std::fabs(static_cast<double>(cf) - static_cast<double>(tf)) /
                         std::max(std::fabs(static_cast<double>(cf)), 1e-6);
      if (rel > 0.05) ++big;
      maxreldiff = std::max(maxreldiff, rel);
    }
  }
  std::string rows;
  for (size_t r = 0; r < per_row.size(); ++r)
    rows += std::to_string(r) + ":" + std::to_string(per_row[r]) + " ";
  MESSAGE("conv parity: " << diff << " bit-diffs of " << cpu_out.size()
          << ", big(>5%) " << big << ", max rel diff " << maxreldiff
          << " | per-row " << rows);
  {
    size_t worst = 0;
    double worstr = 0.0;
    for (size_t i = 0; i < cpu_out.size(); ++i) {
      const float cf = Bf16BitsToFloat(cpu_out[i]), tf = Bf16BitsToFloat(tt_out[i]);
      const double rel = std::fabs(static_cast<double>(cf) - static_cast<double>(tf)) /
                         std::max(std::fabs(static_cast<double>(cf)), 1e-6);
      if (rel > worstr) {
        worstr = rel;
        worst = i;
      }
    }
    const size_t r = worst / H, c = worst % H;
    const int64_t g = c / GSIZE;
    const float cv = Bf16BitsToFloat(cpu_out[worst]), tv = Bf16BitsToFloat(tt_out[worst]);
    const float xv = Bf16BitsToFloat(xb[r * H + c]);
    const float xpv = r ? Bf16BitsToFloat(xb[(r - 1) * H + c]) : 0.0f;
    const float d0v = Bf16BitsToFloat(cb[(((r * SIDES + 0) * TAPS + 0) * GROUPS + g)]);
    const float d1v = Bf16BitsToFloat(cb[(((r * SIDES + 0) * TAPS + 1) * GROUPS + g)]);
    const float b0v = Bf16BitsToFloat(bb_[0 * H + c]);
    const float b1v = Bf16BitsToFloat(bb_[1 * H + c]);
    const int64_t pos = r & 7;
    MESSAGE("worst i=" << worst << " row " << r << " col " << c << " pos " << pos
            << " cpu " << cv << " tt " << tv << " | x " << xv << " xPrev " << xpv
            << " d0 " << d0v << " d1 " << d1v << " b0 " << b0v << " b1 " << b1v
            << " | k0-away " << Bf16AddAway(b0v, d0v) << " cpu-k0 " << Bf16AddEven(b0v, d0v));
  }
  for (size_t i = first; i < cpu_out.size() && i < first + 8; ++i) {
    const size_t r = i / H, c = i % H;
    float cv = Bf16BitsToFloat(cpu_out[i]), tv = Bf16BitsToFloat(tt_out[i]);
    float xv = Bf16BitsToFloat(xb[static_cast<size_t>(r) * H + c]);
    float dv = Bf16BitsToFloat(cb[static_cast<size_t>(((r * SIDES + 0) * TAPS + 0) * GROUPS + c / GSIZE)]);
    float bv = Bf16BitsToFloat(bb_[0 * H + c]);
    MESSAGE("i=" << i << " row " << r << " col " << c << " cpu " << cv << " tt " << tv
            << " x " << xv << " d " << dv << " b " << bv);
  }
  // The envelope: one k-ulp (bf16 ties of b + d land one k-ulp away on the
  // pinned substrate, see the case comment) times the operand scale, plus the
  // rel_rms floor. maxabs and rel_rms are PRINTED for the record.
  double num = 0.0, den = 0.0;
  float maxabs = 0.0f, xmax = 0.0f;
  for (size_t i = 0; i < cpu_out.size(); ++i) {
    const float cf = Bf16BitsToFloat(cpu_out[i]), tf = Bf16BitsToFloat(tt_out[i]);
    const double d = static_cast<double>(tf) - static_cast<double>(cf);
    maxabs = std::max(maxabs, static_cast<float>(std::fabs(d)));
    num += d * d;
    den += static_cast<double>(cf) * static_cast<double>(cf);
    xmax = std::max(xmax, std::fabs(Bf16BitsToFloat(xb[i])));
  }
  const double rel_rms = std::sqrt(num) / std::sqrt(std::max(den, 1e-12));
  MESSAGE("conv parity envelope: rel_rms " << rel_rms << ", maxabs " << maxabs
          << ", xmax " << xmax << ", bound " << 0.05f * xmax);
  CHECK_MESSAGE(maxabs <= 0.05f * xmax + 1e-6f,
                "the TT conv exceeded the tie-rounding envelope: maxabs " << maxabs
                << " > " << 0.05f * xmax);
  CHECK_MESSAGE(rel_rms <= 0.01,
                "the TT conv rel_rms blew past 0.01: " << rel_rms);
}

// TT-vs-CPU parity for vt::PagedAttention at THE DRAFT BLOCK'S SHAPE: one
// request, an 8-row spec query block over a 16-row context, GQA 32q/8kv,
// head_dim 128, bidirectional (the artifact's all-bidirectional sliding layers,
// window not binding), bf16 pool — the exact shape the paged-seam route hands
// the TT kernel and the shape no op gate has ever run on Tenstorrent (the
// existing test_ops_paged_attn.cpp lanes are CPU/CUDA only).
TEST_CASE("qwen38 dflash2 drafter TT: paged attention at the draft block shape "
          "matches the CPU kernel") {
  if (!HasTT()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  constexpr int64_t Hq = 32, Hkv = 8, D = 128, BLOCK = 16;
  constexpr int64_t CTX = 16, QLEN = 8, SEQ = CTX + QLEN;
  constexpr int64_t PAGES = 4;
  const float scale = 1.0f / std::sqrt(static_cast<float>(D));
  std::mt19937 rng(11);
  std::normal_distribution<float> nd(0.0f, 1.0f);
  auto bf = [&](std::vector<float>& v) {
    for (float& e : v) e = Bf16BitsToFloat(FloatToBf16(e = nd(rng)));
  };
  std::vector<uint16_t> qb(static_cast<size_t>(QLEN * Hq * D)),
      pool_kb(static_cast<size_t>(PAGES * BLOCK * Hkv * D)),
      pool_vb(static_cast<size_t>(PAGES * BLOCK * Hkv * D));
  {
    std::vector<float> tmp(qb.size());
    bf(tmp);
    for (size_t i = 0; i < tmp.size(); ++i) qb[i] = FloatToBf16(tmp[i]);
    tmp.resize(pool_kb.size()); bf(tmp);
    for (size_t i = 0; i < tmp.size(); ++i) pool_kb[i] = FloatToBf16(tmp[i]);
    tmp.resize(pool_vb.size()); bf(tmp);
    for (size_t i = 0; i < tmp.size(); ++i) pool_vb[i] = FloatToBf16(tmp[i]);
  }
  std::vector<int32_t> bt = {0, 1, 2, 3};  // one request, pages 0..3
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(SEQ)};
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(QLEN)};

  auto run = [&](vt::DeviceType dt, std::vector<float>* out) {
    vt::Backend& be = vt::GetBackend(dt);
    vt::Queue q = be.CreateQueue();
    auto mk = [&](void* p, std::initializer_list<int64_t> shape, vt::DType dtype) {
      return vt::Tensor::Contiguous(p, dtype, vt::Device{dt, 0}, shape);
    };
    vt::Tensor tq = mk(qb.data(), {QLEN, Hq, D}, vt::DType::kBF16);
    vt::Tensor tk = mk(pool_kb.data(), {PAGES, BLOCK, Hkv, D}, vt::DType::kBF16);
    vt::Tensor tv = mk(pool_vb.data(), {PAGES, BLOCK, Hkv, D}, vt::DType::kBF16);
    vt::Tensor tbt = mk(bt.data(), {1, PAGES}, vt::DType::kI32);
    vt::Tensor tsl = mk(seq_lens.data(), {1}, vt::DType::kI32);
    vt::Tensor tqsl = mk(qsl.data(), {2}, vt::DType::kI32);
    out->assign(static_cast<size_t>(QLEN * Hq * D), 0.0f);
    vt::Tensor to = mk(out->data(), {QLEN, Hq, D}, vt::DType::kF32);
    vt::PagedAttentionArgs pa;
    pa.scale = scale;
    pa.causal = false;  // the artifact's bidirectional sliding layers
    pa.uniform_spec_query_len = static_cast<int32_t>(QLEN);
    pa.query_start_loc_host = qsl.data();
    pa.max_seq_len = static_cast<int32_t>(SEQ);
    vt::PagedAttention(q, to, tq, tk, tv, tbt, tsl, tqsl, pa);
    be.Synchronize(q);
    be.DestroyQueue(q);
  };
  std::vector<float> cpu_out, tt_out;
  run(vt::DeviceType::kCPU, &cpu_out);
  run(vt::DeviceType::kTENSTORRENT, &tt_out);
  double num = 0.0, den = 0.0, maxd = 0.0;
  size_t first = 0;
  for (size_t i = 0; i < cpu_out.size(); ++i) {
    const double d = static_cast<double>(tt_out[i]) - static_cast<double>(cpu_out[i]);
    if (maxd == 0.0 && d != 0.0) first = i;
    maxd = std::max(maxd, std::fabs(d));
    num += d * d;
    den += static_cast<double>(cpu_out[i]) * static_cast<double>(cpu_out[i]);
  }
  const double rel = std::sqrt(num) / std::sqrt(std::max(den, 1e-12));
  MESSAGE("paged-attn parity: rel_rms " << rel << ", max|d| " << maxd << ", first "
          << first << " (cpu " << cpu_out[first] << " tt " << tt_out[first] << ")");
  CHECK_MESSAGE(rel < 0.02,
                "the TT paged-attention kernel diverges from the CPU kernel at "
                "the draft block shape: rel_rms " << rel);
}

// envelope helpers for the conv parity gate (see the case comment)
namespace {
struct ConvParity {
  double rel_rms = 0.0;
  float maxabs = 0.0f;
  size_t diff_count = 0;
};
}  // namespace

// The RED proof: the gate above must DETECT a wrong drafter. One fc output
// row's sign is flipped on our side only (the reference keeps the artifact's
// weights), so the whole context projection rotates; the PCC gate must collapse
// well below the band. A gate that stays green under this mutation measures
// nothing.
// The CPU-backend arm of the same chain: isolates WHICH side of the PCC gate
// moves. If the CPU chain matches the reference, the bundle math and our chain
// agree and the delta is TT-specific; if not, the reference's reading of the
// documented math is what differs.
TEST_CASE("qwen38 dflash2 drafter: the CPU arm of the chain matches the bundle "
          "reference") {
  // NO device required: the fc combine runs on the HOST (bf16-rounded fp64
  // matmul — CombineAuxFeatures' own contract is that bf16 cast), so this case
  // runs CPU-only and isolates the CHAIN SEMANTICS from every device question.
  const std::string wdir0 = WeightsDir();
  if (!fs::exists(wdir0 + "/config.json")) {
    MESSAGE("SKIPPED: pinned drafter checkpoint not found at " << wdir0);
    return;
  }
  Fixture f = MakeFixtureCpuOnly();
  {
    // MakeFixtureCpuOnly leaves cpu_queue null — make the CPU queue here. The
    // earlier "CPU MatmulBT raw-NK path segfaults on this fixture" finding was
    // THIS nullptr, misattributed: a null Queue dereferenced at AppendContextKVHost
    // entry, before any op ran.
    vt::Backend& cpu_be = vt::GetBackend(vt::DeviceType::kCPU);
    auto* cpu_q = new vt::Queue(cpu_be.CreateQueue());  // leaked: must outlive the
    // case AND the static doctest teardown that SIGABRTs on queue destruction
    f.cpu_queue = cpu_q;
  }
  // The CPU arm reuses the D9 host path (PrecomputeContextKV +
  // ForwardBlockLogitsWithPrecomputedKV): the same chain semantics without the
  // D11 device store. The fc combine is the host bf16-rounded matmul — the
  // values CombineAuxFeatures produces on ANY backend (its output is bf16).
  {
    const Mat fc64 = MulWT(FromF32(f.taps), kCtx, kH * kTaps, f.ref.fc, kH);
    fc_out_storage.resize(static_cast<size_t>(kCtx * kH));
    for (size_t i = 0; i < fc_out_storage.size(); ++i)
      fc_out_storage[i] = Bf16BitsToFloat(FloatToBf16(static_cast<float>(fc64[i])));
  }
  const std::vector<float>& fc_out = fc_out_storage;
  Qwen3DFlashModel::PrecomputedContextKV ckv;
  std::vector<int32_t> ctx_pos(static_cast<size_t>(kCtx));
  for (int64_t i = 0; i < kCtx; ++i) ctx_pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  Qwen3DFlashModel::AppendContextKVHost(ckv, fc_out, ctx_pos, f.weights, f.config,
                                        *f.cpu_queue);
  std::vector<int32_t> ids(static_cast<size_t>(kBlock), kMaskId);
  ids[0] = kAnchorId;
  std::vector<int32_t> bpos(static_cast<size_t>(kBlock));
  for (int64_t i = 0; i < kBlock; ++i)
    bpos[static_cast<size_t>(i)] = static_cast<int32_t>(kCtx + i);
  // CONTEXT-FREE probe (W2_REF_NO_CTX=1): the same block with NO context, so
  // the block-internal math (conv/attention/mlp) is gated in isolation from
  // the context interaction.
  const bool no_ctx = std::getenv("W2_REF_NO_CTX") != nullptr;
  std::vector<float> final_out;
  std::vector<std::vector<float>> per_layer;
  std::vector<int32_t> pos_cf(static_cast<size_t>(kBlock));
  for (int64_t i = 0; i < kBlock; ++i) pos_cf[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  auto logits = no_ctx
      ? Qwen3DFlashModel::ForwardBlockLogits(
            ids, pos_cf, {0, static_cast<int32_t>(kBlock)},
            f.weights, f.config, *f.cpu_queue, &per_layer, &final_out)
      : Qwen3DFlashModel::ForwardBlockLogitsWithPrecomputedKV(
      ckv, {0, static_cast<int32_t>(kCtx)}, ids, bpos, {0, static_cast<int32_t>(kBlock)},
      f.weights, f.config, *f.cpu_queue, &per_layer, &final_out);
  REQUIRE(static_cast<int64_t>(final_out.size()) == kBlock * kH);
  auto ours = final_out;
  auto ref = RefForward(f.ref, FromF32(f.taps), f.block_rows);
  // PER-LAYER: the reference's layer-l output vs the chain's, so a divergence
  // names its layer.
  // CONV op probe: my fp64 Gdc vs the C++ CPU kernel on the REAL layer-0
  // weights and the real normed block rows — names the conv as guilty or clean.
  if (std::getenv("W2_REF_CONV_PROBE") != nullptr) {
    const RefLayer& lw0 = f.ref.layers[0];
    Mat hn = Rms(f.block_rows, kBlock, kH, lw0.in_ln);
    Mat kp = MulWT(hn, kBlock, kH, lw0.ac_kp, 1280);
    // the op consumes bf16 operands — round the reference's inputs the same
    // way so this probe checks the fp64 Gdc's EXACT agreement with the CPU op
    hn = RoundBAll(hn);
    kp = RoundBAll(kp);
    Mat refv = Gdc(hn, kBlock, kp, lw0.ac_b00, lw0.ac_b01, 0);
    // bf16 operands through the real op, CPU backend
    std::vector<uint16_t> xb(kBlock * kH), cb(kBlock * 1280), bbv(4 * kH);
    for (int64_t i = 0; i < kBlock * kH; ++i) xb[i] = FloatToBf16(static_cast<float>(hn[i]));
    for (int64_t i = 0; i < kBlock * 1280; ++i) {
      // kernel_projection fp64 [B,1280] IS the op's coefficient tensor values
      cb[i] = FloatToBf16(static_cast<float>(kp[i]));
    }
    for (int64_t i = 0; i < 4 * kH; ++i) bbv[i] = FloatToBf16(static_cast<float>(
        i < 2 * kH ? (i < kH ? lw0.ac_b00[i] : lw0.ac_b01[i - kH])
                   : (i < 3 * kH ? lw0.ac_b10[i - 2 * kH] : lw0.ac_b11[i - 3 * kH])));
    vt::Backend& cbe = vt::GetBackend(vt::DeviceType::kCPU);
    vt::Queue cq = cbe.CreateQueue();
    vt::Tensor tx = vt::Tensor::Contiguous(xb.data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
    vt::Tensor tc = vt::Tensor::Contiguous(cb.data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0},
                                           {kBlock, 2, 2, 320});
    vt::Tensor tb = vt::Tensor::Contiguous(bbv.data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {2, 2, kH});
    std::vector<uint16_t> ob(kBlock * kH, 0);
    vt::Tensor to = vt::Tensor::Contiguous(ob.data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
    vt::DFlashGroupedConvArgs ca;
    ca.block_size = kBlock; ca.taps = 2; ca.num_groups = 320; ca.group_size = 16; ca.side = 0;
    vt::DFlashGroupedConv(cq, to, tx, tc, tb, ca);
    cbe.Synchronize(cq);
    cbe.DestroyQueue(cq);
    double num = 0, den = 0;
    for (int64_t i = 0; i < kBlock * kH; ++i) {
      const double g = Bf16BitsToFloat(ob[static_cast<size_t>(i)]);
      const double r = refv[static_cast<size_t>(i)];
      num += (g - r) * (g - r);
      den += r * r;
    }
    MESSAGE("conv probe: rel_rms(op vs fp64 Gdc) "
            << std::sqrt(num) / std::sqrt(std::max(den, 1e-12)));
    // scalar decomposition of the first big-diff element (the exact bf16
    // arithmetic the CPU kernel is supposed to perform)
    {
      const int64_t rr = 1, cc = 2, gg = cc / 16;
      const float x1 = Bf16BitsToFloat(xb[static_cast<size_t>(rr * kH + cc)]);
      const float x0 = Bf16BitsToFloat(xb[static_cast<size_t>(cc)]);
      const float d0 = Bf16BitsToFloat(cb[static_cast<size_t>(rr * 1280 + gg)]);
      const float d1 = Bf16BitsToFloat(cb[static_cast<size_t>(rr * 1280 + 320 + gg)]);
      const float b0 = Bf16BitsToFloat(bbv[static_cast<size_t>(cc)]);
      const float b1 = Bf16BitsToFloat(bbv[static_cast<size_t>(kH + cc)]);
      const float k0 = Bf16AddEven(b0, d0), k1 = Bf16AddEven(b1, d1);
      const float t0 = Bf16AddEven(k0 * x1, 0.0f);
      const float t1 = Bf16AddEven(k1 * x0, 0.0f);
      const float acc = Bf16AddEven(t0, t1);
      MESSAGE("scalar decomposition row1 col2: k0 " << k0 << " k1 " << k1
              << " t0 " << t0 << " t1 " << t1 << " acc " << acc
              << " | op " << Bf16BitsToFloat(ob[static_cast<size_t>(rr * kH + cc)])
              << " | k0xfp64 " << k0 * x1 << " k1x0fp64 " << k1 * x0
              << " | cand cb[320] " << Bf16BitsToFloat(cb[320])
              << " cb[640] " << Bf16BitsToFloat(cb[640])
              << " cb[960] " << Bf16BitsToFloat(cb[960])
              << " cb[1600] " << Bf16BitsToFloat(cb[1600])
              << " cb[1600-320] " << Bf16BitsToFloat(cb[1280])
              << " cb[2] " << Bf16BitsToFloat(cb[2])
              << " | b00[2] " << Bf16BitsToFloat(bbv[2])
              << " b01[2] " << Bf16BitsToFloat(bbv[kH + 2])
              << " b10[2] " << Bf16BitsToFloat(bbv[2 * kH + 2])
              << " b11[2] " << Bf16BitsToFloat(bbv[3 * kH + 2]));
    }
    for (int64_t i = 0; i < kBlock * kH; ++i) {
      const double g = Bf16BitsToFloat(ob[static_cast<size_t>(i)]);
      const double r = refv[static_cast<size_t>(i)];
      if (std::fabs(g - r) > 0.3 * std::fabs(r) + 1e-3) {
        const int64_t row = i / kH, c = i % kH;
        const int64_t grp = c / 16;
        MESSAGE("conv probe first big diff i=" << i << " row " << row << " col " << c
                << " op " << g << " fp64 " << r << " v " << hn[static_cast<size_t>(row * kH + c)]
                << " vPrev " << (row ? hn[static_cast<size_t>((row - 1) * kH + c)] : 0.0)
                << " d0 " << kp[static_cast<size_t>(row * 1280 + grp)]
                << " d1 " << kp[static_cast<size_t>(row * 1280 + 320 + grp)]
                << " b0 " << lw0.ac_b00[static_cast<size_t>(c)]
                << " b1 " << lw0.ac_b01[static_cast<size_t>(c)]);
        for (int64_t cc = 400; cc < 412; ++cc)
          MESSAGE("  r0 c" << cc << " op "
                  << Bf16BitsToFloat(ob[static_cast<size_t>(cc)]) << " fp64 "
                  << refv[static_cast<size_t>(cc)] << " b0 "
                  << lw0.ac_b00[static_cast<size_t>(cc)] << " d0 "
                  << kp[static_cast<size_t>(cc / 16)]);
        break;
      }
    }
  }

  // ATTENTION op probe: my fp64 attention vs vt::DFlashBlockAttention on CPU
  // at the block shape (bidirectional, one request) with real-shaped randoms.
  if (std::getenv("W2_REF_ATTN_PROBE") != nullptr) {
    constexpr int64_t HQ = kNH, HKV = kNKV;
    std::mt19937 pr(21);
    std::normal_distribution<float> pnd(0.0f, 1.0f);
    std::vector<uint16_t> q3b(kBlock * HQ * kHD), k3b(kBlock * HKV * kHD), v3b(kBlock * HKV * kHD);
    Mat qh_ref(kBlock * HQ * kHD), kh_ref(kBlock * HKV * kHD), vh_ref(kBlock * HKV * kHD);
    for (int64_t i = 0; i < kBlock * HQ * kHD; ++i)
      qh_ref[i] = Bf16BitsToFloat(q3b[i] = FloatToBf16(pnd(pr)));
    for (int64_t i = 0; i < kBlock * HKV * kHD; ++i) {
      kh_ref[i] = Bf16BitsToFloat(k3b[i] = FloatToBf16(pnd(pr)));
      vh_ref[i] = Bf16BitsToFloat(v3b[i] = FloatToBf16(pnd(pr)));
    }
    // fp64 reference: bidirectional full attention, GQA, softmax over all 8 keys
    Mat o_ref(static_cast<size_t>(kBlock * HQ * kHD), 0.0);
    for (int64_t t = 0; t < kBlock; ++t)
      for (int64_t h = 0; h < HQ; ++h) {
        const int64_t kvh = h / (HQ / HKV);
        std::vector<double> sc(kBlock);
        double mx = -1e300;
        for (int64_t s = 0; s < kBlock; ++s) {
          double acc = 0.0;
          for (int64_t d = 0; d < kHD; ++d)
            acc += qh_ref[static_cast<size_t>((t * HQ + h) * kHD + d)] *
                   kh_ref[static_cast<size_t>((s * HKV + kvh) * kHD + d)];
          sc[static_cast<size_t>(s)] = acc * (1.0 / std::sqrt(static_cast<double>(kHD)));
          mx = std::max(mx, sc[static_cast<size_t>(s)]);
        }
        double den = 0.0;
        for (int64_t s = 0; s < kBlock; ++s) {
          sc[static_cast<size_t>(s)] = std::exp(sc[static_cast<size_t>(s)] - mx);
          den += sc[static_cast<size_t>(s)];
        }
        for (int64_t d = 0; d < kHD; ++d) {
          double acc = 0.0;
          for (int64_t s = 0; s < kBlock; ++s)
            acc += sc[static_cast<size_t>(s)] *
                   vh_ref[static_cast<size_t>((s * HKV + kvh) * kHD + d)];
          o_ref[static_cast<size_t>((t * HQ + h) * kHD + d)] = acc / den;
        }
      }
    vt::Backend& cbe = vt::GetBackend(vt::DeviceType::kCPU);
    vt::Queue cq = cbe.CreateQueue();
    auto mk = [&](void* p, std::initializer_list<int64_t> shp) {
      return vt::Tensor::Contiguous(p, vt::DType::kBF16,
                                    vt::Device{vt::DeviceType::kCPU, 0}, shp);
    };
    vt::Tensor tq3 = mk(q3b.data(), {kBlock, HQ, kHD});
    vt::Tensor tk3 = mk(k3b.data(), {kBlock, HKV, kHD});
    vt::Tensor tv3 = mk(v3b.data(), {kBlock, HKV, kHD});
    std::vector<uint16_t> ob(kBlock * HQ * kHD, 0);
    vt::Tensor to3 = mk(ob.data(), {kBlock, HQ, kHD});
    std::vector<int32_t> cuv = {0, static_cast<int32_t>(kBlock)};
    vt::DFlashBlockAttentionArgs pa;
    pa.scale = 1.0f / std::sqrt(static_cast<float>(kHD));
    pa.causal = false;
    pa.sliding_window = 2048;
    pa.cu_seqlens = cuv.data();
    pa.num_reqs = 1;
    vt::DFlashBlockAttention(cq, to3, tq3, tk3, tv3, pa);
    cbe.Synchronize(cq);
    cbe.DestroyQueue(cq);
    double num = 0, den = 0;
    for (size_t i = 0; i < o_ref.size(); ++i) {
      const double g = Bf16BitsToFloat(ob[i]);
      num += (g - o_ref[i]) * (g - o_ref[i]);
      den += o_ref[i] * o_ref[i];
    }
    MESSAGE("attn probe: rel_rms(op vs fp64) " << std::sqrt(num) / std::sqrt(std::max(den, 1e-12)));
  }

  // OP-REPLAY (W2_REF_OP_REPLAY=1): the chain's LAYER-0 op sequence, no ctx,
  // run op by op on the CPU backend with the reference's bf16-rounded weights,
  // each intermediate PCC'd against the fp64 reference. This is the isolation
  // instrument: the stage whose PCC collapses names the op or the wiring.
  if (std::getenv("W2_REF_OP_REPLAY") != nullptr) {
    vt::Backend& cbe = vt::GetBackend(vt::DeviceType::kCPU);
    vt::Queue rcq = cbe.CreateQueue();
    const RefLayer& lw0 = f.ref.layers[0];
    std::vector<std::vector<uint16_t>> arena;  // Contiguous aliases: keep alive
    auto mkbf = [&](const Mat& m, std::initializer_list<int64_t> shape) {
      int64_t n = 1;
      for (int64_t d : shape) n *= d;
      arena.emplace_back(static_cast<size_t>(n));
      std::vector<uint16_t>& b = arena.back();
      for (size_t i = 0; i < b.size(); ++i) b[i] = FloatToBf16(static_cast<float>(m[i]));
      return vt::Tensor::Contiguous(b.data(), vt::DType::kBF16,
                                    vt::Device{vt::DeviceType::kCPU, 0}, shape);
    };
    auto mkbf1 = [&](const Mat& m, int64_t n) {
      arena.emplace_back(static_cast<size_t>(n));
      std::vector<uint16_t>& b = arena.back();
      for (int64_t i = 0; i < n; ++i) b[static_cast<size_t>(i)] = FloatToBf16(static_cast<float>(m[static_cast<size_t>(i)]));
      return vt::Tensor::Contiguous(b.data(), vt::DType::kBF16,
                                    vt::Device{vt::DeviceType::kCPU, 0}, {n});
    };
    auto downf = [&](const vt::Tensor& t, size_t n) {
      std::vector<float> v(n);
      cbe.Copy(rcq, v.data(), t.data, n * 2);
      cbe.Synchronize(rcq);
      std::vector<float> out(n);
      for (size_t i = 0; i < n; ++i) out[i] = Bf16BitsToFloat(
          static_cast<uint16_t>(reinterpret_cast<const uint16_t*>(v.data())[i]));
      return out;
    };
    auto pcc = [&](const std::vector<float>& got, const Mat& refv, const char* name) {
      std::vector<uint16_t> gb(got.size());
      for (size_t i = 0; i < got.size(); ++i) gb[i] = FloatToBf16(got[i]);
      std::vector<float> gf(got.size());
      for (size_t i = 0; i < gf.size(); ++i) gf[i] = Bf16BitsToFloat(gb[i]);
      const PccStats st = Compare(gf, refv, 0, refv.size());
      MESSAGE("replay", std::string(name), ": PCC", st.pcc, ", rel_rms", st.rel_rms);
    };
    // helper: per-head fp64 rope (the reference's head_rope idiom)
    auto head_rope64 = [&](const Mat& x, int64_t heads, const Mat& cos0, const Mat& sin0) {
      Mat y(static_cast<size_t>(kBlock * heads * kHD), 0.0);
      for (int64_t hh = 0; hh < heads; ++hh) {
        Mat slice(static_cast<size_t>(kBlock * kHD), 0.0);
        for (int64_t t = 0; t < kBlock; ++t)
          std::memcpy(&slice[static_cast<size_t>(t * kHD)],
                      &x[static_cast<size_t>((t * heads + hh) * kHD)],
                      sizeof(double) * static_cast<size_t>(kHD));
        Mat roped = RopeApply(slice, kBlock, kHD, cos0, sin0, 0);
        for (int64_t t = 0; t < kBlock; ++t)
          std::memcpy(&y[static_cast<size_t>((t * heads + hh) * kHD)],
                      &roped[static_cast<size_t>(t * kHD)],
                      sizeof(double) * static_cast<size_t>(kHD));
      }
      return y;
    };
    // host fp64 steps for the same stages
    Mat hn64 = Rms(f.block_rows, kBlock, kH, lw0.in_ln);
    Mat hn64r = RoundBAll(hn64);
    Mat kp64 = MulWT(hn64r, kBlock, kH, lw0.ac_kp, 1280);
    kp64 = RoundBAll(kp64);
    Mat attn_in64 = Gdc(hn64r, kBlock, kp64, lw0.ac_b00, lw0.ac_b01, 0);

    vt::Tensor th = mkbf(f.block_rows, {kBlock, kH});
    vt::Tensor w_in = mkbf1(lw0.in_ln, kH);
    arena.emplace_back(static_cast<size_t>(kBlock * kH));
      vt::Tensor dhn = vt::Tensor::Contiguous(arena.back().data(),
                                 vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                 {kBlock, kH});
    vt::RmsNorm(rcq, dhn, th, w_in, vt::RmsNormArgs{1e-6f, false});
    cbe.Synchronize(rcq);
    pcc(downf(dhn, static_cast<size_t>(kBlock * kH)), hn64, "after in_ln");

    vt::Tensor ac_w = mkbf(lw0.ac_kp, {1280, kH});
    arena.emplace_back(static_cast<size_t>(kBlock * 1280));
      vt::Tensor coef = vt::Tensor::Contiguous(arena.back().data(),
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock, 2, 2, 320});
    // MatmulBT needs rank-2 and DFlashGroupedConv needs rank-4: both views ride
    // the SAME arena buffer, so the matmul result IS the conv's coefficients.
    vt::Tensor coef2 = vt::Tensor::Contiguous(coef.data,
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock, 1280});
    vt::MatmulBT(rcq, coef2, dhn, ac_w);
    vt::Tensor ac_base = mkbf(
        [&] { Mat b(4 * kH);
              for (int64_t i = 0; i < kH; ++i) {
                b[i] = lw0.ac_b00[i]; b[kH + i] = lw0.ac_b01[i];
                b[2 * kH + i] = lw0.ac_b10[i]; b[3 * kH + i] = lw0.ac_b11[i];
              }
              return b; }(),
        {2, 2, kH});
    arena.emplace_back(static_cast<size_t>(kBlock * kH));
      vt::Tensor prep = vt::Tensor::Contiguous(arena.back().data(),
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock, kH});
    {
      vt::DFlashGroupedConvArgs a;
      a.block_size = kBlock; a.taps = 2; a.num_groups = 320; a.group_size = 16; a.side = 0;
      vt::DFlashGroupedConv(rcq, prep, dhn, coef, ac_base, a);
    }
    cbe.Synchronize(rcq);
    pcc(downf(prep, static_cast<size_t>(kBlock * kH)), attn_in64, "after ac prepare");

    vt::Tensor wq = mkbf(lw0.q, {kNH * kHD, kH});
    vt::Tensor wk = mkbf(lw0.k, {kNKV * kHD, kH});
    vt::Tensor wv = mkbf(lw0.v, {kNKV * kHD, kH});
    arena.emplace_back(static_cast<size_t>(kBlock * kNH * kHD));
      vt::Tensor qb = vt::Tensor::Contiguous(arena.back().data(),
                                       vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                       {kBlock, kNH * kHD});
    arena.emplace_back(static_cast<size_t>(kBlock * kNKV * kHD));
      vt::Tensor kbb = vt::Tensor::Contiguous(arena.back().data(),
                                        vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                        {kBlock, kNKV * kHD});
    arena.emplace_back(static_cast<size_t>(kBlock * kNKV * kHD));
      vt::Tensor vbb = vt::Tensor::Contiguous(arena.back().data(),
                                        vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                        {kBlock, kNKV * kHD});
    // the same buffer, rank-3 for DFlashBlockAttention (MatmulBT's out must be
    // rank-2, the attention's V must be [T,Hkv,D])
    vt::Tensor vbb3 = vt::Tensor::Contiguous(vbb.data,
                                        vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                        {kBlock, kNKV, kHD});
    vt::MatmulBT(rcq, qb, prep, wq);
    vt::MatmulBT(rcq, kbb, prep, wk);
    vt::MatmulBT(rcq, vbb, prep, wv);
    // q/k per-head norm on [T*heads, Dh]
    {
      arena.emplace_back(static_cast<size_t>(kBlock * kNH * kHD));
      vt::Tensor q2 = vt::Tensor::Contiguous(arena.back().data(),
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock * kNH, kHD});
      arena.emplace_back(static_cast<size_t>(kBlock * kNKV * kHD));
      vt::Tensor k2 = vt::Tensor::Contiguous(arena.back().data(),
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock * kNKV, kHD});
      // copy qb into q2 storage view: RmsNorm writes in place, so give it the
      // reinterpreted buffers by REBUILDING through vt::Tensor views
      std::memcpy(q2.data, qb.data, static_cast<size_t>(kBlock * kNH * kHD) * 2);
      std::memcpy(k2.data, kbb.data, static_cast<size_t>(kBlock * kNKV * kHD) * 2);
      vt::RmsNorm(rcq, q2, q2, mkbf1(lw0.qn, kHD), vt::RmsNormArgs{1e-6f, false});
      vt::RmsNorm(rcq, k2, k2, mkbf1(lw0.kn, kHD), vt::RmsNormArgs{1e-6f, false});
      arena.emplace_back(static_cast<size_t>(kBlock * kNH * kHD));
      vt::Tensor q3 = vt::Tensor::Contiguous(arena.back().data(),
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock, kNH, kHD});
      arena.emplace_back(static_cast<size_t>(kBlock * kNKV * kHD));
      vt::Tensor k3 = vt::Tensor::Contiguous(arena.back().data(),
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock, kNKV, kHD});
      std::memcpy(q3.data, q2.data, static_cast<size_t>(kBlock * kNH * kHD) * 2);
      std::memcpy(k3.data, k2.data, static_cast<size_t>(kBlock * kNKV * kHD) * 2);
      std::vector<int32_t> pos(static_cast<size_t>(kBlock));
      for (int64_t i = 0; i < kBlock; ++i) pos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
      vt::Tensor dpos = vt::Tensor::Contiguous(pos.data(), vt::DType::kI32,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock});
      vt::RopeNeox(rcq, q3, k3, dpos,
                   vt::RopeArgs{static_cast<float>(kRopeTheta), static_cast<int>(kHD), true, {}, false});
      cbe.Synchronize(rcq);
      // fp64: q normed+roped
      Mat q64 = MulWT(attn_in64, kBlock, kH, lw0.q, kNH * kHD);
      q64 = RoundBAll(HeadRms(q64, kBlock, kNH, lw0.qn));
      {
        Mat cos0, sin0;
        RopeTables(kBlock, &cos0, &sin0);
        q64 = RoundBAll(head_rope64(q64, kNH, cos0, sin0));
      }
      pcc(downf(q3, static_cast<size_t>(kBlock * kNH * kHD)), q64, "q normed+roped");
      Mat k64 = MulWT(attn_in64, kBlock, kH, lw0.k, kNKV * kHD);
      k64 = RoundBAll(HeadRms(k64, kBlock, kNKV, lw0.kn));
      {
        Mat cos0, sin0;
        RopeTables(kBlock, &cos0, &sin0);
        k64 = RoundBAll(head_rope64(k64, kNKV, cos0, sin0));
      }
      pcc(downf(k3, static_cast<size_t>(kBlock * kNKV * kHD)), k64, "k normed+roped");

      // attention over the 8 block rows only
      arena.emplace_back(static_cast<size_t>(kBlock * kNH * kHD));
      vt::Tensor ob = vt::Tensor::Contiguous(arena.back().data(),
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock, kNH, kHD});
      // rank-2 view of the SAME buffer for the o_proj MatmulBT (its out must be
      // rank-2; the attention's out must be [T,Hq,D])
      vt::Tensor ob2 = vt::Tensor::Contiguous(ob.data,
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock, kNH * kHD});
      std::vector<int32_t> cuv = {0, static_cast<int32_t>(kBlock)};
      vt::DFlashBlockAttentionArgs pa;
      pa.scale = 1.0f / std::sqrt(static_cast<float>(kHD));
      pa.causal = false;
      pa.sliding_window = 2048;
      pa.cu_seqlens = cuv.data();
      pa.num_reqs = 1;
      vt::DFlashBlockAttention(rcq, ob, q3, k3, vbb3, pa);
      cbe.Synchronize(rcq);
      Mat o64(static_cast<size_t>(kBlock * kNH * kHD));
      // fp64 attention over the same roped q/k, raw v
      Mat v64 = RoundBAll(MulWT(attn_in64, kBlock, kH, lw0.v, kNKV * kHD));
      for (int64_t t = 0; t < kBlock; ++t)
        for (int64_t h = 0; h < kNH; ++h) {
          const int64_t kvh = h / (kNH / kNKV);
          std::vector<double> sc(static_cast<size_t>(kBlock));
          double mx = -1e300;
          for (int64_t s = 0; s < kBlock; ++s) {
            double acc = 0.0;
            for (int64_t d = 0; d < kHD; ++d)
              acc += q64[static_cast<size_t>((t * kNH + h) * kHD + d)] *
                     k64[static_cast<size_t>((s * kNKV + kvh) * kHD + d)];
            sc[static_cast<size_t>(s)] = acc / std::sqrt(static_cast<double>(kHD));
            mx = std::max(mx, sc[static_cast<size_t>(s)]);
          }
          double den = 0.0;
          for (int64_t s = 0; s < kBlock; ++s) {
            sc[static_cast<size_t>(s)] = std::exp(sc[static_cast<size_t>(s)] - mx);
            den += sc[static_cast<size_t>(s)];
          }
          for (int64_t d = 0; d < kHD; ++d) {
            double acc = 0.0;
            for (int64_t s = 0; s < kBlock; ++s)
              acc += sc[static_cast<size_t>(s)] *
                     v64[static_cast<size_t>((s * kNKV + kvh) * kHD + d)];
            o64[static_cast<size_t>(t * kNH * kHD + h * kHD + d)] = acc / den;
          }
        }
      pcc(downf(ob, static_cast<size_t>(kBlock * kNH * kHD)), o64, "attention out");

      vt::Tensor wo = mkbf(lw0.o, {kH, kNH * kHD});
      arena.emplace_back(static_cast<size_t>(kBlock * kH));
      vt::Tensor a2 = vt::Tensor::Contiguous(arena.back().data(),
                                         vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                         {kBlock, kH});
      vt::MatmulBT(rcq, a2, ob2, wo);
      // finish conv side 1 — the chain (DflashConvFinish) reads stream and
      // writes a SEPARATE out buffer, so the replay must not convolve in place
      // either: out[i] would clobber x[i-1] for row i+1's t=1 tap.
      {
        vt::DFlashGroupedConvArgs a;
        a.block_size = kBlock; a.taps = 2; a.num_groups = 320; a.group_size = 16; a.side = 1;
        arena.emplace_back(static_cast<size_t>(kBlock * kH));
        vt::Tensor fin = vt::Tensor::Contiguous(arena.back().data(),
                                           vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                           {kBlock, kH});
        vt::DFlashGroupedConv(rcq, fin, a2, coef, ac_base, a);
        std::memcpy(a2.data, fin.data, static_cast<size_t>(kBlock * kH) * 2);
      }
      cbe.Synchronize(rcq);
      Mat a64 = RoundBAll(MulWT(RoundBAll(o64), kBlock, kNH * kHD, lw0.o, kH));
      Mat acv64 = Gdc(a64, kBlock, kp64, lw0.ac_b10, lw0.ac_b11, 1);
      pcc(downf(a2, static_cast<size_t>(kBlock * kH)), acv64, "after ac finish");

      // ---- MLP HALF of layer 0: residual add -> post_ln -> mc prepare ->
      // gate/up -> silu*up -> down -> mc finish. Same isolation as above.
      Mat h1_64(static_cast<size_t>(kBlock * kH));
      for (int64_t i = 0; i < kBlock * kH; ++i)
        h1_64[static_cast<size_t>(i)] =
            f.block_rows[static_cast<size_t>(i)] + acv64[static_cast<size_t>(i)];
      // the chain's add+RMSNorm: RmsNorm(dh2, attn, w_post, {eps,false}, &res)
      // — fused add+norm, res is the residual stream (= the block rows here)
      arena.emplace_back(static_cast<size_t>(kBlock * kH));
      vt::Tensor dh2 = vt::Tensor::Contiguous(arena.back().data(),
                                          vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                          {kBlock, kH});
      {
        // op-arm residual stream in bf16, exactly what the chain holds
        arena.emplace_back(static_cast<size_t>(kBlock * kH));
        vt::Tensor resb = vt::Tensor::Contiguous(arena.back().data(),
                                            vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                            {kBlock, kH});
        for (int64_t i = 0; i < kBlock * kH; ++i)
          reinterpret_cast<uint16_t*>(resb.data)[static_cast<size_t>(i)] =
              FloatToBf16(static_cast<float>(f.block_rows[static_cast<size_t>(i)]));
        vt::RmsNorm(rcq, dh2, a2, mkbf1(lw0.post_ln, kH), vt::RmsNormArgs{1e-6f, false},
                    &resb);
      }
      cbe.Synchronize(rcq);
      Mat h2_64 = RoundBAll(Rms(h1_64, kBlock, kH, lw0.post_ln));
      pcc(downf(dh2, static_cast<size_t>(kBlock * kH)), h2_64, "after post_ln");

      // mc prepare: coef from h2 (the sublayer INPUT), conv side 0
      arena.emplace_back(static_cast<size_t>(kBlock * 1280));
      vt::Tensor coefm = vt::Tensor::Contiguous(arena.back().data(),
                                          vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                          {kBlock, 1280});
      vt::Tensor mc_w = mkbf(lw0.mc_kp, {1280, kH});
      // rank-4 view of the SAME buffer for the conv (MatmulBT's out is rank-2)
      vt::Tensor coefm4 = vt::Tensor::Contiguous(coefm.data,
                                          vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                          {kBlock, 2, 2, 320});
      vt::MatmulBT(rcq, coefm, dh2, mc_w);
      vt::Tensor mc_base = mkbf(
          [&] { Mat b(4 * kH);
                for (int64_t i = 0; i < kH; ++i) {
                  b[i] = lw0.mc_b00[i]; b[kH + i] = lw0.mc_b01[i];
                  b[2 * kH + i] = lw0.mc_b10[i]; b[3 * kH + i] = lw0.mc_b11[i];
                }
                return b; }(),
          {2, 2, kH});
      arena.emplace_back(static_cast<size_t>(kBlock * kH));
      vt::Tensor mlp_in = vt::Tensor::Contiguous(arena.back().data(),
                                            vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                            {kBlock, kH});
      {
        vt::DFlashGroupedConvArgs a;
        a.block_size = kBlock; a.taps = 2; a.num_groups = 320; a.group_size = 16; a.side = 0;
        vt::DFlashGroupedConv(rcq, mlp_in, dh2, coefm4, mc_base, a);
      }
      cbe.Synchronize(rcq);
      Mat kpm64 = MulWT(h2_64, kBlock, kH, lw0.mc_kp, 1280);
      Mat mlp_in64 = Gdc(h2_64, kBlock, kpm64, lw0.mc_b00, lw0.mc_b01, 0);
      pcc(downf(mlp_in, static_cast<size_t>(kBlock * kH)), mlp_in64, "after mc prepare");

      // gate_up (the merged [2I, kH] GEMM the chain runs), then SiluAndMul
      {
        Mat gu(2 * kIV * kH);
        std::memcpy(&gu[0], lw0.gate.data(), sizeof(double) * lw0.gate.size());
        std::memcpy(&gu[static_cast<size_t>(kIV * kH)], lw0.up.data(),
                    sizeof(double) * lw0.up.size());
        vt::Tensor wgu = mkbf(gu, {2 * kIV, kH});
        arena.emplace_back(static_cast<size_t>(kBlock * 2 * kIV));
        vt::Tensor gate_up = vt::Tensor::Contiguous(arena.back().data(),
                                              vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                              {kBlock, 2 * kIV});
        vt::MatmulBT(rcq, gate_up, mlp_in, wgu);
        arena.emplace_back(static_cast<size_t>(kBlock * kIV));
        vt::Tensor act = vt::Tensor::Contiguous(arena.back().data(),
                                           vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                           {kBlock, kIV});
        vt::SiluAndMul(rcq, act, gate_up);
        cbe.Synchronize(rcq);
        Mat g64 = RoundBAll(MulWT(mlp_in64, kBlock, kH, lw0.gate, kIV));
        Mat u64 = RoundBAll(MulWT(mlp_in64, kBlock, kH, lw0.up, kIV));
        Mat act64(static_cast<size_t>(kBlock * kIV));
        for (int64_t i = 0; i < kBlock * kIV; ++i) {
          const double sv = 1.0 / (1.0 + std::exp(-g64[static_cast<size_t>(i)]));
          act64[static_cast<size_t>(i)] = RoundB(g64[static_cast<size_t>(i)] * sv * u64[static_cast<size_t>(i)]);
        }
        pcc(downf(act, static_cast<size_t>(kBlock * kIV)), act64, "after silu*up");

        // down GEMM, then mc finish over the SAME coefm (side 1)
        arena.emplace_back(static_cast<size_t>(kBlock * kH));
        vt::Tensor dwn = vt::Tensor::Contiguous(arena.back().data(),
                                           vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                           {kBlock, kH});
        vt::MatmulBT(rcq, dwn, act, mkbf(lw0.down, {kH, kIV}));
        {
          vt::DFlashGroupedConvArgs a;
          a.block_size = kBlock; a.taps = 2; a.num_groups = 320; a.group_size = 16; a.side = 1;
          arena.emplace_back(static_cast<size_t>(kBlock * kH));
          vt::Tensor mfin = vt::Tensor::Contiguous(arena.back().data(),
                                              vt::DType::kBF16, vt::Device{vt::DeviceType::kCPU, 0},
                                              {kBlock, kH});
          vt::DFlashGroupedConv(rcq, mfin, dwn, coefm4, mc_base, a);
          cbe.Synchronize(rcq);
          Mat m64 = RoundBAll(MulWT(act64, kBlock, kIV, lw0.down, kH));
          Mat mcv64 = RoundBAll(Gdc(m64, kBlock, kpm64, lw0.mc_b10, lw0.mc_b11, 1));
          pcc(downf(mfin, static_cast<size_t>(kBlock * kH)), mcv64, "after mc finish");
          {
            std::string ms;
            for (int i = 0; i < 8; ++i) ms += std::to_string(mcv64[static_cast<size_t>(i)]) + " ";
            MESSAGE("mcv64 first8: " << ms);
          }
        }
      }
      // STAGE-DUMP comparison (W2_REF_STAGE_DUMP + the product-side
      // VT_DBG_DFLASH_DUMP): read the REAL chain's layer-0 intermediates and
      // PCC each against the fp64 stage values computed above. The first
      // diverging dump names the exact real-chain stage.
      if (std::getenv("W2_REF_STAGE_DUMP") != nullptr) {
        auto read_bin = [&](const char* path, size_t n) {
          std::vector<float> v(n);
          std::FILE* fp = std::fopen(path, "rb");
          if (!fp) return v;
          const size_t got = std::fread(v.data(), 4, n, fp);
          std::fclose(fp);
          if (got != n) v.resize(got);
          return v;
        };
        const Mat hbf64 = FromF32([&] {
          std::vector<float> b(static_cast<size_t>(kBlock * kH));
          for (int64_t i = 0; i < kBlock * kH; ++i)
            b[static_cast<size_t>(i)] = static_cast<float>(f.block_rows[static_cast<size_t>(i)]);
          return b;
        }());
        pcc(read_bin("/tmp/w2-stage-hidden.bin", kBlock * kH), hbf64, "DUMP hidden");
        pcc(read_bin("/tmp/w2-stage-dhn.bin", kBlock * kH), attn_in64, "DUMP dhn (conv-prepared)");
        {
          Mat a64 = RoundBAll(MulWT(RoundBAll(o64), kBlock, kNH * kHD, lw0.o, kH));
          // NOTE: DflashConvFinish REBINDS `attn` in place, so this dump is the
          // CONV-FINISHED output — compare it against acv64, not the raw o_proj.
          {
            Mat a64 = RoundBAll(MulWT(RoundBAll(o64), kBlock, kNH * kHD, lw0.o, kH));
            Mat acv64 = Gdc(a64, kBlock, kp64, lw0.ac_b10, lw0.ac_b11, 1);
            pcc(read_bin("/tmp/w2-stage-attn_out.bin", kBlock * kH), acv64,
                "DUMP attn_out (conv-finished)");
          }
          {
            const std::vector<float> g = read_bin("/tmp/w2-stage-attn_out.bin", kBlock * kH);
            Mat a64p = RoundBAll(MulWT(RoundBAll(o64), kBlock, kNH * kHD, lw0.o, kH));
            std::string gs, rs;
            for (int i = 0; i < 8; ++i) {
              gs += std::to_string(g[static_cast<size_t>(i)]) + " ";
              rs += std::to_string(a64p[static_cast<size_t>(i)]) + " ";
            }
            MESSAGE("attn_out first8 cpp: " << gs << "| fp64: " << rs);
            // per-row PCC names a row-local defect
            for (int r = 0; r < kBlock; ++r) {
              const PccStats sr = Compare(g, a64p, static_cast<size_t>(r * kH),
                                          static_cast<size_t>(kH));
              if (sr.pcc < 0.9)
                MESSAGE("  attn_out row " << r << ": PCC " << sr.pcc << ", rel_rms "
                        << sr.rel_rms);
            }
          }
          Mat h1_64(static_cast<size_t>(kBlock * kH));
          Mat acv64 = Gdc(a64, kBlock, kp64, lw0.ac_b10, lw0.ac_b11, 1);
          for (int64_t i = 0; i < kBlock * kH; ++i)
            h1_64[static_cast<size_t>(i)] =
                f.block_rows[static_cast<size_t>(i)] + acv64[static_cast<size_t>(i)];
          Mat h2_64 = RoundBAll(Rms(h1_64, kBlock, kH, lw0.post_ln));
          pcc(read_bin("/tmp/w2-stage-dh2.bin", kBlock * kH), h2_64, "DUMP dh2");
          const Mat kpm64 = MulWT(h2_64, kBlock, kH, lw0.mc_kp, 1280);
          const Mat mlp_in64 = Gdc(h2_64, kBlock, kpm64, lw0.mc_b00, lw0.mc_b01, 0);
          Mat g64 = RoundBAll(MulWT(mlp_in64, kBlock, kH, lw0.gate, kIV));
          Mat u64 = RoundBAll(MulWT(mlp_in64, kBlock, kH, lw0.up, kIV));
          Mat act64(static_cast<size_t>(kBlock * kIV));
          for (int64_t i = 0; i < kBlock * kIV; ++i) {
            const double sv = 1.0 / (1.0 + std::exp(-g64[static_cast<size_t>(i)]));
            act64[static_cast<size_t>(i)] = RoundB(g64[static_cast<size_t>(i)] * sv * u64[static_cast<size_t>(i)]);
          }
          pcc(read_bin("/tmp/w2-stage-act.bin", kBlock * kIV), act64, "DUMP act");
          Mat m64 = RoundBAll(MulWT(act64, kBlock, kIV, lw0.down, kH));
          pcc(read_bin("/tmp/w2-stage-down.bin", kBlock * kH), m64, "DUMP down (pre mc-finish)");
          pcc(read_bin("/tmp/w2-stage-dh2ln.bin", kBlock * kH), h2_64, "DUMP dh2ln (post-LN)");
          pcc(read_bin("/tmp/w2-stage-mlpc.bin", kBlock * 1280), kpm64, "DUMP mlp_coef");
          {
            Mat mcvx = RoundBAll(Gdc(m64, kBlock, kpm64, lw0.mc_b10, lw0.mc_b11, 1));
            pcc(read_bin("/tmp/w2-stage-mdown.bin", kBlock * kH), mcvx, "DUMP mdown (post mc-finish)");
          }
          // the attention's own q/k (roped) and pre-o_proj output
          pcc(read_bin("/tmp/w2-stage-q3.bin", kBlock * kNH * kHD), q64, "DUMP q3 (roped)");
          pcc(read_bin("/tmp/w2-stage-k3.bin", kBlock * kNKV * kHD), k64, "DUMP k3 (roped)");
          pcc(read_bin("/tmp/w2-stage-v3.bin", kBlock * kNKV * kHD), v64, "DUMP v3");
          pcc(read_bin("/tmp/w2-stage-awttn.bin", kBlock * kNH * kHD), o64, "DUMP attention out");
          // o_proj ON THE DUMPED attention output: if this passes while the
          // chain's own o_proj failed, the chain's o_proj consumed something
          // other than the attention output it dumped.
          {
            const std::vector<float> ain = read_bin("/tmp/w2-stage-awttn.bin",
                                                    kBlock * kNH * kHD);
            arena.emplace_back(static_cast<size_t>(kBlock * kNH * kHD));
            vt::Tensor ibt = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                              vt::Device{vt::DeviceType::kCPU, 0},
                                              {kBlock, kNH * kHD});
            for (size_t i = 0; i < ain.size(); ++i)
              reinterpret_cast<uint16_t*>(ibt.data)[i] = FloatToBf16(ain[i]);
            arena.emplace_back(static_cast<size_t>(kBlock * kH));
            vt::Tensor a2t = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                              vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
            vt::MatmulBT(rcq, a2t, ibt, wo);
            cbe.Synchronize(rcq);
            Mat a64d = RoundBAll(MulWT(RoundBAll(o64), kBlock, kNH * kHD, lw0.o, kH));
            pcc(downf(a2t, static_cast<size_t>(kBlock * kH)), a64d, "o_proj on DUMPED input");
          }
        }
      }
    }
    cbe.DestroyQueue(rcq);
  }

  // CTX-REPLAY (W2_REF_CTX_REPLAY=1): the same op-by-op isolation WITH the
  // 16-row tap context — the one variable the context-free replay above cannot
  // see. Ends by comparing the whole emulated layer-0 MLP output against the
  // fp64 reference's pre-residual down.
  if (std::getenv("W2_REF_CTX_REPLAY") != nullptr) {
    vt::Backend& cbe2 = vt::GetBackend(vt::DeviceType::kCPU);
    vt::Queue rcq = cbe2.CreateQueue();
    const RefLayer& lw0 = f.ref.layers[0];
    std::vector<std::vector<uint16_t>> arena;
    auto mkbf = [&](const Mat& m, std::initializer_list<int64_t> shape) {
      int64_t n = 1;
      for (int64_t d : shape) n *= d;
      arena.emplace_back(static_cast<size_t>(n));
      std::vector<uint16_t>& b = arena.back();
      for (size_t i = 0; i < b.size(); ++i) b[i] = FloatToBf16(static_cast<float>(m[i]));
      return vt::Tensor::Contiguous(b.data(), vt::DType::kBF16,
                                    vt::Device{vt::DeviceType::kCPU, 0}, shape);
    };
    auto mkbf1 = [&](const Mat& m, int64_t n) {
      arena.emplace_back(static_cast<size_t>(n));
      std::vector<uint16_t>& b = arena.back();
      for (int64_t i = 0; i < n; ++i)
        b[static_cast<size_t>(i)] = FloatToBf16(static_cast<float>(m[static_cast<size_t>(i)]));
      return vt::Tensor::Contiguous(b.data(), vt::DType::kBF16,
                                    vt::Device{vt::DeviceType::kCPU, 0}, {n});
    };
    auto downf = [&](const vt::Tensor& t, size_t n) {
      std::vector<float> v(n);
      cbe2.Copy(rcq, v.data(), t.data, n * 2);
      cbe2.Synchronize(rcq);
      std::vector<float> out(n);
      for (size_t i = 0; i < n; ++i) out[i] = Bf16BitsToFloat(
          static_cast<uint16_t>(reinterpret_cast<const uint16_t*>(v.data())[i]));
      return out;
    };
    auto pcc = [&](const std::vector<float>& got, const Mat& refv, const std::string& name) {
      std::vector<float> gf(got.size());
      for (size_t i = 0; i < gf.size(); ++i)
        gf[i] = Bf16BitsToFloat(FloatToBf16(got[i]));
      const PccStats st = Compare(gf, refv, 0, refv.size());
      MESSAGE("ctx-replay " << name << ": PCC " << st.pcc << ", rel_rms " << st.rel_rms);
    };

    // ---- the CONTEXT arm: bf16 RmsNorm of the bf16-rounded fc matmul (what
    // AppendContextKVHost normalizes), then per-layer K/V + k-norm + rope at
    // positions 0..15.
    const Mat hctx64 = Rms(MulWT(FromF32(f.taps), kCtx, kH * kTaps, f.ref.fc, kH),
                           kCtx, kH, f.ref.hnorm);
    Mat fc64 = MulWT(FromF32(f.taps), kCtx, kH * kTaps, f.ref.fc, kH);
    {
      arena.emplace_back(static_cast<size_t>(kCtx * kH));
      std::vector<uint16_t>& b = arena.back();
      for (int64_t i = 0; i < kCtx * kH; ++i)
        b[static_cast<size_t>(i)] = FloatToBf16(static_cast<float>(fc64[static_cast<size_t>(i)]));
      vt::Tensor fcb = vt::Tensor::Contiguous(b.data(), vt::DType::kBF16,
                                          vt::Device{vt::DeviceType::kCPU, 0}, {kCtx, kH});
      arena.emplace_back(static_cast<size_t>(kCtx * kH));
      vt::Tensor hctxb = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                          vt::Device{vt::DeviceType::kCPU, 0}, {kCtx, kH});
      vt::RmsNorm(rcq, hctxb, fcb, mkbf1(f.ref.hnorm, kH), vt::RmsNormArgs{1e-6f, false});
      cbe2.Synchronize(rcq);
      pcc(downf(hctxb, static_cast<size_t>(kCtx * kH)), hctx64, "hctx");

      const int64_t kdim = kNKV * kHD;
      arena.emplace_back(static_cast<size_t>(kCtx * kdim));
      vt::Tensor kcb = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                          vt::Device{vt::DeviceType::kCPU, 0}, {kCtx, kdim});
      // rank-3 view of the SAME buffer for the rope
      vt::Tensor kcb3 = vt::Tensor::Contiguous(kcb.data, vt::DType::kBF16,
                                          vt::Device{vt::DeviceType::kCPU, 0}, {kCtx, kNKV, kHD});
      arena.emplace_back(static_cast<size_t>(kCtx * kdim));
      vt::Tensor vcb = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                          vt::Device{vt::DeviceType::kCPU, 0}, {kCtx, kNKV * kHD});
      vt::MatmulBT(rcq, kcb, hctxb, mkbf(lw0.k, {kNKV * kHD, kH}));
      vt::MatmulBT(rcq, vcb, hctxb, mkbf(lw0.v, {kNKV * kHD, kH}));
      {
        // k-norm over head_dim, then rope at 0..15
        arena.emplace_back(static_cast<size_t>(kCtx * kdim));
        vt::Tensor k2 = vt::Tensor::Contiguous(kcb.data, vt::DType::kBF16,
                                          vt::Device{vt::DeviceType::kCPU, 0},
                                          {kCtx * kNKV, kHD});
        std::memcpy(k2.data, kcb.data, static_cast<size_t>(kCtx * kdim) * 2);
        vt::RmsNorm(rcq, k2, k2, mkbf1(lw0.kn, kHD), vt::RmsNormArgs{1e-6f, false});
        std::memcpy(kcb.data, k2.data, static_cast<size_t>(kCtx * kdim) * 2);
        std::vector<int32_t> cpos(static_cast<size_t>(kCtx));
        for (int64_t i = 0; i < kCtx; ++i) cpos[static_cast<size_t>(i)] = static_cast<int32_t>(i);
        vt::Tensor dcpos = vt::Tensor::Contiguous(cpos.data(), vt::DType::kI32,
                                              vt::Device{vt::DeviceType::kCPU, 0}, {kCtx});
        // RopeNeox ropes BOTH of its first two tensors IN PLACE (the chain
        // hands it q3/k3 and a zeroed rope_scratch for the k output; the op is
        // the q/k pair). Give the second slot a COPY so kcb3 is roped once.
        arena.emplace_back(static_cast<size_t>(kCtx * kdim));
        vt::Tensor krope = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                              vt::Device{vt::DeviceType::kCPU, 0},
                                              {kCtx, kNKV, kHD});
        std::memcpy(krope.data, kcb.data, static_cast<size_t>(kCtx * kdim) * 2);
        vt::RopeNeox(rcq, kcb3, krope, dcpos,
                     vt::RopeArgs{static_cast<float>(kRopeTheta), static_cast<int>(kHD), true, {}, false});
        cbe2.Synchronize(rcq);
      }
      // fp64 companion for the context K/V (normed + roped at 0..15)
      Mat kc64 = RoundBAll(MulWT(hctx64, kCtx, kH, lw0.k, kNKV * kHD));
      Mat vcat64(static_cast<size_t>(kCtx * kNKV * kHD));
      std::memcpy(vcat64.data(), RoundBAll(MulWT(hctx64, kCtx, kH, lw0.v, kNKV * kHD)).data(),
                  sizeof(double) * vcat64.size());
      kc64 = RoundBAll(HeadRms(kc64, kCtx, kNKV, lw0.kn));
      {
        Mat cos0, sin0;
        RopeTables(kCtx + kBlock, &cos0, &sin0);
        Mat y(static_cast<size_t>(kCtx * kNKV * kHD), 0.0);
        for (int64_t hh = 0; hh < kNKV; ++hh) {
          Mat slice(static_cast<size_t>(kCtx * kHD), 0.0);
          for (int64_t t = 0; t < kCtx; ++t)
            std::memcpy(&slice[static_cast<size_t>(t * kHD)],
                        &kc64[static_cast<size_t>((t * kNKV + hh) * kHD)],
                        sizeof(double) * static_cast<size_t>(kHD));
          Mat roped = RopeApply(slice, kCtx, kHD, cos0, sin0, 0);
          for (int64_t t = 0; t < kCtx; ++t)
            std::memcpy(&y[static_cast<size_t>((t * kNKV + hh) * kHD)],
                        &roped[static_cast<size_t>(t * kHD)],
                        sizeof(double) * static_cast<size_t>(kHD));
        }
        kc64 = RoundBAll(y);
      }
      pcc(downf(kcb, static_cast<size_t>(kCtx * kNKV * kHD)), kc64, "ctx k normed+roped");
      // THE CHAIN'S OWN context K for layer 0, against both the fp64 reference
      // and this replay's validated op sequence: names AppendContextKVHost as
      // guilty or clean.
      {
        const int64_t kdim2 = kNKV * kHD;
        std::vector<float> ckvh(static_cast<size_t>(kCtx) * kdim2);
        for (size_t i = 0; i < ckvh.size(); ++i)
          ckvh[i] = Bf16BitsToFloat(ckv.k[0][i]);
        const PccStats st_ref = Compare(ckvh, kc64, 0, ckvh.size());
        const std::vector<float> mine = downf(kcb, ckvh.size());
        const Mat mine64(mine.begin(), mine.end());
        const PccStats st_mine = Compare(ckvh, mine64, 0, ckvh.size());
        MESSAGE("ctx-replay ckv.k[0] vs fp64: PCC " << st_ref.pcc << ", rel_rms "
                << st_ref.rel_rms << " | vs my replay ops: rel_rms " << st_mine.rel_rms);
        std::vector<float> cvh(static_cast<size_t>(kCtx) * kdim2);
        for (size_t i = 0; i < cvh.size(); ++i)
          cvh[i] = Bf16BitsToFloat(ckv.v[0][i]);
        const Mat vc64 = RoundBAll(MulWT(hctx64, kCtx, kH, lw0.v, kNKV * kHD));
        const PccStats stv_ref = Compare(cvh, vc64, 0, cvh.size());
        MESSAGE("ctx-replay ckv.v[0] vs fp64: PCC " << stv_ref.pcc << ", rel_rms " << stv_ref.rel_rms);
      }
      // HARNESS-vs-LOADER weight identity: the fp64 reference and this replay
      // both read the HARNESS's tensor names; the real chain reads the LOADER's.
      // If they disagree, the replay is circular and blind to the divergence.
      {
        const Qwen3DFlashLayerWeights& lw = f.weights.layers[0];
        // merged gate_up halves vs the harness's gate/up (bf16, 2 bytes/elem)
        {
          const Qwen3DFlashLayerWeights& L = lw;
          auto same = [&](const std::vector<uint16_t>& w, const Mat& ref) {
            if (w.size() != ref.size()) return std::string("SIZE ") +
                std::to_string(w.size()) + " vs " + std::to_string(ref.size());
            for (size_t i = 0; i < w.size(); ++i)
              if (Bf16BitsToFloat(w[i]) != static_cast<float>(ref[i]))
                return std::string("DIFFERS at ") + std::to_string(i);
            return std::string("IDENTICAL");
          };
          const size_t half = L.gate_up_proj.bytes.size() / 4;  // uint16 elements
          std::vector<uint16_t> ghalf(half), uhalf(half);
          std::memcpy(ghalf.data(), L.gate_up_proj.bytes.data(), half * 2);
          std::memcpy(uhalf.data(),
                      static_cast<const uint8_t*>(L.gate_up_proj.bytes.data()) + half * 2,
                      half * 2);
          MESSAGE("weights ac_kp: " << same(ToU16(L.attention_conv.kernel_projection.bytes), lw0.ac_kp));
          // base_kernel is [2, taps=2, H]: the harness's b00/b01 are side 0's two
          // tap rows, b10/b11 side 1's. Compare each 5120-elem tap row.
          {
            const auto ac_side0 = BaseSlice(L.attention_conv.base_kernel, 0);
            const auto ac_side1 = BaseSlice(L.attention_conv.base_kernel, 1);
            const auto mc_side0 = BaseSlice(L.mlp_conv.base_kernel, 0);
            const auto mc_side1 = BaseSlice(L.mlp_conv.base_kernel, 1);
            const auto half = [&](const std::vector<uint16_t>& s, int tap) {
              return std::vector<uint16_t>(s.begin() + tap * kH, s.begin() + (tap + 1) * kH);
            };
            MESSAGE("weights ac base s0t0: " << same(half(ac_side0, 0), lw0.ac_b00));
            MESSAGE("weights ac base s0t1: " << same(half(ac_side0, 1), lw0.ac_b01));
            MESSAGE("weights ac base s1t0: " << same(half(ac_side1, 0), lw0.ac_b10));
            MESSAGE("weights ac base s1t1: " << same(half(ac_side1, 1), lw0.ac_b11));
            MESSAGE("weights mc base s0t0: " << same(half(mc_side0, 0), lw0.mc_b00));
            MESSAGE("weights mc base s0t1: " << same(half(mc_side0, 1), lw0.mc_b01));
            MESSAGE("weights mc base s1t0: " << same(half(mc_side1, 0), lw0.mc_b10));
            MESSAGE("weights mc base s1t1: " << same(half(mc_side1, 1), lw0.mc_b11));
          }
          MESSAGE("weights gate: " << same(ghalf, lw0.gate));
          MESSAGE("weights up: " << same(uhalf, lw0.up));
          MESSAGE("weights down: " << same(ToU16(L.down_proj.bytes), lw0.down));
          MESSAGE("weights q: " << same(ToU16(L.qkv_proj.bytes), [&]{
              Mat m(static_cast<size_t>((kNH + 2 * kNKV) * kHD * kH));
              std::memcpy(m.data(), lw0.q.data(), sizeof(double) * lw0.q.size());
              std::memcpy(m.data() + lw0.q.size(), lw0.k.data(), sizeof(double) * lw0.k.size());
              std::memcpy(m.data() + lw0.q.size() + lw0.k.size(), lw0.v.data(),
                          sizeof(double) * lw0.v.size());
              return m; }()));
          MESSAGE("weights o: " << same(ToU16(L.o_proj.bytes), lw0.o));
          MESSAGE("weights in_ln: " << same(ToU16(L.input_layernorm.bytes), lw0.in_ln));
          MESSAGE("weights post_ln: " << same(ToU16(L.post_attention_layernorm.bytes), lw0.post_ln));
          MESSAGE("weights qn: " << same(ToU16(L.q_norm.bytes), lw0.qn));
          MESSAGE("weights kn: " << same(ToU16(L.k_norm.bytes), lw0.kn));
          MESSAGE("weights hnorm: " << same(ToU16(f.weights.hidden_norm.bytes), f.ref.hnorm));
          MESSAGE("weights fc: " << same(ToU16(f.weights.fc.bytes), f.ref.fc));
          (void)ghalf; (void)uhalf;
        }
        MESSAGE("weights dims: ac_kp rank " << lw.attention_conv.kernel_projection.rank
                << " shape " << lw.attention_conv.kernel_projection.shape[0] << "x"
                << lw.attention_conv.kernel_projection.shape[1]
                << " | harness ac_kp " << lw0.ac_kp.size() << " fp64 values"
                << " | base_kernel rank " << lw.attention_conv.base_kernel.rank
                << " | qkv rank " << lw.qkv_proj.rank << " shape "
                << lw.qkv_proj.shape[0] << "x" << lw.qkv_proj.shape[1]
                << " | gate_up shape " << lw.gate_up_proj.shape[0] << "x"
                << lw.gate_up_proj.shape[1]
                << " | conv_taps " << f.weights.conv_taps
                << " conv_block_size " << f.weights.conv_block_size
                << " | fp64 q rows " << lw0.q.size() / kH);
      }

      // ---- the BLOCK arm, identical to the context-free replay through rope,
      // then the COMBINED [ctx; block] attention (cu_seqlens_q semantics).
      Mat hn64 = RoundBAll(Rms(f.block_rows, kBlock, kH, lw0.in_ln));
      Mat kp64 = RoundBAll(MulWT(hn64, kBlock, kH, lw0.ac_kp, 1280));
      Mat attn_in64 = Gdc(hn64, kBlock, kp64, lw0.ac_b00, lw0.ac_b01, 0);
      arena.emplace_back(static_cast<size_t>(kBlock * kH));
      vt::Tensor dhn = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                          vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
      vt::RmsNorm(rcq, dhn, mkbf(f.block_rows, {kBlock, kH}), mkbf1(lw0.in_ln, kH),
                  vt::RmsNormArgs{1e-6f, false});
      arena.emplace_back(static_cast<size_t>(kBlock * 1280));
      vt::Tensor coef = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                          vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, 1280});
      vt::MatmulBT(rcq, coef, dhn, mkbf(lw0.ac_kp, {1280, kH}));
      vt::Tensor coef4 = vt::Tensor::Contiguous(coef.data, vt::DType::kBF16,
                                          vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, 2, 2, 320});
      vt::Tensor ac_base = mkbf(
          [&] { Mat b(4 * kH);
                for (int64_t i = 0; i < kH; ++i) {
                  b[i] = lw0.ac_b00[i]; b[kH + i] = lw0.ac_b01[i];
                  b[2 * kH + i] = lw0.ac_b10[i]; b[3 * kH + i] = lw0.ac_b11[i];
                }
                return b; }(),
          {2, 2, kH});
      arena.emplace_back(static_cast<size_t>(kBlock * kH));
      vt::Tensor hb = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                         vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
      {
        vt::DFlashGroupedConvArgs a;
        a.block_size = kBlock; a.taps = 2; a.num_groups = 320; a.group_size = 16; a.side = 0;
        vt::DFlashGroupedConv(rcq, hb, dhn, coef4, ac_base, a);
      }
      arena.emplace_back(static_cast<size_t>(kBlock * kNH * kHD));
      vt::Tensor qb = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                         vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kNH * kHD});
      arena.emplace_back(static_cast<size_t>(kBlock * kNKV * kHD));
      vt::Tensor kbb = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                         vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kNKV * kHD});
      arena.emplace_back(static_cast<size_t>(kBlock * kNKV * kHD));
      vt::Tensor vbb = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                         vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kNKV * kHD});
      vt::MatmulBT(rcq, qb, hb, mkbf(lw0.q, {kNH * kHD, kH}));
      vt::MatmulBT(rcq, kbb, hb, mkbf(lw0.k, {kNKV * kHD, kH}));
      vt::MatmulBT(rcq, vbb, hb, mkbf(lw0.v, {kNKV * kHD, kH}));
      {
        arena.emplace_back(static_cast<size_t>(kBlock * kNH * kHD));
        vt::Tensor q2 = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock * kNH, kHD});
        arena.emplace_back(static_cast<size_t>(kBlock * kNKV * kHD));
        vt::Tensor k2 = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock * kNKV, kHD});
        std::memcpy(q2.data, qb.data, static_cast<size_t>(kBlock * kNH * kHD) * 2);
        std::memcpy(k2.data, kbb.data, static_cast<size_t>(kBlock * kNKV * kHD) * 2);
        vt::RmsNorm(rcq, q2, q2, mkbf1(lw0.qn, kHD), vt::RmsNormArgs{1e-6f, false});
        vt::RmsNorm(rcq, k2, k2, mkbf1(lw0.kn, kHD), vt::RmsNormArgs{1e-6f, false});
        arena.emplace_back(static_cast<size_t>(kBlock * kNH * kHD));
        vt::Tensor q3 = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kNH, kHD});
        arena.emplace_back(static_cast<size_t>(kBlock * kNKV * kHD));
        vt::Tensor k3 = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kNKV, kHD});
        std::memcpy(q3.data, q2.data, static_cast<size_t>(kBlock * kNH * kHD) * 2);
        std::memcpy(k3.data, k2.data, static_cast<size_t>(kBlock * kNKV * kHD) * 2);
        std::vector<int32_t> bposv(static_cast<size_t>(kBlock));
        for (int64_t i = 0; i < kBlock; ++i)
          bposv[static_cast<size_t>(i)] = static_cast<int32_t>(kCtx + i);
        vt::Tensor dbpos = vt::Tensor::Contiguous(bposv.data(), vt::DType::kI32,
                                              vt::Device{vt::DeviceType::kCPU, 0}, {kBlock});
        vt::RopeNeox(rcq, q3, k3, dbpos,
                     vt::RopeArgs{static_cast<float>(kRopeTheta), static_cast<int>(kHD), true, {}, false});
        cbe2.Synchronize(rcq);

        // COMBINED attention: keys [ctx(16); block(8)], queries = block suffix.
        const int64_t T = kCtx + kBlock;
        arena.emplace_back(static_cast<size_t>(T * kNKV * kHD));
        vt::Tensor kcat = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                            vt::Device{vt::DeviceType::kCPU, 0}, {T, kNKV, kHD});
        arena.emplace_back(static_cast<size_t>(T * kNKV * kHD));
        vt::Tensor vcat = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                            vt::Device{vt::DeviceType::kCPU, 0}, {T, kNKV, kHD});
        std::memcpy(kcat.data, kcb.data, static_cast<size_t>(kCtx * kNKV * kHD) * 2);
        std::memcpy(vcat.data, vcb.data, static_cast<size_t>(kCtx * kNKV * kHD) * 2);
        std::memcpy(reinterpret_cast<uint8_t*>(kcat.data) + static_cast<size_t>(kCtx * kNKV * kHD) * 2,
                    k3.data, static_cast<size_t>(kBlock * kNKV * kHD) * 2);
        std::memcpy(reinterpret_cast<uint8_t*>(vcat.data) + static_cast<size_t>(kCtx * kNKV * kHD) * 2,
                    vbb.data, static_cast<size_t>(kBlock * kNKV * kHD) * 2);
        arena.emplace_back(static_cast<size_t>(kBlock * kNH * kHD));
        vt::Tensor ob = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kNH, kHD});
        std::vector<int32_t> cuv = {0, static_cast<int32_t>(T)};
        std::vector<int32_t> qcu = {0, static_cast<int32_t>(kBlock)};
        vt::DFlashBlockAttentionArgs pa;
        pa.scale = 1.0f / std::sqrt(static_cast<float>(kHD));
        pa.causal = false;
        pa.sliding_window = 2048;
        pa.cu_seqlens = cuv.data();
        pa.cu_seqlens_q = qcu.data();
        pa.num_reqs = 1;
        vt::DFlashBlockAttention(rcq, ob, q3, kcat, vcat, pa);
        cbe2.Synchronize(rcq);
        // fp64 combined attention over 24 keys
        Mat kb64 = RoundBAll(MulWT(attn_in64, kBlock, kH, lw0.k, kNKV * kHD));
        kb64 = RoundBAll(HeadRms(kb64, kBlock, kNKV, lw0.kn));
        {
          Mat cos0, sin0;
          RopeTables(T, &cos0, &sin0);
          Mat y(static_cast<size_t>(kBlock * kNKV * kHD), 0.0);
          for (int64_t hh = 0; hh < kNKV; ++hh) {
            Mat slice(static_cast<size_t>(kBlock * kHD), 0.0);
            for (int64_t t = 0; t < kBlock; ++t)
              std::memcpy(&slice[static_cast<size_t>(t * kHD)],
                          &kb64[static_cast<size_t>((t * kNKV + hh) * kHD)],
                          sizeof(double) * static_cast<size_t>(kHD));
            Mat roped = RopeApply(slice, kBlock, kHD, cos0, sin0, kCtx);
            for (int64_t t = 0; t < kBlock; ++t)
              std::memcpy(&y[static_cast<size_t>((t * kNKV + hh) * kHD)],
                          &roped[static_cast<size_t>(t * kHD)],
                          sizeof(double) * static_cast<size_t>(kHD));
          }
          kb64 = RoundBAll(y);
        }
        Mat q64 = RoundBAll(MulWT(attn_in64, kBlock, kH, lw0.q, kNH * kHD));
        q64 = RoundBAll(HeadRms(q64, kBlock, kNH, lw0.qn));
        {
          Mat cos0, sin0;
          RopeTables(T, &cos0, &sin0);
          Mat y(static_cast<size_t>(kBlock * kNH * kHD), 0.0);
          for (int64_t hh = 0; hh < kNH; ++hh) {
            Mat slice(static_cast<size_t>(kBlock * kHD), 0.0);
            for (int64_t t = 0; t < kBlock; ++t)
              std::memcpy(&slice[static_cast<size_t>(t * kHD)],
                          &q64[static_cast<size_t>((t * kNH + hh) * kHD)],
                          sizeof(double) * static_cast<size_t>(kHD));
            Mat roped = RopeApply(slice, kBlock, kHD, cos0, sin0, kCtx);
            for (int64_t t = 0; t < kBlock; ++t)
              std::memcpy(&y[static_cast<size_t>((t * kNH + hh) * kHD)],
                          &roped[static_cast<size_t>(t * kHD)],
                          sizeof(double) * static_cast<size_t>(kHD));
          }
          q64 = RoundBAll(y);
        }
        Mat kcat64(static_cast<size_t>(T * kNKV * kHD));
        std::memcpy(&kcat64[0], kc64.data(), sizeof(double) * kc64.size());
        std::memcpy(&kcat64[static_cast<size_t>(kCtx * kNKV * kHD)], kb64.data(),
                    sizeof(double) * kb64.size());
        std::memcpy(&vcat64[static_cast<size_t>(kCtx * kNKV * kHD)],
                    RoundBAll(MulWT(attn_in64, kBlock, kH, lw0.v, kNKV * kHD)).data(),
                    sizeof(double) * static_cast<size_t>(kBlock * kNKV * kHD));
        Mat o64ctx(static_cast<size_t>(kBlock * kNH * kHD));
        for (int64_t t = 0; t < kBlock; ++t)
          for (int64_t h = 0; h < kNH; ++h) {
            const int64_t kvh = h / (kNH / kNKV);
            std::vector<double> sc(static_cast<size_t>(T));
            double mx = -1e300;
            for (int64_t s2 = 0; s2 < T; ++s2) {
              double acc = 0.0;
              for (int64_t d = 0; d < kHD; ++d)
                acc += q64[static_cast<size_t>((t * kNH + h) * kHD + d)] *
                       kcat64[static_cast<size_t>((s2 * kNKV + kvh) * kHD + d)];
              sc[static_cast<size_t>(s2)] = acc / std::sqrt(static_cast<double>(kHD));
              mx = std::max(mx, sc[static_cast<size_t>(s2)]);
            }
            double den = 0.0;
            for (int64_t s2 = 0; s2 < T; ++s2) {
              sc[static_cast<size_t>(s2)] = std::exp(sc[static_cast<size_t>(s2)] - mx);
              den += sc[static_cast<size_t>(s2)];
            }
            for (int64_t d = 0; d < kHD; ++d) {
              double acc = 0.0;
              for (int64_t s2 = 0; s2 < T; ++s2)
                acc += sc[static_cast<size_t>(s2)] *
                       vcat64[static_cast<size_t>((s2 * kNKV + kvh) * kHD + d)];
              o64ctx[static_cast<size_t>(t * kNH * kHD + h * kHD + d)] = acc / den;
            }
          }
        pcc(downf(ob, static_cast<size_t>(kBlock * kNH * kHD)), o64ctx, "combined attention out");

        // finish the layer exactly as the chain does
        vt::Tensor ob2 = vt::Tensor::Contiguous(ob.data, vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kNH * kHD});
        arena.emplace_back(static_cast<size_t>(kBlock * kH));
        vt::Tensor a2 = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                           vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
        vt::MatmulBT(rcq, a2, ob2, mkbf(lw0.o, {kH, kNH * kHD}));
        {
          vt::DFlashGroupedConvArgs a;
          a.block_size = kBlock; a.taps = 2; a.num_groups = 320; a.group_size = 16; a.side = 1;
          arena.emplace_back(static_cast<size_t>(kBlock * kH));
          vt::Tensor fin = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                              vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
          vt::DFlashGroupedConv(rcq, fin, a2, coef4, ac_base, a);
          std::memcpy(a2.data, fin.data, static_cast<size_t>(kBlock * kH) * 2);
        }
        cbe2.Synchronize(rcq);
        Mat a64 = RoundBAll(MulWT(RoundBAll(o64ctx), kBlock, kNH * kHD, lw0.o, kH));
        Mat acv64 = Gdc(a64, kBlock, kp64, lw0.ac_b10, lw0.ac_b11, 1);
        pcc(downf(a2, static_cast<size_t>(kBlock * kH)), acv64, "after ac finish (ctx)");

        // fused add + post_ln (the residual stream is the block rows)
        Mat h1_64(static_cast<size_t>(kBlock * kH));
        for (int64_t i = 0; i < kBlock * kH; ++i)
          h1_64[static_cast<size_t>(i)] =
              f.block_rows[static_cast<size_t>(i)] + acv64[static_cast<size_t>(i)];
        arena.emplace_back(static_cast<size_t>(kBlock * kH));
        vt::Tensor dh2 = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                            vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
        {
          arena.emplace_back(static_cast<size_t>(kBlock * kH));
          vt::Tensor resb = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                              vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
          for (int64_t i = 0; i < kBlock * kH; ++i)
            reinterpret_cast<uint16_t*>(resb.data)[static_cast<size_t>(i)] =
                FloatToBf16(static_cast<float>(f.block_rows[static_cast<size_t>(i)]));
          vt::RmsNorm(rcq, dh2, a2, mkbf1(lw0.post_ln, kH), vt::RmsNormArgs{1e-6f, false}, &resb);
        }
        cbe2.Synchronize(rcq);
        Mat h2_64 = RoundBAll(Rms(h1_64, kBlock, kH, lw0.post_ln));
        pcc(downf(dh2, static_cast<size_t>(kBlock * kH)), h2_64, "after post_ln (ctx)");

        arena.emplace_back(static_cast<size_t>(kBlock * 1280));
        vt::Tensor coefm = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                            vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, 1280});
        vt::MatmulBT(rcq, coefm, dh2, mkbf(lw0.mc_kp, {1280, kH}));
        vt::Tensor coefm4 = vt::Tensor::Contiguous(coefm.data, vt::DType::kBF16,
                                             vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, 2, 2, 320});
        vt::Tensor mc_base = mkbf(
            [&] { Mat b(4 * kH);
                  for (int64_t i = 0; i < kH; ++i) {
                    b[i] = lw0.mc_b00[i]; b[kH + i] = lw0.mc_b01[i];
                    b[2 * kH + i] = lw0.mc_b10[i]; b[3 * kH + i] = lw0.mc_b11[i];
                  }
                  return b; }(),
            {2, 2, kH});
        arena.emplace_back(static_cast<size_t>(kBlock * kH));
        vt::Tensor mlp_in = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                              vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
        {
          vt::DFlashGroupedConvArgs a;
          a.block_size = kBlock; a.taps = 2; a.num_groups = 320; a.group_size = 16; a.side = 0;
          vt::DFlashGroupedConv(rcq, mlp_in, dh2, coefm4, mc_base, a);
        }
        cbe2.Synchronize(rcq);
        Mat kpm64 = MulWT(h2_64, kBlock, kH, lw0.mc_kp, 1280);
        Mat mlp_in64 = Gdc(h2_64, kBlock, kpm64, lw0.mc_b00, lw0.mc_b01, 0);
        pcc(downf(mlp_in, static_cast<size_t>(kBlock * kH)), mlp_in64, "after mc prepare (ctx)");

        {
          Mat gu(2 * kIV * kH);
          std::memcpy(&gu[0], lw0.gate.data(), sizeof(double) * lw0.gate.size());
          std::memcpy(&gu[static_cast<size_t>(kIV * kH)], lw0.up.data(),
                      sizeof(double) * lw0.up.size());
          vt::Tensor wgu = mkbf(gu, {2 * kIV, kH});
          arena.emplace_back(static_cast<size_t>(kBlock * 2 * kIV));
          vt::Tensor gate_up = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                                vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, 2 * kIV});
          vt::MatmulBT(rcq, gate_up, mlp_in, wgu);
          arena.emplace_back(static_cast<size_t>(kBlock * kIV));
          vt::Tensor act = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                             vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kIV});
          vt::SiluAndMul(rcq, act, gate_up);
          cbe2.Synchronize(rcq);
          Mat g64 = RoundBAll(MulWT(mlp_in64, kBlock, kH, lw0.gate, kIV));
          Mat u64 = RoundBAll(MulWT(mlp_in64, kBlock, kH, lw0.up, kIV));
          Mat act64(static_cast<size_t>(kBlock * kIV));
          for (int64_t i = 0; i < kBlock * kIV; ++i) {
            const double sv = 1.0 / (1.0 + std::exp(-g64[static_cast<size_t>(i)]));
            act64[static_cast<size_t>(i)] = RoundB(g64[static_cast<size_t>(i)] * sv * u64[static_cast<size_t>(i)]);
          }
          pcc(downf(act, static_cast<size_t>(kBlock * kIV)), act64, "after silu*up (ctx)");

          arena.emplace_back(static_cast<size_t>(kBlock * kH));
          vt::Tensor dwn = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                             vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
          vt::MatmulBT(rcq, dwn, act, mkbf(lw0.down, {kH, kIV}));
          {
            vt::DFlashGroupedConvArgs a;
            a.block_size = kBlock; a.taps = 2; a.num_groups = 320; a.group_size = 16; a.side = 1;
            arena.emplace_back(static_cast<size_t>(kBlock * kH));
            vt::Tensor mfin = vt::Tensor::Contiguous(arena.back().data(), vt::DType::kBF16,
                                                vt::Device{vt::DeviceType::kCPU, 0}, {kBlock, kH});
            vt::DFlashGroupedConv(rcq, mfin, dwn, coefm4, mc_base, a);
            cbe2.Synchronize(rcq);
            Mat m64 = RoundBAll(MulWT(act64, kBlock, kIV, lw0.down, kH));
            Mat mcv64 = RoundBAll(Gdc(m64, kBlock, kpm64, lw0.mc_b10, lw0.mc_b11, 1));
            pcc(downf(mfin, static_cast<size_t>(kBlock * kH)), mcv64, "after mc finish (ctx)");
            {
              // THE REAL CHAIN's with-ctx layer-0 output against THIS fp64:
              const Mat mcvd(mcv64.begin(), mcv64.end());
              const PccStats st_pc = Compare(per_layer[0], mcvd, 0,
                                             static_cast<size_t>(kBlock * kH));
              MESSAGE("DECISIVE-CTX: real chain per_layer[0] vs ctx fp64(mine): PCC "
                      << st_pc.pcc << ", rel_rms " << st_pc.rel_rms);
            }
            {
              auto dumpb = [&](const char* tag, const Mat& m) {
                std::FILE* fp = std::fopen((std::string("/tmp/w2-ctxmine-") + tag + ".bin").c_str(), "wb");
                for (double v : m) {
                  const float f = static_cast<float>(v);
                  std::fwrite(&f, 4, 1, fp);
                }
                std::fclose(fp);
              };
              dumpb("attn_in", attn_in64);
              dumpb("a", a64);
              dumpb("h2", h2_64);
              dumpb("mlp_in", mlp_in64);
              dumpb("act", act64);
              dumpb("m_pre", m64);
              dumpb("mcv", mcv64);
              dumpb("qh", q64);
      dumpb("qpre", RoundBAll(MulWT(attn_in64, kBlock, kH, lw0.q, kNH * kHD)));
      dumpb("qhn", RoundBAll(HeadRms(RoundBAll(MulWT(attn_in64, kBlock, kH, lw0.q, kNH * kHD)), kBlock, kNH, lw0.qn)));
              dumpb("kh", kcat64);
              dumpb("o2d", o64ctx);
            }
          }
        }
      }
    }
    cbe2.DestroyQueue(rcq);
  }

  // THE DECISIVE COMPARISON: the real chain's post-mc-finish layer-0 output
  // (per_layer[0]) against the fp64 mcvx computed by the SAME formulas the
  // replay validated, and against the reference's down — names WHICH SIDE of
  // the reference gate moves.
  {
    const RefLayer& lw0b = f.ref.layers[0];
    Mat hn_b = RoundBAll(Rms(f.block_rows, kBlock, kH, lw0b.in_ln));
    Mat kp_b = MulWT(hn_b, kBlock, kH, lw0b.ac_kp, 1280);
    Mat attn_in_b = Gdc(hn_b, kBlock, kp_b, lw0b.ac_b00, lw0b.ac_b01, 0);
    Mat cos8, sin8;
    RopeTables(kBlock, &cos8, &sin8);
    Mat q_b = RoundBAll(MulWT(attn_in_b, kBlock, kH, lw0b.q, kNH * kHD));
    q_b = RoundBAll(HeadRms(q_b, kBlock, kNH, lw0b.qn));
    {
      Mat y(static_cast<size_t>(kBlock * kNH * kHD), 0.0);
      for (int64_t hh = 0; hh < kNH; ++hh) {
        Mat sl(static_cast<size_t>(kBlock * kHD), 0.0);
        for (int64_t t = 0; t < kBlock; ++t)
          std::memcpy(&sl[static_cast<size_t>(t * kHD)],
                      &q_b[static_cast<size_t>((t * kNH + hh) * kHD)],
                      sizeof(double) * static_cast<size_t>(kHD));
        Mat rp = RopeApply(sl, kBlock, kHD, cos8, sin8, 0);
        for (int64_t t = 0; t < kBlock; ++t)
          std::memcpy(&y[static_cast<size_t>((t * kNH + hh) * kHD)],
                      &rp[static_cast<size_t>(t * kHD)], sizeof(double) * static_cast<size_t>(kHD));
      }
      q_b = RoundBAll(y);
    }
    Mat k_b = RoundBAll(MulWT(attn_in_b, kBlock, kH, lw0b.k, kNKV * kHD));
    k_b = RoundBAll(HeadRms(k_b, kBlock, kNKV, lw0b.kn));
    {
      Mat y(static_cast<size_t>(kBlock * kNKV * kHD), 0.0);
      for (int64_t hh = 0; hh < kNKV; ++hh) {
        Mat sl(static_cast<size_t>(kBlock * kHD), 0.0);
        for (int64_t t = 0; t < kBlock; ++t)
          std::memcpy(&sl[static_cast<size_t>(t * kHD)],
                      &k_b[static_cast<size_t>((t * kNKV + hh) * kHD)],
                      sizeof(double) * static_cast<size_t>(kHD));
        Mat rp = RopeApply(sl, kBlock, kHD, cos8, sin8, 0);
        for (int64_t t = 0; t < kBlock; ++t)
          std::memcpy(&y[static_cast<size_t>((t * kNKV + hh) * kHD)],
                      &rp[static_cast<size_t>(t * kHD)], sizeof(double) * static_cast<size_t>(kHD));
      }
      k_b = RoundBAll(y);
    }
    Mat v_b = RoundBAll(MulWT(attn_in_b, kBlock, kH, lw0b.v, kNKV * kHD));
    Mat o_b(static_cast<size_t>(kBlock * kNH * kHD), 0.0);
    for (int64_t t = 0; t < kBlock; ++t)
      for (int64_t h = 0; h < kNH; ++h) {
        const int64_t kvh = h / (kNH / kNKV);
        std::vector<double> sc(static_cast<size_t>(kBlock));
        double mx = -1e300;
        for (int64_t s2 = 0; s2 < kBlock; ++s2) {
          double acc = 0.0;
          for (int64_t d = 0; d < kHD; ++d)
            acc += q_b[static_cast<size_t>((t * kNH + h) * kHD + d)] *
                   k_b[static_cast<size_t>((s2 * kNKV + kvh) * kHD + d)];
          sc[static_cast<size_t>(s2)] = acc / std::sqrt(static_cast<double>(kHD));
          mx = std::max(mx, sc[static_cast<size_t>(s2)]);
        }
        double den = 0.0;
        for (int64_t s2 = 0; s2 < kBlock; ++s2) {
          sc[static_cast<size_t>(s2)] = std::exp(sc[static_cast<size_t>(s2)] - mx);
          den += sc[static_cast<size_t>(s2)];
        }
        for (int64_t d = 0; d < kHD; ++d) {
          double acc = 0.0;
          for (int64_t s2 = 0; s2 < kBlock; ++s2)
            acc += sc[static_cast<size_t>(s2)] *
                   v_b[static_cast<size_t>((s2 * kNKV + kvh) * kHD + d)];
          o_b[static_cast<size_t>(t * kNH * kHD + h * kHD + d)] = acc / den;
        }
      }
    Mat a_b = RoundBAll(MulWT(RoundBAll(o_b), kBlock, kNH * kHD, lw0b.o, kH));
    Mat acv_b = Gdc(a_b, kBlock, kp_b, lw0b.ac_b10, lw0b.ac_b11, 1);
    Mat h1_b(static_cast<size_t>(kBlock * kH));
    for (int64_t i = 0; i < kBlock * kH; ++i)
      h1_b[static_cast<size_t>(i)] = f.block_rows[static_cast<size_t>(i)] + acv_b[static_cast<size_t>(i)];
    Mat h2_b = RoundBAll(Rms(h1_b, kBlock, kH, lw0b.post_ln));
    Mat kpm_b = MulWT(h2_b, kBlock, kH, lw0b.mc_kp, 1280);
    Mat mlp_in_b = Gdc(h2_b, kBlock, kpm_b, lw0b.mc_b00, lw0b.mc_b01, 0);
    Mat g_b = RoundBAll(MulWT(mlp_in_b, kBlock, kH, lw0b.gate, kIV));
    Mat u_b = RoundBAll(MulWT(mlp_in_b, kBlock, kH, lw0b.up, kIV));
    Mat act_b(static_cast<size_t>(kBlock * kIV));
    for (int64_t i = 0; i < kBlock * kIV; ++i) {
      const double sv = 1.0 / (1.0 + std::exp(-g_b[static_cast<size_t>(i)]));
      act_b[static_cast<size_t>(i)] = RoundB(g_b[static_cast<size_t>(i)] * sv * u_b[static_cast<size_t>(i)]);
    }
    Mat m_b = RoundBAll(MulWT(act_b, kBlock, kIV, lw0b.down, kH));
    const Mat mcv_b = RoundBAll(Gdc(m_b, kBlock, kpm_b, lw0b.mc_b10, lw0b.mc_b11, 1));
    {
      auto dumpb = [&](const char* tag, const Mat& m) {
        std::FILE* fp = std::fopen((std::string("/tmp/w2-mine-") + tag + ".bin").c_str(), "wb");
        for (double v : m) {
          const float f = static_cast<float>(v);
          std::fwrite(&f, 4, 1, fp);
        }
        std::fclose(fp);
      };
      dumpb("attn_in", attn_in_b);
      dumpb("a", a_b);
      dumpb("o2d", o_b);
      dumpb("qh", q_b);
      dumpb("kh", k_b);
      dumpb("h2", h2_b);
      dumpb("mlp_in", mlp_in_b);
      dumpb("act", act_b);
      dumpb("m_pre", m_b);
      dumpb("mcv", mcv_b);
    }
    // per_layer[0] here is the REAL chain's layer-0 post-finish output (the
    // no-ctx arm ran ForwardBlockLogits above)
    {
      const PccStats st_real = Compare(per_layer[0], mcv_b, 0,
                                       static_cast<size_t>(kBlock * kH));
      MESSAGE("DECISIVE: real chain per_layer[0] vs fp64(mine): PCC " << st_real.pcc
              << ", rel_rms " << st_real.rel_rms);
      std::string ms;
      for (int i = 0; i < 8; ++i) ms += std::to_string(mcv_b[static_cast<size_t>(i)]) + " ";
      MESSAGE("mcv_b first8: " << ms << " | cpp first3 " << per_layer[0][0] << " "
              << per_layer[0][1] << " " << per_layer[0][2]);
      // recompute the reference through RefLayerForward HERE and three-way
      // compare: per_layer[0] vs mcv_b (above), per_layer[0] vs RefLayerForward,
      // RefLayerForward vs mcv_b. If the last is not 1.0, RefLayerForward is
      // not what its own formulas say.
      {
        Mat hh = f.block_rows;
        Mat cosx, sinx;
        RopeTables(kBlock, &cosx, &sinx);
        Mat hctx_empty;
        Mat down2 = RefLayerForward(lw0b, hctx_empty, hh, cosx, sinx, 0, 0, nullptr);
        const std::vector<float> down2f(down2.begin(), down2.end());
        const Mat mcv_bf(mcv_b.begin(), mcv_b.end());
        const PccStats s_dm = Compare(down2f, mcv_b, 0, down2.size());
        const PccStats s_pm = Compare(per_layer[0], mcv_b, 0, down2.size());
        const PccStats s_pd = Compare(per_layer[0], down2, 0, down2.size());
        std::string d2s;
        for (int i = 0; i < 8; ++i) d2s += std::to_string(down2[static_cast<size_t>(i)]) + " ";
        MESSAGE("REFLAYER down2 first8: " << d2s << "| PCC(down2, mcv_b) " << s_dm.pcc
                << " | PCC(per_layer, mcv_b) " << s_pm.pcc
                << " | PCC(per_layer, down2) " << s_pd.pcc);
        // bisect: compare per-layer STAGES of down2 against mcv_b by recomputing
        // mcv_b's intermediates... first: is the MLP act the divergence?
        size_t am2 = 0;
        for (size_t i = 1; i < down2.size(); ++i)
          if (std::fabs(down2[i]) > std::fabs(down2[am2])) am2 = i;
        MESSAGE("down2 max at " << am2 << " value " << down2[am2] << " | mcv_b max at "
                << [&]{ size_t a=0; for (size_t i=1;i<mcv_b.size();++i) if (std::fabs(mcv_b[i])>std::fabs(mcv_b[a])) a=i; return a; }()
                << " value " << *std::max_element(mcv_b.begin(), mcv_b.end(),
                    [](double x, double y) { return std::fabs(x) < std::fabs(y); }));
      }
    }
  }

  {
    const int64_t ctx_rows = no_ctx ? 0 : kCtx;
    Mat cos, sin;
    RopeTables(ctx_rows + kBlock, &cos, &sin);
    Mat hctx;
    if (!no_ctx)
      hctx = Rms(MulWT(FromF32(f.taps), kCtx, kH * kTaps, f.ref.fc, kH),
                 kCtx, kH, f.ref.hnorm);
    Mat h = f.block_rows;
    for (size_t l = 0; l < per_layer.size(); ++l) {
      Mat down;
      h = RefLayerForward(f.ref.layers[l], hctx, h, cos, sin, ctx_rows, ctx_rows, &down);
      // per_layer_out[l] is the conv-finished MLP output BEFORE the residual
      // add (qwen3_dflash.cpp:704-711 downloads `down`; the residual add is the
      // NEXT layer's input-norm)
      const PccStats st = Compare(per_layer[l], down, 0, static_cast<size_t>(kBlock * kH));
      if (l == 0) {
        // down vs my independent recomputation of the SAME formulas
        std::string ds;
        for (int i = 0; i < 8; ++i) ds += std::to_string(down[static_cast<size_t>(i)]) + " ";
        const Mat downf(down.begin(), down.end());
        MESSAGE("down first8: " << ds << "| cpp first3 " << per_layer[l][0] << " "
                << per_layer[l][1] << " " << per_layer[l][2]
                << " | down max " << *std::max_element(down.begin(), down.end(),
                    [](double a, double b) { return std::fabs(a) < std::fabs(b); }));
        // where does ref down's max live?
        size_t am = 0;
        for (size_t i = 1; i < down.size(); ++i)
          if (std::fabs(down[i]) > std::fabs(down[am])) am = i;
        MESSAGE("down max at " << am << " (row " << am / kH << " col " << am % kH
                << ") value " << down[am]);
      }
      MESSAGE("layer " << l << " down (ctx " << ctx_rows << "): PCC " << st.pcc
              << ", rel_rms " << st.rel_rms);
      if (l == 0) {
        double mc = 0, md = 0;
        for (int64_t i = 0; i < kBlock * kH; ++i) {
          mc = std::max(mc, std::fabs(down[static_cast<size_t>(i)]));
          md = std::max(md, std::fabs(static_cast<double>(per_layer[l][static_cast<size_t>(i)])));
        }
        MESSAGE("layer0 down max|ref| " << mc << " max|cpp| " << md << " cpp[0..5] "
                << per_layer[l][0] << " " << per_layer[l][1] << " " << per_layer[l][2]
                << " | ref " << down[0] << " " << down[1] << " " << down[2]);
      }
    }
  }
  double worst_pcc = 1.0;
  for (int64_t b = 0; b < kBlock; ++b) {
    const PccStats st = Compare(ours, ref, static_cast<size_t>(b * kH), static_cast<size_t>(kH));
    MESSAGE("row " << b << ": PCC " << st.pcc << ", rel_rms " << st.rel_rms);
    worst_pcc = std::min(worst_pcc, st.pcc);
  }
  CHECK_MESSAGE(worst_pcc >= kPccFloor, "CPU arm diverged: worst-row PCC " << worst_pcc);
}

TEST_CASE("qwen38 dflash2 drafter TT: the PCC gate is RED on a corrupted fc row") {
  if (!HasTT()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  const std::string dir = WeightsDir();
  if (!fs::exists(dir + "/config.json")) {
    MESSAGE("SKIPPED: pinned drafter checkpoint not found at " << dir);
    return;
  }
  Fixture f = MakeFixture();

  // Corrupt OUR side only: negate fc's output row 0 (fc is [H, H*kTaps]).
  for (int64_t j = 0; j < kH * kTaps; ++j) {
    const size_t idx = static_cast<size_t>(j * 2);
    uint16_t b;
    std::memcpy(&b, f.weights.fc.bytes.data() + idx, 2);
    b = static_cast<uint16_t>(b ^ 0x8000);  // flip the bf16 sign bit
    std::memcpy(f.weights.fc.bytes.data() + idx, &b, 2);
  }

  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  auto store = Qwen3DFlashModel::MakeDeviceKVStore(f.config, q, 256);
  auto ours = RunOurForward(f, q, f.weights, store.get());
  auto ref = RefForward(f.ref, FromF32(f.taps), f.block_rows);
  const PccStats all = Compare(ours, ref, 0, static_cast<size_t>(kBlock * kH));
  MESSAGE("corrupted-fc ALL rows: PCC " << all.pcc << ", rel_rms " << all.rel_rms);
  CHECK_MESSAGE(all.pcc < kPccFloor,
                "the corrupted-fc arm stayed above the band (" << all.pcc
                << " >= " << kPccFloor << ") — the gate cannot detect a wrong drafter");
}
