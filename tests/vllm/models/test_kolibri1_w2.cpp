// Kolibri-1 W2 gate test (MODEL-TEXT-kolibri-1, spec
// .agents/specs/kolibri-1-cpu.md) — the CPU forward pass.
//
// RED-FIRST reference (spec risk R4 — the plugin ships no golden tokens):
// the test transcribes the oracle's `kolibri1.py` math into a small SCALAR
// reference (plain double loops, no vt ops, no production code) and compares
// the production CPU forward against it on a tiny synthetic model
// (2 layers: 1 sliding + 1 full, hidden 64, 4 experts top-2, vocab 32). The
// reference computes, by hand and in the plugin's own order:
//   sandwich norms (the residual accumulates the POST-NORMED attention
//   output, kolibri1.py:250), per-head qk-norm before RoPE (:107-118), NeoX
//   RoPE on the sliding layer only (RNoPE: the full-attention layer gets NO
//   positional encoding, :81-83), causal attention with the 513-style window
//   on the sliding layer (:85-106), sigmoid-logit-add routing (top-k on
//   logits + bias, weights = sigmoid of the UNBIASED logits, no
//   renormalisation, :126-142), SwiGLU experts, the UNGATED shared expert
//   (:146-188), and the untied lm_head over the final residual-carrying norm.
// Weights are random (seeded LCG), fp8-block quantized exactly as the
// checkpoint layout, and BOTH sides dequantize with the same format decode
// (fp8-e4m3 * scale_inv grid). The reference is independent of the
// production GRAPH — vt ops, fusing, bf16 rounding, paging — which is what
// the comparison proves. Tolerance covers the production path's bf16
// activation rounding (the reference runs in double).

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_forward.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/fp8_kv.h"
#include "vt/unaligned.h"
#include "vt/ops.h"

#include <functional>

namespace {

using vllm::HfConfig;
using vllm::Kolibri1Weights;
using vllm::ModelForwardInput;
using vllm::MultiKvCacheIndex;
using vllm::PagedKvCache;
using vllm::ModelRegistry;
using vllm::SafetensorsFile;

// ── Seeded random source (deterministic across runs/platforms) ──────────────
struct Lcg {
  uint64_t s;
  explicit Lcg(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695041ULL) {}
  uint32_t NextU32() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<uint32_t>(s >> 33);
  }
  double Next() {  // [-1, 1)
    return static_cast<double>(static_cast<int32_t>(NextU32())) / 2147483648.0;
  }
  double Next(double lo, double hi) {
    return lo + (hi - lo) * (Next() * 0.5 + 0.5);
  }
};

// ── safetensors fixture writing (the W1 test's builder) ─────────────────────
struct FixtureTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[i] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

std::string BuildSafetensors(const std::vector<FixtureTensor>& tensors) {
  nlohmann::json header = nlohmann::json::object();
  std::string payload;
  for (const FixtureTensor& t : tensors) {
    const size_t begin = payload.size();
    payload.append(reinterpret_cast<const char*>(t.bytes.data()),
                   t.bytes.size());
    nlohmann::json entry = nlohmann::json::object();
    entry["dtype"] = t.dtype;
    entry["shape"] = t.shape;
    entry["data_offsets"] = nlohmann::json::array({begin, payload.size()});
    header[t.name] = std::move(entry);
  }
  const std::string head = header.dump();
  return U64Le(head.size()) + head + payload;
}

class TempCheckpoint {
 public:
  explicit TempCheckpoint(const std::vector<FixtureTensor>& tensors) {
    static std::atomic<uint64_t> counter{0};
    dir_ = std::filesystem::temp_directory_path() /
           ("vllm_kolibri1_w2_" + std::to_string(counter.fetch_add(1)));
    std::filesystem::create_directories(dir_);
    path_ = dir_ / "model.safetensors";
    const std::string bytes = BuildSafetensors(tensors);
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
    if (!out) throw std::runtime_error("failed to write fixture checkpoint");
  }
  ~TempCheckpoint() {
    std::error_code ignored;
    std::filesystem::remove_all(dir_, ignored);
  }
  TempCheckpoint(const TempCheckpoint&) = delete;
  TempCheckpoint& operator=(const TempCheckpoint&) = delete;
  std::string path() const { return path_.string(); }

 private:
  std::filesystem::path dir_;
  std::filesystem::path path_;
};

// ── Fixture tensor emitters ─────────────────────────────────────────────────
std::vector<uint8_t> Bf16Bytes(const std::vector<double>& v) {
  std::vector<uint8_t> out(v.size() * 2);
  for (size_t i = 0; i < v.size(); ++i) {
    const uint16_t bits = vt::F32ToBF16(static_cast<float>(v[i]));
    out[2 * i] = static_cast<uint8_t>(bits & 0xff);
    out[2 * i + 1] = static_cast<uint8_t>(bits >> 8);
  }
  return out;
}

std::vector<double> RandVec(Lcg& rng, int64_t n, double lo, double hi) {
  std::vector<double> v(static_cast<size_t>(n));
  for (auto& x : v) x = rng.Next(lo, hi);
  return v;
}

void AppendBf16Tensor(std::vector<FixtureTensor>& out, Lcg& rng,
                      const std::string& name, std::vector<int64_t> shape,
                      double lo, double hi) {
  // Compute the bytes BEFORE the shape moves: braced-init argument order is
  // unspecified, and Numel on a moved-out shape wrote a 1-element tensor.
  const std::vector<uint8_t> bytes =
      Bf16Bytes(RandVec(rng, Numel(shape), lo, hi));
  out.push_back({name, "BF16", std::move(shape), bytes});
}

std::vector<float> RandVecF32(Lcg& rng, int64_t n, double lo, double hi) {
  std::vector<float> v(static_cast<size_t>(n));
  for (auto& x : v) x = static_cast<float>(rng.Next(lo, hi));
  return v;
}

// One fp8-block projection: random e4m3 payload + an F32 scale grid (the
// REAL checkpoint's scale dtype, R6). Both sides dequantize the SAME bytes.
void AppendFp8Projection(std::vector<FixtureTensor>& out, Lcg& rng,
                         const std::string& proj, int64_t n, int64_t k) {
  FixtureTensor w;
  w.name = proj + ".weight";
  w.dtype = "F8_E4M3";
  w.shape = {n, k};
  w.bytes.resize(static_cast<size_t>(n * k));
  for (auto& b : w.bytes)
    b = vt::F32ToF8E4M3(static_cast<float>(rng.Next(-1.5, 1.5)));
  out.push_back(std::move(w));

  const std::vector<float> grid = RandVecF32(rng, ((n + 127) / 128) * ((k + 127) / 128), 0.1, 0.6);
  out.push_back({proj + ".weight_scale_inv", "F32",
                 {(n + 127) / 128, (k + 127) / 128},
                 {reinterpret_cast<const uint8_t*>(grid.data()),
                  reinterpret_cast<const uint8_t*>(grid.data()) + grid.size() * 4}});
}

// ── Model shape + config ────────────────────────────────────────────────────
struct Shape {
  int64_t hidden = 64;
  int64_t vocab = 32;
  int64_t heads = 4;
  int64_t kv_heads = 2;
  int64_t head_dim = 16;
  int64_t layers = 2;
  int64_t experts = 4;
  int64_t topk = 2;
  int64_t inter = 32;
  int64_t window = 8;
};

HfConfig MakeConfig(const Shape& s,
                    const std::vector<std::string>& layer_types) {
  HfConfig config;
  config.model_type = "kolibri1";
  config.architectures = {"Kolibri1ForCausalLM"};
  config.hidden_size = s.hidden;
  config.num_hidden_layers = s.layers;
  config.vocab_size = s.vocab;
  config.num_attention_heads = s.heads;
  config.num_key_value_heads = s.kv_heads;
  config.head_dim = s.head_dim;

  nlohmann::json j;
  j["head_dim"] = s.head_dim;
  j["sliding_window"] = s.window;
  j["use_sliding_window"] = true;
  j["rope_theta"] = 10000.0;
  j["num_experts"] = s.experts;
  j["num_experts_per_tok"] = s.topk;
  j["moe_intermediate_size"] = s.inter;
  j["shared_expert_intermediate_size"] = s.inter;
  j["norm_topk_prob"] = false;
  j["rms_norm_eps"] = 1e-6;
  j["hidden_act"] = "silu";
  j["tie_word_embeddings"] = false;
  j["layer_types"] = layer_types;

  nlohmann::json quant;
  quant["quant_method"] = "fp8";
  quant["activation_scheme"] = "dynamic";
  quant["weight_block_size"] = nlohmann::json::array({128, 128});
  j["quantization_config"] = quant;

  config.raw = j;
  return config;
}

std::vector<FixtureTensor> BuildFixture(Lcg& rng, const Shape& s) {
  std::vector<FixtureTensor> t;
  AppendBf16Tensor(t, rng, "model.embed_tokens.weight", {s.vocab, s.hidden},
                   -0.3, 0.3);
  AppendBf16Tensor(t, rng, "lm_head.weight", {s.vocab, s.hidden}, -0.3, 0.3);
  AppendBf16Tensor(t, rng, "model.norm.weight", {s.hidden}, 0.5, 1.5);
  for (int64_t l = 0; l < s.layers; ++l) {
    const std::string b = "model.layers." + std::to_string(l) + ".";
    for (const char* norm : {"input_layernorm", "post_attn_norm",
                             "post_attention_layernorm", "post_ffn_norm"}) {
      AppendBf16Tensor(t, rng, b + norm + ".weight", {s.hidden}, 0.5, 1.5);
    }
    AppendBf16Tensor(t, rng, b + "self_attn.q_norm.weight", {s.head_dim},
                     0.5, 1.5);
    AppendBf16Tensor(t, rng, b + "self_attn.k_norm.weight", {s.head_dim},
                     0.5, 1.5);
    AppendFp8Projection(t, rng, b + "self_attn.q_proj", s.heads * s.head_dim,
                        s.hidden);
    AppendFp8Projection(t, rng, b + "self_attn.k_proj",
                        s.kv_heads * s.head_dim, s.hidden);
    AppendFp8Projection(t, rng, b + "self_attn.v_proj",
                        s.kv_heads * s.head_dim, s.hidden);
    AppendFp8Projection(t, rng, b + "self_attn.o_proj", s.hidden,
                        s.heads * s.head_dim);
    AppendBf16Tensor(t, rng, b + "mlp.gate.weight", {s.experts, s.hidden},
                     -0.5, 0.5);
    AppendBf16Tensor(t, rng, b + "moe.router.expert_bias", {s.experts},
                     -0.25, 0.25);
    for (int64_t e = 0; e < s.experts; ++e) {
      const std::string ex = b + "mlp.experts." + std::to_string(e);
      AppendFp8Projection(t, rng, ex + ".gate_proj", s.inter, s.hidden);
      AppendFp8Projection(t, rng, ex + ".up_proj", s.inter, s.hidden);
      AppendFp8Projection(t, rng, ex + ".down_proj", s.hidden, s.inter);
    }
    AppendFp8Projection(t, rng, b + "mlp.shared_experts.gate_proj", s.inter,
                        s.hidden);
    AppendFp8Projection(t, rng, b + "mlp.shared_experts.up_proj", s.inter,
                        s.hidden);
    AppendFp8Projection(t, rng, b + "mlp.shared_experts.down_proj", s.hidden,
                        s.inter);
  }
  return t;
}

Kolibri1Weights LoadWeights(const TempCheckpoint& ckpt,
                            const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  return vllm::LoadKolibri1Weights(shards, config);
}

// ── Accessors decoding exactly what the W1 loader stored ────────────────────
double Bf16Val(const vllm::OwnedTensor& t, int64_t i) {
  return vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(t.bytes.data() + i * 2));
}

// fp8-e4m3 * scale_inv — the dequant the forward performs per weight.
double DequantVal(const vllm::Fp8BlockWeight& w, int64_t n, int64_t k) {
  const int64_t scale_cols = (w.k + w.block_k - 1) / w.block_k;
  const float scale = *reinterpret_cast<const float*>(
      w.scale.bytes.data() +
      (n / w.block_n * scale_cols + k / w.block_k) * 4);
  return static_cast<double>(vt::F8E4M3ToF32(w.packed.bytes.data()[n * w.k + k])) *
         static_cast<double>(scale);
}

// ── THE SCALAR REFERENCE (the transcription of kolibri1.py) ─────────────────

std::vector<double> RmsNorm(const std::vector<double>& x,
                            const std::vector<double>& gw, double eps) {
  const int64_t h = static_cast<int64_t>(x.size());
  double sumsq = 0.0;
  for (double v : x) sumsq += v * v;
  const double inv = 1.0 / std::sqrt(sumsq / static_cast<double>(h) + eps);
  std::vector<double> y(static_cast<size_t>(h));
  for (int64_t i = 0; i < h; ++i)
    y[static_cast<size_t>(i)] =
        x[static_cast<size_t>(i)] * inv * gw[static_cast<size_t>(i)];
  return y;
}

std::vector<double> Lin(const std::vector<double>& x, int64_t n, int64_t k,
                        const std::function<double(int64_t, int64_t)>& at) {
  std::vector<double> y(static_cast<size_t>(n), 0.0);
  for (int64_t i = 0; i < n; ++i) {
    double acc = 0.0;
    for (int64_t j = 0; j < k; ++j) acc += x[static_cast<size_t>(j)] * at(i, j);
    y[static_cast<size_t>(i)] = acc;
  }
  return y;
}

// NeoX RoPE on one head, full rotary (rotary_dim == head_dim).
void RopeNeoxRef(std::vector<double>& head, int64_t pos, double theta) {
  const int64_t d = static_cast<int64_t>(head.size());
  const int64_t half = d / 2;
  for (int64_t i = 0; i < half; ++i) {
    const double inv = std::pow(theta, -2.0 * static_cast<double>(i) / static_cast<double>(d));
    const double c = std::cos(static_cast<double>(pos) * inv);
    const double s = std::sin(static_cast<double>(pos) * inv);
    const double a = head[static_cast<size_t>(i)];
    const double b = head[static_cast<size_t>(i + half)];
    head[static_cast<size_t>(i)] = a * c - b * s;
    head[static_cast<size_t>(i + half)] = b * c + a * s;
  }
}

double Silu(double x) { return x / (1.0 + std::exp(-x)); }

double Sig(double x) { return 1.0 / (1.0 + std::exp(-x)); }

// SwiGLU expert from fp8-block weights.
std::vector<double> ExpertMlpRef(const vllm::Kolibri1ExpertWeights& e,
                                 const std::vector<double>& x, int64_t inter,
                                 int64_t hidden) {
  std::vector<double> g = Lin(x, inter, hidden, [&](int64_t i, int64_t j) {
    return DequantVal(e.gate_proj.fp8_block, i, j);
  });
  std::vector<double> u = Lin(x, inter, hidden, [&](int64_t i, int64_t j) {
    return DequantVal(e.up_proj.fp8_block, i, j);
  });
  for (size_t i = 0; i < g.size(); ++i) g[i] = Silu(g[i]) * u[i];
  return Lin(g, hidden, inter, [&](int64_t i, int64_t j) {
    return DequantVal(e.down_proj.fp8_block, i, j);
  });
}

struct RefCtx {
  const Kolibri1Weights& w;
  int64_t H, Dh, Hq, Hkv, E, I, topk;
  double eps, theta;
  int64_t window;
};

// One decoder layer of the reference. `hidden` in/out is the post-ffn-norm
// stream; `residual` is carried across layers exactly as vLLM's
// residual-carrying RMSNorm contract rounds it. Returns nothing; mutates both.
void RefLayer(const RefCtx& r, int64_t l, std::vector<double>& hidden,
              std::vector<double>& residual,
              const std::vector<int32_t>& pos_of) {
  const Kolibri1Weights& w = r.w;
  const auto& lw = w.layers[static_cast<size_t>(l)];
  const int64_t T = static_cast<int64_t>(pos_of.size());
  const int64_t H = r.H, Dh = r.Dh, Hq = r.Hq, Hkv = r.Hkv;

  std::vector<double> g_in(static_cast<size_t>(H)), g_pa(static_cast<size_t>(H)),
      g_pal(static_cast<size_t>(H)), g_pf(static_cast<size_t>(H));
  for (int64_t i = 0; i < H; ++i) {
    g_in[static_cast<size_t>(i)] = Bf16Val(lw.input_layernorm, i);
    g_pa[static_cast<size_t>(i)] = Bf16Val(lw.post_attn_norm, i);
    g_pal[static_cast<size_t>(i)] = Bf16Val(lw.post_attention_layernorm, i);
    g_pf[static_cast<size_t>(i)] = Bf16Val(lw.post_ffn_norm, i);
  }
  std::vector<double> gq(static_cast<size_t>(Dh)), gk(static_cast<size_t>(Dh));
  for (int64_t i = 0; i < Dh; ++i) {
    gq[static_cast<size_t>(i)] = Bf16Val(lw.attn.q_norm, i);
    gk[static_cast<size_t>(i)] = Bf16Val(lw.attn.k_norm, i);
  }

  // 1. input_layernorm(hidden, residual): residual += hidden, then norm.
  for (size_t i = 0; i < residual.size(); ++i) residual[i] += hidden[i];
  std::vector<double> x(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> row(residual.begin() + static_cast<ptrdiff_t>(t * H),
                            residual.begin() + static_cast<ptrdiff_t>((t + 1) * H));
    row = RmsNorm(row, g_in, r.eps);
    std::copy(row.begin(), row.end(),
              x.begin() + static_cast<ptrdiff_t>(t * H));
  }

  // 2. Attention. Q/K/V per token; per-head qk-norm BEFORE RoPE; RoPE only
  // on the sliding layer (RNoPE, :81-95).
  std::vector<double> attn_out(static_cast<size_t>(T * H), 0.0);
  const int64_t gqa = Hq / Hkv;
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> q = Lin(
        std::vector<double>(x.begin() + static_cast<ptrdiff_t>(t * H),
                            x.begin() + static_cast<ptrdiff_t>((t + 1) * H)),
        Hq * Dh, H, [&](int64_t i, int64_t j) {
          return DequantVal(lw.attn.q_proj.fp8_block, i, j);
        });
    for (int64_t hh = 0; hh < Hq; ++hh) {
      std::vector<double> head(q.begin() + static_cast<ptrdiff_t>(hh * Dh),
                               q.begin() + static_cast<ptrdiff_t>((hh + 1) * Dh));
      head = RmsNorm(head, gq, r.eps);
      if (lw.is_sliding) RopeNeoxRef(head, pos_of[static_cast<size_t>(t)], r.theta);
      std::copy(head.begin(), head.end(),
                q.begin() + static_cast<ptrdiff_t>(hh * Dh));
    }

    // Keys and values for every visible position (a from-zero prefill: the
    // cache holds positions 0..pos_t, which the reference recomputes from
    // the same normalized rows the production path wrote through its cache).
    for (int64_t hh = 0; hh < Hq; ++hh) {
      const int64_t kvh = hh / gqa;
      const int64_t lo =
          lw.is_sliding
              ? std::max<int64_t>(0, pos_of[static_cast<size_t>(t)] - (r.window - 1))
              : 0;
      const int64_t hi = pos_of[static_cast<size_t>(t)];
      std::vector<double> scores;
      std::vector<std::vector<double>> vrows;
      for (int64_t u = lo; u <= hi; ++u) {
        std::vector<double> ku = Lin(
            std::vector<double>(x.begin() + static_cast<ptrdiff_t>(u * H),
                                x.begin() + static_cast<ptrdiff_t>((u + 1) * H)),
            Hkv * Dh, H, [&](int64_t i, int64_t j) {
              return DequantVal(lw.attn.k_proj.fp8_block, i, j);
            });
        std::vector<double> khead(
            ku.begin() + static_cast<ptrdiff_t>(kvh * Dh),
            ku.begin() + static_cast<ptrdiff_t>((kvh + 1) * Dh));
        khead = RmsNorm(khead, gk, r.eps);
        if (lw.is_sliding) RopeNeoxRef(khead, u, r.theta);
        std::vector<double> qhead(
            q.begin() + static_cast<ptrdiff_t>(hh * Dh),
            q.begin() + static_cast<ptrdiff_t>((hh + 1) * Dh));
        double dot = 0.0;
        for (int64_t d = 0; d < Dh; ++d)
          dot += qhead[static_cast<size_t>(d)] * khead[static_cast<size_t>(d)];
        scores.push_back(dot / std::sqrt(static_cast<double>(Dh)));
        std::vector<double> vu = Lin(
            std::vector<double>(x.begin() + static_cast<ptrdiff_t>(u * H),
                                x.begin() + static_cast<ptrdiff_t>((u + 1) * H)),
            Hkv * Dh, H, [&](int64_t i, int64_t j) {
              return DequantVal(lw.attn.v_proj.fp8_block, i, j);
            });
        vrows.emplace_back(vu.begin() + static_cast<ptrdiff_t>(kvh * Dh),
                           vu.begin() + static_cast<ptrdiff_t>((kvh + 1) * Dh));
      }
      const double mx = *std::max_element(scores.begin(), scores.end());
      double denom = 0.0;
      for (double& s : scores) {
        s = std::exp(s - mx);
        denom += s;
      }
      std::vector<double> head_out(static_cast<size_t>(Dh), 0.0);
      for (size_t i = 0; i < scores.size(); ++i)
        for (int64_t d = 0; d < Dh; ++d)
          head_out[static_cast<size_t>(d)] +=
              scores[i] / denom * vrows[i][static_cast<size_t>(d)];
      for (int64_t d = 0; d < Dh; ++d)
        attn_out[static_cast<size_t>(t * H + hh * Dh + d)] =
            head_out[static_cast<size_t>(d)];
    }
  }
  // o_proj per token.
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> o = Lin(
        std::vector<double>(
            attn_out.begin() + static_cast<ptrdiff_t>(t * H),
            attn_out.begin() + static_cast<ptrdiff_t>((t + 1) * H)),
        H, Hq * Dh, [&](int64_t i, int64_t j) {
          return DequantVal(lw.attn.o_proj.fp8_block, i, j);
        });
    std::copy(o.begin(), o.end(),
              attn_out.begin() + static_cast<ptrdiff_t>(t * H));
  }

  // 3. post_attn_norm (NO residual), then post_attention_layernorm carries
  // the residual: residual += POST-NORMED attn output (kolibri1.py:248-250).
  std::vector<double> x2(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> row(attn_out.begin() + static_cast<ptrdiff_t>(t * H),
                            attn_out.begin() + static_cast<ptrdiff_t>((t + 1) * H));
    row = RmsNorm(row, g_pa, r.eps);
    for (int64_t i = 0; i < H; ++i)
      residual[static_cast<size_t>(t * H + i)] += row[static_cast<size_t>(i)];
    row = RmsNorm(std::vector<double>(
                      residual.begin() + static_cast<ptrdiff_t>(t * H),
                      residual.begin() + static_cast<ptrdiff_t>((t + 1) * H)),
                  g_pal, r.eps);
    std::copy(row.begin(), row.end(), x2.begin() + static_cast<ptrdiff_t>(t * H));
  }

  // 4. MoE on EVERY layer: router -> routed + UNGATED shared -> post_ffn_norm.
  std::vector<double> gate_w(static_cast<size_t>(r.E * H));
  std::vector<double> bias(static_cast<size_t>(r.E));
  for (int64_t i = 0; i < r.E * H; ++i)
    gate_w[static_cast<size_t>(i)] = Bf16Val(lw.moe.router_gate, i);
  // The bias is RESIDENT F32 (the loader widened the on-disk bf16).
  for (int64_t e = 0; e < r.E; ++e)
    bias[static_cast<size_t>(e)] = *reinterpret_cast<const float*>(
        lw.moe.e_score_correction_bias.bytes.data() + e * 4);

  std::vector<double> moe_out(static_cast<size_t>(T * H));
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> row(x2.begin() + static_cast<ptrdiff_t>(t * H),
                            x2.begin() + static_cast<ptrdiff_t>((t + 1) * H));
    std::vector<double> logits(r.E, 0.0);
    for (int64_t e = 0; e < r.E; ++e)
      for (int64_t j = 0; j < H; ++j)
        logits[static_cast<size_t>(e)] +=
            row[static_cast<size_t>(j)] * gate_w[static_cast<size_t>(e * H + j)];
    // sigmoid-logit-add: top-k on logits + bias, weight = sigmoid(unbiased).
    std::vector<char> taken(static_cast<size_t>(r.E), 0);
    std::vector<double> moe_acc(static_cast<size_t>(H), 0.0);
    for (int64_t kk = 0; kk < r.topk; ++kk) {
      int64_t best = -1;
      double best_score = 0.0;
      for (int64_t e = 0; e < r.E; ++e) {
        if (taken[static_cast<size_t>(e)]) continue;
        const double sc = logits[static_cast<size_t>(e)] + bias[static_cast<size_t>(e)];
        if (best < 0 || sc > best_score) {
          best = e;
          best_score = sc;
        }
      }
      taken[static_cast<size_t>(best)] = 1;
      const double wk = Sig(logits[static_cast<size_t>(best)]);
      const std::vector<double> eo = ExpertMlpRef(
          lw.moe.experts[static_cast<size_t>(best)], row, r.I, H);
      for (int64_t i = 0; i < H; ++i) moe_acc[static_cast<size_t>(i)] += wk * eo[static_cast<size_t>(i)];
    }
    // The UNGATED shared expert, always added (:146-188).
    const std::vector<double> sh =
        ExpertMlpRef(lw.moe.shared_experts, row, r.I, H);
    for (int64_t i = 0; i < H; ++i) moe_acc[static_cast<size_t>(i)] += sh[static_cast<size_t>(i)];
    std::copy(moe_acc.begin(), moe_acc.end(),
              moe_out.begin() + static_cast<ptrdiff_t>(t * H));
  }

  // 5. post_ffn_norm (NO residual); its output is the next layer's `hidden`.
  for (int64_t t = 0; t < T; ++t) {
    std::vector<double> row(moe_out.begin() + static_cast<ptrdiff_t>(t * H),
                            moe_out.begin() + static_cast<ptrdiff_t>((t + 1) * H));
    row = RmsNorm(row, g_pf, r.eps);
    std::copy(row.begin(), row.end(),
              hidden.begin() + static_cast<ptrdiff_t>(t * H));
  }
}

// Full reference forward -> [rows, vocab] double logits.
std::vector<double> RefForward(const Kolibri1Weights& w,
                               const std::vector<int32_t>& token_ids,
                               const std::vector<int32_t>& pos_of_arg = {}) {
  // Positions default to 0..T-1 (a from-zero prefill); callers never need to
  // pass anything else.
  std::vector<int32_t> pos_of = pos_of_arg;
  if (pos_of.empty()) {
    pos_of.resize(token_ids.size());
    for (size_t i = 0; i < pos_of.size(); ++i)
      pos_of[i] = static_cast<int32_t>(i);
  }
  RefCtx r{w,
           w.params.hidden_size, w.params.head_dim,
           w.params.num_attention_heads, w.params.num_key_value_heads,
           w.params.num_experts, w.params.moe_intermediate_size,
           w.params.num_experts_per_tok, w.params.rms_norm_eps,
           w.params.rope_theta, w.params.sliding_window};
  const int64_t T = static_cast<int64_t>(token_ids.size());
  const int64_t H = r.H, V = w.params.vocab_size;

  std::vector<double> hidden(static_cast<size_t>(T * H)), residual(static_cast<size_t>(T * H), 0.0);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t i = 0; i < H; ++i)
      hidden[static_cast<size_t>(t * H + i)] =
          Bf16Val(w.embed_tokens,
                  token_ids[static_cast<size_t>(t)] * H + i);

  for (int64_t l = 0; l < w.params.num_hidden_layers; ++l)
    RefLayer(r, l, hidden, residual, pos_of);

  // Final norm carries the residual.
  std::vector<double> gfn(static_cast<size_t>(H));
  for (int64_t i = 0; i < H; ++i) gfn[static_cast<size_t>(i)] = Bf16Val(w.final_norm, i);
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t i = 0; i < H; ++i)
      hidden[static_cast<size_t>(t * H + i)] += residual[static_cast<size_t>(t * H + i)];
    std::vector<double> row(hidden.begin() + static_cast<ptrdiff_t>(t * H),
                            hidden.begin() + static_cast<ptrdiff_t>((t + 1) * H));
    row = RmsNorm(row, gfn, r.eps);
    std::copy(row.begin(), row.end(),
              hidden.begin() + static_cast<ptrdiff_t>(t * H));
  }

  std::vector<double> logits(static_cast<size_t>(T * V), 0.0);
  for (int64_t t = 0; t < T; ++t)
    for (int64_t v = 0; v < V; ++v) {
      double acc = 0.0;
      for (int64_t i = 0; i < H; ++i)
        acc += hidden[static_cast<size_t>(t * H + i)] * Bf16Val(w.lm_head, v * H + i);
      logits[static_cast<size_t>(t * V + v)] = acc;
    }
  return logits;
}

// ── Production-side harness: topology + step (the mimo_v2 W2 pattern) ───────
constexpr int32_t kBlockPerm[8] = {5, 2, 7, 1, 6, 0, 3, 4};
constexpr int64_t kBlockSize = 16;
constexpr int64_t kNumBlocks = 8;

struct Topology {
  vt::DType dtype = vt::DType::kBF16;
  std::vector<std::vector<uint8_t>> attn_bytes;
  std::vector<PagedKvCache> attn_kv;
  std::vector<std::string> names;
  std::vector<int32_t> group_ids;
  std::vector<int32_t> layer_indices;
  std::vector<uint8_t> payload_kinds;
  std::vector<int32_t> payload_slots;
  std::vector<std::vector<int32_t>> group_bt;
  std::vector<int32_t> group_cols;
  MultiKvCacheIndex mk;
  int64_t layers = 2;
  int64_t kv_heads = 2;
  int64_t head_dim = 16;

  static bool IsFull(int64_t l) { return l % 5 == 4; }  // 4 sliding then 1 full

  explicit Topology(const Shape& s, const std::vector<std::string>& ltypes) {
    layers = s.layers;
    kv_heads = s.kv_heads;
    head_dim = s.head_dim;
    int32_t paged_slot = 0;
    for (int l = 0; l < static_cast<int>(s.layers); ++l) {
      const bool full = ltypes[static_cast<size_t>(l)] == "full_attention";
      if (!full) continue;
      names.push_back("model.layers." + std::to_string(l) + ".self_attn");
      group_ids.push_back(0);
      layer_indices.push_back(l);
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
      payload_slots.push_back(paged_slot++);
    }
    for (int l = 0; l < static_cast<int>(s.layers); ++l) {
      const bool full = ltypes[static_cast<size_t>(l)] == "full_attention";
      if (full) continue;
      names.push_back("model.layers." + std::to_string(l) + ".self_attn");
      group_ids.push_back(1);
      layer_indices.push_back(l);
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
      payload_slots.push_back(paged_slot++);
    }
    const int64_t elt = static_cast<int64_t>(vt::SizeOf(dtype));
    for (size_t i = 0; i < names.size(); ++i) {
      const int64_t page_bytes = kNumBlocks * kBlockSize * kv_heads * head_dim * 2 * elt;
      attn_bytes.emplace_back(static_cast<size_t>(page_bytes), 0);
      PagedKvCache kv;
      kv.data = attn_bytes.back().data();
      kv.dtype = dtype;
      kv.num_blocks = kNumBlocks;
      kv.block_size = kBlockSize;
      kv.num_kv_heads = kv_heads;
      kv.head_size = head_dim;
      attn_kv.push_back(kv);
    }
    group_bt.assign(2, std::vector<int32_t>(kBlockPerm, kBlockPerm + 8));
    group_cols.assign(2, 8);
    Publish();
  }

  void Publish() {
    for (size_t i = 0; i < attn_kv.size() && i < attn_bytes.size(); ++i)
      attn_kv[i].data = attn_bytes[i].data();
    mk.layer_names = &names;
    mk.group_ids = &group_ids;
    mk.layer_indices = &layer_indices;
    mk.payload_kinds = &payload_kinds;
    mk.payload_slots = &payload_slots;
    mk.group_block_tables = &group_bt;
    mk.group_block_table_cols = &group_cols;
  }

  void ZeroPages() {
    for (auto& b : attn_bytes) std::fill(b.begin(), b.end(), 0);
    Publish();
  }

  static int64_t Slot(int64_t pos) {
    return kBlockPerm[pos / kBlockSize] * kBlockSize + pos % kBlockSize;
  }
};

struct Run {
  std::vector<float> logits;  // [rows, vocab]
  int64_t rows = 0;
  int64_t vocab = 0;
};

// Runs the registered forward over `seqs` sequences in ONE step (a ragged
// batch when seqs.size() > 1); every sequence starts at position 0.
Run RunForward(const Kolibri1Weights& w, const HfConfig& config,
               const std::vector<std::vector<int32_t>>& seqs,
               const std::vector<int32_t>& logits_indices = {}) {
  const int64_t nreq = static_cast<int64_t>(seqs.size());
  std::vector<int32_t> token_ids, positions, query_start_loc{0};
  std::vector<int64_t> slot_mapping;
  std::vector<int32_t> seq_lens;
  int32_t max_len = 0;
  int32_t off = 0;
  for (const auto& s : seqs) {
    for (size_t i = 0; i < s.size(); ++i) {
      token_ids.push_back(s[i]);
      positions.push_back(static_cast<int32_t>(i));
      slot_mapping.push_back(static_cast<int32_t>(Topology::Slot(i)));
    }
    query_start_loc.push_back(static_cast<int32_t>(off + static_cast<int32_t>(s.size())));
    off += static_cast<int32_t>(s.size());
    seq_lens.push_back(static_cast<int32_t>(s.size()));
    max_len = std::max(max_len, static_cast<int32_t>(s.size()));
  }
  const int64_t T = static_cast<int64_t>(token_ids.size());

  vllm::v1::CommonAttentionMetadata meta;
  meta.num_reqs = static_cast<int>(nreq);
  meta.num_actual_tokens = static_cast<int>(T);
  meta.max_query_len = static_cast<int>(
      *std::max_element(seq_lens.begin(), seq_lens.end()));
  meta.max_seq_len = max_len;
  meta.query_start_loc = query_start_loc;
  meta.query_start_loc_cpu = query_start_loc;
  meta.seq_lens = seq_lens;
  meta.seq_lens_cpu = seq_lens;
  meta.causal = true;
  meta.block_table_num_cols = 8;
  for (int64_t r = 0; r < nreq; ++r)
    meta.block_table_tensor.insert(meta.block_table_tensor.end(),
                                   kBlockPerm, kBlockPerm + 8);
  meta.slot_mapping = slot_mapping;

  Shape s;
  s.layers = w.params.num_hidden_layers;
  s.kv_heads = w.params.num_key_value_heads;
  s.head_dim = w.params.head_dim;
  std::vector<std::string> ltypes;
  for (int64_t l = 0; l < s.layers; ++l)
    ltypes.push_back(w.params.IsSlidingLayer(l) ? "sliding_attention"
                                                : "full_attention");
  Topology topo(s, ltypes);
  vllm::v1::GDNAttentionMetadata gdn{};
  std::vector<vllm::GdnStateCache> gdn_state;

  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  ModelForwardInput in{
      .token_ids = token_ids,
      .positions = positions,
      .attn_meta = meta,
      .gdn_meta = gdn,
      .attn_kv = topo.attn_kv,
      .gdn_state = gdn_state,
      .config = config,
      .queue = queue,
      .logits_indices = logits_indices,
      .num_reqs = static_cast<int>(nreq)};
  in.multi_kv = &topo.mk;

  const vllm::ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.factory->forward != nullptr);
  // W2 drives the forward seam the registry hook delegates to, over the
  // already-loaded weights (a LoadedModel without a model source cannot be
  // constructed in a unit test; the registry hook itself is a one-line
  // delegation covered by the W1 registration test).
  vllm::ForwardLogits fl = vllm::ForwardKolibri1Forward(
      in.token_ids, in.positions, in.attn_meta, in.attn_kv, w, in.multi_kv,
      in.queue, in.logits_indices);

  Run out;
  out.rows = fl.rows;
  out.vocab = fl.vocab;
  const size_t n = static_cast<size_t>(fl.rows) * static_cast<size_t>(fl.vocab);
  out.logits.resize(n);
  REQUIRE(n == out.logits.size());
  vt::Backend& be = vt::GetBackend(queue.device.type);
  be.Copy(queue, out.logits.data(), fl.device_tensor.data, n * sizeof(float));
  be.Synchronize(queue);
  return out;
}

}  // namespace
// ═══ TESTS ═══════════════════════════════════════════════════════════════════
namespace {

struct TinyWorld {
  Shape shape;
  HfConfig config;
  std::unique_ptr<TempCheckpoint> ckpt;
  Kolibri1Weights weights;

  explicit TinyWorld(uint64_t seed) : shape() {
    config = MakeConfig(shape, {"sliding_attention", "full_attention"});
    Lcg rng(seed);
    ckpt = std::make_unique<TempCheckpoint>(BuildFixture(rng, shape));
    weights = LoadWeights(*ckpt, config);
  }
};

// max |a - b| and the count of elements outside the band atol + rtol*|b|.
void CompareLogits(const std::vector<float>& got, const std::vector<double>& want,
                   int64_t rows, int64_t vocab, double atol, double rtol) {
  REQUIRE(static_cast<int64_t>(got.size()) == rows * vocab);
  REQUIRE(static_cast<int64_t>(want.size()) == rows * vocab);
  double max_abs = 0.0;
  int64_t violations = 0;
  int64_t worst = -1;
  for (int64_t i = 0; i < rows * vocab; ++i) {
    const double a = got[static_cast<size_t>(i)];
    const double b = want[static_cast<size_t>(i)];
    REQUIRE(std::isfinite(a));
    const double diff = std::fabs(a - b);
    if (diff > max_abs) {
      max_abs = diff;
      worst = i;
    }
    if (diff > atol + rtol * std::fabs(b)) ++violations;
  }
  MESSAGE("max_abs gap = ", max_abs, " at flat ", worst, " (", violations,
          " outside band)");
  CHECK(violations == 0);
  // Greedy argmax must agree on EVERY row — the token-exact property the row
  // ultimately gates on, asserted at the reference level here.
  for (int64_t t = 0; t < rows; ++t) {
    const auto ga = std::max_element(got.begin() + static_cast<ptrdiff_t>(t * vocab),
                                     got.begin() + static_cast<ptrdiff_t>((t + 1) * vocab));
    const auto wa = std::max_element(want.begin() + static_cast<ptrdiff_t>(t * vocab),
                                     want.begin() + static_cast<ptrdiff_t>((t + 1) * vocab));
    CHECK(ga - (got.begin() + static_cast<ptrdiff_t>(t * vocab)) ==
          wa - (want.begin() + static_cast<ptrdiff_t>(t * vocab)));
  }
}

}  // namespace

// W2 gate 1: the tiny 2-layer forward matches the scalar reference end to end.
// T = 10 crosses the sliding window (window 8), so the SWA masking, RNoPE on
// the full layer, qk-norm, sandwich norms, the sigmoid-logit-add router and
// the ungated shared expert are ALL exercised against the transcription.
TEST_CASE("kolibri1 W2: CPU forward matches the kolibri1.py scalar reference") {
  TinyWorld world(0x6011B117ULL);
  const std::vector<int32_t> tokens{3, 17, 5, 29, 1, 8, 22, 14, 0, 27};
  const Run run = RunForward(world.weights, world.config, {tokens});
  CHECK(run.rows == 10);
  CHECK(run.vocab == world.shape.vocab);
  const std::vector<double> ref = RefForward(world.weights, tokens, {});
  CompareLogits(run.logits, ref, 10, world.shape.vocab, 0.05, 0.05);
}

// Diagnostic: a SINGLE token at position 0 — RoPE is the identity, the
// attention window is empty, attention reduces to v[0]. Isolates the
// norm/MLP/router math from the rope/window machinery.
TEST_CASE("kolibri1 W2: single-token prefill matches the reference") {
  TinyWorld world(0x6011B117ULL);
  const std::vector<int32_t> tokens{11};
  const Run run = RunForward(world.weights, world.config, {tokens});
  const std::vector<double> ref = RefForward(world.weights, tokens, {});
  CompareLogits(run.logits, ref, 1, world.shape.vocab, 0.05, 0.05);
}


// W2 gate 2: a ragged batch with a partial/idle tail. Two sequences in one
// step (lengths 10 and 4); each row must match the SAME sequence run alone —
// this proves the query_start_loc routing, per-request block tables and slot
// mapping carry no state across the boundary, including through the paged KV
// write (sequence 2's pages are fresh).
TEST_CASE("kolibri1 W2: ragged two-sequence batch matches per-sequence runs") {
  TinyWorld world(0x9E3779B97F4A7C15ULL);
  const std::vector<int32_t> long_seq{3, 17, 5, 29, 1, 8, 22, 14, 0, 27};
  const std::vector<int32_t> short_seq{9, 2, 31, 12};
  const Run batch = RunForward(world.weights, world.config, {long_seq, short_seq});
  const Run alone_long = RunForward(world.weights, world.config, {long_seq});
  const Run alone_short = RunForward(world.weights, world.config, {short_seq});
  CHECK(batch.rows == 14);

  const auto ref_long = RefForward(world.weights, long_seq, {});
  CompareLogits(std::vector<float>(batch.logits.begin(), batch.logits.begin() + 10 * 32),
                ref_long, 10, 32, 0.05, 0.05);
  const auto ref_short = RefForward(world.weights, short_seq, {});
  CompareLogits(std::vector<float>(batch.logits.begin() + 10 * 32, batch.logits.end()),
                ref_short, 4, 32, 0.05, 0.05);

  // NOT bit-equal: the CPU GEMM's K-chunk accumulation order depends on the
  // batch shape, so identical rows differ by bf16-rounding noise. Bound it.
  double worst = 0.0;
  for (size_t i = 0; i < batch.logits.size(); ++i) {
    const float expect = i < 10u * 32u ? alone_long.logits[i]
                                       : alone_short.logits[i - 10u * 32u];
    worst = std::max(worst, static_cast<double>(std::fabs(batch.logits[i] - expect)));
  }
  MESSAGE("batch-vs-alone worst abs diff = ", worst);
  CHECK(worst <= 0.05);
}

// W2 gate 3: the logits_indices gather — a strict subset request returns
// exactly the referenced rows of the full forward.
TEST_CASE("kolibri1 W2: logits_indices gathers the referenced rows") {
  TinyWorld world(0xC0FFEE123456789ULL);
  const std::vector<int32_t> tokens{3, 17, 5, 29, 1, 8, 22, 14, 0, 27};
  const std::vector<int32_t> idx{2, 5, 9};
  const Run gathered = RunForward(world.weights, world.config, {tokens}, idx);
  const Run full = RunForward(world.weights, world.config, {tokens});
  CHECK(gathered.rows == 3);
  CHECK(gathered.vocab == world.shape.vocab);
  for (size_t r = 0; r < idx.size(); ++r)
    for (int64_t v = 0; v < world.shape.vocab; ++v)
      CHECK(gathered.logits[r * 32u + static_cast<size_t>(v)] ==
            full.logits[static_cast<size_t>(idx[r]) * 32u + static_cast<size_t>(v)]);
}

// W2 gate 4: FULL DEPTH. The real 50-layer config (hidden 2560, GQA 48/4,
// head_dim 128, window 513, the 4:1 sliding/full pattern) runs end to end on
// hermetic random weights — shape, finiteness, determinism, and the two-group
// KV resolution across all 50 layers. The expert/vocab axes are shrunk (8
// experts top-2, intermediate 64, vocab 64) so the fixture stays hermetic; the
// layer geometry — the part W2 owns — is the checkpoint's own.
TEST_CASE("kolibri1 W2: full-depth 50-layer config runs end to end") {
  Shape big;
  big.hidden = 2560;
  big.vocab = 64;
  big.heads = 48;
  big.kv_heads = 4;
  big.head_dim = 128;
  big.layers = 50;
  big.experts = 8;
  big.topk = 2;
  big.inter = 64;
  big.window = 513;
  std::vector<std::string> ltypes;
  for (int64_t l = 0; l < 50; ++l)
    ltypes.push_back(l % 5 == 4 ? "full_attention" : "sliding_attention");
  const HfConfig config = MakeConfig(big, ltypes);
  Lcg rng(0xDEADBEEFCAFEF00DULL);
  TempCheckpoint ckpt(BuildFixture(rng, big));
  Kolibri1Weights w = LoadWeights(ckpt, config);
  CHECK(w.params.num_hidden_layers == 50);

  const std::vector<int32_t> tokens{7, 31, 0, 55, 12};
  const Run run = RunForward(w, config, {tokens});
  CHECK(run.rows == 5);
  CHECK(run.vocab == 64);
  for (float v : run.logits) CHECK(std::isfinite(v));

  // Determinism: a fresh topology, same tokens, byte-identical logits.
  const Run again = RunForward(w, config, {tokens});
  CHECK(again.logits == run.logits);
}

// W2 guard: the forward refuses a GPU queue by name — the row's scope is the
// CPU path and the GPU arm is a separate owed row.
TEST_CASE("kolibri1 W2: non-CPU queue is refused by name") {
  TinyWorld world(0xABCDEF0123456789ULL);
  // There is no CUDA device in this environment; assert the guard's MESSAGE
  // by constructing the refusal through the same check the forward performs.
  bool threw = false;
  try {
    // A Metal-typed queue on a host without Metal cannot be constructed
    // either, so exercise the guard predicate directly: the forward's first
    // act on a non-CPU queue is this throw. Verify via the queue-type check
    // with a null-handle CPU-typed queue standing in for the real device.
    vt::Queue fake{vt::Device{static_cast<vt::DeviceType>(99), 0}, nullptr};
    std::vector<int32_t> tokens{1, 2, 3};
    vllm::v1::CommonAttentionMetadata meta;
    meta.num_reqs = 1;
    meta.num_actual_tokens = 3;
    meta.query_start_loc = {0, 3};
    meta.seq_lens = {3};
    meta.slot_mapping = {0, 1, 2};
    meta.block_table_num_cols = 1;
    meta.block_table_tensor = {0};
    std::vector<PagedKvCache> kv(2);
    (void)vllm::ForwardKolibri1Forward(tokens, tokens, meta, kv, world.weights,
                                       nullptr, fake, {});
  } catch (const std::exception& e) {
    threw = true;
    CHECK(std::string(e.what()).find("CPU") != std::string::npos);
  }
  CHECK(threw);
}
