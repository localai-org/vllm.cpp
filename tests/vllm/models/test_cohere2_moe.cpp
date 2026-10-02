// Cohere2MoeForCausalLM (North) gates. Spec: .agents/specs/cohere2-moe.md.
//
//  (a) CONFIG: the released North-Mini-Code-1.0 config.json (committed verbatim)
//      parses to the real geometry and to every per-layer switch; the refusals
//      fire by name; absent keys take the Cohere2MoeConfig class defaults.
//  (b) FORWARD: three tiny checkpoints, written to disk and read back through
//      the production loader, against a torch transcription of the pinned
//      cohere2_moe.py (scripts/cohere2-moe-ref.py). The f32 arm is the tight
//      gate on the arithmetic; the bf16 arm is the production dtype on the bf16
//      envelope. The configs switch every mechanism of the spec on and off.
//  (c) REAL TENSORS (env VLLM_COHERE2_MOE_REAL_DIR): layers 0 (dense prefix,
//      forced RoPE), 1 (sliding MoE) and 4 (full NoPE MoE) plus the final norm of
//      the released checkpoint, against the same transcription.
//  (e) STRUCTURAL (env VLLM_COHERE2_MOE_INDEX): every tensor of the released
//      index is enumerated, and nothing enumerated is missing.
// The runner gate (d) is test_cohere2_moe_registry.cpp.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "cohere2_moe_goldens.inc"
#include "cohere2_moe_tiny_fixture.h"
#include "support/max_abs_diff.h"
#include "vllm/model_executor/layers/attention/attention.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/cohere2_moe.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::Cohere2MoeModel;
using vllm::Cohere2MoeParams;
using vllm::Cohere2MoeWeights;
using vllm::HfConfig;
using vllm::PagedKvCache;
using vllm::SafetensorsFile;
using vllm::v1::CommonAttentionMetadata;
using vt::DType;

namespace {

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

HfConfig ReleasedConfig() {
  return vllm::LoadHfConfig(std::string(COHERE2_MOE_FIXTURE_DIR) + "/config.json");
}

HfConfig WithRaw(const nlohmann::json& raw) {
  return vllm::ParseHfConfig(raw, "cohere2_moe test config");
}

nlohmann::json ReleasedRaw() {
  std::ifstream in(std::string(COHERE2_MOE_FIXTURE_DIR) + "/config.json");
  return nlohmann::json::parse(in);
}

// A tiny checkpoint on disk, loaded back through the production loader.
struct Loaded {
  HfConfig cfg;
  std::unique_ptr<c2m_tiny::TempFile> file;
  std::vector<SafetensorsFile> shards;
  Cohere2MoeWeights w;
  explicit Loaded(const char* json) : cfg(c2m_tiny::ConfigFromJson(json)) {
    const Cohere2MoeParams p = vllm::ParseCohere2MoeParams(cfg);
    file = std::make_unique<c2m_tiny::TempFile>(c2m_tiny::BuildSt(c2m_tiny::SynthTensors(p)),
                                                ".safetensors");
    shards.push_back(SafetensorsFile::Open(file->path()));
    w = vllm::LoadCohere2MoeWeights(shards, cfg);
  }
  void UseF32() {
    w.compute_dtype = DType::kF32;
    w.rope_cos_sin = vllm::BuildCohere2MoeRopeCache(w.params, w.params.max_position, DType::kF32);
  }
};

struct Cache {
  std::vector<std::vector<float>> buf;
  std::vector<PagedKvCache> kv;
  static constexpr int64_t kBlock = 4;
  static constexpr int64_t kBlocks = 8;
  explicit Cache(const Cohere2MoeParams& p) {
    for (int64_t l = 0; l < p.num_layers; ++l)
      buf.emplace_back(static_cast<size_t>(kBlocks * 2 * kBlock * p.num_kv_heads * p.head_dim), 0.0F);
    for (auto& b : buf) {
      PagedKvCache c;
      c.data = b.data();
      c.dtype = DType::kF32;
      c.num_blocks = kBlocks;
      c.block_size = kBlock;
      c.num_kv_heads = p.num_kv_heads;
      c.head_size = p.head_dim;
      kv.push_back(c);
    }
  }
};

// One request whose computed prefix is [0, start) and whose query is
// [start, start + n). Blocks are laid out in REVERSE order (block b of the
// request lives in physical block kBlocks-1-b) so a block-table bug cannot hide.
CommonAttentionMetadata Meta(int64_t start, int64_t n) {
  CommonAttentionMetadata m;
  const int64_t seq = start + n;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(n);
  m.query_start_loc = {0, static_cast<int32_t>(n)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(seq)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(n);
  m.max_seq_len = static_cast<int>(seq);
  m.block_table_num_cols = static_cast<int>(Cache::kBlocks);
  for (int64_t b = 0; b < Cache::kBlocks; ++b)
    m.block_table_tensor.push_back(static_cast<int32_t>(Cache::kBlocks - 1 - b));
  for (int64_t t = start; t < seq; ++t)
    m.slot_mapping.push_back((Cache::kBlocks - 1 - t / Cache::kBlock) * Cache::kBlock +
                             t % Cache::kBlock);
  m.causal = true;
  return m;
}

std::vector<int32_t> Tokens() {
  return std::vector<int32_t>(std::begin(c2m_golden::kTokens), std::end(c2m_golden::kTokens));
}

std::vector<float> Prefill(const Cohere2MoeWeights& w, const std::vector<int32_t>& tok) {
  Cache c(w.params);
  std::vector<int32_t> pos(tok.size());
  for (size_t i = 0; i < pos.size(); ++i) pos[i] = static_cast<int32_t>(i);
  vt::Queue q = Q();
  return Cohere2MoeModel::Forward(tok, pos, Meta(0, static_cast<int64_t>(tok.size())), c.kv, w, q);
}

// Prefill `first` tokens, then decode the rest one at a time through the paged
// cache; returns the logits of every position, row-major, like Prefill.
std::vector<float> Incremental(const Cohere2MoeWeights& w, const std::vector<int32_t>& tok,
                               size_t first) {
  Cache c(w.params);
  vt::Queue q = Q();
  std::vector<float> all;
  size_t done = 0;
  while (done < tok.size()) {
    const size_t n = done == 0 ? first : 1;
    std::vector<int32_t> ids(tok.begin() + static_cast<long>(done),
                             tok.begin() + static_cast<long>(done + n));
    std::vector<int32_t> pos(n);
    for (size_t i = 0; i < n; ++i) pos[i] = static_cast<int32_t>(done + i);
    const std::vector<float> lg = Cohere2MoeModel::Forward(
        ids, pos, Meta(static_cast<int64_t>(done), static_cast<int64_t>(n)), c.kv, w, q);
    all.insert(all.end(), lg.begin(), lg.end());
    done += n;
  }
  return all;
}

double RelMax(const std::vector<float>& got, const float* want) {
  double scale = 0.0;
  for (size_t i = 0; i < got.size(); ++i) scale = std::max(scale, std::abs(static_cast<double>(want[i])));
  return vllm_test::MaxAbsDiff(got, want, got.size()) / scale;
}

// f32 arm bound: the reference and the port both accumulate in f32 over the
// same bf16-valued weights, so the residue is summation order. Measured below
// 2e-6 on all three configs; 2e-5 leaves 10x.
constexpr double kF32Rel = 2e-5;
// bf16 arm bound: the envelope of two bf16 paths that round at the same model
// stores but differ in rounding detail (GEMM accumulation order, RoPE in f32
// against torch's per-op bf16). The reference's OWN bf16-vs-f32 distance on
// these configs is 4.6e-3..2.6e-2 on the two configs whose bf16 routing agrees
// with f32; 3e-2 is that class, not a tuned number.
constexpr double kBf16Rel = 3e-2;

void CheckConfig(const char* json, const float* f32, const float* bf16, const std::string& name) {
  Loaded m(json);
  const std::vector<int32_t> tok = Tokens();
  const std::vector<float> b = Prefill(m.w, tok);
  REQUIRE(b.size() == tok.size() * static_cast<size_t>(m.w.params.vocab_size));
  const double eb = RelMax(b, bf16);
  m.UseF32();
  const std::vector<float> f = Prefill(m.w, tok);
  const double ef = RelMax(f, f32);
  // Decode through the paged cache past the window (W = 4 on every config).
  const std::vector<float> inc = Incremental(m.w, tok, 3);
  const double ei = RelMax(inc, f32);
  MESSAGE("config " << name << ": f32 prefill rel " << ef << ", f32 decode rel " << ei
                    << ", bf16 rel " << eb);
  CHECK(ef < kF32Rel);
  CHECK(ei < kF32Rel);
  CHECK(eb < kBf16Rel);
}

}  // namespace

// ---------------------------------------------------------------- (a) config

TEST_CASE("cohere2_moe config: the released North-Mini-Code-1.0 geometry") {
  const HfConfig cfg = ReleasedConfig();
  CHECK(vllm::ModelRegistry::Resolve(cfg).architecture == "Cohere2MoeForCausalLM");
  const Cohere2MoeParams p = vllm::ParseCohere2MoeParams(cfg);
  CHECK(p.hidden_size == 2048);
  CHECK(p.num_layers == 49);
  CHECK(p.vocab_size == 262144);
  CHECK(p.num_heads == 32);
  CHECK(p.num_kv_heads == 4);
  CHECK(p.head_dim == 128);
  CHECK(p.num_experts == 128);
  CHECK(p.top_k == 8);
  CHECK(p.intermediate_size == 768);
  CHECK(p.prefix_dense_intermediate_size == 3072);
  CHECK(p.num_shared_experts == 0);
  CHECK(p.sigmoid_router);
  CHECK_FALSE(p.norm_topk_prob);
  // Both eps keys ship; rms_norm_eps wins (select_norm_impl).
  CHECK(p.use_rms_norm);
  CHECK(p.norm_eps == doctest::Approx(1e-6));
  CHECK(p.logit_scale == 1.0);
  CHECK(p.rope_theta == 50000.0);
  CHECK(p.max_position == 500000);
  int sliding = 0, rope = 0, dense = 0;
  for (int64_t l = 0; l < p.num_layers; ++l) {
    const size_t i = static_cast<size_t>(l);
    const bool is_sliding = (l % 4) != 0;  // full every 4th layer from 0
    CHECK(p.window[i].has_value() == is_sliding);
    if (is_sliding) CHECK(*p.window[i] == 4097);  // sliding_window + 1
    // RoPE on sliding layers and on the forced-RoPE dense prefix (layer 0).
    CHECK(p.rope[i] == (is_sliding || l == 0));
    // mlp_layer_types is absent in the release: normalized from
    // first_k_dense_replace = 1.
    CHECK(p.dense[i] == (l == 0));
    sliding += is_sliding;
    rope += p.rope[i];
    dense += p.dense[i];
  }
  CHECK(sliding == 36);
  CHECK(rope == 37);
  CHECK(dense == 1);
}

TEST_CASE("cohere2_moe config: refusals by name") {
  const nlohmann::json base = ReleasedRaw();
  auto refuses = [&](nlohmann::json raw, const char* needle) {
    CAPTURE(needle);
    CHECK_THROWS_WITH(vllm::ParseCohere2MoeParams(WithRaw(raw)), doctest::Contains(needle));
  };
  nlohmann::json r = base;
  r["use_qk_norm"] = true;
  refuses(r, "use_qk_norm=true is not supported");
  r = base;
  r["sliding_window"] = nullptr;
  refuses(r, "is sliding_attention but sliding_window is null or 0");
  r = base;
  r["sliding_window"] = 0;
  refuses(r, "is sliding_attention but sliding_window is null or 0");
  r = base;
  r["quantization_config"] = {{"quant_method", "fp8"}};
  refuses(r, "quantization_config");
  r = base;
  r["rope_scaling"] = {{"rope_type", "yarn"}, {"factor", 4.0},
                       {"original_max_position_embeddings", 8192}};
  refuses(r, "rope_scaling 'yarn' is not supported");
  r = base;
  r["num_shared_experts"] = 1;
  r["shared_expert_combination_strategy"] = "max";
  refuses(r, "shared_expert_combination_strategy must be one of");
  r = base;
  r["sliding_window"] = 4096.5;
  refuses(r, "sliding_window must be an integer (got 4096.5)");
  r = base;
  r["layer_types"] = nlohmann::json::array({"full_attention"});
  refuses(r, "layer_types must have num_hidden_layers entries");
}

TEST_CASE("cohere2_moe config: absent keys take the Cohere2MoeConfig defaults") {
  nlohmann::json r = ReleasedRaw();
  for (const char* k : {"layer_types", "rms_norm_eps", "logit_scale", "sliding_window",
                        "expert_selection_fn", "norm_topk_prob", "head_dim"})
    r.erase(k);
  const Cohere2MoeParams p = vllm::ParseCohere2MoeParams(WithRaw(r));
  CHECK_FALSE(p.use_rms_norm);  // LayerNorm with layer_norm_eps
  CHECK(p.norm_eps == doctest::Approx(1e-5));
  CHECK(p.logit_scale == 0.0625);
  CHECK_FALSE(p.sigmoid_router);
  CHECK(p.norm_topk_prob);
  CHECK(p.head_dim == 128);
  // Derived layer_types: the dense prefix (first_k_dense_replace = 1) with
  // prefix pattern 1 is full; the rest follow sliding_window_pattern 4, so the
  // full layers are 4, 8, ... counted from the first MoE layer.
  CHECK_FALSE(p.window[0].has_value());
  CHECK(p.window[1].value() == 4097);
  CHECK_FALSE(p.window[4].has_value());
  CHECK(p.rope[0]);
  CHECK_FALSE(p.rope[4]);
  // An integral float is the same window as the integer.
  nlohmann::json f = ReleasedRaw();
  f["sliding_window"] = 4096.0;
  CHECK(vllm::ParseCohere2MoeParams(WithRaw(f)).window[1].value() == 4097);
}

TEST_CASE("cohere2_moe: GGUF weights are refused by name") {
  const HfConfig cfg = ReleasedConfig();
  vllm::ModelSource src;
  src.kind = vllm::ModelSource::Kind::kGguf;
  CHECK_THROWS_WITH(vllm::ModelRegistry::Load(cfg, src),
                    doctest::Contains("Cohere2MoeForCausalLM does not support GGUF weights"));
}

TEST_CASE("cohere2_moe: the KV-cache spec is one full-attention group") {
  const HfConfig cfg = ReleasedConfig();
  const auto& reg = vllm::ModelRegistry::Resolve(cfg);
  const vllm::v1::KVCacheConfig kv = reg.factory->make_kv_cache(cfg, 16, 4);
  REQUIRE(kv.kv_cache_groups.size() == 1);
  const auto* spec = dynamic_cast<const vllm::v1::FullAttentionSpec*>(
      kv.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(spec != nullptr);
  CHECK(spec->num_kv_heads == 4);
  CHECK(spec->head_size == 128);
}

// ---------------------------------------------------------------- (b) forward

TEST_CASE("cohere2_moe forward: config a (North switches) vs the pinned transcription") {
  CheckConfig(c2m_golden::kConfig_a, c2m_golden::kLogitsF32_a, c2m_golden::kLogitsBf16_a, "a");
}

TEST_CASE("cohere2_moe forward: config b (LayerNorm, shared average, NoPE prefix) vs the "
          "pinned transcription") {
  CheckConfig(c2m_golden::kConfig_b, c2m_golden::kLogitsF32_b, c2m_golden::kLogitsBf16_b, "b");
}

TEST_CASE("cohere2_moe forward: config c (softmax router, shared sum, defaults) vs the "
          "pinned transcription") {
  CheckConfig(c2m_golden::kConfig_c, c2m_golden::kLogitsF32_c, c2m_golden::kLogitsBf16_c, "c");
}

TEST_CASE("cohere2_moe forward: config d (non-contiguous dense layer is NoPE) vs the "
          "pinned transcription") {
  // Layer 2 is dense but not in the contiguous dense prefix, so force_rope is
  // off and the full-attention layer is NoPE (is_prefix_dense_layer, :49-53).
  const Cohere2MoeParams p =
      vllm::ParseCohere2MoeParams(c2m_tiny::ConfigFromJson(c2m_golden::kConfig_d));
  CHECK(p.dense[2]);
  CHECK_FALSE(p.rope[2]);
  CHECK(p.rope[0]);
  CHECK(p.window[1].value() == 4);  // sliding_window 3.0 (an integral float) + 1
  CheckConfig(c2m_golden::kConfig_d, c2m_golden::kLogitsF32_d, c2m_golden::kLogitsBf16_d, "d");
}

TEST_CASE("cohere2_moe forward: disable_sliding_window is refused by name") {
  Loaded m(c2m_golden::kConfig_a);
  vllm::SetDisableSlidingWindow(true);
  CHECK_THROWS_WITH(Prefill(m.w, Tokens()),
                    doctest::Contains("disable_sliding_window is not supported"));
  vllm::ResetDisableSlidingWindowForTesting();
}

// ---------------------------------------------------------------- loader

TEST_CASE("cohere2_moe loader: every shipped tensor is claimed, lm_head is dropped") {
  const HfConfig cfg = c2m_tiny::ConfigFromJson(c2m_golden::kConfig_a);
  const Cohere2MoeParams p = vllm::ParseCohere2MoeParams(cfg);
  auto load = [&](std::vector<c2m_tiny::Tensor> ts) {
    c2m_tiny::TempFile f(c2m_tiny::BuildSt(ts), ".safetensors");
    std::vector<SafetensorsFile> shards;
    shards.push_back(SafetensorsFile::Open(f.path()));
    (void)vllm::LoadCohere2MoeWeights(shards, cfg);
  };
  const std::vector<c2m_tiny::Tensor> base = c2m_tiny::SynthTensors(p);

  std::vector<c2m_tiny::Tensor> ts = base;
  ts.push_back({"lm_head.weight", {p.vocab_size, p.hidden_size}, "BF16",
                std::string(static_cast<size_t>(p.vocab_size * p.hidden_size * 2), '\0')});
  CHECK_NOTHROW(load(ts));  // cohere2_moe.py:489 orig_to_new_prefix {"lm_head.": None}

  ts = base;
  ts.push_back({"model.layers.1.self_attn.q_norm.weight", {p.head_dim}, "BF16",
                std::string(static_cast<size_t>(p.head_dim * 2), '\0')});
  CHECK_THROWS_WITH(load(ts), doctest::Contains(
                                  "model.layers.1.self_attn.q_norm.weight is not a "
                                  "Cohere2MoeForCausalLM weight"));

  ts = base;
  for (auto& t : ts)
    if (t.name == "model.layers.1.mlp.experts.0.up_proj.weight") {
      t.dtype = "F16";
    }
  CHECK_THROWS_WITH(load(ts), doctest::Contains("has dtype F16; only the bf16 checkpoint"));

  ts = base;
  ts.erase(std::remove_if(ts.begin(), ts.end(),
                          [](const c2m_tiny::Tensor& t) {
                            return t.name == "model.layers.3.mlp.experts.7.down_proj.weight";
                          }),
           ts.end());
  CHECK_THROWS_WITH(load(ts), doctest::Contains(
                                  "tensor not found: model.layers.3.mlp.experts.7.down_proj.weight"));
}

// ---------------------------------------------------------------- (c) real

namespace {
std::vector<float> ReadF32(const std::string& path) {
  std::ifstream in(path, std::ios::binary);
  REQUIRE(in.good());
  in.seekg(0, std::ios::end);
  const size_t n = static_cast<size_t>(in.tellg()) / 4;
  in.seekg(0);
  std::vector<float> v(n);
  in.read(reinterpret_cast<char*>(v.data()), static_cast<std::streamsize>(n * 4));
  return v;
}
std::vector<int32_t> ReadI32(const std::string& path) {
  const std::vector<float> raw = ReadF32(path);
  std::vector<int32_t> v(raw.size());
  std::memcpy(v.data(), raw.data(), raw.size() * 4);
  return v;
}
}  // namespace

TEST_CASE("cohere2_moe REAL: North layers 0, 1, 4 and the final norm vs the transcription") {
  const char* dir_env = std::getenv("VLLM_COHERE2_MOE_REAL_DIR");
  if (dir_env == nullptr) {
    MESSAGE("SKIP: VLLM_COHERE2_MOE_REAL_DIR unset");
    return;
  }
  const std::string dir(dir_env);
  const HfConfig cfg = vllm::LoadHfConfig(dir + "/config.json");
  std::vector<SafetensorsFile> shards;
  for (const char* f : {"sub_norm", "sub_s1", "sub_s2", "sub_s4", "sub_s5"})
    shards.push_back(SafetensorsFile::Open(dir + "/" + f + ".safetensors"));
  Cohere2MoeWeights w;
  w.params = vllm::ParseCohere2MoeParams(cfg);
  const std::vector<float> hidden = ReadF32(dir + "/hidden_in.f32");
  const std::vector<int32_t> pos = ReadI32(dir + "/positions.i32");
  const int64_t rows = pos.back() + 1;
  // The final norm weight, copied verbatim (bf16 [H]); the layers below go
  // through the production layer loader.
  {
    const SafetensorsFile& f = shards[0];
    const vllm::StTensor& t = f.Get("model.norm.weight");
    REQUIRE(t.dtype == "BF16");
    w.final_norm.dtype = DType::kBF16;
    w.final_norm.rank = 1;
    w.final_norm.shape[0] = w.params.hidden_size;
    w.final_norm.bytes.resize(t.nbytes);
    std::memcpy(w.final_norm.bytes.data(), t.data, t.nbytes);
  }
  vt::Queue q = Q();
  for (const auto& [dt, tag, bound] :
       {std::tuple{DType::kF32, std::string("f32"), 5e-5},
        std::tuple{DType::kBF16, std::string("bf16"), 3e-2}}) {
    w.compute_dtype = dt;
    w.rope_cos_sin = vllm::BuildCohere2MoeRopeCache(w.params, rows, dt);
    for (int64_t l : {0, 1, 4}) {
      const vllm::Cohere2MoeLayerWeights lw = vllm::LoadCohere2MoeLayerWeights(shards, w.params, l);
      const std::vector<float> got = Cohere2MoeModel::DecoderLayer(w, lw, l, hidden, pos, q);
      const std::vector<float> want =
          ReadF32(dir + "/layer" + std::to_string(l) + "_" + tag + ".f32");
      // Gate the layer's CONTRIBUTION (out - residual), not the output: the
      // residual is passed through unchanged and would dilute a defect in
      // attn + mlp by the ratio of their magnitudes.
      std::vector<float> dgot(got.size()), dwant(want.size());
      for (size_t i = 0; i < got.size(); ++i) {
        dgot[i] = got[i] - hidden[i];
        dwant[i] = want[i] - hidden[i];
      }
      const double e = RelMax(dgot, dwant.data());
      MESSAGE("REAL layer " << l << " (" << tag << "): rel max err of attn+mlp " << e
                            << " (output rel " << RelMax(got, want.data()) << ")");
      CHECK(e < bound);
    }
    const std::vector<float> got = Cohere2MoeModel::FinalNorm(w, hidden, q);
    const std::vector<float> want = ReadF32(dir + std::string("/norm_") + tag + ".f32");
    const double e = RelMax(got, want.data());
    MESSAGE("REAL final norm (" << tag << "): rel max err " << e);
    CHECK(e < bound);
  }
}

// ---------------------------------------------------------------- (e) structural

TEST_CASE("cohere2_moe STRUCTURAL: the released index is fully accounted") {
  const char* idx_env = std::getenv("VLLM_COHERE2_MOE_INDEX");
  if (idx_env == nullptr) {
    MESSAGE("SKIP: VLLM_COHERE2_MOE_INDEX unset");
    return;
  }
  std::ifstream in(idx_env);
  REQUIRE(in.good());
  const nlohmann::json idx = nlohmann::json::parse(in);
  std::set<std::string> shipped;
  for (const auto& [name, shard] : idx.at("weight_map").items()) {
    (void)shard;
    if (name.rfind("lm_head.", 0) == 0) continue;
    shipped.insert(name);
  }
  const Cohere2MoeParams p = vllm::ParseCohere2MoeParams(ReleasedConfig());
  const std::vector<std::string> names = vllm::EnumerateCohere2MoeTensors(p);
  const std::set<std::string> enumerated(names.begin(), names.end());
  CHECK(names.size() == enumerated.size());
  size_t unaccounted = 0, missing = 0;
  for (const std::string& n : shipped) unaccounted += enumerated.count(n) == 0;
  for (const std::string& n : enumerated) missing += shipped.count(n) == 0;
  MESSAGE("STRUCTURAL: shipped " << shipped.size() << ", enumerated " << enumerated.size()
                                 << ", unaccounted " << unaccounted << ", missing " << missing);
  CHECK(shipped.size() == 18730);
  CHECK(unaccounted == 0);
  CHECK(missing == 0);
}
