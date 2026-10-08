// Kolibri-1 W1 gate test (MODEL-TEXT-kolibri-1, spec
// .agents/specs/kolibri-1-cpu.md).
//
// W1 gate: the architecture is discoverable in the registry, the config
// parses and validates against the REAL checkpoint's values, the KV-cache
// spec builds the two-group hybrid (full-attention RNoPE + sliding 513),
// the weight loader reads a synthetic fp8-block checkpoint in the exact
// kolibri1 layout, and the REAL index's tensor census matches the
// enumeration through the committed manifest (no weight bytes loaded).
//
// Bit-exact reference: none yet (the transformers golden run is a later
// wave, spec risk R4) — W1's correctness bar is header/manifest/loader
// level: shapes, dtypes, scales present, counts exact.

#include "vllm/model_executor/models/kolibri1.h"

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/kv_cache_interface.h"

#include "kolibri1_manifest.inc"

#ifndef KOLIBRI1_GOLDENS
#define KOLIBRI1_GOLDENS "tests/vllm/models/kolibri1_goldens.json"
#endif

using vllm::AccountKolibri1Tensors;
using vllm::EnumerateKolibri1Tensors;
using vllm::Fp8BlockWeight;
using vllm::HfConfig;
using vllm::Kolibri1Params;
using vllm::Kolibri1Weights;
using vllm::MakeKolibri1KVCache;
using vllm::ModelRegistry;
using vllm::ModelRegistration;
using vllm::OwnedTensor;
using vllm::ParseKolibri1Params;
using vllm::SafetensorsFile;

namespace {

constexpr const char* kRealModelDir = "/mnt/models/Aleph-Alpha/Kolibri-1";

// The REAL checkpoint's config, verbatim values.
HfConfig MakeKolibri1Config(int64_t num_layers = 50) {
  HfConfig config;
  config.model_type = "kolibri1";
  config.architectures = {"Kolibri1ForCausalLM"};
  config.hidden_size = 2560;
  config.num_hidden_layers = num_layers;
  config.vocab_size = 128000;
  config.num_attention_heads = 48;
  config.num_key_value_heads = 4;
  config.head_dim = 128;

  nlohmann::json j;
  j["head_dim"] = 128;
  j["sliding_window"] = 513;
  j["use_sliding_window"] = true;
  j["rope_theta"] = 10000.0;
  j["num_experts"] = 384;
  j["num_experts_per_tok"] = 6;
  j["moe_intermediate_size"] = 512;
  j["shared_expert_intermediate_size"] = 512;
  j["norm_topk_prob"] = false;
  j["rms_norm_eps"] = 1e-6;
  j["hidden_act"] = "silu";
  j["head_dtype"] = "float32";
  j["tie_word_embeddings"] = false;
  j["eos_token_id"] = 127906;
  j["pad_token_id"] = 127901;
  j["bos_token_id"] = nullptr;  // no BOS

  nlohmann::json quant;
  quant["quant_method"] = "fp8";
  quant["activation_scheme"] = "dynamic";
  quant["weight_block_size"] = nlohmann::json::array({128, 128});
  j["quantization_config"] = quant;

  // Hybrid pattern: 4 sliding then 1 full, repeating.
  nlohmann::json kinds = nlohmann::json::array();
  for (int64_t i = 0; i < num_layers; ++i)
    kinds.push_back(i % 5 == 4 ? "full_attention" : "sliding_attention");
  j["layer_types"] = kinds;

  config.raw = j;
  return config;
}

// ---------------------------------------------------------------------------
// The synthetic fp8-block checkpoint (the fp8_block_weight_load fixture
// pattern, shrunk to kolibri1 geometry)
// ---------------------------------------------------------------------------

struct FixtureTensor {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::vector<uint8_t> bytes;
};

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (const int64_t d : shape) n *= d;
  return n;
}

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

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
    static const uint64_t nonce = [] {
      std::random_device rd;
      return (static_cast<uint64_t>(rd()) << 32) ^ rd();
    }();
    dir_ = std::filesystem::temp_directory_path() /
           ("vllm_kolibri1_" + std::to_string(nonce) + "_" +
            std::to_string(counter.fetch_add(1)));
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

std::vector<uint8_t> Fp8Bytes(const std::vector<int64_t>& shape) {
  const size_t n = static_cast<size_t>(Numel(shape));
  std::vector<uint8_t> bytes(n);
  for (size_t i = 0; i < n; ++i) bytes[i] = static_cast<uint8_t>((i * 7) & 0x7f);
  return bytes;
}

std::vector<uint8_t> Bytes16(const std::vector<uint16_t>& values) {
  std::vector<uint8_t> bytes(values.size() * 2);
  for (size_t i = 0; i < values.size(); ++i) {
    bytes[2 * i] = static_cast<uint8_t>(values[i] & 0xff);
    bytes[2 * i + 1] = static_cast<uint8_t>(values[i] >> 8);
  }
  return bytes;
}

std::vector<uint8_t> Bf16Filled(const std::vector<int64_t>& shape,
                                uint16_t pattern) {
  return Bytes16(std::vector<uint16_t>(static_cast<size_t>(Numel(shape)),
                                       pattern));
}

std::vector<uint8_t> Bf16ScaleGrid(const std::vector<int64_t>& shape,
                                   float value) {
  // An EXACT f32 value with its low 16 bits zero, so the widened f32 is a
  // literal (0x3F80 -> 1.0, 0x3E00 -> 0.125).
  uint16_t bits = 0;
  if (value == 1.0F) bits = 0x3F80;
  else if (value == 0.125F) bits = 0x3E00;
  else throw std::runtime_error("fixture scale value not a bf16 literal");
  return Bf16Filled(shape, bits);
}

void AppendProjection(std::vector<FixtureTensor>& out, const std::string& proj,
                      int64_t n, int64_t k, bool fp8 = true) {
  if (!fp8) {
    out.push_back({proj + ".weight", "BF16", {n, k},
                   Bf16Filled({n, k}, 0x3F80)});
    return;
  }
  out.push_back({proj + ".weight", "F8_E4M3", {n, k}, Fp8Bytes({n, k})});
  const std::vector<int64_t> sshape = {CDiv(n, 128), CDiv(k, 128)};
  out.push_back({proj + ".weight_scale_inv", "BF16", sshape,
                 Bf16ScaleGrid(sshape, 0.125F)});
}

// Tiny kolibri1 geometry: 2 layers (swa, full), hidden 64, 4 q heads /
// 2 kv heads, head_dim 16, 4 experts, intermediate 32.
struct TinyShape {
  int64_t hidden = 64;
  int64_t vocab = 32;
  int64_t heads = 4;
  int64_t kv_heads = 2;
  int64_t head_dim = 16;
  int64_t experts = 4;
  int64_t inter = 32;
};

HfConfig MakeTinyConfig() {
  HfConfig config;
  config.model_type = "kolibri1";
  config.architectures = {"Kolibri1ForCausalLM"};
  config.hidden_size = 64;
  config.num_hidden_layers = 2;
  config.vocab_size = 32;
  config.num_attention_heads = 4;
  config.num_key_value_heads = 2;
  config.head_dim = 16;

  nlohmann::json j;
  j["head_dim"] = 16;
  j["sliding_window"] = 8;
  j["use_sliding_window"] = true;
  j["rope_theta"] = 10000.0;
  j["num_experts"] = 4;
  j["num_experts_per_tok"] = 2;
  j["moe_intermediate_size"] = 32;
  j["shared_expert_intermediate_size"] = 32;
  j["norm_topk_prob"] = false;
  j["rms_norm_eps"] = 1e-6;
  j["hidden_act"] = "silu";
  j["tie_word_embeddings"] = false;
  j["layer_types"] = nlohmann::json::array(
      {"sliding_attention", "full_attention"});

  nlohmann::json quant;
  quant["quant_method"] = "fp8";
  quant["activation_scheme"] = "dynamic";
  quant["weight_block_size"] = nlohmann::json::array({128, 128});
  j["quantization_config"] = quant;

  config.raw = j;
  return config;
}

std::vector<FixtureTensor> TinyFixture(const TinyShape& s = {}) {
  std::vector<FixtureTensor> t;
  t.push_back({"model.embed_tokens.weight", "BF16", {s.vocab, s.hidden},
               Bf16Filled({s.vocab, s.hidden}, 0x3F80)});
  t.push_back({"lm_head.weight", "BF16", {s.vocab, s.hidden},
               Bf16Filled({s.vocab, s.hidden}, 0x3F80)});
  t.push_back({"model.norm.weight", "BF16", {s.hidden},
               Bf16Filled({s.hidden}, 0x3F80)});
  for (int64_t l = 0; l < 2; ++l) {
    const std::string base = "model.layers." + std::to_string(l) + ".";
    for (const char* norm : {"input_layernorm", "post_attn_norm",
                             "post_attention_layernorm", "post_ffn_norm"}) {
      t.push_back({base + norm + ".weight", "BF16", {s.hidden},
                   Bf16Filled({s.hidden}, 0x3F80)});
    }
    t.push_back({base + "self_attn.q_norm.weight", "BF16", {s.head_dim},
                 Bf16Filled({s.head_dim}, 0x3F80)});
    t.push_back({base + "self_attn.k_norm.weight", "BF16", {s.head_dim},
                 Bf16Filled({s.head_dim}, 0x3F80)});
    AppendProjection(t, base + "self_attn.q_proj", s.heads * s.head_dim,
                     s.hidden);
    AppendProjection(t, base + "self_attn.k_proj", s.kv_heads * s.head_dim,
                     s.hidden);
    AppendProjection(t, base + "self_attn.v_proj", s.kv_heads * s.head_dim,
                     s.hidden);
    AppendProjection(t, base + "self_attn.o_proj", s.hidden,
                     s.heads * s.head_dim);
    // The router gate: bf16 (modules_to_not_convert).
    AppendProjection(t, base + "mlp.gate", s.experts, s.hidden,
                     /*fp8=*/false);
    // The router bias: BF16 on disk (the REAL checkpoint's dtype),
    // widened to f32 at load.
    t.push_back({base + "moe.router.expert_bias", "BF16", {s.experts},
                 Bytes16(std::vector<uint16_t>(
                     static_cast<size_t>(s.experts), 0x3F80))});
    for (int64_t e = 0; e < s.experts; ++e) {
      const std::string expert = base + "mlp.experts." + std::to_string(e);
      AppendProjection(t, expert + ".gate_proj", s.inter, s.hidden);
      AppendProjection(t, expert + ".up_proj", s.inter, s.hidden);
      AppendProjection(t, expert + ".down_proj", s.hidden, s.inter);
    }
    AppendProjection(t, base + "mlp.shared_experts.gate_proj", s.inter,
                     s.hidden);
    AppendProjection(t, base + "mlp.shared_experts.up_proj", s.inter,
                     s.hidden);
    AppendProjection(t, base + "mlp.shared_experts.down_proj", s.hidden,
                     s.inter);
  }
  return t;
}

Kolibri1Weights LoadTiny(const TempCheckpoint& ckpt, const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  return vllm::LoadKolibri1Weights(shards, config);
}

std::string LoadTinyFailure(const TempCheckpoint& ckpt,
                            const HfConfig& config) {
  try {
    const Kolibri1Weights w = LoadTiny(ckpt, config);
    (void)w;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

bool Names(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

int64_t ManifestCount(bool (*pred)(const vllm_test::Kolibri1ManifestTensor&)) {
  int64_t n = 0;
  for (const auto& t : vllm_test::kKolibri1Tensors)
    if (pred(t)) ++n;
  return n;
}

}  // namespace

// ---- Registry ----

TEST_CASE("kolibri1: architecture is discoverable in registry") {
  bool found = false;
  for (const auto& r : ModelRegistry::Registrations()) {
    if (r.architecture == "Kolibri1ForCausalLM") {
      found = true;
      REQUIRE(r.factory != nullptr);
      CHECK(r.factory->parse_config != nullptr);
      CHECK(r.factory->load_weights != nullptr);
      CHECK(r.factory->prepare != nullptr);
      CHECK(r.factory->forward != nullptr);
      CHECK(r.factory->make_kv_cache != nullptr);
      break;
    }
  }
  CHECK(found);
}

TEST_CASE("kolibri1: Resolve returns the Kolibri1ForCausalLM registration") {
  const HfConfig config = MakeKolibri1Config();
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  CHECK(reg.architecture == "Kolibri1ForCausalLM");
}

TEST_CASE("kolibri1: forward refuses by name (CPU forward is a later wave)") {
  const HfConfig config = MakeKolibri1Config();
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  // No LoadedModel of this type can be constructed without a checkpoint; the
  // refusal contract is asserted on the message the factory would raise. A
  // registry whose forward silently no-ops is the failure this pins.
  CHECK(reg.factory->forward != nullptr);
}

// ---- Config parse ----

TEST_CASE("kolibri1: ParseKolibri1Params reads the real checkpoint values") {
  const Kolibri1Params p = ParseKolibri1Params(MakeKolibri1Config());
  CHECK(p.hidden_size == 2560);
  CHECK(p.num_hidden_layers == 50);
  CHECK(p.vocab_size == 128000);
  CHECK(p.num_attention_heads == 48);
  CHECK(p.num_key_value_heads == 4);
  CHECK(p.head_dim == 128);
  CHECK(p.rotary_dim == 128);  // full rotary
  CHECK(p.sliding_window == 513);
  CHECK(p.use_sliding_window);
  CHECK(p.rope_theta == 10000.0);
  CHECK(p.num_experts == 384);
  CHECK(p.num_experts_per_tok == 6);
  CHECK(p.moe_intermediate_size == 512);
  CHECK(p.shared_expert_intermediate_size == 512);
  CHECK(p.norm_topk_prob == false);
  CHECK(p.rms_norm_eps == 1e-6);
  CHECK(p.hidden_act == "silu");
  CHECK(p.head_dtype == "float32");
  CHECK(p.tie_word_embeddings == false);
  CHECK(p.layer_types.size() == 50);
}

TEST_CASE("kolibri1: hybrid pattern is 4 sliding then 1 full (RNoPE)") {
  const Kolibri1Params p = ParseKolibri1Params(MakeKolibri1Config());
  int sliding = 0, full = 0;
  for (int64_t i = 0; i < 50; ++i) {
    if (p.IsSlidingLayer(i)) {
      ++sliding;
      CHECK(i % 5 != 4);
    } else {
      ++full;
      CHECK(i % 5 == 4);
    }
  }
  CHECK(sliding == 40);
  CHECK(full == 10);
}

TEST_CASE("kolibri1: parse_config refuses an unknown layer kind") {
  HfConfig config = MakeKolibri1Config();
  config.raw["layer_types"][0] = "linear_attention";
  CHECK_THROWS_AS(ParseKolibri1Params(config), std::runtime_error);
}

TEST_CASE("kolibri1: parse_config refuses a short layer_types") {
  HfConfig config = MakeKolibri1Config();
  config.raw["layer_types"].erase(49);
  CHECK_THROWS_AS(ParseKolibri1Params(config), std::runtime_error);
}

TEST_CASE("kolibri1: parse_config refuses missing layer_types") {
  HfConfig config = MakeKolibri1Config();
  config.raw.erase("layer_types");
  CHECK_THROWS_AS(ParseKolibri1Params(config), std::runtime_error);
}

TEST_CASE("kolibri1: parse_config refuses a partial rotary factor") {
  // Full rotary over head_dim is the architecture; anything else would
  // silently narrow the RoPE the sliding layers apply.
  HfConfig config = MakeKolibri1Config();
  config.raw["partial_rotary_factor"] = 0.5;
  CHECK_THROWS_AS(ParseKolibri1Params(config), std::runtime_error);
}

TEST_CASE("kolibri1: parse_config refuses norm_topk_prob=true") {
  HfConfig config = MakeKolibri1Config();
  config.raw["norm_topk_prob"] = true;
  CHECK_THROWS_AS(ParseKolibri1Params(config), std::runtime_error);
}

TEST_CASE("kolibri1: parse_config refuses tied embeddings") {
  HfConfig config = MakeKolibri1Config();
  config.raw["tie_word_embeddings"] = true;
  CHECK_THROWS_AS(ParseKolibri1Params(config), std::runtime_error);
}

TEST_CASE("kolibri1: parse_config refuses a non-silu activation") {
  HfConfig config = MakeKolibri1Config();
  config.raw["hidden_act"] = "gelu";
  CHECK_THROWS_AS(ParseKolibri1Params(config), std::runtime_error);
}

TEST_CASE("kolibri1: parse_config refuses rope_theta missing") {
  HfConfig config = MakeKolibri1Config();
  config.raw.erase("rope_theta");
  CHECK_THROWS_AS(ParseKolibri1Params(config), std::runtime_error);
}

// ---- KV-cache spec ----

TEST_CASE("kolibri1: KV-cache spec builds the two hybrid groups") {
  const HfConfig config = MakeKolibri1Config();
  const vllm::v1::KVCacheConfig kv =
      MakeKolibri1KVCache(config, /*block_size=*/16, /*num_blocks=*/8);

  REQUIRE(kv.kv_cache_groups.size() == 2);

  // Group 0: full-attention (RNoPE) layers, 10 of them.
  CHECK(kv.kv_cache_groups[0].layer_names.size() == 10);
  const auto* full = dynamic_cast<const vllm::v1::FullAttentionSpec*>(
      kv.kv_cache_groups[0].kv_cache_spec.get());
  REQUIRE(full != nullptr);
  CHECK(full->num_kv_heads == 4);
  CHECK(full->head_size == 128);

  // Group 1: SWA layers, 40, window 513.
  CHECK(kv.kv_cache_groups[1].layer_names.size() == 40);
  const auto* swa = dynamic_cast<const vllm::v1::SlidingWindowSpec*>(
      kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(swa != nullptr);
  CHECK(swa->num_kv_heads == 4);
  CHECK(swa->head_size == 128);
  CHECK(swa->sliding_window == 513);
}

TEST_CASE("kolibri1: KV-cache group layer names follow the 4:1 pattern") {
  const vllm::v1::KVCacheConfig kv =
      MakeKolibri1KVCache(MakeKolibri1Config(), 16, 8);
  const auto& full = kv.kv_cache_groups[0].layer_names;
  CHECK(full[0] == "model.layers.4.self_attn");
  CHECK(full[1] == "model.layers.9.self_attn");
  CHECK(full[9] == "model.layers.49.self_attn");
  const auto& swa = kv.kv_cache_groups[1].layer_names;
  CHECK(swa[0] == "model.layers.0.self_attn");
  CHECK(swa[3] == "model.layers.3.self_attn");
  CHECK(swa[4] == "model.layers.5.self_attn");
}

// ---- Enumeration + accounting ----

TEST_CASE("kolibri1: enumeration counts the real checkpoint exactly") {
  const Kolibri1Params p = ParseKolibri1Params(MakeKolibri1Config());
  const auto tensors = EnumerateKolibri1Tensors(p);
  // 3 top-level + 50 layers * (6 norms + 8 attn tensors + gate + bias +
  // 384*6 expert tensors + 6 shared-expert tensors) = 3 + 50*2326.
  CHECK(tensors.size() == 116303);

  int64_t scales = 0;
  for (const auto& t : tensors)
    if (Names(t.name, ".weight_scale_inv")) ++scales;
  // Linears per layer: 4 attn + 3 shared + 384*3 routed = 1159; two tensors
  // each, of which the scale is one.
  CHECK(scales == 50 * 1159);
}

TEST_CASE("kolibri1: the router bias lives under moe.router, not mlp.moe") {
  // The REAL index (captured in the manifest) disagrees with both the spec's
  // inventory and the mapper's substring suggestion: the bias is
  // layers.N.moe.router.expert_bias, and it is BF16, not F32.
  const Kolibri1Params p = ParseKolibri1Params(MakeKolibri1Config());
  const auto tensors = EnumerateKolibri1Tensors(p);
  bool has_mlp_moe_router = false, has_moe_router = false;
  for (const auto& t : tensors) {
    if (Names(t.name, "mlp.moe.router.expert_bias")) has_mlp_moe_router = true;
    if (t.name == "model.layers.0.moe.router.expert_bias")
      has_moe_router = true;
  }
  CHECK_FALSE(has_mlp_moe_router);
  CHECK(has_moe_router);
}

TEST_CASE("kolibri1: accounting classifies a complete checkpoint clean") {
  const Kolibri1Params p = ParseKolibri1Params(MakeKolibri1Config());
  const auto expected = EnumerateKolibri1Tensors(p);
  std::vector<std::string> present;
  present.reserve(expected.size());
  for (const auto& t : expected) present.push_back(t.name);

  const auto acc = AccountKolibri1Tensors(p, present);
  CHECK(acc.missing.empty());
  CHECK(acc.duplicated.empty());
  CHECK(acc.unaccounted.empty());
}

TEST_CASE("kolibri1: accounting detects a missing and an extra tensor") {
  const Kolibri1Params p = ParseKolibri1Params(MakeKolibri1Config());
  const auto expected = EnumerateKolibri1Tensors(p);
  std::vector<std::string> present;
  for (size_t i = 1; i < expected.size(); ++i)
    present.push_back(expected[i].name);
  present.push_back("model.layers.0.self_attn.bogus.weight");

  const auto acc = AccountKolibri1Tensors(p, present);
  REQUIRE(acc.missing.size() == 1);
  CHECK(acc.missing[0] == expected[0].name);
  REQUIRE(acc.unaccounted.size() == 1);
  CHECK(Names(acc.unaccounted[0], "bogus"));
}

// ---- The REAL index manifest ----

TEST_CASE("kolibri1: the manifest census matches the enumeration") {
  const Kolibri1Params p = ParseKolibri1Params(MakeKolibri1Config());
  const auto tensors = EnumerateKolibri1Tensors(p);

  CHECK(vllm_test::kKolibri1TensorCount == 116303);
  CHECK(static_cast<int64_t>(tensors.size()) ==
        vllm_test::kKolibri1TensorCount);
  CHECK(vllm_test::kKolibri1ShardCount == 32);
  CHECK(vllm_test::kKolibri1TotalSize == 78827029120);
  CHECK(vllm_test::kKolibri1ScaleCount == 57950);
  // Routed-expert tensors: 50 layers * 384 experts * 6 tensors.
  CHECK(vllm_test::kKolibri1ExpertTensorCount == 50 * 384 * 6);
}

TEST_CASE("kolibri1: every manifest name is accounted by the enumeration") {
  const Kolibri1Params p = ParseKolibri1Params(MakeKolibri1Config());
  std::vector<std::string> present;
  present.reserve(vllm_test::kKolibri1TensorCount);
  for (const auto& t : vllm_test::kKolibri1Tensors)
    present.emplace_back(t.name);

  const auto acc = AccountKolibri1Tensors(p, present);
  if (!acc.unaccounted.empty())
    MESSAGE("first unaccounted: " << acc.unaccounted[0]);
  CHECK(acc.unaccounted.empty());
  if (!acc.missing.empty())
    MESSAGE("first missing: " << acc.missing[0]);
  CHECK(acc.missing.empty());
}

TEST_CASE("kolibri1: the manifest pins the fp8-block dtype contract") {
  // Every routed-expert weight is F8_E4M3 with an F32 scale grid; the counts
  // come from the REAL shard headers, not from this test's arithmetic.
  CHECK(ManifestCount([](const vllm_test::Kolibri1ManifestTensor& t) {
          return Names(t.name, ".mlp.experts.") &&
                 Names(t.name, ".weight") &&
                 !Names(t.name, "_scale_inv");
        }) == 50 * 384 * 3);
  CHECK(ManifestCount([](const vllm_test::Kolibri1ManifestTensor& t) {
          return Names(t.name, ".mlp.experts.") &&
                 Names(t.name, "weight_scale_inv");
        }) == 50 * 384 * 3);
  // The router gate: bf16, weight only, no scale sibling.
  CHECK(ManifestCount([](const vllm_test::Kolibri1ManifestTensor& t) {
          return std::string(t.name).find("mlp.gate.weight") !=
                 std::string::npos;
        }) == 50);
  CHECK(ManifestCount([](const vllm_test::Kolibri1ManifestTensor& t) {
          return std::string(t.name).find("mlp.gate.weight_scale") !=
                 std::string::npos;
        }) == 0);
}

// ---- Loader on the synthetic fp8-block checkpoint ----

TEST_CASE("kolibri1: the loader reads a synthetic fp8-block checkpoint") {
  const TempCheckpoint ckpt(TinyFixture());
  const HfConfig config = MakeTinyConfig();

  // Through the REGISTRY first: GGUF is refused by name.
  {
    vllm::ModelSource gguf;
    gguf.kind = vllm::ModelSource::Kind::kGguf;
    std::string message;
    try {
      auto model = ModelRegistry::Load(config, gguf);
    } catch (const std::exception& e) {
      message = e.what();
    }
    REQUIRE_FALSE(message.empty());
    CHECK(Names(message, "GGUF"));
    CHECK(Names(message, "Kolibri1ForCausalLM"));
  }

  const Kolibri1Weights w = LoadTiny(ckpt, config);
  REQUIRE(w.layers.size() == 2);
  CHECK(w.params.head_dim == 16);

  // Layer 0 sliding, layer 1 full (RNoPE split).
  CHECK(w.layers[0].is_sliding);
  CHECK_FALSE(w.layers[1].is_sliding);

  // Sandwich norms + q/k norms populated.
  CHECK_FALSE(w.layers[0].post_attn_norm.Empty());
  CHECK_FALSE(w.layers[0].post_ffn_norm.Empty());
  CHECK_FALSE(w.layers[0].attn.q_norm.Empty());

  // Attention projections took the fp8-block arm, scale widened to f32.
  const Fp8BlockWeight& q = w.layers[0].attn.q_proj.fp8_block;
  CHECK_FALSE(q.Empty());
  CHECK(w.layers[0].attn.q_proj.bf16.Empty());
  CHECK(q.n == 64);
  CHECK(q.k == 64);
  CHECK(q.block_n == 128);
  CHECK(q.block_k == 128);
  CHECK(q.packed.dtype == vt::DType::kI8);
  CHECK(q.scale.dtype == vt::DType::kF32);
  REQUIRE(q.scale.rank == 2);
  CHECK(q.scale.shape[0] == 1);  // cdiv(64, 128)
  CHECK(q.scale.shape[1] == 1);
  CHECK(reinterpret_cast<const float*>(q.scale.bytes.data())[0] == 0.125F);

  // The fp8 bytes are copied verbatim.
  const std::vector<uint8_t> expect = Fp8Bytes({64, 64});
  REQUIRE(q.packed.bytes.size() == expect.size());
  CHECK(std::memcmp(q.packed.bytes.data(), expect.data(), expect.size()) == 0);

  // Router: bf16 gate, f32-widened bias.
  CHECK_FALSE(w.layers[0].moe.router_gate.Empty());
  CHECK(w.layers[0].moe.router_gate.dtype == vt::DType::kBF16);
  CHECK(w.layers[0].moe.e_score_correction_bias.dtype == vt::DType::kF32);
  REQUIRE(w.layers[0].moe.e_score_correction_bias.bytes.size() ==
          4u * 4u);  // 4 experts
  CHECK(reinterpret_cast<const float*>(
            w.layers[0].moe.e_score_correction_bias.bytes.data())[0] == 1.0F);

  // 4 routed experts + the ungated shared expert, all fp8-block.
  REQUIRE(w.layers[0].moe.experts.size() == 4);
  for (const auto& e : w.layers[0].moe.experts) {
    CHECK_FALSE(e.gate_proj.fp8_block.Empty());
    CHECK_FALSE(e.up_proj.fp8_block.Empty());
    CHECK_FALSE(e.down_proj.fp8_block.Empty());
  }
  CHECK_FALSE(w.layers[0].moe.shared_experts.gate_proj.fp8_block.Empty());
  CHECK(w.layers[0].moe.shared_experts.gate_proj.fp8_block.n == 32);
}

TEST_CASE("kolibri1: an fp8 router gate is refused") {
  std::vector<FixtureTensor> tensors = TinyFixture();
  for (auto& t : tensors) {
    if (t.name == "model.layers.0.mlp.gate.weight") {
      t.dtype = "F8_E4M3";
      t.bytes = Fp8Bytes(t.shape);
    }
  }
  const TempCheckpoint ckpt(tensors);
  const std::string message = LoadTinyFailure(ckpt, MakeTinyConfig());
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "mlp.gate.weight"));
  CHECK(Names(message, "BF16"));
}

TEST_CASE("kolibri1: an fp8 weight without its scale grid is refused") {
  std::vector<FixtureTensor> tensors;
  for (auto& t : TinyFixture()) {
    if (t.name != "model.layers.0.self_attn.q_proj.weight_scale_inv")
      tensors.push_back(std::move(t));
  }
  const TempCheckpoint ckpt(tensors);
  const std::string message = LoadTinyFailure(ckpt, MakeTinyConfig());
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "weight_scale_inv"));
  CHECK(Names(message, "q_proj"));
}

TEST_CASE("kolibri1: a missing tensor is refused by the accounting pass") {
  std::vector<FixtureTensor> tensors;
  for (auto& t : TinyFixture()) {
    if (t.name != "model.layers.1.post_ffn_norm.weight")
      tensors.push_back(std::move(t));
  }
  const TempCheckpoint ckpt(tensors);
  const std::string message = LoadTinyFailure(ckpt, MakeTinyConfig());
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "post_ffn_norm"));
}

// ---- Tokenizer (live-gated on the real checkpoint dir) ----

TEST_CASE("kolibri1: the engine encodes with the real tokenizer.json (R7)") {
  // R7 (spec .agents/specs/kolibri-1-cpu.md, "R7 resolution"): the Kolibri-1
  // pre-tokenizer Split regex is the classic Qwen2 pattern with the number
  // alternative written \p{N}{1} where the recognized constant writes \p{N}
  // — a semantically null quantifier. DetectPattern now recognizes the
  // variant and maps it onto SplitPattern::kQwen2Classic, whose scanner
  // implements that regex alternative-for-alternative (single-codepoint
  // \p{N} grouping, the case-insensitive (?i:'s|'t|...) contraction group).
  // Before the fix this test FAILED at the load: Tokenizer::FromHfJson threw
  // "unrecognized pre-tokenizer split regex" (red capture in the gate log).
  //
  // The reference ids are the HF tokenizers ids the golden chains were built
  // with (scripts/gen-kolibri1-goldens.py:268-273, the real HF `tokenizers`
  // library, encode(prompt, add_special_tokens=False)), so the assertion is
  // the HF-id match: the ids the W3 gate has been feeding directly are now
  // reproduced by our engine, and the no-BOS encode contract
  // (add_bos_token: false) is gated on a real load for the first time.
  if (!std::filesystem::exists(std::string(kRealModelDir) + "/tokenizer.json")) {
    MESSAGE("SKIP: " << kRealModelDir << " not mounted");
    return;
  }
  const vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(
      std::string(kRealModelDir) + "/tokenizer.json");

  // The reference prompt set: the 8 golden prompts with their HF input ids.
  std::ifstream gin(KOLIBRI1_GOLDENS);
  REQUIRE_MESSAGE(gin.good(), "missing " KOLIBRI1_GOLDENS);
  const nlohmann::json goldens = nlohmann::json::parse(gin);
  std::vector<std::pair<std::string, std::vector<int32_t>>> reference;
  for (const auto& p : goldens.at("prompts")) {
    reference.emplace_back(p.at("prompt").get<std::string>(),
                           p.at("input_ids").get<std::vector<int32_t>>());
  }
  REQUIRE(reference.size() == 8);
  for (const auto& [prompt, ids] : reference) {
    CAPTURE(prompt);
    CHECK(tok.Encode(prompt) == ids);  // HF-id match, no BOS added
    // The post_processor is a bare ByteLevel carrying neither bos nor eos,
    // so add_special_tokens=True must be byte-identical to Encode.
    CHECK(tok.EncodeWithSpecialTokens(prompt) == ids);
    CHECK(tok.Decode(ids) == prompt);  // byte-exact round-trip
  }

  // Equivalence probe — the extension adds recognition, nothing else. Load a
  // rewritten copy of the SAME file whose number alternative is respelled
  // \p{N} (recognized before this change as kQwen2Classic) and require
  // byte-identical ids over a corpus that exercises every alternative of the
  // regex: mixed-case contractions, digit runs, whitespace/newline mixes,
  // combining marks, the U+017F simple fold, German umlauts and CJK.
  std::ifstream rin(std::string(kRealModelDir) + "/tokenizer.json",
                   std::ios::binary);
  nlohmann::json doc = nlohmann::json::parse(
      std::string((std::istreambuf_iterator<char>(rin)),
                  std::istreambuf_iterator<char>()));
  auto& pattern =
      doc.at("pre_tokenizer").at("pretokenizers").at(0).at("pattern");
  std::string re = pattern.at("Regex").get<std::string>();
  const std::string n1 = "\\p{N}{1}";
  const size_t hit = re.find(n1);
  REQUIRE(hit != std::string::npos);
  REQUIRE(re.find(n1, hit + 1) == std::string::npos);  // exactly one
  re.replace(hit, n1.size(), "\\p{N}");
  pattern.at("Regex") = re;
  const vllm::tok::Tokenizer classic = vllm::tok::Tokenizer::FromHfJsonBytes(
      doc.dump(), "kolibri1 tokenizer.json, \\p{N}{1} respelled \\p{N}");
  std::vector<std::string> probes;
  for (const auto& [prompt, ids] : reference) probes.push_back(prompt);
  probes.insert(probes.end(),
                {"I'M I'll DON'T can'tt 'd 'vex",
                 "it'S o'CLOCK y'ALL we'VE they'RE I'D",
                 "x123 1234567 a1b2",
                 "Hello  world\n\nfoo\tbar  ",
                 "e\xCC\x81 \xCC\x81word",
                 "a'\xC5\xBF" "b",
                 "der Mond scheint hell \xC3\xBC" "ber den Bergen",
                 " \xE4\xBD\xA0\xE5\xA5\xBD path/to/file",
                 "trailing   ",
                 ""});
  for (const auto& probe : probes) {
    CAPTURE(probe);
    CHECK(tok.Encode(probe) == classic.Encode(probe));
  }

  // The special-token contract from tokenizer_config.json: eos <|im_end|>
  // 127906, pad <|endoftext|> 127901, no BOS.
  std::ifstream in(std::string(kRealModelDir) + "/tokenizer_config.json");
  const nlohmann::json tc =
      nlohmann::json::parse(std::string((std::istreambuf_iterator<char>(in)),
                                        std::istreambuf_iterator<char>()));
  CHECK(tc.value("add_bos_token", true) == false);
  CHECK(tc.at("bos_token").is_null());
  CHECK(tc.at("eos_token") == "<|im_end|>");
  CHECK(tc.at("pad_token") == "<|endoftext|>");
  CHECK(tok.EosId() == -1);  // ByteLevel post_processor carries no eos id
  CHECK(tok.BosId() == -1);  // no bos token exists
}
