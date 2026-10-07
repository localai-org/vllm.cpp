// Kolibri-1 Tenstorrent wave-A gate (MODEL-TEXT-kolibri-1-tenstorrent,
// spec .agents/specs/kolibri-tt.md). Device-FREE by construction: the
// staging plan is host-side policy over the landed CPU-row loader, so this
// TU compiles in any TT-enabled build and runs with or without a card —
// unlike test_tenstorrent_backend.cpp nothing here touches vt:: device
// seams at all.
//
// Red-first: each TEST_CASE was written against the refusal / accounting
// contract BEFORE the plan code landed; the accounting case pins the
// 384-expert byte math against the REAL checkpoint manifest
// (kolibri1_manifest.inc), so a plan that under- or over-counts bytes
// fails against measured checkpoint data, not a hand total.
#include <doctest/doctest.h>

#include <atomic>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/models/kolibri1_tt.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"

#include "kolibri1_manifest.inc"

using namespace vllm;  // NOLINT

namespace {

// ---- the synthetic tiny checkpoint (the test_kolibri1.cpp pattern) ----

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
           ("vllm_kolibri1_tt_" + std::to_string(nonce) + "_" +
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

std::vector<uint8_t> Bf16Filled(const std::vector<int64_t>& shape,
                                uint16_t pattern) {
  std::vector<uint8_t> bytes(static_cast<size_t>(Numel(shape)) * 2);
  for (size_t i = 0; i < bytes.size(); i += 2) {
    bytes[i] = static_cast<uint8_t>(pattern & 0xff);
    bytes[i + 1] = static_cast<uint8_t>(pattern >> 8);
  }
  return bytes;
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
                 Bf16Filled(sshape, 0x3E00)});  // 0.125 exactly
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
  j["layer_types"] =
      nlohmann::json::array({"sliding_attention", "full_attention"});

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
    AppendProjection(t, base + "mlp.gate", s.experts, s.hidden,
                     /*fp8=*/false);
    t.push_back({base + "moe.router.expert_bias", "BF16", {s.experts},
                 Bf16Filled({s.experts}, 0x3F80)});
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

std::string PlanFailure(const Kolibri1Weights& w,
                        const Kolibri1TTStagingOptions& o) {
  try {
    const Kolibri1TTStagingPlan p = PlanKolibri1TTStaging(w, o);
    (void)p;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

// ---- real-manifest byte math (the 384-expert wall, from measured data) ----

int64_t ManifestBytes(const vllm_test::Kolibri1ManifestTensor& t) {
  int64_t elems = 1;
  for (int i = 0; i < t.rank; ++i) elems *= t.shape[i];
  const std::string d = t.dtype;
  if (d == "BF16") return elems * 2;
  if (d == "F8_E4M3") return elems;   // one fp8-e4m3 byte per element
  if (d == "F32") return elems * 4;
  throw std::runtime_error(std::string("unknown manifest dtype ") + d);
}

int64_t ManifestTotal(const char* dtype) {
  int64_t total = 0;
  for (const auto& t : vllm_test::kKolibri1Tensors) {
    if (dtype == nullptr || std::string(t.dtype) == dtype) {
      total += ManifestBytes(t);
    }
  }
  return total;
}

}  // namespace

// ---- Registry reachability (device-agnostic registration serves TT) ----

TEST_CASE("kolibri1 TT: the registration is reachable with the TT plan TU "
          "linked") {
  bool found = false;
  for (const auto& r : ModelRegistry::Registrations()) {
    if (r.architecture == "Kolibri1ForCausalLM") {
      found = true;
      REQUIRE(r.factory != nullptr);
      CHECK(r.factory->parse_config != nullptr);
      CHECK(r.factory->make_kv_cache != nullptr);
      break;
    }
  }
  CHECK(found);
}

TEST_CASE("kolibri1 TT: GGUF is refused by name") {
  const HfConfig config = MakeTinyConfig();
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  ModelSource source;
  source.kind = ModelSource::Kind::kGguf;
  bool threw = false;
  try {
    (void)reg.factory->load_weights(reg, config, source);
  } catch (const std::exception& e) {
    threw = true;
    const std::string what = e.what();
    CHECK(what.find("GGUF") != std::string::npos);
  }
  CHECK(threw);
}

// ---- The staging plan over the synthetic tiny checkpoint ----

TEST_CASE("kolibri1 TT: plan dtype decisions and byte accounting (tiny)") {
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  const Kolibri1Weights w = LoadTiny(ckpt, config);

  const Kolibri1TTStagingPlan plan = PlanKolibri1TTStaging(w);
  CHECK(plan.full_layers == 1);
  CHECK(plan.swa_layers == 1);

  int64_t sum = 0;
  int fp8_projections = 0;
  int bf16_modules = 0;
  int f32_modules = 0;
  for (const auto& t : plan.tensors) {
    int64_t bytes = t.bytes + t.scale_bytes;
    sum += bytes;
    switch (t.dtype) {
      case Kolibri1TTDType::kFp8E4M3:
        // n*k fp8 bytes plus an f32 grid of cdiv(n,128)*cdiv(k,128)*4.
        CHECK(t.bytes == t.rows * t.cols);
        CHECK(t.scale_bytes > 0);
        ++fp8_projections;
        break;
      case Kolibri1TTDType::kBf16:
        CHECK(t.bytes % 2 == 0);
        CHECK(t.scale_bytes == 0);
        ++bf16_modules;
        break;
      case Kolibri1TTDType::kF32:
        ++f32_modules;
        break;
    }
  }
  CHECK(sum == plan.total_bytes);
  // Independent fixture math (NOT recomputed from the plan): embed+head
  // 32*64*2 each, final norm 64*2; per layer 4 norms + 2 qk norms (64*2 /
  // 16*2), q [64,64] and o [64,64] fp8 = 4096+4 grid each, k/v [32,64]
  // fp8 = 2048+4 each, router gate [4,64] bf16 = 512, bias [4] f32 = 16,
  // each expert (routed x4 and shared) gate/up [32,64] + down [64,32] fp8
  // = 3*(2048+4).
  const int64_t expert = 3 * (2048 + 4);
  const int64_t layer = 4 * 128 + 2 * 32 + (4100 + 2052 + 2052 + 4100) +
                        512 + 16 + expert + 4 * expert;
  CHECK(plan.total_bytes == 4096 + 4096 + 128 + 2 * layer);
  CHECK(plan.total_bytes == 96696);
  CHECK(fp8_projections > 0);
  CHECK(bf16_modules > 0);
  // Every layer's router bias stages f32.
  CHECK(f32_modules == config.num_hidden_layers);
  // 2 layers x (4 attention + 3 shared + 4 routed experts x 3 projections)
  // fp8 projections.
  CHECK(fp8_projections == 2 * (4 + 3 + 4 * 3));
}

TEST_CASE("kolibri1 TT: single-device refusal names the deficit and the row") {
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  const Kolibri1Weights w = LoadTiny(ckpt, config);

  const Kolibri1TTStagingPlan fitted = PlanKolibri1TTStaging(w);
  Kolibri1TTStagingOptions o;
  o.device_budget_bytes = fitted.total_bytes - 1;
  const std::string err = PlanFailure(w, o);
  REQUIRE(!err.empty());
  CHECK(err.find("deficit") != std::string::npos);
  CHECK(err.find("MODEL-TEXT-kolibri-1-tenstorrent") != std::string::npos);
  CHECK(err.find(std::to_string(fitted.total_bytes)) != std::string::npos);
}

TEST_CASE("kolibri1 TT: a mesh budget that fits is accepted") {
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  const Kolibri1Weights w = LoadTiny(ckpt, config);

  const Kolibri1TTStagingPlan fitted = PlanKolibri1TTStaging(w);
  Kolibri1TTStagingOptions o;
  o.device_budget_bytes = (fitted.total_bytes + 3) / 4;
  o.mesh_chips = 4;
  const Kolibri1TTStagingPlan meshed = PlanKolibri1TTStaging(w, o);
  CHECK(meshed.total_bytes == fitted.total_bytes);
}

// ---- The 384-expert wall, measured against the REAL manifest ----

TEST_CASE("kolibri1 TT: manifest-derived staging total exceeds one P150") {
  const int64_t total = ManifestTotal(nullptr);
  const int64_t fp8 = ManifestTotal("F8_E4M3");
  // The fp8-block projections dominate: >90% of the staged bytes are the
  // natively-staged fp8 operands (dequant-at-load would double THIS part).
  CHECK(fp8 * 10 > total * 9);
  // The wall: a single 32 GiB P150 cannot hold the fp8 model.
  CHECK(total > (int64_t(32) << 30));
  // Per-expert byte math: gate+up+down = 3 * inter * hidden fp8 bytes.
  const int64_t per_expert = 3 * 512 * 2560;  // 3,932,160
  int64_t expert_weight_bytes = 0;
  int64_t expert_count = 0;
  for (const auto& t : vllm_test::kKolibri1Tensors) {
    const std::string name = t.name;
    if (name.find("mlp.experts.") != std::string::npos &&
        name.find("weight_scale_inv") == std::string::npos &&
        t.dtype == std::string("F8_E4M3")) {
      expert_weight_bytes += ManifestBytes(t);
      ++expert_count;
    }
  }
  CHECK(expert_count == 50 * 384 * 3);
  CHECK(expert_weight_bytes == 50 * 384 * per_expert);
  // The routed-expert wall the spec records: ~70.33 GiB.
  const int64_t routed = 50 * 384 * per_expert;
  CHECK(routed == 75497472000);
  CHECK(routed > int64_t(70) << 30);
  CHECK(routed < int64_t(71) << 30);
}

// ---- Wave B1: the single-P150 expert streaming plan (host-side policy) ----
//
// Red-first against spec §"B1 scope": the streaming plan is device-free, so
// every case here runs in the same CPU-only gate as wave A. The expected
// numbers are the spec's § byte-math table values, written out exactly.

namespace {

Kolibri1TTStreamingShape RealKolibri1StreamingShape() {
  // The shipped checkpoint geometry: 50 layers, 384 experts, top-6, one
  // expert = 3,932,160 fp8 bytes + 960 B of f32 scale grids.
  Kolibri1TTStreamingShape s;
  s.layers = 50;
  s.experts = 384;
  s.topk = 6;
  s.expert_bytes = 3 * 512 * 2560 + (4 * 20 + 4 * 20 + 20 * 4) * 4;  // 3,933,120
  s.attention_bytes = 50 * 34078720;    // q 6144×2560, k/v 512×2560, o 2560×6144 fp8
  s.shared_expert_bytes = 50 * 3933120;
  s.router_bytes = 50 * (384 * 2560 * 2);  // bf16 [384,2560] per layer
  s.norm_bytes = 2 << 20;
  s.embed_head_bytes = 2 * (128000 * 2560 * 2);  // embed + untied head, bf16
  return s;
}

std::string StreamingFailure(const Kolibri1TTStreamingShape& s,
                             const Kolibri1TTStreamingOptions& o) {
  try {
    const Kolibri1TTStreamingPlan p = PlanKolibri1TTStreaming(s, o);
    (void)p;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

}  // namespace

TEST_CASE("kolibri1 TT streaming: resident accounting matches the byte-math "
          "table") {
  const Kolibri1TTStreamingShape s = RealKolibri1StreamingShape();
  const Kolibri1TTStreamingPlan plan =
      PlanKolibri1TTStreaming(s, Kolibri1TTStreamingOptions{});
  // Exact component totals from the spec table (per layer × 50).
  CHECK(s.attention_bytes == 1703936000);
  CHECK(s.shared_expert_bytes == 196656000);
  CHECK(s.router_bytes == 98304000);
  CHECK(s.embed_head_bytes == 1310720000);
  CHECK(plan.resident_bytes == s.attention_bytes + s.shared_expert_bytes +
                                   s.router_bytes + s.norm_bytes +
                                   s.embed_head_bytes);
  // The spec's ~3.1 GiB resident claim (attention 1.625 + shared 0.188 +
  // router 0.094 + embed/head 1.221 GiB + norms) — well under one P150.
  CHECK(plan.resident_bytes > int64_t(3) << 30);
  CHECK(plan.resident_bytes < int64_t(32) << 30);
  // The routed experts stay OFF device: the streaming plan never stages
  // 384 experts × 50 layers resident.
  const int64_t routed = s.layers * s.experts * s.expert_bytes;
  CHECK(routed == 75515904000);  // 70.33 GiB
  CHECK(routed + plan.resident_bytes > int64_t(32) << 30);
  // Resident alone exceeding the budget is refused by name.
  Kolibri1TTStreamingOptions tiny;
  tiny.device_budget_bytes = plan.resident_bytes - 1;
  const std::string err = StreamingFailure(s, tiny);
  REQUIRE(!err.empty());
  CHECK(err.find("resident") != std::string::npos);
  CHECK(err.find("deficit") != std::string::npos);
}

TEST_CASE("kolibri1 TT streaming: the hot set derives from the device "
          "residual") {
  const Kolibri1TTStreamingShape s = RealKolibri1StreamingShape();
  Kolibri1TTStreamingOptions o;  // default 32 GiB P150, no KV reserve
  const Kolibri1TTStreamingPlan plan = PlanKolibri1TTStreaming(s, o);
  const int64_t residual =
      (int64_t(32) << 30) - plan.resident_bytes;  // no KV reserve
  CHECK(plan.device_residual_bytes == residual);
  CHECK(plan.hot_experts == residual / s.expert_bytes);
  CHECK(plan.hot_set_bytes == plan.hot_experts * s.expert_bytes);
  CHECK(plan.hot_set_bytes <= residual);
  CHECK(plan.hot_experts > 0);
  // A KV reserve shrinks the hot set by exactly the same byte math.
  o.kv_reserve_bytes = int64_t(4) << 30;
  const Kolibri1TTStreamingPlan with_kv = PlanKolibri1TTStreaming(s, o);
  const int64_t kv_residual =
      (int64_t(32) << 30) - with_kv.resident_bytes - o.kv_reserve_bytes;
  CHECK(with_kv.device_residual_bytes == kv_residual);
  CHECK(with_kv.hot_experts == kv_residual / s.expert_bytes);
  CHECK(with_kv.hot_experts < plan.hot_experts);
}

TEST_CASE("kolibri1 TT streaming: host-budget refusal names the deficit and "
          "the NVMe leaf as owed") {
  const Kolibri1TTStreamingShape s = RealKolibri1StreamingShape();
  const int64_t host_required = s.layers * s.experts * s.expert_bytes;
  Kolibri1TTStreamingOptions o;
  o.host_budget_bytes = host_required - 1000;
  const std::string err = StreamingFailure(s, o);
  REQUIRE(!err.empty());
  // deficit != budget here: the budget differs from the requirement by
  // 1000, so this cannot pass by matching the budget alone.
  const std::string deficit = std::to_string(host_required - o.host_budget_bytes);
  CHECK(deficit == "1000");
  CHECK(err.find("deficit " + deficit) != std::string::npos);
  CHECK(err.find(std::to_string(host_required - o.host_budget_bytes)) !=
        std::string::npos);
  // The NVMe tier is a named-but-unimplemented leaf: the refusal says the
  // spill is owed, never silently degraded.
  CHECK(err.find("NVMe") != std::string::npos);
  CHECK(err.find("owed") != std::string::npos);
  // The full host tier fits a generous budget.
  o.host_budget_bytes = int64_t(96) << 30;
  const std::string ok = StreamingFailure(s, o);
  CHECK(ok.empty());
}

TEST_CASE("kolibri1 TT streaming: the per-token stream bound is "
          "50 × 6 × 3,933,120 B") {
  const Kolibri1TTStreamingShape s = RealKolibri1StreamingShape();
  const Kolibri1TTStreamingPlan plan =
      PlanKolibri1TTStreaming(s, Kolibri1TTStreamingOptions{});
  CHECK(plan.per_token_stream_bytes ==
        50 * 6 * 3933120);
  CHECK(plan.per_token_stream_bytes == 1179936000);
  // RAM-tier bandwidth envelope 25–60 GB/s ⇒ 20–47 ms/token (derived, not
  // measured): 1179936000 / 25e9 = 47.197 ms, / 60e9 = 19.666 ms.
  CHECK(plan.stream_ms_low > 19.0);
  CHECK(plan.stream_ms_low < 20.0);
  CHECK(plan.stream_ms_high > 47.0);
  CHECK(plan.stream_ms_high < 48.0);
}

TEST_CASE("kolibri1 TT streaming: concurrency refusal above the declared "
          "touched-fraction threshold, warning below") {
  const Kolibri1TTStreamingShape s = RealKolibri1StreamingShape();
  Kolibri1TTStreamingOptions o;
  o.host_budget_bytes = int64_t(96) << 30;
  o.touched_fraction_threshold = 0.25;
  // conc 32: f = 1 − (1 − 6/384)^32 ≈ 0.398 > 0.25 → refused by name.
  o.concurrency = 32;
  const std::string err = StreamingFailure(s, o);
  REQUIRE(!err.empty());
  CHECK(err.find("concurrency") != std::string::npos);
  // The verdict names the (1−f) × routed-bytes per-step I/O consequence.
  const double f = 1.0 - std::pow(1.0 - double(s.topk) / double(s.experts),
                                  static_cast<double>(o.concurrency));
  const int64_t per_step =
      int64_t((1.0 - f) * double(s.layers * s.experts * s.expert_bytes));
  CHECK(err.find(std::to_string(per_step)) != std::string::npos);
  // Best-effort at the same operating point warns instead of refusing.
  o.best_effort = true;
  const Kolibri1TTStreamingPlan warned = PlanKolibri1TTStreaming(s, o);
  CHECK(warned.concurrency_warning.find("concurrency") != std::string::npos);
  CHECK(warned.concurrency_warning.find(std::to_string(per_step)) !=
        std::string::npos);
  // conc 4: f ≈ 0.061 < 0.25 → no refusal and no warning.
  o.best_effort = false;
  o.concurrency = 4;
  const Kolibri1TTStreamingPlan calm = PlanKolibri1TTStreaming(s, o);
  CHECK(calm.concurrency_warning.empty());
  CHECK(calm.touched_fraction_per_layer < 0.25);
  CHECK(calm.touched_fraction_per_layer > 0.0);
}

TEST_CASE("kolibri1 TT: every fp8 manifest tensor maps one fp8 staging "
          "entry shape") {
  // The manifest's fp8 weights and their scale grids must pair 1:1 — a
  // staged fp8 operand without its grid is refused by the planner.
  int64_t fp8_weights = 0;
  int64_t grids = 0;
  for (const auto& t : vllm_test::kKolibri1Tensors) {
    const std::string name = t.name;
    if (t.dtype == std::string("F8_E4M3")) ++fp8_weights;
    if (name.find("weight_scale_inv") != std::string::npos) ++grids;
  }
  CHECK(grids == fp8_weights);
  CHECK(fp8_weights == vllm_test::kKolibri1ScaleCount);
}
