// MiMoV2 W5 gate test (MODEL-TEXT-mimo-v2, ISSUE-LOCAL-01M3F82S8ZYCTTDSKPFF5PGAH7).
//
// The W5 gate: the forward hook runs through the production entry point
// (ModelRegistry::Forward), produces finite, non-constant, token-dependent
// logits, and is deterministic across two identical steps.
//
// A tiny MiMoV2 safetensors fixture is written to a temp dir, loaded through
// ModelRegistry::Load, and forwarded through ModelRegistry::Forward. The
// topology mirrors the real model: hybrid full/SWA attention with per-layer
// KV geometry split, layer-0 dense MLP, MoE on the rest.

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/mimo_v2.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace {

// ── deterministic value generator (dots3_note pattern) ──────────────────────
inline uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

inline std::vector<double> Values(int64_t n, uint64_t seed, double amp,
                                  double bias = 0.0) {
  std::vector<double> v(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) {
    const double u =
        static_cast<double>(Mix(seed + static_cast<uint64_t>(i)) >> 40) /
        static_cast<double>(1 << 24);
    const float f = static_cast<float>((u * 2.0 - 1.0) * amp + bias);
    v[static_cast<size_t>(i)] =
        static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(f)));
  }
  return v;
}

// ── safetensors writer (dots3_note pattern) ─────────────────────────────────
struct StOut {
  std::string name;
  std::vector<int64_t> shape;
  std::vector<double> values;
  std::string dtype = "BF16";
};

inline void WriteSafetensors(const std::vector<StOut>& entries,
                             const std::string& path) {
  nlohmann::json header = nlohmann::json::object();
  size_t off = 0;
  for (const StOut& e : entries) {
    size_t n = 1;
    for (int64_t s : e.shape) n *= static_cast<size_t>(s);
    const size_t w = e.dtype == "F32" ? 4u : 2u;
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {off, off + n * w}}};
    off += n * w;
  }
  const std::string hs = header.dump();
  std::ofstream out(path, std::ios::binary);
  const uint64_t hlen = hs.size();
  out.write(reinterpret_cast<const char*>(&hlen), 8);
  out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  for (const StOut& e : entries) {
    for (double v : e.values) {
      if (e.dtype == "F32") {
        const float f = static_cast<float>(v);
        out.write(reinterpret_cast<const char*>(&f), 4);
      } else {
        const uint16_t b = vt::F32ToBF16(static_cast<float>(v));
        out.write(reinterpret_cast<const char*>(&b), 2);
      }
    }
  }
}

// ── tiny model geometry ─────────────────────────────────────────────────────
// 6 layers: [0,1,1,1,1,0] — full at 0 and 5, SWA at 1-4.
// head_dim=48, v_head_dim=32, rotary_dim=16, 4 attention heads.
// full KV: 2 heads; SWA KV: 4 heads. sliding_window=8.
// 4 experts, top-2, moe_intermediate_size=16. intermediate_size=64.
// vocab=32, hidden=32.
constexpr int64_t kHidden = 32;
constexpr int64_t kLayers = 6;
constexpr int64_t kVocab = 32;
constexpr int64_t kHeads = 4;
constexpr int64_t kHeadDim = 48;
constexpr int64_t kVHeadDim = 32;
constexpr int64_t kRotaryDim = 16;
constexpr int64_t kKvFull = 2;
constexpr int64_t kKvSwa = 4;
constexpr int64_t kSlidingWindow = 8;
constexpr int64_t kInter = 64;
constexpr int64_t kExperts = 4;
constexpr int64_t kTopK = 2;
constexpr int64_t kMoeInter = 16;
constexpr int64_t kBlockSize = 4;
constexpr int64_t kNumBlocks = 8;

const std::vector<int>& HybridPattern() {
  static const std::vector<int> p = {0, 1, 1, 1, 1, 0};
  return p;
}

bool IsFull(int l) { return HybridPattern()[static_cast<size_t>(l)] == 0; }
bool IsMoe(int l) { return l != 0; }

// ── config builder ───────────────────────────────────────────────────────────
vllm::HfConfig MakeTinyConfig() {
  vllm::HfConfig config;
  config.model_type = "mimo_v2";
  config.architectures = {"MiMoV2ForCausalLM"};
  config.hidden_size = kHidden;
  config.num_hidden_layers = kLayers;
  config.vocab_size = kVocab;
  config.num_attention_heads = kHeads;

  nlohmann::json j;
  j["head_dim"] = kHeadDim;
  j["v_head_dim"] = kVHeadDim;
  j["sliding_window"] = kSlidingWindow;
  j["partial_rotary_factor"] = static_cast<double>(kRotaryDim) /
                               static_cast<double>(kHeadDim);
  j["num_kv_heads_full"] = kKvFull;
  j["num_kv_heads_swa"] = kKvSwa;
  j["rope_theta_full"] = 1000000.0;
  j["rope_theta_swa"] = 10000.0;
  j["add_swa_attention_sink_bias"] = true;
  j["attention_value_scale"] = 0.5;
  j["intermediate_size"] = kInter;
  j["tie_word_embeddings"] = false;
  j["rms_norm_eps"] = 1e-6;
  j["hidden_act"] = "silu";
  j["hybrid_layer_pattern"] = HybridPattern();
  j["n_routed_experts"] = kExperts;
  j["num_experts_per_tok"] = kTopK;
  j["moe_intermediate_size"] = kMoeInter;
  j["moe_router_dtype"] = "bfloat16";
  j["n_shared_experts"] = false;
  std::vector<int> moe_freq(kLayers, 1);
  moe_freq[0] = 0;
  j["moe_layer_freq"] = moe_freq;

  config.raw = j;
  return config;
}

// ── safetensors fixture builder ──────────────────────────────────────────────
std::vector<StOut> BuildFixtureTensors() {
  std::vector<StOut> out;
  const int64_t H = kHidden;
  const int64_t V = kVocab;
  const int64_t E = kExperts;
  const int64_t MI = kMoeInter;
  const int64_t I = kInter;
  const int64_t Dh = kHeadDim;
  const int64_t Dv = kVHeadDim;
  const int64_t Hq = kHeads;

  auto add = [&](const std::string& name, std::vector<int64_t> shape,
                 uint64_t seed, double amp,
                 std::string dtype = "BF16") {
    int64_t n = 1;
    for (int64_t s : shape) n *= s;
    out.push_back({name, shape, Values(n, seed, amp), dtype});
  };

  // Global tensors
  // embed_tokens: [vocab, hidden] — loaded via LoadBf16Direct (not transposed)
  add("model.embed_tokens.weight", {V, H}, 100, 0.1);
  // final norm: [hidden]
  add("model.norm.weight", {H}, 200, 0.1);
  // lm_head: on-disk [vocab, hidden] → transposed to [hidden, vocab]
  add("lm_head.weight", {V, H}, 300, 0.1);

  for (int64_t l = 0; l < kLayers; ++l) {
    const bool full = IsFull(static_cast<int>(l));
    const int64_t Hkv = full ? kKvFull : kKvSwa;
    const uint64_t base = static_cast<uint64_t>(l) * 1000 + 4000;

    // layernorms
    add("model.layers." + std::to_string(l) + ".input_layernorm.weight", {H},
        base, 0.1);
    add("model.layers." + std::to_string(l) + ".post_attention_layernorm.weight",
        {H}, base + 1, 0.1);

    // attention projections (on-disk [out, in], transposed on load)
    // q_proj: [Hq*Dh, H]
    add("model.layers." + std::to_string(l) + ".self_attn.q_proj.weight",
        {Hq * Dh, H}, base + 10, 0.05);
    // k_proj: [Hkv*Dh, H]
    add("model.layers." + std::to_string(l) + ".self_attn.k_proj.weight",
        {Hkv * Dh, H}, base + 11, 0.05);
    // v_proj: [Hkv*Dv, H]
    add("model.layers." + std::to_string(l) + ".self_attn.v_proj.weight",
        {Hkv * Dv, H}, base + 12, 0.05);
    // o_proj: [H, Hq*Dv] — on-disk [out=H, in=Hq*Dv]
    add("model.layers." + std::to_string(l) + ".self_attn.o_proj.weight",
        {H, Hq * Dv}, base + 13, 0.05);

    // sink_bias on SWA layers only (F32, no .weight suffix)
    if (!full) {
      add("model.layers." + std::to_string(l) + ".self_attn.attention_sink_bias",
          {Hq}, base + 14, 0.1, "F32");
    }

    if (IsMoe(static_cast<int>(l))) {
      // MoE: router_gate [E, H] BF16 (not transposed)
      add("model.layers." + std::to_string(l) + ".mlp.gate.weight", {E, H},
          base + 20, 0.1);
      // e_score_correction_bias [E] F32 (no .weight suffix)
      add("model.layers." + std::to_string(l) + ".mlp.gate.e_score_correction_bias",
          {E}, base + 21, 0.1, "F32");
      // experts
      for (int64_t e = 0; e < E; ++e) {
        const uint64_t es = base + 100 + static_cast<uint64_t>(e) * 10;
        // gate_proj: [MI, H] on-disk (transposed)
        add("model.layers." + std::to_string(l) + ".mlp.experts." +
                std::to_string(e) + ".gate_proj.weight",
            {MI, H}, es, 0.05);
        // up_proj: [MI, H]
        add("model.layers." + std::to_string(l) + ".mlp.experts." +
                std::to_string(e) + ".up_proj.weight",
            {MI, H}, es + 1, 0.05);
        // down_proj: [H, MI]
        add("model.layers." + std::to_string(l) + ".mlp.experts." +
                std::to_string(e) + ".down_proj.weight",
            {H, MI}, es + 2, 0.05);
      }
    } else {
      // Dense MLP (layer 0)
      // gate_proj: [I, H] on-disk (transposed)
      add("model.layers." + std::to_string(l) + ".mlp.gate_proj.weight", {I, H},
          base + 30, 0.05);
      add("model.layers." + std::to_string(l) + ".mlp.up_proj.weight", {I, H},
          base + 31, 0.05);
      add("model.layers." + std::to_string(l) + ".mlp.down_proj.weight", {H, I},
          base + 32, 0.05);
    }
  }

  return out;
}

// ── temp dir ────────────────────────────────────────────────────────────────
std::string MakeTempDir() {
  const auto now = std::chrono::steady_clock::now().time_since_epoch().count();
  std::string dir = "/tmp/mimov2_forward_test_" + std::to_string(now);
  std::filesystem::create_directories(dir);
  return dir;
}

void WriteConfigJson(const std::string& dir, const vllm::HfConfig& config) {
  nlohmann::json doc = config.raw;
  doc["model_type"] = config.model_type;
  doc["architectures"] = config.architectures;
  doc["hidden_size"] = config.hidden_size;
  doc["num_hidden_layers"] = config.num_hidden_layers;
  doc["vocab_size"] = config.vocab_size;
  doc["num_attention_heads"] = config.num_attention_heads;
  std::ofstream(dir + "/config.json") << doc.dump(2);
}

// ── Topology: the published KV cache topology ────────────────────────────────
struct Topology {
  static constexpr int32_t kBlockPerm[8] = {5, 2, 7, 1, 6, 0, 3, 4};

  vt::DType dtype = vt::DType::kBF16;
  std::vector<std::vector<uint8_t>> attn_bytes;
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<std::string> names;
  std::vector<int32_t> group_ids;
  std::vector<int32_t> layer_indices;
  std::vector<uint8_t> payload_kinds;
  std::vector<int32_t> payload_slots;
  std::vector<std::vector<int32_t>> group_bt;
  std::vector<int32_t> group_cols;
  vllm::MultiKvCacheIndex mk;

  Topology() {
    const int64_t elt = static_cast<int64_t>(vt::SizeOf(dtype));
    // Two groups: group 0 = full-attn (layers 0, 5), group 1 = SWA (1,2,3,4).
    // One paged cache per published name, in publication order:
    // group 0 first (layers 0, 5), then group 1 (layers 1,2,3,4).
    // Running paged slot counter over paged entries only.
    int32_t paged_slot = 0;

    // Emit group 0 (full-attn) entries
    for (int l = 0; l < static_cast<int>(kLayers); ++l) {
      if (!IsFull(l)) continue;
      names.push_back("model.layers." + std::to_string(l) + ".self_attn");
      group_ids.push_back(0);
      layer_indices.push_back(l);
      payload_kinds.push_back(
          static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
      payload_slots.push_back(paged_slot++);
    }
    // Emit group 1 (SWA) entries
    for (int l = 0; l < static_cast<int>(kLayers); ++l) {
      if (IsFull(l)) continue;
      names.push_back("model.layers." + std::to_string(l) + ".self_attn");
      group_ids.push_back(1);
      layer_indices.push_back(l);
      payload_kinds.push_back(
          static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
      payload_slots.push_back(paged_slot++);
    }

    // Allocate paged cache buffers: one per paged entry.
    // Page layout: [K: bs*Hkv*Dh_k | V: bs*Hkv*Dh_v] per block.
    for (size_t i = 0; i < names.size(); ++i) {
      const int32_t slot = payload_slots[i];
      const int32_t l = layer_indices[i];
      const int64_t Hkv = IsFull(l) ? kKvFull : kKvSwa;
      const int64_t page_bytes = kNumBlocks * kBlockSize *
                                 (Hkv * kHeadDim + Hkv * kVHeadDim) * elt;
      attn_bytes.emplace_back(static_cast<size_t>(page_bytes), 0);

      vllm::PagedKvCache kv;
      kv.data = attn_bytes[slot].data();
      kv.dtype = dtype;
      kv.num_blocks = kNumBlocks;
      kv.block_size = kBlockSize;
      kv.num_kv_heads = Hkv;
      kv.head_size = kHeadDim;
      kv.head_size_v = kVHeadDim;
      attn_kv.push_back(kv);
    }

    // Block tables: one per group.
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

  static int64_t Slot(int64_t pos) {
    return static_cast<int64_t>(kBlockPerm[pos / kBlockSize]) * kBlockSize +
           pos % kBlockSize;
  }

  void ZeroPages() {
    for (auto& b : attn_bytes) std::fill(b.begin(), b.end(), 0);
    Publish();
  }
};

// ── Step: builds ModelForwardInput ───────────────────────────────────────────
struct Step {
  std::vector<int32_t> token_ids;
  std::vector<int32_t> positions;
  std::vector<int32_t> logits_indices;
  vllm::v1::CommonAttentionMetadata attn_meta{};
  vllm::v1::GDNAttentionMetadata gdn_meta{};
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  vllm::HfConfig config{};
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  int num_reqs = 1;
  const vllm::MultiKvCacheIndex* multi_kv = nullptr;
  std::unique_ptr<Topology> own;

  explicit Step(std::vector<int32_t> ids, std::vector<int32_t> want = {},
                int64_t computed = 0)
      : token_ids(std::move(ids)), logits_indices(std::move(want)) {
    const int64_t T = static_cast<int64_t>(token_ids.size());
    positions.resize(token_ids.size());
    for (size_t i = 0; i < positions.size(); ++i)
      positions[i] = static_cast<int32_t>(computed + static_cast<int64_t>(i));
    attn_meta.num_reqs = 1;
    attn_meta.num_actual_tokens = static_cast<int>(T);
    attn_meta.num_computed_tokens_cpu = {static_cast<int32_t>(computed)};
    attn_meta.seq_lens_cpu = {static_cast<int32_t>(computed + T)};
    attn_meta.seq_lens = attn_meta.seq_lens_cpu;
    attn_meta.query_start_loc = {0, static_cast<int32_t>(T)};
    attn_meta.query_start_loc_cpu = attn_meta.query_start_loc;
    attn_meta.block_table_tensor =
        std::vector<int32_t>(Topology::kBlockPerm,
                              Topology::kBlockPerm + 8);
    attn_meta.block_table_num_cols = 8;
    attn_meta.max_seq_len = static_cast<int>(computed + T);
    attn_meta.max_query_len = static_cast<int>(T);
    attn_meta.causal = true;
    for (int64_t t = 0; t < T; ++t)
      attn_meta.slot_mapping.push_back(Topology::Slot(computed + t));
    own = std::make_unique<Topology>();
    Bind(*own);
  }

  Step& Bind(Topology& t) {
    attn_kv = t.attn_kv;
    multi_kv = &t.mk;
    return *this;
  }

  vllm::ModelForwardInput Get() {
    vllm::ModelForwardInput in{
        .token_ids = token_ids,
        .positions = positions,
        .attn_meta = attn_meta,
        .gdn_meta = gdn_meta,
        .attn_kv = attn_kv,
        .gdn_state = gdn_state,
        .config = config,
        .queue = queue,
        .logits_indices = logits_indices,
        .num_reqs = num_reqs};
    in.multi_kv = multi_kv;
    return in;
  }
};

// ── Gap: max-abs-diff with non-finite guard ──────────────────────────────────
struct Gap {
  double max_abs = 0.0;
  bool any_non_finite = false;

  void operator()(float a, float b) {
    if (!std::isfinite(a) || !std::isfinite(b)) {
      any_non_finite = true;
      return;
    }
    max_abs = std::max(max_abs, static_cast<double>(std::fabs(a - b)));
  }
};

// ── download device logits to host ──────────────────────────────────────────
std::vector<float> DownloadLogits(const vllm::ForwardLogits& fl,
                                  vt::Queue& q) {
  const size_t n = static_cast<size_t>(fl.rows) * static_cast<size_t>(fl.vocab);
  if (!fl.on_device()) {
    return fl.host;
  }
  std::vector<float> out(n);
  vt::Backend& be = vt::GetBackend(q.device.type);
  be.Copy(q, out.data(), fl.device_tensor.data, n * sizeof(float));
  be.Synchronize(q);
  return out;
}

}  // namespace

// ═══ TESTS ═══════════════════════════════════════════════════════════════════

TEST_CASE("mimov2_forward: hook runs and returns real logits") {
  const std::string dir = MakeTempDir();
  const vllm::HfConfig config = MakeTinyConfig();
  WriteConfigJson(dir, config);
  WriteSafetensors(BuildFixtureTensors(), dir + "/model.safetensors");

  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(dir + "/model.safetensors"));
  vllm::ModelSource source =
      vllm::ModelSource::FromSafetensors(shards);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);

  Step step({3, 11, 7, 20}, {3});
  step.config = config;
  const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, step.Get());

  REQUIRE(out.rows == 1);
  REQUIRE(out.vocab == kVocab);

  std::vector<float> logits = DownloadLogits(out, step.queue);
  REQUIRE(static_cast<int64_t>(logits.size()) == kVocab);

  // Finite
  for (float v : logits) CHECK(std::isfinite(v));

  // Non-constant
  float lo = logits[0], hi = logits[0];
  for (float v : logits) {
    lo = std::min(lo, v);
    hi = std::max(hi, v);
  }
  CHECK(hi - lo > 1e-6f);
}

TEST_CASE("mimov2_forward: logits are token-dependent") {
  const std::string dir = MakeTempDir();
  const vllm::HfConfig config = MakeTinyConfig();
  WriteConfigJson(dir, config);
  WriteSafetensors(BuildFixtureTensors(), dir + "/model.safetensors");

  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(dir + "/model.safetensors"));
  vllm::ModelSource source =
      vllm::ModelSource::FromSafetensors(shards);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);

  // Two different tokens at the same position → different logits.
  Step s1({3}, {0});
  s1.config = config;
  const vllm::ForwardLogits o1 = vllm::ModelRegistry::Forward(*model, s1.Get());
  std::vector<float> l1 = DownloadLogits(o1, s1.queue);

  Step s2({11}, {0});
  s2.config = config;
  const vllm::ForwardLogits o2 = vllm::ModelRegistry::Forward(*model, s2.Get());
  std::vector<float> l2 = DownloadLogits(o2, s2.queue);

  REQUIRE(l1.size() == l2.size());
  Gap g;
  for (size_t i = 0; i < l1.size(); ++i) g(l1[i], l2[i]);
  CHECK(!g.any_non_finite);
  CHECK(g.max_abs > 1e-6);
}

TEST_CASE("mimov2_forward: deterministic across two identical steps") {
  const std::string dir = MakeTempDir();
  const vllm::HfConfig config = MakeTinyConfig();
  WriteConfigJson(dir, config);
  WriteSafetensors(BuildFixtureTensors(), dir + "/model.safetensors");

  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(dir + "/model.safetensors"));
  vllm::ModelSource source =
      vllm::ModelSource::FromSafetensors(shards);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);

  // Run the same 4-token prefill twice on FRESH (zeroed) caches.
  Step s1({3, 11, 7, 20}, {3});
  s1.config = config;
  const vllm::ForwardLogits o1 = vllm::ModelRegistry::Forward(*model, s1.Get());
  std::vector<float> l1 = DownloadLogits(o1, s1.queue);

  Step s2({3, 11, 7, 20}, {3});
  s2.config = config;
  const vllm::ForwardLogits o2 = vllm::ModelRegistry::Forward(*model, s2.Get());
  std::vector<float> l2 = DownloadLogits(o2, s2.queue);

  REQUIRE(l1.size() == l2.size());
  Gap g;
  for (size_t i = 0; i < l1.size(); ++i) g(l1[i], l2[i]);
  CHECK(!g.any_non_finite);
  CHECK(g.max_abs == 0.0);  // exact match
}

TEST_CASE("mimov2_forward: multi-token prefill returns T rows") {
  const std::string dir = MakeTempDir();
  const vllm::HfConfig config = MakeTinyConfig();
  WriteConfigJson(dir, config);
  WriteSafetensors(BuildFixtureTensors(), dir + "/model.safetensors");

  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(dir + "/model.safetensors"));
  vllm::ModelSource source =
      vllm::ModelSource::FromSafetensors(shards);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);

  // 4 tokens, ALL positions' logits (empty logits_indices → returns all T rows).
  Step step({3, 11, 7, 20}, {});
  step.config = config;
  const vllm::ForwardLogits out = vllm::ModelRegistry::Forward(*model, step.Get());

  // Empty logits_indices means no gather → forward returns all T rows.
  // The forward itself checks `do_gather = !empty && size < T`.
  // With empty indices, do_gather=false, n_out=T.
  REQUIRE(out.rows == 4);
  REQUIRE(out.vocab == kVocab);

  std::vector<float> logits = DownloadLogits(out, step.queue);
  REQUIRE(static_cast<int64_t>(logits.size()) == 4 * kVocab);

  // All finite
  for (float v : logits) CHECK(std::isfinite(v));

  // Rows differ (token-dependent across positions)
  Gap g;
  for (int64_t v = 0; v < kVocab; ++v) g(logits[v], logits[kVocab + v]);
  CHECK(g.max_abs > 1e-6);
}
