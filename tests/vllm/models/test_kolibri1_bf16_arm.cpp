// Kolibri-1 bf16 arm gate test (MODEL-TEXT-kolibri-1, spec
// .agents/specs/kolibri1-bf16-arm.md).
//
// The fp8 arm is the row's production path; the bf16 arm exists so the fp8
// weight quantization's OWN contribution to the divergence from the primary
// oracle becomes measurable. This file holds the four claims:
//
//   1. a plain bf16 Kolibri-1 checkpoint (no quantization_config) LOADS and
//      the CPU forward produces logits;
//   2. the fp8 arm still loads exactly as before (no regression);
//   3. a scheme that is neither fp8-block nor bf16 is refused BY NAME, and
//      the message names the missing arm;
//   4. the two arms produce the SAME result for the SAME bytes: a synthetic
//      fp8 checkpoint and its exact bf16 dequantization must agree, because
//      the same bytes must ride the same compute class.
//
// The fixture is the W1 tiny geometry (2 layers: 1 sliding + 1 full, hidden
// 64, 4 experts top-2, vocab 32) with a weight pattern chosen so the fp8
// dequant is EXACT: the byte 0x38 + (i % 8) decodes to 1 + (i % 8) / 8,
// which needs three mantissa bits and is therefore exactly representable in
// bf16, and the block scale is 1.0. The bf16 fixture carries the SAME
// values as bf16, so the two arms hold byte-identical weights.

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/kolibri1_forward.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/ops.h"

namespace {

using vllm::HfConfig;
using vllm::Kolibri1Weights;
using vllm::LoadKolibri1Weights;
using vllm::SafetensorsFile;
using vllm::ForwardKolibri1Forward;
using vllm::PagedKvCache;
using vllm::MultiKvCacheIndex;

// The logits band for the same-bytes claim. The two arms hold byte-identical
// weights and ride the same compute class, so the measured gap is expected
// to be exactly zero; the band is a hairline, not a tolerance to widen.
constexpr double kLogitBand = 1e-6;

// ── The fixture ─────────────────────────────────────────────────────────────

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
           ("vllm_kolibri1_bf16_" + std::to_string(nonce) + "_" +
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

// The weight VALUE of element i is 1 + (i % 8) / 8, exactly representable in
// bf16 (three mantissa bits) and in f32.
//
// The fp8 byte for that value: 0x38 + (i % 8) is e4m3 with exponent 7 and
// mantissa (i % 8), so it decodes to 2^0 * (1 + (i % 8) / 8).
uint8_t WeightByte(int64_t i) {
  return static_cast<uint8_t>(0x38 + (i % 8));
}

// The bf16 bits of that value: 0x3F80 is 1.0, and the mantissa field of
// 1 + k / 8 is k << 4.
uint16_t WeightBf16(int64_t i) {
  return static_cast<uint16_t>(0x3F80 + ((i % 8) << 4));
}

struct TinyShape {
  int64_t hidden = 64;
  int64_t vocab = 32;
  int64_t heads = 4;
  int64_t kv_heads = 2;
  int64_t head_dim = 16;
  int64_t experts = 4;
  int64_t inter = 32;
};

// The fp8 arm's projection: the F8_E4M3 bytes plus the F32 scale grid (on
// disk BF16, widened by the loader), exactly the checkpoint's layout.
void AppendFp8Projection(std::vector<FixtureTensor>& out,
                          const std::string& proj, int64_t n, int64_t k) {
  const int64_t numel = n * k;
  std::vector<uint8_t> packed(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i)
    packed[static_cast<size_t>(i)] = WeightByte(i);
  out.push_back({proj + ".weight", "F8_E4M3", {n, k}, packed});
  const std::vector<int64_t> sshape = {CDiv(n, 128), CDiv(k, 128)};
  out.push_back({proj + ".weight_scale_inv", "BF16", sshape,
                 Bf16Filled(sshape, 0x3F80)});  // 1.0
}

// The bf16 arm's projection: the SAME values, as bf16, and NO scale sibling
// (a bf16 checkpoint has no scales).
void AppendBf16Projection(std::vector<FixtureTensor>& out,
                           const std::string& proj, int64_t n, int64_t k) {
  const int64_t numel = n * k;
  std::vector<uint16_t> values(static_cast<size_t>(numel));
  for (int64_t i = 0; i < numel; ++i)
    values[static_cast<size_t>(i)] = WeightBf16(i);
  out.push_back({proj + ".weight", "BF16", {n, k}, Bytes16(values)});
}

HfConfig MakeTinyConfig(bool fp8_scheme) {
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
  if (fp8_scheme) {
    nlohmann::json quant;
    quant["quant_method"] = "fp8";
    quant["activation_scheme"] = "dynamic";
    quant["weight_block_size"] = nlohmann::json::array({128, 128});
    j["quantization_config"] = quant;
  }
  config.raw = j;
  return config;
}

// The shared (arm-independent) tensors: the norms, the embedding, the
// lm_head, the router gate (bf16 in BOTH arms: it is a
// modules_to_not_convert module) and the router bias.
std::vector<FixtureTensor> SharedTensors(const TinyShape& s) {
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
    t.push_back({base + "mlp.gate.weight", "BF16", {s.experts, s.hidden},
                 Bf16Filled({s.experts, s.hidden}, 0x3F80)});
    t.push_back({base + "moe.router.expert_bias", "BF16", {s.experts},
                 Bf16Filled({s.experts}, 0x3F80)});
  }
  return t;
}

std::vector<FixtureTensor> Fp8Fixture(const TinyShape& s = {}) {
  std::vector<FixtureTensor> t = SharedTensors(s);
  for (int64_t l = 0; l < 2; ++l) {
    const std::string base = "model.layers." + std::to_string(l) + ".";
    AppendFp8Projection(t, base + "self_attn.q_proj", s.heads * s.head_dim,
                        s.hidden);
    AppendFp8Projection(t, base + "self_attn.k_proj", s.kv_heads * s.head_dim,
                        s.hidden);
    AppendFp8Projection(t, base + "self_attn.v_proj", s.kv_heads * s.head_dim,
                        s.hidden);
    AppendFp8Projection(t, base + "self_attn.o_proj", s.hidden,
                        s.heads * s.head_dim);
    for (int64_t e = 0; e < s.experts; ++e) {
      const std::string expert = base + "mlp.experts." + std::to_string(e);
      AppendFp8Projection(t, expert + ".gate_proj", s.inter, s.hidden);
      AppendFp8Projection(t, expert + ".up_proj", s.inter, s.hidden);
      AppendFp8Projection(t, expert + ".down_proj", s.hidden, s.inter);
    }
    AppendFp8Projection(t, base + "mlp.shared_experts.gate_proj", s.inter,
                        s.hidden);
    AppendFp8Projection(t, base + "mlp.shared_experts.up_proj", s.inter,
                        s.hidden);
    AppendFp8Projection(t, base + "mlp.shared_experts.down_proj", s.hidden,
                        s.inter);
  }
  return t;
}

std::vector<FixtureTensor> Bf16Fixture(const TinyShape& s = {}) {
  std::vector<FixtureTensor> t = SharedTensors(s);
  for (int64_t l = 0; l < 2; ++l) {
    const std::string base = "model.layers." + std::to_string(l) + ".";
    AppendBf16Projection(t, base + "self_attn.q_proj", s.heads * s.head_dim,
                         s.hidden);
    AppendBf16Projection(t, base + "self_attn.k_proj", s.kv_heads * s.head_dim,
                         s.hidden);
    AppendBf16Projection(t, base + "self_attn.v_proj", s.kv_heads * s.head_dim,
                         s.hidden);
    AppendBf16Projection(t, base + "self_attn.o_proj", s.hidden,
                         s.heads * s.head_dim);
    for (int64_t e = 0; e < s.experts; ++e) {
      const std::string expert = base + "mlp.experts." + std::to_string(e);
      AppendBf16Projection(t, expert + ".gate_proj", s.inter, s.hidden);
      AppendBf16Projection(t, expert + ".up_proj", s.inter, s.hidden);
      AppendBf16Projection(t, expert + ".down_proj", s.hidden, s.inter);
    }
    AppendBf16Projection(t, base + "mlp.shared_experts.gate_proj", s.inter,
                         s.hidden);
    AppendBf16Projection(t, base + "mlp.shared_experts.up_proj", s.inter,
                         s.hidden);
    AppendBf16Projection(t, base + "mlp.shared_experts.down_proj", s.hidden,
                         s.inter);
  }
  return t;
}

Kolibri1Weights Load(const TempCheckpoint& ckpt, const HfConfig& config) {
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  return LoadKolibri1Weights(shards, config);
}

std::string LoadFailure(const TempCheckpoint& ckpt, const HfConfig& config) {
  try {
    const Kolibri1Weights w = Load(ckpt, config);
    (void)w;
    return "";
  } catch (const std::exception& e) {
    return e.what();
  }
}

bool Names(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

// ── The forward harness (the W2/W3 tiny-model shape) ────────────────────────

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

  explicit Topology(const Kolibri1Weights& w) {
    const int64_t layers = w.params.num_hidden_layers;
    const int64_t kv_heads = w.params.num_key_value_heads;
    const int64_t head_dim = w.params.head_dim;
    int32_t paged_slot = 0;
    for (int64_t l = 0; l < layers; ++l) {
      if (w.params.IsSlidingLayer(l)) continue;
      names.push_back("model.layers." + std::to_string(l) + ".self_attn");
      group_ids.push_back(0);
      layer_indices.push_back(static_cast<int32_t>(l));
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
      payload_slots.push_back(paged_slot++);
    }
    for (int64_t l = 0; l < layers; ++l) {
      if (!w.params.IsSlidingLayer(l)) continue;
      names.push_back("model.layers." + std::to_string(l) + ".self_attn");
      group_ids.push_back(1);
      layer_indices.push_back(static_cast<int32_t>(l));
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
      payload_slots.push_back(paged_slot++);
    }
    const int64_t elt = static_cast<int64_t>(vt::SizeOf(dtype));
    for (size_t i = 0; i < names.size(); ++i) {
      const int64_t page_bytes =
          kNumBlocks * kBlockSize * kv_heads * head_dim * 2 * elt;
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
};

// One forward over `tokens` (positions 0..n-1); returns the logits rows.
std::vector<float> RunForward(const Kolibri1Weights& w,
                              const std::vector<int32_t>& tokens) {
  const int64_t T = static_cast<int64_t>(tokens.size());
  std::vector<int32_t> positions(static_cast<size_t>(T));
  std::vector<int64_t> slot_mapping(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) {
    positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
    slot_mapping[static_cast<size_t>(i)] =
        static_cast<int64_t>(kBlockPerm[static_cast<size_t>(i / kBlockSize)] *
                                 kBlockSize +
                             i % kBlockSize);
  }
  std::vector<int32_t> query_start_loc{0, static_cast<int32_t>(T)};
  std::vector<int32_t> seq_lens{static_cast<int32_t>(T)};

  vllm::v1::CommonAttentionMetadata meta;
  meta.num_reqs = 1;
  meta.num_actual_tokens = static_cast<int>(T);
  meta.max_query_len = static_cast<int>(T);
  meta.max_seq_len = static_cast<int>(T);
  meta.query_start_loc = query_start_loc;
  meta.query_start_loc_cpu = query_start_loc;
  meta.seq_lens = seq_lens;
  meta.seq_lens_cpu = seq_lens;
  meta.causal = true;
  meta.block_table_num_cols = 8;
  for (int64_t b = 0; b < kNumBlocks; ++b)
    meta.block_table_tensor.push_back(static_cast<int32_t>(kBlockPerm[b]));
  meta.slot_mapping = slot_mapping;

  Topology topo(w);
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vllm::ForwardLogits fl = ForwardKolibri1Forward(
      tokens, positions, meta, topo.attn_kv, w, &topo.mk, queue, {});
  std::vector<float> out(static_cast<size_t>(fl.rows) *
                         static_cast<size_t>(fl.vocab));
  vt::Backend& be = vt::GetBackend(queue.device.type);
  be.Copy(queue, out.data(), fl.device_tensor.data,
          out.size() * sizeof(float));
  be.Synchronize(queue);
  return out;
}

int32_t Argmax(const std::vector<float>& row, int64_t vocab, int64_t r) {
  const auto begin = row.begin() + static_cast<ptrdiff_t>(r * vocab);
  const auto end = begin + static_cast<ptrdiff_t>(vocab);
  return static_cast<int32_t>(std::distance(begin, std::max_element(begin, end)));
}

}  // namespace

// ═══ TESTS ═══════════════════════════════════════════════════════════════════

TEST_CASE("kolibri1 bf16 arm: a plain bf16 checkpoint loads and the CPU "
          "forward produces logits") {
  const TempCheckpoint ckpt(Bf16Fixture());
  const HfConfig config = MakeTinyConfig(/*fp8_scheme=*/false);

  const Kolibri1Weights w = Load(ckpt, config);
  REQUIRE(w.layers.size() == 2);
  CHECK(w.params.head_dim == 16);
  CHECK(w.layers[0].is_sliding);
  CHECK_FALSE(w.layers[1].is_sliding);

  // Every linear projection took the bf16 arm: no fp8 block, no scale grid.
  const vllm::Kolibri1Projection& q = w.layers[0].attn.q_proj;
  CHECK(q.IsBf16());
  CHECK_FALSE(q.IsFp8Block());
  CHECK(q.bf16.dtype == vt::DType::kBF16);
  REQUIRE(q.bf16.shape[0] == 64);
  REQUIRE(q.bf16.shape[1] == 64);
  // The bf16 bytes are the fixture's, verbatim.
  REQUIRE(q.bf16.bytes.size() == 64u * 64u * 2u);
  for (int64_t i = 0; i < 64 * 64; ++i) {
    const uint16_t got = reinterpret_cast<const uint16_t*>(
        q.bf16.bytes.data())[i];
    CHECK(got == WeightBf16(i));
  }
  // The experts and the shared expert are bf16 too.
  for (const auto& e : w.layers[0].moe.experts) {
    CHECK(e.gate_proj.IsBf16());
    CHECK(e.up_proj.IsBf16());
    CHECK(e.down_proj.IsBf16());
  }
  CHECK(w.layers[0].moe.shared_experts.gate_proj.IsBf16());
  CHECK(w.layers[0].moe.shared_experts.gate_proj.bf16.shape[0] == 32);

  // The forward runs and produces a logits row.
  const std::vector<float> rows = RunForward(w, {1, 2, 3});
  REQUIRE(rows.size() == 3u * 32u);
  for (const float v : rows) CHECK(std::isfinite(v));
  CHECK(std::any_of(rows.begin(), rows.end(), [](float v) { return v != 0.0F; }));
}

TEST_CASE("kolibri1 bf16 arm: the fp8 checkpoint still loads exactly as before") {
  const TempCheckpoint ckpt(Fp8Fixture());
  const HfConfig config = MakeTinyConfig(/*fp8_scheme=*/true);

  const Kolibri1Weights w = Load(ckpt, config);
  REQUIRE(w.layers.size() == 2);

  // The attention projections took the fp8-block arm, with the f32-widened
  // scale grid — unchanged.
  const vllm::Fp8BlockWeight& q = w.layers[0].attn.q_proj.fp8_block;
  CHECK_FALSE(q.Empty());
  CHECK(w.layers[0].attn.q_proj.bf16.Empty());
  CHECK(q.n == 64);
  CHECK(q.k == 64);
  CHECK(q.block_n == 128);
  CHECK(q.block_k == 128);
  CHECK(q.packed.dtype == vt::DType::kI8);
  CHECK(q.scale.dtype == vt::DType::kF32);
  REQUIRE(q.scale.rank == 2);
  CHECK(q.scale.shape[0] == 1);
  CHECK(q.scale.shape[1] == 1);
  CHECK(reinterpret_cast<const float*>(q.scale.bytes.data())[0] == 1.0F);

  // The fp8 bytes are copied verbatim.
  std::vector<uint8_t> expect(64 * 64);
  for (size_t i = 0; i < expect.size(); ++i)
    expect[i] = WeightByte(static_cast<int64_t>(i));
  REQUIRE(q.packed.bytes.size() == expect.size());
  CHECK(std::memcmp(q.packed.bytes.data(), expect.data(), expect.size()) == 0);

  // The router gate stays bf16 in the fp8 arm.
  CHECK(w.layers[0].moe.router_gate.dtype == vt::DType::kBF16);
  CHECK(w.layers[0].moe.e_score_correction_bias.dtype == vt::DType::kF32);

  // The experts are fp8-block.
  REQUIRE(w.layers[0].moe.experts.size() == 4);
  for (const auto& e : w.layers[0].moe.experts) {
    CHECK(e.gate_proj.IsFp8Block());
    CHECK(e.up_proj.IsFp8Block());
    CHECK(e.down_proj.IsFp8Block());
  }
  CHECK(w.layers[0].moe.shared_experts.gate_proj.IsFp8Block());
}

TEST_CASE("kolibri1 bf16 arm: a scheme that is neither fp8-block nor bf16 is "
          "refused by name") {
  // The config names a scheme the row does not implement.
  HfConfig config = MakeTinyConfig(/*fp8_scheme=*/true);
  config.raw["quantization_config"]["quant_method"] = "gptq";
  const TempCheckpoint ckpt(Fp8Fixture());
  const std::string message = LoadFailure(ckpt, config);
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "gptq"));
  // The message names the arms that DO exist, so the reader knows what is
  // missing.
  CHECK(Names(message, "fp8"));
  CHECK(Names(message, "bf16"));
}

TEST_CASE("kolibri1 bf16 arm: a bf16 checkpoint that ships a scale sibling is "
          "refused, and the fp8 arm that loses one is still refused") {
  // A bf16 checkpoint with a scale sibling: the accounting pass must refuse
  // it by name (the bf16 arm expects NO scales).
  std::vector<FixtureTensor> mixed = Bf16Fixture();
  mixed.push_back({"model.layers.0.self_attn.q_proj.weight_scale_inv", "BF16",
                   {1, 1}, Bf16Filled({1, 1}, 0x3F80)});
  const TempCheckpoint mixed_ckpt(mixed);
  const std::string message =
      LoadFailure(mixed_ckpt, MakeTinyConfig(/*fp8_scheme=*/false));
  REQUIRE_FALSE(message.empty());
  CHECK(Names(message, "weight_scale_inv"));
  CHECK(Names(message, "q_proj"));

  // The fp8 arm that loses its scale grid is still refused by name.
  std::vector<FixtureTensor> fp8 = Fp8Fixture();
  std::vector<FixtureTensor> stripped;
  for (auto& t : fp8) {
    if (t.name == "model.layers.0.self_attn.q_proj.weight_scale_inv") continue;
    stripped.push_back(std::move(t));
  }
  const TempCheckpoint stripped_ckpt(stripped);
  const std::string fp8_message =
      LoadFailure(stripped_ckpt, MakeTinyConfig(/*fp8_scheme=*/true));
  REQUIRE_FALSE(fp8_message.empty());
  CHECK(Names(fp8_message, "weight_scale_inv"));
  CHECK(Names(fp8_message, "q_proj"));
}

TEST_CASE("kolibri1 bf16 arm: the fp8 and the bf16 arms agree on the same "
          "bytes") {
  // The two fixtures hold the SAME weight VALUES: the fp8 one as F8_E4M3
  // blocks with a 1.0 scale grid, the bf16 one as the dequantized bf16. The
  // bytes are identical after the load, so the two arms must produce the
  // same result — the same compute class multiplies the same bytes.
  const TempCheckpoint fp8_ckpt(Fp8Fixture());
  const TempCheckpoint bf16_ckpt(Bf16Fixture());
  const Kolibri1Weights fp8 = Load(fp8_ckpt, MakeTinyConfig(true));
  const Kolibri1Weights bf16 = Load(bf16_ckpt, MakeTinyConfig(false));

  // The loaded bytes really are the same: the fp8 arm's decoded weight and
  // the bf16 arm's on-disk weight.
  const vllm::Kolibri1Projection& a = fp8.layers[0].attn.q_proj;
  const vllm::Kolibri1Projection& b = bf16.layers[0].attn.q_proj;
  REQUIRE(a.fp8_block.packed.bytes.size() == 64u * 64u);
  REQUIRE(b.bf16.bytes.size() == 64u * 64u * 2u);
  for (int64_t i = 0; i < 64 * 64; ++i) {
    // The fp8 byte decodes to the bf16 value (the scale is 1.0 and the
    // value needs three mantissa bits, so the decode is exact).
    const uint8_t byte =
        a.fp8_block.packed.bytes.data()[static_cast<size_t>(i)];
    CHECK(byte == WeightByte(i));
    CHECK(reinterpret_cast<const uint16_t*>(b.bf16.bytes.data())[i] ==
          WeightBf16(i));
  }

  // The prefill shape (t == 3) and the decode shape (t == 1) both.
  for (const std::vector<int32_t>& tokens : {std::vector<int32_t>{1, 2, 3},
                                             std::vector<int32_t>{5}}) {
    const std::vector<float> got_fp8 = RunForward(fp8, tokens);
    const std::vector<float> got_bf16 = RunForward(bf16, tokens);
    REQUIRE(got_fp8.size() == got_bf16.size());
    REQUIRE(!got_fp8.empty());
    double worst = 0.0;
    for (size_t i = 0; i < got_fp8.size(); ++i) {
      const double diff =
          std::fabs(static_cast<double>(got_fp8[i]) - got_bf16[i]);
      worst = std::max(worst, diff);
    }
    MESSAGE("t = " << tokens.size() << ": worst fp8-vs-bf16 logit gap " << worst);
    CHECK(worst <= kLogitBand);
    // The greedy token must agree on every row: the bf16 arm must not be
    // WORSE than the fp8 arm.
    const int64_t vocab = 32;
    const int64_t rows = static_cast<int64_t>(got_fp8.size()) / vocab;
    for (int64_t r = 0; r < rows; ++r) {
      CHECK(Argmax(got_fp8, vocab, r) == Argmax(got_bf16, vocab, r));
    }
  }
}
