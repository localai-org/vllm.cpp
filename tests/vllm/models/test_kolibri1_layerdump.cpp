// SCRATCH MEASUREMENT HARNESS (the f32-residual A/B; NOT for main).
//
// Runs ONE teacher-forced 6-token prefill through the production CPU forward,
// twice: once with the residual stream in the model dtype (bf16, the shipped
// arm) and once with the residual stream in f32 (VT_KOLIBRI1_F32_RESIDUAL=1).
// The per-layer quantities are dumped by the env-gated instrumentation in
// kolibri1_forward.cpp (VT_KOLIBRI1_LAYER_DUMP=<file>).
//
// The token ids are the primary oracle's own prompt-0 ids
// (tests/vllm/models/kolibri1_primary_topk_rows.json), the same ids the
// llama.cpp and the mistral.rs layer dumps used, so the three dumps are
// comparable elementwise.
//
// Inert without VT_KOLIBRI1_LAYER_DUMP_DIR.
#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/kolibri1_forward.h"
#include "vllm/v1/kv_cache_interface.h"

namespace {

using vllm::HfConfig;
using vllm::Kolibri1Weights;
using vllm::LoadKolibri1Weights;
using vllm::SafetensorsFile;
using vllm::ForwardKolibri1Forward;

constexpr const char* kRealModelDir = "/mnt/models/Aleph-Alpha/Kolibri-1";
constexpr int64_t kBlockSize = 16;
constexpr int64_t kNumBlocks = 32;

struct Topology {
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
  int64_t layers = 50;
  int64_t kv_heads = 4;
  int64_t head_dim = 128;

  explicit Topology(const Kolibri1Weights& w) {
    layers = w.params.num_hidden_layers;
    kv_heads = w.params.num_key_value_heads;
    head_dim = w.params.head_dim;
    int32_t paged_slot = 0;
    for (int l = 0; l < static_cast<int>(layers); ++l) {
      if (w.params.IsSlidingLayer(l)) continue;
      names.push_back("model.layers." + std::to_string(l) + ".self_attn");
      group_ids.push_back(0);
      layer_indices.push_back(l);
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
      payload_slots.push_back(paged_slot++);
    }
    for (int l = 0; l < static_cast<int>(layers); ++l) {
      if (!w.params.IsSlidingLayer(l)) continue;
      names.push_back("model.layers." + std::to_string(l) + ".self_attn");
      group_ids.push_back(1);
      layer_indices.push_back(l);
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
      payload_slots.push_back(paged_slot++);
    }
    const int64_t elt = static_cast<int64_t>(vt::SizeOf(dtype));
    for (size_t i = 0; i < names.size(); ++i) {
      const int64_t page_bytes =
          kNumBlocks * kBlockSize * kv_heads * head_dim * 2 * elt;
      attn_bytes.emplace_back(static_cast<size_t>(page_bytes), 0);
      vllm::PagedKvCache kv;
      kv.data = attn_bytes.back().data();
      kv.dtype = dtype;
      kv.num_blocks = kNumBlocks;
      kv.block_size = kBlockSize;
      kv.num_kv_heads = kv_heads;
      kv.head_size = head_dim;
      attn_kv.push_back(kv);
    }
    group_bt.assign(1, std::vector<int32_t>());
    for (int64_t b = 0; b < kNumBlocks; ++b)
      group_bt.back().push_back(static_cast<int32_t>(b));
    group_cols.push_back(static_cast<int32_t>(kNumBlocks));
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

// One prefill of `tokens` at positions 0..T-1 into a FRESH topology; returns
// the last row's logits (the caller only needs the forward to have run).
std::vector<float> Prefill(const Kolibri1Weights& w,
                           const std::vector<int32_t>& tokens, Topology& topo) {
  const int64_t T = static_cast<int64_t>(tokens.size());
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> query_start_loc{0, static_cast<int32_t>(T)};
  std::vector<int64_t> slot_mapping(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) slot_mapping[static_cast<size_t>(i)] = i;
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
  meta.block_table_num_cols = static_cast<int>(kNumBlocks);
  for (int64_t b = 0; b < kNumBlocks; ++b)
    meta.block_table_tensor.push_back(static_cast<int32_t>(b));
  meta.slot_mapping = slot_mapping;

  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vllm::ForwardLogits fl = ForwardKolibri1Forward(
      tokens, positions, meta, topo.attn_kv, w, &topo.mk, queue, {});
  std::vector<float> row(static_cast<size_t>(fl.vocab));
  vt::Backend& be = vt::GetBackend(queue.device.type);
  const int64_t last = (fl.rows - 1) * fl.vocab;
  be.Copy(queue, row.data(),
          static_cast<const uint8_t*>(fl.device_tensor.data) +
              last * static_cast<int64_t>(sizeof(float)),
          static_cast<size_t>(fl.vocab) * sizeof(float));
  be.Synchronize(queue);
  return row;
}

}  // namespace

TEST_CASE("kolibri1 layer dump A/B: bf16 residual against f32 residual") {
  const char* dir_c = std::getenv("VT_KOLIBRI1_LAYER_DUMP_DIR");
  if (dir_c == nullptr || dir_c[0] == '\0') {
    MESSAGE("SKIP: VT_KOLIBRI1_LAYER_DUMP_DIR not set (scratch dump is opt-in)");
    return;
  }
  const std::filesystem::path dir(dir_c);
  std::filesystem::create_directories(dir);

  // The primary oracle's prompt-0 ids ("alpha beta gamma delta epsilon zeta").
  const std::vector<int32_t> prompt = {5249, 25079, 74493, 15774, 121598, 116026};

  const HfConfig config =
      vllm::LoadHfConfig(std::string(kRealModelDir) + "/config.json");
  const auto index = nlohmann::json::parse(std::ifstream(
      std::string(kRealModelDir) + "/model.safetensors.index.json"));
  std::set<std::string> shard_names;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)name;
    shard_names.insert(shard.get<std::string>());
  }
  std::vector<SafetensorsFile> shards;
  for (const std::string& shard : shard_names)
    shards.push_back(
        SafetensorsFile::Open(std::string(kRealModelDir) + "/" + shard));
  MESSAGE("loading the real fp8 checkpoint...");
  const Kolibri1Weights w = LoadKolibri1Weights(shards, config);
  shards.clear();

  // ARM A — the shipped arm: the residual stream in the model dtype (bf16).
  unsetenv("VT_KOLIBRI1_F32_RESIDUAL");
  setenv("VT_KOLIBRI1_LAYER_DUMP", (dir / "bf16.tsv").c_str(), 1);
  {
    Topology topo_a(w);
    const std::vector<float> row = Prefill(w, prompt, topo_a);
    REQUIRE(row.size() == 128000u);
    MESSAGE("arm A (bf16 residual) done");
  }
  // ARM B — the lever arm: the residual stream in f32.
  setenv("VT_KOLIBRI1_F32_RESIDUAL", "1", 1);
  setenv("VT_KOLIBRI1_LAYER_DUMP", (dir / "f32.tsv").c_str(), 1);
  {
    Topology topo_b(w);
    const std::vector<float> row = Prefill(w, prompt, topo_b);
    REQUIRE(row.size() == 128000u);
    MESSAGE("arm B (f32 residual) done");
  }
  unsetenv("VT_KOLIBRI1_F32_RESIDUAL");
}
