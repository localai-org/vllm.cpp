// Kolibri-1 three-way numerics capture harness (row/kolibri-three-way).
//
// The primary oracle (the aleph-alpha-inference plugin via vLLM, BFMMLA-class)
// published its per-position top-100 distributions over the 8-prompt battery
// (tests/vllm/models/kolibri1_primary_topk_rows.json, teacher-forced via
// prompt_logprobs=100). This harness teacher-forces OUR CPU forward on the
// SAME chains — prefill the prompt ids, then feed the primary's generated ids
// one at a time — and dumps the FULL f32 logit row (128,000 vocab) at each
// position whose distribution predicted a primary generated token. That is
// the aligned comparison: both sides score position k under the identical
// prefix prompt + gen[0..k-1], so a logit-level distance between the rows is
// a numerics distance, not a chain-divergence artifact.
//
// INERT without the dump directory: the whole body is gated on
// VT_THREEWAY_DUMP_DIR being set, so the default test run is a no-op skip.
// The numerics MODE under test is chosen entirely by environment (the knobs
// are read per forward inside the kernels), so the SAME binary captures the
// production f32 mode (VT_CPU_PAGED_ATTN_NEON unset/0) and the FMA-contract
// draft mode (VT_CPU_PAGED_ATTN_NEON=1 VT_CPU_KOLIBRI_BFDOT_GEMV=1, on the
// row/kolibri-moe-glue-neon tree that carries the knobs).
//
// Output per run: <dir>/manifest.json (model geometry, per-prompt chain and
// row count) and <dir>/prompt<k>.bin — flat little-endian f32 rows,
// rows_total x 128000, prompt-major in fixture order. The analysis script
// (scripts/kolibri1-threeway-analysis.py) reads both.

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
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/kolibri1_forward.h"
#include "vllm/v1/kv_cache_interface.h"

#ifndef KOLIBRI1_PRIMARY_TOPK_ROWS
#define KOLIBRI1_PRIMARY_TOPK_ROWS "tests/vllm/models/kolibri1_primary_topk_rows.json"
#endif

namespace {

using vllm::HfConfig;
using vllm::Kolibri1Weights;
using vllm::LoadKolibri1Weights;
using vllm::SafetensorsFile;
using vllm::ForwardKolibri1Forward;
using Logits = std::vector<float>;

// The checkpoint the capture runs on. The DEFAULT is the released fp8
// checkpoint; `VT_KOLIBRI1_MODEL_DIR` points the SAME instrument at another
// Kolibri-1 checkpoint — the plain bf16 arm (spec
// .agents/specs/kolibri1-bf16-arm.md) measures its own arm with this same
// capture, so the two arms' numbers come from one instrument and one
// fixture.
std::string RealModelDir() {
  const char* env = std::getenv("VT_KOLIBRI1_MODEL_DIR");
  if (env != nullptr && env[0] != '\0') return std::string(env);
  return std::string("/mnt/models/Aleph-Alpha/Kolibri-1");
}
constexpr int64_t kBlockSize = 16;
constexpr int64_t kNumBlocks = 32;  // identity table, 512 slots: prompt + 32 gen

// The bench's persistent-topology pattern (test_kolibri1_decode_bench.cpp),
// sized up to 32 blocks so the longest battery prompt plus its 32 generated
// tokens fit the identity block table.
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

// One forward over `tokens` appended into the persistent topology at
// positions start_pos..; returns the LAST position's logits row. The bench's
// StepLastRow, unchanged except for the block-table extent.
Logits StepLastRow(const Kolibri1Weights& w,
                   const std::vector<int32_t>& tokens, int64_t start_pos,
                   Topology& topo) {
  const int64_t T = static_cast<int64_t>(tokens.size());
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    positions[static_cast<size_t>(i)] = static_cast<int32_t>(start_pos + i);
  std::vector<int32_t> query_start_loc{0, static_cast<int32_t>(T)};
  std::vector<int64_t> slot_mapping(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    slot_mapping[static_cast<size_t>(i)] = (start_pos + i);  // identity slots
  std::vector<int32_t> seq_lens{static_cast<int32_t>(start_pos + T)};

  vllm::v1::CommonAttentionMetadata meta;
  meta.num_reqs = 1;
  meta.num_actual_tokens = static_cast<int>(T);
  meta.max_query_len = static_cast<int>(T);
  meta.max_seq_len = static_cast<int>(start_pos + T);
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
  Logits row(static_cast<size_t>(fl.vocab));
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

// The capture case: teacher-force the PRIMARY chains, dump per-position rows.
TEST_CASE("kolibri1 three-way capture: teacher-forced per-position logits dump") {
  const char* dump_dir_c = std::getenv("VT_THREEWAY_DUMP_DIR");
  if (dump_dir_c == nullptr || dump_dir_c[0] == '\0') {
    MESSAGE("SKIP: VT_THREEWAY_DUMP_DIR not set (capture is opt-in)");
    return;
  }
  const std::filesystem::path dump_dir(dump_dir_c);
  std::filesystem::create_directories(dump_dir);

  const nlohmann::json fixture = nlohmann::json::parse(
      std::ifstream(KOLIBRI1_PRIMARY_TOPK_ROWS));
  REQUIRE(fixture.contains("prompts"));

  const std::string model_dir = RealModelDir();
  const HfConfig config =
      vllm::LoadHfConfig(model_dir + "/config.json");
  const auto index = nlohmann::json::parse(std::ifstream(
      model_dir + "/model.safetensors.index.json"));
  std::set<std::string> shard_names;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)name;
    shard_names.insert(shard.get<std::string>());
  }
  std::vector<SafetensorsFile> shards;
  for (const std::string& shard : shard_names)
    shards.push_back(
        SafetensorsFile::Open(model_dir + "/" + shard));
  MESSAGE("loading the real fp8 checkpoint...");
  const Kolibri1Weights w = LoadKolibri1Weights(shards, config);
  shards.clear();

  Topology topo(w);

  nlohmann::json manifest;
  manifest["model_dir"] = model_dir;
  manifest["fixture"] = KOLIBRI1_PRIMARY_TOPK_ROWS;
  manifest["vocab"] = 128000;
  manifest["prompts"] = nlohmann::json::array();

  size_t rows_total = 0;
  std::ofstream bin;
  for (size_t pi = 0; pi < fixture.at("prompts").size(); ++pi) {
    const auto& p = fixture.at("prompts")[pi];
    std::vector<int32_t> ids;
    for (const auto& v : p.at("input_ids"))
      ids.push_back(v.get<int32_t>());
    std::vector<int32_t> gen;
    for (const auto& v : p.at("generated_ids"))
      gen.push_back(v.get<int32_t>());
    // Scored gen_indices (the primary dropped positions whose emitted token
    // fell outside its own top-100): rows[k].gen_index marks the true step.
    std::vector<int> scored;
    for (const auto& r : p.at("rows"))
      scored.push_back(r.at("gen_index").get<int>());
    std::vector<bool> want(gen.size(), false);
    for (int g : scored) {
      if (g < 0 || g >= static_cast<int>(gen.size()))
        FAIL("scored gen_index out of range");
      want[static_cast<size_t>(g)] = true;
    }
    // Walk the FULL primary chain: prefill predicts gen[0]; feeding gen[k]
    // predicts gen[k+1]. Dump only the scored positions.
    std::vector<Logits> rows;
    std::vector<int> rows_gen_index;
    {
      Logits row = StepLastRow(w, ids, 0, topo);
      if (want[0]) {
        rows.push_back(std::move(row));
        rows_gen_index.push_back(0);
      }
    }
    for (size_t k = 0; k + 1 < gen.size(); ++k) {
      const int64_t pos = static_cast<int64_t>(ids.size());
      ids.push_back(gen[k]);
      Logits row = StepLastRow(w, {gen[k]}, pos, topo);
      if (want[k + 1]) {
        rows.push_back(std::move(row));
        rows_gen_index.push_back(static_cast<int>(k + 1));
      }
    }
    REQUIRE_MESSAGE(rows.size() == scored.size(), "dumped row count must match the fixture scored count");

    // The fixture may skip a generated position when the primary's emitted
    // token fell outside its top-100 (rows[k].gen_index marks the true
    // step). Record the alignment so the analysis joins on gen_index, and
    // feed gen_index-ordered tokens: the chain the model must follow is the
    // primary's FULL generated sequence, not only the scored subset.
    nlohmann::json pj;
    pj["prompt"] = p.at("prompt");
    pj["input_ids"] = p.at("input_ids");
    nlohmann::json gi = nlohmann::json::array();
    for (int g : rows_gen_index) gi.push_back(g);
    pj["gen_index"] = gi;  // scored[k] -> the k-th dumped row predicts gen[g]
    pj["generated_ids"] = p.at("generated_ids");
    const std::string bin_name =
        "prompt" + std::to_string(pi) + ".bin";
    bin.open(dump_dir / bin_name, std::ios::binary);
    REQUIRE(bin);
    for (const Logits& row : rows) {
      REQUIRE(row.size() == 128000u);
      bin.write(reinterpret_cast<const char*>(row.data()),
                static_cast<std::streamsize>(row.size() * sizeof(float)));
      ++rows_total;
    }
    bin.close();
    pj["bin"] = bin_name;
    pj["rows_dumped"] = rows.size();
    manifest["prompts"].push_back(pj);
    MESSAGE("prompt " << pi << ": dumped " << rows.size() << " rows");
  }
  manifest["rows_total"] = rows_total;
  std::ofstream mf(dump_dir / "manifest.json");
  mf << manifest.dump(2) << "\n";
  MESSAGE("wrote " << (dump_dir / "manifest.json").c_str() << " (" << rows_total
                   << " rows)");
}
