// Kolibri-1 W3 gate test (MODEL-TEXT-kolibri-1, spec
// .agents/specs/kolibri-1-cpu.md).
//
// THE ROW'S TOKEN GATE: our CPU forward (W2's, via the real W1 fp8-block
// loader on the REAL checkpoint) runs greedy decode over the same 8 prompts
// the transformers golden run used (tests/vllm/models/kolibri1_goldens.json,
// produced by scripts/gen-kolibri1-goldens.py from the pinned plugin's math
// transcribed to pure torch over the dequantized bf16 reference), and the
// final-step logits are compared against the golden fingerprints.
//
// Tokenization scope: the engine cannot yet tokenize the kolibri1
// pre-tokenizer regex (the W1 recorded gap), so the gate feeds the HF token
// ids from the golden file DIRECTLY as input ids — the honest comparison is
// FORWARD outputs for FIXED ids, which is what a parity gate needs.
//
// Tolerance derivation: both sides carry bf16 activations and bf16-stored
// weights (the goldens read the bf16 dequantized reference; ours dequants
// the fp8 blocks to the same values in bf16), but the accumulation orders
// differ (oneDNN GEMM here vs torch GEMM there), so logits differ by
// bf16-rounding noise. The W2 measurement of the same effect on the scalar
// reference gave max abs logits gap 0.014; the band here is set at
// kLogitBand = 0.35, dominated by the much longer 50-layer/37-token
// accumulation chains. The TOKEN bar is strict: the greedy argmax chain
// (8 prompts x 32 positions = 256 tokens) must match the golden generated
// ids, with near-tie adjudication — a flipped position is reported and
// tolerated only when the golden top-1/top-2 gap is under the band (the two
// candidates are numerically indistinguishable at bf16 noise).

#include <doctest/doctest.h>

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <set>
#include <fstream>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/kolibri1_forward.h"
#include "vllm/v1/kv_cache_interface.h"

#ifndef KOLIBRI1_GOLDENS
#define KOLIBRI1_GOLDENS "tests/vllm/models/kolibri1_goldens.json"
#endif

namespace {

constexpr const char* kRealModelDir = "/mnt/models/Aleph-Alpha/Kolibri-1";
// See the header comment: bf16-rounding noise band, not a derived constant.
constexpr double kLogitBand = 0.35;

using vllm::HfConfig;
using vllm::Kolibri1Weights;
using vllm::LoadKolibri1Weights;
using vllm::SafetensorsFile;
using vllm::ForwardKolibri1Forward;
using Logits = std::vector<float>;

// ── The W2 production-side harness (the mimo_v2 W2 pattern), unchanged ──────
constexpr int32_t kBlockPerm[8] = {5, 2, 7, 1, 6, 0, 3, 4};
constexpr int64_t kBlockSize = 16;
constexpr int64_t kNumBlocks = 8;

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
      if (w.params.IsSlidingLayer(l)) continue;  // the FULL group first
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

  static int32_t Slot(int64_t pos) {
    return kBlockPerm[pos / kBlockSize] * kBlockSize + pos % kBlockSize;
  }
};

// One prefill step for a single sequence at position 0; returns the LAST
// position's logits row (length vocab).
Logits PrefillLastRow(const Kolibri1Weights& w, const HfConfig& config,
                      const std::vector<int32_t>& tokens) {
  (void)config;  // the forward takes the weights struct; the config is
                 // only the registry resolution input
  const int64_t T = static_cast<int64_t>(tokens.size());
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> query_start_loc{0, static_cast<int32_t>(T)};
  std::vector<int64_t> slot_mapping(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i) slot_mapping[static_cast<size_t>(i)] = Topology::Slot(i);
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
  meta.block_table_tensor.insert(meta.block_table_tensor.end(), kBlockPerm,
                                 kBlockPerm + 8);
  meta.slot_mapping = slot_mapping;

  Topology topo(w);
  vllm::v1::GDNAttentionMetadata gdn{};
  std::vector<vllm::GdnStateCache> gdn_state;
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

struct GoldenPrompt {
  std::string prompt;
  std::vector<int32_t> input_ids;
  std::vector<int32_t> generated_ids;
  std::vector<int32_t> topk_ids;
  std::vector<double> topk_logits;
  double logits_sum = 0.0;
  double abs_sum = 0.0;
};

nlohmann::json LoadGoldens() {
  std::ifstream in(KOLIBRI1_GOLDENS);
  REQUIRE_MESSAGE(in, "goldens not found at " << KOLIBRI1_GOLDENS);
  return nlohmann::json::parse(std::string(
      (std::istreambuf_iterator<char>(in)), std::istreambuf_iterator<char>()));
}

}  // namespace

// W3 gate 0: the golden file's shape is sane without any weights loaded.
TEST_CASE("kolibri1 W3: the golden file has the declared prompt set") {
  const nlohmann::json g = LoadGoldens();
  REQUIRE(g.contains("prompts"));
  const auto& prompts = g.at("prompts");
  REQUIRE(prompts.size() == 8);
  int german = 0;
  for (const auto& p : prompts) {
    CHECK(p.at("input_ids").size() > 0);
    CHECK(p.at("generated_ids").size() == 32);
    CHECK(p.at("final_logits").at("topk_ids").size() == 8);
    for (const auto& id : p.at("input_ids")) {
      CHECK(id >= 0);
      CHECK(id < 128000);
    }
    for (const auto& id : p.at("generated_ids")) {
      CHECK(id >= 0);
      CHECK(id < 128000);
    }
    const std::string text = p.at("prompt").get<std::string>();
    if (text.find("Mond") != std::string::npos ||
        text.find("German") != std::string::npos ||
        text.find("Buch") != std::string::npos)
      ++german;
  }
  CHECK(german >= 2);  // the model is English-German; the set covers it
}

// W3 THE TOKEN GATE. Real fp8 checkpoint -> W1 loader -> W2 CPU forward ->
// greedy decode -> compare against the transformers golden fingerprints.
TEST_CASE("kolibri1 W3: CPU forward reproduces the transformers golden run") {
  if (!std::filesystem::exists(std::string(kRealModelDir) +
                               "/model.safetensors.index.json")) {
    MESSAGE("SKIP: " << kRealModelDir << " not mounted");
    return;
  }
  const nlohmann::json g = LoadGoldens();

  // The REAL config through the REAL parser (not the W1 hand-built copy).
  const HfConfig config =
      vllm::LoadHfConfig(std::string(kRealModelDir) + "/config.json");

  // The real loader over the real 32 shards.
  const auto index = nlohmann::json::parse(
      std::ifstream(std::string(kRealModelDir) + "/model.safetensors.index.json"));
  // ONE open per DISTINCT shard: iterating the weight_map itself would try
  // 116303 opens (one per tensor) and blow the fd limit.
  std::set<std::string> shard_names;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)name;
    shard_names.insert(shard.get<std::string>());
  }
  std::vector<SafetensorsFile> shards;
  for (const std::string& shard : shard_names)
    shards.push_back(SafetensorsFile::Open(std::string(kRealModelDir) + "/" + shard));
  MESSAGE("loading the real fp8 checkpoint...");
  const Kolibri1Weights w = LoadKolibri1Weights(shards, config);
  shards.clear();
  MESSAGE("weights loaded");

  int64_t argmax_matches = 0;
  int64_t argmax_total = 0;
  int64_t near_ties = 0;
  int64_t flips = 0;
  double worst_topk = 0.0;
  double worst_sum_gap = 0.0;

  for (const auto& p : g.at("prompts")) {
    GoldenPrompt gp;
    gp.prompt = p.at("prompt").get<std::string>();
    for (const auto& v : p.at("input_ids"))
      gp.input_ids.push_back(v.get<int32_t>());
    for (const auto& v : p.at("generated_ids"))
      gp.generated_ids.push_back(v.get<int32_t>());
    for (const auto& v : p.at("final_logits").at("topk_ids"))
      gp.topk_ids.push_back(v.get<int32_t>());
    for (const auto& v : p.at("final_logits").at("topk_logits"))
      gp.topk_logits.push_back(v.get<double>());
    gp.logits_sum = p.at("final_logits").at("logits_sum").get<double>();
    gp.abs_sum = p.at("final_logits").at("abs_sum").get<double>();

    // Greedy decode, one full prefill per step (identical math to incremental
    // decoding; the goldens do the same).
    std::vector<int32_t> ids = gp.input_ids;
    for (int32_t step = 0; step < 32; ++step) {
      const Logits row = PrefillLastRow(w, config, ids);
      REQUIRE(row.size() == 128000u);
      const int32_t got = static_cast<int32_t>(
          std::max_element(row.begin(), row.end()) - row.begin());
      const int32_t want = gp.generated_ids[static_cast<size_t>(step)];
      ++argmax_total;
      if (got == want) {
        ++argmax_matches;
      } else {
        ++flips;
        // Near-tie adjudication: the golden top-2 gap at this step is not
        // recorded (only the final step is fingerprinted), so a flip is a
        // FAIL unless the flipped logits are within the band of each other.
        const float top1 = row[static_cast<size_t>(want)];
        float second = -std::numeric_limits<float>::infinity();
        for (float v : row)
          if (v > second) second = v;
        MESSAGE("flip at prompt '" << gp.prompt << "' step " << step << ": got "
                                   << got << " want " << want << " (band gap "
                                   << second - top1 << ")");
        if (second - top1 <= kLogitBand) ++near_ties;
      }
      ids.push_back(got);
    }

    // Final-step fingerprint comparison.
    const Logits row = PrefillLastRow(w, config, ids);
    double sum = 0.0, abs_sum = 0.0;
    for (float v : row) {
      sum += v;
      abs_sum += std::fabs(v);
    }
    worst_sum_gap = std::max(worst_sum_gap, std::fabs(sum - gp.logits_sum));
    worst_sum_gap = std::max(worst_sum_gap, std::fabs(abs_sum - gp.abs_sum));
    // The golden top-8 ids, order-insensitive set, each within the band of
    // the corresponding golden logit value.
    for (size_t k = 0; k < gp.topk_ids.size(); ++k) {
      const float got = row[static_cast<size_t>(gp.topk_ids[k])];
      const double diff = std::fabs(static_cast<double>(got) - gp.topk_logits[k]);
      worst_topk = std::max(worst_topk, diff);
      CHECK_MESSAGE(diff <= kLogitBand,
                    "topk logit " << k << " of '" << gp.prompt << "' off by "
                                  << diff);
    }
    MESSAGE("'" << gp.prompt << "': worst topk diff so far " << worst_topk);
  }

  // THE GATE VERDICT, quoted in full.
  MESSAGE("ARGMAX CHAIN: " << argmax_matches << "/" << argmax_total
                           << " positions match the golden greedy decode ("
                           << flips << " flips, " << near_ties
                           << " within the " << kLogitBand << " near-tie band)");
  MESSAGE("FINAL-STEP FINGERPRINT: worst topk logit diff " << worst_topk
                                                           << ", worst sum diff "
                                                           << worst_sum_gap);
  CHECK(argmax_matches == argmax_total);
  CHECK(worst_topk <= kLogitBand);
}
