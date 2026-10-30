// vllm.cpp ORIGINAL — the flint GGUF arm's gate.
// (.agents/specs/flint-gguf-arm.md, row MODEL-TEXT-FLINT wave 2, issue
// ISSUE-LOCAL-01M5PRSC2E4VR54HBSWFPMN37E.)
//
// RED FIRST: at the base of this branch the published container was refused
// by the top-level GGUF dispatch as "not supported by this build" and the
// registry's own loader refused GGUF by name (the W1 refusal,
// src/vllm/model_executor/models/flint.cpp LoadFlintForCausalLM); the
// published Q4_K_M is refused before this row, it loads after.
//
// THE TOKEN GATE: golden greedy chains captured from the PINNED
// llama-cpp-flint oracle (PR #30928 head 1b44e30775dcb2f93952116b05f89e5bd797fb00,
// scripts/gen-flint-goldens.py) on the published Q4_K_M container, replayed
// through the PRODUCTION entry point — ModelRegistry::Resolve, ::Load with
// ModelSource::FromGguf, ::Forward — never by constructing FlintWeights by
// hand. Band: a divergent step is adjudicated by OUR second-best logit gap
// (the kolibri1 contract verbatim); the band is 2.5 nats, the measured
// envelope W1's synthetic gate and the kolibri1 W3 gate use, not re-derived.
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>
#include <doctest/doctest.h>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/model_executor/models/flint.h"
#include "vllm/model_executor/models/flint_gguf_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"

#ifndef FLINT_GOLDENS
#define FLINT_GOLDENS "tests/vllm/models/flint_goldens_q4_k_m.json"
#endif

#ifndef FLINT_GGUF_PATH
#define FLINT_GGUF_PATH "/mnt/models/flint-GGUF/flint-Q4_K_M.gguf"
#endif

namespace {

constexpr const char* kPublishedGguf = FLINT_GGUF_PATH;
// The house band (test_kolibri1_w3.cpp / the W1 synthetic gate), NOT
// re-derived here.
constexpr double kLogitBand = 2.5;

using vllm::HfConfig;
using Logits = std::vector<float>;

nlohmann::json LoadGoldens() {
  std::ifstream in(FLINT_GOLDENS);
  REQUIRE_MESSAGE(in, "goldens not found at " << FLINT_GOLDENS);
  return nlohmann::json::parse(
      std::string((std::istreambuf_iterator<char>(in)),
                  std::istreambuf_iterator<char>()));
}

}  // namespace

TEST_CASE("flint gguf: the config builder maps the file's own metadata") {
  if (!std::filesystem::exists(kPublishedGguf)) {
    MESSAGE("SKIP: " << kPublishedGguf << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(kPublishedGguf);
  REQUIRE(vllm::IsFlintGguf(gguf));
  const HfConfig config = vllm::FlintHfConfigFromGguf(gguf);
  // ParseFlintParams is the ONE validator; the builder's mapping must
  // survive it, including the layer schedule read off the file's own
  // sliding_window_pattern.
  const vllm::FlintParams p = vllm::ParseFlintParams(config);
  CHECK(p.hidden_size == 2048);
  CHECK(p.num_hidden_layers == 49);
  CHECK(p.num_attention_heads == 32);
  CHECK(p.num_key_value_heads == 4);
  CHECK(p.head_dim == 128);
  CHECK(p.sliding_window == 513);
  CHECK(p.vocab_size == 150144);
  CHECK(p.num_experts == 256);
  CHECK(p.num_experts_per_tok == 6);
  CHECK(p.moe_intermediate_size == 512);
  CHECK(p.shared_expert_intermediate_size == 1024);
  CHECK(p.mlp_only_layers == std::vector<int64_t>{0, 1});
  CHECK(p.LayerHasRope(0) == false);
  // The file's own pattern, read off the container header (2026-10-24):
  // 14 full-attention layers at {0, 1, 48} ∪ {l >= 5, l ≡ 1 (mod 4)} — the
  // FFSSSF repeat — 35 sliding. The mapping test pins the FILE, not a
  // re-derived formula.
  for (int64_t l = 0; l < 49; ++l) {
    const bool full = l == 0 || l == 1 || l == 48 || (l >= 5 && l % 4 == 1);
    CHECK(p.IsSlidingLayer(l) == !full);
    CHECK(p.LayerHasRope(l) == p.IsSlidingLayer(l));
  }
}

// The `flint` pre-tokenizer mapping: our Encode must reproduce the ORACLE'S
// OWN prompt ids (the goldens carry them), byte for byte, on every gate
// prompt. A mis-mapped split pattern or vocab would show here.
TEST_CASE("flint gguf: the tokenizer reproduces the oracle's prompt ids") {
  if (!std::filesystem::exists(kPublishedGguf)) {
    MESSAGE("SKIP: " << kPublishedGguf << " not mounted");
    return;
  }
  const nlohmann::json g = LoadGoldens();
  vllm::GgufFile gguf = vllm::GgufFile::Open(kPublishedGguf);
  vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromGguf(gguf);
  for (const auto& p : g.at("prompts")) {
    std::vector<int32_t> want;
    for (const auto& v : p.at("input_ids")) want.push_back(v.get<int32_t>());
    const std::vector<int32_t> got =
        tok.Encode(p.at("prompt").get<std::string>());
    CHECK_MESSAGE(got == want, "prompt '" << p.at("prompt").get<std::string>()
                                          << "'");
  }
}

// THE TOKEN GATE.
TEST_CASE("flint gguf: the published Q4_K_M replays the oracle golden run") {
  if (!std::filesystem::exists(kPublishedGguf)) {
    MESSAGE("SKIP: " << kPublishedGguf << " not mounted");
    return;
  }
  const nlohmann::json g = LoadGoldens();

  vllm::GgufFile gguf = vllm::GgufFile::Open(kPublishedGguf);
  const HfConfig config = vllm::FlintHfConfigFromGguf(gguf);
  MESSAGE("resolving the registry entry...");
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(config);
  REQUIRE(reg.architecture == "FlintForCausalLM");
  MESSAGE("loading the published Q4_K_M container through the registry...");
  vllm::ModelSource source = vllm::ModelSource::FromGguf(
      gguf, vt::DeviceType::kCPU);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);
  MESSAGE("weights loaded");

  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  std::vector<vllm::PagedKvCache> attn_kv;   // the reference forward keeps its own host cache
  std::vector<vllm::GdnStateCache> gdn_state;

  int64_t argmax_matches = 0;
  int64_t argmax_total = 0;
  int64_t near_ties = 0;
  int64_t flips = 0;
  int64_t hard_flips = 0;
  double worst_gap = 0.0;

  for (const auto& p : g.at("prompts")) {
    const std::string prompt = p.at("prompt").get<std::string>();
    std::vector<int32_t> ids;
    for (const auto& v : p.at("input_ids")) ids.push_back(v.get<int32_t>());
    std::vector<int32_t> golden;
    for (const auto& v : p.at("generated_ids")) golden.push_back(v.get<int32_t>());

    // Greedy replay, first divergence adjudicates — the kolibri1 contract.
    bool diverged = false;
    double diverge_gap = 0.0;
    int32_t diverge_step = -1;
    for (int32_t step = 0; step < static_cast<int32_t>(golden.size()) && !diverged; ++step) {
      std::vector<int32_t> positions(static_cast<size_t>(ids.size()));
      for (int64_t i = 0; i < static_cast<int64_t>(ids.size()); ++i)
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
      std::vector<int32_t> logits_indices;
      vllm::v1::CommonAttentionMetadata meta;
      meta.num_reqs = 1;
      meta.num_actual_tokens = static_cast<int>(ids.size());
      meta.max_query_len = static_cast<int>(ids.size());
      meta.max_seq_len = static_cast<int>(ids.size());
      meta.causal = true;
      vllm::v1::GDNAttentionMetadata gdn{};
      vllm::ModelForwardInput input{
          .token_ids = ids,
          .positions = positions,
          .attn_meta = meta,
          .gdn_meta = gdn,
          .attn_kv = attn_kv,
          .gdn_state = gdn_state,
          .config = config,
          .queue = queue,
          .logits_indices = logits_indices,
          .num_reqs = 1,
      };
      vllm::ForwardLogits fl = vllm::ModelRegistry::Forward(*model, input);
      REQUIRE(fl.host.size() ==
              static_cast<size_t>(fl.rows * fl.vocab));
      const int64_t last = (fl.rows - 1) * fl.vocab;
      const int32_t got = static_cast<int32_t>(
          std::max_element(fl.host.begin() + last, fl.host.end()) -
          (fl.host.begin() + last));
      const int32_t want = golden[static_cast<size_t>(step)];
      ++argmax_total;
      if (got == want) {
        ++argmax_matches;
      } else {
        ++flips;
        const float top1 = fl.host[static_cast<size_t>(last + want)];
        float second = -std::numeric_limits<float>::infinity();
        for (int64_t n = last; n < last + fl.vocab; ++n)
          if (fl.host[static_cast<size_t>(n)] > second)
            second = fl.host[static_cast<size_t>(n)];
        diverge_gap = second - top1;
        diverge_step = step;
        worst_gap = std::max(worst_gap, static_cast<double>(diverge_gap));
        diverged = true;
        if (diverge_gap <= kLogitBand) {
          ++near_ties;
        } else {
          ++hard_flips;
          MESSAGE("HARD flip at prompt '" << prompt << "' step " << step
                                          << ": got " << got << " want " << want
                                          << " (gap " << diverge_gap
                                          << " > band " << kLogitBand << ")");
        }
      }
      ids.push_back(got);
    }
    if (diverged)
      MESSAGE("chain '" << prompt << "' diverged at step " << diverge_step
                        << " with gap " << diverge_gap << " (band "
                        << kLogitBand << ") — adjudicated as "
                        << (diverge_gap <= kLogitBand ? "NEAR-TIE"
                                                      : "HARD MISMATCH"));
  }

  MESSAGE("token gate: " << argmax_matches << "/" << argmax_total
                         << " greedy tokens exact, " << flips << " flips ("
                         << near_ties << " near-tie, " << hard_flips
                         << " hard), worst gap " << worst_gap);
  CHECK(hard_flips == 0);
}

// THE KEEP-QUANT PAGED TOKEN GATE (FLINT-KQ-CPU-DISPATCH, spec
// .agents/specs/flint-kq-cpu-dispatch.md). The same llama.cpp oracle goldens,
// but the tree loads with VT_FLINT_KEEP_QUANT=1 and the step enters through
// the PAGED arm (ModelRegistry::Forward with hand-built per-layer KV caches,
// the tensors the LLMEngine runner hands the model) — the arm that refused
// this tree by name before the row. Adjudication is the test_flint_gguf
// contract shape verbatim: llama.cpp uses integer block dots and we may not,
// so token-exactness is NOT expected — every divergence is adjudicated by OUR
// second-best logit gap against the house 2.5-nat band, and the gate is
// 0 hard flips.
namespace {
struct KeepQuantEnv {
  KeepQuantEnv() { setenv("VT_FLINT_KEEP_QUANT", "1", 1); }
  ~KeepQuantEnv() { unsetenv("VT_FLINT_KEEP_QUANT"); }
};
}  // namespace

TEST_CASE("flint keepquant: the paged arm replays the oracle golden in-band") {
  if (!std::filesystem::exists(kPublishedGguf)) {
    MESSAGE("SKIP: " << kPublishedGguf << " not mounted");
    return;
  }
  const nlohmann::json g = LoadGoldens();

  KeepQuantEnv env;
  vllm::GgufFile gguf = vllm::GgufFile::Open(kPublishedGguf);
  const HfConfig config = vllm::FlintHfConfigFromGguf(gguf);
  const vllm::FlintParams p = vllm::ParseFlintParams(config);
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(config);
  REQUIRE(reg.architecture == "FlintForCausalLM");
  vllm::ModelSource source =
      vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kCPU);
  std::unique_ptr<vllm::LoadedModel> model =
      vllm::ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);

  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  // One per-layer KV cache set: [num_blocks, 2, bs, hkv, dh] bf16 pages, the
  // test_flint_paged.cpp KvSet shape at the published container's dims.
  constexpr int64_t kBs = 16;
  constexpr int64_t kNumBlocks = 64;
  const int64_t page =
      2 * kBs * p.num_key_value_heads * p.head_dim *
      static_cast<int64_t>(sizeof(uint16_t));
  std::vector<std::vector<uint8_t>> kv_storage;
  std::vector<vllm::PagedKvCache> attn_kv;
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    kv_storage.emplace_back(static_cast<size_t>(kNumBlocks * page), 0);
    vllm::PagedKvCache kv;
    kv.data = kv_storage.back().data();
    kv.dtype = vt::DType::kBF16;
    kv.num_blocks = kNumBlocks;
    kv.block_size = kBs;
    kv.num_kv_heads = p.num_key_value_heads;
    kv.head_size = p.head_dim;
    attn_kv.push_back(kv);
  }

  int64_t matches = 0;
  int64_t total = 0;
  int64_t near_ties = 0;
  int64_t hard_flips = 0;
  double worst_gap = 0.0;

  for (const auto& pr : g.at("prompts")) {
    const std::string prompt = pr.at("prompt").get<std::string>();
    std::vector<int32_t> ids;
    for (const auto& v : pr.at("input_ids")) ids.push_back(v.get<int32_t>());
    std::vector<int32_t> golden;
    for (const auto& v : pr.at("generated_ids"))
      golden.push_back(v.get<int32_t>());

    bool diverged = false;
    double diverge_gap = 0.0;
    for (int32_t step = 0;
         step < static_cast<int32_t>(golden.size()) && !diverged; ++step) {
      const int64_t t = static_cast<int64_t>(ids.size());
      std::vector<int32_t> positions(static_cast<size_t>(t));
      for (int64_t i = 0; i < t; ++i)
        positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);
      vllm::v1::CommonAttentionMetadata meta;
      meta.num_reqs = 1;
      meta.num_actual_tokens = static_cast<int>(t);
      meta.max_query_len = static_cast<int>(t);
      meta.max_seq_len = static_cast<int>(t);
      meta.seq_lens = {static_cast<int32_t>(t)};
      meta.seq_lens_cpu = meta.seq_lens;
      meta.num_computed_tokens_cpu = {0};
      meta.query_start_loc = {0, static_cast<int32_t>(t)};
      meta.query_start_loc_cpu = meta.query_start_loc;
      meta.causal = true;
      meta.block_table_num_cols = static_cast<int32_t>(kNumBlocks);
      for (int32_t b = 0; b < static_cast<int32_t>(kNumBlocks); ++b)
        meta.block_table_tensor.push_back(b);
      meta.slot_mapping.reserve(static_cast<size_t>(t));
      for (int64_t i = 0; i < t; ++i)
        meta.slot_mapping.push_back(i);
      vllm::v1::GDNAttentionMetadata gdn{};
      std::vector<vllm::GdnStateCache> empty_gdn;
      vllm::ModelForwardInput input{
          .token_ids = ids,
          .positions = positions,
          .attn_meta = meta,
          .gdn_meta = gdn,
          .attn_kv = attn_kv,
          .gdn_state = empty_gdn,
          .config = config,
          .queue = queue,
          .logits_indices = {},
          .num_reqs = 1,
      };
      vllm::ForwardLogits fl = vllm::ModelRegistry::Forward(*model, input);
      // The paged arm returns DEVICE logits (the sampler contract); download.
      std::vector<float> host(static_cast<size_t>(fl.rows * fl.vocab));
      {
        vt::Backend& be = vt::GetBackend(queue.device.type);
        be.Synchronize(queue);
        be.Copy(queue, host.data(), fl.device_tensor.data,
                host.size() * sizeof(float));
      }
      const int64_t last = (fl.rows - 1) * fl.vocab;
      const int32_t got = static_cast<int32_t>(
          std::max_element(host.begin() + last, host.end()) -
          (host.begin() + last));
      const int32_t want = golden[static_cast<size_t>(step)];
      ++total;
      if (got == want) {
        ++matches;
      } else {
        const float top1 = host[static_cast<size_t>(last + want)];
        float second = -std::numeric_limits<float>::infinity();
        for (int64_t n = last; n < last + fl.vocab; ++n)
          if (host[static_cast<size_t>(n)] > second)
            second = host[static_cast<size_t>(n)];
        diverge_gap = static_cast<double>(second - top1);
        worst_gap = std::max(worst_gap, diverge_gap);
        diverged = true;
        if (diverge_gap <= kLogitBand) {
          ++near_ties;
        } else {
          ++hard_flips;
          MESSAGE("HARD flip at prompt '" << prompt << "' step " << step
                                          << ": got " << got << " want " << want
                                          << " (gap " << diverge_gap
                                          << " > band " << kLogitBand << ")");
        }
      }
      ids.push_back(got);
    }
    if (diverged)
      MESSAGE("chain '" << prompt << "' diverged with gap " << diverge_gap
                        << " — adjudicated as "
                        << (diverge_gap <= kLogitBand ? "NEAR-TIE"
                                                      : "HARD MISMATCH"));
  }

  MESSAGE("keepquant paged token gate: " << matches << "/" << total
                                         << " greedy tokens exact, "
                                         << near_ties << " near-tie, "
                                         << hard_flips << " hard, worst gap "
                                         << worst_gap);
  CHECK(hard_flips == 0);
}
