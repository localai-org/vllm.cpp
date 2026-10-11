// Kolibri-1 n-gram SPEC-DECODE bench-level PROBE HARNESS (ISSUE-LOCAL-
// 01M4P22PX6BPRQAY1CK20FZQF6, branch row/kolibri-spec-probe).
//
// WHY A BENCH HARNESS AND NOT THE ENGINE: the engine-level path is blocked by
// ONE precise qwen35-shaped seam — LoadedEngine::MakeKVCacheMaybeSpec
// (src/vllm/entrypoints/model_loader.cpp:1756-1763) builds
// MakeQwen3_5KVCacheSpec for ANY speculative config, ignoring the model's own
// make_kv_cache. For kolibri that injects a GDN/MambaSpec group + fa_draft
// group the model does not have, and the runner refuses at
// runner.cpp:1074 ("MambaSpec state shapes must be positive"). The spec-OFF
// engine loads and runs fine; the spec-ON engine throws at construction.
// The fix (integration plan, NOT this probe): dispatch the widened-spec branch
// on "method needs spec state" (use_eagle()) or "the model's own KV config
// has a recurrent group" — kolibri has none, so its own make_kv_cache is
// already spec-correct for n-gram (the drafts are ordinary scheduled tokens).
//
// WHAT THIS HARNESS IS: the smallest honest demonstration of the mechanism on
// the REAL checkpoint, using only production pieces:
//   - the PRODUCTION proposer: vllm::v1::spec_decode::
//     FindLongestMatchedNgramAndProposeTokens (the landed n-gram matcher);
//   - the PRODUCTION target forward: ForwardKolibri1Forward over the k+1
//     verify sequence [last committed token, draft_0..draft_{k-1}] — the
//     engine's exact verify shape (one causal multi-token forward, the same
//     geometry as a chunked prefill);
//   - the PRODUCTION greedy accept rule: a draft is accepted iff it equals the
//     target argmax at its position; stop at the first mismatch; one
//     bonus/replacement token per verify pass.
// It therefore measures the TRUE accept rate and the TRUE verify-step cost —
// no simulation, no model-in-the-loop shortcuts.
//
// GATES:
//   (i)  EXACTNESS: the spec chain == the plain greedy chain, byte-for-byte,
//        on every prompt (spec decode must not change a single token).
//   (ii) the plain pass reproduces the bench's pinned chain (last token
//        109726) — the harness is the bench's own geometry.
//   (iii) ALIVENESS: drafts proposed and accepted.
// MEASURED: accept rate, tokens per verify pass, decode-phase tok/s of both
// arms at the bench conditions (VLLM_CPP_CPU_THREADS=32,
// VT_KOLIBRI1_DEQUANT_CACHE_MB=32768), and the wall speedup.
//
// Checkpoint-GATED: skips loudly when the real model is absent. CPU only.
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <set>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/kolibri1_forward.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/spec_decode/ngram_proposer.h"

#ifndef KOLIBRI1_GOLDENS
#define KOLIBRI1_GOLDENS "tests/vllm/models/kolibri1_goldens.json"
#endif

namespace {

using vllm::HfConfig;
using vllm::Kolibri1Weights;
using vllm::LoadKolibri1Weights;
using vllm::SafetensorsFile;
using vllm::ForwardKolibri1Forward;
using Logits = std::vector<float>;

constexpr const char* kRealModelDir = "/mnt/models/Aleph-Alpha/Kolibri-1";
constexpr int64_t kBlockSize = 16;
constexpr int64_t kNumBlocks = 16;  // 256 slots: 128-in + 64-out + k drafts
constexpr int kSpecK = 5;           // num_speculative_tokens
constexpr int kSpecMinN = 1;        // prompt_lookup_min
constexpr int kSpecMaxN = 4;        // prompt_lookup_max
constexpr int kMaxModelLen = 262144;
// The bench's token-identity anchor (captured on main 323f81ca6).
constexpr int32_t kAnchorLastToken = 109726;

// The bench's persistent paged topology (identity block table, identity
// slots), unchanged from tests/vllm/models/test_kolibri1_decode_bench.cpp.
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
    for (int g = 0; g < 2; ++g) {
      std::vector<int32_t> bt;
      for (int64_t b = 0; b < kNumBlocks; ++b)
        bt.push_back(static_cast<int32_t>(b));
      group_bt.push_back(bt);
      group_cols.push_back(static_cast<int32_t>(kNumBlocks));
    }
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

// One forward over `tokens` appended at positions start_pos..; returns ALL
// logits rows (one per token) — the verify pass needs every row, the plain
// decode only the last.
std::vector<Logits> StepRows(const Kolibri1Weights& w,
                             const std::vector<int32_t>& tokens,
                             int64_t start_pos, Topology& topo) {
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
  std::vector<Logits> rows(static_cast<size_t>(fl.rows));
  vt::Backend& be = vt::GetBackend(queue.device.type);
  for (int64_t r = 0; r < fl.rows; ++r) {
    rows[static_cast<size_t>(r)].resize(static_cast<size_t>(fl.vocab));
    be.Copy(queue, rows[static_cast<size_t>(r)].data(),
            static_cast<const uint8_t*>(fl.device_tensor.data) +
                r * fl.vocab * static_cast<int64_t>(sizeof(float)),
            static_cast<size_t>(fl.vocab) * sizeof(float));
  }
  be.Synchronize(queue);
  return rows;
}

int32_t Argmax(const Logits& row) {
  return static_cast<int32_t>(
      std::max_element(row.begin(), row.end()) - row.begin());
}

struct SpecRun {
  std::vector<int32_t> chain;   // committed tokens (the full greedy chain)
  double prefill_s = 0.0;
  double decode_s = 0.0;        // verify passes only
  int verifies = 0;             // verify passes after the prefill
  int64_t accepted = 0;
  int64_t proposed = 0;
};

// One greedy request under n-gram spec decode, on the bench topology. The
// verify step is ONE target forward over [last committed, d0..d_{k-1}] —
// the engine's exact verify geometry. The accept walk is the production
// greedy rule (rejection_sampler.h:13-51).
SpecRun RunSpec(const Kolibri1Weights& w, Topology& topo,
                const std::vector<int32_t>& prompt, int max_tokens) {
  SpecRun run;
  std::vector<int32_t> context(prompt);
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<Logits> rows = StepRows(w, context, 0, topo);
  const auto t1 = std::chrono::steady_clock::now();
  run.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  int32_t next = Argmax(rows.back());
  context.push_back(next);
  run.chain.push_back(next);

  const auto d0 = std::chrono::steady_clock::now();
  while (static_cast<int>(run.chain.size()) < max_tokens) {
    // Propose k drafts from the committed context (the production matcher).
    const std::vector<int32_t> drafts =
        vllm::v1::spec_decode::FindLongestMatchedNgramAndProposeTokens(
            context.data(), static_cast<int>(context.size()), kSpecMinN,
            kSpecMaxN, kMaxModelLen, kSpecK);
    run.proposed += static_cast<int64_t>(drafts.size());
    // The verify input: the last committed token + the drafts.
    std::vector<int32_t> input;
    input.reserve(drafts.size() + 1);
    input.push_back(context.back());
    input.insert(input.end(), drafts.begin(), drafts.end());
    const int64_t start_pos = static_cast<int64_t>(context.size()) - 1;
    rows = StepRows(w, input, start_pos, topo);
    // Greedy accept walk over the k+1 rows: row i predicts the token at
    // position start_pos+i+1, compared with input[i+1] (the i-th draft).
    int accepted = 0;
    for (size_t i = 0; i < drafts.size(); ++i) {
      if (Argmax(rows[i]) == input[i + 1]) {
        ++accepted;
      } else {
        break;  // stop at the first mismatch (the sticky-accept rule)
      }
    }
    run.accepted += accepted;
    run.verifies += 1;
    // Commit accepted drafts + the bonus/replacement token (the argmax at
    // the first rejected row, or the last row on a full accept).
    const int emit = std::min(accepted + 1, max_tokens - static_cast<int>(run.chain.size()));
    for (int j = 0; j < emit; ++j) {
      const int32_t tok = (j < accepted) ? input[static_cast<size_t>(j) + 1]
                                         : Argmax(rows[static_cast<size_t>(accepted)]);
      context.push_back(tok);
      run.chain.push_back(tok);
    }
  }
  const auto d1 = std::chrono::steady_clock::now();
  run.decode_s = std::chrono::duration<double>(d1 - d0).count();
  return run;
}

// The plain greedy decode (the bench's own loop), for the exactness gate and
// the baseline timing.
SpecRun RunPlain(const Kolibri1Weights& w, Topology& topo,
                 const std::vector<int32_t>& prompt, int max_tokens) {
  SpecRun run;
  std::vector<int32_t> ids(prompt);
  const auto t0 = std::chrono::steady_clock::now();
  std::vector<Logits> rows = StepRows(w, ids, 0, topo);
  const auto t1 = std::chrono::steady_clock::now();
  run.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  int32_t next = Argmax(rows.back());
  run.chain.push_back(next);
  const auto d0 = std::chrono::steady_clock::now();
  while (static_cast<int>(run.chain.size()) < max_tokens) {
    const int64_t pos = static_cast<int64_t>(ids.size());
    ids.push_back(next);
    rows = StepRows(w, {next}, pos, topo);
    next = Argmax(rows.back());
    run.chain.push_back(next);
    run.verifies += 1;  // one decode step per token
  }
  const auto d1 = std::chrono::steady_clock::now();
  run.decode_s = std::chrono::duration<double>(d1 - d0).count();
  return run;
}

struct Loaded {
  HfConfig config;
  Kolibri1Weights w;
  std::unique_ptr<Topology> topo;
};

Loaded LoadModel() {
  Loaded l;
  l.config = vllm::LoadHfConfig(std::string(kRealModelDir) + "/config.json");
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
  l.w = LoadKolibri1Weights(shards, l.config);
  shards.clear();
  l.topo = std::make_unique<Topology>(l.w);
  return l;
}

void ReportRun(const char* label, const SpecRun& r) {
  const double tok_s =
      r.decode_s > 0.0
          ? static_cast<double>(r.chain.size() - 1) / r.decode_s
          : 0.0;
  MESSAGE(label << ": chain " << r.chain.size() << " tokens in "
                << r.verifies << " steps, decode " << r.decode_s << " s -> "
                << tok_s << " tok/s; drafts accepted " << r.accepted << "/"
                << r.proposed
                << (r.proposed > 0
                        ? " (accept rate " +
                              std::to_string(static_cast<double>(r.accepted) /
                                             r.proposed) +
                              ")"
                        : " (no drafts proposed)"));
}

}  // namespace

TEST_CASE("kolibri1 spec probe: the deterministic bench prompt (plain vs ngram spec)") {
  if (!std::filesystem::exists(std::string(kRealModelDir) +
                               "/model.safetensors.index.json")) {
    MESSAGE("SKIP: " << kRealModelDir << " not mounted");
    return;
  }
  MESSAGE("loading the real fp8 checkpoint...");
  Loaded l = LoadModel();

  std::vector<int32_t> prompt;
  for (int64_t i = 0; i < 128; ++i)
    prompt.push_back(static_cast<int32_t>((i * 7919 + 13) % 128000));
  constexpr int kMaxTokens = 64;

  // Warmup (untimed): the dequant-cache cold misses concentrate in the first
  // forwards; both timed arms below pay them equally otherwise.
  RunPlain(l.w, *l.topo, prompt, /*max_tokens=*/8);

  const SpecRun plain = RunPlain(l.w, *l.topo, prompt, kMaxTokens);
  const SpecRun spec = RunSpec(l.w, *l.topo, prompt, kMaxTokens);

  // GATE (ii): the plain pass reproduces the bench's pinned chain.
  REQUIRE(plain.chain.size() == static_cast<size_t>(kMaxTokens));
  CHECK(plain.chain.back() == kAnchorLastToken);

  // GATE (i): exactness — the spec chain is byte-identical to plain greedy.
  REQUIRE(spec.chain.size() == static_cast<size_t>(kMaxTokens));
  if (spec.chain != plain.chain) {
    size_t shared = 0;
    while (shared < spec.chain.size() && shared < plain.chain.size() &&
           spec.chain[shared] == plain.chain[shared]) {
      ++shared;
    }
    MESSAGE("DIVERGENCE at position " << shared);
  }
  CHECK(spec.chain == plain.chain);

  // GATE (iii): the drafter is alive.
  CHECK(spec.proposed > 0);
  CHECK(spec.accepted > 0);

  ReportRun("bench plain", plain);
  ReportRun("bench spec ", spec);
  const double speedup =
      plain.decode_s > 0.0 ? plain.decode_s / spec.decode_s : 0.0;
  const double spec_tok_s =
      spec.decode_s > 0.0
          ? static_cast<double>(spec.chain.size() - 1) / spec.decode_s
          : 0.0;
  MESSAGE("bench decode speedup spec vs plain: " << speedup
                                                  << "x; spec decode tok/s "
                                                  << spec_tok_s
                                                  << " (baseline 5.76, "
                                                     "target 12)");
}

TEST_CASE("kolibri1 spec probe: the W3 golden prompts, measured through the real forward") {
  if (!std::filesystem::exists(std::string(kRealModelDir) +
                               "/model.safetensors.index.json")) {
    MESSAGE("SKIP: " << kRealModelDir << " not mounted");
    return;
  }
  std::ifstream gf(KOLIBRI1_GOLDENS);
  if (!gf) {
    MESSAGE("SKIP: KOLIBRI1_GOLDENS not readable: " << KOLIBRI1_GOLDENS);
    return;
  }
  const nlohmann::json g = nlohmann::json::parse(gf);
  MESSAGE("loading the real fp8 checkpoint...");
  Loaded l = LoadModel();

  constexpr int kMaxTokens = 32;
  int exact = 0;
  int64_t accepted = 0, proposed = 0, verifies = 0;
  double plain_decode_s = 0.0, spec_decode_s = 0.0;
  int golden_match_full = 0;
  for (size_t i = 0; i < g.at("prompts").size(); ++i) {
    const auto& p = g.at("prompts")[i];
    const std::vector<int32_t> prompt =
        p.at("input_ids").get<std::vector<int32_t>>();
    const std::vector<int32_t> golden =
        p.at("generated_ids").get<std::vector<int32_t>>();

    const SpecRun plain = RunPlain(l.w, *l.topo, prompt, kMaxTokens);
    const SpecRun spec = RunSpec(l.w, *l.topo, prompt, kMaxTokens);
    // GATE (i): exactness on every prompt.
    if (spec.chain == plain.chain) {
      ++exact;
    } else {
      MESSAGE("DIVERGENCE on golden[" << i << "]");
    }
    CHECK(spec.chain == plain.chain);
    accepted += spec.accepted;
    proposed += spec.proposed;
    verifies += spec.verifies;
    plain_decode_s += plain.decode_s;
    spec_decode_s += spec.decode_s;
    // REPORT ONLY: the plain chain vs the torch golden (the W3 relationship;
    // near-tie flips are adjudicated by the W3 gate, not here).
    if (plain.chain == golden) ++golden_match_full;
    size_t shared = 0;
    while (shared < plain.chain.size() && shared < golden.size() &&
           plain.chain[shared] == golden[shared]) {
      ++shared;
    }
    MESSAGE("golden[" << i << "] \"" << p.at("prompt").get<std::string>()
                      << "\": steps " << spec.verifies << "/"
                      << (plain.verifies) << " accepted " << spec.accepted
                      << "/" << spec.proposed << " | plain-vs-torch-golden "
                      << "shared " << shared << "/" << golden.size());
  }
  CHECK(exact == static_cast<int>(g.at("prompts").size()));
  CHECK(proposed > 0);
  CHECK(accepted > 0);

  const double plain_tok_s =
      plain_decode_s > 0.0
          ? static_cast<double>(g.at("prompts").size()) * (kMaxTokens - 1) /
                plain_decode_s
          : 0.0;
  const double spec_tok_s =
      spec_decode_s > 0.0
          ? static_cast<double>(g.at("prompts").size()) * (kMaxTokens - 1) /
                spec_decode_s
          : 0.0;
  MESSAGE("goldens: " << exact << "/" << g.at("prompts").size()
                      << " byte-identical spec vs plain; "
                      << golden_match_full << "/" << g.at("prompts").size()
                      << " plain chains token-identical to the torch golden; "
                         "drafts accepted "
                      << accepted << "/" << proposed << " (accept rate "
                      << (proposed > 0
                              ? static_cast<double>(accepted) / proposed
                              : 0.0)
                      << "), " << verifies << " verify passes; decode tok/s "
                      << plain_tok_s << " -> " << spec_tok_s << " (speedup "
                      << (plain_decode_s > 0.0 ? plain_decode_s / spec_decode_s
                                               : 0.0)
                      << "x; baseline 5.76, target 12)");
}
