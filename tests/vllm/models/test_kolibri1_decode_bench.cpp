// Kolibri-1 incremental-decode production-shape bench and regression gate
// (MODEL-TEXT-kolibri-1; spec .agents/specs/kolibri-1-cpu.md).
//
// test_kolibri1_w3 re-prefills the whole chain per step, so it exercises the
// forward but never the production decode geometry: one prefill over the
// 128-token prompt and then t=1 incremental steps through ONE persistent
// paged KV topology. That is the shape whose block table a misconfigured
// harness under-allocated (the 2026-10-07 cpu_paged_attn.cpp block-table
// refusal), and the shape whose dequant-cache corruption the scratch bench
// caught while W3 could not (docs/bench-evidence/
// kolibri1-dequant-cache-negative-20261007.md). This gate pins it:
//   (i)   the run COMPLETES 64 greedy tokens (any future geometry break —
//         block-table sizing, slot mapping, seq-len bookkeeping — fails here);
//   (ii)  the greedy LAST token equals the anchor captured on main
//         323f81ca6 (the token-identity check);
//   (iii) wall/prefill/decode timing is reported as a MESSAGE only —
//         never asserted, machine load may move it.
// With the NEON lane explicitly opted in (`VT_CPU_PAGED_ATTN_NEON=1`) the
// byte-exact anchor is replaced by the near-tie instrument (spec
// .agents/specs/kolibri1-decode-bench-near-tie-instrument.md): both arms run
// in one process, the scalar arm is the reference, and a differing position
// passes only when the NEON token is inside the scalar top-5 AND its
// teacher-forced gap is within the row's 2.5-nat band. Hard flips fail.
// Deterministic and thread-count-stable in OUTPUT: measured identical across
// 8 and 4 threads and across repeated passes in one process
// (docs/bench-evidence/kolibri1-incremental-baseline-20261007.md).
#include <doctest/doctest.h>

#include <algorithm>
#include <chrono>
#include <cmath>
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

namespace {

using vllm::HfConfig;
using vllm::Kolibri1Weights;
using vllm::LoadKolibri1Weights;
using vllm::SafetensorsFile;
using vllm::ForwardKolibri1Forward;
using Logits = std::vector<float>;

constexpr const char* kRealModelDir = "/mnt/models/Aleph-Alpha/Kolibri-1";
constexpr int64_t kBlockSize = 16;
constexpr int64_t kNumBlocks = 16;  // 128-in + 64-out = 192 tokens -> 12 blocks
constexpr int64_t kPromptLen = 128;
// 63 incremental steps after the prefill token: 64 greedy tokens out.
constexpr int kSteps = 63;
// The token-identity anchor: the greedy last token captured on main
// 323f81ca6 (2026-10-07, this bench, 8 and 4 threads).
constexpr int32_t kAnchorLastToken = 109726;

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
    // Identity block table: logical block b -> physical block b, 16 columns.
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

// One forward over `tokens` appended into the persistent topology at
// positions start_pos..; returns the LAST position's logits row.
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

int32_t Argmax(const Logits& row) {
  return static_cast<int32_t>(
      std::max_element(row.begin(), row.end()) - row.begin());
}

// The near-tie instrument band: the row's already-ratified 2.5-nat
// near-tie band (spec .agents/specs/kolibri1-decode-bench-near-tie-instrument.md;
// the same rule W3 enforces). Top-K membership uses K=5 (W3's).
constexpr double kNearTieBandNats = 2.5;
constexpr int kNearTieTopK = 5;
constexpr const char* kNeonEnv = "VT_CPU_PAGED_ATTN_NEON";

// Mirrors vt::cpu::PagedAttnNeonActive (src/vt/cpu/cpu_paged_attn.cpp:126):
// the lane is opt-in via an explicit `=1`; unset or `=0` is the scalar body.
bool NeonLaneRequested() {
#if defined(__aarch64__)
  const char* e = std::getenv(kNeonEnv);
  return e != nullptr && e[0] == '1';
#else
  return false;
#endif
}

struct ChainRun {
  std::vector<int32_t> chain;
  // Per-position next-token logit rows: rows[0] is the prefill row,
  // rows[s] the s-th incremental step's row. Aligned with `chain`.
  std::vector<Logits> rows;
  double prefill_s = 0.0;
  double decode_s = 0.0;
};

// One full 128-in / 64-greedy-out run over the persistent topology, recording
// every per-step logit row so the near-tie instrument can compare arms.
ChainRun RunChain(const Kolibri1Weights& w, Topology& topo,
                  const std::vector<int32_t>& prompt) {
  ChainRun run;
  std::vector<int32_t> ids(prompt);
  const auto t0 = std::chrono::steady_clock::now();
  Logits row = StepLastRow(w, ids, 0, topo);
  const auto t1 = std::chrono::steady_clock::now();
  int32_t next = Argmax(row);
  run.chain.push_back(next);
  run.rows.push_back(std::move(row));
  for (int step = 1; step <= kSteps; ++step) {
    const int64_t pos = static_cast<int64_t>(ids.size());
    ids.push_back(next);
    row = StepLastRow(w, {next}, pos, topo);
    next = Argmax(row);
    run.chain.push_back(next);
    run.rows.push_back(std::move(row));
  }
  const auto t2 = std::chrono::steady_clock::now();
  run.prefill_s = std::chrono::duration<double>(t1 - t0).count();
  run.decode_s = std::chrono::duration<double>(t2 - t1).count();
  return run;
}

// The lane pass TEACHER-FORCED on the reference chain's tokens: inputs are the
// reference tokens (never the lane's own argmax), so the lane's per-step
// logits share the reference prefix exactly. rows[s] aligns with ref chain
// position s. This is the W3 adjudication input.
ChainRun RunTeacherForced(const Kolibri1Weights& w, Topology& topo,
                          const std::vector<int32_t>& prompt,
                          const std::vector<int32_t>& ref_chain) {
  ChainRun run;
  std::vector<int32_t> ids(prompt);
  Logits row = StepLastRow(w, ids, 0, topo);
  run.chain.push_back(Argmax(row));
  run.rows.push_back(std::move(row));
  for (int step = 1; step <= kSteps; ++step) {
    const int64_t pos = static_cast<int64_t>(ids.size());
    const int32_t fed = ref_chain[static_cast<size_t>(step - 1)];
    ids.push_back(fed);
    row = StepLastRow(w, {fed}, pos, topo);
    run.chain.push_back(Argmax(row));
    run.rows.push_back(std::move(row));
  }
  return run;
}

}  // namespace

TEST_CASE("kolibri1 incremental production shape: 128 in, 64 greedy out, anchored last token") {
  if (!std::filesystem::exists(std::string(kRealModelDir) +
                               "/model.safetensors.index.json")) {
    MESSAGE("SKIP: " << kRealModelDir << " not mounted");
    return;  // same skip convention as test_kolibri1_w3
  }
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

  Topology topo(w);

  // Deterministic prompt: (i*7919+13) % 128000, 128 tokens.
  std::vector<int32_t> ids;
  for (int64_t i = 0; i < kPromptLen; ++i)
    ids.push_back(static_cast<int32_t>((i * 7919 + 13) % 128000));

  if (!NeonLaneRequested()) {
    // LANE OFF (the shipped default): the byte-exact anchor gate exactly as
    // it was before the near-tie instrument (spec design point 4).
    const auto t0 = std::chrono::steady_clock::now();
    const Logits prefill_row = StepLastRow(w, ids, 0, topo);
    const auto t1 = std::chrono::steady_clock::now();
    int32_t next = Argmax(prefill_row);
    std::vector<int32_t> chain{next};
    MESSAGE("prefill first token: " << next);

    for (int step = 1; step <= kSteps; ++step) {
      const int64_t pos = static_cast<int64_t>(ids.size());
      ids.push_back(next);
      const Logits row = StepLastRow(w, {next}, pos, topo);
      next = Argmax(row);
      chain.push_back(next);
    }
    const auto t2 = std::chrono::steady_clock::now();
    const double prefill_s = std::chrono::duration<double>(t1 - t0).count();
    const double decode_s = std::chrono::duration<double>(t2 - t1).count();
    const double wall_s = std::chrono::duration<double>(t2 - t0).count();
    // GATE (i): the production shape completes. A block-table, slot-mapping or
    // seq-len regression anywhere in the incremental path aborts or throws
    // before this line.
    REQUIRE_MESSAGE(chain.size() == static_cast<size_t>(kSteps) + 1,
                    "the incremental run stopped at " << chain.size()
                                                      << " of 64 greedy tokens");
    // GATE (ii): the token-identity anchor. This is the check that catches a
    // silent corruption the fingerprints-only W3 gate missed.
    CHECK_EQ(chain.back(), kAnchorLastToken);

    std::string chain_str;
    for (int32_t t : chain) chain_str += std::to_string(t) + ",";
    MESSAGE("CHAIN: " << chain_str);
    MESSAGE("wall " << wall_s << " s, prefill " << prefill_s << " s, decode "
                    << decode_s << " s, tok/s "
                    << (static_cast<double>(kSteps) / decode_s));
    return;
  }

  // LANE ON: the near-tie instrument (spec
  // .agents/specs/kolibri1-decode-bench-near-tie-instrument.md, W3's
  // methodology, cross-arm). The knob is read per forward call
  // (cpu_paged_attn.cpp:134, uncached getenv), so all passes run in ONE
  // process with the env flipped between them; the scalar arm is the
  // in-process reference.
  //
  // Pass 1: the scalar free-running chain — the reference chain (its
  // teacher-forcing on its own argmax IS the free run) and the byte-exact
  // anchor's tokens.
  // Pass 2: the NEON free-running chain — its tokens mark the DIVERGENCES.
  // Pass 3: the NEON arm TEACHER-FORCED on the reference tokens — per-step
  // lane logits under the SAME prefix as the reference. Every divergence is
  // adjudicated under this common prefix; scoring a free-running lane token
  // against a reference row from a different history is meaningless once the
  // chains split.
  MESSAGE("NEON lane opt-in detected (" << kNeonEnv
                                        << "=1): running the near-tie "
                                           "instrument, scalar arm first");
  setenv(kNeonEnv, "0", 1);
  const ChainRun ref = RunChain(w, topo, ids);
  setenv(kNeonEnv, "1", 1);
  const ChainRun neon_free = RunChain(w, topo, ids);
  const ChainRun neon_tf = RunTeacherForced(w, topo, ids, ref.chain);
  setenv(kNeonEnv, "0", 1);  // restore the scalar default for any later case

  REQUIRE_MESSAGE(ref.chain.size() == static_cast<size_t>(kSteps) + 1,
                  "the scalar arm stopped at " << ref.chain.size()
                                               << " of 64 greedy tokens");
  REQUIRE_MESSAGE(neon_free.chain.size() == static_cast<size_t>(kSteps) + 1,
                  "the neon arm stopped at " << neon_free.chain.size()
                                             << " of 64 greedy tokens");
  REQUIRE_MESSAGE(neon_tf.rows.size() == ref.rows.size(),
                  "row counts diverge: " << neon_tf.rows.size() << " vs "
                                         << ref.rows.size());

  // Adjudication, all under the common (reference) prefix. The lane's
  // teacher-forced argmax at each position is the lane's choice for that
  // position; identical passes, otherwise the nats gap to the reference
  // top-1 and reference top-K membership decide (band 2.5, K=5; hard flips
  // fail). The free-running lane chain is recorded for the divergence count.
  std::string ref_str, neon_str;
  size_t flips = 0, hard_flips = 0, free_divergences = 0;
  double worst_gap = 0.0;
  for (size_t pos = 0; pos < ref.chain.size(); ++pos) {
    ref_str += std::to_string(ref.chain[pos]) + ",";
    neon_str += std::to_string(neon_free.chain[pos]) + ",";
    if (neon_free.chain[pos] != ref.chain[pos]) ++free_divergences;
    const Logits& srow = ref.rows[pos];
    const Logits& nrow = neon_tf.rows[pos];
    // The lane's choice for this position is its TEACHER-FORCED argmax under
    // the common prefix. A free-running divergence that vanishes here is
    // prefix amplification, not a lane defect — scoring the free-running
    // token against the reference row would compare different histories.
    const int32_t lane_tok = Argmax(nrow);
    if (lane_tok == ref.chain[pos]) continue;
    ++flips;
    const int32_t s_top = Argmax(srow);
    // Teacher-forced nats gap under the reference distribution: the logits
    // are raw scores, so the gap to the argmax is the log-softmax difference
    // (the common logsumexp cancels) — the same quantity W3's
    // neartie_gap_mnats measures.
    const double gap =
        static_cast<double>(srow[static_cast<size_t>(s_top)]) -
        static_cast<double>(srow[static_cast<size_t>(lane_tok)]);
    worst_gap = std::max(worst_gap, gap);
    // Top-K membership under the reference arm (ties count as inside).
    int above = 0;
    for (float v : srow)
      if (static_cast<double>(v) >
          static_cast<double>(srow[static_cast<size_t>(lane_tok)]))
        ++above;
    const bool in_topk = above < kNearTieTopK;
    if (gap > kNearTieBandNats || !in_topk) {
      ++hard_flips;
      MESSAGE("HARD FLIP at position " << pos << ": reference "
                                       << ref.chain[pos] << " lane "
                                       << lane_tok << " gap " << gap
                                       << " nats, in-reference-top-K="
                                       << in_topk);
    } else {
      MESSAGE("near-tie flip at position " << pos << ": reference "
                                           << ref.chain[pos] << " lane "
                                           << lane_tok << " gap " << gap
                                           << " nats, in reference top-K");
    }
  }
  MESSAGE("SCALAR CHAIN: " << ref_str);
  MESSAGE("NEON CHAIN:   " << neon_str);
  MESSAGE("NEON TEACHER-FORCED CHAIN: ");
  std::string tf_str;
  for (int32_t t : neon_tf.chain) tf_str += std::to_string(t) + ",";
  MESSAGE(tf_str);
  MESSAGE("near-tie instrument: " << flips << " adjudicated lane flips ("
                                  << free_divergences
                                  << " free-running divergences, prefix "
                                     "amplification collapses under teacher "
                                     "forcing), " << hard_flips
                                  << " HARD flips, worst gap "
                                  << worst_gap << " nats (band "
                                  << kNearTieBandNats << ", top-K "
                                  << kNearTieTopK << ")");
  MESSAGE("wall scalar " << (ref.prefill_s + ref.decode_s)
                         << " s, neon free " << (neon_free.prefill_s + neon_free.decode_s)
                         << " s, neon teacher-forced "
                         << (neon_tf.prefill_s + neon_tf.decode_s) << " s");
  CHECK_EQ(hard_flips, static_cast<size_t>(0));
}
