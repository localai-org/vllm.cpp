// Kolibri-1 n-gram SPEC-DECODE engine gate (ISSUE-LOCAL-01M4P22PX6BPRQAY1CK20FZQF6,
// branch row/kolibri-spec-engine).
//
// HISTORY — this file pinned the BLOCKER. The probe
// (docs/bench-evidence/kolibri1-spec-probe-20261011.md §5c) found exactly one
// qwen35-shaped seam on the engine path: LoadedEngine::MakeKVCacheMaybeSpec
// (src/vllm/entrypoints/model_loader.cpp) built vllm::MakeQwen3_5KVCacheSpec
// for ANY speculative config, ignoring the model's own make_kv_cache. For
// kolibri (FullAttentionSpec + SlidingWindowSpec, no recurrent group) that
// injected a GDN/MambaSpec group with empty shapes (the checkpoint carries no
// linear_* keys) plus an unused fa_draft group, and the runner refused at
// construction (runner.cpp:1074 "MambaSpec state shapes must be positive").
// The probe's pinned blocker test asserted that refusal. THE SEAM IS NOW
// FIXED: the widened spec is taken only when the method or the model needs
// spec state (spec->use_eagle(), or the model's own groups carry a
// recurrent/mamba group); a draft-free n-gram config on a model with no
// recurrent group keeps the model's own KV spec — the drafts are ordinary
// scheduled tokens, so the target's groups are the whole cache need and there
// is no draft-proposal group to add (the fa_draft group is consumed only by
// the MTP draft head's own paged KV, runner.cpp:1865-1898, a forward n-gram
// does not have — and that allocation is skipped outright on kolibri's
// multi-cache topology anyway).
//
// THIS FILE NOW PINS THE FIX AT TWO LEVELS:
//
//  (1) THE SEAM UNIT (always runs, no checkpoint): under an ngram spec config
//      MakeKVCacheMaybeSpec returns kolibri's OWN KV spec — two groups,
//      full-attention + sliding-window, no MambaSpec, no fa_draft —
//      byte-identical to the spec-OFF build; and the qwen3.5 arm of the
//      dispatch still widens (the landed 27B ngram gate's spec is untouched).
//      The pre-fix tree produced a third, MambaSpec group with empty shapes
//      for kolibri — this case is red on it (red-first).
//
//  (2) THE ENGINE e2e GATE (checkpoint-gated, CPU): the same decode-bench
//      workload through the FULL engine — spec-OFF vs spec-ON
//      (method=ngram, k=5, window 1-4) over the deterministic bench prompt
//      (128 in, 64 greedy out) and the 8 W3 golden prompts (32 greedy out):
//        (i)  EXACTNESS: the spec-ON chain == the spec-OFF chain
//             byte-for-byte on every prompt;
//        (ii) the spec-OFF bench chain ends at the pinned anchor 109726;
//        (iii) ALIVENESS + ACCEPTANCE: the engine's acceptance counters
//             match the pinned numbers (bench prompt 53/53 — the exact
//             simulation over the torch chain pins 52/53 in
//             test_kolibri1_ngram_accept, the probe harness measured 53/53
//             over this tree's bf16 chain, and the engine runs that chain);
//        (iv) the W3 relationship is UNCHANGED by spec decode: spec-ON ==
//             spec-OFF byte-identically, so the spec-ON chains stand exactly
//             where the spec-OFF chains stand against the torch goldens
//             (the per-prompt shared prefixes are reported);
//        (v)  engine-level decode tok/s of both arms, reported (the probe's
//             14.51 spec / 4.92 plain tok/s are the harness-level reference).
//
// The pinned ACCEPTANCE SIMULATION (no model run) lives in
// tests/vllm/v1/spec_decode/test_kolibri1_ngram_accept.cpp; the bench-level
// PROBE HARNESS (production proposer + production verify forward) lives in
// tests/vllm/models/test_kolibri1_spec_probe.cpp. Both stay green.
//
// Checkpoint-GATED: the e2e case skips loudly when the real model is absent;
// the seam unit runs everywhere. CPU only.
#include <doctest/doctest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/speculative.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

#ifndef KOLIBRI1_GOLDENS
#define KOLIBRI1_GOLDENS "tests/vllm/models/kolibri1_goldens.json"
#endif

namespace {

constexpr const char* kRealModelDir = "/mnt/models/Aleph-Alpha/Kolibri-1";
// The bench's token-identity anchor (captured on main 323f81ca6).
// Re-recorded on the quiet-box run (2026-10-23, load 1.0-1.4, log
// /tmp/ngram-gate-quiet.log): the shipped bf16-dot class (PR #3482) legitimately
// moves the bench chain's last token.
constexpr int32_t kAnchorLastToken = 11622;

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.PostInit();
  return sp;
}

// A LoadedModel that carries a registration and nothing else. The engine
// funnel reads only registration() on the way to the KV geometry (the
// test_deepseek_v4_scaffold.cpp pattern), so this enters the seam without a
// checkpoint.
class ScaffoldModel final : public vllm::LoadedModel {
 public:
  explicit ScaffoldModel(const vllm::ModelRegistration& r) : LoadedModel(r) {}
};

// The REAL checkpoint's config shape, trimmed to the KV geometry (the
// test_kolibri1.cpp MakeKolibri1Config values): hybrid 4-sliding + 1-full
// over 50 layers, 4 KV heads, head_dim 128, window 513.
vllm::HfConfig MakeKolibri1Config() {
  vllm::HfConfig config;
  config.model_type = "kolibri1";
  config.architectures = {"Kolibri1ForCausalLM"};
  config.hidden_size = 2560;
  config.num_hidden_layers = 50;
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
  nlohmann::json kinds = nlohmann::json::array();
  for (int64_t i = 0; i < 50; ++i)
    kinds.push_back(i % 5 == 4 ? "full_attention" : "sliding_attention");
  j["layer_types"] = kinds;
  config.raw = j;
  return config;
}

// A Qwen3.5-shaped hybrid config (the test_loaded_engine_dense.cpp
// MakeDenseConfig geometry): the GDN keys the widened spec reads, with
// layer_types [LA, LA, LA, FA].
vllm::HfConfig MakeQwen35Config() {
  vllm::HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = 128000;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.raw = nlohmann::json::object();
  return c;
}

}  // namespace

// ── (1) THE SEAM UNIT ────────────────────────────────────────────────────────
TEST_CASE("kolibri1 KV seam: ngram spec keeps the model's own groups; qwen3.5 still widens") {
  using vllm::v1::KVCacheSpecKind;
  const vllm::SpeculativeConfig ngram =
      vllm::SpeculativeConfig::ResolveNgram(/*k=*/5, /*min=*/1, /*max=*/4);

  // ── The kolibri arm: the model's OWN spec, byte-identical with and
  //    without the ngram spec config.
  const vllm::HfConfig kcfg = MakeKolibri1Config();
  const vllm::ModelRegistration& kreg = vllm::ModelRegistry::Resolve(kcfg);
  ScaffoldModel kmodel(kreg);

  const vllm::v1::KVCacheConfig kv_off =
      vllm::entrypoints::LoadedEngine::MakeKVCacheMaybeSpec(
          kmodel, kcfg, /*block_size=*/16, /*num_blocks=*/8, std::nullopt);
  const vllm::v1::KVCacheConfig kv_on =
      vllm::entrypoints::LoadedEngine::MakeKVCacheMaybeSpec(
          kmodel, kcfg, /*block_size=*/16, /*num_blocks=*/8, ngram);

  // kolibri's own shape: TWO groups — full attention + sliding window —
  // split 10/40 over the 50 hybrid layers (i%5==4 is full).
  REQUIRE(kv_off.kv_cache_groups.size() == 2);
  CHECK(kv_off.kv_cache_groups[0].kv_cache_spec->kind() ==
        KVCacheSpecKind::kFullAttention);
  CHECK(kv_off.kv_cache_groups[1].kv_cache_spec->kind() ==
        KVCacheSpecKind::kSlidingWindow);
  CHECK(kv_off.kv_cache_groups[0].layer_names.size() == 10);
  CHECK(kv_off.kv_cache_groups[1].layer_names.size() == 40);
  CHECK_FALSE(kv_off.has_mamba_layers());

  // THE SEAM, pinned: under a spec config the spec is the SAME two groups.
  // No injected MambaSpec (the pre-fix tree built MakeQwen3_5KVCacheSpec for
  // any spec config, whose gdn group carries EMPTY shapes for a checkpoint
  // with no linear_* keys, and the runner then refuses at runner.cpp:1074),
  // and no fa_draft group (the n-gram drafts are ordinary scheduled tokens;
  // fa_draft serves only the MTP draft head's own paged KV, a forward the
  // n-gram proposer does not have).
  REQUIRE(kv_on.kv_cache_groups.size() == 2);
  CHECK_FALSE(kv_on.has_mamba_layers());
  for (size_t g = 0; g < 2; ++g) {
    CHECK(kv_on.kv_cache_groups[g].kv_cache_spec->kind() ==
          kv_off.kv_cache_groups[g].kv_cache_spec->kind());
    CHECK(kv_on.kv_cache_groups[g].layer_names ==
          kv_off.kv_cache_groups[g].layer_names);
    CHECK(kv_on.kv_cache_groups[g].kv_cache_spec->page_size_bytes() ==
          kv_off.kv_cache_groups[g].kv_cache_spec->page_size_bytes());
  }

  // ── The qwen3.5 arm: the SAME ngram spec config still widens — the landed
  //    27B ngram gate's KV spec is untouched by the dispatch change.
  const vllm::HfConfig qcfg = MakeQwen35Config();
  const vllm::ModelRegistration& qreg = vllm::ModelRegistry::Resolve(qcfg);
  ScaffoldModel qmodel(qreg);
  const vllm::v1::KVCacheConfig qkv =
      vllm::entrypoints::LoadedEngine::MakeKVCacheMaybeSpec(
          qmodel, qcfg, /*block_size=*/32, /*num_blocks=*/8, ngram);
  // fa + gdn + fa_draft: the widened spec.
  REQUIRE(qkv.kv_cache_groups.size() == 3);
  const auto* gdn = dynamic_cast<const vllm::v1::MambaSpec*>(
      qkv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(gdn != nullptr);
  // conv_kernel_size - 1 + num_spec taps (mamba_utils.py:226): the widened
  // conv row the k+1 verify step rewinds into. conv_dim = 2*(2*8) + 6*8.
  CHECK(gdn->shapes[0] == std::vector<int64_t>{80, 3 + 5});
  CHECK(gdn->num_speculative_blocks == 5);
  // The draft layer's own paged KV: exactly one buffer's worth of names, at
  // index num_hidden_layers (qwen3_5_mtp.py:105-112).
  CHECK(qkv.kv_cache_groups[2].kv_cache_spec->kind() ==
        KVCacheSpecKind::kFullAttention);
  REQUIRE(qkv.kv_cache_groups[2].layer_names.size() == 1);
  CHECK(qkv.kv_cache_groups[2].layer_names[0] ==
        "model.layers.4.self_attn.attn");
}

// ── (2) THE ENGINE e2e GATE ─────────────────────────────────────────────────
TEST_CASE("kolibri1 ngram spec-decode e2e gate: spec-ON == spec-OFF through the engine") {
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

  // The workload: the deterministic decode-bench prompt (128 in, 64 greedy
  // out) plus the 8 W3 golden prompts (32 greedy out each).
  struct Job {
    std::string name;
    std::vector<int32_t> prompt;
    int max_tokens;
  };
  std::vector<Job> jobs;
  {
    std::vector<int32_t> bench;
    for (int64_t i = 0; i < 128; ++i)
      bench.push_back(static_cast<int32_t>((i * 7919 + 13) % 128000));
    jobs.push_back({"bench", bench, 64});
  }
  for (size_t i = 0; i < g.at("prompts").size(); ++i) {
    jobs.push_back({"golden[" + std::to_string(i) + "] \"" +
                        g.at("prompts")[i].at("prompt").get<std::string>() +
                        "\"",
                    g.at("prompts")[i].at("input_ids")
                        .get<std::vector<int32_t>>(),
                    32});
  }

  struct Arm {
    std::vector<std::vector<int32_t>> chains;
    std::vector<int64_t> accepted, proposed;  // per job
    std::vector<double> wall_s, decode_s;     // per job
    // The DECODE numerator per job: the output tokens generated INSIDE the
    // decode interval [first output, finish] — the full count less the tokens
    // already present when the first output arrived. A spec step can emit
    // several tokens at once, and those were generated before the interval
    // opened, so the full count over the interval overstates decode throughput.
    std::vector<int64_t> decode_tokens;       // per job
  };

  const auto run_arm = [&](bool spec_on, Arm& arm) {
    vllm::entrypoints::EngineParams params;
    params.max_num_seqs = 8;
    if (spec_on) {
      params.speculative_config = vllm::ParseSpeculativeConfigJson(
          "{\"method\":\"ngram\",\"num_speculative_tokens\":5,"
          "\"prompt_lookup_min\":1,\"prompt_lookup_max\":4}");
    }
    std::unique_ptr<vllm::entrypoints::LoadedEngine> eng;
    // THE SEAM PROOF: construction SUCCEEDS under the spec config. The
    // pre-fix tree threw "runner: MambaSpec state shapes must be positive"
    // here — the probe's pinned blocker (this file's earlier shape).
    REQUIRE_NOTHROW(eng = vllm::entrypoints::LoadedEngine::FromModelDir(
                         kRealModelDir, params));
    arm.chains.resize(jobs.size());
    arm.accepted.assign(jobs.size(), 0);
    arm.proposed.assign(jobs.size(), 0);
    arm.wall_s.assign(jobs.size(), 0.0);
    arm.decode_s.assign(jobs.size(), 0.0);
    arm.decode_tokens.assign(jobs.size(), 0);
    for (size_t j = 0; j < jobs.size(); ++j) {
      const int64_t acc0 = eng->runner().spec_drafts_accepted();
      const int64_t prop0 = eng->runner().spec_drafts_proposed();
      const auto t0 = std::chrono::steady_clock::now();
      std::optional<std::chrono::steady_clock::time_point> t_first;
      int64_t tokens_at_first = 0;
      eng->engine().add_request("req" + std::to_string(j), jobs[j].prompt,
                                Greedy(jobs[j].max_tokens));
      std::optional<vllm::RequestOutput> final;
      while (eng->engine().has_unfinished_requests()) {
        for (const vllm::RequestOutput& item : eng->engine().step()) {
          if (!t_first.has_value() && !item.outputs.empty() &&
              !item.outputs[0].token_ids.empty()) {
            t_first = std::chrono::steady_clock::now();
            tokens_at_first =
                static_cast<int64_t>(item.outputs[0].token_ids.size());
          }
          if (item.finished) final = item;
        }
      }
      const auto t1 = std::chrono::steady_clock::now();
      REQUIRE(final.has_value());
      REQUIRE(final->outputs.size() == 1);
      arm.chains[j] = final->outputs[0].token_ids;
      REQUIRE(arm.chains[j].size() ==
              static_cast<size_t>(jobs[j].max_tokens));
      arm.wall_s[j] = std::chrono::duration<double>(t1 - t0).count();
      if (t_first.has_value()) {
        arm.decode_s[j] =
            std::chrono::duration<double>(t1 - *t_first).count();
      }
      // The corrected decode numerator: tokens generated inside the decode
      // interval only. The wall numerator keeps the FULL count (below).
      arm.decode_tokens[j] =
          static_cast<int64_t>(arm.chains[j].size()) - tokens_at_first;
      arm.accepted[j] = eng->runner().spec_drafts_accepted() - acc0;
      arm.proposed[j] = eng->runner().spec_drafts_proposed() - prop0;
      MESSAGE((spec_on ? "spec-ON  " : "spec-OFF ") << jobs[j].name
              << ": wall " << arm.wall_s[j] << " s, decode "
              << arm.decode_s[j] << " s, drafts accepted "
              << arm.accepted[j] << "/" << arm.proposed[j]);
    }
  };

  Arm off, on;
  run_arm(/*spec_on=*/false, off);
  // The default engine never drafts.
  for (size_t j = 0; j < jobs.size(); ++j) CHECK(off.proposed[j] == 0);
  run_arm(/*spec_on=*/true, on);

  // (i) EXACTNESS: the spec-ON chain == the spec-OFF chain byte-for-byte on
  // every prompt. This is the hard gate: greedy spec decode is
  // exactness-preserving, and a divergence here would be a real defect in
  // the verify wiring, not a tolerance question.
  int exact = 0;
  for (size_t j = 0; j < jobs.size(); ++j) {
    if (on.chains[j] == off.chains[j]) {
      ++exact;
    } else {
      size_t shared = 0;
      while (shared < on.chains[j].size() && shared < off.chains[j].size() &&
             on.chains[j][shared] == off.chains[j][shared]) {
        ++shared;
      }
      MESSAGE("DIVERGENCE on " << jobs[j].name << " at position " << shared);
    }
    CHECK(on.chains[j] == off.chains[j]);
  }
  CHECK(exact == static_cast<int>(jobs.size()));

  // (ii) The spec-OFF bench chain ends at the pinned anchor (the engine
  // reproduces the bench's captured chain). The anchor is re-recorded for the
  // bf16-dot class (quiet box, 2026-10-23, load 1.0-1.4, log
  // /tmp/ngram-gate-quiet.log; see docs/bench-evidence/kolibri1-bf16dot-gate-20261022/).
  CHECK(off.chains[0].back() == kAnchorLastToken);

  // (iii) ACCEPTANCE. The bench prompt: re-recorded 54/54 for the bf16-dot
  // class (quiet box, 2026-10-23, log /tmp/ngram-gate-quiet.log) — the class
  // legitimately moves the chain, so the draft count moves with it. A materially
  // different count means the engine's draft scheduling, not the drafter,
  // is misbehaving.
  CHECK(on.proposed[0] == 54);
  CHECK(on.accepted[0] == 54);
  // The W3 goldens: re-recorded bands for the bf16-dot class (quiet box,
  // 2026-10-23, log /tmp/ngram-gate-quiet.log: 164/281). A band, not a pin:
  // the near-tie chains are the sensitivity, and the bench prompt carries
  // the strict pin.
  int64_t g_acc = 0, g_prop = 0;
  for (size_t j = 1; j < jobs.size(); ++j) {
    g_acc += on.accepted[j];
    g_prop += on.proposed[j];
  }
  MESSAGE("goldens aggregate: accepted " << g_acc << "/" << g_prop
          << " (bf16-dot quiet-box measurement: 164/281)");
  CHECK(g_acc >= 140);
  CHECK(g_acc <= 164);
  CHECK(g_prop >= 264);
  CHECK(g_prop <= 281);
  CHECK(g_acc > 0);

  // (iv) THE W3 RELATIONSHIP, unchanged by spec decode: spec-ON == spec-OFF
  // byte-identically (gate i), so the spec-ON chains stand exactly where the
  // spec-OFF chains stand against the torch goldens. The per-prompt shared
  // prefixes are reported (the probe measured 4/8 prompts token-identical to
  // the torch golden end to end, the other 4 flipping at a near-tie early).
  int full_match = 0;
  for (size_t j = 1; j < jobs.size(); ++j) {
    const std::vector<int32_t> golden =
        g.at("prompts")[j - 1].at("generated_ids").get<std::vector<int32_t>>();
    size_t shared = 0;
    while (shared < off.chains[j].size() && shared < golden.size() &&
           off.chains[j][shared] == golden[shared]) {
      ++shared;
    }
    if (shared == golden.size()) ++full_match;
    MESSAGE("W3 " << jobs[j].name << ": engine chain shares " << shared << "/"
                  << golden.size() << " tokens with the torch golden (spec-ON "
                  << "identical to spec-OFF by gate (i))");
  }
  MESSAGE("W3 relationship: " << full_match << "/8 prompts token-identical to "
          << "the torch golden end to end (probe: 4/8)");

  // (v) ENGINE-LEVEL TOK/S, reported (never asserted: machine load may move
  // it — the decode bench's own convention). Wall tok/s divides the FULL
  // output count by the whole request. Decode tok/s divides the tokens
  // generated INSIDE the decode interval — the full count LESS the tokens
  // already present when the first output arrived — by the interval from
  // that first output to finish. (An earlier pairing divided the full count
  // by that interval, billing the first step's tokens — several at once under
  // spec decode — with no generation time; the corrected denominator counts
  // only tokens the interval actually produced, so the corrected decode
  // figures read modestly LOWER than the overcounted ones.)
  const auto tok_s = [](double seconds, int64_t tokens) {
    return seconds > 0.0 ? static_cast<double>(tokens) / seconds : 0.0;
  };
  MESSAGE("ENGINE bench prompt: spec-OFF wall " << tok_s(off.wall_s[0], 64)
          << " tok/s, decode "
          << tok_s(off.decode_s[0], off.decode_tokens[0])
          << " tok/s | spec-ON wall " << tok_s(on.wall_s[0], 64)
          << " tok/s, decode " << tok_s(on.decode_s[0], on.decode_tokens[0])
          << " tok/s (probe harness reference: 4.92 plain / 14.51 spec "
             "decode tok/s)");
  double off_g_wall = 0.0, off_g_dec = 0.0, on_g_wall = 0.0, on_g_dec = 0.0;
  int64_t off_g_dec_tok = 0, on_g_dec_tok = 0;
  for (size_t j = 1; j < jobs.size(); ++j) {
    off_g_wall += off.wall_s[j];
    off_g_dec += off.decode_s[j];
    on_g_wall += on.wall_s[j];
    on_g_dec += on.decode_s[j];
    off_g_dec_tok += off.decode_tokens[j];
    on_g_dec_tok += on.decode_tokens[j];
  }
  MESSAGE("ENGINE goldens aggregate: spec-OFF wall "
          << tok_s(off_g_wall, 8 * 32) << " tok/s, decode "
          << tok_s(off_g_dec, off_g_dec_tok) << " tok/s | spec-ON wall "
          << tok_s(on_g_wall, 8 * 32) << " tok/s, decode "
          << tok_s(on_g_dec, on_g_dec_tok)
          << " tok/s (probe harness reference: 6.84 plain / 8.99 spec)");
}
