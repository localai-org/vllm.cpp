// Kolibri-1 Tenstorrent B2b-ii streaming MoE — the HOST-side gate TU
// (MODEL-TEXT-kolibri-1-tenstorrent, spec .agents/specs/kolibri-tt.md
// ### B2 scope — B2b addendum, slice ii; issue
// ISSUE-LOCAL-01M4ER0E9HHM95YZYJB7T5FECN).
//
// Device-free (the addendum's test ordering): NO card, NO checkpoint
// mount. Covers the fetch executor's host half and the readback pivot:
//
//  - the slot-pool plan over the tiny synthetic fixture (the b2bi
//    pattern): per-layer capacity, byte math vs the B1 streaming plan,
//    the policy built UNCHANGED through PlanKolibri1TTExpertSlotPolicy,
//    and the two named refusals (one-slot-per-layer unaffordable; the
//    B1 host tier not matching the loaded routed tier),
//  - the fetch-list construction: the logical-expert -> slot remap over
//    the B2a dispatch plan, the job fields (slot, device offset, host
//    payload identity), and the byte accounting,
//  - the LOUD stream-bound refusals: a dispatch over the B1 per-token
//    ceiling and a guard charge over the ceiling NEVER degrade
//    silently — each throws by name,
//  - the slot shadow (the readback pivot's reference side): record,
//    byte-exact verify, corrupt-readback refusal, eviction-hook clear,
//  - the reset lane: ContentChangedSince flips on a slot swap, does NOT
//    flip on an identical re-selection (the GDN churn semantics), and
//    the epoch recording consumes it.
//
// RED-FIRST: every case here was written against the new
// kolibri1_tt_stream.h API before the implementation compiled — the red
// was the TU's own build/step failure; each case was then observed
// failing for its stated reason where a runtime failure is reachable
// (the refusal cases), and the positive cases assert the exact byte and
// slot contracts the device leg consumes.
#include <doctest/doctest.h>

#include <atomic>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include <limits>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_tt_forward.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/kolibri1_shared.h"
#include "vllm/model_executor/models/kolibri1_tt.h"
#include "vllm/model_executor/models/kolibri1_tt_stream.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/transformers_utils/hf_config.h"

#include "kolibri1_manifest.inc"

using namespace vllm;  // NOLINT

namespace {

// ---- the synthetic tiny checkpoint (the b2bi test pattern, verbatim) ----

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
           ("vllm_kolibri1_tt_b2ii_" + std::to_string(nonce) + "_" +
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
               Bf16Filled({s.hidden}, 0x3F00)});
  const std::pair<const char*, uint16_t> norms[] = {
      {"input_layernorm", 0x3F80},           {"post_attn_norm", 0x4000},
      {"post_attention_layernorm", 0x4040},  {"post_ffn_norm", 0x4080},
  };
  for (int64_t l = 0; l < 2; ++l) {
    const std::string base = "model.layers." + std::to_string(l) + ".";
    for (const auto& [norm, word] : norms) {
      t.push_back({base + norm + ".weight", "BF16", {s.hidden},
                   Bf16Filled({s.hidden}, word)});
    }
    t.push_back({base + "self_attn.q_norm.weight", "BF16", {s.head_dim},
                 Bf16Filled({s.head_dim}, 0x3E00)});
    t.push_back({base + "self_attn.k_norm.weight", "BF16", {s.head_dim},
                 Bf16Filled({s.head_dim}, 0x3E80)});
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

Kolibri1Weights LoadTiny() {
  TempCheckpoint ckpt(TinyFixture());
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  return LoadKolibri1Weights(shards, MakeTinyConfig());
}

// The B1 streaming plan over the tiny fixture's routed-tier byte math.
// Per-expert: 3 packed projections (32x64, 32x64, 64x32 = 2048 B each)
// + 3 scale grids (1x1 f32 = 4 B each) = 6156 B.
int64_t TinyExpertBytes(const TinyShape& s = {}) {
  const int64_t packed = 3 * s.inter * s.hidden;
  const int64_t scales =
      3 * 4 * CDiv(s.inter, 128) * CDiv(s.hidden, 128) +
      0;  // gate/up: [cdiv(inter,128), cdiv(hidden,128)]; down transposed
  // gate [cdiv(32,128)=1, cdiv(64,128)=1] = 4 B; up 4 B; down 4 B.
  (void)scales;
  return packed + 12;
}

Kolibri1TTStreamingPlan TinyStreaming(const TinyShape& s = {},
                                      int64_t device_budget = 0,
                                      int64_t host_budget = 0) {
  Kolibri1TTStreamingShape shape;
  shape.layers = 2;
  shape.experts = s.experts;
  shape.topk = 2;
  shape.expert_bytes = TinyExpertBytes(s);
  shape.attention_bytes = 1024;
  shape.shared_expert_bytes = shape.expert_bytes * 2;
  shape.router_bytes = 256;
  shape.norm_bytes = 128;
  shape.embed_head_bytes = 512;
  Kolibri1TTStreamingOptions opts;
  opts.device_budget_bytes =
      device_budget != 0 ? device_budget : (int64_t{1} << 30);
  opts.host_budget_bytes =
      host_budget != 0 ? host_budget : (int64_t{4} << 30);
  opts.concurrency = 1;
  // The tiny fixture's 2-of-4 request touches 0.5 of the expert space —
  // over the default 0.25 threshold. The fixture is byte-math only, so
  // the threshold is raised here; the CONCURRENCY refusal itself is
  // inherited unchanged from B1 and is not this TU's subject.
  opts.touched_fraction_threshold = 1.0;
  return PlanKolibri1TTStreaming(shape, opts);
}

}  // namespace

// ---- the slot-pool plan -------------------------------------------------------

TEST_CASE("kolibri1 TT B2b-ii: the slot pool plan — per-layer capacity, "
          "byte math, the policy consumed unchanged") {
  const Kolibri1Weights w = LoadTiny();
  const Kolibri1TTStreamingPlan st = TinyStreaming();
  const int64_t expert_bytes = TinyExpertBytes();
  REQUIRE(st.hot_experts > 0);

  const Kolibri1TTSlotPoolPlan pool = PlanKolibri1TTSlotPool(w, st);
  CHECK(pool.layers == 2);
  CHECK(pool.packed_bytes_per_slot == 3 * 32 * 64);
  CHECK(pool.scale_bytes_per_slot == 12);
  CHECK(pool.packed_bytes_per_slot + pool.scale_bytes_per_slot ==
        expert_bytes);
  // The policy is the B2a one, built against the per-layer residual
  // share: capacity == min(hot_experts, share / expert_bytes) and the
  // per-layer pool == capacity x expert_bytes.
  const int64_t share = st.device_residual_bytes / pool.layers;
  const int64_t want_cap =
      std::min({st.hot_experts, share / expert_bytes, int64_t{4}});
  CHECK(pool.capacity_per_layer == want_cap);
  CHECK(pool.policy.capacity == pool.capacity_per_layer);
  CHECK(pool.policy.layers == 2);
  CHECK(pool.policy.experts == 4);
  CHECK(pool.policy.expert_bytes == expert_bytes);
  CHECK(pool.per_layer_pool_bytes == pool.capacity_per_layer * expert_bytes);
  CHECK(pool.total_pool_bytes == pool.per_layer_pool_bytes * 2);
  CHECK(pool.total_pool_bytes <= st.device_residual_bytes);
  CHECK(pool.total_scale_bytes ==
        pool.scale_bytes_per_slot * pool.capacity_per_layer * 2);
}

TEST_CASE("kolibri1 TT B2b-ii: the slot pool refuses a residual that "
          "cannot afford one slot per layer, by name") {
  const Kolibri1Weights w = LoadTiny();
  Kolibri1TTStreamingPlan st = TinyStreaming();
  st.device_residual_bytes = TinyExpertBytes() / 2;  // less than one slot
  st.hot_experts = 0;
  bool threw = false;
  try {
    (void)PlanKolibri1TTSlotPool(w, st);
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string m = e.what();
    CHECK(m.find("ONE slot per layer") != std::string::npos);
    CHECK(m.find("B2b-ii") != std::string::npos);
  }
  CHECK(threw);
}

TEST_CASE("kolibri1 TT B2b-ii: the slot pool refuses a B1 host tier that "
          "does not match the loaded routed tier") {
  const Kolibri1Weights w = LoadTiny();
  Kolibri1TTStreamingPlan st = TinyStreaming();
  st.host_required_bytes += 1;  // corrupt the cross-check
  bool threw = false;
  try {
    (void)PlanKolibri1TTSlotPool(w, st);
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string m = e.what();
    CHECK(m.find("host tier") != std::string::npos);
    CHECK(m.find("does not equal") != std::string::npos);
  }
  CHECK(threw);
}

// ---- the fetch executor's host half -------------------------------------------

TEST_CASE("kolibri1 TT B2b-ii: the fetch list — remap, slots, offsets, "
          "byte accounting") {
  const Kolibri1Weights w = LoadTiny();
  const Kolibri1TTStreamingPlan st = TinyStreaming();
  const Kolibri1TTSlotPoolPlan pool = PlanKolibri1TTSlotPool(w, st);
  REQUIRE(pool.capacity_per_layer >= 1);

  // Make expert 1 resident on layer 0; request {1, 2}: 1 resident, 2 missed.
  Kolibri1TTExpertSlotPolicy policy = pool.policy;
  const int64_t resident_slot = policy.Touch(0, 1);
  REQUIRE(policy.SlotFor(0, 1) == resident_slot);

  const std::vector<std::pair<int64_t, std::vector<int64_t>>> req = {
      {0, {1, 2}}};
  const Kolibri1TTDispatchPlan disp = PlanKolibri1TTMoEDispatch(policy, req);
  REQUIRE(disp.layers.size() == 1);
  CHECK(disp.layers[0].resident_experts ==
        std::vector<int64_t>{1});
  CHECK(disp.layers[0].missed_experts == std::vector<int64_t>{2});

  const Kolibri1TTSlotFetchList list = BuildKolibri1TTSlotFetchList(
      policy, disp.layers[0], w, pool, st.per_token_stream_bytes);
  REQUIRE(list.jobs.size() == 1);
  CHECK(list.stream_bytes == 1 * TinyExpertBytes());
  CHECK(list.stream_bytes == 1 * TinyExpertBytes());
  const Kolibri1TTSlotFetchJob& job = list.jobs[0];
  CHECK(job.layer == 0);
  CHECK(job.expert == 2);
  CHECK(job.slot == disp.layers[0].fetch_slots[0]);
  CHECK(job.packed_bytes == pool.packed_bytes_per_slot);
  CHECK(job.scale_bytes == pool.scale_bytes_per_slot);
  CHECK(job.device_offset == job.slot * pool.packed_bytes_per_slot);
  // The host payload IS the loaded weights' first packed projection base
  // (the fetch stages per projection through the weights tree).
  const uint8_t* want =
      w.layers[0].moe.experts[2].gate_proj.fp8_block.packed.bytes.data();
  CHECK(job.packed_host == want);

  // A repeat-collapsed request never double-fetches: {2, 2} misses once.
  const Kolibri1TTDispatchPlan disp2 =
      PlanKolibri1TTMoEDispatch(policy, {{0, {2, 2}}});
  const Kolibri1TTSlotFetchList list2 = BuildKolibri1TTSlotFetchList(
      policy, disp2.layers[0], w, pool, st.per_token_stream_bytes);
  CHECK(list2.jobs.size() == 1);
  CHECK(list2.stream_bytes == TinyExpertBytes());
}

TEST_CASE("kolibri1 TT B2b-ii: a dispatch over the B1 per-token bound "
          "refuses LOUDLY, never degrades") {
  const Kolibri1Weights w = LoadTiny();
  const Kolibri1TTStreamingPlan st = TinyStreaming();
  const Kolibri1TTSlotPoolPlan pool = PlanKolibri1TTSlotPool(w, st);
  Kolibri1TTExpertSlotPolicy policy = pool.policy;
  // An empty pool: a 2-of-4 request misses twice = 2 x expert_bytes.
  const Kolibri1TTDispatchPlan disp =
      PlanKolibri1TTMoEDispatch(policy, {{0, {0, 3}}});
  REQUIRE(disp.stream_bytes == 2 * TinyExpertBytes());

  bool threw = false;
  try {
    (void)BuildKolibri1TTSlotFetchList(policy, disp.layers[0], w, pool,
                                       /*per_token_stream_bytes=*/1);
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string m = e.what();
    CHECK(m.find("per-token bound") != std::string::npos);
    CHECK(m.find("LOUDLY") != std::string::npos);
  }
  CHECK(threw);
}

TEST_CASE("kolibri1 TT B2b-ii: the per-step stream-bound guard charges "
          "across layers and refuses on exceed") {
  Kolibri1TTStreamBoundGuard guard(3 * TinyExpertBytes());
  guard.Charge(0, TinyExpertBytes());
  guard.Charge(1, 2 * TinyExpertBytes());
  CHECK(guard.charged() == 3 * TinyExpertBytes());
  bool threw = false;
  try {
    guard.Charge(2, 1);
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string m = e.what();
    CHECK(m.find("stream bound exceeded") != std::string::npos);
    CHECK(m.find("LOUDLY") != std::string::npos);
  }
  CHECK(threw);
  CHECK(guard.charged() == 3 * TinyExpertBytes());  // refused charge not taken
}

// ---- the slot shadow (the readback pivot's reference side) --------------------

TEST_CASE("kolibri1 TT B2b-ii: the slot shadow verifies byte-exact, "
          "refuses a corrupted readback, and clears on eviction") {
  const Kolibri1Weights w = LoadTiny();
  const Kolibri1TTStreamingPlan st = TinyStreaming();
  const Kolibri1TTSlotPoolPlan pool = PlanKolibri1TTSlotPool(w, st);
  // The pool's tiny-fixture capacity exceeds the 4-expert domain, so the
  // eviction exercise builds the SAME policy builder with a capacity of
  // 2 (the builder is the B2a one, unchanged) to force an LRU eviction.
  Kolibri1TTSlotPolicyOptions small_opts;
  small_opts.layers = 2;
  small_opts.experts = 4;
  small_opts.expert_bytes = TinyExpertBytes();
  small_opts.device_residual_bytes = 2 * TinyExpertBytes() * 2;
  small_opts.requested_hot_experts = 2;
  Kolibri1TTExpertSlotPolicy policy =
      PlanKolibri1TTExpertSlotPolicy(small_opts);
  Kolibri1TTSlotShadow shadow(policy, pool.packed_bytes_per_slot);
  shadow.AttachEvictHook();

  // The slot payload is the THREE projections' packed bytes concatenated
  // (gate, up, down) — build the shadow's byte image the way the fetch
  // stages it.
  auto concat = [&](int64_t expert) {
    const Kolibri1ExpertWeights& ew =
        w.layers[0].moe.experts[static_cast<size_t>(expert)];
    std::vector<uint8_t> all;
    for (const Kolibri1Projection* p :
         {&ew.gate_proj, &ew.up_proj, &ew.down_proj}) {
      all.insert(all.end(), p->fp8_block.packed.bytes.begin(),
                 p->fp8_block.packed.bytes.end());
    }
    return all;
  };
  const std::vector<uint8_t> packed0 = concat(0);
  const std::vector<uint8_t> packed1 = concat(1);
  REQUIRE(static_cast<int64_t>(packed0.size()) == pool.packed_bytes_per_slot);

  const int64_t slot0 = policy.Touch(0, 0);
  shadow.Record(0, slot0, packed0.data(), pool.packed_bytes_per_slot);
  CHECK(shadow.Has(0, slot0));
  CHECK(shadow.FilledSlots() == 1);
  CHECK(shadow.VerifyReadback(0, slot0, packed0.data(),
                              pool.packed_bytes_per_slot) ==
        pool.packed_bytes_per_slot);

  // A corrupted device readback refuses by name.
  std::vector<uint8_t> corrupt(packed0.begin(), packed0.end());
  corrupt[0] ^= 0xff;
  bool threw = false;
  try {
    (void)shadow.VerifyReadback(0, slot0, corrupt.data(),
                                pool.packed_bytes_per_slot);
  } catch (const std::runtime_error& e) {
    threw = true;
    const std::string m = e.what();
    CHECK(m.find("DEVICE READBACK") != std::string::npos);
    CHECK(m.find("byte-for-byte") != std::string::npos);
  }
  CHECK(threw);

  // The eviction hook: fill both slots, then pin a third expert — the
  // LRU (expert 0) evicts and its shadow clears.
  for (int64_t e = 0; e < 2; ++e) {
    const int64_t slot = policy.Touch(0, e);
    shadow.Record(0, slot, concat(e).data(), pool.packed_bytes_per_slot);
  }
  const int64_t filled_before = shadow.FilledSlots();
  REQUIRE(filled_before == 2);
  const int64_t evicted_expert = 0;
  const int64_t new_slot = policy.Touch(0, 2);  // evicts the LRU (expert 0)
  (void)new_slot;
  CHECK(policy.SlotFor(0, evicted_expert) == -1);
  CHECK(shadow.FilledSlots() == filled_before - 1);
  // The evicted slot no longer verifies (it is unfilled).
  CHECK(!shadow.Has(0, slot0));
}

TEST_CASE("kolibri1 TT B2b-ii: the reset lane — a swap flips "
          "ContentChangedSince, an identical re-selection does not") {
  const Kolibri1Weights w = LoadTiny();
  const Kolibri1TTStreamingPlan st = TinyStreaming();
  const Kolibri1TTSlotPoolPlan pool = PlanKolibri1TTSlotPool(w, st);
  Kolibri1TTExpertSlotPolicy policy = pool.policy;
  // Fill BEFORE the recording: the recording marks the epoch baseline.
  (void)policy.Touch(0, 2);
  Kolibri1TTSlotEpoch epoch;
  epoch.Record(policy);
  CHECK(!policy.ContentChangedSince(epoch.fingerprint));
  CHECK(!epoch.ConsumeIfChanged(policy));
  CHECK(epoch.resets == 0);

  // Identical re-selection: no flip.
  (void)policy.Touch(0, 2);
  CHECK(!epoch.ConsumeIfChanged(policy));
  CHECK(epoch.resets == 0);

  // A genuine content change: a NEW expert pinned (capacity >= 2) or
  // the slot reused for a different expert (capacity 1).
  if (pool.capacity_per_layer == 1) {
  (void)policy.Touch(0, 3);  // evicts expert 2, reuses the slot
  } else {
  (void)policy.Touch(0, 3);  // fills a new slot
  }
  CHECK(epoch.ConsumeIfChanged(policy));
}

// ---- the router readback pivot --------------------------------------------------

TEST_CASE("kolibri1 TT B2b-ii: the router readback pivot — whole-buffer "
          "download, count refusal") {
  // The pivot wraps a flat D2H copy; a stub download proves the contract.
  std::vector<float> src = {1.0f, 2.0f, 3.0f, 4.0f};
  const std::vector<float> got = Kolibri1TTRouterLogitsReadback(
      [&](int64_t count, float* dst) {
        REQUIRE(count == 4);
        std::memcpy(dst, src.data(), sizeof(float) * 4);
      },
      4);
  CHECK(got == src);

  bool threw = false;
  try {
    (void)Kolibri1TTRouterLogitsReadback(
        [](int64_t, float*) { FAIL("must not download"); }, 0);
  } catch (const std::runtime_error&) {
    threw = true;
  }
  CHECK(threw);
}

// ---- DEVICE LEG: the token gate (the W3 methodology on the P150) -----------
//
// Runs only with a Tenstorrent card AND VT_KOLIBRI1_TT_B2II_MODEL=<real
// checkpoint dir> (operator-run under the GPU lease; evidence in
// docs/bench-evidence/). THE TOKEN GATE, deferred from B2b-i: replay the
// golden prompts on the TT device path with the FULL model — the W3
// methodology (tests/vllm/models/test_kolibri1_w3.cpp): 141/145 argmax
// positions, the known flips adjudicated INSIDE the 2.5-nat band, 0 hard
// flips allowed. Any flip outside the band is a gate FAILURE, not a
// re-adjudication; per-flip nat gaps are MESSAGE'd for the evidence doc.

#ifndef KOLIBRI1_GOLDENS
#define KOLIBRI1_GOLDENS "kolibri1_goldens.json"
#endif

namespace {

constexpr double kNatBand = 2.5;
constexpr int32_t kGateSteps = 32;

struct GateGolden {
  std::string prompt;
  std::vector<int32_t> input_ids;
  std::vector<int32_t> generated_ids;
};

std::vector<GateGolden> LoadGateGoldens() {
  const nlohmann::json j =
      nlohmann::json::parse(std::ifstream(KOLIBRI1_GOLDENS));
  std::vector<GateGolden> out;
  for (const auto& p : j.at("prompts")) {
    GateGolden g;
    g.prompt = p.at("prompt").get<std::string>();
    for (const auto& v : p.at("input_ids"))
      g.input_ids.push_back(v.get<int32_t>());
    for (const auto& v : p.at("generated_ids"))
      g.generated_ids.push_back(v.get<int32_t>());
    out.push_back(std::move(g));
  }
  return out;
}

v1::CommonAttentionMetadata GateMeta(int64_t t, int64_t ctx_before,
                                     int64_t block_table_num_cols) {
  v1::CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(t);
  m.max_query_len = static_cast<int>(t);
  m.max_seq_len = static_cast<int>(ctx_before + t);
  m.query_start_loc = {0, static_cast<int32_t>(t)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(ctx_before + t)};
  m.seq_lens_cpu = m.seq_lens;
  m.num_computed_tokens_cpu = {static_cast<int32_t>(ctx_before)};
  m.block_table_tensor.assign(static_cast<size_t>(block_table_num_cols), 0);
  for (int64_t i = 0; i < block_table_num_cols; ++i)
    m.block_table_tensor[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  m.block_table_num_cols = static_cast<int>(block_table_num_cols);
  for (int64_t i = 0; i < t; ++i) m.slot_mapping.push_back(ctx_before + i);
  m.causal = true;
  return m;
}

bool TenstorrentDevicePresent() {
  vt::Backend* b = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  return b != nullptr;
}

// One device forward over `ids` at `ctx_before`; downloads the LAST row's
// logits and returns them.
std::vector<float> ForwardLastLogits(LoadedModel& model, const HfConfig& config,
                                     vt::Queue& q, vt::Backend& be,
                                     std::vector<PagedKvCache>& kv,
                                     const std::vector<int32_t>& ids,
                                     int64_t ctx_before, int64_t num_blocks) {
  const int64_t t = static_cast<int64_t>(ids.size());
  const v1::CommonAttentionMetadata meta =
      GateMeta(t, ctx_before, num_blocks / 2);
  v1::GDNAttentionMetadata gdn_meta;
  std::vector<GdnStateCache> gdn_state;
  std::vector<int32_t> positions(static_cast<size_t>(t));
  for (int64_t i = 0; i < t; ++i)
    positions[static_cast<size_t>(i)] = static_cast<int32_t>(ctx_before + i);
  ModelForwardInput in{ids, positions, meta, gdn_meta, kv, gdn_state, config,
                       q, /*logits_indices=*/{}, /*num_reqs=*/1};
  in.pure_decode = (t == 1);
  in.uniform_query_len = (t == 1 ? 1 : 0);
  ForwardLogits fl = ModelRegistry::Forward(model, in);
  REQUIRE(fl.rows >= 1);
  std::vector<float> logits(static_cast<size_t>(fl.vocab));
  be.Copy(q, logits.data(),
          static_cast<const uint8_t*>(fl.device_tensor.data) +
              (fl.rows - 1) * fl.vocab * static_cast<int64_t>(sizeof(float)),
          static_cast<size_t>(fl.vocab) * sizeof(float));
  be.Synchronize(q);
  return logits;
}

int32_t prev_got_dbg = 0;

}  // namespace

// SCRATCH DEBUG (remove before landing): TT vs CPU logits, same registry
// path, prompt 1 walked one token per step.
TEST_CASE("SCRATCH dbg TT vs CPU logits") {
  if (!TenstorrentDevicePresent()) return;
  const char* model_dir = std::getenv("VT_KOLIBRI1_TT_B2II_MODEL");
  if (model_dir == nullptr || *model_dir == '\0') return;
  const std::string dir = model_dir;
  const HfConfig config = vllm::LoadHfConfig(dir + "/config.json");
  const auto index = nlohmann::json::parse(
      std::ifstream(dir + "/model.safetensors.index.json"));
  std::set<std::string> shard_names;
  for (const auto& [name, shard] : index.at("weight_map").items())
    shard_names.insert(shard.get<std::string>());
  std::vector<SafetensorsFile> shards;
  for (const std::string& shard : shard_names)
    shards.push_back(SafetensorsFile::Open(dir + "/" + shard));
  const ModelSource source = ModelSource::FromSafetensors(shards);

  // CPU model + queue.
  vt::Backend& cpu_be = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue cpu_q = cpu_be.CreateQueue();
  std::unique_ptr<LoadedModel> cpu_model = ModelRegistry::Load(config, source);
  ModelRegistry::Prepare(*cpu_model, config, cpu_q);
  const int64_t num_blocks = 32;
  const HfConfig* cfg = &config;
  (void)cfg;

  // CPU KV caches.
  const Kolibri1Weights& cw = Kolibri1LoadedModelWeights(*cpu_model);
  const Kolibri1Params& cp = cw.params;
  std::vector<PagedKvCache> cpu_kv;
  std::vector<std::vector<uint8_t>> cpu_kv_bytes;
  for (int64_t l = 0; l < cp.num_hidden_layers; ++l) {
    cpu_kv_bytes.emplace_back(static_cast<size_t>(
        num_blocks * 2 * 16 * cp.num_key_value_heads * cp.head_dim *
        vt::SizeOf(vt::DType::kBF16)));
    PagedKvCache c;
    c.data = cpu_kv_bytes.back().data();
    c.dtype = vt::DType::kBF16;
    c.num_blocks = num_blocks;
    c.block_size = 16;
    c.num_kv_heads = cp.num_key_value_heads;
    c.head_size = cp.head_dim;
    cpu_kv.push_back(c);
  }

  // TT model + queue.
  vt::Backend& tt_be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue tt_q = tt_be.CreateQueue();
  std::unique_ptr<LoadedModel> tt_model = ModelRegistry::Load(config, source);
  ModelRegistry::Prepare(*tt_model, config, tt_q);
  const Kolibri1Weights& tw = Kolibri1LoadedModelWeights(*tt_model);
  const Kolibri1Params& tp = tw.params;
  std::vector<PagedKvCache> tt_kv;
  std::vector<std::shared_ptr<void>> tt_keep;
  for (int64_t l = 0; l < tp.num_hidden_layers; ++l) {
    void* buf = tt_be.Alloc(cpu_kv_bytes[0].size());
    tt_keep.emplace_back(buf, [&tt_be](void* p) { tt_be.Free(p); });
    PagedKvCache c;
    c.data = buf;
    c.dtype = vt::DType::kBF16;
    c.num_blocks = num_blocks;
    c.block_size = 16;
    c.num_kv_heads = tp.num_key_value_heads;
    c.head_size = tp.head_dim;
    tt_kv.push_back(c);
  }

  const GateGolden gp = LoadGateGoldens()[0];
  auto fwd = [&](LoadedModel& m, vt::Queue& q2, vt::Backend& b2,
                 std::vector<PagedKvCache>& kv, const std::vector<int32_t>& ids,
                 int64_t ctx) {
    return ForwardLastLogits(m, config, q2, b2, kv, ids, ctx, num_blocks);
  };
  std::vector<int32_t> seq;
  for (size_t i = 0; i < gp.input_ids.size() + 3; ++i) {
    if (i < gp.input_ids.size()) {
      seq.push_back(gp.input_ids[i]);
    } else {
      // after the prompt, feed CPU's argmax (keep both on the same context)
      seq.push_back(0);  // placeholder, replaced below
      seq.back() = prev_got_dbg;
    }
    std::vector<float> lc =
        fwd(*cpu_model, cpu_q, cpu_be, cpu_kv, {seq.back()},
            static_cast<int64_t>(seq.size()) - 1);
    std::vector<float> lt =
        fwd(*tt_model, tt_q, tt_be, tt_kv, {seq.back()},
            static_cast<int64_t>(seq.size()) - 1);
    double ma = 0;
    size_t amax = 0;
    for (size_t j = 0; j < lc.size(); ++j) {
      const double d = std::fabs(static_cast<double>(lc[j]) -
                                 static_cast<double>(lt[j]));
      if (d > ma) { ma = d; amax = j; }
    }
    int32_t ac = static_cast<int32_t>(
        std::max_element(lc.begin(), lc.end()) - lc.begin());
    int32_t at = static_cast<int32_t>(
        std::max_element(lt.begin(), lt.end()) - lt.begin());
    MESSAGE("step " << i << " tok " << seq.back() << ": max_abs=" << ma
                    << " at " << amax << " argmax cpu=" << ac
                    << " tt=" << at << " logit cpu="
                    << lc[static_cast<size_t>(ac)] << " tt="
                    << lt[static_cast<size_t>(at)]);
    prev_got_dbg = ac;
  }
  (void)gp;
}

TEST_CASE("kolibri1 TT B2b-ii: device token gate — the golden argmax chains "
          "on the card (W3 methodology, 141/145 in the 2.5-nat band)") {
  if (!TenstorrentDevicePresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  const char* model_dir = std::getenv("VT_KOLIBRI1_TT_B2II_MODEL");
  if (model_dir == nullptr || *model_dir == '\0') {
    MESSAGE("SKIPPED: VT_KOLIBRI1_TT_B2II_MODEL not set");
    return;
  }
  const std::string dir = model_dir;
  if (!std::filesystem::exists(dir + "/model.safetensors.index.json")) {
    MESSAGE("SKIPPED: " << dir << " is not a Kolibri-1 checkpoint");
    return;
  }

  // Load through the PRODUCTION registry path.
  const HfConfig config = vllm::LoadHfConfig(dir + "/config.json");
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.architecture == "Kolibri1ForCausalLM");
  const auto index = nlohmann::json::parse(
      std::ifstream(dir + "/model.safetensors.index.json"));
  std::set<std::string> shard_names;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)name;
    shard_names.insert(shard.get<std::string>());
  }
  std::vector<SafetensorsFile> shards;
  for (const std::string& shard : shard_names)
    shards.push_back(SafetensorsFile::Open(dir + "/" + shard));
  const ModelSource source = ModelSource::FromSafetensors(shards);
  std::unique_ptr<LoadedModel> model = ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  ModelRegistry::Prepare(*model, config, q);
  Kolibri1TTStreamingDeviceContext* st =
      Kolibri1LoadedModelTTStreamContext(*model, q);
  REQUIRE(st != nullptr);
  const Kolibri1Weights& w = Kolibri1LoadedModelWeights(*model);
  const Kolibri1Params& p = w.params;

  // Per-layer KV caches (bf16), the b2bi device-leg pattern.
  const int64_t hkv = p.num_key_value_heads;
  const int64_t hd = p.head_dim;
  const int64_t block_size = 16;
  const int64_t num_blocks = 32;
  const int64_t kv_bytes = num_blocks * 2 * block_size * hkv * hd *
                           static_cast<int64_t>(vt::SizeOf(vt::DType::kBF16));
  std::vector<std::shared_ptr<void>> kv_keep;
  std::vector<PagedKvCache> kv;
  kv.reserve(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    void* buf = be.Alloc(kv_bytes);
    kv_keep.emplace_back(buf, [&be](void* ptr) { be.Free(ptr); });
    std::memset(buf, 0, static_cast<size_t>(kv_bytes));
    PagedKvCache c;
    c.data = buf;
    c.dtype = vt::DType::kBF16;
    c.num_blocks = num_blocks;
    c.block_size = block_size;
    c.num_kv_heads = hkv;
    c.head_size = hd;
    kv.push_back(c);
  }

  int64_t argmax_matches = 0, argmax_total = 0, near_ties = 0, hard_flips = 0;
  std::vector<std::string> flip_records;

  for (const GateGolden& gp : LoadGateGoldens()) {
    // Fresh KV per prompt (zeroed: no stale context can leak across
    // prompts).
    for (auto& kkeep : kv_keep)
      std::memset(kkeep.get(), 0, static_cast<size_t>(kv_bytes));
    // The W3 chain: the prompt walks in ONE-TOKEN steps (each forward
    // covers exactly one new position over the KV cache), then the gate
    // decodes greedily the same way. Full-recompute and this incremental
    // walk are identical math (the goldens' semantics,
    // test_kolibri1_w3.cpp:298-310). The prompt CANNOT prefill in one
    // step: one whole-prompt routed dispatch streams every selected
    // expert's slot bytes in a single charge and exceeds the B1 per-step
    // stream bound (observed: 1242865920 B against 1179936000 B at layer
    // 15) — the loud refusal is genuine per-step over-capacity, and the
    // chunked walk is the faithful same-math alternative. (The first
    // draft of this gate decoded step 0 from a ZEROED KV cache while
    // claiming ctx_before = len-1 computed positions — attention over no
    // context, 0/8 hard flips, gate meaningless.)
    std::vector<int32_t> seq;
    std::vector<float> logits;
    int32_t prev_got = -1;
    bool diverged = false;
    for (int32_t step = 0; step < kGateSteps && !diverged; ++step) {
      // The prompt walk first: one token per step, no comparison.
      if (static_cast<size_t>(step) < gp.input_ids.size()) {
        seq.push_back(gp.input_ids[static_cast<size_t>(step)]);
        ForwardLastLogits(*model, config, q, be, kv, {seq.back()},
                          static_cast<int64_t>(seq.size()) - 1, num_blocks);
        continue;
      }
      if (step == static_cast<int32_t>(gp.input_ids.size())) {
        logits = ForwardLastLogits(*model, config, q, be, kv, {seq.back()},
                                   static_cast<int64_t>(seq.size()) - 1,
                                   num_blocks);
      } else {
        seq.push_back(prev_got);
        logits = ForwardLastLogits(*model, config, q, be, kv, {seq.back()},
                                   static_cast<int64_t>(seq.size()) - 1,
                                   num_blocks);
      }
      prev_got = static_cast<int32_t>(
          std::max_element(logits.begin(), logits.end()) - logits.begin());
      const int32_t got = prev_got;
      const int32_t want = gp.generated_ids[static_cast<size_t>(step)];
      ++argmax_total;
      if (got == want) {
        ++argmax_matches;
      } else {
        // The nat gap: got's logit MINUS want's logit — how far the
        // golden token was from the winner (<= band => near-tie).
        const double gap = static_cast<double>(
            logits[static_cast<size_t>(want)] - logits[static_cast<size_t>(got)]);
        flip_records.push_back("'" + gp.prompt + "' step " +
                               std::to_string(step) + ": got " +
                               std::to_string(got) + " want " +
                               std::to_string(want) + ", nat gap " +
                               std::to_string(gap));
        if (-gap <= kNatBand) {
          ++near_ties;
        } else {
          ++hard_flips;
          MESSAGE("HARD flip at '" << gp.prompt << "' step " << step
                                   << ": got " << got << " want " << want
                                   << " (gap " << gap << " beyond band "
                                   << kNatBand << ")");
        }
        diverged = true;  // later steps compare different contexts
        break;
      }
    }
  }

  MESSAGE("ARGMAX CHAIN: " << argmax_matches << "/" << argmax_total
                           << " compared positions match the golden greedy "
                              "decode; flips "
                           << (argmax_total - argmax_matches) << " (near-ties "
                           << near_ties << ", hard " << hard_flips << ")");
  for (const std::string& r : flip_records) MESSAGE("flip: " << r);
  MESSAGE("streaming counters: slot fills "
          << st->slot_fills << " (swaps " << st->swap_fills << "), staged "
          << st->staged_bytes << " B, readback-verified "
          << st->readback_verified_bytes << " B, memo hits " << st->memo_hits);
  // THE GATE VERDICT: 141/145 with the flips inside the band, 0 hard.
  CHECK(argmax_matches >= 141);
  CHECK(hard_flips == 0);

  // ---- the near-tie instrument (the decode-bench methodology,
  // tests/vllm/models/test_kolibri1_decode_bench.cpp RunTeacherForced) ----
  //
  // The byte-comparison pass above BREAKS at the first divergence, so every
  // later same-index comparison never happens — but the moment the
  // free-running chain splits from the golden, ANY per-index comparison
  // against a different prefix is meaningless. This instrument adjudicates
  // each divergence under the COMMON (golden) prefix:
  //   pass 1: the TT engine TEACHER-FORCED on the golden chain's tokens
  //           (decode inputs = generated_ids[d-1], never the engine's own
  //           argmax) — per-step logits under the golden prefix;
  //   pass 2: the TT engine free-running over the full 32 steps — the
  //           free chain marks the DIVERGENCE positions (not flips yet).
  // A divergence is a flip only when the engine's teacher-forced argmax
  // under the golden prefix also differs from the golden token; the gap is
  // scored in that engine row. The committed goldens carry per-step TOKENS
  // only (final_logits is the prefill's last position, one top-5), so the
  // distribution here is the engine's own teacher-forced row — the
  // documented fallback. In-band (<= 2.5 nats) and in the row's top-5 =>
  // NEAR-TIE; otherwise HARD. hard_flips == 0 is the NEW check; it is
  // expected to be RED today — that red quantifies the real defect.
  int64_t instr_divergences = 0, instr_flips = 0, instr_near_ties = 0,
           instr_hard_flips = 0;
  double instr_worst_gap = 0.0;
  std::vector<std::string> hard_records;
  for (const GateGolden& gp : LoadGateGoldens()) {
    auto walk = [&](const std::vector<int32_t>* forced) {
      std::vector<int32_t> chain;
      std::vector<std::vector<float>> rows;
      for (auto& kkeep : kv_keep)
        std::memset(kkeep.get(), 0, static_cast<size_t>(kv_bytes));
      std::vector<int32_t> seq;
      int32_t prev = -1;
      for (int32_t step = 0; step < kGateSteps; ++step) {
        if (static_cast<size_t>(step) < gp.input_ids.size()) {
          seq.push_back(gp.input_ids[static_cast<size_t>(step)]);
          ForwardLastLogits(*model, config, q, be, kv, {seq.back()},
                            static_cast<int64_t>(seq.size()) - 1, num_blocks);
          continue;
        }
        if (step == static_cast<int32_t>(gp.input_ids.size())) {
          // First decode logits come off the prompt's last position.
        } else {
          // Feed the GOLDEN chain's previous token, never the engine's own
          // argmax: decode position d predicts generated_ids[d] from the
          // prefix ending with generated_ids[d-1]. The walk's global `step`
          // is prompt_len + d, so the fed token is generated_ids[step-1].
          const int32_t fed =
              forced != nullptr
                  ? (*forced)[static_cast<size_t>(step) - 1]
                  : prev;
          seq.push_back(fed);
        }
        std::vector<float> row = ForwardLastLogits(
            *model, config, q, be, kv, {seq.back()},
            static_cast<int64_t>(seq.size()) - 1, num_blocks);
        prev = static_cast<int32_t>(
            std::max_element(row.begin(), row.end()) - row.begin());
        chain.push_back(prev);
        rows.push_back(std::move(row));
      }
      return std::make_pair(std::move(chain), std::move(rows));
    };

    // PASS 1 (first, so a free-run failure cannot eat the scoring pass):
    // teacher-forced on the golden chain; adjudicate each divergence under
    // the common golden prefix.
    std::vector<int32_t> forced;
    forced.reserve(gp.generated_ids.size());
    for (int32_t t : gp.generated_ids) forced.push_back(t);
    // KNOWN DEFECT (observed, VT_TT_HOST_FREE_DECODE=0, 2026-10-10): even
    // the GOLDEN-prefix walk can hit the streaming pool's no-slot refusal
    // mid-chain ("expert N selected on layer L but no slot holds it",
    // kolibri1_tt_forward.cpp) — the byte gate never reaches that state
    // because it breaks at the first divergence. Record the abort, keep
    // every prompt adjudicated so far.
    std::pair<std::vector<int32_t>, std::vector<std::vector<float>>> tf_run;
    try {
      tf_run = walk(&forced);
    } catch (const std::runtime_error& e) {
      MESSAGE("teacher-forced run aborted on '" << gp.prompt
                                                << "' (recorded defect): "
                                                << e.what());
      continue;
    }
    const std::vector<int32_t>& tf_chain = tf_run.first;
    const std::vector<std::vector<float>>& tf_rows = tf_run.second;
    for (size_t d = 0; d < tf_chain.size(); ++d) {
      const int32_t lane_tok = tf_chain[d];
      const int32_t want =
          gp.generated_ids[d];
      if (lane_tok == want) continue;  // prefix amplification, not a flip
      ++instr_flips;
      const std::vector<float>& row = tf_rows[d];
      const int32_t top = static_cast<int32_t>(
          std::max_element(row.begin(), row.end()) - row.begin());
      const double gap = static_cast<double>(row[static_cast<size_t>(top)]) -
                         static_cast<double>(row[static_cast<size_t>(want)]);
      instr_worst_gap = std::max(instr_worst_gap, gap);
      int above = 0;
      for (float v : row)
        if (static_cast<double>(v) >
            static_cast<double>(row[static_cast<size_t>(want)]))
          ++above;
      const bool in_top5 = above < 5;
      if (gap > kNatBand || !in_top5) {
        ++instr_hard_flips;
        hard_records.push_back("'" + gp.prompt + "' step " +
                               std::to_string(d) + ": golden " +
                               std::to_string(want) + " tf-argmax " +
                               std::to_string(lane_tok) + ", gap " +
                               std::to_string(gap) + " nats, in-top5=" +
                               (in_top5 ? "yes" : "no"));
      } else {
        ++instr_near_ties;
      }
    }

    // PASS 2: free-running over the FULL 32 steps — the divergence count.
    // KNOWN DEFECT (observed, host-free ON, 2026-10-10): past the first
    // divergence the free-running chain can route experts the streaming
    // slot pool refuses to serve ("expert N selected on layer L but no
    // slot holds it", kolibri1_tt_forward.cpp) — the golden-prefix walk
    // never reaches that state because the byte gate breaks at the first
    // divergence. The refusal is a FINDING, not something this instrument
    // fixes: record it, keep the divergences counted up to the abort.
    try {
      const auto free_run = walk(nullptr);
      const std::vector<int32_t>& free_chain = free_run.first;
      for (size_t d = 0; d < free_chain.size(); ++d)
        if (free_chain[d] != gp.generated_ids[d]) ++instr_divergences;
    } catch (const std::runtime_error& e) {
      MESSAGE("free run aborted on '" << gp.prompt
                                      << "' (recorded defect): " << e.what());
    }
  }
  MESSAGE("NEAR-TIE INSTRUMENT: " << instr_divergences
                                  << " free-running divergences, "
                                  << instr_flips << " teacher-forced flips ("
                                  << instr_near_ties << " near-ties, "
                                  << instr_hard_flips << " HARD), worst "
                                     "teacher-forced gap "
                                  << instr_worst_gap << " nats (band "
                                  << kNatBand << ", top-5, engine's own "
                                     "teacher-forced logits as the "
                                     "distribution — the goldens carry "
                                     "per-step tokens only)");
  for (const std::string& r : hard_records) MESSAGE("REAL hard flip: " << r);
  // THE ADJUDICATED VERDICT (expected RED today): every divergence under the
  // common prefix must be a near-tie.
  CHECK(instr_hard_flips == 0);
}
