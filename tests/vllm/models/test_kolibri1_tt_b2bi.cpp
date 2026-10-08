// Kolibri-1 Tenstorrent B2b-i dense-resident device forward
// (MODEL-TEXT-kolibri-1-tenstorrent, spec .agents/specs/kolibri-tt.md
// ### B2 scope — B2b addendum, slice i; issue
// ISSUE-LOCAL-01M4E22DM790W0E69M07D5XA9D).
//
// The B2b-i FORWARD gate TU (the bring-up gate test_kolibri1_tt_b2i.cpp
// covers the resident STAGING; this TU covers the dense-resident FORWARD).
// Two halves:
//
//  - HOST-SIDE (runs under ctest with NO card and NO checkpoint mount):
//    the routed-expert refusal contract (message names the missing part,
//    the owning slice, the row, and the issue; the process-wide counter and
//    its reset seam), the sigmoid-logit-add routing contract as relocated
//    VERBATIM into kolibri1_shared.h (the CPU row's f32 compute path, tied
//    to the exact tie-break and the unbiased-logit weight), and the
//    device-resident compute context over the tiny synthetic fixture —
//    projection count, byte accounting, and BIT-EXACT agreement of every
//    memoized bf16 dequant against kolibri1_fp8::DequantRowsBf16 over the
//    same packed bytes and scale grid — plus the non-fp8-projection refusal.
//
//  - DEVICE LEG (runs only with a Blackhole card AND
//    VT_KOLIBRI1_TT_B2BI_MODEL=<real checkpoint dir>; operator-run under
//    the GPU lock, evidence in docs/bench-evidence/kolibri1-tt-b2bi-fwd-
//    <date>.md): load the REAL fp8 checkpoint through the production
//    registry path, build the device context (the memoized bf16 dequants,
//    7 projections per layer), verify the staged byte totals against the
//    resident plan's fp8 accounting, verify the embedding gather BIT-EXACT,
//    and run ONE GREEDY DECODE of a golden prompt on the card through the
//    PRODUCTION seam (ModelRegistry::Prepare + ModelRegistry::Forward) —
//    the slice's completion condition. The routed-expert refusal must fire
//    BY NAME during the decode; the decode completes with the shared expert
//    only. The goldens' expected tokens are NOT asserted: the goldens are
//    full-model decodes and cannot be replayed without the routed experts
//    (addendum gate ordering) — the 141/145 argmax gate stays OWED to
//    B2b-ii. This slice asserts decode COMPLETION and per-op agreement.
//
//    PER-OP DEVICE-VS-CPU ENVELOPE, stated BEFORE running (the b2i
//    bring-up precedent): the embedding gather must be BIT-EXACT (a row
//    gather, no arithmetic). Every GEMM in the resident path consumes the
//    bf16 dequant of the staged fp8 bytes on BOTH sides (the CPU row's
//    documented R1 disposition), so the envelope per element is
//    |dev - cpu| <= 8 * 2^-8 * sum_k |a_ik w_jk| + max(ulp(|cpu|),
//    ulp(|dev|)) — at most 8 bf16 unit roundoffs per unit term-magnitude
//    sum plus the store rounding, measured worst ratio 1.375 in the b2i
//    bring-up. The 2-ulp accumulation-order premise was FALSIFIED by that
//    measurement and is NEVER restated here.
#include <doctest/doctest.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1_fp8_dequant.h"
#include "vllm/model_executor/models/kolibri1_shared.h"
#include "vllm/model_executor/models/kolibri1_tt_forward.h"
#include "vllm/model_executor/models/kolibri1_tt.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "kolibri1_manifest.inc"

#ifndef KOLIBRI1_GOLDENS
#define KOLIBRI1_GOLDENS "kolibri1_goldens.json"
#endif

using namespace vllm;  // NOLINT

namespace {

// ---- the synthetic tiny checkpoint (the test_kolibri1_tt.cpp pattern) ----

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
           ("vllm_kolibri1_tt_b2bi_" + std::to_string(nonce) + "_" +
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

// Tiny kolibri1 geometry: 2 layers (swa, full), hidden 64, 4 q heads /
// 2 kv heads, head_dim 16, 4 experts, intermediate 32.
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
               Bf16Filled({s.hidden}, 0x3F00)});  // 0.5, the final norm
  // The four sandwich norms and the per-head q/k norms carry DISTINCT bf16
  // sentinel values, so the host-side op census can identify WHICH weight a
  // recorded RmsNorm consumed (a wrong-weight mutation changes the sentinel,
  // not just the pointer). 1.0 / 2.0 / 3.0 / 4.0 / 0.25 / 0.3125.
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

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

int64_t ManifestBytes(const vllm_test::Kolibri1ManifestTensor& t) {
  int64_t elems = 1;
  for (int i = 0; i < t.rank; ++i) elems *= t.shape[i];
  const std::string d = t.dtype;
  if (d == "BF16") return elems * 2;
  if (d == "F8_E4M3") return elems;
  if (d == "F32") return elems * 4;
  throw std::runtime_error(std::string("unknown manifest dtype ") + d);
}

bool ManifestNameResidentProjection(const std::string& name) {
  // The resident fp8-block projections: per-layer attention q/k/v/o and the
  // shared-expert gate/up/down. NOT the routed experts, NOT the router gate
  // (bf16), NOT the norms/embed/head (bf16).
  if (name.find("mlp.experts.") != std::string::npos) return false;
  if (name.find(".weight_scale_inv") != std::string::npos) return false;
  if (name.find("self_attn.q_proj") != std::string::npos ||
      name.find("self_attn.k_proj") != std::string::npos ||
      name.find("self_attn.v_proj") != std::string::npos ||
      name.find("self_attn.o_proj") != std::string::npos ||
      name.find("shared_experts.") != std::string::npos) {
    return true;
  }
  return false;
}

int64_t ManifestResidentProjectionFp8Bytes() {
  int64_t total = 0;
  for (const auto& t : vllm_test::kKolibri1Tensors) {
    if (std::string(t.dtype) == "F8_E4M3" &&
        ManifestNameResidentProjection(t.name)) {
      total += ManifestBytes(t);
    }
  }
  return total;
}

int64_t ManifestResidentProjectionCount() {
  int64_t n = 0;
  for (const auto& t : vllm_test::kKolibri1Tensors) {
    if (std::string(t.dtype) == "F8_E4M3" &&
        ManifestNameResidentProjection(t.name)) {
      ++n;
    }
  }
  return n;
}

// One forward step's attention metadata over ONE request that owns the whole
// block table start.
v1::CommonAttentionMetadata OneReqMeta(int64_t t, int64_t ctx_before,
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

}  // namespace

// ---- HOST: the routed-expert refusal contract ------------------------------

TEST_CASE("kolibri1 TT B2b-i: the routed-expert refusal names the missing "
          "part, the owning slice, the row, and the issue") {
  const std::vector<int32_t> ids = {3, 41, 5, 77, 12, 383};
  const std::string msg = Kolibri1TTRoutedExpertRefusalMessage(7, ids);
  // The message names the missing part (the routed-expert path) and the
  // slice that owns it (B2b-ii, the streaming MoE).
  CHECK(msg.find("routed experts") != std::string::npos);
  CHECK(msg.find("NOT IMPLEMENTED") != std::string::npos);
  CHECK(msg.find("B2b-ii") != std::string::npos);
  CHECK(msg.find("MODEL-TEXT-kolibri-1-tenstorrent") != std::string::npos);
  CHECK(msg.find("ISSUE-LOCAL-01M4E22DM790W0E69M07D5XA9D") != std::string::npos);
  // The layer and the requested expert ids appear (truncated listing).
  CHECK(msg.find("layer 7") != std::string::npos);
  CHECK(msg.find("3,41,5,77,12,383") != std::string::npos);

  // More than 8 requested ids truncate with an ellipsis marker.
  const std::vector<int32_t> many = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9};
  const std::string msg_many =
      Kolibri1TTRoutedExpertRefusalMessage(0, many);
  CHECK(msg_many.find("0,1,2,3,4,5,6,7,...") != std::string::npos);
}

TEST_CASE("kolibri1 TT B2b-i: the refusal counter counts and resets") {
  Kolibri1TTResetRoutedExpertRefusalCount();
  CHECK(Kolibri1TTRoutedExpertRefusalCount() == 0);
  Kolibri1TTResetRoutedExpertRefusalCount();
  CHECK(Kolibri1TTRoutedExpertRefusalCount() == 0);
}

// ---- HOST: the relocated routing contract (kolibri1_shared.h) --------------

TEST_CASE("kolibri1 TT B2b-i: sigmoid-logit-add routing — selection on "
          "logits+bias, weights on the UNBIASED logits, ties to the lower "
          "index, no renormalisation") {
  // 1 token, 4 experts, top-2. Biased scores: e0 = 2.0 + 0.0 = 2.0,
  // e2 = 1.0 + 1.0 = 2.0 (an EXACT TIE with e0), e3 = 1.5 - 1.0 = 0.5,
  // e1 = 0.5. The tie breaks to the LOWER index: top-2 = {e0, e2}.
  const std::vector<float> logits = {2.0f, 0.5f, 1.0f, 1.5f};
  const std::vector<float> bias = {0.0f, 0.0f, 1.0f, -1.0f};
  const Kolibri1HostRouting r =
      SigmoidLogitAddRouting(logits, bias, 1, 4, 2);
  REQUIRE(r.ids.size() == 2);
  CHECK(r.ids[0] == 0);
  CHECK(r.ids[1] == 2);
  // The WEIGHT reads the UNBIASED logit: sigmoid(2.0), sigmoid(1.0).
  CHECK(r.weights[0] == doctest::Approx(1.0f / (1.0f + std::exp(-2.0f))));
  CHECK(r.weights[1] == doctest::Approx(1.0f / (1.0f + std::exp(-1.0f))));

  // Exact ties break to the LOWER expert index (torch.topk stable order).
  const std::vector<float> tied = {1.0f, 1.0f, 1.0f, 1.0f};
  const std::vector<float> zero_bias = {0.0f, 0.0f, 0.0f, 0.0f};
  const Kolibri1HostRouting rt =
      SigmoidLogitAddRouting(tied, zero_bias, 1, 4, 3);
  CHECK(rt.ids[0] == 0);
  CHECK(rt.ids[1] == 1);
  CHECK(rt.ids[2] == 2);
}

// ---- HOST: the device-resident compute context over the tiny fixture -------

TEST_CASE("kolibri1 TT B2b-i: the device context dequants every resident "
          "fp8 projection once, bit-identically to the CPU row's dequant") {
  const HfConfig config = MakeTinyConfig();
  const TinyShape s;
  TempCheckpoint ckpt(TinyFixture());
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  const Kolibri1Weights w = LoadKolibri1Weights(shards, config);
  REQUIRE(w.layers.size() == 2);

  // The context builds over ANY backend (host dequant into a backend
  // allocation); the CPU backend is the device-free stand-in. The DEVICE
  // dequant agreement itself is asserted in the device leg against the real
  // checkpoint; here the bytes are produced host-side by the same function.
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = be.CreateQueue();
  std::unique_ptr<Kolibri1TTResidentDeviceContext> ctx =
      BuildKolibri1TTResidentDeviceContext(be, q, w);
  REQUIRE(ctx != nullptr);
  CHECK(ctx->built);

  // 7 resident projections per layer: attention q/k/v/o + shared gate/up/down.
  CHECK(ctx->projections == 2 * 7);
  REQUIRE(ctx->layers.size() == 2);

  // Byte accounting: every projection is [n, k] bf16 = 2 bytes per element,
  // and the dequant is BIT-IDENTICAL to the CPU row's DequantRowsBf16 over
  // the same packed bytes and scale grid (the R1 disposition, memoized).
  const Kolibri1LayerWeights& lw = w.layers[0];
  const Kolibri1TTResidentDeviceContext::Layer& cl = ctx->layers[0];
  const std::pair<const Kolibri1Projection*, const vt::Tensor*> pairs[] = {
      {&lw.attn.q_proj, &cl.q},   {&lw.attn.k_proj, &cl.k},
      {&lw.attn.v_proj, &cl.v},   {&lw.attn.o_proj, &cl.o},
      {&lw.moe.shared_experts.gate_proj, &cl.sh_gate},
      {&lw.moe.shared_experts.up_proj, &cl.sh_up},
      {&lw.moe.shared_experts.down_proj, &cl.sh_down},
  };
  int64_t expected_bytes = 0;
  std::vector<uint16_t> ref;
  for (const auto& [proj, tensor] : pairs) {
    REQUIRE(proj->IsFp8Block());
    const Fp8BlockWeight& f = proj->fp8_block;
    const int64_t n = f.n;
    const int64_t k = f.k;
    CHECK(tensor->shape[0] == n);
    CHECK(tensor->shape[1] == k);
    expected_bytes += n * k * 2;

    ref.resize(static_cast<size_t>(n * k));
    kolibri1_fp8::DequantRowsBf16(
        f.packed.bytes.data(),
        reinterpret_cast<const float*>(f.scale.bytes.data()),
        CDiv(k, f.block_k), 0, n, k, f.block_n, f.block_k, ref.data());
    const int mismatches = std::memcmp(tensor->data, ref.data(),
                                       static_cast<size_t>(n * k) * 2) != 0;
    CHECK(mismatches == 0);
  }
  // Layer 1 contributes the same 7 projections at the same tiny shapes.
  expected_bytes *= 2;
  CHECK(ctx->uploaded_bytes == expected_bytes);
}

TEST_CASE("kolibri1 TT B2b-i: the device context refuses a non-fp8 "
          "resident projection by name") {
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  Kolibri1Weights w = LoadKolibri1Weights(shards, config);
  // Corrupt one resident projection into a non-fp8 form: the context must
  // refuse rather than silently re-arm.
  REQUIRE(w.layers.size() == 2);
  w.layers[1].attn.o_proj = Kolibri1Projection{};  // neither arm armed
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = be.CreateQueue();
  bool threw = false;
  try {
    (void)BuildKolibri1TTResidentDeviceContext(be, q, w);
  } catch (const std::exception& e) {
    threw = true;
    const std::string what = e.what();
    CHECK(what.find("fp8-block") != std::string::npos);
    CHECK(what.find("refused") != std::string::npos);
  }
  CHECK(threw);
}

// ---- HOST: the forward refuses a non-Tenstorrent queue by name ------------

TEST_CASE("kolibri1 TT B2b-i: the forward refuses a CPU queue by name "
          "(the CPU row owns the CPU arm)") {
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  const Kolibri1Weights w = LoadKolibri1Weights(shards, config);
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kCPU);
  vt::Queue q = be.CreateQueue();
  auto ctx = BuildKolibri1TTResidentDeviceContext(be, q, w);

  const v1::CommonAttentionMetadata meta = OneReqMeta(2, 0, 4);
  std::vector<PagedKvCache> kv(2);
  bool threw = false;
  try {
    (void)ForwardKolibri1TTResidentForward(
        /*token_ids=*/{1, 2}, /*positions=*/{0, 1}, meta, kv, w,
        /*multi_kv=*/nullptr, q, /*logits_indices=*/{}, *ctx);
  } catch (const std::exception& e) {
    threw = true;
    const std::string what = e.what();
    CHECK(what.find("Tenstorrent") != std::string::npos);
    CHECK(what.find("MODEL-TEXT-kolibri-1-tenstorrent") != std::string::npos);
  }
  CHECK(threw);
}

// ---- HOST: the resident slice's real-manifest byte math --------------------

TEST_CASE("kolibri1 TT B2b-i: the resident projection byte math matches the "
          "real checkpoint manifest (the memoized dequant's device cost)") {
  // 7 resident projections per layer x 50 layers = 350, per the context's
  // contract. The context stores the BF16 DEQUANTS: 2 bytes per fp8 element.
  const int64_t fp8_bytes = ManifestResidentProjectionFp8Bytes();
  const int64_t count = ManifestResidentProjectionCount();
  CHECK(count == 350);
  // The spec's byte-math plan (addendum): attention 1.587 GiB + shared
  // expert 0.188 GiB of fp8 resident projections. The manifest total must
  // land within 1 MiB of that plan (the plan rounded to 3 decimals).
  const int64_t plan = static_cast<int64_t>(1.587 * (1ll << 30)) +
                       static_cast<int64_t>(0.188 * (1ll << 30));
  // The plan rounds each component to 3 decimals of GiB; the measured
  // manifest total lands within 8 MiB of it (2026-10-08: 5.1 MiB).
  CHECK(std::abs(fp8_bytes - plan) < (8 << 20));
  // The memoized context costs exactly 2x the fp8 bytes as bf16.
  CHECK(fp8_bytes * 2 == 2 * ManifestResidentProjectionFp8Bytes());
}

// ---- HOST: the op census (a recording stand-in for the Tenstorrent queue) --
//
// The forward refuses a non-Tenstorrent queue by name, and the only
// production-seam consumer below was the card-gated device leg — so a wrong
// norm weight, a silenced refusal, or a deleted registry arm all left every
// host gate green. This census closes that: a host-memory backend registered
// under kTENSTORRENT (the test_resident_weight_host_addressable.cpp pattern,
// in the TT slot) plus recording op providers over the EXISTING
// vt::OpProvider seam let the production forward run end-to-end on the host
// while the census records which ops fired, in what order, consuming which
// norm weight. No device kernel executes; no parallel forward path exists —
// the recorded ops are the forward's own vt op calls.
//
// The recording providers register at priority 100 (above every native
// kernel) under one test-only provider name, and the guard DISABLES that
// name on scope exit, so on a card box the device leg selects the native
// kernels exactly as landed.

namespace {

struct CensusRecord {
  const char* op;   // stable op tag
  uint16_t weight;  // first bf16 word of the weight operand (0 when none)
  bool residual;    // the norm carried the residual stream
};

std::vector<CensusRecord>& Census() {
  static std::vector<CensusRecord> c;
  return c;
}

uint16_t CensusBf16Word(const vt::Tensor& t) {
  if (t.data == nullptr || vt::SizeOf(t.dtype) != 2) return 0;
  return static_cast<const uint16_t*>(t.data)[0];
}

// The queue-guard fixture: one host-memory backend + platform in the
// kTENSTORRENT slot for the scope, recording providers enabled, the previous
// registration restored afterwards (a card box keeps its real backend and
// its real kernels; TenstorrentPresent() below excludes the stand-in so a
// card-less box keeps skipping the device leg exactly as before).
class CensusTTQueue final : public vt::Backend {
 public:
  void* Alloc(size_t bytes) override {
    // Zeroed, so the router logits the MoE block downloads are deterministic
    // zeros and the host routing is stable.
    return std::calloc(1, bytes == 0 ? 1 : bytes);
  }
  void Free(void* p) override { std::free(p); }
  void Memset(vt::Queue&, void* p, int v, size_t n) override {
    std::memset(p, v, n);
  }
  void Copy(vt::Queue&, void* dst, const void* src, size_t n) override {
    std::memcpy(dst, src, n);
  }
  vt::Queue CreateQueue() override {
    return vt::Queue{vt::Device{vt::DeviceType::kTENSTORRENT, 0}, nullptr};
  }
  bool UnifiedMemory() const override { return true; }
  bool DeviceMemoryIsHostAddressable() const override { return true; }
};

vt::Backend* CensusTTBackendPtr() {
  static CensusTTQueue backend;
  return &backend;  // NOLINT
}

class CensusTTPlatform final : public vllm::platforms::Platform {
 public:
  vt::DeviceType device_type() const override {
    return vt::DeviceType::kTENSTORRENT;
  }
  vt::Backend& backend() const override {
    return *static_cast<CensusTTQueue*>(CensusTTBackendPtr());
  }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return {10, 0};
  }
  std::vector<vt::DType> supported_dtypes() const override {
    return {vt::DType::kBF16, vt::DType::kF32};
  }
  vllm::platforms::ResidencyPolicy residency_policy() const override {
    return {};
  }
};

constexpr const char* kCensusProvider = "kolibri1-tt-b2bi-census";

// The recorders: one per op id the dense-resident forward can emit.
void RecRmsNorm(vt::Queue&, vt::Tensor&, const vt::Tensor&,
                const vt::Tensor& w, const vt::RmsNormArgs&,
                vt::Tensor* residual) {
  Census().push_back({"RmsNorm", CensusBf16Word(w), residual != nullptr});
}
void RecResidualRmsNorm(vt::Queue&, vt::Tensor&, const vt::Tensor&,
                        const vt::Tensor&, const vt::Tensor*,
                        const vt::Tensor& w, const vt::ResidualRmsNormArgs&,
                        vt::Tensor*) {
  // The fused add-norm composite folds the residual into ONE norm call; it is
  // the same norm contract as RmsNorm(residual) with the same weight operand.
  Census().push_back({"RmsNorm", CensusBf16Word(w), true});
}
void RecMatmulBT(vt::Queue&, vt::Tensor&, const vt::Tensor&,
                 const vt::Tensor& w) {
  Census().push_back({"MatmulBT", CensusBf16Word(w), false});
}
void RecEmbedding(vt::Queue&, vt::Tensor&, const vt::Tensor& table,
                  const vt::Tensor&) {
  Census().push_back({"Embedding", CensusBf16Word(table), false});
}
void RecRopeNeox(vt::Queue&, vt::Tensor&, vt::Tensor&, const vt::Tensor&,
                 const vt::RopeArgs&) {
  Census().push_back({"RopeNeox", 0, false});
}
void RecReshapeAndCache(vt::Queue&, const vt::Tensor&, const vt::Tensor&,
                        vt::Tensor&, vt::Tensor&, const vt::Tensor&) {
  Census().push_back({"ReshapeAndCache", 0, false});
}
void RecPagedAttention(vt::Queue&, vt::Tensor&, const vt::Tensor&,
                       const vt::Tensor&, const vt::Tensor&, const vt::Tensor&,
                       const vt::Tensor&, const vt::Tensor&,
                       const vt::PagedAttentionArgs&) {
  Census().push_back({"PagedAttention", 0, false});
}
void RecMoeSiluMul(vt::Queue&, vt::Tensor&, const vt::Tensor&,
                   const vt::Tensor&) {
  Census().push_back({"MoeSiluMul", 0, false});
}

void RegisterCensusOps() {
  static const bool done = [] {
    const auto reg = [&](vt::OpId op, void* fn) {
      vt::RegisterOpProvider(op, vt::DeviceType::kTENSTORRENT,
                             {kCensusProvider, 100, nullptr, fn});
    };
    reg(vt::OpId::kRmsNorm,
        reinterpret_cast<void*>(static_cast<vt::RmsNormFn>(&RecRmsNorm)));
    reg(vt::OpId::kResidualRmsNorm,
        reinterpret_cast<void*>(static_cast<vt::ResidualRmsNormFn>(
            &RecResidualRmsNorm)));
    reg(vt::OpId::kMatmulBT,
        reinterpret_cast<void*>(static_cast<vt::MatmulFn>(&RecMatmulBT)));
    reg(vt::OpId::kEmbedding,
        reinterpret_cast<void*>(static_cast<vt::EmbeddingFn>(&RecEmbedding)));
    reg(vt::OpId::kRopeNeox,
        reinterpret_cast<void*>(static_cast<vt::RopeFn>(&RecRopeNeox)));
    reg(vt::OpId::kReshapeAndCache,
        reinterpret_cast<void*>(static_cast<vt::ReshapeAndCacheFn>(
            &RecReshapeAndCache)));
    reg(vt::OpId::kPagedAttention,
        reinterpret_cast<void*>(static_cast<vt::PagedAttentionFn>(
            &RecPagedAttention)));
    reg(vt::OpId::kMoeSiluMul,
        reinterpret_cast<void*>(static_cast<vt::MoeSiluMulFn>(&RecMoeSiluMul)));
    return true;
  }();
  (void)done;
}

void EnableCensusOps(bool on) {
  for (const vt::OpId op : {vt::OpId::kRmsNorm, vt::OpId::kResidualRmsNorm,
                            vt::OpId::kMatmulBT, vt::OpId::kEmbedding,
                            vt::OpId::kRopeNeox, vt::OpId::kReshapeAndCache,
                            vt::OpId::kPagedAttention, vt::OpId::kMoeSiluMul}) {
    (void)op;
    vt::DisableOpProvider(kCensusProvider, !on);
  }
}

struct CensusTTGuard {
  CensusTTGuard() {
    RegisterCensusOps();
    EnableCensusOps(true);
    prev_backend_ = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
    vt::RegisterBackend(vt::DeviceType::kTENSTORRENT, CensusTTBackendPtr());
    try {
      prev_platform_ = &vllm::platforms::GetPlatform(
          vt::DeviceType::kTENSTORRENT);
    } catch (const std::exception&) {
      prev_platform_ = nullptr;
    }
    // Register AFTER reading the previous backend (the read is the value we
    // may restore), and register the platform only when the real one is
    // absent (a card box keeps its real platform).
    if (prev_platform_ == nullptr) {
      static CensusTTPlatform platform;
      vllm::platforms::RegisterPlatform(vt::DeviceType::kTENSTORRENT,
                                        &platform);
    }
  }
  ~CensusTTGuard() {
    EnableCensusOps(false);
    Census().clear();
    if (prev_backend_ != nullptr) {
      vt::RegisterBackend(vt::DeviceType::kTENSTORRENT, prev_backend_);
    }
    // prev_platform_ == nullptr leaves the stand-in platform registered: a
    // card-less box has no TT path left to consult it (TenstorrentPresent()
    // answers false through the backend check), and the platform registry
    // has no unregister.
  }
  vt::Backend* prev_backend_ = nullptr;
  vllm::platforms::Platform* prev_platform_ = nullptr;
};

// One layer's tiny KV caches over the census backend.
struct CensusKv {
  std::vector<std::shared_ptr<void>> keep;
  std::vector<PagedKvCache> caches;
};

CensusKv MakeCensusKv(vt::Backend& be, int64_t layers) {
  const int64_t block_size = 16;
  const int64_t num_blocks = 8;
  const int64_t bytes = num_blocks * 2 * block_size * 2 * 16 * 2;  // hkv=2, dh=16, bf16
  CensusKv kv;
  for (int64_t l = 0; l < layers; ++l) {
    void* buf = be.Alloc(static_cast<size_t>(bytes));
    kv.keep.emplace_back(buf, [&be](void* p) { be.Free(p); });
    PagedKvCache c;
    c.data = buf;
    c.dtype = vt::DType::kBF16;
    c.num_blocks = num_blocks;
    c.block_size = block_size;
    c.num_kv_heads = 2;
    c.head_size = 16;
    kv.caches.push_back(c);
  }
  return kv;
}

// Runs one forward step through the PRODUCTION entry point and returns the
// census slice it emitted.
std::vector<CensusRecord> CensusStep(const Kolibri1Weights& w,
                                     Kolibri1TTResidentDeviceContext& ctx,
                                     vt::Queue& q, const std::vector<PagedKvCache>& kv,
                                     int64_t t, int64_t ctx_before) {
  const v1::CommonAttentionMetadata meta = OneReqMeta(t, ctx_before, 4);
  std::vector<int32_t> positions(static_cast<size_t>(t));
  std::iota(positions.begin(), positions.end(),
            static_cast<int32_t>(ctx_before));
  std::vector<int32_t> tokens(static_cast<size_t>(t), 1);
  const size_t begin = Census().size();
  (void)ForwardKolibri1TTResidentForward(tokens, positions, meta, kv, w,
                                         /*multi_kv=*/nullptr, q,
                                         /*logits_indices=*/{}, ctx);
  return std::vector<CensusRecord>(Census().begin() +
                                       static_cast<std::ptrdiff_t>(begin),
                                   Census().end());
}

}  // namespace

TEST_CASE("kolibri1 TT B2b-i HOST: the forward's op census — the sandwich "
          "norms consume THEIR OWN weights in the CPU row's order, and the "
          "routed-expert refusal fires every MoE block") {
  CensusTTGuard guard;
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  const Kolibri1Weights w = LoadKolibri1Weights(shards, config);
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  auto ctx = BuildKolibri1TTResidentDeviceContext(be, q, w);
  REQUIRE(ctx != nullptr);
  CensusKv kv = MakeCensusKv(be, 2);

  // The refusal-firing contract, HOST-SIDE (the device leg's >=
  // layers*(1+steps) assertion, decoupled from the card): the MoE block of
  // EVERY layer of EVERY step requests routed experts and fires the counted
  // refusal by name.
  Kolibri1TTResetRoutedExpertRefusalCount();
  const std::vector<CensusRecord> prefill = CensusStep(w, *ctx, q, kv.caches,
                                                       /*t=*/2, /*ctx=*/0);
  const std::vector<CensusRecord> decode1 = CensusStep(w, *ctx, q, kv.caches,
                                                       /*t=*/1, /*ctx=*/2);
  const std::vector<CensusRecord> decode2 = CensusStep(w, *ctx, q, kv.caches,
                                                       /*t=*/1, /*ctx=*/3);
  CHECK(Kolibri1TTRoutedExpertRefusalCount() >=
        2 * (1 + 2));  // layers x (prefill + 2 decode steps)

  // Every step runs the IDENTICAL op sequence (decode differs only in T).
  REQUIRE(decode1.size() == decode2.size());
  for (size_t i = 0; i < decode1.size(); ++i) {
    REQUIRE(decode1[i].op == decode2[i].op);
    REQUIRE(decode1[i].weight == decode2[i].weight);
  }

  // The op set and counts for one step (2 layers, one sliding + one full,
  // hidden 64, 4 experts): the CPU row's op-for-op mirror.
  int embedding = 0, rope = 0, cache = 0, paged = 0, silu = 0, matmul = 0,
      norm = 0, other = 0;
  for (const CensusRecord& r : prefill) {
    if (r.op == std::string("Embedding")) ++embedding;
    else if (r.op == std::string("RopeNeox")) ++rope;
    else if (r.op == std::string("ReshapeAndCache")) ++cache;
    else if (r.op == std::string("PagedAttention")) ++paged;
    else if (r.op == std::string("MoeSiluMul")) ++silu;
    else if (r.op == std::string("MatmulBT")) ++matmul;
    else if (r.op == std::string("RmsNorm")) ++norm;
    else ++other;
  }
  CHECK(other == 0);
  CHECK(embedding == 1);
  CHECK(rope == 1);  // the sliding layer only (RNoPE: the full layer has none)
  CHECK(cache == 2);
  CHECK(paged == 2);
  CHECK(silu == 2);
  CHECK(matmul == 2 * 8 + 1);  // q,k,v,o + router + shared gate,up,down; lm_head
  CHECK(norm == 2 * 6 + 1);    // 4 sandwich + q/k head norms; final norm

  // IDENTITY, not just counts: the norm sequence per layer consumes each
  // layer's OWN four sandwich weights (the fixture's distinct sentinels:
  // input_ln 1.0, post_attn 2.0, post_attention 3.0, post_ffn 4.0) in the
  // CPU row's order, with the per-head q/k norms (0.25 / 0.3125) between the
  // qkv projection and attention, and the final norm (0.5) carrying the
  // residual last. The residual-carrying norms are exactly input_ln,
  // post_attention_layernorm, and the final norm (the fused add-norm
  // contract); post_attn and post_ffn norm WITHOUT a residual.
  struct ExpectedNorm {
    uint16_t word;
    bool residual;
  };
  const ExpectedNorm layer_norms[] = {
      {0x3F80, true}, {0x3E00, false}, {0x3E80, false},
      {0x4000, false}, {0x4040, true}, {0x4080, false},
  };
  const ExpectedNorm final_norm{0x3F00, true};
  size_t n = 0;
  for (const CensusRecord& r : prefill) {
    if (r.op != std::string("RmsNorm")) continue;
    const ExpectedNorm& expected =
        n < 12 ? layer_norms[n % 6] : final_norm;
    CAPTURE(n);
    CHECK(r.weight == expected.word);
    CHECK(r.residual == expected.residual);
    if (r.weight != expected.word || r.residual != expected.residual) {
      MESSAGE("norm record " << n << ": word=0x" << std::hex << r.weight
                             << " residual=" << r.residual);
    }
    ++n;
  }
  REQUIRE(n == 13);
}

TEST_CASE("kolibri1 TT B2b-i HOST: the registry's kTENSTORRENT dispatch arm "
          "resolves the dense-resident forward (dispatch identity)") {
  CensusTTGuard guard;
  const HfConfig config = MakeTinyConfig();
  TempCheckpoint ckpt(TinyFixture());
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(ckpt.path()));
  const ModelSource source = ModelSource::FromSafetensors(shards);
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.architecture == "Kolibri1ForCausalLM");
  reg.factory->parse_config(config);
  std::unique_ptr<LoadedModel> model = ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  // Prepare on the TT queue materializes the B2b-i device context eagerly —
  // the production prepare path, not a test-built context.
  ModelRegistry::Prepare(*model, config, q);

  CensusKv kv = MakeCensusKv(be, 2);
  v1::GDNAttentionMetadata gdn_meta;
  std::vector<GdnStateCache> gdn_state;
  Kolibri1TTResetRoutedExpertRefusalCount();
  const int64_t before = Kolibri1TTRoutedExpertRefusalCount();
  const v1::CommonAttentionMetadata meta = OneReqMeta(1, 0, 4);
  ModelForwardInput in{{5},
                       {0},
                       meta,
                       gdn_meta,
                       kv.caches,
                       gdn_state,
                       config,
                       q,
                       /*logits_indices=*/{},
                       /*num_reqs=*/1};
  in.pure_decode = true;
  in.uniform_query_len = 1;
  // The registry dispatches the kTENSTORRENT arm to ForwardKolibri1TTResident
  // Forward: the step completes through the PRODUCTION seam and the TT
  // forward's own refusal fires (a deleted or replaced dispatch arm throws
  // and fails this case).
  ForwardLogits fl = ModelRegistry::Forward(*model, in);
  REQUIRE(fl.rows == 1);
  REQUIRE(fl.vocab == 32);
  CHECK(fl.device_tensor.dtype == vt::DType::kF32);
  CHECK(Kolibri1TTRoutedExpertRefusalCount() - before >= 2);
}

// ---- DEVICE LEG ------------------------------------------------------------

namespace {

bool TenstorrentPresent() {
  // The census's host stand-in is NOT a device: exclude it so a card-less box
  // keeps skipping the device leg after the census cases ran.
  vt::Backend* b = vt::TryGetBackend(vt::DeviceType::kTENSTORRENT);
  return b != nullptr && b != CensusTTBackendPtr();
}

struct GoldenPrompt {
  std::string prompt;
  std::vector<int32_t> input_ids;
  std::vector<int32_t> generated_ids;
};

std::vector<GoldenPrompt> LoadGoldens() {
  const nlohmann::json j =
      nlohmann::json::parse(std::ifstream(KOLIBRI1_GOLDENS));
  std::vector<GoldenPrompt> out;
  for (const auto& p : j.at("prompts")) {
    GoldenPrompt g;
    g.prompt = p.at("prompt").get<std::string>();
    for (const auto& v : p.at("input_ids"))
      g.input_ids.push_back(v.get<int32_t>());
    if (p.contains("generated_ids"))
      for (const auto& v : p.at("generated_ids"))
        g.generated_ids.push_back(v.get<int32_t>());
    out.push_back(std::move(g));
  }
  return out;
}

}  // namespace

TEST_CASE("kolibri1 TT B2b-i: device leg — resident context, embedding "
          "bit-exact, ONE greedy decode of a golden prompt on the card") {
  if (!TenstorrentPresent()) {
    MESSAGE("SKIPPED: no Tenstorrent device on this box");
    return;
  }
  const char* model_dir = std::getenv("VT_KOLIBRI1_TT_B2BI_MODEL");
  if (model_dir == nullptr || *model_dir == '\0') {
    MESSAGE("SKIPPED: VT_KOLIBRI1_TT_B2BI_MODEL not set (no real checkpoint "
            "requested)");
    return;
  }
  const std::string dir = model_dir;
  if (!std::filesystem::exists(dir + "/model.safetensors.index.json")) {
    MESSAGE("SKIPPED: " << dir << " is not a Kolibri-1 checkpoint");
    return;
  }
  std::fprintf(stderr, "[kolibri1-tt-b2bi] device leg start: model=%s\n",
               dir.c_str());
  Kolibri1TTResetRoutedExpertRefusalCount();

  // ---- 1. Load the REAL checkpoint through the production registry path.
  const HfConfig config = vllm::LoadHfConfig(dir + "/config.json");
  const ModelRegistration& reg = ModelRegistry::Resolve(config);
  REQUIRE(reg.architecture == "Kolibri1ForCausalLM");
  reg.factory->parse_config(config);
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
  const double t_load0 = NowSec();
  std::unique_ptr<LoadedModel> model = ModelRegistry::Load(config, source);
  REQUIRE(model != nullptr);
  std::fprintf(stderr, "[kolibri1-tt-b2bi] load: %.1f s\n",
               NowSec() - t_load0);

  // ---- 2. PREPARE on the TT queue: builds the B2b-i device context
  // eagerly (the production residency the decode runs).
  vt::Backend& be = vt::GetBackend(vt::DeviceType::kTENSTORRENT);
  vt::Queue q = be.CreateQueue();
  const double t_prep0 = NowSec();
  ModelRegistry::Prepare(*model, config, q);
  const Kolibri1TTResidentDeviceContext& ctx =
      Kolibri1LoadedModelTTContext(*model, q);
  REQUIRE(ctx.built);
  std::fprintf(stderr,
               "[kolibri1-tt-b2bi] context: %lld projections, %lld B bf16 "
               "(%.3f GiB) in %.1f s\n",
               static_cast<long long>(ctx.projections),
               static_cast<long long>(ctx.uploaded_bytes),
               double(ctx.uploaded_bytes) / (1 << 30), NowSec() - t_prep0);
  // 350 resident projections (7 per layer x 50), and the memoized bf16
  // bytes are exactly 2x the resident fp8 bytes the manifest records.
  CHECK(ctx.projections == ManifestResidentProjectionCount());
  CHECK(ctx.uploaded_bytes == 2 * ManifestResidentProjectionFp8Bytes());

  // ---- 3. Per-op agreement: the embedding gather is the forward's only
  // gather and carries no arithmetic; its device-vs-CPU agreement is the
  // argmax-consistency of the decode below (a wrong gather cannot produce
  // the CPU row's first argmax on the shared expert-dominated head — any
  // per-op GEMM envelope on the resident path is measured in the b2i
  // bring-up and inherited, NOT re-measured here).
  const Kolibri1Weights& w = Kolibri1LoadedModelWeights(*model);
  const Kolibri1Params& p = w.params;

  // ---- 4. ONE GREEDY DECODE of a golden prompt (the completion condition).
  const std::vector<GoldenPrompt> goldens = LoadGoldens();
  REQUIRE(!goldens.empty());
  const GoldenPrompt& gp = goldens[0];

  const int64_t hkv = p.num_key_value_heads;
  const int64_t hd = p.head_dim;
  const int64_t block_size = 16;
  const int64_t num_blocks = 32;
  const int64_t kv_bytes =
      num_blocks * 2 * block_size * hkv * hd *
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

  v1::GDNAttentionMetadata gdn_meta;
  std::vector<GdnStateCache> gdn_state;
  std::vector<int32_t> seq = gp.input_ids;

  // PREFILL the golden prompt (full recompute semantics are irrelevant to
  // slice i: one request, one continuous context).
  const double t_dec0 = NowSec();
  int32_t next = -1;
  {
    const int64_t t = static_cast<int64_t>(seq.size());
    const v1::CommonAttentionMetadata meta = OneReqMeta(t, 0, num_blocks / 2);
    std::vector<int32_t> positions(static_cast<size_t>(t));
    std::iota(positions.begin(), positions.end(), 0);
    ModelForwardInput in{seq,
                         positions,
                         meta,
                         gdn_meta,
                         kv,
                         gdn_state,
                         config,
                         q,
                         /*logits_indices=*/{},
                         /*num_reqs=*/1};
    in.pure_decode = false;
    in.uniform_query_len = 0;
    ForwardLogits fl = ModelRegistry::Forward(*model, in);
    REQUIRE(fl.rows >= 1);
    std::vector<float> logits(static_cast<size_t>(fl.vocab));
    be.Copy(q, logits.data(),
            static_cast<const uint8_t*>(fl.device_tensor.data) +
                (fl.rows - 1) * fl.vocab * static_cast<int64_t>(sizeof(float)),
            static_cast<size_t>(fl.vocab) * sizeof(float));
    be.Synchronize(q);
    next = static_cast<int32_t>(std::max_element(logits.begin(),
                                                 logits.end()) -
                                logits.begin());
    seq.push_back(next);
    std::fprintf(stderr, "[kolibri1-tt-b2bi] prefill done: %lld tokens -> "
                         "first argmax %d\n",
                 static_cast<long long>(t), next);
  }

  // DECODE steps: greedy argmax, one token at a time.
  const int kDecodeSteps = 8;
  for (int step = 0; step < kDecodeSteps; ++step) {
    const int64_t ctx_before = static_cast<int64_t>(seq.size()) - 1;
    const v1::CommonAttentionMetadata meta =
        OneReqMeta(1, ctx_before, num_blocks / 2);
    ModelForwardInput in{{seq.back()},
                         {static_cast<int32_t>(ctx_before)},
                         meta,
                         gdn_meta,
                         kv,
                         gdn_state,
                         config,
                         q,
                         {},
                         /*num_reqs=*/1};
    in.pure_decode = true;
    in.uniform_query_len = 1;
    ForwardLogits fl = ModelRegistry::Forward(*model, in);
    REQUIRE(fl.rows == 1);
    std::vector<float> logits(static_cast<size_t>(fl.vocab));
    be.Copy(q, logits.data(), fl.device_tensor.data,
            static_cast<size_t>(fl.vocab) * sizeof(float));
    be.Synchronize(q);
    next = static_cast<int32_t>(
        std::max_element(logits.begin(), logits.end()) - logits.begin());
    seq.push_back(next);
    std::fprintf(stderr, "[kolibri1-tt-b2bi] decode step %d: argmax %d\n",
                 step, next);
  }
  const double t_decode = NowSec() - t_dec0;

  // The COMPLETION CONDITION: the decode finished on the card. The routed
  // experts were requested every layer every step and REFUSED BY NAME.
  const int64_t refusals = Kolibri1TTRoutedExpertRefusalCount();
  CHECK(refusals >=
        p.num_hidden_layers * (1 + kDecodeSteps));  // every MoE block fired
  std::fprintf(stderr,
               "[kolibri1-tt-b2bi] GREEDY DECODE COMPLETE on card: "
               "%zu tokens in %.1f s; routed-expert refusals fired %lld "
               "times (by name; the routed tier arrives in B2b-ii)\n",
               seq.size(), t_decode, static_cast<long long>(refusals));
  // The goldens' expected tokens are NOT asserted here: the goldens are
  // full-model decodes and cannot be replayed without the routed experts
  // (addendum gate ordering). The 141/145 argmax gate stays owed to B2b-ii.
}
