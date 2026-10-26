// Flint TT keep-quant device arm — the loader-arm gate.
// (.agents/specs/flint-keepquant.md, row MODEL-TEXT-FLINT canonical, issue
// ISSUE-LOCAL-01M5WCPZC4TWR9GZVJHH73E1XE.)
//
// RED FIRST: at the base of the branch the flint GGUF arm has NO keep-quant
// path — LoadFlintKeepQuantFromGguf does not exist and every weight lands
// bf16-expanded — so this file does not compile at the base (the recorded
// red) and, once the symbol exists, the dtype/byte assertions below are red
// against the bf16 arm.
//
// THE ARM IS OPT-IN THIS WAVE: `VT_FLINT_KEEP_QUANT=1`. The bf16 arm stays
// the default because test_flint_tt_w0.cpp pins the bf16 arm's
// adjudicated-divergent greedy chain; flipping the default re-pins that gate
// with its own per-config chain evidence (the W3 matmul-arm precedent) and
// is the recorded default-flip owed in the spec.
//
// Legs:
//   1. synthetic container: the admitted encoding keeps VERBATIM blocks in
//      [N, K] nk order with the raw byte length; an inadmissible encoding
//      (Q4_0 — no Tenstorrent decode) refuses BY NAME; the env off switch
//      restores the bf16 arm.
//   2. the published flint-Q4_K_M container: the admission census
//      (admitted vs refused-by-name per encoding) and the re-derived budget
//      math — the quantized expert tier RESIDES instead of streaming
//      70.5 GiB.
#include <doctest/doctest.h>

#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <map>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/flint.h"
#include "vllm/model_executor/models/flint_gguf_weights.h"
#include "vllm/model_executor/models/flint_tt.h"
#include "vt/dtype.h"

#ifndef FLINT_GGUF_PATH
#define FLINT_GGUF_PATH "/mnt/models/flint-GGUF/flint-Q4_K_M.gguf"
#endif

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

namespace {

constexpr uint32_t kGgmlF32 = 0;
constexpr uint32_t kGgmlQ4_0 = 2;
constexpr uint32_t kGgmlQ4_K = 12;
// Q4_K: 144 bytes per 256-element block.
constexpr int64_t kH = 256, kV = 32, kHq = 4, kKv = 4, kDh = 64;
constexpr int64_t kQ = kHq * kDh, kKvDim = kKv * kDh;
constexpr int64_t kLayers = 3, kDenseInter = 256;
constexpr int64_t kE = 4, kI = 256, kIs = 256;

std::string Q4KData(int64_t rows, int64_t k) {
  const int64_t nblk = k / 256;
  return std::string(static_cast<size_t>(rows) * static_cast<size_t>(nblk) *
                         144,
                     '\0');
}

std::string F32Data(int64_t n, float v) {
  std::string s(static_cast<size_t>(n) * 4, '\0');
  for (int64_t i = 0; i < n; ++i)
    std::memcpy(s.data() + static_cast<size_t>(i) * 4, &v, 4);
  return s;
}

void AddQ4K(GgufModelBuilder& b, const std::string& name, int64_t out,
            int64_t in) {
  b.AddTensor(name, {static_cast<uint64_t>(in), static_cast<uint64_t>(out)},
              kGgmlQ4_K, Q4KData(out, in));
}

void AddQ4K3D(GgufModelBuilder& b, const std::string& name, int64_t e,
              int64_t out, int64_t in) {
  // The real container's expert tower: gguf dims {in, out, E} (ne order),
  // E planes of [out, in] rows — the layout the loader's tower path reads.
  b.AddTensor(name,
              {static_cast<uint64_t>(in), static_cast<uint64_t>(out),
               static_cast<uint64_t>(e)},
              kGgmlQ4_K, Q4KData(e * out, in));
}

void AddF32T(GgufModelBuilder& b, const std::string& name,
             const std::vector<int64_t>& torch_shape) {
  int64_t n = 1;
  for (int64_t d : torch_shape) n *= d;
  std::vector<uint64_t> dims;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it)
    dims.push_back(static_cast<uint64_t>(*it));
  b.AddTensor(name, dims, kGgmlF32, F32Data(n, 0.5F));
}

// A tiny valid flint GGUF: 3 layers (0-1 dense), Q4_K projections
// (every K is a whole number of 256-element blocks), F32 norms/sinks/router.
GgufModelBuilder SyntheticFlint(uint32_t q_type = kGgmlQ4_K) {
  GgufModelBuilder b;
  b.AddKv(StrKv("general.architecture", "flint"));
  b.AddKv(U32Kv("flint.embedding_length", static_cast<uint32_t>(kH)));
  b.AddKv(U32Kv("flint.block_count", static_cast<uint32_t>(kLayers)));
  b.AddKv(U32Kv("flint.attention.head_count", static_cast<uint32_t>(kHq)));
  b.AddKv(U32Kv("flint.attention.head_count_kv", static_cast<uint32_t>(kKv)));
  b.AddKv(U32Kv("flint.attention.key_length", static_cast<uint32_t>(kDh)));
  b.AddKv(U32Kv("flint.attention.sliding_window", 8));
  b.AddKv(F32Kv("flint.rope.freq_base", 10000.0F));
  b.AddKv(F32Kv("flint.attention.layer_norm_rms_epsilon", 1e-6F));
  b.AddKv(U32Kv("flint.expert_count", static_cast<uint32_t>(kE)));
  b.AddKv(U32Kv("flint.expert_used_count", 2));
  b.AddKv(U32Kv("flint.expert_feed_forward_length", static_cast<uint32_t>(kI)));
  b.AddKv(U32Kv("flint.expert_shared_feed_forward_length",
                static_cast<uint32_t>(kIs)));
  b.AddKv(U32Kv("flint.expert_gating_func", 2));
  b.AddKv(U32Kv("flint.expert_weights_norm", 1));
  b.AddKv(F32Kv("flint.expert_weights_scale", 1.0F));
  b.AddKv(U32Kv("flint.leading_dense_block_count", 2));
  b.AddKv(U32Kv("flint.feed_forward_length",
                static_cast<uint32_t>(kDenseInter)));
  {
    std::vector<bool> sliding(static_cast<size_t>(kLayers), false);
    std::vector<bool> rope(static_cast<size_t>(kLayers), false);
    b.AddKv(gguf_test::BoolArrayKv("flint.attention.sliding_window_pattern",
                                   sliding));
    b.AddKv(gguf_test::BoolArrayKv("flint.attention.rope_pattern", rope));
  }
  const auto blk = [](int64_t l, const char* s) {
    return "blk." + std::to_string(l) + "." + s;
  };
  AddQ4K(b, "token_embd.weight", kV, kH);   // stays the bf16 arm (gather)
  AddQ4K(b, "output.weight", kV, kH);
  AddF32T(b, "output_norm.weight", {kH});
  for (int64_t l = 0; l < kLayers; ++l) {
    AddF32T(b, blk(l, "attn_norm.weight"), {kH});
    AddF32T(b, blk(l, "ffn_norm.weight"), {kH});
    b.AddTensor(blk(l, "attn_q.weight"), {kH, kQ}, q_type, Q4KData(kQ, kH));
    AddQ4K(b, blk(l, "attn_k.weight"), kKvDim, kH);
    AddQ4K(b, blk(l, "attn_v.weight"), kKvDim, kH);
    AddQ4K(b, blk(l, "attn_output.weight"), kH, kQ);
    AddF32T(b, blk(l, "attn_q_norm.weight"), {kDh});
    AddF32T(b, blk(l, "attn_k_norm.weight"), {kDh});
    AddF32T(b, blk(l, "attn_sinks.weight"), {kHq});
    if (l < 2) {
      AddQ4K(b, blk(l, "ffn_gate.weight"), kDenseInter, kH);
      AddQ4K(b, blk(l, "ffn_up.weight"), kDenseInter, kH);
      AddQ4K(b, blk(l, "ffn_down.weight"), kH, kDenseInter);
    } else {
      AddF32T(b, blk(l, "ffn_gate_inp.weight"), {kE, kH});
      AddQ4K3D(b, blk(l, "ffn_gate_exps.weight"), kE, kI, kH);
      AddQ4K3D(b, blk(l, "ffn_up_exps.weight"), kE, kI, kH);
      AddQ4K3D(b, blk(l, "ffn_down_exps.weight"), kE, kH, kI);
      AddQ4K(b, blk(l, "ffn_gate_shexp.weight"), kIs, kH);
      AddQ4K(b, blk(l, "ffn_up_shexp.weight"), kIs, kH);
      AddQ4K(b, blk(l, "ffn_down_shexp.weight"), kH, kIs);
    }
  }
  return b;
}

struct EnvOn {
  EnvOn() { setenv("VT_FLINT_KEEP_QUANT", "1", 1); }
  ~EnvOn() { unsetenv("VT_FLINT_KEEP_QUANT"); }
  EnvOn(const EnvOn&) = delete;
  EnvOn& operator=(const EnvOn&) = delete;
};

int64_t GgmlBlockBytes(uint32_t t) {
  vt::DType dt = vt::DType::kF32;
  if (!vt::BlockDTypeFromGgmlTypeId(t, &dt)) return -1;
  return static_cast<int64_t>(vt::BlockBytes(dt));
}

}  // namespace

TEST_CASE("flint keepquant: the arm keeps verbatim [N,K] blocks") {
  EnvOn on;
  TempFile f(SyntheticFlint().Build());
  vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const auto config = vllm::FlintHfConfigFromGguf(g);
  const vllm::FlintWeights w = vllm::LoadFlintKeepQuantFromGguf(g, config);

  // q_proj: [N=64, K=256] verbatim Q4_K blocks, nk (disk order, no
  // transpose — the MatmulBTQuant orientation).
  const auto& q = w.layers[0].attn.q_proj.w;
  CHECK(q.dtype == vt::DType::kQ4_K);
  CHECK(q.rank == 2);
  CHECK(q.shape[0] == kQ);
  CHECK(q.shape[1] == kH);
  CHECK(q.nk);
  CHECK(static_cast<int64_t>(q.bytes.size()) ==
        kQ * (kH / 256) * GgmlBlockBytes(kGgmlQ4_K));

  // A routed expert cut: whole rows out of the verbatim tower.
  const auto& ex = w.layers[2].moe.experts[1].gate_proj.w;
  CHECK(ex.dtype == vt::DType::kQ4_K);
  CHECK(ex.shape[0] == kI);
  CHECK(ex.shape[1] == kH);
  CHECK(static_cast<int64_t>(ex.bytes.size()) ==
        kI * (kH / 256) * GgmlBlockBytes(kGgmlQ4_K));

  // Float-carrying tensors keep today's path: the embedding expands to bf16
  // (the TT gather of quant rows is not registered), F32 norms land bf16.
  CHECK(w.embed_tokens.dtype == vt::DType::kBF16);
  CHECK(w.layers[0].attn.sinks.dtype == vt::DType::kBF16);

  // The plan's budget math is re-derived: the quantized expert tier counts
  // RAW bytes and RESIDES inside the default budget (bf16 arm: ~50 GiB
  // streamed; here: a few hundred KiB).
  const vllm::FlintTTStagingPlan plan = vllm::PlanFlintTTStaging(w);
  std::fprintf(stderr,
               "[flint-keepquant] synthetic: resident=%lld B expert_tier=%lld "
               "B streamed/layer=%lld B\n",
               static_cast<long long>(plan.resident_bytes),
               static_cast<long long>(plan.expert_tier_bytes),
               static_cast<long long>(plan.streamed_bytes_per_layer));
  CHECK(plan.expert_tier_bytes ==
        static_cast<int64_t>(w.layers[2].moe.experts.size()) *
            (w.layers[2].moe.experts[0].gate_proj.w.bytes.size() +
             w.layers[2].moe.experts[0].up_proj.w.bytes.size() +
             w.layers[2].moe.experts[0].down_proj.w.bytes.size()));
  CHECK(plan.expert_tier_bytes < int64_t(1) << 20);
  CHECK(plan.resident_bytes + plan.expert_tier_bytes <=
        int64_t(32) << 30);
}

TEST_CASE("flint keepquant: an inadmissible encoding refuses BY NAME") {
  EnvOn on;
  TempFile f(SyntheticFlint(kGgmlQ4_0).Build());
  vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const auto config = vllm::FlintHfConfigFromGguf(g);
  bool refused = false;
  try {
    (void)vllm::LoadFlintKeepQuantFromGguf(g, config);
  } catch (const std::exception& e) {
    refused = true;
    const std::string what = e.what();
    CHECK_MESSAGE(what.find("q4_0") != std::string::npos,
                  "the refusal must name the encoding: " << what);
    CHECK_MESSAGE(what.find("attn_q.weight") != std::string::npos,
                  "the refusal must name the tensor: " << what);
  }
  CHECK(refused);
}

TEST_CASE("flint keepquant: the env off switch restores the bf16 arm") {
  TempFile f(SyntheticFlint().Build());
  vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const auto config = vllm::FlintHfConfigFromGguf(g);
  unsetenv("VT_FLINT_KEEP_QUANT");
  const vllm::FlintWeights w = vllm::LoadFlintKeepQuantFromGguf(g, config);
  CHECK(w.layers[0].attn.q_proj.w.dtype == vt::DType::kBF16);
  CHECK(w.layers[2].moe.experts[0].gate_proj.w.dtype == vt::DType::kBF16);
}

TEST_CASE("flint keepquant: the published Q4_K_M census and budget") {
  if (!std::filesystem::exists(FLINT_GGUF_PATH)) {
    MESSAGE("SKIP: " << FLINT_GGUF_PATH << " not mounted");
    return;
  }
  EnvOn on;
  vllm::GgufFile g = vllm::GgufFile::Open(FLINT_GGUF_PATH);
  const auto config = vllm::FlintHfConfigFromGguf(g);
  const vllm::FlintWeights w = vllm::LoadFlintKeepQuantFromGguf(g, config);

  // The census: weights by encoding, admitted vs kept-on-the-bf16-arm.
  std::map<std::string, int64_t> kept, expanded;
  int64_t weight_tensors = 0;
  const auto count = [&](const vllm::OwnedTensor& t) {
    ++weight_tensors;
    if (vt::IsBlockQuant(t.dtype))
      ++kept[vt::Name(t.dtype)];
    else
      ++expanded[vt::Name(t.dtype)];
  };
  count(w.embed_tokens);
  count(w.lm_head);
  for (const auto& lw : w.layers) {
    count(lw.attn.q_proj.w);
    count(lw.attn.k_proj.w);
    count(lw.attn.v_proj.w);
    count(lw.attn.o_proj.w);
    if (!lw.moe.experts.empty()) {
      count(lw.moe.router_gate);
      for (const auto& ex : lw.moe.experts) {
        count(ex.gate_proj.w);
        count(ex.up_proj.w);
        count(ex.down_proj.w);
      }
      count(lw.moe.shared_expert.gate_proj.w);
      count(lw.moe.shared_expert.up_proj.w);
      count(lw.moe.shared_expert.down_proj.w);
    } else {
      count(lw.dense_mlp.gate_proj.w);
      count(lw.dense_mlp.up_proj.w);
      count(lw.dense_mlp.down_proj.w);
    }
  }
  std::fprintf(stderr, "[flint-keepquant] census over %lld weight tensors\n",
               static_cast<long long>(weight_tensors));
  for (const auto& [name, n] : kept)
    std::fprintf(stderr, "[flint-keepquant]   kept      %-8s %lld\n",
                 name.c_str(), static_cast<long long>(n));
  for (const auto& [name, n] : expanded)
    std::fprintf(stderr, "[flint-keepquant]   bf16-arm  %-8s %lld\n",
                 name.c_str(), static_cast<long long>(n));
  CHECK(!kept.empty());

  // THE HEADLINE: the quantized expert tier is a few GiB, not 70.5 GiB, and
  // the whole model RESIDES inside the default 32 GiB budget.
  const vllm::FlintTTStagingPlan plan = vllm::PlanFlintTTStaging(w);
  std::fprintf(stderr,
               "[flint-keepquant] Q4_K_M: resident=%lld B (%.3f GiB) "
               "expert_tier=%lld B (%.3f GiB) streamed/layer=%lld B\n",
               static_cast<long long>(plan.resident_bytes),
               double(plan.resident_bytes) / double(1 << 30),
               static_cast<long long>(plan.expert_tier_bytes),
               double(plan.expert_tier_bytes) / double(1 << 30),
               static_cast<long long>(plan.streamed_bytes_per_layer));
  CHECK(plan.expert_tier_bytes < int64_t(32) << 30);
  CHECK(plan.resident_bytes + plan.expert_tier_bytes <= int64_t(32) << 30);
}
