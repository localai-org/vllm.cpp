// Row NEW-CANDIDATE-PORTS W1 — the olala GGUF weights arm's census test:
// every tensor of the pinned artifacts maps to a role, and counts/shapes
// agree with the parsed config. CPU-only, NO forward (the forward still
// refuses by name). The pinned paths are the fixtures; each case SKIPs
// loudly when its path is absent (the test_new_candidate_parsers pattern).
#include <doctest/doctest.h>

#include <filesystem>
#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/olala_gguf_config.h"
#include "vllm/model_executor/models/olala_gguf_weights.h"
#include "vllm/transformers_utils/hf_config.h"

#ifndef OLALA_BF16_PATH
#define OLALA_BF16_PATH "/mnt/models/olala-v1-mini/Olala-v1.0-mini-7A1B-bf16.gguf"
#endif
#ifndef OLALA_Q6K_PATH
#define OLALA_Q6K_PATH "/mnt/models/olala-v1-mini/Olala-v1.0-mini-7A1B-q6_k.gguf"
#endif

namespace {

bool Present(const std::string& p) { return std::filesystem::exists(p); }

}  // namespace

TEST_CASE("olala: census maps all 989 bf16 tensors, 36 blocks, 29 M + 7 V") {
  if (!Present(OLALA_BF16_PATH)) {
    MESSAGE("SKIP: " << OLALA_BF16_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(OLALA_BF16_PATH);
  vllm::OlalaGgufCensus c = vllm::CensusOlalaGguf(gguf);
  CHECK(c.entries.size() == 989);
  CHECK(c.unmapped.empty());
  CHECK(c.num_layers == 36);
  CHECK(c.m_layers.size() == 29);
  CHECK(c.v_layers.size() == 7);
  CHECK(c.v_layers == std::vector<int64_t>({4, 9, 14, 19, 24, 29, 34}));
  CHECK(c.role_counts[vllm::OlalaTensorRole::kEmbed] == 1);
  CHECK(c.role_counts[vllm::OlalaTensorRole::kLmHead] == 1);
  CHECK(c.role_counts[vllm::OlalaTensorRole::kAttnOutput] == 36);
  CHECK(c.role_counts[vllm::OlalaTensorRole::kFfnUpExps] == 36);
  CHECK(c.role_counts[vllm::OlalaTensorRole::kSsmIn] == 29);
  CHECK(c.role_counts[vllm::OlalaTensorRole::kAttnQ] == 7);
  // bf16 artifact dtype census: 375 BF16 + 614 F32, nothing quantized.
  int64_t bf16 = 0, f32 = 0;
  for (const auto& e : c.entries) {
    if (e.ggml_type == 30) ++bf16;
    if (e.ggml_type == 0) ++f32;
  }
  CHECK(bf16 == 375);
  CHECK(f32 == 614);
}

TEST_CASE("olala: ValidateOlalaGguf accepts the pinned bf16 artifact") {
  if (!Present(OLALA_BF16_PATH)) {
    MESSAGE("SKIP: " << OLALA_BF16_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(OLALA_BF16_PATH);
  vllm::HfConfig cfg = vllm::OlalaHfConfigFromGguf(gguf);
  CHECK_NOTHROW(vllm::ValidateOlalaGguf(gguf, cfg));
  CHECK(cfg.hidden_size == 1536);
  CHECK(cfg.num_hidden_layers == 36);
  CHECK(cfg.num_experts == 256);
  CHECK(cfg.num_attention_heads == 48);
  CHECK(cfg.head_dim == 128);
}

TEST_CASE("olala: q6_k arm census — 303 Q6_K, 614 F32, 36 Q8_0, 36 BF16") {
  if (!Present(OLALA_Q6K_PATH)) {
    MESSAGE("SKIP: " << OLALA_Q6K_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(OLALA_Q6K_PATH);
  vllm::OlalaGgufCensus c = vllm::CensusOlalaGguf(gguf);
  CHECK(c.entries.size() == 989);
  CHECK(c.unmapped.empty());
  int64_t q6k = 0, f32 = 0, q8 = 0, bf16 = 0;
  for (const auto& e : c.entries) {
    if (e.ggml_type == 14) ++q6k;   // Q6_K
    if (e.ggml_type == 0) ++f32;     // F32
    if (e.ggml_type == 8) ++q8;      // Q8_0
    if (e.ggml_type == 30) ++bf16;   // BF16
  }
  CHECK(q6k == 303);
  CHECK(f32 == 614);
  CHECK(q8 == 36);    // the routed up-projection stays Q8_0 in the q6_k arm
  CHECK(bf16 == 36);  // the latent projections stay BF16
  vllm::HfConfig cfg = vllm::OlalaHfConfigFromGguf(gguf);
  CHECK_NOTHROW(vllm::ValidateOlalaGguf(gguf, cfg));
}

TEST_CASE("olala: the forward still refuses by name (W2+ owns the forward)") {
  CHECK(vllm::OlalaForwardRefusal().find("NEW-CANDIDATE-PORTS") !=
        std::string::npos);
}

// ---------------------------------------------------------------------------
// W2b: the model registers, the bf16 GGUF arm loads through the production
// registry seam, the hybrid KV spec publishes 7 V + 29 M layers, and the
// forward boundary refuses naming the unported ops.
// ---------------------------------------------------------------------------
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/olala.h"

TEST_CASE("olala W2b: registry resolves OlalaForCausalLM and the bf16 arm loads via ModelRegistry::Load") {
  if (!Present(OLALA_BF16_PATH)) {
    MESSAGE("SKIP: " << OLALA_BF16_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(OLALA_BF16_PATH);
  vllm::HfConfig cfg = vllm::OlalaHfConfigFromGguf(gguf);
  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(cfg);
  CHECK(reg.architecture == "OlalaForCausalLM");
  reg.factory->parse_config(cfg);
  std::unique_ptr<vllm::LoadedModel> model = vllm::ModelRegistry::Load(
      cfg, vllm::ModelSource::FromGguf(gguf, vt::DeviceType::kCPU));
  REQUIRE(model != nullptr);

  const vllm::OlalaParams p = vllm::ParseOlalaParams(cfg);
  CHECK(p.layers == 36);
  CHECK(p.layer_types.size() == 36);
  CHECK(p.window == 32768);
  CHECK(p.softcap > 149.9);
  CHECK(p.softcap < 150.1);
  CHECK(p.experts == 256);
  CHECK(p.top_k == 6);
  CHECK(p.latent == 384);
  CHECK(p.expert_ff == 768);
  CHECK(p.shared_ff == 1536);
  CHECK(p.routed_scale > 2.449);
  CHECK(p.routed_scale < 2.451);  // f32-stored 2.450000047683716
  CHECK(p.ssm_inner == 3072);
  CHECK(p.d_state == 128);
  CHECK(p.mimo_dim == 4);
  CHECK(p.headdim == 64);
  CHECK(p.tpa_rank == 4);
  CHECK(p.rope_angles == 32);

  // The hybrid KV spec: one sliding-window FA group over the 7 V layers, one
  // MambaSpec recurrent group over the 29 M layers.
  vllm::v1::KVCacheConfig kv = vllm::ModelRegistry::MakeKVCache(*model, cfg, 32, 8);
  REQUIRE(kv.kv_cache_groups.size() == 2);
  size_t v_names = 0, m_names = 0;
  for (const auto& g : kv.kv_cache_groups) {
    if (g.kv_cache_spec->kind() == vllm::v1::KVCacheSpecKind::kSlidingWindow)
      v_names = g.layer_names.size();
    if (g.kv_cache_spec->kind() == vllm::v1::KVCacheSpecKind::kMamba)
      m_names = g.layer_names.size();
  }
  CHECK(v_names == 7);
  CHECK(m_names == 29);

  // The forward boundary names the unported ops.
  const std::string refusal = vllm::OlalaForwardRefusalOps();
  CHECK(refusal.find("mamba3-mimo") != std::string::npos);
  CHECK(refusal.find("differential") != std::string::npos);
  CHECK(refusal.find("geodesic") != std::string::npos);
  CHECK(refusal.find("MoE") != std::string::npos);
  CHECK(refusal.find("NEW-CANDIDATE-PORTS") != std::string::npos);
}

TEST_CASE("olala W2b: weights land in the struct with the census shapes; layer kinds match layer_types") {
  if (!Present(OLALA_BF16_PATH)) {
    MESSAGE("SKIP: " << OLALA_BF16_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(OLALA_BF16_PATH);
  vllm::HfConfig cfg = vllm::OlalaHfConfigFromGguf(gguf);
  vllm::OlalaWeights w = vllm::LoadOlalaGgufWeights(gguf, cfg);
  REQUIRE(w.layers.size() == 36);
  CHECK(w.embed.rank == 2);
  CHECK(w.embed.shape[0] == 151936);
  CHECK(w.embed.shape[1] == 1536);
  CHECK(w.head.rank == 2);
  CHECK(w.head.shape[0] == 151936);
  CHECK(w.head.shape[1] == 1536);
  // M layer 0: ssm set present with the census shapes; V layer 4: attn set.
  const vllm::OlalaLayerWeights& m0 = w.layers[0];
  CHECK(m0.is_m);
  CHECK(m0.ssm_in.rank == 2);
  CHECK(m0.ssm_in.shape[0] == 6288);
  CHECK(m0.ssm_in.shape[1] == 1536);
  CHECK(m0.ssm_mimo_x.rank == 3);
  CHECK(m0.ssm_mimo_x.shape[0] == 48);
  CHECK(m0.ssm_mimo_x.shape[1] == 4);
  CHECK(m0.ssm_mimo_x.shape[2] == 64);
  CHECK(m0.router_weight.rank == 2);
  CHECK(m0.router_weight.shape[0] == 256);
  CHECK(m0.router_weight.shape[1] == 1536);
  CHECK(m0.router_weight.dtype == vt::DType::kF32);
  CHECK(m0.ffn_up_exps.rank == 3);
  CHECK(m0.ffn_up_exps.shape[0] == 256);
  CHECK(m0.ffn_up_exps.shape[1] == 768);
  CHECK(m0.ffn_up_exps.shape[2] == 384);
  const vllm::OlalaLayerWeights& v4 = w.layers[4];
  CHECK_FALSE(v4.is_m);
  CHECK(v4.attn_q.rank == 2);
  CHECK(v4.attn_q.shape[0] == 6144);
  CHECK(v4.attn_q.shape[1] == 1536);
  CHECK(v4.attn_output.shape[0] == 1536);
  CHECK(v4.attn_output.shape[1] == 4608);
  CHECK(v4.attn_softmax_scaler.rank == 1);
  CHECK(v4.attn_softmax_scaler.shape[0] == 48);
  CHECK(m0.attn_output.shape[1] == 3072);
}

TEST_CASE("olala W2b: the q6_k arm refuses by name at load (keep-quant decode owed to W2c)") {
  if (!Present(OLALA_Q6K_PATH)) {
    MESSAGE("SKIP: " << OLALA_Q6K_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(OLALA_Q6K_PATH);
  vllm::HfConfig cfg = vllm::OlalaHfConfigFromGguf(gguf);
  CHECK_THROWS_WITH(vllm::LoadOlalaGgufWeights(gguf, cfg),
                    doctest::Contains("refuse BY NAME"));
}
