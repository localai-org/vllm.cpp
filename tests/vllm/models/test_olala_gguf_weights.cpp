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
