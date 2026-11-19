// Row NEW-CANDIDATE-PARSERS — the CPU-only tests for the three parse-only
// config builders: olala and hy_v4_flash (GGUF) and maxwell_moe (config.json +
// safetensors headers). NO device, NO forwards: each refusal test asserts the
// FORWARD-BOUNDARY message naming row NEW-CANDIDATE-PORTS.
//
// The three pinned artifact paths are the fixtures. Each test SKIPs loudly
// when its path is absent, so CI without /mnt/models stays green (the
// test_flint_gguf pattern).
#include <doctest/doctest.h>

#include <cstdlib>
#include <filesystem>
#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/hy_v4_flash_gguf_config.h"
#include "vllm/model_executor/models/maxwell_config.h"
#include "vllm/model_executor/models/olala_gguf_config.h"
#include "vllm/transformers_utils/hf_config.h"

#ifndef OLALA_GGUF_PATH
#define OLALA_GGUF_PATH "/mnt/models/olala-v1-mini/Olala-v1.0-mini-7A1B-bf16.gguf"
#endif

#ifndef HY4_GGUF_PATH
#define HY4_GGUF_PATH \
  "/mnt/models/hy4-flash-gguf/Hy4-flash-Q4_K_M-00001-of-00002.gguf"
#endif

#ifndef MAXWELL_DIR
#define MAXWELL_DIR "/mnt/models/maxwell-v1-flash/snapshot"
#endif

namespace {

bool Present(const std::string& p) { return std::filesystem::exists(p); }

}  // namespace

TEST_CASE("olala: the builder maps the pinned artifact's standard fields") {
  if (!Present(OLALA_GGUF_PATH)) {
    MESSAGE("SKIP: " << OLALA_GGUF_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(OLALA_GGUF_PATH);
  REQUIRE(vllm::IsOlalaGguf(gguf));
  const vllm::HfConfig c = vllm::OlalaHfConfigFromGguf(gguf);
  CHECK(c.model_type == "olala");
  CHECK(c.num_hidden_layers == 36);
  CHECK(c.hidden_size == 1536);
  CHECK(c.num_attention_heads == 48);
  CHECK(c.num_key_value_heads == 12);
  CHECK(c.head_dim == 128);
  CHECK(c.max_position_embeddings == 100000);
  CHECK(c.sliding_window.value_or(0) == 32768);
  CHECK(c.num_experts == 256);
  CHECK(c.num_experts_per_tok == 6);
  CHECK(c.vocab_size == 151936);  // token_embd.weight shape[0], not a KV
}

TEST_CASE("olala: unmapped KVs are carried verbatim") {
  if (!Present(OLALA_GGUF_PATH)) {
    MESSAGE("SKIP: " << OLALA_GGUF_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(OLALA_GGUF_PATH);
  const vllm::HfConfig c = vllm::OlalaHfConfigFromGguf(gguf);
  REQUIRE(c.raw.contains("gguf_unmapped"));
  const auto& u = c.raw["gguf_unmapped"];
  // Values straight off the committed walk (olala-kvs.txt).
  CHECK(u.contains("olala.ssm.state_size"));
  CHECK(u["olala.ssm.state_size"].get<int>() == 128);
  CHECK(u["olala.tpa_rank"].get<int>() == 4);
  CHECK(u["olala.layer_types"].get<std::string>().size() == 36u);
  CHECK(u["olala.layer_types"].get<std::string>()[4] == 'V');
  CHECK(u["olala.layer_types"].get<std::string>()[35] == 'M');
  CHECK(u.contains("general.size_label"));
  CHECK(u.contains("tokenizer.ggml.pre"));
  // The mapped keys are NOT in the carry.
  CHECK_FALSE(u.contains("olala.block_count"));
  CHECK_FALSE(u.contains("olala.embedding_length"));
  // The count field agrees with the carry's size.
  CHECK(c.raw["gguf_unmapped_count"].get<size_t>() == u.size());
}

TEST_CASE("olala: wrong architecture refuses by name") {
  if (!Present(OLALA_GGUF_PATH)) {
    MESSAGE("SKIP: " << OLALA_GGUF_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(OLALA_GGUF_PATH);
  // Feed a foreign builder this family's file: each builder checks ITS arch.
  CHECK_THROWS_WITH(vllm::HyV4FlashHfConfigFromGguf(gguf),
                    doctest::Contains("hy_v4_flash"));
}

TEST_CASE("olala: the forward refusal names the port row") {
  const std::string r = vllm::OlalaForwardRefusal();
  CHECK(r.find("OlalaHfConfigFromGguf") != std::string::npos);
  CHECK(r.find("NEW-CANDIDATE-PORTS") != std::string::npos);
}

TEST_CASE("hy_v4_flash: the builder maps the pinned artifact's fields") {
  // Shard 2 carries `per_layer_token_embd.weight` at a fork-specific ggml
  // type id (43) the reader does not tabulate, so the SPLIT open refuses
  // before the parser runs. The KV metadata and token_embd live in shard 1
  // (the committed walk), and the parse is a header parse, so this fixture
  // opens shard 1 ALONE. RECORDED DEVIATION: the port row owes the reader a
  // type-43 entry before a full merged open can load.
  setenv("VT_GGUF_NO_SPLIT", "1", /*overwrite=*/1);
  if (!Present(HY4_GGUF_PATH)) {
    MESSAGE("SKIP: " << HY4_GGUF_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(HY4_GGUF_PATH);
  REQUIRE(vllm::IsHyV4FlashGguf(gguf));
  const vllm::HfConfig c = vllm::HyV4FlashHfConfigFromGguf(gguf);
  CHECK(c.model_type == "hy_v4_flash");
  CHECK(c.num_hidden_layers == 43);
  CHECK(c.hidden_size == 3072);
  CHECK(c.intermediate_size == 13824);
  CHECK(c.num_key_value_heads == 1);
  CHECK(c.num_attention_heads == 32);  // head_count[0]; the array is carried
  CHECK(c.head_dim == 576);
  CHECK(c.max_position_embeddings == 1048576);
  CHECK(c.sliding_window.value_or(0) == 512);
  CHECK(c.num_experts == 512);
  CHECK(c.num_experts_per_tok == 16);
  CHECK(c.vocab_size == 120832);
}

TEST_CASE("hy_v4_flash: unmapped KVs (indexer/Engram/PLE) carried verbatim") {
  if (!Present(HY4_GGUF_PATH)) {
    MESSAGE("SKIP: " << HY4_GGUF_PATH << " not mounted");
    return;
  }
  vllm::GgufFile gguf = vllm::GgufFile::Open(HY4_GGUF_PATH);
  const vllm::HfConfig c = vllm::HyV4FlashHfConfigFromGguf(gguf);
  REQUIRE(c.raw.contains("gguf_unmapped"));
  const auto& u = c.raw["gguf_unmapped"];
  CHECK(u.contains("hy_v4_flash.attention.indexer.top_k"));
  CHECK(u["hy_v4_flash.attention.indexer.top_k"].get<int>() == 2048);
  CHECK(u.contains("hy_v4_flash.ple.ngram_size"));
  CHECK(u["hy_v4_flash.ple.ngram_size"].get<int>() == 3);
  CHECK(u.contains("hy_v4_flash.engram.pad_token_id"));
  CHECK(u.contains("hy_v4_flash.hyper_connection.count"));
  // The per-layer head_count ARRAY is carried explicitly (its first entry fed
  // the scalar).
  REQUIRE(c.raw.contains("hy_v4_flash_attention_head_count"));
  CHECK(c.raw["hy_v4_flash_attention_head_count"].size() == 43u);
  CHECK(c.raw["hy_v4_flash_attention_head_count"][2].get<int>() == 64);
  CHECK_FALSE(u.contains("hy_v4_flash.block_count"));
  CHECK_FALSE(u.contains("hy_v4_flash.attention.head_count"));
}

TEST_CASE("hy_v4_flash: the forward refusal names the port row") {
  const std::string r = vllm::HyV4FlashForwardRefusal();
  CHECK(r.find("HyV4FlashHfConfigFromGguf") != std::string::npos);
  CHECK(r.find("NEW-CANDIDATE-PORTS") != std::string::npos);
}

TEST_CASE("maxwell: the builder validates the pinned config + shard headers") {
  if (!Present(MAXWELL_DIR)) {
    MESSAGE("SKIP: " << MAXWELL_DIR << " not mounted");
    return;
  }
  vllm::HfConfig c = vllm::LoadHfConfig(std::string(MAXWELL_DIR) +
                                        "/config.json");
  REQUIRE(vllm::IsMaxwellMoeConfig(c));
  const vllm::HfConfig full = vllm::MaxwellConfigFromJson(c, MAXWELL_DIR);
  CHECK(full.model_type == "maxwell_moe");
  CHECK(full.hidden_size == 2048);
  CHECK(full.num_hidden_layers == 45);
  CHECK(full.num_attention_heads == 32);
  CHECK(full.num_key_value_heads == 4);
  CHECK(full.head_dim == 128);  // attention_head_size, mapped by name
  CHECK(full.num_experts == 256);
  CHECK(full.num_experts_per_tok == 12);
  CHECK(full.vocab_size == 200704);
  CHECK(full.max_position_embeddings == 524288);
  CHECK(full.raw["maxwell_shard_count"].get<size_t>() == 35u);
  CHECK(full.raw.contains("maxwell_header_tensors_seen"));
}

TEST_CASE("maxwell: remote-code-defaulted fields are UNKNOWN, never guessed") {
  if (!Present(MAXWELL_DIR)) {
    MESSAGE("SKIP: " << MAXWELL_DIR << " not mounted");
    return;
  }
  vllm::HfConfig c = vllm::LoadHfConfig(std::string(MAXWELL_DIR) +
                                        "/config.json");
  const vllm::HfConfig full = vllm::MaxwellConfigFromJson(c, MAXWELL_DIR);
  REQUIRE(full.raw.contains("maxwell_unknown"));
  const std::string unknown = full.raw["maxwell_unknown"].dump();
  CHECK(unknown.find("attention_bias") != std::string::npos);
  CHECK(unknown.find("UNKNOWN") != std::string::npos);
}

TEST_CASE("maxwell: the forward refusal names the port row") {
  const std::string r = vllm::MaxwellForwardRefusal();
  CHECK(r.find("MaxwellConfigFromJson") != std::string::npos);
  CHECK(r.find("NEW-CANDIDATE-PORTS") != std::string::npos);
  CHECK(r.find("never executed") != std::string::npos);
}

TEST_CASE("maxwell: a non-maxwell config refuses by name") {
  vllm::HfConfig c;
  c.model_type = "llama";
  CHECK_THROWS_WITH(vllm::MaxwellConfigFromJson(c, "/tmp"),
                    doctest::Contains("maxwell_moe"));
}
