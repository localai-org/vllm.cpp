// MODEL-TEXT-kolibri-1-1 wave U1 — the GGUF architecture-collision guard.
// (ISSUE-LOCAL-01M84HPYRT3BNNF61BKGNRQ740, spec
// .agents/specs/kolibri-1-1-u1-walk-and-trap-guard.md.)
//
// llama.cpp's Kolibri-1.1 converter writes general.architecture="kolibri1",
// the SAME string the 1.0 arm claims. A 1.1 GGUF must be refused BY NAME,
// never silently loaded as 1.0. Every case here is hermetic: a tiny GGUF is
// built in memory by gguf_test::GgufModelBuilder, no mounted artifact is read.
//
// The production-path case drives LoadedEngine::FromModelDir, so deleting
// EITHER the guard OR the kGgufArchArms row in model_loader.cpp makes it red.
#include <doctest/doctest.h>

#include <cstdint>
#include <exception>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/kolibri1_gguf_weights.h"
#include "vllm/transformers_utils/hf_config.h"

using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

namespace {

// Every required 1.0 KV is present with tiny, self-consistent values. With no
// marker KV this file passes Kolibri1HfConfigFromGguf; adding ONE marker must
// flip it into the named 1.1 refusal.
std::string TinyKolibri1Gguf(const char* marker_key = nullptr) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "kolibri1"));
  b.AddKv(gguf_test::U32Kv("kolibri1.block_count", 2));
  b.AddKv(gguf_test::U32Kv("kolibri1.embedding_length", 16));
  b.AddKv(gguf_test::U32Kv("kolibri1.attention.head_count", 4));
  b.AddKv(gguf_test::U32Kv("kolibri1.attention.head_count_kv", 2));
  b.AddKv(gguf_test::U32Kv("kolibri1.attention.key_length", 8));
  b.AddKv(gguf_test::U32Kv("kolibri1.attention.sliding_window", 4));
  b.AddKv(gguf_test::F32Kv("kolibri1.rope.freq_base", 10000.0f));
  b.AddKv(gguf_test::F32Kv("kolibri1.attention.layer_norm_rms_epsilon", 1e-5f));
  b.AddKv(gguf_test::U32Kv("kolibri1.expert_count", 8));
  b.AddKv(gguf_test::U32Kv("kolibri1.expert_used_count", 2));
  b.AddKv(gguf_test::U32Kv("kolibri1.expert_feed_forward_length", 32));
  b.AddKv(gguf_test::U32Kv("kolibri1.expert_shared_feed_forward_length", 32));
  b.AddKv(gguf_test::U32Kv("kolibri1.expert_weights_norm", 0));
  b.AddKv(gguf_test::BoolArrayKv("kolibri1.attention.sliding_window_pattern",
                                 std::vector<bool>{true, false}));
  // ggml dims are inner-first: {hidden, vocab} becomes shape [vocab, hidden].
  b.AddTensor("token_embd.weight", {16, 32}, 0 /* F32 */,
              std::string(16 * 32 * 4, '\0'));

  if (marker_key != nullptr) {
    const std::string key(marker_key);
    if (key == "kolibri1.attention.indexer.types") {
      b.AddKv(gguf_test::BoolArrayKv(key, std::vector<bool>{false, true}));
    } else if (key == "kolibri1.attention.kv_source_map") {
      b.AddKv(gguf_test::I32ArrayKv(key, std::vector<int32_t>{0, 0}));
    } else {
      b.AddKv(gguf_test::U32Kv(key, 2048));
    }
  }
  return b.Build();
}

const char* const kMarkers[] = {
    "kolibri1.attention.indexer.top_k",
    "kolibri1.attention.indexer.types",
    "kolibri1.attention.kv_source_map",
    "kolibri1.attention.indexer.candidate_block_size",
    "kolibri1.attention.indexer.candidate_top_k_blocks",
    "kolibri1.leading_dense_block_count",
};

}  // namespace

TEST_CASE("kolibri1 gguf: 1.1 marker KVs are refused by name on the production path") {
  for (const char* key : kMarkers) {
    gguf_test::TempFile f(TinyKolibri1Gguf(key));
    bool refused = false;
    try {
      LoadedEngine::FromModelDir(f.path(), EngineParams{});
    } catch (const std::exception& e) {
      const std::string message = e.what();
      refused = true;
      CHECK(message.find("kolibri1_1") != std::string::npos);
      CHECK(message.find("MODEL-TEXT-kolibri-1-1") != std::string::npos);
      CHECK(message.find(key) != std::string::npos);
    }
    CHECK(refused);
  }
}

TEST_CASE("kolibri1 gguf: a file with no 1.1 marker still parses as 1.0") {
  // Control: the guard must not reject a genuine 1.0 header.
  gguf_test::TempFile f(TinyKolibri1Gguf());
  const vllm::GgufFile gguf = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Kolibri1HfConfigFromGguf(gguf);
  CHECK(config.model_type == "kolibri1");
  CHECK(config.num_hidden_layers == 2);
  CHECK(config.hidden_size == 16);
}
