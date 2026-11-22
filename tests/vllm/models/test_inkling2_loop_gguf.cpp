// Row NEW-CANDIDATE-PORTS W0 — CPU-only tests for (a) the two missing ggml
// type ids in the GGUF reader and (b) the inkling2_loop config builder.
//
// The pinned artifact path is the fixture; a missing artifact exits with
// CTest's SKIP_RETURN_CODE (77, wired by vllm_cpp_add_test) so CI without
// /mnt/models reports Skipped, never a false green (issue #463).
#include <doctest/doctest.h>

#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <string>

#include <nlohmann/json.hpp>

#include "vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/inkling2_loop_gguf_config.h"
#include "vllm/transformers_utils/hf_config.h"

#ifndef INKLING2_APEX_GGUF_PATH
#define INKLING2_APEX_GGUF_PATH \
  "/mnt/models/inkling2-mini-apex-gguf/Inkling-2-Mini-APEX-Balanced.gguf"
#endif

namespace {

// The compiled-in path is the pinned artifact. The env override exists only so
// the 77-style skip can be exercised on a host where the artifact IS mounted
// (point it at a missing path); normal CI just uses the compiled-in default.
const char* ApexPath() {
  const char* override = std::getenv("VT_INKLING2_APEX_GGUF_PATH");
  return override != nullptr && override[0] != '\0' ? override
                                                    : INKLING2_APEX_GGUF_PATH;
}

[[noreturn]] void SkipWithoutApex() {
  std::fprintf(stderr,
               "SKIP: %s not mounted; CTest reports this as Skipped (exit 77)\n",
               ApexPath());
  std::exit(77);
}

}  // namespace

TEST_CASE("gguf: type ids 3 (Q4_1) and 7 (Q5_1) are tabulated") {
  // Geometry from llama.cpp ggml-common.h / gguf-py GGML_QUANT_SIZES:
  // block_q4_1 = f16 d + f16 m + 16 qs = 20 bytes per 32 elements;
  // block_q5_1 = f16 d + f16 m + u8 qh + 16 qs = 21 bytes per 32 elements.
  const vllm::GgmlTypeTraits& q4_1 = vllm::GgmlTraits(3);
  CHECK(q4_1.block_elems == 32);
  CHECK(q4_1.block_bytes == 20);
  CHECK(std::string(q4_1.name) == "Q4_1");

  const vllm::GgmlTypeTraits& q5_1 = vllm::GgmlTraits(7);
  CHECK(q5_1.block_elems == 32);
  CHECK(q5_1.block_bytes == 21);
  CHECK(std::string(q5_1.name) == "Q5_1");
}

TEST_CASE("gguf: the APEX Balanced artifact opens with its 704 tensors") {
  if (!std::filesystem::exists(ApexPath())) SkipWithoutApex();
  // Before the type-table fix this throws
  // `gguf: tensor "blk.10.ffn_down_exps.weight" has unknown ggml type id 7`.
  const vllm::GgufFile gguf = vllm::GgufFile::Open(ApexPath());
  const vllm::GgufValue* arch = gguf.FindKv("general.architecture");
  REQUIRE(arch != nullptr);
  CHECK(std::get<std::string>(arch->v) == "inkling2_loop");
  CHECK(gguf.Tensors().size() == 704);

  int type7 = 0;
  for (const vllm::GgufTensorInfo& t : gguf.Tensors()) {
    if (t.ggml_type == 7) {
      ++type7;
      CHECK(t.name.find("ffn_down_exps.weight") != std::string::npos);
    }
  }
  // The committed walk records 14 type-7 tensors, all routed experts.
  CHECK(type7 == 14);

  const vllm::GgufTensorInfo& emb = gguf.Get("token_embd.weight");
  REQUIRE(emb.shape.size() == 2);
  CHECK(emb.shape[0] == 201024);  // vocab (ggml dims reversed to row-major)
  CHECK(emb.shape[1] == 2304);    // hidden
}

TEST_CASE("inkling2_loop: the builder maps the pinned artifact's fields") {
  if (!std::filesystem::exists(ApexPath())) SkipWithoutApex();
  const vllm::GgufFile gguf = vllm::GgufFile::Open(ApexPath());
  REQUIRE(vllm::IsInkling2LoopGguf(gguf));
  const vllm::HfConfig c = vllm::Inkling2LoopHfConfigFromGguf(gguf);
  CHECK(c.model_type == "inkling2_loop");
  REQUIRE(c.architectures.size() == 1);
  CHECK(c.architectures[0] == "Inkling2LoopForConditionalGeneration");
  CHECK(c.num_hidden_layers == 24);
  CHECK(c.hidden_size == 2304);
  CHECK(c.intermediate_size == 1152);
  CHECK(c.num_attention_heads == 24);   // head_count[0] (per-layer array)
  CHECK(c.num_key_value_heads == 8);    // head_count_kv[0] (per-layer array)
  CHECK(c.head_dim == 128);
  CHECK(c.num_experts == 128);
  CHECK(c.num_experts_per_tok == 8);
  CHECK(c.moe_intermediate_size == 1152);
  CHECK(c.max_position_embeddings == 1048576);
  CHECK(c.sliding_window.value_or(0) == 1024);
  CHECK(c.vocab_size == 201024);  // token_embd.weight shape[0], not a KV
  CHECK(c.rms_norm_eps > 0.0);
}

TEST_CASE("inkling2_loop: loop_config mirrors the upstream loop fields") {
  if (!std::filesystem::exists(ApexPath())) SkipWithoutApex();
  const vllm::GgufFile gguf = vllm::GgufFile::Open(ApexPath());
  const vllm::HfConfig c = vllm::Inkling2LoopHfConfigFromGguf(gguf);
  REQUIRE(c.raw.contains("loop_config"));
  const auto& loop = c.raw["loop_config"];
  // Upstream vllm/models/inkling2/configs.py spellings, sourced from the GGUF
  // walk: loop.start/end/count/gate_rank/varying.
  CHECK(loop["core_start"].get<int>() == 6);
  CHECK(loop["core_end"].get<int>() == 18);
  CHECK(loop["num_passes"].get<int>() == 3);
  CHECK(loop["gate_rank"].get<int>() == 64);
  CHECK(loop["is_varloop"].get<bool>() == false);
  // The per-layer arrays cannot fit a scalar HfConfig field: the first entry
  // populated it above and the whole arrays are carried explicitly.
  REQUIRE(c.raw.contains("inkling2_loop_attention_head_count"));
  CHECK(c.raw["inkling2_loop_attention_head_count"].size() == 24u);
}

TEST_CASE("inkling2_loop: unmapped KVs are carried verbatim") {
  if (!std::filesystem::exists(ApexPath())) SkipWithoutApex();
  const vllm::GgufFile gguf = vllm::GgufFile::Open(ApexPath());
  const vllm::HfConfig c = vllm::Inkling2LoopHfConfigFromGguf(gguf);
  REQUIRE(c.raw.contains("gguf_unmapped"));
  const auto& u = c.raw["gguf_unmapped"];
  CHECK(u.contains("inkling2_loop.attention.sliding_window_pattern"));
  CHECK(u["inkling2_loop.attention.sliding_window_pattern"].size() == 24u);
  CHECK(u.contains("inkling2_loop.vocab_size_unpadded"));
  CHECK(u["inkling2_loop.vocab_size_unpadded"].get<int>() == 200058);
  CHECK(u.contains("tokenizer.ggml.pre"));
  CHECK(u["tokenizer.ggml.pre"].get<std::string>() == "inkling2");
  CHECK(u.contains("general.quantization_version"));
  // Mapped keys are NOT in the carry.
  CHECK_FALSE(u.contains("inkling2_loop.block_count"));
  CHECK_FALSE(u.contains("inkling2_loop.loop.start"));
  CHECK(c.raw["gguf_unmapped_count"].get<size_t>() == u.size());
}

TEST_CASE("inkling2_loop: the forward refusal names the port row") {
  const std::string r = vllm::Inkling2LoopForwardRefusal();
  CHECK(r.find("Inkling2LoopHfConfigFromGguf") != std::string::npos);
  CHECK(r.find("NEW-CANDIDATE-PORTS") != std::string::npos);
  CHECK(r.find("forward boundary") != std::string::npos);
}

TEST_CASE("inkling2_loop: a wrong architecture refuses by name") {
  // Hermetic: no artifact needed. Feed a tiny synthetic GGUF with a foreign
  // arch to this family's builder, which must check ITS own arch.
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "llama"));
  gguf_test::TempFile f(b.Build());
  const vllm::GgufFile gguf = vllm::GgufFile::Open(f.path());
  CHECK_THROWS_WITH(vllm::Inkling2LoopHfConfigFromGguf(gguf),
                    doctest::Contains("inkling2_loop"));
}
