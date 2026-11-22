// Row NEW-CANDIDATE-PORTS W0 — CPU-only tests for (a) the two missing ggml
// type ids in the GGUF reader and (b) the inkling2_loop config builder.
//
// The pinned artifact path is the fixture; a missing artifact exits with
// CTest's SKIP_RETURN_CODE (77, wired by vllm_cpp_add_test) so CI without
// /mnt/models reports Skipped, never a false green (issue #463).
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <filesystem>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/inkling2_loop_gguf_config.h"
#include "vllm/transformers_utils/hf_config.h"

using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

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

// A tiny, fully-valid inkling2_loop GGUF built entirely in memory. No mount and
// no large artifact: every KV the parser consumes is written with the shared
// gguf_test builder. The defaults satisfy all three loop_config validations,
// so the production refusal at the loader's forward boundary is the FIRST
// thing that can throw. A test overrides one field to drive a single
// validation, and the builder otherwise stays identical.
struct TinyInkling2 {
  int block_count = 4;
  int hidden = 64;
  int vocab = 32;
  int core_start = 1;
  int core_end = 3;
  int num_passes = 3;
  int gate_rank = 64;
  bool is_varloop = false;
};

std::string TinyInkling2Gguf(const TinyInkling2& f = {}) {
  gguf_test::GgufModelBuilder b;
  b.AddKv(gguf_test::StrKv("general.architecture", "inkling2_loop"));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.block_count", f.block_count));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.embedding_length", f.hidden));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.feed_forward_length", 128));
  b.AddKv(gguf_test::I32ArrayKv(
      "inkling2_loop.attention.head_count",
      std::vector<int32_t>(f.block_count, 4)));
  b.AddKv(gguf_test::I32ArrayKv(
      "inkling2_loop.attention.head_count_kv",
      std::vector<int32_t>(f.block_count, 2)));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.attention.key_length", 16));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.expert_count", 8));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.expert_used_count", 2));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.expert_feed_forward_length", 128));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.context_length", 1024));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.attention.sliding_window", 512));
  b.AddKv(gguf_test::F32Kv(
      "inkling2_loop.attention.layer_norm_rms_epsilon", 1e-5f));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.vocab_size", f.vocab));
  // ggml dims are inner-first: {hidden, vocab} reverses to shape
  // [vocab, hidden], which is what the parser cross-checks against vocab_size
  // and embedding_length.
  b.AddTensor("token_embd.weight",
              {static_cast<uint64_t>(f.hidden), static_cast<uint64_t>(f.vocab)},
              0 /* F32 */,
              std::string(static_cast<size_t>(f.hidden) * f.vocab * 4, '\0'));
  b.AddKv(gguf_test::BoolArrayKv(
      "inkling2_loop.attention.sliding_window_pattern",
      std::vector<bool>(f.block_count, false)));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.loop.start", f.core_start));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.loop.end", f.core_end));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.loop.count", f.num_passes));
  b.AddKv(gguf_test::U32Kv("inkling2_loop.loop.gate_rank", f.gate_rank));
  b.AddKv(gguf_test::BoolKv("inkling2_loop.loop.varying", f.is_varloop));
  return b.Build();
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

TEST_CASE("inkling2_loop: FromModelDir reaches the loader row and refuses") {
  // The production path a real `.gguf` argument takes. A direct call to the
  // builder proves the parser works; it says nothing about whether the loader
  // ever calls it. Drive LoadedEngine::FromModelDir with the tiny valid file so
  // the arch table selects this family's builder and the forward-boundary
  // refusal fires. Delete EITHER the kGgufArchArms row OR the refusal block and
  // the thrown message loses "NEW-CANDIDATE-PORTS" -- this case goes red.
  gguf_test::TempFile f(TinyInkling2Gguf());
  try {
    LoadedEngine::FromModelDir(f.path(), EngineParams{});
    FAIL("FromModelDir must refuse the inkling2_loop forward boundary");
  } catch (const std::exception& e) {
    const std::string message = e.what();
    CHECK(message.find("NEW-CANDIDATE-PORTS") != std::string::npos);
    CHECK(message.find("forward boundary") != std::string::npos);
  }
}

TEST_CASE("inkling2_loop: loop core layers outside the block count are refused") {
  // Synthetic, no mount. core_end 5 > block_count 4 -- upstream configs.py
  // requires 0 < core_start < core_end <= num_hidden_layers.
  TinyInkling2 f;
  f.core_end = 5;
  gguf_test::TempFile file(TinyInkling2Gguf(f));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  CHECK_THROWS_WITH(vllm::Inkling2LoopHfConfigFromGguf(gguf),
                    doctest::Contains("loop core layers are out of range"));
}

TEST_CASE("inkling2_loop: gate_rank 0 is refused") {
  // Upstream configs.py requires gate_rank >= 1.
  TinyInkling2 f;
  f.gate_rank = 0;
  gguf_test::TempFile file(TinyInkling2Gguf(f));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  CHECK_THROWS_WITH(vllm::Inkling2LoopHfConfigFromGguf(gguf),
                    doctest::Contains("loop gate_rank must be >= 1"));
}

TEST_CASE("inkling2_loop: a single fixed-loop pass is refused") {
  // A non-varloop checkpoint needs at least two passes (configs.py
  // check_loop_options). A varloop with one pass would be accepted.
  TinyInkling2 f;
  f.num_passes = 1;
  f.is_varloop = false;
  gguf_test::TempFile file(TinyInkling2Gguf(f));
  const vllm::GgufFile gguf = vllm::GgufFile::Open(file.path());
  CHECK_THROWS_WITH(vllm::Inkling2LoopHfConfigFromGguf(gguf),
                    doctest::Contains("at least 2 passes"));
}
