// M0.10 Task 2: LoadedEngine::FromModelDir routes a `.gguf` path to the GGUF
// loader (not the safetensors directory path). A full end-to-end engine build
// from a real GGUF is dgx-pending; here we prove the branch selection: a
// `.gguf` file is opened as GGUF (a bad-magic .gguf surfaces the reader's
// "magic" error, which the directory path can never produce), and a non-.gguf
// non-directory path still takes the directory branch.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#include "gguf_builder.h"
#include "vllm/entrypoints/model_loader.h"
// dots3_note.h is a MODEL-PRIVATE header under src/ (W1 ships nothing on the
// public ABI). It is here for ONE symbol: `Dots3NoteGgufRefusal`, so the
// REACHABLE door can be compared to its owner byte-for-byte rather than by
// substring. Same arrangement test_dots3_note_scaffold uses.
#include "vllm/model_executor/models/dots3_note.h"

using gguf_test::TempFile;
using vllm::entrypoints::EngineParams;
using vllm::entrypoints::LoadedEngine;

namespace {

// The smallest GGUF that reaches the entrypoint's architecture dispatch: a
// valid v3 header plus the ONE kv the dispatch reads. No tensors and no vocab
// on purpose — the refusal has to fire before any tokenizer or weight I/O, and
// a fixture that carries them could not tell a refusal at the dispatch apart
// from one further down.
std::string GgufWithArchitecture(const std::string& arch) {
  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.architecture", arch));
  return builder.Build();
}

// What FromModelDir threw, as text. CHECK_THROWS_WITH_AS proves a substring is
// PRESENT; the point of #809 is also what must be ABSENT, and only the caught
// message can answer that.
std::string RefusalFor(const std::string& gguf_bytes) {
  TempFile file(gguf_bytes);
  try {
    LoadedEngine::FromModelDir(file.path(), EngineParams{});
  } catch (const std::exception& e) {
    return e.what();
  }
  return "";
}

class TempDir {
 public:
  TempDir() {
    static int counter = 0;
    path_ = std::filesystem::temp_directory_path() /
            ("vllm_model_registry_reject_" + std::to_string(counter++));
    std::filesystem::create_directories(path_);
  }
  ~TempDir() { std::filesystem::remove_all(path_); }
  const std::filesystem::path& path() const { return path_; }

 private:
  std::filesystem::path path_;
};

}  // namespace

TEST_CASE("FromModelDir routes .gguf files to the GGUF reader") {
  // A .gguf file with a corrupt magic: the GGUF branch is taken, so the
  // reader's "magic" error surfaces (the directory branch cannot produce it).
  TempFile bad(std::string("GGML") + std::string(60, '\0'));
  CHECK_THROWS_WITH_AS(LoadedEngine::FromModelDir(bad.path(), EngineParams{}),
                       doctest::Contains("magic"), std::runtime_error);
}

TEST_CASE("FromModelDir keeps the directory path for non-.gguf inputs") {
  CHECK_THROWS_WITH_AS(
      LoadedEngine::FromModelDir("/nonexistent/model/dir", EngineParams{}),
      doctest::Contains("not a directory"), std::runtime_error);
}

// #809. The GGUF architecture dispatch used to have no default: after the
// `deepseek4` and `muse-glimmer` arms it FELL THROUGH to `vllm::HfConfigFromGguf`
// — qwen3_5's config builder, which hard-asserts its own three keys
// (qwen3_5_gguf_weights.cpp). So every GGUF outside those five names was refused
// with "qwen3_5 gguf: unexpected architecture", naming a model unrelated to the
// file the user passed and sending the reader into an unrelated translation
// unit. These cases drive `LoadedEngine::FromModelDir` — the REAL path a `.gguf`
// argument takes — not a hand-built ModelSource.
TEST_CASE("An unsupported GGUF architecture is refused BY NAME, not qwen3_5's") {
  const std::string message = RefusalFor(GgufWithArchitecture("mamba-unported"));
  REQUIRE_FALSE(message.empty());

  // It names the file's OWN architecture...
  CHECK(message.find("mamba-unported") != std::string::npos);
  // ...and the set this build actually supports, so the reader can see whether
  // the file is convertible to something loadable rather than guessing.
  CHECK(message.find("deepseek4") != std::string::npos);
  CHECK(message.find("muse-glimmer") != std::string::npos);
  CHECK(message.find("qwen35") != std::string::npos);
  CHECK(message.find("qwen35moe") != std::string::npos);
  CHECK(message.find("qwen3next") != std::string::npos);
  // And it does NOT hand the reader one model's parser as the explanation.
  CHECK(message.find("qwen3_5 gguf:") == std::string::npos);
}

TEST_CASE("A GGUF with no general.architecture says so, by name") {
  // The old fall-through reported this as qwen3_5's "general.architecture must
  // be a string" too. There is no architecture to select, and the message that
  // says that must not be a specific model's.
  gguf_test::GgufModelBuilder builder;
  builder.AddKv(gguf_test::StrKv("general.name", "no-arch-key"));
  const std::string message = RefusalFor(builder.Build());
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("general.architecture") != std::string::npos);
  CHECK(message.find("deepseek4") != std::string::npos);
  CHECK(message.find("qwen3_5 gguf:") == std::string::npos);
}

TEST_CASE("A supported GGUF architecture still reaches its own builder") {
  // The guard on the change: adding an explicit default must not have moved a
  // SUPPORTED key onto the refusal path. A `qwen35` file with none of the
  // geometry keys reaches qwen3_5's builder and fails on the FIRST MISSING KEY,
  // which only that builder can report — so this proves the arm was taken, not
  // merely that something threw.
  const std::string message = RefusalFor(GgufWithArchitecture("qwen35"));
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("qwen3_5 gguf: missing metadata key") != std::string::npos);
  CHECK(message.find("is not supported by this build") == std::string::npos);
}

TEST_CASE("a qwen4exp GGUF reaches ITS OWN builder through the dispatch") {
  // MODEL-MM-QWEN4-EXP W6a, and the REACHABILITY case for the new arm: the
  // dispatch row is only real if a `qwen4exp` file actually lands on
  // `Qwen4ExpHfConfigFromGguf`. It is proven the way the `qwen35` case above is
  // — by the message. A file with the architecture key and no geometry fails on
  // the FIRST MISSING KEY, and only that builder reports that key with that
  // prefix, so this shows the arm was TAKEN rather than that something threw.
  const std::string message = RefusalFor(GgufWithArchitecture("qwen4exp"));
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("qwen4_exp gguf: missing metadata key") !=
        std::string::npos);
  CHECK(message.find("qwen4exp.embedding_length") != std::string::npos);
  // Not the unsupported-architecture refusal any more...
  CHECK(message.find("is not supported by this build") == std::string::npos);
  // ...and emphatically not qwen3_5's, which is the #809 defect: that builder
  // asserts its own three architectures by name, so routing a fourth family
  // there would blame a model the user never mentioned.
  CHECK(message.find("qwen3_5 gguf:") == std::string::npos);
}

TEST_CASE("a glm5next GGUF reaches ITS OWN builder through the dispatch") {
  // MODEL-MM-GLM53-FLASH W1, #2067 -- the REACHABILITY case for O9. W7a's
  // converter (`scripts/convert-glm5-next-gguf.py`) writes
  // `general.architecture = glm5next`, and until this row existed the file it
  // wrote was refused as merely unrecognized: the converter shipped a
  // capability no production entry point reached.
  //
  // Proven by the MESSAGE, exactly as the `qwen35` and `qwen4exp` cases above
  // are. A file carrying the architecture key and no geometry fails on the
  // FIRST MISSING KEY, and only this family's builder reports that key with
  // that prefix -- so this shows the arm was TAKEN, not merely that something
  // threw. `LoadedEngine::FromModelDir` is the real path a `.gguf` argument
  // takes; nothing here hand-builds a ModelSource.
  const std::string message = RefusalFor(GgufWithArchitecture("glm5next"));
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("glm5_next gguf: missing metadata key") !=
        std::string::npos);
  // `block_count` and not `embedding_length`: the builder reads the layer
  // count FIRST, because every per-layer schedule it validates is measured
  // against it. Naming the key the builder actually reached is what makes this
  // a proof that the arm was taken rather than a proof that a string appears.
  CHECK(message.find("glm5next.block_count") != std::string::npos);
  // Not the unsupported-architecture refusal any more...
  CHECK(message.find("is not supported by this build") == std::string::npos);
  // ...and not qwen3_5's, which is the #809 defect this dispatch table exists
  // to prevent.
  CHECK(message.find("qwen3_5 gguf:") == std::string::npos);
}

TEST_CASE("a glm-dsa GGUF reaches ITS OWN builder through the dispatch") {
  // MODEL-TEXT-deepseek-v2-glm-moe-dsa-for-causal-lm W2, #2214 -- the
  // REACHABILITY case for the `glm-dsa` arm, in the same shape as the three
  // above. GLM-5.3 is the first row here whose family vLLM DOES implement at
  // the pin, and whose GGUF our own DeepSeek-V2 loader refuses outright
  // (`deepseek_v2_registry.cpp`: "does not support GGUF weights"), so the arm
  // is a new row rather than a `deepseek2` one.
  //
  // Proven by the MESSAGE. A file carrying the architecture key and no geometry
  // fails on the FIRST MISSING KEY, and only this family's builder reports that
  // key with that prefix. `block_count` and not `embedding_length` because the
  // builder resolves the backbone depth first: `block_count` counts the
  // multi-token-prediction block and `num_hidden_layers` does not.
  const std::string message = RefusalFor(GgufWithArchitecture("glm-dsa"));
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("glm-dsa gguf: missing metadata key") != std::string::npos);
  CHECK(message.find("glm-dsa.block_count") != std::string::npos);
  // Not the unsupported-architecture refusal any more...
  CHECK(message.find("is not supported by this build") == std::string::npos);
  // ...and not qwen3_5's, which is the #809 defect this dispatch table exists
  // to prevent.
  CHECK(message.find("qwen3_5 gguf:") == std::string::npos);
}

TEST_CASE("a laguna GGUF reaches ITS OWN builder through the dispatch") {
  // MODEL-TEXT-laguna-laguna-for-causal-lm, #2841: the registry's Laguna GGUF
  // arm was unreachable from FromModelDir, which refused `laguna` as an
  // unsupported architecture. Proven by the message, as the rows above are:
  // only the Laguna resolve names `laguna.embedding_length` as its first
  // missing key.
  const std::string message = RefusalFor(GgufWithArchitecture("laguna"));
  REQUIRE_FALSE(message.empty());
  CHECK(message.find("laguna gguf: missing required key laguna.embedding_length") !=
        std::string::npos);
  CHECK(message.find("is not supported by this build") == std::string::npos);
  CHECK(message.find("qwen3_5 gguf:") == std::string::npos);
}

TEST_CASE("FromModelDir rejects an unknown dense architecture before loading") {
  // The rejection must fire during architecture resolution, BEFORE any tokenizer
  // or weight I/O — so the arch must be one the registry does NOT know. (Note:
  // LlamaForCausalLM, which this case originally used, is now a SUPPORTED arch,
  // so it would sail past resolve and fail later on a missing tokenizer instead.)
  // Gemma4ForCausalLM is a real-but-still-unported Gemma arch: it is absent from
  // the registry, so Resolve throws the "not supported" error up front.
  TempDir dir;
  nlohmann::json config{
      {"model_type", "gemma"},
      {"architectures", nlohmann::json::array({"Gemma4ForCausalLM"})},
      {"hidden_size", 8},
      {"num_hidden_layers", 1},
      {"num_attention_heads", 1},
      {"vocab_size", 8},
  };
  std::ofstream(dir.path() / "config.json") << config.dump();

  CHECK_THROWS_WITH_AS(
      LoadedEngine::FromModelDir(dir.path().string(), EngineParams{}),
      "Model architectures ['Gemma4ForCausalLM'] are not supported for now. "
      "Supported architectures: "
      "dict_keys(['BoundaryExtractor', 'ClmModel', 'Cohere2MoeForCausalLM', 'CohereForCausalLM', 'CuaS1Forms', 'DeepseekV2ForCausalLM', "
      "'DeepseekV41ForCausalLM', 'DeepseekV4ForCausalLM', 'Dots3NoteForCausalLM', 'Gemma2ForCausalLM', 'Gemma3ForCausalLM', "
      "'Gemma4ForConditionalGeneration', 'Gemma4UnifiedForConditionalGeneration', 'GemmaForCausalLM', 'Glm4ForCausalLM', "
      "'Glm4MoeLiteForCausalLM', 'Glm5NextForConditionalGeneration', 'GlmMoeDsaForCausalLM', 'GraniteForCausalLM', "
      "'InternLM2ForCausalLM', 'InternLM3ForCausalLM', 'KevModel', 'KimiK3ForConditionalGeneration', 'KimiLinearForCausalLM', "
      "'LagunaForCausalLM', 'LayaModel', 'LlamaForCausalLM', 'LlamaModel', 'MiMoV2ForCausalLM', 'MiniCPM3ForCausalLM', "
      "'MiniCPMForCausalLM', 'MistralForCausalLM', 'MuseGlimmerForCausalLM', 'MuseGlimmerForConditionalGeneration', "
      "'NemotronHForCausalLM', 'NemotronH_Nano_Omni_Reasoning_V3', 'NemotronH_Nano_VL_V2', 'NimbleModel', 'OPTForCausalLM', "
      "'Olmo2ForCausalLM', 'Olmo3ForCausalLM', 'ParakeetForCTC', 'ParakeetForRNNT', 'ParakeetForTDT', 'Phi3ForCausalLM', "
      "'PhiForCausalLM', 'Qwen3ForCausalLM', 'Qwen3MoeForCausalLM', 'Qwen3VLForConditionalGeneration', 'Qwen3_5ForCausalLM', "
      "'Qwen3_5ForConditionalGeneration', 'Qwen3_5MoeForCausalLM', 'Qwen3_5MoeForConditionalGeneration', 'Qwen4ExpForConditionalGeneration', "
      "'SpanExtractor', 'StableLmForCausalLM', 'Tev1Model', 'XorModel'])",
      std::runtime_error);
}

// MODEL-MM-dots3-note W9a (#2882). The dots3-note GGUF refusal was CAREFUL and
// UNREACHABLE: `LoadDots3NoteForCausalLM` refused `Kind::kGguf` by name, naming
// the row and the brick, but
// `src/vllm/entrypoints/model_loader.cpp::HfConfigFromGgufDispatch` runs long
// before the same function's `ModelSource::FromGguf`, so a real `dots3note` file
// died at the build-level default naming neither the model, the row, the brick,
// nor what is owed. These cases drive `LoadedEngine::FromModelDir` — the entry
// point every server and CLI `.gguf` argument takes — exactly as the #809 cases
// above do, and they are the whole gate for this slice: it adds no arithmetic,
// so its gate is the refusal's REACHABILITY and its TEXT (spec §4.19.4).
TEST_CASE("a dots3note GGUF is refused by dots3-note's OWN message") {
  const std::string message = RefusalFor(GgufWithArchitecture("dots3note"));
  REQUIRE_FALSE(message.empty());

  // It names the MODEL...
  CHECK(message.find("Dots3NoteForCausalLM") != std::string::npos);
  CHECK(message.find("dots3-note") != std::string::npos);
  // ...the ROW that owes the arm...
  CHECK(message.find("MODEL-MM-dots3-note-dots3-note-for-causal-lm") !=
        std::string::npos);
  // ...the BRICK...
  CHECK(message.find("W9") != std::string::npos);
  // ...and the spec section a reader lands on.
  CHECK(message.find(".agents/specs/dots3-note.md") != std::string::npos);
  // And it is NOT the build-level default, which names none of those.
  CHECK(message.find("is not supported by this build") == std::string::npos);
}

// The SECOND defect #2882 names, and it is a defect in PRODUCT OUTPUT rather
// than in a record: the W1 message told the operator that llama.cpp has no
// `dots3_note` architecture and therefore no converter to reuse. False at
// llama.cpp `master` — `LLM_ARCH_DOTS3NOTE -> "dots3note"`
// (`src/llama-arch.cpp:114`), merged by `ggml-org/llama.cpp`
// 5a32f7b66ef6cfb3e60deea26e3454cc6ad3438c ("model: add dots3-note", #27060,
// 2026-08-21) and 54ee5ee643f29abba6852903ddfdb688c2361b5b ("mtmd: support
// dots3-note vision+audio", #27524, 2026-08-22), with the converter at
// `conversion/dots3.py`. A record correction that leaves the lie in the throw
// is not a fix, so the SHAs are asserted in the message a user actually reads.
TEST_CASE("the dots3note GGUF refusal tells the truth about llama.cpp") {
  const std::string message = RefusalFor(GgufWithArchitecture("dots3note"));
  REQUIRE_FALSE(message.empty());

  // The two merge commits, so the reader can check the claim instead of
  // trusting it.
  CHECK(message.find("5a32f7b66ef6cfb3e60deea26e3454cc6ad3438c") !=
        std::string::npos);
  CHECK(message.find("54ee5ee643f29abba6852903ddfdb688c2361b5b") !=
        std::string::npos);
  // The false claim is GONE, in either of the two spellings W1 used.
  CHECK(message.find("llama.cpp has no") == std::string::npos);
  CHECK(message.find("no upstream converter to reuse") == std::string::npos);
  // And it does NOT oversell the position: this build still cannot read one.
  // W9b/W9c/W9e/W9f own that (spec §4.19.5).
  CHECK(message.find("cannot read it yet") != std::string::npos);
}

// The fresh review of #2882 measured this hole and it is the reason this case
// exists. `test_dots3_note_scaffold` asserts that the FACTORY guard's bytes
// equal `Dots3NoteGgufRefusal()` -- but the factory guard is the door THIS SLICE
// PROVES UNREACHABLE, so that assertion holds the one door a real file never
// arrives at. Measured on the reviewed head: appending `" (dots3note)"` to the
// string `HfConfigFromGgufDispatch` throws left BOTH suites fully green. "The
// two doors cannot drift" was a claim about one door.
//
// This case holds the OTHER one, and it is the only byte-exact assertion on
// what a user actually reads: FromModelDir's thrown message, whole, against the
// single owner both doors are supposed to borrow from. A substring set cannot
// do this -- every substring above still matched while the mutated build
// printed a different message.
TEST_CASE("the REACHABLE dots3note refusal is byte-identical to its ONE owner") {
  CHECK(RefusalFor(GgufWithArchitecture("dots3note")) ==
        vllm::Dots3NoteGgufRefusal());
}

// `IsDots3NoteGguf` was proved to FIRE (delete the dispatch branch, or return
// `false`, and the cases above go red) and proved not to fire for EVERYTHING
// (return `true` unconditionally and the `mamba-unported` case above goes red).
// The narrow half was open: nothing fed it a NEAR MISS, so loosening its `==`
// to a prefix match left both suites green. That is the "right reason" half of
// the gate, and these two architectures close it.
//
// `dots3note-moe` is the prefix case -- a plausible future sibling whose GGUF is
// NOT this model's. `dots3_note` is the underscore spelling W1's own refusal
// text used, which is the separator-normalising variant of the same error. A
// file carrying either is not a dots3-note file, so both must reach the
// build-level default that names the file's own architecture, and neither may
// be claimed by a row that does not owe it.
TEST_CASE("a NEAR-MISS GGUF architecture is NOT claimed by dots3-note") {
  for (const char* arch_c : {"dots3note-moe", "dots3_note"}) {
    const std::string arch(arch_c);
    CAPTURE(arch);
    const std::string message = RefusalFor(GgufWithArchitecture(arch));
    REQUIRE_FALSE(message.empty());

    // The build-level default, naming the file's OWN architecture...
    CHECK(message.find("is not supported by this build") != std::string::npos);
    CHECK(message.find(arch) != std::string::npos);
    // ...and NOT dots3-note's refusal, in either the substring or the byte form.
    CHECK(message.find("MODEL-MM-dots3-note-dots3-note-for-causal-lm") ==
          std::string::npos);
    CHECK(message != vllm::Dots3NoteGgufRefusal());
  }
}
