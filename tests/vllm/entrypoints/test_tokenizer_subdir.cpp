// The tokenizer/ subdirectory fallback, entered through the production loader.
//
// `convaiinnovations/laya` (MODEL-LAYA) keeps its tokenizer at
// `<model_dir>/tokenizer/tokenizer.json`. Before the fallback the loader read
// only `<model_dir>/tokenizer.json`, found nothing, built an EMPTY tokenizer,
// and every /v1/systemone request then failed with
// `tokenizer: symbol "c" not in vocab`.
// ISSUE-LOCAL-01M3SDXGYYTS9FDXKZDAE24N0R.
//
// How the loader cases observe which file was read: every tokenizer file here is
// malformed JSON, and `Tokenizer::FromHfJson` names the path it failed to parse.
// The model directory has a valid config.json and no weights, so a load that
// reads no tokenizer file fails later, on the absent weights, with a message
// that names no tokenizer. The error message therefore says exactly which
// tokenizer file `LoadedEngine::FromModelDir` opened, if any.
//
// Red before the fix: the subdirectory-only case fails on the weights instead
// of on tokenizer/tokenizer.json.
#include <doctest/doctest.h>

#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>

#include "support/process_id.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/transformers_utils/tokenizer_files.h"

namespace fs = std::filesystem;

namespace {

// A fresh, empty directory under the temp dir, removed on scope exit.
struct TempModelDir {
  explicit TempModelDir(const std::string& stem)
      : path(fs::temp_directory_path() /
             (stem + "-" + std::to_string(vllm_test::ProcessId()))) {
    fs::remove_all(path);
    fs::create_directories(path);
  }
  ~TempModelDir() {
    std::error_code ec;
    fs::remove_all(path, ec);
  }
  fs::path path;
};

void WriteFile(const fs::path& p, const std::string& body) {
  fs::create_directories(p.parent_path());
  std::ofstream(p, std::ios::binary) << body;
}

// A registered architecture with exactly the keys `LoadHfConfig` requires, as in
// test_placement_reach.cpp. No shards, so a load that passes the tokenizer step
// throws on the weights.
void WriteConfig(const fs::path& dir) {
  WriteFile(dir / "config.json", R"({
    "architectures": ["Qwen3MoeForCausalLM"],
    "model_type": "qwen3_moe",
    "hidden_size": 64,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "head_dim": 16,
    "intermediate_size": 128,
    "vocab_size": 256,
    "num_hidden_layers": 2,
    "rms_norm_eps": 1e-6
  })");
}

constexpr const char* kMalformed = "{ this is not json";

// The message FromModelDir throws for `dir`. The load must throw: there are no
// weights, and every tokenizer file present is malformed.
std::string LoadError(const fs::path& dir) {
  vllm::entrypoints::EngineParams params;
  try {
    (void)vllm::entrypoints::LoadedEngine::FromModelDir(dir.string(), params);
  } catch (const std::exception& e) {
    return e.what();
  }
  FAIL("FromModelDir loaded a directory that has no weights");
  return {};
}

bool Contains(const std::string& haystack, const std::string& needle) {
  return haystack.find(needle) != std::string::npos;
}

}  // namespace

TEST_CASE("tokenizer subdir: FromModelDir reads tokenizer/tokenizer.json when the root file is absent") {
  TempModelDir d("vllm-cpp-tok-subdir-only");
  WriteConfig(d.path);
  WriteFile(d.path / "tokenizer" / "tokenizer.json", kMalformed);

  const std::string err = LoadError(d.path);
  INFO("error: " << err);
  CHECK(Contains(err, "JSON parse error in"));
  CHECK(Contains(err, (d.path / "tokenizer" / "tokenizer.json").string()));
}

TEST_CASE("tokenizer subdir: the root tokenizer.json wins over tokenizer/tokenizer.json") {
  TempModelDir d("vllm-cpp-tok-root-wins");
  WriteConfig(d.path);
  WriteFile(d.path / "tokenizer.json", kMalformed);
  WriteFile(d.path / "tokenizer" / "tokenizer.json", kMalformed);

  const std::string err = LoadError(d.path);
  INFO("error: " << err);
  CHECK(Contains(err, "JSON parse error in"));
  CHECK(Contains(err, (d.path / "tokenizer.json").string()));
  CHECK_FALSE(Contains(err, (d.path / "tokenizer" / "tokenizer.json").string()));
}

TEST_CASE("tokenizer subdir: with neither file the load reads no tokenizer, as before") {
  TempModelDir d("vllm-cpp-tok-neither");
  WriteConfig(d.path);
  // A tokenizer/ directory without tokenizer.json is not a tokenizer.
  fs::create_directories(d.path / "tokenizer");
  WriteFile(d.path / "tokenizer" / "vocab.txt", "a\n");

  const std::string err = LoadError(d.path);
  INFO("error: " << err);
  CHECK_FALSE(Contains(err, "tokenizer:"));
}

TEST_CASE("ResolveTokenizerFile: root wins, the subdirectory is only a fallback") {
  TempModelDir d("vllm-cpp-tok-resolve");
  for (const std::string name : {"tokenizer.json", "tokenizer_config.json"}) {
    CAPTURE(name);
    const fs::path root = d.path / name;
    const fs::path sub = d.path / "tokenizer" / name;

    // Neither: the root path, so callers keep their "absent" handling.
    CHECK(vllm::ResolveTokenizerFile(d.path, name) == root);

    // Subdirectory only.
    WriteFile(sub, "{}");
    CHECK(vllm::ResolveTokenizerFile(d.path, name) == sub);

    // Both: the root file.
    WriteFile(root, "{}");
    CHECK(vllm::ResolveTokenizerFile(d.path, name) == root);

    // Root only.
    fs::remove(sub);
    CHECK(vllm::ResolveTokenizerFile(d.path, name) == root);
    fs::remove(root);
  }

  // A directory named like the file under tokenizer/ is not a fallback.
  fs::create_directories(d.path / "tokenizer" / "tokenizer.json");
  CHECK(vllm::ResolveTokenizerFile(d.path, "tokenizer.json") ==
        d.path / "tokenizer.json");
}
