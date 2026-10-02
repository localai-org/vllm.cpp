// vllm.cpp original (row
// MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation,
// .agents/specs/muse-glimmer-parity.md); no upstream mirror: vLLM ships no Muse
// Glimmer token test at the pin.
//
// THE PAGED-ENGINE Muse Glimmer GGUF GREEDY GATE (the text tower).
// Drives the standard 16-prompt battery through the full paged LLMEngine via
// LoadedEngine::FromModelDir on `Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf`
// (meta-models/Muse-Glimmer-30B-GGUF @ 70bf1b61ac09f91b24d39038091b41c582bc5d7a,
// 16756683904 bytes) and adjudicates against a golden
// captured by the REGISTERED llama.cpp oracle at its pin b10451 on the SAME
// file (scripts/llamacpp/llamacpp_oracle.cpp). The golden is quant-matched and
// llama.cpp-denominated: it is not a vLLM result, and the vLLM-denominated
// golden is owed (spec `## Owed`).
//
// GATE FORM, as the other paged-engine gates: our ids must equal the committed
// anchor `our_ids.npy` exactly (a drift is a regression suspect), and wherever
// they differ from the oracle's own greedy, the oracle's teacher-forced gap on
// OUR prefix must stay inside kNearTieMnats. A cell outside the band is a real
// forward divergence and fails the gate.
//
// Goldens: tests/parity/goldens/muse_glimmer_30b_q4km/
//   p<i>_prompt.i32, greedy_ids.npy, our_ids.npy, neartie_gap_mnats.npy.
// Checkpoint-gated: VLLM_MUSE_GGUF_PARITY names the file; absent, a loud SKIP.
// BOOTSTRAP: VT_DUMP_IDS=1 with no anchor writes p<i>_prompt.i32 (the ids the
// engine itself fed) and our_ids.i32, for the oracle driver to consume.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <memory>
#include <string>
#include <vector>

#include "npy.h"
#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

constexpr int32_t kNearTieMnats = 500;
constexpr int kTokens = 32;

const std::vector<std::string>& Prompts() {
  static const std::vector<std::string> p = {
      "The capital of France is",
      "Once upon a time,",
      "In the beginning God created",
      "The quick brown fox jumps over",
      "def fibonacci(n):",
      "Water boils at a temperature of",
      "The theory of relativity was developed by",
      "To be or not to be, that is",
      "The largest planet in our solar system is",
      "Machine learning is a subfield of",
      "The mitochondria is the powerhouse of",
      "Roses are red, violets are",
      "The first president of the United States was",
      "E equals m c",
      "A journey of a thousand miles begins with",
      "The chemical symbol for gold is",
  };
  return p;
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.ignore_eos = true;
  sp.PostInit();
  return sp;
}

const int32_t* AsI32(const parity::NpyArray& a) {
  return reinterpret_cast<const int32_t*>(a.data.data());
}

std::vector<int32_t> ReadI32(const fs::path& path) {
  std::ifstream f(path, std::ios::binary);
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<int32_t> v(raw.size() / 4);
  std::memcpy(v.data(), raw.data(), v.size() * 4);
  return v;
}

void WriteI32(const fs::path& path, const std::vector<int32_t>& v) {
  std::FILE* f = std::fopen(path.string().c_str(), "wb");
  REQUIRE(f != nullptr);
  std::fwrite(v.data(), sizeof(int32_t), v.size(), f);
  std::fclose(f);
}

vllm::entrypoints::EngineParams GateParams() {
  vllm::entrypoints::EngineParams p;
  p.device = vllm::Device::kCPU;
  p.max_model_len = 512;  // the battery needs under 64 tokens of context
  p.max_num_seqs = 1;
  return p;
}

}  // namespace

TEST_CASE("Muse Glimmer 30B Q4_K_M GGUF paged-engine greedy gate vs llama.cpp b10451") {
  const char* gguf_env = std::getenv("VLLM_MUSE_GGUF_PARITY");
  const std::string label = "muse-glimmer-30b-q4km";
  if (gguf_env == nullptr || gguf_env[0] == '\0') {
    MESSAGE(label << ": SKIP, VLLM_MUSE_GGUF_PARITY unset (set it to "
                     "Muse-Glimmer-30B-KQuant-17GB-Q4_K_M.gguf, "
                     "meta-models/Muse-Glimmer-30B-GGUF @ 70bf1b61, 16756683904 bytes)");
    return;
  }
  const std::string gguf(gguf_env);
  const fs::path gdir = fs::path(PARITY_GOLDENS_DIR) / "muse_glimmer_30b_q4km";
  const bool dump = std::getenv("VT_DUMP_IDS") != nullptr;
  const bool have_anchor = fs::exists(gdir / "our_ids.npy") &&
                           fs::exists(gdir / "neartie_gap_mnats.npy") &&
                           fs::exists(gdir / "greedy_ids.npy");
  const int64_t N = static_cast<int64_t>(Prompts().size());

  if (!have_anchor) {
    if (!dump) {
      MESSAGE(label << ": SKIP, golden absent; run under VT_DUMP_IDS=1, then "
                       "scripts/llamacpp/llamacpp_oracle.cpp on the same file");
      return;
    }
    fs::create_directories(gdir);
    MESSAGE(label << ": BOOTSTRAP via FromModelDir(" << gguf << ")...");
    auto le = vllm::entrypoints::LoadedEngine::FromModelDir(gguf, GateParams());
    std::vector<int32_t> buf(static_cast<size_t>(N * kTokens), -1);
    for (int64_t i = 0; i < N; ++i) {
      const vllm::RequestOutput out = le->engine().generate(
          Prompts()[static_cast<size_t>(i)], Greedy(kTokens), "boot" + std::to_string(i));
      REQUIRE(out.outputs.size() == 1);
      WriteI32(gdir / ("p" + std::to_string(i) + "_prompt.i32"), out.prompt_token_ids);
      const std::vector<int32_t>& got = out.outputs[0].token_ids;
      REQUIRE(static_cast<int>(got.size()) == kTokens);
      for (int j = 0; j < kTokens; ++j)
        buf[static_cast<size_t>(i * kTokens + j)] = got[static_cast<size_t>(j)];
    }
    WriteI32(gdir / "our_ids.i32", buf);
    MESSAGE(label << ": BOOTSTRAP wrote the prompt ids and our ids under " << gdir.string());
    return;
  }

  const parity::NpyArray g = parity::LoadNpy((gdir / "greedy_ids.npy").string());
  const parity::NpyArray o = parity::LoadNpy((gdir / "our_ids.npy").string());
  const parity::NpyArray gap = parity::LoadNpy((gdir / "neartie_gap_mnats.npy").string());
  REQUIRE(g.dtype == "<i4");
  REQUIRE(o.dtype == "<i4");
  REQUIRE(gap.dtype == "<i4");
  REQUIRE(g.shape.size() == 2);
  REQUIRE(g.shape[0] == N);
  const int64_t T = g.shape[1];
  REQUIRE(T == kTokens);
  REQUIRE(o.shape == g.shape);
  REQUIRE(gap.shape == g.shape);
  const int32_t* gd = AsI32(g);
  const int32_t* od = AsI32(o);
  const int32_t* gapd = AsI32(gap);

  MESSAGE(label << ": loading via FromModelDir(" << gguf << ")...");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(gguf, GateParams());

  int strict_exact = 0, neartie_only = 0, fail = 0;
  int32_t worst_gap = 0;
  int worst_i = -1, worst_j = -1;
  for (int64_t i = 0; i < N; ++i) {
    // The ids the golden was captured on, fed verbatim: the gate compares the
    // forward, never a tokenizer.
    const std::vector<int32_t> prompt = ReadI32(gdir / ("p" + std::to_string(i) + "_prompt.i32"));
    REQUIRE_FALSE(prompt.empty());
    const vllm::RequestOutput out =
        loaded->engine().generate(prompt, Greedy(kTokens), "gate" + std::to_string(i));
    REQUIRE(out.finished);
    REQUIRE(out.outputs.size() == 1);
    const std::vector<int32_t>& got = out.outputs[0].token_ids;
    REQUIRE(static_cast<int64_t>(got.size()) == T);

    int first_div = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != od[i * T + j]) { first_div = static_cast<int>(j); break; }
    }
    REQUIRE_MESSAGE(first_div < 0, label << " anchor drift prompt[" << i << "] tok="
                                         << first_div << " engine="
                                         << (first_div < 0 ? -1 : got[static_cast<size_t>(first_div)])
                                         << " anchor="
                                         << (first_div < 0 ? -1 : od[i * T + first_div]));
    bool exact = true, prompt_ok = true;
    int first_bad = -1;
    for (int64_t j = 0; j < T; ++j) {
      if (got[static_cast<size_t>(j)] != gd[i * T + j]) exact = false;
      const int32_t mn = gapd[i * T + j];
      if (mn > worst_gap) { worst_gap = mn; worst_i = static_cast<int>(i); worst_j = static_cast<int>(j); }
      if (mn > kNearTieMnats) { prompt_ok = false; if (first_bad < 0) first_bad = static_cast<int>(j); }
    }
    if (!prompt_ok) {
      ++fail;
      MESSAGE(label << " FORWARD DIVERGENCE prompt[" << i << "] tok=" << first_bad
                    << " our=" << got[static_cast<size_t>(first_bad)]
                    << " llama.cpp greedy=" << gd[i * T + first_bad]
                    << " gap=" << (gapd[i * T + first_bad] / 1000.0) << " nats");
    } else if (exact) {
      ++strict_exact;
    } else {
      ++neartie_only;
    }
    CHECK(prompt_ok);
  }
  MESSAGE(label << " gate: " << (strict_exact + neartie_only) << "/" << N
                << " prompts PASS (token-exact vs llama.cpp b10451 greedy: " << strict_exact
                << "/" << N << "; near-tie band only: " << neartie_only << "/" << N
                << "; max gap " << (worst_gap / 1000.0) << " nats @ prompt[" << worst_i
                << "] tok=" << worst_j << "; " << fail << " forward-divergent)");
  REQUIRE(fail == 0);
}
