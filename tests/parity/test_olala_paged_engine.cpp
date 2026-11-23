// vllm.cpp original (NEW-CANDIDATE-PORTS wiring unit, issue
// ISSUE-LOCAL-01M835QTXAG5P2SD26QZYT0Z5W); no upstream mirror.
//
// THE PAGED-ENGINE olala GREEDY REACHABILITY GATE — proof that the registered
// production decode path actually runs, per AGENTS.md §"Nothing lands dead".
// Drives the W2d prompt through the FULL paged engine stack
// (LoadedEngine::FromModelDir → scheduler → ForwardOlalaForCausalLM →
// OlalaPagedForward → on-device... host sampler) and requires the generated
// ids to be EXACT against BOTH:
//   * the W2d bf16 token-gate golden (llama-cpp-olala @ bd20ca9b — the same 12
//     ids test_olala_ops.cpp pins; q8_0-safe2 shares the bf16 sequence), and
//   * OlalaGenerateHost on the same artifact, in-process.
// A drift in either column is a real paged-forward divergence (state gather/
// scatter, page addressing, raw-shift chain) and fails the gate.
//
// Checkpoint-gated: OLALA_BF16_PARITY names the GGUF (default the mounted
// bf16 artifact). Absent, a loud SKIP.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/model_executor/models/olala.h"
#include "vllm/model_executor/models/olala_gguf_config.h"
#include "vllm/model_executor/models/olala_gguf_weights.h"
#include "vllm/model_executor/models/olala_ops.h"
#include "vllm/sampling_params.h"

namespace fs = std::filesystem;

namespace {

#ifndef OLALA_BF16_PARITY_DEFAULT
#define OLALA_BF16_PARITY_DEFAULT \
  "/mnt/models/olala-v1-mini/Olala-v1.0-mini-7A1B-bf16.gguf"
#endif

// The W2d/W2e battery prompt + the llama-cpp-olala bf16 golden continuation
// (test_olala_ops.cpp kGoldenQ6K — the bf16, q6_k and q8_0-safe2 gates share
// this EXACT 12-id sequence).
constexpr int64_t kGolden[] = {151644, 872,    198,    15588,  220,    151645,
                               151644, 77091,  151646, 34484,  151654, 1654};
constexpr int kNGen = 6;

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.ignore_eos = true;
  sp.PostInit();
  return sp;
}

}  // namespace

TEST_CASE("olala paged-engine greedy reachability gate (EXACT vs host + golden)") {
  const char* env = std::getenv("OLALA_BF16_PARITY");
  const std::string gguf = env != nullptr && env[0] != '\0'
                               ? std::string(env)
                               : std::string(OLALA_BF16_PARITY_DEFAULT);
  if (!fs::exists(gguf)) {
    MESSAGE("SKIP: " << gguf
                     << " not mounted (set OLALA_BF16_PARITY to the pinned "
                        "OVHaiLLM/Olala-v1.0-mini-7A1B bf16 GGUF)");
    return;
  }

  // The host block reference on the SAME artifact: the operand the W2d EXACT
  // gate adjudicates. The engine must reproduce it exactly.
  vllm::GgufFile gguf_file = vllm::GgufFile::Open(gguf);
  vllm::HfConfig cfg = vllm::OlalaHfConfigFromGguf(gguf_file);
  vllm::OlalaParams p = vllm::ParseOlalaParams(cfg);
  vllm::OlalaWeights w = vllm::LoadOlalaGgufWeights(gguf_file, cfg);
  std::vector<int64_t> ref_ids(kGolden, kGolden + 6);
  vllm::OlalaGenerateHost(w, p, ref_ids, kNGen);
  REQUIRE(static_cast<int64_t>(ref_ids.size()) == 6 + kNGen);

  // THE PRODUCTION PATH: the registered engine, default configuration.
  vllm::entrypoints::EngineParams ep;
  ep.device = vllm::Device::kCPU;
  ep.max_model_len = 512;  // the battery needs under 64 tokens of context
  ep.max_num_seqs = 1;
  MESSAGE("loading the engine via FromModelDir(" << gguf << ")...");
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(gguf, ep);
  std::vector<int32_t> prompt(kGolden, kGolden + 6);
  const vllm::RequestOutput out =
      loaded->engine().generate(prompt, Greedy(kNGen), "olala-reach");
  REQUIRE(out.finished);
  REQUIRE(out.outputs.size() == 1);
  const std::vector<int32_t>& got = out.outputs[0].token_ids;
  REQUIRE(static_cast<int64_t>(got.size()) == kNGen);

  bool exact_ref = true, exact_golden = true;
  for (int j = 0; j < kNGen; ++j) {
    CAPTURE(j);
    CHECK(got[static_cast<size_t>(j)] == ref_ids[static_cast<size_t>(6 + j)]);
    CHECK(got[static_cast<size_t>(j)] == kGolden[6 + j]);
    if (got[static_cast<size_t>(j)] != ref_ids[static_cast<size_t>(6 + j)])
      exact_ref = false;
    if (got[static_cast<size_t>(j)] != kGolden[6 + j]) exact_golden = false;
  }
  CHECK(exact_ref);
  CHECK(exact_golden);
}
