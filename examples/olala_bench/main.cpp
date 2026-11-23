// olala-bench — the p150-olala-bf16 axis A/B driver (spec
// .agents/specs/p150-benchmark-axes.md, axes A1-A3+A8). Drives the PRODUCTION
// paged-engine decode path (LoadedEngine::FromModelDir → LLMEngine →
// ModelRegistry::Forward → OlalaPagedForward) over the W2d token-gate prompt,
// greedy, 64 output tokens, and reports prompt-eval and decode throughput with
// a prefill-vs-decode split measured from two timed generate calls per leg
// (max_tokens=1 isolates prefill; the 64-token leg's decode time is its
// remainder). No internal-model shortcut: this is the shipped engine path.
//
//   olala-bench --model <bf16.gguf> [--runs 3] [--tokens 64] [--max-len 512]
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vllm/entrypoints/model_loader.h"
#include "vllm/sampling_params.h"

namespace {

// The W2d token-gate prompt (test_olala_paged_engine.cpp kGolden[:6]).
constexpr int64_t kPrompt[] = {151644, 872, 198, 15588, 220, 151645};
constexpr int kNPrompt = 6;

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

vllm::SamplingParams Greedy(int max_tokens) {
  vllm::SamplingParams sp;
  sp.temperature = 0.0;
  sp.max_tokens = max_tokens;
  sp.ignore_eos = true;
  sp.PostInit();
  return sp;
}

}  // namespace

int main(int argc, char** argv) {
  std::string model;
  int runs = 3, tokens = 64, max_len = 512;
  for (int i = 1; i < argc; ++i) {
    std::string f = argv[i];
    auto next = [&]() { return std::string(argv[++i]); };
    if (f == "--model") model = next();
    else if (f == "--runs") runs = std::atoi(next().c_str());
    else if (f == "--tokens") tokens = std::atoi(next().c_str());
    else if (f == "--max-len") max_len = std::atoi(next().c_str());
    else { std::fprintf(stderr, "unknown flag %s\n", f.c_str()); return 2; }
  }
  if (model.empty()) { std::fprintf(stderr, "usage: --model <gguf>\n"); return 2; }

  vllm::entrypoints::EngineParams ep;
  ep.device = vllm::Device::kCPU;
  ep.max_model_len = max_len;
  ep.max_num_seqs = 1;
  const double t0 = NowSec();
  auto loaded = vllm::entrypoints::LoadedEngine::FromModelDir(model, ep);
  const double load_s = NowSec() - t0;
  std::printf("model=%s load_s=%.3f\n", model.c_str(), load_s);

  for (int r = 0; r < runs; ++r) {
    std::vector<int32_t> prompt(kPrompt, kPrompt + kNPrompt);
    double tp = NowSec();
    loaded->engine().generate(prompt, Greedy(1), "olala-bench-p" + std::to_string(r));
    const double prefill_s = NowSec() - tp;
    tp = NowSec();
    const vllm::RequestOutput out = loaded->engine().generate(
        prompt, Greedy(tokens), "olala-bench-d" + std::to_string(r));
    const double total_s = NowSec() - tp;
    const double decode_s = total_s - prefill_s;
    std::printf("run=%d prefill_s=%.4f prefill_tok_s=%.4f decode_s=%.4f "
                "decode_tok_s=%.4f n_out=%zu ids=",
                r, prefill_s, kNPrompt / prefill_s, decode_s,
                (tokens - 1) / decode_s, out.outputs[0].token_ids.size());
    for (int32_t id : out.outputs[0].token_ids) std::printf("%d,", id);
    std::printf("\n");
    std::fflush(stdout);
  }
  return 0;
}
