// Synthetic reachability regression only. Published-checkpoint validation is separate.
// Reuses the pinned GGUF metadata and deterministic weights from the public-loader fixture.
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdio>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>
#include "capi/rocm_quant_gather_fixture.h"
#include "vllm.h"
#include "vt/ops.h"

namespace {
struct Result {
  std::array<int32_t, 4> ids{};
  std::vector<float> logits;
  bool invalid = false;
  int calls = 0;
};
void Require(bool value, const std::string& message) {
  if (!value) throw std::runtime_error(message);
}
void Capture(const int32_t*, int32_t, float* logits, int32_t vocab, void* data) {
  auto& result = *static_cast<Result*>(data);
  if (vocab != 128 || result.calls >= 4) { result.invalid = true; return; }
  for (int32_t i = 0; i < vocab; ++i) {
    if (!std::isfinite(logits[i])) result.invalid = true;
    result.logits.push_back(logits[i]);
  }
  ++result.calls;
}
Result Run(const std::string& path, int device, int prompt_size) {
  auto model = vllm_model_params_default();
  model.model_path = path.c_str();
  model.device = device;
  model.block_size = 16;
  model.num_blocks = 16;
  model.max_model_len = 64;
  model.max_num_seqs = 1;
  vllm_engine* raw = nullptr;
  const auto load_status = vllm_engine_load(&model, &raw);
  Require(load_status == VLLM_OK, "public load: " + std::string(vllm_last_error()));
  std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(raw, vllm_engine_free);
  Result result;
  result.logits.reserve(4 * 128);
  auto sampling = vllm_sampling_params_default();
  sampling.temperature = 0;
  sampling.max_tokens = 4;
  sampling.ignore_eos = 1;
  sampling.logits_processor = Capture;
  sampling.logits_processor_user_data = &result;
  std::vector<int32_t> prompt(static_cast<size_t>(prompt_size));
  for (int i = 0; i < prompt_size; ++i) prompt[static_cast<size_t>(i)] = (i * 19 + 1) % 128;
  int32_t count = 0;
  const auto before = vt::GetReferenceTierHits();
  const auto complete_status = vllm_complete_tokens(engine.get(), prompt.data(), prompt_size, &sampling,
                              result.ids.data(), 4, &count, nullptr);
  Require(complete_status == VLLM_OK, "public completion: " + std::string(vllm_last_error()));
  Require(count == 4 && result.calls == 4 && !result.invalid,
          "expected four tokens and finite complete logits");
  if (device == 2) Require(vt::GetReferenceTierHits() == before, "CUDA used CPU reference fallback");
  return result;
}
}
int main() {
  int count = 0;
  if (cudaGetDeviceCount(&count) != cudaSuccess || count == 0) return 77;
  try {
    const rocm_gather_test::Format q4{vt::DType::kQ4_K, 12, "Q4_K"};
    for (uint32_t head_dim : {64u, 128u, 256u}) {
      gguf_test::TempFile file(rocm_gather_test::BuildModel(q4, true, nullptr, nullptr, head_dim));
      for (int prompt : {1, 16, 37}) {
        const auto cpu = Run(file.path(), 1, prompt);
        const auto gpu = Run(file.path(), 2, prompt);
        Require(cpu.ids == gpu.ids, "CPU/CUDA greedy token mismatch");
        std::printf("synthetic public CPU/CUDA head_dim=%u prompt=%d ids=", head_dim, prompt);
        for (auto id : gpu.ids) std::printf(" %d", id);
        std::puts("");
      }
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
