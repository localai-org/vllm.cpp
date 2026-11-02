// Real-checkpoint gate through the public ABI. No synthetic model is created.
// Usage: test_cuda_legacy_capi MODEL.gguf PROMPT.i32 GENERATED_TOKENS [REPETITIONS]
#include "vllm.h"

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

int main(int argc, char** argv) {
  if (argc == 1) return 77;  // The operator must supply a real pinned artifact.
  try {
    if (argc < 4 || argc > 5) throw std::runtime_error("expected MODEL PROMPT.i32 GENERATED [REPETITIONS]");
    const int generated = std::stoi(argv[3]);
    const int repetitions = argc == 5 ? std::stoi(argv[4]) : 1;
    if (generated < 1 || generated > 1024 || repetitions < 1 || repetitions > 100)
      throw std::runtime_error("invalid generated-token count or repetitions");
    std::ifstream input(argv[2], std::ios::binary | std::ios::ate);
    const auto bytes = input.tellg();
    if (!input || bytes <= 0 || bytes % 4 != 0 || bytes / 4 > 1024)
      throw std::runtime_error("prompt must contain 1 to 1024 little-endian int32 token IDs");
    input.seekg(0);
    std::vector<int32_t> prompt(static_cast<size_t>(bytes) / 4);
    for (auto& id : prompt) {
      unsigned char b[4];
      if (!input.read(reinterpret_cast<char*>(b), 4)) throw std::runtime_error("short prompt read");
      const uint32_t value = uint32_t(b[0]) | uint32_t(b[1]) << 8 | uint32_t(b[2]) << 16 | uint32_t(b[3]) << 24;
      if (value > static_cast<uint32_t>(std::numeric_limits<int32_t>::max()))
        throw std::runtime_error("negative prompt token ID");
      id = static_cast<int32_t>(value);
    }
    auto model = vllm_model_params_default();
    model.model_path = argv[1];
    model.device = 2;  // Require CUDA. A CPU replacement must fail.
    model.max_num_seqs = 1;
    model.max_model_len = 256;
    if (prompt.size() + static_cast<size_t>(generated) > 256)
      throw std::runtime_error("prompt and completion exceed the 256-token gate context");
    model.block_size = 16;
    model.num_blocks = 32;  // Include the allocator's reserved null block.
    model.kv_cache_memory_bytes = 0;
    vllm_engine* raw = nullptr;
    if (vllm_engine_load(&model, &raw) != VLLM_OK) throw std::runtime_error(vllm_last_error());
    std::unique_ptr<vllm_engine, decltype(&vllm_engine_free)> engine(raw, vllm_engine_free);
    auto sampling = vllm_sampling_params_default();
    sampling.temperature = 0;
    sampling.max_tokens = generated;
    sampling.ignore_eos = 1;
    sampling.has_seed = 1;
    sampling.seed = 0;
    std::vector<int32_t> ids(static_cast<size_t>(generated));
    for (int repetition = 0; repetition < repetitions; ++repetition) {
      int32_t count = 0;
      const auto start = std::chrono::steady_clock::now();
      if (vllm_complete_tokens(engine.get(), prompt.data(), static_cast<int32_t>(prompt.size()),
                               &sampling, ids.data(), generated, &count, nullptr) != VLLM_OK)
        throw std::runtime_error(vllm_last_error());
      const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
      if (count != generated) throw std::runtime_error("completion returned fewer tokens than requested");
      std::printf("repetition=%d prompt_tokens=%zu generated_tokens=%d seconds=%.9f ids=", repetition, prompt.size(), count, seconds);
      for (int32_t i = 0; i < count; ++i) std::printf("%s%d", i ? "," : "", ids[static_cast<size_t>(i)]);
      std::puts("");
    }
  } catch (const std::exception& error) {
    std::fprintf(stderr, "%s\n", error.what());
    return 1;
  }
}
