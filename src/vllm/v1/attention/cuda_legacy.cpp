// Local capability descriptor for existing portable CUDA paged attention.
// Secondary: llama.cpp 62318cd57, ggml/src/ggml-cuda/fattn.cu:757-775.
// This registers metadata only; AttnBlock and vt::PagedAttention own execution.
#include "vllm/v1/attention/registry.h"
#include <stdexcept>

namespace vllm::v1 {
namespace {
class CudaLegacyAttentionBackend final : public AttentionBackend {
 public:
  std::string get_name() const override { return "CUDA_LEGACY"; }
  std::vector<vt::DType> supported_dtypes() const override {
    return {vt::DType::kBF16, vt::DType::kF32};
  }
  std::vector<std::string> supported_kv_cache_dtypes() const override {
    return {"auto", "bfloat16"};
  }
  std::vector<int> get_supported_kernel_block_sizes() const override { return {16}; }
  bool supports_head_size(int size) const override { return size > 0 && size <= 256 && size % 8 == 0; }
  bool supports_compute_capability(const platforms::DeviceCapability& cap) const override {
    return (cap.major == 6 && (cap.minor == 0 || cap.minor == 1)) || (cap.major == 7 && cap.minor == 0);
  }
  bool supports_kv_connector() const override { return false; }
  bool supports_attn_type(const std::string& type) const override { return type == "decoder"; }
  std::vector<int64_t> get_kv_cache_shape(int64_t blocks, int64_t size, int64_t heads,
                                        int64_t dim, const std::string& = "auto") const override {
    if (size % 16 != 0) throw std::invalid_argument("Block size must be a multiple of 16.");
    return {blocks, 2, size, heads, dim};
  }
};
std::unique_ptr<AttentionBackend> MakeLegacy() { return std::make_unique<CudaLegacyAttentionBackend>(); }
const AttentionBackendRegistrar kLegacy{vt::DeviceType::kCUDA, "CUDA_LEGACY", MakeLegacy};
}
}
