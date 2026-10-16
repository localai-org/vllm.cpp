// Platform values follow vLLM platforms/xpu.py @ e126687a9a. Kernels and USM
// ownership are native VT implementations; Torch's memory allocator is not used.
#include "vllm/platforms/interface.h"
#include "vllm/v1/attention/registry.h"
#include "vt/xpu.h"
#include <cstdlib>
#include <string_view>

namespace vllm::platforms {
namespace {
// Native generic paged attention, with the engine's NHD cache layout. This
// descriptor does not borrow FlashAttention's kernel/capability declarations.
class XpuAttentionBackend final : public v1::AttentionBackend {
 public:
  std::string get_name() const override { return "XPU_ATTN"; }
  std::vector<int64_t> get_kv_cache_shape(int64_t blocks, int64_t page, int64_t heads,
                                        int64_t dim, const std::string&) const override {
    return {blocks, 2, page, heads, dim};
  }
  std::vector<DType> supported_dtypes() const override { return {DType::kF32, DType::kBF16, DType::kF16}; }
  std::vector<std::string> supported_kv_cache_dtypes() const override {
    return {"auto", "float16", "bfloat16", "fp8", "fp8_e4m3"};
  }
  std::vector<int> get_supported_kernel_block_sizes() const override { return {16}; }
  bool supports_head_size(int head_size) const override { return head_size > 0 && head_size <= 256; }
  bool supports_non_causal() const override { return true; }
  bool supports_sliding_window() const override { return true; }
};
const v1::AttentionBackendRegistrar attention{
    DeviceType::kXPU, "XPU_ATTN", []() -> std::unique_ptr<v1::AttentionBackend> {
      return std::make_unique<XpuAttentionBackend>();
    }};
class XpuPlatform final : public Platform {
 public:
  DeviceType device_type() const override { return DeviceType::kXPU; }
  Backend& backend() const override { return vt::GetBackend(DeviceType::kXPU); }
  DeviceCapability get_device_capability() const override { return {}; }
  std::vector<DType> supported_dtypes() const override { return {DType::kBF16, DType::kF16, DType::kF32}; }
  bool needs_weight_staging() const override { return true; }
  bool support_static_graph_mode() const override {
    const char* setting = std::getenv("VT_XPU_GRAPH");
    return setting && std::string_view(setting) == "1" && backend().SupportsGraphCapture();
  }
  bool static_graph_requires_persistent_inputs() const override { return true; }
  int max_static_graph_batch_size() const override { return 4; }
  ResidencyPolicy residency_policy() const override {
    ResidencyPolicy p;
    p.release_host_weights_after_upload = true;
    p.device_memory_total_bytes = vt::xpu::GetMemoryInfo().total_bytes;
    return p;
  }
  // Native text path: EXL3 projections, F32 GDN state and float/E4M3 paged KV.
  // Compact MTP3 is implemented in the bounded tested text envelope.
  // Vision is unsupported; full reference qualification remains incomplete.
  bool supports_model_architecture(std::string_view architecture) const override {
    return architecture == "Qwen3_5ForConditionalGeneration";
  }
  std::vector<std::string> get_attn_backend_priority(const AttnSelectorConfig& cfg) const override {
    if (cfg.use_mla || cfg.use_sparse) return {};
    return {"XPU_ATTN"};
  }
};
struct Registrar {
  Registrar() {
    if (vt::xpu::DeviceCount() == 0) return;
    static XpuPlatform platform;
    RegisterPlatform(DeviceType::kXPU, &platform);
  }
} registrar;
}  // namespace
}  // namespace vllm::platforms
