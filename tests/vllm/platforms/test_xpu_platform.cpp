#include <doctest/doctest.h>
#include "vllm/platforms/interface.h"
#include "vllm/v1/attention/registry.h"

TEST_CASE("XPU platform selects native generic attention without FA2 or unsupported cache formats") {
  auto& platform = vllm::platforms::GetPlatform(vt::DeviceType::kXPU);
  CHECK(platform.supports_model_architecture("Qwen3_5ForConditionalGeneration"));
  CHECK_FALSE(platform.supports_model_architecture("Qwen3_5MoeForConditionalGeneration"));
  CHECK_FALSE(platform.supports_fa2_attention());
  CHECK_FALSE(platform.get_device_capability().present());
  CHECK(platform.needs_weight_staging());
  CHECK_FALSE(platform.host_memory_is_device_addressable());
  vllm::platforms::AttnSelectorConfig cfg;
  cfg.head_size = 256; cfg.num_heads = 24; cfg.block_size = 16;
  cfg.dtype = vt::DType::kF32; cfg.kv_cache_dtype = "bfloat16";
  CHECK(vllm::v1::SelectAttentionBackendName(platform, "", cfg) == "XPU_ATTN");
  CHECK_NOTHROW(vllm::v1::CheckKvCacheShape(vt::DeviceType::kXPU, "XPU_ATTN", 4, 16, 4, 256, false));
  cfg.kv_cache_dtype = "fp8";
  CHECK(vllm::v1::SelectAttentionBackendName(platform, "", cfg) == "XPU_ATTN");
  cfg.kv_cache_dtype = "fp8_e4m3";
  CHECK(vllm::v1::SelectAttentionBackendName(platform, "", cfg) == "XPU_ATTN");
  cfg.kv_cache_dtype = "fp8_e5m2";
  CHECK_THROWS(vllm::v1::SelectAttentionBackendName(platform, "", cfg));
  cfg.kv_cache_dtype = "auto"; cfg.use_mla = true;
  CHECK_THROWS(vllm::v1::SelectAttentionBackendName(platform, "", cfg));
}
