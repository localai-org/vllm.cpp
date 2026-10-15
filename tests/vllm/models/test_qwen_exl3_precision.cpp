#include <doctest/doctest.h>

#include "vllm/model_executor/models/qwen3_5_dense.h"

TEST_CASE("Qwen EXL3 resolves pinned FP16 execution despite BF16 export dtype") {
  vllm::HfConfig config;
  config.torch_dtype = "bfloat16";
  config.mamba_ssm_dtype = "float32";
  const auto policy = vllm::ResolveQwen3_5DensePrecision(config, false, true);
  CHECK(policy.activation == vt::DType::kF16);
  CHECK(policy.dense_weight == vt::DType::kF16);
  CHECK(policy.kv_auto == vt::DType::kF16);
  CHECK(policy.gdn_conv_state == vt::DType::kF16);
  CHECK(policy.gdn_recurrent_state == vt::DType::kF32);
  CHECK(policy.sampler == vt::DType::kF32);
  const auto ordinary = vllm::ResolveQwen3_5DensePrecision(config, false);
  CHECK(ordinary.activation == vt::DType::kBF16);
  CHECK_THROWS_AS(vllm::ResolveQwen3_5DensePrecision(config, true), std::runtime_error);
  config.torch_dtype = "float16";
  CHECK(vllm::ResolveQwen3_5DensePrecision(config, true).activation == vt::DType::kF16);
}

TEST_CASE("Qwen EXL3 precision rejects conflicting formats and non-FP32 recurrence") {
  vllm::HfConfig config;
  config.torch_dtype = "float16";
  config.mamba_ssm_dtype = "float32";
  CHECK_THROWS_AS(vllm::ResolveQwen3_5DensePrecision(config, true, true), std::runtime_error);
  config.mamba_ssm_dtype = "float16";
  CHECK_THROWS_AS(vllm::ResolveQwen3_5DensePrecision(config, false, true), std::runtime_error);
}
