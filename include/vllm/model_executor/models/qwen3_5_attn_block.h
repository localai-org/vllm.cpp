#pragma once

#include <memory>
#include <optional>
#include <vector>

#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"

namespace vllm {

// Owning handles for the existing model-owned full-attention implementation.
// Metadata is uploaded once per step and must outlive every consuming layer.
struct FullAttnStepInputs { std::shared_ptr<void> impl; };
struct FullAttnBlockOutput {
  vt::Tensor tensor;
  std::shared_ptr<void> storage;
};

FullAttnStepInputs BuildFullAttnStepInputs(
    vt::Queue& queue, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& metadata, const HfConfig& config,
    std::optional<vt::DType> activation_dtype = std::nullopt);

FullAttnBlockOutput RunFullAttnBlockPaged(
    vt::Queue& queue, const FullAttnLayerWeights& weights,
    const HfConfig& config, const vt::Tensor& hidden,
    const FullAttnStepInputs& step,
    const v1::CommonAttentionMetadata& metadata, const PagedKvCache& cache,
    int64_t tokens, std::optional<vt::DType> activation_dtype = std::nullopt);

}  // namespace vllm
