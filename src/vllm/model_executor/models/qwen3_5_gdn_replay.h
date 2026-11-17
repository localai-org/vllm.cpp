// Private bounded diagnostic entry points; no public C ABI or serving mode.
#pragma once
#include <cstdint>
#include <vector>
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {
struct HfConfig;
struct GdnStateCache;
namespace v1 {
struct CommonAttentionMetadata;
struct GDNAttentionMetadata;
}

// Exactly the normal dense GDN loader with its existing FP16 storage policy.
GdnLayerWeights LoadQwen3_5DenseGdnFp16ForDiagnostics(
    const TensorResolver& get, const std::string& layer_base);

// Invoke the production paged block on already normalized FP16 inputs and
// caller-owned FP16 Conv/FP32 SSM caches. Return one selected FP16 output row;
// mutated caches remain on device for the caller's bounded observation.
std::vector<uint16_t> ReplayQwen3_5GdnBlockFp16ForDiagnostics(
    vt::Queue& queue, const GdnLayerWeights& weights, const HfConfig& config,
    const vt::Tensor& normalized, const GdnStateCache& state,
    const v1::CommonAttentionMetadata& attention,
    const v1::GDNAttentionMetadata& gdn, const std::vector<int32_t>& positions,
    int64_t selected_token_row);
}  // namespace vllm
