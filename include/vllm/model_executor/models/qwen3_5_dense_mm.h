#pragma once

namespace vt { struct Queue; }
namespace vllm {
struct ForwardLogits;
struct ModelForwardInput;
struct Qwen3_5DenseWeights;
struct HfConfig;
struct MmEmbedInputs;
struct MmForwardBuffers;

// Bounded native image serving. Admission and the registered forward must
// agree so excess clients queue before any unsupported batch reaches XPU.
inline constexpr int kNativeQwen3_5MaxConcurrentRequests = 4;

// Native EXL3 XPU FP16 eager C1-C4 forward over already merged device embeddings
// and axis-major [3,T] positions. Cache-write positions/metadata remain the
// ordinary runner values; M-RoPE selects only rotation coefficients. Inputs
// must be ready on the main queue and their owners retained through its use.
// The caller also owns the usual model, KV and GDN lifetimes. This dependency
// does not register an image-serving hook or qualify full vision/MTP yet.
ForwardLogits Qwen3_5DenseForwardEmbeddings(const ModelForwardInput& input,
                                          const Qwen3_5DenseWeights& weights);

// Embed authoritative token IDs and splice ordered FP16 device visual-row
// slices using the runner's mask. No image activations are downloaded. Borrowed
// source owners must survive their queued copies; the registered encoder hook
// is responsible for recording those uses. Returned storage owns both outputs.
MmForwardBuffers Qwen3_5DenseEmbedMultimodal(
    const Qwen3_5DenseWeights& weights, const HfConfig& config,
    vt::Queue& queue, const MmEmbedInputs& inputs);
}  // namespace vllm
