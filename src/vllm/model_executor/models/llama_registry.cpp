// Llama-3.x (`LlamaForCausalLM`) registry TU — the cross-family dense additive
// bring-up. Self-registers "LlamaForCausalLM" via REGISTER_VLLM_MODEL and owns the
// arch-specific entry points: the config hook, the full-attention-ONLY KV-cache
// spec, the LoadedModel subclass, and the factory. Mirrors the qwen3_dense.cpp
// seam (new TU + one in-TU REGISTER line -> ZERO shared-array edit).
//
// The forward + KV topology are REUSED VERBATIM from the Qwen3-dense path
// (LlamaModel == Qwen3DenseModel): Llama is that forward with qk-norm skipped and
// llama3 rope-scaling applied, both handled additively inside the shared
// dense_attn_block.h AttnBlock. So this TU only wires the loader + forward hooks.
// See .agents/specs/sweep-llama-3.2.md.
//
// Registry ALIASES (mirror vLLM 0.25.0 registry.py which maps these arch strings
// onto ("llama", "LlamaForCausalLM") — the SAME model class): "InternLM3ForCausalLM"
// (registry.py:134) is InternLM3, a plain Llama arch (RMSNorm + NeoX + GQA +
// SiLU-SwiGLU, dynamic-NTK rope) — NOT InternLM2 (which has the fused-wqkv
// interleaved split); the internlm3 checkpoint config carries no sliding_window.
// Both aliases reuse kLlamaFactory/kLlamaInfo VERBATIM — zero delta beyond the
// arch-string registration. NOTE on Yi: modern Yi checkpoints (Yi-1.5-*, Yi-Coder-*)
// declare architectures:["LlamaForCausalLM"] directly, so they resolve to the
// llama_dense registration below with NO alias; vLLM 0.25.0 registers no
// "YiForCausalLM", so we add none (mirror the oracle). See
// .agents/specs/sweep-recent-dense-batch.md (trivial tail).
#include "vllm/model_executor/models/model_registry.h"

#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/llama.h"
#include "vllm/model_executor/models/llama_gguf_weights.h"
#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits (shared carrier)
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::DeviceTokenIdsScope
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py _ModelInfo for Llama: text generation, NOT hybrid (pure
// full-attention), NOT multimodal (text-only checkpoint).
inline constexpr ModelInfo kLlamaInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// Opaque owned model: holds the loaded Llama dense weights (shared dense
// container); the forward reuses the Qwen3-dense forward machinery.
class LlamaLoadedModel final : public LoadedModel {
 public:
  LlamaLoadedModel(const ModelRegistration& registration, LlamaWeights weights)
      : LoadedModel(registration), weights_(std::move(weights)) {}

  const LlamaWeights& weights() const { return weights_; }
  // W7: the shared pure-dense decode CUDA-graph driver state (LlamaModel ==
  // Qwen3DenseModel, so this is the SAME driver the Qwen3-dense path uses).
  std::unique_ptr<Qwen3DenseDecodeGraph>& decode_graph() { return decode_graph_; }

 private:
  LlamaWeights weights_;
  std::unique_ptr<Qwen3DenseDecodeGraph> decode_graph_;
};

std::unique_ptr<LoadedModel> LoadLlamaForCausalLM(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    if (!source.gguf) throw std::runtime_error("GGUF model source is empty");
    const auto policy = GgufLoadPolicy::FromEnv(source.device, detail::ActDType(source.device));
    return std::make_unique<LlamaLoadedModel>(registration,
        LoadLlamaFromGguf(*source.gguf, config, &policy));
  }
  // Existing BF16 safetensors arm.
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture LlamaForCausalLM does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<LlamaLoadedModel>(
      registration, LoadLlamaForCausalLMWeights(*source.safetensors, config));
}

void PrepareLlamaForCausalLM(LoadedModel& model, const HfConfig& config,
                             vt::Queue& queue) {
  (void)model;
  (void)config;
  (void)queue;
}

ForwardLogits ForwardLlamaForCausalLM(LoadedModel& model,
                                      const ModelForwardInput& input) {
  // ROW-SERVE-ASYNC-DENSE-MIRROR (sibling scope): same guard qwen3_dense.cpp
  // establishes. This family routes through the SAME shared EmbedInto (decode
  // graph + eager Forward/ForwardDevice), so without it the async runner's
  // device-resident ids are ignored and the stale host `token_ids` races the
  // combine's device write — the #31 batch-1 token-0 degeneration. Null on every
  // non-async-CUDA path, RAII-scoped, byte-identical when the mirror is off.
  const detail::DeviceTokenIdsScope device_ids_scope(
      input.device_token_ids, static_cast<int64_t>(input.token_ids.size()));
  auto& llama = ModelAs<LlamaLoadedModel>(model, "LlamaForCausalLM");
  const LlamaWeights& weights = llama.weights();
  // Shared pure-dense decode CUDA-graph (opt-in via VLLM_CPP_QWEN3_DENSE_DECODE_
  // GRAPH); std::nullopt falls through to the byte-identical eager path below.
  if (auto fl = DenseDecodeGraphForward(llama.decode_graph(), weights, input))
    return std::move(*fl);
  // DEVICE-resident logits (sampler-on-device) on the gather path; HOST logits on
  // the opt-out. Llama is pure full-attention (input.gdn_* unused).
  if (input.gather_logits) {
    return LlamaModel::ForwardDevice(input.token_ids, input.positions,
                                     input.attn_meta, input.attn_kv, weights,
                                     input.config, input.queue,
                                     input.logits_indices);
  }
  return HostLogits(
      LlamaModel::Forward(input.token_ids, input.positions, input.attn_meta,
                          input.attn_kv, weights, input.config, input.queue,
                          input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kLlamaFactory{
    .parse_config = &ParseLlamaForCausalLMConfig,
    .load_weights = &LoadLlamaForCausalLM,
    .prepare = &PrepareLlamaForCausalLM,
    .forward = &ForwardLlamaForCausalLM,
    .make_kv_cache = &MakeLlamaForCausalLMKVCache,
    .is_dense_model = true,
    .consumes_device_token_ids = true,
};

}  // namespace

void ParseLlamaForCausalLMConfig(const HfConfig& config) {
  // LoadHfConfig already materializes + validates every consumed Llama field
  // (including the llama3 rope_scaling dictionary). No-op hook (mirrors
  // ParseQwen3ForCausalLMConfig) — the family normalization/validation seam.
  (void)config;
}

v1::KVCacheConfig MakeLlamaForCausalLMKVCache(const HfConfig& config,
                                              int block_size, int num_blocks) {
  // Pure dense: exactly ONE full-attention KV group, NO MambaSpec/GDN group.
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::KVCacheConfig kv;
  kv.num_blocks = num_blocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<v1::FullAttentionSpec>(
          block_size, num_kv_heads, head_dim, v1::ResolveKvCacheDType()));
  return kv;
}

REGISTER_VLLM_MODEL(llama_dense, "LlamaForCausalLM", kLlamaFactory, kLlamaInfo)

// InternLM3 (`InternLM3ForCausalLM`, registry.py:134) is a plain Llama arch in vLLM
// 0.25.0 — the internlm3-8b-instruct checkpoint is RMSNorm + NeoX + GQA(kv=2) +
// SiLU-SwiGLU with dynamic-NTK rope (factor 6.0, identity within the trained
// window), no biases, untied lm_head — all handled by the shared dense path. Alias
// only; zero forward/loader delta.
REGISTER_VLLM_MODEL(internlm3_llama, "InternLM3ForCausalLM", kLlamaFactory, kLlamaInfo)

}  // namespace vllm
