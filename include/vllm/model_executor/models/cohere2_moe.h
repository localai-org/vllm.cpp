// Cohere2MoeForCausalLM (North, `CohereLabs/North-Mini-Code-1.0`): the parsed
// parameters, the weight set, the loader and the forward.
//
// Ported from vllm/model_executor/models/cohere2_moe.py @ a7c23ac96d (545 lines)
// with the Cohere LayerNorm of commandr.py @ the same pin. The spec is
// .agents/specs/cohere2-moe.md; every mechanism below cites its upstream line.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;

// Every field the forward reads, resolved ONCE from the checkpoint config with
// the defaults of transformers' `Cohere2MoeConfig` (the HF config class vLLM
// instantiates, so an absent key reaches `cohere2_moe.py` as that default, never
// as the `getattr` fallback written in the model file).
struct Cohere2MoeParams {
  int64_t hidden_size = 0;
  int64_t num_layers = 0;
  int64_t vocab_size = 0;
  int64_t num_heads = 0;
  int64_t num_kv_heads = 0;
  int64_t head_dim = 0;
  // Routed-expert width, and the unit of the shared-expert width
  // (`intermediate_size * num_shared_experts`, cohere2_moe.py:291).
  int64_t intermediate_size = 0;
  // Dense-prefix MLP width (cohere2_moe.py:351-360); intermediate_size when the
  // key is absent or null (Cohere2MoeMLP falls back at :120-124).
  int64_t prefix_dense_intermediate_size = 0;
  int64_t num_experts = 0;
  int64_t top_k = 0;
  int64_t num_shared_experts = 0;
  // shared_expert_combination_strategy == "average": the WHOLE MoE output
  // (routed + shared) is halved (cohere2_moe.py:326-327). Only meaningful when
  // num_shared_experts > 0 (:301-303 leaves the strategy None otherwise).
  bool shared_average = false;
  // expert_selection_fn == "sigmoid" selects token_choice_with_bias
  // (cohere2_moe.py:270-276, :56-72); anything else is FusedMoE's default
  // softmax top-k.
  bool sigmoid_router = false;
  bool norm_topk_prob = true;  // renormalize (cohere2_moe.py:311)
  // select_norm_impl (cohere2_moe.py:97-103): RMSNorm when rms_norm_eps is set,
  // else the weight-only Cohere LayerNorm with layer_norm_eps.
  bool use_rms_norm = false;
  float norm_eps = 1e-5F;
  double logit_scale = 0.0625;
  double rope_theta = 10000.0;
  int64_t max_position = 0;
  // Per layer, from layer_types and mlp_layer_types.
  // `window`: `sliding_window + 1` on a sliding_attention layer (:205-211),
  // nullopt on a full-attention layer. `rope`: `sliding_window or force_rope`
  // (:242). `dense`: mlp_layer_types[l] == "dense" (:351).
  std::vector<std::optional<int64_t>> window;
  std::vector<bool> rope;
  std::vector<bool> dense;
};

// Resolve and validate. Refuses by name every configuration the pinned file
// cannot run or that this port does not carry (use_qk_norm, rope scaling, a
// quantization_config, a sliding layer without a window).
Cohere2MoeParams ParseCohere2MoeParams(const HfConfig& config);

struct Cohere2MoeAttnWeights {
  OwnedTensor qkv_proj;  // raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v)
  OwnedTensor o_proj;    // raw-NK [H, Hq*Dh]
};

// A SwiGLU MLP: the dense prefix MLP, one shared-expert MLP, or one routed
// expert. gate|up are stacked into one [2I, H] owner (cohere2_moe.py:484-487).
struct Cohere2MoeMlpWeights {
  OwnedTensor gate_up_proj;  // raw-NK [2I, H]
  OwnedTensor down_proj;     // raw-NK [H, I]
};

struct Cohere2MoeSparseWeights {
  OwnedTensor router;                         // raw-NK [E, H] (`mlp.gate`)
  std::vector<Cohere2MoeMlpWeights> experts;  // E routed experts
  bool has_shared = false;
  Cohere2MoeMlpWeights shared;  // [2*I*num_shared, H] / [H, I*num_shared]
};

struct Cohere2MoeLayerWeights {
  OwnedTensor input_layernorm;  // [H]; the ONE norm of the parallel block
  Cohere2MoeAttnWeights attn;
  bool dense = false;
  Cohere2MoeMlpWeights mlp;     // valid iff dense
  Cohere2MoeSparseWeights moe;  // valid iff !dense
};

struct Cohere2MoeWeights {
  Cohere2MoeParams params;
  // The activation dtype of the forward. bf16 is the checkpoint dtype and the
  // production path. f32 is the ARITHMETIC arm of the synthetic and real-tensor
  // gates only: the loader never sets it, so no served path is wider than the
  // checkpoint.
  vt::DType compute_dtype = vt::DType::kBF16;
  OwnedTensor embed_tokens;  // [vocab, H]; also the tied logits weight
  OwnedTensor final_norm;    // [H]
  OwnedTensor rope_cos_sin;  // [max_position, head_dim], [cos|sin], compute dtype
  std::vector<Cohere2MoeLayerWeights> layers;
};

// get_rope(head_dim, max_position, rope_theta, is_neox_style=False)'s cos|sin
// cache (cohere2_moe.py:198-203), rows [0, rows) in `dtype`.
OwnedTensor BuildCohere2MoeRopeCache(const Cohere2MoeParams& params, int64_t rows,
                                     vt::DType dtype);

// The whole checkpoint. Every shipped tensor is claimed or refused by name;
// `lm_head.*` is dropped (cohere2_moe.py:489).
Cohere2MoeWeights LoadCohere2MoeWeights(const std::vector<SafetensorsFile>& shards,
                                        const HfConfig& config);

// One decoder layer from `shards` (the real-tensor gate loads a subset).
Cohere2MoeLayerWeights LoadCohere2MoeLayerWeights(
    const std::vector<SafetensorsFile>& shards, const Cohere2MoeParams& params,
    int64_t layer);

// Every tensor name a checkpoint for `params` ships, in load order. The
// structural gate compares it with a released index.
std::vector<std::string> EnumerateCohere2MoeTensors(const Cohere2MoeParams& params);

class Cohere2MoeModel {
 public:
  // Paged forward over `attn_kv` (one cache per layer). Returns f32 logits
  // [rows, vocab], rows = logits_indices.size() or T.
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Cohere2MoeWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const Cohere2MoeWeights& weights,
      vt::Queue& queue, const std::vector<int32_t>& logits_indices = {});

  // ONE decoder layer of the same code path, over a single-request prefill of
  // `hidden_in` [T, H] (f32 host values, fed at the compute dtype) at
  // `positions`, into a scratch cache. Returns the layer output [T, H] as f32.
  // `layer_idx` selects the window/RoPE/MLP kind from `weights.params`; the
  // layer weights are passed explicitly so a gate can load a subset of layers.
  static std::vector<float> DecoderLayer(const Cohere2MoeWeights& weights,
                                         const Cohere2MoeLayerWeights& layer,
                                         int64_t layer_idx,
                                         const std::vector<float>& hidden_in,
                                         const std::vector<int32_t>& positions,
                                         vt::Queue& queue);

  // The final norm alone, [T, H] -> [T, H] f32.
  static std::vector<float> FinalNorm(const Cohere2MoeWeights& weights,
                                      const std::vector<float>& hidden_in,
                                      vt::Queue& queue);
};

void ParseCohere2MoeForCausalLMConfig(const HfConfig& config);

v1::KVCacheConfig MakeCohere2MoeKVCache(const HfConfig& config, int block_size,
                                        int num_blocks);

}  // namespace vllm
