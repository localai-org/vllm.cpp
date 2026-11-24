// Kolibri-1 — typed config params parsed from HfConfig.
//
// (MODEL-TEXT-kolibri-1 W1) Aleph-Alpha Kolibri-1, model_type `kolibri1`,
// mirrored from the aleph-alpha-inference plugin pinned at 049a6a7bd240
// (`aleph_alpha_inference/kolibri1.py`). CPU path only; see
// .agents/specs/kolibri-1-cpu.md.
//
// The architecture is a hybrid SWA/full-attention MoE with RNoPE: sliding-
// window layers carry standard RoPE (theta 10000, full rotary over
// head_dim=128) and full-attention layers carry NO positional encoding at
// all — the inverse of the Mistral-style convention (kolibri1.py:81-83,
// 91-95). Every layer is MoE (384 routed experts, top-6, no renormalisation,
// sigmoid-logit-add routing) with one ungated shared expert.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

struct Kolibri1Params {
  int64_t hidden_size = 0;          // 2560
  int64_t num_hidden_layers = 0;    // 50
  int64_t vocab_size = 0;           // 128000
  int64_t num_attention_heads = 0;  // 48
  int64_t num_key_value_heads = 0;  // 4 (GQA 12:1)
  int64_t head_dim = 0;             // 128; 48*128 = 6144 q-size
  int64_t sliding_window = 0;       // 513, sliding layers only
  bool use_sliding_window = false;  // true in the checkpoint

  // Per-layer attention kind, from config.layer_types: "sliding_attention"
  // or "full_attention". Length == num_hidden_layers. RNoPE: RoPE lives ONLY
  // on the sliding layers.
  std::vector<std::string> layer_types;

  // Kolibri-1.1 additions. Empty (or zero) for 1.0, so every existing arm
  // keeps its exact current behaviour. `indexed_attention` is a THIRD layer
  // kind, deliberately not folded into IsSlidingLayer.
  std::vector<int64_t> index_layers;
  std::vector<int64_t> kv_source_layers;
  std::vector<int64_t> mlp_only_layers;
  int64_t mlp_only_intermediate_size = 0;

  bool IsSlidingLayer(int64_t i) const { return layer_types[i] == "sliding_attention"; }
  bool IsIndexedLayer(int64_t i) const {
    return i >= 0 && i < static_cast<int64_t>(layer_types.size()) &&
           layer_types[static_cast<size_t>(i)] == "indexed_attention";
  }
  bool IsDenseLayer(int64_t i) const {
    for (int64_t l : mlp_only_layers) {
      if (l == i) return true;
    }
    return false;
  }
  // Effective KV owner of layer i: itself for 1.0, the mapped source for 1.1.
  int64_t KvSourceLayer(int64_t i) const {
    if (i >= 0 && i < static_cast<int64_t>(kv_source_layers.size()))
      return kv_source_layers[static_cast<size_t>(i)];
    return i;
  }

  // RoPE: flat rope_theta (R2 — the plugin reads
  // rope_parameters["rope_theta"] through the transformers Qwen3MoeConfig
  // base; the checkpoint ships the flat field and this parser derives the
  // same value from it). Full rotary: rotary_dim == head_dim; a
  // partial_rotary_factor other than 1.0 is refused.
  double rope_theta = 0.0;  // 10000.0
  int64_t rotary_dim = 0;   // == head_dim

  // MoE: every layer is MoE (no dense layers). Routing is sigmoid-logit-add
  // (kolibri1.py:126-142): selection on logits + e_score_correction_bias,
  // weights = sigmoid of the UNBIASED logits, norm_topk_prob=false means NO
  // weight renormalisation.
  int64_t num_experts = 0;                    // 384
  int64_t num_experts_per_tok = 0;            // 6
  int64_t moe_intermediate_size = 0;          // 512
  int64_t shared_expert_intermediate_size = 0;  // 512
  bool norm_topk_prob = false;                // false => renormalize=False

  // Norm / activation
  double rms_norm_eps = 0.0;  // 1e-6
  std::string hidden_act;     // "silu"

  // head_dtype: "float32" in the checkpoint — an attention-head compute-dtype
  // hint the plugin does not read (it relies on vLLM defaults). R5: treated
  // as a model-path annotation, reconciled with the oracle at gate time.
  std::string head_dtype;

  bool tie_word_embeddings = false;  // false: lm_head untied
};

// Parses and validates every Kolibri-1 config field. Throws on anything the
// CPU forward cannot represent.
Kolibri1Params ParseKolibri1Params(const HfConfig& config);

// Validates config (delegates to ParseKolibri1Params, void return).
void ParseKolibri1Config(const HfConfig& config);

// Builds the hybrid KV-cache spec: two groups (full-attention + SWA, uniform
// head_dim 128, window 513 on the sliding group).
v1::KVCacheConfig MakeKolibri1KVCache(const HfConfig& config, int block_size,
                                      int num_blocks);

// Named refusal for the Kolibri-1.1 sparse forward (U3 work). Exposed so the
// refusal policy is unit-testable without constructing a loaded model; the
// production forward calls it before any device dispatch.
void RefuseKolibri11SparseForward(const Kolibri1Params& p);

}  // namespace vllm
