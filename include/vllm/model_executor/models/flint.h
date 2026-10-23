// RecursiveSI Flint (`FlintForCausalLM`) — typed config params, weight
// structs, and the reference forward declarations.
//
// (MODEL-TEXT-FLINT W1, spec .agents/specs/flint-cpu-gguf.md.) Flint,
// published 2026-10-21: 49 layers, hidden 2048, 32 query heads against 4 KV
// heads at head_dim 128, vocab 150144, rms eps 1e-6. Hybrid FFSSSF
// sliding/full attention (sliding_window 513) with NO RoPE on the
// full-attention layers (config `no_rope_layer_indices`; the GGUF arm carries
// the same fact as the `rope_pattern` array). MoE: 256 experts / 6 per token,
// sigmoid routing WITH renormalisation, one UNGATED shared expert always
// added (`shared_expert_intermediate_size` 1024 = 2 x moe_intermediate 512,
// the GGUF `expert_shared_count` 2), layers 0-1 dense (`mlp_only_layers`).
// Per-head attention sinks (`self_attn.softmax_offset`, one scalar per query
// head) folded into the softmax denominator. Standard pre-norm residual —
// NOT Kolibri-1's sandwich.
//
// Grounding: llama.cpp PR #30928 head `1b44e30775dcb2f93952116b05f89e5bd797fb00`
// (`conversion/flint.py`, `src/models/flint.cpp`) — the pinned
// [`llama-cpp-flint`](../../../../.agents/oracles/llama-cpp-flint.md) oracle —
// and the RecursiveSI `config.json` field names the converter reads. vLLM
// implements nothing for this architecture.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/kv_cache_interface.h"

namespace vllm {

class SafetensorsFile;

struct FlintParams {
  int64_t hidden_size = 0;          // 2048
  int64_t num_hidden_layers = 0;    // 49
  int64_t vocab_size = 0;           // 150144
  int64_t num_attention_heads = 0;  // 32
  int64_t num_key_value_heads = 0;  // 4 (GQA 8:1)
  int64_t head_dim = 0;             // 128
  int64_t sliding_window = 0;       // 513, sliding layers only
  double rope_theta = 0.0;          // 10000.0
  double rms_norm_eps = 0.0;        // 1e-6
  std::string hidden_act;           // "silu"

  // Per-layer attention kind, from config.layer_types: "sliding_attention"
  // or "full_attention". Length == num_hidden_layers.
  std::vector<std::string> layer_types;
  bool IsSlidingLayer(int64_t i) const { return layer_types[i] == "sliding_attention"; }

  // Layers with NO RoPE (config.no_rope_layer_indices): the FULL-attention
  // layers. Sliding layers carry full rotary (rotary_dim == head_dim).
  std::vector<int64_t> no_rope_layer_indices;
  bool LayerHasRope(int64_t i) const;

  // MoE. `mlp_only_layers` must be {0, .., n-1} (the leading dense layers);
  // the converter refuses anything else and so does this parser.
  std::vector<int64_t> mlp_only_layers;
  int64_t num_experts = 0;                    // 256
  int64_t num_experts_per_tok = 0;            // 6
  int64_t moe_intermediate_size = 0;          // 512
  int64_t shared_expert_intermediate_size = 0;  // 1024
  bool IsDenseLayer(int64_t i) const;

  bool tie_word_embeddings = false;  // false: untied lm_head
};

// Parses and validates every Flint config field. Throws on anything the
// forward cannot represent.
FlintParams ParseFlintParams(const HfConfig& config);

// Validates config (delegates to ParseFlintParams, void return).
void ParseFlintConfig(const HfConfig& config);

// Builds the hybrid KV-cache spec: two groups (sliding-window with window 513
// and full-attention), uniform head_dim and num_key_value_heads across both.
v1::KVCacheConfig MakeFlintKVCache(const HfConfig& config, int block_size,
                                   int num_blocks);

// One linear projection, transposed to Matmul-B [K=in, N=out] bf16.
struct FlintLinear {
  OwnedTensor w;  // bf16 [K, N], nk == false

  bool Empty() const { return w.rank == 0; }
};

struct FlintExpertWeights {
  FlintLinear gate_proj;  // [H, I]
  FlintLinear up_proj;    // [H, I]
  FlintLinear down_proj;  // [I, H]
};

// SwiGLU dense MLP (layers 0-1): the same three projections at the dense
// intermediate size (6144).
struct FlintMlpWeights {
  FlintLinear gate_proj;
  FlintLinear up_proj;
  FlintLinear down_proj;
};

struct FlintAttnWeights {
  FlintLinear q_proj;  // [H, hq*dh]
  FlintLinear k_proj;  // [H, hkv*dh]
  FlintLinear v_proj;  // [H, hkv*dh]
  FlintLinear o_proj;  // [hq*dh, H]
  // Per-head q/k RMS norms over the head dim, applied BEFORE RoPE.
  OwnedTensor q_norm;  // bf16 [dh]
  OwnedTensor k_norm;  // bf16 [dh]
  // Per-head attention sinks (`self_attn.softmax_offset`): one scalar per
  // query head, added as exp(sink) to the softmax denominator.
  OwnedTensor sinks;   // bf16 [hq]
};

struct FlintMoeWeights {
  OwnedTensor router_gate;  // bf16 [H, E] (transposed from disk [E, H])
  std::vector<FlintExpertWeights> experts;  // [num_experts]
  FlintMlpWeights shared_expert;            // ungated, always added
};

struct FlintLayerWeights {
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  FlintAttnWeights attn;
  // Exactly one of dense_mlp / moe is populated, per IsDenseLayer.
  FlintMlpWeights dense_mlp;
  FlintMoeWeights moe;
};

struct FlintWeights {
  FlintParams params;
  OwnedTensor embed_tokens;  // bf16 [vocab, H]
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab] Matmul-B, untied
  std::vector<FlintLayerWeights> layers;
};

// Accounting result: every on-disk tensor must be classified.
struct FlintAccounting {
  std::vector<std::string> missing;
  std::vector<std::string> duplicated;
  std::vector<std::string> unaccounted;
};

// Enumerates every expected CHECKPOINT-side tensor name for `p`.
std::vector<std::string> EnumerateFlintTensors(const FlintParams& p);

// Classifies every on-disk name against the expected set.
FlintAccounting AccountFlintTensors(const FlintParams& p,
                                    const std::vector<std::string>& present);

// Loads all weights from the safetensors shards. Performs full accounting
// before materialization; throws on missing/duplicated/unaccounted tensors.
FlintWeights LoadFlintWeights(const std::vector<SafetensorsFile>& shards,
                              const HfConfig& config);

// The CPU REFERENCE forward arm (spec step 2). Single-sequence, host-side:
// the numerics mirror the pinned oracle exactly — q/k RMSNorm over the head
// dim BEFORE RoPE, attention scores in f32 with the per-head sink folded into
// the softmax denominator (Σexp(S_j)·V_j / (Σexp(S_j) + exp(sink))), RoPE on
// the sliding layers only, sigmoid-gated top-6 routing WITH renormalisation,
// the ungated shared expert always added, layers 0-1 dense. Linear outputs
// and norms round to bf16 (the model dtype every layer inherits); the score,
// router and softmax f32 wideness is the architecture's own, named in the
// spec. The engine's PAGED KV-cache integration is a later wave of the row
// (see the spec's ## Owed and the commit body): this arm keeps its own host
// cache, which is why it refuses multi-sequence input BY NAME.
class FlintModel {
 public:
  // token_ids/positions: one flattened single sequence (query_start_loc with
  // exactly one request). Returns [T, vocab] f32 logits.
  static std::vector<float> Forward(const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& positions,
                                    const FlintWeights& weights,
                                    const HfConfig& config);
};

}  // namespace vllm
