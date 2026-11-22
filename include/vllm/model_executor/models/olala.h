// vllm.cpp ORIGINAL — the `olala` (OlalaForCausalLM) model TU declarations, row
// NEW-CANDIDATE-PORTS W2b. See .agents/specs/new-candidate-ports-olala-w2.md
// (architecture map) and the W1 census
// (docs/bench-evidence/new-candidate-ports-olala-20261120/olala-namemap.md).
//
// The reference is the model author's own code: OVHaiLLM/Olala-v1.0-mini-7A1B
// @ f781f12498adc9bdd71b7dd59d1311434369c684, modeling_olala.py
// (oracle id `olala-transformers`). GGUF names come from the W1 walk; the
// mapping below cites the reference class each tensor feeds.
#pragma once

#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace vllm {

// Geometry the forward consumes, read from the parsed HfConfig (the GGUF
// builder maps the standard fields; the rest ride in config.raw under
// gguf_unmapped with their GGUF spelling).
struct OlalaParams {
  int64_t hidden = 0;          // olala.embedding_length
  int64_t layers = 0;          // olala.block_count
  int64_t vocab = 0;           // token_embd rows
  int64_t heads = 0;           // olala.attention.head_count
  int64_t kv_heads = 0;        // olala.attention.head_count_kv
  int64_t head_dim = 0;        // olala.attention.key_length
  int64_t window = 0;          // olala.attention.sliding_window (slw_wsize)
  double softcap = 0.0;        // olala.attn_logit_softcapping
  int64_t experts = 0;         // olala.expert_count
  int64_t top_k = 0;           // olala.expert_used_count
  int64_t latent = 0;          // olala.moe_latent_size
  int64_t expert_ff = 0;       // olala.expert_feed_forward_length
  int64_t shared_ff = 0;       // olala.expert_shared_feed_forward_length
  double routed_scale = 0.0;   // olala.expert_weights_scale
  // M-layer (Mamba3-MIMO) geometry:
  int64_t ssm_inner = 0;       // olala.ssm.inner_size (d_inner = 2*hidden on the artifact)
  int64_t d_state = 0;         // olala.ssm.state_size
  int64_t mimo_dim = 0;        // olala.mamba_mimo_dim
  int64_t headdim = 0;         // olala.mamba_headdim
  int64_t tpa_rank = 0;        // olala.tpa_rank
  int64_t rope_angles = 0;     // olala.num_rope_angles
  // elementwise mixer gate bias: silu(attn_gate @ x + gate_bias) scales the V
  // mixer output before mixer_proj (modeling_olala.py:1777-1786, use_gate /
  // gate_type "elementwise" / gate_bias = olala.gate_bias; fork
  // olala.cpp:3795-3798 ggml_scale_bias(..., 1.0, gate_b) then ggml_silu)
  double gate_bias = 0.0;      // olala.gate_bias
  // layer schedule: 'M'/'V' per layer, from olala.layer_types (carried verbatim
  // under config.raw["gguf_unmapped"]["olala.layer_types"]); the W1 tensor walk
  // (attn_q presence) agrees on the pinned artifact.
  std::string layer_types;
};

OlalaParams ParseOlalaParams(const HfConfig& config);

// Per-layer weights. Every tensor the W1 census maps lands here; the layer
// kind comes from layer_types (V = differential attention, M = Mamba3-MIMO).
struct OlalaLayerWeights {
  // common (all 36 layers):
  OwnedTensor attn_output;    // blk.N.attn_output.weight  (mixer_proj, modeling_olala.py:1679)
  OwnedTensor router_bias;    // blk.N.exp_probs_b         (expert_bias, :1503) F32
  OwnedTensor router_weight;  // blk.N.ffn_gate_inp.weight (moe_gate, :1478) F32
  OwnedTensor ffn_up_exps;    // blk.N.ffn_up_exps.weight  [256,768,384] experts up/gate
  OwnedTensor ffn_down_exps;  // blk.N.ffn_down_exps.weight [256,384,768] experts down
  OwnedTensor ffn_up_shexp;   // blk.N.ffn_up_shexp.weight  [1536,1536] shared MLP
  OwnedTensor ffn_down_shexp; // blk.N.ffn_down_shexp.weight
  OwnedTensor ffn_latent_up;  // blk.N.ffn_latent_up.weight  [1536,384] MoE up_proj (:1480)
  OwnedTensor ffn_latent_down;// blk.N.ffn_latent_down.weight [384,1536] MoE down_proj
  OwnedTensor geo_mixer_bias, geo_mixer_scale;  // [1] F32 geodesic_mixer (:1689)
  OwnedTensor geo_mlp_bias, geo_mlp_scale;      // [1] F32 geodesic_mlp (:1690)
  // M-only (29 layers, OlalaMamba3MimoFast :933):
  OwnedTensor ssm_in, ssm_in_dyn;              // [6288,1536] / [1056,1536] in_proj/in_proj_dyn (:987,:993)
  OwnedTensor ssm_b_bias, ssm_b_norm;          // [48,4,128] / [128]
  OwnedTensor ssm_c_bias, ssm_c_norm;
  OwnedTensor ssm_d, ssm_dt_bias;              // [48]
  OwnedTensor ssm_m3_bias, ssm_m3_misc, ssm_m3_mxz, ssm_m3_norms;
  OwnedTensor ssm_mimo_x, ssm_mimo_z, ssm_mimo_o;  // [48,4,64]
  // V-only (7 layers, OlalaDifferentialTensorProductAttentionV2 :498):
  OwnedTensor attn_q;          // [6144,1536]
  OwnedTensor attn_gate;       // [4608,1536] elementwise silu gate (:1653)
  OwnedTensor attn_q_norm, attn_k_norm;       // [128] F32
  OwnedTensor attn_lambda, attn_shift_k, attn_shift_v;  // [12,1536]
  OwnedTensor attn_softmax_scaler;            // [48] F32
  OwnedTensor attn_wa_k, attn_wa_v, attn_wb_k, attn_wb_v;  // [48,1536] / [512,1536]
  bool is_m = false;
};

struct OlalaWeights {
  OwnedTensor embed;  // token_embd.weight [vocab, hidden]
  OwnedTensor head;   // output.weight     [vocab, hidden] (untied)
  std::vector<OlalaLayerWeights> layers;
};

// W2b: loads the bf16/F32 arm of a census-validated olala GGUF. Block-quant
// tensors (the q6_k/q5_k_m/q8_0-safe2 arms) refuse BY NAME here — the
// keep-quant decode arm is W2c work. Throws on any census-unmapped tensor.
OlalaWeights LoadOlalaGgufWeights(const GgufFile& gguf, const HfConfig& config);

// The forward boundary text for W2b: weights load, the forward refuses naming
// each unported op. W2c replaces this with the real forward.
std::string OlalaForwardRefusalOps();

}  // namespace vllm
