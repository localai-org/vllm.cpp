// vllm.cpp ORIGINAL — the `olala` host ops, row NEW-CANDIDATE-PORTS W2c-1.
// Reference: modeling_olala.py @ f781f124 (OVHaiLLM/Olala-v1.0-mini-7A1B,
// oracle olala-transformers) — each function cites the reference lines.
#pragma once

#include <cstdint>

#include "vllm/model_executor/models/olala.h"

namespace vllm {

// OlalaGeodesicNorm.forward (modeling_olala.py:1596-1617), host f32.
// x, g: [rows, cols] row-major; scale/bias: the geo_* [1] scalars; layer_idx
// is the 0-based block index (the reference's 1-based layer_idx+1 divisor).
void OlalaGeodesicNormHost(float* out, const float* x, const float* g,
                           int64_t rows, int64_t cols, float scale,
                           float bias, int64_t layer_idx);

// OlalaMoE.forward (modeling_olala.py:1512-1552) + TorchMoEMLP
// (:1448-1469) + OlalaMLP (:1414-1418), host f32 compute over the W2b-loaded
// weights. x: [tokens, hidden] f32; out: [tokens, hidden] f32.
// Router: logits = x @ ffn_gate_inp^T + exp_probs_b (f32), sigmoid, top-k,
// probs normalized * expert_weights_scale. Routed: latent down [latent,hidden],
// per-expert h = relu(xl @ up_e)^2, y = h @ down_e, fp32 weighted accumulate,
// latent up [hidden, latent]. Shared: relu(ffn_up_shexp @ x)^2 @ ffn_down_shexp.
// out = shared + routed.
void OlalaMoeHost(float* out, const float* x, int64_t tokens, int64_t hidden,
                  const OlalaLayerWeights& lw, const OlalaParams& p);

}  // namespace vllm

namespace vllm {

// W2c-2: the V-layer mixer core, OlalaDifferentialTensorProductAttentionV2
// (modeling_olala.py:498-745), host f32, pre-mixer_proj. x: [len, hidden];
// positions: [len] (0-based, one document); out: [len, signal_heads*head_dim].
// NoPE on this artifact (rope_theta=0); qk-norm is plain RMSNorm
// (zero_centered_gamma weight is zeros); scalable softmax scales q by
// scaler[h] * log(min(pos+1, window)); softmax scale = 1/head_dim
// (use_completed_p=True, :727); softcap tanh (:469-470); causal + left window
// wsize (:472-491); GQA repeat_interleave (:464-465); differential merge with
// sigmoid(lambda) (:734-740).
void OlalaVAttentionHost(float* out, const float* x, const int64_t* positions,
                         int64_t len, const OlalaLayerWeights& lw,
                         const OlalaParams& p);

}  // namespace vllm

namespace vllm {

// W2c-3: one Mamba3-MIMO decode token step (modeling_olala.py:1265-1370 step +
// _mamba3_rotary_step_torch :861-868 + _mamba3_step_torch :871-889), host f32.
// State layout (allocate_inference_cache :1372-1404): angle [H, A] f32,
// ssm [H, P, N] f32, k_state [R, H, N], v_state [H, P]. u: [hidden]; out: [H*P]
// pre-mixer_proj. Prefill = stepping per token (the chunked kernel matches the
// step to bf16 noise, :1148).
struct OlalaMambaState {
  std::vector<float> angle;   // [H * A]
  std::vector<float> ssm;     // [H * P * N]
  std::vector<float> k_state; // [R * H * N]
  std::vector<float> v_state; // [H * P]
};

void OlalaMambaStepHost(float* out, OlalaMambaState& st, const float* u,
                        const OlalaLayerWeights& lw, const OlalaParams& p);

}  // namespace vllm
