// Gemma-4 NaFlex SigLIP2 vision tower (`Gemma4VisionModel` + the vision
// `Gemma4MultimodalEmbedder`) — MODEL-GEMMA4 G2-impl. A standalone additive
// forward that turns the NaFlex image-processor output (pre-patchified
// `pixel_values [P, 3*16^2]` + per-patch `(x,y)` position ids) into the 256
// projected soft tokens that scatter into the text embedding stream.
//
// Ported 1:1 from transformers/models/gemma4/modeling_gemma4.py (the tower vLLM
// runs in eager per gemma4_mm.py's docstring):
//   Gemma4VisionPatchEmbedder (:575) — input_proj Linear(768->768,bias=False),
//     pixel scale 2*(x-0.5) (:612), learned 2-D pos-embed lookup
//     position_embedding_table[2,10240,768] x_emb+y_emb (:602-604), padding
//     zeroed (:605);
//   Gemma4VisionRotaryEmbedding (:701) — theta 100, spatial_dim=head_dim//2=32,
//     inv_freq over arange(0,32,2)/32 (:749), per-axis cos|sin (:757-778);
//   apply_multidimensional_rope (:855) — head_dim 64 split into ndim=2 parts of
//     32 ch each, each a NeoX rope over its axis' positions;
//   Gemma4VisionAttention (:911) — scaling=1.0 (:921, NOT 1/sqrt(d)), q/k RMSNorm
//     + weight-less v RMSNorm (:929-931), MHA (kv_heads==heads==12), o_proj;
//   Gemma4VisionEncoderLayer (:980) — Gemma-2 SANDWICH norms
//     input_ln->attn->post_attn_ln->+res ; pre_ff_ln->mlp->post_ff_ln->+res;
//   Gemma4VisionMLP (:685) — GeGLU gate/up/down, gelu_pytorch_tanh;
//   Gemma4VisionPooler (:618) — avg-pool-by-position over a k^2 grid via one_hot
//     weights (:631-656), * sqrt(hidden) in fp32 (:681), padding strip (:2071);
//   Gemma4MultimodalEmbedder (:2078) — RMSNorm(with_scale=False) ->
//     Linear(768->2560,bias=False).
//
// Numeric contract mirrors the M2a Qwen3-VL tower: production dtype bf16 (all
// GEMMs/attn/norm-affine bf16, softmax + RMS accumulation f32; the pooler's
// sqrt(hidden) scale is fp32 per upstream because it can exceed the fp16 range).
// The 2-D pos-embed lookup and the rope cos|sin cache are deterministic host
// precomputes (f32) consumed on device.
//
// KEY LAYOUT FACT (measured on the committed E4B image ref): the NaFlex
// processor emits the VALID patches CONTIGUOUS at the front (padding (-1,-1) is a
// trailing block), so the bidirectional encoder — whose mask excludes padding
// keys — is exactly a full non-causal attention over the leading `n_valid`
// patches. The tower therefore runs only the valid prefix (padding contributes
// nothing to any valid soft token: its pos-embed is zeroed and the pooler
// masked_fill-s it to 0), and the four stage captures are compared on the valid
// rows.
#pragma once

#include <array>
#include <cstdint>
#include <limits>
#include <vector>

#include "vt/backend.h"

namespace vllm::multimodal {

struct Gemma4VisionConfig {
  int64_t hidden_size = 768;
  int64_t num_heads = 12;
  int64_t num_kv_heads = 12;  // MHA (kv_groups==1)
  int64_t depth = 16;
  int64_t intermediate_size = 3072;
  int64_t head_dim = 64;
  int64_t patch_size = 16;
  int64_t pooling_kernel_size = 3;
  int64_t position_embedding_size = 10240;
  int64_t text_hidden_size = 2560;  // E4B text_config.hidden_size
  float rms_norm_eps = 1e-6f;
  double rope_theta = 100.0;
  bool use_clipped_linears = false;
  // Loader accepts only the existing tanh-GELU, unbiased, unstandardized path.
  float attention_dropout = 0.0f;  // Retained; inference does not apply dropout.

  int64_t patch_dim() const { return 3 * patch_size * patch_size; }  // 768
};

// Gemma4ClippableLinear QAT activation clamp (use_clipped_linears==True for E4B):
// out = clamp(linear(clamp(x, in_min, in_max)), out_min, out_max). The bounds
// are per-linear scalars stored in the checkpoint. patch_embedder.input_proj and
// embed_vision.embedding_projection are PLAIN nn.Linear (no clip).
struct Clip {
  float in_min = -std::numeric_limits<float>::infinity(),
        in_max = std::numeric_limits<float>::infinity();    // +/-inf default == no-op
  float out_min = -std::numeric_limits<float>::infinity(),
        out_max = std::numeric_limits<float>::infinity();
};

// All weights host-side row-major RAW BF16 BITS (`uint16_t`, as torch stores:
// a Linear weight is [out,in]). bf16 because `Gemma4VisionForward` puts every
// one of them through `MakeDevBf16` before its first GEMM, so an f32 store would
// cost 2x the checkpoint's bytes and buy nothing (#1359, and see
// `qwen3_vl_vision.h` for the mirror argument and the bit-identity proof).
struct Gemma4VisionBlockWeights {
  std::vector<uint16_t> input_ln, post_attn_ln, pre_ff_ln, post_ff_ln;  // [H]
  std::vector<uint16_t> q_proj, k_proj, v_proj, o_proj;  // Q [nh*hd,H], K/V [nkv*hd,H], O [H,nh*hd]
  std::vector<uint16_t> q_norm, k_norm;                  // [head_dim]
  std::vector<uint16_t> gate_proj, up_proj;              // [I,H]
  std::vector<uint16_t> down_proj;                       // [H,I]
  // Independent QAT bounds per source ClippableLinear; equality is not required.
  Clip q_clip, k_clip, v_clip, o_clip, gate_clip, up_clip, down_clip;
};

struct Gemma4VisionWeights {
  std::vector<uint16_t> input_proj;              // [H, patch_dim]  (768x768)
  // f32, DELIBERATELY, and the only such field here — the same exception
  // `Qwen3VLVisionWeights::pos_embed_w` carries. `Gemma4VisionForward` SUMS the
  // x and y rows of this table on the host (`gemma4_vision.cpp:199-210`) and
  // narrows only the sum, so the stored values reach arithmetic and narrowing
  // the store would move the result. Reconciling it onto the mirror is owed
  // with the Qwen3-VL one (`.agents/specs/vision-tower-dtype-polarity.md` §4.3).
  std::vector<float> position_embedding_table;   // [2*pos_embed_size*H]
  std::vector<Gemma4VisionBlockWeights> blocks;  // depth
  std::vector<uint16_t> embed_projection;        // [text_hidden, H]  (2560x768)
};

// Per-stage captures for the G2-impl unit gates (all host f32, valid rows only).
struct Gemma4VisionCapture {
  std::vector<float> patch_embed_out;  // [n_valid, H]
  std::vector<float> attention_out;  // last block, pre-o-projection [n_valid,H]
  std::vector<float> mlp_out;        // last block, post-norm [n_valid,H]
  std::vector<float> encoder_out;      // [n_valid, H]
  std::vector<float> pooled;           // [n_soft, H]  (post sqrt(hidden) fp32, bf16-rounded)
  std::vector<float> projected;        // [n_soft, text_hidden]
};

// Validate the actual tower input boundary before backend allocation; returns
// the nonempty contiguous valid prefix. Positions encode the padding mask.
int64_t ValidateGemma4VisionInput(const std::vector<float>& pixels,
                                 const std::vector<int64_t>& positions,
                                 int64_t n_total, const Gemma4VisionConfig& cfg);

// Runs the tower on one image. pixel_values is [n_total, patch_dim] host f32 in
// [0,1]; position_ids is [n_total, 2] host i64 ((x,y); (-1,-1)==padding, trailing
// contiguous). Returns the projected soft tokens [n_soft, text_hidden] host f32
// (== embed_vision output == the masked-scatter merge input). If capture !=
// nullptr the four stage tensors are filled.
std::vector<float> Gemma4VisionForward(const std::vector<float>& pixel_values,
                                       const std::vector<int64_t>& position_ids,
                                       int64_t n_total, const Gemma4VisionWeights& w,
                                       const Gemma4VisionConfig& cfg, vt::Backend& backend,
                                       Gemma4VisionCapture* capture = nullptr);

}  // namespace vllm::multimodal
