// Gemma-4 USM-Conformer AUDIO tower (`Gemma4AudioModel`) + the audio
// `Gemma4MultimodalEmbedder` (`embed_audio`) — MODEL-GEMMA4 G3. A standalone
// additive, host-side FLOAT32 forward that turns the log-mel `input_features`
// [T, feature_size] (A1's product; a dumped golden here) into the projected audio
// soft tokens [S, text_hidden] that scatter into the text embedding stream.
//
// Ported 1:1 from transformers/models/gemma4/modeling_gemma4.py @ 5.13.1
// (vLLM loads this tower via AutoModel.from_config and runs it eager —
// gemma4_mm.py:1063,1481):
//   Gemma4AudioModel.forward (:1985-2014) — subsample -> rel_pos_enc -> N layers
//     -> output_proj(Linear 1024->1536, bias=True);
//   Gemma4AudioSubSampleConvProjection (:385-412) — mask-zero -> Conv2d(1->128,
//     k3,s2,p1) -> LN(128,bias=False) -> ReLU -> mask[::2] -> Conv2d(128->32,
//     k3,s2,p1) -> LN(32) -> ReLU -> mask[::2] -> reshape[S, (128/4)*32=1024] ->
//     input_proj_linear(1024->1024);
//   Gemma4AudioRelPositionalEncoding (:218-246) — inv_timescales(512),
//     position_ids=arange(ctx//2,-1,-1) [13], pos=[sin|cos] -> [13,1024];
//   Gemma4AudioLayer (:525-573) — ff1(half-step 0.5) -> clamp+norm_pre_attn ->
//     self_attn -> clamp+norm_post_attn+res -> lconv1d -> ff2 -> clamp+norm_out;
//   Gemma4AudioAttention (:249-354) — CHUNKED LOCAL attention: q_scale=
//     (hd^-0.5)/ln2 * softplus(per_dim_scale), k_scale=ln(1+e)/ln2, blocks of
//     chunk=12 with a context window [past=ctx_left-1=12, future=ctx_right=0]
//     (context_size=24), Transformer-XL relative bias (relative_k_proj over the
//     13 rel-pos embeddings + _rel_shift), attn=tanh(aw/softcap)*softcap (50),
//     sliding-window+padding mask -> softmax(f32) -> post proj;
//   Gemma4AudioLightConv1d (:484-522) — pre_norm -> linear_start(1024->2048) ->
//     GLU -> depthwise CAUSAL Conv1d(k=5, left_pad=4) -> conv_norm -> silu ->
//     linear_end -> +res;
//   Gemma4AudioFeedForward (:415-447) — pre_norm -> ffw1(1024->4096) -> silu ->
//     ffw2(4096->1024) -> post_norm -> *0.5 -> +res;
//   Gemma4RMSNorm (:197-215) — x.float(), mean(x^2)+eps, *pow(-0.5), *weight;
//   Gemma4ClippableLinear (:168-194, use_clipped_linears=True) — clamp(x,in) ->
//     Linear(bias=False) -> clamp(out), FINITE trained per-linear scalar bounds;
//   Gemma4MultimodalEmbedder (gemma4_mm.py:908-960) — RMSNorm(no-weight) ->
//     Linear(1536->text_hidden, bias=False).
//
// NUMERIC CONTRACT: host FLOAT32 throughout — the reference tower is dumped in
// f32 (scripts/mm/g3_audio_tower_ref.py) so the USM-Conformer MATH (chunked attn +
// rel-shift + GLU conv + softcap + clamps + half-step residuals) is the sole
// parity variable, gated f32-vs-f32 per-stage (RED-first). Production dtype is
// bf16; a device-resident bf16 forward + the audio->text e2e merge are the named
// residuals (mirroring the G2-impl vision tower-in-isolation cadence).
#pragma once

#include <cstdint>
#include <vector>
#include <limits>
#include <string>

namespace vllm::multimodal {

struct Gemma4AudioConfig {
  bool use_clipped_linears = true;
  std::string hidden_act = "silu";
  int64_t hidden_size = 1024;
  int64_t num_layers = 12;
  int64_t num_heads = 8;
  int64_t head_dim = 128;          // hidden/num_heads
  int64_t output_proj_dims = 1536;
  int64_t conv_kernel_size = 5;    // depthwise light-conv kernel
  int64_t sub_ch0 = 128;           // subsampling_conv_channels[0]
  int64_t sub_ch1 = 32;            // subsampling_conv_channels[1]
  int64_t feature_size = 128;      // mel feature dim (Conv2d spatial width)
  int64_t chunk_size = 12;         // attention_chunk_size
  int64_t context_left = 13;       // attention_context_left
  int64_t context_right = 0;       // attention_context_right
  int64_t text_hidden_size = 2560; // text_config.hidden_size
  float attention_logit_cap = 50.0f;
  float attention_invalid_logits_value = -1e9f;
  double gradient_clipping = 1e10;
  float residual_weight = 0.5f;    // FFN half-step post scale
  float rms_norm_eps = 1e-6f;

  int64_t max_past_horizon() const { return context_left - 1; }          // 12
  int64_t max_future_horizon() const { return context_right; }           // 0
  int64_t context_size() const { return chunk_size + max_past_horizon() + max_future_horizon(); }  // 24
  int64_t ffn_dim() const { return hidden_size * 4; }                    // 4096
};

// Gemma4ClippableLinear QAT clamp; disabled clipping has true infinite bounds.
struct AClip {
  float in_min = -std::numeric_limits<float>::infinity(), in_max = std::numeric_limits<float>::infinity();
  float out_min = -std::numeric_limits<float>::infinity(), out_max = std::numeric_limits<float>::infinity();
};

// A clippable Linear: weight [out,in] row-major f32 + the activation clamp.
struct ClipLinear {
  std::vector<float> w;  // [out, in]
  AClip clip;
};

struct Gemma4AudioFFWeights {
  ClipLinear ffw1, ffw2;                        // 1024->4096, 4096->1024
  std::vector<float> pre_ln, post_ln;           // RMSNorm weights [H]
};

struct Gemma4AudioLConvWeights {
  ClipLinear linear_start, linear_end;          // 1024->2048, 1024->1024
  std::vector<float> depthwise;                 // [H,1,K]
  std::vector<float> pre_ln, conv_norm;         // RMSNorm [H]
};

struct Gemma4AudioAttnWeights {
  ClipLinear q_proj, k_proj, v_proj, post;      // all 1024->1024
  std::vector<float> relative_k_proj;           // [H,H] (bias-free)
  std::vector<float> per_dim_scale;             // [head_dim]
};

struct Gemma4AudioLayerWeights {
  Gemma4AudioFFWeights ff1, ff2;
  Gemma4AudioAttnWeights attn;
  Gemma4AudioLConvWeights lconv;
  std::vector<float> norm_pre_attn, norm_post_attn, norm_out;  // RMSNorm [H]
};

struct Gemma4AudioWeights {
  // subsample
  std::vector<float> sub0_conv;   // [128,1,3,3]
  std::vector<float> sub0_norm;   // [128]  (LayerNorm weight, no bias)
  std::vector<float> sub1_conv;   // [32,128,3,3]
  std::vector<float> sub1_norm;   // [32]
  std::vector<float> input_proj;  // [1024,1024]
  std::vector<Gemma4AudioLayerWeights> layers;
  std::vector<float> output_proj_w;  // [1536,1024]
  std::vector<float> output_proj_b;  // [1536]
  std::vector<float> embed_proj;     // [text_hidden,1536] (embed_audio, bias-free)
};

// Per-stage captures for the G3 unit gates (all host f32).
struct Gemma4AudioCapture {
  std::vector<float> subsample_out;       // [S, H]
  std::vector<float> position_embeddings; // [P, H]  (P = ctx//2+1)
  std::vector<float> block0;              // [S, H]
  std::vector<float> block_mid;           // [S, H]  (layer N/2)
  std::vector<float> block_last;          // [S, H]  (layer N-1)
  std::vector<float> output_proj;         // [S, output_proj_dims]
};

// Optional embed_projection borrows the separately loaded connector without
// copying the full tower; null uses w.embed_proj for existing callers.
// Runs the USM-Conformer audio tower + audio projector on one clip.
//   input_features : [T, feature_size] host f32 (log-mel; A1's product)
//   feature_mask   : [T] host i32 (1=valid, 0=padding) — may be empty (all valid)
// Returns the projected audio soft tokens [S, text_hidden] host f32 (== the
// masked-scatter merge input). If capture != nullptr the stage tensors are filled.
std::vector<float> Gemma4AudioForward(const std::vector<float>& input_features, int64_t n_frames,
                                      const std::vector<int32_t>& feature_mask,
                                      const Gemma4AudioWeights& w, const Gemma4AudioConfig& cfg,
                                      Gemma4AudioCapture* capture = nullptr,
                                      const std::vector<float>* embed_projection = nullptr);

}  // namespace vllm::multimodal
