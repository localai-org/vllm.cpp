// Qwen3-VL vision tower (`Qwen3_VisionTransformer`) — M2a standalone forward.
//
// Ported from: vllm/model_executor/models/qwen3_vl.py @ e24d1b24
//   Qwen3_VisionPatchEmbed (:347), Qwen3_VisionBlock (:413),
//   Qwen3_VisionPatchMerger (:467), Qwen3_VisionTransformer (:519),
//   forward (:800), pos_embed_interpolate_native (:277), rot_pos_emb (:667),
//   vision attention Qwen2_5_VisionAttention (qwen2_5_vl.py:345),
//   ApplyRotaryEmb.forward_static (rotary_embedding/common.py:125).
//
// This is the M2a increment: the vision tower proven faithful to vLLM in
// isolation (image token-exact e2e is M2c, after the M2b MRoPE/DeepStack text
// backbone). Pure additive TU — it does NOT touch the shared model runner /
// registry, so the text SACRED gates are byte-identical by construction.
//
// Numeric contract: production model dtype is bf16 (patch-embed/attn/mlp/merger
// GEMMs bf16, softmax/norm accumulation f32). The pos-embed bilinear interp and
// the vision-rope cos|sin are computed HOST-side in f32 then consumed on device
// (deterministic precompute; vLLM computes them on GPU — gated within a stated
// bf16 tolerance). Vision RoPE is a full-64-dim NeoX rope with per-token cos|sin
// [L,32] (partial_rotary_factor 0.5 builds the 32-wide table from a 16-freq
// cache over the 2 spatial axes); it reuses vt::RopeFromCache with rotary_dim=64
// and a [L,64]=[cos|sin] cache.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <map>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/tensor.h"

namespace vllm::multimodal {

struct Qwen3VLVisionConfig {
  int64_t hidden_size = 1024;
  int64_t num_heads = 16;
  int64_t depth = 24;
  int64_t intermediate_size = 4096;
  int64_t out_hidden_size = 2560;
  int64_t patch_size = 16;
  int64_t temporal_patch_size = 2;
  int64_t spatial_merge_size = 2;
  int64_t num_position_embeddings = 2304;
  int64_t in_channels = 3;
  std::vector<int> deepstack_visual_indexes = {5, 11, 17};
  float norm_eps = 1e-6f;

  int64_t head_dim() const { return hidden_size / num_heads; }
  int64_t merge_unit() const { return spatial_merge_size * spatial_merge_size; }
  int64_t num_grid_per_side() const {
    // isqrt(num_position_embeddings)
    int64_t r = 0;
    while ((r + 1) * (r + 1) <= num_position_embeddings) ++r;
    return r;
  }
};

// All weights are host-side row-major RAW BF16 BITS (`uint16_t`, as stored by
// torch: a Linear weight is [out, in]). LayerNorm weight/bias are [dim].
//
// BF16, NOT f32, and that is the model dtype rather than a compression (#1359).
// Upstream has no ViT dtype of its own: `Qwen3_VisionTransformer.dtype` IS
// `patch_embed.proj.weight.dtype` (qwen3_vl.py:633-634) under
// `set_default_torch_dtype(model_config.dtype)` (base_loader.py:53), so the
// tower is whatever the all-BF16 checkpoint loaded as. These stores held f32
// until #1359 and every consumer narrowed them straight back with
// `vt::F32ToBF16` before its first GEMM, so the widening cost exactly 2x the
// checkpoint's bytes and bought nothing. Narrowing is BIT-IDENTICAL, not merely
// within tolerance: `BF16ToF32` is a 16-bit shift, so `F32ToBF16`'s
// round-to-nearest-even addend cannot carry back into bit 16
// (`src/vt/dtype.cpp:317-326`; proven exhaustively in
// `tests/vllm/models/test_vision_tower_dtype.cpp`).
struct VisionBlockWeights {
  std::vector<uint16_t> norm1_w, norm1_b;   // [hidden]
  std::vector<uint16_t> norm2_w, norm2_b;   // [hidden]
  std::vector<uint16_t> qkv_w, qkv_b;       // qkv_w [3*hidden, hidden], qkv_b [3*hidden]
  std::vector<uint16_t> proj_w, proj_b;     // [hidden, hidden], [hidden]
  std::vector<uint16_t> fc1_w, fc1_b;       // [inter, hidden], [inter]
  std::vector<uint16_t> fc2_w, fc2_b;       // [hidden, inter], [hidden]
};

struct VisionMergerWeights {
  bool use_postshuffle_norm = false;        // main merger false; deepstack true
  std::vector<uint16_t> norm_w, norm_b;     // [context_dim] (false) or [4*context] (true)
  std::vector<uint16_t> fc1_w, fc1_b;       // [4*context, 4*context]
  std::vector<uint16_t> fc2_w, fc2_b;       // [out_hidden, 4*context], [out_hidden]
};

struct Qwen3VLVisionWeights {
  std::vector<uint16_t> patch_proj_w, patch_proj_b;  // [hidden, C*tp*p*p], [hidden]
  // f32, DELIBERATELY, and the only such field here: the pos-embed table is the
  // one weight whose values reach arithmetic before anything narrows them —
  // `VisionPosEmbedInterpolate` runs the bilinear gather and sum on the host in
  // f32 (mirroring pos_embed_interpolate_native, qwen3_vl.py:277-344) — so
  // narrowing the store would move the interpolated table. 9,437,184 B on
  // Qwen3-VL-4B, 0.57% of the tower. This is WIDER than upstream, which casts
  // the coefficients down to the model dtype at :335 and gathers a model-dtype
  // embedding at :337; reconciling it onto the mirror moves tower numbers and is
  // owed separately (`.agents/specs/vision-tower-dtype-polarity.md` §4.3).
  std::vector<float> pos_embed_w;                    // [num_position_embeddings, hidden]
  std::vector<VisionBlockWeights> blocks;            // depth
  VisionMergerWeights merger;
  std::vector<VisionMergerWeights> deepstack_mergers;  // len(deepstack_visual_indexes)
};

// Optional intermediate capture for the M2a unit gates (all host f32). When a
// pointer to this is passed, the forward downloads the gated stages; production
// (M2c) passes nullptr and pays nothing.
struct Qwen3VLVisionCapture {
  std::vector<float> pos_embeds;                  // [L, hidden]
  std::vector<float> rotary_cos, rotary_sin;      // [L, head_dim/2]
  std::vector<float> patch_embed_out;             // [L, hidden]
  std::vector<float> block0_out;                  // [L, hidden]
  std::vector<float> merger_out;                  // [Nmerge, out_hidden]
  std::vector<std::vector<float>> deepstack_out;  // 3 x [Nmerge, out_hidden]
  // Optional typed-route attribution boundaries, empty in the legacy helper.
  std::vector<float> merger_input, merger_norm_out, merger_fc1_out, merger_gelu_out;
  std::vector<std::vector<float>> block_outputs;  // typed-route attribution only
  // Empty preserves historical all-block output capture. Nonempty captures
  // only these output slots and their suboperations; production passes nullptr.
  std::vector<int64_t> selected_blocks;
  std::map<int64_t, std::map<std::string, std::vector<float>>> block_boundaries;
};

// Runs the tower on one image. pixel_values is [L, C*tp*p*p] host bf16 bits
// (uint16 raw bf16), grid_thw = [t,h,w]. Returns the full tower output
// [Nmerge, out_hidden*(1+num_deepstack)] as host f32. If capture != nullptr, the
// gated intermediate stages are filled.
std::vector<float> Qwen3VLVisionForward(const std::vector<uint16_t>& pixel_values_bf16,
                                        const std::array<int64_t, 3>& grid_thw,
                                        const Qwen3VLVisionWeights& w,
                                        const Qwen3VLVisionConfig& cfg, vt::Backend& backend,
                                        Qwen3VLVisionCapture* capture = nullptr);

// --- Device-resident tower weights (the production/fast path) -----------------
// The M2a host-weights forward above converted every weight host f32->bf16 and
// re-uploaded ~0.5 GiB per call, INSIDE the timed forward — that host marshalling
// (not the ViT kernels) dominated the ~2.1 s vs vLLM's ~0.25 s encode (which runs
// on weights already resident on GPU, mirror `Qwen3_VisionTransformer` whose
// nn.Linears are loaded once). PrepareVisionDeviceWeights does that conversion +
// upload ONCE; the resident-weights forward then runs pure GEMMs/attention with
// only the tiny per-image (pixel / pos-embed / rope) uploads. Numerically
// BIT-IDENTICAL to the host overload (same bf16 weight bytes, same GEMM order).
struct Qwen3VLVisionDeviceWeights;

// The default preserves the existing BF16 tower. Explicit XPU FP16 preparation
// numerically converts checkpoint BF16 values and uploads a typed position
// table as well; it is not accepted by the legacy BF16-input/host-output forward
// below. Backend ownership must outlive the prepared weights.
std::shared_ptr<Qwen3VLVisionDeviceWeights> PrepareVisionDeviceWeights(
    const Qwen3VLVisionWeights& host_w, const Qwen3VLVisionConfig& cfg, vt::Backend& backend,
    vt::DType execution_dtype = vt::DType::kBF16);

// Fixed-grid, queue-bound FP16 XPU scratch, reused across images and all blocks.
// The backend must outlive the workspace and every output. Destroying scratch
// waits its latest encode event; ordinary forward submission is asynchronous.
struct Qwen3VLVisionWorkspace;
std::shared_ptr<Qwen3VLVisionWorkspace> PrepareVisionWorkspace(
    const std::array<int64_t,3>& grid_thw, const Qwen3VLVisionConfig& cfg,
    vt::Backend& backend, vt::Queue& queue);

// Owned [visual_tokens,out_hidden_size] FP16 device output. Before consumption
// on another queue call WaitOn; after submitting each consumer call RecordUse.
// The shared owner retains weights/output until encode and all consumers finish.
// It does not retain scratch; independent outputs survive workspace reuse.
class Qwen3VLVisionDeviceOutput {
 public:
  struct State;
  explicit Qwen3VLVisionDeviceOutput(std::shared_ptr<State> state);
  const vt::Tensor& tensor() const;
  void WaitOn(vt::Queue& queue) const;
  void RecordUse(vt::Queue& queue) const;
 private:
  std::shared_ptr<State> state_;
};

// Native image-only route (t=1), without DeepStack. Input is a contiguous FP16
// device tensor, retained by the caller through encode completion. Optional
// captures download intermediates for tests; nullptr performs no readbacks.
Qwen3VLVisionDeviceOutput Qwen3VLVisionForwardDevice(
    const vt::Tensor& pixel_values,
    std::shared_ptr<const Qwen3VLVisionDeviceWeights> weights,
    Qwen3VLVisionWorkspace& workspace, vt::Backend& backend, vt::Queue& queue,
    Qwen3VLVisionCapture* capture = nullptr);

// Synchronous diagnostic only: run one existing typed block on an exact device
// input, using the production block implementation and workspace RoPE. Does
// not accept test operands in serving or replace any production activation.
// Optional reference outputs replace scratch AFTER recording each native
// stage, isolating the next operator from upstream drift. Caller owns tensors.
void Qwen3VLVisionReplayBlockDevice(
    const vt::Tensor& block_input,
    std::shared_ptr<const Qwen3VLVisionDeviceWeights> weights,
    Qwen3VLVisionWorkspace& workspace, vt::Backend& backend, vt::Queue& queue,
    int64_t block, Qwen3VLVisionCapture& capture,
    const std::map<std::string, vt::Tensor>* reference_outputs = nullptr);

// Synchronous diagnostic of the shared typed merger on an exact tower output.
// Optional norm/fc1/gelu references replace scratch AFTER native capture to
// isolate the following operator. Normal inference cannot supply references.
void Qwen3VLVisionReplayMergerDevice(
    const vt::Tensor& merger_input,
    std::shared_ptr<const Qwen3VLVisionDeviceWeights> weights,
    Qwen3VLVisionWorkspace& workspace, vt::Backend& backend, vt::Queue& queue,
    Qwen3VLVisionCapture& capture,
    const std::map<std::string, vt::Tensor>* reference_outputs = nullptr);

std::vector<float> Qwen3VLVisionForward(const std::vector<uint16_t>& pixel_values_bf16,
                                        const std::array<int64_t, 3>& grid_thw,
                                        const Qwen3VLVisionDeviceWeights& dw,
                                        const Qwen3VLVisionConfig& cfg, vt::Backend& backend,
                                        Qwen3VLVisionCapture* capture = nullptr);

// Host precompute helpers (exposed for direct unit-gating).
// pos_embed bilinear interpolation + spatial-merge reorder → [L, hidden] f32.
std::vector<float> VisionPosEmbedInterpolate(const std::vector<float>& pos_embed_w,
                                             const std::array<int64_t, 3>& grid_thw,
                                             const Qwen3VLVisionConfig& cfg);
// vision-rope cos|sin tables → each [L, head_dim/2] f32.
void VisionRopeCosSin(const std::array<int64_t, 3>& grid_thw, const Qwen3VLVisionConfig& cfg,
                      std::vector<float>* cos, std::vector<float>* sin);

}  // namespace vllm::multimodal
