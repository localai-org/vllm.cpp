// RADIO vision tower (`RadioModel`) and the Nemotron Nano VL image projector
// (pixel shuffle v2 + `mlp1`), the image half of `NemotronH_Nano_VL_V2` and
// `NemotronH_Nano_Omni_Reasoning_V3`.
//
// Ported from vLLM at the parity pin `e126687a9a`:
//   RadioPosEmbedForGrid         <- radio.py:386-421 (_get_pos_embeddings, CPE)
//   RadioVisionTower::Forward    <- radio.py:198-212 (ViTPatchGenerator.forward,
//                                   dynamic path), :265-312 (apply_pos_enc_dynamic,
//                                   cls_token_dynamic), :493-520 (encoder layer
//                                   and encoder), :579-644 (per-image mask),
//                                   :745-774 (_extract_final), and
//                                   intern_vit.py:145-290 (attention, MLP)
//   NanoNemotronVLPixelShuffle   <- nano_nemotron_vl.py:1012-1048 (v2)
//   NanoNemotronVLProjector      <- nano_nemotron_vl.py:955-976 (mlp1)
// The configuration (ViT-H/16, LayerNorm eps 1e-6, erf GELU, qkv bias) is
// `vllm/transformers_utils/configs/radio.py:12-110`.
//
// ─── THE SILENT TRAPS (each is mutation-gated in test_nano_nemotron_vl_vision) ─
//   1. The CPE positional table is bilinearly interpolated to a SQUARE
//      `max(grid_h, grid_w)` grid and THEN cropped to `grid_h x grid_w`
//      (radio.py:401-410). Interpolating straight to the grid keeps every shape.
//   2. Every image gets its OWN copy of the 4 CLS + 6 register tokens in front
//      of its patches, and its own attention segment (radio.py:295-312,
//      :579-589). The 10 rows are stripped per image before the shuffle.
//   3. Pixel shuffle v2 permutes (0, 1, 3, 2, 4, 5). The v1 order transposes
//      the image and keeps the shape and the multiset of values.
//   4. The table is interpolated in f32 and cast back to the model dtype
//      BEFORE it is added to the patch embeddings (radio.py:403-408).
//
// The layer-scale `ls1`/`ls2` are 1.0: `initializer_factor` defaults to 1.0
// (configs/radio.py:69), the released checkpoint ships no `ls*` tensor, and
// vLLM's loader skips them (radio.py:732-734).
#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"

namespace vllm::multimodal {

// `RadioConfig` for `vit_huge_patch16_224` (configs/radio.py:12-17) with the
// released `C-RADIOv4-H` arguments: cls_token_per_teacher over 4 unique
// teachers, register_multiple 10, cpe_max_size 2048.
struct RadioVisionConfig {
  int64_t hidden_size = 1280;
  int64_t num_attention_heads = 16;
  int64_t num_hidden_layers = 32;
  int64_t intermediate_size = 5120;
  int64_t patch_size = 16;
  // ClsToken (radio.py:60-106): num_cls_tokens CLS rows, then
  // `register_multiple - num_cls_tokens % register_multiple` register rows.
  int64_t num_cls_tokens = 4;
  int64_t num_registers = 6;
  // The positional table's grid: max_input_dims / patch_size
  // (radio.py:140-158), 2048 / 16 = 128.
  int64_t pos_rows = 128;
  int64_t pos_cols = 128;
  // `max_input_dims != input_dims` (radio.py:144). True for every released
  // checkpoint (preferred_resolution 768 < cpe_max_size 2048).
  bool cpe_mode = true;
  float layer_norm_eps = 1e-6f;
  // Production dtype: the model dtype, bf16. f32 exists so the per-stage gate
  // can pin the arithmetic without the bf16 rounding envelope.
  vt::DType compute_dtype = vt::DType::kBF16;

  int64_t head_dim() const { return hidden_size / num_attention_heads; }
  int64_t num_skip() const { return num_cls_tokens + num_registers; }
  int64_t patch_dim() const { return 3 * patch_size * patch_size; }
};

// Host weights as bf16 bit patterns (the checkpoint's storage dtype), torch
// layout: a Linear weight is [out, in].
struct RadioBlockWeights {
  std::vector<uint16_t> norm1_w, norm1_b;  // [H]
  std::vector<uint16_t> qkv_w, qkv_b;      // [3H, H], [3H]
  std::vector<uint16_t> proj_w, proj_b;    // [H, H], [H]
  std::vector<uint16_t> norm2_w, norm2_b;  // [H]
  std::vector<uint16_t> fc1_w, fc1_b;      // [I, H], [I]
  std::vector<uint16_t> fc2_w, fc2_b;      // [H, I], [H]
};

struct RadioVisionWeights {
  std::vector<uint16_t> embedder_w;  // [H, 3*p*p], bias-free (radio.py:162-164)
  std::vector<uint16_t> pos_embed;   // [pos_rows * pos_cols, H]
  std::vector<uint16_t> cls_token;   // [num_skip, H]
  std::vector<RadioBlockWeights> blocks;
};

// One image, already preprocessed and patchified: `patches` is
// [grid_h * grid_w, 3*p*p] in the (c, yy, xx) patch order of
// `DynamicResolutionImageTiler.stack` (processors/nano_nemotron_vl.py:555-566).
struct RadioImage {
  std::vector<float> patches;
  int64_t grid_h = 0;
  int64_t grid_w = 0;
};

// `_get_pos_embeddings` for one grid: [grid_h * grid_w, H] f32. The table is
// read as bf16 (the checkpoint dtype); when `round_to_bf16` is set the result
// is rounded to bf16 exactly as `.to(pos_embed.dtype)` does in the production
// dtype.
std::vector<float> RadioPosEmbedForGrid(const std::vector<uint16_t>& pos_embed,
                                        int64_t grid_h, int64_t grid_w,
                                        const RadioVisionConfig& cfg,
                                        bool round_to_bf16);

// The tower with its weights uploaded once. `Forward` runs every image through
// the patch embedder, the positional table and the 32 blocks, with one
// attention segment per image, and returns the patch features with the CLS and
// register rows stripped: [sum_i grid_h_i * grid_w_i, H] f32 (values rounded
// to the compute dtype).
class RadioVisionTower {
 public:
  RadioVisionTower(const RadioVisionWeights& weights, const RadioVisionConfig& cfg,
                   vt::Backend& backend);
  ~RadioVisionTower();
  RadioVisionTower(const RadioVisionTower&) = delete;
  RadioVisionTower& operator=(const RadioVisionTower&) = delete;

  std::vector<float> Forward(const std::vector<RadioImage>& images) const;
  const RadioVisionConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

// `pixel_shuffle` ps_version "v2" (nano_nemotron_vl.py:1012-1029) on ONE image:
// `features` is [grid_h * grid_w, dim]; returns
// [(grid_h/r) * (grid_w/r), dim * r * r] with r = 1 / downsample_ratio.
std::vector<float> NanoNemotronVLPixelShuffle(const std::vector<float>& features,
                                              int64_t grid_h, int64_t grid_w,
                                              int64_t dim, int64_t r);

struct NanoNemotronVLProjectorConfig {
  int64_t in_dim = 1280 * 4;        // vit_hidden_size * r^2
  int64_t hidden_dim = 20480;       // projector_hidden_size
  int64_t out_dim = 2688;           // text_config.hidden_size
  float rms_norm_eps = 1e-5f;       // nano_nemotron_vl.py:964
  vt::DType compute_dtype = vt::DType::kBF16;
};

struct NanoNemotronVLProjectorWeights {
  std::vector<uint16_t> norm_w;  // mlp1.0.weight [in_dim]
  std::vector<uint16_t> fc1_w;   // mlp1.1.weight [hidden_dim, in_dim], no bias
  std::vector<uint16_t> fc2_w;   // mlp1.3.weight [out_dim, hidden_dim], no bias
};

// `mlp1`: RMSNorm -> Linear -> ReLU squared -> Linear, weights uploaded once.
class NanoNemotronVLProjector {
 public:
  NanoNemotronVLProjector(const NanoNemotronVLProjectorWeights& weights,
                          const NanoNemotronVLProjectorConfig& cfg,
                          vt::Backend& backend);
  ~NanoNemotronVLProjector();
  NanoNemotronVLProjector(const NanoNemotronVLProjector&) = delete;
  NanoNemotronVLProjector& operator=(const NanoNemotronVLProjector&) = delete;

  // `x` is [rows, in_dim] f32 (values representable in the compute dtype);
  // returns [rows, out_dim] f32 rounded to the compute dtype.
  std::vector<float> Forward(const std::vector<float>& x, int64_t rows) const;
  const NanoNemotronVLProjectorConfig& config() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm::multimodal
