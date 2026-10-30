// Kolibri-1 — weight structs and loader declarations (private header).
//
// (MODEL-TEXT-kolibri-1 W1, spec .agents/specs/kolibri-1-cpu.md) The weight
// structures mirror the checkpoint's tensor layout: sandwich (post-norm)
// norms per layer — input_layernorm, post_attn_norm,
// post_attention_layernorm, post_ffn_norm — GQA attention with per-head
// q/k RMS norms, and an MoE block on EVERY layer (384 routed experts top-6
// plus one ungated shared expert).
//
// THE RELEASED CHECKPOINT IS FP8 BLOCK-QUANTIZED (spec risk R1):
// quantization_config.weight_block_size = [128, 128] and every linear
// projection ships a weight_scale_inv sibling. The loader stores the fp8
// bytes RAW beside an f32-widened scale grid (the house Fp8BlockWeight, the
// rung dense_weight_loaders.h::LoadFp8BlockRaw loads) — the dequant-at-load
// vs CPU-fp8-GEMM decision for the forward arm is the forward wave's and is
// recorded there. `mlp.gate` (the router) is bf16 and lives in
// modules_to_not_convert; a router that ships fp8 is refused.
//
// THE PLAIN BF16 ARM (spec .agents/specs/kolibri1-bf16-arm.md): a checkpoint
// with NO quantization_config is a bf16 checkpoint — the upstream
// `Aleph-Alpha/Kolibri-1-BF16` artifact. Its weights are read as they are:
// no scale siblings, no dequant, and the same bf16 compute class the fp8
// arm's decoded weights ride. The scheme is resolved ONCE (see
// Kolibri1WeightScheme) and every later decision reads it, so the two arms
// cannot be confused with each other.
//
// NAME RECONCILIATION (the mapper, kolibri1.py:258-266): the HF layout puts
// the router gate at `mlp.gate`, the routed experts at `mlp.experts.N`, the
// shared expert at `mlp.shared_experts`, and the router bias OUTSIDE mlp at
// `moe.router.expert_bias` — the spec's `mlp.moe.*` inventory was wrong and
// the W1 manifest test pins the real names. This loader enumerates and
// consumes the CHECKPOINT's own names directly, so the reconciliation is
// the enumeration itself rather than a rename pass.
#pragma once

#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/kolibri1.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor, Fp8BlockWeight
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

class LoadedModel;  // model_registry.h

// The checkpoint's WEIGHT SCHEME, resolved once from the config (spec
// .agents/specs/kolibri1-bf16-arm.md).
//
//   kFp8Block  the released checkpoint: F8_E4M3 weights on the [128, 128]
//              block grid with an f32 `weight_scale_inv` sibling each. The
//              weights stay compressed at load and are dequantized in the
//              forward.
//   kBf16      a plain bf16 checkpoint: BF16 weights, NO scale siblings, no
//              dequant. The upstream `Aleph-Alpha/Kolibri-1-BF16` artifact
//              carries no `quantization_config` at all.
//
// Every other scheme is refused BY NAME, with a message that names the arms
// that do exist.
enum class Kolibri1WeightScheme {
  kFp8Block,
  kBf16,
};

// The scheme's name, for the refusal messages and for the record.
const char* Kolibri1WeightSchemeName(Kolibri1WeightScheme scheme);

// Resolves the checkpoint's scheme from its config's raw JSON. A config with
// no `quantization_config` is the plain bf16 arm; the fp8 block [128, 128]
// dynamic config is the fp8 arm; anything else throws, naming the scheme it
// found and the arms that exist.
Kolibri1WeightScheme ResolveKolibri1WeightScheme(const nlohmann::json& raw);

// One linear projection. Exactly one slot is populated: the fp8-block arm
// (quantized checkpoint) or the bf16 arm (a plain bf16 checkpoint's
// projections, and the fp8 arm's modules_to_not_convert modules: the router
// gate).
struct Kolibri1Projection {
  Fp8BlockWeight fp8_block;
  OwnedTensor bf16;

  bool IsFp8Block() const { return !fp8_block.Empty(); }
  bool IsBf16() const { return !bf16.Empty(); }
  bool Empty() const { return fp8_block.Empty() && bf16.Empty(); }
};

// SwiGLU expert projections (routed and shared experts share the shape:
// moe_intermediate 512 / shared_expert_intermediate 512).
struct Kolibri1ExpertWeights {
  Kolibri1Projection gate_proj;
  Kolibri1Projection up_proj;
  Kolibri1Projection down_proj;
};

struct Kolibri1AttnWeights {
  Kolibri1Projection q_proj;
  Kolibri1Projection k_proj;
  Kolibri1Projection v_proj;
  Kolibri1Projection o_proj;
  // Per-head q/k RMS norms before RoPE (kolibri1.py:107-118): [head_dim].
  OwnedTensor q_norm;
  OwnedTensor k_norm;
};

struct Kolibri1MoeWeights {
  // Router: `mlp.gate.weight`, BF16 [num_experts, hidden_size] — the
  // modules_to_not_convert module. Upstream runs it fp32
  // (kolibri1.py:171-177).
  OwnedTensor router_gate;
  // `model.layers.N.moe.router.expert_bias` (on disk, BF16 in the real
  // checkpoint) — the e_score_correction_bias the mapper renames
  // (kolibri1.py:258-261); resident f32 after a lossless widening.
  OwnedTensor e_score_correction_bias;
  std::vector<Kolibri1ExpertWeights> experts;  // [num_experts]
  Kolibri1ExpertWeights shared_experts;        // ungated, always added
};

// One backbone layer. EVERY layer is MoE; is_sliding selects the attention
// kind (RNoPE: RoPE only on sliding layers).
struct Kolibri1LayerWeights {
  bool is_sliding = false;

  // Sandwich (post-norm) layout: input_layernorm -> attn -> post_attn_norm
  // -> residual via post_attention_layernorm -> MoE -> post_ffn_norm ->
  // residual (kolibri1.py:210-253).
  OwnedTensor input_layernorm;
  OwnedTensor post_attn_norm;
  OwnedTensor post_attention_layernorm;
  OwnedTensor post_ffn_norm;

  Kolibri1AttnWeights attn;
  Kolibri1MoeWeights moe;
};

struct Kolibri1Weights {
  Kolibri1Params params;

  OwnedTensor embed_tokens;  // BF16 [vocab_size, hidden_size]
  OwnedTensor lm_head;       // BF16, untied
  OwnedTensor final_norm;    // BF16 [hidden_size]

  std::vector<Kolibri1LayerWeights> layers;
};

// Accounting result: every on-disk tensor must be classified.
struct Kolibri1Accounting {
  std::vector<std::string> missing;
  std::vector<std::string> duplicated;
  std::vector<std::string> unaccounted;
};

// A named expected tensor, tagged with its consumer for diagnostics.
struct Kolibri1Tensor {
  std::string name;
  std::string consumer;
};

// Enumerates every expected CHECKPOINT-side tensor name for `p` under
// `scheme`: the fp8-block layout (".weight" + ".weight_scale_inv" per linear
// projection, bf16 ".weight" for the router gate, norms, embedding,
// lm_head) or the plain bf16 layout (".weight" only per linear projection —
// a bf16 checkpoint has no scales). This is the mlp.moe.* vs mlp.gate
// reconciliation: the checkpoint's own names.
std::vector<Kolibri1Tensor> EnumerateKolibri1Tensors(
    const Kolibri1Params& p, Kolibri1WeightScheme scheme);

// Classifies every on-disk name against the expected set.
Kolibri1Accounting AccountKolibri1Tensors(
    const Kolibri1Params& p, Kolibri1WeightScheme scheme,
    const std::vector<std::string>& present);

// Loads all weights from the safetensors shards (the 32-shard index resolves
// through the per-shard headers). Performs full accounting before
// materialization; throws on missing/duplicated/unaccounted tensors, on a
// weight scheme that is neither fp8 block [128, 128] nor plain bf16, on a
// projection whose dtype contradicts the resolved scheme, and on a router
// gate that is not bf16.
Kolibri1Weights LoadKolibri1Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// The checked accessor for the weights a ModelRegistry::Load produced: the
// B2b device waves receive the registry's LoadedModel (Prepare/Forward) and
// read the loaded tree back out through this seam instead of re-loading.
// Refuses by name when `model` is not a Kolibri1ForCausalLM load.
const Kolibri1Weights& Kolibri1LoadedModelWeights(LoadedModel& model);

}  // namespace vllm
