// vllm.cpp original — the DENSE Qwen3.6-27B text gate
// (unsloth/Qwen3.6-27B-NVFP4, arch Qwen3_5ForConditionalGeneration, text_config
// model_type "qwen3_5_text"). See .agents/specs/qwen27b-w4a4-notes.md.
//
// The 27B shares the 35B hybrid backbone WHOLESALE (GDN linear-attention + gated
// full-attention + Gemma (1+w) RMSNorm + mRoPE->NeoX); it REUSES the 35B's
// GdnLayerWeights / FullAttnLayerWeights sub-structs and the GdnBlock /
// FullAttnBlock forward helpers verbatim. The ONLY structural change is the
// per-layer sparse-MoE block being replaced by a DENSE SwiGLU MLP (gate/up/down,
// intermediate 17408): down( silu(gate(x)) * up(x) ). See notes §2.
//
// Quant: compressed-tensors NVFP4 W4A4 (notes §3). For the CPU correctness path
// each quantized Linear is MATERIALIZED to bf16 at load via the CT weight-dequant
// reference (DequantCtNvfp4WeightToF32, multiply by 1/weight_global_scale) and
// the existing bf16 forward carries it — the true GB10 fp4xfp4 GEMM is a later,
// GPU-gated step (notes §5 steps 5-7). The activation-quant round-trip is dropped
// on this correctness path (bf16 activations), matching the notes' §5 step-6a
// FAST PATH; that is a tiny numeric deviation vs true W4A4, validated later vs
// the pip-vLLM oracle golden.
//
// Which Linears are quantized (notes §3.6): QUANTIZED (W4A4) = every dense-MLP
// gate/up/down_proj, every self_attn q/k/v/o_proj, and the GDN linear_attn
// out_proj. NOT quantized (bf16 on disk) = the GDN in_proj_{qkv,z,a,b}, conv1d,
// A_log, dt_bias, all norms, embed_tokens, lm_head, mtp.*, and visual.*.
#pragma once

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/config/multimodal.h"
#include "vllm/model_executor/models/qwen3_5.h"  // PagedKvCache, GdnStateCache + v1 attention metadata
#include <functional>

#include "vllm/model_executor/layers/quantization/fp8_block_quant.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor, Gdn/FullAttn weights, TensorResolver
#include "vllm/config/multimodal.h"                  // MultiModalConfig (the tower-skip borrow)
#include "vllm/model_executor/models/qwen3_vl_vision.h"  // MODEL-QWEN35-DENSE-VL-EXL3: the dense arm's tower
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

// Dense SwiGLU MLP (replaces the 35B MoE block). Synthetic/legacy projections
// use Matmul-B [in,out], ordinary BF16 checkpoints use one raw-NK gate/up
// owner, and compressed checkpoints retain NVFP4 residents.
struct DenseMlpWeights {
  OwnedTensor gate_proj;  // bf16 [H, I]
  OwnedTensor up_proj;    // bf16 [H, I]
  OwnedTensor down_proj;  // bf16 [I, H]

  // Plain-BF16 vLLM topology: gate/up checkpoint rows concatenated into one raw
  // torch-Linear [2I,H] owner (nk=true), consumed by one MatmulBT + SiluAndMul.
  // The legacy split fields above remain the synthetic-test/diagnostic fallback.
  OwnedTensor gate_up_proj;

  // W4A4 fp4-resident variants (compressed-tensors NVFP4, notes §5 step-6a). On
  // the real 27B CUDA load these are populated (kept in the on-disk [N=out,K=in]
  // orientation vt::MatmulNvfp4 reads) and the bf16 fields above are left EMPTY;
  // the synthetic CPU tests populate the bf16 fields and leave these empty.
  Nvfp4Weight gate_proj_fp4;  // [N=I, K=H]
  Nvfp4Weight up_proj_fp4;    // [N=I, K=H]
  Nvfp4Weight down_proj_fp4;  // [N=H, K=I]

  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): block-wise (128x128) FP8 MLP
  // projections. This block had NO fp8 rung at all before that row, so a
  // block-wise MLP fell through to `LoadMergedBf16RawNK` and died on
  // "expected BF16". Loaded as separate shards and MERGED at first use (#1189
  // M6, `gate_up_fp8_block_merged` below): block scales concatenate losslessly
  // along N, so the merge is a device-residency step rather than a loader one.
  Fp8BlockWeight gate_proj_fp8_block;  // [N=I, K=H]
  Fp8BlockWeight up_proj_fp8_block;    // [N=I, K=H]
  Fp8BlockWeight down_proj_fp8_block;  // [N=H, K=I]

  // MODEL-FP8-BLOCK-MERGED (#1189 M6): vLLM's MergedColumnParallelLinear
  // `gate_up_proj` as ONE device operand, gate rows first. Block scales
  // concatenate losslessly along N, so this is the whole merge — no alpha
  // vector, no shared-scale guard, no opt-in flag, which is what the per-tensor
  // fp8 QKV merge beside it needs and this one does not. Built lazily-once by
  // `dense_fp8_block::ResidentFp8BlockMerged`; the per-shard residents are then
  // never built. Empty on every non-block owner.
  Fp8BlockMergedResident gate_up_fp8_block_merged;

  // CUDA resident for vLLM's MergedColumnParallelLinear gate_up_proj. The
  // checkpoint stores gate/up separately; production concatenates their packed
  // rows and linear block scales once, then keeps only the combined packed
  // operand and combined swizzled scale resident. Mutable matches Nvfp4Weight's
  // lazy device-resident handles. The split weights remain host-resident for
  // VT_FP4_MERGED_GATE_UP=0 and VT_FP4_FULL_TACTICS=0 diagnostics.
  mutable std::shared_ptr<void> d_gate_up_packed;
  mutable std::shared_ptr<void> d_gate_up_scale_sw;
  // One physical merged projection owns one device alpha. Keep the exact merged
  // host scalar in model-lifetime storage because Backend::Copy is asynchronous.
  mutable float gate_up_alpha = 0.0F;
  mutable std::shared_ptr<void> d_gate_up_alpha;

  // QUANT-EXL3 (#2181) / MODEL-QWEN35-EXL3 (#2495 item 3): the exllamav3
  // trellis arm of the dense SwiGLU MLP. gate and up stay SEPARATE for the
  // same reason the attention shards do -- a trellis merge on the output dim
  // interleaves per input tile -- and `layers::Exl3MlpGateUpMethod` is the
  // shared seam that consumes the pair, so the model never learns which scheme
  // it bound. Exactly one representation is populated per layer.
  Exl3Weight gate_proj_exl3;  // [K=H, N=I]
  Exl3Weight up_proj_exl3;    // [K=H, N=I]
  Exl3Weight down_proj_exl3;  // [K=I, N=H]

  // `down_proj_exl3`, mirroring `Qwen3DenseMlpWeights::IsExl3` in `qwen3.h`:
  // down is the projection that has no merged twin in any arm, so it is the one
  // whose presence cannot be confused with a partially built resident.
  bool IsExl3() const { return !down_proj_exl3.Empty(); }
};

// Exact scalar processing for a two-shard CT NVFP4 MergedColumnParallelLinear.
// Mirrors compressed_tensors_w4a4_nvfp4.py:95-138: max each logical-shard
// divisor first, reciprocate once, then multiply the two reciprocals into alpha.
struct DenseGateUpGlobals {
  float input_global_scale_inv = 0.0F;  // max on-disk input divisor
  float weight_global_scale = 0.0F;     // reciprocal of max on-disk weight divisor
  float alpha = 0.0F;
};

DenseGateUpGlobals MergeDenseGateUpGlobals(const Nvfp4Weight& gate,
                                           const Nvfp4Weight& up);

// One dense decoder layer: input/post norms + one attention variant + dense MLP.
struct Qwen3_5DenseLayerWeights {
  bool is_linear_attention = false;
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  GdnLayerWeights gdn;                    // valid iff is_linear_attention
  FullAttnLayerWeights attn;             // valid iff !is_linear_attention
  DenseMlpWeights mlp;                   // every layer has a dense MLP
};

// Whole dense-model text weights. The CHECKPOINT may store the head BF16, FP8
// (per-channel scale) or ModelOpt NVFP4 — the 27B NVFP4 publishers disagree, and
// revisions of one repo disagree with each other (issue #164). BF16/FP8 are
// materialized into `lm_head`, NVFP4 stays PACKED in `lm_head_fp4`
// (PERF-27B-LMHEAD-FP4, issue #213); exactly one is populated.
struct Qwen3_5DenseWeights {
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (NOT transposed; embed lookup)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab]  (dequantized -> Matmul-B layout)
  // NVFP4-resident output head [N=vocab, K=H], kept in the on-disk orientation the
  // fp4 GEMMs read. Mirrors Qwen3_5MoeWeights::lm_head_fp4 and vLLM's own decision
  // to leave the head quantized: get_quant_method accepts ParallelLMHead
  // (modelopt.py:2508-2536) over the bare `lm_head` key (modelopt.py:2491-2496),
  // so ModelOptNvFp4W4A16LinearMethod — pinning MarlinNvFp4LinearKernel
  // (modelopt.py:1249,1283-1284) — resolves it and logits_processor._apply_head
  // calls quant_method.apply every step (logits_processor.py:98-133). Empty on
  // every BF16/FP8/GGUF/tied checkpoint.
  Nvfp4Weight lm_head_fp4;
  // Mirrors tie_word_embeddings: logits reuse embed_tokens as raw [V,H]
  // torch-Linear storage, so no second host/device owner is created.
  bool tied_lm_head = false;
  // MODEL-QWEN35-EXL3 (#2495 item 5): an EXL3 output head, kept in the trellis
  // rather than materialized. Its WIDTH is its own and is read from the tensor,
  // never from `quantization_config.bits`: `turboderp/Llama-3.2-1B-Instruct-exl3`
  // @ 3.0bpw ships a SIX-bit head under a config that says 3.0, and a reader
  // that trusts the scalar decodes it at the wrong width with every shape check
  // still passing (`Exl3Weight::Bits`). Empty on every other head storage.
  Exl3Weight lm_head_exl3;
  std::vector<Qwen3_5DenseLayerWeights> layers;

  // MODEL-QWEN35-DENSE-VL-EXL3 (ISSUE-LOCAL-01M3AHX9DQX8HNE32G80C9VGMJ): the
  // checkpoint's vision tower, loaded by `LoadQwen3_5Dense` when the index
  // carries `model.visual.*` tensors. Before this member the dense loader
  // SILENTLY DROPPED those 333 bf16 tensors — a vision-inclusive checkpoint
  // loaded as a text-only model with no diagnostic, so no production path
  // could ever answer an image prompt. Unquantized-but-required: the tower is
  // bf16 on the EXL3 27B even while the text arm is trellis-quantized (see
  // `IsQwen27QuantizedLinear`, which already exempted the prefix). EMPTY (and
  // `has_visual` false) on every text-only checkpoint, which is what keeps a
  // text load byte-identical.
  multimodal::Qwen3VLVisionWeights visual;
  // The tower's geometry, the checkpoint's `vision_config` as the family
  // builder expresses it (`Qwen3_5DenseVisionConfig`); valid iff `has_visual`.
  multimodal::Qwen3VLVisionConfig visual_cfg;
  bool has_visual = false;
  // #607 L3, the TOWER SKIP: `model.visual.*` was present and deliberately left
  // unread because every modality the tower serves was at limit 0. Distinct
  // from a text-only checkpoint, where this and `has_visual` are both false.
  bool vision_skipped = false;
};

// True iff the projection named `name` is a W4A4-quantized Linear in the 27B
// (notes §3.6). `name` is the module path WITHOUT the trailing ".weight*" (e.g.
// "model.language_model.layers.0.mlp.gate_proj"). Encodes the checkpoint's
// config.json `ignore` list: the quantized set is the dense-MLP {gate,up,down}
// proj, the self_attn {q,k,v,o} proj, and the GDN linear_attn out_proj; every
// other Linear (GDN in_proj_*, lm_head, mtp.*, visual.*, ...) is bf16.
bool IsQwen27QuantizedLinear(const std::string& name);

// Materialize one compressed-tensors NVFP4 W4A4 Linear to an owned bf16 tensor
// in Matmul-B layout [in, out]. Reads `<proj>.weight_packed` (U8 [out, in/2]),
// `<proj>.weight_scale` (F8_E4M3 [out, in/16]) and `<proj>.weight_global_scale`
// (F32 scalar divisor); dequants to f32 via DequantCtNvfp4WeightToF32 (which
// reciprocates the global scale), rounds to bf16, and transposes. Exposed for
// unit testing. The `<proj>.input_global_scale` (activation divisor) is ignored
// on this bf16-activation correctness path (notes §3.4 / §5 step-6a).
// `lm_head` across the three storage forms the 27B NVFP4 publishers actually ship
// (issue #164): BF16, FP8 `+_scale` (per-output-channel or per-tensor), and NVFP4
// `+_scale` `+_scale_2`/`+_global_scale`. Always returns bf16 [in, out] Matmul-B,
// so a BF16 head is byte-identical to the previous LoadBf16Transposed call.
// `has` probes optional companion tensors. Exported for the loader gate.
OwnedTensor LoadLmHeadAnyDtype(const TensorResolver& get,
                               const std::function<bool(const std::string&)>& has,
                               const std::string& name);

// PERF-27B-LMHEAD-FP4 (issue #213). Load the dense output head into EXACTLY ONE
// of the two owners: a ModelOpt/compressed-tensors NVFP4 head stays PACKED in
// `fp4_out`, every other storage form is materialized bf16 [in, out] into
// `bf16_out` by LoadLmHeadAnyDtype. `proj` omits the trailing ".weight".
void LoadDenseLmHead(const TensorResolver& get,
                     const std::function<bool(const std::string&)>& has,
                     const std::string& proj, OwnedTensor& bf16_out,
                     Nvfp4Weight& fp4_out);

// SPEC-DFLASH2-QUANT-LMHEAD (#1628). The ROUTING QUESTION `LoadDenseLmHead`
// asks — "does this head take the packed arm?" — exported so a SECOND consumer
// of the same head asks it rather than re-deriving it. `LoadDenseLmHead` is its
// only other caller, so there is one predicate and not two descriptions of one.
//
// The second consumer is the DFlash/DSpark draft, which owns no head and runs
// the TARGET's (`LoadDflashSharedLmHead`, qwen3_dflash.h). Its shared-head read
// used to test the STORED DTYPE, which cannot separate a head that was
// dequantized into something the target does not compute with from a head kept
// packed and computed with natively.
bool DenseLmHeadTakesNvfp4(const std::function<bool(const std::string&)>& has,
                           const std::string& proj);

// MODEL-DFLASH2-NVFP4 (#2758). The per-projection NVFP4 probe and reader,
// exported for the DFlash2 draft loader, which reads the SAME ModelOpt spelling
// from a DIFFERENT checkpoint.
//
// `IsNvfp4Projection` answers `has(<proj>.weight_packed) ||
// has(<proj>.weight_scale_2)`, which is the union of the two spellings a
// producer emits: compressed-tensors writes `.weight_packed` plus a
// `.weight_global_scale` DIVISOR, ModelOpt writes `.weight` plus a
// `.weight_scale_2` SCALE. `LoadNvfp4AnyNaming` reads whichever of the two the
// checkpoint ships and normalises both onto one `Nvfp4Weight`, including the
// W4A16-unless-VT_MODELOPT_W4A4 activation decision.
//
// EXPORTED RATHER THAN RE-DERIVED, for the reason stated on
// `DenseLmHeadTakesNvfp4` above: the draft and the target must not be able to
// disagree about what an NVFP4 projection IS. `proj` omits the trailing
// ".weight".
bool IsNvfp4Projection(const std::function<bool(const std::string&)>& has,
                       const std::string& proj);
Nvfp4Weight LoadNvfp4AnyNaming(const TensorResolver& get,
                               const std::function<bool(const std::string&)>& has,
                               const std::string& proj);

// True when the checkpoint ships an EXPLICIT head under either naming
// (`<proj>.weight`, or `<proj>.weight_packed` for compressed-tensors NVFP4);
// false means `tie_word_embeddings`.
bool DenseCheckpointHasLmHead(const std::function<bool(const std::string&)>& has,
                              const std::string& proj);

// VT_LMHEAD_FP4 (default ON): the in-binary rollback for the packed head. `0`
// restores the dequantize-at-load owner, so the A/B is same-binary.
bool DenseLmHeadFp4Enabled();

// VT_MODELOPT_W4A4 (default 0): consume a projection's on-disk activation
// divisor, so `Nvfp4Weight::IsTrueW4A4()` flips and the weight routes to the
// fp4-ACTIVATION GEMM instead of the weight-only one (docs/ENVIRONMENT.md).
// ONE reader in the tree, exported by MODEL-DFLASH2-NVFP4 (#2758) so the DFlash
// draft loader reports the arm this build actually takes rather than reading
// the same environment variable a second time.
bool ModelOptW4A4OptIn();

Fp8Weight LoadFp8RawShared(const TensorResolver& get, const std::string& proj);

OwnedTensor MaterializeCtNvfp4Bf16Transposed(const TensorResolver& get,
                                             const std::string& proj);

// Load and concatenate raw BF16 torch-Linear weights `[N_i,K]` along their
// output rows, preserving the exact listed order and setting `nk=true` for
// vt::MatmulBT. This is vLLM MergedColumnParallelLinear's physical ownership
// rule for GDN `in_proj_ba` now and `in_proj_qkvz` in the separately gated W2.
// Exposed for the focused loader contract.
OwnedTensor LoadMergedBf16RawNK(const TensorResolver& get,
                                const std::vector<std::string>& names);

// Load the dense checkpoint's GDN block. Exposed so the loader regression can
// assert the one-owner invariant (`in_proj_ba` populated, split b/a empty)
// without manufacturing unrelated attention/MLP tensors.
GdnLayerWeights LoadQwen3_5DenseGdn(const TensorResolver& get,
                                    const std::string& layer_base);

// Load one dense decoder layer. `layer_type` is "linear_attention" or
// "full_attention". Prefix is "{backbone_prefix}layers.{layer_idx}.", defaulting
// to the VL spelling every checkpoint we gate today uses, so this seam is
// byte-identical for the 27B/35B/Coder callers; `LoadQwen3_5Dense` passes the
// prefix it resolved ONCE from the shard index
// (`ResolveQwen3_5BackbonePrefix`). Routes each Linear to ordinary BF16 or
// compressed NVFP4 based on tensor presence.
Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(
    const TensorResolver& get, const std::string& layer_type, int64_t layer_idx,
    const std::string& backbone_prefix = std::string(kQwen3_5VlBackbonePrefix));

// The same load with an EXPLICIT presence probe — what `LoadQwen3_5Dense` calls
// per layer, and the cheapest one to pass when the caller HAS a name index.
// FIX-PROBE-CANNOT-SAY-NO (#1258): the resolver-only overload above used to
// answer `has` with a constant `true`, which forced every routed projection down
// the compressed-tensors spelling and, after #1189 M3 taught the loader to
// cross-check the config against the tensors, made that seam refuse every
// checkpoint that has no block-wise FP8 scale — which is all of them but one.
// It now derives the probe from the resolver (`dense_loaders::
// ProbeThroughResolver`), so both overloads answer the same question truthfully
// and differ only in cost. Exposed so the loader gate can drive a whole
// synthetic layer through the SAME routing production takes.
//
// A probe handed here must be able to answer `false`: this overload refuses one
// that cannot (`dense_loaders::CheckProbeCanAnswerNo`), because a predicate that
// only says yes reports every optional tensor as present and the next guard to
// ask about one blames the checkpoint.
// MODEL-FP8-BLOCK-WEIGHT (#1189 M3): `block` carries the checkpoint's declared
// `weight_block_size`, `activation_scheme` and `modules_to_not_convert`, read
// ONCE by `LoadQwen3_5Dense` from the quantization config. A default-constructed
// value means "not block-wise", which is byte-identical to the routing before
// that row. The two seams above default to it; the production loader passes the
// value it read, because a dtype probe alone cannot detect a config/tensor
// DISAGREEMENT and that is where a silent wrong-scale bug lives.
Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(
    const TensorResolver& get, const std::function<bool(const std::string&)>& has,
    const std::string& layer_type, int64_t layer_idx,
    const std::string& backbone_prefix,
    const Fp8BlockQuantConfig& block);

Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(
    const TensorResolver& get, const std::function<bool(const std::string&)>& has,
    const std::string& layer_type, int64_t layer_idx,
    const std::string& backbone_prefix = std::string(kQwen3_5VlBackbonePrefix));

// Full dense-model load across the given shards. Uses config.num_hidden_layers
// and config.layer_types. A checkpoint that carries `model.visual.*` loads the
// tower through `LoadQwen3_5DenseVision`; `mm_config` is the engine's modality
// limits, and when every modality the tower serves is at limit 0 the tower is
// left unread and `vision_skipped` is set instead. A null `mm_config` (every
// non-engine caller) loads the tower as before. The checkpoint's MTP head is
// intentionally loaded on demand by LoadQwen3_5MTP when speculative decoding
// is enabled; it is not part of the always-resident target weights.
Qwen3_5DenseWeights LoadQwen3_5Dense(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config,
                                     vt::Queue* load_queue = nullptr,
                                     const MultiModalConfig* mm_config = nullptr);

// Host-lifetime helpers for ordinary dense CUDA models. The release function
// drops only tensors whose authoritative raw/F32 device representation exists;
// unused fallbacks stay host-resident. The caller synchronizes first.
// MODEL-FP8-BLOCK-LINEAR (#1189 M4), the M4/M5 seam. Throws by name when the
// load produced a block-wise FP8 weight AND `device` has no block-scaled GEMM
// to run it with. Inert on a device that has one, which is what M4 changed:
// the forward reads these weights now, so the remaining gap is the kernel and
// not the wiring. Called from `PrepareQwen3_5Dense`, i.e.
// `ModelRegistry::Prepare`, so the refusal lands before the first forward and
// before any graph capture. M5 (`489a9a4c0`) NARROWED this rather than deleting
// it: the mainloop-scaled CUTLASS kernel covers `VT_CUTLASS_FP8_ARCHS` (12.0a,
// 12.1a) only, so a CUDA arch outside that cell is still refused here by name.
void RefuseUnrunnableQwen3_5DenseFp8Block(const Qwen3_5DenseWeights& weights,
                                          vt::DeviceType device);

bool IsPlainBf16Qwen3_5Dense(const Qwen3_5DenseWeights& weights);
size_t ReleaseResidentQwen3_5DenseHostWeights(Qwen3_5DenseWeights& weights);

// Dense single-sequence reference forward (text path). Mirrors
// Qwen3_5Model::ForwardDense but runs the dense SwiGLU MLP in place of the MoE
// block; reuses the 35B GDN + gated-attention + norm forward helpers. Returns
// logits [T, vocab] f32 (T = token_ids.size()). CPU or CUDA per `queue`.
class Qwen3_5DenseModel {
 public:
  // Eagerly create every raw/F32 resident needed by the currently loaded dense
  // layers. Used by the bounded direct-device loader before releasing host
  // staging bytes; normal forwards remain lazy.
  static void PrepareBf16Resident(const Qwen3_5DenseWeights& weights,
                                  vt::Queue& queue);

  // PERF-27B-LMHEAD-FP4 (issue #213). Build the resident form of the packed
  // `lm_head_fp4` THIS backend's logits GEMM consumes: the Marlin W4A16 repack
  // on CUDA (PRE-CAPTURE), else the dequantized bf16 [K,N] operand. Inert when
  // the head is not packed. Called from the registry `prepare` hook.
  static void PrepareLmHeadResident(const Qwen3_5DenseWeights& weights,
                                    vt::Queue& queue);
  // PERF-27B-GDN-FP8-QKVZ. Build the merged N-concatenated [qkv;z] FP8 operand
  // for every eligible GDN layer NOW — at model prepare, before any forward and
  // therefore before any CUDA-graph capture. A resident built inside stream
  // capture allocates and copies mid-capture, which aborts the capture; this is
  // the same reason PERF-27B-LMHEAD-FP4 builds its Marlin operand up front.
  // No-op on a non-staging (CPU) queue, on a non-FP8 owner, and whenever the
  // merge is not selected — the split path then never pays for the operand.
  static void PrepareGdnFp8Resident(const Qwen3_5DenseWeights& weights,
                                    const HfConfig& config, vt::Queue& queue);

  // Batched PAGED dense forward — the 27B analogue of Qwen3_5Model::Forward.
  // Same signature/structure (paged KV cache for the full-attn layers, batched
  // GDN recurrent state for the GDN layers, the f32 residual thread), reusing the
  // 35B GDN/FullAttn paged machinery VERBATIM with the dense SwiGLU MLP
  // (RunDenseLayerPaged) in place of the MoE block. One PagedKvCache per full-attn
  // layer + one GdnStateCache per GDN layer, in layer order. Returns
  // [num_actual_tokens, vocab] f32 logits (lm_head applied). Runs on `queue`'s
  // device. See qwen3_5.h::Qwen3_5Model::Forward for the metadata contract.
  // `logits_indices` (optional): identical semantics to
  // Qwen3_5Model::Forward — gather the per-request last-token hidden rows
  // on-device before lm_head (prefill/mixed) so the return is [num_reqs, vocab].
  static std::vector<float> Forward(const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& positions,
                                    const v1::CommonAttentionMetadata& attn_meta,
                                    const v1::GDNAttentionMetadata& gdn_meta,
                                    const std::vector<PagedKvCache>& attn_kv,
                                    const std::vector<GdnStateCache>& gdn_state,
                                    const Qwen3_5DenseWeights& weights,
                                    const HfConfig& config, vt::Queue& queue,
                                    const std::vector<int32_t>& logits_indices = {});

  // MODEL-KEV Phase 3: the pooling forward — the same embed + layer stack as
  // Forward, stopping after the final GemmaRMSNorm (+ optional gather) with NO
  // lm_head, mirroring Qwen3DenseModel::ForwardHidden (qwen3.cpp:570-592).
  // The [n_out, H] f32 rows are downloaded to the host carrier for the kev
  // PointerHead readout, whose ops are host-side.
  static ForwardLogits ForwardHidden(
      const std::vector<int32_t>& token_ids,
      const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const v1::GDNAttentionMetadata& gdn_meta,
      const std::vector<PagedKvCache>& attn_kv,
      const std::vector<GdnStateCache>& gdn_state,
      const Qwen3_5DenseWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // DEVICE-resident variant of Forward (sampler-on-device hot path): same contract
  // as Forward but returns the lm_head output as a pool-backed DEVICE buffer
  // (ForwardLogits::device_*) with NO full-logits D2H. See
  // Qwen3_5Model::ForwardDevice.
  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const v1::GDNAttentionMetadata& gdn_meta,
      const std::vector<PagedKvCache>& attn_kv,
      const std::vector<GdnStateCache>& gdn_state,
      const Qwen3_5DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  // ForwardDevice + the DRAFTER hidden-state tap (SPEC-MTP I5c). Byte-identical
  // logits to ForwardDevice, and additionally moves the full [num_actual_tokens, H]
  // post-final-norm hidden into `*hidden_out` (device-owning) for the MTP drafter's
  // propose(). `hidden_out` may be null. Not wired into the runner until I5d.
  static ForwardLogits ForwardDeviceTap(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const v1::GDNAttentionMetadata& gdn_meta,
      const std::vector<PagedKvCache>& attn_kv,
      const std::vector<GdnStateCache>& gdn_state,
      const Qwen3_5DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
      Qwen3_5MTPHiddenStates* hidden_out,
      const std::vector<int32_t>& logits_indices = {});

  // ForwardDevice + the DFlash multi-layer aux hidden taps (SPEC-DFLASH D1,
  // DF-AUX-TAPS). Byte-identical logits to ForwardDevice; additionally captures
  // (hidden + residual) at each `aux_out->layer_ids` boundary into
  // `aux_out->tensor` = [T, H×taps] (concat order = layer_ids), mirroring vLLM's
  // eagle3 aux capture (see Qwen3_5AuxTaps). `aux_out` may be null (then exactly
  // ForwardDevice). Not wired into the runner until DFlash D4; byte-identical when
  // unused.
  static ForwardLogits ForwardDeviceMultiTap(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const v1::GDNAttentionMetadata& gdn_meta,
      const std::vector<PagedKvCache>& attn_kv,
      const std::vector<GdnStateCache>& gdn_state,
      const Qwen3_5DenseWeights& weights, const HfConfig& config, vt::Queue& queue,
      Qwen3_5AuxTaps* aux_out,
      const std::vector<int32_t>& logits_indices = {});

  // Dense single-sequence reference forward (M0.9 anchor). Runs the whole model
  // for a single non-paged sequence and returns logits [T, vocab] f32 (T =
  // token_ids.size()). Retained as the paged==dense parity reference.
  static std::vector<float> ForwardDense(const std::vector<int32_t>& token_ids,
                                         const std::vector<int32_t>& positions,
                                         const Qwen3_5DenseWeights& weights,
                                         const HfConfig& config,
                                         vt::Queue& queue);

  // MODEL-KEV Phase 5a: single-sequence hidden-state forward. Mirrors
  // ForwardDense but stops after the final GemmaRMSNorm (no lm_head) and
  // returns [T, H] f32 hidden states for the kev PointerHead readout.
  static std::vector<float> ForwardDenseHidden(
      const std::vector<int32_t>& token_ids,
      const std::vector<int32_t>& positions,
      const Qwen3_5DenseWeights& weights,
      const HfConfig& config,
      vt::Queue& queue);

  // MODEL-NIMBLE: single-sequence forward that returns the LAST position's
  // logits only, [vocab] f32. Mirrors HF `model(..., logits_to_keep=1)`, which
  // is what the Nimble reference scorer calls: the final-norm row T-1 is
  // gathered on-device (the same GatherRows the paged Forward uses for
  // logits_indices) and only that row goes through DenseLogitsF32D. ForwardDense
  // would materialize [T, vocab] f32, about 1 GB at T=1000 on a 248320 vocab.
  static std::vector<float> ForwardDenseLastLogits(
      const std::vector<int32_t>& token_ids,
      const std::vector<int32_t>& positions,
      const Qwen3_5DenseWeights& weights,
      const HfConfig& config,
      vt::Queue& queue);
};

// M3-b — single-image, single-sequence GREEDY image->text generation on the
// Qwen3.6-27B GDN-hybrid backbone (Qwen3_5ForConditionalGeneration). The forked
// VL forward, gated on multimodal input so a text-only request is byte-identical
// (it reuses Qwen3_5DenseModel::Forward's DenseForwardLayers machinery with two
// mm-only points active): (1) inputs_embeds — embed(prompt_ids) then scatter the
// vision tower's merger output [N,H] into the image_token rows; (2) MRoPE — the
// 16 full-attention layers' cos|sin cache is built from Qwen3VLGetRopeIndex
// positions [3,T] + config.mrope_section ([11,11,10]) interleaved, instead of the
// 1-D RoPE cache. NO DeepStack (empty deepstack_visual_indexes on the 27B). GDN
// (linear_attention) layers carry no rope. Runs the whole model on `queue`'s
// device (CUDA), allocating its own paged KV (full-attn) + GDN recurrent state,
// and greedy-decodes up to max_new_tokens (stops on eos_token_id).
//
// prompt_ids : placeholder-expanded model input ids (image_token_id repeated N
//              times at the image span).
// mm_main    : the tower merger output [N, H] (== hidden_size == out_hidden 5120),
//              host f32; scattered (bf16-rounded) into the image rows.
// grid_thw   : the LLM-grid source (t,h,w) for get_rope_index.
std::vector<int32_t> Qwen3_5VLGenerateGreedy(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::array<int64_t, 3>& grid_thw, int32_t image_token_id,
    int32_t eos_token_id, const Qwen3_5DenseWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens);

// M3d — single-VIDEO, single-sequence GREEDY video->text generation on the
// Qwen3.6-27B GDN-hybrid backbone. Mirrors Qwen3_5VLGenerateGreedy through the
// shared GDN-hybrid VL core; the two video differences (as in the 4B M3c split)
// are (a) the merge mask is on video_token_id across all frames and (b) the MRoPE
// prefill positions come from Qwen3VLGetRopeIndexVideo (per-frame, timestamp-
// interleaved scan). NO DeepStack (empty deepstack_visual_indexes on the 27B).
//
// prompt_ids  : placeholder-expanded model input ids (per-frame timestamp tokens
//               + vision_start + video_token*Nf + vision_end, x grid_t).
// mm_main     : the tower merger output [N, H] over ALL video tokens
//               (N = grid_t*(h/merge)*(w/merge)); scattered into the video rows.
// grid_thw    : the video (t,h,w) patch grid for get_rope_index.
std::vector<int32_t> Qwen3_5VLGenerateGreedyVideo(
    const std::vector<int32_t>& prompt_ids, const std::vector<float>& mm_main,
    const std::array<int64_t, 3>& grid_thw, int32_t video_token_id,
    int32_t vision_start_token_id, int32_t vision_end_token_id,
    int32_t eos_token_id, const Qwen3_5DenseWeights& weights,
    const HfConfig& config, vt::Queue& queue, int max_new_tokens);

// Replay one dense decoder layer from a captured combined residual stream.
// This is the dense sibling of Qwen3_5ReplayLayer and is the focused parity
// seam for the real 27B resident W4A4 layer goldens.
std::vector<float> Qwen3_5ReplayDenseLayer(
    const Qwen3_5DenseLayerWeights& layer, const HfConfig& config,
    const std::vector<float>& hidden_in, const std::vector<int32_t>& positions,
    int64_t seqlen, vt::Queue& queue);

// Qwen3.5 DENSE decode CUDA-graph driver — the dense sibling of
// Qwen3_5DecodeGraph (qwen3_5.h). It serves both the 27B NVFP4 and ordinary
// BF16 dense checkpoints. Captures the PURE-DECODE dense forward once per PADDED batch size
// (kDecodeGraphSizes {1,2,4,8,16,32,64}, capped at max_num_reqs) and replays it
// per token, removing the per-step host tax (the ~62k cudaMalloc/step + kernel
// launch overhead) that the eager dense decode paid every step. The embedding is
// kept outside the capture region and run per step into a persistent hidden
// buffer; every per-call scratch is pool-backed / resident (DevicePool + the fp4
// GEMM StreamScratch pools + resident weights) so the captured region does zero
// cudaMalloc after a cold pre-warm. Bit-identical to Qwen3_5DenseModel::Forward
// for the same inputs/caches (same op sequence: DenseEmbedInto + DenseForward
// Layers). VLLM_CPP_CUDAGRAPH=0 disables capture (always eager). The 35B MoE
// graph (Qwen3_5DecodeGraph) is a separate driver and is untouched.
class Qwen3_5DenseDecodeGraph {
 public:
  // max_num_reqs == the runner's max_num_seqs (== the GDN state-cache slot count);
  // the padded decode batch is capped at this so it never exceeds the mamba/GDN
  // state cache (mirrors vLLM's decode cudagraph dispatcher, compilation.py).
  Qwen3_5DenseDecodeGraph(const Qwen3_5DenseWeights& weights, const HfConfig& config,
                          vt::Queue queue, int64_t max_num_reqs);
  ~Qwen3_5DenseDecodeGraph();
  Qwen3_5DenseDecodeGraph(const Qwen3_5DenseDecodeGraph&) = delete;
  Qwen3_5DenseDecodeGraph& operator=(const Qwen3_5DenseDecodeGraph&) = delete;

  // One PURE-DECODE step. Returns the [B, vocab] f32 logits as a DEVICE-resident
  // ForwardLogits (the captured graph's output stays on device — a view over the
  // slot's persistent logits buffer; the eager fallback owns a pool block), fed
  // straight to the sampler with NO full-logits D2H. Bit-identical to
  // Qwen3_5DenseModel::Forward for the same inputs/caches. The caller must only
  // route pure-decode batches here (all query_len==1, no prefill).
  ForwardLogits Step(const std::vector<int32_t>& token_ids,
                     const std::vector<int32_t>& positions,
                     const v1::CommonAttentionMetadata& attn_meta,
                     const v1::GDNAttentionMetadata& gdn_meta,
                     const std::vector<PagedKvCache>& attn_kv,
                     const std::vector<GdnStateCache>& gdn_state,
                     // SPEC-DSPARK W8 (#442): non-null captures the DFlash/DSpark
                     // aux hidden taps into this slot's PERSISTENT [S, H*taps]
                     // buffer and points `aux_out->tensor` at it. The view is
                     // valid until this slot's next replay, the same contract the
                     // returned logits already carry. Null keeps the pure-decode
                     // behavior byte-identical.
                     Qwen3_5AuxTaps* aux_out = nullptr);

  // Diagnostics (A/B + tests): is a graph currently captured, and how many
  // replays have run since the last (re)capture.
  bool captured() const;
  int64_t replay_count() const;

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace vllm
