// DFlash draft model (`DFlashDraftModel` -> DFlashQwen3ForCausalLM /
// DFlashQwen3Model) for block-diffusion speculative decoding. SPEC-DFLASH D2,
// DF-DRAFT-MODEL.
//
// Ported from vllm/model_executor/models/qwen3_dflash.py @ 555967922
// (vLLM 0.26.0.dev0): DFlashQwen3Attention (:149-263), DFlashQwen3DecoderLayer
// (:266-342), DFlashQwen3Model (:345-661), DFlashQwen3ForCausalLM (:664-855).
//
// The draft is a PLAIN Qwen3-dense decoder (NOT the Qwen3.5 gated full-attention):
// per-layer q/k/v/o proj + per-head q_norm/k_norm RMSNorm, standard (non-gemma)
// input/post RMSNorm, NeoX RoPE (theta 1e7), SwiGLU gate/up/down MLP, NO GDN, NO
// MoE, NO attention output gate. The z-lab/Qwen3.6-27B-DFlash card: 5 layers
// (config.layer_types = 4x sliding_attention window 2048 + 1x full_attention),
// hidden 5120, 32 q-heads / 8 kv-heads / head_dim 128, vocab 248320,
// mask_token_id 248070, target_layer_ids [1,16,31,46,61] (5 aux taps, fc input
// 5120x5 -> 5120).
//
// The ONE genuinely new brick is the attention: the FULL-attention layer attends
// BIDIRECTIONALLY (non-causal) across the uniform (1+k) query block; the SWA
// layers are causal within their window. This routes through the new
// vt::DFlashBlockAttention primitive (ops.h) rather than the causal
// vt::PagedAttention every other model uses. Per-layer causality is resolved from
// config exactly as vLLM _resolve_layer_attention (:109-169) + _dflash_layer_causal
// (:58-67), both @ vllm-project/vllm#52816 head
// `19c9351904df4c63042671bc67a866ca48dc7d6f`: a DECLARED top-level `is_causal`
// wins, else a declared `dflash_config.causal`, else a layer is causal iff its
// DECLARED `layer_types[i]` is `sliding_attention`. That last arm reads the
// declared value and not the resolved layer type, so `dflash_config.use_swa` --
// which forces SWA onto every layer -- moves the WINDOW and never the causality
// (#1366). Both explicit arms test presence and then coerce, as upstream's
// `bool(...)` does, so a checkpoint spelling the value `0` is honoured. The
// top-level arm is BEYOND-PIN (SPEC-DFLASH2 W1, #1314); no DFlash1 checkpoint
// declares the key, so their resolution is unchanged by it. See
// ResolveQwen3DFlashAttnModes.
//
// Context-KV precompute (qwen3_dflash.py:548-619 precompute_and_store_context_kv)
// and prepare_dflash_inputs are D3 (DF-DRAFT-KV-PREP); this header/cpp owns the
// draft model, the fc combine, the mask embedding, the block forward, and the
// loader. The D2 isolation gate exercises the CONTEXT-FREE block forward (the
// query block attends only to itself) which fully exercises the new non-causal
// primitive + the per-layer routing + fc + mask-embed + logits.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3.h"             // Qwen3DenseMlpWeights-style ops
#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor, TensorResolver
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"
#include "vt/tensor.h"

namespace vllm {

class SafetensorsFile;

// Per-layer attention mode resolved from the draft config, mirroring
// _resolve_layer_attention (qwen3_dflash.py:86-146). `causal` is false for a
// full-attention layer (BIDIRECTIONAL in-block) and true for a sliding-window
// layer (causal within `sliding_window`).
struct Qwen3DFlashLayerAttnMode {
  bool causal = false;
  int64_t sliding_window = 0;  // >0 for SWA layers; 0 for full layers
};

// SPEC-DFLASH2 W2 (#1314): the grouped dynamic depthwise convolution that wraps
// ONE sublayer of a DFlash2 draft block. EMPTY on a DFlash1 draft, which is what
// `Qwen3DFlashWeights::IsDflash2` reads.
//
// BEYOND-PIN, from `DFlashGroupedConv` (vllm/model_executor/models/qwen3_dflash2.py
// @ vllm-project/vllm#52816 head `19c9351904df4c63042671bc67a866ca48dc7d6f`).
//
// Both tensors are named exactly as the published checkpoint stores them
// (`z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`,
// safetensors header read 2026-08-19):
//   layers.N.attention_conv.base_kernel               bf16 (2, 2, 5120)
//   layers.N.attention_conv.kernel_projection.weight  bf16 (1280, 5120)
// and the same pair under `mlp_conv`.
//
// `base_kernel` dim 0 is the SIDE -- 0 = `prepare` (before the sublayer), 1 =
// `finish` (after it) -- and NOT a tap. On this checkpoint `taps` is also 2, so
// the two axes are indistinguishable by shape and only the port note separates
// them. `kernel_projection` maps hidden -> `2 * taps * num_groups` (1280 =
// 2*2*320 at hidden 5120 / conv_group_size 16), i.e. ONE projection of the
// sublayer input carrying BOTH sides' deltas.
struct Qwen3DFlashConvWeights {
  OwnedTensor base_kernel;        // bf16 [2, taps, H]; dim 0 is the SIDE
  OwnedTensor kernel_projection;  // bf16 raw-NK [2*taps*num_groups, H], nk
  bool Empty() const { return base_kernel.bytes.empty(); }
};

// SPEC-DFLASH2 W3 (#1314): the CANDIDATE SELECTOR of a DFlash2 draft. EMPTY on a
// DFlash1 draft, which is what `Qwen3DFlashWeights::IsDflash2` reads.
//
// BEYOND-PIN, from `CandidateSelector` and `DFlash2Qwen3ForCausalLM`
// (vllm/model_executor/models/qwen3_dflash2.py:231-356 @
// vllm-project/vllm#52816 head `66e5414c6d75a8529473d977f7458c140bbab8a0`). The
// PR head MOVED from `19c9351904df4c63042671bc67a866ca48dc7d6f` on 2026-08-19
// (#1404). `_score_edges`, `CandidateSelector` and every line of
// `compute_candidates` except the LM-head guard are BYTE-IDENTICAL at the two
// heads; the two things that changed are ported or recorded on
// `Dflash2SelectorWeights::kNonPortSetModelTag` below.
//
// The three tensors are named exactly as the published checkpoint stores them
// (`z-lab/Qwen3.8-27B-DFlash2` @ `50307d4c4cde6860d4eee73e2547cd786fe8e8a4`,
// safetensors header read 2026-08-19):
//   candidate_selector.hidden_projection.weight  bf16 (256, 5120)
//   candidate_selector.predecessor_codebook      bf16 (248320, 256)   127 MB
//   candidate_selector.successor_codebook        bf16 (248320, 256)   127 MB
//
// The two codebooks are ~254 MB resident that the DFlash1 lane never allocates
// (`## Risks/decisions` D5). Upstream stores them bf16 and so do we; whether a
// narrower dtype would serve is NOT decided by this row.
struct Dflash2SelectorWeights {
  OwnedTensor hidden_projection;     // bf16 raw-NK [rank, H], nk
  OwnedTensor predecessor_codebook;  // bf16 [vocab, rank]
  OwnedTensor successor_codebook;    // bf16 [vocab, rank]
  int64_t rank = 0;    // dflash_config.selector_rank (256 on both drafts)
  int64_t top_k = 0;   // dflash_config.selector_top_k (16 on both drafts)
  // The two OUTPUT SCALARS, applied to the candidate VALUES in
  // `compute_candidates` BEFORE the selector scores them
  // (`DFlash2Qwen3ForCausalLM.compute_candidates` @ that head). A wrong value
  // reorders the top-K and moves acceptance without raising anything, which is
  // this row's signature invisible defect, so `## Risks/decisions` D9 requires
  // them gated against a checkpoint that SETS them:
  // `z-lab/Muse-Glimmer-30B-DFlash2` ships 0.19611613513818404 and 20.0, while
  // `z-lab/Qwen3.8-27B-DFlash2` ships neither and takes both defaults (#1327).
  float output_multiplier = 1.0f;
  // `float(dflash_config.get("final_logit_softcapping") or 0.0)`, then DISABLED
  // when not > 0 — upstream's own `softcap if softcap > 0 else None`. So 0 here
  // means "no softcap" and never "cap at zero".
  float final_logit_softcapping = 0.0f;
  bool Empty() const { return predecessor_codebook.bytes.empty(); }
  // A DELIBERATE NON-PORT, recorded rather than skipped in silence. Upstream
  // wraps the selector's construction in `set_model_tag("dflash2_candidate_selector")`
  // because `CandidateSelector` carries its own `@support_torch_compile` and is
  // built while the draft's model tag is still active, so without a tag of its
  // own the two share a torch.compile cache namespace and the selector loads the
  // draft's graph. It is the only behavioural change #52816 made to this file
  // between the two heads. This engine has no torch.compile and no compile
  // cache, so there is nothing for a tag to disambiguate.
  static constexpr const char* kNonPortSetModelTag = "dflash2_candidate_selector";
};

// One DFlash draft decoder layer: input/post standard RMSNorm + plain Qwen3
// attention (merged qkv, per-head q/k norm, NeoX RoPE) + SwiGLU MLP. Weights are
// kept in the on-disk torch-Linear [N=out,K=in] orientation (nk=true) for
// vt::MatmulBT, exactly like the Qwen3-dense loader.
struct Qwen3DFlashLayerWeights {
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  OwnedTensor qkv_proj;  // bf16 raw-NK [Hq*Dh + 2*Hkv*Dh, H] (rows q|k|v), nk
  OwnedTensor o_proj;    // bf16 raw-NK [H, Hq*Dh], nk
  OwnedTensor q_norm;    // bf16 [head_dim] per-head RMSNorm
  OwnedTensor k_norm;    // bf16 [head_dim]
  OwnedTensor gate_up_proj;  // bf16 raw-NK [2*I, H] (rows gate|up), nk
  OwnedTensor down_proj;     // bf16 raw-NK [H, I], nk
  Qwen3DFlashLayerAttnMode attn_mode;
  // SPEC-DFLASH2 W2 (#1314): the two grouped convolutions of a DFlash2 block,
  // wrapping the attention and the MLP sublayer respectively. Both EMPTY on a
  // DFlash1 draft, and the forward then runs byte-for-byte as before.
  Qwen3DFlashConvWeights attention_conv;
  Qwen3DFlashConvWeights mlp_conv;
  // MODEL-DFLASH2-EXL3 (#2495 item 7): the SEVEN per-layer trellises of an EXL3
  // draft. All EMPTY on a bf16 or GGUF draft, which is what leaves every
  // `MakeLinearMethod` / `MakeMlpGateUpMethod` call in the three forward bodies
  // bound to the unquantized method it was bound to before.
  //
  // SEVEN AND NOT FIVE, because a trellis does not row-stack. `qkv_proj` and
  // `gate_up_proj` above are MERGED owners built by concatenating output rows;
  // joining two trellises on the output dim INTERLEAVES per input tile and is a
  // real transform, deferred by `specs/quant-exl3-shared.md` `## Owed`. The
  // artifact agrees: `Mia-AiLab/Qwen3.8-27B-DFlash2-EXL3-5.0bpw` ships six
  // independently fitted trellises per layer with six independently fitted
  // `svh` vectors and no merged operand anywhere in the file.
  Exl3Weight q_proj_exl3;     // [K=H, N=Hq*Dh]
  Exl3Weight k_proj_exl3;     // [K=H, N=Hkv*Dh]
  Exl3Weight v_proj_exl3;     // [K=H, N=Hkv*Dh]
  Exl3Weight o_proj_exl3;     // [K=Hq*Dh, N=H]
  Exl3Weight gate_proj_exl3;  // [K=H, N=I]
  Exl3Weight up_proj_exl3;    // [K=H, N=I]
  Exl3Weight down_proj_exl3;  // [K=I, N=H]
  // MODEL-DFLASH2-NVFP4 (#2758): the SEVEN per-layer NVFP4 weights of a
  // ModelOpt-quantized draft, in the same shape and for the same reason as the
  // seven trellises above. All EMPTY on a bf16, GGUF or EXL3 draft.
  //
  // SEVEN AND NOT FIVE, and here the argument is the producer's rather than the
  // format's. NVFP4 CAN row-stack -- `dense_loaders::LoadMergedCtNvfp4W4A16`
  // does it for compressed-tensors, where a merged operand is what the producer
  // emits -- but it costs a `max()` over the shards' DIVISORS and a rescale of
  // every group scale, and `maurienne-ai/Qwen3.8-27B-DFlash2-NVFP4-RTNcal`
  // @ `bd7a9342` ships q, k and v as three modules with three independently
  // fitted `weight_scale_2` scalars. The target's own ModelOpt path in this tree
  // reads q/k/v/o unmerged for the same reason (`LoadDenseAttn`'s
  // `load_projection`, qwen3_5_dense_weights.cpp), so this is the tree's
  // established NVFP4 ownership shape and not a deferral.
  Nvfp4Weight q_proj_fp4;     // [N=Hq*Dh, K=H]
  Nvfp4Weight k_proj_fp4;     // [N=Hkv*Dh, K=H]
  Nvfp4Weight v_proj_fp4;     // [N=Hkv*Dh, K=H]
  Nvfp4Weight o_proj_fp4;     // [N=H, K=Hq*Dh]
  Nvfp4Weight gate_proj_fp4;  // [N=I, K=H]
  Nvfp4Weight up_proj_fp4;    // [N=I, K=H]
  Nvfp4Weight down_proj_fp4;  // [N=H, K=I]
};

// Whole DFlash draft weights. The draft owns the fc aux-combine, the
// hidden_norm (applied to combined target features before the context-KV proj,
// D3), the final norm, and an optional dedicated mask embedding.
//
// It does NOT own its embedding table, and this comment used to say it did
// (#1946). The z-lab checkpoint ships none: the draft runs the TARGET's table,
// which the loader read into `embed_tokens` here and which
// `BindDflashDraftSharedEmbed` now rebinds onto the target's own tensor, exactly
// as the MTP head has always shared it. The head is still the draft's own copy,
// and its dedup is owed (see the spec's `## Owed`).
struct Qwen3DFlashWeights {
  OwnedTensor embed_tokens;  // bf16 [vocab, H] (embed lookup)
  // #1946: the TARGET's table, borrowed, once `BindDflashDraftSharedEmbed` has
  // rebound it — the mirror of upstream's `del draft_inner.embed_tokens;
  // draft_inner.embed_tokens = target_embed`
  // (vllm/v1/worker/gpu/spec_decode/dflash/utils.py:64-74 @ b389ac29465b33f9e9c534df221ea3c129e9793f).
  //
  // WHY A POINTER AND NOT A COPY. `ResidentWeight` caches the host->device
  // upload on the OwnedTensor itself (`if (!w.d_dev)`), so two OwnedTensors over
  // the same bytes are two device allocations however the HOST bytes are shared
  // — which is what W9 (#1849) fixed and what it could not fix. One tensor is
  // one `d_dev`, allocated by whichever side forwards first and never
  // reallocated, so the address cannot depend on load order and stays fixed for
  // the life of any graph that reads it.
  //
  // BORROWED from the target LoadedModel, which outlives the draft (LoadedEngine
  // declares `model_` ahead of `dflash_draft_` for exactly this). Null on a
  // draft that was never rebound: an unbound draft still runs its own
  // `embed_tokens`, which is what every pre-#1946 path did.
  //
  // ONCE REBOUND, `embed_tokens` IS EMPTY, and reading it instead of
  // `EmbedTable()` is refused by `ResidentWeight` by name (dense_attn_block.h).
  // That refusal is a check #1953 added; it is NOT something the
  // empty tensor gets for free. `vt::Embedding` validates ranks, shapes, dtypes,
  // contiguity and device and never the byte length, and the shape it sees comes
  // from the caller, so before that check an empty table produced a null host
  // alias or a zero-byte device allocation read as [vocab, H].
  const OwnedTensor* shared_embed_tokens = nullptr;
  // The table the draft's four embed lookups actually gather from. The ONE
  // accessor, so a rebind cannot reach three of the four sites.
  const OwnedTensor& EmbedTable() const {
    return shared_embed_tokens != nullptr ? *shared_embed_tokens : embed_tokens;
  }
  // QWEN38-DFLASH2-STAGE-C W3: the head twin of the embed rebind above. The
  // target's own lm_head owner, borrowed, once `BindDflashDraftSharedHead` has
  // rebound it. Null on a draft that was never rebound: the draft then runs its
  // own `lm_head`, exactly as every pre-W3 path did.
  const OwnedTensor* shared_lm_head = nullptr;
  // The table the draft's block-logits GEMM actually reads. The ONE accessor,
  // so a rebind cannot reach a second copy through a stale field.
  const OwnedTensor& LmHead() const {
    return shared_lm_head != nullptr ? *shared_lm_head : lm_head;
  }
  OwnedTensor fc;            // bf16 raw-NK [H, H*num_taps], nk (combine_hidden_states)
  // MODEL-DFLASH2-EXL3 (#2495 item 7): the aux-combine trellis, and the ONE
  // field that says whether this draft is an EXL3 draft at all. Empty on every
  // bf16 and GGUF draft.
  Exl3Weight fc_exl3;        // [K=H*num_taps, N=H]
  OwnedTensor hidden_norm;   // bf16 [H] (normed before context-KV proj, D3)
  OwnedTensor final_norm;    // bf16 [H] (model.norm)
  OwnedTensor lm_head;       // bf16 raw-NK [draft_vocab, H], nk
  // SPEC-DFLASH2-QUANT-LMHEAD (#1628): the SHARED head when the target stores it
  // as ModelOpt/compressed-tensors NVFP4. EXACTLY ONE of `lm_head` and
  // `lm_head_fp4` is populated, which is the one-owner rule
  // `Qwen3_5DenseWeights` already applies to the target's own head
  // (`LoadDenseLmHead`, qwen3_5_dense.h). The draft's logits GEMM routes on it,
  // so a packed head is COMPUTED WITH rather than widened at load: the DFlash2
  // selector's whole input is the target head's exact top-K, and widening the
  // head is the case `RefuseQuantizedDflash2LmHead` refuses.
  Nvfp4Weight lm_head_fp4;   // NVFP4 [N=draft_vocab, K=H] (else empty)
  // MODEL-QWEN35-EXL3-HEAD (#2495 item 6): the SHARED head when the target
  // stores it as an exllamav3 trellis. THE THIRD owner of the same
  // exactly-one-of rule above, and it is packed for the same reason
  // `lm_head_fp4` is: `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships
  // `lm_head.{trellis,suh,svh,mul1}` and NO `lm_head.weight` at all
  // (k=5120, n=248320, bits 6, codebook 2 — read from its safetensors header by
  // range request, #2569), so there is nothing to read as dense floats and a
  // dequantized copy would be a 2.543 GB buffer of a tensor the target itself
  // never widens. Upstream needs no branch here because its head is an
  // `nn.Module`: `compute_candidates` -> `LogitsProcessor.get_top_k_tokens` ->
  // `_apply_head` -> `lm_head.quant_method.apply` IS the target's own logits
  // path (logits_processor.py:241-286,:132-142 @ the merged
  // vllm-project/vllm#52816 head `b389ac29`).
  Exl3Weight lm_head_exl3;   // trellis [K=H, N=draft_vocab] (else empty)
  // Optional dedicated mask embedding [H] substituted at mask_token_id
  // (has_separate_mask_embedding). Empty for the z-lab 27B (in-vocab mask token).
  OwnedTensor mask_embedding;
  std::vector<Qwen3DFlashLayerWeights> layers;
  int64_t num_taps = 0;         // len(target_layer_ids); fc input = H*num_taps
  int32_t mask_token_id = -1;   // dflash_config.mask_token_id (248070 for 27B)
  int64_t draft_vocab_size = 0;
  // SPEC-DFLASH2 W2 (#1314): the conv geometry. `conv_taps` is
  // `dflash_config.conv_kernel_size` and is 0 on a DFlash1 draft, which is what
  // makes it the DFlash2 discriminator here -- a DFlash1 checkpoint declares
  // none of these keys and carries no conv tensor.
  int64_t conv_taps = 0;        // dflash_config.conv_kernel_size (2 on both drafts)
  int64_t conv_group_size = 0;  // dflash_config.conv_group_size (16 on both drafts)
  // The QUERY block the conv masks its taps against: `1 + num_speculative_tokens`,
  // NOT `dflash_config.block_size`. Upstream sizes it from the speculative config
  // (`DFlash2Qwen3DecoderLayer.__init__` @ vllm-project/vllm#52816 head
  // `66e5414c6d75a8529473d977f7458c140bbab8a0`) and the checkpoint key only ever
  // supplies that value's DEFAULT, through `k`.
  //
  // `LoadQwen3DFlash` DOES NOT FILL IT. Whoever knows the resolved `k` is the
  // only writer -- `LoadDflashDraft` in production -- and until then it is 0,
  // which every DFlash2 forward refuses by name. W3 and earlier seeded it from
  // the checkpoint key so a direct caller had a usable value, and that is what
  // made the loader's own assignment ungateable: deleting it left a plausible
  // block behind and no gate could see the difference (spec `## Owed` O5,
  // mutation-proven green by W2). SPEC-DFLASH2 W4 (#1314) removes the seed.
  int64_t conv_block_size = 0;
  // SPEC-DFLASH2 W3 (#1314): TRUE when the SHARED lm_head above was produced by
  // DEQUANTIZING a quantized target tensor rather than read as dense floats.
  // Only the GGUF target arm can set it (`LoadGgufSharedEmbedAndHeadBf16`
  // dequantizes a q6_K/NVFP4 `output.weight` to bf16); the safetensors arm
  // refuses a non-BF16 head one layer up in `LoadNamedBf16`. It exists because
  // the DFlash2 selector's whole input is the target head's EXACT top-K, which a
  // dequantized head does not produce -- the case upstream's LM-head guard
  // refuses by name. See `RefuseQuantizedDflash2LmHead`
  // (qwen3_dflash2.h). Read only by that guard; the DFlash1 lane is unaffected,
  // which is why the flag is recorded rather than refused at load.
  bool lm_head_dequantized = false;
  // SPEC-DFLASH2 W3 (#1314): the candidate selector. EMPTY on a DFlash1 draft.
  // It is loaded whenever `IsDflash2()`, because a DFlash2 checkpoint that
  // carried the conv and no selector is not a shape this port knows how to run.
  Dflash2SelectorWeights candidate_selector;
  bool IsDflash2() const { return conv_taps > 0; }
  // MODEL-DFLASH2-EXL3 (#2495 item 7): THE arm discriminator, and there is only
  // ONE of it. Every projection in all three forward bodies asks this same
  // question, exactly as they all ask `IsDflash2()`.
  //
  // KEYED ON `fc_exl3` ALONE, and that is the point rather than a convenience.
  // A per-field probe -- "is THIS projection's trellis populated?" -- lets a
  // half-populated container read as EXL3 at one call site and as bf16 at the
  // next, which loads, runs, and drafts from a model that is neither. On a
  // lossless verify that is invisible: the emitted tokens stay the target's and
  // only acceptance falls. The loader closes the other half of the door by
  // CONTROL FLOW rather than by a second check: once `fc` classifies the
  // checkpoint the per-layer rung is unconditional, so a layer missing a
  // trellis throws `tensor not found` at load. `IsExl3()` true therefore
  // implies every forward site has one to read.
  bool IsExl3() const { return !fc_exl3.Empty(); }
  // MODEL-DFLASH2-NVFP4 (#2758): the same one-field question for the NVFP4 arm,
  // and it CANNOT be asked of `fc`. `fc` is the one projection every DFlash
  // draft carries, which is what makes it the EXL3 classifier -- and it is
  // exactly the projection the published NVFP4 drafter EXCLUDES from
  // quantization, so on that artifact `fc` is BF16 and says nothing. The
  // question is therefore asked of a projection the artifact really quantizes.
  //
  // DERIVED AND NOT STORED, so it cannot disagree with the owners it describes.
  // The loader reads all seven per-layer weights unconditionally once the
  // draft's `quantization_config` classified the checkpoint, and refuses by name
  // when a module the declaration says is quantized ships no NVFP4 operands --
  // so a populated `layers[0].q_proj_fp4` means all of them are populated.
  bool IsNvfp4() const {
    return !layers.empty() && !layers.front().q_proj_fp4.Empty();
  }
};

// Load the z-lab DFlash draft checkpoint. The on-disk names follow vLLM's
// DFlashQwen3Model.load_weights + hf_to_vllm_mapper (qwen3_dflash.py:347-356,
// 772-855): per layer `model.layers.N.self_attn.{q,k,v,o}_proj.weight`,
// `.self_attn.{q,k}_norm.weight`, `.mlp.{gate,up,down}_proj.weight`,
// `.{input_layernorm,post_attention_layernorm}.weight`; top-level
// `model.embed_tokens.weight`, `model.fc.weight`, `model.hidden_norm.weight`,
// `model.norm.weight`, `lm_head.weight`. q/k/v are concatenated into one qkv_proj
// and gate/up into one gate_up_proj (the vLLM stacked mapping). All draft tensors
// are BF16. `num_taps`/`mask_token_id`/`draft_vocab_size` come from the resolved
// draft config.
//
// MODEL-DFLASH2-EXL3 (#2495 item 7): `has` is the tensor-PRESENCE predicate the
// EXL3 rung needs, because upstream's own `Linear.is_exl3_storage` asks for
// `{key}.trellis` TOGETHER WITH `{key}.suh` and `{key}.svh`
// (`exllamav3/modules/linear.py:385-389`) and `TensorResolver` can only throw.
// An EMPTY `has` -- the default -- answers "this container holds no EXL3
// module", which is what keeps the GGUF draft and the DSpark backbone
// byte-unchanged: neither format can carry a trellis, and neither should grow a
// probe for one. The shards overload below builds `has` with the SAME
// `model.`-prefixed fallback its resolver uses, so the two cannot disagree
// about whether a tensor exists.
Qwen3DFlashWeights LoadQwen3DFlash(const TensorResolver& get, const HfConfig& config,
                                   int64_t num_taps, int32_t mask_token_id,
                                   const std::function<bool(const std::string&)>& has = {});
Qwen3DFlashWeights LoadQwen3DFlash(const std::vector<SafetensorsFile>& shards,
                                   const HfConfig& config, int64_t num_taps,
                                   int32_t mask_token_id);

// SPEC-DFLASH2-QUANT-LMHEAD (#1628). Fill the draft's SHARED `lm_head` from the
// TARGET's safetensors shards, taking the SAME arm the target's own loader takes.
//
// The DFlash/DSpark draft owns no head: it runs the TARGET's over its own hidden
// states. Through W5 this read was a single `LoadNamedBf16("lm_head.weight")`
// inside `SharedHeadSource` (model_loader.cpp), so it refused ANY head not
// stored as dense bf16 — including the ModelOpt NVFP4 head of
// `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`, which is the only checkpoint the
// BENCH-QWEN38-27B-SOTA campaign's speculative arm has (#1628). It refused on
// the STORED DTYPE, which cannot separate the two states `## Risks/decisions`
// D12 is about: a head DEQUANTIZED into something the target does not compute
// with, and a head kept PACKED and computed with natively.
//
// So the routing decision is `LoadDenseLmHead`'s (qwen3_5_dense.h), asked once
// here for the draft exactly as the target asks it for itself: a
// ModelOpt/compressed-tensors NVFP4 head lands PACKED in `head_fp4`, every other
// storage form lands as raw-NK bf16 in `head_bf16`. Upstream reaches the same
// place by construction — at the MERGED #52816 head `b389ac29` `compute_candidates`
// carries no quant-method check at all and goes through
// `LogitsProcessor.get_top_k_tokens` -> `_apply_head` ->
// `lm_head.quant_method.apply` (logits_processor.py:241-286,132-142), which IS
// the target's own logits path.
//
// MODEL-QWEN35-EXL3-HEAD (#2495 item 6) adds the THIRD arm and the fourth
// parameter: an exllamav3 trellis head lands PACKED in `head_exl3`. The
// parameter is REQUIRED and has no default, for the reason
// `SharedHeadSource::LoadInto` gives at length for the two it already has: a
// defaulted owner is what silently turns an arm off and leaves every gate green.
// `nullptr` means "this lane cannot compute with a trellis head", and the read
// then refuses the checkpoint by name rather than returning an empty head.
//
// EXACTLY ONE output is populated. Throws, naming the target shards AND the arms
// it tried, when no arm matches (#2569 — this function used to fall off the end
// of its bf16 loop and return silently, leaving the refusal to whichever caller
// happened to check emptiness); throws naming the arm when the head is NVFP4
// with an ACTIVATION scale in force (`VT_MODELOPT_W4A4=1`), because the
// fp4-activation GEMM is not the W4A16 dispatcher this draft's logits GEMM takes.
void LoadDflashSharedLmHead(const std::vector<SafetensorsFile>& shards,
                            OwnedTensor* head_bf16, Nvfp4Weight* head_fp4,
                            Exl3Weight* head_exl3);

// SPEC-DFLASH2 W9 (#1849). Fill the draft's SHARED embedding table from the
// TARGET's safetensors shards, borrow-first: a whole-range verbatim bf16 read
// the draft never mutates, so it takes the fail-closed direct-upload seam
// (`BorrowStTensorBytes`) and views the file mapping instead of holding a
// ~2.54 GB anonymous copy of bytes the target's own loader already borrows.
// Same lookup (exact name, first shard that has it), same "is not BF16"
// refusal, same EMPTY-on-absence contract as the loader-local `LoadNamedBf16`
// read it replaces; `nk` stays false (the [vocab, H] gather-table
// orientation). Exported so the borrow is gated at the exact function
// `SharedHeadSource::LoadInto` calls.
OwnedTensor LoadDflashSharedEmbedBf16(const std::vector<SafetensorsFile>& shards,
                                      const std::string& name);

// Resolve the per-layer attention modes from config.layer_types, the optional
// dflash_config overrides, and the optional top-level `is_causal`. Exposed for
// the loader + tests. Mirrors _resolve_layer_attention (qwen3_dflash.py:86-146 @
// the parity pin 555967922, :109-169 @ vllm-project/vllm#52816 head
// 19c9351904df4c63042671bc67a866ca48dc7d6f, identical body at both) and
// _dflash_layer_causal (:58-64 @ the pin, :58-67 @ that head, where the top-level
// `is_causal` arm is added), whose precedence the definition documents in full.
std::vector<Qwen3DFlashLayerAttnMode> ResolveQwen3DFlashAttnModes(const HfConfig& config);

// Build a DFlash draft HfConfig from the draft checkpoint's own config.json (the
// real nested {block_size, dflash_config:{mask_token_id,target_layer_ids},
// layer_types, ...}). Mirrors the D3 parity harness MakeConfig; kept manual
// rather than routed through LoadHfConfig so the DFlashDraftModel architecture
// and the nested dflash_config parse deterministically. It was `MakeDflashDraftConfig`
// in the loader's anonymous namespace until SPEC-DFLASH2 W1 moved it here.
//
// It lives beside ResolveQwen3DFlashAttnModes because the two are one decision
// split in half: this copies the named keys the draft resolution reads, and that
// function reads them. A key this builder drops is a key that resolution can
// never see, whatever the checkpoint declares -- which is why `is_causal` is
// carried here and gated in tests/vllm/v1/spec_decode/test_dflash_causality.cpp.
// The loader (src/vllm/entrypoints/model_loader.cpp, LoadDflashDraft) is the
// production caller.
HfConfig MakeQwen3DFlashDraftConfig(const nlohmann::json& c);

// D11 (Part A) — DEVICE-RESIDENT append-only draft-KV store. Opaque handle (defined
// in qwen3_dflash.cpp) holding, per request, the projected bf16 K/V for every draft
// layer as device chunks that persist ACROSS verify steps. It supersedes D9's
// host-vector PrecomputedContextKV + UploadContextKV re-upload: AppendContextKVDevice
// projects ONLY the newly-accepted rows on-device and keeps their bf16 K/V resident
// (no D->H download/H->D re-upload each step), and ForwardBlockLogitsWithDeviceKV runs
// the block forward straight off the device store. It is BIT-IDENTICAL to the host
// path by per-row projection independence (same PrecomputeContextKVDevice, same
// ascending-position append order, same IndexCopy concat), so tokens+acceptance are
// unchanged; it is the capture-ready substrate the Part B paged-attention kernel + the
// Part C static-shape CUDA-graph draft step will read (block-table over these slots).
struct DflashDeviceKVStore;

// The DFlash draft forward. D2 owns the CONTEXT-FREE block forward (the isolation
// gate): each request's uniform (1+k) query block attends only to itself through
// vt::DFlashBlockAttention (non-causal full / causal SWA per layer). D3 extends
// this to attend over pre-inserted context K/V.
class Qwen3DFlashModel {
 public:
  // SPEC-DFLASH2 W8 (#1837) — DEVICE handles out of a block forward. The
  // candidate selector's whole input is the block forward's logits and its
  // post-final-norm hidden, and through W7 both crossed the host boundary
  // (~17 MB/step at the published shapes) only to be re-uploaded. Upstream's
  // `_generate_draft` (dflash2/speculator.py @ b389ac2946) never downloads
  // either. The views are pool-backed: `keep_*` owns the storage when the call
  // allocated it (`DBuf::ReleaseShared`), and is EMPTY when the store's
  // persistent CUDA-graph output buffers own it — those live as long as the
  // store, and the selector consumes the views in the same step.
  struct DflashBlockDeviceOut {
    vt::Tensor logits;                  // [Tq, draft_vocab] f32, device
    vt::Tensor hidden;                  // [Tq, H] bf16, device (post-final-norm)
    std::shared_ptr<void> keep_logits;
    std::shared_ptr<void> keep_hidden;
  };

  // SPEC-DFLASH2 W8 (#1838) — a device-resident combined-features buffer (the
  // fc output), same pool-backed ownership shape as DflashBlockDeviceOut.
  struct DflashCombinedDevice {
    vt::Tensor tensor;                  // [T, H] bf16, device
    std::shared_ptr<void> keep;
  };

  // The fc aux-combine (combine_hidden_states, qwen3_dflash.py:750-770): a bias-
  // free Linear [H*num_taps]->[H] over the D1 multi-tap `[T, H*num_taps]` output
  // (column order = ascending target_layer_ids). Returns [T,H] bf16 as a device
  // buffer's host download for parity checks. This is the combined feature that
  // D3 will normalize (hidden_norm) and project into the context KV cache.
  // Since W8 it is a marshaling shell over CombineAuxFeaturesDevice — one GEMM
  // implementation — and stays bit-identical because bf16->f32->bf16 round
  // trips are exact.
  static std::vector<float> CombineAuxFeatures(const std::vector<float>& aux_features,
                                               int64_t T, const Qwen3DFlashWeights& weights,
                                               const HfConfig& config, vt::Queue& queue);

  // SPEC-DFLASH2 W8 (#1838): the SAME fc aux-combine, straight off the runner's
  // device-resident aux tap (`Qwen3_5AuxTaps.tensor`, bf16 [T, H*num_taps]) with
  // no host round trip: one vt::MatmulBT, no casts. Mirrors upstream's
  // `combine_hidden_states(torch.cat(aux_hidden_states, dim=-1))` consuming the
  // target's device tensors (dflash/speculator.py::propose @ b389ac2946). The
  // dtype is ASSERTED bf16 by name rather than assumed, which is what the old
  // host loop did silently.
  static DflashCombinedDevice CombineAuxFeaturesDevice(const vt::Tensor& aux_bf16,
                                                       const Qwen3DFlashWeights& weights,
                                                       const HfConfig& config,
                                                       vt::Queue& queue);

  // CONTEXT-FREE block forward -> [T, draft_vocab] f32 logits (the D2 isolation
  // gate). `input_ids` are the mask-block token ids (anchor + k mask_token_id per
  // request, T = num_reqs*(1+k)); `positions` the intra-context positions; `cu`
  // the per-request block boundaries (length num_reqs+1). Each block attends only
  // to itself: full-attention layers BIDIRECTIONAL, SWA layers causal-in-window.
  // Mask slots embed via embed_tokens[mask_token_id] (or the dedicated
  // mask_embedding when present), mirroring embed_input_ids (:432-438).
  // `per_layer_out` (if non-null) receives each decoder layer's post-MLP hidden
  // [T,H] (column-major flattened, one entry per layer); `final_out` (if non-null)
  // receives the post-final-norm hidden [T,H] fed to lm_head. Both are the exact
  // per-stage tensors the D2 draft-parity gate compares against the dumped vLLM
  // reference (scripts/spec/d2_dflash_draft_ref.py). Pass nullptr in production.
  static std::vector<float> ForwardBlockLogits(
      const std::vector<int32_t>& input_ids, const std::vector<int32_t>& positions,
      const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      std::vector<std::vector<float>>* per_layer_out = nullptr,
      std::vector<float>* final_out = nullptr);

  // ---- D3 (DF-DRAFT-KV-PREP) ----------------------------------------------
  //
  // Context-KV precompute (precompute_and_store_context_kv, qwen3_dflash.py:
  // 548-619). Projects the TARGET's per-token combined features (the fc output of
  // D1's multi-tap, or the last target hidden) to K/V for EVERY draft attention
  // layer, k-norms + NeoX-RoPE's K, and returns the per-layer context K/V that D3
  // pre-inserts into the draft's KV cache so the draft never re-runs context. The
  // path is: normed = RMSNorm(context_states, hidden_norm); for each layer l,
  // K_l = kv_proj_l.k(normed) (rows [q_size, q_size+kv_size) of qkv_proj), then
  // RMSNorm(K_l, k_norm_l) over head_dim, then RoPE(K_l, context_positions);
  // V_l = kv_proj_l.v(normed) (rows [q_size+kv_size, q_size+2*kv_size), no norm,
  // no RoPE). Mirrors _build_context_kv_buffers (:440-460) + _project_context_kv
  // (:505-534) + _normalize_context_k (:536-546) + the fused RoPE (:582-597); the
  // fused multi-layer GEMM/grouped-norm are perf fusions, so per-layer here is
  // numerically the same bf16 projection vLLM computes. `context_states` is
  // [num_ctx, H] f32; `context_positions` is [num_ctx] absolute positions.
  struct ContextKV {
    // Per draft attention layer: K normed+RoPE'd, V raw. Each is [num_ctx, Hkv,
    // Dh] flattened f32 (the paged-cache insert layout the D3 forward reads back).
    std::vector<std::vector<float>> k;
    std::vector<std::vector<float>> v;
    int64_t num_ctx = 0;
  };
  static ContextKV PrecomputeContextKV(const std::vector<float>& context_states,
                                       const std::vector<int32_t>& context_positions,
                                       const Qwen3DFlashWeights& weights,
                                       const HfConfig& config, vt::Queue& queue);

  // CONTEXT-AWARE block forward (the D3 isolation gate): the uniform (1+k) query
  // block of each request attends over its pre-inserted context K/V (from
  // PrecomputeContextKV) PLUS the in-block K/V. Full-attention layers are
  // BIDIRECTIONAL within the block and see all context; SWA layers are causal in
  // window. Context is laid out before the block per request, so the landed
  // vt::DFlashBlockAttention's offset-based mask is exact when context+query
  // positions are contiguous (the z-lab 27B SWA window 2048 >> any block so SWA
  // degenerates to plain causal-over-[context;block]); this reuses the D2
  // primitive UNCHANGED (no new kernel). `ctx_cu`/`cu` are the per-request
  // context/query boundaries (length num_reqs+1). Returns [Tq, draft_vocab] f32
  // logits over the query tokens (Tq = num_reqs*(1+k)). `per_layer_out`/`final_out`
  // capture the query-token per-stage hidden for the parity dump (as ForwardBlockLogits).
  static std::vector<float> ForwardBlockLogitsWithContext(
      const std::vector<float>& context_states, const std::vector<int32_t>& context_positions,
      const std::vector<int32_t>& ctx_cu, const std::vector<int32_t>& block_input_ids,
      const std::vector<int32_t>& block_positions, const std::vector<int32_t>& cu,
      const Qwen3DFlashWeights& weights, const HfConfig& config, vt::Queue& queue,
      std::vector<std::vector<float>>* per_layer_out = nullptr,
      std::vector<float>* final_out = nullptr);

  // ---- D9 (persistent paged draft-KV) -------------------------------------
  //
  // PERSISTENT context-KV store: the per-layer draft context K/V held as bf16 host
  // buffers ACROSS verify steps, so each step projects ONLY the newly-accepted rows
  // (AppendContextKVHost) instead of re-projecting the whole growing context every
  // step (the O(context^2) recompute PrecomputeContextKV does). This is the perf
  // form of vLLM's append-only paged draft KV cache (precompute_and_store_context_kv
  // writes only the step's NEW tokens, dflash/speculator.py:358,548-619). BIT-IDENTICAL
  // to the full recompute by per-row independence: hidden_norm is a per-row RMSNorm,
  // the KV GEMM an independent per-row dot product, k_norm per-row-per-head, and RoPE
  // per-row at that row's absolute position — so projecting a row once (when accepted)
  // yields the SAME bf16 bits as projecting it later inside a C-row batch. Tokens +
  // acceptance are therefore UNCHANGED; only the recompute cost drops O(ctx^2) -> O(ctx).
  struct PrecomputedContextKV {
    // Per draft attention layer: bf16 K (normed+RoPE'd) / V (raw), each flat
    // [num_ctx*Hkv*Dh] in ascending-position row order (== ctx_cu order per request).
    std::vector<std::vector<uint16_t>> k;
    std::vector<std::vector<uint16_t>> v;
    int64_t num_ctx = 0;
  };

  // Project `count` NEW context rows (combined features `new_features` [count,H] f32
  // at absolute `new_positions` [count]) to per-layer bf16 K/V and APPEND them to
  // `store` (one entry per config.num_hidden_layers, growing num_ctx by count). Reuses
  // the EXACT projection the full recompute runs (same ops, same order, each row RoPE'd
  // at its own position), so the appended bits equal the recompute's for those rows.
  static void AppendContextKVHost(PrecomputedContextKV& store,
                                  const std::vector<float>& new_features,
                                  const std::vector<int32_t>& new_positions,
                                  const Qwen3DFlashWeights& weights, const HfConfig& config,
                                  vt::Queue& queue);

  // CONTEXT-AWARE block forward over a PRECOMPUTED (persistent) context KV store —
  // identical to ForwardBlockLogitsWithContext EXCEPT it uploads `ckv`'s per-layer bf16
  // K/V instead of re-projecting the context each step (D9). `ctx_cu`/`cu` are the
  // per-request context/query boundaries (length num_reqs+1); ckv.num_ctx == ctx_cu.back().
  // The K is already normed+RoPE'd and V raw (as PrecomputeContextKVDevice produces), so
  // the [context; block] attention (the UNCHANGED D2 primitive) sees bit-identical inputs.
  static std::vector<float> ForwardBlockLogitsWithPrecomputedKV(
      const PrecomputedContextKV& ckv, const std::vector<int32_t>& ctx_cu,
      const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
      const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
      vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out = nullptr,
      std::vector<float>* final_out = nullptr);

  // ---- D11 Part A (device-resident append-only draft-KV store) ------------
  //
  // #1919: HOW BIG THE STORE IS, AND WHO DECIDED. Until #1919 the answer was a
  // compile-time `kDflashMaxCtxSlots = 4096` with no relation to the
  // `max_model_len` the engine advertises and admits, so a server at
  // `--max-model-len 12288` accepted an 8K-token prompt and then threw out of
  // the middle of an EngineCore step. Upstream keeps NO private store: the
  // DFlash draft's context K/V goes into the engine's own paged KV cache
  // (`vllm/model_executor/models/qwen3_dflash.py:604-620` at pin `5559679229`)
  // whose block tables are `cdiv(max_model_len, block_size)`
  // (`vllm/v1/worker/gpu/model_runner.py:426,444`), and its per-step bound is
  // `min(max_seq_len + num_query_per_req, max_model_len)`
  // (`vllm/v1/worker/gpu/spec_decode/dflash/speculator.py:331-333`).
  //
  // So the capacity is resolved from the engine's own context, plus the
  // `num_query_per_req` headroom the (1+k) query block needs — the W11 paged
  // route writes that block at slots `[C, C+Tq)`, and
  // `ClassifyDflashBlockAttn` declines the fast route below it.
  //
  // A CAP IS STILL REQUIRED, because `max_model_len` alone can be absurd: the
  // pool is per request, per draft layer, bf16, and doubled for K and V, so a
  // 262144-token context is about a gigabyte per concurrent request and this
  // box has OOM-rebooted from unbounded device residency (#1647).
  //
  // THE BUDGET IS A TOTAL, NOT A PER-REQUEST ALLOWANCE, and that is the point
  // of it. The runner builds ONE store per BATCH ROW, so peak device residency
  // is `bytes_per_request * max_num_reqs` — a per-request budget of 256 MiB was
  // an 8 GiB aggregate at the `--max-num-seqs 32` `docs/USAGE.md` itself shows,
  // none of it accounted by `gpu_memory_utilization`, which is the same
  // unbounded-residency shape #1647 names. So `kDflashCtxTotalBudgetBytes`
  // bounds the SUM across concurrent requests and the per-request share is
  // `total / max_num_reqs`. `VT_DFLASH_CTX_MAX_TOKENS` overrides the cap in
  // TOKENS per request, and the aggregate arithmetic is then the operator's
  // own. A capped resolution is ANNOUNCED at startup rather than applied
  // silently, together with the aggregate it costs, and a request that outgrows
  // the store falls back to the non-speculative path instead of refusing or
  // throwing.
  //
  // Every field is reported rather than derived by the caller, so the startup
  // line can state what it compared against what instead of printing one
  // number.
  struct DflashCtxStoreSizing {
    int64_t slots = 0;             // resolved capacity, a multiple of page_size
    int64_t want_slots = 0;        // max_model_len + num_query_per_req, page-rounded
    int64_t budget_slots = 0;      // what the byte budget (or the override) allows
    int64_t page_size = 0;         // rows per paged context page
    int64_t bytes_per_slot = 0;    // K+V, all draft layers, one context row
    int64_t bytes_per_request = 0; // slots * bytes_per_slot
    int64_t max_num_reqs = 0;      // concurrent batch rows == concurrent stores
    int64_t bytes_total = 0;       // bytes_per_request * max_num_reqs, the residency
    int64_t budget_bytes = 0;      // the TOTAL budget the cap was taken from
    bool overridden = false;       // VT_DFLASH_CTX_MAX_TOKENS named the cap
    bool capped = false;           // slots < want_slots
  };

  // Resolve the store's capacity for one draft geometry and one engine context.
  // Pure host arithmetic, no device read, so a CPU gate covers CUDA exactly.
  //
  // `max_num_reqs` is REQUIRED and has no default, for the same reason
  // `MakeDeviceKVStore`'s capacity has none: the budget it divides is the
  // aggregate one, and a defaulted 1 would silently restore the per-request
  // budget this parameter exists to remove.
  static DflashCtxStoreSizing ResolveCtxStoreSizing(const HfConfig& config,
                                                    int64_t max_model_len,
                                                    int64_t num_query_per_req,
                                                    int64_t max_num_reqs);

  // Create an empty per-request device KV store for `config.num_hidden_layers`
  // draft layers, holding `max_ctx_slots` context rows. Held by the runner
  // across verify steps (shared_ptr so the runner header needs only the forward
  // declaration).
  //
  // `max_ctx_slots` is REQUIRED and has no default on purpose (#1919). A
  // defaulted overload would let a call site keep the old fixed capacity
  // silently, which is the defect this parameter exists to remove.
  static std::shared_ptr<DflashDeviceKVStore> MakeDeviceKVStore(const HfConfig& config,
                                                                vt::Queue& queue,
                                                                int64_t max_ctx_slots);

  // Project the `count` NEW context rows (`new_features` [count,H] f32 at absolute
  // `new_positions` [count]) to per-layer bf16 K/V ON DEVICE and append them (as a
  // resident chunk) to `store`, growing num_ctx by count. Reuses the EXACT projection
  // (PrecomputeContextKVDevice) the host AppendContextKVHost runs, but keeps the bits
  // on device (no download). Per-row independence => bit-identical to the host store.
  static void AppendContextKVDevice(DflashDeviceKVStore& store,
                                    const std::vector<float>& new_features,
                                    const std::vector<int32_t>& new_positions,
                                    const Qwen3DFlashWeights& weights, const HfConfig& config,
                                    vt::Queue& queue);

  // SPEC-DFLASH2 W8 (#1838): the SAME projection+append, fed DEVICE-side. Gathers
  // the accepted-prefix rows `rows` (host i32 indices into `combined`, ascending
  // committed-position order — the rejection output already determines them) with
  // one vt::IndexSelect and runs the identical projection+scatter tail
  // AppendContextKVDevice runs, so the appended bits are the host path's exactly:
  // the host path's f32 detour around the same bf16 source was an exact round
  // trip. `combined` is the [T, H] bf16 CombineAuxFeaturesDevice output.
  static void AppendContextKVDeviceRows(DflashDeviceKVStore& store,
                                        const vt::Tensor& combined,
                                        const std::vector<int32_t>& rows,
                                        const std::vector<int32_t>& new_positions,
                                        const Qwen3DFlashWeights& weights,
                                        const HfConfig& config, vt::Queue& queue);

  // Number of context rows currently resident in a device store.
  static int64_t DeviceKVNumCtx(const DflashDeviceKVStore& store);

  // How many context rows the store can hold in total (#1919). The runner reads
  // it to decide, BEFORE appending, whether this request still fits — which is
  // what turns the capacity condition from a thrown `VT_CHECK` inside an
  // EngineCore step into a per-request fallback to the non-speculative path.
  static int64_t DeviceKVCapacity(const DflashDeviceKVStore& store);

  // CONTEXT-AWARE block forward over PERSISTENT DEVICE stores (the D11 production
  // path): concatenates the `stores` (one per propose request, in ctx_cu order) into
  // one combined device ContextKVDev ON DEVICE (no host round-trip) and runs the SAME
  // downstream core (ForwardWithCtxKVDev) the D9 host path used, so the result is
  // BIT-IDENTICAL to ForwardBlockLogitsWithPrecomputedKV given identical appends.
  // `ctx_cu`/`cu` are the per-request context/query boundaries (length num_reqs+1);
  // ctx_cu.back() == sum of the stores' num_ctx.
  // SPEC-DFLASH2 W8 (#1837): `device_out`, when non-null, receives the logits
  // and the post-final-norm hidden as DEVICE handles and the HOST return vector
  // comes back EMPTY — downloading it is the round trip #1837 measures.
  // Requesting `device_out` does NOT disqualify the single-request PAGED branch
  // (unlike `final_out`, whose host contract is what cost a DFlash2 draft the
  // D13 paged forward and its CUDA-graph capture), so a DFlash2 propose runs
  // paged+captured exactly as a DFlash1 one does. `device_out` together with
  // `final_out` is refused by name: no caller wants the same hidden both
  // resident and downloaded.
  static std::vector<float> ForwardBlockLogitsWithDeviceKV(
      const std::vector<DflashDeviceKVStore*>& stores, const std::vector<int32_t>& ctx_cu,
      const std::vector<int32_t>& block_input_ids, const std::vector<int32_t>& block_positions,
      const std::vector<int32_t>& cu, const Qwen3DFlashWeights& weights, const HfConfig& config,
      vt::Queue& queue, std::vector<std::vector<float>>* per_layer_out = nullptr,
      std::vector<float>* final_out = nullptr,
      DflashBlockDeviceOut* device_out = nullptr);
};

// prepare_dflash_inputs (dflash/speculator.py:472-687, the _prepare_dflash_inputs_kernel
// Triton kernel + its launcher). Ported as a HOST op (pure integer index arithmetic
// — exactly what the Triton kernel computes, no float math), so it is bit-exact and
// needs no CUDA/sanitizer. For each request it builds the (1+k) query block (anchor =
// bonus/verified token at offset 0, then k mask_token_id rows), the query
// positions/slots, the context positions/slots from the target block table, the
// per-mask sample maps, and seq_lens = last_valid_pos+1+num_query_per_req. Rejected
// positions are excluded via valid_ctx_end = ctx_end - num_rejected (:518), so the
// bonus/query positions anchor at the last ACCEPTED token. DISTINCT from MTP's
// shift-splice prepare_prefill_inputs.
struct DflashPrepareBatch {
  // Target batch inputs (mirror the kernel's pointer args).
  std::vector<int32_t> target_query_start_loc;  // [num_reqs+1] target token boundaries (ctx)
  std::vector<int64_t> target_positions;        // [num_target_tokens] absolute positions
  std::vector<int32_t> idx_mapping;             // [num_reqs] req_idx -> req_state_idx
  std::vector<int32_t> last_sampled;            // [max_num_reqs] indexed by req_state_idx
  std::vector<int32_t> next_prefill_tokens;     // [max_num_reqs] indexed by req_state_idx
  std::vector<int32_t> num_sampled;             // [num_reqs]
  std::vector<int32_t> num_rejected;            // [num_reqs]
  std::vector<int32_t> block_table;             // [max_num_reqs * block_table_stride]
  int32_t block_table_stride = 0;
  int32_t block_size = 0;                        // paged KV block size
  int32_t parallel_drafting_token_id = 0;        // = mask_token_id
  int32_t num_query_per_req = 0;                  // 1 + k
  int32_t num_speculative_steps = 0;             // k
  int32_t max_num_reqs = 0;
  int32_t max_num_tokens = 0;
  int32_t max_model_len = 0;
  bool sample_from_anchor = false;               // DFlash: false (anchor = bonus token)
};
struct DflashPrepareOutputs {
  std::vector<int32_t> input_ids;             // [num_reqs*num_query_per_req]
  std::vector<int64_t> query_positions;       // [num_reqs*num_query_per_req] (clamped max_model_len-1)
  std::vector<int32_t> query_start_loc;       // [max_num_reqs+1]
  std::vector<int32_t> seq_lens;              // [max_num_reqs]
  std::vector<int64_t> query_slot_mapping;    // [max_num_tokens] (PAD_SLOT_ID=-1 past active)
  std::vector<int64_t> context_positions;     // [num_target_tokens]
  std::vector<int64_t> context_slot_mapping;  // [num_target_tokens]
  std::vector<int64_t> sample_indices;        // [max_num_reqs*num_speculative_steps]
  std::vector<int64_t> sample_pos;            // [max_num_reqs*num_speculative_steps]
  std::vector<int32_t> sample_idx_mapping;    // [max_num_reqs*num_speculative_steps] (-1 past active)
};
DflashPrepareOutputs PrepareDflashInputs(const DflashPrepareBatch& batch);

}  // namespace vllm
