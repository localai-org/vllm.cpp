// Ported from vllm/model_executor/models/qwen3_5_mtp.py @
// e24d1b24fe96a56ba8b0d653efa076d03eb95d6c.
//
// Qwen3.5/3.6 MTP draft model used by the k=1 speculative-decoding path. The
// checkpoint-owned `mtp.*` tensors are loaded separately from the target model,
// while embed_tokens and the original lm_head reference the target weights as
// load_eagle_model does upstream (v1/worker/gpu/spec_decode/eagle/utils.py).
// GPTQ/XPU additionally owns a separate INT4 draft-head copy.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_dense.h"

namespace vllm {

enum class Qwen3_5MTPKind : uint8_t { kDense, kMoe };

// Complete 128-column Hadamard blocks from the packed target head, in the
// supplied production-list order. token_ids maps compact columns to global IDs.
struct Exl3DraftHead {
  Exl3Weight weight;
  OwnedTensor token_ids;  // I32 [65536]
  int64_t target_vocab = 0;
  bool Empty() const { return weight.Empty(); }
};

Exl3DraftHead BuildExl3DraftHead(const Exl3Weight& target_head,
                                const nlohmann::json& subset,
                                int64_t target_vocab);
Exl3DraftHead LoadExl3DraftHead(const std::vector<SafetensorsFile>& shards,
                               const nlohmann::json& subset,
                               int64_t target_vocab);

// The checkpoint-owned portion of Qwen3_5MultiTokenPredictor. Both gate
// checkpoints have one layer today, but the vector mirrors upstream's
// `mtp_num_hidden_layers` and spec_step_idx modulo selection.
struct Qwen3_5MTPWeights {
  Qwen3_5MTPKind kind = Qwen3_5MTPKind::kDense;
  OwnedTensor fc;  // bf16 raw torch Linear [H,2H], nk=true
  Gptq4Weight fc_gptq4;  // one-time packed GPTQ draft FC on the XPU path
  Gptq4Weight draft_lm_head_gptq4;  // separate draft copy; target head stays FP16
  // MODEL-QWEN35-EXL3-HEAD (#2495 item 5): the fc-cat projection as an
  // exllamav3 trellis [K=2H, N=H]. `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`
  // quantizes it -- `mtp.fc.trellis I16 [640, 320, 64]`, K=10240, N=5120,
  // bits 4, with a `.mul1` marker (codebook 2), read from the shard's own
  // safetensors header by range request. It is NOT remainder: it is the ninth
  // EXL3 tensor of the head, and the one the forward's own `fc.nk` precondition
  // refused outright. Exactly one of {`fc`, `fc_exl3`} is populated.
  Exl3Weight fc_exl3;
  Exl3DraftHead draft_head_exl3;
  OwnedTensor pre_fc_norm_embedding;  // [H], FP16 for XPU EXL3, otherwise BF16
  OwnedTensor pre_fc_norm_hidden;     // [H], Gemma RMSNorm
  OwnedTensor final_norm;             // [H], Gemma RMSNorm
  std::vector<Qwen3_5DenseLayerWeights> dense_layers;
  std::vector<Qwen3_5MoeLayerWeights> moe_layers;

  int64_t NumLayers() const;

  // ONE truth for "is this head's EXL3 arm active", keyed on the TRELLIS alone
  // and never on a config scalar, mirroring `FullAttnLayerWeights::IsExl3`
  // (keyed on `q_proj_exl3`) and `DenseMlpWeights::IsExl3` (on
  // `down_proj_exl3`). The loader fills `fc_exl3` together with the layer
  // projections, so a half-populated container cannot read as an arm; the
  // predicate answers the same question the loader's own `IsExl3Projection`
  // probe asked about `mtp.fc`.
  bool IsExl3() const { return !fc_exl3.Empty(); }
  bool IsGptq4Draft() const { return fc_gptq4.k != 0; }
};

// The productive XPU recipe quantizes the BF16 MTP linears and a separate
// FP16 draft-head copy once after checkpoint loading. This mutates only the
// draft container; the target's FP16 lm_head remains untouched.
void PackQwen3_5MTPGptqDraft(Qwen3_5MTPWeights& weights,
                             const StTensor& target_head);

// The MTP head depth, read from the checkpoint config (`mtp_num_hidden_layers`,
// under `text_config` when the config nests it). Defaults to 1, which is what
// both Qwen3.5/3.6 gate checkpoints ship. Exported because the GGUF head loader
// (`SPEC-MTP-GGUF`) resolves the same value from the same place, after
// HfConfigFromGguf republishes it from `<arch>.nextn_predict_layers`.
int64_t NumMtpLayers(const HfConfig& config);

// Whether the checkpoint gives the MTP head its OWN embedding table rather than
// sharing the target's (`mtp_use_dedicated_embeddings`). Both gate checkpoints
// set it false, and the loaders reject the true case.
bool UsesDedicatedEmbeddings(const HfConfig& config);

// Load only `mtp.*` tensors through an existing checkpoint resolver. XPU dense
// EXL3 casts the checkpoint norm remainders directly to the production FP16
// boundary. Other execution paths retain the strict BF16 remainder reader.
// Dedicated embeddings are rejected for this bounded
// Qwen3.6 leaf; both gate checkpoints set mtp_use_dedicated_embeddings=false.
//
// MODEL-QWEN35-EXL3-HEAD (#2495 item 5) adds the EXL3 arm and, with it, `has`.
// It is REQUIRED and has no default: an EXL3 projection ships no `.weight`, so
// selecting the arm means asking whether `<proj>.trellis` EXISTS, and a
// defaulted "assume nothing is quantized" predicate is exactly the silent-off
// shape `SharedHeadSource::LoadInto` documents at length. Dropping it is a
// compile error, not a green run that quietly loads no EXL3 head.
Qwen3_5MTPWeights LoadQwen3_5MTP(const TensorResolver& get,
                                 const std::function<bool(const std::string&)>& has,
                                 const HfConfig& config,
                                 Qwen3_5MTPKind kind,
                                 vt::DeviceType execution_device = vt::DeviceType::kCPU);

// Multi-shard convenience overload. It indexes the shard headers, then calls
// the resolver overload. Normal target-model loaders intentionally do not call
// this: vLLM loads the draft only when speculative decoding is enabled.
Qwen3_5MTPWeights LoadQwen3_5MTP(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    Qwen3_5MTPKind kind,
    vt::DeviceType execution_device = vt::DeviceType::kCPU);

// Owning device/CPU buffer for the MTP forward's direct hidden-state return.
// This mirrors Qwen3_5MTP.forward returning hidden states (not a tuple); the
// caller may then apply the target-shared lm_head through ComputeLogits().
struct Qwen3_5MTPHiddenStates {
  std::shared_ptr<void> storage;
  vt::Tensor tensor;  // bf16 generic or f16 XPU GPTQ [T,H]
  // Graph output leases retain their buffers and a producer completion event.
  // A consumer on another queue must wait before reading and retain storage
  // until its own queued reads complete. Ordinary eager returns leave this empty.
  std::shared_ptr<vt::Event> producer_ready_event = {};
  void WaitReady(vt::Queue& consumer_queue) const;
};

class Qwen3_5MTPModel {
 public:
  // The target and config must outlive this lightweight sharing wrapper.
  Qwen3_5MTPModel(const Qwen3_5MTPWeights& weights,
                  const Qwen3_5DenseWeights& target,
                  const HfConfig& config);
  Qwen3_5MTPModel(const Qwen3_5MTPWeights& weights,
                  const Qwen3_5MoeWeights& target,
                  const HfConfig& config);

  // The original head remains shared; GPTQ/XPU also has a packed draft copy.
  bool has_own_embed_tokens() const { return false; }
  bool has_own_lm_head() const { return false; }
  const OwnedTensor& embed_tokens() const { return *embed_tokens_; }
  const OwnedTensor* lm_head() const { return lm_head_; }
  const Nvfp4Weight* lm_head_fp4() const { return lm_head_fp4_; }
  // MODEL-QWEN35-EXL3-HEAD (#2495 item 5). The draft shares the TARGET's head,
  // so it must see the TRELLIS one too, and compute with it packed rather than
  // widen it -- which is what upstream does without a branch, because its head
  // is an `nn.Module` and `_apply_head` calls `lm_head.quant_method.apply`
  // (logits_processor.py:132-142). NULL on the MoE target: `Qwen3_5MoeWeights`
  // has no EXL3 head owner, and an absent field is said out loud here rather
  // than added as something nothing fills.
  const Exl3Weight* lm_head_exl3() const { return lm_head_exl3_; }

  // input_ids/positions and target_hidden_states all have T rows. The target
  // hidden states are the target model's post-final-norm output, matching
  // qwen3_5_mtp.py:129-165. `spec_step_idx` selects one MTP layer modulo depth.
  Qwen3_5MTPHiddenStates Forward(
      const std::vector<int32_t>& input_ids,
      const std::vector<int32_t>& positions,
      const vt::Tensor& target_hidden_states, vt::Queue& queue,
      int64_t spec_step_idx = 0) const;

  // PAGED MTP draft forward (SPEC-MTP I5c). Same head math as Forward (the
  // fc-cat-norm over embed(input_ids) + target_hidden), but the single
  // full_attention decoder layer reads/writes the DRAFT KV-cache layer through
  // the paged attention path (vt::ReshapeAndCache + vt::PagedAttention) using the
  // target's block table / slot mapping — mirroring qwen3_5_mtp.py:129-165 driven
  // over the paged attention backend the target layers already use, exactly what
  // AutoRegressiveSpeculator._prefill (speculator.py:332-370) runs each step. The
  // MTP head is layer_type="full_attention" (qwen3_5_mtp.py:105-112) so no GDN
  // path is ever taken; `attn_meta` carries slot_mapping / block_table / seq_lens
  // / query_start_loc and `draft_kv` is the head's own paged K/V layer (index
  // num_hidden_layers upstream). Over a single-request 1-block KV with a trivial
  // slot map this reproduces Forward's dense result (the paged==dense anchor).
  Qwen3_5MTPHiddenStates ForwardPaged(
      const std::vector<int32_t>& input_ids,
      const std::vector<int32_t>& positions,
      const vt::Tensor& target_hidden_states,
      const v1::CommonAttentionMetadata& attn_meta, PagedKvCache& draft_kv,
      vt::Queue& queue, int64_t spec_step_idx = 0,
      const vt::Tensor* inputs_embeds = nullptr) const;

  // inputs_embeds, when supplied, are already merged [T,H] in the hidden-state
  // dtype/device. The caller retains their owner until queued draft reads finish.
  // The pinned V2 image draft preserves ordinary 1-D positions here; target
  // M-RoPE axes are a separate input contract. Null keeps the existing lookup.

  // Gather `rows` of a [T,H] device hidden-state tensor into a fresh,
  // owning [rows.size(), H] buffer of the same dtype (SPEC-MTP-K-GT-1, #81).
  //
  // The mirror of `self.hidden_states[:num_reqs] = hidden_states[
  // last_token_indices]` (autoregressive/speculator.py:367-371 @ 555967922),
  // which is how the draft PREFILL hands its per-request hidden state to the
  // first multi-step draft DECODE. It lives on the model, not in the speculator,
  // because the device-buffer pool and the row-copy idiom are private to
  // qwen3_5.cpp; a copy in the speculator would be a second, parallel path to
  // the same allocator. `hidden_states` must be contiguous bf16 [T,H] (or
  // f16 on XPU) on the queue's device and every row index must be in [0, T).
  Qwen3_5MTPHiddenStates GatherHiddenRows(const vt::Tensor& hidden_states,
                                          const std::vector<int64_t>& rows,
                                          vt::Queue& queue) const;

  // XPU EXL3 returns [selected_rows,65536] compact logits with this stable map.
  // GatherHiddenRows must precede this call when only some forward rows sample.
  const Exl3DraftHead& compact_head() const { return weights_->draft_head_exl3; }
  // Apply the packed draft head when present, else the shared target head.
  ForwardLogits ComputeLogits(const vt::Tensor& hidden_states,
                              vt::Queue& queue) const;
  // Native XPU EXL3 selection copies only chosen global IDs, after completion.
  std::vector<int32_t> SelectDraftTokens(const ForwardLogits& logits,
                                        vt::Queue& queue) const;

  // Standalone parity convenience: Forward + shared lm_head + one host download.
  std::vector<float> ForwardLogitsHost(
      const std::vector<int32_t>& input_ids,
      const std::vector<int32_t>& positions,
      const vt::Tensor& target_hidden_states, vt::Queue& queue,
      int64_t spec_step_idx = 0) const;

 private:
  const Qwen3_5MTPWeights* weights_ = nullptr;
  const HfConfig* config_ = nullptr;
  const OwnedTensor* embed_tokens_ = nullptr;
  const OwnedTensor* lm_head_ = nullptr;
  const Nvfp4Weight* lm_head_fp4_ = nullptr;
  const Exl3Weight* lm_head_exl3_ = nullptr;
};

}  // namespace vllm
