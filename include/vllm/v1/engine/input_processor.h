// Ported from: vllm/v1/engine/input_processor.py @ e24d1b24
// (InputProcessor.__init__ + process_inputs, the T0 text path).
//
// Scope (M1.8 Task 2): turn a text prompt + SamplingParams into an
// EngineCoreRequest the EngineCore (Task 1) can schedule. This is the T0 slice of
// upstream InputProcessor.process_inputs (input_processor.py:242-385):
//   _validate_params  -> RUN SamplingParams::PostInit()/Verify()  (see below)
//   preprocess/tokenize -> tokenizer.Encode(prompt) -> prompt_token_ids
//   sampling_params.clone() (implicit: params passed by value)
//   default max_tokens = model_config.max_model_len - len(prompt)  when unset
//   update_from_generation_config(generation_config, eos_token_id)  (:323)
//   update_from_tokenizer(tokenizer)                                (:328)
//   -> EngineCoreRequest(request_id, prompt_token_ids, sampling_params, ...)
//
// THE M1.1 CARRY THIS CLOSES: our SamplingParams (M1.1) deferred __post_init__
// to "the constructing unit — M1.8". The InputProcessor IS that unit:
// ValidateParams runs PostInit() (which normalizes in place AND runs Verify()),
// mirroring upstream, where __post_init__ ran at SamplingParams construction and
// process_inputs then calls params.verify(model_config, ...).
//
// DEVIATIONS vs the pinned API (recorded, use OUR names):
//   - __init__ takes VllmConfig (from which it pulls model_config,
//     generation_config_fields and a renderer holding the tokenizer). We hold a
//     tokenizer + HfConfig reference directly (the T0 deps), deriving
//     max_model_len from HfConfig.max_position_embeddings and the primary eos +
//     secondary eos ids from HfConfig.raw["eos_token_id"] (int OR list) with a
//     Tokenizer::EosId() fallback. HfConfig has no max_model_len override
//     (rope-scaling etc.), so max_position_embeddings stands in for it at T0.
//   - process_inputs signature reordered to (request_id, prompt, params,
//     arrival_time): only the text prompt + SamplingParams path is kept.
//   - update_from_generation_config: sets eos_token_id, adds the eos id(s) to
//     all_stop_token_ids (for MinTokens masking) and merges the SECONDARY eos
//     ids into stop_token_ids, matching sampling_params.py:627-655 (ROAD-V1-C7
//     wired all_stop_token_ids; it was previously dropped).
//   - update_from_tokenizer tokenizes bad_words into bad_words_token_ids
//     (sampling_params.py:659-698), ROAD-V1-C7 (previously a no-op stub while
//     bad_words was deferred on SamplingParams).
//
//   - _validate_model_inputs -> _validate_prompt_len (input_processor.py:387-432
//     @ 555967922) is LANDED for the decoder arm: an empty prompt, and one at or
//     past max_model_len, are refused with InputValidationError, which the
//     OpenAI server answers with HTTP 400. It is what stops an unservable prompt
//     from reaching the scheduler and sitting in `waiting` forever with an idle
//     GPU. The ENCODER arm (mm_encoder_cache_size) and the out-of-vocab check
//     remain deferred with the encoder/decoder split.
//
// DEFERRED (marked; matches upstream so re-adding is mechanical): dict/EngineInput
// prompts, prompt_embeds, encoder/decoder split, multimodal (mm_features),
// pooling (PoolingParams), LoRA, data_parallel_rank validation, request-id
// randomization (assign_request_id), the _validate_model_inputs out-of-vocab
// check and its encoder arm, current_platform.validate_request, trace_headers,
// priority, resumable.
#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/engine/validation_error.h"  // InputValidationError (was defined here)

namespace vllm::tok {
class Tokenizer;  // vllm/tokenizer/tokenizer.h
}

namespace vllm::v1 {

// InputValidationError — the refusal this processor throws — now lives in
// vllm/v1/engine/validation_error.h, included above, so that the multimodal
// limit check can throw the SAME type without including this header. Same name,
// same namespace: every existing thrower and catcher is unaffected.

class InputProcessor {
 public:
  // __init__ (T0 deps): the tokenizer + HfConfig the processor reads. The
  // TOKENIZER must outlive the InputProcessor; the HfConfig need not — it is
  // fully consumed here (max_model_len + eos ids are derived up front and the
  // config is not retained).
  //
  // `max_model_len_override` (> 0) supplies the ALREADY-RESOLVED serving length
  // in place of the checkpoint's own context window. Upstream reads
  // model_config.max_model_len, which is exactly that resolved value; our
  // HfConfig only carries the raw max_position_embeddings, so LoadedEngine
  // threads its resolved max_model_len_ (the `--max-model-len` override, or the
  // KV-pool auto-fit) through here. 0 keeps the historical
  // derive-from-config behaviour, so every existing construction is unchanged.
  InputProcessor(const tok::Tokenizer& tokenizer, const HfConfig& config,
                 int64_t max_model_len_override = 0);

  // process_inputs (text path): validate + tokenize + build the request.
  // `params` is taken BY VALUE (upstream clones it); PostInit()/eos-wiring
  // mutate the local copy, never the caller's. Throws std::runtime_error via
  // SamplingParams::Verify() on invalid params. arrival_time defaults to the
  // wall clock (upstream time.time()). `priority` (upstream process_inputs
  // priority arg) is carried onto EngineCoreRequest.priority; default 0.
  EngineCoreRequest process_inputs(
      const std::string& request_id, const std::string& prompt,
      SamplingParams params,
      std::optional<double> arrival_time = std::nullopt,
      int priority = 0) const;

  // process_inputs for a PRE-TOKENIZED prompt (vLLM `TokensPrompt`: a request
  // built from prompt_token_ids directly, skipping tokenization). Mirrors the
  // input_preprocessor token-prompt branch. Strictly ADDITIVE: the string
  // overload above is UNCHANGED and every existing caller keeps the tokenize
  // path byte-for-byte. Used to gate a model whose tokenizer family we have not
  // ported yet (e.g. InternLM2's non-standard fast BPE) by feeding the oracle's
  // exact prompt token ids, isolating the forward from tokenization.
  EngineCoreRequest process_inputs_tokens(
      const std::string& request_id, std::vector<int32_t> prompt_token_ids,
      SamplingParams params,
      std::optional<double> arrival_time = std::nullopt,
      int priority = 0) const;

  // process_inputs for a MULTIMODAL prompt (ROAD-V1-MM MM-SERVE-ENGINE). The
  // `prompt_token_ids` are the ALREADY placeholder-EXPANDED ids (one image_pad /
  // audio_pad per feature slot) produced by the serving-side mm processor
  // (chat_mm RouteImageRgb / RouteAudioWav) and `mm_features` the matching
  // per-item specs (mm-hash + span + encoder input). Mirrors upstream
  // input_processor.py:333-379, where process_inputs builds the EngineCoreRequest
  // with mm_features from the mm-processor output alongside the expanded
  // prompt_token_ids. Strictly ADDITIVE: identical to process_inputs_tokens (no
  // tokenization; validate / default-max_tokens / eos+stop wiring byte-for-byte)
  // EXCEPT it also carries mm_features onto the request. An EMPTY mm_features
  // vector makes it byte-identical to process_inputs_tokens (every downstream mm
  // hook a no-op), so the text path is never perturbed.
  EngineCoreRequest process_inputs_mm(
      const std::string& request_id, std::vector<int32_t> prompt_token_ids,
      std::vector<multimodal::MultiModalFeatureSpec> mm_features,
      SamplingParams params,
      std::optional<double> arrival_time = std::nullopt,
      int priority = 0) const;

  // The resolved max_model_len that ValidatePromptLen refuses against.
  int64_t max_model_len() const { return max_model_len_; }

  // The pre-tokenization byte bound for a prompt this processor will encode:
  // Tokenizer::MaxPromptBytes(max_model_len()), 0 when unbounded. The chat
  // handler applies it to the RENDERED prompt before the encode, which is where
  // vLLM's _text_len_check runs (vllm/renderers/params.py:342-370,386-399 @
  // e126687a9a). Out of line: this header only forward-declares Tokenizer.
  size_t max_prompt_bytes() const;

  // The number of prompts the text `process_inputs` overload has handed to the
  // tokenizer since construction. It is an observation point, not a policy: a
  // request boundary that must refuse a prompt BEFORE the encode is proved by
  // this count staying unchanged (SERVE-REQUEST-LENGTH-GUARD). Relaxed, because
  // it orders nothing.
  uint64_t num_prompt_encodes() const {
    return num_prompt_encodes_.load(std::memory_order_relaxed);
  }

  // The same resolved model EOS set used for request stop handling. Grammar
  // backends must recognize secondary/chat EOS ids too, even when the tokenizer
  // JSON has no EOS post-processor.
  const std::vector<int32_t>& eos_token_ids() const {
    return generation_config_eos_ids_;
  }

 private:
  // _validate_params: runs SamplingParams::PostInit() (normalize + Verify) —
  // this closes the M1.1 deferred-__post_init__ carry.
  void ValidateParams(SamplingParams& params) const;
  // _validate_prompt_len (input_processor.py:387-432), decoder arm: refuse an
  // empty prompt and one at or past max_model_len. Throws InputValidationError,
  // which the OpenAI server answers with HTTP 400.
  void ValidatePromptLen(std::size_t prompt_len) const;
  // update_from_generation_config (T0 subset: eos_token_id + secondary stop ids).
  void UpdateFromGenerationConfig(SamplingParams& params) const;
  // update_from_tokenizer (T0 no-op: bad_words is deferred).
  void UpdateFromTokenizer(SamplingParams& params) const;

  const tok::Tokenizer& tokenizer_;
  // NOTE: no `const HfConfig&` member. The constructor consumes the HfConfig
  // ENTIRELY at construction — max_model_len_, eos_token_id_ and
  // generation_config_eos_ids_ below are everything this class needs from it —
  // so the reference it used to hold was never read. Clang's
  // -Wunused-private-field flagged it while building on macOS
  // (BACKEND-METAL-MLX W0). Removed rather than suppressed, which also drops a
  // lifetime obligation: the HfConfig no longer has to outlive the processor.
  // model_config.max_model_len (T0: HfConfig.max_position_embeddings).
  int64_t max_model_len_ = 0;
  // The primary eos id (renderer.get_eos_token_id()): the eos_ids list head, or
  // the tokenizer's eos, or unset.
  std::optional<int32_t> eos_token_id_;
  // generation_config["eos_token_id"] as a list (int is a 1-element list).
  std::vector<int32_t> generation_config_eos_ids_;
  // See num_prompt_encodes().
  mutable std::atomic<uint64_t> num_prompt_encodes_{0};
};

}  // namespace vllm::v1
