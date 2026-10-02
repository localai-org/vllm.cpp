// Ported from: vllm/v1/engine/input_processor.py @ e24d1b24
// See include/vllm/v1/engine/input_processor.h for scope, deviations and
// deferrals.
#include "vllm/v1/engine/input_processor.h"

#include <algorithm>
#include <chrono>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/tokenizer/tokenizer.h"

namespace vllm::v1 {
namespace {

// Wall-clock seconds since the epoch, mirroring upstream time.time().
double NowSeconds() {
  const auto now = std::chrono::system_clock::now().time_since_epoch();
  return std::chrono::duration<double>(now).count();
}

}  // namespace

InputProcessor::InputProcessor(const tok::Tokenizer& tokenizer,
                               const HfConfig& config,
                               int64_t max_model_len_override)
    : tokenizer_(tokenizer) {
  // model_config.max_model_len. HfConfig has no dedicated max_model_len (rope
  // scaling etc. are deferred), so max_position_embeddings stands in at T0 —
  // unless the caller already resolved a serving length (LoadedEngine's
  // --max-model-len override / KV auto-fit), which is what upstream's
  // model_config.max_model_len actually is.
  max_model_len_ = max_model_len_override > 0 ? max_model_len_override
                                              : config.max_position_embeddings;

  // renderer.get_eos_token_id() + generation_config["eos_token_id"]: derive the
  // primary eos id and the secondary eos-id list from config.json's
  // "eos_token_id" (int OR list), falling back to the tokenizer's own eos.
  bool found = false;
  if (config.raw.is_object()) {
    auto it = config.raw.find("eos_token_id");
    if (it != config.raw.end() && !it->is_null()) {
      if (it->is_number_integer()) {
        const auto id = it->get<int32_t>();
        eos_token_id_ = id;
        generation_config_eos_ids_.push_back(id);
        found = true;
      } else if (it->is_array()) {
        for (const auto& e : *it) {
          if (e.is_number_integer()) {
            generation_config_eos_ids_.push_back(e.get<int32_t>());
          }
        }
        if (!generation_config_eos_ids_.empty()) {
          eos_token_id_ = generation_config_eos_ids_.front();
          found = true;
        }
      }
    }
  }
  if (!found && tokenizer_.EosId() >= 0) {
    eos_token_id_ = tokenizer_.EosId();
    generation_config_eos_ids_.push_back(tokenizer_.EosId());
    found = true;
  }
  // Neither source named an eos: take tokenizer_config.json's eos_token, which
  // is upstream's primary source (renderers/base.py:310-317 @ 5559679229). It
  // must be exactly one token, or it is not guessed at. Every checkpoint that
  // already resolved an id above is untouched (ISSUE-LOCAL-01M3RTGVTN34YQFBR1KZH117XA).
  if (!found && !config.tokenizer_eos_token.empty()) {
    const std::vector<int32_t> ids = tokenizer_.Encode(config.tokenizer_eos_token);
    if (ids.size() == 1) {
      eos_token_id_ = ids.front();
      generation_config_eos_ids_.push_back(ids.front());
    }
  }

  // Upstream draws update_from_generation_config's id set from
  // generation_config.json (ModelConfig.try_get_generation_config), which is
  // usually a SUPERSET of config.json's: Gemma-4-26B ships config.json
  // [1, 106] against generation_config.json [1, 106, 50], and a port reading
  // only config.json never stops on 50. Union them, appending only ids not
  // already present so the PRIMARY eos id resolved above -- which is
  // generation_config_eos_ids_.front() on the list path -- keeps its position.
  // from_model_config's ids stand in for the file when it is absent; the
  // loader leaves them empty whenever the file exists.
  std::vector<int32_t> secondary = config.generation_config_eos_ids;
  secondary.insert(secondary.end(), config.model_config_eos_ids.begin(),
                   config.model_config_eos_ids.end());
  for (const int32_t id : secondary) {
    if (std::find(generation_config_eos_ids_.begin(),
                  generation_config_eos_ids_.end(),
                  id) == generation_config_eos_ids_.end()) {
      generation_config_eos_ids_.push_back(id);
    }
  }
}

void InputProcessor::ValidatePromptLen(std::size_t prompt_len) const {
  // Ported from vllm/v1/engine/input_processor.py:387-432 @ 555967922
  // (InputProcessor._validate_prompt_len), decoder arm.
  //
  // HARNESS ADAPTATION: upstream's early-out is `skip_prompt_length_check`, a
  // multimodal-processor property. Ours is `max_model_len_ <= 0`, which means
  // the config carried no context length at all — upstream's ModelConfig always
  // resolves a positive max_model_len so it has no such state, and checking
  // against 0 would reject every prompt. The encoder arm (mm_encoder_cache_size)
  // is deferred with the rest of the encoder/decoder split.
  if (max_model_len_ <= 0) {
    return;
  }

  // input_processor.py:395-396.
  if (prompt_len == 0) {
    throw InputValidationError("The decoder prompt cannot be empty");
  }

  const int64_t len = static_cast<int64_t>(prompt_len);
  // input_processor.py:404-421. Upstream's `suggestion` has a multimodal
  // variant; we always take the text one, because the mm path here receives an
  // ALREADY placeholder-expanded id stream, so `prompt_len` is the text + image
  // token count upstream's mm suggestion is describing.
  if (len > max_model_len_) {
    throw InputValidationError(
        "The decoder prompt (length " + std::to_string(len) +
        ") is longer than the maximum model length of " +
        std::to_string(max_model_len_) +
        ". Make sure that `max_model_len` is no smaller than the number of "
        "text tokens.");
  }
  // input_processor.py:423-432: exactly at the limit leaves no room for the at
  // least one output token every generate request asks for.
  if (len == max_model_len_) {
    throw InputValidationError(
        "The decoder prompt (length " + std::to_string(len) +
        ") plus the number of requested output tokens (at least 1) is longer "
        "than the maximum model length of " +
        std::to_string(max_model_len_) +
        ". Make sure that `max_model_len` is no smaller than the number of "
        "text tokens (prompt + requested output tokens).");
  }
}

void InputProcessor::ValidateParams(SamplingParams& params) const {
  // Upstream _validate_params calls params.verify(model_config, ...) after
  // __post_init__ already ran at construction. Our SamplingParams deferred
  // __post_init__ to this constructing unit (M1.1), so PostInit() both
  // normalizes the params AND runs Verify() — closing that carry.
  params.PostInit();

  // #249 — _validate_logprobs (sampling_params.py:755-770). Upstream REJECTS a
  // request whose logprobs exceed the cap; it does not clamp. Without this a
  // single `{"logprobs": 999999}` reached GatherLogprobs, which partial_sorts k
  // entries out of a vocab-sized index array and walked off the end — one
  // unauthenticated request killed the server process.
  //
  // Our ModelConfig carries no separate max_logprobs, so the cap is the vocab
  // size, which is upstream's own meaning for max_logprobs == -1.
  const int64_t vocab = static_cast<int64_t>(tokenizer_.VocabSize());
  const auto over_cap = [vocab](int n) { return vocab > 0 && n > vocab; };
  if (params.logprobs.has_value() && over_cap(*params.logprobs)) {
    throw std::runtime_error(
        "Requested sample logprobs of " + std::to_string(*params.logprobs) +
        ", which is greater than max allowed: " + std::to_string(vocab) + ".");
  }
  if (params.prompt_logprobs.has_value() && over_cap(*params.prompt_logprobs)) {
    throw std::runtime_error(
        "Requested prompt logprobs of " + std::to_string(*params.prompt_logprobs) +
        ", which is greater than max allowed: " + std::to_string(vocab) + ".");
  }
}

void InputProcessor::UpdateFromGenerationConfig(SamplingParams& params) const {
  // sampling_params.py:627-655. Sets eos_token_id, adds eos id(s) to
  // all_stop_token_ids (so the MinTokens processor masks them), and merges the
  // SECONDARY eos ids into stop_token_ids.
  if (!params.ignore_eos) {
    params.eos_token_id = eos_token_id_;
  }

  // The primary eos id feeds min_tokens masking (sampling_params.py:638-641).
  if (eos_token_id_.has_value()) {
    params.all_stop_token_ids.insert(*eos_token_id_);
  }

  if (generation_config_eos_ids_.empty()) {
    return;
  }
  std::set<int32_t> eos_ids(generation_config_eos_ids_.begin(),
                            generation_config_eos_ids_.end());
  // The primary eos id is handled separately for stopping; don't duplicate it.
  if (eos_token_id_.has_value()) {
    eos_ids.erase(*eos_token_id_);
  }
  if (!eos_ids.empty()) {
    // The full eos set contributes to min_tokens masking regardless of
    // ignore_eos (sampling_params.py:653).
    params.all_stop_token_ids.insert(eos_ids.begin(), eos_ids.end());
    if (!params.ignore_eos) {
      for (int32_t id : params.stop_token_ids) {
        eos_ids.insert(id);
      }
      params.stop_token_ids.assign(eos_ids.begin(), eos_ids.end());
    }
  }
}

void InputProcessor::UpdateFromTokenizer(SamplingParams& params) const {
  // sampling_params.py:659-698 (update_from_tokenizer): tokenize bad_words into
  // per-word token-id n-grams the sampler masks. Each word is encoded both
  // without and with a leading space (add_prefix_space) to catch it at the start
  // of and mid-text; the prefix-space variant is kept only when it produces a
  // genuinely different same-length token sequence.
  if (params.bad_words.empty()) {
    return;
  }
  std::vector<std::vector<int32_t>> bad_ids;
  for (const std::string& bad_word : params.bad_words) {
    // lstrip the word (add_prefix_space controls the leading space instead).
    size_t start = bad_word.find_first_not_of(" \t\n\r\f\v");
    const std::string stripped =
        start == std::string::npos ? std::string() : bad_word.substr(start);
    for (int variant = 0; variant < 2; ++variant) {
      const bool add_prefix_space = variant == 1;
      const std::string prompt =
          (add_prefix_space ? std::string(" ") : std::string()) + stripped;
      // Encode() adds no special tokens (== add_special_tokens=False).
      std::vector<int32_t> ids = tokenizer_.Encode(prompt);
      if (ids.empty()) continue;
      if (!add_prefix_space) {
        bad_ids.push_back(std::move(ids));
      } else if (!bad_ids.empty()) {
        const std::vector<int32_t>& last = bad_ids.back();
        if (!last.empty() && ids[0] != last[0] && ids.size() == last.size()) {
          bad_ids.push_back(std::move(ids));
        }
      }
    }
  }

  // Vocabulary-range check (sampling_params.py:683-698).
  const int32_t max_token_id = tokenizer_.VocabSize() - 1;
  for (const std::vector<int32_t>& ngram : bad_ids) {
    for (int32_t token_id : ngram) {
      if (token_id < 0 || token_id > max_token_id) {
        throw std::runtime_error(
            "The model vocabulary size is " +
            std::to_string(max_token_id + 1) +
            ", but a token specified as bad is out of range. All token id "
            "values should satisfy 0 <= token_id <= " +
            std::to_string(max_token_id) + ".");
      }
    }
  }

  params.bad_words_token_ids = std::move(bad_ids);
}

size_t InputProcessor::max_prompt_bytes() const {
  return tokenizer_.MaxPromptBytes(max_model_len_);
}

EngineCoreRequest InputProcessor::process_inputs(
    const std::string& request_id, const std::string& prompt,
    SamplingParams params, std::optional<double> arrival_time,
    int priority) const {
  // _validate_params: run PostInit()/Verify() on the (cloned) params.
  ValidateParams(params);

  const double t = arrival_time.has_value() ? *arrival_time : NowSeconds();

  // input_preprocessor.preprocess -> tokenize (text path only). vLLM tokenizes
  // prompts with HF's default `add_special_tokens=True`, so the tokenizer's
  // post_processor template is APPLIED here. This is a no-op for every Qwen
  // tokenizer (their ByteLevel post_processor declares no bos/eos) and supplies
  // the prepended `</s>` that OPT's TemplateProcessing declares.
  num_prompt_encodes_.fetch_add(1, std::memory_order_relaxed);
  std::vector<int32_t> prompt_token_ids =
      tokenizer_.EncodeWithSpecialTokens(prompt);

  // _validate_model_inputs -> _validate_prompt_len (input_processor.py:452).
  // BEFORE the max_tokens default below, which subtracts the prompt length and
  // would otherwise go negative for exactly the prompts this rejects.
  ValidatePromptLen(prompt_token_ids.size());

  // params is already our clone (passed by value). If unset max_tokens, then
  // generate up to the max_model_len (input_processor.py:317-321).
  if (!params.max_tokens.has_value()) {
    const int64_t seq_len = static_cast<int64_t>(prompt_token_ids.size());
    params.max_tokens = static_cast<int>(max_model_len_ - seq_len);
  }

  UpdateFromGenerationConfig(params);
  UpdateFromTokenizer(params);

  EngineCoreRequest request;
  request.request_id = request_id;
  request.prompt_token_ids = std::move(prompt_token_ids);
  request.sampling_params = std::move(params);
  request.arrival_time = t;
  request.priority = priority;
  return request;
}

EngineCoreRequest InputProcessor::process_inputs_tokens(
    const std::string& request_id, std::vector<int32_t> prompt_token_ids,
    SamplingParams params, std::optional<double> arrival_time,
    int priority) const {
  // Identical to process_inputs EXCEPT the prompt is already tokenized (vLLM
  // TokensPrompt): no tokenizer_.EncodeWithSpecialTokens call, and no
  // post_processor template applied (the caller supplies the exact ids, special
  // tokens included). Every other step (validate, default max_tokens, eos/stop
  // wiring, request assembly) is byte-for-byte the string path.
  ValidateParams(params);

  // _validate_prompt_len, same as the string path (input_processor.py:452).
  ValidatePromptLen(prompt_token_ids.size());

  const double t = arrival_time.has_value() ? *arrival_time : NowSeconds();

  if (!params.max_tokens.has_value()) {
    const int64_t seq_len = static_cast<int64_t>(prompt_token_ids.size());
    params.max_tokens = static_cast<int>(max_model_len_ - seq_len);
  }

  UpdateFromGenerationConfig(params);
  UpdateFromTokenizer(params);

  EngineCoreRequest request;
  request.request_id = request_id;
  request.prompt_token_ids = std::move(prompt_token_ids);
  request.sampling_params = std::move(params);
  request.arrival_time = t;
  request.priority = priority;
  return request;
}

EngineCoreRequest InputProcessor::process_inputs_mm(
    const std::string& request_id, std::vector<int32_t> prompt_token_ids,
    std::vector<multimodal::MultiModalFeatureSpec> mm_features,
    SamplingParams params, std::optional<double> arrival_time,
    int priority) const {
  // Multimodal path: the prompt is the ALREADY placeholder-EXPANDED id stream
  // (the serving-side mm processor ran and consumed the media), so like
  // process_inputs_tokens there is no tokenizer call. The ONLY delta vs the
  // tokens path is that mm_features is carried onto the EngineCoreRequest
  // (upstream input_processor.py:370-379 sets mm_features alongside
  // prompt_token_ids). Every other step is byte-for-byte identical.
  ValidateParams(params);

  // _validate_prompt_len over the EXPANDED id stream, which is the text +
  // placeholder token count upstream's multimodal suggestion text describes.
  ValidatePromptLen(prompt_token_ids.size());

  const double t = arrival_time.has_value() ? *arrival_time : NowSeconds();

  if (!params.max_tokens.has_value()) {
    const int64_t seq_len = static_cast<int64_t>(prompt_token_ids.size());
    params.max_tokens = static_cast<int>(max_model_len_ - seq_len);
  }

  UpdateFromGenerationConfig(params);
  UpdateFromTokenizer(params);

  EngineCoreRequest request;
  request.request_id = request_id;
  request.prompt_token_ids = std::move(prompt_token_ids);
  request.sampling_params = std::move(params);
  request.arrival_time = t;
  request.priority = priority;
  request.mm_features = std::move(mm_features);
  return request;
}

}  // namespace vllm::v1
