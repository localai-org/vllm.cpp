// Ported from: vllm/parser/gemma4.py @ 555967922 (vLLM 0.26.0.dev0).
#include "vllm/parser/gemma4.h"

#include <cstddef>
#include <algorithm>

namespace vllm::parser {

namespace {

// gemma4.py:38-41 channel/tool markers.
const std::string CHANNEL_START = "<|channel>";
const std::string CHANNEL_END = "<channel|>";
// gemma4.py:383-384 thought prefix/token.
const std::string THOUGHT_PREFIX = "thought\n";
const std::string THOUGHT_TOKEN = "thought";

bool contains(const std::string& hay, const std::string& needle) {
  return hay.find(needle) != std::string::npos;
}
bool starts_with(const std::string& s, const std::string& p) {
  return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

}  // namespace

// Pinned a7c23ac96d gemma4.py:418-437,480-526. No synthetic ID fallback.
Gemma4Parser::Gemma4Parser(const engine::EngineTokenizer* tokenizer, bool thinking)
    : engine::ParserEngine(engine::gemma4_config(), tokenizer),
      thinking_(thinking) {
  if (!tokenizer) return;
  const auto& vocab = tokenizer->get_vocab();
  for (const auto* marker : {"<channel|>", "<|tool_call>", "<|turn>",
                             "<|tool_response>"}) {
    const auto it = vocab.find(marker);
    if (it != vocab.end()) prompt_boundary_ids_.push_back(it->second);
  }
}

void Gemma4Parser::adjust_initial_state_from_prompt(
    const std::vector<int>& prompt_token_ids) {
  if (!reasoning_start_token_id_) return;
  for (auto it = prompt_token_ids.rbegin(); it != prompt_token_ids.rend(); ++it) {
    if (*it == *reasoning_start_token_id_) {
      prepare_streaming_state(engine::ParserState::REASONING);
      return;
    }
    if (std::find(prompt_boundary_ids_.begin(), prompt_boundary_ids_.end(), *it) !=
        prompt_boundary_ids_.end()) return;
  }
}

// Text-literal specialization of a7c23ac96d parser_engine.py:650-674.
// Gemma's CONTENT -> THINK_START opens reasoning; THINK_END and TOOL_START
// leave it. The latest of these markers or a turn boundary decides the tail.
// This text query does not claim original-vocabulary parity.
bool Gemma4Parser::is_reasoning_end(const std::string& text) const {
  std::size_t latest = std::string::npos;
  bool result = !thinking_;
  const auto consider = [&](const std::string& marker, bool ended) {
    const auto pos = text.rfind(marker);
    if (pos != std::string::npos &&
        (latest == std::string::npos || pos > latest)) {
      latest = pos;
      result = ended;
    }
  };
  consider(CHANNEL_END, true);
  consider("<|tool_call>", true);
  consider(CHANNEL_START, false);
  consider("<|turn>", !thinking_);
  consider("<|tool_response>", !thinking_);
  return result;
}

void Gemma4Parser::reset(std::optional<engine::ParserState> initial_state) {
  engine::ParserEngine::reset(initial_state);
  reasoning_text_.clear();
  prefix_stripped_ = false;
  is_first_feed_ = true;
}

std::pair<std::string, std::vector<int>> Gemma4Parser::preprocess_feed(
    const std::string& delta_text, const std::vector<int>& delta_token_ids) {
  if (!is_first_feed_) return {delta_text, delta_token_ids};
  is_first_feed_ = false;

  if (delta_text.empty() || engine_state() != engine::ParserState::CONTENT ||
      !reasoning_start_token_id_ || !reasoning_end_token_id_)
    return {delta_text, delta_token_ids};

  if (contains(delta_text, CHANNEL_START)) return {delta_text, delta_token_ids};

  const bool needs_injection = contains(delta_text, CHANNEL_END) ||
                               starts_with(delta_text, THOUGHT_PREFIX) ||
                               delta_text == THOUGHT_TOKEN;
  if (!needs_injection) return {delta_text, delta_token_ids};

  std::string text = CHANNEL_START + delta_text;
  std::vector<int> ids = delta_token_ids;
  if (!ids.empty()) ids.insert(ids.begin(), *reasoning_start_token_id_);
  return {text, ids};
}

std::optional<engine::oai::DeltaMessage> Gemma4Parser::events_to_delta(
    const std::vector<engine::SemanticEvent>& events, bool finished) {
  std::optional<engine::oai::DeltaMessage> delta =
      engine::ParserEngine::events_to_delta(events, finished);
  if (!delta || !delta->reasoning) return delta;
  if (prefix_stripped_) return delta;

  reasoning_text_ += *delta->reasoning;
  const std::size_t prefix_len = THOUGHT_PREFIX.size();

  if (starts_with(reasoning_text_, THOUGHT_PREFIX)) {
    std::size_t prev_reasoning_len =
        reasoning_text_.size() - delta->reasoning->size();
    if (prev_reasoning_len >= prefix_len) {
      prefix_stripped_ = true;
      return delta;
    }
    std::size_t chars_of_prefix_in_delta = prefix_len - prev_reasoning_len;
    std::string stripped = delta->reasoning->substr(chars_of_prefix_in_delta);
    if (!stripped.empty()) {
      prefix_stripped_ = true;
      delta->reasoning = stripped;
      return delta;
    }
    if (reasoning_text_.size() >= prefix_len) {
      prefix_stripped_ = true;
      delta->reasoning = std::nullopt;
      const bool has_tools = delta->tool_calls && !delta->tool_calls->empty();
      if (delta->content || has_tools) return delta;
      return std::nullopt;
    }
    return std::nullopt;
  }

  // _GEMMA4_THOUGHT_PREFIX.startswith(self._reasoning_text)
  if (reasoning_text_.size() <= prefix_len &&
      THOUGHT_PREFIX.compare(0, reasoning_text_.size(), reasoning_text_) == 0) {
    if (finished) prefix_stripped_ = true;
    return std::nullopt;
  }

  prefix_stripped_ = true;
  delta->reasoning = reasoning_text_;
  return delta;
}

std::pair<std::optional<std::string>, std::optional<std::string>>
Gemma4Parser::extract_reasoning(const std::string& model_output,
                                const engine::ParserRequest& request) {
  std::pair<std::optional<std::string>, std::optional<std::string>> base =
      engine::ParserEngine::extract_reasoning(model_output, request);
  // Build the result reasoning fresh (single assignment) — mirrors
  // `return reasoning or None`, dropping an empty post-strip result.
  std::optional<std::string> reasoning;
  if (base.first && !base.first->empty()) {
    const std::string& r = *base.first;
    if (starts_with(r, THOUGHT_PREFIX)) {
      std::string stripped = r.substr(THOUGHT_PREFIX.size());
      if (!stripped.empty()) reasoning = std::move(stripped);
    } else if (r == THOUGHT_TOKEN) {  // _GEMMA4_THOUGHT_PREFIX.rstrip()
      // -> None
    } else {
      reasoning = r;
    }
  }
  return {reasoning, base.second};
}

}  // namespace vllm::parser
