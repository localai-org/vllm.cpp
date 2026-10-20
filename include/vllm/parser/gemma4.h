// Ported from: vllm/parser/gemma4.py @ 555967922 (vLLM 0.26.0.dev0).
//
// Gemma4Parser: the assembly-layer ParserEngine subclass for Gemma-4's
// channel-based (<|channel>thought…<channel|>) reasoning + <|tool_call> tool
// format. Two assembly-core hooks beyond the base engine:
//   * _preprocess_feed (gemma4.py:424) injects the intrinsic <|channel> opener
//     on the first feed when the stream elides it (starts with `thought\n` or
//     carries a bare <channel|>), so reasoning is recognised.
//   * _events_to_delta (gemma4.py:530) strips the intrinsic `thought\n` channel
//     prefix from streamed reasoning (the header is part of the format).
// It also strips the same prefix in the non-streaming extract_reasoning path
// (gemma4.py:572). The custom key:value arg scanner lives in gemma4_config().
#pragma once

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/engine/parser_engine.h"

namespace vllm::parser {

class Gemma4Parser : public engine::ParserEngine {
 public:
  explicit Gemma4Parser(const engine::EngineTokenizer* tokenizer = nullptr,
                        bool thinking = false);

  void adjust_initial_state_from_prompt(
      const std::vector<int>& prompt_token_ids) override;

  bool is_reasoning_end(const std::string& text) const override;

 protected:
  // gemma4.py:418 _reset — also clears the prefix-strip / first-feed state.
  void reset(std::optional<engine::ParserState> initial_state) override;

  // gemma4.py:424 _preprocess_feed — inject <|channel> on the first feed.
  std::pair<std::string, std::vector<int>> preprocess_feed(
      const std::string& delta_text,
      const std::vector<int>& delta_token_ids) override;

  // gemma4.py:530 _events_to_delta — strip the `thought\n` reasoning prefix.
  std::optional<engine::oai::DeltaMessage> events_to_delta(
      const std::vector<engine::SemanticEvent>& events, bool finished) override;

  // gemma4.py:572 extract_reasoning — strip the `thought\n` prefix (non-stream).
  std::pair<std::optional<std::string>, std::optional<std::string>>
  extract_reasoning(const std::string& model_output,
                    const engine::ParserRequest& request) override;

 private:
  bool thinking_;
  std::vector<int> prompt_boundary_ids_;

  // gemma4.py:414-416 per-request state.
  std::string reasoning_text_;
  bool prefix_stripped_ = false;
  bool is_first_feed_ = true;
};

}  // namespace vllm::parser
