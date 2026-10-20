// Ported from: vllm/parser/engine/adapters.py:35
// (ParserEngineReasoningAdapter) + vllm/parser/engine/registered_adapters.py:48
// (Qwen3ParserReasoningAdapter, re-exported by
// vllm/reasoning/qwen3_engine_reasoning_parser.py) @ 555967922
// (vLLM 0.26.0.dev0).
//
// Group D of the reasoning-parser registry (see .agents/specs/
// reasoning-parsers.md): the families whose reasoning parser is NOT a
// hand-rolled text splitter but a thin FACE over the shared streaming parser
// ENGINE (`src/vllm/parser/engine/`, the already-landed TOOLS-STREAMING-PARSER
// work). One engine config per family; this adapter re-shapes it into the
// ReasoningParser contract, exactly as upstream's adapter re-shapes a
// ParserEngine into the legacy ReasoningParser ABC.
//
// Why an engine face and not another BaseThinkingReasoningParser: the engine
// grammar knows things a `<think>`/`</think>` string split cannot —
// qwen3's `<tool_call>` is an IMPLICIT reasoning end with no `</think>` at all
// (qwen3.py:137), a duplicate `</think>` in the content span is absorbed
// (qwen3.py:132), and the family's initial state (REASONING for thinking-on)
// decides whether a marker-less stream is reasoning or content. Re-deriving any
// of that in the text seam would be the parallel path the shared-seam rule
// forbids.
//
// DEVIATIONS (all inherited from the TEXT-ONLY reasoning seam documented in
// reasoning_parsers/abstract.h):
//   - is_reasoning_end takes the accumulated TEXT, not token ids; it delegates
//     to ParserEngine::is_reasoning_end(text) (parser_engine.py:595 in text
//     form).
//   - the streaming signature drops the three token-ID spans, so the engine is
//     fed text-only deltas; the incremental lexer already handles a marker
//     split across deltas, which is what the token-ID path bought upstream.
//   - the upstream ctor takes a tokenizer and the request's
//     chat_template_kwargs; the name-only factory cannot carry request kwargs
//     (same deviation as deepseek_v3 — threading is W4), so `thinking` is a
//     constructor argument that defaults to upstream's default (True).
//   - adjust_request / extract_content_ids / count_reasoning_tokens /
//     get_streaming_fallback_content are not on our ReasoningParser ABC and are
//     not modelled (token-ID and request-plumbing surfaces, W4).
#ifndef VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_PARSER_ENGINE_ADAPTER_H_
#define VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_PARSER_ENGINE_ADAPTER_H_

#include <memory>
#include <optional>
#include <string>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/entrypoints/openai/reasoning_parsers/abstract.h"
#include "vllm/parser/engine/parser_engine.h"

namespace vllm::entrypoints::openai {

// adapters.py:35 (ParserEngineReasoningAdapter). Holds ONE engine for the whole
// request, like every streaming reasoning parser on this seam.
class ParserEngineReasoningAdapter : public ReasoningParser {
 public:
  explicit ParserEngineReasoningAdapter(
      std::unique_ptr<vllm::parser::engine::ParserEngine> engine);
  ~ParserEngineReasoningAdapter() override;

  void adjust_initial_state_from_prompt(
      const std::vector<int>& prompt_token_ids) override;

  // adapters.py:69 extract_reasoning (under _skip_tool_parsing).
  ExtractedReasoning extract_reasoning(
      const std::string& model_output,
      const ChatCompletionRequest& request) override;

  // adapters.py:77 extract_reasoning_streaming (under _skip_tool_parsing).
  std::optional<DeltaMessage> extract_reasoning_streaming(
      const std::string& previous_text, const std::string& current_text,
      const std::string& delta_text,
      const ChatCompletionRequest& request) override;

  // adapters.py:60 is_reasoning_end.
  bool is_reasoning_end(const std::string& text) const override;

 protected:
  std::unique_ptr<vllm::parser::engine::ParserEngine> engine_;
};

// registered_adapters.py:48 — make_adapters(Qwen3Parser) reasoning half,
// re-exported as qwen3_engine_reasoning_parser.Qwen3ParserReasoningAdapter and
// registered under BOTH "qwen3" (__init__.py:115) and "mimo" (:87).
class Qwen3ParserReasoningAdapter final : public ParserEngineReasoningAdapter {
 public:
  // `thinking` mirrors chat_template_kwargs["enable_thinking"] (qwen3.py:226),
  // whose upstream default is True.
  explicit Qwen3ParserReasoningAdapter(bool thinking = true);
};

// Pinned a7c23ac96 registered_adapters.py:43-46; Gemma defaults false.
class Gemma4ParserReasoningAdapter final : public ParserEngineReasoningAdapter {
 public:
  explicit Gemma4ParserReasoningAdapter(
      bool thinking = false,
      const vllm::parser::engine::EngineTokenizer* tokenizer = nullptr);
};

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_REASONING_PARSERS_PARSER_ENGINE_ADAPTER_H_
