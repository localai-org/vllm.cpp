#include "vllm/parser/gemma4.h"
// See parser_engine_adapter.h. Ported from vllm/parser/engine/adapters.py:35 +
// registered_adapters.py:48 @ 555967922.
#include "vllm/entrypoints/openai/reasoning_parsers/parser_engine_adapter.h"

#include <memory>
#include <optional>
#include <string>
#include <utility>

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/qwen3.h"

namespace vllm::entrypoints::openai {

namespace pe = vllm::parser::engine;

namespace {

// adapters.py:51 _skip_tool_parsing — a scoped contextmanager that forces the
// engine to treat tool markers as ordinary content for the duration of a
// reasoning call, restoring the previous value afterwards. The reasoning face
// must not consume the tool body: `<tool_call>` still ENDS reasoning (the
// engine emits REASONING_END + TEXT_CHUNK for it), but everything after it
// stays verbatim content for the tool parser to handle later.
class SkipToolParsing {
 public:
  explicit SkipToolParsing(pe::ParserEngine& engine)
      : engine_(engine), saved_(engine.skip_tool_parsing()) {
    engine_.set_skip_tool_parsing(true);
  }
  ~SkipToolParsing() { engine_.set_skip_tool_parsing(saved_); }

  SkipToolParsing(const SkipToolParsing&) = delete;
  SkipToolParsing& operator=(const SkipToolParsing&) = delete;

 private:
  pe::ParserEngine& engine_;
  bool saved_;
};

}  // namespace

ParserEngineReasoningAdapter::ParserEngineReasoningAdapter(
    std::unique_ptr<pe::ParserEngine> engine)
    : engine_(std::move(engine)) {}

ParserEngineReasoningAdapter::~ParserEngineReasoningAdapter() = default;

// adapters.py:69.
ExtractedReasoning ParserEngineReasoningAdapter::extract_reasoning(
    const std::string& model_output, const ChatCompletionRequest& /*request*/) {
  SkipToolParsing guard(*engine_);
  // ParserEngine::extract_reasoning (parser_engine.py:490) reads nothing off
  // the request, so the engine-side request subset is left at its defaults.
  auto [reasoning, content] =
      engine_->extract_reasoning(model_output, pe::ParserRequest{});
  return ExtractedReasoning{reasoning, content};
}

// adapters.py:77.
std::optional<DeltaMessage>
ParserEngineReasoningAdapter::extract_reasoning_streaming(
    const std::string& /*previous_text*/, const std::string& /*current_text*/,
    const std::string& delta_text, const ChatCompletionRequest& /*request*/) {
  SkipToolParsing guard(*engine_);
  return engine_->extract_reasoning_streaming(delta_text);
}

// adapters.py:60.
bool ParserEngineReasoningAdapter::is_reasoning_end(
    const std::string& text) const {
  return engine_->is_reasoning_end(text);
}

// registered_adapters.py:48 — make_adapters(Qwen3Parser).
Qwen3ParserReasoningAdapter::Qwen3ParserReasoningAdapter(bool thinking)
    : ParserEngineReasoningAdapter(std::make_unique<vllm::parser::Qwen3Parser>(
          pe::qwen3_config(thinking, "qwen3"), thinking)) {}

void ParserEngineReasoningAdapter::adjust_initial_state_from_prompt(
    const std::vector<int>& prompt_token_ids) {
  // Pinned a7c23ac96 adapters.py:68-69: direct forwarding, caller owns timing.
  engine_->adjust_initial_state_from_prompt(prompt_token_ids);
}

Gemma4ParserReasoningAdapter::Gemma4ParserReasoningAdapter(
    bool thinking, const pe::EngineTokenizer* tokenizer)
    : ParserEngineReasoningAdapter(
          std::make_unique<vllm::parser::Gemma4Parser>(tokenizer, thinking)) {}

}  // namespace vllm::entrypoints::openai
