// Ported from: vllm/parser/parser_manager.py + registered_adapters.py @
// 555967922 (vLLM 0.26.0.dev0).
#include "vllm/parser/parser_manager.h"

#include "vllm/parser/engine/configs.h"
#include "vllm/parser/gemma4.h"
#include "vllm/parser/glm47_moe.h"
#include "vllm/parser/inkling.h"
#include "vllm/parser/kimi_k2.h"
#include "vllm/parser/qwen3.h"

namespace vllm::parser {

std::unique_ptr<engine::ParserEngine> get_parser_engine(const std::string& name) {
  return get_parser_engine(name, name != "gemma4");
}

std::unique_ptr<engine::ParserEngine> get_parser_engine(
    const std::string& name, bool thinking,
    const engine::EngineTokenizer* tokenizer) {
  // qwen3 + seed_oss share the qwen3 grammar AND the qwen3 class: upstream
  // seed_oss.py is `class SeedOssParser(Qwen3Parser)` overriding only the four
  // wrapper token strings, which live in the config. Qwen3Parser adds the
  // thinking-off passthrough (qwen3.py:247) and the unpaired-tool-call
  // reasoning end (qwen3.py:256); both read their literals from the config, so
  // one class serves both wrapper spellings. The tool-assembly behaviour is
  // unchanged — neither override is on the tool path.
  if (name == "qwen3") {
    return std::make_unique<Qwen3Parser>(engine::qwen3_config(thinking, "qwen3"),
                                         thinking, tokenizer);
  }
  if (name == "seed_oss") {
    return std::make_unique<Qwen3Parser>(
        engine::qwen3_config(thinking, "seed_oss", "<seed:think>",
                             "</seed:think>", "<seed:tool_call>",
                             "</seed:tool_call>"),
        thinking, tokenizer);
  }
  // kimi_k2 needs the native-header name/id parsing overrides (kimi_k2.py).
  if (name == "kimi_k2") {
    return std::make_unique<KimiK2Parser>(thinking, tokenizer);
  }
  // minimax_m2 / deepseek_v4 / deepseek_v32 / nemotron_v3: the arg_converter and
  // all assembly behavior live in the config, so a base ParserEngine over the
  // right config suffices (the deepseek subclasses' _convert_args wrapper-unwrap
  // degenerates to the config converter under the no-tool-schema model — same
  // residual as _fix_arg_types; see parser_engine.h).
  if (name == "minimax_m2") {
    return std::make_unique<engine::ParserEngine>(engine::minimax_m2_config(),
                                                  tokenizer);
  }
  if (name == "deepseek_v4") {
    return std::make_unique<engine::ParserEngine>(
        engine::deepseek_v4_config(thinking), tokenizer);
  }
  if (name == "deepseek_v32") {
    return std::make_unique<engine::ParserEngine>(engine::deepseek_v32_config(),
                                                  tokenizer);
  }
  if (name == "nemotron_v3") {
    return std::make_unique<engine::ParserEngine>(
        engine::nemotron_v3_config(thinking), tokenizer);
  }
  // glm47_moe strips the function name via the emit_name_delta / handle_tool_end
  // hooks (glm47_moe.py).
  if (name == "glm47_moe") {
    return std::make_unique<Glm47MoeParser>(thinking, tokenizer);
  }
  // gemma4 adds the _preprocess_feed (channel-opener injection) + _events_to_delta
  // (thought-prefix strip) hooks. Thinking selects only reasoning-end policy;
  // the initial assembly state remains CONTENT in either mode.
  if (name == "gemma4") {
    return std::make_unique<Gemma4Parser>(tokenizer, thinking);
  }
  // inkling adds the "args" wrapper-key unwrap + trailing-text single-pass flush
  // hooks (inkling.py); MESSAGE_HEADER-initial, no reasoning tokens.
  if (name == "inkling") {
    return std::make_unique<InklingParser>(tokenizer);
  }
  return nullptr;
}

}  // namespace vllm::parser
