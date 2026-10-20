// Ported from: vllm/parser/parser_manager.py (ParserManager.get_parser) +
// vllm/parser/engine/registered_adapters.py @ 555967922 (vLLM 0.26.0.dev0).
//
// The name -> engine-backed ParserEngine dispatch: the 0.26 replacement for the
// legacy per-family ToolParserManager for the engine-backed formats. Given a
// registered format name (see engine_backed_names()), returns a freshly
// constructed assembled ParserEngine (or the KimiK2Parser subclass), or nullptr
// for an unregistered name.
#pragma once

#include <memory>
#include <string>

#include "vllm/parser/engine/parser_engine.h"
#include "vllm/parser/engine/token_id_scanner.h"

namespace vllm::parser {

// Returns an assembled parser for `name`, or nullptr if the name is not an
// engine-backed format. `thinking` selects the reasoning-enabled config
// variant; `tokenizer` (optional) is only needed for token-ID-driven streams.
std::unique_ptr<engine::ParserEngine> get_parser_engine(
    const std::string& name, bool thinking,
    const engine::EngineTokenizer* tokenizer = nullptr);

// Omitted thinking is false for Gemma; all other families retain true.
std::unique_ptr<engine::ParserEngine> get_parser_engine(const std::string& name);

}  // namespace vllm::parser
