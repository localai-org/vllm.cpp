// Qwen3-VL's multimodal CHAT seam, registered on its own architecture.
//
// This is the mirror of `qwen3_vl.py:1673-1678` @ 5559679229bc, where the
// registration sits on the model class and edits no shared table. Before
// ENG-MM-INPUT-PIPELINE (#2475) this body was fifty lines inside
// `server_main.cpp`'s `main`, reached by asking whether
// `<model_dir>/preprocessor_config.json` existed and never asking which
// architecture was loaded.
//
// Nothing about Qwen3-VL's behaviour changes here. The processor config is read
// from the same two files with the same loader, the seam body is the same
// `MakeQwen3VLImageChatFn`, and the per-modality ceiling is the same
// `Qwen3VLChatSupportedMmLimits()`. That function does NOT move: it stays
// declared in `chat_mm.h` and defined in `chat_mm.cpp`. What moves here is the
// CALL, which `main` used to make. Reading this architecture's own ceiling from
// this architecture's own factory is what upstream does when it takes
// `info.supported_mm_limits` off the model class (context.py:378-390).
#include <filesystem>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/multimodal/processing/context.h"
#include "vllm/multimodal/qwen3vl_processor.h"

namespace vllm::entrypoints::openai {

namespace {

namespace fs = std::filesystem;

// TU-local, matching server_main.cpp:127-144. A checkpoint path arrives as UTF-8
// and `fs::path` is `wchar_t`-based on Windows.
fs::path NativeUtf8Path(const std::string& value) {
#if defined(_WIN32)
  const std::u8string utf8(reinterpret_cast<const char8_t*>(value.data()),
                           value.size());
  return fs::path(utf8);
#else
  return fs::path(value);
#endif
}

std::string PathUtf8(const fs::path& path) {
#if defined(_WIN32)
  const std::u8string utf8 = path.u8string();
  return std::string(reinterpret_cast<const char*>(utf8.data()), utf8.size());
#else
  return path.string();
#endif
}

MultiModalChatSeam MakeQwen3VLChatSeam(const MultiModalChatContext& ctx) {
  if (ctx.tokenizer == nullptr || ctx.mm_config == nullptr || !ctx.prompt_fn ||
      !ctx.codec) {
    // Refuse by name rather than dereference. A context assembled without one
    // of these is a caller defect, and the install's catch turns it into a
    // refusing seam rather than a crash.
    throw std::runtime_error(
        "Qwen3-VL multimodal chat seam: the install context is incomplete "
        "(tokenizer, multimodal config, chat-prompt renderer and image codec "
        "are all required)");
  }

  const std::string preprocessor_config_path =
      PathUtf8(NativeUtf8Path(ctx.model_dir) / "preprocessor_config.json");
  if (!fs::exists(NativeUtf8Path(preprocessor_config_path))) {
    // Named, not silent. The live case is a GGUF Qwen3-VL served with
    // `--mmproj`, whose directory carries no `preprocessor_config.json`: the
    // server used to start text-only and answer image requests as text.
    throw std::runtime_error(
        "Qwen3-VL multimodal chat seam: '" + preprocessor_config_path +
        "' is missing; the image processor's patch/merge geometry and pixel "
        "bounds are read from it");
  }

  // Both are captured by the returned chat function, so the seam owns its own
  // state and the server keeps no locals for it.
  auto proc = std::make_shared<multimodal::Qwen3VLImageProcessor>(
      multimodal::LoadQwen3VLProcessorConfig(
          preprocessor_config_path, ctx.config_path, ctx.served_model_name));
  // #607 L2 / #686: the engine's limits (`--limit-mm-per-prompt`,
  // `--language-model-only`) folded by min() against this seam's own ceiling, so
  // a three-image request is HTTP 400 with upstream's message instead of being
  // served with its first image. The MultiModalConfig is held BY REFERENCE
  // (context.h:105); the engine owns it and outlives the seam.
  auto info = std::make_shared<multimodal::BaseProcessingInfo>(
      *ctx.mm_config, Qwen3VLChatSupportedMmLimits());

  MultiModalChatSeam seam;
  seam.allowed_limits = info->AllowedMmLimits();
  seam.detail = "Qwen3-VL processor from " + preprocessor_config_path;
  MultiModalRequestChatFn body = MakeQwen3VLImageRequestChatFn(
      *proc, *ctx.tokenizer, ctx.prompt_fn, ctx.codec, *info);
  seam.request_chat_fn = [proc, info, body = std::move(body)](
      const std::vector<ChatMessage>& messages,
      const std::vector<ChatCompletionToolsParam>& tools,
      const nlohmann::ordered_json& kwargs)
      -> std::optional<multimodal::MultiModalInputs> {
    return body(messages, tools, kwargs);
  };
  return seam;
}

}  // namespace

REGISTER_VLLM_MM_CHAT(qwen3_vl, "Qwen3VLForConditionalGeneration",
                      &MakeQwen3VLChatSeam)

}  // namespace vllm::entrypoints::openai
