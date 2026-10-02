// The `NemotronH_Nano_VL_V2` / `NemotronH_Nano_Omni_Reasoning_V3` multimodal
// chat seam (row MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2,
// .agents/specs/nano-nemotron-vl-radio.md). One translation unit with the
// REGISTER_VLLM_MM_CHAT lines and zero edits to a shared array, as
// `mm_chat_qwen3vl.cpp` is.
//
// Ported from vLLM `e126687a9a`:
//   NanoNemotronVLProcessingInfo.get_supported_mm_limits (:238-242), image arm
//   BaseNanoNemotronVLProcessor._preprocess_image (processors/...:676-741)
//   NanoNemotronVLMultiModalProcessor._get_prompt_repl_image (:427-480)
// and the released chat template's image prefix
// (chat_template.jinja:207-238 @ e5e9932441de940c9a62185c870ea5bcd4cd24e2),
// which puts `<image>\n` (one image) or `<image 1><image> <image 2><image>\n`
// (several) IN FRONT of the message text, whatever the part order was.
//
// THE PROMPT IS RENDERED BY THE SERVER'S TEMPLATE, as for Qwen3-VL: each
// multimodal message is handed to it as the string that template itself would
// have built from the content parts (`NanoNemotronVLImageMmContent`), so the
// template's string branch renders it unchanged.
#include <algorithm>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/model_executor/models/nano_nemotron_vl.h"
#include "vllm/multimodal/nano_nemotron_vl_processor.h"
#include "vllm/multimodal/processing/context.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/input_processor.h"  // InputValidationError -> HTTP 400

namespace vllm::entrypoints::openai {
namespace {

std::map<std::string, std::optional<int>> NanoNemotronVLChatSupportedMmLimits() {
  // IMAGE only, unlimited (the tiler budgets every image of a prompt
  // together, compute_params :464-550). Audio and video are absent, which
  // context.py:414-415 reads as limit 0: their towers are not ported.
  return {{"image", std::nullopt}};
}

int32_t ResolveId(const vllm::tok::Tokenizer& tokenizer, const std::string& text) {
  for (const vllm::tok::SpecialToken& t : tokenizer.AddedTokens()) {
    if (t.text == text) return t.id;
  }
  throw std::runtime_error("Nemotron Nano VL multimodal chat seam: this checkpoint's "
                           "tokenizer has no added token '" +
                           text + "', which the image expansion writes");
}

MultiModalChatSeam MakeNanoNemotronVLChatSeam(const MultiModalChatContext& ctx) {
  if (ctx.tokenizer == nullptr || ctx.mm_config == nullptr || ctx.config == nullptr ||
      !ctx.prompt_fn || !ctx.codec) {
    throw std::runtime_error(
        "Nemotron Nano VL multimodal chat seam: the install context is incomplete "
        "(tokenizer, multimodal config, resolved model config, chat-prompt renderer "
        "and image codec are all required)");
  }
  const NanoNemotronVLParams params = ParseNanoNemotronVLParams(*ctx.config);
  multimodal::NanoNemotronVLProcessorConfig proc = params.processor;
  if (ctx.max_model_len <= 0) {
    throw std::runtime_error(
        "Nemotron Nano VL multimodal chat seam: the engine's max_model_len was not "
        "supplied, and the dynamic tiler's token budget is defined against it "
        "(processors/nano_nemotron_vl.py:330-331)");
  }
  proc.max_model_len = ctx.max_model_len;
  multimodal::NanoNemotronVLTokenIds ids;
  ids.img_start = ResolveId(*ctx.tokenizer, params.img_start_token);
  ids.img_context = ResolveId(*ctx.tokenizer, params.img_context_token);
  ids.img_end = ResolveId(*ctx.tokenizer, params.img_end_token);

  auto info = std::make_shared<const multimodal::BaseProcessingInfo>(
      *ctx.mm_config, NanoNemotronVLChatSupportedMmLimits());
  const vllm::tok::Tokenizer& tokenizer = *ctx.tokenizer;
  const std::string context_token = params.img_context_token;
  const std::string model_id = ctx.served_model_name;

  MultiModalChatSeam seam;
  seam.allowed_limits = info->AllowedMmLimits();
  seam.detail = "Nemotron Nano VL dynamic-resolution processor (RADIO tower, "
                "max_num_patches " + std::to_string(proc.max_num_patches) + ")";
  seam.chat_fn = [info, proc, ids, &tokenizer, context_token, model_id,
                  prompt_fn = ctx.prompt_fn, codec = ctx.codec](
                     const std::vector<ChatMessage>& messages)
      -> std::optional<multimodal::MultiModalInputs> {
    ValidateChatMmLimits(*info, messages);

    // The image parts are collected from the CALLER's messages, which outlive
    // this call; `rendered` drops its parts below.
    std::vector<const ChatContentPart*> image_parts;
    std::vector<ChatMessage> rendered = messages;
    for (size_t i = 0; i < messages.size(); ++i) {
      const ChatMessage& src = messages[i];
      if (!src.content_parts.has_value()) continue;
      int n_images = 0;
      std::string text;
      for (const ChatContentPart& part : *src.content_parts) {
        if (part.type == "text") {
          text += part.text;
        } else if (part.type == "image_url") {
          ++n_images;
          image_parts.push_back(&part);
        }
      }
      rendered[i].content = multimodal::NanoNemotronVLImageMmContent(n_images, text);
      rendered[i].content_parts.reset();
    }
    if (image_parts.empty()) return std::nullopt;

    // #1681: chat_template_kwargs cannot reach the mm seam (owed there).
    const std::string prompt =
        prompt_fn(rendered, /*add_generation_prompt=*/true, {},
                  nlohmann::ordered_json::object());
    const std::vector<int32_t> prompt_ids = tokenizer.EncodeWithSpecialTokens(prompt);
    // `sans_images = text[0].replace("<image>", "")` and
    // `len(tokenizer(sans_images, add_special_tokens=False).input_ids)`
    // (:689-692). `Encode` is that call: added tokens in the text are still
    // parsed, and no post_processor BOS/EOS is added, so a BOS-prepending
    // tokenizer does not shrink the tiler's budget by one token.
    std::string sans = prompt;
    for (size_t at = sans.find(context_token); at != std::string::npos;
         at = sans.find(context_token, at)) {
      sans.erase(at, context_token.size());
    }
    const int64_t text_len = static_cast<int64_t>(tokenizer.Encode(sans).size());

    std::vector<DecodedImageRgb> decoded;
    decoded.reserve(image_parts.size());
    for (const ChatContentPart* part : image_parts) {
      try {
        decoded.push_back(codec(DecodeImageUrlPart(*part)));
      } catch (const std::exception& e) {
        throw vllm::v1::InputValidationError(std::string("Nemotron Nano VL chat image: ") +
                                              e.what());
      }
    }
    std::vector<multimodal::NanoNemotronVLImageRgb> images;
    for (const DecodedImageRgb& d : decoded) {
      images.push_back({d.rgb.data(), d.height, d.width});
    }
    try {
      return multimodal::NanoNemotronVLPrepareInputs(prompt_ids, text_len, images, proc, ids,
                                                     model_id);
    } catch (const std::invalid_argument& e) {
      throw vllm::v1::InputValidationError(std::string("Nemotron Nano VL chat image: ") +
                                            e.what());
    }
  };
  return seam;
}

}  // namespace

REGISTER_VLLM_MM_CHAT(nano_nemotron_vl_v2, "NemotronH_Nano_VL_V2",
                      &MakeNanoNemotronVLChatSeam)
REGISTER_VLLM_MM_CHAT(nano_nemotron_omni_v3, "NemotronH_Nano_Omni_Reasoning_V3",
                      &MakeNanoNemotronVLChatSeam)

}  // namespace vllm::entrypoints::openai
