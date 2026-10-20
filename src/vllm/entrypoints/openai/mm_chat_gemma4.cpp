// The `Gemma4ForConditionalGeneration` multimodal chat seam (row
// MODEL-MM-gemma4-mm-gemma4-for-conditional-generation,
// .agents/specs/gemma4-e4b-google-baseline.md).
//
// T10B: image processing through the full serving path. The seam
// decodes image_url parts, runs the Gemma4ImageProcessor, expands
// placeholder tokens, and builds MultiModalInputs. Audio and video
// content parts are refused with a clear message.
//
// Text-only requests pass through unchanged (return nullopt).
#include <algorithm>
#include <cstring>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/model_executor/models/gemma4.h"
#include "vllm/multimodal/gemma4_image_processor.h"
#include "vllm/multimodal/hasher.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm::entrypoints::openai {
namespace {

using json = nlohmann::ordered_json;

// Check whether any message carries audio or video parts.
bool HasAudioOrVideoParts(const std::vector<ChatMessage>& messages) {
  for (const auto& m : messages) {
    if (!m.content_parts.has_value()) continue;
    for (const auto& p : *m.content_parts) {
      if (p.type == "input_audio" || p.type == "video_url") return true;
    }
  }
  return false;
}

MultiModalChatSeam MakeGemma4ChatSeam(const MultiModalChatContext& ctx) {
  if (ctx.tokenizer == nullptr || ctx.config == nullptr) {
    throw std::runtime_error(
        "Gemma4 multimodal chat seam: the install context is incomplete "
        "(tokenizer and resolved model config are required)");
  }
  if (!ctx.prompt_fn) {
    throw std::runtime_error(
        "Gemma4 multimodal chat seam: prompt renderer is required");
  }
  if (!ctx.codec) {
    throw std::runtime_error(
        "Gemma4 multimodal chat seam: image codec is required");
  }

  // Resolve special token IDs from config.json (T03B).
  const auto spec_ids = Gemma4SpecialTokenIds(*ctx.config);
  const int32_t boi_id = spec_ids.boi_token_id;
  const int32_t eoi_id = spec_ids.eoi_token_id;
  const int32_t image_token_id = spec_ids.image_token_id;

  const std::string model_id = ctx.served_model_name;
  const auto& tokenizer = *ctx.tokenizer;
  const auto prompt_fn = ctx.prompt_fn;
  const auto codec = ctx.codec;

  MultiModalChatSeam seam;
  seam.detail =
      "Gemma4 multimodal chat seam (image processing, audio/video pending)";
  seam.allowed_limits = {{"image", -1}};

  seam.chat_fn = [boi_id, eoi_id, image_token_id, model_id, &tokenizer,
                  prompt_fn, codec](
                     const std::vector<ChatMessage>& messages)
      -> std::optional<multimodal::MultiModalInputs> {
    // Walk messages: collect image parts and build text-only rendered messages.
    std::vector<const ChatContentPart*> image_parts;
    std::vector<ChatMessage> rendered = messages;
    bool any_mm = false;
    for (size_t i = 0; i < messages.size(); ++i) {
      if (!messages[i].content_parts.has_value()) continue;
      std::string text;
      int n_images = 0;
      for (const auto& part : *messages[i].content_parts) {
        if (part.type == "text") {
          text += part.text;
        } else if (part.type == "image_url") {
          ++n_images;
          image_parts.push_back(&part);
          any_mm = true;
        } else {
          any_mm = true;
        }
      }
      // Build text content with per-image placeholder. The Gemma4
      // chat template inserts a <IMG_TOKEN> per image.
      std::string content;
      for (int j = 0; j < n_images; ++j) {
        content += "<IMG_TOKEN>";
      }
      content += text;
      rendered[i].content = content;
      rendered[i].content_parts.reset();
    }

    if (!any_mm) return std::nullopt;

    // Refuse audio/video.
    if (HasAudioOrVideoParts(messages)) {
      throw vllm::v1::InputValidationError(
          "Gemma4 multimodal chat: audio and video processing in the chat "
          "path is not yet implemented. Only image_url content parts are "
          "supported.");
    }

    // Render the prompt via the server's chat template.
    const std::string prompt =
        prompt_fn(rendered, /*add_generation_prompt=*/true, {}, json::object());

    // Tokenize the rendered prompt.
    const std::vector<int32_t> prompt_ids =
        tokenizer.EncodeWithSpecialTokens(prompt);

    // Decode and process each image.
    multimodal::Gemma4ImageProcessor processor;
    std::vector<multimodal::Gemma4ImageFeatures> features;
    std::vector<DecodedImageRgb> decoded;
    for (const ChatContentPart* part : image_parts) {
      DecodedMedia media = DecodeImageUrlPart(*part);
      DecodedImageRgb rgb = codec(media);
      auto feat = processor.ProcessImageRGB(
          std::span<const uint8_t>(rgb.rgb.data(), rgb.rgb.size()),
          rgb.height, rgb.width);
      features.push_back(std::move(feat));
      decoded.push_back(std::move(rgb));
    }

    // Expand placeholders: replace each image_token_id in prompt_ids with
    // [boi] + [image_token] * N + [eoi], where N = num_soft_tokens.
    std::vector<int32_t> expanded;
    std::vector<multimodal::MultiModalFeatureSpec> specs;
    size_t img_idx = 0;
    for (size_t i = 0; i < prompt_ids.size(); ++i) {
      if (prompt_ids[i] == image_token_id) {
        if (img_idx >= features.size()) {
          throw std::invalid_argument(
              "Gemma4 mm chat: more image_token placeholders than images");
        }
        const auto& feat = features[img_idx];
        const int n = static_cast<int>(feat.num_soft_tokens);

        int offset = static_cast<int>(expanded.size());
        expanded.push_back(boi_id);
        for (int j = 0; j < n; ++j) {
          expanded.push_back(image_token_id);
        }
        expanded.push_back(eoi_id);

        multimodal::MultiModalFeatureSpec spec;
        const auto& rgb = decoded[img_idx];
        spec.mm_hash = multimodal::MultiModalHasher::HashImageRGB(
            model_id, rgb.rgb.data(), rgb.height, rgb.width);
        spec.modality = "image";
        spec.offset = offset + 1;  // skip boi
        spec.length = n;
        spec.data =
            std::make_shared<multimodal::ImageKwargs>(feat.patches);
        specs.push_back(std::move(spec));
        ++img_idx;
      } else {
        expanded.push_back(prompt_ids[i]);
      }
    }

    multimodal::MultiModalInputs result;
    result.prompt_token_ids = std::move(expanded);
    result.mm_features = std::move(specs);
    // Every decoded image must have a matching placeholder in the prompt.
    if (img_idx != features.size()) {
      throw std::invalid_argument(
          "Gemma4 mm chat: fewer image_token placeholders than images — "
          "the chat template did not expand all image parts");
    }
    return result;
  };

  return seam;
}

}  // namespace

REGISTER_VLLM_MM_CHAT(gemma4, "Gemma4ForConditionalGeneration",
                       &MakeGemma4ChatSeam)

}  // namespace vllm::entrypoints::openai
