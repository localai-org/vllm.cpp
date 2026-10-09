// ROAD-V1-MM serving W1 — OpenAI chat multimodal content-part DECODE + ROUTE.
// See include/vllm/entrypoints/openai/chat_mm.h for the ported-from map and the
// named residuals.
#include "vllm/entrypoints/openai/chat_mm.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/engine/input_processor.h"

namespace vllm::entrypoints::openai {

namespace {

// RFC 4648 reverse alphabet: value 0..63 for a base64 char, -1 invalid, -2 skip
// (ASCII whitespace), -3 padding '='.
int Base64Value(unsigned char c) {
  if (c >= 'A' && c <= 'Z') return c - 'A';
  if (c >= 'a' && c <= 'z') return c - 'a' + 26;
  if (c >= '0' && c <= '9') return c - '0' + 52;
  if (c == '+') return 62;
  if (c == '/') return 63;
  if (c == '=') return -3;
  if (c == ' ' || c == '\t' || c == '\r' || c == '\n') return -2;
  return -1;
}

}  // namespace

std::vector<uint8_t> DecodeBase64(const std::string& b64) {
  std::vector<uint8_t> out;
  out.reserve((b64.size() / 4) * 3);
  int quad[4];
  int n = 0;         // how many real (non-skip) symbols collected in this group
  int pad = 0;       // padding chars seen in this group
  bool ended = false;
  for (const char ch : b64) {
    const int v = Base64Value(static_cast<unsigned char>(ch));
    if (v == -2) continue;  // whitespace
    if (v == -1) {
      throw std::runtime_error("DecodeBase64: invalid base64 character");
    }
    if (ended) {
      // Data after a completed padded group is malformed.
      throw std::runtime_error("DecodeBase64: trailing data after padding");
    }
    if (v == -3) {  // padding
      if (n < 2) throw std::runtime_error("DecodeBase64: misplaced padding");
      quad[n++] = 0;
      ++pad;
    } else {
      if (pad > 0) throw std::runtime_error("DecodeBase64: data after padding");
      quad[n++] = v;
    }
    if (n == 4) {
      const uint32_t triple = (static_cast<uint32_t>(quad[0]) << 18) |
                              (static_cast<uint32_t>(quad[1]) << 12) |
                              (static_cast<uint32_t>(quad[2]) << 6) |
                              static_cast<uint32_t>(quad[3]);
      out.push_back(static_cast<uint8_t>((triple >> 16) & 0xFF));
      if (pad < 2) out.push_back(static_cast<uint8_t>((triple >> 8) & 0xFF));
      if (pad < 1) out.push_back(static_cast<uint8_t>(triple & 0xFF));
      n = 0;
      if (pad > 0) ended = true;
      pad = 0;
    }
  }
  if (n != 0) {
    throw std::runtime_error("DecodeBase64: truncated base64 group");
  }
  return out;
}

DecodedMedia DecodeDataUri(const std::string& uri) {
  // RFC 2397: data:[<mediatype>][;base64],<payload>
  static const std::string kScheme = "data:";
  if (uri.rfind(kScheme, 0) != 0) {
    throw std::runtime_error(
        "DecodeDataUri: not a data: URI (http(s) media fetch is a named "
        "residual)");
  }
  const std::string::size_type comma = uri.find(',');
  if (comma == std::string::npos) {
    throw std::runtime_error("DecodeDataUri: malformed data URI (no comma)");
  }
  // Header between "data:" and the comma: <mediatype>[;base64].
  const std::string header =
      uri.substr(kScheme.size(), comma - kScheme.size());
  const std::string payload = uri.substr(comma + 1);

  const std::string::size_type semi = header.find(';');
  DecodedMedia media;
  media.media_type = header.substr(0, semi);  // may be empty
  const bool is_base64 =
      header.size() >= 7 && header.compare(header.size() - 7, 7, ";base64") == 0;
  if (!is_base64) {
    throw std::runtime_error(
        "DecodeDataUri: only ;base64 data URIs are supported");
  }
  media.bytes = DecodeBase64(payload);
  return media;
}

bool HasMultiModalParts(const ChatMessage& m) {
  if (!m.content_parts.has_value()) return false;
  for (const ChatContentPart& p : *m.content_parts) {
    if (p.type != "text") return true;
  }
  return false;
}

DecodedMedia DecodeImageUrlPart(const ChatContentPart& part) {
  // Bound the encoded allocation BEFORE DecodeDataUri copies the payload and
  // DecodeBase64 reserves its output. No image URL triggers outbound I/O.
  constexpr std::size_t max_uri = ((kMaxImageContainerBytes + 2) / 3) * 4 + 256;
  if (part.url.size() > max_uri) {
    throw vllm::v1::InputValidationError("multimodal image: encoded input exceeds byte limit");
  }
  try {
    DecodedMedia media = DecodeDataUri(part.url);
    if (media.bytes.size() > kMaxImageContainerBytes) {
      throw vllm::v1::InputValidationError("multimodal image: container exceeds byte limit");
    }
    return media;
  } catch (const vllm::v1::InputValidationError&) {
    throw;
  } catch (const std::runtime_error& e) {
    throw vllm::v1::InputValidationError(e.what());
  }
}

DecodedMedia DecodeInputAudioPart(const ChatContentPart& part) {
  DecodedMedia media;
  media.media_type = part.audio_format.empty()
                         ? std::string()
                         : "audio/" + part.audio_format;
  media.bytes = DecodeBase64(part.audio_data);
  return media;
}

multimodal::MultiModalInputs RouteAudioWav(
    const multimodal::WhisperAudioProcessor& proc, const DecodedMedia& audio,
    const std::vector<int32_t>& prompt_ids) {
  const multimodal::AudioProcessorConfig& cfg = proc.config();

  const multimodal::DecodedAudio decoded =
      multimodal::DecodeWavPcm16Mono(audio.bytes.data(), audio.bytes.size());
  multimodal::AudioKwargs kw = proc.ProcessWaveform(
      decoded.samples.data(), static_cast<int64_t>(decoded.samples.size()),
      decoded.sampling_rate);

  const int num_audio_tokens = cfg.max_source_positions;
  std::vector<std::array<int, 2>> placeholders;
  std::vector<int32_t> expanded = multimodal::ExpandAudioPlaceholders(
      prompt_ids, cfg.audio_placeholder_id, {num_audio_tokens}, &placeholders);

  multimodal::MultiModalInputs out;
  out.prompt_token_ids = std::move(expanded);
  if (!placeholders.empty()) {
    multimodal::MultiModalFeatureSpec spec;
    spec.modality = "audio";
    spec.offset = placeholders[0][0];
    spec.length = placeholders[0][1];
    spec.audio_data = std::make_shared<multimodal::AudioKwargs>(std::move(kw));
    spec.mm_hash = proc.HashAudio(
        decoded.samples.data(), static_cast<int64_t>(decoded.samples.size()));
    out.mm_features.push_back(std::move(spec));
  }
  return out;
}

multimodal::MultiModalInputs RouteImageRgb(
    const multimodal::Qwen3VLImageProcessor& proc, const uint8_t* rgb,
    int64_t height, int64_t width, const std::vector<int32_t>& prompt_ids) {
  const multimodal::Qwen3VLProcessorConfig& cfg = proc.config();

  multimodal::ImageKwargs kw;
  try {
    kw = proc.ProcessImage(rgb, height, width);
  } catch (const std::invalid_argument& e) {
    throw vllm::v1::InputValidationError(e.what());
  }
  const std::array<int64_t, 3> grid = kw.image_grid_thw;

  std::vector<std::array<int64_t, 3>> grids{grid};
  std::vector<std::array<int, 2>> placeholders;
  std::vector<int32_t> expanded = multimodal::ExpandImagePlaceholders(
      prompt_ids, cfg.image_token_id, cfg.merge_size, grids, &placeholders);

  multimodal::MultiModalInputs out;
  out.prompt_token_ids = std::move(expanded);
  if (!placeholders.empty()) {
    multimodal::MultiModalFeatureSpec spec;
    spec.modality = "image";
    spec.offset = placeholders[0][0];
    spec.length = placeholders[0][1];
    spec.mm_hash = proc.HashImage(rgb, height, width);
    spec.data = std::make_shared<multimodal::ImageKwargs>(std::move(kw));
    out.mm_features.push_back(std::move(spec));
  }
  return out;
}

std::string ImagePlaceholderString() {
  // qwen3_vl.py:1716.
  return "<|vision_start|><|image_pad|><|vision_end|>";
}

std::string VideoPlaceholderString() {
  // qwen3_vl.py:1718.
  return "<|vision_start|><|video_pad|><|vision_end|>";
}

std::string AudioPlaceholderString(int index) {
  // qwen2_audio.py:335 — f"Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>".
  return "Audio " + std::to_string(index) +
         ": <|audio_bos|><|AUDIO|><|audio_eos|>";
}

std::string ChatPlaceholderFor(const ChatContentPart& part, int audio_index) {
  if (part.type == "image_url") return ImagePlaceholderString();
  if (part.type == "video_url") return VideoPlaceholderString();
  if (part.type == "input_audio" || part.type == "audio_url") {
    return AudioPlaceholderString(audio_index);
  }
  return std::string();  // "text" and unrouted residual kinds carry no marker
}

std::vector<std::string> CollectChatPlaceholders(const ChatMessage& message) {
  std::vector<std::string> markers;
  if (!message.content_parts.has_value()) return markers;
  int audio_index = 0;
  for (const ChatContentPart& part : *message.content_parts) {
    const bool is_audio =
        part.type == "input_audio" || part.type == "audio_url";
    if (is_audio) ++audio_index;
    std::string marker = ChatPlaceholderFor(part, audio_index);
    if (!marker.empty()) markers.push_back(std::move(marker));
  }
  return markers;
}

std::string BuildMarkerInjectedContent(const ChatMessage& message) {
  // Bare-string content: nothing to inject (byte-identical text path).
  if (!message.content_parts.has_value()) {
    return message.content.value_or(std::string());
  }
  std::string out;
  int audio_index = 0;
  for (const ChatContentPart& part : *message.content_parts) {
    if (part.type == "text") {
      out += part.text;
      continue;
    }
    const bool is_audio =
        part.type == "input_audio" || part.type == "audio_url";
    if (is_audio) ++audio_index;
    // The mm marker at THIS part's position (empty for unrouted residual kinds,
    // which then contribute nothing — matching the text-only fallback).
    out += ChatPlaceholderFor(part, audio_index);
  }
  return out;
}

// chat_utils.py:1478-1483 (MM_PARSER_MAP), reduced to the part types protocol.h
// parses. See the header for why `*_embeds` passes straight through.
std::optional<std::string> ChatPartModality(const ChatContentPart& part) {
  if (part.type == "image_url") return std::string("image");
  if (part.type == "video_url") return std::string("video");
  if (part.type == "input_audio" || part.type == "audio_url") {
    return std::string("audio");
  }
  // "text" is not multimodal input; anything else that names an embeds kind is
  // handed to ValidateTrackedChatItem verbatim so it owns the suffix strip.
  if (part.type.size() > 7 &&
      part.type.compare(part.type.size() - 7, 7, "_embeds") == 0) {
    return part.type;
  }
  return std::nullopt;
}

// chat_utils.py:648-662.
void ValidateChatMmLimits(const multimodal::BaseProcessingInfo& info,
                          const std::vector<ChatMessage>& messages) {
  // The running item count, keyed by the ORIGINAL (as-written) modality —
  // `self._items_by_modality[original_modality]` (chat_utils.py:625,652). Note
  // it is NOT keyed by the input modality: `image` and `image_embeds` keep
  // separate counters upstream, and only the LIMIT is looked up under the
  // stripped name (:633,662). Keying on the stripped name here would refuse a
  // request upstream serves.
  //
  // The map lives for the whole `messages` walk, not per message, because
  // upstream's tracker does: two images across two turns of one request are two
  // images, so the limit cannot be evaded by splitting them up.
  std::map<std::string, int> items_by_modality;
  for (const ChatMessage& message : messages) {
    if (!message.content_parts.has_value()) continue;
    for (const ChatContentPart& part : *message.content_parts) {
      const std::optional<std::string> modality = ChatPartModality(part);
      if (!modality.has_value()) continue;
      // The count INCLUDES this item (`len(...) + 1`, :652): upstream validates
      // the length the item WOULD make, so the refusal names the limit the
      // request crossed rather than the one it was already at.
      const int num_items = ++items_by_modality[*modality];
      info.ValidateTrackedChatItem(*modality, num_items);
    }
  }
}

std::map<std::string, std::optional<int>> Qwen3VLChatSupportedMmLimits() {
  return {{"image", std::optional<int>(1)}};
}

std::function<std::optional<multimodal::MultiModalInputs>(
    const std::vector<ChatMessage>&)>
MakeQwen3VLImageChatFn(const multimodal::Qwen3VLImageProcessor& proc,
                       const vllm::tok::Tokenizer& tokenizer,
                       ChatPromptRenderFn prompt_fn, ImageCodecFn codec,
                       const multimodal::BaseProcessingInfo& info) {
  auto body = MakeQwen3VLImageRequestChatFn(proc, tokenizer, std::move(prompt_fn), std::move(codec), info);
  return [body = std::move(body)](const std::vector<ChatMessage>& messages) {
    return body(messages, {}, nlohmann::ordered_json::object());
  };
}

std::function<std::optional<multimodal::MultiModalInputs>(
    const std::vector<ChatMessage>&,
    const std::vector<ChatCompletionToolsParam>&,
    const nlohmann::ordered_json&)>
MakeQwen3VLImageRequestChatFn(const multimodal::Qwen3VLImageProcessor& proc,
                            const vllm::tok::Tokenizer& tokenizer,
                            ChatPromptRenderFn prompt_fn, ImageCodecFn codec,
                            const multimodal::BaseProcessingInfo& info) {
  return [&proc, &tokenizer, &info, prompt_fn = std::move(prompt_fn),
          codec = std::move(codec)](const std::vector<ChatMessage>& messages,
              const std::vector<ChatCompletionToolsParam>& tools,
              const nlohmann::ordered_json& chat_template_kwargs)
             -> std::optional<multimodal::MultiModalInputs> {
    // STEP 0 (#607 L2, #686): the per-item limit check, BEFORE anything is
    // decoded or dropped. chat_utils.py:662 validates as it tracks, for the same
    // reason: refusing costs nothing, and truncating is invisible.
    ValidateChatMmLimits(info, messages);

    // Preserve conversation and content-part order. The processing info keeps
    // legacy Qwen3-VL at one image; native Qwen3.5 supplies its own ceiling.
    std::vector<const ChatContentPart*> image_parts;
    for (const ChatMessage& m : messages) {
      if (!m.content_parts.has_value()) continue;
      for (const ChatContentPart& part : *m.content_parts) {
        if (part.type == "image_url") {
          image_parts.push_back(&part);
        }
      }
    }
    if (image_parts.empty()) return std::nullopt;

    // 1. Render the templated prompt with the placeholder marker injected at the
    //    mm part position (the real chat template wraps <|im_start|>… around it).
    std::vector<ChatMessage> rendered = messages;
    for (ChatMessage& m : rendered) {
      if (m.content_parts.has_value()) {
        m.content = BuildMarkerInjectedContent(m);
        m.content_parts.reset();
      }
    }
    const std::string prompt =
        prompt_fn(rendered, /*add_generation_prompt=*/true, tools,
                  chat_template_kwargs);

    // 2. Tokenize WITH special tokens: each <|image_pad|> marker becomes one
    //    image_token_id (added tokens matched leftmost-longest).
    const std::vector<int32_t> prompt_ids =
        tokenizer.EncodeWithSpecialTokens(prompt);

    // 3. Preprocess serially so decoded RGB buffers do not accumulate beside
    //    the retained typed patches. Expand all markers together, then pair
    //    each ordered grid/hash/patch owner with its exact placeholder span.
    multimodal::MultiModalInputs out;
    std::vector<std::array<int64_t, 3>> grids;
    for (const auto* part : image_parts) {
      const DecodedMedia media = DecodeImageUrlPart(*part);
      const DecodedImageRgb img = codec(media);
      multimodal::MultiModalFeatureSpec feature;
      feature.modality = "image";
      try {
        feature.data = std::make_shared<multimodal::ImageKwargs>(
            proc.ProcessImage(img.rgb.data(), img.height, img.width));
      } catch (const std::invalid_argument& e) {
        throw vllm::v1::InputValidationError(e.what());
      }
      feature.mm_hash = proc.HashImage(img.rgb.data(), img.height, img.width);
      grids.push_back(feature.data->image_grid_thw);
      out.mm_features.push_back(std::move(feature));
    }
    std::vector<std::array<int, 2>> placeholders;
    out.prompt_token_ids = multimodal::ExpandImagePlaceholders(
        prompt_ids, proc.config().image_token_id, proc.config().merge_size,
        grids, &placeholders);
    for (size_t i = 0; i < out.mm_features.size(); ++i) {
      out.mm_features[i].offset = placeholders[i][0];
      out.mm_features[i].length = placeholders[i][1];
    }
    return out;
  };
}

}  // namespace vllm::entrypoints::openai
