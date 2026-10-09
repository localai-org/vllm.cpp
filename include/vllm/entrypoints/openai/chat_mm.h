// OpenAI chat multimodal content-part DECODE + ROUTE (ROAD-V1-MM serving W1).
//
// Ported from:
//   - vllm/entrypoints/chat_utils.py: MM_PARSER_MAP:1478 +
//     _parse_chat_message_content_mm_part:1524 (the content-part schema — the
//     PARSE half lives in protocol.cpp / ChatContentPart);
//   - vllm/multimodal/media.py + vllm/multimodal/utils.py:35-113 (the data: URI /
//     base64 media encode; we mirror the decode: data:{mimetype};base64,{b64}).
//
// This is the CPU-reachable half of wiring multimodal into the OpenAI server: the
// base64 / data-URI DECODE and the ROUTE of the decoded bytes through the EXISTING
// single-sequence mm processors (multimodal/qwen3vl_processor.h + audio_processor.h)
// to produce a MultiModalInputs (placeholder-expanded prompt ids + mm_features).
// It runs entirely on CPU with only the processor CONFIG (no model weights).
//
// Optional native PNG/JPEG decoding lives in image_codec.cpp and is shared by
// the HTTP and C API entrypoints. Architecture-specific processor/model hooks
// still determine whether an image can reach a model. HTTP(S) fetching remains
// unsupported; decoding a data URI never introduces outbound access.
#ifndef VLLM_ENTRYPOINTS_OPENAI_CHAT_MM_H_
#define VLLM_ENTRYPOINTS_OPENAI_CHAT_MM_H_

#include <cstdint>
#include <cstddef>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/multimodal/audio_processor.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/multimodal/processing/context.h"
#include "vllm/multimodal/qwen3vl_processor.h"

namespace vllm {
namespace tok {
class Tokenizer;  // vllm/tokenizer/tokenizer.h (the placeholder->id mapping)
}  // namespace tok
}  // namespace vllm

namespace vllm::entrypoints::openai {

// A decoded media payload: the declared MIME type plus the raw bytes.
struct DecodedMedia {
  std::string media_type;      // e.g. "image/png", "audio/wav" (empty if absent)
  std::vector<uint8_t> bytes;  // raw decoded bytes
};

// Compressed/container input and decoded RGB are bounded separately from the
// processor's processed-image area cap. These are native input admission limits.
inline constexpr std::size_t kMaxImageContainerBytes = 32 * 1024 * 1024;
inline constexpr uint64_t kMaxDecodedImagePixels = 16 * 1024 * 1024;
inline constexpr uint64_t kMaxDecodedImageExtent = 32768;

// Standard base64 decode (RFC 4648 alphabet, '=' padding). ASCII whitespace
// (space/tab/CR/LF) is ignored so wrapped payloads decode. Throws
// std::runtime_error on an invalid character or a truncated group.
std::vector<uint8_t> DecodeBase64(const std::string& b64);

// Decode an RFC 2397 data: URI — data:[<mediatype>][;base64],<payload>. Only the
// ;base64 form is supported (the shape OpenAI clients emit); throws on a
// non-`data:` URI (an http(s) fetch is a named residual) or a non-base64 data URI.
DecodedMedia DecodeDataUri(const std::string& uri);

// Whether a chat message carries any non-text (multimodal) content part.
bool HasMultiModalParts(const ChatMessage& m);

// Decode an image_url content part's `url` (a data: URI) to its raw bytes. The
// bytes are the container-format payload; DefaultImageCodec produces RGB and
// dimensions. Invalid/overlimit image URLs raise InputValidationError (HTTP 400).
DecodedMedia DecodeImageUrlPart(const ChatContentPart& part);

// Decode an input_audio content part's base64 `data` to raw bytes (a container
// such as WAV). audio_url data: URIs go through DecodeDataUri instead.
DecodedMedia DecodeInputAudioPart(const ChatContentPart& part);

// Route a decoded PCM16 WAV through the EXISTING Whisper audio processor to
// produce the mm inputs + placeholder-expanded prompt, exactly as the single-
// sequence path does (DecodeWavPcm16Mono -> ProcessWaveform ->
// ExpandAudioPlaceholders). `prompt_ids` is the pre-tokenized prompt carrying one
// audio_placeholder_id per audio item (the tokenizer text path is out of scope,
// mirroring the M2c/A-track e2e fixtures). Throws on a non-PCM16/non-mono WAV.
multimodal::MultiModalInputs RouteAudioWav(
    const multimodal::WhisperAudioProcessor& proc, const DecodedMedia& audio,
    const std::vector<int32_t>& prompt_ids);

// Route already-decoded RGB image bytes (HWC uint8, height*width*3) through the
// EXISTING Qwen3-VL image processor (ProcessImage -> ExpandImagePlaceholders).
// DefaultImageCodec supplies the container decode before this processor call.
// `prompt_ids` carries one image_token_id per image item.
multimodal::MultiModalInputs RouteImageRgb(
    const multimodal::Qwen3VLImageProcessor& proc, const uint8_t* rgb,
    int64_t height, int64_t width, const std::vector<int32_t>& prompt_ids);

// ── Chat-template placeholder insertion (MM-SERVE-ENGINE) ──────────────────
//
// Ported from: vllm/entrypoints/chat_utils.py:886 `_add_placeholder` +
// :627 `get_placeholder_str`, and the per-model markers
// vllm/model_executor/models/qwen3_vl.py:1714 (image/video) +
// qwen2_audio.py:333 (audio). When a chat request carries a multimodal content
// part, vLLM inserts the model's placeholder STRING into the templated prompt at
// the part position; the tokenizer then maps it to ONE placeholder token, which
// the mm processor EXPANDS to N (= the grid/feature count) — the token-level
// expansion our RouteImageRgb / RouteAudioWav already perform
// (ExpandImagePlaceholders / ExpandAudioPlaceholders). These helpers own the
// STRING-level marker (the serving-layer half of the mirror); turning the marker
// into token ids needs the model tokenizer (MM-SERVE-E2E).

// The Qwen3-VL IMAGE placeholder marker: "<|vision_start|><|image_pad|><|vision_end|>"
// (qwen3_vl.py:1716). The single <|image_pad|> is what ExpandImagePlaceholders
// replaces with prod(grid_thw)/merge^2 copies of image_token_id.
std::string ImagePlaceholderString();

// The Qwen3-VL VIDEO placeholder marker: "<|vision_start|><|video_pad|><|vision_end|>"
// (qwen3_vl.py:1718).
std::string VideoPlaceholderString();

// The Qwen2-Audio placeholder marker: "Audio {i}: <|audio_bos|><|AUDIO|><|audio_eos|>"
// (qwen2_audio.py:335). `index` is the 1-based audio item number vLLM formats in.
std::string AudioPlaceholderString(int index);

// The placeholder marker for ONE parsed content part (get_placeholder_str
// dispatch): image_url -> ImagePlaceholderString; input_audio / audio_url ->
// AudioPlaceholderString(audio_index); "text" (and unrouted residual kinds) ->
// empty. `audio_index` is the running 1-based audio counter (Qwen numbers audios).
std::string ChatPlaceholderFor(const ChatContentPart& part, int audio_index);

// Collect the placeholder markers for a message's mm parts IN ORDER (mirrors the
// per-message mm_placeholder_storage, chat_utils.py:891). Text parts contribute
// no marker; audio parts are numbered 1..k. Empty when the message has no mm
// parts (a bare-string content is byte-identical — content_parts is nullopt).
std::vector<std::string> CollectChatPlaceholders(const ChatMessage& message);

// Build the marker-INJECTED content string for one message: walk its
// content_parts IN ORDER, appending each text part's text and each mm part's
// placeholder MARKER (ChatPlaceholderFor) at its position. Mirrors vLLM
// interleaving the placeholder string into the message content at the mm part
// offset (chat_utils.py:886 `_add_placeholder`); the single <|image_pad|> in the
// marker is what the tokenizer maps to ONE image_token_id, which the mm processor
// then EXPANDS to N. A bare-string message (content_parts nullopt) returns its
// content unchanged (byte-identical text path).
std::string BuildMarkerInjectedContent(const ChatMessage& message);

// ── The per-item LIMIT check on the chat path (#607 wave L2, #686) ─────────
//
// Ported from: vllm/entrypoints/chat_utils.py:630-662 — the tracker that counts
// every multimodal content part it parses and validates the RUNNING count
// against the configured limit before the request reaches the engine.
// `BaseProcessingInfo::ValidateTrackedChatItem` (#607 L1) is the check itself;
// what follows is the WALK that reaches it, which is the half L1 deliberately
// left out because nothing constructed a config on a live request yet.

// The tracked mm modality for one content part — the modality
// `validate_num_items` is asked about, or nullopt for a part that is not
// multimodal input at all. Mirrors MM_PARSER_MAP (chat_utils.py:1478) reduced to
// the part types this server's protocol.h parses:
//   image_url             -> "image"      (chat_utils.py:1479)
//   video_url             -> "video"      (chat_utils.py:1483)
//   input_audio/audio_url -> "audio"      (chat_utils.py:1481-1482)
//   *_embeds              -> passed THROUGH verbatim, because
//                            ValidateTrackedChatItem owns the suffix strip and
//                            the enable_mm_embeds escape (chat_utils.py:635,653-660)
//   text (and anything else) -> nullopt
std::optional<std::string> ChatPartModality(const ChatContentPart& part);

// The WALK (chat_utils.py:648-662): every mm content part of every message, IN
// ORDER, maintaining the running per-modality count and validating each item as
// it is counted. The count is cumulative ACROSS messages, exactly as upstream's
// tracker is, so a limit cannot be evaded by splitting the items over two turns.
//
// Throws vllm::v1::InputValidationError — the type api_server.cpp:185,252 maps
// to HTTP 400 — carrying upstream's own message text. This is what turns the
// silent truncation of #686 into upstream's refusal: before it, chat_mm.cpp took
// the FIRST image part and dropped the rest without a word.
void ValidateChatMmLimits(const multimodal::BaseProcessingInfo& info,
                          const std::vector<ChatMessage>& messages);

// The Qwen3-VL IMAGE chat seam's OWN supported limits — the `min()` fold's other
// operand (context.py:392-405), which #686 recorded as undeclared.
//
// Upstream's Qwen3-VL declares image and video UNLIMITED —
// `get_supported_mm_limits` is not defined on Qwen3VLProcessingInfo at all
// (qwen3_vl.py:848 subclasses Qwen2VLProcessingInfo); it is INHERITED from
// qwen2_vl.py:851-852, `return {"image": None, "video": None}` — because its
// processor handles N of each. Ours handles exactly ONE image
// (MakeQwen3VLImageChatFn locates a single image part) and no video or audio at
// all, so the honest ceiling is {"image": 1} and every other modality is
// ABSENT — which context.py:414-415 reads as "not supported", limit 0. That is
// not a policy choice, it is this seam's implemented arm stated as a NUMBER. A
// user limit can only LOWER it (the fold is a min), so
// `--limit-mm-per-prompt image=99` still refuses the second image.
//
// WHAT THIS DOES NOT YET SATISFY (#758, found in the #749 review). AGENTS.md
// asks that an unimplemented arm be "refused with a message naming the missing
// piece". The number is stated here, but the message a client receives is
// upstream's generic "At most 0 video(s) may be provided in one prompt." — which
// names nothing, and which a user cannot tell apart from an operator having set
// `--limit-mm-per-prompt '{"video": 0}'`. The only signal today is by OMISSION:
// ValidateNumItems withholds the "Set `--limit-mm-per-prompt` to increase this
// limit." hint when raising the configured limit would not help. Changing the
// text is a deliberate divergence from a verbatim-ported message that three
// suites assert byte-for-byte, so it is owed to #758 with its own spec rather
// than folded in here.
//
// When the multi-image / video arms land they raise these numbers here, and
// nothing else changes.
std::map<std::string, std::optional<int>> Qwen3VLChatSupportedMmLimits();

// ── The multimodal chat SEAM BODY (MM-SERVE-E2E) ───────────────────────────
//
// A decoded RGB image: raw HWC uint8 (height*width*3) and dimensions, after the
// reference HTTP media loader's orientation and default color conversion.
struct DecodedImageRgb {
  std::vector<uint8_t> rgb;
  int64_t height = 0;
  int64_t width = 0;
};

// Image codec seam: decoded media bytes -> raw RGB + dims. Throws on an
// unsupported/malformed/overlimit container with InputValidationError. Native
// PNG/JPEG support is optional at build time; diagnostic raw RGB remains usable.
using ImageCodecFn = std::function<DecodedImageRgb(const DecodedMedia&)>;

// THE PRODUCTION CODEC, and there is exactly one.
//
// VLLM_CPP_IMAGE_CODECS=ON selects system libpng/libjpeg. The default OFF build
// refuses PNG/JPEG by name; raw RGB still works with the same bounded inputs.
//
// It lives here rather than as a lambda inside `server_main.cpp` because it is
// not the server BINARY's, it is the LIBRARY's: `vllm_chat` installs the same
// multimodal chat seam through `include/vllm.h` and has to hand every factory
// the identical codec, or one entry point of one library would accept an image
// the other refuses. It belongs to neither of them and to no architecture,
// which is why every `MultiModalChatContext` carries it rather than growing one
// per model.
ImageCodecFn DefaultImageCodec();

// The chat-prompt renderer seam (structurally IDENTICAL to serving_chat.h
// ChatPromptFn — kept local so chat_mm.h need not pull serving_chat.h). The
// server's real chat-template renderer (MakeChatTemplatePromptFn) plugs in here.
using ChatPromptRenderFn = std::function<std::string(
    const std::vector<ChatMessage>&, bool,
    const std::vector<ChatCompletionToolsParam>&,
    const nlohmann::ordered_json&)>;

// Build the Qwen3-VL IMAGE multimodal chat seam body (the MultiModalChatFn the
// server sets via set_multimodal_chat_fn). The returned function turns chat
// `messages` into the engine's placeholder-EXPANDED MultiModalInputs:
//   1. inject the image placeholder marker at each image part's position
//      (BuildMarkerInjectedContent) and render the templated prompt via
//      `prompt_fn` (the real chat template);
//   2. tokenize the rendered prompt WITH special tokens — the tokenizer maps the
//      single <|image_pad|> marker to ONE `proc.config().image_token_id`
//      (tokenizer.h EncodeWithSpecialTokens, added tokens matched
//      leftmost-longest);
//   3. decode/preprocess each image serially, expand all marker ids to their
//      ordered grid/merge^2 spans and retain every image's patch owner/hash in
//      the mm_features the engine carries onto Request.mm_features.
// Returns nullopt when no message carries an image part (the text path stays
// byte-identical). `proc`, `tokenizer` and `info` must outlive the returned
// function (the server owns them for the process lifetime, like
// set_beam_search_tokenizer).
//
// STEP 0, ahead of everything above (#607 L2, #686): ValidateChatMmLimits(info,
// messages). IMAGE-only is no longer a silent truncation — a request carrying
// more images than `info` allows, or any video/audio part, is REFUSED with
// upstream's message and reaches the client as HTTP 400. `info`'s supported
// limits are model-scoped (legacy Qwen3VLChatSupportedMmLimits is one image,
// native Qwen3.5 supplies two); its MultiModalConfig
// is where `--limit-mm-per-prompt` / `--language-model-only` land, so
// --language-model-only makes this seam answer every image request with
// "At most 0 image(s) may be provided in one prompt."
//
// This is the seam-body half of MM-SERVE-E2E: it produces the token-correct
// engine input; the GPU worker consuming Request.mm_features through the vision
// tower + merge + MRoPE/DeepStack forward is the remaining residual (the engine
// model runner has no mm-forward path yet — the M2c Qwen3VLGenerateGreedy driver
// runs it standalone, outside ModelRegistry::Forward).
std::function<std::optional<multimodal::MultiModalInputs>(
    const std::vector<ChatMessage>&)>
MakeQwen3VLImageChatFn(const multimodal::Qwen3VLImageProcessor& proc,
                       const vllm::tok::Tokenizer& tokenizer,
                       ChatPromptRenderFn prompt_fn, ImageCodecFn codec,
                       const multimodal::BaseProcessingInfo& info);

// Same image processor/limits, preserving per-request tools/template options.
// The messages-only function above wraps this with empty options for existing
// direct callers; production Qwen chat seams install this form.
std::function<std::optional<multimodal::MultiModalInputs>(
    const std::vector<ChatMessage>&,
    const std::vector<ChatCompletionToolsParam>&,
    const nlohmann::ordered_json&)>
MakeQwen3VLImageRequestChatFn(const multimodal::Qwen3VLImageProcessor& proc,
                            const vllm::tok::Tokenizer& tokenizer,
                            ChatPromptRenderFn prompt_fn, ImageCodecFn codec,
                            const multimodal::BaseProcessingInfo& info);

}  // namespace vllm::entrypoints::openai

#endif  // VLLM_ENTRYPOINTS_OPENAI_CHAT_MM_H_
