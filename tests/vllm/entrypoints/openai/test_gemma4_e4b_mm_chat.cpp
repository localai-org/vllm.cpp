// Tests for the Gemma4 mm chat seam registration
// (mm_chat_gemma4.cpp), entered through
// `MultiModalChatRegistry::Find` and `MakeSeam`.
//
// T10A registers the seam. The seam refuses multimodal requests with
// a clear message naming the architecture. Text-only requests pass
// through unchanged.

#include <doctest/doctest.h>

#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/transformers_utils/hf_config.h"

namespace oai = vllm::entrypoints::openai;

namespace {

oai::ChatContentPart TextPart(const std::string& text) {
  oai::ChatContentPart p;
  p.type = "text";
  p.text = text;
  return p;
}

oai::ChatContentPart ImagePart() {
  oai::ChatContentPart p;
  p.type = "image_url";
  p.url = "data:image/png;base64,iVBOR";
  return p;
}

oai::ChatMessage UserWith(std::vector<oai::ChatContentPart> parts) {
  oai::ChatMessage m;
  m.role = "user";
  m.content_parts = std::move(parts);
  return m;
}

}  // namespace

// T10A: the seam is registered for the architecture.
TEST_CASE("gemma4 mm chat: seam is registered for Gemma4ForConditionalGeneration") {
  const oai::MultiModalChatRegistration* reg =
      oai::MultiModalChatRegistry::Find("Gemma4ForConditionalGeneration");
  REQUIRE(reg != nullptr);
  CHECK(reg->make_seam != nullptr);
}

// T10A: the factory refuses when the context is incomplete (no tokenizer/config).
TEST_CASE("gemma4 mm chat: factory throws on incomplete context") {
  oai::MultiModalChatContext ctx;
  ctx.architecture = "Gemma4ForConditionalGeneration";
  // tokenizer and config are null — the factory validates and throws.
  REQUIRE_THROWS_AS(oai::MultiModalChatRegistry::MakeSeam(ctx),
                     std::runtime_error);
}

// T10A: the factory returns a seam with a non-null chat_fn when the context
// is valid (tokenizer and config present). Closes the mutation-3 coverage
// gap: without this test, setting seam.chat_fn = nullptr in the factory is
// not caught — the "refusing fn" test builds chat_fn directly via
// MakeRefusingMultiModalChatFn and never goes through the factory.
TEST_CASE("gemma4 mm chat: factory returns seam with non-null chat_fn") {
  oai::MultiModalChatContext ctx;
  ctx.architecture = "Gemma4ForConditionalGeneration";
  // Load the real config fixture — the factory reads special token IDs
  // from config via Gemma4SpecialTokenIds.
  static const auto cfg = vllm::LoadHfConfig(
      std::string(GEMMA4_E4B_FIXTURE_DIR) + "/gemma4_e4b_google/config.json");
  int tok_sentinel = 0;
  ctx.tokenizer = reinterpret_cast<const vllm::tok::Tokenizer*>(&tok_sentinel);
  ctx.config = &cfg;
  ctx.prompt_fn = [](const std::vector<oai::ChatMessage>&, bool,
                      const std::vector<oai::ChatCompletionToolsParam>&,
                      const nlohmann::ordered_json&) -> std::string {
    return "";
  };
  ctx.codec = oai::DefaultImageCodec();

  auto seam = oai::MultiModalChatRegistry::MakeSeam(ctx);
  REQUIRE(seam.chat_fn != nullptr);

  // The seam must still refuse audio/video requests.
  oai::ChatContentPart audio_part;
  audio_part.type = "input_audio";
  audio_part.url = "data:audio/wav;base64,UklGRiQ=";
  REQUIRE_THROWS_AS(
      seam.chat_fn({UserWith({audio_part, TextPart("transcribe")})}),
      vllm::v1::InputValidationError);
}

// T10A: the refusing chat fn passes text through and refuses multimodal.
TEST_CASE("gemma4 mm chat: refusing fn text pass-through and mm refusal") {
  auto chat_fn = oai::MakeRefusingMultiModalChatFn(
      "Gemma4ForConditionalGeneration",
      "image, audio, and video processing in the chat path is not yet "
      "implemented");

  // Text-only: returns nullopt (pass-through).
  auto text_out = chat_fn({UserWith({TextPart("hello")})});
  CHECK_FALSE(text_out.has_value());

  // Multimodal: throws InputValidationError with architecture name.
  REQUIRE_THROWS_AS(
      chat_fn({UserWith({ImagePart(), TextPart("what is this?")})}),
      vllm::v1::InputValidationError);
  try {
    chat_fn({UserWith({ImagePart(), TextPart("what is this?")})});
  } catch (const vllm::v1::InputValidationError& e) {
    CHECK(std::string(e.what()).find("Gemma4") != std::string::npos);
  }
}
