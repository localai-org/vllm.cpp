// Tests for the Gemma4 mm chat seam image path (T10B).
// Exercises the full serving path: chat messages → rendered prompt →
// tokenized → image decoded → image processed → placeholders expanded →
// MultiModalInputs built.

#include <doctest/doctest.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/multimodal/gemma4_image_processor.h"
#include "vllm/config/multimodal.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"

namespace oai = vllm::entrypoints::openai;
using json = nlohmann::ordered_json;

namespace {

// Gemma4 special token IDs — small test values that fit within
// the test tokenizer's vocab range.
constexpr int32_t kImageToken = 302;
constexpr int32_t kBoiToken = 300;
constexpr int32_t kEoiToken = 301;

// Build a minimal tokenizer that has the Gemma4 special tokens as added
// tokens, so EncodeWithSpecialTokens produces the correct placeholder IDs.
vllm::tok::Tokenizer BuildTokenizer() {
  const std::string path =
      (std::filesystem::temp_directory_path() / "vllm_gemma4_mmchat_tok.json")
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array({
      {{"id", 300}, {"content", "<start_of_image>"}, {"special", true}},
      {{"id", 301}, {"content", "<end_of_image>"}, {"special", true}},
      {{"id", 302}, {"content", "<IMG_TOKEN>"}, {"special", true}},
      {{"id", 303}, {"content", "<bos>"}, {"special", true}},
      {{"id", 304}, {"content", "<eos>"}, {"special", true}},
  });
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {{"type", "Metaspace"}, {"replacement", "_"},
                          {"prepend_scheme", "always"}, {"split", true}};
  json vocab_arr = json::array();
  vocab_arr.push_back({"_", -3.0});  // Metaspace replacement char
  vocab_arr.push_back({"user", -1.0});
  vocab_arr.push_back({"assistant", -1.0});
  vocab_arr.push_back({"what", -1.0});
  vocab_arr.push_back({"is", -1.0});
  vocab_arr.push_back({"this", -1.0});
  vocab_arr.push_back({"?", -1.0});
  for (int b = 0; b < 256; ++b) {
    char buf[8];
    std::snprintf(buf, sizeof(buf), "<0x%02X>", b);
    vocab_arr.push_back({std::string(buf), 0.0});
  }
  doc["model"] = {{"type", "Unigram"}, {"byte_fallback", true},
                  {"vocab", vocab_arr}, {"merges", json::array()}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  auto tok = vllm::tok::Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

const vllm::tok::Tokenizer& Tok() {
  static const auto tok = BuildTokenizer();
  return tok;
}

// Build a raw-RGB data URI for a solid-color square image.
std::string RawRgbDataUri(int64_t side, int seed) {
  std::vector<uint8_t> rgb(static_cast<size_t>(side * side * 3));
  for (size_t i = 0; i < rgb.size(); i += 3) {
    rgb[i] = static_cast<uint8_t>((seed * 37 + i) & 0xFF);
    rgb[i + 1] = static_cast<uint8_t>((seed * 53 + i) & 0xFF);
    rgb[i + 2] = static_cast<uint8_t>((seed * 71 + i) & 0xFF);
  }
  static const char kB64[] =
      "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::string out;
  for (size_t i = 0; i < rgb.size(); i += 3) {
    uint32_t v = (static_cast<uint32_t>(rgb[i]) << 16) |
                 (i + 1 < rgb.size() ? static_cast<uint32_t>(rgb[i + 1]) << 8 : 0) |
                 (i + 2 < rgb.size() ? static_cast<uint32_t>(rgb[i + 2]) : 0);
    out.push_back(kB64[(v >> 18) & 63]);
    out.push_back(kB64[(v >> 12) & 63]);
    out.push_back(i + 1 < rgb.size() ? kB64[(v >> 6) & 63] : '=');
    out.push_back(i + 2 < rgb.size() ? kB64[v & 63] : '=');
  }
  return "data:image/x-raw-rgb;base64," + out;
}

oai::ChatContentPart TextPart(const std::string& text) {
  oai::ChatContentPart p;
  p.type = "text";
  p.text = text;
  return p;
}

oai::ChatContentPart ImagePart(int64_t side, int seed) {
  oai::ChatContentPart p;
  p.type = "image_url";
  p.url = RawRgbDataUri(side, seed);
  return p;
}

oai::ChatMessage UserWith(std::vector<oai::ChatContentPart> parts) {
  oai::ChatMessage m;
  m.role = "user";
  m.content_parts = std::move(parts);
  return m;
}

struct Ctx {
  vllm::HfConfig config;
  vllm::MultiModalConfig mm_config;
  oai::MultiModalChatContext ctx;
  std::shared_ptr<std::vector<std::string>> seen =
      std::make_shared<std::vector<std::string>>();

  Ctx() {
    // Load the real Gemma4 config fixture, then override the special
    // token IDs with small test values that the test tokenizer supports.
    auto real = vllm::LoadHfConfig(
        std::string(GEMMA4_E4B_FIXTURE_DIR) + "/gemma4_e4b_google/config.json");
    // Override token IDs — the Gemma4SpecialTokenIds constructor reads
    // from config.raw. Real IDs are in the 256K range; use small IDs
    // that the test tokenizer can handle.
    real.raw["image_token_id"] = 302;
    real.raw["boi_token_id"] = 300;
    real.raw["eoi_token_id"] = 301;
    config = real;

    ctx.architecture = "Gemma4ForConditionalGeneration";
    ctx.model_dir = std::string(GEMMA4_E4B_FIXTURE_DIR) + "/gemma4_e4b_google";
    ctx.config_path = ctx.model_dir + "/config.json";
    ctx.served_model_name = "gemma4-e4b";
    ctx.tokenizer = &Tok();
    auto rec = seen;
    ctx.prompt_fn = [rec](const std::vector<oai::ChatMessage>& msgs, bool,
                          const std::vector<oai::ChatCompletionToolsParam>&,
                          const json&) -> std::string {
      std::string out;
      for (const auto& m : msgs) {
        rec->push_back(m.content.value_or(""));
        out += "<bos>" + m.role + " " + m.content.value_or("") + "<eos>";
      }
      out += "<bos>assistant";
      return out;
    };
    ctx.codec = oai::DefaultImageCodec();
    ctx.config = &config;
    ctx.mm_config = &mm_config;
    ctx.max_model_len = 16384;
  }
};

}  // namespace

// T10B RED: the seam processes an image and returns MultiModalInputs.
// Before the implementation, MakeSeam returns a refusing seam whose
// chat_fn throws InputValidationError on any multimodal input.
TEST_CASE("gemma4 mm chat: image is processed and placeholders expanded") {
  Ctx c;
  const oai::MultiModalChatSeam seam = oai::MultiModalChatRegistry::MakeSeam(c.ctx);
  REQUIRE(seam.chat_fn);

  const auto out = seam.chat_fn(
      {UserWith({TextPart("what is this?"), ImagePart(48, 1)})});
  REQUIRE(out.has_value());
  // One image → one feature spec.
  REQUIRE(out->mm_features.size() == 1);
  const auto& f = out->mm_features[0];
  CHECK(f.modality == "image");
  CHECK(f.length > 0);
  // The expanded prompt must contain boi before the image tokens.
  CHECK(out->prompt_token_ids[static_cast<size_t>(f.offset - 1)] == kBoiToken);
  // Every token in the feature range must be image_token.
  for (int i = 0; i < f.length; ++i) {
    CHECK(out->prompt_token_ids[static_cast<size_t>(f.offset + i)] == kImageToken);
  }
  // The token after the feature range must be eoi.
  CHECK(out->prompt_token_ids[static_cast<size_t>(f.offset + f.length)] ==
        kEoiToken);
}

// T10B: text-only requests still pass through unchanged.
TEST_CASE("gemma4 mm chat: text-only passes through after image upgrade") {
  Ctx c;
  const oai::MultiModalChatSeam seam = oai::MultiModalChatRegistry::MakeSeam(c.ctx);
  const auto out = seam.chat_fn({UserWith({TextPart("hello")})});
  CHECK_FALSE(out.has_value());
}
