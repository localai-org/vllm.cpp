// The Nemotron Nano VL / Omni multimodal CHAT seam
// (mm_chat_nano_nemotron_vl.cpp), entered through
// `MultiModalChatRegistry::MakeSeam` with the context the server fills in:
//   * the registration exists for both released architecture names;
//   * a message with image parts is rendered with the released template's
//     image prefix ("<image>\n", or "<image 1><image> <image 2><image>\n");
//   * each `<image>` is expanded to `<img>` + `<image>` x N + `</img>` with N
//     from the dynamic tiler, and one feature per image carries the patch grid;
//   * a text-only conversation passes through (nullopt);
//   * a context without the engine's max_model_len is refused at install.
#include <doctest/doctest.h>

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <functional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/config/multimodal.h"
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/entrypoints/openai/protocol.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/multimodal/nano_nemotron_vl_processor.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"

namespace oai = vllm::entrypoints::openai;

namespace {

using json = nlohmann::ordered_json;

constexpr const char* kArch = "NemotronH_Nano_Omni_Reasoning_V3";
constexpr int32_t kBos = 2, kImage = 18, kImgStart = 19, kImgEnd = 20;

json ByteLevelPreTokenizer() {
  return json{
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
}

// The released ids of the three image tokens (tokenizer_config.json @
// e5e99324: 18 <image>, 19 <img>, 20 </img>), plus a few letters.
//
// `with_bos` adds a post_processor that PREPENDS a BOS, the one tokenizer shape
// on which HF `add_special_tokens=False` and `add_special_tokens=True` give
// different lengths for the same text.
vllm::tok::Tokenizer BuildTokenizer(bool with_bos = false) {
  const std::string path =
      (std::filesystem::temp_directory_path() /
       (with_bos ? "vllm_nnvl_mmchat_tok_bos.json" : "vllm_nnvl_mmchat_tok.json"))
          .string();
  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array({
      {{"id", 0}, {"content", "<|im_start|>"}, {"special", true}},
      {{"id", 1}, {"content", "<|im_end|>"}, {"special", true}},
      {{"id", kBos}, {"content", "<s>"}, {"special", true}},
      {{"id", kImage}, {"content", "<image>"}, {"special", true}},
      {{"id", kImgStart}, {"content", "<img>"}, {"special", true}},
      {{"id", kImgEnd}, {"content", "</img>"}, {"special", true}},
  });
  if (with_bos) {
    doc["post_processor"] = {
        {"type", "TemplateProcessing"},
        {"single", json::array({{{"SpecialToken", {{"id", "<s>"}, {"type_id", 0}}}},
                                {{"Sequence", {{"id", "A"}, {"type_id", 0}}}}})},
        {"pair", json::array({{{"SpecialToken", {{"id", "<s>"}, {"type_id", 0}}}},
                              {{"Sequence", {{"id", "A"}, {"type_id", 0}}}},
                              {{"Sequence", {{"id", "B"}, {"type_id", 1}}}}})},
        {"special_tokens",
         {{"<s>", {{"id", "<s>"}, {"ids", json::array({kBos})}, {"tokens", json::array({"<s>"})}}}}}};
  }
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = ByteLevelPreTokenizer();
  // The whole byte-level alphabet, so any rendered prompt encodes.
  json vocab = json::object();
  int id = 30;
  for (int b = 0; b < 256; ++b) {
    vocab[vllm::tok::MapBytesToUnicode(std::string(1, static_cast<char>(b)))] = id++;
  }
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array()}};
  std::ofstream(path, std::ios::binary) << doc.dump();
  vllm::tok::Tokenizer tok = vllm::tok::Tokenizer::FromHfJson(path);
  std::remove(path.c_str());
  return tok;
}

const vllm::tok::Tokenizer& Tok() {
  static const vllm::tok::Tokenizer t = BuildTokenizer();
  return t;
}

const vllm::tok::Tokenizer& TokWithBos() {
  static const vllm::tok::Tokenizer t = BuildTokenizer(/*with_bos=*/true);
  return t;
}

std::string RawRgbDataUri(int64_t side, int seed) {
  static const char* kB64 = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
  std::vector<uint8_t> rgb(static_cast<size_t>(side * side * 3));
  for (size_t i = 0; i < rgb.size(); ++i)
    rgb[i] = static_cast<uint8_t>((i * 7 + static_cast<size_t>(seed) * 53) % 251);
  std::string out;
  for (size_t i = 0; i < rgb.size(); i += 3) {
    const uint32_t v = (static_cast<uint32_t>(rgb[i]) << 16) |
                       (i + 1 < rgb.size() ? static_cast<uint32_t>(rgb[i + 1]) << 8 : 0U) |
                       (i + 2 < rgb.size() ? static_cast<uint32_t>(rgb[i + 2]) : 0U);
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

// The context the server fills in, with a stand-in renderer that records what
// it was asked to render (a ChatML shape, which is what the released template
// produces around the content).
struct Ctx {
  vllm::HfConfig config;
  vllm::MultiModalConfig mm_config;
  oai::MultiModalChatContext ctx;
  std::shared_ptr<std::vector<std::string>> seen = std::make_shared<std::vector<std::string>>();
  // The last whole prompt the renderer produced.
  std::shared_ptr<std::string> prompt = std::make_shared<std::string>();

  explicit Ctx(int64_t max_model_len = 16384, const vllm::tok::Tokenizer* tok = &Tok()) {
    config = vllm::LoadHfConfig(std::string(NANO_NEMOTRON_VL_FIXTURE_DIR) +
                                "/nemotron_nano_omni_v3/config.json");
    ctx.architecture = kArch;
    ctx.model_dir = "/nonexistent/nemotron-omni";
    ctx.config_path = ctx.model_dir + "/config.json";
    ctx.served_model_name = "nemotron-omni";
    ctx.tokenizer = tok;
    auto rec = seen;
    auto last = prompt;
    ctx.prompt_fn = [rec, last](const std::vector<oai::ChatMessage>& msgs, bool add_gen,
                          const std::vector<oai::ChatCompletionToolsParam>&,
                          const json&) -> std::string {
      std::string out;
      for (const auto& m : msgs) {
        rec->push_back(m.content.value_or(""));
        out += "<|im_start|>" + m.role + "\n" + m.content.value_or("") + "<|im_end|>\n";
      }
      if (add_gen) out += "<|im_start|>assistant\n";
      *last = out;
      return out;
    };
    ctx.codec = oai::DefaultImageCodec();
    ctx.config = &config;
    ctx.mm_config = &mm_config;
    ctx.max_model_len = max_model_len;
  }
};

}  // namespace

TEST_CASE("nano-nemotron-vl mm chat: both released architectures have a seam") {
  for (const char* arch : {"NemotronH_Nano_Omni_Reasoning_V3", "NemotronH_Nano_VL_V2"}) {
    const oai::MultiModalChatRegistration* reg = oai::MultiModalChatRegistry::Find(arch);
    REQUIRE(reg != nullptr);
    CHECK(reg->make_seam != nullptr);
  }
}

TEST_CASE("nano-nemotron-vl mm chat: one image is prefixed, expanded and featurized") {
  Ctx c;
  const oai::MultiModalChatSeam seam = oai::MultiModalChatRegistry::MakeSeam(c.ctx);
  REQUIRE(seam.chat_fn);
  const auto out = seam.chat_fn({UserWith({TextPart("\nwhat is it"), ImagePart(48, 1)})});
  REQUIRE(out.has_value());
  REQUIRE(c.seen->size() == 1);
  CHECK((*c.seen)[0] == "<image>\nwhat is it");
  REQUIRE(out->mm_features.size() == 1);
  const auto& f = out->mm_features[0];
  // 48x48 under min_num_patches 1024: grid 3x3 is upscaled to 32x32 -> 256 rows.
  CHECK(f.data->image_grid_thw[1] == 32);
  CHECK(f.data->image_grid_thw[2] == 32);
  CHECK(f.length == 256);
  const auto& ids = out->prompt_token_ids;
  REQUIRE(f.offset >= 1);
  CHECK(ids[static_cast<size_t>(f.offset - 1)] == kImgStart);
  for (int i = 0; i < f.length; ++i) CHECK(ids[static_cast<size_t>(f.offset + i)] == kImage);
  CHECK(ids[static_cast<size_t>(f.offset + f.length)] == kImgEnd);
  CHECK(std::count(ids.begin(), ids.end(), kImage) == 256);
}

TEST_CASE("nano-nemotron-vl mm chat: two images get the numbered prefix and two features") {
  Ctx c;
  const oai::MultiModalChatSeam seam = oai::MultiModalChatRegistry::MakeSeam(c.ctx);
  const auto out =
      seam.chat_fn({UserWith({ImagePart(48, 1), TextPart("compare"), ImagePart(64, 2)})});
  REQUIRE(out.has_value());
  REQUIRE(c.seen->size() == 1);
  CHECK((*c.seen)[0] == "<image 1><image> <image 2><image>\ncompare");
  REQUIRE(out->mm_features.size() == 2);
  CHECK(out->mm_features[0].offset < out->mm_features[1].offset);
  CHECK(out->mm_features[0].mm_hash != out->mm_features[1].mm_hash);
}

TEST_CASE("nano-nemotron-vl mm chat: a text-only conversation is not intercepted") {
  Ctx c;
  const oai::MultiModalChatSeam seam = oai::MultiModalChatRegistry::MakeSeam(c.ctx);
  oai::ChatMessage m;
  m.role = "user";
  m.content = "hello";
  CHECK_FALSE(seam.chat_fn({m}).has_value());
}

TEST_CASE("nano-nemotron-vl mm chat: an install without max_model_len is refused") {
  Ctx c(/*max_model_len=*/0);
  bool refused = false;
  try {
    (void)oai::MultiModalChatRegistry::MakeSeam(c.ctx);
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("max_model_len") != std::string::npos;
  }
  CHECK(refused);
}

// The text length the tiler budgets against is upstream's
// `len(tokenizer(sans_images, add_special_tokens=False).input_ids)`
// (processors/nano_nemotron_vl.py:689-692 @ e126687a9a): the added special
// tokens in the text are still PARSED, but no BOS/EOS is added. On a tokenizer
// whose post_processor prepends a BOS the two spellings differ by one token,
// and one token is four patches of budget. This case puts the budget exactly
// at a 40x40 grid for a 640x640 image, so a text length one token too long
// shrinks the grid to 40x38.
TEST_CASE("nano-nemotron-vl mm chat: the tiler's text length adds no special tokens") {
  const vllm::tok::Tokenizer& tok = TokWithBos();
  const auto messages = std::vector<oai::ChatMessage>{
      UserWith({TextPart("\nwhat is it"), ImagePart(640, 3)})};
  // The rendered prompt does not depend on max_model_len: a first seam reads it.
  Ctx probe(/*max_model_len=*/16384, &tok);
  REQUIRE(oai::MultiModalChatRegistry::MakeSeam(probe.ctx).chat_fn(messages).has_value());
  std::string sans = *probe.prompt;
  for (size_t at = sans.find("<image>"); at != std::string::npos; at = sans.find("<image>", at))
    sans.erase(at, 7);
  const int64_t text_len = static_cast<int64_t>(tok.Encode(sans).size());
  // The tokenizer DOES distinguish the two spellings, or this case proves nothing.
  REQUIRE(static_cast<int64_t>(tok.EncodeWithSpecialTokens(sans).size()) == text_len + 1);

  // tokens_available = max_model_len - text_len - 4 (:330-331) = 400, which is
  // 1600 patches: exactly the 40x40 grid a 640x640 image asks for at patch 16.
  Ctx c(/*max_model_len=*/400 + text_len + 4, &tok);
  const auto out = oai::MultiModalChatRegistry::MakeSeam(c.ctx).chat_fn(messages);
  REQUIRE(out.has_value());
  REQUIRE(out->mm_features.size() == 1);
  const auto& f = out->mm_features[0];
  CHECK(f.data->image_grid_thw[1] == 40);
  CHECK(f.data->image_grid_thw[2] == 40);
  CHECK(f.length == 400);
}
