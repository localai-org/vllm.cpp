// Synthetic loader contract, not original-checkpoint inference. Original names
// follow the pinned Google config in fixtures/gemma4_e4b_google. Dimensions are
// reduced to avoid allocating the original vocabulary and PLE tables.
#include <doctest/doctest.h>
#include "vllm/multimodal/gemma4_audio_processor.h"
#include <filesystem>
#include <functional>
#include <cmath>
#include <limits>
#include <cstring>
#include <fstream>
#include <random>
#include <nlohmann/json.hpp>
#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"

namespace {
using nlohmann::json;
using Block = vllm::multimodal::Gemma4VisionBlockWeights;
struct BlockEntry {
  const char* name;
  std::vector<int64_t> shape;
  std::vector<uint16_t> Block::* member;
};
const std::vector<BlockEntry> inventory = {
  {"self_attn.q_proj.linear.weight", {4, 3}, &Block::q_proj},
  {"self_attn.k_proj.linear.weight", {2, 3}, &Block::k_proj},
  {"self_attn.v_proj.linear.weight", {2, 3}, &Block::v_proj},
  {"self_attn.o_proj.linear.weight", {3, 4}, &Block::o_proj},
  {"mlp.gate_proj.linear.weight", {5, 3}, &Block::gate_proj},
  {"mlp.up_proj.linear.weight", {5, 3}, &Block::up_proj},
  {"mlp.down_proj.linear.weight", {3, 5}, &Block::down_proj},
  {"input_layernorm.weight", {3}, &Block::input_ln},
  {"post_attention_layernorm.weight", {3}, &Block::post_attn_ln},
  {"pre_feedforward_layernorm.weight", {3}, &Block::pre_ff_ln},
  {"post_feedforward_layernorm.weight", {3}, &Block::post_ff_ln},
  {"self_attn.q_norm.weight", {2}, &Block::q_norm},
  {"self_attn.k_norm.weight", {2}, &Block::k_norm},
};
std::string block_prefix(int layer) {
  return "model.vision_tower.encoder.layers." + std::to_string(layer) + ".";
}
struct Fixture {
  std::filesystem::path path;
  std::filesystem::path directory;
  json header = json::object();
  std::string bytes;
  Fixture() {
    for (int attempt = 0; attempt < 32; ++attempt) {
      directory = std::filesystem::temp_directory_path() /
          ("e4b-loader-" + std::to_string(std::random_device{}()));
      if (std::filesystem::create_directory(directory)) break;
      directory.clear();
    }
    if (directory.empty()) throw std::runtime_error("cannot create fixture directory");
    path = directory / "model.safetensors";
    add("model.vision_tower.patch_embedder.input_proj.weight", {3, 3});
    add("model.vision_tower.patch_embedder.position_embedding_table", {2, 2, 3});
    for (int layer = 0; layer < 2; ++layer)
      for (const auto& e : inventory) add(block_prefix(layer) + e.name, e.shape);
    for (int layer = 0; layer < 2; ++layer)
      for (size_t i = 0; i < 7; ++i) {
        std::string base = inventory[i].name;
        base.resize(base.size() - std::string("linear.weight").size());
        for (const auto* bound : {"input_min", "input_max", "output_min", "output_max"})
          add(block_prefix(layer) + base + bound, {});
      }
    add("model.language_model.embed_tokens.weight", {3, 2});
    add("model.language_model.norm.weight", {2});
    const std::string p = "model.language_model.layers.0.";
    for (const auto* n : {"input_layernorm", "post_attention_layernorm",
                         "pre_feedforward_layernorm", "post_feedforward_layernorm"})
      add(p + n + ".weight", {2});
    for (const auto* n : {"q_proj", "k_proj", "v_proj", "o_proj"})
      add(p + "self_attn." + n + ".weight", {2, 2});
    for (const auto* n : {"q_norm", "k_norm"})
      add(p + "self_attn." + n + ".weight", {2});
    for (const auto* n : {"gate_proj", "up_proj", "down_proj"})
      add(p + "mlp." + n + ".weight", {2, 2});
  }
  ~Fixture() { std::error_code ec; std::filesystem::remove_all(directory, ec); }
  void add(const std::string& name, std::vector<int64_t> shape) {
    size_t n = 1;
    for (auto d : shape) n *= static_cast<size_t>(d);
    const size_t start = bytes.size();
    // BF16 1.0, serialized little-endian.
    for (size_t i = 0; i < n; ++i) { bytes.push_back('\x80'); bytes.push_back('\x3f'); }
    header[name] = {{"dtype", "BF16"}, {"shape", shape},
                    {"data_offsets", {start, bytes.size()}}};
  }
  std::vector<vllm::SafetensorsFile> open() {
    auto h = header.dump();
    while (h.size() % 8) h.push_back(' ');
    std::ofstream f(path, std::ios::binary);
    for (unsigned i = 0; i < 8; ++i) f.put(static_cast<char>((uint64_t(h.size()) >> (8*i)) & 255));
    f << h << bytes;
    f.close();
    std::vector<vllm::SafetensorsFile> shards;
    shards.push_back(vllm::SafetensorsFile::Open(path.string()));
    return shards;
  }
};
vllm::HfConfig config() {
  vllm::HfConfig c;
  c.hidden_size = 2; c.vocab_size = 3; c.num_hidden_layers = 1;
  c.num_attention_heads = 1; c.num_key_value_heads = 1; c.head_dim = 2;
  c.raw = {{"text_config", json::object()}};
  return c;
}
}
TEST_CASE("E4B loader: absent tie flag defaults to tied without head allocation") {
  Fixture f; auto shards = f.open();
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, config());
  CHECK(w.tie_word_embeddings); CHECK(w.lm_head.bytes.empty());
  REQUIRE(w.layers.size() == 1); CHECK(w.layers[0].head_dim == 2);
  CHECK_FALSE(w.layers[0].is_kv_shared); CHECK_FALSE(w.layers[0].k_eq_v);
}
TEST_CASE("E4B loader: false tie flag requires lm_head") {
  Fixture f; auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = false;
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains("lm_head.weight"));
}
TEST_CASE("E4B loader: untied head has vocab by hidden source shape") {
  Fixture f; f.add("lm_head.weight", {3, 2});
  const uint16_t values[] = {0x3f80, 0x4000, 0x4040, 0x4080, 0x40a0, 0x40c0};
  const size_t start = f.bytes.size() - sizeof(values);
  for (size_t i = 0; i < 6; ++i) {
    f.bytes[start + 2*i] = static_cast<char>(values[i] & 255);
    f.bytes[start + 2*i + 1] = static_cast<char>(values[i] >> 8);
  }
  auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = false;
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
  CHECK_FALSE(w.tie_word_embeddings); CHECK(w.lm_head.rank == 2);
  CHECK(w.lm_head.dtype == vt::DType::kBF16);
  CHECK(w.lm_head.shape[0] == 2); CHECK(w.lm_head.shape[1] == 3);
  REQUIRE(w.lm_head.bytes.size() == 12);
  uint16_t actual[6]; std::memcpy(actual, w.lm_head.bytes.data(), sizeof(actual));
  const uint16_t expected[] = {0x3f80, 0x4040, 0x40a0, 0x4000, 0x4080, 0x40c0};
  for (size_t i = 0; i < 6; ++i) CHECK(actual[i] == expected[i]);
  std::vector<vllm::SafetensorsFile> second;
  second.push_back(vllm::SafetensorsFile::Open(f.path.string()));
  auto owned = std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(second));
  auto ow = vllm::LoadGemma4ForConditionalGenerationWeightsOwned(owned, c);
  CHECK(ow.lm_head.dtype == vt::DType::kBF16);
  REQUIRE(ow.lm_head.bytes.size() == w.lm_head.bytes.size());
  CHECK(std::memcmp(ow.lm_head.bytes.data(), w.lm_head.bytes.data(), 12) == 0);
}
TEST_CASE("E4B loader: correctly shaped F16 untied head is refused by both overloads") {
  Fixture f; f.add("lm_head.weight", {3, 2});
  // F16 has the same byte width: only the dtype changes, not shape or payload.
  f.header["lm_head.weight"]["dtype"] = "F16";
  auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = false;
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains("BF16"));
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains("lm_head.weight"));
  auto owned = std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards));
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeightsOwned(owned, c),
                    doctest::Contains("BF16"));
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeightsOwned(owned, c),
                    doctest::Contains("lm_head.weight"));
}
TEST_CASE("E4B loader: malformed untied head shape is refused") {
  for (const auto& shape : std::vector<std::vector<int64_t>>{{2, 3}, {6}, {3, 1}}) {
    Fixture f; f.add("lm_head.weight", shape); auto shards = f.open(); auto c = config();
    c.raw["text_config"]["tie_word_embeddings"] = false;
    CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                      doctest::Contains("lm_head.weight"));
  }
}
TEST_CASE("E4B loader: tied flag ignores an unused separate head") {
  Fixture f; f.add("lm_head.weight", {1}); auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = true;
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
  CHECK(w.tie_word_embeddings); CHECK(w.lm_head.bytes.empty());
}
TEST_CASE("E4B loader: invalid tie flag is not silently defaulted") {
  for (const auto& flag : std::vector<json>{"false", 0, json::array(), json::object()}) {
    Fixture f; auto shards = f.open(); auto c = config();
    c.raw["text_config"]["tie_word_embeddings"] = flag;
    CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                      doctest::Contains("tie_word_embeddings"));
  }
}
TEST_CASE("E4B loader: null tie flag retains the legacy default") {
  Fixture f; auto shards = f.open(); auto c = config();
  c.raw["text_config"]["tie_word_embeddings"] = nullptr;
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
  CHECK(w.tie_word_embeddings); CHECK(w.lm_head.bytes.empty());
}
TEST_CASE("E4B loader: head rejection uses original E4B config dimensions") {
  // No original-size tensor allocation: an intentionally tiny head is refused
  // against the independently pinned original config dimensions.
  const auto c0 = vllm::LoadHfConfig(E4B_CONFIG_PATH);
  CHECK(c0.hidden_size == 2560); CHECK(c0.vocab_size == 262144);
  CHECK(c0.num_hidden_layers == 42);
  CHECK(c0.raw["text_config"]["hidden_size_per_layer_input"] == 256);
  CHECK(c0.raw["text_config"]["num_kv_shared_layers"] == 18);
  CHECK(c0.raw["text_config"]["tie_word_embeddings"] == true);
  Fixture f; f.add("lm_head.weight", {3, 2}); auto shards = f.open(); auto c = c0;
  c.raw["text_config"]["tie_word_embeddings"] = false;
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains("lm_head.weight"));
}

TEST_CASE("E4B config: special token IDs match real Google fixture") {
  const auto c0 = vllm::LoadHfConfig(E4B_CONFIG_PATH);
  const auto ids = vllm::Gemma4SpecialTokenIds(c0);
  CHECK(ids.audio_token_id == 258881);
  CHECK(ids.boa_token_id == 256000);
  CHECK(ids.boi_token_id == 255999);
  CHECK(ids.eoa_token_id == 258883);
  CHECK(ids.eoa_token_index == 258883);
  CHECK(ids.eoi_token_id == 258882);
  CHECK(ids.image_token_id == 258880);
  CHECK(ids.video_token_id == 258884);
  CHECK(ids.bos_token_id == 2);
  CHECK(ids.pad_token_id == 0);
  CHECK(ids.vocab_size == 262144);
  // eos_token_id is a list at top level [1, 106]
  CHECK(ids.eos_token_ids.size() == 2);
  CHECK(ids.eos_token_ids[0] == 1);
  CHECK(ids.eos_token_ids[1] == 106);
}

namespace {
const std::string vision_proj = "model.embed_vision.embedding_projection.weight";
const std::string audio_proj = "model.embed_audio.embedding_projection.weight";
vllm::HfConfig mm_config() {
  auto c = config();
  c.raw["vision_config"] = {{"hidden_size", 3}, {"patch_size", 1}, {"position_embedding_size", 2}, {"rms_norm_eps", 1e-6}, {"num_hidden_layers", 2}, {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"head_dim", 2}, {"intermediate_size", 5},
                            {"use_clipped_linears", true}};
  c.raw["audio_config"] = {{"hidden_size", 4}, {"output_proj_dims", 5},
                           {"rms_norm_eps", 1e-6}, {"num_hidden_layers", 2}, {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"head_dim", 2}, {"intermediate_size", 5}};
  return c;
}
using Audio = vllm::multimodal::Gemma4AudioWeights;
struct AudioEntry {
  const char* name;
  std::vector<int64_t> shape;
  std::vector<float> Audio::*member;
};
std::vector<AudioEntry> audio_inventory(int64_t h = 4, int64_t a = 4, int64_t b = 2) {
  return {{"layer0.conv.weight", {a,1,3,3}, &Audio::sub0_conv},
          {"layer0.norm.weight", {a}, &Audio::sub0_norm},
          {"layer1.conv.weight", {b,a,3,3}, &Audio::sub1_conv},
          {"layer1.norm.weight", {b}, &Audio::sub1_norm},
          {"input_proj_linear.weight", {h,(a/4)*b}, &Audio::input_proj}};
}
const std::string audio_prefix = "model.audio_tower.subsample_conv_projection.";
using AL = vllm::multimodal::Gemma4AudioLayerWeights;
struct AE { std::string name; std::vector<int64_t> shape;
  std::function<const std::vector<float>&(const AL&)> get; };
std::vector<AE> ai(int64_t h=4, int64_t heads=2) {
  const auto q=(h/heads)*heads;
  return {
    {"feed_forward1.ffw_layer_1.linear.weight",{4*h,h},[](const AL& a)->const std::vector<float>& { return a.ff1.ffw1.w; }},
    {"feed_forward1.ffw_layer_2.linear.weight",{h,4*h},[](const AL& a)->const std::vector<float>& { return a.ff1.ffw2.w; }},
    {"feed_forward1.pre_layer_norm.weight",{h},[](const AL& a)->const std::vector<float>& { return a.ff1.pre_ln; }},
    {"feed_forward1.post_layer_norm.weight",{h},[](const AL& a)->const std::vector<float>& { return a.ff1.post_ln; }},
    {"feed_forward2.ffw_layer_1.linear.weight",{4*h,h},[](const AL& a)->const std::vector<float>& { return a.ff2.ffw1.w; }},
    {"feed_forward2.ffw_layer_2.linear.weight",{h,4*h},[](const AL& a)->const std::vector<float>& { return a.ff2.ffw2.w; }},
    {"feed_forward2.pre_layer_norm.weight",{h},[](const AL& a)->const std::vector<float>& { return a.ff2.pre_ln; }},
    {"feed_forward2.post_layer_norm.weight",{h},[](const AL& a)->const std::vector<float>& { return a.ff2.post_ln; }},
    {"self_attn.q_proj.linear.weight",{q,h},[](const AL& a)->const std::vector<float>& { return a.attn.q_proj.w; }},
    {"self_attn.k_proj.linear.weight",{q,h},[](const AL& a)->const std::vector<float>& { return a.attn.k_proj.w; }},
    {"self_attn.v_proj.linear.weight",{q,h},[](const AL& a)->const std::vector<float>& { return a.attn.v_proj.w; }},
    {"self_attn.post.linear.weight",{h,h},[](const AL& a)->const std::vector<float>& { return a.attn.post.w; }},
    {"self_attn.relative_k_proj.weight",{q,h},[](const AL& a)->const std::vector<float>& { return a.attn.relative_k_proj; }},
    {"self_attn.per_dim_scale",{h/heads},[](const AL& a)->const std::vector<float>& { return a.attn.per_dim_scale; }},
    {"lconv1d.linear_start.linear.weight",{2*h,h},[](const AL& a)->const std::vector<float>& { return a.lconv.linear_start.w; }},
    {"lconv1d.linear_end.linear.weight",{h,h},[](const AL& a)->const std::vector<float>& { return a.lconv.linear_end.w; }},
    {"lconv1d.depthwise_conv1d.weight",{h,1,5},[](const AL& a)->const std::vector<float>& { return a.lconv.depthwise; }},
    {"lconv1d.pre_layer_norm.weight",{h},[](const AL& a)->const std::vector<float>& { return a.lconv.pre_ln; }},
    {"lconv1d.conv_norm.weight",{h},[](const AL& a)->const std::vector<float>& { return a.lconv.conv_norm; }},
    {"norm_pre_attn.weight",{h},[](const AL& a)->const std::vector<float>& { return a.norm_pre_attn; }},
    {"norm_post_attn.weight",{h},[](const AL& a)->const std::vector<float>& { return a.norm_post_attn; }},
    {"norm_out.weight",{h},[](const AL& a)->const std::vector<float>& { return a.norm_out; }},
  };
}
std::vector<std::pair<std::string,const vllm::multimodal::AClip*>> audio_clips(const AL& a) {
 return {
  {"feed_forward1.ffw_layer_1",&a.ff1.ffw1.clip},
  {"feed_forward1.ffw_layer_2",&a.ff1.ffw2.clip},
  {"feed_forward2.ffw_layer_1",&a.ff2.ffw1.clip},
  {"feed_forward2.ffw_layer_2",&a.ff2.ffw2.clip},
  {"self_attn.q_proj",&a.attn.q_proj.clip},
  {"self_attn.k_proj",&a.attn.k_proj.clip},
  {"self_attn.v_proj",&a.attn.v_proj.clip},
  {"self_attn.post",&a.attn.post.clip},
  {"lconv1d.linear_start",&a.lconv.linear_start.clip},
  {"lconv1d.linear_end",&a.lconv.linear_end.clip},
 };
}
void add_blocks(Fixture& f, int64_t h=4, int layers=2, int heads=2) {
  AL empty;
  for(int l=0;l<layers;++l) {
    const auto p="model.audio_tower.layers."+std::to_string(l)+".";
    int id=0;
    for(const auto& e:ai(h,heads)) {
      f.add(p+e.name,e.shape);
      const auto start=f.header[p+e.name]["data_offsets"][0].get<size_t>();
      size_t n=1; for(auto d:e.shape)n*=d;
      for(size_t i=0;i<n;++i) {uint16_t bits=0x3000+l*1024+id++;
        std::memcpy(f.bytes.data()+start+2*i,&bits,2);}
    }
    int ci=0;
    for(const auto& e:audio_clips(empty)) {
      int bi=0;
      for(const auto* b:{"input_min","input_max","output_min","output_max"}) {
        const auto name=p+e.first+"."+b; f.add(name,{});
        uint16_t bits=0x3e00+l*128+ci*4+bi++;
        auto start=f.header[name]["data_offsets"][0].get<size_t>();
        std::memcpy(f.bytes.data()+start,&bits,2);
      }
      ++ci;
    }
  }
  f.add("model.audio_tower.output_proj.weight",{h==4?5:h,h});
  f.add("model.audio_tower.output_proj.bias",{h==4?5:h});
  int oi=0;
  for(auto name:{"model.audio_tower.output_proj.weight","model.audio_tower.output_proj.bias"}) {
    auto start=f.header[name]["data_offsets"][0].get<size_t>();
    auto end=f.header[name]["data_offsets"][1].get<size_t>();
    for(auto off=start;off<end;off+=2) {uint16_t bits=0x4200+oi++;
      std::memcpy(f.bytes.data()+off,&bits,2);}
  }
}
void add_audio(Fixture& f, int64_t h, int64_t a = 128, int64_t b = 32) {
  for (const auto& e : audio_inventory(h,a,b)) f.add(audio_prefix + e.name, e.shape);
  add_blocks(f,h);
}
vllm::HfConfig audio_config() {
  auto c = config();
  c.raw["audio_config"] = {{"hidden_size",4},{"output_proj_dims",5},
      {"num_hidden_layers",2},{"num_attention_heads",2},{"rms_norm_eps",0.0001},{"subsampling_conv_channels",{4,2}}};
  return c;
}
void reject_mm(Fixture& f, const vllm::HfConfig& c, const std::string& diagnostic) {
  auto shards = f.open();
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeights(shards, c),
                    doctest::Contains(diagnostic));
  auto owned = std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards));
  CHECK_THROWS_WITH(vllm::LoadGemma4ForConditionalGenerationWeightsOwned(owned, c),
                    doctest::Contains(diagnostic));
}
}
TEST_CASE("E4B projectors: configured modalities require their weights") {
  Fixture f; auto c = mm_config();
  reject_mm(f, c, vision_proj);
  f.add(vision_proj, {2, 3});
  reject_mm(f, c, audio_proj);
}
TEST_CASE("E4B projectors: malformed config is refused by both loaders") {
  for (const auto* key : {"vision_config", "audio_config"}) {
    for (const json& bad : std::vector<json>{false, 7, "bad", json::array()}) {
      Fixture f; auto c = config(); c.raw[key] = bad;
      reject_mm(f, c, key);
    }
    for (const auto* field : {"hidden_size", "rms_norm_eps"}) {
      for (const json& bad : std::vector<json>{nullptr, false, -1, 0, "bad"}) {
        Fixture f; auto c = config();
        c.raw[key] = {{"hidden_size", 3}, {"patch_size", 1}, {"position_embedding_size", 2}, {"rms_norm_eps", 1e-6}, {"num_hidden_layers", 2}, {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"head_dim", 2}, {"intermediate_size", 5}};
        c.raw[key][field] = bad;
        reject_mm(f, c, field);
      }
    }
  }
  for (const json& bad : std::vector<json>{false, -1, 1.5, "bad"}) {
    Fixture f; auto c = config(); c.raw["audio_config"] = mm_config().raw["audio_config"];
    c.raw["audio_config"]["output_proj_dims"] = bad;
    reject_mm(f, c, "output_proj_dims");
  }
}
TEST_CASE("E4B projectors: exact rank and orientation are required") {
  for (const auto* key : {"vision_config", "audio_config"}) {
    const auto name = std::string(key) == "vision_config" ? vision_proj : audio_proj;
    for (const auto& shape : std::vector<std::vector<int64_t>>{{3, 2}, {6}, {2, 1}}) {
      Fixture f; auto c = config(); c.raw[key] = {{"hidden_size", 3}, {"patch_size", 1}, {"position_embedding_size", 2}, {"rms_norm_eps", 1e-6}, {"num_hidden_layers", 2}, {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"head_dim", 2}, {"intermediate_size", 5}};
      f.add(name, shape); reject_mm(f, c, name);
    }
    Fixture f; auto c = config(); c.raw[key] = {{"hidden_size", 3}, {"patch_size", 1}, {"position_embedding_size", 2}, {"rms_norm_eps", 1e-6}, {"num_hidden_layers", 2}, {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"head_dim", 2}, {"intermediate_size", 5}};
    f.add(name, {2, 3}); f.header[name]["dtype"] = "F16";
    reject_mm(f, c, "BF16");
  }
}
TEST_CASE("E4B projectors: owned raw BF16 layout and full config survive shard destruction") {
  for (bool owned : {false, true}) {
    vllm::Gemma4Weights w;
    auto c = mm_config();
    {
      Fixture f; f.add(vision_proj, {2, 3}); f.add(audio_proj, {2, 5}); add_audio(f,4);
      const size_t start = f.header[vision_proj]["data_offsets"][0].get<size_t>();
      for (size_t i = 0; i < 6; ++i) {
        f.bytes[start + 2*i] = 0; f.bytes[start + 2*i + 1] = char(0x3f + i);
      }
      auto shards = f.open();
      w = owned ? vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
          std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)), c)
          : vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
    }
    CHECK(w.vision.config == c.raw["vision_config"]);
    CHECK(w.audio.config == c.raw["audio_config"]);
    for (const auto* p : {&w.vision.embedding_projection, &w.audio.embedding_projection}) {
      CHECK(p->dtype == vt::DType::kBF16); CHECK(p->rank == 2); CHECK(p->shape[0] == 2);
    }
    CHECK(w.vision.embedding_projection.shape[1] == 3);
    CHECK(w.audio.embedding_projection.shape[1] == 5);
    REQUIRE(w.vision.embedding_projection.bytes.size() == 12);
    REQUIRE(w.audio.embedding_projection.bytes.size() == 20);
    uint16_t values[6]; std::memcpy(values, w.vision.embedding_projection.bytes.data(), 12);
    for (size_t i = 0; i < 6; ++i) CHECK(values[i] == ((0x3f + i) << 8));
  }
}
TEST_CASE("E4B projectors: null modalities and audio output width fallback") {
  Fixture f; auto c = config(); c.raw["vision_config"] = nullptr; c.raw["audio_config"] = nullptr;
  auto shards = f.open();
  auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
  CHECK(w.vision.config.is_null()); CHECK(w.audio.embedding_projection.bytes.empty());
  for (const json& output : std::vector<json>{nullptr, 0, 3}) {
    Fixture a; a.add(audio_proj, {2, 3}); auto ac = config();
    ac.raw["audio_config"] = {{"hidden_size", 3}, {"patch_size", 1}, {"position_embedding_size", 2}, {"rms_norm_eps", 1e-6}, {"num_hidden_layers", 2}, {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"head_dim", 2}, {"intermediate_size", 5}, {"output_proj_dims", output}};
    add_audio(a,3); auto as = a.open(); auto aw = vllm::LoadGemma4ForConditionalGenerationWeights(as, ac);
    CHECK(aw.audio.embedding_projection.shape[1] == 3);
    ac.raw["audio_config"].erase("output_proj_dims");
    CHECK(vllm::LoadGemma4ForConditionalGenerationWeights(as, ac).audio.embedding_projection.bytes.size() == 12);
  }
}
TEST_CASE("E4B projectors: original config dimensions reject tiny projector without allocation") {
  auto original = vllm::LoadHfConfig(E4B_CONFIG_PATH);
  for (const auto* key : {"vision_config", "audio_config"}) {
    auto c = config(); c.raw[key] = original.raw[key]; c.hidden_size = original.hidden_size;
    Fixture f; f.add(std::string(key) == "vision_config" ? vision_proj : audio_proj, {2, 3});
    reject_mm(f, c, "encoder_output");
  }
}
TEST_CASE("E4B projectors: scaled FP8 materialization remains supported") {
  for (const auto& name : {vision_proj, audio_proj}) {
    Fixture f; auto c = config();
    c.raw[name == vision_proj ? "vision_config" : "audio_config"] =
        {{"hidden_size", 3}, {"patch_size", 1}, {"position_embedding_size", 2}, {"rms_norm_eps", 1e-6}, {"num_hidden_layers", 2}, {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"head_dim", 2}, {"intermediate_size", 5}};
    const size_t start = f.bytes.size(); f.bytes.append(6, '\x38'); // E4M3 1.0
    f.header[name] = {{"dtype", "F8_E4M3"}, {"shape", {2, 3}}, {"data_offsets", {start, start+6}}};
    f.add(name + "_scale", {1}); // BF16 1.0
    if (name == audio_proj) add_audio(f,3);
    auto shards = f.open();
    auto w = vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
    const auto& p = name == vision_proj ? w.vision.embedding_projection : w.audio.embedding_projection;
    REQUIRE(p.bytes.size() == 12); CHECK(p.dtype == vt::DType::kBF16);
    uint16_t values[6]; std::memcpy(values, p.bytes.data(), 12);
    for (auto v : values) CHECK(v == 0x3f80);
  }
}

namespace {
const std::string patch_input = "model.vision_tower.patch_embedder.input_proj.weight";
const std::string patch_position = "model.vision_tower.patch_embedder.position_embedding_table";
vllm::HfConfig patch_config() {
  auto c = config();
  c.raw["vision_config"] = {{"hidden_size", 3}, {"patch_size", 1}, {"position_embedding_size", 2}, {"rms_norm_eps", 1e-6}, {"num_hidden_layers", 2}, {"num_attention_heads", 2}, {"num_key_value_heads", 1}, {"head_dim", 2}, {"intermediate_size", 5},
      {"patch_size", 1}, {"position_embedding_size", 2}};
  return c;
}
}
TEST_CASE("E4B vision patches: required inventory is not silently discarded") {
  Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
  f.header.erase(patch_input); f.header.erase(patch_position);
  reject_mm(f, c, patch_input);
  f.add(patch_input, {3, 3}); reject_mm(f, c, patch_position);
}
TEST_CASE("E4B vision patches: malformed config shapes and dtype are refused") {
  for (const auto* key : {"patch_size", "position_embedding_size"}) {
    for (const json& value : std::vector<json>{nullptr, false, 0, -1, 1.5, "bad", UINT64_MAX}) {
      Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
      c.raw["vision_config"][key] = value; reject_mm(f, c, key);
    }
    Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
    c.raw["vision_config"].erase(key); reject_mm(f, c, key);
  }
  for (const auto& name : {patch_input, patch_position}) {
    for (int defect = 0; defect < 3; ++defect) {
      Fixture f; f.add(vision_proj, {2, 3});
      f.add(patch_input, {3, 3}); f.add(patch_position, {2, 2, 3});
      if (defect == 0) f.header[name]["shape"] = name == patch_input ? json{9} : json{12};
      if (defect == 1) f.header[name]["shape"] = name == patch_input ? json{1, 9} : json{3, 2, 2};
      if (defect == 2) f.header[name]["dtype"] = "F16";
      reject_mm(f, patch_config(), defect == 2 ? "BF16" : name);
    }
  }
}
TEST_CASE("E4B vision patches: owned raw layout and FP8 scaling survive shard destruction") {
  for (bool owned : {false, true}) {
    for (bool fp8 : {false, true}) {
      vllm::Gemma4Weights w;
      auto c = patch_config();
      {
        Fixture f; f.add(vision_proj, {2, 3});
        for (const auto& name : {patch_input, patch_position}) {
          const size_t count = name == patch_input ? 9 : 12;
          const size_t start = f.bytes.size();
          if (fp8 && name == patch_input) {
            f.bytes.append(count, '\x38'); // E4M3 1.0, scale 2.0
            f.header[name]["dtype"] = "F8_E4M3";
            f.header[name]["data_offsets"] = {start, start + count};
            f.add(name + "_scale", {1});
            f.bytes[f.bytes.size()-2] = 0; f.bytes.back() = '\x40';
          } else {
            for (size_t i = 0; i < count; ++i) {
              f.bytes.push_back(char(i)); f.bytes.push_back('\x3f');
            }
            f.header[name]["data_offsets"] = {start, start + count*2};
          }
        }
        auto shards = f.open();
        w = owned ? vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
            std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)), c)
            : vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
      }
      CHECK(w.vision.config == c.raw["vision_config"]);
      const auto& patch = w.vision_patch_weights;
      REQUIRE(patch.input_proj.size() == 9);
      REQUIRE(patch.position_embedding_table.size() == 12);
      CHECK(patch.blocks.size() == 2); CHECK(patch.embed_projection.empty());
      for (size_t i = 0; i < patch.input_proj.size(); ++i)
        CHECK(patch.input_proj[i] == (fp8 ? 0x4000 : 0x3f00 + i));
      for (size_t i = 0; i < patch.position_embedding_table.size(); ++i)
        CHECK(patch.position_embedding_table[i] == vt::BF16ToF32(uint16_t(0x3f00 + i)));
    }
  }
}
TEST_CASE("E4B vision patches: absent modality and dimension overflow stay bounded") {
  for (bool owned : {false, true}) {
    Fixture f; auto c = config();
    f.header.erase(patch_input); f.header.erase(patch_position);
    auto shards = f.open();
    auto w = owned ? vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
        std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)), c)
        : vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
    CHECK(w.vision_patch_weights.blocks.empty());
    CHECK(w.vision_patch_weights.input_proj.empty());
    CHECK(w.vision_patch_weights.position_embedding_table.empty());
  }
  for (const auto* key : {"patch_size", "position_embedding_size"}) {
    Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
    c.raw["vision_config"][key] = INT64_MAX;
    reject_mm(f, c, key);
  }
  // The patch area fits int64, but the BF16 projection byte count does not.
  {
    Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
    c.raw["vision_config"]["patch_size"] = 1000000000;
    reject_mm(f, c, "patch_size tensor size overflow");
  }
  // The dense materializer supports FP8 matrices, not rank-three tables.
  {
    Fixture f; f.add(vision_proj, {2, 3});
    const size_t start = f.bytes.size(); f.bytes.append(12, '\x38');
    f.header[patch_position]["dtype"] = "F8_E4M3";
    f.header[patch_position]["data_offsets"] = {start, start + 12};
    f.add(patch_position + "_scale", {1});
    reject_mm(f, patch_config(), "expected 2-D weight for FP8");
  }
  auto original = vllm::LoadHfConfig(E4B_CONFIG_PATH);
  Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
  c.raw["vision_config"]["patch_size"] = original.raw["vision_config"]["patch_size"];
  c.raw["vision_config"]["position_embedding_size"] = original.raw["vision_config"]["position_embedding_size"];
  reject_mm(f, c, patch_input);
}

TEST_CASE("E4B vision blocks: complete per-layer raw layout survives shard destruction") {
  for (bool owned : {false, true}) for (bool fp8 : {false, true}) {
    vllm::Gemma4Weights w;
    {
      Fixture f; f.add(vision_proj, {2, 3});
      for (int layer = 0; layer < 2; ++layer) {
        for (size_t j = 0; j < inventory.size(); ++j) {
          const auto& e = inventory[j]; const auto name = block_prefix(layer) + e.name;
          size_t count = 1; for (auto d : e.shape) count *= size_t(d);
          const auto start = f.bytes.size();
          if (fp8 && e.shape.size() == 2) {
            f.bytes.append(count, '\x38');
            f.header[name]["dtype"] = "F8_E4M3";
            f.header[name]["data_offsets"] = {start, start + count};
            f.add(name + "_scale", {1});
            f.bytes[f.bytes.size()-2] = 0; f.bytes.back() = '\x40';
          } else {
            for (size_t k = 0; k < count; ++k) {
              uint16_t bits = uint16_t(0x3f00 + layer * 128 + j * 5 + k);
              f.bytes.push_back(char(bits & 255)); f.bytes.push_back(char(bits >> 8));
            }
            f.header[name]["data_offsets"] = {start, start + 2*count};
          }
        }
      }
      auto shards = f.open();
      w = owned ? vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
          std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)), patch_config())
          : vllm::LoadGemma4ForConditionalGenerationWeights(shards, patch_config());
    }
    w.shards_keepalive.reset();
    REQUIRE(w.vision_patch_weights.blocks.size() == 2);
    for (size_t layer = 0; layer < 2; ++layer) for (size_t j = 0; j < inventory.size(); ++j) {
      const auto& e = inventory[j]; const auto& values = w.vision_patch_weights.blocks[layer].*e.member;
      size_t count = 1; for (auto d : e.shape) count *= size_t(d);
      REQUIRE(values.size() == count);
      for (size_t k = 0; k < count; ++k)
        CHECK(values[k] == (fp8 && e.shape.size() == 2 ? 0x4000 : 0x3f00 + layer*128 + j*5 + k));
    }
  }
}
TEST_CASE("E4B vision blocks: every layer tensor requires exact shape and dtype") {
  for (int layer = 0; layer < 2; ++layer) for (const auto& e : inventory) {
    const auto name = block_prefix(layer) + e.name;
    for (int defect = 0; defect < 3; ++defect) {
      Fixture f; f.add(vision_proj, {2, 3});
      if (defect == 0) f.header.erase(name);
      if (defect == 1) f.add(name, {1});
      if (defect == 2) f.header[name]["dtype"] = "F16";
      reject_mm(f, patch_config(), defect == 2 ? "BF16" : name);
    }
  }
}
TEST_CASE("E4B vision blocks: invalid dimensions and overflow fail before allocation") {
  for (const auto* key : {"num_hidden_layers", "num_attention_heads", "num_key_value_heads", "head_dim", "intermediate_size"}) {
    for (const json& bad : std::vector<json>{nullptr, false, 0, -1, 1.5, "bad", UINT64_MAX}) {
      Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
      c.raw["vision_config"][key] = bad; reject_mm(f, c, key);
    }
    Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
    c.raw["vision_config"].erase(key); reject_mm(f, c, key);
  }
  for (const auto* key : {"num_hidden_layers", "num_attention_heads", "intermediate_size"}) {
    Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
    c.raw["vision_config"][key] = INT64_MAX; reject_mm(f, c, "overflow");
  }
  // Existing positive heads overflow the product before any matrix-size check.
  {
    Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
    c.raw["vision_config"]["head_dim"] = INT64_MAX;
    reject_mm(f, c, "gemma4: vision_config head dimension overflow");
  }
  // Head products fit int64, but matrix bytes overflow with hidden_size=3.
  {
    Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
    c.raw["vision_config"]["num_attention_heads"] = INT64_MAX / 4;
    reject_mm(f, c, "block tensor size overflow");
  }
  Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
  c.raw["vision_config"]["num_key_value_heads"] = 3;
  reject_mm(f, c, "head divisibility");
}

namespace {
const std::vector<vllm::multimodal::Clip Block::*> clips = {
  &Block::q_clip, &Block::k_clip, &Block::v_clip, &Block::o_clip,
  &Block::gate_clip, &Block::up_clip, &Block::down_clip};
const std::vector<std::string> bounds = {"input_min", "input_max", "output_min", "output_max"};
std::string clip_name(int layer, size_t i, size_t j) {
  std::string base = inventory[i].name;
  base.resize(base.size() - std::string("linear.weight").size());
  return block_prefix(layer) + base + bounds[j];
}
void set_bound(Fixture& f, const std::string& name, uint16_t bits) {
  auto offset = f.header[name]["data_offsets"][0].get<size_t>();
  f.bytes[offset] = static_cast<char>(bits & 255);
  f.bytes[offset + 1] = static_cast<char>(bits >> 8);
}
vllm::Gemma4Weights load_fixture(Fixture& f, const vllm::HfConfig& c, bool owned) {
  auto shards = f.open();
  if (owned) return vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
      std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)), c);
  return vllm::LoadGemma4ForConditionalGenerationWeights(shards, c);
}
}
TEST_CASE("E4B clipping: each scalar and layer survives both loader lifetimes") {
  for (bool owned : {false, true}) {
    vllm::Gemma4Weights w;
    {
      Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
      c.raw["vision_config"]["use_clipped_linears"] = true;
      for (int l = 0; l < 2; ++l) for (size_t i = 0; i < 7; ++i)
        for (size_t j = 0; j < 4; ++j)
          set_bound(f, clip_name(l, i, j), uint16_t(0x3f80 + 32*l + 4*i + j));
      w = load_fixture(f, c, owned);
    }
    REQUIRE(w.vision_patch_weights.blocks.size() == 2);
    for (int l = 0; l < 2; ++l) for (size_t i = 0; i < 7; ++i) {
      auto v = w.vision_patch_weights.blocks[l].*clips[i];
      std::vector<float> actual = {v.in_min, v.in_max, v.out_min, v.out_max};
      for (size_t j = 0; j < 4; ++j)
        CHECK(actual[j] == 1.0f + float(32*l + 4*i + j)/128.0f);
    }
  }
}
TEST_CASE("E4B clipping: required scalar inventory rejects malformed bounds") {
  for (int l = 0; l < 2; ++l) for (size_t i = 0; i < 7; ++i)
    for (size_t j = 0; j < 4; ++j) for (int bad = 0; bad < 5; ++bad) {
      Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
      c.raw["vision_config"]["use_clipped_linears"] = true;
      auto name = clip_name(l,i,j);
      if (bad == 0) f.header.erase(name);
      if (bad == 1) f.header[name]["shape"] = {1};
      if (bad == 2) f.header[name]["dtype"] = "F16";
      if (bad == 3) set_bound(f,name,0x7fc0); // NaN
      if (bad == 4) set_bound(f,name,j%2 == 0 ? 0x4000 : 0x0000);
      reject_mm(f,c,bad == 4 ? "clip bounds order" : name);
    }
}
TEST_CASE("E4B clipping: infinite defaults and equal endpoints are valid") {
  for (bool owned : {false,true}) for (bool enabled : {false,true}) {
    Fixture f; f.add(vision_proj, {2,3}); auto c = patch_config();
    if (enabled) c.raw["vision_config"]["use_clipped_linears"] = true;
    for (int l=0;l<2;++l) for(size_t i=0;i<7;++i) for(size_t j=0;j<4;++j) {
      auto name=clip_name(l,i,j);
      if (enabled) set_bound(f,name,j%2 == 0 ? 0xff80 : 0x7f80);
      else f.header.erase(name);
    }
    auto w=load_fixture(f,c,owned);
    for (const auto& b : w.vision_patch_weights.blocks) for(auto member:clips) {
      auto v=b.*member;
      CHECK(v.in_min == -std::numeric_limits<float>::infinity());
      CHECK(v.out_min == -std::numeric_limits<float>::infinity());
      CHECK(v.in_max == std::numeric_limits<float>::infinity());
      CHECK(v.out_max == std::numeric_limits<float>::infinity());
    }
  }
}
TEST_CASE("E4B typed vision config: invalid execution parameters are refused") {
  for (const auto& item : std::vector<std::pair<std::string,json>>{
    {"pooling_kernel_size",0}, {"pooling_kernel_size",1.5}, {"pooling_kernel_size",INT64_MAX},
    {"rms_norm_eps",1e-100}, {"rms_norm_eps",1e100},
    {"use_clipped_linears",1}, {"standardize",true}, {"attention_bias",true},
    {"hidden_activation","relu"}, {"attention_dropout",-0.1}, {"attention_dropout",1.0},
    {"rope_parameters",{{"rope_type","linear"},{"rope_theta",100}}},
    {"rope_parameters",{{"rope_type","default"},{"rope_theta",0}}}}) {
    Fixture f; f.add(vision_proj,{2,3}); auto c=patch_config();
    c.raw["vision_config"][item.first]=item.second; reject_mm(f,c,item.first);
  }
}
TEST_CASE("E4B typed vision config: defaults overrides and absence survive both loaders") {
  for (bool owned : {false, true}) {
    Fixture absent;
    CHECK_FALSE(load_fixture(absent, config(), owned).vision_config.has_value());
    for (bool custom : {false, true}) {
      vllm::Gemma4Weights w;
      {
        Fixture f; f.add(vision_proj, {2, 3}); auto c = patch_config();
        if (custom) {
          c.raw["vision_config"]["pooling_kernel_size"] = 2;
          c.raw["vision_config"]["rope_parameters"] = {{"rope_type", "default"}, {"rope_theta", 250.0}};
          c.raw["vision_config"]["rms_norm_eps"] = 0.125;
          c.raw["vision_config"]["attention_dropout"] = 0.25;
          c.raw["vision_config"]["use_clipped_linears"] = true;
        } else {
          c.raw["vision_config"]["use_clipped_linears"] = false;
          // Explicit false must ignore even poisoned, otherwise-unused bounds.
          set_bound(f, clip_name(0, 0, 0), 0x7fc0);
        }
        w = load_fixture(f, c, owned);
      }
      REQUIRE(w.vision_config.has_value());
      const auto& c = *w.vision_config;
      CHECK(c.hidden_size == 3); CHECK(c.text_hidden_size == 2);
      CHECK(c.num_heads == 2); CHECK(c.num_kv_heads == 1);
      CHECK(c.head_dim == 2); CHECK(c.depth == 2);
      CHECK(c.intermediate_size == 5); CHECK(c.patch_size == 1);
      CHECK(c.patch_dim() == 3); CHECK(c.position_embedding_size == 2);
      CHECK(c.pooling_kernel_size == (custom ? 2 : 3));
      CHECK(c.rope_theta == (custom ? 250.0 : 100.0));
      CHECK(c.rms_norm_eps == (custom ? 0.125f : 1e-6f));
      CHECK(c.attention_dropout == (custom ? 0.25f : 0.0f));
      CHECK(c.use_clipped_linears == custom);
      if (custom) for (const auto& b : w.vision_patch_weights.blocks)
        for (auto member : clips) {
          const auto v = b.*member;
          CHECK(v.in_min == 1); CHECK(v.in_max == 1);
          CHECK(v.out_min == 1); CHECK(v.out_max == 1);
        }
    }
  }
}
TEST_CASE("E4B typed vision config: malformed types and nonfinite numbers fail closed") {
  for (const auto* key : {"pooling_kernel_size", "use_clipped_linears", "standardize",
                          "attention_bias", "attention_dropout", "hidden_activation"})
    for (const auto& value : {json(nullptr), json("bad"), json::array()}) {
      Fixture f; f.add(vision_proj, {2,3}); auto c = patch_config();
      c.raw["vision_config"][key] = value; reject_mm(f, c, key);
    }
  for (double value : {std::numeric_limits<double>::infinity(),
                       std::numeric_limits<double>::quiet_NaN()})
    for (const auto* key : {"rms_norm_eps", "attention_dropout", "rope_parameters"}) {
      Fixture f; f.add(vision_proj, {2,3}); auto c = patch_config();
      c.raw["vision_config"][key] = std::string(key) == "rope_parameters"
          ? json{{"rope_theta", value}} : json(value);
      reject_mm(f, c, key);
    }
  for (const auto& value : {json(false), json::array(), json("bad")}) {
    Fixture f; f.add(vision_proj, {2,3}); auto c = patch_config();
    c.raw["vision_config"]["rope_parameters"] = value;
    reject_mm(f, c, "rope_parameters");
  }
}

TEST_CASE("E4B audio frontend: owned contents layout and configuration") {
  for (bool owned : {false,true}) {
    vllm::Gemma4Weights w;
    {
      Fixture f; add_audio(f,4,4,2); f.add(audio_proj,{2,5});
      int tensor = 0;
      for (const auto& e : audio_inventory()) {
        const auto start = f.header[audio_prefix+e.name]["data_offsets"][0].get<size_t>();
        size_t n=1; for (auto d:e.shape) n*=d;
        for (size_t i=0;i<n;++i) {
          uint16_t bits = static_cast<uint16_t>(0x3f00 + 128*tensor + i);
          std::memcpy(f.bytes.data()+start+2*i,&bits,2);
        }
        ++tensor;
      }
      auto shards=f.open(); auto c=audio_config();
      w=owned ? vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
          std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)),c)
          : vllm::LoadGemma4ForConditionalGenerationWeights(shards,c);
    }
    w.shards_keepalive.reset();
    REQUIRE(w.audio_frontend_config.has_value());
    const auto& c=*w.audio_frontend_config;
    CHECK(c.hidden_size==4); CHECK(c.sub_ch0==4); CHECK(c.sub_ch1==2);
    CHECK(c.input_width==2); CHECK(c.rms_norm_eps==doctest::Approx(0.0001));
    int tensor=0;
    for (const auto& e:audio_inventory()) {
      size_t n=1; for(auto d:e.shape)n*=d;
      const auto& values=w.audio_frontend_weights.*(e.member); REQUIRE(values.size()==n);
      for(size_t i=0;i<n;++i) {
        uint32_t bits=uint32_t(0x3f00+128*tensor+i)<<16; float expected;
        std::memcpy(&expected,&bits,4); CHECK(values[i]==expected);
      }
      ++tensor;
    }
    CHECK(w.audio_frontend_weights.layers.size()==2);
    CHECK(w.audio_frontend_weights.output_proj_w.size()==20);
  }
}
TEST_CASE("E4B audio frontend: missing malformed and wrong dtype inventory") {
  for(const auto& e:audio_inventory()) for(int bad=0;bad<3;++bad) {
    Fixture f; add_audio(f,4,4,2); f.add(audio_proj,{2,5});
    const auto name=audio_prefix+e.name;
    if(bad==0) f.header.erase(name);
    if(bad==1) { auto shape=e.shape; shape.push_back(1); f.header[name]["shape"]=shape; }
    if(bad==2) f.header[name]["dtype"]="F16";
    reject_mm(f,audio_config(),bad==2 ? "gemma4: " + name + " requires BF16" : name);
  }
}
TEST_CASE("E4B audio frontend: valid scaled FP8 input projection requires BF16") {
  Fixture f; add_audio(f,4,4,2); f.add(audio_proj,{2,5});
  const std::string name = audio_prefix + "input_proj_linear.weight";
  const size_t start = f.bytes.size();
  f.bytes.append(8, '\x38'); // Eight E4M3 1.0 values, [out,in] = [4,2].
  f.header[name] = {{"dtype", "F8_E4M3"}, {"shape", {4,2}},
                    {"data_offsets", {start, start+8}}};
  f.add(name + "_scale", {1}); // Valid per-tensor BF16 1.0 scale.
  // The shared materializer accepts this encoding for vision. Only the
  // frontend-specific guard may reject it, before that materializer runs.
  reject_mm(f,audio_config(),"gemma4: " + name + " requires BF16");
}
TEST_CASE("E4B audio frontend: config and precise overflow refusals") {
  for (const auto* key : {"hidden_size", "rms_norm_eps"}) {
    Fixture f; f.add(audio_proj,{2,5}); auto c=audio_config();
    c.raw["audio_config"].erase(key); reject_mm(f,c,key);
  }
  {
    Fixture f; f.add(audio_proj,{2,5}); auto c=audio_config();
    c.raw["audio_config"]["hidden_size"]=INT64_MAX;
    reject_mm(f,c,"audio_config frontend tensor size overflow");
  }
  for(const json& bad:std::vector<json>{nullptr,false,4,"bad",json::array(),{4},{4,2,1},{0,2},{3,2},{4,-1},{4,2.5},{4,true}}) {
    Fixture f; f.add(audio_proj,{2,5}); auto c=audio_config();
    c.raw["audio_config"]["subsampling_conv_channels"]=bad;
    reject_mm(f,c,"subsampling_conv_channels");
  }
  for(double eps:{1e-300,1e300}) {
    Fixture f; f.add(audio_proj,{2,5}); auto c=audio_config();
    c.raw["audio_config"]["rms_norm_eps"]=eps;
    reject_mm(f,c,"audio_config rms_norm_eps");
  }
  for(const json& channels:std::vector<json>{{INT64_MAX,2},{4,INT64_MAX},{int64_t(1)<<32,int64_t(1)<<32}}) {
    Fixture f; f.add(audio_proj,{2,5}); auto c=audio_config();
    c.raw["audio_config"]["subsampling_conv_channels"]=channels;
    reject_mm(f,c,"audio_config frontend tensor size overflow");
  }
}
TEST_CASE("E4B audio frontend: absent and source default channels") {
  for(bool owned:{false,true}) for(bool absent:{false,true}) {
    Fixture f; auto c=config();
    if(!absent) {c=audio_config(); c.raw["audio_config"].erase("subsampling_conv_channels");
      add_audio(f,4); f.add(audio_proj,{2,5});}
    auto shards=f.open();
    auto w=owned ? vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
        std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)),c)
        : vllm::LoadGemma4ForConditionalGenerationWeights(shards,c);
    CHECK(w.audio_frontend_config.has_value()==!absent);
    if(!absent) { REQUIRE(w.audio_frontend_config); CHECK(w.audio_frontend_config->sub_ch0==128);
      CHECK(w.audio_frontend_config->sub_ch1==32); CHECK(w.audio_frontend_config->input_width==1024); }
    else CHECK(w.audio_frontend_weights.sub0_conv.empty());
  }
}

TEST_CASE("E4B audio blocks: complete owned inventory and exact widening") {
 for(bool owned:{false,true}) {
  vllm::Gemma4Weights w;
  { Fixture f; add_audio(f,4,4,2); f.add(audio_proj,{2,5}); auto c=audio_config();
    auto shards=f.open();
    if(owned) w=vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
      std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)),c);
    else w=vllm::LoadGemma4ForConditionalGenerationWeights(shards,c);
  }
  w.shards_keepalive.reset();
  REQUIRE(w.audio_frontend_weights.layers.size()==2);
  for(int l=0;l<2;++l) {
    const auto& a=w.audio_frontend_weights.layers[l]; int id=0;
    for(const auto& e:ai()) {
      size_t n=1;for(auto d:e.shape)n*=d; REQUIRE(e.get(a).size()==n);
      for(auto v:e.get(a)) {uint32_t bits=uint32_t(0x3000+l*1024+id++)<<16;
        float expected;std::memcpy(&expected,&bits,4);CHECK(v==expected);}
    }
    int ci=0;
    for(const auto& e:audio_clips(a)) {int bi=0;
      for(auto v:{e.second->in_min,e.second->in_max,e.second->out_min,e.second->out_max}) {
        uint32_t bits=uint32_t(0x3e00+l*128+ci*4+bi++)<<16;float expected;
        std::memcpy(&expected,&bits,4);CHECK(v==expected);}
      ++ci;
    }
  }
  REQUIRE(w.audio_frontend_weights.output_proj_w.size()==20);
  REQUIRE(w.audio_frontend_weights.output_proj_b.size()==5);
  int oi=0;
  for(const auto* values:{&w.audio_frontend_weights.output_proj_w,&w.audio_frontend_weights.output_proj_b})
    for(float v:*values) {uint32_t bits=uint32_t(0x4200+oi++)<<16;float expected;
      std::memcpy(&expected,&bits,4);CHECK(v==expected);}
  REQUIRE(w.audio_config);const auto& c=*w.audio_config;
  CHECK(c.num_layers==2);CHECK(c.num_heads==2);CHECK(c.head_dim==2);
  CHECK(c.hidden_size==4);CHECK(c.output_proj_dims==5);CHECK(c.text_hidden_size==2);
  CHECK(c.conv_kernel_size==5);CHECK(c.chunk_size==12);CHECK(c.context_left==13);
  CHECK(c.context_right==0);CHECK(c.residual_weight==0.5f);
  CHECK(c.attention_logit_cap==50);CHECK(c.attention_invalid_logits_value==-1e9f);
  CHECK(c.gradient_clipping==1e10);CHECK(c.hidden_act=="silu");CHECK(c.use_clipped_linears);
  CHECK(c.sub_ch0==4);CHECK(c.sub_ch1==2);CHECK(c.rms_norm_eps==0.0001f);
 }
}
TEST_CASE("E4B audio blocks: every tensor fails closed") {
 AL empty; std::vector<std::string> names;
 for(int l=0;l<2;++l) {
  auto p="model.audio_tower.layers."+std::to_string(l)+".";
  for(const auto& e:ai()) names.push_back(p+e.name);
  for(const auto& e:audio_clips(empty))for(auto b:{"input_min","input_max","output_min","output_max"})
    names.push_back(p+e.first+"."+b);
 }
 names.push_back("model.audio_tower.output_proj.weight");
 names.push_back("model.audio_tower.output_proj.bias");
 for(const auto& name:names)for(int bad=0;bad<3;++bad) {
  Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});
  if(bad==0)f.header.erase(name);
  if(bad==1)f.header[name]["shape"].push_back(1);
  if(bad==2)f.header[name]["dtype"]="F16";
  reject_mm(f,audio_config(),bad==2?"gemma4: "+name+" requires BF16":name);
 }
}

namespace {
void set_audio_bits(Fixture& f,const std::string& name,uint16_t bits) {
 auto start=f.header[name]["data_offsets"][0].get<size_t>();
 std::memcpy(f.bytes.data()+start,&bits,2);
}
vllm::Gemma4Weights load_audio_fixture(Fixture& f,const vllm::HfConfig& c,bool owned) {
 auto shards=f.open();
 if(owned)return vllm::LoadGemma4ForConditionalGenerationWeightsOwned(
   std::make_shared<const std::vector<vllm::SafetensorsFile>>(std::move(shards)),c);
 return vllm::LoadGemma4ForConditionalGenerationWeights(shards,c);
}
}
TEST_CASE("E4B audio blocks: every bound rejects NaN and reversed pairs") {
 AL empty;
 for(int l=0;l<2;++l)for(const auto& e:audio_clips(empty)) {
  auto p="model.audio_tower.layers."+std::to_string(l)+"."+e.first;
  for(auto b:{"input_min","input_max","output_min","output_max"}) {
   Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});
   set_audio_bits(f,p+"."+b,0x7fc1);
   reject_mm(f,audio_config(),"gemma4: "+p+"."+b+" must not be NaN");
  }
  for(auto b:{"input","output"}) {
   Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});
   set_audio_bits(f,p+"."+b+"_min",0x7f80);
   reject_mm(f,audio_config(),"gemma4: "+p+" "+b+"_min exceeds "+b+"_max");
  }
 }
}
TEST_CASE("E4B audio blocks: disabled clipping and infinite equal bounds") {
 for(bool owned:{false,true})for(int mode=0;mode<3;++mode) {
  Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();AL empty;
  if(mode==0)c.raw["audio_config"]["use_clipped_linears"]=false;
  for(int l=0;l<2;++l)for(const auto& e:audio_clips(empty))
   for(auto b:{"input_min","input_max","output_min","output_max"}) {
    auto name="model.audio_tower.layers."+std::to_string(l)+"."+e.first+"."+b;
    if(mode==0)f.header.erase(name);
    else set_audio_bits(f,name,mode==2?0x4000:(std::string(b).find("min")!=std::string::npos?0xff80:0x7f80));
   }
  auto w=load_audio_fixture(f,c,owned);REQUIRE(w.audio_config);
  CHECK(w.audio_config->use_clipped_linears==(mode!=0));
  for(const auto& a:w.audio_frontend_weights.layers)for(const auto& e:audio_clips(a)) {
   CHECK(e.second->in_min==(mode==2?2.f:-std::numeric_limits<float>::infinity()));
   CHECK(e.second->out_min==e.second->in_min);
   CHECK(e.second->in_max==(mode==2?2.f:std::numeric_limits<float>::infinity()));
   CHECK(e.second->out_max==e.second->in_max);
  }
 }
}
TEST_CASE("E4B audio blocks: typed config rejects malformed values and overflow") {
 for(auto key:{"num_hidden_layers","num_attention_heads","conv_kernel_size","attention_chunk_size",
               "attention_context_left","attention_context_right"})
  for(const auto& bad:std::vector<json>{nullptr,true,"2",-1,1.5,json::array(),uint64_t(-1)}) {
   Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
   c.raw["audio_config"][key]=bad;reject_mm(f,c,std::string("audio_config ")+key+" invalid integer");
  }
 for(auto key:{"num_hidden_layers","num_attention_heads","conv_kernel_size","attention_chunk_size","attention_context_left"}) {
  Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
  c.raw["audio_config"][key]=0;reject_mm(f,c,std::string("audio_config ")+key+" invalid integer");
 }
 for(auto key:{"residual_weight","attention_logit_cap","attention_invalid_logits_value","gradient_clipping"})
  for(const auto& bad:std::vector<json>{nullptr,true,"2",1e300,std::numeric_limits<double>::quiet_NaN(),std::numeric_limits<double>::infinity()}) {
   Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
   c.raw["audio_config"][key]=bad;reject_mm(f,c,std::string("audio_config ")+key+" invalid finite F32 value");
  }
 for(auto key:{"attention_logit_cap","gradient_clipping"})for(double bad:{0.,-1.,1e-300}) {
  Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
  c.raw["audio_config"][key]=bad;reject_mm(f,c,std::string("audio_config ")+key+" invalid finite F32 value");
 }
 for(auto key:{"num_hidden_layers","conv_kernel_size"}) {
  Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
  c.raw["audio_config"][key]=INT64_MAX;
  f.header.erase("model.audio_tower.layers.0.feed_forward1.ffw_layer_1.linear.weight");
  reject_mm(f,c,"audio_config block tensor size overflow");
 }
 for(auto key:{"attention_chunk_size","attention_context_left","attention_context_right"}) {
  Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
  c.raw["audio_config"][key]=INT64_MAX;reject_mm(f,c,"audio_config attention context overflow");
 }
 for(const auto& bad:std::vector<json>{nullptr,0,"false",json::array()}) {
  Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
  c.raw["audio_config"]["use_clipped_linears"]=bad;
  reject_mm(f,c,"audio_config use_clipped_linears must be boolean");
 }
 {Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
  c.raw["audio_config"]["hidden_act"]=nullptr;reject_mm(f,c,"audio_config hidden_act must be a string");}
 {Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
  c.raw["audio_config"]["num_attention_heads"]=5;reject_mm(f,c,"audio_config num_attention_heads exceeds hidden_size");}
}
TEST_CASE("E4B audio blocks: config overrides and absent inventory") {
 for(bool owned:{false,true}) {
  Fixture f;add_audio(f,4,4,2);f.add(audio_proj,{2,5});auto c=audio_config();
  auto& j=c.raw["audio_config"];j["residual_weight"]=-0.25;j["attention_logit_cap"]=7;
  j["attention_invalid_logits_value"]=-2;j["gradient_clipping"]=32;
  j["attention_chunk_size"]=3;j["attention_context_left"]=2;j["attention_context_right"]=1;
  j["hidden_act"]="relu";
  auto w=load_audio_fixture(f,c,owned);REQUIRE(w.audio_config);const auto& a=*w.audio_config;
  CHECK(a.residual_weight==-0.25f);CHECK(a.attention_logit_cap==7);
  CHECK(a.attention_invalid_logits_value==-2);CHECK(a.gradient_clipping==32);
  CHECK(a.chunk_size==3);CHECK(a.context_left==2);CHECK(a.context_right==1);CHECK(a.hidden_act=="relu");
  CHECK(w.audio.config==j);
  auto none=load_audio_fixture(f,config(),owned);CHECK_FALSE(none.audio_config);
  CHECK(none.audio_frontend_weights.layers.empty());CHECK(none.audio_frontend_weights.output_proj_b.empty());
 }
}

TEST_CASE("E4B audio blocks: source layer and head defaults with twelve tiny blocks") {
 for(bool owned:{false,true}) {
  Fixture f;add_audio(f,8,4,2);add_blocks(f,8,12,8);f.add(audio_proj,{2,8});
  auto c=audio_config();auto& j=c.raw["audio_config"];
  j["hidden_size"]=8;j.erase("num_hidden_layers");j.erase("num_attention_heads");
  j.erase("output_proj_dims");
  auto w=load_audio_fixture(f,c,owned);REQUIRE(w.audio_config);
  CHECK(w.audio_config->num_layers==12);CHECK(w.audio_config->num_heads==8);
  CHECK(w.audio_config->head_dim==1);CHECK(w.audio_config->output_proj_dims==8);
  REQUIRE(w.audio_frontend_weights.layers.size()==12);
  for(const auto& a:w.audio_frontend_weights.layers) {
   CHECK(a.attn.per_dim_scale.size()==1);CHECK(a.ff1.ffw1.w.size()==256);
  }
 }
}

#include "vllm/model_executor/models/gemma4_loaded.h"
TEST_CASE("E4B execution: shared registry dispatch reaches loaded audio and projector") {
  Fixture f; auto c = audio_config(); add_audio(f,4,4,2); f.add(audio_proj,{2,5});
  // Independent scalar fixture: zero output matrix, bias [1,2,3,4,5].
  // Connector rows select first and last normalized bias components.
  auto set = [&](const std::string& name, const std::vector<float>& values) {
    const size_t start=f.header[name]["data_offsets"][0].get<size_t>();
    for(size_t i=0;i<values.size();++i) {
      auto bits=vt::F32ToBF16(values[i]); std::memcpy(f.bytes.data()+start+2*i,&bits,2);
    }
  };
  set("model.audio_tower.output_proj.weight",std::vector<float>(20,0));
  set("model.audio_tower.output_proj.bias",{1,2,3,4,5});
  set(audio_proj,{1,0,0,0,0, 0,0,0,0,1});
  auto shards=f.open(); auto weights=vllm::LoadGemma4ForConditionalGenerationWeights(shards,c);
  const auto& factory=vllm::kGemma4MmHooks;
  const vllm::ModelRegistration registration{"Gemma4ForConditionalGeneration",&factory,{}};
  vllm::Gemma4LoadedModel model(registration,std::move(weights));
  CHECK(vllm::ModelRegistry::SupportsMmInputs(model));
  vt::Queue queue{};
  vllm::multimodal::MultiModalFeatureSpec item;
  item.modality="audio"; item.length=2;
  item.audio_data=std::make_shared<vllm::multimodal::AudioKwargs>();
  auto& a=*item.audio_data; a.n_mels=4; a.n_frames=8;
  a.input_features.resize(32); for(size_t i=0;i<32;++i)a.input_features[i]=float(i)/32;
  a.feature_mask={1,1,1,1,1,0,0,0};
  auto out=vllm::ModelRegistry::EncodeMm(model,c,queue,item);
  REQUIRE(out.storage); REQUIRE(out.embeds.shape[0]==2); REQUIRE(out.embeds.shape[1]==2);
  REQUIRE(out.embeds.dtype==vt::DType::kF32);
  const float scale=1/std::sqrt(11.0f+0.0001f);
  for(int i=0;i<2;++i) {
    CHECK(out.embeds.Ptr<float>()[2*i]==doctest::Approx(scale));
    CHECK(out.embeds.Ptr<float>()[2*i+1]==doctest::Approx(5*scale));
  }
  auto reject=[&](const char* message){CHECK_THROWS_WITH(vllm::ModelRegistry::EncodeMm(model,c,queue,item),doctest::Contains(message));};
  a.feature_mask={1}; reject("mask");
  a.feature_mask={1,0,1,1,1,0,0,0}; reject("prefix");
  a.feature_mask={1,1,2,1,1,0,0,0}; reject("mask");
  a.feature_mask=std::vector<int32_t>(8,0); reject("valid");
  a.feature_mask.clear(); item.length=1; reject("length"); item.length=2;
  a.num_chunks=2; reject("chunk"); a.num_chunks=1;
  a.input_features.pop_back(); reject("shape"); a.input_features.push_back(0);
  item.modality="image"; reject("audio"); item.modality="audio";
  queue.device.type=vt::DeviceType::kCUDA; reject("CPU");
  queue.device.type=vt::DeviceType::kCPU;
  a.n_frames=INT64_MAX; reject("shape"); a.n_frames=8;
  a.n_mels=5; a.input_features.resize(40); reject("width");
  a.n_mels=4; a.input_features.resize(32);
  a.input_features[0]=std::numeric_limits<float>::quiet_NaN(); reject("nonfinite");
  a.input_features[0]=0;
  a.num_tokens=1; reject("length"); a.num_tokens=0;
  // Loaded configurations which the reference cannot faithfully execute.
  for (const auto& [key,value] : std::vector<std::pair<std::string,int>>{
      {"attention_context_right",1},{"attention_context_left",1},{"num_attention_heads",3}}) {
    auto badc=c; badc.raw["audio_config"][key]=value;
    Fixture bad; add_audio(bad,4,4,2); bad.add(audio_proj,{2,5});
    if(key=="num_attention_heads") add_blocks(bad,4,2,3);
    auto bs=bad.open(); auto bw=vllm::LoadGemma4ForConditionalGenerationWeights(bs,badc);
    vllm::Gemma4LoadedModel bm(registration,std::move(bw));
    CHECK_THROWS_WITH(vllm::ModelRegistry::EncodeMm(bm,badc,queue,item),doctest::Contains("config"));
  }
  // Output owns its storage independently of the model/fixture.
  auto saved=out.storage; CHECK(out.embeds.Ptr<float>()[0]==doctest::Approx(scale));
}

TEST_CASE("E4B execution: real tower responds to features and strips trailing padding") {
  Fixture f; auto c=audio_config(); add_audio(f,4,4,2); f.add(audio_proj,{2,5});
  c.raw["audio_config"]["use_clipped_linears"]=false;
  auto put=[&](const std::string& name, std::vector<float> values) {
    auto offset=f.header[name]["data_offsets"][0].get<size_t>();
    for(float x:values) {auto bits=vt::F32ToBF16(x); std::memcpy(f.bytes.data()+offset,&bits,2);offset+=2;}
  };
  // Sparse center taps preserve a sign-dependent channel direction through
  // both real subsampling convolutions; zero blocks leave only their norms.
  std::vector<float> conv0(36,0); conv0[4]=1; conv0[13]=-1;
  put(audio_prefix+"layer0.conv.weight",conv0);
  std::vector<float> conv1(72,0);conv1[4]=1;conv1[36+13]=1;
  put(audio_prefix+"layer1.conv.weight",conv1);
  put(audio_prefix+"input_proj_linear.weight",{1,0,0,1,0,0,0,0});
  for(int l=0;l<2;++l) for(const auto& e:ai(4,2)) {
    size_t n=1; for(auto d:e.shape)n*=d;
    const bool norm=std::string(e.name).find("norm")!=std::string::npos;
    put("model.audio_tower.layers."+std::to_string(l)+"."+e.name,std::vector<float>(n,norm?1:0));
  }
  put("model.audio_tower.output_proj.weight",{1,0,0,0, 0,1,0,0, 0,0,0,0, 0,0,0,0, 0,0,0,0});
  put("model.audio_tower.output_proj.bias",std::vector<float>(5,0));
  put(audio_proj,{1,0,0,0,0, 0,1,0,0,0});
  auto shards=std::make_shared<const std::vector<vllm::SafetensorsFile>>(f.open());
  auto w=vllm::LoadGemma4ForConditionalGenerationWeightsOwned(shards,c);
  const vllm::ModelRegistration registration{"Gemma4ForConditionalGeneration",&vllm::kGemma4MmHooks,{}};
  vllm::Gemma4LoadedModel model(registration,std::move(w)); shards.reset();
  vt::Queue q{}; vllm::multimodal::MultiModalFeatureSpec item;
  item.modality="audio";item.length=2;item.audio_data=std::make_shared<vllm::multimodal::AudioKwargs>();
  auto& a=*item.audio_data;a.n_frames=8;a.n_mels=4;a.feature_mask={1,1,1,1,1,0,0,0};
  a.input_features=std::vector<float>(32,1);
  // Mel zero at the first output sample is negative, while flat row-major
  // reinterpretation would sample positive data for the second output.
  a.input_features[0]=-1;a.input_features[4]=-1;
  auto out=vllm::ModelRegistry::EncodeMm(model,c,q,item);
  CHECK(out.embeds.Ptr<float>()[0]==doctest::Approx(0));
  CHECK(out.embeds.Ptr<float>()[1]>2.2f);
  CHECK(out.embeds.Ptr<float>()[2]==doctest::Approx(0));
  CHECK(out.embeds.Ptr<float>()[3]>2.2f);
  // Independent scalar: input [0,v,0,0], two RMS norms, output selects it,
  // then weightless RMS over five channels -> sqrt(5) to within eps effects.
  CHECK(out.embeds.Ptr<float>()[1]==doctest::Approx(std::sqrt(5.0f)).epsilon(0.001));
  const auto padded=a.input_features;
  a.input_features.clear();for(int m=0;m<4;++m)for(int t=0;t<5;++t)a.input_features.push_back(padded[m*8+t]);
  a.n_frames=5;a.feature_mask.clear();
  auto trimmed=vllm::ModelRegistry::EncodeMm(model,c,q,item);
  for(int i=0;i<4;++i) CHECK(trimmed.embeds.Ptr<float>()[i]==out.embeds.Ptr<float>()[i]);
  a.input_features[0]=1;a.input_features[4]=1;
  auto changed=vllm::ModelRegistry::EncodeMm(model,c,q,item);
  CHECK(changed.embeds.Ptr<float>()[0]>2.2f);
  CHECK(changed.embeds.Ptr<float>()[1]==doctest::Approx(0));
}

TEST_CASE("E4B execution: small nonzero blocks distinguish fixed and configured epsilon") {
  // HF a08ace4: RMSNorm :198, FF :423-424, layer :535-537 use 1e-6;
  // frontend :394-399 and light-conv :499-500 use config epsilon.
  // One token, one active hidden channel: attention probability is exactly 1.
  // All stored weights below are exactly representable in BF16. No production
  // math helper or capture is used to compute the independent scalar oracle.
  for (double eps : {0.0001, 0.01}) for (bool owned : {false, true}) {
    CAPTURE(eps); CAPTURE(owned);
    Fixture f; auto c=audio_config(); add_audio(f,4,4,2); f.add(audio_proj,{2,5});
    c.raw["audio_config"]["rms_norm_eps"]=eps;
    c.raw["audio_config"]["use_clipped_linears"]=false;
    auto sparse=[&](const std::string& name,
                    std::initializer_list<std::pair<size_t,float>> entries) {
      const auto start=f.header[name]["data_offsets"][0].get<size_t>();
      const auto end=f.header[name]["data_offsets"][1].get<size_t>();
      std::fill(f.bytes.begin()+start,f.bytes.begin()+end,0);
      for (auto [i,x]:entries) {
        REQUIRE(start+2*i+2<=end);
        const auto bits=vt::F32ToBF16(x);
        std::memcpy(f.bytes.data()+start+2*i,&bits,2);
      }
    };
    sparse(audio_prefix+"layer0.conv.weight",{{4,0.125f}});
    sparse(audio_prefix+"layer0.norm.weight",{{0,1}});
    sparse(audio_prefix+"layer1.conv.weight",{{4,0.125f}});
    sparse(audio_prefix+"layer1.norm.weight",{{0,1}});
    sparse(audio_prefix+"input_proj_linear.weight",{{0,1.0f/128}});
    for (int l=0;l<2;++l) {
      const auto p="model.audio_tower.layers."+std::to_string(l)+".";
      for (const auto& e:ai()) {
        if (e.name.find("norm")!=std::string::npos)
          sparse(p+e.name,{{0,1.0f/128}});
        else if (e.name=="lconv1d.linear_end.linear.weight")
          sparse(p+e.name,{{0,8}}); // expose both configured light-conv norms
        else if (e.name=="lconv1d.depthwise_conv1d.weight")
          sparse(p+e.name,{{4,0.125f}}); // last causal tap
        else if (e.name=="lconv1d.linear_start.linear.weight")
          sparse(p+e.name,{{0,0.125f},{16,0.125f}}); // GLU value and gate
        else sparse(p+e.name,{{0,0.125f}});
      }
    }
    sparse("model.audio_tower.output_proj.weight",{{0,0.125f}});
    sparse("model.audio_tower.output_proj.bias",{{1,1.0f/256}});
    sparse(audio_proj,{{0,1},{6,1}});
    auto weights=load_audio_fixture(f,c,owned);
    const vllm::ModelRegistration registration{
        "Gemma4ForConditionalGeneration",&vllm::kGemma4MmHooks,{}};
    vllm::Gemma4LoadedModel model(registration,std::move(weights));
    vt::Queue q{}; vllm::multimodal::MultiModalFeatureSpec item;
    item.modality="audio"; item.length=1;
    item.audio_data=std::make_shared<vllm::multimodal::AudioKwargs>();
    auto& a=*item.audio_data; a.n_frames=1; a.n_mels=4;
    a.input_features={1.0f/32,0,0,0};
    const auto out=vllm::ModelRegistry::EncodeMm(model,c,q,item);
    REQUIRE(out.embeds.shape[0]==1); REQUIRE(out.embeds.shape[1]==2);

    // LayerNorm([u,0,0,0]) then LayerNorm([v,0]), ReLU and projection.
    const double u=(1.0/32)/8;
    const double v=(3*u/4)/std::sqrt(3*u*u/16+eps)/8;
    double x=(v/2)/std::sqrt(v*v/4+eps)/128;
    const auto rms=[](double z,double e) {return z/std::sqrt(z*z/4+e)/128;};
    const auto silu=[](double z) {return z/(1+std::exp(-z));};
    const auto ff=[&](double z) {
      return z+0.5*rms(silu(rms(z,1e-6)/8)/8,1e-6);
    };
    for (int l=0;l<2;++l) {
      x=ff(x);
      x+=rms(rms(x,1e-6)/64,1e-6); // V, one-token attention, post
      const double g=rms(x,eps)/8;
      x+=8*silu(rms((g/(1+std::exp(-g)))/8,eps));
      x=rms(ff(x),1e-6);
    }
    const double y=x/8, bias=1.0/256;
    const double denom=std::sqrt((y*y+bias*bias)/5+eps);
    const double expected0=y/denom, expected1=bias/denom;
    CAPTURE(expected0); CAPTURE(expected1);
    CHECK(out.embeds.Ptr<float>()[0]==doctest::Approx(expected0).epsilon(2e-5));
    CHECK(out.embeds.Ptr<float>()[1]==doctest::Approx(expected1).epsilon(2e-5));
  }
}

TEST_CASE("E4B preprocessing: waveform reaches loaded 128-mel encoder") {
  Fixture f; auto c=audio_config();
  c.raw["audio_config"]["subsampling_conv_channels"]={128,2};
  add_audio(f,4,128,2); f.add(audio_proj,{2,5});
  auto set=[&](const std::string& name, const std::vector<float>& values) {
    const size_t start=f.header[name]["data_offsets"][0].get<size_t>();
    for(size_t i=0;i<values.size();++i) {
      auto bits=vt::F32ToBF16(values[i]); std::memcpy(f.bytes.data()+start+2*i,&bits,2);
    }
  };
  set("model.audio_tower.output_proj.weight",std::vector<float>(20,0));
  set("model.audio_tower.output_proj.bias",{1,2,3,4,5});
  set(audio_proj,{1,0,0,0,0, 0,0,0,0,1});
  // H=2: BF16 sqrt(2) is 0x3fb5 (181/128). The selected text
  // coordinate 0x3f06 (67/128) gives 12127/16384, rounding to 0x3f3d.
  // Using unrounded sqrt(2) instead rounds to 0x3f3e. See the spec's
  // bounded shared merge contract and Gemma ForwardBody/EmbedScaledBf16.
  const size_t text_offset=f.header["model.language_model.embed_tokens.weight"]["data_offsets"][0].get<size_t>();
  const uint16_t scale_probe=0x3f06;
  std::memcpy(f.bytes.data()+text_offset+3*sizeof(uint16_t),&scale_probe,sizeof(scale_probe));
  auto shards=f.open(); auto w=vllm::LoadGemma4ForConditionalGenerationWeights(shards,c);
  const vllm::ModelRegistration registration{"Gemma4ForConditionalGeneration",&vllm::kGemma4MmHooks,{}};
  vllm::Gemma4LoadedModel model(registration,std::move(w));
  vllm::multimodal::Gemma4AudioProcessor processor;
  std::vector<float> waveform(800,0); waveform[17]=.5;
  vllm::multimodal::MultiModalFeatureSpec item; item.modality="audio";
  item.audio_data=std::make_shared<vllm::multimodal::AudioKwargs>(processor.ProcessWaveform(waveform.data(),800,16000));
  // Five padded frames but four valid: token count must NOT round padded frames.
  REQUIRE(item.audio_data->n_frames==5); REQUIRE(item.audio_data->num_tokens==1);
  CHECK(item.audio_data->feature_mask==std::vector<int32_t>{1,1,1,1,0});
  item.length=1; vt::Queue q{};
  const auto out=vllm::ModelRegistry::EncodeMm(model,c,q,item);
  REQUIRE(out.embeds.shape[0]==1); REQUIRE(out.embeds.shape[1]==2);
  CHECK(out.embeds.Ptr<float>()[0]==doctest::Approx(1/std::sqrt(11.0001)));
  CHECK(out.embeds.Ptr<float>()[1]==doctest::Approx(5/std::sqrt(11.0001)));
  std::vector<int32_t> ids{1,2,0};
  std::vector<char> mask{0,1,0};
  std::vector<vt::Tensor> slices{out.embeds};
  vllm::MmEmbedInputs inputs; inputs.token_ids=&ids;
  inputs.is_mm_embed=&mask; inputs.mm_embeds=&slices;
  c.raw["text_config"]["vocab_size_per_layer_input"]=1;
  auto merged=vllm::ModelRegistry::EmbedMm(model,c,q,inputs);
  REQUIRE(merged.mm.inputs_embeds.dtype==vt::DType::kBF16);
  REQUIRE(merged.mm.inputs_embeds.shape[0]==3);
  REQUIRE(merged.mm.ple_token_ids.dtype==vt::DType::kI32);
  CHECK(merged.mm.inputs_embeds.Ptr<uint16_t>()[0]==0x3fb5);
  CHECK(merged.mm.inputs_embeds.Ptr<uint16_t>()[1]==0x3f3d);
  CHECK(merged.mm.inputs_embeds.Ptr<uint16_t>()[4]==0x3fb5);
  CHECK(merged.mm.inputs_embeds.Ptr<uint16_t>()[2]==vt::F32ToBF16(1/std::sqrt(11.0001)));
  CHECK(merged.mm.inputs_embeds.Ptr<uint16_t>()[3]==vt::F32ToBF16(5/std::sqrt(11.0001)));
  for(int i=0;i<3;++i) CHECK(merged.mm.ple_token_ids.Ptr<int32_t>()[i]==0);
  item.length=2; CHECK_THROWS_WITH(vllm::ModelRegistry::EncodeMm(model,c,q,item),doctest::Contains("placeholder length"));
  item.length=0;
  *item.audio_data=processor.ProcessWaveform(waveform.data(),160,16000);
  CHECK_THROWS_WITH(vllm::ModelRegistry::EncodeMm(model,c,q,item),doctest::Contains("no valid frames"));
  CHECK(vllm::ModelRegistry::SupportsMmInputs(model));
}

TEST_CASE("E4B merge: ordered chunk slices validation ownership and readiness") {
  Fixture f; auto c=audio_config(); add_audio(f,4,4,2); f.add(audio_proj,{2,5});
  // Nonuniform checkpoint rows expose incorrect token lookup/order and scaling.
  // Token 1 is initially masked; the text-only call below exposes the same
  // H=2 rounding discriminator (67/128)*(181/128) -> BF16 0x3f3d.
  const uint16_t table[]{0x3f80,0x4000,0x3f06,0x4080,0x40a0,0x40c0};
  const size_t offset=f.header["model.language_model.embed_tokens.weight"]["data_offsets"][0].get<size_t>();
  std::memcpy(f.bytes.data()+offset,table,sizeof(table));
  auto shards=f.open();
  auto w=vllm::LoadGemma4ForConditionalGenerationWeights(shards,c);
  const vllm::ModelRegistration reg{"Gemma4ForConditionalGeneration",&vllm::kGemma4MmHooks,{}};
  vllm::Gemma4LoadedModel model(reg,std::move(w)); vt::Queue q{};
  std::vector<int32_t> ids{2,1,0,2}; std::vector<char> mask{0,1,1,0};
  std::vector<float> full{99,99,2,3,4,5};
  auto tensor=vt::Tensor::Contiguous(full.data()+2,vt::DType::kF32,q.device,{1,2});
  auto second=tensor; second.data=full.data()+4;
  std::vector<vt::Tensor> slices{tensor,second};
  vllm::MmEmbedInputs in; in.token_ids=&ids; in.is_mm_embed=&mask; in.mm_embeds=&slices;
  auto call=[&]{return vllm::ModelRegistry::EmbedMm(model,c,q,in);};
  auto result=call();
  CHECK(result.mm.inputs_embeds.Ptr<uint16_t>()[0]==0x40e2);
  CHECK(result.mm.inputs_embeds.Ptr<uint16_t>()[1]==0x4108);
  CHECK(result.mm.inputs_embeds.Ptr<uint16_t>()[2]==0x4000);
  CHECK(result.mm.inputs_embeds.Ptr<uint16_t>()[4]==0x4080);
  CHECK(result.mm.ple_token_ids.Ptr<int32_t>()[0]==2);
  CHECK(result.mm.ple_token_ids.Ptr<int32_t>()[1]==0);
  CHECK(result.mm.ple_token_ids.Ptr<int32_t>()[2]==0);
  CHECK(result.mm.ple_token_ids.Ptr<int32_t>()[3]==2);
  full[2]=8; ids[0]=0;
  CHECK(result.mm.inputs_embeds.Ptr<uint16_t>()[2]==0x4000);
  CHECK(result.mm.ple_token_ids.Ptr<int32_t>()[0]==2);
  for(int bad:{-1,3}) {ids[0]=bad; CHECK_THROWS(call());} ids[0]=2;
  mask[0]=2; CHECK_THROWS(call()); mask[0]=0;
  mask.pop_back(); CHECK_THROWS(call()); mask.push_back(0);
  slices.push_back(second); CHECK_THROWS(call()); slices.pop_back();
  slices.pop_back(); CHECK_THROWS(call()); slices.push_back(second);
  for(int mode=0;mode<7;++mode) {
    auto saved=slices[0];
    if(mode==0)slices[0].rank=1;
    if(mode==1)slices[0].shape[0]=0;
    if(mode==2)slices[0].shape[1]=3;
    if(mode==3)slices[0].dtype=vt::DType::kBF16;
    if(mode==4)slices[0].stride[0]=3;
    if(mode==5)slices[0].data=nullptr;
    if(mode==6)slices[0].device.type=vt::DeviceType::kCUDA;
    CHECK_THROWS(call()); slices[0]=saved;
  }
  full[2]=std::numeric_limits<float>::infinity(); CHECK_THROWS(call());full[2]=2;
  std::vector<int32_t> positions{0};in.mrope_positions=&positions;CHECK_THROWS(call());in.mrope_positions=nullptr;
  in.host_token_ids_stale=true;CHECK_THROWS(call());in.host_token_ids_stale=false;
  q.device.type=vt::DeviceType::kCUDA;CHECK_THROWS(call());q.device.type=vt::DeviceType::kCPU;
  ids.clear();mask.clear();slices.clear();CHECK_THROWS(call());
  ids={1};mask={0};auto text=call();CHECK(text.mm.ple_token_ids.Ptr<int32_t>()[0]==1);
  CHECK(text.mm.inputs_embeds.Ptr<uint16_t>()[0]==0x3f3d);
  CHECK(vllm::ModelRegistry::SupportsMmInputs(model));
  vllm::ModelFactory factory{};
  const vllm::ModelRegistration other{"Gemma4ForConditionalGeneration",&factory,{}};
  vllm::Gemma4LoadedModel legacy(other,vllm::Gemma4Weights{});
  CHECK(factory.mm_inputs_ready);
  CHECK_FALSE(vllm::ModelRegistry::SupportsMmInputs(legacy));
  factory.encode_mm=vllm::kGemma4MmHooks.encode_mm;
  CHECK_FALSE(vllm::ModelRegistry::SupportsMmInputs(legacy));
  factory.embed_mm=vllm::kGemma4MmHooks.embed_mm;
  CHECK(vllm::ModelRegistry::SupportsMmInputs(legacy));
  factory.mm_inputs_ready=false;
  CHECK_FALSE(vllm::ModelRegistry::SupportsMmInputs(legacy));
}

// T09: the factory the registry uses advertises the multimodal input path.
// kGemma4Factory copies kGemma4MmHooks as its base, so the mm_inputs_ready
// flag in kGemma4MmHooks propagates to the factory the runner sees.  Before
// the flip this CHECK fails: mm_inputs_ready is false and SupportsMmInputs
// returns false even though encode_mm and embed_mm are both non-null.
TEST_CASE("E4B T09: kGemma4MmHooks advertises mm_inputs_ready to the runner") {
  CHECK(vllm::kGemma4MmHooks.mm_inputs_ready);
  CHECK(vllm::kGemma4MmHooks.encode_mm != nullptr);
  CHECK(vllm::kGemma4MmHooks.embed_mm != nullptr);
  // A model constructed with kGemma4MmHooks must pass SupportsMmInputs —
  // the gate the runner checks before dispatching multimodal inputs.
  vllm::Gemma4Weights w;
  const vllm::ModelRegistration reg{"Gemma4ForConditionalGeneration",
                                    &vllm::kGemma4MmHooks, {}};
  vllm::Gemma4LoadedModel model(reg, std::move(w));
  CHECK(vllm::ModelRegistry::SupportsMmInputs(model));
}
