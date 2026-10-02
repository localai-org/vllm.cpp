// MODEL-CLM: CLM against the Contrastive-LM/CLM reference
// (.agents/specs/clm.md, "Fidelity repair").
//
// Evidence, in order of how much it rests on the reference:
//   1. clm_goldens.inc is produced by RUNNING the reference @ bb42c6c
//      (scripts/gen-clm-goldens.py): make_head outputs, HeadPair's scale,
//      build_pairs texts, answer_from_logits, Engine.answer refusals, and a
//      whole Engine.answer run over a tiny head with raw embeddings.
//   2. The heads load through the real loader from a safetensors file with
//      the checkpoint's own tensor names.
//   3. A tiny ClmModel directory (the converter's layout) loads through
//      vllm_engine_load, and vllm_decide must equal clm::Answer over an
//      independent ForwardHidden last row.
// With VLLM_CPP_CLM_MODEL_DIR pointing at a converted CLM-v0.1-8B, the real
// checkpoint must also reproduce the reference's answers on five requests.
//
// The first port's 13 tests built synthetic weights in the engine's own
// layout and recomputed the confidence formula inline, so they checked the
// engine against itself; none of them is kept.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "clm_goldens.inc"
#include "qwen3_5_decision_fixture.h"
#include "vllm.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/clm.h"
#include "vllm/model_executor/models/clm_inference.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace {

namespace fs = std::filesystem;
using ojson = nlohmann::ordered_json;
using vllm::clm::HeadParams;

const ojson& Goldens() {
  static const ojson g = ojson::parse(kClmGoldens);
  return g;
}

// ── Scratch safetensors writer ──────────────────────────────────────────────
struct StEntry {
  std::string name;
  std::string dtype;  // "F32" | "BF16"
  std::vector<int64_t> shape;
  std::vector<float> data;
};

void WriteSafetensors(const std::vector<StEntry>& entries, const fs::path& path) {
  nlohmann::json header = nlohmann::json::object();
  std::string blob;
  for (const StEntry& e : entries) {
    std::string bytes;
    if (e.dtype == "F32") {
      bytes.resize(e.data.size() * 4);
      std::memcpy(bytes.data(), e.data.data(), bytes.size());
    } else {
      bytes.resize(e.data.size() * 2);
      for (size_t i = 0; i < e.data.size(); ++i) {
        const uint16_t b = vt::F32ToBF16(e.data[i]);
        std::memcpy(bytes.data() + 2 * i, &b, 2);
      }
    }
    header[e.name] = {{"dtype", e.dtype},
                      {"shape", e.shape},
                      {"data_offsets", {blob.size(), blob.size() + bytes.size()}}};
    blob += bytes;
  }
  std::string hs = header.dump();
  while ((8 + hs.size()) % 8 != 0) hs += ' ';  // aligned, like the reference writer
  std::ofstream out(path, std::ios::binary);
  const uint64_t n = hs.size();
  out.write(reinterpret_cast<const char*>(&n), 8);
  out.write(hs.data(), static_cast<std::streamsize>(hs.size()));
  out.write(blob.data(), static_cast<std::streamsize>(blob.size()));
}

struct ScratchDir {
  fs::path dir;
  explicit ScratchDir(const std::string& tag) {
    dir = fs::temp_directory_path() /
          ("vllm_clm_" + tag + "_" + std::to_string(std::random_device{}()));
    fs::create_directories(dir);
  }
  ~ScratchDir() {
    std::error_code ec;
    fs::remove_all(dir, ec);
  }
};

HeadParams ParamsOf(const ojson& cfg, double logit_scale = 0.0) {
  HeadParams p;
  p.hidden_size = cfg["hidden"].get<int64_t>();
  p.width = cfg["width"].get<int64_t>();
  p.depth = cfg["depth"].get<int64_t>();
  p.proj_dim = cfg["proj"].get<int64_t>();
  p.layernorm = cfg["layernorm"].get<bool>();
  p.residual = cfg["residual"].get<bool>();
  const std::string a = cfg["activation"].get<std::string>();
  p.activation = a == "gelu" ? vllm::clm::Activation::kGelu
                 : a == "relu" ? vllm::clm::Activation::kRelu
                               : vllm::clm::Activation::kSilu;
  p.logit_scale = logit_scale;
  return p;
}

// The golden's state-dict tensors under <prefix>., F32.
void AddHead(std::vector<StEntry>& out, const std::string& prefix, const ojson& tensors) {
  for (auto it = tensors.begin(); it != tensors.end(); ++it) {
    out.push_back({prefix + "." + it.key(), "F32",
                   it.value()["shape"].get<std::vector<int64_t>>(),
                   it.value()["data"].get<std::vector<float>>()});
  }
}

vllm::clm::HeadWeights LoadHeads(const std::vector<StEntry>& entries,
                                 const HeadParams& p) {
  ScratchDir d("head");
  WriteSafetensors(entries, d.dir / "head.safetensors");
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open((d.dir / "head.safetensors").string()));
  return vllm::clm::LoadHeadWeights(shards, p);
}

// Same keys in the same order; strings and objects equal; numbers to `eps`.
void CheckJson(const ojson& got, const ojson& want, double eps) {
  if (want.is_object()) {
    REQUIRE(got.is_object());
    std::vector<std::string> gk, wk;
    for (auto it = got.begin(); it != got.end(); ++it) gk.push_back(it.key());
    for (auto it = want.begin(); it != want.end(); ++it) wk.push_back(it.key());
    CHECK(gk == wk);
    for (auto it = want.begin(); it != want.end(); ++it) {
      if (got.contains(it.key())) CheckJson(got[it.key()], it.value(), eps);
    }
  } else if (want.is_number()) {
    REQUIRE(got.is_number());
    CHECK(std::abs(got.get<double>() - want.get<double>()) <= eps);
  } else {
    CHECK(got == want);
  }
}

// Embeddings from the golden: the raw (not normalized) vector the reference
// run handed its l2. A text the golden has no vector for is a wrong text.
vllm::clm::EmbedFn GoldenEmbed(const ojson& vectors) {
  return [&vectors](const std::string& text, int64_t* tokens) {
    if (!vectors.contains(text)) {
      throw std::runtime_error("the reference never encoded the text: " + text);
    }
    *tokens += 1;
    return vectors[text].get<std::vector<float>>();
  };
}

}  // namespace

// ── 1. Reference goldens ────────────────────────────────────────────────────

TEST_CASE("clm.heads.match_make_head_loaded_by_the_checkpoint_names") {
  for (const ojson& c : Goldens()["heads"]) {
    const HeadParams p = ParamsOf(c["cfg"]);
    std::vector<StEntry> e;
    AddHead(e, "state_head", c["tensors"]);
    AddHead(e, "action_head", c["tensors"]);
    const vllm::clm::HeadWeights hw = LoadHeads(e, p);
    for (size_t i = 0; i < c["inputs"].size(); ++i) {
      const auto x = c["inputs"][i].get<std::vector<float>>();
      const auto want = c["outputs"][i].get<std::vector<float>>();
      for (const auto* head : {&hw.state_head, &hw.action_head}) {
        const std::vector<float> got = vllm::clm::ClmMlpHeadForward(*head, p, x);
        REQUIRE(got.size() == want.size());
        for (size_t k = 0; k < want.size(); ++k) CHECK(got[k] == doctest::Approx(want[k]).epsilon(1e-5));
      }
    }
  }
}

TEST_CASE("clm.heads.the_first_ports_tensor_names_are_refused_by_name") {
  const ojson& c = Goldens()["heads"][0];
  const HeadParams p = ParamsOf(c["cfg"]);
  const std::map<std::string, std::string> old_names = {
      {"inp", "0"}, {"norms.0", "2"}, {"hidden.0", "4"}, {"out", "6"}};
  std::vector<StEntry> e;
  for (auto it = c["tensors"].begin(); it != c["tensors"].end(); ++it) {
    const std::string k = it.key();
    const std::string module = k.substr(0, k.rfind('.'));
    const std::string param = k.substr(k.rfind('.') + 1);
    for (const char* prefix : {"state_head", "action_head"}) {
      e.push_back({std::string(prefix) + "." + old_names.at(module) + "." + param, "F32",
                   it.value()["shape"].get<std::vector<int64_t>>(),
                   it.value()["data"].get<std::vector<float>>()});
    }
  }
  CHECK_THROWS_WITH_AS(LoadHeads(e, p),
                       doctest::Contains("'state_head.inp.weight' is missing"),
                       std::runtime_error);
}

TEST_CASE("clm.heads.a_missing_config_key_is_refused_not_defaulted") {
  nlohmann::json raw = {{"clm_hidden_size", 8}, {"clm_width", 6}, {"clm_depth", 3},
                        {"clm_projection_dim", 4}, {"clm_activation", "gelu"},
                        {"clm_layernorm", true}, {"clm_residual", false},
                        {"clm_logit_scale", 4.6}};
  CHECK(vllm::clm::ParseHeadParams(raw, 8).logit_scale == 4.6);
  raw.erase("clm_logit_scale");
  CHECK_THROWS_WITH_AS(vllm::clm::ParseHeadParams(raw, 8),
                       doctest::Contains("clm_logit_scale"), std::runtime_error);
}

TEST_CASE("clm.scale.is_exp_then_clamp_as_HeadPair_computes_it") {
  for (const ojson& c : Goldens()["scale"]) {
    INFO("logit_scale " << c["logit_scale"].get<double>());
    CHECK(vllm::clm::ClmScale(c["logit_scale"].get<double>()) ==
          doctest::Approx(c["scale"].get<double>()).epsilon(1e-7));
  }
  // The published checkpoint's value: the clamp applies after exp.
  CHECK(vllm::clm::ClmScale(Goldens()["scale"][0]["logit_scale"].get<double>()) == 100.0F);
}

TEST_CASE("clm.pairs.match_build_pairs") {
  for (const ojson& c : Goldens()["pairs"]) {
    for (auto it = c["pairs"].begin(); it != c["pairs"].end(); ++it) {
      INFO("question " << it.key());
      const vllm::clm::Pair p =
          vllm::clm::BuildPair(c["body"]["state"], c["body"]["questions"][it.key()]);
      CHECK(p.state_text == it.value()["state_text"].get<std::string>());
      CHECK(p.keys == it.value()["keys"].get<std::vector<std::string>>());
      CHECK(p.candidates == it.value()["candidates"].get<std::vector<std::string>>());
    }
  }
}

TEST_CASE("clm.answers.match_answer_from_logits") {
  for (const ojson& c : Goldens()["answers"]) {
    const vllm::clm::Pair p = vllm::clm::BuildPair("s", c["question"]);
    const ojson got = vllm::clm::AnswerFromLogits(
        c["question"], p.keys, c["logits"].get<std::vector<double>>());
    CheckJson(got, c["answer"], 1e-12);
  }
}

TEST_CASE("clm.request.refuses_what_the_reference_refuses") {
  const vllm::clm::HeadWeights none;
  const HeadParams p;
  auto never = [](const std::string&, int64_t*) -> std::vector<float> {
    throw std::logic_error("a refused request must not reach the encoder");
  };
  for (const ojson& c : Goldens()["refusals"]) {
    INFO(c["body"].dump());
    CHECK_THROWS_WITH_AS(vllm::clm::Answer(none, p, c["body"], never),
                         c["error"].get<std::string>().c_str(), vllm::clm::RequestError);
  }
}

TEST_CASE("clm.pipeline.matches_Engine_answer_from_raw_embeddings") {
  const ojson& g = Goldens()["pipeline"];
  const HeadParams p = ParamsOf(g["cfg"], g["logit_scale"].get<double>());
  std::vector<StEntry> e;
  AddHead(e, "state_head", g["state_head"]);
  AddHead(e, "action_head", g["action_head"]);
  const vllm::clm::HeadWeights hw = LoadHeads(e, p);
  for (const ojson& c : g["cases"]) {
    INFO(c["body"].dump());
    const ojson got = vllm::clm::Answer(hw, p, c["body"], GoldenEmbed(g["vectors"]));
    CheckJson(got["answers"], c["answers"], 1e-5);
    CHECK(got["usage"]["billing_units"] == c["body"]["questions"].size());
  }
}

// ── 2. A ClmModel directory through vllm_engine_load and vllm_decide ────────

namespace {

vllm::HfConfig TinyConfig() {
  vllm::HfConfig c;
  c.num_hidden_layers = 2;
  c.hidden_size = 8;  // the pipeline golden's head width in
  c.num_attention_heads = 2;
  c.num_key_value_heads = 1;
  c.head_dim = 8;
  c.rotary_dim = 8;
  c.intermediate_size = 16;
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 1000000.0;
  c.vocab_size = 260;  // the byte tokenizer's ids
  return c;
}

std::vector<float> Fill(int64_t n, uint32_t seed, float scale) {
  std::mt19937 rng(seed);
  std::uniform_real_distribution<float> d(-scale, scale);
  std::vector<float> v(static_cast<size_t>(n));
  for (float& x : v) x = d(rng);
  return v;
}

// The converter's layout: config.json naming ClmModel with the clm_* keys,
// the base's shards, head.safetensors, and the tokenizer.
void WriteTinyClmDir(const fs::path& dir) {
  const vllm::HfConfig c = TinyConfig();
  const ojson& g = Goldens()["pipeline"];
  const int64_t H = c.hidden_size, Dh = c.head_dim, I = c.intermediate_size, V = c.vocab_size;
  const int64_t q = c.num_attention_heads * Dh, kv = c.num_key_value_heads * Dh;
  std::vector<StEntry> base;
  uint32_t seed = 1;
  auto add = [&](const std::string& n, std::vector<int64_t> shape, float scale, float bias = 0.0F) {
    int64_t numel = 1;
    for (int64_t d : shape) numel *= d;
    std::vector<float> v = Fill(numel, seed++, scale);
    for (float& x : v) x += bias;
    base.push_back({n, "BF16", std::move(shape), std::move(v)});
  };
  add("model.embed_tokens.weight", {V, H}, 0.5F);
  add("model.norm.weight", {H}, 0.2F, 1.0F);
  add("lm_head.weight", {V, H}, 0.3F);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const std::string pre = "model.layers." + std::to_string(l) + ".";
    add(pre + "input_layernorm.weight", {H}, 0.2F, 1.0F);
    add(pre + "post_attention_layernorm.weight", {H}, 0.2F, 1.0F);
    add(pre + "self_attn.q_proj.weight", {q, H}, 0.4F);
    add(pre + "self_attn.k_proj.weight", {kv, H}, 0.4F);
    add(pre + "self_attn.v_proj.weight", {kv, H}, 0.4F);
    add(pre + "self_attn.o_proj.weight", {H, q}, 0.4F);
    add(pre + "self_attn.q_norm.weight", {Dh}, 0.2F, 1.0F);
    add(pre + "self_attn.k_norm.weight", {Dh}, 0.2F, 1.0F);
    add(pre + "mlp.gate_proj.weight", {I, H}, 0.4F);
    add(pre + "mlp.up_proj.weight", {I, H}, 0.4F);
    add(pre + "mlp.down_proj.weight", {H, I}, 0.4F);
  }
  WriteSafetensors(base, dir / "model.safetensors");
  std::vector<StEntry> heads;
  AddHead(heads, "state_head", g["state_head"]);
  AddHead(heads, "action_head", g["action_head"]);
  WriteSafetensors(heads, dir / "head.safetensors");
  const ojson& cfg = g["cfg"];
  const nlohmann::json config = {
      {"architectures", {"ClmModel"}}, {"model_type", "qwen3"},
      {"hidden_size", H}, {"num_hidden_layers", c.num_hidden_layers},
      {"num_attention_heads", c.num_attention_heads},
      {"num_key_value_heads", c.num_key_value_heads}, {"head_dim", Dh},
      {"intermediate_size", I}, {"vocab_size", V}, {"rms_norm_eps", 1e-6},
      {"rope_theta", 1000000.0}, {"max_position_embeddings", 4096},
      {"tie_word_embeddings", false}, {"attention_bias", false},
      {"hidden_act", "silu"}, {"torch_dtype", "bfloat16"},
      {"clm_hidden_size", H}, {"clm_width", cfg["width"]}, {"clm_depth", cfg["depth"]},
      {"clm_projection_dim", cfg["proj"]}, {"clm_activation", cfg["activation"]},
      {"clm_layernorm", cfg["layernorm"]}, {"clm_residual", cfg["residual"]},
      {"clm_logit_scale", g["logit_scale"]}};
  std::ofstream(dir / "config.json") << config.dump(2);
  std::ofstream(dir / "tokenizer.json") << qwen3_5_decision_fixture::ByteTokenizerJson();
}

// The independent encoder: the dense ForwardHidden over an F32 cache, and
// its LAST row.
vllm::clm::EmbedFn IndependentEmbed(const std::vector<vllm::SafetensorsFile>& shards,
                                    const vllm::HfConfig& config,
                                    const vllm::Qwen3DenseWeights& w) {
  return [&shards, &config, &w](const std::string& text, int64_t* tokens) {
    (void)shards;
    const std::vector<int32_t> ids =
        qwen3_5_decision_fixture::ByteTokenizer().EncodeWithSpecialTokens(text);
    *tokens += static_cast<int64_t>(ids.size());
    const int64_t T = static_cast<int64_t>(ids.size());
    const int64_t bs = 16, nb = (T + bs - 1) / bs;
    std::vector<std::vector<float>> buf;
    std::vector<vllm::PagedKvCache> kvs;
    for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
      buf.emplace_back(static_cast<size_t>(nb * 2 * bs * config.num_key_value_heads *
                                           config.head_dim), 0.0F);
    }
    for (auto& b : buf) {
      vllm::PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = vt::DType::kF32;
      kv.num_blocks = nb;
      kv.block_size = bs;
      kv.num_kv_heads = config.num_key_value_heads;
      kv.head_size = config.head_dim;
      kvs.push_back(kv);
    }
    vllm::v1::CommonAttentionMetadata m;
    m.num_reqs = 1;
    m.num_actual_tokens = static_cast<int>(T);
    m.query_start_loc = {0, static_cast<int32_t>(T)};
    m.query_start_loc_cpu = m.query_start_loc;
    m.seq_lens = {static_cast<int32_t>(T)};
    m.seq_lens_cpu = m.seq_lens;
    m.max_query_len = static_cast<int>(T);
    m.max_seq_len = static_cast<int>(T);
    m.block_table_num_cols = static_cast<int>(nb);
    for (int64_t b = 0; b < nb; ++b) m.block_table_tensor.push_back(static_cast<int32_t>(b));
    for (int64_t t = 0; t < T; ++t) m.slot_mapping.push_back(t);
    m.causal = true;
    std::vector<int32_t> pos(ids.size());
    std::iota(pos.begin(), pos.end(), 0);
    vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
    const vllm::ForwardLogits out =
        vllm::Qwen3DenseModel::ForwardHidden(ids, pos, m, kvs, w, config, q);
    const size_t H = static_cast<size_t>(config.hidden_size);
    return std::vector<float>(out.host.end() - static_cast<std::ptrdiff_t>(H), out.host.end());
  };
}

}  // namespace

TEST_CASE("clm.decide.vllm_decide_on_a_converted_dir_equals_an_independent_forward") {
  ScratchDir d("model");
  WriteTinyClmDir(d.dir);
  const std::string dir = d.dir.string();

  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = dir.c_str();
  vllm_engine* eng = nullptr;
  REQUIRE_MESSAGE(vllm_engine_load(&mp, &eng) == VLLM_OK, vllm_last_error());

  // Independent side: the same files through the dense loader, the head
  // loader, and an F32-cache forward whose last row is the embedding.
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open((d.dir / "head.safetensors").string()));
  shards.push_back(vllm::SafetensorsFile::Open((d.dir / "model.safetensors").string()));
  const vllm::HfConfig config = vllm::LoadHfConfig(dir + "/config.json");
  const vllm::Qwen3DenseWeights w = vllm::LoadQwen3ForCausalLMWeights(shards, config);
  const HeadParams p = vllm::clm::ParseHeadParams(config.raw, config.hidden_size);
  const vllm::clm::HeadWeights hw = vllm::clm::LoadHeadWeights(shards, p);
  const vllm::clm::EmbedFn embed = IndependentEmbed(shards, config, w);

  for (const ojson& c : Goldens()["pipeline"]["cases"]) {
    INFO(c["body"].dump());
    char* out = nullptr;
    REQUIRE_MESSAGE(vllm_decide(eng, c["body"].dump().c_str(), &out) == VLLM_OK,
                    vllm_last_error());
    const ojson got = ojson::parse(out);
    vllm_decide_free(out);
    const ojson want = vllm::clm::Answer(hw, p, c["body"], embed);
    // The engine's KV cache is bf16 where this side's is f32.
    CheckJson(got["answers"], want["answers"], 2e-3);
    CHECK(got["usage"] == want["usage"]);
  }

  // A refused request is the caller's fault, with the reference's text.
  char* out = nullptr;
  CHECK(vllm_decide(eng, R"({"state":"s","questions":{}})", &out) ==
        VLLM_ERR_INVALID_ARGUMENT);
  CHECK(std::string(vllm_last_error()).find("questions must not be empty") !=
        std::string::npos);
  vllm_engine_free(eng);
}

TEST_CASE("clm.decide.the_last_row_is_the_embedding") {
  // The tiny model's first and last rows differ, so a pooling that reads the
  // wrong row moves the answer; this pins the independent side above to the
  // row the reference's LAST pooling takes.
  ScratchDir d("rows");
  WriteTinyClmDir(d.dir);
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open((d.dir / "model.safetensors").string()));
  const vllm::HfConfig config = vllm::LoadHfConfig((d.dir / "config.json").string());
  const vllm::Qwen3DenseWeights w = vllm::LoadQwen3ForCausalLMWeights(shards, config);
  int64_t tokens = 0;
  const std::vector<float> last = IndependentEmbed(shards, config, w)("abc", &tokens);
  const std::vector<float> first = IndependentEmbed(shards, config, w)("a", &tokens);
  double diff = 0.0;
  for (size_t i = 0; i < last.size(); ++i) diff += std::abs(last[i] - first[i]);
  CHECK(diff > 1e-3);
}

// ── 3. The published checkpoint (opt-in) ───────────────────────────────────

TEST_CASE("clm.real.converted_checkpoint_reproduces_the_reference") {
  const char* dir = std::getenv("VLLM_CPP_CLM_MODEL_DIR");
  if (dir == nullptr || *dir == '\0') {
    MESSAGE("VLLM_CPP_CLM_MODEL_DIR is not set; the real-checkpoint gate is skipped");
    return;
  }
  const ojson& real = Goldens()["real"];
  REQUIRE(real.is_object());
  vllm_model_params mp = vllm_model_params_default();
  mp.model_path = dir;
  vllm_engine* eng = nullptr;
  REQUIRE_MESSAGE(vllm_engine_load(&mp, &eng) == VLLM_OK, vllm_last_error());
  double worst = 0.0;
  for (const ojson& c : real["cases"]) {
    char* out = nullptr;
    REQUIRE_MESSAGE(vllm_decide(eng, c["body"].dump().c_str(), &out) == VLLM_OK,
                    vllm_last_error());
    const ojson got = ojson::parse(out)["answers"];
    vllm_decide_free(out);
    for (auto it = c["answers"].begin(); it != c["answers"].end(); ++it) {
      const ojson& want = it.value();
      const ojson& g = got[it.key()];
      INFO(it.key());
      CHECK(g["type"] == want["type"]);
      if (want["type"] == "noul") {
        // The argmax must agree; the reference's own bf16 and fp32 runs
        // differ by up to 0.049 on these requests.
        CHECK((g["noul"].get<double>() >= 0.5) == (want["noul"].get<double>() >= 0.5));
        worst = std::max(worst, std::abs(g["noul"].get<double>() - want["noul"].get<double>()));
        continue;
      }
      if (want["type"] == "choice") CHECK(g["choice"] == want["choice"]);
      for (auto p = want["probabilities"].begin(); p != want["probabilities"].end(); ++p) {
        worst = std::max(worst, std::abs(g["probabilities"][p.key()].get<double>() -
                                         p.value().get<double>()));
      }
    }
  }
  MESSAGE("max probability difference vs the reference: " << worst);
  CHECK(worst < 0.06);
  vllm_engine_free(eng);
}
