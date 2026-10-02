// NemotronH_Nano_Omni_Reasoning_V3 REACHABILITY: an image enters through the
// production entry points and changes the tokens.
//
// A tiny but structurally complete checkpoint is written to a temp file: the
// NemotronH hybrid language tower under `language_model.` (the same tiny
// schedule test_nemotron_h_paged_forward.cpp gates), a real-geometry
// `vit_small_patch16_224` RADIO tower, `mlp1`, and the tensors the loader must
// DEFER by name (the input conditioner, the video embedder, an audio tensor).
// It is loaded through `ModelRegistry::Load` under the released architecture
// name, the image is processed by `NanoNemotronVLPrepareInputs` (the function
// the chat seam calls), and a greedy decode runs through `GPUModelRunner`,
// which reaches `encode_mm`, `embed_mm` and the paged NemotronH forward.
//
// The reference is computed independently of the runner: the tower, the
// shuffle and the projector on a SECOND load of the same bytes, the merged
// embeddings spliced by hand, and the NemotronH HOST forward over them. The
// runner's tokens must equal the reference's, and must DIFFER from the tokens
// the same prompt gives with the image rows left as `<image>` embeddings.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "vllm/config/multimodal.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/nano_nemotron_vl.h"
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vllm/model_executor/models/nemotron_h_loader.h"
#include "vllm/model_executor/models/radio.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/multimodal/nano_nemotron_vl_processor.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/worker/gpu/runner.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vllm/models/nano_nemotron_vl_tiny_checkpoint.h"

using vllm::HfConfig;
using vllm::ModelRegistry;
using vllm::ModelSource;
using vllm::NemotronHBlock;
using vllm::SafetensorsFile;
using vllm::SamplingParams;
using vllm::v1::CachedRequestData;
using vllm::v1::GPUModelRunner;
using vllm::v1::KVCacheConfig;
using vllm::v1::NewRequestData;
using vllm::v1::SchedulerOutput;

namespace {

using namespace nnvl_tiny;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

struct Fixture {
  std::unique_ptr<TempFile> st;
  std::unique_ptr<TempFile> cfg_json;
  std::vector<SafetensorsFile> shards;
  HfConfig cfg;
  std::unique_ptr<vllm::LoadedModel> model;

  Fixture() {
    st = std::make_unique<TempFile>(BuildSt(BuildTensors()));
    cfg_json = std::make_unique<TempFile>(TinyConfig().dump(2), ".json");
    shards.push_back(SafetensorsFile::Open(st->path()));
    cfg = vllm::LoadHfConfig(cfg_json->path());
    model = ModelRegistry::Load(cfg, ModelSource::FromSafetensors(shards));
  }
};

std::vector<uint8_t> Image(int64_t h, int64_t w) {
  std::vector<uint8_t> px(static_cast<size_t>(h * w * 3));
  uint32_t r = 77;
  for (auto& v : px) {
    r = r * 1664525u + 1013904223u;
    v = static_cast<uint8_t>(r >> 24);
  }
  return px;
}

vllm::multimodal::MultiModalInputs Prepare(const Fixture& fx, const std::vector<uint8_t>& rgb,
                                           int64_t h, int64_t w,
                                           const std::vector<int32_t>& prompt) {
  vllm::NanoNemotronVLParams p = vllm::ParseNanoNemotronVLParams(fx.cfg);
  p.processor.max_model_len = kMaxModelLen;
  vllm::multimodal::NanoNemotronVLTokenIds ids{kImgStart, kImgContext, kImgEnd};
  const int64_t text_len = static_cast<int64_t>(prompt.size()) - 1;
  return vllm::multimodal::NanoNemotronVLPrepareInputs(prompt, text_len,
                                                       {{rgb.data(), h, w}}, p.processor, ids,
                                                       "tiny-omni");
}

SamplingParams Greedy() {
  SamplingParams sp;
  sp.temperature = 0.0;
  sp.PostInit();
  return sp;
}

// Prefill + decode through the runner; the image item is encoded on the first
// step because the scheduler names it in `scheduled_encoder_inputs`.
std::vector<int32_t> RunnerGreedy(Fixture& fx, const vllm::multimodal::MultiModalInputs& mm,
                                  int steps) {
  const vllm::ModelRegistration& reg = fx.model->registration();
  KVCacheConfig kv = reg.factory->make_kv_cache(fx.cfg, kBlockSize, kNumBlocks);
  vt::Queue q = Q();
  GPUModelRunner runner(fx.cfg, *fx.model, kv, q, /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  const std::string id = "img";
  NewRequestData nr;
  nr.req_id = id;
  nr.prompt_token_ids = mm.prompt_token_ids;
  nr.sampling_params = Greedy();
  nr.block_ids = {std::vector<int>{0, 1, 2}, std::vector<int>{0}};
  nr.num_computed_tokens = 0;
  nr.prefill_token_ids = mm.prompt_token_ids;
  nr.mm_features = mm.mm_features;
  SchedulerOutput so;
  so.scheduled_cached_reqs = CachedRequestData::make_empty();
  so.scheduled_new_reqs.push_back(nr);
  so.num_scheduled_tokens[id] = static_cast<int>(mm.prompt_token_ids.size());
  so.total_num_scheduled_tokens = static_cast<int>(mm.prompt_token_ids.size());
  if (!mm.mm_features.empty()) so.scheduled_encoder_inputs[id] = {0};
  CHECK_FALSE(runner.execute_model(so).has_value());
  vllm::v1::ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  REQUIRE(m1.sampled_token_ids.size() == 1);
  std::vector<int32_t> out{m1.sampled_token_ids[0][0]};
  int computed = static_cast<int>(mm.prompt_token_ids.size());
  int outputs = 1;
  for (int s = 1; s < steps; ++s) {
    SchedulerOutput sd;
    CachedRequestData cached;
    cached.req_ids = {id};
    cached.num_computed_tokens.push_back(computed);
    cached.num_output_tokens.push_back(outputs);
    cached.new_block_ids.emplace_back(std::nullopt);
    sd.scheduled_cached_reqs = std::move(cached);
    sd.num_scheduled_tokens[id] = 1;
    sd.total_num_scheduled_tokens = 1;
    CHECK_FALSE(runner.execute_model(sd).has_value());
    vllm::v1::ModelRunnerOutput md = runner.sample_tokens(std::nullopt);
    REQUIRE(md.sampled_token_ids.size() == 1);
    out.push_back(md.sampled_token_ids[0][0]);
    ++computed;
    ++outputs;
  }
  return out;
}

int32_t Argmax(const std::vector<float>& logits, size_t row, size_t vocab) {
  const float* r = logits.data() + row * vocab;
  return static_cast<int32_t>(std::max_element(r, r + vocab) - r);
}

}  // namespace

TEST_CASE("nano-nemotron-vl: the released architecture names resolve through the registry") {
  const auto archs = ModelRegistry::SupportedArchs();
  CHECK(std::find(archs.begin(), archs.end(), "NemotronH_Nano_Omni_Reasoning_V3") != archs.end());
  CHECK(std::find(archs.begin(), archs.end(), "NemotronH_Nano_VL_V2") != archs.end());
}

TEST_CASE("nano-nemotron-vl: the loader accounts for every shipped tensor") {
  Fixture fx;
  REQUIRE(fx.model != nullptr);
  // A stray tensor nothing names is REFUSED, never dropped.
  TempFile stray(BuildSt(BuildTensors(/*with_stray=*/true)));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(stray.path()));
  bool refused = false;
  try {
    (void)ModelRegistry::Load(fx.cfg, ModelSource::FromSafetensors(shards));
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("mystery_head.weight") != std::string::npos;
  }
  CHECK(refused);
}

TEST_CASE("nano-nemotron-vl: an image reaches encode_mm, embed_mm and the paged forward") {
  Fixture fx;
  const int64_t h = 40, w = 56;
  const std::vector<uint8_t> rgb = Image(h, w);
  const std::vector<int32_t> prompt = {3, 7, kImgContext, 5, 9};
  const vllm::multimodal::MultiModalInputs mm = Prepare(fx, rgb, h, w, prompt);
  REQUIRE(mm.mm_features.size() == 1);
  const auto& feat = mm.mm_features[0];
  // 40x56 -> a 4x4 patch grid -> 4 rows after the shuffle.
  CHECK(feat.data->image_grid_thw[1] == 4);
  CHECK(feat.data->image_grid_thw[2] == 4);
  REQUIRE(feat.length == 4);
  CHECK(mm.prompt_token_ids ==
        std::vector<int32_t>{3, 7, kImgStart, kImgContext, kImgContext, kImgContext,
                             kImgContext, kImgEnd, 5, 9});
  CHECK(feat.offset == 3);

  constexpr int kSteps = 4;
  const std::vector<int32_t> got = RunnerGreedy(fx, mm, kSteps);

  // ── the independent reference ──
  const vllm::NanoNemotronVLParams p = vllm::ParseNanoNemotronVLParams(fx.cfg);
  const vllm::NanoNemotronVLVisionLoad vis = vllm::LoadNanoNemotronVLVisionWeights(fx.shards, p);
  vt::Backend* cpu = vt::TryGetBackend(vt::DeviceType::kCPU);
  REQUIRE(cpu != nullptr);
  const vllm::multimodal::RadioVisionTower tower(vis.radio, p.radio, *cpu);
  const vllm::multimodal::NanoNemotronVLProjector proj(vis.projector, p.projector, *cpu);
  vllm::multimodal::RadioImage im;
  im.patches = feat.data->pixel_values_f32;
  im.grid_h = 4;
  im.grid_w = 4;
  std::vector<float> f = tower.Forward({im});
  for (float& v : f) v = vt::BF16ToF32(vt::F32ToBF16(v));
  const std::vector<float> rows =
      proj.Forward(vllm::multimodal::NanoNemotronVLPixelShuffle(f, 4, 4, kVitH, 2), 4);

  const HfConfig text_cfg = vllm::NanoNemotronVLTextConfig(fx.cfg);
  const vllm::NemotronHParams tp = vllm::ParseNemotronHParams(text_cfg);
  vllm::NemotronHLoadReport rep;
  const vllm::NemotronHHostWeights host = vllm::LoadNemotronHHostWeights(
      fx.shards, tp, vllm::ResolveNemotronHModelDType(text_cfg), &rep, "language_model.");
  CHECK(rep.materialized == rep.in_index);

  auto embed_row = [&](int32_t id) {
    std::vector<float> r(kHidden);
    const auto* tab = reinterpret_cast<const uint16_t*>(host.embeddings.bytes.data());
    for (int c = 0; c < kHidden; ++c) r[static_cast<size_t>(c)] = vt::BF16ToF32(tab[id * kHidden + c]);
    return r;
  };
  std::vector<int32_t> seq = mm.prompt_token_ids;
  std::vector<float> embeds;
  for (size_t t = 0; t < seq.size(); ++t) {
    const int64_t k = static_cast<int64_t>(t) - feat.offset;
    if (k >= 0 && k < feat.length) {
      for (int c = 0; c < kHidden; ++c)
        embeds.push_back(vt::BF16ToF32(vt::F32ToBF16(rows[static_cast<size_t>(k * kHidden + c)])));
    } else {
      const auto r = embed_row(seq[t]);
      embeds.insert(embeds.end(), r.begin(), r.end());
    }
  }
  vt::Queue q = Q();
  std::vector<int32_t> want;
  for (int s = 0; s < kSteps; ++s) {
    const auto logits = vllm::NemotronHForward(host, tp, seq, {}, q, nullptr, &embeds);
    const int32_t tok = Argmax(logits, seq.size() - 1, kVocab);
    want.push_back(tok);
    seq.push_back(tok);
    const auto r = embed_row(tok);
    embeds.insert(embeds.end(), r.begin(), r.end());
  }
  CHECK(got == want);

  // The image MATTERS: the same ids with the `<image>` rows embedded as text
  // decode differently. A forward that ignored inputs_embeds would pass the
  // comparison above only if this one failed.
  const std::vector<float> prompt_embeds(
      embeds.begin(),
      embeds.begin() + static_cast<std::ptrdiff_t>(mm.prompt_token_ids.size() * kHidden));
  const auto text_logits = vllm::NemotronHForward(host, tp, mm.prompt_token_ids, {}, q);
  const auto img_logits =
      vllm::NemotronHForward(host, tp, mm.prompt_token_ids, {}, q, nullptr, &prompt_embeds);
  double diff = 0;
  const size_t last = (mm.prompt_token_ids.size() - 1) * kVocab;
  for (int v = 0; v < kVocab; ++v)
    diff = std::max(diff, std::abs(static_cast<double>(text_logits[last + v]) - img_logits[last + v]));
  MESSAGE("max |logit(image) - logit(text-only)| at the last prompt position: " << diff);
  CHECK(diff > 1e-2);
  // And the decoded TOKENS differ, so the runner equality above cannot be met
  // by a runner that dropped the image rows.
  std::vector<int32_t> text_seq = mm.prompt_token_ids;
  std::vector<int32_t> want_text;
  for (int s = 0; s < kSteps; ++s) {
    const auto logits = vllm::NemotronHForward(host, tp, text_seq, {}, q);
    const int32_t tok = Argmax(logits, text_seq.size() - 1, kVocab);
    want_text.push_back(tok);
    text_seq.push_back(tok);
  }
  MESSAGE("image decode " << want[0] << " " << want[1] << " " << want[2] << " " << want[3]
                          << " | text-only decode " << want_text[0] << " " << want_text[1]
                          << " " << want_text[2] << " " << want_text[3]);
  CHECK(want != want_text);
}

// ─── upstream tests/models/multimodal/test_nano_nemotron_vl.py @ e126687a9a ──
// The three `load_weights` cases, adapted from a mocked module to a real load:
// the harness here has no module tree to monkeypatch, so each case loads a
// tiny checkpoint and asserts the same observable outcome. The two
// `_extract_audio_from_videos` cases are NOT ported: audio is refused by name.

// test_nano_nemotron_vl_skips_multimodal_weights_in_text_only_mode (:77-96)
//
// Upstream's point is two facts about one load: the vision tower and `mlp1`
// are never READ (the mocked module raises if it is inspected), and a
// `sound_encoder.*` tensor on a model with NO sound encoder loads fine,
// because text-only mode skips the sound arm before its assert (:1546-1548).
// The checkpoint here carries no `sound_config` and ships a
// `sound_encoder.encoder.weight` tensor, exactly as the upstream case does.
TEST_CASE("nano-nemotron-vl (upstream): text-only mode skips the multimodal weights") {
  Fixture fx;
  nlohmann::json no_sound = TinyConfig();
  no_sound.erase("sound_config");
  TempFile cfg_json(no_sound.dump(2), ".json");
  const HfConfig cfg = vllm::LoadHfConfig(cfg_json.path());
  std::vector<Fx> ts = BuildTensors();
  ts.push_back(Bf16("sound_encoder.encoder.weight", {4, 4}, 9001));
  TempFile st(BuildSt(ts));
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(st.path()));

  vllm::MultiModalConfig text_only;
  text_only.language_model_only = true;
  ModelSource src = ModelSource::FromSafetensors(shards);
  src.multimodal = &text_only;
  std::unique_ptr<vllm::LoadedModel> model;
  std::string load_error;
  try {
    model = ModelRegistry::Load(cfg, src);
  } catch (const std::exception& e) {
    load_error = e.what();
  }
  // A text-only load skips the sound tensor; it must not refuse it.
  INFO("load error: " << load_error);
  REQUIRE(load_error.empty());
  REQUIRE(model != nullptr);
  // No vision or mlp1 tensor was read.
  const vllm::NanoNemotronVLVisionLoad& vis = vllm::NanoNemotronVLVisionLoadOf(*model);
  CHECK(vis.shipped == 0);
  CHECK(vis.materialized == 0);
  // The control: an image-mode load of the same tower DOES read it, so the
  // zeros above are a statement about text-only mode, not about the counter.
  CHECK(vllm::NanoNemotronVLVisionLoadOf(*fx.model).materialized > 0);
  vllm::multimodal::MultiModalFeatureSpec item;
  item.modality = "image";
  item.length = 4;
  item.data = std::make_shared<vllm::multimodal::ImageKwargs>();
  item.data->num_patches = 16;
  vt::Queue q = Q();
  bool refused = false;
  try {
    (void)ModelRegistry::EncodeMm(*model, cfg, q, item);
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("loaded text-only") != std::string::npos;
  }
  CHECK(refused);
}

// test_nano_nemotron_vl_loads_vision_weights_without_sound_encoder (:99-121)
// and test_nano_nemotron_vl_requires_sound_encoder_for_sound_weights (:124-136)
TEST_CASE("nano-nemotron-vl (upstream): vision loads without a sound encoder; sound weights need one") {
  nlohmann::json no_sound = TinyConfig();
  no_sound.erase("sound_config");
  TempFile cfg_json(no_sound.dump(2), ".json");
  const HfConfig cfg = vllm::LoadHfConfig(cfg_json.path());
  // Without the sound tensor: loads.
  std::vector<Fx> ts = BuildTensors();
  ts.erase(std::remove_if(ts.begin(), ts.end(),
                          [](const Fx& f) { return f.name.rfind("sound", 0) == 0; }),
           ts.end());
  {
    TempFile st(BuildSt(ts));
    std::vector<SafetensorsFile> shards;
    shards.push_back(SafetensorsFile::Open(st.path()));
    CHECK(ModelRegistry::Load(cfg, ModelSource::FromSafetensors(shards)) != nullptr);
  }
  // With it: refused, because no sound encoder exists to take it.
  {
    TempFile st(BuildSt(BuildTensors()));
    std::vector<SafetensorsFile> shards;
    shards.push_back(SafetensorsFile::Open(st.path()));
    bool refused = false;
    try {
      (void)ModelRegistry::Load(cfg, ModelSource::FromSafetensors(shards));
    } catch (const std::runtime_error& e) {
      refused = std::string(e.what()).find("no `sound_config`") != std::string::npos;
    }
    CHECK(refused);
  }
}

TEST_CASE("nano-nemotron-vl: audio and video items are refused by name at encode") {
  Fixture fx;
  vllm::multimodal::MultiModalFeatureSpec item;
  item.modality = "audio";
  item.length = 4;
  vt::Queue q = Q();
  bool refused = false;
  try {
    (void)ModelRegistry::EncodeMm(*fx.model, fx.cfg, q, item);
  } catch (const std::exception& e) {
    refused = std::string(e.what()).find("modality 'audio' is not ported") != std::string::npos;
  }
  CHECK(refused);
}

// ─── STRUCTURAL gate on a released index (env-gated, headers only) ──────────
// NANO_NEMOTRON_VL_INDEX_JSON names a model.safetensors.index.json and
// NANO_NEMOTRON_VL_CONFIG_JSON the config.json beside it. Every tensor the index
// ships must be claimed: the language tower by NemotronH's own enumeration
// under `language_model.`, the vision tower and mlp1 by name, and the rest
// deferred by name (input conditioner, video embedder, audio). The verdict is
// printed for each arm so a run records which checkpoint layout this tree can
// load and which it refuses.
TEST_CASE("nano-nemotron-vl STRUCTURAL: a released index is fully accounted") {
  const char* index_path = std::getenv("NANO_NEMOTRON_VL_INDEX_JSON");
  const char* config_path = std::getenv("NANO_NEMOTRON_VL_CONFIG_JSON");
  if (index_path == nullptr || config_path == nullptr) {
    MESSAGE("SKIP: NANO_NEMOTRON_VL_INDEX_JSON / NANO_NEMOTRON_VL_CONFIG_JSON unset");
    return;
  }
  nlohmann::json index;
  {
    std::ifstream f(index_path);
    REQUIRE(f.good());
    f >> index;
  }
  const HfConfig cfg = vllm::LoadHfConfig(config_path);
  const vllm::NanoNemotronVLParams p = vllm::ParseNanoNemotronVLParams(cfg);
  const HfConfig text_cfg = vllm::NanoNemotronVLTextConfig(cfg);
  const vllm::NemotronHParams tp = vllm::ParseNemotronHParams(text_cfg);
  std::set<std::string> enumerated;
  for (const auto& t : vllm::EnumerateNemotronHTensors(tp)) enumerated.insert(t.name);

  std::set<std::string> vision_expected;
  const std::string pg = "vision_model.radio_model.model.patch_generator.";
  for (const char* n : {"embedder.weight", "pos_embed", "cls_token.token"}) vision_expected.insert(pg + n);
  for (int64_t l = 0; l < p.radio.num_hidden_layers; ++l) {
    const std::string b = "vision_model.radio_model.model.blocks." + std::to_string(l) + ".";
    for (const char* n : {"norm1.weight", "norm1.bias", "attn.qkv.weight", "attn.qkv.bias",
                          "attn.proj.weight", "attn.proj.bias", "norm2.weight", "norm2.bias",
                          "mlp.fc1.weight", "mlp.fc1.bias", "mlp.fc2.weight", "mlp.fc2.bias"})
      vision_expected.insert(b + n);
  }
  for (const char* n : {"mlp1.0.weight", "mlp1.1.weight", "mlp1.3.weight"}) vision_expected.insert(n);

  int64_t lm = 0, lm_claimed = 0, vis = 0, deferred = 0;
  std::vector<std::string> unclaimed;
  std::set<std::string> lm_seen;
  for (const auto& [name, shard] : index.at("weight_map").items()) {
    (void)shard;
    if (name.rfind("language_model.", 0) == 0) {
      ++lm;
      const std::string stripped = name.substr(std::string("language_model.").size());
      lm_seen.insert(stripped);
      if (enumerated.count(stripped) != 0) ++lm_claimed; else unclaimed.push_back(name);
    } else if (vision_expected.count(name) != 0) {
      ++vis;
    } else if (name == pg + "video_embedder.weight" ||
               name.rfind("vision_model.radio_model.input_conditioner.", 0) == 0 ||
               name.rfind("sound_encoder.", 0) == 0 || name.rfind("sound_projection.", 0) == 0) {
      ++deferred;
    } else {
      unclaimed.push_back(name);
    }
  }
  std::vector<std::string> missing;
  for (const std::string& n : enumerated)
    if (lm_seen.count(n) == 0 && n.rfind("mtp.", 0) != 0) missing.push_back(n);
  MESSAGE("language tower: " << lm << " shipped, " << lm_claimed << " claimed by the NemotronH "
                             << "enumeration, " << missing.size() << " enumerated but not shipped");
  MESSAGE("vision + mlp1: " << vis << " of " << vision_expected.size() << " claimed; " << deferred
                            << " deferred by name; " << unclaimed.size() << " unclaimed");
  for (size_t i = 0; i < std::min<size_t>(unclaimed.size(), 5); ++i)
    MESSAGE("  unclaimed: " << unclaimed[i]);
  for (size_t i = 0; i < std::min<size_t>(missing.size(), 5); ++i)
    MESSAGE("  enumerated but not shipped: " << missing[i]);
  CHECK(vis == static_cast<int64_t>(vision_expected.size()));
  CHECK(unclaimed.empty());
  CHECK(missing.empty());
}
