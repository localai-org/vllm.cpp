// Muse Glimmer perception encoder: the FIRST reference run on the released
// tensors (row MODEL-MM-muse-glimmer-muse-glimmer-for-conditional-generation,
// .agents/specs/muse-glimmer-parity.md).
//
// Env-gated, because the tensors are ~3.7 GB and are not committed:
//   MUSE_GLIMMER_VISION_REAL_DIR     a directory holding the released
//                                    `config.json` and safetensors files that
//                                    carry every `model.vision_*` tensor of
//                                    meta-models/Muse-Glimmer-30B @ a4e59da5
//   MUSE_GLIMMER_VISION_REAL_GOLDEN  the directory
//                                    `scripts/mm/muse_glimmer_vision_ref.py
//                                    --real-dir ... --out-dir ...` wrote
//
// The weights reach the tower through `LoadMuseGlimmerVisionTower`, the slot
// loader the full-model load uses, and the soft tokens come out of
// `MuseGlimmerEncodePixelGroups`, the function the registered multimodal
// forward calls: encoder -> adapter -> vision_projection -> perception_emb_norm.
// The reference is a torch transcription of the pinned vLLM `a7c23ac96d`
// formulas. It is NOT the vLLM runtime, and no image-to-text claim follows.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/muse_glimmer.h"
#include "vllm/model_executor/models/muse_glimmer_vision.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

namespace {

std::vector<float> ReadF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE(f.good());
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<float> out(raw.size() / sizeof(float));
  std::memcpy(out.data(), raw.data(), out.size() * sizeof(float));
  return out;
}

// max |a - b| / max |b|
double RelMaxErr(const std::vector<float>& a, const std::vector<float>& b) {
  REQUIRE(a.size() == b.size());
  double num = 0.0, den = 1e-12;
  for (size_t i = 0; i < a.size(); ++i) {
    num = std::max(num, std::abs(static_cast<double>(a[i]) - b[i]));
    den = std::max(den, std::abs(static_cast<double>(b[i])));
  }
  return num / den;
}

struct RowCosine {
  double worst = 1.0;
  double mean = 0.0;
};

RowCosine RowCosines(const std::vector<float>& a, const std::vector<float>& b, size_t width) {
  REQUIRE(a.size() == b.size());
  REQUIRE(width > 0);
  RowCosine out;
  const size_t rows = a.size() / width;
  for (size_t r = 0; r < rows; ++r) {
    double dot = 0, na = 0, nb = 0;
    for (size_t c = 0; c < width; ++c) {
      const double x = a[r * width + c], y = b[r * width + c];
      dot += x * y;
      na += x * x;
      nb += y * y;
    }
    const double cs = dot / std::sqrt(na * nb + 1e-300);
    out.worst = std::min(out.worst, cs);
    out.mean += cs / static_cast<double>(rows);
  }
  return out;
}

}  // namespace

TEST_CASE("muse glimmer vision REAL: tower, adapter, projection vs the pinned formulas") {
  const char* dir = std::getenv("MUSE_GLIMMER_VISION_REAL_DIR");
  const char* gold = std::getenv("MUSE_GLIMMER_VISION_REAL_GOLDEN");
  if (dir == nullptr || gold == nullptr) {
    MESSAGE("SKIP: MUSE_GLIMMER_VISION_REAL_DIR / MUSE_GLIMMER_VISION_REAL_GOLDEN unset");
    return;
  }
  const std::string d(dir), g(gold);
  nlohmann::json meta;
  {
    std::ifstream f(g + "/meta.json");
    REQUIRE(f.good());
    f >> meta;
  }
  const vllm::HfConfig cfg = vllm::LoadHfConfig(d + "/config.json");
  vllm::MuseGlimmerWeights w;
  w.params = vllm::ParseMuseGlimmerParams(cfg);
  CHECK(w.params.text.normalize_tok_embeddings == meta["normalize_tok_embeddings"].get<bool>());
  std::vector<vllm::SafetensorsFile> shards;
  for (const auto& e : std::filesystem::directory_iterator(d)) {
    if (e.path().extension() == ".safetensors") shards.push_back(vllm::SafetensorsFile::Open(e.path().string()));
  }
  REQUIRE_FALSE(shards.empty());
  w.vision = vllm::LoadMuseGlimmerVisionTower(shards, w.params);
  REQUIRE(w.vision.loaded);

  const int64_t h = meta["height"].get<int64_t>(), wd = meta["width"].get<int64_t>();
  vllm::multimodal::MuseGlimmerVisionImage image;
  image.pixels = ReadF32(g + "/pixels_f32.bin");
  image.channels = 3;
  image.height = h;
  image.width = wd;
  REQUIRE(static_cast<int64_t>(image.pixels.size()) == 3 * h * wd);
  const int64_t H = w.params.text.hidden_size;
  vt::Backend* cpu = vt::TryGetBackend(vt::DeviceType::kCPU);
  REQUIRE(cpu != nullptr);
  vt::Queue q{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};

  for (const char* arm : {"f32", "bf16"}) {
    CAPTURE(std::string(arm));
    const bool f32 = std::string(arm) == "f32";
    w.vision.cfg.compute_dtype = f32 ? vt::DType::kF32 : vt::DType::kBF16;

    // Stage by stage, through the tower's own capture.
    vllm::multimodal::MuseGlimmerVisionCapture cap;
    const std::vector<float> tower = vllm::multimodal::MuseGlimmerVisionForward(
        {image}, w.vision.encoder, w.vision.cfg, *cpu, &cap);
    const std::vector<float> ln_pre_ref = ReadF32(g + "/ln_pre_" + arm + ".bin");
    REQUIRE(cap.ln_pre_out.size() == 1);
    const double e_ln = RelMaxErr(cap.ln_pre_out[0], ln_pre_ref);
    const double e_block0 = RelMaxErr(cap.block0_out, ReadF32(g + "/block0_" + arm + ".bin"));
    const double e_tower = RelMaxErr(tower, ReadF32(g + "/tower_" + arm + ".bin"));
    // The adapter on OUR tower output, so its error isolates the adapter only
    // when the tower already agrees (the f32 arm).
    const std::vector<float> adapted = vllm::multimodal::MuseGlimmerVisionAdapterForward(
        tower, static_cast<int64_t>(tower.size()) / w.vision.cfg.output_dim, w.vision.adapter,
        w.vision.cfg, *cpu);
    const double e_adapter = RelMaxErr(adapted, ReadF32(g + "/adapter_" + arm + ".bin"));
    // The production entry point: encoder -> adapter -> projection -> norm.
    const std::vector<float> soft = vllm::MuseGlimmerEncodePixelGroups({image}, w, q);
    const std::vector<float> soft_ref = ReadF32(g + "/soft_" + arm + ".bin");
    const double e_soft = RelMaxErr(soft, soft_ref);
    const RowCosine cos_soft = RowCosines(soft, soft_ref, static_cast<size_t>(H));
    MESSAGE(std::string(arm) << ": ln_pre rel " << e_ln << ", block0 rel " << e_block0
                             << ", tower rel " << e_tower << ", adapter rel " << e_adapter
                             << ", soft tokens rel " << e_soft << ", soft-token cosine worst "
                             << cos_soft.worst << " mean " << cos_soft.mean << " over "
                             << soft.size() / static_cast<size_t>(H) << " rows");
    if (f32) {
      // The arithmetic gate: same bf16 weights, f32 math on both sides.
      CHECK(e_ln < 1e-4);
      CHECK(e_block0 < 1e-4);
      CHECK(e_tower < 1e-3);
      CHECK(e_adapter < 1e-3);
      CHECK(e_soft < 1e-3);
    } else {
      // 50 bf16 blocks move a few rows a long way from the f32 result on BOTH
      // sides: the reference's own bf16 arm reaches a worst soft-token cosine
      // of 0.970 against its f32 arm. So the production arm is gated against
      // the f32 TRUTH, and must be at least as close to it as the reference's
      // bf16 transcription is (a small allowance for which rows drift).
      const std::vector<float> truth = ReadF32(g + "/soft_f32.bin");
      const RowCosine ours = RowCosines(soft, truth, static_cast<size_t>(H));
      const RowCosine theirs = RowCosines(soft_ref, truth, static_cast<size_t>(H));
      MESSAGE("bf16 vs the f32 truth: ours worst " << ours.worst << " mean " << ours.mean
                                                   << "; reference bf16 worst " << theirs.worst
                                                   << " mean " << theirs.mean);
      CHECK(ours.mean >= theirs.mean - 1e-3);
      CHECK(ours.worst >= theirs.worst - 0.02);
      // Cosine is blind to scale (a dropped output norm keeps it high), so the
      // magnitude is bounded too: our relative max error against the f32 truth
      // may not exceed twice the reference bf16 arm's own.
      const double e_ours = RelMaxErr(soft, truth);
      const double e_theirs = RelMaxErr(soft_ref, truth);
      MESSAGE("bf16 rel max err vs the f32 truth: ours " << e_ours << ", reference bf16 "
                                                          << e_theirs);
      CHECK(e_ours <= 2.0 * e_theirs);
    }
  }
}
