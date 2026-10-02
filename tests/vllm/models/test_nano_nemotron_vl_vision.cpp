// Nemotron Nano VL / Omni image path: per-stage numeric gate.
//
// Compares every stage of the processor (nano_nemotron_vl_processor.cpp), the
// RADIO tower and the mlp1 projector (radio.cpp) against
// `nano_nemotron_vl_goldens.inc`, which `scripts/mm/nano_nemotron_vl_ref.py`
// derives in torch from a transcription of vLLM `e126687a9a`
// (radio.py, intern_vit.py, nano_nemotron_vl.py and
// processors/nano_nemotron_vl.py; the resize and the positional interpolation
// are torch's own F.interpolate).
//
// The synthetic weights are an explicit LCG rounded to bf16, reproduced
// bit-exactly here, so no weight blob is committed. The f32 arm pins the
// arithmetic; the bf16 arm is the production dtype and is gated on the bf16
// envelope.
//
// The REAL arm runs only when NANO_NEMOTRON_VL_REAL_WEIGHTS names a
// safetensors file carrying the released `vision_model.*` and `mlp1.*`
// tensors and NANO_NEMOTRON_VL_REAL_GOLDEN names the directory the script's
// `--real-weights` mode wrote. CI never needs the asset.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

#include "doctest/doctest.h"
#include "vllm/model_executor/models/nano_nemotron_vl.h"
#include "vllm/model_executor/models/radio.h"
#include "vllm/multimodal/nano_nemotron_vl_processor.h"
#include "vllm/multimodal/processing/processor.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/backend.h"
#include "vt/dtype.h"

#include "nano_nemotron_vl_goldens.inc"

namespace {

using vllm::multimodal::NanoNemotronVLProcessorConfig;
using vllm::multimodal::NanoNemotronVLProjector;
using vllm::multimodal::NanoNemotronVLProjectorConfig;
using vllm::multimodal::NanoNemotronVLProjectorWeights;
using vllm::multimodal::NanoNemotronVLTileParams;
using vllm::multimodal::RadioImage;
using vllm::multimodal::RadioVisionConfig;
using vllm::multimodal::RadioVisionTower;
using vllm::multimodal::RadioVisionWeights;

// ─── the generator's LCG (scripts/mm/nano_nemotron_vl_ref.py:lcg) ───────────
std::vector<float> Lcg(uint32_t seed, size_t n, double scale) {
  std::vector<float> out(n);
  uint32_t s = seed;
  for (size_t i = 0; i < n; ++i) {
    s = s * 1664525u + 1013904223u;
    const double u = static_cast<double>(s >> 8) / 16777216.0;
    const float f = static_cast<float>((u * 2.0 - 1.0) * scale);
    out[i] = vt::BF16ToF32(vt::F32ToBF16(f));
  }
  return out;
}

std::vector<uint8_t> LcgImage(uint32_t seed, int64_t h, int64_t w) {
  std::vector<uint8_t> out(static_cast<size_t>(h * w * 3));
  uint32_t s = seed;
  for (auto& v : out) {
    s = s * 1664525u + 1013904223u;
    v = static_cast<uint8_t>((s >> 24) & 0xFF);
  }
  return out;
}

std::vector<uint16_t> Bits(const std::vector<float>& f) {
  std::vector<uint16_t> o(f.size());
  for (size_t i = 0; i < f.size(); ++i) o[i] = vt::F32ToBF16(f[i]);
  return o;
}

std::vector<float> OnePlus(const std::vector<float>& f) {
  std::vector<float> o(f.size());
  for (size_t i = 0; i < f.size(); ++i) o[i] = vt::BF16ToF32(vt::F32ToBF16(1.0f + f[i]));
  return o;
}

// SYN in the generator.
constexpr int64_t kH = 64, kHeads = 4, kLayers = 2, kInter = 128, kP = 4;
constexpr int64_t kProjHidden = 96, kLlmHidden = 48;

RadioVisionConfig SynRadio(vt::DType dt) {
  RadioVisionConfig c;
  c.hidden_size = kH;
  c.num_attention_heads = kHeads;
  c.num_hidden_layers = kLayers;
  c.intermediate_size = kInter;
  c.patch_size = kP;
  c.num_cls_tokens = 4;
  c.num_registers = 6;
  c.pos_rows = c.pos_cols = 8;
  c.cpe_mode = true;
  c.layer_norm_eps = 1e-6f;
  c.compute_dtype = dt;
  return c;
}

struct SynWeights {
  RadioVisionWeights radio;
  NanoNemotronVLProjectorWeights proj;
};

SynWeights MakeSyn() {
  uint32_t seed = 1000;
  auto nxt = [&](size_t n, double scale) { return Lcg(++seed, n, scale); };
  SynWeights w;
  w.radio.embedder_w = Bits(nxt(kH * 3 * kP * kP, 0.2));
  w.radio.pos_embed = Bits(nxt(8 * 8 * kH, 0.5));
  w.radio.cls_token = Bits(nxt(10 * kH, 0.5));
  for (int64_t l = 0; l < kLayers; ++l) {
    vllm::multimodal::RadioBlockWeights b;
    b.norm1_w = Bits(OnePlus(nxt(kH, 0.1)));
    b.norm1_b = Bits(nxt(kH, 0.1));
    b.qkv_w = Bits(nxt(3 * kH * kH, 0.15));
    b.qkv_b = Bits(nxt(3 * kH, 0.1));
    b.proj_w = Bits(nxt(kH * kH, 0.15));
    b.proj_b = Bits(nxt(kH, 0.1));
    b.norm2_w = Bits(OnePlus(nxt(kH, 0.1)));
    b.norm2_b = Bits(nxt(kH, 0.1));
    b.fc1_w = Bits(nxt(kInter * kH, 0.15));
    b.fc1_b = Bits(nxt(kInter, 0.1));
    b.fc2_w = Bits(nxt(kH * kInter, 0.1));
    b.fc2_b = Bits(nxt(kH, 0.1));
    w.radio.blocks.push_back(std::move(b));
  }
  w.proj.norm_w = Bits(OnePlus(nxt(4 * kH, 0.1)));
  w.proj.fc1_w = Bits(nxt(kProjHidden * 4 * kH, 0.1));
  w.proj.fc2_w = Bits(nxt(kLlmHidden * kProjHidden, 0.1));
  return w;
}

NanoNemotronVLProcessorConfig SynProc() {
  NanoNemotronVLProcessorConfig c;
  c.patch_size = kP;
  c.min_num_patches = 16;
  c.max_num_patches = 64;
  c.max_model_len = 4096;
  return c;
}

NanoNemotronVLProjectorConfig SynProj(vt::DType dt) {
  NanoNemotronVLProjectorConfig c;
  c.in_dim = 4 * kH;
  c.hidden_dim = kProjHidden;
  c.out_dim = kLlmHidden;
  c.compute_dtype = dt;
  return c;
}

// max |a - b| / max(max |b|, floor)
double RelMaxErr(const std::vector<float>& a, const float* b, size_t n) {
  REQUIRE(a.size() == n);
  double num = 0.0, den = 1e-6;
  for (size_t i = 0; i < n; ++i) {
    num = std::max(num, std::abs(static_cast<double>(a[i]) - b[i]));
    den = std::max(den, std::abs(static_cast<double>(b[i])));
  }
  return num / den;
}

double MaxAbsErr(const std::vector<float>& a, const float* b, size_t n) {
  REQUIRE(a.size() == n);
  double m = 0.0;
  for (size_t i = 0; i < n; ++i) m = std::max(m, std::abs(static_cast<double>(a[i]) - b[i]));
  return m;
}

vt::Backend& Cpu() {
  vt::Backend* cpu = vt::TryGetBackend(vt::DeviceType::kCPU);
  REQUIRE(cpu != nullptr);
  return *cpu;
}

// The processed pixels of the two synthetic images and their grids.
struct SynImages {
  std::vector<NanoNemotronVLTileParams> params;
  std::vector<std::vector<uint8_t>> rgb;
};

SynImages ProcessSyn() {
  SynImages s;
  const NanoNemotronVLProcessorConfig cfg = SynProc();
  std::vector<std::pair<int64_t, int64_t>> sizes;
  for (int i = 0; i < 2; ++i) {
    const int64_t h = kNnvlImgHW[2 * i], w = kNnvlImgHW[2 * i + 1];
    s.rgb.push_back(LcgImage(i == 0 ? 7u : 9u, h, w));
    sizes.emplace_back(w, h);
  }
  s.params = vllm::multimodal::NanoNemotronVLComputeParams(
      sizes, vllm::multimodal::NanoNemotronVLTokensAvailable(cfg, kNnvlTextLen), cfg);
  return s;
}

std::vector<RadioImage> TowerInputs(const SynImages& s) {
  std::vector<RadioImage> images;
  const float* px[2] = {kNnvlPixBf16_0, kNnvlPixBf16_1};
  for (int i = 0; i < 2; ++i) {
    const auto& p = s.params[static_cast<size_t>(i)];
    const int64_t h = p.grid_h * kP, w = p.grid_w * kP;
    std::vector<float> chw(px[i], px[i] + 3 * h * w);
    RadioImage im;
    im.patches = vllm::multimodal::NanoNemotronVLPatchify(chw, h, w, kP);
    im.grid_h = p.grid_h;
    im.grid_w = p.grid_w;
    images.push_back(std::move(im));
  }
  return images;
}

std::vector<float> ShuffleAll(const std::vector<float>& feats, const SynImages& s) {
  std::vector<float> out;
  size_t off = 0;
  for (const auto& p : s.params) {
    const size_t n = static_cast<size_t>(p.grid_h * p.grid_w * kH);
    std::vector<float> one(feats.begin() + static_cast<std::ptrdiff_t>(off),
                           feats.begin() + static_cast<std::ptrdiff_t>(off + n));
    const auto sh = vllm::multimodal::NanoNemotronVLPixelShuffle(one, p.grid_h, p.grid_w, kH, 2);
    out.insert(out.end(), sh.begin(), sh.end());
    off += n;
  }
  return out;
}

}  // namespace

TEST_CASE("nano-nemotron-vl processor: tiling params match the tiler") {
  const SynImages s = ProcessSyn();
  REQUIRE(s.params.size() == 2);
  for (size_t i = 0; i < 2; ++i) {
    CHECK(s.params[i].grid_w == kNnvlParams[3 * i + 0]);
    CHECK(s.params[i].grid_h == kNnvlParams[3 * i + 1]);
    CHECK(s.params[i].num_embeddings == kNnvlParams[3 * i + 2]);
  }
}

TEST_CASE("nano-nemotron-vl processor: python round is half-to-even") {
  // 32 / 16 + 0.5 = 2.5 rounds to 2 (half to even), not 3. A 32x32 image under
  // a generous budget keeps a 2x2 grid; half-away-from-zero would give 3x3 and
  // the even rounding would then grow it to 4x4.
  NanoNemotronVLProcessorConfig cfg;
  cfg.patch_size = 16;
  cfg.min_num_patches = 1;
  cfg.max_num_patches = 1000;
  auto [p, n] = vllm::multimodal::NanoNemotronVLProcessMedia(32, 32, 1000, cfg);
  CHECK(p.grid_w == 2);
  CHECK(p.grid_h == 2);
  CHECK(n == 4);
}

TEST_CASE("nano-nemotron-vl processor: antialiased resize + normalize match torch") {
  const SynImages s = ProcessSyn();
  const NanoNemotronVLProcessorConfig cfg = SynProc();
  const float* f32[2] = {kNnvlPixF32_0, kNnvlPixF32_1};
  const float* b16[2] = {kNnvlPixBf16_0, kNnvlPixBf16_1};
  for (int i = 0; i < 2; ++i) {
    const int64_t h = kNnvlImgHW[2 * i], w = kNnvlImgHW[2 * i + 1];
    const auto& p = s.params[static_cast<size_t>(i)];
    const size_t n = static_cast<size_t>(3 * p.grid_h * kP * p.grid_w * kP);
    const auto a = vllm::multimodal::NanoNemotronVLResizeNormalize(
        s.rgb[static_cast<size_t>(i)].data(), h, w, p, cfg, false);
    CAPTURE(i);
    // f32 end to end: only summation-order noise separates us from torch.
    CHECK(MaxAbsErr(a, f32[i], n) < 2e-5);
    const auto b = vllm::multimodal::NanoNemotronVLResizeNormalize(
        s.rgb[static_cast<size_t>(i)].data(), h, w, p, cfg, true);
    // bf16: at most one ulp on a value that sits on a rounding boundary.
    size_t mismatched = 0;
    for (size_t k = 0; k < n; ++k)
      if (b[k] != b16[i][k]) {
        ++mismatched;
        CHECK(std::abs(b[k] - b16[i][k]) <= std::abs(b16[i][k]) * (1.0f / 128.0f) + 1e-6f);
      }
    CHECK(mismatched * 100 <= n);
  }
}

TEST_CASE("nano-nemotron-vl RADIO: CPE table is square-interpolated then cropped") {
  const SynWeights w = MakeSyn();
  const RadioVisionConfig cfg = SynRadio(vt::DType::kF32);
  const auto pe = vllm::multimodal::RadioPosEmbedForGrid(w.radio.pos_embed, 5, 11, cfg, false);
  CHECK(MaxAbsErr(pe, kNnvlPosNonSquare, static_cast<size_t>(5 * 11 * kH)) < 1e-6);
}

TEST_CASE("nano-nemotron-vl RADIO tower + projector, f32 arm") {
  const SynWeights w = MakeSyn();
  const SynImages s = ProcessSyn();
  vt::Backend& backend = Cpu();
  const RadioVisionTower tower(w.radio, SynRadio(vt::DType::kF32), backend);
  const auto feats = tower.Forward(TowerInputs(s));
  const size_t nf = static_cast<size_t>((s.params[0].grid_h * s.params[0].grid_w +
                                          s.params[1].grid_h * s.params[1].grid_w) * kH);
  CHECK(RelMaxErr(feats, kNnvlFeatsF32, nf) < 1e-4);

  const auto sh = ShuffleAll(feats, s);
  const NanoNemotronVLProjector proj(w.proj, SynProj(vt::DType::kF32), backend);
  const int64_t rows = s.params[0].num_embeddings + s.params[1].num_embeddings;
  const auto out = proj.Forward(sh, rows);
  CHECK(RelMaxErr(out, kNnvlProjF32, static_cast<size_t>(rows * kLlmHidden)) < 1e-4);
}

TEST_CASE("nano-nemotron-vl RADIO tower + projector, bf16 (production) arm") {
  const SynWeights w = MakeSyn();
  const SynImages s = ProcessSyn();
  vt::Backend& backend = Cpu();
  const RadioVisionTower tower(w.radio, SynRadio(vt::DType::kBF16), backend);
  const auto feats = tower.Forward(TowerInputs(s));
  const size_t nf = static_cast<size_t>((s.params[0].grid_h * s.params[0].grid_w +
                                          s.params[1].grid_h * s.params[1].grid_w) * kH);
  CHECK(RelMaxErr(feats, kNnvlFeatsBf16, nf) < 3e-2);

  const auto sh = ShuffleAll(feats, s);
  const NanoNemotronVLProjector proj(w.proj, SynProj(vt::DType::kBF16), backend);
  const int64_t rows = s.params[0].num_embeddings + s.params[1].num_embeddings;
  const auto out = proj.Forward(sh, rows);
  CHECK(RelMaxErr(out, kNnvlProjBf16, static_cast<size_t>(rows * kLlmHidden)) < 3e-2);
}

TEST_CASE("nano-nemotron-vl pixel shuffle v2 keeps the image orientation") {
  // A 4x6 grid of 1-wide rows valued by their (y, x): output (oy, ox) must hold
  // rows (2oy + ry, 2ox + rx) in (ry, rx) order. v1 would transpose it.
  std::vector<float> f;
  for (int y = 0; y < 4; ++y)
    for (int x = 0; x < 6; ++x) f.push_back(static_cast<float>(10 * y + x));
  const auto o = vllm::multimodal::NanoNemotronVLPixelShuffle(f, 4, 6, 1, 2);
  REQUIRE(o.size() == 24);
  // Every output row, in the v2 order (nano_nemotron_vl.py:1028). The
  // first and last rows are the same under v1's `permute(0, 3, 1, 2, 4, 5)`
  // (:1026), so the off-diagonal rows are what tell the two apart: v1 puts
  // inputs (2,0),(2,1),(3,0),(3,1) at output row 1, where v2 puts
  // (0,2),(0,3),(1,2),(1,3).
  const std::vector<float> want = {0,  1,  10, 11,   // (oy=0, ox=0)
                                   2,  3,  12, 13,   // (oy=0, ox=1)
                                   4,  5,  14, 15,   // (oy=0, ox=2)
                                   20, 21, 30, 31,   // (oy=1, ox=0)
                                   22, 23, 32, 33,   // (oy=1, ox=1)
                                   24, 25, 34, 35};  // (oy=1, ox=2)
  CHECK(o == want);
  CHECK(o[1 * 4 + 0] == 2.0f);
  CHECK(o[1 * 4 + 2] == 12.0f);
}

TEST_CASE("nano-nemotron-vl prompt expansion: <image> -> <img> <image>*N </img>") {
  // ids: <image>=18, <img>=19, </img>=20 (the released tokenizer).
  const auto rep = vllm::multimodal::NanoNemotronVLImageReplacement(19, 18, 20, {3, 2});
  std::vector<vllm::multimodal::AppliedPromptUpdate> applied;
  const auto out =
      vllm::multimodal::ApplyPromptReplacements({5, 18, 7, 18, 9}, {rep}, &applied);
  const std::vector<int32_t> want = {5, 19, 18, 18, 18, 20, 7, 19, 18, 18, 20, 9};
  CHECK(out == want);
  REQUIRE(applied.size() == 2);
  CHECK(applied[0].offset == 2);
  CHECK(applied[0].length == 3);
  CHECK(applied[1].offset == 8);
  CHECK(applied[1].length == 2);
}

TEST_CASE("nano-nemotron-vl chat content mirrors the released template") {
  using vllm::multimodal::NanoNemotronVLImageMmContent;
  CHECK(NanoNemotronVLImageMmContent(1, "\nWhat is this?") == "<image>\nWhat is this?");
  CHECK(NanoNemotronVLImageMmContent(2, "Compare") ==
        "<image 1><image> <image 2><image>\nCompare");
  CHECK(NanoNemotronVLImageMmContent(1, "see <image> here") == "see <image> here");
}

TEST_CASE("nano-nemotron-vl config: the released Omni config parses") {
  const vllm::HfConfig cfg = vllm::LoadHfConfig(std::string(NANO_NEMOTRON_VL_FIXTURE_DIR) +
                                                "/nemotron_nano_omni_v3/config.json");
  const vllm::NanoNemotronVLParams p = vllm::ParseNanoNemotronVLParams(cfg);
  CHECK(p.radio.hidden_size == 1280);
  CHECK(p.radio.num_hidden_layers == 32);
  CHECK(p.radio.num_attention_heads == 16);
  CHECK(p.radio.num_cls_tokens == 4);
  CHECK(p.radio.num_registers == 6);
  CHECK(p.radio.pos_rows == 128);
  CHECK(p.radio.cpe_mode);
  CHECK(p.projector.in_dim == 5120);
  CHECK(p.projector.hidden_dim == 20480);
  CHECK(p.projector.out_dim == 2688);
  CHECK(p.processor.min_num_patches == 1024);
  CHECK(p.processor.max_num_patches == 13312);
  CHECK(p.has_sound);
  CHECK(p.has_video_embedder);
  const vllm::HfConfig text = vllm::NanoNemotronVLTextConfig(cfg);
  CHECK(text.hidden_size == 2688);
}

TEST_CASE("nano-nemotron-vl config: the static-tiling 12B VL V2 is refused by name") {
  // The released file carries `time_step_limit: [0.0, Infinity]`, a Python
  // JSON literal the strict parser rejects before any architecture code runs
  // (ISSUE-LOCAL-01M3RZ6SFVDJRX1Z5GZG1CKJ7G, owed). Replace it so the test
  // reaches the refusal this row owns.
  std::ifstream f(std::string(NANO_NEMOTRON_VL_FIXTURE_DIR) + "/nemotron_nano_vl_v2_12b/config.json");
  REQUIRE(f.good());
  std::string text((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  const size_t at = text.find("Infinity");
  REQUIRE(at != std::string::npos);
  text.replace(at, 8, "1e30");
  const vllm::HfConfig cfg = vllm::ParseHfConfig(nlohmann::json::parse(text), "12b-vl-v2");
  bool refused = false;
  try {
    (void)vllm::ParseNanoNemotronVLParams(cfg);
  } catch (const std::runtime_error& e) {
    refused = std::string(e.what()).find("static InternVL tiling") != std::string::npos;
  }
  CHECK(refused);
}

// ─── the REAL arm (env-gated) ───────────────────────────────────────────────
namespace {
std::vector<float> ReadF32(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE(f.good());
  std::vector<char> raw((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  std::vector<float> out(raw.size() / sizeof(float));
  std::memcpy(out.data(), raw.data(), out.size() * sizeof(float));
  return out;
}
}  // namespace

TEST_CASE("nano-nemotron-vl REAL checkpoint: processor, tower and projector") {
  const char* wpath = std::getenv("NANO_NEMOTRON_VL_REAL_WEIGHTS");
  const char* gdir = std::getenv("NANO_NEMOTRON_VL_REAL_GOLDEN");
  if (wpath == nullptr || gdir == nullptr) {
    MESSAGE("SKIP: NANO_NEMOTRON_VL_REAL_WEIGHTS / NANO_NEMOTRON_VL_REAL_GOLDEN unset");
    return;
  }
  const std::string g(gdir);
  nlohmann::json meta;
  {
    std::ifstream f(g + "/meta.json");
    REQUIRE(f.good());
    f >> meta;
  }
  const vllm::HfConfig cfg = vllm::LoadHfConfig(std::string(NANO_NEMOTRON_VL_FIXTURE_DIR) +
                                                "/nemotron_nano_omni_v3/config.json");
  vllm::NanoNemotronVLParams p = vllm::ParseNanoNemotronVLParams(cfg);
  p.processor.max_model_len = meta["max_model_len"].get<int64_t>();
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(wpath));
  const vllm::NanoNemotronVLVisionLoad load = vllm::LoadNanoNemotronVLVisionWeights(shards, p);
  CHECK(load.materialized == 3 + 32 * 12 + 3);
  CHECK(load.materialized + static_cast<int64_t>(load.deferred.size()) == load.shipped);

  const int64_t ih = meta["img_h"].get<int64_t>(), iw = meta["img_w"].get<int64_t>();
  std::vector<uint8_t> rgb(static_cast<size_t>(ih * iw * 3));
  {
    std::ifstream f(g + "/image_u8.bin", std::ios::binary);
    f.read(reinterpret_cast<char*>(rgb.data()), static_cast<std::streamsize>(rgb.size()));
    REQUIRE(f.good());
  }
  const auto params = vllm::multimodal::NanoNemotronVLComputeParams(
      {{iw, ih}},
      vllm::multimodal::NanoNemotronVLTokensAvailable(p.processor,
                                                      meta["text_len"].get<int64_t>()),
      p.processor);
  REQUIRE(params.size() == 1);
  CHECK(params[0].grid_w == meta["grid_w"].get<int64_t>());
  CHECK(params[0].grid_h == meta["grid_h"].get<int64_t>());
  CHECK(params[0].num_embeddings == meta["num_embeddings"].get<int64_t>());

  const auto pix = vllm::multimodal::NanoNemotronVLResizeNormalize(rgb.data(), ih, iw,
                                                                   params[0], p.processor, true);
  const auto pix_ref = ReadF32(g + "/pixels_f32.bin");
  size_t mism = 0;
  for (size_t k = 0; k < pix.size(); ++k) mism += pix[k] != pix_ref[k];
  MESSAGE("real pixels: " << mism << " of " << pix.size() << " differ by a bf16 ulp");
  CHECK(mism * 100 <= pix.size());

  vt::Backend& backend = Cpu();
  const RadioVisionTower tower(load.radio, p.radio, backend);
  RadioImage im;
  const int64_t H = params[0].grid_h * 16, W = params[0].grid_w * 16;
  // Feed the REFERENCE pixels so the tower is gated on its own error alone.
  im.patches = vllm::multimodal::NanoNemotronVLPatchify(pix_ref, H, W, 16);
  im.grid_h = params[0].grid_h;
  im.grid_w = params[0].grid_w;
  const auto feats = tower.Forward({im});
  const auto feats_ref = ReadF32(g + "/feats_f32.bin");
  const double fe = RelMaxErr(feats, feats_ref.data(), feats_ref.size());
  MESSAGE("real tower rel max err (bf16): " << fe);
  CHECK(fe < 8e-2);  // the bf16 envelope of 32 blocks; the f32 arm below is the tight gate

  // The f32 arm on the same bf16 weights and pixels: an arithmetic defect
  // shows here at full size, where the bf16 arm's envelope would hide it.
  {
    vllm::multimodal::RadioVisionConfig c32 = p.radio;
    c32.compute_dtype = vt::DType::kF32;
    const RadioVisionTower t32(load.radio, c32, backend);
    const auto f32 = t32.Forward({im});
    const auto f32_ref = ReadF32(g + "/feats_f32arm.bin");
    const double e32 = RelMaxErr(f32, f32_ref.data(), f32_ref.size());
    MESSAGE("real tower rel max err (f32 arm): " << e32);
    CHECK(e32 < 1e-4);
    vllm::multimodal::NanoNemotronVLProjectorConfig pc32 = p.projector;
    pc32.compute_dtype = vt::DType::kF32;
    const NanoNemotronVLProjector p32(load.projector, pc32, backend);
    const auto o32 = p32.Forward(
        vllm::multimodal::NanoNemotronVLPixelShuffle(f32_ref, params[0].grid_h,
                                                     params[0].grid_w, 1280, 2),
        params[0].num_embeddings);
    const auto o32_ref = ReadF32(g + "/proj_f32arm.bin");
    const double pe32 = RelMaxErr(o32, o32_ref.data(), o32_ref.size());
    MESSAGE("real projector rel max err (f32 arm): " << pe32);
    CHECK(pe32 < 2e-5);
  }

  // The projector on the REFERENCE features, so its own error is isolated too.
  const auto sh = vllm::multimodal::NanoNemotronVLPixelShuffle(feats_ref, params[0].grid_h,
                                                               params[0].grid_w, 1280, 2);
  const NanoNemotronVLProjector proj(load.projector, p.projector, backend);
  const auto out = proj.Forward(sh, params[0].num_embeddings);
  const auto proj_ref = ReadF32(g + "/proj_f32.bin");
  const double pe = RelMaxErr(out, proj_ref.data(), proj_ref.size());
  MESSAGE("real projector rel max err (bf16): " << pe);
  CHECK(pe < 5e-2);

  // Cosine similarity per row of the end-to-end chain (our pixels, our tower,
  // our projector) against the reference chain.
  RadioImage ours;
  ours.patches = vllm::multimodal::NanoNemotronVLPatchify(pix, H, W, 16);
  ours.grid_h = params[0].grid_h;
  ours.grid_w = params[0].grid_w;
  const auto chain = proj.Forward(
      vllm::multimodal::NanoNemotronVLPixelShuffle(tower.Forward({ours}), params[0].grid_h,
                                                   params[0].grid_w, 1280, 2),
      params[0].num_embeddings);
  double worst = 1.0;
  for (int64_t r = 0; r < params[0].num_embeddings; ++r) {
    double dot = 0, na = 0, nb = 0;
    for (int64_t c = 0; c < 2688; ++c) {
      const double a = chain[static_cast<size_t>(r * 2688 + c)];
      const double b = proj_ref[static_cast<size_t>(r * 2688 + c)];
      dot += a * b;
      na += a * a;
      nb += b * b;
    }
    worst = std::min(worst, dot / std::sqrt(na * nb + 1e-30));
  }
  MESSAGE("real end-to-end worst row cosine: " << worst);
  CHECK(worst > 0.99);
}
