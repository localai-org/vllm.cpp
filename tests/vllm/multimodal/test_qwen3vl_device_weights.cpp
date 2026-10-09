#include <doctest/doctest.h>

#include <array>
#include <cstdlib>
#include <cstring>
#include <map>
#include <stdexcept>
#include <vector>

#include "vllm/model_executor/models/qwen3_vl_vision.h"
#include "vt/dtype.h"
#include "vt/xpu.h"

namespace {
using namespace vllm::multimodal;
using vt::DType;

// Independent, hand-specified numeric equivalents; these patterns would fail
// if checkpoint BF16 bytes were merely reinterpreted as FP16.
constexpr std::array<uint16_t, 5> kBf{0x3f80, 0x3f81, 0xc000, 0x3f00, 0};
constexpr std::array<uint16_t, 5> kHalf{0x3c00, 0x3c08, 0xc000, 0x3800, 0};

std::vector<uint16_t> Pattern(size_t n, bool half) {
  std::vector<uint16_t> result(n);
  for (size_t i = 0; i < n; ++i) result[i] = half ? kHalf[i % 5] : kBf[i % 5];
  return result;
}

Qwen3VLVisionConfig Config() {
  Qwen3VLVisionConfig c;
  c.hidden_size = 4; c.num_heads = 1; c.depth = 1;
  c.intermediate_size = 6; c.out_hidden_size = 3;
  c.patch_size = 2; c.temporal_patch_size = 1; c.in_channels = 1;
  c.num_position_embeddings = 4; c.deepstack_visual_indexes = {0};
  return c;
}

Qwen3VLVisionWeights Weights() {
  Qwen3VLVisionWeights w;
  w.patch_proj_w = Pattern(16, false); w.patch_proj_b = Pattern(4, false);
  for (uint16_t bits : Pattern(16, false)) w.pos_embed_w.push_back(vt::BF16ToF32(bits));
  VisionBlockWeights b;
  b.norm1_w = b.norm1_b = b.norm2_w = b.norm2_b = Pattern(4, false);
  b.qkv_w = Pattern(48, false); b.qkv_b = Pattern(12, false);
  b.proj_w = Pattern(16, false); b.proj_b = Pattern(4, false);
  b.fc1_w = Pattern(24, false); b.fc1_b = Pattern(6, false);
  b.fc2_w = Pattern(24, false); b.fc2_b = Pattern(4, false);
  w.blocks.push_back(b);
  w.merger.norm_w = w.merger.norm_b = Pattern(4, false);
  w.merger.fc1_w = Pattern(256, false); w.merger.fc1_b = Pattern(16, false);
  w.merger.fc2_w = Pattern(48, false); w.merger.fc2_b = Pattern(3, false);
  auto ds = w.merger;
  ds.use_postshuffle_norm = true;
  ds.norm_w = ds.norm_b = Pattern(16, false);
  w.deepstack_mergers.push_back(ds);
  return w;
}

// Mock copies deliberately read sources only at Synchronize, exercising the
// upload owner's lifetime. With a real backend this observes actual device
// bytes after its queue drains. No learned computation is mocked or qualified.
class UploadRecorder final : public vt::Backend {
 public:
  explicit UploadRecorder(vt::DeviceType type, vt::Backend* real = nullptr)
      : type(type), real(real) {}
  void* Alloc(size_t n) override {
    if (++allocations == fail_allocation) throw std::runtime_error("injected allocation failure");
    void* p = real ? real->Alloc(n) : std::malloc(n == 0 ? 1 : n);
    live[p] = n;
    return p;
  }
  void Free(void* p) override {
    if (!p) return;
    for (const auto& c : pending) if (c.dst == p) ++premature_frees;
    live.erase(p);
    if (real) real->Free(p); else std::free(p);
  }
  void Memset(vt::Queue& q, void* p, int v, size_t n) override {
    if (real) real->Memset(q, p, v, n); else std::memset(p, v, n);
  }
  void Copy(vt::Queue& q, void* dst, const void* src, size_t n) override {
    if (real) real->Copy(q, dst, src, n);
    pending.push_back({dst, src, n});
  }
  vt::Queue CreateQueue() override {
    ++created;
    return real ? real->CreateQueue() : vt::Queue{{type, 0}, nullptr};
  }
  void DestroyQueue(vt::Queue& q) override {
    ++destroyed;
    if (real) real->DestroyQueue(q);
  }
  void Synchronize(vt::Queue& q) override {
    if (real) real->Synchronize(q);
    for (const auto& c : pending) {
      if (!real) std::memcpy(c.dst, c.src, c.n);
      snapshots.emplace_back(c.n / sizeof(uint16_t));
      if (real) real->Copy(q, snapshots.back().data(), c.dst, c.n);
      else std::memcpy(snapshots.back().data(), c.dst, c.n);
    }
    if (real) real->Synchronize(q);
    pending.clear();
  }
  bool UnifiedMemory() const override { return real ? real->UnifiedMemory() : true; }
  struct Transfer { void* dst; const void* src; size_t n; };
  vt::DeviceType type;
  vt::Backend* real;
  std::map<void*, size_t> live;
  std::vector<Transfer> pending;
  std::vector<std::vector<uint16_t>> snapshots;
  size_t allocations = 0, fail_allocation = 0;
  int created = 0, destroyed = 0, premature_frees = 0;
};

void CheckUploads(const UploadRecorder& b, bool half) {
  std::vector<size_t> counts{16, 4};
  if (half) counts.push_back(16);  // typed device position table
  const std::vector<size_t> block{4,4,4,4,48,12,16,4,24,6,24,4};
  counts.insert(counts.end(), block.begin(), block.end());
  const std::vector<size_t> mergers{4,4,256,16,48,3,16,16,256,16,48,3};
  counts.insert(counts.end(), mergers.begin(), mergers.end());
  REQUIRE(b.snapshots.size() == counts.size());
  for (size_t i = 0; i < counts.size(); ++i) CHECK(b.snapshots[i] == Pattern(counts[i], half));
  CHECK(b.created == 1); CHECK(b.destroyed == 1);
  CHECK(b.pending.empty()); CHECK(b.premature_frees == 0);
}
}  // namespace

TEST_CASE("qwen3vl device weights: default BF16 and explicit FP16 numeric uploads") {
  auto w = Weights();
  const auto c = Config();
  for (auto type : {vt::DeviceType::kCPU, vt::DeviceType::kCUDA, vt::DeviceType::kROCM}) {
    UploadRecorder b(type);
    auto dw = PrepareVisionDeviceWeights(w, c, b);
    CheckUploads(b, false);
    dw.reset(); CHECK(b.live.empty());
  }
  UploadRecorder b(vt::DeviceType::kXPU);
  auto dw = PrepareVisionDeviceWeights(w, c, b, DType::kF16);
  CheckUploads(b, true);
  CHECK_THROWS_WITH_AS(Qwen3VLVisionForward({}, {1,2,2}, *dw, c, b),
      doctest::Contains("legacy BF16 forward cannot consume FP16 weights"), std::runtime_error);
  CHECK(b.created == 1);  // refuse before creating a forward queue
  dw.reset(); CHECK(b.live.empty());
}

TEST_CASE("qwen3vl device weights: failed preparation drains uploads and releases queue") {
  for (bool half : {false, true}) {
    for (size_t fail : {size_t{5}, size_t{17}, size_t{25}}) {
     UploadRecorder b(vt::DeviceType::kXPU);
     b.fail_allocation = fail;
     CHECK_THROWS_AS(PrepareVisionDeviceWeights(Weights(), Config(), b,
                     half ? DType::kF16 : DType::kBF16), std::runtime_error);
     CHECK(b.snapshots.size() == fail - 1);
     for (const auto& snapshot : b.snapshots) CHECK(snapshot == Pattern(snapshot.size(), half));
     CHECK(b.live.empty()); CHECK(b.pending.empty()); CHECK(b.premature_frees == 0);
     CHECK(b.created == 1); CHECK(b.destroyed == 1);
    }
   }
  auto w = Weights();
  for (auto type : {vt::DeviceType::kCPU, vt::DeviceType::kCUDA, vt::DeviceType::kROCM}) {
    UploadRecorder b(type);
    CHECK_THROWS_AS(PrepareVisionDeviceWeights(w, Config(), b, DType::kF16), std::runtime_error);
    CHECK(b.live.empty()); CHECK(b.created == 1); CHECK(b.destroyed == 1);
  }
  UploadRecorder b(vt::DeviceType::kCPU), other(vt::DeviceType::kCPU);
  CHECK_THROWS_AS(PrepareVisionDeviceWeights(w, Config(), b, DType::kF32), std::runtime_error);
  CHECK(b.created == 0);
  auto dw = PrepareVisionDeviceWeights(w, Config(), b);
  CHECK_THROWS_WITH_AS(Qwen3VLVisionForward({}, {1,2,2}, *dw, Config(), other),
      doctest::Contains("prepared weights belong to another backend"), std::runtime_error);
  CHECK(other.created == 0);
  w.patch_proj_w.clear();
  CHECK_THROWS_AS(PrepareVisionDeviceWeights(w, Config(), other), std::runtime_error);
  CHECK(other.live.empty()); CHECK(other.created == 1); CHECK(other.destroyed == 1);
}

TEST_CASE("qwen3vl device weights: actual XPU upload and owner release") {
#if defined(VLLM_CPP_XPU)
  auto* gpu = vt::TryGetBackend(vt::DeviceType::kXPU);
  REQUIRE(gpu != nullptr);
  const auto before = vt::xpu::GetMemoryInfo().allocated_bytes;
  UploadRecorder b(vt::DeviceType::kXPU, gpu);
  const auto w = Weights();
  auto dw = PrepareVisionDeviceWeights(w, Config(), b, DType::kF16);
  CheckUploads(b, true);
  dw.reset(); CHECK(b.live.empty());
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == before);
#else
  MESSAGE("SKIP: actual GPU uploads require an XPU build; mock lifecycle tests run separately");
#endif
}
