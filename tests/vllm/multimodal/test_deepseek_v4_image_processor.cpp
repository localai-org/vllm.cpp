// Ported from deepseek-ai/DeepSeek-V4-Flash-Vision-Exp
// inference/image_processor.py at revision
// 86f746b36186f0e567729a5c06a8c918caba82a9. The BF16 word goldens below were
// produced by executing that pinned file with Pillow 12.1.1 and torch 2.11.0.
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <memory>
#include <new>
#include <span>
#include <string>
#include <vector>

#include "doctest/doctest.h"
#include "vllm/multimodal/deepseek_v4_processor.h"

#include "vllm/multimodal/hasher.h"

namespace allocation_probe {

thread_local bool enabled = false;
thread_local size_t allocations = 0;
thread_local size_t bytes = 0;

}  // namespace allocation_probe

// THE COMPLETE SET OF REPLACEABLE GLOBAL ALLOCATION FUNCTIONS. The probe
// replaces `operator new` with a `std::malloc`-based one so it can count every
// allocation, and pairs it with a `std::free`-based `operator delete`. That
// pairing is only sound if EVERY replaceable allocation form routes through
// the same malloc/free pair: the nothrow and aligned forms included. The
// pre-fix file replaced only the throwing forms, so libstdc++'s
// `std::get_temporary_buffer` (used by `std::stable_sort` inside
// `vllm::OrderedRegistry` at static-init time) allocated through the DEFAULT
// nothrow `operator new` and was freed by the replaced `operator delete` with
// `std::free` — an operator-new/operator-delete mismatch ASan refuses
// (alloc-dealloc-mismatch, ISSUE-LOCAL-01M4FK4BJFZMCSG04WD0CAY5D2).
//
// NOT INLINED, for two reasons that point the same way. A probe the optimizer
// inlines is a probe it can elide, and an uncounted allocation reads as a
// PASS on the very assertion these operators exist to make. And GCC pairs the
// inlined `std::malloc` here with the inlined `std::free` below across an
// unrelated caller and reports `-Wmismatched-new-delete`, which is a false
// positive on a deliberate replacement rather than a defect to silence.
__attribute__((noinline))
static void* ProbeAlloc(std::size_t size, std::size_t alignment) {
  if (size == 0) size = 1;
  // Over-aligned requests need a genuinely aligned block; plain malloc only
  // guarantees __STDCPP_DEFAULT_NEW_ALIGNMENT__. std::aligned_alloc requires
  // the size to be a multiple of the alignment, so round up. Every path here
  // is released by std::free in the matching operator delete below.
  void* pointer =
      alignment > __STDCPP_DEFAULT_NEW_ALIGNMENT__
          ? std::aligned_alloc(alignment, (size + alignment - 1) & ~(alignment - 1))
          : std::malloc(size);
  if (pointer == nullptr) return nullptr;
  if (allocation_probe::enabled) {
    ++allocation_probe::allocations;
    allocation_probe::bytes += size;
  }
  return pointer;
}

__attribute__((noinline))
void* operator new(std::size_t size) {
  if (void* pointer = ProbeAlloc(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__)) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size) {
  return ::operator new(size);
}

void* operator new(std::size_t size, const std::nothrow_t&) noexcept {
  return ProbeAlloc(size, __STDCPP_DEFAULT_NEW_ALIGNMENT__);
}

void* operator new[](std::size_t size, const std::nothrow_t&) noexcept {
  return ::operator new(size, std::nothrow);
}

void* operator new(std::size_t size, std::align_val_t alignment) {
  if (void* pointer = ProbeAlloc(size, static_cast<std::size_t>(alignment))) {
    return pointer;
  }
  throw std::bad_alloc();
}

void* operator new[](std::size_t size, std::align_val_t alignment) {
  return ::operator new(size, alignment);
}

void* operator new(std::size_t size, std::align_val_t alignment,
                   const std::nothrow_t&) noexcept {
  return ProbeAlloc(size, static_cast<std::size_t>(alignment));
}

void* operator new[](std::size_t size, std::align_val_t alignment,
                     const std::nothrow_t&) noexcept {
  return ::operator new(size, alignment, std::nothrow);
}

__attribute__((noinline))
void operator delete(void* pointer) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer) noexcept {
  ::operator delete(pointer);
}

void operator delete(void* pointer, std::size_t) noexcept {
  ::operator delete(pointer);
}

void operator delete[](void* pointer, std::size_t) noexcept {
  ::operator delete[](pointer);
}

void operator delete(void* pointer, const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, const std::nothrow_t&) noexcept {
  ::operator delete(pointer, std::nothrow);
}

void operator delete(void* pointer, std::align_val_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::align_val_t) noexcept {
  ::operator delete[](pointer, std::nothrow);
}

void operator delete(void* pointer, std::align_val_t, std::size_t) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::align_val_t, std::size_t) noexcept {
  ::operator delete[](pointer, std::nothrow);
}

void operator delete(void* pointer, std::align_val_t,
                     const std::nothrow_t&) noexcept {
  std::free(pointer);
}

void operator delete[](void* pointer, std::align_val_t,
                       const std::nothrow_t&) noexcept {
  ::operator delete[](pointer, std::nothrow);
}

namespace {

using vllm::multimodal::BuildDeepSeekV4ImageBlock;
using vllm::multimodal::DeepSeekV4ImageProcessor;
using vllm::multimodal::DeepSeekV4ProcessorConfig;
using vllm::multimodal::GridTokens;
using vllm::multimodal::MakeDeepSeekV4MmHash;
using vllm::multimodal::PrepareDeepSeekV4Inputs;
using vllm::multimodal::SafeResize;
using vllm::multimodal::SolveResizeRatio;

class AllocationProbe {
 public:
  AllocationProbe() {
    allocation_probe::allocations = 0;
    allocation_probe::bytes = 0;
    allocation_probe::enabled = true;
  }

  ~AllocationProbe() { allocation_probe::enabled = false; }

  size_t allocations() const { return allocation_probe::allocations; }
  size_t bytes() const { return allocation_probe::bytes; }
};

DeepSeekV4ProcessorConfig TinyConfig() {
  DeepSeekV4ProcessorConfig cfg;
  cfg.patch_size = 2;
  cfg.downsample_ratio = 1;
  cfg.max_image_tokens = 128;
  cfg.min_pixels = 0;
  cfg.max_width_height_ratio = 8;
  cfg.vocab_size = 100;
  return cfg;
}

std::vector<uint8_t> IdentityRgb() {
  return {0,   10,  20, 64,  74,  84,
          128, 138, 148, 255, 245, 235};
}

DeepSeekV4ProcessorConfig OracleConfig(int64_t min_pixels,
                                       int64_t max_image_tokens,
                                       int64_t max_width_height_ratio) {
  DeepSeekV4ProcessorConfig cfg;
  cfg.patch_size = 1;
  cfg.downsample_ratio = 1;
  cfg.max_image_tokens = max_image_tokens;
  cfg.min_pixels = min_pixels;
  cfg.max_width_height_ratio = max_width_height_ratio;
  cfg.vocab_size = 100;
  return cfg;
}

std::vector<uint8_t> FormulaRgb(int64_t height, int64_t width, uint8_t seed) {
  std::vector<uint8_t> rgb(static_cast<size_t>(height * width * 3));
  for (size_t i = 0; i < rgb.size(); ++i) {
    rgb[i] = static_cast<uint8_t>(
        static_cast<uint64_t>(seed) * 53 + i * 97 + i * i * 17);
  }
  return rgb;
}

uint64_t HashBf16Words(const std::vector<uint16_t>& words) {
  uint64_t hash = 1469598103934665603ULL;
  for (const uint16_t word : words) {
    hash ^= static_cast<uint8_t>(word);
    hash *= 1099511628211ULL;
    hash ^= static_cast<uint8_t>(word >> 8);
    hash *= 1099511628211ULL;
  }
  return hash;
}

}  // namespace

TEST_CASE("deepseek-v4 grid token accounting includes N-layout padding") {
  const auto one = GridTokens(42, 42, 14, 3);
  CHECK(one.n_llm_h == 1);
  CHECK(one.n_llm_w == 1);
  CHECK(one.num_tokens == 6);

  const auto even = GridTokens(84, 126, 14, 3);
  CHECK(even.n_llm_h == 2);
  CHECK(even.n_llm_w == 3);
  CHECK(even.num_tokens == 10);

  const auto odd = GridTokens(126, 84, 14, 3);
  CHECK(odd.n_llm_h == 3);

  CHECK(odd.n_llm_w == 2);
  CHECK(odd.num_tokens == 14);
}

TEST_CASE("deepseek-v4 processor validates its complete configuration once") {
  auto cfg = TinyConfig();
  cfg.patch_size = 0;
  CHECK_THROWS_WITH_AS(
      static_cast<void>(DeepSeekV4ImageProcessor(cfg)),
      "DeepSeek-V4 processor patch size must be positive",
      std::invalid_argument);
  cfg = TinyConfig();
  cfg.downsample_ratio = 0;
  CHECK_THROWS_WITH_AS(
      static_cast<void>(DeepSeekV4ImageProcessor(cfg)),
      "DeepSeek-V4 processor downsample ratio must be positive",
      std::invalid_argument);
  for (const int64_t invalid_budget : {5, 6, 8}) {
    cfg = TinyConfig();
    cfg.max_image_tokens = invalid_budget;
    CHECK_THROWS_WITH_AS(
        static_cast<void>(DeepSeekV4ImageProcessor(cfg)),
        "DeepSeek-V4 processor image token budget must be at least 9",
        std::invalid_argument);
  }
  cfg = OracleConfig(0, 9, 8);
  const std::array<uint8_t, 3> minimum_rgb = {0, 127, 255};
  const auto minimum_image = DeepSeekV4ImageProcessor(cfg).ProcessImage(
      std::span<const uint8_t>(minimum_rgb), 1, 1);
  CHECK(minimum_image.image_grid_thw ==
        std::array<int64_t, 3>{1, 1, 1});
  CHECK(minimum_image.num_patches == 1);
  CHECK(minimum_image.pixel_values_bf16.size() == 3);
  cfg = TinyConfig();
  cfg.min_pixels = -1;
  CHECK_THROWS_WITH_AS(
      static_cast<void>(DeepSeekV4ImageProcessor(cfg)),
      "DeepSeek-V4 processor minimum pixels must not be negative",
      std::invalid_argument);
  cfg = TinyConfig();
  cfg.max_width_height_ratio = -1;
  CHECK_THROWS_WITH_AS(
      static_cast<void>(DeepSeekV4ImageProcessor(cfg)),
      "DeepSeek-V4 processor width-height ratio must not be negative",
      std::invalid_argument);
  cfg = TinyConfig();
  cfg.vocab_size = std::numeric_limits<int32_t>::max();
  CHECK_THROWS_WITH_AS(
      static_cast<void>(DeepSeekV4ImageProcessor(cfg)),
      "DeepSeek-V4 processor vocabulary size is invalid",
      std::invalid_argument);
}

TEST_CASE("deepseek-v4 RGB span must have the exact checked byte extent") {
  const auto rgb = IdentityRgb();
  const DeepSeekV4ImageProcessor processor(TinyConfig());
  CHECK_THROWS_WITH_AS(
      processor.ProcessImage(
          std::span<const uint8_t>(rgb.data(), rgb.size() - 1), 2, 2),
      "DeepSeek-V4 RGB byte extent does not equal height*width*3",
      std::invalid_argument);
  std::vector<uint8_t> oversized = rgb;
  oversized.push_back(0);
  CHECK_THROWS_WITH_AS(
      processor.ProcessImage(std::span<const uint8_t>(oversized), 2, 2),
      "DeepSeek-V4 RGB byte extent does not equal height*width*3",
      std::invalid_argument);
  CHECK_THROWS_WITH_AS(
      processor.ProcessImage(
          std::span<const uint8_t>(rgb),
          std::numeric_limits<int64_t>::max(),
          std::numeric_limits<int64_t>::max()),
      "DeepSeek-V4 RGB byte extent overflow", std::overflow_error);
}

TEST_CASE("deepseek-v4 geometry arithmetic rejects overflow by name") {
  CHECK_THROWS_WITH_AS(
      GridTokens(1, std::numeric_limits<int64_t>::max(), 1, 1),
      "DeepSeek-V4 grid token count overflow", std::overflow_error);
}

TEST_CASE("deepseek-v4 resize ratio preserves all three upstream branches") {
  const auto tall = SolveResizeRatio(20000, 100, 14, 3, 384);
  CHECK(tall.n_llm_h == 190);
  CHECK(tall.n_llm_w == 1);
  CHECK(tall.height == 7980);
  CHECK(tall.width == 42);
  CHECK(tall.num_tokens == 382);

  const auto wide = SolveResizeRatio(100, 100000, 14, 3, 384);
  CHECK(wide.n_llm_h == 2);
  CHECK(wide.n_llm_w == 190);
  CHECK(wide.height == 84);
  CHECK(wide.width == 7980);
  CHECK(wide.num_tokens == 386);

  const auto regular = SolveResizeRatio(300, 400, 14, 3, 384);
  CHECK(regular.n_llm_h == 16);
  CHECK(regular.n_llm_w == 22);
  CHECK(regular.height == 672);
  CHECK(regular.width == 896);
  CHECK(regular.num_tokens == 370);
}

TEST_CASE("deepseek-v4 safe resize reserves compression padding and loops") {
  const auto resized = SafeResize(714, 840, 714, 840, 14, 3, 384);
  CHECK(resized.n_llm_h == 16);
  CHECK(resized.n_llm_w == 19);
  CHECK(resized.height == 672);
  CHECK(resized.width == 784);
  CHECK(GridTokens(resized.height, resized.width, 14, 3).num_tokens <= 381);
}

TEST_CASE("deepseek-v4 RGB normalize and patchify produce exact BF16 bytes") {
  const auto rgb = IdentityRgb();
  const auto image = DeepSeekV4ImageProcessor(TinyConfig()).ProcessImage(
      std::span<const uint8_t>(rgb), 2, 2);
  CHECK(image.image_grid_thw == std::array<int64_t, 3>{1, 1, 1});
  CHECK(image.num_patches == 1);
  CHECK(image.patch_feature_dim == 12);
  const std::vector<uint16_t> golden = {
      49024, 48895, 15233, 16256, 49004, 48855,
      15785, 16236, 48984, 48815, 15909, 16216};
  CHECK(image.pixel_values_bf16 == golden);
  CHECK(image.pixel_values_f32.empty());
}

TEST_CASE("deepseek-v4 identity processing allocates only its BF16 output") {
  const auto rgb = IdentityRgb();
  const DeepSeekV4ImageProcessor processor(TinyConfig());
  vllm::multimodal::ImageKwargs image;
  size_t allocations = 0;
  size_t bytes = 0;
  {
    const AllocationProbe probe;
    image = processor.ProcessImage(std::span<const uint8_t>(rgb), 2, 2);
    allocations = probe.allocations();
    bytes = probe.bytes();
  }
  CHECK(allocations == 1);
  CHECK(bytes == image.pixel_values_bf16.size() * sizeof(uint16_t));
}

TEST_CASE("deepseek-v4 non-wide images keep aspect ratio and pad with RGB 127") {
  const std::vector<uint8_t> rgb = {
      0, 10, 20, 64, 74, 84, 128, 138, 148,
      255, 245, 235, 192, 182, 172, 32, 42, 52};
  const auto image = DeepSeekV4ImageProcessor(TinyConfig()).ProcessImage(
      std::span<const uint8_t>(rgb), 2, 3);
  CHECK(image.image_grid_thw == std::array<int64_t, 3>{1, 1, 2});
  const std::vector<uint16_t> golden = {
      49024, 48895, 16256, 16130, 49004, 48855, 16236, 16091,
      48984, 48815, 16216, 16051, 15233, 48001, 48960, 48001,
      15785, 48001, 48940, 48001, 15909, 48001, 48920, 48001};
  CHECK(image.pixel_values_bf16 == golden);
}

TEST_CASE("deepseek-v4 minimum pixels upscale before patchification") {
  auto cfg = TinyConfig();
  cfg.min_pixels = 64;
  const auto rgb = IdentityRgb();
  const auto image = DeepSeekV4ImageProcessor(cfg).ProcessImage(
      std::span<const uint8_t>(rgb), 2, 2);
  CHECK(image.image_grid_thw == std::array<int64_t, 3>{1, 4, 4});
  CHECK(image.num_patches == 16);
  CHECK(image.patch_feature_dim == 12);
  REQUIRE(image.pixel_values_bf16.size() == 192);
  CHECK(image.pixel_values_bf16[0] == 49024);
  CHECK(image.pixel_values_bf16[11] == 49004);
  CHECK(image.pixel_values_bf16[75] == 48401);
  CHECK(image.pixel_values_bf16[96] == 48855);
  CHECK(image.pixel_values_bf16[191] == 16256);
}

TEST_CASE("deepseek-v4 Pillow bicubic resize matrix is byte exact") {
  const std::vector<uint8_t> seed_zero_rgb = {
      95, 130, 194, 217, 207, 235, 15, 163, 33, 215, 217, 130, 248, 189, 16,
      69, 184, 232, 205, 78, 169, 61, 125, 10, 29, 240, 66, 19, 182, 39,
      59, 4, 59, 81, 222, 44, 120, 122, 50, 208, 78, 28, 64, 166, 121};
  const auto reviewer_case =
      DeepSeekV4ImageProcessor(OracleConfig(77, 128, 8))
          .ProcessImage(std::span<const uint8_t>(seed_zero_rgb), 3, 5);
  REQUIRE(reviewer_case.pixel_values_bf16.size() == 198);
  CHECK(reviewer_case.image_grid_thw == std::array<int64_t, 3>{1, 6, 11});
  CHECK(reviewer_case.pixel_values_bf16[4] == 15877);
  CHECK(reviewer_case.pixel_values_bf16[12] == 48944);
  CHECK(HashBf16Words(reviewer_case.pixel_values_bf16) ==
        0xd328b8bac9ec61ecULL);

  struct Fixture {
    int64_t height;
    int64_t width;
    uint8_t seed;
    int64_t min_pixels;
    int64_t max_image_tokens;
    int64_t max_width_height_ratio;
    int64_t output_height;
    int64_t output_width;
    size_t output_words;
    uint64_t hash;
  };
  const std::array<Fixture, 5> fixtures = {{
      {2, 2, 1, 64, 128, 8, 8, 8, 192, 0x80a05e437d354185ULL},
      {13, 9, 2, 0, 40, 8, 5, 4, 60, 0x29a3fcc29a8f4d25ULL},
      {9, 15, 3, 0, 32, 8, 2, 3, 18, 0x9c7473c6224a81d2ULL},
      {3, 17, 4, 0, 64, 3, 3, 9, 81, 0x0280afcf5897e5a8ULL},
      {11, 7, 5, 0, 24, 8, 4, 2, 24, 0xe44671b5eb36727dULL},
  }};
  for (const Fixture& fixture : fixtures) {
    const auto rgb =
        FormulaRgb(fixture.height, fixture.width, fixture.seed);
    const auto image =
        DeepSeekV4ImageProcessor(OracleConfig(
            fixture.min_pixels, fixture.max_image_tokens,
            fixture.max_width_height_ratio))
            .ProcessImage(std::span<const uint8_t>(rgb), fixture.height,
                          fixture.width);
    CHECK(image.image_grid_thw ==
          std::array<int64_t, 3>{1, fixture.output_height,
                                 fixture.output_width});
    REQUIRE(image.pixel_values_bf16.size() == fixture.output_words);
    CHECK(HashBf16Words(image.pixel_values_bf16) == fixture.hash);
  }
}

TEST_CASE("deepseek-v4 wide rule is asymmetric") {
  auto cfg = TinyConfig();
  cfg.max_width_height_ratio = 2;
  std::vector<uint8_t> wide;
  for (int y = 0; y < 2; ++y) {
    for (int x = 0; x < 8; ++x) {
      wide.push_back(static_cast<uint8_t>(x * 20));
      wide.push_back(static_cast<uint8_t>(y * 60));
      wide.push_back(static_cast<uint8_t>(10 + x + y));
    }
  }
  const auto wide_image = DeepSeekV4ImageProcessor(cfg).ProcessImage(
      std::span<const uint8_t>(wide), 2, 8);
  CHECK(wide_image.image_grid_thw == std::array<int64_t, 3>{1, 1, 2});
  const std::vector<uint16_t> wide_golden = {
      49002, 48926, 49002, 48926, 49024, 49024, 48904, 48904,
      49002, 49000, 49000, 48998, 48787, 15425, 48787, 15425,
      49024, 49024, 48904, 48904, 48994, 48992, 48992, 48990};
  CHECK(wide_image.pixel_values_bf16 == wide_golden);

  std::vector<uint8_t> tall;
  for (int y = 0; y < 8; ++y) {
    for (int x = 0; x < 2; ++x) {
      tall.push_back(static_cast<uint8_t>(x * 60));
      tall.push_back(static_cast<uint8_t>(y * 20));
      tall.push_back(static_cast<uint8_t>(10 + x + y));
    }
  }
  const auto tall_image = DeepSeekV4ImageProcessor(cfg).ProcessImage(
      std::span<const uint8_t>(tall), 8, 2);
  CHECK(tall_image.image_grid_thw == std::array<int64_t, 3>{1, 4, 1});
  CHECK(tall_image.num_patches == 4);
}

TEST_CASE("deepseek-v4 exact wide threshold uses direct resize") {
  auto cfg = TinyConfig();
  cfg.max_width_height_ratio = 2;
  const auto rgb = FormulaRgb(3, 6, 111);
  const auto image = DeepSeekV4ImageProcessor(cfg).ProcessImage(
      std::span<const uint8_t>(rgb), 3, 6);
  CHECK(image.image_grid_thw == std::array<int64_t, 3>{1, 2, 3});
  REQUIRE(image.pixel_values_bf16.size() == 72);
  CHECK(HashBf16Words(image.pixel_values_bf16) ==
        0xab9bbef0bbb70c6aULL);
}

TEST_CASE("deepseek-v4 image block preserves start padding and row-pair order") {
  const auto block = BuildDeepSeekV4ImageBlock(2, 3, 0);
  CHECK(block.types ==
        std::vector<int64_t>{1, 1, 1, 0, 2, 2, 2, 2, 2, 2, 3, 3, 4});
  CHECK(block.permutation == std::vector<int64_t>{0, 3, 1, 4, 2, 5});

  const auto shifted = BuildDeepSeekV4ImageBlock(3, 2, 3);
  CHECK(shifted.types ==
        std::vector<int64_t>{0, 2, 2, 2, 2, 3, 3, 2, 1, 2, 1, 3, 1, 4});
  CHECK(shifted.permutation == std::vector<int64_t>{0, 2, 1, 3, 4, 5});
}

TEST_CASE("deepseek-v4 image block emits final pair-alignment padding") {
  const auto block = BuildDeepSeekV4ImageBlock(2, 2, 0);
  CHECK(block.types ==
        std::vector<int64_t>{1, 1, 1, 0, 2, 2, 2, 2, 3, 3, 1, 1, 4});
  CHECK(block.permutation == std::vector<int64_t>{0, 2, 1, 3});
  CHECK(GridTokens(2, 2, 1, 1).num_tokens ==
        static_cast<int64_t>(block.types.size()) - 3);
}

TEST_CASE("deepseek-v4 placeholders expand multiple images in source order") {
  auto cfg = TinyConfig();
  const auto rgb1 = IdentityRgb();
  std::vector<uint8_t> rgb2 = rgb1;
  rgb2[0] = 255;
  DeepSeekV4ImageProcessor processor(cfg);
  auto image1 = std::make_shared<vllm::multimodal::ImageKwargs>(
      processor.ProcessImage(std::span<const uint8_t>(rgb1), 2, 2));
  auto image2 = std::make_shared<vllm::multimodal::ImageKwargs>(
      processor.ProcessImage(std::span<const uint8_t>(rgb2), 2, 2));
  const std::string hash1 = processor.HashImage(
      std::span<const uint8_t>(rgb1), 2, 2);
  const std::string hash2 = processor.HashImage(
      std::span<const uint8_t>(rgb2), 2, 2);
  const auto inputs = PrepareDeepSeekV4Inputs(
      {7, 42, 8, 42, 9}, 42, {{image1, hash1}, {image2, hash2}}, cfg);
  CHECK(inputs.prompt_token_ids ==
        std::vector<int32_t>{7, 101, 101, 100, 102, 101, 103, 101, 104,
                             8, 101, 100, 102, 101, 103, 101, 104, 9});
  REQUIRE(inputs.mm_features.size() == 2);
  CHECK(inputs.mm_features[0].offset == 1);
  CHECK(inputs.mm_features[0].length == 8);
  CHECK(inputs.mm_features[0].data == image1);
  CHECK(inputs.mm_features[1].offset == 10);
  CHECK(inputs.mm_features[1].length == 7);
  CHECK(inputs.mm_features[1].data == image2);

  // THE KEY, and this is the assertion the two images exist to make.
  // `Scheduler::try_schedule_encoder_inputs` skips a feature whose `mm_hash` it
  // already scheduled this step, and the runner's `encoder_cache_` is a global
  // map on the same string, so two features that share a key are ONE image to
  // both: the second placeholder is filled from the first image's rows and the
  // answer is fluent. `rgb2` differs from `rgb1` in ONE byte, which is the
  // smallest difference a content hash has to see.
  CHECK_FALSE(hash1.empty());
  CHECK(hash1 != hash2);
  CHECK(inputs.mm_features[0].mm_hash != inputs.mm_features[1].mm_hash);
  CHECK(inputs.mm_features[0].mm_hash.rfind(hash1, 0) == 0);
  CHECK(inputs.mm_features[1].mm_hash.rfind(hash2, 0) == 0);
}

// THE OTHER HALF OF THE KEY, and it is why this architecture cannot use
// Qwen3-VL's content-only one. `BuildDeepSeekV4ImageBlock` reads
// `compress_pad = 3 - start_position % 4`, so ONE image at two offsets is two
// different blocks -- here 8 tokens and 7. A key that did not carry the offset
// term would make the scheduler serve the 8-token block's rows under the
// 7-token span.
TEST_CASE("deepseek-v4 one image at two offsets gets two DIFFERENT feature keys") {
  auto cfg = TinyConfig();
  const auto rgb = IdentityRgb();
  DeepSeekV4ImageProcessor processor(cfg);
  auto image = std::make_shared<vllm::multimodal::ImageKwargs>(
      processor.ProcessImage(std::span<const uint8_t>(rgb), 2, 2));
  const std::string content =
      processor.HashImage(std::span<const uint8_t>(rgb), 2, 2);

  const auto inputs = PrepareDeepSeekV4Inputs(
      {7, 42, 8, 42, 9}, 42, {{image, content}, {image, content}}, cfg);
  REQUIRE(inputs.mm_features.size() == 2);
  // The PREMISE: the two spans really are different lengths, so a shared key
  // would be a length error and not merely a content one.
  REQUIRE(inputs.mm_features[0].length != inputs.mm_features[1].length);
  CHECK(inputs.mm_features[0].mm_hash != inputs.mm_features[1].mm_hash);

  // ...and the SAME image at the SAME residue is the same key, which is the
  // cache hit that makes the scheduler's dedup correct rather than merely safe.
  CHECK(MakeDeepSeekV4MmHash(content, 2, 2, 3) ==
        MakeDeepSeekV4MmHash(content, 2, 2, 3));
  CHECK(MakeDeepSeekV4MmHash(content, 2, 2, 3) !=
        MakeDeepSeekV4MmHash(content, 2, 2, 2));
  CHECK(MakeDeepSeekV4MmHash(content, 2, 2, 3) !=
        MakeDeepSeekV4MmHash(content, 3, 2, 3));
  CHECK(MakeDeepSeekV4MmHash(content, 2, 2, 3) !=
        MakeDeepSeekV4MmHash(content, 2, 3, 3));
}

// A feature with no key is refused where it is BUILT, not discovered three
// components downstream as a wrong answer.
TEST_CASE("deepseek-v4 an image item with no content hash is refused by name") {
  auto cfg = TinyConfig();
  const auto rgb = IdentityRgb();
  DeepSeekV4ImageProcessor processor(cfg);
  auto image = std::make_shared<vllm::multimodal::ImageKwargs>(
      processor.ProcessImage(std::span<const uint8_t>(rgb), 2, 2));
  CHECK_THROWS_AS(
      PrepareDeepSeekV4Inputs({42}, 42, {{image, ""}}, cfg),
      std::invalid_argument);
}

// The content key is the hasher's, over the same bytes, and the model_id is a
// NAMESPACE: two engines serving different checkpoints must not collide in a
// process-global cache.
TEST_CASE("deepseek-v4 the content hash is the shared hasher's, keyed on model_id") {
  const auto rgb = IdentityRgb();
  auto cfg = TinyConfig();
  cfg.model_id = "deepseek-a";
  const DeepSeekV4ImageProcessor a(cfg);
  cfg.model_id = "deepseek-b";
  const DeepSeekV4ImageProcessor b(cfg);
  const std::string ha = a.HashImage(std::span<const uint8_t>(rgb), 2, 2);
  CHECK(ha == vllm::multimodal::MultiModalHasher::HashImageRGB(
                  "deepseek-a", rgb.data(), 2, 2));
  CHECK(ha != b.HashImage(std::span<const uint8_t>(rgb), 2, 2));
  // And a buffer that does not match its declared dimensions is refused rather
  // than read past its end.
  CHECK_THROWS_AS(a.HashImage(std::span<const uint8_t>(rgb), 2, 3),
                  std::invalid_argument);
}

TEST_CASE("deepseek-v4 placeholder and image counts must match") {
  const auto image = std::make_shared<vllm::multimodal::ImageKwargs>();
  CHECK_THROWS_WITH_AS(
      PrepareDeepSeekV4Inputs({1, 2}, 42, {{image, "c0"}}, TinyConfig()),
      "Found 0 image tokens but got 1 images", std::invalid_argument);
  CHECK_THROWS_WITH_AS(
      PrepareDeepSeekV4Inputs({42, 42}, 42, {{image, "c0"}}, TinyConfig()),
      "Found 2 image tokens but got 1 images", std::invalid_argument);
}

TEST_CASE("deepseek-v4 image block arithmetic rejects overflow before allocation") {
  CHECK_THROWS_WITH_AS(
      BuildDeepSeekV4ImageBlock(std::numeric_limits<int64_t>::max(), 1, 0),
      "DeepSeek-V4 image block size overflow", std::overflow_error);
  CHECK_THROWS_WITH_AS(
      BuildDeepSeekV4ImageBlock(1, std::numeric_limits<int64_t>::max(), 0),
      "DeepSeek-V4 image block size overflow", std::overflow_error);
}

TEST_CASE("deepseek-v4 placeholder expansion validates image shape and extent") {
  auto wrong_shape = std::make_shared<vllm::multimodal::ImageKwargs>();
  wrong_shape->image_grid_thw = {1, 1, 1};
  wrong_shape->num_patches = 2;
  wrong_shape->patch_feature_dim = 12;
  wrong_shape->pixel_values_bf16 = {0, 0};
  CHECK_THROWS_WITH_AS(
      PrepareDeepSeekV4Inputs({42}, 42, {{wrong_shape, "c0"}}, TinyConfig()),
      "DeepSeek-V4 image input shape is invalid", std::invalid_argument);

  auto wrong_feature_width =
      std::make_shared<vllm::multimodal::ImageKwargs>();
  wrong_feature_width->image_grid_thw = {1, 1, 1};
  wrong_feature_width->num_patches = 1;
  wrong_feature_width->patch_feature_dim = 1;
  wrong_feature_width->pixel_values_bf16 = {0};
  CHECK_THROWS_WITH_AS(
      PrepareDeepSeekV4Inputs({42}, 42, {{wrong_feature_width, "c0"}}, TinyConfig()),
      "DeepSeek-V4 image feature width does not match patch size",
      std::invalid_argument);

  auto wrong_extent = std::make_shared<vllm::multimodal::ImageKwargs>();
  wrong_extent->image_grid_thw = {1, 1, 1};
  wrong_extent->num_patches = 1;
  wrong_extent->patch_feature_dim = 12;
  CHECK_THROWS_WITH_AS(
      PrepareDeepSeekV4Inputs({42}, 42, {{wrong_extent, "c0"}}, TinyConfig()),
      "DeepSeek-V4 image BF16 extent does not match shape",
      std::invalid_argument);
}

TEST_CASE("deepseek-v4 placeholder expansion narrows before block allocation") {
  auto huge = std::make_shared<vllm::multimodal::ImageKwargs>();
  huge->image_grid_thw = {1, 1, std::numeric_limits<int32_t>::max() / 2};
  huge->num_patches = huge->image_grid_thw[2];
  huge->patch_feature_dim = 12;
  CHECK_THROWS_WITH_AS(
      PrepareDeepSeekV4Inputs({42}, 42, {{huge, "c0"}}, TinyConfig()),
      "DeepSeek-V4 image feature length exceeds int range",
      std::overflow_error);
}
