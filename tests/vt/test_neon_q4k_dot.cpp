// FLINT-NEON-Q4K-DOT — bit-identity suite for the NEON Q4_K x Q8_K vec_dot.
//
// vllm.cpp original (vt runtime); the row's spec is the contract: the NEON
// kernel (cpu_quant_dot_q4k_neon.cpp, invoked through the same VecDotFn
// signature the serving dispatch uses) must be BIT-IDENTICAL to the scalar
// reference (cpu_quant_dot.cpp, quants.c:645 order) on EVERY input. The
// reference is `QuantQ4KPortableVecDot()` — the TRUE scalar, immune to
// VT_CPU_Q4K_DOT, because on a DotProd host the selected vec_dot IS the NEON
// kernel and comparing through the traits table would compare it with itself.
//
// The comparison is an f32 BIT compare: no tolerance, no envelope. Cases:
// deterministic-xorshift randomized blocks (thousands), plus the adversarial
// grid the spec names — scale extremes (6-bit scales 0 and 63, per-sub-block
// max/min mixes), nibble saturation (all 0x0 / all 0xF / alternating), zero
// sub-blocks, zero super-block scales, denormal-adjacent f16 scales
// (bits 1..4), q8 extremes (+-128), and multi-block super-blocks with
// per-block adversarial mixing. On non-DotProd hosts the file SKIPS LOUDLY:
// the NEON kernel cannot exist there and the reference would be compared with
// itself.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/quant.h"

namespace {

constexpr int kK = 256;  // kQK_K

struct SuperBlock {
  std::vector<uint8_t> x;  // one BlockQ4_K, 144 bytes
  std::vector<uint8_t> y;  // one BlockQ8_K, 292 bytes
};

uint32_t Xorshift32(uint32_t* state) {
  uint32_t x = *state;
  x ^= x << 13;
  x ^= x >> 17;
  x ^= x << 5;
  *state = x;
  return x;
}

// Build one super-block from raw ingredients; bsums computed exactly as the
// Q8_K quantizer does (sum of quants in groups of 16) and the activation
// delta pinned to 1.0f, so the f16 d/dmin ingredients are the only scales.
SuperBlock MakeBlock(const uint8_t* nib, const int8_t* q8, uint16_t d,
                     uint16_t dmin, const uint8_t* x_scales) {
  SuperBlock b;
  b.x.assign(144, 0);
  b.y.assign(292, 0);
  std::memcpy(b.x.data() + 0, &d, 2);
  std::memcpy(b.x.data() + 2, &dmin, 2);
  std::memcpy(b.x.data() + 4, x_scales, 12);
  for (int l = 0; l < 128; ++l) {
    b.x[16 + l] =
        static_cast<uint8_t>(nib[2 * l] | (nib[2 * l + 1] << 4));
  }
  std::memcpy(b.y.data() + 0, q8, kK);
  const float d_act = 1.0f;
  std::memcpy(b.y.data() + kK, &d_act, 4);
  for (int j = 0; j < 16; ++j) {
    int16_t s = 0;
    for (int l = 0; l < 16; ++l) {
      s = static_cast<int16_t>(s + q8[16 * j + l]);
    }
    std::memcpy(b.y.data() + kK + 4 + 2 * j, &s, 2);
  }
  return b;
}

float RunDot(vt::cpu::VecDotFn fn, const std::vector<SuperBlock>& blocks) {
  std::vector<uint8_t> x;
  std::vector<uint8_t> y;
  for (const SuperBlock& b : blocks) {
    x.insert(x.end(), b.x.begin(), b.x.end());
    y.insert(y.end(), b.y.begin(), b.y.end());
  }
  float s = 0.0f;
  fn(static_cast<int>(blocks.size()) * kK, &s, 0, x.data(), 0, y.data(), 0, 1);
  return s;
}

uint32_t FloatBits(float f) {
  uint32_t bits;
  std::memcpy(&bits, &f, 4);
  return bits;
}

void CheckBitIdentity(const std::vector<SuperBlock>& blocks,
                      const std::string& label) {
  CAPTURE(label);
  const float ref = RunDot(vt::cpu::QuantQ4KPortableVecDot(), blocks);
  const float got = RunDot(vt::cpu::QuantQ4KNeonVecDot(), blocks);
  REQUIRE(FloatBits(ref) == FloatBits(got));
}

SuperBlock RandomBlock(uint32_t* rng) {
  uint8_t nib[kK];
  int8_t q8[kK];
  for (int l = 0; l < kK; ++l) {
    nib[l] = static_cast<uint8_t>(Xorshift32(rng) & 0xF);
  }
  for (int l = 0; l < kK; ++l) {
    q8[l] = static_cast<int8_t>(static_cast<int>(Xorshift32(rng) % 256) - 128);
  }
  uint8_t x_scales[12];
  for (int l = 0; l < 12; ++l) {
    x_scales[l] = static_cast<uint8_t>(Xorshift32(rng) & 0xFF);
  }
  const uint16_t d = static_cast<uint16_t>(Xorshift32(rng) & 0xFFFF);
  const uint16_t dmin = static_cast<uint16_t>(Xorshift32(rng) & 0xFFFF);
  return MakeBlock(nib, q8, d, dmin, x_scales);
}

SuperBlock PatternBlock(uint16_t d, uint16_t dmin, int nibble_mode,
                        int q8_mode, uint8_t scale_byte) {
  uint8_t nib[kK];
  int8_t q8[kK];
  for (int l = 0; l < kK; ++l) {
    switch (nibble_mode) {
      case 0: nib[l] = 0x0; break;
      case 1: nib[l] = 0xF; break;
      case 2: nib[l] = static_cast<uint8_t>(l & 1 ? 0xF : 0x0); break;
      default: nib[l] = static_cast<uint8_t>(l % 3);
    }
  }
  for (int l = 0; l < kK; ++l) {
    switch (q8_mode) {
      case 0: q8[l] = 0; break;
      case 1: q8[l] = 127; break;
      case 2: q8[l] = -128; break;
      case 3: q8[l] = static_cast<int8_t>(l % 2 ? -128 : 127); break;
      default: q8[l] = static_cast<int8_t>((l * 37) % 7 - 3);
    }
  }
  uint8_t x_scales[12];
  for (int l = 0; l < 12; ++l) x_scales[l] = scale_byte;
  return MakeBlock(nib, q8, d, dmin, x_scales);
}

}  // namespace

TEST_CASE("FLINT-NEON-Q4K-DOT: tier availability is coherent") {
  // The explicit getters are the test seam; the dispatch flips to NEON exactly
  // when the tier is live.
  CHECK(vt::cpu::QuantQ4KNeonActive() ==
        (vt::cpu::QuantQ4KNeonVecDot() != nullptr));
  if (!vt::cpu::QuantQ4KNeonActive()) {
    MESSAGE("NEON Q4_K dot tier not live on this host; bit-identity skipped");
  }
}

TEST_CASE("FLINT-NEON-Q4K-DOT: randomized blocks are bit-identical") {
  if (!vt::cpu::QuantQ4KNeonActive()) return;
  // 4096 single-super-block cases across four seeds...
  for (uint32_t seed = 1; seed <= 4; ++seed) {
    uint32_t rng = seed * 0x9E3779B9U;
    for (int rep = 0; rep < 1024; ++rep) {
      const std::string label =
          "seed=" + std::to_string(seed) + " rep=" + std::to_string(rep);
      CheckBitIdentity({RandomBlock(&rng)}, label);
    }
  }
  // ...plus 1024 multi-super-block cases (1..4 blocks, nb>1 exercises the
  // cross-block float accumulation order).
  uint32_t rng = 0xC0FFEE11U;
  for (int rep = 0; rep < 1024; ++rep) {
    std::vector<SuperBlock> blocks;
    const int nb = 1 + static_cast<int>(Xorshift32(&rng) % 4);
    for (int i = 0; i < nb; ++i) blocks.push_back(RandomBlock(&rng));
    CheckBitIdentity(blocks, "multi rep=" + std::to_string(rep));
  }
}

TEST_CASE("FLINT-NEON-Q4K-DOT: adversarial grid is bit-identical") {
  if (!vt::cpu::QuantQ4KNeonActive()) return;
  // d/dmin scale extremes: zero, denormal-adjacent (f16 bits 1..4), smallest
  // normal (0x0400), one (0x3C00), and the largest finite f16 (0x7BFF).
  const uint16_t kDValues[] = {0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
                               0x0400, 0x3C00, 0x7BFF};
  // scale_byte drives the 6-bit scales/mins; 0x00 = all-zero, 0xFF decodes to
  // the max 63 for every sub-block.
  for (uint16_t d : kDValues) {
    for (uint16_t dmin : kDValues) {
      for (int nib_mode = 0; nib_mode < 4; ++nib_mode) {
        for (int q8_mode = 0; q8_mode < 5; ++q8_mode) {
          for (uint8_t sb : {uint8_t{0x00}, uint8_t{0xFF}}) {
            const SuperBlock b =
                PatternBlock(d, dmin, nib_mode, q8_mode, sb);
            CheckBitIdentity(
                {b}, "d=" + std::to_string(d) + " dmin=" + std::to_string(dmin) +
                         " nib=" + std::to_string(nib_mode) +
                         " q8=" + std::to_string(q8_mode) +
                         " sb=" + std::to_string(sb));
          }
        }
      }
    }
  }
}

TEST_CASE("FLINT-NEON-Q4K-DOT: per-sub-block scale extremes") {
  if (!vt::cpu::QuantQ4KNeonActive()) return;
  // Mixed max/min per-sub-block scales land on specific utmp bytes; sweep a
  // per-byte 00/FF choice over the 12 scale bytes (4096 combinations) with
  // saturated and zero nibbles.
  uint8_t nib[kK];
  int8_t q8[kK];
  for (int mode = 0; mode < 2; ++mode) {
    for (int l = 0; l < kK; ++l) nib[l] = mode ? 0xF : 0x0;
    for (int l = 0; l < kK; ++l) q8[l] = mode ? 127 : 0;
    for (uint32_t mask = 0; mask < 4096; ++mask) {
      uint8_t x_scales[12];
      for (int l = 0; l < 12; ++l) {
        x_scales[l] = (mask & (1u << l)) ? 0xFF : 0x00;
      }
      const SuperBlock b = MakeBlock(nib, q8, 0x3C00, 0x3C00, x_scales);
      CheckBitIdentity({b}, "scalemask mode=" + std::to_string(mode) +
                                " mask=" + std::to_string(mask));
    }
  }
}

TEST_CASE("FLINT-NEON-Q4K-DOT: dispatch agrees with the tier") {
  // Dispatch composes the Arm selector with the outer x86 AVX2 selector.
  // Preserve each selector's current environment override and verify the
  // production surface consumed by GEMM.
  const vt::cpu::VecDotFn portable = vt::cpu::QuantQ4KPortableVecDot();
  vt::cpu::VecDotFn selected = vt::cpu::SelectQuantQ4KVecDot(portable);
#if defined(__x86_64__) || defined(_M_X64)
  selected = vt::cpu::SelectQuantKVecDot(vt::DType::kQ4_K, selected);
#endif
  CHECK(vt::cpu::BlockVecDot(vt::DType::kQ4_K) == selected);
  CHECK(vt::cpu::QuantTraits(vt::DType::kQ4_K).vec_dot == selected);
  // Separately: in auto mode the dispatch IS the NEON kernel exactly when the
  // HWCAP dotprod probe is live (QuantQ4KNeonVecDot() is that probe). A
  // `portable` env override legitimately suppresses it, so this arm only runs
  // when the env does not force the scalar tier.
  const char* dot_env = std::getenv("VT_CPU_Q4K_DOT");
  const bool forced_portable =
      dot_env != nullptr && std::strcmp(dot_env, "portable") == 0;
  if (!forced_portable && vt::cpu::QuantQ4KNeonActive()) {
    CHECK(vt::cpu::SelectQuantQ4KVecDot(portable) ==
          vt::cpu::QuantQ4KNeonVecDot());
  } else {
    CHECK(vt::cpu::SelectQuantQ4KVecDot(portable) == portable);
  }
  // Traits must execute the selected kernel. The scalar and NEON tiers also
  // require portable bit identity. AVX2 arithmetic has its independent oracle
  // and tolerance in test_ops_quant_dot, including adversarial inputs.
  uint32_t rng = 0xABCDEF01U;
  for (int rep = 0; rep < 256; ++rep) {
    std::vector<SuperBlock> blocks;
    for (int i = 0; i < 3; ++i) blocks.push_back(RandomBlock(&rng));
    std::vector<uint8_t> x;
    std::vector<uint8_t> y;
    for (const SuperBlock& b : blocks) {
      x.insert(x.end(), b.x.begin(), b.x.end());
      y.insert(y.end(), b.y.begin(), b.y.end());
    }
    float via_traits = 0.0f;
    vt::cpu::QuantTraits(vt::DType::kQ4_K).vec_dot(
        3 * kK, &via_traits, 0, x.data(), 0, y.data(), 0, 1);
    const float ref = RunDot(vt::cpu::QuantQ4KPortableVecDot(), blocks);
    CAPTURE(rep);
    const float via_selected = RunDot(selected, blocks);
    REQUIRE(FloatBits(via_traits) == FloatBits(via_selected));
    if (selected == portable || selected == vt::cpu::QuantQ4KNeonVecDot()) {
      REQUIRE(FloatBits(via_traits) == FloatBits(ref));
    }
  }
}
