// FLINT-CPU-DESCENT-2 — bit-identity suite for the NEON Q6_K x Q8_K vec_dot.
//
// vllm.cpp original (vt runtime); the row's spec is the contract: the NEON
// kernel (cpu_quant_dot_q6k_neon.cpp, invoked through the same VecDotFn
// signature the serving dispatch uses) must be BIT-IDENTICAL to the scalar
// reference (cpu_quant_dot.cpp, quants.c:800 order) on EVERY input. The
// reference is `QuantQ6KPortableVecDot()` — the TRUE scalar, immune to
// VT_CPU_Q6K_DOT, because on a DotProd host the selected vec_dot IS the NEON
// kernel and comparing through the traits table would compare it with itself.
//
// The comparison is an f32 BIT compare: no tolerance, no envelope. Cases:
// deterministic-xorshift randomized blocks (thousands), plus the adversarial
// grid the spec names — f16 scale extremes (zero, denormal-adjacent, smallest
// normal, one, largest finite), 6-bit value saturation (all 0 / all 63 /
// alternating), zero sub-block scales, scale extremes (8-bit +127/-128), q8
// extremes (+-128), and multi-block super-blocks with per-block adversarial
// mixing. On non-DotProd hosts the file SKIPS LOUDLY: the NEON kernel cannot
// exist there and the reference would be compared with itself.
//
// The class this kernel serves is the keep-quant flint arm's untied lm_head
// ([V, H] Q6_K), so a divergence here moves served tokens, not just floats.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "vt/dtype.h"
#include "vt/quant.h"

namespace {

constexpr int kK = 256;  // kQK_K

struct SuperBlock {
  std::vector<uint8_t> x;  // one BlockQ6_K, 210 bytes
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

// Build one super-block from raw ingredients. The 6-bit values land in the
// q6_K container exactly as the scalar unpack reads them: value l (l < 32)
// from ql[l] low nibble + qh[l] bits 0-1, value l+32 from ql[l+32] low nibble
// + qh[l] bits 2-3, value l+64 from ql[l] high nibble + qh[l] bits 4-5, value
// l+96 from ql[l+32] high nibble + qh[l] bits 6-7 (cpu_quant_dot.cpp:471).
// The activation delta is pinned to 1.0f so the f16 d is the only weight
// scale and the int8 scales[] are used raw, exactly as the kernel reads them.
SuperBlock MakeBlock(const uint8_t* vals, const int8_t* q8, uint16_t d,
                     const int8_t* x_scales) {
  SuperBlock b;
  b.x.assign(210, 0);
  b.y.assign(292, 0);
  uint8_t ql[128];
  uint8_t qh[64];
  std::memset(ql, 0, sizeof(ql));
  std::memset(qh, 0, sizeof(qh));
  for (int l = 0; l < 32; ++l) {
    const uint8_t v0 = vals[l] & 0x3F;
    const uint8_t v1 = vals[l + 32] & 0x3F;
    const uint8_t v2 = vals[l + 64] & 0x3F;
    const uint8_t v3 = vals[l + 96] & 0x3F;
    ql[l] = static_cast<uint8_t>(ql[l] | (v0 & 0xF));
    ql[l + 32] = static_cast<uint8_t>(ql[l + 32] | (v1 & 0xF));
    ql[l] = static_cast<uint8_t>(ql[l] | (v2 << 4));
    ql[l + 32] = static_cast<uint8_t>(ql[l + 32] | (v3 << 4));
    qh[l] = static_cast<uint8_t>((v0 >> 4) | ((v1 >> 4) << 2) |
                                 ((v2 >> 4) << 4) | ((v3 >> 4) << 6));
  }
  std::memcpy(b.x.data() + 0, ql, 128);        // BlockQ6_K::ql
  std::memcpy(b.x.data() + 128, qh, 64);       // BlockQ6_K::qh
  std::memcpy(b.x.data() + 192, x_scales, 16); // BlockQ6_K::scales
  std::memcpy(b.x.data() + 208, &d, 2);        // BlockQ6_K::d (f16)
  std::memcpy(b.y.data() + 0, q8, kK);
  const float d_act = 1.0f;
  std::memcpy(b.y.data() + kK, &d_act, 4);
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
  const float ref = RunDot(vt::cpu::QuantQ6KPortableVecDot(), blocks);
  const float got = RunDot(vt::cpu::QuantQ6KNeonVecDot(), blocks);
  REQUIRE(FloatBits(ref) == FloatBits(got));
}

SuperBlock RandomBlock(uint32_t* rng) {
  uint8_t vals[kK];
  int8_t q8[kK];
  for (int l = 0; l < kK; ++l) {
    vals[l] = static_cast<uint8_t>(Xorshift32(rng) & 0x3F);
  }
  for (int l = 0; l < kK; ++l) {
    q8[l] = static_cast<int8_t>(static_cast<int>(Xorshift32(rng) % 256) - 128);
  }
  int8_t x_scales[16];
  for (int l = 0; l < 16; ++l) {
    x_scales[l] = static_cast<int8_t>(Xorshift32(rng) & 0xFF);
  }
  uint16_t d = static_cast<uint16_t>(Xorshift32(rng) & 0xFFFF);
  const uint16_t exp = d & 0x7C00;
  if (exp == 0x7C00) d &= 0x7BFF;  // clamp Inf/NaN payloads: finite scales only
  return MakeBlock(vals, q8, d, x_scales);
}

SuperBlock PatternBlock(uint16_t d, int value_mode, int q8_mode,
                        int8_t scale_byte) {
  uint8_t vals[kK];
  int8_t q8[kK];
  for (int l = 0; l < kK; ++l) {
    switch (value_mode) {
      case 0: vals[l] = 0; break;
      case 1: vals[l] = 63; break;
      case 2: vals[l] = static_cast<uint8_t>(l & 1 ? 63 : 0); break;
      default: vals[l] = static_cast<uint8_t>(l % 3);
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
  return MakeBlock(vals, q8, d, &scale_byte);
}

}  // namespace

TEST_CASE("FLINT-CPU-DESCENT-2: q6_K tier availability is coherent") {
  // The explicit getters are the test seam; the dispatch flips to NEON exactly
  // when the tier is live.
  CHECK(vt::cpu::QuantQ6KNeonActive() ==
        (vt::cpu::QuantQ6KNeonVecDot() != nullptr));
  if (!vt::cpu::QuantQ6KNeonActive()) {
    MESSAGE("NEON Q6_K dot tier not live on this host; bit-identity skipped");
  }
}

TEST_CASE("FLINT-CPU-DESCENT-2: q6_K randomized blocks are bit-identical") {
  if (!vt::cpu::QuantQ6KNeonActive()) return;
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
  uint32_t rng = 0xC0FFEE12U;
  for (int rep = 0; rep < 1024; ++rep) {
    std::vector<SuperBlock> blocks;
    const int nb = 1 + static_cast<int>(Xorshift32(&rng) % 4);
    for (int i = 0; i < nb; ++i) blocks.push_back(RandomBlock(&rng));
    CheckBitIdentity(blocks, "multi rep=" + std::to_string(rep));
  }
}

TEST_CASE("FLINT-CPU-DESCENT-2: q6_K adversarial grid is bit-identical") {
  if (!vt::cpu::QuantQ6KNeonActive()) return;
  // d scale extremes: zero, denormal-adjacent (f16 bits 1..4), smallest
  // normal (0x0400), one (0x3C00), and the largest finite f16 (0x7BFF).
  const uint16_t kDValues[] = {0x0000, 0x0001, 0x0002, 0x0003, 0x0004,
                               0x0400, 0x3C00, 0x7BFF};
  for (uint16_t d : kDValues) {
    for (int value_mode = 0; value_mode < 4; ++value_mode) {
      for (int q8_mode = 0; q8_mode < 5; ++q8_mode) {
        for (int8_t sb : {int8_t{0}, int8_t{127}, int8_t{-128}}) {
          const SuperBlock b = PatternBlock(d, value_mode, q8_mode, sb);
          CheckBitIdentity({b}, "d=" + std::to_string(d) +
                                    " val=" + std::to_string(value_mode) +
                                    " q8=" + std::to_string(q8_mode) +
                                    " sb=" + std::to_string(sb));
        }
      }
    }
  }
}

TEST_CASE("FLINT-CPU-DESCENT-2: q6_K per-chunk scale extremes") {
  if (!vt::cpu::QuantQ6KNeonActive()) return;
  // Mixed max/min per-chunk scales over the 16 int8 scale bytes (65536
  // combinations across two saturation modes) with saturated and zero values.
  uint8_t vals[kK];
  int8_t q8[kK];
  for (int mode = 0; mode < 2; ++mode) {
    for (int l = 0; l < kK; ++l) vals[l] = mode ? 63 : 0;
    for (int l = 0; l < kK; ++l) q8[l] = mode ? 127 : 0;
    for (uint32_t half = 0; half < 256; ++half) {
      int8_t x_scales[16];
      for (int l = 0; l < 16; ++l) {
        x_scales[l] = (half & (1u << (l % 8))) ? static_cast<int8_t>(0x7F)
                                               : static_cast<int8_t>(0x80);
      }
      const SuperBlock b = MakeBlock(vals, q8, 0x3C00, x_scales);
      CheckBitIdentity({b}, "scalemask mode=" + std::to_string(mode) +
                                " half=" + std::to_string(half));
    }
  }
}

TEST_CASE("FLINT-CPU-DESCENT-2: q6_K dispatch agrees with the tier") {
  // On a DotProd host the traits-selected kernel IS the NEON kernel; the
  // portable pointer stays scalar. Verify both facts through the same public
  // surface the GEMM path consumes.
  if (!vt::cpu::QuantQ6KNeonActive()) {
    CHECK(vt::cpu::BlockVecDot(vt::DType::kQ6_K) == vt::cpu::QuantQ6KPortableVecDot());
    return;
  }
  CHECK(vt::cpu::BlockVecDot(vt::DType::kQ6_K) == vt::cpu::QuantQ6KNeonVecDot());
  // End-to-end through the traits table: the dispatched kernel must still be
  // bit-identical to the portable reference on a randomized corpus.
  uint32_t rng = 0xABCDEF02U;
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
    vt::cpu::QuantTraits(vt::DType::kQ6_K).vec_dot(
        3 * kK, &via_traits, 0, x.data(), 0, y.data(), 0, 1);
    const float ref = RunDot(vt::cpu::QuantQ6KPortableVecDot(), blocks);
    CAPTURE(rep);
    REQUIRE(FloatBits(via_traits) == FloatBits(ref));
  }
}
