// Kolibri-1 fp8-block dequant: NEON-vs-scalar BITWISE identity doctest
// (MODEL-TEXT-kolibri-1 perf row, Phase 3 NEON).
//
// The dequant kernel (src/vllm/model_executor/models/kolibri1_fp8_dequant.h)
// must be bit-identical elementwise to the scalar reference
// vt::F8E4M3ToF32 x scale, stored via vt::F32ToBF16 — the W3 token gate
// replays fixed argmax chains over the dequantized bf16 weights, so any
// bit divergence anywhere in the 256-value e4m3fn domain can flip a
// near-tie adjudication. This gate is exhaustive over the byte domain:
// all 256 e4m3fn values x representative block scales (1.0, a
// subnormal-range 2^-14, a large 3.0e34, an exact negative, a tie-pinning
// 1.015625 with 64 exact ties, a full-mantissa 1.1), compared as
// raw uint16 bit patterns, plus boundary chunks (odd K, small blocks) of
// the full row driver.

#include <doctest/doctest.h>

#include <cstdint>
#include <cstring>
#include <iomanip>
#include <ios>
#include <vector>

#include "vllm/model_executor/models/kolibri1_fp8_dequant.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"

namespace {

uint16_t ScalarOne(uint8_t b, float s) {
  return vt::F32ToBF16(vt::F8E4M3ToF32(b) * s);
}

// A deterministic spread of scales per block: hits the subnormal-weight
// range, the large-normal range, negatives, zero, and 1.0.
float ScaleForBlock(int64_t blk) {
  static const float kScales[] = {
      1.0f, 0x1p-14f /*subnormal-weight range*/, 3.0e34f /*large*/, -0.875f, 0.0f, 0x1p-9f, 448.0f};
  return kScales[static_cast<size_t>(blk) % 7];
}

}  // namespace

TEST_CASE(
    "kolibri1 fp8 dequant: NEON decode is bit-identical to the scalar "
    "reference over ALL 256 e4m3fn values") {
  std::vector<uint8_t> bytes(256);
  for (int i = 0; i < 256; ++i) bytes[static_cast<size_t>(i)] = static_cast<uint8_t>(i);

  // 1.015625f (0x3F820000) produces 64 exact bf16 ties across the 256
  // bytes, pinning the kernel's round-to-nearest-even tie break (a
  // tie-away kernel diverges on 32 of them). 1.1f (0x3F8CCCCD) is a
  // full-mantissa scale, pinning the scale's precision (a bf16-rounded
  // scale diverges on 96) and the scalar's f32-mul-then-bf16
  // double-rounding order.
  const float scales2[] = {1.0f, 0x1p-14f, 3.0e34f, -1.5f, 0.0f, 0x1p-9f,
                           1.015625f /*64 exact ties: pins RNE*/,
                           1.1f /*full mantissa: pins scale precision*/};
  for (float s : scales2) {
    // Pad past 16 so a full vector read never overruns. Two block scales
    // (K = 256, block_k = 128), both set to s.
    const float grid[2] = {s, s};
    std::vector<uint8_t> buf(bytes);
    buf.resize(256 + 16, 0xAB);
    std::vector<uint16_t> got(256, 0xBEEF), want(256, 0xDEAD);
    // 16-byte chunks (the NEON unit), then the scalar driver covers any
    // remainder through the same chunking.
    vllm::kolibri1_fp8::DequantRowsBf16(buf.data(), grid, 1, 0, 1, 256, 1, 128, got.data());
    for (int i = 0; i < 256; ++i) {
      want[static_cast<size_t>(i)] = ScalarOne(bytes[static_cast<size_t>(i)], s);
    }
    CHECK_EQ(std::memcmp(got.data(), want.data(), 512), 0);
    if (std::memcmp(got.data(), want.data(), 512) != 0) {
      for (int i = 0; i < 256; ++i) {
        if (got[static_cast<size_t>(i)] != want[static_cast<size_t>(i)]) {
          MESSAGE("divergence at byte 0x", std::hex, i, " scale ", s, ": got 0x",
                  got[static_cast<size_t>(i)], " want 0x", want[static_cast<size_t>(i)]);
          break;
        }
      }
    }
  }
}

TEST_CASE(
    "kolibri1 fp8 dequant: row driver is bit-identical to the scalar "
    "reference across ragged blocks and odd K") {
  // 5 rows x 45 cols, block 2 x 16: chunks cross neither block rows nor
  // block cols; K = 45 leaves a 13-wide tail chunk after the NEON bodies.
  const int64_t kN = 5, kK = 45, kBN = 2, kBK = 16;
  std::vector<uint8_t> bytes(static_cast<size_t>(kN * kK));
  for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i);
  const int64_t sc_cols = (kK + kBK - 1) / kBK;
  const int64_t sc_rows = (kN + kBN - 1) / kBN;
  std::vector<float> scales(static_cast<size_t>(sc_rows * sc_cols));
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = ScaleForBlock(static_cast<int64_t>(i) % sc_cols) *
                (0x1p-3f * static_cast<float>(i % 5) + 0.5f);
  }
  std::vector<uint16_t> got(static_cast<size_t>(kN * kK), 0xBEEF),
      want(static_cast<size_t>(kN * kK));
  vllm::kolibri1_fp8::DequantRowsBf16(bytes.data(), scales.data(), sc_cols, 0, kN, kK, kBN, kBK,
                                      got.data());
  for (int64_t n = 0; n < kN; ++n) {
    for (int64_t k = 0; k < kK; ++k) {
      const size_t idx = static_cast<size_t>(n * kK + k);
      want[idx] = ScalarOne(bytes[idx], scales[static_cast<size_t>((n / kBN) * sc_cols + k / kBK)]);
    }
  }
  CHECK_EQ(std::memcmp(got.data(), want.data(), want.size() * 2), 0);
}

TEST_CASE(
    "kolibri1 fp8 dequant: partial row ranges partition bit-identically "
    "(the pool contract at the kernel level)") {
  const int64_t kN = 4, kK = 64, kBN = 4, kBK = 32;
  std::vector<uint8_t> bytes(static_cast<size_t>(kN * kK));
  for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = static_cast<uint8_t>(i * 7);
  std::vector<float> scales(static_cast<size_t>(kN) * 2);
  for (size_t i = 0; i < scales.size(); ++i) {
    scales[i] = ScaleForBlock(static_cast<int64_t>(i));
  }
  std::vector<uint16_t> whole(static_cast<size_t>(kN * kK)), split(static_cast<size_t>(kN * kK), 0);
  vllm::kolibri1_fp8::DequantRowsBf16(bytes.data(), scales.data(), 2, 0, kN, kK, kBN, kBK,
                                      whole.data());
  vllm::kolibri1_fp8::DequantRowsBf16(bytes.data(), scales.data(), 2, 0, 1, kK, kBN, kBK,
                                      split.data());
  vllm::kolibri1_fp8::DequantRowsBf16(bytes.data(), scales.data(), 2, 1, 3, kK, kBN, kBK,
                                      split.data());
  vllm::kolibri1_fp8::DequantRowsBf16(bytes.data(), scales.data(), 2, 3, kN, kK, kBN, kBK,
                                      split.data());
  CHECK_EQ(std::memcmp(whole.data(), split.data(), whole.size() * 2), 0);
}
