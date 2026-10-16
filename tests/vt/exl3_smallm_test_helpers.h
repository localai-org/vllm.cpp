#pragma once
#include "xpu_test_helpers.h"
#include "vt/exl3_grouped.h"
#include "vt/unaligned.h"
#include <limits>

namespace exl3_smallm_test {
using vt::DType;
using xpu_test::Buffer;
using xpu_test::Queue;

inline double Relative(const std::vector<float>& got, const std::vector<float>& expected) {
  REQUIRE(got.size() == expected.size());
  double error = 0, norm = 0;
  bool finite = true;
  for (size_t i = 0; i < got.size(); ++i) {
    finite &= std::isfinite(got[i]) && std::isfinite(expected[i]);
    const double delta = double(got[i]) - expected[i];
    error += delta * delta;
    norm += double(expected[i]) * expected[i];
  }
  REQUIRE(finite);
  REQUIRE(norm > 0);
  return std::sqrt(error / norm);
}

struct Scratch {
  vt::Exl3SmallMPlan plan;
  Buffer had, parts;
  Scratch(vt::Queue& q, int m, int k, int n, int bits, int groups)
      : plan(vt::PlanExl3SmallM(m, k, n, bits)),
        had(q, DType::kF16, {groups, k / 16, plan.padded_rows, 16}),
        parts(q, DType::kF32, {plan.splits, m, n}) {
    // Stale/poisoned allocations must be fully overwritten, including DPAS
    // padded rows which the caller has not promised to initialize.
    std::vector<uint16_t> poison(had.bytes / 2, 0x7e00);
    had.upload(poison.data());
    parts.put(std::vector<float>(parts.bytes / 4,
                                 std::numeric_limits<float>::quiet_NaN()));
  }
};

inline void CheckPadding(Scratch& scratch, int m, int k, int groups) {
  const auto raw = scratch.had.download();
  bool zero = true;
  for (int g = 0; g < groups; ++g) for (int tile = 0; tile < k / 16; ++tile)
    for (int row = m; row < scratch.plan.padded_rows; ++row)
      for (int col = 0; col < 16; ++col) {
        const size_t offset = (((size_t(g) * (k / 16) + tile) *
                                 scratch.plan.padded_rows + row) * 16 + col) * 2;
        zero &= vt::LoadUnaligned<uint16_t>(raw.data() + offset) == 0;
      }
  CHECK(zero);
}

}  // namespace exl3_smallm_test
