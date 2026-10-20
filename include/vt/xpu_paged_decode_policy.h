#pragma once
#include "vt/paged_attn_route.h"
#include "vt/dtype.h"
#include <cstdlib>
#include <string_view>

namespace vt {
inline constexpr const char* kXpuLongDecodeDefault = "2";
inline constexpr const char* kXpuLongDecodeSplitsDefault = "32";
// Qualified pure B2..4 Q1; explicit0 retains the prior generic fallback.
inline constexpr const char* kXpuBatchedQ1Default = "1";
inline bool XpuBatchedQ1Enabled() {
  const auto* setting = std::getenv("VT_XPU_XE2_BATCHED_Q1");
  const std::string_view flag = setting ? setting : kXpuBatchedQ1Default;
  VT_CHECK(flag == "0" || flag == "1", "VT_XPU_XE2_BATCHED_Q1 must be0/1");
  return flag == "1";
}
inline bool XpuPureBatchedQ1Shape(int64_t tokens, int64_t requests,
                                const int32_t* host_offsets) {
  // Total Q == B also describes valid ragged packets such as [0,0,2].
  // Existing host metadata must prove every width; never download device data.
  if (requests < 2 || requests > 4 || tokens != requests || !host_offsets)
    return false;
  for (int64_t r = 0; r <= requests; ++r)
    if (host_offsets[r] != r) return false;
  return true;
}
inline XpuPagedDecodePlan XpuPagedDecodePlanFromEnvironment(int64_t length, int64_t page) {
  const auto* enabled = std::getenv("VT_XPU_XE2_LONG_DECODE");
  const auto* parts = std::getenv("VT_XPU_XE2_DECODE_SPLITS");
  const std::string_view flag = enabled ? enabled : kXpuLongDecodeDefault;
  const std::string_view splits = parts ? parts : kXpuLongDecodeSplitsDefault;
  // Mode2 extends Q1 decode to at most128 physical pages. Mode0/1 retain
  // the original policies as explicit rollback options.
  VT_CHECK(flag == "0" || flag == "1" || flag == "2",
           "VT_XPU_XE2_LONG_DECODE must be0/1/2");
  VT_CHECK(splits == "16" || splits == "32", "VT_XPU_XE2_DECODE_SPLITS must be16/32");
  return PagedAttnXpuDecodePlan(length, page, flag != "0",
                               splits == "16" ? 16 : 32, flag == "2");
}
}  // namespace vt
