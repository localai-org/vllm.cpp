#pragma once
#include "vt/paged_attn_route.h"
#include "vt/dtype.h"
#include <cstdlib>
#include <string_view>

namespace vt {
inline constexpr const char* kXpuLongDecodeDefault = "2";
inline constexpr const char* kXpuLongDecodeSplitsDefault = "32";
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
