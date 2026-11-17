#pragma once

namespace vt {
// Approved FP32 reduction reassociation for bounded speculative GDN. Preserve
// an explicit 0 control; graph keys and operator dispatch share this default.
inline constexpr const char* kXpuGdnSpecSubgroupDefault = "1";
}  // namespace vt
