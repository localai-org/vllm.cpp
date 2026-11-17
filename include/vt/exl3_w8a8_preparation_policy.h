#pragma once
#include "vt/dtype.h"
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace vt {
// Internal bounded experiment, independent of the accounted device budget.
// Parse separately so invalid settings can be tested without resetting the
// process-wide policy used by both dispatch and the preparation pool.
inline size_t ParseExl3W8A8PreparationLimit(const char* value) {
  if (!value || std::string_view(value) == "72") return size_t{72} * 1024 * 1024;
  VT_CHECK(std::string_view(value) == "64",
           "VT_XPU_W8A8_PREPARE_MAX_MIB must be64/72");
  return size_t{64} * 1024 * 1024;
}

inline size_t Exl3W8A8PreparationLimitBytes() {
  static const size_t limit = [] {
    const auto bytes = ParseExl3W8A8PreparationLimit(
        std::getenv("VT_XPU_W8A8_PREPARE_MAX_MIB"));
    // Cold initialization only; no per-operator logging or environment parse.
    std::fprintf(stderr, "XPU EXL3 W8A8 preparation maximum: %zu MiB (on demand)\n",
                 bytes / (1024 * 1024));
    return bytes;
  }();
  return limit;
}

inline bool Exl3W8A8PreparationAdmitted(size_t bytes) {
  return bytes > 0 && bytes <= Exl3W8A8PreparationLimitBytes();
}

// Do not let an exhausted/over-accounted budget wrap into apparent capacity.
inline bool Exl3W8A8GrowthFits(size_t budget, size_t allocated, size_t graph,
                              size_t current, size_t requested) {
  return allocated <= budget && graph <= budget - allocated &&
         (requested <= current || requested - current <= budget - allocated - graph);
}
}  // namespace vt
