#pragma once
#include <cstddef>

namespace vt {
inline bool XpuDeviceAllocationFits(std::size_t budget, std::size_t allocated,
                                    std::size_t graph, std::size_t requested) {
  // Check each subtraction before using the remainder: over-accounting must
  // refuse allocation rather than wrapping into apparently available bytes.
  return allocated <= budget && graph <= budget - allocated &&
         requested <= budget - allocated - graph;
}
}  // namespace vt
