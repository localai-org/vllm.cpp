#pragma once
#include "vt/dtype.h"
#include <algorithm>
#include <span>
#include <vector>

namespace vt {
struct Exl3W8A8Panel {
  int first_column, columns, source_group;
};

struct Exl3W8A8PanelCapacity {
  int columns;
  size_t bytes;
};

// Only output-column decomposition changes. A panel never crosses an input
// scale/Hadamard source group, even if that group reappears later in the map.
inline std::vector<Exl3W8A8Panel> PlanExl3W8A8Panels(
    std::span<const int32_t> source_map, int groups, int max_columns) {
  VT_CHECK(max_columns == 128 || max_columns == 1024 || max_columns == 2048,
           "EXL3 W8A8 panel width must be128/1024/2048");
  VT_CHECK(groups > 0 && !source_map.empty() &&
               source_map.size() <= size_t(std::numeric_limits<int>::max() / 128),
           "EXL3 W8A8 invalid panel map geometry");
  // Validate the entire map before returning a runnable decomposition.
  for (int group : source_map)
    VT_CHECK(group >= 0 && group < groups, "EXL3 W8A8 shard_of_nb group out of range");
  std::vector<Exl3W8A8Panel> panels;
  for (size_t begin = 0; begin < source_map.size();) {
    size_t end = begin + 1;
    while (end < source_map.size() && end - begin < size_t(max_columns / 128) &&
           source_map[end] == source_map[begin]) ++end;
    panels.push_back({int(begin * 128), int((end - begin) * 128), source_map[begin]});
    begin = end;
  }
  return panels;
}

// One reusable int8 panel, not one expanded matrix or one panel per layer.
// The caller must separately own and retire this capacity on queue completion.
inline Exl3W8A8PanelCapacity PlanExl3W8A8PanelCapacity(
    int64_t k, int64_t n, int max_columns) {
  VT_CHECK(max_columns == 128 || max_columns == 1024 || max_columns == 2048,
           "EXL3 W8A8 panel width must be128/1024/2048");
  VT_CHECK(k > 0 && n > 0 && k % 128 == 0 && n % 128 == 0 &&
               k <= std::numeric_limits<int32_t>::max() / (127 * 127) &&
               n <= std::numeric_limits<int>::max(),
           "EXL3 W8A8 invalid bounded panel geometry");
  const int columns = int(std::min(n, int64_t(max_columns)));
  VT_CHECK(size_t(k) <= std::numeric_limits<size_t>::max() / size_t(columns),
           "EXL3 W8A8 panel size overflow");
  return {columns, size_t(k) * size_t(columns)};
}
}  // namespace vt
