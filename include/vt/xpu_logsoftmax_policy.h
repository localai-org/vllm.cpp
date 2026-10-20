#pragma once
#include "vt/dtype.h"
#include <cstdlib>
#include <cstring>
namespace vt {
inline constexpr const char* kXpuRawLogsoftmaxSplitDefault = "1";
inline bool XpuRawLogsoftmaxSplit(int64_t rows, int64_t vocab) {
  if (rows != 1 || vocab != 248320) return false;
  const char* value = std::getenv("VT_XPU_RAW_LOGSOFTMAX_SPLIT");
  if (!value) value = kXpuRawLogsoftmaxSplitDefault;
  VT_CHECK(std::strcmp(value,"0") == 0 || std::strcmp(value,"1") == 0,
           "VT_XPU_RAW_LOGSOFTMAX_SPLIT must be 0/1");
  return std::strcmp(value,"1") == 0;
}
}  // namespace vt
