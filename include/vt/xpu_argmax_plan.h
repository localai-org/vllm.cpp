#pragma once
#include "vt/dtype.h"
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace vt {
inline constexpr std::size_t kXpuArgmaxPartialBytes = 16;
inline constexpr std::size_t kXpuArgmaxMaxWorkspaceBytes = 16 * 1024 * 1024;

inline std::size_t XpuTiledArgmaxWorkspaceBytes(int64_t rows, int64_t vocab, int tile) {
  VT_CHECK(tile == 4096 || tile == 8192, "XPU tiled argmax tile must be 4096/8192");
  VT_CHECK(rows >= 0 && vocab > 0, "XPU tiled argmax requires nonnegative rows and positive vocab");
  if (rows == 0) return 0;
  const uint64_t tiles = 1 + uint64_t(vocab - 1) / uint64_t(tile);
  constexpr auto maximum = kXpuArgmaxMaxWorkspaceBytes / kXpuArgmaxPartialBytes;
  VT_CHECK(uint64_t(rows) <= maximum && tiles <= maximum / uint64_t(rows),
           "XPU tiled argmax workspace exceeds 16 MiB");
  VT_CHECK(uint64_t(rows) <= std::numeric_limits<std::size_t>::max() / sizeof(float) / uint64_t(vocab),
           "XPU tiled argmax logits byte size overflow");
  return std::size_t(uint64_t(rows) * tiles) * kXpuArgmaxPartialBytes;
}
struct XpuMappedSelectionLayout {
  std::size_t id_bytes, scratch_offset, scratch_bytes, total_bytes;
};
inline std::size_t XpuCompactLogprobsWorkspaceBytes(int64_t rows, int64_t vocab) {
  // Integer count partials share the argmax planner's checked shape bound.
  return XpuTiledArgmaxWorkspaceBytes(rows, vocab, 4096) / kXpuArgmaxPartialBytes * sizeof(int32_t);
}
inline bool XpuCompactLogprobs0Enabled(int64_t rows, int64_t vocab) {
  const char* value = std::getenv("VT_XPU_COMPACT_LOGPROBS0");
  if (!value) value = "0";
  VT_CHECK(std::strcmp(value,"0") == 0 || std::strcmp(value,"1") == 0,
           "VT_XPU_COMPACT_LOGPROBS0 must be 0/1");
  return std::strcmp(value,"1") == 0 && (rows == 1 || rows == 4) && vocab == 248320;
}
inline XpuMappedSelectionLayout XpuMappedSelectionBuffers(int64_t rows, int64_t vocab, int tile) {
  const auto scratch = XpuTiledArgmaxWorkspaceBytes(rows, vocab, tile);
  VT_CHECK(uint64_t(rows) <= (std::numeric_limits<std::size_t>::max() - 63) / sizeof(int32_t),
           "XPU mapped selection ID bytes overflow");
  const auto ids = std::size_t(rows) * sizeof(int32_t);
  const auto offset = (ids + 63) & ~std::size_t{63};
  VT_CHECK(scratch <= std::numeric_limits<std::size_t>::max() - offset,
           "XPU mapped selection allocation bytes overflow");
  return {ids, offset, scratch, offset + scratch};
}
// Opt-in caller experiment until complete MTP serving/ownership qualification.
inline int XpuMappedArgmaxTile(int64_t rows, int64_t vocab, int64_t target_vocab) {
  const char* value = std::getenv("VT_XPU_MAPPED_ARGMAX_TILE");
  if (!value) value = "0";
  int tile = 0;
  if (std::strcmp(value, "4096") == 0) tile = 4096;
  else if (std::strcmp(value, "8192") == 0) tile = 8192;
  else VT_CHECK(std::strcmp(value, "0") == 0, "VT_XPU_MAPPED_ARGMAX_TILE must be 0/4096/8192");
  return (rows == 1 || rows == 4) && vocab == 65536 && target_vocab == 248320 ? tile : 0;
}
// Separate opt-in for the asynchronous verification owner; target sampler
// qualification does not qualify this lifetime or the acceptance consumer.
inline int XpuRejectionArgmaxTile(int64_t rows, int64_t vocab) {
  const char* value = std::getenv("VT_XPU_REJECTION_ARGMAX_TILE");
  if (!value) value = "0";
  int tile = 0;
  if (std::strcmp(value, "4096") == 0) tile = 4096;
  else if (std::strcmp(value, "8192") == 0) tile = 8192;
  else VT_CHECK(std::strcmp(value, "0") == 0, "VT_XPU_REJECTION_ARGMAX_TILE must be 0/4096/8192");
  return (rows == 1 || rows == 4) && vocab == 248320 ? tile : 0;
}
// Internal A/B selector. Tile4096 passed exactness, ownership and paired
// serving gates; only the qualified real decode forms enter.
inline constexpr const char* kXpuTargetArgmaxDefaultTile = "4096";
inline int XpuTargetArgmaxTile(int64_t rows, int64_t vocab) {
  const char* value = std::getenv("VT_XPU_TARGET_ARGMAX_TILE");
  if (!value) value = kXpuTargetArgmaxDefaultTile;
  int tile = 0;
  if (std::strcmp(value, "4096") == 0) tile = 4096;
  else if (std::strcmp(value, "8192") == 0) tile = 8192;
  else VT_CHECK(std::strcmp(value, "0") == 0, "VT_XPU_TARGET_ARGMAX_TILE must be 0/4096/8192");
  return (rows == 1 || rows == 4) && vocab == 248320 ? tile : 0;
}
}  // namespace vt
