#include <doctest/doctest.h>
#include "vt/xpu_argmax_plan.h"
#include <initializer_list>
#include <limits>
#include <string>

TEST_CASE("XPU tiled argmax plan: bounded real shapes and tile tails") {
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(1, 248320, 4096) == 976);
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(4, 248320, 4096) == 3904);
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(1, 248320, 8192) == 496);
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(4, 248320, 8192) == 1984);
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(1, 65536, 4096) == 256);
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(1, 4095, 4096) == 16);
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(1, 4096, 4096) == 16);
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(1, 4097, 4096) == 32);
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(0, 248320, 4096) == 0);
  CHECK(vt::XpuTiledArgmaxWorkspaceBytes(1048576, 4096, 4096) == 16 * 1024 * 1024);
  CHECK_THROWS(vt::XpuTiledArgmaxWorkspaceBytes(1048577, 4096, 4096));
  for (int tile : {0, 128, 4095, 16384}) CHECK_THROWS(vt::XpuTiledArgmaxWorkspaceBytes(1, 248320, tile));
  CHECK_THROWS(vt::XpuTiledArgmaxWorkspaceBytes(-1, 248320, 4096));
  CHECK_THROWS(vt::XpuTiledArgmaxWorkspaceBytes(1, 0, 4096));
  CHECK_THROWS(vt::XpuTiledArgmaxWorkspaceBytes(1, -1, 4096));
  const auto maximum = std::numeric_limits<int64_t>::max();
  CHECK_THROWS(vt::XpuTiledArgmaxWorkspaceBytes(maximum, 248320, 4096));
  CHECK_THROWS(vt::XpuTiledArgmaxWorkspaceBytes(1, maximum, 8192));
}

TEST_CASE("XPU tiled target policy: qualified default shape guard and invalid control") {
  const char* value = std::getenv("VT_XPU_TARGET_ARGMAX_TILE");
  const bool had_value = value != nullptr;
  const std::string saved = value ? value : "";
  unsetenv("VT_XPU_TARGET_ARGMAX_TILE");
  CHECK(vt::XpuTargetArgmaxTile(1, 248320) == 4096);
  for (const char* tile : {"4096", "8192"}) {
    setenv("VT_XPU_TARGET_ARGMAX_TILE", tile, 1);
    CHECK(vt::XpuTargetArgmaxTile(1, 248320) == std::atoi(tile));
    CHECK(vt::XpuTargetArgmaxTile(4, 248320) == std::atoi(tile));
    for (int rows : {0, 2, 3, 16, 4096}) CHECK(vt::XpuTargetArgmaxTile(rows, 248320) == 0);
    for (int vocab : {1, 4096, 8192, 65536, 248319, 248321}) CHECK(vt::XpuTargetArgmaxTile(1, vocab) == 0);
  }
  setenv("VT_XPU_TARGET_ARGMAX_TILE", "0", 1);
  CHECK(vt::XpuTargetArgmaxTile(1, 248320) == 0);
  setenv("VT_XPU_TARGET_ARGMAX_TILE", "1", 1);
  CHECK_THROWS(vt::XpuTargetArgmaxTile(1, 248320));
  if (had_value) setenv("VT_XPU_TARGET_ARGMAX_TILE", saved.c_str(), 1);
  else unsetenv("VT_XPU_TARGET_ARGMAX_TILE");
}

TEST_CASE("XPU mapped selection: checked composite owner layout and opt-in policy") {
  for (int tile : {4096, 8192}) for (int rows : {1, 4}) {
    const auto p = vt::XpuMappedSelectionBuffers(rows, 65536, tile);
    CHECK(p.id_bytes == size_t(rows) * 4);
    CHECK(p.scratch_offset == 64);
    CHECK(p.scratch_bytes == size_t(rows) * (65536 / tile) * 16);
    CHECK(p.total_bytes == p.scratch_offset + p.scratch_bytes);
    CHECK(p.scratch_offset % 16 == 0);
  }
  const auto empty = vt::XpuMappedSelectionBuffers(0, 65536, 4096);
  CHECK(empty.total_bytes == 0);
  CHECK_THROWS(vt::XpuMappedSelectionBuffers(-1, 65536, 4096));
  CHECK_THROWS(vt::XpuMappedSelectionBuffers(std::numeric_limits<int64_t>::max(), 65536, 4096));
  const char* old = std::getenv("VT_XPU_MAPPED_ARGMAX_TILE");
  const bool had = old != nullptr; const std::string saved = old ? old : "";
  unsetenv("VT_XPU_MAPPED_ARGMAX_TILE");
  CHECK(vt::XpuMappedArgmaxTile(1, 65536, 248320) == 0);
  for (const char* tile : {"4096", "8192"}) {
    setenv("VT_XPU_MAPPED_ARGMAX_TILE", tile, 1);
    CHECK(vt::XpuMappedArgmaxTile(1, 65536, 248320) == std::atoi(tile));
    CHECK(vt::XpuMappedArgmaxTile(4, 65536, 248320) == std::atoi(tile));
    CHECK(vt::XpuMappedArgmaxTile(2, 65536, 248320) == 0);
    CHECK(vt::XpuMappedArgmaxTile(1, 65535, 248320) == 0);
    CHECK(vt::XpuMappedArgmaxTile(1, 65536, 248319) == 0);
  }
  setenv("VT_XPU_MAPPED_ARGMAX_TILE", "1", 1);
  CHECK_THROWS(vt::XpuMappedArgmaxTile(1, 65536, 248320));
  if (had) setenv("VT_XPU_MAPPED_ARGMAX_TILE", saved.c_str(), 1);
  else unsetenv("VT_XPU_MAPPED_ARGMAX_TILE");
}

TEST_CASE("XPU rejection selection: independent opt-in policy and owner bound") {
  const char* old = std::getenv("VT_XPU_REJECTION_ARGMAX_TILE");
  const bool had = old != nullptr; const std::string saved = old ? old : "";
  unsetenv("VT_XPU_REJECTION_ARGMAX_TILE");
  CHECK(vt::XpuRejectionArgmaxTile(4,248320) == 0);
  for (const char* tile : {"4096","8192"}) {
    setenv("VT_XPU_REJECTION_ARGMAX_TILE",tile,1);
    for (int rows : {1,4}) CHECK(vt::XpuRejectionArgmaxTile(rows,248320) == std::atoi(tile));
    for (int rows : {0,2,3,10,4096}) CHECK(vt::XpuRejectionArgmaxTile(rows,248320) == 0);
    CHECK(vt::XpuRejectionArgmaxTile(4,65536) == 0);
  }
  CHECK(vt::XpuMappedSelectionBuffers(1,248320,4096).total_bytes - sizeof(int32_t) == 1036);
  CHECK(vt::XpuMappedSelectionBuffers(4,248320,4096).total_bytes - 4*sizeof(int32_t) == 3952);
  setenv("VT_XPU_REJECTION_ARGMAX_TILE","0",1);
  CHECK(vt::XpuRejectionArgmaxTile(4,248320) == 0);
  setenv("VT_XPU_REJECTION_ARGMAX_TILE","1",1);
  CHECK_THROWS(vt::XpuRejectionArgmaxTile(4,248320));
  if (had) setenv("VT_XPU_REJECTION_ARGMAX_TILE",saved.c_str(),1);
  else unsetenv("VT_XPU_REJECTION_ARGMAX_TILE");
}

TEST_CASE("XPU compact logprobs0: count workspace and independent opt-in policy") {
  CHECK(vt::XpuCompactLogprobsWorkspaceBytes(1,248320)==244);
  CHECK(vt::XpuCompactLogprobsWorkspaceBytes(4,248320)==976);
  CHECK(vt::XpuCompactLogprobsWorkspaceBytes(1,4097)==8);
  CHECK(vt::XpuCompactLogprobsWorkspaceBytes(0,248320)==0);
  CHECK_THROWS(vt::XpuCompactLogprobsWorkspaceBytes(-1,248320));
  CHECK_THROWS(vt::XpuCompactLogprobsWorkspaceBytes(std::numeric_limits<int64_t>::max(),248320));
  const char* old=std::getenv("VT_XPU_COMPACT_LOGPROBS0");
  const bool had=old!=nullptr;const std::string saved=old?old:"";
  unsetenv("VT_XPU_COMPACT_LOGPROBS0");CHECK_FALSE(vt::XpuCompactLogprobs0Enabled(1,248320));
  setenv("VT_XPU_COMPACT_LOGPROBS0","1",1);
  CHECK(vt::XpuCompactLogprobs0Enabled(1,248320));CHECK(vt::XpuCompactLogprobs0Enabled(4,248320));
  for(int rows:{0,2,3,8,4096})CHECK_FALSE(vt::XpuCompactLogprobs0Enabled(rows,248320));
  CHECK_FALSE(vt::XpuCompactLogprobs0Enabled(1,65536));
  setenv("VT_XPU_COMPACT_LOGPROBS0","0",1);CHECK_FALSE(vt::XpuCompactLogprobs0Enabled(1,248320));
  setenv("VT_XPU_COMPACT_LOGPROBS0","2",1);CHECK_THROWS(vt::XpuCompactLogprobs0Enabled(1,248320));
  if(had)setenv("VT_XPU_COMPACT_LOGPROBS0",saved.c_str(),1);else unsetenv("VT_XPU_COMPACT_LOGPROBS0");
}
