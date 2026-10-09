#include <doctest/doctest.h>
#include "vt/exl3_w8a8_panel_plan.h"
#include "vt/exl3_grouped.h"
#include <cstdlib>
#include <optional>
#include <string>

TEST_CASE("EXL3 W8A8 model selection: exact internal widths and invalid admission") {
  constexpr auto key = "VT_XPU_EXL3_W8A8_PANEL_COLUMNS";
  struct Restore {
    const char* key;
    std::optional<std::string> old;
    ~Restore() { if (old) setenv(key, old->c_str(), 1); else unsetenv(key); }
  } restore{key, std::getenv(key) ? std::optional<std::string>(std::getenv(key)) : std::nullopt};
  REQUIRE(unsetenv(key) == 0);
  CHECK(vt::Exl3W8A8ModelPanelColumns() == 1024);
  for (const auto width : {"128", "1024", "2048"}) {
    REQUIRE(setenv(key, width, 1) == 0);
    CHECK(vt::Exl3W8A8ModelPanelColumns() == std::atoi(width));
  }
  for (const auto invalid : {"", "256", "1024junk", " 2048", "-128"}) {
    REQUIRE(setenv(key, invalid, 1) == 0);
    CHECK_THROWS_WITH_AS(vt::Exl3W8A8ModelPanelColumns(), doctest::Contains("must be128/1024/2048"),
                         std::runtime_error);
  }
}

TEST_CASE("EXL3 W8A8 panels: nonmonotonic groups retain tails and exact coverage") {
  std::vector<int32_t> map(9, 0);
  map.insert(map.end(), 6, 1);
  map.push_back(0);
  for (int width : {128, 1024, 2048}) {
    const auto panels = vt::PlanExl3W8A8Panels(map, 2, width);
    int next = 0;
    for (const auto& p : panels) {
      CHECK(p.first_column == next);
      CHECK(p.columns > 0);
      CHECK(p.columns <= width);
      CHECK(p.columns % 128 == 0);
      for (int c = p.first_column; c < p.first_column + p.columns; c += 128)
        CHECK(map[c / 128] == p.source_group);
      next += p.columns;
    }
    CHECK(next == 2048);
    CHECK(panels.size() == (width == 128 ? 16 : width == 1024 ? 4 : 3));
  }
  const auto tail = vt::PlanExl3W8A8Panels(map, 2, 1024);
  CHECK(tail[0].columns == 1024);
  CHECK(tail[1].columns == 128);
  CHECK(tail[2].columns == 768);
  CHECK(tail[3].first_column == 1920);
  CHECK(tail[3].source_group == 0);
}

TEST_CASE("EXL3 W8A8 panels: real gate-up geometry and bounded reusable capacity") {
  std::vector<int32_t> map(136, 0);  // each17408-column source is independent
  map.insert(map.end(), 136, 1);
  for (int width : {128, 1024, 2048}) {
    const auto panels = vt::PlanExl3W8A8Panels(map, 2, width);
    CHECK(panels.size() == (width == 128 ? 272 : width == 1024 ? 34 : 18));
    for (const auto& p : panels) {
      CHECK(p.first_column / 17408 == p.source_group);
      CHECK((p.first_column + p.columns - 1) / 17408 == p.source_group);
    }
  }
  const auto full = vt::PlanExl3W8A8PanelCapacity(17408, 34816, 2048);
  CHECK(full.columns == 2048);
  CHECK(full.bytes == size_t(34) * 1024 * 1024);
  CHECK(vt::PlanExl3W8A8PanelCapacity(17408, 34816, 128).bytes == size_t(2228224));
  const auto short_panel = vt::PlanExl3W8A8PanelCapacity(5120, 128, 2048);
  CHECK(short_panel.columns == 128);
  CHECK(short_panel.bytes == size_t(5120 * 128));
  // Public legacy planner remains an unchanged128-column control until the
  // wider kernel/allocation path is qualified on real GPU operands.
  CHECK(vt::PlanExl3W8A8(256, 17408, 34816, 2, 4).weight_panel_bytes ==
        vt::PlanExl3W8A8PanelCapacity(17408, 34816, 128).bytes);
  for (int width : {1024, 2048}) {
    const auto p = vt::PlanExl3W8A8(1600, 17408, 34816, 2, 4, width);
    CHECK(p.padded_rows == 1792);
    CHECK(p.weight_panel_columns == width);
    CHECK(p.weight_panel_bytes == size_t(17408) * width);
    CHECK(p.workspace_bytes == vt::PlanExl3W8A8(1600, 17408, 34816, 2, 4).workspace_bytes);
  }
}

TEST_CASE("EXL3 W8A8 panels: invalid source and unsafe geometry are refused") {
  CHECK_THROWS(vt::PlanExl3W8A8Panels({}, 2, 1024));
  const std::vector<int32_t> negative{0, -1}, invalid{0, 2}, valid{0};
  CHECK_THROWS(vt::PlanExl3W8A8Panels(negative, 2, 1024));
  CHECK_THROWS(vt::PlanExl3W8A8Panels(invalid, 2, 1024));
  CHECK_THROWS(vt::PlanExl3W8A8Panels(valid, 0, 1024));
  CHECK_THROWS(vt::PlanExl3W8A8Panels(valid, 1, 256));
  CHECK_THROWS(vt::PlanExl3W8A8PanelCapacity(5120, 128, 4096));
  CHECK_THROWS(vt::PlanExl3W8A8PanelCapacity(5130, 128, 1024));
  CHECK_THROWS(vt::PlanExl3W8A8PanelCapacity(5120, 64, 1024));
  CHECK_THROWS(vt::PlanExl3W8A8PanelCapacity(133248, 128, 1024));
}
