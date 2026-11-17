#include <doctest/doctest.h>
#include "vt/xpu_memory_policy.h"
#include <limits>

TEST_CASE("XPU immutable table admission: exhausted accounting cannot wrap") {
  CHECK(vt::XpuDeviceAllocationFits(100, 80, 10, 10));
  CHECK_FALSE(vt::XpuDeviceAllocationFits(100, 80, 10, 11));
  CHECK(vt::XpuDeviceAllocationFits(100, 100, 0, 0));
  CHECK_FALSE(vt::XpuDeviceAllocationFits(100, 100, 0, 1));
  CHECK_FALSE(vt::XpuDeviceAllocationFits(100, 101, 0, 1));
  CHECK_FALSE(vt::XpuDeviceAllocationFits(100, 80, 21, 1));
  CHECK_FALSE(vt::XpuDeviceAllocationFits(100, 0, 101, 1));
  const auto maximum = std::numeric_limits<std::size_t>::max();
  CHECK_FALSE(vt::XpuDeviceAllocationFits(100, 80, maximum, 1));
  CHECK_FALSE(vt::XpuDeviceAllocationFits(100, maximum, 80, 1));
  CHECK_FALSE(vt::XpuDeviceAllocationFits(maximum, maximum, 1, 1));
  CHECK(vt::XpuDeviceAllocationFits(maximum, maximum - 131072, 0, 131072));
  CHECK_FALSE(vt::XpuDeviceAllocationFits(maximum, maximum - 131072, 1, 131072));
}
