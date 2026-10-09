#include "xpu_test_helpers.h"
#include "vt/xpu/xpu_common.h"
#include <stdexcept>

TEST_CASE("XPU W8A8 workspaces: readers growth exceptions and queue retirement") {
  xpu_test::Queue first(vt::DeviceType::kXPU), second(vt::DeviceType::kXPU);
  for (bool preparation : {false, true}) {
    CAPTURE(preparation);
    auto lease = [&](vt::Queue& q, size_t bytes, const std::function<void(void*)>& launch) {
      return preparation ? vt::xpu::WithExl3W8A8Preparation(q, bytes, launch)
                         : vt::xpu::WithExl3W8A8Workspace(q, bytes, launch);
    };
    for (int step = 0; step < 12; ++step) {
      CAPTURE(step);
      const size_t bytes = step < 4 ? 65536 : 131072;
      xpu_test::Buffer a(first.q, vt::DType::kI8, {int64_t(bytes)});
      xpu_test::Buffer b(second.q, vt::DType::kI8, {int64_t(bytes)});
      auto submit = [&](vt::Queue& q, xpu_test::Buffer& out, int value, bool fail) {
        return lease(q, bytes, [&](void* storage) {
          auto& native = vt::xpu::NativeQueue(q);
          native.memset(storage, value, bytes);
          native.memcpy(out.tensor.data, storage, bytes);
          if (fail) throw std::runtime_error("lease callback failure");
        });
      };
      REQUIRE(submit(first.q, a, 17, false));
      // No caller synchronization/download between the two consumers. Alternate
      // same-queue and cross-queue reuse, including the first capacity growth.
      auto& next = step % 2 ? first.q : second.q;
      REQUIRE(submit(next, b, 93, false));
      auto check = [](const xpu_test::Buffer& out, unsigned char value) {
        const auto raw = out.download();
        CHECK(std::all_of(raw.begin(), raw.end(), [=](unsigned char x) { return x == value; }));
      };
      check(a, 17); check(b, 93);
      CHECK_THROWS_WITH_AS(submit(first.q, a, 41, true),
                           "lease callback failure", std::runtime_error);
      REQUIRE(submit(second.q, b, 67, false));
      check(a, 41); check(b, 67);
      {
        xpu_test::Queue retiring(vt::DeviceType::kXPU);
        REQUIRE(submit(retiring.q, a, 29, false));
      } // Retire the originating queue before another queue reuses the pool.
      REQUIRE(submit(second.q, b, 83, false));
      check(a, 29); check(b, 83);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}
