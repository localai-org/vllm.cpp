#include <doctest/doctest.h>
#include "vllm/v1/worker/gpu/async_output.h"
#include "vt/xpu.h"
#include "vt/ops.h"
#include <array>

TEST_CASE("PR11 XPU async readback retains request ordering, padding and reusable slots") {
  const vt::Device device{vt::DeviceType::kXPU, 0};
  auto main = vt::CreateQueue(device), copy = vt::CreateQueue(device);
  auto& backend = vt::GetBackend(device);
  const auto before = vt::xpu::GetMemoryInfo();
  {
    vllm::v1::AsyncOutputPool pool(device, 4, 2);
    const auto baseline = vt::xpu::GetMemoryInfo();
    vllm::v1::AsyncOutputSlot* last_slot = nullptr;
    auto make = [&](std::vector<std::string> requests, std::array<int64_t, 4> tokens, std::vector<int32_t> discard) {
      auto* slot = pool.Acquire();
      last_slot = slot;
      backend.Copy(main, slot->device_sampled_ids, tokens.data(), requests.size() * sizeof(int64_t));
      vllm::v1::ModelRunnerOutput skeleton;
      skeleton.req_ids = requests;
      for (size_t i = 0; i < requests.size(); ++i) skeleton.req_id_to_index[requests[i]] = int(i);
      return std::make_unique<vllm::v1::AsyncGPUModelRunnerOutput>(
          std::move(skeleton), device, pool, slot, int(requests.size()), main, copy, std::move(discard));
    };
    for (int step = 0; step < 16; ++step) {
      auto first = make({"a", "b", "c", "d"}, {11, 22, 33, 44}, {1});
      // A later batch has finished b, condensed the rows and admitted e while
      // the previous batch's transfer still owns its original ordering.
      auto second = make({"d", "a", "e"}, {45, 12, 55, -1}, {});
      const auto later = second->get_output(), earlier = first->get_output();
      CHECK(earlier.req_ids == std::vector<std::string>{"a", "b", "c", "d"});
      CHECK(earlier.sampled_token_ids == std::vector<std::vector<int32_t>>{{11}, {}, {33}, {44}});
      CHECK(later.req_ids == std::vector<std::string>{"d", "a", "e"});
      CHECK(later.sampled_token_ids == std::vector<std::vector<int32_t>>{{45}, {12}, {55}});
      CHECK_THROWS(first->get_output()); // exactly one consume, no duplicate gather
      CHECK(pool.num_slots() == 2);
      CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == baseline.allocated_bytes);
      CHECK(vt::xpu::GetMemoryInfo().pinned_bytes == baseline.pinned_bytes);
    }
    auto empty = make({}, {}, {}); CHECK(empty->get_output().sampled_token_ids.empty());
    auto cancelled = make({"x"}, {91, 0, 0, 0}, {}); cancelled.reset();
    CHECK(backend.QueryEvent(last_slot->ready_event));
    auto resumed = make({"x"}, {92, 0, 0, 0}, {});
    CHECK(resumed->get_output().sampled_token_ids == std::vector<std::vector<int32_t>>{{92}});
  }
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == before.allocated_bytes);
  CHECK(vt::xpu::GetMemoryInfo().pinned_bytes == before.pinned_bytes);
  vt::DestroyQueue(copy); vt::DestroyQueue(main);
}
