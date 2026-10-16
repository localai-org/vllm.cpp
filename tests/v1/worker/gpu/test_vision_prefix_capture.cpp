#include <doctest/doctest.h>

#include <vector>

#include "vllm/v1/worker/gpu/vision_prefix_capture.h"

TEST_CASE("vision prefix capture selects the actual MTP query column") {
  using vllm::detail::VisionPrefixQueryColumn;
  const std::vector<int32_t> wanted{1835, 8029, 271, 3074, 2702};
  const std::vector<int32_t> before{1835, 8029, 271, 3074};
  const std::vector<int32_t> query{3074, 2702, 13, 248044};
  CHECK(VisionPrefixQueryColumn(wanted, before, query, 3, 3, true) == 1);
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, query, 3, 3, false));
  CHECK(VisionPrefixQueryColumn(wanted, wanted, std::vector<int32_t>{2702, 13, 248044, 248045},
                               4, 3, false) == 0);
  CHECK(VisionPrefixQueryColumn(wanted, wanted, std::vector<int32_t>{2702}, 4, 0, false) == 0);
  for (int column = 0; column <= 3; ++column) {
    const std::vector<int32_t> packet{10, 11, 12, 13};
    const std::vector<int32_t> prefix(packet.begin(), packet.begin() + column + 1);
    CHECK(VisionPrefixQueryColumn(prefix, std::vector<int32_t>{10}, packet, 0, 3, true) == column);
  }
}

TEST_CASE("vision prefix capture rejects incompatible logical and physical prefixes") {
  using vllm::detail::VisionPrefixQueryColumn;
  const std::vector<int32_t> wanted{10, 11, 12}, before{10, 11}, query{11, 12, 13, 14};
  CHECK_FALSE(VisionPrefixQueryColumn({}, before, query, 1, 3, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, {}, query, 1, 3, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, query, -1, 3, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, query, 0, 3, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, query, 2, 3, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, query, 1, -1, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, query, 1, 4, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, std::vector<int32_t>{11}, 1, 3, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, std::vector<int32_t>{11}, 1, 0, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, std::vector<int32_t>{10, 11, 12, 13}, query, 3, 3, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, std::vector<int32_t>{10, 99}, query, 1, 3, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, std::vector<int32_t>{99, 12, 13, 14}, 1, 3, true));
  CHECK_FALSE(VisionPrefixQueryColumn(wanted, before, std::vector<int32_t>{11, 99, 13, 14}, 1, 3, true));
}

TEST_CASE("vision prefix capture requires accepted leading draft tokens") {
  using vllm::detail::VisionPrefixReachableGreedy;
  const std::vector<int32_t> query{3074, 2702, 13, 248044};
  const std::vector<int32_t> target{2702, 13, 248044, 248045};
  for (int column = 0; column <= 3; ++column)
    CHECK(VisionPrefixReachableGreedy(target, query, column));
  CHECK_FALSE(VisionPrefixReachableGreedy(std::vector<int32_t>{2702, 2542, 248044, 248045}, query, 2));
  CHECK_FALSE(VisionPrefixReachableGreedy(std::vector<int32_t>{99, 13, 248044, 248045}, query, 1));
  CHECK_FALSE(VisionPrefixReachableGreedy(target, query, -1));
  CHECK_FALSE(VisionPrefixReachableGreedy(target, query, 4));
  CHECK_FALSE(VisionPrefixReachableGreedy({}, query, 0));
  CHECK_FALSE(VisionPrefixReachableGreedy({}, {}, 0));
  CHECK_FALSE(VisionPrefixReachableGreedy(std::vector<int32_t>{11, 12, 13, 14, 15},
                                        std::vector<int32_t>{10, 11, 12, 13, 14}, 0));
}

TEST_CASE("Vision greedy preview does not assert actual sampler acceptance") {
  const std::vector<int32_t> query{3074, 2702, 13, 248046};
  CHECK(vllm::detail::VisionGreedyAcceptedDrafts(std::vector<int32_t>{2542, 13, 248044, 198}, query) == 0);
  CHECK(vllm::detail::VisionGreedyAcceptedDrafts(std::vector<int32_t>{2702, 2542, 248044, 198}, query) == 1);
  CHECK(vllm::detail::VisionGreedyAcceptedDrafts(std::vector<int32_t>{2702, 13, 248044, 198}, query) == 2);
  CHECK(vllm::detail::VisionGreedyAcceptedDrafts(std::vector<int32_t>{2702, 13, 248046, 198}, query) == 3);
  CHECK_FALSE(vllm::detail::VisionGreedyAcceptedDrafts(std::vector<int32_t>{}, query).has_value());
  CHECK_FALSE(vllm::detail::VisionGreedyAcceptedDrafts(std::vector<int32_t>{}, std::vector<int32_t>{}).has_value());
  CHECK_FALSE(vllm::detail::VisionGreedyAcceptedDrafts(std::vector<int32_t>{1,2,3,4,5},
      std::vector<int32_t>{1,2,3,4,5}).has_value());
  CHECK(vllm::detail::VisionGreedyAcceptedDrafts(std::vector<int32_t>{271}, std::vector<int32_t>{16327}) == 0);
}
