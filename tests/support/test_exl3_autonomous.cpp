#include <doctest/doctest.h>

#include <limits>

#include "support/exl3_autonomous.h"

TEST_CASE("EXL3 autonomous C1 feeds native head choices and resets request state") {
  int state = 99, resets = 0;
  std::vector<std::vector<int32_t>> inputs, positions;
  const auto run = [&] {
    return exl3_test::RunAutonomousC1({1, 3, 5}, 16, {6}, 8,
        [&] { state = 0; ++resets; inputs.clear(); positions.clear(); },
        [&](const auto& ids, const auto& pos, bool prefill) {
          CHECK(prefill == (state == 0));
          inputs.push_back(ids); positions.push_back(pos);
          // Deliberately unrelated to the prompt IDs. The next head also
          // depends on feedback, so fixture-driven continuation cannot pass.
          std::vector<float> logits(8, -10);
          const int chosen = state == 0 ? 4 : ids[0] == 4 ? 2 : 6;
          logits[chosen] = 10;
          ++state;
          return exl3_test::GreedyToken(logits);
        });
  };
  const auto first = run();
  CHECK(first.output_ids == std::vector<int32_t>{4, 2, 6});
  CHECK(inputs == std::vector<std::vector<int32_t>>{{1, 3, 5}, {4}, {2}});
  CHECK(positions == std::vector<std::vector<int32_t>>{{0, 1, 2}, {3}, {4}});
  CHECK(first.stop_reason == "eos");
  CHECK(first.forward_calls == 3);
  const auto second = run();
  CHECK(resets == 2);
  CHECK(second.output_ids == first.output_ids);
  CHECK(second.forward_calls == first.forward_calls);
}

TEST_CASE("EXL3 autonomous C1 counts EOS and enforces exact output budget") {
  int calls = 0, resets = 0;
  const auto run = [&](int limit, const std::vector<int32_t>& eos) {
    return exl3_test::RunAutonomousC1({1}, limit, eos, 8,
        [&] { ++resets; calls = 0; },
        [&](const auto&, const auto&, bool) { ++calls; return 3; });
  };
  const auto zero = run(0, {});
  CHECK(zero.output_ids.empty()); CHECK(calls == 0); CHECK(resets == 1);
  const auto one = run(1, {});
  CHECK(one.output_ids == std::vector<int32_t>{3});
  CHECK(one.stop_reason == "output_limit"); CHECK(calls == 1);
  const auto capped = run(4, {});
  CHECK(capped.output_ids.size() == 4); CHECK(calls == 4);
  const auto eos = run(16, {3, 6});
  CHECK(eos.output_ids == std::vector<int32_t>{3});
  CHECK(eos.stop_reason == "eos"); CHECK(calls == 1);
}

TEST_CASE("EXL3 autonomous C1 rejects invalid IDs and nonfinite heads") {
  int calls = 0;
  const auto reset = [&] { ++calls; };
  const auto forward = [](const auto&, const auto&, bool) { return 8; };
  CHECK_THROWS_AS(exl3_test::RunAutonomousC1({}, 1, {}, 8, reset, forward), std::invalid_argument);
  CHECK_THROWS_AS(exl3_test::RunAutonomousC1({8}, 1, {}, 8, reset, forward), std::invalid_argument);
  CHECK_THROWS_AS(exl3_test::RunAutonomousC1({1}, -1, {}, 8, reset, forward), std::invalid_argument);
  CHECK_THROWS_AS(exl3_test::RunAutonomousC1({1}, 1, {-1}, 8, reset, forward), std::invalid_argument);
  CHECK(calls == 0);
  CHECK_THROWS_AS(exl3_test::RunAutonomousC1({1}, 1, {}, 8, reset, forward), std::runtime_error);
  CHECK(exl3_test::GreedyToken({-2, 5, 5, 1}) == 1);
  CHECK_THROWS_AS(exl3_test::GreedyToken({}), std::runtime_error);
  CHECK_THROWS_AS(exl3_test::GreedyToken({0, std::numeric_limits<float>::quiet_NaN()}), std::runtime_error);
  CHECK_THROWS_AS(exl3_test::GreedyToken({std::numeric_limits<float>::infinity(), 0}), std::runtime_error);
}
