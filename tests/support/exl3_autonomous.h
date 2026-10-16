#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace exl3_test {

struct AutonomousResult {
  std::vector<int32_t> output_ids;
  std::string stop_reason;
  int forward_calls = 0;
};

// Global vocabulary order gives the lowest global ID on a logit tie.
inline int32_t GreedyToken(const std::vector<float>& logits) {
  if (logits.empty() || !std::all_of(logits.begin(), logits.end(),
                                   [](float x) { return std::isfinite(x); }))
    throw std::runtime_error("autonomous head requires nonempty finite logits");
  return static_cast<int32_t>(std::max_element(logits.begin(), logits.end()) - logits.begin());
}

// The caller owns model/cache resources and must reset them before each request.
// Forward receives only the real prompt or the last native-selected global ID;
// this driver has no continuation fixture or access to future reference tokens.
template <class Reset, class Forward>
AutonomousResult RunAutonomousC1(const std::vector<int32_t>& prompt, int output_limit,
                                const std::vector<int32_t>& eos_ids, int vocab_size,
                                Reset reset, Forward forward) {
  const auto valid_id = [vocab_size](int32_t id) { return id >= 0 && id < vocab_size; };
  if (prompt.empty() || output_limit < 0 || vocab_size <= 0 ||
      !std::all_of(prompt.begin(), prompt.end(), valid_id) ||
      !std::all_of(eos_ids.begin(), eos_ids.end(), valid_id))
    throw std::invalid_argument("invalid autonomous prompt, budget or vocabulary");
  reset();
  AutonomousResult result;
  result.stop_reason = "output_limit";
  for (int emitted = 0; emitted < output_limit; ++emitted) {
    const bool prefill = emitted == 0;
    std::vector<int32_t> ids = prefill ? prompt : std::vector<int32_t>{result.output_ids.back()};
    std::vector<int32_t> positions(ids.size());
    std::iota(positions.begin(), positions.end(),
              prefill ? 0 : static_cast<int32_t>(prompt.size() + emitted - 1));
    const int32_t chosen = forward(ids, positions, prefill);
    ++result.forward_calls;
    if (!valid_id(chosen)) throw std::runtime_error("native head returned an invalid global token ID");
    result.output_ids.push_back(chosen);
    if (std::find(eos_ids.begin(), eos_ids.end(), chosen) != eos_ids.end()) {
      result.stop_reason = "eos";
      break;
    }
  }
  return result;
}

}  // namespace exl3_test
