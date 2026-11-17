#pragma once

// Private, opt-in diagnostic selection. A proposed prefix is only a candidate
// here; full target logits and the actual rejection result must independently
// prove that its preceding draft tokens were accepted.
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>

namespace vllm::detail {

inline std::optional<int> VisionPrefixQueryColumn(
    std::span<const int32_t> wanted, std::span<const int32_t> committed,
    std::span<const int32_t> query, int computed, int drafts,
    bool allow_packet_prefix) {
  if (wanted.empty() || committed.empty() || drafts < 0 || drafts > 3 ||
      computed < 0 || static_cast<std::size_t>(computed) + 1 != committed.size() ||
      committed.size() > wanted.size() || query.size() != static_cast<std::size_t>(drafts + 1) ||
      !std::equal(committed.begin(), committed.end(), wanted.begin())) return std::nullopt;
  const std::size_t column = wanted.size() - committed.size();
  if (column > static_cast<std::size_t>(drafts) || (column != 0 && !allow_packet_prefix))
    return std::nullopt;
  for (std::size_t i = 0; i <= column; ++i)
    if (query[i] != wanted[static_cast<std::size_t>(computed) + i]) return std::nullopt;
  return static_cast<int>(column);
}

// Prediction only: actual sampler acceptance remains the authoritative result.
inline std::optional<int> VisionGreedyAcceptedDrafts(std::span<const int32_t> argmax_ids,
                                                    std::span<const int32_t> query) {
  if (query.empty() || query.size() > 4 || argmax_ids.size() != query.size()) return std::nullopt;
  int accepted = 0;
  while (static_cast<std::size_t>(accepted + 1) < query.size() &&
         argmax_ids[static_cast<std::size_t>(accepted)] == query[static_cast<std::size_t>(accepted) + 1])
    ++accepted;
  return accepted;
}

inline bool VisionPrefixReachableGreedy(std::span<const int32_t> argmax_ids,
                                       std::span<const int32_t> query, int column) {
  if (column < 0 || query.empty() || query.size() > 4 || argmax_ids.size() != query.size() ||
      static_cast<std::size_t>(column) >= query.size()) return false;
  for (int i = 0; i < column; ++i)
    if (argmax_ids[static_cast<std::size_t>(i)] != query[static_cast<std::size_t>(i) + 1]) return false;
  return true;
}

}  // namespace vllm::detail
