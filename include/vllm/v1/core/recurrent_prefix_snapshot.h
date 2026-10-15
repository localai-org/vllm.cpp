#pragma once
#include <algorithm>
#include <cstdint>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace vllm::v1 {
// Shared scheduler/runner metadata for a bounded recurrent prefix cache.
// The runner owns the GPU rows. A row becomes visible only after its copy has
// completed; admitted readers pin it until their private running state is ready.
// Published snapshots are never used as the mutable decode state.
class RecurrentPrefixSnapshotIndex {
 public:
  struct Snapshot { int slot; std::string hash; int tokens; uint64_t generation; };
  RecurrentPrefixSnapshotIndex(int capacity, int block_tokens)
      : entries_(CheckedCapacity(capacity)), block_tokens_(block_tokens) {
    if (block_tokens <= 0) throw std::invalid_argument("prefix snapshot block size must be positive");
  }
  int capacity() const { return static_cast<int>(entries_.size()); }
  int block_tokens() const { return block_tokens_; }
  bool Contains(const std::string& hash) const {
    std::lock_guard lock(mutex_); return Find(hash) >= 0;
  }
  bool Pin(const std::string& request, const std::string& hash) {
    std::lock_guard lock(mutex_);
    const int slot = Find(hash);
    if (slot < 0) return false;
    const auto old = readers_.find(request);
    if (old != readers_.end() && old->second == slot) return true;
    ReleaseLocked(request);
    readers_.emplace(request, slot); ++entries_[slot].readers; entries_[slot].used = ++clock_;
    return true;
  }
  std::optional<Snapshot> Pinned(const std::string& request) const {
    std::lock_guard lock(mutex_);
    const auto it = readers_.find(request);
    if (it == readers_.end()) return std::nullopt;
    const auto& e = entries_[it->second];
    return Snapshot{it->second, e.hash, e.tokens, e.used};
  }
  void Release(const std::string& request) {
    std::lock_guard lock(mutex_); ReleaseLocked(request);
  }
  // Reserve removes an evicted entry from lookups BEFORE its GPU row changes.
  // A cache miss is preferable to modifying a snapshot with pending readers.
  std::optional<Snapshot> Reserve(const std::string& hash, int tokens) {
    std::lock_guard lock(mutex_);
    if (tokens <= 0 || tokens % block_tokens_ != 0 || hash.empty())
      throw std::invalid_argument("prefix snapshot must end at a hashed token boundary");
    for (const auto& e : entries_) if (e.hash == hash) return std::nullopt;
    int slot = -1;
    for (int i = 0; i < capacity(); ++i) {
      const auto& e = entries_[i];
      if (e.readers || e.writing) continue;
      if (slot < 0 || e.used < entries_[slot].used) slot = i;
    }
    if (slot < 0) return std::nullopt;
    entries_[slot] = Entry{hash, tokens, 0, ++clock_, false, true};
    return Snapshot{slot, hash, tokens, entries_[slot].used};
  }
  void Publish(const Snapshot& snapshot) {
    std::lock_guard lock(mutex_);
    auto& e = CheckedWrite(snapshot);
    e.writing = false; e.ready = true;
  }
  void Abort(const Snapshot& snapshot) {
    std::lock_guard lock(mutex_); CheckedWrite(snapshot) = Entry{};
  }
  bool Reset() { return Reset([] { return true; }); }
  template<class ResetPages>
  bool Reset(ResetPages reset_pages) {
    std::lock_guard lock(mutex_);
    if (!readers_.empty() || std::any_of(entries_.begin(), entries_.end(), [](const Entry& e) { return e.writing; })) return false;
    // Coordinate the KV-page reset under the same lock: a refused reset must
    // leave both the published snapshots and their attention pages intact.
    if (!reset_pages()) return false;
    std::fill(entries_.begin(), entries_.end(), Entry{}); return true;
  }
 private:
  struct Entry {
    std::string hash;
    int tokens = 0, readers = 0;
    uint64_t used = 0;
    bool ready = false, writing = false;
  };
  static size_t CheckedCapacity(int value) {
    if (value <= 0) throw std::invalid_argument("prefix snapshot capacity must be positive");
    return static_cast<size_t>(value);
  }
  int Find(const std::string& hash) const {
    for (int i = 0; i < capacity(); ++i) if (entries_[i].ready && entries_[i].hash == hash) return i;
    return -1;
  }
  void ReleaseLocked(const std::string& request) {
    const auto it = readers_.find(request);
    if (it != readers_.end()) { --entries_[it->second].readers; readers_.erase(it); }
  }
  Entry& CheckedWrite(const Snapshot& s) {
    if (s.slot < 0 || s.slot >= capacity()) throw std::logic_error("invalid prefix snapshot slot");
    auto& e = entries_[s.slot];
    if (!e.writing || e.hash != s.hash || e.tokens != s.tokens || e.used != s.generation)
      throw std::logic_error("stale prefix snapshot publication");
    return e;
  }
  mutable std::mutex mutex_;
  std::vector<Entry> entries_;
  std::unordered_map<std::string, int> readers_;
  int block_tokens_;
  uint64_t clock_ = 0;
};
}  // namespace vllm::v1
