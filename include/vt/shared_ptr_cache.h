#pragma once

#include <atomic>
#include <memory>
#include <utility>

namespace vt {
// A copyable owner slot with atomic publication. Readers must retain Load()'s
// snapshot while using its payload; there is deliberately no raw-pointer API.
// Copies share the published owner, but subsequent publication is independent.
template <class T>
class SharedPtrCache {
 public:
  SharedPtrCache() = default;
  SharedPtrCache(const SharedPtrCache& other) : value_(other.Load()) {}
  SharedPtrCache& operator=(const SharedPtrCache& other) {
    if (this != &other) Store(other.Load());
    return *this;
  }
  SharedPtrCache(SharedPtrCache&& other) noexcept
      : value_(other.value_.exchange({})) {}
  SharedPtrCache& operator=(SharedPtrCache&& other) noexcept {
    if (this != &other) Store(other.value_.exchange({}));
    return *this;
  }
  std::shared_ptr<T> Load() const { return value_.load(); }
  void Store(std::shared_ptr<T> value) { value_.store(std::move(value)); }
  void Reset() { Store({}); }

 private:
  std::atomic<std::shared_ptr<T>> value_;
};
}  // namespace vt
