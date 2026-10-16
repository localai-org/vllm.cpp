#include <doctest/doctest.h>
#include <future>
#include <vector>
#include "vt/shared_ptr_cache.h"

TEST_CASE("SharedPtrCache: copy and move preserve independent owning slots") {
  vt::SharedPtrCache<const int> cache;
  CHECK_FALSE(cache.Load());
  auto value = std::make_shared<const int>(17);
  std::weak_ptr<const int> lifetime = value;
  cache.Store(value);
  auto copied = cache;
  vt::SharedPtrCache<const int> assigned;
  assigned = cache;
  value.reset(); cache.Reset();
  CHECK_FALSE(lifetime.expired());
  CHECK(*copied.Load() == 17);
  CHECK(copied.Load() == assigned.Load());
  std::vector<vt::SharedPtrCache<const int>> slots;
  slots.push_back(std::move(copied));
  CHECK_FALSE(copied.Load());
  slots.resize(8);  // Owning weight containers relocate these slots.
  vt::SharedPtrCache<const int> moved;
  moved = std::move(slots[0]);
  CHECK_FALSE(slots[0].Load());
  CHECK(moved.Load() == assigned.Load());
  auto& self = moved;
  moved = self; moved = std::move(self);
  CHECK(*moved.Load() == 17);
  assigned.Reset(); moved.Reset();
  CHECK(lifetime.expired());
}

TEST_CASE("SharedPtrCache: concurrent replacement retains the reader generation") {
  vt::SharedPtrCache<const int> cache;
  auto first = std::make_shared<const int>(41);
  std::weak_ptr<const int> lifetime = first;
  cache.Store(first); first.reset();
  std::promise<void> acquired, release;
  auto ready = acquired.get_future();
  auto released = release.get_future();
  auto consumer = std::async(std::launch::async, [&] {
    const auto snapshot = cache.Load();
    acquired.set_value();
    released.wait();
    return *snapshot;
  });
  ready.wait();
  cache.Store(std::make_shared<const int>(43));
  CHECK(*cache.Load() == 43);
  CHECK_FALSE(lifetime.expired());
  cache.Reset();
  CHECK_FALSE(lifetime.expired());
  release.set_value();
  CHECK(consumer.get() == 41);
  CHECK(lifetime.expired());
}
