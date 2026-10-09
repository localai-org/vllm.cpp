#include <doctest/doctest.h>
#include "vllm/v1/core/recurrent_prefix_snapshot.h"

TEST_CASE("PR11 recurrent snapshots publish only completed copies and pin pending readers") {
  vllm::v1::RecurrentPrefixSnapshotIndex cache(2, 16);
  auto a = cache.Reserve("a", 32); REQUIRE(a.has_value());
  CHECK_FALSE(cache.Contains("a")); CHECK_FALSE(cache.Pin("r", "a"));
  cache.Publish(*a); REQUIRE(cache.Contains("a")); REQUIRE(cache.Pin("r", "a"));
  REQUIRE(cache.Pin("s", "a")); // concurrent requests share an immutable snapshot
  CHECK(cache.Pinned("r")->slot == a->slot);
  CHECK(cache.Pinned("r")->tokens == 32);
  CHECK_FALSE(cache.Reserve("a", 32).has_value());
  auto b = cache.Reserve("b", 64); REQUIRE(b.has_value()); cache.Publish(*b);
  REQUIRE(cache.Pin("t", "b"));
  CHECK_FALSE(cache.Reserve("c", 48).has_value()); // bounded, all rows pinned
  CHECK_FALSE(cache.Reset());
  cache.Release("r"); CHECK_FALSE(cache.Reserve("c", 48).has_value());
  cache.Release("s");
  auto c = cache.Reserve("c", 48); REQUIRE(c.has_value());
  CHECK(c->slot == a->slot); CHECK_FALSE(cache.Contains("a"));
  CHECK(cache.Contains("b")); CHECK_FALSE(cache.Contains("c"));
  cache.Abort(*c); CHECK_FALSE(cache.Contains("c"));
  auto retry = cache.Reserve("c", 48); REQUIRE(retry.has_value());
  CHECK_THROWS(cache.Publish(*c));
  cache.Abort(*retry);
  cache.Release("t"); REQUIRE(cache.Reset());
  CHECK_FALSE(cache.Contains("b")); CHECK_FALSE(cache.Pinned("t").has_value());
}
TEST_CASE("PR11 prefix snapshot cancellation and repeated admission release one pin") {
  vllm::v1::RecurrentPrefixSnapshotIndex cache(1, 16);
  auto a = cache.Reserve("a", 16); cache.Publish(*a);
  REQUIRE(cache.Pin("r", "a")); REQUIRE(cache.Pin("r", "a"));
  cache.Release("r"); cache.Release("r");
  auto b = cache.Reserve("b", 32); REQUIRE(b.has_value()); cache.Publish(*b);
  CHECK_FALSE(cache.Pin("r", "a")); REQUIRE(cache.Pin("r", "b"));
  CHECK(cache.Pinned("r")->tokens == 32);
  CHECK_THROWS(cache.Reserve("invalid", 17));
  CHECK_THROWS(cache.Reserve("invalid", 0));
}
