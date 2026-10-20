// SPEC-DFLASH2 W10 (#1857) — the PagedAttention lane-classification seam.
//
// `include/vt/paged_attn_route.h` is the host half of the CUDA dispatch's
// prefill/decode class split, factored out so a box without a GPU can pin the
// ROUTING of a uniform-qlen speculative verify: classified + admissible ⇒ the
// DECODE class (split-KV), everything else ⇒ byte-identical to the shipped
// `num_tokens > num_reqs` predicate.
//
// Mutation targets: the shape guard's `> 1` (a pure decode must never be
// "uniform spec"), the `q * num_reqs` product (a stale field over a rewritten
// batch must never route), and the is-prefill polarity (dropping the
// spec-as-decode conjunct re-reds the verify-routing case).
#include <doctest/doctest.h>
#include <initializer_list>
#include <cstdlib>
#include <string>

#include "vt/paged_attn_route.h"
#include "vt/xpu_paged_decode_policy.h"

using vt::PagedAttnIsPrefill;
using vt::PagedAttnUniformSpecShape;

TEST_CASE("route: experimental batched Q1 requires exact live host widths") {
  const char* key = "VT_XPU_XE2_BATCHED_Q1";
  const bool present = std::getenv(key) != nullptr;
  const std::string saved = present ? std::getenv(key) : "";
  struct Restore {
    const char* key; bool present; std::string saved;
    ~Restore() { if (present) setenv(key, saved.c_str(), 1); else unsetenv(key); }
  } restore{key, present, saved};
  REQUIRE(unsetenv(key) == 0);
  CHECK(vt::XpuBatchedQ1Enabled());
  REQUIRE(setenv(key, "0", 1) == 0);
  CHECK_FALSE(vt::XpuBatchedQ1Enabled());
  REQUIRE(setenv(key, "1", 1) == 0);
  CHECK(vt::XpuBatchedQ1Enabled());
  const int32_t uniform[] = {0, 1, 2, 3, 4, 5};
  for (int requests : {2, 3, 4}) {
    CHECK(vt::XpuPureBatchedQ1Shape(requests, requests, uniform));
    CHECK_FALSE(vt::XpuPureBatchedQ1Shape(requests, requests, nullptr));
    CHECK_FALSE(vt::XpuPureBatchedQ1Shape(requests * 4, requests, uniform));
    CHECK_FALSE(vt::XpuPureBatchedQ1Shape(INT64_MAX, requests, uniform));
  }
  for (int requests : {0, 1, 5})
    CHECK_FALSE(vt::XpuPureBatchedQ1Shape(requests, requests, uniform));
  for (const auto& offsets : {std::initializer_list<int32_t>{0, 0, 2},
       {0, 2, 2}, {1, 1, 2}, {0, 1, 3}})
    CHECK_FALSE(vt::XpuPureBatchedQ1Shape(2, 2, offsets.begin()));
  const int32_t mixed[] = {0, 1, 4096};
  CHECK_FALSE(vt::XpuPureBatchedQ1Shape(4096, 2, mixed));
  for (const char* invalid : {"", "2", "01", "1 ", "auto"}) {
    REQUIRE(setenv(key, invalid, 1) == 0);
    CHECK_THROWS(vt::XpuBatchedQ1Enabled());
  }
}

TEST_CASE("route: experimental long Q1 environment is bounded and preserves defaults") {
  struct Restore {
    std::string flag, splits;
    bool had_flag, had_splits;
    ~Restore() {
      if (had_flag) setenv("VT_XPU_XE2_LONG_DECODE", flag.c_str(), 1);
      else unsetenv("VT_XPU_XE2_LONG_DECODE");
      if (had_splits) setenv("VT_XPU_XE2_DECODE_SPLITS", splits.c_str(), 1);
      else unsetenv("VT_XPU_XE2_DECODE_SPLITS");
    }
  } restore{std::getenv("VT_XPU_XE2_LONG_DECODE") ? std::getenv("VT_XPU_XE2_LONG_DECODE") : "",
            std::getenv("VT_XPU_XE2_DECODE_SPLITS") ? std::getenv("VT_XPU_XE2_DECODE_SPLITS") : "",
            std::getenv("VT_XPU_XE2_LONG_DECODE") != nullptr,
            std::getenv("VT_XPU_XE2_DECODE_SPLITS") != nullptr};
  REQUIRE(setenv("VT_XPU_XE2_DECODE_SPLITS", "32", 1) == 0);
  // The local B70 default now includes the qualified bounded long route.
  // Explicit mode1 remains the rollback to the original three-page policy.
  REQUIRE(unsetenv("VT_XPU_XE2_LONG_DECODE") == 0);
  CHECK(vt::XpuPagedDecodePlanFromEnvironment(65491, 1600).max_keys == 65600);
  CHECK(vt::XpuPagedDecodePlanFromEnvironment(65491, 1600).splits == 32);
  CHECK(vt::XpuPagedDecodePlanFromEnvironment(204801, 1600).max_keys == 0);
  REQUIRE(setenv("VT_XPU_XE2_LONG_DECODE", "1", 1) == 0);
  CHECK(vt::XpuPagedDecodePlanFromEnvironment(65491, 1600).max_keys == 0);
  REQUIRE(setenv("VT_XPU_XE2_LONG_DECODE", "2", 1) == 0);
  for (const int64_t page : {1600, 1664}) {
    for (const char* splits : {"16", "32"}) {
      REQUIRE(setenv("VT_XPU_XE2_DECODE_SPLITS", splits, 1) == 0);
      for (const int64_t length : {960, 961, 4096, 4800, 4801, 16384,
                                   65491, 131070, 199673}) {
        const auto plan = vt::XpuPagedDecodePlanFromEnvironment(length, page);
        CHECK(plan.max_keys == (length <= 960 ? 960 : ((length - 1) / page + 1) * page));
        CHECK(plan.splits == (length <= 960 ? 1 : std::stoi(splits)));
      }
      CHECK(vt::XpuPagedDecodePlanFromEnvironment(128 * page, page).max_keys == 128 * page);
      CHECK(vt::XpuPagedDecodePlanFromEnvironment(128 * page + 1, page).max_keys == 0);
    }
  }
  CHECK(vt::XpuPagedDecodePlanFromEnvironment(INT64_MAX, 1600).max_keys == 0);
  CHECK(vt::XpuPagedDecodePlanFromEnvironment(65491, 16).max_keys == 0);
  CHECK(vt::XpuPagedDecodePlanFromEnvironment(-1, 1600).max_keys == 0);
}

TEST_CASE("route: XPU long C1 decode buckets and bounded splits") {
  for (int64_t page : {1600, 1664}) {
    for (int64_t length : {1, 960}) {
      const auto p = vt::PagedAttnXpuDecodePlan(length, page, false);
      CHECK(p.max_keys == 960); CHECK(p.splits == 1);
    }
    for (int splits : {16, 32}) for (int64_t length : {961, 4095, 4096, 4352, 4800}) {
      const auto p = vt::PagedAttnXpuDecodePlan(length, page, true, splits);
      CHECK(p.max_keys == ((length + page - 1) / page) * page);
      CHECK(p.splits == splits);
      CHECK(vt::PagedAttnXpuDecodePlan(length, page, false, splits).max_keys == 0);
    }
    CHECK(vt::PagedAttnXpuDecodePlan(page * 3, page, true).max_keys == page * 3);
    CHECK(vt::PagedAttnXpuDecodePlan(page * 3 + 1, page, true).max_keys == 0);
    CHECK(vt::PagedAttnXpuDecodePlan(4096, page, true, 8).max_keys == 0);
    CHECK(vt::PagedAttnXpuDecodePlan(INT64_MAX, page, true).max_keys == 0);
    CHECK(vt::PagedAttnXpuDecodePlan(0, page, true).max_keys == 0);
    CHECK(vt::PagedAttnXpuDecodePlan(-1, page, true).max_keys == 0);
  }
  CHECK(vt::PagedAttnXpuDecodePlan(961, 0, true).max_keys == 0);
  CHECK(vt::PagedAttnXpuDecodePlan(961, 16, true).max_keys == 0);
}

TEST_CASE("route: XPU captured attention policy changes at actual host boundaries") {
  CHECK(vt::PagedAttnXpuShortDecodeBound(960));
  CHECK_FALSE(vt::PagedAttnXpuShortDecodeBound(961));
  CHECK_FALSE(vt::PagedAttnXpuLongSplitBound(4095));
  CHECK(vt::PagedAttnXpuLongSplitBound(4096));
  // Real MTP3 serving crosses this page while its table shape stays fixed.
  CHECK(vt::PagedAttnXpuActivePages(4096, 1600) == 3);
  CHECK(vt::PagedAttnXpuActivePages(4800, 1600) == 3);
  CHECK(vt::PagedAttnXpuActivePages(4801, 1600) == 4);
  CHECK(vt::PagedAttnXpuActivePages(4992, 1664) == 3);
  CHECK(vt::PagedAttnXpuActivePages(4993, 1664) == 4);
  CHECK(vt::PagedAttnXpuActivePages(0, 1600) == 0);
  CHECK(vt::PagedAttnXpuActivePages(4096, 0) == 0);
}

TEST_CASE("route: the classified uniform verify shape is admitted") {
  // The #1857 measured shape: 1 request, q=9 verify (k=8).
  CHECK(PagedAttnUniformSpecShape(/*num_tokens=*/9, /*num_reqs=*/1, /*uq=*/9));
  // Batched verify: 4 requests at q=3.
  CHECK(PagedAttnUniformSpecShape(/*num_tokens=*/12, /*num_reqs=*/4, /*uq=*/3));
}

TEST_CASE("route: unclassified, stale and degenerate shapes are refused") {
  // Not classified (the default 0): never a spec batch.
  CHECK_FALSE(PagedAttnUniformSpecShape(9, 1, 0));
  // q == 1 is a pure decode, which routes decode by shape already.
  CHECK_FALSE(PagedAttnUniformSpecShape(4, 4, 1));
  // STALE FIELD over a padded pure-decode rebuild: the BuildPaddedDecode
  // rewrites make num_tokens == num_reqs, and S == q*S only at q == 1.
  CHECK_FALSE(PagedAttnUniformSpecShape(/*num_tokens=*/16, /*num_reqs=*/16, /*uq=*/9));
  // Shape/classification mismatch (a batch that shrank or grew).
  CHECK_FALSE(PagedAttnUniformSpecShape(/*num_tokens=*/5, /*num_reqs=*/1, /*uq=*/9));
  CHECK_FALSE(PagedAttnUniformSpecShape(/*num_tokens=*/9, /*num_reqs=*/2, /*uq=*/9));
  // Degenerate batch.
  CHECK_FALSE(PagedAttnUniformSpecShape(0, 0, 9));
}

TEST_CASE("route: a spec-as-decode batch takes the DECODE class") {
  // THE #1857 ROUTING PIN: the uniform-qlen verify must NOT be prefill when the
  // decode lane admits it. Before W10 the dispatch had no spec term and this
  // batch rode the prefill flash ladder at num_splits=1.
  CHECK_FALSE(PagedAttnIsPrefill(/*num_tokens=*/9, /*num_reqs=*/1,
                                 /*spec_as_decode=*/true));
  CHECK_FALSE(PagedAttnIsPrefill(/*num_tokens=*/12, /*num_reqs=*/4,
                                 /*spec_as_decode=*/true));
}

TEST_CASE("route: without the spec admission the shipped predicate holds verbatim") {
  // Prefill: more tokens than requests.
  CHECK(PagedAttnIsPrefill(9, 1, /*spec_as_decode=*/false));
  CHECK(PagedAttnIsPrefill(5, 2, /*spec_as_decode=*/false));
  // Pure decode: one token per request.
  CHECK_FALSE(PagedAttnIsPrefill(4, 4, /*spec_as_decode=*/false));
  CHECK_FALSE(PagedAttnIsPrefill(1, 1, /*spec_as_decode=*/false));
}

TEST_CASE("route: XPU packed verification graph bound is stable and overflow safe") {
  CHECK(vt::PagedAttnXpuPackedVerifyBound(4096, 1600) == 4800);
  CHECK(vt::PagedAttnXpuPackedVerifyBound(4800, 1600) == 4800);
  CHECK(vt::PagedAttnXpuPackedVerifyBound(4801, 1600) == 6400);
  CHECK(vt::PagedAttnXpuPackedVerifyBound(4096, 1664) == 4992);
  CHECK(vt::PagedAttnXpuPackedVerifyBound(262144, 1600) == 262400);
  CHECK(vt::PagedAttnXpuPackedVerifyBound(0, 1600) == 0);
  CHECK(vt::PagedAttnXpuPackedVerifyBound(4, 0) == 4);
  CHECK(vt::PagedAttnXpuPackedVerifyBound(INT32_MAX, 1600) == INT32_MAX);
  CHECK(vt::PagedAttnXpuPackedVerifyBound(4, INT64_MAX) == 4);
}
