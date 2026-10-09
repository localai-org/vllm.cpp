// Kolibri-1 in-process NUMA interleaving for the weight arenas
// (ISSUE-LOCAL-01M4GQ82X99D6JNBWDPX7ATJPY).
//
// MEASURED BASELINE (docs/bench-evidence/kolibri1-perf-next-lever-20261008.md,
// PR #3423 repair section): the external wrapper `numactl --interleave=all`
// plus 32 threads lifts decode from 3.17-3.21 to 4.69 tok/s (+47%). The
// mechanism is allocation placement, not scheduling: the loader and the
// dequant cache first-touch their pages on the loading thread's node, so a
// 32-thread decode spread over this host's 4 NUMA nodes streams most weight
// bytes REMOTE. The lever is therefore an allocation policy
// (`set_mempolicy(MPOL_INTERLEAVE)` scoped over the weight allocations), not
// a thread-placement change — and it must hold in-process, without numactl.
//
// The three cases:
//
//  1. POLICY ENGAGED: inside `vllm::kolibri1_numa::PolicyGuard` the calling task's
//     mempolicy is MPOL_INTERLEAVE over the full node mask; outside it the
//     previous policy is restored. `VT_KOLIBRI1_NUMA_INTERLEAVE=0` disables
//     the guard (the knob the A/B legs flip).
//  2. PAGES INTERLEAVED: a 128 MiB touched allocation inside the guard has
//     resident pages on MORE THAN ONE NUMA node (get_mempolicy MPOL_F_NODE |
//     MPOL_F_ADDR per page). The same allocation outside the guard, touched
//     single-threaded, sits on exactly one node — the first-touch shape the
//     lever removes. `numactl --interleave=all` cannot reach this binary's
//     allocations made before exec replay, which is precisely why the policy
//     must be in-process.
//
// NON-GOAL (pinned elsewhere, unchanged): the decode's numeric identity.
// Interleaving moves pages, never bytes; the W3 token gate
// (anchor 109726, chain md5 8de1463a) covers numerics at the final config.

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <set>
#include <vector>

#include "vllm/model_executor/models/kolibri1_dequant_cache.h"
#include "vllm/model_executor/models/kolibri1_numa.h"

#if defined(__linux__)

namespace {

std::vector<int> TouchedPageNodes(const void* addr, size_t bytes) {
  const size_t page = 4096;
  std::vector<int> nodes;
  const uintptr_t base = reinterpret_cast<uintptr_t>(addr);
  for (uintptr_t a = base; a < base + bytes; a += 64 * page) {
    int node = -1;
    if (vllm::kolibri1_numa::PageNode(reinterpret_cast<const void*>(a), &node) &&
        node >= 0) {
      nodes.push_back(node);
    }
  }
  return nodes;
}

size_t Distinct(const std::vector<int>& v) {
  return std::set<int>(v.begin(), v.end()).size();
}

}  // namespace

TEST_CASE("kolibri1 NUMA interleave: policy engages and pages interleave") {
  // 1. Default (knob unset == ENABLED): the guard sets MPOL_INTERLEAVE.
  SUBCASE("guard sets MPOL_INTERLEAVE by default") {
    REQUIRE(vllm::kolibri1_numa::Enabled());
    vllm::kolibri1_numa::PolicyGuard guard;
    CHECK(vllm::kolibri1_numa::CurrentPolicyIsInterleave());
  }
  // 1b. Knob off: the guard is inert.
  SUBCASE("VT_KOLIBRI1_NUMA_INTERLEAVE=0 disables the guard") {
    REQUIRE_EQ(setenv("VT_KOLIBRI1_NUMA_INTERLEAVE", "0", 1), 0);
    vllm::kolibri1_numa::PolicyGuard guard;
    CHECK_FALSE(vllm::kolibri1_numa::CurrentPolicyIsInterleave());
    REQUIRE_EQ(unsetenv("VT_KOLIBRI1_NUMA_INTERLEAVE"), 0);
  }
  // 2. The interleave actually scatters touched pages across nodes.
  SUBCASE("touched allocation spans multiple nodes under the guard") {
    const size_t bytes = 128ull << 20;
    std::vector<uint8_t> buf;
    {
      vllm::kolibri1_numa::PolicyGuard guard;
      buf.resize(bytes);
      for (size_t i = 0; i < bytes; i += 4096) buf[i] = 1;
    }
    CHECK(Distinct(TouchedPageNodes(buf.data(), bytes)) > 1);
  }
  // 2b. Outside the guard, single-threaded first-touch keeps the pages on
  // one node (the default-allocation shape that costs the +47%).
  SUBCASE("touched allocation outside the guard stays on one node") {
    const size_t bytes = 128ull << 20;
    std::vector<uint8_t> buf(bytes);
    for (size_t i = 0; i < bytes; i += 4096) buf[i] = 1;
    CHECK(Distinct(TouchedPageNodes(buf.data(), bytes)) == 1);
  }
}

// The allocation SITE, not just the policy: a cache miss's decoded block —
// the bytes the decode actually streams — lands interleaved. (The knob-off
// site behavior is deliberately NOT pinned here: the miss decode
// first-touches through the multi-threaded host pool, so even an inert
// guard can scatter pages; the off leg is the A/B bench's check instead.)
TEST_CASE("kolibri1 NUMA interleave: dequant-cache arena pages interleave") {
  namespace kdc = vllm::kolibri1_dequant_cache;
  vllm::Fp8BlockWeight w;
  w.n = 8192;
  w.k = 8192;  // 128 MiB of decoded bf16 — enough pages to span the nodes
  w.block_n = 128;
  w.block_k = 128;
  const int64_t sc_cols = (w.k + w.block_k - 1) / w.block_k;
  const int64_t sc_rows = (w.n + w.block_n - 1) / w.block_n;
  w.packed.bytes = vllm::OwnedBytes(std::vector<uint8_t>(
      static_cast<size_t>(w.n) * static_cast<size_t>(w.k), 0));
  w.packed.dtype = vt::DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = w.n;
  w.packed.shape[1] = w.k;
  w.scale.bytes = vllm::OwnedBytes(std::vector<uint8_t>(
      static_cast<size_t>(sc_rows * sc_cols) * sizeof(float), 0));
  w.scale.dtype = vt::DType::kF32;
  w.scale.rank = 2;
  w.scale.shape[0] = sc_rows;
  w.scale.shape[1] = sc_cols;

  kdc::Cache cache(512ull << 20);
  const auto lease = cache.GetOrDequant(w);
  REQUIRE(lease);
  CHECK(Distinct(TouchedPageNodes(lease.data(),
                                  lease.size() * sizeof(uint16_t))) > 1);
}

#endif  // __linux__
