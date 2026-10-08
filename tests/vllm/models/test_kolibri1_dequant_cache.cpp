// Kolibri-1 fp8-block dequant cache: the re-land gate
// (ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2, after the pool-lease hardening).
//
// The first implementation of this lever was reverted when the row's token
// gate went red: a live cache entry was found holding ANOTHER tensor's bytes
// (`CACHE MISMATCH n=512 k=2560`), traced to the pool handing a block that a
// `DBuf::ReleaseShared` lease still owned to a fresh allocation — the
// `POOL DOUBLE-HAND-OUT` (docs/bench-evidence/
// kolibri1-dequant-cache-negative-20261007.md). The re-land stores each entry
// in bytes the cache OWNS, so the aliasing shape cannot be constructed at all.
// These three cases pin exactly that:
//
//  1. BYTE IDENTITY: a cache hit returns bit-identical raw-uint16 bytes to a
//     fresh kernel decode — cold and hot, over all 256 e4m3 bytes x
//     representative scales (the same domain the NEON dequant gate covers).
//  2. BUDGET INVARIANCE: the same decode sequence produces the identical
//     output chain with the cache disabled (budget 0), a budget that forces
//     LRU evictions, and a larger budget, in ONE binary — the gate the first
//     attempt lacked.
//  3. OWNERSHIP: a hit entry's bytes are unchanged after intervening pool
//     Get/Put traffic of the same size class (the exact shape that killed v1;
//     on the hardened pool, a second return now throws instead).
//  4. LRU VICTIM: the evicting budget evicts the LEAST recently used entry,
//     not the most recent one (distinguished by which key hits next).
//  5. DEFAULT OFF: `ProcessCache()` with `VT_KOLIBRI1_DEQUANT_CACHE_MB`
//     unset is disabled (budget 0), pinned in a fork/exec'd helper whose
//     environment provably lacks the variable.
//
// KNOWN LIMITS: byte-identity cannot catch a key weakened to scale-pointer
// only (mitigated by weight-owned process-lifetime storage, not by a test),
// and the ownership guarantee's teeth are by-construction rather than
// asserted — a future reader need not rediscover either.

#include <doctest/doctest.h>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <sys/wait.h>
#include <unistd.h>

#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf
#include "vllm/model_executor/models/kolibri1_dequant_cache.h"
#include "vllm/model_executor/models/kolibri1_fp8_dequant.h"
#include "vt/device.h"

namespace {

using vllm::Fp8BlockWeight;
using vllm::kolibri1_dequant_cache::Cache;

// A deterministic fp8-block weight: packed bytes are (i * 7 + seed) and the
// scale grid walks the same spread of scales the dequant kernel gate uses.
float ScaleForBlock(size_t i) {
  static const float kScales[] = {
      1.0f, 0x1p-14f, 3.0e34f, -1.5f, 0.0f, 0x1p-9f, 1.015625f, 1.1f};
  return kScales[i % 8];
}

Fp8BlockWeight MakeWeight(int64_t n, int64_t k, int64_t bn, int64_t bk,
                          int seed) {
  Fp8BlockWeight w;
  w.n = n;
  w.k = k;
  w.block_n = bn;
  w.block_k = bk;
  w.packed.bytes = vllm::OwnedBytes(
      std::vector<uint8_t>(static_cast<size_t>(n * k), 0));
  w.packed.dtype = vt::DType::kI8;
  w.packed.rank = 2;
  w.packed.shape[0] = n;
  w.packed.shape[1] = k;
  auto* p = w.packed.bytes.data();
  for (int64_t i = 0; i < n * k; ++i) {
    p[i] = static_cast<uint8_t>(i * 7 + seed);
  }
  const int64_t sc_cols = (k + bk - 1) / bk;
  const int64_t sc_rows = (n + bn - 1) / bn;
  w.scale.bytes = vllm::OwnedBytes(std::vector<uint8_t>(
      static_cast<size_t>(sc_rows * sc_cols) * sizeof(float), 0));
  w.scale.dtype = vt::DType::kF32;
  w.scale.rank = 2;
  w.scale.shape[0] = sc_rows;
  w.scale.shape[1] = sc_cols;
  auto* s = reinterpret_cast<float*>(w.scale.bytes.data());
  for (int64_t i = 0; i < sc_rows * sc_cols; ++i) {
    s[i] = ScaleForBlock(static_cast<size_t>(i)) *
           (0x1p-3f * static_cast<float>((i + seed) % 5) + 0.5f);
  }
  return w;
}

// The reference: a fresh kernel decode straight off the weight, no cache.
std::vector<uint16_t> RefDecode(const Fp8BlockWeight& w) {
  std::vector<uint16_t> out(static_cast<size_t>(w.n) * static_cast<size_t>(w.k));
  const int64_t sc_cols = (w.k + w.block_k - 1) / w.block_k;
  vllm::kolibri1_fp8::DequantRowsBf16(
      w.packed.bytes.data(), reinterpret_cast<const float*>(w.scale.bytes.data()),
      sc_cols, 0, w.n, w.k, w.block_n, w.block_k, out.data());
  return out;
}

uint64_t ChainStep(uint64_t h, const uint16_t* p, size_t count) {
  for (size_t i = 0; i < count; ++i) {
    h ^= static_cast<uint64_t>(p[i]);
    h *= 1099511628211ull;
  }
  return h;
}

}  // namespace

// ── 1. Byte identity, cold and hot, over the full e4m3 byte domain ──────────
TEST_CASE("kolibri1 dequant cache: hits serve byte-identical raw-uint16 bytes "
          "to a fresh decode") {
  // One row carrying ALL 256 e4m3 byte values (padded past the block so a
  // vector read never overruns), one weight per representative scale, K=256,
  // block_k=128 — the same domain shape the NEON dequant gate pins.
  const int64_t kK = 256 + 16;
  const float scales[] = {1.0f, 0x1p-14f, 3.0e34f, -1.5f, 0.0f, 0x1p-9f,
                          1.015625f, 1.1f};
  std::vector<Fp8BlockWeight> weights;
  for (int si = 0; si < 8; ++si) {
    weights.push_back(MakeWeight(1, kK, 1, 128, si));
    // Force the exact byte domain into the packed row.
    auto* p = weights.back().packed.bytes.data();
    for (int i = 0; i < 256; ++i) p[i] = static_cast<uint8_t>(i);
    // And the SAME scale for the whole row, so the reference is exact.
    auto* s = reinterpret_cast<float*>(weights.back().scale.bytes.data());
    s[0] = s[1] = scales[si];
  }
  // Budget comfortably above any entry: hits are expected.
  Cache cache(1u << 20);
  for (int round = 0; round < 2; ++round) {
    for (const auto& w : weights) {
      const uint16_t* got = cache.GetOrDequant(w);
      const auto want = RefDecode(w);
      CHECK_EQ(std::memcmp(got, want.data(), want.size() * 2), 0);
    }
  }
  // Round 0 was all misses, round 1 all hits.
  CHECK_EQ(vllm::kolibri1_dequant_cache::Counters().hits >= 8, true);
  // A second hit on the same weight returns the SAME stable pointer.
  CHECK_EQ(cache.GetOrDequant(weights[0]),
           cache.GetOrDequant(weights[0]));
}

// ── 2. Budget invariance: identical chain at off / evicting / large budget ──
TEST_CASE("kolibri1 dequant cache: the decode chain is budget-invariant "
          "across off, evicting, and large budgets in one binary") {
  // Eight distinct weights; the access order revisits early weights AFTER the
  // later ones, so a small budget must evict and re-decode them.
  std::vector<Fp8BlockWeight> weights;
  for (int i = 0; i < 8; ++i) weights.push_back(MakeWeight(8, 512, 4, 128, i));
  const size_t kEntry = 8 * 512 * 2;  // bytes per entry
  const std::vector<size_t> budgets = {0, kEntry * 2 /*forces eviction*/,
                                       kEntry * 16 /*no eviction*/};
  // Access order: 0..7 then 3,5,1,7,0,4,2,6.
  const int order[] = {0, 1, 2, 3, 4, 5, 6, 7, 3, 5, 1, 7, 0, 4, 2, 6};
  std::vector<uint64_t> chains;
  for (size_t b : budgets) {
    Cache cache(b);
    uint64_t h = 1469598103934665603ull;
    for (int idx : order) {
      const uint16_t* got = cache.GetOrDequant(weights[static_cast<size_t>(idx)]);
      h = ChainStep(h, got, static_cast<size_t>(weights[static_cast<size_t>(idx)].n) *
                                   static_cast<size_t>(weights[static_cast<size_t>(idx)].k));
    }
    chains.push_back(h);
  }
  // All three chains identical, and equal to the no-cache kernel chain.
  uint64_t ref = 1469598103934665603ull;
  for (int idx : order) {
    const auto want = RefDecode(weights[static_cast<size_t>(idx)]);
    ref = ChainStep(ref, want.data(), want.size());
  }
  for (size_t i = 0; i < chains.size(); ++i) {
    CHECK_EQ(chains[i], ref);
  }
  // The evicting budget actually evicted and still served identical bytes.
  Cache evicting(kEntry * 2);
  for (int idx : order) evicting.GetOrDequant(weights[static_cast<size_t>(idx)]);
  CHECK_EQ(vllm::kolibri1_dequant_cache::Counters().evictions > 0, true);
  CHECK_EQ(vllm::kolibri1_dequant_cache::Counters().hits > 0, true);
}

// ── 3. Ownership: hit bytes survive heavy pool Get/Put traffic ──────────────
TEST_CASE("kolibri1 dequant cache: a hit entry's bytes are unchanged after "
          "intervening pool traffic (the v1 aliasing shape)") {
  const int64_t kN = 512, kK = 2560;  // the shape the CACHE MISMATCH named
  const Fp8BlockWeight w = MakeWeight(kN, kK, 128, 128, 3);
  const size_t entry_bytes = static_cast<size_t>(kN * kK) * 2;
  Cache cache(entry_bytes * 4);
  const uint16_t* first = cache.GetOrDequant(w);
  const auto ref = RefDecode(w);
  REQUIRE_EQ(std::memcmp(first, ref.data(), ref.size() * 2), 0);

  // Heavy same-size-class pool traffic, each block filled with a pattern
  // BEFORE it is returned, so a re-handed block cannot pass by luck.
  vt::Queue queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vllm::dense_attn::Dev d{vt::GetBackend(vt::DeviceType::kCPU), queue};
  for (int i = 0; i < 8; ++i) {
    vllm::dense_attn::DBuf block(d, vt::DType::kBF16, {kN, kK});
    REQUIRE_EQ(block.bytes(), entry_bytes);
    auto* p = static_cast<uint16_t*>(block.ptr());
    for (size_t j = 0; j < entry_bytes / 2; ++j) p[j] = static_cast<uint16_t>(0xDEAD + j);
  }

  // The entry survived: same pointer, same bytes.
  const uint16_t* again = cache.GetOrDequant(w);
  CHECK_EQ(again, first);
  CHECK_EQ(std::memcmp(again, ref.data(), ref.size() * 2), 0);
}

// ── 4. LRU victim selection: the least recently used entry goes first ───────
TEST_CASE("kolibri1 dequant cache: the evicting budget evicts the LRU entry, "
          "not the most recent one") {
  std::vector<Fp8BlockWeight> weights;
  for (int i = 0; i < 3; ++i) {
    weights.push_back(MakeWeight(8, 512, 4, 128, 100 + i));
  }
  const size_t kEntry = 8 * 512 * 2;  // bytes per entry
  Cache cache(kEntry * 2);            // room for exactly two entries
  auto& counters = vllm::kolibri1_dequant_cache::Counters();
  const uint64_t misses0 = counters.misses;
  const uint64_t evictions0 = counters.evictions;

  cache.GetOrDequant(weights[0]);
  cache.GetOrDequant(weights[1]);
  cache.GetOrDequant(weights[0]);  // TOUCH: recency order is now 0, then 1
  CHECK_EQ(counters.misses - misses0, 2);

  // One insert over budget: exactly one eviction, and it must take the LRU
  // victim (weight 1). An MRU policy would evict the just-touched weight 0.
  cache.GetOrDequant(weights[2]);
    CHECK_EQ(counters.evictions - evictions0, 1);
  CHECK_EQ(cache.resident_bytes(), kEntry * 2);

  // The touched weight 0 is still resident: this is a HIT, not a fresh miss.
  cache.GetOrDequant(weights[0]);
  CHECK_EQ(counters.misses - misses0, 3);
  // And the LRU victim weight 1 was the one evicted: this IS a fresh miss.
  cache.GetOrDequant(weights[1]);
  CHECK_EQ(counters.misses - misses0, 4);
}

// ── 5. Default off: ProcessCache() with the env var unset is disabled ───────
TEST_CASE("kolibri1 dequant cache: ProcessCache() defaults to a disabled "
          "cache when VT_KOLIBRI1_DEQUANT_CACHE_MB is unset") {
  // The budget is read ONCE per process at the static init inside
  // ProcessCache(). To pin the DEFAULT (and not whatever this process
  // happens to carry, or an earlier test case has already latched), the
  // assertion runs in a fork/exec'd helper binary launched with an
  // environment that provably lacks the variable; the helper exits 0 iff
  // budget_bytes() == 0 and !enabled().
  REQUIRE_EQ(std::getenv("VT_KOLIBRI1_DEQUANT_CACHE_MB"),
             static_cast<char*>(nullptr));
  const pid_t pid = fork();
  REQUIRE(pid >= 0);
  if (pid == 0) {
    char* envp[] = {nullptr};  // empty environment: the var is provably unset
    execle(KOLIBRI_DEQUANT_CACHE_DEFAULT_HELPER, "helper", static_cast<char*>(nullptr),
           envp);
    _exit(127);  // exec failed — not a pass
  }
  int status = 0;
  REQUIRE_EQ(waitpid(pid, &status, 0), pid);
  REQUIRE(WIFEXITED(status));
  CHECK_EQ(WEXITSTATUS(status), 0);
}
