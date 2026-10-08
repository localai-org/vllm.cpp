// Kolibri-1 fp8-block dequant cache (MODEL-TEXT-kolibri-1, re-land of
// ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2).
//
// The fp8-block arm of `LinearBT` re-dequants every used weight PER CALL
// (~1.4 G elements per decode step, 48% of gate-forward time per the profile).
// The dequant is a pure function of the weight bytes, so its result can be
// cached per weight and reused across decode steps. This is the SECOND
// implementation; the first was reverted when the row's token gate went red.
// Post-mortem: docs/bench-evidence/kolibri1-dequant-cache-negative-20261007.md.
//
// WHAT WAS WRONG WITH V1, AND WHY THIS ONE CANNOT REPEAT IT. V1 stored each
// entry's decoded block in a POOL BLOCK held through `DBuf::ReleaseShared` (the
// shared-ptr lease). The pool did not know a leased block was alive, handed the
// same address to a later `DBuf`, and a live cache entry suddenly contained
// another tensor's bytes — the `POOL DOUBLE-HAND-OUT` / `CACHE MISMATCH`
// corruption. The pool is now hardened (a second return of a leased block
// throws), but the real fix is OWNERSHIP: an entry here OWNS ITS BYTES in an
// independent `std::vector<uint16_t>` sized to the block. No entry ever holds a
// pointer into the pool, a `DBuf`, or any memory the cache does not itself
// allocate, so a cache hit can never alias pool traffic by construction.
//
// KEYING. The key is STABLE WEIGHT IDENTITY: the data pointers of the weight's
// own packed/scale `OwnedTensor` storage plus the shape/tiling scalars. Those
// buffers are owned by `Kolibri1Weights` and live for the process, so the
// pointer pair identifies one physical weight for the cache's life. The key
// NEVER names a pool block address, a `DBuf`, or any transient allocation.
//
// LIFETIME CONTRACT. On a hit, `GetOrDequant` returns a pointer into the
// entry's OWNED vector. That pointer is stable until the NEXT call on the same
// `Cache` object (any call — a later call may evict the entry an earlier
// pointer named). The caller must consume the bytes, including any GEMM over
// them, before its next cache call; the single serial decode caller in
// `LinearBT` satisfies this by scope. Concurrent cache users are serialized by
// the cache's mutex, so two threads never observe each other's eviction.
//
// BUDGET. `VT_KOLIBRI1_DEQUANT_CACHE_MB` sets the total budget in MiB across
// all entries. Default 0 == the cache is DISABLED: `GetOrDequant` still decodes
// (the same code path, so behavior and bit patterns are identical), but stores
// nothing, so main-line behavior is unchanged unless the var is set. An entry
// larger than the whole budget is never cached (it would evict everything else
// on every insert).
//
// COUNTERS. hits / misses / evictions / decode_calls are exported through the
// VT_KOLIBRI1_PROFILE report (kolibri1_forward.cpp).
#ifndef VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_DEQUANT_CACHE_H_
#define VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_DEQUANT_CACHE_H_

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "vllm/model_executor/models/host_parallel.h"  // the ONE pool (#1664)
#include "vllm/model_executor/models/kolibri1_fp8_dequant.h"  // DequantRowsBf16
#include "vllm/model_executor/models/kolibri1_weights.h"      // Fp8BlockWeight

namespace vllm {
namespace kolibri1_dequant_cache {

// The per-process hit/miss/eviction bookkeeping. Touched only from the serial
// decode caller (and the cache mutex), so plain integers suffice.
struct CacheCounters {
  uint64_t hits = 0;
  uint64_t misses = 0;
  uint64_t evictions = 0;
  uint64_t decode_calls = 0;
};

inline CacheCounters& Counters() {
  static CacheCounters c;
  return c;
}

namespace detail {

struct Key {
  const void* packed;  // w.packed.bytes.data() — weight-owned, process-lifetime
  const void* scale;   // w.scale.bytes.data() — weight-owned, process-lifetime
  int64_t n;
  int64_t k;
  int64_t block_n;
  int64_t block_k;

  bool operator==(const Key& o) const {
    return packed == o.packed && scale == o.scale && n == o.n && k == o.k &&
           block_n == o.block_n && block_k == o.block_k;
  }
};

struct KeyHash {
  size_t operator()(const Key& key) const {
    // FNV-1a over the members. Identity only — never a content hash (content
    // hashing 1.4 G packed bytes per call would cost more than the decode).
    size_t h = 1469598103934665603ull;
    const auto mix = [&h](size_t v) {
      h ^= v;
      h *= 1099511628211ull;
    };
    mix(reinterpret_cast<size_t>(key.packed));
    mix(reinterpret_cast<size_t>(key.scale));
    mix(static_cast<size_t>(key.n));
    mix(static_cast<size_t>(key.k));
    mix(static_cast<size_t>(key.block_n));
    mix(static_cast<size_t>(key.block_k));
    return h;
  }
};

}  // namespace detail

// One LRU cache of decoded bf16 blocks, keyed on stable weight identity. Each
// entry OWNS its bytes (see the file comment). Not copyable or movable: the
// returned pointers name memory this object owns.
class Cache {
 public:
  explicit Cache(size_t budget_bytes) : budget_(budget_bytes) {}

  // Returns the decoded bf16 [n, k] block for `w`: cache-owned bytes on a hit,
  // freshly decoded cache-owned bytes on a miss (the entry is inserted when it
  // fits), or freshly decoded SCRATCH bytes when the cache is disabled
  // (budget 0) or the block exceeds the whole budget. The decode is the exact
  // `kolibri1_fp8::DequantRowsBf16` body, so every path is bit-identical to
  // the uncached kernel by construction.
  //
  // The returned pointer is valid until the next call on this Cache (see the
  // lifetime contract in the file comment).
  const uint16_t* GetOrDequant(const Fp8BlockWeight& w) {
    std::lock_guard<std::mutex> lock(mutex_);
    Counters().decode_calls++;
    const detail::Key key{w.packed.bytes.data(), w.scale.bytes.data(),
                          w.n, w.k, w.block_n, w.block_k};
    const size_t count = static_cast<size_t>(w.n) * static_cast<size_t>(w.k);
    const size_t byte_count = count * sizeof(uint16_t);
    if (auto it = map_.find(key); it != map_.end()) {
      Touch(it->second);
      Counters().hits++;
      return it->second->bytes.data();
    }
    Counters().misses++;
    if (budget_ == 0 || byte_count > budget_) {
      // Disabled or oversized: decode into the scratch, store nothing. Same
      // decode body, so the returned bytes are bit-identical to a stored hit.
      DecodeInto(w, scratch_);
      return scratch_.data();
    }
    // Evict LRU entries until the new entry fits (checked against the budget,
    // not current residency, so one insert never thrashes more than needed).
    while (resident_ + byte_count > budget_ && !lru_.empty()) {
      resident_ -= lru_.back().bytes.size() * sizeof(uint16_t);
      map_.erase(lru_.back().key);
      lru_.pop_back();
      Counters().evictions++;
    }
    Entry entry;
    entry.key = key;
    entry.bytes.resize(count);
    DecodeInto(w, entry.bytes);
    resident_ += byte_count;
    lru_.push_front(std::move(entry));
    map_.emplace(key, lru_.begin());
    return lru_.front().bytes.data();
  }

  size_t budget_bytes() const { return budget_; }
  size_t resident_bytes() const { return resident_; }
  // False when the budget is 0: `LinearBT` then keeps the ORIGINAL threaded
  // pool decode (main-line behavior byte-for-byte, host threads included)
  // instead of routing through this cache's serial decode.
  bool enabled() const { return budget_ != 0; }
  size_t entries() const { return map_.size(); }

 private:
  struct Entry {
    detail::Key key;
    std::vector<uint16_t> bytes;  // OWNED — never a view into any pool block
  };
  using Lru = std::list<Entry>;
  using Map = std::unordered_map<detail::Key, Lru::iterator, detail::KeyHash>;

  // The exact uncached decode, partitioned over output rows through the ONE
  // host pool exactly as `DequantFp8Block` does it: the loop is elementwise
  // over OUTPUT rows only, so any partition is bit-identical to the serial
  // loop by construction (the pool determinism contract), and a cold miss
  // costs the host no more than the uncached path it replaces. Measured
  // without this: a serial cold decode tripled the bench prefill (44 s vs
  // 12.8 s) and ate most of the decode win.
  static void DecodeInto(const Fp8BlockWeight& w, std::vector<uint16_t>& dst) {
    const int64_t scale_cols = (w.k + w.block_k - 1) / w.block_k;
    const auto* src = w.packed.bytes.data();
    const auto* sc = reinterpret_cast<const float*>(w.scale.bytes.data());
    dst.resize(static_cast<size_t>(w.n) * static_cast<size_t>(w.k));
    host_parallel::ForOutputRows(w.n, w.k, [&](int64_t n0, int64_t n1) {
      kolibri1_fp8::DequantRowsBf16(src, sc, scale_cols, n0, n1, w.k,
                                    w.block_n, w.block_k, dst.data());
    });
  }

  void Touch(Lru::iterator it) {
    if (it != lru_.begin()) lru_.splice(lru_.begin(), lru_, it);
  }

  const size_t budget_;
  size_t resident_ = 0;
  Lru lru_;  // front == most recently used
  Map map_;
  std::vector<uint16_t> scratch_;  // the disabled/oversized decode target
  std::mutex mutex_;
};

// The process-wide cache `LinearBT` consults, configured ONCE from
// `VT_KOLIBRI1_DEQUANT_CACHE_MB` (default 0 == disabled).
inline Cache& ProcessCache() {
  static const size_t budget = [] {
    size_t mb = 0;
    if (const char* v = std::getenv("VT_KOLIBRI1_DEQUANT_CACHE_MB")) {
      mb = static_cast<size_t>(strtoull(v, nullptr, 10));
    }
    return mb * 1024 * 1024;
  }();
  static Cache cache(budget);
  return cache;
}

}  // namespace kolibri1_dequant_cache
}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_DEQUANT_CACHE_H_
