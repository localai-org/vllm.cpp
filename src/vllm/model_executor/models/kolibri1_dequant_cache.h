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
// WHAT THE 2026-10-08 REVIEW REPAIR (maint-bot on PR #3414,
// ISSUE-LOCAL-01M4CVDDHAFD7R1QCK9F493SWZ) CHANGED, AND WHY. The v2 cache had
// two lifetime holes, both closed here:
//
//   P1a — IDENTITY OUTLIVED THE WEIGHTS. The key was the weight's own
//   packed/scale data pointers plus the shape scalars, but the process-wide
//   `ProcessCache()` retains entries past the owning model's lifetime: after
//   unload+reload, the allocator hands the freed same-size weight buffers back
//   at the SAME addresses, and the reloaded model with the same geometry was
//   served the PREVIOUS model's decoded bytes. The key now also carries the
//   MODEL GENERATION (see KEYING below), bumped by the Kolibri1 loader at every
//   model load, so entries from an earlier model incarnation can never be hit
//   again — even at reused addresses.
//
//   P1b — HIT BYTES WERE UNPROTECTED DURING THE GEMM. `GetOrDequant` returned
//   a raw pointer into the entry and unlocked; a concurrent caller (distinct
//   engines share the static cache) could evict the entry and reuse its bytes
//   before the consuming `vt::MatmulBT` finished. `GetOrDequant` now returns a
//   `Lease`: shared ownership of the decoded bytes (see LIFETIME below). The
//   caller holds the lease across the GEMM; eviction drops the cache's
//   reference but a live lease keeps the bytes alive, so a caller mid-GEMM
//   never sees them freed or reused. Serializing the whole lookup-and-consume
//   under the cache mutex was rejected: it would hold the lock across the
//   GEMM — the expensive half — and serialize distinct engines' GEMMs.
//
// KEYING. The key is STABLE WEIGHT IDENTITY: the data pointers of the weight's
// own packed/scale `OwnedTensor` storage plus the shape/tiling scalars plus
// the MODEL GENERATION current at lookup. The buffers are owned by
// `Kolibri1Weights`, and the generation scopes them to one model incarnation:
// `Kolibri1LoadedModel`'s constructor bumps the process-wide generation at
// every model load, so a reload that reuses the previous model's addresses
// (the allocator's ordinary same-size behavior) produces different keys and
// fresh decodes. Stale-generation entries are never touched again, so they sit
// at the LRU tail and age out under ordinary budget pressure — no teardown
// path is needed, and none is wanted: a model destructor invalidating the
// process-lifetime static cache would couple unload to static destruction
// order. The key NEVER names a pool block address, a `DBuf`, or any transient
// allocation.
//
// LIFETIME CONTRACT. On any call, `GetOrDequant` returns a `Lease` — shared
// ownership of the decoded bf16 bytes. The leased bytes stay alive and
// unchanged until the lease is released (end of the holder's scope), whatever
// the cache does in between: eviction, other callers, scratch traffic. The
// bytes are owned through a `shared_ptr`, so eviction freeing the cache's
// reference cannot free a leased entry — the guarantee is by construction,
// the same ownership argument that fixed v1. The budget counts an evicted
// entry as evicted, so a live lease transiently overshoots the budget by at
// most the one in-flight GEMM's weight; leases are released right after the
// GEMM, so the overshoot is bounded and short. The single serial decode
// caller in `LinearBT` holds its lease across the one `vt::MatmulBT` and
// releases it at scope end. Concurrent cache users are serialized by the
// cache's mutex for the lookup/decode/insert itself; the lease extends the
// protection past the unlock, which the mutex alone never did.
//
// BUDGET. `VT_KOLIBRI1_DEQUANT_CACHE_MB` sets the total budget in MiB across
// all entries. Default 0 == the cache is DISABLED: `GetOrDequant` still decodes
// (the same code path, so behavior and bit patterns are identical), but stores
// nothing, so main-line behavior is unchanged unless the var is set. An entry
// larger than the whole budget is never cached (it would evict everything else
// on every insert). Because a returned Lease must own its bytes outright, the
// disabled/oversized path decodes into a FRESH per-call allocation (the old
// shared member scratch could be clobbered by the next call, which the lease
// contract forbids); `LinearBT` never calls `GetOrDequant` when the cache is
// disabled, so main-line behavior and its allocation profile are unchanged.
//
// COUNTERS. hits / misses / evictions / decode_calls are exported through the
// VT_KOLIBRI1_PROFILE report (kolibri1_forward.cpp).
#ifndef VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_DEQUANT_CACHE_H_
#define VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_DEQUANT_CACHE_H_

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <list>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <utility>
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

// The model generation (maint-bot P1a on PR #3414). The Kolibri1 loader bumps
// it at every model load (`Kolibri1LoadedModel`'s constructor); the cache key
// carries the generation current at lookup, so entries decoded for an earlier
// model incarnation can never be hit again — even when the next load reuses
// the previous model's weight-buffer addresses. Atomic because a load can
// race another engine's lookups; relaxed is enough (the cache mutex orders the
// key construction, the bump only has to be visible eventually — and it is,
// before the new model's first forward).
inline std::atomic<uint64_t>& ModelGenerationStorage() {
  static std::atomic<uint64_t> generation{0};
  return generation;
}
inline uint64_t ModelGeneration() {
  return ModelGenerationStorage().load(std::memory_order_relaxed);
}
inline uint64_t BumpModelGeneration() {
  return ModelGenerationStorage().fetch_add(1, std::memory_order_relaxed) + 1;
}

namespace detail {

struct Key {
  const void* packed;  // w.packed.bytes.data() — weight-owned storage
  const void* scale;   // w.scale.bytes.data() — weight-owned storage
  int64_t n;
  int64_t k;
  int64_t block_n;
  int64_t block_k;
  uint64_t generation;  // the model incarnation the entry belongs to (P1a)

  bool operator==(const Key& o) const {
    return packed == o.packed && scale == o.scale && n == o.n && k == o.k &&
           block_n == o.block_n && block_k == o.block_k &&
           generation == o.generation;
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
    mix(static_cast<size_t>(key.generation));
    return h;
  }
};

}  // namespace detail

// The ownership lease returned by `GetOrDequant` (maint-bot P1b on PR #3414).
// A Lease shares ownership of the decoded bytes: they stay alive and unchanged
// until the LAST lease (and the cache's own reference, if the entry is still
// resident) is gone, so a caller holding a lease across its GEMM is never
// exposed to an eviction or a scratch reuse. Move-only semantics are not
// required — copying a lease legitimately extends the bytes' lifetime.
class Lease {
 public:
  Lease() = default;
  const uint16_t* data() const { return bytes_ ? bytes_->data() : nullptr; }
  size_t size() const { return bytes_ ? bytes_->size() : 0; }
  explicit operator bool() const { return static_cast<bool>(bytes_); }

 private:
  friend class Cache;
  explicit Lease(std::shared_ptr<std::vector<uint16_t>> bytes)
      : bytes_(std::move(bytes)) {}
  std::shared_ptr<std::vector<uint16_t>> bytes_;
};

// One LRU cache of decoded bf16 blocks, keyed on stable weight identity. Each
// entry OWNS its bytes (see the file comment). Not copyable or movable: the
// returned leases name memory this object co-owns.
class Cache {
 public:
  explicit Cache(size_t budget_bytes) : budget_(budget_bytes) {}

  // Returns the decoded bf16 [n, k] block for `w` as a Lease: cache-owned
  // bytes on a hit, freshly decoded lease-owned bytes on a miss (the entry is
  // inserted when it fits), or freshly decoded lease-owned bytes when the
  // cache is disabled (budget 0) or the block exceeds the whole budget. The
  // decode is the exact `kolibri1_fp8::DequantRowsBf16` body, so every path is
  // bit-identical to the uncached kernel by construction.
  //
  // The leased bytes are valid until the lease is released (see the lifetime
  // contract in the file comment) — NOT merely until the next call on this
  // Cache.
  Lease GetOrDequant(const Fp8BlockWeight& w) {
    std::lock_guard<std::mutex> lock(mutex_);
    Counters().decode_calls++;
    const detail::Key key{w.packed.bytes.data(), w.scale.bytes.data(),
                          w.n, w.k, w.block_n, w.block_k, ModelGeneration()};
    const size_t count = static_cast<size_t>(w.n) * static_cast<size_t>(w.k);
    const size_t byte_count = count * sizeof(uint16_t);
    if (auto it = map_.find(key); it != map_.end()) {
      Touch(it->second);
      Counters().hits++;
      return Lease(it->second->bytes);
    }
    Counters().misses++;
    // Every path decodes into LEASE-OWNED bytes: the returned lease must own
    // its bytes outright, so even the disabled/oversized path allocates per
    // call (a shared member scratch could be clobbered by the next call,
    // which the lease contract forbids).
    auto bytes = std::make_shared<std::vector<uint16_t>>();
    DecodeInto(w, *bytes);
    if (budget_ == 0 || byte_count > budget_) {
      // Disabled or oversized: decode into the lease, store nothing. Same
      // decode body, so the bytes are bit-identical to a stored hit.
      return Lease(std::move(bytes));
    }
    // Evict LRU entries until the new entry fits (checked against the budget,
    // not current residency, so one insert never thrashes more than needed).
    // Eviction drops the cache's reference; a live lease keeps the bytes
    // alive (shared ownership), so a caller mid-GEMM never sees them freed
    // or reused. The budget counts the entry as evicted, so at most the
    // in-flight leased entry transiently overshoots the budget.
    while (resident_ + byte_count > budget_ && !lru_.empty()) {
      resident_ -= lru_.back().bytes->size() * sizeof(uint16_t);
      map_.erase(lru_.back().key);
      lru_.pop_back();
      Counters().evictions++;
    }
    Entry entry;
    entry.key = key;
    entry.bytes = bytes;
    resident_ += byte_count;
    lru_.push_front(std::move(entry));
    map_.emplace(key, lru_.begin());
    return Lease(std::move(bytes));
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
    // OWNED — never a view into any pool block. Held through a shared_ptr so
    // a Lease returned to a caller keeps the bytes alive across eviction.
    std::shared_ptr<std::vector<uint16_t>> bytes;
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
