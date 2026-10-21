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
// all entries. `0` DISABLES the cache: `GetOrDequant` still decodes (the same
// code path, so behavior and bit patterns are identical), but stores nothing,
// and `LinearBT` keeps the ORIGINAL threaded pool decode byte-for-byte. An
// explicit value sets the budget EXACTLY (that is how the 2026-10-20
// measurement below was taken: 16 GiB). The DEFAULT — the variable UNSET —
// is ON, with the budget DERIVED from the host's own memory report (THE
// DEFAULT BUDGET below). An entry larger than the whole budget is never
// cached (it would evict everything else on every insert). Because a returned
// Lease must own its bytes outright, the disabled/oversized path decodes into
// a FRESH per-call allocation (the old shared member scratch could be
// clobbered by the next call, which the lease contract forbids).
//
// COUNTERS. hits / misses / evictions / decode_calls / alloc_failures are
// exported through the VT_KOLIBRI1_PROFILE report (kolibri1_forward.cpp).
//
// THE DEFAULT BUDGET (2026-10-20, corrected 2026-10-21). The cache is worth
// MORE THAN 2x on CPU decode: 4.02 tok/s with the cache OFF against 9.07
// tok/s with a 16 GiB budget (2026-10-20, node 3, aarch64, 3.4 GHz, 32
// threads, empty-prompt 32-token decode; prefill p512 43.33 tok/s with the
// cache off). Shipping it OFF by default shipped the SLOW configuration, so
// the default is ON and the budget is DERIVED from the host's own memory
// report. Two MEASURED facts shape the policy.
//
// THE KNEE. The budget is worth its memory only at or above 16 GiB. At 8.25
// GiB the measured hit rate was 33% with 16068 evictions, and the decode
// throughput (3.84 tok/s) was indistinguishable from the cache being off
// (3.62 tok/s). A budget below the knee buys nothing for the memory it
// takes.
//
// THE SAMPLING MOMENT. The budget was read on FIRST USE of the cache, and
// the first use happens DURING the 74 GB checkpoint load: the page cache is
// full by then, so MemAvailable reads 66 GiB where the host's idle reading
// is ~212 GiB. The divisor then yields 8.25 GiB, below the knee, and the
// default ships the slow configuration by accident.
//
// So the policy reads the host at PROCESS START, before our own load
// depresses the reading, and floors the budget at the knee when the host can
// afford it:
//
//     honest_bytes = max(start_bytes, plan_bytes)
//     budget_bytes = clamp(honest_bytes / 8, 1 GiB, 32 GiB)
//     if honest_bytes >= 64 GiB: budget_bytes = max(budget_bytes, 16 GiB)
//
//   * `start_bytes` is MemAvailable read by a NAMESPACE-SCOPE initialiser,
//     which runs before main and before any model load. `plan_bytes` is the
//     reading at plan time. The MAXIMUM is the honest one: a reading lower
//     than an earlier one says that something took memory in between, and
//     the policy does not let its own checkpoint load talk the budget down.
//   * Both readings are `/proc/meminfo` MemAvailable, read through the house
//     helper `vllm::v1::host_available_memory_bytes()`
//     (include/vllm/v1/core/kv_cache_utils.h) — the SAME probe the
//     DeepSeek-V4 EXL3 host-residency budget uses, so the tree keeps ONE
//     memory reader.
//   * The DIVISOR (8) leaves seven eighths of the available memory for the
//     checkpoint itself (74 GB of fp8 weights), the KV cache and the OS.
//   * The KNEE FLOOR applies at or above the HOST THRESHOLD (64 GiB): a host
//     that reports 64 GiB as handable can afford a 16 GiB cache, which is a
//     quarter of what it reports. Below the threshold the divisor result
//     stands with the 1 GiB floor: the host is scarce, and its scarcity
//     speaks.
//   * The FLOOR (1 GiB) keeps the lever ON everywhere: a host with less than
//     8 GiB available still gets a cache, because re-dequantising every
//     weight on every forward costs far more than a gigabyte of residency.
//   * The CAP (32 GiB) bounds the footprint: the measured budget-sensitivity
//     table had its knee still open at 32 GiB (+5-6% over 16 GiB), so the
//     cap captures nearly the whole win while keeping the cache a small
//     fraction of the model it serves.
//   * An UNKNOWN report (MemAvailable unreadable -> 0) uses the FIXED 4 GiB
//     nominal: an unknown budget is not a reason to refuse, and not a reason
//     to hand out the cap. The nominal sits BELOW the knee on purpose: a
//     host that does not report its memory is not known to afford the knee.
//
// The budget is a BOUND, not a reservation: entries are allocated one at a
// time and evicted under the bound, so the process is never committed to the
// whole budget up front.
//
// THE FALLBACK. A cache that cannot get its bytes must NOT take the forward
// down with it. An allocation failure degrades the process to the UNCACHED
// path (the original threaded pool decode), which is slower and needs no
// residency: the cache drops its entries (handing the memory back), reports
// itself disabled, and every later forward takes the uncached path without
// trying again. The decoded BYTES are the same either way — the fallback
// changes the ROUTE, never the arithmetic.
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
#include <new>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/host_parallel.h"  // the ONE pool (#1664)
#include "vllm/v1/core/kv_cache_utils.h"  // host_available_memory_bytes
#include "vllm/model_executor/models/kolibri1_numa.h"  // the interleave policy
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
  // The DEGRADATIONS (THE FALLBACK): an allocation the cache could not
  // honour, after which the process runs the uncached path.
  uint64_t alloc_failures = 0;
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

// The PROCESS-START reading (THE SAMPLING MOMENT). A function-local static,
// so the reading is taken EXACTLY once, and the namespace-scope initialiser
// below forces it to be taken at process start instead of on first use.
inline uint64_t& StartAvailableBytesStorage() {
  static uint64_t value = [] {
    const int64_t available = vllm::v1::host_available_memory_bytes();
    return available > 0 ? static_cast<uint64_t>(available) : 0;
  }();
  return value;
}

}  // namespace detail

// THE PROCESS-START READING. The namespace-scope dynamic initialiser runs
// before main and before any model load, so the reading is the host's own
// availability and not what our own 74 GB checkpoint load left behind. The
// plan uses the MAXIMUM of this reading and the reading at plan time.
inline const uint64_t kStartAvailableBytes = detail::StartAvailableBytesStorage();

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
  // `budget_source_bytes` is the HONEST available-memory reading the budget
  // was derived from (0 when the budget was set explicitly, or when the host
  // does not report one). `budget_start_bytes` is the PROCESS-START reading
  // and `budget_plan_bytes` the reading at plan time (THE SAMPLING MOMENT).
  // They are kept so a reader can check the DERIVATION without re-reading a
  // number that has moved since.
  explicit Cache(size_t budget_bytes, uint64_t budget_source_bytes = 0,
                 uint64_t budget_start_bytes = 0,
                 uint64_t budget_plan_bytes = 0)
      : budget_(budget_bytes),
        budget_source_bytes_(budget_source_bytes),
        budget_start_bytes_(budget_start_bytes),
        budget_plan_bytes_(budget_plan_bytes) {}

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
    // The decoded block is WEIGHT-ARENA traffic the decode streams: park its
    // pages interleaved over the NUMA nodes (kolibri1_numa.h) so a multi-node
    // decode reads them local. Allocation policy only — bytes identical.
    [[maybe_unused]] const kolibri1_numa::PolicyGuard numa_guard;
    std::shared_ptr<std::vector<uint16_t>> bytes;
    try {
      bytes = std::make_shared<std::vector<uint16_t>>();
      DecodeInto(w, *bytes);
    } catch (const std::bad_alloc&) {
      // THE FALLBACK: the allocator would not give the cache its bytes. The
      // forward continues on the UNCACHED path (the caller sees an empty
      // lease and takes the original threaded pool decode), and the cache
      // degrades itself so the next forward does not try again.
      Degrade();
      return Lease();
    } catch (const std::length_error&) {
      // A weight whose decoded size overflows the allocator's limit: the same
      // route as a failed allocation, and the same degradation.
      Degrade();
      return Lease();
    }
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
  uint64_t budget_source_bytes() const { return budget_source_bytes_; }
  uint64_t budget_start_bytes() const { return budget_start_bytes_; }
  uint64_t budget_plan_bytes() const { return budget_plan_bytes_; }
  size_t resident_bytes() const { return resident_; }
  // False when the budget is 0, and after a DEGRADATION (THE FALLBACK):
  // `LinearBT` then keeps the ORIGINAL threaded pool decode (main-line
  // behavior byte-for-byte, host threads included) instead of routing
  // through this cache's serial decode.
  bool enabled() const {
    return budget_ != 0 && !degraded_.load(std::memory_order_relaxed);
  }
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

  // THE FALLBACK: drop every entry, hand the memory back, and report the
  // cache disabled. The entries' bytes are freed here unless a caller still
  // holds a lease (shared ownership keeps them alive until it does). Called
  // with the cache mutex held.
  void Degrade() {
    map_.clear();
    lru_.clear();
    resident_ = 0;
    degraded_.store(true, std::memory_order_relaxed);
    Counters().alloc_failures++;
  }

  const size_t budget_;
  const uint64_t budget_source_bytes_;
  const uint64_t budget_start_bytes_;
  const uint64_t budget_plan_bytes_;
  std::atomic<bool> degraded_{false};
  size_t resident_ = 0;
  Lru lru_;  // front == most recently used
  Map map_;
  std::mutex mutex_;
};

// THE DEFAULT BUDGET POLICY (THE DEFAULT BUDGET above). The constants are
// exposed so the gate can pin the policy — floor, cap, divisor, knee,
// threshold — without restating the numbers, and so the documentation and
// the code cannot drift.
inline constexpr uint64_t kDefaultBudgetDivisor = 8;
inline constexpr uint64_t kDefaultBudgetMinBytes = 1ull << 30;   // 1 GiB
inline constexpr uint64_t kDefaultBudgetMaxBytes = 32ull << 30;  // 32 GiB
// The nominal used when the host does not report its available memory.
inline constexpr uint64_t kDefaultBudgetNominalBytes = 4ull << 30;  // 4 GiB
// The MEASURED KNEE: the budget below which the cache thrashes and the
// decode throughput is indistinguishable from the cache being off.
inline constexpr uint64_t kDefaultBudgetKneeBytes = 16ull << 30;  // 16 GiB
// The HOST THRESHOLD: at or above this honest available reading the host can
// afford the knee (16 GiB is a quarter of 64 GiB).
inline constexpr uint64_t kDefaultBudgetHostThresholdBytes = 64ull << 30;  // 64 GiB

// The HONEST available-memory reading: the MAXIMUM of the PROCESS-START
// reading and the reading at plan time (THE SAMPLING MOMENT). A reading
// lower than an earlier one says that something took memory in between; the
// policy does not let its own checkpoint load, or anyone else's, talk the
// budget down.
inline uint64_t HonestAvailableBytes(uint64_t start_bytes, uint64_t plan_bytes) {
  return start_bytes > plan_bytes ? start_bytes : plan_bytes;
}

// The PURE policy: the budget for a given HONEST available-memory reading.
// Pure on purpose: the gate pins it without reading the host, so a policy
// change is a deterministic red, not a race against the machine's memory.
inline uint64_t DefaultBudgetBytesFor(uint64_t available_bytes) {
  if (available_bytes == 0) return kDefaultBudgetNominalBytes;  // UNKNOWN
  uint64_t derived = available_bytes / kDefaultBudgetDivisor;
  if (derived < kDefaultBudgetMinBytes) derived = kDefaultBudgetMinBytes;
  // The KNEE FLOOR: a host that can afford the measured knee must not be
  // served a budget below it, because a sub-knee budget buys nothing for the
  // memory it takes. Below the HOST THRESHOLD the divisor result stands: the
  // host is scarce, and its scarcity speaks.
  if (available_bytes >= kDefaultBudgetHostThresholdBytes &&
      derived < kDefaultBudgetKneeBytes) {
    derived = kDefaultBudgetKneeBytes;
  }
  if (derived > kDefaultBudgetMaxBytes) derived = kDefaultBudgetMaxBytes;
  return derived;
}

// How the process-wide cache's budget was decided.
struct BudgetPlan {
  size_t bytes = 0;           // the budget itself
  uint64_t source_bytes = 0;  // the HONEST available-memory reading it came
                              // from (0 when the operator set the budget
                              // exactly, or when the host reports none)
  uint64_t start_bytes = 0;   // the PROCESS-START reading (0 when the
                              // operator set the budget exactly)
  uint64_t plan_bytes = 0;    // the reading at plan time (0 when the
                              // operator set the budget exactly)
};

// The env knob, parsed. A plain non-negative decimal number of MiB parses;
// anything else (an empty value, a word, a negative number, a value with a
// suffix) does NOT, and an unparseable value falls back to the DEFAULT
// policy rather than to "disabled": the default turns this lever ON, so a
// typo must not silently switch it off.
inline bool ParseBudgetMiB(const char* value, uint64_t* out_mib) {
  if (value == nullptr || out_mib == nullptr) return false;
  const auto space = [](char c) {
    return c == ' ' || c == '\t' || c == '\n' || c == '\r' || c == '\v' ||
           c == '\f';
  };
  const char* p = value;
  while (*p != '\0' && space(*p)) ++p;
  const char* end = p;
  while (*end != '\0') ++end;
  while (end > p && space(end[-1])) --end;
  if (end == p) return false;
  uint64_t v = 0;
  for (const char* q = p; q != end; ++q) {
    if (*q < '0' || *q > '9') return false;
    v = v * 10 + static_cast<uint64_t>(*q - '0');
    // 2^32 MiB is 4 PiB: past any real budget, and small enough that the
    // MiB -> byte conversion below cannot overflow a 64-bit size_t.
    if (v > (1ull << 32)) return false;
  }
  *out_mib = v;
  return true;
}

// The process-wide cache `LinearBT` consults, configured ONCE:
//   * `VT_KOLIBRI1_DEQUANT_CACHE_MB` UNSET -> the DEFAULT budget (ON).
//   * `VT_KOLIBRI1_DEQUANT_CACHE_MB=0`     -> DISABLED.
//   * `VT_KOLIBRI1_DEQUANT_CACHE_MB=N`     -> exactly N MiB.
//   * an unparseable value                   -> the DEFAULT budget.
inline Cache& ProcessCache() {
  static const BudgetPlan plan = [] {
    const char* v = std::getenv("VT_KOLIBRI1_DEQUANT_CACHE_MB");
    uint64_t mib = 0;
    if (v == nullptr || !ParseBudgetMiB(v, &mib)) {
      // The DEFAULT: ON, with a budget derived from the host's own memory
      // report (THE DEFAULT BUDGET). The reading is the HONEST one: the
      // MAXIMUM of the PROCESS-START reading (taken before our own
      // checkpoint load depressed the host's report) and the reading at plan
      // time.
      const int64_t available = vllm::v1::host_available_memory_bytes();
      BudgetPlan p;
      p.start_bytes = kStartAvailableBytes;
      p.plan_bytes = available > 0 ? static_cast<uint64_t>(available) : 0;
      p.source_bytes = HonestAvailableBytes(p.start_bytes, p.plan_bytes);
      p.bytes = static_cast<size_t>(DefaultBudgetBytesFor(p.source_bytes));
      return p;
    }
    const uint64_t bytes = mib * 1024ull * 1024ull;
    BudgetPlan p;
    p.bytes = bytes > static_cast<uint64_t>(SIZE_MAX)
                  ? SIZE_MAX
                  : static_cast<size_t>(bytes);
    return p;
  }();
  static Cache cache(plan.bytes, plan.source_bytes, plan.start_bytes,
                     plan.plan_bytes);
  return cache;
}

}  // namespace kolibri1_dequant_cache
}  // namespace vllm

#endif  // VLLM_MODEL_EXECUTOR_MODELS_KOLIBRI1_DEQUANT_CACHE_H_
