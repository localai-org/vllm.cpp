// Shared process-wide caching device allocator (DevicePool) — extracted VERBATIM
// from the Qwen3.6 forward (qwen3_5.cpp) so the dense Qwen3 forward (qwen3.cpp)
// reuses the SAME pooled-scratch machinery instead of raw per-op Backend
// Alloc/Free. The relocation was byte-for-byte the qwen3_5.cpp definitions; what
// has changed since is that a pool is now bound to ONE DEVICE (see below), and
// the accessors take the backend that names it.
//
// Rationale: both cudaMalloc AND cudaFree SYNCHRONIZE the whole device, so the
// per-op DBuf alloc/free churn in a forward (thousands of tiny scratch buffers per
// step) is itself a sync storm. This pool reuses freed blocks (size-class keyed)
// instead of hitting cudaMalloc/cudaFree, so after a brief warm-up almost every
// DBuf lifetime is sync-free. Reuse is safe under the forward's single-queue
// (single-stream) ordering: a block returned to the pool is only handed back out
// on the same queue, and CUDA stream ordering guarantees the op that last touched
// the block has completed before any reused op runs — no host sync needed. Blocks
// are never returned to the driver (leak at process exit, like the cublasLt
// workspace); the pool is bounded by the forward's peak concurrent scratch.
//
// ONE POOL PER DEVICE (#516, .agents/specs/pool-device-key.md). Until this was
// fixed there was ONE pool for the whole process and its free list was keyed by
// byte size class alone, so a block allocated through one backend was handed to
// the next caller of that size class whatever device it was running on. One
// fault, two symptoms, chosen by direction: a cudaMalloc block reaching a
// CPU-backend forward SIGSEGVs host-side (and compute-sanitizer is CLEAN,
// because the fault is not on the device), while a host block reaching a CUDA
// forward returned a UNIFORM 0x7fff0000 quiet NaN — computed and propagated, not
// garbage read. vLLM never had the bug because it never had the design: its
// allocation handle carries the device as field 0
// (vllm/device_allocator/__init__.py:12-14 @ pin 555967922) and torch's cache is
// per-device by construction (c10/cuda/CUDACachingAllocator.h:118-172).
//
// So a `DevicePool` is BOUND to one backend, `Pool(b)` resolves the pool for a
// device, and there is deliberately NO way to spell "the pool" without naming a
// device. The `vt::Backend*` IS the device identity: the registry hands out
// exactly one Backend* per Device{type,index} (vt/backend.h, kMaxDevicesPerType),
// and GetBackend(type) and GetBackend(Device{type,0}) return the identical
// pointer — so the device enters the key with no new virtual on vt::Backend and
// therefore no edit to any backend implementation.
#pragma once

#include <atomic>
#include <bit>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "vt/backend.h"

namespace vllm {

// SIZE-CLASS BUCKETING (prefill sync-cudaMalloc kill). Keying the pool on the
// EXACT byte size defeats reuse during prefill: under continuous batching each
// engine step processes a different token count T (prefill-chunk tokens + decode
// tokens), so every [T,*] scratch is a byte size that step-1 never saw — an
// exact-key MISS -> a SYNCHRONOUS cudaMalloc (device-serializing, does NOT
// overlap compute) on the forward's host thread, thousands per prefill. Rounding
// the request UP to a size class (keep the top `kClassBits` significant bits;
// <=1/2^kClassBits ~ 6.25% over-allocation) makes nearby-T scratch of the same
// op share ONE block, so after warm-up almost every prefill DBuf is a pool hit.
// The returned block is >= the requested bytes (the Tensor view uses only the
// logical prefix), and Put/Get round identically so a block always returns to
// its own class bucket. VT_POOL_EXACT=1 restores exact keying (A/B measurement).
class DevicePool {
 public:
  // A pool serves exactly ONE device, named at construction. Resolve one with
  // Pool(b) / AuxPool(b) rather than building your own; a directly-constructed
  // pool is for tests that want an isolated free list.
  explicit DevicePool(vt::Backend& b) : backend_(&b) {}
  DevicePool(const DevicePool&) = delete;
  DevicePool& operator=(const DevicePool&) = delete;
  static bool SupportsCapture() { return !Bypass(); }

  // BEST-FIT BORROW (#1922). The guarantee a caller gets, stated as a ratio:
  // while it holds a block borrowed from a larger class it holds at most this
  // many times the bytes it asked for.
  //
  // IT IS NOT WHAT STOPS THE LOOP, and this comment used to argue the opposite.
  // The `probe > limit` test in `Get` is unreachable for every request of at
  // least `2^kClassBits` bytes: the ladder has `2^kClassBits` rungs per octave,
  // so `kBorrowMaxSteps` steps land EXACTLY on `kBorrowMaxRatio * key` —
  // including across an octave boundary, where the rung width doubles and the
  // steps left cover the same distance. `probe == limit` is not `probe >
  // limit`, so the step budget always runs out first and the ratio never fires.
  // Measured on this tree: raising this constant from 2 to 16 ALONE left the
  // focused gate 12/12 `SUCCESS`, and only ratio 16 together with
  // `kBorrowMaxSteps` 64 moved anything.
  //
  // So the two constants are one bound written from two ends, and the
  // `static_assert` beside `kBorrowMaxSteps` is what makes that true rather
  // than hoped for: edit either alone and this header stops compiling.
  //
  // PUBLIC so a caller can read the guarantee it is given. A memory gate
  // asserts the LITERAL bound instead of naming this constant, because an
  // assertion written against the constant widens itself when the constant is
  // raised — the tautology shape `.agents/verification.md` names.
  static constexpr size_t kBorrowMaxRatio = 2;

  // Size-class rounding is `private static`, and `tests/vt/test_cpu_isa_x86.cpp`
  // exercises it directly (including the overflow throw) without a backend to
  // build a pool on — this is that seam. It is deliberately `static`, so binding
  // a pool to a device did not change it.
  static size_t SizeClassForTest(size_t bytes) { return ClassOf(bytes); }

  void* Get(vt::Backend& b, size_t bytes) {
    RequireOwnDevice(b, "Get");
    // BYPASS lane (VT_POOL_BYPASS=1) — the pool is a DETECTOR BLIND SPOT and
    // this is how you see through it. Two ways it hides a real defect from
    // compute-sanitizer:
    //   1. size-class rounding hands back a block up to 6.25% LARGER than the
    //      logical tensor, so a write past the last row lands inside the same
    //      driver allocation and memcheck reports nothing;
    //   2. blocks are never returned to the driver, so a use-after-free of a
    //      released DBuf reads memory that is still legally mapped — and, worse,
    //      may already have been handed to an unrelated op.
    // Under bypass every Get is an EXACT-size driver allocation and every Put is
    // a real Free, which restores both boundaries for the detector. It is a
    // debugging lane only: it reinstates the per-op cudaMalloc/cudaFree sync
    // storm this pool exists to remove, so it is never a timing configuration.
    if (Bypass()) return b.Alloc(bytes);
    const size_t key = ClassOf(bytes);
    void* hit = nullptr;
    {
      std::lock_guard<std::mutex> lk(mu_);
      ClassState& cs = classes_[key];
      ++cs.live;
      if (cs.live - cs.base > cs.peak) cs.peak = cs.live - cs.base;
      if (!cs.free.empty()) {
        hit = cs.free.back();
        cs.free.pop_back();
        retained_ -= key;
        block_class_[hit] = key;
        ++hits_;
      } else {
        // BEST FIT OVER THE RETAINED POOL (#1922). A block held free in a LARGER
        // class already satisfies this request, and refusing to lend it is what
        // made retention a function of how many distinct shapes the traffic has
        // shown rather than of how much one step concurrently needs. Measured on
        // this tree: twelve sequential requests through `LoadedEngine::generate`,
        // the LARGEST one first so every later request demanded strictly less,
        // still grew the process heap from 1.71 MiB to 4.24 MiB — every buffer
        // the later requests needed had already been allocated and returned, and
        // none of it could be reused because a freed block could only ever serve
        // its own class.
        //
        // MIRROR. torch's caching allocator, which is the allocator vLLM's
        // activations come out of, searches its cached pool for the SMALLEST
        // block at least as large as the request before it asks the driver
        // (`c10/cuda/CUDACachingAllocator.cpp::get_free_block`). This is that
        // search, over the class ladder instead of over a sorted block set, and
        // it is the one structural difference that made a bounded upstream
        // working set unbounded here.
        //
        // The borrow is BOUNDED at kBorrowMaxRatio, so a caller never holds more
        // than twice the bytes it asked for while it holds a borrowed block. The
        // line that DELIVERS that bound is the `kBorrowMaxSteps` budget, not the
        // `probe > limit` test beside it — see `kBorrowMaxRatio`, which is the
        // same bound written from the other end and is `static_assert`ed against
        // this budget. The `limit` test is kept because it is what makes the
        // guarantee hold for a request below `2^kClassBits` bytes, where the
        // ladder keys exactly and 16 rungs is more than one octave.
        //
        // Upstream bounds the same waste with `kMaxSplitSize` plus its
        // small/large pool split; we cannot split a driver allocation, so this
        // pair is the whole bound.
        //
        // The block keeps its OWN class: `block_class_` records what the driver
        // actually allocated, and `Put` returns it there. A borrow is therefore a
        // loan and never a demotion — the large class gets its block back and can
        // still serve a large request — which is what keeps the borrow from
        // starving the class it came from.
        if (BorrowEnabled()) {
          size_t probe = key;
          const size_t limit =
              (key > std::numeric_limits<size_t>::max() / kBorrowMaxRatio)
                  ? key
                  : key * kBorrowMaxRatio;
          for (int step = 0; step < kBorrowMaxSteps; ++step) {
            probe = NextClassAbove(probe);
            if (probe == 0 || probe > limit) break;
            auto it = classes_.find(probe);
            if (it == classes_.end() || it->second.free.empty()) continue;
            hit = it->second.free.back();
            it->second.free.pop_back();
            retained_ -= probe;
            block_class_[hit] = probe;
            ++hits_;
            break;
          }
        }
        if (hit == nullptr) ++misses_;
      }
    }
    if (hit != nullptr) {
      // The block changes tenants here: whatever the previous DBuf left at
      // `hit` is dead. Backends that keep per-pointer residency must drop it —
      // a fresh Alloc would have registered the block anew, and a pool HIT
      // bypasses Alloc entirely (Tenstorrent keys its f32 device shadows on
      // the host pointer; a stale shadow was downloaded into an unrelated
      // tensor, #1715). Applies to BORROWED blocks identically: a loan from a
      // larger class had its own previous tenant. Outside the pool mutex: the
      // notification takes the backend's own locks.
      b.OnScratchBlockAcquired(hit);
      return hit;
    }
    void* fresh = nullptr;
    try {
      fresh = b.Alloc(key);
    } catch (...) {
      // The block was never handed out, so it must not count as live: a Get that
      // threw has no matching Put, and a leaked `live` would make every later
      // demand profile read one block too high for this class forever.
      std::lock_guard<std::mutex> lk(mu_);
      ClassState& cs = classes_[key];
      if (cs.live > 0) --cs.live;
      throw;
    }
    {
      std::lock_guard<std::mutex> lk(mu_);
      block_class_[fresh] = key;
    }
    return fresh;
  }
  // ── THE LEASE (cross-step carriers) ──────────────────────────────────────
  // `DBuf::ReleaseShared()` registers the block it hands to a shared_ptr here,
  // and the carrier's deleter returns it through `PutLeased`, which erases the
  // lease and then performs the ordinary uncapped return. A lease is one
  // pointer per outstanding carrier — bounded by live cross-step hand-offs —
  // and it exists so the pool can tell a legal return from a SECOND owner's
  // return of a live block, which is the aliasing the kolibri-1 dequant-cache
  // experiment observed (`POOL DOUBLE-HAND-OUT`) and which this set makes
  // refuse-able instead of silent.
  //
  // The registration happens on ReleaseShared, not on Get: a plain scratch
  // DBuf has exactly the one owner its destructor models, and only a block
  // whose ownership leaves the DBuf lifetime needs the extra record.
  void NoteLease(void* p) {
    if (p == nullptr) return;
    std::lock_guard<std::mutex> lk(mu_);
    leased_.insert(p);
  }

  // The carrier deleter's return: discharge the lease, then the ordinary
  // uncapped Put. A carrier whose block was never leased here — or whose lease
  // was already discharged — is the same double-owner defect seen from the
  // carrier's side, and is refused by name.
  void PutLeased(vt::Backend& b, size_t bytes, void* p) {
    RequireOwnDevice(b, "PutLeased");
    if (!DischargeLease(p)) throw std::logic_error(LeaseMessage(p, "PutLeased"));
    Put(b, bytes, p);
  }

  // Uncapped retention (deliberately-retained cross-step buffers: the device
  // logits / MTP hidden handed off via a shared_ptr deleter). Bytes are always
  // returned to the free list — the cross-step buffers are not cap-evicted.
  //
  // This used to take no backend at all, which is how ~28 copy-pasted
  // `shared_ptr` deleters came to name neither the device nor the pool: they
  // closed over a byte count and called `Pool().Put(alloc, q)`, so a block from
  // ANY device was returned to the one global pool. The aux-stream half of that
  // was LATENT rather than live: the deleter would have returned an AuxPool
  // block to the main pool, but no `Release()` site sat under an
  // `ActivePoolScope` — the four scope regions are leaf-ward of all nine of
  // them — so the wrong-pool return was reachable only by adding a site, which
  // is precisely the mistake that then costs a debugging campaign.
  // `DBuf::ReleaseShared()` is now the only way to build that carrier and it
  // captures the buffer's own pool and backend, so neither half can come back
  // (#516).
  void Put(vt::Backend& b, size_t bytes, void* p) {
    RequireOwnDevice(b, "Put");
    // THE LEASE GUARD (ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2). A block whose
    // `DBuf::ReleaseShared()` carrier is still alive is OWNED by that carrier;
    // the only return the pool accepts for it is the carrier's own, which goes
    // through `PutLeased` below and erases the lease first. Any other Put of a
    // leased block is a second owner returning a live block — the kolibri-1
    // dequant-cache defect: the block entered the free list, the next `Get`
    // handed it to a new owner, and the carrier's bytes became another
    // tensor's data (`POOL DOUBLE-HAND-OUT`). Before this guard the second
    // return was indistinguishable from a legal one and was accepted in
    // silence; now it is a hard runtime throw in BOTH lanes — under bypass the
    // same Put would free a live block to the driver, the louder spelling of
    // the same defect. The check runs before the bypass branch so the refusal
    // does not depend on the lane.
    if (TakeLeaseRefusal(p)) throw std::logic_error(LeaseMessage(p, "Put"));
    // Bypass: free for real so a later use-after-free traps.
    if (Bypass()) {
      b.Free(p);
      return;
    }
    const size_t key = ClassOf(bytes);
    std::lock_guard<std::mutex> lk(mu_);
    NoteReturned(key);
    // The DEMAND accounting above is the requesting class's; the FREE LIST is
    // the block's own (#1922). A borrowed block returns to the class the driver
    // allocated it at, never to the smaller class that borrowed it.
    const size_t owner = TakeBlockClass(p, key);
    retained_ += owner;
    classes_[owner].free.push_back(p);
  }
  // Cap-aware retention for the high-frequency forward scratch (DBuf). The soft
  // cap comes from the platform's residency_policy().device_pool_cap_bytes
  // (BACKEND-PLATFORM item 2). cap == 0 is UNCAPPED — the identical fast path as
  // the uncapped Put above and thus behavior-preserving on GB10 (cap 0 today).
  // When a discrete GPU sets a bound, scratch over the cap is freed to the driver
  // rather than pooled, so the reuse pool self-limits without a model edit.
  void Put(vt::Backend& b, size_t bytes, void* p, size_t cap) {
    RequireOwnDevice(b, "Put");
    // The lease guard, on the capped overload identically — see the uncapped
    // Put above.
    if (TakeLeaseRefusal(p)) throw std::logic_error(LeaseMessage(p, "Put"));
    if (Bypass()) {
      b.Free(p);
      return;
    }
    const size_t key = ClassOf(bytes);
    {
      std::lock_guard<std::mutex> lk(mu_);
      NoteReturned(key);
      // Same split as the uncapped Put: demand is the requester's class, the
      // free list is the block's own (#1922).
      const size_t owner = TakeBlockClass(p, key);
      if (cap == 0 || retained_ + owner <= cap) {
        retained_ += owner;
        classes_[owner].free.push_back(p);
        return;
      }
    }
    b.Free(p);  // over the soft cap: to the driver, outside the lock
  }

  // ── Retention accounting (#1922) ──────────────────────────────────────────
  // What this pool has asked the DRIVER for, and what it is holding free.
  //
  // `misses` is the gate-relevant number and the reason this accessor exists.
  // Bytes on the heap are the sum of everything in the process, so a memory
  // gate written against them measures every unrelated shape-keyed cache in the
  // tree as well and needs a tolerance nobody can justify. `misses` counts
  // exactly one thing: a request this pool could not serve from what it already
  // held, and therefore asked the backend for. A server that has already served
  // its largest request must stop making them, and before #1922 it did not.
  //
  // `VT_POOL_STATS` prints hits and misses at destruction, which answers the
  // same question one request too late to gate on.
  struct Stats {
    uint64_t hits = 0;
    uint64_t misses = 0;  // driver allocations this pool has made
    size_t retained_bytes = 0;
    size_t classes = 0;
    size_t live_blocks = 0;
  };
  Stats stats() const {
    std::lock_guard<std::mutex> lk(mu_);
    Stats s;
    s.hits = hits_.load();
    s.misses = misses_.load();
    s.retained_bytes = retained_;
    s.classes = classes_.size();
    s.live_blocks = block_class_.size();
    return s;
  }

  // Release every RETAINED block back to the driver, and report the bytes freed.
  //
  // The pool earns its keep WITHIN a phase, where the same size classes recur
  // every step and cudaMalloc/cudaFree would be a sync storm. Across a PHASE
  // CHANGE the retained classes are the wrong shapes for what comes next, so on
  // an UNCAPPED pool (`device_pool_cap_bytes == 0`, which is GB10/Thor today)
  // they are pure headroom loss at exactly the moment the next phase wants its
  // own working set. Draining at that boundary costs one cudaFree per retained
  // block, once, and is what keeps a big-canvas MiniMax-H3 VAE decode from
  // meeting the driver's OOM on top of 50 steps of denoise scratch.
  //
  // SAFETY: a class's free list only ever holds blocks a DBuf already returned, so nothing
  // live is touched. Under VT_POOL_BYPASS the free list is always empty (Put
  // frees straight through) and this is a no-op. And because a pool now holds
  // ONE device's blocks, `b.Free` is guaranteed to be the allocator that made
  // them — before #516 a drain could hand one device's block to another's Free.
  size_t Drain(vt::Backend& b) {
    RequireOwnDevice(b, "Drain");
    std::lock_guard<std::mutex> lk(mu_);
    size_t freed = 0;
    for (auto& entry : classes_) {
      for (void* p : entry.second.free) {
        b.Free(p);
        freed += entry.first;
      }
      entry.second.free.clear();
    }
    retained_ = 0;
    return freed;
  }

  // ── CAPTURE PRE-GROW (#1380) ───────────────────────────────────────────────
  // A `cudaMalloc` inside a captured region ABORTS the capture, so every
  // allocation the captured forward makes has to be a pool HIT. Warming the pool
  // by running the same forward eagerly first is necessary and NOT sufficient,
  // and the missing half is what #1380 measured: the capture RETAINS its
  // `[S, vocab]` logits, so one block never comes back to the free list, and the
  // NEXT capture at that shape is one block short. Since the pool is keyed by
  // SIZE CLASS rather than by tensor, the block the next capture then misses on
  // need not be the logits at all -- on `thor:gpu0` it was `dconv`, the GDN
  // causal-conv output (`qwen3_5.cpp:4665`), which at the gate's shape lands in
  // the SAME class as the retained logits.
  //
  // So the driver cannot pre-grow by naming one tensor. It asks the pool what
  // the step it just ran actually demanded, and asks for that to be servable.
  // `StepDemand` is the per-class PEAK number of blocks live at one time above
  // the baseline the last `MarkStepBoundary` recorded -- the transient demand,
  // not the retention -- so the profile is a property of the FORWARD and not of
  // whatever happened to be resident when it ran.
  using StepDemand = std::vector<std::pair<size_t, int64_t>>;

  // Close the current step for demand accounting. The next step's peak is
  // measured from the blocks that are live right now.
  void MarkStepBoundary() {
    std::lock_guard<std::mutex> lk(mu_);
    for (auto& e : classes_) {
      e.second.peak = 0;
      e.second.base = e.second.live;
    }
  }

  // The per-class transient demand measured since the last `MarkStepBoundary`.
  // A driver takes this at the end of the EAGER step it runs at a shape and
  // hands it back at that shape's capture, which is why it is a value rather
  // than pool state: two shapes interleave through one pool, and last-step state
  // would answer for whichever step ran most recently instead of for this one.
  StepDemand StepDemandProfile() const {
    std::lock_guard<std::mutex> lk(mu_);
    StepDemand out;
    out.reserve(classes_.size());
    for (const auto& e : classes_)
      if (e.second.peak > 0) out.emplace_back(e.first, e.second.peak);
    return out;
  }

  // Make the free list able to serve `demand` without touching the driver, and
  // report how many blocks that cost. Call it OUTSIDE the captured region; a
  // block created here is a `cudaMalloc` and would abort a capture in progress.
  // Idempotent, and a no-op once the free list is deep enough — which is the
  // steady state, so a warm server pays nothing for it.
  //
  // IT IGNORES THE SOFT CAP, AND UNDER A NON-ZERO ONE THAT REOPENS #1380 BY THE
  // OTHER API. This function adds to `retained_` without consulting `cap`,
  // while `Put` above frees a returned block STRAIGHT TO THE DRIVER once
  // `retained_ + key > cap`. A `cudaFree` inside a captured region aborts the
  // capture exactly as a `cudaMalloc` does — and `Put` is called from INSIDE the
  // captured forward, every time a scratch `DBuf` there goes out of scope. So a
  // pre-grow that pushed `retained_` over the cap would arm the abort it exists
  // to prevent, at the first block the capture returned.
  //
  // Nothing can reach that today and this is a hazard note, not a live defect:
  // every platform resolves `residency_policy().device_pool_cap_bytes` to 0,
  // and `cap == 0` is the uncapped fast path in `Put`. WHOEVER FIRST SETS A
  // NON-ZERO CAP OWES THE FIX HERE — either pass `cap` in and refuse to grow
  // past it, or make the captured region's `Put` cap-exempt. Recorded under
  // `## Owed` in `.agents/specs/eng-cudagraph-break.md`; owner row
  // `ENG-CUDAGRAPH-BREAK`.
  // #2274: THE COUNTERPART TO PreGrowForCapture, and the invariant it restores.
  //
  // A capture BAKES the addresses of the blocks it allocates. Those blocks are
  // freed back to the free list when `ForwardLayers` returns, and the replay then
  // writes through the baked pointers. `qwen3_5.cpp` states the assumption:
  // working scratch is "still SAFELY shared between the two graphs -- they replay
  // sequentially on one stream". That covers the two GRAPHS. It does not cover a
  // THIRD party allocating the same size class in between, which is what a newly
  // created DFlash2 draft store does when it takes a block for its block table --
  // and the next replay then writes f32 activations over that table.
  //
  // PreGrow guarantees `demand` free blocks per class BEFORE the capture; this
  // takes the same count back OUT of the free list AFTER it, so the blocks the
  // graph baked cannot be handed to anyone else while the graph lives. The
  // holder returns them on graph reset. Memory cost is one step's scratch per
  // live graph, which the graph was always going to need.
  std::vector<std::pair<size_t, void*>> PinForGraph(vt::Backend& b, const StepDemand& demand) {
    RequireOwnDevice(b, "PinForGraph");
    std::vector<std::pair<size_t, void*>> pinned;
    if (Bypass()) return pinned;  // no free list to pin out of
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& want : demand) {
      ClassState& cs = classes_[want.first];
      for (int64_t taken = 0; taken < want.second && !cs.free.empty(); ++taken) {
        void* p = cs.free.back();
        cs.free.pop_back();
        retained_ -= want.first;
        pinned.emplace_back(want.first, p);
      }
    }
    return pinned;
  }

  // Give the pinned blocks back when the graph that baked them is gone.
  void UnpinForGraph(vt::Backend& b, const std::vector<std::pair<size_t, void*>>& pinned) {
    // Empty FIRST: a release site is a destructor, and `RequireOwnDevice` throws.
    // A slot that never captured has nothing to give back, so it must not be able
    // to raise on the teardown path.
    if (pinned.empty()) return;
    RequireOwnDevice(b, "UnpinForGraph");
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& kv : pinned) {
      classes_[kv.first].free.push_back(kv.second);
      retained_ += kv.first;
    }
  }

  size_t PreGrowForCapture(vt::Backend& b, const StepDemand& demand) {
    RequireOwnDevice(b, "PreGrowForCapture");
    if (Bypass()) return 0;  // no free list to grow; every Get is a driver call
    std::vector<size_t> missing;
    {
      std::lock_guard<std::mutex> lk(mu_);
      for (const auto& want : demand) {
        ClassState& cs = classes_[want.first];
        for (int64_t have = static_cast<int64_t>(cs.free.size()); have < want.second; ++have)
          missing.push_back(want.first);
      }
    }
    for (size_t key : missing) {
      void* p = b.Alloc(key);
      std::lock_guard<std::mutex> lk(mu_);
      classes_[key].free.push_back(p);
      retained_ += key;
    }
    return missing.size();
  }

  ~DevicePool() {
    if (std::getenv("VT_POOL_STATS") != nullptr) {
      const uint64_t h = hits_.load(), m = misses_.load();
      const double rate = (h + m) ? 100.0 * static_cast<double>(h) / static_cast<double>(h + m) : 0.0;
      // The backend pointer identifies WHICH device's pool this line is about:
      // a mixed-backend process now prints one line per device, and two lines
      // with no way to tell them apart would be worse than one wrong line.
      std::fprintf(stderr,
                   "[DevicePool backend=%p] hits=%llu misses(cudaMalloc)=%llu hit-rate=%.2f%% "
                   "distinct-classes=%zu\n",
                   static_cast<const void*>(backend_),
                   static_cast<unsigned long long>(h), static_cast<unsigned long long>(m),
                   rate, classes_.size());
    }
  }

 private:
  // The device check, on EVERY pool operation. A hard runtime throw and NOT an
  // `assert`: the SACRED gate builds are Release/NDEBUG, where an assert is
  // compiled out and the pre-fix behavior — a block silently crossing devices —
  // would come straight back. It is one predictable compare against a member,
  // against a `cudaMalloc` this pool exists to avoid.
  //
  // The only way to reach it is an `ActivePoolScope` pointing at another
  // device's pool, which is precisely the mistake this row exists to make
  // impossible to make quietly.
  void RequireOwnDevice(vt::Backend& b, const char* op) const {
    if (&b == backend_) return;
    char msg[192];
    std::snprintf(msg, sizeof(msg),
                  "DevicePool::%s called with backend %p on a pool bound to backend %p: a scratch "
                  "block must never cross devices (see .agents/specs/pool-device-key.md, #516)",
                  op, static_cast<const void*>(&b), static_cast<const void*>(backend_));
    throw std::logic_error(msg);
  }

  // VT_POOL_BYPASS=1 turns every Get/Put into a raw driver Alloc/Free (see Get).
  // Read once: it must not change between an allocation and its matching free,
  // or a pooled block would be handed to Backend::Free (or a driver block leaked
  // into the free list).
  static bool Bypass() {
    static const bool on = [] {
      const char* e = std::getenv("VT_POOL_BYPASS");
      return e != nullptr && e[0] == '1';
    }();
    return on;
  }

  // How many leading significant bits a size class keeps, and therefore how
  // many rungs the ladder has per octave: `1 << kClassBits`. It is a class
  // member rather than a local of `ClassOf` because the borrow's step budget is
  // derived from it, and the `static_assert` below is what derives it.
  static constexpr int kClassBits = 4;  // <=6.25% over-allocation per class

  // Round `bytes` up so it keeps at most kClassBits leading significant bits.
  // Exact keying when VT_POOL_EXACT=1 (A/B). Small sizes (< 2^kClassBits) key
  // exactly — there are few of them and the waste would be proportionally large.
  static size_t ClassOf(size_t bytes) {
    static const bool exact = [] {
      const char* e = std::getenv("VT_POOL_EXACT");
      return e != nullptr && e[0] == '1';
    }();
    if (exact || bytes == 0) return bytes == 0 ? 1 : bytes;
    const int msb = static_cast<int>(std::bit_width(bytes)) - 1;
    if (msb < kClassBits) return bytes;
    const int shift = msb - kClassBits;
    const size_t mask = (static_cast<size_t>(1) << shift) - 1;
    if (bytes > std::numeric_limits<size_t>::max() - mask) {
      throw std::overflow_error("DevicePool size class rounding overflow");
    }
    return (bytes + mask) & ~mask;  // round up to a multiple of 2^shift
  }

  // How many rungs of the class ladder the borrow may climb, and the constraint
  // that ACTUALLY binds the loop — `kBorrowMaxRatio` says why. One octave is
  // `1 << kClassBits` rungs, so this budget is `log2(kBorrowMaxRatio)` octaves
  // of them, which is the same bound the ratio states.
  static constexpr int kBorrowMaxSteps = 16;

  // The two spellings of one bound, held together. Without this, raising
  // `kBorrowMaxRatio` alone changed NOTHING (the ratio test cannot fire first)
  // and raising `kBorrowMaxSteps` alone widened the borrow past the documented
  // guarantee with no gate anywhere — the ratio was a dead constant that a
  // memory gate was nevertheless asserting against.
  //
  // It compares THREE different constants, so it cannot degenerate into reading
  // `16 == 16`: edit any one of them on its own and this fails, naming which.
  static_assert(kBorrowMaxRatio >= 2 && std::has_single_bit(kBorrowMaxRatio),
                "DevicePool: kBorrowMaxRatio is an octave count and must be a power of two");
  static_assert(kBorrowMaxSteps ==
                    (static_cast<int>(std::bit_width(kBorrowMaxRatio)) - 1) * (1 << kClassBits),
                "DevicePool: kBorrowMaxSteps and kBorrowMaxRatio must state the SAME bound. "
                "One octave of the class ladder is (1 << kClassBits) rungs, so the step "
                "budget is log2(kBorrowMaxRatio) octaves of them. Change one and change both.");

  // The next class strictly above `k`. `ClassOf` is idempotent on a class key,
  // so the successor is the class of one byte more. Returns 0 on overflow,
  // which the caller reads as "no further rung".
  static size_t NextClassAbove(size_t k) {
    if (k == std::numeric_limits<size_t>::max()) return 0;
    try {
      return ClassOf(k + 1);
    } catch (const std::overflow_error&) {
      return 0;
    }
  }

  // VT_POOL_BORROW=0 restores the pre-#1922 exact-class-only reuse for a
  // same-binary A/B. Read once, for the reason `Bypass()` gives: it must not
  // change between an allocation and its matching free.
  static bool BorrowEnabled() {
    static const bool on = [] {
      const char* e = std::getenv("VT_POOL_BORROW");
      return !(e != nullptr && e[0] == '0');
    }();
    return on;
  }

  // One block of `key` came back: the requesting class's live count falls.
  // Separated from the free-list return because the two classes can differ
  // (#1922), and because holding a `ClassState&` across the `classes_[owner]`
  // insert below would be a reference into a container that can rehash.
  // Caller holds `mu_`.
  void NoteReturned(size_t key) {
    ClassState& cs = classes_[key];
    if (cs.live > 0) --cs.live;
  }

  // Caller holds no lock. True when the block is currently leased; the caller
  // turns the hit into the throw WITHOUT discharging, so the carrier's own
  // later return is still the accepted one.
  bool TakeLeaseRefusal(void* p) {
    if (p == nullptr) return false;
    std::lock_guard<std::mutex> lk(mu_);
    return leased_.count(p) != 0;
  }
  // Caller holds no lock. True when the lease existed and is now discharged.
  bool DischargeLease(void* p) {
    if (p == nullptr) return false;
    std::lock_guard<std::mutex> lk(mu_);
    return leased_.erase(p) != 0;
  }
  std::string LeaseMessage(void* p, const char* op) {
    // A per-call string, not a member buffer: two defect-path Puts on
    // different threads format concurrently, and a shared buffer would race.
    // The string is copied into the exception before the statement ends, so
    // a local outlives the throw.
    char buf[192];
    std::snprintf(buf, sizeof(buf),
                  "DevicePool::%s: block %p is still leased (its DBuf::ReleaseShared carrier "
                  "is alive) — a second return of a live block is the POOL DOUBLE-HAND-OUT "
                  "defect, not a legal return",
                  op, static_cast<const void*>(p));
    return std::string(buf);
  }

  // The class the DRIVER allocated `p` at, and forget it. `fallback` covers a
  // block this pool never handed out through `Get` — which is not reachable
  // today, and is answered with the requesting class rather than with a throw
  // so that a future direct-insert path degrades to the pre-#1922 behaviour
  // instead of aborting a forward. Caller holds `mu_`.
  size_t TakeBlockClass(void* p, size_t fallback) {
    auto it = block_class_.find(p);
    if (it == block_class_.end()) return fallback;
    const size_t owner = it->second;
    block_class_.erase(it);
    return owner;
  }

  // One size class: its free blocks, and the demand accounting `StepDemand`
  // reads. `live` counts blocks handed out and not yet returned; `base` is what
  // `live` was at the last step boundary; `peak` is the largest `live - base`
  // since then, which is the number of blocks of this class a repeat of that
  // step needs on the free list.
  struct ClassState {
    std::vector<void*> free;
    int64_t live = 0;
    int64_t base = 0;
    int64_t peak = 0;
  };

  mutable std::mutex mu_;
  // THE DEVICE, and the reason this class exists in this shape. Every block in
  // a class's free list was allocated by this backend and will be freed by it;
  // nothing else may draw from or return to this pool.
  vt::Backend* backend_;
  std::unordered_map<size_t, ClassState> classes_;
  // The class the driver allocated each LIVE block at (#1922). One entry per
  // block currently handed out — the step's working set, not the history — so
  // it is bounded by concurrent liveness and empties itself as blocks return.
  // It exists because `Put` is told the caller's byte count and a borrowed
  // block's own size is not derivable from that.
  // Blocks whose ReleaseShared carrier is still alive. One entry per live
  // cross-step hand-off; the carrier's deleter erases it.
  std::unordered_map<void*, size_t> block_class_;
  std::unordered_set<void*> leased_;
  size_t retained_ = 0;  // bytes (class-rounded) held free, for the soft cap
  std::atomic<uint64_t> hits_{0};
  std::atomic<uint64_t> misses_{0};
};

namespace detail {

// Process-wide table of per-device pools. Tiny by construction: one entry per
// `vt::Backend*` the process ever allocates through, i.e. one per
// Device{type,index}. Entries are never erased, which is what lets Pool()'s
// memo below hold a raw pointer.
class PoolTable {
 public:
  DevicePool& For(vt::Backend& b) {
    std::lock_guard<std::mutex> lk(mu_);
    for (const auto& e : pools_)
      if (e.first == &b) return *e.second;
    pools_.emplace_back(&b, std::unique_ptr<DevicePool>(new DevicePool(b)));
    return *pools_.back().second;
  }

 private:
  std::mutex mu_;
  std::vector<std::pair<vt::Backend*, std::unique_ptr<DevicePool>>> pools_;
};

inline PoolTable& MainPoolTable() {
  static PoolTable t;
  return t;
}

// The (backend -> pool) resolution, memoized per thread. A DBuf resolves its
// pool on EVERY construction — thousands per forward step — and this pool's
// whole purpose is to avoid a synchronizing cudaMalloc, so paying a lock and a
// scan for it would be self-defeating. A process drives one device per host
// thread at a time, so the steady state here is a single pointer compare.
inline DevicePool& MemoizedPool(PoolTable& table, vt::Backend& b,
                                vt::Backend*& last_backend, DevicePool*& last_pool) {
  if (last_backend == &b) return *last_pool;
  DevicePool& p = table.For(b);
  last_backend = &b;
  last_pool = &p;
  return p;
}

}  // namespace detail

// THE scratch pool for a device. There is deliberately no no-argument spelling:
// "the pool" without a device is the defect (#516), not an ergonomic shortcut.
inline DevicePool& Pool(vt::Backend& b) {
  thread_local vt::Backend* last_backend = nullptr;
  thread_local DevicePool* last_pool = nullptr;
  return detail::MemoizedPool(detail::MainPoolTable(), b, last_backend, last_pool);
}

// --- Aux-stream scratch pool (ENG-MOE-SHARED-AUX) ----------------------------
// The MoE shared-expert overlap (MoeBlockFusedMarlinCuda) issues the shared MLP
// on a SECOND CUDA stream concurrent with the routed experts on the main stream.
// The main `Pool()` above is only reuse-safe under SINGLE-stream ordering: a
// block returned to the free list is handed back out on the same stream, so CUDA
// stream ordering guarantees the block's last op has completed before any reuse.
// Two streams sharing one pool BREAKS that invariant — a scratch block the aux
// stream freed (e.g. the shared gate/up transient) could be handed to a routed
// GEMM on the main stream and written WHILE the aux kernel still reads it (a
// cross-stream RAW/WAR race compute-sanitizer flags). vLLM sidesteps this with
// its STREAM-AWARE caching allocator (record_stream); we keep the simple pool and
// instead give the aux stream its OWN pool. Each pool then only ever serves ONE
// stream, so single-stream ordering holds within it and the two streams never
// share a live block. Blocks are handed back to the pool they came from (DBuf
// stores its owning pool), so a buffer allocated in the aux region and destroyed
// after the join still returns to the aux pool.
// The aux pool is per-device too: the stream distinction and the device
// distinction are independent, and a process with two devices running the MoE
// overlap needs one aux pool per device, not one shared between them.
#ifdef VT_MARLIN_NVFP4  // only the Marlin MoE overlap path draws from AuxPool
namespace detail {
inline PoolTable& AuxPoolTable() {
  static PoolTable t;
  return t;
}
}  // namespace detail
inline DevicePool& AuxPool(vt::Backend& b) {
  thread_local vt::Backend* last_backend = nullptr;
  thread_local DevicePool* last_pool = nullptr;
  return detail::MemoizedPool(detail::AuxPoolTable(), b, last_backend, last_pool);
}
#endif

// Thread-local OVERRIDE of the scratch pool a DBuf constructs from. Null means
// "this device's own Pool(b)" — the default, and the only correct default, since
// a thread-local cannot know which device the next DBuf will be built on. The
// aux-stream overlap region swaps it to AuxPool(b) for the duration of the
// shared-expert issue via ActivePoolScope. Single host thread drives the
// forward, and the aux ops are issued in one contiguous block, so the swap is a
// simple RAII stack.
//
// An override pointing at ANOTHER device's pool is no longer a silent
// corruption: the pool checks its backend on every operation and throws.
inline DevicePool*& ActivePoolOverride() {
  thread_local DevicePool* p = nullptr;
  return p;
}
inline DevicePool& ActivePool(vt::Backend& b) {
  DevicePool* const override_pool = ActivePoolOverride();
  return override_pool != nullptr ? *override_pool : Pool(b);
}
struct ActivePoolScope {
  DevicePool* prev;
  explicit ActivePoolScope(DevicePool* p) : prev(ActivePoolOverride()) {
    ActivePoolOverride() = p;
  }
  ~ActivePoolScope() { ActivePoolOverride() = prev; }
  ActivePoolScope(const ActivePoolScope&) = delete;
  ActivePoolScope& operator=(const ActivePoolScope&) = delete;
};

}  // namespace vllm
