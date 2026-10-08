// vllm.cpp original (the shared scratch pool is a vt-runtime deviation, porting
// inventory §9.1); vLLM has no mirror because it never had this design — its
// allocation handle carries the device as field 0
// (`vllm/device_allocator/__init__.py:12-14` @ pin 555967922) and torch's cache
// is per-device by construction (`c10/cuda/CUDACachingAllocator.h:118-172`).
//
// THE GATE FOR #516 (.agents/specs/pool-device-key.md): `vllm::Pool()` was a
// process-wide free list keyed by BYTE SIZE CLASS ONLY. The device was not in
// the key, so a block allocated through one backend was handed to a `DBuf`
// running on another. One fault, two symptoms, chosen by direction:
//
//   cudaMalloc block -> CPU-backend forward : SIGSEGV, compute-sanitizer CLEAN
//                                             (the fault is host-side)
//   host block       -> CUDA forward        : SILENT all-NaN output
//
// This file is the DIRECT gate. It needs no GPU, no checkpoint and no NAS: two
// distinguishable fake backends stand in for two devices (the technique
// tests/vt/test_backend_multidevice.cpp and tests/vt/test_reference_tier.cpp
// already use), and every case goes through `dense_attn::DBuf` — the seam every
// production forward allocates from — so it holds the same path the LTX-2.5
// device suite crashes on rather than a paraphrase of it.
//
// Two test-side properties are load-bearing and neither is a style choice.
// (1) A fake backend NEVER returns a block to the C allocator: these cases
// compare pointer identity ACROSS a free, and a freed pointer is not a value you
// may reason about. (2) A fake backend is never destroyed before exit, so no
// stack or heap address is ever reused — the pool is keyed on the backend's
// identity, and an address reused by a later case would make one case's pool
// answer another case's question.
//
// THE TWO DEBUG LANES ARE GREEN HERE, ON PURPOSE. `VT_POOL_BYPASS=1` removes the
// free list and `VT_POOL_EXACT=1` removes size-class rounding; the spec's own
// §10 hands `VT_POOL_BYPASS=1` to whoever picks up the unattributed dgx failures
// as "the cheap discriminator that needs no second build". A suite that reds
// under the very lane it tells you to use is a suite that costs its next reader
// an hour deciding whether the red is theirs. So: a case whose subject is REUSE
// states what the ACTIVE lane does — a pooled hit by default, a fresh driver
// block under bypass — and the case whose subject is the SIZE CLASS states
// sharing by default and SEPARATION under `VT_POOL_EXACT` (spec §5 T1.3's
// second clause, which had no assertion until now). Both env vars are read ONCE
// into a function-local static inside `DevicePool`, so a process is in exactly
// one lane for its whole life and no case can toggle them.
#include <doctest/doctest.h>

#include <atomic>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <set>
#include <stdexcept>
#include <thread>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"

namespace {

using vllm::dense_attn::DBuf;
using vllm::dense_attn::Dev;
using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;

// A host-memory backend that remembers WHICH allocations are its own, so a case
// can ask the question that matters — "did this block come from THIS device?" —
// instead of inferring it from a pointer that merely happens to differ.
//
// ITS OWN BOOKKEEPING IS LOCKED, and that is a correctness requirement of the
// concurrent case at the bottom of this file rather than defensive habit.
// `DevicePool::Get` calls `Alloc` OUTSIDE its own mutex, on purpose — a driver
// allocation is the slow path and holding the pool lock across it would
// serialise every other class. So two threads really do enter `Alloc` at once,
// and an unlocked `owned_.push_back` would be a data race in the FIXTURE.
// ThreadSanitizer would then report this file instead of its subject, which is
// the failure mode where a broken instrument returns a verdict about the code.
class TagBackend final : public Backend {
 public:
  ~TagBackend() override {
    for (void* p : owned_) std::free(p);
  }
  void* Alloc(size_t bytes) override {
    void* p = std::malloc(bytes == 0 ? 1 : bytes);
    std::lock_guard<std::mutex> lk(mu_);
    owned_.push_back(p);
    ++allocs_;
    return p;
  }
  // Deliberately does NOT std::free: see the file header. The block stays valid
  // and stays owned; only the fact of the Free is recorded.
  void Free(void* p) override {
    std::lock_guard<std::mutex> lk(mu_);
    freed_.push_back(p);
    ++frees_;
  }
  void Memset(Queue&, void* p, int v, size_t bytes) override { std::memset(p, v, bytes); }
  void Copy(Queue&, void* dst, const void* src, size_t bytes) override {
    std::memcpy(dst, src, bytes);
  }
  Queue CreateQueue() override { return Queue{}; }
  bool UnifiedMemory() const override { return true; }

  bool Owns(const void* p) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const void* q : owned_)
      if (q == p) return true;
    return false;
  }
  bool WasFreed(const void* p) const {
    std::lock_guard<std::mutex> lk(mu_);
    for (const void* q : freed_)
      if (q == p) return true;
    return false;
  }
  int allocs() const {
    std::lock_guard<std::mutex> lk(mu_);
    return allocs_;
  }
  int frees() const {
    std::lock_guard<std::mutex> lk(mu_);
    return frees_;
  }

 private:
  mutable std::mutex mu_;
  std::vector<void*> owned_;
  std::vector<void*> freed_;
  int allocs_ = 0;
  int frees_ = 0;
};

// Process-lifetime fakes: see the file header, property (2).
TagBackend& NewBackend() {
  static std::vector<std::unique_ptr<TagBackend>> keep;
  keep.push_back(std::make_unique<TagBackend>());
  return *keep.back();
}

// Two devices of the same TYPE. `Device{type,index}` is exactly how the backend
// registry addresses discrete devices (vt/backend.h, kMaxDevicesPerType), and
// keeping the type equal keeps the platform lookup the DBuf constructor performs
// (`ResolveDevicePoolPolicy`) on a registered platform, so these cases test the
// pool and nothing else.
Queue QueueOn(int32_t index) {
  Queue q;
  q.device = Device{DeviceType::kCPU, index};
  q.handle = nullptr;
  return q;
}

// The two debug lanes, spelled EXACTLY as `DevicePool::Bypass()` and
// `DevicePool::ClassOf()` spell them ("=1", first character only), so this file
// cannot disagree with the header about which lane a process is in.
bool PoolBypass() {
  const char* e = std::getenv("VT_POOL_BYPASS");
  return e != nullptr && e[0] == '1';
}
bool PoolExact() {
  const char* e = std::getenv("VT_POOL_EXACT");
  return e != nullptr && e[0] == '1';
}

}  // namespace

// ═══════════════════════════════════════════════════════════════════════════
// THE DEFECT. A block freed on device 0 must never be handed to device 1.
//
// Every case below uses its OWN size class, so no case can be decided by what an
// earlier one left in a free list — including under `--order-by=rand`.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("device pool: a block freed on one device is NEVER handed to another") {
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  Queue qb = QueueOn(1);

  // Identical shape and dtype on both devices, so both land in the SAME size
  // class. That is the whole precondition: at differing size classes the two
  // arms never trade blocks, which is why the f32 LTX-2.5 arms could not reach
  // this and the bf16 ones could.
  const std::vector<int64_t> shape{1024};  // 4096 bytes
  void* on_a = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, shape);
    on_a = x.ptr();
  }  // returned to device 0's free list here

  void* on_b = nullptr;
  {
    DBuf y(Dev{b, qb}, DType::kF32, shape);
    on_b = y.ptr();
  }

  REQUIRE(on_a != nullptr);
  REQUIRE(on_b != nullptr);
  // The pointer identity IS the defect: before the device entered the key these
  // were the same block, and device 1 then wrote through device 0's allocation.
  CHECK(on_b != on_a);
  // ...and the stronger statement that identity check stands in for: each block
  // came out of its OWN device's allocator.
  CHECK(a.Owns(on_a));
  CHECK(b.Owns(on_b));
  CHECK_FALSE(b.Owns(on_a));
  CHECK_FALSE(a.Owns(on_b));
}

// The same ordering the other way round. The direction decides the SYMPTOM (a
// host block reaching a device forward is the silent-NaN direction; a device
// block reaching a host forward is the SIGSEGV one), so a fix that separates
// them in only one direction is not a fix.
TEST_CASE("device pool: the reverse direction is separated too") {
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  Queue qb = QueueOn(1);
  const std::vector<int64_t> shape{4096};  // 8192 bytes @ bf16

  void* on_b = nullptr;
  {
    DBuf y(Dev{b, qb}, DType::kBF16, shape);
    on_b = y.ptr();
  }
  void* on_a = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kBF16, shape);
    on_a = x.ptr();
  }
  CHECK(on_a != on_b);
  CHECK(b.Owns(on_b));
  CHECK(a.Owns(on_a));
  CHECK_FALSE(a.Owns(on_b));
}

// ═══════════════════════════════════════════════════════════════════════════
// ...AND THE POOL MUST STILL BE A POOL. Without the two cases below, "fixed"
// would be indistinguishable from `VT_POOL_BYPASS=1`, which also passes every
// case above and reinstates the per-op cudaMalloc/cudaFree sync storm the pool
// exists to remove. The fix has to separate the devices WITHOUT ending reuse.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("device pool: reuse on ONE device still returns the identical block") {
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  const std::vector<int64_t> shape{4096};  // 16384 bytes @ f32

  void* first = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, shape);
    first = x.ptr();
  }
  const int allocs_after_first = a.allocs();
  void* second = nullptr;
  {
    DBuf y(Dev{a, qa}, DType::kF32, shape);
    second = y.ptr();
  }
  if (PoolBypass()) {
    // The bypass lane HAS no free list — every Get is an exact-size driver
    // Alloc and every Put a real Free, which is the whole reason it exists (it
    // restores the allocation boundaries compute-sanitizer needs). Reuse is
    // therefore the wrong expectation here, not a regression, and asserting the
    // lane's own behavior is what keeps the lane usable as a discriminator.
    CHECK(second != first);
    CHECK(a.allocs() == allocs_after_first + 1);
    CHECK(a.WasFreed(first));
  } else {
    CHECK(second == first);                   // a pool HIT...
    CHECK(a.allocs() == allocs_after_first);  // ...proven by the allocator counter
  }
}

TEST_CASE("device pool: size-class rounding still lets nearby sizes share a block") {
  // 32,400 and 32,768 bytes round to the same class (kClassBits=4 keeps the top
  // four significant bits), which is what makes a prefill of a different token
  // count a pool hit instead of a synchronous cudaMalloc. Preserved
  // deliberately: `VT_POOL_EXACT=1` (reuse kept, rounding removed) was MEASURED
  // still red, so the rounding is not the fault and is not what this row
  // changes.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);

  void* first = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, {8100});  // 32,400 bytes
    first = x.ptr();
  }
  const int allocs_after_first = a.allocs();
  void* second = nullptr;
  {
    DBuf y(Dev{a, qa}, DType::kF32, {8192});  // 32,768 bytes, same class
    second = y.ptr();
  }
  const size_t lo_class = vllm::DevicePool::SizeClassForTest(32400);
  const size_t hi_class = vllm::DevicePool::SizeClassForTest(32768);
  if (PoolBypass()) {
    CHECK(second != first);  // no free list at all, so nothing to share
  } else if (PoolExact()) {
    // Spec §5 T1.3's second clause, "`VT_POOL_EXACT` still separates them",
    // which had no assertion anywhere until this one. Exact keying is the A/B
    // arm that MEASURED the rounding innocent (still red), so it has to keep
    // being a real second behavior and not merely a variable nobody reads.
    CHECK(lo_class == 32400);
    CHECK(hi_class == 32768);
    CHECK(lo_class != hi_class);
    CHECK(second != first);
    CHECK(a.allocs() == allocs_after_first + 1);
  } else {
    CHECK(lo_class == hi_class);  // ...one class...
    CHECK(lo_class == 32768);     // ...and it is the larger of the two
    CHECK(second == first);
    CHECK(a.allocs() == allocs_after_first);
  }
}

// ═══════════════════════════════════════════════════════════════════════════
// The three properties that follow from "a pool is bound to one device", each
// of which was silently wrong before and none of which the DBuf cases above
// would catch on their own.
// ═══════════════════════════════════════════════════════════════════════════

TEST_CASE("device pool: Drain frees ONE device's blocks, through ITS OWN backend") {
  // Before the device entered the key there was one free list, so the MiniMax-H3
  // phase-change drain (minimax_h3_pipeline.cpp) handed every retained block to
  // whichever backend it was called with — including another device's blocks, to
  // an allocator that never made them.
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  Queue qb = QueueOn(1);
  const std::vector<int64_t> shape{16384};  // 65,536 bytes @ f32

  void* held_by_b = nullptr;
  {
    DBuf x(Dev{a, qa}, DType::kF32, shape);
  }
  {
    DBuf y(Dev{b, qb}, DType::kF32, shape);
    held_by_b = y.ptr();
  }
  if (PoolBypass()) {
    // Bypass frees straight through, so both destructors already returned their
    // block to their OWN backend and there is nothing retained to drain. That is
    // the same device-scoped property this case is about, enforced by the
    // absence of a free list rather than by the key — worth stating, because it
    // is the arm the spec sends a reader to when a drain is under suspicion.
    CHECK(a.frees() == 1);
    CHECK(b.frees() == 1);
    CHECK(vllm::Pool(a).Drain(a) == 0);
    return;
  }
  REQUIRE(a.frees() == 0);
  REQUIRE(b.frees() == 0);

  const size_t drained = vllm::Pool(a).Drain(a);
  CHECK(drained > 0);
  CHECK(a.frees() == 1);
  CHECK(b.frees() == 0);  // device 1's retained block was NOT freed by device 0

  // ...and device 1's free list is intact: its next request of that class is
  // still the same block, i.e. the drain did not quietly empty it.
  {
    DBuf y(Dev{b, qb}, DType::kF32, shape);
    CHECK(y.ptr() == held_by_b);
  }
}

TEST_CASE("device pool: a pool bound to another device is REFUSED, not served") {
  // `ActivePoolScope` is the one remaining way to hand a DBuf a pool that is not
  // its device's. It is a legitimate seam — the aux CUDA stream uses it — so it
  // stays, and the pool checks the backend instead. A hard throw, not an
  // `assert`: the gate builds are Release/NDEBUG, where an assert is compiled
  // out and the silent cross-device hand-off would come straight back.
  TagBackend& a = NewBackend();
  TagBackend& b = NewBackend();
  Queue qa = QueueOn(0);
  vllm::DevicePool bs_pool(b);  // a pool that belongs to device 1

  const vllm::ActivePoolScope wrong(&bs_pool);
  CHECK_THROWS_AS(DBuf(Dev{a, qa}, DType::kF32, {64}), std::logic_error);
}

TEST_CASE("device pool: ReleaseShared returns the block to the pool it CAME FROM") {
  // The cross-step carrier (device logits, MTP hidden states, MoE scratch) used
  // to be built by hand at ~28 sites, from a deleter that closed over the byte
  // count ALONE and called `Pool().Put(alloc, q)`. So it returned the block to
  // the one global pool whatever device it came from — and, separately, whatever
  // POOL it came from, which silently drained the aux-stream pool into the main
  // one. `ReleaseShared()` captures both.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  const std::vector<int64_t> shape{12288};  // 49,152 bytes @ f32

  void* raw = nullptr;
  {
    std::shared_ptr<void> carrier;
    {
      DBuf x(Dev{a, qa}, DType::kF32, shape);
      raw = x.ptr();
      carrier = x.ReleaseShared();
    }
    // The DBuf is gone but the carrier holds the block: this device's free list
    // must NOT have it yet, so a same-class request allocates afresh.
    DBuf other(Dev{a, qa}, DType::kF32, shape);
    CHECK(other.ptr() != raw);
  }
  if (PoolBypass()) {
    // No free list, so the block does not come back. What the lane still proves
    // — and it is the half that matters — is that the carrier's deleter ran
    // against the buffer's OWN backend rather than an ambient one.
    CHECK(a.WasFreed(raw));
    return;
  }
  // Carrier dropped -> the block is back in THIS device's pool.
  {
    DBuf again(Dev{a, qa}, DType::kF32, shape);
    CHECK(again.ptr() == raw);
  }
}

TEST_CASE("device pool: a scoped pool's block returns to the SCOPED pool, not the device's") {
  // The aux-stream shape, which the hand-written deleters got wrong on every
  // path that used them: a block drawn under an ActivePoolScope and handed to a
  // shared_ptr must come back to the SCOPED pool. Returning it to the device's
  // main pool is how a second stream's block ends up in the first stream's free
  // list — the race AuxPool() exists to prevent.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  vllm::DevicePool scoped(a);  // same DEVICE, different pool
  const std::vector<int64_t> shape{24576};  // 98,304 bytes @ f32

  void* raw = nullptr;
  {
    std::shared_ptr<void> carrier;
    {
      const vllm::ActivePoolScope scope(&scoped);
      DBuf x(Dev{a, qa}, DType::kF32, shape);
      raw = x.ptr();
      carrier = x.ReleaseShared();
    }  // scope ends BEFORE the carrier is dropped, deliberately
  }

  if (PoolBypass()) {
    // Neither pool retains anything under bypass; the deleter freed the block
    // to the backend it was allocated from, which is all this lane can show.
    CHECK(a.WasFreed(raw));
    return;
  }
  // The device's main pool must not have acquired it...
  {
    DBuf from_main(Dev{a, qa}, DType::kF32, shape);
    CHECK(from_main.ptr() != raw);
  }
  // ...the scoped pool must have.
  {
    const vllm::ActivePoolScope scope(&scoped);
    DBuf from_scoped(Dev{a, qa}, DType::kF32, shape);
    CHECK(from_scoped.ptr() == raw);
  }
}

TEST_CASE("device pool: a backend whose PLATFORM is unregistered is REFUSED, not defaulted") {
  // The residency policy is memoized PER DEVICE TYPE now (spec §4 D5), where it
  // used to be one function-local static for the process. That closed the
  // ambient-device hole one layer above the pool, and it opened a new failure
  // mode with it: `platforms::GetPlatform` VT_CHECK-throws for an unregistered
  // type, so a DBuf on such a backend now throws where it previously inherited
  // whichever device happened to resolve FIRST. That is the correct answer — a
  // `device_pool_cap_bytes` read off another platform is a wrong number, not a
  // default — but a new throw with no test is a new throw nobody has run.
  //
  // kXPU is the type with no `RegisterPlatform` call anywhere in the tree (grep
  // src/vllm/platforms/*.cpp: cpu, cuda, rocm, vulkan, metal, tenstorrent). The
  // REQUIRE_FALSE states that as the precondition it is, so the day an XPU
  // platform lands this case goes RED and asks to be re-pointed rather than
  // quietly asserting nothing.
  REQUIRE_FALSE(vllm::platforms::HasPlatform(DeviceType::kXPU));
  TagBackend& a = NewBackend();
  Queue q;
  q.device = Device{DeviceType::kXPU, 0};
  q.handle = nullptr;
  CHECK_THROWS_AS(DBuf(Dev{a, q}, DType::kF32, {64}), std::runtime_error);
}

// ═══════════════════════════════════════════════════════════════════════════
// #1380: THE CAPTURE PRE-GROW SERVES THE STEP'S DEMAND, NOT ONE BLOCK.
//
// A `cudaMalloc` inside a captured region aborts the capture, so a decode-graph
// driver has to make every allocation the captured forward performs a pool HIT.
// Running the forward eagerly once first is necessary and not sufficient: the
// capture RETAINS its `[S, vocab]` logits, so one block never returns to the
// free list and the NEXT capture at that shape is one block short.
//
// `Qwen3_5DenseDecodeGraph` used to answer that by allocating and freeing ONE
// `[S, vocab]` block before `BeginCapture`. That reasons about a TENSOR while
// the pool reasons about a SIZE CLASS, and it is one block where the forward may
// hold several of that class live at once. Measured on `thor:gpu0` (sm_110): the
// second ring slot's capture of a speculative shape threw
// `cudaMalloc: operation not permitted when stream is capturing` inside `dconv`,
// the GDN causal-conv output, whose `[T, conv_dim]` bf16 lands in the same class
// as the retained `[S, vocab]` f32 logits at that shape, and the measured peak
// for the class was TWO.
//
// This case is that arithmetic with no GPU: one eager step holding two blocks of
// one class at once, one block retained the way a captured graph retains its
// logits, and then the pair of allocations the captured region would make.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("device pool: a capture pre-grow serves the STEP's demand, not one block") {
  if (PoolBypass()) {
    // The bypass lane has no free list at all — every Get is a driver Alloc —
    // so "the free list is deep enough" is not a question it can be asked. The
    // lane stays usable as a discriminator instead of reddening here.
    MESSAGE("SKIP: VT_POOL_BYPASS=1 removes the free list this case measures");
    return;
  }
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  vllm::DevicePool& pool = vllm::Pool(a);
  const std::vector<int64_t> shape{2048};  // 8192 bytes @ f32 — this case's class
  const size_t key = vllm::DevicePool::SizeClassForTest(8192);

  // ONE eager step at this shape. It holds TWO blocks of the class live at the
  // same time and returns both, which is exactly what a forward does when its
  // working scratch and its output land in one class.
  pool.MarkStepBoundary();
  {
    DBuf scratch(Dev{a, qa}, DType::kF32, shape);
    DBuf out(Dev{a, qa}, DType::kF32, shape);
    CHECK(scratch.ptr() != out.ptr());
  }
  const vllm::DevicePool::StepDemand demand = pool.StepDemandProfile();
  int64_t peak = 0;
  for (const auto& e : demand)
    if (e.first == key) peak = e.second;
  CHECK(peak == 2);  // the PROFILE, and it is the number the pre-grow needs

  // The previous capture retained its logits: one block leaves circulation for
  // good, exactly as `SizeSlot::logits` does. The free list is now ONE deep
  // against a demand of two, which is the whole defect.
  auto retained = std::make_unique<DBuf>(Dev{a, qa}, DType::kF32, shape);

  const int allocs_before = a.allocs();
  pool.PreGrowForCapture(a, demand);
  const int allocs_after_pregrow = a.allocs();
  CHECK(allocs_after_pregrow == allocs_before + 1);  // it really was one short

  // THE GATE. This is the captured region: two blocks of the class, live at once,
  // and NOT ONE of them may reach the driver.
  {
    DBuf x(Dev{a, qa}, DType::kF32, shape);
    DBuf y(Dev{a, qa}, DType::kF32, shape);
    CHECK(x.ptr() != y.ptr());
    CHECK(a.allocs() == allocs_after_pregrow);
  }

  // Idempotent: a free list that is already deep enough costs nothing, which is
  // the steady state on a warm server.
  const int allocs_before_second = a.allocs();
  pool.PreGrowForCapture(a, demand);
  CHECK(a.allocs() == allocs_before_second);

  // A profile is a property of the FORWARD, not of what is resident when it
  // runs. Re-measure with the retained block still live and the peak must read
  // two again, because the baseline moved with it. Without that subtraction the
  // profile would read three here and the pre-grow would over-allocate on every
  // capture for the rest of the process.
  pool.MarkStepBoundary();
  {
    DBuf scratch(Dev{a, qa}, DType::kF32, shape);
    DBuf out(Dev{a, qa}, DType::kF32, shape);
  }
  int64_t peak_with_retention = 0;
  for (const auto& e : pool.StepDemandProfile())
    if (e.first == key) peak_with_retention = e.second;
  CHECK(peak_with_retention == 2);
  retained.reset();
}

// ═══════════════════════════════════════════════════════════════════════════
// CONCURRENCY. Every case above this line runs on one thread, and until now so
// did every case anywhere that touches `DevicePool` — the pool held a mutex
// that nothing in the tree had ever contended. `sanitize-cpu (thread)` runs
// this suite, so the file was NOMINALLY covered by ThreadSanitizer while giving
// it no concurrent access to observe, which is not coverage.
//
// This matters now rather than in the abstract: `--max-num-seqs > 1` is the
// configuration the #1922 follow-up ladder measures, and the borrow added in
// that row (#1930) put two NEW pieces of shared mutable state behind that mutex
// — the `block_class_` map and the cross-class `retained_` arithmetic — where
// before there was one free list per key and nothing that read another class's.
//
// WHAT THE CASE ASSERTS, and why each assertion can fail on its own:
//
//   no double issue   — two threads must never hold one block at the same time.
//                       A `live` set guarded by ITS OWN mutex catches that
//                       directly, and the byte pattern each thread writes into
//                       its block catches it a second time, as a data race
//                       ThreadSanitizer reports on the block itself rather than
//                       as a count this file has to be trusted to keep.
//   no lost block     — after quiesce a SEQUENTIAL replay of the same ladder
//                       must reach the driver zero times. Whatever served a
//                       size during the concurrent phase, its own class or a
//                       larger one it borrowed from, is free again now; a block
//                       dropped on the floor, or returned to the wrong class,
//                       is a miss here.
//   retention adds up — `retained_` is maintained by three different lines
//                       (`Get`'s exact hit, `Get`'s borrow, `Put`), each
//                       adjusting by a DIFFERENT class than the caller named.
//                       `Drain` recounts the free lists independently and must
//                       report the same bytes.
//
// It is deliberately NOT a #1922 gate. It is green with the borrow on, with
// `VT_POOL_BORROW=0`, and with `VT_POOL_EXACT=1` — a concurrency invariant that
// only held in one lane would be a worse property, not a stronger test.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("device pool: concurrent Get/Put never double-issues, loses, or miscounts a block") {
  if (PoolBypass()) {
    // No free list, no shared class state, and `stats()` counts nothing: the
    // bypass lane hands every Get straight to the driver, so the invariants
    // below are not questions it can be asked.
    MESSAGE("SKIP: VT_POOL_BYPASS=1 removes the free list this case measures");
    return;
  }

  TagBackend& b = NewBackend();
  // ISOLATED (the header's stated use for a directly-constructed pool), so the
  // counts below are this case's own and no other case can perturb them.
  vllm::DevicePool pool(b);

  // Four octaves, and each rung appears twice: once ON a class boundary and
  // once three-quarters of the way up the one below it. The second form is what
  // makes the BORROW path live — it asks for less than a class the other
  // threads keep returning — while the first keeps the exact-hit path busy.
  constexpr size_t kBase = 1u << 16;
  const std::vector<size_t> ladder = {
      kBase,     kBase * 3 / 4,     kBase * 2, kBase * 2 * 3 / 4,
      kBase * 4, kBase * 4 * 3 / 4, kBase * 8, kBase * 8 * 3 / 4,
  };
  constexpr int kThreads = 8;
  constexpr int kIters = 200;

  // NO DOCTEST ASSERTION RUNS ON A WORKER. `REQUIRE` throws, and an exception
  // that escapes a `std::thread` calls `std::terminate` — the suite would die
  // rather than report. Every worker observation is an atomic counter that the
  // main thread checks after the join.
  std::mutex live_mu;
  std::set<void*> live;
  size_t peak_live = 0;  // guarded by live_mu
  std::atomic<int> double_issued{0};
  std::atomic<int> clobbered{0};
  std::atomic<int> null_block{0};
  std::atomic<uint64_t> gets{0};
  std::atomic<int> arrived{0};

  // Write both ends of the LOGICAL extent and read them back. A borrowed block
  // is larger than the request, so a defect that handed the tail of one block
  // to a second caller would not show at the head. Bounded rather than the
  // whole block: under ThreadSanitizer every byte written is a shadow update,
  // and the invariant does not need megabytes to hold. This is also the second,
  // independent detector for a double issue — if two threads hold one block,
  // these writes race on the block itself and TSan reports it directly, rather
  // than the case having to be trusted to keep its own count.
  auto stamp_and_verify = [&](void* p, size_t bytes, unsigned char tag) {
    const size_t touch = bytes < 4096 ? bytes : 4096;
    auto* q = static_cast<unsigned char*>(p);
    std::memset(q, tag, touch);
    std::memset(q + bytes - touch, tag, touch);
    for (size_t k = 0; k < touch; k += 337) {
      if (q[k] != tag || q[bytes - touch + k] != tag) ++clobbered;
    }
  };
  auto enter = [&](void* p) {
    std::lock_guard<std::mutex> lk(live_mu);
    if (!live.insert(p).second) ++double_issued;
    if (live.size() > peak_live) peak_live = live.size();
  };
  // Leave the live set BEFORE the block re-enters the pool: after `Put`
  // another thread may legitimately hold it, and an erase after that would
  // score a correct hand-off as a double issue.
  auto leave = [&](void* p) {
    std::lock_guard<std::mutex> lk(live_mu);
    live.erase(p);
  };

  std::vector<std::thread> workers;
  workers.reserve(kThreads);
  for (int t = 0; t < kThreads; ++t) {
    workers.emplace_back([&, t] {
      const auto tag = static_cast<unsigned char>(t + 1);

      // ── Phase A: every thread holds a block AT THE SAME TIME ──────────────
      // A barrier, so that contention is a property of the case rather than of
      // how the scheduler happened to interleave it. Without this the whole
      // run could serialise, every assertion below would still pass, and the
      // case would report green having measured nothing concurrent.
      {
        void* p = pool.Get(b, ladder[static_cast<size_t>(t) % ladder.size()]);
        if (p == nullptr) {
          ++null_block;
        } else {
          enter(p);
          stamp_and_verify(p, ladder[static_cast<size_t>(t) % ladder.size()], tag);
        }
        arrived.fetch_add(1, std::memory_order_acq_rel);
        while (arrived.load(std::memory_order_acquire) < kThreads) std::this_thread::yield();
        if (p != nullptr) {
          stamp_and_verify(p, ladder[static_cast<size_t>(t) % ladder.size()], tag);
          leave(p);
          pool.Put(b, ladder[static_cast<size_t>(t) % ladder.size()], p);
        }
        ++gets;
      }

      // ── Phase B: unsynchronised churn across the whole ladder ─────────────
      uint64_t x = 0x9E3779B97F4A7C15ULL * static_cast<uint64_t>(t + 1);
      for (int i = 0; i < kIters; ++i) {
        x = x * 6364136223846793005ULL + 1442695040888963407ULL;
        const size_t bytes = ladder[(x >> 33) % ladder.size()];
        void* p = pool.Get(b, bytes);
        ++gets;  // counted at the CALL, so `hits + misses == gets` holds even
                 // on the path below that cannot use the block.
        if (p == nullptr) {
          ++null_block;
          continue;
        }
        enter(p);
        stamp_and_verify(p, bytes, tag);
        leave(p);
        pool.Put(b, bytes, p);
      }
    });
  }
  for (auto& w : workers) w.join();

  CHECK(null_block.load() == 0);
  CHECK(double_issued.load() == 0);
  CHECK(clobbered.load() == 0);
  CHECK(live.empty());
  // The barrier makes this exact rather than likely: all eight blocks were out
  // at once, so the case cannot report green over a run that never overlapped.
  CHECK(peak_live == kThreads);

  const vllm::DevicePool::Stats after = pool.stats();
  // Nothing is still handed out: `block_class_` holds the live working set and
  // empties itself as blocks return, so a leaked entry is a leaked block.
  CHECK(after.live_blocks == 0);
  // Every Get was answered exactly once, by the free list or by the driver.
  CHECK(after.hits + after.misses == gets.load());
  // And the driver was asked exactly as often as the pool recorded a miss.
  CHECK(static_cast<uint64_t>(b.allocs()) == after.misses);
  // Phase A alone forces one driver allocation per thread: nothing is free when
  // it starts and no block is returned until every thread holds one.
  CHECK(after.misses >= static_cast<uint64_t>(kThreads));

  // NO BLOCK WAS LOST, AND EVERY BORROWED BLOCK WENT HOME. One at a time now,
  // so exactly one block is live at any moment and everything the concurrent
  // phase allocated is on a free list. Reaching the driver here means a block
  // is gone, or is sitting in a class no request of its own size can find.
  const uint64_t misses_before_replay = after.misses;
  for (size_t bytes : ladder) {
    void* p = pool.Get(b, bytes);
    REQUIRE(p != nullptr);
    pool.Put(b, bytes, p);
  }
  CHECK(pool.stats().misses == misses_before_replay);

  // RETENTION ADDS UP. `Drain` recounts the free lists from the class keys
  // themselves, which is an independent path to the number `retained_` has been
  // maintaining incrementally across three call sites and two classes per
  // borrow.
  const size_t retained = pool.stats().retained_bytes;
  CHECK(retained > 0);
  CHECK(pool.Drain(b) == retained);
  CHECK(pool.stats().retained_bytes == 0);
  // Every block the driver ever gave this pool came back to it.
  CHECK(b.frees() == b.allocs());
}

// ═══════════════════════════════════════════════════════════════════════════
// THE EMPTY BUFFER (#1904). `DBuf() = default` names a state that used to be
// reachable only by moving from a buffer, and it is NOT the same state: the
// move constructor clears `p_` alone, so a moved-from buffer keeps a live `b_`,
// while a default-constructed one has none. `Zero` and `Download` dereference
// `b_`, so on this state they were a null dereference on a legally
// constructible object.
//
// TWO CASES BECAUSE THERE ARE TWO CHECKS. A fresh review deleted BOTH checks
// and every suite in the tree stayed green -- the same "a guard whose deletion
// no test can see" shape this row already caught once, one commit earlier, in
// its own platform guard. Deleting either check now reds its own case.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("device pool: DBuf::Zero on an EMPTY buffer is refused by name") {
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  DBuf empty;
  REQUIRE(empty.ptr() == nullptr);
  REQUIRE(empty.bytes() == 0);
  CHECK_THROWS_WITH_AS(empty.Zero(Dev{a, qa}),
                       doctest::Contains("DBuf::Zero on an EMPTY buffer"), std::exception);
  // And the backend was never touched: a check that threw AFTER dispatching
  // would be a message wrapped around the fault it exists to replace.
  CHECK(a.allocs() == 0);
}

TEST_CASE("device pool: DBuf::Download from an EMPTY buffer is refused by name") {
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  DBuf empty;
  float host = 1.0f;
  CHECK_THROWS_WITH_AS(empty.Download(Dev{a, qa}, &host),
                       doctest::Contains("DBuf::Download from an EMPTY buffer"), std::exception);
  // The destination is untouched, which is what distinguishes a refusal from a
  // zero-byte copy that silently succeeded.
  CHECK(host == 1.0f);
}

// ═══════════════════════════════════════════════════════════════════════════
// #2435 — the RANK BOUND on the glue's tensor writer.
//
// `dense_attn::MakeTensor` writes `t.shape[i]` and `t.stride[i]` for every `i`
// below the shape's size, over two `int64_t[4]` arrays. It had no bound, so a
// rank-5 shape wrote past both. `vt::Tensor::Contiguous` — the only other
// writer of those arrays — has always had one; this is the parallel path
// closing.
//
// WHY THIS CASE EXISTS WHEN A SANITIZER LANE ALREADY CATCHES IT. That lane
// catches it by ABORTING: `-fno-sanitize-recover=all` kills the process at the
// first finding, so `test_qwen4_exp_layer_loop` produced one UBSan report and
// not one doctest assertion. A finding whose only witness is a lane that is
// `continue-on-error` and that runs on two of a dozen jobs is a finding that
// can come back unnoticed. This case is the one that fires everywhere.
//
// It asserts the REFUSAL and the boundary on both sides: rank 4 is still
// served, rank 5 throws by name. A bound written as `<` instead of `<=` passes
// the throw half and reds the serve half, which is the mutation that a
// one-sided case would miss.
// ═══════════════════════════════════════════════════════════════════════════
TEST_CASE("dense_attn::MakeTensor refuses a rank above vt::kMaxRank") {
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  std::vector<int64_t> host(2 * 3 * 4 * 5 * 6, 0);

  // Rank 4 is the boundary and is SERVED — the shape, the strides and the
  // element count all come out as they always did.
  const vt::Tensor t4 = vllm::dense_attn::MakeTensor(
      host.data(), DType::kI64, Device{DeviceType::kCPU, 0}, {2, 3, 4, 5});
  CHECK(t4.rank == 4);
  CHECK(t4.shape[0] == 2);
  CHECK(t4.shape[3] == 5);
  CHECK(t4.stride[0] == 60);
  CHECK(t4.stride[3] == 1);
  CHECK(t4.Numel() == 120);
  // And the storage markers are untouched. `stride[4]` of a rank-5 write lands
  // exactly here, so this is the assertion that separates "the arrays are the
  // right size" from "nothing else moved".
  CHECK(t4.repacked == false);
  CHECK(t4.q8_0_aligned == false);
  CHECK(t4.elem_kn_repacked == false);

  // Rank 5 is refused BY NAME rather than written past the end.
  CHECK_THROWS_WITH_AS(
      vllm::dense_attn::MakeTensor(host.data(), DType::kI64,
                                   Device{DeviceType::kCPU, 0}, {2, 3, 4, 5, 6}),
      doctest::Contains("exceeds vt::Tensor's kMaxRank"), std::exception);

  // The same bound through `DBuf`, which is how every rank-5 caller in this
  // tree actually reached it: the refusal must arrive BEFORE the pool hands out
  // a block, or a throw past the allocation leaks it.
  CHECK_THROWS_WITH_AS(
      DBuf(Dev{a, qa}, DType::kI64, std::vector<int64_t>{2, 3, 4, 5, 6}),
      doctest::Contains("exceeds vt::Tensor's kMaxRank"), std::exception);
  // AND THE BLOCK WAS NEVER DRAWN. `DBuf`'s constructor calls `pool_->Get`
  // before `MakeTensor`, and a constructor that throws never runs its own
  // destructor, so a refusal raised by `MakeTensor` alone would strand the
  // allocation — trading an out-of-bounds write for a leak. This is the
  // assertion that pins the check AHEAD of the pool, and it is what reds if the
  // constructor's own `CheckRank` is removed while `MakeTensor` keeps its.
  CHECK(a.allocs() == 0);
}

// ═══════════════════════════════════════════════════════════════════════════
// THE LEASE (ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2). A kolibri-1 fp8-dequant
// cache held decoded blocks via `DBuf::ReleaseShared()` — the cross-step
// carrier — while forward traffic kept allocating scratch of the same size
// class. Its diagnostics observed `POOL DOUBLE-HAND-OUT`: a fresh `DBuf` was
// handed a pointer a live cache entry still owned, and the entry's bytes came
// back as another tensor's data (`CACHE MISMATCH n=512 k=2560`). For the pool
// to hand a live-leased block out, that block must have entered a free list
// while its carrier was alive — i.e. SOMEONE returned it twice, and the pool
// accepted the second return in silence.
//
// THE INSTRUMENT IS PER-TEST, NOT GLOBAL. The raw Put/Get free-list tripwire
// tried during the cache experiment was unsound: a static set spanning pool
// instances printed 2.45 M false positives with the cache disabled. Here each
// case holds its own live-lease set, on ONE backend and ONE pool, and checks
// every allocation it receives against exactly the leases it created.
//
// The two cases below use their OWN size class (55,296 bytes) so no earlier
// case's free list can decide them, including under `--order-by=rand`.
// ═══════════════════════════════════════════════════════════════════════════

namespace {

// The size class every lease case below uses, and the byte count a same-class
// `DBuf`/`Put` names. 13,824 f32 elements round UP to 55,296 — distinct from
// every shape used above this point in the file.
const std::vector<int64_t> kLeaseShape{13824};
constexpr size_t kLeaseBytes = 13824 * sizeof(float);

}  // namespace

TEST_CASE("device pool: the dequant-cache lease pattern keeps every live lease unaliased") {
  // THE PATTERN, at pool level: allocate, lease out via the carrier, cycle
  // same-class scratch traffic through Get/Put while the lease is outstanding,
  // and never see the leased block again until the carrier dies. This is the
  // sound instrument the cache experiment lacked; it is GREEN on a sound pool
  // and stays green as the regression guard.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  if (PoolBypass()) return;  // no free list to alias through in this lane

  std::vector<void*> leased;
  std::shared_ptr<void> carrier;
  {
    DBuf w(Dev{a, qa}, DType::kF32, kLeaseShape);
    std::memset(w.ptr(), 0xAA, kLeaseBytes);
    leased.push_back(w.ptr());
    carrier = w.ReleaseShared();
  }
  REQUIRE(carrier != nullptr);

  // Same-class traffic cycles several times: each block is filled with its own
  // tag, returned, and re-handed. None of it may ever be the leased block, and
  // the leased bytes must survive untouched.
  for (int round = 0; round < 4; ++round) {
    DBuf scratch(Dev{a, qa}, DType::kF32, kLeaseShape);
    for (void* p : leased) CHECK(scratch.ptr() != p);
    std::memset(scratch.ptr(), static_cast<unsigned char>(round), kLeaseBytes);
    CHECK(static_cast<const unsigned char*>(leased[0])[0] == 0xAA);
    CHECK(static_cast<const unsigned char*>(leased[0])[kLeaseBytes - 1] == 0xAA);
  }
  // Dropping the carrier hands the block back; the next request reuses it.
  carrier.reset();
  DBuf again(Dev{a, qa}, DType::kF32, kLeaseShape);
  CHECK(again.ptr() == leased[0]);
}

TEST_CASE("device pool: a block returned while a LEASE is outstanding is REFUSED, not double-handed") {
  // The defect the cache experiment's tripwire fired on. The observed sequence
  // needs a second return of a block whose carrier is still alive — a Put the
  // rightful owner did not make. Before the fix the pool accepted it in
  // silence: the block entered the free list, the next `Get` handed it to a
  // new owner, and the carrier's bytes became another tensor's data. The pool
  // now refuses that return BY NAME, at the moment of the second Put, while
  // the legal carrier return still works.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  vllm::DevicePool& pool = vllm::Pool(a);

  void* leased_ptr = nullptr;
  std::shared_ptr<void> carrier;
  {
    DBuf w(Dev{a, qa}, DType::kF32, kLeaseShape);
    leased_ptr = w.ptr();
    std::memset(leased_ptr, 0xAA, kLeaseBytes);
    carrier = w.ReleaseShared();
  }
  REQUIRE(leased_ptr != nullptr);

  // THE SECOND RETURN. Nothing in the tree does this on purpose; the defect is
  // that doing it BY ACCIDENT was indistinguishable from a legal return. The
  // refusal must fire in BOTH lanes: under bypass the bogus Put would free a
  // live block to the driver, which is the same defect with a louder symptom.
  CHECK_THROWS_WITH_AS(
      pool.Put(a, kLeaseBytes, leased_ptr),
      doctest::Contains("still leased"), std::exception);

  // The refusal left the block OUT of the free list: the next request cannot
  // be handed the leased block, and a write through the new block cannot reach
  // the carrier's bytes.
  {
    DBuf fresh(Dev{a, qa}, DType::kF32, kLeaseShape);
    CHECK_MESSAGE(fresh.ptr() != leased_ptr,
                  "POOL DOUBLE-HAND-OUT: a live-leased block was handed to a new owner");
    std::memset(fresh.ptr(), 0xBB, kLeaseBytes);
    CHECK(static_cast<const unsigned char*>(leased_ptr)[0] == 0xAA);
    CHECK(static_cast<const unsigned char*>(leased_ptr)[kLeaseBytes - 1] == 0xAA);
  }

  // AND THE LEGAL RETURN STILL WORKS: the carrier's own deleter hands the
  // block back and the pool reuses it, exactly as the ReleaseShared case above
  // pins. A refusal that broke reuse would be `VT_POOL_BYPASS=1` wearing a fix
  // costume.
  carrier.reset();
  if (PoolBypass()) {
    CHECK(a.WasFreed(leased_ptr));
    return;
  }
  DBuf again(Dev{a, qa}, DType::kF32, kLeaseShape);
  CHECK(again.ptr() == leased_ptr);
}

TEST_CASE("device pool: the CAPPED Put refuses a live-leased block too (the DBuf destructor path)") {
  // The 4-argument (cap-aware) Put is the overload `DBuf::~DBuf` actually calls
  // (dense_device_glue.h): it is the LIVE path for ordinary scratch teardown,
  // not an edge case. The refusal guard must sit on it identically, and the
  // refusal must fire in BOTH lanes: under bypass the bogus capped Put frees a
  // live block to the driver — the same defect with a louder symptom.
  TagBackend& a = NewBackend();
  Queue qa = QueueOn(0);
  vllm::DevicePool& pool = vllm::Pool(a);

  // A cap large enough that the legal path POOLS the block rather than freeing
  // it, so the post-refusal assertions below probe the free list, not the
  // driver.
  const size_t cap = kLeaseBytes * 8;

  void* leased_ptr = nullptr;
  std::shared_ptr<void> carrier;
  {
    DBuf w(Dev{a, qa}, DType::kF32, kLeaseShape);
    leased_ptr = w.ptr();
    std::memset(leased_ptr, 0xAA, kLeaseBytes);
    carrier = w.ReleaseShared();
  }
  REQUIRE(leased_ptr != nullptr);

  // THE SECOND RETURN through the capped overload — the shape an accidental
  // extra teardown of a live DBuf produces.
  CHECK_THROWS_WITH_AS(
      pool.Put(a, kLeaseBytes, leased_ptr, cap),
      doctest::Contains("still leased"), std::exception);

  // The refusal left the block OUT of the capped free list: the next same-class
  // request cannot be handed the leased block, and a write through the new
  // block cannot reach the carrier's bytes.
  {
    DBuf fresh(Dev{a, qa}, DType::kF32, kLeaseShape);
    CHECK_MESSAGE(fresh.ptr() != leased_ptr,
                  "POOL DOUBLE-HAND-OUT via the capped Put: a live-leased block was handed to a new owner");
    std::memset(fresh.ptr(), 0xBB, kLeaseBytes);
    CHECK(static_cast<const unsigned char*>(leased_ptr)[0] == 0xAA);
    CHECK(static_cast<const unsigned char*>(leased_ptr)[kLeaseBytes - 1] == 0xAA);
  }

  // AND THE LEGAL CAPPED RETURN STILL WORKS: the carrier's own deleter goes
  // through the same capped overload and the block is reusable.
  carrier.reset();
  if (PoolBypass()) {
    CHECK(a.WasFreed(leased_ptr));
    return;
  }
  DBuf again(Dev{a, qa}, DType::kF32, kLeaseShape);
  CHECK(again.ptr() == leased_ptr);
}
