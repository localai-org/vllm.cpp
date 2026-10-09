// vllm.cpp original; see qwen3_5_weights.h. Weight naming/quant verified
// against nvidia/Qwen3.6-35B-A3B-NVFP4 snapshot 491c2f1e
// (.agents/specs/qwen36-forward-notes.md §6).
#include "vllm/model_executor/models/qwen3_5_weights.h"

#include <atomic>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <new>  // over-aligned operator new/delete (kDeviceAliasAlignment)
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"  // ReadF32Scalar (#1181)
#include "vllm/model_executor/models/qwen3_vl.h"  // LoadQwen3VLVisionWeights (#891)
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/unaligned.h"

namespace vllm {

int64_t OwnedTensor::Numel() const {
  if (rank == 0) return 0;
  int64_t n = 1;
  for (int i = 0; i < rank; ++i) n *= shape[i];
  return n;
}

uint64_t OwnedTensor::TowerUid() const {
  // See the field comment: an ADDRESS is not an identity for a cache that
  // outlives the model, because the allocator reuses addresses. A counter is,
  // because it never goes backwards.
  static std::atomic<uint64_t> next{1};
  const uint8_t* p = bytes.data();
  if (tower_uid == 0 || tower_uid_for != p) {
    tower_uid = next.fetch_add(1, std::memory_order_relaxed);
    tower_uid_for = p;
  }
  return tower_uid;
}

void OwnedTensor::ReleaseHost() const {
  // Free the host mirror once the device-resident copy is authoritative
  // (residency_policy().release_host_weights_after_upload; BACKEND-PLATFORM
  // item 2). Logically const — only the dead host buffer is reclaimed — so a
  // narrow const_cast, consistent with the mutable lazy-device-residency design.
  auto& self = *const_cast<OwnedTensor*>(this);
  if (self.bytes.empty()) return;
  // A BORROWED buffer owns no anonymous pages: the bytes live in the GGUF mmap
  // (clean, file-backed, reclaimable by the kernel without our help) or in an
  // expansion another tensor still references. madvise'ing them away would be
  // wrong on both counts, so releasing one means only dropping the keep-alive.
  if (self.bytes.borrowed()) {
    self.bytes.Reset();
    self.mmap_src = nullptr;  // nothing borrows the mapping through this tensor now
    self.mmap_src_bytes = 0;
    // The FILE descriptor goes with the mapping. It describes where these host
    // bytes came from, and there are no host bytes now. Left set, it outlives
    // its own subject: `bytes.data()` is null, so an expert-stream slice would
    // pread the recorded offset (which belongs to a tensor nothing is reading
    // any more) and would key every released tower on TowerId(nullptr) == 0.
    self.mmap_fd = -1;
    self.mmap_file_offset = 0;
    self.host_released = true;
    return;
  }
#if defined(__unix__) || defined(__APPLE__)
  // RETURN THE PHYSICAL PAGES TO THE OS NOW. std::vector's free() alone does not:
  // glibc raises its dynamic mmap threshold as large blocks are freed, so the
  // ~0.5 MB per-expert fp4 allocations end up served from the sbrk arena, where
  // free() only returns them to the free-list — RSS/PSS stay resident (measured:
  // the 35B serving PSS did NOT drop from the logical swap alone). madvise(
  // MADV_DONTNEED) drops the resident anonymous pages immediately (the bytes are
  // dead — the device Marlin resident is authoritative; a stray later read would
  // fault fresh zero pages, but nothing reads a released host weight). Interior
  // whole pages only, so the free()-metadata boundary words are untouched. This
  // mirrors the LOAD-SAFETENSORS windowed release (safetensors_reader.cpp:317).
  if (!self.bytes.empty()) {
    const long ps_l = ::sysconf(_SC_PAGESIZE);
    const auto ps = static_cast<uintptr_t>(ps_l > 0 ? ps_l : 4096);
    const auto begin = reinterpret_cast<uintptr_t>(self.bytes.data());
    const uintptr_t end = begin + self.bytes.size();
    const uintptr_t page_begin = (begin + ps - 1) & ~(ps - 1);
    const uintptr_t page_end = end & ~(ps - 1);
    if (page_end > page_begin) {
      ::madvise(reinterpret_cast<void*>(page_begin),
                static_cast<size_t>(page_end - page_begin), MADV_DONTNEED);
    }
  }
#endif
  // Reset() forces the underlying vector to deallocate its capacity.
  self.bytes.Reset();
  self.host_released = true;
}

// ENG-LOAD-DIRECT-UPLOAD (issue #150). A weight whose bytes BORROW the
// safetensors mmap has just been uploaded to the device, so its source range is
// consumed-and-dead exactly as a copied one is: drop its resident (clean,
// file-backed) pages, mirroring the loaders' windowed release. Correctness-safe
// on any backend -- the mapping is PROT_READ MAP_PRIVATE, so a later read of the
// still-valid borrowed view simply re-faults it from the file.
namespace {

void ReleaseDirectUploadSource(const OwnedTensor& w) {
  if (w.mmap_src == nullptr) return;
  // Deliberately NOT MaybeReleaseSourcePages: that call site is the host-copy
  // BYTE COUNTER (every copy helper reaches it after materializing a range),
  // and these bytes were never materialized on the host -- they went straight to
  // the device. Counting them there would report the very copy this row removes.
  if (LoadWindowedReleaseEnabled())
    ReleaseSourcePages(w.mmap_src, w.mmap_src_bytes);
}

}  // namespace

namespace {

// Drop the resident anonymous pages of `[p, p + nb)` without touching the
// allocator's boundary metadata. Interior WHOLE pages only, for the reason
// spelled out in `ReleaseHost` above: glibc's `free()` alone often leaves the
// pages resident on the sbrk arena's free list, and the whole subject here is
// the RSS.
void DropResidentInteriorPages(const uint8_t* p, size_t nb) {
#if defined(__unix__) || defined(__APPLE__)
  if (p == nullptr || nb == 0) return;
  const long ps_l = ::sysconf(_SC_PAGESIZE);
  const auto ps = static_cast<uintptr_t>(ps_l > 0 ? ps_l : 4096);
  const auto begin = reinterpret_cast<uintptr_t>(p);
  const uintptr_t end = begin + nb;
  const uintptr_t page_begin = (begin + ps - 1) & ~(ps - 1);
  const uintptr_t page_end = end & ~(ps - 1);
  if (page_end > page_begin) {
    ::madvise(reinterpret_cast<void*>(page_begin),
              static_cast<size_t>(page_end - page_begin), MADV_DONTNEED);
  }
#else
  (void)p;
  (void)nb;
#endif
}

}  // namespace

namespace {

// ATOMIC, although the function that feeds it is single-threaded by
// precondition. The re-homing branch below genuinely requires one thread — it
// destroys the source vector — and saying so is honest. These counters are a
// different thing: a DIAGNOSTIC that `HostAliasSnapshot` publishes to any
// caller, so a plain `uint64_t` incremented on the model path and read
// elsewhere is a data race in the standard's terms whatever the precondition
// says about the branch. Relaxed ordering costs nothing measurable here (five
// increments against a memcpy of a weight) and makes the snapshot well defined.
struct AtomicHostAliasStats {
  std::atomic<uint64_t> aliased_in_place_bytes{0};
  std::atomic<uint64_t> rehomed_bytes{0};
  std::atomic<uint64_t> declined_borrow_bytes{0};
  std::atomic<uint64_t> declined_other_bytes{0};
  std::atomic<uint64_t> calls{0};
};

AtomicHostAliasStats& AliasStats() {
  static AtomicHostAliasStats s;
  return s;
}

}  // namespace

HostAliasStats HostAliasSnapshot() {
  const AtomicHostAliasStats& s = AliasStats();
  HostAliasStats out;
  out.aliased_in_place_bytes = s.aliased_in_place_bytes.load(std::memory_order_relaxed);
  out.rehomed_bytes = s.rehomed_bytes.load(std::memory_order_relaxed);
  out.declined_borrow_bytes = s.declined_borrow_bytes.load(std::memory_order_relaxed);
  out.declined_other_bytes = s.declined_other_bytes.load(std::memory_order_relaxed);
  out.calls = s.calls.load(std::memory_order_relaxed);
  return out;
}

bool HostWeightAliasEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_QWEN35_ALIAS_HOST_WEIGHTS");
    return !(e != nullptr && e[0] == '0');
  }();
  return on;
}

bool StagingFitsModel(size_t model_weight_bytes, size_t device_total_bytes,
                      size_t reserve_bytes) {
  if (model_weight_bytes == 0 || device_total_bytes == 0) return false;
  // Overflow-safe: compare in the domain that cannot wrap.
  if (model_weight_bytes > (device_total_bytes - reserve_bytes) / 2) return false;
  return 2 * model_weight_bytes + reserve_bytes <= device_total_bytes;
}

namespace {
std::atomic<bool> g_stage_owned{false};
}  // namespace

void SetSafetensorsWeightBudget(size_t model_weight_bytes, size_t device_total_bytes) {
  static const size_t reserve = [] {
    const char* e = std::getenv("VT_QWEN35_STAGE_RESERVE_BYTES");
    if (e == nullptr || e[0] == '\0') return static_cast<size_t>(12ull << 30);
    const long long v = std::atoll(e);
    return v > 0 ? static_cast<size_t>(v) : static_cast<size_t>(12ull << 30);
  }();
  // `=1` pins the retag on (the #1299 escape hatch); `=0` forces staging.
  const char* pin = std::getenv("VT_QWEN35_ALIAS_HOST_WEIGHTS");
  if (pin != nullptr && pin[0] == '1') { g_stage_owned.store(false); return; }
  if (pin != nullptr && pin[0] == '0') { g_stage_owned.store(true); return; }
  g_stage_owned.store(
      StagingFitsModel(model_weight_bytes, device_total_bytes, reserve));
}

bool StageOwnedWeightsToDevice() { return g_stage_owned.load(); }

bool MakeHostBytesDeviceAliasable(const OwnedTensor& w,
                                  HostAliasOutcome* outcome) {
  AtomicHostAliasStats& st = AliasStats();
  st.calls.fetch_add(1, std::memory_order_relaxed);
  const uint64_t nbytes = static_cast<uint64_t>(w.bytes.size());
  auto report = [&](HostAliasOutcome o, bool ok) {
    if (outcome != nullptr) *outcome = o;
    switch (o) {
      case HostAliasOutcome::kAliasedInPlace:
        st.aliased_in_place_bytes.fetch_add(nbytes, std::memory_order_relaxed);
        break;
      case HostAliasOutcome::kRehomed:
        st.rehomed_bytes.fetch_add(nbytes, std::memory_order_relaxed);
        break;
      case HostAliasOutcome::kDeclinedBorrow:
        st.declined_borrow_bytes.fetch_add(nbytes, std::memory_order_relaxed);
        break;
      default:
        st.declined_other_bytes.fetch_add(nbytes, std::memory_order_relaxed);
        break;
    }
    return ok;
  };
  if (!HostWeightAliasEnabled())
    return report(HostAliasOutcome::kDeclinedDisabled, false);
  // Nothing to alias. The caller refuses this by name rather than handing a
  // kernel a null pointer; see ResidentWeight.
  if (w.bytes.empty()) return report(HostAliasOutcome::kDeclinedEmpty, false);
  if (reinterpret_cast<uintptr_t>(w.bytes.data()) % kDeviceAliasAlignment == 0) {
    // ISSUE #150's WINDOWED RELEASE STILL HAS TO HAPPEN, and this branch is the
    // third path that can skip it. A safetensors direct-upload borrow whose
    // offset happens to be a multiple of 256 is aliased here and never reaches
    // `AdoptDeviceBytesAsHost`, which is the only other caller of
    // `ReleaseDirectUploadSource` — so its consumed source pages would stay
    // resident, data-dependently, for roughly one borrow in eight. The comment
    // on the ordering inside `AdoptDeviceBytesAsHost` insists the release
    // happens on EVERY path including the `VT_ADOPT_DEVICE_BYTES=0` arm, and
    // this honours the same rule. A no-op unless `mmap_src` is set, which the
    // GGUF loader never does.
    //
    // ONCE, AND THE MEMO IS THE RECORD ITSELF (a fresh review of #1299 caught
    // the repeat). `AdoptDeviceBytesAsHost` calls this from behind
    // `if (!w.d_dev)`, so it ran exactly once per weight. THIS branch has no
    // memo: `ResidentWeight` re-tests the alignment on every call, about 1,361
    // times per forward step on the target checkpoint. Left uncleared,
    // `mmap_src` would make every one of those steps `madvise(MADV_DONTNEED)`
    // the very pages the GPU is about to read, and the kernel would fault them
    // straight back in. Correctness survives that; throughput does not. Clearing
    // the record IS the memo, and it says the true thing: these source pages are
    // spent. It also matches what `ReleaseHost` and the adoption branch below
    // already do after they consume the same record. The borrow itself is
    // untouched and stays a valid, re-faultable PROT_READ MAP_PRIVATE view.
    ReleaseDirectUploadSource(w);
    auto& consumed = *const_cast<OwnedTensor*>(&w);
    consumed.mmap_src = nullptr;
    consumed.mmap_src_bytes = 0;
    return report(HostAliasOutcome::kAliasedInPlace, true);
  }
  // A borrow owns no anonymous pages, so re-homing it would ADD residency
  // instead of removing it, and a tied pair's shared expansion must keep its one
  // keep-alive. Same reasoning, and the same answer, as `ReleaseHost`'s and
  // `AdoptDeviceBytesAsHost`'s borrowed branches.
  //
  // ONE EXCEPTION TO "A BORROW OWNS NO ANONYMOUS PAGES" NOW EXISTS, AND IT IS
  // THE ONE THIS FUNCTION CREATES. The block below turns an OWNED buffer into a
  // BORROWED one whose keep-alive is an over-aligned `operator new` block —
  // anonymous memory. That does not change the answer here (such a buffer is
  // already aligned and returns above), but it does mean the sentence is no
  // longer universally true, and the three places that reason from it —
  // `ReleaseHost`'s borrowed branch, `AdoptDeviceBytesAsHost`'s, and this one —
  // are now reasoning about GGUF mappings and tied expansions specifically. The
  // consequence worth naming: `ReleaseHost()` on a re-homed weight drops the
  // keep-alive rather than madvising, which frees the block through the deleter
  // and is correct, but it does not take the `MADV_DONTNEED` path. Nothing calls
  // `ReleaseHost` on a dense weight today; `HostMirrorIsRedundant` is what keeps
  // the one caller that could from doing it to an aliased one.
  if (w.bytes.borrowed())
    return report(HostAliasOutcome::kDeclinedBorrow, false);

  // SINGLE-THREADED BY PRECONDITION, stated rather than enforced. The
  // re-pointing below destroys the source vector, so two threads reaching it for
  // the same weight would race — and unlike `AdoptDeviceBytesAsHost`, which
  // hides behind `if (!w.d_dev)`, this branch has no memo and re-tests alignment
  // on every call. It is safe because the first touch of every weight happens
  // inside one forward on one thread, which is the same assumption the `d_dev`
  // memo two branches down has always made. A model that ever builds residents
  // from several threads must add a `call_once` here and there.
  auto& self = *const_cast<OwnedTensor*>(&w);
  const size_t nb = self.bytes.size();
  // Over-aligned `operator new` rather than `aligned_alloc`/`posix_memalign`:
  // it is the one spelling that is standard C++17 AND available on MSVC, which
  // this tree still compiles for, and it does not require the size to be a
  // multiple of the alignment.
  void* p = ::operator new(nb, std::align_val_t{kDeviceAliasAlignment});
  // The keep-alive is built IMMEDIATELY, before anything that can throw, so the
  // block is owned from the instant it exists. `shared_ptr`'s own control-block
  // allocation is the throwing step, and holding a raw `p` across it is how an
  // allocation leaks on a path nobody tests.
  std::shared_ptr<const void> keep(static_cast<const void*>(p), [](const void* q) {
    ::operator delete(const_cast<void*>(q), std::align_val_t{kDeviceAliasAlignment});
  });
  std::memcpy(p, self.bytes.data(), nb);
  // ORDER: release the OLD pages while they are still mapped, then re-point.
  // The assignment below destroys the vector that owns them, and madvise'ing a
  // range after it has been unmapped is at best a silent no-op and at worst
  // discards whatever mapped into the hole first — the same trap
  // `AdoptDeviceBytesAsHost` documents at length.
  DropResidentInteriorPages(self.bytes.data(), nb);
  self.bytes = OwnedBytes::Borrow(static_cast<const uint8_t*>(p), nb, std::move(keep));
  return report(HostAliasOutcome::kRehomed, true);
}

OwnedTensor BorrowWholeOwnedTensor(OwnedTensor& src) {
  // An empty source has nothing to keep alive, and `OwnedBytes::Borrow` refuses
  // a null owner by name. A block adapter legitimately passes an unpopulated
  // arm through (the GGUF path leaves `in_proj_ba` and `in_proj_qkvz` empty),
  // so this is a normal case and not a refusal.
  if (src.bytes.empty()) return src;
  // BEFORE `data()`: `KeepAlive()` moves an owned vector into a refcounted
  // holder, and although `std::vector`'s move preserves the heap address,
  // ordering the two calls means this view never depends on that.
  std::shared_ptr<const void> keep = src.bytes.KeepAlive();
  OwnedTensor v;
  v.bytes = OwnedBytes::Borrow(src.bytes.data(), src.bytes.size(), std::move(keep));
  v.dtype = src.dtype;
  v.weight_value_dtype = src.weight_value_dtype;
  v.rank = src.rank;
  for (int i = 0; i < src.rank; ++i) v.shape[i] = src.shape[i];
  v.nk = src.nk;
  // EVERY LAYOUT MARKER THE KERNEL KEYS ON. Dropping `repacked` makes
  // `kMatmulBTQuant` read i8mm-interleaved bytes as plain q8_0, which is the
  // CIQ-G7 all-zero-token failure; the other three have the same shape.
  v.repacked = src.repacked;
  v.q8_0_aligned = src.q8_0_aligned;
  v.elem_kn_repacked = src.elem_kn_repacked;
  v.mmap_fd = src.mmap_fd;
  v.mmap_file_offset = src.mmap_file_offset;
  // `mmap_src` is deliberately NOT carried. It is the direct-upload release
  // record, and it names bytes exactly one owner may release; a view that
  // inherited it would let `ReleaseDirectUploadSource` run twice.
  return v;
}

namespace {

struct AtomicBorrowReleaseStats {
  std::atomic<uint64_t> calls{0};
  std::atomic<uint64_t> bytes{0};
};

AtomicBorrowReleaseStats& BorrowReleaseStatsRef() {
  static AtomicBorrowReleaseStats s;
  return s;
}

}  // namespace

BorrowReleaseStats BorrowReleaseSnapshot() {
  const AtomicBorrowReleaseStats& s = BorrowReleaseStatsRef();
  BorrowReleaseStats out;
  out.calls = s.calls.load(std::memory_order_relaxed);
  out.bytes = s.bytes.load(std::memory_order_relaxed);
  return out;
}

bool MaybeReleaseStagedBorrowSource(vt::Backend& backend, vt::Queue& queue,
                                    const OwnedTensor& w, bool host_addressable) {
  // See the header for each of the three. The ORDER matters only in that the
  // cheap, allocation-free tests come before the synchronize: a backend this
  // does not apply to must not pay a stream sync per weight to find that out.
  //
  // BOTH host-addressability predicates, because they answer different
  // questions and a weight can reach this arm with them disagreeing. The
  // header says which is which.
  if (host_addressable || backend.DeviceMemoryIsHostAddressable()) return false;
  if (w.bytes.empty() || !w.bytes.borrowed()) return false;
  if (w.mmap_fd < 0) return false;
  // THE STAGING COPY IS ASYNCHRONOUS. `RocmBackend::Copy` is `hipMemcpyAsync` on
  // `queue`'s stream, and the CUDA backend's is the same shape. Releasing the
  // source under a live DMA would be a correctness bug, not a residency one, so
  // the copy is waited for here. This runs once per weight (see the `d_dev`
  // memo note in the header), so it is a first-forward cost and not a per-step
  // one.
  backend.Synchronize(queue);
  const size_t nb = w.bytes.size();
  DropResidentInteriorPages(w.bytes.data(), nb);
  AtomicBorrowReleaseStats& st = BorrowReleaseStatsRef();
  st.calls.fetch_add(1, std::memory_order_relaxed);
  st.bytes.fetch_add(static_cast<uint64_t>(nb), std::memory_order_relaxed);
  return true;
}

void AdoptDeviceBytesAsHost(vt::Backend& backend, const OwnedTensor& w) {
  if (w.d_dev == nullptr) return;
  // ENG-LOAD-DIRECT-UPLOAD: a direct-upload borrow is the ONE borrow that may be
  // adopted. Its bytes are the file mapping, not an anonymous copy and not a
  // tied pair's shared expansion, so re-pointing it at the (host-addressable)
  // device allocation loses nothing and stops the page cache from holding a
  // second copy of the model for the rest of the process. Where the device
  // allocation is NOT host-addressable the borrow stays as it is (a valid,
  // re-faultable view) and only the resident pages are dropped.
  if (w.mmap_src != nullptr && w.bytes.borrowed()) {
    // ORDERING IS LOAD-BEARING, AND IT IS THE FIRST STATEMENT FOR A REASON.
    //
    // (1) Release BEFORE re-pointing `bytes`. The assignment below overwrites
    // the OwnedBytes that carries this tensor's keep-alive on
    // `StTensor::mapping`. By adoption time the shard's `SafetensorsFile` is
    // long gone (`LoadedEngine::FromModelDir` drops `shards` at the end of the
    // load; adoption runs lazily at first forward use), so for the LAST
    // adopted weight of a shard that keep-alive is the last reference and
    // `~Mapping` munmaps SYNCHRONOUSLY inside the assignment. Releasing after
    // it would madvise a range that is no longer mapped -- observed under
    // strace as `munmap(...) = 0` followed by `madvise(..., MADV_DONTNEED) =
    // -1 ENOMEM`. That is a silent no-op only for as long as nothing else
    // mmaps into the hole first; on a private anonymous mapping landing there,
    // MADV_DONTNEED DISCARDS live data. `ReleaseHost()`'s borrowed branch
    // already orders it this way.
    //
    // (2) Release BEFORE the two early returns. The page release and the
    // adoption are two independent levers: a backend without host-addressable
    // device memory, and the `VT_ADOPT_DEVICE_BYTES=0` A/B, must both still
    // drop the consumed source pages. Otherwise the documented adoption knob
    // silently turns off the direct-upload release as well.
    ReleaseDirectUploadSource(w);
    // Not host-addressable: the borrow STAYS as it is -- a valid, re-faultable
    // PROT_READ MAP_PRIVATE view -- so `mmap_src` is deliberately left set.
    if (!backend.DeviceMemoryIsHostAddressable()) return;
    if (const char* v = std::getenv("VT_ADOPT_DEVICE_BYTES");
        v != nullptr && v[0] == '0') {
      return;
    }
    auto& self = *const_cast<OwnedTensor*>(&w);
    const size_t nb = self.bytes.size();
    std::shared_ptr<const void> keep(w.d_dev,
                                     static_cast<const void*>(w.d_dev.get()));
    self.bytes = OwnedBytes::Borrow(static_cast<const uint8_t*>(w.d_dev.get()),
                                    nb, std::move(keep));
    self.mmap_src = nullptr;
    self.mmap_src_bytes = 0;
    // `bytes` now points at DEVICE memory, so the file offset no longer
    // describes it. See the note in ReleaseHost: a descriptor that outlives its
    // mapping reads as a valid source and is not one.
    self.mmap_fd = -1;
    self.mmap_file_offset = 0;
    return;
  }
  if (!backend.DeviceMemoryIsHostAddressable()) return;
  // A BORROWED buffer owns no anonymous pages (a GGUF mmap is clean and
  // file-backed; a shared expansion is a tied pair's single copy), so adopting
  // would reclaim nothing and would break the tie. Same reasoning as
  // ReleaseHost's borrowed branch.
  if (w.bytes.empty() || w.bytes.borrowed()) return;
  if (const char* v = std::getenv("VT_ADOPT_DEVICE_BYTES"); v != nullptr && v[0] == '0') {
    return;
  }
  auto& self = *const_cast<OwnedTensor*>(&w);
  const size_t nb = self.bytes.size();
#if defined(__unix__) || defined(__APPLE__)
  // Drop the resident anonymous pages BEFORE the vector is destroyed, for
  // exactly the reason ReleaseHost above does it: glibc raises its dynamic mmap
  // threshold as large blocks are freed, so free() alone leaves many weight
  // buffers on the sbrk arena free-list with their pages still resident, and
  // the whole point here is the RSS. Interior whole pages only, so free()'s
  // boundary metadata is untouched.
  {
    const long ps_l = ::sysconf(_SC_PAGESIZE);
    const auto ps = static_cast<uintptr_t>(ps_l > 0 ? ps_l : 4096);
    const auto begin = reinterpret_cast<uintptr_t>(self.bytes.data());
    const uintptr_t end = begin + nb;
    const uintptr_t page_begin = (begin + ps - 1) & ~(ps - 1);
    const uintptr_t page_end = end & ~(ps - 1);
    if (page_end > page_begin) {
      ::madvise(reinterpret_cast<void*>(page_begin),
                static_cast<size_t>(page_end - page_begin), MADV_DONTNEED);
    }
  }
#endif
  // The keep-alive is the device allocation's own shared_ptr, so the borrowed
  // view cannot outlive the bytes it points at: the aliasing constructor shares
  // d_dev's control block, and the buffer is freed through the vt Backend only
  // when BOTH the weight's d_dev and this view are gone.
  std::shared_ptr<const void> keep(w.d_dev, static_cast<const void*>(w.d_dev.get()));
  self.bytes = OwnedBytes::Borrow(static_cast<const uint8_t*>(w.d_dev.get()), nb,
                                  std::move(keep));
}

namespace {

// nullopt => env-driven; set => forced (test seam).
std::optional<bool>& DirectUploadOverride() {
  static std::optional<bool> value;
  return value;
}

}  // namespace

void detail::SetLoadDirectUploadOverrideForTesting(std::optional<bool> value) {
  DirectUploadOverride() = value;
}

bool LoadDirectUploadEnabled() {
  const std::optional<bool> forced = DirectUploadOverride();
  if (forced.has_value()) return *forced;
  // Process-cached: read the env once. DEFAULT ON; "=0" rolls back to the
  // copy-then-upload behavior (same-binary A/B, house convention).
  static const bool enabled = [] {
    const char* e = std::getenv("VT_LOAD_DIRECT_UPLOAD");
    return e == nullptr || e[0] != '0';
  }();
  return enabled;
}

bool BorrowStTensorBytes(OwnedTensor& o, const StTensor& t, vt::DType dtype,
                         const std::vector<int64_t>& shape) {
  if (!LoadDirectUploadEnabled()) return false;
  // FAIL CLOSED on anything that is not a whole-range verbatim view of a live
  // mapping: no keep-alive (a synthetic StTensor), no data, or a destination
  // whose element count times dtype width is not EXACTLY the source span. The
  // caller then runs its ordinary copy, so a mismatch can only cost the lever,
  // never correctness.
  if (t.mapping == nullptr || t.data == nullptr || t.nbytes == 0) return false;
  const auto rank = static_cast<int>(shape.size());
  if (rank <= 0 || rank > vt::kMaxRank) return false;
  size_t numel = 1;
  for (const int64_t dim : shape) {
    if (dim <= 0) return false;
    const auto d = static_cast<size_t>(dim);
    if (numel > SIZE_MAX / d) return false;
    numel *= d;
  }
  const size_t elem = vt::SizeOf(dtype);
  if (elem == 0 || numel > SIZE_MAX / elem) return false;
  if (numel * elem != t.nbytes) return false;

  o.dtype = dtype;
  o.rank = rank;
  for (int i = 0; i < rank; ++i) o.shape[i] = shape[static_cast<size_t>(i)];
  o.bytes = OwnedBytes::Borrow(t.data, t.nbytes, t.mapping);
  o.mmap_src = t.data;
  o.mmap_src_bytes = t.nbytes;
  load_stats::AddBorrowed(t.nbytes);
  return true;
}

vt::Tensor OwnedTensor::View() const {
  VT_CHECK(!host_released,
           "OwnedTensor::View: host bytes were released after device upload");
  return ViewOn(const_cast<uint8_t*>(bytes.data()), vt::Device{});
}

vt::Tensor OwnedTensor::ViewOn(void* data, vt::Device device,
                               std::vector<int64_t> view_shape) const {
  if (view_shape.empty()) view_shape.assign(shape, shape + rank);
  VT_CHECK(view_shape.size() <= vt::kMaxRank,
           "OwnedTensor::ViewOn: invalid rank");
  vt::Tensor t;
  t.data = data;
  t.dtype = dtype;
  t.device = device;
  // ViewOn carries ONLY weight_value_dtype. The LAYOUT markers (repacked,
  // q8_0_aligned, elem_kn_repacked) are deliberately NOT inherited here: each
  // ResidentWeight arm owns a different marker set, and copying them centrally
  // silently changed the CPU-alias lane for every model that shares this helper.
  // See expert_stream_seam.h, which documents that q8_0_aligned must not reach
  // a resident view.
  t.weight_value_dtype = weight_value_dtype;
  t.rank = static_cast<int>(view_shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    VT_CHECK(view_shape[i] >= 0, "OwnedTensor::ViewOn: negative dimension");
    t.shape[i] = view_shape[i];
    t.stride[i] = stride;
    stride *= view_shape[i];
  }
  VT_CHECK(t.rank == 0 || stride == Numel(), "OwnedTensor::ViewOn: numel mismatch");
  return t;
}

namespace {

OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "qwen3_5 weights: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

// The per-tensor scale read, from the shared seam. It used to be a local copy
// bounded by `nbytes >= sizeof(float)`, a FLOOR, so an array was read as
// element 0 and any dtype was reinterpreted (#1181).
using dense_loaders::ReadF32Scalar;

// src bf16 [rows, cols] -> dst bf16 [cols, rows].
//
// `src` is `const void*`, not `const uint16_t*`, because the caller below hands
// it a pointer INTO the mmap'd safetensors payload. A tensor's offset there is
// the running byte total of everything ahead of it, so a bf16 tensor that
// follows an odd-length one starts on an odd byte and the typed pointer is
// undefined to form or load through (issue #627). `vt::LoadUnaligned` is the
// project's seam for that — the same one `dense_loaders::ReadF32Scalar` uses
// and `dense_loaders::TransposeBf16` already uses for this exact loop.
// The strided form. `src_pitch` is the distance in ELEMENTS between successive
// source rows, which is `cols` for a dense 2-D tensor and something larger when
// the block is a column-slice of a wider one (the stacked gate/up halves below).
void TransposeBf16Strided(const void* src, int64_t rows, int64_t cols,
                          int64_t src_pitch, uint16_t* dst) {
  const auto* bytes = static_cast<const uint8_t*>(src);
  for (int64_t r = 0; r < rows; ++r) {
    const uint8_t* src_row = bytes + r * src_pitch * 2;
    for (int64_t c = 0; c < cols; ++c) {
      dst[c * rows + r] = vt::LoadUnaligned<uint16_t>(src_row + c * 2);
    }
  }
}

void TransposeBf16(const void* src, int64_t rows, int64_t cols, uint16_t* dst) {
  TransposeBf16Strided(src, rows, cols, cols, dst);
}

// Verbatim (non-transposing) copy of a `rows` x `cols` bf16 block whose source
// rows are `src_pitch` elements apart, into a dense [rows, cols] destination.
// Same unaligned discipline as above: a stacked expert slice is an arbitrary
// byte offset into the mmap'd payload (#627).
void CopyBf16Strided(const void* src, int64_t rows, int64_t cols,
                     int64_t src_pitch, uint16_t* dst) {
  const auto* bytes = static_cast<const uint8_t*>(src);
  for (int64_t r = 0; r < rows; ++r) {
    const uint8_t* src_row = bytes + r * src_pitch * 2;
    for (int64_t c = 0; c < cols; ++c) {
      dst[r * cols + c] = vt::LoadUnaligned<uint16_t>(src_row + c * 2);
    }
  }
}

// BF16 tensor copied verbatim (optionally reshaped): shape override lets the
// [conv_dim,1,K] conv weight collapse to [conv_dim,K] with no data change.
OwnedTensor LoadBf16Direct(const TensorResolver& get, const std::string& name,
                           const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 weights: expected BF16 for " + name);
  std::vector<int64_t> shape =
      shape_override.empty() ? t.shape : shape_override;
  OwnedTensor borrowed;
  // ENG-LOAD-DIRECT-UPLOAD (issue #150): whole-range verbatim copy; qualifies.
  if (BorrowStTensorBytes(borrowed, t, vt::DType::kBF16, shape)) return borrowed;
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  VT_CHECK(t.nbytes == o.bytes.size(),
           "qwen3_5 weights: byte-size mismatch for " + name);
  std::memcpy(o.bytes.data(), t.data, t.nbytes);
  // LOAD-SAFETENSORS: source range now copied-then-dead; drop its resident pages
  // so the owned mirror never double-resides with the mmap (spec §page-lifetime).
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// BF16 [out, in] -> owned bf16 [in, out] (Matmul-B layout).
OwnedTensor LoadBf16Transposed(const TensorResolver& get,
                               const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 weights: expected BF16 for " + name);
  VT_CHECK(t.shape.size() == 2,
           "qwen3_5 weights: expected 2-D weight for " + name);
  const int64_t out_dim = t.shape[0];
  const int64_t in_dim = t.shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(t.data, out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// BF16 [n] -> owned f32 [n] (A_log / dt_bias; upcast is lossless).
OwnedTensor LoadBf16ToF32(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16", "qwen3_5 weights: expected BF16 for " + name);
  VT_CHECK(t.shape.size() == 1,
           "qwen3_5 weights: expected 1-D tensor for " + name);
  const int64_t n = t.shape[0];
  OwnedTensor o = MakeOwned(vt::DType::kF32, {n});
  // Unaligned: `t.data` is an arbitrary byte offset into the mmap (#627).
  const uint8_t* src = t.data;
  auto* dst = reinterpret_cast<float*>(o.bytes.data());
  for (int64_t i = 0; i < n; ++i) {
    dst[i] = vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(src + i * 2));
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// Native-precision dense projections — keep the 35B's per-tensor FP8 W8A8 attn
// (q/k/v/o) + GDN (in_proj_qkv/z, out_proj) weights RESIDENT in fp8 (LoadFp8Raw)
// + load the static input_scale, so the forward runs a native fp8 W8A8 GEMM
// (static per-tensor activation quant + fp8 tensor cores — either the cutlass
// sm120 kernel or, by default, cuBLASLt fp8; see MatmulFp8CutlassD /
// DenseCublasLtFp8Enabled). Mirrors vLLM's actual scheme (the checkpoint IS
// W8A8) and HALVES those projections' weight bytes vs bf16.
//   DEFAULT ON (VT_DENSE_NATIVE) on a CUDA+cutlass build. VT_DENSE_NATIVE=0 (or
// the legacy VT_FP8_CUTLASS=0) restores the dequant-fp8->bf16-AT-LOAD path
// (LoadFp8Transposed into the bf16 fields + cublas bf16 W8A16) — the previous
// default, kept for the parent's A/B (and the only path on a build WITHOUT the
// fp8 cutlass kernel, guarded by VT_CUTLASS_FP8 so we never route to an
// uncompiled GEMM). The dense NVFP4 sinks (shared-expert + lm_head) are already
// native (fp4-resident + Marlin W4A16) and not gated here.
bool DenseNativeEnabled() {
#ifdef VT_CUTLASS_FP8
  const char* dn = std::getenv("VT_DENSE_NATIVE");
  if (dn != nullptr && dn[0] == '0') return false;
  const char* legacy = std::getenv("VT_FP8_CUTLASS");  // back-compat opt-out
  if (legacy != nullptr && legacy[0] == '0') return false;
  return true;
#else
  return false;
#endif
}

// Per-tensor FP8 projection `<proj>.weight` F8_E4M3 [out,in] + `.weight_scale`
// scalar + `.input_scale` scalar -> RAW fp8-resident W8A8 Fp8Weight (no dequant,
// no transpose: the bytes stay in the on-disk [N=out, K=in] orientation the
// cutlass W8A8 GEMM reads directly). `alpha = input_scale * weight_scale` is
// precomputed (per-tensor scalars) — the folded scalar vt::MatmulFp8Cutlass
// multiplies the fp8 accumulator by (mirror vLLM ScaledEpilogue for per-tensor).
Fp8Weight LoadFp8Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "F8_E4M3",
           "qwen3_5 weights: expected F8_E4M3 for " + proj + ".weight");
  VT_CHECK(w.shape.size() == 2,
           "qwen3_5 weights: expected 2-D weight for " + proj);
  Fp8Weight r;
  r.n = w.shape[0];
  r.k = w.shape[1];
  r.weight_scale = ReadF32Scalar(get, proj + ".weight_scale");
  r.input_scale = ReadF32Scalar(get, proj + ".input_scale");
  r.alpha = r.input_scale * r.weight_scale;
  r.packed = MakeOwned(vt::DType::kI8, {r.n, r.k});
  VT_CHECK(w.nbytes == r.packed.bytes.size(),
           "qwen3_5 weights: fp8 byte-size mismatch for " + proj);
  std::memcpy(r.packed.bytes.data(), w.data, w.nbytes);
  MaybeReleaseSourcePages(w.data, w.nbytes);
  return r;
}

// Per-tensor FP8 projection `<proj>.weight` [out,in] + `<proj>.weight_scale`
// scalar -> owned bf16 [in, out] (dequant then transpose).
OwnedTensor LoadFp8Transposed(const TensorResolver& get,
                              const std::string& proj) {
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "F8_E4M3",
           "qwen3_5 weights: expected F8_E4M3 for " + proj + ".weight");
  VT_CHECK(w.shape.size() == 2,
           "qwen3_5 weights: expected 2-D weight for " + proj);
  const int64_t out_dim = w.shape[0];
  const int64_t in_dim = w.shape[1];
  const float scale = ReadF32Scalar(get, proj + ".weight_scale");

  std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
  DequantFp8ToBf16(w.data, scale, out_dim * in_dim, dq.data());
  MaybeReleaseSourcePages(w.data, w.nbytes);

  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(dq.data(), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// NVFP4 W4A16 projection `<proj>.weight` U8 [out,in/2] + `.weight_scale` F8
// [out,in/16] + `.weight_scale_2` f32 scalar -> RAW fp4-resident Nvfp4Weight
// (M2.2b). No dequant, no transpose: the bytes are kept in the on-disk
// [N=out, K=in] orientation vt::MatmulNvfp4 reads directly. This is the
// storage refactor that removes the ~40-min CPU dequant + ~70GB bf16 tensors.
Nvfp4Weight LoadNvfp4Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& w = get(proj + ".weight");
  VT_CHECK(w.dtype == "U8",
           "qwen3_5 weights: expected U8 for " + proj + ".weight");
  VT_CHECK(w.shape.size() == 2,
           "qwen3_5 weights: expected 2-D packed weight for " + proj);
  const int64_t out_dim = w.shape[0];
  const int64_t in_dim = w.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 weights: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "qwen3_5 weights: expected F8_E4M3 for " + proj + ".weight_scale");
  const float ws2 = ReadF32Scalar(get, proj + ".weight_scale_2");

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.scale2 = ws2;
  r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  VT_CHECK(w.nbytes == r.packed.bytes.size(),
           "qwen3_5 weights: packed byte-size mismatch for " + proj);
  std::memcpy(r.packed.bytes.data(), w.data, w.nbytes);
  MaybeReleaseSourcePages(w.data, w.nbytes);
  r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  VT_CHECK(ws.nbytes == r.scale.bytes.size(),
           "qwen3_5 weights: scale byte-size mismatch for " + proj);
  std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
  MaybeReleaseSourcePages(ws.data, ws.nbytes);
  return r;
}

GdnLayerWeights LoadGdn(const TensorResolver& get, const std::string& base,
                        MoeProjDtype in_dtype) {
  const std::string la = base + "linear_attn.";
  GdnLayerWeights g;
  // THE BF16 GDN TOWER (issue #864) -- what `Qwen/Qwen3.6-35B-A3B` and
  // `Qwen/Qwen3.8-2.4T-A95B` actually ship, both of which carry ZERO
  // `weight_scale` tensors. Selected by tensor PRESENCE
  // (`ResolveQwen3_5MoeTowerDtypes`), NOT by `DenseNativeEnabled()`: that lever
  // switches fp8-resident against fp8-dequant and both of its arms would read
  // an fp8 tensor here.
  //
  // The destination is the SAME bf16 Matmul-B `[in, out]` field the
  // `VT_DENSE_NATIVE=0` dequant arm below fills (`qwen3_5_weights.h:289-312`),
  // so the forward is reached unchanged -- `ProjectGdn*` already selects the
  // bf16 field when the fp8 one is empty, which is the path the GGUF and
  // synthetic loaders have always taken. The transpose runs through
  // `LoadBf16Transposed` -> `TransposeBf16Strided` -> `vt::LoadUnaligned`, so a
  // tower tensor at an odd safetensors offset is read the one legal way (#627).
  if (in_dtype == MoeProjDtype::kBf16) {
    g.in_proj_qkv = LoadBf16Transposed(get, la + "in_proj_qkv.weight");
    g.in_proj_z = LoadBf16Transposed(get, la + "in_proj_z.weight");
    g.out_proj = LoadBf16Transposed(get, la + "out_proj.weight");
  } else if (DenseNativeEnabled()) {
    // W8A8 fp8 (35B), DEFAULT: keep raw fp8 + input_scale, run the native fp8
    // GEMM. VT_DENSE_NATIVE=0 restores dequant-at-load into the bf16 fields
    // (parent A/B).
    g.in_proj_qkv_fp8 = LoadFp8Raw(get, la + "in_proj_qkv");
    g.in_proj_z_fp8 = LoadFp8Raw(get, la + "in_proj_z");
    g.out_proj_fp8 = LoadFp8Raw(get, la + "out_proj");
  } else {
    g.in_proj_qkv = LoadFp8Transposed(get, la + "in_proj_qkv");
    g.in_proj_z = LoadFp8Transposed(get, la + "in_proj_z");
    g.out_proj = LoadFp8Transposed(get, la + "out_proj");
  }
  // GDN-MOE-PACKED-BA (#1169). vLLM owns ONE physical in_proj_ba on every
  // Qwen3.5/3.6 GDN layer, dense and MoE alike (packed_modules_mapping on
  // Qwen3_5ForCausalLMBase, qwen3_5.py:297; shard order b=0, a=1 at :217-218
  // @ 555967922). Mirror it here as the dense loader does at
  // qwen3_5_dense_weights.cpp:514: one [2*Hv, H] nk owner with rows [b; a], and
  // the split fields deliberately empty -- ProjectGdnBA slices the owner where
  // it needs the legacy pair, and never retains duplicate bytes. This is what
  // makes `has_packed_ba` true on the 35B (#1169). Independent of the tower
  // branch above: in_proj_b / in_proj_a are bf16 on every Qwen3.5/3.6
  // checkpoint (the modelopt_mixed 35B keeps them on the ignore list), which is
  // why the dense loader merges them unconditionally too.
  g.in_proj_ba = dense_loaders::LoadMergedBf16RawNK(
      get, {la + "in_proj_b.weight", la + "in_proj_a.weight"});
  // conv1d.weight ships [conv_dim,1,K]; collapse the singleton to [conv_dim,K].
  const StTensor& conv = get(la + "conv1d.weight");
  VT_CHECK(conv.shape.size() == 3 && conv.shape[1] == 1,
           "qwen3_5 weights: unexpected conv1d shape");
  g.conv1d_weight =
      LoadBf16Direct(get, la + "conv1d.weight", {conv.shape[0], conv.shape[2]});
  g.a_log = LoadBf16ToF32(get, la + "A_log");
  g.dt_bias = LoadBf16ToF32(get, la + "dt_bias");
  g.norm_weight = LoadBf16Direct(get, la + "norm.weight");
  return g;
}

FullAttnLayerWeights LoadAttn(const TensorResolver& get,
                              const std::string& base, MoeProjDtype in_dtype) {
  const std::string sa = base + "self_attn.";
  FullAttnLayerWeights a;
  // THE BF16 ATTENTION TOWER (issue #864). Same reasoning as `LoadGdn` above:
  // selected by tensor presence, landing in the SAME bf16 Matmul-B `[in, out]`
  // fields the fp8-dequant arm fills (`qwen3_5_weights.h:347-350`), which the
  // GGUF and synthetic loaders already populate and the forward already reads.
  if (in_dtype == MoeProjDtype::kBf16) {
    a.q_proj = LoadBf16Transposed(get, sa + "q_proj.weight");
    a.k_proj = LoadBf16Transposed(get, sa + "k_proj.weight");
    a.v_proj = LoadBf16Transposed(get, sa + "v_proj.weight");
    a.o_proj = LoadBf16Transposed(get, sa + "o_proj.weight");
  } else if (DenseNativeEnabled()) {
    // W8A8 fp8 (35B), DEFAULT: keep raw fp8 + input_scale, run the native fp8
    // GEMM. VT_DENSE_NATIVE=0 restores dequant-at-load into the bf16 fields
    // (parent A/B).
    a.q_proj_fp8 = LoadFp8Raw(get, sa + "q_proj");
    a.k_proj_fp8 = LoadFp8Raw(get, sa + "k_proj");
    a.v_proj_fp8 = LoadFp8Raw(get, sa + "v_proj");
    a.o_proj_fp8 = LoadFp8Raw(get, sa + "o_proj");
  } else {
    a.q_proj = LoadFp8Transposed(get, sa + "q_proj");
    a.k_proj = LoadFp8Transposed(get, sa + "k_proj");
    a.v_proj = LoadFp8Transposed(get, sa + "v_proj");
    a.o_proj = LoadFp8Transposed(get, sa + "o_proj");
  }
  a.q_norm = LoadBf16Direct(get, sa + "q_norm.weight");
  a.k_norm = LoadBf16Direct(get, sa + "k_norm.weight");
  return a;
}

// The routed-expert host copies (E × gate/up/down NVFP4, MakeOwned+memcpy). Split
// out so the 35B load-phase peak-PSS interleave can materialize EXACTLY ONE
// layer's experts just before the device Marlin build + host free (the
// `load_layer_experts` closure in LoadQwen3_5Moe), instead of all N layers'
// experts coexisting on the host.
void LoadMoeExpertsInto(const TensorResolver& get, const std::string& mlp,
                        int64_t num_experts, MoeBlockWeights& m) {
  m.expert_gate_fp4.reserve(static_cast<size_t>(num_experts));
  m.expert_up_fp4.reserve(static_cast<size_t>(num_experts));
  m.expert_down_fp4.reserve(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    const std::string ex = mlp + "experts." + std::to_string(e) + ".";
    m.expert_gate_fp4.push_back(LoadNvfp4Raw(get, ex + "gate_proj"));
    m.expert_up_fp4.push_back(LoadNvfp4Raw(get, ex + "up_proj"));
    m.expert_down_fp4.push_back(LoadNvfp4Raw(get, ex + "down_proj"));
  }
}

// The 3-D STACKED, UNQUANTIZED (bf16) routed-expert arm (issue #740) — the
// layout every PUBLISHED Qwen3.5-family MoE repo ships, as opposed to the
// per-expert NVFP4 requant `LoadMoeExpertsInto` above reads.
//
// THE SLICING ORDER IS UPSTREAM'S, NOT INFERRED FROM SHAPES. At the parity pin
// `555967922`, `vllm/model_executor/layers/fused_moe/routed_experts.py`:
//
//   :1081-1082   (f"{w13}weight", f"experts.{gate_up}", 0, "w1"),
//                (f"{w13}weight", f"experts.{gate_up}", 1, "w3"),
//   :923-928     if shard_id in {"w1", "w3"}:
//                    if fused_weight.shape[-1] != unpadded_hidden:
//                        fused_weight = fused_weight.transpose(-1, -2)
//                    # Repurpose expert_id for deconcatenating w1 and w3
//                    experts_shard = fused_weight.chunk(2, dim=1)[expert_id]
//   :929-933     else:
//                    if fused_weight.shape[-2] != unpadded_hidden:
//                        fused_weight = fused_weight.transpose(-1, -2)
//                    experts_shard = fused_weight
//   :942         loaded_experts = experts_shard.unbind()  <- stride is dim 0
//
// The third tuple element is normally an expert id; for the two FUSED entries it
// is repurposed as the CHUNK INDEX, which is upstream's own comment at :927. So
// chunk 0 is `w1` (gate) and chunk 1 is `w3` (up).
//
// So `gate_up_proj` is normalized to `[E, 2I, H]` (LAST dim hidden) and split in
// half along dim 1, FIRST half gate and second half up; `down_proj` is
// normalized to `[E, H, I]` (dim -2 hidden); and expert `e` is slice `e` of dim
// 0 in both. The `shape[-1] != hidden` / `shape[-2] != hidden` tests are
// upstream's own, and are mirrored here rather than replaced by a shape guess —
// a plausible-looking guess loads cleanly and produces wrong logits.
//
// THREE INDEPENDENT UPSTREAM CONFIRMATIONS that chunk 0 is the gate, because
// this is the one bit that a token gate would catch and nothing here would:
//   :494-500   `_load_w13` narrows w1 to offset 0 of the destination and w3 to
//              offset `shard_size`, with :656 `SHARD_ID_TO_SHARDED_DIM`
//              {"w1": 0, "w2": 1, "w3": 0}.
//   unquantized_fused_moe_method.py:97-106 allocates `w13_weight` as
//              [num_experts, 2 * intermediate, hidden] — the canonical form.
//   activation.py:118-143 `SiluAndMul` is `silu(x[..., :d]) * x[..., d:]` with
//              `d = x.shape[-1] // 2`, so SiLU lands on the FIRST half of the
//              fused operand. Rows [0, I) are the gate independently of any
//              loader bookkeeping.
//
// And HUGGINGFACE DECLARES THE AXIS ORDER OUTRIGHT, which is what makes
// upstream's runtime `shape[-1] != hidden` probe a compatibility branch rather
// than the authority — transformers 5.3.0
// models/qwen3_5_moe/modeling_qwen3_5_moe.py:812-844:
//   :820  self.gate_up_proj = nn.Parameter(
//             torch.empty(num_experts, 2 * intermediate_dim, hidden_dim))
//   :821  self.down_proj = nn.Parameter(
//             torch.empty(num_experts, hidden_dim, intermediate_dim))
//   :842  gate, up = linear(current_state, self.gate_up_proj[e]).chunk(2, -1)
// `F.linear(x, W)` is `x @ W.T`, so output column j is row j of W: `gate` is
// rows [0, I) of `gate_up_proj[e]`. `modular_qwen3_5_moe.py:164` names the same
// split in its TP plan — "experts.gate_up_proj": "packed_colwise".
//
// Corroborated against the real published indices, read from each shard's own
// safetensors header 2026-08-14 (the generator and the captured manifest are
// tests/vllm/models/fixtures/gen_qwen3_5_stacked_shapes.py and
// qwen3_5_stacked_shapes.inc):
//   Qwen/Qwen3.6-35B-A3B  (H 2048, moe_intermediate 512, 256 experts)
//     gate_up_proj BF16 [256, 1024, 2048], down_proj BF16 [256, 2048, 512]
//   Qwen/Qwen3.8-2.4T-A95B (H 8192, moe_intermediate 2048, 512 experts)
//     gate_up_proj BF16 [512, 4096, 8192], down_proj BF16 [512, 8192, 2048]
// Both are the CANONICAL orientation, and in both the sibling per-expert-shaped
// `shared_expert.gate_proj.weight` pins which of I and H is which.
// `gemma4_weights.cpp:194-199` asserts the same two shapes in tree.
//
// Destination convention is the loader's existing bf16 one
// (qwen3_5_weights.h:420-422): gate/up `[H, I]`, down `[I, H]` — transposed
// from the checkpoint's `[out, in]`, exactly what the GGUF and synthetic arms
// already produce, so the forward's bf16 MoE path is unchanged.
void LoadMoeExpertsStackedBf16Into(const TensorResolver& get,
                                   const std::string& mlp, int64_t num_experts,
                                   int64_t hidden, MoeBlockWeights& m) {
  VT_CHECK(hidden > 0,
           "qwen3_5 weights: hidden_size must be > 0 to resolve the 3-D stacked "
           "routed-expert orientation for " +
               mlp + "experts.gate_up_proj");
  const std::string gu_name = mlp + "experts.gate_up_proj";
  const std::string dn_name = mlp + "experts.down_proj";
  const StTensor& gu = get(gu_name);
  const StTensor& dn = get(dn_name);
  // Refused by dtype, by name: a stacked expert tensor that is NOT bf16 is a
  // quantized stacked layout, which is a further owed arm and not this one.
  VT_CHECK(gu.dtype == "BF16",
           "qwen3_5 weights: only BF16 3-D stacked routed experts are "
           "implemented -- " +
               gu_name + " is " + gu.dtype +
               ". A quantized stacked expert layout is OWED: see "
               ".agents/specs/moe-bf16-stacked-experts.md.");
  VT_CHECK(dn.dtype == "BF16",
           "qwen3_5 weights: only BF16 3-D stacked routed experts are "
           "implemented -- " +
               dn_name + " is " + dn.dtype +
               ". A quantized stacked expert layout is OWED: see "
               ".agents/specs/moe-bf16-stacked-experts.md.");
  VT_CHECK(gu.shape.size() == 3 && dn.shape.size() == 3,
           "qwen3_5 weights: expected 3-D stacked routed experts for " + mlp +
               "experts.{gate_up_proj,down_proj}");
  VT_CHECK(gu.shape[0] == num_experts && dn.shape[0] == num_experts,
           "qwen3_5 weights: stacked routed experts must have num_experts (" +
               std::to_string(num_experts) + ") on dim 0 for " + mlp +
               "experts.{gate_up_proj,down_proj}");

  // Orientation, upstream's test and upstream's tie-break: "last dim is hidden"
  // is checked FIRST, so a degenerate config where 2I == H resolves the same way
  // it does there.
  const bool gu_canonical = gu.shape[2] == hidden;  // [E, 2I, H]
  VT_CHECK(gu_canonical || gu.shape[1] == hidden,
           "qwen3_5 weights: " + gu_name +
               " has hidden_size on neither trailing dim; cannot resolve the "
               "gate/up split orientation");
  const int64_t two_i = gu_canonical ? gu.shape[1] : gu.shape[2];
  VT_CHECK(two_i > 0 && two_i % 2 == 0,
           "qwen3_5 weights: " + gu_name +
               " must carry an EVEN gate|up dimension (gate and up are its two "
               "halves)");
  const int64_t inter = two_i / 2;
  const bool dn_canonical = dn.shape[1] == hidden;  // [E, H, I]
  VT_CHECK(dn_canonical || dn.shape[2] == hidden,
           "qwen3_5 weights: " + dn_name +
               " has hidden_size on neither trailing dim");
  VT_CHECK((dn_canonical ? dn.shape[2] : dn.shape[1]) == inter,
           "qwen3_5 weights: " + dn_name +
               " intermediate dim disagrees with half of " + gu_name);

  m.expert_gate.reserve(static_cast<size_t>(num_experts));
  m.expert_up.reserve(static_cast<size_t>(num_experts));
  m.expert_down.reserve(static_cast<size_t>(num_experts));
  for (int64_t e = 0; e < num_experts; ++e) {
    OwnedTensor gate = MakeOwned(vt::DType::kBF16, {hidden, inter});
    OwnedTensor up = MakeOwned(vt::DType::kBF16, {hidden, inter});
    OwnedTensor down = MakeOwned(vt::DType::kBF16, {inter, hidden});
    auto* gate_dst = reinterpret_cast<uint16_t*>(gate.bytes.data());
    auto* up_dst = reinterpret_cast<uint16_t*>(up.bytes.data());
    auto* down_dst = reinterpret_cast<uint16_t*>(down.bytes.data());
    if (gu_canonical) {
      // [E, 2I, H]: each half is a contiguous [I, H] block; transpose to [H, I].
      const uint8_t* base = gu.data + e * two_i * hidden * 2;
      TransposeBf16Strided(base, inter, hidden, hidden, gate_dst);
      TransposeBf16Strided(base + inter * hidden * 2, inter, hidden, hidden,
                           up_dst);
    } else {
      // [E, H, 2I]: each half is a COLUMN block of an [H, 2I] slab, and the
      // destination [H, I] is that block verbatim — the two transposes upstream
      // composes (normalize, then our [out,in]->[in,out]) cancel exactly.
      const uint8_t* base = gu.data + e * hidden * two_i * 2;
      CopyBf16Strided(base, hidden, inter, two_i, gate_dst);
      CopyBf16Strided(base + inter * 2, hidden, inter, two_i, up_dst);
    }
    if (dn_canonical) {
      // [E, H, I] -> [I, H].
      TransposeBf16Strided(dn.data + e * hidden * inter * 2, hidden, inter,
                           inter, down_dst);
    } else {
      // [E, I, H] is already the destination orientation.
      CopyBf16Strided(dn.data + e * inter * hidden * 2, inter, hidden, hidden,
                      down_dst);
    }
    m.expert_gate.push_back(std::move(gate));
    m.expert_up.push_back(std::move(up));
    m.expert_down.push_back(std::move(down));
  }
  // Both source ranges are now copied-then-dead, same as every other helper.
  MaybeReleaseSourcePages(gu.data, gu.nbytes);
  MaybeReleaseSourcePages(dn.data, dn.nbytes);
}

MoeBlockWeights LoadMoe(const TensorResolver& get, const std::string& base,
                        int64_t num_experts, bool with_experts,
                        MoeExpertLayout layout, int64_t hidden,
                        MoeProjDtype shared_dtype) {
  const std::string mlp = base + "mlp.";
  MoeBlockWeights m;
  m.router_gate = LoadBf16Transposed(get, mlp + "gate.weight");
  m.shared_gate = LoadBf16Transposed(get, mlp + "shared_expert_gate.weight");
  // M2.2b: the NVFP4 expert + shared projections are kept fp4-resident (raw
  // packed + scales, no dequant/transpose); the bf16 expert_*/shared_*_proj
  // fields stay EMPTY. The forward calls vt::MatmulNvfp4 on the fp4 fields.
  // with_experts=false DEFERS the routed-expert loop (the ~16.9 GiB host
  // consumer) to the per-layer streaming path; the small shared projections are
  // always loaded eagerly (retained through the device build).
  if (with_experts) {
    if (layout == MoeExpertLayout::kStackedBf16) {
      LoadMoeExpertsStackedBf16Into(get, mlp, num_experts, hidden, m);
    } else {
      LoadMoeExpertsInto(get, mlp, num_experts, m);
    }
  }
  const std::string se = mlp + "shared_expert.";
  // THE BF16 SHARED EXPERT (issue #864). Destination convention is the bf16 one
  // the header declares (`qwen3_5_weights.h:432-434`): gate/up `[H, Is]`, down
  // `[Is, H]`, i.e. transposed from the checkpoint's `[out, in]` -- the same
  // fields the GGUF and synthetic arms fill, so `MoeBlock`'s bf16 branch reads
  // it unchanged. Exactly one of the bf16 and fp4 sets is populated.
  if (shared_dtype == MoeProjDtype::kBf16) {
    m.shared_gate_proj = LoadBf16Transposed(get, se + "gate_proj.weight");
    m.shared_up_proj = LoadBf16Transposed(get, se + "up_proj.weight");
    m.shared_down_proj = LoadBf16Transposed(get, se + "down_proj.weight");
  } else {
    m.shared_gate_proj_fp4 = LoadNvfp4Raw(get, se + "gate_proj");
    m.shared_up_proj_fp4 = LoadNvfp4Raw(get, se + "up_proj");
    m.shared_down_proj_fp4 = LoadNvfp4Raw(get, se + "down_proj");
  }
  return m;
}

// Full decoder layer minus (optionally) the routed experts. Shared by the public
// LoadQwen3_5MoeLayer (with_experts=true) and the deferred streaming load.
Qwen3_5MoeLayerWeights LoadLayerImpl(const TensorResolver& get,
                                     const std::string& layer_type,
                                     int64_t layer_idx, int64_t num_experts,
                                     bool with_experts,
                                     const std::string& backbone_prefix,
                                     MoeExpertLayout layout, int64_t hidden,
                                     const Qwen3_5MoeTowerDtypes& tower) {
  const std::string base =
      backbone_prefix + "layers." + std::to_string(layer_idx) + ".";
  Qwen3_5MoeLayerWeights layer;
  layer.input_layernorm = LoadBf16Direct(get, base + "input_layernorm.weight");
  layer.post_attention_layernorm =
      LoadBf16Direct(get, base + "post_attention_layernorm.weight");
  if (layer_type == "linear_attention") {
    layer.is_linear_attention = true;
    layer.gdn = LoadGdn(get, base, tower.gdn);
  } else if (layer_type == "full_attention") {
    layer.is_linear_attention = false;
    layer.attn = LoadAttn(get, base, tower.attn);
  } else {
    VT_CHECK(false, "qwen3_5 weights: unknown layer_type " + layer_type);
  }
  layer.moe = LoadMoe(get, base, num_experts, with_experts, layout, hidden,
                      tower.shared_expert);
  return layer;
}

// True iff any name in `names` is a backbone tensor under `prefix`. Only the
// three structural backbone spellings vote (see qwen3_5_weights.h): the
// vision tower (`model.visual.*`) and the top-level `lm_head.*` / `mtp.*` are
// deliberately NOT backbone names, so they cannot decide the namespace.
bool HasBackboneUnder(const std::vector<std::string>& names,
                      std::string_view prefix) {
  const std::string embed = std::string(prefix) + "embed_tokens.weight";
  const std::string norm = std::string(prefix) + "norm.weight";
  const std::string layers = std::string(prefix) + "layers.";
  for (const std::string& name : names) {
    if (name == embed || name == norm) return true;
    if (name.compare(0, layers.size(), layers) == 0) return true;
  }
  return false;
}

// --- Routed-expert layout: resolved once, and what remains REFUSED BY NAME ---
//
// This loader reads TWO routed-expert layouts (issue #740):
//
//   kPerExpertNvfp4  `LoadMoeExpertsInto` -> `LoadNvfp4Raw`, which hard-requires
//                    a `U8` `.weight`, an `F8_E4M3` `.weight_scale` and a
//                    `.weight_scale_2`. What an NVFP4 requant ships, and what
//                    every gated row (27B / 35B / Coder) reads today.
//   kStackedBf16     `LoadMoeExpertsStackedBf16Into`, ONE 3-D bf16 tensor per
//                    projection. What the PUBLISHED repos ship: read off the
//                    live safetensors indices, `Qwen/Qwen3.8-2.4T-A95B` has 93x
//                    `mlp.experts.gate_up_proj` + 93x `.down_proj` and ZERO
//                    `weight_scale` / `input_scale` tensors, and
//                    `Qwen/Qwen3.6-35B-A3B` is the same under the VL prefix.
//
// WHAT IS STILL NOT IMPLEMENTED, and is refused rather than discovered: an
// unquantized PER-EXPERT layout, a non-bf16 stacked one (refused by dtype in the
// stacked reader itself), an NVFP4 attention or GDN tower, an FP8 shared expert,
// and an FP8 `lm_head`. Every one of those already FAILED before #864 -- at a
// raw dtype complaint from inside a reader -- so refusing them by name is
// strictly a better report of the same unsupported checkpoint.
//
// The bf16 tower, bf16 shared expert and bf16 `lm_head` are NO LONGER refused:
// issue #864 implements all four, which is what makes a published Qwen bf16 MoE
// repo load. The refusals below were narrowed for exactly those shapes and for
// nothing else.
//
// AGENTS.md: an arm that is not implemented "is refused with a message naming
// the missing piece ... never left to be discovered later". Left alone, such a
// load dies at `LoadNvfp4Raw(get, "lm_head")` with "expected U8 for
// lm_head.weight", which reads as a corrupt checkpoint (issue #490).
//
// ANCHORED AT `<backbone>layers.`, DELIBERATELY. The one checkpoint the 35B row
// gates, `nvidia/Qwen3.6-35B-A3B-NVFP4`, DOES carry the stacked spelling — under
// the top-level `mtp.` draft-head prefix, which is neither `model.layers.` nor
// `model.language_model.layers.`. Broadening either the scan or the resolution
// below to every `.mlp.experts.` name would flip that checkpoint's whole model
// onto the stacked arm. The synthetic fixture in
// tests/vllm/models/test_qwen3_8_text_only.cpp carries those two `mtp.` names so
// the regression is CPU-visible.
static const std::string kMoeExpertLayoutHelp =
    " This loader implements two routed-expert layouts: 3-D STACKED BF16 -- "
    "<layer>.mlp.experts.{gate_up_proj,down_proj}, which is what the published "
    "Qwen/Qwen3.8-2.4T-A95B and Qwen/Qwen3.6-35B-A3B repos ship; and per-expert "
    "NVFP4 -- <layer>.mlp.experts.<e>.{gate,up,down}_proj.weight (U8 packed) + "
    ".weight_scale (F8_E4M3) + .weight_scale_2, which is what an NVFP4 requant "
    "(e.g. nvidia/Qwen3.6-35B-A3B-NVFP4) ships. Outside the routed experts it "
    "reads a BF16 or per-tensor-FP8 attention/GDN tower, a BF16 or NVFP4 shared "
    "expert, and a BF16 or NVFP4 lm_head, each resolved by tensor presence. "
    "Per-expert-but-unquantized experts, a non-BF16 stacked expert tensor, an "
    "NVFP4 attention/GDN tower, an FP8 shared expert and an FP8 lm_head are all "
    "still OWED, not silently unsupported: see "
    ".agents/specs/moe-bf16-tower-arms.md.";

// True iff `name` is a routed-expert tensor under this checkpoint's backbone,
// and if so whether it is the PER-EXPERT spelling (`experts.<digit>...`) rather
// than the stacked one (`experts.gate_up_proj` / `experts.down_proj`).
bool ClassifyRoutedExpertName(const std::string& name,
                              const std::string& layers, bool* per_expert) {
  static const std::string kExperts = ".mlp.experts.";
  if (name.compare(0, layers.size(), layers) != 0) return false;
  const size_t at = name.find(kExperts);
  if (at == std::string::npos) return false;
  const size_t rest = at + kExperts.size();
  if (rest >= name.size()) return false;
  *per_expert = std::isdigit(static_cast<unsigned char>(name[rest])) != 0;
  return true;
}

// Inert on every supported layout — every name it inspects already has to exist
// for the load to succeed at all.
void CheckMoeQuantLayoutSupported(const std::vector<std::string>& names,
                                  const std::string& backbone,
                                  MoeExpertLayout layout,
                                  const Qwen3_5MoeTowerDtypes& tower) {
  const std::string& kRequired = kMoeExpertLayoutHelp;
  const std::string layers = backbone + "layers.";
  const std::string weight = ".weight";
  const std::unordered_set<std::string> present(names.begin(), names.end());
  for (const std::string& name : names) {
    bool per_expert = false;
    if (!ClassifyRoutedExpertName(name, layers, &per_expert)) continue;
    // The stacked arm carries no scale tensors by construction; its remaining
    // requirement is a dtype, which only the reader can see, so it refuses
    // there (`LoadMoeExpertsStackedBf16Into`) and not here.
    if (!per_expert || layout != MoeExpertLayout::kPerExpertNvfp4) continue;
    if (name.size() > weight.size() &&
        name.compare(name.size() - weight.size(), weight.size(), weight) == 0 &&
        present.count(name + "_scale") == 0) {
      VT_CHECK(false,
               "qwen3_5 weights: unquantized routed experts are not implemented "
               "for the safetensors MoE arm -- \"" +
                   name + "\" has no \"" + name + "_scale\" beside it." +
                   kRequired);
    }
  }
  // ...and the three NON-routed components, refused by the dtype the probe
  // RESOLVED rather than discovered as a complaint from inside a reader (#490).
  // Each of these already failed before #864; naming it is the whole change.
  // Names the namespace-scope constant, not the function-local `kRequired`
  // reference bound to it: a local reference is odr-used here and would need a
  // capture, which MSVC enforces (#1068).
  const auto refuse = [](const char* what, MoeProjDtype got,
                         const char* supported) {
    VT_CHECK(false, std::string("qwen3_5 weights: a ") +
                        MoeProjDtypeName(got) + " " + what +
                        " is not implemented for the safetensors MoE arm -- it "
                        "reads " +
                        supported + " there." + kMoeExpertLayoutHelp);
  };
  if (tower.gdn == MoeProjDtype::kNvfp4) {
    refuse("GDN tower (<layer>.linear_attn.{in_proj_qkv,in_proj_z,out_proj})",
           tower.gdn, "BF16 or per-tensor FP8");
  }
  if (tower.attn == MoeProjDtype::kNvfp4) {
    refuse("attention tower (<layer>.self_attn.{q,k,v,o}_proj)", tower.attn,
           "BF16 or per-tensor FP8");
  }
  if (tower.shared_expert == MoeProjDtype::kFp8) {
    refuse("shared expert (<layer>.mlp.shared_expert.{gate,up,down}_proj)",
           tower.shared_expert, "BF16 or NVFP4");
  }
  if (tower.lm_head == MoeProjDtype::kFp8) {
    // The DENSE loader dequantizes an FP8 head (`LoadLmHeadAnyDtype`); the MoE
    // arm does not, and says so rather than dying on a dtype assert.
    refuse("lm_head", tower.lm_head, "BF16 or NVFP4");
  }
}

}  // namespace

// ONE routed-expert layout decision for the whole checkpoint, mirroring
// `ResolveQwen3_5BackbonePrefix`. A per-lookup fallback would let a checkpoint
// bind some layers from each layout and still appear to succeed, so a mixed
// index is refused here exactly as a mixed namespace is.
//
// An index with NO routed-expert names at all resolves to the per-expert arm —
// the status quo before #740 — and the load then fails at the first missing
// tensor, which is what it did before and is a truthful report of the index.
MoeExpertLayout ResolveQwen3_5MoeExpertLayout(
    const std::vector<std::string>& tensor_names,
    const std::string& backbone_prefix) {
  const std::string layers = backbone_prefix + "layers.";
  bool saw_per_expert = false;
  bool saw_stacked = false;
  for (const std::string& name : tensor_names) {
    bool per_expert = false;
    if (!ClassifyRoutedExpertName(name, layers, &per_expert)) continue;
    (per_expert ? saw_per_expert : saw_stacked) = true;
  }
  VT_CHECK(!(saw_per_expert && saw_stacked),
           "qwen3_5 weights: checkpoint carries BOTH routed-expert spellings "
           "under \"" +
               layers +
               "\" -- per-expert \"experts.<e>.<proj>\" AND stacked "
               "\"experts.gate_up_proj\"; refusing a mixed expert layout rather "
               "than binding some experts from each." +
               kMoeExpertLayoutHelp);
  return saw_stacked ? MoeExpertLayout::kStackedBf16
                     : MoeExpertLayout::kPerExpertNvfp4;
}

const char* MoeProjDtypeName(MoeProjDtype dtype) {
  switch (dtype) {
    case MoeProjDtype::kBf16:
      return "BF16";
    case MoeProjDtype::kFp8:
      return "per-tensor FP8";
    case MoeProjDtype::kNvfp4:
      return "NVFP4";
  }
  return "unknown";
}

// THE DENSE ARM'S LADDER, MIRRORED EXACTLY (qwen3_5_dense_weights.cpp:358-360
// `IsNvfp4Projection`, :475-484 `load_projection`). See the header for why the
// order matters and what binds the two in test.
MoeProjDtype ClassifyQwen3_5Projection(const TensorDtypeProbe& dtype_of,
                                       const std::string& proj) {
  // 1. NVFP4 under EITHER spelling: compressed-tensors names the packed weight
  //    `weight_packed`; ModelOpt keeps `.weight` and adds `weight_scale_2`.
  //    Probing only one of the two missed nvidia/Qwen3.6-27B-NVFP4 entirely.
  if (!dtype_of(proj + ".weight_packed").empty() ||
      !dtype_of(proj + ".weight_scale_2").empty()) {
    return MoeProjDtype::kNvfp4;
  }
  // 2. per-tensor FP8, decided by the WEIGHT'S OWN DTYPE and not by the presence
  //    of a scale: a `.weight` that is F8_E4M3 with no `weight_scale` beside it
  //    is a broken fp8 projection, and it must fail as one (naming the missing
  //    scale) rather than be re-read as bf16 and produce garbage.
  const std::string weight_dtype = dtype_of(proj + ".weight");
  if (weight_dtype == "F8_E4M3") return MoeProjDtype::kFp8;
  // 3. otherwise plain BF16 -- what every published Qwen bf16 MoE repo ships.
  //
  // ...but ONLY if the weight really is bf16. A `.weight` that is U8 has packed
  // 4-bit codes in it and is an NVFP4 projection MISSING ITS GLOBAL SCALE, and
  // falling through to the bf16 arm reports that as "expected BF16 for
  // <proj>.weight" -- the exact shape of complaint issue #490 exists to stop,
  // because it reads as a corrupt checkpoint rather than as an absent tensor.
  // The DENSE arm has the same hole (`LoadBf16RawNK` asserts the dtype); naming
  // the missing companion is strictly a better report of the same refusal and
  // changes no answer for a well-formed projection.
  VT_CHECK(weight_dtype.empty() || weight_dtype == "BF16",
           "qwen3_5 weights: \"" + proj + ".weight\" is " + weight_dtype +
               ", which this loader can only read as a QUANTIZED projection, "
               "and the tensor that says which is missing: an NVFP4 weight "
               "needs \"" +
               proj + ".weight_scale_2\" (ModelOpt) or the compressed-tensors "
                      "\"" +
               proj + ".weight_packed\" spelling beside it, and an F8_E4M3 "
                      "weight needs \"" +
               proj + ".weight_scale\"." + kMoeExpertLayoutHelp);
  return MoeProjDtype::kBf16;
}

bool Qwen3_5ProjectionPresent(const TensorDtypeProbe& dtype_of,
                              const std::string& proj) {
  return !dtype_of(proj + ".weight").empty() ||
         !dtype_of(proj + ".weight_packed").empty();
}

namespace {

// One component's vote. `resolved` is the decision so far and `owner` the
// projection that cast it, so a disagreement can name BOTH sides.
struct TowerVote {
  MoeProjDtype resolved = MoeProjDtype::kBf16;
  std::string owner;
  bool seen = false;
};

void CastTowerVote(const TensorDtypeProbe& dtype_of, const std::string& proj,
                   const char* component, TowerVote& vote) {
  if (!Qwen3_5ProjectionPresent(dtype_of, proj)) return;
  const MoeProjDtype got = ClassifyQwen3_5Projection(dtype_of, proj);
  if (!vote.seen) {
    vote.resolved = got;
    vote.owner = proj;
    vote.seen = true;
    return;
  }
  VT_CHECK(got == vote.resolved,
           std::string("qwen3_5 weights: the ") + component +
               " disagrees with itself about its quantization -- \"" +
               vote.owner + "\" is " + MoeProjDtypeName(vote.resolved) +
               " but \"" + proj + "\" is " + MoeProjDtypeName(got) +
               ". This loader resolves each component ONCE per checkpoint and "
               "refuses a component bound half from each dtype, exactly as it "
               "refuses a mixed weight namespace and a mixed routed-expert "
               "layout: a half-quantized tower loads cleanly and produces wrong "
               "logits rather than an error." +
               kMoeExpertLayoutHelp);
}

}  // namespace

Qwen3_5MoeTowerDtypes ResolveQwen3_5MoeTowerDtypes(
    const TensorDtypeProbe& dtype_of, const std::string& backbone_prefix,
    const std::vector<std::string>& layer_types) {
  TowerVote gdn;
  TowerVote attn;
  TowerVote shared;
  TowerVote head;
  for (size_t l = 0; l < layer_types.size(); ++l) {
    const std::string base =
        backbone_prefix + "layers." + std::to_string(l) + ".";
    if (layer_types[l] == "linear_attention") {
      const std::string la = base + "linear_attn.";
      for (const char* p : {"in_proj_qkv", "in_proj_z", "out_proj"}) {
        CastTowerVote(dtype_of, la + p, "GDN tower", gdn);
      }
    } else if (layer_types[l] == "full_attention") {
      const std::string sa = base + "self_attn.";
      for (const char* p : {"q_proj", "k_proj", "v_proj", "o_proj"}) {
        CastTowerVote(dtype_of, sa + p, "attention tower", attn);
      }
    }
    // Every Qwen3.5 MoE layer carries the shared expert, whichever attention
    // kind it is.
    const std::string se = base + "mlp.shared_expert.";
    for (const char* p : {"gate_proj", "up_proj", "down_proj"}) {
      CastTowerVote(dtype_of, se + p, "shared expert", shared);
    }
  }
  // The head is TOP-LEVEL (unprefixed) in both published spellings. A checkpoint
  // with no head at all is the tied-word-embeddings case; it casts no vote and
  // the load then fails at the reader, exactly as it did before.
  CastTowerVote(dtype_of, "lm_head", "lm_head", head);

  Qwen3_5MoeTowerDtypes tower;
  if (gdn.seen) tower.gdn = gdn.resolved;
  if (attn.seen) tower.attn = attn.resolved;
  if (shared.seen) tower.shared_expert = shared.resolved;
  if (head.seen) tower.lm_head = head.resolved;
  return tower;
}

std::string ResolveQwen3_5BackbonePrefix(
    const std::vector<std::string>& tensor_names) {
  // `model.language_model.` is tested FIRST because it is also a `model.`
  // name: a plain "starts with model." test would match both spellings.
  const bool vl = HasBackboneUnder(tensor_names, kQwen3_5VlBackbonePrefix);
  // ...so the canonical probe must EXCLUDE the VL-prefixed names, which the
  // backbone spellings above already do (`model.language_model.` is neither
  // `model.embed_tokens.weight`, nor `model.norm.weight`, nor `model.layers.`).
  const bool flat = HasBackboneUnder(tensor_names, kQwen3_5TextBackbonePrefix);
  VT_CHECK(!(vl && flat),
           "qwen3_5 weights: checkpoint carries backbone tensors under BOTH "
           "\"model.language_model.\" and \"model.\"; refusing a mixed weight "
           "namespace rather than binding half the model from each");
  VT_CHECK(vl || flat,
           "qwen3_5 weights: no Qwen3.5 backbone tensors found under either "
           "\"model.language_model.\" or \"model.\"");
  return std::string(vl ? kQwen3_5VlBackbonePrefix
                        : kQwen3_5TextBackbonePrefix);
}

// External-linkage seam so the DENSE loader can keep an FP8 GDN tower native.
Fp8Weight LoadFp8RawShared(const TensorResolver& get, const std::string& proj) {
  return LoadFp8Raw(get, proj);
}

// --- The load plan (issue #740) ---------------------------------------------
//
// One function per loader helper, in the same order and with the same names, so
// the two can be read side by side. Nothing below allocates a tensor or
// dereferences a weight byte; the whole 4.8 TB 2.4T plan is ~1600 strings.
namespace {

void PlanBf16(std::vector<PlannedTensor>& p, const std::string& name,
              std::vector<int64_t> shape) {
  p.push_back({name, "BF16", std::move(shape), /*shape_enforced=*/false});
}

// Mirror of `LoadFp8Raw` / `LoadFp8Transposed`. Which of the two runs is
// `DenseNativeEnabled()`, and it is a REQUEST-SET difference, not a cosmetic
// one: the raw arm additionally reads `.input_scale`. The plan calls the same
// predicate rather than assuming an arm, so a build where the two disagree is
// impossible by construction.
void PlanFp8(std::vector<PlannedTensor>& p, const std::string& proj,
             std::vector<int64_t> shape) {
  p.push_back({proj + ".weight", "F8_E4M3", std::move(shape), false});
  p.push_back({proj + ".weight_scale", "F32", {}, false});
  if (DenseNativeEnabled()) {
    p.push_back({proj + ".input_scale", "F32", {}, false});
  }
}

// Mirror of `LoadNvfp4Raw`: U8 [out, in/2] + F8_E4M3 [out, in/16] + an F32
// scalar. `out`/`in` are the LOGICAL projection dims; the packed shapes are
// derived here the same way the reader derives them in reverse.
void PlanNvfp4(std::vector<PlannedTensor>& p, const std::string& proj,
               int64_t out_dim, int64_t in_dim) {
  const bool known = out_dim > 0 && in_dim > 0;
  p.push_back({proj + ".weight", "U8",
               known ? std::vector<int64_t>{out_dim, in_dim / 2}
                     : std::vector<int64_t>{},
               false});
  p.push_back({proj + ".weight_scale", "F8_E4M3",
               known ? std::vector<int64_t>{out_dim, in_dim / 16}
                     : std::vector<int64_t>{},
               false});
  p.push_back({proj + ".weight_scale_2", "F32", {}, false});
}

// Mirror of the TOWER dispatch `LoadGdn` / `LoadAttn` perform (issue #864). A
// BF16 projection asks for ONE tensor where an FP8 one asks for two or three,
// so the arm is a request-set difference and the plan must take the same one.
void PlanTowerProjection(std::vector<PlannedTensor>& p, const std::string& proj,
                         std::vector<int64_t> shape, MoeProjDtype dtype) {
  if (dtype == MoeProjDtype::kBf16) {
    PlanBf16(p, proj + ".weight", std::move(shape));
    return;
  }
  VT_CHECK(dtype == MoeProjDtype::kFp8,
           "qwen3_5 weights: the MoE attention/GDN tower reads BF16 or "
           "per-tensor FP8; there is no plan for a " +
               std::string(MoeProjDtypeName(dtype)) + " " + proj);
  PlanFp8(p, proj, std::move(shape));
}

// Mirror of the SINK dispatch (`LoadMoe`'s shared expert, and `lm_head` in
// `LoadQwen3_5Moe`): BF16 `[out, in]` verbatim, or the NVFP4 triple.
void PlanSinkProjection(std::vector<PlannedTensor>& p, const std::string& proj,
                        int64_t out_dim, int64_t in_dim, MoeProjDtype dtype) {
  if (dtype == MoeProjDtype::kBf16) {
    const bool known = out_dim > 0 && in_dim > 0;
    PlanBf16(p, proj + ".weight",
             known ? std::vector<int64_t>{out_dim, in_dim}
                   : std::vector<int64_t>{});
    return;
  }
  VT_CHECK(dtype == MoeProjDtype::kNvfp4,
           "qwen3_5 weights: the MoE shared expert and lm_head read BF16 or "
           "NVFP4; there is no plan for a " +
               std::string(MoeProjDtypeName(dtype)) + " " + proj);
  PlanNvfp4(p, proj, out_dim, in_dim);
}

// Mirror of `LoadGdn`.
void PlanGdn(std::vector<PlannedTensor>& p, const HfConfig& c,
             const std::string& base, MoeProjDtype tower_dtype) {
  const std::string la = base + "linear_attn.";
  const int64_t h = c.hidden_size;
  // The GDN input projection packs q, k and v: two key groups of
  // linear_key_head_dim plus the value group. Confirmed against BOTH published
  // repos' headers — 2.4T [20480, 8192] (2*16*128 + 128*128) and 35B
  // [8192, 2048] (2*16*128 + 32*128).
  const int64_t qkv =
      2 * c.linear_num_key_heads * c.linear_key_head_dim +
      c.linear_num_value_heads * c.linear_value_head_dim;
  const int64_t v_dim = c.linear_num_value_heads * c.linear_value_head_dim;
  PlanTowerProjection(p, la + "in_proj_qkv", {qkv, h}, tower_dtype);
  PlanTowerProjection(p, la + "in_proj_z", {v_dim, h}, tower_dtype);
  PlanTowerProjection(p, la + "out_proj", {h, v_dim}, tower_dtype);
  PlanBf16(p, la + "in_proj_b.weight", {c.linear_num_value_heads, h});
  PlanBf16(p, la + "in_proj_a.weight", {c.linear_num_value_heads, h});
  // `LoadGdn` REQUIRES rank 3 with a singleton middle before collapsing it.
  PlanBf16(p, la + "conv1d.weight", {qkv, 1, c.linear_conv_kernel_dim});
  // A_log / dt_bias are per VALUE HEAD; norm.weight is per value-head DIM. The
  // 2.4T cannot tell those apart (both 128) — the 35B can, and does: A_log [32]
  // against norm.weight [128].
  PlanBf16(p, la + "A_log", {c.linear_num_value_heads});
  PlanBf16(p, la + "dt_bias", {c.linear_num_value_heads});
  PlanBf16(p, la + "norm.weight", {c.linear_value_head_dim});
}

// Mirror of `LoadAttn`. The four projections carry NO shape: their output width
// depends on `attn_output_gate`, which `HfConfig` does not carry, so this
// planner declines to state one rather than state a wrong one.
void PlanAttn(std::vector<PlannedTensor>& p, const HfConfig& c,
              const std::string& base, MoeProjDtype tower_dtype) {
  const std::string sa = base + "self_attn.";
  PlanTowerProjection(p, sa + "q_proj", {}, tower_dtype);
  PlanTowerProjection(p, sa + "k_proj", {}, tower_dtype);
  PlanTowerProjection(p, sa + "v_proj", {}, tower_dtype);
  PlanTowerProjection(p, sa + "o_proj", {}, tower_dtype);
  PlanBf16(p, sa + "q_norm.weight", {c.head_dim});
  PlanBf16(p, sa + "k_norm.weight", {c.head_dim});
}

// Mirror of `LoadMoe` + `LoadMoeExpertsInto` / `LoadMoeExpertsStackedBf16Into`.
void PlanMoe(std::vector<PlannedTensor>& p, const HfConfig& c,
             const std::string& base, MoeExpertLayout layout,
             MoeProjDtype shared_dtype) {
  const std::string mlp = base + "mlp.";
  const int64_t h = c.hidden_size;
  const int64_t inter = c.moe_intermediate_size;
  PlanBf16(p, mlp + "gate.weight", {c.num_experts, h});
  PlanBf16(p, mlp + "shared_expert_gate.weight", {1, h});
  if (layout == MoeExpertLayout::kStackedBf16) {
    // THE ONE SHAPE THE LOADER ITSELF ENFORCES, and the one this row added.
    // Upstream's canonical orientation: `gate_up_proj` is [E, 2I, H] with hidden
    // LAST and gate the FIRST half of dim 1; `down_proj` is [E, H, I] with
    // hidden at dim -2 (routed_experts.py:923-933, :1081-1082 @ 555967922).
    // The reader also accepts the transposed spelling, exactly as upstream's
    // `shape[-1] != hidden` / `shape[-2] != hidden` probes do, so this is the
    // shape a PUBLISHED repo has rather than the only one that would load.
    p.push_back({mlp + "experts.gate_up_proj", "BF16",
                 {c.num_experts, 2 * inter, h}, /*shape_enforced=*/true});
    p.push_back({mlp + "experts.down_proj", "BF16",
                 {c.num_experts, h, inter}, /*shape_enforced=*/true});
  } else {
    for (int64_t e = 0; e < c.num_experts; ++e) {
      const std::string ex = mlp + "experts." + std::to_string(e) + ".";
      PlanNvfp4(p, ex + "gate_proj", inter, h);
      PlanNvfp4(p, ex + "up_proj", inter, h);
      PlanNvfp4(p, ex + "down_proj", h, inter);
    }
  }
  const std::string se = mlp + "shared_expert.";
  const int64_t si = c.shared_expert_intermediate_size;
  PlanSinkProjection(p, se + "gate_proj", si, h, shared_dtype);
  PlanSinkProjection(p, se + "up_proj", si, h, shared_dtype);
  PlanSinkProjection(p, se + "down_proj", h, si, shared_dtype);
}

}  // namespace

std::vector<PlannedTensor> PlanQwen3_5MoeLoad(const HfConfig& config,
                                              const std::string& backbone_prefix,
                                              MoeExpertLayout layout,
                                              Qwen3_5MoeTowerDtypes tower) {
  // The same two preconditions `LoadQwen3_5Moe` checks before it reads anything.
  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 weights: layer_types size must equal num_hidden_layers");
  VT_CHECK(config.num_experts > 0,
           "qwen3_5 weights: num_experts must be > 0 for the MoE model");

  std::vector<PlannedTensor> p;
  PlanBf16(p, backbone_prefix + "embed_tokens.weight",
           {config.vocab_size, config.hidden_size});
  PlanBf16(p, backbone_prefix + "norm.weight", {config.hidden_size});
  PlanSinkProjection(p, "lm_head", config.vocab_size, config.hidden_size,
                     tower.lm_head);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    const std::string base =
        backbone_prefix + "layers." + std::to_string(l) + ".";
    PlanBf16(p, base + "input_layernorm.weight", {config.hidden_size});
    PlanBf16(p, base + "post_attention_layernorm.weight", {config.hidden_size});
    const std::string& type = config.layer_types[static_cast<size_t>(l)];
    if (type == "linear_attention") {
      PlanGdn(p, config, base, tower.gdn);
    } else if (type == "full_attention") {
      PlanAttn(p, config, base, tower.attn);
    } else {
      VT_CHECK(false, "qwen3_5 weights: unknown layer_type " + type);
    }
    PlanMoe(p, config, base, layout, tower.shared_expert);
  }
  return p;
}

Qwen3_5MoeLayerWeights LoadQwen3_5MoeLayer(const TensorResolver& get,
                                           const std::string& layer_type,
                                           int64_t layer_idx,
                                           int64_t num_experts,
                                           const std::string& backbone_prefix,
                                           MoeExpertLayout layout,
                                           int64_t hidden,
                                           Qwen3_5MoeTowerDtypes tower) {
  return LoadLayerImpl(get, layer_type, layer_idx, num_experts,
                       /*with_experts=*/true, backbone_prefix, layout, hidden,
                       tower);
}

Qwen3_5MoeWeights LoadQwen3_5Moe(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    std::shared_ptr<const std::vector<SafetensorsFile>> shards_owner) {
  // Build a name -> shard index from each shard's own header. The target loader
  // does not request mtp.*: LoadQwen3_5MTP loads that optional draft head only
  // when speculative decoding is enabled. The resolver throws on missing names.
  // The map is heap-owned (shared_ptr) so the deferred per-layer expert loader
  // below can reuse it without rebuilding — its entries point into `shards`,
  // which `shards_owner` keeps mmap'd.
  auto where =
      std::make_shared<std::unordered_map<std::string, const SafetensorsFile*>>();
  std::vector<std::string> all_names;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      (*where)[name] = &shard;
      all_names.push_back(name);
    }
  }
  // ONE namespace decision for the whole checkpoint (qwen3_5_weights.h): the
  // VL-nested spelling for the wrappers we gate, the flat `model.` spelling for
  // a text-only arm, and a refusal for a mixed index.
  const std::string backbone = ResolveQwen3_5BackbonePrefix(all_names);
  // ...and ONE decision about the routed-expert layout (issue #740), from the
  // same index and in the same shape: per-expert NVFP4 or 3-D stacked bf16,
  // decided here and THREADED, never re-probed per lookup.
  const MoeExpertLayout expert_layout =
      ResolveQwen3_5MoeExpertLayout(all_names, backbone);
  // ...and ONE decision per NON-routed component (issue #864): the GDN tower,
  // the attention tower, the shared expert and `lm_head`, each resolved from the
  // same index by the same ladder the DENSE loader uses, and THREADED. A
  // per-lookup probe could answer differently per layer, which is how a
  // checkpoint ends up half quantized and half bf16 while still appearing to
  // load. Reading only the header, so no weight byte is touched here.
  const TensorDtypeProbe dtype_of =
      [&where](const std::string& name) -> std::string {
    auto it = where->find(name);
    if (it == where->end()) return std::string();
    return it->second->Get(name).dtype;
  };
  const Qwen3_5MoeTowerDtypes tower =
      ResolveQwen3_5MoeTowerDtypes(dtype_of, backbone, config.layer_types);
  // ...and then, before any tensor is touched, an arm we do not implement is
  // refused by name rather than discovered as a dtype complaint about `lm_head`
  // (issue #490).
  CheckMoeQuantLayoutSupported(all_names, backbone, expert_layout, tower);
  const TensorResolver get =
      [where](const std::string& name) -> const StTensor& {
    auto it = where->find(name);
    VT_CHECK(it != where->end(), "qwen3_5 weights: tensor not found: " + name);
    return it->second->Get(name);
  };

  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 weights: layer_types size must equal num_hidden_layers");
  VT_CHECK(config.num_experts > 0,
           "qwen3_5 weights: num_experts must be > 0 for the MoE model");

  // With a shared owner we DEFER the routed-expert host copies: load every layer
  // WITHOUT its experts now (small: norms, attn/GDN, router, shared expert), and
  // install a closure that materializes ONE layer's experts on demand during
  // PrepareMarlinResident — bounding peak host residency to a single layer's
  // ~256 experts instead of all N layers coexisting (the 35B load-phase peak-PSS
  // lever). Without an owner the shards may be released right after this returns,
  // so the experts must be loaded eagerly (GGUF/synthetic/borrowed paths).
  const bool defer_experts = shards_owner != nullptr;

  Qwen3_5MoeWeights w;
  w.embed_tokens = LoadBf16Direct(get, backbone + "embed_tokens.weight");
  w.final_norm = LoadBf16Direct(get, backbone + "norm.weight");
  // THE BF16 OUTPUT HEAD (issue #864): `lm_head.weight` [vocab, H] -> owned bf16
  // [H, vocab], the Matmul-B orientation `DenseLmHead`/`MoeLogits` already reads
  // on the GGUF and tied paths (`qwen3_5_weights.h:469`, `qwen3_5.cpp:7101`).
  // Exactly one of `lm_head` and `lm_head_fp4` is populated, and the NVFP4 arm
  // is byte-unchanged for every gated row.
  if (tower.lm_head == MoeProjDtype::kBf16) {
    w.lm_head = LoadBf16Transposed(get, "lm_head.weight");
  } else {
    w.lm_head_fp4 = LoadNvfp4Raw(get, "lm_head");  // M2.2b fp4-resident
  }
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    w.layers.push_back(LoadLayerImpl(get,
                                     config.layer_types[static_cast<size_t>(l)],
                                     l, config.num_experts,
                                     /*with_experts=*/!defer_experts, backbone,
                                     expert_layout, config.hidden_size, tower));
  }

  if (defer_experts) {
    const int64_t num_experts = config.num_experts;
    const int64_t hidden = config.hidden_size;
    // Captures the shared name-map AND the shards owner (keepalive): the mmap'd
    // shards stay valid until PrepareMarlinResident resets this closure after the
    // last layer is built. Does NOT capture the (movable) Qwen3_5MoeWeights — the
    // target MoE block is passed in by reference, so the closure survives the
    // model's move into the LoadedModel.
    // `backbone` is captured BY VALUE: the closure outlives this frame, and it
    // must keep using the ONE namespace resolved above rather than re-deciding.
    // `expert_layout` is captured for exactly the same reason (#740) — the
    // layout is a property of the CHECKPOINT, decided once from its index, and a
    // closure that re-probed per layer could answer differently per layer.
    w.load_layer_experts = [where, shards_owner, num_experts, backbone,
                            expert_layout, hidden](int64_t layer,
                                                   MoeBlockWeights& moe) {
      const TensorResolver g =
          [where](const std::string& name) -> const StTensor& {
        auto it = where->find(name);
        VT_CHECK(it != where->end(),
                 "qwen3_5 weights: tensor not found: " + name);
        return it->second->Get(name);
      };
      const std::string mlp =
          backbone + "layers." + std::to_string(layer) + ".mlp.";
      if (expert_layout == MoeExpertLayout::kStackedBf16) {
        LoadMoeExpertsStackedBf16Into(g, mlp, num_experts, hidden, moe);
      } else {
        LoadMoeExpertsInto(g, mlp, num_experts, moe);
      }
    };
  }
  return w;
}

// ── The MoE arm's VISION TOWER (issue #891) ──────────────────────────────────
// See the header for why this exists and why it refuses rather than shrugs.

bool HasQwen3_5MoeVisionTower(const std::vector<SafetensorsFile>& shards) {
  static const std::string kVisual = "model.visual.";
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names())
      if (name.compare(0, kVisual.size(), kVisual) == 0) return true;
  return false;
}

// The tower geometry for a Qwen3.5 conditional-generation checkpoint, shared by
// BOTH arms (full argument in qwen3_5_weights.h, above the two public wrappers).
// The family is NOT one tower: the published 27B/35B checkpoints ship
// depth 27 / hidden 1152 / intermediate 4304, while Qwen3.5-4B ships
// depth 24 / hidden 1024 / intermediate 4096 (its vision_config mirrors the
// Qwen3-VL-4B tower). Every member declares these values in config.json's
// `vision_config`, so the constants below are ONLY the fallback for a config
// that omits the block — the checkpoint's own declaration is authoritative
// (ISSUE-LOCAL-01M3SQMXYKT2RV2D6MHCY2FCAC: hardcoding the big towers made the
// loader index blocks.0..26 against a 24-block 4B checkpoint and fatal on
// `model.visual.blocks.24.norm1.weight`).
multimodal::Qwen3VLVisionConfig Qwen3_5FamilyVisionConfig(
    const HfConfig& config) {
  multimodal::Qwen3VLVisionConfig v;
  v.hidden_size = 1152;
  v.num_heads = 16;
  v.depth = 27;
  v.intermediate_size = 4304;
  // The merger writes straight into the text residual stream, so the tower's
  // output width IS the text hidden size (2048 on the 35B MoE, 5120 on the 27B
  // dense) — the ONE field the two arms disagree on.
  v.out_hidden_size = config.hidden_size;
  v.patch_size = 16;
  v.temporal_patch_size = 2;
  v.spatial_merge_size = 2;
  v.num_position_embeddings = 2304;
  v.in_channels = 3;
  v.deepstack_visual_indexes = {};  // NO DeepStack on this family.
  v.norm_eps = 1e-6f;
  // The values above are the 27B/35B tower, kept as the fallback for a config
  // that omits a key. Upstream sizes the tower from config.vision_config
  // (qwen3_vl.py:536-628 @ 5559679229), and the 0.8B/2B/4B checkpoints ship a
  // smaller one (0.8B: depth 12, hidden 768), so the checkpoint's value wins.
  // ISSUE-LOCAL-01M3RT4GVEY4QBE5AYBT8RDM89.
  const auto vc = config.raw.find("vision_config");
  if (vc != config.raw.end() && vc->is_object()) {
    const auto read = [&vc](const char* key, int64_t& field) {
      const auto it = vc->find(key);
      if (it != vc->end() && it->is_number_integer()) field = it->get<int64_t>();
    };
    read("hidden_size", v.hidden_size);
    read("num_heads", v.num_heads);
    read("depth", v.depth);
    read("intermediate_size", v.intermediate_size);
    read("out_hidden_size", v.out_hidden_size);
    read("patch_size", v.patch_size);
    read("temporal_patch_size", v.temporal_patch_size);
    read("spatial_merge_size", v.spatial_merge_size);
    read("num_position_embeddings", v.num_position_embeddings);
    read("in_channels", v.in_channels);
    const auto ds = vc->find("deepstack_visual_indexes");
    if (ds != vc->end() && ds->is_array())
      v.deepstack_visual_indexes = ds->get<std::vector<int>>();
    // norm_eps is a float, outside the int64_t `read` lambda; a checkpoint
    // that publishes it wins over the 1e-6f fallback.
    const auto ne = vc->find("norm_eps");
    if (ne != vc->end() && ne->is_number())
      v.norm_eps = ne->get<float>();
  }
  return v;
}

multimodal::Qwen3VLVisionConfig Qwen3_5MoeVisionConfig(const HfConfig& config) {
  return Qwen3_5FamilyVisionConfig(config);
}

multimodal::Qwen3VLVisionConfig Qwen3_5DenseVisionConfig(const HfConfig& config) {
  return Qwen3_5FamilyVisionConfig(config);
}

bool HasQwen3_5DenseVisionTower(const std::vector<SafetensorsFile>& shards) {
  // Identical probe to the MoE arm's: the tower prefix is a FAMILY property
  // (`model.visual.*` on every Qwen3.5 *ForConditionalGeneration checkpoint),
  // not an arm property.
  return HasQwen3_5MoeVisionTower(shards);
}

multimodal::Qwen3VLVisionWeights LoadQwen3_5DenseVision(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  VT_CHECK(HasQwen3_5DenseVisionTower(shards),
           "qwen3_5 dense vision: this checkpoint carries NO `model.visual.*` "
           "tensors, so it has no vision tower and cannot answer an image or "
           "video prompt. A Qwen3.5-family *ForConditionalGeneration checkpoint "
           "publishes its tower as `model.visual.patch_embed.proj.{weight,bias}`,"
           " `model.visual.pos_embed.weight`, `model.visual.blocks.<0..depth-1>."
           "{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}."
           "{weight,bias}` and `model.visual.merger.*` (333 tensors on "
           "Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw). Load the vision-inclusive repo; "
           "text-only repos declare `vision_config` but ship no `visual.*` "
           "weights.");
  return LoadQwen3VLVisionWeights(shards, Qwen3_5DenseVisionConfig(config));
}

multimodal::Qwen3VLVisionWeights LoadQwen3_5MoeVision(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  VT_CHECK(HasQwen3_5MoeVisionTower(shards),
           "qwen3_5 moe vision: this checkpoint carries NO `model.visual.*` "
           "tensors, so it has no vision tower and cannot answer an image or "
           "video prompt. A Qwen3.5-family *ForConditionalGeneration checkpoint "
           "publishes its tower as `model.visual.patch_embed.proj.{weight,bias}`,"
           " `model.visual.pos_embed.weight`, `model.visual.blocks.<0..depth-1>."
           "{norm1,norm2,attn.qkv,attn.proj,mlp.linear_fc1,mlp.linear_fc2}."
           "{weight,bias}` and `model.visual.merger.*` (333 tensors on "
           "Qwen/Qwen3.6-35B-A3B). Load the vision-inclusive repo; the text-only "
           "and NVFP4-requant repos (e.g. nvidia/Qwen3.6-35B-A3B-NVFP4) declare "
           "`vision_config` but ship no `visual.*` weights.");
  return LoadQwen3VLVisionWeights(shards, Qwen3_5MoeVisionConfig(config));
}

}  // namespace vllm
