#if !defined(_WIN32)
#include <unistd.h>
#endif
#include <cerrno>
// ENG-EXPERT-STREAM W3. See expert_streamer.h for why the destination is an
// interface and why the fill is synchronous.
#include "vllm/model_executor/expert_streamer.h"

#include <stdexcept>
#include <string>

namespace vllm {

ExpertStreamer::ExpertStreamer(ExpertSlotCache& cache, ExpertSlotStore& store)
    : cache_(cache), store_(store) {
  if (store.slot_count() != cache.capacity()) {
    // The cache hands out slot indices in [0, capacity). If the store holds
    // fewer, a perfectly ordinary eviction would write past the end of the slot
    // array, so this disagreement is refused at construction rather than on the
    // first unlucky step.
    throw std::invalid_argument(
        "expert streamer: store holds " + std::to_string(store.slot_count()) +
        " slots but the cache has capacity " + std::to_string(cache.capacity()));
  }
}

ExpertStreamer::Result ExpertStreamer::EnsureFile(const ExpertKey& key, int fd,
                                                 size_t file_offset,
                                                 size_t bytes) {
#if defined(_WIN32)
  (void)fd;
  (void)file_offset;
  (void)bytes;
  (void)key;
  throw std::invalid_argument("expert streamer: EnsureFile needs pread");
#else
  Result out;
  // Size first, for the same reason as the other two overloads: a slice that
  // cannot be stored must not evict a resident one on its way to being refused.
  if (bytes > store_.slot_bytes()) {
    throw std::invalid_argument(
        "expert streamer: expert span is " + std::to_string(bytes) +
        " bytes but a slot holds " + std::to_string(store_.slot_bytes()));
  }
  if (fd < 0) throw std::invalid_argument("expert streamer: bad descriptor");

  const ExpertAcquisition acq = cache_.Acquire(key);
  if (acq.slot < 0) return out;
  if (acq.slot >= store_.slot_count()) {
    throw std::out_of_range(
        "expert streamer: cache returned slot " + std::to_string(acq.slot) +
        " but the store holds " + std::to_string(store_.slot_count()));
  }

  out.slot = acq.slot;
  if (acq.hit) {
    out.hit = true;  // resident: no syscall at all, which is the point
    return out;
  }

  // pread in a loop: a short read is legal and must be finished, not accepted.
  // The destination is whatever the store says is host-writable, so on the host
  // store the bytes never pass through a staging buffer or the page tables of
  // the mapping at all. On a store whose slots are DEVICE memory that pointer
  // cannot be the slot -- `vt::Backend::DeviceMemoryIsHostAddressable()` is
  // false for CUDA -- so `SlotForWrite` hands back a staging buffer and
  // `CommitSlot` below publishes it. Without that pair a device store could not
  // be filled by this function at all, which is issue #1124's third piece and
  // the reason the contract change rides in W1 rather than after it.
  //
  // THE ACQUISITION IS UNDONE IF THE READ THROWS, and that is the whole reason
  // this loop sits inside a try. Acquire must run first, because the read needs
  // a destination, so at this point the cache already says the key is resident.
  // A throw from here would unwind past that claim and leave the entry standing
  // over a slot holding `done` correct bytes and `bytes - done` bytes of the
  // expert that used to live there. Nothing reads the exception as data: the
  // next acquisition of the same key is an ordinary HIT, no read is issued
  // because a hit moves no bytes, and the GEMM multiplies half of one expert
  // spliced onto half of another. That is silent, plausible, and wrong, which is
  // the exact failure this row exists to prevent. Undoing the acquisition turns
  // it into a retryable miss instead.
  uint8_t* dst = store_.SlotForWrite(acq.slot);
  size_t done = 0;
  try {
    while (done < bytes) {
      const ssize_t n = ::pread(fd, dst + done, bytes - done,
                                static_cast<off_t>(file_offset + done));
      if (n < 0) {
        if (errno == EINTR) continue;
        throw std::runtime_error("expert streamer: pread failed with errno " +
                                 std::to_string(errno));
      }
      if (n == 0) {
        throw std::runtime_error(
            "expert streamer: short read, " + std::to_string(done) + " of " +
            std::to_string(bytes) + " bytes at offset " +
            std::to_string(file_offset));
      }
      done += static_cast<size_t>(n);
    }
    // INSIDE the try, deliberately. Publishing is the last step of the fill and
    // it can fail for the same class of reason the read can -- a device copy
    // that throws leaves the slot holding whatever it held before, under a cache
    // entry that already claims the key is resident. That is the same silent,
    // plausible and wrong outcome the loop above is wrapped for, so it takes the
    // same undo.
    store_.CommitSlot(acq.slot, bytes);
  } catch (...) {
    cache_.Invalidate(key);
    throw;
  }

  bytes_filled_ += static_cast<int64_t>(bytes);
  ++fills_;
  out.filled = true;
  return out;
#endif
}

ExpertStreamer::Result ExpertStreamer::EnsureSpan(const ExpertKey& key,
                                                  const uint8_t* src,
                                                  size_t bytes) {
  Result out;

  // Same ordering rule as the tensor overload, and for the same reason: check
  // the size BEFORE acquiring, or a slice that cannot be stored still evicts a
  // resident expert on its way to being refused.
  if (bytes > store_.slot_bytes()) {
    throw std::invalid_argument(
        "expert streamer: expert span is " + std::to_string(bytes) +
        " bytes but a slot holds " + std::to_string(store_.slot_bytes()));
  }
  if (src == nullptr && bytes > 0) {
    throw std::invalid_argument("expert streamer: null span with bytes > 0");
  }

  const ExpertAcquisition acq = cache_.Acquire(key);
  if (acq.slot < 0) return out;  // exhausted: caller reports, never works around
  if (acq.slot >= store_.slot_count()) {
    throw std::out_of_range(
        "expert streamer: cache returned slot " + std::to_string(acq.slot) +
        " but the store holds " + std::to_string(store_.slot_count()));
  }

  out.slot = acq.slot;
  if (acq.hit) {
    out.hit = true;  // resident: no bytes move, which is the entire saving
    return out;
  }

  // THE ACQUISITION IS UNDONE IF THE WRITE THROWS, for the same reason the read
  // in `EnsureFile` is wrapped, one call earlier. Acquire must run first because
  // the write needs a slot, so by the time `WriteSlot` throws the cache already
  // says the key is resident -- over a slot still holding the expert that
  // acquisition just evicted. A throw that escapes leaves that entry standing,
  // the next request for the key is an ordinary HIT, no bytes move because a hit
  // moves none, and the GEMM multiplies the evicted expert. Silent, plausible
  // and wrong. Before ENG-EXPERT-STREAM-DEVICE W1 no store could reach this:
  // every `WriteSlot` was a memcpy. `DeviceExpertSlotStore::WriteSlot` calls
  // `vt::Backend::Copy` and `Synchronize`, and both throw on CUDA.
  try {
    store_.WriteSlot(acq.slot, src, bytes);
  } catch (...) {
    cache_.Invalidate(key);
    throw;
  }
  bytes_filled_ += static_cast<int64_t>(bytes);
  ++fills_;
  out.filled = true;
  return out;
}

ExpertStreamer::Result ExpertStreamer::Ensure(const ExpertKey& key,
                                              const GgufTensorInfo& tensor,
                                              const GgufExpertLayout& layout) {
  Result out;

  // The size check happens BEFORE the cache is touched. Acquiring first would
  // evict a resident expert to make room for one that cannot be stored, so a
  // configuration error would also destroy a good entry.
  if (layout.expert_bytes > store_.slot_bytes()) {
    throw std::invalid_argument(
        "expert streamer: expert is " + std::to_string(layout.expert_bytes) +
        " bytes but a slot holds " + std::to_string(store_.slot_bytes()) +
        " for " + tensor.name);
  }

  const ExpertAcquisition acq = cache_.Acquire(key);
  if (acq.slot < 0) {
    // The cache refused: every slot is in use by this step. Reported by
    // returning an invalid slot with neither flag set, so the caller cannot
    // mistake it for a hit or a fill.
    return out;
  }
  if (acq.slot >= store_.slot_count()) {
    throw std::out_of_range(
        "expert streamer: cache returned slot " + std::to_string(acq.slot) +
        " but the store holds " + std::to_string(store_.slot_count()));
  }

  out.slot = acq.slot;
  if (acq.hit) {
    // Resident already. No span is computed and no bytes move: this is the
    // saving the whole row exists to produce.
    out.hit = true;
    return out;
  }

  const GgufExpertSpan span = GgufExpertSpanOf(tensor, layout, key.expert);
  // Wrapped for the same reason as `EnsureSpan` above: a `try` on one entry
  // point leaves the other exactly as exposed as it was.
  try {
    store_.WriteSlot(acq.slot, span.data, span.bytes);
  } catch (...) {
    cache_.Invalidate(key);
    throw;
  }
  bytes_filled_ += static_cast<int64_t>(span.bytes);
  ++fills_;
  out.filled = true;
  return out;
}

}  // namespace vllm
