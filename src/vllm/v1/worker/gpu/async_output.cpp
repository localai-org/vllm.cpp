// Ported from: vllm/v1/worker/gpu/async_utils.py:12-70 +
// vllm/v1/worker/gpu/gpu_model_runner.py:242-332 @ e24d1b24. See async_output.h
// for scope, device-neutrality, and the token-exactness argument.
#include "vllm/v1/worker/gpu/async_output.h"

#include <cstdint>
#include <utility>

namespace vllm::v1 {

// ─── AsyncOutputPool ────────────────────────────────────────────────────────
// Persistent, reusable slots. Every device/pinned/event allocation happens here
// (ctor / rare grow) and is freed ONCE in the dtor — never on the decode hot
// path. Mirrors torch's caching device + pinned allocators (vLLM does no
// per-step raw cudaMalloc/cudaHostAlloc/cudaEventCreate on the sampled-id path).

AsyncOutputPool::AsyncOutputPool(vt::Device device, int capacity_elems,
                                 int initial_slots)
    : device_(device),
      backend_(&vt::GetBackend(device.type)),
      capacity_elems_(capacity_elems < 1 ? 1 : capacity_elems) {
  for (int i = 0; i < (initial_slots < 1 ? 1 : initial_slots); ++i) {
    slots_.push_back(MakeSlot());
  }
}

AsyncOutputPool::~AsyncOutputPool() {
  for (auto& slot : slots_) {
    if (slot == nullptr) continue;
    backend_->DestroyEvent(slot->fork_event);
    backend_->DestroyEvent(slot->ready_event);
    if (slot->pinned_host != nullptr) backend_->FreePinned(slot->pinned_host);
    if (slot->device_sampled_ids != nullptr) {
      vt::Free(device_, slot->device_sampled_ids);
    }
  }
}

std::unique_ptr<AsyncOutputSlot> AsyncOutputPool::MakeSlot() {
  auto slot = std::make_unique<AsyncOutputSlot>();
  const size_t bytes = static_cast<size_t>(capacity_elems_) * sizeof(int64_t);
  slot->device_sampled_ids = vt::Alloc(device_, bytes);
  slot->pinned_host = static_cast<int64_t*>(backend_->AllocPinned(bytes));
  slot->fork_event = backend_->CreateEvent();
  slot->ready_event = backend_->CreateEvent();
  slot->capacity = capacity_elems_;
  slot->in_use = false;
  return slot;
}

AsyncOutputSlot* AsyncOutputPool::Acquire() {
  for (auto& slot : slots_) {
    if (!slot->in_use) {
      slot->in_use = true;
      return slot.get();
    }
  }
  // Rare miss (more in flight than pre-allocated): grow by one. This is a
  // one-time cudaMalloc/cudaHostAlloc, NOT a per-step cost — the pool never
  // shrinks, so steady state converges to zero growth.
  slots_.push_back(MakeSlot());
  AsyncOutputSlot* slot = slots_.back().get();
  slot->in_use = true;
  return slot;
}

void AsyncOutputPool::Release(AsyncOutputSlot* slot) {
  if (slot != nullptr) slot->in_use = false;
}

// ─── AsyncGPUModelRunnerOutput ──────────────────────────────────────────────

AsyncGPUModelRunnerOutput::AsyncGPUModelRunnerOutput(
    ModelRunnerOutput skeleton, vt::Device device, AsyncOutputPool& pool,
    AsyncOutputSlot* slot, int num_reqs, vt::Queue& main_q, vt::Queue& copy_q,
    std::vector<int32_t> invalid_req_indices)
    : output_(std::move(skeleton)),
      backend_(&vt::GetBackend(device.type)),
      device_(device),
      pool_(&pool),
      slot_(slot),
      num_reqs_(num_reqs),
      invalid_req_indices_(std::move(invalid_req_indices)) {
  // copy_stream.wait_stream(default_stream) (async_utils.py:29): make the copy
  // queue wait for the sampling work on the main queue, WITHOUT blocking the
  // host. Implemented as record-on-main + queue-wait-on-copy using the slot's
  // persistent fork event (re-recorded each step; no per-step CreateEvent).
  backend_->RecordEvent(slot_->fork_event, main_q);
  backend_->QueueWaitEvent(copy_q, slot_->fork_event);

  // async_copy_to_np (async_utils.py:32,108-109): the NON-BLOCKING D2H of the
  // sampled ids into the slot's persistent pinned host buffer. The main queue is
  // never synchronized.
  const size_t bytes = static_cast<size_t>(num_reqs_) * sizeof(int64_t);
  if (bytes != 0) {
    backend_->Copy(copy_q, slot_->pinned_host, slot_->device_sampled_ids, bytes);
  }

  // copy_event.record(copy_stream) (async_utils.py:44): the completion event the
  // host will later wait on (and only this — never the main queue).
  backend_->RecordEvent(slot_->ready_event, copy_q);
}

AsyncGPUModelRunnerOutput::~AsyncGPUModelRunnerOutput() { ReleaseSlot(); }

void AsyncGPUModelRunnerOutput::ReleaseSlot() {
  if (slot_ != nullptr && pool_ != nullptr) {
    // Cancellation can destroy an output before get_output() waits its DMA.
    // Reusing either buffer or re-recording the events before that wait races
    // the old copy with the next batch, even though no caller wants its ids.
    if (!consumed_) backend_->SynchronizeEvent(slot_->ready_event);
    pool_->Release(slot_);
    slot_ = nullptr;
  }
}

ModelRunnerOutput AsyncGPUModelRunnerOutput::get_output() {
  VT_CHECK(!consumed_ && slot_ != nullptr,
           "async output has already been consumed; sampled tokens may be gathered only once");
  // async_utils.py:48 self.copy_event.synchronize(): the ONE blocking wait, and
  // it waits the COPY queue's event — the main queue was never blocked, so the
  // next step's forward overlaps this copy.
  backend_->SynchronizeEvent(slot_->ready_event);

  // async_utils.py:53-58 / gpu_model_runner.py:301-304: materialize the ragged
  // per-request sampled_token_ids from the (now host-resident) ids. T0 non-spec
  // decode is exactly one bonus token per request ([num_reqs, 1]).
  output_.sampled_token_ids.assign(static_cast<size_t>(num_reqs_), {});
  for (int i = 0; i < num_reqs_; ++i) {
    output_.sampled_token_ids[static_cast<size_t>(i)] = {
        static_cast<int32_t>(slot_->pinned_host[i])};
  }
  // discard_request_mask (vllm/v1/outputs.py:303-304): clear the sampled token
  // for any request still consuming its known prefill tokens this step. The
  // scheduler expects EMPTY token ids for a still-prefilling request
  // (scheduler.py:1888-1890); leaving the (garbage) prefill-position sample in
  // would append a spurious output token and, under async scheduling, underflow
  // num_output_placeholders.
  for (const int32_t i : invalid_req_indices_) {
    if (i >= 0 && i < num_reqs_) {
      output_.sampled_token_ids[static_cast<size_t>(i)].clear();
    }
  }
  consumed_ = true;
  // Return the slot to the pool now that both its device buffer (copied) and its
  // pinned buffer (read above) are done with — the copy completed at the event
  // synchronize, so the buffers are free for the next Acquire().
  ReleaseSlot();
  return std::move(output_);
}

}  // namespace vllm::v1
