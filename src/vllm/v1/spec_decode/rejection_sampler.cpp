// Ported from: vllm/v1/worker/gpu/spec_decode/rejection_sampler.py +
// rejection_sampler_utils.py::rejection_sample @ e24d1b24. See
// include/vllm/v1/spec_decode/rejection_sampler.h for the exact greedy accept
// rule (with upstream file:line for every clause) and the deferred seams.
#include "vllm/v1/spec_decode/rejection_sampler.h"

#include <cstddef>
#include <limits>
#include <utility>

#include "vllm/v1/sample/device_scratch.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#ifdef VLLM_CPP_XPU
#include "vt/xpu_argmax_plan.h"
#include "vt/xpu_sampling.h"
#endif

namespace vllm::v1 {

// ─── RejectionSamplerDeviceOutput ───────────────────────────────────────────
// The accept walk's outputs, still on the device. See the header for why this
// object owns the kernel's INPUTS as well as its outputs.

RejectionSamplerDeviceOutput::~RejectionSamplerDeviceOutput() { Release(); }

RejectionSamplerDeviceOutput::RejectionSamplerDeviceOutput(
    RejectionSamplerDeviceOutput&& other) noexcept
    : backend_(other.backend_),
      device_(other.device_),
      sampled_(other.sampled_),
      num_sampled_(other.num_sampled_),
      target_argmax_(other.target_argmax_),
      num_reqs_(other.num_reqs_),
      width_(other.width_),
      verify_q_(other.verify_q_),
      copy_q_(other.copy_q_),
      draft_(std::move(other.draft_)),
      cu_(std::move(other.cu_)) {
  other.backend_ = nullptr;
  other.sampled_ = nullptr;
  other.num_sampled_ = nullptr;
  other.target_argmax_ = nullptr;
  other.num_reqs_ = 0;
  other.width_ = 0;
  // The moved-from object owns nothing, so it must not drain either.
  other.verify_q_ = nullptr;
  other.copy_q_ = nullptr;
}

RejectionSamplerDeviceOutput& RejectionSamplerDeviceOutput::operator=(
    RejectionSamplerDeviceOutput&& other) noexcept {
  if (this == &other) return *this;
  Release();
  backend_ = other.backend_;
  device_ = other.device_;
  sampled_ = other.sampled_;
  num_sampled_ = other.num_sampled_;
  target_argmax_ = other.target_argmax_;
  num_reqs_ = other.num_reqs_;
  width_ = other.width_;
  verify_q_ = other.verify_q_;
  copy_q_ = other.copy_q_;
  draft_ = std::move(other.draft_);
  cu_ = std::move(other.cu_);
  other.backend_ = nullptr;
  other.sampled_ = nullptr;
  other.num_sampled_ = nullptr;
  other.target_argmax_ = nullptr;
  other.num_reqs_ = 0;
  other.width_ = 0;
  other.verify_q_ = nullptr;
  other.copy_q_ = nullptr;
  return *this;
}

void RejectionSamplerDeviceOutput::Release() {
  if (backend_ != nullptr) {
    // DRAIN BEFORE FREE, and the header says why at length. Short version: the
    // five owned buffers are read and written by kernels and copies that are
    // still queued when `verify` returns, so freeing them is only safe after the
    // caller has waited. Nothing in the type system makes a caller wait, and the
    // one caller that matters destroys this object in a block scope
    // (`GPUModelRunner::sample_tokens_async`) whose wait A2-4 is about to move
    // out. So the destructor waits for the object's OWN work: the queue the
    // accept walk was issued on, plus the queue of the last `CopyToHost` when
    // that is a different one. A caller that already waited pays a drain on an
    // idle queue; a caller that did not pays a real one instead of emitting a
    // wrong accept prefix that no token gate in this tree can see (#1366).
    // On CPU both calls are the base no-op, so this changes nothing here.
    if (verify_q_ != nullptr) backend_->Synchronize(*verify_q_);
    if (copy_q_ != nullptr && copy_q_ != verify_q_) backend_->Synchronize(*copy_q_);
    if (sampled_ != nullptr) backend_->Free(sampled_);
    if (num_sampled_ != nullptr) backend_->Free(num_sampled_);
    // The argmax scratch is freed HERE and nowhere else. It was a process-global
    // in the CUDA backend, freed by whichever later call needed more rows, which
    // is a free under a still-queued accept kernel (SPEC-DFLASH2 A2-2, #2802).
    if (target_argmax_ != nullptr) backend_->Free(target_argmax_);
  }
  sampled_ = nullptr;
  num_sampled_ = nullptr;
  target_argmax_ = nullptr;
  verify_q_ = nullptr;
  copy_q_ = nullptr;
  // The staging goes with them: it is only alive because the kernel reads it.
  draft_.reset();
  cu_.reset();
}

void RejectionSamplerDeviceOutput::CopyToHost(vt::Queue& q, int32_t* sampled_out,
                                              int32_t* num_sampled_out) const {
  if (num_reqs_ == 0 || backend_ == nullptr) return;
  // Recorded so the destructor can drain this queue too when the caller frees
  // the object before waiting (see Release).
  copy_q_ = &q;
  // Two copies, NO drain (SPEC-DFLASH2 W8 #1837 kept the single-drain property;
  // A2-2 moves the drain itself out to the caller). Both buffers come off the
  // same kernel on the same queue, so ONE wait — wherever the caller puts it —
  // orders both reads.
  backend_->Copy(q, sampled_out, sampled_,
                 static_cast<size_t>(num_reqs_ * width_) * sizeof(int32_t));
  backend_->Copy(q, num_sampled_out, num_sampled_,
                 static_cast<size_t>(num_reqs_) * sizeof(int32_t));
}

// ─── RejectionSampler ───────────────────────────────────────────────────────

RejectionSamplerDeviceOutput RejectionSampler::verify(
    vt::Queue& q, const vt::Tensor& logits,
    const std::vector<int32_t>& draft_sampled,
    const std::vector<int32_t>& cu_num_logits) const {
  VT_CHECK(logits.rank == 2, "rejection_sampler: logits must be [num_logits, vocab]");
  VT_CHECK(logits.dtype == vt::DType::kF32, "rejection_sampler: logits must be f32");
  VT_CHECK(cu_num_logits.size() >= 1,
           "rejection_sampler: cu_num_logits must have num_reqs+1 entries");
  const int64_t num_reqs = static_cast<int64_t>(cu_num_logits.size()) - 1;
  const int64_t num_logits = logits.shape[0];
  VT_CHECK(num_reqs == 0 || cu_num_logits.back() == static_cast<int32_t>(num_logits),
           "rejection_sampler: cu_num_logits.back() must equal the expanded logits rows");
  VT_CHECK(draft_sampled.size() == static_cast<size_t>(num_logits),
           "rejection_sampler: draft_sampled must have one entry per expanded logits row");

  RejectionSamplerDeviceOutput dev_out;
  if (num_reqs == 0) return dev_out;

  // `sampled` row width: upstream sizes it num_speculative_steps + 1
  // (rejection_sampler_utils.py:1026-1028). Widen if a request somehow carries
  // more expanded rows than the configured k (defensive; the scheduler clamps).
  int64_t width = static_cast<int64_t>(num_speculative_steps_) + 1;
  for (int64_t r = 0; r < num_reqs; ++r) {
    const int64_t n = cu_num_logits[static_cast<size_t>(r) + 1] -
                      cu_num_logits[static_cast<size_t>(r)];
    VT_CHECK(n >= 1, "rejection_sampler: every request needs at least the bonus logit row");
    if (n > width) width = n;
  }

  const vt::Device dev = logits.device;
  vt::Backend& backend = vt::GetBackend(dev.type);
  VT_CHECK(num_logits >= 0 && uint64_t(num_logits) <=
               std::numeric_limits<size_t>::max() / sizeof(int32_t),
           "rejection_sampler: argmax allocation bytes overflow");
  size_t argmax_bytes = static_cast<size_t>(num_logits) * sizeof(int32_t);
#ifdef VLLM_CPP_XPU
  const int tile = dev.type == vt::DeviceType::kXPU ?
      vt::XpuRejectionArgmaxTile(num_logits, logits.shape[1]) : 0;
  size_t scratch_offset = 0, scratch_bytes = 0;
  if (tile) {
    // Same I32-ID-prefix/aligned-partial layout as mapped selection. This is
    // still this verification's private allocation, never a growing global.
    const auto layout = vt::XpuMappedSelectionBuffers(num_logits, logits.shape[1], tile);
    argmax_bytes = layout.total_bytes;
    scratch_offset = layout.scratch_offset;
    scratch_bytes = layout.scratch_bytes;
  }
#endif
  dev_out.backend_ = &backend;
  dev_out.device_ = dev;
  dev_out.num_reqs_ = num_reqs;
  dev_out.width_ = width;
  // Staging copies can already be queued if a later allocation throws. Retain
  // the existing completion-before-free guarantee on that unwind path too.
  dev_out.verify_q_ = &q;
  dev_out.draft_ = std::make_unique<DeviceScratch>(dev, q, draft_sampled.data(),
                                                   vt::DType::kI32,
                                                   std::initializer_list<int64_t>{num_logits});
  dev_out.cu_ = std::make_unique<DeviceScratch>(dev, q, cu_num_logits.data(),
                                                vt::DType::kI32,
                                                std::initializer_list<int64_t>{num_reqs + 1});
  dev_out.sampled_ =
      backend.Alloc(static_cast<size_t>(num_reqs * width) * sizeof(int32_t));
  dev_out.num_sampled_ =
      backend.Alloc(static_cast<size_t>(num_reqs) * sizeof(int32_t));
  // The accept walk's own scratch, allocated PER CALL and owned by this object
  // for the same reason its outputs are (SPEC-DFLASH2 A2-2, #2802): the argmax
  // kernel writes it and the accept kernel reads it, both after this function
  // returns. A shared grow-only buffer in the backend was the previous shape and
  // it freed this window's memory whenever a later, larger call grew it. The
  // per-step allocation is not a new cost class either — the two lines above
  // already allocate per call.
  dev_out.target_argmax_ =
      backend.Alloc(argmax_bytes);

  vt::Tensor sampled_t = vt::Tensor::Contiguous(dev_out.sampled_, vt::DType::kI32,
                                                dev, {num_reqs, width});
  vt::Tensor num_sampled_t =
      vt::Tensor::Contiguous(dev_out.num_sampled_, vt::DType::kI32, dev, {num_reqs});
  vt::Tensor target_argmax_t =
      vt::Tensor::Contiguous(dev_out.target_argmax_, vt::DType::kI32, dev, {num_logits});
#ifdef VLLM_CPP_XPU
  if (tile) {
    auto scratch = vt::Tensor::Contiguous(
        static_cast<unsigned char*>(dev_out.target_argmax_) + scratch_offset,
        vt::DType::kI8, dev, {static_cast<int64_t>(scratch_bytes)});
    vt::xpu::TiledGreedyRejectionSample(q, sampled_t, num_sampled_t, target_argmax_t,
        logits, dev_out.draft_->tensor(), dev_out.cu_->tensor(), scratch, tile);
  } else
#endif
  vt::GreedyRejectionSample(q, sampled_t, num_sampled_t, target_argmax_t, logits,
                            dev_out.draft_->tensor(), dev_out.cu_->tensor());
  // NOTHING is copied and NO queue is waited on here. That is the whole split,
  // and it is now true of the backend too: the op no longer reaches a global it
  // has to free, so no allocator call inside it can synchronize the device.
  return dev_out;
}

RejectionSamplerOutput RejectionSampler::finalize(
    const std::vector<int32_t>& host_sampled, int64_t width,
    const std::vector<int32_t>& host_num_sampled,
    const std::vector<int32_t>& cu_num_logits,
    const std::vector<char>& is_chunked_prefilling) {
  VT_CHECK(cu_num_logits.size() >= 1,
           "rejection_sampler: cu_num_logits must have num_reqs+1 entries");
  const int64_t num_reqs = static_cast<int64_t>(cu_num_logits.size()) - 1;

  RejectionSamplerOutput out;
  out.sampled_token_ids.resize(static_cast<size_t>(num_reqs));
  out.num_sampled.assign(static_cast<size_t>(num_reqs), 0);
  out.num_rejected.assign(static_cast<size_t>(num_reqs), 0);
  if (num_reqs == 0) return out;
  VT_CHECK(host_num_sampled.size() == static_cast<size_t>(num_reqs),
           "rejection_sampler: finalize needs one num_sampled entry per request");
  VT_CHECK(host_sampled.size() == static_cast<size_t>(num_reqs * width),
           "rejection_sampler: finalize needs a [num_reqs, width] sampled buffer");

  // get_num_sampled_and_rejected (gpu/input_batch.py:408-453): num_rejected =
  // num_logits - num_sampled; a still-chunked-prefilling row samples nothing and
  // rejects nothing.
  for (int64_t r = 0; r < num_reqs; ++r) {
    const size_t ur = static_cast<size_t>(r);
    const int32_t row_logits = cu_num_logits[ur + 1] - cu_num_logits[ur];
    int32_t ns = host_num_sampled[ur];
    const bool prefilling = ur < is_chunked_prefilling.size() && is_chunked_prefilling[ur] != 0;
    if (prefilling) {
      out.num_sampled[ur] = 0;
      out.num_rejected[ur] = 0;
      continue;
    }
    out.num_sampled[ur] = ns;
    out.num_rejected[ur] = row_logits - ns;
    out.sampled_token_ids[ur].reserve(static_cast<size_t>(ns));
    for (int32_t j = 0; j < ns; ++j) {
      out.sampled_token_ids[ur].push_back(host_sampled[ur * static_cast<size_t>(width) +
                                                       static_cast<size_t>(j)]);
    }
  }
  return out;
}

RejectionSamplerOutput RejectionSampler::forward(
    vt::Queue& q, const vt::Tensor& logits, const std::vector<int32_t>& draft_sampled,
    const std::vector<int32_t>& cu_num_logits,
    const std::vector<char>& is_chunked_prefilling) const {
  // The SAME-QUEUE route: issue the walk, copy on `q`, drain `q`. Byte-for-byte
  // what this function did before the A2-2 split, and it stays the route every
  // synchronous caller takes. The async verify arm calls `verify` / `CopyToHost`
  // / `finalize` directly so it can put the wait on a copy queue instead.
  RejectionSamplerDeviceOutput dev_out =
      verify(q, logits, draft_sampled, cu_num_logits);
  const int64_t num_reqs = dev_out.num_reqs();
  if (num_reqs == 0) {
    return finalize({}, dev_out.width(), {}, cu_num_logits, is_chunked_prefilling);
  }
  const int64_t width = dev_out.width();
  std::vector<int32_t> host_sampled(static_cast<size_t>(num_reqs * width));
  std::vector<int32_t> host_num_sampled(static_cast<size_t>(num_reqs));
  dev_out.CopyToHost(q, host_sampled.data(), host_num_sampled.data());
  vt::GetBackend(dev_out.device().type).Synchronize(q);
  return finalize(host_sampled, width, host_num_sampled, cu_num_logits,
                  is_chunked_prefilling);
}

}  // namespace vllm::v1
