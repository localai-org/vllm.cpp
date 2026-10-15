// Shared pooled device-scratch glue for dense-transformer forwards.
//
// Extracted VERBATIM (behavior-preserving) from include/vllm/model_executor/
// models/dense_attn_block.h so the NVFP4 W4A16 GEMM helpers
// (dense_nvfp4_gemm.h) can sit BENEATH the attention block that consumes them
// rather than above it — dense_attn_block.h now includes both this header and
// dense_nvfp4_gemm.h, which would otherwise be an include cycle. The definitions
// below are byte-for-byte the dense_attn_block.h originals and stay in the SAME
// namespace (`vllm::dense_attn`), so every existing `using namespace
// dense_attn;` consumer (qwen3.cpp, qwen3_moe.cpp, opt.cpp) resolves exactly as
// before and every dense forward remains BYTE-IDENTICAL.
//
// Contents (all `vllm::dense_attn`):
//   Dev                     — {Backend&, Queue&} device handle pair.
//   MakeTensor / Reshape    — non-owning strided views.
//   DevicePoolPolicy        — platform residency policy for the scratch pool.
//   DBuf                    — move-only pooled device allocation + tensor view.
#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "vllm/model_executor/models/device_pool.h"       // DevicePool/Pool/ActivePool
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/device.h"  // kNumDeviceTypes
#include "vt/dtype.h"   // VT_CHECK
#include "vt/ops.h"
#include "vt/tensor.h"  // vt::kMaxRank

namespace vllm {
namespace dense_attn {

using vt::Backend;
using vt::DType;
using vt::Queue;
using vt::Tensor;

struct Dev {
  Backend& b;
  Queue& q;
  // A loaded model can override the default backend activation precision.
  std::optional<DType> activation_dtype;
  Dev(Backend& backend, Queue& queue, std::optional<DType> activation = std::nullopt)
      : b(backend), q(queue), activation_dtype(activation) {}
};

// The one rank bound, in one place, because it has TWO callers that must not
// disagree: `MakeTensor` below, and `DBuf`'s constructor, which has to refuse
// BEFORE it draws a pool block (see the comment there).
inline void CheckRank(size_t rank) {
  VT_CHECK(rank <= static_cast<size_t>(vt::kMaxRank),
           "dense_attn: rank exceeds vt::Tensor's kMaxRank (4); vt::Tensor "
           "stores shape and stride in fixed int64_t[4] arrays and a wider rank "
           "writes past both");
}

// THE RANK BOUND (#2435) is `CheckRank` above, and it is the same one
// `vt::Tensor::Contiguous` (src/vt/tensor.cpp:19-20) has always carried.
// `vt::Tensor` fixes `kMaxRank = 4` and stores `int64_t shape[4]` and
// `int64_t stride[4]` (include/vt/tensor.h:12); the loop below indexes both by
// `i` up to `shape.size() - 1`, so a rank-5 shape wrote eight bytes past the
// end of each.
//
// IT IS NOT A HARMLESS OVERRUN AND IT IS NOT A CRASH, which is why no value
// gate could see it. `shape[4]` lands on `stride[0]`, which the `i == 0`
// iteration then rewrites correctly, so the shape damage heals itself.
// `stride[4]` lands on the three STORAGE MARKERS, and the first iteration
// writes `acc == 1` there — setting `Tensor::repacked` on a tensor nothing ever
// repacked. That marker is what `kMatmulBTQuant` reads to choose the i8mm
// interleaved gemm over the plain one, so the damage is a wrong kernel choice
// waiting for a weight, not a fault. `test_qwen4_exp_layer_loop` reported the
// oracle match and aborted under `-fno-sanitize-recover=all` before a single
// assertion ran; that abort is what has reddened `sanitize-cpu
// (address,undefined)` on every open pull request.
//
// This throws rather than truncating. A silently truncated rank is the same
// class of defect one level quieter: the buffer would still be sized from the
// full product, and every consumer would read a tensor whose shape does not
// describe its bytes.
inline Tensor MakeTensor(void* data, DType dt, vt::Device dev,
                         const std::vector<int64_t>& shape) {
  CheckRank(shape.size());
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t acc = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = acc;
    acc *= t.shape[i];
  }
  return t;
}

inline Tensor Reshape(const Tensor& src, const std::vector<int64_t>& shape) {
  return MakeTensor(src.data, src.dtype, src.device, shape);
}

// The device-scratch residency policy (BACKEND-PLATFORM item 2), resolved from
// the running device's platform. The DevicePool soft cap is platform data (0 ==
// uncapped, GB10 today ⇒ pool behavior byte-for-byte unchanged). Mirrors
// qwen3_5.cpp.
//
// Memoized PER DEVICE TYPE, not once per process. The previous single
// function-local static cached whichever device asked FIRST and then applied its
// cap to every later device — the same ambient-device assumption #516 fixed one
// layer down, and a mixed-backend process would have run a CUDA DBuf under the
// CPU platform's policy. DBuf is a per-op hot path, so the virtual dispatch is
// still paid at most once per device type.
//
// A backend whose platform was never REGISTERED now throws out of
// `platforms::GetPlatform` instead of inheriting whichever device asked first.
// That is the point: a residency cap read off another platform is a wrong
// number wearing a default's clothes. Gated by
// tests/vllm/models/test_device_pool.cpp.
struct DevicePoolPolicy {
  size_t cap_bytes = 0;  // residency_policy().device_pool_cap_bytes (0 == uncapped)
};
inline DevicePoolPolicy ResolveDevicePoolPolicy(const Dev& d) {
  // Stored as cap+1 so that 0 means "not resolved yet" and a genuine cap of 0
  // (every platform today) still caches. Racing threads resolve the same device
  // type to the same value, so the benign double-resolve needs no lock.
  static std::array<std::atomic<size_t>, vt::kNumDeviceTypes> cached{};
  // Same bound, same place, as platforms::Index() (src/vllm/platforms/
  // platform.cpp) applies to this identical value before indexing ITS registry.
  // An out-of-range DeviceType is only reachable by a cast, and the two lookups
  // must not disagree about whether that is a throw or a stray write.
  const size_t idx = static_cast<size_t>(d.q.device.type);
  VT_CHECK(idx < vt::kNumDeviceTypes, "invalid device type");
  const size_t seen = cached[idx].load(std::memory_order_relaxed);
  if (seen != 0) return DevicePoolPolicy{seen - 1};
  const auto rp = vllm::platforms::GetPlatform(d.q.device.type).residency_policy();
  cached[idx].store(rp.device_pool_cap_bytes + 1, std::memory_order_relaxed);
  return DevicePoolPolicy{rp.device_pool_cap_bytes};
}

// Owned device allocation + tensor view, routed through the SHARED DevicePool so
// the buffer's storage is reused rather than freed to the driver (avoiding the
// per-op cudaMalloc/cudaFree sync). Move-only, RAII. Ported verbatim from the
// qwen3_5.cpp pooled DBuf (device_pool.h Pool()/ActivePool()).
class DBuf {
 public:
  // THE EMPTY BUFFER, named rather than only reachable (#1904). An owner that is
  // default-constructed and allocates later -- the LTX-2.5 video VAE's
  // `VaeStore` is the first -- otherwise has to reach for `std::optional<DBuf>`
  // to hold one of these.
  //
  // It does not widen what the pool promises: an empty `DBuf` owns no block, and
  // `~DBuf`, `operator=(DBuf&&)` and `ReleaseShared()` each guard on
  // `p_ != nullptr`, so there is no path by which a pooled allocation leaks or
  // is returned twice through it.
  //
  // IT IS NOT THE MOVED-FROM STATE, and an earlier draft of this comment said it
  // was. A fresh review read the move constructor below and refuted it: that one
  // clears `p_` ALONE, so a moved-from buffer keeps a live `b_`, a live `pool_`,
  // its original `bytes()`, and a `t()` still describing the block it gave away.
  // This state clears everything, `b_` included. The difference that matters is
  // `Zero()` and `Download()`, which dereference `b_`: those two now check it
  // rather than fault, because a default-constructed object is a legal one and
  // two of its own public methods were undefined on it.
  //
  // WHY IT IS NOT `std::optional<DBuf>` AT THE CALL SITE. It was, for one
  // compile: g++ 13.3.0 at -O3 reports `-Wmaybe-uninitialized` through
  // `std::optional`'s disengaged union storage for every member the move
  // constructor at the line below reads, and this tree builds with `-Werror`.
  // The warning is a false positive on libstdc++'s guarded move, but the
  // alternative to suppressing it is this constructor, which is the better
  // object anyway.
  DBuf() = default;

  DBuf(Dev d, DType dt, const std::vector<int64_t>& shape,
       const void* host = nullptr)
      : b_(&d.b) {
    // THE RANK BOUND FIRST, ahead of the pool block (#2435). `MakeTensor` at
    // the bottom of this body is the writer that refuses, and by then
    // `pool_->Get` has already handed out an allocation. A constructor that
    // throws never runs its own destructor, so the block would be stranded —
    // the refusal would trade an out-of-bounds write for a leak. Checking here
    // costs one comparison on a path that is about to allocate anyway.
    CheckRank(shape.size());
    int64_t numel = 1;
    for (int64_t s : shape) numel *= s;
    bytes_ = static_cast<size_t>(numel) * vt::SizeOf(dt);
    alloc_bytes_ = bytes_ == 0 ? 1 : bytes_;
    cap_ = ResolveDevicePoolPolicy(d).cap_bytes;
    // THIS DEVICE's pool, unless an ActivePoolScope overrides it (the aux
    // stream). Remembered so the block returns to the pool it came from even if
    // this DBuf outlives the scope. See device_pool.h: there is no
    // device-less pool to fall back on.
    pool_ = &ActivePool(*b_);
    p_ = pool_->Get(*b_, alloc_bytes_);
    t_ = MakeTensor(p_, dt, d.q.device, shape);
    if (host != nullptr && bytes_ > 0) b_->Copy(d.q, p_, host, bytes_);
  }
  ~DBuf() { if (p_ != nullptr) pool_->Put(*b_, alloc_bytes_, p_, cap_); }
  DBuf(const DBuf&) = delete;
  DBuf& operator=(const DBuf&) = delete;
  DBuf(DBuf&& o) noexcept
      : b_(o.b_), pool_(o.pool_), p_(o.p_), bytes_(o.bytes_),
        alloc_bytes_(o.alloc_bytes_), cap_(o.cap_), t_(o.t_) {
    o.p_ = nullptr;
  }
  DBuf& operator=(DBuf&& o) noexcept {
    if (this != &o) {
      if (p_ != nullptr) pool_->Put(*b_, alloc_bytes_, p_, cap_);
      b_ = o.b_;
      pool_ = o.pool_;
      p_ = o.p_;
      bytes_ = o.bytes_;
      alloc_bytes_ = o.alloc_bytes_;
      cap_ = o.cap_;
      t_ = o.t_;
      o.p_ = nullptr;
    }
    return *this;
  }

  Tensor& t() { return t_; }
  const Tensor& t() const { return t_; }
  void* ptr() { return p_; }
  // The const read of the same pointer. Purely additive, and it is what lets a
  // const-qualified accessor on an owner that HOLDS a DBuf hand its bytes out
  // without a const_cast — the LTX-2.5 video VAE's `VaeStore::ptr() const`
  // (#1904) is the first such owner. `t() const` and `bytes() const` already
  // read this object through a const reference; the raw pointer had no such
  // spelling only because no caller had needed one.
  const void* ptr() const { return p_; }
  size_t bytes() const { return bytes_; }
  size_t alloc_bytes() const { return alloc_bytes_; }
  // The two methods that dereference `b_`. Both check it, because `DBuf()` is a
  // constructible object whose `b_` is null (#1904 fresh review); without this
  // an empty buffer reaching either one is a null dereference rather than a
  // message. A MOVED-FROM buffer is a different shape and reaches neither check:
  // its `b_` is still live and only its `p_` is null.
  void Zero(Dev d) {
    VT_CHECK(b_ != nullptr, "DBuf::Zero on an EMPTY buffer: it owns no allocation and names no "
                            "backend. Assign an allocated DBuf before writing to it.");
    b_->Memset(d.q, p_, 0, bytes_);
  }
  void Download(Dev d, void* host) {
    VT_CHECK(b_ != nullptr, "DBuf::Download from an EMPTY buffer: it owns no allocation and names "
                            "no backend. Assign an allocated DBuf before reading from it.");
    b_->Copy(d.q, host, p_, bytes_);
    b_->Synchronize(d.q);
  }
  // Relinquish the pool block WITHOUT returning it (dtor becomes a no-op); the
  // caller takes over the Put obligation for alloc_bytes(). Prefer
  // ReleaseShared() below, which discharges that obligation correctly by
  // construction.
  void* Release() {
    void* p = p_;
    p_ = nullptr;
    return p;
  }

  // Move the block into a shared_ptr that returns it to THIS buffer's own pool
  // and backend when the last owner drops it — the carrier every cross-step
  // hand-off (device logits, MTP hidden states, MoE scratch) wants.
  //
  // It replaces ~28 copies of a hand-written deleter that closed over the byte
  // count ALONE and called `Pool().Put(alloc, q)`. That idiom named neither the
  // device nor the pool, so it returned every such block to the one global pool
  // — a block from another device (#516), which was LIVE, and a block drawn
  // from the aux-stream pool, which was not: none of the nine `Release()` sites
  // sat inside or under any of the four `ActivePoolScope` regions, so the old
  // deleter and the buffer's own `pool_` always agreed in practice. It was a
  // hazard one new call site away from being real, and it is gone either way,
  // because the carrier now captures the pool it came from rather than
  // re-deriving it.
  std::shared_ptr<void> ReleaseShared() {
    DevicePool* const pool = pool_;
    Backend* const b = b_;
    const size_t alloc = alloc_bytes_;
    void* const p = Release();
    // A moved-from or already-released buffer owns nothing; a shared_ptr built
    // over a null pointer with a custom deleter would still RUN that deleter and
    // push null into the free list.
    if (p == nullptr) return {};
    // REGISTER THE LEASE with the block's own pool, and let the deleter return
    // it through `PutLeased`, which discharges the lease before the ordinary
    // Put. The pool then knows the block is owned by this carrier, and any
    // OTHER return of the live block — the second Put that made a cached block
    // re-handed out while its entry still owned it (the kolibri-1 dequant-cache
    // `POOL DOUBLE-HAND-OUT`, ISSUE-LOCAL-01M4BEH8ZH59TF9E0A7YRNTJJ2) — is
    // refused at the Put instead of silently aliasing the carrier's bytes.
    pool->NoteLease(p);
    return std::shared_ptr<void>(p, [pool, b, alloc](void* q) { pool->PutLeased(*b, alloc, q); });
  }

 private:
  Backend* b_ = nullptr;
  DevicePool* pool_ = nullptr;
  void* p_ = nullptr;
  size_t bytes_ = 0;
  size_t alloc_bytes_ = 0;
  size_t cap_ = 0;
  Tensor t_{};
};

// ── The ONE install of an f32-upcast weight (#2711) ─────────────────────────
//
// `ResidentWeightF32` exists TWICE -- here in `dense_attn_block.h`, which 49
// translation units under `src/vllm/model_executor/models/` include, and as a
// private twin in `qwen3_5.cpp` that stays private on purpose (`:790`). Both
// carried the same defect, so a repair applied once left it standing. This is
// the body both now install through, and it is the reason there is one repair
// rather than two.
//
// WHY `f` IS TAKEN BY VALUE AND NAMED. It is the copy SOURCE, and its lifetime
// is the entire bug. `Backend::Copy` is ASYNCHRONOUS on both device backends --
// `cudaMemcpyAsync` (`src/vt/cuda/cuda_backend.cu:116-118`) and `hipMemcpyAsync`
// (`src/vt/rocm/rocm_backend.hip:269-271`) -- and this source is ordinary
// PAGEABLE heap memory, which the driver may read at any point until the queue
// drains. Both callers used to build it in a function-local vector and let it
// die at their closing brace, with nothing in between that drained anything.
//
// THE `Synchronize` IS THE FIX, and the tree already states its rule for the
// same hazard: `glm5_next_kv.cpp:143-150` drains on EVERY span rather than once
// at the end, because a deferred wait "would hand the driver a pageable source
// that the next iteration has already overwritten." Same hazard, now handled the
// same way.
//
// WHY NOT KEEP THE SOURCE ALIVE INSTEAD. `vt::Backend` has no completion
// callback and nothing polls `Event`, so "alive until the copy retires"
// degenerates to "alive for the model's lifetime" -- a permanent host allocation
// per upcast weight, reinstating exactly the second host copy
// `AdoptDeviceBytesAsHost` and `ReleaseResidentQwen3_5DenseHostWeights` exist to
// drop. The drain is memoised on `d_dev_f32`, so it runs ONCE per distinct
// weight per process rather than per forward or per token.
//
// THE SIBLING `ResidentWeight` DOES NOT DRAIN, AND IS RIGHT NOT TO. Its source
// is `w.bytes`, an owned buffer or a file mapping that outlives the call. The
// asymmetry between the two helpers is the whole defect, not an oversight in
// the other one.
inline void InstallResidentF32(Dev d, const OwnedTensor& w,
                               std::vector<float> f) {
  // Aliasing is a CPU property and not a not-CUDA one (#125, #1946): the
  // predicate this replaced was `!is_cuda()`, which handed a plain heap pointer
  // to kMETAL, kVULKAN and kXPU. All three reach the upload arm below.
  if (vllm::platforms::GetPlatform(d.q.device.type).is_cpu()) {
    auto* buf = new std::vector<float>(std::move(f));
    w.d_dev_f32 = std::shared_ptr<void>(buf->data(), [buf](void*) { delete buf; });
    return;
  }
  const size_t nb = f.size() * sizeof(float);
  void* p = d.b.Alloc(nb);
  d.b.Copy(d.q, p, f.data(), nb);
  // BEFORE `f` GOES OUT OF SCOPE, and before this function returns. See above.
  d.b.Synchronize(d.q);
  Backend* bk = &d.b;
  w.d_dev_f32 = std::shared_ptr<void>(p, [bk](void* q) { bk->Free(q); });
}

}  // namespace dense_attn
}  // namespace vllm
