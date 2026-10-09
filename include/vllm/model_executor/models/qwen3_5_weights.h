// vllm.cpp original (in-memory weight container for the Qwen3.6-35B-A3B MoE
// gate model); no 1:1 upstream mirror — the pinned load path is
// AutoWeightsLoader over nn.Module params (models/qwen3_5.py @ e24d1b24), which
// this replaces with an explicit owned-tensor struct for the M0.9 forward.
//
// Loads the real 35B checkpoint (nvidia/Qwen3.6-35B-A3B-NVFP4) into owned host
// bf16 tensors. Quant schemes per weight class (verified against the real ckpt,
// .agents/specs/qwen36-forward-notes.md §6):
//   - MoE experts + shared_expert + lm_head : NVFP4 W4A16 (DequantNvfp4ToBf16)
//   - attention (q/k/v/o) + GDN (in_proj_qkv/z, out_proj) : per-tensor FP8
//     (DequantFp8ToBf16) — NOT bf16 as the task first assumed
//   - everything else (embeds, norms, router gate, conv1d, A_log/dt_bias,
//     in_proj_a/b) : bf16 (A_log/dt_bias upcast to f32)
//
// All 2-D projection weights are stored TRANSPOSED to vt::Matmul's B layout
// [in, out] (on-disk torch layout is [out, in]). Host checkpoint ownership stays
// per logical projection. On CUDA the production 27B path additionally builds
// resident packed gate_up and full-attention QKV operands, mirroring vLLM's
// MergedColumnParallelLinear/QKVParallelLinear topology at TP=1; diagnostic
// toggles retain the split residents.
#pragma once

#include <cstddef>  // size_t, for kDeviceAliasAlignment
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/owned_bytes.h"
#include "vllm/model_executor/models/qwen3_vl_vision.h"  // MoE vision tower (#891)
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vt {
class Backend;
}  // namespace vt

namespace vllm {

// One owned, contiguous, host tensor: heap bytes + shape/dtype. View() builds a
// fresh vt::Tensor over the current buffer, so it stays valid across moves/
// reallocations of the owning struct (no cached raw pointer to dangle).
struct OwnedTensor {
  // Heap bytes, OR a read-only borrowed view (GGUF mmap / a bf16 expansion
  // shared with a second tensor) that carries its own keep-alive. See
  // owned_bytes.h; the read API is the std::vector<uint8_t> subset this tree
  // uses, so readers are unaffected by which residency a weight has.
  OwnedBytes bytes;
  vt::DType dtype = vt::DType::kF32;
  int rank = 0;
  int64_t shape[vt::kMaxRank] = {0, 0, 0, 0};

  // Matmul-weight orientation. false (default): Matmul-B [K=in, N=out] (the
  // loader transposed the disk tensor; row-major x row-major vt::Matmul).
  // true: raw torch Linear [N=out, K=in] as on disk (LoadBf16Raw) — consumed
  // via vt::MatmulBT, the cuBLASLt TN fast path (see ops.h MatmulBT).
  bool nk = false;

  // CIQ G7: the block-quant bytes were REPACKED at load into the CPU i8mm
  // interleave (q8_0 -> block_q8_0x4). Propagated to vt::Tensor::repacked by
  // View(); the CPU quant GEMM keys on it. Only set on an i8mm process, only for
  // repack-eligible q8_0 weights, and only on the copy residency (the transform
  // mutates the buffer, so it cannot ride the read-only mmap borrow).
  bool repacked = false;

  // Brick 4 (DeepSeek-V4 last-mile): the Q8_0 blocks were repacked at load into
  // the CUDA coalesced-load layout ([all qs | all scales], 16-byte-aligned qs).
  // Carried to vt::Tensor::q8_0_aligned by View(); the CUDA Q8_0 GEMM keys on it.
  // Same total bytes; set only on the owned (copy) residency (the transform
  // rewrites the buffer, so it cannot ride the read-only mmap borrow).
  bool q8_0_aligned = false;
  // KERNEL-GEMM-CPU-TILED lever 2: the [N,K] elementwise bytes were transposed
  // to [K,N] at load. Carried to vt::Tensor::elem_kn_repacked; shape stays
  // [N,K]. Opt-in (VT_CPU_ELEM_KN_REPACK=1) and CPU-only, see the flag comment.
  bool elem_kn_repacked = false;

  // A synchronized direct-device load may discard the host staging buffer while
  // retaining an authoritative d_dev/d_dev_f32 copy. Empty() is used as model
  // dispatch metadata, so host reclamation must not make a populated weight look
  // absent. Mutable because ReleaseHost is logically const, like lazy residency.
  mutable bool host_released = false;

  // ENG-EXPERT-STREAM-DEVICE W0c (issue #1124). The expert-stream lane has
  // claimed this tower: its slices are served from HOST slot storage, and the
  // tower itself must therefore NEVER be staged into device memory.
  //
  // Why a flag and not a name test. `ResidentWeight` is handed an OwnedTensor
  // and has no idea what the loader called it, and the refusal has to fire in
  // `ResidentWeight` because that is where the 1.1875 GiB `d.b.Alloc` is. A
  // process-global set of streamed tower ids would answer the same question at
  // the cost of a lookup in the decode path of every weight; one bool on the
  // tensor answers it for free.
  //
  // Set by `KqExpertSlice` the first time the lane serves this tower, and read
  // by `ResidentWeight`'s staging branch, which throws by name. That refusal is
  // a TRIPWIRE, not a path with a production caller: with the lane on, the
  // grouped-MoE route that would stage a tower is already disabled
  // (`Qwen35GroupedMoeEnabled`), and W0c's own fallback reads the tower's host
  // bytes in place. It exists because the failure it guards is issue #1123 —
  // 48 towers staged, death partway through layer 16 of 93 — and that failure
  // is silent until the allocator runs out.
  //
  // Mutable for the same reason `d_dev` is: claiming a tower for the lane is
  // logically const, like lazy residency.
  mutable bool expert_streamed = false;

  // ENG-LOAD-DIRECT-UPLOAD (issue #150). Non-null when `bytes` BORROWS a
  // read-only safetensors mmap taken verbatim from the checkpoint instead of
  // being copied into an owned buffer, and records the exact source range so
  // the windowed page release can run once a device copy exists. It is the
  // discriminator between the two borrow producers: this one may be adopted
  // onto the device allocation after upload, whereas a GGUF-mmap or
  // tied-expansion borrow must not be (see AdoptDeviceBytesAsHost).
  mutable const void* mmap_src = nullptr;
  mutable size_t mmap_src_bytes = 0;
  // Where those borrowed bytes physically live, for a consumer that wants to
  // READ them rather than fault them in: the owning shard's descriptor and the
  // byte offset of this weight within that shard. Set only on the mmap-borrow
  // path; -1 means "no descriptor, read through the mapping".
  //
  // Expert streaming is the consumer. Filling a slot by memcpy from the mapping
  // traps every 4 KiB page on the way, which is the whole reason W4 moved no
  // faster than the mmap path it replaced; a pread lands the slice in one call.
  mutable int mmap_fd = -1;
  mutable size_t mmap_file_offset = 0;

  // A process-unique identity for the BUFFER this tensor currently points at,
  // for a cache that outlives the model.
  //
  // READ THAT LITERALLY: the identity is keyed on `bytes.data()`, so it is an
  // identity for the address, not for the contents. Replacing a buffer's bytes
  // IN PLACE — same address, different weights — keeps the old uid, and the
  // cache would then serve the old entries for the new contents. Nothing does
  // that today: a tower's `bytes` is assigned once when the model loads and is
  // only ever replaced wholesale, which moves the address. This comment says
  // where the guarantee stops rather than rounding it up, because #1066 was
  // caused by a comment on this exact field that rounded it up (it claimed a
  // base pointer was a stable identity, which is true for one model's life and
  // false for the cache's). `test_qwen36_weights` pins both halves.
  //
  // The expert slot cache is a process-lifetime singleton keyed by (tower,
  // expert), and it used to derive the tower half from the buffer's ADDRESS.
  // That is stable for one model's life, which is what its comment claimed, but
  // the cache is not scoped to one model's life: free a model, load another, and
  // the allocator hands a new tower an address the old one used. Every entry the
  // cache still holds for that address is then served to a DIFFERENT model's
  // expert, silently, as a hit that moves no bytes. Measured on two synthetic
  // models in one process: 21 distinct addresses for 24 towers, and 20 of 222
  // slices returned another tower's bytes.
  //
  // A counter cannot collide because it never goes backwards, and re-stamping
  // when `bytes` moves keeps a COPY of a tensor from inheriting the original's
  // identity along with a different buffer. Assigned lazily, so a weight that is
  // never streamed never pays for one.
  mutable uint64_t tower_uid = 0;
  mutable const uint8_t* tower_uid_for = nullptr;

  // `tower_uid`, assigned on first use and re-assigned if `bytes` has moved.
  // Never returns 0, so a caller can treat 0 as "no identity".
  uint64_t TowerUid() const;

  bool Empty() const { return bytes.empty() && !host_released; }
  bool HasHostBytes() const { return !bytes.empty(); }
  int64_t Numel() const;
  // Contiguous view over the current buffer (host/CPU device).
  vt::Tensor View() const;
  // Build every host/resident descriptor from the same storage and value metadata.
  vt::Tensor ViewOn(void* data, vt::Device device,
                    std::vector<int64_t> shape = {}) const;
  std::optional<vt::DType> weight_value_dtype;

  // Return the host byte buffer to the OS once an authoritative device-resident
  // copy exists — the platform residency behavior
  // `residency_policy().release_host_weights_after_upload`
  // (platforms/interface.h; BACKEND-PLATFORM item 2). Logically const: the
  // tensor's VALUE is unchanged, only the now-dead host mirror is freed. This
  // mirrors the existing mutable lazy-device-upload residency design (d_dev
  // above is populated on a const weight). swap-with-empty (not
  // clear()) guarantees the std::vector capacity is actually deallocated. After
  // release View()/bytes must not be read; shape/dtype metadata is retained and
  // Empty() remains false so device-resident dispatch continues to see it.
  void ReleaseHost() const;

  // True when the host view is an adopted alias of this tensor's own device
  // allocation (host-addressable devices: AdoptDeviceBytesAsHost re-points
  // `bytes` at it and the block becomes the view's keep-alive).
  bool HostViewIsDeviceTwin() const {
    return d_dev != nullptr && bytes.borrowed() && mmap_src == nullptr &&
           bytes.data() == static_cast<const uint8_t*>(d_dev.get());
  }

  // Lazily-populated device-resident copies (CUDA forward only; null on host or
  // before first use). Uploaded ONCE and reused across every forward step so the
  // model's bf16/f32 weights (embed table, norms, attention/GDN projections,
  // router) stop re-uploading per op. d_dev holds the raw-dtype bytes; d_dev_f32
  // holds a bf16->f32 upcast (the CUDA norm/conv kernels want f32 when the
  // activation is f32). The shared_ptr deleter frees through the vt Backend.
  mutable std::shared_ptr<void> d_dev;
  mutable std::shared_ptr<void> d_dev_f32;
};

// ADOPT the device-resident copy AS the host buffer, where the backend says its
// allocations are host-addressable (vt::Backend::DeviceMemoryIsHostAddressable).
//
// THE DEFECT THIS CLOSES (BACKEND-VULKAN-LOADMEM). `ResidentWeight` uploads a
// weight and keeps `bytes` as well, so on Vulkan the model became resident
// TWICE. That is invisible on a discrete GPU -- the two copies are in different
// memories -- but GB10 is unified, so both come out of the same 119 GiB of
// system RAM. MEASURED on Qwen3-4B (7.6 GiB on disk): 8.622 GiB of Vulkan
// allocation and 16.392 GiB of process VmHWM, i.e. a whole second copy, and
// 17.1 GiB off MemAvailable. Extrapolated to the 27B (50.89 GiB) that is over
// 100 GiB, which is why loading it could take the machine down rather than fail.
// `src/vllm/platforms/vulkan.cpp` even REASONED that there is "exactly one copy
// of the bytes"; this is what makes that true.
//
// It is an ADOPTION, not a free: `bytes` is re-pointed at the device allocation
// (persistently mapped, host-coherent) and keeps it alive through `d_dev`, so
// every existing `.bytes` reader -- `ResidentWeightF32`'s upcast, the portable
// CPU reference tier, `Numel`/`View` -- reads the SAME bytes it read before,
// from the surviving copy. Nothing is dropped that anyone could still want, so
// unlike `ReleaseHost()` this needs no "is the device path committed" proof.
//
// A no-op unless the backend opts in, and a no-op on an already-BORROWED buffer
// (a GGUF mmap or a tied-weight expansion): those own no anonymous pages, and a
// tied pair must keep sharing one keep-alive.
//
// `VT_ADOPT_DEVICE_BYTES=0` is the same-binary A/B back to the two-copy
// behavior (house convention for a default-on residency change).
void AdoptDeviceBytesAsHost(vt::Backend& backend, const OwnedTensor& w);

// Drop the resident host pages a just-STAGED weight's borrowed source span still
// holds, on a device whose kernels cannot read host memory.
//
// WHAT IT IS FOR. `ResidentWeight`'s staging arm copies the weight to the device
// and leaves every source page mapped for the process lifetime. For a GGUF
// keep-quant load that is the WHOLE MODEL: 65.488 GiB on the 67.56 GiB
// `Qwen3.8-Flash-Next UD-IQ1_S`, faulted in by the load-time prefault and read
// exactly once. `ReleaseHost()`'s borrowed branch declines to `madvise` those
// pages, arguing that clean file-backed pages are reclaimable without our help.
// That argument holds against the page reclaimer and FAILS against the KFD,
// which walks what this process has resident: on gfx1151 the load then wedges
// forever in `svm_range_set_attr` with a 31 GiB host
// (.agents/specs/rocm-host-residency-after-upload.md).
//
// This is llama.cpp's `unmap_fragment` (`src/llama-mmap.cpp:492-507`, called at
// `src/llama-model-loader.cpp:1683-1694` after the offloaded tensors are set)
// expressed as `MADV_DONTNEED` rather than `munmap`, which is the spelling this
// tree already uses for the same job in `GgufFile::DropSpanResidency`,
// `ReleaseHost` and `AdoptDeviceBytesAsHost`. The borrow itself is untouched and
// stays a valid, re-faultable `PROT_READ MAP_PRIVATE` view, so a later read
// re-faults the identical bytes from the file. The cost of being wrong is a page
// fault, never a wrong token.
//
// THREE PRECONDITIONS, ALL CHECKED HERE.
//
//  1. The device cannot dereference host storage. TWO predicates answer that,
//     they are not the same question, and asking only one is a defect this
//     file's own gate caught. `vllm::platforms::Platform::
//     host_memory_is_device_addressable()` is what `ResidentWeight` selects the
//     ALIAS arm on (issues #125, #1299): may a kernel follow a host pointer.
//     `vt::Backend::DeviceMemoryIsHostAddressable()` is the converse: is a
//     DEVICE allocation host-dereferenceable (GB10 answers false while being
//     physically unified). A weight can reach the staging arm on a platform
//     whose answer is YES -- a misaligned borrow declines the alias and falls
//     through -- and releasing its pages there would drop memory the kernels
//     may still read directly. So the CALLER passes its already-computed
//     platform answer in as `host_addressable` and the helper refuses on
//     either. One predicate, computed once, exactly as #2406 made
//     `QuantRepackForDevice` take its `dev`: a second spelling is how a
//     refusal and its route predicate come to disagree about one weight.
//
//     ONLY THE PLATFORM HALF IS GATED, and that is recorded rather than
//     implied. Deleting `backend.DeviceMemoryIsHostAddressable()` here leaves
//     the focused suite green: the fake backend in
//     `test_resident_weight_host_addressable.cpp` answers false
//     unconditionally, so no case ever presents the disagreeing pair. Owed in
//     `ISSUE-LOCAL-01M2CCNA0S74WT5WBV50B3VD0W`.
//  2. `bytes` is BORROWED. An owned buffer is `ReleaseHost`'s business.
//  3. `mmap_fd >= 0`. This is the discriminator that makes the call SAFE, and it
//     is not a convenience. `MADV_DONTNEED` on a file-backed private mapping
//     drops re-faultable pages; on ANONYMOUS memory it ZEROES them. The other
//     borrow producer in this tree is a tied pair's shared bf16 expansion, which
//     is anonymous and whose keep-alive both tensors share. Only the mmap-borrow
//     path sets `mmap_fd` (`SourceOfSpan`, qwen3_5_gguf_weights.cpp), so asking
//     for it is asking "are these pages backed by a file I can re-read".
//
// IT MUST BE CALLED FROM BEHIND THE `d_dev` MEMO, and #1299 is why: the caller
// reaches `ResidentWeight` about 1,361 times per forward step on that
// checkpoint, so a release that re-tested its condition on every call would
// `MADV_DONTNEED` the pages the GPU is about to read on every step and the
// kernel would fault them straight back in. Correctness survives that;
// throughput does not. The one production call site is inside
// `if (!w.d_dev)` and `BorrowReleaseSnapshot().calls` is what makes that
// checkable rather than asserted. THERE ARE TWO SUCH CALL SITES AND BOTH ARE
// PRODUCTION: the `ResidentWeight` in the unnamed namespace of `qwen3_5.cpp`
// (Qwen3.5's dense weights, and the shared MoE seam's keep-quant expert towers
// through `KqResidentSlice` and `KqGrouped`), and `dense_attn::ResidentWeight`
// in `dense_attn_block.h`, which is the one every `qwen4_exp` attention, norm,
// hyper-connection, PLE and lm_head weight takes. Wiring only the first left
// host `RssFile` on `strix:gpu0` at 21.08 GB through the forward.
//
// It SYNCHRONIZES `queue` before releasing anything, because the staging copy is
// `hipMemcpyAsync` on a stream and dropping the source pages under a live DMA is
// a correctness bug rather than a residency one. The synchronize is skipped
// entirely when the preconditions do not hold, so a backend this does not apply
// to pays nothing. THE SYNCHRONIZE IS ALSO UNGATED, for a harness reason:
// `HostBackend::Copy` is a synchronous `memcpy` and the class does not override
// `Synchronize`, so no case in this tree can express a live DMA. Same issue.
//
// `host_addressable` is the caller's `host_memory_is_device_addressable()`
// answer for the queue's device. Returns true when pages were released.
bool MaybeReleaseStagedBorrowSource(vt::Backend& backend, vt::Queue& queue,
                                    const OwnedTensor& w, bool host_addressable);

// What `MaybeReleaseStagedBorrowSource` has done in this process. `calls` counts
// the releases that HAPPENED, not the invocations that declined, for the same
// reason `NoteGgufPrefaultedSpan` counts after the fact: an `madvise` changes no
// byte, so a count of the ones that ran is the only thing that separates a
// release from a skip. Read it at a stated point; it is cumulative.
struct BorrowReleaseStats {
  uint64_t calls = 0;
  uint64_t bytes = 0;
};
BorrowReleaseStats BorrowReleaseSnapshot();

// The whole of `src` as a ZERO-COPY view: same bytes, shape, dtype, `nk` and
// layout markers, with a keep-alive on `src`'s buffer.
//
// WHY THIS EXISTS RATHER THAN `OwnedTensor b = a;`. `OwnedTensor`'s implicit
// copy DEEP-COPIES an owned `OwnedBytes`, and a per-step block adapter that
// spells the pass-through as assignment therefore reallocates the weight on
// every step. That is not only traffic. `ResidentWeight`'s host-alias arm hands
// a device kernel `w.bytes.data()` directly, so the copy's buffer becomes the
// GEMM's operand, and the copy dies at the end of the layer's scope while the
// GEMM is still queued: `Warp illegal address` inside cuBLASLt, with every
// declared extent correct (issue #2476, `.agents/specs/gdn-qkvz-operand-lifetime.md`).
// The view keeps the operand owned by the model, which is the invariant.
//
// `src` is NON-CONST because `OwnedBytes::KeepAlive()` converts an owned buffer
// into a shared read-only one in place. Nothing may write through `src` after
// that, which is already true of every weight this serves: they are read-only
// from the end of the load. `KeepAlive()` is taken BEFORE `data()`, so the view
// can never depend on `std::vector`'s move preserving the heap address.
OwnedTensor BorrowWholeOwnedTensor(OwnedTensor& src);

// The alignment a HOST pointer must meet before a device kernel may be handed it
// in place of the `Backend::Alloc` pointer it would otherwise have received.
//
// 256, because that is cuBLASLt's documented
// `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES` DEFAULT, which this tree never
// sets, and it dominates every explicit pointer gate in the tree. Measured, not
// assumed: `grep -rn MIN_ALIGNMENT src/vt/` finds nothing, so the 256 default
// applies to every cuBLASLt matmul this tree issues.
//
// THE POINTER GATES, ENUMERATED. Two earlier revisions of this paragraph said
// there was exactly one, in `cuda_matmul_nvfp4.cu`, asking for 16. That was
// wrong both times. `grep -rn 'reinterpret_cast<uintptr_t>' src/vt/cuda/` plus
// `grep -rn PointerAligned src/vt/` finds at least seven, across four files:
//
//   | site | asks | operand |
//   |---|---|---|
//   | `cuda_nvfp4_sm12x.cu:401` `PointerAligned(gate_up, 32)`   | **32** | activation |
//   | `cuda_nvfp4_sm12x.cu:401` `PointerAligned(packed, 8)`     | 8  | weight |
//   | `cuda_matmul_nvfp4.cu:204` `(prow) & 0xf`                 | 16 | weight row |
//   | `cuda_matmul_nvfp4.cu:1757,1778` `(p) & 0xF`              | 16 | scratch |
//   | `cuda_matmul_nvfp4.cu:1841` `(out) & 0x7`                 | 8  | output |
//   | `cuda_laguna.cu:67,70` `LagFastNormAligned{16,8}`         | 16 | weight/act |
//   | `cuda_ops.cu:370,395` `aligned16(w.data)`                 | 16 | NORM WEIGHT |
//
// The strictest is 32, and `cuda_ops.cu` is the one that binds a WEIGHT pointer
// on the ordinary decode path rather than a packed-arm buffer. The `% 32` and
// `% 64` tests near `cuda_matmul_nvfp4.cu` still read as alignment gates and
// still are not: they check a DIMENSION (`d`, `dv`), not an address.
//
// The conclusion is unchanged and is now correct at the widest of them: 256
// dominates 32 as comfortably as it dominated 16. It is also what `cudaMalloc`
// returns in practice, though CUDA guarantees only "suitably aligned" and
// current devices return more — so "indistinguishable from a `cudaMalloc`
// pointer" is the intuition, and "at least what every consumer is promised" is
// the claim.
//
// WHAT ALIGNMENT DOES AND DOES NOT BUY. It makes the substitution CORRECT: no
// kernel can fault or mis-vectorise on this pointer that would not have on the
// other. It does not make the two pointers indistinguishable in every respect,
// and two in-tree facts say so. `src/vllm/model_executor/models/laguna.cpp`
// records a MEASURED GB10 penalty for reading system-allocated memory from the
// GPU rather than a `cudaMalloc` allocation, worst on a long-K low-parallelism
// GEMV — a consumer telling them apart by BANDWIDTH, which is why
// `VT_QWEN35_ALIAS_HOST_WEIGHTS` exists below. And the Vulkan and Metal backends
// resolve a tensor pointer against their own allocation tables and throw if it
// is outside them, telling them apart by IDENTITY; harmless only because neither
// overrides `host_memory_is_device_addressable()`, so this argument is scoped to
// backends that take raw pointers. Deriving a smaller number would mean
// enumerating every kernel that ever binds a weight and being right about all of
// them, and the enumeration does not close: the widest thing any of them
// dereferences is a 16-byte `cp.async` granule
// (`src/vt/cuda/cuda_matmul_nvfp4.cu`, whose shape gate assumes an aligned base
// rather than checking it), while cuBLASLt is PROMISED 256 —
// `CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES` defaults to 256 and this tree
// never sets it (`src/vt/cuda/cuda_matmul.cu`). Matching the allocator instead
// of the consumers makes the whole question go away, and it costs one memcpy
// that REPLACES the host->device copy it removes.
//
// CAN THE SUBSTITUTION MOVE A LOGIT? MEASURED, AND THE ANSWER IS NO.
// An earlier revision of this comment worried that "the heuristic may pick an
// algorithm on the strength of a promise a 16-aligned pointer breaks", and a
// fresh review was right that the worry was recorded here and never measured.
// It is measured now, on `thor:gpu0` (NVIDIA Thor `sm_110`, driver 13020,
// cuBLASLt 130101), which answers this file's own predicate TRUE
// (`cudaDevAttrPageableMemoryAccess = 1`, `cudaDevAttrIntegrated = 1`) and is
// therefore a member of the population this branch serves. Six shapes off the
// target checkpoint (`embedding_length = 8192`; M = 1, 5 and 32) crossed with
// BOTH formulations this tree issues — the row-major NN `MatmulKernelCuda`,
// where the weight is operand B, and the column-major TN `MatmulBTKernelCuda`,
// where it is operand A — for 12 measurements, `PROBE_FAILURES=0`:
//
//   * The heuristic CANNOT see a pointer. `cublasLtMatmulAlgoGetHeuristic` takes
//     (handle, desc, four layouts, preference) and no operands, so alignment
//     reaches it only through the preference. 12/12 identical on a repeated
//     call, 12/12 identical with the 256 promise stated EXPLICITLY, and 12/12
//     identical when the promise is weakened to 16 — the selection does not move
//     on alignment at all. Five DIFFERENT algo configurations appear across the
//     six shapes (tiles 393/537/573/576, workspaces 0 to 5,242,896), so the
//     instrument does discriminate; it simply does not discriminate on this.
//   * The OUTPUT is bit-exact. Running `cublasLtMatmul` with the same algo over
//     the same bytes, the weight operand once from `cudaMalloc` and once from a
//     256-aligned host block, gives byte-identical results: 12/12 with zero
//     differing elements, every status `SUCCESS`.
//
// So a 256-aligned host pointer cannot change a logit.
//
// AND THE SAME PROBE RAN ON THE GB10 ITSELF — the silicon the token gate ran on,
// so this is no longer a Thor result read across to another part. `rc` job
// `7c7a05e9-be87-48f4-94ae-1bbe0340f063` on `dgx:gpu0`, 2026-08-19 17:47 UTC,
// `NVIDIA GB10 sm_121`, driver 580.173.02, cuBLASLt 130101, the predicate
// re-derived in the job's own output (`pageableMemoryAccess=1 integrated=1`).
// Same six shapes, same two formulations, 12 measurements, `PROBE_EXIT=0`,
// `PROBE_FAILURES=0`: repeated heuristic call identical 12/12, the promise
// weakened to 16 moving nothing 12/12, and `cublasLtMatmul` output bit-exact
// 12/12 (`differing=0`). At least five distinct algo configurations appear
// across the twelve and they DIFFER from Thor's, so the heuristic was
// re-resolved rather than replayed.
//
// WHAT THAT ESTABLISHES, AND WHAT IT DOES NOT. It EXCLUDES this branch as the
// cause of the row's CUDA-versus-CPU token divergence, measured on the target
// silicon. It does not IDENTIFY the cause, and no reader of this block may take
// it as if it did: excluding one cause is not identifying another. An earlier
// revision of this comment named the two arms' GEMM arithmetic as the cause,
// which the row's own spec forbids asserting — that is the STANDING HYPOTHESIS,
// together with a greedy path whose top-2 margin at the divergent step —
// step 9 — is 0.022802 logits, about 0.1 % of the winning logit, and it is
// NOT MEASURED. This comment used to name 0.264709 logits here. That is
// step 7's margin, and step 7 is a step both arms AGREE on: the divergence
// point was transcribed wrong in `.agents/benchmark-record.md`'s W0f entry,
// which drops token id `7172` twice (#1783). Naming the first tensor whose
// values differ between the arms at that step, and the operation that
// produced it, is carried under `## Owed` in
// `.agents/specs/expert-stream-device-slots.md`.
//
// One more thing the probe does not license. The 16-aligned arm also came back
// bit-exact at these shapes, which is NOT a reason to lower this constant: the
// enumeration above still does not close, and a promise kept by luck at twelve
// shapes is not a promise. The literal below is pinned by a case in
// `tests/vllm/model_executor/test_resident_weight_host_addressable.cpp`, so
// lowering it reds a gate instead of passing every one of them silently.
inline constexpr size_t kDeviceAliasAlignment = 256;

// Make `w.bytes` safe to hand to a device kernel directly, and say whether it
// worked. On return `true`, `w.bytes.data()` is non-null and aligned to
// `kDeviceAliasAlignment`. On `false` the caller must fall back to staging, and
// nothing has changed.
//
// THREE CASES, and the middle one is the point (ENG-EXPERT-STREAM-DEVICE W0f,
// issue #1299).
//
//   * ALREADY ALIGNED — true, and nothing is copied. A GGUF mmap borrow lands
//     here whenever its tensor offset happens to be a multiple of 256; GGUF's
//     `general.alignment` guarantees only 32, so this is luck rather than a
//     contract, and the fallback below is what makes that acceptable.
//   * OWNED AND MISALIGNED — the bytes are moved into a `kDeviceAliasAlignment`
//     allocation and `w.bytes` is re-pointed at it, keeping the new block alive
//     the way `AdoptDeviceBytesAsHost` keeps the device block alive. A plain
//     `std::vector<uint8_t>` from glibc is 16-byte aligned and no more (a large
//     block is an mmap chunk, so it lands at page+16), which is exactly what the
//     GDN V-head reorder's ~44.6 GiB of bf16-expanded `attn_qkv` / `ssm_out`
//     arrive as. Without this they could never be aliased and W0f would move no
//     bytes at all.
//   * BORROWED AND MISALIGNED — false. A borrow owns no anonymous pages: it is a
//     clean, file-backed GGUF mapping or a tied pair's single shared expansion.
//     Copying it would CREATE the anonymous residency this change exists to
//     remove, and would break the tie. Staging is the right answer for it.
//
// Logically const, like the lazy device residency beside it: only where the
// bytes live changes, never what they are.
// The outcomes, so a caller and a log can say WHICH one happened.
enum class HostAliasOutcome {
  kAliasedInPlace,    // already aligned; nothing allocated and nothing copied
  kRehomed,           // an OWNED misaligned buffer moved into an aligned block
  kDeclinedBorrow,    // a misaligned BORROW; the caller must stage
  kDeclinedEmpty,     // no host bytes at all
  kDeclinedDisabled,  // VT_QWEN35_ALIAS_HOST_WEIGHTS=0
};

bool MakeHostBytesDeviceAliasable(const OwnedTensor& w,
                                  HostAliasOutcome* outcome = nullptr);

// Bytes seen by `MakeHostBytesDeviceAliasable`, split by outcome, since process
// start.
//
// WHY A COUNTER AND NOT AN INFERENCE FROM RSS. W0f's first device attempt was
// read only through `free -m`, and what it showed — about 47 GB appearing in
// 30 seconds at the first forward — is equally consistent with "the branch
// declined and staged as before", with "the branch re-homed and the old pages
// did not come back", and with "something else allocated". Those three call for
// three different changes, and no amount of staring at an RSS curve chooses
// between them. This says how many bytes took each outcome. It is printed
// PERIODICALLY rather than at exit, because the process it measures is one the
// memory guard kills before any exit handler runs.
//
// PER CALL, NOT PER WEIGHT, AND THE DIFFERENCE IS THE WHOLE READING. There is no
// memo on the alias branch: `ResidentWeight` re-enters it for every weight on
// every forward step, so a weight aliased 32 times is counted 32 times. These are
// therefore BYTES SEEN — traffic — and they become a residency statement only
// when read at a stated point. The recorded 60.793 GiB is one such reading: the
// first-forward totals at call 1361, where re-homing plateaus and every dense
// weight has been seen exactly once. Quoted without that qualifier the same
// number is a traffic count, and after two decode steps the counter has passed
// 200 GiB on a checkpoint whose resident dense half is 60.8.
struct HostAliasStats {
  uint64_t aliased_in_place_bytes = 0;
  uint64_t rehomed_bytes = 0;
  uint64_t declined_borrow_bytes = 0;
  uint64_t declined_other_bytes = 0;
  uint64_t calls = 0;
};
HostAliasStats HostAliasSnapshot();

// The same-binary A/B back to the staging behaviour, per the house convention
// for a default-on residency change that `VT_ADOPT_DEVICE_BYTES` and
// `VT_MOE_HOST_FREE` already follow. `VT_QWEN35_ALIAS_HOST_WEIGHTS=0` makes
// every call decline, so one build can measure both arms — which matters more
// here than usual, because `src/vllm/model_executor/models/laguna.cpp` records
// a MEASURED GB10 penalty for reading system-allocated memory from the GPU
// rather than a `cudaMalloc` allocation, worst on a long-K low-parallelism
// GEMV. This branch installs exactly that retag by default, and without a knob
// W0e could not tell a decode regression from the workload.
bool HostWeightAliasEnabled();

// May the host mirror of `w` be released, because a DEVICE copy exists to be
// authoritative in its place?
//
// THE INVARIANT A USE-AFTER-FREE TAUGHT US (issue #1299). `MoeBlockBf16Cuda`
// captures `ResidentWeight(...).data` for every expert into a device-resident
// pointer table, uploads the table once, and then releases the host mirrors. It
// justified that with "once the device copy exists it is authoritative and
// nothing reads the host bytes again", which was true while `ResidentWeight`
// had two behaviours. It has three: on a host-addressable platform it ALIASES,
// so the captured pointers ARE `w.bytes.data()` and releasing them frees memory
// the resident table still points at, for the model's lifetime and from inside
// captured graphs. A fresh review demonstrated it with a scratch case that takes
// SIGSEGV.
//
// The question is therefore not "did we upload" but "is there something else to
// read", and `d_dev` already answers it: null on exactly the arm that aliases,
// non-null on every arm that staged. Named rather than inlined so the release
// sites state the invariant they depend on, and so a gate can mutate it.
inline bool HostMirrorIsRedundant(const OwnedTensor& w) {
  return w.d_dev != nullptr;
}

// Lazily-built per-weight DEVICE-RESIDENT state, OWNED BY THE WEIGHT (issue
// #237).
//
// The forward paths build per-weight constants once — arrays of per-expert
// device pointers, Marlin repacks, token row maps — and reuse them on every
// subsequent step. That state used to live in `static` maps keyed on the ADDRESS
// of the weight, built on first touch and never erased, on the assumption that a
// process holds one engine for its lifetime.
//
// The assumption is false and it corrupts output. Destroy a `LoadedEngine` and
// build another in the same process, and the allocator can hand the new weights
// the address the old ones had; the new weights then inherit an entry marked
// ready whose device pointers were `cudaFree`d with the old engine. Nothing in
// this tree destroys the CUDA context, so those pointers stay *mapped* — they
// just now belong to whatever the new engine allocated there, which is how it
// surfaced: not a crash, but zeroed and corrupted output token ids, only in a
// test ordering that builds a second engine.
//
// Holding the state here ties it to the weights it describes, so it cannot
// outlive them and an address cannot be inherited. Deliberately opaque: the
// resident types are CUDA-path implementation details of the model .cpp files,
// and `shared_ptr<void>` keeps them out of this header while still running the
// correct destructor.
//
// This does NOT change the lifetime of the DEVICE allocations those types point
// at. They were leaked for the process before and are leaked now; freeing them
// is a separate question about backend teardown order, and widening this fix to
// touch that would put a shutdown-ordering hazard on the critical path of a
// correctness fix.
struct ResidentSlot {
  // Mutable because building the resident state is logically const: it is a
  // cache of what the weight already contains, populated on first use from a
  // const forward.
  mutable std::shared_ptr<void> state;
};

// Device-resident NVFP4 W4A16 weight (M2.2b). The modelopt packed fp4 codes +
// fp8-e4m3 group scales + per-tensor scale, kept RAW in the ORIGINAL torch
// [N=out_features, K=in_features] orientation vt::MatmulNvfp4 expects (NOT
// transposed to Matmul-B [in,out], and NOT dequanted to bf16). Keeping the
// ~22GB fp4 as fp4 avoids the M2.2-profile CPU dequant (~40 min) + the ~70GB
// bf16 host tensors. On the CUDA path the forward uploads packed+scale to the
// GPU ONCE (lazily, on first use — the mutable device handles below) and reads
// them in place across every step; on the host path it dequants for reference.
// EXL3 (exllamav3 trellis) storage for ONE quantized linear — QUANT-EXL3
// (#2181). Beside `Nvfp4Weight` and for the same reason: a quantized weight is
// data that model containers hold, while the METHOD that multiplies it lives in
// `layers/quantization/exl3.h`. Putting the struct there instead would make
// `qwen3.h` include `linear.h`, which includes `dense_attn_block.h`, which
// includes `qwen3.h`.
//
// THREE tensors, not four. The `mcg` int32 marker some checkpoints also carry
// is a codebook SELECTOR that is never read at inference
// (`exl3_lib/quantize.py:1414-1424`); the loader resolves it into `codebook`
// below, and the stock `turboderp/*-exl3` artifacts ship no `mcg` tensor at all
// — `Linear.is_exl3_storage` requires only `{key}.trellis` with `suh|su` and
// `svh|sv` (`modules/linear.py:385-389`). There are NO SCALES: `exl3.py:38`
// says so ("scale is no longer used"), and a reader looking for one is reading
// a different format.
struct Exl3Weight {
  // I8 [k/16, n/16, 32*bits] — the SAME BYTES the checkpoint stores as
  // `I16 [k/16, n/16, 16*bits]`, held at byte width because that is the shape
  // `vt::Exl3Gemm` reads and because `vt::DType` has no 16-bit integer.
  OwnedTensor trellis;
  OwnedTensor suh;   // F16 [k]  input-side Hadamard sign vector
  OwnedTensor svh;   // F16 [n]  output-side Hadamard sign vector
  // NO DEFAULT ON PURPOSE. An implicit codebook is exactly what shipped a
  // wrong decode: `= 1` here would silently give MCG to every hand-constructed
  // `Exl3Weight`, which is the same shape as reading marker ABSENCE as MCG.
  // -1 is not a codebook, so anything that forgets to set it refuses at
  // `Exl3DecodeCodeword` by name instead of decoding to plausible garbage.
  int codebook = -1;  // 0 == 3INST, 1 == MCG, 2 == mul1; SET IT EXPLICITLY
  // HALF-INTEGER RATE (K+0.5), BACKEND-ROCM frac rates. `bits` then carries the
  // integer KA of `K = KA + 0.5` and the codebook is ALWAYS 2 (mul1) — the
  // oracle defines the fractional rates for `mul1` only (`frac.cu`), and the
  // loader refuses anything else by name. `words % 16 == 8` ⇔ `half`, the same
  // signal `Bits()` reads off the byte width (`last % 32 == 16`).
  bool half = false;

  bool Empty() const { return trellis.Empty(); }

  // k and n from the trellis geometry rather than from a config: a 16x16 tile
  // packs 256 weights, so dim 0 counts input tiles and dim 1 output tiles
  // (`exl3.py:47`).
  int64_t InFeatures() const { return trellis.shape[0] * 16; }
  int64_t OutFeatures() const { return trellis.shape[1] * 16; }

  // BITS ARE PER TENSOR, and `quantization_config.bits` is NOT this number.
  // Measured on `turboderp/Llama-3.2-1B-Instruct-exl3` @ `3.0bpw`: the body is
  // 3-bit while `lm_head.trellis [128, 8016, 96]` is SIX-bit, under a config
  // that says `bits: 3.0`. A reader that trusts the config scalar decodes that
  // head at the wrong width and NO shape check catches it, because the tensor
  // is self-consistent at either reading — the bytes are there and only the
  // values come out wrong.
  int Bits() const {
    VT_CHECK(trellis.rank == 3,
             "exl3: trellis must be 3-D [k/16, n/16, 16*bits] (exl3.py:47), got rank " +
                 std::to_string(trellis.rank));
    const int64_t last = trellis.shape[2];
    // Two byte widths, and the partition is total. INTEGER rates borrow the
    // trellis at `32*bits` bytes (`last % 32 == 0`); FRACTIONAL rates
    // (K = KA + 0.5, mul1) at `32*KA + 16` bytes (`last % 32 == 16`) — the
    // stored 16*KA+8 uint16 words are held at byte width exactly like the
    // integer tiles. `last % 32 == 16` with `KA = last / 32` is the same
    // signal the loader reads off the on-disk word count (`words % 16 == 8`);
    // for `OrcaSAQ-2-27B-EXL3-3.21bpw` that is 56 words = 112 bytes = 3.5 bpw
    // (`exllamav3_ext/quant/frac.cu`, `bpb16 = 16*KA + 8` words per tile).
    VT_CHECK(last > 0 && (last % 32 == 0 || last % 32 == 16),
             "exl3: trellis last dim must be 32*bits BYTES (16*bits i16 words on disk) or "
             "32*KA+16 bytes for a half-integer rate (K+0.5, mul1), got " +
                 std::to_string(last));
    const int64_t bits = last / 32;
    VT_CHECK(bits >= 1 && bits <= 8,
             "exl3: bits must be in [1, 8]; the trellis last dim " + std::to_string(last) +
                 " implies " + std::to_string(bits));
    return static_cast<int>(bits);
  }
};

struct Nvfp4Weight {
  OwnedTensor packed;   // i8 [N, K/2]   two 4-bit E2M1 codes per byte
  OwnedTensor scale;    // i8 [N, K/16]  one fp8-e4m3 scale per 16-elem group
  float scale2 = 0.0F;  // per-tensor weight global scale (1/divisor), multiplied
  int64_t n = 0;        // out_features
  int64_t k = 0;        // in_features (K % 16 == 0)
  bool Empty() const { return packed.Empty(); }

  // Drop this weight's device resident. `packed.d_dev`/`scale.d_dev` are the
  // only owners, so the resets free; an adopted host view (see
  // HostViewIsDeviceTwin) must go first because it holds the same block alive.
  void ReleaseResident() const {
    if (packed.HostViewIsDeviceTwin()) packed.ReleaseHost();
    if (scale.HostViewIsDeviceTwin()) scale.ReleaseHost();
    packed.d_dev.reset();
    scale.d_dev.reset();
  }

  // Block-scale FORMAT. Default = NVFP4: group_size 16, fp8-e4m3 `scale`, a
  // per-tensor `scale2` global. is_mxfp4 selects compressed-tensors MXFP4
  // (`mxfp4-pack-quantized`): group_size 32, E8M0 (UE8M0) `scale` [N, K/32], NO
  // global (scale2 unused). Same E2M1 `packed`. Set ONLY by the dense MXFP4
  // loader; the 27B/35B NVFP4 gate paths never touch it (default false) so their
  // Marlin repack/GEMM are byte-unchanged.
  int group_size = 16;
  bool is_mxfp4 = false;

  // TRUE W4A4 fields (27B compressed-tensors NVFP4; notes §7). Populated ONLY on
  // the 27B CT load (LoadCtNvfp4Raw); left 0 for the 35B modelopt W4A16 weights
  // (which have no activation quant) so `IsTrueW4A4()` gates the 27B alone.
  //   input_global_scale_inv = the ON-DISK activation divisor (2688/amax_act),
  //     passed DIRECTLY to vt::ScaledFp4Quant.
  //   alpha = (1/input_divisor)·(1/weight_divisor) = scale2/input_global_scale_inv,
  //     the single scalar vt::MatmulNvfp4Fp4 multiplies the fp4xfp4 accumulator by.
  // Keep the original on-disk weight divisor as well as its reciprocal. Fused
  // logical linears (qkv/gate_up) take max(divisors) BEFORE reciprocating in
  // vLLM; reconstructing the divisor from scale2 can lose one float ULP.
  float weight_global_scale_inv = 0.0F;
  float input_global_scale_inv = 0.0F;
  float alpha = 0.0F;
  // True when the activation-quant globals were loaded (27B true-W4A4 path).
  bool IsTrueW4A4() const { return alpha > 0.0F; }

  // Device residency lives on `packed.d_dev`/`scale.d_dev` (the generic raw-twin
  // slot): one owner per allocation, no second Nvfp4Weight handle.
  // Lazily-populated SWIZZLED weight block scale for the cutlass sm120a fp4 GEMM
  // path (VT_NVFP4_CUTLASS): [round_up(n,128), round_up(k/16,4)] in the cutlass
  // `scale.d_dev` (the generic raw-twin slot) via vt::SwizzleBlockscale.
  mutable std::shared_ptr<void> d_scale_sw;
  // vLLM/FlashInfer-compatible model-owned f32 alpha for the true-W4A4 CUTLASS
  // path. Uploaded once from the persistent `alpha` member; the diagnostic host
  // scalar path leaves this null.
  mutable std::shared_ptr<void> d_alpha;
  // OPT-IN lifetime residency for the DEQUANTIZED bf16 [K=in, N=out] Matmul-B
  // operand the backends with NO fp4 GEMM multiply against (CPU / Vulkan / Metal /
  // HIP / Tenstorrent; CUDA never dequantizes). Default OFF, and it must stay a
  // per-WEIGHT opt-in: the operand is a bf16 expansion of ~4x the packed bytes, so
  // holding one per tower projection is the double-residency that OOM-reboots a
  // Spark on Vulkan (#203). The dense loader opts in the OUTPUT HEAD alone — one
  // weight, re-read whole every step (~2.54 GB a step rebuilt per call at the
  // 27B's 248320x5120); everything else keeps a per-call copy.
  bool keep_dequant_b = false;
  mutable std::shared_ptr<void> d_dequant_b;

  // Resident Marlin constants (issue #237; see ResidentSlot). `resident_marlin`
  // is this weight's own repack; `resident_marlin_pair` is the fused gate+up
  // repack, held on the GATE weight of the pair (it is the pair's cache key).
  ResidentSlot resident_marlin;
  ResidentSlot resident_marlin_pair;
};

// Device-resident per-tensor FP8 (W8A8) weight — the 35B attn q/k/v/o + GDN
// in_proj_qkv/z/out_proj projections (checkpoint: weight F8_E4M3 + f32
// weight_scale + f32 input_scale, activations W8A8 static per-tensor). Raw IEEE
// fp8-e4m3fn bytes kept in the ORIGINAL torch [N=out_features, K=in_features]
// orientation the cutlass W8A8 GEMM reads directly (NOT transposed to Matmul-B
// [in,out], NOT dequanted to bf16 — halves the projection's device memory vs the
// bf16 field and defers all scaling into the GEMM). On the CUDA path the forward
// uploads the bytes ONCE (lazily; the mutable device handle below) and reads them
// in place across every step. Populated only on the real 35B CUDA load with the
// cutlass W8A8 path enabled (LoadFp8Raw); the bf16 field it replaces is left
// EMPTY. `alpha = input_scale * weight_scale` is precomputed at load (both scales
// are per-tensor scalars) — the single fused scalar vt::MatmulFp8Cutlass applies.
struct Fp8Weight {
  OwnedTensor packed;         // i8 [N, K]  one fp8-e4m3fn byte per element
  float weight_scale = 0.0F;  // per-tensor weight_scale (dequant(w) = f8(w)*this)
  float input_scale = 0.0F;   // per-tensor activation scale (quant a = a/this)
  float alpha = 0.0F;         // input_scale * weight_scale (folded GEMM scalar)
  int64_t n = 0;              // out_features
  int64_t k = 0;              // in_features
  bool Empty() const { return packed.Empty(); }

  // Lazily-populated device-resident copy (CUDA forward only; null on host or
  // before first use). The shared_ptr deleter frees through the vt Backend.
  mutable std::shared_ptr<void> d_packed;
};

// Block-wise (fine-grained) FP8 weight — MODEL-FP8-BLOCK-WEIGHT, #1189 M3, spec
// `.agents/specs/model-fp8-block-weight.md`. One fp8-e4m3fn scale per
// `block_n` x `block_k` tile of the weight, the layout `Qwen/Qwen3.8-27B-FP8`
// ships and vLLM registers as `weight_scale_inv`
// (`vllm/model_executor/layers/quantization/fp8.py:378-379,511` @ `555967922`).
//
// A SIBLING of `Fp8Weight` above, deliberately, not an extension of it. That
// struct is three host floats whose whole point is the `alpha = input_scale *
// weight_scale` folded once at load; a block scheme has NO `input_scale` at all
// (the activation scheme is `dynamic`, and the target checkpoint ships zero such
// tensors) and its weight scale is a 2-D tensor, so there is no value `alpha`
// could take. Adding an optional tensor to `Fp8Weight` would make every existing
// reader of `alpha` — the cutlass and cuBLASLt fp8 wrappers, the merged-QKV alpha
// vector, `PrepareGdnFp8Resident` — carry a silent which-arm branch, and the one
// that forgets it returns a plausible number instead of an error. A distinct type
// makes the wrong call site fail to COMPILE. The shape mirrors `Nvfp4Weight`
// above: an OwnedTensor scale beside the packed values plus lazy device handles.
//
// `scale` is f32 and that is the MIRROR rather than a widening
// (`.agents/porting.md` §"Mirror the memory format"). Upstream allocates the
// parameter `torch.float32` (`utils/fp8_utils.py:1276,1283-1296`; `scale_dtype`
// is None unless `is_scale_e8m0`, which `Fp8Config` does not define) and loads
// the checkpoint tensor into it with `self.data.copy_()`
// (`vllm/model_executor/parameter.py:97`), a CONVERTING copy. A `BF16` tensor on
// disk is therefore widened to f32 once, at load, losslessly. `LoadFp8BlockRaw`
// switches on the on-disk dtype explicitly and has no default branch, because
// #1181 landed a guard for a reader that memcpy'd four bytes whatever the dtype
// was.
//
// `block_n`/`block_k` are carried ON the weight rather than looked up from the
// config at use time: the consumer needs them per GEMM, and a weight that knows
// its own geometry cannot be paired with the wrong one.
//
// CONSUMED BY THE DENSE FORWARD since #1189 milestone M4 (`281b4bc76`), which
// landed `Fp8BlockLinearMethod` and the wiring it names. All ten projections
// in `qwen3_5.cpp` are read, through THREE entry points rather than one.
// `dense_fp8_block::MatmulFp8BlockScaledD` reads eight of them: `o_proj`,
// `down_proj`, the three GDN projections, and q/k/v whenever the split path
// runs. M6 (`836c13c35`) added the other two: `MatmulFp8BlockMergedD` reads
// q/k/v as one operand when the consumer accepts row-strided views, and
// `Fp8BlockGateUpSwiGLUD` is the ONLY reader of `gate_proj_fp8_block` and
// `up_proj_fp8_block`. `PrepareQwen3_5Dense` no longer refuses a populated
// weight -- it refuses a DEVICE with no block-scaled GEMM, which after M5
// (`489a9a4c0`) means a CUDA arch outside `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a).
//
// The CUDA kernel has still NEVER EXECUTED on hardware and there is no token
// gate. That debt is real and is recorded in
// `.agents/specs/vt-matmul-fp8-block-cuda.md`; nothing here narrows it.
struct Fp8BlockWeight {
  OwnedTensor packed;  // i8  [N, K]  one fp8-e4m3fn byte per element, verbatim
  OwnedTensor scale;   // f32 [cdiv(N, block_n), cdiv(K, block_k)]
  int64_t n = 0;       // out_features
  int64_t k = 0;       // in_features
  int64_t block_n = 0;
  int64_t block_k = 0;
  bool Empty() const { return packed.Empty(); }

  // Lazily-populated device-resident copies (CUDA forward only; null on host or
  // before first use). Declared here so M4/M5 upload through the same seam every
  // other quantized weight uses; owed and unpopulated at this merge commit.
  mutable std::shared_ptr<void> d_packed;
  mutable std::shared_ptr<void> d_scale;
};

// The N-concatenated device operand of a MERGED block-wise FP8 group —
// MODEL-FP8-BLOCK-MERGED (#1189 milestone M6, spec
// `.agents/specs/model-fp8-block-merged.md`).
//
// vLLM loads `gate_proj`/`up_proj` into ONE MergedColumnParallelLinear and
// `q`/`k`/`v` into ONE QKVParallelLinear, so one GEMM runs where this tree ran
// two and three. Unlike the per-tensor fp8 case beside it, no alpha vector is
// needed: a block scale is indexed by `n / block_n`, so the shard scale grids
// row-concatenate exactly when each shard's rows begin on a block boundary,
// which is upstream's own merged-partition rule (`fp8_utils.py:1229-1244`).
//
// Built lazily-once by `dense_fp8_block::ResidentFp8BlockMerged`, exactly like
// `Fp8BlockWeight::d_packed`. The per-shard residents are then never built, so
// the merged arm costs no duplicate device bytes. Empty on every non-block
// owner.
struct Fp8BlockMergedResident {
  mutable std::shared_ptr<void> d_packed;  // i8  [sum N_i, K]
  mutable std::shared_ptr<void> d_scale;   // f32 [sum cdiv(N_i,bn), cdiv(K,bk)]
};

// Gated-DeltaNet (linear_attention) layer weights. Projections in Matmul-B
// layout [in, out]; conv1d [conv_dim, K]; a_log/dt_bias f32 [Hv]; norm bf16.
struct GdnLayerWeights {
  OwnedTensor in_proj_qkv;    // bf16 [H, conv_dim]  (FP8 dequant + T)
  OwnedTensor in_proj_z;      // bf16 [H, value_dim] (FP8 dequant + T)
  OwnedTensor in_proj_b;      // bf16 [H, Hv]        (bf16 + T)
  OwnedTensor in_proj_a;      // bf16 [H, Hv]        (bf16 + T)
  // The production owner for vLLM's MergedColumnParallelLinear `in_proj_ba`:
  // raw torch Linear orientation [2*Hv,H], rows [b,a], nk=true. Both
  // safetensors loaders populate this and leave in_proj_b/a empty (dense since
  // KERNEL-GDN-PACKED-DECODE W1, MoE since GDN-MOE-PACKED-BA, #1169). The split
  // rollback slices this owner. The GGUF (#1793) and synthetic paths retain the
  // legacy fields above and leave this empty.
  OwnedTensor in_proj_ba;
  // Qwen3.6-27B production owner for vLLM's MergedColumnParallelLinear
  // `in_proj_qkvz` (W2): raw torch Linear orientation [conv_dim+value_dim, H],
  // rows in exact [q,k,v,z] order (the checkpoint's in_proj_qkv already stacks
  // q|k|v per vLLM's stacked mapping (0,1,2) qwen3_5.py:203-207; in_proj_z
  // appends the z rows as shard 3), nk=true. The real dense loader populates
  // this and leaves in_proj_qkv/in_proj_z empty; the split rollback slices
  // this owner. 35B (FP8 qkv/z) / GGUF / synthetic paths retain the legacy
  // fields above and leave this empty.
  OwnedTensor in_proj_qkvz;
  OwnedTensor conv1d_weight;  // bf16 [conv_dim, K]  (bf16, NOT transposed)
  OwnedTensor a_log;          // f32  [Hv]
  OwnedTensor dt_bias;        // f32  [Hv]
  OwnedTensor norm_weight;    // bf16 [Dv]           (RMSNormGated)
  OwnedTensor out_proj;       // bf16 [value_dim, H] (FP8 dequant + T)
  bool out_proj_tiled = false;  // T25: weight kept in tiled Q5_K order; permute input at runtime

  // ── DEFERRED V-HEAD PERMUTATION (MODEL-MM-QWEN4-EXP,
  //    ISSUE-LOCAL-01M2ENTH6YA5FWEDY6CFHF4NAM) ───────────────────────────────
  //
  // 0 on EVERY loader but the `qwen4exp` GGUF one, and 0 means "the weights
  // above are already in HuggingFace GROUPED V-head order", which is what every
  // other checkpoint in this tree ships and what the GDN block has always
  // assumed. Nothing changes for them.
  //
  // Non-zero is the number of KEY heads, and it says that `in_proj_qkv`'s
  // TRAILING V rows, `in_proj_z`, `in_proj_b`, `in_proj_a` and `out_proj`'s
  // COLUMNS are in the converter's TILED V-head order instead — because undoing
  // that order on the weight forces a k-quant superblock to be dequantized at
  // load (4.152 GB/step of bf16 against 1.508 GB in the file, 61% of this
  // model's decode weight traffic). The block undoes it on the projection
  // VECTORS instead, which is an exact re-indexing of the same values.
  //
  // EVERY OTHER V-INDEXED WEIGHT STAYS GROUPED — `conv1d_weight`'s V channels,
  // `a_log`, `dt_bias` — because they are f32/bf16 already and cost nothing to
  // permute at load. So the permutation the block applies is confined to the
  // four projection outputs and the out-projection input; the conv, the
  // recurrence, the gated norm and both persistent state caches see exactly the
  // grouped order they see today. `norm_weight` is [Dv] and head-agnostic.
  int64_t v_head_perm_key_heads = 0;

  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): block-wise FP8 GDN projections. The
  // target checkpoint lists the GDN small tensors under
  // `modules_to_not_convert`, so these stay empty for it; the rung exists
  // because the SITE probes `F8_E4M3` and a block-wise weight is one.
  Fp8BlockWeight in_proj_qkv_fp8_block;
  Fp8BlockWeight in_proj_z_fp8_block;
  Fp8BlockWeight out_proj_fp8_block;

  // 27B W4A4 fp4-resident variant of out_proj (compressed-tensors NVFP4, notes
  // §3.6). When populated (real 27B CUDA load) the forward calls vt::MatmulNvfp4
  // on it and out_proj above is left EMPTY; the 35B / synthetic loaders populate
  // the bf16 out_proj and leave this empty. Exactly one is filled.
  Nvfp4Weight out_proj_fp4;   // [N=H, K=value_dim]

  // 35B fp8-resident W8A8 variants (per-tensor FP8). Populated BY DEFAULT on the
  // real 35B CUDA+cutlass load (VT_DENSE_NATIVE); the bf16 in_proj_qkv/z/out_proj
  // above are then left EMPTY and the forward calls the native fp8 GEMM (cuBLASLt
  // fp8 by default, or cutlass fp8 under VT_DENSE_CUBLASLT_FP8=0). VT_DENSE_NATIVE
  // =0 flips back to the bf16 fields. The 27B (bf16 in_proj + fp4 out_proj) and
  // GGUF/synthetic (bf16) loaders leave these empty. Forward checks fp8, fp4, bf16.
  Fp8Weight in_proj_qkv_fp8;  // [N=conv_dim, K=H]
  Fp8Weight in_proj_z_fp8;    // [N=value_dim, K=H]
  Fp8Weight out_proj_fp8;     // [N=H, K=value_dim]

  // PERF-27B-GDN-FP8-QKVZ: the FP8 analogue of `in_proj_qkvz`. vLLM runs ONE
  // merged qkvz GEMM per GDN layer, so the two RAW fp8 shards above are
  // N-concatenated ONCE into a single device operand — i8 [conv_dim+value_dim,
  // H], qkv rows first — and the forward issues one fp8 GEMM instead of two.
  // Built lazily-once (and eagerly, PRE-CAPTURE, by
  // Qwen3_5DenseModel::PrepareGdnFp8Resident) exactly like Fp8Weight::d_packed;
  // the shard-local `d_packed` residents are then never built, so the merged
  // arm costs no duplicate device bytes. `d_qkvz_fp8_alpha` is the f32
  // [conv_dim+value_dim] per-output-column folded alpha and stays NULL in the
  // common case where both shards fold the SAME alpha (it is then folded into
  // the GEMM scalar instead). Empty on every non-fp8 owner.
  mutable std::shared_ptr<void> d_qkvz_fp8_packed;
  mutable std::shared_ptr<void> d_qkvz_fp8_alpha;

  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4) — the EXL3 (exllamav3 trellis) arm of
  // this tower, and it is THREE projections rather than the merged qkvz owner
  // above.
  //
  // `Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw` ships `in_proj_qkv` and `in_proj_z` as
  // two INDEPENDENTLY quantized trellises: `in_proj_qkv.svh [10240]` and
  // `in_proj_z.svh [6144]`, each fitted at quantization time against its own
  // projection. The bf16 loader N-concatenates the two `.weight` shards because
  // concatenating bf16 rows is a byte copy. A trellis is a 16x16-tiled
  // bitstream whose two Hadamard sign vectors are per projection, so the same
  // concatenation is defined only when both shards share `suh` bit-for-bit, and
  // nothing in the calibration gives a reason to expect that. There is no
  // merged-trellis operand in this tree either (`quant-exl3-shared.md` records
  // "no merged EXL3 QKV or gate_up" as owed).
  //
  // So the arm MIRRORS THE ARTIFACT and issues two in-projection GEMMs. That is
  // the shape `ProjectGdnQkvz` already has a precedent for: the split native-FP8
  // arm the 35B runs takes exactly this path when the merged owner is empty.
  //
  // Exactly one representation is ever populated. When these are filled, the
  // bf16 `in_proj_qkvz`/`out_proj` and every FP8 and NVFP4 field above are empty.
  Exl3Weight in_proj_qkv_exl3;  // [K=H,         N=conv_dim]
  Exl3Weight in_proj_z_exl3;    // [K=H,         N=value_dim]
  Exl3Weight out_proj_exl3;     // [K=value_dim, N=H]

  // ONE truth for the arm, keyed on the first projection alone, exactly as
  // `FullAttnLayerWeights::IsExl3` keys on `q_proj_exl3`. A predicate that
  // OR-ed the three would report a half-populated container as an arm, which is
  // the state no loader may produce and no consumer may accept.
  bool IsExl3() const { return !in_proj_qkv_exl3.Empty(); }
};

// Full (dense causal) attention layer weights.
struct FullAttnLayerWeights {
  OwnedTensor q_proj;   // bf16 [H, 2*Hq*Dh]  (FP8 dequant + T; output-gate doubled)
  OwnedTensor k_proj;   // bf16 [H, Hkv*Dh]
  OwnedTensor v_proj;   // bf16 [H, Hkv*Dh]
  OwnedTensor o_proj;   // bf16 [Hq*Dh, H]    (FP8 dequant + T)
  OwnedTensor q_norm;   // bf16 [Dh]
  OwnedTensor k_norm;   // bf16 [Dh]

  // 27B W4A4 fp4-resident variants of q/k/v/o_proj (compressed-tensors NVFP4,
  // notes §3.6). Populated on the real 27B CUDA load; the 35B / synthetic
  // loaders populate the bf16 fields above and leave these empty (exactly one
  // set filled). Each kept in the on-disk [N=out, K=in] orientation MatmulNvfp4
  // reads directly.
  Nvfp4Weight q_proj_fp4;  // [N=2*Hq*Dh, K=H]
  Nvfp4Weight k_proj_fp4;  // [N=Hkv*Dh,  K=H]
  Nvfp4Weight v_proj_fp4;  // [N=Hkv*Dh,  K=H]
  Nvfp4Weight o_proj_fp4;  // [N=H,       K=Hq*Dh]

  // CUDA resident for vLLM's QKVParallelLinear. The checkpoint owns logical
  // Q/K/V shards separately; production concatenates their packed rows and
  // linear block scales once, then keeps the combined packed operand and
  // combined swizzled scale resident. The split weights remain available for
  // VT_FP4_MERGED_QKV=0 and non-CUTLASS diagnostics.
  mutable std::shared_ptr<void> d_qkv_packed;
  mutable std::shared_ptr<void> d_qkv_scale_sw;
  // Merged QKV owns the max-before-reciprocal alpha as one physical device
  // scalar, matching its one physical projection. The host member is persistent
  // storage for the asynchronous H2D copy.
  mutable float qkv_alpha = 0.0F;
  mutable std::shared_ptr<void> d_qkv_alpha;

  // 35B fp8-resident W8A8 variants (per-tensor FP8). Populated BY DEFAULT on the
  // real 35B CUDA+cutlass load (VT_DENSE_NATIVE); the bf16 q/k/v/o_proj above are
  // then left EMPTY and the forward calls the native fp8 GEMM (cuBLASLt fp8 by
  // default, or cutlass fp8 under VT_DENSE_CUBLASLT_FP8=0). VT_DENSE_NATIVE=0
  // flips back to the bf16 fields. The 27B (fp4) and GGUF/synthetic (bf16)
  // loaders leave these empty. Forward checks fp8, then fp4, then bf16.
  Fp8Weight q_proj_fp8;  // [N=2*Hq*Dh, K=H]
  Fp8Weight k_proj_fp8;  // [N=Hkv*Dh,  K=H]
  Fp8Weight v_proj_fp8;  // [N=Hkv*Dh,  K=H]
  Fp8Weight o_proj_fp8;  // [N=H,       K=Hq*Dh]

  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): the block-wise (128x128) FP8 variants,
  // populated by the `weight_scale_inv` rung in `qwen3_5_dense_weights.cpp`
  // BEFORE the per-tensor rung, because a block-wise weight is also `F8_E4M3`
  // (#1166). The bf16, fp4 and per-tensor fp8 slots are left EMPTY when these
  // are populated, and vice versa. M4 (#1189, `281b4bc76`) landed the linear
  // method and the forward that reads them, so `PrepareQwen3_5Dense` refuses an
  // unrunnable DEVICE rather than a populated weight.
  Fp8BlockWeight q_proj_fp8_block;  // [N=2*Hq*Dh, K=H]
  Fp8BlockWeight k_proj_fp8_block;  // [N=Hkv*Dh,  K=H]
  Fp8BlockWeight v_proj_fp8_block;  // [N=Hkv*Dh,  K=H]
  Fp8BlockWeight o_proj_fp8_block;  // [N=H,       K=Hq*Dh]

  // CUDA resident for the FP8 (W8A8) analog of QKVParallelLinear (VT_FP8_MERGED
  // _QKV, opt-in). The checkpoint owns logical Q/K/V shards separately with a
  // shared per-tensor input_scale but per-projection weight_scale; production
  // N-concatenates their RAW fp8 bytes into one [Nq+Nk+Nv, K] operand kept
  // resident (below), runs ONE fp8 GEMM with alpha=1 (raw f32 accumulation), and
  // applies each output column's folded scalar (input_scale * that shard's
  // weight_scale) through the resident per-column alpha vector — byte-identical
  // to the three separate per-shard GEMMs when the GEMM accumulation matches.
  // The split fp8 weights above remain available for VT_FP8_MERGED_QKV=0.
  mutable std::shared_ptr<void> d_qkv_fp8_packed;  // i8 [Nq+Nk+Nv, K] raw e4m3fn
  mutable std::shared_ptr<void> d_qkv_fp8_alpha;   // f32 [Nq+Nk+Nv] per-column

  // MODEL-FP8-BLOCK-MERGED (#1189 M6): the BLOCK-wise analog of the resident
  // above, and a much smaller one. Block scales concatenate losslessly along N,
  // so there is no alpha vector and no shared-input_scale guard; the merged
  // operand is the three shards' bytes and scale grids end to end. Default ON
  // wherever the site exposes packed q/k/v views, because there is nothing to
  // trade off. Empty on every non-block owner.
  Fp8BlockMergedResident qkv_fp8_block_merged;

  // QUANT-EXL3 (#2181) / MODEL-QWEN35-EXL3 (#2495 item 3): the exllamav3
  // trellis arm of this tower. Q/K/V stay SEPARATE where the bf16, NVFP4 and
  // block-FP8 arms above hold (or build) one merged owner: a trellis is
  // `[k/16, n/16, 32*bits]`, so joining on the output dim interleaves per input
  // tile rather than row-stacking, which is a real transform and owed its own
  // gate (`## Owed` in `specs/quant-exl3-shared.md`). Exactly one of {bf16,
  // fp4, per-tensor fp8, block fp8, exl3} is populated per layer.
  Exl3Weight q_proj_exl3;  // [K=H,       N=2*Hq*Dh]
  Exl3Weight k_proj_exl3;  // [K=H,       N=Hkv*Dh]
  Exl3Weight v_proj_exl3;  // [K=H,       N=Hkv*Dh]
  Exl3Weight o_proj_exl3;  // [K=Hq*Dh,   N=H]

  // `q_proj_exl3` and not "any of the four": the loader fills all four together
  // or none, and asking about the FIRST one keeps this predicate the same
  // question the loader's own `IsExl3Projection(has, "...q_proj")` probe asked.
  bool IsExl3() const { return !q_proj_exl3.Empty(); }
};

// Exact scalar processing for the three-shard CT NVFP4 QKVParallelLinear.
// Mirrors compressed_tensors_w4a4_nvfp4.py:95-138 and QKVParallelLinear's
// logical-shard loader: take each maximum divisor before reciprocating once.
struct FullAttnQkvGlobals {
  float input_global_scale_inv = 0.0F;  // max on-disk input divisor
  float weight_global_scale = 0.0F;     // reciprocal of max weight divisor
  float alpha = 0.0F;
};

FullAttnQkvGlobals MergeFullAttnQkvGlobals(const Nvfp4Weight& q,
                                           const Nvfp4Weight& k,
                                           const Nvfp4Weight& v);

// Sparse-MoE block (router + per-expert MLP + shared expert). Per-expert and
// shared projections are NVFP4-dequant'd and stored separately (gate/up/down),
// all in Matmul-B layout.
struct MoeBlockWeights {
  OwnedTensor router_gate;   // bf16 [H, E]  (bf16 + T)
  OwnedTensor shared_gate;   // bf16 [H, 1]  (bf16 + T)
  std::vector<OwnedTensor> expert_gate;  // E * bf16 [H, I]
  std::vector<OwnedTensor> expert_up;    // E * bf16 [H, I]
  std::vector<OwnedTensor> expert_down;  // E * bf16 [I, H]
  // A3 keep-quant grouped-MoE fold: when the GGUF experts stay block-quant (the
  // keep-quant arm of LoadExpertsT slices WITHOUT transpose), the whole stacked
  // [E*N, K] block tensor is loaded ONCE here (memory-neutral: replaces the E
  // per-expert `expert_*` copies, not additive) so the MoE forward can issue ONE
  // vt::MatmulBTQuantGrouped per {gate,up,down} instead of E per-expert matvecs
  // (mirror of laguna W9). EMPTY on fp4 / bf16-expand (those keep the vectors above).
  OwnedTensor expert_gate_kq;  // keep-quant stacked [E*I, H]
  OwnedTensor expert_up_kq;    // keep-quant stacked [E*I, H]
  OwnedTensor expert_down_kq;  // keep-quant stacked [E*H, I]
  OwnedTensor shared_gate_proj;  // bf16 [H, Is]
  OwnedTensor shared_up_proj;    // bf16 [H, Is]
  OwnedTensor shared_down_proj;  // bf16 [Is, H]

  // M2.2b fp4-resident variants of the NVFP4 expert/shared projections. When
  // populated (real-checkpoint CUDA load) the forward calls vt::MatmulNvfp4 on
  // these and the bf16 fields above are left EMPTY; the synthetic / GGUF loaders
  // populate the bf16 fields and leave these empty. Exactly one set is filled.
  std::vector<Nvfp4Weight> expert_gate_fp4;  // E * [N=I, K=H]
  std::vector<Nvfp4Weight> expert_up_fp4;    // E * [N=I, K=H]
  std::vector<Nvfp4Weight> expert_down_fp4;  // E * [N=H, K=I]
  Nvfp4Weight shared_gate_proj_fp4;  // [N=Is, K=H]
  Nvfp4Weight shared_up_proj_fp4;    // [N=Is, K=H]
  Nvfp4Weight shared_down_proj_fp4;  // [N=H, K=Is]

  // Resident MoE constants, one slot per forward path (issue #237; see
  // ResidentSlot). Exactly one is populated on a given engine — whichever path
  // this block's experts route through — and all three die with the block.
  ResidentSlot resident_fused;   // MoeFusedResident   (fp4 fused)
  ResidentSlot resident_bf16;    // MoeBf16Resident    (bf16 fast)
  ResidentSlot resident_marlin;  // MoeMarlinResident  (Marlin grouped)
};

// One decoder layer: input/post norms + one attention variant + the MoE block.
struct Qwen3_5MoeLayerWeights {
  bool is_linear_attention = false;
  OwnedTensor input_layernorm;           // bf16 [H]
  OwnedTensor post_attention_layernorm;  // bf16 [H]
  GdnLayerWeights gdn;                    // valid iff is_linear_attention
  FullAttnLayerWeights attn;             // valid iff !is_linear_attention
  MoeBlockWeights moe;                   // all 35B layers are MoE
};

// Whole-model weights.
struct Qwen3_5MoeWeights {
  OwnedTensor embed_tokens;  // bf16 [vocab, H]  (NOT transposed; embed lookup)
  OwnedTensor final_norm;    // bf16 [H]
  OwnedTensor lm_head;       // bf16 [H, vocab]  (bf16/GGUF path; empty when fp4)
  Nvfp4Weight lm_head_fp4;   // [N=vocab, K=H]   (M2.2b fp4-resident; else empty)
  std::vector<Qwen3_5MoeLayerWeights> layers;

  // Deferred routed-expert host materialization — the 35B load-phase peak-PSS
  // interleave (ENG-EXPERT-STREAM / ENG-MOE-HOSTFREE follow-up). When set, the
  // per-layer `moe.expert_*_fp4` host copies are NOT loaded at Load time; instead
  // PrepareMarlinResident calls this per layer IMMEDIATELY BEFORE the device
  // Marlin build + host free, so at most ONE layer's routed-expert host copies
  // coexist (peak host residency ~ one layer, not all N layers ×256 experts). The
  // closure owns the mmap'd safetensors shards (its keepalive); resetting it after
  // the whole model is built returns them. Null on the GGUF / synthetic / Borrow
  // paths (experts already materialized eagerly at load). Non-marlin / non-CUDA /
  // VT_NVFP4_MARLIN=0 falls back to a bulk host materialization (correctness for
  // the wmma/CPU forward that reads these host bytes). `mutable`: materialization
  // is a logically const residency op, like the lazy device-upload handles above.
  mutable std::function<void(int64_t /*layer*/, MoeBlockWeights& /*moe*/)>
      load_layer_experts;
};

// Resolves a tensor name to its StTensor (across shards). Throws if absent.
using TensorResolver = std::function<const StTensor&(const std::string&)>;

// --- Backbone weight namespace (MODEL-TEXT-qwen3-5-*-for-causal-lm) -----------
//
// A Qwen3.5-family checkpoint publishes its text backbone under ONE of two
// spellings. The multimodal wrappers we already gate (Qwen3.6-27B / 35B-A3B /
// Coder) nest it under `model.language_model.`; the TEXT-ONLY arms
// (`Qwen3_5ForCausalLM` / `Qwen3_5MoeForCausalLM`, e.g.
// `Qwen/Qwen3.8-2.4T-A95B`) publish it flat under `model.`. The BACKBONE names
// are otherwise identical — same `mlp.shared_expert_gate.weight`, same
// top-level `lm_head`.
//
// THE PREFIX IS NOT THE ONLY THING BETWEEN THIS LOADER AND A PUBLISHED
// CHECKPOINT, and an earlier revision of this comment wrongly implied it was.
// The published Qwen3.5-family MoE repos ship 3-D STACKED routed experts
// (`...mlp.experts.gate_up_proj` / `.down_proj`) and carry no quantization
// scales at all, while `LoadQwen3_5Moe` reads ONLY per-expert NVFP4. That arm
// is OWED and is refused by name (`CheckMoeExpertLayoutSupported`,
// `qwen3_5_weights.cpp`). The DENSE loader is different: it routes BF16 vs FP8
// vs NVFP4 per projection by tensor presence, so it may genuinely read a flat
// bf16 checkpoint. Resolving the namespace is what THIS seam does; it is not a
// support claim for either published checkpoint.
//
// Upstream normalizes the two with ONE mapper —
//   WeightsMapper(orig_to_new_prefix={"model.language_model.": "model."})
//   (vllm/model_executor/models/qwen3_5.py:296-300 @ `ad5d29db7`, PR #50210,
//    which is AHEAD OF our `555967922` parity pin and recorded as such) —
// so `model.` is canonical and the VL spelling is its accepted alias.
inline constexpr std::string_view kQwen3_5VlBackbonePrefix =
    "model.language_model.";
inline constexpr std::string_view kQwen3_5TextBackbonePrefix = "model.";

// Decides which of the two the checkpoint uses, ONCE, from the shard index, so
// every subsequent lookup in a load uses one namespace. Deliberately NOT a
// per-lookup fallback: a fallback would let a checkpoint bind half its tensors
// from one namespace and half from the other and still appear to load.
//
// Only BACKBONE spellings vote — `<prefix>embed_tokens.weight`,
// `<prefix>norm.weight` and `<prefix>layers.`. `model.visual.*` (the
// vision-inclusive 27B/35B towers) and the top-level `lm_head.*` / `mtp.*`
// therefore cast no vote, which is what keeps a vision checkpoint from looking
// like a flat text one.
//
// Throws std::runtime_error when BOTH namespaces carry backbone tensors (a
// mixed index is refused, never half-loaded) and when NEITHER does.
std::string ResolveQwen3_5BackbonePrefix(
    const std::vector<std::string>& tensor_names);

// --- ENG-LOAD-DIRECT-UPLOAD (issue #150) -------------------------------------
//
// THE DEFECT THIS CLOSES. Loading a checkpoint copies the weights TWICE: the
// loader `memcpy`s each tensor out of the read-only safetensors mmap into an
// owned anonymous buffer, and `ResidentWeight` later copies that buffer into a
// device allocation and drops it. For a 27B that is two full passes over tens
// of GiB where one would do, and the intermediate pass also costs the kernel a
// fresh anonymous page (and its zero-fill) for every page of the model.
//
// THE MECHANISM. For a tensor the device consumes VERBATIM, the owned buffer is
// never allocated at all: `bytes` borrows the mapping (OwnedBytes::Borrow, with
// StTensor::mapping as the keep-alive, so the mapping cannot be unmapped out
// from under it), and the device upload reads straight from the file mapping.
//
// WHAT QUALIFIES, BY CONSTRUCTION. Only a call site that would have performed a
// plain `memcpy` of the WHOLE source range into a freshly allocated destination
// of the SAME size may call this — no transpose, no dtype conversion, no
// dequant, no concatenation of several sources into one buffer, and no
// load-time repack (`repacked`/`q8_0_aligned`/`elem_kn_repacked`, which mutate
// the buffer and are set only on the GGUF path). The size identity is re-checked
// here (`numel(shape) * sizeof(dtype) == t.nbytes`) and the call FAILS CLOSED,
// returning false so the caller runs its normal copy, whenever anything does
// not line up. A reshape is fine: it changes no byte.
//
// Returns true when `o` was made to borrow (its dtype/rank/shape are then set
// from `dtype`/`shape` and its bytes are the mapping's); false when the caller
// must fall back to its existing copy. `VT_LOAD_DIRECT_UPLOAD=0` forces false
// (same-binary A/B, house convention).
bool BorrowStTensorBytes(OwnedTensor& o, const StTensor& t, vt::DType dtype,
                         const std::vector<int64_t>& shape);

// Process-cached gate behind `BorrowStTensorBytes`. Exposed so a test can assert
// which arm it is measuring.
bool LoadDirectUploadEnabled();

namespace detail {
// Test-only override of the direct-upload decision, bypassing the env cache so
// one test binary can exercise both arms. std::nullopt restores the default.
void SetLoadDirectUploadOverrideForTesting(std::optional<bool> value);
}  // namespace detail

// How a checkpoint spells its ROUTED experts (issue #740). Resolved ONCE per
// checkpoint from the shard index and threaded, exactly as the backbone
// namespace is: a per-lookup fallback would let one checkpoint bind half its
// experts from each layout and still appear to load.
//
//  kPerExpertNvfp4  `<layer>.mlp.experts.<e>.{gate,up,down}_proj.weight` U8 +
//                   `.weight_scale` F8_E4M3 + `.weight_scale_2` -- what an
//                   NVFP4 requant (nvidia/Qwen3.6-35B-A3B-NVFP4) ships and what
//                   every gated row reads today. Populates `expert_*_fp4`.
//  kStackedBf16     `<layer>.mlp.experts.{gate_up_proj,down_proj}`, ONE 3-D bf16
//                   tensor per projection holding every expert -- what the
//                   PUBLISHED repos (Qwen/Qwen3.8-2.4T-A95B,
//                   Qwen/Qwen3.6-35B-A3B) ship. Populates the bf16 `expert_*`.
enum class MoeExpertLayout { kPerExpertNvfp4, kStackedBf16 };

// Decides which of the two a checkpoint uses, ONCE, from its shard index —
// the routed-expert sibling of `ResolveQwen3_5BackbonePrefix`, and public for
// the same reason: a caller that wants to know what a published index implies
// must ask the same question the loader asks, not a paraphrase of it.
//
// Only names under `<backbone_prefix>layers.` vote. The top-level `mtp.` draft
// head carries the STACKED spelling even in `nvidia/Qwen3.6-35B-A3B-NVFP4`,
// whose model is per-expert NVFP4, so a scan that counted every `.mlp.experts.`
// name would flip that checkpoint's whole model onto the wrong arm.
//
// Throws when BOTH spellings appear under the backbone (a mixed index is
// refused, never half-bound). An index with NEITHER resolves to the per-expert
// arm — the status quo — and the load then fails at its first missing tensor.
MoeExpertLayout ResolveQwen3_5MoeExpertLayout(
    const std::vector<std::string>& tensor_names,
    const std::string& backbone_prefix);

// --- Everything that is NOT a routed expert (issue #864) ---------------------
//
// #740 gave the MoE loader the published repos' routed experts. It did not make
// a published repo LOAD: `Qwen/Qwen3.6-35B-A3B` and `Qwen/Qwen3.8-2.4T-A95B`
// carry ZERO `weight_scale` / `input_scale` / `scale_inv` tensors anywhere, and
// the loader additionally hard-required per-tensor FP8 for the GDN and
// attention towers and NVFP4 for the shared expert and `lm_head`. Those four
// components are what this enum and struct route by tensor PRESENCE.
//
// `DenseNativeEnabled()` does NOT cover this and is deliberately not widened:
// it switches fp8-RESIDENT against fp8-DEQUANT and BOTH of its branches assume
// an fp8 input, so it is a build/env A/B lever with recorded evidence attached,
// not a layout probe.
enum class MoeProjDtype {
  kBf16,   // `<proj>.weight` BF16, no scales -- what a published repo ships
  kFp8,    // `<proj>.weight` F8_E4M3 + `.weight_scale` (+ `.input_scale`)
  kNvfp4,  // ModelOpt `<proj>.weight` U8 + `.weight_scale` F8 + `.weight_scale_2`
};

const char* MoeProjDtypeName(MoeProjDtype dtype);

// The safetensors dtype string of `name` ("BF16", "F8_E4M3", "U8", ...), or an
// EMPTY string when the checkpoint index has no such tensor.
//
// One callback rather than the dense loader's `TensorResolver` + `TensorExists`
// pair, because presence is `!dtype_of(name).empty()` and the dense ladder needs
// exactly those two questions -- so a caller that has only an INDEX (a manifest,
// a plan audit) can answer both without materializing an `StTensor`.
using TensorDtypeProbe = std::function<std::string(const std::string&)>;

// THE DENSE ARM'S LADDER, AND IT MUST STAY THE DENSE ARM'S LADDER.
// `qwen3_5_dense_weights.cpp`'s `load_projection` (:475-484) asks, in order:
//   1. `IsNvfp4Projection`  = `has(<proj>.weight_packed) || has(<proj>.weight_scale_2)`
//      -- compressed-tensors spells the packed weight `weight_packed`; ModelOpt
//      spells the global scale `weight_scale_2`. Probing only one missed
//      `nvidia/Qwen3.6-27B-NVFP4` entirely (:342-356).
//   2. `get(<proj>.weight).dtype == "F8_E4M3"` -- the `modelopt_mixed` tower.
//   3. otherwise BF16.
// This is that ladder and nothing else. If the two ever disagree about one
// projection, a checkpoint could route differently through two loaders in the
// same build, so `test_qwen3_8_text_only` binds them by loading the SAME
// synthetic projection through the dense loader and comparing which slot it
// filled.
MoeProjDtype ClassifyQwen3_5Projection(const TensorDtypeProbe& dtype_of,
                                       const std::string& proj);

// True iff the checkpoint carries this projection at all, under either spelling
// (mirror of `DenseCheckpointHasLmHead`: a compressed-tensors projection's only
// weight tensor is `<proj>.weight_packed`).
bool Qwen3_5ProjectionPresent(const TensorDtypeProbe& dtype_of,
                              const std::string& proj);

// The four non-routed-expert components, each resolved ONCE per checkpoint and
// THREADED -- the same discipline `ResolveQwen3_5BackbonePrefix` and
// `ResolveQwen3_5MoeExpertLayout` follow, and for the same reason: a per-lookup
// probe would let one checkpoint bind some layers quantized and some bf16 and
// still appear to load.
//
// The defaults are the STATUS QUO before #864 -- per-tensor FP8 towers, NVFP4
// shared expert and head -- so every existing caller of the defaulted seams
// below is unchanged by construction.
//
// WHY FOUR INDEPENDENT DECISIONS AND NOT ONE. A checkpoint that is quantized in
// one component and bf16 in another is ordinary upstream, not a defect:
// `nvidia/Qwen3.6-27B-NVFP4` is `modelopt_mixed` (FP8 attention tower next to
// NVFP4 MLP and a BF16 GDN in-projection) and the dense arm reads it by asking
// per projection. Collapsing the four into one decision would refuse that shape
// and diverge from the ladder above, which is the one thing the spec's stop
// condition forbids. What IS refused is a component that disagrees with ITSELF
// -- see `ResolveQwen3_5MoeTowerDtypes`.
struct Qwen3_5MoeTowerDtypes {
  MoeProjDtype gdn = MoeProjDtype::kFp8;     // linear_attn in_proj_qkv/z, out_proj
  MoeProjDtype attn = MoeProjDtype::kFp8;    // self_attn q/k/v/o_proj
  MoeProjDtype shared_expert = MoeProjDtype::kNvfp4;  // mlp.shared_expert.*
  MoeProjDtype lm_head = MoeProjDtype::kNvfp4;
};

// Resolves all four from the checkpoint index, walking EVERY layer of
// `layer_types` (GDN projections on `linear_attention` layers, attention
// projections on `full_attention` ones, the shared expert on both) plus the
// top-level `lm_head`.
//
// Throws when one component disagrees with itself -- layer 0's `q_proj` BF16
// against layer 4's F8_E4M3, or `gate_proj` BF16 against `down_proj` NVFP4 --
// naming both projections and both dtypes. That is the "half from each" failure
// the once-per-checkpoint discipline exists to prevent, and unlike a missing
// tensor it would otherwise produce wrong logits rather than an error.
//
// A component with no projections present at all keeps its default, so an index
// that simply lacks a tensor still fails at the reader with `tensor not found`.
Qwen3_5MoeTowerDtypes ResolveQwen3_5MoeTowerDtypes(
    const TensorDtypeProbe& dtype_of, const std::string& backbone_prefix,
    const std::vector<std::string>& layer_types);

// --- The load PLAN (issue #740, .agents/specs/moe-bf16-stacked-experts.md) ---
//
// WHAT PROBLEM THIS SOLVES. `Qwen/Qwen3.8-2.4T-A95B` is ~4.8 TB over 213 shards.
// Nothing here can hold it, so "the reader works at 35B" is the only byte-level
// evidence available — and on its own it does not show that the 2.4T's OWN
// names, shapes and dtypes resolve, nor that the per-expert offset arithmetic
// survives its dimensions (one layer's `gate_up_proj` is 34,359,738,368 bytes,
// which overflows int32 by four orders of magnitude).
//
// So: walk the whole load for a config WITHOUT allocating or reading a single
// weight byte, and report every tensor it would fetch. Checked against the
// published `model.safetensors.index.json`, that answers "would it load on
// hardware that can hold it?" for exactly the part that can be answered without
// the hardware.
//
// WHAT IT DELIBERATELY DOES NOT CLAIM: a generated token, throughput, memory
// headroom, or that any allocation path survives at that scale.
//
// THE PLAN IS ONLY WORTH ANYTHING IF IT IS A PROJECTION OF THE LOADER RATHER
// THAN A SECOND MODEL OF IT. `PlanQwen3_5MoeLoad` therefore mirrors
// `LoadQwen3_5Moe` helper for helper, including the `DenseNativeEnabled()`
// decision that adds `.input_scale` to every FP8 projection, and the test suite
// binds the two: it builds a synthetic checkpoint from the plan ALONE, requires
// the production loader to read it, and then removes each planned tensor in turn
// and requires the load to fail naming exactly that tensor. A plan entry the
// loader does not want, or a tensor it wants that the plan omits, fails there.
struct PlannedTensor {
  std::string name;
  // The safetensors dtype string the load hard-requires (`LoadBf16Direct` wants
  // BF16, `LoadFp8Raw` F8_E4M3, `LoadNvfp4Raw` U8 + F8_E4M3 + F32, ...).
  std::string dtype;
  // The shape the CONFIG implies. EMPTY when neither the loader nor the config
  // determines it: the attention projections' output width depends on
  // `attn_output_gate`, which `HfConfig` does not carry, so this planner states
  // no shape for them rather than a plausible one that is wrong on every real
  // checkpoint (the 2.4T's `q_proj` is [32768, 8192], twice heads*head_dim).
  std::vector<int64_t> shape;
  // True when the LOADER ITSELF checks this shape against config, rather than
  // reading it off the header. Only the 3-D stacked routed experts do — which is
  // exactly the arithmetic this row added, so it is the one shape whose
  // agreement with the published index is a statement about the reader and not
  // about this planner.
  bool shape_enforced = false;
};

// Every tensor `LoadQwen3_5Moe` would fetch for `config`, in load order, without
// touching a weight byte. `backbone_prefix` and `layout` are what
// `ResolveQwen3_5BackbonePrefix` / `ResolveQwen3_5MoeExpertLayout` resolved from
// the index. Does NOT include `mtp.*`: `LoadQwen3_5MTP` loads that optional
// draft head separately, and only when speculative decoding is enabled.
//
// Identical for the eager and DEFERRED (`load_layer_experts`) residency paths —
// deferring changes WHEN the routed experts are read, never WHICH.
//
// `tower` is what `ResolveQwen3_5MoeTowerDtypes` resolved for the four
// non-routed-expert components; it changes the REQUEST SET (an FP8 projection
// asks for two or three tensors where a BF16 one asks for a single `.weight`),
// so a plan built with the wrong one is not a projection of the loader.
std::vector<PlannedTensor> PlanQwen3_5MoeLoad(
    const HfConfig& config, const std::string& backbone_prefix,
    MoeExpertLayout layout, Qwen3_5MoeTowerDtypes tower = {});

// Load one decoder layer's weights from real tensors. `layer_type` is
// "linear_attention" or "full_attention"; `num_experts` drives the expert loop.
// Exercised on real data by the Task 3 unit test (both layer types live in
// shard 1). Prefix is "{backbone_prefix}layers.{layer_idx}.", and the default
// is the VL spelling every checkpoint we gate today uses, so this seam is
// byte-identical for the 27B/35B/Coder callers. `LoadQwen3_5Moe` passes the
// prefix it resolved once from the shard index.
//
// `layout` likewise defaults to the arm every gated caller uses, so this seam
// stays byte-identical for them. `hidden` is read ONLY by the stacked arm, which
// needs it to resolve the 3-D tensors' orientation the way upstream does; it is
// unused (and may be 0) for the per-expert arm.
//
// `tower` likewise defaults to the arm every gated caller reads (FP8 towers,
// NVFP4 shared expert), so this seam stays byte-identical for them.
Qwen3_5MoeLayerWeights LoadQwen3_5MoeLayer(
    const TensorResolver& get, const std::string& layer_type, int64_t layer_idx,
    int64_t num_experts,
    const std::string& backbone_prefix = std::string(kQwen3_5VlBackbonePrefix),
    MoeExpertLayout layout = MoeExpertLayout::kPerExpertNvfp4,
    int64_t hidden = 0, Qwen3_5MoeTowerDtypes tower = {});

// Full-model load: resolves every param across the given shards (name -> shard
// looked up from each file's own header), dequantizes/transposes, and returns
// owned host bf16 tensors. Uses config.num_hidden_layers, config.layer_types,
// config.num_experts.
//
// `shards_owner` (optional): when non-null it SHARES ownership of `shards`
// (`shards_owner.get() == &shards`), which lets the loader DEFER the routed-MoE
// experts' host copies — the returned weights carry a `load_layer_experts`
// closure (holding `shards_owner` as its keepalive) that PrepareMarlinResident
// drives per layer to bound peak host residency (35B load-phase peak-PSS lever).
// When null the experts are loaded EAGERLY (the mmap'd shards may then be
// released as soon as this returns) — used by GGUF/synthetic/borrowed paths and
// any caller that will drop the shards before Prepare.
Qwen3_5MoeWeights LoadQwen3_5Moe(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    std::shared_ptr<const std::vector<SafetensorsFile>> shards_owner = nullptr);

// ── The MoE arm's VISION TOWER (issue #891, .agents/specs/moe-vision-tower.md) ─
//
// `LoadQwen3_5Moe` above reads the TEXT backbone only. `Qwen/Qwen3.6-35B-A3B`
// ships 333 `model.visual.*` tensors alongside it, and until this seam existed
// they were simply not read — the load succeeded and produced a text-only model,
// which is the silent-drop failure class this project keeps rediscovering.
//
// Upstream composes the SAME tower on both arms: `Qwen3_5MoeForConditionalGener
// ation` and `Qwen3_5ForConditionalGeneration` each hold a `Qwen3_VisionTransfor
// mer` (pinned vLLM `qwen3_5.py`), so this is the SHARED
// `LoadQwen3VLVisionWeights` reader (`qwen3_vl.h`) the dense 27B arm already
// gates at image 32/32 + video 32/32 — deliberately NOT a second tower.

// The tower geometry for a Qwen3.6 conditional-generation checkpoint. Mirrors
// the checkpoint's `config.json::vision_config`; `Qwen/Qwen3.6-35B-A3B` and
// `Qwen/Qwen3.6-27B` publish the SAME tower (depth 27, hidden 1152, heads 16,
// intermediate 4304, patch 16, temporal patch 2, spatial merge 2, 2304 position
// embeddings, EMPTY `deepstack_visual_indexes`) and differ only in
// `out_hidden_size`, which is the text backbone's hidden size (2048 on the 35B
// MoE, 5120 on the 27B dense) because the merger writes straight into the text
// residual stream. That one field is therefore taken from `config.hidden_size`
// rather than hardcoded. `HfConfig` does not parse `vision_config` (the dense
// arm's gate hardcodes the same numbers in-test).
//
// NO DeepStack: `deepstack_visual_indexes: []` compiles that path out for this
// family upstream (`qwen3_vl.py:1709-1716`), so the tower output is exactly
// [N, out_hidden_size].
multimodal::Qwen3VLVisionConfig Qwen3_5MoeVisionConfig(const HfConfig& config);

// Load the MoE arm's vision tower from the SAME shards the text backbone came
// from, through the shared `LoadQwen3VLVisionWeights`.
//
// REFUSES BY NAME when the checkpoint carries no `model.visual.*` tensor at all:
// a Qwen3.5-family `*ForConditionalGeneration` checkpoint without a tower cannot
// answer an image or video prompt, and returning an empty tower would let it
// answer from text alone and still emit plausible tokens. AGENTS.md: an arm that
// is not implemented "is refused with a message naming the missing piece ...
// never left to be discovered later". `nvidia/Qwen3.6-35B-A3B-NVFP4` is exactly
// such a checkpoint (`vision_config` declared, `visual.*` weights absent).
multimodal::Qwen3VLVisionWeights LoadQwen3_5MoeVision(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// True iff any shard carries a `model.visual.` tensor. Exposed so a caller can
// route to the text-only path deliberately instead of discovering the refusal.
bool HasQwen3_5MoeVisionTower(const std::vector<SafetensorsFile>& shards);

// ── MODEL-QWEN35-DENSE-VL-EXL3: the DENSE arm's vision tower. ────────────────
// Upstream composes the SAME `Qwen3_VisionTransformer` on
// `Qwen3_5ForConditionalGeneration` as on the MoE arm (pinned vLLM
// `qwen3_5.py`), over the same 333 `model.visual.*` tensors. So these mirror
// the MoE wrappers above onto the dense arm through the SAME shared
// `LoadQwen3VLVisionWeights` reader — no second tower — and the geometry is
// one family builder (`Qwen3_5FamilyVisionConfig`) the two wrappers share. The
// dense 27B arm is already token-gated on this tower at image 32/32 +
// video 32/32; what this row adds is the LOADER SEAM the dense loader never
// had, so the EXL3 27B checkpoint's `model.visual.*` tensors stop being
// silently unread.
multimodal::Qwen3VLVisionConfig Qwen3_5DenseVisionConfig(const HfConfig& config);

// Load the dense arm's vision tower from the SAME shards the text backbone
// came from. REFUSES BY NAME when the checkpoint carries no `model.visual.*`
// tensor at all — the identical refusal discipline the MoE wrapper applies.
multimodal::Qwen3VLVisionWeights LoadQwen3_5DenseVision(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// True iff any shard carries a `model.visual.` tensor (the family probe).
bool HasQwen3_5DenseVisionTower(const std::vector<SafetensorsFile>& shards);

// THE STAGING POLICY. The three declarations below decide ONE question: do this
// model's dense weights get a true device copy, or do they keep the retag above?
//
// WHY THIS EXISTS, MEASURED. The retag above costs real decode throughput on
// GB10: interleaved A/B on `dgx:gpu0`, one boot, Qwen3.8-27B bf16 + DFlash2
// k=7 at concurrency 1, alias ON vs alias OFF, four warm repeats each ->
// 11.677 / 11.693 tok/s aliased against 14.288 / 14.337 staged, **+22.5%**,
// the two arms agreeing to 0.14% and 0.34%. That is the same mechanism
// `laguna.cpp:130-132` records and the same 20-30% band that took Laguna and
// DeepSeek-V4 from behind their references to ahead of them.
//
// WHY IT IS NOT A BLANKET DEFAULT FLIP. This file also serves
// `Qwen3.8-2.4T-A95B`, and #1299 measured that model exhausting a 119.631 GiB
// box precisely because the CUDA arm paid for its weights TWICE — host bytes
// plus a device copy. Staging is the right default for a model that fits and
// is fatal for one that does not, so the question has to be asked of the BOX
// and of this model's own total, rather than answered once for the file.
//
// THE RULE, IN FULL. `StagingFitsModel` below is arithmetic over two STABLE
// numbers: `2 * model_weight_bytes + reserve_bytes <= device_total_bytes`,
// where `reserve_bytes` comes from `VT_QWEN35_STAGE_RESERVE_BYTES` (12 GiB by
// default). `SetSafetensorsWeightBudget` latches that answer ONCE per process.
// NOTHING here is asked per weight, and no fraction of LIVE FREE memory is read
// — an earlier form did both, and the comment on `SetSafetensorsWeightBudget`
// below records what that cost. A 50 GiB model on a 119.6 GiB box clears the
// rule and stages; the 2.4T checkpoint cannot and never does, keeping exactly
// the behaviour #1299 shipped.

// Does a model of `model_weight_bytes` leave room for a SECOND, device-resident
// copy of itself on a device of `device_total_bytes`, keeping `reserve_bytes`
// for the KV cache and activations?
//
// PURE, so it is gateable without a device.
//
// THE FACTOR OF TWO IS THE WHOLE POINT. On a platform that does not release the
// host mirror after upload -- which is every unified-memory part, and which is
// what `ResidencyPolicy::release_host_weights_after_upload` records -- a staged
// weight is ADDITIVE: the host bytes stay and the device copy joins them. That
// second copy is exactly what made #1299's 2.4T checkpoint exhaust a 119.631 GiB
// box, and it is exactly what buys +21.1% for a model that fits.
bool StagingFitsModel(size_t model_weight_bytes, size_t device_total_bytes,
                      size_t reserve_bytes);

// Latch the decision for this process, ONCE, from two STABLE numbers. Called by
// the safetensors loader, which is the only place that knows the model's total.
//
// WHY ONCE AND NOT PER WEIGHT. The first attempt at this asked
// `Backend::DeviceMemoryInfo` per weight and compared LIVE FREE against a
// fraction of total. Free falls as the host mirror loads, so the floor degraded
// into "stage the first N GiB, then stop": measured 8.1% of an available 21.1%
// on the committed gate. `platforms/interface.h:66-69` had already written the
// rule -- the budget is "TOTAL rather than FREE, because `free` at load time
// carries the page cache and whatever else the box is doing, which would make a
// load-time verdict a function of contention."
//
// A model that never calls this keeps the retag, which is what every GGUF path
// does today and is how #1299's checkpoint stays untouched.
void SetSafetensorsWeightBudget(size_t model_weight_bytes, size_t device_total_bytes);

// The latched answer. False until `SetSafetensorsWeightBudget` says otherwise.
bool StageOwnedWeightsToDevice();

}  // namespace vllm
