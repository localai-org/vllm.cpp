// Faithful 1:1 port of vllm/platforms/interface.py:134-1290 (class Platform) @
// pin 5559679229 — the device-capability / memory-model seam. (Re-anchored from
// `:134-229 @ e24d1b24`, the pin retired at W5: the class is unchanged in the
// parts we mirror, and the cited `supported_dtypes` / `get_device_capability`
// members moved rather than changing.) See
// .agents/porting-inventory.md §9 note 8 (the platforms/ tree is a faithful
// mirror of the upstream seam, NOT a deviation) and
// .agents/specs/extensibility-platform-seam-2026-07-18.md.
#pragma once

#include <cstddef>
#include <string>
#include <string_view>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace vllm::platforms {

using vt::Backend;
using vt::DeviceType;
using vt::DType;

// Mirrors vllm/platforms/interface.py:69-131 (class DeviceCapability): a device's
// compute capability as (major, minor). `present() == false` models a platform
// with no queryable capability (CPU) — upstream get_device_capability() -> None.
struct DeviceCapability {
  int major = -1;
  int minor = -1;
  bool present() const { return major >= 0; }
  // interface.py:124 to_int — <major><minor>, assuming a single-digit minor.
  int to_int() const { return major * 10 + minor; }
};

// Discrete-vs-unified memory residency policy — the PR #4 discrete-memory debt
// made DATA and, since item 2 (BACKEND-PLATFORM), CONSUMED by the model residency
// path (qwen3_5.cpp) rather than decided inline. The host-free / load-stream /
// device-scratch-pool decisions read these fields from
// `GetPlatform(<obj>.device.type).residency_policy()`, so a new (e.g. discrete)
// GPU changes ONLY its platform's policy values and the model code is UNCHANGED.
// On GB10/unified today CUDA advertises release=true (the routed MoE experts'
// host mirror IS freed after the device Marlin build — ENG-MOE-HOSTFREE /
// -LOADSTREAM) + pool=true/uncapped (the DevicePool); CPU advertises the
// unified-host retain/no-pool defaults below.
struct ResidencyPolicy {
  // May host-side weight bytes be freed once a device-resident copy exists?
  // A discrete GPU reclaims host RAM; unified/host platforms keep the one copy.
  // Consumed by ShouldReleaseHostWeights / ShouldInterleaveLoadStream below.
  bool release_host_weights_after_upload = false;
  // Is device scratch served from a reuse pool (qwen3_5.cpp DevicePool) rather
  // than freed straight to the driver? cudaMalloc/cudaFree are expensive and
  // illegal under graph capture, so CUDA pools; host platforms do not need it.
  bool uses_device_memory_pool = false;
  // Optional soft cap (bytes) on pooled device scratch; 0 == uncapped (today).
  // Consumed by the DevicePool: a discrete GPU sets a bound; GB10 leaves it 0.
  size_t device_pool_cap_bytes = 0;
  // Total device memory this platform can draw weight allocations from (bytes).
  // 0 == UNKNOWN, and unknown must never be read as "unlimited" or as "nothing
  // fits": a caller that cannot learn the budget declines to decide. Consumed by
  // the load-time GGUF fit refusal (`gguf_device_fit.h`, issue #1123).
  //
  // TOTAL rather than FREE, because `free` at load time carries the page cache
  // and whatever else the box is doing, which would make a load-time verdict a
  // function of contention.
  //
  // This is NOT `vt::Backend::DeviceMemoryInfo`, which is a live free/total
  // probe whose only consumer is `Gemma4MoE`'s device-expert LRU and which
  // `CudaBackend` does not override at all -- so wiring CUDA into that seam
  // would wake a landed residency policy that is currently dead on CUDA, which
  // is a behaviour change with its own measurement to make (issue #1126).
  // Probed once at platform registration; on a GB10 `cudaMemGetInfo` reports
  // 128452956160 (119.631 GiB) where `nvidia-smi` reports `[N/A]`.
  size_t device_memory_total_bytes = 0;
};

// Residency DECISIONS derived from a ResidencyPolicy — the single, testable place
// the host-weight-release and per-layer load-stream-interleave questions are
// answered from platform data (BACKEND-PLATFORM item 2). Consumed by
// qwen3_5.cpp; unit-tested in tests/vllm/platforms/test_platform.cpp so the
// wiring the model uses is exercised on the CPU tier. The kernel-path predicate
// `marlin_committed` (is the Marlin resident the committed compute path, so the
// host-reading wmma fallback can never re-read the freed bytes) is ORTHOGONAL to
// the platform POLICY and is supplied by the caller; `host_free_env` is the
// VT_MOE_HOST_FREE same-binary A/B override (house convention). A discrete GPU
// flips ResidencyPolicy and both this logic and the model are unchanged.
inline bool ShouldReleaseHostWeights(const ResidencyPolicy& policy,
                                     bool marlin_committed, bool host_free_env) {
  return marlin_committed && policy.release_host_weights_after_upload &&
         host_free_env;
}
inline bool ShouldInterleaveLoadStream(const ResidencyPolicy& policy,
                                       bool marlin_committed) {
  return marlin_committed && policy.release_host_weights_after_upload;
}

// The CUDA platform's residency policy, assembled from the values its registrar
// probes. `src/vllm/platforms/cuda.cpp` calls this and supplies
// `cudaMemGetInfo`'s `total`; it holds no policy of its own.
//
// A free function HERE, rather than a body inside that file, because that
// translation unit compiles ONLY in a CUDA build. While the assembly lived there,
// nothing on a host without a CUDA toolkit could reach it — no test, and no
// mutation, which is why "delete the `device_memory_total_bytes` assignment" was
// recorded as an owed mutation by #1123 instead of being proven. This header
// compiles everywhere, so tests/vllm/platforms/test_platform.cpp pins the
// assembly on every host and `cuda.cpp` is left holding only the probe it alone
// can make (#1136). The probe itself is still not mutation-proven; a CUDA build
// is the only thing that reaches the `cudaMemGetInfo` call.
//
// `device_memory_total_bytes` is 0 when the probe failed, which the load-time
// GGUF fit refusal reads as UNKNOWN and never as "nothing fits".
inline ResidencyPolicy CudaResidencyPolicy(size_t device_memory_total_bytes) {
  ResidencyPolicy p;
  p.release_host_weights_after_upload = true;  // freed after the Marlin build
  p.uses_device_memory_pool = true;            // qwen3_5.cpp DevicePool
  p.device_pool_cap_bytes = 0;                 // uncapped
  p.device_memory_total_bytes = device_memory_total_bytes;
  return p;
}

// A 1:1 port of `vllm/v1/attention/selector.py::AttentionSelectorConfig`
// (`:24-39`) @ pin `5559679229` — the ONE request description upstream feeds to
// BOTH `vllm/platforms/cuda.py::_get_backend_priorities` (`:83-163`) and
// `vllm/v1/attention/backend.py::AttentionBackend.validate_configuration`
// (`:320-393`). The capability itself is NOT a field — it is the platform's own
// `get_device_capability()`, exactly as upstream passes it in beside the config
// (`cuda.py:381-384`).
//
// It lives under `platforms/` rather than under `v1/` because
// `Platform::get_attn_backend_priority` takes it and `platforms/` must not depend
// on `v1/`. That is also why `attn_type` below is a STRING: upstream's
// `AttentionType` is a `str` enum (`backend.py:38-46`) and `supports_attn_type`
// takes a `str` (`:292-298`), so the string IS the faithful shape;
// `vllm::v1::AttentionTypeName` converts our enum to it.
//
// Every default is upstream's, so every existing caller
// (`get_attn_backend_priority()` with no argument, and the runner's dense walk)
// keeps its exact selection behavior.
struct AttnSelectorConfig {
  // model_config.use_mla — the MLA branch of _get_backend_priorities:93-142.
  // On our gate models (Qwen3 dense / GDN) this is false.
  bool use_mla = false;
  // Sparse (DSA / V3.2 indexer) attention. NOT a _get_backend_priorities input
  // upstream — it is the `is_sparse()` check in validate_configuration that
  // REJECTS a sparse backend for a dense request and vice versa. Carried here so
  // the selector can apply that filter (see SelectAttentionBackendName): this is
  // the seam a future DSA/sparse-MLA backend slots into with ZERO selector edit.
  bool use_sparse = false;
  // cuda.py:105 `num_heads is not None and num_heads <= 16` (sm_100 sparse-MLA
  // ordering only). 0 == unknown, upstream's `None`.
  int num_heads = 0;
  // cuda.py:98 `is_quantized_kv_cache(kv_cache_dtype)` (sm_100 sparse-MLA
  // ordering only). Our KV cache is bf16/f32 today, so false. Upstream DERIVES
  // this from `kv_cache_dtype` below; it stays a separate field because the MLA
  // priority rows and their tests already read it, and the two are asserted
  // consistent nowhere — recorded as such rather than silently unified.
  bool quantized_kv_cache = false;

  // ─── selector.py:25-28 — the four inputs upstream makes REQUIRED ───────────
  // The attention head dimension. 0 is accepted by every predicate whose
  // supported list is empty, which is upstream's answer for an unconstrained
  // backend (`backend.py:159-161`).
  int head_size = 0;
  // The model/query dtype (`supports_dtype`, backend.py:163-165). Upstream has
  // no `None` here. The runner does not yet supply it (issue #1332 M4), so on
  // the production path it keeps this default.
  DType dtype = DType::kBF16;
  // The KV-cache quantization name ("auto", "fp8", "fp8_e4m3", ...). EMPTY is
  // upstream's `None`, which `supports_kv_cache_dtype` accepts outright
  // (`backend.py:167-173`).
  std::string kv_cache_dtype;
  // The framework block size. 0 is upstream's `None`, accepted outright
  // (`backend.py:176-178`).
  int block_size = 0;

  // ─── selector.py:30-39 — the request flags, upstream defaults ─────────────
  bool has_sink = false;
  bool use_mm_prefix = false;
  bool use_per_head_quant_scales = false;
  // `AttentionType` is a `str` enum upstream; "decoder" is `AttentionType.DECODER`.
  std::string attn_type = "decoder";
  bool has_sliding_window = false;
  // Non-causal (bidirectional) decoder attention — DFlash's drafter sets it.
  // Also a `_get_backend_priorities` input at this pin (`cuda.py:88,148`), which
  // is issue #1333.
  bool use_non_causal = false;
  bool use_batch_invariant = false;
  bool use_kv_connector = false;
  bool use_pcp = false;
};

// Faithful port of vllm/platforms/interface.py:134-229 `class Platform`. Exposes
// the capability queries the engine/model code branches on. A Platform COMPOSES
// a vt::Backend (include/vt/backend.h) for the concrete device ops — it does NOT
// replace it: the memory-model queries delegate to the backend, while
// capability/dtype/residency queries are platform metadata.
class Platform {
 public:
  virtual ~Platform() = default;

  virtual DeviceType device_type() const = 0;

  // interface.py:189-202 is_cuda / is_cpu.
  bool is_cuda() const { return device_type() == DeviceType::kCUDA; }
  bool is_cpu() const { return device_type() == DeviceType::kCPU; }

  // The composed vt::Backend (index 0). Memory-model queries delegate here.
  virtual Backend& backend() const = 0;

  // True when host and device share one address space (backend.h:45). CPU and
  // GB10/Apple are unified; a discrete GPU is not.
  bool is_unified_memory() const { return backend().UnifiedMemory(); }

  // Graph/command capture capability (backend.h:80 SupportsGraphCapture).
  bool supports_graph_capture() const { return backend().SupportsGraphCapture(); }

  // interface.py:420-431 get_device_capability. **THE UNIT IS AN NVIDIA SM
  // VERSION**, and upstream's docstring says so outright: "Stateless version of
  // torch.cuda.get_device_capability". Every predicate written against this value
  // — FlashAttentionBackend::supports_compute_capability (flash_attn.py:200-202,
  // `>= (8, 0)`), CudaPlatform::supports_fp8 (`has_device_capability(8, 9)`) — is
  // a statement about SM versions and about nothing else.
  //
  // A PLATFORM WITH NO SM VERSION REPORTS ABSENT (`present() == false`). It must
  // never answer in a unit of its own, however natural that unit is on the
  // device. That is upstream's rule, applied by upstream to itself in
  // xpu.py:228-234: "capacity format differs from cuda's and will cause
  // unexpected failure, so use None directly".
  //
  // Absent is also what makes ONE shared selector correct. Upstream reaches
  // `validate_configuration` only from CudaPlatform (cuda.py:381,410, guarded by
  // `assert device_capability is not None` at :404) and RocmPlatform
  // (rocm.py:531,558); every other platform has its own get_attn_backend_cls and
  // never evaluates the SM predicate (cpu.py:75-87). Our
  // SelectAttentionBackendName is shared across every DeviceType, so what
  // upstream gets from its callers we get from this contract: absent skips the
  // predicate exactly as upstream's CPU/XPU platforms skip it, present feeds it
  // an SM value exactly as upstream's CUDA/ROCm platforms do.
  //
  // Answering in a foreign unit was #1823: Metal reported the Apple GPU family
  // and Vulkan the Vulkan API version, and an SM-8.0 bar was applied to both.
  // tests/vllm/platforms/test_platform.cpp gates the contract for every
  // registered platform.
  virtual DeviceCapability get_device_capability() const = 0;

  // interface.py:433-454 has_device_capability — is this platform >= a required
  // (major, minor)? False when the platform has no queryable capability (CPU).
  bool has_device_capability(int major, int minor) const;

  // interface.py:481-493 is_device_capability_family — is the device capability
  // any <major>.x (CUDA-13 "family" architecture semantics, e.g. 10.x, 11.x,
  // 12.x)? Argument is a full capability int (e.g. 120), mirroring upstream's
  // `(current_capability.to_int() // 10) == (capability // 10)`. False when the
  // platform has no queryable capability (CPU). Non-virtual: derived from
  // get_device_capability(), so a new platform answers it for free. Upstream uses
  // it for ROCm gfx-family / CUDA sm-family gating; carried for the ROCm port.
  bool is_device_capability_family(int capability) const;

  // --- Portable capability predicates (interface.py:914-984,1051-1072) --------
  //
  // The capability half of the Platform seam the audit found "accidentally
  // absent" (accelerator-seam-audit.md §"Our baseline" 1, row S3 / §11.5): the
  // fp8/fp4/attention/graph capability queries our shared layer branches on TODAY
  // via a raw `device.type == kCUDA` test. Making them Platform POLICY (which is
  // where vLLM owns them) is what lets a `device==kCUDA` fast-path gate become a
  // capability predicate: byte-identical today (CUDA answers true, every other
  // platform answers the base false — exactly what `device==kCUDA` returned), yet
  // genuinely DECOUPLING (a future accelerator with the fast path answers true for
  // itself, with no edit at the call site). This is the audit's class-D fix, and
  // it is where the deferred fp4/fp8 gates (S4 §9.3) convert, since their bespoke
  // ops are dual-registered CPU+CUDA and so cannot use `vt::OpRegistered` (S6 §11).
  //
  // Base defaults mirror upstream's `Platform` base (all False); the CudaPlatform
  // leg overrides with the vLLM-mirrored answers, and its definition lives in the
  // DSR-allowlisted CUDA platform file, so the removed call-site `kCUDA` is a real
  // decoupling drop, not a moved comment. These are POLICY on Platform, never
  // implementation-selection — that stays `vt::OpProvider` (audit Risks/dec. 6).

  // interface.py:933 + cuda.py:562 supports_fp8 — does this platform support FP8
  // types? CUDA: has_device_capability(8,9); every other platform: false.
  virtual bool supports_fp8() const { return false; }

  // nvfp4_utils.py:56 cutlass_fp4_supported (+ csrc nvfp4_scaled_mm_entry.cu:71)
  // — is the CUTLASS/FlashInfer NVFP4 true-W4A4 fp4-activation fast path available
  // here? CUDA: compute capability in [100,130); every other platform: false.
  virtual bool cutlass_fp4_supported() const { return false; }

  // interface.py:977 + cuda.py:570 opaque_attention_op — is attention registered
  // as one giant opaque custom op on this platform? CUDA: true; else false.
  virtual bool opaque_attention_op() const { return false; }

  // interface.py:914 + cuda.py:675 is_integrated_gpu — is the GPU an integrated
  // (UMA) device sharing system memory with the CPU? CUDA: probed once at
  // registration (torch is_integrated analogue, cudaDevAttrIntegrated); else
  // false. Surface parity for the ROCm/memory-reporting port; unwired today.
  virtual bool is_integrated_gpu() const { return false; }

  // ENG-EXPERT-STREAM-DEVICE W0b (issue #1124). May a KERNEL on this platform
  // DEREFERENCE ordinary host storage — a `std::vector<uint8_t>`, an mmap'd
  // file page — with no staging copy, no registration and no map/unmap call?
  //
  // This is a PROBED PHYSICAL PROPERTY, never a device name, an architecture
  // string or a compute capability. CUDA answers from
  // `cudaDevAttrPageableMemoryAccess AND cudaDevAttrIntegrated`, probed once at
  // registration; ROCm from the `pageable_memory_access` + `integrated` pair it
  // already probes. The conjunction is the point: a DISCRETE card with HMM
  // reports pageable access alone and would then serve every expert slice over
  // PCIe with page migration, which is the opposite of what the caller wants.
  //
  // WHY NOT AN EXISTING PREDICATE. `is_cpu()` is what this lifts. Every one of
  // the other four answers a different question:
  //
  //   needs_weight_staging()  is TRUE on CUDA everywhere, GB10 included, and is
  //                           a PROGRAMMING-MODEL policy — the model's forward
  //                           binds distinct device pointers. It would gate
  //                           nothing here, and the two deliberately DISAGREE on
  //                           GB10 (stages: yes; can read host memory: yes).
  //   is_unified_memory()     answers the OPPOSITE question: CUDA on GB10
  //                           reports unified because host and device address one
  //                           physical pool, while a `cudaMalloc` pointer is
  //                           still not host-dereferenceable — see
  //                           `include/vt/backend.h::DeviceMemoryIsHostAddressable`.
  //   is_integrated_gpu()     alone is insufficient:
  //                           `src/vllm/platforms/rocm.cpp::is_integrated_gpu`
  //                           records integrated-WITHOUT-pageable-access as a
  //                           real device class, and on it a host pointer faults.
  //   Backend::DeviceMemoryIsHostAddressable()
  //                           is the MIRROR of this one and not a substitute. It
  //                           asks whether a pointer from `Alloc()` may be
  //                           dereferenced by the HOST; this asks whether a host
  //                           pointer may be dereferenced by the DEVICE. GB10
  //                           answers false to that one and true to this one.
  //
  // Base false, because being wrong here hands a host pointer to a device kernel.
  // A platform must OPT IN from a probe, exactly like the backend seam above.
  virtual bool host_memory_is_device_addressable() const { return false; }

  // interface.py:1058 + cuda.py:662 support_static_graph_mode — does this platform
  // support static (CUDA-graph) capture mode? CUDA: true; else false. Distinct
  // from supports_graph_capture() (the backend-level capability); this is the
  // platform POLICY gate, mirroring upstream.
  virtual bool support_static_graph_mode() const { return false; }
  // Some graph backends require metadata uploads outside the recorded region.
  virtual bool static_graph_requires_persistent_inputs() const { return false; }
  virtual int max_static_graph_batch_size() const { return 64; }

  // Model-specific admission can be narrower than the platform-wide default.
  virtual bool support_static_graph_for_model(const std::vector<std::string>& /*architectures*/,
                                              int /*device_index*/) const {
    return support_static_graph_mode();
  }

  // #1625/#2812: whether a decode family needs the EXPLICIT capture opt-in on
  // this platform. TT's static-graph mode is default-on since the #1625 flip,
  // but only the Qwen3-dense family carries committed captured-arm gate
  // evidence (the 0.6B capture pair; deterministic 4B and Mistral runs).
  // Every OTHER driver conjuncts this policy query and keeps the pre-flip
  // opt-in polarity until its captured arm is brought up (#2812 — the
  // Qwen3.5-GDN arm TT_FATALs mid-capture today). Qwen3-dense does not call
  // it. Trivially false elsewhere: CUDA's captured decode is the proven
  // standard arm.
  virtual bool static_graph_requires_opt_in() const { return false; }

  // Architecture-aware overload for drivers whose ONE graph class serves
  // several model families: the Qwen3-dense graph also serves the Mistral,
  // Llama and InternLM2 registries, so the driver passes its checkpoint's
  // `architectures` and the platform carves default capture to exactly the
  // evidence family. The base ignores the architectures.
  virtual bool static_graph_requires_opt_in(
      const std::vector<std::string>& /*architectures*/) const {
    return static_graph_requires_opt_in();
  }

  // --- Residency / attention fast-path POLICY (accelerator-seam S7) -----------
  //
  // Two more capability predicates the shared layer branches on TODAY via a raw
  // `device.type == kCUDA` test, hoisted to Platform POLICY (audit §12.3, work
  // row S7). Same discipline as the S3 block above: byte-identical today (CUDA
  // answers true, every other platform the base false — exactly what
  // `device==kCUDA` returned), yet genuinely DECOUPLING (a future accelerator
  // answers for itself, with no edit at the call site). Definitions live in the
  // DSR-allowlisted CUDA platform leg, so the removed call-site `kCUDA` is a real
  // decoupling drop. POLICY on Platform, never implementation-selection
  // (audit Risks/dec. 6).

  // Does this platform require host tensors to be STAGED into a distinct
  // device-resident buffer before a kernel can read them? This is the CUDA
  // programming-model residency policy, NOT a physical-memory probe:
  // is_unified_memory()/is_integrated_gpu() are the WRONG predicate because GB10
  // is physically unified yet the CUDA path STILL stages (device pointers are a
  // distinct address the kernels bind), so a memory-property probe would FLIP the
  // path (audit §12.3). CUDA: true; every other platform: base false — exactly
  // what `device==kCUDA` answered. Governs the model's device-resident forward as
  // a whole: resident-weight upload (ResidentWeight), device-resident state I/O
  // (IndexedGdnStateIoEnabled), the merged/packed GDN projections (which slice a
  // device-resident packed-GEMM owner) and the direct-device-load precondition —
  // all the SAME "operands live in a staged device buffer" policy, versus the
  // host-resident direct-view reference path. A future discrete GPU answers true
  // (it stages); a future direct-host accelerator answers false (it reads host
  // memory in place) — the decoupling a bare `kCUDA` cannot express.
  virtual bool needs_weight_staging() const { return false; }

  // BACKEND-ROCM, issue #1934. Does a load on this platform allocate BOUNDED,
  // budget-checkable device memory for resident weights — the narrower
  // PHYSICAL question `needs_weight_staging()` deliberately does NOT answer.
  //
  // `needs_weight_staging()`'s own doc above says it "governs the model's
  // device-resident forward as a whole", listing `ResidentWeight` among what it
  // gates — that listing is STALE. `qwen3_5.cpp::ResidentWeight` was fixed
  // under issue #125 to key its alias-vs-upload branch on `is_cpu()`, not
  // `needs_weight_staging()`, precisely because the old predicate answered
  // false for every non-CUDA device (Vulkan, Metal, XPU, ROCm) and each of
  // them aliased a HOST pointer into a DEVICE kernel. So today, on ANY
  // non-CPU platform, `ResidentWeight::Alloc` genuinely allocates bounded
  // device memory regardless of `needs_weight_staging()` — confirmed on ROCm
  // by issue #1870's own reproduction, a real `hipMalloc: out of memory`.
  //
  // What `needs_weight_staging()` correctly still gates is a DIFFERENT
  // question: should the OPTIMIZED device-resident forward run — the indexed
  // GDN state-I/O kernels (though `IndexedGdnStateIoEnabled` already
  // special-cases a non-staging, non-CPU device by checking op registration
  // directly, so ROCm already takes the fast arm there without this method),
  // the merged/packed GDN projections, and the fp8/bf16 GDN resident-prep
  // passes. Those default to the ROW-COPY REFERENCE path today on ROCm, and
  // this method changes NONE of them: flipping `needs_weight_staging()`
  // itself was considered and rejected (see the row's spec) because at least
  // one of those consumers has no op-registration fallback and would silently
  // assume kernels a device might not have, exactly the failure mode
  // `IndexedGdnStateIoEnabled` was written to avoid for the ONE consumer that
  // already checks.
  //
  // Consumed by the ONE production call site of `CheckDeviceWeightFit`
  // (`gguf_device_fit.h`, issue #1123): the load-time refusal only needs to
  // know "will this load draw from a bounded device memory pool, and do we
  // know its size" — not "should the fully-optimized forward run".
  //
  // Default DELEGATES to `needs_weight_staging()`, so CUDA's answer (true) is
  // unchanged and every platform that overrides neither method reads exactly
  // as it did before this method existed — a pure additive seam.
  virtual bool allocates_bounded_device_memory() const {
    return needs_weight_staging();
  }

  // Does this platform have the fused flash-attention-2 (native-bf16) attention
  // fast path? The FA2 dispatch (qwen3_5.cpp GdnBlockPaged / full-attn preamble)
  // emits bf16 q/k and a bf16 attention output — the combo the vendored CUDA
  // split-KV FA2 kernel (cuda_paged_attn.cu:2494) consumes — versus the f32
  // graph-captured fallback. Upstream expresses this as an attention-backend
  // choice (FLASH_ATTN is a registered CUDA AttentionBackendEnum entry,
  // registry.py:34-120); here the kernel lands one layer lower (the
  // vt::PagedAttention op), so `vt::OpRegistered` cannot see the FA2 SUB-kernel
  // (audit class D, §9.3) and the choice is a Platform predicate. CUDA: true;
  // every other platform: base false — exactly what `device==kCUDA` answered. A
  // future accelerator with a bf16 flash-attention kernel answers true.
  virtual bool supports_fa2_attention() const { return false; }

  // interface.py:181-187 supported_dtypes. The FIRST entry is the default
  // fallback for the platform ("auto" dtype resolution).
  virtual std::vector<DType> supported_dtypes() const = 0;

  // Discrete-vs-unified residency/memory-model policy (folds the PR #4 debt).
  virtual ResidencyPolicy residency_policy() const = 0;

  // Can this platform actually RUN the model registered under `architecture`?
  //
  // WHY THIS EXISTS (BACKEND-METAL-MLX work row M3a). "A partial backend is a
  // supported, tested state" (src/vt/ops.cpp:104-111) has always been true one
  // layer down — `vt::GetOp` throws for an unregistered op. But once
  // model_loader.cpp::SelectQueue started asking the Platform seam instead of
  // hardcoding kCUDA, "which device does this process run on" stopped being the
  // same question as "which device can run THIS model". Metal implements exactly
  // the ops OPT-125m needs; pointing a Qwen3-dense engine at it produces a
  // late, obscure failure inside a kernel bind ("table points outside every
  // Metal allocation") instead of a clean fall-back to the CPU reference.
  //
  // The default is TRUE and is the right default for a COMPLETE backend: CUDA
  // (74/75 ops) and CPU keep their exact behaviour, and this is never consulted
  // for them in any way that changes an outcome. A PARTIAL backend overrides it
  // and names the architectures whose ops it has actually registered — which is
  // a claim the backend can honour, and which shrinks to nothing the moment the
  // remaining kernels land.
  //
  // Deliberately keyed on the ARCHITECTURE STRING rather than an OpId manifest:
  // the registration already carries the architecture, the manifest would have
  // to be maintained in lockstep with every forward, and a stale manifest would
  // fail in exactly the silent way this is meant to prevent.
  virtual bool supports_model_architecture(std::string_view architecture) const {
    (void)architecture;
    return true;
  }

  // Capability-ordered attention-backend priority: the ordered list of backend
  // NAMES this platform prefers, highest-priority first. Mirrors
  // vllm/platforms/cuda.py::_get_backend_priorities (the ordered
  // AttentionBackendEnum list) and is consumed by
  // vllm::v1::SelectAttentionBackendName (mirror of get_attn_backend_cls), which
  // returns the first REGISTERED name. The base default is empty (no preference);
  // CudaPlatform/CpuPlatform override with the vLLM-mirrored lists. (Item 4 —
  // attention-backend registry seam; formerly a stub returning 0.)
  //
  // `cfg` carries upstream's `use_mla` (+ the sparse/num_heads/kv-dtype inputs).
  // The DEFAULT argument lives here only: overrides declare the parameter
  // without a default, and every call through a `Platform&` picks this one up,
  // so the pre-MLA zero-argument call sites are unchanged.
  virtual std::vector<std::string> get_attn_backend_priority(
      const AttnSelectorConfig& cfg = AttnSelectorConfig{}) const {
    (void)cfg;
    return {};
  }

  // Capability-ordered MLA *prefill* backend priority — a separate selector
  // upstream: vllm/v1/attention/backends/mla/prefill/selector.py:47-76
  // `_get_mla_prefill_backend_priorities`. Distinct from the decode/dense list
  // above because MLA runs a materialized-MHA prefill against a different
  // backend family. Base default empty; CudaPlatform mirrors the upstream lists.
  virtual std::vector<std::string> get_mla_prefill_backend_priority() const {
    return {};
  }
};

// Self-registered per DeviceType, copying the RegisterBackend/GetBackend
// static-init idiom (src/vt/backend.cpp). All registration completes before
// main() runs (platforms register via static initializers); afterwards these
// are lock-free reads with no synchronization.
void RegisterPlatform(DeviceType type, Platform* platform);
// The platform for a specific DeviceType (throws if none registered).
Platform& GetPlatform(DeviceType type);
bool HasPlatform(DeviceType type);

// Find a registered platform by vt::DeviceTypeName(), or nullptr when that
// canonical user-facing name has no usable platform in this process. This is
// the shared explicit-device seam: callers name a platform without branching
// on a backend-specific DeviceType literal.
Platform* FindPlatformByName(std::string_view name);

// The process's active compute platform: the highest-priority registered
// accelerator, else CPU — mirrors vLLM resolving `current_platform` by probing
// for an accelerator and falling back to CPU. This answers ONLY the process-level
// "what accelerator is this process on" question. It is NOT a per-tensor device
// test: on a GPU box it returns the CUDA platform regardless of the object being
// operated on, so a CPU queue/tensor on a GPU box would wrongly read as CUDA. For
// per-object device dispatch use `GetPlatform(<obj>.device.type)` (BACKEND-PLATFORM).
Platform& CurrentPlatform();

// The ORDER CurrentPlatform() probes device types in, exposed so it can be
// ASSERTED rather than trusted (BACKEND-ROCM W0). This walk is the one place
// adding a platform is not additive: a DeviceType missing from it registers
// fine, answers every query correctly, and is then simply never selected — and
// unlike a missing enum case, no compiler diagnostic fires. Exposing it lets
// tests/vllm/platforms/test_platform.cpp assert that every DeviceType appears,
// so the next backend cannot reintroduce that silence. Returns a pointer to a
// static array of `count` entries.
const DeviceType* CurrentPlatformPriority(size_t& count);

// ENG-EXPERT-STREAM-DEVICE W0b, issue #1378. The DECISION behind
// `CudaPlatform::host_memory_is_device_addressable()`, as a pure function of the
// two probed attribute values, so it can be checked without the hardware.
//
// It exists because the conjunction was gated by nothing. The rule is declared
// load-bearing in three places -- the base predicate's own comment above,
// `src/vllm/platforms/cuda.cpp`'s probe, and this row's spec -- and the only CUDA
// box this project can reach reports BOTH attributes as 1, so `test_platform`
// could assert the implication (`!hmda || integrated`) and the board value and
// still pass with either term deleted. Both mutations were run in the fresh
// review of #1377 and both came back GREEN. A rule that no reachable input can
// falsify is a comment, not a guard.
//
// The device class the conjunction is FOR is the one nobody here owns: a discrete
// card with HMM answers `pageable = 1, integrated = 0`, and on it the lane would
// serve ~6.95 GB of expert slices per token over PCIe with page migration. Taking
// the arguments as the raw `int`s `cudaDeviceGetAttribute` writes, rather than as
// `bool`s, keeps this callable with exactly the values the probe produced --
// including the 0 a FAILED probe leaves behind, which is the conservative answer
// and is part of what the cases pin.
//
// The ROCm leg answers the identical conjunction inside
// `vt::rocm::HostMemoryIsDeviceAddressable` (`src/vt/rocm/rocm_backend.hip`). It
// does NOT call this function and cannot: `vt` does not depend on `vllm`, and
// inverting that layering to share four bytes of boolean algebra would be the
// worse trade. The duplication is recorded here rather than left to be found.
bool HostMemoryIsDeviceAddressableFromAttrs(int pageable_memory_access,
                                            int integrated);

}  // namespace vllm::platforms
