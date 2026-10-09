// vllm.cpp original; see qwen3_5_dense.h. Loader for the DENSE Qwen3.6-27B text
// gate (compressed-tensors NVFP4 W4A4). Mirrors the 35B loader structure
// (qwen3_5_weights.cpp) but routes each Linear bf16 vs W4A4-materialized-to-bf16
// by name (notes §3.6) and swaps the MoE block for the dense SwiGLU MLP.
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_gdn_replay.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <limits>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "vllm/config/multimodal.h"
#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_config.h"
#include "vllm/model_executor/layers/quantization/fp8_block_quant.h"
#include "vllm/model_executor/layers/quantization/modelopt_mixed_precision.h"
#include "vllm/model_executor/models/dense_fp8_block_gemm.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/interfaces.h"  // #607 L3 SkipTowerForModalities
#include "vllm/platforms/interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/unaligned.h"

namespace vllm {

// The generic BF16 copy/transpose/merge routines now live in the shared
// dense_weight_loaders.h (SEAM GAP #3 extraction) so qwen3_weights.cpp reuses
// them. Re-expose them unqualified for the 27B call sites below — behavior and
// loaded bytes are byte-identical to the former anon-namespace copies.
using dense_loaders::LoadBf16Direct;
using dense_loaders::LoadBf16Transposed;
using dense_loaders::MakeOwned;
// The per-tensor scale read (#1181). The local copy this replaces checked
// neither the element count nor the dtype.
using dense_loaders::ReadF32Scalar;

namespace {

using TensorExists = std::function<bool(const std::string&)>;

// GPTQ stores its unquantized text remainder as FP16. Keep those values in
// their checkpoint dtype; the FP16 execution path is wired separately.
OwnedTensor LoadGptqF16(const TensorResolver& get, const std::string& name,
                        const std::vector<int64_t>& shape, bool nk = false) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "F16" && t.shape == shape,
           "gptq4: expected F16 tensor with configured shape for " + name);
  OwnedTensor out = dense_loaders::LoadF16Direct(get, name, shape);
  out.nk = nk;
  return out;
}

OwnedTensor LoadGptqF16MergedNK(const TensorResolver& get,
                                const std::string& first,
                                const std::string& second, int64_t first_n,
                                int64_t second_n, int64_t k) {
  const StTensor& a = get(first);
  const StTensor& b = get(second);
  const std::vector<int64_t> first_shape{first_n, k};
  const std::vector<int64_t> second_shape{second_n, k};
  VT_CHECK(a.dtype == "F16" && b.dtype == "F16" &&
               a.shape == first_shape && b.shape == second_shape &&
               a.nbytes == static_cast<size_t>(first_n * k) * 2 &&
               b.nbytes == static_cast<size_t>(second_n * k) * 2,
           "gptq4: invalid merged FP16 BA geometry");
  OwnedTensor out = MakeOwned(vt::DType::kF16, {first_n + second_n, k});
  out.nk = true;
  std::memcpy(out.bytes.data(), a.data, a.nbytes);
  std::memcpy(out.bytes.data() + a.nbytes, b.data, b.nbytes);
  MaybeReleaseSourcePages(a.data, a.nbytes);
  MaybeReleaseSourcePages(b.data, b.nbytes);
  return out;
}

OwnedTensor LoadGptqF16VectorAsF32(const TensorResolver& get,
                                   const std::string& name, int64_t n) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "F16" && t.shape == std::vector<int64_t>{n} &&
               t.nbytes == static_cast<size_t>(n) * 2,
           "gptq4: invalid FP16 vector for " + name);
  OwnedTensor out = MakeOwned(vt::DType::kF32, {n});
  for (int64_t i = 0; i < n; ++i) {
    const float value = vt::F16ToF32(
        vt::LoadUnaligned<uint16_t>(t.data + i * 2));
    std::memcpy(out.bytes.data() + i * 4, &value, 4);
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return out;
}

// The GDN in-projections stay RAW in the on-disk torch Linear [out, in]
// orientation (nk=true, via LoadMergedBf16RawNK), consumed by vt::MatmulBT —
// the cuBLASLt TN fast path (K contiguous in both operands, the layout vLLM's
// F.linear hits; nvjet TNNN on GB10). Measured 2026-07-10: the transposed
// [K,N] Matmul-B orientation forces cuBLASLt onto NNNN/sm80-cutlass kernels —
// 27B GDN in_proj site 2.29 vs vLLM 1.80 us/tok. Raw NK also skips the
// load-time host transpose.

// Model-dtype vectors are BF16 in the NVFP4 gate checkpoint but F32 in ordinary
// Qwen3.5 dense checkpoints. Normalize them to the representation consumed by
// the existing forward without changing the quantized path.
OwnedTensor LoadModelBf16Direct(
    const TensorResolver& get, const std::string& name,
    const std::vector<int64_t>& shape_override = {}) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16" || t.dtype == "F32",
           "qwen3_5 dense: expected BF16 or F32 for " + name);
  const std::vector<int64_t> shape =
      shape_override.empty() ? t.shape : shape_override;
  // ENG-LOAD-DIRECT-UPLOAD (issue #150): ONLY the BF16 arm is verbatim. The F32
  // arm below CONVERTS f32 -> bf16 and therefore can never borrow; the dtype
  // test is what keeps the two apart, and BorrowStTensorBytes independently
  // rejects the f32 source anyway (its span is twice the bf16 destination).
  OwnedTensor borrowed;
  if (t.dtype == "BF16" &&
      BorrowStTensorBytes(borrowed, t, vt::DType::kBF16, shape)) {
    return borrowed;
  }
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  if (t.dtype == "BF16") {
    VT_CHECK(t.nbytes == o.bytes.size(),
             "qwen3_5 dense: byte-size mismatch for " + name);
    std::memcpy(o.bytes.data(), t.data, t.nbytes);
  } else {
    VT_CHECK(t.nbytes == static_cast<size_t>(o.Numel()) * sizeof(float),
             "qwen3_5 dense: byte-size mismatch for " + name);
    auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
    const auto* src = static_cast<const uint8_t*>(t.data);
    for (int64_t i = 0; i < o.Numel(); ++i) {
      float value = 0.0F;
      std::memcpy(&value, src + static_cast<size_t>(i) * sizeof(value),
                  sizeof(value));
      dst[i] = vt::F32ToBF16(value);
    }
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

// Only the scoped XPU policy materializes EXL3 remainders as FP16.
// Other backends retain their model-dtype reader, including F16-to-BF16.
OwnedTensor LoadModelVectorForScheme(const TensorResolver& get, bool exl3,
                                     const std::string& name, bool fp16 = false,
                                     const std::vector<int64_t>& shape = {}) {
  if (exl3 && fp16) return dense_loaders::LoadF16Direct(get, name, shape);
  if (exl3 && get(name).dtype == "F16")
    return dense_loaders::LoadF16AsBf16Direct(get, name, shape);
  return LoadModelBf16Direct(get, name, shape);
}

OwnedTensor LoadBf16RawNK(const TensorResolver& get,
                          const std::string& name) {
  OwnedTensor out = LoadBf16Direct(get, name);
  VT_CHECK(out.rank == 2,
           "qwen3_5 dense: expected 2-D weight for " + name);
  out.nk = true;
  return out;
}

// BF16/F32 [n] -> owned f32 [n] (A_log / dt_bias).
OwnedTensor LoadToF32(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  VT_CHECK(t.dtype == "BF16" || t.dtype == "F32",
           "qwen3_5 dense: expected BF16 or F32 for " + name);
  VT_CHECK(t.shape.size() == 1,
           "qwen3_5 dense: expected 1-D tensor for " + name);
  const int64_t n = t.shape[0];
  OwnedTensor o = MakeOwned(vt::DType::kF32, {n});
  auto* dst = reinterpret_cast<float*>(o.bytes.data());
  if (t.dtype == "F32") {
    VT_CHECK(t.nbytes == o.bytes.size(),
             "qwen3_5 dense: byte-size mismatch for " + name);
    std::memcpy(dst, t.data, t.nbytes);
  } else {
    // Unaligned: `t.data` is an arbitrary byte offset into the mmap (#627).
    for (int64_t i = 0; i < n; ++i) {
      dst[i] = vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(t.data + i * 2));
    }
  }
  MaybeReleaseSourcePages(t.data, t.nbytes);
  return o;
}

bool DirectDeviceLoadEligible(vt::Queue* queue) {
  if (queue == nullptr ||
      !platforms::GetPlatform(queue->device.type).needs_weight_staging()) {
    return false;
  }
  const char* release = std::getenv("VT_RELEASE_HOST_WEIGHTS");
  if (release != nullptr && release[0] == '0') return false;
  const char* direct = std::getenv("VT_DIRECT_DEVICE_LOAD");
  if (direct != nullptr && direct[0] == '0') return false;
  const auto& platform = platforms::GetPlatform(queue->device.type);
  // LOAD-MODELOPT-NVFP4-BORROW (issue #1647). This used to also require
  // `!platform.is_unified_memory()`, which is TRUE on GB10, so
  // `StageAndReleaseLoadedDense` -- the only caller of
  // `ReleaseResidentQwen3_5DenseHostWeights` -- never ran on the one device in
  // this fleet whose host and device memory come out of the same 119 GiB.
  //
  // TWO PREDICATES DISAGREED ABOUT ONE DEVICE, and the staging one is right.
  // `needs_weight_staging()` is asked at the top of this function and is
  // unconditionally true on CUDA; `src/vllm/platforms/cuda.cpp` says why in its
  // own words -- the CUDA path binds distinct device-resident buffers
  // "regardless of GB10 being physically unified". `interface.h` says the same
  // from the other side: `is_unified_memory()` "answers the OPPOSITE question",
  // because a `cudaMalloc` pointer there is still not host-dereferenceable.
  // `ResidentNvfp4` settles it by construction: it allocates and copies with no
  // aliasing branch, so on GB10 there really are two buffers.
  //
  // THIS DOES NOT ARM THE ALIASING USE-AFTER-FREE the release site's comment
  // records. That site asks the invariant `HostMirrorIsRedundant(tensor)` --
  // `d_dev != nullptr` -- rather than the `d_dev || d_dev_f32` proxy, and
  // `ResidentWeight`'s alias branch leaves `d_dev` null, so an aliased weight is
  // never released. Its own conclusion is that the disjunct is redundant on the
  // staging arm; what changes here is that the invariant now does the work
  // instead of an accident of platform naming.
  return platform.residency_policy().release_host_weights_after_upload;
}

void StageAndReleaseLoadedDense(Qwen3_5DenseWeights& weights,
                                vt::Queue& queue) {
  Qwen3_5DenseModel::PrepareBf16Resident(weights, queue);
  vt::GetBackend(queue.device.type).Synchronize(queue);
  (void)ReleaseResidentQwen3_5DenseHostWeights(weights);
}

// One compressed-tensors NVFP4 W4A4 Linear -> RAW fp4-resident Nvfp4Weight kept
// in the on-disk [N=out, K=in] orientation vt::MatmulNvfp4 reads directly (notes
// §5 step-6a — the throughput path; NO bf16 materialization). Reads the CT
// tensor names `<proj>.weight_packed` (U8 [out,in/2]), `<proj>.weight_scale`
// (F8_E4M3 [out,in/16], LINEAR non-swizzled) and `<proj>.weight_global_scale`
// (F32 scalar). The CT global scale is stored as a DIVISOR, so scale2 is its
// RECIPROCAL (notes §3.3) — the ONLY math delta vs the modelopt LoadNvfp4Raw;
// the byte encoding (E2M1 nibbles + fp8-e4m3 group-16 scale) is identical, so the
// existing M2.7 tensor-core GEMM carries these bit-for-bit as the CPU dequant
// reference (DequantCtNvfp4WeightToF32). Activation-quant is dropped (bf16
// activations, W4A16-style) — the on-disk `input_global_scale` is not read.
Nvfp4Weight LoadCtNvfp4Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8",
           "qwen3_5 dense: expected U8 weight_packed for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "qwen3_5 dense: expected 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "qwen3_5 dense: expected F8_E4M3 weight_scale for " + proj);
  const float wgs_disk = ReadF32Scalar(get, proj + ".weight_global_scale");
  VT_CHECK(wgs_disk != 0.0F,
           "qwen3_5 dense: zero weight_global_scale (divisor) for " + proj);

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.weight_global_scale_inv = wgs_disk;  // retain exact divisor for fused linears
  r.scale2 = 1.0F / wgs_disk;  // CT stores 1/scale (divisor) -> reciprocate
  // TRUE W4A4 (notes §7): the on-disk activation divisor drives vt::ScaledFp4Quant
  // (used DIRECTLY), and alpha folds both reciprocated globals for the fp4xfp4
  // GEMM: alpha = (1/input_divisor)·(1/weight_divisor). input_global_scale is a
  // per-tensor F32 scalar present on every 27B quantized Linear (§3.2).
  const float igs_disk = ReadF32Scalar(get, proj + ".input_global_scale");
  VT_CHECK(igs_disk != 0.0F,
           "qwen3_5 dense: zero input_global_scale (divisor) for " + proj);
  r.input_global_scale_inv = igs_disk;   // on-disk divisor, used directly
  r.alpha = r.scale2 * (1.0F / igs_disk);
  // ENG-LOAD-DIRECT-UPLOAD (issue #150): both payloads are verbatim.
  if (!BorrowStTensorBytes(r.packed, packed, vt::DType::kI8,
                           {out_dim, in_dim / 2})) {
    r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
    VT_CHECK(packed.nbytes == r.packed.bytes.size(),
             "qwen3_5 dense: packed byte-size mismatch for " + proj);
    std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
    MaybeReleaseSourcePages(packed.data, packed.nbytes);
  }
  if (!BorrowStTensorBytes(r.scale, ws, vt::DType::kI8,
                           {out_dim, in_dim / 16})) {
    r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
    VT_CHECK(ws.nbytes == r.scale.bytes.size(),
             "qwen3_5 dense: scale byte-size mismatch for " + proj);
    std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
    MaybeReleaseSourcePages(ws.data, ws.nbytes);
  }
  return r;
}

// --- lm_head dtype dispatch (issue #164) --------------------------------------
// The 27B NVFP4 publishers do NOT agree on the OUTPUT HEAD, and the head is not
// a compressed-tensors Linear, so none of the scheme probes above cover it:
//
//   BF16     `lm_head.weight` [V,H]                                (transpose)
//   F8_E4M3  `lm_head.weight` [V,H] + `.weight_scale` [V,1] or []  (per-row/scalar)
//   U8       `lm_head.weight` [V,H/2] + `.weight_scale` F8 [V,H/16]
//                                     + `.weight_scale_2` f32       (ModelOpt NVFP4)
//
// This loader was written against `unsloth/Qwen3.6-27B-NVFP4` @890bdef7, which
// ships a BF16 head — the snapshot every recorded 27B-NVFP4 benchmark ran on, so
// those numbers are unaffected by this change. @ccdaab7e later re-quantized the
// head to FP8 with a PER-OUTPUT-CHANNEL scale, and nvidia/Qwen3.6-27B-NVFP4 ships
// a ModelOpt NVFP4 head; both hit the old unconditional BF16 assert.
//
// BF16 and FP8 heads land on the SAME bf16 [in, out] Matmul-B operand the logits
// GEMM already consumes, so a BF16 head stays byte-exact. The NVFP4 form no
// longer reaches this function: LoadDenseLmHead routes it to the PACKED
// `lm_head_fp4` (PERF-27B-LMHEAD-FP4, issue #213), and the U8 branch below
// survives ONLY as the VT_LMHEAD_FP4=0 in-binary rollback.
//
// ModelOpt vs compressed-tensors global-scale convention: CT stores the value as
// a DIVISOR and `DequantCtNvfp4WeightToF32` reciprocates it internally, whereas
// ModelOpt's `weight_scale_2` IS the scale (qwen3_5_weights.cpp:272 assigns it to
// `scale2` directly). Passing `1/weight_scale_2` as the "disk divisor" makes the
// shared CT dequant compute the ModelOpt scale exactly.
}  // namespace

OwnedTensor LoadLmHeadAnyDtype(const TensorResolver& get, const TensorExists& has,
                               const std::string& name) {
  const StTensor& w = get(name);
  VT_CHECK(w.shape.size() == 2, "qwen3_5 dense: expected 2-D weight for " + name);

  if (w.dtype == "BF16") {
    return LoadBf16Transposed(get, name);  // unchanged byte-for-byte
  }

  if (w.dtype == "F8_E4M3") {
    const int64_t out_dim = w.shape[0];
    const int64_t in_dim = w.shape[1];
    // Per-output-channel [V,1] (unsloth @ccdaab7e) or a single per-tensor scalar.
    // Stored BF16 there, F32 elsewhere; normalize both to f32 rows.
    std::vector<float> row_scale(static_cast<size_t>(out_dim), 1.0F);
    VT_CHECK(has(name + "_scale"),
             "qwen3_5 dense: FP8 " + name + " requires " + name + "_scale");
    const StTensor& sc = get(name + "_scale");
    const int64_t n_scale =
        static_cast<int64_t>(sc.nbytes) / (sc.dtype == "BF16" ? 2 : 4);
    VT_CHECK(n_scale == out_dim || n_scale == 1,
             "qwen3_5 dense: " + name + "_scale must be per-tensor or [out,1]");
    for (int64_t r = 0; r < out_dim; ++r) {
      const int64_t i = (n_scale == 1) ? 0 : r;
      if (sc.dtype == "BF16") {
        uint16_t h = 0;
        std::memcpy(&h, static_cast<const uint8_t*>(sc.data) + i * 2, 2);
        const uint32_t bits = static_cast<uint32_t>(h) << 16;
        std::memcpy(&row_scale[static_cast<size_t>(r)], &bits, sizeof(float));
      } else {
        std::memcpy(&row_scale[static_cast<size_t>(r)],
                    static_cast<const uint8_t*>(sc.data) + i * 4, sizeof(float));
      }
    }
    std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
    for (int64_t r = 0; r < out_dim; ++r) {
      DequantFp8ToBf16(static_cast<const uint8_t*>(w.data) + r * in_dim,
                       row_scale[static_cast<size_t>(r)], in_dim,
                       dq.data() + static_cast<size_t>(r) * in_dim);
    }
    MaybeReleaseSourcePages(w.data, w.nbytes);
    OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
    dense_loaders::TransposeBf16(dq.data(), out_dim, in_dim,
                                 reinterpret_cast<uint16_t*>(o.bytes.data()));
    return o;
  }

  if (w.dtype == "U8") {
    const int64_t out_dim = w.shape[0];
    const int64_t in_dim = w.shape[1] * 2;
    VT_CHECK(in_dim % 16 == 0,
             "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + name);
    const StTensor& ws = get(name + "_scale");
    VT_CHECK(ws.dtype == "F8_E4M3",
             "qwen3_5 dense: expected F8_E4M3 " + name + "_scale");
    // ModelOpt spells the global scale `weight_scale_2`; compressed-tensors
    // spells it `weight_global_scale` and stores the reciprocal.
    float disk_divisor = 0.0F;
    if (has(name + "_scale_2")) {
      const float ws2 = ReadF32Scalar(get, name + "_scale_2");
      VT_CHECK(ws2 != 0.0F, "qwen3_5 dense: zero " + name + "_scale_2");
      disk_divisor = 1.0F / ws2;  // ModelOpt scale -> CT divisor convention
    } else {
      VT_CHECK(has(name + "_global_scale"),
               "qwen3_5 dense: NVFP4 " + name + " requires " + name +
                   "_scale_2 (ModelOpt) or " + name + "_global_scale (CT)");
      disk_divisor = ReadF32Scalar(get, name + "_global_scale");
      VT_CHECK(disk_divisor != 0.0F,
               "qwen3_5 dense: zero " + name + "_global_scale (divisor)");
    }
    std::vector<float> f32(static_cast<size_t>(out_dim) * in_dim);
    DequantCtNvfp4WeightToF32(static_cast<const uint8_t*>(w.data),
                              static_cast<const uint8_t*>(ws.data), disk_divisor,
                              out_dim, in_dim, f32.data());
    MaybeReleaseSourcePages(w.data, w.nbytes);
    std::vector<uint16_t> dq(static_cast<size_t>(out_dim) * in_dim);
    for (size_t i = 0; i < f32.size(); ++i) {
      uint32_t bits = 0;
      std::memcpy(&bits, &f32[i], sizeof(bits));
      // round-to-nearest-even f32 -> bf16, matching DequantFp8ToBf16.
      const uint32_t lsb = (bits >> 16) & 1U;
      bits += 0x7FFFU + lsb;
      dq[i] = static_cast<uint16_t>(bits >> 16);
    }
    OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
    dense_loaders::TransposeBf16(dq.data(), out_dim, in_dim,
                                 reinterpret_cast<uint16_t*>(o.bytes.data()));
    return o;
  }

  VT_CHECK(false, "qwen3_5 dense: unsupported dtype '" + w.dtype + "' for " +
                      name + "; supported: BF16, F8_E4M3 (+_scale), "
                      "U8 NVFP4 (+_scale and _scale_2/_global_scale)");
  return OwnedTensor{};
}

// compressed-tensors and ModelOpt both ship NVFP4, with different names AND a
// different global-scale convention:
//
//   compressed-tensors  <proj>.weight_packed U8 + .weight_scale F8
//                       + .weight_global_scale F32 (a DIVISOR)
//   ModelOpt            <proj>.weight        U8 + .weight_scale F8
//                       + .weight_scale_2    F32 (the SCALE itself)
//
// nvidia/Qwen3.6-27B-NVFP4 is ModelOpt, so every `has(<proj>.weight_packed)`
// probe missed it and the whole tower fell through to the BF16 path and died at
// the first U8 tensor. LoadCtNvfp4Raw already reciprocates internally, so the
// ModelOpt scale is passed through as 1/weight_scale_2 to land on the same math
// (identical to the lm_head conversion in LoadLmHeadAnyDtype).
//
// MODEL-DFLASH2-NVFP4 (#2758): EXPORTED, and no longer file-local. The DFlash2
// draft loader asks the SAME question about the SAME spelling for its own
// weights, and a second copy of this probe is the "two descriptions of one
// rule" failure AGENTS.md `## Changing the rules or a checker` names. It is the
// argument `DenseLmHeadTakesNvfp4` already makes one screen down, for the same
// consumer. Declared in `qwen3_5_dense.h`; the bodies are unchanged.
bool IsNvfp4Projection(const std::function<bool(const std::string&)>& has,
                       const std::string& proj) {
  return has(proj + ".weight_packed") || has(proj + ".weight_scale_2");
}

Nvfp4Weight LoadNvfp4AnyNaming(const TensorResolver& get,
                               const std::function<bool(const std::string&)>& has,
                               const std::string& proj) {
  if (has(proj + ".weight_packed")) return LoadCtNvfp4Raw(get, proj);

  const StTensor& packed = get(proj + ".weight");
  VT_CHECK(packed.dtype == "U8",
           "qwen3_5 dense: expected U8 ModelOpt weight for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "qwen3_5 dense: expected 2-D ModelOpt weight for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3",
           "qwen3_5 dense: expected F8_E4M3 weight_scale for " + proj);
  const float ws2 = ReadF32Scalar(get, proj + ".weight_scale_2");
  VT_CHECK(ws2 != 0.0F, "qwen3_5 dense: zero weight_scale_2 for " + proj);

  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.scale2 = ws2;                       // ModelOpt stores the scale directly
  r.weight_global_scale_inv = 1.0F / ws2;  // the CT-convention divisor
  // W4A16 unless the checkpoint also carries an activation scale. ModelOpt
  // spells it `input_scale`; leaving alpha at 0 keeps IsTrueW4A4() false so the
  // weight routes to the W4A16 dispatcher, matching vLLM's use_a16 branch.
  // A/B(VT_MODELOPT_W4A4=1): default W4A16. ModelOpt ships `input_scale` on every
  // projection, but consuming it flips IsTrueW4A4() and routes to the
  // fp4-activation GEMM; leaving alpha at 0 keeps the weight-only dispatcher.
  const bool w4a4_opt_in = ModelOptW4A4OptIn();
  if (w4a4_opt_in && has(proj + ".input_scale")) {
    const float is = ReadF32Scalar(get, proj + ".input_scale");
    if (is != 0.0F) {
      r.input_global_scale_inv = is;
      r.alpha = r.scale2 * (1.0F / is);
    }
  }
  // ENG-LOAD-DIRECT-UPLOAD (issue #150), widened to the ModelOpt SPELLING by
  // LOAD-MODELOPT-NVFP4-BORROW (issue #1647). Both payloads are verbatim, byte
  // for byte identical to what `LoadCtNvfp4Raw` borrows twenty lines up: the
  // same E2M1 nibbles and the same fp8-e4m3 group-16 scale, differing only in
  // the tensor NAMES and in a scalar global-scale convention. Nothing about the
  // ranges made this arm ineligible; it was simply never given the attempt.
  //
  // WHAT IT COST. `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121` declares
  // `quant_method: "modelopt"`, so all 193 of its NVFP4 modules take this
  // branch. Copying put ~21 GiB into ANONYMOUS heap where upstream holds
  // reclaimable page cache (`weight_utils.py:969-974,1247` at the pin
  // `5559679229bc961848b121ccdeaa8fa5d79bec98` streams a file-backed mmap into a
  // device parameter allocated before any file is opened,
  // `base_loader.py:52-58`). The expensive half is downstream:
  // `AdoptDeviceBytesAsHost` is gated on
  // `w.mmap_src != nullptr && w.bytes.borrowed()`, so `ResidentNvfp4`'s
  // post-upload adoption -- which releases the consumed source pages and, on a
  // host-addressable device, re-points `bytes` at the device allocation -- was a
  // SILENT NO-OP for every weight loaded here. On GB10, where a device
  // allocation IS host RAM, that is a packed host copy plus a packed device copy
  // out of one 119 GiB pool, and the box reboots.
  //
  // The fallback below is unchanged and still runs whenever the borrow declines,
  // which `BorrowStTensorBytes` does for a synthetic tensor, a size that does not
  // line up, or `VT_LOAD_DIRECT_UPLOAD=0`. The lever can only cost residency,
  // never correctness.
  if (!BorrowStTensorBytes(r.packed, packed, vt::DType::kI8,
                           {out_dim, in_dim / 2})) {
    r.packed = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
    VT_CHECK(packed.nbytes == r.packed.bytes.size(),
             "qwen3_5 dense: ModelOpt packed byte-size mismatch for " + proj);
    std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
    MaybeReleaseSourcePages(packed.data, packed.nbytes);
  }
  if (!BorrowStTensorBytes(r.scale, ws, vt::DType::kI8,
                           {out_dim, in_dim / 16})) {
    r.scale = MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
    VT_CHECK(ws.nbytes == r.scale.bytes.size(),
             "qwen3_5 dense: ModelOpt scale byte-size mismatch for " + proj);
    std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
    MaybeReleaseSourcePages(ws.data, ws.nbytes);
  }
  return r;
}

namespace {

// MODEL-FP8-BLOCK-WEIGHT (#1189 M3), spec
// `.agents/specs/model-fp8-block-weight.md`. Does this projection take the
// block-wise (fine-grained 128x128) FP8 arm?
//
// READ THE CONFIG AND THE TENSORS, AND REFUSE WHEN THEY DISAGREE. A dtype probe
// alone answers "which arm" and can never answer "do these two sources agree",
// and the disagreement is where a silent wrong-scale bug lives: #1166 measured a
// `[96, 40]` block grid passing a per-tensor scale reader's byte floor and being
// applied to the whole `[N, K]` weight, stopped only by the tensor NAME. The
// checkpoint's `modules_to_not_convert` is the other half -- 882 entries on
// `Qwen/Qwen3.8-27B-FP8` @`017b9c7a`, 636 of them outside the vision tower --
// which a probe reproduces only by accident (#1614; the "~400" this line
// carried was wrong by more than 2.2x however the list is sliced).
//
// Six combinations, four of them refused by name; the table is in the spec.
bool IsFp8BlockProjection(const TensorExists& has, const std::string& proj,
                          const std::string& weight_dtype,
                          const Fp8BlockQuantConfig& block) {
  const bool has_scale_inv = has(proj + ".weight_scale_inv");
  const bool excluded = block.ExcludesModule(proj);

  if (!block.block_quant) {
    VT_CHECK(!has_scale_inv,
             "qwen3_5 dense: '" + proj +
                 ".weight_scale_inv' is present, which is the block-wise "
                 "(fine-grained) FP8 scale, but the checkpoint's "
                 "quantization_config declares no weight_block_size. The "
                 "tensors and the config disagree and there is no block "
                 "geometry to read the scale with; refusing rather than "
                 "guessing 128x128");
    return false;
  }

  if (excluded) {
    VT_CHECK(!has_scale_inv,
             "qwen3_5 dense: '" + proj +
                 "' is listed in quantization_config.modules_to_not_convert, "
                 "so it must be unquantized, yet it ships '" + proj +
                 ".weight_scale_inv'. The tensors and the config disagree; "
                 "refusing rather than picking one of them");
    return false;
  }

  if (weight_dtype == "F8_E4M3") {
    VT_CHECK(has_scale_inv,
             "qwen3_5 dense: the checkpoint declares "
             "quantization_config.weight_block_size and '" +
                 proj +
                 ".weight' is F8_E4M3, but '" + proj +
                 ".weight_scale_inv' is missing. Block-wise (fine-grained) FP8 "
                 "stores its scale under that name and only that name "
                 "(vllm fp8.py:378-379,511), so this projection cannot be "
                 "dequantized. Either the module belongs in "
                 "modules_to_not_convert or the shard is incomplete");
  }
  if (!has_scale_inv) return false;

  // An `input_scale` cannot coexist with a dynamic activation scheme: upstream
  // registers one only when `act_q_static` (`fp8.py:381-384`), which block quant
  // asserts against outright (`fp8.py:367`). `Qwen/Qwen3.8-27B-FP8` ships zero.
  VT_CHECK(!has(proj + ".input_scale"),
           "qwen3_5 dense: '" + proj +
               ".input_scale' is present beside a block-wise (fine-grained) "
               "FP8 weight whose quantization_config declares "
               "activation_scheme \"dynamic\". A dynamic scheme quantizes "
               "activations at run time and has no static input scale; the "
               "tensors and the config disagree");
  return true;
}

// Shared unquantized GDN parameters. The scoped XPU policy loads BA, Conv
// and gated norm as FP16, independently of their stored remainder dtype.
// A_log and dt_bias are retained as FP32 computation inputs; recurrent state
// precision is resolved separately. Other schemes keep their existing reader.
void LoadGdnSmallTensors(const TensorResolver& get, const std::string& la,
                         bool allow_f16, GdnLayerWeights& g, bool fp16 = false) {
  // QUALIFIED: `qwen3_5_dense.h` declares a two-argument `vllm::`
  // LoadMergedBf16RawNK forwarder for the focused loader contract, and
  // unqualified lookup finds that one first.
  g.in_proj_ba = allow_f16 && fp16
      ? dense_loaders::LoadMergedF16RawNK(get, {la + "in_proj_b.weight", la + "in_proj_a.weight"})
      : dense_loaders::LoadMergedBf16RawNK(get, {la + "in_proj_b.weight", la + "in_proj_a.weight"},
                                        /*tp=*/nullptr, allow_f16);
  // conv1d.weight ships [conv_dim,1,K]; collapse the singleton to [conv_dim,K].
  const StTensor& conv = get(la + "conv1d.weight");
  VT_CHECK(conv.shape.size() == 3 && conv.shape[1] == 1,
           "qwen3_5 dense: unexpected conv1d shape");
  g.conv1d_weight = allow_f16 && fp16
      ? dense_loaders::LoadF16Direct(get, la + "conv1d.weight", {conv.shape[0], conv.shape[2]})
      : LoadBf16Direct(get, la + "conv1d.weight", {conv.shape[0], conv.shape[2]});
  g.a_log = LoadToF32(get, la + "A_log");
  g.dt_bias = LoadToF32(get, la + "dt_bias");
  g.norm_weight = LoadModelVectorForScheme(get, allow_f16, la + "norm.weight", fp16);
}

GdnLayerWeights LoadGdnDense(const TensorResolver& get, const TensorExists& has,
                             const std::string& base,
                             const Fp8BlockQuantConfig& block, bool fp16 = false) {
  const std::string la = base + "linear_attn.";
  GdnLayerWeights g;
  // MODEL-QWEN35-GDN-EXL3 (#2495 item 4). FIRST and EXCLUSIVE, for the reason
  // `LoadAttnDense`'s EXL3 rung is: an EXL3 projection ships NO `.weight` at
  // all, so every probe below -- `IsNvfp4Projection`, the
  // `get(name + ".weight").dtype` read, the block-FP8 cross-check -- would
  // either mis-route it or die asking for a tensor the checkpoint correctly
  // does not ship. Selection is by upstream's own storage predicate
  // (`Linear.is_exl3_storage`, `modules/linear.py:385-389`) rather than by a
  // config flag, so a checkpoint that quantizes only some layers loads each one
  // the way it is actually stored.
  //
  // THIS REPLACES A REFUSAL. Until this rung existed the three projections were
  // refused by name, which was the honest behaviour then: the alternative was
  // falling through to `get(la + "in_proj_qkv.weight")` and dying on "tensor
  // not found", a sentence about a checkpoint that is complete.
  //
  // THREE projections and no merged qkvz owner, because that is what the
  // artifact ships. See `GdnLayerWeights::in_proj_qkv_exl3` for why the trellis
  // pair cannot be N-concatenated the way the bf16 pair is.
  if (dense_loaders::IsExl3Projection(has, la + "in_proj_qkv")) {
    g.in_proj_qkv_exl3 = dense_loaders::LoadExl3(get, has, la + "in_proj_qkv");
    g.in_proj_z_exl3 = dense_loaders::LoadExl3(get, has, la + "in_proj_z");
    g.out_proj_exl3 = dense_loaders::LoadExl3(get, has, la + "out_proj");
    // GEOMETRY, CHECKED rather than trusted. A trellis is self-consistent at
    // more than one reading of its own shape, so a transposed or mismatched
    // projection is invisible until the numbers are wrong. `conv1d.weight`'s
    // leading dim is the checkpoint's own statement of `conv_dim`, and
    // `out_proj`'s K is its statement of `value_dim`; both are read from the
    // artifact rather than from the config, so a config that disagrees with the
    // tensors fails here instead of silently picking one of them.
    const int64_t hidden = g.in_proj_qkv_exl3.InFeatures();
    VT_CHECK(g.in_proj_z_exl3.InFeatures() == hidden &&
                 g.out_proj_exl3.OutFeatures() == hidden,
             "qwen3_5 dense: EXL3 GDN projections disagree about the hidden "
             "size at " + la);
    VT_CHECK(g.out_proj_exl3.InFeatures() == g.in_proj_z_exl3.OutFeatures(),
             "qwen3_5 dense: EXL3 GDN out_proj K and in_proj_z N disagree "
             "about value_dim at " + la);
    VT_CHECK(get(la + "conv1d.weight").shape[0] ==
                 g.in_proj_qkv_exl3.OutFeatures(),
             "qwen3_5 dense: EXL3 GDN in_proj_qkv N and conv1d's channel count "
             "disagree about conv_dim at " + la);
    LoadGdnSmallTensors(get, la, /*allow_f16=*/true, g, fp16);
    return g;
  }
  // in_proj_{qkv,z,a,b}: bf16 (ignore list, notes §3.6). Kept raw [N,K]
  // (nk=true -> vt::MatmulBT TN fast path). Mirror vLLM's TWO physical
  // MergedColumnParallelLinear owners (qwen3_5.py:203-210 stacked mapping +
  // qwen_gdn_linear_attn.py:481-496): in_proj_qkvz in exact [q,k,v,z] row
  // order (the checkpoint's in_proj_qkv already stacks q|k|v; z appended) and
  // in_proj_ba in exact [b,a] row order. The rollback paths take non-owning
  // row slices of these owners, so the split fields deliberately stay empty.
  // FP8 checkpoints (nvidia/Qwen3.6-27B-NVFP4 is `modelopt_mixed`) keep the GDN
  // shards NATIVE. Merging forces one dtype, and dequantizing fp8 -> bf16 DOUBLES
  // this tower's resident bytes (6.72 -> 13.44 GiB measured), which both slows
  // decode and starves the KV pool. `ProjectGdnQkvz` already carries the
  // separate-fp8 arm the 35B runs and selects it when the merged owner is empty.
  const StTensor& qkv_probe = get(la + "in_proj_qkv.weight");
  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): the block-wise rung goes BEFORE the
  // per-tensor one at every site that probes `F8_E4M3`, because a block-wise
  // weight IS `F8_E4M3` and fell into the per-tensor arm (#1166).
  if (IsFp8BlockProjection(has, la + "in_proj_qkv", qkv_probe.dtype, block)) {
    g.in_proj_qkv_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, la + "in_proj_qkv", block.block_n, block.block_k);
    g.in_proj_z_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, la + "in_proj_z", block.block_n, block.block_k);
  } else if (qkv_probe.dtype == "F8_E4M3") {
    g.in_proj_qkv_fp8 = LoadFp8RawShared(get, la + "in_proj_qkv");
    g.in_proj_z_fp8 = LoadFp8RawShared(get, la + "in_proj_z");
  } else {
    g.in_proj_qkvz = LoadMergedBf16RawNK(
        get, {la + "in_proj_qkv.weight", la + "in_proj_z.weight"});
  }
  // NVFP4 checkpoints use compressed tensors; ordinary checkpoints use raw
  // torch-Linear BF16 [N,K].
  if (IsNvfp4Projection(has, la + "out_proj")) {
    g.out_proj_fp4 = LoadNvfp4AnyNaming(get, has, la + "out_proj");
  } else if (IsFp8BlockProjection(has, la + "out_proj",
                                  get(la + "out_proj.weight").dtype, block)) {
    g.out_proj_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, la + "out_proj", block.block_n, block.block_k);
  } else if (get(la + "out_proj.weight").dtype == "F8_E4M3") {
    // Same rule as the in_proj shards above, and for the same measured reason:
    // the bf16 arm dequantizes this tower and then runs it as a cuBLAS `gemvx`,
    // which the decode profile shows costing far more than the bytes justify.
    // `ProjectGdnOut` already selects `out_proj_fp8` when it is populated.
    g.out_proj_fp8 = LoadFp8RawShared(get, la + "out_proj");
  } else {
    g.out_proj = LoadBf16RawNK(get, la + "out_proj.weight");
  }
  LoadGdnSmallTensors(get, la, /*allow_f16=*/false, g);
  return g;
}

FullAttnLayerWeights LoadAttnDense(const TensorResolver& get,
                                   const TensorExists& has,
                                   const std::string& base,
                                   const Fp8BlockQuantConfig& block, bool fp16 = false) {
  const std::string sa = base + "self_attn.";
  FullAttnLayerWeights a;
  // MODEL-QWEN35-EXL3 (#2495 item 3). FIRST and exclusive, and selected by
  // upstream's own storage predicate rather than by a config flag, so a
  // checkpoint that quantizes only some layers loads each one the way it is
  // actually stored (`Linear.is_exl3_storage`, `modules/linear.py:385-389`).
  // An EXL3 projection has NO `.weight` at all, so every probe below --
  // `IsNvfp4Projection`, the `get(name + ".weight").dtype` read, the block
  // cross-check -- would either mis-route it or die asking for a tensor the
  // checkpoint correctly does not ship. Mirrors `LoadLlamaLayer`
  // (`llama_weights.cpp:73-85`), which is the working exemplar.
  if (dense_loaders::IsExl3Projection(has, sa + "q_proj")) {
    a.q_proj_exl3 = dense_loaders::LoadExl3(get, has, sa + "q_proj");
    a.k_proj_exl3 = dense_loaders::LoadExl3(get, has, sa + "k_proj");
    a.v_proj_exl3 = dense_loaders::LoadExl3(get, has, sa + "v_proj");
    a.o_proj_exl3 = dense_loaders::LoadExl3(get, has, sa + "o_proj");
    // The per-head q/k RMSNorm weights are NOT quantized; they ship beside the
    // trellis at the checkpoint's unquantized dtype.
    a.q_norm = LoadModelVectorForScheme(get, /*exl3=*/true, sa + "q_norm.weight", fp16);
    a.k_norm = LoadModelVectorForScheme(get, /*exl3=*/true, sa + "k_norm.weight", fp16);
    return a;
  }
  // Three forms, not two. `modelopt_mixed` checkpoints quantize this tower to
  // FP8 W8A8 while leaving the MLP NVFP4, and a projection that matches neither
  // the NVFP4 probe nor an FP8 dtype is genuinely BF16. Without the middle
  // branch an FP8 tower fell through to `LoadBf16RawNK`, which dequantizes it:
  // 1.562 GiB of FP8 became 3.12 GiB of BF16 re-read every decode step and
  // executed as cuBLAS `gemvx`. The `*_fp8` slots and their `MatmulFp8Cutlass*`
  // consumers already exist and are what the 35B runs.
  // FOUR forms. MODEL-FP8-BLOCK-WEIGHT (#1189 M3) inserts the block-wise rung
  // BEFORE the per-tensor one, because a block-wise weight is also `F8_E4M3`
  // and therefore entered the per-tensor arm, which then asked for a
  // `weight_scale` the checkpoint spells `weight_scale_inv` (#1166).
  const auto load_projection = [&](const std::string& name, Nvfp4Weight& fp4,
                                   Fp8BlockWeight& fp8_block, Fp8Weight& fp8,
                                   OwnedTensor& plain) {
    if (IsNvfp4Projection(has, name)) {
      fp4 = LoadNvfp4AnyNaming(get, has, name);
      return;
    }
    const std::string dtype = get(name + ".weight").dtype;
    if (IsFp8BlockProjection(has, name, dtype, block)) {
      fp8_block = dense_loaders::LoadFp8BlockRaw(get, name, block.block_n,
                                                 block.block_k);
    } else if (dtype == "F8_E4M3") {
      fp8 = LoadFp8RawShared(get, name);
    } else {
      plain = LoadBf16RawNK(get, name + ".weight");
    }
  };
  load_projection(sa + "q_proj", a.q_proj_fp4, a.q_proj_fp8_block, a.q_proj_fp8,
                  a.q_proj);
  load_projection(sa + "k_proj", a.k_proj_fp4, a.k_proj_fp8_block, a.k_proj_fp8,
                  a.k_proj);
  load_projection(sa + "v_proj", a.v_proj_fp4, a.v_proj_fp8_block, a.v_proj_fp8,
                  a.v_proj);
  load_projection(sa + "o_proj", a.o_proj_fp4, a.o_proj_fp8_block, a.o_proj_fp8,
                  a.o_proj);
  a.q_norm = LoadModelBf16Direct(get, sa + "q_norm.weight");
  a.k_norm = LoadModelBf16Direct(get, sa + "k_norm.weight");
  return a;
}

// Dense SwiGLU MLP: gate/up/down all W4A4-quantized -> fp4-resident (§5 6a).
DenseMlpWeights LoadDenseMlp(const TensorResolver& get, const TensorExists& has,
                             const std::string& base,
                             const Fp8BlockQuantConfig& block) {
  const std::string mlp = base + "mlp.";
  DenseMlpWeights m;
  // MODEL-QWEN35-EXL3 (#2495 item 3). FIRST and exclusive, for the same reason
  // the attention arm is: an EXL3 projection ships no `.weight`, so every probe
  // below reads a tensor that is not there. Keep the checkpoint shards here;
  // the scoped FP16 method lazily builds the model-owned packed gate/up group
  // before its first GPU use.
  if (dense_loaders::IsExl3Projection(has, mlp + "gate_proj")) {
    m.gate_proj_exl3 = dense_loaders::LoadExl3(get, has, mlp + "gate_proj");
    m.up_proj_exl3 = dense_loaders::LoadExl3(get, has, mlp + "up_proj");
    m.down_proj_exl3 = dense_loaders::LoadExl3(get, has, mlp + "down_proj");
    return m;
  }
  if (IsNvfp4Projection(has, mlp + "gate_proj")) {
    m.gate_proj_fp4 = LoadNvfp4AnyNaming(get, has, mlp + "gate_proj");
    m.up_proj_fp4 = LoadNvfp4AnyNaming(get, has, mlp + "up_proj");
    m.down_proj_fp4 = LoadNvfp4AnyNaming(get, has, mlp + "down_proj");
  } else if (IsFp8BlockProjection(has, mlp + "gate_proj",
                                  get(mlp + "gate_proj.weight").dtype, block)) {
    // MODEL-FP8-BLOCK-WEIGHT (#1189 M3). This block had NO fp8 rung at all, so
    // a block-wise MLP fell through to `LoadMergedBf16RawNK` and died on
    // "expected BF16". Loaded UNMERGED: block scales concatenate losslessly
    // along N, so merging gate+up is simpler here than in the per-tensor case
    // and #1189 M6 owns it rather than this row guessing at the layout.
    m.gate_proj_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, mlp + "gate_proj", block.block_n, block.block_k);
    m.up_proj_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, mlp + "up_proj", block.block_n, block.block_k);
    m.down_proj_fp8_block = dense_loaders::LoadFp8BlockRaw(
        get, mlp + "down_proj", block.block_n, block.block_k);
  } else {
    m.gate_up_proj = dense_loaders::LoadMergedBf16RawNK(
        get, {mlp + "gate_proj.weight", mlp + "up_proj.weight"});
    m.down_proj = LoadBf16RawNK(get, mlp + "down_proj.weight");
  }
  return m;
}

}  // namespace

DenseGptq4LayerWeights LoadQwen3_5DenseGptq4Projections(
    const TensorResolver& get, const TensorExists& has,
    const HfConfig& config, int64_t layer_idx,
    const std::string& backbone_prefix) {
  dense_loaders::CheckProbeCanAnswerNo(
      has, "LoadQwen3_5DenseGptq4Projections");
  VT_CHECK(layer_idx >= 0 &&
               layer_idx < static_cast<int64_t>(config.layer_types.size()),
           "gptq4: layer index is outside the text layer configuration");
  const int64_t hidden = config.hidden_size;
  const int64_t intermediate = config.intermediate_size;
  const int64_t value_dim =
      config.linear_num_value_heads * config.linear_value_head_dim;
  VT_CHECK(hidden > 0 && intermediate > 0 &&
               config.num_attention_heads > 0 &&
               config.num_key_value_heads > 0 && config.head_dim > 0 &&
               config.linear_num_key_heads > 0 &&
               config.linear_key_head_dim > 0 &&
               config.linear_num_value_heads > 0 &&
               config.linear_value_head_dim > 0 &&
               hidden <= (1 << 20) && intermediate <= (1 << 20) &&
               config.num_attention_heads <= (1 << 20) &&
               config.num_key_value_heads <= (1 << 20) &&
               config.head_dim <= (1 << 20) &&
               config.linear_num_key_heads <= (1 << 20) &&
               config.linear_key_head_dim <= (1 << 20) &&
               config.linear_num_value_heads <= (1 << 20) &&
               config.linear_value_head_dim <= (1 << 20),
           "gptq4: incomplete text projection geometry");

  const std::string base = backbone_prefix + "layers." +
                           std::to_string(layer_idx) + ".";
  const auto check = [&has](const std::string& projection) {
    VT_CHECK(has(projection + ".qweight"),
             "gptq4: missing packed projection " + projection);
    VT_CHECK(!has(projection + ".weight") &&
                 !has(projection + ".weight_packed") &&
                 !has(projection + ".trellis"),
             "gptq4: conflicting weight representation for " + projection);
  };
  DenseGptq4LayerWeights result;
  const std::string mlp = base + "mlp.";
  const std::string gate = mlp + "gate_proj";
  const std::string up = mlp + "up_proj";
  const std::string down = mlp + "down_proj";
  for (const auto& projection : {gate, up, down}) check(projection);
  result.mlp_gate_up = LoadMergedGptq4Weight(
      get, {gate, up}, hidden, {intermediate, intermediate});
  result.mlp_down = LoadGptq4Weight(get, down, intermediate, hidden);

  const std::string& layer_type =
      config.layer_types[static_cast<size_t>(layer_idx)];
  if (layer_type == "linear_attention") {
    const std::string la = base + "linear_attn.";
    const std::string qkv = la + "in_proj_qkv";
    const std::string z = la + "in_proj_z";
    const std::string out = la + "out_proj";
    for (const auto& projection : {qkv, z, out}) check(projection);
    const int64_t qkv_dim =
        2 * config.linear_num_key_heads * config.linear_key_head_dim + value_dim;
    result.gdn_qkvz = LoadMergedGptq4Weight(
        get, {qkv, z}, hidden, {qkv_dim, value_dim});
    result.gdn_out = LoadGptq4Weight(get, out, value_dim, hidden);
  } else if (layer_type == "full_attention") {
    const std::string sa = base + "self_attn.";
    const std::string q = sa + "q_proj";
    const std::string k = sa + "k_proj";
    const std::string v = sa + "v_proj";
    const std::string out = sa + "o_proj";
    for (const auto& projection : {q, k, v, out}) check(projection);
    const int64_t q_dim = 2 * config.num_attention_heads * config.head_dim;
    const int64_t kv_dim = config.num_key_value_heads * config.head_dim;
    result.attn_qkv = LoadMergedGptq4Weight(
        get, {q, k, v}, hidden, {q_dim, kv_dim, kv_dim});
    result.attn_out = LoadGptq4Weight(
        get, out, config.num_attention_heads * config.head_dim, hidden);
  } else {
    VT_CHECK(false, "gptq4: unsupported text layer type " + layer_type);
  }
  return result;
}

std::vector<DenseGptq4LayerWeights> LoadQwen3_5DenseGptq4TextProjections(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    vt::Queue* load_queue) {
  VT_CHECK(config.raw.contains("quantization_config") &&
               config.raw["quantization_config"].is_object(),
           "gptq4: missing quantization_config");
  const auto& quant = config.raw["quantization_config"];
  VT_CHECK(quant.value("quant_method", std::string()) == "gptq" &&
               quant.value("format", std::string()) == "gptq" &&
               quant.value("bits", 0) == 4 &&
               quant.value("group_size", 0) == 128 &&
               quant.value("sym", false) &&
               !quant.value("desc_act", true) &&
               quant.value("pack_dtype", std::string()) == "int32",
           "gptq4: unsupported quantization configuration");
  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "gptq4: incomplete text layer inventory");
  VT_CHECK(load_queue == nullptr ||
               load_queue->device.type == vt::DeviceType::kXPU,
           "gptq4: resident text projection loader requires XPU");

  std::unordered_map<std::string, const SafetensorsFile*> where;
  std::vector<std::string> all_names;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      VT_CHECK(where.emplace(name, &shard).second,
               "gptq4: duplicate tensor name " + name);
      all_names.push_back(name);
    }
  }
  const std::string backbone = ResolveQwen3_5BackbonePrefix(all_names);
  const TensorResolver get = [&where](const std::string& name)
      -> const StTensor& {
    const auto it = where.find(name);
    VT_CHECK(it != where.end(), "gptq4: missing tensor " + name);
    return it->second->Get(name);
  };
  const TensorExists has = [&where](const std::string& name) {
    return where.find(name) != where.end();
  };
  size_t expected_matrices = 0;
  std::unordered_set<std::string> expected_layer_names;
  const auto allow_projection = [&expected_layer_names](const std::string& proj) {
    for (const char* suffix : {".qweight", ".scales", ".qzeros", ".g_idx"})
      expected_layer_names.insert(proj + suffix);
  };
  for (size_t layer = 0; layer < config.layer_types.size(); ++layer) {
    const auto& type = config.layer_types[layer];
    VT_CHECK(type == "linear_attention" || type == "full_attention",
             "gptq4: unsupported text layer type " + type);
    expected_matrices += type == "linear_attention" ? 6 : 7;
    const std::string base = backbone + "layers." + std::to_string(layer) + ".";
    expected_layer_names.insert(base + "input_layernorm.weight");
    expected_layer_names.insert(base + "post_attention_layernorm.weight");
    for (const char* proj : {"gate_proj", "up_proj", "down_proj"})
      allow_projection(base + "mlp." + proj);
    if (type == "linear_attention") {
      const std::string la = base + "linear_attn.";
      for (const char* proj : {"in_proj_qkv", "in_proj_z", "out_proj"})
        allow_projection(la + proj);
      for (const char* dense : {"in_proj_b.weight", "in_proj_a.weight",
                                "conv1d.weight", "A_log", "dt_bias",
                                "norm.weight"})
        expected_layer_names.insert(la + dense);
    } else {
      const std::string sa = base + "self_attn.";
      for (const char* proj : {"q_proj", "k_proj", "v_proj", "o_proj"})
        allow_projection(sa + proj);
      expected_layer_names.insert(sa + "q_norm.weight");
      expected_layer_names.insert(sa + "k_norm.weight");
    }
  }
  size_t found_matrices = 0;
  const std::string layer_prefix = backbone + "layers.";
  for (const auto& name : all_names) {
    if (name.rfind(layer_prefix, 0) == 0) {
      VT_CHECK(expected_layer_names.count(name) != 0,
               "gptq4: unexpected text layer tensor " + name);
      if (name.size() >= 8 &&
          name.compare(name.size() - 8, 8, ".qweight") == 0)
        ++found_matrices;
    }
  }
  VT_CHECK(found_matrices == expected_matrices,
           "gptq4: text qweight inventory does not match layer topology");

  std::vector<DenseGptq4LayerWeights> layers;
  layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t layer = 0; layer < config.num_hidden_layers; ++layer) {
    layers.push_back(LoadQwen3_5DenseGptq4Projections(
        get, has, config, layer, backbone));
    if (load_queue != nullptr) layers.back().PrepareResident(*load_queue);
  }
  return layers;
}

DenseExecutionPrecision ResolveQwen3_5DensePrecision(
    const HfConfig& config, bool gptq4_checkpoint, bool exl3_checkpoint) {
  DenseExecutionPrecision policy;
  VT_CHECK(!(gptq4_checkpoint && exl3_checkpoint), "dense: conflicting GPTQ/EXL3 precision policy");
  if (!gptq4_checkpoint && !exl3_checkpoint) return policy;
  VT_CHECK(!gptq4_checkpoint || config.torch_dtype == "float16",
           "gptq4: expected declared text dtype float16, got '" +
               config.torch_dtype + "' from " + config.dtype_source);
  VT_CHECK(config.mamba_ssm_dtype == "float32" ||
               config.mamba_ssm_dtype == "float",
           "dense FP16: expected FP32 recurrent state from mamba_ssm_dtype");
  policy.activation = vt::DType::kF16;
  policy.dense_weight = vt::DType::kF16;
  policy.kv_auto = vt::DType::kF16;
  policy.gdn_conv_state = vt::DType::kF16;
  policy.gdn_recurrent_state = vt::DType::kF32;
  return policy;
}

namespace {

Qwen3_5DenseWeights LoadGptq4DenseText(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config,
    const TensorResolver& get, const TensorExists& has,
    const std::string& backbone, vt::Queue* load_queue) {
  VT_CHECK(config.hidden_size > 0 && config.vocab_size > 0 &&
               config.linear_num_key_heads > 0 &&
               config.linear_key_head_dim > 0 &&
               config.linear_num_value_heads > 0 &&
               config.linear_value_head_dim > 0 &&
               config.linear_conv_kernel_dim > 0 && config.head_dim > 0,
           "gptq4: incomplete unquantized text geometry");
  VT_CHECK(has("lm_head.weight") && !has("lm_head.qweight") &&
               !has(backbone + "embed_tokens.qweight"),
           "gptq4: expected separate unquantized FP16 head and embedding");
  for (const auto& shard : shards) {
    for (const auto& name : shard.Names()) {
      const bool head = name.rfind("lm_head.", 0) == 0;
      const bool embedding =
          name.rfind(backbone + "embed_tokens.", 0) == 0;
      const bool final_norm = name.rfind(backbone + "norm.", 0) == 0;
      if (head || embedding || final_norm) {
        const std::string expected = head ? "lm_head.weight"
            : embedding ? backbone + "embed_tokens.weight"
                        : backbone + "norm.weight";
        VT_CHECK(name == expected,
                 "gptq4: unexpected unquantized text tensor " + name);
      }
    }
  }
  Qwen3_5DenseWeights result;
  result.gptq4_checkpoint = true;
  result.precision = ResolveQwen3_5DensePrecision(config, true);
  auto packed = LoadQwen3_5DenseGptq4TextProjections(
      shards, config,
      load_queue != nullptr && load_queue->device.type == vt::DeviceType::kXPU
          ? load_queue : nullptr);
  const int64_t hidden = config.hidden_size;
  const int64_t value_dim = config.linear_num_value_heads *
                            config.linear_value_head_dim;
  const int64_t conv_dim = 2 * config.linear_num_key_heads *
                               config.linear_key_head_dim + value_dim;
  result.embed_tokens = LoadGptqF16(
      get, backbone + "embed_tokens.weight", {config.vocab_size, hidden});
  result.final_norm = LoadGptqF16(get, backbone + "norm.weight", {hidden});
  result.lm_head = LoadGptqF16(
      get, "lm_head.weight", {config.vocab_size, hidden}, true);
  result.layers.reserve(packed.size());
  for (size_t i = 0; i < packed.size(); ++i) {
    const std::string base = backbone + "layers." + std::to_string(i) + ".";
    Qwen3_5DenseLayerWeights layer;
    layer.gptq4 = std::move(packed[i]);
    layer.input_layernorm = LoadGptqF16(
        get, base + "input_layernorm.weight", {hidden});
    layer.post_attention_layernorm = LoadGptqF16(
        get, base + "post_attention_layernorm.weight", {hidden});
    layer.is_linear_attention = config.layer_types[i] == "linear_attention";
    if (layer.is_linear_attention) {
      const std::string la = base + "linear_attn.";
      layer.gdn.in_proj_ba = LoadGptqF16MergedNK(
          get, la + "in_proj_b.weight", la + "in_proj_a.weight",
          config.linear_num_value_heads, config.linear_num_value_heads,
          hidden);
      layer.gdn.conv1d_weight = LoadGptqF16(
          get, la + "conv1d.weight",
          {conv_dim, 1, config.linear_conv_kernel_dim});
      layer.gdn.a_log = LoadGptqF16VectorAsF32(
          get, la + "A_log", config.linear_num_value_heads);
      layer.gdn.dt_bias = LoadGptqF16VectorAsF32(
          get, la + "dt_bias", config.linear_num_value_heads);
      layer.gdn.norm_weight = LoadGptqF16(
          get, la + "norm.weight", {config.linear_value_head_dim});
    } else {
      const std::string sa = base + "self_attn.";
      layer.attn.q_norm = LoadGptqF16(
          get, sa + "q_norm.weight", {config.head_dim});
      layer.attn.k_norm = LoadGptqF16(
          get, sa + "k_norm.weight", {config.head_dim});
    }
    result.layers.push_back(std::move(layer));
  }
  return result;
}

}  // namespace

// VT_MODELOPT_W4A4 (default 0): consume a projection's on-disk activation
// divisor, setting `alpha` and so flipping `IsTrueW4A4()` to the fp4-ACTIVATION
// GEMM (docs/ENVIRONMENT.md). ONE reader, so the spellings cannot drift --
// which is why MODEL-DFLASH2-NVFP4 (#2758) exported it rather than reading the
// same variable a second time from the DFlash draft loader.
bool ModelOptW4A4OptIn() {
  const char* w4a4 = std::getenv("VT_MODELOPT_W4A4");
  return w4a4 != nullptr && w4a4[0] == '1';
}

bool DenseLmHeadFp4Enabled() {
  const char* v = std::getenv("VT_LMHEAD_FP4");
  return v == nullptr || v[0] != '0';
}

bool DenseLmHeadTakesNvfp4(const TensorExists& has, const std::string& proj) {
  return DenseLmHeadFp4Enabled() && IsNvfp4Projection(has, proj);
}

void LoadDenseLmHead(const TensorResolver& get, const TensorExists& has,
                     const std::string& proj, OwnedTensor& bf16_out,
                     Nvfp4Weight& fp4_out) {
  fp4_out = Nvfp4Weight{};
  bf16_out = OwnedTensor{};
  // PERF-27B-LMHEAD-FP4 (issue #213). An NVFP4 head stays PACKED, through the SAME
  // LoadNvfp4AnyNaming every other NVFP4 projection takes, so the ModelOpt vs
  // compressed-tensors global-scale convention is handled in exactly one place.
  // vLLM's mixed scheme likewise resolves a quantized head (modelopt.py:2491-2496).
  if (DenseLmHeadTakesNvfp4(has, proj)) {
    fp4_out = LoadNvfp4AnyNaming(get, has, proj);
    // The head is W4A16, whatever the naming. `LoadNvfp4AnyNaming` decides
    // activation-quant per SPELLING — the ModelOpt arm ignores `input_scale`
    // unless VT_MODELOPT_W4A4=1, but `LoadCtNvfp4Raw` consumes
    // `input_global_scale` UNCONDITIONALLY, correct for a TOWER projection of the
    // 27B compressed-tensors checkpoint, which really is W4A4. An output head is
    // not one: vLLM resolves it through ModelOptNvFp4W4A16LinearMethod, which
    // DELETES input_scale (modelopt.py:1365; registered at :1358) and pins
    // MarlinNvFp4LinearKernel (modelopt.py:1249,1283-1284). A set alpha would (a)
    // take the fp4-activation GEMM vLLM refuses here and (b) make
    // PrepareLmHeadResident early-return, skipping the pre-capture Marlin build.
    if (!ModelOptW4A4OptIn()) {
      fp4_out.input_global_scale_inv = 0.0F;
      fp4_out.alpha = 0.0F;
    }
    // The ONE weight that opts into a lifetime dequant-B resident where there is
    // no fp4 GEMM (qwen3_5_weights.h): the head is re-read whole every step.
    fp4_out.keep_dequant_b = true;
    return;
  }
  bf16_out = LoadLmHeadAnyDtype(get, has, proj + ".weight");
}

bool DenseCheckpointHasLmHead(const TensorExists& has, const std::string& proj) {
  // A bare `<proj>.weight` probe misses a compressed-tensors head, whose only
  // weight tensor is `<proj>.weight_packed`; no head at all means
  // `tie_word_embeddings`, so missing CT ties the logits to the embedding table.
  return has(proj + ".weight") || IsNvfp4Projection(has, proj);
}

OwnedTensor LoadMergedBf16RawNK(const TensorResolver& get,
                                const std::vector<std::string>& names) {
  // Extracted to the shared dense_weight_loaders.h (SEAM GAP #3); this retains
  // the public vllm::LoadMergedBf16RawNK API (used by the 27B GDN loader below
  // and test_qwen27_dense_forward) as a byte-identical thin forward.
  return dense_loaders::LoadMergedBf16RawNK(get, names);
}

GdnLayerWeights LoadQwen3_5DenseGdnFp16ForDiagnostics(
    const TensorResolver& get, const std::string& layer_base) {
  const TensorExists has = dense_loaders::ProbeThroughResolver(get);
  dense_loaders::CheckProbeCanAnswerNo(has, "LoadQwen3_5DenseGdnFp16ForDiagnostics");
  return LoadGdnDense(get, has, layer_base, Fp8BlockQuantConfig{}, true);
}

GdnLayerWeights LoadQwen3_5DenseGdn(const TensorResolver& get,
                                    const std::string& layer_base) {
  // Public focused-loader seam historically describes the 27B checkpoint, which
  // is NVFP4 and declares no `weight_block_size`, so the default-constructed
  // block config here is the truthful one and the routing is unchanged.
  //
  // `has` ASKS THE RESOLVER (#1256). It used to answer `true` for every name,
  // which was harmless while the only caller of `has` was a dtype probe that
  // went on to `get` the tensor anyway and would throw on a name that was not
  // there. M3's config/tensor cross-check made it harmful: it asks
  // `has(proj + ".weight_scale_inv")` WITHOUT then fetching it, so a stub that
  // says yes to everything reports a block-wise scale on a checkpoint that has
  // none, and `IsFp8BlockProjection` refuses the disagreement it just invented.
  // `TensorResolver` throws on a missing tensor, so probing it is the honest
  // answer and the only one available at this seam.
  //
  // FIX-PROBE-CANNOT-SAY-NO (#1258): the `try`/`catch` this used to spell out
  // now lives ONCE in `dense_loaders`, because the sibling seam below needs the
  // identical probe and two copies of a subtle thing is the next defect.
  const TensorExists has = dense_loaders::ProbeThroughResolver(get);
  dense_loaders::CheckProbeCanAnswerNo(has, "LoadQwen3_5DenseGdn");
  return LoadGdnDense(get, has, layer_base, Fp8BlockQuantConfig{});
}

bool IsQwen27QuantizedLinear(const std::string& name) {
  // Never quantized regardless of suffix (notes §3.6 `ignore`). The
  // `model.visual.` exemption is the vision tower the EXL3 27B ships UNQUANTIZED
  // beside its trellis text arm — unquantized-but-required, and since
  // MODEL-QWEN35-DENSE-VL-EXL3 actually LOADED by `LoadQwen3_5Dense` instead of
  // being left unread.
  if (name.rfind("mtp.", 0) == 0) return false;
  if (name.find("model.visual.") != std::string::npos) return false;
  if (name.find(".linear_attn.in_proj_") != std::string::npos) return false;
  if (name.find("lm_head") != std::string::npos) return false;
  // Quantized set: dense-MLP gate/up/down, self_attn q/k/v/o, GDN out_proj.
  auto ends_with = [&name](const char* suf) {
    const std::string s(suf);
    return name.size() >= s.size() &&
           name.compare(name.size() - s.size(), s.size(), s) == 0;
  };
  if (name.find(".mlp.") != std::string::npos &&
      (ends_with(".gate_proj") || ends_with(".up_proj") ||
       ends_with(".down_proj")))
    return true;
  if (name.find(".self_attn.") != std::string::npos &&
      (ends_with(".q_proj") || ends_with(".k_proj") || ends_with(".v_proj") ||
       ends_with(".o_proj")))
    return true;
  if (ends_with(".linear_attn.out_proj")) return true;
  return false;
}

OwnedTensor MaterializeCtNvfp4Bf16Transposed(const TensorResolver& get,
                                             const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8",
           "qwen3_5 dense: expected U8 weight_packed for " + proj);
  VT_CHECK(packed.shape.size() == 2,
           "qwen3_5 dense: expected 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0,
           "qwen3_5 dense: NVFP4 in_dim must be a multiple of 16 for " + proj);
  const StTensor& wscale = get(proj + ".weight_scale");
  VT_CHECK(wscale.dtype == "F8_E4M3",
           "qwen3_5 dense: expected F8_E4M3 weight_scale for " + proj);
  const float wgs_disk = ReadF32Scalar(get, proj + ".weight_global_scale");

  // Dequant to f32 [out, in] (the divisor is reciprocated inside), then round to
  // bf16 while transposing to Matmul-B layout [in, out].
  std::vector<float> f32(static_cast<size_t>(out_dim) * in_dim);
  DequantCtNvfp4WeightToF32(reinterpret_cast<const uint8_t*>(packed.data),
                            reinterpret_cast<const uint8_t*>(wscale.data),
                            wgs_disk, out_dim, in_dim, f32.data());
  MaybeReleaseSourcePages(packed.data, packed.nbytes);
  MaybeReleaseSourcePages(wscale.data, wscale.nbytes);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t r = 0; r < out_dim; ++r)
    for (int64_t c = 0; c < in_dim; ++c)
      dst[c * out_dim + r] =
          vt::F32ToBF16(f32[static_cast<size_t>(r) * in_dim + c]);
  return o;
}

Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(
    const TensorResolver& get, const TensorExists& has,
    const std::string& layer_type, int64_t layer_idx,
    const std::string& backbone_prefix, const Fp8BlockQuantConfig& block,
    bool exl3_fp16) {
  // FIX-PROBE-CANNOT-SAY-NO (#1258): every dense-layer path funnels through this
  // overload, so one check here covers the production loader, both resolver-only
  // seams, and whatever probe a caller supplies next.
  dense_loaders::CheckProbeCanAnswerNo(has, "LoadQwen3_5DenseLayer");
  const std::string base =
      backbone_prefix + "layers." + std::to_string(layer_idx) + ".";
  Qwen3_5DenseLayerWeights layer;
  // MODEL-QWEN35-EXL3 (#2495 item 3): the two decoder norms are unquantized and
  // ship at the checkpoint's own remainder dtype, which is F16 on an EXL3
  // artifact. The probe asks about the projections of THIS layer, because
  // exllamav3 decides the scheme per Linear and a `linear_attention` layer has
  // no `self_attn.q_proj` to ask about at all.
  const bool exl3 =
      dense_loaders::IsExl3Projection(has, base + "mlp.gate_proj") ||
      dense_loaders::IsExl3Projection(has, base + "self_attn.q_proj");
  layer.input_layernorm =
      LoadModelVectorForScheme(get, exl3, base + "input_layernorm.weight", exl3_fp16);
  layer.post_attention_layernorm = LoadModelVectorForScheme(
      get, exl3, base + "post_attention_layernorm.weight", exl3_fp16);
  if (layer_type == "linear_attention") {
    layer.is_linear_attention = true;
    layer.gdn = LoadGdnDense(get, has, base, block, exl3_fp16);
  } else if (layer_type == "full_attention") {
    layer.is_linear_attention = false;
    layer.attn = LoadAttnDense(get, has, base, block, exl3_fp16);
  } else {
    VT_CHECK(false, "qwen3_5 dense: unknown layer_type " + layer_type);
  }
  layer.mlp = LoadDenseMlp(get, has, base, block);
  return layer;
}

Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(
    const TensorResolver& get, const TensorExists& has,
    const std::string& layer_type, int64_t layer_idx,
    const std::string& backbone_prefix) {
  return LoadQwen3_5DenseLayer(get, has, layer_type, layer_idx, backbone_prefix,
                               Fp8BlockQuantConfig{});
}

Qwen3_5DenseLayerWeights LoadQwen3_5DenseLayer(
    const TensorResolver& get, const std::string& layer_type, int64_t layer_idx,
    const std::string& backbone_prefix) {
  // FIX-PROBE-CANNOT-SAY-NO (#1258). This was the SECOND always-true stub named
  // by #1256; #1257 fixed only the sibling above. It was not latent: the
  // `qwen36_gdn_layer_27b` isolated-layer golden replays through THIS overload
  // (`tests/parity/test_op_parity.cpp`) with `layer_type == "linear_attention"`,
  // which routes to `LoadGdnDense`, whose first act is the
  // `IsFp8BlockProjection` cross-check a constant `true` lies to. That leg is
  // SKIPped without the pinned 27B snapshot, which is why no CPU lane saw it.
  //
  // The compressed-tensors parity fixture this seam was written for really does
  // carry `weight_packed` on every routed projection, so a truthful probe leaves
  // its routing where it was and merely stops inventing the tensors it lacks.
  const TensorExists has = dense_loaders::ProbeThroughResolver(get);
  return LoadQwen3_5DenseLayer(get, has, layer_type, layer_idx,
                               backbone_prefix);
}

Qwen3_5DenseWeights LoadQwen3_5Dense(const std::vector<SafetensorsFile>& shards,
                                     const HfConfig& config,
                                     vt::Queue* load_queue,
                                     const MultiModalConfig* mm_config) {
  std::unordered_map<std::string, const SafetensorsFile*> where;
  std::vector<std::string> all_names;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      where[name] = &shard;
      all_names.push_back(name);
    }
  }

  // PERF-QWEN35-STAGE-WEIGHTS (#2327/#2342): decide ONCE, here, whether this
  // checkpoint's dense weights may be staged device-resident instead of
  // ATS-retagged. This is the only place that knows the model's TOTAL, and the
  // budget must come from a stable number rather than live free memory --
  // measured, a live-free floor captured 8.1% of an available 21.1% because
  // `free` falls as the host mirror loads (`platforms/interface.h:66-69` states
  // the rule; this is what ignoring it costs).
  //
  // Only the SAFETENSORS loader calls this. A GGUF checkpoint never does, so it
  // keeps the retag unconditionally — which is how #1299's `Qwen3.8-2.4T-A95B
  // UD-Q1_0` stays exactly as it is, and why the file total is a sound proxy
  // here: a bf16 safetensors weight occupies its file size in host memory,
  // while a GGUF one expands on load and would not.
  {
    size_t model_weight_bytes = 0;
    for (const SafetensorsFile& shard : shards)
      for (const std::string& name : shard.Names())
        model_weight_bytes += shard.Get(name).nbytes;
    size_t device_total = 0;
    if (load_queue != nullptr) {
      device_total = vllm::platforms::GetPlatform(load_queue->device.type)
                         .residency_policy()
                         .device_memory_total_bytes;
    }
    SetSafetensorsWeightBudget(model_weight_bytes, device_total);
  }
  // ONE namespace decision for the whole checkpoint (qwen3_5_weights.h): the
  // VL-nested spelling for the wrappers we gate, the flat `model.` spelling for
  // a text-only arm, and a refusal for a mixed index.
  const std::string backbone = ResolveQwen3_5BackbonePrefix(all_names);
  const TensorResolver get =
      [&where](const std::string& name) -> const StTensor& {
    auto it = where.find(name);
    VT_CHECK(it != where.end(), "qwen3_5 dense: tensor not found: " + name);
    return it->second->Get(name);
  };
  const TensorExists has = [&where](const std::string& name) {
    return where.find(name) != where.end();
  };
  // FIX-PROBE-CANNOT-SAY-NO (#1258). Truthful by construction today; checked
  // anyway, because the guard's value is that it does not depend on how the
  // probe was built — a name index filled from the wrong shard list fails here
  // too.
  dense_loaders::CheckProbeCanAnswerNo(has, "LoadQwen3_5Dense");

  VT_CHECK(config.num_hidden_layers > 0 &&
               static_cast<int64_t>(config.layer_types.size()) ==
                   config.num_hidden_layers,
           "qwen3_5 dense: layer_types size must equal num_hidden_layers");

  // A GPTQ projection has no `.weight`. Route by its storage spelling before
  // any BF16/FP8 loader attempts to resolve that absent tensor.
  const bool declared_gptq = config.raw.contains("quantization_config") &&
      config.raw["quantization_config"].is_object() &&
      config.raw["quantization_config"].value(
          "quant_method", std::string()) == "gptq";
  if (declared_gptq ||
      has(backbone + "layers.0.mlp.gate_proj.qweight")) {
    return LoadGptq4DenseText(shards, config, get, has, backbone, load_queue);
  }

  // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): ONE read of the quantization config for
  // the whole checkpoint, validated here rather than per projection. It carries
  // the block geometry the rung needs and the `modules_to_not_convert` list the
  // rung cross-checks against, which a per-tensor dtype probe cannot supply.
  // Default-constructed (`block_quant == false`) on every non-block checkpoint,
  // which leaves the routing below byte-identical to before this row.
  const Fp8BlockQuantConfig block = ReadFp8BlockQuantConfig(config);

  // QUANT-QWEN38-27B-NVFP4-ARM (#821). The SECOND whole-checkpoint read of the
  // quantization config, for the compressed-tensors spelling this file's
  // per-projection dtype probe cannot resolve.
  //
  // `unsloth/Qwen3.8-27B-NVFP4` @ `7d6f8d4d...` declares
  // `format: "mixed-precision"`: layers 0-55's MLP is NVFP4 W4A4 and layers
  // 56-63's MLP is FP8 W8A8, under the SAME module names. Which arm a
  // projection takes is a REGEX OVER LAYER INDICES in `config_groups`, so no
  // dtype probe can recover it — and its FP8 group ships a PER-OUTPUT-CHANNEL
  // BF16 `weight_scale` with DYNAMIC per-token activations, which this file's
  // `LoadFp8Raw` rung cannot represent. Before this call the loader entered the
  // per-tensor arm anyway and died on `tensor not found:
  // ...in_proj_qkv.input_scale`, a sentence about a checkpoint that is
  // complete: the artifact ships ZERO `*.input_scale` because its activation
  // scheme is dynamic, exactly as its config says.
  //
  // Empty on every checkpoint that is not compressed-tensors, and on every
  // compressed-tensors checkpoint whose groups this build can execute, so every
  // arm that loaded before this row still loads byte-identically.
  const std::string ct_refusal =
      layers::compressed_tensors::RefusalForHfConfigRaw(config.raw, all_names);
  VT_CHECK(ct_refusal.empty(), "qwen3_5 dense: " + ct_refusal);

  // QUANT-QWEN38-27B-NVFP4-ARM W5 (#821, campaign #1574). The THIRD
  // whole-checkpoint read, for the OTHER `mixed-precision` spelling: ModelOpt's.
  //
  // Two 27B artifacts in this family declare `quant_method: "modelopt"`,
  // `quant_algo: "MIXED_PRECISION"` and a `quantized_layers` map of exact
  // module names — `nvidia/Qwen3.6-27B-NVFP4` @ `0893e160`, which this tree has
  // loaded and measured since #466, and `r0b0tlab/Qwen3.8-27B-NVFP4-MTP-sm121`
  // @ `36f717a2`. Both split 401 Linears into 208 per-tensor STATIC FP8 and 193
  // W4A16_NVFP4 group_size 16. Neither is compressed-tensors, so the read above
  // stops at `quant_method` and answers "" for both, and until this call
  // NOTHING in this tree read either config: every routing decision for them
  // was made by probing which tensor NAMES are present.
  //
  // That probe cannot be wrong about the bytes and it can be wrong about the
  // CHECKPOINT, in both directions. A module the producer declared FP8 that
  // ships an NVFP4 spelling takes the NVFP4 arm; a module declared unquantized
  // that ships scales takes a quantized arm. Both load, both produce plausible
  // numbers, and both match tokens while moving the wrong bytes — the one
  // defect class `AGENTS.md` §"Inherit vLLM defaults" says a token gate cannot
  // see. So the declaration is CROSS-CHECKED against the shipped spelling and a
  // disagreement is refused by name.
  //
  // This CHECKS the probe rather than replacing it: no checkpoint whose config
  // and tensors agree takes a different arm than it did before this line, and
  // every non-ModelOpt checkpoint answers "" without reading anything. Routing
  // by the declared algorithm instead of by the probe is a larger change that
  // would move a gate model's arm; it is listed under `## Owed` in
  // `.agents/specs/qwen38-27b-quant-arms.md` and tracked by its own issue.
  const nlohmann::json* quant =
      layers::compressed_tensors::QuantizationConfigOf(config.raw);
  if (quant != nullptr) {
    const std::string modelopt_refusal =
        layers::modelopt::RefusalForQuantizationConfig(*quant, all_names);
    VT_CHECK(modelopt_refusal.empty(), "qwen3_5 dense: " + modelopt_refusal);

    // QUANT-QWEN38-27B-NVFP4-ARM W7 (#2760). The FOURTH whole-checkpoint read,
    // and the only one that is not a refusal.
    //
    // A THIRD ModelOpt artifact of this model declares an algorithm this build
    // does not execute, and the refusal above cannot see it.
    // `RadixArk/Qwen3.8-27B-NVFP4` @ `554ebba9` -- the target
    // `pangoleen/qwen3.8-27b-dgx-spark-dflash2` serves, and the artifact
    // `r0b0tlab/...-MTP-sm121` was derived from -- declares
    // `{"quant_algo": "NVFP4", "group_size": 16}` on all 193 of its NVFP4
    // modules, declares `config_groups.group_1.input_activations` 4-bit static,
    // and ships `<proj>.input_scale` on every one. `Refusal` accepts that,
    // correctly: its question is whether the shipped SPELLING matches the
    // declaration, and it does. The arm the loader then takes is the one
    // `VT_MODELOPT_W4A4` decides, and it defaults to the weight-only W4A16
    // dispatcher.
    //
    // SO WE RUN W4A16 ON A W4A4 CHECKPOINT, AND UNTIL THIS LINE WE PRINTED
    // NOTHING. The weight bytes per step are identical either way, so no
    // throughput axis moves and no roofline changes; the activation path and the
    // numerics differ, and a token gate cannot see that because W4A16 of a W4A4
    // checkpoint is numerically plausible. This does not refuse and does not
    // move any arm -- the checkpoint loads exactly as it did -- it makes the
    // divergence VISIBLE DEBT instead of a silent one. Whether the W4A4 path
    // should run at all is a lease-side question with a same-binary A/B and a
    // token gate attached; it is `## Owed` in
    // `.agents/specs/qwen38-27b-quant-arms.md`, tracked by #2760.
    const std::string activation_notice =
        layers::modelopt::ActivationArmNoticeForQuantizationConfig(
            *quant, all_names, ModelOptW4A4OptIn());
    if (!activation_notice.empty()) {
      std::fprintf(stderr, "qwen3_5 dense: %s\n", activation_notice.c_str());
    }
  }

  Qwen3_5DenseWeights w;

  // MODEL-QWEN35-DENSE-VL-EXL3 (ISSUE-LOCAL-01M3AHX9DQX8HNE32G80C9VGMJ). THE
  // SILENT DROP STOPS HERE, and BEFORE any backbone read, so an incomplete
  // tower is diagnosed before anything else can fail. A vision-inclusive
  // checkpoint carries `model.visual.*` tensors this loader historically never
  // read: it returned a text-only `Qwen3_5DenseWeights` and no message
  // anywhere said the tower was gone, so an image prompt had no path and no
  // diagnostic. Now the tower is REQUIRED when present and loaded through the
  // SHARED reader the MoE arm and the 27B token gates already use
  // (`LoadQwen3VLVisionWeights`); an index that NAMES the tower but ships it
  // incomplete is a loud refusal naming the first missing tensor, never a
  // quiet text-only model. A text-only checkpoint takes none of this:
  // `has_visual` stays false and the load is byte-identical (spec gate 2).
  //
  // #607 L3: the engine's modality limits are the ONE input to the skip.
  // `--language-model-only` and `--limit-mm-per-prompt '{"image":0,"video":0}'`
  // both land on `mm_config`; a null config (every non-engine caller) loads the
  // tower as before. The index is probed EITHER WAY, so "left unread" stays
  // distinguishable from "this checkpoint carries none".
  if (HasQwen3_5DenseVisionTower(shards)) {
    if (SkipTowerForModalities(mm_config, {"image", "video"})) {
      w.vision_skipped = true;
    } else {
      w.visual = LoadQwen3_5DenseVision(shards, config);
      w.visual_cfg = Qwen3_5DenseVisionConfig(config);
      w.has_visual = true;
    }
  }
  // MODEL-QWEN35-EXL3 (#2495 items 3 and 5). ONE whole-checkpoint question, and
  // it is asked of the TENSORS rather than of `quantization_config`: exllamav3
  // records the scheme per Linear, and an artifact that quantizes only part of
  // the model is a shape this loader must read correctly. Layer 0 of this
  // family is `linear_attention` and has no `self_attn`, so the MLP is the
  // probe that always exists.
  const bool exl3_checkpoint =
      dense_loaders::IsExl3Projection(has, backbone + "layers.0.mlp.gate_proj") ||
      dense_loaders::IsExl3Projection(has,
                                      backbone + "layers.0.self_attn.q_proj") ||
      dense_loaders::IsExl3Projection(has, "lm_head");
  w.exl3_checkpoint = exl3_checkpoint;
  const bool xpu_exl3 = exl3_checkpoint && load_queue != nullptr &&
      load_queue->device.type == vt::DeviceType::kXPU;
  w.precision = ResolveQwen3_5DensePrecision(config, false, xpu_exl3);
  // Preserve the ordinary backend path; the pinned FP16 policy belongs to XPU.
  w.embed_tokens = xpu_exl3
      ? dense_loaders::LoadF16Direct(get, backbone + "embed_tokens.weight")
      : (exl3_checkpoint && get(backbone + "embed_tokens.weight").dtype == "F16")
          ? dense_loaders::LoadF16AsBf16Direct(get, backbone + "embed_tokens.weight")
          : LoadBf16Direct(get, backbone + "embed_tokens.weight");
  w.final_norm =
      LoadModelVectorForScheme(get, exl3_checkpoint, backbone + "norm.weight", xpu_exl3);
  // The 27B owns an explicit head; smaller Qwen3.5 checkpoints tie logits to
  // the embedding table and omit lm_head.weight.
  if (dense_loaders::IsExl3Projection(has, "lm_head")) {
    // MODEL-QWEN35-EXL3 (#2495 item 5), mirroring `llama_weights.cpp:152-161`.
    // A REAL quantized head is preferred over a tied embedding table EVEN when
    // the config declares `tie_word_embeddings: true`: the publisher quantized
    // a separate head at its own width, so the head is what it intends to be
    // used. That width comes from the tensor and never from a config scalar --
    // `turboderp/Llama-3.2-1B-Instruct-exl3` @ 3.0bpw ships a SIX-bit head
    // under a config that says 3.0, and no shape check can catch a reader that
    // trusts the scalar.
    w.lm_head_exl3 = dense_loaders::LoadExl3(get, has, "lm_head");
  } else if (DenseCheckpointHasLmHead(has, "lm_head")) {
    LoadDenseLmHead(get, has, "lm_head", w.lm_head, w.lm_head_fp4);
  } else {
    w.tied_lm_head = true;
    w.embed_tokens.nk = true;
  }
  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  bool direct_device = DirectDeviceLoadEligible(load_queue);
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    w.layers.push_back(LoadQwen3_5DenseLayer(
        get, has, config.layer_types[static_cast<size_t>(l)], l, backbone,
        block, xpu_exl3));
    if (direct_device) {
      direct_device = IsPlainBf16Qwen3_5Dense(w);
      if (direct_device) StageAndReleaseLoadedDense(w, *load_queue);
    }
  }
  // MODEL-QWEN35-DENSE-VL-EXL3: the tower itself was loaded at the TOP of this
  // function (before any backbone read) so an incomplete tower is diagnosed
  // first; nothing further to do here.
  return w;
}

// MODEL-FP8-BLOCK-LINEAR (#1189 M4), the M4/M5 seam, narrowing the M3/M4 one,
// because the dense forward knew only
// fp4, per-tensor fp8 and bf16 and an unwired block-wise checkpoint would fall
// through to an EMPTY bf16 tensor. The forward reads all ten projections now,
// through THREE entry points rather than one:
// `dense_fp8_block::MatmulFp8BlockScaledD` reads eight of them,
// `MatmulFp8BlockMergedD` reads q/k/v as one operand, and
// `Fp8BlockGateUpSwiGLUD` is the only reader of `gate_proj` and `up_proj`. So
// what is left to refuse is a DEVICE with no kernel. M5 (`489a9a4c0`) added the
// mainloop-scaled CUTLASS kernel for `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a), so
// this now fires for a CUDA arch outside that cell. It stays inert on CPU,
// where `vt::MatmulFp8BlockScaled` is registered as a correctness reference.
// This runs from `PrepareQwen3_5Dense`, i.e.
// `ModelRegistry::Prepare`, which every runner calls unconditionally before the
// first forward and before any graph capture
// (`src/vllm/v1/worker/gpu/runner.cpp:414,455`), so the user is told before a
// capture rather than inside the first GEMM. M5 NARROWED this rather than
// deleting it: an arch outside the CUTLASS cell must still be refused by name.
void RefuseUnrunnableQwen3_5DenseFp8Block(const Qwen3_5DenseWeights& weights,
                                          vt::DeviceType device) {
  if (dense_fp8_block::BlockFp8Runnable(device)) return;
  for (size_t l = 0; l < weights.layers.size(); ++l) {
    const Qwen3_5DenseLayerWeights& layer = weights.layers[l];
    const std::string base = "model.layers." + std::to_string(l) + ".";
    const auto check = [&base, device](const Fp8BlockWeight& w,
                                       const std::string& suffix) {
      if (!w.Empty()) RefuseUnrunnableFp8BlockWeight(base + suffix, device);
    };
    if (layer.is_linear_attention) {
      check(layer.gdn.in_proj_qkv_fp8_block, "linear_attn.in_proj_qkv");
      check(layer.gdn.in_proj_z_fp8_block, "linear_attn.in_proj_z");
      check(layer.gdn.out_proj_fp8_block, "linear_attn.out_proj");
    } else {
      check(layer.attn.q_proj_fp8_block, "self_attn.q_proj");
      check(layer.attn.k_proj_fp8_block, "self_attn.k_proj");
      check(layer.attn.v_proj_fp8_block, "self_attn.v_proj");
      check(layer.attn.o_proj_fp8_block, "self_attn.o_proj");
    }
    check(layer.mlp.gate_proj_fp8_block, "mlp.gate_proj");
    check(layer.mlp.up_proj_fp8_block, "mlp.up_proj");
    check(layer.mlp.down_proj_fp8_block, "mlp.down_proj");
  }
}

bool IsPlainBf16Qwen3_5Dense(const Qwen3_5DenseWeights& weights) {
  // A PACKED head (PERF-27B-LMHEAD-FP4) is not plain bf16: this staging path
  // only knows how to stage OwnedTensors.
  if (!weights.lm_head_fp4.Empty()) return false;
  // MODEL-QWEN35-EXL3 (#2495 item 5): a trellis head is not plain bf16 either.
  // This predicate gates the DIRECT-DEVICE staging path, which knows only how
  // to stage `OwnedTensor`s; an EXL3 weight reaches the device through
  // `dense_attn::ResidentWeight` on three separate members instead.
  if (!weights.lm_head_exl3.Empty()) return false;
  for (const Qwen3_5DenseLayerWeights& layer : weights.layers) {
    if (!layer.gptq4.Empty()) return false;
    if (layer.mlp.IsExl3()) return false;
    if (!layer.is_linear_attention && layer.attn.IsExl3()) return false;
    // MODEL-QWEN35-GDN-EXL3 (#2495 item 4): the GDN tower, for exactly the
    // reason the two lines above exist. Without this term a checkpoint whose
    // linear-attention tower is EXL3 while its dense half is bf16 reads as
    // plain bf16, takes the direct-device staging path, and that path stages
    // `OwnedTensor` members only -- it would walk past the trellis, `suh` and
    // `svh` entirely and leave the tower unstaged. The published artifact
    // quantizes both halves, so `mlp.IsExl3()` already answers for it; the arm
    // is resolved PER PROJECTION from the tensors, so correctness here must not
    // depend on which halves one artifact happens to quantize.
    if (layer.is_linear_attention && layer.gdn.IsExl3()) return false;
    if (!layer.mlp.gate_proj_fp4.Empty() || !layer.mlp.up_proj_fp4.Empty() ||
        !layer.mlp.down_proj_fp4.Empty()) {
      return false;
    }
    // MODEL-FP8-BLOCK-WEIGHT (#1189 M3): a block-wise FP8 projection is not
    // plain bf16 either, and the direct-device staging path only knows how to
    // stage OwnedTensors.
    if (!layer.mlp.gate_proj_fp8_block.Empty() ||
        !layer.mlp.up_proj_fp8_block.Empty() ||
        !layer.mlp.down_proj_fp8_block.Empty()) {
      return false;
    }
    if (layer.is_linear_attention) {
      if (!layer.gdn.out_proj_fp4.Empty() ||
          !layer.gdn.in_proj_qkv_fp8.Empty() ||
          !layer.gdn.in_proj_z_fp8.Empty() ||
          !layer.gdn.out_proj_fp8.Empty() ||
          !layer.gdn.in_proj_qkv_fp8_block.Empty() ||
          !layer.gdn.in_proj_z_fp8_block.Empty() ||
          !layer.gdn.out_proj_fp8_block.Empty()) {
        return false;
      }
    } else if (!layer.attn.q_proj_fp8_block.Empty() ||
               !layer.attn.k_proj_fp8_block.Empty() ||
               !layer.attn.v_proj_fp8_block.Empty() ||
               !layer.attn.o_proj_fp8_block.Empty() ||
               !layer.attn.q_proj_fp4.Empty() ||
               !layer.attn.k_proj_fp4.Empty() ||
               !layer.attn.v_proj_fp4.Empty() ||
               !layer.attn.o_proj_fp4.Empty() ||
               !layer.attn.q_proj_fp8.Empty() ||
               !layer.attn.k_proj_fp8.Empty() ||
               !layer.attn.v_proj_fp8.Empty() ||
               !layer.attn.o_proj_fp8.Empty()) {
      return false;
    }
  }
  return true;
}

size_t ReleaseResidentQwen3_5DenseHostWeights(
    Qwen3_5DenseWeights& weights) {
  size_t released = 0;
  const auto release = [&released](OwnedTensor& tensor) {
    // `HostMirrorIsRedundant`, NOT `d_dev || d_dev_f32` (the second instance of
    // the use-after-free W0f introduced; #1299).
    //
    // `d_dev_f32` is a bf16->f32 UPCAST into a separate device allocation. It is
    // not a copy of these bytes and it can never stand in for them, so it never
    // made a host mirror redundant — it only looked like it did while every
    // weight that had one also had a `d_dev`. On the aliasing arm that stopped
    // being true: `PrepareBf16Resident` passes exactly four weights to BOTH
    // `raw()` and `f32()` — `gdn.conv1d_weight`, `gdn.norm_weight`,
    // `attn.q_norm`, `attn.k_norm` — and there `raw()` ALIASES (leaving `d_dev`
    // null) while `f32()` allocates (setting `d_dev_f32`). The disjunction then
    // passes and frees the very bytes the aliased raw tensor points at.
    //
    // NOTHING REACHES THAT COMBINATION, AND SINCE #1647 THE INVARIANT IS THE
    // ONLY REASON. This paragraph used to name a second one:
    // `DirectDeviceLoadEligible` also required `!platform.is_unified_memory()`,
    // and on CUDA `is_unified_memory()` and `host_memory_is_device_addressable()`
    // are the SAME `pageable && integrated` conjunction, computed independently
    // in `src/vt/cuda/cuda_backend.cu` and `src/vllm/platforms/cuda.cpp`. That
    // equality was an accident — no rule states it, no document records it and
    // no gate holds it — and the sentence written here said so and concluded
    // that asking the invariant instead of a proxy costs nothing. #1647 removed
    // the accident, because `is_unified_memory()` was switching the whole
    // stage-and-release lane off on the one device that needs it. What is left
    // is the invariant itself, doing the work it was already doing: on the
    // aliasing arm `ResidentWeight` leaves `d_dev` null, so `HostMirrorIsRedundant`
    // answers false and an aliased weight is never released. The `d_dev_f32`
    // disjunct stays deleted, and it stays redundant on the staging arm, where
    // every weight with a `d_dev_f32` has a `d_dev` too.
    if (tensor.HasHostBytes() && HostMirrorIsRedundant(tensor)) {
      released += tensor.bytes.size();
      tensor.ReleaseHost();
    }
  };
  const auto release_gdn = [&release](GdnLayerWeights& gdn) {
    release(gdn.in_proj_qkv);
    release(gdn.in_proj_z);
    release(gdn.in_proj_qkvz);
    release(gdn.in_proj_b);
    release(gdn.in_proj_a);
    release(gdn.in_proj_ba);
    release(gdn.conv1d_weight);
    release(gdn.a_log);
    release(gdn.dt_bias);
    release(gdn.norm_weight);
    release(gdn.out_proj);
  };
  const auto release_attn = [&release](FullAttnLayerWeights& attn) {
    release(attn.q_proj);
    release(attn.k_proj);
    release(attn.v_proj);
    release(attn.o_proj);
    release(attn.q_norm);
    release(attn.k_norm);
  };

  release(weights.embed_tokens);
  release(weights.final_norm);
  release(weights.lm_head);
  for (Qwen3_5DenseLayerWeights& layer : weights.layers) {
    release(layer.input_layernorm);
    release(layer.post_attention_layernorm);
    if (layer.is_linear_attention)
      release_gdn(layer.gdn);
    else
      release_attn(layer.attn);
    release(layer.mlp.gate_proj);
    release(layer.mlp.up_proj);
    release(layer.mlp.gate_up_proj);
    release(layer.mlp.down_proj);
    const auto release_gptq = [&release](Gptq4Weight& weight) {
      release(weight.qweight);
      release(weight.scales);
      release(weight.zero_point);
    };
    release_gptq(layer.gptq4.gdn_qkvz);
    release_gptq(layer.gptq4.gdn_out);
    release_gptq(layer.gptq4.attn_qkv);
    release_gptq(layer.gptq4.attn_out);
    release_gptq(layer.gptq4.mlp_gate_up);
    release_gptq(layer.gptq4.mlp_down);
  }
  return released;
}

}  // namespace vllm
