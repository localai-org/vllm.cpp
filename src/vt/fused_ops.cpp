#include "vt/fused_ops.h"

#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

#if defined(VLLM_CPP_HIP)
#include "vt/rocm/rocm_gelu_mul_sep.h"
#include "vt/rocm/rocm_gemma4_expert_geglu.h"
#include "vt/rocm/rocm_matmul_batch.h"
#include "vt/rocm/rocm_rmsnorm_plus_add.h"
#endif

namespace vt {

void RmsNormPlusAdd(Queue& q, Tensor& out, const Tensor& x, const Tensor& w,
                    const Tensor& addend, const RmsNormArgs& args) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::RmsNormPlusAddRocm(q, out, x, w, addend, args);
    return;
  }
#endif
  // Composed reference (CPU / non-ROCm): out = rmsnorm(x) + addend via tmp in out.
  // Use out as scratch for rmsnorm then add — requires out != addend alias.
  RmsNorm(q, out, x, w, args);
  Add(q, out, out, addend);
}

void DualRmsNormPlusRes(Queue& q, Tensor& out, const Tensor& x1, const Tensor& w1,
                        const Tensor& x2, const Tensor& w2, const Tensor& w3,
                        const Tensor& residual, const RmsNormArgs& args) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::DualRmsNormPlusResRocm(q, out, x1, w1, x2, w2, w3, residual, args);
    return;
  }
#endif
  if (q.device.type != DeviceType::kCPU) {
    throw std::runtime_error("vt::DualRmsNormPlusRes: non-ROCm GPU path not registered");
  }
  VT_CHECK(x1.rank == 2 && x1.shape[0] > 0 && x1.shape[1] > 0 &&
               (x1.dtype == DType::kBF16 || x1.dtype == DType::kF32),
           "DualRmsNormPlusRes: expected a nonempty BF16/F32 matrix");
  // Validate the count before IsContiguous/Numel can multiply signed extents.
  VT_CHECK(x1.shape[0] <= std::numeric_limits<int64_t>::max() / x1.shape[1],
           "DualRmsNormPlusRes: matrix element count overflow");
  const uint64_t elements = static_cast<uint64_t>(x1.shape[0]) *
                            static_cast<uint64_t>(x1.shape[1]);
  const size_t element_bytes = SizeOf(x1.dtype);
  VT_CHECK(elements <= std::numeric_limits<size_t>::max() / element_bytes,
           "DualRmsNormPlusRes: matrix byte count overflow");
  const size_t bytes = static_cast<size_t>(elements) * element_bytes;
  for (const Tensor* t : std::initializer_list<const Tensor*>{&x2, &residual, &out}) {
    VT_CHECK(t->rank == 2 && t->shape[0] == x1.shape[0] &&
                 t->shape[1] == x1.shape[1] && t->dtype == x1.dtype &&
                 t->device == q.device && t->IsContiguous(),
             "DualRmsNormPlusRes: matrix shape, dtype or device mismatch");
  }
  // CPU scratch retains each intermediate's declared dtype rounding. Do not
  // write out until both inputs and all weights are consumed: out may alias them.
  const size_t words = bytes / sizeof(uint32_t) +
                       (bytes % sizeof(uint32_t) != 0);
  std::vector<uint32_t> storage1(words), storage2(words);
  Tensor n1 = Tensor::Contiguous(storage1.data(), x1.dtype, q.device,
                                {x1.shape[0], x1.shape[1]});
  Tensor n2 = Tensor::Contiguous(storage2.data(), x1.dtype, q.device,
                                {x1.shape[0], x1.shape[1]});
  RmsNorm(q, n1, x1, w1, args);
  RmsNorm(q, n2, x2, w2, args);
  Add(q, n1, n1, n2);
  RmsNorm(q, n2, n1, w3, args);
  Add(q, out, n2, residual);
}

void GeluMulSeparate(Queue& q, void* out, const void* gate, const void* up, int64_t n,
                     DType dtype) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::GeluMulSeparateRocm(q, out, gate, up, n, dtype);
    return;
  }
#endif
  // Composed reference (CPU / non-ROCm), same shape as RmsNormPlusAdd above.
  //
  // This is NOT an optional fast path: gemma4.cpp's `ple > 0` block calls this
  // from the SHARED forward, so throwing here aborted Gemma-4 on its first
  // layer on every non-ROCm backend (issue #377). vt::GeluAndMul computes
  // exactly this math -- gelu_tanh(gate) * up -- but wants the two halves
  // adjacent as one [rows, 2D] tensor, so the compose stages them into a
  // temporary [1, 2n] laid out as [gate | up] and runs the shipped kernel on
  // it. Whatever GeluAndMul does, this matches it by construction, on every
  // backend that registers it.
  if (n <= 0) return;
  Backend& b = GetBackend(q.device.type);
  const size_t elem = SizeOf(dtype);
  const size_t half = static_cast<size_t>(n) * elem;
  void* tmp = b.Alloc(2 * half);
  if (tmp == nullptr) {
    throw std::runtime_error("vt::GeluMulSeparate: temporary allocation failed");
  }
  try {
    b.Copy(q, tmp, gate, half);
    b.Copy(q, static_cast<char*>(tmp) + half, up, half);
    Tensor tin = Tensor::Contiguous(tmp, dtype, q.device, {1, 2 * n});
    Tensor tout = Tensor::Contiguous(out, dtype, q.device, {1, n});
    GeluAndMul(q, tout, tin);
    // The temporary is read by work queued on `q`, so it must outlive that
    // work: on an async backend the Free below would otherwise race the
    // kernel. Synchronous backends (CPU) no-op this.
    b.Synchronize(q);
  } catch (...) {
    b.Free(tmp);
    throw;
  }
  b.Free(tmp);
}

bool HasMatmulBTAlphaBeta(const Queue& q) {
#if defined(VLLM_CPP_HIP)
  return q.device.type == DeviceType::kROCM;
#else
  (void)q;
  return false;
#endif
}

void MatmulBTAlphaBeta(Queue& q, void* out, const void* a, const void* b, int M, int N, int K,
                       float alpha, float beta, DType dtype) {
#if defined(VLLM_CPP_HIP)
  // Dispatch on the predicate rather than on a second copy of its condition, so
  // `HasMatmulBTAlphaBeta` cannot drift from what this function actually does.
  if (HasMatmulBTAlphaBeta(q)) {
    rocm::MatmulBTAlphaBetaRocm(q, out, a, b, M, N, K, alpha, beta, dtype);
    return;
  }
#endif
  (void)out;
  (void)a;
  (void)b;
  (void)M;
  (void)N;
  (void)K;
  (void)alpha;
  (void)beta;
  (void)dtype;
  // Two different absences reach this line, and telling a caller the wrong one
  // sends them to fix the wrong thing.
  //
  // A kROCM queue arrives here only when the build was configured without
  // `-DVLLM_CPP_HIP`, so the 'rocm' arm exists in the tree and is compiled out.
  // That is a build-configuration problem, not a missing kernel, and it is not
  // #1205 — the previous "ROCm-only in this build" got exactly this case right.
  if (q.device.type == DeviceType::kROCM) {
    throw std::runtime_error(
        "vt::MatmulBTAlphaBeta: the 'rocm' arm "
        "(src/vt/rocm/rocm_matmul_hipblaslt.hip) is compiled out of this build; "
        "reconfigure with -DVLLM_CPP_HIP to enable it.");
  }
  // Every other device arrives here because no such kernel was ever written:
  // the only implementation in the tree is rocm::MatmulBTAlphaBetaRocm. Name the
  // device that asked, name the one arm that exists, and name the issue that
  // owes the rest — "ROCm-only" alone left the caller unable to tell a missing
  // kernel from a missing build flag. Reaching this on CUDA is issue #1205 and
  // blocks #1126 step 1: waking Gemma4's device-expert LRU would route decode
  // into ExpertGeGLUDeviceAccum, which lands here outside the upload's
  // try/catch. `EnsureGemma4Fp8ExpertOnDevice` now refuses before that upload
  // (gemma4_moe.cpp), so this throw is the backstop rather than the guard.
  throw std::runtime_error(
      std::string("vt::MatmulBTAlphaBeta: no implementation for device '") +
      DeviceTypeName(q.device.type) + "'; no '" + DeviceTypeName(q.device.type) +
      "' kernel has been written and the only arm in the tree is 'rocm' "
      "(src/vt/rocm/rocm_matmul_hipblaslt.hip); see issue #1205.");
}

bool HasMatmulBTFp8Channel(const Queue& q) {
#if defined(VLLM_CPP_HIP)
  return q.device.type == DeviceType::kROCM;
#else
  (void)q;
  return false;
#endif
}

void MatmulBTFp8Channel(Queue& q, void* out, const void* a, const void* b_fp8,
                        const void* scale_bf16, int M, int N, int K, float alpha, float beta) {
#if defined(VLLM_CPP_HIP)
  // Dispatch on the predicate rather than on a second copy of its condition, so
  // `HasMatmulBTFp8Channel` cannot drift from what this function actually does.
  if (HasMatmulBTFp8Channel(q)) {
    rocm::MatmulBTFp8ChannelRocm(q, out, a, b_fp8, scale_bf16, M, N, K, alpha, beta);
    return;
  }
#endif
  (void)q;
  (void)out;
  (void)a;
  (void)b_fp8;
  (void)scale_bf16;
  (void)M;
  (void)N;
  (void)K;
  (void)alpha;
  (void)beta;
  throw std::runtime_error("vt::MatmulBTFp8Channel: ROCm-only in this build");
}

bool HasDequantFp8ChannelBf16(const Queue& q) {
#if defined(VLLM_CPP_HIP)
  return q.device.type == DeviceType::kROCM;
#else
  (void)q;
  return false;
#endif
}

void DequantFp8ChannelBf16(Queue& q, void* out_bf16, const void* fp8, const void* scale_bf16,
                           int N, int K) {
#if defined(VLLM_CPP_HIP)
  // Dispatch on the predicate, not on a second copy of its condition.
  if (HasDequantFp8ChannelBf16(q)) {
    rocm::DequantFp8ChannelBf16Rocm(q, out_bf16, fp8, scale_bf16, N, K);
    return;
  }
#endif
  (void)q;
  (void)out_bf16;
  (void)fp8;
  (void)scale_bf16;
  (void)N;
  (void)K;
  throw std::runtime_error("vt::DequantFp8ChannelBf16: ROCm-only in this build");
}

bool ExpertGeGLUBf16TopKM1(Queue& q, void* ysum, const void* x, const void* const* w_gu,
                           const void* const* w_dn, const float* wts, int G, int I, int H) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    return rocm::ExpertGeGLUBf16TopKM1Rocm(q, ysum, x, w_gu, w_dn, wts, G, I, H);
  }
#endif
  (void)q;
  (void)ysum;
  (void)x;
  (void)w_gu;
  (void)w_dn;
  (void)wts;
  (void)G;
  (void)I;
  (void)H;
  return false;
}

bool ExpertGeGLUFp8TopKM1(Queue& q, void* ysum, const void* x, const void* const* fp8_gu,
                          const void* const* s_gu, const void* const* fp8_dn,
                          const void* const* s_dn, const float* wts, int G, int I, int H) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    return rocm::ExpertGeGLUFp8TopKM1Rocm(q, ysum, x, fp8_gu, s_gu, fp8_dn, s_dn, wts, G, I, H);
  }
#endif
  (void)q;
  (void)ysum;
  (void)x;
  (void)fp8_gu;
  (void)s_gu;
  (void)fp8_dn;
  (void)s_dn;
  (void)wts;
  (void)G;
  (void)I;
  (void)H;
  return false;
}

bool ExpertGeGLUFp8TopKIndexed(Queue& q, void* ysum, const void* x, const void* gu_base,
                               const void* dn_base, const void* sgu_base, const void* sdn_base,
                               const int32_t* idx_dev, const float* wts_dev, int G, int I, int H) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    return rocm::ExpertGeGLUFp8TopKIndexedRocm(q, ysum, x, gu_base, dn_base, sgu_base, sdn_base,
                                               idx_dev, wts_dev, G, I, H);
  }
#endif
  (void)q; (void)ysum; (void)x; (void)gu_base; (void)dn_base; (void)sgu_base; (void)sdn_base;
  (void)idx_dev; (void)wts_dev; (void)G; (void)I; (void)H;
  return false;
}

void ApplyExpertScaleRw(Queue& q, float* rw_dev, const int32_t* ri_dev, const float* escale_dev,
                        int G, int E) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::ApplyExpertScaleRwRocm(q, rw_dev, ri_dev, escale_dev, G, E);
    return;
  }
#endif
  (void)q; (void)rw_dev; (void)ri_dev; (void)escale_dev; (void)G; (void)E;
}

bool PrewarmExpertGeGLUFp8TopK(int dev, int G, int I, int H) {
#if defined(VLLM_CPP_HIP)
  return rocm::PrewarmExpertGeGLUFp8TopKIndexedRocm(dev, G, I, H);
#else
  (void)dev; (void)G; (void)I; (void)H;
  return false;
#endif
}

void MoeGatherRows(Queue& q, void* out_bf16, const void* in_bf16, const int32_t* token_ids_dev,
                   int n, int H) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::MoeGatherRowsRocm(q, out_bf16, in_bf16, token_ids_dev, n, H);
    return;
  }
#endif
  (void)q;
  (void)out_bf16;
  (void)in_bf16;
  (void)token_ids_dev;
  (void)n;
  (void)H;
  throw std::runtime_error("vt::MoeGatherRows: ROCm-only in this build");
}

void MoeWeightedScatterAdd(Queue& q, void* acc_bf16, const void* y_bf16,
                           const int32_t* token_ids_dev, const float* weights_dev, int n, int H) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::MoeWeightedScatterAddRocm(q, acc_bf16, y_bf16, token_ids_dev, weights_dev, n, H);
    return;
  }
#endif
  (void)q;
  (void)acc_bf16;
  (void)y_bf16;
  (void)token_ids_dev;
  (void)weights_dev;
  (void)n;
  (void)H;
  throw std::runtime_error("vt::MoeWeightedScatterAdd: ROCm-only in this build");
}

void MoeZeroBf16(Queue& q, void* buf_bf16, int64_t nelem) {
#if defined(VLLM_CPP_HIP)
  if (q.device.type == DeviceType::kROCM) {
    rocm::MoeZeroBf16Rocm(q, buf_bf16, nelem);
    return;
  }
#endif
  (void)q;
  (void)buf_bf16;
  (void)nelem;
  throw std::runtime_error("vt::MoeZeroBf16: ROCm-only in this build");
}

}  // namespace vt
