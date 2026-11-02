// CUDA keep-quant GEMM gate (QUANT-GGUF-CIQ-GEMM-CUDA). The kCUDA provider for
// `OpId::kMatmulBTQuant` (src/vt/cuda/cuda_quant_dot.cu) is measured against the
// LANDED CPU keep-quant reference (src/vt/cpu/cpu_quant_gemm.cpp — the oracle,
// itself gated by test_ops_quant_dot.cpp) and against an INDEPENDENT f64
// dequantize-then-dot, on the DeepSeek-V4 encodings IQ2_XXS / IQ3_XXS / Q2_K
// plus the four other Q8_K-family k-quants the CUDA kernel also serves.
//
// The GATE (per the campaign): the CUDA kernel's Q8_K activation quant and the
// whole INTEGER dot are bit-identical to the CPU reference by construction, so
// CUDA-vs-CPU is asserted at a TIGHT NMSE (1e-6, f32 out) — only the per-super-
// block float scale sum is reassociated (warp reduction vs the CPU sequential
// add). CUDA-vs-f64-dequant is asserted at the SAME band test_ops_quant_dot uses
// (5e-4, test-backend-ops.cpp:4277). A wrong codebook index / scale unpack /
// sign would blow both bands (RED-first): the tables live device-side, so a
// transcription slip in cuda_quant_iq_tables.cuh surfaces here as an IQ2/IQ3
// divergence, not silently.
//
// Skips cleanly (returns) when no CUDA backend is present so the CPU CI leg is
// green; it only asserts on a real GB10/CUDA device.
#include <doctest/doctest.h>

#ifdef VLLM_CPP_CUDA
#include <cuda_runtime.h>  // cudaStream_t for the Brick 12 pair/group-diag externs
#endif

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <iterator>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/quant.h"      // vt::cpu::BlockToFloat
#include "vt/tensor.h"

// The ORACLE's own numbers for the two encodings this row adds: real weight
// bytes out of the staged GLM-5.3-Flash artifact, the oracle's own Q8_K
// activation blocks, and the f32 result llama.cpp `b10451` computed from them.
#include "iq2xs_iq4xs_dot_golden.h"
#include "iq2xs_iq4xs_golden_vectors.h"

// The production GGUF reader and the production residency decision, so the
// reachability case below can start at a real file header instead of at a
// hand-built block. `../vllm/gguf_builder.h` is the same cross-directory
// include tests/vllm/test_gguf_dequant.cpp already makes in the other
// direction for the golden vectors above.
#include "../vllm/gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"

#ifdef VLLM_CPP_CUDA
// The device-codebook seal: the CPU tables it is measured against, and the copy
// out of device memory (only the CUDA TU that defines them can address them).
#include "vt/cpu/cpu_quant_iq_tables.h"
#include "vt/cuda/cuda_iq_table_seal.h"
#endif

// UNCONDITIONAL: `CurrentPlatform()` is used at the `RouteGgufTensor` call in
// the IQ header case below, which is NOT inside `VLLM_CPP_CUDA`. This include
// spent one CI cycle inside that guard and broke `build-newest-gcc` and
// `build-test-cpu` while every CPU-only focused gate stayed green, because no
// suite in that target list builds this TU.
#include "vllm/platforms/interface.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Queue;
using vt::Tensor;

// Brick 12 (ds4-gap launch consolidation): the paired + block-diagonal Q8_0 decode GEMV
// kernels under test (external linkage from cuda_quant_dot.cu). The A/B gates below prove
// they are BYTE-IDENTICAL to the launches they consolidate. These two call the .cu
// symbols DIRECTLY (not through the OpProvider seam), so both the <cuda_runtime.h>
// declaration and the gates themselves exist only in a CUDA build — a CPU-only or
// sanitizer build has neither the toolkit header nor the symbols to link against.
#ifdef VLLM_CPP_CUDA
namespace vt::cuda {
void MatmulQ8_0PairCuda(vt::Tensor& out0, vt::Tensor& out1, const vt::Tensor& a,
                        const vt::Tensor& b0, const vt::Tensor& b1, cudaStream_t s);
void MatmulQ8_0GroupDiagCuda(vt::Tensor& out, const vt::Tensor& a, const vt::Tensor& b,
                             int64_t ng, cudaStream_t s);
}  // namespace vt::cuda
#endif  // VLLM_CPP_CUDA

namespace {

// test-backend-ops.cpp:4277 — the MUL_MAT NMSE band the CPU keep-quant is gated
// on; the CUDA kernel must meet the same against the f64 dequant reference.
constexpr double kMaxNmseErr = 5e-4;
// CUDA-vs-CPU-oracle: integer core identical, so only float reassociation drifts.
constexpr double kMaxNmseVsCpu = 1e-6;

bool HasCuda() {
  try {
    vt::GetBackend(DeviceType::kCUDA);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kCUDA, 0}; }

struct WeightCase {
  DType dtype;
  int64_t block_elems;
  int64_t block_bytes;
  int d_off;
  int dmin_off;
  const char* name;
  // Ceiling for the f64-DEQUANT case only (0 means use kMaxNmseErr). That
  // comparison holds a Q8_K-quantized activation against an f32 reference, so
  // it measures ACTIVATION error, whose reach depends on the weight
  // distribution. It does NOT bound the kernel. The CUDA-vs-CPU bound below is
  // what gates this port, it is shared by every case, and it is not relaxed.
  double nmse_ref_max = 0.0;
};

// Same table as test_ops_quant_dot.cpp: the Q8_K-family encodings the CUDA kernel
// serves natively PLUS Q8_0 (its own 32-block Q8_0-activation path — the DeepSeek-V4
// MLA/o-LoRA/shared-expert/lm_head weights) (offsets from ggml-common.h, restated
// independently).
const WeightCase kCases[] = {
    {DType::kIQ2_XXS, 256, 66, 0, -1, "iq2_xxs"},   // DeepSeek-V4 gate/up
    {DType::kIQ3_XXS, 256, 98, 0, -1, "iq3_xxs"},   // DeepSeek-V4 down
    {DType::kIQ2_S, 256, 82, 0, -1, "iq2_s"},       // DeepSeek-V4 UD-IQ2_M gate/up
    // The two sub-2-bit codebooks the Qwen3.8-2.4T checkpoints store their
    // routed experts in (96.92 % of each model). Without a CUDA arm these fall
    // to the CPU path and still emit CORRECT tokens, just at CPU speed, which
    // no token gate can see -- so device parity is gated HERE.
    // Measured, not guessed. Both are ternary lanes times a per-32 scale, so a
    // super-block spans a wider dynamic range than a 4-6 bit codebook while
    // Q8_K gives the activation ONE scale per 256 elements. iq1_s lands at
    // 5.2399e-4 on the thinnest shape (m=4, n=1 is FOUR dot products), against
    // a 5e-4 default -- the CPU test measures 5.2398e-4 at the SAME shape, and
    // that near-identity is what shows this is the activation term and not the
    // device kernel. 2e-3 is ~4x the residual and an order of magnitude below
    // the weakest decode defect (2.7e-2, measured by mutation on the CPU arm).
    {DType::kIQ1_S, 256, 50, 0, -1, "iq1_s", 2e-3},       // Qwen3.8 UD-IQ1_S experts
    {DType::kIQ1_XXXS, 256, 38, 0, -1, "iq1_xxxs", 2e-3},  // Qwen3.8 UD-Q1_0 experts
    // QUANT-CUDA-IQ4XS-IQ2XS (#2260). The two encodings the staged
    // `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL arm stores 85 of its 1412 tensors
    // in -- 82 IQ2_XS (`blk.N.ffn_{gate,up}_exps.weight`) and 3 IQ4_XS -- and
    // IQ4_XS is ALSO the last encoding the GLM-5.3 non-flash UD-IQ1_S arm had
    // no device kernel for, its other five all being here already. Adding the
    // rows drives all five gates in this file, and the fused gate+up+SwiGLU one
    // is the RED: with no `case` in that switch the seam THROWS by name, which
    // is why both models say `--device cpu` today.
    //
    // Offsets restated from ggml-common.h @ b10451 rather than copied:
    //   iq2_xs :388-393  d@0  qs@2 (u16[32])  scales@66 (u8[8])        (74 B)
    //   iq4_xs :454-460  d@0  scales_h@2 (u16)  scales_l@4  qs@8      (136 B)
    // Neither carries a `dmin`; IQ4_XS's per-sub-block delta is `d * (ls - 32)`
    // spliced out of scales_l/scales_h, so there is no second f16 field to name.
    {DType::kIQ2_XS, 256, 74, 0, -1, "iq2_xs"},     // GLM-5.3-Flash gate/up exps
    {DType::kIQ4_XS, 256, 136, 0, -1, "iq4_xs"},    // GLM-5.3-Flash + GLM-5.3 down exps
    {DType::kQ2_K, 256, 84, 80, 82, "q2_K"},        // DeepSeek-V4 UD-Q2_K_XL
    {DType::kQ3_K, 256, 110, 108, -1, "q3_K"},
    {DType::kQ4_K, 256, 144, 0, 2, "q4_K"},
    {DType::kQ5_K, 256, 176, 0, 2, "q5_K"},
    {DType::kQ6_K, 256, 210, 208, -1, "q6_K"},
    {DType::kQ8_0, 32, 34, 0, -1, "q8_0"},          // DeepSeek-V4 AProj/SExp/Out (Q8_0-act path)
    // QUANT-CUDA-KEEPQUANT-32B (#2419). The 32-element Q8_0-ACTIVATION lane.
    // These three cannot be served by the Q8_K GEMM the twelve rows above use --
    // their vec_dot pairs a 32-element block with a `BlockQ8_0`, not a 256-element
    // super-block with a `BlockQ8_K` -- so they run on a second templated GEMM
    // (`IsCuda32BlockKeepQuantSupported`).
    //
    // IQ4_NL and Q5_0 are what the released `unsloth/Qwen3.8-Flash-Next-GGUF`
    // UD-IQ1_S stores its routed-expert towers in, and that is forced rather than
    // chosen: `expert_feed_forward_length` is 640, so every routed expert row is
    // indivisible by 256 and NO K-quant can encode it.
    //
    // Adding the rows drives five gates in this file. The dense, grouped and
    // fused-MoE value gates say the new kernels are CORRECT; none of them can say
    // the device ran, because the pre-change tree drained the stream and returned
    // the same numbers from the CPU arm. The CAPTURE gate is the discriminator --
    // `cudaStreamSynchronize` on a capturing stream fails -- and the fused-MoE
    // gate is a second RED, because that seam THREW by name for these dtypes.
    //
    // Offsets restated from ggml-common.h @ b10451 rather than copied:
    //   q4_0   :213  d@0  qs@2 (u8[16])                 (18 B)
    //   q5_0   :229  d@0  qh@2 (u8[4])  qs@6 (u8[16])   (22 B)
    //   iq4_nl :447  d@0  qs@2 (u8[16])                 (18 B)
    // None carries a `dmin`, so `dmin_off` is -1 on all three; `d_off` is 0.
    {DType::kIQ4_NL, 32, 18, 0, -1, "iq4_nl"},      // Qwen3.8-Flash-Next experts
    {DType::kQ5_0, 32, 22, 0, -1, "q5_0"},          // Qwen3.8-Flash-Next experts
    {DType::kQ4_0, 32, 18, 0, -1, "q4_0"},          // same lane, one more case
    // NOTE (owed, #2419): MXFP4 (39, UD-IQ2_M ffn_down) is NOT here — it is the
    // fourth 32-element encoding and the lane above is the seam it needs, but no
    // artifact gated here uses it, so it still CPU-fallbacks. The CPU dot IS
    // gated in tests/vt/test_ops_quant_dot.cpp.
};

void GenerateData(float offset, size_t n, float* dst) {
  for (size_t i = 0; i < n; i++)
    dst[i] = 0.1F + 2 * std::cos(static_cast<float>(i) + offset);
}

std::vector<uint8_t> RandomBlocks(const WeightCase& c, int64_t nblocks,
                                  uint32_t seed) {
  std::mt19937 rng(seed);
  std::vector<uint8_t> bytes(static_cast<size_t>(nblocks * c.block_bytes));
  for (uint8_t& b : bytes) b = static_cast<uint8_t>(rng() & 0xFF);
  for (int64_t i = 0; i < nblocks; ++i) {
    uint8_t* blk = bytes.data() + i * c.block_bytes;
    auto put_f16 = [&](int off, float v) {
      const uint16_t h = vt::F32ToF16(v);
      std::memcpy(blk + off, &h, sizeof(h));
    };
    const float jitter = 1.0F + 0.05F * static_cast<float>(i % 7);
    if (c.d_off >= 0) put_f16(c.d_off, 0.0125F * jitter);
    if (c.dmin_off >= 0) put_f16(c.dmin_off, 0.0075F * jitter);
    // IQ1_S and IQ1_XXXS carry their per-32 sub-block scale INSIDE the weight
    // (qh bits 12-14, and the sc nibble's bits 0-2). Uniformly random bits
    // spread neighbouring 32-groups over a 15x scale range, which no encoder
    // emits, and Q8_K gives the ACTIVATION one scale per 256 elements -- so the
    // f64-dequant bound below would measure that synthetic dynamic range rather
    // than the kernel. The scale still VARIES per sub-block, so a kernel that
    // dropped or hoisted it is still caught; the delta sign bit stays random.
    // Mirrors the identical narrowing in tests/vt/test_ops_quant_dot.cpp.
    if (c.dtype == DType::kIQ1_S) {
      for (int ib = 0; ib < 8; ++ib) {
        uint16_t qh = 0;
        std::memcpy(&qh, blk + 34 + 2 * ib, sizeof(qh));
        const uint16_t ls = static_cast<uint16_t>(2 + ((i + ib) % 3));
        qh = static_cast<uint16_t>((qh & 0x8FFFU) | (ls << 12));
        std::memcpy(blk + 34 + 2 * ib, &qh, sizeof(qh));
      }
    }
    if (c.dtype == DType::kIQ1_XXXS) {
      for (int ib = 0; ib < 8; ++ib) {
        uint8_t& byte = blk[34 + ib / 2];
        const int shift = 4 * (ib & 1);
        const uint8_t ls = static_cast<uint8_t>(2 + ((i + ib) % 3));
        const uint8_t keep_sign = static_cast<uint8_t>((byte >> shift) & 0x8);
        byte = static_cast<uint8_t>((byte & ~(0xFU << shift)) |
                                    ((keep_sign | ls) << shift));
      }
    }
  }
  return bytes;
}

Tensor DevTensor(void* p, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = p;
  t.dtype = dt;
  t.device = Gpu();
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

bool PascalQ5Oracle(DType dtype, int64_t m) {
#ifdef VLLM_CPP_CUDA
  cudaDeviceProp prop{};
  REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
  return prop.major == 6 && dtype == DType::kQ5_0 && m == 1;
#else
  return false;
#endif
}

bool PascalQ6Oracle(DType dtype, int64_t m) {
#ifdef VLLM_CPP_CUDA
  cudaDeviceProp prop{};
  REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
  return prop.major == 6 && dtype == DType::kQ6_K && m == 1;
#else
  return false;
#endif
}

void Q6GpuOracle(std::vector<float>& result, const std::vector<float>& a,
                 const std::vector<uint8_t>& wq, int64_t n, int64_t k) {
  std::vector<float> decoded(n * k), reconstructed(k);
  vt::cpu::BlockToFloat(DType::kQ6_K)(wq.data(), decoded.data(), n * k);
  for (int64_t b = 0; b < k / 32; ++b) {
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(a[b * 32 + j]));
    const float d = amax / 127.0f;
    const float hd = vt::F16ToF32(vt::F32ToF16(d));
    for (int j = 0; j < 32; ++j)
      reconstructed[b * 32 + j] = d == 0 ? 0 : std::round(a[b * 32 + j] / d) * hd;
  }
  for (int64_t row = 0; row < n; ++row) {
    double sum = 0;
    for (int64_t j = 0; j < k; ++j) sum += double(decoded[row * k + j]) * reconstructed[j];
    result[row] = static_cast<float>(sum);
  }
}

bool PascalQ4KOracle(DType dtype, int64_t m) {
#ifdef VLLM_CPP_CUDA
  const char* e = std::getenv("VT_PASCAL_Q4K_DECODE");
  if (e != nullptr && e[0] == '0') return false;
  cudaDeviceProp prop{};
  REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
  return prop.major == 6 && dtype == DType::kQ4_K && m == 1;
#else
  return false;
#endif
}
void Q4KGpuOracle(std::vector<float>& result, const std::vector<float>& a,
                  const std::vector<uint8_t>& wq, int64_t n, int64_t k) {
  std::vector<float> decoded(n * k), reconstructed(k);
  vt::cpu::BlockToFloat(DType::kQ4_K)(wq.data(), decoded.data(), n * k);
  for (int64_t b = 0; b < k / 32; ++b) {
    float amax = 0;
    for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(a[b * 32 + j]));
    const float d = amax / 127.0f;
    const float hd = vt::F16ToF32(vt::F32ToF16(d));
    for (int j = 0; j < 32; ++j)
      reconstructed[b * 32 + j] = d == 0 ? 0 : std::round(a[b * 32 + j] / d) * hd;
  }
  for (int64_t row = 0; row < n; ++row) {
    double sum = 0;
    for (int64_t j = 0; j < k; ++j) sum += double(decoded[row * k + j]) * reconstructed[j];
    result[row] = static_cast<float>(sum);
  }
}
void Q5GpuOracle(std::vector<float>& cpu_out, const std::vector<float>& a,
                 const std::vector<uint8_t>& wq, int64_t n, int64_t nb) {
            for (int64_t row = 0; row < n; ++row) {
              float total = 0;
              for (int64_t block = 0; block < nb; ++block) {
                float tree[32], amax = 0;
                for (int j = 0; j < 32; ++j) {
                  tree[j] = a[block * 32 + j];
                  amax = std::max(amax, std::abs(tree[j]));
                }
                for (int mask : {16, 8, 4, 2, 1}) {
                  float old[32]; std::copy(tree, tree + 32, old);
                  for (int j = 0; j < 32; ++j) tree[j] = old[j] + old[j ^ mask];
                }
                const float d = amax / 127.0f;
                const float hd = vt::F16ToF32(vt::F32ToF16(d));
                const float hs = vt::F16ToF32(vt::F32ToF16(tree[0]));
                const uint8_t* wb = wq.data() + (row * nb + block) * 22;
                uint16_t wd; std::memcpy(&wd, wb, 2);
                uint32_t qh; std::memcpy(&qh, wb + 2, 4);
                for (int part = 0; part < 2; ++part) {
                  int dot = 0;
                  for (int j = part * 8; j < part * 8 + 8; ++j) {
                    const int lo = (wb[6 + j] & 15) | (((qh >> j) & 1) << 4);
                    const int hi = (wb[6 + j] >> 4) | (((qh >> (j + 16)) & 1) << 4);
                    dot += lo * (d == 0 ? 0 : static_cast<int>(std::round(a[block * 32 + j] / d)));
                    dot += hi * (d == 0 ? 0 : static_cast<int>(std::round(a[block * 32 + j + 16] / d)));
                  }
                  total += vt::F16ToF32(wd) * (dot * hd - 8.0f * hs);
                }
              }
              cpu_out[row] = total;
            }
}

}  // namespace

TEST_CASE("CUDA keep-quant GEMM == CPU reference and f64 dequant (Q8_K family + Q8_0)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  for (const WeightCase& c : kCases) {
    // Decode (M=1) through prefill (M=512), plus an odd N that catches a warp/
    // chunking assumption; K = 8 super-blocks (model-ish).
    const int64_t k = 8 * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{4}, int64_t{32}, int64_t{512}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(std::string(c.name));
        CAPTURE(m);
        CAPTURE(k);
        CAPTURE(n);

        std::vector<uint8_t> wq =
            RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());

        // --- CPU oracle (the landed keep-quant kernel over host tensors) ------
        std::vector<float> cpu_out(static_cast<size_t>(m * n), 0.0F);
        {
          Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {m, k});
          Tensor bt =
              Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
          bt.dtype = c.dtype;
          Tensor ot =
              Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {m, n});
          vt::MatmulBTQuant(cq, ot, at, bt);
        }

        if (PascalQ5Oracle(c.dtype, m))
          Q5GpuOracle(cpu_out, a, wq, n, k / c.block_elems);
        if (PascalQ6Oracle(c.dtype, m)) Q6GpuOracle(cpu_out, a, wq, n, k);
        if (PascalQ4KOracle(c.dtype, m)) Q4KGpuOracle(cpu_out, a, wq, n, k);


        // --- CUDA path (device tensors, unified pool) -------------------------
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        void* d_w = gpu.Alloc(wq.size());
        void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        Tensor bt = DevTensor(d_w, c.dtype, {n, k});
        Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
        vt::MatmulBTQuant(gq, ot, at, bt);
        std::vector<float> cuda_out(static_cast<size_t>(m * n), 0.0F);
        gpu.Copy(gq, cuda_out.data(), d_o, cuda_out.size() * sizeof(float));
        gpu.Synchronize(gq);
        gpu.Free(d_a);
        gpu.Free(d_w);
        gpu.Free(d_o);

        // --- f64 independent reference: decode weight + f32 activation dot -----
        std::vector<float> w(static_cast<size_t>(n * k));
        vt::cpu::BlockToFloat(c.dtype)(wq.data(), w.data(), n * k);

        double num_ref = 0, den_ref = 0, num_cpu = 0, den_cpu = 0;
        for (int64_t i = 0; i < m; ++i) {
          for (int64_t jj = 0; jj < n; ++jj) {
            double ref = 0;
            for (int64_t p = 0; p < k; ++p)
              ref += static_cast<double>(a[static_cast<size_t>(i * k + p)]) *
                     static_cast<double>(w[static_cast<size_t>(jj * k + p)]);
            const double got =
                cuda_out[static_cast<size_t>(i * n + jj)];
            const double cpu = cpu_out[static_cast<size_t>(i * n + jj)];
            num_ref += (got - ref) * (got - ref);
            den_ref += ref * ref;
            num_cpu += (got - cpu) * (got - cpu);
            den_cpu += cpu * cpu;
            REQUIRE(std::isfinite(got));
          }
        }
        const double nmse_ref = den_ref > 0 ? num_ref / den_ref : num_ref;
        const double nmse_cpu = den_cpu > 0 ? num_cpu / den_cpu : num_cpu;
        CAPTURE(nmse_ref);
        CAPTURE(nmse_cpu);
        const double ref_ceiling =
            c.nmse_ref_max > 0 ? c.nmse_ref_max : kMaxNmseErr;
        CAPTURE(ref_ceiling);
        // Q8_1 original-sum rounding has its own exact oracle gate.
        if (!PascalQ5Oracle(c.dtype, m)) CHECK(nmse_ref <= ref_ceiling);   // quantization error vs f64 dequant
        CHECK(nmse_cpu <= kMaxNmseVsCpu);  // matches the CPU oracle (int core exact)
      }
    }
  }
  gpu.DestroyQueue(gq);
}

// Brick 4 (last-mile): the CUDA coalesced-Q8_0 layout (RepackQ8_0Cuda + the
// q8_0_aligned kernel) must be BIT-IDENTICAL to the plain in-place Q8_0 path — it
// is a byte permutation of the SAME int8 + f16 scale values feeding the SAME dp4a
// dot. RED-first: a wrong deinterleave offset (qs/scale section) or a mis-aligned
// int4 load would diverge the two outputs.
TEST_CASE("Brick 4: CUDA Q8_0 aligned (coalesced) layout == plain Q8_0 (bit-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0-aligned gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const int64_t k = 8 * c.block_elems;  // 8 blocks
  for (int64_t m : {int64_t{1}, int64_t{4}}) {
    for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
      CAPTURE(m);
      CAPTURE(n);
      std::vector<uint8_t> wq = RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
      std::vector<float> a(static_cast<size_t>(m * k));
      GenerateData(1.0F, a.size(), a.data());

      void* d_a = gpu.Alloc(a.size() * sizeof(float));
      gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
      Tensor at = DevTensor(d_a, DType::kF32, {m, k});

      // plain in-place Q8_0
      void* d_w = gpu.Alloc(wq.size());
      void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
      gpu.Copy(gq, d_w, wq.data(), wq.size());
      Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});
      Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
      vt::MatmulBTQuant(gq, ot, at, bt);
      std::vector<float> plain(static_cast<size_t>(m * n), 0.0F);
      gpu.Copy(gq, plain.data(), d_o, plain.size() * sizeof(float));
      gpu.Synchronize(gq);

      // aligned (coalesced) layout: repack the SAME bytes + flag the tensor
      std::vector<uint8_t> wq_al = wq;
      vt::cpu::RepackQ8_0Cuda(wq_al.data(), n, k);
      void* d_wa = gpu.Alloc(wq_al.size());
      void* d_oa = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
      gpu.Copy(gq, d_wa, wq_al.data(), wq_al.size());
      Tensor bta = DevTensor(d_wa, DType::kQ8_0, {n, k});
      bta.q8_0_aligned = true;
      Tensor ota = DevTensor(d_oa, DType::kF32, {m, n});
      vt::MatmulBTQuant(gq, ota, at, bta);
      std::vector<float> aligned(static_cast<size_t>(m * n), 0.0F);
      gpu.Copy(gq, aligned.data(), d_oa, aligned.size() * sizeof(float));
      gpu.Synchronize(gq);

      for (size_t i = 0; i < plain.size(); ++i) {
        REQUIRE(std::isfinite(aligned[i]));
        CHECK(aligned[i] == plain[i]);  // BIT-IDENTICAL
      }
      gpu.Free(d_a);
      gpu.Free(d_w);
      gpu.Free(d_o);
      gpu.Free(d_wa);
      gpu.Free(d_oa);
    }
  }
  gpu.DestroyQueue(gq);
}

// Lever 1 (ds4-gap): the ds4-preq activation-quant grid (QuantizeQ8_0PreqKernel,
// one warp per 32-block, VT_V4_Q8_PREQ_QUANT default-ON) must be BIT-IDENTICAL to
// the legacy one-thread-per-block QuantizeQ8_0Kernel (=0). Both compute the same
// amax (MAX-reduction is associative + exact), same d = amax/127, same roundf ->
// byte-identical Q8_0 scratch feeding the same integer dot. RED-first: a wrong
// warp-reduce or lane->element mapping would diverge the two outputs.
TEST_CASE("Lever 1: CUDA Q8_0 preq-quant grid == legacy quant grid (bit-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 preq-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const int64_t k = 8 * c.block_elems;  // 8 blocks
  for (int64_t m : {int64_t{1}, int64_t{4}}) {
    for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
      CAPTURE(m);
      CAPTURE(n);
      std::vector<uint8_t> wq = RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
      std::vector<float> a(static_cast<size_t>(m * k));
      GenerateData(1.0F, a.size(), a.data());
      void* d_a = gpu.Alloc(a.size() * sizeof(float));
      gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
      Tensor at = DevTensor(d_a, DType::kF32, {m, k});
      void* d_w = gpu.Alloc(wq.size());
      gpu.Copy(gq, d_w, wq.data(), wq.size());
      Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});

      auto run = [&](const char* flag) {
        setenv("VT_V4_Q8_PREQ_QUANT", flag, 1);
        void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
        Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
        vt::MatmulBTQuant(gq, ot, at, bt);
        std::vector<float> out(static_cast<size_t>(m * n), 0.0F);
        gpu.Copy(gq, out.data(), d_o, out.size() * sizeof(float));
        gpu.Synchronize(gq);
        gpu.Free(d_o);
        return out;
      };
      std::vector<float> legacy = run("0");  // one-thread-per-block
      std::vector<float> preq = run("1");    // ds4-preq warp-per-block
      unsetenv("VT_V4_Q8_PREQ_QUANT");
      for (size_t i = 0; i < legacy.size(); ++i) {
        REQUIRE(std::isfinite(preq[i]));
        CHECK(preq[i] == legacy[i]);  // BIT-IDENTICAL
      }
      gpu.Free(d_a);
      gpu.Free(d_w);
    }
  }
  gpu.DestroyQueue(gq);
}

// Lever 2 / Brick 11 (ds4-gap): the sub-warp Q8_0 GEMV (QuantDotGemmQ8_0SubwarpKernel,
// VT_V4_Q8_SUBWARP=1) splits the 32-lane warp into LANES-wide subgroups (nb≤16→8,
// nb≤48→16, else 32), one output per subgroup. The integer __dp4a accumulation is
// order-independent → `sumi` is bit-exact for ANY LANES; only the final float scale-sum
// re-associates across fewer lanes → a characterized NEAR-TIE (NMSE≤5e-4). The big-K
// (nb=224 → LANES=32) case must be BYTE-IDENTICAL to the plain kernel (same lane→block
// map + same 32-wide reduce) — that byte-identity is the RED-first anchor: a wrong
// SubwarpMask/reduction/lane-map would diverge LANES=32 from plain and fail hard.
TEST_CASE("Lever 2: CUDA Q8_0 sub-warp GEMV == plain (bit-exact big-K, NMSE≤5e-4 short-K)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 sub-warp gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const double kMaxNmse = 5e-4;  // ratified DeepSeek-V4 near-tie band
  // nb = 16 (LANES=8, short-K MLA/kv), 48 (LANES=16, q_b/LoRA), 224 (LANES=32, big-K).
  for (int64_t nb : {int64_t{16}, int64_t{48}, int64_t{224}}) {
    const int64_t k = nb * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{3}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}}) {
        CAPTURE(nb);
        CAPTURE(m);
        CAPTURE(n);
        std::vector<uint8_t> wq = RandomBlocks(c, n * nb, 0x1EAF2U);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        void* d_w = gpu.Alloc(wq.size());
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});

        auto run = [&](const char* flag) {
          setenv("VT_V4_Q8_SUBWARP", flag, 1);
          void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
          Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
          vt::MatmulBTQuant(gq, ot, at, bt);
          std::vector<float> out(static_cast<size_t>(m * n), 0.0F);
          gpu.Copy(gq, out.data(), d_o, out.size() * sizeof(float));
          gpu.Synchronize(gq);
          gpu.Free(d_o);
          return out;
        };
        std::vector<float> plain = run("0");    // full 32-lane warp per output
        std::vector<float> subw = run("1");     // sub-warp tiling
        unsetenv("VT_V4_Q8_SUBWARP");
        double num = 0.0, den = 0.0;
        for (size_t i = 0; i < plain.size(); ++i) {
          REQUIRE(std::isfinite(subw[i]));
          const double d = static_cast<double>(subw[i]) - plain[i];
          num += d * d;
          den += static_cast<double>(plain[i]) * plain[i];
          if (nb == 224) CHECK(subw[i] == plain[i]);  // LANES=32 byte-identical (RED-first)
        }
        const double nmse = den > 0 ? num / den : num;
        CAPTURE(nmse);
        CHECK(nmse <= kMaxNmse);  // near-tie (short-K re-associates the float scale-sum)
        gpu.Free(d_a);
        gpu.Free(d_w);
      }
    }
  }
  gpu.DestroyQueue(gq);
}

// Brick 13 (ds4-gap ILP lever): the N-output-rows-per-warp Q8_0 GEMV
// (QuantDotGemmQ8_0MultiRowKernel, VT_V4_Q8_ILP=2|4) computes NROWS consecutive output
// columns per warp, sharing the activation load and issuing NROWS independent weight-load
// streams. UNLIKE the sub-warp lever (Brick 11), each output row still uses the FULL 32-lane
// warp reduce + the identical __dp4a order + the identical f16-scale fold → the result is
// BYTE-IDENTICAL to the plain one-row-per-warp kernel (not merely a near-tie). That
// byte-identity is the RED-first anchor: a wrong j0/tail/grid map or a re-associated reduce
// would diverge from plain and fail hard. Covers short-K (nb=16), mid-K (nb=48), big-K
// (nb=224); n both divisible and NOT divisible by NROWS (tail-row path); m=1 (decode) and m=3.
TEST_CASE("Brick 13: CUDA Q8_0 ILP multi-row GEMV == plain (byte-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 ILP gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  for (int64_t nb : {int64_t{16}, int64_t{48}, int64_t{224}}) {
    const int64_t k = nb * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{3}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}, int64_t{17}}) {  // 7,17 exercise tail
        CAPTURE(nb);
        CAPTURE(m);
        CAPTURE(n);
        std::vector<uint8_t> wq = RandomBlocks(c, n * nb, 0x13B1CU);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        void* d_w = gpu.Alloc(wq.size());
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});

        auto run = [&](const char* flag) {
          setenv("VT_V4_Q8_ILP", flag, 1);
          void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
          Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
          vt::MatmulBTQuant(gq, ot, at, bt);
          std::vector<float> out(static_cast<size_t>(m * n), 0.0F);
          gpu.Copy(gq, out.data(), d_o, out.size() * sizeof(float));
          gpu.Synchronize(gq);
          gpu.Free(d_o);
          return out;
        };
        std::vector<float> plain = run("1");   // one row per warp (baseline)
        std::vector<float> ilp2 = run("2");     // 2 rows per warp
        std::vector<float> ilp4 = run("4");     // 4 rows per warp
        unsetenv("VT_V4_Q8_ILP");
        for (size_t i = 0; i < plain.size(); ++i) {
          REQUIRE(std::isfinite(ilp2[i]));
          REQUIRE(std::isfinite(ilp4[i]));
          CHECK(ilp2[i] == plain[i]);  // byte-identical (RED-first)
          CHECK(ilp4[i] == plain[i]);
        }
        gpu.Free(d_a);
        gpu.Free(d_w);
      }
    }
  }
  gpu.DestroyQueue(gq);
}

// Brick 14 (ds4 raw-mechanism lever): the INTRA-ROW multi-block register-PREFETCH Q8_0 GEMV
// (QuantDotGemmQ8_0PrefetchKernel, VT_V4_Q8_PREFETCH=2|4) hoists PF super-block loads into
// registers before the dependent __dp4a chains, keeping the warp→output map (one row per warp)
// UNCHANGED — unlike the Brick-13 multi-ROW ILP. Each lane still visits its blocks in the SAME
// ascending order, runs the SAME 8×__dp4a per block, the SAME 32-wide warp reduce, and the SAME
// f16-scale fold → the result is BYTE-IDENTICAL to the plain one-row-per-warp kernel. That
// byte-identity is the RED-first anchor: a wrong group/tail stride or a re-associated accumulate
// would diverge from plain and fail hard. nb chosen to exercise groups that are NOT a multiple of
// 32*PF (tail-block path): nb=16 (< one PF=2 group), nb=48, nb=224; n both 1 and tail-y.
TEST_CASE("Brick 14: CUDA Q8_0 register-prefetch GEMV == plain (byte-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 prefetch gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  for (int64_t nb : {int64_t{16}, int64_t{48}, int64_t{224}}) {
    const int64_t k = nb * c.block_elems;
    for (int64_t m : {int64_t{1}, int64_t{3}}) {
      for (int64_t n : {int64_t{1}, int64_t{7}, int64_t{16}, int64_t{17}}) {
        CAPTURE(nb);
        CAPTURE(m);
        CAPTURE(n);
        std::vector<uint8_t> wq = RandomBlocks(c, n * nb, 0x14B14U);
        std::vector<float> a(static_cast<size_t>(m * k));
        GenerateData(1.0F, a.size(), a.data());
        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        Tensor at = DevTensor(d_a, DType::kF32, {m, k});
        void* d_w = gpu.Alloc(wq.size());
        gpu.Copy(gq, d_w, wq.data(), wq.size());
        Tensor bt = DevTensor(d_w, DType::kQ8_0, {n, k});

        auto run = [&](const char* flag) {
          setenv("VT_V4_Q8_PREFETCH", flag, 1);
          void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
          Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
          vt::MatmulBTQuant(gq, ot, at, bt);
          std::vector<float> out(static_cast<size_t>(m * n), 0.0F);
          gpu.Copy(gq, out.data(), d_o, out.size() * sizeof(float));
          gpu.Synchronize(gq);
          gpu.Free(d_o);
          return out;
        };
        std::vector<float> plain = run("1");   // one row per warp, no prefetch (baseline)
        std::vector<float> pf2 = run("2");      // prefetch depth 2
        std::vector<float> pf4 = run("4");      // prefetch depth 4
        unsetenv("VT_V4_Q8_PREFETCH");
        for (size_t i = 0; i < plain.size(); ++i) {
          REQUIRE(std::isfinite(pf2[i]));
          REQUIRE(std::isfinite(pf4[i]));
          CHECK(pf2[i] == plain[i]);  // byte-identical (RED-first)
          CHECK(pf4[i] == plain[i]);
        }
        gpu.Free(d_a);
        gpu.Free(d_w);
      }
    }
  }
  gpu.DestroyQueue(gq);
}

// ─── the GROUPED seams (kMatmulBTQuantGrouped, kMoeGateUpSwiGLUGrouped) ──────
//
// Three dispatch switches consume IsCudaKeepQuantSupported. #967 taught the
// predicate to say yes for IQ1_S and IQ1_XXXS and extended ONE of them. The
// grouped GEMM uses that same predicate to SKIP its CPU fallback and then ran a
// `switch (w)` with no case for either dtype and no default: it quantized the
// activation, launched NOTHING, and returned. `cudaGetLastError()` reported
// success, because a launch that never happened cannot fail, so the output
// tensor kept whatever it already held. That is the DEFAULT routed-expert path
// of Qwen3.8-2.4T (qwen3_5_gguf_weights.cpp -> qwen3_5.cpp KqGrouped ->
// vt::MatmulBTQuantGrouped), and the two dtypes are 96.92 % of that checkpoint.
// The fused gate+up+SwiGLU seam had the same hole, where it turned a NAMED
// "gate/up must be the SAME CUDA keep-quant dtype" refusal into a silent no-op.
//
// Each gate below asserts TWO things, and it needs both:
//   1. the output is not the POISON written into the device buffer before the
//      call. This is the only assertion that can see "no kernel ran"; every
//      value-comparison gate in this file would have read a zeroed allocation as
//      a merely inaccurate result.
//   2. it matches the CPU grouped golden (src/vt/cpu/cpu_quant_gemm.cpp, itself
//      gated by test_ops_quant_dot.cpp) to kMaxNmseVsCpu. That is what catches a
//      kernel that DID run, on the wrong expert row, scale, or codebook.
// The f64 dequant band is deliberately not repeated here: the grouped kernel is
// the dense kernel with one different weight-row index (same DotSuperblock, same
// FinalFactor, same warp reduce), so the dense case above already measures the
// quantization error per dtype and repeating it would only re-measure Q8_K.
namespace {

// Distinctive, never a plausible dot product, and the value the independent
// review used on GB10 when it measured this defect.
constexpr float kPoison = -12345.0F;

// P routed rows over E experts into N columns. P=1 is decode with one expert;
// N=7 puts an odd column count against the 32-lane warp tiling; the `bcast` arm
// (activation rows == 1 while P > 1) is the shape the routed MoE actually takes,
// where every selected expert reads ONE quantized hidden.
struct GroupedShape {
  int64_t P;
  int64_t n;
  int64_t E;
  bool bcast;
};
const GroupedShape kGroupedShapes[] = {
    {1, 1, 2, false}, {1, 16, 3, false}, {4, 7, 3, false}, {4, 7, 3, true}, {8, 16, 4, true},
};

std::vector<int32_t> ExpertIds(int64_t P, int64_t E) {
  std::vector<int32_t> ids(static_cast<size_t>(P));
  for (int64_t p = 0; p < P; ++p)
    ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 3 + 1) % E);
  return ids;
}

// num/den of the CUDA-vs-CPU NMSE, plus how many outputs still carry the poison.
struct GroupedVerdict {
  double nmse = 0.0;
  int64_t poisoned = 0;
  int64_t nonfinite = 0;
};

GroupedVerdict Compare(const std::vector<float>& got, const std::vector<float>& ref) {
  GroupedVerdict v;
  double num = 0, den = 0;
  for (size_t i = 0; i < got.size(); ++i) {
    const double g = got[i], r = ref[i];
    if (got[i] == kPoison) ++v.poisoned;
    if (!std::isfinite(g)) ++v.nonfinite;
    num += (g - r) * (g - r);
    den += r * r;
  }
  v.nmse = den > 0 ? num / den : num;
  return v;
}

}  // namespace

// The CPU twin of the probe below, and the reason to trust it. Both CUDA gates
// that follow return early on a host without a device, so on a CPU-only runner
// they assert NOTHING and doctest still prints SUCCESS. This case runs
// everywhere: it drives the SAME poisoned-buffer instrument through the CPU
// grouped golden, which must overwrite every element and reproduce the dense
// per-expert result byte for byte. If the instrument itself were broken -- a
// poison that is never written, a comparison that reads the wrong buffer -- it
// would be broken here too, where it can be seen without a GPU.
TEST_CASE("grouped keep-quant golden writes over a poisoned buffer (CPU arm)") {
  Queue cq{Cpu(), nullptr};
  int64_t combos = 0;
  for (const WeightCase& c : kCases) {
    const int64_t k = 8 * c.block_elems;
    for (const GroupedShape& g : kGroupedShapes) {
      const std::string case_name(c.name);
      CAPTURE(case_name);
      CAPTURE(g.P);
      CAPTURE(g.n);
      CAPTURE(g.bcast);
      ++combos;

      const int64_t arows = g.bcast ? 1 : g.P;
      const int64_t wrows = g.E * g.n;
      std::vector<uint8_t> wq = RandomBlocks(c, wrows * (k / c.block_elems), 0x5EEDU);
      std::vector<float> a(static_cast<size_t>(arows * k));
      GenerateData(1.0F, a.size(), a.data());
      std::vector<int32_t> ids = ExpertIds(g.P, g.E);
      const size_t outn = static_cast<size_t>(g.P * g.n);
      const size_t row_bytes = static_cast<size_t>(k / c.block_elems) * c.block_bytes;

      std::vector<float> got(outn, kPoison);
      {
        Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {arows, k});
        Tensor wt = Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {wrows, k});
        wt.dtype = c.dtype;
        Tensor et = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {g.P});
        Tensor ot = Tensor::Contiguous(got.data(), DType::kF32, Cpu(), {g.P, g.n});
        vt::MatmulBTQuantGrouped(cq, ot, at, wt, et);
      }

      // Independent reconstruction: the dense keep-quant GEMM over the row block
      // expert_ids[p] selects, one row of output at a time.
      std::vector<float> ref(outn, kPoison);
      for (int64_t p = 0; p < g.P; ++p) {
        const int64_t e = ids[static_cast<size_t>(p)];
        Tensor at = Tensor::Contiguous(a.data() + static_cast<size_t>(g.bcast ? 0 : p * k),
                                       DType::kF32, Cpu(), {1, k});
        Tensor wt = Tensor::Contiguous(wq.data() + static_cast<size_t>(e * g.n) * row_bytes,
                                       DType::kF32, Cpu(), {g.n, k});
        wt.dtype = c.dtype;
        Tensor ot = Tensor::Contiguous(ref.data() + static_cast<size_t>(p * g.n), DType::kF32,
                                       Cpu(), {1, g.n});
        vt::MatmulBTQuant(cq, ot, at, wt);
      }

      int64_t poisoned = 0;
      for (float v : got)
        if (v == kPoison) ++poisoned;
      CAPTURE(poisoned);
      CHECK(poisoned == 0);
      CHECK(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0);
    }
  }
  CAPTURE(combos);
  CHECK(combos == static_cast<int64_t>(std::size(kCases) * std::size(kGroupedShapes)));
  CHECK(combos > 0);
}

TEST_CASE("CUDA grouped keep-quant GEMM == CPU grouped golden and it WRITES the output") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA grouped keep-quant gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  int64_t combos = 0;
  for (const WeightCase& c : kCases) {
    const int64_t k = 8 * c.block_elems;
    for (const GroupedShape& g : kGroupedShapes) {
      const std::string case_name(c.name);
      CAPTURE(case_name);
      CAPTURE(g.P);
      CAPTURE(g.n);
      CAPTURE(g.E);
      CAPTURE(g.bcast);
      CAPTURE(k);
      ++combos;

      const int64_t arows = g.bcast ? 1 : g.P;
      std::vector<uint8_t> wq = RandomBlocks(c, g.E * g.n * (k / c.block_elems), 0x5EEDU);
      std::vector<float> a(static_cast<size_t>(arows * k));
      GenerateData(1.0F, a.size(), a.data());
      std::vector<int32_t> ids = ExpertIds(g.P, g.E);
      const size_t outn = static_cast<size_t>(g.P * g.n);

      // --- CPU golden (the landed grouped keep-quant kernel over host tensors)
      std::vector<float> cpu_out(outn, kPoison);
      {
        Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {arows, k});
        Tensor wt = Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {g.E * g.n, k});
        wt.dtype = c.dtype;
        Tensor et = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {g.P});
        Tensor ot = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {g.P, g.n});
        vt::MatmulBTQuantGrouped(cq, ot, at, wt, et);
      }

      // --- CUDA path, over a POISONED output buffer --------------------------
      void* d_a = gpu.Alloc(a.size() * sizeof(float));
      void* d_w = gpu.Alloc(wq.size());
      void* d_e = gpu.Alloc(ids.size() * sizeof(int32_t));
      void* d_o = gpu.Alloc(outn * sizeof(float));
      std::vector<float> poison(outn, kPoison);
      gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
      gpu.Copy(gq, d_w, wq.data(), wq.size());
      gpu.Copy(gq, d_e, ids.data(), ids.size() * sizeof(int32_t));
      gpu.Copy(gq, d_o, poison.data(), poison.size() * sizeof(float));
      gpu.Synchronize(gq);
      Tensor at = DevTensor(d_a, DType::kF32, {arows, k});
      Tensor wt = DevTensor(d_w, c.dtype, {g.E * g.n, k});
      Tensor et = DevTensor(d_e, DType::kI32, {g.P});
      Tensor ot = DevTensor(d_o, DType::kF32, {g.P, g.n});
      vt::MatmulBTQuantGrouped(gq, ot, at, wt, et);
      std::vector<float> cuda_out(outn, 0.0F);
      gpu.Copy(gq, cuda_out.data(), d_o, cuda_out.size() * sizeof(float));
      gpu.Synchronize(gq);
      gpu.Free(d_a);
      gpu.Free(d_w);
      gpu.Free(d_e);
      gpu.Free(d_o);

      const GroupedVerdict v = Compare(cuda_out, cpu_out);
      CAPTURE(v.nmse);
      CAPTURE(v.poisoned);
      CHECK(v.poisoned == 0);   // a dispatch that launches nothing lands HERE
      CHECK(v.nonfinite == 0);
      CHECK(v.nmse <= kMaxNmseVsCpu);
    }
  }
  // doctest prints "SUCCESS!" for a loop that never ran. Say how many it ran.
  CAPTURE(combos);
  CHECK(combos == static_cast<int64_t>(std::size(kCases) * std::size(kGroupedShapes)));
  CHECK(combos > 0);
  gpu.DestroyQueue(gq);
}

TEST_CASE("CUDA fused MoE gate+up+SwiGLU == CPU golden and it WRITES the output") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA fused-SwiGLU gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  int64_t combos = 0;
  for (const WeightCase& c : kCases) {
    // The fused kernel is Q8_K-activation only; Q8_0 has no fused arm and the
    // op REFUSES it by name (there is no CPU fallback behind this seam).
    if (c.dtype == DType::kQ8_0) continue;
    const int64_t k = 8 * c.block_elems;
    for (const GroupedShape& g : kGroupedShapes) {
      // limit=+inf is the plain silu(g)*u MLP; a finite limit exercises both
      // clamp arms of the epilogue.
      for (float limit : {std::numeric_limits<float>::infinity(), 3.0F}) {
        const std::string case_name(c.name);
        CAPTURE(case_name);
        CAPTURE(g.P);
        CAPTURE(g.n);
        CAPTURE(g.bcast);
        CAPTURE(limit);
        ++combos;

        const int64_t arows = g.bcast ? 1 : g.P;
        const int64_t wrows = g.E * g.n;
        std::vector<uint8_t> gw = RandomBlocks(c, wrows * (k / c.block_elems), 0x5EEDU);
        std::vector<uint8_t> uw = RandomBlocks(c, wrows * (k / c.block_elems), 0xC0FFEEU);
        std::vector<float> a(static_cast<size_t>(arows * k));
        GenerateData(1.0F, a.size(), a.data());
        std::vector<int32_t> ids = ExpertIds(g.P, g.E);
        const size_t outn = static_cast<size_t>(g.P * g.n);

        std::vector<float> cpu_out(outn, kPoison);
        {
          Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {arows, k});
          Tensor gt = Tensor::Contiguous(gw.data(), DType::kF32, Cpu(), {wrows, k});
          Tensor ut = Tensor::Contiguous(uw.data(), DType::kF32, Cpu(), {wrows, k});
          gt.dtype = c.dtype;
          ut.dtype = c.dtype;
          Tensor et = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {g.P});
          Tensor ot = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {g.P, g.n});
          vt::MoeGateUpSwiGLUGrouped(cq, ot, at, gt, ut, et, limit);
        }

        void* d_a = gpu.Alloc(a.size() * sizeof(float));
        void* d_g = gpu.Alloc(gw.size());
        void* d_u = gpu.Alloc(uw.size());
        void* d_e = gpu.Alloc(ids.size() * sizeof(int32_t));
        void* d_o = gpu.Alloc(outn * sizeof(float));
        std::vector<float> poison(outn, kPoison);
        gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
        gpu.Copy(gq, d_g, gw.data(), gw.size());
        gpu.Copy(gq, d_u, uw.data(), uw.size());
        gpu.Copy(gq, d_e, ids.data(), ids.size() * sizeof(int32_t));
        gpu.Copy(gq, d_o, poison.data(), poison.size() * sizeof(float));
        gpu.Synchronize(gq);
        Tensor at = DevTensor(d_a, DType::kF32, {arows, k});
        Tensor gt = DevTensor(d_g, c.dtype, {wrows, k});
        Tensor ut = DevTensor(d_u, c.dtype, {wrows, k});
        Tensor et = DevTensor(d_e, DType::kI32, {g.P});
        Tensor ot = DevTensor(d_o, DType::kF32, {g.P, g.n});
        vt::MoeGateUpSwiGLUGrouped(gq, ot, at, gt, ut, et, limit);
        std::vector<float> cuda_out(outn, 0.0F);
        gpu.Copy(gq, cuda_out.data(), d_o, cuda_out.size() * sizeof(float));
        gpu.Synchronize(gq);
        gpu.Free(d_a);
        gpu.Free(d_g);
        gpu.Free(d_u);
        gpu.Free(d_e);
        gpu.Free(d_o);

        const GroupedVerdict v = Compare(cuda_out, cpu_out);
        CAPTURE(v.nmse);
        CAPTURE(v.poisoned);
        CHECK(v.poisoned == 0);
        CHECK(v.nonfinite == 0);
        // The SwiGLU epilogue can drive a clamped output to exactly zero, which
        // makes the denominator small on a thin shape; the band is the same
        // CUDA-vs-CPU one because the integer core is the same.
        CHECK(v.nmse <= kMaxNmseVsCpu);
      }
    }
  }
  CAPTURE(combos);
  CHECK(combos == static_cast<int64_t>((std::size(kCases) - 1) * std::size(kGroupedShapes) * 2));
  CHECK(combos > 0);
  gpu.DestroyQueue(gq);
}

TEST_CASE("CUDA keep-quant GEMM registers the native kCUDA provider") {
  // The registration is what flips the GGUF loader's keep-quant default ON on a
  // CUDA device (GgufQuantComputeAvailable -> OpRegistered(kMatmulBTQuant,kCUDA))
  // so DeepSeek-V4's experts dispatch to the GPU. Present only in a CUDA build.
  if (!HasCuda()) return;
  CHECK(vt::OpRegistered(vt::OpId::kMatmulBTQuant, DeviceType::kCUDA));
}

// ─── QUANT-CUDA-IQ4XS-IQ2XS (#2260): the two gates the kCases rows cannot be ──
//
// Adding IQ2_XS and IQ4_XS to `kCases` drives the five gates above, and four of
// them compare the CUDA arm against OUR OWN CPU kernel. That is the right bound
// for a port -- the integer core is bit-identical by construction -- but it is
// consistency, not correctness: a defect present in both arms agrees with
// itself. And the fifth, "vs an f64 dequantize-then-dot", reads OUR decoder.
//
// The two cases below are the ones that answer a question the others cannot.

namespace {

// tests/vt/test_ops_quant_dot.cpp::MakeDotActivation, restated rather than
// shared: this file has no dependency on that translation unit, and the signal
// is the load-bearing half of the golden's provenance. Every value is an
// integer in [-1024, 1023] over 64, so it is exact in binary32 on any compiler
// and the same bytes come out of any build.
void MakeDotActivation(int n, uint32_t seed, float* x) {
  uint32_t st = seed;
  for (int i = 0; i < n; ++i) {
    st = st * 1664525U + 1013904223U;
    const int32_t v = static_cast<int32_t>((st >> 16) & 0x7ffU) - 1024;
    x[i] = static_cast<float>(v) / 64.0F;
  }
}

uint32_t FloatBits(float f) {
  uint32_t u = 0;
  std::memcpy(&u, &f, sizeof(u));
  return u;
}

float BitsFloat(uint32_t u) {
  float f = 0.0F;
  std::memcpy(&f, &u, sizeof(f));
  return f;
}

// One oracle-gated device case, shared by both encodings. `weights` is FOUR
// whole super-blocks read out of the staged artifact; `expected_per_block` and
// `expected_total` are what llama.cpp b10451's OWN vec_dot returned for them
// against the oracle's OWN Q8_K activation blocks.
//
// WHAT IS COMPARED AGAINST WHAT, in words, because a reader cannot audit wiring
// they cannot see:
//   * at k=256 the CUDA GEMM's f32 output is compared BIT FOR BIT against the
//     ORACLE's per-super-block number. One super-block means ONE contributing
//     warp lane, the other 31 contribute an exact +0.0f, and `FinalFactor` is a
//     power of two -- so no reassociation exists to explain a difference away.
//     Nothing in this comparison reads our decoder, our CPU vec_dot, or our
//     activation encoder: the device quantizes the f32 signal itself, and if it
//     produced anything but the oracle's Q8_K bytes these bits would move.
//   * at k=1024 FOUR lanes contribute, and `QuantDotGemmKernel` reduces them
//     with __shfl_down_sync at offsets 16,8,4,2,1 -- which over four live lanes
//     is (v0+v2)+(v1+v3), where the oracle accumulates ((v0+v1)+v2)+v3. So the
//     PRIMARY assertion is bit equality against the oracle's own four numbers
//     recombined in THAT order, and a SECONDARY one bounds the difference from
//     the oracle's sequential total by the reassociation error. If the primary
//     ever fails while the secondary holds, the warp reduction moved; that is a
//     different finding from a wrong codebook and it should not read the same.
void CheckCudaOracleDot(DType dtype, const char* name, const uint8_t* weights,
                        size_t wbytes, uint32_t act_seed, uint32_t expected_total,
                        const uint32_t (&expected_per_block)[4]) {
  constexpr int kBlocks = 4;
  constexpr int kK = 256 * kBlocks;
  CAPTURE(name);

  // The pairing is a CLAIM and it has been got wrong before: IQ4_XS shares
  // IQ4_NL's 16-entry codebook but NOT its 32-element block, so it dots Q8_K.
  REQUIRE(vt::cpu::QuantTraits(dtype).vec_dot_type == DType::kQ8_K);
  const size_t block_bytes = wbytes / kBlocks;
  REQUIRE(block_bytes * kBlocks == wbytes);

  std::vector<float> act(kK);
  MakeDotActivation(kK, act_seed, act.data());

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();

  void* d_a = gpu.Alloc(act.size() * sizeof(float));
  void* d_w = gpu.Alloc(wbytes);
  void* d_o = gpu.Alloc(kBlocks * sizeof(float));
  gpu.Copy(gq, d_a, act.data(), act.size() * sizeof(float));
  gpu.Copy(gq, d_w, weights, wbytes);
  gpu.Synchronize(gq);

  // --- k=256, one super-block at a time: BIT FOR BIT against the oracle ------
  for (int b = 0; b < kBlocks; ++b) {
    CAPTURE(b);
    const float poison = kPoison;
    gpu.Copy(gq, d_o, &poison, sizeof(float));
    gpu.Synchronize(gq);
    Tensor at = DevTensor(static_cast<uint8_t*>(d_a) + b * 256 * sizeof(float),
                          DType::kF32, {1, 256});
    Tensor wt = DevTensor(static_cast<uint8_t*>(d_w) + b * block_bytes, dtype, {1, 256});
    Tensor ot = DevTensor(d_o, DType::kF32, {1, 1});
    vt::MatmulBTQuant(gq, ot, at, wt);
    float got = 0.0F;
    gpu.Copy(gq, &got, d_o, sizeof(float));
    gpu.Synchronize(gq);
    // isfinite FIRST: every comparison against NaN is false, so an all-NaN
    // forward reads as agreement to any mismatch counter.
    REQUIRE(std::isfinite(got));
    CHECK(got != kPoison);
    CAPTURE(got);
    CAPTURE(BitsFloat(expected_per_block[b]));
    CHECK(FloatBits(got) == expected_per_block[b]);
  }

  // --- k=1024, all four at once ---------------------------------------------
  {
    const float poison = kPoison;
    gpu.Copy(gq, d_o, &poison, sizeof(float));
    gpu.Synchronize(gq);
    Tensor at = DevTensor(d_a, DType::kF32, {1, kK});
    Tensor wt = DevTensor(d_w, dtype, {1, kK});
    Tensor ot = DevTensor(d_o, DType::kF32, {1, 1});
    vt::MatmulBTQuant(gq, ot, at, wt);
    float got = 0.0F;
    gpu.Copy(gq, &got, d_o, sizeof(float));
    gpu.Synchronize(gq);
    REQUIRE(std::isfinite(got));
    CHECK(got != kPoison);

    const float p0 = BitsFloat(expected_per_block[0]);
    const float p1 = BitsFloat(expected_per_block[1]);
    const float p2 = BitsFloat(expected_per_block[2]);
    const float p3 = BitsFloat(expected_per_block[3]);
    const float tree = (p0 + p2) + (p1 + p3);   // the warp reduction's order
    CAPTURE(got);
    CAPTURE(tree);
    CHECK(FloatBits(got) == FloatBits(tree));

    // The reassociation bound, stated from the data rather than chosen: four
    // f32 additions over these magnitudes, each carrying at most one ULP of the
    // running sum. It is printed as a MARGIN so a reader sees how much room the
    // gate actually had, not only that it passed.
    const float seq = BitsFloat(expected_total);
    const double mag = static_cast<double>(std::fabs(p0)) + std::fabs(p1) +
                       std::fabs(p2) + std::fabs(p3);
    const double bound = 4.0 * 1.1920929e-7 * mag;
    const double margin = std::fabs(static_cast<double>(got) - seq);
    CAPTURE(seq);
    CAPTURE(margin);
    CAPTURE(bound);
    CHECK(margin <= bound);
  }

  gpu.Free(d_a);
  gpu.Free(d_w);
  gpu.Free(d_o);
  gpu.DestroyQueue(gq);
}

}  // namespace

TEST_CASE("CUDA IQ2_XS / IQ4_XS dot the ORACLE's own numbers on REAL checkpoint bytes") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; CUDA oracle-dot gate skipped");
    return;
  }
  // Weights: `blk.3.ffn_gate_exps.weight` and `blk.11.ffn_down_exps.weight` of
  // the staged `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL artifact. Expected
  // values: llama.cpp b10451 `ggml/src/ggml-cpu/quants.c:948`
  // `ggml_vec_dot_iq2_xs_q8_K_generic` and `:1283`
  // `ggml_vec_dot_iq4_xs_q8_K_generic`. Provenance, the reproduction recipe and
  // what these particular blocks exercise are in iq2xs_iq4xs_dot_golden.h.
  CheckCudaOracleDot(DType::kIQ2_XS, "iq2_xs", vllm_test::kIq2xsGoldenBlocks,
                     std::size(vllm_test::kIq2xsGoldenBlocks), 0x2247U,
                     vllm_test::kIq2xsDotExpectedBits,
                     vllm_test::kIq2xsDotPerBlockBits);
  CheckCudaOracleDot(DType::kIQ4_XS, "iq4_xs", vllm_test::kIq4xsGoldenBlocks,
                     std::size(vllm_test::kIq4xsGoldenBlocks), 0x4247U,
                     vllm_test::kIq4xsDotExpectedBits,
                     vllm_test::kIq4xsDotPerBlockBits);
}

// ─── the gate that separates "ran on the device" from "drained to the host" ──
//
// `IsCudaKeepQuantSupported` returning false is NOT a refusal: the dense and
// grouped seams call `cudaStreamSynchronize` and then run the CPU keep-quant
// kernel over the same unified-memory tensors. On GB10 that produces CORRECT
// numbers at host speed, so every value comparison in this file reads it as a
// pass, and the only visible symptom is a throughput deficit nobody attributed.
//
// Stream capture is what tells them apart, and it is not a contrivance: decode
// here is graph-captured, so a keep-quant GEMM that cannot be captured is a
// keep-quant GEMM the decode path cannot use. `cudaStreamSynchronize` on a
// capturing stream fails, which invalidates the capture -- so this case is RED
// for exactly the dtypes that fall back and green for the ones that do not.
// ─── from a REAL GGUF header to the device kernel, in one chain ─────────────
//
// Every other gate in this file hands the GEMM bytes this test made up. That
// proves the kernel and says nothing about whether a checkpoint can reach it,
// and the two have come apart here before: a dtype can decode, keep its blocks,
// and still be absent from the device dispatch, which is exactly the state
// #2260 found and fixed.
//
// So this one starts where a user starts. It writes a GGUF whose tensors are
// REAL bytes of the staged `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL artifact,
// opens it with the production reader (`GgufFile::Open` resolves a single-shard
// path through `GgufFile::OpenOne`), asks the production residency decision what
// to do with each tensor, and then dots the bytes the READER handed back --
// never a local copy -- on the device. The expected value is the pinned
// oracle's own, so a defect anywhere along that chain moves it.
//
// The residency half runs everywhere, including the CPU CI leg; only the GEMM
// half needs a device.
//
// It is NOT a red-first discriminator and must not be read as one. Before the
// kernels landed, `vt::MatmulBTQuant` on a CUDA queue drained the stream and ran
// the CPU keep-quant kernel over the same unified tensors, which returns the
// oracle's bits exactly -- so this case would have PASSED on the broken tree.
// That is the point being made: correctness gates cannot see a host fallback,
// which is why the capture case below exists and why this one is about the
// CHAIN. Its red comes from the reachability mutation.
TEST_CASE("a REAL GGUF header carries IQ2_XS / IQ4_XS to the CUDA keep-quant GEMM") {
  gguf_test::GgufModelBuilder b;
  // ne0 is the fastest-varying dim: 1024 elements is four whole 256-element
  // super-blocks of either encoding, which is exactly the golden slice.
  b.AddTensor("blk.3.ffn_gate_exps.weight", {1024}, 17,
              std::string(reinterpret_cast<const char*>(vllm_test::kIq2xsGoldenBlocks),
                          sizeof(vllm_test::kIq2xsGoldenBlocks)));
  b.AddTensor("blk.11.ffn_down_exps.weight", {1024}, 23,
              std::string(reinterpret_cast<const char*>(vllm_test::kIq4xsGoldenBlocks),
                          sizeof(vllm_test::kIq4xsGoldenBlocks)));
  const gguf_test::TempFile f(b.Build());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());

  struct Arm {
    const char* tensor;
    uint32_t ggml_type;
    DType dtype;
    size_t block_bytes;
    uint32_t seed;
    const uint32_t* per_block;
  };
  const Arm arms[] = {
      {"blk.3.ffn_gate_exps.weight", 17U, DType::kIQ2_XS, 74U, 0x2247U,
       vllm_test::kIq2xsDotPerBlockBits},
      {"blk.11.ffn_down_exps.weight", 23U, DType::kIQ4_XS, 136U, 0x4247U,
       vllm_test::kIq4xsDotPerBlockBits},
  };

  const bool cuda = HasCuda();
  if (!cuda) MESSAGE("no CUDA backend on this host; only the residency half of this chain runs");

  int64_t dotted = 0;
  for (const Arm& arm : arms) {
    const std::string tensor(arm.tensor);
    CAPTURE(tensor);
    const vllm::GgufTensorInfo& t = g.Get(arm.tensor);
    REQUIRE(t.ggml_type == arm.ggml_type);
    REQUIRE(t.nbytes == 4U * arm.block_bytes);

    // The production residency decision. It must answer keep-quant, because an
    // EXPANDED tensor never reaches a keep-quant GEMM at all -- which is the
    // other way this chain can break, and the way #2245 found it broken.
    // `[out=1, in=1024]` is the `kMatmulWeight` orientation `KeepQuantKDim`
    // reads (shape[1] is K); the real artifact stores these towers stacked as
    // `kStackedExpertWeight` `[E, out, in]`, which takes shape[2] and is
    // already gated in tests/vllm/test_gguf_keep_quant.cpp. Both land on the
    // same encoding rule, and this file exercises the one whose K matches the
    // four super-blocks it then dots.
    const vllm::GgufResidency res = vllm::RouteGgufTensor(
        /*keep_quant=*/true, /*keep_f16=*/true, /*nvfp4_fp4=*/false,
        /*cpu_ref=*/false, vllm::GgufTensorRole::kMatmulWeight,
        arm.ggml_type, {1, 1024}, vllm::platforms::CurrentPlatform().device_type());
    CHECK(res == vllm::GgufResidency::kKeepQuant);

    if (!cuda) continue;
    Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
    Queue gq = gpu.CreateQueue();
    std::vector<float> act(1024);
    MakeDotActivation(1024, arm.seed, act.data());

    void* d_a = gpu.Alloc(act.size() * sizeof(float));
    void* d_w = gpu.Alloc(t.nbytes);
    void* d_o = gpu.Alloc(sizeof(float));
    gpu.Copy(gq, d_a, act.data(), act.size() * sizeof(float));
    gpu.Copy(gq, d_w, t.data, t.nbytes);   // the READER's bytes, not a copy of ours
    gpu.Synchronize(gq);

    for (int blk = 0; blk < 4; ++blk) {
      CAPTURE(blk);
      const float poison = kPoison;
      gpu.Copy(gq, d_o, &poison, sizeof(float));
      gpu.Synchronize(gq);
      Tensor at = DevTensor(static_cast<uint8_t*>(d_a) + blk * 256 * sizeof(float),
                            DType::kF32, {1, 256});
      Tensor wt = DevTensor(static_cast<uint8_t*>(d_w) + blk * arm.block_bytes,
                            arm.dtype, {1, 256});
      Tensor ot = DevTensor(d_o, DType::kF32, {1, 1});
      vt::MatmulBTQuant(gq, ot, at, wt);
      float got = 0.0F;
      gpu.Copy(gq, &got, d_o, sizeof(float));
      gpu.Synchronize(gq);
      REQUIRE(std::isfinite(got));
      CHECK(got != kPoison);
      CAPTURE(got);
      CHECK(FloatBits(got) == arm.per_block[blk]);
      ++dotted;
    }
    gpu.Free(d_a);
    gpu.Free(d_w);
    gpu.Free(d_o);
    gpu.DestroyQueue(gq);
  }
  CAPTURE(dotted);
  CHECK(dotted == (cuda ? 8 : 0));
}

TEST_CASE("CUDA keep-quant runs every kCases dtype INSIDE a stream capture") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; keep-quant capture gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  if (!gpu.SupportsGraphCapture()) {
    MESSAGE("this CUDA backend does not support graph capture; gate skipped");
    return;
  }

  int64_t captured = 0;
  for (const WeightCase& c : kCases) {
    const std::string case_name(c.name);
    CAPTURE(case_name);
    const int64_t k = 8 * c.block_elems;
    const int64_t n = 4;

    std::vector<uint8_t> wq = RandomBlocks(c, n * (k / c.block_elems), 0x5EEDU);
    std::vector<float> a(static_cast<size_t>(k));
    GenerateData(1.0F, a.size(), a.data());

    // A queue per dtype: a capture that fails leaves its stream in an invalid
    // capture state, and sharing one queue would spread the first failure over
    // every dtype after it and make the RED unreadable.
    Queue gq = gpu.CreateQueue();
    void* d_a = gpu.Alloc(a.size() * sizeof(float));
    void* d_w = gpu.Alloc(wq.size());
    void* d_o = gpu.Alloc(static_cast<size_t>(n) * sizeof(float));
    gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
    gpu.Copy(gq, d_w, wq.data(), wq.size());
    gpu.Synchronize(gq);
    Tensor at = DevTensor(d_a, DType::kF32, {1, k});
    Tensor wt = DevTensor(d_w, c.dtype, {n, k});
    Tensor ot = DevTensor(d_o, DType::kF32, {1, n});

    // Warm up OUTSIDE the capture, at the SAME shape, so the per-stream Q8_K
    // scratch already exists and this gate measures the dispatch rather than
    // the allocator. The eager result is also the reference the replay must
    // reproduce bit for bit.
    vt::MatmulBTQuant(gq, ot, at, wt);
    std::vector<float> eager(static_cast<size_t>(n), 0.0F);
    gpu.Copy(gq, eager.data(), d_o, eager.size() * sizeof(float));
    gpu.Synchronize(gq);

    std::vector<float> poison(static_cast<size_t>(n), kPoison);
    gpu.Copy(gq, d_o, poison.data(), poison.size() * sizeof(float));
    gpu.Synchronize(gq);

    std::string threw;
    void* graph = nullptr;
    try {
      gpu.BeginCapture(gq);
      vt::MatmulBTQuant(gq, ot, at, wt);
      graph = gpu.EndCaptureGraph(gq);
      gpu.ReplayGraph(gq, graph);
      gpu.Synchronize(gq);
    } catch (const std::exception& e) {
      threw = e.what();
      try {                       // close a capture the throw left open
        if (graph == nullptr) graph = gpu.EndCaptureGraph(gq);
      } catch (const std::exception&) {
        graph = nullptr;
      }
    }
    CAPTURE(threw);
    CHECK(threw.empty());

    if (threw.empty()) {
      std::vector<float> replayed(static_cast<size_t>(n), 0.0F);
      gpu.Copy(gq, replayed.data(), d_o, replayed.size() * sizeof(float));
      gpu.Synchronize(gq);
      int64_t poisoned = 0, nonfinite = 0, differ = 0;
      for (size_t i = 0; i < replayed.size(); ++i) {
        if (replayed[i] == kPoison) ++poisoned;
        if (!std::isfinite(replayed[i])) ++nonfinite;
        if (FloatBits(replayed[i]) != FloatBits(eager[i])) ++differ;
      }
      CAPTURE(poisoned);
      CAPTURE(nonfinite);
      CAPTURE(differ);
      CHECK(poisoned == 0);
      CHECK(nonfinite == 0);
      CHECK(differ == 0);   // a captured graph replays the eager result exactly
      ++captured;
    }
    if (graph != nullptr) gpu.DestroyGraph(graph);
    gpu.Free(d_a);
    gpu.Free(d_w);
    gpu.Free(d_o);
    gpu.DestroyQueue(gq);
  }
  // Say how many dtypes actually got captured; a loop that captured none would
  // otherwise print the same SUCCESS as one that captured all of them.
  CAPTURE(captured);
  CHECK(captured == static_cast<int64_t>(std::size(kCases)));
}

#ifdef VLLM_CPP_CUDA
// Brick 12 (ds4-gap "launch consolidation") A/B gate — PAIRED Q8_0 decode GEMV. The
// paired kernel (one launch, two weights, one shared activation) must be BYTE-IDENTICAL
// to running the plain Q8_0 GEMV twice (via vt::MatmulBTQuant). RED-first: a wrong
// activation-load factoring, a swapped weight row-stride, or a reduction re-order would
// diverge the two outputs (this is the wq_a+wkv / shared-gate+up decode fusion).
// ─── QUANT-CUDA-KEEPQUANT-32B (#2419): the two things `kCases` cannot say ────
//
// 1. THE LANE STRIDE. Every `kCases` gate runs `k = 8 * block_elems`, which for a
//    32-element encoding is nb = 8 blocks. The dense and grouped kernels walk
//    `for (b = lane; b < nb; b += 32)`, so at nb = 8 only lanes 0-7 ever enter
//    the body and NO LANE TAKES THE LOOP TWICE. A defect in the stride, in the
//    per-lane block address, or in the warp reduce's treatment of idle lanes is
//    byte-for-byte invisible at that shape. This case runs nb = 40 > 32, where
//    lanes 0-7 take the loop twice and lanes 8-31 take it once.
//
// 2. THE RELEASED GEOMETRY. K = 640 is the shipped Qwen3.8-Flash-Next
//    `expert_feed_forward_length`, and it is the reason this lane exists at all:
//    640 is 20 whole 32-element blocks and is NOT a whole number of 256-element
//    super-blocks, so the Q8_K seams refuse that shape BY NAME whatever the
//    dtype carried. A gate that only ever ran K % 256 == 0 cannot tell that
//    refusal apart from a missing `case`.
//
// What this case observes: that the device numbers track the CPU arm at shapes
// the other gates do not reach. What it CANNOT observe, exactly like every value
// gate in this file: whether the device ran. The capture case above is the only
// discriminator for that, and this one is deliberately not dressed up as a
// second.
TEST_CASE("32-block keep-quant lane: crosses the lane stride, and takes K = 640") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; 32-block lane gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // The three dtypes the 32-block lane serves, named here rather than filtered
  // out of `kCases`: a filter that matched nothing would leave this case running
  // zero shapes and still printing SUCCESS.
  const WeightCase k32[] = {
      {DType::kIQ4_NL, 32, 18, 0, -1, "iq4_nl"},
      {DType::kQ5_0, 32, 22, 0, -1, "q5_0"},
      {DType::kQ4_0, 32, 18, 0, -1, "q4_0"},
  };
  // 640 = the released expert row (20 blocks, and NOT a multiple of 256).
  // 1280 = 40 blocks, which is the first shape past the 32-lane stride.
  const int64_t kKs[] = {640, 1280};

  int64_t combos = 0;
  for (const WeightCase& c : k32) {
    // Worst margin over this dtype's shapes. A gate that only says "within
    // tolerance" cannot be read for HOW MUCH room it had, so a later tightening
    // or a slow drift has nothing to be compared against. Reported below.
    double worst_ratio = 0.0, worst_diff = 0.0;
    for (int64_t k : kKs) {
      const int64_t nb = k / c.block_elems;
      for (int64_t m : {int64_t{1}, int64_t{5}}) {
        for (int64_t n : {int64_t{1}, int64_t{33}}) {
          const std::string case_name(c.name);
          CAPTURE(case_name);
          CAPTURE(k);
          CAPTURE(nb);
          CAPTURE(m);
          CAPTURE(n);
          ++combos;

          std::vector<uint8_t> wq = RandomBlocks(c, n * nb, 0x5EEDU);
          std::vector<float> a(static_cast<size_t>(m * k));
          GenerateData(1.0F, a.size(), a.data());

          std::vector<float> cpu_out(static_cast<size_t>(m * n), 0.0F);
          {
            Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {m, k});
            Tensor bt = Tensor::Contiguous(wq.data(), DType::kF32, Cpu(), {n, k});
            bt.dtype = c.dtype;
            Tensor ot = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {m, n});
            vt::MatmulBTQuant(cq, ot, at, bt);
          }

          if (PascalQ5Oracle(c.dtype, m)) Q5GpuOracle(cpu_out, a, wq, n, nb);

          void* d_a = gpu.Alloc(a.size() * sizeof(float));
          void* d_w = gpu.Alloc(wq.size());
          void* d_o = gpu.Alloc(static_cast<size_t>(m * n) * sizeof(float));
          gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
          gpu.Copy(gq, d_w, wq.data(), wq.size());
          // Poison, so a kernel that never launched is a FAILURE and not a
          // buffer that happened to hold the right zeros.
          std::vector<float> poison(static_cast<size_t>(m * n), kPoison);
          gpu.Copy(gq, d_o, poison.data(), poison.size() * sizeof(float));
          Tensor at = DevTensor(d_a, DType::kF32, {m, k});
          Tensor bt = DevTensor(d_w, c.dtype, {n, k});
          Tensor ot = DevTensor(d_o, DType::kF32, {m, n});
          vt::MatmulBTQuant(gq, ot, at, bt);
          std::vector<float> cuda_out(static_cast<size_t>(m * n), 0.0F);
          gpu.Copy(gq, cuda_out.data(), d_o, cuda_out.size() * sizeof(float));
          gpu.Synchronize(gq);
          gpu.Free(d_a);
          gpu.Free(d_w);
          gpu.Free(d_o);

          double max_abs_diff = 0, max_abs_cpu = 0, num = 0, den = 0;
          int64_t poisoned = 0;
          for (size_t i = 0; i < cuda_out.size(); ++i) {
            const double got = cuda_out[i], cpu = cpu_out[i];
            if (cuda_out[i] == kPoison) ++poisoned;
            REQUIRE(std::isfinite(got));
            max_abs_diff = std::max(max_abs_diff, std::abs(got - cpu));
            max_abs_cpu = std::max(max_abs_cpu, std::abs(cpu));
            num += (got - cpu) * (got - cpu);
            den += cpu * cpu;
          }
          const double nmse = den > 0 ? num / den : num;
          // DERIVED, not fitted to this fixture. Both sides accumulate nb f32
          // terms; they differ only in ASSOCIATION -- the CPU arm sums blocks
          // sequentially, the kernel sums them per lane and then down a 32-wide
          // warp tree. Worst-case f32 accumulation error over nb terms is
          // nb * eps relative, so the bound is 8 * nb * eps with the 8 as
          // headroom. It GROWS with nb, which is the direction the error grows;
          // a constant here would tighten as K rose and start failing on the
          // long rows this lane exists to serve.
          const double eps = 1.1920929e-7;  // FLT_EPSILON
          const double tol = 8.0 * static_cast<double>(nb) * eps;
          const double bound = tol * max_abs_cpu;
          CAPTURE(max_abs_diff);
          CAPTURE(max_abs_cpu);
          CAPTURE(bound);
          CAPTURE(nmse);
          CAPTURE(poisoned);
          CHECK(poisoned == 0);
          CHECK(max_abs_diff <= bound);
          CHECK(nmse <= kMaxNmseVsCpu);
          if (bound > 0 && max_abs_diff / bound > worst_ratio) worst_ratio = max_abs_diff / bound;
          if (max_abs_diff > worst_diff) worst_diff = max_abs_diff;
        }
      }
    }
    // ALWAYS printed, unlike CAPTURE, which doctest emits only for a FAILING
    // assertion -- so a green run reported no number at all and the measured
    // device-vs-CPU margin could not be quoted from a passing gate.
    // std::string, NOT the bare `c.name`: doctest stringifies a `const char*`
    // as a BOOL, so the first run of this line labelled all three dtypes "1"
    // and the margins could only have been attributed by trusting loop order.
    MESSAGE("32-block lane " << std::string(c.name) << ": worst device-vs-CPU max|diff| = "
                             << worst_diff << ", worst fraction of the derived bound = "
                             << worst_ratio);
  }
  // A loop that ran nothing prints the same SUCCESS as one that ran everything.
  CAPTURE(combos);
  CHECK(combos == static_cast<int64_t>(std::size(k32) * std::size(kKs) * 2 * 2));
  CHECK(combos > 0);
  gpu.DestroyQueue(gq);
}

// The grouped 32-block lane assigned ONE WARP PER OUTPUT ELEMENT, so at the
// released qwen4_exp shape (hidden 2560 -> nb = 80) its inner loop ran
// `for (b = lane; b < 80; b += 32)` -- 2.50 iterations per lane -- and then paid a
// five-step warp reduction to emit ONE float. It measured 41.6 GB/s, 15.2% of
// peak, the lowest row in the decode bandwidth table
// (ISSUE-LOCAL-01M2TZ9AE416FW4GH23V09783Q). VT_V4_W32_COLS=2|4|8 selects
// QuantDotGemmGrouped32MultiColKernel, which gives one warp C CONSECUTIVE output
// columns against ONE broadcast activation row.
//
// BYTE-IDENTITY is the RED-first anchor, and here it is genuinely reachable
// rather than aspirational: every output element keeps its OWN accumulator and
// its OWN ascending block order (b = lane, lane+32, ...), and the reduction tree
// is the same five __shfl_down_sync steps. Only the warp->output map changes, so
// NOTHING is re-associated. A wrong group stride, a mis-sized tail, or a
// re-associated accumulate all diverge from the width-1 arm and fail hard here.
//
// The shapes are chosen for the tail, which is where a multi-column map breaks:
// n = 5 and n = 13 divide by NONE of 2, 4, 8; n = 1 is a single-column row whose
// group is short at every width; n = 64 divides all three. nb = 20 is the
// released expert row (K = 640) and nb = 80 is the released hidden (K = 2560),
// the shape this change exists for. Both broadcast polarities run, because
// `bcast` picks the activation row inside the loop and a multi-column warp reads
// that row C times.
TEST_CASE("32-block grouped multi-COLUMN GEMV == one warp per output (byte-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; 32-block multi-column gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();

  const WeightCase k32[] = {
      {DType::kIQ4_NL, 32, 18, 0, -1, "iq4_nl"},
      {DType::kQ5_0, 32, 22, 0, -1, "q5_0"},
      {DType::kQ4_0, 32, 18, 0, -1, "q4_0"},
  };
  constexpr int64_t kE = 5;
  constexpr int64_t kP = 3;

  int64_t combos = 0;
  for (const WeightCase& c : k32) {
    const std::string case_name(c.name);
    for (int64_t nb : {int64_t{20}, int64_t{80}}) {
      const int64_t k = nb * c.block_elems;
      for (int64_t n : {int64_t{1}, int64_t{5}, int64_t{13}, int64_t{64}}) {
        for (bool bcast : {false, true}) {
          CAPTURE(case_name);
          CAPTURE(nb);
          CAPTURE(n);
          CAPTURE(bcast);
          ++combos;

          const int64_t arows = bcast ? 1 : kP;
          std::vector<uint8_t> wq = RandomBlocks(c, kE * n * nb, 0xC015U);
          std::vector<float> a(static_cast<size_t>(arows * k));
          GenerateData(1.0F, a.size(), a.data());
          std::vector<int32_t> ids = ExpertIds(kP, kE);
          const size_t outn = static_cast<size_t>(kP * n);

          void* d_a = gpu.Alloc(a.size() * sizeof(float));
          void* d_w = gpu.Alloc(wq.size());
          void* d_e = gpu.Alloc(ids.size() * sizeof(int32_t));
          gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
          gpu.Copy(gq, d_w, wq.data(), wq.size());
          gpu.Copy(gq, d_e, ids.data(), ids.size() * sizeof(int32_t));
          gpu.Synchronize(gq);
          Tensor at = DevTensor(d_a, DType::kF32, {arows, k});
          Tensor wt = DevTensor(d_w, c.dtype, {kE * n, k});
          Tensor et = DevTensor(d_e, DType::kI32, {kP});

          auto run = [&](const char* flag, const char* warps) {
            setenv("VT_V4_W32_COLS", flag, 1);
            setenv("VT_V4_W32_WARPS", warps, 1);
            void* d_o = gpu.Alloc(outn * sizeof(float));
            // Poison, so an arm whose grid computed to zero blocks -- the exact
            // way a wrong `njg` fails -- reads as a FAILURE and not as a buffer
            // that happened to agree.
            const std::vector<float> poison(outn, kPoison);
            gpu.Copy(gq, d_o, poison.data(), poison.size() * sizeof(float));
            Tensor ot = DevTensor(d_o, DType::kF32, {kP, n});
            vt::MatmulBTQuantGrouped(gq, ot, at, wt, et);
            std::vector<float> out(outn, 0.0F);
            gpu.Copy(gq, out.data(), d_o, out.size() * sizeof(float));
            gpu.Synchronize(gq);
            gpu.Free(d_o);
            return out;
          };
          // The baseline is the PRE-CHANGE launch exactly: one warp per output,
          // four warps per block.
          const std::vector<float> w1 = run("1", "4");
          const std::vector<float> w2 = run("2", "4");
          const std::vector<float> w4 = run("4", "4");
          const std::vector<float> w8 = run("8", "4");
          // The warps-per-block axis, which changes only how many warps share a
          // block and must therefore move nothing. Run at BOTH ends: width 1 (the
          // untouched kernel) and width 4 (the new one, where a tail group also
          // has to survive a different grid).
          const std::vector<float> b1 = run("1", "1");
          const std::vector<float> b8 = run("4", "8");
          unsetenv("VT_V4_W32_COLS");
          unsetenv("VT_V4_W32_WARPS");

          for (size_t i = 0; i < outn; ++i) {
            REQUIRE(w1[i] != kPoison);
            REQUIRE(w2[i] != kPoison);
            REQUIRE(w4[i] != kPoison);
            REQUIRE(w8[i] != kPoison);
            REQUIRE(b1[i] != kPoison);
            REQUIRE(b8[i] != kPoison);
            REQUIRE(std::isfinite(w1[i]));
            CHECK(w2[i] == w1[i]);  // byte-identical, NOT a tolerance
            CHECK(w4[i] == w1[i]);
            CHECK(w8[i] == w1[i]);
            CHECK(b1[i] == w1[i]);
            CHECK(b8[i] == w1[i]);
          }
          gpu.Free(d_a);
          gpu.Free(d_w);
          gpu.Free(d_e);
        }
      }
    }
  }
  // doctest prints SUCCESS for a loop that ran nothing. Say how many it ran.
  CAPTURE(combos);
  CHECK(combos == static_cast<int64_t>(std::size(k32)) * 2 * 4 * 2);
  CHECK(combos > 0);
  gpu.DestroyQueue(gq);
}

TEST_CASE("Brick 12: CUDA Q8_0 PAIR == two separate matmuls (bit-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 pair gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  cudaStream_t s = static_cast<cudaStream_t>(gq.handle);
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const int64_t k = 8 * c.block_elems;  // 8 blocks (256), a model-ish decode K
  const int64_t nbk = k / c.block_elems;
  // (n0,n1): unequal out-dims (the real q_a=qlr vs kv_a=hd; gate=up=mi), and 1-row edge.
  for (auto nn : {std::pair<int64_t, int64_t>{16, 16}, {16, 7}, {7, 16}, {1, 24}, {24, 1}}) {
    const int64_t n0 = nn.first, n1 = nn.second;
    CAPTURE(n0);
    CAPTURE(n1);
    std::vector<uint8_t> w0 = RandomBlocks(c, n0 * nbk, 0xA11CEU);
    std::vector<uint8_t> w1 = RandomBlocks(c, n1 * nbk, 0xB0BAU);
    std::vector<float> a(static_cast<size_t>(k));
    GenerateData(1.0F, a.size(), a.data());

    void* d_a = gpu.Alloc(a.size() * sizeof(float));
    void* d_w0 = gpu.Alloc(w0.size());
    void* d_w1 = gpu.Alloc(w1.size());
    void* d_o0 = gpu.Alloc(static_cast<size_t>(n0) * sizeof(float));
    void* d_o1 = gpu.Alloc(static_cast<size_t>(n1) * sizeof(float));
    void* d_r0 = gpu.Alloc(static_cast<size_t>(n0) * sizeof(float));
    void* d_r1 = gpu.Alloc(static_cast<size_t>(n1) * sizeof(float));
    gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
    gpu.Copy(gq, d_w0, w0.data(), w0.size());
    gpu.Copy(gq, d_w1, w1.data(), w1.size());

    Tensor at = DevTensor(d_a, DType::kF32, {1, k});
    Tensor w0t = DevTensor(d_w0, DType::kQ8_0, {n0, k});
    Tensor w1t = DevTensor(d_w1, DType::kQ8_0, {n1, k});
    Tensor o0 = DevTensor(d_o0, DType::kF32, {1, n0});
    Tensor o1 = DevTensor(d_o1, DType::kF32, {1, n1});
    Tensor r0 = DevTensor(d_r0, DType::kF32, {1, n0});
    Tensor r1 = DevTensor(d_r1, DType::kF32, {1, n1});

    vt::cuda::MatmulQ8_0PairCuda(o0, o1, at, w0t, w1t, s);  // ONE paired launch
    vt::MatmulBTQuant(gq, r0, at, w0t);                     // reference: two separate launches
    vt::MatmulBTQuant(gq, r1, at, w1t);

    std::vector<float> got0(static_cast<size_t>(n0)), got1(static_cast<size_t>(n1));
    std::vector<float> ref0(static_cast<size_t>(n0)), ref1(static_cast<size_t>(n1));
    gpu.Copy(gq, got0.data(), d_o0, got0.size() * sizeof(float));
    gpu.Copy(gq, got1.data(), d_o1, got1.size() * sizeof(float));
    gpu.Copy(gq, ref0.data(), d_r0, ref0.size() * sizeof(float));
    gpu.Copy(gq, ref1.data(), d_r1, ref1.size() * sizeof(float));
    gpu.Synchronize(gq);
    gpu.Free(d_a); gpu.Free(d_w0); gpu.Free(d_w1);
    gpu.Free(d_o0); gpu.Free(d_o1); gpu.Free(d_r0); gpu.Free(d_r1);

    CHECK(std::memcmp(got0.data(), ref0.data(), got0.size() * sizeof(float)) == 0);
    CHECK(std::memcmp(got1.data(), ref1.data(), got1.size() * sizeof(float)) == 0);
  }
  gpu.DestroyQueue(gq);
}

// Brick 12 (ds4-gap "row-split consolidation") A/B gate — BLOCK-DIAGONAL grouped Q8_0
// output-LoRA GEMV. The single consolidated launch (all ng groups) must be
// BYTE-IDENTICAL to the ng separate per-group slice GEMVs (each over a disjoint ipg-wide
// activation slice + its olr weight rows). RED-first: a wrong group activation offset or
// a mis-computed rows-per-group would diverge. This is the resident wo_a fusion (the 344
// launches/step, 53% of the 646).
TEST_CASE("Brick 12: CUDA Q8_0 block-diagonal o-LoRA == per-group loop (bit-identical)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; Q8_0 group-diagonal gate skipped");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  cudaStream_t s = static_cast<cudaStream_t>(gq.handle);
  const WeightCase c = {DType::kQ8_0, 32, 34, 0, -1, "q8_0"};
  const size_t row_bytes = static_cast<size_t>(1) * c.block_bytes;  // per 32-block
  // (ng, olr, ipg): ipg must be a multiple of 32; models use ng=8.
  struct Shape { int64_t ng, olr, ipg; };
  for (Shape sh : {Shape{4, 8, 64}, Shape{8, 16, 96}, Shape{2, 7, 32}}) {
    const int64_t ng = sh.ng, olr = sh.olr, ipg = sh.ipg;
    const int64_t nb_g = ipg / c.block_elems;
    const int64_t total_rows = ng * olr;
    CAPTURE(ng);
    CAPTURE(olr);
    CAPTURE(ipg);
    std::vector<uint8_t> w = RandomBlocks(c, total_rows * nb_g, 0xC0FFEEU);
    std::vector<float> o(static_cast<size_t>(ng * ipg));
    GenerateData(0.5F, o.size(), o.data());

    void* d_o = gpu.Alloc(o.size() * sizeof(float));
    void* d_w = gpu.Alloc(w.size());
    void* d_z = gpu.Alloc(static_cast<size_t>(total_rows) * sizeof(float));   // consolidated
    void* d_zr = gpu.Alloc(static_cast<size_t>(total_rows) * sizeof(float));  // per-group ref
    gpu.Copy(gq, d_o, o.data(), o.size() * sizeof(float));
    gpu.Copy(gq, d_w, w.data(), w.size());

    Tensor ot = DevTensor(d_o, DType::kF32, {1, ng * ipg});
    Tensor wt = DevTensor(d_w, DType::kQ8_0, {total_rows, ipg});
    Tensor zt = DevTensor(d_z, DType::kF32, {1, total_rows});
    vt::cuda::MatmulQ8_0GroupDiagCuda(zt, ot, wt, ng, s);  // ONE block-diagonal launch

    // reference: ng separate slice GEMVs (mirrors GemmRowSliceInto in the model forward)
    for (int64_t gp = 0; gp < ng; ++gp) {
      Tensor a_slice =
          DevTensor(static_cast<float*>(d_o) + gp * ipg, DType::kF32, {1, ipg});
      Tensor w_slice = DevTensor(static_cast<uint8_t*>(d_w) +
                                     static_cast<size_t>(gp * olr) * nb_g * row_bytes,
                                 DType::kQ8_0, {olr, ipg});
      Tensor z_slice =
          DevTensor(static_cast<float*>(d_zr) + gp * olr, DType::kF32, {1, olr});
      vt::MatmulBTQuant(gq, z_slice, a_slice, w_slice);
    }

    std::vector<float> got(static_cast<size_t>(total_rows)), ref(static_cast<size_t>(total_rows));
    gpu.Copy(gq, got.data(), d_z, got.size() * sizeof(float));
    gpu.Copy(gq, ref.data(), d_zr, ref.size() * sizeof(float));
    gpu.Synchronize(gq);
    gpu.Free(d_o); gpu.Free(d_w); gpu.Free(d_z); gpu.Free(d_zr);

    CHECK(std::memcmp(got.data(), ref.data(), got.size() * sizeof(float)) == 0);
  }
  gpu.DestroyQueue(gq);
}

// The device-codebook drift seal cuda_quant_iq_tables.cuh has claimed since it
// landed. `src/vt/cuda/cuda_quant_iq_tables.cuh` is a hand transcription of
// `src/vt/cpu/cpu_quant_iq_tables.h`, and nothing compared the two: the CPU
// tests digest the HOST symbols, and no gate reads `vt::cuda::d_iq1s_grid` at
// all. The value gates in this file only see a drifted entry if a weight sample
// happens to address it, which is chance, not coverage: replaying the
// std::mt19937(0x5EED) stream those gates use, 266 of the 2048 d_iq1s_grid
// entries (13.0 %) are never addressed, so drifting entry 0 is caught while
// drifting entry 3 is green at 150032/150032 assertions.
//
// Byte equality is the right assertion because it is the actual contract: the
// device tables are DERIVED from the CPU ones, deliberately in the same u64
// layout, so any difference at all is a transcription defect.
TEST_CASE("CUDA device codebooks == the CPU host tables (byte-exact)") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; device-codebook seal skipped");
    return;
  }
  vt::GetBackend(DeviceType::kCUDA);  // ensure a live CUDA context for the copy
  auto snap = std::make_unique<vt::cuda::IqTableSnapshot>();
  vt::cuda::SnapshotIqTablesFromDevice(snap.get());

  int sealed = 0;
  // `name` is a const& parameter, so the caller's temporary outlives the CAPTURE;
  // a `CAPTURE(std::string(...))` would dangle, because doctest reads the
  // captured expression at FAILURE time, not at capture time.
  auto seal = [&](const std::string& name, const void* dev, const void* host, size_t bytes) {
    CAPTURE(name);
    CHECK(std::memcmp(dev, host, bytes) == 0);
    ++sealed;
  };
  seal("d_kmask_iq2xs", snap->kmask_iq2xs, vt::cpu::kKmaskIq2xs, sizeof(snap->kmask_iq2xs));
  seal("d_ksigns_iq2xs", snap->ksigns_iq2xs, vt::cpu::kKsignsIq2xs, sizeof(snap->ksigns_iq2xs));
  seal("d_iq1s_grid", snap->iq1s_grid, vt::cpu::kIq1sGrid, sizeof(snap->iq1s_grid));
  seal("d_iq1xxxs_grid", snap->iq1xxxs_grid, vt::cpu::kIq1xxxsGrid, sizeof(snap->iq1xxxs_grid));
  seal("d_iq2xxs_grid", snap->iq2xxs_grid, vt::cpu::kIq2xxsGrid, sizeof(snap->iq2xxs_grid));
  seal("d_iq3xxs_grid", snap->iq3xxs_grid, vt::cpu::kIq3xxsGrid, sizeof(snap->iq3xxs_grid));
  seal("d_iq2s_grid", snap->iq2s_grid, vt::cpu::kIq2sGrid, sizeof(snap->iq2s_grid));
  // QUANT-CUDA-IQ4XS-IQ2XS. d_iq2xs_grid is the one table here a SIBLING could
  // be mistaken for and still index in range: iq2xxs (256), iq2xs (512) and
  // iq2s (1024) all hold 8 bytes per entry, so a kernel reading the wrong one
  // returns a plausible magnitude rather than misbehaving.
  seal("d_iq2xs_grid", snap->iq2xs_grid, vt::cpu::kIq2xsGrid, sizeof(snap->iq2xs_grid));
  seal("d_kvalues_iq4nl", snap->kvalues_iq4nl, vt::cpu::kValuesIq4nl,
       sizeof(snap->kvalues_iq4nl));
  seal("d_kvalues_mxfp4", snap->kvalues_mxfp4, vt::cpu::kValuesMxfp4,
       sizeof(snap->kvalues_mxfp4));
  // Say how many tables were examined: a seal that compared nothing would
  // otherwise print the same "SUCCESS!" as one that compared all eight.
  CAPTURE(sealed);
  CHECK(sealed == 10);
  // And the extents themselves, so a snapshot that silently shrank is a failure
  // rather than a shorter memcmp that trivially passes.
  CHECK(sizeof(snap->iq1s_grid) == sizeof(vt::cpu::kIq1sGrid));
  CHECK(sizeof(snap->iq1xxxs_grid) == sizeof(vt::cpu::kIq1xxxsGrid));
  CHECK(sizeof(snap->iq2s_grid) == sizeof(vt::cpu::kIq2sGrid));
  CHECK(sizeof(snap->iq2xs_grid) == sizeof(vt::cpu::kIq2xsGrid));
  CHECK(sizeof(snap->kvalues_iq4nl) == sizeof(vt::cpu::kValuesIq4nl));
}
#endif  // VLLM_CPP_CUDA

// ─── GLM-5.3-Flash's PUBLISHED geometry, on its four real encoding triples ───
//
// Every grouped case above runs at E<=4, n<=16, k=8 blocks. That is the right
// shape for a codebook gate and it is the wrong shape for a residency one: the
// numbers it exercises are four to eight orders of magnitude below the ones
// this model actually presents, so nothing above can see an index or an offset
// that only misbehaves when the tower is large.
//
// `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL presents `n_routed_experts` 288,
// `moe_intermediate_size` 2048 and `hidden_size` 4096, so ONE routed-expert
// tower is 288 * 2048 * 4096 = 2,415,919,104 elements. **That is past INT32**,
// and it is past it by element count while every BYTE offset in the same tower
// (287 * 2048 * 1184 = 696 MB for IQ2_XS) still fits — so a 32-bit element
// count truncates while a 32-bit byte offset does not, and only the real
// geometry separates them.
//
// THE FOUR TRIPLES ARE THE CHECKPOINT'S, read off its GGUF tensor headers
// across all four shards on 2026-08-30 and recorded in
// `.agents/specs/glm5-next-flash.md` §W9a:
//   (IQ2_XS, IQ2_XS, IQ3_XXS) x39   (IQ2_XS, IQ2_XS, IQ4_XS)  x2
//   (IQ3_XXS, IQ3_XXS, IQ4_XS) x1   (Q2_K,   Q2_K,   Q3_K)    x1
// gate and up share a dtype in all 43 sparse blocks, which is what lets the
// FUSED seam represent this checkpoint at all; the down tower is separately
// encoded and runs through `vt::MatmulBTQuantGrouped`.
//
// P = 16 is two tokens at the published `num_experts_per_tok` of 8. Two tokens
// rather than one, because a single token cannot distinguish an activation
// gather that reads the wrong row.
namespace {

const WeightCase& CaseFor(DType dt) {
  for (const WeightCase& c : kCases)
    if (c.dtype == dt) return c;
  FAIL("no WeightCase for the requested dtype");
  return kCases[0];
}

struct Glm5Triple {
  DType gate_up;
  DType down;
  const char* what;
};
const Glm5Triple kGlm5Triples[] = {
    {DType::kIQ2_XS, DType::kIQ3_XXS, "blk.3-44 (39 layers)"},
    {DType::kIQ2_XS, DType::kIQ4_XS, "blk.12, blk.44"},
    {DType::kIQ3_XXS, DType::kIQ4_XS, "blk.11"},
    {DType::kQ2_K, DType::kQ3_K, "blk.45"},
};

}  // namespace

TEST_CASE("CUDA grouped keep-quant at GLM-5.3-Flash's published MoE geometry") {
  if (!HasCuda()) {
    MESSAGE("no CUDA backend on this host; GLM-5.3-Flash geometry gate skipped");
    return;
  }
  // The published constants, named rather than spelled inline so a reader can
  // check them against configuration_glm5_next.py and the GGUF kv.
  constexpr int64_t kE = 288;     // n_routed_experts
  constexpr int64_t kI = 2048;    // moe_intermediate_size
  constexpr int64_t kH = 4096;    // hidden_size
  constexpr int64_t kTopK = 8;    // num_experts_per_tok
  constexpr int64_t kTokens = 2;
  constexpr int64_t kP = kTokens * kTopK;
  static_assert(static_cast<int64_t>(kE) * kI * kH > 2147483647LL,
                "the point of this case is a tower past INT32 elements");

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  int64_t triples = 0;
  for (const Glm5Triple& tr : kGlm5Triples) {
    const std::string what(tr.what);
    CAPTURE(what);
    const WeightCase& cg = CaseFor(tr.gate_up);
    const WeightCase& cd = CaseFor(tr.down);
    CAPTURE(cg.name);
    CAPTURE(cd.name);
    ++triples;

    // gate/up are [E * I, H]; down is [E * H, I]. Both are the stacked towers
    // `LoadStackedExperts` writes and `AdmitMoeQuantBanks` views.
    const int64_t gu_rows = kE * kI;
    const int64_t dn_rows = kE * kH;
    std::vector<uint8_t> gw = RandomBlocks(cg, gu_rows * (kH / cg.block_elems), 0x51EEDU);
    std::vector<uint8_t> uw = RandomBlocks(cg, gu_rows * (kH / cg.block_elems), 0x52EEDU);
    std::vector<uint8_t> dw = RandomBlocks(cd, dn_rows * (kI / cd.block_elems), 0x53EEDU);

    // Distinct experts across the 16 slots, spread over the full 288 so the
    // top of the range is exercised and not just the first few rows.
    std::vector<int32_t> ids(static_cast<size_t>(kP));
    for (int64_t p = 0; p < kP; ++p)
      ids[static_cast<size_t>(p)] = static_cast<int32_t>((p * 37 + 5) % kE);
    ids[static_cast<size_t>(kP - 1)] = static_cast<int32_t>(kE - 1);  // the last row

    std::vector<float> a(static_cast<size_t>(kP * kH));
    GenerateData(1.0F, a.size(), a.data());

    const size_t midn = static_cast<size_t>(kP * kI);
    const size_t outn = static_cast<size_t>(kP * kH);
    constexpr float kLimit = 10.0F;  // config.swiglu_limit on this checkpoint

    std::vector<float> cpu_mid(midn, kPoison);
    std::vector<float> cpu_out(outn, kPoison);
    {
      Tensor at = Tensor::Contiguous(a.data(), DType::kF32, Cpu(), {kP, kH});
      Tensor gt = Tensor::Contiguous(gw.data(), DType::kF32, Cpu(), {gu_rows, kH});
      Tensor ut = Tensor::Contiguous(uw.data(), DType::kF32, Cpu(), {gu_rows, kH});
      gt.dtype = cg.dtype;
      ut.dtype = cg.dtype;
      Tensor et = Tensor::Contiguous(ids.data(), DType::kI32, Cpu(), {kP});
      Tensor mt = Tensor::Contiguous(cpu_mid.data(), DType::kF32, Cpu(), {kP, kI});
      vt::MoeGateUpSwiGLUGrouped(cq, mt, at, gt, ut, et, kLimit);
      Tensor dt = Tensor::Contiguous(dw.data(), DType::kF32, Cpu(), {dn_rows, kI});
      dt.dtype = cd.dtype;
      Tensor ot = Tensor::Contiguous(cpu_out.data(), DType::kF32, Cpu(), {kP, kH});
      vt::MatmulBTQuantGrouped(cq, ot, mt, dt, et);
    }

    void* d_a = gpu.Alloc(a.size() * sizeof(float));
    void* d_g = gpu.Alloc(gw.size());
    void* d_u = gpu.Alloc(uw.size());
    void* d_d = gpu.Alloc(dw.size());
    void* d_e = gpu.Alloc(ids.size() * sizeof(int32_t));
    void* d_m = gpu.Alloc(midn * sizeof(float));
    void* d_o = gpu.Alloc(outn * sizeof(float));
    const std::vector<float> poison_mid(midn, kPoison);
    const std::vector<float> poison_out(outn, kPoison);
    gpu.Copy(gq, d_a, a.data(), a.size() * sizeof(float));
    gpu.Copy(gq, d_g, gw.data(), gw.size());
    gpu.Copy(gq, d_u, uw.data(), uw.size());
    gpu.Copy(gq, d_d, dw.data(), dw.size());
    gpu.Copy(gq, d_e, ids.data(), ids.size() * sizeof(int32_t));
    gpu.Copy(gq, d_m, poison_mid.data(), poison_mid.size() * sizeof(float));
    gpu.Copy(gq, d_o, poison_out.data(), poison_out.size() * sizeof(float));
    gpu.Synchronize(gq);

    Tensor at = DevTensor(d_a, DType::kF32, {kP, kH});
    Tensor gt = DevTensor(d_g, cg.dtype, {gu_rows, kH});
    Tensor ut = DevTensor(d_u, cg.dtype, {gu_rows, kH});
    Tensor dt = DevTensor(d_d, cd.dtype, {dn_rows, kI});
    Tensor et = DevTensor(d_e, DType::kI32, {kP});
    Tensor mt = DevTensor(d_m, DType::kF32, {kP, kI});
    Tensor ot = DevTensor(d_o, DType::kF32, {kP, kH});
    vt::MoeGateUpSwiGLUGrouped(gq, mt, at, gt, ut, et, kLimit);
    vt::MatmulBTQuantGrouped(gq, ot, mt, dt, et);

    std::vector<float> cuda_mid(midn, 0.0F);
    std::vector<float> cuda_out(outn, 0.0F);
    gpu.Copy(gq, cuda_mid.data(), d_m, cuda_mid.size() * sizeof(float));
    gpu.Copy(gq, cuda_out.data(), d_o, cuda_out.size() * sizeof(float));
    gpu.Synchronize(gq);
    gpu.Free(d_a); gpu.Free(d_g); gpu.Free(d_u); gpu.Free(d_d);
    gpu.Free(d_e); gpu.Free(d_m); gpu.Free(d_o);

    const GroupedVerdict vm = Compare(cuda_mid, cpu_mid);
    const GroupedVerdict vo = Compare(cuda_out, cpu_out);
    CAPTURE(vm.nmse);
    CAPTURE(vm.poisoned);
    CAPTURE(vm.nonfinite);
    CAPTURE(vo.nmse);
    CAPTURE(vo.poisoned);
    CAPTURE(vo.nonfinite);
    // Both halves, because a fused seam that silently launched nothing would
    // leave the poison in `mid` and the down GEMM would then faithfully
    // propagate a poisoned input into a finite-looking output.
    CHECK(vm.poisoned == 0);
    CHECK(vo.poisoned == 0);
    CHECK(vm.nonfinite == 0);
    CHECK(vo.nonfinite == 0);
    CHECK(vm.nmse <= kMaxNmseVsCpu);
    CHECK(vo.nmse <= kMaxNmseVsCpu);
  }
  CHECK(triples == static_cast<int64_t>(std::size(kGlm5Triples)));
  gpu.DestroyQueue(gq);
}

TEST_CASE("Pascal Q5_0 decode uses the GPU oracle original activation sum") {
#ifdef VLLM_CPP_CUDA
  if (!HasCuda()) return;
  cudaDeviceProp prop{};
  REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
  if (prop.major != 6) return;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue q = gpu.CreateQueue();
  // llama-cpp 62318cd5 quantize.cu:64-101 and vecdotq.cuh:175-203.
  // Zero unsigned weight quants isolate the original-sum correction term.
  std::vector<float> a(32);
  for (int i = 0; i < 32; ++i) a[i] = (i == 0) ? 1.0f : 0.013f * (i % 5 - 1);
  std::vector<float> tree = a;
  for (int mask : {16, 8, 4, 2, 1}) {
    const auto prior = tree;
    for (int i = 0; i < 32; ++i) tree[i] = prior[i] + prior[i ^ mask];
  }
  const float expected = -16.0f * vt::F16ToF32(vt::F32ToF16(tree[0]));
  std::vector<uint8_t> w(22, 0);
  const uint16_t one = vt::F32ToF16(1.0f);
  std::memcpy(w.data(), &one, 2);
  void* da = gpu.Alloc(128);
  void* dw = gpu.Alloc(22);
  void* dout = gpu.Alloc(4);
  gpu.Copy(q, da, a.data(), 128);
  gpu.Copy(q, dw, w.data(), 22);
  Tensor at = DevTensor(da, DType::kF32, {1, 32});
  Tensor wt = DevTensor(dw, DType::kQ5_0, {1, 32});
  Tensor ot = DevTensor(dout, DType::kF32, {1, 1});
  vt::MatmulBTQuant(q, ot, at, wt);
  float actual = 0;
  gpu.Copy(q, &actual, dout, 4);
  gpu.Synchronize(q);
  gpu.Free(da); gpu.Free(dw); gpu.Free(dout);
  CHECK(actual == doctest::Approx(expected).epsilon(1e-6));
#endif
}

TEST_CASE("Pascal Q6_K decode matches independently reconstructed GPU Q8_1 activations") {
#ifdef VLLM_CPP_CUDA
  if (!HasCuda()) return;
  cudaDeviceProp prop{};
  REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
  if (prop.major != 6) return;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue q = gpu.CreateQueue();
  const WeightCase c{DType::kQ6_K, 256, 210, 208, -1, "q6_K"};
  for (int64_t k : {256, 1536, 2048}) {
    CAPTURE(k);
    constexpr int64_t n = 7;
    const auto w = RandomBlocks(c, n * k / 256, 351);
    std::vector<float> dequant(n * k), a(k), reconstructed(k);
    vt::cpu::BlockToFloat(c.dtype)(w.data(), dequant.data(), n * k);
    for (int64_t j = 0; j < k; ++j) a[j] = std::sin(j * .31f) * (1 + (j / 32) % 8);
    for (int64_t b = 0; b < k / 32; ++b) {
      float amax = 0;
      for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(a[b * 32 + j]));
      const float d = amax / 127.0f;
      const float hd = vt::F16ToF32(vt::F32ToF16(d));
      for (int j = 0; j < 32; ++j)
        reconstructed[b * 32 + j] = std::round(a[b * 32 + j] / d) * hd;
    }
    void* da = gpu.Alloc(k * 4); void* dw = gpu.Alloc(w.size()); void* dout = gpu.Alloc(n * 4);
    gpu.Copy(q, da, a.data(), k * 4); gpu.Copy(q, dw, w.data(), w.size());
    Tensor at = DevTensor(da, DType::kF32, {1, k});
    Tensor wt = DevTensor(dw, c.dtype, {n, k});
    Tensor ot = DevTensor(dout, DType::kF32, {1, n});
    vt::MatmulBTQuant(q, ot, at, wt);
    std::vector<float> actual(n);
    gpu.Copy(q, actual.data(), dout, n * 4); gpu.Synchronize(q);
    gpu.Free(da); gpu.Free(dw); gpu.Free(dout);
    for (int64_t row = 0; row < n; ++row) {
      double expected = 0;
      for (int64_t j = 0; j < k; ++j) expected += double(dequant[row * k + j]) * reconstructed[j];
      CHECK(actual[row] == doctest::Approx(expected).epsilon(2e-5).scale(1));
    }
  }
  gpu.DestroyQueue(q);
#endif
}

TEST_CASE("Pascal Q4_K decode matches independently reconstructed GPU Q8_1 activations") {
#ifdef VLLM_CPP_CUDA
  if (!HasCuda()) return;
  {
    const char* e = std::getenv("VT_PASCAL_Q4K_DECODE");
    if (e != nullptr && e[0] == '0') return;
  }
  cudaDeviceProp prop{};
  REQUIRE(cudaGetDeviceProperties(&prop, 0) == cudaSuccess);
  if (prop.major != 6) return;
  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue q = gpu.CreateQueue();
  const WeightCase& c = CaseFor(DType::kQ4_K);
  for (int64_t k : {256, 2048, 5632}) {
    CAPTURE(k);
    constexpr int64_t n = 7;
    const auto w = RandomBlocks(c, n * k / 256, 0x4B1DU);
    std::vector<float> dequant(n * k), a(k), reconstructed(k);
    vt::cpu::BlockToFloat(c.dtype)(w.data(), dequant.data(), n * k);
    for (int64_t j = 0; j < k; ++j) a[j] = std::sin(j * .37f) * (1 + (j / 32) % 8);
    for (int64_t b = 0; b < k / 32; ++b) {
      float amax = 0;
      for (int j = 0; j < 32; ++j) amax = std::max(amax, std::abs(a[b * 32 + j]));
      const float d = amax / 127.0f;
      const float hd = vt::F16ToF32(vt::F32ToF16(d));
      for (int j = 0; j < 32; ++j)
        reconstructed[b * 32 + j] = std::round(a[b * 32 + j] / d) * hd;
    }
    void* da = gpu.Alloc(k * 4); void* dw = gpu.Alloc(w.size()); void* dout = gpu.Alloc(n * 4);
    gpu.Copy(q, da, a.data(), k * 4); gpu.Copy(q, dw, w.data(), w.size());
    Tensor at = DevTensor(da, DType::kF32, {1, k});
    Tensor wt = DevTensor(dw, c.dtype, {n, k});
    Tensor ot = DevTensor(dout, DType::kF32, {1, n});
    vt::MatmulBTQuant(q, ot, at, wt);
    std::vector<float> actual(n);
    gpu.Copy(q, actual.data(), dout, n * 4); gpu.Synchronize(q);
    gpu.Free(da); gpu.Free(dw); gpu.Free(dout);
    for (int64_t row = 0; row < n; ++row) {
      double expected = 0;
      for (int64_t j = 0; j < k; ++j) expected += double(dequant[row * k + j]) * reconstructed[j];
      CHECK(actual[row] == doctest::Approx(expected).epsilon(2e-5).scale(1));
    }
  }
  gpu.DestroyQueue(q);
#endif
}
