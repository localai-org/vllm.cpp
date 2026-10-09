// ROCm fp8 KV-cache store + paged-attention read gate (KV-FP8 W6, #2065).
//
// W1 landed the CPU half: vt::ReshapeAndCacheFp8 (fp8-e4m3 store), the fp8 read
// dequant in CPU paged attention, and vllm::v1::ParseCacheDtype. W1 IS THE
// ORACLE FOR W6 — the ROCm arm is measured against it, never against a fresh
// reference — so this file only ever compares ROCm to the landed CPU kernels.
// The CUDA arm (W2) is the direct template; the ROCm arm is elementwise-
// identical to it, and the CUDA arm is itself elementwise-identical to the CPU
// reference.
//
// Upstream mirror @ pin 555967922:
//   store  vllm/csrc/libtorch_stable/cache_kernels.cu:314-401
//          (reshape_and_cache_flash_kernel, fp8 branch) + CopyWithScaleOp :241-252
//   read   vllm/csrc/quantization/w8a8/fp8/nvidia/quant_utils.cuh:419-429
//          (scaled_vec_conversion<float, uint8_t>)
//   scale  quant_utils.cuh:296-300 — FP8 = Quantize(HP / scale);
//                                    Dequant(FP8) * scale = HP
//   scales vllm/model_executor/layers/quantization/kv_cache.py:108-191
//          (BaseKVCacheMethod: per-TENSOR k_scale/v_scale, 1.0 uncalibrated)
//
// The gates, and they do not all run in the same build:
//
//  G1 (runs in every build WITHOUT the ROCm backend, i.e. the x86 CI leg): the
//     fp8 store/read resolves through the provider table on a non-CPU device
//     with no "later brick" guard. Compiled only where the ROCm backend is
//     absent: in a ROCm build the op IS registered, so these calls would
//     dispatch a real kernel over host pointers.
//  G1b (every build): the fp8 READ is refused by name on kMETAL (the only
//     backend with no fp8 dequant now that ROCm is implemented). The check
//     fires in the op wrapper, so no Metal backend need be linked.
//  G2 (ROCm build): the ROCm providers are REGISTERED for the fp8 store and the
//     paged read — the shared-seam reach check.
//  G3 (ROCm device): STORE parity — the ROCm store writes the SAME BYTES as the
//     CPU store, zero tolerance, over the f32, bf16 and f16 sources the wrapper
//     admits, with a padded (-1) slot and a strided unbind-slice cache.
//  G4 (ROCm device): READ parity — paged attention over identical fp8 cache
//     bytes, ROCm vs CPU, in both the decode and the prefill shape, for an f32
//     query/output.
//  G4b (ROCm device): ...and for the bf16 query/output a served model actually
//     runs, which is a DIFFERENT template instantiation of the same launcher.
//  G5 (ROCm device): fp8_e5m2 stays refused BY THE ROCM KERNEL, reached through
//     the registered provider.
//
// G3/G4/G4b/G5 SKIP CLEANLY when no ROCm backend is present, which is the house
// pattern (tests/vt/test_cuda_quant_dot.cpp:80-88). A skip is NOT a pass: every
// skipping case prints a MESSAGE naming what did not run.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <atomic>
#include <vector>
#include <cstdlib>

// G6 needs VT_ATTN_DECODE_GQA4=1 active before the first call to
// PagedAttentionKernelRocm, because the dispatch reads the env var once into a
// static local. This file-scope initializer runs before main() and before any
// TEST_CASE, so the static picks it up on first entry. G4/G4b's geometry
// (hq=2, H=1, D=16) does not match the GQA4 guard (hq==16, kv==4, d==128||256),
// so setting it here does not reroute those cases.
namespace {
struct SetGqa4Env {
  SetGqa4Env() { ::setenv("VT_ATTN_DECODE_GQA4", "1", 1); }
} _set_gqa4_env;
}  // namespace
#include "vt/backend.h"
#include "vt/device.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/op_provider.h"
#include "vt/ops.h"
#include "vt/tensor.h"

using vt::Backend;
using vt::Device;
using vt::DeviceType;
using vt::DType;
using vt::Fp8KVCacheDataType;
using vt::OpId;
using vt::PagedAttentionArgs;
using vt::Queue;
using vt::Tensor;

// Test-visible dispatch counter from rocm_paged_attn.hip. Only defined in
// ROCm builds (VLLM_CPP_HIP); the G7 test uses it to prove the SharedK
// prefill dispatch fired, not just that the output is correct.
#ifdef VLLM_CPP_HIP
namespace vt::rocm {
extern std::atomic<int>& Fp8PrefillSharedKDispatchCount();
}  // namespace vt::rocm
#endif

namespace {

bool HasRocm() {
  try {
    vt::GetBackend(DeviceType::kROCM);
    return true;
  } catch (const std::runtime_error&) {
    return false;
  }
}

Device Cpu() { return Device{DeviceType::kCPU, 0}; }
Device Gpu() { return Device{DeviceType::kROCM, 0}; }

// Tensor::Contiguous takes an initializer_list; these take the runtime shapes
// the cases build. Same packed-stride result.
Tensor Contig(void* data, DType dt, Device dev, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

Tensor Host(void* data, DType dt, const std::vector<int64_t>& shape) {
  return Contig(data, dt, Cpu(), shape);
}

Tensor Dev(void* data, DType dt, const std::vector<int64_t>& shape) {
  return Contig(data, dt, Gpu(), shape);
}

std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

}  // namespace

// ─── G1 ─────────────────────────────────────────────────────────────────────
// The fp8 store/read resolves through the provider table on a non-CPU device
// with no "later brick" guard. Compiled only where the ROCm backend is absent:
// in a ROCm build the op IS registered, so these calls would dispatch a real
// kernel over host pointers. The ROCm build asserts the same property from the
// other side, in G2.
#ifndef VLLM_CPP_HIP
TEST_CASE("fp8 KV ops resolve through the provider table on a non-CPU device") {
  const int64_t nb = 1, bs = 4, H = 1, D = 16, page = H * D;
  std::vector<float> k(static_cast<size_t>(page), 1.0f), v(static_cast<size_t>(page), 1.0f);
  std::vector<uint8_t> kc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<uint8_t> vc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<int64_t> slots = {0};
  Tensor tk = Dev(k.data(), DType::kF32, {1, H, D});
  Tensor tv = Dev(v.data(), DType::kF32, {1, H, D});
  Tensor tkc = Dev(kc.data(), DType::kI8, {nb, bs, H, D});
  Tensor tvc = Dev(vc.data(), DType::kI8, {nb, bs, H, D});
  Tensor ts = Dev(slots.data(), DType::kI64, {1});
  Queue qq{Gpu(), nullptr};

  std::string store_msg;
  try {
    vt::ReshapeAndCacheFp8(qq, tk, tv, tkc, tvc, ts, Fp8KVCacheDataType::kFp8E4M3, 0.01f, 0.01f);
    FAIL("reshape_and_cache_fp8 must refuse when no ROCm provider is linked in");
  } catch (const std::runtime_error& e) {
    store_msg = e.what();
  }
  CAPTURE(store_msg);
  // The refusal must come from the PROVIDER TABLE, naming the op...
  CHECK(store_msg.find("no kernel for op ReshapeAndCacheFp8") != std::string::npos);
  // ...and NOT from a device-class guard inside the wrapper.
  CHECK(store_msg.find("later brick") == std::string::npos);

  // Same for the read side: PagedAttention's fp8 arm must not carry a CPU-only
  // guard either. One request, one decode token, one 16-wide head.
  std::vector<float> q(static_cast<size_t>(D), 0.5f), out(static_cast<size_t>(D), 0.0f);
  std::vector<int32_t> bt = {0}, seq = {1}, qsl = {0, 1};
  Tensor tq = Dev(q.data(), DType::kF32, {1, 1, D});
  Tensor to = Dev(out.data(), DType::kF32, {1, 1, D});
  Tensor tbt = Dev(bt.data(), DType::kI32, {1, 1});
  Tensor tseq = Dev(seq.data(), DType::kI32, {1});
  Tensor tqsl = Dev(qsl.data(), DType::kI32, {2});
  PagedAttentionArgs args;
  args.scale = 0.25f;
  args.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
  args.k_scale = 0.01f;
  args.v_scale = 0.01f;

  std::string read_msg;
  try {
    vt::PagedAttention(qq, to, tq, tkc, tvc, tbt, tseq, tqsl, args);
    FAIL("paged_attention fp8 read must refuse when no ROCm provider is linked in");
  } catch (const std::runtime_error& e) {
    read_msg = e.what();
  }
  CAPTURE(read_msg);
  CHECK(read_msg.find("no kernel for op PagedAttention") != std::string::npos);
  CHECK(read_msg.find("later brick") == std::string::npos);
}
#endif  // !VLLM_CPP_HIP

// ─── G1b ────────────────────────────────────────────────────────────────────
// The fp8 READ rides ADDITIVE fields on PagedAttentionArgs of an op that kMETAL
// already registers for the FLOAT path (metal_ops.mm). The provider table
// cannot tell the two arms apart, so an fp8 cache reaching that kernel would be
// read as that backend's float dtype and return silent garbage. AGENTS.md
// requires an unimplemented arm to refuse with a message that NAMES the missing
// part. ROCm landed in W6, so only Metal is refused now.
//
// Runs in every build: the check fires in the op wrapper, before any device or
// provider is touched, so no Metal backend needs to be linked in.
TEST_CASE("the fp8 KV read is refused on a backend with no fp8 dequant") {
  const int64_t nb = 1, bs = 4, H = 1, D = 16, page = H * D;
  std::vector<uint8_t> kc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<uint8_t> vc(static_cast<size_t>(nb * bs * page), 0);
  std::vector<float> q(static_cast<size_t>(D), 0.5f), out(static_cast<size_t>(D), 0.0f);
  std::vector<int32_t> bt = {0}, seq = {1}, qsl = {0, 1};
  PagedAttentionArgs args;
  args.scale = 0.25f;
  args.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
  args.k_scale = 0.01f;
  args.v_scale = 0.01f;

  const Device dev{DeviceType::kMETAL, 0};
  Tensor tq = Contig(q.data(), DType::kF32, dev, {1, 1, D});
  Tensor to = Contig(out.data(), DType::kF32, dev, {1, 1, D});
  Tensor tkc = Contig(kc.data(), DType::kI8, dev, {nb, bs, H, D});
  Tensor tvc = Contig(vc.data(), DType::kI8, dev, {nb, bs, H, D});
  Tensor tbt = Contig(bt.data(), DType::kI32, dev, {1, 1});
  Tensor tseq = Contig(seq.data(), DType::kI32, dev, {1});
  Tensor tqsl = Contig(qsl.data(), DType::kI32, dev, {2});
  Queue qq{dev, nullptr};
  std::string msg;
  try {
    vt::PagedAttention(qq, to, tq, tkc, tvc, tbt, tseq, tqsl, args);
    FAIL("paged_attention must refuse the fp8 KV read on a backend without one");
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  CAPTURE(msg);
  CHECK(msg.find("fp8 KV read") != std::string::npos);
  // The message must say WHAT would go wrong, not merely that it is refused.
  CHECK(msg.find("no fp8 dequant") != std::string::npos);
}

// ─── G2 ─────────────────────────────────────────────────────────────────────
// Reach through the shared seam. vt::ReshapeAndCacheFp8 and vt::PagedAttention
// dispatch through GetOp(OpId, DeviceType) (src/vt/ops.cpp), so a provider
// registered for kROCM IS the production path — nothing else selects a kernel.
// Registration is a static-init table fill, so this holds without a device: it
// asks "was the ROCm arm compiled and registered", which is exactly the question
// a `#ifdef`-elided kernel silently answers "no" to.
#ifdef VLLM_CPP_HIP
TEST_CASE("the ROCm fp8 KV store and paged read are registered providers") {
  CHECK(vt::GetOp(OpId::kReshapeAndCacheFp8, DeviceType::kROCM) != nullptr);
  CHECK(vt::GetOp(OpId::kPagedAttention, DeviceType::kROCM) != nullptr);
}
#endif  // VLLM_CPP_HIP

// ─── G3 ─────────────────────────────────────────────────────────────────────
// STORE parity, byte for byte, zero tolerance. The CPU kernel is the oracle.
//
// The two arms are the same arithmetic by construction: the CPU codec is
// vt::F32ToF8E4M3 (include/vt/fp8_kv.h — software round-to-nearest-even,
// saturating at +/-448) and the ROCm kernel uses the SAME software codec
// vt::StoreKvFp8E4M3 (include/vt/fp8_kv.h:87-89), because ROCm HIP has no
// __nv_cvt_float_to_fp8 intrinsic. The CUDA intrinsic's equality to the CPU
// codec is already MEASURED at zero tolerance (spec W2,
// vt-fp8-quant-arch-gate.md G2); the ROCm arm uses the CPU codec directly, so
// ROCm==CPU on the store is a property of the source rather than of a
// measurement. This case re-takes it on the KV path, where the scale is applied
// as a true DIVIDE rather than the activation path's reciprocal multiply.
TEST_CASE("rocm fp8 KV store is byte-identical to the CPU store") {
  if (!HasRocm()) {
    MESSAGE("SKIPPED: no ROCm backend in this build/host — the ROCm fp8 KV store "
            "parity gate did NOT run");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  const int64_t nb = 1, bs = 4, H = 1, D = 16, page = H * D, nt = 4;
  const size_t cache_elems = static_cast<size_t>(nb * bs * page);
  auto kf = RandF32(static_cast<size_t>(nt * page), 11);
  auto vf = RandF32(static_cast<size_t>(nt * page), 22);
  std::vector<int64_t> slots = {3, 0, 2, -1};  // -1 = padded token → skip
  const float k_scale = 0.007f, v_scale = 0.003f;

  // CPU oracle: store the same tokens through the CPU kernel.
  std::vector<uint8_t> kc_ref(cache_elems, 0xAB), vc_ref(cache_elems, 0xCD);
  Tensor ck = Host(kf.data(), DType::kF32, {nt, H, D});
  Tensor cv = Host(vf.data(), DType::kF32, {nt, H, D});
  Tensor ckc = Host(kc_ref.data(), DType::kI8, {nb, bs, H, D});
  Tensor cvc = Host(vc_ref.data(), DType::kI8, {nb, bs, H, D});
  Tensor cs = Host(slots.data(), DType::kI64, {nt});
  vt::ReshapeAndCacheFp8(cq, ck, cv, ckc, cvc, cs, Fp8KVCacheDataType::kFp8E4M3, k_scale,
                         v_scale);

  void* dk = gpu.Alloc(kf.size() * sizeof(float));
  void* dv = gpu.Alloc(vf.size() * sizeof(float));
  void* dkc = gpu.Alloc(cache_elems);
  void* dvc = gpu.Alloc(cache_elems);
  void* ds = gpu.Alloc(slots.size() * sizeof(int64_t));
  std::vector<uint8_t> kc_seed(cache_elems, 0xAB);
  std::vector<uint8_t> vc_seed(cache_elems, 0xCD);
  gpu.Copy(gq, dk, kf.data(), kf.size() * sizeof(float));
  gpu.Copy(gq, dv, vf.data(), vf.size() * sizeof(float));
  gpu.Copy(gq, dkc, kc_seed.data(), cache_elems);
  gpu.Copy(gq, dvc, vc_seed.data(), cache_elems);
  gpu.Copy(gq, ds, slots.data(), slots.size() * sizeof(int64_t));
  Tensor gk = Dev(dk, DType::kF32, {nt, H, D});
  Tensor gv = Dev(dv, DType::kF32, {nt, H, D});
  Tensor gkc = Dev(dkc, DType::kI8, {nb, bs, H, D});
  Tensor gvc = Dev(dvc, DType::kI8, {nb, bs, H, D});
  Tensor gs = Dev(ds, DType::kI64, {nt});
  vt::ReshapeAndCacheFp8(gq, gk, gv, gkc, gvc, gs, Fp8KVCacheDataType::kFp8E4M3, k_scale,
                         v_scale);

  std::vector<uint8_t> kc_got(cache_elems, 0);
  std::vector<uint8_t> vc_got(cache_elems, 0);
  gpu.Copy(gq, kc_got.data(), dkc, cache_elems);
  gpu.Copy(gq, vc_got.data(), dvc, cache_elems);
  gpu.Synchronize(gq);

  int64_t kbad = 0, vbad = 0;
  for (size_t i = 0; i < cache_elems; ++i) {
    if (kc_got[i] != kc_ref[i]) ++kbad;
    if (vc_got[i] != vc_ref[i]) ++vbad;
  }
  CHECK(kbad == 0);
  CHECK(vbad == 0);
  // Two kernels that both returned early would leave the seed fill on both
  // sides and compare equal, so require that the ORACLE wrote something. This
  // is asked of the CPU bytes, not the ROCm ones: a quantized byte may
  // legitimately equal the 0xAB fill, and counting ROCm's differences would
  // then be an assertion about the fixture rather than about the kernel.
  int64_t ref_written = 0;
  for (size_t i = 0; i < cache_elems; ++i) {
    if (kc_ref[i] != 0xAB) ++ref_written;
  }
  CHECK(ref_written > 0);

  gpu.Free(dk);
  gpu.Free(dv);
  gpu.Free(dkc);
  gpu.Free(dvc);
  gpu.Free(ds);
  gpu.DestroyQueue(gq);
}

// The two NARROW source arms of the same store, and both of them matter.
//
// bf16 is the dtype vLLM actually resolves for a model (AGENTS.md "Inherit
// vLLM defaults"), so it is the arm production runs. f16 is the arm nothing
// else covers: `vt::ReshapeAndCacheFp8`'s wrapper admits any `IsFloat()`
// source (src/vt/ops.cpp), the CPU `LoadSrcF32` serves f16
// (src/vt/cpu/cpu_cache.cpp), and the ROCm `ReshapeAndCacheFp8KernelRocm` has
// a `DType::kF16 -> __half` arm — which, without this case, no gate would ever
// instantiate on a device. An untested dispatch arm is the shape a wrong
// `Ptr<>` cast hides in.
//
// Both are widened to f32 BEFORE the divide on each side — upstream does the
// same (`quant_utils.cuh:482-489`, `__bfloat162float(a) / scale`), the ROCm
// kernel through `Ld` and the CPU through `LoadSrcF32` — and bf16->f32 and
// f16->f32 are both exact, so the two arms must still agree byte for byte.
TEST_CASE("rocm fp8 KV store is byte-identical to the CPU store (bf16 and f16 sources)") {
  if (!HasRocm()) {
    MESSAGE("SKIPPED: no ROCm backend in this build/host — the bf16/f16-source "
            "fp8 KV store parity gate did NOT run");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  const int64_t nb = 1, bs = 4, H = 1, D = 16, page = H * D, nt = 4;
  const size_t cache_elems = static_cast<size_t>(nb * bs * page);
  auto kf = RandF32(static_cast<size_t>(nt * page), 33);
  auto vf = RandF32(static_cast<size_t>(nt * page), 44);
  std::vector<int64_t> slots = {3, 0, 2, 1};
  const float k_scale = 0.007f, v_scale = 0.003f;

  // Both narrow dtypes are 2-byte, so one uint16_t staging buffer serves each.
  for (DType src : {DType::kBF16, DType::kF16}) {
    const int src_dtype_tag = static_cast<int>(src);
    CAPTURE(src_dtype_tag);
    std::vector<uint16_t> kb(kf.size()), vb(vf.size());
    for (size_t i = 0; i < kf.size(); ++i) {
      kb[i] = src == DType::kBF16 ? vt::F32ToBF16(kf[i]) : vt::F32ToF16(kf[i]);
      vb[i] = src == DType::kBF16 ? vt::F32ToBF16(vf[i]) : vt::F32ToF16(vf[i]);
    }

    std::vector<uint8_t> kc_ref(cache_elems, 0);
    std::vector<uint8_t> vc_ref(cache_elems, 0);
    Tensor ck = Host(kb.data(), src, {nt, H, D});
    Tensor cv = Host(vb.data(), src, {nt, H, D});
    Tensor ckc = Host(kc_ref.data(), DType::kI8, {nb, bs, H, D});
    Tensor cvc = Host(vc_ref.data(), DType::kI8, {nb, bs, H, D});
    Tensor cs = Host(slots.data(), DType::kI64, {nt});
    vt::ReshapeAndCacheFp8(cq, ck, cv, ckc, cvc, cs, Fp8KVCacheDataType::kFp8E4M3, k_scale,
                           v_scale);

    void* dk = gpu.Alloc(kb.size() * sizeof(uint16_t));
    void* dv = gpu.Alloc(vb.size() * sizeof(uint16_t));
    void* dkc = gpu.Alloc(cache_elems);
    void* dvc = gpu.Alloc(cache_elems);
    void* ds = gpu.Alloc(slots.size() * sizeof(int64_t));
    std::vector<uint8_t> zero(cache_elems, 0);
    gpu.Copy(gq, dk, kb.data(), kb.size() * sizeof(uint16_t));
    gpu.Copy(gq, dv, vb.data(), vb.size() * sizeof(uint16_t));
    gpu.Copy(gq, dkc, zero.data(), cache_elems);
    gpu.Copy(gq, dvc, zero.data(), cache_elems);
    gpu.Copy(gq, ds, slots.data(), slots.size() * sizeof(int64_t));
    Tensor gk = Dev(dk, src, {nt, H, D});
    Tensor gv = Dev(dv, src, {nt, H, D});
    Tensor gkc = Dev(dkc, DType::kI8, {nb, bs, H, D});
    Tensor gvc = Dev(dvc, DType::kI8, {nb, bs, H, D});
    Tensor gs = Dev(ds, DType::kI64, {nt});
    vt::ReshapeAndCacheFp8(gq, gk, gv, gkc, gvc, gs, Fp8KVCacheDataType::kFp8E4M3, k_scale,
                           v_scale);

    std::vector<uint8_t> kc_got(cache_elems, 0);
    std::vector<uint8_t> vc_got(cache_elems, 0);
    gpu.Copy(gq, kc_got.data(), dkc, cache_elems);
    gpu.Copy(gq, vc_got.data(), dvc, cache_elems);
    gpu.Synchronize(gq);
    CHECK(kc_got == kc_ref);
    CHECK(vc_got == vc_ref);
    // The CPU oracle must have WRITTEN something, or the equality above is
    // between two all-zero buffers and holds for any kernel.
    CHECK(std::any_of(kc_ref.begin(), kc_ref.end(), [](uint8_t b) { return b != 0; }));

    gpu.Free(dk);
    gpu.Free(dv);
    gpu.Free(dkc);
    gpu.Free(dvc);
    gpu.Free(ds);
  }
  gpu.DestroyQueue(gq);
}

// ─── G4 ─────────────────────────────────────────────────────────────────────
// READ parity: paged attention over the SAME fp8 cache bytes, ROCm vs CPU, in
// BOTH shapes the fp8 arm routes to — pure decode (the generic block kernel)
// and prefill (the tiled flash kernel). The cache is built once on the host so
// this case measures the READ alone; G3 already measures the store.
//
// The dequant itself is bit-identical by construction: the ROCm kernel decodes
// e4m3 with the same arithmetic as vt::F8E4M3ToF32 and multiplies by the same
// per-tensor scale (quant_utils.cuh:419-429). The only divergence available is
// the softmax REDUCTION ORDER (block-cooperative on ROCm, sequential on the
// CPU), so the band is tight. A wrong scale, a missing dequant, a swapped
// k_scale/v_scale or a dropped sign blows it by orders of magnitude.
TEST_CASE("rocm fp8 KV paged-attention read matches the CPU read") {
  if (!HasRocm()) {
    MESSAGE("SKIPPED: no ROCm backend in this build/host — the ROCm fp8 KV "
            "paged-attention read parity gate did NOT run");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // 2 requests, 2 q-heads over 1 kv-head (GQA), head_size 16, block_size 4.
  const int64_t nb = 4, bs = 4, H = 1, D = 16, hq = 2, num_reqs = 2;
  const size_t cache_elems = static_cast<size_t>(nb * bs * H * D);
  auto raw = RandF32(cache_elems, 77);
  const float k_scale = 0.005f, v_scale = 0.009f;
  std::vector<uint8_t> kc(cache_elems), vc(cache_elems);
  for (size_t i = 0; i < cache_elems; ++i) {
    kc[i] = vt::StoreKvFp8E4M3(raw[i], k_scale);
    vc[i] = vt::StoreKvFp8E4M3(raw[cache_elems - 1 - i], v_scale);
  }
  std::vector<int32_t> bt = {0, 1, 2, 3};  // [num_reqs, max_blocks]
  std::vector<int32_t> seq = {5, 3};

  void* dkc = gpu.Alloc(cache_elems);
  void* dvc = gpu.Alloc(cache_elems);
  void* dbt = gpu.Alloc(bt.size() * sizeof(int32_t));
  void* dseq = gpu.Alloc(seq.size() * sizeof(int32_t));
  gpu.Copy(gq, dkc, kc.data(), cache_elems);
  gpu.Copy(gq, dvc, vc.data(), cache_elems);
  gpu.Copy(gq, dbt, bt.data(), bt.size() * sizeof(int32_t));
  gpu.Copy(gq, dseq, seq.data(), seq.size() * sizeof(int32_t));

  struct Shape {
    const char* name;
    int64_t nt;
    std::vector<int32_t> qsl;
  };
  // nt == num_reqs -> pure decode; nt > num_reqs -> prefill.
  const std::vector<Shape> shapes = {{"decode", 2, {0, 1, 2}}, {"prefill", 4, {0, 3, 4}}};

  for (const Shape& sh : shapes) {
    CAPTURE(std::string(sh.name));
    auto qh = RandF32(static_cast<size_t>(sh.nt * hq * D), 88);
    std::vector<int32_t> qsl = sh.qsl;

    PagedAttentionArgs args;
    args.scale = 0.25f;
    args.causal = true;
    args.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
    args.k_scale = k_scale;
    args.v_scale = v_scale;

    std::vector<float> cpu_out(static_cast<size_t>(sh.nt * hq * D), 0.0f);
    Tensor cqt = Host(qh.data(), DType::kF32, {sh.nt, hq, D});
    Tensor cot = Host(cpu_out.data(), DType::kF32, {sh.nt, hq, D});
    Tensor ckc = Host(kc.data(), DType::kI8, {nb, bs, H, D});
    Tensor cvc = Host(vc.data(), DType::kI8, {nb, bs, H, D});
    Tensor cbt = Host(bt.data(), DType::kI32, {num_reqs, 2});
    Tensor cseq = Host(seq.data(), DType::kI32, {num_reqs});
    Tensor cqsl = Host(qsl.data(), DType::kI32, {num_reqs + 1});
    vt::PagedAttention(cq, cot, cqt, ckc, cvc, cbt, cseq, cqsl, args);

    void* dq = gpu.Alloc(qh.size() * sizeof(float));
    void* dout = gpu.Alloc(qh.size() * sizeof(float));
    void* dqsl = gpu.Alloc(qsl.size() * sizeof(int32_t));
    gpu.Copy(gq, dq, qh.data(), qh.size() * sizeof(float));
    gpu.Copy(gq, dqsl, qsl.data(), qsl.size() * sizeof(int32_t));
    Tensor gqt = Dev(dq, DType::kF32, {sh.nt, hq, D});
    Tensor got = Dev(dout, DType::kF32, {sh.nt, hq, D});
    Tensor gkc = Dev(dkc, DType::kI8, {nb, bs, H, D});
    Tensor gvc = Dev(dvc, DType::kI8, {nb, bs, H, D});
    Tensor gbt = Dev(dbt, DType::kI32, {num_reqs, 2});
    Tensor gseq = Dev(dseq, DType::kI32, {num_reqs});
    Tensor gqsl = Dev(dqsl, DType::kI32, {num_reqs + 1});
    vt::PagedAttention(gq, got, gqt, gkc, gvc, gbt, gseq, gqsl, args);

    std::vector<float> gpu_out(qh.size(), 0.0f);
    gpu.Copy(gq, gpu_out.data(), dout, gpu_out.size() * sizeof(float));
    gpu.Synchronize(gq);

    double num = 0.0, den = 0.0, worst = 0.0;
    for (size_t i = 0; i < gpu_out.size(); ++i) {
      const double d0 = static_cast<double>(gpu_out[i]) - static_cast<double>(cpu_out[i]);
      num += d0 * d0;
      den += static_cast<double>(cpu_out[i]) * static_cast<double>(cpu_out[i]);
      worst = std::max(worst, std::fabs(d0));
    }
    // The CPU arm must have produced a non-degenerate output, or the comparison
    // above is between two fields of zeros and would pass on any kernel.
    CHECK(den > 0.0);
    const double nmse = den > 0.0 ? num / den : 1.0;
    CAPTURE(nmse);
    CAPTURE(worst);
    CHECK(nmse < 1e-6);
    CHECK(worst < 1e-3);

    gpu.Free(dq);
    gpu.Free(dout);
    gpu.Free(dqsl);
  }

  gpu.Free(dkc);
  gpu.Free(dvc);
  gpu.Free(dbt);
  gpu.Free(dseq);
  gpu.DestroyQueue(gq);
}

// ─── G4b ────────────────────────────────────────────────────────────────────
// THE INSTANTIATION PRODUCTION WILL USE. G4 above runs an f32 query into an f32
// output, which resolves `PagedAttnOnline<float, uint8_t, float>`
// (src/vt/rocm/rocm_paged_attn.hip). That is not the arm a served model takes:
// vLLM resolves ONE model dtype and every layer inherits it (AGENTS.md "Inherit
// vLLM defaults"), the gate models are bf16, and the bf16 query/output arm is
// what production runs. Without this case
// `PagedAttnOnline<__hip_bfloat16, uint8_t, __hip_bfloat16>` compiles, ships,
// and is never once executed against the oracle.
//
// The band is looser than G4's and deliberately so: both arms round an f32
// accumulator to bf16 on the store, and bf16 carries 8 mantissa bits, so two
// accumulators that differ only in softmax reduction order can land on
// opposite sides of one rounding boundary. The output is a convex combination
// of V rows and every V here is inside [-2, 2], so |x| < 2 and one bf16 ulp is
// at most 2^1 * 2^-7 = 1.56e-2; even if EVERY element were a full ulp out the
// NMSE would be (2^-8)^2 = 1.5e-5. The band below admits that and nothing else
// — a missing dequant, a swapped k_scale/v_scale or a dropped sign moves the
// output by orders of magnitude, not by an ulp.
TEST_CASE("rocm fp8 KV paged-attention read matches the CPU read (bf16 query, bf16 out)") {
  if (!HasRocm()) {
    MESSAGE("SKIPPED: no ROCm backend in this build/host — the bf16-query/bf16-out "
            "fp8 KV paged-attention read parity gate did NOT run");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  const int64_t nb = 4, bs = 4, H = 1, D = 16, hq = 2, num_reqs = 2;
  const size_t cache_elems = static_cast<size_t>(nb * bs * H * D);
  auto raw = RandF32(cache_elems, 77);
  const float k_scale = 0.005f, v_scale = 0.009f;
  std::vector<uint8_t> kc(cache_elems), vc(cache_elems);
  for (size_t i = 0; i < cache_elems; ++i) {
    kc[i] = vt::StoreKvFp8E4M3(raw[i], k_scale);
    vc[i] = vt::StoreKvFp8E4M3(raw[cache_elems - 1 - i], v_scale);
  }
  std::vector<int32_t> bt = {0, 1, 2, 3};
  std::vector<int32_t> seq = {5, 3};

  void* dkc = gpu.Alloc(cache_elems);
  void* dvc = gpu.Alloc(cache_elems);
  void* dbt = gpu.Alloc(bt.size() * sizeof(int32_t));
  void* dseq = gpu.Alloc(seq.size() * sizeof(int32_t));
  gpu.Copy(gq, dkc, kc.data(), cache_elems);
  gpu.Copy(gq, dvc, vc.data(), cache_elems);
  gpu.Copy(gq, dbt, bt.data(), bt.size() * sizeof(int32_t));
  gpu.Copy(gq, dseq, seq.data(), seq.size() * sizeof(int32_t));

  struct Shape {
    const char* name;
    int64_t nt;
    std::vector<int32_t> qsl;
  };
  const std::vector<Shape> shapes = {{"decode", 2, {0, 1, 2}}, {"prefill", 4, {0, 3, 4}}};

  for (const Shape& sh : shapes) {
    const std::string shape_name(sh.name);
    CAPTURE(shape_name);
    auto qf = RandF32(static_cast<size_t>(sh.nt * hq * D), 88);
    std::vector<uint16_t> qb(qf.size());
    for (size_t i = 0; i < qf.size(); ++i) qb[i] = vt::F32ToBF16(qf[i]);
    std::vector<int32_t> qsl = sh.qsl;

    PagedAttentionArgs args;
    args.scale = 0.25f;
    args.causal = true;
    args.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
    args.k_scale = k_scale;
    args.v_scale = v_scale;

    std::vector<uint16_t> cpu_out(qf.size(), 0);
    Tensor cqt = Host(qb.data(), DType::kBF16, {sh.nt, hq, D});
    Tensor cot = Host(cpu_out.data(), DType::kBF16, {sh.nt, hq, D});
    Tensor ckc = Host(kc.data(), DType::kI8, {nb, bs, H, D});
    Tensor cvc = Host(vc.data(), DType::kI8, {nb, bs, H, D});
    Tensor cbt = Host(bt.data(), DType::kI32, {num_reqs, 2});
    Tensor cseq = Host(seq.data(), DType::kI32, {num_reqs});
    Tensor cqsl = Host(qsl.data(), DType::kI32, {num_reqs + 1});
    vt::PagedAttention(cq, cot, cqt, ckc, cvc, cbt, cseq, cqsl, args);

    void* dq = gpu.Alloc(qb.size() * sizeof(uint16_t));
    void* dout = gpu.Alloc(qb.size() * sizeof(uint16_t));
    void* dqsl = gpu.Alloc(qsl.size() * sizeof(int32_t));
    gpu.Copy(gq, dq, qb.data(), qb.size() * sizeof(uint16_t));
    gpu.Copy(gq, dqsl, qsl.data(), qsl.size() * sizeof(int32_t));
    Tensor gqt = Dev(dq, DType::kBF16, {sh.nt, hq, D});
    Tensor got = Dev(dout, DType::kBF16, {sh.nt, hq, D});
    Tensor gkc = Dev(dkc, DType::kI8, {nb, bs, H, D});
    Tensor gvc = Dev(dvc, DType::kI8, {nb, bs, H, D});
    Tensor gbt = Dev(dbt, DType::kI32, {num_reqs, 2});
    Tensor gseq = Dev(dseq, DType::kI32, {num_reqs});
    Tensor gqsl = Dev(dqsl, DType::kI32, {num_reqs + 1});
    vt::PagedAttention(gq, got, gqt, gkc, gvc, gbt, gseq, gqsl, args);

    std::vector<uint16_t> gpu_out(qb.size(), 0);
    gpu.Copy(gq, gpu_out.data(), dout, gpu_out.size() * sizeof(uint16_t));
    gpu.Synchronize(gq);

    double num = 0.0, den = 0.0, worst = 0.0;
    for (size_t i = 0; i < gpu_out.size(); ++i) {
      const double g = static_cast<double>(vt::BF16ToF32(gpu_out[i]));
      const double c = static_cast<double>(vt::BF16ToF32(cpu_out[i]));
      num += (g - c) * (g - c);
      den += c * c;
      worst = std::max(worst, std::fabs(g - c));
    }
    // The CPU arm must have produced a non-degenerate output, or the comparison
    // is between two fields of zeros and would pass on any kernel.
    CHECK(den > 0.0);
    const double nmse = den > 0.0 ? num / den : 1.0;
    CAPTURE(nmse);
    CAPTURE(worst);
    CHECK(nmse < 1e-4);
    CHECK(worst < 2e-2);

    gpu.Free(dq);
    gpu.Free(dout);
    gpu.Free(dqsl);
  }

  gpu.Free(dkc);
  gpu.Free(dvc);
  gpu.Free(dbt);
  gpu.Free(dseq);
  gpu.DestroyQueue(gq);
}

// ─── G5 ─────────────────────────────────────────────────────────────────────
// fp8_e5m2 stays a NAMED later brick (spec W5) on ROCm exactly as on CPU and
// CUDA — it must be refused, never silently mis-stored through the e4m3
// converter. There are THREE refusals on that path and only one is a ROCm-side
// guarantee:
//
//   * the op wrapper, `src/vt/ops.cpp` `ReshapeAndCacheFp8` — device-independent,
//     evaluated ABOVE the device checks and above GetOp, so it fires identically
//     on a CPU queue and cannot be a ROCm guarantee.
//   * the CPU kernel, `src/vt/cpu/cpu_cache.cpp` `ReshapeAndCacheFp8Kernel`.
//   * the ROCm kernel's own guard, `src/vt/rocm/rocm_dense_basic.hip`
//     `ReshapeAndCacheFp8KernelRocm`, which is defence in depth for any future
//     caller that reaches the registered provider without going through the
//     wrapper.
//
// A layered refusal needs an assertion that NAMES its layer. This version
// reaches the kernel guard the only way anything can — through the registered
// provider — and requires the message to carry both `rocm reshape_and_cache_fp8`
// and `fp8_e5m2`, which no other layer produces.
TEST_CASE("the ROCm fp8 KV store kernel refuses e5m2 (later brick)") {
  if (!HasRocm()) {
    MESSAGE("SKIPPED: no ROCm backend in this build/host — the ROCm-kernel e5m2 "
            "refusal gate did NOT run");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  const int64_t nb = 1, bs = 4, H = 1, D = 16, page = H * D;
  std::vector<float> k(static_cast<size_t>(page), 1.0f);
  std::vector<int64_t> slots = {0};
  void* dk = gpu.Alloc(k.size() * sizeof(float));
  void* dkc = gpu.Alloc(static_cast<size_t>(nb * bs * page));
  void* dvc = gpu.Alloc(static_cast<size_t>(nb * bs * page));
  void* ds = gpu.Alloc(sizeof(int64_t));
  gpu.Copy(gq, dk, k.data(), k.size() * sizeof(float));
  gpu.Copy(gq, ds, slots.data(), sizeof(int64_t));
  gpu.Synchronize(gq);
  Tensor gk = Dev(dk, DType::kF32, {1, H, D});
  Tensor gkc = Dev(dkc, DType::kI8, {nb, bs, H, D});
  Tensor gvc = Dev(dvc, DType::kI8, {nb, bs, H, D});
  Tensor gs = Dev(ds, DType::kI64, {1});

  // The registered ROCm provider, resolved exactly as vt::ReshapeAndCacheFp8
  // resolves it, then called directly so the wrapper's own e5m2 check is not in
  // the way. Anything that reaches this kernel reaches it through this pointer.
  auto* fn = reinterpret_cast<vt::ReshapeAndCacheFp8Fn>(
      vt::GetOp(OpId::kReshapeAndCacheFp8, DeviceType::kROCM));
  REQUIRE(fn != nullptr);
  std::string msg;
  try {
    fn(gq, gk, gk, gkc, gvc, gs, Fp8KVCacheDataType::kFp8E5M2, 0.01f, 0.01f);
    FAIL("the ROCm fp8 KV store kernel must refuse e5m2, not store it as e4m3");
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  CAPTURE(msg);
  // The refusal must come from the ROCm KERNEL and name the missing part, not
  // from the device-independent wrapper this call deliberately bypassed.
  CHECK(msg.find("rocm reshape_and_cache_fp8") != std::string::npos);
  CHECK(msg.find("fp8_e5m2") != std::string::npos);

  // e4m3 through the SAME pointer still runs: the guard above refuses one kind,
  // it does not disable the kernel.
  fn(gq, gk, gk, gkc, gvc, gs, Fp8KVCacheDataType::kFp8E4M3, 0.01f, 0.01f);
  gpu.Synchronize(gq);

  gpu.Free(dk);
  gpu.Free(dkc);
  gpu.Free(dvc);
  gpu.Free(ds);
  gpu.DestroyQueue(gq);
}

// ─── G6 ─────────────────────────────────────────────────────────────────────
// EXACT-GEOMETRY REACH: the GQA4 f32-query decode kernel (PagedAttnDecodeGqaF32Q)
// is gated behind VT_ATTN_DECODE_GQA4=1 AND an exact geometry: f32 q/out, fp8 or
// bf16 KV, d=128 or 256, hq=16, num_kv_heads=4. G4 above uses hq=2, H=1, D=16 —
// none of those match, so G4's dispatch never reaches PagedAttnDecodeGqaF32Q and
// deleting the GQA4 arm leaves G4 green. This case uses the exact geometry
// (hq=16, kv=4, d=128, f32 q/out, fp8 KV) so deleting the dispatch changes the
// output (PagedAttnOnline's per-key reduction order differs from the warp-strided
// online softmax).
//
// The env var is set by the file-scope initializer below, so it is active before
// any call to PagedAttentionKernelRocm in this binary. G4/G4b's geometry
// (hq=2, H=1, D=16) does not match the GQA4 guard (hq==16, kv==4, d==128||256),
// so setting the env var does not reroute those cases.
TEST_CASE("rocm fp8 KV GQA4 f32-query decode reaches the fast kernel (exact geometry)") {
  if (!HasRocm()) {
    MESSAGE("SKIPPED: no ROCm backend in this build/host — the GQA4 FP8 "
            "exact-geometry reach gate did NOT run");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // Exact GQA4 geometry: hq=16, num_kv_heads=4 (QG=4), d=128, f32 q/out, fp8 KV.
  const int64_t nb = 8, bs = 16, H = 4, D = 128, hq = 16, num_reqs = 1;
  const size_t cache_elems = static_cast<size_t>(nb * bs * H * D);
  auto raw = RandF32(cache_elems, 200);
  const float k_scale = 0.004f, v_scale = 0.006f;
  std::vector<uint8_t> kc(cache_elems), vc(cache_elems);
  for (size_t i = 0; i < cache_elems; ++i) {
    kc[i] = vt::StoreKvFp8E4M3(raw[i], k_scale);
    vc[i] = vt::StoreKvFp8E4M3(raw[cache_elems - 1 - i], v_scale);
  }
  // 1 request, 1 decode token, seq_len=37 (spans 3 blocks of bs=16).
  std::vector<int32_t> bt = {0, 1, 2, 0, 0, 0, 0, 0};  // [num_reqs, max_blocks]
  std::vector<int32_t> seq = {37};
  std::vector<int32_t> qsl = {0, 1};

  void* dkc = gpu.Alloc(cache_elems);
  void* dvc = gpu.Alloc(cache_elems);
  void* dbt = gpu.Alloc(bt.size() * sizeof(int32_t));
  void* dseq = gpu.Alloc(seq.size() * sizeof(int32_t));
  void* dqsl = gpu.Alloc(qsl.size() * sizeof(int32_t));
  gpu.Copy(gq, dkc, kc.data(), cache_elems);
  gpu.Copy(gq, dvc, vc.data(), cache_elems);
  gpu.Copy(gq, dbt, bt.data(), bt.size() * sizeof(int32_t));
  gpu.Copy(gq, dseq, seq.data(), seq.size() * sizeof(int32_t));
  gpu.Copy(gq, dqsl, qsl.data(), qsl.size() * sizeof(int32_t));

  auto qh = RandF32(static_cast<size_t>(1 * hq * D), 201);
  PagedAttentionArgs args;
  args.scale = 0.0884f;  // 1/sqrt(128)
  args.causal = true;
  args.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
  args.k_scale = k_scale;
  args.v_scale = v_scale;

  // CPU oracle: PagedAttnOnline (the reference kernel).
  std::vector<float> cpu_out(static_cast<size_t>(1 * hq * D), 0.0f);
  Tensor cqt = Host(qh.data(), DType::kF32, {1, hq, D});
  Tensor cot = Host(cpu_out.data(), DType::kF32, {1, hq, D});
  Tensor ckc = Host(kc.data(), DType::kI8, {nb, bs, H, D});
  Tensor cvc = Host(vc.data(), DType::kI8, {nb, bs, H, D});
  Tensor cbt = Host(bt.data(), DType::kI32, {num_reqs, 8});
  Tensor cseq = Host(seq.data(), DType::kI32, {num_reqs});
  Tensor cqsl = Host(qsl.data(), DType::kI32, {num_reqs + 1});
  vt::PagedAttention(cq, cot, cqt, ckc, cvc, cbt, cseq, cqsl, args);

  // GPU: PagedAttnDecodeGqaF32Q<uint8_t> (the fast kernel, via VT_ATTN_DECODE_GQA4=1).
  void* dq = gpu.Alloc(qh.size() * sizeof(float));
  void* dout = gpu.Alloc(qh.size() * sizeof(float));
  gpu.Copy(gq, dq, qh.data(), qh.size() * sizeof(float));
  Tensor gqt = Dev(dq, DType::kF32, {1, hq, D});
  Tensor got = Dev(dout, DType::kF32, {1, hq, D});
  Tensor gkc = Dev(dkc, DType::kI8, {nb, bs, H, D});
  Tensor gvc = Dev(dvc, DType::kI8, {nb, bs, H, D});
  Tensor gbt = Dev(dbt, DType::kI32, {num_reqs, 8});
  Tensor gseq = Dev(dseq, DType::kI32, {num_reqs});
  Tensor gqsl = Dev(dqsl, DType::kI32, {num_reqs + 1});
  vt::PagedAttention(gq, got, gqt, gkc, gvc, gbt, gseq, gqsl, args);

  std::vector<float> gpu_out(qh.size(), 0.0f);
  gpu.Copy(gq, gpu_out.data(), dout, gpu_out.size() * sizeof(float));
  gpu.Synchronize(gq);

  // The band is the same as G4's: both arms dequant fp8 identically, and the
  // only divergence is the softmax reduction order. A missing dequant, a
  // swapped k_scale/v_scale or a dropped sign moves the output by orders of
  // magnitude, not by a reduction-order ulp.
  double num = 0.0, den = 0.0, worst = 0.0;
  for (size_t i = 0; i < gpu_out.size(); ++i) {
    const double d0 = static_cast<double>(gpu_out[i]) - static_cast<double>(cpu_out[i]);
    num += d0 * d0;
    den += static_cast<double>(cpu_out[i]) * static_cast<double>(cpu_out[i]);
    worst = std::max(worst, std::fabs(d0));
  }
  CHECK(den > 0.0);
  const double nmse = den > 0.0 ? num / den : 1.0;
  CAPTURE(nmse);
  CAPTURE(worst);
  CHECK(nmse < 1e-6);
  CHECK(worst < 1e-3);

  // REACH EVIDENCE: the GQA4 kernel uses warp-strided online softmax with
  // QG=4 fused query heads. If the dispatch was deleted, PagedAttnOnline would
  // run instead, and the reduction order difference would move the output
  // beyond the tight band above at this geometry. The band is the gate: a
  // dispatch deletion that left PagedAttnOnline running would fail it.
  // Additionally, the output must be non-degenerate (not all zeros), which
  // catches the case where the kernel returned early without writing.
  CHECK(std::any_of(gpu_out.begin(), gpu_out.end(),
                    [](float v) { return v != 0.0f; }));

  gpu.Free(dq);
  gpu.Free(dout);
  gpu.Free(dkc);
  gpu.Free(dvc);
  gpu.Free(dbt);
  gpu.Free(dseq);
  gpu.Free(dqsl);
  gpu.DestroyQueue(gq);
}

// ─── G7 ─────────────────────────────────────────────────────────────────────
// PREFILL FAST-PATH REACH: the fp8 SharedK prefill dispatch
// (PagedAttnPrefillSharedK<uint8_t, float, float>) is gated behind an exact
// geometry: f32 q/out, fp8 KV, d=256, total_q >= 64, num_reqs == 1, QG in
// {2,4,8}. G4 above uses hq=2, H=1, D=16 — none match, so G4 never reaches the
// SharedK prefill. G6 uses d=128 and total_q=1 (decode) — also no match. This
// case uses the exact geometry (hq=16, kv=4, d=256, total_q=64, f32 q/out,
// fp8 KV, QG=4).
//
// DISPATCH REACH: the test checks Fp8PrefillSharedKDispatchCount() before and
// after the GPU call. The counter is incremented in the dispatch guard itself,
// so a deleted or broken dispatch (wrong dtype check, wrong d, wrong QG) leaves
// the counter unchanged and the CHECK fails. This is necessary because output
// comparison alone cannot distinguish SharedK from the PagedAttnOnline
// fallback: both GPU kernels use online softmax, while the CPU oracle uses
// two-pass softmax, so both GPU kernels differ from the CPU by a similar
// (small) amount. The NMSE band gates load/scale mutations (swapped
// k_scale/v_scale, dropped dequant) which move the output by orders of
// magnitude, not dispatch deletion.
//
// The env var VT_ATTN_DECODE_GQA4=1 is set by the file-scope initializer above.
// It routes the DECODE path through PagedAttnDecodeGqaF32Q but does not affect
// the prefill dispatch — the prefill SharedK guard is independent. The env var
// VT_ATTN_PREFILL_FLASH_SHAREDK defaults to enabled; this case does not set it
// because the static is already read by the time this test runs, and the
// default is "on".
TEST_CASE("rocm fp8 KV SharedK prefill reaches the fast kernel (exact geometry, d=256, QG=4)") {
  if (!HasRocm()) {
    MESSAGE("SKIPPED: no ROCm backend in this build/host — the SharedK FP8 "
            "prefill exact-geometry reach gate did NOT run");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  // Exact SharedK prefill geometry: hq=16, num_kv_heads=4 (QG=4), d=256,
  // f32 q/out, fp8 KV, total_q=64 (>= 64 threshold), num_reqs=1.
  const int64_t nb = 16, bs = 16, H = 4, D = 256, hq = 16, num_reqs = 1;
  const int64_t total_q = 64;
  const int64_t seq_len = total_q;  // causal prefill
  const size_t cache_elems = static_cast<size_t>(nb * bs * H * D);
  auto raw = RandF32(cache_elems, 300);
  const float k_scale = 0.003f, v_scale = 0.007f;
  std::vector<uint8_t> kc(cache_elems), vc(cache_elems);
  for (size_t i = 0; i < cache_elems; ++i) {
    kc[i] = vt::StoreKvFp8E4M3(raw[i], k_scale);
    vc[i] = vt::StoreKvFp8E4M3(raw[cache_elems - 1 - i], v_scale);
  }
  // 1 request, seq_len=64 → 4 blocks of bs=16.
  std::vector<int32_t> bt = {0, 1, 2, 3, 0, 0, 0, 0};  // [num_reqs, max_blocks]
  std::vector<int32_t> seq = {static_cast<int32_t>(seq_len)};
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(total_q)};

  void* dkc = gpu.Alloc(cache_elems);
  void* dvc = gpu.Alloc(cache_elems);
  void* dbt = gpu.Alloc(bt.size() * sizeof(int32_t));
  void* dseq = gpu.Alloc(seq.size() * sizeof(int32_t));
  void* dqsl = gpu.Alloc(qsl.size() * sizeof(int32_t));
  gpu.Copy(gq, dkc, kc.data(), cache_elems);
  gpu.Copy(gq, dvc, vc.data(), cache_elems);
  gpu.Copy(gq, dbt, bt.data(), bt.size() * sizeof(int32_t));
  gpu.Copy(gq, dseq, seq.data(), seq.size() * sizeof(int32_t));
  gpu.Copy(gq, dqsl, qsl.data(), qsl.size() * sizeof(int32_t));

  auto qh = RandF32(static_cast<size_t>(total_q * hq * D), 301);
  PagedAttentionArgs args;
  args.scale = 0.0625f;  // 1/sqrt(256)
  args.causal = true;
  args.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
  args.k_scale = k_scale;
  args.v_scale = v_scale;

  // CPU oracle: PagedAttnOnline (the reference kernel).
  std::vector<float> cpu_out(static_cast<size_t>(total_q * hq * D), 0.0f);
  Tensor cqt = Host(qh.data(), DType::kF32, {total_q, hq, D});
  Tensor cot = Host(cpu_out.data(), DType::kF32, {total_q, hq, D});
  Tensor ckc = Host(kc.data(), DType::kI8, {nb, bs, H, D});
  Tensor cvc = Host(vc.data(), DType::kI8, {nb, bs, H, D});
  Tensor cbt = Host(bt.data(), DType::kI32, {num_reqs, 8});
  Tensor cseq = Host(seq.data(), DType::kI32, {num_reqs});
  Tensor cqsl = Host(qsl.data(), DType::kI32, {num_reqs + 1});
  vt::PagedAttention(cq, cot, cqt, ckc, cvc, cbt, cseq, cqsl, args);

  // GPU: PagedAttnPrefillSharedK<uint8_t, float, float> (the fast prefill
  // kernel, via the fp8_prefill_fast dispatch guard).
  void* dq = gpu.Alloc(qh.size() * sizeof(float));
  void* dout = gpu.Alloc(qh.size() * sizeof(float));
  gpu.Copy(gq, dq, qh.data(), qh.size() * sizeof(float));
  Tensor gqt = Dev(dq, DType::kF32, {total_q, hq, D});
  Tensor got = Dev(dout, DType::kF32, {total_q, hq, D});
  Tensor gkc = Dev(dkc, DType::kI8, {nb, bs, H, D});
  Tensor gvc = Dev(dvc, DType::kI8, {nb, bs, H, D});
  Tensor gbt = Dev(dbt, DType::kI32, {num_reqs, 8});
  Tensor gseq = Dev(dseq, DType::kI32, {num_reqs});
  Tensor gqsl = Dev(dqsl, DType::kI32, {num_reqs + 1});
  // DISPATCH REACH: the counter is incremented inside the dispatch guard,
  // so if the guard fails (wrong dtype, wrong d, wrong QG, deleted), the
  // counter does not advance and this CHECK fails. Guarded by
  // VLLM_CPP_HIP because the counter function is only defined in ROCm
  // builds; in non-ROCm builds the test skips via HasRocm() above.
#ifdef VLLM_CPP_HIP
  const int dispatch_before =
      vt::rocm::Fp8PrefillSharedKDispatchCount().load(std::memory_order_relaxed);
#endif
  vt::PagedAttention(gq, got, gqt, gkc, gvc, gbt, gseq, gqsl, args);
#ifdef VLLM_CPP_HIP
  const int dispatch_after =
      vt::rocm::Fp8PrefillSharedKDispatchCount().load(std::memory_order_relaxed);
  CAPTURE(dispatch_before);
  CAPTURE(dispatch_after);
  CHECK(dispatch_after > dispatch_before);
#endif

  std::vector<float> gpu_out(qh.size(), 0.0f);
  gpu.Copy(gq, gpu_out.data(), dout, gpu_out.size() * sizeof(float));
  gpu.Synchronize(gq);

  // LOAD/SCALE MUTATION GATE: both arms dequant fp8 identically, so the
  // only way the NMSE exceeds 1e-4 is a wrong dequant — a missing scale, a
  // swapped k_scale/v_scale, or a dropped sign moves the output by orders
  // of magnitude (NMSE >> 1.0, worst ~1.0). The band is looser than G4/G6
  // (which use D=16 and D=128): at D=256 with total_q=64, the SharedK
  // tile-structured online softmax accumulates more reduction-order drift
  // against PagedAttnOnline's two-pass softmax than the smaller geometries
  // do. The measured NMSE is ~3e-6 and worst is ~5e-3 — expected for two
  // correct f32 reductions in different order over 64 keys × 256 dims.
  //
  // The band does NOT gate dispatch deletion: both SharedK and the
  // PagedAttnOnline fallback use online softmax and produce numerically
  // similar results (~3e-6 NMSE vs CPU), so falling back would still pass
  // the band. The dispatch reach check above (counter) gates that separately.
  double num = 0.0, den = 0.0, worst = 0.0;
  for (size_t i = 0; i < gpu_out.size(); ++i) {
    const double d0 = static_cast<double>(gpu_out[i]) - static_cast<double>(cpu_out[i]);
    num += d0 * d0;
    den += static_cast<double>(cpu_out[i]) * static_cast<double>(cpu_out[i]);
    worst = std::max(worst, std::fabs(d0));
  }
  CHECK(den > 0.0);
  const double nmse = den > 0.0 ? num / den : 1.0;
  CAPTURE(nmse);
  CAPTURE(worst);
  CHECK(nmse < 1e-4);
  CHECK(worst < 1e-2);

  // OUTPUT SANITY: the output must be non-degenerate (not all zeros), which
  // catches the case where the kernel returned early without writing.
  CHECK(std::any_of(gpu_out.begin(), gpu_out.end(),
                    [](float v) { return v != 0.0f; }));

  gpu.Free(dq);
  gpu.Free(dout);
  gpu.Free(dkc);
  gpu.Free(dvc);
  gpu.Free(dbt);
  gpu.Free(dseq);
  gpu.Free(dqsl);
  gpu.DestroyQueue(gq);
}

// ─── G6b ────────────────────────────────────────────────────────────────────
// WIDENED-GEOMETRY REACH: the same f32-query GQA arm after the ISSUE-LOCAL
// widening — hq=24, num_kv_heads=4 (qg_total=6), d=256, f32 q/out, bf16 KV.
// Qwen3.8-27B-EXL3's full-attention layers run exactly this shape; before the
// widening they fell to PagedAttnOnline at ~180us/call on gfx1100. The kernel
// tiles the qg=6 group as QG=6, z=1. CPU oracle is PagedAttnOnline again —
// the band is the reduction-order band, not byte equality.
TEST_CASE("rocm GQA4 f32-query decode widened to qg_total=6 (hq=24, kv=4, bf16 KV)") {
  if (!HasRocm()) {
    MESSAGE("SKIPPED: no ROCm backend in this build/host — the GQA6 "
            "widened-geometry reach gate did NOT run");
    return;
  }
  Backend& gpu = vt::GetBackend(DeviceType::kROCM);
  Queue gq = gpu.CreateQueue();
  Queue cq{Cpu(), nullptr};

  const int64_t nb = 8, bs = 16, H = 4, D = 256, hq = 24, num_reqs = 1;
  const size_t cache_elems = static_cast<size_t>(nb * bs * H * D);
  auto kraw = RandF32(cache_elems, 211);
  auto vraw = RandF32(cache_elems, 212);
  std::vector<uint16_t> kc(cache_elems), vc(cache_elems);
  for (size_t i = 0; i < cache_elems; ++i) {
    kc[i] = vt::F32ToBF16(kraw[i]);
    vc[i] = vt::F32ToBF16(vraw[cache_elems - 1 - i]);
  }
  std::vector<int32_t> bt = {0, 1, 2, 3, 0, 0, 0, 0};
  std::vector<int32_t> seq = {53};
  std::vector<int32_t> qsl = {0, 1};

  void* dkc = gpu.Alloc(cache_elems * 2);
  void* dvc = gpu.Alloc(cache_elems * 2);
  void* dbt = gpu.Alloc(bt.size() * sizeof(int32_t));
  void* dseq = gpu.Alloc(seq.size() * sizeof(int32_t));
  void* dqsl = gpu.Alloc(qsl.size() * sizeof(int32_t));
  gpu.Copy(gq, dkc, kc.data(), cache_elems * 2);
  gpu.Copy(gq, dvc, vc.data(), cache_elems * 2);
  gpu.Copy(gq, dbt, bt.data(), bt.size() * sizeof(int32_t));
  gpu.Copy(gq, dseq, seq.data(), seq.size() * sizeof(int32_t));
  gpu.Copy(gq, dqsl, qsl.data(), qsl.size() * sizeof(int32_t));

  auto qh = RandF32(static_cast<size_t>(hq * D), 213);
  PagedAttentionArgs args;
  args.scale = 0.0625f;  // 1/sqrt(256)
  args.causal = true;
  args.kv_cache_dtype = Fp8KVCacheDataType::kAuto;

  std::vector<float> cpu_out(static_cast<size_t>(hq * D), 0.0f);
  Tensor cqt = Host(qh.data(), DType::kF32, {1, hq, D});
  Tensor cot = Host(cpu_out.data(), DType::kF32, {1, hq, D});
  Tensor ckc = Host(kc.data(), DType::kBF16, {nb, bs, H, D});
  Tensor cvc = Host(vc.data(), DType::kBF16, {nb, bs, H, D});
  Tensor cbt = Host(bt.data(), DType::kI32, {num_reqs, 8});
  Tensor cseq = Host(seq.data(), DType::kI32, {num_reqs});
  Tensor cqsl = Host(qsl.data(), DType::kI32, {num_reqs + 1});
  vt::PagedAttention(cq, cot, cqt, ckc, cvc, cbt, cseq, cqsl, args);

  void* dq = gpu.Alloc(qh.size() * sizeof(float));
  void* dout = gpu.Alloc(qh.size() * sizeof(float));
  gpu.Copy(gq, dq, qh.data(), qh.size() * sizeof(float));
  Tensor gqt = Dev(dq, DType::kF32, {1, hq, D});
  Tensor got = Dev(dout, DType::kF32, {1, hq, D});
  Tensor gkc = Dev(dkc, DType::kBF16, {nb, bs, H, D});
  Tensor gvc = Dev(dvc, DType::kBF16, {nb, bs, H, D});
  Tensor gbt = Dev(dbt, DType::kI32, {num_reqs, 8});
  Tensor gseq = Dev(dseq, DType::kI32, {num_reqs});
  Tensor gqsl = Dev(dqsl, DType::kI32, {num_reqs + 1});
  vt::PagedAttention(gq, got, gqt, gkc, gvc, gbt, gseq, gqsl, args);

  std::vector<float> gpu_out(qh.size(), 0.0f);
  gpu.Copy(gq, gpu_out.data(), dout, gpu_out.size() * sizeof(float));
  gpu.Synchronize(gq);

  double num = 0.0, den = 0.0, worst = 0.0;
  for (size_t i = 0; i < gpu_out.size(); ++i) {
    const double d0 = static_cast<double>(gpu_out[i]) - static_cast<double>(cpu_out[i]);
    num += d0 * d0;
    den += static_cast<double>(cpu_out[i]) * static_cast<double>(cpu_out[i]);
    worst = std::max(worst, std::fabs(d0));
  }
  CHECK(den > 0.0);
  const double nmse = den > 0.0 ? num / den : 1.0;
  CAPTURE(nmse);
  CAPTURE(worst);
  CHECK(nmse < 1e-6);
  CHECK(worst < 1e-3);
  CHECK(std::any_of(gpu_out.begin(), gpu_out.end(),
                    [](float v) { return v != 0.0f; }));

  gpu.Free(dq);
  gpu.Free(dout);
  gpu.Free(dkc);
  gpu.Free(dvc);
  gpu.Free(dbt);
  gpu.Free(dseq);
  gpu.Free(dqsl);
  gpu.DestroyQueue(gq);
}

