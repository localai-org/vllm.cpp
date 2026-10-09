#pragma once
// The ROCm backend's `__device__` half/bfloat16 codec — ONE copy.
//
// These four functions are transcriptions of `vt::F16ToF32`, `vt::BF16ToF32`,
// `vt::F32ToF16` and `vt::F32ToBF16` (`include/vt/dtype.h`), which are the host
// definitions every CPU reference in this tree rounds through. They are NOT
// `__half2float` / `__float2half` / `__bfloat162float` / `__float2bfloat16`,
// and the difference is the point: a device gate that asserts BYTE equality
// against a CPU arm is asserting equality with those host functions, so the
// device has to run the same arithmetic and not merely a conversion that agrees
// on the common cases. The hardware conversion instruction is a perf lever, not
// a correctness requirement -- the same sentence `Dp4a` carries in
// `rocm_grouped_gemm.hip`.
//
// They lived in that file's anonymous namespace until `BACKEND-ROCM-EXL3`
// needed the same two conversions for the trellis decode. A second copy of a
// codec that must agree byte for byte is exactly the drift this repository
// files bugs about, so the copy moved here rather than being duplicated.
#include <hip/hip_runtime.h>

#include <cstdint>

namespace vt::rocm {

// gfx11 hardware arms. v_cvt_f32_f16 and v_cvt_f16_f32 (round-to-nearest-even,
// fp16 denormals kept: the default MODE) equal the transcriptions below on
// EVERY input bit pattern — test_rocm_f16_codec walks all 2^32 f32 and 2^16
// fp16 patterns against them. The one difference, the hardware quieting a
// signalling fp16 NaN (f32 bit 22), is xored back out. Inline asm rather
// than __float2half: fptrunc(fmul) may select v_fma_mixlo_f16, one rounding
// instead of two. Constant arguments take the transcription so they fold.
#if defined(__HIP_DEVICE_COMPILE__) && defined(__GFX11__)
#define VT_ROCM_F16_CODEC_HW 1
#endif

// `vt::F16ToF32` (dtype.h): fp16 bits -> f32, exact for every input.
__device__ inline float DF16ToF32(uint16_t h) {
#if defined(VT_ROCM_F16_CODEC_HW)
  if (!__builtin_constant_p(h)) {
    float r;
    asm("v_cvt_f32_f16 %0, %1" : "=v"(r) : "v"(static_cast<uint32_t>(h)));
    const uint32_t snan = static_cast<uint32_t>((h & 0x7fffu) - 0x7c01u) < 0x1ffu;
    return __uint_as_float(__float_as_uint(r) ^ (snan << 22));
  }
#endif
  uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
  uint32_t exp = (h >> 10) & 0x1F;
  uint32_t mant = h & 0x3FF;
  if (exp == 0x1F) return __int_as_float(sign | 0x7F800000 | (mant << 13));
  if (exp == 0) {
    if (mant == 0) return __int_as_float(sign);
    int shift = 0;
    while ((mant & 0x400) == 0) { mant <<= 1; ++shift; }
    mant &= 0x3FF;
    return __int_as_float(sign | ((113 - shift) << 23) | (mant << 13));
  }
  return __int_as_float(sign | ((exp + 112) << 23) | (mant << 13));
}

// `vt::BF16ToF32` (dtype.h): the bf16 bit pattern IS the f32's top half.
__device__ inline float DBF16ToF32(uint16_t b) {
  return __int_as_float(static_cast<uint32_t>(b) << 16);
}

// `vt::F32ToBF16`: round-to-nearest-even, with a NaN kept as a quiet NaN.
__device__ inline uint16_t DF32ToBF16(float f) {
  uint32_t u = __float_as_int(f);
  if ((u & 0x7F800000) == 0x7F800000 && (u & 0x7FFFFF))
    return static_cast<uint16_t>((u >> 16) | 0x0040);
  uint32_t rounding = 0x7FFF + ((u >> 16) & 1);
  return static_cast<uint16_t>((u + rounding) >> 16);
}

// `vt::F32ToF16`: round-to-nearest-even, including the subnormal arm and the
// overflow-to-infinity arm. The `rem == mid && (half & 1)` tie is what makes it
// EVEN rather than away-from-zero, and it is the tie the host function takes.
__device__ inline uint16_t DF32ToF16(float f) {
#if defined(VT_ROCM_F16_CODEC_HW)
  if (!__builtin_constant_p(f)) {
    uint32_t r;
    asm("v_cvt_f16_f32 %0, %1" : "=v"(r) : "v"(f));
    return static_cast<uint16_t>(r);
  }
#endif
  uint32_t u = __float_as_uint(f);
  uint16_t sign = static_cast<uint16_t>((u >> 16) & 0x8000);
  int32_t exp = static_cast<int32_t>((u >> 23) & 0xFF) - 127 + 15;
  uint32_t mant = u & 0x7FFFFF;
  if (((u >> 23) & 0xFF) == 0xFF)
    return static_cast<uint16_t>(sign | 0x7C00 | (mant ? 0x200 | (mant >> 13) : 0));
  if (exp >= 0x1F) return static_cast<uint16_t>(sign | 0x7C00);
  if (exp <= 0) {
    if (exp < -10) return sign;
    mant |= 0x800000;
    uint32_t shift = static_cast<uint32_t>(14 - exp);
    uint32_t half = mant >> shift;
    uint32_t rem = mant & ((1u << shift) - 1);
    uint32_t mid = 1u << (shift - 1);
    if (rem > mid || (rem == mid && (half & 1))) ++half;
    return static_cast<uint16_t>(sign | half);
  }
  uint32_t half = static_cast<uint32_t>(exp << 10) | (mant >> 13);
  uint32_t rem = mant & 0x1FFF;
  if (rem > 0x1000 || (rem == 0x1000 && (half & 1))) ++half;
  return static_cast<uint16_t>(sign | half);
}

}  // namespace vt::rocm
