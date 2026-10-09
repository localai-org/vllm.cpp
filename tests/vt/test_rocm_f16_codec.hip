// rocm_f16_codec.h's device DF16ToF32 / DF32ToF16 against the dtype.h
// transcriptions on EVERY input bit pattern: all 2^16 fp16 and all 2^32 f32
// values, NaNs included. On gfx11 the header routes non-constant arguments to
// v_cvt_f32_f16 / v_cvt_f16_f32; this is the proof that the hardware arm is
// byte-identical to the arithmetic every CPU reference rounds through.
// Exit 77 without a ROCm device.
#include <hip/hip_runtime.h>

#include <cstdint>
#include <cstdio>

#include "vt/dtype.h"
#include "vt/rocm/rocm_f16_codec.h"

namespace {

// The software transcriptions, verbatim (the header's non-gfx11 arm).
__device__ float RefF16ToF32(uint16_t h) {
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

__device__ uint16_t RefF32ToF16(float f) {
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

// out[0]: f32->f16 mismatches, out[1]: fp16->f32 mismatches, out[2]: inputs
// visited (coverage), first[0/1]: first mismatching input.
__global__ void CodecAll(unsigned long long* out, uint32_t* first, uint32_t* f16_dev) {
  const uint64_t stride = (uint64_t)gridDim.x * blockDim.x;
  unsigned long long visited = 0;
  for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < (1ull << 32);
       i += stride) {
    const uint32_t u = static_cast<uint32_t>(i);
    const float f = __uint_as_float(u);
    if (vt::rocm::DF32ToF16(f) != RefF32ToF16(f)) {
      atomicAdd(&out[0], 1ull);
      atomicCAS(&first[0], 0xffffffffu, u);
    }
    if (u < 65536u) {
      const uint16_t h = static_cast<uint16_t>(u);
      const uint32_t got = __float_as_uint(vt::rocm::DF16ToF32(h));
      f16_dev[u] = got;
      if (got != __float_as_uint(RefF16ToF32(h))) {
        atomicAdd(&out[1], 1ull);
        atomicCAS(&first[1], 0xffffffffu, u);
      }
    }
    ++visited;
  }
  atomicAdd(&out[2], visited);
}

bool Hip(hipError_t e, const char* what) {
  if (e == hipSuccess) return true;
  std::fprintf(stderr, "%s: %s\n", what, hipGetErrorString(e));
  return false;
}

}  // namespace

int main() {
  int devices = 0;
  if (hipGetDeviceCount(&devices) != hipSuccess || devices == 0) {
    std::printf("no ROCm device; skipping\n");
    return 77;
  }
  unsigned long long* out = nullptr;
  uint32_t* first = nullptr;
  uint32_t* f16_dev = nullptr;
  if (!Hip(hipMalloc(&out, 3 * sizeof(unsigned long long)), "alloc") ||
      !Hip(hipMalloc(&first, 2 * sizeof(uint32_t)), "alloc") ||
      !Hip(hipMalloc(&f16_dev, 65536 * sizeof(uint32_t)), "alloc") ||
      !Hip(hipMemset(out, 0, 3 * sizeof(unsigned long long)), "memset") ||
      !Hip(hipMemset(first, 0xff, 2 * sizeof(uint32_t)), "memset"))
    return 1;
  CodecAll<<<96 * 64, 256>>>(out, first, f16_dev);
  unsigned long long h[3];
  uint32_t fh[2];
  static uint32_t f16_host[65536];
  if (!Hip(hipGetLastError(), "launch") ||
      !Hip(hipMemcpy(h, out, sizeof(h), hipMemcpyDeviceToHost), "copy") ||
      !Hip(hipMemcpy(fh, first, sizeof(fh), hipMemcpyDeviceToHost), "copy") ||
      !Hip(hipMemcpy(f16_host, f16_dev, sizeof(f16_host), hipMemcpyDeviceToHost), "copy"))
    return 1;
  // The device fp16 decode against the HOST vt::F16ToF32 too: the function
  // every CPU reference rounds through.
  unsigned long long host_mismatch = 0;
  for (uint32_t u = 0; u < 65536u; ++u) {
    const float r = vt::F16ToF32(static_cast<uint16_t>(u));
    uint32_t rb;
    __builtin_memcpy(&rb, &r, 4);
    host_mismatch += rb != f16_host[u];
  }
  std::printf("visited %llu of %llu f32 patterns\n", h[2], 1ull << 32);
  std::printf("f32->f16 mismatches %llu (first 0x%08x)\n", h[0], fh[0]);
  std::printf("f16->f32 mismatches %llu (first 0x%04x), vs host vt::F16ToF32 %llu\n", h[1],
              fh[1], host_mismatch);
  const bool ok = h[2] == (1ull << 32) && h[0] == 0 && h[1] == 0 && host_mismatch == 0;
  std::printf("%s\n", ok ? "PASS" : "FAIL");
  (void)hipFree(out);
  (void)hipFree(first);
  (void)hipFree(f16_dev);
  return ok ? 0 : 1;
}
