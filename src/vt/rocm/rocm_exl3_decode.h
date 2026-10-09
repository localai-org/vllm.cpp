// Shared EXL3 trellis decode — the scalar codec three ROCm EXL3 TUs use.
//
// Source of every function: the portable CPU reference's device transcription
// in rocm_exl3.hip (itself a 1:1 port of exllamav3 @
// 2398c05635fbbad01a0a51dce63c85c6c8a8450e — exl3_dq.cuh:15-31, codebook.cuh,
// quantize.py:22-42). Extracted so rocm_exl3.hip (the fused GEMM), its GEMV
// sibling and rocm_exl3_recon.hip (the reconstruct+GEMM arm) decode identically
// rather than three copies drifting.
//
// __device__ inline — header-only by construction, same shape as
// rocm_f16_codec.h which it depends on.
#pragma once

#include <cstdint>

#include "vt/rocm/rocm_f16_codec.h"

namespace vt::rocm {

// cpu_exl3_dequant.cpp RoundHalf.
__device__ inline float Exl3RoundHalf(float v) { return DF16ToF32(DF32ToF16(v)); }

// The tile's uint32 view (exl3_dq.cuh:25-26), assembled by hand so the trellis
// may sit at any alignment inside the loaded buffer.
__device__ inline uint32_t Exl3TileWord32(const uint16_t* tile, int index) {
  return static_cast<uint32_t>(tile[2 * index]) |
         (static_cast<uint32_t>(tile[2 * index + 1]) << 16);
}

// Exl3TileCodeword, verbatim. `+ 256*bits` keeps the tail-biting wrap
// non-negative; the `% words32` on the word index is the wrap itself.
__device__ inline uint16_t Exl3TileCodeword(const uint16_t* tile, int bits, int t) {
  const int words32 = bits * 256 / 32;
  const int b0 = t * bits + bits - 16 + 256 * bits;
  const int b1 = b0 + 16;
  const int i0 = b0 / 32;
  const int i1 = (b1 - 1) / 32;
  const int s0 = (i1 + 1) * 32 - b1;
  const uint32_t a = Exl3TileWord32(tile, i0 % words32);
  const uint32_t b = Exl3TileWord32(tile, i1 % words32);
  const uint64_t merged = (static_cast<uint64_t>(a) << 32) | b;
  return static_cast<uint16_t>((merged >> s0) & 0xffffu);
}

// The HALF-INTEGER twin (BACKEND-ROCM frac rates), same read shape as
// `Exl3TileCodeword` above and a device transcription of the host
// `vt::Exl3FracTileCodeword` (cpu_exl3_dequant.cpp). Positions alternate
// KA / KA+1 state bits with the extra bit on odd positions (mask 0xAAAA), so
// position t's window ends at ring bit
//   S(t) = KA*(t+1) + ((t&15)+1)/2 + 8*(t/16)
// and the ring is tail-biting mod tile_bits = 256*KA + 128 — the FRAC ring
// length, not a copied +256*bits (that copy is the off-by-128 the spec warns
// about near the wrap).
__device__ inline uint16_t Exl3FracTileCodeword(const uint16_t* tile, int ka, int t) {
  const int tile_bits = 256 * ka + 128;
  const int words32 = tile_bits / 32;  // 8*KA + 4
  const int b0 = ka * (t + 1) + ((t & 15) + 1) / 2 + 8 * (t / 16) + tile_bits - 16;
  const int b1 = b0 + 16;
  const int i0 = b0 / 32;
  const int i1 = (b1 - 1) / 32;
  const int s0 = (i1 + 1) * 32 - b1;
  const uint32_t a = Exl3TileWord32(tile, i0 % words32);
  const uint32_t b = Exl3TileWord32(tile, i1 % words32);
  const uint64_t merged = (static_cast<uint64_t>(a) << 32) | b;
  return static_cast<uint16_t>((merged >> s0) & 0xffffu);
}

// Exl3DecodeCodeword (codebook.cuh:56-90). All THREE codebooks, because an AMD
// box has no reason to see fewer artifacts than an NVIDIA one:
//   cb 0  3INST, the DEFAULT — a checkpoint that ships neither an `mcg` nor a
//         `mul1` marker lands here, which is every stock turboderp/*-exl3
//   cb 1  MCG, which the SparkInfer DeepSeek-V4 artifact marks
//   cb 2  mul1, a DIFFERENT SHAPE and not a third multiplier: the product's
//         four UNSIGNED BYTES are summed into a fixed accumulator, the sum is
//         REINTERPRETED as an fp16 bit pattern, and an fp16 affine map turns
//         it into the codebook value
// `codebook` is validated on the host before launch, so there is no fall-off
// arm here; a device VT_CHECK does not exist and a silent wrong decode is
// exactly the failure this family documents (right distribution, no
// correlation).
__device__ inline float Exl3DecodeCodeword(uint16_t codeword, int codebook) {
  uint32_t x = static_cast<uint32_t>(codeword);
  if (codebook == 2) {
    x *= 0x83DCD12Du;
    // `__dp4a(x, 0x01010101u, acc)` == acc + the four UNSIGNED bytes of x.
    // v_dot4_u32_u8 folds the four masks/adds into one instruction; integer
    // arithmetic is exact either way. 0x6400 is chosen so the
    // reinterpretation is EXACT: the fp16 binade [1024, 2048) has an ULP of
    // exactly 1.0 and the byte sum is at most 1020, so the pattern never
    // leaves that binade.
    const uint32_t sum = __builtin_amdgcn_udot4(x, 0x01010101u, 0x6400u, false);
    const float h = DF16ToF32(static_cast<uint16_t>(sum));
    // BIT PATTERNS, not the rounded decimals upstream's comments carry:
    // 0x1eee is 887/131072 and 0xc931 is -1329/128.
    const float k_inv = DF16ToF32(static_cast<uint16_t>(0x1eeeu));
    const float k_bias = DF16ToF32(static_cast<uint16_t>(0xc931u));
    // Upstream ends in __hfma, ONE rounding. Evaluating the product and the
    // sum separately in f32 reproduces it exactly (the CPU arm proves why over
    // all 1021 reachable sums), and -ffp-contract=off keeps the compiler from
    // fusing them into a different single rounding.
    return Exl3RoundHalf(h * k_inv + k_bias);
  }
  if (codebook == 0) {
    x *= 89226354u;
    x += 64248484u;
  } else {
    x *= 0xCBAC1FEDu;
  }
  x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
  // The halves of x are the two f16 codebook contributions; their f16 sum
  // is the weight. f32-add of two f16-exact values is exact in f32, so RN
  // back to f16 equals __hadd — one V_ADD_F16 replaces both DF16ToF32
  // converts, the f32 add, and Exl3RoundHalf.
  const half2 pair = __builtin_bit_cast(half2, x);
  return __hadd(__low2half(pair), __high2half(pair));
}

// mul1 pair via the u32xu8 byte-dot (codebook.cuh:25-41,76-89). The constants
// are BIT PATTERNS: 0x1eee = 887/131072, 0xc931 = -1329/128, and 0x6400's
// binade makes the integer reinterpretation exact (cuda_exl3.cu:281-304).
// __builtin_amdgcn_udot4 is v_dot4_u32_u8 — acc + the four UNSIGNED bytes of
// x in one instruction; __dp4a is not in this HIP's device headers. One
// packed __hfma2 ends the pair where the scalar arm runs two f32 chains;
// the CPU exhaustively proves both roundings equal, so bits are unchanged.
//
// Only the LOW 16 bits of x0/x1 are read; the high halves may hold anything.
//
// gfx11: v_mul_lo_u32 issues at quarter rate and v_dot4_u32_u8 at half rate
// (measured, review/main/irate.hip), and those two were ~70% of the decode's
// issue cycles. The codeword is 16 bits, so x*0x83DCD12D mod 2^32 splits into
//   P      = x * 0xD12D                    exact in 32 bits (v_mad_u32_u16)
//   P.hi16 = (x * 0x83DC + P.hi16) mod 2^16 (v_mad_u16, op_sel dest-hi keeps
//                                            P.lo16 — ISA VOP3 OPSEL[3])
// and the byte sums come from v_sad_u8 (|byte - 0| summed, + acc); the pair's
// second sum lands in the high half through v_sad_hi_u8, so the packed fp16
// pattern needs no perm. Six full-rate ops replace 2 mul_lo + 2 dot4 + 1 pack.
// Bit-identical over all 2^16 codewords with garbage high halves
// (review/main/hash16.hip, test_exl3_rocm).
__device__ __forceinline__ half2 Exl3DecodePairCb2Dp4a(uint32_t x0,
                                                        uint32_t x1) {
#if defined(__GFX11__)
  uint32_t p0, p1, packed;
  asm("v_mad_u32_u16 %0, %1, 0xd12d, 0" : "=v"(p0) : "v"(x0));
  asm("v_mad_u32_u16 %0, %1, 0xd12d, 0" : "=v"(p1) : "v"(x1));
  asm("v_mad_u16 %0, %1, 0x83dc, %0 op_sel:[0,0,1,1]" : "+v"(p0) : "v"(x0));
  asm("v_mad_u16 %0, %1, 0x83dc, %0 op_sel:[0,0,1,1]" : "+v"(p1) : "v"(x1));
  asm("v_sad_u8 %0, %1, 0, 0x64006400" : "=v"(packed) : "v"(p0));
  asm("v_sad_hi_u8 %0, %1, 0, %0" : "+v"(packed) : "v"(p1));
  const half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
  const half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
  return __hfma2(__builtin_bit_cast(half2, packed), k_inv_h2, k_bias_h2);
#else
  x0 = (x0 & 0xffffu) * 0x83DCD12Du;
  x1 = (x1 & 0xffffu) * 0x83DCD12Du;
  const uint32_t sum0 = __builtin_amdgcn_udot4(x0, 0x01010101u, 0x6400u, false);
  const uint32_t sum1 = __builtin_amdgcn_udot4(x1, 0x01010101u, 0x6400u, false);
  const half2 k_inv_h2 = __half2half2(__ushort_as_half(0x1eee));
  const half2 k_bias_h2 = __half2half2(__ushort_as_half(0xc931));
  union {
    uint16_t u16;
    half h;
  } h0{static_cast<uint16_t>(sum0)}, h1{static_cast<uint16_t>(sum1)};
  return __hfma2(__halves2half2(h0.h, h1.h), k_inv_h2, k_bias_h2);
#endif
}

// Exl3TileRowMajorIndex (quantize.py:28-42). `t / 8` is the tensor-core lane,
// `t % 8` its eight fragment slots. The host's kRowOffset[8] = {0,1,8,9,0,1,8,9}
// is written as arithmetic here so no constant array lands in scratch.
__device__ inline int Exl3TileRowMajorIndex(int t) {
  const int lane = t >> 3;
  const int sub = t & 7;
  const int q = sub & 3;
  const int row_offset = q < 2 ? q : q + 6;  // 0,1,8,9
  const int r = (lane % 4) * 2 + row_offset;
  const int c = lane / 4 + (sub < 4 ? 0 : 8);
  return r * 16 + c;
}

// The INVERSE of Exl3TileRowMajorIndex: the codeword index `t` whose decoded
// weight lands at row-major position (r, c) inside the 16x16 tile. The
// reconstruct kernel enumerates output elements, not codewords, so each
// codeword is decoded exactly once and writes stay coalesced. Derived from the
// map's own arithmetic: row_offset is {0,1} for r < 8 and {8,9} for r >= 8,
// lane%4 is r/2 mod 4 in both halves.
__device__ inline int Exl3TileRowMajorIndexInv(int r, int c) {
  const int l4 = (r < 8) ? (r >> 1) : ((r - 8) >> 1);
  const int ro = r - 2 * l4;                 // 0,1 or 8,9
  const int q = (ro < 2) ? ro : (ro - 6);    // undo the q -> q+6 shift
  const int lane = (c & 7) * 4 + l4;
  const int sub = q + ((c & 8) ? 4 : 0);
  return lane * 8 + sub;
}

}  // namespace vt::rocm
