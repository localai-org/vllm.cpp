#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

// Native device equivalents of cpu_exl3_dequant.cpp. The file format uses
// cyclic 16-bit windows and tensor-core order inside each 16x16 tile, even
// when the execution kernel itself uses no matrix instructions.
namespace vt::xpu::exl3 {
inline float Half(uint16_t bits) { return static_cast<float>(sycl::bit_cast<sycl::half>(bits)); }
inline float RoundHalf(float value) { return static_cast<float>(sycl::half(value)); }
inline float Decode(uint16_t codeword, int codebook) {
  uint32_t x = codeword;
  if (codebook == 2) {
    x *= 0x83DCD12Du;
    const uint32_t sum = (x & 255u) + ((x >> 8) & 255u) +
                         ((x >> 16) & 255u) + (x >> 24);
    // The exact half affine product/sum fits F32; round once as __hfma does.
    return RoundHalf(Half(static_cast<uint16_t>(0x6400u + sum)) * Half(0x1eeeu) + Half(0xc931u));
  }
  if (codebook == 0) x = x * 89226354u + 64248484u;
  else x *= 0xCBAC1FEDu;
  x = (x & 0x8fff8fffu) ^ 0x3b603b60u;
  return RoundHalf(Half(static_cast<uint16_t>(x)) + Half(static_cast<uint16_t>(x >> 16)));
}
inline uint32_t Word(const unsigned char* bytes, int index) {
  bytes += index * 4;
  return uint32_t(bytes[0]) | (uint32_t(bytes[1]) << 8) |
         (uint32_t(bytes[2]) << 16) | (uint32_t(bytes[3]) << 24);
}
inline uint16_t Codeword(const unsigned char* tile, int bits, int t) {
  const int words = bits * 8;
  const int begin = t * bits + bits - 16 + 256 * bits;
  const int end = begin + 16;
  const int first = begin / 32, last = (end - 1) / 32;
  const uint64_t pair = (uint64_t(Word(tile, first % words)) << 32) | Word(tile, last % words);
  return static_cast<uint16_t>(pair >> ((last + 1) * 32 - end));
}
inline int RowMajor(int t) {
  const int lane = t >> 3, sub = t & 7;
  const int row = (lane & 3) * 2 + (sub & 1) + ((sub & 2) * 4);
  const int col = lane / 4 + ((sub & 4) * 2);
  return row * 16 + col;
}
inline int Fragment(int row, int col) {
  return (((col & 7) * 4 + ((row & 7) >> 1)) * 8) +
         (row & 1) + ((row >> 3) * 2) + ((col >> 3) * 4);
}
// Aligned specialization used by the packed kernels and the XMX probe.
template<int Bits>
uint16_t PackedCodeword(const uint32_t* tile, int inner, int col) {
  const int t = Fragment(inner, col);
  const int end = t * Bits + Bits + 256 * Bits;
  const int first = (end - 16) / 32, last = (end - 1) / 32;
  const int shift = (last + 1) * 32 - end;
  const uint32_t hi = tile[first % (8 * Bits)], lo = tile[last % (8 * Bits)];
  return static_cast<uint16_t>(shift == 0 ? lo : (lo >> shift) | (hi << (32 - shift)));
}

}  // namespace vt::xpu::exl3
