#pragma once
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

namespace vt::xpu::detail {
// Register-only producer for the checked long-row W8A8 path. The table already
// contains FP16(SiLU(gate)); multiply in F32 and narrow to FP16 once more before
// any suh multiplication. Invalid lanes become private zeros and invalidate
// the checked row. They must never be published as successful activations.
struct SiluMulFp16Block {
  sycl::ext::intel::esimd::simd<sycl::half, 128> values;
  sycl::ext::intel::esimd::simd_mask<128> finite;
};

ESIMD_INLINE SiluMulFp16Block LoadSiluMulFp16(
    const sycl::half* gate_up, size_t offset, int width,
    const uint16_t* silu_table) {
  using namespace sycl::ext::intel::esimd;
  auto gates = block_load<uint16_t, 128>(
      reinterpret_cast<const uint16_t*>(gate_up) + offset);
  auto ups = block_load<uint16_t, 128>(
      reinterpret_cast<const uint16_t*>(gate_up) + offset + width);
  simd_mask<128> finite = ((gates & 0x7c00u) != 0x7c00u) &
                          ((ups & 0x7c00u) != 0x7c00u);
  gates.merge(uint16_t{0}, !finite);
  ups.merge(uint16_t{0}, !finite);
  simd<uint16_t, 128> activations;
#pragma unroll
  for (int part = 0; part < 4; ++part) {
    const simd<uint16_t, 32> indices = gates.template select<32, 1>(32 * part);
    const simd<uint32_t, 32> offsets = convert<uint32_t>(indices) * 2u;
    activations.template select<32, 1>(32 * part) =
        gather<uint16_t, 32>(silu_table, offsets);
  }
  const simd<sycl::half, 128> rounded_silu = activations.bit_cast_view<sycl::half>();
  const simd<sycl::half, 128> up_values = ups.bit_cast_view<sycl::half>();
  auto product = convert<float>(rounded_silu) * convert<float>(up_values);
  // Match checked preparation's RNE overflow boundary, also under fast-math.
  finite &= ((product.bit_cast_view<uint32_t>() & 0x7fffffffu) < 0x477ff000u);
  product.merge(0.0f, !finite);
  simd<sycl::half, 128> rounded = convert<sycl::half>(product);
  finite &= ((rounded.bit_cast_view<uint16_t>() & 0x7c00u) != 0x7c00u);
  rounded.merge(sycl::half{0.0f}, !finite);
  return {rounded, finite};
}
}  // namespace vt::xpu::detail
