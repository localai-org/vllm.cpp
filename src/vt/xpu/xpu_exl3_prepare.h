// Checked preparation adapted from exl3xpu c59d944 csrc/exl3_ops.sycl.
// Copyright (c) 2026 0xSero. MIT license: third_party/exl3xpu/LICENSE.
#pragma once
#include "xpu_silu_input.h"
#include <exl3xpu/exl3_esimd.h>

namespace vt::xpu::detail {
// Only the checked producer uses this. For full physical rows, each group/row
// stores every 128-wide INT8 block and exactly one FP32 scale, even on invalid
// rows (private finite-zero payload/scale1). There are no alignment holes:
// padded_rows is a multiple256, K a multiple128, and scales have4-byte elements.
// Partial padding deliberately keeps BOTH original initializations.
inline void InitializePreparedRows(Queue& q,
    const ::exl3::HadInQ8Kernel<sycl::half>& input, size_t qbytes,
    bool skip_full_initialization) {
  if (skip_full_initialization && input.M == input.Ms) return;
  auto zero = NativeQueue(q).memset(input.xq,0,qbytes);
  auto scales = NativeQueue(q).parallel_for(sycl::range<1>(size_t(input.S)*input.Ms),
      [=](sycl::id<1> i) {input.sx[i[0]]=1.0f;});
  RecordProfileEvent(q,"exl3_w8a8_initialize_quant",zero);
  RecordProfileEvent(q,"exl3_w8a8_initialize_scales",scales);
}

struct TensorActivationInput {
  ESIMD_INLINE SiluMulFp16Block Load(const ::exl3::HadInQ8Kernel<sycl::half>& input,
                                     int row, int block) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    auto value = block_load<sycl::half, 128>(
        input.x + size_t(row) * input.x_stride + block * 128);
    return {value, (value.bit_cast_view<uint16_t>() & 0x7c00u) != 0x7c00u};
  }
};
struct SiluMulActivationInput {
  const uint16_t* table;
  ESIMD_INLINE SiluMulFp16Block Load(const ::exl3::HadInQ8Kernel<sycl::half>& input,
                                     int row, int block) const SYCL_ESIMD_KERNEL {
    return LoadSiluMulFp16(input.x, size_t(row) * input.x_stride + block * 128,
                          input.Kdim, table);
  }
};
// Empty in ordinary serving; test-only observers instantiate separate kernels.
struct IgnorePreparationCheckpoints {
  ESIMD_INLINE void Record(const ::exl3::HadInQ8Kernel<sycl::half>&, int, int, int,
      const sycl::ext::intel::esimd::simd<sycl::half, 128>&,
      const sycl::ext::intel::esimd::simd<sycl::half, 128>&,
      const sycl::ext::intel::esimd::simd<sycl::half, 128>&) const SYCL_ESIMD_KERNEL {}
};

template<int TPR, int NB, class Producer = TensorActivationInput,
         class Observer = IgnorePreparationCheckpoints>
struct PrepareRows {
  ::exl3::HadInQ8Kernel<sycl::half> input;
  uint32_t* valid;
  Producer producer{};
  Observer observer{};
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    slm_init<TPR * (sizeof(float) + sizeof(uint32_t))>();
    const int t = it.get_local_id(0), row = it.get_group(0);
    const int m = row % input.M, g = row / input.M, blocks = input.Kdim / 128;
    simd<sycl::half, 128 * NB> buf;
    simd<float, 128> mx = 0.0f;
    bool finite = true;
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const int kb = t + j * TPR;
      if (kb < blocks) {
        const auto produced = producer.Load(input, m, kb);
        const auto xi = produced.values;
        auto su = block_load<sycl::half, 128>(input.suh + size_t(g) * input.Kdim + kb * 128);
        bool block_finite = produced.finite.all() &&
                            !((su.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any();
        auto v = convert<float>(xi) * convert<float>(su);
        block_finite &= !((v.template bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
        if (!block_finite) v = 0.0f;
        simd<sycl::half, 128> product = convert<sycl::half>(v);
        block_finite &= !((product.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any();
        v = convert<float>(product);
        ::exl3::fwht128(v);
        v *= ::exl3::kRsqrt128;
        block_finite &= !((v.template bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
        if (!block_finite) v = 0.0f;
        simd<sycl::half, 128> vh = convert<sycl::half>(v);
        block_finite &= !((vh.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any();
        if (!block_finite) vh = sycl::half(0.0f);
        observer.Record(input, m, g, kb, xi, product, vh);
        finite &= block_finite;
        buf.template select<128, 1>(j * 128) = vh;
        mx = max(mx, abs(convert<float>(vh)));
      }
    }
    slm_scalar_store<float>(t * sizeof(float), hmax<float>(mx));
    slm_scalar_store<uint32_t>(TPR * sizeof(float) + t * sizeof(uint32_t), finite ? 1u : 0u);
    barrier();
    const bool row_finite = (slm_block_load<uint32_t, TPR>(TPR * sizeof(float)) == 1u).all();
    const float amax = hmax<float>(slm_block_load<float, TPR>(0));
    const float scale = row_finite && amax > 0.0f ? amax / 127.0f : 1.0f;
    const float inv = row_finite ? 1.0f / scale : 0.0f;
    if (t == 0) {
      input.sx[size_t(g) * input.Ms + m] = scale;
      if (!row_finite)
        atomic_update<atomic_op::store, uint32_t, 1>(valid, simd<uint32_t, 1>(0u),
                                                    simd<uint32_t, 1>(0u));
    }
#pragma unroll
    for (int j = 0; j < NB; ++j) {
      const int kb = t + j * TPR;
      if (kb < blocks) {
        const simd<sycl::half, 128> b = buf.template select<128, 1>(j * 128);
        const auto rounded = rnde<float>(convert<float>(b) * inv);
        block_store<int8_t, 128>(input.xq + (size_t(g) * input.Ms + m) * input.Kdim + kb * 128,
                                 convert<int8_t>(rounded));
      }
    }
  }
};

}  // namespace vt::xpu::detail
