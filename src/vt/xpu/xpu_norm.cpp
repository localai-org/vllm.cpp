#include "xpu_common.h"
#include "xpu_kernels.h"
#include "xpu_qk_norm.h"
namespace vt::xpu {
namespace {
template <bool Residual>
float Gemma5120Value(View src, View res, int64_t row, int col) {
#pragma clang fp contract(off)
  const float value = Load(src, row * src.stride[0] + col);
  if constexpr (Residual)
    return value + Load(res, row * res.stride[0] + col);
  return value;
}

float Gemma5120OutputValue(float value, float inverse, float weight) {
#pragma clang fp contract(off)
  // Keep each producer F32 boundary with native arithmetic. The XPU target
  // also uses -fno-fast-math/-ffp-contract=off; no fused multiply/add or
  // reassociation is permitted. Avoid external directed-rounding library calls
  // in this finite FP16-operand path.
  const float effective_weight = 1.0f + weight;
  const float normalized = value * inverse;
  return normalized * effective_weight;
}

template <bool Residual>
void Gemma5120ShortRowsKernel(Queue& q, View dst, View src, View w, View res,
                              int64_t rows, int width, float eps) {
  // One work-group per row distributes the original virtual-lane partials.
  // SLM exchanges preserve the same descending chunk tree, then the same
  // ascending SG16 tree for each half of the producer's logical SG32.
  constexpr int lanes = 16;
  const int workgroup = width / 2, chunks = width / 32;
  // The public RMSNorm contract validates contiguous rows, and this route
  // admits F16 operands only. Capture typed pointers instead of dynamic View
  // loads/stores without changing the virtual-lane reduction or F32 sum.
  const auto* input = static_cast<const sycl::half*>(src.data);
  const auto* weight = static_cast<const sycl::half*>(w.data);
  auto* residual = static_cast<sycl::half*>(res.data);
  auto* output = static_cast<sycl::half*>(dst.data);
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> sums(sycl::range<1>(width + 1), h);
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * workgroup),
                                    sycl::range<1>(workgroup)),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
#pragma clang fp contract(off)
      const int64_t row = item.get_group_linear_id();
      const int64_t base = row * 5120;
      const int local = item.get_local_linear_id();
      const int chunk = local / lanes, lane = local % lanes;
      for (int half = 0; half < 2; ++half) {
        float regs[4] = {};
        for (int col = 4 * (32 * chunk + lane + half * lanes); col < 5120;
             col += 4 * width) {
          for (int j = 0; j < 4; ++j) {
            float value = static_cast<float>(input[base + col + j]);
            if constexpr (Residual)
              value += static_cast<float>(residual[base + col + j]);
            regs[j] += value * value;
          }
        }
        sums[half * workgroup + local] = ((regs[0] + regs[1]) + regs[2]) + regs[3];
      }
      item.barrier(sycl::access::fence_space::local_space);
      for (int offset = chunks / 2; offset > 0; offset /= 2) {
        if (chunk < offset) {
          sums[local] += sums[local + offset * lanes];
          sums[workgroup + local] += sums[workgroup + local + offset * lanes];
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      if (chunk == 0) {
        auto group = item.get_sub_group();
        float a = sums[lane], b = sums[workgroup + lane];
        for (int offset = 1; offset < lanes; offset *= 2) {
          a += sycl::shift_group_left(group, a, offset);
          b += sycl::shift_group_left(group, b, offset);
        }
        const float mean = sycl::group_broadcast(group, a + b, 0) * (1.0f / 5120.0f);
        if (lane == 0) sums[width] = sycl::rsqrt(mean + eps);
      }
      item.barrier(sycl::access::fence_space::local_space);
      const float inverse = sums[width];
      for (int col = local; col < 5120; col += workgroup) {
        float value = static_cast<float>(input[base + col]);
        if constexpr (Residual) {
          value += static_cast<float>(residual[base + col]);
          residual[base + col] = static_cast<sycl::half>(value);
        }
        output[base + col] = static_cast<sycl::half>(Gemma5120OutputValue(
            value, inverse, static_cast<float>(weight[col])));
      }
    });
  });
  RecordProfileEvent(q, "rms_norm_gemma5120_fp16", event);
}

template <bool Residual>
void Gemma5120Kernel(Queue& q, View dst, View src, View w, View res,
                     int64_t rows, float eps) {
  // Pinned Torch ReduceConfig: max WG1024, logical SG32, contiguous vec4.
  // Output count determines group_height; group_x_reduce first halves the
  // virtual lanes down to32, then uses ascending subgroup offsets.
  int height = 1;
  while (height < 32 && height * 2 <= rows) height *= 2;
  const int width = 1024 / height;
  if (rows > 0 && rows <= 16) {
    Gemma5120ShortRowsKernel<Residual>(q, dst, src, w, res, rows, width, eps);
    return;
  }
  constexpr int lanes = 16, workgroup = 128;
  const auto global = ((rows * lanes + workgroup - 1) / workgroup) * workgroup;
  const auto event = NativeQueue(q).parallel_for(
      sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
#pragma clang fp contract(off)
    const int64_t row = item.get_global_linear_id() / lanes;
    if (row >= rows) return;
    const int lane = item.get_local_linear_id() % lanes;
    float first[32], second[32];
    const int chunks = width / 32;
    for (int chunk = 0; chunk < chunks; ++chunk) {
      float partial[2];
      for (int half = 0; half < 2; ++half) {
        float regs[4] = {};
        for (int col = 4 * (32 * chunk + lane + half * lanes); col < 5120;
             col += 4 * width) {
          for (int j = 0; j < 4; ++j) {
            const float value = Gemma5120Value<Residual>(src, res, row, col + j);
            regs[j] += value * value;
          }
        }
        partial[half] = ((regs[0] + regs[1]) + regs[2]) + regs[3];
      }
      first[chunk] = partial[0]; second[chunk] = partial[1];
    }
    for (int offset = chunks / 2; offset > 0; offset /= 2)
      for (int chunk = 0; chunk < offset; ++chunk) {
        first[chunk] += first[chunk + offset];
        second[chunk] += second[chunk + offset];
      }
    auto group = item.get_sub_group();
    float a = first[0], b = second[0];
    for (int offset = 1; offset < lanes; offset *= 2) {
      a += sycl::shift_group_left(group, a, offset);
      b += sycl::shift_group_left(group, b, offset);
    }
    // MeanOps projects with an F32 reciprocal factor, not division by D.
    const float mean = sycl::group_broadcast(group, a + b, 0) * (1.0f / 5120.0f);
    const float inverse = sycl::rsqrt(mean + eps);
    for (int col = lane; col < 5120; col += lanes) {
      const float value = Gemma5120Value<Residual>(src, res, row, col);
      if constexpr (Residual) Store(res, row * res.stride[0] + col, value);
      Store(dst, row * dst.stride[0] + col,
            Gemma5120OutputValue(value, inverse, Load(w, col)));
    }
  });
  RecordProfileEvent(q, "rms_norm_gemma5120_fp16", event);
}
}  // namespace

void RmsNormKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                   const RmsNormArgs& args, Tensor* residual) {
  TraceXpuOp(OpId::kRmsNorm, q, {&out, &x, &weight, residual});
  FloatTensor(out); FloatTensor(x); FloatTensor(weight);
  VT_CHECK(x.shape[1] > 0, "XPU RMSNorm requires positive hidden width");
  if (residual) {
    FloatTensor(*residual);
    VT_CHECK(!Overlap(*residual, weight), "XPU RMSNorm residual may not alias weights");
    VT_CHECK(!Overlap(*residual, x) || residual->data == x.data, "XPU RMSNorm partial residual/input alias");
  }
  WithOutput(q, out, {&x, &weight, residual}, [&](Tensor& target) {
    const View dst(target), src(x), w(weight), res(residual ? *residual : x);
    const auto width = x.shape[1]; const bool has_res = residual != nullptr;
    const auto eps = args.eps; const bool gemma = args.gemma;
    if (args.qk_fp16) {
      const int64_t rows = x.shape[0];
      const auto sizes = NativeQueue(q).get_device().get_info<sycl::info::device::sub_group_sizes>();
      if (std::find(sizes.begin(), sizes.end(), 16) != sizes.end()) {
        constexpr int lanes = 16, workgroup = 128;
        const auto global = ((rows * lanes + workgroup - 1) / workgroup) * workgroup;
        const auto event = NativeQueue(q).parallel_for(
            sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
          const int64_t row = item.get_global_linear_id() / lanes;
          if (row >= rows) return;
          const int lane = item.get_local_linear_id() % lanes;
          const int64_t base = row * src.stride[0];
          const float mean = ProducerQkMean256(src, base, item.get_sub_group(), lane, rows);
          const float inverse = sycl::rsqrt(mean + eps);
          for (int col = lane; col < 256; col += lanes)
            Store(dst, row * dst.stride[0] + col,
                  ProducerQkNormValue(Load(src, base + col), inverse, Load(w, col), true));
        });
        RecordProfileEvent(q, "rms_norm_qk_fp16_subgroup", event);
      } else {
        const auto event = NativeQueue(q).parallel_for(sycl::range<1>(rows), [=](sycl::id<1> item) {
          const int64_t row = item[0], base = row * src.stride[0];
          const float mean = ProducerQkMean256Scalar(src, base, rows);
          const float inverse = sycl::rsqrt(mean + eps);
          for (int col = 0; col < 256; ++col)
            Store(dst, row * dst.stride[0] + col,
                  ProducerQkNormValue(Load(src, base + col), inverse, Load(w, col), true));
        });
        RecordProfileEvent(q, "rms_norm_qk_fp16_scalar", event);
      }
      return;
    }
    if (width == 5120 && gemma && src.dtype == DType::kF16 &&
        dst.dtype == DType::kF16 && w.dtype == DType::kF16 &&
        (!has_res || res.dtype == DType::kF16)) {
      const auto device = NativeQueue(q).get_device();
      const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
      if (device.get_info<sycl::info::device::max_work_group_size>() == 1024 &&
          !sizes.empty() && *std::min_element(sizes.begin(), sizes.end()) == 16 &&
          *std::max_element(sizes.begin(), sizes.end()) == 32) {
        if (has_res) Gemma5120Kernel<true>(q, dst, src, w, res, x.shape[0], eps);
        else Gemma5120Kernel<false>(q, dst, src, w, res, x.shape[0], eps);
        return;
      }
    }
    // Pinned EXL3 FP16 GemmaRMSNorm uses the native producer IR:
    // normalize x.float()+res.float(), but return that sum narrowed as res.
    // Do not normalize a reread of the FP16 store. Other dtype/weight modes
    // retain their existing residual-rounding contract.
    const bool fp32_sum = has_res && gemma && src.dtype == DType::kF16 &&
                          dst.dtype == DType::kF16 && res.dtype == DType::kF16;
    // Wide decode rows must distribute the reduction over a work-group. A
    // single work-item reading 5120 elements twice dominates B70 token time.
    if (width >= 256) {
      constexpr size_t kWorkGroup = 256;
      const auto event = NativeQueue(q).parallel_for(
          sycl::nd_range<1>(sycl::range<1>(x.shape[0] * kWorkGroup),
                            sycl::range<1>(kWorkGroup)),
          [=](sycl::nd_item<1> item) {
        const auto row = item.get_group(0);
        const auto lane = item.get_local_id(0);
        float partial = 0.0f;
        for (int64_t col = lane; col < width; col += kWorkGroup) {
          float value = Load(src, row * src.stride[0] + col);
          if (has_res) {
            const auto off = row * res.stride[0] + col;
            value += Load(res, off);
            if (!fp32_sum) {
              value = Round(res.dtype, value);
              Store(res, off, value);
            }
          }
          partial += value * value;
        }
        const float sum = sycl::reduce_over_group(
            item.get_group(), partial, sycl::plus<float>());
        const float scale = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
        sycl::group_barrier(item.get_group());
        for (int64_t col = lane; col < width; col += kWorkGroup) {
          float value = has_res ? Load(res, row * res.stride[0] + col)
                               : Load(src, row * src.stride[0] + col);
          if (fp32_sum) {
            value += Load(src, row * src.stride[0] + col);
            Store(res, row * res.stride[0] + col, value);
          }
          const float weight_value = gemma ? 1.0f + Load(w, col) : Load(w, col);
          Store(dst, row * dst.stride[0] + col, value * scale * weight_value);
        }
      });
      RecordProfileEvent(q, "rms_norm", event);
      return;
    }
    // Keep the sequential reduction for narrow rows and its exact FP32 sum.
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(x.shape[0]), [=](sycl::id<1> item) {
      const auto row = item[0];
      float sum = 0;
      for (int64_t col = 0; col < width; ++col) {
        float value = Load(src, row * src.stride[0] + col);
        if (has_res) {
          const auto off = row * res.stride[0] + col;
          value += Load(res, off);
          if (!fp32_sum) {
            value = Round(res.dtype, value);
            Store(res, off, value);
          }
        }
        sum += value * value;
      }
      const float scale = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
      for (int64_t col = 0; col < width; ++col) {
        float value = has_res ? Load(res, row * res.stride[0] + col)
                             : Load(src, row * src.stride[0] + col);
        if (fp32_sum) {
          value += Load(src, row * src.stride[0] + col);
          Store(res, row * res.stride[0] + col, value);
        }
        const float weight_value = gemma ? 1.0f + Load(w, col) : Load(w, col);
        Store(dst, row * dst.stride[0] + col, value * scale * weight_value);
      }
    });
    RecordProfileEvent(q, "rms_norm", event);
  });
}
}  // namespace vt::xpu
