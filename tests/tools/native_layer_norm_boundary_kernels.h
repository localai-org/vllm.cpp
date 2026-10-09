// Diagnostic-only N/N1 variants. No production dispatch or golden injection.
// Native sequence copied from src/vt/xpu/xpu_vision_ops.cpp at 8f0cdcb72.
// Ordinary LayerNorm adapted from intel/torch-xpu-ops
// bc294243807debb350dca694049ba7a806dfdceb. Copyright 2020-2026 Intel
// Corporation, Apache-2.0; see third_party/torch_xpu_layer_norm/LICENSE.
#pragma once
#include "vt/xpu/xpu_common.h"
#include "vt/xpu/xpu_kernels.h"
#include <sycl/ext/intel/math.hpp>

namespace vt::xpu::norm_probe {
namespace {
struct Moments { float mean = 0, m2 = 0, count = 0; };

Moments AddSample(Moments a, float value, bool vectorized) {
  const float count = a.count + 1.0f;
  const float delta = value - a.mean;
  const float mean = vectorized ? sycl::fma(delta, sycl::native::recip(count), a.mean)
                                : a.mean + delta / count;
  return {mean, sycl::fma(delta, value - mean, a.m2), count};
}

Moments Merge(Moments a, Moments b, bool vectorized) {
  if (!vectorized) {
    if (a.count == 0) return b;
    if (b.count == 0) return a;
    const float delta = b.mean - a.mean;
    const float count = a.count + b.count;
    const float fraction = b.count / count;
    return {sycl::fma(delta, fraction, a.mean),
            sycl::fma(delta * delta * a.count, fraction, a.m2 + b.m2), count};
  }
  // The vectorized reference combines current=B, shuffled/shared=A.
  const float count = a.count + b.count;
  if (count == 0) return {};
  const float inverse = sycl::native::recip(count);
  const float na = b.count * inverse, nb = a.count * inverse;
  const float delta = a.mean - b.mean;
  return {sycl::fma(na, b.mean, nb * a.mean),
          sycl::fma(delta * delta * b.count, nb, b.m2 + a.m2), count};
}
Moments Shuffle(sycl::sub_group group, Moments value, int offset) {
  return {sycl::shift_group_left(group, value.mean, offset),
          sycl::shift_group_left(group, value.m2, offset),
          sycl::shift_group_left(group, value.count, offset)};
}
}  // namespace

template<bool materialize, bool observe, bool correct_division = false>
void Native(Queue& q, Tensor& out, const Tensor& x,
                     const Tensor* weight, const Tensor* bias,
                     const LayerNormArgs& args, float* statistics, float* pre_cast) {
  TraceXpuOp(OpId::kLayerNorm, q, {&out, &x, weight, bias});
  FloatTensor(out); FloatTensor(x);
  if (weight) FloatTensor(*weight);
  if (bias) FloatTensor(*bias);
  const int64_t width = x.shape[x.rank - 1];
  const int64_t rows = x.Numel() / width;
  if (!rows) return;
  constexpr int simd = 32;
  const bool vectorized = width % 4 == 0 && width <= (1 << 24);
  int lanes = 512;  // pinned reference tail reduction group size
  if (vectorized) {
    lanes = NativeQueue(q).get_device().get_info<sycl::info::device::max_work_group_size>();
    while (lanes > width / 4 && lanes > simd) lanes /= 2;
  }
  const int groups = lanes / simd;
  const bool affine = weight != nullptr, has_bias = bias != nullptr;
  const float eps = args.eps;
  WithOutput(q, out, {&x, weight, bias}, [&](Tensor& target) {
    const View dst(target), src(x);
    const View gamma(weight ? *weight : x), beta(bias ? *bias : x);
    const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
      sycl::local_accessor<Moments, 1> partial(sycl::range<1>(groups), h);
      sycl::local_accessor<float, 1> published;
      if constexpr (materialize)
        published = sycl::local_accessor<float, 1>(sycl::range<1>(2), h);
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>(rows * lanes), sycl::range<1>(lanes)),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t base = item.get_group_linear_id() * width;
        const int lane = item.get_local_linear_id();
        const auto sg = item.get_sub_group();
        const int sg_lane = sg.get_local_linear_id(), sg_id = sg.get_group_linear_id();
        Moments moments;
        if (vectorized) {
          for (int64_t c = lane * 4; c < width; c += lanes * 4)
            for (int k = 0; k < 4; ++k) moments = AddSample(moments, Load(src, base + c + k), true);
          for (int offset = simd / 2; offset; offset /= 2)
            moments = Merge(moments, Shuffle(sg, moments, offset), true);
        } else {
          for (int64_t c = lane; c < width; c += lanes)
            moments = AddSample(moments, Load(src, base + c), false);
          for (int offset = 1; offset < simd; offset *= 2) {
            const auto other = Shuffle(sg, moments, offset);
            if (sg_lane < simd - offset) moments = Merge(moments, other, false);
          }
        }
        if (sg_lane == 0) partial[sg_id] = moments;
        item.barrier(sycl::access::fence_space::local_space);
        if (vectorized) {
          for (int offset = groups / 2; offset; offset /= 2) {
            if (sg_lane == 0 && sg_id < offset)
              moments = Merge(moments, partial[sg_id + offset], true);
            item.barrier(sycl::access::fence_space::local_space);
            if (sg_lane == 0 && sg_id < offset) partial[sg_id] = moments;
            item.barrier(sycl::access::fence_space::local_space);
          }
        } else if (lane == 0) {
          for (int group = 1; group < groups; ++group)
            moments = Merge(moments, partial[group], false);
          partial[0] = moments;
        }
        item.barrier(sycl::access::fence_space::local_space);
        if constexpr (materialize) {
          if (lane == 0) {
            published[0] = partial[0].mean;
            published[1] = partial[0].m2 / static_cast<float>(width);
          }
          item.barrier(sycl::access::fence_space::local_space);
        }
        const float mean = materialize ? published[0] : partial[0].mean;
        // Candidate2: reproduce the bound reference's FP32 rounded divide,
        // with no queue/program-wide precision option and no wider storage.
        const float variance = correct_division
            ? sycl::ext::intel::math::fdiv_rn(partial[0].m2, static_cast<float>(width))
            : (materialize ? published[1] : partial[0].m2 / static_cast<float>(width));
        const float denominator = variance + eps;
        const float inverse = sycl::rsqrt(denominator);
        if constexpr (observe) {
          if (lane == 0) {
            const int64_t row = item.get_group_linear_id();
            statistics[row * 6] = mean;
            statistics[row * 6 + 1] = partial[0].m2;
            statistics[row * 6 + 2] = variance;
            statistics[row * 6 + 3] = denominator;
            statistics[row * 6 + 4] = inverse;
            statistics[row * 6 + 5] = partial[0].count;
          }
        }
        for (int64_t c = lane; c < width; c += lanes) {
          float value = (Load(src, base + c) - mean) * inverse;
          if (affine && has_bias)
            value = sycl::fma(Load(gamma, c), value, Load(beta, c));
          else if (affine) value *= Load(gamma, c);
          else if (has_bias) value += Load(beta, c);
          if constexpr (observe) pre_cast[base + c] = value;
          Store(dst, base + c, value);
        }
      });
    });
    RecordProfileEvent(q, "vision_layer_norm", event);
  });
}

}  // namespace vt::xpu::norm_probe

namespace vt::xpu::norm_probe {
// D: minimal Torch-free vectorized forward adaptation of pinned Intel source.
// Keep its ordinary expressions, ND shape, local normalized variance and vec4
// accesses. Effective flags remain native strict flags: D is not installed R.
struct WelfordDataLN {
  float mean = 0.f, sigma2 = 0.f, count = 0.f;
};
inline WelfordDataLN Online(float val, WelfordDataLN curr) {
  float delta = val - curr.mean;
  float new_count = curr.count + 1.f;
  float new_mean = curr.mean + delta * sycl::native::recip(new_count);
  return {new_mean, curr.sigma2 + delta * (val - new_mean), new_count};
}
inline WelfordDataLN Combine(WelfordDataLN dataB, WelfordDataLN dataA) {
  float delta = dataB.mean - dataA.mean;
  float count = dataA.count + dataB.count;
  if (count <= 0.f) return {};
  auto coef = sycl::native::recip(count);
  auto nA = dataA.count * coef, nB = dataB.count * coef;
  return {nA * dataA.mean + nB * dataB.mean,
          dataA.sigma2 + dataB.sigma2 + delta * delta * dataA.count * nB, count};
}
struct alignas(8) Half4 { sycl::half val[4]; };
template<bool observe>
struct SourceShaped {
  int N;
  float eps;
  const sycl::half* __restrict X;
  const sycl::half* gamma;
  const sycl::half* beta;
  sycl::half* Y;
  float* stats;
  float* pre_cast;
  sycl::local_accessor<float, 1> buf;

  [[sycl::reqd_sub_group_size(32)]]
  void operator()(sycl::nd_item<2> item) const {
    auto row = item.get_group(1);
    auto* X_vec = reinterpret_cast<const Half4*>(X + row * N);
    auto* gamma_vec = reinterpret_cast<const Half4*>(gamma);
    auto* beta_vec = reinterpret_cast<const Half4*>(beta);
    auto* Y_vec = reinterpret_cast<Half4*>(Y + row * N);
    const int numx = item.get_local_range(1) * item.get_local_range(0);
    const int thrx = item.get_local_linear_id();
    WelfordDataLN wd;
    for (int i = thrx; i < N / 4; i += numx) {
      Half4 data = X_vec[i];
#pragma unroll
      for (int ii = 0; ii < 4; ++ii) wd = Online(float(data.val[ii]), wd);
    }
    auto sg = item.get_sub_group();
    for (int offset = 16; offset; offset >>= 1) {
      WelfordDataLN other{sycl::shift_group_left(sg, wd.mean, offset),
                         sycl::shift_group_left(sg, wd.sigma2, offset),
                         sycl::shift_group_left(sg, wd.count, offset)};
      wd = Combine(wd, other);
    }
    float m2 = 0.f, count = 0.f;
    if (item.get_local_range(0) > 1) {
      auto addr_offset = item.get_local_range(0);
      for (int offset = item.get_local_range(0) / 2; offset; offset /= 2) {
        if (item.get_local_id(1) == 0 && item.get_local_id(0) >= size_t(offset) &&
            item.get_local_id(0) < size_t(2 * offset)) {
          const int wrt_y = item.get_local_id(0) - offset;
          buf[2 * wrt_y] = wd.mean;
          buf[2 * wrt_y + 1] = wd.sigma2;
          buf[wrt_y + addr_offset] = wd.count;
        }
        sycl::group_barrier(item.get_group());
        if (item.get_local_id(1) == 0 && item.get_local_id(0) < size_t(offset)) {
          const int rd_y = item.get_local_id(0);
          wd = Combine(wd, {buf[2 * rd_y], buf[2 * rd_y + 1], buf[rd_y + addr_offset]});
        }
        sycl::group_barrier(item.get_group());
      }
      if (thrx == 0) {
        m2 = wd.sigma2; count = wd.count;
        buf[0] = wd.mean;
        buf[1] = wd.sigma2 / float(N);
      }
      sycl::group_barrier(item.get_group());
      wd = {buf[0], buf[1], 0.f};
    } else {
      m2 = sycl::select_from_group(sg, wd.sigma2, 0);
      count = sycl::select_from_group(sg, wd.count, 0);
      wd = {sycl::select_from_group(sg, wd.mean, 0), m2 / float(N), 0.f};
    }
    const float denominator = wd.sigma2 + eps;
    const float rstd = sycl::rsqrt(denominator);
    for (int i = thrx; i < N / 4; i += numx) {
      Half4 data = X_vec[i], out;
#pragma unroll
      for (int ii = 0; ii < 4; ++ii) {
        float value = float(gamma_vec[i].val[ii]) *
            (rstd * (float(data.val[ii]) - wd.mean)) + float(beta_vec[i].val[ii]);
        if constexpr (observe) pre_cast[row * N + i * 4 + ii] = value;
        out.val[ii] = sycl::half(value);
      }
      Y_vec[i] = out;
    }
    if constexpr (observe) {
      if (thrx == 0) {
        stats[row * 6] = wd.mean;
        stats[row * 6 + 1] = m2;
        stats[row * 6 + 2] = wd.sigma2;
        stats[row * 6 + 3] = denominator;
        stats[row * 6 + 4] = rstd;
        stats[row * 6 + 5] = count;
      }
    }
  }
};

template<bool observe>
int Adapted(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
            const Tensor& bias, float eps, float* stats, float* pre_cast) {
  using Kernel = SourceShaped<observe>;
  auto& native = NativeQueue(q);
  auto bundle = sycl::get_kernel_bundle<sycl::bundle_state::executable>(
      native.get_context(), {native.get_device()}, {sycl::get_kernel_id<Kernel>()});
  auto kernel = bundle.get_kernel(sycl::get_kernel_id<Kernel>());
  int maximum = kernel.template get_info<sycl::info::kernel_device_specific::work_group_size>(native.get_device());
  int width = x.shape[x.rank - 1], lanes = maximum;
  while (lanes > width / 4 && lanes > 32) lanes >>= 1;
  VT_CHECK(lanes % 32 == 0 && width % 4 == 0, "D requires aligned vectorized dispatch");
  native.submit([&](sycl::handler& h) {
    sycl::local_accessor<float, 1> buf(sycl::range<1>((lanes / 32) * 2), h);
    Kernel functor{width, eps, static_cast<const sycl::half*>(x.data),
                   static_cast<const sycl::half*>(weight.data),
                   static_cast<const sycl::half*>(bias.data),
                   static_cast<sycl::half*>(out.data), stats, pre_cast, buf};
    h.parallel_for(sycl::nd_range<2>({size_t(lanes / 32), size_t(x.Numel() / width * 32)},
                                   {size_t(lanes / 32), 32}), functor);
  });
  return maximum;
}
}  // namespace vt::xpu::norm_probe
