// Ordinary LayerNorm adapted from intel/torch-xpu-ops
// bc294243807debb350dca694049ba7a806dfdceb, LayerNormKernels.cpp,
// SharedReduceOps.h and GroupReduceUtils.h. Copyright 2020-2026 Intel
// Corporation, Apache-2.0; see third_party/torch_xpu_layer_norm/LICENSE.
// Explicit FP32 Welford moments, affine arithmetic and one final storage cast.
#include "xpu_common.h"
#include "xpu_kernels.h"
#include <sycl/ext/intel/math.hpp>

namespace vt::xpu {
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

void LayerNormKernel(Queue& q, Tensor& out, const Tensor& x,
                     const Tensor* weight, const Tensor* bias,
                     const LayerNormArgs& args) {
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
        const float mean = partial[0].mean;
        // The pinned Torch program uses correctly rounded FP32 division.
        // Default SYCL division permits 2.5 ULP even under our strict flags.
        // Enforce only this normalization boundary; text math and the native
        // reciprocal-based Welford tree retain their existing arithmetic.
        const float variance = sycl::ext::intel::math::fdiv_rn(
            partial[0].m2, static_cast<float>(width));
        const float inverse = sycl::rsqrt(variance + eps);
        for (int64_t c = lane; c < width; c += lanes) {
          float value = (Load(src, base + c) - mean) * inverse;
          if (affine && has_bias)
            value = sycl::fma(Load(gamma, c), value, Load(beta, c));
          else if (affine) value *= Load(gamma, c);
          else if (has_bias) value += Load(beta, c);
          Store(dst, base + c, value);
        }
      });
    });
    RecordProfileEvent(q, "vision_layer_norm", event);
  });
}

namespace {
// ActivationGeluKernel.cpp at the same Intel source pin as LayerNorm.
// Torch's opmath type is FP32 for FP16/BF16; only the output is narrowed.
template<bool tanh_approximation>
void GeluKernel(Queue& q, Tensor& out, const Tensor& x) {
  TraceXpuOp(tanh_approximation ? OpId::kGeluTanh : OpId::kGeluErf, q, {&out, &x});
  FloatTensor(out); FloatTensor(x);
  if (!out.Numel()) return;
  const auto launch = [&](Tensor& target) {
    const View dst(target), src(x);
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const int64_t i = item[0];
      const float value = Load(src, i);
      float result;
      if constexpr (tanh_approximation) {
        constexpr float beta = 0.7978845608028654f;
        const float cube = value * value * value;
        const float inner = beta * sycl::fma(0.044715f, cube, value);
        result = 0.5f * value * (1.0f + ::tanhf(inner));
      } else {
        constexpr float alpha = 0.7071067811865476f;
        result = value * 0.5f * (1.0f + sycl::erf(value * alpha));
      }
      Store(dst, i, result);
    });
    RecordProfileEvent(q, tanh_approximation ? "vision_gelu_tanh" : "vision_gelu_erf", event);
  };
  // Exact in-place pointwise execution cannot clobber another lane's input.
  // Partial overlaps or differently sized storage still require a snapshot.
  if (out.data == x.data && out.dtype == x.dtype) launch(out);
  else WithOutput(q, out, {&x}, launch);
}
}  // namespace

void GeluTanhKernel(Queue& q, Tensor& out, const Tensor& x) { GeluKernel<true>(q, out, x); }
void GeluErfKernel(Queue& q, Tensor& out, const Tensor& x) { GeluKernel<false>(q, out, x); }

// vLLM Qwen3_VisionTransformer's executed Triton interpolation path. FP16
// coefficient casts and contracted half FMA order follow the executed kernel
// and captured TTIR/LLVM IR; coordinate math stays FP32. No host learned math.
void VisionPosEmbedInterpolateKernel(Queue& q, Tensor& out, const Tensor& table,
                                     const VisionPosEmbedArgs& args) {
  TraceXpuOp(OpId::kVisionPosEmbedInterpolate, q, {&out, &table});
  const View dst(out), src(table);
  const int64_t width = out.shape[1], h = args.h, w = args.w;
  const int64_t side = args.grid_side, merge = args.merge_size;
  // Python passes these double-computed ratios as FP32 runtime kernel scalars.
  const float hs = h > 1 ? static_cast<float>(static_cast<double>(side - 1) / (h - 1)) : 0;
  const float ws = w > 1 ? static_cast<float>(static_cast<double>(side - 1) / (w - 1)) : 0;
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
    const int64_t i = item[0], d = i % width;
    const int64_t spatial = (i / width) % (h * w);
    const int64_t block = spatial / (merge * merge), local = spatial % (merge * merge);
    const int64_t row = (block / (w / merge)) * merge + local / merge;
    const int64_t col = (block % (w / merge)) * merge + local % merge;
    const float hf = static_cast<float>(row) * hs, wf = static_cast<float>(col) * ws;
    const int64_t r0 = static_cast<int64_t>(sycl::floor(hf)), c0 = static_cast<int64_t>(sycl::floor(wf));
    const int64_t r1 = sycl::min(r0 + 1, side - 1), c1 = sycl::min(c0 + 1, side - 1);
    const float dh = hf - static_cast<float>(r0), dw = wf - static_cast<float>(c0);
    const float w11 = dh * dw, w10 = dh - w11, w01 = dw - w11;
    const float w00 = 1.0f - dh - w01;
    const sycl::half e00(Load(src, (r0 * side + c0) * width + d));
    const sycl::half e01(Load(src, (r0 * side + c1) * width + d));
    const sycl::half e10(Load(src, (r1 * side + c0) * width + d));
    const sycl::half e11(Load(src, (r1 * side + c1) * width + d));
    // The device compiler contracts p00 into (p00+p01), then p10 and p11
    // into the successive sums. p01 rounds before that first half FMA.
    const sycl::half p01(sycl::half(w01) * e01);
    sycl::half value = sycl::fma(sycl::half(w00), e00, p01);
    value = sycl::fma(sycl::half(w10), e10, value);
    value = sycl::fma(sycl::half(w11), e11, value);
    Store(dst, i, static_cast<float>(value));
  });
  RecordProfileEvent(q, "vision_pos_embed_interpolate", event);
}

void VisionRopeGridKernel(Queue& q, Tensor& out, const Tensor& base,
                          const VisionRopeGridArgs& args) {
  TraceXpuOp(OpId::kVisionRopeGrid, q, {&out, &base});
  auto* dst = static_cast<uint16_t*>(out.data);
  const auto* src = static_cast<const uint16_t*>(base.data);
  const int64_t width = out.shape[1], f = base.shape[1] / 2;
  const int64_t h = args.h, w = args.w, merge = args.merge_size;
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
    const int64_t i = item[0], d = i % width;
    const int64_t spatial = (i / width) % (h * w);
    const int64_t block = spatial / (merge * merge), local = spatial % (merge * merge);
    const int64_t row = (block / (w / merge)) * merge + local / merge;
    const int64_t col = (block % (w / merge)) * merge + local % merge;
    const int64_t axis = (d / f) % 2 == 0 ? row : col;
    const int64_t source_column = (d / (2 * f)) * f + d % f;
    dst[i] = src[axis * (2 * f) + source_column];
  });
  RecordProfileEvent(q, "vision_rope_grid", event);
}

void QkvSplitKernel(Queue& queue, Tensor& q, Tensor& k, Tensor& v, const Tensor& qkv) {
  const int64_t rows = qkv.shape[0];
  const int64_t qd = q.Numel() / rows, kd = k.Numel() / rows, vd = v.Numel() / rows;
  VT_CHECK(qd > 0 && kd > 0 && vd > 0,
           "XPU QKV split requires positive output widths");
  VT_CHECK(!Overlap(q, qkv) && !Overlap(k, qkv) && !Overlap(v, qkv) &&
               !Overlap(q, k) && !Overlap(q, v) && !Overlap(k, v),
           "XPU QKV split requires non-overlapping input/output storage");
  TraceXpuOp(OpId::kQkvSplit, queue, {&q, &k, &v, &qkv});
  const View qs(q), ks(k), vs(v), src(qkv);
  const int64_t width = qd + kd + vd;
  const size_t bytes = SizeOf(qkv.dtype);
  const auto event = NativeQueue(queue).parallel_for(sycl::range<1>(qkv.Numel()), [=](sycl::id<1> item) {
    const int64_t i = item[0], row = i / width, col = i % width;
    const auto dst = col < qd ? qs : col < qd + kd ? ks : vs;
    const int64_t out_index = col < qd ? row * qd + col :
        col < qd + kd ? row * kd + col - qd : row * vd + col - qd - kd;
    // Copy storage bits, including signed zero, subnormals and NaN payloads.
    for (size_t b = 0; b < bytes; ++b)
      static_cast<char*>(dst.data)[out_index * bytes + b] =
          static_cast<const char*>(src.data)[i * bytes + b];
  });
  RecordProfileEvent(queue, "vision_qkv_split", event);
}

}  // namespace vt::xpu
