#include "xpu_common.h"
#include "xpu_kernels.h"
#include <sycl/ext/intel/math.hpp>
#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <string_view>

namespace vt::xpu {
namespace {
void CheckOffsets(Queue& q, const Tensor& qsl, int64_t sequences, int64_t tokens) {
  const auto* offsets = static_cast<const int32_t*>(qsl.data);
  CheckDeviceMetadata(q, [=] {
    if (offsets[0] != 0 || offsets[sequences] != tokens) return false;
    for (int64_t i = 0; i < sequences; ++i)
      if (offsets[i] < 0 || offsets[i + 1] < offsets[i] || offsets[i + 1] > tokens) return false;
    return true;
  }, "XPU GDN/conv invalid sequence offsets", {&qsl});
}
void CheckSlots(Queue& q, const Tensor& indices, int64_t slots, bool allow_null, bool unique) {
  const auto* ids = static_cast<const int32_t*>(indices.data);
  const auto rows = indices.Numel();
  CheckDeviceMetadata(q, [=] {
    for (int64_t i = 0; i < rows; ++i) {
      if (ids[i] >= slots || (!allow_null && ids[i] < 0)) return false;
      if (unique && ids[i] >= 0)
        for (int64_t j = 0; j < i; ++j) if (ids[j] == ids[i]) return false;
    }
    return true;
  }, "XPU GDN/conv invalid or duplicate state slot", {&indices});
}
inline bool Initial(View flags, int64_t row) {
  return flags.dtype == DType::kI8 ? static_cast<const int8_t*>(flags.data)[row] != 0
                                 : static_cast<const int32_t*>(flags.data)[row] != 0;
}
inline float Silu(float x) { return x / (1.0f + sycl::exp(-x)); }
}

void CausalConv1dFwdKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                           const Tensor* bias, Tensor& state, const Tensor& qsl,
                           const Tensor& initial, const CausalConv1dArgs& args) {
  TraceXpuOp(OpId::kCausalConv1dFwd, q, {&out, &x, &weight, bias, &state, &qsl, &initial});
  const auto sequences = state.shape[0], channels = x.shape[1], taps = weight.shape[1];
  const auto state_width = state.shape[2], width = taps - 1;
  FloatTensor(state);
  VT_CHECK(!Overlap(state, x) && !Overlap(state, out) && !Overlap(state, weight)
               && (!bias || !Overlap(state, *bias)), "XPU conv state must have separate storage");
  CheckOffsets(q, qsl, sequences, x.shape[0]);
  WithOutput(q, out, {&x, &weight, bias}, [&](Tensor& target) {
    const View src(x), dst(target), w(weight), b(bias ? *bias : weight), flags(initial);
    const auto* offsets = static_cast<const int32_t*>(qsl.data);
    const View cache(state);
    const bool has_bias = bias != nullptr, activation = args.silu_activation;
    static const bool parallel_prefill = [] {
      const char* value = std::getenv("VT_XPU_CONV_PREFILL_PARALLEL");
      return !value || std::string_view(value) != "0";
    }();
    // Long single-sequence prefills have independent output tokens. Keep the
    // history write in a later launch so early outputs still see the old state.
    if (parallel_prefill && sequences == 1 && x.shape[0] >= std::max<int64_t>(512, width)) {
      const auto tokens = x.shape[0];
      const auto output_event = NativeQueue(q).parallel_for(sycl::range<1>(tokens * channels), [=](sycl::id<1> item) {
        const int64_t t = item[0] / channels, channel = item[0] % channels;
        const bool use_history = Initial(flags, 0);
        float acc = has_bias ? Load(b, channel) : 0.0f;
        for (int64_t j = 0; j < taps; ++j) {
          const auto token = t - width + j;
          const float value = token >= 0 ? Load(src, token * src.stride[0] + channel)
                                        : use_history ? Load(cache, channel * state_width + width + token) : 0.0f;
          acc += Load(w, channel * taps + j) * value;
        }
        Store(dst, t * channels + channel, activation ? Silu(acc) : acc);
      });
      RecordProfileEvent(q, "conv1d_prefill", output_event);
      if (width) {
        const auto state_event = NativeQueue(q).parallel_for(sycl::range<1>(channels * width), [=](sycl::id<1> item) {
          const int64_t channel = item[0] / width, j = item[0] % width;
          Store(cache, channel * state_width + j,
                Load(src, (tokens - width + j) * src.stride[0] + channel));
        });
        RecordProfileEvent(q, "conv1d_prefill_state", state_event);
      }
      return;
    }
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(sequences * channels), [=](sycl::id<1> item) {
      const int64_t seq = item[0] / channels, channel = item[0] % channels;
      const int64_t begin = offsets[seq], length = offsets[seq + 1] - begin;
      const auto old = (seq * channels + channel) * state_width;
      const bool keep = Initial(flags, seq);
      for (int64_t t = 0; t < length; ++t) {
        float acc = has_bias ? Load(b, channel) : 0.0f;
        for (int64_t j = 0; j < taps; ++j) {
          const auto token = t - width + j;
          const float value = token >= 0 ? Load(src, (begin + token) * src.stride[0] + channel)
                                        : keep ? Load(cache, old + width + token) : 0.0f;
          acc += Load(w, channel * taps + j) * value;
        }
        Store(dst, (begin + t) * channels + channel, activation ? Silu(acc) : acc);
      }
      // For a short sequence, old entries are read above the write cursor.
      // History always contains raw activations, never the convolved output.
      for (int64_t j = 0; j < width; ++j) {
        const auto token = length - width + j;
        const float value = token >= 0 ? Load(src, (begin + token) * src.stride[0] + channel)
                                      : keep ? Load(cache, old + width + token) : 0.0f;
        Store(cache, old + j, value);
      }
    });
    RecordProfileEvent(q, "conv1d_prefill", event);
  });
}

void CausalConv1dUpdateKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& weight,
                              const Tensor* bias, Tensor& state, const Tensor* indices,
                              const CausalConv1dArgs& args) {
  TraceXpuOp(OpId::kCausalConv1dUpdate, q, {&out, &x, &weight, bias, &state, indices});
  const auto batch = x.shape[0], channels = x.shape[1], taps = weight.shape[1];
  const auto state_width = state.shape[2], width = taps - 1;
  FloatTensor(state);
  VT_CHECK(!Overlap(state, x) && !Overlap(state, out) && !Overlap(state, weight)
               && (!bias || !Overlap(state, *bias)), "XPU conv state must have separate storage");
  if (indices) CheckSlots(q, *indices, state.shape[0], true, true);
  WithOutput(q, out, {&x, &weight, bias, indices}, [&](Tensor& target) {
    const View src(x), dst(target), w(weight), b(bias ? *bias : weight);
    const View cache(state);
    const auto* ids = indices ? static_cast<const int32_t*>(indices->data) : nullptr;
    const bool has_bias = bias != nullptr, activation = args.silu_activation;
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(batch * channels), [=](sycl::id<1> item) {
      const int64_t row = item[0] / channels, channel = item[0] % channels;
      const int64_t slot = ids ? ids[row] : row;
      if (slot < 0) return;  // VT conv contract leaves null-slot output unchanged.
      const auto history = (slot * channels + channel) * state_width;
      const float current = Load(src, row * src.stride[0] + channel);
      float acc = has_bias ? Load(b, channel) : 0.0f;
      for (int64_t j = 0; j < width; ++j) acc += Load(w, channel * taps + j) * Load(cache, history + j);
      acc += Load(w, channel * taps + width) * current;
      Store(dst, row * channels + channel, activation ? Silu(acc) : acc);
      for (int64_t j = 0; j + 1 < width; ++j) Store(cache, history + j, Load(cache, history + j + 1));
      if (width) Store(cache, history + width - 1, current);
    });
    RecordProfileEvent(q, "conv1d_decode", event);
  }, true);
}
void CausalConv1dSpecUpdateKernel(Queue& q, Tensor& out, const Tensor& x,
                                  const Tensor& weight, const Tensor* bias,
                                  Tensor& state, const Tensor& indices,
                                  const Tensor& accepted, const Tensor& qsl,
                                  const CausalConv1dArgs& args) {
  TraceXpuOp(OpId::kCausalConv1dSpecUpdate, q,
             {&out, &x, &weight, bias, &state, &indices, &accepted, &qsl});
  const int64_t requests = qsl.shape[0] - 1, channels = x.shape[1];
  const int64_t taps = weight.shape[1], width = taps - 1;
  const int64_t state_len = state.shape[2];
  const int64_t max_query_len = state_len - width + 1;
  FloatTensor(state);
  VT_CHECK(!Overlap(state, x) && !Overlap(state, out) && !Overlap(state, weight) &&
               (!bias || !Overlap(state, *bias)) && !Overlap(state, indices) &&
               !Overlap(state, accepted) && !Overlap(state, qsl),
           "XPU spec conv state must have separate storage");
  const auto* offsets = static_cast<const int32_t*>(qsl.data);
  const auto* ids = static_cast<const int32_t*>(indices.data);
  const auto* num_accepted = static_cast<const int32_t*>(accepted.data);
  const int64_t tokens = x.shape[0], slots = state.shape[0];
  CheckDeviceMetadata(q, [=] {
    if (offsets[0] != 0 || offsets[requests] != tokens) return false;
    for (int64_t i = 0; i < requests; ++i) {
      if (offsets[i] < 0 || offsets[i + 1] < offsets[i] ||
          offsets[i + 1] > tokens ||
          offsets[i + 1] - offsets[i] > max_query_len ||
          num_accepted[i] < 1 || num_accepted[i] > max_query_len ||
          ids[i] >= slots) return false;
      if (ids[i] >= 0)
        for (int64_t j = 0; j < i; ++j)
          if (ids[j] == ids[i]) return false;
    }
    return true;
  }, "XPU spec conv invalid offsets, accepted count or state slot",
      {&qsl, &indices, &accepted});
  WithOutput(q, out, {&x, &weight, bias, &indices, &accepted, &qsl},
             [&](Tensor& target) {
    const View src(x), dst(target), w(weight), b(bias ? *bias : weight);
    const View cache(state);
    const bool has_bias = bias != nullptr, activation = args.silu_activation;
    const auto event = NativeQueue(q).parallel_for(
        sycl::range<1>(requests * channels), [=](sycl::id<1> item) {
      const int64_t request = item[0] / channels, channel = item[0] % channels;
      const int64_t slot = ids[request];
      if (slot < 0) return;
      const int64_t begin = offsets[request], length = offsets[request + 1] - begin;
      if (!length) return;
      const int64_t off = static_cast<int64_t>(num_accepted[request]) - 1;
      const int64_t base = (slot * channels + channel) * state_len;
      for (int64_t t = 0; t < length; ++t) {
        float acc = has_bias ? Load(b, channel) : 0.0f;
        for (int64_t j = 0; j < taps; ++j) {
          const int64_t pos = t + j;
          const float value = pos < width
              ? Load(cache, base + off + pos)
              : Load(src, (begin + pos - width) * src.stride[0] + channel);
          acc += Load(w, channel * taps + j) * value;
        }
        Store(dst, (begin + t) * channels + channel,
              activation ? Silu(acc) : acc);
      }
      // Pinned causal_conv1d.py:845,1221 uses an EFFECTIVE state width of
      // (taps-1)+(length-1), independent of physical speculative capacity.
      // Retain width-1 valid history elements, then this step's provisional
      // tokens. Using physical state_len-length puts short queries after stale
      // rejected tokens. The spare tail stays untouched and inaccessible.
      const int64_t keep = width - 1;
      for (int64_t j = 0; j < keep; ++j) {
        const int64_t source = off + j + 1;
        Store(cache, base + j,
              source < state_len ? Load(cache, base + source) : 0.0f);
      }
      for (int64_t t = 0; t < length; ++t)
        Store(cache, base + keep + t,
              Load(src, (begin + t) * src.stride[0] + channel));
    });
    RecordProfileEvent(q, "conv1d_spec_decode", event);
  }, true);
}
void GdnPostConvKernel(Queue& q, Tensor& qo, Tensor& ko, Tensor& vo, Tensor& go, Tensor& bo,
                        const Tensor& conv, const Tensor& araw, const Tensor& braw,
                        const Tensor& alog, const Tensor& bias, const GdnPostConvArgs& args) {
  TraceXpuOp(OpId::kGdnPostConv, q, {&qo, &ko, &vo, &go, &bo, &conv, &araw, &braw, &alog, &bias});
  const int64_t tokens = conv.shape[0], hk = qo.shape[1], dk = qo.shape[2];
  const int64_t hv = vo.shape[1], dv = vo.shape[2], keys = hk * dk, values = hv * dv;
  const View src(conv), qs(qo), ks(ko), vs(vo), gs(go), bs(bo), a(araw), b(braw), al(alog), dt(bias);
  const float eps = args.eps;
  if (args.xpu_fp16_prefill) {
    // The pinned _xpu_C Conv binary uses SIMD32 (its launch lambda does not
    // inherit the source functor's SIMD16 annotation). Each Q/K head occupies
    // one subgroup with four contiguous FP32 features per lane. The separate
    // libgdn_attn recurrence kernels do use SIMD16; do not conflate the two.
    const auto event = NativeQueue(q).submit([&](sycl::handler& handler) {
      handler.parallel_for(sycl::nd_range<1>(tokens * (hk + hv) * 64, 64),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
        const int64_t group = item.get_group(0);
        const int64_t token = group / (hk + hv), head = group % (hk + hv);
        const int lane = item.get_local_id(0);
        const int64_t base = token * (2 * keys + values);
        if (head < hk) {
          const bool is_q = lane < 32;
          const int feature = (lane % 32) * 4;
          float value[4];
          for (int i = 0; i < 4; ++i) {
            value[i] = Load(src, base + (is_q ? 0 : keys) + head * dk + feature + i);
          }
          // Both pinned Q/K quadratsums square feature 1 before contracting
          // feature 0, then 2 and 3. The independent FP32 producer replay
          // distinguishes this from starting with feature 0.
          float square = value[1] * value[1];
          square = sycl::fma(value[0], value[0], square);
          square = sycl::fma(value[2], value[2], square);
          square = sycl::fma(value[3], value[3], square);
          const auto sg = item.get_sub_group();
          const float total = sycl::reduce_over_group(sg, square, sycl::plus<float>());
          // Match the pinned Conv binary, not just its source expression:
          // Q uses SQRT(sum+eps), SQRT(D), MUL, then INV; K uses RSQRT.
          // The producer executes these as subgroup-uniform scalar math.
          // On B70, varying RSQRT differs even for identical input bits.
          // Compute on the leader and broadcast to preserve its rounding.
          float inv = 0;
          if (sg.get_local_linear_id() == 0)
            inv = is_q
                ? sycl::native::recip(sycl::native::sqrt(total + eps) *
                                      sycl::native::sqrt(float(dk)))
                : sycl::rsqrt(total + eps);
          inv = sycl::group_broadcast(sg, inv, 0);
          for (int i = 0; i < 4; ++i)
            Store(is_q ? qs : ks, (token * hk + head) * dk + feature + i, value[i] * inv);
        } else {
          const int64_t h = head - hk;
          for (int i = lane; i < dv; i += 64)
            Store(vs, (token * hv + h) * dv + i, Load(src, base + 2 * keys + h * dv + i));
          if (lane == 0) {
            const float x = Load(a, token * a.stride[0] + h) + Load(dt, h);
            const float softplus = x < 20.0f ? sycl::log(1.0f + sycl::exp(x)) : x;
            Store(gs, token * hv + h, softplus * -sycl::exp(Load(al, h)));
            Store(bs, token * hv + h, 1.0f / (1.0f + sycl::exp(-Load(b, token * b.stride[0] + h))));
          }
        }
      });
    });
    RecordProfileEvent(q, "gdn_postconv_fp16_prefill", event);
    return;
  }
  const char* subgroup_setting = std::getenv("VT_XPU_GDN_POSTCONV_SUBGROUP");
  const auto device = NativeQueue(q).get_device();
  const bool subgroup = (!subgroup_setting ||
      std::string_view(subgroup_setting) == "1") &&
      conv.dtype == DType::kF16 && qo.dtype == DType::kF16 &&
      ko.dtype == DType::kF16 && vo.dtype == DType::kF16 &&
      hk == 16 && hv == 48 && dk == 128 && dv == 128 &&
      device.has(sycl::aspect::ext_intel_device_id) &&
      device.get_info<sycl::ext::intel::info::device::device_id>() == 57891;
  if (subgroup) {
    constexpr int SG = 16;
    const auto event = NativeQueue(q).parallel_for(
        sycl::nd_range<1>(tokens * (hk + hv) * SG, SG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const int64_t group = item.get_group(0);
      const int64_t token = group / (hk + hv), head = group % (hk + hv);
      const int lane = item.get_local_id(0);
      const int64_t base = token * (2 * keys + values);
      if (head < hk) {
        float qss = 0, kss = 0;
        for (int64_t i = lane; i < dk; i += SG) {
          const float qv = Load(src, base + head * dk + i);
          const float kv = Load(src, base + keys + head * dk + i);
          qss += qv * qv; kss += kv * kv;
        }
        const auto sg = item.get_sub_group();
        qss = sycl::reduce_over_group(sg, qss, sycl::plus<float>());
        kss = sycl::reduce_over_group(sg, kss, sycl::plus<float>());
        const float qi = 1.0f / sycl::sqrt(qss + eps);
        const float ki = 1.0f / sycl::sqrt(kss + eps);
        for (int64_t i = lane; i < dk; i += SG) {
          Store(qs, (token * hk + head) * dk + i,
                Load(src, base + head * dk + i) * qi);
          Store(ks, (token * hk + head) * dk + i,
                Load(src, base + keys + head * dk + i) * ki);
        }
      } else {
        const int64_t h = head - hk;
        for (int64_t i = lane; i < dv; i += SG)
          Store(vs, (token * hv + h) * dv + i,
                Load(src, base + 2 * keys + h * dv + i));
        if (lane == 0) {
          const float x = Load(a, token * a.stride[0] + h) + Load(dt, h);
          const float softplus = x > 20.0f ? x : sycl::log1p(sycl::exp(x));
          Store(gs, token * hv + h, -sycl::exp(Load(al, h)) * softplus);
          Store(bs, token * hv + h,
                1.0f / (1.0f + sycl::exp(-Load(b, token * b.stride[0] + h))));
        }
      }
    });
    RecordProfileEvent(q, "gdn_postconv_subgroup", event);
    return;
  }
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(tokens * (hk + hv)), [=](sycl::id<1> item) {
    const int64_t token = item[0] / (hk + hv), head = item[0] % (hk + hv);
    const int64_t base = token * (2 * keys + values);
    if (head < hk) {
      float qss = 0, kss = 0;
      for (int64_t i = 0; i < dk; ++i) {
        const float qv = Load(src, base + head * dk + i), kv = Load(src, base + keys + head * dk + i);
        qss += qv * qv; kss += kv * kv;
      }
      const float qi = 1.0f / sycl::sqrt(qss + eps), ki = 1.0f / sycl::sqrt(kss + eps);
      for (int64_t i = 0; i < dk; ++i) {
        Store(qs, (token * hk + head) * dk + i, Load(src, base + head * dk + i) * qi);
        Store(ks, (token * hk + head) * dk + i, Load(src, base + keys + head * dk + i) * ki);
      }
    } else {
      const auto h = head - hk;
      for (int64_t i = 0; i < dv; ++i)
        Store(vs, (token * hv + h) * dv + i, Load(src, base + 2 * keys + h * dv + i));
      const float x = Load(a, token * a.stride[0] + h) + Load(dt, h);
      const float softplus = x > 20.0f ? x : sycl::log1p(sycl::exp(x));
      Store(gs, token * hv + h, -sycl::exp(Load(al, h)) * softplus);
      Store(bs, token * hv + h, 1.0f / (1.0f + sycl::exp(-Load(b, token * b.stride[0] + h))));
    }
  });
  RecordProfileEvent(q, "gdn_postconv", event);
}

namespace {
void Recurrence(Queue& q, Tensor& out, const Tensor& qi, const Tensor& ki, const Tensor& vi,
                  const Tensor& g, const Tensor& beta, Tensor& state, const Tensor* qsl,
                  const Tensor* indices, float scale) {
  VT_CHECK(state.dtype == DType::kF32, "XPU GDN recurrence requires F32 state");
  for (const auto* t : std::initializer_list<const Tensor*>{&out, &qi, &ki, &vi, &g, &beta})
    VT_CHECK(!Overlap(state, *t), "XPU GDN state must have separate storage");
  const bool decode = qsl == nullptr;
  const auto rows = decode ? qi.shape[0] : state.shape[0];
  const auto hv = state.shape[1], dv = state.shape[2], dk = state.shape[3], hk = qi.shape[1];
  if (qsl) CheckOffsets(q, *qsl, rows, qi.shape[0]);
  if (indices) CheckSlots(q, *indices, state.shape[0], true, true);
  WithOutput(q, out, {&qi, &ki, &vi, &g, &beta}, [&](Tensor& target) {
    const View dst(target), qs(qi), ks(ki), vs(vi), gs(g), bs(beta);
    const auto* offsets = qsl ? static_cast<const int32_t*>(qsl->data) : nullptr;
    const auto* ids = indices ? static_cast<const int32_t*>(indices->data) : nullptr;
    auto* cache = static_cast<float*>(state.data);
    enum class DecodeMode { kAuto, kReference, kSubgroup };
    static const DecodeMode decode_mode = [] {
      const char* value = std::getenv("VT_XPU_GDN_DECODE");
      const std::string_view name = value ? value : "auto";
      VT_CHECK(name == "auto" || name == "reference" || name == "subgroup", "Invalid VT_XPU_GDN_DECODE");
      return name == "reference" ? DecodeMode::kReference :
             name == "subgroup" ? DecodeMode::kSubgroup : DecodeMode::kAuto;
    }();
    const bool shape_ok = decode && dk == 128 && dv == 128 &&
        qi.dtype == DType::kF16 && ki.dtype == DType::kF16 && vi.dtype == DType::kF16;
    bool subgroup = false;
    if (shape_ok && decode_mode != DecodeMode::kReference) {
      const auto sizes = NativeQueue(q).get_device().get_info<sycl::info::device::sub_group_sizes>();
      subgroup = std::find(sizes.begin(), sizes.end(), 16) != sizes.end();
    }
    VT_CHECK(!decode || decode_mode != DecodeMode::kSubgroup || subgroup,
             "VT_XPU_GDN_DECODE=subgroup requires F16 Dk=Dv=128 and SG16");
    if (subgroup) {
      // One subgroup owns a complete value row; each lane retains eight K
      // positions until both reductions finish, then writes state once.
      constexpr int lanes = 16, tile = 8, workgroup = 128;
      const auto count = rows * hv * dv;
      const auto global = ((count * lanes + workgroup - 1) / workgroup) * workgroup;
      const auto event = NativeQueue(q).parallel_for(
          sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t index = item.get_global_linear_id() / lanes;
        if (index >= count) return;
        const int lane = item.get_local_linear_id() % lanes;
        const int64_t row = index / (hv * dv), head = (index / dv) % hv, value = index % dv;
        const int64_t slot = ids ? ids[row] : row;
        if (slot < 0) {
          if (lane == 0) Store(dst, index, 0.0f);
          return;
        }
        const int64_t token = row, kh = head / (hv / hk);
        const int64_t kbase = (token * hk + kh) * dk;
        auto* s = cache + ((slot * hv + head) * dv + value) * dk;
        const float decay = sycl::exp(Load(gs, token * hv + head));
        float local_state[tile], local_key[tile], prediction = 0.0f;
        for (int j = 0; j < tile; ++j) {
          const int col = lane + lanes * j;
          local_state[j] = s[col] * decay;
          local_key[j] = Load(ks, kbase + col);
          prediction += local_state[j] * local_key[j];
        }
        prediction = sycl::reduce_over_group(item.get_sub_group(), prediction, sycl::plus<float>());
        const float delta = (Load(vs, index) - prediction) * Load(bs, token * hv + head);
        float output = 0.0f;
        for (int j = 0; j < tile; ++j) {
          const int col = lane + lanes * j;
          local_state[j] += delta * local_key[j];
          output += local_state[j] * (Load(qs, kbase + col) * scale);
          s[col] = local_state[j];
        }
        output = sycl::reduce_over_group(item.get_sub_group(), output, sycl::plus<float>());
        if (lane == 0) Store(dst, index, output);
      });
      RecordProfileEvent(q, "gdn_decode_recurrence_subgroup", event);
      return;
    }
    // One work-item owns one value row of S. Tokens remain sequential; no
    // extra q/k normalization or gate transformation occurs inside recurrence.
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(rows * hv * dv), [=](sycl::id<1> item) {
      const int64_t row = item[0] / (hv * dv), head = (item[0] / dv) % hv, value = item[0] % dv;
      const int64_t kh = head / (hv / hk), slot = ids ? ids[row] : row;
      if (slot < 0) { Store(dst, (row * hv + head) * dv + value, 0.0f); return; }
      auto* s = cache + ((slot * hv + head) * dv + value) * dk;
      const int64_t first = decode ? row : offsets[row], end = decode ? row + 1 : offsets[row + 1];
      for (int64_t token = first; token < end; ++token) {
        const int64_t kbase = (token * hk + kh) * dk;
        const float decay = sycl::exp(Load(gs, token * hv + head));
        float prediction = 0;
        for (int64_t j = 0; j < dk; ++j) {
          s[j] *= decay;
          prediction += s[j] * Load(ks, kbase + j);
        }
        const float delta = (Load(vs, (token * hv + head) * dv + value) - prediction) * Load(bs, token * hv + head);
        float output = 0;
        for (int64_t j = 0; j < dk; ++j) {
          s[j] += delta * Load(ks, kbase + j);
          output += s[j] * (Load(qs, kbase + j) * scale);
        }
        Store(dst, (token * hv + head) * dv + value, output);
      }
    });
    RecordProfileEvent(q, decode ? "gdn_decode_recurrence" : "gdn_prefill_recurrence", event);
  });
}
}
void GdnPrefillKernel(Queue& q, Tensor& out, const Tensor& qi, const Tensor& ki, const Tensor& vi,
                       const Tensor& g, const Tensor& beta, Tensor& state, const Tensor& qsl,
                       const GdnArgs& args) {
  TraceXpuOp(OpId::kGdnPrefill, q, {&out, &qi, &ki, &vi, &g, &beta, &state, &qsl});
  enum class Mode { kAuto, kReference, kChunked };
  static const Mode mode = [] {
    const char* value = std::getenv("VT_XPU_GDN_PREFILL");
    const std::string_view name = value ? value : "auto";
    VT_CHECK(name == "auto" || name == "reference" || name == "chunked", "Invalid VT_XPU_GDN_PREFILL");
    return name == "auto" ? Mode::kAuto : name == "chunked" ? Mode::kChunked : Mode::kReference;
  }();
#ifdef VLLM_CPP_XPU_XE2_GDN
  static const bool native_requested = [] {
    const char* value = std::getenv("VT_XPU_GDN_NATIVE");
    const std::string_view setting = value ? value : "0";
    VT_CHECK(setting == "0" || setting == "1", "Invalid VT_XPU_GDN_NATIVE");
    return setting == "1";
  }();
  const bool native_selected = native_requested &&
      GdnNativePrefillKernel(q, out, qi, ki, vi, g, beta, state, qsl, args);
#else
  const bool native_selected = false;
#endif
  bool chunked = mode == Mode::kChunked;
  if (mode == Mode::kAuto && qi.shape[1] == 16 && state.shape[1] == 48) {
    const auto device = NativeQueue(q).get_device();
    chunked = std::string_view(__VERSION__) == "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" &&
        device.get_info<sycl::info::device::driver_version>() == "1.17.39758+10" &&
        device.get_platform().get_info<sycl::info::platform::version>() == "1.17";
  }
  const bool supported = GdnChunkedPrefillEnabled();
  const bool eligible = chunked && supported && qi.shape[0] >= 64;
  const bool selected = !native_selected && eligible &&
      GdnChunkedPrefillKernel(q, out, qi, ki, vi, g, beta, state, qsl, args);
  if (const char* trace = std::getenv("VT_XPU_TRACE_FAST_PATH");
      trace != nullptr && trace[0] == '1' && trace[1] == '\0') {
    const char* reason = (selected || native_selected) ? "eligible" :
        !chunked ? "mode_or_stack_gate" : !supported ? "kernel_disabled" :
        qi.shape[0] < 64 ? "short_query" : "kernel_declined";
    const char* mode_name = mode == Mode::kAuto ? "auto" :
                            mode == Mode::kChunked ? "chunked" : "reference";
    std::fprintf(stderr,
                 "{\"event\":\"xpu_fast_path\",\"operator\":\"gdn_prefill\","
                 "\"selected\":\"%s\",\"reason\":\"%s\","
                 "\"mode\":\"%s\",\"tokens\":%lld}\n",
                 native_selected ? "native" : selected ? "chunked" : "reference",
                 reason, mode_name,
                 static_cast<long long>(qi.shape[0]));
  }
  if (selected || native_selected) return;
  Recurrence(q, out, qi, ki, vi, g, beta, state, &qsl, nullptr, args.scale);
}
void GdnDecodeKernel(Queue& q, Tensor& out, const Tensor& qi, const Tensor& ki, const Tensor& vi,
                      const Tensor& g, const Tensor& beta, Tensor& state, const Tensor* indices,
                      const GdnArgs& args) {
  TraceXpuOp(OpId::kGdnDecode, q, {&out, &qi, &ki, &vi, &g, &beta, &state, indices});
  Recurrence(q, out, qi, ki, vi, g, beta, state, nullptr, indices, args.scale);
}
void GdnSpecDecodeKernel(Queue& q, Tensor& out, const Tensor& qi,
                         const Tensor& ki, const Tensor& vi, const Tensor& g,
                         const Tensor& beta, Tensor& state, const Tensor& qsl,
                         const Tensor& indices, const Tensor& accepted,
                         const GdnArgs& args) {
  TraceXpuOp(OpId::kGdnSpecDecode, q,
             {&out, &qi, &ki, &vi, &g, &beta, &state, &qsl, &indices, &accepted});
  const int64_t requests = indices.shape[0], cols = indices.shape[1];
  const int64_t tokens = qi.shape[0], slots = state.shape[0];
  const int64_t hk = qi.shape[1], hv = state.shape[1];
  const int64_t dv = state.shape[2], dk = state.shape[3];
  VT_CHECK(state.dtype == DType::kF32, "XPU spec GDN requires F32 state");
  VT_CHECK(dk > 0 && dk <= 128, "XPU spec GDN supports Dk <= 128");
  for (const auto* operand :
       std::initializer_list<const Tensor*>{&out, &qi, &ki, &vi, &g, &beta,
                                            &qsl, &indices, &accepted})
    VT_CHECK(!Overlap(state, *operand), "XPU spec GDN state must have separate storage");
  const auto* offsets = static_cast<const int32_t*>(qsl.data);
  const auto* ids = static_cast<const int32_t*>(indices.data);
  const auto* nat = static_cast<const int32_t*>(accepted.data);
  CheckDeviceMetadata(q, [=] {
    if (offsets[0] != 0 || offsets[requests] != tokens) return false;
    for (int64_t r = 0; r < requests; ++r) {
      if (offsets[r] < 0 || offsets[r + 1] < offsets[r] ||
          offsets[r + 1] > tokens || offsets[r + 1] - offsets[r] > cols ||
          nat[r] < 1 || nat[r] > cols) return false;
      for (int64_t c = 0; c < cols; ++c) {
        const int32_t slot = ids[r * cols + c];
        if (slot >= slots) return false;
        if (slot < 0) continue;
        for (int64_t prior = 0; prior < r; ++prior)
          for (int64_t pc = 0; pc < cols; ++pc)
            if (ids[prior * cols + pc] == slot) return false;
      }
    }
    return true;
  }, "XPU spec GDN invalid offsets, accepted count or state slot",
      {&qsl, &indices, &accepted});
  WithOutput(q, out, {&qi, &ki, &vi, &g, &beta, &qsl, &indices, &accepted},
             [&](Tensor& target) {
    const View dst(target), qs(qi), ks(ki), vs(vi), gs(g), bs(beta);
    auto* cache = static_cast<float*>(state.data);
    const float scale = args.scale;
    const int64_t items = requests * hv * dv;
    if (!items) return;
    const char* slm_setting = std::getenv("VT_XPU_GDN_SPEC_SLM");
    const std::string_view slm_mode = slm_setting ? slm_setting : "1";
    VT_CHECK(slm_mode == "0" || slm_mode == "1", "Invalid VT_XPU_GDN_SPEC_SLM");
    bool slm_selected = slm_mode == "1" && requests <= 4 && cols <= 4 && hk == 16 && hv == 48 &&
        dk == 128 && dv == 128 && qi.dtype == DType::kF16 && ki.dtype == DType::kF16 &&
        vi.dtype == DType::kF16 && g.dtype == DType::kF32 && beta.dtype == DType::kF32 &&
        std::getenv("VT_XPU_GDN_SPEC_WG") == nullptr;
    if (slm_selected) {
      const auto device = NativeQueue(q).get_device();
      const auto sizes = device.get_info<sycl::info::device::sub_group_sizes>();
      slm_selected = device.get_info<sycl::info::device::local_mem_size>() >= 32768 &&
          device.get_info<sycl::info::device::max_work_group_size>() >= 64 &&
          std::find(sizes.begin(), sizes.end(), 32) != sizes.end();
    }
    if (slm_selected) {
      const char* typed_setting = std::getenv("VT_XPU_GDN_SPEC_SLM_TYPED");
      const std::string_view typed_mode = typed_setting ? typed_setting : "1";
      VT_CHECK(typed_mode == "0" || typed_mode == "1", "Invalid VT_XPU_GDN_SPEC_SLM_TYPED");
      const bool typed_selected = typed_mode == "1" && target.dtype == DType::kF16 &&
          target.IsContiguous() && qi.IsContiguous() && ki.IsContiguous() &&
          vi.IsContiguous() && g.IsContiguous() && beta.IsContiguous();
      // Each WG owns 64 independent value rows within one head/request.
      // Contiguous transfers use [K,value xor low K bits] SLM so both transfer
      // and value-row accesses spread over the local addresses. Each
      // value owner retains the original ascending-K scalar accumulation.
      // No private128-element state, parallel-K reduction or precision change.
      constexpr int values = 64, keys = 128;
      // Typed memory removes dynamic View dtype branches from the inner-K
      // loops. Preserve the same SLM layout and ascending-K arithmetic.
      const auto* query_half = static_cast<const sycl::half*>(qi.data);
      const auto* key_half = static_cast<const sycl::half*>(ki.data);
      const auto* value_half = static_cast<const sycl::half*>(vi.data);
      const auto* gates_float = static_cast<const float*>(g.data);
      const auto* beta_float = static_cast<const float*>(beta.data);
      auto* output_half = static_cast<sycl::half*>(target.data);
      const auto launch = [&]<bool Typed>() {
        return NativeQueue(q).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> s(sycl::range<1>(values * keys), h);
        h.parallel_for(sycl::nd_range<1>(items, values),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(32)]] {
#pragma clang fp contract(off)
          const int64_t index = item.get_group_linear_id() * values;
          const int lane = item.get_local_linear_id();
          const int64_t request = index / (hv * dv), head = (index / dv) % hv;
          const int64_t first_value = index % dv, value = first_value + lane;
          const int64_t first = offsets[request], last = offsets[request + 1];
          const int32_t initial = ids[request * cols + nat[request] - 1];
          const int64_t out_channel = head * dv + value;
          const auto write = [&](int64_t pos, float x) {
            if constexpr (Typed) output_half[pos] = sycl::half(x);
            else Store(dst, pos, x);
          };
          if (initial < 0) {
            for (int64_t token = first; token < last; ++token)
              write(token * hv * dv + out_channel, 0.0f);
            return;  // Entire WG has the same request and initial slot.
          }
          if (first == last) return;
          const int64_t initial_base = ((int64_t(initial) * hv + head) * dv + first_value) * keys;
          for (int pos = lane; pos < values * keys; pos += values)
            s[(pos % keys) * values + ((pos / keys) ^ (pos % values))] = cache[initial_base + pos];
          item.barrier(sycl::access::fence_space::local_space);
          const int64_t key_head = head / (hv / hk);
          for (int64_t token = first; token < last; ++token) {
            const int64_t key_base = (token * hk + key_head) * keys;
            const float gate = Typed ? gates_float[token * hv + head] : Load(gs, token * hv + head);
            const float decay = sycl::exp(gate);
            float prediction = 0.0f;
#pragma unroll 1
            for (int j = 0; j < keys; ++j) {
              const int address = j * values + (lane ^ (j % values));
              const float decayed = s[address] * decay;
              s[address] = decayed;
              const float key_value = Typed ? static_cast<float>(key_half[key_base + j])
                                            : Load(ks, key_base + j);
              prediction += decayed * key_value;
            }
            const float value_input = Typed ? static_cast<float>(value_half[token * hv * dv + out_channel])
                                            : Load(vs, token * hv * dv + out_channel);
            const float beta_input = Typed ? beta_float[token * hv + head] : Load(bs, token * hv + head);
            const float delta = (value_input - prediction) * beta_input;
            float output = 0.0f;
#pragma unroll 1
            for (int j = 0; j < keys; ++j) {
              const int address = j * values + (lane ^ (j % values));
              const float key_value = Typed ? static_cast<float>(key_half[key_base + j])
                                            : Load(ks, key_base + j);
              const float query_value = Typed ? static_cast<float>(query_half[key_base + j])
                                              : Load(qs, key_base + j);
              const float updated = s[address] + delta * key_value;
              s[address] = updated;
              output += updated * (query_value * scale);
            }
            write(token * hv * dv + out_channel, output);
            // A different lane transfers this owner's state row. Complete all
            // updates first, then all snapshot reads before the next token.
            item.barrier(sycl::access::fence_space::local_space);
            const int32_t snapshot = ids[request * cols + token - first];
            if (snapshot >= 0) {
              const int64_t base = ((int64_t(snapshot) * hv + head) * dv + first_value) * keys;
              for (int pos = lane; pos < values * keys; pos += values)
                cache[base + pos] = s[(pos % keys) * values + ((pos / keys) ^ (pos % values))];
            }
            item.barrier(sycl::access::fence_space::global_and_local);
          }
        });
      });
      };
      const auto event = typed_selected ? launch.operator()<true>() : launch.operator()<false>();
      RecordProfileEvent(q, typed_selected ? "gdn_spec_decode_slm_typed" : "gdn_spec_decode_slm", event);
      return;
    }
    const auto work = [=](int64_t index) {
      const int64_t request = index / (hv * dv);
      const int64_t head = (index / dv) % hv, value = index % dv;
      const int64_t first = offsets[request], last = offsets[request + 1];
      const int32_t initial = ids[request * cols + nat[request] - 1];
      const int64_t out_channel = head * dv + value;
      if (initial < 0) {
        for (int64_t token = first; token < last; ++token)
          Store(dst, token * hv * dv + out_channel, 0.0f);
        return;
      }
      if (first == last) return;
      float s[128];
      const int64_t initial_base = ((static_cast<int64_t>(initial) * hv + head) * dv + value) * dk;
      for (int64_t j = 0; j < dk; ++j) s[j] = cache[initial_base + j];
      const int64_t key_head = head / (hv / hk);
      for (int64_t token = first; token < last; ++token) {
        const int64_t key_base = (token * hk + key_head) * dk;
        const float decay = sycl::exp(Load(gs, token * hv + head));
        float prediction = 0.0f;
        for (int64_t j = 0; j < dk; ++j) {
          s[j] *= decay;
          prediction += s[j] * Load(ks, key_base + j);
        }
        const float delta =
            (Load(vs, token * hv * dv + out_channel) - prediction) *
            Load(bs, token * hv + head);
        float output = 0.0f;
        for (int64_t j = 0; j < dk; ++j) {
          s[j] += delta * Load(ks, key_base + j);
          output += s[j] * (Load(qs, key_base + j) * scale);
        }
        Store(dst, token * hv * dv + out_channel, output);
        const int32_t snapshot = ids[request * cols + token - first];
        if (snapshot >= 0) {
          const int64_t base = ((static_cast<int64_t>(snapshot) * hv + head) * dv + value) * dk;
          for (int64_t j = 0; j < dk; ++j) cache[base + j] = s[j];
        }
      }
    };
    const char* setting = std::getenv("VT_XPU_GDN_SPEC_WG");
    // The 27B MTP verification shape benefits from explicit B70 workgroups.
    // Keep other shapes on the existing range launch until they are measured.
    const int wg = setting ? std::atoi(setting) :
        requests == 1 && hk == 16 && hv == 48 && dk == 128 && dv == 128 &&
                tokens <= 5 ? 64 : 0;
    VT_CHECK(wg == 0 || wg == 32 || wg == 64 || wg == 128 || wg == 256,
             "VT_XPU_GDN_SPEC_WG must be 0, 32, 64, 128 or 256");
    sycl::event event;
    if (wg == 0) {
      event = NativeQueue(q).parallel_for(sycl::range<1>(items),
          [=](sycl::id<1> item) { work(item[0]); });
    } else {
      const int64_t rounded = ((items + wg - 1) / wg) * wg;
      event = NativeQueue(q).parallel_for(
          sycl::nd_range<1>(rounded, wg), [=](sycl::nd_item<1> item) {
            const int64_t index = static_cast<int64_t>(item.get_global_id(0));
            if (index < items) work(index);
          });
    }
    RecordProfileEvent(q, wg ? "gdn_spec_decode_wg" : "gdn_spec_decode", event);
  });
}
void RmsNormGatedKernel(Queue& q, Tensor& out, const Tensor& x, const Tensor& gate,
                         const Tensor& weight, const RmsNormGatedArgs& args) {
  TraceXpuOp(OpId::kRmsNormGated, q, {&out, &x, &gate, &weight});
  const auto width = x.shape[x.rank - 1], rows = x.Numel() / width;
  const auto group = gate.rank == 3 ? gate.shape[1] : 1;
  const auto eps = args.eps; const bool sigmoid = args.sigmoid_gate;
  WithOutput(q, out, {&x, &gate, &weight}, [&](Tensor& target) {
    const View src(x), dst(target), z(gate), w(weight);
    enum class NormMode { kAuto, kReference, kSubgroup };
    static const NormMode norm_mode = [] {
      const char* value = std::getenv("VT_XPU_GDN_GATED_NORM");
      const std::string_view name = value ? value : "auto";
      VT_CHECK(name == "auto" || name == "reference" || name == "subgroup",
               "Invalid VT_XPU_GDN_GATED_NORM");
      return name == "reference" ? NormMode::kReference :
             name == "subgroup" ? NormMode::kSubgroup : NormMode::kAuto;
    }();
    bool subgroup = false;
    if (width == 128 && norm_mode != NormMode::kReference) {
      const auto sizes = NativeQueue(q).get_device().get_info<sycl::info::device::sub_group_sizes>();
      subgroup = std::find(sizes.begin(), sizes.end(), 16) != sizes.end();
    }
    VT_CHECK(norm_mode != NormMode::kSubgroup || subgroup,
             "VT_XPU_GDN_GATED_NORM=subgroup requires width 128 and SG16");
    if (subgroup && src.dtype == DType::kF16 && z.dtype == DType::kF16 &&
        (w.dtype == DType::kF32 || w.dtype == DType::kF16) && !sigmoid && rows >= 32) {
      // Original eager GDN: D128 non-vectorized Torch mean, virtual SG32,
      // F32 rsqrt/SiLU and separate multiply boundaries before narrowing.
      // Typed operands keep this producer route separate from the generic
      // dtype and reciprocal-sqrt alternatives.
      const auto* input = static_cast<const sycl::half*>(src.data);
      const auto* gate_data = static_cast<const sycl::half*>(z.data);
      const char* table_setting = std::getenv("VT_XPU_GDN_GATED_SILU_TABLE");
      // Exact F32 SiLU intermediate for F16 gates, without the separate
      // SwiGLU table's F16 rounding. Setting0 retains the producer expression.
      const std::string_view table_mode = table_setting ? table_setting : "1";
      VT_CHECK(table_mode == "0" || table_mode == "1", "Invalid VT_XPU_GDN_GATED_SILU_TABLE");
      auto launch = [&]<bool cached>(const float* table) {
        constexpr int lanes = 16, workgroup = 128;
        const auto global = ((rows * lanes + workgroup - 1) / workgroup) * workgroup;
        const auto event = NativeQueue(q).parallel_for(
            sycl::nd_range<1>(global, workgroup),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
#pragma clang fp contract(off)
          const int64_t row = item.get_global_linear_id() / lanes;
          if (row >= rows) return;
          const int lane = item.get_local_linear_id() % lanes;
          auto sg = item.get_sub_group();
          float halves[2];
          for (int half = 0; half < 2; ++half) {
            float partial = 0.0f;
            for (int j = 0; j < 4; ++j) {
              const float value = input[row * 128 + lane + half * lanes + 32 * j];
              partial += value * value;
            }
            halves[half] = partial;
          }
          for (int offset = 1; offset < lanes; offset *= 2) {
            halves[0] += sycl::shift_group_left(sg, halves[0], offset);
            halves[1] += sycl::shift_group_left(sg, halves[1], offset);
          }
          const float mean = sycl::group_broadcast(sg, halves[0] + halves[1], 0) / 128.0f;
          const float inverse = sycl::rsqrt(mean + eps);
          const auto gbase = (row / group) * z.stride[0] + (row % group) * 128;
          for (int j = 0; j < 8; ++j) {
            const int col = lane + lanes * j;
            const float value = input[row * 128 + col];
            const float g = gate_data[gbase + col];
            float act;
            if constexpr (cached) {
              const auto bits = sycl::bit_cast<uint16_t>(gate_data[gbase + col]);
              act = (bits & 0x7c00) != 0x7c00 ? table[bits]
                  : sycl::ext::intel::math::fdiv_rn(g, 1.0f + sycl::exp(-g));
            } else act = sycl::ext::intel::math::fdiv_rn(g, 1.0f + sycl::exp(-g));
            const float normalized = value * inverse;
            const float weighted = normalized * Load(w, col);
            Store(dst, row * 128 + col, weighted * act);
          }
        });
        RecordProfileEvent(q, cached ? "gdn_gated_norm_fp16_producer_table" : "gdn_gated_norm_fp16_producer", event);
      };
      if (table_mode == "1" && WithGatedSiluTable(q, [&](void* storage) {
        auto* table = static_cast<float*>(storage);
        const auto event = NativeQueue(q).parallel_for(sycl::nd_range<1>(65536, 128),
            [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
#pragma clang fp contract(off)
          const auto bits = uint16_t(item.get_global_linear_id());
          const float g = float(sycl::bit_cast<sycl::half>(bits));
          table[bits] = sycl::ext::intel::math::fdiv_rn(g, 1.0f + sycl::exp(-g));
        });
        RecordProfileEvent(q, "gdn_gated_silu_table_build", event);
        return event;
      }, [&](const void* storage) { launch.template operator()<true>(static_cast<const float*>(storage)); })) return;
      launch.template operator()<false>(nullptr);
      return;
    }
    if (subgroup) {
      constexpr int lanes = 16, tile = 8, workgroup = 128;
      const auto global = ((rows * lanes + workgroup - 1) / workgroup) * workgroup;
      const auto event = NativeQueue(q).parallel_for(
          sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(workgroup)),
          [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
        const int64_t row = item.get_global_linear_id() / lanes;
        if (row >= rows) return;
        const int lane = item.get_local_linear_id() % lanes;
        float values[tile], sum = 0.0f;
        for (int j = 0; j < tile; ++j) {
          values[j] = Load(src, row * width + lane + lanes * j);
          sum += values[j] * values[j];
        }
        sum = sycl::reduce_over_group(item.get_sub_group(), sum, sycl::plus<float>());
        const float inv = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
        const auto gbase = (row / group) * z.stride[0] + (row % group) * width;
        for (int j = 0; j < tile; ++j) {
          const int col = lane + lanes * j;
          const float value = Load(z, gbase + col);
          const float act = sigmoid ? 1.0f / (1.0f + sycl::exp(-value)) : Silu(value);
          Store(dst, row * width + col, values[j] * inv * Load(w, col) * act);
        }
      });
      RecordProfileEvent(q, "gdn_gated_norm_subgroup", event);
      return;
    }
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(rows), [=](sycl::id<1> item) {
      const auto row = item[0];
      float sum = 0;
      for (int64_t j = 0; j < width; ++j) { const float v = Load(src, row * width + j); sum += v * v; }
      const float inv = 1.0f / sycl::sqrt(sum / static_cast<float>(width) + eps);
      const auto gbase = (row / group) * z.stride[0] + (row % group) * width;
      for (int64_t j = 0; j < width; ++j) {
        const float value = Load(z, gbase + j);
        const float act = sigmoid ? 1.0f / (1.0f + sycl::exp(-value)) : Silu(value);
        Store(dst, row * width + j, Load(src, row * width + j) * inv * Load(w, j) * act);
      }
    });
    RecordProfileEvent(q, "gdn_gated_norm", event);
  });
}
void GdnStateGatherKernel(Queue& q, Tensor& working, const Tensor& cache,
                           const Tensor& indices, const Tensor* initial) {
  TraceXpuOp(OpId::kGdnStateGather, q, {&working, &cache, &indices, initial});
  CheckSlots(q, indices, cache.shape[0], false, false);
  const auto inner = working.shape[working.rank - 1], physical = cache.shape[cache.rank - 1];
  const auto row_size = working.Numel() / indices.Numel(), mid = row_size / inner;
  WithOutput(q, working, {&cache, &indices, initial}, [&](Tensor& target) {
    const View dst(target), src(cache), flags(initial ? *initial : indices);
    const auto* ids = static_cast<const int32_t*>(indices.data); const bool has_flags = initial != nullptr;
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(working.Numel()), [=](sycl::id<1> item) {
      const auto row = item[0] / row_size, col = item[0] % row_size;
      const auto from = (ids[row] * mid + col / inner) * physical + col % inner;
      Store(dst, item[0], !has_flags || Initial(flags, row) ? Load(src, from) : 0.0f);
    });
    RecordProfileEvent(q, "gdn_state_gather", event);
  });
}
void GdnStateScatterKernel(Queue& q, Tensor& cache, const Tensor& working, const Tensor& indices) {
  TraceXpuOp(OpId::kGdnStateScatter, q, {&cache, &working, &indices});
  CheckSlots(q, indices, cache.shape[0], false, true);
  const auto inner = working.shape[working.rank - 1], physical = cache.shape[cache.rank - 1];
  const auto row_size = working.Numel() / indices.Numel(), mid = row_size / inner;
  WithOutput(q, cache, {&working, &indices}, [&](Tensor& target) {
    const View dst(target), src(working); const auto* ids = static_cast<const int32_t*>(indices.data);
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(working.Numel()), [=](sycl::id<1> item) {
      const auto row = item[0] / row_size, col = item[0] % row_size;
      const auto to = (ids[row] * mid + col / inner) * physical + col % inner;
      Store(dst, to, Load(src, item[0]));
    });
    RecordProfileEvent(q, "gdn_state_scatter", event);
  }, true);
}
}  // namespace vt::xpu
