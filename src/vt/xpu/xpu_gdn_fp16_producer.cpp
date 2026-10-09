#include "xpu_common.h"
#include "xpu_kernels.h"
#include "chunk_gated_delta_rule_fp16_producer.hpp"
#include "gated_delta_rule_fp16_decode.hpp"
#include "vt/gdn_fp16_plan.h"

#include <cmath>

namespace vt::xpu {
namespace {
constexpr int Hk = 16, Hv = 48, D = 128;
using T = cutlass::half_t;
}  // namespace
void GdnPrefillRawGateKernel(Queue& queue, Tensor& out, const Tensor& qi,
    const Tensor& ki, const Tensor& vi, const Tensor& raw_a, const Tensor& beta,
    const Tensor& a_log, const Tensor& dt_bias, Tensor& state, const Tensor& qsl,
    const GdnArgs&) {
  auto& native = NativeQueue(queue);
  const int sequences = int(state.shape[0]);
  const auto plan = PlanGdnFp16Batch(qi.shape[0], sequences);
  const int tokens = plan.tokens, capacity = plan.capacity;
  const auto device = native.get_device();
  VT_CHECK(device.has(sycl::aspect::ext_intel_device_id) &&
               device.get_info<sycl::ext::intel::info::device::device_id>() == 57891 &&
               device.has(sycl::aspect::ext_intel_matrix),
           "gdn_prefill_raw_gate: requires the B70 matrix device");
  for (const Tensor* input : {&qi, &ki, &vi, &raw_a, &beta, &a_log, &dt_bias, &qsl}) {
    VT_CHECK(!Overlap(out, *input) && !Overlap(state, *input),
             "gdn_prefill_raw_gate: output/state overlaps an input");
  }
  VT_CHECK(!Overlap(out, state), "gdn_prefill_raw_gate: output overlaps state");
  TraceXpuOp(OpId::kGdnPrefillRawGate, queue,
             {&out, &qi, &ki, &vi, &raw_a, &beta, &a_log, &dt_bias, &state, &qsl});
  RecordGraphWrite(queue, state.data, Span(state));
  const auto* offsets = static_cast<const int32_t*>(qsl.data);
  CheckDeviceMetadata(queue, [=] {
    if (offsets[0] != 0 || offsets[sequences] != tokens) return false;
    for (int s = 0; s < sequences; ++s)
      if (offsets[s + 1] <= offsets[s]) return false;
    return true;
  }, "gdn_prefill_raw_gate: offsets must cover nonempty logical sequences", {&qsl});

  const bool submitted = WithGdnNativeWorkspace(queue, plan.bytes, [&](void* storage) {
    auto* cursor = static_cast<char*>(storage);
    auto* q = reinterpret_cast<T*>(cursor + plan.q_offset);
    auto* k = reinterpret_cast<T*>(cursor + plan.k_offset);
    auto* v = reinterpret_cast<T*>(cursor + plan.v_offset);
    auto* A = reinterpret_cast<T*>(cursor + plan.a_matrix_offset);
    auto* w = reinterpret_cast<T*>(cursor + plan.w_offset);
    auto* u = reinterpret_cast<T*>(cursor + plan.u_offset);
    auto* a = reinterpret_cast<float*>(cursor + plan.raw_a_offset);
    auto* b = reinterpret_cast<float*>(cursor + plan.beta_offset);
    auto* bias = reinterpret_cast<sycl::half*>(cursor + plan.bias_offset);
    auto* index = reinterpret_cast<int*>(cursor + plan.index_offset);
    auto* initial = reinterpret_cast<bool*>(cursor + plan.initial_offset);
    // The producer reads its zero-padded physical capacity at tile tails.
    native.memset(storage, 0, plan.bytes);
    if (sequences == 1) {
      native.memcpy(q, qi.data, size_t(tokens) * Hk * D * sizeof(T));
      native.memcpy(k, ki.data, size_t(tokens) * Hk * D * sizeof(T));
      native.memcpy(v, vi.data, size_t(tokens) * Hv * D * sizeof(T));
    } else {
      const auto* qs = static_cast<const T*>(qi.data);
      const auto* ks = static_cast<const T*>(ki.data);
      const auto* vs = static_cast<const T*>(vi.data);
      // The donor addresses concatenated64-token chunks, not the unpadded
      // token stream. Pack each sequence independently on device; no request
      // serialization, host prefix readback or extra workspace reservation.
      native.parallel_for(sycl::range<1>(size_t(tokens) * (2 * Hk + Hv) * D), [=](sycl::id<1> id) {
        const int t = id[0] / ((2 * Hk + Hv) * D);
        const int feature = id[0] % ((2 * Hk + Hv) * D);
        int start = 0, virtual_start = 0;
        for (int s = 0; s < sequences; ++s) {
          const int end = offsets[s + 1];
          if (t < end) break;
          virtual_start += (end - start + 63) / 64 * 64;
          start = end;
        }
        const int virtual_t = virtual_start + t - start;
        if (feature < Hk * D) q[virtual_t * Hk * D + feature] = qs[t * Hk * D + feature];
        else if (feature < 2 * Hk * D) {
          const int col = feature - Hk * D;
          k[virtual_t * Hk * D + col] = ks[t * Hk * D + col];
        } else {
          const int col = feature - 2 * Hk * D;
          v[virtual_t * Hv * D + col] = vs[t * Hv * D + col];
        }
      });
    }
    const View av(raw_a), bv(beta), dv(dt_bias);
    native.parallel_for(sycl::range<1>(size_t(Hv) * tokens), [=](sycl::id<1> id) {
      const int h = id[0] / tokens, t = id[0] % tokens;
      int start = 0, virtual_start = 0;
      for (int s = 0; s < sequences; ++s) {
        const int end = offsets[s + 1];
        if (t < end) break;
        virtual_start += (end - start + 63) / 64 * 64;
        start = end;
      }
      const int virtual_t = virtual_start + t - start;
      a[h * capacity + virtual_t] = Load(av, t * av.stride[0] + h);
      b[h * capacity + virtual_t] = Load(bv, t * bv.stride[0] + h);
      if (t == 0) bias[h] = sycl::half(Load(dv, h));
    });
    // The caller has prepared this working state: zero for a fresh request,
    // or gathered persistent FP32 values for continuation. Always consume it.
    native.parallel_for(sycl::range<1>(sequences), [=](sycl::id<1> id) {
      index[id] = int(id[0]); initial[id] = true;
    });
    gdn::fp16_producer::kernel_launcher<T, float>(native,
        static_cast<T*>(out.data), q, k, v, A, w, u, b, a,
        static_cast<const float*>(a_log.data), reinterpret_cast<const T*>(bias),
        static_cast<float*>(state.data), Hv * D * D, offsets, index, initial,
        nullptr, sequences, capacity, Hk, D, Hv, D);
  });
  VT_CHECK(submitted, "gdn_prefill_raw_gate: native workspace unavailable");
}
void GdnPackedDecodeKernel(Queue& queue, Tensor& out, const Tensor& mixed,
    const Tensor& raw_a, const Tensor& raw_b, const Tensor& a_log,
    const Tensor& dt_bias, Tensor& state, const Tensor& indices,
    const GdnArgs& args) {
  auto& native = NativeQueue(queue);
  const auto device = native.get_device();
  VT_CHECK(device.has(sycl::aspect::ext_intel_device_id) &&
               device.get_info<sycl::ext::intel::info::device::device_id>() == 57891,
           "gdn_packed_decode: FP16 producer requires the B70");
  VT_CHECK(mixed.dtype == DType::kF16 && mixed.shape[0] == 1 &&
               mixed.shape[1] == (2 * Hk + Hv) * D &&
               raw_a.dtype == DType::kF16 && raw_b.dtype == DType::kF16 &&
               out.dtype == DType::kF16 && state.dtype == DType::kF32 &&
               a_log.dtype == DType::kF32 && dt_bias.dtype == DType::kF32 &&
               state.shape[1] == Hv && state.shape[2] == D && state.shape[3] == D &&
               state.shape[0] > 0 &&
               std::abs(args.scale - 1.0f / std::sqrt(float(D))) < 1e-6f,
           "gdn_packed_decode: qualified FP16 producer geometry is C1/Hk16/Hv48/D128/F32 state");
  for (const Tensor* input : {&mixed, &raw_a, &raw_b, &a_log, &dt_bias, &indices}) {
    VT_CHECK(!Overlap(out, *input) && !Overlap(state, *input),
             "gdn_packed_decode: output/state overlaps an input");
  }
  VT_CHECK(!Overlap(out, state), "gdn_packed_decode: output overlaps state");
  TraceXpuOp(OpId::kGdnPackedDecode, queue,
             {&out, &mixed, &raw_a, &raw_b, &a_log, &dt_bias, &state, &indices});
  RecordGraphWrite(queue, state.data, Span(state));
  const auto* index = static_cast<const int32_t*>(indices.data);
  const int64_t slots = state.shape[0];
  CheckDeviceMetadata(queue, [=] { return index[0] >= 0 && index[0] < slots; },
                      "gdn_packed_decode: invalid active state slot", {&indices});
  // The backend reserves the maximum supported completion-owned capacity
  // even when decode runs first. These small metadata offsets remain fixed.
  const bool submitted = WithGdnNativeWorkspace(queue, 256, [&](void* storage) {
    auto* bias = static_cast<sycl::half*>(storage);
    auto* offsets = reinterpret_cast<int*>(static_cast<char*>(storage) + 128);
    const View bv(dt_bias);
    native.parallel_for(sycl::range<1>(Hv), [=](sycl::id<1> h) {
      bias[h] = sycl::half(Load(bv, h));
    });
    native.single_task([=] { offsets[0] = 0; offsets[1] = 1; });
    const auto* input = static_cast<const sycl::half*>(mixed.data);
    using Kernel = gdn::fp16_decode_producer::gated_delta_rule_kernel<sycl::half, float, 4>;
    native.parallel_for(Kernel::get_nd_range(1, Hv, D),
        Kernel(static_cast<sycl::half*>(out.data), input, input + Hk * D,
            input + 2 * Hk * D, static_cast<const sycl::half*>(raw_b.data),
            static_cast<const sycl::half*>(raw_a.data),
            static_cast<const float*>(a_log.data), bias,
            static_cast<float*>(state.data), Hv * D * D, offsets, nullptr,
            index, nullptr, nullptr, 1, 1, Hk, D, Hv, D));
  });
  VT_CHECK(submitted, "gdn_packed_decode: native workspace unavailable");
}
}  // namespace vt::xpu
