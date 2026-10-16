#include "xpu_common.h"
#include "xpu_kernels.h"
#include "chunk_gated_delta_rule_kernels_xe2.hpp"

#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <string_view>

namespace vt::xpu {
namespace {
bool Contiguous(const Tensor& t) {
  int64_t stride = 1;
  for (int dim = t.rank - 1; dim >= 0; --dim) {
    if (t.stride[dim] != stride) return false;
    stride *= t.shape[dim];
  }
  return true;
}
size_t Align64(size_t n) { return (n + 63) & ~size_t(63); }
}  // namespace

bool GdnNativePrefillKernel(Queue& queue, Tensor& out, const Tensor& qi,
                            const Tensor& ki, const Tensor& vi, const Tensor& g,
                            const Tensor& beta, Tensor& state, const Tensor& qsl,
                            const GdnArgs& args) {
  // Keep the scratch bound at one 4K macro segment while carrying the FP32
  // state across segments on the same in-order queue.
  constexpr int Segment = 4096, Hk = 16, Hv = 48, D = 128, C = 64;
  const int64_t tokens = qi.shape[0];
  const auto device = NativeQueue(queue).get_device();
  if (!device.has(sycl::aspect::ext_intel_device_id) ||
      device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
      std::string_view(__VERSION__) !=
          "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" ||
      device.get_info<sycl::info::device::driver_version>() != "1.17.39758+10" ||
      device.get_platform().get_info<sycl::info::platform::version>() != "1.17" ||
      !device.has(sycl::aspect::ext_intel_matrix) ||
      qi.rank != 3 || ki.rank != 3 || vi.rank != 3 || out.rank != 3 ||
      g.rank != 2 || beta.rank != 2 || state.rank != 4 || qsl.rank != 1 ||
      qi.dtype != DType::kF16 || ki.dtype != DType::kF16 ||
      vi.dtype != DType::kF16 || out.dtype != DType::kF16 ||
      g.dtype != DType::kF32 || beta.dtype != DType::kF32 ||
      state.dtype != DType::kF32 || qsl.dtype != DType::kI32 ||
      (tokens != Segment && tokens != 2 * Segment && tokens != 4 * Segment) ||
      qi.shape[1] != Hk || qi.shape[2] != D ||
      ki.shape[0] != tokens || ki.shape[1] != Hk || ki.shape[2] != D ||
      vi.shape[0] != tokens || vi.shape[1] != Hv || vi.shape[2] != D ||
      out.shape[0] != tokens || out.shape[1] != Hv || out.shape[2] != D ||
      g.shape[0] != tokens || g.shape[1] != Hv ||
      beta.shape[0] != tokens || beta.shape[1] != Hv ||
      state.shape[0] != 1 || state.shape[1] != Hv ||
      state.shape[2] != D || state.shape[3] != D || qsl.shape[0] != 2 ||
      !Contiguous(qi) || !Contiguous(ki) || !Contiguous(vi) ||
      !Contiguous(out) || !Contiguous(g) || !Contiguous(beta) ||
      !Contiguous(state) || !Contiguous(qsl) ||
      std::abs(args.scale - 1.0f / std::sqrt(float(D))) > 1e-6f)
    return false;

  for (const Tensor* input : {&qi, &ki, &vi, &g, &beta, &qsl}) {
    if (Overlap(out, *input) || Overlap(state, *input)) return false;
  }
  if (Overlap(out, state)) return false;
  const auto* offsets = static_cast<const int32_t*>(qsl.data);
  CheckDeviceMetadata(queue, [=] { return offsets[0] == 0 && offsets[1] == tokens; },
                      "XPU native GDN requires one full macro-segment sequence", {&qsl});

  const size_t a_bytes = size_t(Hv) * Segment * C * sizeof(sycl::half);
  const size_t w_bytes = size_t(Hv) * Segment * D * sizeof(sycl::half);
  const size_t u_bytes = w_bytes;
  const size_t q_bytes = size_t(Segment) * Hk * D * sizeof(sycl::half);
  const size_t gates_bytes = size_t(Segment) * Hv * sizeof(float);
  const size_t scratch_bytes = Align64(a_bytes) + Align64(w_bytes) +
      Align64(u_bytes) + Align64(q_bytes) + 2 * Align64(gates_bytes) + 192;
  return WithGdnNativeWorkspace(queue, scratch_bytes, [&](void* storage) {
    auto* cursor = static_cast<char*>(storage);
    auto take = [&](size_t n) {
      void* result = cursor;
      cursor += Align64(n);
      return result;
    };
    auto* A = static_cast<cutlass::half_t*>(take(a_bytes));
    auto* w = static_cast<cutlass::half_t*>(take(w_bytes));
    auto* u = static_cast<cutlass::half_t*>(take(u_bytes));
    auto* q_scaled = static_cast<cutlass::half_t*>(take(q_bytes));
    auto* a = static_cast<float*>(take(gates_bytes));
    auto* b = static_cast<float*>(take(gates_bytes));
    auto* segment_offsets = static_cast<int*>(take(64));
    auto* cache_index = static_cast<int*>(take(64));
    auto* has_initial = reinterpret_cast<bool*>(cursor);
    auto& native = NativeQueue(queue);
    std::optional<sycl::event> begin;
    if (ProfileQueueEventsEnabled()) begin = native.ext_oneapi_submit_barrier();
    const auto host_start = HostProfileSpansEnabled() ? HostProfileClockNs() : 0;
    native.single_task([=] {
      segment_offsets[0] = 0;
      segment_offsets[1] = Segment;
      cache_index[0] = 0;
      has_initial[0] = true;
    });
    auto* q_out = reinterpret_cast<sycl::half*>(q_scaled);
    for (int base = 0; base < tokens; base += Segment) {
      native.memset(A, 0, a_bytes + w_bytes + u_bytes);
      const auto* q_in = static_cast<const sycl::half*>(qi.data) + base * Hk * D;
      native.parallel_for(sycl::range<1>(size_t(Segment) * Hk * D),
                          [=](sycl::id<1> i) {
        q_out[i] = sycl::half(float(q_in[i]) * args.scale);
      });
      gdn::kernel_launcher<cutlass::half_t, float>(
          native, static_cast<cutlass::half_t*>(out.data) + base * Hv * D,
          q_scaled,
          static_cast<const cutlass::half_t*>(ki.data) + base * Hk * D,
          static_cast<const cutlass::half_t*>(vi.data) + base * Hv * D,
          A, w, u, b, a,
          static_cast<const float*>(g.data) + base * Hv,
          static_cast<const float*>(beta.data) + base * Hv,
          static_cast<float*>(state.data), Hv * D * D, segment_offsets,
          cache_index, has_initial, nullptr, 1, Segment, Hk, D, Hv, D);
    }
    if (begin) {
      const auto end = native.ext_oneapi_submit_barrier();
      RecordProfileSpan(queue, "gdn_native_prefill", *begin, end, host_start,
                        "macro4096,Hv48,Hk16,D128,F16,F32-state");
    }
  });
}
}  // namespace vt::xpu
