#pragma once

#include <array>
#include <atomic>
#include <cstddef>
#include <cstdint>

#include "vllm/model_executor/model_loader/gptq4_weight.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vt/ops.h"
#ifdef VLLM_CPP_XPU
#include "vt/xpu.h"
#endif

namespace vllm::dense_gptq4 {

enum class Projection : size_t {
  kGdnQkvz,
  kGdnBa,
  kGdnOut,
  kAttnQkv,
  kAttnOut,
  kMlpGateUp,
  kMlpDown,
  kLmHead,
  kMtpFc,
  kMtpHead,
  kCount,
};

inline constexpr size_t kProjectionCount = static_cast<size_t>(Projection::kCount);
inline constexpr std::array<const char*, kProjectionCount> kProjectionNames{
    "gdn_qkvz", "gdn_ba", "gdn_out", "attn_qkv", "attn_out",
    "mlp_gate_up", "mlp_down", "lm_head", "mtp_fc", "mtp_head"};

struct DispatchCounts {
  std::array<uint64_t, kProjectionCount> calls{};
};

inline std::array<std::atomic<uint64_t>, kProjectionCount>& Counters() {
  static std::array<std::atomic<uint64_t>, kProjectionCount> counts{};
  return counts;
}

inline DispatchCounts GetDispatchCounts() {
  DispatchCounts result;
  for (size_t i = 0; i < kProjectionCount; ++i)
    result.calls[i] = Counters()[i].load(std::memory_order_relaxed);
  return result;
}

inline dense_attn::DBuf Packed(dense_attn::Dev d, const vt::Tensor& input,
                               const Gptq4Weight& weight, Projection projection) {
  VT_CHECK(d.q.device.type == vt::DeviceType::kXPU &&
               input.dtype == vt::DType::kF16 && input.rank == 2 &&
               input.IsContiguous() && input.device == d.q.device &&
               weight.k == input.shape[1] && weight.n > 0,
           "gptq4: packed projection requires contiguous XPU FP16 [M,K] and matching owner");
  const Gptq4ResidentViews resident = PrepareGptq4Resident(weight, d.q);
  dense_attn::DBuf output(d, vt::DType::kF16, {input.shape[0], weight.n});
#ifdef VLLM_CPP_XPU
  const vt::xpu::ProfileMatrixScope profile_projection(
      kProjectionNames[static_cast<size_t>(projection)]);
#endif
  vt::MatmulGptq4W4A16(d.q, output.t(), input, resident.qweight,
                       resident.scales, resident.zero_point, weight.group_size);
  Counters()[static_cast<size_t>(projection)].fetch_add(1, std::memory_order_relaxed);
  return output;
}

inline dense_attn::DBuf Dense(dense_attn::Dev d, const vt::Tensor& input,
                              const vt::Tensor& weight, Projection projection) {
  VT_CHECK(d.q.device.type == vt::DeviceType::kXPU &&
               input.dtype == vt::DType::kF16 && input.rank == 2 &&
               input.IsContiguous() && input.device == d.q.device &&
               weight.dtype == vt::DType::kF16 && weight.rank == 2 &&
               weight.IsContiguous() && weight.device == d.q.device &&
               weight.shape[1] == input.shape[1] && weight.shape[0] > 0,
           "gptq4: dense projection requires contiguous XPU FP16 [M,K] x [N,K]");
  dense_attn::DBuf output(d, vt::DType::kF16, {input.shape[0], weight.shape[0]});
#ifdef VLLM_CPP_XPU
  const vt::xpu::ProfileMatrixScope profile_projection(
      kProjectionNames[static_cast<size_t>(projection)]);
#endif
  vt::MatmulDenseF16(d.q, output.t(), input, weight);
  Counters()[static_cast<size_t>(projection)].fetch_add(1, std::memory_order_relaxed);
  return output;
}

inline dense_attn::DBuf Mlp(dense_attn::Dev d, const vt::Tensor& input,
                            const Gptq4Weight& gate_up,
                            const Gptq4Weight& down, int64_t intermediate) {
  VT_CHECK(intermediate > 0 && gate_up.n == 2 * intermediate &&
               down.k == intermediate && down.n == input.shape[1],
           "gptq4: merged gate/up and down geometry must match the dense MLP");
  dense_attn::DBuf both = Packed(d, input, gate_up, Projection::kMlpGateUp);
  dense_attn::DBuf activated(d, vt::DType::kF16,
                             {input.shape[0], intermediate});
  vt::SiluAndMul(d.q, activated.t(), both.t());
  return Packed(d, activated.t(), down, Projection::kMlpDown);
}

inline dense_attn::DBuf AttentionQkv(dense_attn::Dev d,
                                     const vt::Tensor& input,
                                     const Gptq4Weight& weight,
                                     int64_t query_gate_width,
                                     int64_t kv_width) {
  VT_CHECK(query_gate_width > 0 && kv_width > 0 &&
               weight.n == query_gate_width + 2 * kv_width,
           "gptq4: merged attention Q/K/V geometry mismatch");
  return Packed(d, input, weight, Projection::kAttnQkv);
}

}  // namespace vllm::dense_gptq4
