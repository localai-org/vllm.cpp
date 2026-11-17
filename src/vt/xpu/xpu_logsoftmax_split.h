#pragma once
#include "xpu_common.h"
#include <array>
#include <limits>
namespace vt::xpu::detail {
struct IgnoreLogsoftmaxCheckpoints {
  void Maximum(float) const {}
  void Exponential(int64_t, float) const {}
  void LaneSum(int64_t, float) const {}
  void LogSumExp(float) const {}
};
// The ordinary Normalize<1> lane sequence/reductions are unchanged. Only its
// pointwise exp and final output stores move to independent parallel kernels.
// Caller owns distinct input/output and two-float state through all consumers.
// No allocation, retirement, host completion or extra vocabulary buffer.
template<class Observer = IgnoreLogsoftmaxCheckpoints>
std::array<sycl::event,4> SubmitSplitRawLogsoftmax(Queue& q, float* dst,
    const float* src, float* stats, int64_t vocab, Observer observer = {}) {
  auto& native = NativeQueue(q);
  auto maximum = native.parallel_for(sycl::nd_range<1>(128,128),
      [=](sycl::nd_item<1> item) {
    const int64_t lane = item.get_local_id(0);
    float value = -std::numeric_limits<float>::infinity();
    for (int64_t j = lane; j < vocab; j += 128) value = sycl::fmax(value,src[j]);
    value = sycl::reduce_over_group(item.get_group(),value,sycl::maximum<float>());
    if (lane == 0) { stats[0] = value; observer.Maximum(value); }
  });
  auto exponentials = native.parallel_for(sycl::range<1>(vocab),[=](sycl::id<1> item) {
    const int64_t j = item[0];
    const float value = sycl::exp(src[j]-stats[0]);
    dst[j] = value; observer.Exponential(j,value);
  });
  auto sum = native.parallel_for(sycl::nd_range<1>(128,128),
      [=](sycl::nd_item<1> item) {
    const int64_t lane = item.get_local_id(0);
    float value = 0;
    for (int64_t j = lane; j < vocab; j += 128) value += dst[j];
    observer.LaneSum(lane,value);
    value = sycl::reduce_over_group(item.get_group(),value,sycl::plus<float>());
    if (lane == 0) {
      const float lse = stats[0]+sycl::log(value);
      stats[1] = lse; observer.LogSumExp(lse);
    }
  });
  auto output = native.parallel_for(sycl::range<1>(vocab),[=](sycl::id<1> item) {
    dst[item[0]] = src[item[0]]-stats[1];
  });
  return {maximum,exponentials,sum,output};
}
}  // namespace vt::xpu::detail
