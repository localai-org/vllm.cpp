// Native wrappers adapted from exl3xpu c59d944 csrc/exl3_ops.sycl.
// Copyright (c) 2026 0xSero. MIT license: third_party/exl3xpu/LICENSE.
#pragma clang diagnostic push
// Shared headers declare FP8/attention NaN constants that this translation
// unit never uses. Keep the donor's fast-math scope on its ESIMD kernels.
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#include "xpu_common.h"
#include "xpu_kernels.h"
#include "vt/exl3_grouped.h"
#include "vt/xpu_graph_metadata.h"
#pragma clang diagnostic pop
#include <exl3xpu/exl3_esimd.h>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>
#include <nlohmann/json.hpp>
#include <cstdio>
#include <cstdlib>
#include <string_view>

namespace vt::xpu {
namespace {
template<class Kernel> struct LargeGrf {
  Kernel kernel;
  void operator()(sycl::nd_item<1> item) const SYCL_ESIMD_KERNEL { kernel(item); }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<256>};
  }
};
template<class Kernel>
void Launch(Queue& q, int64_t count, Kernel kernel, const char* stage) {
  constexpr int local = 8;
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>((count + local - 1) / local * local),
                                    sycl::range<1>(local)), kernel);
  });
  RecordProfileEvent(q, stage, event);
}

template<int Bits, int MB, int NT>
void Dpas(Queue& q, const Tensor& in, const Tensor& trellis, const Tensor& shard,
          Tensor& partials, int m, int k, int n, const Exl3SmallMPlan& plan) {
  const int strips = n / 16 / NT, blocks = (m + MB - 1) / MB;
  ::exl3::DpasKernel<Bits, 2, MB, NT> kernel{
      static_cast<const sycl::half*>(in.data), static_cast<const uint32_t*>(trellis.data),
      static_cast<const int*>(shard.data), static_cast<float*>(partials.data),
      m, k, n, n / 16, plan.tile_rows_per_split, strips, blocks, blocks * MB, {}, 0};
  const int64_t count = int64_t(strips) * blocks * plan.splits;
  if constexpr (MB >= 40) {
    // Same default 256-GRF domain as the pinned MB40/48/64 launchers.
    const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
      h.parallel_for(sycl::nd_range<1>(sycl::range<1>((count + 7) / 8 * 8),
                                      sycl::range<1>(8)), LargeGrf{kernel});
    });
    RecordProfileEvent(q, "exl3_smallm_dpas", event);
  } else {
    Launch(q, count, kernel, "exl3_smallm_dpas");
  }
}

template<int Bits, int MR, int NT>
void Gemv(Queue& q, const Tensor& in, const Tensor& trellis, const Tensor& shard,
          Tensor& partials, int m, int k, int n, const Exl3SmallMPlan& plan) {
  const int strips = n / 16 / NT;
  ::exl3::GemvKernel<Bits, 2, MR, NT> kernel{
      static_cast<const sycl::half*>(in.data), static_cast<const uint32_t*>(trellis.data),
      static_cast<const int*>(shard.data), static_cast<float*>(partials.data),
      m, k, n, n / 16, plan.tile_rows_per_split, strips, {}, 0};
  Launch(q, int64_t(strips) * plan.splits, kernel, "exl3_smallm_gemv");
}

template<int Bits>
void Gemm(Queue& q, const Tensor& in, const Tensor& trellis, const Tensor& shard,
          Tensor& partials, int m, int k, int n, const Exl3SmallMPlan& plan) {
  if (plan.vector) {
    if (m == 1) Gemv<Bits, 1, Bits == 4 ? 8 : 4>(q, in, trellis, shard, partials, m, k, n, plan);
    else Gemv<Bits, 2, 4>(q, in, trellis, shard, partials, m, k, n, plan);
    return;
  }
  switch (plan.row_block) {
    case 8: Dpas<Bits, 8, 4>(q, in, trellis, shard, partials, m, k, n, plan); break;
    case 16: Dpas<Bits, 16, 4>(q, in, trellis, shard, partials, m, k, n, plan); break;
    case 24: Dpas<Bits, 24, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    case 32: Dpas<Bits, 32, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    case 40: Dpas<Bits, 40, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    case 48: Dpas<Bits, 48, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    case 64: Dpas<Bits, 64, 2>(q, in, trellis, shard, partials, m, k, n, plan); break;
    default: VT_CHECK(false, "EXL3 SmallM invalid DPAS row block");
  }
}
}

void Exl3GroupedLinearKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& trellis,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    Tensor& in_had, Tensor& partials, const Exl3GroupedLinearArgs& args) {
  TraceXpuOp(OpId::kExl3GroupedLinear, q,
             {&out, &in, &trellis, &suh, &svh, &shard, &in_had, &partials});
  const int m = int(in.shape[0]), k = int(in.shape[1]), n = int(out.shape[1]),
            groups = int(suh.shape[0]);
  const auto plan = PlanExl3SmallM(m, k, n, args.bits);
  VT_CHECK(reinterpret_cast<uintptr_t>(trellis.data) % alignof(uint32_t) == 0,
           "EXL3 SmallM requires word-aligned packed storage");
  for (const Tensor* t : std::initializer_list<const Tensor*>{
           &out, &in, &suh, &svh, &in_had, &partials})
    VT_CHECK(reinterpret_cast<uintptr_t>(t->data) % 16 == 0,
             "EXL3 SmallM vector storage requires 16-byte alignment");
  for (const Tensor* dst : {&out, &in_had, &partials}) {
    for (const Tensor* src : {&in, &trellis, &suh, &svh, &shard})
      VT_CHECK(!Overlap(*dst, *src), "EXL3 SmallM writable storage may not overlap operands");
  }
  VT_CHECK(!Overlap(out, in_had) && !Overlap(out, partials) && !Overlap(in_had, partials),
           "EXL3 SmallM output and scratch may not overlap");
  const auto* mapping = static_cast<const int32_t*>(shard.data);
  if (args.model_map) {
    // The versioned model certificate validated these immutable routing bytes.
    // Capture still rejects writes into the map and pins its allocation owner.
    RecordGraphImmutableRead(q, shard.data, Span(shard), args.model_map->ResidentOwner(),
                             "EXL3 SmallM immutable source map");
  } else {
    CheckDeviceMetadata(q, [=] {
      for (int nb = 0; nb < n / 128; ++nb)
        if (mapping[nb] < 0 || mapping[nb] >= groups) return false;
      return true;
    }, "EXL3 SmallM shard_of_nb group out of range", {&shard});
  }

  for (const Tensor* t : {&out, &in_had, &partials}) RecordGraphWrite(q, t->data, Span(*t));
  // Poisoned padding must never reach DPAS. The producer zeros the entire
  // blocked buffer when it has padded rows, then writes only the active rows.
  if (plan.padded_rows != m) NativeQueue(q).memset(in_had.data, 0, in_had.Bytes());
  ::exl3::HadInKernel<sycl::half> had{
      static_cast<const sycl::half*>(in.data), static_cast<const sycl::half*>(suh.data),
      static_cast<sycl::half*>(in_had.data), m, k, groups, k, plan.padded_rows, 0};
  Launch(q, int64_t(groups) * m * (k / 128), had, "exl3_smallm_input_hadamard");
  if (args.bits == 4) Gemm<4>(q, in_had, trellis, shard, partials, m, k, n, plan);
  else Gemm<6>(q, in_had, trellis, shard, partials, m, k, n, plan);
  ::exl3::HadOutKernel<sycl::half> tail{
      static_cast<const float*>(partials.data), static_cast<const sycl::half*>(svh.data),
      static_cast<sycl::half*>(out.data), m, n, plan.splits, n};
  Launch(q, int64_t(m) * (n / 128), tail, "exl3_smallm_output_hadamard");

  const char* setting = std::getenv("VT_XPU_EXL3_TRACE");
  if (setting && std::string_view(setting) == "1") {
    const nlohmann::json event = {{"event", "exl3_smallm_dispatch"}, {"queue_id", q.id},
        {"matrix", args.debug_name ? args.debug_name : ""}, {"m", m}, {"k", k}, {"n", n},
        {"groups", groups}, {"bits", args.bits}, {"codebook", args.codebook},
        {"leaf", plan.vector ? "producer_gemv" : "producer_dpas"},
        {"row_block", plan.row_block}, {"padded_m", plan.padded_rows},
        {"splits", plan.splits}, {"tile_rows_per_split", plan.tile_rows_per_split},
        {"input_dtype", Name(in.dtype)}, {"output_dtype", Name(out.dtype)}};
    std::fprintf(stderr, "EXL3_SMALLM_DISPATCH %s\n", event.dump().c_str());
  }
}
}  // namespace vt::xpu
