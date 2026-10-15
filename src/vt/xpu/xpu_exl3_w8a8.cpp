// Native wrappers adapted from exl3xpu c59d944 csrc/exl3_ops.sycl.
// Copyright (c) 2026 0xSero. MIT license: third_party/exl3xpu/LICENSE.
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wnan-infinity-disabled"
#include "xpu_common.h"
#include "xpu_kernels.h"
#include "xpu_gptq4.h"
#include "vt/exl3_grouped.h"
#include "vt/exl3_w8a8_panel_plan.h"
#pragma clang diagnostic pop
#include <exl3xpu/exl3_esimd.h>
#include <nlohmann/json.hpp>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <string_view>
#include <vector>

namespace vt::xpu {
#ifdef VLLM_CPP_XPU_ONEDNN
namespace {
template<class Kernel>
void Launch(Queue& q, int64_t count, int local, Kernel kernel, const char* stage) {
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    h.parallel_for(sycl::nd_range<1>(sycl::range<1>((count + local - 1) / local * local),
                                    sycl::range<1>(local)), kernel);
  });
  RecordProfileEvent(q, stage, event);
}

// The donor's float->INT8 conversion has no defined NaN/Inf contract. Reject
// those inputs, including overflow at either FP16 Hadamard rounding boundary,
// before writing any output. Bit tests remain valid in this fast-math TU.
struct ValidateRows {
  ::exl3::HadInQ8Kernel<sycl::half> input;
  uint32_t* valid;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    const int kb_n = input.Kdim / 128;
    const size_t id = it.get_global_id(0);
    if (id >= size_t(input.S) * input.M * kb_n) return;
    const int kb = int(id % kb_n), m = int(id / kb_n % input.M);
    const int g = int(id / kb_n / input.M);
    const auto xi = block_load<uint16_t, 128>(
        reinterpret_cast<const uint16_t*>(input.x) + size_t(m) * input.Kdim + kb * 128);
    const auto su = block_load<uint16_t, 128>(
        reinterpret_cast<const uint16_t*>(input.suh) + size_t(g) * input.Kdim + kb * 128);
    bool finite = !((xi & 0x7c00u) == 0x7c00u).any() &&
                  !((su & 0x7c00u) == 0x7c00u).any();
    // Check the product rounding separately: a subsequent max reduction is
    // not a valid detector for a NaN that some lanes may ignore.
    auto product_f32 =
        convert<float>(block_load<sycl::half, 128>(input.x + size_t(m) * input.Kdim + kb * 128)) *
        convert<float>(block_load<sycl::half, 128>(input.suh + size_t(g) * input.Kdim + kb * 128));
    // 65520 is the round-to-nearest FP16 overflow boundary. In this fast-math
    // TU a half round-trip can be optimized away, so validate the F32 bits
    // before narrowing as well as checking the resulting half bits.
    finite &= !((product_f32.bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
    simd<sycl::half, 128> product = convert<sycl::half>(product_f32);
    finite &= !((product.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any();
    // Mirror blk's transform, checking the second boundary before its half
    // conversion for the same reason. The producer itself remains unchanged.
    auto transformed = convert<float>(product);
    ::exl3::fwht128(transformed);
    transformed *= ::exl3::kRsqrt128;
    finite &= !((transformed.bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
    transformed = convert<float>(convert<sycl::half>(transformed));
    finite &= !((transformed.bit_cast_view<uint32_t>() & 0x7f800000u) == 0x7f800000u).any();
    if (!finite)
      atomic_update<atomic_op::store, uint32_t, 1>(valid, simd<uint32_t, 1>(0u),
                                                  simd<uint32_t, 1>(0u));
  }
};

struct ValidateOutputScale {
  const uint16_t* bits;
  int n;
  uint32_t* valid;
  void operator()(sycl::nd_item<1> it) const SYCL_ESIMD_KERNEL {
    using namespace sycl::ext::intel::esimd;
    const size_t block = it.get_global_id(0);
    if (block >= size_t(n / 128)) return;
    const auto values = block_load<uint16_t, 128>(bits + block * 128);
    if (((values & 0x7c00u) == 0x7c00u).any())
      atomic_update<atomic_op::store, uint32_t, 1>(valid, simd<uint32_t, 1>(0u),
                                                  simd<uint32_t, 1>(0u));
  }
};

// Adapt the donor's HadInQ8WgKernel: keep its FP16 register tiles, SLM maximum
// and quantization order, adding both checked FP16 boundaries. All stores are
// private until the host has checked the existing two-word status. Bad blocks
// become finite zeros before narrowing; a bad row uses scale1/inv0, so there
// is never a float-to-INT8 conversion of NaN, Inf or an out-of-range value.
template<int TPR, int NB>
struct PrepareRows {
  ::exl3::HadInQ8Kernel<sycl::half> input;
  uint32_t* valid;
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
        auto xi = block_load<sycl::half, 128>(input.x + size_t(m) * input.x_stride + kb * 128);
        auto su = block_load<sycl::half, 128>(input.suh + size_t(g) * input.Kdim + kb * 128);
        bool block_finite = !((xi.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any() &&
                            !((su.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any();
        auto v = convert<float>(xi) * convert<float>(su);
        block_finite &= !((v.bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
        if (!block_finite) v = 0.0f;
        simd<sycl::half, 128> product = convert<sycl::half>(v);
        block_finite &= !((product.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any();
        v = convert<float>(product);
        ::exl3::fwht128(v);
        v *= ::exl3::kRsqrt128;
        block_finite &= !((v.bit_cast_view<uint32_t>() & 0x7fffffffu) >= 0x477ff000u).any();
        if (!block_finite) v = 0.0f;
        simd<sycl::half, 128> vh = convert<sycl::half>(v);
        block_finite &= !((vh.bit_cast_view<uint16_t>() & 0x7c00u) == 0x7c00u).any();
        if (!block_finite) vh = sycl::half(0.0f);
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

template<int Bits>
void Reconstruct(Queue& q, const Tensor& tr, Tensor& panel, int k, int n,
                 int first_column, int columns) {
  ::exl3::ReconstructKernel<Bits, 2, 8, int8_t> kernel{
      static_cast<const uint32_t*>(tr.data), static_cast<int8_t*>(panel.data),
      n / 16, first_column / 16, columns, columns / 128, k / 16, 127.0f / 3.453125f};
  Launch(q, int64_t(k / 16) * (columns / 128), 8, kernel,
         "exl3_w8a8_weight_reconstruct");
}
struct PreparedActivation {
  void* storage;
  size_t bytes;
};

void DispatchW8A8(Queue& q, Tensor& out, const Tensor& in, const Tensor& tr,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    Tensor& workspace, Tensor& panel, const Exl3GroupedLinearArgs& args,
    const PreparedActivation* activation = nullptr) {
  const int m = int(in.shape[0]), k = int(in.shape[1]), n = int(out.shape[1]);
  const int groups = int(suh.shape[0]);
  auto plan = PlanExl3W8A8(m, k, n, groups, args.bits, args.w8a8_panel_columns);
  if (activation) {
    // Only the private model descriptor omits published xq/sx storage. Keep
    // the identical padded F16 Y, weight scale/status region and panel layout.
    plan.workspace_bytes -= plan.intermediate_offset;
    plan.weight_scale_offset -= plan.intermediate_offset;
    plan.intermediate_offset = 0;
  }
  if (workspace.rank == 0 && panel.rank == 0) {
    VT_CHECK(plan.weight_panel_bytes <= SIZE_MAX - plan.workspace_bytes,
             "EXL3 W8A8 shared scratch size overflow");
    const char* direct_setting = std::getenv("VT_XPU_W8A8_DIRECT_PREPARE");
    const std::string_view direct_mode = direct_setting ? direct_setting : "1";
    VT_CHECK(!args.model_map || direct_mode == "0" || direct_mode == "1",
             "Invalid VT_XPU_W8A8_DIRECT_PREPARE");
    const char* prepare_setting = std::getenv("VT_XPU_W8A8_PREPARE");
    if (args.model_map && direct_mode == "1" && k / 128 <= 144 &&
        (!prepare_setting || std::string_view(prepare_setting) == "1") &&
        plan.intermediate_offset <= 64 * 1024 * 1024) {
      const size_t compact_bytes = plan.workspace_bytes - plan.intermediate_offset;
      bool consumed = false;
      // Retain the existing workspace -> preparation lock order on all queues.
      // The preparation callback now encloses every oneDNN and tail consumer;
      // its completion fence retires the direct view before reuse or growth.
      (void)WithExl3W8A8Workspace(q, compact_bytes + plan.weight_panel_bytes,
          [&](void* storage) {
        consumed = WithExl3W8A8Preparation(q, plan.intermediate_offset,
            [&](void* prepared) {
          auto scratch = Tensor::Contiguous(storage, DType::kI8, q.device,
                                             {int64_t(compact_bytes)});
          auto weights = Tensor::Contiguous(static_cast<uint8_t*>(storage) + compact_bytes,
              DType::kI8, q.device, {k, plan.weight_panel_columns});
          const PreparedActivation view{prepared, plan.intermediate_offset};
          DispatchW8A8(q, out, in, tr, suh, svh, shard, scratch, weights, args, &view);
        });
      });
      if (consumed) return;
      // Insufficient preparation budget retains the original checked path.
      // The compact attempt has not written output, scratch or panel data.
    }
    const bool acquired = WithExl3W8A8Workspace(q, plan.workspace_bytes + plan.weight_panel_bytes,
        [&](void* storage) {
          auto scratch = Tensor::Contiguous(storage, DType::kI8, q.device,
                                             {int64_t(plan.workspace_bytes)});
          auto weights = Tensor::Contiguous(static_cast<uint8_t*>(storage) + plan.workspace_bytes,
              DType::kI8, q.device, {k, plan.weight_panel_columns});
          // Operand admission already ran before entering the registered op.
          // These exact plan-sized private descriptors satisfy scratch layout;
          // the recursive kernel retains all alignment/overlap/arithmetic checks.
          // Keep the model certificate across this private workspace recursion.
          DispatchW8A8(q, out, in, tr, suh, svh, shard, scratch, weights, args);
        });
    VT_CHECK(acquired, "EXL3 W8A8 shared scratch exceeds device memory budget");
    return;
  }
  TraceXpuOp(OpId::kExl3GroupedW8A8, q,
             {&out, &in, &tr, &suh, &svh, &shard, &workspace, &panel});
  for (const Tensor* t : std::initializer_list<const Tensor*>{
           &out, &in, &tr, &suh, &svh, &shard, &workspace, &panel})
    VT_CHECK(reinterpret_cast<uintptr_t>(t->data) % 16 == 0,
             "EXL3 W8A8 storage requires 16-byte alignment");
  for (const Tensor* dst : {&out, &workspace, &panel})
    for (const Tensor* src : {&in, &tr, &suh, &svh, &shard})
      VT_CHECK(!Overlap(*dst, *src), "EXL3 W8A8 writable storage may not overlap operands");
  VT_CHECK(!Overlap(out, workspace) && !Overlap(out, panel) && !Overlap(workspace, panel),
           "EXL3 W8A8 output and scratch may not overlap");
  if (activation) {
    const auto storage = Tensor::Contiguous(activation->storage, DType::kI8, q.device,
                                            {int64_t(activation->bytes)});
    VT_CHECK(reinterpret_cast<uintptr_t>(storage.data) % 16 == 0,
             "EXL3 W8A8 storage requires 16-byte alignment");
    for (const Tensor* t : std::initializer_list<const Tensor*>{
             &out, &in, &tr, &suh, &svh, &shard, &workspace, &panel})
      VT_CHECK(!Overlap(storage, *t), "EXL3 W8A8 prepared storage may not overlap operands");
  }

  // Eager initial implementation: mapping is small, and the completion wait
  // also ensures that the panel's previous use has retired before reuse.
  std::vector<Exl3W8A8Panel> checked_panels;
  if (!args.model_map) {
    const auto start = HostProfileSpansEnabled() ? HostProfileClockNs() : 0;
    std::vector<int32_t> mapping(n / 128);
    auto& backend = GetBackend(q.device);
    backend.Copy(q, mapping.data(), shard.data, mapping.size() * sizeof(int32_t));
    backend.Synchronize(q);
    checked_panels = PlanExl3W8A8Panels(mapping, groups, args.w8a8_panel_columns);
    if (start) RecordHostProfileSpan(q, "exl3_w8a8_map_readback_and_plan", start,
        HostProfileClockNs());
  }
  const auto& panels = args.model_map ? args.model_map->Panels(args.w8a8_panel_columns)
                                    : checked_panels;
  auto& backend = GetBackend(q.device);
  const auto* sv_bits = static_cast<const uint16_t*>(svh.data);

  auto* bytes = static_cast<uint8_t*>(workspace.data);
  auto* xq = activation ? static_cast<int8_t*>(activation->storage)
                       : reinterpret_cast<int8_t*>(bytes + plan.activation_offset);
  auto* sx = activation
      ? reinterpret_cast<float*>(static_cast<uint8_t*>(activation->storage) + size_t(groups) * plan.padded_rows * k)
      : reinterpret_cast<float*>(bytes + plan.row_scale_offset);
  auto* y = reinterpret_cast<sycl::half*>(bytes + plan.intermediate_offset);
  auto* sw = reinterpret_cast<float*>(bytes + plan.weight_scale_offset);
  // Both GPU checks use spare words in the existing 64-byte scale region.
  // Read their results together before output/panel/activation writes. This
  // avoids a separate metadata allocation, readback and retirement drain.
  auto* valid = reinterpret_cast<uint32_t*>(sw + 1);
  NativeQueue(q).single_task([=] { valid[0] = 1; valid[1] = 1; });
  Launch(q, n / 128, 8, ValidateOutputScale{sv_bits, n, valid + 1},
         "exl3_w8a8_validate_svh");
  ::exl3::HadInQ8Kernel<sycl::half> input{
      static_cast<const sycl::half*>(in.data), static_cast<const sycl::half*>(suh.data),
      xq, sx, m, k, groups, k, plan.padded_rows};
  const auto check_status = [&] {
    std::array<uint32_t, 2> finite{};
    backend.Copy(q, finite.data(), valid, sizeof(finite));
    backend.Synchronize(q);
    VT_CHECK(finite[1], "EXL3 W8A8 requires finite svh");
    VT_CHECK(finite[0], "EXL3 W8A8 requires finite inputs and finite FP16 transformed rows");
  };
  const char* prepare_setting = std::getenv("VT_XPU_W8A8_PREPARE");
  bool prepared = false;
  const size_t qbytes = size_t(groups) * plan.padded_rows * k;
  const size_t sbytes = size_t(groups) * plan.padded_rows * sizeof(float);
  const auto prepare_checked = [&](void* storage) {
    auto candidate = input;
    candidate.xq = static_cast<int8_t*>(storage);
    candidate.sx = reinterpret_cast<float*>(static_cast<uint8_t*>(storage) + qbytes);
    NativeQueue(q).memset(candidate.xq, 0, qbytes);
    NativeQueue(q).parallel_for(sycl::range<1>(size_t(groups) * plan.padded_rows),
        [=](sycl::id<1> i) { candidate.sx[i[0]] = 1.0f; });
    if (k / 128 <= 48)
      Launch(q, int64_t(groups) * m * 8, 8, PrepareRows<8, 6>{candidate, valid},
             "exl3_w8a8_prepare_checked");
    else
      Launch(q, int64_t(groups) * m * 16, 16, PrepareRows<16, 9>{candidate, valid},
             "exl3_w8a8_prepare_checked");
    check_status();
    return candidate;
  };
  // Exact checked preparation is the default;0 restores the original route.
  // Unsupported K or insufficient private budget keeps the existing checks.
  if (activation) {
    VT_CHECK(activation->bytes >= qbytes + sbytes, "EXL3 W8A8 prepared storage too small");
    (void)prepare_checked(activation->storage);
    prepared = true;
  } else if ((!prepare_setting || std::string_view(prepare_setting) == "1") && k / 128 <= 144) {
    prepared = WithExl3W8A8Preparation(q, qbytes + sbytes, [&](void* storage) {
      const auto candidate = prepare_checked(storage);
      // Public preparation stays untouched on either validation failure.
      RecordGraphWrite(q, workspace.data, Span(workspace));
      const auto qe = NativeQueue(q).memcpy(xq, candidate.xq, qbytes);
      const auto se = NativeQueue(q).memcpy(sx, candidate.sx, sbytes);
      RecordProfileEvent(q, "exl3_w8a8_prepared_commit", qe);
      RecordProfileEvent(q, "exl3_w8a8_prepared_commit", se);
    });
  }
  if (!prepared) {
    Launch(q, int64_t(groups) * m * (k / 128), 8, ValidateRows{input, valid},
         "exl3_w8a8_validate_rows");
    check_status();
  }

  for (const Tensor* t : {&out, &workspace, &panel}) RecordGraphWrite(q, t->data, Span(*t));
  // Poison/stale padding is never consumed by GEMM. The active producer writes
  // every byte of each real row. For zero rows its exact fallback is scale1.
  if (!prepared) {
    NativeQueue(q).memset(xq, 0, size_t(groups) * plan.padded_rows * k);
    NativeQueue(q).parallel_for(sycl::range<1>(size_t(groups) * plan.padded_rows),
                            [=](sycl::id<1> i) { sx[i[0]] = 1.0f; });
  }
  NativeQueue(q).single_task([=] { *sw = 3.453125f / 127.0f; });
  if (!prepared) {
    if (k / 128 <= 48) {
      ::exl3::HadInQ8WgKernel<sycl::half, 8, 6> had{
          input.x, input.suh, xq, sx, m, k, groups, k, plan.padded_rows};
      Launch(q, int64_t(groups) * m * 8, 8, had, "exl3_w8a8_input_quantize");
    } else if (k / 128 <= 144) {
      ::exl3::HadInQ8WgKernel<sycl::half, 16, 9> had{
          input.x, input.suh, xq, sx, m, k, groups, k, plan.padded_rows};
      Launch(q, int64_t(groups) * m * 16, 16, had, "exl3_w8a8_input_quantize");
    } else {
      Launch(q, int64_t(groups) * m, 8, input, "exl3_w8a8_input_quantize");
    }
  }

  const auto weight_scale = Tensor::Contiguous(sw, DType::kF32, q.device, {1});
  for (const auto& part : panels) {
    if (args.bits == 4) Reconstruct<4>(q, tr, panel, k, n, part.first_column, part.columns);
    else Reconstruct<6>(q, tr, panel, k, n, part.first_column, part.columns);
    const int group = part.source_group;
    const auto a = Tensor::Contiguous(xq + size_t(group) * plan.padded_rows * k,
        DType::kI8, q.device, {plan.padded_rows, k});
    const auto scales = Tensor::Contiguous(sx + size_t(group) * plan.padded_rows,
        DType::kF32, q.device, {plan.padded_rows});
    auto dst = Tensor::Contiguous(y + part.first_column, DType::kF16, q.device,
                                  {plan.padded_rows, part.columns});
    dst.stride[0] = n;
    // Reconstruction writes a compact K*width prefix, including short tails.
    const auto weight_view = Tensor::Contiguous(panel.data, DType::kI8, q.device,
                                                {k, part.columns});
    Exl3W8A8Matmul(q, dst, a, weight_view, scales, weight_scale);
  }
  // oneDNN rounds to F16 before the output Hadamard, as the production donor
  // does. The fallback I32 HadOutQ8 recipe is a different arithmetic route.
  ::exl3::HadOutKernel<sycl::half, sycl::half> tail{
      y, static_cast<const sycl::half*>(svh.data), static_cast<sycl::half*>(out.data),
      m, n, 1, n};
  Launch(q, int64_t(m) * (n / 128), 8, tail, "exl3_w8a8_output_hadamard");
  const char* setting = std::getenv("VT_XPU_EXL3_TRACE");
  if (setting && std::string_view(setting) == "1") {
    const nlohmann::json event = {{"event", "exl3_w8a8_dispatch"}, {"queue_id", q.id},
        {"matrix", args.debug_name ? args.debug_name : ""}, {"m", m}, {"padded_m", plan.padded_rows},
        {"k", k}, {"n", n}, {"groups", groups}, {"bits", args.bits},
        {"leaf", "signed_int8_onednn_f16_hadamard"}, {"intermediate_stride", n},
        {"weight_panel_bytes", plan.weight_panel_bytes}, {"workspace_bytes", plan.workspace_bytes},
        {"weight_panel_columns", plan.weight_panel_columns}, {"panel_submissions", panels.size()},
        {"direct_preparation", activation != nullptr},
        {"input_dtype", Name(in.dtype)}, {"output_dtype", Name(out.dtype)}};
    std::fprintf(stderr, "EXL3_W8A8_DISPATCH %s\n", event.dump().c_str());
  }
}
}  // namespace

void Exl3GroupedW8A8Kernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& tr,
    const Tensor& suh, const Tensor& svh, const Tensor& shard,
    Tensor& workspace, Tensor& panel, const Exl3GroupedLinearArgs& args) {
  DispatchW8A8(q, out, in, tr, suh, svh, shard, workspace, panel, args);
}
#else
void Exl3GroupedW8A8Kernel(Queue&, Tensor&, const Tensor&, const Tensor&,
    const Tensor&, const Tensor&, const Tensor&, Tensor&, Tensor&,
    const Exl3GroupedLinearArgs&) {
  VT_CHECK(false, "EXL3 W8A8 requires VLLM_CPP_XPU_ONEDNN=ON (oneDNN 3.13.0)");
}
#endif
}  // namespace vt::xpu
