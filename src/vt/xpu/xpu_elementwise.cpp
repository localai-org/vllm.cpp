#include "xpu_common.h"
#include "xpu_kernels.h"
#include <sycl/ext/intel/math.hpp>
#include <cstdlib>
#include <string_view>

namespace vt::xpu {
void TraceXpuOp(OpId op, Queue& q, std::initializer_list<const Tensor*> tensors) {
  vt::TraceOpTensors(op, q, tensors);
  const auto mark = [&](size_t i) {
    if (i < tensors.size() && tensors.begin()[i]) {
      const auto& tensor = *tensors.begin()[i]; RecordGraphWrite(q, tensor.data, Span(tensor));
    }
  };
  switch (op) {
    case OpId::kQkvSplit: mark(0); mark(1); mark(2); break;
    case OpId::kVisionRopeApply: mark(0); mark(1); break;
    case OpId::kReshapeAndCache: case OpId::kReshapeAndCacheFp8: mark(2); mark(3); break;
    case OpId::kGdnPostConv: for (size_t i = 0; i < 5; ++i) mark(i); break;
    case OpId::kAttnGateSplit: case OpId::kRopeNeox: case OpId::kRopeFromCache: mark(0); mark(1); break;
    case OpId::kGdnDecode: case OpId::kGdnPrefill: mark(0); mark(6); break;
    case OpId::kCausalConv1dFwd: case OpId::kCausalConv1dUpdate: mark(0); mark(4); break;
    case OpId::kRmsNorm: mark(0); mark(3); break;
    case OpId::kExl3Gemm: mark(0); mark(5); break;
    default: mark(0); break;
  }
}
void CopyKernel(Queue& q, Tensor& out, const Tensor& in) {
  TraceXpuOp(OpId::kCopy, q, {&out, &in});
  const auto n = out.Numel();
  if (!n) return;
  const size_t bytes = SizeOf(out.dtype);
  VT_CHECK(out.dtype == in.dtype ||
               ((in.dtype == DType::kF32 || in.dtype == DType::kF16 || in.dtype == DType::kBF16)
                && (out.dtype == DType::kF32 || out.dtype == DType::kF16 || out.dtype == DType::kBF16)),
           "XPU copy: unsupported conversion");
  WithOutput(q, out, {&in}, [&](Tensor& target) {
    const View dst(target), src(in);
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(n), [=](sycl::id<1> item) {
      const auto i = item[0];
      const auto di = dst.offset(i), si = src.offset(i);
      if (dst.dtype == src.dtype) {
        for (size_t b = 0; b < bytes; ++b)
          static_cast<char*>(dst.data)[di * bytes + b] = static_cast<const char*>(src.data)[si * bytes + b];
      } else Store(dst, di, Load(src, si));
    });
    RecordProfileEvent(q, "copy_convert", event);
  });
}
void AddKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) {
  TraceXpuOp(OpId::kAdd, q, {&out, &a, &b});
  FloatTensor(out); FloatTensor(a); FloatTensor(b);
  WithOutput(q, out, {&a, &b}, [&](Tensor& target) {
    const View dst(target), av(a), bv(b);
    const bool broadcast = b.rank == 1 && a.rank != 1;
    const auto width = a.shape[a.rank - 1];
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto i = item[0];
      Store(dst, dst.offset(i), Load(av, av.offset(i)) + Load(bv, bv.offset(broadcast ? i % width : i)));
    });
    RecordProfileEvent(q, "add", event);
  });
}
void MoeSiluMulKernel(Queue& q, Tensor& out, const Tensor& gate, const Tensor& up) {
  TraceXpuOp(OpId::kMoeSiluMul, q, {&out, &gate, &up});
  FloatTensor(out); FloatTensor(gate); FloatTensor(up);
  WithOutput(q, out, {&gate, &up}, [&](Tensor& target) {
    const View dst(target), gv(gate), uv(up);
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto i = item[0]; const float g = Load(gv, gv.offset(i));
      const float act = Round(gv.dtype, g / (1.0f + sycl::exp(-g)));
      Store(dst, dst.offset(i), act * Load(uv, uv.offset(i)));
    });
    RecordProfileEvent(q, "moe_silu_mul", event);
  });
}
void SiluAndMulKernel(Queue& q, Tensor& out, const Tensor& in) {
  TraceXpuOp(OpId::kSiluAndMul, q, {&out, &in});
  FloatTensor(out); FloatTensor(in);
  WithOutput(q, out, {&in}, [&](Tensor& target) {
    const View dst(target), src(in);
    const auto width = out.shape[1];
    const char* typed = std::getenv("VT_XPU_SILU_FP16_TYPED");
    if (target.dtype == DType::kF16 && in.dtype == DType::kF16 &&
        target.IsContiguous() && in.IsContiguous() && out.shape[0] >= 64 &&
        width > 0 && (typed == nullptr || std::string_view(typed) != "0")) {
      const auto* input = static_cast<const sycl::half*>(in.data);
      auto* output = static_cast<sycl::half*>(target.data);
      constexpr size_t local = 256;
      const size_t columns = (size_t(width) + local - 1) / local * local;
      const char* table_setting = std::getenv("VT_XPU_SILU_FP16_TABLE");
      // Exact device-generated FP16 materialization; small-row decode retains
      // its original route. Setting0 preserves the typed expression for A/B.
      const std::string_view table_mode = table_setting ? table_setting : "1";
      VT_CHECK(table_mode == "0" || table_mode == "1", "Invalid VT_XPU_SILU_FP16_TABLE");
      if (table_mode == "1" && WithFp16SiluTable(q, [&](void* storage) {
        auto* table = static_cast<sycl::half*>(storage);
        const auto event = NativeQueue(q).parallel_for(
            sycl::nd_range<1>(65536, local), [=](sycl::nd_item<1> item) {
          const auto bits = uint16_t(item.get_global_linear_id());
          const float gate = float(sycl::bit_cast<sycl::half>(bits));
          const float denominator = 1.0f + sycl::exp(-gate);
          table[bits] = sycl::half(sycl::ext::intel::math::fdiv_rn(gate, denominator));
        });
        RecordProfileEvent(q, "silu_fp16_table_build", event);
        return event;
      }, [&](const void* storage) {
        const auto* table = static_cast<const sycl::half*>(storage);
        const auto event = NativeQueue(q).parallel_for(
            sycl::nd_range<2>(sycl::range<2>(size_t(out.shape[0]), columns),
                              sycl::range<2>(1, local)),
            [=](sycl::nd_item<2> item) {
          const int64_t row = item.get_global_id(0), col = item.get_global_id(1);
          if (col >= width) return;
          const int64_t offset = row * (2 * width) + col;
          const auto bits = sycl::bit_cast<uint16_t>(input[offset]);
          float rounded;
          if ((bits & 0x7c00) == 0x7c00) {
            // Preserve the original expression for Inf/NaN gate operands.
            const float gate = float(input[offset]);
            const float denominator = 1.0f + sycl::exp(-gate);
            rounded = float(sycl::half(sycl::ext::intel::math::fdiv_rn(gate, denominator)));
          } else rounded = float(table[bits]);
          output[row * width + col] = sycl::half(rounded * float(input[offset + width]));
        });
        RecordProfileEvent(q, "silu_and_mul_fp16_table", event);
      })) return;
      const auto event = NativeQueue(q).parallel_for(
          sycl::nd_range<2>(sycl::range<2>(size_t(out.shape[0]), columns),
                            sycl::range<2>(1, local)),
          [=](sycl::nd_item<2> item) {
        const int64_t row = item.get_global_id(0), col = item.get_global_id(1);
        if (col >= width) return;
        const int64_t offset = row * (2 * width) + col;
        const float gate = static_cast<float>(input[offset]);
        const float denominator = 1.0f + sycl::exp(-gate);
        const float silu = sycl::ext::intel::math::fdiv_rn(gate, denominator);
        const float rounded = static_cast<float>(sycl::half(silu));
        output[row * width + col] = sycl::half(rounded * static_cast<float>(input[offset + width]));
      });
      RecordProfileEvent(q, "silu_and_mul_fp16_typed", event);
      return;
    }
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto i = item[0]; const auto off = (i / width) * src.stride[0] + i % width;
      const float gate = Load(src, off);
      // Preserve the eager FP16 SiLU boundary before multiplying by up.
      // An approximate division can cross a half midpoint on real MLP rows.
      const float denominator = 1.0f + sycl::exp(-gate);
      const float silu = src.dtype == DType::kF16 && dst.dtype == DType::kF16
          ? sycl::ext::intel::math::fdiv_rn(gate, denominator) : gate / denominator;
      Store(dst, dst.offset(i), Round(src.dtype, silu) * Load(src, off + width));
    });
    RecordProfileEvent(q, "silu_and_mul", event);
  });
}
void SigmoidGateKernel(Queue& q, Tensor& out, const Tensor& attn, const Tensor& gate) {
  TraceXpuOp(OpId::kSigmoidGateBf16, q, {&out, &attn, &gate});
  FloatTensor(out); FloatTensor(attn); FloatTensor(gate);
  WithOutput(q, out, {&attn, &gate}, [&](Tensor& target) {
    const View dst(target), av(attn), gv(gate);
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto i = item[0];
      if (dst.dtype == DType::kF16 && av.dtype == DType::kF16) {
        // Eager FP16 attention multiplies by a materialized FP16 sigmoid.
        // Keep the gate input's F32 values, then narrow only this result.
        const float value = Load(gv, gv.offset(i));
        const float sigmoid = Round(DType::kF16,
            sycl::ext::intel::math::fdiv_rn(1.0f, 1.0f + sycl::exp(-value)));
        Store(dst, dst.offset(i), Load(av, av.offset(i)) * sigmoid);
        return;
      }
      Store(dst, dst.offset(i), Load(av, av.offset(i)) * (1.0f / (1.0f + sycl::exp(-Load(gv, gv.offset(i))))));
    });
    RecordProfileEvent(q, "sigmoid_gate", event);
  });
}
namespace {
int64_t Index(View idx, int64_t row) {
  return idx.dtype == DType::kI64 ? static_cast<const int64_t*>(idx.data)[idx.offset(row)]
                                : static_cast<const int32_t*>(idx.data)[idx.offset(row)];
}
void CheckIndices(Queue& q, const Tensor& idx, int64_t limit) {
  if (!idx.Numel()) return;
  const View ids(idx); const auto rows = idx.Numel();
  CheckDeviceMetadata(q, [=] {
    for (int64_t r = 0; r < rows; ++r) if (Index(ids, r) < 0 || Index(ids, r) >= limit) return false;
    return true;
  }, "XPU index out of range", {&idx});
}
void Rows(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx, bool scatter, bool embedding) {
  TraceXpuOp(embedding ? OpId::kEmbedding : scatter ? OpId::kIndexCopy : OpId::kIndexSelect, q, {&out, &in, &idx});
  CheckIndices(q, idx, scatter ? out.shape[0] : in.shape[0]);
  if (idx.Numel() == 0) return;
  if (embedding) { FloatTensor(out); FloatTensor(in); }
  const int64_t rows = idx.Numel();
  const int64_t width = (scatter ? in.Numel() : out.Numel()) / rows;
  const size_t bytes = SizeOf(out.dtype);
  WithOutput(q, out, {&in, &idx}, [&](Tensor& target) {
    const View dst(target), src(in), ids(idx);
    if (scatter && rows > 128) {
      // One group per source row: check last-write-wins once collectively,
      // rather than scanning all later indices again for every column.
      // No temporary allocation/readback; the same group then copies the row.
      constexpr size_t local = 256;
      const auto event = NativeQueue(q).parallel_for(
          sycl::nd_range<1>(sycl::range<1>(size_t(rows) * local), sycl::range<1>(local)),
          [=](sycl::nd_item<1> item) {
        const int64_t row = item.get_group_linear_id();
        const int64_t lane = item.get_local_linear_id();
        const int64_t index = Index(ids, row);
        bool later = false;
        for (int64_t r = row + 1 + lane; r < rows; r += local) {
          if (Index(ids, r) == index) { later = true; break; }
        }
        if (sycl::any_of_group(item.get_group(), later)) return;
        for (int64_t col = lane; col < width; col += local) {
          const int64_t di = index * dst.stride[0] + col;
          const int64_t si = row * src.stride[0] + col;
          for (size_t b = 0; b < bytes; ++b)
            static_cast<char*>(dst.data)[di * bytes + b] =
                static_cast<const char*>(src.data)[si * bytes + b];
        }
      });
      RecordProfileEvent(q, "index_copy_row_group", event);
      return;
    }
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(rows * width), [=](sycl::id<1> item) {
      const auto row = item[0] / width, col = item[0] % width;
      const int64_t index = Index(ids, row);
      if (scatter) {
        // Keep the existing launch for small scatters/decode shapes.
        for (int64_t r = row + 1; r < rows; ++r) if (Index(ids, r) == index) return;
      }
      const int64_t di = (scatter ? index : row) * dst.stride[0] + col;
      const int64_t si = (scatter ? row : index) * src.stride[0] + col;
      if (embedding && dst.dtype != src.dtype) Store(dst, di, Load(src, si));
      else for (size_t b = 0; b < bytes; ++b)
        static_cast<char*>(dst.data)[di * bytes + b] = static_cast<const char*>(src.data)[si * bytes + b];
    });
    RecordProfileEvent(q, embedding ? "embedding" : scatter ? "index_copy" : "index_select", event);
  }, scatter);
}
void Matmul(Queue& q, Tensor& out, const Tensor& a, const Tensor& b, bool transpose) {
  TraceXpuOp(transpose ? OpId::kMatmulBT : OpId::kMatmul, q, {&out, &a, &b});
  FloatTensor(out); FloatTensor(a); FloatTensor(b);
  WithOutput(q, out, {&a, &b}, [&](Tensor& target) {
    const View dst(target), av(a), bv(b);
    const auto n = out.shape[1], k = a.shape[1];
    const auto event = NativeQueue(q).parallel_for(sycl::range<1>(out.Numel()), [=](sycl::id<1> item) {
      const auto row = item[0] / n, col = item[0] % n;
      float sum = 0;
      for (int64_t inner = 0; inner < k; ++inner) {
        const auto bi = transpose ? col * bv.stride[0] + inner * bv.stride[1]
                                  : inner * bv.stride[0] + col * bv.stride[1];
        sum += Load(av, row * av.stride[0] + inner * av.stride[1]) * Load(bv, bi);
      }
      Store(dst, row * dst.stride[0] + col * dst.stride[1], sum);
    });
    RecordProfileEvent(q, transpose ? "matmul_bt" : "matmul", event);
  });
}
}
void IndexSelectKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx) { Rows(q, out, in, idx, false, false); }
void IndexCopyKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx) { Rows(q, out, in, idx, true, false); }
void EmbeddingKernel(Queue& q, Tensor& out, const Tensor& in, const Tensor& idx) { Rows(q, out, in, idx, false, true); }
void MatmulKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) { Matmul(q, out, a, b, false); }
void MatmulBTKernel(Queue& q, Tensor& out, const Tensor& a, const Tensor& b) { Matmul(q, out, a, b, true); }
}  // namespace vt::xpu
