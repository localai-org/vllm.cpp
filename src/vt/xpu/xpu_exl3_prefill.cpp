#include "xpu_common.h"
#include "xpu_exl3.h"
#include "xpu_kernels.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include "xpu_exl3_register_gemm.h"
#include <cstdlib>
#include <string_view>

namespace vt::xpu {
namespace {
constexpr size_t kWorkspaceBytes = 32 * 1024 * 1024;
constexpr int kBM = 32, kBN = 64, kBK = 32;

bool UseRegisterPanel(bool matrix, bool all_rows, int64_t rows) {
  static const std::string_view mode = [] {
    const char* value = std::getenv("VT_XPU_EXL3_PANEL_KERNEL");
    return std::string_view(value ? value : "auto");
  }();
  VT_CHECK(mode == "auto" || mode == "reference" || mode == "register",
           "Invalid EXL3 panel kernel selector");
  return matrix && (mode == "register" || (mode == "auto" && all_rows && rows >= 512));
}

template<int Bits>
void DecodePanel(Queue& q, sycl::half* panel, const uint32_t* packed,
                 int64_t k, int64_t n, int64_t base_k, int64_t column, int64_t width) {
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(k * width), [=](sycl::id<1> i) {
    const int64_t r = base_k + i[0] / width, c = column + i[0] % width;
    const auto* tile = packed + ((r / 16) * (n / 16) + c / 16) * (8 * Bits);
    panel[i[0]] = sycl::half(exl3::Decode(exl3::PackedCodeword<Bits>(tile, r % 16, c % 16), 2));
  });
  RecordProfileEvent(q, "exl3_panel_decode", event);
}

// Eight 16-lane subgroups cover one 32x64 tile. Two SLM stages overlap the
// next global loads with current matrix work; all subgroups use each stage
// before the workgroup barrier permits its reuse. Accumulators remain F32.
void PanelGemm(Queue& q, float* out, const sycl::half* input, const sycl::half* panel,
               int64_t m, int64_t k, int64_t leading_k, int64_t width, bool first) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  const int64_t tiles_n = width / kBN, groups = ((m + kBM - 1) / kBM) * tiles_n;
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<sycl::half> a(2 * kBM * kBK, h), b(2 * kBK * kBN, h);
    h.parallel_for(sycl::nd_range<1>(groups * 128, 128), [=](sycl::nd_item<1> item)
                     [[sycl::reqd_sub_group_size(16)]] {
      auto sg = item.get_sub_group();
      const int lane = item.get_local_id(0), group = lane / 16;
      const int mr = (group / 4) * 16, nc = (group % 4) * 16;
      const int64_t row = item.get_group(0) / tiles_n * kBM;
      const int64_t column = item.get_group(0) % tiles_n * kBN;
      auto stage = [&](int slot, int64_t base) {
        for (int i = lane; i < kBM * kBK; i += 128)
          a[slot * kBM * kBK + i] = row + i / kBK < m ?
              input[(row + i / kBK) * leading_k + base + i % kBK] : sycl::half(0);
        for (int i = lane; i < kBK * kBN; i += 128)
          b[slot * kBK * kBN + i] = panel[(base + i / kBN) * width + column + i % kBN];
      };
      mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> ja;
      mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> jb;
      mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> jc;
      const auto output = sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(out + (row + mr) * width + column + nc);
      if (first) mx::joint_matrix_fill(sg, jc, 0.0f);
      else mx::joint_matrix_load(sg, jc, output, width, mx::layout::row_major);
      stage(0, 0);
      item.barrier(sycl::access::fence_space::local_space);
      for (int64_t base = 0; base < k; base += kBK) {
        const int slot = (base / kBK) % 2;
        if (base + kBK < k) stage(1 - slot, base + kBK);
        #pragma unroll
        for (int offset = 0; offset < kBK; offset += 16) {
          mx::joint_matrix_load(sg, ja, a.template get_multi_ptr<sycl::access::decorated::no>() +
              slot * kBM * kBK + mr * kBK + offset, kBK);
          mx::joint_matrix_load(sg, jb, b.template get_multi_ptr<sycl::access::decorated::no>() +
              slot * kBK * kBN + offset * kBN + nc, kBN);
          mx::joint_matrix_mad(sg, jc, ja, jb, jc);
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      mx::joint_matrix_store(sg, jc, output, width, mx::layout::row_major);
    });
  });
  RecordProfileEvent(q, "exl3_panel_gemm", event);
}

template<int rows>
void ExactPanelGemm(Queue& q, float* out, const sycl::half* input, const sycl::half* panel,
                    int64_t m, int64_t k, int64_t leading_k, int64_t width, bool first) {
  const auto event = NativeQueue(q).parallel_for(sycl::range<1>(((m + rows - 1) / rows) * width), [=](sycl::id<1> item) {
    const int64_t row = (item[0] / width) * rows, col = item[0] % width;
    float acc[rows] = {};
    #pragma unroll
    for (int r = 0; r < rows; ++r)
      if (!first && row + r < m) acc[r] = out[(row + r) * width + col];
    for (int64_t inner = 0; inner < k; ++inner) {
      const float weight = static_cast<float>(panel[inner * width + col]);
      #pragma unroll
      for (int r = 0; r < rows; ++r) if (row + r < m) {
        const float value = static_cast<float>(input[(row + r) * leading_k + inner]);
        if (value != 0.0f) acc[r] += value * weight;
      }
    }
    #pragma unroll
    for (int r = 0; r < rows; ++r) if (row + r < m) out[(row + r) * width + col] = acc[r];
  });
  RecordProfileEvent(q, "exl3_panel_exact_gemm", event);
}
}

bool Exl3PrefillKernel(Queue& q, Tensor& out, const Tensor& in_had,
                       const Tensor& trellis, const Tensor& svh, int bits, bool matrix, bool all_rows) {
  const int64_t m = in_had.shape[0], k = in_had.shape[1], n = out.shape[1];
  const auto device = NativeQueue(q).get_device();
  if (!device.has(sycl::aspect::ext_intel_device_id) ||
      device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
      !device.has(sycl::aspect::ext_intel_matrix) || m < 1 || m > 6656) return false;
  // PERF-02 selects all-row text prefill at M>=512 on this B70. The register
  // kernel passed the fixed model gates and wins for every measured text family.
  const bool register_panel = UseRegisterPanel(matrix, all_rows, m);
  // The all-row schedule keeps every M row in the result panel, so each
  // compressed weight panel is decoded once across all rows.
  constexpr int64_t k_per_panel = 1024;
  const int64_t columns = all_rows ? 1024 : 4096;
  const int64_t rows_per_panel = all_rows ? m : 256;
  const int64_t padded_rows = ((rows_per_panel + kBM - 1) / kBM) * kBM;
  if (all_rows && !matrix) return false;
  if (columns * (2 * k_per_panel + 4 * padded_rows) > static_cast<int64_t>(kWorkspaceBytes))
    return false;
  return WithExl3Workspace(q, kWorkspaceBytes, [&](void* workspace) {
    auto* panel = static_cast<sycl::half*>(workspace);
    auto* result = reinterpret_cast<float*>(panel + k_per_panel * columns);
    const auto* packed = static_cast<const uint32_t*>(trellis.data);
    const auto* input = static_cast<const sycl::half*>(in_had.data);
    for (int64_t row = 0; row < m; row += rows_per_panel) {
      const int64_t rows = std::min(rows_per_panel, m - row);
      auto output = out;
      output.data = static_cast<char*>(out.data) + row * n * SizeOf(out.dtype);
      output.shape[0] = rows;
      for (int64_t column = 0; column < n; column += columns) {
        const int64_t width = std::min(columns, n - column);
        for (int64_t base = 0; base < k; base += k_per_panel) {
          const int64_t count = std::min(k_per_panel, k - base);
          switch (bits) {
            case 3: DecodePanel<3>(q, panel, packed, count, n, base, column, width); break;
            case 4: DecodePanel<4>(q, panel, packed, count, n, base, column, width); break;
            case 5: DecodePanel<5>(q, panel, packed, count, n, base, column, width); break;
            case 6: DecodePanel<6>(q, panel, packed, count, n, base, column, width); break;
            default: VT_CHECK(false, "Unsupported EXL3 prefill width");
          }
          if (register_panel)
            RegisterPanelGemm(q, result, input + row * k + base, panel, rows, count, k, width, base == 0);
          else if (matrix) PanelGemm(q, result, input + row * k + base, panel, rows, count, k, width, base == 0);
          else if (width <= 1024)
            ExactPanelGemm<4>(q, result, input + row * k + base, panel, rows, count, k, width, base == 0);
          else ExactPanelGemm<8>(q, result, input + row * k + base, panel, rows, count, k, width, base == 0);
        }
        auto raw = Tensor::Contiguous(result, DType::kF32, q.device, {rows, width});
        Exl3OutputHadPanel(q, output, raw, svh, column);
      }
    }
  });
}
}  // namespace vt::xpu
