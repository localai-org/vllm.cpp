// Register-blocked B70 EXL3 panel GEMM, included by xpu_exl3_prefill.cpp.
#pragma once

namespace vt::xpu {
namespace {
// Four subgroups/WG. Each subgroup owns 16x64 outputs as eight native 8x16
// accumulators, reusing each A fragment across four N fragments and each B
// fragment across two M fragments. No workgroup barrier in the K loop.
void RegisterPanelGemm(Queue& q, float* out, const sycl::half* input,
                      const sycl::half* panel, int64_t m, int64_t k,
                      int64_t leading_k, int64_t width, bool first) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  constexpr int kSG = 16, kGroups = 4, kRows = 16, kCols = 64, kInner = 16;
  const int64_t tiles_n = width / kCols;
  const int64_t groups = ((m + kRows * kGroups - 1) / (kRows * kGroups)) * tiles_n;
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    // Only the final partial subgroup tile uses this 2-KiB allocation.
    // Each subgroup owns a disjoint slab, so subgroup barriers suffice.
    sycl::local_accessor<sycl::half> tail_a(kGroups * kRows * kInner, h);
    h.parallel_for(sycl::nd_range<1>(groups * kSG * kGroups, kSG * kGroups),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      auto sg = item.get_sub_group();
      const int lane = sg.get_local_linear_id(), sub = sg.get_group_linear_id();
      const int64_t row = (item.get_group(0) / tiles_n) * (kRows * kGroups) + sub * kRows;
      const int64_t column = (item.get_group(0) % tiles_n) * kCols;
      if (row >= m) return;  // Uniform per subgroup; no workgroup barriers below.
      mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 8, 16,
                       mx::layout::row_major> a[2];
      mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16,
                       mx::layout::row_major> b[4];
      mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 8, 16> c[2][4];
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
          if (first) mx::joint_matrix_fill(sg, c[r][n], 0.0f);
          else mx::joint_matrix_load(sg, c[r][n],
              sycl::address_space_cast<sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(out + (row + 8 * r) * width + column + 16 * n),
              width, mx::layout::row_major);
        }
      }
      for (int64_t base = 0; base < k; base += kInner) {
        if (row + kRows <= m) {
          #pragma unroll
          for (int r = 0; r < 2; ++r)
            mx::joint_matrix_load(sg, a[r],
                sycl::address_space_cast<sycl::access::address_space::global_space,
                    sycl::access::decorated::no>(input + (row + 8 * r) * leading_k + base), leading_k);
        } else {
          for (int i = lane; i < kRows * kInner; i += kSG)
            tail_a[sub * kRows * kInner + i] = row + i / kInner < m
                ? input[(row + i / kInner) * leading_k + base + i % kInner] : sycl::half(0);
          sycl::group_barrier(sg);
          #pragma unroll
          for (int r = 0; r < 2; ++r)
            mx::joint_matrix_load(sg, a[r],
                tail_a.template get_multi_ptr<sycl::access::decorated::no>() +
                    sub * kRows * kInner + r * 8 * kInner, kInner);
        }
        #pragma unroll
        for (int n = 0; n < 4; ++n) {
          mx::joint_matrix_load(sg, b[n],
              sycl::address_space_cast<sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(panel + base * width + column + n * 16), width);
          #pragma unroll
          for (int r = 0; r < 2; ++r) mx::joint_matrix_mad(sg, c[r][n], a[r], b[n], c[r][n]);
        }
        // Prevent a tail subgroup from overwriting SLM while any lane still
        // consumes its current A fragment. Full tiles never execute a barrier.
        if (row + kRows > m) sycl::group_barrier(sg);
      }
      #pragma unroll
      for (int r = 0; r < 2; ++r) {
        #pragma unroll
        for (int n = 0; n < 4; ++n)
          mx::joint_matrix_store(sg, c[r][n],
              sycl::address_space_cast<sycl::access::address_space::global_space,
                  sycl::access::decorated::no>(out + (row + 8 * r) * width + column + 16 * n),
              width, mx::layout::row_major);
      }
      // Stores may include up to 15 padded rows. The caller reserves ceil(M/32)
      // rows; the unchanged output Hadamard consumes only the actual M rows.
    });
  });
  RecordProfileEvent(q, "exl3_panel_gemm", event);
}
}  // namespace
}  // namespace vt::xpu
