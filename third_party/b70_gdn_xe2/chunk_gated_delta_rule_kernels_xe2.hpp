// Adapted from vllm-project/vllm-xpu-kernels release/0.1.15.4
// commit ddf336d86e3c8602888572a3502f951abd51df12 (Apache-2.0).
// The native SYCL kernel body is retained; the PyTorch dispatcher is omitted.
#pragma once

#include <cstdint>
#include <sycl/sycl.hpp>

#include "gemm.hpp"
#include "gdn_attn_utils.h"

namespace gdn {
using namespace cute;
static constexpr int MaxThreadsPerSM = 512;
static constexpr int elem_per_item = 2;
static constexpr int sub_group_size = 16;
static constexpr float eps = 0.000001f;
static constexpr int chunk_size = gdn::chunk_size_xe2;

struct chunk_gemm_policy_64x64x32_2x1 {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_2, _1, _1>, Stride<_1, _1, _0>>;
};

struct chunk_gemm_policy_64x64x32_2x2 {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_2, _2, _1>, Stride<_2, _1, _0>>;
};

struct chunk_gemm_policy_64x64x32_4x2 {
  using WGTile = Shape<_64, _64, _32>;
  using SGLayout = Layout<Shape<_4, _2, _1>, Stride<_2, _1, _0>>;
};

struct chunk_gemm_policy_16x16x16 {
  using WGTile = Shape<_16, _16, _16>;
  using SGLayout = Layout<Shape<_1, _1, _1>, Stride<_1, _1, _0>>;
};

using chunk_gemm_policy_compute_A = chunk_gemm_policy_64x64x32_2x2;
using chunk_gemm_policy_inverse = chunk_gemm_policy_16x16x16;
using chunk_gemm_policy_compute_wu = chunk_gemm_policy_64x64x32_2x1;
using chunk_gemm_policy_fwd_o = chunk_gemm_policy_64x64x32_4x2;

CUTE_DEVICE float
act_softplus(float& x, float beta = 1.0f, float threshold = 20.0f) {
  if (beta * x < threshold) {
    return sycl::log(1.0f + sycl::exp(beta * x)) / beta;
  } else
    return x;
}

template <typename T>
CUTE_DEVICE void chunk_prepare_kernel(
    float* a,
    float* b,
    const float* g_token_major,
    const float* beta_token_major,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_v_heads,
    const int head_v_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int group_id = item.get_group(1);
  int group_range = item.get_group_range(1);
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_range = sg.get_group_linear_range();
  int sg_local_id = sg.get_local_linear_id();

  int total_sg_range = group_range * sg_range;
  int total_sg_id = group_id * sg_range + sg_id;

  int pre_chunks = 0;
  const int chunk_range = total_sg_range / num_v_heads;
  int chunk_id = total_sg_id % chunk_range;
  const int v_head_id = total_sg_id / chunk_range;
  // When total_sg_range is not a multiple of num_v_heads, the top sub-group(s)
  // get v_head_id >= num_v_heads and would write past the end of a[num_v_heads,
  // ...] (OOB).
  if (v_head_id >= num_v_heads) return;

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int chunk_start_offset = chunk_id * chunk_size;

      // assume that (chunk_size % sub_group_size == 0)
      constexpr int local_num = chunk_size / sub_group_size;
      float g_local[local_num] = {};
      float g_local_sum = 0.0f;
      CUTE_UNROLL
      for (int c = 0; c < local_num; ++c) {
        const int token = chunk_start_offset + sg_local_id * local_num + c;
        g_local[c] = g_token_major[token * num_v_heads + v_head_id];
        b[token + v_head_id * total_virtual_seqlen] =
            beta_token_major[token * num_v_heads + v_head_id];
      }
      CUTE_UNROLL
      for (int c = 0; c < local_num; ++c) {
        g_local_sum += g_local[c];
      }
      g_local_sum =
          sycl::inclusive_scan_over_group(sg, g_local_sum, sycl::plus<float>());
      CUTE_UNROLL
      for (int c = local_num - 1; c >= 0; --c) {
        a[(chunk_start_offset + sg_local_id * local_num + c) +
          v_head_id * total_virtual_seqlen] = g_local_sum;
        g_local_sum -= g_local[c];
      }

      chunk_id += chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_compute_A_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,
    T* A,
    const T* k,
    const T* v,
    const float* b,
    const float* a,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  int local_range = item.get_local_range(2);
  int chunk_id = item.get_group(1);
  const int global_chunk_range = item.get_group_range(1);

  auto sg = item.get_sub_group();
  int sg_local_id = sg.get_local_linear_id();

  float* slm_mem = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());
  float* g_slm_ptr = slm_mem;

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;

  auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int m_tile_start = 0;
  int n_tile_start = 0;
  int m_sg_start = sg_local_m_coord * SG_M;
  int n_sg_start = sg_local_n_coord * SG_N;

  int pre_chunks = 0;

  const int kv_ratio = num_v_heads / num_k_heads;

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int chunk_start_offset = chunk_id * chunk_size;

      for (int v_head_id = 0; v_head_id < num_v_heads; ++v_head_id) {
        // WAR fence: sub-groups from the previous (chunk, v_head)
        // iteration may still be reading g_slm_ptr in the exp()
        // epilogue; all must arrive before any sub-group refills the
        // SLM (g_slm_ptr) for this iteration.
        sycl::group_barrier(item.get_group());
        CUTE_UNROLL
        for (int e = local_id; e < chunk_size; e += local_range) {
          g_slm_ptr[e] =
              a[(chunk_start_offset + e) + v_head_id * total_virtual_seqlen];
        }

        sycl::group_barrier(item.get_group());

        auto k_ptr = k +
                     static_cast<int64_t>(chunk_start_offset) * num_k_heads *
                         head_k_dim +
                     (v_head_id / kv_ratio) * head_k_dim;
        auto K_tensor_shape = make_shape(chunk_size, head_k_dim);
        auto K_tensor = make_tensor(
            make_gmem_ptr(k_ptr),
            make_layout(
                K_tensor_shape, make_stride(head_k_dim * num_k_heads, _1{})));

        auto A_ptr = A +
                     static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                         chunk_size +
                     chunk_start_offset * chunk_size;
        auto A_tensor_shape = make_shape(chunk_size, chunk_size);
        auto A_tensor = make_tensor(
            make_gmem_ptr(A_ptr),
            make_layout(A_tensor_shape, make_stride(chunk_size, _1{})));

        Tensor cA = make_identity_tensor(A_tensor.shape());
        Tensor gA_C =
            local_tile(cA, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});

        auto copy_A_c = get_block_2d_copy_D<void>(mma, A_tensor);
        auto thr_copy_A_c = copy_A_c.get_slice(local_id);
        auto tCrA_c = thr_copy_A_c.partition_sg_fragment_S(gA_C);
        auto tCgA_c = thr_copy_A_c.partition_D(gA_C);
        auto tSrA_c = thr_mma.partition_sg_fragment_C(gA_C);

        clear(tSrA_c);
        gemm_TTS(K_tensor, K_tensor, tSrA_c, 0, 0, mma);

        CUTE_UNROLL
        for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
          int n_idx =
              n_tile_start + n_sg_start + sn * sub_group_size + sg_local_id;
          CUTE_UNROLL
          for (int sm = 0; sm < SG_M; ++sm) {
            int m_idx = m_tile_start + m_sg_start + sm;
            float beta_value =
                b[(chunk_start_offset + m_idx) +
                  v_head_id * total_virtual_seqlen];

            tSrA_c(sn * SG_M + sm) *=
                sycl::exp(g_slm_ptr[(m_idx)] - g_slm_ptr[n_idx]) * beta_value;
            if (m_idx == n_idx) {
              tSrA_c(sn * SG_M + sm) = 1.0f;
            }
            if (m_idx < n_idx) {
              tSrA_c(sn * SG_M + sm) = 0.0f;
            }
          }
        }

        reorder(tSrA_c, tCrA_c);
        copy(copy_A_c, tCrA_c, tCgA_c);
      }
      chunk_id += global_chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

template <typename T>
CUTE_DEVICE void chunk_inverse_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,
    T* A,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  int local_range = item.get_local_range(2);
  int chunk_id = item.get_group(1);
  const int global_chunk_range = item.get_group_range(1);

  // l2norm for q, k
  int group_id = item.get_group(1);
  int group_range = item.get_group_range(1);
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_range = sg.get_group_linear_range();
  int sg_local_id = sg.get_local_linear_id();

  float* slm_mem = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());

  float* A_ptr_load = slm_mem;
  float* A_ptr_save = A_ptr_load + chunk_size * chunk_size;

  int pre_chunks = 0;

  const int kv_ratio = num_v_heads / num_k_heads;

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int chunk_start_offset = chunk_id * chunk_size;

      for (int v_head_id = 0; v_head_id < num_v_heads; ++v_head_id) {
        auto A_ptr = A +
                     static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                         chunk_size +
                     chunk_start_offset * chunk_size;

        CUTE_UNROLL
        for (int m_idx = sg_id; m_idx < chunk_size; m_idx += sg_range) {
          CUTE_UNROLL
          for (int n_idx = sg_local_id; n_idx < m_idx;
               n_idx += sub_group_size) {
            A_ptr_load[m_idx * chunk_size + n_idx] =
                A_ptr[m_idx * chunk_size + n_idx];
          }
        }

        CUTE_UNROLL
        for (int idx = local_id; idx < chunk_size; idx += local_range) {
          A_ptr_save[idx * chunk_size + idx] = 1.0f;
        }
        sycl::group_barrier(item.get_group());

        // inverse
        CUTE_UNROLL
        for (int n_idx = local_id; n_idx < chunk_size; n_idx += local_range) {
          CUTE_UNROLL
          for (int m_idx = n_idx + 1; m_idx < chunk_size; ++m_idx) {
            float sum =
                static_cast<float>(A_ptr_load[m_idx * chunk_size + n_idx]);
            CUTE_UNROLL
            for (int loop_idx = n_idx + 1; loop_idx < m_idx; ++loop_idx) {
              sum +=
                  static_cast<float>(
                      A_ptr_save[loop_idx * chunk_size + n_idx]) *
                  static_cast<float>(A_ptr_load[m_idx * chunk_size + loop_idx]);
            }
            A_ptr_save[m_idx * chunk_size + n_idx] = -sum;
          }
        }

        sycl::group_barrier(item.get_group());

        CUTE_UNROLL
        for (int m_idx = sg_id; m_idx < chunk_size; m_idx += sg_range) {
          CUTE_UNROLL
          for (int n_idx = sg_local_id; n_idx <= m_idx;
               n_idx += sub_group_size) {
            A_ptr[m_idx * chunk_size + n_idx] =
                static_cast<T>(A_ptr_save[m_idx * chunk_size + n_idx]);
          }
        }
      }
      chunk_id += global_chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_inverse_opt_kernel(
    T* A,
    const int* query_start_loc,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  int local_range = item.get_local_range(2);
  int v_head_id = item.get_group(1) % num_v_heads;
  int chunk_id = item.get_group(1) / num_v_heads;
  const int global_chunk_range = item.get_group_range(1) / num_v_heads;

  // l2norm for q, k
  int group_id = item.get_group(1);
  int group_range = item.get_group_range(1);
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_range = sg.get_group_linear_range();
  int sg_local_id = sg.get_local_linear_id();

  int pre_chunks = 0;

  const int kv_ratio = num_v_heads / num_k_heads;

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int chunk_start_offset = chunk_id * chunk_size;

      auto A_ptr =
          A +
          static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size +
          chunk_start_offset * chunk_size;

      CUTE_UNROLL
      for (int i = 0; i < 4; ++i) {
        int offset = i * 16;
        T* A_ptr_xx = A_ptr + offset * chunk_size + offset;
        float A_local[16];
        float A_other[16];
        float A_sum;
        CUTE_UNROLL
        for (int e = 0; e < sg_local_id + 1; ++e) {
          A_local[e] = 0.0f;
        }

        T A_load[16];
        CUTE_UNROLL
        for (int e = 0; e < sg_local_id; ++e) {
          A_load[e] = A_ptr_xx[sg_local_id * chunk_size + e];
        }

        CUTE_UNROLL
        for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
          CUTE_UNROLL
          for (int nn_idx = 0; nn_idx < mm_idx; ++nn_idx) {
            float send_value = static_cast<float>(A_load[nn_idx]);
            float receive_value = sycl::group_broadcast(sg, send_value, mm_idx);
            if (sg_local_id == nn_idx) {
              A_local[mm_idx] = receive_value;
            }
          }
        }

        CUTE_UNROLL
        for (int mm_idx = 1; mm_idx < 16; ++mm_idx) {
          A_sum = 0.0f;
          CUTE_UNROLL
          for (int e = 1; e < mm_idx + 1; ++e) {
            A_other[e] = sycl::group_broadcast(sg, A_local[mm_idx], e);
          }

          CUTE_UNROLL
          for (int e = 1; e < mm_idx + 1; ++e) {
            A_sum += A_local[e] * A_other[e];
          }

          A_local[mm_idx] = -A_local[mm_idx] - A_sum;
        }

        CUTE_UNROLL
        for (int e = sg_local_id + 1; e < 16; ++e) {
          A_ptr_xx[e * chunk_size + sg_local_id] = static_cast<T>(A_local[e]);
        }
      }

      auto A_ptr_11 = A_ptr;

      auto A_ptr_21 = A_ptr + 16 * chunk_size;
      auto A_ptr_22 = A_ptr + 16 * chunk_size + 16;

      auto A_ptr_31 = A_ptr + 32 * chunk_size;
      auto A_ptr_32 = A_ptr + 32 * chunk_size + 16;
      auto A_ptr_33 = A_ptr + 32 * chunk_size + 32;

      auto A_ptr_41 = A_ptr + 48 * chunk_size;
      auto A_ptr_42 = A_ptr + 48 * chunk_size + 16;
      auto A_ptr_43 = A_ptr + 48 * chunk_size + 32;
      auto A_ptr_44 = A_ptr + 48 * chunk_size + 48;

      auto A_XX_tensor_shape = make_shape(16, 16);

      auto A_11_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_11),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

      auto A_21_tensor = make_tensor(
          make_gmem_ptr(A_ptr_21),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_21_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_21),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_22_tensor = make_tensor(
          make_gmem_ptr(A_ptr_22),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_22_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_22),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

      auto A_31_tensor = make_tensor(
          make_gmem_ptr(A_ptr_31),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_31_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_31),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_32_tensor = make_tensor(
          make_gmem_ptr(A_ptr_32),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_32_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_32),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_33_tensor = make_tensor(
          make_gmem_ptr(A_ptr_33),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_33_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_33),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));

      auto A_41_tensor = make_tensor(
          make_gmem_ptr(A_ptr_41),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_41_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_41),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_42_tensor = make_tensor(
          make_gmem_ptr(A_ptr_42),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_42_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_42),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_43_tensor = make_tensor(
          make_gmem_ptr(A_ptr_43),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));
      auto A_43_tensor_T = make_tensor(
          make_gmem_ptr(A_ptr_43),
          make_layout(A_XX_tensor_shape, make_stride(_1{}, chunk_size)));
      auto A_44_tensor = make_tensor(
          make_gmem_ptr(A_ptr_44),
          make_layout(A_XX_tensor_shape, make_stride(chunk_size, _1{})));

      Tensor cA = make_identity_tensor(A_XX_tensor_shape);
      Tensor cB = make_identity_tensor(A_XX_tensor_shape);
      Tensor cC = make_identity_tensor(A_XX_tensor_shape);
      Tensor gA = local_tile(cA, select<0, 2>(wg_tile), make_coord(0, _));
      Tensor gB = local_tile(cB, select<1, 2>(wg_tile), make_coord(0, _));
      Tensor gC =
          local_tile(cC, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
      auto tCrA = thr_mma.partition_sg_fragment_A(gA(_, _, 0));
      auto tCrB = thr_mma.partition_sg_fragment_B(gB(_, _, 0));
      auto tCrC = thr_mma.partition_sg_fragment_C(gC);

      auto copy_D_21 = get_block_2d_copy_D<void>(mma, A_21_tensor);
      auto thr_copy_D_21 = copy_D_21.get_slice(local_id);
      auto tCrD_21 = thr_copy_D_21.partition_sg_fragment_S(gC);
      auto tCgD_21 = thr_copy_D_21.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_22_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrA);
      clear(tCrC);
      gemm_STS(tCrA, A_11_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_21);
      copy(copy_D_21, tCrD_21, tCgD_21);

      auto copy_D_31 = get_block_2d_copy_D<void>(mma, A_31_tensor);
      auto thr_copy_D_31 = copy_D_31.get_slice(local_id);
      auto tCrD_31 = thr_copy_D_31.partition_sg_fragment_S(gC);
      auto tCgD_31 = thr_copy_D_31.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_31_tensor, A_11_tensor_T, tCrC, 0, 0, mma);
      gemm_TTS(A_32_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrD_31);
      copy(copy_D_31, tCrD_31, tCgD_31);
      clear(tCrC);
      gemm_TTS(A_33_tensor, A_31_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_31);
      copy(copy_D_31, tCrD_31, tCgD_31);

      auto copy_D_41 = get_block_2d_copy_D<void>(mma, A_41_tensor);
      auto thr_copy_D_41 = copy_D_41.get_slice(local_id);
      auto tCrD_41 = thr_copy_D_41.partition_sg_fragment_S(gC);
      auto tCgD_41 = thr_copy_D_41.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_41_tensor, A_11_tensor_T, tCrC, 0, 0, mma);
      gemm_TTS(A_42_tensor, A_21_tensor_T, tCrC, 0, 0, mma);
      gemm_TTS(A_43_tensor, A_31_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrD_41);
      copy(copy_D_41, tCrD_41, tCgD_41);
      clear(tCrC);
      gemm_TTS(A_44_tensor, A_41_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_41);
      copy(copy_D_41, tCrD_41, tCgD_41);

      auto copy_D_32 = get_block_2d_copy_D<void>(mma, A_32_tensor);
      auto thr_copy_D_32 = copy_D_32.get_slice(local_id);
      auto tCrD_32 = thr_copy_D_32.partition_sg_fragment_S(gC);
      auto tCgD_32 = thr_copy_D_32.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_33_tensor, A_32_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrA);
      clear(tCrC);
      gemm_STS(tCrA, A_22_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_32);
      copy(copy_D_32, tCrD_32, tCgD_32);

      auto copy_D_42 = get_block_2d_copy_D<void>(mma, A_42_tensor);
      auto thr_copy_D_42 = copy_D_42.get_slice(local_id);
      auto tCrD_42 = thr_copy_D_42.partition_sg_fragment_S(gC);
      auto tCgD_42 = thr_copy_D_42.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_42_tensor, A_22_tensor_T, tCrC, 0, 0, mma);
      gemm_TTS(A_43_tensor, A_32_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrD_42);
      copy(copy_D_42, tCrD_42, tCgD_42);
      clear(tCrC);
      gemm_TTS(A_44_tensor, A_42_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_42);
      copy(copy_D_42, tCrD_42, tCgD_42);

      auto copy_D_43 = get_block_2d_copy_D<void>(mma, A_43_tensor);
      auto thr_copy_D_43 = copy_D_43.get_slice(local_id);
      auto tCrD_43 = thr_copy_D_43.partition_sg_fragment_S(gC);
      auto tCgD_43 = thr_copy_D_43.partition_D(gC);
      clear(tCrC);
      gemm_TTS(A_44_tensor, A_43_tensor_T, tCrC, 0, 0, mma);
      reorder(tCrC, tCrA);
      clear(tCrC);
      gemm_STS(tCrA, A_33_tensor_T, tCrC, 0, 0, mma);
      CUTE_UNROLL
      for (int i = 0; i < tCrC.size(); ++i) {
        tCrC(i) *= -1.0f;
      }
      reorder(tCrC, tCrD_43);
      copy(copy_D_43, tCrD_43, tCgD_43);

      chunk_id += global_chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

template <typename T, class TiledMMA>
CUTE_DEVICE void chunk_compute_wu_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,
    T* A,
    T* w,
    T* u,
    const T* q,
    const T* k,
    const T* v,
    const float* b,
    const float* a,
    const float* A_log,
    const T* dt_bias,
    const int* query_start_loc,
    const bool* has_initial_state,
    const int total_virtual_seqlen,
    const int batch_size,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  int local_range = item.get_local_range(2);
  int chunk_id = item.get_group(1);
  const int global_chunk_range = item.get_group_range(1);

  // l2norm for q, k
  int group_id = item.get_group(1);
  int group_range = item.get_group_range(1);
  auto sg = item.get_sub_group();
  int sg_id = sg.get_group_linear_id();
  int sg_range = sg.get_group_linear_range();
  int sg_local_id = sg.get_local_linear_id();

  float* slm_mem = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());
  float* A_log_slm_ptr = slm_mem;
  float* dt_bias_slm_ptr = A_log_slm_ptr + num_v_heads;
  float* g_slm_ptr = dt_bias_slm_ptr + num_v_heads;
  float* beta_slm_ptr = g_slm_ptr + chunk_size;

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();
  auto thr_mma = mma.get_slice(local_id);

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;

  auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int m_tile_start = 0;
  int n_tile_start = 0;
  int m_sg_start = sg_local_m_coord * SG_M;
  int n_sg_start = sg_local_n_coord * SG_N;

  using TileShape = decltype(mma.tile_mnk());

  int pre_chunks = 0;

  const int kv_ratio = num_v_heads / num_k_heads;

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const bool initial_state = has_initial_state[batch_id];
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;
    const int cumsum_chunks = pre_chunks + current_chunks;

    if (chunk_id >= cumsum_chunks) {
      pre_chunks = cumsum_chunks;
      continue;
    }

    while (chunk_id < cumsum_chunks) {
      const int chunk_start_offset = chunk_id * chunk_size;

      for (int v_head_id = 0; v_head_id < num_v_heads; ++v_head_id) {
        // WAR fence: previous iteration's sub-groups may still be
        // reading beta_slm_ptr/g_slm_ptr; all must arrive before any
        // sub-group refills the SLM for this iteration.
        sycl::group_barrier(item.get_group());
        CUTE_UNROLL
        for (int e = local_id; e < chunk_size; e += local_range) {
          float beta_value =
              b[(chunk_start_offset + e) + v_head_id * total_virtual_seqlen];
          float a_value =
              a[(chunk_start_offset + e) + v_head_id * total_virtual_seqlen];
          beta_slm_ptr[e] = beta_value;
          g_slm_ptr[e] = sycl::exp(a_value) * beta_value;
        }

        sycl::group_barrier(item.get_group());

        auto A_ptr = A +
                     static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                         chunk_size +
                     chunk_start_offset * chunk_size;
        auto A_tensor_shape = make_shape(chunk_size, chunk_size);
        auto A_tensor = make_tensor(
            make_gmem_ptr(A_ptr),
            make_layout(A_tensor_shape, make_stride(chunk_size, _1{})));

        auto v_ptr = v +
                     static_cast<int64_t>(chunk_start_offset) * num_v_heads *
                         head_v_dim +
                     v_head_id * head_v_dim;
        auto V_tensor_T_shape = make_shape(head_v_dim, chunk_size);
        auto V_tensor_T = make_tensor(
            make_gmem_ptr(v_ptr),
            make_layout(
                V_tensor_T_shape, make_stride(_1{}, head_v_dim * num_v_heads)));
        auto U_ptr = u +
                     static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                         head_v_dim +
                     chunk_start_offset * head_v_dim;
        auto U_tensor_shape = make_shape(chunk_size, head_v_dim);
        auto U_tensor = make_tensor(
            make_gmem_ptr(U_ptr),
            make_layout(U_tensor_shape, make_stride(head_v_dim, _1{})));

        Tensor cU = make_identity_tensor(U_tensor.shape());
        auto copy_U_c = get_block_2d_copy_D<void>(mma, U_tensor);
        auto thr_copy_U_c = copy_U_c.get_slice(local_id);

        for (int dv = 0; dv < head_v_dim / chunk_size; ++dv) {
          Tensor gU_C =
              local_tile(cU, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});
          auto tCrU_c = thr_copy_U_c.partition_sg_fragment_S(gU_C);
          auto tCgU_c = thr_copy_U_c.partition_D(gU_C);
          auto tSrU_c = thr_mma.partition_sg_fragment_C(gU_C);
          clear(tSrU_c);
          gemm_TTS_k_multi(
              A_tensor, V_tensor_T, tSrU_c, 0, dv, mma, beta_slm_ptr);
          reorder(tSrU_c, tCrU_c);
          copy(copy_U_c, tCrU_c, tCgU_c);
        }

        if (((chunk_id - pre_chunks) != 0) || initial_state) {
          auto k_ptr = k +
                       static_cast<int64_t>(chunk_start_offset) * num_k_heads *
                           head_k_dim +
                       (v_head_id / kv_ratio) * head_k_dim;
          auto K_tensor_T_shape = make_shape(head_k_dim, chunk_size);
          auto K_tensor_T = make_tensor(
              make_gmem_ptr(k_ptr),
              make_layout(
                  K_tensor_T_shape,
                  make_stride(_1{}, head_k_dim * num_k_heads)));
          auto W_ptr = w +
                       static_cast<int64_t>(v_head_id) * total_virtual_seqlen *
                           head_k_dim +
                       chunk_start_offset * head_k_dim;
          auto W_tensor_shape = make_shape(chunk_size, head_k_dim);
          auto W_tensor = make_tensor(
              make_gmem_ptr(W_ptr),
              make_layout(W_tensor_shape, make_stride(head_k_dim, _1{})));

          Tensor cW = make_identity_tensor(W_tensor.shape());
          auto copy_W_c = get_block_2d_copy_D<void>(mma, W_tensor);
          auto thr_copy_W_c = copy_W_c.get_slice(local_id);

          for (int dk = 0; dk < head_k_dim / chunk_size; ++dk) {
            Tensor gW_C = local_tile(
                cW, wg_tile, make_coord(0, dk, 0), Step<_1, _1, X>{});
            auto tCrW_c = thr_copy_W_c.partition_sg_fragment_S(gW_C);
            auto tCgW_c = thr_copy_W_c.partition_D(gW_C);
            auto tSrW_c = thr_mma.partition_sg_fragment_C(gW_C);
            clear(tSrW_c);
            gemm_TTS_k_multi(
                A_tensor, K_tensor_T, tSrW_c, 0, dk, mma, g_slm_ptr);
            reorder(tSrW_c, tCrW_c);
            copy(copy_W_c, tCrW_c, tCgW_c);
          }
        }
      }
      chunk_id += global_chunk_range;
    }
    pre_chunks = cumsum_chunks;
  }
}

template <typename T, typename StateT, class TiledMMA>
CUTE_DEVICE void chunk_fwd_o_kernel(
    const sycl::local_accessor<float, 1>& slm_mem_const,  // [3 * chunk_size]
    T* core_attn_out,  // [total_seqlen, num_v_heads, head_v_dim]
    T* A,  // [num_v_heads, total_virtual_seqlen, chunk_size], temp O2 buffer
    T* w,  // [num_v_heads, total_virtual_seqlen, head_k_dim]
    T* u,  // [num_v_heads, total_virtual_seqlen, head_v_dim]
    const T* q,  // [total_virtual_seqlen, num_k_heads, head_k_dim]
    const T* k,  // [total_virtual_seqlen, num_k_heads, head_k_dim]
    const float* a,
    StateT*
        ssm_state,  // [cache_batch_size, num_v_heads, head_v_dim, head_k_dim]
    const int ssm_state_stride_0,
    const int* query_start_loc,
    const int* cache_indices,
    const bool* has_initial_state,
    const int* token_indx,
    const int batch_size,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  auto item = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
  int local_id = item.get_local_linear_id();
  int current_batch_id = item.get_group(0);
  int v_head_id = item.get_group(1);
  int local_range = item.get_local_range(2);

  auto sg = item.get_sub_group();
  int sg_local_id = sg.get_local_linear_id();

  float* slm_mem = static_cast<float*>(
      slm_mem_const.template get_multi_ptr<sycl::access::decorated::no>()
          .get());
  float* g_slm_ptr = slm_mem;
  float* g_multi_slm_ptr = slm_mem + chunk_size;
  float* g_exp_slm_ptr = g_multi_slm_ptr + chunk_size;

  TiledMMA mma{};
  auto wg_tile = mma.tile_mnk();

  static constexpr auto tile_m = get<0>(wg_tile);
  static constexpr auto tile_n = get<1>(wg_tile);

  static constexpr auto ATOM_M =
      get<1>(typename TiledMMA::ThrLayoutVMNK{}.shape());
  static constexpr auto ATOM_N =
      get<2>(typename TiledMMA::ThrLayoutVMNK{}.shape());

  static constexpr auto SG_M = tile_m / ATOM_M;  // BLK_M / ATOM_M;
  static constexpr auto SG_N = tile_n / ATOM_N;  // BLK_N / ATOM_N;

  auto sg_local_m_coord = cutlass::get_sub_group_id() / ATOM_N;
  auto sg_local_n_coord = cutlass::get_sub_group_id() % ATOM_N;
  int m_tile_start = 0;
  int n_tile_start = 0;
  int m_sg_start = sg_local_m_coord * SG_M;
  int n_sg_start = sg_local_n_coord * SG_N;

  const int kv_ratio = num_v_heads / num_k_heads;
  const int kv_head_id = v_head_id / kv_ratio;

  int pre_chunks = 0;

  for (int batch_id = 0; batch_id < batch_size; ++batch_id) {
    const bool initial_state = has_initial_state[batch_id];
    const int seq_start_offset = query_start_loc[batch_id];
    const int seq_end_offset = query_start_loc[batch_id + 1];
    const int seq_len = seq_end_offset - seq_start_offset;

    const int current_chunks = (seq_len + chunk_size - 1) / chunk_size;

    if (current_batch_id != batch_id) {
      pre_chunks += current_chunks;
      continue;
    }

    StateT* ssm_state_ptr =
        ssm_state +
        static_cast<int64_t>(cache_indices[batch_id]) * ssm_state_stride_0 +
        v_head_id * head_v_dim * head_k_dim;

    for (int chunk_id = 0; chunk_id < current_chunks; ++chunk_id) {
      const bool has_prev_state = (chunk_id != 0) || initial_state;
      const int out_chunk_offset = seq_start_offset + chunk_id * chunk_size;
      const int chunk_offset = (pre_chunks + chunk_id) * chunk_size;

      int current_chunk_size = chunk_size;
      if ((chunk_id + 1) * chunk_size > seq_len) {
        current_chunk_size = seq_len - chunk_id * chunk_size;
      }

      float g_last_value =
          a[(chunk_offset + current_chunk_size - 1) +
            v_head_id * total_virtual_seqlen];
      float g_last_value_exp = sycl::native::exp(g_last_value);
      // WAR fence: the previous chunk iteration's sub-groups may
      // still be reading g_slm_ptr/g_multi_slm_ptr/g_exp_slm_ptr in
      // their output/state GEMM epilogues; all must arrive before any
      // sub-group refills these SLM buffers for this chunk.
      sycl::group_barrier(item.get_group());
      CUTE_UNROLL
      for (int e = local_id; e < current_chunk_size; e += local_range) {
        float g_cumsum_value =
            a[(chunk_offset + e) + v_head_id * total_virtual_seqlen];
        g_slm_ptr[e] = g_cumsum_value;
        g_multi_slm_ptr[e] = sycl::native::exp(g_last_value - g_cumsum_value);
        g_exp_slm_ptr[e] = sycl::native::exp(g_cumsum_value);
      }

      CUTE_UNROLL
      for (int e = current_chunk_size + local_id; e < chunk_size;
           e += local_range) {
        g_slm_ptr[e] = 0.0f;
        g_multi_slm_ptr[e] = 0.0f;
        g_exp_slm_ptr[e] = 0.0f;
      }
      sycl::group_barrier(item.get_group());

      auto W_ptr = w + v_head_id * total_virtual_seqlen * head_k_dim +
                   chunk_offset * head_k_dim;
      auto W_tensor_shape = make_shape(chunk_size, head_k_dim);
      auto W_tensor = make_tensor(
          make_gmem_ptr(W_ptr),
          make_layout(W_tensor_shape, make_stride(head_k_dim, _1{})));

      auto U_ptr = u + v_head_id * total_virtual_seqlen * head_v_dim +
                   chunk_offset * head_v_dim;
      auto U_tensor_shape = make_shape(chunk_size, head_v_dim);
      auto U_tensor = make_tensor(
          make_gmem_ptr(U_ptr),
          make_layout(U_tensor_shape, make_stride(head_v_dim, _1{})));

      StateT* S_ptr = ssm_state_ptr;
      auto S_tensor_shape = make_shape(head_v_dim, head_k_dim);
      auto S_tensor = make_tensor(
          make_gmem_ptr(S_ptr),
          make_layout(S_tensor_shape, make_stride(head_k_dim, _1{})));

      Tensor cU = make_identity_tensor(U_tensor.shape());

      auto copy_U_c = get_block_2d_copy_C<void>(mma, U_tensor);
      auto copy_U_d = get_block_2d_copy_D<void>(mma, U_tensor);

      auto thr_copy_U_c = copy_U_c.get_slice(local_id);
      auto thr_copy_U_d = copy_U_d.get_slice(local_id);

      auto thr_mma = mma.get_slice(local_id);

      auto q_ptr =
          q + chunk_offset * num_k_heads * head_k_dim + kv_head_id * head_k_dim;
      auto Q_tensor_shape = make_shape(current_chunk_size, head_k_dim);
      auto Q_tensor = make_tensor(
          make_gmem_ptr(q_ptr),
          make_layout(
              Q_tensor_shape, make_stride(head_k_dim * num_k_heads, _1{})));

      auto k_ptr =
          k + chunk_offset * num_k_heads * head_k_dim + kv_head_id * head_k_dim;
      auto K_tensor_shape = make_shape(chunk_size, head_k_dim);
      auto K_tensor = make_tensor(
          make_gmem_ptr(k_ptr),
          make_layout(
              K_tensor_shape, make_stride(head_k_dim * num_k_heads, _1{})));

      auto O2_ptr =
          A +
          static_cast<int64_t>(v_head_id) * total_virtual_seqlen * chunk_size +
          chunk_offset * chunk_size;
      auto O2_tensor_shape = make_shape(current_chunk_size, chunk_size);
      auto O2_tensor = make_tensor(
          make_gmem_ptr(O2_ptr),
          make_layout(O2_tensor_shape, make_stride(chunk_size, _1{})));
      Tensor cO2 = make_identity_tensor(O2_tensor_shape);
      auto copy_O2_c = get_block_2d_copy_D<void>(mma, O2_tensor);
      auto thr_copy_O2_c = copy_O2_c.get_slice(local_id);

      Tensor gO2_C =
          local_tile(cO2, wg_tile, make_coord(0, 0, 0), Step<_1, _1, X>{});
      auto tCrO2_c = thr_copy_O2_c.partition_sg_fragment_S(gO2_C);
      auto tCgO2_c = thr_copy_O2_c.partition_D(gO2_C);
      auto tSrO2_c = thr_mma.partition_sg_fragment_C(gO2_C);

      // QK MMA: compute O2 = mask(Q × K^T) first (no dependency on WS)
      clear(tSrO2_c);
      gemm_TTS(Q_tensor, K_tensor, tSrO2_c, 0, 0, mma);

      CUTE_UNROLL
      for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
        int n_idx =
            n_tile_start + n_sg_start + sn * sub_group_size + sg_local_id;
        CUTE_UNROLL
        for (int sm = 0; sm < SG_M; ++sm) {
          int m_idx = m_tile_start + m_sg_start + sm;
          tSrO2_c(sn * SG_M + sm) *=
              sycl::native::exp(g_slm_ptr[(m_idx)] - g_slm_ptr[n_idx]);
          if (m_idx < n_idx) {
            tSrO2_c(sn * SG_M + sm) = 0.0f;
          }
        }
      }
      reorder(tSrO2_c, tCrO2_c);
      copy(copy_O2_c, tCrO2_c, tCgO2_c);

      auto U_tensor_T_shape = make_shape(head_v_dim, chunk_size);
      auto U_tensor_T = make_tensor(
          make_gmem_ptr(U_ptr),
          make_layout(U_tensor_T_shape, make_stride(_1{}, head_v_dim)));
      auto O_ptr =
          core_attn_out +
          (token_indx ? token_indx[out_chunk_offset] : out_chunk_offset) *
              num_v_heads * head_v_dim +
          v_head_id * head_v_dim;
      auto O_tensor_shape = make_shape(current_chunk_size, head_v_dim);
      auto O_tensor = make_tensor(
          make_gmem_ptr(O_ptr),
          make_layout(
              O_tensor_shape, make_stride(num_v_heads * head_v_dim, _1{})));

      Tensor cO = make_identity_tensor(O_tensor.shape());
      auto copy_O_c = get_block_2d_copy_D<void>(mma, O_tensor);
      auto thr_copy_O_c = copy_O_c.get_slice(local_id);

      if (has_prev_state) {
        // Fused WS+QS dv loop: S[dv] is loaded once per k_tile and reused
        // in registers for both WS (W×S) and QS (Q×S) DPAS operations.
        for (int dv = 0; dv < head_v_dim / chunk_size; ++dv) {
          // --- Fused WS+QS MMA: share S[dv] load in registers ---
          Tensor gU_C =
              local_tile(cU, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});
          auto tSrU_d = thr_mma.partition_sg_fragment_C(gU_C);
          clear(tSrU_d);

          Tensor gO_C =
              local_tile(cO, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});
          auto tSrO_c = thr_mma.partition_sg_fragment_C(gO_C);
          clear(tSrO_c);

          // Fused gemm: W×S[dv] -> tSrU_d, Q×S[dv] -> tSrO_c
          // S is loaded once and reused in registers for both
          gemm_TTS_fused_2A(
              W_tensor, Q_tensor, S_tensor, tSrU_d, tSrO_c, 0, 0, dv, mma);

          // --- WS epilogue: U_new[dv] = U_old[dv] - W×S[dv] ---
          auto tCrU_d = thr_copy_U_d.partition_sg_fragment_S(gU_C);
          auto tCgU_d = thr_copy_U_d.partition_D(gU_C);
          auto tCrU_c_save = thr_copy_U_c.partition_sg_fragment_D(gU_C);
          reorder(tSrU_d, tCrU_c_save);

          auto tCgU_c = thr_copy_U_c.partition_S(gU_C);
          auto tCrU_c = thr_copy_U_c.partition_sg_fragment_D(gU_C);
          copy(copy_U_c, tCgU_c, tCrU_c);

          CUTE_UNROLL
          for (int i = 0; i < tCrU_c_save.size(); ++i) {
            tCrU_c(i) -= tCrU_c_save(i);
          }

          reorder(tCrU_c, tCrU_d);
          copy(copy_U_d, tCrU_d, tCgU_d);

          // --- QS epilogue: apply exp(g) scaling ---
          CUTE_UNROLL
          for (int sn = 0; sn < SG_N / sub_group_size; ++sn) {
            int n_idx =
                n_tile_start + n_sg_start + sn * sub_group_size + sg_local_id;
            CUTE_UNROLL
            for (int sm = 0; sm < SG_M; ++sm) {
              int m_idx = m_tile_start + m_sg_start + sm;
              tSrO_c(sn * SG_M + sm) *= g_exp_slm_ptr[(m_idx)];
            }
          }

          // Barrier: ensure U_new[dv] is visible across all subgroups
          // before O2U reads it via U_tensor_T.
          sycl::group_barrier(item.get_group());

          // --- O2U MMA: O[dv] += O1[dv] + O2 × U_T[dv] ---
          gemm_TTS(O2_tensor, U_tensor_T, tSrO_c, 0, dv, mma);
          auto tCrO_c = thr_copy_O_c.partition_sg_fragment_S(gO_C);
          auto tCgO_c = thr_copy_O_c.partition_D(gO_C);
          reorder(tSrO_c, tCrO_c);
          copy(copy_O_c, tCrO_c, tCgO_c);
        }
      }

      auto K_tensor_T_shape = make_shape(head_k_dim, chunk_size);
      auto K_tensor_T = make_tensor(
          make_gmem_ptr(k_ptr),
          make_layout(
              K_tensor_T_shape, make_stride(_1{}, head_k_dim * num_k_heads)));

      Tensor cS = make_identity_tensor(S_tensor.shape());
      auto copy_S_c = get_block_2d_copy_C<void>(mma, S_tensor);
      auto copy_S_d = get_block_2d_copy_D<void>(mma, S_tensor);
      auto thr_copy_S_c = copy_S_c.get_slice(local_id);
      auto thr_copy_S_d = copy_S_d.get_slice(local_id);

      for (int dv = 0; dv < head_v_dim / chunk_size; ++dv) {
        for (int dk = 0; dk < head_k_dim / chunk_size; ++dk) {
          Tensor gS_C =
              local_tile(cS, wg_tile, make_coord(dv, dk, 0), Step<_1, _1, X>{});
          auto tCrS_d = thr_copy_S_d.partition_sg_fragment_S(gS_C);
          auto tCgS_d = thr_copy_S_d.partition_D(gS_C);
          auto tSrS_d = thr_mma.partition_sg_fragment_C(gS_C);

          // Seed accumulator with exp(g_last) * S_prev when previous state
          // exists; otherwise start from zeros.
          if (has_prev_state) {
            auto tCgS_c = thr_copy_S_c.partition_S(gS_C);
            auto tCrS_c = thr_copy_S_c.partition_sg_fragment_D(gS_C);
            copy(copy_S_c, tCgS_c, tCrS_c);

            reorder(tCrS_c, tSrS_d);
            CUTE_UNROLL
            for (int i = 0; i < tCrS_c.size(); ++i) {
              tSrS_d(i) *= g_last_value_exp;
            }
          } else {
            clear(tSrS_d);
          }

          gemm_TTS_k_multi(
              U_tensor_T, K_tensor_T, tSrS_d, dv, dk, mma, g_multi_slm_ptr);
          reorder(tSrS_d, tCrS_d);
          copy(copy_S_d, tCrS_d, tCgS_d);
        }
      }

      if (!has_prev_state) {
        for (int dv = 0; dv < head_v_dim / chunk_size; ++dv) {
          Tensor gO_C =
              local_tile(cO, wg_tile, make_coord(0, dv, 0), Step<_1, _1, X>{});
          auto tCrO_c = thr_copy_O_c.partition_sg_fragment_S(gO_C);
          auto tCgO_c = thr_copy_O_c.partition_D(gO_C);
          auto tSrO_c = thr_mma.partition_sg_fragment_C(gO_C);

          clear(tSrO_c);
          gemm_TTS(O2_tensor, U_tensor_T, tSrO_c, 0, dv, mma);
          reorder(tSrO_c, tCrO_c);
          copy(copy_O_c, tCrO_c, tCgO_c);
        }
      }
    }
    pre_chunks += current_chunks;
  }
}

template <typename T, typename StateTag>
class ChunkPrepareKernel;

template <typename T, typename StateTag>
class ChunkComputeAKernel;

template <typename T, typename StateTag>
class ChunkInverseOptKernel;

template <typename T, typename StateTag>
class ChunkInverseKernel;

template <typename T, typename StateTag>
class ChunkComputeWUKernel;

template <typename T, typename StateT>
class ChunkFwdOKernel;

template <typename T, typename StateT>
void kernel_launcher(
    sycl::queue& queue,
    T* core_attn_out,
    const T* q,
    const T* k,
    const T* v,
    T* A,
    T* w,
    T* u,
    float* b,
    float* a,
    const float* g_token_major,
    const float* beta_token_major,
    StateT* ssm_state,
    const int ssm_state_stride_0,
    const int* query_start_loc,
    const int* cache_indices,
    const bool* has_initial_state,
    const int* token_indx,
    const int batch_size,
    const int total_virtual_seqlen,
    const int num_k_heads,
    const int head_k_dim,
    const int num_v_heads,
    const int head_v_dim) {
  using Element_non_CV = cutlass::platform::remove_cv_t<T>;
  auto op = XE_DPAS_TT<8, float, Element_non_CV>{};

  int sm_count =
      cutlass::KernelHardwareInfo::query_device_multiprocessor_count(0);
  static constexpr int MaxThreadsPerSM = 512;

  namespace syclex = sycl::ext::oneapi::experimental;
  namespace intelex = sycl::ext::intel::experimental;

  syclex::properties kernel_props{
      syclex::sub_group_size<cute::detail::subgroup_size>,
      intelex::grf_size<256>};

  // prepare data for A, W, U compute
  sycl::range<3> local_prepare(1, 1, MaxThreadsPerSM);
  sycl::range<3> global_prepare(1, sm_count, 1);
  int slm_size_prepare = num_v_heads * 2 + chunk_size;

  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for<ChunkPrepareKernel<T, StateT>>(
        sycl::nd_range<3>{global_prepare * local_prepare, local_prepare},
        kernel_props,
        [=](auto) {
          chunk_prepare_kernel<T>(
              a,
              b,
              g_token_major,
              beta_token_major,
              query_start_loc,
              total_virtual_seqlen,
              batch_size,
              num_v_heads,
              head_v_dim);
        });
  });

  // compute A
  using WGTileComputeA = chunk_gemm_policy_compute_A::WGTile;
  using SGLayoutComputeA = chunk_gemm_policy_compute_A::SGLayout;
  using MMAComputeA = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTileComputeA>,
      SGLayoutComputeA>::TiledMMA;
  auto mmaComputeA = MMAComputeA{};
  int MaxThreadsPerWorkgroupComputeA = size(mmaComputeA);
  sycl::range<3> local_compute_A(1, 1, MaxThreadsPerWorkgroupComputeA);
  sycl::range<3> global_compute_A(
      1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroupComputeA, 1);
  int slm_size_compute_A = chunk_size;

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_mem(
        sycl::range<1>(slm_size_compute_A), cgh);
    cgh.parallel_for<ChunkComputeAKernel<T, StateT>>(
        sycl::nd_range<3>{global_compute_A * local_compute_A, local_compute_A},
        kernel_props,
        [=](auto) {
          chunk_compute_A_kernel<T, MMAComputeA>(
              local_mem,
              A,
              k,
              v,
              b,
              a,
              query_start_loc,
              total_virtual_seqlen,
              batch_size,
              num_k_heads,
              head_k_dim,
              num_v_heads,
              head_v_dim);
        });
  });

  if (true) {  // This adapter is gated to the B70 Xe2 device.
    using WGTileInverse = chunk_gemm_policy_inverse::WGTile;
    using SGLayoutInverse = chunk_gemm_policy_inverse::SGLayout;
    using MMAInverse = typename TiledMMAHelper<
        MMA_Atom<decltype(op)>,
        Layout<WGTileInverse>,
        SGLayoutInverse>::TiledMMA;
    auto mma_inverse = MMAInverse{};
    int MaxThreadsPerWorkgroupInverse = size(mma_inverse);

    sycl::range<3> local_inverse(1, 1, MaxThreadsPerWorkgroupInverse);
    sycl::range<3> global_inverse(
        1,
        (sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroupInverse +
         num_v_heads - 1) /
            num_v_heads * num_v_heads,
        1);

    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for<ChunkInverseOptKernel<T, StateT>>(
          sycl::nd_range<3>{global_inverse * local_inverse, local_inverse},
          kernel_props,
          [=](auto) {
            chunk_inverse_opt_kernel<T, MMAInverse>(
                A,
                query_start_loc,
                total_virtual_seqlen,
                batch_size,
                num_k_heads,
                head_k_dim,
                num_v_heads,
                head_v_dim);
          });
    });
  } else {
    // PVC has acc issue of sycl tla, so use native implementation for inverse
    // Once issue is fixed, remove this workaround and use the same MMA-based
    // kernel as BMG
    int inverse_items = 16 * 2;
    sycl::range<3> local_inverse(1, 1, inverse_items);
    sycl::range<3> global_inverse(
        1, sm_count * MaxThreadsPerSM / inverse_items, 1);
    int slm_size_inverse = chunk_size * chunk_size * 2;

    queue.submit([&](sycl::handler& cgh) {
      sycl::local_accessor<float, 1> local_mem(
          sycl::range<1>(slm_size_inverse), cgh);
      cgh.parallel_for<ChunkInverseKernel<T, StateT>>(
          sycl::nd_range<3>{global_inverse * local_inverse, local_inverse},
          kernel_props,
          [=](auto) {
            chunk_inverse_kernel<T>(
                local_mem,
                A,
                query_start_loc,
                total_virtual_seqlen,
                batch_size,
                num_k_heads,
                head_k_dim,
                num_v_heads,
                head_v_dim);
          });
    });
  }

  // compute W U
  using WGTileComputeWU = chunk_gemm_policy_compute_wu::WGTile;
  using SGLayoutComputeWU = chunk_gemm_policy_compute_wu::SGLayout;
  using MMAComputeWU = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTileComputeWU>,
      SGLayoutComputeWU>::TiledMMA;
  auto mmaComputeWU = MMAComputeWU{};
  int MaxThreadsPerWorkgroupComputeWU = size(mmaComputeWU);
  sycl::range<3> local_compute_wu(1, 1, MaxThreadsPerWorkgroupComputeWU);
  sycl::range<3> global_compute_wu(
      1, sm_count * MaxThreadsPerSM / MaxThreadsPerWorkgroupComputeWU, 1);
  int slm_size_compute_wu = num_v_heads * 2 + chunk_size * 2;

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_mem(
        sycl::range<1>(slm_size_compute_wu), cgh);
    cgh.parallel_for<ChunkComputeWUKernel<T, StateT>>(
        sycl::nd_range<3>{
            global_compute_wu * local_compute_wu, local_compute_wu},
        kernel_props,
        [=](auto) {
          chunk_compute_wu_kernel<T, MMAComputeWU>(
              local_mem,
              A,
              w,
              u,
              q,
              k,
              v,
              b,
              a,
              nullptr,
              nullptr,
              query_start_loc,
              has_initial_state,
              total_virtual_seqlen,
              batch_size,
              num_k_heads,
              head_k_dim,
              num_v_heads,
              head_v_dim);
        });
  });

  // compute O
  using WGTileFwdO = chunk_gemm_policy_fwd_o::WGTile;
  using SGLayoutFwdO = chunk_gemm_policy_fwd_o::SGLayout;
  using MMAFwdO = typename TiledMMAHelper<
      MMA_Atom<decltype(op)>,
      Layout<WGTileFwdO>,
      SGLayoutFwdO>::TiledMMA;
  auto mmaFwdO = MMAFwdO{};
  int MaxThreadsPerWorkgroupFwdO = size(mmaFwdO);
  sycl::range<3> local_fwd_o(1, 1, MaxThreadsPerWorkgroupFwdO);
  sycl::range<3> global_fwd_o(batch_size, num_v_heads, 1);
  int slm_size_fwd_o = chunk_size + chunk_size + chunk_size;

  queue.submit([&](sycl::handler& cgh) {
    sycl::local_accessor<float, 1> local_mem(
        sycl::range<1>(slm_size_fwd_o), cgh);
    cgh.parallel_for<ChunkFwdOKernel<T, StateT>>(
        sycl::nd_range<3>{global_fwd_o * local_fwd_o, local_fwd_o},
        kernel_props,
        [=](auto) {
          chunk_fwd_o_kernel<T, StateT, MMAFwdO>(
              local_mem,
              core_attn_out,
              A,
              w,
              u,
              q,
              k,
              a,
              ssm_state,
              ssm_state_stride_0,
              query_start_loc,
              cache_indices,
              has_initial_state,
              token_indx,
              batch_size,
              total_virtual_seqlen,
              num_k_heads,
              head_k_dim,
              num_v_heads,
              head_v_dim);
        });
  });
}

}  // namespace gdn
