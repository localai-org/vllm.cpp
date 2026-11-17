/***************************************************************************************************
 * Copyright (C) 2025 Intel Corporation, All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice,
 *this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 *ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 *LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 *CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 *SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 *INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 *CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 *ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 *POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/

#pragma once

#include "cutlass/cutlass.h"
#include "cutlass/gemm/dispatch_policy.hpp"

#include "cute/algorithm/functional.hpp"
#include "cute/algorithm/gemm.hpp"
#include "cute/algorithm/subgroup_algorithms.hpp"
#include "cute/atom/mma_atom.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"

namespace cutlass::fmha {

template <int Stages>
class XeDefault {};  // Default FMHA mainloop, P in registers.

};  // namespace cutlass::fmha

namespace cutlass::fmha::collective {

static inline void sbarrier_wait() { asm volatile("sbarrier.wait\n"); }

static inline void sbarrier_signal() { asm volatile("sbarrier.signal\n"); }

static inline void gfence() { asm volatile("lsc_fence.ugm.none.group\n"); }

static inline void barrier() {
  asm volatile("lsc_fence.ugm.none.group\n");
  asm volatile("barrier\n");
}

using namespace cute;

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    class DispatchPolicy_,
    bool CausalMask_,
    bool LocalMask_,
    bool PagedKV_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void>  // Optional TiledCopy for loading V
struct FMHAFwdMainloop {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool CausalMask_,
    bool LocalMask_,
    bool PagedKV_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_>
struct FMHAFwdMainloop<
    XeDefault<Stages>,
    CausalMask_,
    LocalMask_,
    PagedKV_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool Fp8KV =
      is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool LocalMask = LocalMask_;
  static constexpr bool PagedKV = PagedKV_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    void* const scale_k;
    void* const scale_v;

    // Paged KV Cache
    const int* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int local_left, local_right;
    // Physical stride between paged blocks in seq-position units.
    int page_stride_elements;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  FMHAFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.scale_k,
        args.scale_v,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.local_left,
        args.local_right,
        args.page_stride_elements};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  CUTLASS_DEVICE int get_paged_idx(int K, int idx_b) {
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int b_offset = idx_b * params.max_pages_per_seq;
    int page_local_idx = K * get<1>(TileShapeQK{}) / params.page_size;

    // Clamp page_local_idx to the valid range [0, max_pages_per_seq - 1]
    if (page_local_idx >= params.max_pages_per_seq) {
      page_local_idx = params.max_pages_per_seq - 1;
    }

    int block_idx = params.ptr_page_table[b_offset + page_local_idx];
    int page_stride_tiles = params.page_stride_elements / get<1>(TileShapeQK{});
    return block_idx * page_stride_tiles + K % tiles_per_page;
  }

  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int const& idx_b,       // WG tile indices: (B)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int blk_k1_causal,
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int blk_local_l_safe,
      int blk_local_r_safe) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    constexpr int tile_k = get<1>(TileShapeQK{});

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)

    auto K_2D_0 = K_2D;
    auto V_2D_0 = V_2D;

    // --- Xe 2D-block surface-height split
    // ------------------------------------- The Xe LSC 2D block-load descriptor
    // stores the surface height in a 24-bit field, so a (rank-2) K/V surface
    // can address at most 2^24 rows. A paged KV cache may span many more rows
    // (total_seqlen_kv = num_blocks * page_stride_elements), which overflows
    // that field and makes the hardware wrap addresses. To keep the original
    // rank-2 load/prefetch logic we split the K and V surfaces into
    // page-aligned sub-surfaces, each at most 2^24 rows tall, and route every
    // physical tile to the surface that physically contains it.

    // The split state and the two `make_*_surface` lambdas are declared here at
    // function scope (rather than inside the `if constexpr` below) so that the
    // per-tile routing inside the K loop can reuse them. They are only ever
    // exercised when `has_large_surface` is true; otherwise they stay unused.
    [[maybe_unused]] int tiles_per_surface = 0x3fffffff;
    [[maybe_unused]] int rows_per_surface = 0;
    [[maybe_unused]] int k_rows_total = get<0>(K_2D.shape());
    [[maybe_unused]] int v_cols_total = get<1>(V_2D.shape());
    [[maybe_unused]] int64_t k_surf_shift = 0;
    [[maybe_unused]] int64_t v_surf_shift = 0;
    [[maybe_unused]] int k_pre_surf = -1;

    // Lazily materialize an arbitrary overflow sub-surface `surf` (>= 1). The
    // base is shifted by `surf` full surfaces and the height/width is the real
    // remaining extent of that surface, so the 2D block load never reads past
    // the cache. These are only ever built inside the K loop's cold branch, so
    // supporting any number of sub-surfaces adds at most one live descriptor
    // and the common path keeps its original register/descriptor footprint.
    [[maybe_unused]] auto make_k_surface = [&](int surf) {
      int64_t rem = int64_t(k_rows_total) - int64_t(surf) * rows_per_surface;
      int rows = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (rows < 1) {
        rows = 1;
      }
      return make_tensor(
          K_2D.data() + int64_t(surf) * k_surf_shift,
          make_layout(make_shape(rows, get<1>(K_2D.shape())), K_2D.stride()));
    };
    [[maybe_unused]] auto make_v_surface = [&](int surf) {
      int64_t rem = int64_t(v_cols_total) - int64_t(surf) * rows_per_surface;
      int cols = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (cols < 1) {
        cols = 1;
      }
      return make_tensor(
          V_2D.data() + int64_t(surf) * v_surf_shift,
          make_layout(make_shape(get<0>(V_2D.shape()), cols), V_2D.stride()));
    };

    if constexpr (has_large_surface) {
      constexpr int kXeMaxSurfaceRows = 1 << 24;
      if constexpr (PagedKV) {
        int page_stride_tiles = params.page_stride_elements / tile_k;
        int blocks_per_surface =
            kXeMaxSurfaceRows / params.page_stride_elements;
        if (blocks_per_surface < 1) {
          blocks_per_surface = 1;
        }
        tiles_per_surface = blocks_per_surface * page_stride_tiles;
        rows_per_surface = tiles_per_surface * tile_k;
      }

      // Build the (rank-2) K/V surface 0. Surface 0 keeps the original base
      // pointer and is the only surface addressed on the hot path; overflow
      // surfaces are materialized lazily (see make_k_surface/make_v_surface).
      int k_rows0, v_cols0;
      if constexpr (PagedKV) {
        k_rows0 =
            k_rows_total < rows_per_surface ? k_rows_total : rows_per_surface;
        k_surf_shift = int64_t(rows_per_surface) * get<0>(K_2D.stride());
        v_cols0 =
            v_cols_total < rows_per_surface ? v_cols_total : rows_per_surface;
        v_surf_shift = int64_t(rows_per_surface) * get<1>(V_2D.stride());
      } else {
        k_rows0 = k_rows_total;
        v_cols0 = v_cols_total;
      }

      K_2D_0 = make_tensor(
          K_2D.data(),
          make_layout(
              make_shape(k_rows0, get<1>(K_2D.shape())), K_2D.stride()));
      V_2D_0 = make_tensor(
          V_2D.data(),
          make_layout(
              make_shape(get<0>(V_2D.shape()), v_cols0), V_2D.stride()));

      cK = make_identity_tensor(make_shape(
          PagedKV ? rows_per_surface : k_rows_total,
          get<1>(K_2D.shape())));  // (k,d)
      cV = make_identity_tensor(make_shape(
          get<0>(V_2D.shape()),
          PagedKV ? rows_per_surface : v_cols_total));  // (v,k)
    }

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gK = local_tile(
        cK,
        TileShapeQK{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,K,D)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
    Tensor gV_split = local_tile(
        gV,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D_0};
    TiledCopyV copy_v{V_2D_0};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));
    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));
    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);
    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_0);
    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    /* Masking scalars */
    int lane_id = thr_id % intel::sg_size;
    constexpr int sg_tile_q = get<0>(TileShapeQK{}) / SGPerWG::value;
    int row_base = get<0>(blk_qv) * get<0>(TileShapeQK{}) +
                   (thr_id / intel::sg_size) * sg_tile_q;

    // PagedKV
    [[maybe_unused]] int page_idx = blk_k0;
    [[maybe_unused]] int next_page_idx = blk_k0;
    if constexpr (PagedKV) {
      next_page_idx = get_paged_idx(blk_k0, idx_b);
    }

    /* Q prefetch */
    CUTLASS_PRAGMA_UNROLL
    for (int D = 0; D < size<3>(pQgQ); D++) {
#ifndef VT_XE2_PREFILL_NO_PREFETCH
      prefetch(prefetch_q, pQgQ(_, _, _, D));
#endif
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    bool check_remainder_k = (seq_len % tile_k != 0);

    // FP8 KV Scale (per-tensor): fold scale_k into the softmax scale and
    // defer scale_v to a single post-loop output rescale. Mathematically
    // identical to per-element K/V rescaling (S = Q·K is linear in K and
    // O = sum P_i·V_i is linear in V), but removes both rescales from the
    // inner loop. The remaining fp8 -> ElementQ conversion is performed by
    // reorder() using the subgroup-vectorized cast primitive.
    ElementS effective_scale = params.scale;
    float scale_v = 1.f;
    if constexpr (Fp8KV) {
      // Null scale pointers denote unit scales for VT's narrow FP8 path.
      if (params.scale_k && params.scale_v) {
        const float scale_k = *static_cast<const float*>(params.scale_k);
        scale_v = *static_cast<const float*>(params.scale_v);
        effective_scale = params.scale * ElementS(scale_k);
      }
    }

    [[maybe_unused]] auto route_current = [&](int cur) -> int {
      int k_surf = PagedKV ? cur / tiles_per_surface : 0;
      int local = cur - k_surf * tiles_per_surface;
      if (k_pre_surf != k_surf) {
        k_pre_surf = k_surf;

        auto K_2D_n = make_k_surface(k_surf);
        auto V_2D_n = make_v_surface(k_surf);

        copy_k = TiledCopyK{K_2D_n};
        copy_v = TiledCopyV{V_2D_n};

        auto thr_copy_k_n = copy_k.get_slice(thr_id);
        auto thr_copy_v_n = copy_v.get_slice(thr_id);

        prefetch_k = make_block_2d_prefetch(copy_k);
        prefetch_v =
            make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_n);

        Tensor cK_n = make_identity_tensor(K_2D_n.shape());  // (k,d)
        Tensor cV_n = make_identity_tensor(V_2D_n.shape());  // (v,k)

        Tensor gK_n = local_tile(
            cK_n,
            TileShapeQK{},
            make_coord(_, _, _),
            Step<X, _1, _1>{});  // (k,d,K,D)
        Tensor gV_n = local_tile(
            cV_n, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
        Tensor gV_split_n = local_tile(
            gV_n,
            TileShapePV{},
            make_coord(_, _, 0),
            Step<X, _1, _1>{});  // (v,k,VV,K)

        tKgK = thr_copy_k_n.partition_S(gK_n);        // (atom_val,k',d',K,D)
        tVgV = thr_copy_v_n.partition_S(gV_split_n);  // (atom_val,v',k',VV,K)

        pKgK = prefetch_k.get_slice(thr_id).partition_S(gK_n);
        pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_n);
      }
      return local;
    };

    if constexpr (!has_large_surface) {
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(pKgK); D++) {
#ifndef VT_XE2_PREFILL_NO_PREFETCH
        prefetch(prefetch_k, pKgK(_, _, _, next_page_idx, D));
#endif
      }
    }

    /* Main loop, blocked in k. */
    for (int K = blk_k0; K < blk_k1; K++) {
      barrier_arrive(ScopeWorkgroup);

      bool need_causal = false;
      if constexpr (CausalMask) {
        need_causal = K >= blk_k1_causal;
      }

      int load_idx;
      if constexpr (has_large_surface) {
        int cur = PagedKV ? get_paged_idx(K, idx_b) : K;
        load_idx = route_current(cur);
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKgK); D++) {
#ifndef VT_XE2_PREFILL_NO_PREFETCH
          prefetch(prefetch_k, pKgK(_, _, _, load_idx, D));
#endif
        }
      } else {
        page_idx = next_page_idx;
        next_page_idx = K + 1;
        // next paged_idx
        if constexpr (PagedKV) {
          next_page_idx = get_paged_idx(next_page_idx, idx_b);
        }
        load_idx = PagedKV ? page_idx : K;
      }

      auto tKgK_cache = tKgK(_, _, _, load_idx, _);
      auto tVgV_cache = tVgV(_, _, _, _, load_idx);

      /* V prefetch for GEMM 2 */
#ifndef VT_XE2_PREFILL_NO_PREFETCH
      prefetch(prefetch_v, pVgV(_, _, _, load_idx));
#endif

      /* GEMM 1: S = Q * K^T */
      clear(tSrS); /* TODO: fuse w/ initial gemm call */
      CUTLASS_PRAGMA_UNROLL
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);
        reorder(tQrQ, tSrQ);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_k has been folded into effective_scale above.
        reorder(tKrK, tSrK);
        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* Causal masking and k remainder masking */
      if constexpr (CausalMask) {
        if (need_causal) {
          constexpr int kTileK = tile_k;
          constexpr int n_reps = kTileK / intel::sg_size;
          constexpr int elems_per_n = tSrS.size() / n_reps;
          int k_base = K * kTileK;
          CUTLASS_PRAGMA_UNROLL
          for (int n = 0; n < n_reps; n++) {
            int col = k_base + n * intel::sg_size + lane_id;
            int causal_bound = col - full_tile_offset - row_base;
            CUTLASS_PRAGMA_UNROLL
            for (int j = 0; j < elems_per_n; j++) {
              if (j < causal_bound) {
                tSrS(n * elems_per_n + j) = ElementS(-INFINITY);
              }
            }
          }
        }
      }
      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = K * tile_k + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }
      /* Local masking */
      if constexpr (LocalMask) {
        const bool need_left = (K < blk_local_l_safe);
        const bool need_right = (K > blk_local_r_safe);
        if (need_left || need_right) {
          Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
          Tensor gP = local_tile(
              cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
          auto cS_thread = thr_mma_qk.partition_C(gP);
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tSrS.size(); ++i) {
            int row_idx = get<0>(cS_thread(i));
            int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
            bool left_mask = col_idx < row_idx - params.local_left;
            bool right_mask = col_idx > row_idx + params.local_right;
            if (left_mask || right_mask) {
              tSrS(i) = ElementS(-INFINITY);
            }
          }
        }
      }

      /* Apply softmax */
      auto rescale =
          softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_v is applied once after the K loop below.
        reorder(tVrV, tArV);
        if (K != blk_k0) {
          CUTLASS_PRAGMA_UNROLL
          for (int i = 0; i < tArA.size() / VTiles; i++)
            tArA(_, _, _, VV)(i) *= broadcast<0>(rescale, tArA, i);
        }
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      if constexpr (!has_large_surface) {
        CUTLASS_PRAGMA_UNROLL
        for (int D = 0; D < size<4>(pKgK); D++) {
#ifndef VT_XE2_PREFILL_NO_PREFETCH
          prefetch(prefetch_k, pKgK(_, _, _, next_page_idx, D));
#endif
        }
      }

      barrier_wait(ScopeWorkgroup);
    }

    // Apply the per-tensor V scale once after the K loop.
    if constexpr (Fp8KV) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArA.size(); ++i) {
        tArA(i) *= ElementA(scale_v);
      }
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  FragSRow softmax(
      ElementS scale,      // Effective softmax scale (fp8 K scale folded in)
      bool first_block,    // First softmax block?
      FragS& tS,           // Softmax src/dst block
      FragSRow& tS_max,    // Softmax row-wise max accumulator
      FragSRow& tS_sum) {  // Softmax row-wise sum accumulator

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    FragSRow rescale;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      ElementS new_max = sycl::max(tS_max(i), scale * tS_bmax(i));
      rescale(i) = sycl::native::exp2(tS_max(i) - new_max);
      tS_max(i) = new_max;
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++) {
      tS(i) = sycl::native::exp2(scale * tS(i) - broadcast<0>(tS_max, tS, i));
    }

    /* Rescale existing S sums */
    if (!first_block) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_sum.size(); i++) {
        tS_sum(i) *= rescale(i);
      }
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_sum.size(); i++) {
      tS_sum(i) += tS_bsum(i);
    }
    return rescale;
  }
};

template <
    class DispatchPolicy_,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,  // Tiling for Q*K GEMM
    class TiledMMAPV_,  // Tiling for P*V GEMM
    int VTiles_,        // # of tiles in V dimension
    class TensorQ_,     // Global Q/K/V tensors
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_ = void,  // Optional TiledCopy for loading Q
    class TiledCopyK_ = void,  // Optional TiledCopy for loading K
    class TiledCopyV_ = void,  // Optional TiledCopy for loading V
    bool LocalMask_ = false>
struct DecodeFwdMainloop {
  static_assert(
      cutlass::detail::dependent_false<DispatchPolicy_>,
      "Could not find a mainloop specialization.");
};

/////////////////////////////////////////////////////////////////////////////////////////////////

template <
    int Stages,
    bool PagedKV_,
    bool CausalMask_,
    class TiledMMAQK_,
    class TiledMMAPV_,
    int VTiles_,
    class TensorQ_,
    class TensorK_,
    class TensorV_,
    class TiledCopyQ_,
    class TiledCopyK_,
    class TiledCopyV_,
    bool LocalMask_>
struct DecodeFwdMainloop<
    XeDefault<Stages>,
    PagedKV_,
    CausalMask_,
    TiledMMAQK_,
    TiledMMAPV_,
    VTiles_,
    TensorQ_,
    TensorK_,
    TensorV_,
    TiledCopyQ_,
    TiledCopyK_,
    TiledCopyV_,
    LocalMask_> {
  //
  // Type Aliases
  //
  using TiledMMAQK = TiledMMAQK_;
  using TiledMMAPV = TiledMMAPV_;
  using TileShapeQK = decltype(TiledMMAQK{}.tile_mnk());
  using TileShapePV = decltype(TiledMMAPV{}.tile_mnk());
  static constexpr int VTiles = VTiles_;
  using SubgroupLayoutQK = decltype(TiledMMAQK{}.get_atom_layout_mnk());
  using SGPerWG = decltype(product(
      take<1, 4>(shape(typename TiledMMAQK::ThrLayoutVMNK{}))));

  using TensorQ = TensorQ_;
  using TensorK = TensorK_;
  using TensorV = TensorV_;

  using ElementQ = typename TensorQ::engine_type::value_type;
  using ElementK = typename TensorK::engine_type::value_type;

  using TensorQ2D =
      decltype(TensorQ_{}(append<rank_v<TensorQ_>>(make_coord(_, _), 0)));
  using TensorK2D =
      decltype(TensorK_{}(append<rank_v<TensorK_>>(make_coord(_, _), 0)));
  using TensorV2D =
      decltype(TensorV_{}(append<rank_v<TensorV_>>(make_coord(_, _), 0)));

  using TiledCopyQ = conditional_t<
      is_void_v<TiledCopyQ_>,
      decltype(make_block_2d_copy_A(TiledMMAQK{}, TensorQ2D{})),
      TiledCopyQ_>;
  using TiledCopyK = conditional_t<
      is_void_v<TiledCopyK_>,
      decltype(make_block_2d_copy_B(TiledMMAQK{}, TensorK2D{})),
      TiledCopyK_>;
  using TiledCopyV = conditional_t<
      is_void_v<TiledCopyV_>,
      decltype(make_block_2d_copy_B(TiledMMAPV{}, TensorV2D{})),
      TiledCopyV_>;

  // TODO: static_asserts on TiledMMAPV here...

  //
  // Accumulator types
  //
  // FragS:    accumulator for Q*K MMA
  // FragO:    accumulator for P*V MMAs.
  //           Note: v mode may be split into multiple pieces
  //             to reduce register pressure.
  // Frag*Row types are reductions of the corresponding Frag* types
  //   over rows.
  //
  template <typename TiledMMA>
  using FragC = decltype(TiledMMA{}.get_slice(0).partition_sg_fragment_C(
      make_identity_tensor(select<0, 1>(TiledMMA{}.tile_mnk()))));

  using FragS = FragC<TiledMMAQK>;
  using FragSRow = decltype(reduce<1>(FragS{}, sycl::plus<void>{}));
  using FragSCol = decltype(reduce<0>(FragS{}, sycl::plus<void>{}));
  using ElementS = typename TiledMMAQK::ValTypeD;

  using SingleFragA = FragC<TiledMMAPV>;  // (atom val,q',v')
  using FragA =
      expand_sg_fragment_t<SingleFragA, 1, VTiles>;  // (atom val,q',v',VV)
  using FragARow = decltype(reduce<1>(FragA{}, sycl::plus<void>{}));
  // static_assert(is_same_v<decltype(FragSRow{}.shape()), float>, "dtype
  // mismatched");
  using ElementA = typename TiledMMAPV::ValTypeD;

  static constexpr bool PagedKV = PagedKV_;
  static constexpr bool CausalMask = CausalMask_;
  static constexpr bool Fp8KV =
      is_any_of_v<ElementK, float_e5m2_t, float_e4m3_t>;
  static constexpr bool LocalMask = LocalMask_;

  // User-facing arguments
  struct Arguments {
    ElementS const scale;
    void* const scale_k;
    void* const scale_v;
    // Paged KV Cache
    int const* ptr_page_table;
    int page_size;
    int max_pages_per_seq;
    int total_seqlen_kv;
    // Local Mask
    int window_size_left;
    int window_size_right;
    // Physical stride between paged blocks in seq-position units
    int page_stride_elements;
  };

  // Kernel-facing parameters
  using Params = Arguments;

  // SLM data
  struct SharedStorage {};

  Params params;

  //
  // Methods
  //

  DecodeFwdMainloop(Params const& params_, SharedStorage&) : params(params_) {}

  static constexpr Params
  to_underlying_arguments(Arguments const& args, void* /* workspace */) {
    constexpr double kLog2e = 1.4426950408889634074;  // log_2(e)
    ElementS val = args.scale * static_cast<ElementS>(kLog2e);
    return Params{
        val,
        args.scale_k,
        args.scale_v,
        args.ptr_page_table,
        args.page_size,
        args.max_pages_per_seq,
        args.total_seqlen_kv,
        args.window_size_left,
        args.window_size_right,
        args.page_stride_elements};
  }

  CUTLASS_HOST_DEVICE static bool can_implement(Arguments const&) {
    return true;
  }

  template <bool has_large_surface, typename QVCoord>
  CUTLASS_DEVICE void operator()(
      TensorQ2D const& Q_2D,  // (q,d)
      TensorK2D const& K_2D,  // (k,d)
      TensorV2D const& V_2D,  // (d,k)
      FragA& tArA,            // Output accumulator (q,v)
      FragARow& tA_max,       // Softmax row-wise max accumulator
      FragARow& tA_sum,       // Softmax row-wise sum accumulator
      QVCoord blk_qv,         // WG tile indices: (Q,V)
      int const& idx_b,       // WG tile indices: (B)
      int blk_k0,             // K block range: [K0,K1)
      int blk_k1,
      int total_blk,  // Total # of K blocks
      int thr_id,
      int seq_len,
      int full_tile_offset,
      int discard_seq_coord) {
    using namespace sycl::ext::oneapi::this_work_item;

    // Short dimension names:
    //    q = sequence len dimension for Q
    //    k = sequence len dimension for K
    //    d = head size dimension for K/Q
    //    v = head size dimension for V
    //   VV = MMA tile indices for V
    // Capital letters (Q, K, ...) refer to WG block indices.
    // Primed letters (q', k', ...) refer to atom block indices.

    auto tile_shape_v =
        make_shape(get<1>(TileShapePV{}) * C<VTiles>{}, get<2>(TileShapePV{}));

    /* Create proxy coordinate tensors for Q/K/P/V */
    Tensor cQ = make_identity_tensor(Q_2D.shape());               // (q,d)
    Tensor cK = make_identity_tensor(K_2D.shape());               // (k,d)
    Tensor cV = make_identity_tensor(V_2D.shape());               // (v,k)
    Tensor cP = make_identity_tensor(take<0, 2>(TileShapeQK{}));  // (q,k)

    auto K_2D_0 = K_2D;
    auto V_2D_0 = V_2D;

    // --- Xe 2D-block surface-height split
    // ------------------------------------- The Xe LSC 2D block-load descriptor
    // stores the surface height in a 24-bit field, so a (rank-2) K/V surface
    // can address at most 2^24 rows. A paged KV cache may span many more rows
    // (total_seqlen_kv = num_blocks * page_stride_elements), which overflows
    // that field and makes the hardware wrap addresses. To keep the original
    // rank-2 load/prefetch logic we split the K and V surfaces into
    // page-aligned sub-surfaces, each at most 2^24 rows tall, and route every
    // physical tile to the surface that physically contains it.

    // The split state and the two `make_*_surface` lambdas are declared here at
    // function scope (rather than inside the `if constexpr` below) so that the
    // per-tile routing inside the K loop can reuse them. They are only ever
    // exercised when `has_large_surface` is true; otherwise they stay unused.
    constexpr int tile_k = get<1>(TileShapeQK{});
    [[maybe_unused]] int tiles_per_surface = 0x3fffffff;
    [[maybe_unused]] int rows_per_surface = 0;
    [[maybe_unused]] int k_rows_total = get<0>(K_2D.shape());
    [[maybe_unused]] int v_cols_total = get<1>(V_2D.shape());
    [[maybe_unused]] int64_t k_surf_shift = 0;
    [[maybe_unused]] int64_t v_surf_shift = 0;
    [[maybe_unused]] int k_pre_surf = -1;

    // Lazily materialize an arbitrary overflow sub-surface `surf` (>= 1). The
    // base is shifted by `surf` full surfaces and the height/width is the real
    // remaining extent of that surface, so the 2D block load never reads past
    // the cache. These are only ever built inside the K loop's cold branch, so
    // supporting any number of sub-surfaces adds at most one live descriptor
    // and the common path keeps its original register/descriptor footprint.
    [[maybe_unused]] auto make_k_surface = [&](int surf) {
      int64_t rem = int64_t(k_rows_total) - int64_t(surf) * rows_per_surface;
      int rows = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (rows < 1) {
        rows = 1;
      }
      return make_tensor(
          K_2D.data() + int64_t(surf) * k_surf_shift,
          make_layout(make_shape(rows, get<1>(K_2D.shape())), K_2D.stride()));
    };
    [[maybe_unused]] auto make_v_surface = [&](int surf) {
      int64_t rem = int64_t(v_cols_total) - int64_t(surf) * rows_per_surface;
      int cols = rem > rows_per_surface ? rows_per_surface : int(rem);
      if (cols < 1) {
        cols = 1;
      }
      return make_tensor(
          V_2D.data() + int64_t(surf) * v_surf_shift,
          make_layout(make_shape(get<0>(V_2D.shape()), cols), V_2D.stride()));
    };

    if constexpr (has_large_surface) {
      constexpr int kXeMaxSurfaceRows = 1 << 24;
      if constexpr (PagedKV) {
        int page_stride_tiles = params.page_stride_elements / tile_k;
        int blocks_per_surface =
            kXeMaxSurfaceRows / params.page_stride_elements;
        if (blocks_per_surface < 1) {
          blocks_per_surface = 1;
        }
        tiles_per_surface = blocks_per_surface * page_stride_tiles;
        rows_per_surface = tiles_per_surface * tile_k;
      }

      // Build the (rank-2) K/V surface 0. Surface 0 keeps the original base
      // pointer and is the only surface addressed on the hot path; overflow
      // surfaces are materialized lazily (see make_k_surface/make_v_surface).
      int k_rows0, v_cols0;
      if constexpr (PagedKV) {
        k_rows0 =
            k_rows_total < rows_per_surface ? k_rows_total : rows_per_surface;
        k_surf_shift = int64_t(rows_per_surface) * get<0>(K_2D.stride());
        v_cols0 =
            v_cols_total < rows_per_surface ? v_cols_total : rows_per_surface;
        v_surf_shift = int64_t(rows_per_surface) * get<1>(V_2D.stride());
      } else {
        k_rows0 = k_rows_total;
        v_cols0 = v_cols_total;
      }

      K_2D_0 = make_tensor(
          K_2D.data(),
          make_layout(
              make_shape(k_rows0, get<1>(K_2D.shape())), K_2D.stride()));
      V_2D_0 = make_tensor(
          V_2D.data(),
          make_layout(
              make_shape(get<0>(V_2D.shape()), v_cols0), V_2D.stride()));

      cK = make_identity_tensor(make_shape(
          PagedKV ? rows_per_surface : k_rows_total,
          get<1>(K_2D.shape())));  // (k,d)
      cV = make_identity_tensor(make_shape(
          get<0>(V_2D.shape()),
          PagedKV ? rows_per_surface : v_cols_total));  // (v,k)
    }

    /* Partition global tensors into workgroup tiles */
    Tensor gQ = local_tile(
        cQ, TileShapeQK{}, append(blk_qv, _), Step<_1, X, _1>{});  // (q,d,D)
    Tensor gK = local_tile(
        cK,
        TileShapeQK{},
        make_coord(_, _, _),
        Step<X, _1, _1>{});  // (k,d,K,D)
    Tensor gV =
        local_tile(cV, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
    Tensor gV_split = local_tile(
        gV,
        TileShapePV{},
        make_coord(_, _, 0),
        Step<X, _1, _1>{});  // (v,k,VV,K)

    /* Create global -> register copies */
    TiledCopyQ copy_q{Q_2D};
    TiledCopyK copy_k{K_2D_0};
    TiledCopyV copy_v{V_2D_0};

    /* Create MMAs */
    TiledMMAQK mma_qk{};
    TiledMMAPV mma_pv{};

    /* Slice TiledCopy/TiledMMA operations down to to work-item level */
    auto thr_copy_q = copy_q.get_slice(thr_id);
    auto thr_copy_k = copy_k.get_slice(thr_id);
    auto thr_copy_v = copy_v.get_slice(thr_id);
    auto thr_mma_qk = mma_qk.get_slice(thr_id);
    auto thr_mma_pv = mma_pv.get_slice(thr_id);

    /* Partition coordinate tensors for copy */
    auto tQgQ = thr_copy_q.partition_S(gQ);        // (atom_val,q',d',D)
    auto tKgK = thr_copy_k.partition_S(gK);        // (atom_val,k',d',K,D)
    auto tVgV = thr_copy_v.partition_S(gV_split);  // (atom_val,v',k',VV,K)

    /* Create register fragments for MMA and copies */
    auto tQrQ = thr_copy_q.partition_sg_fragment_D(gQ(_, _, 0));
    auto tSrQ = thr_mma_qk.partition_sg_fragment_A(gQ(_, _, 0));

    auto tKrK = thr_copy_k.partition_sg_fragment_D(gK(_, _, 0, 0));
    auto tSrK = thr_mma_qk.partition_sg_fragment_B(gK(_, _, 0, 0));

    auto tSrS = thr_mma_qk.partition_sg_fragment_C(cP);
    auto tArP = thr_mma_pv.partition_sg_fragment_A(cP);

    auto tVrV = thr_copy_v.partition_sg_fragment_D(gV_split(_, _, 0, 0));
    auto tArV = thr_mma_pv.partition_sg_fragment_B(gV_split(_, _, 0, 0));

    /* Create TiledCopy objects for prefetches */
    auto prefetch_q = make_block_2d_prefetch(copy_q);
    auto prefetch_k = make_block_2d_prefetch(copy_k);
    auto prefetch_v =
        make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_0);

    /* Partition global tensors for prefetch */
    auto pQgQ = prefetch_q.get_slice(thr_id).partition_S(gQ);
    auto pKgK = prefetch_k.get_slice(thr_id).partition_S(gK);
    auto pVgV = prefetch_v.get_slice(thr_id).partition_S(gV);

    // ------
    // Kernel
    // ------

    // PagedKV
    int tiles_per_page = params.page_size / get<1>(TileShapeQK{});
    int page_stride_tiles = params.page_stride_elements / get<1>(TileShapeQK{});
    int tile_idx = blk_k0;
    int b_offset = idx_b * params.max_pages_per_seq;
    if constexpr (PagedKV) {
      int page_local_idx = tile_idx * get<1>(TileShapeQK{}) / params.page_size;
      int block_idx = params.ptr_page_table[b_offset + page_local_idx];
      tile_idx = block_idx * page_stride_tiles + tile_idx % tiles_per_page;
    }

    /* Initialization steps for first block: Q/K prefetch, O init */
    /* TODO: limit D prefetch for large head size, and reorder K prefetches */
    for (int D = 0; D < size<3>(pQgQ); D++) {
      prefetch(prefetch_q, pQgQ(_, _, _, D));
    }

    [[maybe_unused]] auto update_surface_info = [&](int& next_page_idx) {
      // Route the current physical tile to its sub-surface (see split above).
      int k_surf = PagedKV ? next_page_idx / tiles_per_surface : 0;
      next_page_idx = next_page_idx - k_surf * tiles_per_surface;

      if (k_pre_surf != k_surf) {
        k_pre_surf = k_surf;

        auto K_2D_n = make_k_surface(k_surf);
        auto V_2D_n = make_v_surface(k_surf);

        copy_k = TiledCopyK{K_2D_n};
        copy_v = TiledCopyV{V_2D_n};

        auto thr_copy_k_n = copy_k.get_slice(thr_id);
        auto thr_copy_v_n = copy_v.get_slice(thr_id);

        prefetch_k = make_block_2d_prefetch(copy_k);
        prefetch_v =
            make_block_2d_prefetch<SGPerWG::value>(tile_shape_v, V_2D_n);

        Tensor cK_n = make_identity_tensor(K_2D_n.shape());  // (k,d)
        Tensor cV_n = make_identity_tensor(V_2D_n.shape());  // (v,k)

        Tensor gK_n = local_tile(
            cK_n,
            TileShapeQK{},
            make_coord(_, _, _),
            Step<X, _1, _1>{});  // (k,d,K,D)
        Tensor gV_n = local_tile(
            cV_n, tile_shape_v, make_coord(get<1>(blk_qv), _));  // (v,k,K)
        Tensor gV_split_n = local_tile(
            gV_n,
            TileShapePV{},
            make_coord(_, _, 0),
            Step<X, _1, _1>{});  // (v,k,VV,K)

        tKgK = thr_copy_k_n.partition_S(gK_n);        // (atom_val,k',d',K,D)
        tVgV = thr_copy_v_n.partition_S(gV_split_n);  // (atom_val,v',k',VV,K)

        pKgK = prefetch_k.get_slice(thr_id).partition_S(gK_n);
        pVgV = prefetch_v.get_slice(thr_id).partition_S(gV_n);
      }
    };

    if constexpr (has_large_surface) {
      update_surface_info(tile_idx);
    }

    for (int D = 0; D < size<4>(pKgK); D++) {
      prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
    }

    clear(tArA);
    fill(tA_max, cutlass::platform::numeric_limits<ElementA>::lowest());
    clear(tA_sum);

    /* Check if */
    bool check_remainder_k = (seq_len % get<1>(TileShapeQK{}) != 0);

    // FP8 KV Scale (per-tensor): fold scale_k into the softmax scale and
    // defer scale_v to a single post-loop output rescale. Mathematically
    // identical to per-element K/V rescaling (S = Q·K is linear in K and
    // O = sum P_i·V_i is linear in V), but removes both rescales from the
    // inner loop. The remaining fp8 -> ElementQ conversion is performed by
    // reorder() using the subgroup-vectorized cast primitive.
    ElementS effective_scale = params.scale;
    float scale_v = 1.f;
    if constexpr (Fp8KV) {
      // Null scale pointers denote unit scales for VT's narrow FP8 path.
      if (params.scale_k && params.scale_v) {
        const float scale_k = *static_cast<const float*>(params.scale_k);
        scale_v = *static_cast<const float*>(params.scale_v);
        effective_scale = params.scale * ElementS(scale_k);
      }
    }

    /* Main loop, blocked in k. */
    int next_tile_idx;
    for (int K = blk_k0; K < blk_k1; K++) {
      /* Split barrier to keep threads together */
      // barrier_arrive(ScopeWorkgroup);

      auto tKgK_cache =
          PagedKV ? tKgK(_, _, _, tile_idx, _) : tKgK(_, _, _, K, _);
      auto tVgV_cache =
          PagedKV ? tVgV(_, _, _, _, tile_idx) : tVgV(_, _, _, _, K);

      /* GEMM 1: S = K * Q */
      clear(tSrS); /* TODO: fuse w/ initial gemm call */
      for (int D = 0; D < size<4>(tKgK); D++) {
        copy(copy_q, tQgQ(_, _, _, D), tQrQ);
        copy(copy_k, tKgK_cache(_, _, _, D), tKrK);

        reorder(tQrQ, tSrQ);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_k has been folded into effective_scale above.
        reorder(tKrK, tSrK);

        cute::gemm(mma_qk, tSrQ, tSrK, tSrS);
      }

      /* V prefetch for GEMM 2 */
      prefetch(prefetch_v, pVgV(_, _, _, tile_idx));

      /* Causal masking */
      // No Causal masking in decoding
      // if constexpr (CausalMask) {
      //   if (K == blk_k1 - 1) {
      //     // Need to get global col and row indices to mask the elements
      //     Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
      //     Tensor gP = local_tile(cPgP, take<0,2>(TileShapeQK{}),
      //     make_coord(get<0>(blk_qv), K)); auto cS_thread =
      //     thr_mma_qk.partition_C(gP); CUTLASS_PRAGMA_UNROLL for (int i = 0; i
      //     < tSrS.size(); ++i) {
      //       int row_idx = get<0>(cS_thread(i));
      //       int col_idx = get<1>(cS_thread(i));
      //       if (col_idx - full_tile_offset > row_idx - discard_seq_coord) {
      //         tSrS(i) = ElementS(-INFINITY);
      //       }
      //     }
      //   }
      // }

      /* Local/sliding window masking */
      if constexpr (LocalMask) {
        // For decode, all packed GQA heads share the same KV position
        // (seq_len_kv - 1). Use a fixed decode row for all elements.
        int decode_row = seq_len - 1 - full_tile_offset;
        Tensor cPgP = make_identity_tensor(make_shape(seq_len, seq_len));
        Tensor gP = local_tile(
            cPgP, take<0, 2>(TileShapeQK{}), make_coord(get<0>(blk_qv), K));
        auto cS_thread = thr_mma_qk.partition_C(gP);
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); ++i) {
          int col_idx = get<1>(cS_thread(i)) - full_tile_offset;
          bool left_mask = col_idx < decode_row - params.window_size_left;
          bool right_mask = col_idx > decode_row + params.window_size_right;
          if (left_mask || right_mask) {
            tSrS(i) = ElementS(-INFINITY);
          }
        }
      }

      /* k masking for remainder tiles */
      if (check_remainder_k && K == blk_k1 - 1) {
        FragSCol k_rem_mask;
        int k = get<0>(tKgK(0, 0, 0, K, 0)) + get_sub_group().get_local_id()[0];
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < k_rem_mask.size(); i++, k += intel::sg_size) {
          k_rem_mask(i) =
              (k < seq_len) ? ElementS(sycl::nan(0u)) : ElementS(-INFINITY);
        }
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < tSrS.size(); i++) {
          tSrS(i) = sycl::fmin(tSrS(i), broadcast<1>(k_rem_mask, tSrS, i));
        }
      }

      /* Apply softmax and scaling */
      softmax(effective_scale, K == blk_k0, tSrS, tA_max, tA_sum, tArA);
      reorder(tSrS, tArP);

      /* GEMM 2: A += P * V, split in v dimension */
      CUTLASS_PRAGMA_UNROLL
      for (int VV = 0; VV < VTiles; VV++) {
        copy(copy_v, tVgV_cache(_, _, _, VV), tVrV);
        // reorder() performs the (vectorized) fp8 -> ElementQ cast; the
        // per-tensor scale_v is applied once after the K loop below.
        reorder(tVrV, tArV);
        cute::gemm(mma_pv, tArP, tArV, tArA(_, _, _, VV));
      }

      barrier();

      // next tile_idx
      next_tile_idx = K + 1;
      if constexpr (PagedKV) {
        int next_page_local_idx =
            next_tile_idx * get<1>(TileShapeQK{}) / params.page_size;
        if (next_page_local_idx < params.max_pages_per_seq) {
          int next_block_idx =
              params.ptr_page_table[b_offset + next_page_local_idx];
          next_tile_idx = next_block_idx * page_stride_tiles +
                          next_tile_idx % tiles_per_page;
        } else {
          // set to last page
          next_tile_idx = params.max_pages_per_seq * tiles_per_page - 1;
        }
      }
      tile_idx = next_tile_idx;

      if constexpr (has_large_surface) {
        update_surface_info(tile_idx);
      }

      /* K prefetch */
      for (int D = 0; D < size<4>(pKgK); D++) {
        prefetch(prefetch_k, pKgK(_, _, _, tile_idx, D));
      }

      // barrier_wait(ScopeWorkgroup);
    }

    // Apply the per-tensor V scale once after the K loop.
    if constexpr (Fp8KV) {
      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tArA.size(); ++i) {
        tArA(i) *= ElementA(scale_v);
      }
    }
  }

  // Single step of blocked softmax.
  CUTLASS_DEVICE
  void softmax(
      ElementS scale,    // Effective softmax scale (fp8 K scale folded in)
      bool first_block,  // First softmax block?
      FragS& tS,         // Softmax src/dst block
      FragSRow& tS_max,  // Softmax row-wise max accumulator
      FragSRow& tS_sum,  // Softmax row-wise sum accumulator
      FragA& tA) {       // O accumulator (for rescaling)

    /* Compute row-wise maxima for this block */
    auto tS_bmax = reduce<1>(tS, sycl::maximum{});

    /* Update (scaled) maxima */
    auto tS_prev_max = tS_max;
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS_max.size(); i++) {
      tS_max(i) = sycl::max(tS_max(i), scale * tS_bmax(i));
    }

    /* Scale S and subtract maxima, then exponentiate */
    CUTLASS_PRAGMA_UNROLL
    for (int i = 0; i < tS.size(); i++)
      tS(i) = sycl::native::exp2(scale * tS(i) - broadcast<0>(tS_max, tS, i));

    /* Rescale existing S sums and O accumulator */
    if (!first_block) {
      FragSRow rescale;

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tS_max.size(); i++) {
        rescale(i) = sycl::native::exp2(tS_prev_max(i) - tS_max(i));
        tS_sum(i) *= rescale(i);
      }

      CUTLASS_PRAGMA_UNROLL
      for (int i = 0; i < tA.size(); i++)
        tA(i) *= broadcast<0>(rescale, tA, i);
    }

    /* Update sums */
    auto tS_bsum = reduce<1>(tS, sycl::plus<void>{});
    for (int i = 0; i < tS_sum.size(); i++)
      tS_sum(i) += tS_bsum(i);
  }
};

template <typename SGLayoutQK>
CUTLASS_HOST_DEVICE constexpr auto get_sg_layout_pv(SGLayoutQK const&) {
  return make_layout(
      get<0>(SGLayoutQK{}), Layout<_1, _0>{}, get<1>(SGLayoutQK{}));
}

// Get a P*V TiledMMA given K*Q tile size and SG configuration, for mainloops
//   not supporting S data interchange among subgroups (e.g. XeDefault).
template <
    typename MMAOp,
    typename WGTileQK,
    typename SGLayoutQK,
    typename TileV>
CUTLASS_HOST_DEVICE constexpr auto get_tiled_mma_pv(
    MMAOp const&,
    WGTileQK const& wg_tile_qk,
    SGLayoutQK const& sg_layout_qk,
    TileV const&) {
  using TileQ = decltype(get<0>(wg_tile_qk));
  using TileK = decltype(get<1>(wg_tile_qk));

  using WGTilePV = Shape<TileQ, TileV, TileK>;
  using SGLayoutPV = decltype(get_sg_layout_pv(sg_layout_qk));

  static_assert(
      size(SGLayoutPV{}) == size(SGLayoutQK{}),
      "Q*K cannot be parallelized in the head size dimension");

  return TiledMMAHelper<MMAOp, WGTilePV, SGLayoutPV>{};
}

}  // namespace cutlass::fmha::collective

/////////////////////////////////////////////////////////////////////////////////////////////////
