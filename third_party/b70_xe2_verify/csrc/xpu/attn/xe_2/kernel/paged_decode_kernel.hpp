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
#include "cutlass/gemm/gemm.h"
#include "cutlass/kernel_hardware_info.hpp"

#include "cute/util/type_traits.hpp"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "../collective/chunk_prefill_mainloop.hpp"
#include "../collective/chunk_prefill_epilogue.hpp"

namespace cutlass::fmha::kernel {

using namespace cute;

///////////////////////////////////////////////////////////////////////////////
template <bool IsVarLen_ = false>
struct DecodeProblemShape {
  using SeqLenType = cute::
      conditional_t<IsVarLen_, cutlass::fmha::collective::VariableLength, int>;
  int batch;
  int num_heads_q, num_heads_kv;
  SeqLenType seq_len_qo, seq_len_kv;
  int head_size_qk, head_size_vo;
};

///////////////////////////////////////////////////////////////////////////////

template <
    class ProblemShape_,
    class CollectiveMainloop_,
    class CollectiveEpilogue_,
    class TileScheduler_>
class XeFMHAFwdSplitKVKernel {
 public:
  //
  // Type Aliases
  //
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len =
      cutlass::fmha::collective::is_variable_length_v<
          typename ProblemShape::SeqLenType>;
  using CollectiveMainloop = CollectiveMainloop_;
  using MainloopArguments = typename CollectiveMainloop::Arguments;
  using MainloopParams = typename CollectiveMainloop::Params;

  using TiledMMAQK = typename CollectiveMainloop::TiledMMAQK;
  using TiledMMAPV = typename CollectiveMainloop::TiledMMAPV;
  using TileShapeQK = typename CollectiveMainloop::TileShapeQK;
  using TileShapePV = typename CollectiveMainloop::TileShapePV;
  using SubgroupLayoutQK = typename CollectiveMainloop::SubgroupLayoutQK;
  using ElementQ = typename CollectiveMainloop::TensorQ::element_type;
  using ElementK = typename CollectiveMainloop::TensorK::element_type;
  using ElementV = typename CollectiveMainloop::TensorV::element_type;

  using StrideQ = decltype(stride(typename CollectiveMainloop::TensorQ{}));
  using StrideK = decltype(stride(typename CollectiveMainloop::TensorK{}));
  using StrideV = decltype(stride(typename CollectiveMainloop::TensorV{}));

  using SGPerWG = typename CollectiveMainloop::SGPerWG;

  using FragA = typename CollectiveMainloop::FragA;
  using FragARow = typename CollectiveMainloop::FragARow;

  // Tile scheduler derived types
  using TileScheduler = TileScheduler_;
  using TileSchedulerParams = typename TileScheduler::Params;

  // Epilogue derived types
  using CollectiveEpilogue = CollectiveEpilogue_;
  using EpilogueArguments = typename CollectiveEpilogue::Arguments;
  using EpilogueParams = typename CollectiveEpilogue::Params;

  using TileShapeO = typename CollectiveEpilogue::TileShapeO;
  using ElementO = typename CollectiveEpilogue::TensorO::element_type;
  using ElementLSE = typename CollectiveEpilogue::ElementLSE;
  using StrideO = decltype(stride(typename CollectiveEpilogue::TensorO{}));

  // Kernel level shared memory storage
  using MainloopSharedStorage = typename CollectiveMainloop::SharedStorage;
  using EpilogueSharedStorage = typename CollectiveEpilogue::SharedStorage;
  union SharedStorage {
    MainloopSharedStorage mainloop;
    EpilogueSharedStorage epilogue;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

  static constexpr int max_num_kv_splits = SGPerWG::value * intel::sg_size;
  static constexpr int dpas_max_repeat_count = 8;
  static constexpr bool Sink = CollectiveEpilogue::Sink;
  using ElementSink = typename CollectiveEpilogue::ElementSink;

  static constexpr int kXeMaxSurfaceRows = 1 << 24;

  // Device side arguments
  struct KernelArguments {
    ProblemShape shape;
    const ElementQ* Q;
    StrideQ dQ;
    const ElementK* K;
    StrideK dK;
    const ElementV* V;
    StrideV dV;
    ElementO* Oaccum;
    StrideO dOaccum;
    ElementLSE* exp_sums;
    StrideO dExp_sums;
    ElementLSE* max_logits;
    StrideO dMax_logits;

    const ElementSink* sm_sink;

    // per-batch mask: true = prefill, false = decode; nullptr = process all
    const bool* is_prefill;
    const int* splits_per_seq =
        nullptr;  // per-seq split counts; null => use global num_kv_splits
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    MainloopArguments mainloop{};
    EpilogueArguments epilogue{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1;  // no split by default
  };

  // Kernel entry point API
  struct Params {
    KernelParams kernel;
    MainloopParams mainloop;
    EpilogueParams epilogue;
    TileSchedulerParams scheduler;
  };

  //
  // Methods
  //

  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        CollectiveMainloop::to_underlying_arguments(args.mainloop, workspace),
        CollectiveEpilogue::to_underlying_arguments(args.epilogue, workspace),
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    if (!is_var_len && args.kernel.shape.seq_len_qo != 1) {
      // decode only
      return false;
    }

    if (args.num_kv_splits > max_num_kv_splits) {
      return false;
    }

    return CollectiveMainloop::can_implement(args.mainloop) &&
           CollectiveEpilogue::can_implement(args.epilogue);
  }

  static int get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(
        params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(
      ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
#ifdef B70_VERIFY_MASK
      // The M04 extension packs all verification rows into the query-head
      // dimension and therefore exposes one physical decode row to the
      // scheduler, irrespective of the caller's original q_len metadata.
      return Shape<int, int>{
          1, problem_shape.seq_len_kv.cumulative_length[batch]};
#else
      auto q_len = cutlass::fmha::collective::apply_variable_length(
          Shape<VariableLength>{problem_shape.seq_len_qo}, batch);
      return Shape<int, int>{
          get<0>(q_len), problem_shape.seq_len_kv.cumulative_length[batch]};
#endif
    } else {
      return Shape<int, int>{
          problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;
    int head_group_q = s.num_heads_q / s.num_heads_kv;

    int thr_id = int(ThreadIdxX());
    int sub_group_id = thr_id / intel::sg_size;
    int q_sg_tile = get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})));

    auto cS = make_identity_tensor(take<0, 2>(TiledMMAQK{}.tile_mnk()));
    auto tScS = TiledMMAQK{}.get_slice(thr_id).partition_C(cS);
    auto q_offset_wi = get<0>(tScS(0));
    auto q_offset_sg = group_broadcast(
        sycl::ext::oneapi::this_work_item::get_sub_group(), q_offset_wi, 0);

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits_;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto
          [blk_q,
           blk_v,
           head,
           idx_b,
           idx_kv_split,
           wl_tile_start,
           wl_tile_count] =
              tile_scheduler
                  .get_block_coord();  // (Q,V,h,b,split,tile_start,tile_count)

      // Skip prefill batches when is_prefill mask is provided
      if (p.is_prefill != nullptr && p.is_prefill[idx_b]) continue;

      auto blk_qv = make_coord(blk_q, blk_v);
      int head_q_start = head * head_group_q;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;
      // Decode packs the GQA head-group into the Q/row dimension (seq_len_qo
      // is always 1). blk_q tiles head_group_q in steps of the packed-Q tile
      // size, so skip work-groups whose first head-group row is out of range.
      if (blk_q * get<0>(TileShapeQK{}) >= head_group_q) continue;

      auto offset = cute::min(seq_len_qo, seq_len_kv);
      auto discard_seq_coord = seq_len_qo - offset;
#ifdef B70_VERIFY_MASK
      // M04 experiment: Causal=true is reserved for packed speculative
      // verification. Each KV head owns six ordinary GQA query heads; the
      // packed Q group concatenates 2..5 verification rows. Pass the row
      // count through the otherwise-unused negative decode coordinate.
      if constexpr (CollectiveMainloop::CausalMask) {
        discard_seq_coord = -(head_group_q / 6);
      }
#endif
      auto full_tile_offset = seq_len_kv - offset;
      int seq_coord =
          cute::min(seq_len_qo, (blk_q * get<0>(TileShapeQK{}) + q_offset_sg));

      if (CollectiveMainloop::CausalMask && seq_coord < discard_seq_coord)
        continue;
      // For decode window_size_right doesn't have effect
      const int seq_len = seq_len_kv;
      // For decode, all packed GQA heads are at position seq_len_kv - 1.
      // Use seq_len - 1 (= seq_len_kv - 1) as the decode position for
      // k_block0 to match ReduceSplitK's computation.
      const int k_block0 =
          CollectiveMainloop::LocalMask
              ? cute::max(seq_len - 1 - params.mainloop.window_size_left, 0) /
                    get<1>(TileShapeQK{})
              : 0;
      const int k_blocks = cute::ceil_div(seq_len, get<1>(TileShapeQK{}));
      const int windowed_k_blocks = k_blocks - k_block0;

      int offset_q = 0, offset_k = 0, offset_v = 0, offset_o = 0;
      int offset_exp_sums = 0, offset_max_logits = 0;
      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;

        // Use Q's actual per-token stride (get<0>(dQ)) instead of assuming
        // the packed value num_heads_q * head_size_qk so non-contiguous Q
        // tensors (e.g. strided slices, permuted/transposed views, slices
        // of a wider buffer) are read correctly.
        offset_q = get<0>(p.dQ) * qo_cumulative[idx_b];
        offset_o = s.num_heads_q * s.head_size_vo * num_kv_splits *
                   qo_cumulative[idx_b];
        offset_exp_sums = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        offset_max_logits =
            s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];

        // for gqa packing, seq_len_qo must be 1
        seq_len_qo = 1;
      }

      // neglect seq_len_qo since it's always 1 for decode
      auto batch_dim = is_var_len ? 1 : s.batch;
      auto shape_Q =
          make_shape(head_group_q, s.head_size_qk, s.num_heads_kv, batch_dim);
      // shape
      auto total_seqlen_kv = params.mainloop.total_seqlen_kv;
      auto shape_K = make_shape(
          total_seqlen_kv, s.head_size_qk, s.num_heads_kv, batch_dim);
      auto shape_V = make_shape(
          s.head_size_vo, total_seqlen_kv, s.num_heads_kv, batch_dim);

      auto shape_O = make_shape(
          head_group_q,
          s.head_size_vo,
          s.num_heads_kv,
          num_kv_splits,
          batch_dim);
      auto shape_exp_sums =
          make_shape(head_group_q, num_kv_splits, s.num_heads_kv, batch_dim);
      auto shape_max_logits =
          make_shape(head_group_q, num_kv_splits, s.num_heads_kv, batch_dim);
      auto shape_sink = make_shape(s.num_heads_kv, head_group_q);

      int kv_split_offset;
      int num_effective_kv_blocks;
      int seq_num_kv_splits;
      bool is_single_split;

      if (wl_tile_start >= 0) {
        // Compact grid: use pre-computed tile range from work_list.
        // Python (build_decode_split_plan) has already folded the
        // single-split heuristic and balanced assignment into the plan.
        kv_split_offset = k_block0 + wl_tile_start;
        num_effective_kv_blocks = wl_tile_count;
        seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                ? p.splits_per_seq[idx_b]
                                : num_kv_splits;
        is_single_split = (seq_num_kv_splits <= 1);
      } else {
        // Legacy path: compute split range on the fly
        seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                ? p.splits_per_seq[idx_b]
                                : num_kv_splits;

        if (idx_kv_split >= seq_num_kv_splits) {
          continue;
        }

        int num_blocks_per_split =
            cute::ceil_div(windowed_k_blocks, seq_num_kv_splits);

        constexpr int tile_n = get<1>(TileShapeQK{});
        constexpr int kMinBlocksForSplit = (tile_n <= 64) ? 32 : 128;
        is_single_split =
            (seq_num_kv_splits > 1) && (windowed_k_blocks < kMinBlocksForSplit);

        if (is_single_split) {
          if (idx_kv_split > 0) {
            continue;
          }
          kv_split_offset = k_block0;
          num_effective_kv_blocks = windowed_k_blocks;
        } else {
          kv_split_offset = k_block0 + idx_kv_split * num_blocks_per_split;
          num_effective_kv_blocks = cute::min(
              windowed_k_blocks - idx_kv_split * num_blocks_per_split,
              num_blocks_per_split);
        }
      }

      if (num_effective_kv_blocks <= 0) {
        continue;
      }

      auto dcQ = const_cast<ElementQ*>(p.Q + offset_q);
      auto dcK = const_cast<ElementK*>(p.K);
      auto dcV = const_cast<ElementV*>(p.V);
      auto ptrO = p.Oaccum + offset_o;
      auto ptrExp_sums = p.exp_sums + offset_exp_sums;
      auto ptrMax_logits = p.max_logits + offset_max_logits;

      // Q layout uses the tensor's actual head stride (get<2>(dQ)) so
      // non-contiguous Q (e.g. permuted heads or sliced from a wider
      // head-dim buffer) is read correctly. The GQA grouping splits the
      // num_heads_q dim into (num_heads_kv outer, head_group_q inner) which
      // assumes heads are regularly strided by q_stride_heads.
      auto layout_q = make_layout(
          shape_Q,
          make_stride(
              get<2>(p.dQ),
              _1{},
              static_cast<int>(head_group_q) * get<2>(p.dQ),
              get<3>(p.dQ)));
      auto layout_k = make_layout(
          shape_K, make_stride(get<0>(p.dK), _1{}, get<2>(p.dK), get<3>(p.dK)));
      auto layout_v = make_layout(
          shape_V, make_stride(_1{}, get<1>(p.dV), get<2>(p.dV), get<3>(p.dV)));

      auto layout_o = make_ordered_layout(shape_O, Step<_1, _0, _2, _3, _4>{});
      auto layout_exp_sums =
          make_ordered_layout(shape_exp_sums, Step<_1, _0, _2, _3>{});
      auto layout_max_logits =
          make_ordered_layout(shape_max_logits, Step<_1, _0, _2, _3>{});
      auto layout_sink = make_ordered_layout(shape_sink, Step<_1, _0>{});

      Tensor Q = make_tensor(make_gmem_ptr(dcQ), layout_q);
      Tensor K = make_tensor(make_gmem_ptr(dcK), layout_k);
      Tensor V = make_tensor(make_gmem_ptr(dcV), layout_v);
      Tensor O = make_tensor(make_gmem_ptr(ptrO), layout_o);
      Tensor exp_sums =
          make_tensor(make_gmem_ptr(ptrExp_sums), layout_exp_sums);
      Tensor max_logits =
          make_tensor(make_gmem_ptr(ptrMax_logits), layout_max_logits);
      Tensor sinks = make_tensor(
          make_gmem_ptr(const_cast<ElementSink*>(p.sm_sink)), layout_sink);

      // O accumulator types
      FragA tArA;
      FragARow tA_max, tA_sum;

      // Main loop
      int l_coord = is_var_len ? 0 : idx_b;

      int start_blk = kv_split_offset;
      int end_blk = kv_split_offset + num_effective_kv_blocks;

      CollectiveMainloop mainloop(params.mainloop, shared_storage.mainloop);
      bool has_large_surface = kXeMaxSurfaceRows < total_seqlen_kv;
      if (has_large_surface) {
        mainloop.template operator()<true>(
            Q(_, _, head, l_coord),
            K(_, _, head, l_coord),
            V(_, _, head, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            idx_b,
            start_blk,
            end_blk,
            k_blocks,
            thr_id,
            seq_len,
            full_tile_offset,
            discard_seq_coord);
      } else {
        mainloop.template operator()<false>(
            Q(_, _, head, l_coord),
            K(_, _, head, l_coord),
            V(_, _, head, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            idx_b,
            start_blk,
            end_blk,
            k_blocks,
            thr_id,
            seq_len,
            full_tile_offset,
            discard_seq_coord);
      }

      if constexpr (
          !is_empty_v<MainloopSharedStorage> &&
          !is_empty_v<EpilogueSharedStorage>) {
        sycl::group_barrier(get_work_group<3>());
      }

      // Epilogue
      CollectiveEpilogue epilogue{params.epilogue, shared_storage.epilogue};
      if constexpr (Sink) {
        auto sinks_per_kv = sinks(head, _);
        epilogue(
            O(_, _, head, idx_kv_split, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            thr_id,
            exp_sums(_, _, head, l_coord),
            max_logits(_, _, head, l_coord),
            idx_kv_split,
            head_group_q,
            sinks_per_kv,
            num_kv_splits,
            is_single_split);
      } else {
        epilogue(
            O(_, _, head, idx_kv_split, l_coord),
            tArA,
            tA_max,
            tA_sum,
            blk_qv,
            thr_id,
            exp_sums(_, _, head, l_coord),
            max_logits(_, _, head, l_coord),
            idx_kv_split,
            head_group_q,
            sinks,
            num_kv_splits,
            is_single_split);
      }
    }
  }
};

template <class ProblemShape_, class TileScheduler_, class FMHAKernel_>
class ReduceSplitK {
 public:
  using ProblemShape = ProblemShape_;
  using VariableLength = cutlass::fmha::collective::VariableLength;
  static constexpr bool is_var_len =
      cutlass::fmha::collective::is_variable_length_v<
          typename ProblemShape::SeqLenType>;
  using TileScheduler = TileScheduler_;
  static_assert(
      is_same_v<
          TileScheduler,
          cutlass::fmha::kernel::XeReduceSplitKTileScheduler>,
      "ReduceSplitK kernel requires XeReduceSplitKTileScheduler");
  using TileSchedulerParams = typename TileScheduler::Params;

  using ElementO = typename FMHAKernel_::ElementO;
  using StrideO = typename FMHAKernel_::StrideO;
  using TileShapeO = typename FMHAKernel_::TileShapeO;
  using TileShapeQK = typename FMHAKernel_::TileShapeQK;

  using ElementLSE = typename FMHAKernel_::ElementLSE;

  using SGPerWG = typename FMHAKernel_::SGPerWG;

  // num values (head_dim) processed by each thread
  constexpr static int num_vals_per_thread =
      int(get<1>(TileShapeO{}) / (SGPerWG::value * intel::sg_size));

  //
  // Types
  //

  struct KernelArguments {
    ProblemShape shape;
    // outputs:
    ElementO* O;
    StrideO dO;
    // below are inputs
    // TODO: whether same dtype as output or accum?
    const ElementO* Oaccum;
    StrideO dOaccum;
    const ElementLSE* exp_sums;
    StrideO dExp_sums;
    const ElementLSE* max_logits;
    StrideO dMax_logits;
    int window_size_left = -1;

    // per-batch mask: true = prefill, false = decode; nullptr = process all
    const bool* is_prefill;
    const int* splits_per_seq = nullptr;  // per-seq split counts
  };
  using KernelParams = KernelArguments;

  struct Arguments {
    KernelArguments kernel{};
    KernelHardwareInfo hw_info{};
    int num_kv_splits = -1;  // no split by default
  };

  /// Params structure
  struct Params {
    KernelParams kernel;
    TileSchedulerParams scheduler;
  };

  struct SharedStorage {
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits>
        max_logits_slm_array;
    cutlass::Array<ElementLSE, FMHAKernel_::max_num_kv_splits>
        exp_sums_slm_array;
  };

  static constexpr int SharedStorageSize =
      is_empty_v<SharedStorage> ? size_t(0) : sizeof(SharedStorage);

 public:
  static Params
  to_underlying_arguments(Arguments const& args, void* workspace) {
    return {
        args.kernel,
        TileScheduler::to_underlying_arguments(
            args.kernel.shape, args.hw_info, TileShapeO{}, args.num_kv_splits)};
  }

  static bool can_implement(Arguments const& args) {
    // only support decode
    if (!is_var_len && args.kernel.shape.seq_len_qo > 1) {
      return false;
    }

    if (args.num_kv_splits > FMHAKernel_::max_num_kv_splits) {
      return false;
    }
    return true;
  }

  static int get_workspace_size(Arguments const& args) { return 0; }

  static cutlass::Status initialize_workspace(
      Arguments const& args,
      void* workspace = nullptr,
      cudaStream_t stream = nullptr,
      CudaHostAdapter* cuda_adapter = nullptr) {
    return Status::kSuccess;
  }

  static dim3 get_grid_shape(Params const& params) {
    return TileScheduler::template get_grid_shape<SGPerWG::value>(
        params.scheduler);
  }

  static dim3 get_block_shape() {
    return dim3(SGPerWG::value * intel::sg_size, 1, 1);
  }

  CUTLASS_DEVICE
  Shape<int, int> get_sequence_length_shape(
      ProblemShape const& problem_shape, int const& batch) {
    if constexpr (is_var_len) {
#ifdef B70_VERIFY_MASK
      return Shape<int, int>{
          1, problem_shape.seq_len_kv.cumulative_length[batch]};
#else
      auto q_len = cutlass::fmha::collective::apply_variable_length(
          Shape<VariableLength>{problem_shape.seq_len_qo}, batch);
      return Shape<int, int>{
          get<0>(q_len), problem_shape.seq_len_kv.cumulative_length[batch]};
#endif
    } else {
      return Shape<int, int>{
          problem_shape.seq_len_qo, problem_shape.seq_len_kv};
    }
  }

  /// Perform a reduction
  CUTLASS_DEVICE
  void operator()(Params const& params, char* smem_buf) {
    using namespace sycl::ext::oneapi::this_work_item;

    SharedStorage& shared_storage = *reinterpret_cast<SharedStorage*>(smem_buf);

    auto& p = params.kernel;
    ProblemShape const& s = p.shape;

    int thr_id = int(ThreadIdxX());
    int sub_group_id = thr_id / intel::sg_size;
    int tid_in_sg = thr_id % intel::sg_size;

    TileScheduler tile_scheduler{params.scheduler};
    auto num_kv_splits = params.scheduler.num_kv_splits;

    auto batch_dim = is_var_len ? 1 : s.batch;
    auto num_heads_q = s.num_heads_q;
    auto head_size_vo = s.head_size_vo;

    CUTLASS_PRAGMA_NO_UNROLL
    for (; tile_scheduler.is_valid(); ++tile_scheduler) {
      auto [seq_idx, head_q, idx_b] = tile_scheduler.get_block_coord();

      // Skip prefill batches when is_prefill mask is provided
      if (p.is_prefill != nullptr && p.is_prefill[idx_b]) continue;

      auto sequence_length_shape = get_sequence_length_shape(s, idx_b);
      auto [seq_len_qo, seq_len_kv] = sequence_length_shape;

      // when varlen enabled, use largest seq_len_qo to decide work group num
      if (seq_idx >= seq_len_qo) continue;

      const int k_blocks = cute::ceil_div(seq_len_kv, get<1>(TileShapeQK{}));
      // Sliding window: skip blocks before the window
      constexpr bool LocalMask = FMHAKernel_::CollectiveMainloop::LocalMask;
      const int k_block0 =
          LocalMask ? cute::max(seq_len_kv - 1 - p.window_size_left, 0) /
                          get<1>(TileShapeQK{})
                    : 0;
      const int windowed_k_blocks = k_blocks - k_block0;
      // Per-sequence adaptive split count
      int seq_num_kv_splits = (p.splits_per_seq != nullptr)
                                  ? p.splits_per_seq[idx_b]
                                  : num_kv_splits;

      int num_blocks_per_split =
          cute::ceil_div(windowed_k_blocks, seq_num_kv_splits);

      // is_single_split is a heuristic owned by the FMHA kernel; when the
      // host provides splits_per_seq, the host has already applied the same
      // policy (see build_decode_split_plan in flash_attn_interface.py), so
      // trust it directly. Otherwise (legacy path), mirror the FMHA kernel.
      int effective_splits;
      if (p.splits_per_seq != nullptr) {
        effective_splits = seq_num_kv_splits;
      } else {
        constexpr int tile_n = get<1>(typename FMHAKernel_::TileShapeQK{});
        constexpr int kMinBlocksForSplit = (tile_n <= 64) ? 32 : 128;
        bool is_single_split =
            (seq_num_kv_splits > 1) && (windowed_k_blocks < kMinBlocksForSplit);
        effective_splits = is_single_split ? 1 : seq_num_kv_splits;
      }

      int offset_o = 0, offset_o_accum = 0;
      int offset_exp_sums = 0, offset_max_logits = 0;

      if constexpr (is_var_len) {
        auto qo_cumulative = s.seq_len_qo.cumulative_length;

        offset_o_accum = s.num_heads_q * s.head_size_vo * num_kv_splits *
                         qo_cumulative[idx_b];
        offset_exp_sums = s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];
        offset_max_logits =
            s.num_heads_q * num_kv_splits * qo_cumulative[idx_b];

        offset_o = s.num_heads_q * s.head_size_vo * qo_cumulative[idx_b];
      }

      auto shape_O =
          make_shape(seq_len_qo, head_size_vo, num_heads_q, batch_dim);
      auto shape_Oaccum = is_var_len ? make_shape(
                                           seq_len_qo,
                                           head_size_vo,
                                           num_heads_q * num_kv_splits,
                                           batch_dim)
                                     : make_shape(
                                           seq_len_qo,
                                           head_size_vo,
                                           num_heads_q * num_kv_splits,
                                           batch_dim);

      auto shape_exp_sums =
          make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch_dim);
      auto shape_max_logits =
          make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch_dim);

      auto dcOaccum = const_cast<ElementO*>(p.Oaccum + offset_o_accum);
      auto ptrExp_sums = const_cast<ElementLSE*>(p.exp_sums + offset_exp_sums);
      auto ptrMax_logits =
          const_cast<ElementLSE*>(p.max_logits + offset_max_logits);
      auto ptrO = p.O + offset_o;

      auto stride_o = is_var_len
                          ? cutlass::make_cute_packed_stride(StrideO{}, shape_O)
                          : p.dO;
      auto stride_o_accum =
          is_var_len ? cutlass::make_cute_packed_stride(StrideO{}, shape_Oaccum)
                     : p.dOaccum;
      auto stride_exp_sums = is_var_len ? cutlass::make_cute_packed_stride(
                                              StrideO{}, shape_exp_sums)
                                        : p.dExp_sums;
      auto stride_max_logits = is_var_len ? cutlass::make_cute_packed_stride(
                                                StrideO{}, shape_max_logits)
                                          : p.dMax_logits;

      Tensor Oaccum = make_tensor(
          make_gmem_ptr(dcOaccum), make_layout(shape_Oaccum, stride_o_accum));
      Tensor O =
          make_tensor(make_gmem_ptr(ptrO), make_layout(shape_O, stride_o));

      Tensor exp_sums = make_tensor(
          make_gmem_ptr(ptrExp_sums),
          make_layout(shape_exp_sums, stride_exp_sums));
      Tensor max_logits = make_tensor(
          make_gmem_ptr(ptrMax_logits),
          make_layout(shape_max_logits, stride_max_logits));

      int l_coord = is_var_len ? 0 : idx_b;

      // Step 1: reduce max logits across different partitions
      // store into SLM for later use

      ElementLSE global_max_logits{
          cutlass::platform::numeric_limits<ElementLSE>::lowest()};
      ElementLSE global_exp_sums{0};
      // only first subgroup participates
      if (thr_id < effective_splits &&
          thr_id * num_blocks_per_split < windowed_k_blocks) {
        ElementLSE cur_max_logit = max_logits(seq_idx, thr_id, head_q, l_coord);
        global_max_logits = sycl::max(global_max_logits, cur_max_logit);
        shared_storage.max_logits_slm_array[thr_id] = cur_max_logit;

        ElementLSE cur_exp_sum = exp_sums(seq_idx, thr_id, head_q, l_coord);
        shared_storage.exp_sums_slm_array[thr_id] = cur_exp_sum;
      }

      // barrier for SLM writes finished
      sycl::group_barrier(get_work_group<3>());

      // reduce across wg
      global_max_logits = reduce_over_group(
          get_work_group<1>(), global_max_logits, sycl::maximum<>());

      // broadcast to all other threads
      global_max_logits =
          sycl::group_broadcast(get_work_group<1>(), global_max_logits, 0);

      for (int idx = thr_id; idx < s.head_size_vo;
           idx += SGPerWG::value * intel::sg_size) {
        ElementLSE acc = 0;
        global_exp_sums = 0;
        for (int i = 0; i < effective_splits; ++i) {
          if (i * num_blocks_per_split >= windowed_k_blocks) {
            break;
          }
          ElementLSE local_max_logit = shared_storage.max_logits_slm_array[i];
          ElementLSE local_exp_sum = shared_storage.exp_sums_slm_array[i];

          // Skip splits with no valid data (short sequences treated as
          // single-split have exp_sums=0 / max_logits=-inf for unused splits).
          if (local_exp_sum <= ElementLSE(0)) continue;

          ElementLSE rescale =
              sycl::native::exp2(local_max_logit - global_max_logits);

          // Partial outputs are unnormalized (not divided by exp_sum in the
          // epilogue), so combine them directly with the rescale factor.
          ElementLSE o_accum_val = static_cast<ElementLSE>(
              Oaccum(seq_idx, idx, i * num_heads_q + head_q, l_coord));
          acc += o_accum_val * rescale;

          // update global exp sum
          global_exp_sums += local_exp_sum * rescale;
        }

        ElementLSE inv_global_exp_sums = 1. / global_exp_sums;

        acc *= inv_global_exp_sums;
        O(seq_idx, idx, head_q, l_coord) = static_cast<ElementO>(acc);
      }
    }
  }
};

}  // namespace cutlass::fmha::kernel
