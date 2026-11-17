// SPDX-License-Identifier: Apache-2.0
// Derived from vllm-xpu-kernels 6d92b1bfbf32767ecda8e819613eb151e70030ad,
// csrc/xpu/attn/xe_2/chunk_prefill.hpp. The Torch policy dispatcher was
// omitted; this copy retains the selected kernel configuration and launcher.
#pragma once

#include "cutlass/epilogue/collective/default_epilogue.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/GPU_Clock.hpp"
#include <cute/tensor.hpp>
#include <random>
#include <stdexcept>

#include "cutlass/util/command_line.h"
#include "cutlass/util/device_memory.h"
#include "cutlass/util/reference/device/gemm_complex.h"
#include "cutlass/util/reference/device/tensor_compare.h"

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "collective/chunk_prefill_scheduler.hpp"
#include "collective/chunk_prefill_epilogue.hpp"
#include "kernel/chunk_prefill_kernel.hpp"



using namespace cute;

struct chunk_prefill_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  void* k_scale;
  void* v_scale;
  float sm_scale;
  void* sm_sink;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;
  int max_blocks_per_seq;
  int block_size;
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_varlen = false;
  bool is_paged = false;
  bool is_causal = false;
  bool is_local = false;
  bool is_sink = false;
  // softmax_lse output (nullptr when not requested)
  float* softmax_lse = nullptr;
  int lse_stride = 0;  // stride along head dim (= total_seqlen_q)
  // Q/O strides in CUTLASS order: (seq, head_size=1, heads, batch)
  int q_stride_seq = 0;
  int q_stride_heads = 0;
  int q_stride_batch = 0;
  // K strides in CUTLASS order: (seq, head_size=1, heads, batch)
  int k_stride_seq = 0;
  int k_stride_heads = 0;
  int k_stride_batch = 0;
  // V strides in CUTLASS order: (head_size=1, seq, heads, batch)
  int v_stride_seq = 0;
  int v_stride_heads = 0;
  int v_stride_batch = 0;
  // O strides in CUTLASS order: (seq, head_size=1, heads, batch)
  int o_stride_seq = 0;
  int o_stride_heads = 0;
  int o_stride_batch = 0;
  // per-batch mask: true = prefill, false = decode; nullptr = process all
  void* is_prefill = nullptr;
  int page_stride_elements = 0;
};

template <class FMHAKernel, bool isVarLen>
struct KernelLauncher {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<isVarLen>;
  using ProblemShapeTypeInit = cutlass::fmha::kernel::FMHAProblemShape<false>;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;

  ProblemShapeType initialize(const chunk_prefill_args_t& args) {
    ProblemShapeType shape;
    ProblemShapeTypeInit shape_init;
    auto batch = shape.batch = shape_init.batch = args.batch_size;
    auto num_heads_q = shape.num_heads_q = shape_init.num_heads_q =
        args.num_heads_q;
    auto num_heads_kv = shape.num_heads_kv = shape_init.num_heads_kv =
        args.num_heads_k;
    auto head_size_qk = shape.head_size_qk = shape_init.head_size_qk =
        args.head_size;
    auto head_size_vo = shape.head_size_vo = shape_init.head_size_vo =
        args.head_size;

    if constexpr (isVarLen) {
      batch = shape_init.batch = 1;
      shape_init.seq_len_qo = args.total_seqlen_q;
      shape_init.seq_len_kv = args.total_seqlen_k;

      shape.seq_len_qo =
          cutlass::fmha::collective::VariableLength{args.max_queries};
      shape.seq_len_qo.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_q);
      shape.seq_len_kv =
          cutlass::fmha::collective::VariableLength{args.max_keys};
      shape.seq_len_kv.cumulative_length =
          reinterpret_cast<int*>(args.cu_seqlens_k);
    } else {
      shape.seq_len_qo = shape_init.seq_len_qo = args.max_queries;
      shape.seq_len_kv = shape_init.seq_len_kv = args.max_keys;
    }

    // Use actual tensor strides instead of packed strides
    stride_Q = StrideQ{};
    get<0>(stride_Q) = args.q_stride_seq;
    get<2>(stride_Q) = args.q_stride_heads;
    get<3>(stride_Q) = args.q_stride_batch;

    stride_K = StrideK{};
    get<0>(stride_K) = args.k_stride_seq;
    get<2>(stride_K) = args.k_stride_heads;
    get<3>(stride_K) = args.k_stride_batch;

    stride_V = StrideV{};
    get<1>(stride_V) = args.v_stride_seq;
    get<2>(stride_V) = args.v_stride_heads;
    get<3>(stride_V) = args.v_stride_batch;

    stride_O = StrideO{};
    get<0>(stride_O) = args.o_stride_seq;
    get<2>(stride_O) = args.o_stride_heads;
    get<3>(stride_O) = args.o_stride_batch;

    return shape;
  }

  sycl::event
  run(sycl::queue& queue,
      const chunk_prefill_args_t& args,
      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(args);

    typename FMHAKernel::Arguments arguments{
        {shape,
         reinterpret_cast<ElementQ*>(args.query),
         stride_Q,
         reinterpret_cast<ElementK*>(args.key),
         stride_K,
         reinterpret_cast<ElementV*>(args.value),
         stride_V,
         reinterpret_cast<ElementO*>(args.out),
         stride_O,
         reinterpret_cast<ElementQ*>(args.sm_sink),
         args.softmax_lse,
         args.lse_stride,
         static_cast<const bool*>(args.is_prefill)},
        {args.sm_scale,
         args.k_scale,
         args.v_scale,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k,
         args.window_size_left,
         args.window_size_right,
         args.page_stride_elements},
        {},
        hw_info};

    // The selected forward kernel requests no global workspace. Keep the
    // adapter on VT's queue/device and reject a future policy that changes it.
    if (FMHAKernel::get_workspace_size(arguments) != 0)
      throw std::runtime_error("Xe2 prefill unexpectedly needs a workspace");

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params = FMHAKernel::to_underlying_arguments(arguments, nullptr);

    return run(queue, params);
  }

  static sycl::event run(sycl::queue& queue, typename FMHAKernel::Params params) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // configure smem size and carveout
    int smem_size = FMHAKernel::SharedStorageSize;

    const auto sycl_block = compat::dim3(block.x, block.y, block.z);
    const auto sycl_grid = compat::dim3(grid.x, grid.y, grid.z);

    // Launch parameters depend on whether SYCL compiler supports work-group
    // scratch memory extension
    compat::experimental::launch_properties launch_props{
        syclex::work_group_scratch_size(smem_size),
    };
    compat::experimental::kernel_properties kernel_props{
        syclex::sub_group_size<cute::intel::sg_size>, intelex::grf_size<256>};
    compat::experimental::launch_policy policy{
        sycl_grid, sycl_block, launch_props, kernel_props};
    return compat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
        policy, queue, params);
  }
};

template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_, /* void -> default */
    int PipelineStages,
    bool Paged = false,
    bool Causal = false,
    bool Local = false,
    bool Sink = false,
    bool SoftmaxLSE = false,
    typename ElementQ = bfloat16_t,
    typename ElementK = bfloat16_t,
    typename ElementV = bfloat16_t,
    typename ElementO = bfloat16_t,
    typename MMAOperation_ = void, /* void -> default */
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void, /* void -> default block 2D */
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void>
struct FMHAConfig {
  static constexpr int SGTileQ =
      get<0>(shape_div(TileShapeQK{}, shape(SubgroupLayoutQK{})))();
  using MMAOperation = cute::conditional_t<
      is_void_v<MMAOperation_>,
      XE_DPAS_TT<cute::gcd(SGTileQ, 8), float, ElementQ>,
      MMAOperation_>;
  using SubgroupLayoutPV = cute::conditional_t<
      is_void_v<SubgroupLayoutPV_>,
      decltype(cutlass::fmha::collective::get_sg_layout_pv(SubgroupLayoutQK{})),
      SubgroupLayoutPV_>;

  template <class Scheduler>
  static sycl::event run(sycl::queue& queue, const chunk_prefill_args_t& args) {
    constexpr bool VarLen = true;
    // constexpr bool Paged = true;
    cutlass::KernelHardwareInfo hw_info;

    using ProblemShapeType = cutlass::fmha::kernel::FMHAProblemShape<VarLen>;

    using TiledMMAQK = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapeQK>,
        SubgroupLayoutQK>::TiledMMA;
    using TiledMMAPV = typename TiledMMAHelper<
        MMA_Atom<MMAOperation>,
        Layout<TileShapePV>,
        SubgroupLayoutPV>::TiledMMA;

    static_assert(
        get<0>(TileShapeOutput{}) == get<0>(TileShapePV{}),
        "Output tile and P*V tile have different sizes in Q dimension");
    constexpr int VTiles = get<1>(TileShapeOutput{}) / get<1>(TileShapePV{});

    auto make_dummy_tensor = [&](auto val, auto stride) {
      return make_tensor(
          make_gmem_ptr(&val),
          make_layout(repeat<rank_v<decltype(stride)>>(1), stride));
    };

    using TensorQ = decltype(make_dummy_tensor(ElementQ{}, StrideQ{}));
    using TensorK = decltype(make_dummy_tensor(ElementK{}, StrideK{}));
    using TensorV = decltype(make_dummy_tensor(ElementV{}, StrideV{}));
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideO{}));

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::FMHAFwdMainloop<
        MainloopDispatchPolicy,
        Causal,
        Local,
        Paged,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV>;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::FMHAFwdEpilogue<
        Sink,
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        GmemTiledCopyO>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler,
        SoftmaxLSE>;

    KernelLauncher<FMHAKernel, VarLen> launcher;

    return launcher.run(queue, args, hw_info);
  }

  static sycl::event
  kernel_dispatch(sycl::queue& queue, const chunk_prefill_args_t& args) {
    return run<cutlass::fmha::kernel::XeFHMAIndividualTileScheduler>(
        queue, args);
  }
};
