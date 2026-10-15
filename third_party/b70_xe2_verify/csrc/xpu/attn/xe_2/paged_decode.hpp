// SPDX-License-Identifier: Apache-2.0
// Narrow Torch-free derivative of vllm-xpu-kernels 6d92b1bfbf32767ecda8e819613eb151e70030ad.
#pragma once

#include "flash_attention_v2/collective/fmha_fusion.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <cute/tensor.hpp>
#include <stdexcept>

#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

#include "collective/chunk_prefill_scheduler.hpp"
#include "collective/chunk_prefill_epilogue.hpp"
#include "kernel/paged_decode_kernel.hpp"

#define TORCH_CHECK(condition, ...) \
  do { if (!(condition)) throw std::runtime_error("B70 M04 invalid shape or stride"); } while (false)

using namespace cute;

struct paged_decode_args_t {
  void* query;
  void* key;
  void* value;
  void* out;
  void* tem_out;
  void* exp_sums;
  void* max_logits;
  void* block_table;
  void* cu_seqlens_q;
  void* cu_seqlens_k;
  int max_queries;
  int max_keys;
  int total_seqlen_q;
  int total_seqlen_k;
  void* k_scale = nullptr;
  void* v_scale = nullptr;
  float sm_scale;
  void* sm_sink;
  int batch_size;
  int num_heads_q;
  int num_heads_k;
  int head_size;
  int v_head_size;
  int max_blocks_per_seq;
  int block_size;
  int window_size_left = -1;
  int window_size_right = -1;
  bool is_varlen = false;
  bool is_paged = false;
  bool is_causal = false;
  bool is_local = false;
  bool is_sink = false;
  int num_kv_splits = 1;
  const int* splits_per_seq =
      nullptr;  // per-seq split counts; null => use global num_kv_splits
  const void* work_list =
      nullptr;        // DecodeWorkItem[total_wgs]; null => old grid
  int total_wgs = 0;  // sum(splits_per_seq); 0 => old grid
  // KV cache strides [num_blocks, block_size, num_heads_kv, head_size]
  int64_t k_stride_page = 0;
  int64_t k_stride_seq = 0;
  int64_t k_stride_heads = 0;
  int64_t v_stride_page = 0;
  int64_t v_stride_seq = 0;
  int64_t v_stride_heads = 0;
  // per-batch mask: true = prefill, false = decode; nullptr = process all
  void* is_prefill = nullptr;
  // Q strides. Varlen Q is [total_seq, num_heads_q, head_size]; non-varlen Q
  // is [batch, num_heads_q, seq, head_size]. The kernel always assumes the
  // head_size dim has stride 1 (checked at the API boundary). The other
  // strides are passed explicitly so non-contiguous Q (slice/permute/sliced
  // buffer) is supported. q_stride_batch is unused for varlen (batch=1) and
  // populated as 0 in that case.
  int64_t q_stride_seq = 0;
  int64_t q_stride_heads = 0;
  int64_t q_stride_batch = 0;
  int page_stride_elements = 0;
};

template <class FMHAKernel, class ReductionSplitKernel, bool isVarLen>
struct DecodeKernelLauncher {
  using StrideQ = typename FMHAKernel::StrideQ;
  using StrideK = typename FMHAKernel::StrideK;
  using StrideV = typename FMHAKernel::StrideV;
  using StrideO = typename FMHAKernel::StrideO;

  using ElementQ = typename FMHAKernel::ElementQ;
  using ElementK = typename FMHAKernel::ElementK;
  using ElementV = typename FMHAKernel::ElementV;
  using ElementO = typename FMHAKernel::ElementO;
  using ElementLSE = typename FMHAKernel::ElementLSE;

  using CollectiveMainloop = typename FMHAKernel::CollectiveMainloop;
  using ElementS = typename CollectiveMainloop::ElementS;

  using ProblemShapeType = cutlass::fmha::kernel::DecodeProblemShape<isVarLen>;
  using ProblemShapeTypeInit = cutlass::fmha::kernel::DecodeProblemShape<false>;

  /// Initialization
  StrideQ stride_Q;
  StrideK stride_K;
  StrideV stride_V;
  StrideO stride_O;
  StrideO stride_Oaccum;
  StrideO stride_exp_sums;
  StrideO stride_max_logits;

  int num_kv_splits;

  ProblemShapeType initialize(const paged_decode_args_t& args) {
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
        args.v_head_size;

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

    auto seq_len_qo = shape_init.seq_len_qo;
    auto seq_len_kv = shape_init.seq_len_kv;

    num_kv_splits = args.num_kv_splits;

    if (args.q_stride_seq > 0 || args.q_stride_heads > 0 ||
        args.q_stride_batch > 0) {
      // Use actual strides from the Q tensor (supports non-contiguous Q
      // such as strided row slices, permuted/transposed views, or slices
      // of a wider buffer). The head_dim stride is _1 and is enforced at
      // the API boundary.
      constexpr int64_t kIntMax =
          static_cast<int64_t>(std::numeric_limits<int>::max());
      TORCH_CHECK(
          args.q_stride_seq <= kIntMax && args.q_stride_heads <= kIntMax &&
              args.q_stride_batch <= kIntMax,
          "Q stride exceeds int32 max (",
          kIntMax,
          "): q_stride_seq=",
          args.q_stride_seq,
          " q_stride_heads=",
          args.q_stride_heads,
          " q_stride_batch=",
          args.q_stride_batch);
      stride_Q = StrideQ{
          static_cast<int>(args.q_stride_seq),
          _1{},
          static_cast<int>(args.q_stride_heads),
          static_cast<int>(args.q_stride_batch)};
    } else {
      stride_Q = cutlass::make_cute_packed_stride(
          StrideQ{},
          cute::make_shape(seq_len_qo, head_size_qk, num_heads_q, batch));
    }
    if (args.k_stride_seq > 0) {
      // Use actual strides from KV cache tensors (supports non-contiguous
      // layouts such as MLA combined KV cache)
      constexpr int64_t kIntMax =
          static_cast<int64_t>(std::numeric_limits<int>::max());
      TORCH_CHECK(
          args.k_stride_seq <= kIntMax && args.k_stride_heads <= kIntMax &&
              args.k_stride_page <= kIntMax && args.v_stride_seq <= kIntMax &&
              args.v_stride_heads <= kIntMax && args.v_stride_page <= kIntMax,
          "KV cache stride exceeds int32 max (",
          kIntMax,
          "): k_stride_seq=",
          args.k_stride_seq,
          " k_stride_heads=",
          args.k_stride_heads,
          " k_stride_page=",
          args.k_stride_page,
          " v_stride_seq=",
          args.v_stride_seq,
          " v_stride_heads=",
          args.v_stride_heads,
          " v_stride_page=",
          args.v_stride_page);
      stride_K = StrideK{
          static_cast<int>(args.k_stride_seq),
          _1{},
          static_cast<int>(args.k_stride_heads),
          static_cast<int>(args.k_stride_page)};
      stride_V = StrideV{
          _1{},
          static_cast<int>(args.v_stride_seq),
          static_cast<int>(args.v_stride_heads),
          static_cast<int>(args.v_stride_page)};
    } else {
      stride_K = cutlass::make_cute_packed_stride(
          StrideK{},
          cute::make_shape(seq_len_kv, head_size_qk, num_heads_kv, batch));
      stride_V = cutlass::make_cute_packed_stride(
          StrideV{},
          cute::make_shape(head_size_vo, seq_len_kv, num_heads_kv, batch));
    }
    stride_O = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, head_size_vo, num_heads_q, batch));
    stride_Oaccum = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(
            seq_len_qo, head_size_vo, num_heads_q * num_kv_splits, batch));

    stride_exp_sums = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch));

    stride_max_logits = cutlass::make_cute_packed_stride(
        StrideO{},
        cute::make_shape(seq_len_qo, num_kv_splits, num_heads_q, batch));

    return shape;
  }

  cutlass::Status
  run(sycl::queue& queue,
      const paged_decode_args_t& args,
      const cutlass::KernelHardwareInfo& hw_info) {
    ProblemShapeType shape = initialize(args);

    typename FMHAKernel::Arguments arguments{
        {
            shape,
            reinterpret_cast<ElementQ*>(args.query),
            stride_Q,
            reinterpret_cast<ElementK*>(args.key),
            stride_K,
            reinterpret_cast<ElementV*>(args.value),
            stride_V,
            reinterpret_cast<ElementO*>(args.tem_out),
            stride_Oaccum,
            reinterpret_cast<ElementLSE*>(args.exp_sums),
            stride_exp_sums,
            reinterpret_cast<ElementLSE*>(args.max_logits),
            stride_max_logits,
            reinterpret_cast<ElementQ*>(args.sm_sink),
            static_cast<const bool*>(args.is_prefill),
            args.splits_per_seq,
        },
        {args.sm_scale,
         args.k_scale,
         args.v_scale,
         static_cast<int*>(args.block_table),
         args.block_size,
         args.max_blocks_per_seq,
         args.total_seqlen_k,
         args.window_size_left,
         args.window_size_right,
         // page_stride_elements: physical stride between paged blocks in
         // seq-position units. It includes interleaved and cross-layer gaps.
         args.page_stride_elements},
        {},
        hw_info,
        args.num_kv_splits};

    typename ReductionSplitKernel::Arguments reduce_arg{
        {shape,
         reinterpret_cast<ElementO*>(args.out),
         stride_O,
         reinterpret_cast<ElementO*>(args.tem_out),
         stride_Oaccum,
         reinterpret_cast<ElementLSE*>(args.exp_sums),
         stride_exp_sums,
         reinterpret_cast<ElementLSE*>(args.max_logits),
         stride_max_logits,
         args.window_size_left,
         static_cast<const bool*>(args.is_prefill),
         args.splits_per_seq},
        hw_info,
        args.num_kv_splits};

    // Define device-global scratch memory
    size_t workspace_size = FMHAKernel::get_workspace_size(arguments);
    size_t reduce_workspace_size =
        ReductionSplitKernel::get_workspace_size(reduce_arg);
    if (workspace_size != 0 || reduce_workspace_size != 0)
      throw std::runtime_error("B70 M04 unexpectedly needs a global workspace");

    if (!FMHAKernel::can_implement(arguments)) {
      std::cout << "Invalid Problem Size: " << args.batch_size << 'x'
                << args.num_heads_q << 'x' << args.max_queries << 'x'
                << args.max_keys << 'x' << args.head_size << 'x'
                << args.head_size << std::endl;
      return cutlass::Status::kErrorInvalidProblem;
    }

    // Initialize the workspace
    FMHAKernel::initialize_workspace(arguments, nullptr);

    // Convert host-side arguments to device-side arguments to be passed to the
    // kernel
    auto params =
        FMHAKernel::to_underlying_arguments(arguments, nullptr);
    auto reduce_params = ReductionSplitKernel::to_underlying_arguments(
        reduce_arg, nullptr);

    // Compact grid: patch scheduler params with work_list if available
    if (args.work_list != nullptr && args.total_wgs > 0) {
      params.scheduler.work_list =
          reinterpret_cast<const cutlass::fmha::kernel::DecodeWorkItem*>(
              args.work_list);
      params.scheduler.total_wgs = args.total_wgs;
      // Override grid.z to compact size
      params.scheduler.grid.z = args.total_wgs * args.num_heads_k;
    }

    ReductionSplitKernel::initialize_workspace(
        reduce_arg, nullptr);
    run(queue, params, reduce_params, args.num_kv_splits > 1);

    return cutlass::Status::kSuccess;
  }

  static void
  run(sycl::queue& queue,
      typename FMHAKernel::Params params,
      typename ReductionSplitKernel::Params reduce_params,
      bool need_reduce) {
    namespace syclex = sycl::ext::oneapi::experimental;
    namespace intelex = sycl::ext::intel::experimental;

    dim3 const block = FMHAKernel::get_block_shape();
    dim3 const grid = FMHAKernel::get_grid_shape(params);

    // cute::print("Launching FMHAKernel with grid: "); cute::print("%d x %d x
    // %d ", grid.x, grid.y, grid.z); cute::print("and block: ");
    // cute::print("%d x %d x %d\n", block.x, block.y, block.z);

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
    compat::experimental::launch<cutlass::device_kernel<FMHAKernel>>(
        policy, queue, params);

    // event.wait();

    if (need_reduce) {
      dim3 const reduce_grid =
          ReductionSplitKernel::get_grid_shape(reduce_params);
      int reduce_smem_size = ReductionSplitKernel::SharedStorageSize;
      const auto reduce_sycl_block = compat::dim3(block.x, block.y, block.z);
      const auto reduce_sycl_grid =
          compat::dim3(reduce_grid.x, reduce_grid.y, reduce_grid.z);
      compat::experimental::launch_properties launch_props_reduce{
          syclex::work_group_scratch_size(reduce_smem_size),
      };
      compat::experimental::launch_policy reduce_policy{
          reduce_sycl_grid,
          reduce_sycl_block,
          launch_props_reduce,
          kernel_props};

      compat::experimental::launch<
          cutlass::device_kernel<ReductionSplitKernel>>(
          reduce_policy, queue, reduce_params);
      // reduce_event.wait();
    }
  }
};

template <
    typename TileShapeQK,
    typename TileShapePV,
    typename TileShapeOutput,
    typename SubgroupLayoutQK,
    typename SubgroupLayoutPV_, /* void -> default */
    int PipelineStages,
    bool Causal = false,
    bool Local = false,
    bool Sink = false,
    typename ElementQ = bfloat16_t,
    typename ElementK = bfloat16_t,
    typename ElementV = bfloat16_t,
    typename ElementO = float,
    typename MMAOperation_ = void, /* void -> default */
    typename StrideQ = Stride<int, _1, int, int>,
    typename StrideK = Stride<int, _1, int, int>,
    typename StrideV = Stride<_1, int, int, int>,
    typename StrideO = Stride<int, _1, int, int>,
    typename StrideOaccum = Stride<int, _1, int, int>,
    typename GmemTiledCopyQ = void, /* void -> default block 2D */
    typename GmemTiledCopyK = void,
    typename GmemTiledCopyV = void,
    typename GmemTiledCopyO = void>
struct PagedDecodeConfig {
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
  static void run(sycl::queue& queue, const paged_decode_args_t& args) {
    constexpr bool VarLen = true;
    constexpr bool Paged = true;
    cutlass::KernelHardwareInfo hw_info;
    hw_info.sm_count =
        cutlass::KernelHardwareInfo::query_device_multiprocessor_count(
            hw_info.device_id);

    using ProblemShapeType = cutlass::fmha::kernel::DecodeProblemShape<VarLen>;

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
    using TensorO = decltype(make_dummy_tensor(ElementO{}, StrideOaccum{}));
    using TensorLSE = decltype(make_dummy_tensor(float{}, StrideO{}));

    // Mainloop
    using MainloopDispatchPolicy = cutlass::fmha::XeDefault<PipelineStages>;
    using CollectiveMainloop = cutlass::fmha::collective::DecodeFwdMainloop<
        MainloopDispatchPolicy,
        Paged,
        Causal,
        TiledMMAQK,
        TiledMMAPV,
        VTiles,
        TensorQ,
        TensorK,
        TensorV,
        GmemTiledCopyQ,
        GmemTiledCopyK,
        GmemTiledCopyV,
        Local>;

    // Epilogue
    using CollectiveEpilogue = cutlass::fmha::collective::DecodeFwdEpilogue<
        CollectiveMainloop,
        TileShapeOutput,
        TensorO,
        TensorLSE,
        void,
        Sink>;

    using FMHAKernel = cutlass::fmha::kernel::XeFMHAFwdSplitKVKernel<
        ProblemShapeType,
        CollectiveMainloop,
        CollectiveEpilogue,
        Scheduler>;

    using ReduceSplitKernel = cutlass::fmha::kernel::ReduceSplitK<
        ProblemShapeType,
        cutlass::fmha::kernel::XeReduceSplitKTileScheduler,
        FMHAKernel>;

    DecodeKernelLauncher<FMHAKernel, ReduceSplitKernel, VarLen> launcher;

    launcher.run(queue, args, hw_info);
  }

  static void
  kernel_dispatch(sycl::queue& queue, const paged_decode_args_t& args) {
    return run<cutlass::fmha::kernel::DecodeTileScheduler>(queue, args);
  }
};

#undef TORCH_CHECK
