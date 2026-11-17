#include "xpu_common.h"
#include "xpu_kernels.h"
#include "vt/xpu_paged_decode_policy.h"
#define B70_VERIFY_MASK 1
#include "csrc/xpu/attn/xe_2/paged_decode.hpp"

#include <cstdint>
#include <limits>
#include <string_view>

namespace vt::xpu {
namespace {
using namespace cute;
using DecodeQ8 = PagedDecodeConfig<Shape<_8, _64, _64>,
    Shape<_8, _32, _64>, Shape<_8, _256>, Layout<Shape<_1, _4, _1>>,
    void, 1, false, false, false, half_t, float_e4m3_t, float_e4m3_t, half_t>;
}  // namespace

bool CanUsePagedAttentionXe2Decode(Queue& q, const Tensor& out, const Tensor& query,
    const Tensor& key_cache, const Tensor& value_cache, const Tensor& block_table,
    const Tensor& seq_lens, const Tensor& query_start_loc,
    const PagedAttentionArgs& args) {
  const auto& device = NativeQueue(q).get_device();
  const int64_t page = key_cache.shape[1];
  const auto plan = XpuPagedDecodePlanFromEnvironment(args.max_seq_len, page);
  const int64_t requests = seq_lens.Numel();
  const bool batched = query.rank == 3 && seq_lens.rank == 1 &&
      query_start_loc.rank == 1 && query_start_loc.dtype == DType::kI32 &&
      query_start_loc.Numel() == requests + 1 &&
      XpuPureBatchedQ1Shape(query.shape[0], requests, args.query_start_loc_host) &&
      XpuBatchedQ1Enabled();
  if ((!batched && (requests != 1 || query.shape[0] != 1)) ||
      query.rank != 3 || out.rank != 3 || query.shape[0] != requests ||
      query.dtype != DType::kF16 || out.dtype != DType::kF16 ||
      query.shape[1] != 24 || query.shape[2] != 256 ||
      out.shape[0] != requests || out.shape[1] != 24 || out.shape[2] != 256 ||
      query.stride[0] != 24 * 256 || query.stride[1] != 256 ||
      query.stride[2] != 1 || out.stride[0] != 24 * 256 ||
      out.stride[1] != 256 || out.stride[2] != 1 ||
      key_cache.rank != 4 || value_cache.rank != 4 ||
      key_cache.dtype != DType::kI8 || value_cache.dtype != DType::kI8 ||
      args.kv_cache_dtype != Fp8KVCacheDataType::kFp8E4M3 ||
      key_cache.shape[0] != value_cache.shape[0] ||
      (page != 1600 && page != 1664) || value_cache.shape[1] != page ||
      key_cache.shape[2] != 4 || value_cache.shape[2] != 4 ||
      key_cache.shape[3] != 256 || value_cache.shape[3] != 256 ||
      (key_cache.stride[1] != 4 * 256 && key_cache.stride[1] != 4 * 512) ||
      value_cache.stride[1] != key_cache.stride[1] ||
      key_cache.stride[2] != key_cache.stride[1] / 4 ||
      value_cache.stride[2] != key_cache.stride[2] ||
      key_cache.stride[3] != 1 || value_cache.stride[3] != 1 ||
      key_cache.stride[0] != value_cache.stride[0] ||
      key_cache.stride[0] % key_cache.stride[1] != 0 ||
      block_table.rank != 2 || block_table.dtype != DType::kI32 ||
      block_table.shape[0] != requests || block_table.stride[1] != 1 ||
      (batched && block_table.stride[0] != block_table.shape[1]) ||
      seq_lens.rank != 1 || seq_lens.dtype != DType::kI32 ||
      (batched && (seq_lens.stride[0] != 1 || query_start_loc.stride[0] != 1 ||
                   key_cache.stride[0] / key_cache.stride[1] < page)) ||
      query_start_loc.rank != 1 || query_start_loc.dtype != DType::kI32 ||
      query_start_loc.Numel() != requests + 1 ||
      plan.max_keys == 0 || block_table.shape[1] < (plan.max_keys + page - 1) / page ||
      !args.causal || args.window_size || args.logits_soft_cap != 0 ||
      args.k_scale != 1.0f || args.v_scale != 1.0f || args.scale != 1.0f / 16 ||
      (reinterpret_cast<uintptr_t>(query.data) & 15) ||
      (reinterpret_cast<uintptr_t>(out.data) & 15) ||
      (reinterpret_cast<uintptr_t>(key_cache.data) & 15) ||
      (reinterpret_cast<uintptr_t>(value_cache.data) & 15) ||
      !device.has(sycl::aspect::ext_intel_device_id) ||
      device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
      std::string_view(__VERSION__) !=
          "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" ||
      device.get_info<sycl::info::device::driver_version>() != "1.17.39758+10" ||
      key_cache.shape[0] > std::numeric_limits<int>::max() /
          (key_cache.stride[0] / key_cache.stride[1]))
    return false;

  return true;
}

bool PagedAttentionXe2DecodeKernel(Queue& q, Tensor& out, const Tensor& query,
    const Tensor& key_cache, const Tensor& value_cache, const Tensor& block_table,
    const Tensor& seq_lens, const Tensor& query_start_loc,
    const PagedAttentionArgs& args) {
  if (!CanUsePagedAttentionXe2Decode(q, out, query, key_cache, value_cache,
                                   block_table, seq_lens, query_start_loc, args)) return false;
  const int64_t page = key_cache.shape[1];
  const auto plan = XpuPagedDecodePlanFromEnvironment(args.max_seq_len, page);
  const int64_t requests = seq_lens.Numel();
  const bool multiple = plan.splits > 1;
  constexpr size_t reservation = 16 * 1024 * 1024;
  const size_t temp_per_request = multiple ? size_t(plan.splits) * 24 * 256 * sizeof(uint16_t) : 0;
  const size_t stats_per_request = size_t(plan.splits) * 24 * sizeof(float);
  const size_t per_request = temp_per_request + 2 * stats_per_request;
  // Admission bounds requests to1..4 and splits to1/16/32. Check the sum
  // before multiplying by B, within the actual fixed allocation below.
  if (!per_request || size_t(requests) > reservation / per_request) return false;
  const size_t temp_bytes = size_t(requests) * temp_per_request;
  const size_t stats_bytes = size_t(requests) * stats_per_request;
  // The existing16 MiB graph-owned reservation never moves. Bounded Q1 uses
  // disjoint batched F16 partials / F32 sums / maxima; one split still aliases
  // its sole partial to output. No new host readback or per-token allocation.
  return WithAttentionWorkspace(q, 16 * 1024 * 1024, [&](void* storage) {
    auto* bytes = static_cast<uint8_t*>(storage);
    auto* sums = reinterpret_cast<float*>(bytes + temp_bytes);
    auto* maxima = reinterpret_cast<float*>(bytes + temp_bytes + stats_bytes);
    paged_decode_args_t donor{};
    donor.query = query.data; donor.key = key_cache.data; donor.value = value_cache.data;
    donor.out = out.data; donor.tem_out = multiple ? storage : out.data;
    donor.exp_sums = sums; donor.max_logits = maxima;
    donor.block_table = block_table.data;
    donor.cu_seqlens_q = query_start_loc.data; donor.cu_seqlens_k = seq_lens.data;
    donor.max_queries = 1; donor.max_keys = plan.max_keys;
    donor.total_seqlen_q = requests;
    donor.total_seqlen_k = key_cache.shape[0] *
        (key_cache.stride[0] / key_cache.stride[1]);
    // Null scales denote1 inside the FP8 donor; all other scales are rejected.
    donor.sm_scale = args.scale;
    donor.batch_size = requests; donor.num_heads_q = 24; donor.num_heads_k = 4;
    donor.head_size = 256; donor.v_head_size = 256;
    donor.max_blocks_per_seq = block_table.shape[1]; donor.block_size = page;
    donor.is_varlen = true; donor.is_paged = true; donor.num_kv_splits = plan.splits;
    donor.q_stride_seq = query.stride[0]; donor.q_stride_heads = query.stride[1];
    donor.k_stride_page = key_cache.stride[0]; donor.k_stride_seq = key_cache.stride[1];
    donor.k_stride_heads = key_cache.stride[2];
    donor.v_stride_page = value_cache.stride[0]; donor.v_stride_seq = value_cache.stride[1];
    donor.v_stride_heads = value_cache.stride[2];
    donor.page_stride_elements = key_cache.stride[0] / key_cache.stride[1];
    const bool profile = (multiple || requests > 1) && ProfileQueueEventsEnabled();
    sycl::event begin;
    const auto submitted = profile ? HostProfileClockNs() : 0;
    if (profile) begin = NativeQueue(q).ext_oneapi_submit_barrier();
    DecodeQ8::kernel_dispatch(NativeQueue(q), donor);
    if (profile) {
      const auto end = NativeQueue(q).ext_oneapi_submit_barrier();
      RecordProfileSpan(q, requests > 1 ? "attention_decode_xe2_batch" : "attention_decode_xe2_long",
          begin, end, submitted, "bound=" + std::to_string(plan.max_keys) +
          " splits=" + std::to_string(plan.splits) + " requests=" + std::to_string(requests));
    }
  });
}
}  // namespace vt::xpu
