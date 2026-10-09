#include "xpu_common.h"
#include "xpu_kernels.h"
#include "vt/paged_attn_route.h"
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

bool PagedAttentionXe2DecodeKernel(Queue& q, Tensor& out, const Tensor& query,
    const Tensor& key_cache, const Tensor& value_cache, const Tensor& block_table,
    const Tensor& seq_lens, const Tensor& query_start_loc,
    const PagedAttentionArgs& args) {
  const auto& device = NativeQueue(q).get_device();
  const int64_t page = key_cache.shape[1];
  if (query.rank != 3 || out.rank != 3 || query.shape[0] != 1 ||
      query.dtype != DType::kF16 || out.dtype != DType::kF16 ||
      query.shape[1] != 24 || query.shape[2] != 256 ||
      out.shape[0] != 1 || out.shape[1] != 24 || out.shape[2] != 256 ||
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
      block_table.shape[0] != 1 || block_table.stride[1] != 1 ||
      seq_lens.rank != 1 || seq_lens.dtype != DType::kI32 || seq_lens.Numel() != 1 ||
      query_start_loc.rank != 1 || query_start_loc.dtype != DType::kI32 ||
      query_start_loc.Numel() != 2 ||
      !PagedAttnXpuShortDecodeBound(args.max_seq_len) || block_table.shape[1] < 1 ||
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

  // One split below16 KV tiles, matching the original short-decode policy.
  // The shared reservation remains16 MiB so later Split-K can reuse it.
  // Queue completion owns it; no host wait, readback, Q packing or scales copy.
  return WithAttentionWorkspace(q, 16 * 1024 * 1024, [&](void* storage) {
    auto* stats = static_cast<float*>(storage);
    paged_decode_args_t donor{};
    donor.query = query.data; donor.key = key_cache.data; donor.value = value_cache.data;
    donor.out = out.data; donor.tem_out = out.data;
    donor.exp_sums = stats; donor.max_logits = stats + 24;
    donor.block_table = block_table.data;
    donor.cu_seqlens_q = query_start_loc.data; donor.cu_seqlens_k = seq_lens.data;
    donor.max_queries = 1; donor.max_keys = args.max_seq_len;
    donor.total_seqlen_q = 1;
    donor.total_seqlen_k = key_cache.shape[0] *
        (key_cache.stride[0] / key_cache.stride[1]);
    // Null scales denote1 inside the FP8 donor; all other scales are rejected.
    donor.sm_scale = args.scale;
    donor.batch_size = 1; donor.num_heads_q = 24; donor.num_heads_k = 4;
    donor.head_size = 256; donor.v_head_size = 256;
    donor.max_blocks_per_seq = block_table.shape[1]; donor.block_size = page;
    donor.is_varlen = true; donor.is_paged = true; donor.num_kv_splits = 1;
    donor.q_stride_seq = query.stride[0]; donor.q_stride_heads = query.stride[1];
    donor.k_stride_page = key_cache.stride[0]; donor.k_stride_seq = key_cache.stride[1];
    donor.k_stride_heads = key_cache.stride[2];
    donor.v_stride_page = value_cache.stride[0]; donor.v_stride_seq = value_cache.stride[1];
    donor.v_stride_heads = value_cache.stride[2];
    donor.page_stride_elements = key_cache.stride[0] / key_cache.stride[1];
    DecodeQ8::kernel_dispatch(NativeQueue(q), donor);
  });
}
}  // namespace vt::xpu
