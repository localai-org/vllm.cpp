#include "xpu_common.h"
#include "xpu_kernels.h"
#include "csrc/xpu/attn/xe_2/chunk_prefill.hpp"

#include <cstdlib>
#include <cstdint>
#include <limits>
#include <string_view>

namespace vt::xpu {
namespace {
using namespace cute;
using Xe2Fp8Prefill = FMHAConfig<
    Shape<_256, _32, _32>, Shape<_256, _32, _32>, Shape<_256, _256>,
    Layout<Shape<_32, _1, _1>>, void, 2, true, true, false, false, false,
    half_t, float_e4m3_t, float_e4m3_t, half_t>;
}  // namespace

bool PagedAttentionXe2PrefillKernel(Queue& q, Tensor& out, const Tensor& query,
    const Tensor& key_cache, const Tensor& value_cache, const Tensor& block_table,
    const Tensor& seq_lens, const Tensor& query_start_loc,
    const PagedAttentionArgs& args) {
  const char* setting = std::getenv("VT_XPU_XE2_PREFILL");
  if (setting && std::string_view(setting) != "1") return false;
  const auto device = NativeQueue(q).get_device();
  constexpr int dim = 256, q_heads = 24, kv_heads = 4;
  const auto tokens = query.shape[0], page = key_cache.shape[1];
  // The native causal offset uses KV length minus query length. Limit this
  // qualified range to one request, at most 8K KV and a 4K existing prefix;
  // the remaining shape/layout checks below still apply.
  const int64_t prefix = args.max_seq_len - tokens;
  const bool initial_prefill = prefix == 0 && tokens >= 2048 && tokens <= 8192;
  const bool continuation = prefix > 0 && prefix <= 4096 &&
      tokens >= 2048 && tokens <= 4096 && args.max_seq_len <= 8192;
  if (continuation) {
    const char* continuation_setting = std::getenv("VT_XPU_XE2_CONTINUATION");
    if (continuation_setting && std::string_view(continuation_setting) != "1")
      return false;
  }
  if (!(initial_prefill || continuation) || query.rank != 3 || out.rank != 3 ||
      query.dtype != DType::kF16 || out.dtype != DType::kF16 ||
      query.shape[1] != q_heads || query.shape[2] != dim ||
      out.shape[0] != query.shape[0] || out.shape[1] != query.shape[1] ||
      out.shape[2] != query.shape[2] ||
      query.stride[0] != q_heads * dim || query.stride[1] != dim ||
      query.stride[2] != 1 || out.stride[0] != q_heads * dim ||
      out.stride[1] != dim || out.stride[2] != 1 ||
      key_cache.rank != 4 || value_cache.rank != 4 ||
      key_cache.dtype != DType::kI8 || value_cache.dtype != DType::kI8 ||
      args.kv_cache_dtype != Fp8KVCacheDataType::kFp8E4M3 ||
      key_cache.shape[0] != value_cache.shape[0] ||
      key_cache.shape[1] != value_cache.shape[1] ||
      key_cache.shape[2] != kv_heads || value_cache.shape[2] != kv_heads ||
      key_cache.shape[3] != dim || value_cache.shape[3] != dim ||
      (page != 64 && page != 1600 && page != 1664) ||
      key_cache.stride[1] != kv_heads * dim ||
      value_cache.stride[1] != kv_heads * dim ||
      key_cache.stride[2] != dim || value_cache.stride[2] != dim ||
      key_cache.stride[3] != 1 || value_cache.stride[3] != 1 ||
      key_cache.stride[0] != value_cache.stride[0] ||
      key_cache.stride[0] % key_cache.stride[1] != 0 ||
      key_cache.shape[0] <= 0 ||
      key_cache.stride[0] / key_cache.stride[1] < page ||
      key_cache.shape[0] > std::numeric_limits<int>::max() /
          (key_cache.stride[0] / key_cache.stride[1]) ||
      block_table.rank != 2 || block_table.dtype != DType::kI32 ||
      block_table.shape[0] != 1 || block_table.stride[1] != 1 ||
      block_table.shape[1] < (args.max_seq_len + page - 1) / page ||
      block_table.shape[1] > std::numeric_limits<int>::max() ||
      seq_lens.rank != 1 || seq_lens.dtype != DType::kI32 ||
      seq_lens.Numel() != 1 || query_start_loc.rank != 1 ||
      query_start_loc.dtype != DType::kI32 || query_start_loc.Numel() != 2 ||
      !args.causal || args.window_size || args.logits_soft_cap != 0 ||
      args.k_scale != 1.0f || args.v_scale != 1.0f ||
      args.scale != 1.0f / 16 ||
      (reinterpret_cast<uintptr_t>(query.data) & 15) ||
      (reinterpret_cast<uintptr_t>(out.data) & 15) ||
      (reinterpret_cast<uintptr_t>(key_cache.data) & 15) ||
      (reinterpret_cast<uintptr_t>(value_cache.data) & 15) ||
      !device.has(sycl::aspect::ext_intel_device_id) ||
      device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
      std::string_view(__VERSION__) !=
          "Intel(R) oneAPI DPC++/C++ Compiler 2026.1.1 (2026.1.1.20260724)" ||
      device.get_info<sycl::info::device::driver_version>() != "1.17.39758+10")
    return false;

  chunk_prefill_args_t donor{};
  donor.query = query.data;
  donor.key = key_cache.data;
  donor.value = value_cache.data;
  donor.out = out.data;
  donor.block_table = block_table.data;
  donor.cu_seqlens_q = query_start_loc.data;
  donor.cu_seqlens_k = seq_lens.data;
  donor.max_queries = tokens;
  donor.max_keys = args.max_seq_len;
  donor.total_seqlen_q = tokens;
  const auto page_row_step = key_cache.stride[0] / key_cache.stride[1];
  donor.total_seqlen_k = key_cache.shape[0] * page_row_step;
  donor.k_scale = nullptr;  // Vendored policy interprets null as unit scale.
  donor.v_scale = nullptr;
  donor.sm_scale = args.scale;
  donor.batch_size = 1;
  donor.num_heads_q = q_heads;
  donor.num_heads_k = kv_heads;
  donor.head_size = dim;
  donor.max_blocks_per_seq = block_table.shape[1];
  donor.block_size = page;
  donor.is_varlen = true;
  donor.is_paged = true;
  donor.is_causal = true;
  donor.page_stride_elements = page_row_step;
  donor.q_stride_seq = query.stride[0];
  donor.q_stride_heads = query.stride[1];
  donor.o_stride_seq = out.stride[0];
  donor.o_stride_heads = out.stride[1];
  donor.k_stride_seq = key_cache.stride[1];
  donor.k_stride_heads = key_cache.stride[2];
  donor.v_stride_seq = value_cache.stride[1];
  donor.v_stride_heads = value_cache.stride[2];
  const auto event = Xe2Fp8Prefill::kernel_dispatch(NativeQueue(q), donor);
  RecordProfileEvent(q, "attention_prefill_xe2", event);
  return true;
}
}  // namespace vt::xpu
