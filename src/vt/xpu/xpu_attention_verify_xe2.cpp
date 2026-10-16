#include "xpu_common.h"
#include "xpu_kernels.h"
#define B70_VERIFY_MASK 1
#include "csrc/xpu/attn/xe_2/paged_decode.hpp"

#include <cstdint>
#include <limits>
#include <string_view>

namespace vt::xpu {
namespace {
using namespace cute;
using VerifyQ8 = PagedDecodeConfig<Shape<_8, _64, _64>,
    Shape<_8, _32, _64>, Shape<_8, _256>, Layout<Shape<_1, _4, _1>>,
    void, 1, true, false, false, half_t, float_e4m3_t, float_e4m3_t, half_t>;

constexpr size_t Align(size_t bytes) { return (bytes + 63) & ~size_t{63}; }
}  // namespace

int64_t PagedAttentionXe2VerifyQueryLength(int64_t tokens, int64_t requests,
                                         const int32_t* host_offsets) {
  if (requests == 1) return tokens >= 2 && tokens <= 5 ? tokens : 0;
  // Original C2/C3/C4 Q4 fixtures qualify uniform partial batches as requests
  // finish. Do not infer uniformity from shape or the speculative routing hint.
  // Missing/ragged host metadata
  // leaves the generic route available without an extra D2H readback.
  if (requests < 2 || requests > 4 || tokens != requests * 4 || !host_offsets) return 0;
  for (int64_t r = 0; r <= requests; ++r)
    if (host_offsets[r] != r * 4) return 0;
  return 4;
}

bool CanUsePagedAttentionXe2Verify(Queue& q, const Tensor& out, const Tensor& query,
    const Tensor& key_cache, const Tensor& value_cache, const Tensor& block_table,
    const Tensor& seq_lens, const Tensor& query_start_loc,
    const PagedAttentionArgs& args) {
  const auto& device = NativeQueue(q).get_device();
  const int64_t tokens = query.shape[0];
  const int64_t page = key_cache.shape[1];
  const int64_t requests = seq_lens.Numel();
  const int64_t rows = PagedAttentionXe2VerifyQueryLength(
      tokens, requests, args.query_start_loc_host);
  if (!rows || query.rank != 3 || out.rank != 3 ||
      query.dtype != DType::kF16 || out.dtype != DType::kF16 ||
      query.shape[1] != 24 || query.shape[2] != 256 ||
      out.shape[0] != tokens || out.shape[1] != 24 || out.shape[2] != 256 ||
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
      key_cache.stride[0] / key_cache.stride[1] < page ||
      block_table.rank != 2 || block_table.dtype != DType::kI32 ||
      block_table.shape[0] != requests || block_table.stride[1] != 1 ||
      (requests > 1 && block_table.stride[0] != block_table.shape[1]) ||
      seq_lens.rank != 1 || seq_lens.dtype != DType::kI32 ||
      query_start_loc.rank != 1 ||
      query_start_loc.dtype != DType::kI32 || query_start_loc.Numel() != requests + 1 ||
      args.max_seq_len < rows ||
      block_table.shape[1] < (args.max_seq_len + page - 1) / page ||
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
      device.get_info<sycl::info::device::driver_version>() != "1.17.39758+10" ||
      key_cache.shape[0] > std::numeric_limits<int>::max() /
          (key_cache.stride[0] / key_cache.stride[1]))
    return false;
  return true;
}

bool PagedAttentionXe2VerifyKernel(Queue& q, Tensor& out, const Tensor& query,
    const Tensor& key_cache, const Tensor& value_cache, const Tensor& block_table,
    const Tensor& seq_lens, const Tensor& query_start_loc,
    const PagedAttentionArgs& args) {
  if (!CanUsePagedAttentionXe2Verify(q, out, query, key_cache, value_cache,
                                    block_table, seq_lens, query_start_loc, args)) return false;
  const int64_t tokens = query.shape[0], page = key_cache.shape[1], requests = seq_lens.Numel();
  const int64_t rows = PagedAttentionXe2VerifyQueryLength(tokens, requests, args.query_start_loc_host);

  const int splits = rows == 2 ? 32 : rows == 3 ? 8 : 16;
  const size_t packed_bytes = size_t(tokens) * 24 * 256 * sizeof(uint16_t);
  const size_t temp_bytes = packed_bytes * splits;
  const size_t stats_bytes = size_t(tokens) * 24 * splits * sizeof(float);
  const size_t total = 2 * Align(packed_bytes) + Align(temp_bytes) +
      2 * Align(stats_bytes) + Align(2 * sizeof(float)) + Align(size_t(requests + 1) * sizeof(int32_t));
  // Split-K may reuse this queue later and needs the full persistent workspace.
  VT_CHECK(total <= 16 * 1024 * 1024, "verification workspace exceeds Split-K allocation");
  return WithAttentionWorkspace(q, 16 * 1024 * 1024, [&](void* storage) {
    auto* ptr = static_cast<unsigned char*>(storage);
    auto* packed_q = reinterpret_cast<uint16_t*>(ptr); ptr += Align(packed_bytes);
    auto* packed_out = reinterpret_cast<uint16_t*>(ptr); ptr += Align(packed_bytes);
    auto* temp = reinterpret_cast<uint16_t*>(ptr); ptr += Align(temp_bytes);
    auto* sums = reinterpret_cast<float*>(ptr); ptr += Align(stats_bytes);
    auto* maxima = reinterpret_cast<float*>(ptr); ptr += Align(stats_bytes);
    auto* scales = reinterpret_cast<float*>(ptr); ptr += Align(2 * sizeof(float));
    auto* packed_offsets = reinterpret_cast<int32_t*>(ptr);
    // Logical {0,Q,...,B*Q} stays caller-owned. Like the original wrapper,
    // donor offsets {0,1,...,B} describe one packed physical row per request.
    NativeQueue(q).parallel_for(sycl::range<1>(requests + 1), [=](sycl::id<1> i) {
      if (i[0] < 2) scales[i[0]] = 1.0f;
      packed_offsets[i[0]] = int32_t(i[0]);
    });
    const auto* src = static_cast<const uint16_t*>(query.data);
    const int count = int(tokens * 24 * 256);
    const auto pack = NativeQueue(q).parallel_for(sycl::range<1>(count),
        [=](sycl::id<1> item) {
      const int index = item[0], dim = index % 256;
      const int request = index / (int(rows) * 24 * 256);
      const int head = (index / 256) % (int(rows) * 24);
      const int kv_head = head / (int(rows) * 6);
      const int row = (head / 6) % int(rows);
      const int group_head = head % 6;
      packed_q[index] = src[((request * int(rows) + row) * 24 +
                             kv_head * 6 + group_head) * 256 + dim];
    });
    RecordProfileEvent(q, "attention_verify_pack", pack);

    paged_decode_args_t donor{};
    donor.query = packed_q;
    donor.key = key_cache.data;
    donor.value = value_cache.data;
    donor.out = packed_out;
    donor.tem_out = temp;
    donor.exp_sums = sums;
    donor.max_logits = maxima;
    donor.block_table = block_table.data;
    donor.cu_seqlens_q = packed_offsets;
    donor.cu_seqlens_k = seq_lens.data;
    donor.max_queries = 1;
    donor.max_keys = args.max_seq_len;
    donor.total_seqlen_q = requests;
    donor.total_seqlen_k = key_cache.shape[0] *
        (key_cache.stride[0] / key_cache.stride[1]);
    donor.k_scale = scales;
    donor.v_scale = scales + 1;
    donor.sm_scale = args.scale;
    donor.batch_size = requests;
    donor.num_heads_q = rows * 24;
    donor.num_heads_k = 4;
    donor.head_size = 256;
    donor.v_head_size = 256;
    donor.max_blocks_per_seq = block_table.shape[1];
    donor.block_size = page;
    donor.is_varlen = true;
    donor.is_paged = true;
    donor.is_causal = true;
    donor.num_kv_splits = splits;
    donor.q_stride_seq = rows * 24 * 256;
    donor.q_stride_heads = 256;
    donor.k_stride_page = key_cache.stride[0];
    donor.k_stride_seq = key_cache.stride[1];
    donor.k_stride_heads = key_cache.stride[2];
    donor.v_stride_page = value_cache.stride[0];
    donor.v_stride_seq = value_cache.stride[1];
    donor.v_stride_heads = value_cache.stride[2];
    donor.page_stride_elements = key_cache.stride[0] / key_cache.stride[1];
    VerifyQ8::kernel_dispatch(NativeQueue(q), donor);

    auto* dst = static_cast<uint16_t*>(out.data);
    const auto unpack = NativeQueue(q).parallel_for(sycl::range<1>(count),
        [=](sycl::id<1> item) {
      const int index = item[0], dim = index % 256;
      const int head = index / 256;
      const int request = head / (int(rows) * 24);
      const int row = (head / 24) % int(rows), kv_head = (head / 6) % 4;
      const int group_head = head % 6;
      dst[index] = packed_out[(((request * 4 + kv_head) * int(rows) + row) * 6 +
                                group_head) * 256 + dim];
    });
    RecordProfileEvent(q, "attention_verify_unpack", unpack);
  });
}
}  // namespace vt::xpu
