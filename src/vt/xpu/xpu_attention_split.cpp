#include "xpu_common.h"
#include "xpu_fp8.h"
#include "xpu_kernels.h"
#include "vt/paged_attn_route.h"
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <string_view>

namespace vt::xpu {
bool PagedAttentionSplitKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& key_cache,
                               const Tensor& value_cache, const Tensor& block_table,
                               const Tensor& seq_lens, const Tensor& query_start_loc,
                               const PagedAttentionArgs& args) {
  constexpr int SG = 16, Components = 16, Reuse = 2;
  constexpr size_t Workspace = 16 * 1024 * 1024;
  const int64_t tokens = query.shape[0], heads = query.shape[1], dim = query.shape[2];
  const int64_t kvheads = key_cache.shape[2], ratio = heads / kvheads;
  const int64_t page = key_cache.shape[1], capacity = block_table.shape[1] * page;
  const auto sizes = NativeQueue(q).get_device().get_info<sycl::info::device::sub_group_sizes>();
  const auto device = NativeQueue(q).get_device();
  const bool b70_fp8 = key_cache.dtype == DType::kI8 &&
      args.kv_cache_dtype == Fp8KVCacheDataType::kFp8E4M3 &&
      device.has(sycl::aspect::ext_intel_device_id) &&
      device.get_info<sycl::ext::intel::info::device::device_id>() == 57891;
  // Q32 prefill needs at least 32 packed queries. Keep the 21..31 gap on
  // Split-K for the qualified model shape instead of falling to the generic
  // per-head reference kernel.
  const char* extended_setting = std::getenv("VT_XPU_ATTN_SPLIT_EXTENDED");
  const bool extended_queries = (!extended_setting ||
      std::string_view(extended_setting) == "1") &&
      b70_fp8 && heads == 24 && kvheads == 4 &&
      dim == 256 && (page == 1600 || page == 1664) &&
      args.causal && !args.window_size &&
      args.logits_soft_cap == 0 && args.k_scale == 1.0f &&
      args.v_scale == 1.0f;
  if (tokens < 1 || tokens > (extended_queries ? 31 : 20) ||
      heads > 24 || dim > SG * Components ||
      std::find(sizes.begin(), sizes.end(), SG) == sizes.end()) return false;
  // Short E4M3 contexts need more independent page slices to occupy the B70.
  // F16/BF16 and other devices retain the original 256-token partitioning.
  int span = b70_fp8 ? 32 : 256;
  if (const char* setting = std::getenv("VT_XPU_ATTN_SPLIT_SPAN")) {
    span = std::atoi(setting);
    VT_CHECK(span == 32 || span == 64 || span == 128 || span == 256,
             "VT_XPU_ATTN_SPLIT_SPAN must be 32, 64, 128 or 256");
  }
  // Long B70 FP8 decode benefits from more independent slices. Keep short
  // contexts and other formats on the established partition count.
  const int64_t context = args.max_seq_len > 0 ? args.max_seq_len : capacity;
  int max_parts = b70_fp8 && PagedAttnXpuLongSplitBound(context) ? 256 : 32;
  if (const char* setting = std::getenv("VT_XPU_ATTN_SPLIT_MAX_PARTS")) {
    max_parts = std::atoi(setting);
    VT_CHECK(max_parts == 32 || max_parts == 64 || max_parts == 128 || max_parts == 256,
             "VT_XPU_ATTN_SPLIT_MAX_PARTS must be 32, 64, 128 or 256");
  }
  const int64_t stride = dim + 2, pairs = (ratio + Reuse - 1) / Reuse;
  const int64_t budget_parts = Workspace / (tokens * heads * stride * sizeof(float));
  if (budget_parts < 1) return false;
  // The block table may reserve pages beyond the active sequence. Plan from
  // the active length rounded to a KV page so a padded table cannot inflate
  // Split-K work, while retaining the occupancy of the compact table path.
  const char* active_page_setting = std::getenv("VT_XPU_ATTN_SPLIT_ACTIVE_PAGE_CAP");
  const bool active_page_cap = (!active_page_setting ||
      std::string_view(active_page_setting) == "1") &&
      b70_fp8 && heads == 24 && kvheads == 4 && dim == 256 &&
      (page == 1600 || page == 1664) && args.max_seq_len > 0 && args.causal &&
      !args.window_size && args.logits_soft_cap == 0 &&
      args.k_scale == 1.0f && args.v_scale == 1.0f;
  const int64_t planned_capacity = active_page_cap ?
      std::min(capacity, PagedAttnXpuActivePages(context, page) * page) : capacity;
  const int64_t parts = std::min({int64_t{max_parts}, budget_parts,
      std::max(int64_t{1}, (planned_capacity + span - 1) / span)});
  if (const char* trace = std::getenv("VT_XPU_TRACE_SPLIT_PLAN");
      trace && std::string_view(trace) == "1")
    std::fprintf(stderr,
        "{\"event\":\"xpu_split_plan\",\"tokens\":%lld,"
        "\"active_context\":%lld,\"page\":%lld,"
        "\"block_table_capacity\":%lld,\"planned_capacity\":%lld,"
        "\"span\":%d,\"parts\":%lld}\n",
        static_cast<long long>(tokens), static_cast<long long>(context),
        static_cast<long long>(page), static_cast<long long>(capacity),
        static_cast<long long>(planned_capacity), span,
        static_cast<long long>(parts));
  const View qs(query), kc(key_cache), vc(value_cache), dst(out), bt(block_table);
  const auto* table = static_cast<const int32_t*>(bt.data);
  const auto* lengths = static_cast<const int32_t*>(seq_lens.data);
  const auto* offsets = static_cast<const int32_t*>(query_start_loc.data);
  const float scale = args.scale, cap = args.logits_soft_cap, kscale = args.k_scale, vscale = args.v_scale;
  const bool causal = args.causal;
  const int64_t left = args.window_size ? args.window_size->left : -1;
  const int64_t right = args.window_size ? args.window_size->right : -1;
  return WithAttentionWorkspace(q, Workspace, [&](void* storage) {
    auto* partial = static_cast<float*>(storage);
    const int64_t groups = tokens * kvheads * pairs * parts;
    const auto partial_event = NativeQueue(q).parallel_for(sycl::nd_range<1>(groups * SG, SG),
        [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
      const auto sg = item.get_sub_group();
      const int lane = item.get_local_id(0);
      int64_t group = item.get_group(0), part = group % parts; group /= parts;
      const int64_t pair = group % pairs; group /= pairs;
      const int64_t kh = group % kvheads, token = group / kvheads, head = kh * ratio + pair * Reuse;
      int64_t request = 0;
      while (token >= offsets[request + 1]) ++request;
      const int64_t position = lengths[request] - (offsets[request + 1] - offsets[request]) + token - offsets[request];
      const int64_t first = left < 0 ? 0 : sycl::max(int64_t{0}, position - left);
      int64_t last = causal ? position : int64_t(lengths[request]) - 1;
      if (right >= 0) last = sycl::min(last, position + right);
      const int64_t span = (last - first + 1 + parts - 1) / parts;
      const int64_t begin = first + part * span, end = sycl::min(last + 1, begin + span);
      float qv[Reuse][Components] = {}, acc[Reuse][Components] = {};
      float maximum[Reuse], denominator[Reuse] = {};
      #pragma unroll
      for (int r = 0; r < Reuse; ++r) {
        maximum[r] = -std::numeric_limits<float>::infinity();
        #pragma unroll
        for (int c = 0; c < Components; ++c)
          if (pair * Reuse + r < ratio && lane + c * SG < dim)
            qv[r][c] = Load(qs, (token * heads + head + r) * dim + lane + c * SG);
      }
      for (int64_t key = begin; key < end; ++key) {
        const int64_t block = table[request * bt.stride[0] + (key / page) * bt.stride[1]];
        const int64_t kb = block * kc.stride[0] + (key % page) * kc.stride[1] + kh * kc.stride[2];
        const int64_t vb = block * vc.stride[0] + (key % page) * vc.stride[1] + kh * vc.stride[2];
        float kval[Components], vval[Components];
        #pragma unroll
        for (int c = 0; c < Components; ++c) {
          kval[c] = lane + c * SG < dim ? LoadKV(kc, kb + lane + c * SG, kscale) : 0;
          vval[c] = lane + c * SG < dim ? LoadKV(vc, vb + lane + c * SG, vscale) : 0;
        }
        // Adjacent GQA heads share every K/V load; tail heads are masked.
        #pragma unroll
        for (int r = 0; r < Reuse; ++r) {
          float dot = 0;
          #pragma unroll
          for (int c = 0; c < Components; ++c) dot += qv[r][c] * kval[c];
          float score = sycl::reduce_over_group(sg, dot, sycl::plus<float>()) * scale;
          if (cap > 0) score = cap * sycl::tanh(score / cap);
          const float next = sycl::max(maximum[r], score);
          const float old = sycl::exp(maximum[r] - next), p = sycl::exp(score - next);
          denominator[r] = denominator[r] * old + p;
          #pragma unroll
          for (int c = 0; c < Components; ++c) acc[r][c] = acc[r][c] * old + p * vval[c];
          maximum[r] = next;
        }
      }
      #pragma unroll
      for (int r = 0; r < Reuse; ++r) if (pair * Reuse + r < ratio) {
        auto* result = partial + ((token * heads + head + r) * parts + part) * stride;
        if (lane == 0) { result[dim] = maximum[r]; result[dim + 1] = denominator[r]; }
        #pragma unroll
        for (int c = 0; c < Components; ++c) if (lane + c * SG < dim) result[lane + c * SG] = acc[r][c];
      }
    });
    RecordProfileEvent(q, "attention_split_partial", partial_event);
    const char* reduce_setting = std::getenv("VT_XPU_ATTN_SPLIT_REDUCE");
    const std::string_view reduce_mode = reduce_setting ? reduce_setting : "auto";
    VT_CHECK(reduce_mode == "auto" || reduce_mode == "scalar" ||
                 reduce_mode == "cooperative",
             "VT_XPU_ATTN_SPLIT_REDUCE must be auto, scalar or cooperative");
    if (b70_fp8 && args.kv_cache_dtype == Fp8KVCacheDataType::kFp8E4M3 &&
        heads == 24 && dim == 256 && reduce_mode != "scalar") {
      constexpr int ReduceLanes = 64;
      const int64_t component_groups = (dim + ReduceLanes - 1) / ReduceLanes;
      const int64_t groups = tokens * heads * component_groups;
      const auto reduce_event = NativeQueue(q).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> weights(sycl::range<1>(256), h);
        h.parallel_for(sycl::nd_range<1>(groups * ReduceLanes, ReduceLanes),
            [=](sycl::nd_item<1> item) {
          const auto group = item.get_group(0);
          const int64_t row = group / component_groups;
          const int64_t component = group % component_groups * ReduceLanes +
                                    item.get_local_id(0);
          const int lane = item.get_local_id(0);
          const auto* src = partial + row * parts * stride;
          float local_max = -std::numeric_limits<float>::infinity();
          for (int p = lane; p < parts; p += ReduceLanes)
            if (src[p * stride + dim + 1] > 0)
              local_max = sycl::max(local_max, src[p * stride + dim]);
          const auto workgroup = item.get_group();
          const float maximum = sycl::reduce_over_group(
              workgroup, local_max, sycl::maximum<float>{});
          float local_sum = 0;
          for (int p = lane; p < parts; p += ReduceLanes) {
            const float denominator = src[p * stride + dim + 1];
            const float factor = denominator > 0 ?
                sycl::exp(src[p * stride + dim] - maximum) : 0.0f;
            weights[p] = factor;
            local_sum += factor * denominator;
          }
          const float sum = sycl::reduce_over_group(
              workgroup, local_sum, sycl::plus<float>{});
          item.barrier(sycl::access::fence_space::local_space);
          if (component < dim) {
            float value = 0;
            for (int p = 0; p < parts; ++p)
              value += weights[p] * src[p * stride + component];
            Store(dst, row * dim + component, value / sum);
          }
        });
      });
      RecordProfileEvent(q, "attention_split_reduce_cooperative", reduce_event);
      return;
    }
    const auto reduce_event = NativeQueue(q).parallel_for(sycl::range<1>(tokens * heads * dim), [=](sycl::id<1> item) {
      const int64_t row = item[0] / dim, d = item[0] % dim;
      const auto* src = partial + row * parts * stride;
      float maximum = -std::numeric_limits<float>::infinity();
      for (int p = 0; p < parts; ++p) if (src[p * stride + dim + 1] > 0)
        maximum = sycl::max(maximum, src[p * stride + dim]);
      float sum = 0, value = 0;
      for (int p = 0; p < parts; ++p) if (src[p * stride + dim + 1] > 0) {
        const float factor = sycl::exp(src[p * stride + dim] - maximum);
        sum += factor * src[p * stride + dim + 1];
        value += factor * src[p * stride + d];
      }
      Store(dst, item[0], value / sum);
    });
    RecordProfileEvent(q, "attention_split_reduce", reduce_event);
  });
}
}  // namespace vt::xpu
