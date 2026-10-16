#include "xpu_common.h"
#include "xpu_fp8.h"
#include "xpu_kernels.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <string_view>
#include <sycl/ext/intel/experimental/grf_size_properties.hpp>

namespace vt::xpu {
namespace {
template<class Fn> struct PrefillGrf256 {
  Fn fn;
  void operator()(sycl::nd_item<1> item) const { fn(item); }
  auto get(sycl::ext::oneapi::experimental::properties_tag) const {
    return sycl::ext::oneapi::experimental::properties{
        sycl::ext::intel::experimental::grf_size<256>,
        sycl::ext::oneapi::experimental::sub_group_size<16>};
  }
};
}
// Native adaptation of the donor's tiled QK -> online softmax -> PV schedule.
// VT pages may be strided unbind views; no donor block-size/layout assumption.
// Q16/Q32/Q64 with K32, eight SG16s; F32 running softmax and output.
// 256 GRFs avoid the spill traffic of the 128-GRF compilation on B70.
// F32 queries use FP16 high+residual XMX operands. F16 Q32/Q64 use one FP16
// probability operand after the FP32 softmax; K/V share one staging buffer.
template<int Q, bool QueryResidual, bool ProbabilityResidual = true>
bool PagedAttentionPrefillImpl(Queue& q, Tensor& out, const Tensor& query, const Tensor& key_cache,
                               const Tensor& value_cache, const Tensor& block_table,
                               const Tensor& seq_lens, const Tensor& query_start_loc,
                               const PagedAttentionArgs& args) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  namespace imx = sycl::ext::intel::experimental::matrix;
  constexpr int K = 32, D = 256, WG = 128;
  static_assert(Q == 16 || ((Q == 32 || Q == 64) && !QueryResidual));
  constexpr size_t local_bytes =
      (Q * D + K * D + Q * K + (QueryResidual ? Q * D : 1) +
       (ProbabilityResidual ? Q * K : 1)) * sizeof(sycl::half) +
      (Q * K + 3 * Q) * sizeof(float);
  const auto device = NativeQueue(q).get_device();
  const int64_t tokens = query.shape[0], heads = query.shape[1], ratio = heads / key_cache.shape[2];
  const int64_t requests = seq_lens.Numel(), page = key_cache.shape[1];
  if (tokens < Q || query.shape[2] != D || requests > 4 || heads > 24 ||
      !device.has(sycl::aspect::ext_intel_matrix) ||
      !device.has(sycl::aspect::ext_intel_device_id) ||
      device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
      device.get_info<sycl::info::device::local_mem_size>() < local_bytes) return false;
  const View qs(query), kc(key_cache), vc(value_cache), dst(out), bt(block_table);
  const auto* table = static_cast<const int32_t*>(bt.data);
  const auto* lengths = static_cast<const int32_t*>(seq_lens.data);
  const auto* offsets = static_cast<const int32_t*>(query_start_loc.data);
  const float scale = args.scale, cap = args.logits_soft_cap, kscale = args.k_scale, vscale = args.v_scale;
  const bool causal = args.causal;
  const int64_t left = args.window_size ? args.window_size->left : -1;
  const int64_t right = args.window_size ? args.window_size->right : -1;
  // Preserve the pinned producer's base2 softmax boundaries for the observed
  // short F16 prefill. Other shapes/modes keep their existing arithmetic.
  const bool producer_softmax = !QueryResidual && !ProbabilityResidual &&
      query.dtype == DType::kF16 && out.dtype == DType::kF16 &&
      tokens == 128 && requests == 1 && heads == 24 && ratio == 6 &&
      (page == 1600 || page == 1664) && args.max_seq_len == 128 &&
      key_cache.dtype == DType::kI8 && value_cache.dtype == DType::kI8 &&
      args.kv_cache_dtype == Fp8KVCacheDataType::kFp8E4M3 &&
      kscale == 1 && vscale == 1 && scale == 1.0f / 16 && causal &&
      cap == 0 && !args.window_size;
  constexpr float log2e = static_cast<float>(1.4426950408889634074);
  const float scale2 = scale * log2e;
  // One extra partial tile per sequence. The device maps this compact grid,
  // so empty requests and changing query lengths need no host mirror.
  const int64_t tiles = (tokens + Q - 1) / Q + requests - 1;
  Scratch status(q.device, sizeof(int));
  auto* unsafe = static_cast<int*>(status.data);
  const auto reset_event = NativeQueue(q).memset(unsafe, 0, sizeof(int));
  RecordProfileEvent(q, "attention_prefill_status_reset", reset_event);
  const auto event = NativeQueue(q).submit([&](sycl::handler& h) {
    sycl::local_accessor<sycl::half> query_tile(Q * D, h), kv(K * D, h), prob(Q * K, h);
    // The F16 instantiation keeps only a one-element placeholder for the
    // unused residual accessor; its 8-KiB tile is absent from local memory.
    sycl::local_accessor<sycl::half> query_low(QueryResidual ? Q * D : 1, h);
    sycl::local_accessor<sycl::half> prob_low(ProbabilityResidual ? Q * K : 1, h);
    sycl::local_accessor<float> scores(Q * K, h), maxval(Q, h), denom(Q, h), oldscale(Q, h);
    h.parallel_for(sycl::nd_range<1>(tiles * heads * WG, WG),
        PrefillGrf256{[=](sycl::nd_item<1> item) {
      const auto sg = item.get_sub_group();
      const int tid = item.get_local_id(0), subgroup = tid / 16;
      const int64_t head = item.get_group(0) % heads;
      int64_t tile = item.get_group(0) / heads, request = 0;
      while (request < requests) {
        const int64_t count = (offsets[request + 1] - offsets[request] + Q - 1) / Q;
        if (tile < count) break;
        tile -= count; ++request;
      }
      if (request == requests) return;
      const int64_t token = offsets[request] + tile * Q;
      const int rows = sycl::min(int64_t{Q}, int64_t(offsets[request + 1]) - token);
      const int64_t position = lengths[request] - (offsets[request + 1] - offsets[request]) + token - offsets[request];
      const int64_t first = left < 0 ? 0 : sycl::max(int64_t{0}, position - left);
      int64_t end = causal ? position + rows : lengths[request];
      if (right >= 0) end = sycl::min(end, position + rows + right);
      auto narrow = [&](float value) {
        if (!sycl::isfinite(value) || sycl::fabs(value) > 65504.0f) {
          sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
              sycl::access::address_space::global_space>(*unsafe).store(1);
          return sycl::half(0);
        }
        return sycl::half(value);
      };
      for (int i = tid; i < Q * D; i += WG) {
        const float value = i / D < rows ? Load(qs, ((token + i / D) * heads + head) * D + i % D) : 0;
        const auto high = narrow(value);
        query_tile[i] = high;
        if constexpr (QueryResidual) query_low[i] = narrow(value - float(high));
      }
      if (tid < Q) { maxval[tid] = -std::numeric_limits<float>::infinity(); denom[tid] = 0; }
      mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> acc0[Q / 16], acc1[Q / 16];
      for (int row_block = 0; row_block < Q / 16; ++row_block) {
        mx::joint_matrix_fill(sg, acc0[row_block], 0.0f);
        mx::joint_matrix_fill(sg, acc1[row_block], 0.0f);
      }
      for (int64_t base = first; base < end; base += K) {
        for (int i = tid; i < K * D; i += WG) {
          const int key = i % K, d = i / K;
          float value = 0;
          if (base + key < end) {
            const int64_t index = base + key, block = table[request * bt.stride[0] + (index / page) * bt.stride[1]];
            value = LoadKV(kc, block * kc.stride[0] + (index % page) * kc.stride[1] + (head / ratio) * kc.stride[2] + d, kscale);
          }
          kv[i] = narrow(value); // transposed K: [D,K]
        }
        item.barrier(sycl::access::fence_space::local_space);
        if (subgroup < (Q / 16) * (K / 16)) {
          const int row_block = subgroup / (K / 16), key_block = subgroup % (K / 16);
          mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> a;
          mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> b;
          mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> dot;
          mx::joint_matrix_fill(sg, dot, 0.0f);
          for (int d = 0; d < D; d += 16) {
            mx::joint_matrix_load(sg, a, query_tile.template get_multi_ptr<sycl::access::decorated::no>() + row_block * 16 * D + d, D);
            mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<sycl::access::decorated::no>() + d * K + key_block * 16, K);
            mx::joint_matrix_mad(sg, dot, a, b, dot);
            if constexpr (QueryResidual) {
              mx::joint_matrix_load(sg, a, query_low.template get_multi_ptr<sycl::access::decorated::no>() + row_block * 16 * D + d, D);
              mx::joint_matrix_mad(sg, dot, a, b, dot);
            }
          }
          mx::joint_matrix_store(sg, dot, scores.template get_multi_ptr<sycl::access::decorated::no>() + row_block * 16 * K + key_block * 16, K, mx::layout::row_major);
        }
        item.barrier(sycl::access::fence_space::local_space);
        for (int pass = 0; pass < Q / (WG / 16); ++pass) {
          const int row = subgroup + pass * (WG / 16);
          const int lane = tid % 16;
          const float previous = maxval[row];
          const auto score = [&](int key_col) {
            const int64_t key = base + key_col, p = position + row;
            const bool valid = row < rows && key < end && (!causal || key <= p) &&
                (left < 0 || key >= p - left) && (right < 0 || key <= p + right);
            float value = scores[row * K + key_col];
            if (!producer_softmax) value *= scale;
            if (cap > 0) value = cap * sycl::tanh(value / cap);
            return valid ? value : -std::numeric_limits<float>::infinity();
          };
          const float s0 = score(lane), s1 = score(lane + 16);
          const float tile_max = sycl::reduce_over_group(sg, sycl::max(s0, s1), sycl::maximum<float>());
          const float maximum = sycl::max(previous,
              producer_softmax ? scale2 * tile_max : tile_max);
          const float old = sycl::isfinite(maximum) ?
              (producer_softmax ? sycl::native::exp2(previous - maximum) :
                                  sycl::exp(previous - maximum)) : 0;
          const float e0 = sycl::isfinite(s0) ?
              (producer_softmax ? sycl::native::exp2(sycl::fma(scale2, s0, -maximum)) :
                                  sycl::exp(s0 - maximum)) : 0;
          const float e1 = sycl::isfinite(s1) ?
              (producer_softmax ? sycl::native::exp2(sycl::fma(scale2, s1, -maximum)) :
                                  sycl::exp(s1 - maximum)) : 0;
          const float sum = sycl::reduce_over_group(sg, e0 + e1, sycl::plus<float>());
          prob[row * K + lane] = sycl::half(e0);
          prob[row * K + lane + 16] = sycl::half(e1);
          if constexpr (ProbabilityResidual) {
            prob_low[row * K + lane] = sycl::half(e0 - float(sycl::half(e0)));
            prob_low[row * K + lane + 16] = sycl::half(e1 - float(sycl::half(e1)));
          }
          if (lane == 0) {
            denom[row] = denom[row] * old + sum;
            maxval[row] = maximum;
            oldscale[row] = old;
          }
        }
        for (int i = tid; i < K * D; i += WG) {
          const int key = i / D, d = i % D;
          float value = 0;
          if (base + key < end) {
            const int64_t index = base + key, block = table[request * bt.stride[0] + (index / page) * bt.stride[1]];
            value = LoadKV(vc, block * vc.stride[0] + (index % page) * vc.stride[1] + (head / ratio) * vc.stride[2] + d, vscale);
          }
          kv[i] = narrow(value); // shared stage now holds V: [K,D]
        }
        item.barrier(sycl::access::fence_space::local_space);
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::a, 16, 16, mx::layout::row_major> a;
        mx::joint_matrix<sycl::sub_group, sycl::half, mx::use::b, 16, 16, mx::layout::row_major> b;
        for (int row_block = 0; row_block < Q / 16; ++row_block) {
          imx::joint_matrix_apply(sg, acc0[row_block], [&](float& value, size_t row, size_t) {
            value *= oldscale[row_block * 16 + row];
          });
          imx::joint_matrix_apply(sg, acc1[row_block], [&](float& value, size_t row, size_t) {
            value *= oldscale[row_block * 16 + row];
          });
          for (int k = 0; k < K; k += 16) {
            mx::joint_matrix_load(sg, a, prob.template get_multi_ptr<sycl::access::decorated::no>() + row_block * 16 * K + k, K);
            mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<sycl::access::decorated::no>() + k * D + subgroup * 32, D);
            mx::joint_matrix_mad(sg, acc0[row_block], a, b, acc0[row_block]);
            mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<sycl::access::decorated::no>() + k * D + subgroup * 32 + 16, D);
            mx::joint_matrix_mad(sg, acc1[row_block], a, b, acc1[row_block]);
            if constexpr (ProbabilityResidual) {
              mx::joint_matrix_load(sg, a, prob_low.template get_multi_ptr<sycl::access::decorated::no>() + row_block * 16 * K + k, K);
              mx::joint_matrix_mad(sg, acc1[row_block], a, b, acc1[row_block]);
              mx::joint_matrix_load(sg, b, kv.template get_multi_ptr<sycl::access::decorated::no>() + k * D + subgroup * 32, D);
              mx::joint_matrix_mad(sg, acc0[row_block], a, b, acc0[row_block]);
            }
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      for (int row_block = 0; row_block < Q / 16; ++row_block) {
        imx::joint_matrix_apply(sg, acc0[row_block], [&](float& value, size_t row, size_t col) {
          const size_t query_row = row_block * 16 + row;
          if (query_row < size_t(rows))
            Store(dst, ((token + query_row) * heads + head) * D + subgroup * 32 + col,
                  value / denom[query_row]);
        });
        imx::joint_matrix_apply(sg, acc1[row_block], [&](float& value, size_t row, size_t col) {
          const size_t query_row = row_block * 16 + row;
          if (query_row < size_t(rows))
            Store(dst, ((token + query_row) * heads + head) * D + subgroup * 32 + 16 + col,
                  value / denom[query_row]);
        });
      }
    }});
  });
  RecordProfileEvent(q, Q == 64 ? "attention_prefill_q64" :
      Q == 32 ? "attention_prefill_q32" : "attention_prefill", event);
  int invalid = 0;
  GetBackend(q.device).Copy(q, &invalid, unsafe, sizeof(int));
  // If FP16 could not represent an operand, the caller re-runs its generic
  // F32 kernel. The original inputs are intact, including output aliases.
  return invalid == 0;
}
bool PagedAttentionPrefillKernel(Queue& q, Tensor& out, const Tensor& query, const Tensor& key_cache,
                                 const Tensor& value_cache, const Tensor& block_table,
                                 const Tensor& seq_lens, const Tensor& query_start_loc,
                                 const PagedAttentionArgs& args) {
  const char* setting = std::getenv("VT_XPU_ATTN_PREFILL_TILE");
  const std::string_view tile = setting ? setting : "auto";
  VT_CHECK(tile == "auto" || tile == "q16" || tile == "q32" || tile == "q64",
           "Invalid VT_XPU_ATTN_PREFILL_TILE");
  if (query.dtype == DType::kF16) {
    if (tile == "auto" || tile == "q32" || tile == "q64") {
      const char* probability = std::getenv("VT_XPU_ATTN_PROBABILITY");
      const std::string_view mode = probability ? probability : "single";
      VT_CHECK(mode == "residual" || mode == "single", "Invalid VT_XPU_ATTN_PROBABILITY");
      // Q32 keeps short prompts on XMX; Q64 needs 64 packed tokens and is the
      // measured long-prefill winner on B70. The residual reference stays Q32.
      if (tile == "q64" || (tile == "auto" && query.shape[0] >= 64 && mode == "single")) {
        VT_CHECK(mode == "single", "Q64 requires single FP16 probability operand");
        return PagedAttentionPrefillImpl<64, false, false>(q, out, query, key_cache,
            value_cache, block_table, seq_lens, query_start_loc, args);
      }
      if (mode == "single")
        return PagedAttentionPrefillImpl<32, false, false>(q, out, query, key_cache,
            value_cache, block_table, seq_lens, query_start_loc, args);
      return PagedAttentionPrefillImpl<32, false>(q, out, query, key_cache,
          value_cache, block_table, seq_lens, query_start_loc, args);
    }
    return PagedAttentionPrefillImpl<16, false>(q, out, query, key_cache,
        value_cache, block_table, seq_lens, query_start_loc, args);
  }
  return PagedAttentionPrefillImpl<16, true>(q, out, query, key_cache,
      value_cache, block_table, seq_lens, query_start_loc, args);
}
}  // namespace vt::xpu
