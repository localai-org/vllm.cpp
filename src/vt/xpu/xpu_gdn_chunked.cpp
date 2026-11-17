#include "xpu_common.h"
#include "xpu_kernels.h"
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdlib>
#include <optional>
#include <string_view>
#include <type_traits>

namespace vt::xpu {
namespace {
constexpr int C = 64, D = 128, MaxHeads = 48;
constexpr size_t WorkspaceBytes = 16 * 1024 * 1024;
using BF = sycl::ext::oneapi::bfloat16;
bool QkXmxEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_GDN_QK");
    const std::string_view mode = value ? value : "xmx";
    VT_CHECK(mode == "reference" || mode == "xmx", "Invalid VT_XPU_GDN_QK");
    return mode == "xmx";
  }();
  return enabled;
}
bool DeltaCrossEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_GDN_DELTA_CROSS");
    const std::string_view mode = value ? value : "tile4";
    VT_CHECK(mode == "reference" || mode == "tile4", "Invalid VT_XPU_GDN_DELTA_CROSS");
    return mode == "tile4";
  }();
  return enabled;
}
bool IntraStateTileEnabled() {
  static const bool enabled = [] {
    const char* value = std::getenv("VT_XPU_GDN_INTRA_STATE");
    const std::string_view mode = value ? value : "tile4";
    VT_CHECK(mode == "reference" || mode == "tile4", "Invalid VT_XPU_GDN_INTRA_STATE");
    return mode == "tile4";
  }();
  return enabled;
}

// One or two chunks for one sequence at a time. K/Q retain their input
// precision as XMX operands. The inverse, W/U, deltas, old state and all state
// arithmetic stay F32.
template <typename Input>
struct ChunkScratch {
  Input *q, *k, *kt, *inverse_xmx, *weighted_k, *weighted_v;
  Input *inverse_xmx_lo, *weighted_k_lo, *weighted_v_lo;
  Input *weighted_delta_hi, *weighted_delta_lo;
  float *g, *exp_g, *tail_decay, *lower, *qk, *inverse, *w, *u, *delta, *cross, *state_t;
  ChunkScratch(void* storage, int heads) {
    auto* cursor = static_cast<char*>(storage);
    auto fp16 = [&](size_t count) { auto* p = reinterpret_cast<Input*>(cursor); cursor += count * sizeof(Input); return p; };
    auto fp = [&](size_t count) { auto* p = reinterpret_cast<float*>(cursor); cursor += count * sizeof(float); return p; };
    q = fp16(heads * C * D); k = fp16(heads * C * D); kt = fp16(heads * D * C);
    inverse_xmx = fp16(heads * C * C);
    weighted_k = fp16(heads * C * D); weighted_v = fp16(heads * C * D);
    g = fp(heads * C); exp_g = fp(heads * C); tail_decay = fp(heads * C);
    lower = fp(heads * C * C); qk = fp(heads * C * C); inverse = fp(heads * C * C);
    w = fp(heads * C * D); u = fp(heads * C * D); delta = fp(heads * C * D);
    cross = fp(heads * C * D);
    state_t = fp(heads * D * D);
    // Lower is dead after inversion; delta is first written after W/U.
    inverse_xmx_lo = reinterpret_cast<Input*>(lower);
    weighted_k_lo = reinterpret_cast<Input*>(delta);
    weighted_v_lo = weighted_k_lo + heads * C * D;
    // Cross is dead after the output kernel and can hold the state operands.
    weighted_delta_hi = reinterpret_cast<Input*>(cross);
    weighted_delta_lo = weighted_delta_hi + heads * D * C;
    VT_CHECK(size_t(cursor - static_cast<char*>(storage)) <= WorkspaceBytes, "GDN workspace overflow");
  }
};
static_assert(MaxHeads * ((5 * C * D + C * C) * sizeof(BF) +
    (3 * C + 3 * C * C + 4 * C * D + D * D) * sizeof(float)) <= WorkspaceBytes);

template <typename Input, int Batch = 1>
void Prepare(Queue& queue, ChunkScratch<Input> s0, ChunkScratch<Input> s1,
             View qi, View ki, View gates, const float* state,
             const int32_t* offsets, int sequence, int base, int heads, int key_heads) {
  auto& q = NativeQueue(queue);
  // VT has already normalized q/k and transformed g/beta. Only a chunk-local
  // prefix sum belongs here; do not repeat donor softplus, sigmoid or L2Norm.
  // Gate sums and input/state copies are independent, so one launch prepares both.
  // Transpose the F32 state through a padded local tile for contiguous global writes.
  constexpr int tile_size = 16, tiles_per_head = (D / tile_size) * (D / tile_size);
  const auto prepare_event = q.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float> state_tile(tile_size * (tile_size + 1), handler);
    handler.parallel_for(sycl::nd_range<1>(Batch * heads * tiles_per_head * tile_size * tile_size,
                                          tile_size * tile_size),
        [=](sycl::nd_item<1> item) {
      const int group = item.get_group(0), lane = item.get_local_id(0);
      const int chunk = group / (heads * tiles_per_head);
      const int head_tile = group % (heads * tiles_per_head);
      const int h = head_tile / tiles_per_head, tile = head_tile % tiles_per_head;
      const ChunkScratch<Input> s = chunk ? s1 : s0;
      const int v = (tile / (D / tile_size)) * tile_size + lane / tile_size;
      const int k = (tile % (D / tile_size)) * tile_size + lane % tile_size;
      const int inner = v * D + k;
      if (tile == 0 && lane == 0) {
        const int first = offsets[sequence] + base + chunk * C, end = offsets[sequence + 1];
        float sum = 0;
        for (int i = 0; i < C; ++i) {
          if (first + i < end) sum += Load(gates, (first + i) * heads + h);
          s.g[h * C + i] = sum;
        }
        for (int i = 0; i < C; ++i) {
          s.exp_g[h * C + i] = sycl::exp(s.g[h * C + i]);
          s.tail_decay[h * C + i] = sycl::exp(sum - s.g[h * C + i]);
        }
      }
      if (chunk == 0)
        state_tile[(lane / tile_size) * (tile_size + 1) + lane % tile_size] =
            state[(sequence * heads + h) * D * D + inner];
      if (inner < C * D) {
        const int row = inner / D, d = inner % D;
        const int token = offsets[sequence] + base + chunk * C + row;
        const bool valid = token < offsets[sequence + 1];
        const int kh = h / (heads / key_heads);
        const Input qv(valid ? Load(qi, (token * key_heads + kh) * D + d) : 0.0f);
        const Input kv(valid ? Load(ki, (token * key_heads + kh) * D + d) : 0.0f);
        s.q[h * C * D + inner] = qv; s.k[h * C * D + inner] = kv;
        s.kt[(h * D + d) * C + row] = kv;
      }
      if (chunk == 0) {
        item.barrier(sycl::access::fence_space::local_space);
        const int out_v = (tile / (D / tile_size)) * tile_size + lane % tile_size;
        const int out_k = (tile % (D / tile_size)) * tile_size + lane / tile_size;
        s.state_t[(h * D + out_k) * D + out_v] =
            state_tile[(lane % tile_size) * (tile_size + 1) + lane / tile_size];
      }
    });
  });
  RecordProfileEvent(queue, Batch == 1 ? "gdn_chunk_prepare" : "gdn_chunk_prepare_pair",
                     prepare_event);
}

template <typename Input>
void TransposeState(Queue& queue, ChunkScratch<Input> s, const float* state,
                    int sequence, int heads) {
  constexpr int tile_size = 16, tiles_per_head = (D / tile_size) * (D / tile_size);
  const auto event = NativeQueue(queue).submit([&](sycl::handler& handler) {
    sycl::local_accessor<float> tile(tile_size * (tile_size + 1), handler);
    handler.parallel_for(sycl::nd_range<1>(heads * tiles_per_head * tile_size * tile_size,
                                          tile_size * tile_size),
        [=](sycl::nd_item<1> item) {
      const int group = item.get_group(0), lane = item.get_local_id(0);
      const int h = group / tiles_per_head, block = group % tiles_per_head;
      const int v = (block / (D / tile_size)) * tile_size + lane / tile_size;
      const int k = (block % (D / tile_size)) * tile_size + lane % tile_size;
      tile[(lane / tile_size) * (tile_size + 1) + lane % tile_size] =
          state[(sequence * heads + h) * D * D + v * D + k];
      item.barrier(sycl::access::fence_space::local_space);
      const int out_v = (block / (D / tile_size)) * tile_size + lane % tile_size;
      const int out_k = (block % (D / tile_size)) * tile_size + lane / tile_size;
      s.state_t[(h * D + out_k) * D + out_v] =
          tile[(lane % tile_size) * (tile_size + 1) + lane / tile_size];
    });
  });
  RecordProfileEvent(queue, "gdn_chunk_state_transpose", event);
}

// Compute K K^T using native BF16/F16 XMX, F32 accumulation. Each SG16
// owns a 16x16 tile; no state operand is narrowed for this operation.
template <typename Input, int Batch = 1>
void Dots(Queue& queue, ChunkScratch<Input> s0, ChunkScratch<Input> s1, int heads) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  const bool qk_xmx = QkXmxEnabled();
  const auto event = NativeQueue(queue).parallel_for(sycl::nd_range<1>(Batch * heads * 16 * 16, 16),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
    auto sg = item.get_sub_group();
    const int chunk = item.get_group(0) / (heads * 16);
    const int tile = item.get_group(0) % (heads * 16);
    const ChunkScratch<Input> s = chunk ? s1 : s0;
    const int h = (tile / 16) % heads, row = (tile % 16) / 4 * 16, col = tile % 4 * 16;
    const Input* a = s.k + (h * C + row) * D;
    const Input* q = s.q + (h * C + row) * D;
    const Input* b = s.kt + h * D * C + col;
    float* out = s.lower + (h * C + row) * C + col;
    float* qk = s.qk + (h * C + row) * C + col;
    mx::joint_matrix<sycl::sub_group, Input, mx::use::a, 16, 16, mx::layout::row_major> ja;
    mx::joint_matrix<sycl::sub_group, Input, mx::use::b, 16, 16, mx::layout::row_major> jb;
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> acc;
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> qacc;
    mx::joint_matrix_fill(sg, acc, 0.0f);
    if (qk_xmx) mx::joint_matrix_fill(sg, qacc, 0.0f);
    for (int d = 0; d < D; d += 16) {
      mx::joint_matrix_load(sg, ja, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(a + d), D);
      mx::joint_matrix_load(sg, jb, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(b + d * C), C);
      mx::joint_matrix_mad(sg, acc, ja, jb, acc);
      if (qk_xmx) {
        mx::joint_matrix_load(sg, ja, sycl::address_space_cast<sycl::access::address_space::global_space,
            sycl::access::decorated::no>(q + d), D);
        mx::joint_matrix_mad(sg, qacc, ja, jb, qacc);
      }
    }
    mx::joint_matrix_store(sg, acc, sycl::address_space_cast<sycl::access::address_space::global_space,
        sycl::access::decorated::no>(out), C, mx::layout::row_major);
    if (qk_xmx)
      mx::joint_matrix_store(sg, qacc, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(qk), C, mx::layout::row_major);
  });
  RecordProfileEvent(queue, Batch == 1
      ? (qk_xmx ? "gdn_chunk_dots_qk" : "gdn_chunk_dots")
      : (qk_xmx ? "gdn_chunk_dots_qk_pair" : "gdn_chunk_dots_pair"), event);
}

template <typename Input>
void System(Queue& queue, ChunkScratch<Input> s, View beta, const int32_t* offsets,
            int sequence, int base, int heads, float scale) {
  auto& q = NativeQueue(queue);
  const bool qk_xmx = QkXmxEnabled();
  const auto system_event = q.parallel_for(sycl::range<1>(heads * C * C), [=](sycl::id<1> item) {
    const int index = item[0];
    const int h = index / (C * C), row = (index / C) % C, col = index % C;
    const int token = offsets[sequence] + base + row;
    const bool valid = token < offsets[sequence + 1];
    const float decay = valid && row >= col ? sycl::exp(s.g[h * C + row] - s.g[h * C + col]) : 0;
    s.lower[index] = valid && row > col ? s.lower[index] * Load(beta, token * heads + h) * decay : 0;
    // The XMX path factors out scale after the FP32 dot. The reference path
    // preserves VT's per-element FP32 q*scale rounding for comparison.
    if (qk_xmx) {
      s.qk[index] = valid && row >= col ? s.qk[index] * scale * decay : 0;
    } else {
      float dot = 0;
      if (valid && row >= col) for (int d = 0; d < D; ++d)
        dot += static_cast<float>(s.k[(h * C + col) * D + d]) *
            (static_cast<float>(s.q[(h * C + row) * D + d]) * scale);
      s.qk[index] = dot * decay;
    }
  });
  RecordProfileEvent(queue, "gdn_chunk_system", system_event);
  enum class InverseMode { kReference, kSlm, kBlocked };
  static const InverseMode inverse_mode = [] {
    const char* value = std::getenv("VT_XPU_GDN_INVERSE");
    const std::string_view mode = value ? value : "blocked";
    VT_CHECK(mode == "slm" || mode == "reference" || mode == "blocked",
             "Invalid VT_XPU_GDN_INVERSE");
    return mode == "blocked" ? InverseMode::kBlocked
        : mode == "slm" ? InverseMode::kSlm : InverseMode::kReference;
  }();
  if (inverse_mode == InverseMode::kBlocked) {
    constexpr int block = 8, wg = 128;
    const auto inverse_event = q.submit([&](sycl::handler& handler) {
      sycl::local_accessor<float> lower(C * C, handler), inverse(C * C, handler);
      sycl::local_accessor<float> product(block * C, handler);
      handler.parallel_for(sycl::nd_range<1>(heads * wg, wg), [=](sycl::nd_item<1> item) {
        const int h = item.get_group(0), tid = item.get_local_id(0);
        for (int i = tid; i < C * C; i += wg) {
          lower[i] = s.lower[h * C * C + i];
          inverse[i] = 0.0f;
        }
        item.barrier(sycl::access::fence_space::local_space);
        for (int base = 0; base < C; base += block) {
          if (tid < block) {
            const int col = base + tid;
            for (int row = base; row < base + block; ++row) {
              float value = row == col ? 1.0f : 0.0f;
              if (row > col) {
                value = -lower[row * C + col];
                for (int j = col + 1; j < row; ++j)
                  value -= lower[row * C + j] * inverse[j * C + col];
              }
              inverse[row * C + col] = value;
            }
          }
          item.barrier(sycl::access::fence_space::local_space);
          for (int i = tid; i < block * base; i += wg) {
            const int row = i / base, col = i % base;
            float value = 0.0f;
            for (int j = 0; j < base; ++j)
              value += lower[(base + row) * C + j] * inverse[j * C + col];
            product[row * C + col] = value;
          }
          item.barrier(sycl::access::fence_space::local_space);
          for (int i = tid; i < block * base; i += wg) {
            const int row = i / base, col = i % base;
            float value = 0.0f;
            for (int j = 0; j <= row; ++j)
              value -= inverse[(base + row) * C + base + j] * product[j * C + col];
            inverse[(base + row) * C + col] = value;
          }
          item.barrier(sycl::access::fence_space::local_space);
        }
        for (int i = tid; i < C * C; i += wg)
          s.inverse[h * C * C + i] = inverse[i];
      });
    });
    RecordProfileEvent(queue, "gdn_chunk_inverse_blocked", inverse_event);
    return;
  }
  if (inverse_mode == InverseMode::kSlm) {
    const auto inverse_event = q.submit([&](sycl::handler& handler) {
      sycl::local_accessor<float> tile(C * C, handler);
      handler.parallel_for(sycl::nd_range<1>(heads * C, C), [=](sycl::nd_item<1> item) {
        const int h = item.get_group(0), col = item.get_local_id(0);
        const auto* lower = s.lower + h * C * C;
        // A column has no dependency on another column. Keep its intermediate
        // values in SLM, then write the completed inverse to the workspace.
        for (int row = 0; row < C; ++row) {
          float value = row == col ? 1.0f : 0.0f;
          if (row > col) {
            value = -lower[row * C + col];
            for (int j = col + 1; j < row; ++j)
              value -= lower[row * C + j] * tile[j * C + col];
          }
          tile[row * C + col] = value;
        }
        for (int row = 0; row < C; ++row)
          s.inverse[(h * C + row) * C + col] = tile[row * C + col];
      });
    });
    RecordProfileEvent(queue, "gdn_chunk_inverse_slm", inverse_event);
    return;
  }
  // Each work-item solves one column of (I + lower)^-1. Dependencies stay in
  // that column, so neither subgroup barriers nor cross-workgroup waits occur.
  const auto inverse_event = q.parallel_for(sycl::range<1>(heads * C), [=](sycl::id<1> item) {
    const int index = item[0];
    const int h = index / C, col = index % C;
    auto* inverse = s.inverse + h * C * C;
    const auto* lower = s.lower + h * C * C;
    for (int row = 0; row < C; ++row) {
      float value = row == col ? 1.0f : 0.0f;
      if (row > col) {
        value = -lower[row * C + col];
        for (int j = col + 1; j < row; ++j) value -= lower[row * C + j] * inverse[j * C + col];
      }
      inverse[row * C + col] = value;
    }
  });
  RecordProfileEvent(queue, "gdn_chunk_inverse", inverse_event);
}

// The two chunk-local systems are independent of the recurrent state.
template <typename Input>
void SystemPair(Queue& queue, ChunkScratch<Input> first, ChunkScratch<Input> second,
                View beta, const int32_t* offsets, int sequence, int base,
                int heads, float scale) {
  auto& q = NativeQueue(queue);
  const auto system_event = q.parallel_for(sycl::range<1>(2 * heads * C * C),
      [=](sycl::id<1> item) {
    const int chunk = item[0] / (heads * C * C);
    const int index = item[0] % (heads * C * C);
    const ChunkScratch<Input> s = chunk ? second : first;
    const int h = index / (C * C), row = index / C % C, col = index % C;
    const int token = offsets[sequence] + base + chunk * C + row;
    const bool valid = token < offsets[sequence + 1];
    const float decay = valid && row >= col
        ? sycl::exp(s.g[h * C + row] - s.g[h * C + col]) : 0;
    s.lower[index] = valid && row > col
        ? s.lower[index] * Load(beta, token * heads + h) * decay : 0;
    s.qk[index] = valid && row >= col ? s.qk[index] * scale * decay : 0;
  });
  RecordProfileEvent(queue, "gdn_chunk_system_pair", system_event);
  constexpr int block = 8, wg = 128;
  const auto inverse_event = q.submit([&](sycl::handler& handler) {
    sycl::local_accessor<float> lower(C * C, handler), inverse(C * C, handler);
    sycl::local_accessor<float> product(block * C, handler);
    handler.parallel_for(sycl::nd_range<1>(2 * heads * wg, wg),
        [=](sycl::nd_item<1> item) {
      const int group = item.get_group(0), chunk = group / heads;
      const int h = group % heads, tid = item.get_local_id(0);
      const ChunkScratch<Input> s = chunk ? second : first;
      for (int i = tid; i < C * C; i += wg) {
        lower[i] = s.lower[h * C * C + i];
        inverse[i] = 0.0f;
      }
      item.barrier(sycl::access::fence_space::local_space);
      for (int base = 0; base < C; base += block) {
        if (tid < block) {
          const int col = base + tid;
          for (int row = base; row < base + block; ++row) {
            float value = row == col ? 1.0f : 0.0f;
            if (row > col) {
              value = -lower[row * C + col];
              for (int j = col + 1; j < row; ++j)
                value -= lower[row * C + j] * inverse[j * C + col];
            }
            inverse[row * C + col] = value;
          }
        }
        item.barrier(sycl::access::fence_space::local_space);
        for (int i = tid; i < block * base; i += wg) {
          const int row = i / base, col = i % base;
          float value = 0.0f;
          for (int j = 0; j < base; ++j)
            value += lower[(base + row) * C + j] * inverse[j * C + col];
          product[row * C + col] = value;
        }
        item.barrier(sycl::access::fence_space::local_space);
        for (int i = tid; i < block * base; i += wg) {
          const int row = i / base, col = i % base;
          float value = 0.0f;
          for (int j = 0; j <= row; ++j)
            value -= inverse[(base + row) * C + base + j] * product[j * C + col];
          inverse[(base + row) * C + col] = value;
        }
        item.barrier(sycl::access::fence_space::local_space);
      }
      for (int i = tid; i < C * C; i += wg)
        s.inverse[h * C * C + i] = inverse[i];
    });
  });
  RecordProfileEvent(queue, "gdn_chunk_inverse_blocked_pair", inverse_event);
}

template <typename Input>
void ComputeWUXmx(Queue& queue, ChunkScratch<Input> s, View vi, View beta,
                  const int32_t* offsets, int sequence, int base, int heads) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  const auto prepare = NativeQueue(queue).parallel_for(sycl::range<1>(heads * C * D),
      [=](sycl::id<1> item) {
    const int index = item[0], h = index / (C * D), row = index / D % C, d = index % D;
    const int first = offsets[sequence] + base;
    const bool valid = first + row < offsets[sequence + 1];
    const float b = valid ? Load(beta, (first + row) * heads + h) : 0.0f;
    const float wk = static_cast<float>(s.k[index]) * b * s.exp_g[h * C + row];
    const float wv = valid ? Load(vi, ((first + row) * heads + h) * D + d) * b : 0.0f;
    const Input wk_hi(wk), wv_hi(wv);
    s.weighted_k[index] = wk_hi;
    s.weighted_v[index] = wv_hi;
    s.weighted_k_lo[index] = Input(wk - static_cast<float>(wk_hi));
    s.weighted_v_lo[index] = Input(wv - static_cast<float>(wv_hi));
    if (d < C) {
      const int at = (h * C + row) * C + d;
      const float inverse = s.inverse[at];
      const Input inverse_hi(inverse);
      s.inverse_xmx[at] = inverse_hi;
      s.inverse_xmx_lo[at] = Input(inverse - static_cast<float>(inverse_hi));
    }
  });
  RecordProfileEvent(queue, "gdn_chunk_wu_xmx_prepare", prepare);
  const auto gemm = NativeQueue(queue).parallel_for(
      sycl::nd_range<1>(heads * (C / 16) * (D / 16) * 16, 16),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
    auto sg = item.get_sub_group();
    const int tile = item.get_group(0), h = tile / ((C / 16) * (D / 16));
    const int row = tile / (D / 16) % (C / 16) * 16, col = tile % (D / 16) * 16;
    mx::joint_matrix<sycl::sub_group, Input, mx::use::a, 16, 16, mx::layout::row_major> a, a_lo;
    mx::joint_matrix<sycl::sub_group, Input, mx::use::b, 16, 16, mx::layout::row_major> bk, bv, bk_lo, bv_lo;
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> w, u;
    mx::joint_matrix_fill(sg, w, 0.0f);
    mx::joint_matrix_fill(sg, u, 0.0f);
    for (int j = 0; j < C; j += 16) {
      mx::joint_matrix_load(sg, a, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(s.inverse_xmx + (h * C + row) * C + j), C);
      mx::joint_matrix_load(sg, a_lo, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(s.inverse_xmx_lo + (h * C + row) * C + j), C);
      mx::joint_matrix_load(sg, bk, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(s.weighted_k + (h * C + j) * D + col), D);
      mx::joint_matrix_load(sg, bv, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(s.weighted_v + (h * C + j) * D + col), D);
      mx::joint_matrix_load(sg, bk_lo, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(s.weighted_k_lo + (h * C + j) * D + col), D);
      mx::joint_matrix_load(sg, bv_lo, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(s.weighted_v_lo + (h * C + j) * D + col), D);
      mx::joint_matrix_mad(sg, w, a, bk, w);
      mx::joint_matrix_mad(sg, u, a, bv, u);
      mx::joint_matrix_mad(sg, w, a_lo, bk, w);
      mx::joint_matrix_mad(sg, u, a_lo, bv, u);
      mx::joint_matrix_mad(sg, w, a, bk_lo, w);
      mx::joint_matrix_mad(sg, u, a, bv_lo, u);
      mx::joint_matrix_mad(sg, w, a_lo, bk_lo, w);
      mx::joint_matrix_mad(sg, u, a_lo, bv_lo, u);
    }
    mx::joint_matrix_store(sg, w, sycl::address_space_cast<sycl::access::address_space::global_space,
        sycl::access::decorated::no>(s.w + (h * C + row) * D + col), D, mx::layout::row_major);
    mx::joint_matrix_store(sg, u, sycl::address_space_cast<sycl::access::address_space::global_space,
        sycl::access::decorated::no>(s.u + (h * C + row) * D + col), D, mx::layout::row_major);
  });
  RecordProfileEvent(queue, "gdn_chunk_wu_xmx", gemm);
}

template <typename Input>
void ComputeWU(Queue& queue, ChunkScratch<Input> s, View vi, View beta, const int32_t* offsets,
               int sequence, int base, int heads) {
  static const std::string_view mode = [] {
    const char* value = std::getenv("VT_XPU_GDN_WU");
    const std::string_view requested = value ? value : "auto";
    VT_CHECK(requested == "auto" || requested == "reference" || requested == "tile4" ||
             requested == "xmx",
             "Invalid VT_XPU_GDN_WU");
    const std::string_view mode = requested == "auto"
        ? (std::is_same_v<Input, sycl::half> ? "xmx" : "tile4") : requested;
    VT_CHECK((mode != "xmx" || std::is_same_v<Input, sycl::half>),
             "XPU GDN W/U XMX requires F16 input");
    return mode;
  }();
  if (mode == "xmx") return ComputeWUXmx(queue, s, vi, beta, offsets, sequence, base, heads);
  if (mode != "reference") {
    constexpr int width = 4;
    const auto event = NativeQueue(queue).parallel_for(
        sycl::range<1>(heads * C * (D / width)), [=](sycl::id<1> item) {
      const int index = item[0];
      const int h = index / (C * (D / width));
      const int row = (index / (D / width)) % C, d = (index % (D / width)) * width;
      const int first = offsets[sequence] + base;
      const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - first));
      float w[width] = {}, u[width] = {};
      if (row < n) for (int j = 0; j <= row; ++j) {
        const float inv = s.inverse[(h * C + row) * C + j];
        const float b = Load(beta, (first + j) * heads + h);
        const float decay = s.exp_g[h * C + j];
        for (int col = 0; col < width; ++col) {
          w[col] += inv * (static_cast<float>(s.k[(h * C + j) * D + d + col]) * b * decay);
          u[col] += inv * (Load(vi, ((first + j) * heads + h) * D + d + col) * b);
        }
      }
      for (int col = 0; col < width; ++col) {
        s.w[(h * C + row) * D + d + col] = w[col];
        s.u[(h * C + row) * D + d + col] = u[col];
      }
    });
    RecordProfileEvent(queue, "gdn_chunk_wu_tile4", event);
    return;
  }
  const auto event = NativeQueue(queue).parallel_for(sycl::range<1>(heads * C * D), [=](sycl::id<1> item) {
    const int index = item[0];
    const int h = index / (C * D), row = (index / D) % C, d = index % D;
    const int first = offsets[sequence] + base;
    const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - first));
    float w = 0, u = 0;
    if (row < n) for (int j = 0; j <= row; ++j) {
      const float inv = s.inverse[(h * C + row) * C + j];
      const float b = Load(beta, (first + j) * heads + h);
      w += inv * (static_cast<float>(s.k[(h * C + j) * D + d]) * b * s.exp_g[h * C + j]);
      u += inv * (Load(vi, ((first + j) * heads + h) * D + d) * b);
    }
    s.w[index] = w; s.u[index] = u;
  });
  RecordProfileEvent(queue, "gdn_chunk_wu", event);
}

template <typename Input>
void ComputeStateXmx(Queue& queue, ChunkScratch<Input> s, float* state,
                     const int32_t* offsets, int sequence, int base, int heads) {
  namespace mx = sycl::ext::oneapi::experimental::matrix;
  auto& q = NativeQueue(queue);
  const auto prepare = q.parallel_for(sycl::range<1>(heads * D * C), [=](sycl::id<1> item) {
    const int index = item[0], h = index / (D * C), v = index / C % D, j = index % C;
    const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - offsets[sequence] - base));
    const float weighted = j < n
        ? s.delta[(h * C + j) * D + v] * s.tail_decay[h * C + j] : 0.0f;
    const Input hi(weighted);
    s.weighted_delta_hi[index] = hi;
    s.weighted_delta_lo[index] = Input(weighted - static_cast<float>(hi));
  });
  RecordProfileEvent(queue, "gdn_chunk_state_xmx_prepare", prepare);
  const auto gemm = q.parallel_for(
      sycl::nd_range<1>(heads * (D / 16) * (D / 16) * 16, 16),
      [=](sycl::nd_item<1> item) [[sycl::reqd_sub_group_size(16)]] {
    const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - offsets[sequence] - base));
    if (!n) return;
    auto sg = item.get_sub_group();
    const int tile = item.get_group(0), h = tile / ((D / 16) * (D / 16));
    const int v = tile / (D / 16) % (D / 16) * 16, d = tile % (D / 16) * 16;
    mx::joint_matrix<sycl::sub_group, Input, mx::use::a, 16, 16, mx::layout::row_major> a, a_lo;
    mx::joint_matrix<sycl::sub_group, Input, mx::use::b, 16, 16, mx::layout::row_major> b;
    mx::joint_matrix<sycl::sub_group, float, mx::use::accumulator, 16, 16> product;
    mx::joint_matrix_fill(sg, product, 0.0f);
    for (int j = 0; j < C; j += 16) {
      mx::joint_matrix_load(sg, a, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(s.weighted_delta_hi + (h * D + v) * C + j), C);
      mx::joint_matrix_load(sg, a_lo, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(s.weighted_delta_lo + (h * D + v) * C + j), C);
      mx::joint_matrix_load(sg, b, sycl::address_space_cast<sycl::access::address_space::global_space,
          sycl::access::decorated::no>(s.k + (h * C + j) * D + d), D);
      mx::joint_matrix_mad(sg, product, a, b, product);
      mx::joint_matrix_mad(sg, product, a_lo, b, product);
    }
    float* out = state + (sequence * heads + h) * D * D + v * D + d;
    mx::joint_matrix_store(sg, product, sycl::address_space_cast<sycl::access::address_space::global_space,
        sycl::access::decorated::no>(out), D, mx::layout::row_major);
    item.barrier(sycl::access::fence_space::global_space);
    const float decay = s.exp_g[h * C + n - 1];
    for (int i = item.get_local_id(0); i < 16 * 16; i += 16) {
      const int row = i / 16, col = i % 16;
      out[row * D + col] += s.state_t[(h * D + d + col) * D + v + row] * decay;
    }
  });
  RecordProfileEvent(queue, "gdn_chunk_state_xmx", gemm);
}

template <typename Input>
void OutputState(Queue& queue, ChunkScratch<Input> s, View output, float* state, const int32_t* offsets,
                 int sequence, int base, int heads, float scale) {
  auto& q = NativeQueue(queue);
  const bool delta_cross = DeltaCrossEnabled();
  if (delta_cross) {
    constexpr int width = 4;
    const auto event = q.parallel_for(sycl::range<1>(heads * C * (D / width)), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (C * (D / width));
      const int row = (index / (D / width)) % C, v = (index % (D / width)) * width;
      float prediction[width] = {}, cross[width] = {};
      for (int d = 0; d < D; ++d) {
        const float w = s.w[(h * C + row) * D + d];
        const float qscaled = static_cast<float>(s.q[(h * C + row) * D + d]) * scale;
        for (int col = 0; col < width; ++col) {
          const float old = s.state_t[(h * D + d) * D + v + col];
          prediction[col] += w * old;
          cross[col] += old * qscaled;
        }
      }
      for (int col = 0; col < width; ++col) {
        const int at = (h * C + row) * D + v + col;
        s.delta[at] = s.u[at] - prediction[col];
        s.cross[at] = cross[col];
      }
    });
    RecordProfileEvent(queue, "gdn_chunk_delta_cross_tile4", event);
  } else {
    const auto delta_event = q.parallel_for(sycl::range<1>(heads * C * D), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (C * D), row = (index / D) % C, v = index % D;
      float prediction = 0;
      for (int d = 0; d < D; ++d)
        prediction += s.w[(h * C + row) * D + d] * s.state_t[(h * D + d) * D + v];
      s.delta[index] = s.u[index] - prediction;
    });
    RecordProfileEvent(queue, "gdn_chunk_delta", delta_event);
  }
  const bool intra_state_tile = IntraStateTileEnabled();
  if (intra_state_tile && delta_cross) {
    constexpr int width = 4;
    const auto event = q.parallel_for(sycl::range<1>(heads * C * (D / width)), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (C * (D / width));
      const int row = (index / (D / width)) % C, v = (index % (D / width)) * width;
      const int token = offsets[sequence] + base + row;
      if (token >= offsets[sequence + 1]) return;
      float intra[width] = {};
      for (int j = 0; j <= row; ++j) {
        const float qk = s.qk[(h * C + row) * C + j];
        for (int col = 0; col < width; ++col)
          intra[col] += qk * s.delta[(h * C + j) * D + v + col];
      }
      for (int col = 0; col < width; ++col) {
        const int at = (h * C + row) * D + v + col;
        Store(output, (token * heads + h) * D + v + col,
              s.cross[at] * s.exp_g[h * C + row] + intra[col]);
      }
    });
    RecordProfileEvent(queue, "gdn_chunk_output_tile4", event);
  } else {
    const auto output_event = q.parallel_for(sycl::range<1>(heads * C * D), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (C * D), row = (index / D) % C, v = index % D;
      const int token = offsets[sequence] + base + row;
      if (token >= offsets[sequence + 1]) return;
      float cross = 0, intra = 0;
      if (delta_cross) cross = s.cross[index];
      else for (int d = 0; d < D; ++d)
        cross += s.state_t[(h * D + d) * D + v] * (static_cast<float>(s.q[(h * C + row) * D + d]) * scale);
      for (int j = 0; j <= row; ++j)
        intra += s.qk[(h * C + row) * C + j] * s.delta[(h * C + j) * D + v];
      Store(output, (token * heads + h) * D + v, cross * s.exp_g[h * C + row] + intra);
    });
    RecordProfileEvent(queue, "gdn_chunk_output", output_event);
  }
  static const std::string_view state_mode = [] {
    const char* value = std::getenv("VT_XPU_GDN_STATE");
    const std::string_view requested = value ? value : "auto";
    VT_CHECK(requested == "auto" || requested == "tile4" || requested == "xmx",
             "Invalid VT_XPU_GDN_STATE");
    const std::string_view mode = requested == "auto"
        ? (std::is_same_v<Input, sycl::half> ? "xmx" : "tile4") : requested;
    VT_CHECK((mode != "xmx" || std::is_same_v<Input, sycl::half>),
             "XPU GDN state XMX requires F16 input");
    return mode;
  }();
  if (state_mode == "xmx")
    return ComputeStateXmx(queue, s, state, offsets, sequence, base, heads);
  if (intra_state_tile) {
    constexpr int width = 4;
    const auto event = q.parallel_for(sycl::range<1>(heads * D * (D / width)), [=](sycl::id<1> item) {
      const int index = item[0], h = index / (D * (D / width));
      const int v = (index / (D / width)) % D, d = (index % (D / width)) * width;
      const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - offsets[sequence] - base));
      if (!n) return;
      float value[width];
      for (int col = 0; col < width; ++col)
        value[col] = s.state_t[(h * D + d + col) * D + v] * s.exp_g[h * C + n - 1];
      for (int j = 0; j < n; ++j) {
        const float weighted_delta = s.delta[(h * C + j) * D + v] * s.tail_decay[h * C + j];
        for (int col = 0; col < width; ++col)
          value[col] += weighted_delta * static_cast<float>(s.k[(h * C + j) * D + d + col]);
      }
      for (int col = 0; col < width; ++col)
        state[(sequence * heads + h) * D * D + v * D + d + col] = value[col];
    });
    RecordProfileEvent(queue, "gdn_chunk_state_tile4", event);
    return;
  }
  const auto state_event = q.parallel_for(sycl::range<1>(heads * D * D), [=](sycl::id<1> item) {
    const int h = item[0] / (D * D), v = (item[0] / D) % D, d = item[0] % D;
    const int n = sycl::min(C, sycl::max(0, offsets[sequence + 1] - offsets[sequence] - base));
    if (!n) return;
    float value = s.state_t[(h * D + d) * D + v] * s.exp_g[h * C + n - 1];
    for (int j = 0; j < n; ++j)
      value += s.delta[(h * C + j) * D + v] * s.tail_decay[h * C + j] *
          static_cast<float>(s.k[(h * C + j) * D + d]);
    state[(sequence * heads + h) * D * D + v * D + d] = value;
  });
  RecordProfileEvent(queue, "gdn_chunk_state", state_event);
}
}

template <typename Input>
bool RunChunked(Queue& queue, Tensor& out, const Tensor& qi, const Tensor& ki,
                const Tensor& vi, const Tensor& g, const Tensor& beta, Tensor& state,
                const Tensor& qsl, const GdnArgs& args, int heads, int key_heads,
                int sequences, int tokens) {
  const char* requested = std::getenv("VT_XPU_GDN_BATCH");
  const std::string_view batch_mode = requested ? requested : "2";
  VT_CHECK(batch_mode == "1" || batch_mode == "2", "Invalid VT_XPU_GDN_BATCH");
  const char* inverse = std::getenv("VT_XPU_GDN_INVERSE");
  const bool compatible = QkXmxEnabled() &&
      (!inverse || std::string_view(inverse) == "blocked");
  if (requested && batch_mode == "2")
    VT_CHECK(compatible, "XPU GDN batch 2 requires XMX QK and blocked inverse");
  const bool batch_two = batch_mode == "2" && compatible;
  return WithGdnWorkspace(queue, WorkspaceBytes * (batch_two ? 2 : 1), [&](void* storage) {
    ChunkScratch<Input> scratch(storage, heads);
    std::optional<ChunkScratch<Input>> second;
    if (batch_two) second.emplace(static_cast<char*>(storage) + WorkspaceBytes, heads);
    WithOutput(queue, out, {&qi, &ki, &vi, &g, &beta}, [&](Tensor& target) {
      const auto* offsets = static_cast<const int32_t*>(qsl.data);
      for (int sequence = 0; sequence < sequences; ++sequence) {
        int base = 0;
        if (batch_two && sequences == 1) for (; base + C < tokens; base += 2 * C) {
          Prepare<Input, 2>(queue, scratch, *second, View(qi), View(ki), View(g),
                            static_cast<float*>(state.data), offsets, sequence, base,
                            heads, key_heads);
          Dots<Input, 2>(queue, scratch, *second, heads);
          SystemPair(queue, scratch, *second, View(beta), offsets, sequence, base,
                     heads, args.scale);
          ComputeWU(queue, scratch, View(vi), View(beta), offsets, sequence, base, heads);
          OutputState(queue, scratch, View(target), static_cast<float*>(state.data), offsets,
                      sequence, base, heads, args.scale);
          TransposeState(queue, *second, static_cast<float*>(state.data), sequence, heads);
          ComputeWU(queue, *second, View(vi), View(beta), offsets, sequence, base + C, heads);
          OutputState(queue, *second, View(target), static_cast<float*>(state.data), offsets,
                      sequence, base + C, heads, args.scale);
        }
        for (; base < tokens; base += C) {
          Prepare(queue, scratch, scratch, View(qi), View(ki), View(g),
                  static_cast<float*>(state.data), offsets, sequence, base, heads, key_heads);
          Dots(queue, scratch, scratch, heads);
          System(queue, scratch, View(beta), offsets, sequence, base, heads, args.scale);
          ComputeWU(queue, scratch, View(vi), View(beta), offsets, sequence, base, heads);
          OutputState(queue, scratch, View(target), static_cast<float*>(state.data), offsets,
                      sequence, base, heads, args.scale);
        }
      }
    });
  });
}

bool GdnChunkedPrefillKernel(Queue& queue, Tensor& out, const Tensor& qi, const Tensor& ki,
                            const Tensor& vi, const Tensor& g, const Tensor& beta, Tensor& state,
                            const Tensor& qsl, const GdnArgs& args) {
  const auto device = NativeQueue(queue).get_device();
  const int heads = state.shape[1], sequences = state.shape[0], key_heads = qi.shape[1], tokens = qi.shape[0];
  if (state.shape[2] != D || state.shape[3] != D || heads < 1 || heads > MaxHeads ||
      sequences < 1 || sequences > 4 || tokens < 1 || tokens > 6656 ||
      (qi.dtype != DType::kBF16 && qi.dtype != DType::kF16) ||
      ki.dtype != qi.dtype || vi.dtype != qi.dtype ||
      !device.has(sycl::aspect::ext_intel_device_id) ||
      device.get_info<sycl::ext::intel::info::device::device_id>() != 57891 ||
      !device.has(sycl::aspect::ext_intel_matrix)) return false;
  VT_CHECK(state.dtype == DType::kF32, "XPU chunked GDN requires F32 state");
  for (const auto* t : std::initializer_list<const Tensor*>{&out, &qi, &ki, &vi, &g, &beta})
    VT_CHECK(!Overlap(state, *t), "XPU GDN state must have separate storage");
  const auto* offsets = static_cast<const int32_t*>(qsl.data);
  CheckDeviceMetadata(queue, [=] {
    if (offsets[0] != 0 || offsets[sequences] != tokens) return false;
    for (int i = 0; i < sequences; ++i) if (offsets[i] < 0 || offsets[i] > offsets[i + 1]) return false;
    return true;
  }, "XPU chunked GDN invalid sequence offsets", {&qsl});
  if (qi.dtype == DType::kF16)
    return RunChunked<sycl::half>(queue, out, qi, ki, vi, g, beta, state, qsl,
                                  args, heads, key_heads, sequences, tokens);
  return RunChunked<BF>(queue, out, qi, ki, vi, g, beta, state, qsl,
                        args, heads, key_heads, sequences, tokens);
}
}  // namespace vt::xpu
