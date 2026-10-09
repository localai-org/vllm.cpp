// Native Torch-free port from release/0.1.15.4 ddf336d86e3c8602888572a3502f951abd51df12 (Apache-2.0).
#pragma once

#include <sycl/sycl.hpp>


namespace gdn::fp16_decode_producer {
static constexpr int sub_group_size = 32;
template <typename T, typename StateT, int k_bucket_size>
struct gated_delta_rule_kernel {
 public:
  static constexpr int group_size = 256;
  static constexpr int sg_per_group = group_size / sub_group_size;
  static constexpr int v_dim_per_sg = 4;
  static constexpr int v_dim_per_group = v_dim_per_sg * sg_per_group;
  static constexpr float eps = 0.000001;

  gated_delta_rule_kernel(
      T* core_attn_out,
      const T* q,
      const T* k,
      const T* v,
      const T* b,
      const T* a,
      const float* A_log,
      const T* dt_bias,
      StateT* ssm_state,
      const int ssm_state_stride_0,
      const int* query_start_loc,
      const int* token_indx,
      const int* cache_indices,
      const bool* has_initial_state,
      const int* num_accepted_tokens,
      const int batch_size,
      const int total_seqlen,
      const int num_k_heads,
      const int head_k_dim,
      const int num_v_heads,
      const int head_v_dim)
      : core_attn_out(core_attn_out),
        q(q),
        k(k),
        v(v),
        b(b),
        a(a),
        A_log(A_log),
        dt_bias(dt_bias),
        ssm_state(ssm_state),
        ssm_state_stride_0(ssm_state_stride_0),
        query_start_loc(query_start_loc),
        token_indx(token_indx),
        cache_indices(cache_indices),
        has_initial_state(has_initial_state),
        num_accepted_tokens(num_accepted_tokens),
        batch_size(batch_size),
        total_seqlen(total_seqlen),
        num_k_heads(num_k_heads),
        head_k_dim(head_k_dim),
        num_v_heads(num_v_heads),
        head_v_dim(head_v_dim) {}

  static inline sycl::nd_range<3> get_nd_range(
      const int batch_size, const int num_v_heads, const int head_v_dim) {
    int num_v_bucket = (head_v_dim + v_dim_per_group - 1) / v_dim_per_group;
    sycl::range<3> local(1, 1, group_size);
    sycl::range<3> global(batch_size, num_v_heads, num_v_bucket);
    return sycl::nd_range<3>(global * local, local);
  }

  static inline float act_sigmoid(float& x) {
    return 1.0f / (1.0f + sycl::exp(-x));
  }

  static inline float
  act_softplus(float& x, float beta = 1.0f, float threshold = 20.0f) {
    if (beta * x < threshold) {
      return sycl::log(1.0f + sycl::exp(beta * x)) / beta;
    } else
      return x;
  }

  [[sycl::reqd_sub_group_size(sub_group_size)]] void
  operator()(sycl::nd_item<3> item) const {
    int batch_id = item.get_group(0);
    int num_v_heads_id = item.get_group(1);
    int v_bucket_id = item.get_group(2);

    auto sg = item.get_sub_group();
    int sg_id = sg.get_group_id();
    int sg_local_id = sg.get_local_id();

    // assume num_v_heads is always bigger than num_k_heads
    int kv_ratio = num_v_heads / num_k_heads;
    int head_v_dim_id = v_bucket_id * v_dim_per_group + sg_id * v_dim_per_sg;

    if (head_v_dim_id >= head_v_dim) {
      return;
    }

    const float scale = 1.0f / sycl::sqrt(float(head_k_dim));
    float A_log_local = A_log[num_v_heads_id];
    float dt_bias_local = dt_bias[num_v_heads_id];
    A_log_local = -sycl::exp(A_log_local);

    float state_local[v_dim_per_sg * k_bucket_size];
    float q_local[k_bucket_size];
    float k_local[k_bucket_size];
    float v_local[v_dim_per_sg];

    StateT* ssm_state_ptr =
        ssm_state +
        static_cast<int64_t>(cache_indices[batch_id]) * ssm_state_stride_0;

    // load state
    if (has_initial_state == nullptr || has_initial_state[batch_id]) {
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] =
              static_cast<float>(ssm_state_ptr
                                     [num_v_heads_id * head_k_dim * head_v_dim +
                                      (k_bucket_size * sg_local_id + i) +
                                      (head_v_dim_id + j) * head_k_dim]);
        }
      }
    } else {
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
#pragma unroll
        for (int j = 0; j < v_dim_per_sg; ++j) {
          state_local[i * v_dim_per_sg + j] = 0.0f;
        }
      }
    }

    int seq_start_offset = query_start_loc[batch_id];
    int seq_end_offset = query_start_loc[batch_id + 1];

    // The state of each token is calculated iteratively.
    for (int t = seq_start_offset; t < seq_end_offset; ++t) {
      // act beta(t), g(t)
      float b_local = b[t * num_v_heads + num_v_heads_id];
      float beta = act_sigmoid(b_local);
      float a_local = a[t * num_v_heads + num_v_heads_id] + dt_bias_local;
      float g = sycl::exp(A_log_local * act_softplus(a_local));

      float q_sum = 0.0f;
      float k_sum = 0.0f;
// load q(t), k(t) and l2norm
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] =
            q[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        k_local[i] =
            k[t * num_k_heads * head_k_dim +
              (num_v_heads_id / kv_ratio) * head_k_dim +
              (k_bucket_size * sg_local_id + i)];
        q_sum += q_local[i] * q_local[i];
        k_sum += k_local[i] * k_local[i];
      }
      q_sum = sycl::reduce_over_group(sg, q_sum, sycl::plus<>());
      k_sum = sycl::reduce_over_group(sg, k_sum, sycl::plus<>());
      q_sum += eps;
      k_sum += eps;
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        q_local[i] /= sycl::sqrt(q_sum);
        q_local[i] *= scale;
        k_local[i] /= sycl::sqrt(k_sum);
      }

      float kv_mem[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = 0.0f;
      }

#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          state_local[j * k_bucket_size + i] *= g;
          kv_mem[j] += state_local[j * k_bucket_size + i] * k_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        kv_mem[i] = sycl::reduce_over_group(sg, kv_mem[i], sycl::plus<>());
      }

#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        v_local[i] =
            v[t * num_v_heads * head_v_dim + num_v_heads_id * head_v_dim +
              head_v_dim_id + i];
      }
      float delta[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        delta[i] = (v_local[i] - kv_mem[i]) * beta;
      }

      float res[v_dim_per_sg];
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = 0.0f;
      }
#pragma unroll
      for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
        for (int i = 0; i < k_bucket_size; ++i) {
          // get S(t)
          state_local[j * k_bucket_size + i] += k_local[i] * delta[j];
          // get O(t)
          res[j] += state_local[j * k_bucket_size + i] * q_local[i];
        }
      }
#pragma unroll
      for (int i = 0; i < v_dim_per_sg; ++i) {
        res[i] = sycl::reduce_over_group(sg, res[i], sycl::plus<>());
      }

      // store O(t) -- core_attn_out is the GLOBAL active buffer; remap
      // local token id `t` through token_indx if it was provided.
      if (sg_local_id == 0) {
        const int global_t = (token_indx != nullptr) ? token_indx[t] : t;
#pragma unroll
        for (int i = 0; i < v_dim_per_sg; ++i) {
          core_attn_out
              [global_t * num_v_heads * head_v_dim +
               num_v_heads_id * head_v_dim + head_v_dim_id + i] = res[i];
        }
      }
    }

// update state
#pragma unroll
    for (int j = 0; j < v_dim_per_sg; ++j) {
#pragma unroll
      for (int i = 0; i < k_bucket_size; ++i) {
        ssm_state_ptr
            [num_v_heads_id * head_k_dim * head_v_dim +
             (k_bucket_size * sg_local_id + i) +
             (head_v_dim_id + j) * head_k_dim] =
                static_cast<StateT>(state_local[j * k_bucket_size + i]);
      }
    }
  }

 private:
  T* core_attn_out;
  const T* q;
  const T* k;
  const T* v;
  const T* b;
  const T* a;
  const float* A_log;
  const T* dt_bias;
  StateT* ssm_state;
  const int ssm_state_stride_0;
  const int* query_start_loc;
  const int* token_indx;
  const int* cache_indices;
  const bool* has_initial_state;
  const int* num_accepted_tokens;
  const int batch_size;
  const int total_seqlen;
  const int num_k_heads;
  const int head_k_dim;
  const int num_v_heads;
  const int head_v_dim;
};


} // namespace gdn::fp16_decode_producer
