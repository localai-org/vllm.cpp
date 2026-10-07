// Paged-attention PREFILL A/B: paged vs dense KV source, bf16 vs fp8 cache, at
// the 27B full-attention shape (hq=32, hk=4, d=256, block_size=32).
//
// WHY AN ISOLATED SWEEP EXISTS: the engine-level fp8 prefill numbers (28s vs 8s
// at 8k) conflate the attention kernel, the per-layer dequant scratch, the D2H
// sync and the driver allocator. This runs ONE vt::PagedAttention call per arm
// on synthetic data, so the number is the kernel cost and nothing else.
//
// ARMS (one process per arm; the dispatch knobs are process-static):
//   BENCH_KV=bf16            paged bf16 cache        (the bf16 engine baseline)
//   BENCH_KV=fp8             paged fp8 cache         (dense scratch ON by default)
//   BENCH_KV=fp8 VT_ATTN_FP8_DENSE=0   per-read dequant inside the kernel
//   BENCH_KV=bf16 BENCH_DENSE=1        dense bf16 + identity table (layout control)
//
// ONE arm per process, so this binary cannot compare arms to each other. It
// prints a checksum of the arm's output, which makes a wildly wrong arm visible
// in the logs; it is not a parity gate. Parity is gated by
// `tests/vt/test_ops_paged_attn.cpp`, where each arm is compared against an f32
// reference on the exact dequantized values (< 5e-2 max abs err).
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/ops.h"

using vt::Backend;
using vt::DeviceType;
using vt::DType;
using vt::Fp8KVCacheDataType;
using vt::PagedAttentionArgs;
using vt::Queue;
using vt::Tensor;

namespace {

Tensor MakeT(void* data, DType dt, const std::vector<int64_t>& shape) {
  Tensor t;
  t.data = data;
  t.dtype = dt;
  t.device = vt::Device{DeviceType::kCUDA, 0};
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

std::vector<float> RandF32(size_t n, uint32_t seed) {
  std::vector<float> v(n);
  uint32_t s = seed;
  for (auto& x : v) {
    s = s * 1664525u + 1013904223u;
    x = (static_cast<float>(s >> 8) / static_cast<float>(1u << 24)) * 4.0f - 2.0f;
  }
  return v;
}

struct Buf {
  Backend& b;
  void* p = nullptr;
  size_t bytes = 0;
  Buf(Backend& backend, size_t n) : b(backend), bytes(n) { p = b.Alloc(n == 0 ? 1 : n); }
  ~Buf() { b.Free(p); }
  Buf(const Buf&) = delete;
  Buf& operator=(const Buf&) = delete;
};

}  // namespace

int main(int argc, char** argv) {
  const int64_t T = argc > 1 ? std::atoll(argv[1]) : 8192;
  const int64_t Hq = argc > 2 ? std::atoll(argv[2]) : 32;
  const int64_t Hk = argc > 3 ? std::atoll(argv[3]) : 4;
  const int64_t D = argc > 4 ? std::atoll(argv[4]) : 256;
  const int64_t BS = argc > 5 ? std::atoll(argv[5]) : 32;
  const int reps = argc > 6 ? std::atoi(argv[6]) : 3;
  const char* kv_env = std::getenv("BENCH_KV");
  const bool fp8 = kv_env != nullptr && std::strcmp(kv_env, "fp8") == 0;
  const char* dense_env = std::getenv("BENCH_DENSE");
  const bool dense_bf16 = !fp8 && dense_env != nullptr && dense_env[0] == '1';
  const float scale = std::pow(static_cast<float>(D), -0.5f);
  const int64_t N = T / BS;  // one request, identity block order

  Backend& gpu = vt::GetBackend(DeviceType::kCUDA);
  Queue q = gpu.CreateQueue();

  auto q_host = RandF32(static_cast<size_t>(T * Hq * D), 2024);
  auto k_host = RandF32(static_cast<size_t>(N * BS * Hk * D), 137);
  auto v_host = RandF32(static_cast<size_t>(N * BS * Hk * D), 179);
  std::vector<int32_t> block_table(static_cast<size_t>(N));
  for (int64_t i = 0; i < N; ++i) block_table[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<int32_t> seq_lens = {static_cast<int32_t>(T)};
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};

  Buf dq(gpu, static_cast<size_t>(T * Hq * D) * sizeof(float));
  Buf dbt(gpu, block_table.size() * sizeof(int32_t));
  Buf dsl(gpu, seq_lens.size() * sizeof(int32_t));
  Buf dqsl(gpu, qsl.size() * sizeof(int32_t));
  Buf dout(gpu, static_cast<size_t>(T * Hq * D) * sizeof(float));
  gpu.Copy(q, dq.p, q_host.data(), dq.bytes);
  gpu.Copy(q, dbt.p, block_table.data(), dbt.bytes);
  gpu.Copy(q, dsl.p, seq_lens.data(), dsl.bytes);
  gpu.Copy(q, dqsl.p, qsl.data(), dqsl.bytes);

  // Cache: paged [N, BS, Hk, D] (bf16 or fp8), or dense [1, T, Hk, D] bf16.
  const int64_t cache_elems = dense_bf16 ? T * Hk * D : N * BS * Hk * D;
  Buf kcache(gpu, static_cast<size_t>(cache_elems) * (fp8 ? 1 : 2));
  Buf vcache(gpu, static_cast<size_t>(cache_elems) * (fp8 ? 1 : 2));
  Tensor kt, vt;
  if (fp8) {
    std::vector<uint8_t> k8(k_host.size()), v8(v_host.size());
    for (size_t i = 0; i < k_host.size(); ++i) {
      k8[i] = vt::StoreKvFp8E4M3(k_host[i], 1.0f);
      v8[i] = vt::StoreKvFp8E4M3(v_host[i], 1.0f);
    }
    gpu.Copy(q, kcache.p, k8.data(), k8.size());
    gpu.Copy(q, vcache.p, v8.data(), v8.size());
    kt = MakeT(kcache.p, DType::kI8, {N, BS, Hk, D});
    vt = MakeT(vcache.p, DType::kI8, {N, BS, Hk, D});
  } else {
    std::vector<uint16_t> kb(k_host.size()), vb(v_host.size());
    for (size_t i = 0; i < k_host.size(); ++i) {
      kb[i] = vt::F32ToBF16(k_host[i]);
      vb[i] = vt::F32ToBF16(v_host[i]);
    }
    gpu.Copy(q, kcache.p, kb.data(), kb.size() * 2);
    gpu.Copy(q, vcache.p, vb.data(), vb.size() * 2);
    const std::vector<int64_t> shape =
        dense_bf16 ? std::vector<int64_t>{1, T, Hk, D} : std::vector<int64_t>{N, BS, Hk, D};
    kt = MakeT(kcache.p, DType::kBF16, shape);
    vt = MakeT(vcache.p, DType::kBF16, shape);
  }

  // Dense arm: one block per request, block_size = T, identity table.
  int32_t* bt_ptr = static_cast<int32_t*>(dbt.p);
  const int64_t bt_elems = N;
  const int64_t block_size = dense_bf16 ? T : BS;

  PagedAttentionArgs args{scale, /*causal=*/true};
  if (fp8) {
    args.kv_cache_dtype = Fp8KVCacheDataType::kFp8E4M3;
    args.k_scale = 1.0f;
    args.v_scale = 1.0f;
  }

  Tensor q_t = MakeT(dq.p, DType::kF32, {T, Hq, D});
  Tensor out_t = MakeT(dout.p, DType::kF32, {T, Hq, D});
  Tensor bt_t = MakeT(bt_ptr, DType::kI32, {1, dense_bf16 ? 1 : bt_elems});
  Tensor sl_t = MakeT(dsl.p, DType::kI32, {1});
  Tensor qsl_t = MakeT(dqsl.p, DType::kI32, {2});

  std::vector<float> got(static_cast<size_t>(T * Hq * D));
  double ms = 0.0;
  for (int r = 0; r < reps + 2; ++r) {
    const auto t0 = std::chrono::steady_clock::now();
    vt::PagedAttention(q, out_t, q_t, kt, vt, bt_t, sl_t, qsl_t, args);
    gpu.Synchronize(q);
    const double dt =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
    if (r >= 2) ms += dt;
  }
  ms /= reps;
  gpu.Copy(q, got.data(), dout.p, got.size() * sizeof(float));
  gpu.Synchronize(q);

  const char* arm = fp8 ? (std::getenv("VT_ATTN_FP8_DENSE") != nullptr ? "fp8/per-read" : "fp8/dense")
                        : (dense_bf16 ? "bf16/dense" : "bf16/paged");  std::printf("arm=%-12s T=%lld hq=%lld hk=%lld d=%lld bs=%lld  %.1f ms  %.0f tok/s\n", arm,
              static_cast<long long>(T), static_cast<long long>(Hq), static_cast<long long>(Hk),
              static_cast<long long>(D), static_cast<long long>(block_size), ms, T / (ms / 1000.0));
  // Checksum so a wrong-but-fast arm is visible.
  double sum = 0.0;
  for (float x : got) sum += x;
  std::printf("  checksum=%.6f\n", sum);
  gpu.DestroyQueue(q);
  return 0;
}
