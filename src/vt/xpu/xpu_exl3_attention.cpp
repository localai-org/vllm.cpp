// SDPA graph transcribed from exl3xpu c59d9442, csrc/exl3_ops.sycl and
// exl3xpu/fp8kv_prefill.py (MIT; Copyright 2026 exl3xpu contributors).
// Native ownership/accounting and exact paged-cache addressing are local.
#include "xpu_common.h"
#include "xpu_fp8.h"
#include "xpu_gptq4.h"
#include "xpu_kernels.h"

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <cmath>
#include <cstdio>
#include <map>
#include <memory>
#include <mutex>
#include <tuple>

namespace vt::xpu {
namespace {
using namespace dnnl::graph;
using LT = logical_tensor;
struct Entry {
  compiled_partition partition;
  std::vector<LT> inputs, outputs;
};
using Key = std::tuple<int64_t, int64_t, int64_t, int64_t>;
struct Cache {
  // Execution is eager and completed before this lock is released; eviction
  // cannot retire a compiled partition that still has device consumers.
  std::mutex mutex;
  std::map<Key, std::shared_ptr<Entry>> entries;
};
std::mutex caches_mutex;
std::map<int, std::unique_ptr<Cache>> caches;
constexpr size_t kPartitionCapacity = 16;
Cache& DeviceCache(int device) {
  std::lock_guard<std::mutex> lock(caches_mutex);
  auto& cache = caches[device];
  if (!cache) cache = std::make_unique<Cache>();
  return *cache;
}
std::shared_ptr<Entry> Partition(Cache& cache, dnnl::engine& engine,
                                 int64_t Q, int64_t L, int64_t G, int64_t D) {
  const Key signature{Q, L, G, D};
  if (auto it = cache.entries.find(signature); it != cache.entries.end()) return it->second;
  // Bound the library's additional compiled-partition cache as well.
  static std::once_flag library_cache;
  std::call_once(library_cache, [] { set_compiled_partition_cache_capacity(kPartitionCapacity); });
  using dt = LT::data_type;
  size_t id = 0;
  LT query(id++, dt::f16, {1, 1, G, Q, D}, {G*Q*D, G*Q*D, Q*D, D, 1});
  LT key(id++, dt::f16, {1, 1, 1, L, D}, {L*D, L*D, L*D, D, 1});
  LT score(id++, dt::f32, {1, 1, G, Q, L}, LT::layout_type::strided);
  op bmm1(id++, op::kind::MatMul, "bmm1");
  bmm1.set_attr<bool>(op::attr::transpose_b, true);
  bmm1.add_inputs({query, key}); bmm1.add_outputs({score});
  LT scale(id++, dt::f16, LT::dims{1}, LT::layout_type::strided);
  LT scaled(id++, dt::f32, {1, 1, G, Q, L}, LT::layout_type::strided);
  op divide(id++, op::kind::Divide, "scale");
  divide.add_inputs({score, scale}); divide.add_outputs({scaled});
  LT row(id++, dt::s32, {1, 1, G, Q, L}, LT::layout_type::strided);
  op rows(id++, op::kind::GenIndex, "row");
  rows.set_attr<int64_t>(op::attr::axis, -2); rows.add_inputs({scaled}); rows.add_outputs({row});
  LT lk(id++, dt::s32, 0, LT::layout_type::strided, LT::property_type::host_scalar);
  LT ra(id++, dt::s32, {1, 1, G, Q, L}, LT::layout_type::strided);
  op add(id++, op::kind::Add, "row+L"); add.add_inputs({row, lk}); add.add_outputs({ra});
  LT lq(id++, dt::s32, 0, LT::layout_type::strided, LT::property_type::host_scalar);
  LT rs(id++, dt::s32, {1, 1, G, Q, L}, LT::layout_type::strided);
  op sub(id++, op::kind::Subtract, "-Q"); sub.add_inputs({ra, lq}); sub.add_outputs({rs});
  LT col(id++, dt::s32, {1, 1, G, Q, L}, LT::layout_type::strided);
  op cols(id++, op::kind::GenIndex, "col");
  cols.set_attr<int64_t>(op::attr::axis, -1); cols.add_inputs({scaled}); cols.add_outputs({col});
  LT ge(id++, dt::boolean, {1, 1, G, Q, L}, LT::layout_type::strided);
  op compare(id++, op::kind::GreaterEqual, "ge");
  compare.add_inputs({rs, col}); compare.add_outputs({ge});
  LT negative_inf(id++, dt::f32, LT::dims{1}, LT::layout_type::strided);
  LT masked(id++, dt::f32, {1, 1, G, Q, L}, LT::layout_type::strided);
  op select(id++, op::kind::Select, "sel");
  select.add_inputs({ge, scaled, negative_inf}); select.add_outputs({masked});
  LT probs(id++, dt::f16, {1, 1, G, Q, L}, LT::layout_type::strided);
  op softmax(id++, op::kind::SoftMax, "softmax");
  softmax.set_attr<int64_t>(op::attr::axis, -1);
  softmax.set_attr<std::string>(op::attr::mode, "inf_as_zero");
  softmax.add_inputs({masked}); softmax.add_outputs({probs});
  LT value(id++, dt::f16, {1, 1, 1, L, D}, {L*D, L*D, L*D, D, 1});
  LT output(id++, dt::f16, {1, 1, G, Q, D}, {G*Q*D, G*Q*D, Q*D, D, 1});
  op bmm2(id++, op::kind::MatMul, "bmm2"); bmm2.add_inputs({probs, value}); bmm2.add_outputs({output});
  graph graph(dnnl::engine::kind::gpu);
  for (auto& operation : {bmm1, divide, rows, add, sub, cols, compare, select, softmax, bmm2})
    graph.add_op(operation);
  graph.finalize();
  auto parts = graph.get_partitions();
  VT_CHECK(parts.size() == 1, "EXL3 oneDNN SDPA graph did not fuse to one partition");
  std::vector<LT> inputs{query, key, scale, lk, lq, negative_inf, value}, outputs{output};
  auto compiled = parts[0].compile(inputs, outputs, engine);
  auto entry = std::make_shared<Entry>(Entry{compiled, inputs, outputs});
  if (cache.entries.size() == kPartitionCapacity) cache.entries.erase(cache.entries.begin());
  cache.entries.emplace(signature, entry);
  return entry;
}
}  // namespace

bool PagedAttentionExl3OneDnnKernel(Queue& q, Tensor& out, const Tensor& query,
    const Tensor& key_cache, const Tensor& value_cache, const Tensor& block_table,
    const Tensor& seq_lens, const Tensor& query_start_loc, const PagedAttentionArgs& args) {
  if (query.dtype != DType::kF16 || out.dtype != DType::kF16 ||
      query.shape[1] != 24 || query.shape[2] != 256 || key_cache.shape[2] != 4 ||
      key_cache.dtype != DType::kI8 || value_cache.dtype != DType::kI8 ||
      args.kv_cache_dtype != Fp8KVCacheDataType::kFp8E4M3 || !args.causal ||
      args.logits_soft_cap != 0 ||
      (args.window_size && (args.window_size->left != -1 || args.window_size->right != -1)) ||
      !std::isfinite(args.scale) || args.scale <= 0 ||
      !std::isfinite(args.k_scale) || args.k_scale <= 0 ||
      !std::isfinite(args.v_scale) || args.v_scale <= 0) return false;
  const float inverse = 1.0f / args.scale;
  if (!std::isfinite(inverse) || inverse <= 0 || inverse > 65504) return false;
  const auto requests = seq_lens.Numel();
  if (requests < 1 || requests > 4) return false;
  auto& backend = GetBackend(q.device);
  std::vector<int32_t> lengths(requests), offsets(requests + 1);
  // Exact device lengths, never max_seq_len / a CPU capacity upper bound.
  backend.Copy(q, lengths.data(), seq_lens.data, lengths.size()*sizeof(int32_t));
  backend.Copy(q, offsets.data(), query_start_loc.data, offsets.size()*sizeof(int32_t));
  backend.Synchronize(q);
  for (int64_t r = 0; r < requests; ++r) {
    const auto rows = offsets[r+1] - offsets[r];
    if (rows != 0 && (rows < 2 || rows > 4096 || lengths[r] < rows || lengths[r] > 262144))
      return false;
  }
  auto& cache = DeviceCache(q.device.index);
  std::lock_guard<std::mutex> execution(cache.mutex);
  auto& engine = Exl3OneDnnEngine(q);
  auto& stream = Exl3OneDnnStream(q);
  const View qs(query), ks(key_cache), vs(value_cache), bt(block_table), dest(out);
  constexpr int64_t G = 6, D = 256;
  const auto page = key_cache.shape[1];
  const float key_scale = args.k_scale, value_scale = args.v_scale;
  for (int64_t request = 0; request < requests; ++request) {
    const int64_t rows = offsets[request+1] - offsets[request], length = lengths[request];
    if (!rows) continue;
    const int64_t padded = (rows + 255) / 256 * 256, first = offsets[request];
    // One KV head at a time. Caller-owned buffers are released only after the
    // SDPA and the explicit stride-aware output copy have completed.
    Scratch storage(q.device, size_t(2*length*D + 2*G*padded*D)*2 + 64);
    auto* kb = static_cast<sycl::half*>(storage.data);
    auto* vb = kb + length*D;
    auto* qg = vb + length*D;
    auto* og = qg + G*padded*D;
    auto* divisor = og + G*padded*D;
    auto* ninf = reinterpret_cast<float*>(divisor + 8);
    const uint16_t inverse_half = F32ToF16(inverse);
    const float negative_inf = -std::numeric_limits<float>::infinity();
    backend.Copy(q, divisor, &inverse_half, sizeof(inverse_half));
    backend.Copy(q, ninf, &negative_inf, sizeof(negative_inf));
    backend.Memset(q, qg, 0, size_t(G*padded*D)*2);
    auto entry = Partition(cache, engine, padded, length, G, D);
    const auto scratch_lt = entry->partition.get_scratchpad_logical_tensor();
    auto* scratch = Exl3OneDnnScratchpad(q, scratch_lt.get_mem_size());
    for (int64_t head = 0; head < 4; ++head) {
      NativeQueue(q).parallel_for(sycl::range<1>(length*D), [=](sycl::id<1> item) {
        const int64_t token = item[0]/D, dim = item[0]%D;
        const int64_t block = static_cast<const int32_t*>(bt.data)[
            request*bt.stride[0] + (token/page)*bt.stride[1]];
        const auto ki = block*ks.stride[0] + (token%page)*ks.stride[1] + head*ks.stride[2] + dim*ks.stride[3];
        const auto vi = block*vs.stride[0] + (token%page)*vs.stride[1] + head*vs.stride[2] + dim*vs.stride[3];
        kb[item[0]] = sycl::half(LoadKV(ks, ki, key_scale));
        vb[item[0]] = sycl::half(LoadKV(vs, vi, value_scale));
      });
      NativeQueue(q).parallel_for(sycl::range<1>(G*rows*D), [=](sycl::id<1> item) {
        const int64_t dim = item[0]%D, row = (item[0]/D)%rows, group = item[0]/(rows*D);
        qg[(group*padded + padded-rows+row)*D + dim] = sycl::half(Load(qs,
            (first+row)*qs.stride[0] + (head*G+group)*qs.stride[1] + dim*qs.stride[2]));
      });
      int32_t lk = static_cast<int32_t>(length), lq = static_cast<int32_t>(padded);
      const auto& ins = entry->inputs;
      std::vector<tensor> inputs{tensor(ins[0], engine, qg), tensor(ins[1], engine, kb),
          tensor(ins[2], engine, divisor), tensor::make_scalar_tensor(ins[3], &lk),
          tensor::make_scalar_tensor(ins[4], &lq), tensor(ins[5], engine, ninf), tensor(ins[6], engine, vb)};
      std::vector<tensor> outputs{tensor(entry->outputs[0], engine, og)};
      auto done = dnnl::graph::sycl_interop::execute(entry->partition, stream, inputs, outputs,
          tensor(scratch_lt, engine, scratch));
      done.wait_and_throw();
      RecordProfileEvent(q, "exl3_onednn_sdpa", done);
      auto copy = NativeQueue(q).parallel_for(sycl::range<1>(G*rows*D), [=](sycl::id<1> item) {
        const int64_t dim = item[0]%D, row = (item[0]/D)%rows, group = item[0]/(rows*D);
        static_cast<sycl::half*>(dest.data)[(first+row)*dest.stride[0] +
            (head*G+group)*dest.stride[1] + dim*dest.stride[2]] =
            og[(group*padded + padded-rows+row)*D + dim];
      });
      copy.wait_and_throw();
    }
    if (const char* trace = std::getenv("VT_XPU_TRACE_FAST_PATH"); trace && trace[0] == '1')
      std::fprintf(stderr, "EXL3_SDPA exact_k=%lld logical_q=%lld padded_q=%lld request=%lld partitions=%zu capacity=%zu scale=%g k_scale=%g v_scale=%g scratch=%zu\n",
          static_cast<long long>(length), static_cast<long long>(rows), static_cast<long long>(padded),
          static_cast<long long>(request), cache.entries.size(), kPartitionCapacity,
          args.scale, args.k_scale, args.v_scale, scratch_lt.get_mem_size());
  }
  return true;
}
}  // namespace vt::xpu
