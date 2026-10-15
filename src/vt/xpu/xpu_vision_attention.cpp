#include "xpu_vision_attention.h"
#include "xpu_common.h"
#include "xpu_gptq4.h"

#include <oneapi/dnnl/dnnl_graph.hpp>
#include <oneapi/dnnl/dnnl_graph_sycl.hpp>
#include <cmath>
#include <bit>
#include <cstdio>
#include <cstdlib>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <vector>

namespace vt::xpu {
namespace {
using namespace dnnl::graph;
using LT = logical_tensor;
constexpr size_t kCapacity = 8;
struct Entry {
  compiled_partition partition;
  std::vector<LT> inputs, outputs;
  LT scratch;
  size_t scratch_bytes;
  // In-order queue's latest event covers every earlier use on that queue.
  std::map<uint64_t, sycl::event> consumers;
};
struct Cache { std::map<int64_t, std::shared_ptr<Entry>> entries; };
std::mutex cache_mutex;
std::map<int, Cache> caches;

void Prune(Entry& entry) {
  for (auto it = entry.consumers.begin(); it != entry.consumers.end();) {
    if (it->second.get_info<sycl::info::event::command_execution_status>() ==
        sycl::info::event_command_status::complete) it = entry.consumers.erase(it);
    else ++it;
  }
}

std::shared_ptr<Entry> GetEntry(Cache& cache, dnnl::engine& engine, int64_t length) {
  if (auto it = cache.entries.find(length); it != cache.entries.end()) return it->second;
  static std::once_flag library_cache;
  std::call_once(library_cache, [] { set_compiled_partition_cache_capacity(16); });
  constexpr int64_t heads = 16, dim = 72;
  const LT::dims shape{1, heads, length, dim};
  const LT::dims strides{length*heads*dim, dim, heads*dim, 1};
  const LT::dims score_shape{1, heads, length, length};
  using dt = LT::data_type;
  size_t id = 0;
  LT query(id++,dt::f16,shape,strides),key(id++,dt::f16,shape,strides);
  LT score(id++,dt::f32,score_shape,LT::layout_type::strided);
  op qk(id++,op::kind::MatMul,"vision_qk");
  qk.set_attr<bool>(op::attr::transpose_b,true);
  qk.add_inputs({query,key});qk.add_outputs({score});
  LT scale(id++,dt::f32,LT::dims{1},LT::layout_type::strided);
  LT scaled(id++,dt::f32,score_shape,LT::layout_type::strided);
  op multiply(id++,op::kind::Multiply,"vision_scale");
  multiply.add_inputs({score,scale});multiply.add_outputs({scaled});
  LT probabilities(id++,dt::f16,score_shape,LT::layout_type::strided);
  op softmax(id++,op::kind::SoftMax,"vision_softmax");
  softmax.set_attr<int64_t>(op::attr::axis,-1);
  softmax.set_attr<std::string>(op::attr::mode,"inf_as_zero");
  softmax.add_inputs({scaled});softmax.add_outputs({probabilities});
  LT value(id++,dt::f16,shape,strides),output(id++,dt::f16,shape,strides);
  op pv(id++,op::kind::MatMul,"vision_pv");
  pv.add_inputs({probabilities,value});pv.add_outputs({output});
  graph graph(dnnl::engine::kind::gpu);
  for (const auto& operation : {qk,multiply,softmax,pv}) graph.add_op(operation);
  graph.finalize();
  auto partitions=graph.get_partitions();
  VT_CHECK(partitions.size()==1 && partitions[0].is_supported(),
           "XPU vision SDPA must fuse into one supported partition");
  std::vector<LT> inputs{query,key,scale,value},outputs{output};
  auto partition=partitions[0].compile(inputs,outputs,engine);
  auto scratch=partition.get_scratchpad_logical_tensor();
  const size_t bytes=scratch.get_mem_size();
  // Refuse a library decomposition needing quadratic score/probability space
  // before allocating/executing. The allowance is linear in the input size.
  const size_t bound=65536+size_t(length*heads*dim)*sizeof(uint16_t);
  VT_CHECK(bytes<=bound,"XPU vision SDPA scratch exceeds its linear memory bound");
  auto entry=std::make_shared<Entry>(Entry{partition,inputs,outputs,scratch,bytes,{}});
  if (cache.entries.size()==kCapacity) {
    auto victim=cache.entries.begin();
    // Retiring a plan is exceptional, not a wait on every inference call.
    for (auto& [_,event] : victim->second->consumers) event.wait_and_throw();
    cache.entries.erase(victim);
  }
  cache.entries.emplace(length,entry);
  return entry;
}
struct ProfileAnchor { void operator()() const {} };
}

void VisionAttentionDenseFlashKernel(Queue& q, Tensor& out, const Tensor& query,
    const Tensor& key, const Tensor& value, const AttentionArgs& args) {
  VT_CHECK(query.dtype==DType::kF16 && out.dtype==DType::kF16 &&
               key.dtype==DType::kF16 && value.dtype==DType::kF16,
           "XPU vision dense flash requires FP16 operands");
  VT_CHECK(query.shape[1]==16 && key.shape[1]==16 && value.shape[1]==16 &&
               query.shape[2]==72 && query.shape[0]>0 && query.shape[0]<=16384,
           "XPU vision dense flash requires bounded [L,16,72] operands");
  VT_CHECK(!args.causal && std::isfinite(args.scale) && args.scale>0,
           "XPU vision dense flash requires non-causal attention and finite positive scale");
  VT_CHECK(!Overlap(out,query) && !Overlap(out,key) && !Overlap(out,value),
           "XPU vision dense flash output must not overlap Q/K/V");
  TraceXpuOp(OpId::kAttentionDenseFlash,q,{&out,&query,&key,&value});
  std::lock_guard<std::mutex> lock(cache_mutex);
  // The reference-matching FP16 route is selected by device/layout admission,
  // independently of the text-prefill policy. Other devices retain oneDNN.
#ifdef VLLM_CPP_XPU_XE2_VISION
  if (VisionAttentionXe2Kernel(q,out,query,key,value,args)) return;
#endif
  auto& engine=Exl3OneDnnEngine(q);
  auto& stream=Exl3OneDnnStream(q);
  auto entry=GetEntry(caches[q.device.index],engine,query.shape[0]);
  Prune(*entry);
  const size_t aligned=(entry->scratch_bytes+63)/64*64;
  auto* storage=static_cast<char*>(Exl3OneDnnScratchpad(q,aligned+64));
  auto* scalar=reinterpret_cast<float*>(storage+aligned);
  const float scale=args.scale;
  NativeQueue(q).single_task([=] { *scalar=scale; });
  const auto& inputs=entry->inputs;
  std::vector<tensor> ins{tensor(inputs[0],engine,query.data),tensor(inputs[1],engine,key.data),
      tensor(inputs[2],engine,scalar),tensor(inputs[3],engine,value.data)};
  std::vector<tensor> outs{tensor(entry->outputs[0],engine,out.data)};
  std::optional<sycl::event> begin;
  if (ProfileQueueEventsEnabled()) begin=NativeQueue(q).single_task(ProfileAnchor{});
  const auto host_start=HostProfileClockNs();
  auto done=dnnl::graph::sycl_interop::execute(entry->partition,stream,ins,outs,
      tensor(entry->scratch,engine,storage));
  const auto host_duration=HostProfileClockNs()-host_start;
  entry->consumers.insert_or_assign(q.id,done);
  if (begin) {
    const auto end=NativeQueue(q).single_task(ProfileAnchor{});
    RecordProfileSpan(q,"vision_onednn_sdpa",*begin,end,host_duration,
                     "L="+std::to_string(query.shape[0])+" scratch="+std::to_string(entry->scratch_bytes));
  }
  if (const char* trace=std::getenv("VT_XPU_TRACE_FAST_PATH");trace && trace[0]=='1')
    std::fprintf(stderr,"VISION_SDPA L=%lld H=16 D=72 scratch=%zu bound=%zu plans=%zu scale_bits=0x%08x\n",
        static_cast<long long>(query.shape[0]),entry->scratch_bytes,
        65536+size_t(query.Numel())*2,caches[q.device.index].entries.size(),std::bit_cast<uint32_t>(args.scale));
}
}
