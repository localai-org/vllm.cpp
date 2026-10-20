#include "xpu_test_helpers.h"
#include "vt/xpu_sampling.h"
#include "vt/xpu_logsoftmax_policy.h"
#include "vt/xpu/xpu_logsoftmax_split.h"
#include "vt/breakable_graph.h"
#include "vt/xpu.h"
#include "vllm/v1/sample/sampler.h"
#include <bit>
#include <cstdlib>
#include <limits>
#include <string>
namespace {
using xpu_test::Buffer;using xpu_test::Queue;using vt::DType;
struct Flag {
 const char* name;bool had;std::string old;
 Flag(const char* n,const char* value):name(n),had(std::getenv(n)),old(had?std::getenv(n):""){setenv(n,value,1);}
 ~Flag(){if(had)setenv(name,old.c_str(),1);else unsetenv(name);}
};
struct Snapshot {
 float* exponential;float* lanes;float* stats;
 void Maximum(float x) const {stats[0]=x;}
 void Exponential(int64_t j,float x) const {exponential[j]=x;}
 void LaneSum(int64_t lane,float x) const {lanes[lane]=x;}
 void LogSumExp(float x) const {stats[1]=x;}
};
struct Stages {
 Buffer exponentials,lanes,stats;
 Stages(vt::Queue& q,int64_t n):exponentials(q,DType::kF32,{n}),lanes(q,DType::kF32,{128}),stats(q,DType::kF32,{2}){}
 Snapshot observer(){return {static_cast<float*>(exponentials.tensor.data),static_cast<float*>(lanes.tensor.data),static_cast<float*>(stats.tensor.data)};}
};
// Verbatim Normalize<1> reduction, with independent diagnostic checkpoint stores.
void Original(vt::Queue& q,Buffer& out,const Buffer& in,Stages& stages){
 const auto n=in.tensor.shape[1];const auto* src=static_cast<const float*>(in.tensor.data);auto* dst=static_cast<float*>(out.tensor.data);const auto observer=stages.observer();
 vt::xpu::NativeQueue(q).parallel_for(sycl::nd_range<1>(128,128),[=](sycl::nd_item<1> item){
  const int64_t lane=item.get_local_id(0);float maximum=-std::numeric_limits<float>::infinity();
  for(int64_t j=lane;j<n;j+=128)maximum=sycl::fmax(maximum,src[j]);
  maximum=sycl::reduce_over_group(item.get_group(),maximum,sycl::maximum<float>());
  if(lane==0)observer.Maximum(maximum);
  float sum=0;
  for(int64_t j=lane;j<n;j+=128){const float x=sycl::exp(src[j]-maximum);observer.Exponential(j,x);sum+=x;}
  observer.LaneSum(lane,sum);
  sum=sycl::reduce_over_group(item.get_group(),sum,sycl::plus<float>());
  const float lse=maximum+sycl::log(sum);if(lane==0)observer.LogSumExp(lse);
  for(int64_t j=lane;j<n;j+=128)dst[j]=src[j]-lse;
 });
}
std::vector<float> Values(int n,int kind){
 std::vector<float> v(n);for(int i=0;i<n;++i)v[i]=(int((i*37u)%32749)-16374)*0.00390625f;
 if(kind==1)std::fill(v.begin(),v.end(),0.f);
 if(kind==2){std::fill(v.begin(),v.end(),-0.f);v.back()=0.f;}
 if(kind==3)std::fill(v.begin(),v.end(),-std::numeric_limits<float>::infinity());
 if(kind==4){std::fill(v.begin(),v.end(),-1.f);v.back()=std::numeric_limits<float>::infinity();}
 if(kind==5)v.back()=std::bit_cast<float>(uint32_t{0x7fc00035});
 if(kind==6)for(int i=0;i<n;++i)v[i]=i%2?88.5f:-104.f;
 return v;
}
void Detect(vt::Queue& q,Buffer& input){
 const auto n=input.tensor.shape[1];Buffer old(q,DType::kF32,{1,n}),normal(q,DType::kF32,{1,n}),candidate(q,DType::kF32,{1,n}),state(q,DType::kF32,{2});Stages a(q,n),b(q,n);
 vt::ComputeLogprobs(q,normal.tensor,input.tensor);Original(q,old,input,a);
 vt::xpu::detail::SubmitSplitRawLogsoftmax(q,static_cast<float*>(candidate.tensor.data),static_cast<const float*>(input.tensor.data),static_cast<float*>(state.tensor.data),n,b.observer());
 xpu_test::SameBytes(old.download(),normal.download());xpu_test::SameBytes(candidate.download(),normal.download());
 xpu_test::SameBytes(a.exponentials.download(),b.exponentials.download());xpu_test::SameBytes(a.lanes.download(),b.lanes.download());xpu_test::SameBytes(a.stats.download(),b.stats.download());
 // Diagnostic stores must not be required for the production no-observer result.
 vt::xpu::detail::SubmitSplitRawLogsoftmax(q,static_cast<float*>(candidate.tensor.data),static_cast<const float*>(input.tensor.data),static_cast<float*>(state.tensor.data),n);
 xpu_test::SameBytes(candidate.download(),normal.download());
 if(n==248320){vt::xpu::SplitRawLogprobsM1(q,candidate.tensor,input.tensor,state.tensor);xpu_test::SameBytes(candidate.download(),normal.download());}
}
}
TEST_CASE("XPU split raw logsoftmax: exact exp lane sums LSE and full snapshots"){
 Queue gpu(vt::DeviceType::kXPU);
 for(int n:{1,127,128,129,248319,248320})for(int kind=0;kind<7;++kind){
  CAPTURE(n);
  CAPTURE(kind);
  Buffer input(gpu.q,DType::kF32,{1,n});input.put(Values(n,kind));const auto before=input.download();Detect(gpu.q,input);xpu_test::SameBytes(input.download(),before);
 }
}
TEST_CASE("XPU split raw logsoftmax: policy and atomic metadata alias rejection"){
 Flag flag("VT_XPU_RAW_LOGSOFTMAX_SPLIT","0");unsetenv(flag.name);CHECK(vt::XpuRawLogsoftmaxSplit(1,248320));setenv(flag.name,"0",1);CHECK_FALSE(vt::XpuRawLogsoftmaxSplit(1,248320));setenv(flag.name,"1",1);CHECK(vt::XpuRawLogsoftmaxSplit(1,248320));
 for(int n:{0,2,4})CHECK_FALSE(vt::XpuRawLogsoftmaxSplit(n,248320));CHECK_FALSE(vt::XpuRawLogsoftmaxSplit(1,248319));
 for(const char* bad:{"","2","01"," 1","1x"}){setenv(flag.name,bad,1);CHECK_THROWS(vt::XpuRawLogsoftmaxSplit(1,248320));CHECK_FALSE(vt::XpuRawLogsoftmaxSplit(4,248320));}
 Queue gpu(vt::DeviceType::kXPU);constexpr int n=248320;Buffer input(gpu.q,DType::kF32,{1,n}),output(gpu.q,DType::kF32,{1,n}),state(gpu.q,DType::kF32,{2});input.put(Values(n,0));output.put(std::vector<float>(n,123.f));state.put({42.f,43.f});
 const auto old=input.download(),old_out=output.download(),old_state=state.download();
 const auto refuse=[&](vt::Tensor in,vt::Tensor out,vt::Tensor scratch){CHECK_THROWS(vt::xpu::SplitRawLogprobsM1(gpu.q,out,in,scratch));xpu_test::SameBytes(input.download(),old);xpu_test::SameBytes(output.download(),old_out);xpu_test::SameBytes(state.download(),old_state);};
 auto bad=input.tensor;bad.stride[0]++;refuse(bad,output.tensor,state.tensor);
 bad=input.tensor;bad.shape[1]--;refuse(bad,output.tensor,state.tensor);
 bad=output.tensor;bad.dtype=DType::kF16;refuse(input.tensor,bad,state.tensor);
 bad=state.tensor;bad.shape[0]=1;refuse(input.tensor,output.tensor,bad);
 bad=state.tensor;bad.data=static_cast<unsigned char*>(bad.data)+1;refuse(input.tensor,output.tensor,bad);
 bad=state.tensor;bad.data=input.tensor.data;refuse(input.tensor,output.tensor,bad);
 bad=state.tensor;bad.data=output.tensor.data;refuse(input.tensor,output.tensor,bad);
 bad=output.tensor;bad.data=input.tensor.data;refuse(input.tensor,bad,state.tensor);
 bad=input.tensor;bad.device.index=1;refuse(bad,output.tensor,state.tensor);
 bad=input.tensor;bad.data=nullptr;refuse(bad,output.tensor,state.tensor);
}
TEST_CASE("XPU split raw logsoftmax: independent graph queues and consumer owners"){
 Queue a(vt::DeviceType::kXPU),b(vt::DeviceType::kXPU);constexpr int n=248320;
 Buffer ia(a.q,DType::kF32,{1,n}),ib(b.q,DType::kF32,{1,n}),oa(a.q,DType::kF32,{1,n}),ob(b.q,DType::kF32,{1,n}),sa(a.q,DType::kF32,{2}),sb(b.q,DType::kF32,{2}),ref(a.q,DType::kF32,{1,n});
 auto& backend=vt::GetBackend(a.q.device);REQUIRE(backend.SupportsGraphCapture());ia.put(Values(n,0));ib.put(Values(n,1));
 const auto apply=[&](int i){vt::xpu::SplitRawLogprobsM1(i?b.q:a.q,i?ob.tensor:oa.tensor,i?ib.tensor:ia.tensor,i?sb.tensor:sa.tensor);};apply(0);apply(1);backend.Synchronize(a.q);backend.Synchronize(b.q);
 vt::BreakableGraph graphs[2];for(int i=0;i<2;++i){vt::GraphCaptureScope capture(backend,i?b.q:a.q,graphs[i],vt::GraphCaptureMode::kFull);apply(i);}REQUIRE(graphs[0].captured());REQUIRE(graphs[1].captured());
 const auto memory=vt::xpu::GetMemoryInfo();
 for(int kind:{0,2,5}){
  ia.put(Values(n,kind));ib.put(Values(n,kind));auto event=backend.CreateEvent();backend.RecordEvent(event,a.q);backend.QueueWaitEvent(b.q,event);
  graphs[1].Replay(b.q);graphs[0].Replay(a.q);vt::ComputeLogprobs(a.q,ref.tensor,ia.tensor);const auto expected=ref.download();xpu_test::SameBytes(oa.download(),expected);xpu_test::SameBytes(ob.download(),expected);backend.DestroyEvent(event);
 }
 const auto after=vt::xpu::GetMemoryInfo();CHECK(after.allocated_bytes==memory.allocated_bytes);CHECK(after.graph_device_bytes==memory.graph_device_bytes);CHECK(after.graph_count==memory.graph_count);for(auto& graph:graphs)graph.Reset();
}
TEST_CASE("XPU split raw logsoftmax: full sampler raw-before-processors and API fallbacks"){
 Queue gpu(vt::DeviceType::kXPU);Flag flag("VT_XPU_RAW_LOGSOFTMAX_SPLIT","0"),compact("VT_XPU_COMPACT_LOGPROBS0","0");constexpr int n=248320;
 Buffer input(gpu.q,DType::kF32,{1,n});auto values=Values(n,0);vllm::v1::Sampler control,candidate;
 for(int kind=0;kind<6;++kind){
  vllm::v1::SamplingMetadata sm;sm.all_greedy=true;sm.all_random=false;sm.max_num_logprobs=0;
  sm.logit_bias[0][n-1]=1000.f;
  if(kind==1)sm.max_num_logprobs=3;
  if(kind==2)sm.max_num_logprobs.reset();
  if(kind==3)sm.logprob_token_ids=std::map<int,std::vector<int32_t>>{{0,{3,n-1}}};
  if(kind==4)setenv(compact.name,"1",1);
  if(kind==5){sm.all_greedy=false;sm.all_random=true;sm.temperature=std::vector<float>{.7f};sm.generators[0]=1234;sm.output_token_positions={11};}
  input.put(values);setenv(flag.name,"0",1);auto a=control.forward(gpu.q,input.tensor,sm);
  input.put(values);setenv(flag.name,"1",1);auto b=candidate.forward(gpu.q,input.tensor,sm);
  CHECK(a.sampled_token_ids==b.sampled_token_ids);REQUIRE(a.logprobs_tensors.has_value()==b.logprobs_tensors.has_value());
  if(a.logprobs_tensors){auto& x=*a.logprobs_tensors;auto& y=*b.logprobs_tensors;CHECK(x.logprob_token_ids==y.logprob_token_ids);CHECK(x.selected_token_ranks==y.selected_token_ranks);CHECK(x.logprobs.size()==y.logprobs.size());CHECK(std::memcmp(x.logprobs.data(),y.logprobs.data(),x.logprobs.size()*4)==0);}
 }
 const auto warm=vt::xpu::GetMemoryInfo();
 for(int i=0;i<2;++i){vllm::v1::SamplingMetadata sm;sm.all_greedy=true;sm.all_random=false;sm.max_num_logprobs=0;input.put(values);(void)candidate.forward(gpu.q,input.tensor,sm);}
 const auto steady=vt::xpu::GetMemoryInfo();CHECK(steady.allocated_bytes==warm.allocated_bytes);CHECK(steady.pinned_bytes==warm.pinned_bytes);
}
