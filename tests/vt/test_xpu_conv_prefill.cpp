#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/xpu_conv_policy.h"
#include "vt/breakable_graph.h"
#include <array>
#include <cstdlib>
#include <string>
#include <limits>
#include <map>

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
struct Policy {
  const char* name="VT_XPU_CONV_PREFILL_TYPED_TILE";
  bool had=std::getenv(name)!=nullptr;
  std::string old=had?std::getenv(name):"";
  ~Policy(){if(had)setenv(name,old.c_str(),1);else unsetenv(name);}
  void Set(int tile){REQUIRE(setenv(name,std::to_string(tile).c_str(),1)==0);}
};
struct Fixture {
  static constexpr int channels=10240;
  vt::Queue& q;
  int tokens,stride;
  Buffer input,weight,output,state,offsets,initial,bias;
  std::vector<float> history;
  std::vector<uint32_t> poison;
  Fixture(vt::Queue& queue,int n,int row_stride,bool byte_flag):q(queue),tokens(n),stride(row_stride),
    input(q,DType::kF16,{n+1,row_stride}),weight(q,DType::kF16,{channels,4}),
    output(q,DType::kF32,{n+1,channels}),state(q,DType::kF32,{2,channels,3}),
    offsets(q,DType::kI32,{2}),initial(q,byte_flag?DType::kI8:DType::kI32,{1}),bias(q,DType::kF32,{channels}) {
    std::vector<uint16_t> data(size_t(n+1)*stride,0x7e55);
    for(int t=0;t<n;++t) for(int c=0;c<channels;++c)
      data[size_t(t)*stride+c]=vt::F32ToF16(float(int((t*17+c*7)%127)-63)*.015625f);
    constexpr uint16_t special[]{0,0x8000,1,0x0400,0x7bff,0x7c00,0xfc00,0x7e01,0xfe55};
    if(n)for(size_t c=0;c<std::size(special);++c)data[c]=special[c];
    input.upload(data.data());weight.put(xpu_test::Values(channels*4,9,.125f));
    bias.put(xpu_test::Values(channels,7,.03125f));
    history=xpu_test::Values(2*channels*3,11,.015625f);
    for(size_t i=channels*3;i<history.size();++i)history[i]=std::numeric_limits<float>::quiet_NaN();
    poison.assign(size_t(n+1)*channels,0x7fa54321);
    const int32_t cu[]{0,n};offsets.upload(cu);
  }
  vt::Tensor Input(){auto t=input.tensor;t.shape[0]=tokens;t.shape[1]=channels;return t;}
  vt::Tensor Output(){auto t=output.tensor;t.shape[0]=tokens;return t;}
  vt::Tensor State(){auto t=state.tensor;t.shape[0]=1;return t;}
  void Reset(bool keep){
    state.put(history);output.upload(poison.data());
    if(initial.tensor.dtype==DType::kI8){const int8_t flag=keep;initial.upload(&flag);}
    else{const int32_t flag=keep;initial.upload(&flag);}
  }
  void Apply(bool has_bias,bool silu){
    auto x=Input(),out=Output(),cache=State();
    vt::CausalConv1dFwd(q,out,x,weight.tensor,has_bias?&bias.tensor:nullptr,cache,offsets.tensor,initial.tensor,{silu});
  }
};
void CheckRoute(int tile,bool admitted) {
  const auto events=vt::xpu::DrainProfileEvents();
  if(!std::getenv("VT_XPU_PROFILE") || std::string(std::getenv("VT_XPU_PROFILE"))!="1")return;
  const std::string output="conv1d_prefill_typed_"+std::to_string(tile);
  CHECK(std::count_if(events.begin(),events.end(),[&](const auto& e){return e.stage==output;})==(admitted?1:0));
  CHECK(std::count_if(events.begin(),events.end(),[](const auto& e){return e.stage=="conv1d_prefill_typed_state";})==(admitted?1:0));
}
}

TEST_CASE("XPU typed conv prefill: exact old GPU output history tails and bounded admission") {
  Queue gpu(vt::DeviceType::kXPU);Policy policy;
  const std::array<int,11> lengths{0,1,2,3,4,511,512,513,4095,4096,4097};
  for(size_t index=0;index<lengths.size();++index)for(int stride:{10240,16384}){
    const int n=lengths[index];CAPTURE(n);
    CAPTURE(stride);
    Fixture f(gpu.q,n,stride,index%2==0);
    const bool keep=(index+(stride==16384))%2,has_bias=index%2,silu=index%3!=0;
    policy.Set(0);f.Reset(keep);f.Apply(has_bias,silu);
    const auto expected=f.output.download(),history=f.state.download(),input=f.input.download();
    for(int tile:{1,2,4}){
      CAPTURE(tile);
      policy.Set(tile);f.Reset(keep);(void)vt::xpu::DrainProfileEvents();
      f.Apply(has_bias,silu);
      xpu_test::SameBytes(f.output.download(),expected);
      xpu_test::SameBytes(f.state.download(),history);
      xpu_test::SameBytes(f.input.download(),input);
      CheckRoute(tile,n>=512&&n<=4096);
    }
  }
}

namespace {
vt::Tensor Rows(vt::Tensor t,int start,int count) {
  t.data=static_cast<unsigned char*>(t.data)+size_t(start)*t.stride[0]*vt::SizeOf(t.dtype);
  t.shape[0]=count;return t;
}
}

TEST_CASE("XPU typed conv prefill: full split prefill decode and subsequent continuation are exact") {
  Queue gpu(vt::DeviceType::kXPU);Policy policy;
  for(int n:{512,513,4096})for(bool keep:{false,true}) {
    CAPTURE(n);
    CAPTURE(keep);
    Fixture f(gpu.q,n,16384,true);
    policy.Set(0);f.Reset(keep);f.Apply(true,true);
    const auto expected=f.output.download(),state=f.state.download();
    Buffer reference_state(gpu.q,DType::kF32,{2,Fixture::channels,3}),
      next(gpu.q,DType::kF16,{1,Fixture::channels}),expected_next(gpu.q,DType::kF32,{1,Fixture::channels});
    reference_state.upload(state.data());next.put(xpu_test::Values(Fixture::channels,19,.0625f));
    policy.Set(4);f.Reset(keep);
    const int prefix=n-1;const int32_t offsets[]{0,prefix};f.offsets.upload(offsets);
    auto in=Rows(f.Input(),0,prefix),out=Rows(f.Output(),0,prefix),cache=f.State();
    vt::CausalConv1dFwd(gpu.q,out,in,f.weight.tensor,&f.bias.tensor,cache,f.offsets.tensor,f.initial.tensor,{true});
    in=Rows(f.Input(),prefix,1);out=Rows(f.Output(),prefix,1);
    vt::CausalConv1dUpdate(gpu.q,out,in,f.weight.tensor,&f.bias.tensor,cache,{true});
    xpu_test::SameBytes(f.output.download(),expected);xpu_test::SameBytes(f.state.download(),state);
    auto reference=reference_state.tensor;reference.shape[0]=1;
    vt::CausalConv1dUpdate(gpu.q,expected_next.tensor,next.tensor,f.weight.tensor,&f.bias.tensor,reference,{true});
    out=Rows(f.output.tensor,n,1);
    vt::CausalConv1dUpdate(gpu.q,out,next.tensor,f.weight.tensor,&f.bias.tensor,cache,{true});
    const auto full=f.output.download();const auto next_bytes=expected_next.download();
    xpu_test::SameBytes({full.end()-next_bytes.size(),full.end()},next_bytes);
    xpu_test::SameBytes(f.state.download(),reference_state.download());
  }
}

TEST_CASE("XPU typed conv prefill: overlapping FP16 input FP32 output retains original raw history") {
  Queue gpu(vt::DeviceType::kXPU);Policy policy;
  Fixture f(gpu.q,512,2*Fixture::channels,true);
  const auto original=f.input.download();
  auto in=f.Input(),cache=f.State();
  auto out=vt::Tensor::Contiguous(f.input.tensor.data,DType::kF32,gpu.q.device,{512,Fixture::channels});
  policy.Set(0);f.Reset(true);
  vt::CausalConv1dFwd(gpu.q,out,in,f.weight.tensor,&f.bias.tensor,cache,f.offsets.tensor,f.initial.tensor,{true});
  const auto expected=f.input.download(),state=f.state.download();
  policy.Set(4);f.input.upload(original.data());f.Reset(true);
  (void)vt::xpu::DrainProfileEvents();
  vt::CausalConv1dFwd(gpu.q,out,in,f.weight.tensor,&f.bias.tensor,cache,f.offsets.tensor,f.initial.tensor,{true});
  xpu_test::SameBytes(f.input.download(),expected);xpu_test::SameBytes(f.state.download(),state);
  CheckRoute(4,true);
}

TEST_CASE("XPU typed conv prefill: invalid offsets and state aliases preserve outputs") {
  Queue gpu(vt::DeviceType::kXPU);Policy policy;policy.Set(4);
  Fixture f(gpu.q,512,16384,false);f.Reset(true);
  const auto before=f.output.download(),state=f.state.download(),input=f.input.download(),weight=f.weight.download();
  auto in=f.Input(),out=f.Output(),cache=f.State();
  const auto refuse=[&](vt::Tensor src,vt::Tensor dst,vt::Tensor working,const vt::Tensor* bias){
    CHECK_THROWS(vt::CausalConv1dFwd(gpu.q,dst,src,f.weight.tensor,bias,working,f.offsets.tensor,f.initial.tensor,{true}));
    xpu_test::SameBytes(f.output.download(),before);xpu_test::SameBytes(f.state.download(),state);
    xpu_test::SameBytes(f.input.download(),input);xpu_test::SameBytes(f.weight.download(),weight);
  };
  auto bad=cache;bad.data=in.data;refuse(in,out,bad,nullptr);
  bad=out;bad.data=cache.data;refuse(in,bad,cache,nullptr);
  bad=cache;bad.data=f.weight.tensor.data;refuse(in,out,bad,nullptr);
  bad=f.bias.tensor;bad.data=cache.data;refuse(in,out,cache,&bad);
  for(const auto& offsets:{std::array<int32_t,2>{1,512},{0,511},{0,513},{0,-1}}){
    f.offsets.upload(offsets.data());refuse(in,out,cache,nullptr);
  }
}

TEST_CASE("XPU typed conv prefill: unsupported dtypes physical history and batches keep fallback") {
  Queue gpu(vt::DeviceType::kXPU);Policy policy;
  constexpr int c=Fixture::channels,n=512;
  Fixture f(gpu.q,n,16384,true);
  Buffer input_f32(gpu.q,DType::kF32,{n,c}),weight_f32(gpu.q,DType::kF32,{c,4}),
    state_f16(gpu.q,DType::kF16,{2,c,3}),wide(gpu.q,DType::kF32,{2,c,6}),
    bias_f16(gpu.q,DType::kF16,{c}),batch_offsets(gpu.q,DType::kI32,{3}),batch_flags(gpu.q,DType::kI8,{2});
  input_f32.put(xpu_test::Values(n*c,1));weight_f32.put(xpu_test::Values(c*4,9,.125f));
  bias_f16.put(xpu_test::Values(c,7,.03125f));
  const int32_t cu[]{0,128,n};batch_offsets.upload(cu);const int8_t flags[]{1,0};batch_flags.upload(flags);
  for(int mode=0;mode<6;++mode){
    CAPTURE(mode);
    std::vector<unsigned char> expected,expected_state;
    for(int tile:{0,4}){
      policy.Set(tile);f.Reset(true);state_f16.put(f.history);wide.put(xpu_test::Values(2*c*6,5));
      auto in=mode==0?input_f32.tensor:f.Input(),w=mode==1?weight_f32.tensor:f.weight.tensor,
        cache=mode==2?state_f16.tensor:mode==3?wide.tensor:f.State(),out=f.Output();
      if(mode==2||mode==3)cache.shape[0]=1;
      if(mode==5)cache=f.state.tensor;
      const vt::Tensor* bias=mode==4?&bias_f16.tensor:nullptr;
      auto& qsl=mode==5?batch_offsets.tensor:f.offsets.tensor;auto& initial=mode==5?batch_flags.tensor:f.initial.tensor;
      (void)vt::xpu::DrainProfileEvents();
      vt::CausalConv1dFwd(gpu.q,out,in,w,bias,cache,qsl,initial,{true});
      const auto output=f.output.download();
      const auto history=mode==2?state_f16.download():mode==3?wide.download():f.state.download();
      if(!tile){expected=output;expected_state=history;}
      else{xpu_test::SameBytes(output,expected);xpu_test::SameBytes(history,expected_state);CheckRoute(4,false);}
    }
  }
}

TEST_CASE("XPU typed conv prefill: qualified default selects tile4 and explicit overrides remain available") {
  Policy policy;unsetenv(policy.name);REQUIRE(vt::XpuConvPrefillTypedTile()==4);
  Queue gpu(vt::DeviceType::kXPU);
  Fixture f(gpu.q,512,16384,true);
  policy.Set(0);f.Reset(true);f.Apply(true,true);
  const auto expected=f.output.download(),history=f.state.download();
  unsetenv(policy.name);f.Reset(true);(void)vt::xpu::DrainProfileEvents();
  f.Apply(true,true);
  xpu_test::SameBytes(f.output.download(),expected);
  xpu_test::SameBytes(f.state.download(),history);
  CheckRoute(4,true);
  for(int tile:{0,1,2,4}){policy.Set(tile);CHECK(vt::XpuConvPrefillTypedTile()==tile);}
  setenv(policy.name,"3",1);CHECK_THROWS(vt::XpuConvPrefillTypedTile());
}

TEST_CASE("XPU typed conv prefill: independent graphs see changed inputs flags and reject stale offsets") {
  Queue first(vt::DeviceType::kXPU),second(vt::DeviceType::kXPU);Policy policy;
  auto& backend=vt::GetBackend(first.q.device);REQUIRE(backend.SupportsGraphCapture());
  Fixture a(first.q,512,16384,true),b(second.q,512,16384,false),reference(first.q,512,16384,true);
  Fixture* fixtures[]{&a,&b};vt::BreakableGraph graphs[2];
  policy.Set(4);
  for(int i=0;i<2;++i){
    auto& f=*fixtures[i];f.Reset(true);f.Apply(false,true);backend.Synchronize(f.q);
    vt::GraphCaptureScope scope(backend,f.q,graphs[i],vt::GraphCaptureMode::kFull);f.Apply(false,true);
  }
  REQUIRE(graphs[0].captured());REQUIRE(graphs[1].captured());
  const auto memory=vt::xpu::GetMemoryInfo();
  for(int mode=0;mode<4;++mode){
    for(int i=0;i<2;++i){
      auto& f=*fixtures[i];f.Reset((mode+i)%2!=0);
      auto bytes=f.input.download();const uint16_t value=vt::F32ToF16(float(mode+i+1)*.125f);
      std::memcpy(bytes.data()+16*sizeof(uint16_t),&value,sizeof(value));f.input.upload(bytes.data());
    }
    graphs[1].Replay(second.q);graphs[0].Replay(first.q);
    for(int i=0;i<2;++i){
      auto& f=*fixtures[i];reference.input.upload(f.input.download().data());reference.Reset((mode+i)%2!=0);
      policy.Set(0);reference.Apply(false,true);
      xpu_test::SameBytes(f.output.download(),reference.output.download());
      xpu_test::SameBytes(f.state.download(),reference.state.download());
    }
  }
  a.Reset(true);const auto out=a.output.download(),state=a.state.download();
  const int32_t bad[]{0,511};a.offsets.upload(bad);
  CHECK_THROWS(graphs[0].Replay(first.q));
  xpu_test::SameBytes(a.output.download(),out);xpu_test::SameBytes(a.state.download(),state);
  const int32_t valid[]{0,512};a.offsets.upload(valid);
  reference.input.upload(a.input.download().data());reference.Reset(true);policy.Set(0);reference.Apply(false,true);
  graphs[0].Replay(first.q);
  xpu_test::SameBytes(a.output.download(),reference.output.download());
  xpu_test::SameBytes(a.state.download(),reference.state.download());
  backend.Synchronize(first.q);backend.Synchronize(second.q);
  const auto after=vt::xpu::GetMemoryInfo();
  CHECK(after.allocated_bytes==memory.allocated_bytes);CHECK(after.pinned_bytes==memory.pinned_bytes);
  CHECK(after.graph_count==memory.graph_count);CHECK(after.graph_device_bytes==memory.graph_device_bytes);
  for(auto& graph:graphs)graph.Reset();
}

TEST_CASE("XPU typed conv prefill: retains existing eager validation host operations") {
  Queue gpu(vt::DeviceType::kXPU);Policy policy;Fixture f(gpu.q,512,16384,true);
  std::map<std::string,int> control;
  for(int tile:{0,4}){
    policy.Set(tile);f.Reset(true);(void)vt::xpu::DrainHostProfileRecords();f.Apply(false,true);
    const auto records=vt::xpu::DrainHostProfileRecords();std::map<std::string,int> counts;
    for(const auto& record:records)++counts[record.stage];
    if(!tile)control=counts;else CHECK(counts==control);
    if(std::getenv("VT_XPU_HOST_PROFILE")&&std::string(std::getenv("VT_XPU_HOST_PROFILE"))=="1")
      CHECK(counts["metadata_d2h_wait"]==1);
    // Completion here belongs to the test, outside the observed submission.
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
  }
}
