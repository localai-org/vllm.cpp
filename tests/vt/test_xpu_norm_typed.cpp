#include "xpu_test_helpers.h"
#include "vt/xpu.h"
#include "vt/xpu_norm_policy.h"
#include "vt/breakable_graph.h"
#include <array>
#include <cstdlib>
#include <string>

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
struct NormPolicy {
  static constexpr const char* name="VT_XPU_GEMMA_LARGE_TYPED";
  bool had=std::getenv(name)!=nullptr;
  std::string old=had?std::getenv(name):"";
  ~NormPolicy(){if(had)setenv(name,old.c_str(),1);else unsetenv(name);}
  void Set(bool on){REQUIRE(setenv(name,on?"1":"0",1)==0);}
};
bool NormProfiling(){
  return std::getenv("VT_XPU_PROFILE") && std::string(std::getenv("VT_XPU_PROFILE"))=="1";
}
void NormRoute(bool expected){
  const auto events=vt::xpu::DrainProfileEvents();
  // Exact functional checks still run without diagnostic queue timestamps.
  // The separately registered admission proof is an explicit framework skip
  // without profiling; qualification runs it with VT_XPU_PROFILE=1.
  if(!NormProfiling())return;
  CHECK(std::count_if(events.begin(),events.end(),[](const auto& e){return e.stage=="rms_norm_gemma5120_typed_fp16";})==(expected?1:0));
}
struct NormFixture {
  vt::Queue& q;int rows;
  Buffer input,weight,output,residual;
  std::vector<unsigned char> original_input,original_residual,poison;
  NormFixture(vt::Queue& queue,int count):q(queue),rows(count),input(q,DType::kF16,{count+1,5120}),
    weight(q,DType::kF16,{5120}),output(q,DType::kF16,{count+1,5120}),residual(q,DType::kF16,{count+1,5120}){
    auto x=xpu_test::Values(size_t(count+1)*5120,7,.125f);
    auto r=xpu_test::Values(x.size(),11,.015625f);
    // Nonzero values on opposite sides of half rounding boundaries; the
    // normalization must consume the F32 sum, not the narrowed residual.
    constexpr uint16_t bits[]{0,0x8000,1,0x8001,0x0400,0x3c00,0xbc00,0x7bff,0xfbff,0x3555,0xb555};
    for(size_t i=0;i<x.size();i+=127)x[i]=vt::F16ToF32(bits[(i/127)%std::size(bits)]);
    input.put(x);residual.put(r);weight.put(xpu_test::Values(5120,9,.0625f));
    original_input=input.download();original_residual=residual.download();poison.assign(output.bytes,0xa5);
  }
  void Reset(){input.upload(original_input.data());residual.upload(original_residual.data());output.upload(poison.data());}
  vt::Tensor Active(vt::Tensor t){t.shape[0]=rows;return t;}
  void Apply(bool has_res,float eps=1e-6f,int alias=0){
    auto x=Active(input.tensor),out=Active(output.tensor),res=Active(residual.tensor);
    if(alias==1)out=x;
    if(alias==2)out=res;
    if(alias==3)res=x;
    vt::RmsNorm(q,out,x,weight.tensor,{eps,true},has_res?&res:nullptr);
  }
};
}
TEST_CASE("XPU typed large norm: exact boundaries residual precision and poisoned capacity"){
  Queue gpu(vt::DeviceType::kXPU);NormPolicy policy;
  for(int rows:{1,4,16,17,31,32,33,4096,4097})for(bool residual:{false,true}){
    CAPTURE(rows);
    CAPTURE(residual);NormFixture f(gpu.q,rows);
    policy.Set(false);f.Reset();f.Apply(residual);
    const auto output=f.output.download(),res=f.residual.download(),input=f.input.download();
    policy.Set(true);f.Reset();(void)vt::xpu::DrainProfileEvents();f.Apply(residual);
    xpu_test::SameBytes(f.output.download(),output);xpu_test::SameBytes(f.residual.download(),res);
    xpu_test::SameBytes(f.input.download(),input);NormRoute(rows>=17&&rows<=4096);
  }
}
TEST_CASE("XPU typed large norm: allowed aliases epsilon and exceptional operands remain exact"){
  Queue gpu(vt::DeviceType::kXPU);NormPolicy policy;
  for(int rows:{17,32,33})for(int alias:{0,1,2,3})for(float eps:{0.f,1e-6f,1e-3f}){
    CAPTURE(rows);
    CAPTURE(alias);
    CAPTURE(eps);NormFixture f(gpu.q,rows);
    for(bool exceptional:{false,true}){
      if(exceptional){auto bits=f.original_input;const uint16_t v[]{0x7c00,0xfc00,0x7e01,0xfe55};std::memcpy(bits.data(),v,sizeof(v));f.original_input=bits;}
      policy.Set(false);f.Reset();f.Apply(true,eps,alias);
      const auto output=f.output.download(),res=f.residual.download(),input=f.input.download();
      policy.Set(true);f.Reset();(void)vt::xpu::DrainProfileEvents();f.Apply(true,eps,alias);
      xpu_test::SameBytes(f.output.download(),output);xpu_test::SameBytes(f.residual.download(),res);
      xpu_test::SameBytes(f.input.download(),input);NormRoute(true);
    }
  }
}
TEST_CASE("XPU typed large norm: unsupported dtypes and mode retain fallback invalid geometry writes nothing"){
  Queue gpu(vt::DeviceType::kXPU);NormPolicy policy;NormFixture f(gpu.q,33);
  Buffer wide_x(gpu.q,DType::kF32,{33,5120}),wide_w(gpu.q,DType::kF32,{5120}),wide_r(gpu.q,DType::kF32,{33,5120});
  wide_x.put(xpu_test::Values(33*5120));wide_w.put(xpu_test::Values(5120,9));const auto initial=xpu_test::Values(33*5120,11);
  for(int mode:{0,1,2,3}){
    auto x=mode==0?wide_x.tensor:f.Active(f.input.tensor),w=mode==1?wide_w.tensor:f.weight.tensor;
    auto res=mode==2?wide_r.tensor:f.Active(f.residual.tensor),out=f.Active(f.output.tensor);
    std::vector<unsigned char> expected,history;
    for(bool on:{false,true}){
      policy.Set(on);f.Reset();wide_r.put(initial);(void)vt::xpu::DrainProfileEvents();
      vt::RmsNorm(gpu.q,out,x,w,{1e-6f,mode!=3},&res);
      const auto got=f.output.download(),r=mode==2?wide_r.download():f.residual.download();
      if(!on){expected=got;history=r;}else{xpu_test::SameBytes(got,expected);xpu_test::SameBytes(r,history);NormRoute(false);}
    }
  }
  policy.Set(true);f.Reset();auto x=f.Active(f.input.tensor),out=f.Active(f.output.tensor),res=f.Active(f.residual.tensor);
  for(int bad:{0,1,2}){
    auto bx=x,br=res,bw=f.weight.tensor;
    if(bad==0)bx.stride[0]+=1;
    if(bad==1)br.stride[0]+=1;
    if(bad==2)bw.data=res.data;
    CHECK_THROWS(vt::RmsNorm(gpu.q,out,bx,bw,{1e-6f,true},&br));
    xpu_test::SameBytes(f.output.download(),f.poison);xpu_test::SameBytes(f.residual.download(),f.original_residual);
  }
}
TEST_CASE("XPU typed large norm: unrounded residual normalization has an explicit negative oracle"){
  Queue gpu(vt::DeviceType::kXPU);NormPolicy policy;NormFixture f(gpu.q,33);
  f.input.put(std::vector<float>(34*5120,1.f));
  auto r=std::vector<float>(34*5120);for(size_t i=0;i<r.size();++i)r[i]=i%2?.0007f:.0003f;
  f.residual.put(r);f.weight.put(std::vector<float>(5120,0.f));
  f.original_input=f.input.download();f.original_residual=f.residual.download();
  policy.Set(false);f.Reset();f.Apply(true);const auto expected=f.output.download(),history=f.residual.download();
  Buffer wrong(gpu.q,DType::kF16,{34,5120});wrong.upload(f.poison.data());
  auto rounded=f.Active(f.residual.tensor),wrong_out=f.Active(wrong.tensor);
  vt::RmsNorm(gpu.q,wrong_out,rounded,f.weight.tensor,{1e-6f,true},nullptr);
  REQUIRE(wrong.download()!=expected);
  policy.Set(true);f.Reset();(void)vt::xpu::DrainProfileEvents();f.Apply(true);
  xpu_test::SameBytes(f.output.download(),expected);xpu_test::SameBytes(f.residual.download(),history);NormRoute(true);
}
TEST_CASE("XPU typed large norm: independent graphs retain changed inputs and exact residual continuation"){
  Queue first(vt::DeviceType::kXPU),second(vt::DeviceType::kXPU);NormPolicy policy;
  auto& backend=vt::GetBackend(first.q.device);REQUIRE(backend.SupportsGraphCapture());
  NormFixture a(first.q,31),b(second.q,33);NormFixture* fixtures[]{&a,&b};vt::BreakableGraph graphs[2];
  policy.Set(true);
  for(int i=0;i<2;++i){auto& f=*fixtures[i];f.Reset();f.Apply(true);backend.Synchronize(f.q);
    {vt::GraphCaptureScope scope(backend,f.q,graphs[i],vt::GraphCaptureMode::kFull);f.Apply(true);}
    REQUIRE(graphs[i].captured());
  }
  const auto before=vt::xpu::GetMemoryInfo();
  for(int iteration=0;iteration<3;++iteration)for(int i=0;i<2;++i){
    auto& f=*fixtures[i];auto x=xpu_test::Values(size_t(f.rows+1)*5120,iteration+17,.25f);
    auto r=xpu_test::Values(x.size(),iteration+7,.00390625f);
    f.input.put(x);f.residual.put(r);f.original_input=f.input.download();f.original_residual=f.residual.download();
    policy.Set(false);f.Reset();f.Apply(true);const auto expected=f.output.download(),history=f.residual.download();
    policy.Set(true);f.Reset();graphs[i].Replay(f.q);backend.Synchronize(f.q);
    xpu_test::SameBytes(f.output.download(),expected);xpu_test::SameBytes(f.residual.download(),history);
    // A second replay consumes the live updated residual, not a hidden snapshot.
    f.original_residual=history;policy.Set(false);f.Reset();f.Apply(true);
    const auto continued=f.output.download(),continued_history=f.residual.download();
    policy.Set(true);f.Reset();graphs[i].Replay(f.q);backend.Synchronize(f.q);
    xpu_test::SameBytes(f.output.download(),continued);xpu_test::SameBytes(f.residual.download(),continued_history);
  }
  const auto after=vt::xpu::GetMemoryInfo();CHECK(after.allocated_bytes==before.allocated_bytes);CHECK(after.pinned_bytes==before.pinned_bytes);
  for(auto& graph:graphs)graph.Reset();
}

TEST_CASE("XPU typed large norm: selected default and explicit control policy fail invalid values"){
  NormPolicy policy;unsetenv(policy.name);CHECK(vt::XpuGemmaLargeTyped());
  policy.Set(false);CHECK_FALSE(vt::XpuGemmaLargeTyped());policy.Set(true);CHECK(vt::XpuGemmaLargeTyped());
  setenv(policy.name,"2",1);CHECK_THROWS(vt::XpuGemmaLargeTyped());
  Queue gpu(vt::DeviceType::kXPU);NormFixture f(gpu.q,33);f.Reset();
  CHECK_THROWS(f.Apply(true));
  xpu_test::SameBytes(f.output.download(),f.poison);
  xpu_test::SameBytes(f.residual.download(),f.original_residual);
}

TEST_CASE("XPU typed large norm: instrumented admission proof" * doctest::skip(!NormProfiling())){
  Queue gpu(vt::DeviceType::kXPU);NormPolicy policy;
  for(int rows:{16,17,31,32,33,4096,4097}){
    CAPTURE(rows);
    NormFixture f(gpu.q,rows);
    policy.Set(true);f.Reset();(void)vt::xpu::DrainProfileEvents();f.Apply(true);
    (void)f.output.download();NormRoute(rows>=17&&rows<=4096);
  }
}
