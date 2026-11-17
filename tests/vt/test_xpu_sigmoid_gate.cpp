#include "xpu_test_helpers.h"
#include "vt/xpu_sigmoid_gate_policy.h"
#include "vt/xpu.h"
#include "vt/breakable_graph.h"
#include <bit>
#include <cstdlib>
#include <limits>
#include <string>
namespace {
using xpu_test::Buffer;using xpu_test::Queue;using vt::DType;
struct GateFlag {
 const char* key="VT_XPU_SIGMOID_GATE_TYPED";bool had=std::getenv(key);std::string old=had?std::getenv(key):"";
 ~GateFlag(){if(had)setenv(key,old.c_str(),1);else unsetenv(key);}
 void set(const char* x){setenv(key,x,1);}
};
void CompareGate(vt::Queue& q,vt::Tensor& out,const vt::Tensor& attn,const vt::Tensor& gate,Buffer& owner){
 GateFlag flag;flag.set("0");vt::SigmoidGateBf16(q,out,attn,gate);const auto expected=owner.download();
 flag.set("1");vt::SigmoidGateBf16(q,out,attn,gate);xpu_test::SameBytes(owner.download(),expected);
}
}
TEST_CASE("XPU typed sigmoid gate: exhaustive FP16-valued F32 gates preserve all output bits"){
 Queue gpu(vt::DeviceType::kXPU);constexpr int m=512,k=6144;constexpr size_t count=size_t(m)*k;
 Buffer attn(gpu.q,DType::kF16,{m,k}),gate(gpu.q,DType::kF32,{m,k}),output(gpu.q,DType::kF16,{int64_t(count+32)});
 auto out=vt::Tensor::Contiguous(static_cast<uint16_t*>(output.tensor.data)+16,DType::kF16,gpu.q.device,{m,k});
 std::vector<float> gv(count);for(size_t i=0;i<count;++i)gv[i]=vt::F16ToF32(uint16_t(i));gate.upload(gv.data());
 std::vector<uint16_t> av(count);
 for(uint16_t value:{uint16_t{0},uint16_t{0x8000},uint16_t{1},uint16_t{0x3c00},uint16_t{0xbc00},uint16_t{0x7bff},uint16_t{0x7c00},uint16_t{0xfc00},uint16_t{0x7e35}}){
  CAPTURE(value);
  std::fill(av.begin(),av.end(),value);attn.upload(av.data());output.upload(std::vector<unsigned char>(output.bytes,0xa5).data());
  CompareGate(gpu.q,out,attn.tensor,gate.tensor,output);
  const auto got=output.download();CHECK(std::all_of(got.begin(),got.begin()+32,[](auto b){return b==0xa5;}));CHECK(std::all_of(got.end()-32,got.end(),[](auto b){return b==0xa5;}));
  xpu_test::SameBytes(attn.download(),std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(av.data()),reinterpret_cast<const unsigned char*>(av.data())+av.size()*2));
 }
}
TEST_CASE("XPU typed sigmoid gate: F32 gate rounding strides aliases and small fallback"){
 Queue gpu(vt::DeviceType::kXPU);constexpr int k=6144;
 for(int rows:{1,4,511,512,4095,4096})for(int padding:{0,16}){
  CAPTURE(rows);
  CAPTURE(padding);
  Buffer attn(gpu.q,DType::kF16,{rows,k}),gate(gpu.q,DType::kF32,{rows,k+padding}),output(gpu.q,DType::kF16,{rows,k});
  auto view=gate.tensor;view.shape[1]=k;
  std::vector<uint16_t> av(size_t(rows)*k);for(size_t i=0;i<av.size();++i)av[i]=uint16_t((i*37)%65536);attn.upload(av.data());
  std::vector<float> gv(size_t(rows)*(k+padding));for(size_t i=0;i<gv.size();++i)gv[i]=std::bit_cast<float>(uint32_t(i*1664525u+1013904223u));gate.upload(gv.data());
  const auto before=gate.download();
  output.upload(std::vector<unsigned char>(output.bytes,0xa5).data());
  if(padding){
    const auto old=output.download();GateFlag rejected;
    for(const char* mode:{"0","1"}){rejected.set(mode);CHECK_THROWS(vt::SigmoidGateBf16(gpu.q,output.tensor,attn.tensor,view));xpu_test::SameBytes(output.download(),old);}
    xpu_test::SameBytes(gate.download(),before);continue;
  }
  CompareGate(gpu.q,output.tensor,attn.tensor,view,output);xpu_test::SameBytes(gate.download(),before);
  // The public contract is equal element COUNT, not identical matrix shapes.
  auto flat_gate=vt::Tensor::Contiguous(gate.tensor.data,DType::kF32,gpu.q.device,{1,int64_t(rows)*k});
  auto flat_attn=vt::Tensor::Contiguous(attn.tensor.data,DType::kF16,gpu.q.device,{1,int64_t(rows)*k});
  CompareGate(gpu.q,output.tensor,flat_attn,flat_gate,output);
  // Same public in-place behavior; both use existing WithOutput alias scratch.
  GateFlag flag;flag.set("0");vt::SigmoidGateBf16(gpu.q,attn.tensor,attn.tensor,view);const auto expected=attn.download();attn.upload(av.data());flag.set("1");vt::SigmoidGateBf16(gpu.q,attn.tensor,attn.tensor,view);xpu_test::SameBytes(attn.download(),expected);
 }
}
TEST_CASE("XPU typed sigmoid gate: admission policy and rejection before output mutation"){
 GateFlag flag;unsetenv(flag.key);CHECK_FALSE(vt::XpuSigmoidGateTyped(4096,6144));flag.set("1");CHECK(vt::XpuSigmoidGateTyped(512,6144));CHECK(vt::XpuSigmoidGateTyped(4096,6144));
 for(int m:{0,1,4,511,4097})CHECK_FALSE(vt::XpuSigmoidGateTyped(m,6144));CHECK_FALSE(vt::XpuSigmoidGateTyped(4096,6143));
 Queue gpu(vt::DeviceType::kXPU);Buffer attn(gpu.q,DType::kF16,{512,6144}),gate(gpu.q,DType::kF32,{512,6144}),out(gpu.q,DType::kF16,{512,6144});attn.put(std::vector<float>(512*6144,1.f));gate.put(std::vector<float>(512*6144,.5f));out.upload(std::vector<unsigned char>(out.bytes,0xa5).data());const auto before=out.download();
 for(const char* mode:{"0","1"}){
  flag.set(mode);auto wrong=gate.tensor;wrong.dtype=DType::kF16;
  CHECK_THROWS(vt::SigmoidGateBf16(gpu.q,out.tensor,attn.tensor,wrong));xpu_test::SameBytes(out.download(),before);
 }
 for(const char* bad:{"","bad","01"," 1","2"}){flag.set(bad);CHECK_THROWS(vt::XpuSigmoidGateTyped(4096,6144));CHECK_FALSE(vt::XpuSigmoidGateTyped(4,6144));CHECK_THROWS(vt::SigmoidGateBf16(gpu.q,out.tensor,attn.tensor,gate.tensor));xpu_test::SameBytes(out.download(),before);}
}
TEST_CASE("XPU typed sigmoid gate: independent graph queues own changing F32 inputs"){
 GateFlag flag;flag.set("1");Queue a(vt::DeviceType::kXPU),b(vt::DeviceType::kXPU);constexpr int m=512,k=6144;
 Buffer ia(a.q,DType::kF16,{m,k}),ib(b.q,DType::kF16,{m,k}),ga(a.q,DType::kF32,{m,k}),gb(b.q,DType::kF32,{m,k}),oa(a.q,DType::kF16,{m,k}),ob(b.q,DType::kF16,{m,k}),reference(a.q,DType::kF16,{m,k});
 auto av=ga.tensor,bv=gb.tensor;av.shape[1]=bv.shape[1]=k;auto& backend=vt::GetBackend(a.q.device);REQUIRE(backend.SupportsGraphCapture());
 ia.put(std::vector<float>(m*k,.5f));ib.put(std::vector<float>(m*k,.5f));ga.put(std::vector<float>(m*k,.25f));gb.put(std::vector<float>(m*k,.25f));
 const auto apply=[&](int i){vt::SigmoidGateBf16(i?b.q:a.q,i?ob.tensor:oa.tensor,i?ib.tensor:ia.tensor,i?bv:av);};apply(0);apply(1);backend.Synchronize(a.q);backend.Synchronize(b.q);
 vt::BreakableGraph graphs[2];for(int i=0;i<2;++i){vt::GraphCaptureScope capture(backend,i?b.q:a.q,graphs[i],vt::GraphCaptureMode::kFull);apply(i);}REQUIRE(graphs[0].captured());REQUIRE(graphs[1].captured());const auto memory=vt::xpu::GetMemoryInfo();
 for(float value:{-.5f,.0001f,12.f}){
  ga.put(std::vector<float>(m*k,value));gb.put(std::vector<float>(m*k,value));graphs[1].Replay(b.q);graphs[0].Replay(a.q);flag.set("0");vt::SigmoidGateBf16(a.q,reference.tensor,ia.tensor,av);flag.set("1");const auto expected=reference.download();xpu_test::SameBytes(oa.download(),expected);xpu_test::SameBytes(ob.download(),expected);
 }
 const auto after=vt::xpu::GetMemoryInfo();CHECK(after.allocated_bytes==memory.allocated_bytes);CHECK(after.graph_count==memory.graph_count);CHECK(after.graph_device_bytes==memory.graph_device_bytes);for(auto& g:graphs)g.Reset();
}

TEST_CASE("XPU typed sigmoid gate: rounding the F32 gate is a detecting negative oracle"){
 GateFlag flag;Queue gpu(vt::DeviceType::kXPU);constexpr int rows=512,width=6144;
 Buffer attn(gpu.q,DType::kF16,{rows,width}),gate(gpu.q,DType::kF32,{rows,width}),out(gpu.q,DType::kF16,{rows,width});
 attn.put(std::vector<float>(rows*width,1.f));
 const float original=1.0006f,rounded=vt::F16ToF32(vt::F32ToF16(original));REQUIRE(original!=rounded);
 gate.put(std::vector<float>(rows*width,original));flag.set("0");vt::SigmoidGateBf16(gpu.q,out.tensor,attn.tensor,gate.tensor);const auto expected=out.download();
 gate.put(std::vector<float>(rows*width,rounded));vt::SigmoidGateBf16(gpu.q,out.tensor,attn.tensor,gate.tensor);CHECK(out.download()!=expected);
 gate.put(std::vector<float>(rows*width,original));flag.set("1");vt::SigmoidGateBf16(gpu.q,out.tensor,attn.tensor,gate.tensor);xpu_test::SameBytes(out.download(),expected);
}
