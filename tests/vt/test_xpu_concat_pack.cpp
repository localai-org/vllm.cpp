#include "xpu_test_helpers.h"
#include "vt/xpu_copy.h"
#include "vt/xpu.h"
#include "vt/breakable_graph.h"
#include "vllm/model_executor/models/mtp_concat.h"
#include <limits>
#include <string>
namespace {
using xpu_test::Queue;using xpu_test::Buffer;using vt::DType;
std::vector<unsigned char> Words(size_t bytes,int salt){std::vector<unsigned char>x(bytes);for(size_t i=0;i<bytes/2;++i){uint16_t value=uint16_t(i*37+salt);std::memcpy(x.data()+2*i,&value,2);}return x;}
std::vector<unsigned char> Reference(const std::vector<unsigned char>&a,const std::vector<unsigned char>&b,int rows,int hidden){size_t width=size_t(hidden)*2;std::vector<unsigned char>x(size_t(rows+1)*2*width,0xa5);for(int r=0;r<rows;++r){std::memcpy(x.data()+size_t(2*r)*width,a.data()+r*width,width);std::memcpy(x.data()+size_t(2*r+1)*width,b.data()+r*width,width);}return x;}
struct Flag{const char* key;bool had;std::string value;explicit Flag(const char*k):key(k),had(std::getenv(k)!=nullptr),value(had?std::getenv(k):""){}~Flag(){if(had)setenv(key,value.c_str(),1);else unsetenv(key);}};
}
TEST_CASE("XPU concat pack: all raw FP16 words tails zero rows and narrow model admission"){
 Queue queue(vt::DeviceType::kXPU);auto&q=queue.q;
 for(int rows:{0,1,4,511,512,513,4095,4096,4097})for(int hidden:{7,5120}){
  CAPTURE(rows);
  CAPTURE(hidden);
  Buffer a(q,DType::kF16,{std::max(rows,1),hidden}),b(q,DType::kF16,{std::max(rows,1),hidden}),out(q,DType::kF16,{rows+1,2*hidden});
  auto av=a.tensor,bv=b.tensor,ov=out.tensor;av.shape[0]=bv.shape[0]=ov.shape[0]=rows;
  const auto aa=Words(a.bytes,17),bb=Words(b.bytes,65513),poison=std::vector<unsigned char>(out.bytes,0xa5);a.upload(aa.data());b.upload(bb.data());out.upload(poison.data());
  vt::xpu::ConcatRowsF16(q,ov,av,bv);const auto expected=Reference(aa,bb,rows,hidden);xpu_test::SameBytes(out.download(),expected);xpu_test::SameBytes(a.download(),aa);xpu_test::SameBytes(b.download(),bb);
  out.upload(poison.data());(void)vt::xpu::DrainProfileEvents();
  const bool route=vllm::detail::ConcatMtpNormalized(vt::GetBackend(q.device),q,ov,av,bv,false,true);
  const bool admitted=rows>=512&&rows<=4096&&hidden==5120;CHECK(route==admitted);xpu_test::SameBytes(out.download(),expected);
  const auto events=vt::xpu::DrainProfileEvents();if(std::getenv("VT_XPU_PROFILE")&&std::string(std::getenv("VT_XPU_PROFILE"))=="1")CHECK(std::count_if(events.begin(),events.end(),[](const auto&e){return e.stage=="concat_rows_f16_pack";})==(admitted?1:0));
 }
 Flag flag("VLLM_QWEN35_MTP_CONCAT_PACK");unsetenv(flag.key);CHECK_FALSE(vllm::detail::MtpConcatPackEnabled());setenv(flag.key,"1",1);CHECK(vllm::detail::MtpConcatPackEnabled());setenv(flag.key,"0",1);CHECK_FALSE(vllm::detail::MtpConcatPackEnabled());setenv(flag.key,"2",1);CHECK_THROWS(vllm::detail::MtpConcatPackEnabled());
}
TEST_CASE("XPU concat pack: every descriptor dtype alignment and alias refusal precedes writes"){
 Queue queue(vt::DeviceType::kXPU);auto&q=queue.q;Buffer a(q,DType::kF16,{4,7}),b(q,DType::kF16,{4,7}),out(q,DType::kF16,{4,14});const auto poison=Words(out.bytes,31);out.upload(poison.data());
 const auto refuse=[&](vt::Tensor o,vt::Tensor x,vt::Tensor y){CHECK_THROWS(vt::xpu::ConcatRowsF16(q,o,x,y));xpu_test::SameBytes(out.download(),poison);};
 auto bad=a.tensor;bad.rank=1;refuse(out.tensor,bad,b.tensor);bad=a.tensor;bad.stride[0]=9;refuse(out.tensor,bad,b.tensor);bad=a.tensor;bad.stride[0]=-7;refuse(out.tensor,bad,b.tensor);bad=out.tensor;bad.stride[0]=1;refuse(bad,a.tensor,b.tensor);
 bad=a.tensor;bad.data=out.tensor.data;refuse(out.tensor,bad,b.tensor);bad=b.tensor;bad.data=static_cast<unsigned char*>(out.tensor.data)+2;refuse(out.tensor,a.tensor,bad);
 bad=a.tensor;bad.data=static_cast<unsigned char*>(a.tensor.data)+1;refuse(out.tensor,bad,b.tensor);bad=out.tensor;bad.data=static_cast<unsigned char*>(out.tensor.data)+1;refuse(bad,a.tensor,b.tensor);
 bad=a.tensor;bad.device.index=1;refuse(out.tensor,bad,b.tensor);bad=b.tensor;bad.shape[0]=3;refuse(out.tensor,a.tensor,bad);bad=a.tensor;bad.shape[1]=std::numeric_limits<int64_t>::max();refuse(out.tensor,bad,b.tensor);
 auto av=a.tensor,bv=b.tensor,ov=out.tensor;av.shape[0]=bv.shape[0]=ov.shape[0]=std::numeric_limits<int64_t>::max();refuse(ov,av,bv);
 av=a.tensor;bv=b.tensor;ov=out.tensor;av.dtype=bv.dtype=ov.dtype=DType::kBF16;refuse(ov,av,bv);bad=a.tensor;bad.data=nullptr;refuse(out.tensor,bad,b.tensor);
 bad=a.tensor;bad.data=reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-4);refuse(out.tensor,bad,b.tensor);
}
TEST_CASE("XPU concat pack: independent graph owners keep queued producer consumer ordering"){
 Queue first(vt::DeviceType::kXPU),second(vt::DeviceType::kXPU);auto&backend=vt::GetBackend(first.q.device);REQUIRE(backend.SupportsGraphCapture());constexpr int rows=512,hidden=5120;
 struct Fixture{vt::Queue&q;Buffer sa,sb,a,b,out,consumer;explicit Fixture(vt::Queue&queue):q(queue),sa(q,DType::kF16,{rows,hidden}),sb(q,DType::kF16,{rows,hidden}),a(q,DType::kF16,{rows,hidden}),b(q,DType::kF16,{rows,hidden}),out(q,DType::kF16,{rows,2*hidden}),consumer(q,DType::kF16,{rows,2*hidden}){}void Run(){auto&x=vt::GetBackend(q.device);x.Copy(q,a.tensor.data,sa.tensor.data,a.bytes);x.Copy(q,b.tensor.data,sb.tensor.data,b.bytes);vt::xpu::ConcatRowsF16(q,out.tensor,a.tensor,b.tensor);x.Copy(q,consumer.tensor.data,out.tensor.data,out.bytes);}}a(first.q),b(second.q);
 Fixture*fs[]{&a,&b};vt::BreakableGraph graphs[2];
 for(int i=0;i<2;++i){auto&f=*fs[i];auto bits=Words(f.a.bytes,i);f.sa.upload(bits.data());f.sb.upload(bits.data());f.Run();backend.Synchronize(f.q);vt::GraphCaptureScope capture(backend,f.q,graphs[i],vt::GraphCaptureMode::kFull);f.Run();}
 REQUIRE(graphs[0].captured());REQUIRE(graphs[1].captured());const auto warm=vt::xpu::GetMemoryInfo();
 for(int repeat=0;repeat<3;++repeat){std::vector<unsigned char>expected[2];for(int i=0;i<2;++i){auto&f=*fs[i];auto aa=Words(f.a.bytes,repeat+i),bb=Words(f.b.bytes,65500-repeat-i);f.sa.upload(aa.data());f.sb.upload(bb.data());expected[i]=Reference(aa,bb,rows,hidden);expected[i].resize(f.out.bytes);}graphs[1].Replay(second.q);graphs[0].Replay(first.q);for(int i=0;i<2;++i)xpu_test::SameBytes(fs[i]->consumer.download(),expected[i]);}
 const auto end=vt::xpu::GetMemoryInfo();CHECK(end.allocated_bytes==warm.allocated_bytes);CHECK(end.pinned_bytes==warm.pinned_bytes);CHECK(end.graph_count==warm.graph_count);backend.Synchronize(first.q);backend.Synchronize(second.q);
}
