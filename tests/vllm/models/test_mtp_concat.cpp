#include "../../vt/xpu_test_helpers.h"
#include "vllm/model_executor/models/mtp_concat.h"
#include "vt/breakable_graph.h"
#ifdef VLLM_CPP_XPU
#include "vt/xpu.h"
#endif
#include <array>
#include <limits>
#include <string>
using xpu_test::Buffer;
using xpu_test::Queue;
bool ConcatMtpNormalized(vt::Queue& q,vt::Tensor& out,const vt::Tensor& left,const vt::Tensor& right,bool strided) {
  return vllm::detail::ConcatMtpNormalized(vt::GetBackend(q.device),q,out,left,right,strided);
}
namespace {
std::vector<unsigned char> Pattern(size_t bytes, int salt) {
  std::vector<unsigned char> out(bytes);
  for(size_t i=0;i<bytes/2;++i){const uint16_t word=uint16_t(i*37+salt);std::memcpy(out.data()+2*i,&word,2);}
  return out;
}
std::vector<unsigned char> Expected(const std::vector<unsigned char>& a,
                                   const std::vector<unsigned char>& b,int rows,int h,size_t element) {
  const size_t width=size_t(h)*element;
  std::vector<unsigned char> out(size_t(rows+1)*2*width,0xa5);
  for(int r=0;r<rows;++r){std::memcpy(out.data()+2*r*width,a.data()+r*width,width);std::memcpy(out.data()+(2*r+1)*width,b.data()+r*width,width);}
  return out;
}
void Exact(vt::DeviceType type,int rows,int h,vt::DType dtype) {
  Queue queue(type);auto& q=queue.q;
  Buffer left(q,dtype,{std::max(rows,1),h}),right(q,dtype,{std::max(rows,1),h}),output(q,dtype,{rows+1,2*h});
  const auto a=Pattern(left.bytes,0),b=Pattern(right.bytes,65513);
  if(rows){left.upload(a.data());right.upload(b.data());}
  const auto poison=std::vector<unsigned char>(output.bytes,0xa5);output.upload(poison.data());
  auto out=output.tensor;out.shape[0]=rows;
  auto left_view=left.tensor,right_view=right.tensor;
  left_view.shape[0]=right_view.shape[0]=rows;
  const bool route=ConcatMtpNormalized(q,out,left_view,right_view,true);
  CHECK(route==(type==vt::DeviceType::kXPU && dtype==vt::DType::kF16 && h==5120 && rows>=512 && rows<=4096));
  const auto actual=output.download();xpu_test::SameBytes(actual,Expected(a,b,rows,h,vt::SizeOf(dtype)));
  if(rows){xpu_test::SameBytes(left.download(),a);xpu_test::SameBytes(right.download(),b);}
  output.upload(poison.data());CHECK_FALSE(ConcatMtpNormalized(q,out,left_view,right_view,false));
  xpu_test::SameBytes(output.download(),actual);
}
}
TEST_CASE("MTP concat: portable storage exactness empty small rows and old dtypes") {
  for(auto dtype:{vt::DType::kF16,vt::DType::kBF16,vt::DType::kF32})
    for(int n:{0,1,4,512}){
      CAPTURE(n);
      CAPTURE(int(dtype));
      Exact(vt::DeviceType::kCPU,n,7,dtype);
    }
}
TEST_CASE("MTP concat: descriptor overflow stride and alias refusal before writes") {
  Queue queue(vt::DeviceType::kCPU);auto& q=queue.q;
  Buffer left(q,vt::DType::kF16,{4,7}),right(q,vt::DType::kF16,{4,7}),output(q,vt::DType::kF16,{4,14});
  const auto poison=Pattern(output.bytes,17);output.upload(poison.data());
  auto check=[&](vt::Tensor o,vt::Tensor a,vt::Tensor b){CHECK_THROWS(ConcatMtpNormalized(q,o,a,b,true));xpu_test::SameBytes(output.download(),poison);};
  auto bad=left.tensor;bad.data=output.tensor.data;check(output.tensor,bad,right.tensor);
  bad=right.tensor;bad.data=output.tensor.data;check(output.tensor,left.tensor,bad);
  bad=left.tensor;bad.data=static_cast<unsigned char*>(output.tensor.data)+2;check(output.tensor,bad,right.tensor);
  bad=output.tensor;bad.data=static_cast<unsigned char*>(left.tensor.data)+2;check(bad,left.tensor,right.tensor);
  bad=left.tensor;bad.stride[0]=-7;check(output.tensor,bad,right.tensor);
  bad=left.tensor;bad.stride[0]=9;check(output.tensor,bad,right.tensor);
  bad=output.tensor;bad.stride[0]=1;check(bad,left.tensor,right.tensor);
  bad=right.tensor;bad.dtype=vt::DType::kBF16;check(output.tensor,left.tensor,bad);
  bad=left.tensor;bad.rank=1;check(output.tensor,bad,right.tensor);
  bad=left.tensor;bad.device.index=1;check(output.tensor,bad,right.tensor);
  bad=right.tensor;bad.shape[0]=3;check(output.tensor,left.tensor,bad);
  bad=left.tensor;bad.shape[1]=std::numeric_limits<int64_t>::max();check(output.tensor,bad,right.tensor);
  auto a=left.tensor,b=right.tensor,o=output.tensor;
  a.shape[0]=b.shape[0]=o.shape[0]=std::numeric_limits<int64_t>::max();check(o,a,b);
  bad=left.tensor;bad.data=reinterpret_cast<void*>(std::numeric_limits<uintptr_t>::max()-4);check(output.tensor,bad,right.tensor);
  bad=left.tensor;bad.data=nullptr;check(output.tensor,bad,right.tensor);
}
TEST_CASE("MTP concat: selected policy rejects malformed flags and observer is scoped") {
  const char* key="VLLM_QWEN35_MTP_CONCAT_STRIDED";const char* previous=std::getenv(key);const bool had=previous;const std::string old=had?previous:"";
  struct Restore{const char* key;bool had;std::string old;~Restore(){if(had)setenv(key,old.c_str(),1);else unsetenv(key);}}restore{key,had,old};
  unsetenv(key);CHECK_FALSE(vllm::detail::MtpConcatStridedEnabled());
  setenv(key,"0",1);CHECK_FALSE(vllm::detail::MtpConcatStridedEnabled());setenv(key,"1",1);CHECK(vllm::detail::MtpConcatStridedEnabled());
  setenv(key,"2",1);CHECK_THROWS(vllm::detail::MtpConcatStridedEnabled());
  CHECK(vllm::detail::mtp_head_tensor_observer==nullptr);
  {vllm::detail::MtpHeadTensorObserverScope observer(+[](vt::Queue&,const char*,const vt::Tensor&){});CHECK(vllm::detail::mtp_head_tensor_observer!=nullptr);}
  CHECK(vllm::detail::mtp_head_tensor_observer==nullptr);
}
#ifdef VLLM_CPP_XPU
TEST_CASE("MTP concat: exact XPU long boundaries and exceptional FP16 payloads") {
  for(int n:{0,1,4,511,512,513,4095,4096,4097}){CAPTURE(n);Exact(vt::DeviceType::kXPU,n,5120,vt::DType::kF16);}
  Exact(vt::DeviceType::kXPU,512,5120,vt::DType::kBF16);
  Exact(vt::DeviceType::kXPU,512,5120,vt::DType::kF32);
  Exact(vt::DeviceType::kXPU,512,5121,vt::DType::kF16);
}
TEST_CASE("MTP concat: independent graph owners queued producers and consumers") {
  Queue first(vt::DeviceType::kXPU),second(vt::DeviceType::kXPU);
  auto& backend=vt::GetBackend(first.q.device);REQUIRE(backend.SupportsGraphCapture());
  constexpr int n=512,h=5120;
  struct Fixture {
    vt::Queue& q;Buffer seed_left,seed_right,left,right,out,consumer;
    explicit Fixture(vt::Queue& queue):q(queue),seed_left(q,vt::DType::kF16,{n,h}),seed_right(q,vt::DType::kF16,{n,h}),left(q,vt::DType::kF16,{n,h}),right(q,vt::DType::kF16,{n,h}),out(q,vt::DType::kF16,{n,2*h}),consumer(q,vt::DType::kF16,{n,2*h}){}
    void Apply(){auto& b=vt::GetBackend(q.device);b.Copy(q,left.tensor.data,seed_left.tensor.data,left.bytes);b.Copy(q,right.tensor.data,seed_right.tensor.data,right.bytes);REQUIRE(ConcatMtpNormalized(q,out.tensor,left.tensor,right.tensor,true));b.Copy(q,consumer.tensor.data,out.tensor.data,out.bytes);}
  }a(first.q),b(second.q);
  Fixture* fs[]{&a,&b};vt::BreakableGraph graphs[2];
  for(int i=0;i<2;++i){auto& f=*fs[i];const auto seed=Pattern(f.left.bytes,i);f.seed_left.upload(seed.data());f.seed_right.upload(seed.data());f.Apply();backend.Synchronize(f.q);vt::GraphCaptureScope scope(backend,f.q,graphs[i],vt::GraphCaptureMode::kFull);f.Apply();}
  REQUIRE(graphs[0].captured());REQUIRE(graphs[1].captured());const auto warm=vt::xpu::GetMemoryInfo();
  for(int repeat=0;repeat<3;++repeat){
    std::array<std::vector<unsigned char>,2> expected;
    for(int i=0;i<2;++i){auto& f=*fs[i];const auto x=Pattern(f.left.bytes,repeat*11+i),y=Pattern(f.right.bytes,65500-repeat-i);f.seed_left.upload(x.data());f.seed_right.upload(y.data());expected[i]=Expected(x,y,n,h,2);expected[i].resize(f.out.bytes);}
    graphs[1].Replay(second.q);graphs[0].Replay(first.q);
    for(int i=0;i<2;++i)xpu_test::SameBytes(fs[i]->consumer.download(),expected[i]);
  }
  const auto end=vt::xpu::GetMemoryInfo();CHECK(end.allocated_bytes==warm.allocated_bytes);CHECK(end.pinned_bytes==warm.pinned_bytes);CHECK(end.graph_count==warm.graph_count);
  backend.Synchronize(first.q);backend.Synchronize(second.q);
}
#endif
