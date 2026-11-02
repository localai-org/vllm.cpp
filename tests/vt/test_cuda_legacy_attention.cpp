// Pinned llama.cpp 62318cd57 ggml-cuda/fattn-vec.cuh:86-118,274,439-496:
// warp-reduced GQA attention, causal/window masks, register online softmax.
// Enter the registered operator; no kernel or implementation header is called.
#include <cuda_runtime.h>
#include "vt/ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>

namespace {
void Check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
  void* p = nullptr;
  explicit Buffer(size_t n) { Check(cudaMallocManaged(&p, n)); }
  ~Buffer() { cudaFree(p); }
};
void Put(void* p, vt::DType dtype, size_t i, float f) {
  if (dtype == vt::DType::kF32) static_cast<float*>(p)[i] = f;
  else static_cast<uint16_t*>(p)[i] = vt::F32ToBF16(f);
}
float Get(void* p, vt::DType dtype, size_t i) {
  return dtype == vt::DType::kF32 ? static_cast<float*>(p)[i] : vt::BF16ToF32(static_cast<uint16_t*>(p)[i]);
}
void Run(int d, int seq, vt::DType input, vt::DType output, int group, bool window, bool causal, bool padded = false) {
  constexpr int hq = 32, bs = 16, blocks = 17;
  const int hk = hq / group;
  const int row = hk*d + (padded ? 1 : 0);
  Buffer query(hq*d*vt::SizeOf(input)), key(blocks*2*bs*row*vt::SizeOf(input));
  Buffer value(blocks*2*bs*row*vt::SizeOf(input)), out(hq*d*vt::SizeOf(output));
  Buffer table(blocks*2*sizeof(int32_t)), lens(sizeof(int32_t)), starts(2*sizeof(int32_t));
  for (int i=0;i<hq*d;++i) Put(query.p,input,i, std::sin(static_cast<float>(i)*0.13f)*0.7f);
  for (int i=0;i<blocks*2*bs*row;++i) {
    Put(key.p,input,i,std::sin(static_cast<float>(i)*0.07f));
    Put(value.p,input,i,std::cos(static_cast<float>(i)*0.11f));
  }
  for (int i=0;i<blocks;++i) static_cast<int32_t*>(table.p)[2*i] = blocks-i-1;
  static_cast<int32_t*>(lens.p)[0] = seq;
  static_cast<int32_t*>(starts.p)[0] = 0; static_cast<int32_t*>(starts.p)[1] = 1;
  const vt::Device gpu{vt::DeviceType::kCUDA,0}; vt::Queue q{gpu,nullptr};
  auto tq=vt::Tensor::Contiguous(query.p,input,gpu,{1,hq,d});
  auto tk=vt::Tensor::Contiguous(key.p,input,gpu,{blocks,bs,hk,d});
  auto tv=vt::Tensor::Contiguous(value.p,input,gpu,{blocks,bs,hk,d});
  tk.stride[0] = 2*bs*row; tv.stride[0] = 2*bs*row;
  tk.stride[1] = row; tv.stride[1] = row;
  auto to=vt::Tensor::Contiguous(out.p,output,gpu,{1,hq,d});
  auto tb=vt::Tensor::Contiguous(table.p,vt::DType::kI32,gpu,{1,blocks});
  tb.stride[0] *= 2; tb.stride[1] = 2;
  auto tl=vt::Tensor::Contiguous(lens.p,vt::DType::kI32,gpu,{1});
  auto ts=vt::Tensor::Contiguous(starts.p,vt::DType::kI32,gpu,{2});
  vt::PagedAttentionArgs args; args.scale=1.0f/std::sqrt(static_cast<float>(d)); args.causal=causal;
  if(window) { args.window_size=vt::AttentionWindow{7,0}; args.logits_soft_cap=0.7f; }
  auto launch=[&] { vt::PagedAttention(q,to,tq,tk,tv,tb,tl,ts,args); };
  const auto hits=vt::GetReferenceTierHits();
  launch(); Check(cudaDeviceSynchronize());
  if(vt::GetReferenceTierHits()!=hits) throw std::runtime_error("CPU fallback");
  // Independent two-pass host reference over the exact rounded operand bytes.
  for(int h=0;h<hq;++h) {
    const int lo=window?std::max(0,seq-8):0;
    std::vector<double> scores(static_cast<size_t>(seq)); double max=-1e300,sum=0;
    for(int j=lo;j<seq;++j) {
      const int base=((blocks-1-j/bs)*2*bs+j%bs)*row+(h/group)*d;
      double dot=0;
      for(int e=0;e<d;++e) dot+=static_cast<double>(Get(query.p,input,h*d+e))*Get(key.p,input,base+e);
      double score=dot*args.scale;
      if(args.logits_soft_cap>0) score=args.logits_soft_cap*std::tanh(score/args.logits_soft_cap);
      scores[static_cast<size_t>(j)]=score; max=std::max(max,score);
    }
    for(int j=lo;j<seq;++j) { scores[static_cast<size_t>(j)]=std::exp(scores[static_cast<size_t>(j)]-max); sum+=scores[static_cast<size_t>(j)]; }
    for(int e=0;e<d;++e) {
      double expected=0;
      for(int j=lo;j<seq;++j) {
        const int base=((blocks-1-j/bs)*2*bs+j%bs)*row+(h/group)*d;
        expected+=scores[static_cast<size_t>(j)]*Get(value.p,input,base+e)/sum;
      }
      const float actual=Get(out.p,output,h*d+e);
      const float tolerance=output==vt::DType::kBF16?0.003f:0.00001f;
      if(!std::isfinite(actual)||std::abs(actual-expected)>tolerance) throw std::runtime_error("paged attention numerical mismatch");
    }
  }
  // Warm after CPU managed-memory reads, then time only registered GPU work.
  for(int i=0;i<8;++i) { launch(); }
  Check(cudaDeviceSynchronize());
  cudaEvent_t begin{},end{}; Check(cudaEventCreate(&begin));Check(cudaEventCreate(&end));
  Check(cudaEventRecord(begin));for(int i=0;i<40;++i) launch();Check(cudaEventRecord(end));Check(cudaEventSynchronize(end));
  float ms=0;Check(cudaEventElapsedTime(&ms,begin,end));cudaEventDestroy(begin);cudaEventDestroy(end);
  std::printf("d=%d seq=%d input=%d output=%d group=%d window=%d causal=%d us=%.6f\n",d,seq,static_cast<int>(input),static_cast<int>(output),group,window,causal,ms*1000/40);
}
}
int main() {
  int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0) return 77;
  cudaDeviceProp p{};Check(cudaGetDeviceProperties(&p,0)); if(p.major!=6)return 77;
  try {
    for(int d:{64,128}) for(int seq:{1,7,16,17,128,256})
      for(auto input:{vt::DType::kF32,vt::DType::kBF16}) for(auto output:{vt::DType::kF32,vt::DType::kBF16})
        for(int group:{1,4}) for(bool window:{false,true}) Run(d,seq,input,output,group,window,true);
    Run(64,17,vt::DType::kBF16,vt::DType::kF32,4,true,false);
    for(int d:{64,128}) for(auto input:{vt::DType::kF32,vt::DType::kBF16})
      Run(d,17,input,vt::DType::kF32,4,false,true,true);
  } catch(const std::exception& e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
