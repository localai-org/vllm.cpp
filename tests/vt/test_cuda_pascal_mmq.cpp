// llama.cpp 62318cd57 mmq.cuh:60-76 and quantize.cu:457-550.
// Independent scalar D4 reference through the production registered operator.
#include <cuda_runtime.h>
#include "vt/ops.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace {
void Check(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Buffer {
  void* p = nullptr;
  explicit Buffer(size_t n) { Check(cudaMallocManaged(&p,n)); }
  ~Buffer() { cudaFree(p); }
};
float Bits(uint32_t bits) { float x;std::memcpy(&x,&bits,4);return x; }
float WScale(const uint8_t* b) { uint16_t d;std::memcpy(&d,b,2);return vt::F16ToF32(d); }
int WQuant(const uint8_t* b, int j, bool q5) {
  if(!q5) return static_cast<int8_t>(b[2+j]);
  uint32_t hi;std::memcpy(&hi,b+2,4);
  return ((b[6+j%16] >> (j>=16?4:0))&15) + int((hi>>j)&1)*16 -16;
}
float Read(void* p, vt::DType dt, size_t i) {
  return dt==vt::DType::kF32?static_cast<float*>(p)[i]:vt::BF16ToF32(static_cast<uint16_t*>(p)[i]);
}
void Write(void* p, vt::DType dt, size_t i, float x) {
  if(dt==vt::DType::kF32) static_cast<float*>(p)[i]=x;
  else static_cast<uint16_t*>(p)[i]=vt::F32ToBF16(x);
}
std::vector<uint8_t> Load(const char* path) {
  std::ifstream f(path,std::ios::binary|std::ios::ate);
  if(!f) throw std::runtime_error("missing fixture");
  const auto n=f.tellg();f.seekg(0);std::vector<uint8_t> v(static_cast<size_t>(n));
  if(!f.read(reinterpret_cast<char*>(v.data()),n))throw std::runtime_error("short fixture");
  return v;
}
void Run(int m,int n,int k,bool q5,vt::DType in,vt::DType out,
         const std::vector<float>& x,const std::vector<uint8_t>& w,
         const std::vector<float>* oracle=nullptr) {
  const int nb=k/32,bytes=q5?22:34;
  Buffer a(static_cast<size_t>(m*k)*vt::SizeOf(in)),b(w.size()),o(static_cast<size_t>(m*n)*vt::SizeOf(out));
  for(int i=0;i<m*k;++i)Write(a.p,in,i,x[static_cast<size_t>(i)]);
  std::memcpy(b.p,w.data(),w.size());
  const vt::Device device{vt::DeviceType::kCUDA,0};vt::Queue queue{device,nullptr};
  auto at=vt::Tensor::Contiguous(a.p,in,device,{m,k});
  auto wt=vt::Tensor::Contiguous(b.p,q5?vt::DType::kQ5_0:vt::DType::kQ8_0,device,{n,k});
  auto ot=vt::Tensor::Contiguous(o.p,out,device,{m,n});
  const auto hits=vt::GetReferenceTierHits();vt::MatmulBTQuant(queue,ot,at,wt);Check(cudaDeviceSynchronize());
  if(vt::GetReferenceTierHits()!=hits)throw std::runtime_error("unexpected CPU fallback");
  double error=0,max_error=0;
  for(int r=0;r<m;++r)for(int c=0;c<n;++c) {
    double expected=0,magnitude=0;
    for(int block=0;block<nb;++block) {
      float vals[32],amax=0;
      for(int j=0;j<32;++j){vals[j]=Read(a.p,in,static_cast<size_t>(r*k+block*32+j));amax=std::max(amax,std::abs(vals[j]));}
      const bool d4=m>8;
      const float inv=d4?(amax==0?0:127.0f/amax):(amax==0?0:1.0f/(amax/127.0f));
      const float scale=d4?(inv==0?0:1.0f/inv):vt::F16ToF32(vt::F32ToF16(amax/127.0f));
      const auto* wb=w.data()+(c*nb+block)*bytes;
      int dot=0;
      for(int j=0;j<32;++j) {
        const int quant=amax==0?0:static_cast<int>(std::round(d4?vals[j]*inv:(q5&&m==1?vals[j]/(amax/127.0f):vals[j]*inv)));
        dot+=WQuant(wb,j,q5)*quant;
      }
      if(q5&&m==1) {
        float tree[32];std::copy(vals,vals+32,tree);
        for(int mask:{16,8,4,2,1}){float prior[32];std::copy(tree,tree+32,prior);for(int j=0;j<32;++j)tree[j]=prior[j]+prior[j^mask];}
        int unsigned_dot=0;
        for(int j=0;j<32;++j) unsigned_dot+=(WQuant(wb,j,true)+16)*(amax==0?0:static_cast<int>(std::round(vals[j]/(amax/127.0f))));
        expected+=WScale(wb)*(unsigned_dot*scale-16.0f*vt::F16ToF32(vt::F32ToF16(tree[0])));
      } else {
        const double term=static_cast<double>(WScale(wb)*scale)*dot;
        expected+=term; magnitude+=std::abs(term);
      }
    }
    float want=static_cast<float>(expected);
    if(out==vt::DType::kBF16)want=vt::BF16ToF32(vt::F32ToBF16(want));
    const float actual=Read(o.p,out,static_cast<size_t>(r*n+c));
    const double tolerance=(out==vt::DType::kBF16?0.008:0.00001)*std::abs(want)+0.000003*magnitude+1e-11;
    if(!std::isfinite(actual)||std::abs(actual-want)>tolerance) {
      std::fprintf(stderr,"m=%d n=%d k=%d q5=%d in=%d out=%d row=%d col=%d actual=%.9g expected=%.9g\n",m,n,k,q5,int(in),int(out),r,c,actual,want);
      throw std::runtime_error("registered Pascal MMQ differs from independent activation contract");
    }
    if(oracle){const double e=actual-(*oracle)[static_cast<size_t>(r*n+c)];error+=e*e;max_error=std::max(max_error,std::abs(e));}
  }
  if(oracle&&(std::sqrt(error/(m*n))>1e-6||max_error>1e-5))
    throw std::runtime_error("actual projection differs from captured pinned CUDA oracle");
  std::printf("PASS m=%d n=%d k=%d q5=%d in=%d out=%d",m,n,k,q5,int(in),int(out));
  if(oracle)std::printf(" oracle_rmse=%.9g oracle_max=%.9g",std::sqrt(error/(m*n)),max_error);
  std::puts("");
}
std::vector<uint8_t> Weights(int n,int k,bool q5) {
  const int bytes=q5?22:34;std::vector<uint8_t>w(static_cast<size_t>(n*k/32*bytes));
  for(int b=0;b<n*k/32;++b) {
    auto* block=w.data()+b*bytes;
    const float raw_scale=b%7==0?0.0f:b%7==1?0.000000059604644775390625f:
        (b%3==0?-1.0f:1.0f)*(.03125f*(b%7+1));
    const uint16_t scale=vt::F32ToF16(raw_scale);
    std::memcpy(block,&scale,2);
    uint32_t hi=0;
    for(int j=0;j<32;++j) {
      const int value=(b*3+j*7)%(q5?32:255)-(q5?16:127);
      if(q5){const int u=value+16;block[6+j%16]|=static_cast<uint8_t>((u&15)<<(j>=16?4:0));hi|=uint32_t(u>>4)<<j;}
      else block[2+j]=static_cast<uint8_t>(static_cast<int8_t>(value));
    }
    if(q5)std::memcpy(block+2,&hi,4);
  }
  return w;
}
}
int main(int argc,char** argv) {
  int count=0;if(cudaGetDeviceCount(&count)!=cudaSuccess||count==0)return 77;
  cudaDeviceProp prop{};Check(cudaGetDeviceProperties(&prop,0));if(prop.major!=6||prop.minor<1)return 77;
  try {
    if(argc==8) {
      const int m=std::stoi(argv[4]),n=std::stoi(argv[5]),k=std::stoi(argv[6]);const bool q5=std::stoi(argv[7])!=0;
      auto aa=Load(argv[1]),w=Load(argv[2]),gold=Load(argv[3]);
      if(aa.size()!=size_t(m*k*4)||gold.size()!=size_t(m*n*4)||w.size()!=size_t(n*k/32*(q5?22:34)))throw std::runtime_error("actual fixture shape mismatch");
      std::vector<float>x(m*k),oracle(m*n);std::memcpy(x.data(),aa.data(),aa.size());std::memcpy(oracle.data(),gold.data(),gold.size());
      Run(m,n,k,q5,vt::DType::kF32,vt::DType::kF32,x,w,&oracle);return 0;
    }
    for(bool q5:{false,true})for(int m:{1,8,9,16,128})for(int k:{32,576})
      for(auto in:{vt::DType::kF32,vt::DType::kBF16})for(auto out:{vt::DType::kF32,vt::DType::kBF16}) {
        std::vector<float>x(m*k);
        for(int r=0;r<m;++r)for(int j=0;j<k;++j) {
          float value=std::sin(j*.31f)*(1+(r%7));
          if(r==0&&m>1)value=0;
          if(r==1)value*=1e-7f;
          if(r==2)value=j%32==0?Bits(0x3fa25c7f):j%32==1?Bits(0x3c7575a9):0;
          x[static_cast<size_t>(r*k+j)]=value;
        }
        Run(m,7,k,q5,in,out,x,Weights(7,k,q5));
      }
  }catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 1;}
}
