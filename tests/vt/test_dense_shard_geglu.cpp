// CPU contract for original-format GeGLU shards.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include "vllm/model_executor/layers/linear.h"
#include "vt/backend.h"

int main() {
  using namespace vt;
  Queue q{{DeviceType::kCPU, 0}, nullptr};
  constexpr int K=32, N=3, M=2;
  std::vector<uint16_t> x(M*K);
  for (int i=0;i<M*K;++i) x[i]=F32ToBF16(float(i%7-3)/8);
  Tensor a=Tensor::Contiguous(x.data(),DType::kBF16,q.device,{M,K});
  std::vector<vllm::OwnedTensor> parts(2);
  for(int j=0;j<2;++j) {
    auto& w=parts[j]; w.dtype=j?DType::kQ8_0:DType::kQ4_0;
    w.rank=2; w.shape[0]=N; w.shape[1]=K; w.nk=true;
    w.bytes.resize(N*RowSizeBytes(w.dtype,K));
    for(int r=0;r<N;++r) {
      auto* p=w.bytes.data()+r*RowSizeBytes(w.dtype,K);
      uint16_t scale=F32ToF16(float(r+1)/16); std::memcpy(p,&scale,2);
      for(int k=0;k<(j?32:16);++k) p[2+k]=j?uint8_t(k%9-4):uint8_t((k%8)|((15-k%8)<<4));
    }
  }
  std::vector<uint16_t> gu(M*2*N), expected(M*N);
  std::vector<uint16_t> g(M*N),u(M*N);
  Tensor products[]={Tensor::Contiguous(g.data(),DType::kBF16,q.device,{M,N}),
                     Tensor::Contiguous(u.data(),DType::kBF16,q.device,{M,N})};
  for(int j=0;j<2;++j) { auto w=parts[j].View(); MatmulBTQuant(q,products[j],a,w); }
  for(int m=0;m<M;++m) { std::memcpy(gu.data()+m*2*N,g.data()+m*N,N*2); std::memcpy(gu.data()+m*2*N+N,u.data()+m*N,N*2); }
  Tensor merged=Tensor::Contiguous(gu.data(),DType::kBF16,q.device,{M,2*N});
  Tensor ref=Tensor::Contiguous(expected.data(),DType::kBF16,q.device,{M,N});
  GeluAndMul(q,ref,merged);
  vllm::dense_attn::Dev d{GetBackend(DeviceType::kCPU),q};
  vllm::layers::DenseShardMlpGateUpGeluMethod method(&parts,N);
  auto result=method.Apply(d,a);
  if(std::memcmp(result.ptr(),expected.data(),expected.size()*2)) throw std::runtime_error("mixed GeGLU mismatch");
  Tensor weights[]={parts[0].View(),parts[1].View()};
  bool rejected=false;
  try { MergedGemmDense(q,kDenseGateUpGeluShards,products,a,weights,2,nullptr); } catch(const std::exception&) {rejected=true;}
  if(!rejected) throw std::runtime_error("missing epilogue output accepted");
  auto bad=kDenseGateUpGeluShards;bad.arity=3;rejected=false;
  try { MergedGemmDense(q,bad,products,a,weights,2,&ref); } catch(const std::exception&) {rejected=true;}
  if(!rejected) throw std::runtime_error("bad arity accepted");
  std::vector<uint16_t> sw(M*N); Tensor swref=Tensor::Contiguous(sw.data(),DType::kBF16,q.device,{M,N});
  // Form the reference before dispatch can overwrite either contraction.
  MoeSiluMul(q,swref,products[0],products[1]);
  std::fill(g.begin(),g.end(),uint16_t{0x7fc1});
  std::fill(u.begin(),u.end(),uint16_t{0x7fc1});
  std::fill(expected.begin(),expected.end(),uint16_t{0x7fc1});
  MergedGemmDense(q,kDenseGateUpShards,products,a,weights,2,&ref);
  if(std::memcmp(sw.data(),expected.data(),sw.size()*2)) throw std::runtime_error("SwiGLU changed");
  // Q, K and V use distinct widths and compact formats. Standalone contractions
  // are the reference for the shared dispatch, with separate poisoned outputs.
  constexpr int widths[]={N,2,4};
  std::vector<vllm::OwnedTensor> qkv_parts(3);
  std::vector<std::vector<uint16_t>> qkv_expected(3),qkv_actual(3);
  Tensor qkv_weights[3],qkv_outputs[3];
  for(int j=0;j<3;++j) {
    auto& w=qkv_parts[j];w.dtype=j==1?DType::kQ8_0:DType::kQ4_0;
    w.rank=2;w.shape[0]=widths[j];w.shape[1]=K;w.nk=true;
    const size_t row_bytes=RowSizeBytes(w.dtype,K);
    w.bytes.resize(widths[j]*row_bytes);
    for(int r=0;r<widths[j];++r) {
      auto* p=w.bytes.data()+r*row_bytes;
      uint16_t scale=F32ToF16(float((j+1)*(r+1))/32);
      std::memcpy(p,&scale,2);
      for(int k=0;k<(j==1?32:16);++k)
        p[2+k]=j==1?uint8_t((k+r)%11-5):
                   uint8_t(((k+j+r)%16)|(((3*k+j+2*r)%16)<<4));
    }
    qkv_weights[j]=w.View();
    qkv_expected[j].resize(M*widths[j]);
    qkv_actual[j].assign(M*widths[j],uint16_t{0x7fc1});
    Tensor reference=Tensor::Contiguous(qkv_expected[j].data(),DType::kBF16,
                                       q.device,{M,widths[j]});
    MatmulBTQuant(q,reference,a,qkv_weights[j]);
    qkv_outputs[j]=Tensor::Contiguous(qkv_actual[j].data(),DType::kBF16,
                                     q.device,{M,widths[j]});
  }
  MergedGemmGroup none{3,MergedEpilogue::kNone,kNoMergedFastOp,"qkv"};
  MergedGemmDense(q,none,qkv_outputs,a,qkv_weights,3);
  for(int j=0;j<3;++j)
    for(size_t i=0;i<qkv_expected[j].size();++i)
      if(qkv_actual[j][i]!=qkv_expected[j][i])
        throw std::runtime_error("no-epilogue QKV shard "+std::to_string(j)+
                                 " element "+std::to_string(i)+" mismatch");
  std::cout<<"mixed Q4/Q8 GeGLU, guards, unchanged SwiGLU and no epilogue PASS\n";
}
