// SPDX-License-Identifier: Apache-2.0
// Vision policy adapted from vllm-xpu-kernels ddf336d86e3c8602888572a3502f951abd51df12,
// fmha_utils.hpp chunk_policy_head96 and xe_2/chunk_prefill.hpp FP16 dispatch.
// The existing Torch-free Xe2 collective retains its original BSD-3-Clause
// notices. No Torch tensors or library dependency enter this native launcher.
#include "xpu_common.h"
#include "xpu_gptq4.h"
#include "xpu_vision_attention.h"
#include "csrc/xpu/attn/xe_2/chunk_prefill.hpp"
#include <cstdio>
#include <bit>
#include <cstdlib>
#include <initializer_list>

namespace vt::xpu {
namespace {
using namespace cute;
using VisionPolicy=FMHAConfig<
    Shape<_128,_32,_32>,Shape<_128,_32,_32>,Shape<_128,_96>,
    Layout<Shape<_8,_1,_1>>,void,2,false,false,false,false,false,
    half_t,half_t,half_t,half_t>;
}

bool VisionAttentionXe2Kernel(Queue& q,Tensor& out,const Tensor& query,
    const Tensor& key,const Tensor& value,const AttentionArgs& args) {
  const auto device=NativeQueue(q).get_device();
  if (!device.has(sycl::aspect::ext_intel_device_id) ||
      device.get_info<sycl::ext::intel::info::device::device_id>()!=57891 ||
      !device.has(sycl::aspect::ext_intel_matrix)) return false;
  for (const auto* tensor : std::initializer_list<const Tensor*>{&out,&query,&key,&value})
    if (reinterpret_cast<uintptr_t>(tensor->data)&15) return false;
  const int length=static_cast<int>(query.shape[0]);
  // The queue-owned 64-byte workspace is reused, with in-order dependencies.
  // One GPU write replaces host positions, metadata readback and per-call frees.
  auto* offsets=static_cast<int32_t*>(Exl3OneDnnScratchpad(q,64));
  NativeQueue(q).single_task([=] { offsets[0]=0;offsets[1]=length; });
  chunk_prefill_args_t donor{};
  donor.query=query.data;donor.key=key.data;donor.value=value.data;donor.out=out.data;
  donor.cu_seqlens_q=offsets;donor.cu_seqlens_k=offsets;
  donor.max_queries=donor.max_keys=donor.total_seqlen_q=donor.total_seqlen_k=length;
  donor.sm_scale=args.scale;donor.batch_size=1;
  donor.num_heads_q=donor.num_heads_k=16;donor.head_size=72;
  donor.is_varlen=true;donor.is_paged=false;donor.is_causal=false;
  donor.q_stride_seq=query.stride[0];donor.q_stride_heads=query.stride[1];
  donor.k_stride_seq=key.stride[0];donor.k_stride_heads=key.stride[1];
  donor.v_stride_seq=value.stride[0];donor.v_stride_heads=value.stride[1];
  donor.o_stride_seq=out.stride[0];donor.o_stride_heads=out.stride[1];
  const auto event=VisionPolicy::kernel_dispatch(NativeQueue(q),donor);
  RecordProfileEvent(q,"vision_xe2_sdpa",event);
  if (const char* trace=std::getenv("VT_XPU_TRACE_FAST_PATH");trace && trace[0]=='1')
    std::fprintf(stderr,"VISION_XE2_SDPA L=%d H=16 D=72 workspace=64 scale_bits=0x%08x\n",length,
                 std::bit_cast<uint32_t>(donor.sm_scale));
  return true;
}
}
