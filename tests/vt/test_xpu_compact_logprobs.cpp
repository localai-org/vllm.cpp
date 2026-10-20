#include "xpu_test_helpers.h"
#include "vt/xpu_sampling.h"
#include "vt/xpu_argmax_plan.h"
#include "vt/xpu.h"
#include "vt/breakable_graph.h"
#include "vllm/v1/sample/sampler.h"
#include <bit>
#include <cstdlib>
#include <limits>
#include <string>

namespace {
using xpu_test::Buffer;
using xpu_test::Queue;
using vt::DType;
using vt::xpu::SelectedLogprobRecord;
constexpr float Inf = std::numeric_limits<float>::infinity();
constexpr float NaN = std::numeric_limits<float>::quiet_NaN();
std::vector<float> Values(int rows, int vocab, int mode) {
  auto data = xpu_test::Values(size_t(rows) * vocab,mode,.125f);
  for (int r=0; r<rows; ++r) {
    float* row = data.data()+size_t(r)*vocab;
    if (mode == 1) std::fill(row,row+vocab,-100.f); // Rounded subtraction ties.
    if (mode == 2) { std::fill(row,row+vocab,-Inf); row[vocab-1]=Inf; }
    if (mode == 3) std::fill(row,row+vocab,NaN);
    if (mode == 4) { std::fill(row,row+vocab,0.f); row[vocab-1]=-0.f; }
    if (mode == 5) { row[0]=NaN; row[vocab-1]=std::bit_cast<float>(uint32_t{0x7fa12345}); }
  }
  return data;
}
std::vector<SelectedLogprobRecord> Expected(const std::vector<float>& data, const std::vector<int64_t>& ids, int vocab) {
  std::vector<SelectedLogprobRecord> records(ids.size());
  for (size_t r=0; r<ids.size(); ++r) {
    const auto id = ids[r];
    records[r] = {id,0,-1};
    if (id<0 || id>=vocab) continue;
    const float* row = data.data()+r*vocab;
    std::memcpy(&records[r].logprob,row+id,sizeof(float));
    int rank=0; for (int c=0; c<vocab; ++c) rank += row[c]>=row[id];
    records[r].rank=rank;
  }
  return records;
}
}

TEST_CASE("XPU compact logprobs0: exact stored values ranks exceptional bits and tails") {
  Queue gpu(vt::DeviceType::kXPU);
  for (int rows : {1,4}) for (int vocab : {1,4095,4096,4097,8193,248320}) {
    CAPTURE(rows);
    CAPTURE(vocab);
    const auto bytes = vt::XpuCompactLogprobsWorkspaceBytes(rows,vocab);
    Buffer input(gpu.q,DType::kF32,{rows+1,vocab}), selected(gpu.q,DType::kI64,{rows+2}),
        output(gpu.q,DType::kI8,{int64_t(rows*sizeof(SelectedLogprobRecord)+32)}),
        workspace(gpu.q,DType::kI8,{int64_t(bytes+32)});
    auto in=input.tensor; in.shape[0]=rows;
    auto ids=vt::Tensor::Contiguous(static_cast<int64_t*>(selected.tensor.data)+1,DType::kI64,gpu.q.device,{rows});
    auto out=vt::Tensor::Contiguous(static_cast<unsigned char*>(output.tensor.data)+16,DType::kI8,gpu.q.device,{int64_t(rows*sizeof(SelectedLogprobRecord))});
    auto scratch=vt::Tensor::Contiguous(static_cast<unsigned char*>(workspace.tensor.data)+16,DType::kI8,gpu.q.device,{int64_t(bytes)});
    for (int mode=0; mode<8; ++mode) {
      CAPTURE(mode);
      auto data=Values(rows,vocab,mode);
      std::vector<int64_t> wanted(rows);
      for (int r=0; r<rows; ++r) wanted[r]=mode==6 ? -1 : mode==7 ? vocab : mode%2 ? vocab-1 : (r*997)%vocab;
      auto expected=Expected(data,wanted,vocab);
      data.resize(size_t(rows+1)*vocab,NaN);input.put(data);
      auto padded=wanted;padded.insert(padded.begin(),-987);padded.push_back(-987);selected.upload(padded.data());
      output.upload(std::vector<unsigned char>(output.bytes,0xa5).data());
      workspace.upload(std::vector<unsigned char>(workspace.bytes,0x9d).data());
      const auto before=input.download(), before_ids=selected.download();
      (void)vt::xpu::DrainProfileEvents();
      vt::xpu::CompactRawLogprobs0(gpu.q,out,in,ids,scratch);
      const auto result=output.download();
      std::vector<unsigned char> expected_bytes(rows*sizeof(SelectedLogprobRecord));
      std::memcpy(expected_bytes.data(),expected.data(),expected_bytes.size());
      xpu_test::SameBytes({result.begin()+16,result.end()-16},expected_bytes);
      CHECK(std::all_of(result.begin(),result.begin()+16,[](auto b){return b==0xa5;}));
      CHECK(std::all_of(result.end()-16,result.end(),[](auto b){return b==0xa5;}));
      const auto work=workspace.download();
      CHECK(std::all_of(work.begin(),work.begin()+16,[](auto b){return b==0x9d;}));
      CHECK(std::all_of(work.end()-16,work.end(),[](auto b){return b==0x9d;}));
      xpu_test::SameBytes(input.download(),before);xpu_test::SameBytes(selected.download(),before_ids);
      if (std::getenv("VT_XPU_PROFILE")) {
        const auto events=vt::xpu::DrainProfileEvents();
        for (const char* stage : {"compact_logprobs0_rank_partials","compact_logprobs0_gather"})
          CHECK(std::count_if(events.begin(),events.end(),[&](const auto& e){return e.stage==stage;})==1);
      }
    }
  }
}

TEST_CASE("XPU compact logprobs0: invalid metadata scratch and aliases refuse before writes") {
  Queue gpu(vt::DeviceType::kXPU);
  constexpr int rows=4,vocab=8193;
  const auto required=vt::XpuCompactLogprobsWorkspaceBytes(rows,vocab);
  Buffer input(gpu.q,DType::kF32,{rows,vocab}), ids(gpu.q,DType::kI64,{rows}),
      output(gpu.q,DType::kI8,{int64_t(rows*sizeof(SelectedLogprobRecord))}), work(gpu.q,DType::kI8,{int64_t(required)});
  input.put(Values(rows,vocab,0));ids.upload(std::vector<int64_t>(rows,0).data());
  output.upload(std::vector<unsigned char>(output.bytes,0xa5).data());work.upload(std::vector<unsigned char>(work.bytes,0x9d).data());
  const auto before=input.download(), before_ids=ids.download(), before_out=output.download(), before_work=work.download();
  const auto refuse=[&](vt::Tensor in, vt::Tensor selected, vt::Tensor out, vt::Tensor scratch) {
    CHECK_THROWS(vt::xpu::CompactRawLogprobs0(gpu.q,out,in,selected,scratch));
    xpu_test::SameBytes(input.download(),before);xpu_test::SameBytes(ids.download(),before_ids);
    xpu_test::SameBytes(output.download(),before_out);xpu_test::SameBytes(work.download(),before_work);
  };
  auto bad=work.tensor;--bad.shape[0];refuse(input.tensor,ids.tensor,output.tensor,bad);
  bad=work.tensor;bad.data=static_cast<unsigned char*>(bad.data)+1;refuse(input.tensor,ids.tensor,output.tensor,bad);
  bad=work.tensor;bad.shape[0]=vt::kXpuArgmaxMaxWorkspaceBytes;refuse(input.tensor,ids.tensor,output.tensor,bad);
  bad=output.tensor;--bad.shape[0];refuse(input.tensor,ids.tensor,bad,work.tensor);
  bad=output.tensor;bad.data=static_cast<unsigned char*>(bad.data)+4;refuse(input.tensor,ids.tensor,bad,work.tensor);
  bad=output.tensor;bad.dtype=DType::kF32;refuse(input.tensor,ids.tensor,bad,work.tensor);
  bad=ids.tensor;bad.dtype=DType::kI32;refuse(input.tensor,bad,output.tensor,work.tensor);
  bad=ids.tensor;bad.shape[0]=3;refuse(input.tensor,bad,output.tensor,work.tensor);
  bad=input.tensor;bad.stride[0]+=1;refuse(bad,ids.tensor,output.tensor,work.tensor);
  bad=input.tensor;bad.dtype=DType::kF16;refuse(bad,ids.tensor,output.tensor,work.tensor);
  bad=input.tensor;bad.device.index=1;refuse(bad,ids.tensor,output.tensor,work.tensor);
  bad=input.tensor;bad.shape[0]=std::numeric_limits<int64_t>::max();
  auto huge=ids.tensor;huge.shape[0]=bad.shape[0];refuse(bad,huge,output.tensor,work.tensor);
  bad=input.tensor;bad.shape[1]=std::numeric_limits<int64_t>::max();refuse(bad,ids.tensor,output.tensor,work.tensor);
  for (void* pointer : {input.tensor.data,ids.tensor.data,output.tensor.data}) {
    bad=work.tensor;bad.data=pointer;refuse(input.tensor,ids.tensor,output.tensor,bad);
  }
  for (void* pointer : {input.tensor.data,ids.tensor.data}) {
    bad=output.tensor;bad.data=pointer;refuse(input.tensor,ids.tensor,bad,work.tensor);
  }
  auto in=input.tensor, selected=ids.tensor, out=output.tensor, scratch=work.tensor;
  in.shape[0]=selected.shape[0]=out.shape[0]=scratch.shape[0]=0;
  out.data=scratch.data=nullptr;
  (void)vt::xpu::DrainProfileEvents();
  CHECK_NOTHROW(vt::xpu::CompactRawLogprobs0(gpu.q,out,in,selected,scratch));
  CHECK(vt::xpu::DrainProfileEvents().empty());
  xpu_test::SameBytes(output.download(),before_out);xpu_test::SameBytes(work.download(),before_work);
}

TEST_CASE("XPU compact logprobs0: independent graph queues consume changing snapshot and IDs") {
  Queue first(vt::DeviceType::kXPU),second(vt::DeviceType::kXPU);
  auto& backend=vt::GetBackend(first.q.device);
  REQUIRE(backend.SupportsGraphCapture());
  constexpr int rows=4,vocab=8193;
  const auto bytes=vt::XpuCompactLogprobsWorkspaceBytes(rows,vocab);
  Buffer a(first.q,DType::kF32,{rows,vocab}), b(second.q,DType::kF32,{rows,vocab}),
    ia(first.q,DType::kI64,{rows}), ib(second.q,DType::kI64,{rows}),
    oa(first.q,DType::kI8,{rows*sizeof(SelectedLogprobRecord)}),ob(second.q,DType::kI8,{rows*sizeof(SelectedLogprobRecord)}),
    wa(first.q,DType::kI8,{int64_t(bytes)}),wb(second.q,DType::kI8,{int64_t(bytes)});
  a.put(Values(rows,vocab,0));b.put(Values(rows,vocab,1));
  ia.upload(std::vector<int64_t>(rows,0).data());ib.upload(std::vector<int64_t>(rows,vocab-1).data());
  const auto apply=[&](int i) {vt::xpu::CompactRawLogprobs0(i?second.q:first.q,i?ob.tensor:oa.tensor,
      i?b.tensor:a.tensor,i?ib.tensor:ia.tensor,i?wb.tensor:wa.tensor);};
  (void)vt::xpu::DrainHostProfileRecords();
  for (int i=0;i<2;++i) apply(i);
  CHECK(vt::xpu::DrainHostProfileRecords().empty());
  backend.Synchronize(first.q);backend.Synchronize(second.q);
  vt::BreakableGraph graphs[2];
  for (int i=0;i<2;++i) {
    vt::GraphCaptureScope capture(backend,i?second.q:first.q,graphs[i],vt::GraphCaptureMode::kFull);apply(i);
  }
  REQUIRE(graphs[0].captured());REQUIRE(graphs[1].captured());
  const auto memory=vt::xpu::GetMemoryInfo();
  for (int mode=0;mode<6;++mode) {
    auto av=Values(rows,vocab,mode),bv=Values(rows,vocab,(mode+3)%6);
    std::vector<int64_t> ai{0,vocab-1,4096,17},bi{vocab-1,0,17,4096};
    if(mode%2) {ai[0]=-1;bi[1]=vocab;}
    a.put(av);b.put(bv);ia.upload(ai.data());ib.upload(bi.data());
    (void)vt::xpu::DrainHostProfileRecords();
    graphs[1].Replay(second.q);graphs[0].Replay(first.q);
    // The existing graph wrapper records submission itself. Reject every
    // allocation/readback/wait record, while retaining its normal submit span.
    const auto host=vt::xpu::DrainHostProfileRecords();
    CHECK(std::all_of(host.begin(),host.end(),[](const auto& r){return r.stage=="graph_compute_submit";}));
    for (int i=0;i<2;++i) {
      const auto expected=Expected(i?bv:av,i?bi:ai,vocab);
      std::vector<unsigned char> raw(expected.size()*sizeof(SelectedLogprobRecord));std::memcpy(raw.data(),expected.data(),raw.size());
      xpu_test::SameBytes((i?ob:oa).download(),raw);
    }
  }
  backend.Synchronize(first.q);backend.Synchronize(second.q);
  const auto after=vt::xpu::GetMemoryInfo();
  CHECK(after.allocated_bytes==memory.allocated_bytes);CHECK(after.pinned_bytes==memory.pinned_bytes);
  CHECK(after.graph_count==memory.graph_count);CHECK(after.graph_device_bytes==memory.graph_device_bytes);
  for(auto& graph:graphs) graph.Reset(); // Retire borrowers before buffer owners.
}

TEST_CASE("XPU compact logprobs0: sampler preserves snapshot rounded ranks and fallback modes") {
  Queue first(vt::DeviceType::kXPU),second(vt::DeviceType::kXPU);
  struct Restore {
    bool had=std::getenv("VT_XPU_COMPACT_LOGPROBS0")!=nullptr;
    std::string old=had?std::getenv("VT_XPU_COMPACT_LOGPROBS0"):"";
    ~Restore(){if(had)setenv("VT_XPU_COMPACT_LOGPROBS0",old.c_str(),1);else unsetenv("VT_XPU_COMPACT_LOGPROBS0");}
  } restore;
  const auto initial=vt::xpu::GetMemoryInfo();
  for (auto mode : {vllm::v1::LogprobsMode::kRawLogprobs,vllm::v1::LogprobsMode::kProcessedLogprobs,
                    vllm::v1::LogprobsMode::kRawLogits,vllm::v1::LogprobsMode::kProcessedLogits}) {
    vllm::v1::Sampler sampler(mode);
    for (int rows : {1,4,1}) for (int requested : {-2,0,1,2}) {
      auto& q=rows==4?second.q:first.q;
      constexpr int vocab=248320;
      Buffer input(q,DType::kF32,{rows,vocab});
      std::vector<float> data(size_t(rows)*vocab,-20);
      for(int r=0;r<rows;++r){data[size_t(r)*vocab]=0;data[size_t(r)*vocab+1]=-1e-8f;}
      vllm::v1::SamplingMetadata sm;
      if(requested!=-2)sm.max_num_logprobs=requested==2?0:requested;
      if(requested==2)sm.logprob_token_ids=std::map<int,std::vector<int32_t>>{{0,{0,7}}};
      // A processor changes the sampled token, while raw LPs must still come
      // from the pre-processor snapshot. Other rows retain a rounded-logprob tie.
      if(rows==4)sm.logit_bias[0][7]=50;
      if(rows==4 && requested==0 && mode==vllm::v1::LogprobsMode::kRawLogprobs) {
        // This upstream mask uses1 for EXCLUDED tokens.
        sm.allowed_token_ids_mask.emplace(rows,std::vector<uint8_t>(vocab,0));
        std::fill((*sm.allowed_token_ids_mask)[2].begin(),(*sm.allowed_token_ids_mask)[2].end(),1);
        (*sm.allowed_token_ids_mask)[2][5]=0;
        sm.no_penalties=false;
        sm.prompt_token_ids.emplace(rows,std::vector<int32_t>{});
        sm.output_token_ids.resize(rows);sm.output_token_ids[3]={0};
        sm.frequency_penalties.assign(rows,0);sm.frequency_penalties[3]=1;
        sm.presence_penalties.assign(rows,0);sm.repetition_penalties.assign(rows,1);
      }
      setenv("VT_XPU_COMPACT_LOGPROBS0","0",1);input.put(data);
      const auto control=sampler.forward(q,input.tensor,sm);
      setenv("VT_XPU_COMPACT_LOGPROBS0","1",1);input.put(data);
      (void)vt::xpu::DrainProfileEvents();(void)vt::xpu::DrainHostProfileRecords();
      const auto candidate=sampler.forward(q,input.tensor,sm);
      const auto events=vt::xpu::DrainProfileEvents();
      const auto host=vt::xpu::DrainHostProfileRecords();
      const bool eligible=mode==vllm::v1::LogprobsMode::kRawLogprobs && requested==0;
      CHECK(candidate.sampled_token_ids==control.sampled_token_ids);
      REQUIRE(candidate.logprobs_tensors.has_value()==control.logprobs_tensors.has_value());
      if(control.logprobs_tensors){
        const auto& a=*control.logprobs_tensors;const auto& b=*candidate.logprobs_tensors;
        CHECK(a.num_positions==b.num_positions);CHECK(a.num_tokens_per_position==b.num_tokens_per_position);
        CHECK(a.logprob_token_ids==b.logprob_token_ids);CHECK(a.selected_token_ranks==b.selected_token_ranks);
        REQUIRE(a.logprobs.size()==b.logprobs.size());
        CHECK(std::memcmp(a.logprobs.data(),b.logprobs.data(),a.logprobs.size()*sizeof(float))==0);
        if(eligible&&rows==1) CHECK(b.selected_token_ranks[0]==2); // Raw-logit rank would be1.
        if(eligible&&rows==4) CHECK(b.logprob_token_ids[0]==7);
        if(eligible&&rows==4) CHECK(b.logprob_token_ids[2]==5);
        if(eligible&&rows==4) CHECK(b.logprob_token_ids[3]==1);
      }
      if(std::getenv("VT_XPU_PROFILE")) for(const char* stage:{"compact_logprobs0_rank_partials","compact_logprobs0_gather"})
        CHECK(std::count_if(events.begin(),events.end(),[&](const auto& e){return e.stage==stage;})==(eligible?1:0));
      if(eligible) CHECK(std::none_of(host.begin(),host.end(),[](const auto& r){return r.stage=="staged_d2h_wait" && r.copy_bytes>=248320*sizeof(float);}));
    }
    if(mode==vllm::v1::LogprobsMode::kRawLogprobs) {
      Buffer input(first.q,DType::kF32,{4,248320});
      input.put(std::vector<float>(4*248320,0));
      vllm::v1::SamplingMetadata sm;sm.max_num_logprobs=0;
      setenv("VT_XPU_COMPACT_LOGPROBS0","1",1);
      for(int i=0;i<2;++i)sampler.forward(first.q,input.tensor,sm);
      const auto warm=vt::xpu::GetMemoryInfo();
      for(int i=0;i<8;++i)sampler.forward(first.q,input.tensor,sm);
      const auto after=vt::xpu::GetMemoryInfo();
      CHECK(after.allocated_bytes==warm.allocated_bytes);CHECK(after.pinned_bytes==warm.pinned_bytes);
      CHECK(after.graph_count==warm.graph_count);CHECK(after.graph_device_bytes==warm.graph_device_bytes);
    }
  }
  const auto retired=vt::xpu::GetMemoryInfo();
  CHECK(retired.allocated_bytes==initial.allocated_bytes);CHECK(retired.pinned_bytes==initial.pinned_bytes);
}

TEST_CASE("XPU compact logprobs0: sampler preserves nonfinite log-softmax payloads") {
  Queue gpu(vt::DeviceType::kXPU);
  const char* old=std::getenv("VT_XPU_COMPACT_LOGPROBS0");
  const bool had=old!=nullptr;const std::string saved=old?old:"";
  struct Restore{bool had;std::string saved;~Restore(){if(had)setenv("VT_XPU_COMPACT_LOGPROBS0",saved.c_str(),1);else unsetenv("VT_XPU_COMPACT_LOGPROBS0");}} restore{had,saved};
  vllm::v1::Sampler sampler;vllm::v1::SamplingMetadata sm;sm.max_num_logprobs=0;
  Buffer input(gpu.q,DType::kF32,{1,248320});
  for(int mode : {0,2,3,4,5}) {
    auto data=Values(1,248320,mode);
    setenv("VT_XPU_COMPACT_LOGPROBS0","0",1);input.put(data);
    const auto a=sampler.forward(gpu.q,input.tensor,sm);
    setenv("VT_XPU_COMPACT_LOGPROBS0","1",1);input.put(data);
    const auto b=sampler.forward(gpu.q,input.tensor,sm);
    CHECK(a.sampled_token_ids==b.sampled_token_ids);
    REQUIRE(a.logprobs_tensors.has_value());REQUIRE(b.logprobs_tensors.has_value());
    CHECK(a.logprobs_tensors->logprob_token_ids==b.logprobs_tensors->logprob_token_ids);
    CHECK(a.logprobs_tensors->selected_token_ranks==b.logprobs_tensors->selected_token_ranks);
    CHECK(std::memcmp(a.logprobs_tensors->logprobs.data(),b.logprobs_tensors->logprobs.data(),sizeof(float))==0);
  }
}
