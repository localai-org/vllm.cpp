// Actual four-geometry tower comparison; diagnostic readbacks only.
#include <algorithm>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>
#include "vllm/entrypoints/openai/chat_mm.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vt/op_provider.h"
#include "vt/unaligned.h"
#include "vt/xpu.h"

namespace {
using json = nlohmann::json;
void Require(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}
std::vector<unsigned char> Read(const std::filesystem::path& path, uint64_t limit) {
  const auto length = std::filesystem::file_size(path);
  Require(length > 0 && length <= limit, "file exceeds diagnostic budget");
  std::ifstream input(path, std::ios::binary);
  Require(input.good(), "cannot open diagnostic input");
  std::vector<unsigned char> bytes{std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
  Require(bytes.size() == length, "incomplete diagnostic input");
  return bytes;
}
struct Queue {
  vt::Queue value = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~Queue() { vt::DestroyQueue(value); }
};
struct Buffer {
  vt::Tensor value;
  Buffer(vt::Device device, int64_t rows, int64_t columns)
      : value(vt::Tensor::Contiguous(nullptr, vt::DType::kF16, device, {rows, columns})) {
    value.data = vt::Alloc(device, value.Bytes());
  }
  ~Buffer() { vt::Free(value.device, value.data); }
  Buffer(const Buffer&) = delete;
};
std::vector<unsigned char> Boundary(const std::filesystem::path& root, const json& entry,
                                    const char* dtype) {
  const auto name = entry.at("file").get<std::string>();
  Require(std::filesystem::path(name).filename() == name && entry.at("dtype") == dtype,
          "invalid boundary path/dtype");
  uint64_t count = 1;
  for (const auto& dim : entry.at("shape")) {
    const auto size = dim.get<uint64_t>();
    Require(size > 0 && size <= 16384 * 1536 && count <= (16384 * 1536) / size,
            "invalid boundary shape");
    count *= size;
  }
  const auto bytes = Read(root / name, 256 * 1024 * 1024);
  Require(bytes.size() == count * (std::string(dtype) == "torch.int64" ? 8 : 2),
          "boundary shape/byte mismatch");
  return bytes;
}
json Compare(const std::vector<float>& actual, const std::vector<unsigned char>& reference,
             double relative_limit, double absolute_limit) {
  Require(actual.size() * 2 == reference.size(), "native/reference boundary shape differs");
  double error = 0, norm = 0, max_abs = 0;
  bool finite = true, storage_equal = true;
  for (size_t i = 0; i < actual.size(); ++i) {
    const auto bits = vt::LoadUnaligned<uint16_t>(reference.data() + 2 * i);
    const float expected = vt::F16ToF32(bits);
    const double delta = static_cast<double>(actual[i]) - expected;
    finite &= std::isfinite(actual[i]) && std::isfinite(expected);
    storage_equal &= vt::F32ToF16(actual[i]) == bits;
    error += delta * delta; norm += static_cast<double>(expected) * expected;
    max_abs = std::max(max_abs, std::abs(delta));
  }
  const double relative = std::sqrt(error / std::max(norm, 1e-30));
  const bool exact = relative_limit == 0 && absolute_limit == 0;
  return {{"pass", finite && relative <= relative_limit && max_abs <= absolute_limit && (!exact || storage_equal)},
          {"elements", actual.size()}, {"finite", finite}, {"storage_exact", storage_equal},
          {"rel_l2", relative}, {"max_abs", max_abs},
          {"rel_l2_limit", relative_limit}, {"max_abs_limit", absolute_limit}};
}
json DetailedCompare(const std::vector<float>& actual, const std::vector<unsigned char>& reference,
                     int64_t columns) {
  auto report=Compare(actual,reference,0,0);
  report.erase("pass");report.erase("rel_l2_limit");report.erase("max_abs_limit");
  Require(columns>0 && actual.size()%columns==0,"invalid diagnostic row geometry");
  std::vector<double> errors(actual.size()),row_error(actual.size()/columns),row_norm(row_error.size());
  std::vector<double> group_error((columns+63)/64),group_norm(group_error.size());
  std::vector<size_t> top;
  size_t exact=0,finite=0;uint32_t maximum_ulp=0;
  const auto ordered=[](uint16_t bits) {return bits&0x8000 ? 0x8000-int(bits&0x7fff) : 0x8000+int(bits);};
  for (size_t i=0;i<actual.size();++i) {
    const auto bits=vt::LoadUnaligned<uint16_t>(reference.data()+2*i);
    const auto native_bits=vt::F32ToF16(actual[i]);
    const double expected=vt::F16ToF32(bits),delta=double(actual[i])-expected;
    if (!std::isfinite(actual[i]) || !std::isfinite(expected)) continue;
    ++finite;exact+=native_bits==bits;
    maximum_ulp=std::max(maximum_ulp,uint32_t(std::abs(ordered(native_bits)-ordered(bits))));
    errors[i]=std::abs(delta);row_error[i/columns]+=delta*delta;row_norm[i/columns]+=expected*expected;
    group_error[(i%columns)/64]+=delta*delta;group_norm[(i%columns)/64]+=expected*expected;
    if (top.size()<10 || errors[i]>errors[top.back()]) {
      top.push_back(i);
      std::sort(top.begin(),top.end(),[&](size_t a,size_t b){return errors[a]>errors[b] || (errors[a]==errors[b] && a<b);});
      if (top.size()>10) top.pop_back();
    }
  }
  json worst=json::array();
  for (auto i:top) {
    const auto bits=vt::LoadUnaligned<uint16_t>(reference.data()+2*i);
    const auto native_bits=vt::F32ToF16(actual[i]);
    worst.push_back({{"row",i/columns},{"channel",i%columns},{"native",actual[i]},
                    {"reference",vt::F16ToF32(bits)},{"abs_error",errors[i]},
                    {"fp16_ulp_distance",std::abs(ordered(native_bits)-ordered(bits))}});
  }
  report["exact_elements"]=exact;report["finite_elements"]=finite;report["top_errors"]=worst;
  report["fp16_max_ulp_distance"]=maximum_ulp;
  report["ulp_rule"]="ordered finite FP16 storage values; signed zeros have equal ordered distance";
  if (finite==actual.size()) {
    std::sort(errors.begin(),errors.end());
    report["absolute_error_quantiles"]={{"p50",errors[(errors.size()-1)/2]},
       {"p90",errors[size_t((errors.size()-1)*.9)]},{"p99",errors[size_t((errors.size()-1)*.99)]},
       {"max",errors.back()}};
  }
  report["relative_l2_by_visual_row"]=json::array();
  for (size_t i=0;i<row_error.size();++i)
    report["relative_l2_by_visual_row"].push_back(std::sqrt(row_error[i]/std::max(row_norm[i],1e-30)));
  report["relative_l2_by_64_channel_group"]=json::array();
  for (size_t i=0;i<group_error.size();++i)
    report["relative_l2_by_64_channel_group"].push_back(std::sqrt(group_error[i]/std::max(group_norm[i],1e-30)));
  return report;
}

// This utility is the only caller supplying frozen block inputs. The block
// itself is shared with normal typed inference, not reimplemented here.
void RunBlocks(const std::filesystem::path& model,const std::filesystem::path& reference,
               const std::string& selection,const std::filesystem::path& operators,json& report,
               const std::filesystem::path& stage_reference = {}, bool computed_only = false) {
  using namespace vllm;using namespace vllm::multimodal;
  std::vector<int64_t> layers;
  std::istringstream selected(selection);std::string token;
  while (std::getline(selected,token,',')) {
    Require(!token.empty() && token.find_first_not_of("0123456789")==std::string::npos,"invalid block selection");
    const auto block=std::stoll(token);
    Require(block>=0 && block<27 && std::find(layers.begin(),layers.end(),block)==layers.end(),"invalid/repeated block selection");
    layers.push_back(block);
  }
  Require(!layers.empty() && layers.size()<=5,"select one to five existing blocks");
  std::ifstream input(reference/"vision-boundaries-capture.json");const auto manifest=json::parse(input);
  Require(manifest.at("worker").at("encoder_calls")==2 && manifest.at("worker").at("captures").size()==382,
          "existing two-image all-block capture required");
  std::map<std::string,json> entries;
  for (const auto& entry:manifest.at("worker").at("captures")) {
    const auto name=entry.at("name").get<std::string>();
    if (name=="tower-input-0" && entries.contains(name)) break; // First captured image only.
    Require(entries.emplace(name,entry).second,"duplicate first-image boundary");
  }
  const auto raw=[&](const std::string& name) {return Boundary(reference,entries.at(name),"torch.float16");};
  const auto cfg=Qwen3_5DenseVisionConfig(LoadHfConfig((model/"config.json").string()));
  Require(cfg.depth==27 && cfg.hidden_size==1152 && cfg.head_dim()==72 && cfg.intermediate_size==4304,
          "qualified checkpoint tower required");
  auto& backend=vt::GetBackend(vt::DeviceType::kXPU);Queue queue;
  const auto index=LoadSafetensorsIndex((model/"model.safetensors.index.json").string());
  std::set<std::string> names;for (const auto& [key,file]:index) if (key.rfind("model.visual.",0)==0) names.insert(file);
  std::vector<SafetensorsFile> shards;for (const auto& name:names) shards.push_back(SafetensorsFile::Open((model/name).string()));
  auto host=LoadQwen3VLVisionWeights(shards,cfg);
  auto weights=PrepareVisionDeviceWeights(host,cfg,backend,vt::DType::kF16);host={};
  auto workspace=PrepareVisionWorkspace({1,24,32},cfg,backend,queue.value);
  Buffer pixels(queue.value.device,768,1536);const auto pixel_bytes=raw("tower-input-0");
  Require(pixel_bytes.size()==pixels.value.Bytes(),"wrong captured patch shape");
  backend.Copy(queue.value,pixels.value.data,pixel_bytes.data(),pixel_bytes.size());
  Qwen3VLVisionCapture composed;composed.selected_blocks=layers;
  report["reference_outputs_injected"]=!computed_only && (!operators.empty() || !stage_reference.empty());
  report["computed_only"]=computed_only;
  if (!computed_only && operators.empty() && stage_reference.empty()) {
    auto output=Qwen3VLVisionForwardDevice(pixels.value,weights,*workspace,backend,queue.value,&composed);
    report["early_boundaries"]={{"patch",DetailedCompare(composed.patch_embed_out,raw("patch-output"),1152)},
        {"position",DetailedCompare(composed.pos_embeds,raw("encoder-metadata-pos_embeds"),1152)}};
  } else {
    if (!operators.empty()) Require(layers==std::vector<int64_t>{0},"isolated operator mode requires only block0");
    else if (!stage_reference.empty()) Require(layers.size()==1 && (layers[0]==12 || layers[0]==26),"isolated captured block requires block12/26");
    report["composed_tower_executed"]=false;
  }
  Buffer block_input(queue.value.device,768,1152);
  report["blocks"]=json::array();
  for (auto block:layers) {
    const auto prefix="block"+std::to_string(block);
    const auto bytes=raw(prefix+"-input-0");Require(bytes.size()==block_input.value.Bytes(),"wrong captured block input shape");
    backend.Copy(queue.value,block_input.value.data,bytes.data(),bytes.size());
    Qwen3VLVisionCapture local;
    Qwen3VLVisionReplayBlockDevice(block_input.value,weights,*workspace,backend,queue.value,block,local);
    json block_report{{"block",block},{"reference_input_sha256",entries.at(prefix+"-input-0").at("sha256")},
        {"same_input_output",DetailedCompare(local.block_boundaries.at(block).at("output"),raw(prefix+"-output"),1152)},
        {"same_input_suboperations",json::object()}};
    if (!computed_only && operators.empty() && stage_reference.empty())
      block_report["composed_output"]=DetailedCompare(composed.block_outputs.at(block),raw(prefix+"-output"),1152);
    for (const auto& [stage,suffix]:std::vector<std::pair<std::string,std::string>>{
        {"input","-input-0"},{"norm1","-norm1-output"},{"projection","-attn-output"},
        {"residual1","-norm2-input-0"},{"norm2","-norm2-output"},{"fc2","-mlp-output"}}) {
      if (entries.contains(prefix+suffix)) block_report["same_input_suboperations"][stage]=
          DetailedCompare(local.block_boundaries.at(block).at(stage),raw(prefix+suffix),1152);
    }
    if (!stage_reference.empty()) {
      std::ifstream stream(stage_reference/"capture.json");const auto captured=json::parse(stream);
      Require(captured.at("status")=="CAPTURED" && captured.at("block")==block &&
                  captured.at("reference_input_sha256")==entries.at(prefix+"-input-0").at("sha256"),
              "wrong isolated reference block/input");
      const auto& stages=captured.at("stages");
      const std::map<std::string,int64_t> columns{{"norm1",1152},{"qkv",3456},{"q",1152},{"k",1152},
          {"v",1152},{"rotated_q",1152},{"rotated_k",1152},{"attention",1152},{"projection",1152},
          {"residual1",1152},{"norm2",1152},{"fc1",4304},{"gelu",4304},{"fc2",1152}};
      std::vector<std::unique_ptr<Buffer>> reference_buffers;
      std::map<std::string,vt::Tensor> reference_outputs;
      std::map<std::string,std::vector<unsigned char>> stage_bytes;
      for (const auto& [stage,width]:columns) {
        auto bytes=Boundary(stage_reference,stages.at(stage),"torch.float16");
        if (!computed_only) {
          auto buffer=std::make_unique<Buffer>(queue.value.device,768,width);
          Require(bytes.size()==buffer->value.Bytes(),"wrong captured operator geometry");
          backend.Copy(queue.value,buffer->value.data,bytes.data(),bytes.size());
          reference_outputs.emplace(stage,buffer->value);reference_buffers.push_back(std::move(buffer));
        }
        stage_bytes.emplace(stage,std::move(bytes));
      }
      Qwen3VLVisionCapture isolated;
      if (!computed_only) Qwen3VLVisionReplayBlockDevice(block_input.value,weights,*workspace,backend,queue.value,block,isolated,&reference_outputs);
      block_report["exact_upstream_stage_replay"]=json::object();
      block_report["local_stage_chain"]=json::object();
      for (const auto& [stage,width]:columns) {
        if (!computed_only) block_report["exact_upstream_stage_replay"][stage]=
            DetailedCompare(isolated.block_boundaries.at(block).at(stage),stage_bytes.at(stage),width);
        block_report["local_stage_chain"][stage]=
            DetailedCompare(local.block_boundaries.at(block).at(stage),stage_bytes.at(stage),width);
      }
      const auto expected=Boundary(stage_reference,stages.at("output"),"torch.float16");
      if (!computed_only) {
        block_report["exact_upstream_final_add"]=DetailedCompare(isolated.block_boundaries.at(block).at("output"),expected,1152);
        // Minimal causal replay: retain every native operator/activation except
        // the two already attributed norm boundaries. This is diagnostic only.
        std::map<std::string,vt::Tensor> norm_outputs{{"norm1",reference_outputs.at("norm1")},
                                                    {"norm2",reference_outputs.at("norm2")}};
        Qwen3VLVisionCapture norm_isolated;
        Qwen3VLVisionReplayBlockDevice(block_input.value,weights,*workspace,backend,queue.value,block,norm_isolated,&norm_outputs);
        block_report["only_norm_references_output"]=DetailedCompare(
            norm_isolated.block_boundaries.at(block).at("output"),expected,1152);
      }
      block_report["reference_block_matches_original"]=captured.at("matches_original_block_output");
      block_report["operator_reference_scope"]="actual installed pinned block on exact original input, all internal stages observed; diagnostic capture, frozen original unchanged";
    }
    if (block==0 && !operators.empty()) {
      const auto document=[&](const char* name) {std::ifstream stream(operators/name);return json::parse(stream);};
      const auto qkv=document("qkv-real-replays.json").at("cases").at(0).at("files");
      const auto attention=document("attention-real-replays.json").at("cases").at(0).at("files");
      const auto operator_raw=[&](const json& entry,int64_t columns) {
        const auto filename=entry.at("file").get<std::string>();
        Require(std::filesystem::path(filename).filename()==filename,"invalid operator reference filename");
        const auto bytes=Read(operators/filename,16*1024*1024);
        Require(bytes.size()==size_t(768*columns*2),"wrong operator reference size");return bytes;
      };
      std::map<std::string,std::pair<int64_t,std::vector<unsigned char>>> stages{
        {"norm1",{1152,raw("block0-norm1-output")}},
        {"qkv",{3456,operator_raw(qkv.at("merged"),3456)}},
        {"q",{1152,operator_raw(qkv.at("q"),1152)}},
        {"k",{1152,operator_raw(qkv.at("k"),1152)}},
        {"v",{1152,operator_raw(qkv.at("v"),1152)}},
        {"rotated_q",{1152,operator_raw(attention.at("q"),1152)}},
        {"rotated_k",{1152,operator_raw(attention.at("k"),1152)}},
        {"attention",{1152,operator_raw(attention.at("expected"),1152)}},
        {"projection",{1152,raw("block0-attn-output")}},
        {"residual1",{1152,raw("block0-norm2-input-0")}},
        {"norm2",{1152,raw("block0-norm2-output")}},
        {"fc2",{1152,raw("block0-mlp-output")}}};
      if (computed_only) {
        for (const auto& [stage,value]:stages)
          block_report["local_stage_chain"][stage]=DetailedCompare(local.block_boundaries.at(0).at(stage),value.second,value.first);
      } else {
        std::vector<std::unique_ptr<Buffer>> reference_buffers;
        std::map<std::string,vt::Tensor> reference_outputs;
        for (const auto& [stage,value]:stages) {
          auto buffer=std::make_unique<Buffer>(queue.value.device,768,value.first);
          backend.Copy(queue.value,buffer->value.data,value.second.data(),value.second.size());
          reference_outputs.emplace(stage,buffer->value);reference_buffers.push_back(std::move(buffer));
        }
        Qwen3VLVisionCapture isolated;
        Qwen3VLVisionReplayBlockDevice(block_input.value,weights,*workspace,backend,queue.value,0,isolated,&reference_outputs);
        block_report["exact_upstream_stage_replay"]=json::object();
        for (const auto& [stage,value]:stages) {
          auto metric=DetailedCompare(isolated.block_boundaries.at(0).at(stage),value.second,value.first);
          metric["input_contract"]=stage=="fc2" ? "exact reference norm2 input; native FC1/GELU/FC2 group" :
              "computed native stage after preceding reference outputs replace scratch";
          block_report["exact_upstream_stage_replay"][stage]=metric;
        }
        block_report["exact_upstream_final_add"]=DetailedCompare(isolated.block_boundaries.at(0).at("output"),raw("block0-output"),1152);
      }
      block_report["operator_reference_scope"]="QKV/RoPE/attention are existing standalone pinned executions on worker-derived inputs; projection/residual/norm/MLP are actual worker boundaries. Not a new full-worker suboperation capture.";
    }
    report["blocks"].push_back(block_report);
  }
  if (!computed_only && operators.empty() && stage_reference.empty()) report["merger_composed"]=DetailedCompare(composed.merger_out,raw("merger-output"),5120);
  report["reference_tier_hits"]=vt::GetReferenceTierHits();
  Require(vt::GetReferenceTierHits()==0,"reference provider fallback executed");
  report["status"]="DIAGNOSTIC";
}

void RunMerger(const std::filesystem::path& model,const std::filesystem::path& reference,json& report) {
  using namespace vllm;using namespace vllm::multimodal;
  std::ifstream input(reference/"vision-boundaries-capture.json");const auto manifest=json::parse(input);
  Require(manifest.at("worker").at("encoder_calls")==2 && manifest.at("worker").at("captures").size()==382,
          "existing two-image all-block capture required");
  std::map<std::string,json> entries;
  for (const auto& entry:manifest.at("worker").at("captures")) {
    const auto name=entry.at("name").get<std::string>();
    if (name=="tower-input-0" && entries.contains(name)) break;
    Require(entries.emplace(name,entry).second,"duplicate first-image boundary");
  }
  const auto raw=[&](const char* name) {return Boundary(reference,entries.at(name),"torch.float16");};
  const auto cfg=Qwen3_5DenseVisionConfig(LoadHfConfig((model/"config.json").string()));
  Require(cfg.depth==27 && cfg.hidden_size==1152 && cfg.out_hidden_size==5120 && cfg.merge_unit()==4,
          "qualified checkpoint merger required");
  const auto input_bytes=raw("merger-input-0");
  const std::map<std::string,std::tuple<int64_t,int64_t,std::vector<unsigned char>>> stages{
      {"norm",{768,1152,raw("merger-norm-output")}},
      {"fc1",{192,4608,raw("merger-fc1-output-0")}},
      {"gelu",{192,4608,raw("merger-fc2-input-0")}},
      {"output",{192,5120,raw("merger-output")}}};
  auto& backend=vt::GetBackend(vt::DeviceType::kXPU);Queue queue;
  const auto index=LoadSafetensorsIndex((model/"model.safetensors.index.json").string());
  std::set<std::string> names;for (const auto& [key,file]:index) if (key.rfind("model.visual.",0)==0) names.insert(file);
  std::vector<SafetensorsFile> shards;for (const auto& name:names) shards.push_back(SafetensorsFile::Open((model/name).string()));
  auto host=LoadQwen3VLVisionWeights(shards,cfg);
  auto weights=PrepareVisionDeviceWeights(host,cfg,backend,vt::DType::kF16);host={};
  auto workspace=PrepareVisionWorkspace({1,24,32},cfg,backend,queue.value);
  Buffer merger_input(queue.value.device,768,1152);
  Require(input_bytes.size()==merger_input.value.Bytes(),"wrong captured merger input shape");
  backend.Copy(queue.value,merger_input.value.data,input_bytes.data(),input_bytes.size());
  Qwen3VLVisionCapture local,repeat,isolated;
  Qwen3VLVisionReplayMergerDevice(merger_input.value,weights,*workspace,backend,queue.value,local);
  Qwen3VLVisionReplayMergerDevice(merger_input.value,weights,*workspace,backend,queue.value,repeat);
  report["same_input_repeat_storage_exact"]=local.merger_norm_out==repeat.merger_norm_out &&
      local.merger_fc1_out==repeat.merger_fc1_out && local.merger_gelu_out==repeat.merger_gelu_out &&
      local.merger_out==repeat.merger_out;
  std::vector<std::unique_ptr<Buffer>> reference_buffers;
  std::map<std::string,vt::Tensor> reference_outputs;
  for (const auto& [stage,value]:stages) if (stage!="output") {
    const auto& [rows,columns,bytes]=value;
    auto buffer=std::make_unique<Buffer>(queue.value.device,rows,columns);
    Require(bytes.size()==buffer->value.Bytes(),"wrong merger stage geometry");
    backend.Copy(queue.value,buffer->value.data,bytes.data(),bytes.size());
    reference_outputs.emplace(stage,buffer->value);reference_buffers.push_back(std::move(buffer));
  }
  Qwen3VLVisionReplayMergerDevice(merger_input.value,weights,*workspace,backend,queue.value,isolated,&reference_outputs);
  report["same_input_merger"]=json::object();report["exact_upstream_stage_replay"]=json::object();
  const auto captures=[](const Qwen3VLVisionCapture& cap) {
    return std::map<std::string,const std::vector<float>*>{{"norm",&cap.merger_norm_out},
        {"fc1",&cap.merger_fc1_out},{"gelu",&cap.merger_gelu_out},{"output",&cap.merger_out}};
  };
  const auto local_stages=captures(local),isolated_stages=captures(isolated);
  for (const auto& [stage,value]:stages) {
    const auto& [rows,columns,bytes]=value;
    report["same_input_merger"][stage]=DetailedCompare(*local_stages.at(stage),bytes,columns);
    auto metric=DetailedCompare(*isolated_stages.at(stage),bytes,columns);
    metric["input_contract"]="exact reference input of this individual operator; native capture precedes scratch replacement";
    report["exact_upstream_stage_replay"][stage]=metric;
  }
  report["reference_input_sha256"]=entries.at("merger-input-0").at("sha256");
  report["composed_tower_executed"]=false;
  report["reference_tier_hits"]=vt::GetReferenceTierHits();
  Require(vt::GetReferenceTierHits()==0,"reference provider fallback executed");
  report["status"]="DIAGNOSTIC";
}

void Run(const std::filesystem::path& model, const std::filesystem::path& fixtures,
         const std::filesystem::path& reference, const std::string& selected_case, json& report) {
  using namespace vllm;
  using namespace vllm::multimodal;
  using namespace vllm::entrypoints::openai;
  std::ifstream input(reference / "manifest.json");
  const auto manifest = json::parse(input);
  Require(manifest.at("status") == "CAPTURED" && manifest.at("cases").size() == 4,
          "complete actual four-geometry capture required");
  const std::vector<std::string> order{"orbit", "comet-unaligned", "portrait", "maximum"};
  Require(selected_case.empty() || std::find(order.begin(), order.end(), selected_case) != order.end(),
          "unknown frozen geometry selection");
  for (size_t i = 0; i < order.size(); ++i)
    Require(manifest.at("cases").at(i).at("name") == order[i], "wrong frozen geometry order");
  const auto hf = LoadHfConfig((model / "config.json").string());
  const auto tower = Qwen3_5DenseVisionConfig(hf);
  Require(tower.hidden_size == 1152 && tower.depth == 27 && tower.num_heads == 16 &&
          tower.head_dim() == 72 && tower.out_hidden_size == 5120 &&
          tower.deepstack_visual_indexes.empty(), "qualified Qwen3.5 checkpoint required");
  auto config = LoadQwen3VLProcessorConfig((model / "preprocessor_config.json").string(),
                                         (model / "config.json").string(), "native-compare");
  config.max_pixels = std::min(config.max_pixels, kNativeQwen3_5MaxImagePixels);
  config.torchvision_bicubic_resize = true;
  config.pixel_dtype = ImagePixelDType::kF16;
  config.retain_pixel_values_f32 = false;
  const Qwen3VLImageProcessor processor(config);
  const auto codec = DefaultImageCodec();
  Require(vt::xpu::DeviceCount() > 0, "native XPU required");
  auto& backend = vt::GetBackend(vt::DeviceType::kXPU);
  Queue queue;
  const auto index = LoadSafetensorsIndex((model / "model.safetensors.index.json").string());
  std::set<std::string> names;
  for (const auto& [key, file] : index) if (key.rfind("model.visual.", 0) == 0) names.insert(file);
  std::vector<SafetensorsFile> shards;
  for (const auto& name : names) shards.push_back(SafetensorsFile::Open((model / name).string()));
  auto host = LoadQwen3VLVisionWeights(shards, tower);
  auto weights = PrepareVisionDeviceWeights(host, tower, backend, vt::DType::kF16);
  host = {};
  bool passed = true;
  for (size_t i = 0; i < order.size(); ++i) {
    if (!selected_case.empty() && order[i] != selected_case) continue;
    const auto& entry = manifest.at("cases").at(i);
    Require(entry.at("name") == order[i], "wrong frozen geometry order");
    const auto path = entry.at("image").get<std::string>();
    Require(path == "native_vision_http/orbit.png" || path == "native_vision_http/comet-unaligned.png" ||
            path == "native_vision_qualification/field-day-portrait.png" ||
            path == "native_vision_http/orbit-max-2048.png", "wrong frozen fixture path");
    const auto bytes = Read(fixtures / path, kMaxImageContainerBytes);
    const auto decoded = codec(DecodedMedia{"image/png", {bytes.begin(), bytes.end()}});
    const auto patches = processor.ProcessImage(decoded.rgb.data(), decoded.height, decoded.width);
    const auto& boundaries = entry.at("boundaries");
    const auto pixels = Boundary(reference, boundaries.at("pixels"), "torch.float16");
    const auto grid = Boundary(reference, boundaries.at("grid"), "torch.int64");
    Require(grid.size() == 24, "one-image grid required");
    bool grid_equal = true;
    for (int axis = 0; axis < 3; ++axis)
      grid_equal &= patches.image_grid_thw[axis] == vt::LoadUnaligned<int64_t>(grid.data() + axis * 8);
    const bool input_exact = pixels.size() == patches.pixel_values_f16.size() * 2 &&
        std::memcmp(pixels.data(), patches.pixel_values_f16.data(), pixels.size()) == 0;
    json case_report{{"name", order[i]}, {"image_sha256", entry.at("image_sha256")},
                     {"grid_thw", patches.image_grid_thw}, {"pixels_storage_exact", input_exact},
                     {"grid_exact", grid_equal}, {"boundaries", json::object()}};
    // Compare same reference input even if host preprocessing itself fails.
    Require(pixels.size() == static_cast<size_t>(patches.num_patches * patches.patch_feature_dim) * 2,
            "processor/reference input geometry differs");
    auto workspace = PrepareVisionWorkspace(patches.image_grid_thw, tower, backend, queue.value);
    Buffer device_pixels(queue.value.device, patches.num_patches, patches.patch_feature_dim);
    backend.Copy(queue.value, device_pixels.value.data, pixels.data(), pixels.size());
    Qwen3VLVisionCapture capture;
    auto output = Qwen3VLVisionForwardDevice(device_pixels.value, weights, *workspace, backend, queue.value, &capture);
    for (const auto& [name, values, rel, abs] :
         std::vector<std::tuple<std::string, const std::vector<float>*, double, double>>{
          {"patch", &capture.patch_embed_out, 3e-4, .02}, {"position", &capture.pos_embeds, 0, 0},
          {"cos", &capture.rotary_cos, 0, 0}, {"sin", &capture.rotary_sin, 0, 0},
          {"block0", &capture.block0_out, 1e-3, .05}, {"merger", &capture.merger_out, 3e-3, .05}}) {
      case_report["boundaries"][name] = Compare(*values, Boundary(reference, boundaries.at(name), "torch.float16"), rel, abs);
      passed &= case_report["boundaries"][name]["pass"].get<bool>();
    }
    passed &= input_exact && grid_equal;
    backend.Synchronize(queue.value);
    const auto memory = vt::xpu::GetMemoryInfo();
    case_report["backend_live_bytes"] = memory.allocated_bytes;
    case_report["backend_peak_bytes"] = memory.peak_allocated_bytes;
    report["cases"].push_back(case_report);
    std::cout << "NATIVE_TOWER_COMPARISON " << case_report.dump() << '\n';
  }
  report["reference_tier_hits"] = vt::GetReferenceTierHits();
  Require(vt::GetReferenceTierHits() == 0, "reference provider fallback executed");
  report["status"] = passed ? "PASS" : "FAIL";
}
}  // namespace

int main(int argc, char** argv) {
  if (argc==5 && std::string(argv[1])=="--merger") {
    if (std::filesystem::exists(argv[4])) {std::cerr<<"output exists; preserve evidence\n";return 2;}
    json report{{"status","FAIL"},{"scope","first-image exact-reference-input merger and isolated suboperations; descriptive metrics, no gate waiver"}};
    try {RunMerger(argv[2],argv[3],report);}
    catch (const std::exception& error) {report["error"]=error.what();}
    std::ofstream output(argv[4]);if (!output.good()) return 2;
    output<<report.dump(2)<<'\n';return report["status"]=="DIAGNOSTIC" ? 0 : 1;
  }
  const bool computed_only = argc>2 && std::string(argv[1])=="--blocks" && std::string(argv[argc-1])=="--computed-only";
  if (computed_only) --argc;
  if ((argc == 6 || argc == 7 || (argc==8 && std::string(argv[6])=="--stage-reference")) && std::string(argv[1])=="--blocks") {
    if (std::filesystem::exists(argv[4])) {std::cerr<<"output exists; preserve evidence\n";return 2;}
    json report{{"status","FAIL"},{"scope","selected first-image composed and exact-reference-input block replay; descriptive metrics, no gate waiver"}};
    try {RunBlocks(argv[2],argv[3],argv[5],argc==7 ? argv[6] : "",report,argc==8 ? argv[7] : "",computed_only);}
    catch (const std::exception& error) {report["error"]=error.what();}
    std::ofstream output(argv[4]);if (!output.good()) return 2;
    output<<report.dump(2)<<'\n';return report["status"]=="DIAGNOSTIC" ? 0 : 1;
  }
  if (argc != 5 && argc != 6) {
    std::cerr << "usage: native-vision-compare MODEL_DIR FIXTURE_ROOT REFERENCE_CAPTURE_DIR OUTPUT_JSON [CASE]\n";
    return 2;
  }
  if (std::filesystem::exists(argv[4])) {
    std::cerr << "output exists; preserve evidence\n"; return 2;
  }
  json report{{"status", "FAIL"}, {"cases", json::array()},
              {"scope", "four same-input actual tower boundaries; fixed predeclared FP16 gates; not serving performance"},
              {"reference_hash_validation", "supervisor must verify manifest/fixture/boundary SHA-256 before execution"}};
  const std::string selected_case = argc == 6 ? argv[5] : "";
  if (!selected_case.empty()) {
    report["selected_case"] = selected_case;
    report["scope"] = "one selected same-input tower geometry; fixed predeclared FP16 gates; not full geometry qualification";
  }
  try { Run(argv[1], argv[2], argv[3], selected_case, report); }
  catch (const std::exception& error) { report["status"] = "FAIL"; report["error"] = error.what(); }
  std::ofstream output(argv[4]);
  if (!output.good()) { std::cerr << "cannot write output\n"; return 2; }
  output << report.dump(2) << '\n';
  return report["status"] == "PASS" ? 0 : 1;
}
