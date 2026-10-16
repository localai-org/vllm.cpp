#include "xpu_test_helpers.h"
#include "support/native_vision_external_artifacts.h"
#include "vllm/model_executor/models/qwen3_vl.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_vl_vision_attention.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vt/unaligned.h"
#include "vt/xpu.h"

#include <cstdlib>
#include <bit>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <set>
#include <nlohmann/json.hpp>

namespace {
using namespace vllm::multimodal;
using xpu_test::Buffer;
using vt::DType;
using json=nlohmann::json;
Qwen3VLVisionConfig Config(bool real) {
  Qwen3VLVisionConfig c;
  c.hidden_size=1152;c.num_heads=16;c.depth=real?27:1;
  c.intermediate_size=real?4304:8;c.out_hidden_size=real?5120:4;
  c.patch_size=real?16:1;c.temporal_patch_size=real?2:1;c.in_channels=real?3:1;
  c.num_position_embeddings=real?2304:4;c.deepstack_visual_indexes.clear();
  return c;
}
std::vector<unsigned char> Read(const std::string& path) {
  std::ifstream f(path,std::ios::binary);REQUIRE_MESSAGE(f.good(),path);
  return {std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>()};
}
std::vector<unsigned char> Download(vt::Backend& b,vt::Queue& q,const Qwen3VLVisionDeviceOutput& out) {
  out.WaitOn(q);
  std::vector<unsigned char> bytes(out.tensor().Bytes());
  b.Copy(q,bytes.data(),out.tensor().data,bytes.size());out.RecordUse(q);b.Synchronize(q);
  return bytes;
}
void Compare(const std::vector<float>& actual,const std::vector<unsigned char>& raw,
             double relative_limit,double absolute_limit,const std::string& label) {
  REQUIRE(actual.size()*2==raw.size());
  double error=0,norm=0,absolute=0;bool finite=true;
  for (size_t i=0;i<actual.size();++i) {
    const float ref=vt::F16ToF32(vt::LoadUnaligned<uint16_t>(raw.data()+i*2));
    const double d=static_cast<double>(actual[i])-ref;
    finite &= std::isfinite(actual[i]) && std::isfinite(ref);
    error+=d*d;norm+=static_cast<double>(ref)*ref;absolute=std::max(absolute,std::abs(d));
  }
  const double relative=std::sqrt(error/std::max(norm,1e-30));
  INFO(label);
  CAPTURE(relative);
  CAPTURE(absolute);
  CHECK(finite);CHECK(relative<=relative_limit);CHECK(absolute<=absolute_limit);
  if (relative_limit==0 && absolute_limit==0) {
    std::vector<uint16_t> bits(actual.size());
    for (size_t i=0;i<bits.size();++i) bits[i]=vt::F32ToF16(actual[i]);
    const auto* begin=reinterpret_cast<const unsigned char*>(bits.data());
    xpu_test::SameBytes({begin,begin+bits.size()*2},raw);
  }
  std::cout<<"VISION_TOWER "<<label<<" rel_l2="<<relative<<" max_abs="<<absolute<<'\n';
  if (const char* env=std::getenv("VT_B70_VISION_OPERATOR_DUMP_DIR")) {
    std::vector<uint16_t> bits(actual.size());
    for (size_t i=0;i<bits.size();++i) bits[i]=vt::F32ToF16(actual[i]);
    std::ofstream file(std::string(env)+"/tower-native-"+label+".float16",std::ios::binary);
    REQUIRE(file.good());file.write(reinterpret_cast<const char*>(bits.data()),bits.size()*2);
  }
}
Qwen3VLVisionWeights ConstantOutputWeights(const Qwen3VLVisionConfig& c) {
  Qwen3VLVisionWeights w;
  const int64_t H=c.hidden_size,I=c.intermediate_size,ctx=H*c.merge_unit();
  w.patch_proj_w.resize(H);w.patch_proj_b.resize(H);w.pos_embed_w.resize(4*H);
  VisionBlockWeights block;
  block.norm1_w.assign(H,vt::F32ToBF16(1));block.norm1_b.resize(H);
  block.norm2_w=block.norm1_w;block.norm2_b.resize(H);
  block.qkv_w.resize(3*H*H);block.qkv_b.resize(3*H);
  block.proj_w.resize(H*H);block.proj_b.resize(H);
  block.fc1_w.resize(I*H);block.fc1_b.resize(I);
  block.fc2_w.resize(H*I);block.fc2_b.resize(H);w.blocks.push_back(std::move(block));
  w.merger.norm_w.assign(H,vt::F32ToBF16(1));w.merger.norm_b.resize(H);
  w.merger.fc1_w.resize(ctx*ctx);w.merger.fc1_b.resize(ctx);
  w.merger.fc2_w.resize(c.out_hidden_size*ctx);
  w.merger.fc2_b={vt::F32ToBF16(0.25f),vt::F32ToBF16(-0.5f),vt::F32ToBF16(1),0};
  return w;
}
}

TEST_CASE("XPU typed vision attention: model-free reference scalar contract") {
  std::ifstream input(std::string(XPU_VISION_FIXTURE_DIR) + "/attention.json");
  REQUIRE(input.good());
  json reference; input >> reference;
  REQUIRE(reference.at("cases").size() == 3);
  const auto args = TypedVisionAttentionArgs(72);
  CHECK_FALSE(args.causal);
  for (const auto& entry : reference.at("cases")) {
    REQUIRE(entry.at("shape")[2] == 72);
    const float pinned = entry.at("scale").get<float>();
    CHECK(std::bit_cast<uint32_t>(pinned) == 0x3df15befu);
    CHECK(std::bit_cast<uint32_t>(args.scale) == std::bit_cast<uint32_t>(pinned));
  }
  CHECK_THROWS_AS(TypedVisionAttentionArgs(0), std::invalid_argument);
  CHECK_THROWS_AS(TypedVisionAttentionArgs(-1), std::invalid_argument);
  std::cout << "VISION_SCALAR_CONTRACT head_dim=72 scale=" << std::setprecision(17)
            << args.scale << " bits=0x" << std::hex << std::bit_cast<uint32_t>(args.scale)
            << std::dec << '\n';
}

TEST_CASE("XPU registered vision: encoder output and GPU merge retain live parents") {
  auto& b=vt::GetBackend(vt::DeviceType::kXPU);
  const auto baseline=vt::xpu::GetMemoryInfo();
  const auto baseline_pool=vllm::Pool(b).stats();
  std::shared_ptr<vllm::MmEncoderLifetime> deliberately_retained;
  {
    xpu_test::Queue encode(vt::DeviceType::kXPU),consumer(vt::DeviceType::kXPU);
    auto cfg=Config(false);
    vllm::HfConfig text; text.hidden_size=4; text.vocab_size=16;
    text.raw["quantization_config"]={{"quant_method","exl3"}};
    vllm::Qwen3_5DenseWeights weights;
    weights.exl3_checkpoint=true; weights.precision.activation=DType::kF16;
    weights.has_visual=true; weights.visual_cfg=cfg; weights.visual=ConstantOutputWeights(cfg);
    weights.embed_tokens.dtype=DType::kF16; weights.embed_tokens.rank=2;
    weights.embed_tokens.shape[0]=16; weights.embed_tokens.shape[1]=4;
    weights.embed_tokens.bytes.resize(16*4*2);
    auto model=vllm::MakeQwen3_5DenseLoadedModel(std::move(weights));
    REQUIRE(vllm::ModelRegistry::SupportsMmInputs(*model));
    vllm::multimodal::MultiModalFeatureSpec image;
    image.offset=1; image.length=1; image.data=std::make_shared<vllm::multimodal::ImageKwargs>();
    image.data->image_grid_thw={1,2,2}; image.data->num_patches=4;
    image.data->patch_feature_dim=1; image.data->pixel_dtype=vllm::multimodal::ImagePixelDType::kF16;
    image.data->pixel_values_f16={vt::F32ToF16(1),vt::F32ToF16(2),vt::F32ToF16(3),vt::F32ToF16(4)};
    auto wrong=image; wrong.length=2;
    CHECK_THROWS_AS(vllm::ModelRegistry::EncodeMm(*model,text,encode.q,wrong),std::runtime_error);
    auto first=vllm::ModelRegistry::EncodeMm(*model,text,encode.q,image);
    auto second=vllm::ModelRegistry::EncodeMm(*model,text,encode.q,image);
    REQUIRE(first.lifetime); REQUIRE(second.lifetime);
    CHECK(first.embeds.data!=second.embeds.data);
    REQUIRE(first.embeds.dtype==DType::kF16); REQUIRE(first.embeds.shape[0]==1);
    const std::vector<int32_t> ids{3,0,4},positions{0,1,2,0,1,2,0,1,2};
    const std::vector<char> mask{0,1,0}; const std::vector<vt::Tensor> slices{first.embeds};
    std::vector<std::shared_ptr<vllm::MmEncoderLifetime>> owners{first.lifetime};
    vllm::MmEmbedInputs inputs;
    inputs.token_ids=&ids; inputs.mm_embeds=&slices; inputs.is_mm_embed=&mask;
    inputs.mrope_positions=&positions;
    CHECK_THROWS_WITH_AS(vllm::ModelRegistry::EmbedMm(*model,text,consumer.q,inputs),
                        doctest::Contains("live source owners required"),std::runtime_error);
    inputs.mm_lifetimes=&owners;
    auto merged=vllm::ModelRegistry::EmbedMm(*model,text,consumer.q,inputs);
    // Drop cache, input-feature and model owners before reading the queued merge.
    std::weak_ptr<vllm::MmEncoderLifetime> weak=first.lifetime;
    std::weak_ptr<vllm::MmEncoderLifetime> second_weak=second.lifetime;
    std::weak_ptr<vllm::multimodal::ImageKwargs> request_input=image.data;
    if (const char* retain=std::getenv("VT_B70_VISION_RETAIN_OWNER"); retain && retain[0]=='1')
      deliberately_retained=first.lifetime; // Negative qualification test only.
    first={}; owners.clear(); image.data.reset(); wrong.data.reset(); model.reset();
    CHECK(request_input.expired());
    CHECK_FALSE(weak.expired());
    std::vector<uint16_t> actual(12),expected(12);
    expected[4]=vt::F32ToF16(.25f); expected[5]=vt::F32ToF16(-.5f);
    expected[6]=vt::F32ToF16(1);
    b.Copy(consumer.q,actual.data(),merged.mm.inputs_embeds.data,actual.size()*2); b.Synchronize(consumer.q);
    CHECK(actual==expected);
    second.lifetime->WaitOn(consumer.q);
    std::vector<uint16_t> other(4);
    b.Copy(consumer.q,other.data(),second.embeds.data,8);
    second.lifetime->RecordUse(consumer.q); b.Synchronize(consumer.q);
    CHECK(other==std::vector<uint16_t>(expected.begin()+4,expected.begin()+8));
    second={}; CHECK(second_weak.expired());
    merged.storage.clear(); CHECK(weak.expired());
  }
  const auto before_drain=vt::xpu::GetMemoryInfo();
  CHECK(before_drain.graph_count==baseline.graph_count);
  const auto pool=vllm::Pool(b).stats();
  CHECK(pool.live_blocks==baseline_pool.live_blocks);
  // Diagnostic teardown only, after model, consumers and queues were destroyed.
  const auto released=vllm::Pool(b).Drain(b);
  const auto final=vt::xpu::GetMemoryInfo();
  CHECK(final.allocated_bytes==baseline.allocated_bytes-baseline_pool.retained_bytes);
  CHECK(final.graph_count==baseline.graph_count);
  std::cout<<"VISION_OWNER_TEARDOWN baseline="<<baseline.allocated_bytes
           <<" before_pool_drain="<<before_drain.allocated_bytes
           <<" released_pool="<<released<<" final="<<final.allocated_bytes<<'\n';
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU registered vision: request M-RoPE matches executed Python positions") {
  std::ifstream file(std::string(XPU_VISION_FIXTURE_DIR)+"/request-mrope/request-mrope.json");
  REQUIRE(file.good()); json reference; file>>reference;
  vllm::HfConfig cfg; cfg.hidden_size=4;
  cfg.raw["quantization_config"]={{"quant_method","exl3"}};
  vllm::Qwen3_5DenseWeights weights;
  weights.exl3_checkpoint=true; weights.precision.activation=DType::kF16;
  weights.has_visual=true; weights.visual_cfg=Config(false);
  auto model=vllm::MakeQwen3_5DenseLoadedModel(std::move(weights));
  for (const auto& c : reference.at("cases")) {
    const auto tokens=c.at("tokens").get<std::vector<int32_t>>();
    std::vector<vllm::multimodal::MultiModalFeatureSpec> features;
    for (const auto& span : c.at("spans")) {
      vllm::multimodal::MultiModalFeatureSpec item;
      item.offset=span.at("offset"); item.length=span.at("length");
      item.data=std::make_shared<vllm::multimodal::ImageKwargs>();
      item.data->image_grid_thw=span.at("grid").get<std::array<int64_t,3>>();
      features.push_back(std::move(item));
    }
    const auto actual=vllm::ModelRegistry::MropePromptPositionsFor(*model,cfg,tokens,features);
    const auto rows=c.at("positions").get<std::vector<std::vector<int32_t>>>();
    std::vector<int32_t> expected;
    for (const auto& row : rows) expected.insert(expected.end(),row.begin(),row.end());
    CHECK(actual.positions==expected); CHECK(actual.delta==c.at("delta").get<int64_t>());
  }
  cfg.architectures={"Qwen3_5ForCausalLM"};
  vllm::Qwen3_5DenseWeights text;
  auto text_model=vllm::MakeQwen3_5DenseLoadedModel(std::move(text),cfg);
  CHECK_FALSE(vllm::ModelRegistry::SupportsMmInputs(*text_model));
  CHECK_FALSE(vllm::ModelRegistry::UsesMrope(*text_model));
}

TEST_CASE("XPU typed vision tower: independent outputs survive scratch reuse and consumers") {
  auto& b=vt::GetBackend(vt::DeviceType::kXPU);
  const auto baseline=vt::xpu::GetMemoryInfo().allocated_bytes;
  {
    xpu_test::Queue encode(vt::DeviceType::kXPU),consumer(vt::DeviceType::kXPU);
    const auto cfg=Config(false);auto host=ConstantOutputWeights(cfg);
    auto weights=PrepareVisionDeviceWeights(host,cfg,b,DType::kF16);
    auto ws=PrepareVisionWorkspace({1,2,2},cfg,b,encode.q);
    Buffer pixels(encode.q,DType::kF16,{4,1});pixels.put({1,2,3,4});
    CHECK_THROWS_AS(Qwen3VLVisionForwardDevice(pixels.tensor,nullptr,*ws,b,encode.q),std::runtime_error);
    CHECK_THROWS_AS(Qwen3VLVisionForwardDevice(pixels.tensor,weights,*ws,b,consumer.q),std::runtime_error);
    auto invalid_pixels=pixels.tensor;invalid_pixels.dtype=DType::kBF16;
    CHECK_THROWS_AS(Qwen3VLVisionForwardDevice(invalid_pixels,weights,*ws,b,encode.q),std::runtime_error);
    CHECK_THROWS_AS(PrepareVisionWorkspace({2,2,2},cfg,b,encode.q),std::runtime_error);
    CHECK_THROWS_AS(PrepareVisionWorkspace({1,128,130},cfg,b,encode.q),std::runtime_error);
    auto invalid_cfg=cfg;invalid_cfg.deepstack_visual_indexes={0};
    CHECK_THROWS_AS(PrepareVisionWorkspace({1,2,2},invalid_cfg,b,encode.q),std::runtime_error);
    auto first=Qwen3VLVisionForwardDevice(pixels.tensor,weights,*ws,b,encode.q);
    auto second=Qwen3VLVisionForwardDevice(pixels.tensor,weights,*ws,b,encode.q);
    Qwen3VLVisionCapture selected;selected.selected_blocks={0};
    auto diagnostic=Qwen3VLVisionForwardDevice(pixels.tensor,weights,*ws,b,encode.q,&selected);
    REQUIRE(selected.block_outputs.size()==1);
    REQUIRE(selected.block_boundaries.size()==1);
    CHECK(selected.block_boundaries.at(0).at("output")==selected.block_outputs[0]);
    CHECK(selected.block_boundaries.at(0).at("input")==selected.block_outputs[0]);
    selected.selected_blocks={1};
    CHECK_THROWS_AS(Qwen3VLVisionForwardDevice(pixels.tensor,weights,*ws,b,encode.q,&selected),std::runtime_error);
    Buffer block_input(encode.q,DType::kF16,{4,1152});
    const std::vector<float> replay_values(4*1152,0.5f);block_input.put(replay_values);
    Qwen3VLVisionCapture local;
    CHECK_THROWS_AS(Qwen3VLVisionReplayBlockDevice(block_input.tensor,weights,*ws,b,consumer.q,0,local),std::runtime_error);
    CHECK_THROWS_AS(Qwen3VLVisionReplayBlockDevice(block_input.tensor,weights,*ws,b,encode.q,-1,local),std::runtime_error);
    auto wrong_input=block_input.tensor;wrong_input.dtype=DType::kBF16;
    CHECK_THROWS_AS(Qwen3VLVisionReplayBlockDevice(wrong_input,weights,*ws,b,encode.q,0,local),std::runtime_error);
    Qwen3VLVisionReplayBlockDevice(block_input.tensor,weights,*ws,b,encode.q,0,local);
    CHECK(local.block_boundaries.at(0).at("input")==replay_values);
    CHECK(local.block_boundaries.at(0).at("output")==replay_values);
    CHECK(block_input.floats()==replay_values);
    Buffer reference_projection(encode.q,DType::kF16,{4,1152});
    reference_projection.put(std::vector<float>(4*1152,0.25f));
    std::map<std::string,vt::Tensor> reference_outputs{{"projection",reference_projection.tensor}};
    Qwen3VLVisionReplayBlockDevice(block_input.tensor,weights,*ws,b,encode.q,0,local,&reference_outputs);
    // Captures retain the computed native result BEFORE the diagnostic override.
    CHECK(local.block_boundaries.at(0).at("projection")==std::vector<float>(4*1152,0));
    CHECK(local.block_boundaries.at(0).at("output")==std::vector<float>(4*1152,0.75f));
    reference_outputs={{"unknown-stage",reference_projection.tensor}};
    CHECK_THROWS_AS(Qwen3VLVisionReplayBlockDevice(block_input.tensor,weights,*ws,b,encode.q,0,local,&reference_outputs),std::runtime_error);
    reference_outputs={{"projection",pixels.tensor}};
    CHECK_THROWS_AS(Qwen3VLVisionReplayBlockDevice(block_input.tensor,weights,*ws,b,encode.q,0,local,&reference_outputs),std::runtime_error);
    Qwen3VLVisionCapture merger;
    CHECK_THROWS_AS(Qwen3VLVisionReplayMergerDevice(block_input.tensor,weights,*ws,b,consumer.q,merger),std::runtime_error);
    CHECK_THROWS_AS(Qwen3VLVisionReplayMergerDevice(wrong_input,weights,*ws,b,encode.q,merger),std::runtime_error);
    Qwen3VLVisionReplayMergerDevice(block_input.tensor,weights,*ws,b,encode.q,merger);
    CHECK(merger.merger_input==replay_values);
    CHECK(merger.merger_norm_out==std::vector<float>(4*1152,0));
    CHECK(merger.merger_out==std::vector<float>{0.25f,-0.5f,1,0});
    Buffer reference_fc1(encode.q,DType::kF16,{1,4*1152});
    reference_fc1.put(std::vector<float>(4*1152,1));
    reference_outputs={{"norm",reference_projection.tensor},{"fc1",reference_fc1.tensor}};
    Qwen3VLVisionReplayMergerDevice(block_input.tensor,weights,*ws,b,encode.q,merger,&reference_outputs);
    CHECK(merger.merger_norm_out==std::vector<float>(4*1152,0));
    CHECK(merger.merger_fc1_out==std::vector<float>(4*1152,0));
    CHECK(merger.merger_gelu_out.size()==4*1152);
    CHECK(merger.merger_gelu_out[0]>0.8f);
    CHECK(block_input.floats()==replay_values);
    reference_outputs={{"unknown-stage",reference_projection.tensor}};
    CHECK_THROWS_AS(Qwen3VLVisionReplayMergerDevice(block_input.tensor,weights,*ws,b,encode.q,merger,&reference_outputs),std::runtime_error);
    reference_outputs={{"norm",pixels.tensor}};
    CHECK_THROWS_AS(Qwen3VLVisionReplayMergerDevice(block_input.tensor,weights,*ws,b,encode.q,merger,&reference_outputs),std::runtime_error);
    Qwen3VLVisionCapture ordinary;
    auto after_replay=Qwen3VLVisionForwardDevice(pixels.tensor,weights,*ws,b,encode.q,&ordinary);
    CHECK(ordinary.block0_out==std::vector<float>(4*1152,0));
    CHECK(ordinary.merger_gelu_out==std::vector<float>(4*1152,0));
    CHECK(first.tensor().data!=second.tensor().data);
    CHECK(first.tensor().dtype==DType::kF16);
    CHECK(first.tensor().shape[0]==1);CHECK(first.tensor().shape[1]==4);
    // Scratch and caller's weight handle can disappear before a consumer;
    // outputs retain their own memory/weights and completion events.
    ws.reset();weights.reset();
    const std::vector<uint16_t> expected{vt::F32ToF16(0.25f),vt::F32ToF16(-0.5f),vt::F32ToF16(1),0};
    const auto expected_bytes=std::vector<unsigned char>(reinterpret_cast<const unsigned char*>(expected.data()),
        reinterpret_cast<const unsigned char*>(expected.data())+expected.size()*2);
    xpu_test::SameBytes(Download(b,consumer.q,first),expected_bytes);
    xpu_test::SameBytes(Download(b,encode.q,second),expected_bytes);
    auto invalid=vt::Queue{{vt::DeviceType::kCPU,0},nullptr};
    CHECK_THROWS_AS(first.WaitOn(invalid),std::runtime_error);
    CHECK_THROWS_AS(first.RecordUse(invalid),std::runtime_error);
  }
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==baseline);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU typed vision tower: real checkpoint and actual two-image worker boundaries" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR","VT_B70_VISION_MODEL_DIR"}))) {
  const char* env=exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");
  const char* model=exl3_test::CaseExternalEnvironment("VT_B70_VISION_MODEL_DIR");

  const std::string dir=env;
  const auto reference=native_vision_test::ReferenceDocument(dir+"/vision-boundaries-capture.json");
  const auto index=vllm::LoadSafetensorsIndex(std::string(model)+"/model.safetensors.index.json");
  std::set<std::string> names;
  for (const auto& [key,file] : index) if (key.rfind("model.visual.",0)==0) names.insert(file);
  std::vector<vllm::SafetensorsFile> shards;
  for (const auto& file : names) shards.push_back(vllm::SafetensorsFile::Open(std::string(model)+"/"+file));
  const auto cfg=Config(true);auto host=vllm::LoadQwen3VLVisionWeights(shards,cfg);
  auto& b=vt::GetBackend(vt::DeviceType::kXPU);xpu_test::Queue encode(vt::DeviceType::kXPU),consumer(vt::DeviceType::kXPU);
  auto weights=PrepareVisionDeviceWeights(host,cfg,b,DType::kF16);
  host={}; // No host mirror needed during tower execution.
  auto ws=PrepareVisionWorkspace({1,24,32},cfg,b,encode.q);
  Buffer pixels(encode.q,DType::kF16,{768,1536});
  for (size_t image=0;image<2;++image) {
    const size_t base=image*35;
    const auto boundary=[&](size_t offset) {
      return Read(dir+"/"+reference.at("worker").at("captures")[base+offset].at("file").get<std::string>());
    };
    const auto input=boundary(0);REQUIRE(input.size()==pixels.bytes);pixels.upload(input.data());
    Qwen3VLVisionCapture cap;
    auto output=Qwen3VLVisionForwardDevice(pixels.tensor,weights,*ws,b,encode.q,&cap);
    const std::string label="image"+std::to_string(image);
    // Stricter FP16 pipeline gates fixed before candidate execution; the old
    // BF16 tower has wider 0.005/0.012/0.08 relative gates. Position/RoPE remain
    // storage-exact; final logits/token quality are later serving gates.
    Compare(cap.patch_embed_out,boundary(3),3e-4,0.02,label+"-patch");
    Compare(cap.pos_embeds,boundary(4),0,0,label+"-position");
    Compare(cap.rotary_cos,boundary(5),0,0,label+"-cos");
    Compare(cap.rotary_sin,boundary(6),0,0,label+"-sin");
    Compare(cap.block0_out,boundary(26),1e-3,0.05,label+"-block0");
    Compare(cap.merger_out,boundary(34),3e-3,0.05,label+"-merger");
    if (const char* dump=std::getenv("VT_B70_VISION_OPERATOR_DUMP_DIR")) {
      REQUIRE(cap.block_outputs.size()==27);
      for (size_t block=0;block<cap.block_outputs.size();++block) {
        const auto& values=cap.block_outputs[block];std::vector<uint16_t> bits(values.size());
        for (size_t i=0;i<bits.size();++i) bits[i]=vt::F32ToF16(values[i]);
        std::ofstream file(std::string(dump)+"/tower-native-"+label+"-block"+std::to_string(block)+".float16",std::ios::binary);
        REQUIRE(file.good());file.write(reinterpret_cast<const char*>(bits.data()),bits.size()*2);
      }
      for (const auto& [name,values] : std::vector<std::pair<std::string,const std::vector<float>*>>{
          {"merger-input",&cap.merger_input},{"merger-norm",&cap.merger_norm_out},
          {"merger-fc1",&cap.merger_fc1_out},{"merger-gelu",&cap.merger_gelu_out}}) {
        std::vector<uint16_t> bits(values->size());
        for (size_t i=0;i<bits.size();++i) bits[i]=vt::F32ToF16((*values)[i]);
        std::ofstream file(std::string(dump)+"/tower-native-"+label+"-"+name+".float16",std::ios::binary);
        REQUIRE(file.good());file.write(reinterpret_cast<const char*>(bits.data()),bits.size()*2);
      }
    }
    const auto first=Download(b,consumer.q,output);
    auto repeat=Qwen3VLVisionForwardDevice(pixels.tensor,weights,*ws,b,encode.q);
    xpu_test::SameBytes(Download(b,consumer.q,repeat),first);
    xpu_test::SameBytes(Download(b,consumer.q,output),first);
    xpu_test::SameBytes(pixels.download(),input);
  }
  CHECK(vt::GetReferenceTierHits()==0);
}
