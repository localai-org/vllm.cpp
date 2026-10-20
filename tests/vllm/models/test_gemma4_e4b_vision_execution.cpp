#include <doctest/doctest.h>
#include <cstring>
#include <cmath>
#include <limits>
#include <algorithm>
#include <fstream>
#include <nlohmann/json.hpp>
#include "vllm/model_executor/models/gemma4_loaded.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/multimodal/gemma4_image_processor.h"

namespace {
using nlohmann::json;
json golden() {
  std::ifstream f(std::string(E4B_VISION_FIXTURES)+"/oracle.json");
  return json::parse(f);
}
vllm::HfConfig config(const json& j) {
  vllm::HfConfig c;
  c.hidden_size=2; c.vocab_size=3; c.num_hidden_layers=1;
  c.num_attention_heads=1; c.num_key_value_heads=1; c.head_dim=2;
  c.raw={{"text_config",json::object()},{"vision_config",j.at("config")}};
  return c;
}
vllm::Gemma4Weights weights(const vllm::HfConfig& c) {
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(std::string(E4B_VISION_FIXTURES)+"/model.safetensors"));
  return vllm::LoadGemma4ForConditionalGenerationWeights(shards,c);
}
void compare(const std::vector<float>& actual,const json& expected,float tolerance) {
  const auto ref=expected.get<std::vector<float>>(); REQUIRE(actual.size()==ref.size());
  float error=0; for(size_t i=0;i<ref.size();++i) {
    REQUIRE(std::isfinite(actual[i]));
    error=std::max(error,std::abs(actual[i]-ref[i]));
  }
  INFO("max absolute error="<<error); CHECK(error<=tolerance);
}
}
TEST_CASE("E4B vision: independent nonzero BF16 Torch stages") {
  const auto j=golden(); auto c=config(j); auto w=weights(c);
  auto cfg=*w.vision_config; auto vw=w.vision_patch_weights;
  const auto& connector=w.vision.embedding_projection;
  vw.embed_projection.resize(connector.bytes.size()/2);
  std::memcpy(vw.embed_projection.data(),connector.bytes.data(),connector.bytes.size());
  const auto& small=j.at("small");
  auto pixels=small.at("pixels").get<std::vector<float>>();
  auto pos=small.at("positions").get<std::vector<int64_t>>();
  vllm::multimodal::Gemma4VisionCapture capture;
  auto& backend=vt::GetBackend(vt::DeviceType::kCPU);
  auto out=vllm::multimodal::Gemma4VisionForward(pixels,pos,18,vw,cfg,backend,&capture);
  compare(capture.patch_embed_out,small.at("patch"),0.004f);
  compare(capture.attention_out,small.at("attention"),0.032f);
  compare(capture.mlp_out,small.at("mlp"),0.032f);
  compare(capture.encoder_out,small.at("encoder"),0.032f);
  compare(capture.pooled,small.at("pooled"),0.032f);
  compare(out,small.at("projected"),0.008f);
  for (int mode=0; mode<6; ++mode) {
    auto bad=cfg;
    if(mode==0)bad.num_kv_heads=2;
    if(mode==1)bad.hidden_size=5;
    if(mode==2)bad.head_dim=2;
    if(mode==3)bad.rms_norm_eps=0;
    if(mode==4)bad.rope_theta=0;
    if(mode==5)bad.num_heads=INT64_MAX;
    CHECK_THROWS_WITH(vllm::multimodal::Gemma4VisionForward(pixels,pos,18,vw,bad,backend),
                      doctest::Contains("execution config"));
  }
  auto broken=vw; broken.blocks[0].k_proj.pop_back();
  CHECK_THROWS_WITH(vllm::multimodal::Gemma4VisionForward(pixels,pos,18,broken,cfg,backend),
                    doctest::Contains("weight shape"));
  auto badpos=pos;badpos[2]=0;
  CHECK_THROWS(vllm::multimodal::Gemma4VisionForward(pixels,badpos,18,vw,cfg,backend));
  pixels[0]=std::numeric_limits<float>::quiet_NaN();
  CHECK_THROWS(vllm::multimodal::Gemma4VisionForward(pixels,pos,18,vw,cfg,backend));

}
TEST_CASE("E4B vision: decoded RGB reaches loaded shared registry and PLE merge") {
  auto j=golden(); auto c=config(j); auto w=weights(c);
  const vllm::ModelRegistration reg{"Gemma4ForConditionalGeneration",&vllm::kGemma4MmHooks,{}};
  vllm::Gemma4LoadedModel model(reg,std::move(w));
  std::vector<uint8_t> rgb(384*384*3);
  for(size_t i=0;i<rgb.size();++i)rgb[i]=static_cast<uint8_t>((i*17+i/7)%256);
  auto features=vllm::multimodal::Gemma4ImageProcessor(70).ProcessImageRGB(rgb,384,384);
  vllm::multimodal::MultiModalFeatureSpec item;
  item.modality="image";item.length=static_cast<int>(features.num_soft_tokens);
  item.data=std::make_shared<vllm::multimodal::ImageKwargs>(std::move(features.patches));
  vt::Queue q{};
  auto out=vllm::ModelRegistry::EncodeMm(model,c,q,item);
  REQUIRE(out.embeds.dtype==vt::DType::kF32); REQUIRE(out.embeds.shape[0]==item.length);
  REQUIRE(out.embeds.shape[1]==2); REQUIRE(out.storage);
  compare(std::vector<float>(out.embeds.Ptr<float>(),out.embeds.Ptr<float>()+item.length*2),j.at("image_projected"),0.016f);
  std::vector<int32_t> ids(static_cast<size_t>(item.length)+2,1); ids.front()=2;
  std::vector<char> mask(ids.size(),1);mask.front()=mask.back()=0;
  std::vector<vt::Tensor> slices{out.embeds};
  vllm::MmEmbedInputs in;in.token_ids=&ids;in.is_mm_embed=&mask;in.mm_embeds=&slices;
  auto merged=vllm::ModelRegistry::EmbedMm(model,c,q,in);
  for(int i=0;i<item.length;++i) {
    CHECK(merged.mm.ple_token_ids.Ptr<int32_t>()[i+1]==0);
    for(int h=0;h<2;++h)CHECK(merged.mm.inputs_embeds.Ptr<uint16_t>()[(i+1)*2+h]==vt::F32ToBF16(out.embeds.Ptr<float>()[i*2+h]));
  }
  CHECK(merged.mm.ple_token_ids.Ptr<int32_t>()[0]==2);
  CHECK(merged.mm.ple_token_ids.Ptr<int32_t>()[ids.size()-1]==1);
  CHECK(vllm::ModelRegistry::SupportsMmInputs(model));
  auto original=*item.data;
  for(int mode=0;mode<8;++mode) {
    if(mode==0)item.data->image_grid_thw[0]=2;
    if(mode==1)item.data->image_grid_thw[1]=INT64_MAX;
    if(mode==2)item.data->image_grid_thw[2]=0;
    if(mode==3)item.data->pixel_values_bf16.push_back(0);
    if(mode==4)item.data->pixel_values_f32.pop_back();
    if(mode==5)item.data->pixel_values_f32.back()=1;
    if(mode==6)item.data->patch_feature_dim=4;
    if(mode==7)item.data->num_patches=INT64_MAX;
    CHECK_THROWS(vllm::ModelRegistry::EncodeMm(model,c,q,item));
    *item.data=original;
  }
  item.length++; CHECK_THROWS(vllm::ModelRegistry::EncodeMm(model,c,q,item));item.length--;
  item.modality="video";CHECK_THROWS(vllm::ModelRegistry::EncodeMm(model,c,q,item));item.modality="image";
  q.device.type=vt::DeviceType::kCUDA;CHECK_THROWS(vllm::ModelRegistry::EncodeMm(model,c,q,item));
}

TEST_CASE("E4B vision: round pooled average before non-power-of-two scale") {
  std::ifstream f(std::string(E4B_VISION_FIXTURES)+"/pool_rounding.json");
  const auto j=json::parse(f);
  vllm::multimodal::Gemma4VisionConfig cfg;
  cfg.hidden_size=12; cfg.num_heads=cfg.num_kv_heads=3; cfg.head_dim=4;
  cfg.depth=1; cfg.intermediate_size=6; cfg.position_embedding_size=6;
  cfg.text_hidden_size=2;
  REQUIRE(j.at("width")==cfg.hidden_size);
  REQUIRE(j.at("heads")==cfg.num_heads);
  REQUIRE(j.at("head_dim")==cfg.head_dim);
  vllm::multimodal::Gemma4VisionWeights w;
  w.input_proj.resize(12*768);
  for(int h=0;h<12;++h)w.input_proj[h*768+h]=vt::F32ToBF16(1);
  w.position_embedding_table=j.at("position_table").get<std::vector<float>>();
  w.embed_projection.assign(2*12,vt::F32ToBF16(0.125f));
  w.blocks.resize(1);
  auto& block=w.blocks[0];
  for(auto* v:{&block.q_proj,&block.k_proj,&block.v_proj})
    v->assign(12*12,vt::F32ToBF16(0.125f));
  // Isolate pooling without bypassing any production stage or provider.
  block.o_proj.resize(12*12);
  for(auto* v:{&block.gate_proj,&block.up_proj})v->assign(6*12,vt::F32ToBF16(0.125f));
  block.down_proj.resize(12*6);
  for(auto* v:{&block.input_ln,&block.post_attn_ln,&block.pre_ff_ln,&block.post_ff_ln})
    v->assign(12,vt::F32ToBF16(1));
  for(auto* v:{&block.q_norm,&block.k_norm})v->assign(4,vt::F32ToBF16(1));
  vllm::multimodal::Gemma4VisionCapture capture;
  auto& backend=vt::GetBackend(vt::DeviceType::kCPU);
  const auto out=vllm::multimodal::Gemma4VisionForward(
      j.at("pixels").get<std::vector<float>>(),j.at("positions").get<std::vector<int64_t>>(),
      18,w,cfg,backend,&capture);
  REQUIRE(out.size()==4);
  compare(capture.patch_embed_out,j.at("encoder"),0);
  compare(capture.encoder_out,j.at("encoder"),0);
  compare(capture.pooled,j.at("pooled"),0);
  const auto expected=j.at("pooled").get<std::vector<float>>();
  const auto late=j.at("late_rounding").get<std::vector<float>>();
  REQUIRE(expected.size()==late.size());
  CHECK(expected!=late);
}

TEST_CASE("E4B vision: exact storage for every tower weight") {
  const auto j=golden(); const auto c=config(j); const auto loaded=weights(c);
  const auto cfg=*loaded.vision_config;
  auto w=loaded.vision_patch_weights;
  w.embed_projection.resize(loaded.vision.embedding_projection.bytes.size()/2);
  std::memcpy(w.embed_projection.data(),loaded.vision.embedding_projection.bytes.data(),
              loaded.vision.embedding_projection.bytes.size());
  const auto pixels=j.at("small").at("pixels").get<std::vector<float>>();
  const auto pos=j.at("small").at("positions").get<std::vector<int64_t>>();
  auto& backend=vt::GetBackend(vt::DeviceType::kCPU);
  auto& b=w.blocks[0];
  auto reject_sizes=[&](auto& storage,const char* message) {
    const auto original=storage;
    for(int delta:{-1,1}) {
      CAPTURE(delta); CAPTURE(message);
      storage.resize(original.size()+delta);
      CHECK_THROWS_WITH(vllm::multimodal::Gemma4VisionForward(pixels,pos,18,w,cfg,backend),
                        doctest::Contains(message));
      storage=original;
    }
  };
  int index=0;
  for(auto* v:{&b.q_proj,&b.k_proj,&b.v_proj,&b.o_proj}) {
    CAPTURE(index); reject_sizes(*v,"attention weight shape"); ++index;
  }
  for(auto* v:{&b.gate_proj,&b.up_proj,&b.down_proj})reject_sizes(*v,"MLP weight shape");
  for(auto* v:{&b.input_ln,&b.post_attn_ln,&b.pre_ff_ln,&b.post_ff_ln})reject_sizes(*v,"norm weight shape");
  for(auto* v:{&b.q_norm,&b.k_norm})reject_sizes(*v,"head norm shape");
  reject_sizes(w.input_proj,"weight shape");
  reject_sizes(w.position_embedding_table,"weight shape");
  reject_sizes(w.embed_projection,"weight shape");
  reject_sizes(w.blocks,"weight shape");
}

TEST_CASE("E4B vision: shared registry rejects malformed raw connector") {
  const auto j=golden(); const auto c=config(j);
  const vllm::ModelRegistration reg{"Gemma4ForConditionalGeneration",&vllm::kGemma4MmHooks,{}};
  vllm::multimodal::MultiModalFeatureSpec item;
  item.modality="image"; item.length=2;
  item.data=std::make_shared<vllm::multimodal::ImageKwargs>();
  item.data->num_patches=18; item.data->patch_feature_dim=768;
  item.data->image_grid_thw={1,3,6};
  item.data->pixel_values_f32=j.at("small").at("pixels").get<std::vector<float>>();
  vt::Queue q{};
  for(int mode=0;mode<9;++mode) {
    CAPTURE(mode);
    auto w=weights(c); auto& p=w.vision.embedding_projection;
    if(mode==0)p.dtype=vt::DType::kF32;
    if(mode==1)p.rank=1;
    if(mode==2)--p.shape[0];
    if(mode==3)++p.shape[0];
    if(mode==4)--p.shape[1];
    if(mode==5)++p.shape[1];
    if(mode>=6) {
      // Own a copy before resizing; the loader may return borrowed storage.
      std::vector<uint8_t> bytes(p.bytes.begin(),p.bytes.end());
      if(mode==6)bytes.resize(bytes.size()-1);
      if(mode==7)bytes.resize(bytes.size()+1);
      if(mode==8)bytes.resize(bytes.size()+2);
      p.bytes=vllm::OwnedBytes(std::move(bytes));
    }
    vllm::Gemma4LoadedModel model(reg,std::move(w));
    CHECK_THROWS_WITH(vllm::ModelRegistry::EncodeMm(model,c,q,item),
                      doctest::Contains("BF16 connector shape"));
  }
}
