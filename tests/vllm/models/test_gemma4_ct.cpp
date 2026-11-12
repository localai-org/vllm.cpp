// Packed group32 integer weights follow the Q4_0/Q8_0 converter contract.
#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>
#include <nlohmann/json.hpp>
#include "doctest/doctest.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vt/backend.h"
#include "vt/dtype.h"
namespace {
using nlohmann::json;
struct Fixture {
 std::filesystem::path dir = std::filesystem::temp_directory_path()/("gemma4-ct-fixture-"+std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
 json header = json::object(); std::vector<uint8_t> payload;
 Fixture() { std::filesystem::create_directories(dir); }
 ~Fixture() { std::filesystem::remove_all(dir); }
 template<class T> void Add(const std::string& name,const std::string& dtype,
                            std::vector<int64_t> shape,const std::vector<T>& data) {
  size_t off=payload.size(), count=data.size()*sizeof(T);
  payload.resize(off+count); std::memcpy(payload.data()+off,data.data(),count);
  header[name]={{"dtype",dtype},{"shape",shape},{"data_offsets",{off,off+count}}};
 }
 void Half(const std::string& name,std::vector<int64_t> shape,float value=1) {
  size_t n=1; for(auto s:shape)n*=static_cast<size_t>(s);
  Add(name,"F16",shape,std::vector<uint16_t>(n,vt::F32ToF16(value)));
 }
 void Packed(const std::string& name,int bits,int n=32,int k=32) {
  std::vector<int32_t> codes(static_cast<size_t>(n*k*bits/32));
  uint32_t seed=2166136261u;
  for(unsigned char c:name)seed=(seed^c)*16777619u;
  for(size_t i=0;i<codes.size();++i) {
   uint32_t word=0; for(int j=0;j<32/bits;++j)
    word|=static_cast<uint32_t>((seed+(i*19+j*13)+(i/7)*23)%static_cast<size_t>(1u<<bits))<<(j*bits);
   std::memcpy(&codes[i],&word,4);
  }
  Add(name+".weight_packed","I32",{n,k*bits/32},codes);
  Add(name+".weight_shape","I32",{2},std::vector<int32_t>{n,k});
  std::vector<uint16_t> scales(static_cast<size_t>(n*k/32));
  for(size_t i=0;i<scales.size();++i)scales[i]=vt::F32ToF16(i%2?-0.03125f:0.03125f);
  Add(name+".weight_scale","F16",{n,k/32},scales);
 }
 void Save() {
  std::string h=header.dump(); while(h.size()%8)h+=' ';
  uint64_t len=h.size();std::ofstream st(dir/"model.safetensors",std::ios::binary);
  st.write(reinterpret_cast<char*>(&len),8);st<<h;
  st.write(reinterpret_cast<char*>(payload.data()),static_cast<std::streamsize>(payload.size()));
 }
 vllm::HfConfig Write() {
  const std::string b="model.language_model.", l=b+"layers.0.";
  std::vector<uint16_t> embedding(32*32);
  for(size_t i=0;i<embedding.size();++i)
    embedding[i]=vt::F32ToF16((static_cast<int>((i*13+i/32)%31)-15)*0.03125f);
  Add(b+"embed_tokens.weight","F16",{32,32},embedding); Half(b+"norm.weight",{32});
  for(const auto* s:{"input_layernorm.weight","post_attention_layernorm.weight",
   "pre_feedforward_layernorm.weight","post_feedforward_layernorm.weight",
   "pre_feedforward_layernorm_2.weight","post_feedforward_layernorm_1.weight",
   "post_feedforward_layernorm_2.weight"}) Half(l+s,{32});
  Half(l+"self_attn.q_norm.weight",{32}); Half(l+"self_attn.k_norm.weight",{32});
  Packed(l+"self_attn.q_proj",4); Packed(l+"self_attn.k_proj",4);
  Packed(l+"self_attn.v_proj",4); Packed(l+"self_attn.o_proj",8);
  Packed(l+"mlp.gate_proj",4); Packed(l+"mlp.up_proj",4); Packed(l+"mlp.down_proj",4);
  Half(l+"router.scale",{32}); Half(l+"router.proj.weight",{2,32},0.125f);
  Half(l+"router.per_expert_scale",{2});
  for(int e=0;e<2;++e)for(const auto* s:{"gate_proj","up_proj","down_proj"})
    Packed(l+"experts."+std::to_string(e)+"."+s,4);
  Save();
  json weights={{"type","int"},{"strategy","group"},{"num_bits",4},
    {"group_size",32},{"symmetric",true},{"dynamic",false}};
  json groups={{"w4",{{"format","pack-quantized"},{"weights",weights},
    {"input_activations",nullptr},{"targets",{"re:.*(q_proj|k_proj|v_proj|gate_proj|up_proj|down_proj)$"}}}}};
  weights["num_bits"]=8;
  groups["w8"]={{"format","pack-quantized"},{"weights",weights},
    {"input_activations",nullptr},{"targets",{l+"self_attn.o_proj"}}};
  json text={{"hidden_size",32},{"intermediate_size",32},{"head_dim",32},
    {"global_head_dim",32},{"num_attention_heads",1},{"num_key_value_heads",1},
    {"num_hidden_layers",1},{"vocab_size",32},{"max_position_embeddings",32},
    {"rms_norm_eps",1e-6},{"hidden_activation","gelu_pytorch_tanh"},
    {"enable_moe_block",true},{"num_experts",2},{"top_k_experts",1},
    {"moe_intermediate_size",32},{"hidden_size_per_layer_input",0},
    {"layer_types",{"full_attention"}},{"tie_word_embeddings",true}};
  json c={{"model_type","gemma4"},{"architectures",{"Gemma4ForConditionalGeneration"}},
    {"text_config",text},{"quantization_config",{{"quant_method","compressed-tensors"},
    {"format","pack-quantized"},{"config_groups",groups}}}};
  std::ofstream(dir/"config.json")<<c.dump();
  return vllm::LoadHfConfig((dir/"config.json").string());
 }
};
}
TEST_CASE("gemma4 CT production loader accepts mixed packed integers and F16") {
 Fixture f; auto cfg=f.Write();
 auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
 shards->push_back(vllm::SafetensorsFile::Open((f.dir/"model.safetensors").string()));
 auto w=vllm::LoadGemma4ForConditionalGenerationWeightsOwned(shards,cfg);
 CHECK(w.layers.size()==1); CHECK_FALSE(w.layers[0].moe.experts.Empty());
 CHECK(w.layers[0].attn.o_proj.dtype==vt::DType::kQ8_0);
}

namespace {
std::vector<float> Run(Fixture& f,const vllm::HfConfig& cfg) {
 auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
 shards->push_back(vllm::SafetensorsFile::Open((f.dir/"model.safetensors").string()));
 auto model=vllm::ModelRegistry::Load(cfg,vllm::ModelSource::FromSafetensorsOwned(shards));
 auto q=vt::CreateQueue({vt::DeviceType::kCPU,0});
 std::vector<uint16_t> kvbytes(2*16*32,0);
 vllm::PagedKvCache kv;kv.data=kvbytes.data();kv.dtype=vt::DType::kBF16;
 kv.num_blocks=1;kv.block_size=16;kv.num_kv_heads=1;kv.head_size=32;
 std::vector<vllm::PagedKvCache> caches{kv};
 std::vector<vllm::GdnStateCache> states;
 vllm::v1::CommonAttentionMetadata meta;meta.num_reqs=1;meta.num_actual_tokens=1;
 meta.query_start_loc={0,1};meta.query_start_loc_cpu={0,1};meta.seq_lens={1};
 meta.seq_lens_cpu={1};meta.max_query_len=1;meta.max_seq_len=1;
 meta.block_table_num_cols=1;meta.block_table_tensor={0};meta.slot_mapping={0};meta.causal=true;
 vllm::v1::GDNAttentionMetadata gdn;
 std::vector<int32_t> ids{2},pos{0},last{0};
 vllm::ModelForwardInput input{.token_ids=ids,.positions=pos,.attn_meta=meta,.gdn_meta=gdn,
  .attn_kv=caches,.gdn_state=states,.config=cfg,.queue=q,.logits_indices=last,
  .num_reqs=1,.gather_logits=false};
 auto out=vllm::ModelRegistry::Forward(*model,input);
 return out.host;
}
}
TEST_CASE("gemma4 CT packed projections and experts reach registered CPU forward") {
 Fixture f; auto cfg=f.Write();
 auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
 shards->push_back(vllm::SafetensorsFile::Open((f.dir/"model.safetensors").string()));
 auto out = Run(f,cfg);
 REQUIRE(out.size()==32);
 for(float v:out)CHECK(std::isfinite(v));
 CHECK(std::abs(out[0])>0.01f);
}

TEST_CASE("gemma4 CT rejects incompatible schemes before reading packed values") {
 for(const auto* field:{"group_size","symmetric","output_activations"}) {
  Fixture f;auto cfg=f.Write();
  auto& group=cfg.raw["quantization_config"]["config_groups"]["w4"];
  if(std::string(field)=="group_size")group["weights"][field]=64;
  else if(std::string(field)=="symmetric")group["weights"][field]=false;
  else group[field]={{"type","int"},{"num_bits",8}};
  auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
  shards->push_back(vllm::SafetensorsFile::Open((f.dir/"model.safetensors").string()));
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Load(cfg,vllm::ModelSource::FromSafetensorsOwned(shards)),
    doctest::Contains("self_attn.q_proj"),std::runtime_error);
 }
}

TEST_CASE("gemma4 CT rejects padded tensor shapes that disagree with config") {
 Fixture f;auto cfg=f.Write();cfg.intermediate_size=64;
 auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
 shards->push_back(vllm::SafetensorsFile::Open((f.dir/"model.safetensors").string()));
 CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Load(cfg,vllm::ModelSource::FromSafetensorsOwned(shards)),
    doctest::Contains("mlp.gate_proj"),std::runtime_error);
}

TEST_CASE("gemma4 CT matches independent dequantized weights and reaches selected experts") {
 Fixture packed;auto cfg=packed.Write();
 const auto actual=Run(packed,cfg);
 Fixture dense;dense.Write();
 std::vector<std::string> names;
 for(auto it=dense.header.begin();it!=dense.header.end();++it)
   if(it.key().ends_with(".weight_packed"))names.push_back(it.key());
 for(const auto& key:names) {
  const std::string module=key.substr(0,key.size()-14);
  const auto& meta=dense.header.at(key);
  const int bits=module.ends_with("self_attn.o_proj")?8:4;
  const size_t offset=meta["data_offsets"][0].get<size_t>();
  std::vector<uint16_t> values(32*32);
  for(size_t i=0;i<values.size();++i) {
    uint32_t word=0;std::memcpy(&word,dense.payload.data()+offset+(i/(32/bits))*4,4);
    int code=static_cast<int>((word>>((i%(32/bits))*bits))&((1u<<bits)-1));
    float scale=(i/32)%2?-0.03125f:0.03125f;
    values[i]=vt::F32ToBF16(static_cast<float>(code-(1<<(bits-1)))*scale);
  }
  // Independent dense BF16 fixture reuses the same CT loader's ignored arm.
  dense.Add(module+".weight","BF16",{32,32},values);
 }
 auto dense_cfg=cfg;dense_cfg.raw["quantization_config"]["ignore"]={"re:.*"};dense.Save();
 auto expected=Run(dense,dense_cfg);
 REQUIRE(actual.size()==expected.size());
 // Native block GEMM quantizes activation blocks to Q8_0. The independent
 // dense reference keeps activations BF16, so compare with an absolute bound.
 float max_error=0;
 for(size_t i=0;i<actual.size();++i)max_error=std::max(max_error,std::abs(actual[i]-expected[i]));
 std::printf("gemma4 CT compact-vs-dense maximum logit error: %.9g\n", max_error);
 CHECK(max_error<0.05f);
 // Removing the selected expert contribution must change the full model logits.
 auto zero_cfg=cfg;
 const std::string scale="model.language_model.layers.0.router.per_expert_scale";
 const size_t offset=packed.header.at(scale)["data_offsets"][0].get<size_t>();
 std::memset(packed.payload.data()+offset,0,4);packed.Save();
 auto zero=Run(packed,zero_cfg);
 float difference=0;for(size_t i=0;i<actual.size();++i)difference=std::max(difference,std::abs(actual[i]-zero[i]));
 std::printf("gemma4 CT expert contribution maximum logit difference: %.9g\n", difference);
 CHECK(difference>0.01f);
}

TEST_CASE("gemma4 CT preserves native Q4 and Q8 block bytes exactly") {
 Fixture f;auto cfg=f.Write();
 auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
 shards->push_back(vllm::SafetensorsFile::Open((f.dir/"model.safetensors").string()));
 auto w=vllm::LoadGemma4ForConditionalGenerationWeightsOwned(shards,cfg);
 CHECK(w.embed_tokens.dtype==vt::DType::kF16);
 for(int bits:{4,8}) {
  const std::string module="model.language_model.layers.0.self_attn."+
    std::string(bits==4?"q_proj":"o_proj");
  const auto& source=shards->front().Get(module+".weight_packed");
  const auto& scale=shards->front().Get(module+".weight_scale");
  const auto& weight=bits==4?w.layers[0].attn.qkv_parts[0]:w.layers[0].attn.o_proj;
  const size_t block_bytes=bits==4?18:34;
  REQUIRE(weight.bytes.size()==32*block_bytes);
  for(size_t row=0;row<32;++row) {
   const uint8_t* dst=weight.bytes.data()+row*block_bytes;
   CHECK(std::memcmp(dst,scale.data+row*2,2)==0);
   for(int k=0;k<32;++k) {
    uint32_t word=0;std::memcpy(&word,source.data+row*(bits*4)+(k/(32/bits))*4,4);
    const unsigned code=(word>>((k%(32/bits))*bits))&((1u<<bits)-1);
    const unsigned actual=bits==4?((dst[2+k%16]>>((k/16)*4))&15u):(dst[2+k]^128u);
    CHECK(actual==code);
   }
  }
 }
}

TEST_CASE("gemma4 CT refuses unsupported bit widths and declared tensor metadata") {
 for(const auto* defect:{"bits","weight_shape","scale_dtype","ignored","short_shape"}) {
  Fixture f;auto cfg=f.Write();
  const std::string module="model.language_model.layers.0.self_attn.q_proj";
  if(std::string(defect)=="bits")
   cfg.raw["quantization_config"]["config_groups"]["w4"]["weights"]["num_bits"]=3;
  else if(std::string(defect)=="ignored")
   cfg.raw["quantization_config"]["ignore"]={module};
  else if(std::string(defect)=="scale_dtype") {
   f.header[module+".weight_scale"]["dtype"]="BF16";f.Save();
  } else if(std::string(defect)=="short_shape") {
   auto& metadata=f.header[module+".weight_shape"];
   metadata["shape"]={1};
   metadata["data_offsets"][1]=metadata["data_offsets"][0].get<size_t>()+4;
   f.Save();
  } else {
   const size_t off=f.header[module+".weight_shape"]["data_offsets"][0].get<size_t>();
   int32_t wrong=64;std::memcpy(f.payload.data()+off,&wrong,4);f.Save();
  }
  auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
  shards->push_back(vllm::SafetensorsFile::Open((f.dir/"model.safetensors").string()));
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Load(cfg,vllm::ModelSource::FromSafetensorsOwned(shards)),
    doctest::Contains("self_attn.q_proj"),std::runtime_error);
 }
}

TEST_CASE("gemma4 CT explicit V overrides the global sharing flag") {
 const std::string module="model.language_model.layers.0.self_attn.v_proj";
 for(bool plain_v:{false,true}) {
  Fixture f;auto cfg=f.Write();
  if(plain_v) {
   f.Half(module+".weight",{32,32},0.1875f);
   cfg.raw["quantization_config"]["ignore"]={module};
   for(const auto* suffix:{".weight_packed",".weight_scale",".weight_shape"})
    f.header.erase(module+suffix);
   f.Save();
  }
  cfg.raw["text_config"]["attention_k_eq_v"]=false;
  const auto distinct=Run(f,cfg);
  cfg.raw["text_config"]["attention_k_eq_v"]=true;
  const auto stale_flag=Run(f,cfg);
  REQUIRE(distinct.size()==stale_flag.size());
  for(size_t i=0;i<distinct.size();++i)CHECK(distinct[i]==stale_flag[i]);
  auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
  shards->push_back(vllm::SafetensorsFile::Open((f.dir/"model.safetensors").string()));
  auto w=vllm::LoadGemma4ForConditionalGenerationWeightsOwned(shards,cfg);
  CHECK_FALSE(w.layers[0].k_eq_v);
 }
}

TEST_CASE("gemma4 CT absent V shares K independently of the global flag") {
 const std::string base="model.language_model.layers.0.self_attn.";
 Fixture tied;auto cfg=tied.Write();
 for(const auto* suffix:{".weight_packed",".weight_scale",".weight_shape"}) {
  const auto& key=tied.header.at(base+"k_proj"+suffix);
  const auto& value=tied.header.at(base+"v_proj"+suffix);
  const size_t src=key["data_offsets"][0].get<size_t>();
  const size_t dst=value["data_offsets"][0].get<size_t>();
  const size_t count=key["data_offsets"][1].get<size_t>()-src;
  std::memcpy(tied.payload.data()+dst,tied.payload.data()+src,count);
 }
 tied.Save();
 cfg.raw["text_config"]["attention_k_eq_v"]=false;
 const auto explicit_tie=Run(tied,cfg);
 for(const auto* suffix:{".weight_packed",".weight_scale",".weight_shape"})
  tied.header.erase(base+"v_proj"+suffix);
 tied.Save();
 for(bool flag:{false,true}) {
  cfg.raw["text_config"]["attention_k_eq_v"]=flag;
  const auto implicit_tie=Run(tied,cfg);
  REQUIRE(explicit_tie.size()==implicit_tie.size());
  for(size_t i=0;i<explicit_tie.size();++i)CHECK(explicit_tie[i]==implicit_tie[i]);
  auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
  shards->push_back(vllm::SafetensorsFile::Open((tied.dir/"model.safetensors").string()));
  auto w=vllm::LoadGemma4ForConditionalGenerationWeightsOwned(shards,cfg);
  CHECK(w.layers[0].k_eq_v);
 }
}

TEST_CASE("gemma4 CT rejects declared KV quantization through production loading") {
 Fixture f;auto cfg=f.Write();
 auto shards=std::make_shared<std::vector<vllm::SafetensorsFile>>();
 shards->push_back(vllm::SafetensorsFile::Open((f.dir/"model.safetensors").string()));
 auto load=[&] {return vllm::ModelRegistry::Load(cfg,vllm::ModelSource::FromSafetensorsOwned(shards));};
 CHECK_NOTHROW(load());
 cfg.raw["quantization_config"]["kv_cache_scheme"]=nullptr;
 CHECK_NOTHROW(load());
 for(const json& declaration:{json{{"type","float"},{"num_bits",8},{"strategy","tensor"},
                                  {"dynamic",false},{"symmetric",true}},json::object(),json("fp8")}) {
  cfg.raw["quantization_config"]["kv_cache_scheme"]=declaration;
  CHECK_THROWS_WITH_AS(load(),doctest::Contains("kv_cache_scheme"),std::runtime_error);
 }
}
