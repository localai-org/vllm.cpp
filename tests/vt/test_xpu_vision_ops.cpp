#include "xpu_test_helpers.h"
#include "support/native_vision_external_artifacts.h"
#include "vt/unaligned.h"
#include "vt/xpu.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_vl_vision_attention.h"

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <nlohmann/json.hpp>

namespace {
using xpu_test::Buffer;
using vt::DType;
using json = nlohmann::json;

std::vector<unsigned char> Read(const std::string& path) {
  std::ifstream f(path, std::ios::binary);
  REQUIRE_MESSAGE(f.good(), path);
  return {std::istreambuf_iterator<char>(f), std::istreambuf_iterator<char>()};
}
json Document(const std::string& path) {
  return native_vision_test::ReferenceDocument(path);
}

std::vector<float> Floats(const std::vector<unsigned char>& bytes, DType dtype) {
  const size_t unit = vt::SizeOf(dtype);
  REQUIRE(bytes.size() % unit == 0);
  std::vector<float> out(bytes.size() / unit);
  for (size_t i = 0; i < out.size(); ++i) {
    if (dtype == DType::kF32) out[i] = vt::LoadUnaligned<float>(bytes.data() + i * unit);
    else {
      const auto bits = vt::LoadUnaligned<uint16_t>(bytes.data() + i * unit);
      out[i] = dtype == DType::kF16 ? vt::F16ToF32(bits) : vt::BF16ToF32(bits);
    }
  }
  return out;
}

void Compare(const Buffer& result, const std::vector<unsigned char>& expected,
             double relative_limit, double absolute_limit, const std::string& label,
             std::string_view op = "LAYERNORM") {
  const auto actual = result.floats(), ref = Floats(expected, result.tensor.dtype);
  REQUIRE(actual.size() == ref.size());
  double squared = 0.0, norm = 0.0, max_abs = 0.0;
  bool finite = true;
  size_t per_element_failures = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    finite &= std::isfinite(actual[i]) && std::isfinite(ref[i]);
    const double delta = static_cast<double>(actual[i]) - ref[i];
    squared += delta * delta; norm += static_cast<double>(ref[i]) * ref[i];
    max_abs = std::max(max_abs, std::abs(delta));
    if (std::abs(delta) > absolute_limit + relative_limit * std::abs(ref[i]))
      ++per_element_failures;
  }
  const double relative = norm > 0.0 ? std::sqrt(squared / norm) : std::sqrt(squared);
  if (const char* dump = std::getenv("VT_B70_VISION_OPERATOR_DUMP_DIR")) {
    const auto bytes = result.download();
    const std::string prefix = op == "LAYERNORM" ? "layernorm-native-" :
        op == "DENSE" ? "dense-native-" : op == "POSITION" ? "position-native-" :
        op == "QKV" ? "qkv-native-" : op == "ROPE" ? "vision-rope-native-" :
        op == "ATTENTION" ? "attention-native-" : "gelu-native-";
    std::ofstream file(std::string(dump) + "/" + prefix + label + ".bin",
                       std::ios::binary);
    REQUIRE(file.good());
    file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
  }
  INFO(label);
  CAPTURE(relative);
  CAPTURE(max_abs);
  CHECK(finite);
  if (op == "ATTENTION") {
    CAPTURE(per_element_failures);
    CHECK(per_element_failures == 0);
  } else {
    CHECK(relative <= relative_limit); CHECK(max_abs <= absolute_limit);
  }
  std::cout << "VISION_" << op << " " << label << " rel_l2=" << relative
            << " max_abs=" << max_abs << '\n';
}

void AttentionProfile(const std::string& label) {
  json events=json::array(),host=json::array();
  for (const auto& event : vt::xpu::DrainProfileEvents()) {
    events.push_back({{"stage",event.stage},{"queue_id",event.queue_id},
                      {"duration_ns",event.end_ns-event.start_ns}});
    std::cout << "VISION_ATTENTION_PROFILE " << label << ' ' << event.stage
              << " ns=" << event.end_ns-event.start_ns << '\n';
  }
  for (const auto& event : vt::xpu::DrainHostProfileRecords())
    host.push_back({{"stage",event.stage},{"duration_ns",event.end_steady_ns-event.start_steady_ns},
                    {"copy_bytes",event.copy_bytes}});
  if (const char* dump=std::getenv("VT_B70_VISION_OPERATOR_DUMP_DIR")) {
    std::ofstream file(std::string(dump)+"/attention-profile-"+label+".json");
    REQUIRE(file.good()); file << json{{"device_events",events},{"host_events",host}}.dump(2) << '\n';
  }
}
}  // namespace

TEST_CASE("XPU language M-RoPE cache: executed three-axis FP16 coefficients") {
  const std::string dir = std::string(XPU_VISION_FIXTURE_DIR) + "/language-mrope";
  const auto reference = Document(dir + "/language-mrope.json");
  REQUIRE(reference.at("contract").at("bytes") == "exact");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  for (const auto& c : reference.at("cases")) {
    const int64_t t = c.at("tokens");
    Buffer positions(gpu.q,DType::kI64,{3,t}), cache(gpu.q,DType::kF32,{t,64}),
        half(gpu.q,DType::kF16,{t,64});
    const auto source = Read(dir + "/" + c.at("files").at("positions").at("file").get<std::string>());
    const auto expected = Read(dir + "/" + c.at("files").at("expected").at("file").get<std::string>());
    REQUIRE(source.size() == positions.bytes);
    REQUIRE(expected.size() == half.bytes);
    positions.upload(source.data());
    vt::RopeArgs args{};
    args.rotary_dim = 64; args.base = 10000000.f;
    args.linear_scaling_factor = 1.f; args.fp16_intermediates = true;
    args.mrope_section = c.at("section").get<std::array<int32_t,3>>();
    args.mrope_interleaved = c.at("interleaved");
    vt::RopeCosSinCache(gpu.q,cache.tensor,positions.tensor,args);
    vt::CastF16(gpu.q,half.tensor,cache.tensor);
    xpu_test::SameBytes(half.download(),expected);
    const auto first = cache.download();
    const auto before = vt::xpu::GetMemoryInfo().allocated_bytes;
    vt::RopeCosSinCache(gpu.q,cache.tensor,positions.tensor,args);
    xpu_test::SameBytes(cache.download(),first);
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == before);
    xpu_test::SameBytes(positions.download(),source);

    // Both integer widths are device inputs. Equal axes must preserve the
    // existing 1-D producer, including its F32 storage before the F16 cast.
    Buffer i32(gpu.q,DType::kI32,{3,t});
    std::vector<int32_t> ids(3*t);
    for (size_t i=0;i<ids.size();++i)
      ids[i] = static_cast<int32_t>(vt::LoadUnaligned<int64_t>(source.data()+8*i));
    i32.upload(ids.data());
    vt::RopeCosSinCache(gpu.q,cache.tensor,i32.tensor,args);
    xpu_test::SameBytes(cache.download(),first);
    if (c.at("equal_axes").get<bool>()) {
      auto one = i32.tensor; one.rank=1; one.shape[0]=t; one.stride[0]=1;
      vt::RopeCosSinCache(gpu.q,cache.tensor,one,args);
      xpu_test::SameBytes(cache.download(),first);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU language M-RoPE cache: reject invalid axes, sections and overlap") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer positions(gpu.q,DType::kI32,{3,2}), cache(gpu.q,DType::kF32,{2,64});
  const int32_t ids[] = {0,1,0,2,0,3}; positions.upload(ids);
  cache.put(std::vector<float>(128,7));
  const auto initial = cache.download(), source = positions.download();
  vt::RopeArgs args{}; args.rotary_dim=64; args.base=10000000.f;
  args.mrope_section={11,11,10}; args.mrope_interleaved=true;
  auto bad=positions.tensor; bad.shape[0]=2;
  CHECK_THROWS_AS(vt::RopeCosSinCache(gpu.q,cache.tensor,bad,args),std::runtime_error);
  bad=positions.tensor; bad.shape[1]=1;
  CHECK_THROWS_AS(vt::RopeCosSinCache(gpu.q,cache.tensor,bad,args),std::runtime_error);
  bad=positions.tensor; bad.stride[0]=3;
  CHECK_THROWS_AS(vt::RopeCosSinCache(gpu.q,cache.tensor,bad,args),std::runtime_error);
  for (const auto& section : {std::array<int32_t,3>{11,11,9},
                             std::array<int32_t,3>{-1,22,11},
                             std::array<int32_t,3>{INT32_MAX,INT32_MAX,2}}) {
    auto invalid=args; invalid.mrope_section=section;
    CHECK_THROWS_AS(vt::RopeCosSinCache(gpu.q,cache.tensor,positions.tensor,invalid),std::runtime_error);
  }
  auto alias=cache.tensor; alias.data=positions.tensor.data;
  CHECK_THROWS_AS(vt::RopeCosSinCache(gpu.q,alias,positions.tensor,args),std::runtime_error);
  alias.data=static_cast<unsigned char*>(positions.tensor.data)+4;
  CHECK_THROWS_AS(vt::RopeCosSinCache(gpu.q,alias,positions.tensor,args),std::runtime_error);
  xpu_test::SameBytes(cache.download(),initial);
  xpu_test::SameBytes(positions.download(),source);
  xpu_test::Queue cpu(vt::DeviceType::kCPU);
  Buffer cp(cpu.q,DType::kI32,{3,2}), cc(cpu.q,DType::kF32,{2,64});
  cp.upload(ids);
  CHECK_THROWS_AS(vt::RopeCosSinCache(cpu.q,cc.tensor,cp.tensor,args),std::runtime_error);
  CHECK(vt::GetReferenceTierHits() == 0);
}

namespace {
void FusedAttentionReferences(const std::vector<std::pair<std::string,json>>& sources,
                              const std::string& prefix) {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::OpRegistered(vt::OpId::kAttentionDenseFlash,vt::DeviceType::kXPU));
  size_t number=0;
  for (const auto& [dir,reference] : sources) {
    REQUIRE(reference.at("contract").at("relative")==0.001);
    REQUIRE(reference.at("contract").at("absolute")==0.001);
    for (const auto& c : reference.at("cases")) {
      const int64_t t=c.at("shape")[0];
      Buffer q(gpu.q,DType::kF16,{t,16,72}),k(gpu.q,DType::kF16,{t,16,72}),
          v(gpu.q,DType::kF16,{t,16,72}),out(gpu.q,DType::kF16,{t,16,72});
      const auto bytes=[&](const char* name) { return Read(dir+"/"+c.at("files").at(name).at("file").get<std::string>()); };
      const auto qi=bytes("q"),ki=bytes("k"),vi=bytes("v"),expected=bytes("expected");
      REQUIRE(qi.size()==q.bytes);REQUIRE(ki.size()==k.bytes);REQUIRE(vi.size()==v.bytes);REQUIRE(expected.size()==out.bytes);
      q.upload(qi.data());k.upload(ki.data());v.upload(vi.data());
      const vt::AttentionArgs args{c.at("scale").get<float>(),false};
      // Warm compilation and queue-owned scratch once. Production repeated
      // submissions below are asynchronous; readbacks are test verification.
      vt::AttentionDenseFlash(gpu.q,out.tensor,q.tensor,k.tensor,v.tensor,args);
      (void)out.download();
      AttentionProfile(prefix+"-warm"+std::to_string(number));
      const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
      vt::AttentionDenseFlash(gpu.q,out.tensor,q.tensor,k.tensor,v.tensor,args);
      const auto label=prefix+std::to_string(number);
      Compare(out,expected,0.001,0.001,label,"ATTENTION");
      const auto first=out.download();
      // The optional Xe2 policy matched all recorded executed-reference bytes.
      // Freeze that stronger standalone result without changing tower gates.
      if (const char* candidate=std::getenv("VT_XPU_VISION_XE2"); candidate && candidate[0]=='1')
        xpu_test::SameBytes(first,expected);
      vt::AttentionDenseFlash(gpu.q,out.tensor,q.tensor,k.tensor,v.tensor,args);
      xpu_test::SameBytes(out.download(),first);
      xpu_test::SameBytes(q.download(),qi);xpu_test::SameBytes(k.download(),ki);xpu_test::SameBytes(v.download(),vi);
      CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
      AttentionProfile(label);
      ++number;
    }
  }
  CHECK(vt::GetReferenceTierHits()==0);
}
}

TEST_CASE("XPU vision fused attention: executed synthetic fixtures") {
  const std::string fixtures=XPU_VISION_FIXTURE_DIR;
  FusedAttentionReferences({{fixtures,Document(fixtures+"/attention.json")}},"fused-synthetic");
}

TEST_CASE("XPU vision fused attention: separate real-input references" * doctest::test_suite("vision-external") *
          doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR"}))) {
  const std::string dir=exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");
  FusedAttentionReferences({{dir,Document(dir+"/attention-real-replays.json")}},"fused-real");
}

TEST_CASE("XPU vision fused attention: refuse unsupported geometry, masks and overlap") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer q(gpu.q,DType::kF16,{2,16,72}),k(gpu.q,DType::kF16,{2,16,72}),
      v(gpu.q,DType::kF16,{2,16,72}),out(gpu.q,DType::kF16,{2,16,72});
  q.put(std::vector<float>(q.bytes/2,1));k.put(std::vector<float>(k.bytes/2,2));v.put(std::vector<float>(v.bytes/2,3));
  const auto qi=q.download(),ki=k.download(),vi=v.download();
  for (const auto& input : {q.tensor,k.tensor,v.tensor}) {
    auto alias=out.tensor;alias.data=input.data;
    CHECK_THROWS_AS(vt::AttentionDenseFlash(gpu.q,alias,q.tensor,k.tensor,v.tensor,{0.1f,false}),std::runtime_error);
    alias.data=static_cast<char*>(input.data)+2;
    CHECK_THROWS_AS(vt::AttentionDenseFlash(gpu.q,alias,q.tensor,k.tensor,v.tensor,{0.1f,false}),std::runtime_error);
  }
  CHECK_THROWS_AS(vt::AttentionDenseFlash(gpu.q,out.tensor,q.tensor,k.tensor,v.tensor,{0.1f,true}),std::runtime_error);
  CHECK_THROWS_AS(vt::AttentionDenseFlash(gpu.q,out.tensor,q.tensor,k.tensor,v.tensor,{std::numeric_limits<float>::infinity(),false}),std::runtime_error);
  auto a=q.tensor,b=k.tensor,c=v.tensor,d=out.tensor;
  a.shape[0]=b.shape[0]=c.shape[0]=d.shape[0]=16385;
  CHECK_THROWS_AS(vt::AttentionDenseFlash(gpu.q,d,a,b,c,{0.1f,false}),std::runtime_error);
  a=q.tensor;b=k.tensor;c=v.tensor;d=out.tensor;
  a.dtype=b.dtype=c.dtype=d.dtype=DType::kBF16;
  CHECK_THROWS_AS(vt::AttentionDenseFlash(gpu.q,d,a,b,c,{0.1f,false}),std::runtime_error);
  a=q.tensor;b=k.tensor;c=v.tensor;d=out.tensor;
  a.shape[2]=b.shape[2]=c.shape[2]=d.shape[2]=64;
  a.stride[1]=b.stride[1]=c.stride[1]=d.stride[1]=64;
  a.stride[0]=b.stride[0]=c.stride[0]=d.stride[0]=16*64;
  CHECK_THROWS_AS(vt::AttentionDenseFlash(gpu.q,d,a,b,c,{0.1f,false}),std::runtime_error);
  xpu_test::SameBytes(q.download(),qi);xpu_test::SameBytes(k.download(),ki);xpu_test::SameBytes(v.download(),vi);
  auto cpu=vt::Queue{{vt::DeviceType::kCPU,0},nullptr};
  a=q.tensor;b=k.tensor;c=v.tensor;d=out.tensor;
  a.device=b.device=c.device=d.device=cpu.device;
  CHECK_THROWS_AS(vt::AttentionDenseFlash(cpu,d,a,b,c,{0.1f,false}),std::runtime_error);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision fused attention: maximum patches and independent queue lifetimes") {
  const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
  {
    xpu_test::Queue first(vt::DeviceType::kXPU),second(vt::DeviceType::kXPU);
    struct Pending {
      Buffer q,k,v,out;
      float value;
      Pending(vt::Queue& queue,int64_t length,float constant)
          : q(queue,DType::kF16,{length,16,72}),k(queue,DType::kF16,{length,16,72}),
            v(queue,DType::kF16,{length,16,72}),out(queue,DType::kF16,{length,16,72}),value(constant) {
        q.put(std::vector<float>(q.bytes/2,0));k.put(std::vector<float>(k.bytes/2,0));
        v.put(std::vector<float>(v.bytes/2,value));
      }
      void submit() { vt::AttentionDenseFlash(q.q,out.tensor,q.tensor,k.tensor,v.tensor,{1.0f/std::sqrt(72.0f),false}); }
      void verify() {
        const auto result=out.floats();
        // Uniform attention over a constant V has the same constant output,
        // independently of sequence length or the library's tiling choices.
        CHECK(std::all_of(result.begin(),result.end(),[&](float x) { return std::isfinite(x) && std::abs(x-value)<=0.001f; }));
      }
    };
    std::vector<std::unique_ptr<Pending>> pending;
    for (int64_t length=1;length<=9;++length) {
      pending.push_back(std::make_unique<Pending>(first.q,length,0.5f));
      pending.push_back(std::make_unique<Pending>(second.q,length,-0.25f));
    }
    // Submit both queues without readbacks, including more than the plan-cache
    // capacity. Retiring a plan must preserve every outstanding consumer.
    for (auto& item : pending) item->submit();
    for (auto& item : pending) item->verify();
    pending.clear();
    Pending maximum(first.q,16384,0.5f);
    maximum.submit();maximum.verify();
    const auto warm=vt::xpu::GetMemoryInfo().allocated_bytes;
    maximum.submit();maximum.verify();
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==warm);
    AttentionProfile("fused-lifecycle-max");
  }
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision attention: executed non-causal FP16 fixtures") {
  const std::string dir=XPU_VISION_FIXTURE_DIR;
  const auto reference=Document(dir+"/attention.json");
  REQUIRE(reference.at("fp8_enabled")==false);
  REQUIRE(reference.at("contract").at("relative")==0.001);
  REQUIRE(reference.at("contract").at("absolute")==0.001);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::OpRegistered(vt::OpId::kAttention,vt::DeviceType::kXPU));
  size_t number=0;
  for (const auto& c : reference.at("cases")) {
    REQUIRE(c.at("causal")==false);
    const int64_t t=c.at("shape")[0];
    Buffer q(gpu.q,DType::kF16,{t,16,72}),k(gpu.q,DType::kF16,{t,16,72}),
        v(gpu.q,DType::kF16,{t,16,72}),out(gpu.q,DType::kF16,{t,16,72});
    const auto bytes=[&](const char* name) { return Read(dir+"/"+c.at("files").at(name).at("file").get<std::string>()); };
    const auto qi=bytes("q"),ki=bytes("k"),vi=bytes("v"),expected=bytes("expected");
    REQUIRE(qi.size()==q.bytes);REQUIRE(ki.size()==k.bytes);REQUIRE(vi.size()==v.bytes);REQUIRE(expected.size()==out.bytes);
    q.upload(qi.data());k.upload(ki.data());v.upload(vi.data());
    const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
    const vt::AttentionArgs args{c.at("scale").get<float>(),false};
    vt::Attention(gpu.q,out.tensor,q.tensor,k.tensor,v.tensor,args);
    const auto label="fixture"+std::to_string(number);
    Compare(out,expected,0.001,0.001,label,"ATTENTION");
    const auto first=out.download();
    vt::Attention(gpu.q,out.tensor,q.tensor,k.tensor,v.tensor,args);
    xpu_test::SameBytes(out.download(),first);
    xpu_test::SameBytes(q.download(),qi);xpu_test::SameBytes(k.download(),ki);xpu_test::SameBytes(v.download(),vi);
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
    AttentionProfile(label);
    ++number;
  }
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision attention: separate real-input image replays" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR"}))) {
  const char* env=exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");

  const std::string dir=env;
  const auto reference=Document(dir+"/attention-real-replays.json");
  REQUIRE(reference.at("cases").size()==2);
  REQUIRE(reference.at("contract").at("relative")==0.001);
  REQUIRE(reference.at("contract").at("absolute")==0.001);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer q(gpu.q,DType::kF16,{768,16,72}),k(gpu.q,DType::kF16,{768,16,72}),
      v(gpu.q,DType::kF16,{768,16,72}),out(gpu.q,DType::kF16,{768,16,72});
  for (const auto& c : reference.at("cases")) {
    REQUIRE(c.at("causal")==false);
    const auto bytes=[&](const char* name) { return Read(dir+"/"+c.at("files").at(name).at("file").get<std::string>()); };
    const auto qi=bytes("q"),ki=bytes("k"),vi=bytes("v"),expected=bytes("expected");
    REQUIRE(qi.size()==q.bytes);REQUIRE(ki.size()==k.bytes);REQUIRE(vi.size()==v.bytes);REQUIRE(expected.size()==out.bytes);
    q.upload(qi.data());k.upload(ki.data());v.upload(vi.data());
    const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
    const vt::AttentionArgs args{c.at("scale").get<float>(),false};
    vt::Attention(gpu.q,out.tensor,q.tensor,k.tensor,v.tensor,args);
    const auto label="image"+std::to_string(c.at("image").get<int>());
    Compare(out,expected,0.001,0.001,label,"ATTENTION");
    const auto first=out.download();
    vt::Attention(gpu.q,out.tensor,q.tensor,k.tensor,v.tensor,args);
    xpu_test::SameBytes(out.download(),first);
    xpu_test::SameBytes(q.download(),qi);xpu_test::SameBytes(k.download(),ki);xpu_test::SameBytes(v.download(),vi);
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
    AttentionProfile(label);
  }
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision LayerNorm: pinned operator, dtype, tails and alias") {
  const std::string dir = XPU_VISION_FIXTURE_DIR;
  const auto manifest = Document(dir + "/layernorm.json");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::OpRegistered(vt::OpId::kLayerNorm, vt::DeviceType::kXPU));
  size_t number = 0;
  for (const auto& c : manifest.at("cases")) {
    const auto name = c.at("dtype").get<std::string>();
    const auto dtype = name == "float16" ? DType::kF16 : name == "bfloat16" ? DType::kBF16 : DType::kF32;
    const int64_t rows = c.at("shape")[0], width = c.at("shape")[1];
    const auto bytes = [&](const char* field) { return Read(dir + "/" + c.at("files").at(field).at("file").get<std::string>()); };
    const auto input = bytes("input"), expected = bytes("expected");
    Buffer x(gpu.q, dtype, {std::max<int64_t>(rows, 1), width});
    Buffer y(gpu.q, dtype, {std::max<int64_t>(rows, 1), width});
    // Tensor::Contiguous requires positive allocation dimensions. Exercise the
    // operator's empty-batch no-op using a zero-row view of a live allocation.
    if (rows == 0) {
      x.tensor.shape[0] = y.tensor.shape[0] = 0;
      x.bytes = y.bytes = 0;
    }
    Buffer w(gpu.q, dtype, {width}), b(gpu.q, dtype, {width});
    REQUIRE(x.bytes == input.size()); REQUIRE(y.bytes == expected.size());
    x.upload(input.data());
    const bool affine = c.at("affine");
    if (affine) { w.upload(bytes("weight").data()); b.upload(bytes("bias").data()); }
    const vt::LayerNormArgs args{c.at("eps").get<float>()};
    const auto contract = manifest.at("contract").at(name);
    const std::string label = std::to_string(number++) + ":" + name + ":" + std::to_string(width);
    vt::LayerNorm(gpu.q, y.tensor, x.tensor, affine ? &w.tensor : nullptr, affine ? &b.tensor : nullptr, args);
    Compare(y, expected, contract.at("rel_l2"), contract.at("max_abs"), label);
    const auto first = y.download();
    vt::LayerNorm(gpu.q, y.tensor, x.tensor, affine ? &w.tensor : nullptr, affine ? &b.tensor : nullptr, args);
    xpu_test::SameBytes(y.download(), first);
    vt::LayerNorm(gpu.q, x.tensor, x.tensor, affine ? &w.tensor : nullptr, affine ? &b.tensor : nullptr, args);
    xpu_test::SameBytes(x.download(), first);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision GELU: pinned variants, storage dtypes, tails and alias") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::OpRegistered(vt::OpId::kGeluTanh, vt::DeviceType::kXPU));
  REQUIRE(vt::OpRegistered(vt::OpId::kGeluErf, vt::DeviceType::kXPU));
  const std::string dir = XPU_VISION_FIXTURE_DIR;
  const auto manifest = Document(dir + "/gelu.json");
  size_t number = 0;
  for (const auto& c : manifest.at("cases")) {
    const auto name = c.at("dtype").get<std::string>();
    const auto dtype = name == "float16" ? DType::kF16 : name == "bfloat16" ? DType::kBF16 : DType::kF32;
    const int64_t rows = c.at("shape")[0], width = c.at("shape")[1];
    const auto bytes = [&](const char* field) {
      return Read(dir + "/" + c.at("files").at(field).at("file").get<std::string>());
    };
    const auto input = bytes("input"), expected = bytes("expected");
    const bool tanh = c.at("approximate") == "tanh";
    const vt::ReluFn op = tanh ? &vt::GeluTanh : &vt::GeluErf;
    const auto contract = manifest.at("contract").at(name);
    const std::string label = std::to_string(number++) + ":" + name;
    Buffer x(gpu.q, dtype, {rows, width}), y(gpu.q, dtype, {rows, width});
    REQUIRE(x.bytes == input.size()); REQUIRE(y.bytes == expected.size());
    x.upload(input.data());
    op(gpu.q, y.tensor, x.tensor);
    Compare(y, expected, contract.at("rel_l2"), contract.at("max_abs"), label,
            tanh ? "GELU_TANH" : "GELU_ERF");
    const auto first = y.download();
    op(gpu.q, y.tensor, x.tensor);
    xpu_test::SameBytes(y.download(), first);
    op(gpu.q, x.tensor, x.tensor);
    xpu_test::SameBytes(x.download(), first);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision GELU: invalid shape, rank, stride, dtype and device refuse") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer x(gpu.q, DType::kF16, {3, 37}), y(gpu.q, DType::kF16, {3, 37});
  for (const vt::ReluFn op : {&vt::GeluTanh, &vt::GeluErf}) {
    auto wrong = y.tensor; wrong.shape[1] -= 1;
    CHECK_THROWS(op(gpu.q, wrong, x.tensor));
    wrong = x.tensor; wrong.rank = 0;
    CHECK_THROWS(op(gpu.q, y.tensor, wrong));
    wrong = x.tensor; wrong.stride[0] += 1;
    CHECK_THROWS(op(gpu.q, y.tensor, wrong));
    wrong = y.tensor; wrong.dtype = DType::kI32;
    CHECK_THROWS(op(gpu.q, wrong, x.tensor));
    wrong = x.tensor; wrong.device = {vt::DeviceType::kCPU, 0};
    CHECK_THROWS(op(gpu.q, y.tensor, wrong));
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision GELU: partial overlap preserves unread input") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const std::string dir = XPU_VISION_FIXTURE_DIR;
  const auto manifest = Document(dir + "/gelu.json");
  for (const int index : {0, 3}) {
    const auto& c = manifest.at("cases")[index];
    auto bytes = [&](const char* field) {
      return Read(dir + "/" + c.at("files").at(field).at("file").get<std::string>());
    };
    const auto input = bytes("input");
    auto expected = bytes("expected"); expected.resize(37 * sizeof(uint16_t));
    Buffer storage(gpu.q, DType::kF16, {38}), y(gpu.q, DType::kF16, {37});
    storage.upload(input.data());
    auto x = vt::Tensor::Contiguous(storage.tensor.data, DType::kF16, gpu.q.device, {37});
    auto out = vt::Tensor::Contiguous(static_cast<unsigned char*>(storage.tensor.data) + 2,
                                     DType::kF16, gpu.q.device, {37});
    const vt::ReluFn op = index == 0 ? &vt::GeluTanh : &vt::GeluErf;
    op(gpu.q, out, x);
    vt::Copy(gpu.q, y.tensor, out);
    Compare(y, expected, 3e-4, 0.002, "partial:" + std::to_string(index), "GELU");
    const auto after = storage.download();
    CHECK(after[0] == input[0]); CHECK(after[1] == input[1]);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("vision GELU: existing CPU output dtype contract is preserved") {
  xpu_test::Queue cpu(vt::DeviceType::kCPU);
  const std::string dir = XPU_VISION_FIXTURE_DIR;
  const auto manifest = Document(dir + "/gelu.json");
  for (const auto& c : manifest.at("cases")) {
    const auto name = c.at("dtype").get<std::string>();
    const auto dtype = name == "float16" ? DType::kF16 : name == "bfloat16" ? DType::kBF16 : DType::kF32;
    const int64_t rows = c.at("shape")[0], width = c.at("shape")[1];
    Buffer x(cpu.q, dtype, {rows, width}), y(cpu.q, dtype, {rows, width});
    const vt::ReluFn op = c.at("approximate") == "tanh" ? &vt::GeluTanh : &vt::GeluErf;
    if (dtype == DType::kF16) {
      CHECK_THROWS(op(cpu.q, y.tensor, x.tensor));
      continue;
    }
    const auto input = Read(dir + "/" + c.at("files").at("input").at("file").get<std::string>());
    const auto expected = Read(dir + "/" + c.at("files").at("expected").at("file").get<std::string>());
    x.upload(input.data()); op(cpu.q, y.tensor, x.tensor);
    const auto contract = manifest.at("contract").at(name);
    Compare(y, expected, contract.at("rel_l2"), contract.at("max_abs"),
            "cpu:" + name + ":" + c.at("approximate").get<std::string>(), "GELU");
  }
}

TEST_CASE("XPU vision GELU: standalone reference on real merger boundary inputs" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR"}))) {
  const char* capture = exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");

  const std::string dir = capture;
  const auto manifest = Document(dir + "/gelu-real-merger.json");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(manifest.at("cases").size() == 2);
  size_t number = 0;
  for (const auto& c : manifest.at("cases")) {
    REQUIRE(c.at("input").at("dtype") == "torch.float16");
    REQUIRE((c.at("input").at("shape").get<std::vector<int64_t>>() ==
             std::vector<int64_t>{192, 4608}));
    const auto input = Read(dir + "/" + c.at("input").at("file").get<std::string>());
    const auto expected = Read(dir + "/" + c.at("expected").get<std::string>());
    Buffer x(gpu.q, DType::kF16, {192, 4608}), y(gpu.q, DType::kF16, {192, 4608});
    REQUIRE(x.bytes == input.size()); REQUIRE(y.bytes == expected.size());
    x.upload(input.data());
    vt::GeluErf(gpu.q, y.tensor, x.tensor);
    Compare(y, expected, manifest.at("contract").at("rel_l2"),
            manifest.at("contract").at("max_abs"), "merger:image" + std::to_string(number++),
            "GELU_ERF");
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision LayerNorm: reference-rounded FP32 variance regression") {
  const std::string dir = std::string(XPU_VISION_FIXTURE_DIR) + "/rounded-divide";
  const auto manifest = Document(dir + "/rounding.json");
  REQUIRE(manifest.at("schema") == 1);
  REQUIRE(manifest.at("cases").size() == 2);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  for (const auto& c : manifest.at("cases")) {
    INFO(c.at("label").get<std::string>());
    REQUIRE(c.at("shape") == json::array({1,1,1152}));
    REQUIRE(c.at("minimal_R_matches_original_full_shape_row").get<bool>());
    const auto bytes = [&](const char* role) {
      const auto name = c.at("files").at(role).at("file").get<std::string>();
      REQUIRE(name.find('/') == std::string::npos);
      return Read(dir + "/" + name);
    };
    const auto input = bytes("input"), weight = bytes("weight"), bias = bytes("bias"), expected = bytes("expected");
    Buffer x(gpu.q,DType::kF16,{1,1,1152}), y(gpu.q,DType::kF16,{1,1,1152}),
        w(gpu.q,DType::kF16,{1152}), b(gpu.q,DType::kF16,{1152});
    REQUIRE(input.size() == x.bytes); REQUIRE(expected.size() == y.bytes);
    REQUIRE(weight.size() == w.bytes); REQUIRE(bias.size() == b.bytes);
    x.upload(input.data()); w.upload(weight.data()); b.upload(bias.data());
    const vt::LayerNormArgs args{c.at("eps").get<float>()};
    vt::LayerNorm(gpu.q,y.tensor,x.tensor,&w.tensor,&b.tensor,args);
    xpu_test::SameBytes(y.download(),expected);
    vt::LayerNorm(gpu.q,x.tensor,x.tensor,&w.tensor,&b.tensor,args);
    xpu_test::SameBytes(x.download(),expected);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision LayerNorm: invalid rank, stride, affine shape and epsilon refuse") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer x(gpu.q, DType::kF16, {3, 37}), y(gpu.q, DType::kF16, {3, 37});
  Buffer w(gpu.q, DType::kF16, {36});
  CHECK_THROWS(vt::LayerNorm(gpu.q, y.tensor, x.tensor, &w.tensor, nullptr, {1e-6f}));
  CHECK_THROWS(vt::LayerNorm(gpu.q, y.tensor, x.tensor, nullptr, nullptr, {-1.0f}));
  auto strided = x.tensor; strided.stride[0] += 1;
  CHECK_THROWS(vt::LayerNorm(gpu.q, y.tensor, strided, nullptr, nullptr, {1e-6f}));
  auto wrong = y.tensor; wrong.shape[1] -= 1;
  CHECK_THROWS(vt::LayerNorm(gpu.q, wrong, x.tensor, nullptr, nullptr, {1e-6f}));
  auto scalar = x.tensor;
  scalar.rank = 0;
  CHECK_THROWS(vt::LayerNorm(gpu.q, scalar, scalar, nullptr, nullptr, {1e-6f}));
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision LayerNorm: actual executed tower boundary inputs" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR","VT_B70_VISION_MODEL_DIR"}))) {
  const char* capture = exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");
  const char* checkpoint = exl3_test::CaseExternalEnvironment("VT_B70_VISION_MODEL_DIR");

  const std::string dir = capture, model = checkpoint;
  const auto index = vllm::LoadSafetensorsIndex(model + "/model.safetensors.index.json");
  const auto manifest = Document(dir + "/vision-boundaries-capture.json");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  for (const std::string stage : {"block0-norm1", "block0-norm2", "merger-norm"}) {
    const std::string key = stage == "merger-norm" ? "model.visual.merger.norm" :
        stage == "block0-norm1" ? "model.visual.blocks.0.norm1" : "model.visual.blocks.0.norm2";
    auto file = vllm::SafetensorsFile::Open(model + "/" + index.at(key + ".weight"));
    auto convert = [&](const std::string& suffix) {
      const auto& stored = file.Get(key + suffix);
      REQUIRE(stored.dtype == "BF16"); REQUIRE(stored.shape == std::vector<int64_t>{1152});
      std::vector<uint16_t> half(1152);
      for (size_t i = 0; i < half.size(); ++i)
        half[i] = vt::F32ToF16(vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(stored.data + i * 2)));
      return half;
    };
    const auto weights = convert(".weight"), bias = convert(".bias");
    Buffer w(gpu.q, DType::kF16, {1152}), b(gpu.q, DType::kF16, {1152});
    w.upload(weights.data()); b.upload(bias.data());
    std::vector<json> inputs, outputs;
    for (const auto& entry : manifest.at("worker").at("captures")) {
      if (entry.at("name") == stage + "-input-0") inputs.push_back(entry);
      if (entry.at("name") == stage + "-output") outputs.push_back(entry);
    }
    REQUIRE(inputs.size() == 2); REQUIRE(outputs.size() == 2);
    for (size_t i = 0; i < inputs.size(); ++i) {
      REQUIRE(inputs[i].at("dtype") == "torch.float16");
      REQUIRE((inputs[i].at("shape").get<std::vector<int64_t>>() == std::vector<int64_t>{768, 1, 1152}));
      Buffer x(gpu.q, DType::kF16, {768, 1, 1152}), y(gpu.q, DType::kF16, {768, 1, 1152});
      const auto input = Read(dir + "/" + inputs[i].at("file").get<std::string>());
      const auto expected = Read(dir + "/" + outputs[i].at("file").get<std::string>());
      REQUIRE(x.bytes == input.size()); REQUIRE(y.bytes == expected.size()); x.upload(input.data());
      vt::LayerNorm(gpu.q, y.tensor, x.tensor, &w.tensor, &b.tensor, {1e-6f});
      // Declared before native comparison, matching the FP16 operator cases.
      Compare(y, expected, 3e-4, 0.002, stage + ":image" + std::to_string(i));
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision positions: executed production fixtures and repeatability") {
  const std::string dir = XPU_VISION_FIXTURE_DIR;
  const auto manifest = Document(dir + "/position.json");
  REQUIRE(manifest.at("route") == "triton_pos_embed_interpolate");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::OpRegistered(vt::OpId::kVisionPosEmbedInterpolate, vt::DeviceType::kXPU));
  CHECK(vt::GetReferenceTierHits() == 0);
  size_t i = 0;
  for (const auto& c : manifest.at("cases")) {
    const auto grid = c.at("grid").get<std::vector<int64_t>>();
    const int64_t side = c.at("side"), hidden = c.at("hidden"), merge = c.at("merge");
    Buffer table(gpu.q, DType::kF16, {side*side, hidden});
    Buffer y(gpu.q, DType::kF16, {grid[0]*grid[1]*grid[2], hidden});
    const auto input = Read(dir + "/" + c.at("files").at("table").at("file").get<std::string>());
    const auto expected = Read(dir + "/" + c.at("files").at("expected").at("file").get<std::string>());
    REQUIRE(table.bytes == input.size()); REQUIRE(y.bytes == expected.size());
    table.upload(input.data());
    const vt::VisionPosEmbedArgs args{grid[0], grid[1], grid[2], side, merge};
    vt::VisionPosEmbedInterpolate(gpu.q, y.tensor, table.tensor, args);
    Compare(y, expected, manifest.at("contract").at("rel_l2"),
            manifest.at("contract").at("max_abs"), std::to_string(i++), "POSITION");
    const auto first = y.download();
    vt::VisionPosEmbedInterpolate(gpu.q, y.tensor, table.tensor, args);
    xpu_test::SameBytes(first, y.download());
    xpu_test::SameBytes(first, expected);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision positions: grid, dtype, strides, aliases and device validation") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer table(gpu.q, DType::kF16, {4, 3}), y(gpu.q, DType::kF16, {4, 3});
  const vt::VisionPosEmbedArgs good{1,2,2,2,2};
  const std::array<vt::VisionPosEmbedArgs, 8> invalid{{
      {0,2,2,2,2}, {1,0,2,2,2}, {1,2,2,0,2}, {1,2,2,2,0},
      {1,2,2,2,3}, {2,2,2,2,2}, {1,2,2,3,2}, {1,32770,2,2,2}}};
  for (const auto& args : invalid)
    CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,y.tensor,table.tensor,args), std::runtime_error);
  auto bad = table.tensor; bad.dtype = DType::kBF16;
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,y.tensor,bad,good), std::runtime_error);
  bad = y.tensor; bad.dtype = DType::kF32;
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,bad,table.tensor,good), std::runtime_error);
  bad = table.tensor; bad.stride[0] += 1;
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,y.tensor,bad,good), std::runtime_error);
  bad = y.tensor; bad.stride[1] = 2;
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,bad,table.tensor,good), std::runtime_error);
  bad = table.tensor; bad.device.index += 1;
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,y.tensor,bad,good), std::runtime_error);
  bad = table.tensor; bad.rank = 1;
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,y.tensor,bad,good), std::runtime_error);
  bad = y.tensor; bad.shape[1] = 2;
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,bad,table.tensor,good), std::runtime_error);
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,table.tensor,table.tensor,good), std::runtime_error);
  bad = table.tensor; bad.data = static_cast<char*>(bad.data) + 2;
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,bad,table.tensor,good), std::runtime_error);
  bad = table.tensor; bad.shape[0] = std::numeric_limits<int64_t>::max();
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(gpu.q,y.tensor,bad,good), std::runtime_error);
  auto cpu = vt::Queue{{vt::DeviceType::kCPU,0}, nullptr};
  CHECK_THROWS_AS(vt::VisionPosEmbedInterpolate(cpu,y.tensor,table.tensor,good), std::runtime_error);
  Buffer adjacent(gpu.q,DType::kF16,{24});
  adjacent.put(std::vector<float>(24,1));
  auto left = vt::Tensor::Contiguous(adjacent.tensor.data,DType::kF16,gpu.q.device,{4,3});
  auto right = left; right.data = static_cast<char*>(left.data) + 24;
  vt::VisionPosEmbedInterpolate(gpu.q,right,left,good);
  CHECK(adjacent.floats() == std::vector<float>(24,1));
}

TEST_CASE("XPU vision positions: actual worker position boundaries" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR"}))) {
  const char* env = exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");

  const std::string dir = env;
  const auto manifest = Document(dir + "/position-real-replays.json");
  REQUIRE(manifest.at("route") == "triton_pos_embed_interpolate");
  REQUIRE(manifest.at("cases").size() == 2);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  CHECK(vt::GetReferenceTierHits() == 0);
  Buffer table(gpu.q, DType::kF16, {2304, 1152}), y(gpu.q,DType::kF16,{768,1152});
  const auto input = Read(dir + "/" + manifest.at("table").at("file").get<std::string>());
  REQUIRE(input.size() == table.bytes); table.upload(input.data());
  const auto boundaries = Document(dir + "/vision-boundaries-capture.json");
  std::vector<json> patches, block_inputs;
  for (const auto& entry : boundaries.at("worker").at("captures")) {
    if (entry.at("name") == "patch-output") patches.push_back(entry);
    if (entry.at("name") == "block0-norm1-input-0") block_inputs.push_back(entry);
  }
  REQUIRE(patches.size() == 2); REQUIRE(block_inputs.size() == 2);
  const auto before = vt::xpu::GetMemoryInfo().allocated_bytes;
  for (const auto& c : manifest.at("cases")) {
    REQUIRE(c.at("standalone_matches_worker_bytes").get<bool>());
    const auto& expected = c.at("expected");
    REQUIRE(expected.at("dtype") == "torch.float16");
    REQUIRE((expected.at("shape").get<std::vector<int64_t>>() == std::vector<int64_t>{768,1152}));
    const auto reference = Read(dir + "/" + expected.at("file").get<std::string>());
    REQUIRE(reference.size() == y.bytes);
    vt::VisionPosEmbedInterpolate(gpu.q,y.tensor,table.tensor,{1,24,32,48,2});
    Compare(y,reference,3e-4,0.002,"real:image"+std::to_string(c.at("image").get<int>()),"POSITION");
    const auto first = y.download();
    vt::VisionPosEmbedInterpolate(gpu.q,y.tensor,table.tensor,{1,24,32,48,2});
    xpu_test::SameBytes(first,y.download());
    xpu_test::SameBytes(first,reference);
    // The patch projection was independently qualified. Compose its captured
    // result with native positions/Add and compare the actual first norm input.
    const auto i = c.at("image").get<size_t>();
    REQUIRE(i < 2);
    REQUIRE(patches[i].at("dtype") == "torch.float16");
    REQUIRE(block_inputs[i].at("dtype") == "torch.float16");
    Buffer patch(gpu.q,DType::kF16,{768,1152}), added(gpu.q,DType::kF16,{768,1152});
    const auto patch_bytes = Read(dir + "/" + patches[i].at("file").get<std::string>());
    const auto norm_input = Read(dir + "/" + block_inputs[i].at("file").get<std::string>());
    REQUIRE(patch.bytes == patch_bytes.size()); REQUIRE(added.bytes == norm_input.size());
    patch.upload(patch_bytes.data());
    vt::Add(gpu.q,added.tensor,patch.tensor,y.tensor);
    Compare(added,norm_input,0,0,"patch-add:image"+std::to_string(i),"POSITION");
    xpu_test::SameBytes(added.download(),norm_input);
  }
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes == before);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision QKV: executed split fixtures preserve storage bits") {
  const std::string dir = XPU_VISION_FIXTURE_DIR;
  const auto manifest = Document(dir + "/qkv.json");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::OpRegistered(vt::OpId::kQkvSplit,vt::DeviceType::kXPU));
  for (const auto& c : manifest.at("cases")) {
    const auto name = c.at("dtype").get<std::string>();
    const auto dtype = name == "float16" ? DType::kF16 : name == "bfloat16" ? DType::kBF16 : DType::kF32;
    const int64_t rows = c.at("rows");
    const auto widths = c.at("widths").get<std::vector<int64_t>>();
    REQUIRE(widths.size() == 3);
    Buffer input(gpu.q,dtype,{rows,widths[0]+widths[1]+widths[2]});
    Buffer q(gpu.q,dtype,{rows,widths[0]}), k(gpu.q,dtype,{rows,widths[1]}), v(gpu.q,dtype,{rows,widths[2]});
    const auto source = Read(dir + "/" + c.at("files").at("input").at("file").get<std::string>());
    REQUIRE(input.bytes == source.size()); input.upload(source.data());
    auto qview = q.tensor.View({rows,1,widths[0]});
    vt::QkvSplit(gpu.q,qview,k.tensor,v.tensor,input.tensor);
    for (const auto& entry : {std::pair{"q",&q},std::pair{"k",&k},std::pair{"v",&v}})
      xpu_test::SameBytes(entry.second->download(),Read(dir + "/" + c.at("files").at(entry.first).at("file").get<std::string>()));
    const auto first_q=q.download(),first_k=k.download(),first_v=v.download();
    vt::QkvSplit(gpu.q,qview,k.tensor,v.tensor,input.tensor);
    xpu_test::SameBytes(q.download(),first_q); xpu_test::SameBytes(k.download(),first_k); xpu_test::SameBytes(v.download(),first_v);
    xpu_test::SameBytes(input.download(),source);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision QKV: aliases, metadata and legacy CPU dtype contract") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer input(gpu.q,DType::kF16,{2,9}), q(gpu.q,DType::kF16,{2,3}),
      k(gpu.q,DType::kF16,{2,3}), v(gpu.q,DType::kF16,{2,3});
  input.put({1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18});
  const auto initial = input.download();
  const std::array<vt::Tensor,3> original{q.tensor,k.tensor,v.tensor};
  for (size_t i=0;i<3;++i) {
    auto views=original; views[i].data=static_cast<char*>(input.tensor.data)+2;
    CHECK_THROWS_AS(vt::QkvSplit(gpu.q,views[0],views[1],views[2],input.tensor),std::runtime_error);
  }
  for (size_t a=0;a<3;++a) for (size_t b=a+1;b<3;++b) {
    auto views=original; views[b].data=static_cast<char*>(views[a].data)+2;
    CHECK_THROWS_AS(vt::QkvSplit(gpu.q,views[0],views[1],views[2],input.tensor),std::runtime_error);
  }
  xpu_test::SameBytes(input.download(),initial);
  auto bad=input.tensor; bad.stride[1]=2;
  CHECK_THROWS_AS(vt::QkvSplit(gpu.q,q.tensor,k.tensor,v.tensor,bad),std::runtime_error);
  bad=q.tensor; bad.stride[1]=2;
  CHECK_THROWS_AS(vt::QkvSplit(gpu.q,bad,k.tensor,v.tensor,input.tensor),std::runtime_error);
  bad=k.tensor; bad.dtype=DType::kF32;
  CHECK_THROWS_AS(vt::QkvSplit(gpu.q,q.tensor,bad,v.tensor,input.tensor),std::runtime_error);
  bad=v.tensor; bad.device.index+=1;
  CHECK_THROWS_AS(vt::QkvSplit(gpu.q,q.tensor,k.tensor,bad,input.tensor),std::runtime_error);
  bad=input.tensor; bad.shape[1]=8;
  CHECK_THROWS_AS(vt::QkvSplit(gpu.q,q.tensor,k.tensor,v.tensor,bad),std::runtime_error);
  auto empty=q.tensor; empty.shape[1]=0; empty.stride[0]=0;
  bad=input.tensor; bad.shape[1]=6;
  CHECK_THROWS_AS(vt::QkvSplit(gpu.q,empty,k.tensor,v.tensor,bad),std::runtime_error);
  Buffer adjacent(gpu.q,DType::kF16,{18});
  auto qa=vt::Tensor::Contiguous(adjacent.tensor.data,DType::kF16,gpu.q.device,{2,3});
  auto ka=qa,va=qa; ka.data=static_cast<char*>(qa.data)+12; va.data=static_cast<char*>(qa.data)+24;
  vt::QkvSplit(gpu.q,qa,ka,va,input.tensor);
  CHECK((adjacent.floats()==std::vector<float>{1,2,3,10,11,12,4,5,6,13,14,15,7,8,9,16,17,18}));
  xpu_test::Queue cpu(vt::DeviceType::kCPU);
  for (const auto dtype : {DType::kBF16,DType::kF32,DType::kF16}) {
    Buffer a(cpu.q,dtype,{2,3}), cq(cpu.q,dtype,{2,1}), ck(cpu.q,dtype,{2,1}), cv(cpu.q,dtype,{2,1});
    a.put({1,2,3,4,5,6});
    if (dtype==DType::kF16) {
      CHECK_THROWS_AS(vt::QkvSplit(cpu.q,cq.tensor,ck.tensor,cv.tensor,a.tensor),std::runtime_error);
    } else {
      vt::QkvSplit(cpu.q,cq.tensor,ck.tensor,cv.tensor,a.tensor);
      CHECK((cq.floats()==std::vector<float>{1,4})); CHECK((ck.floats()==std::vector<float>{2,5})); CHECK((cv.floats()==std::vector<float>{3,6}));
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision QKV: real first-block FP16 projection and split" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR","VT_B70_VISION_MODEL_DIR"}))) {
  const char* reference_env=exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");
  const char* model_env=exl3_test::CaseExternalEnvironment("VT_B70_VISION_MODEL_DIR");

  const std::string dir=reference_env,model=model_env;
  const auto manifest=Document(dir+"/qkv-real-replays.json");
  REQUIRE(manifest.at("cases").size()==2);
  const std::string key=manifest.at("key");
  const auto index=vllm::LoadSafetensorsIndex(model+"/model.safetensors.index.json");
  auto file=vllm::SafetensorsFile::Open(model+"/"+index.at(key+".weight"));
  const auto convert=[&](const char* suffix,const std::vector<int64_t>& shape) {
    const auto& tensor=file.Get(key+suffix);
    REQUIRE(tensor.dtype=="BF16"); REQUIRE(tensor.shape==shape);
    std::vector<uint16_t> half(tensor.nbytes/2);
    for (size_t i=0;i<half.size();++i) half[i]=vt::F32ToF16(vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(tensor.data+2*i)));
    return half;
  };
  const auto weight=convert(".weight",{3456,1152}),bias=convert(".bias",{3456});
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer w(gpu.q,DType::kF16,{3456,1152}),b(gpu.q,DType::kF16,{3456});
  w.upload(weight.data()); b.upload(bias.data());
  Buffer a(gpu.q,DType::kF16,{768,1152}),merged(gpu.q,DType::kF16,{768,3456}),
      q(gpu.q,DType::kF16,{768,16,72}),k(gpu.q,DType::kF16,{768,16,72}),v(gpu.q,DType::kF16,{768,16,72});
  const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
  for (const auto& c : manifest.at("cases")) {
    REQUIRE(c.at("input").at("dtype")=="torch.float16");
    REQUIRE((c.at("input").at("shape").get<std::vector<int64_t>>()==std::vector<int64_t>{768,1,1152}));
    const auto input=Read(dir+"/"+c.at("input").at("file").get<std::string>());
    REQUIRE(input.size()==a.bytes); a.upload(input.data());
    vt::MatmulDenseF16(gpu.q,merged.tensor,a.tensor,w.tensor,&b.tensor);
    vt::QkvSplit(gpu.q,q.tensor,k.tensor,v.tensor,merged.tensor);
    const auto image=c.at("image").get<int>();
    for (const auto& entry : {std::pair{"merged",&merged},std::pair{"q",&q},std::pair{"k",&k},std::pair{"v",&v}}) {
      const auto expected=Read(dir+"/"+c.at("files").at(entry.first).at("file").get<std::string>());
      REQUIRE(expected.size()==entry.second->bytes);
      Compare(*entry.second,expected,manifest.at("contract").at("rel_l2"),
              manifest.at("contract").at("max_abs"),std::string(entry.first)+":image"+std::to_string(image),"QKV");
    }
    const auto first=merged.download(),first_q=q.download(),first_k=k.download(),first_v=v.download();
    vt::MatmulDenseF16(gpu.q,merged.tensor,a.tensor,w.tensor,&b.tensor);
    vt::QkvSplit(gpu.q,q.tensor,k.tensor,v.tensor,merged.tensor);
    xpu_test::SameBytes(merged.download(),first); xpu_test::SameBytes(q.download(),first_q);
    xpu_test::SameBytes(k.download(),first_k); xpu_test::SameBytes(v.download(),first_v);
  }
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision RoPE apply: executed FP16 operator fixtures") {
  const std::string dir=XPU_VISION_FIXTURE_DIR;
  const auto reference=Document(dir+"/vision-rope.json");
  REQUIRE(reference.at("is_neox_style")==true);
  REQUIRE(reference.at("enable_fp32_compute")==false);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::OpRegistered(vt::OpId::kVisionRopeApply,vt::DeviceType::kXPU));
  size_t number=0;
  for (const auto& c : reference.at("cases")) {
    const int64_t t=c.at("shape")[0],h=c.at("shape")[1],d=c.at("shape")[2];
    Buffer q(gpu.q,DType::kF16,{t,h,d}),k(gpu.q,DType::kF16,{t,h,d}),cache(gpu.q,DType::kF16,{t,d});
    const auto bytes=[&](const char* name) { return Read(dir+"/"+c.at("files").at(name).at("file").get<std::string>()); };
    const auto qi=bytes("q"),ki=bytes("k"),coeff=bytes("cache");
    REQUIRE(qi.size()==q.bytes); REQUIRE(ki.size()==k.bytes); REQUIRE(coeff.size()==cache.bytes);
    q.upload(qi.data()); k.upload(ki.data()); cache.upload(coeff.data());
    const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
    vt::VisionRopeApply(gpu.q,q.tensor,k.tensor,cache.tensor);
    for (const auto& entry : {std::pair{"q_out",&q},std::pair{"k_out",&k}}) {
      const auto expected=bytes(entry.first);
      Compare(*entry.second,expected,0,0,std::string(entry.first)+":fixture"+std::to_string(number),"ROPE");
      xpu_test::SameBytes(entry.second->download(),expected);
    }
    const auto firstq=q.download(),firstk=k.download();
    q.upload(qi.data()); k.upload(ki.data());
    vt::VisionRopeApply(gpu.q,q.tensor,k.tensor,cache.tensor);
    xpu_test::SameBytes(q.download(),firstq); xpu_test::SameBytes(k.download(),firstk);
    xpu_test::SameBytes(cache.download(),coeff);
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
    ++number;
  }
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision RoPE apply: shapes, aliases and immutable coefficients") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer q(gpu.q,DType::kF16,{2,1,4}),k(gpu.q,DType::kF16,{2,1,4}),cache(gpu.q,DType::kF16,{2,4});
  q.put(std::vector<float>(8,1)); k.put(std::vector<float>(8,2)); cache.put(std::vector<float>(8,1));
  const auto qi=q.download(),ki=k.download(),coeff=cache.download();
  for (int index=0;index<3;++index) {
    auto a=q.tensor,b=k.tensor,c=cache.tensor;
    (index==0?a:index==1?b:c).dtype=DType::kBF16;
    CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,a,b,c),std::runtime_error);
    a=q.tensor;b=k.tensor;c=cache.tensor;
    (index==0?a:index==1?b:c).device.index+=1;
    CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,a,b,c),std::runtime_error);
    a=q.tensor;b=k.tensor;c=cache.tensor;
    (index==0?a:index==1?b:c).stride[0]+=1;
    CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,a,b,c),std::runtime_error);
    a=q.tensor;b=k.tensor;c=cache.tensor;
    (index==0?a:index==1?b:c).rank=1;
    CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,a,b,c),std::runtime_error);
  }
  auto bad=k.tensor; bad.shape[0]=1;
  CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,q.tensor,bad,cache.tensor),std::runtime_error);
  bad=k.tensor; bad.shape[1]=2;
  CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,q.tensor,bad,cache.tensor),std::runtime_error);
  bad=cache.tensor;bad.shape[1]=3;
  CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,q.tensor,k.tensor,bad),std::runtime_error);
  auto a=q.tensor,b=k.tensor,c=cache.tensor;
  a.shape[2]=b.shape[2]=c.shape[1]=3;
  CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,a,b,c),std::runtime_error);
  a=q.tensor;b=k.tensor;c=cache.tensor;
  a.shape[0]=b.shape[0]=c.shape[0]=std::numeric_limits<int64_t>::max();
  CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,a,b,c),std::runtime_error);
  for (int pair=0;pair<3;++pair) for (int offset : {0,2}) {
    a=q.tensor;b=k.tensor;c=cache.tensor;
    auto& dst=pair==0?b:c;
    const auto& src=pair==2?b:a;
    dst.data=static_cast<char*>(src.data)+offset;
    CHECK_THROWS_AS(vt::VisionRopeApply(gpu.q,a,b,c),std::runtime_error);
  }
  auto cpu=vt::Queue{{vt::DeviceType::kCPU,0},nullptr};
  CHECK_THROWS_AS(vt::VisionRopeApply(cpu,q.tensor,k.tensor,cache.tensor),std::runtime_error);
  xpu_test::SameBytes(q.download(),qi); xpu_test::SameBytes(k.download(),ki); xpu_test::SameBytes(cache.download(),coeff);
  Buffer adjacent(gpu.q,DType::kF16,{24});
  std::vector<float> values{1,1,1,1,1,1,1,1,2,2,2,2,2,2,2,2,1,1,0,0,1,1,0,0};
  adjacent.put(values);
  a=vt::Tensor::Contiguous(adjacent.tensor.data,DType::kF16,gpu.q.device,{2,1,4});
  b=a;b.data=static_cast<char*>(a.data)+16;
  c=vt::Tensor::Contiguous(static_cast<char*>(a.data)+32,DType::kF16,gpu.q.device,{2,4});
  vt::VisionRopeApply(gpu.q,a,b,c);
  CHECK(adjacent.floats()==values);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision RoPE apply: GPU spatial cache and real-input QK" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR"}))) {
  const char* env=exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");

  const std::string dir=env;
  const auto reference=Document(dir+"/vision-rope-reference.json");
  REQUIRE(reference.at("cases").size()==2);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer positions(gpu.q,DType::kI32,{32}),base32(gpu.q,DType::kF32,{32,36}),
      base16(gpu.q,DType::kF16,{32,36}),cache(gpu.q,DType::kF16,{768,72}),
      q(gpu.q,DType::kF16,{768,16,72}),k(gpu.q,DType::kF16,{768,16,72});
  std::array<int32_t,32> ids{};for(size_t i=0;i<ids.size();++i)ids[i]=i;
  positions.upload(ids.data());
  vt::RopeArgs args;args.rotary_dim=36;args.linear_scaling_factor=1;args.fp16_intermediates=true;
  vt::RopeCosSinCache(gpu.q,base32.tensor,positions.tensor,args);
  vt::CastF16(gpu.q,base16.tensor,base32.tensor);
  vt::VisionRopeGrid(gpu.q,cache.tensor,base16.tensor,{1,24,32,2});
  const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
  for (const auto& c : reference.at("cases")) {
    const auto qi=Read(dir+"/"+c.at("q_in").at("file").get<std::string>());
    const auto ki=Read(dir+"/"+c.at("k_in").at("file").get<std::string>());
    REQUIRE(qi.size()==q.bytes);REQUIRE(ki.size()==k.bytes);
    q.upload(qi.data());k.upload(ki.data());
    vt::VisionRopeApply(gpu.q,q.tensor,k.tensor,cache.tensor);
    for (const auto& entry : {std::pair{"q_out",&q},std::pair{"k_out",&k}}) {
      const auto expected=Read(dir+"/"+c.at(entry.first).at("file").get<std::string>());
      Compare(*entry.second,expected,0,0,std::string(entry.first)+":aligned-image"+std::to_string(c.at("image").get<int>()),"ROPE");
      xpu_test::SameBytes(entry.second->download(),expected);
    }
    const auto firstq=q.download(),firstk=k.download();
    q.upload(qi.data());k.upload(ki.data());
    vt::VisionRopeApply(gpu.q,q.tensor,k.tensor,cache.tensor);
    xpu_test::SameBytes(q.download(),firstq);xpu_test::SameBytes(k.download(),firstk);
  }
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision RoPE grid: merge ordering, frames and storage bits") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::OpRegistered(vt::OpId::kVisionRopeGrid,vt::DeviceType::kXPU));
  const std::array<vt::VisionRopeGridArgs,4> cases{{
      {1,4,6,2},{2,2,4,2},{2,6,3,3},{3,1,1,1}}};
  constexpr int64_t f=5;
  const std::array<uint16_t,8> patterns{{0,0x8000,0x7c00,0xfc00,0x7e21,0x7c01,1,0x3c00}};
  for (const auto& args : cases) {
    const int64_t p=std::max(args.h,args.w),tokens=args.t*args.h*args.w;
    std::vector<uint16_t> source(p*2*f),expected;
    for (size_t i=0;i<source.size();++i) source[i]=patterns[i%patterns.size()] ^ uint16_t(i/8);
    // Independent forward traversal of frame -> merged block -> local pixel.
    for (int64_t frame=0;frame<args.t;++frame)
      for (int64_t top=0;top<args.h;top+=args.merge_size)
        for (int64_t left=0;left<args.w;left+=args.merge_size)
          for (int64_t y=top;y<top+args.merge_size;++y)
            for (int64_t x=left;x<left+args.merge_size;++x)
              for (int64_t half=0;half<2;++half)
                for (int64_t axis : {y,x})
                  for (int64_t d=0;d<f;++d) expected.push_back(source[axis*2*f+half*f+d]);
    Buffer base(gpu.q,DType::kF16,{p,2*f}),out(gpu.q,DType::kF16,{tokens,4*f});
    base.upload(source.data());
    const auto* begin=reinterpret_cast<const unsigned char*>(expected.data());
    const std::vector<unsigned char> bytes(begin,begin+expected.size()*2);
    const auto original=base.download();
    const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
    vt::VisionRopeGrid(gpu.q,out.tensor,base.tensor,args);
    xpu_test::SameBytes(out.download(),bytes);
    vt::VisionRopeGrid(gpu.q,out.tensor,base.tensor,args);
    xpu_test::SameBytes(out.download(),bytes);
    xpu_test::SameBytes(base.download(),original);
    CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
  }
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision RoPE grid: bounded shapes and alias validation") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer base(gpu.q,DType::kF16,{2,4}),out(gpu.q,DType::kF16,{4,8});
  const vt::VisionRopeGridArgs good{1,2,2,2};
  const std::array<vt::VisionRopeGridArgs,9> invalid{{
      {0,2,2,2},{1,0,2,2},{1,2,0,2},{1,2,2,0},{1,2,2,3},
      {2,2,2,2},{1,4,2,2},{1,32770,2,2},{1,2,32770,2}}};
  for (const auto& args : invalid)
    CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,out.tensor,base.tensor,args),std::runtime_error);
  auto bad=base.tensor; bad.dtype=DType::kBF16;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,out.tensor,bad,good),std::runtime_error);
  bad=out.tensor; bad.dtype=DType::kF32;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,bad,base.tensor,good),std::runtime_error);
  bad=base.tensor; bad.rank=1;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,out.tensor,bad,good),std::runtime_error);
  bad=out.tensor; bad.shape[1]=7;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,bad,base.tensor,good),std::runtime_error);
  bad=base.tensor; bad.shape[1]=3;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,out.tensor,bad,good),std::runtime_error);
  bad=base.tensor; bad.stride[0]+=1;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,out.tensor,bad,good),std::runtime_error);
  bad=out.tensor; bad.stride[1]=2;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,bad,base.tensor,good),std::runtime_error);
  bad=base.tensor; bad.device.index+=1;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,out.tensor,bad,good),std::runtime_error);
  bad=out.tensor; bad.device.index+=1;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,bad,base.tensor,good),std::runtime_error);
  bad=out.tensor; bad.shape[0]=std::numeric_limits<int64_t>::max();
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,bad,base.tensor,good),std::runtime_error);
  bad=base.tensor; bad.data=out.tensor.data;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,out.tensor,bad,good),std::runtime_error);
  bad=base.tensor; bad.data=static_cast<char*>(out.tensor.data)+2;
  CHECK_THROWS_AS(vt::VisionRopeGrid(gpu.q,out.tensor,bad,good),std::runtime_error);
  auto cpu=vt::Queue{{vt::DeviceType::kCPU,0},nullptr};
  CHECK_THROWS_AS(vt::VisionRopeGrid(cpu,out.tensor,base.tensor,good),std::runtime_error);
  Buffer adjacent(gpu.q,DType::kF16,{40}); adjacent.put(std::vector<float>(40,1));
  auto src=vt::Tensor::Contiguous(adjacent.tensor.data,DType::kF16,gpu.q.device,{2,4});
  auto dst=vt::Tensor::Contiguous(static_cast<char*>(src.data)+16,DType::kF16,gpu.q.device,{4,8});
  vt::VisionRopeGrid(gpu.q,dst,src,good);
  CHECK(adjacent.floats()==std::vector<float>(40,1));
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision RoPE grid: GPU base through captured worker coefficients" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR"}))) {
  const char* env=exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");

  const std::string dir=env;
  const auto reference=Document(dir+"/vision-rope-reference.json");
  const auto cos=Read(dir+"/"+reference.at("combined_cos").at("file").get<std::string>());
  const auto sin=Read(dir+"/"+reference.at("combined_sin").at("file").get<std::string>());
  REQUIRE(cos.size()==768*36*2); REQUIRE(sin.size()==cos.size());
  std::vector<unsigned char> expected;
  for (size_t row=0;row<768;++row) {
    expected.insert(expected.end(),cos.begin()+row*72,cos.begin()+(row+1)*72);
    expected.insert(expected.end(),sin.begin()+row*72,sin.begin()+(row+1)*72);
  }
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer positions(gpu.q,DType::kI32,{32}),cache(gpu.q,DType::kF32,{32,36}),
      base(gpu.q,DType::kF16,{32,36}),out(gpu.q,DType::kF16,{768,72});
  std::array<int32_t,32> ids{}; for (size_t i=0;i<ids.size();++i) ids[i]=i;
  positions.upload(ids.data());
  vt::RopeArgs args; args.rotary_dim=36; args.linear_scaling_factor=1; args.fp16_intermediates=true;
  const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
  vt::RopeCosSinCache(gpu.q,cache.tensor,positions.tensor,args);
  vt::CastF16(gpu.q,base.tensor,cache.tensor);
  vt::VisionRopeGrid(gpu.q,out.tensor,base.tensor,{1,24,32,2});
  Compare(out,expected,0,0,"spatial-grid","ROPE");
  xpu_test::SameBytes(out.download(),expected);
  vt::VisionRopeGrid(gpu.q,out.tensor,base.tensor,{1,24,32,2});
  xpu_test::SameBytes(out.download(),expected);
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision RoPE: executing reference base cache" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR"}))) {
  const char* env=exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");

  const std::string dir=env;
  const auto reference=Document(dir+"/vision-rope-reference.json");
  REQUIRE(reference.at("initialization").at(0).at("computed_device")=="xpu:0");
  REQUIRE((reference.at("base_cache_f32").at("shape").get<std::vector<int64_t>>()==std::vector<int64_t>{32,36}));
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer positions(gpu.q,DType::kI32,{32}),cache(gpu.q,DType::kF32,{32,36}),half(gpu.q,DType::kF16,{32,36});
  std::array<int32_t,32> ids{}; for (size_t i=0;i<ids.size();++i) ids[i]=i;
  positions.upload(ids.data());
  vt::RopeArgs args; args.rotary_dim=36; args.linear_scaling_factor=1; args.fp16_intermediates=true;
  vt::RopeCosSinCache(gpu.q,cache.tensor,positions.tensor,args);
  const auto f32=Read(dir+"/"+reference.at("base_cache_f32").at("file").get<std::string>());
  Compare(cache,f32,3e-6,1e-5,"base-f32","ROPE");
  vt::CastF16(gpu.q,half.tensor,cache.tensor);
  const auto f16=Read(dir+"/"+reference.at("base_cache_f16").at("file").get<std::string>());
  Compare(half,f16,0,0,"base-f16","ROPE");
  xpu_test::SameBytes(half.download(),f16);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision RoPE: real-input rotations with captured coefficients" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR"}))) {
  const char* env=exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");

  const std::string dir=env;
  const auto reference=Document(dir+"/vision-rope-reference.json");
  REQUIRE(reference.at("cases").size()==2);
  REQUIRE(reference.at("rotation_method").at("enable_fp32_compute")==false);
  const auto cos=Floats(Read(dir+"/"+reference.at("combined_cos").at("file").get<std::string>()),DType::kF16);
  const auto sin=Floats(Read(dir+"/"+reference.at("combined_sin").at("file").get<std::string>()),DType::kF16);
  REQUIRE(cos.size()==768*36); REQUIRE(sin.size()==768*36);
  // Diagnostic input loading only: widen captured FP16 coefficients exactly to
  // exercise the existing FP32-cache consumer. Production materialization is
  // still pending; this is not a host cache implementation for serving.
  std::vector<float> values(768*72);
  for (size_t t=0;t<768;++t) for (size_t d=0;d<36;++d) {
    values[t*72+d]=cos[t*36+d]; values[t*72+36+d]=sin[t*36+d];
  }
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer cache(gpu.q,DType::kF32,{768,72}),positions(gpu.q,DType::kI32,{768}),
      q(gpu.q,DType::kF16,{768,16,72}),k(gpu.q,DType::kF16,{768,16,72});
  cache.put(values);
  std::array<int32_t,768> ids{}; for (size_t t=0;t<ids.size();++t) ids[t]=t;
  positions.upload(ids.data());
  vt::RopeArgs args; args.rotary_dim=72; args.fp16_intermediates=true;
  const auto before=vt::xpu::GetMemoryInfo().allocated_bytes;
  for (const auto& c : reference.at("cases")) {
    const auto input_q=Read(dir+"/"+c.at("q_in").at("file").get<std::string>());
    const auto input_k=Read(dir+"/"+c.at("k_in").at("file").get<std::string>());
    REQUIRE(input_q.size()==q.bytes); REQUIRE(input_k.size()==k.bytes);
    q.upload(input_q.data()); k.upload(input_k.data());
    vt::RopeFromCache(gpu.q,q.tensor,&k.tensor,positions.tensor,cache.tensor,args);
    for (const auto& entry : {std::pair{"q_out",&q},std::pair{"k_out",&k}}) {
      const auto expected=Read(dir+"/"+c.at(entry.first).at("file").get<std::string>());
      REQUIRE(expected.size()==entry.second->bytes);
      Compare(*entry.second,expected,0,0,std::string(entry.first)+":image"+std::to_string(c.at("image").get<int>()),"ROPE");
      xpu_test::SameBytes(entry.second->download(),expected);
    }
    const auto first_q=q.download(),first_k=k.download();
    q.upload(input_q.data()); k.upload(input_k.data());
    vt::RopeFromCache(gpu.q,q.tensor,&k.tensor,positions.tensor,cache.tensor,args);
    xpu_test::SameBytes(q.download(),first_q); xpu_test::SameBytes(k.download(),first_k);
  }
  CHECK(vt::xpu::GetMemoryInfo().allocated_bytes==before);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU vision dense FP16: executed linear fixtures and repeatability") {
  const std::string dir = XPU_VISION_FIXTURE_DIR;
  const auto manifest = Document(dir + "/dense.json");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  REQUIRE(vt::OpRegistered(vt::OpId::kMatmulDenseF16, vt::DeviceType::kXPU));
  size_t number = 0;
  for (const auto& c : manifest.at("cases")) {
    const int64_t m = c.at("m"), k = c.at("k"), n = c.at("n");
    auto bytes = [&](const char* field) {
      return Read(dir + "/" + c.at("files").at(field).at("file").get<std::string>());
    };
    Buffer a(gpu.q, DType::kF16, {m, k}), w(gpu.q, DType::kF16, {n, k});
    Buffer b(gpu.q, DType::kF16, {n}), y(gpu.q, DType::kF16, {m, n});
    const auto input = bytes("input"), weight = bytes("weight"), expected = bytes("expected");
    REQUIRE(a.bytes == input.size()); REQUIRE(w.bytes == weight.size()); REQUIRE(y.bytes == expected.size());
    a.upload(input.data()); w.upload(weight.data());
    const bool affine = c.at("bias");
    if (affine) b.upload(bytes("bias").data());
    vt::MatmulDenseF16(gpu.q, y.tensor, a.tensor, w.tensor, affine ? &b.tensor : nullptr);
    Compare(y, expected, manifest.at("contract").at("rel_l2"),
            manifest.at("contract").at("max_abs"), std::to_string(number++), "DENSE");
    const auto first = y.download();
    vt::MatmulDenseF16(gpu.q, y.tensor, a.tensor, w.tensor, affine ? &b.tensor : nullptr);
    xpu_test::SameBytes(y.download(), first);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision dense FP16: partial aliases refuse and adjacent views work") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  Buffer storage(gpu.q, DType::kF16, {4}), weights(gpu.q, DType::kF16, {5});
  Buffer biases(gpu.q, DType::kF16, {3});
  storage.put({1, 2, 7, 8}); weights.put({1, 0, 0, 1, 9}); biases.put({0, 0, 0});
  const auto view = [&](void* ptr, std::initializer_list<int64_t> shape) {
    return vt::Tensor::Contiguous(ptr, DType::kF16, gpu.q.device, shape);
  };
  auto a = view(storage.tensor.data, {1, 2});
  auto w = view(weights.tensor.data, {2, 2});
  auto b = view(biases.tensor.data, {2});
  auto output = view(static_cast<unsigned char*>(storage.tensor.data) + 2, {1, 2});
  CHECK_THROWS_AS(vt::MatmulDenseF16(gpu.q, output, a, w), std::runtime_error);
  output = view(static_cast<unsigned char*>(weights.tensor.data) + 2, {1, 2});
  CHECK_THROWS_AS(vt::MatmulDenseF16(gpu.q, output, a, w), std::runtime_error);
  output = view(static_cast<unsigned char*>(biases.tensor.data) + 2, {1, 2});
  CHECK_THROWS_AS(vt::MatmulDenseF16(gpu.q, output, a, w, &b), std::runtime_error);
  // Distinct live ranges in one allocation are legal, even though the owning
  // allocation is shared. These four half elements retain both input and output.
  storage.put({1, 2, 7, 8});
  weights.put({1, 0, 0, 1, 9});
  output = view(static_cast<unsigned char*>(storage.tensor.data) + 4, {1, 2});
  CHECK_NOTHROW(vt::MatmulDenseF16(gpu.q, output, a, w));
  const auto after = storage.floats();
  CHECK((after == std::vector<float>{1, 2, 1, 2}));
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision dense FP16: actual patch and merger projection boundaries" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR","VT_B70_VISION_MODEL_DIR"}))) {
  const char* capture = exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");
  const char* checkpoint = exl3_test::CaseExternalEnvironment("VT_B70_VISION_MODEL_DIR");

  const std::string dir = capture, model = checkpoint;
  const auto manifest = Document(dir + "/dense-real-projections.json");
  const auto index = vllm::LoadSafetensorsIndex(model + "/model.safetensors.index.json");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const char* profile = std::getenv("VT_XPU_PROFILE");
  const bool profiled = profile && std::string_view(profile) == "1";
  if (profiled) vt::xpu::DrainProfileEvents(gpu.q.device.index);
  for (const std::string stage : {"patch", "merger-fc1", "merger-fc2"}) {
    std::vector<json> cases;
    for (const auto& c : manifest.at("cases")) if (c.at("stage") == stage) cases.push_back(c);
    REQUIRE(cases.size() == 2);
    const int64_t m = stage == "patch" ? 768 : 192;
    const int64_t k = stage == "patch" ? 1536 : 4608;
    const int64_t n = stage == "patch" ? 1152 : stage == "merger-fc1" ? 4608 : 5120;
    const std::string key = cases[0].at("key");
    auto file = vllm::SafetensorsFile::Open(model + "/" + index.at(key + ".weight"));
    const auto convert = [&](const std::string& suffix, const std::vector<int64_t>& shape) {
      const auto& stored = file.Get(key + suffix);
      REQUIRE(stored.dtype == "BF16"); REQUIRE(stored.shape == shape);
      std::vector<uint16_t> half(stored.nbytes / 2);
      for (size_t i = 0; i < half.size(); ++i)
        half[i] = vt::F32ToF16(vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(stored.data + 2 * i)));
      return half;
    };
    const auto weight = convert(".weight", stage == "patch" ?
        std::vector<int64_t>{1152, 3, 2, 16, 16} : std::vector<int64_t>{n, k});
    const auto bias = convert(".bias", {n});
    Buffer w(gpu.q, DType::kF16, {n, k}), b(gpu.q, DType::kF16, {n});
    REQUIRE(w.bytes == weight.size() * 2); REQUIRE(b.bytes == bias.size() * 2);
    w.upload(weight.data()); b.upload(bias.data());
    for (size_t i = 0; i < cases.size(); ++i) {
      const auto& c = cases[i];
      REQUIRE(c.at("input").at("dtype") == "torch.float16");
      REQUIRE((c.at("input").at("shape").get<std::vector<int64_t>>() == std::vector<int64_t>{m, k}));
      REQUIRE((c.at("expected").at("shape").get<std::vector<int64_t>>() == std::vector<int64_t>{m, n}));
      const auto input = Read(dir + "/" + c.at("input").at("file").get<std::string>());
      const auto expected = Read(dir + "/" + c.at("expected").at("file").get<std::string>());
      Buffer a(gpu.q, DType::kF16, {m, k}), y(gpu.q, DType::kF16, {m, n});
      REQUIRE(a.bytes == input.size()); REQUIRE(y.bytes == expected.size()); a.upload(input.data());
      const std::string label = stage + ":image" + std::to_string(i);
      vt::xpu::ProfileMatrixScope scope(label.c_str());
      vt::MatmulDenseF16(gpu.q, y.tensor, a.tensor, w.tensor, &b.tensor);
      Compare(y, expected, manifest.at("contract").at("rel_l2"),
              manifest.at("contract").at("max_abs"), label, "DENSE");
      const auto first = y.download();
      vt::MatmulDenseF16(gpu.q, y.tensor, a.tensor, w.tensor, &b.tensor);
      xpu_test::SameBytes(y.download(), first);
    }
  }
  if (profiled) {
    size_t dense_spans = 0;
    for (const auto& record : vt::xpu::DrainProfileEvents(gpu.q.device.index)) {
      if (record.stage != "onednn_dense_f16_stream") continue;
      ++dense_spans;
      CHECK(record.stream_span);
      std::cout << "VISION_DENSE_ROUTE " << record.matrix << '\n';
    }
    CHECK(dense_spans == 12);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU vision block attribution: actual block0 attention and MLP inputs" * doctest::test_suite("vision-external") * doctest::skip(exl3_test::OptionalExternalEnvironmentMissing({"VT_B70_VISION_REFERENCE_DIR","VT_B70_VISION_MODEL_DIR"}))) {
  const char* capture=exl3_test::CaseExternalEnvironment("VT_B70_VISION_REFERENCE_DIR");
  const char* checkpoint=exl3_test::CaseExternalEnvironment("VT_B70_VISION_MODEL_DIR");

  const std::string dir=capture,model=checkpoint;
  const auto reference=Document(dir+"/vision-boundaries-capture.json");
  const auto index=vllm::LoadSafetensorsIndex(model+"/model.safetensors.index.json");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto weight=[&](const std::string& key,std::initializer_list<int64_t> shape) {
    auto file=vllm::SafetensorsFile::Open(model+"/"+index.at(key));
    const auto& stored=file.Get(key);
    REQUIRE(stored.dtype=="BF16");REQUIRE(stored.shape==std::vector<int64_t>(shape));
    std::vector<uint16_t> half(stored.nbytes/2);
    for (size_t i=0;i<half.size();++i)
      half[i]=vt::F32ToF16(vt::BF16ToF32(vt::LoadUnaligned<uint16_t>(stored.data+2*i)));
    auto out=std::make_unique<Buffer>(gpu.q,DType::kF16,shape);
    REQUIRE(out->bytes==stored.nbytes);out->upload(half.data());return out;
  };
  const std::string prefix="model.visual.blocks.0.";
  auto qkv_w=weight(prefix+"attn.qkv.weight",{3456,1152});
  auto qkv_b=weight(prefix+"attn.qkv.bias",{3456});
  auto proj_w=weight(prefix+"attn.proj.weight",{1152,1152});
  auto proj_b=weight(prefix+"attn.proj.bias",{1152});
  auto fc1_w=weight(prefix+"mlp.linear_fc1.weight",{4304,1152});
  auto fc1_b=weight(prefix+"mlp.linear_fc1.bias",{4304});
  auto fc2_w=weight(prefix+"mlp.linear_fc2.weight",{1152,4304});
  auto fc2_b=weight(prefix+"mlp.linear_fc2.bias",{1152});
  Buffer input(gpu.q,DType::kF16,{768,1152}),qkv(gpu.q,DType::kF16,{768,3456}),
      queries(gpu.q,DType::kF16,{768,1152}),keys(gpu.q,DType::kF16,{768,1152}),
      values(gpu.q,DType::kF16,{768,1152}),cache(gpu.q,DType::kF16,{768,72}),
      attention(gpu.q,DType::kF16,{768,16,72}),projected(gpu.q,DType::kF16,{768,1152}),
      fc1(gpu.q,DType::kF16,{768,4304}),fc2(gpu.q,DType::kF16,{768,1152});
  for (size_t image=0;image<2;++image) {
    const auto boundary=[&](size_t offset,const char* name) {
      const auto& entry=reference.at("worker").at("captures")[image*35+offset];
      REQUIRE(entry.at("name")==name);return Read(dir+"/"+entry.at("file").get<std::string>());
    };
    const auto input_bytes=boundary(16,"block0-attn-input-0");
    const auto expected_attention=boundary(21,"block0-attn-output");
    const auto cos=boundary(5,"encoder-metadata-rotary_pos_emb_cos");
    const auto sin=boundary(6,"encoder-metadata-rotary_pos_emb_sin");
    REQUIRE(input_bytes.size()==input.bytes);REQUIRE(cos.size()==768*36*2);REQUIRE(sin.size()==cos.size());
    std::vector<unsigned char> coefficients(cache.bytes);
    for (size_t row=0;row<768;++row) {
      std::copy_n(cos.data()+row*72,72,coefficients.data()+row*144);
      std::copy_n(sin.data()+row*72,72,coefficients.data()+row*144+72);
    }
    cache.upload(coefficients.data());input.upload(input_bytes.data());
    for (const bool fused : {true,false}) {
      vt::MatmulDenseF16(gpu.q,qkv.tensor,input.tensor,qkv_w->tensor,&qkv_b->tensor);
      vt::QkvSplit(gpu.q,queries.tensor,keys.tensor,values.tensor,qkv.tensor);
      auto q=queries.tensor.View({768,16,72}),k=keys.tensor.View({768,16,72}),v=values.tensor.View({768,16,72});
      vt::VisionRopeApply(gpu.q,q,k,cache.tensor);
      const auto args=vllm::multimodal::TypedVisionAttentionArgs(72);
      if (fused) vt::AttentionDenseFlash(gpu.q,attention.tensor,q,k,v,args);
      else vt::Attention(gpu.q,attention.tensor,q,k,v,args);
      vt::MatmulDenseF16(gpu.q,projected.tensor,attention.tensor.View({768,1152}),proj_w->tensor,&proj_b->tensor);
      // Fixed same-input submodule bounds use the already qualified dense
      // projection contract. This isolates attention from upstream norm drift;
      // it neither changes nor replaces the complete-tower qualification gates.
      Compare(projected,expected_attention,3e-4,0.02,
          "block0-attn-"+std::string(fused?"fused":"generic")+"-image"+std::to_string(image),"DENSE");
      xpu_test::SameBytes(input.download(),input_bytes);
    }
    const auto mlp_input=boundary(24,"block0-mlp-input-0");
    const auto mlp_output=boundary(25,"block0-mlp-output");
    REQUIRE(mlp_input.size()==input.bytes);input.upload(mlp_input.data());
    vt::MatmulDenseF16(gpu.q,fc1.tensor,input.tensor,fc1_w->tensor,&fc1_b->tensor);
    vt::GeluTanh(gpu.q,fc1.tensor,fc1.tensor);
    vt::MatmulDenseF16(gpu.q,fc2.tensor,fc1.tensor,fc2_w->tensor,&fc2_b->tensor);
    Compare(fc2,mlp_output,3e-4,0.02,"block0-mlp-image"+std::to_string(image),"DENSE");
    xpu_test::SameBytes(input.download(),mlp_input);
  }
  CHECK(vt::GetReferenceTierHits()==0);
}
