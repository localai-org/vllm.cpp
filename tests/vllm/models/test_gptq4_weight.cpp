#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gptq4_weight.h"
#include "vllm/model_executor/models/dense_gptq4_linear.h"
#include "vllm/model_executor/models/dense_weight_loaders.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/fp8_kv.h"
#include "vt/op_provider.h"
#include "vt/unaligned.h"
#include "vt/xpu.h"
#include <nlohmann/json.hpp>


namespace {

struct Fixture {
  std::map<std::string, std::vector<uint8_t>> storage;
  std::map<std::string, vllm::StTensor> tensors;

  template <typename T>
  void Add(const std::string& name, const std::string& dtype,
           std::vector<int64_t> shape, const std::vector<T>& values) {
    auto& bytes = storage[name];
    bytes.resize(values.size() * sizeof(T));
    std::memcpy(bytes.data(), values.data(), bytes.size());
    auto& tensor = tensors[name];
    tensor.dtype = dtype;
    tensor.shape = std::move(shape);
    tensor.data = bytes.data();
    tensor.nbytes = bytes.size();
  }

  void AddProjection(const std::string& name, int n, int seed, int k = 256) {
    std::vector<uint32_t> words(k / 8 * n);
    for (int p = 0; p < k / 8; ++p)
      for (int output = 0; output < n; ++output)
        words[p * n + output] = static_cast<uint32_t>(seed + p * 100 + output);
    std::vector<uint16_t> scales(k / 128 * n);
    for (int group = 0; group < k / 128; ++group)
      for (int output = 0; output < n; ++output)
        scales[group * n + output] =
            vt::F32ToF16(static_cast<float>(seed + group * 10 + output + 1));
    std::vector<uint32_t> qzeros(k / 128 * n / 8, 0x77777777u);
    std::vector<int32_t> g_idx(k);
    for (int i = 0; i < k; ++i) g_idx[i] = i / 128;
    Add(name + ".qweight", "I32", {k / 8, n}, words);
    Add(name + ".scales", "F16", {k / 128, n}, scales);
    Add(name + ".qzeros", "I32", {k / 128, n / 8}, qzeros);
    Add(name + ".g_idx", "I32", {k}, g_idx);
  }

  vllm::TensorResolver Resolver() const {
    return [this](const std::string& name) -> const vllm::StTensor& {
      return tensors.at(name);
    };
  }

  std::string WriteSafetensors() const {
    const auto nonce = std::chrono::steady_clock::now()
                           .time_since_epoch().count();
    const std::string path =
        (std::filesystem::temp_directory_path() /
         ("test_gptq4_weight_" + std::to_string(nonce) + ".safetensors"))
            .string();
    nlohmann::json header = nlohmann::json::object();
    size_t offset = 0;
    for (const auto& [name, tensor] : tensors) {
      header[name] = {{"dtype", tensor.dtype},
                      {"shape", tensor.shape},
                      {"data_offsets", {offset, offset + tensor.nbytes}}};
      offset += tensor.nbytes;
    }
    std::string encoded = header.dump();
    encoded.append((8 - encoded.size() % 8) % 8, ' ');
    std::ofstream file(path, std::ios::binary);
    const uint64_t header_size = encoded.size();
    file.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    file.write(encoded.data(), encoded.size());
    for (const auto& [name, bytes] : storage)
      file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    file.close();
    return path;
  }
};

TEST_CASE("dense BF16 and FP16 draft weights pack as symmetric GPTQ4 G128") {
  Fixture fixture;
  std::vector<uint16_t> bf16(8 * 128);
  std::vector<uint16_t> f16(8 * 128);
  for (int row = 0; row < 8; ++row) {
    for (int col = 0; col < 128; ++col) {
      const float value = col == 0 ? -7.0F : col == 1 ? 7.0F
          : col == 2 ? 0.5F : col == 3 ? -0.5F : 0.0F;
      bf16[row * 128 + col] = vt::F32ToBF16(value);
      f16[row * 128 + col] = vt::F32ToF16(value);
    }
  }
  fixture.Add("b", "BF16", {8, 128}, bf16);
  fixture.Add("f", "F16", {8, 128}, f16);
  const auto packed = vllm::QuantizeMergedGptq4Weight(
      {fixture.tensors.at("b"), fixture.tensors.at("f")}, 128);
  CHECK(packed.n == 16);
  CHECK(packed.k == 128);
  CHECK(packed.zero_point.bytes.data()[0] == 8);
  CHECK(vt::LoadUnaligned<uint32_t>(packed.qweight.bytes.data()) ==
        0x888888F1u);
  CHECK(vt::LoadUnaligned<uint32_t>(packed.qweight.bytes.data() + 8 * 16 * 4) ==
        0x888888F1u);
  CHECK(vt::F16ToF32(vt::LoadUnaligned<uint16_t>(packed.scales.bytes.data())) ==
        1.0F);
}

uint32_t Word(const vllm::OwnedTensor& tensor, size_t index) {
  return vt::LoadUnaligned<uint32_t>(tensor.bytes.data() + index * 4);
}

uint16_t Scale(const vllm::OwnedTensor& tensor, size_t index) {
  return vt::LoadUnaligned<uint16_t>(tensor.bytes.data() + index * 2);
}

#ifdef VLLM_CPP_TEST_GPTQ4_XPU
vllm::OwnedTensor ConstantHalf(std::vector<int64_t> shape, float value,
                               bool nk = false) {
  auto result = vllm::dense_loaders::MakeOwned(vt::DType::kF16, shape);
  result.nk = nk;
  const uint16_t bits = vt::F32ToF16(value);
  for (size_t i = 0; i < result.bytes.size(); i += 2)
    std::memcpy(result.bytes.data() + i, &bits, 2);
  return result;
}

vllm::Gptq4Weight ConstantPacked(int64_t k, int64_t n,
                                  float scale = 0.00390625f) {
  vllm::Gptq4Weight result;
  result.k = k;
  result.n = n;
  result.qweight = vllm::dense_loaders::MakeOwned(vt::DType::kI32,
                                                  {n, k / 8});
  result.scales = vllm::dense_loaders::MakeOwned(vt::DType::kF16,
                                                 {k / 128, n});
  result.zero_point = vllm::dense_loaders::MakeOwned(vt::DType::kI8, {1});
  const uint32_t words = 0x99999999u;
  const uint16_t scale_bits = vt::F32ToF16(scale);
  for (size_t i = 0; i < result.qweight.bytes.size(); i += 4)
    std::memcpy(result.qweight.bytes.data() + i, &words, 4);
  for (size_t i = 0; i < result.scales.bytes.size(); i += 2)
    std::memcpy(result.scales.bytes.data() + i, &scale_bits, 2);
  result.zero_point.bytes.data()[0] = 8;
  return result;
}

#endif

}  // namespace

TEST_CASE("GPTQ4 loader transposes packed words and merges scales by group") {
  Fixture fixture;
  fixture.AddProjection("gate", 8, 1);
  fixture.AddProjection("up", 16, 2);
  const auto weight = vllm::LoadMergedGptq4Weight(
      fixture.Resolver(), {"gate", "up"}, 256, {8, 16});
  CHECK(weight.k == 256);
  CHECK(weight.n == 24);
  CHECK(weight.group_size == 128);
  CHECK(weight.packing_version == vllm::Gptq4Weight::kPackingVersion);
  CHECK(weight.disk_zero_offset == 1);
  CHECK(weight.qweight.dtype == vt::DType::kI32);
  CHECK(weight.qweight.shape[0] == 24);
  CHECK(weight.qweight.shape[1] == 32);
  CHECK(weight.scales.shape[0] == 2);
  CHECK(weight.scales.shape[1] == 24);
  CHECK(weight.zero_point.bytes.size() == 1);
  CHECK(weight.zero_point.bytes.data()[0] == 8);
  for (int p = 0; p < 32; ++p) {
    for (int output = 0; output < 24; ++output) {
      const int source_output = output < 8 ? output : output - 8;
      const int seed = output < 8 ? 1 : 2;
      CHECK(Word(weight.qweight, output * 32 + p) ==
            static_cast<uint32_t>(seed + p * 100 + source_output));
    }
  }
  for (int group = 0; group < 2; ++group) {
    for (int output = 0; output < 24; ++output) {
      const int source_output = output < 8 ? output : output - 8;
      const int seed = output < 8 ? 1 : 2;
      CHECK(Scale(weight.scales, group * 24 + output) ==
            vt::F32ToF16(
                static_cast<float>(seed + group * 10 + source_output + 1)));
    }
  }
}

TEST_CASE("GPTQ4 loader rejects changed zero, ordering and geometry") {
  Fixture fixture;
  fixture.AddProjection("q", 8, 1);
  CHECK_NOTHROW(vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8));
  CHECK_THROWS_AS(vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 16),
                  std::runtime_error);
  fixture.storage["q.qzeros"][0] = 0;
  CHECK_THROWS_AS(vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8),
                  std::runtime_error);
  fixture.storage["q.qzeros"][0] = 0x77;
  fixture.storage["q.g_idx"][0] = 1;
  CHECK_THROWS_AS(vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8),
                  std::runtime_error);
}

TEST_CASE("GPTQ4 resident uploads once and releases the host mirror") {
  Fixture fixture;
  fixture.AddProjection("q", 8, 1);
  auto weight = vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8);
  REQUIRE(weight.ResidentBytes() == 0);
  const auto before = vllm::load_stats::Snapshot().device_upload_bytes;
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kCPU, 0});
  const auto first = vllm::PrepareGptq4Resident(weight, queue, false);
  CHECK(weight.ResidentBytes() == 1057);
  CHECK(weight.qweight.HasHostBytes());
  CHECK(std::memcmp(first.qweight.data, weight.qweight.bytes.data(),
                    weight.qweight.bytes.size()) == 0);
  CHECK(std::memcmp(first.scales.data, weight.scales.bytes.data(),
                    weight.scales.bytes.size()) == 0);
  CHECK(vllm::load_stats::Snapshot().device_upload_bytes - before == 1057);
  const auto second = vllm::PrepareGptq4Resident(weight, queue, true);
  CHECK(second.qweight.data == first.qweight.data);
  CHECK(second.scales.data == first.scales.data);
  CHECK(second.zero_point.data == first.zero_point.data);
  CHECK(!weight.qweight.HasHostBytes());
  CHECK(!weight.scales.HasHostBytes());
  CHECK(!weight.zero_point.HasHostBytes());
  CHECK(!weight.qweight.Empty());
  CHECK(vllm::load_stats::Snapshot().device_upload_bytes - before == 1057);
  vt::DestroyQueue(queue);
}

TEST_CASE("GPTQ4 model host-release visitor includes packed residents") {
  Fixture fixture;
  fixture.AddProjection("q", 8, 1);
  vllm::Qwen3_5DenseWeights model;
  model.layers.resize(1);
  auto& weight = model.layers[0].gptq4.mlp_down;
  weight = vllm::LoadGptq4Weight(fixture.Resolver(), "q", 256, 8);
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kCPU, 0});
  (void)vllm::PrepareGptq4Resident(weight, queue, false);
  REQUIRE(weight.qweight.HasHostBytes());
  CHECK(vllm::ReleaseResidentQwen3_5DenseHostWeights(model) == 1057);
  CHECK(!weight.qweight.HasHostBytes());
  CHECK(!weight.scales.HasHostBytes());
  CHECK(!weight.zero_point.HasHostBytes());
  CHECK(weight.ResidentBytes() == 1057);
  CHECK(vllm::ReleaseResidentQwen3_5DenseHostWeights(model) == 0);
  vt::DestroyQueue(queue);
}

TEST_CASE("GPTQ4 text inventory loads only the declared packed projections") {
  Fixture fixture;
  const std::string base = "model.language_model.layers.0.";
  fixture.AddProjection(base + "mlp.gate_proj", 128, 1, 128);
  fixture.AddProjection(base + "mlp.up_proj", 128, 2, 128);
  fixture.AddProjection(base + "mlp.down_proj", 128, 3, 128);
  fixture.AddProjection(base + "linear_attn.in_proj_qkv", 384, 4, 128);
  fixture.AddProjection(base + "linear_attn.in_proj_z", 128, 5, 128);
  fixture.AddProjection(base + "linear_attn.out_proj", 128, 6, 128);
  const std::vector<uint16_t> one(128, vt::F32ToF16(1.0f));
  fixture.Add("model.language_model.embed_tokens.weight", "F16", {16, 128},
              std::vector<uint16_t>(16 * 128, vt::F32ToF16(0.5f)));
  fixture.Add("model.language_model.norm.weight", "F16", {128}, one);
  fixture.Add("lm_head.weight", "F16", {16, 128},
              std::vector<uint16_t>(16 * 128, vt::F32ToF16(0.25f)));
  fixture.Add(base + "input_layernorm.weight", "F16", {128}, one);
  fixture.Add(base + "post_attention_layernorm.weight", "F16", {128}, one);
  fixture.Add(base + "linear_attn.in_proj_b.weight", "F16", {1, 128}, one);
  fixture.Add(base + "linear_attn.in_proj_a.weight", "F16", {1, 128},
              std::vector<uint16_t>(128, vt::F32ToF16(2.0f)));
  fixture.Add(base + "linear_attn.conv1d.weight", "F16", {384, 1, 4},
              std::vector<uint16_t>(384 * 4, vt::F32ToF16(0.5f)));
  fixture.Add(base + "linear_attn.A_log", "F16", {1},
              std::vector<uint16_t>{vt::F32ToF16(2.0f)});
  fixture.Add(base + "linear_attn.dt_bias", "F16", {1},
              std::vector<uint16_t>{vt::F32ToF16(3.0f)});
  fixture.Add(base + "linear_attn.norm.weight", "F16", {128}, one);
  const std::string path = fixture.WriteSafetensors();
  std::vector<vllm::SafetensorsFile> shards;
  shards.push_back(vllm::SafetensorsFile::Open(path));
  vllm::HfConfig config;
  config.hidden_size = 128;
  config.vocab_size = 16;
  config.intermediate_size = 128;
  config.num_hidden_layers = 1;
  config.num_attention_heads = 1;
  config.num_key_value_heads = 1;
  config.head_dim = 128;
  config.linear_num_key_heads = 1;
  config.linear_key_head_dim = 128;
  config.linear_num_value_heads = 1;
  config.linear_value_head_dim = 128;
  config.linear_conv_kernel_dim = 4;
  config.torch_dtype = "float16";
  config.mamba_ssm_dtype = "float32";
  config.layer_types = {"linear_attention"};
  config.raw = {{"quantization_config",
                 {{"quant_method", "gptq"}, {"format", "gptq"},
                  {"bits", 4}, {"group_size", 128}, {"sym", true},
                  {"desc_act", false}, {"pack_dtype", "int32"}}}};
  const auto layers =
      vllm::LoadQwen3_5DenseGptq4TextProjections(shards, config);
  REQUIRE(layers.size() == 1);
  CHECK(layers[0].gdn_qkvz.n == 512);
  CHECK(layers[0].gdn_out.k == 128);
  CHECK(layers[0].mlp_gate_up.n == 256);
  CHECK(layers[0].attn_qkv.k == 0);
  CHECK(layers[0].ResidentBytes() == 0);
  const auto full = vllm::LoadQwen3_5Dense(shards, config);
  REQUIRE(full.gptq4_checkpoint);
  CHECK(full.precision.activation == vt::DType::kF16);
  CHECK(full.precision.dense_weight == vt::DType::kF16);
  CHECK(full.precision.kv_auto == vt::DType::kF16);
  CHECK(full.precision.gdn_conv_state == vt::DType::kF16);
  CHECK(full.precision.gdn_recurrent_state == vt::DType::kF32);
  REQUIRE(full.layers.size() == 1);
  CHECK(full.layers[0].gptq4.gdn_qkvz.n == 512);
  CHECK(full.layers[0].gdn.in_proj_ba.dtype == vt::DType::kF16);
  CHECK(full.layers[0].gdn.in_proj_ba.nk);
  CHECK(full.layers[0].gdn.in_proj_ba.shape[0] == 2);
  CHECK(full.layers[0].gdn.in_proj_ba.shape[1] == 128);
  CHECK(Scale(full.layers[0].gdn.in_proj_ba, 0) == vt::F32ToF16(1.0f));
  CHECK(Scale(full.layers[0].gdn.in_proj_ba, 128) == vt::F32ToF16(2.0f));
  CHECK(full.layers[0].gdn.a_log.dtype == vt::DType::kF32);
  CHECK(full.lm_head.dtype == vt::DType::kF16);
  CHECK(full.lm_head.nk);
  CHECK(!full.embed_tokens.Empty());
  CHECK(full.layers[0].mlp.gate_up_proj.Empty());
  fixture.Add(base + "mlp.gate_proj.weight", "F16", {128, 128},
              std::vector<uint16_t>(128 * 128, vt::F32ToF16(1.0f)));
  const std::string conflict_path = fixture.WriteSafetensors();
  std::vector<vllm::SafetensorsFile> conflict_shards;
  conflict_shards.push_back(vllm::SafetensorsFile::Open(conflict_path));
  CHECK_THROWS_AS(vllm::LoadQwen3_5Dense(conflict_shards, config),
                  std::runtime_error);
  std::remove(conflict_path.c_str());
  config.raw["quantization_config"]["desc_act"] = true;
  CHECK_THROWS_AS(vllm::LoadQwen3_5DenseGptq4TextProjections(shards, config),
                  std::runtime_error);
  std::remove(path.c_str());
}

TEST_CASE("GPTQ4 precision policy preserves ordinary dense defaults") {
  vllm::HfConfig config;
  config.torch_dtype = "bfloat16";
  const auto ordinary = vllm::ResolveQwen3_5DensePrecision(config, false);
  CHECK(ordinary.activation == vt::DType::kBF16);
  CHECK(ordinary.dense_weight == vt::DType::kBF16);
  CHECK(ordinary.kv_auto == vt::DType::kBF16);
  CHECK(ordinary.sampler == vt::DType::kF32);
  CHECK_THROWS_AS(vllm::ResolveQwen3_5DensePrecision(config, true),
                  std::runtime_error);
  config.torch_dtype = "float16";
  config.mamba_ssm_dtype = "float32";
  const auto gptq = vllm::ResolveQwen3_5DensePrecision(config, true);
  CHECK(gptq.activation == vt::DType::kF16);
  CHECK(gptq.gdn_recurrent_state == vt::DType::kF32);
}

TEST_CASE("GPTQ4 cache spec stores FP16 KV and convolution with FP32 recurrence") {
  vllm::HfConfig config;
  config.torch_dtype = "float16";
  config.mamba_ssm_dtype = "float32";
  config.raw = {{"quantization_config", {{"quant_method", "gptq"}}}};
  config.num_key_value_heads = 4;
  config.head_dim = 256;
  config.linear_num_key_heads = 16;
  config.linear_num_value_heads = 48;
  config.linear_key_head_dim = 128;
  config.linear_value_head_dim = 128;
  config.linear_conv_kernel_dim = 4;
  auto model = vllm::MakeQwen3_5DenseLoadedModel(vllm::Qwen3_5DenseWeights{});
  const auto kv = vllm::ModelRegistry::MakeKVCache(*model, config, 16, 8);
  REQUIRE(kv.kv_cache_groups.size() == 2);
  const auto* attn = dynamic_cast<const vllm::v1::FullAttentionSpec*>(
      kv.kv_cache_groups[0].kv_cache_spec.get());
  const auto* mamba = dynamic_cast<const vllm::v1::MambaSpec*>(
      kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(attn != nullptr);
  REQUIRE(mamba != nullptr);
  CHECK(attn->dtype == vt::DType::kF16);
  REQUIRE(mamba->dtypes.size() == 2);
  CHECK(mamba->dtypes[0] == vt::DType::kF16);
  CHECK(mamba->dtypes[1] == vt::DType::kF32);
  config.raw = nlohmann::json::object();
  const auto ordinary = vllm::ModelRegistry::MakeKVCache(*model, config, 16, 8);
  const auto* ordinary_mamba = dynamic_cast<const vllm::v1::MambaSpec*>(
      ordinary.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(ordinary_mamba != nullptr);
  CHECK(ordinary_mamba->dtypes[0] == vt::DType::kBF16);
}

#ifdef VLLM_CPP_TEST_GPTQ4_XPU
TEST_CASE("GPTQ4 model boundaries keep F16 hidden tap and F32 gathered logits") {
  const auto owned_half = [](std::vector<int64_t> shape,
                             const std::vector<float>& values, bool nk = false) {
    vllm::OwnedTensor weight;
    weight.dtype = vt::DType::kF16;
    weight.rank = static_cast<int>(shape.size());
    weight.nk = nk;
    for (int i = 0; i < weight.rank; ++i) weight.shape[i] = shape[i];
    weight.bytes.resize(values.size() * sizeof(uint16_t));
    for (size_t i = 0; i < values.size(); ++i) {
      const uint16_t bits = vt::F32ToF16(values[i]);
      std::memcpy(weight.bytes.data() + i * sizeof(bits), &bits, sizeof(bits));
    }
    return weight;
  };
  vllm::HfConfig config;
  config.hidden_size = 4;
  config.vocab_size = 6;
  config.num_hidden_layers = 0;
  config.rms_norm_eps = 1e-6;
  vllm::Qwen3_5DenseWeights weights;
  weights.gptq4_checkpoint = true;
  weights.precision.activation = vt::DType::kF16;
  weights.precision.gdn_conv_state = vt::DType::kF16;
  const std::vector<float> embedding{
      0.25f, 0.5f, -0.25f, 0.125f,
      0.5f, -0.25f, 0.125f, 0.375f,
      -0.375f, 0.25f, 0.5f, -0.125f,
      0.125f, 0.375f, -0.5f, 0.25f,
      0.375f, 0.125f, 0.25f, -0.5f,
      -0.25f, 0.5f, 0.375f, 0.125f};
  const std::vector<float> gamma{0.0f, 0.125f, -0.25f, 0.5f};
  const std::vector<float> head{
      0.5f, 0.25f, -0.125f, 0.375f,
      -0.25f, 0.5f, 0.375f, -0.125f,
      0.125f, -0.5f, 0.25f, 0.375f,
      0.375f, 0.125f, -0.25f, 0.5f,
      -0.125f, 0.375f, 0.5f, 0.25f,
      0.25f, -0.125f, 0.375f, -0.5f};
  weights.embed_tokens = owned_half({6, 4}, embedding);
  weights.final_norm = owned_half({4}, gamma);
  weights.lm_head = owned_half({6, 4}, head, true);
  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  const std::vector<int32_t> ids{1, 2, 3}, positions{0, 1, 2};
  vllm::v1::CommonAttentionMetadata attn;
  attn.num_reqs = 1;
  attn.num_actual_tokens = 3;
  attn.query_start_loc = {0, 3};
  attn.seq_lens = {3};
  attn.block_table_num_cols = 1;
  attn.block_table_tensor = {0};
  attn.slot_mapping = {0, 1, 2};
  vllm::Qwen3_5MTPHiddenStates tap;
  const auto head_before = vllm::dense_gptq4::GetDispatchCounts();
  const auto logits = vllm::Qwen3_5DenseModel::ForwardDeviceTap(
      ids, positions, attn, {}, {}, {}, weights, config, queue,
      &tap, {0, 2});
  const auto head_after = vllm::dense_gptq4::GetDispatchCounts();
  const auto head_index = static_cast<size_t>(vllm::dense_gptq4::Projection::kLmHead);
  const char* head_route = std::getenv("VT_GPTQ4_LM_HEAD_ROUTE");
  const bool reference_head = head_route != nullptr &&
                              std::strcmp(head_route, "reference_matmul_bt") == 0;
  CHECK(head_after.calls[head_index] - head_before.calls[head_index] ==
        (reference_head ? 0u : 1u));
  REQUIRE(tap.tensor.dtype == vt::DType::kF16);
  REQUIRE(tap.tensor.shape[0] == 3);
  REQUIRE(tap.tensor.shape[1] == 4);
  REQUIRE(logits.on_device());
  REQUIRE(logits.device_tensor.dtype == vt::DType::kF32);
  REQUIRE(logits.rows == 2);
  std::vector<uint16_t> tap_bits(12);
  std::vector<float> actual_logits(12);
  auto& backend = vt::GetBackend(queue.device);
  backend.Copy(queue, tap_bits.data(), tap.tensor.data, tap_bits.size() * 2);
  backend.Copy(queue, actual_logits.data(), logits.device_tensor.data,
               actual_logits.size() * sizeof(float));
  backend.Synchronize(queue);
  for (int t = 0; t < 3; ++t) {
    const int token = ids[t];
    float square = 0;
    for (int j = 0; j < 4; ++j)
      square += embedding[token * 4 + j] * embedding[token * 4 + j];
    const float inv = 1.0f / std::sqrt(square / 4.0f + 1e-6f);
    for (int j = 0; j < 4; ++j) {
      const float expected = vt::F16ToF32(vt::F32ToF16(
          embedding[token * 4 + j] * inv * (1.0f + gamma[j])));
      const float actual = vt::F16ToF32(tap_bits[t * 4 + j]);
      CHECK(std::abs(actual - expected) < 0.001f);
    }
  }
  for (int row = 0; row < 2; ++row) for (int vocab = 0; vocab < 6; ++vocab) {
    float expected = 0;
    const int selected = row == 0 ? 0 : 2;
    for (int j = 0; j < 4; ++j)
      expected += vt::F16ToF32(tap_bits[selected * 4 + j]) * head[vocab * 4 + j];
    CHECK(std::abs(actual_logits[row * 6 + vocab] - expected) < 0.001f);
  }
  vt::DestroyQueue(queue);
}

TEST_CASE("GPTQ4 two-layer XPU prefill dispatches every packed text projection") {
  vllm::HfConfig config;
  config.hidden_size = 128;
  config.intermediate_size = 128;
  config.vocab_size = 6;
  config.num_hidden_layers = 2;
  config.layer_types = {"linear_attention", "full_attention"};
  config.num_attention_heads = 2;
  config.num_key_value_heads = 1;
  config.head_dim = 64;
  config.rotary_dim = 32;
  config.rope_theta = 1000000.0;
  config.rms_norm_eps = 1e-6;
  config.linear_num_key_heads = 1;
  config.linear_num_value_heads = 2;
  config.linear_key_head_dim = 64;
  config.linear_value_head_dim = 64;
  config.linear_conv_kernel_dim = 4;

  vllm::Qwen3_5DenseWeights weights;
  weights.gptq4_checkpoint = true;
  weights.precision.activation = vt::DType::kF16;
  weights.precision.gdn_conv_state = vt::DType::kF16;
  weights.embed_tokens = ConstantHalf({6, 128}, 0.03125f);
  weights.final_norm = ConstantHalf({128}, 0.0f);
  weights.lm_head = ConstantHalf({6, 128}, 0.015625f, true);
  weights.layers.resize(2);
  for (auto& layer : weights.layers) {
    layer.input_layernorm = ConstantHalf({128}, 0.0f);
    layer.post_attention_layernorm = ConstantHalf({128}, 0.0f);
    layer.gptq4.mlp_gate_up = ConstantPacked(128, 256);
    layer.gptq4.mlp_down = ConstantPacked(128, 128);
  }
  auto& gdn = weights.layers[0];
  gdn.is_linear_attention = true;
  gdn.gptq4.gdn_qkvz = ConstantPacked(128, 384);
  gdn.gptq4.gdn_out = ConstantPacked(128, 128);
  gdn.gdn.in_proj_ba = ConstantHalf({4, 128}, 0.00390625f, true);
  gdn.gdn.conv1d_weight = ConstantHalf({256, 1, 4}, 0.125f);
  gdn.gdn.a_log = vllm::dense_loaders::MakeOwned(vt::DType::kF32, {2});
  gdn.gdn.dt_bias = vllm::dense_loaders::MakeOwned(vt::DType::kF32, {2});
  const float a_log[] = {0.0f, 0.0f}, dt_bias[] = {-2.0f, -2.0f};
  std::memcpy(gdn.gdn.a_log.bytes.data(), a_log, sizeof(a_log));
  std::memcpy(gdn.gdn.dt_bias.bytes.data(), dt_bias, sizeof(dt_bias));
  gdn.gdn.norm_weight = ConstantHalf({64}, 0.0f);
  auto& attn = weights.layers[1];
  attn.gptq4.attn_qkv = ConstantPacked(128, 384);
  attn.gptq4.attn_out = ConstantPacked(128, 128);
  attn.attn.q_norm = ConstantHalf({64}, 0.0f);
  attn.attn.k_norm = ConstantHalf({64}, 0.0f);

  vt::Queue queue = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  auto& backend = vt::GetBackend(queue.device);
  constexpr int T = 64, block_size = 128;
  constexpr size_t kv_bytes = 2 * 2 * block_size * 64 * 2;
  constexpr size_t ssm_bytes = 2 * 2 * 64 * 64 * 4;
  constexpr size_t conv_bytes = 2 * 256 * 3 * 2;
  void* kv_data = vt::Alloc(queue.device, kv_bytes);
  void* ssm_data = vt::Alloc(queue.device, ssm_bytes);
  void* conv_data = vt::Alloc(queue.device, conv_bytes);
  backend.Memset(queue, kv_data, 0, kv_bytes);
  backend.Memset(queue, ssm_data, 0, ssm_bytes);
  backend.Memset(queue, conv_data, 0, conv_bytes);
  vllm::PagedKvCache kv;
  kv.data = kv_data;
  kv.dtype = vt::DType::kF16;
  kv.num_blocks = 2;
  kv.block_size = block_size;
  kv.num_kv_heads = 1;
  kv.head_size = 64;
  vllm::GdnStateCache state;
  state.ssm_state = vt::Tensor::Contiguous(
      ssm_data, vt::DType::kF32, queue.device, {2, 2, 64, 64});
  state.conv_state = vt::Tensor::Contiguous(
      conv_data, vt::DType::kF16, queue.device, {2, 256, 3});

  std::vector<int32_t> ids(T), positions(T);
  for (int i = 0; i < T; ++i) {
    ids[i] = 1 + i % 5;
    positions[i] = i;
  }
  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = T;
  am.query_start_loc = {0, T};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {T};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = T;
  am.max_seq_len = T;
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0};
  am.slot_mapping.assign(positions.begin(), positions.end());
  am.causal = true;
  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = T;
  gm.num_actual_tokens = T;
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, T};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, T};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  const auto conv = vllm::v1::ComputeCausalConv1dMetadata(
      *gm.non_spec_query_start_loc);
  gm.batch_ptr = conv.batch_ptr;
  gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;

  const auto before = vllm::dense_gptq4::GetDispatchCounts();
  vllm::Qwen3_5MTPHiddenStates tap;
  std::vector<vllm::PagedKvCache> kv_caches{kv};
  std::vector<vllm::GdnStateCache> gdn_caches{state};
  const std::vector<int32_t> logits_indices{T - 1};
  auto model = vllm::BorrowQwen3_5DenseLoadedModel(weights);
  vllm::ModelRegistry::Prepare(*model, config, queue);
  vllm::ModelForwardInput input{ids, positions, am, gm, kv_caches, gdn_caches,
                                config, queue, logits_indices};
  input.num_reqs = 1;
  input.hidden_tap = &tap;
  const auto logits = vllm::ModelRegistry::Forward(*model, input);
  REQUIRE(logits.on_device());
  REQUIRE(logits.rows == 1);
  REQUIRE(logits.vocab == 6);
  CHECK(tap.tensor.dtype == vt::DType::kF16);
  std::vector<float> values(6);
  backend.Copy(queue, values.data(), logits.device_tensor.data, values.size() * 4);
  backend.Synchronize(queue);
  for (float value : values) CHECK(std::isfinite(value));
  const auto after = vllm::dense_gptq4::GetDispatchCounts();
  using vllm::dense_gptq4::Projection;
  const auto calls = [&](Projection projection) {
    const size_t index = static_cast<size_t>(projection);
    return after.calls[index] - before.calls[index];
  };
  CHECK(calls(Projection::kGdnQkvz) == 1);
  CHECK(calls(Projection::kGdnBa) == 1);
  CHECK(calls(Projection::kGdnOut) == 1);
  CHECK(calls(Projection::kAttnQkv) == 1);
  CHECK(calls(Projection::kAttnOut) == 1);
  CHECK(calls(Projection::kMlpGateUp) == 2);
  CHECK(calls(Projection::kMlpDown) == 2);
  if (std::getenv("VLLM_CPP_GPTQ4_FP8_TEST") != nullptr) {
    void* fp8_kv_data = vt::Alloc(queue.device, kv_bytes / 2);
    void* fp8_ssm_data = vt::Alloc(queue.device, ssm_bytes);
    void* fp8_conv_data = vt::Alloc(queue.device, conv_bytes);
    backend.Memset(queue, fp8_kv_data, 0, kv_bytes / 2);
    backend.Memset(queue, fp8_ssm_data, 0, ssm_bytes);
    backend.Memset(queue, fp8_conv_data, 0, conv_bytes);
    auto fp8_kv = kv;
    fp8_kv.data = fp8_kv_data;
    fp8_kv.dtype = vt::DType::kI8;
    fp8_kv.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
    fp8_kv.k_scale = 1.0f;
    fp8_kv.v_scale = 1.0f;
    auto fp8_state = state;
    fp8_state.ssm_state.data = fp8_ssm_data;
    fp8_state.conv_state.data = fp8_conv_data;
    std::vector<vllm::PagedKvCache> fp8_caches{fp8_kv};
    std::vector<vllm::GdnStateCache> fp8_states{fp8_state};
    vt::EnableOpProviderCallStats(true);
    const auto before_store = vt::GetOpProviderStats(
        vt::OpId::kReshapeAndCacheFp8, vt::DeviceType::kXPU).selections;
    vllm::ModelForwardInput fp8_input{
        ids, positions, am, gm, fp8_caches, fp8_states,
        config, queue, logits_indices};
    fp8_input.num_reqs = 1;
    const auto fp8_logits = vllm::ModelRegistry::Forward(*model, fp8_input);
    REQUIRE(fp8_logits.on_device());
    std::vector<float> fp8_values(6);
    backend.Copy(queue, fp8_values.data(), fp8_logits.device_tensor.data,
                 fp8_values.size() * sizeof(float));
    backend.Synchronize(queue);
    const auto after_store = vt::GetOpProviderStats(
        vt::OpId::kReshapeAndCacheFp8, vt::DeviceType::kXPU).selections;
    vt::EnableOpProviderCallStats(false);
    CHECK(after_store > before_store);
    for (int token = 0; token < 6; ++token) {
      CHECK(std::isfinite(fp8_values[token]));
      CHECK(std::abs(fp8_values[token] - values[token]) < 0.05f);
    }
    vt::Free(queue.device, fp8_conv_data);
    vt::Free(queue.device, fp8_ssm_data);
    vt::Free(queue.device, fp8_kv_data);
  }
  if (std::getenv("VLLM_CPP_GPTQ4_GRAPH_TEST") != nullptr) {
    std::vector<int32_t> second_ids(8), second_positions(8);
    for (int token = 0; token < 8; ++token) {
      second_ids[token] = 2 + token % 4;
      second_positions[token] = token;
    }
    auto second_am = am;
    second_am.num_actual_tokens = 8;
    second_am.query_start_loc = {0, 8};
    second_am.query_start_loc_cpu = second_am.query_start_loc;
    second_am.seq_lens = {8};
    second_am.seq_lens_cpu = second_am.seq_lens;
    second_am.max_query_len = 8;
    second_am.max_seq_len = 8;
    second_am.block_table_tensor = {1};
    second_am.slot_mapping.clear();
    for (int token = 0; token < 8; ++token)
      second_am.slot_mapping.push_back(block_size + token);
    auto second_gm = gm;
    second_gm.num_prefill_tokens = 8;
    second_gm.num_actual_tokens = 8;
    second_gm.non_spec_state_indices_tensor = std::vector<int32_t>{1};
    second_gm.non_spec_query_start_loc = std::vector<int32_t>{0, 8};
    second_gm.prefill_query_start_loc = std::vector<int32_t>{0, 8};
    second_gm.prefill_state_indices = std::vector<int32_t>{1};
    const auto second_conv = vllm::v1::ComputeCausalConv1dMetadata(
        *second_gm.non_spec_query_start_loc);
    second_gm.batch_ptr = second_conv.batch_ptr;
    second_gm.token_chunk_offset_ptr = second_conv.token_chunk_offset_ptr;
    const std::vector<int32_t> second_indices{7};
    vllm::ModelForwardInput second_input{
        second_ids, second_positions, second_am, second_gm, kv_caches,
        gdn_caches, config, queue, second_indices};
    second_input.num_reqs = 1;
    REQUIRE(vllm::ModelRegistry::Forward(*model, second_input).on_device());
    backend.Synchronize(queue);
    const auto captures_before = backend.GraphsCaptured();
    const auto replays_before = backend.GraphReplays();
    void* reference_kv_data = vt::Alloc(queue.device, kv_bytes);
    void* reference_ssm_data = vt::Alloc(queue.device, ssm_bytes);
    void* reference_conv_data = vt::Alloc(queue.device, conv_bytes);
    backend.Copy(queue, reference_kv_data, kv_data, kv_bytes);
    backend.Copy(queue, reference_ssm_data, ssm_data, ssm_bytes);
    backend.Copy(queue, reference_conv_data, conv_data, conv_bytes);
    backend.Synchronize(queue);
    auto reference_kv = kv;
    reference_kv.data = reference_kv_data;
    auto reference_state = state;
    reference_state.ssm_state.data = reference_ssm_data;
    reference_state.conv_state.data = reference_conv_data;
    std::vector<vllm::PagedKvCache> reference_kv_caches{reference_kv};
    std::vector<vllm::GdnStateCache> reference_gdn_caches{reference_state};
    const auto decode_step = [&](int step, int state_slot, int position,
                                 bool graph_route) {
      const std::vector<int32_t> token_ids{2 + step % 4};
      const std::vector<int32_t> token_positions{position};
      auto decode_am = am;
      decode_am.num_actual_tokens = 1;
      decode_am.query_start_loc = {0, 1};
      decode_am.query_start_loc_cpu = decode_am.query_start_loc;
      decode_am.seq_lens = {position + 1};
      decode_am.seq_lens_cpu = decode_am.seq_lens;
      decode_am.max_query_len = 1;
      decode_am.max_seq_len = position + 1;
      decode_am.block_table_tensor = {state_slot};
      decode_am.slot_mapping = {state_slot * block_size + position};
      vllm::v1::GDNAttentionMetadata decode_gm;
      decode_gm.num_decodes = 1;
      decode_gm.num_decode_tokens = 1;
      decode_gm.num_actual_tokens = 1;
      decode_gm.non_spec_state_indices_tensor = std::vector<int32_t>{state_slot};
      decode_gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
      const std::vector<int32_t> decode_indices{0};
      vllm::ModelForwardInput step_input{
          token_ids, token_positions, decode_am, decode_gm,
          graph_route ? kv_caches : reference_kv_caches,
          graph_route ? gdn_caches : reference_gdn_caches,
          config, queue, decode_indices};
      step_input.num_reqs = 1;
      step_input.gdn_state_slots = 2;
      step_input.pure_decode = graph_route;
      step_input.uniform_query_len = graph_route ? 1 : 0;
      const auto result = vllm::ModelRegistry::Forward(*model, step_input);
      REQUIRE(result.on_device());
      std::vector<float> values(6);
      backend.Copy(queue, values.data(), result.device_tensor.data, values.size() * 4);
      backend.Synchronize(queue);
      for (float value : values) CHECK(std::isfinite(value));
      return values;
    };
    const std::vector<int> graph_slots{0, 1, 1, 0, 0, 1, 1, 0};
    std::vector<int> graph_positions;
    int next_position[2] = {T, 8};
    for (int slot : graph_slots)
      graph_positions.push_back(next_position[slot]++);
    std::vector<std::vector<float>> graphed;
    for (size_t step = 0; step < graph_slots.size(); ++step)
      graphed.push_back(decode_step(static_cast<int>(step), graph_slots[step],
                                    graph_positions[step], true));
    CHECK(backend.GraphsCaptured() >= captures_before + 2);
    CHECK(backend.GraphReplays() > replays_before);
    for (size_t step = 0; step < graph_slots.size(); ++step) {
      const auto eager = decode_step(static_cast<int>(step), graph_slots[step],
                                     graph_positions[step], false);
      for (int token = 0; token < 6; ++token)
        CHECK(std::abs(graphed[step][token] - eager[token]) < 0.01f);
    }
    model.reset();  // Drop captured pointers before their buffers and queue.
    vt::Free(queue.device, reference_conv_data);
    vt::Free(queue.device, reference_ssm_data);
    vt::Free(queue.device, reference_kv_data);
  }
  vt::Free(queue.device, conv_data);
  vt::Free(queue.device, ssm_data);
  vt::Free(queue.device, kv_data);
  vt::DestroyQueue(queue);
}
#endif
