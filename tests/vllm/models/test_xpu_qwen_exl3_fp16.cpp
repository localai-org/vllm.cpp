// Synthetic native execution check, not pinned-checkpoint/producer parity.
// Exercises the dense model's FP16 policy through GDN, attention, MLP and head.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <chrono>
#include <cstring>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <tuple>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/dense_exl3_linear.h"
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_dense_mm.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/model_executor/models/qwen3_5_gdn_block.h"
#include "vllm/model_executor/models/qwen3_5_attn_block.h"
#include "vllm/model_executor/models/act_dump.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vt/exl3_fixture.h"
#include "vt/unaligned.h"
#include "vt/xpu_test_helpers.h"
#include "vllm/tokenizer/tokenizer.h"
#include "support/exl3_autonomous.h"
#include "vt/xpu.h"

namespace {
using vt::DType;
using vllm::OwnedTensor;

OwnedTensor Own(DType dtype, std::initializer_list<int64_t> shape,
                const void* data, size_t bytes) {
  OwnedTensor w;
  w.dtype = dtype;
  w.rank = static_cast<int>(shape.size());
  std::copy(shape.begin(), shape.end(), w.shape);
  w.bytes.resize(bytes);
  std::memcpy(w.bytes.data(), data, bytes);
  return w;
}

OwnedTensor Values(DType dtype, std::initializer_list<int64_t> shape, uint32_t seed) {
  int64_t n = 1;
  for (int64_t dim : shape) n *= dim;
  exl3_test::Rng rng;
  rng.s = seed;
  std::vector<float> f(static_cast<size_t>(n));
  for (auto& v : f) v = rng.next(0.04f);
  if (dtype == DType::kF32) return Own(dtype, shape, f.data(), f.size() * 4);
  std::vector<uint16_t> h(f.size());
  for (size_t i = 0; i < f.size(); ++i) h[i] = vt::F32ToF16(f[i]);
  return Own(dtype, shape, h.data(), h.size() * 2);
}

vllm::Exl3Weight Packed(int64_t k, int64_t n, int bits, uint32_t seed,
                        const std::string& name) {
  auto f = exl3_test::MakeFixture(k, n, bits, seed);
  for (auto& v : f.svh) v = vt::F32ToF16(0.03125f);
  vllm::Exl3Weight w;
  w.name = name;
  w.codebook = 2;
  w.trellis = Own(DType::kI8, {k / 16, n / 16, 32 * bits},
                   f.trellis.data(), f.trellis.size() * 2);
  w.suh = Own(DType::kF16, {k}, f.suh.data(), f.suh.size() * 2);
  w.svh = Own(DType::kF16, {n}, f.svh.data(), f.svh.size() * 2);
  return w;
}

vllm::HfConfig Config() {
  vllm::HfConfig c;
  c.hidden_size = c.vocab_size = c.intermediate_size = 128;
  c.num_hidden_layers = 2;
  c.num_attention_heads = c.num_key_value_heads = 1;
  c.head_dim = 128;
  c.linear_num_key_heads = c.linear_num_value_heads = 1;
  c.linear_key_head_dim = c.linear_value_head_dim = 128;
  c.linear_conv_kernel_dim = 4;
  c.layer_types = {"linear_attention", "full_attention"};
  c.torch_dtype = "bfloat16";  // exported config; runtime policy is FP16
  c.mamba_ssm_dtype = "float32";
  c.rms_norm_eps = 1e-6;
  c.rope_theta = 10000;
  c.rotary_dim = 64;
  c.max_position_embeddings = 8;
  return c;
}

vllm::Qwen3_5DenseWeights Weights(const vllm::HfConfig& c, bool split_ba) {
  vllm::Qwen3_5DenseWeights w;
  w.exl3_checkpoint = true;
  w.precision = vllm::ResolveQwen3_5DensePrecision(c, false, true);
  w.embed_tokens = Values(DType::kF16, {128, 128}, 1);
  w.final_norm = Values(DType::kF16, {128}, 2);
  w.lm_head_exl3 = Packed(128, 128, 6, 3, "lm_head");
  for (int layer = 0; layer < 2; ++layer) {
    vllm::Qwen3_5DenseLayerWeights l;
    const uint32_t seed = 100 + 100 * layer;
    const std::string prefix = "model.language_model.layers." + std::to_string(layer) + ".";
    l.input_layernorm = Values(DType::kF16, {128}, seed);
    l.post_attention_layernorm = Values(DType::kF16, {128}, seed + 1);
    l.mlp.gate_proj_exl3 = Packed(128, 128, 4, seed + 2, prefix + "mlp.gate_proj");
    l.mlp.up_proj_exl3 = Packed(128, 128, 4, seed + 3, prefix + "mlp.up_proj");
    l.mlp.down_proj_exl3 = Packed(128, 128, 4, seed + 4, prefix + "mlp.down_proj");
    l.is_linear_attention = layer == 0;
    if (l.is_linear_attention) {
      auto& g = l.gdn;
      g.in_proj_qkv_exl3 = Packed(128, 384, 4, seed + 5, prefix + "linear_attn.in_proj_qkv");
      g.in_proj_z_exl3 = Packed(128, 128, 4, seed + 6, prefix + "linear_attn.in_proj_z");
      g.out_proj_exl3 = Packed(128, 128, 4, seed + 7, prefix + "linear_attn.out_proj");
      g.in_proj_ba = Values(DType::kF16, {2, 128}, seed + 8);
      g.in_proj_ba.nk = true;
      if (split_ba) {
        g.in_proj_b = Own(DType::kF16, {1, 128}, g.in_proj_ba.bytes.data(), 256);
        g.in_proj_a = Own(DType::kF16, {1, 128}, g.in_proj_ba.bytes.data() + 256, 256);
        g.in_proj_b.nk = g.in_proj_a.nk = true;
        g.in_proj_ba = OwnedTensor{};
      }
      g.conv1d_weight = Values(DType::kF16, {384, 4}, seed + 9);
      g.norm_weight = Values(DType::kF16, {128}, seed + 10);
      g.a_log = Values(DType::kF32, {1}, seed + 11);
      g.dt_bias = Values(DType::kF32, {1}, seed + 12);
    } else {
      auto& a = l.attn;
      a.q_proj_exl3 = Packed(128, 256, 4, seed + 5, prefix + "self_attn.q_proj");
      a.k_proj_exl3 = Packed(128, 128, 4, seed + 6, prefix + "self_attn.k_proj");
      a.v_proj_exl3 = Packed(128, 128, 4, seed + 7, prefix + "self_attn.v_proj");
      a.o_proj_exl3 = Packed(128, 128, 4, seed + 8, prefix + "self_attn.o_proj");
      a.q_norm = Values(DType::kF16, {128}, seed + 9);
      a.k_norm = Values(DType::kF16, {128}, seed + 10);
    }
    w.layers.push_back(std::move(l));
  }
  return w;
}
}  // namespace

namespace {
std::vector<float> CapturedFloats(const vllm::StTensor& value) {
  REQUIRE((value.dtype == "F16" || value.dtype == "F32"));
  const size_t size = value.dtype == "F16" ? 2 : 4;
  std::vector<float> result(value.nbytes/size);
  for (size_t i = 0; i < result.size(); ++i) {
    if (size == 4) std::memcpy(&result[i], value.data + i*4, 4);
    else { uint16_t bit; std::memcpy(&bit, value.data + i*2, 2); result[i] = vt::F16ToF32(bit); }
  }
  return result;
}

void CapturedClose(const std::string& stage, const std::vector<float>& got,
                   const vllm::StTensor& expected, float rtol, float atol) {
  const auto want = CapturedFloats(expected);
  REQUIRE(got.size() == want.size());
  double error = 0, norm = 0; float max_error = 0; size_t different = 0;
  bool finite = true;
  for (size_t i = 0; i < got.size(); ++i) {
    finite &= std::isfinite(got[i]) && std::isfinite(want[i]);
    const double delta = double(got[i]) - want[i]; error += delta*delta; norm += double(want[i])*want[i];
    max_error = std::max(max_error, std::abs(got[i] - want[i])); different += got[i] != want[i];
  }
  REQUIRE(finite); REQUIRE(norm > 0);
  std::cout << "REAL_BLOCK_STAGE stage=" << stage << " relative=" << std::sqrt(error/norm)
            << " max_error=" << max_error << " different_values=" << different << '\n';
  CHECK(std::sqrt(error/norm) < 2e-3);
  xpu_test::Close(got, want, rtol, atol);
}

struct RealLayer0 {
  vllm::HfConfig config;
  std::optional<vllm::SafetensorsFile> shard, oracle;
  vllm::Qwen3_5DenseLayerWeights layer;
  RealLayer0(int layer_index = 0, const char* oracle_name = "real_block_oracle.safetensors") {
    const char* model = std::getenv("VT_B70_EXL3_MODEL");
    const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
    if (!model || !captures) {
      std::cerr << "SKIP: set VT_B70_EXL3_MODEL and VT_B70_EXL3_S1_FIXTURES.\n";
      std::exit(77);
    }
    config = vllm::LoadHfConfig((std::filesystem::path(model) / "config.json").string());
    shard = vllm::SafetensorsFile::Open((std::filesystem::path(model) / "model-00001-of-00002.safetensors").string());
    const auto get = [&](const std::string& name) -> const vllm::StTensor& { return shard->Get(name); };
    const auto has = [&](const std::string& name) {
      return std::find(shard->Names().begin(), shard->Names().end(), name) != shard->Names().end();
    };
    REQUIRE((layer_index == 0 || layer_index == 1 || layer_index == 20 || layer_index == 21));
    layer = vllm::LoadQwen3_5DenseLayer(get, has, "linear_attention", layer_index, "model.language_model.");
    oracle = vllm::SafetensorsFile::Open((std::filesystem::path(captures) / oracle_name).string());
    REQUIRE(config.hidden_size == 5120);
  }
};

void ExportGdnState(const std::string& phase, const xpu_test::Buffer& conv,
                    const xpu_test::Buffer& ssm) {
  const char* directory = std::getenv("VT_B70_EXL3_STATE_OUTPUT");
  if (!directory) return;
  const std::filesystem::path path(directory);
  std::filesystem::create_directories(path);
  nlohmann::json layout;
  for (const auto& entry : {std::make_pair("conv", &conv), std::make_pair("ssm", &ssm)}) {
    const auto& tensor = entry.second->tensor;
    const auto bytes = entry.second->download();
    const auto file = path / (phase + "-" + entry.first + ".bin");
    REQUIRE(!std::filesystem::exists(file));
    std::ofstream output(file, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    output.close(); REQUIRE(output.good());
    layout[entry.first] = {{"dtype", vt::Name(tensor.dtype)},
                          {"shape", std::vector<int64_t>(tensor.shape, tensor.shape + tensor.rank)},
                          {"stride", std::vector<int64_t>(tensor.stride, tensor.stride + tensor.rank)},
                          {"bytes", bytes.size()}, {"file", file.filename().string()}};
  }
  layout["active_slot"] = 4;
  const auto metadata = path / (phase + "-layout.json");
  REQUIRE(!std::filesystem::exists(metadata));
  std::ofstream report(metadata); report << layout.dump(2) << '\n';
  report.close(); REQUIRE(report.good());
}
}  // namespace

TEST_CASE("XPU EXL3 real attention gate: identical P128 D1 D29 operands") {
  const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!captures) std::exit(77);
  const auto dir = std::filesystem::path(captures);
  const auto early = vllm::SafetensorsFile::Open((dir / "real_attention3_oracle.safetensors").string());
  const auto later = vllm::SafetensorsFile::Open((dir / "repeat-0-attention3.safetensors").string());
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  for (const std::string phase : {"p128", "d1", "d29"}) {
    CAPTURE(phase);
    const auto& oracle = phase == "d29" ? later : early;
    const auto& attention = oracle.Get(phase + "_attention_output");
    const auto& gate = oracle.Get(phase + "_gate");
    const auto& expected = oracle.Get(phase + "_gated_attention");
    REQUIRE(attention.dtype == "F16");
    REQUIRE(gate.dtype == "F16");
    REQUIRE(expected.dtype == "F16");
    const int rows = phase == "p128" ? 128 : 1;
    REQUIRE(attention.shape == std::vector<int64_t>{rows, 6144});
    const auto values = CapturedFloats(attention);
    const std::vector<unsigned char> wanted(expected.data, expected.data + expected.nbytes);
    for (bool alias : {false, true}) {
      CAPTURE(alias);
      xpu_test::Buffer a(gpu.q, DType::kF16, {rows, 6144});
      xpu_test::Buffer z(gpu.q, DType::kF32, {rows, 6144});
      xpu_test::Buffer out(gpu.q, DType::kF16, {rows, 6144});
      a.put(values); z.put(CapturedFloats(gate));
      vt::SigmoidGateBf16(gpu.q, alias ? a.tensor : out.tensor, a.tensor, z.tensor);
      const auto actual = alias ? a.download() : out.download();
      size_t mismatches = 0;
      for (size_t i = 0; i < actual.size(); i += 2)
        mismatches += std::memcmp(actual.data() + i, wanted.data() + i, 2) != 0;
      std::cout << "ATTENTION_GATE_IDENTICAL " << phase << " alias=" << alias
                << " half_differences=" << mismatches << '\n';
      CHECK(mismatches == 0);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real attention core: identical D29 operands and FP8 state") {
  const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!captures) std::exit(77);
  const auto oracle = vllm::SafetensorsFile::Open(
      (std::filesystem::path(captures) / "repeat-0-attention3.safetensors").string());
  const auto& query = oracle.Get("d29_q_rope");
  const auto& expected = oracle.Get("d29_attention_output");
  const auto& key = oracle.Get("d29_key_bytes_after");
  const auto& value = oracle.Get("d29_value_bytes_after");
  REQUIRE(query.dtype == "F16");
  REQUIRE(expected.dtype == "F16");
  REQUIRE(key.dtype == "U8");
  REQUIRE(value.dtype == "U8");
  REQUIRE(query.nbytes == 6144 * 2);
  REQUIRE(expected.nbytes == query.nbytes);
  REQUIRE(key.shape == std::vector<int64_t>{157, 4, 256});
  REQUIRE(value.shape == key.shape);
  const auto& block = oracle.Get("d29_block_table");
  REQUIRE(block.dtype == "I32");
  REQUIRE(block.nbytes == sizeof(int32_t));
  int32_t physical_block;
  std::memcpy(&physical_block, block.data, sizeof(physical_block));
  REQUIRE(physical_block == 1);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  // Same initialized logical state in the original interleaved and native
  // planar layouts. Padding remains poisoned; neither inference nor cache
  // construction is replaced by this isolated consumer regression.
  for (int64_t page : {1600, 1664}) {
  const int64_t block_bytes = page * 4 * 512;
  for (uint8_t padding : {uint8_t{0x7f}, uint8_t{0}})
  for (bool alias : {false, true})
  for (bool interleaved : {false, true}) {
    CAPTURE(page);
    CAPTURE(alias);
    CAPTURE(int(padding));
    CAPTURE(interleaved);
    xpu_test::Buffer storage(gpu.q, DType::kI8, {2, page, 4, 512});
    std::vector<uint8_t> state(2 * block_bytes, padding);
    auto kc = storage.tensor;
    kc.shape[3] = 256;
    if (!interleaved) { kc.stride[1] = 4 * 256; kc.stride[2] = 256; }
    auto vc = kc;
    vc.data = static_cast<uint8_t*>(vc.data) + (interleaved ? 256 : page * 4 * 256);
    for (int64_t p = 0; p < 157; ++p) for (int64_t h = 0; h < 4; ++h) {
      const auto logical = (p * 4 + h) * 256;
      const auto at = physical_block * block_bytes + p * kc.stride[1] + h * kc.stride[2];
      std::memcpy(state.data() + at, key.data + logical, 256);
      std::memcpy(state.data() + at + (interleaved ? 256 : page * 4 * 256),
                  value.data + logical, 256);
    }
    storage.upload(state.data());
    xpu_test::Buffer q(gpu.q, DType::kF16, {1, 24, 256});
    xpu_test::Buffer out(gpu.q, DType::kF16, {1, 24, 256});
    xpu_test::Buffer table(gpu.q, DType::kI32, {1, 1});
    xpu_test::Buffer lengths(gpu.q, DType::kI32, {1});
    xpu_test::Buffer offsets(gpu.q, DType::kI32, {2});
    q.upload(query.data); table.upload(block.data);
    lengths.upload(oracle.Get("d29_seq_lens").data);
    offsets.upload(oracle.Get("d29_query_start_loc").data);
    vt::PagedAttentionArgs args;
    args.scale = 1.0f / 16.0f; args.causal = true; args.max_seq_len = 157;
    args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
    args.k_scale = 1.0f; args.v_scale = 1.0f;
    const int32_t host_offsets[] = {0, 1}; args.query_start_loc_host = host_offsets;
    vt::PagedAttention(gpu.q, alias ? q.tensor : out.tensor, q.tensor, kc, vc, table.tensor,
                        lengths.tensor, offsets.tensor, args);
    const auto actual = alias ? q.download() : out.download();
    size_t mismatches = 0, nonfinite = 0;
    for (size_t i = 0; i < actual.size(); i += 2) {
      mismatches += std::memcmp(actual.data() + i, expected.data + i, 2) != 0;
      uint16_t bits;
      std::memcpy(&bits, actual.data() + i, sizeof(bits));
      nonfinite += !std::isfinite(vt::F16ToF32(bits));
    }
    std::cout << "ATTENTION_CORE_D29 page=" << page << " alias=" << alias
              << " interleaved=" << interleaved
              << " padding=" << int(padding) << " half_differences=" << mismatches
              << " nonfinite=" << nonfinite << '\n';
    CHECK(storage.download() == state);
    CHECK(nonfinite == 0);
    CHECK(mismatches == 0);
  }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

static void RunRealAttentionCore(bool strict_halves, int64_t page = 1600) {
  const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!captures) { std::cerr << "Set VT_B70_EXL3_S1_FIXTURES.\n"; std::exit(77); }
  const auto dir = std::filesystem::path(captures);
  const auto oracle = vllm::SafetensorsFile::Open((dir / "real_attention3_oracle.safetensors").string());
  std::ifstream record_stream(dir / "real_attention3_oracle.json");
  REQUIRE(record_stream.good());
  const auto record = nlohmann::json::parse(record_stream);
  REQUIRE(record.at("active_cache_continuity_exact").get<bool>());
  const int32_t pblock = *reinterpret_cast<const int32_t*>(oracle.Get("p128_block_table").data);
  const int32_t dblock = *reinterpret_cast<const int32_t*>(oracle.Get("d1_block_table").data);
  REQUIRE(pblock >= 0); REQUIRE(dblock == pblock);
  const int64_t blocks = int64_t(pblock) + 1;
  constexpr int64_t heads = 24, kvheads = 4, dim = 256;
  const int64_t block_bytes = page * kvheads * 2 * dim;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  // Match the observed interleaved physical layout, including head/page
  // strides. Every inactive byte starts poisoned and must remain untouched.
  xpu_test::Buffer storage(gpu.q, DType::kI8, {blocks, page, kvheads, 2 * dim});
  std::vector<uint8_t> poison(static_cast<size_t>(blocks * block_bytes), 0x7f);
  storage.upload(poison.data());
  auto kc = storage.tensor;
  kc.shape[3] = dim;
  auto vc = kc;
  vc.data = static_cast<uint8_t*>(vc.data) + dim;
  for (const std::string phase : {"p128", "d1"}) {
    const int64_t rows = phase == "p128" ? 128 : 1;
    const int64_t length = phase == "p128" ? 128 : 129;
    const auto& step = record.at("steps").at(phase == "p128" ? 0 : 1);
    const float kscale = step.at("scales").at("_k_scale").at("values").get<float>();
    const float vscale = step.at("scales").at("_v_scale").at("values").get<float>();
    REQUIRE(kscale > 0); REQUIRE(vscale > 0);
    const auto get = [&](const std::string& label) -> const vllm::StTensor& {
      return oracle.Get(phase + "_" + label);
    };
    REQUIRE(get("q_rope").dtype == "F16");
    REQUIRE(get("k_rope").dtype == "F16");
    REQUIRE(get("value").dtype == "F16");
    xpu_test::Buffer query(gpu.q, DType::kF16, {rows, heads, dim});
    xpu_test::Buffer key(gpu.q, DType::kF16, {rows, kvheads, dim});
    xpu_test::Buffer value(gpu.q, DType::kF16, {rows, kvheads, dim});
    xpu_test::Buffer output(gpu.q, DType::kF16, {rows, heads, dim});
    xpu_test::Buffer slots(gpu.q, DType::kI64, {rows});
    xpu_test::Buffer table(gpu.q, DType::kI32, {1, 1});
    xpu_test::Buffer lengths(gpu.q, DType::kI32, {1});
    xpu_test::Buffer offsets(gpu.q, DType::kI32, {2});
    query.upload(get("q_rope").data); key.upload(get("k_rope").data);
    value.upload(get("value").data);
    std::vector<int64_t> mapped_slots(static_cast<size_t>(rows));
    for (int64_t i = 0; i < rows; ++i) {
      int64_t slot;
      std::memcpy(&slot, get("slot_mapping").data + i * sizeof(slot), sizeof(slot));
      // Preserve the captured logical positions when testing another page size.
      mapped_slots[i] = slot < 0 ? slot : (slot / 1600) * page + slot % 1600;
    }
    slots.upload(mapped_slots.data());
    table.upload(get("block_table").data); lengths.upload(get("seq_lens").data);
    offsets.upload(get("query_start_loc").data);
    vt::ReshapeAndCacheFp8(gpu.q, key.tensor, value.tensor, kc, vc, slots.tensor,
                           vt::Fp8KVCacheDataType::kFp8E4M3, kscale, vscale);
    const auto raw = storage.download();
    const auto& expected_k = get("key_bytes_after");
    const auto& expected_v = get("value_bytes_after");
    REQUIRE(expected_k.dtype == "U8"); REQUIRE(expected_v.dtype == "U8");
    REQUIRE(expected_k.nbytes == static_cast<size_t>(length * kvheads * dim));
    REQUIRE(expected_v.nbytes == expected_k.nbytes);
    size_t key_diff = 0, value_diff = 0, inactive_diff = 0;
    for (int64_t b = 0; b < blocks; ++b)
      for (int64_t p = 0; p < page; ++p)
        for (int64_t h = 0; h < kvheads; ++h)
          for (int64_t c = 0; c < 2 * dim; ++c) {
            const size_t physical = static_cast<size_t>(b * block_bytes + p * kvheads * 2 * dim + h * 2 * dim + c);
            if (b == pblock && p < length) {
              const size_t logical = static_cast<size_t>((p * kvheads + h) * dim + c % dim);
              if (c < dim) key_diff += raw[physical] != expected_k.data[logical];
              else value_diff += raw[physical] != expected_v.data[logical];
            } else inactive_diff += raw[physical] != 0x7f;
          }
    std::cout << "REAL_ATTN_BYTES phase=" << phase << " key_diff=" << key_diff
              << " value_diff=" << value_diff << " inactive_diff=" << inactive_diff << '\n';
    CHECK(key_diff == 0); CHECK(value_diff == 0); CHECK(inactive_diff == 0);
    vt::PagedAttentionArgs args;
    args.scale = 1.0f / 16.0f;
    args.causal = true;
    args.kv_cache_dtype = vt::Fp8KVCacheDataType::kFp8E4M3;
    args.k_scale = kscale; args.v_scale = vscale;
    args.max_seq_len = length;
    const int32_t host_offsets[] = {0, static_cast<int32_t>(rows)};
    args.query_start_loc_host = host_offsets;
    vt::PagedAttention(gpu.q, output.tensor, query.tensor, kc, vc, table.tensor,
                        lengths.tensor, offsets.tensor, args);
    const auto got = output.floats();
    const auto want = CapturedFloats(get("attention_output"));
    REQUIRE(got.size() == want.size());
    size_t finite_failures = 0, band_failures = 0; float maximum = 0;
    for (size_t i = 0; i < got.size(); ++i) {
      finite_failures += !std::isfinite(got[i]) || !std::isfinite(want[i]);
      const float delta = std::abs(got[i] - want[i]);
      maximum = std::max(maximum, delta);
      band_failures += delta > 0.003f + 0.01f * std::abs(want[i]);
    }
    std::cout << "REAL_ATTN_CORE phase=" << phase << " max_error=" << maximum
              << " band_failures=" << band_failures << '\n';
    CHECK(finite_failures == 0); CHECK(band_failures == 0);
    if (strict_halves) {
      const auto actual = output.download();
      const auto& expected = get("attention_output");
      REQUIRE(expected.dtype == "F16");
      REQUIRE(actual.size() == expected.nbytes);
      size_t mismatches = 0;
      for (size_t i = 0; i < actual.size(); i += 2)
        mismatches += std::memcmp(actual.data() + i, expected.data + i, 2) != 0;
      std::cout << "ATTENTION_CORE_STRICT phase=" << phase
                << " half_differences=" << mismatches << '\n';
      CHECK(mismatches == 0);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real attention: identical original operands FP8 bytes P128 D1") {
  RunRealAttentionCore(false);
}

TEST_CASE("XPU EXL3 real attention core: strict P128 D1 original endpoints") {
  for (const int64_t page : {1600, 1664}) RunRealAttentionCore(true, page);
}

TEST_CASE("XPU EXL3 attention RoPE: actual FP16 operands and coefficients") {
  const char* fixtures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  if (!fixtures || !model) std::exit(77);
  const auto oracle = vllm::SafetensorsFile::Open(
      (std::filesystem::path(fixtures) / "real_attention3_rope_oracle.safetensors").string());
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto shard = vllm::SafetensorsFile::Open(
      (std::filesystem::path(model) / "model-00001-of-00002.safetensors").string());
  const auto get = [&](const std::string& n) -> const vllm::StTensor& { return shard.Get(n); };
  const auto has = [&](const std::string& n) {
    return std::find(shard.Names().begin(), shard.Names().end(), n) != shard.Names().end();
  };
  auto layer = vllm::LoadQwen3_5DenseLayer(get, has, "full_attention", 3, "model.language_model.");
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const auto qw = vllm::dense_attn::ResidentWeightF32(d, layer.attn.q_norm, {256});
  const auto kw = vllm::dense_attn::ResidentWeightF32(d, layer.attn.k_norm, {256});
  std::vector<float> coefficients = CapturedFloats(oracle.Get("p128_rope_cos_sin"));
  const auto last = CapturedFloats(oracle.Get("d1_rope_cos_sin"));
  coefficients.insert(coefficients.end(), last.begin(), last.end());
  REQUIRE(coefficients.size() == 129 * 64);
  xpu_test::Buffer cache(gpu.q, DType::kF32, {129, 64});
  cache.put(coefficients);
  vt::RopeArgs args{10000000.0f, 64};
  args.fp16_intermediates = true;
  for (const std::string phase : {"p128", "d1"}) {
    const int64_t rows = phase == "p128" ? 128 : 1;
    xpu_test::Buffer query(gpu.q, DType::kF16, {rows, 24, 256});
    xpu_test::Buffer key(gpu.q, DType::kF16, {rows, 4, 256});
    xpu_test::Buffer positions(gpu.q, DType::kI32, {rows});
    std::vector<int32_t> ids(rows);
    std::iota(ids.begin(), ids.end(), phase == "p128" ? 0 : 128);
    positions.upload(ids.data());
    query.upload(oracle.Get(phase + "_rope_q_input").data);
    key.upload(oracle.Get(phase + "_rope_k_input").data);
    vt::RopeFromCache(gpu.q, query.tensor, &key.tensor, positions.tensor, cache.tensor, args);
    for (const auto& [label, actual] : {std::pair<std::string, std::vector<uint8_t>>{"q", query.download()},
                                       {"k", key.download()}}) {
      const auto& expected = oracle.Get(phase + "_" + label + "_rope");
      const std::vector<uint8_t> want(expected.data, expected.data + expected.nbytes);
      xpu_test::SameBytes(actual, want);
    }
    xpu_test::Buffer merged(gpu.q, DType::kF16, {rows, 14336});
    xpu_test::Buffer gates(gpu.q, DType::kF32, {rows, 24, 256});
    merged.upload(oracle.Get(phase + "_qkv_output").data);
    auto qgate = merged.tensor.Slice(1, 0, 12288);
    auto raw_key = merged.tensor.Slice(1, 12288, 13312);
    auto active_cache = cache.tensor.Slice(0, phase == "p128" ? 0 : 128,
                                         phase == "p128" ? 128 : 129);
    vt::AttnQkNormRopeGate(gpu.q, query.tensor, key.tensor, gates.tensor,
                          qgate, raw_key, qw, kw, active_cache,
                          vt::RmsNormArgs{1e-6f, true}, args);
    CapturedClose(phase + ".fp16_fused_q", query.floats(), oracle.Get(phase + "_q_rope"), 0.01f, 0.003f);
    CapturedClose(phase + ".fp16_fused_k", key.floats(), oracle.Get(phase + "_k_rope"), 0.01f, 0.003f);
  }
  xpu_test::Buffer positions(gpu.q, DType::kI32, {129});
  std::vector<int32_t> ids(129); std::iota(ids.begin(), ids.end(), 0); positions.upload(ids.data());
  xpu_test::Buffer generated(gpu.q, DType::kF32, {129, 64});
  args.linear_scaling_factor = 1.0f;
  vt::RopeCosSinCache(gpu.q, generated.tensor, positions.tensor, args);
  auto values = generated.floats();
  size_t different = 0;
  for (size_t i = 0; i < values.size(); ++i) {
    const auto half = vt::F32ToF16(values[i]);
    if (vt::F16ToF32(half) == coefficients[i]) continue;
    if (different < 16)
      std::cout << "REAL_ROPE_COEFFICIENT_DIFFERENCE position=" << i / 64
                << " column=" << i % 64 << " generated=" << std::hexfloat << values[i]
                << " captured_half=" << coefficients[i] << std::defaultfloat
                << " generated_half_bits=" << half
                << " captured_half_bits=" << vt::F32ToF16(coefficients[i]) << '\n';
    ++different;
  }
  std::cout << "REAL_ROPE_COEFFICIENTS half_differences=" << different << '\n';
  CHECK(different == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 QK norm: IEEE signed-zero products") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  constexpr int rows = 2, dim = 256;
  std::vector<float> values(rows * dim);
  const float pattern[] = {-0.5f, 0.5f, -0.0f, 0.0f};
  std::vector<uint16_t> expected(values.size());
  for (size_t i = 0; i < values.size(); ++i) {
    values[i] = pattern[(i + i / 32) % 4];
    expected[i] = vt::F32ToF16(values[i]) & 0x8000u;
  }
  xpu_test::Buffer input(gpu.q, DType::kF16, {rows, dim}); input.put(values);
  xpu_test::Buffer weight(gpu.q, DType::kF32, {dim});
  weight.put(std::vector<float>(dim, -1.0f));
  xpu_test::Buffer output(gpu.q, DType::kF16, {rows, dim});
  vt::RmsNorm(gpu.q, output.tensor, input.tensor, weight.tensor,
              vt::RmsNormArgs{1e-6f, true, true});
  std::vector<uint8_t> expected_bytes(expected.size() * 2);
  std::memcpy(expected_bytes.data(), expected.data(), expected_bytes.size());
  xpu_test::SameBytes(output.download(), expected_bytes);
}

static void RunRealKvPreimage(bool benchmark) {
  const char* fixtures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  if (!fixtures || !model) std::exit(77);
  const auto oracle = vllm::SafetensorsFile::Open(
      (std::filesystem::path(fixtures) / "target-repeats/repeat-0.safetensors").string());
  std::vector<vllm::SafetensorsFile> shards;
  for (const char* name : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
    shards.push_back(vllm::SafetensorsFile::Open((std::filesystem::path(model) / name).string()));
  const auto has = [&](const std::string& name) {
    for (const auto& shard : shards)
      if (std::find(shard.Names().begin(), shard.Names().end(), name) != shard.Names().end()) return true;
    return false;
  };
  const auto get = [&](const std::string& name) -> const vllm::StTensor& {
    for (const auto& shard : shards)
      if (std::find(shard.Names().begin(), shard.Names().end(), name) != shard.Names().end()) return shard.Get(name);
    throw std::runtime_error("missing layer43 weight: " + name);
  };
  const auto layer = vllm::LoadQwen3_5DenseLayer(get, has, "full_attention", 43, "model.language_model.");
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const auto qw = vllm::dense_attn::ResidentWeightF32(d, layer.attn.q_norm, {256});
  const auto kw = vllm::dense_attn::ResidentWeightF32(d, layer.attn.k_norm, {256});
  const auto compare = [&](const std::string& stage, const xpu_test::Buffer& buffer,
                           const vllm::StTensor& expected) {
    const auto actual = buffer.download();
    REQUIRE(expected.dtype == "F16");
    REQUIRE(actual.size() == expected.nbytes);
    size_t different = 0;
    for (size_t i = 0; i < actual.size(); i += 2)
      different += std::memcmp(actual.data() + i, expected.data + i, 2) != 0;
    std::cout << "KV_PREIMAGE stage=" << stage << " half_differences=" << different << '\n';
    if (const char* output_dir = std::getenv("VT_B70_EXL3_DIAGNOSTIC_OUTPUT")) {
      std::filesystem::create_directories(output_dir);
      const auto path = std::filesystem::path(output_dir) / (stage + ".f16");
      REQUIRE_FALSE(std::filesystem::exists(path));
      std::ofstream output(path, std::ios::binary);
      output.write(reinterpret_cast<const char*>(actual.data()), actual.size());
      output.close(); REQUIRE(output.good());
    }
    CHECK(different == 0);
  };
  vt::RopeArgs rope{10000000.0f, 64};
  rope.fp16_intermediates = true;
  const bool profiled = std::getenv("VT_XPU_PROFILE") &&
      std::string(std::getenv("VT_XPU_PROFILE")) == "1";
  nlohmann::json timing{{"schema", "b70-exl3-layer43-complete-preamble-cost-v1"},
      {"profiled", profiled}, {"iterations_per_batch", 128}, {"warmup_calls", 16},
      {"cases", nlohmann::json::array()},
      {"scope", "Actual held-out original P128/D1 operands and layer43 weights; "
          "complete eager preamble submissions and batch completion wait. "
          "Not serving, graph, Q4 verifier or Python performance parity."}};
  for (const std::string phase : {"p128", "d1"}) {
    const auto captured = [&](const char* stage) -> const vllm::StTensor& {
      return oracle.Get(phase + "_l43_preimage_" + stage);
    };
    const int64_t rows = phase == "p128" ? 128 : 1;
    xpu_test::Buffer raw_key(gpu.q, DType::kF16, {rows * 4, 256});
    xpu_test::Buffer normalized(gpu.q, DType::kF16, {rows * 4, 256});
    raw_key.upload(captured("k_norm_input").data);
    vt::RmsNorm(gpu.q, normalized.tensor, raw_key.tensor, kw,
                vt::RmsNormArgs{1e-6f, true, true});
    compare(phase + ".standalone_norm", normalized, captured("k_norm_output"));

    xpu_test::Buffer query(gpu.q, DType::kF16, {rows, 24, 256});
    query.put(std::vector<float>(rows * 24 * 256, 0.0f));
    xpu_test::Buffer key(gpu.q, DType::kF16, {rows, 4, 256});
    key.upload(captured("rope_k_input").data);
    xpu_test::Buffer positions(gpu.q, DType::kI32, {rows});
    std::vector<int32_t> ids(rows); std::iota(ids.begin(), ids.end(), 0);
    positions.upload(ids.data());
    xpu_test::Buffer cache(gpu.q, DType::kF32, {rows, 64});
    cache.put(CapturedFloats(captured("rope_cos_sin")));
    vt::RopeFromCache(gpu.q, query.tensor, &key.tensor, positions.tensor, cache.tensor, rope);
    compare(phase + ".standalone_rope", key, captured("k_rope"));

    xpu_test::Buffer merged(gpu.q, DType::kF16, {rows, 14336});
    merged.upload(captured("qkv_output").data);
    const auto qgate = merged.tensor.Slice(1, 0, 12288);
    const auto input_key = merged.tensor.Slice(1, 12288, 13312);
    xpu_test::Buffer gates(gpu.q, DType::kF32, {rows, 24, 256});
    const auto invoke = [&] {
      vt::AttnQkNormRopeGate(gpu.q, query.tensor, key.tensor, gates.tensor, qgate, input_key,
                            qw, kw, cache.tensor, vt::RmsNormArgs{1e-6f, true}, rope);
    };
    invoke();
    if (benchmark) {
      for (int i = 0; i < 16; ++i) invoke();
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      (void)vt::xpu::DrainProfileEvents();
      nlohmann::json wall = nlohmann::json::array();
      nlohmann::json device = nlohmann::json::array();
      for (int batch = 0; batch < 3; ++batch) {
        const auto start = std::chrono::steady_clock::now();
        for (int i = 0; i < 128; ++i) invoke();
        vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
        wall.push_back(std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count() / 128);
        const auto events = vt::xpu::DrainProfileEvents();
        if (profiled) {
          REQUIRE(events.size() == 128);
          double ms = 0;
          for (const auto& event : events) {
            REQUIRE((event.stage == "attn_qk_norm_rope_gate_subgroup" ||
                     event.stage == "attn_qk_norm_rope_gate"));
            ms += (event.end_ns - event.start_ns) / 1e6;
          }
          device.push_back(ms / 128);
        } else CHECK(events.empty());
      }
      timing["cases"].push_back({{"phase", phase}, {"rows", rows},
          {"complete_wall_ms_per_call", wall}, {"device_ms_per_call", device}});
    }
    compare(phase + ".fused_norm_rope", key, captured("k_rope"));
  }
  if (benchmark) {
    const char* directory = std::getenv("VT_B70_EXL3_DIAGNOSTIC_OUTPUT");
    REQUIRE(directory != nullptr);
    const auto path = std::filesystem::path(directory) / "preamble-timing.json";
    REQUIRE_FALSE(std::filesystem::exists(path));
    std::ofstream output(path); REQUIRE(output.good());
    output << timing.dump(2) << '\n';
    output.close(); REQUIRE(output.good());
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 attention layer43: actual KV signed-zero preimage") {
  RunRealKvPreimage(false);
}

TEST_CASE("XPU EXL3 attention layer43: bounded complete preamble cost") {
  RunRealKvPreimage(true);
}

TEST_CASE("XPU EXL3 real attention mixer: own cache P128 D1 from original hidden") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* fixtures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  const char* dumps = vllm::actdump::StreamDir();
  if (!model || !fixtures || !dumps) {
    std::cerr << "Set VT_B70_EXL3_MODEL, VT_B70_EXL3_S1_FIXTURES and VT_DUMP_ACT.\n";
    std::exit(77);
  }
  const auto oracle = vllm::SafetensorsFile::Open(
      (std::filesystem::path(fixtures) / "real_attention3_oracle.safetensors").string());
  const auto shard = vllm::SafetensorsFile::Open(
      (std::filesystem::path(model) / "model-00001-of-00002.safetensors").string());
  const auto config = vllm::LoadHfConfig((std::filesystem::path(model) / "config.json").string());
  const auto get = [&](const std::string& n) -> const vllm::StTensor& { return shard.Get(n); };
  const auto has = [&](const std::string& n) {
    return std::find(shard.Names().begin(), shard.Names().end(), n) != shard.Names().end();
  };
  auto layer = vllm::LoadQwen3_5DenseLayer(get, has, "full_attention", 3, "model.language_model.");
  REQUIRE(layer.attn.IsExl3()); REQUIRE(config.hidden_size == 5120);
  int32_t active;
  std::memcpy(&active, oracle.Get("p128_block_table").data, sizeof(active));
  REQUIRE(active >= 0); REQUIRE(active < 212);
  const int64_t blocks = active + 1;
  constexpr int64_t page = 1600, hkv = 4, dim = 256, columns = hkv * dim;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  xpu_test::Buffer storage(gpu.q, DType::kI8, {blocks, 2, page, columns});
  std::vector<uint8_t> poison(storage.bytes, 0x7f);
  storage.upload(poison.data());
  vllm::PagedKvCache cache;
  cache.data = storage.tensor.data; cache.dtype = DType::kI8;
  cache.num_blocks = blocks; cache.block_size = page;
  cache.num_kv_heads = hkv; cache.head_size = dim;
  cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  cache.k_scale = cache.v_scale = 1.0f;
  for (int step_idx = 0; step_idx < 2; ++step_idx) {
    const std::string phase = step_idx ? "d1" : "p128";
    const int32_t rows = step_idx ? 1 : 128;
    const auto captured = [&](const std::string& label) -> const vllm::StTensor& {
      return oracle.Get(phase + "_" + label);
    };
    REQUIRE(captured("positions").dtype == "I64");
    REQUIRE(captured("positions").shape == std::vector<int64_t>{3, rows});
    std::vector<int32_t> positions(rows);
    for (int axis = 0; axis < 3; ++axis) for (int32_t row = 0; row < rows; ++row) {
      int64_t p;
      std::memcpy(&p, captured("positions").data + (axis * rows + row) * 8, 8);
      CHECK(p == (step_idx ? 128 : row)); positions[row] = static_cast<int32_t>(p);
    }
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1; am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = {0, rows};
    am.seq_lens = am.seq_lens_cpu = {step_idx ? 129 : 128};
    am.num_computed_tokens_cpu = {step_idx ? 128 : 0};
    am.max_query_len = rows; am.max_seq_len = am.seq_lens[0];
    am.block_table_num_cols = 1; am.block_table_tensor = {active};
    am.slot_mapping.resize(rows);
    REQUIRE(captured("slot_mapping").dtype == "I64");
    REQUIRE(captured("slot_mapping").nbytes == static_cast<size_t>(rows) * 8);
    std::memcpy(am.slot_mapping.data(), captured("slot_mapping").data, rows * 8);
    int32_t expected_block;
    std::memcpy(&expected_block, captured("block_table").data, 4);
    CHECK(expected_block == active);
    for (int32_t row = 0; row < rows; ++row)
      CHECK(am.slot_mapping[row] == int64_t(active) * page + positions[row]);
    vllm::dense_attn::DBuf input(d, DType::kF16, {rows, 5120}, captured("input_norm_output").data);
    auto step = vllm::BuildFullAttnStepInputs(gpu.q, positions, am, config, DType::kF16);
    const vllm::actdump::LayerScope scope(step_idx, 3);
    const auto output = vllm::RunFullAttnBlockPaged(gpu.q, layer.attn, config,
        input.t(), step, am, cache, rows, DType::kF16);
    REQUIRE(output.tensor.dtype == DType::kF16);
    const auto read_stage = [&](const std::string& label, const std::string& dtype, int64_t width) {
      const auto path = std::filesystem::path(dumps) /
          ("s" + std::to_string(step_idx) + "_l3_attn_" + label + ".bin");
      const size_t bytes = static_cast<size_t>(rows * width) * (dtype == "F16" ? 2 : 4);
      REQUIRE(std::filesystem::file_size(path) == bytes);
      std::vector<uint8_t> raw(bytes);
      std::ifstream stream(path, std::ios::binary);
      stream.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
      REQUIRE(stream.good());
      vllm::StTensor tensor; tensor.dtype = dtype; tensor.data = raw.data(); tensor.nbytes = bytes;
      return CapturedFloats(tensor);
    };
    const auto qkv = CapturedFloats(captured("qkv_output"));
    REQUIRE(qkv.size() == static_cast<size_t>(rows) * 14336);
    for (const auto& [label, start, width] :
         {std::tuple<std::string, int, int>{"qgate", 0, 12288}, {"key_raw", 12288, 1024},
          {"value", 13312, 1024}}) {
      std::vector<float> want;
      for (int32_t row = 0; row < rows; ++row)
        want.insert(want.end(), qkv.begin() + row * 14336 + start,
                    qkv.begin() + row * 14336 + start + width);
      const auto actual = read_stage(label, "F16", width);
      CHECK(actual == want);
    }
    const auto compare = [&](const std::string& label, const std::vector<float>& actual,
                             const std::vector<float>& expected) {
      REQUIRE(actual.size() == expected.size());
      size_t failures = 0, different = 0, nonfinite = 0; float maximum = 0;
      double error = 0, norm = 0;
      for (size_t i = 0; i < actual.size(); ++i) {
        nonfinite += !std::isfinite(actual[i]) || !std::isfinite(expected[i]);
        const float delta = std::abs(actual[i] - expected[i]);
        failures += delta > 0.003f + 0.01f * std::abs(expected[i]);
        different += actual[i] != expected[i]; maximum = std::max(maximum, delta);
        error += double(delta) * delta; norm += double(expected[i]) * expected[i];
      }
      std::cout << "REAL_ATTN_MIXER phase=" << phase << " stage=" << label
                << " relative=" << std::sqrt(error/norm) << " max_error=" << maximum
                << " different=" << different << " band_failures=" << failures << '\n';
      CHECK(nonfinite == 0); CHECK(failures == 0); CHECK(std::sqrt(error/norm) < 2e-3);
    };
    for (const auto& [label, dtype, width, reference] : {
         std::tuple<std::string, std::string, int64_t, std::string>{"q_rope", "F16", 6144, "q_rope"},
         {"k_rope", "F16", 1024, "k_rope"}, {"gate", "F32", 6144, "gate"},
         {"core", "F16", 6144, "attention_output"}, {"gated", "F16", 6144, "gated_attention"}})
      compare(label, read_stage(label, dtype, width), CapturedFloats(captured(reference)));
    xpu_test::Buffer copy(gpu.q, DType::kF16, {rows, 5120});
    vt::Copy(gpu.q, copy.tensor, output.tensor);
    compare("mixer", copy.floats(), CapturedFloats(captured("mixer_output")));
    const auto raw = storage.download();
    size_t key_diff = 0, value_diff = 0, inactive_diff = 0;
    for (int64_t b = 0; b < blocks; ++b) for (int64_t plane = 0; plane < 2; ++plane)
      for (int64_t p = 0; p < page; ++p) for (int64_t c = 0; c < columns; ++c) {
        const size_t offset = static_cast<size_t>(((b * 2 + plane) * page + p) * columns + c);
        if (b == active && p < am.seq_lens[0]) {
          const auto& expected = captured(plane ? "value_bytes_after" : "key_bytes_after");
          const bool differs = raw[offset] != expected.data[p * columns + c];
          if (plane) value_diff += differs; else key_diff += differs;
        } else inactive_diff += raw[offset] != 0x7f;
      }
    std::cout << "REAL_ATTN_MIXER_BYTES phase=" << phase << " key_diff=" << key_diff
              << " value_diff=" << value_diff << " inactive_diff=" << inactive_diff << '\n';
    CHECK(key_diff == 0); CHECK(value_diff == 0); CHECK(inactive_diff == 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block GDN prep: original FP32 Conv normalization boundary") {
  RealLayer0 real;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const auto producer = vllm::SafetensorsFile::Open(
      (std::filesystem::path(std::getenv("VT_B70_EXL3_S1_FIXTURES")) / "real_gdn_oracle.safetensors").string());
  const auto& raw_qkvz = real.oracle->Get("p128_qkvz_output");
  const auto& raw_ba = real.oracle->Get("p128_ba_output");
  vllm::dense_attn::DBuf qkvz(d, DType::kF16, {128, 16384}, raw_qkvz.data);
  vllm::dense_attn::DBuf ba(d, DType::kF16, {128, 96}, raw_ba.data);
  auto input = qkvz.t().Slice(1, 0, 10240);
  auto b = ba.t().Slice(1, 0, 48), a = ba.t().Slice(1, 48, 96);
  xpu_test::Buffer conv(gpu.q, DType::kF32, {128, 10240});
  xpu_test::Buffer state(gpu.q, DType::kF32, {1, 10240, 3});
  state.put(std::vector<float>(10240*3, 7));
  xpu_test::Buffer qsl(gpu.q, DType::kI32, {2}), initial(gpu.q, DType::kI32, {1});
  const int32_t offsets[] = {0, 128}, cold[] = {0};
  qsl.upload(offsets); initial.upload(cold);
  auto cw = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.conv1d_weight, {10240, 4});
  auto alog = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.a_log, {48});
  auto bias = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.dt_bias, {48});
  vt::CausalConv1dFwd(gpu.q, conv.tensor, input, cw, nullptr, state.tensor,
                     qsl.tensor, initial.tensor, vt::CausalConv1dArgs{true});
  if (const char* dir = vllm::actdump::StreamDir()) {
    const auto path = std::filesystem::path(dir) / "s0_l0_gdn_prep_conv.bin";
    VT_CHECK(!std::filesystem::exists(path), "GDN prep dump must not overwrite an existing receipt");
    const auto bytes = conv.download();
    vllm::actdump::WriteBlob("VT_DUMP_ACT", dir, 0, 0, "gdn_prep_conv",
                            DType::kF32, 128, 10240, bytes.data(), bytes.size());
  }
  xpu_test::Buffer q(gpu.q, DType::kF16, {128, 16, 128});
  xpu_test::Buffer k(gpu.q, DType::kF16, {128, 16, 128});
  xpu_test::Buffer v(gpu.q, DType::kF16, {128, 48, 128});
  xpu_test::Buffer g(gpu.q, DType::kF32, {128, 48});
  xpu_test::Buffer beta(gpu.q, DType::kF32, {128, 48});
  vt::GdnPostConv(gpu.q, q.tensor, k.tensor, v.tensor, g.tensor, beta.tensor,
                  conv.tensor, a, b, alog, bias, vt::GdnPostConvArgs{1e-6f, true});
  for (const auto& [label, buffer] :
       std::vector<std::pair<std::string, xpu_test::Buffer*>>{{"q", &q}, {"k", &k}, {"v", &v}}) {
    CAPTURE(label);
    auto expected = CapturedFloats(producer.Get("p128_conv_" + label));
    const auto actual = buffer->floats();
    expected.resize(actual.size());  // discard original zero-padded capacity
    const auto different = std::inner_product(actual.begin(), actual.end(), expected.begin(),
        size_t{0}, std::plus<size_t>(), [](float x, float y) { return size_t(x != y); });
    std::cout << "REAL_GDN_PREP stage=" << label << " different_values=" << different << '\n';
    CHECK(different == 0);
  }
  const auto producer_beta = CapturedFloats(producer.Get("p128_conv_b"));
  const auto actual_beta = beta.floats();
  const int padded_rows = static_cast<int>(producer.Get("p128_conv_b").shape[1]);
  for (int t = 0; t < 128; ++t) for (int h = 0; h < 48; ++h)
    CHECK(actual_beta[t*48+h] == producer_beta[h*padded_rows+t]);
  CHECK(vt::GetReferenceTierHits() == 0);
}

static void RunRealGdnRawGate(int layer_index) {
  RealLayer0 real(layer_index);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const auto producer = vllm::SafetensorsFile::Open(
      (std::filesystem::path(std::getenv("VT_B70_EXL3_S1_FIXTURES")) / "real_gdn_oracle.safetensors").string());
  const auto upload = [&](const char* name, int heads) {
    return vllm::dense_attn::DBuf(d, DType::kF16, {128, heads, 128},
                                 producer.Get(name).data);
  };
  auto q = upload("p128_conv_q", 16), k = upload("p128_conv_k", 16);
  auto v = upload("p128_conv_v", 48);
  const auto& raw_ba = real.oracle->Get("p128_ba_output");
  vllm::dense_attn::DBuf ba(d, DType::kF16, {128, 96}, raw_ba.data);
  auto raw_a = ba.t().Slice(1, 48, 96);  // Exercise the actual merged BA row stride.
  const auto captured_beta = CapturedFloats(producer.Get("p128_conv_b"));
  std::vector<float> token_beta(128 * 48);
  for (int t = 0; t < 128; ++t) for (int h = 0; h < 48; ++h)
    token_beta[t * 48 + h] = captured_beta[h * 191 + t];
  xpu_test::Buffer beta(gpu.q, DType::kF32, {128, 48});
  beta.put(token_beta);
  auto alog = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.a_log, {48});
  auto bias = vllm::dense_attn::ResidentWeight(d, real.layer.gdn.dt_bias, {48});
  xpu_test::Buffer state(gpu.q, DType::kF32, {1, 48, 128, 128});
  state.put(std::vector<float>(48 * 128 * 128, 0));
  xpu_test::Buffer out(gpu.q, DType::kF16, {128, 48, 128});
  xpu_test::Buffer qsl(gpu.q, DType::kI32, {2});
  const int32_t offsets[] = {0, 128};
  qsl.upload(offsets);
  vt::GdnPrefillRawGate(gpu.q, out.tensor, q.t(), k.t(), v.t(), raw_a,
                        beta.tensor, alog, bias, state.tensor, qsl.tensor,
                        vt::GdnArgs{1.0f});
  for (const auto& [label, buffer, capture] :
       std::vector<std::tuple<const char*, xpu_test::Buffer*, const char*>>{
           {"core", &out, "p128_core"}, {"state", &state, "p128_ssm_state"}}) {
    const auto actual = buffer->floats();
    const auto expected = CapturedFloats(producer.Get(capture));
    REQUIRE(actual.size() == expected.size());
    const auto different = std::inner_product(actual.begin(), actual.end(), expected.begin(),
        size_t{0}, std::plus<size_t>(), [](float x, float y) { return size_t(x != y); });
    std::cout << "REAL_GDN_RAW_GATE stage=" << label << " different_values=" << different << '\n';
    CHECK(different == 0);
  }
  // Continue from the native P128 result, never a captured-state replacement.
  // A padded packed row and strided BA views retain the real input contracts.
  constexpr int slots = 5, active = 4, state_elements = 48 * 128 * 128;
  xpu_test::Buffer cache(gpu.q, DType::kF32, {slots, 48, 128, 128});
  cache.put(std::vector<float>(slots * state_elements, 0.125f));
  xpu_test::Buffer indices(gpu.q, DType::kI32, {1});
  const int32_t index[] = {active};
  indices.upload(index);
  vt::GdnStateScatter(gpu.q, cache.tensor, state.tensor, indices.tensor);
  std::vector<uint16_t> packed(10304, 0x7e00);  // poisoned physical tail
  std::memcpy(packed.data(), producer.Get("d1_conv_q").data, 2048 * 2);
  std::memcpy(packed.data() + 2048, producer.Get("d1_conv_k").data, 2048 * 2);
  std::memcpy(packed.data() + 4096, producer.Get("d1_conv_v").data, 6144 * 2);
  vllm::dense_attn::DBuf mixed_owner(d, DType::kF16, {1, 10304}, packed.data());
  auto mixed = mixed_owner.t().Slice(1, 0, 10240);
  std::vector<uint16_t> decode_ba(96);
  std::memcpy(decode_ba.data(), producer.Get("d1_conv_b").data, 48 * 2);
  std::memcpy(decode_ba.data() + 48, producer.Get("d1_conv_a").data, 48 * 2);
  vllm::dense_attn::DBuf decode_ba_owner(d, DType::kF16, {1, 96}, decode_ba.data());
  auto decode_b = decode_ba_owner.t().Slice(1, 0, 48);
  auto decode_a = decode_ba_owner.t().Slice(1, 48, 96);
  xpu_test::Buffer decode_out(gpu.q, DType::kF16, {1, 48, 128});
  vt::GdnPackedDecode(gpu.q, decode_out.tensor, mixed, decode_a, decode_b,
                      alog, bias, cache.tensor, indices.tensor,
                      vt::GdnArgs{1.0f / std::sqrt(128.0f)});
  const auto decode_actual = decode_out.floats();
  const auto decode_expected = CapturedFloats(producer.Get("d1_core"));
  REQUIRE(decode_actual.size() == decode_expected.size());
  const auto core_different = std::inner_product(decode_actual.begin(), decode_actual.end(),
      decode_expected.begin(), size_t{0}, std::plus<size_t>(),
      [](float x, float y) { return size_t(x != y); });
  const auto cache_actual = cache.floats();
  const auto state_expected = CapturedFloats(producer.Get("d1_ssm_state"));
  REQUIRE(state_expected.size() == state_elements);
  const auto state_different = std::inner_product(cache_actual.begin() + active * state_elements,
      cache_actual.begin() + (active + 1) * state_elements, state_expected.begin(),
      size_t{0}, std::plus<size_t>(), [](float x, float y) { return size_t(x != y); });
  const auto inactive_different = std::count_if(cache_actual.begin(),
      cache_actual.begin() + active * state_elements, [](float x) { return x != 0.125f; });
  std::cout << "REAL_GDN_NATIVE_CHAIN D1_core_diff=" << core_different
            << " D1_state_diff=" << state_different
            << " inactive_diff=" << inactive_different << '\n';
  CHECK(core_different == 0);
  CHECK(state_different == 0);
  CHECK(inactive_different == 0);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real GDN raw gate: native FP16 P128 D1 state chain") {
  RunRealGdnRawGate(0);
}

TEST_CASE("XPU EXL3 real layer1 GDN raw gate: original projections P128 D1") {
  RunRealGdnRawGate(1);
}

static void RunRealGdnStateChain(int layer_index) {
  RealLayer0 real(layer_index);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  constexpr int slots = 5, active = 4, channels = 10240, heads = 48, dim = 128, history = 3;
  REQUIRE(real.config.linear_num_value_heads == heads);
  REQUIRE(real.config.linear_value_head_dim == dim);
  REQUIRE(real.config.linear_key_head_dim == dim);
  REQUIRE(real.config.linear_conv_kernel_dim == history+1);
  // Preserve the captured discrete slot4. Inactive slots are initialized to
  // witnesses; cold active contents are deliberately nonzero and must be ignored.
  xpu_test::Buffer conv(gpu.q, DType::kF16, {slots, channels, history});
  xpu_test::Buffer ssm(gpu.q, DType::kF32, {slots, heads, dim, dim});
  std::vector<float> conv_seed(slots*channels*history, 0.125f);
  std::vector<float> ssm_seed(slots*heads*dim*dim, 0.125f);
  std::fill(conv_seed.begin()+active*channels*history, conv_seed.end(), -2.0f);
  std::fill(ssm_seed.begin()+active*heads*dim*dim, ssm_seed.end(), 7.0f);
  conv.put(conv_seed); ssm.put(ssm_seed);
  ExportGdnState("p128-initial", conv, ssm);
  vllm::GdnStateCache state; state.conv_state = conv.tensor; state.ssm_state = ssm.tensor;
  for (bool prefill : {true, false}) {
    CAPTURE(prefill);
    const vllm::actdump::LayerScope dump_scope(prefill ? 0 : 1, layer_index);
    const int rows = prefill ? 128 : 1;
    const std::string phase = prefill ? "p128" : "d1";
    const auto& x = real.oracle->Get(phase + "_input_norm_output");
    REQUIRE(x.dtype == "F16"); REQUIRE(x.shape == std::vector<int64_t>{rows, 5120});
    vllm::dense_attn::DBuf input(d, DType::kF16, {rows, 5120}, x.data);
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1; am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = {0, rows};
    am.seq_lens = am.seq_lens_cpu = {prefill ? 128 : 129};
    am.max_query_len = rows; am.max_seq_len = am.seq_lens[0];
    am.block_table_num_cols = 1; am.block_table_tensor = {0};
    am.slot_mapping.assign(rows, 0);
    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = rows;
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{active};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, rows};
    if (prefill) {
      gm.num_prefills = 1; gm.num_prefill_tokens = rows;
      gm.has_initial_state = std::vector<uint8_t>{0};
      gm.prefill_query_start_loc = std::vector<int32_t>{0, rows};
      gm.prefill_state_indices = std::vector<int32_t>{active};
      gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto chunks = vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = chunks.batch_ptr; gm.token_chunk_offset_ptr = chunks.token_chunk_offset_ptr;
    } else { gm.num_decodes = gm.num_decode_tokens = 1; }
    const auto& captured_pos = real.oracle->Get(phase + "_positions");
    REQUIRE(captured_pos.dtype == "I64"); REQUIRE(captured_pos.shape == std::vector<int64_t>{3, rows});
    std::vector<int32_t> positions(rows);
    for (int i = 0; i < rows; ++i) for (int axis = 0; axis < 3; ++axis) {
      int64_t p; std::memcpy(&p, captured_pos.data + (axis*rows+i)*8, 8);
      CHECK(p == (prefill ? i : 128)); positions[i] = static_cast<int32_t>(p);
    }
    auto step = vllm::BuildGdnStepInputs(gpu.q, positions, am, gm, slots);
    auto out = vllm::RunGdnBlockPaged(gpu.q, real.layer.gdn, real.config,
                                    input.t(), step, gm, state, rows, nullptr, DType::kF16);
    REQUIRE(out.tensor.dtype == DType::kF16);
    xpu_test::Buffer copy(gpu.q, DType::kF16, {rows, 5120});
    vt::Copy(gpu.q, copy.tensor, out.tensor);
    // Export actual final physical states before a frozen comparison can abort
    // this case. The independent matched-state D1 case is collected separately.
    ExportGdnState(phase + "-final", conv, ssm);
    CapturedClose(phase+".mixer", copy.floats(), real.oracle->Get(phase+"_mixer_output"), 0.01f, 0.003f);
    const auto cv = conv.floats(), ss = ssm.floats();
    CHECK(std::equal(cv.begin(), cv.begin()+active*channels*history, conv_seed.begin()));
    CHECK(std::equal(ss.begin(), ss.begin()+active*heads*dim*dim, ssm_seed.begin()));
    std::vector<float> active_conv(channels*history);
    for (int c = 0; c < channels; ++c) for (int h = 0; h < history; ++h)
      active_conv[h*channels+c] = cv[(active*channels+c)*history+h];
    const auto expected_conv = CapturedFloats(real.oracle->Get(phase+"_conv_state_after"));
    CHECK(active_conv == expected_conv);
    const std::vector<float> active_ssm(ss.begin()+active*heads*dim*dim, ss.end());
    // Existing strict FP32 GDN state comparison band, fixed before first run.
    CapturedClose(phase+".ssm", active_ssm, real.oracle->Get(phase+"_ssm_state_after"), 1e-4f, 1e-5f);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block GDN: captured P128 and D1 mixer/state continuation") {
  RunRealGdnStateChain(0);
}

TEST_CASE("XPU EXL3 real layer1 GDN: identical input P128 and native-state D1") {
  RunRealGdnStateChain(1);
}

static void RunMatchedRealGdnD1(int layer_index) {
  RealLayer0 real(layer_index);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  constexpr int slots = 5, active = 4, channels = 10240, heads = 48, dim = 128, history = 3;
  // This independent D1 gate restores identical original starting states.
  // It does not qualify P128 or replace the native-state continuation test.
  const auto captured_conv = CapturedFloats(real.oracle->Get("d1_conv_state_before"));
  const auto captured_ssm = CapturedFloats(real.oracle->Get("d1_ssm_state_before"));
  REQUIRE(captured_conv.size() == channels * history);
  REQUIRE(captured_ssm.size() == heads * dim * dim);
  std::vector<float> conv_seed(slots * channels * history, 0.125f);
  std::vector<float> ssm_seed(slots * heads * dim * dim, 0.125f);
  for (int c = 0; c < channels; ++c) for (int h = 0; h < history; ++h)
    conv_seed[(active * channels + c) * history + h] = captured_conv[h * channels + c];
  std::copy(captured_ssm.begin(), captured_ssm.end(), ssm_seed.begin() + active * heads * dim * dim);
  xpu_test::Buffer conv(gpu.q, DType::kF16, {slots, channels, history});
  xpu_test::Buffer ssm(gpu.q, DType::kF32, {slots, heads, dim, dim});
  conv.put(conv_seed); ssm.put(ssm_seed);
  ExportGdnState("matched-d1-initial", conv, ssm);
  vllm::GdnStateCache state; state.conv_state = conv.tensor; state.ssm_state = ssm.tensor;
  vllm::dense_attn::DBuf input(d, DType::kF16, {1, 5120},
                              real.oracle->Get("d1_input_norm_output").data);
  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = am.num_actual_tokens = 1;
  am.query_start_loc = am.query_start_loc_cpu = {0, 1};
  am.seq_lens = am.seq_lens_cpu = {129};
  am.max_query_len = 1; am.max_seq_len = 129;
  am.block_table_num_cols = 1; am.block_table_tensor = {0}; am.slot_mapping = {0};
  vllm::v1::GDNAttentionMetadata gm;
  gm.num_actual_tokens = gm.num_decodes = gm.num_decode_tokens = 1;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{active};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
  auto step = vllm::BuildGdnStepInputs(gpu.q, {128}, am, gm, slots);
  const vllm::actdump::LayerScope dump_scope(1, layer_index);
  auto out = vllm::RunGdnBlockPaged(gpu.q, real.layer.gdn, real.config,
                                  input.t(), step, gm, state, 1, nullptr, DType::kF16);
  REQUIRE(out.tensor.dtype == DType::kF16);
  xpu_test::Buffer copy(gpu.q, DType::kF16, {1, 5120});
  vt::Copy(gpu.q, copy.tensor, out.tensor);
  ExportGdnState("matched-d1-final", conv, ssm);
  CapturedClose("matched_d1.mixer", copy.floats(), real.oracle->Get("d1_mixer_output"), 0.01f, 0.003f);
  const auto cv = conv.floats(), ss = ssm.floats();
  CHECK(std::equal(cv.begin(), cv.begin() + active * channels * history, conv_seed.begin()));
  CHECK(std::equal(ss.begin(), ss.begin() + active * heads * dim * dim, ssm_seed.begin()));
  std::vector<float> active_conv(channels * history);
  for (int c = 0; c < channels; ++c) for (int h = 0; h < history; ++h)
    active_conv[h * channels + c] = cv[(active * channels + c) * history + h];
  CHECK(active_conv == CapturedFloats(real.oracle->Get("d1_conv_state_after")));
  const std::vector<float> active_ssm(ss.begin() + active * heads * dim * dim, ss.end());
  CapturedClose("matched_d1.ssm", active_ssm, real.oracle->Get("d1_ssm_state_after"), 1e-4f, 1e-5f);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block GDN: D1 from matched original Conv and SSM state") {
  RunMatchedRealGdnD1(0);
}

TEST_CASE("XPU EXL3 real layer1 GDN: independent matched-state D1") {
  RunMatchedRealGdnD1(1);
}

TEST_CASE("XPU EXL3 real gated norm: identical P128 D1 operands and strided alias") {
  const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!captures) std::exit(77);
  auto oracle = vllm::SafetensorsFile::Open(
      (std::filesystem::path(captures) / "intermediates.safetensors").string());
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  constexpr int heads = 48, width = 128;
  for (const std::string phase : {"p128", "d1"}) {
    const int tokens = phase == "p128" ? 128 : 1;
    const auto& xv = oracle.Get(phase + "_x");
    const auto& zv = oracle.Get(phase + "_z");
    const auto& wv = oracle.Get(phase + "_weight");
    const auto& expected = oracle.Get(phase + "_endpoint");
    REQUIRE(xv.dtype == "F16"); REQUIRE(zv.dtype == "F16");
    REQUIRE(wv.dtype == "F32"); REQUIRE(expected.dtype == "F16");
    REQUIRE(xv.shape == std::vector<int64_t>{tokens * heads, width});
    const auto xvalues = CapturedFloats(xv), zvalues = CapturedFloats(zv);
    const std::vector<unsigned char> wanted(expected.data, expected.data + expected.nbytes);
    for (auto weight_type : {DType::kF32, DType::kF16})
    for (bool padded : {false, true}) for (bool alias : {false, true}) {
      CAPTURE(phase);
      CAPTURE(weight_type);
      CAPTURE(padded);
      CAPTURE(alias);
      // Actual merged QKVZ gate uses token stride16384, with6144 gate values.
      const int physical_heads = padded ? 128 : heads;
      xpu_test::Buffer x(gpu.q, DType::kF16, {tokens, heads, width});
      xpu_test::Buffer z(gpu.q, DType::kF16, {tokens, physical_heads, width});
      xpu_test::Buffer w(gpu.q, weight_type, {width});
      xpu_test::Buffer out(gpu.q, DType::kF16, {tokens, heads, width});
      x.put(xvalues); w.put(CapturedFloats(wv));
      CHECK(w.floats() == CapturedFloats(wv));
      std::vector<float> gate(tokens * physical_heads * width, 3.0f);
      for (int t = 0; t < tokens; ++t)
        std::copy_n(zvalues.begin() + t * heads * width, heads * width,
                    gate.begin() + t * physical_heads * width);
      z.put(gate); z.tensor.shape[1] = heads;
      auto& target = alias ? x.tensor : out.tensor;
      vt::RmsNormGated(gpu.q, target, x.tensor, z.tensor, w.tensor, {1e-6f, false});
      const auto actual = alias ? x.download() : out.download();
      size_t mismatches = 0;
      for (size_t i = 0; i < actual.size(); i += 2)
        mismatches += std::memcmp(actual.data() + i, wanted.data() + i, 2) != 0;
      std::cout << "GATED_NORM_IDENTICAL " << phase << " padded=" << padded
                << " alias=" << alias << " half_differences=" << mismatches << '\n';
      CHECK(mismatches == 0);
      if (!padded && !alias) {
        xpu_test::Buffer diagnostic(gpu.q, DType::kF32, {tokens, heads, width});
        vt::RmsNormGated(gpu.q, diagnostic.tensor, x.tensor, z.tensor, w.tensor, {1e-6f, false});
        const auto bytes = diagnostic.download();
        const auto& product = oracle.Get(phase + "_product");
        REQUIRE(product.dtype == "F32");
        CHECK(bytes == std::vector<unsigned char>(product.data, product.data + product.nbytes));
        if (const char* directory = std::getenv("VT_B70_EXL3_STATE_OUTPUT")) {
          std::filesystem::create_directories(directory);
          const auto path = std::filesystem::path(directory) /
              (phase + "-gated-product-" + std::string(vt::Name(weight_type)) + ".f32");
          REQUIRE(!std::filesystem::exists(path));
          std::ofstream output(path, std::ios::binary);
          output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
          output.close(); REQUIRE(output.good());
        }
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

static void RunRealGdnD29Detail(int layer_index) {
  // Isolated consumer attribution. Original states are confined to this test;
  // product inference and the failing own-state trajectory remain untouched.
  RealLayer0 real(layer_index, "repeat-0.safetensors");
  const std::string key = "d29_l" + std::to_string(layer_index);
  const std::string label = "matched-D29-layer" + std::to_string(layer_index);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  constexpr int slots = 5, active = 4, channels = 10240, heads = 48, dim = 128, history = 3;
  const auto& original_input = real.oracle->Get(key + "_post_input_norm");
  REQUIRE(original_input.dtype == "F16");
  REQUIRE(original_input.shape == std::vector<int64_t>{1, 5120});
  const auto& original_conv = real.oracle->Get(key + "_detail_conv_before");
  const auto& original_ssm = real.oracle->Get(key + "_detail_ssm_before");
  REQUIRE(original_conv.dtype == "F16");
  REQUIRE(original_conv.shape == std::vector<int64_t>{history, channels});
  REQUIRE(original_ssm.dtype == "F32");
  REQUIRE(original_ssm.shape == std::vector<int64_t>{heads, dim, dim});
  const auto captured_conv = CapturedFloats(original_conv);
  const auto captured_ssm = CapturedFloats(original_ssm);
  std::vector<float> conv_seed(slots * channels * history, 0.125f);
  std::vector<float> ssm_seed(slots * heads * dim * dim, 0.125f);
  for (int c = 0; c < channels; ++c) for (int h = 0; h < history; ++h)
    conv_seed[(active * channels + c) * history + h] = captured_conv[h * channels + c];
  std::copy(captured_ssm.begin(), captured_ssm.end(), ssm_seed.begin() + active * heads * dim * dim);
  xpu_test::Buffer conv(gpu.q, DType::kF16, {slots, channels, history});
  xpu_test::Buffer ssm(gpu.q, DType::kF32, {slots, heads, dim, dim});
  conv.put(conv_seed); ssm.put(ssm_seed);
  ExportGdnState(label + "-initial", conv, ssm);
  vllm::GdnStateCache state; state.conv_state = conv.tensor; state.ssm_state = ssm.tensor;
  vllm::dense_attn::DBuf input(d, DType::kF16, {1, 5120}, original_input.data);
  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = am.num_actual_tokens = 1;
  am.query_start_loc = am.query_start_loc_cpu = {0, 1};
  am.seq_lens = am.seq_lens_cpu = {157};
  am.max_query_len = 1; am.max_seq_len = 157;
  am.block_table_num_cols = 1; am.block_table_tensor = {0}; am.slot_mapping = {0};
  vllm::v1::GDNAttentionMetadata gm;
  gm.num_actual_tokens = gm.num_decodes = gm.num_decode_tokens = 1;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{active};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
  auto step = vllm::BuildGdnStepInputs(gpu.q, {156}, am, gm, slots);
  const vllm::actdump::LayerScope dump_scope(29, layer_index);
  auto out = vllm::RunGdnBlockPaged(gpu.q, real.layer.gdn, real.config,
                                  input.t(), step, gm, state, 1, nullptr, DType::kF16);
  REQUIRE(out.tensor.dtype == DType::kF16);
  xpu_test::Buffer copy(gpu.q, DType::kF16, {1, 5120});
  vt::Copy(gpu.q, copy.tensor, out.tensor);
  ExportGdnState(label + "-final", conv, ssm);
  CapturedClose(label + ".mixer", copy.floats(),
                real.oracle->Get(key + "_detail_mixer_output"), 0.01f, 0.003f);
  const auto& expected_mixer = real.oracle->Get(key + "_detail_mixer_output");
  CHECK(copy.download() == std::vector<unsigned char>(expected_mixer.data,
      expected_mixer.data + expected_mixer.nbytes));
  const auto cv = conv.floats(), ss = ssm.floats();
  CHECK(std::equal(cv.begin(), cv.begin() + active * channels * history, conv_seed.begin()));
  CHECK(std::equal(ss.begin(), ss.begin() + active * heads * dim * dim, ssm_seed.begin()));
  std::vector<float> active_conv(channels * history);
  for (int c = 0; c < channels; ++c) for (int h = 0; h < history; ++h)
    active_conv[h * channels + c] = cv[(active * channels + c) * history + h];
  CHECK(active_conv == CapturedFloats(real.oracle->Get(key + "_detail_conv_after")));
  const std::vector<float> active_ssm(ss.begin() + active * heads * dim * dim, ss.end());
  CapturedClose(label + ".ssm", active_ssm,
                real.oracle->Get(key + "_detail_ssm_after"), 1e-4f, 1e-5f);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real GDN layer1: D29 identical original input and active states") {
  RunRealGdnD29Detail(1);
}

TEST_CASE("XPU EXL3 real GDN layer21: D29 identical original input and active states") {
  RunRealGdnD29Detail(21);
}

static void RunRealBaRepeats(int layer_index) {
  RealLayer0 real(layer_index);
  const char* directory = std::getenv("VT_B70_EXL3_STATE_OUTPUT");
  if (!directory) std::exit(77);
  const std::filesystem::path path(directory);
  std::filesystem::create_directories(path);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  const auto& owner = real.layer.gdn.in_proj_ba;
  REQUIRE(owner.nk); REQUIRE(owner.dtype == DType::kF16);
  REQUIRE(owner.shape[0] == 96); REQUIRE(owner.shape[1] == 5120);
  const auto weight = vllm::dense_attn::ResidentWeight(d, owner);
  vt::xpu::DrainProfileEvents();
  nlohmann::json result = {{"purpose", "Actual production VT MatmulDenseF16 fixed-input repeatability gate; not oracle parity/performance"},
                           {"layer_index", layer_index},
                           {"activation_dtype", "f16"}, {"weight_shape", {96, 5120}},
                           {"weight_strides", {5120, 1}}, {"phases", nlohmann::json::object()}};
  for (const auto& phase : {std::string("p128"), std::string("d1")}) {
    const int rows = phase == "p128" ? 128 : 1;
    const auto& hidden = real.oracle->Get(phase + "_input_norm_output");
    REQUIRE(hidden.dtype == "F16"); REQUIRE(hidden.shape == std::vector<int64_t>{rows, 5120});
    vllm::dense_attn::DBuf input(d, DType::kF16, {rows, 5120}, hidden.data);
    xpu_test::Buffer output(gpu.q, DType::kF16, {rows, 96});
    std::vector<std::vector<uint16_t>> repeats;
    for (int repeat = 0; repeat < 3; ++repeat) {
      vt::MatmulDenseF16(gpu.q, output.tensor, input.t(), weight);
      const auto bytes = output.download();
      std::vector<uint16_t> bits(bytes.size() / 2);
      std::memcpy(bits.data(), bytes.data(), bytes.size()); repeats.push_back(bits);
      const auto file = path / (phase + "-vt-ba-repeat" + std::to_string(repeat) + ".f16");
      REQUIRE(!std::filesystem::exists(file));
      std::ofstream raw(file, std::ios::binary);
      raw.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      raw.close(); REQUIRE(raw.good());
      REQUIRE(std::all_of(bits.begin(), bits.end(), [](uint16_t v) { return std::isfinite(vt::F16ToF32(v)); }));
    }
    auto& summary = result["phases"][phase];
    summary["shape"] = {rows, 96}; summary["pair_half_differences"] = nlohmann::json::array();
    for (int first = 0; first < 3; ++first) for (int second = first + 1; second < 3; ++second) {
      const auto different = std::inner_product(repeats[first].begin(), repeats[first].end(),
          repeats[second].begin(), size_t{0}, std::plus<size_t>(),
          [](uint16_t a, uint16_t b) { return size_t(a != b); });
      summary["pair_half_differences"].push_back({{"first", first}, {"second", second}, {"different", different}});
      std::cout << "ACTUAL_VT_BA phase=" << phase << " first=" << first << " second=" << second
                << " half_differences=" << different << '\n';
      CHECK(different == 0);
    }
  }
  result["selected_implementations"] = nlohmann::json::array();
  for (const auto& event : vt::xpu::DrainProfileEvents())
    if (event.stage == "onednn_dense_f16_stream")
      result["selected_implementations"].push_back(event.matrix);
  const auto file = path / "actual-vt-ba-repeats.json";
  REQUIRE(!std::filesystem::exists(file));
  std::ofstream report(file); report << result.dump(2) << '\n'; report.close(); REQUIRE(report.good());
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real BA: actual VT identical operands three repeats P128 D1") {
  RunRealBaRepeats(0);
}

TEST_CASE("XPU EXL3 real layer1 BA: actual VT identical operands three repeats P128 D1") {
  RunRealBaRepeats(1);
}

TEST_CASE("XPU EXL3 real block MLP: captured P128 and D1 complete native MLP") {
  RealLayer0 real;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  vllm::OwnedTensor empty;
  for (const std::string phase : {"p128", "d1"}) {
    CAPTURE(phase);
    const int rows = phase == "p128" ? 128 : 1;
    const auto& x = real.oracle->Get(phase+"_post_norm_output");
    REQUIRE(x.dtype == "F16"); REQUIRE(x.shape == std::vector<int64_t>{rows, 5120});
    vllm::dense_attn::DBuf input(d, DType::kF16, {rows, 5120}, x.data);
    auto act = vllm::dense_exl3::GateUp(d, input.t(), empty,
        real.layer.mlp.gate_proj_exl3, real.layer.mlp.up_proj_exl3,
        real.config.intermediate_size, &real.layer.mlp.gate_up_exl3);
    std::vector<uint16_t> raw(rows*real.config.intermediate_size);
    act.Download(d, raw.data());
    std::vector<float> activated(raw.size());
    for (size_t i = 0; i < raw.size(); ++i) activated[i] = vt::F16ToF32(raw[i]);
    CapturedClose(phase+".swiglu", activated, real.oracle->Get(phase+"_swiglu_output"), 0.002f, 1e-4f);
    auto down = vllm::dense_exl3::Linear(d, act.t(), empty, real.layer.mlp.down_proj_exl3, DType::kF16);
    std::vector<uint16_t> down_raw(rows*5120); down.Download(d, down_raw.data());
    std::vector<float> got(down_raw.size());
    for (size_t i = 0; i < down_raw.size(); ++i) got[i] = vt::F16ToF32(down_raw[i]);
    CapturedClose(phase+".down", got, real.oracle->Get(phase+"_hidden_out"), 0.002f, 1e-4f);
    CHECK(empty.Empty());
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real layer20 MLP: identical P128 projection and activation operands") {
  RealLayer0 real(20, "repeat-0.safetensors");
  const char* output = std::getenv("VT_B70_EXL3_DIAGNOSTIC_OUTPUT");
  REQUIRE(output != nullptr);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  constexpr int rows = 128, hidden = 5120, intermediate = 17408;
  REQUIRE(real.config.intermediate_size == intermediate);
  auto& w = real.layer.mlp;
  REQUIRE(w.IsExl3());
  const auto observe = [&](const std::string& stage, vllm::dense_attn::DBuf& result,
                            const std::string& key, int width) {
    const auto& expected = real.oracle->Get(key);
    REQUIRE(expected.dtype == "F16");
    REQUIRE(expected.shape == std::vector<int64_t>{rows, width});
    REQUIRE(result.t().dtype == DType::kF16);
    std::vector<uint16_t> raw(rows * width); result.Download(d, raw.data());
    size_t different = 0;
    for (size_t i = 0; i < raw.size(); ++i)
      different += raw[i] != vt::LoadUnaligned<uint16_t>(expected.data + i * 2);
    const auto path = std::filesystem::path(output) / ("mlp20-" + stage + ".f16");
    REQUIRE_FALSE(std::filesystem::exists(path));
    std::ofstream file(path, std::ios::binary);
    file.write(reinterpret_cast<const char*>(raw.data()), raw.size() * 2);
    file.close(); REQUIRE(file.good());
    std::cout << "MLP20_SAME_INPUT stage=" << stage << " different_halves=" << different << '\n';
    CHECK(different == 0);
  };
  const auto upload = [&](const std::string& key, int width) {
    const auto& value = real.oracle->Get(key);
    REQUIRE(value.dtype == "F16");
    REQUIRE(value.shape == std::vector<int64_t>{rows, width});
    return vllm::dense_attn::DBuf(d, DType::kF16, {rows, width}, value.data);
  };
  auto input = upload("p128_l20_post_attn_norm", hidden);
  auto gate_up = vllm::dense_exl3::GroupedLinear(d, input.t(),
      {&w.gate_proj_exl3, &w.up_proj_exl3}, w.gate_up_exl3);
  observe("gate-up", gate_up, "p128_l20_detail_gate_up", 2 * intermediate);
  auto original_gate_up = upload("p128_l20_detail_gate_up", 2 * intermediate);
  vllm::dense_attn::DBuf isolated_act(d, DType::kF16, {rows, intermediate});
  vt::SiluAndMul(gpu.q, isolated_act.t(), original_gate_up.t());
  observe("isolated-swiglu", isolated_act, "p128_l20_detail_swiglu", intermediate);
  vllm::OwnedTensor empty;
  auto chain_act = vllm::dense_exl3::GateUp(d, input.t(), empty, w.gate_proj_exl3,
      w.up_proj_exl3, intermediate, &w.gate_up_exl3);
  observe("chain-swiglu", chain_act, "p128_l20_detail_swiglu", intermediate);
  auto original_act = upload("p128_l20_detail_swiglu", intermediate);
  auto isolated_down = vllm::dense_exl3::Linear(d, original_act.t(), empty,
      w.down_proj_exl3, DType::kF16);
  observe("isolated-down", isolated_down, "p128_l20_detail_down", hidden);
  auto chain_down = vllm::dense_exl3::Linear(d, chain_act.t(), empty,
      w.down_proj_exl3, DType::kF16);
  observe("chain-down", chain_down, "p128_l20_detail_down", hidden);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block Gemma: FP32 sum before FP16 residual rounding") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  // The independent producer IR computes norm(x.float()+res.float()) and
  // returns that sum narrowed separately as residual. These exact half inputs
  // distinguish that boundary from normalizing the narrowed residual, in both
  // the narrow leaf and the actual target's 5120-column work-group reduction.
  for (int width : {2, 5120}) for (int alias : {0, 1, 2}) {
    CAPTURE(width);
    CAPTURE(alias);
    xpu_test::Buffer input(gpu.q, DType::kF16, {1, width});
    xpu_test::Buffer residual(gpu.q, DType::kF16, {1, width});
    xpu_test::Buffer weight(gpu.q, DType::kF16, {width});
    xpu_test::Buffer output(gpu.q, DType::kF16, {1, width});
    std::vector<float> res(width), expected(width), stored(width);
    for (int i = 0; i < width; ++i) {
      res[i] = i % 2 ? 0.000244140625f : 0.00341796875f;
      expected[i] = i % 2 ? 0.99853515625f : 1.001953125f;
      stored[i] = i % 2 ? 1.0f : 1.00390625f;
    }
    input.put(std::vector<float>(width, 1)); residual.put(res);
    weight.put(std::vector<float>(width, 0));
    auto target = alias == 1 ? input.tensor : alias == 2 ? residual.tensor : output.tensor;
    vt::RmsNorm(gpu.q, target, input.tensor, weight.tensor,
                vt::RmsNormArgs{1e-6f, true}, &residual.tensor);
    std::vector<uint16_t> got(width);
    vt::GetBackend(gpu.q.device).Copy(gpu.q, got.data(), target.data, width*2);
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    bool same = true;
    for (int i = 0; i < width; ++i) same &= got[i] == vt::F32ToF16(expected[i]);
    CHECK(same);
    // If output aliases residual, the output replaces the updated residual
    // after both old inputs have been consumed; otherwise both remain visible.
    CHECK(residual.floats() == (alias == 2 ? expected : stored));
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real block Gemma: captured P128 and D1 normalization stages") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* captures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!model || !captures) {
    std::cerr << "SKIP: set VT_B70_EXL3_MODEL and VT_B70_EXL3_S1_FIXTURES.\n";
    std::exit(77);
  }
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto config = vllm::LoadHfConfig((std::filesystem::path(model) / "config.json").string());
  auto shard = vllm::SafetensorsFile::Open((std::filesystem::path(model) / "model-00001-of-00002.safetensors").string());
  const auto get = [&](const std::string& name) -> const vllm::StTensor& { return shard.Get(name); };
  const auto has = [&](const std::string& name) {
    return std::find(shard.Names().begin(), shard.Names().end(), name) != shard.Names().end();
  };
  const auto layer = vllm::LoadQwen3_5DenseLayer(get, has, "linear_attention", 0, "model.language_model.");
  auto oracle = vllm::SafetensorsFile::Open((std::filesystem::path(captures) / "real_block_oracle.safetensors").string());
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  REQUIRE(config.hidden_size == 5120);
  for (const std::string phase : {"p128", "d1"}) for (bool post : {false, true}) {
    CAPTURE(phase);
    CAPTURE(post);
    const int64_t rows = phase == "p128" ? 128 : 1;
    const auto& input = oracle.Get(phase + (post ? "_post_norm_input" : "_hidden_in"));
    const auto& expected = oracle.Get(phase + (post ? "_post_norm_output" : "_input_norm_output"));
    REQUIRE(input.dtype == "F16"); REQUIRE(expected.dtype == "F16");
    REQUIRE(input.shape == std::vector<int64_t>{rows, 5120});
    REQUIRE(expected.shape == input.shape);
    xpu_test::Buffer x(gpu.q, DType::kF16, {rows, 5120});
    xpu_test::Buffer out(gpu.q, DType::kF16, {rows, 5120});
    xpu_test::Buffer res(gpu.q, DType::kF16, {rows, 5120});
    x.upload(input.data);
    if (post) res.upload(oracle.Get(phase + "_post_norm_residual_input").data);
    const auto w = vllm::dense_attn::ResidentWeight(d, post ? layer.post_attention_layernorm : layer.input_layernorm);
    vt::RmsNorm(gpu.q, out.tensor, x.tensor, w,
                vt::RmsNormArgs{static_cast<float>(config.rms_norm_eps), true}, post ? &res.tensor : nullptr);
    const auto actual = out.floats();
    std::vector<float> want(actual.size());
    double error = 0, norm = 0; float max_error = 0; size_t different = 0;
    bool finite = true;
    const auto bytes = out.download();
    for (size_t i = 0; i < want.size(); ++i) {
      uint16_t bit; std::memcpy(&bit, expected.data + i*2, 2); want[i] = vt::F16ToF32(bit);
      finite &= std::isfinite(actual[i]);
      const double delta = double(actual[i]) - want[i]; error += delta*delta; norm += double(want[i])*want[i];
      max_error = std::max(max_error, std::abs(actual[i] - want[i]));
      different += std::memcmp(bytes.data() + i*2, expected.data + i*2, 2) != 0;
    }
    REQUIRE(finite);
    REQUIRE(norm > 0);
    std::cout << "REAL_BLOCK_NORM phase=" << phase << " post=" << post
              << " relative=" << std::sqrt(error/norm) << " max_error=" << max_error
              << " different_half_values=" << different << '\n';
    // Fixed independent producer IR fused_add_rms_norm tolerance.
    CHECK(std::sqrt(error/norm) < 2e-3);
    xpu_test::Close(actual, want, 2e-3f, 1e-2f);
    if (post) {
      const auto& expected_res = oracle.Get(phase + "_post_norm_residual_output");
      REQUIRE(expected_res.shape == input.shape);
      xpu_test::SameBytes(res.download(), std::vector<unsigned char>(expected_res.data, expected_res.data + expected_res.nbytes));
    }
    // The producer F16 Gemma path must preserve the same captured boundary
    // when WithOutput stages an output that aliases either read operand.
    const std::vector<unsigned char> expected_bytes(expected.data, expected.data + expected.nbytes);
    xpu_test::SameBytes(bytes, expected_bytes);
    for (int alias : {1, 2}) {
      if (alias == 2 && !post) continue;
      CAPTURE(alias);
      x.upload(input.data);
      if (post) res.upload(oracle.Get(phase + "_post_norm_residual_input").data);
      auto& target = alias == 1 ? x : res;
      vt::RmsNorm(gpu.q, target.tensor, x.tensor, w,
                  vt::RmsNormArgs{static_cast<float>(config.rms_norm_eps), true},
                  post ? &res.tensor : nullptr);
      xpu_test::SameBytes(target.download(), expected_bytes);
      if (post && alias == 1) {
        const auto& expected_res = oracle.Get(phase + "_post_norm_residual_output");
        xpu_test::SameBytes(res.download(), std::vector<unsigned char>(
            expected_res.data, expected_res.data + expected_res.nbytes));
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 Gemma5120: real short rows aliases strides and poisoned padding") {
  const char* path = std::getenv("VT_B70_EXL3_GEMMA_ROWS_FIXTURE");
  if (!path) std::exit(77);
  const std::filesystem::path fixture(path);
  auto metadata = fixture; metadata.replace_extension(".json");
  std::ifstream record(metadata);
  const auto cases = nlohmann::json::parse(record).at("cases");
  const auto oracle = vllm::SafetensorsFile::Open(fixture.string());
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  for (const auto& entry : cases) {
    const auto label = entry.at("label").get<std::string>();
    CAPTURE(label);
    const int rows = entry.at("physical_rows"), active = entry.at("logical_rows");
    const bool residual = entry.at("residual");
    REQUIRE(active > 0); REQUIRE(active <= rows);
    const auto& input = oracle.Get(label + "_input");
    const auto& expected = oracle.Get(label + "_output");
    REQUIRE(input.dtype == "F16"); REQUIRE(expected.dtype == "F16");
    REQUIRE(input.shape == std::vector<int64_t>{rows, 5120});
    xpu_test::Buffer weight(gpu.q, DType::kF16, {5120});
    weight.upload(oracle.Get(label + "_weight").data);
    for (bool strided : {false, true}) for (int alias : {0, 1, 2}) {
      if (alias == 2 && !residual) continue;
      CAPTURE(strided);
      CAPTURE(alias);
      const int leading = 5120 + (strided ? 16 : 0), border = strided ? 1 : 0, offset = strided ? 8 : 0;
      xpu_test::Buffer xb(gpu.q, DType::kF16, {rows + 2 * border, leading});
      xpu_test::Buffer rb(gpu.q, DType::kF16, {rows + 2 * border, leading});
      xpu_test::Buffer ob(gpu.q, DType::kF16, {rows + 2 * border, leading});
      std::vector<uint16_t> poison((rows + 2 * border) * leading, 0x7e00);
      const auto upload = [&](xpu_test::Buffer& owner, const vllm::StTensor& value) {
        auto data = poison;
        for (int row = 0; row < active; ++row)
          std::memcpy(data.data() + (row + border) * leading + offset,
                      value.data + row * 5120 * 2, 5120 * 2);
        // Inactive physical rows stay poisoned. They must not affect a logical
        // row, even when physical M changes the producer reduction geometry.
        owner.upload(data.data());
      };
      upload(xb, input); ob.upload(poison.data()); rb.upload(poison.data());
      if (residual) upload(rb, oracle.Get(label + "_residual"));
      auto x = xb.tensor.Slice(0, border, border + rows).Slice(1, offset, offset + 5120);
      auto res = rb.tensor.Slice(0, border, border + rows).Slice(1, offset, offset + 5120);
      auto out = alias == 1 ? x : alias == 2 ? res :
          ob.tensor.Slice(0, border, border + rows).Slice(1, offset, offset + 5120);
      if (strided) {
        // The public VT RMSNorm contract requires contiguous rows. Keep this
        // unsupported layout explicit and prove refusal precedes any write.
        const auto before_x = xb.download(), before_res = rb.download(), before_out = ob.download();
        CHECK_THROWS_WITH_AS(vt::RmsNorm(gpu.q, out, x, weight.tensor,
            vt::RmsNormArgs{1e-6f, true}, residual ? &res : nullptr),
            doctest::Contains("rmsnorm: contiguous required"), std::runtime_error);
        xpu_test::SameBytes(xb.download(), before_x);
        xpu_test::SameBytes(rb.download(), before_res);
        xpu_test::SameBytes(ob.download(), before_out);
        continue;
      }
      vt::RmsNorm(gpu.q, out, x, weight.tensor, vt::RmsNormArgs{1e-6f, true}, residual ? &res : nullptr);
      const auto logical = [&](const xpu_test::Buffer& owner) {
        const auto storage = owner.download();
        std::vector<unsigned char> data(active * 5120 * 2);
        for (int row = 0; row < active; ++row)
          std::memcpy(data.data() + row * 5120 * 2,
                      storage.data() + ((row + border) * leading + offset) * 2, 5120 * 2);
        return data;
      };
      const auto want = [&](const vllm::StTensor& value) {
        return std::vector<unsigned char>(value.data, value.data + active * 5120 * 2);
      };
      xpu_test::SameBytes(logical(alias == 1 ? xb : alias == 2 ? rb : ob), want(expected));
      if (residual) xpu_test::SameBytes(logical(rb), want(oracle.Get(
          label + (alias == 2 ? "_output" : "_stored_residual"))));
      if (alias != 1) xpu_test::SameBytes(logical(xb), want(input));
      for (const auto* owner : {&xb, &rb, &ob}) {
        const auto storage = owner->download();
        bool guards_unchanged = true;
        for (int row = 0; row < rows + 2 * border; ++row) for (int col = 0; col < leading; ++col) {
          if (row >= border && row < border + rows && col >= offset && col < offset + 5120) continue;
          uint16_t bits; std::memcpy(&bits, storage.data() + (row * leading + col) * 2, 2);
          guards_unchanged &= bits == 0x7e00;
        }
        CHECK(guards_unchanged);
      }
    }
    std::cout << "GEMMA_SHORT_ROWS " << label << " PASS\n";
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 Gemma5120: bounded same-input operator timing") {
  const char* path = std::getenv("VT_B70_EXL3_GEMMA_ROWS_FIXTURE");
  const char* output = std::getenv("VT_B70_EXL3_GEMMA_PERF_OUTPUT");
  if (!path || !output) std::exit(77);
  REQUIRE_FALSE(std::filesystem::exists(output));
  const bool profiled = std::getenv("VT_XPU_PROFILE") &&
      std::string(std::getenv("VT_XPU_PROFILE")) == "1";
  const std::filesystem::path fixture(path);
  auto metadata = fixture; metadata.replace_extension(".json");
  std::ifstream record(metadata);
  const auto cases = nlohmann::json::parse(record).at("cases");
  const auto oracle = vllm::SafetensorsFile::Open(fixture.string());
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  nlohmann::json report{{"schema", "b70-exl3-gemma5120-operator-timing-v1"},
      {"profiled", profiled}, {"iterations", 128}, {"cases", nlohmann::json::array()},
      {"scope", "Identical real operands; complete norm with D2D residual reset per call. "
          "Batch completion wait only; not serving performance or compiled-original qualification."}};
  for (const auto& entry : cases) {
    const auto label = entry.at("label").get<std::string>();
    CAPTURE(label);
    const int rows = entry.at("physical_rows");
    const bool residual = entry.at("residual");
    xpu_test::Buffer x(gpu.q, DType::kF16, {rows, 5120});
    xpu_test::Buffer r(gpu.q, DType::kF16, {rows, 5120});
    xpu_test::Buffer initial(gpu.q, DType::kF16, {rows, 5120});
    xpu_test::Buffer out(gpu.q, DType::kF16, {rows, 5120});
    xpu_test::Buffer weight(gpu.q, DType::kF16, {5120});
    x.upload(oracle.Get(label + "_input").data);
    weight.upload(oracle.Get(label + "_weight").data);
    if (residual) initial.upload(oracle.Get(label + "_residual").data);
    const auto invoke = [&] {
      if (residual) vt::Copy(gpu.q, r.tensor, initial.tensor);
      vt::RmsNorm(gpu.q, out.tensor, x.tensor, weight.tensor,
                  vt::RmsNormArgs{1e-6f, true}, residual ? &r.tensor : nullptr);
    };
    for (int i = 0; i < 16; ++i) invoke();
    vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
    (void)vt::xpu::DrainProfileEvents();
    nlohmann::json wall = nlohmann::json::array();
    nlohmann::json event_times = nlohmann::json::array();
    for (int batch = 0; batch < 3; ++batch) {
      const auto start = std::chrono::steady_clock::now();
      for (int i = 0; i < 128; ++i) invoke();
      vt::GetBackend(gpu.q.device).Synchronize(gpu.q);
      wall.push_back(std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - start).count() / 128);
      auto events = vt::xpu::DrainProfileEvents();
      if (profiled) {
        size_t norms = 0;
        double ms = 0;
        for (const auto& event : events) if (event.stage == "rms_norm_gemma5120_fp16") {
          ++norms; ms += (event.end_ns - event.start_ns) / 1e6;
        }
        REQUIRE(norms == 128);
        event_times.push_back(ms / 128);
      } else CHECK(events.empty());
    }
    const auto want = [&](const std::string& suffix) {
      const auto& tensor = oracle.Get(label + suffix);
      return std::vector<unsigned char>(tensor.data, tensor.data + rows * 5120 * 2);
    };
    xpu_test::SameBytes(out.download(), want("_output"));
    xpu_test::SameBytes(x.download(), want("_input"));
    if (residual) xpu_test::SameBytes(r.download(), want("_stored_residual"));
    report["cases"].push_back({{"label", label}, {"rows", rows}, {"residual", residual},
        {"complete_wall_ms_per_call", wall}, {"norm_event_ms_per_call", event_times}});
  }
  CHECK(vt::GetReferenceTierHits() == 0);
  std::ofstream stream(output); REQUIRE(stream.good());
  stream << report.dump(2) << '\n';
}

static void ExportAttention3Kv(const std::string& phase, vt::Queue& q,
                               const vllm::PagedKvCache& cache, int length) {
  const char* directory = std::getenv("VT_B70_EXL3_STATE_OUTPUT");
  REQUIRE(directory != nullptr);
  REQUIRE(cache.dtype == DType::kI8);
  REQUIRE(cache.num_blocks == 1);
  REQUIRE((length == 156 || length == 157));
  REQUIRE(cache.num_kv_heads == 4);
  REQUIRE(cache.head_size == 256);
  std::filesystem::create_directories(directory);
  nlohmann::json layout;
  for (int which = 0; which < 2; ++which) {
    auto source = vllm::dense_attn::KvSlice(cache, q.device, which);
    source.shape[1] = length;
    xpu_test::Buffer compact(q, DType::kI8, {1, length, 4, 256});
    vt::Copy(q, compact.tensor, source);
    const auto bytes = compact.download();
    const std::string label = which == 0 ? "key" : "value";
    const auto path = std::filesystem::path(directory) / (phase + "-" + label + ".bin");
    REQUIRE(!std::filesystem::exists(path));
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    output.close(); REQUIRE(output.good());
    layout[label] = {{"dtype", "U8"}, {"logical_shape", {length, 4, 256}},
        {"source_shape", std::vector<int64_t>(source.shape, source.shape + source.rank)},
        {"source_strides", std::vector<int64_t>(source.stride, source.stride + source.rank)},
        {"scale", which == 0 ? cache.k_scale : cache.v_scale}};
  }
  const auto path = std::filesystem::path(directory) / (phase + ".json");
  REQUIRE(!std::filesystem::exists(path));
  std::ofstream output(path); output << layout.dump(2) << '\n';
  output.close(); REQUIRE(output.good());
}

static void CheckHeldOutGdnStates(const std::string& phase, const std::string& stage,
                                 vt::Queue& q, const vllm::Qwen3_5DenseWeights& weights,
                                 const std::vector<vllm::GdnStateCache>& states,
                                 const vllm::SafetensorsFile& oracle) {
  const char* directory = std::getenv("VT_B70_EXL3_STATE_OUTPUT");
  REQUIRE(directory != nullptr);
  std::filesystem::create_directories(directory);
  constexpr size_t conv_elements = 10240 * 3, ssm_elements = 48 * 128 * 128;
  auto& backend = vt::GetBackend(q.device.type);
  size_t state_index = 0;
  for (size_t layer = 0; layer < weights.layers.size(); ++layer) {
    if (!weights.layers[layer].is_linear_attention) continue;
    const auto& state = states.at(state_index++);
    REQUIRE(state.conv_state.dtype == DType::kF16);
    REQUIRE(state.ssm_state.dtype == DType::kF32);
    REQUIRE(state.conv_state.IsContiguous());
    REQUIRE(state.ssm_state.IsContiguous());
    REQUIRE(state.conv_state.shape[0] == 5);
    REQUIRE(state.ssm_state.shape[0] == 5);
    const std::string key = phase + "_l" + std::to_string(layer) + "_" + stage;
    std::vector<uint16_t> conv(conv_elements), canonical(conv_elements);
    std::vector<float> ssm(ssm_elements);
    backend.Copy(q, conv.data(), static_cast<const uint16_t*>(state.conv_state.data) +
                 4 * conv_elements, conv_elements * 2);
    backend.Copy(q, ssm.data(), static_cast<const float*>(state.ssm_state.data) +
                 4 * ssm_elements, ssm_elements * 4);
    backend.Synchronize(q);
    // Original Conv cache is [history,channel]; native is [channel,history].
    for (size_t channel = 0; channel < 10240; ++channel)
      for (size_t history = 0; history < 3; ++history)
        canonical[history * 10240 + channel] = conv[channel * 3 + history];
    const auto& expected_conv = oracle.Get(key + "_conv");
    REQUIRE(expected_conv.dtype == "F16");
    REQUIRE((expected_conv.shape == std::vector<int64_t>{3, 10240}));
    const auto& expected_ssm = oracle.Get(key + "_ssm");
    REQUIRE(expected_ssm.dtype == "F32");
    REQUIRE((expected_ssm.shape == std::vector<int64_t>{48, 128, 128}));
    // Export only initialized active bytes, before any comparison can fail.
    for (const auto& entry : {std::make_tuple("conv.f16", static_cast<const void*>(canonical.data()),
                                            conv_elements * 2),
                              std::make_tuple("ssm.f32", static_cast<const void*>(ssm.data()),
                                            ssm_elements * 4)}) {
      const auto file = std::filesystem::path(directory) / (key + "-" + std::get<0>(entry));
      REQUIRE_FALSE(std::filesystem::exists(file));
      std::ofstream output(file, std::ios::binary);
      output.write(static_cast<const char*>(std::get<1>(entry)), std::get<2>(entry));
      output.close(); REQUIRE(output.good());
    }
    CHECK(std::memcmp(canonical.data(), expected_conv.data, conv_elements * 2) == 0);
    const auto expected = CapturedFloats(expected_ssm);
    REQUIRE(expected.size() == ssm.size());
    bool finite = true;
    size_t different = 0, pointwise_failures = 0;
    double max_error = 0;
    for (size_t i = 0; i < ssm.size(); ++i) {
      finite &= std::isfinite(ssm[i]) && std::isfinite(expected[i]);
      const double error = std::abs(double(ssm[i]) - expected[i]);
      max_error = std::max(max_error, error);
      different += ssm[i] != expected[i];
      pointwise_failures += error > 1e-5 + 1e-4 * std::abs(double(expected[i]));
    }
    std::cout << "HELD_OUT_GDN_STATE key=" << key << " different=" << different
              << " max_error=" << max_error << " pointwise_failures=" << pointwise_failures << '\n';
    REQUIRE(finite);
    CHECK(pointwise_failures == 0);
  }
  REQUIRE(state_index == 48);
}

static void CheckHeldOutAttentionKv(const std::string& phase, const std::string& stage,
                                    vt::Queue& q, const vllm::Qwen3_5DenseWeights& weights,
                                    const std::vector<vllm::PagedKvCache>& caches, int length,
                                    const vllm::SafetensorsFile& oracle) {
  const char* directory = std::getenv("VT_B70_EXL3_STATE_OUTPUT");
  REQUIRE(directory != nullptr);
  REQUIRE((length == 128 || length == 129));
  size_t cache_index = 0;
  for (size_t layer = 0; layer < weights.layers.size(); ++layer) {
    if (weights.layers[layer].is_linear_attention) continue;
    const auto& cache = caches.at(cache_index++);
    REQUIRE(cache.dtype == DType::kI8);
    REQUIRE(cache.fp8_kind == vt::Fp8KVCacheDataType::kFp8E4M3);
    REQUIRE(cache.num_blocks == 1); REQUIRE(cache.block_size == 1600);
    REQUIRE(cache.num_kv_heads == 4); REQUIRE(cache.head_size == 256);
    REQUIRE(cache.k_scale == 1.0f); REQUIRE(cache.v_scale == 1.0f);
    const std::string key = phase + "_l" + std::to_string(layer) + "_" + stage;
    for (int which = 0; which < 2; ++which) {
      auto source = vllm::dense_attn::KvSlice(cache, q.device, which);
      source.shape[1] = length;
      xpu_test::Buffer compact(q, DType::kI8, {1, length, 4, 256});
      vt::Copy(q, compact.tensor, source);
      const auto bytes = compact.download();
      const std::string label = which == 0 ? "key" : "value";
      const auto& expected = oracle.Get(key + "_" + label);
      REQUIRE(expected.dtype == "U8");
      REQUIRE((expected.shape == std::vector<int64_t>{length, 4, 256}));
      REQUIRE(expected.nbytes == bytes.size());
      const auto file = std::filesystem::path(directory) / (key + "-" + label + ".u8");
      REQUIRE_FALSE(std::filesystem::exists(file));
      std::ofstream output(file, std::ios::binary);
      output.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      output.close(); REQUIRE(output.good());
      const size_t different = std::inner_product(bytes.begin(), bytes.end(), expected.data,
          size_t{0}, std::plus<size_t>(), [](auto a, auto b) { return size_t(a != b); });
      std::cout << "HELD_OUT_ATTENTION_KV key=" << key << '_' << label
                << " bytes=" << bytes.size() << " different=" << different << '\n';
      CHECK(different == 0);
    }
  }
  REQUIRE(cache_index == 16);
}

static void RunRealEagerTarget(int decode_steps, int diagnostic_stop = -1,
                               bool attention3_detail = false, int gdn_detail_layer = 1,
                               bool gdn_history = false, bool held_out_states = false,
                               bool held_out_kv = false) {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* fixtures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  if (!model || !fixtures) std::exit(77);
  const std::filesystem::path model_dir(model), receipts(fixtures);
  REQUIRE((decode_steps == 1 || decode_steps == 64));
  REQUIRE((diagnostic_stop == -1 || (decode_steps == 64 && (diagnostic_stop == 29 || diagnostic_stop == 1)) ||
           (decode_steps == 1 && diagnostic_stop == 1)));
  const bool diagnostic = diagnostic_stop >= 0;
  if (held_out_kv) REQUIRE(held_out_states);
  if (held_out_states) {
    REQUIRE(decode_steps == 1); REQUIRE(diagnostic_stop == 1);
    REQUIRE_FALSE(attention3_detail); REQUIRE_FALSE(gdn_history);
    REQUIRE(std::getenv("VT_B70_EXL3_STATE_OUTPUT") != nullptr);
  }
  if (gdn_history) { REQUIRE(diagnostic_stop == 1); REQUIRE(gdn_detail_layer == 21); }
  const vllm::actdump::StepSelectionScope dump_selection(
      diagnostic && diagnostic_stop != 1 ? diagnostic_stop : -1);
  const vllm::actdump::StageLayerSelectionScope kv_preimage_selection(
      held_out_kv ? 43 : vllm::actdump::SelectedStageLayer());
  const char* diagnostic_output = std::getenv("VT_B70_EXL3_DIAGNOSTIC_OUTPUT");
  if (diagnostic && !diagnostic_output) std::exit(77);
  // Full D64 qualification can keep immutable reference fixtures read-only
  // while writing its complete outputs to an isolated receipt directory.
  const auto output_dir = diagnostic_output ? std::filesystem::path(diagnostic_output) : receipts;
  if (diagnostic_output) std::filesystem::create_directories(output_dir);
  const auto capture_dir = receipts / (decode_steps == 1 ? "target-repeats" : "target-d64");
  const int repeats = held_out_states ? 1 : (decode_steps == 1 ? 3 : 1);
  std::ifstream record(capture_dir / "repeat-0.json");
  const auto captured_record = nlohmann::json::parse(record);
  const auto captured_ids = captured_record.at("output_ids").at(0)
                                .get<std::vector<int32_t>>();
  REQUIRE(captured_ids.size() == static_cast<size_t>(decode_steps + 1));
  std::vector<int32_t> prompt_ids;
  if (held_out_states) {
    const auto& prompt = captured_record.at("actual_input_witnesses").at("p128").at("token_ids");
    REQUIRE(prompt.is_array()); REQUIRE(prompt.size() == 128);
    for (const auto& token : prompt) {
      REQUIRE(token.is_number_integer()); REQUIRE_FALSE(token.is_boolean());
      REQUIRE(token.get<int64_t>() >= 0); REQUIRE(token.get<int64_t>() < 248320);
      prompt_ids.push_back(token.get<int32_t>());
    }
    const auto& observations = captured_record.at("all_gdn_state_records");
    REQUIRE(observations.size() == 192);
    for (const std::string phase : {"p128", "d1"})
      for (int layer = 0; layer < 64; ++layer) if (layer % 4 != 3)
        for (const std::string stage : {"before", "after"}) {
          const auto& entry = observations.at(phase + "_l" + std::to_string(layer) + "_" + stage);
          CHECK(entry.at("phase") == phase); CHECK(entry.at("layer") == layer);
          CHECK(entry.at("stage") == stage);
          CHECK(entry.at("copied") == !(phase == "p128" && stage == "before"));
        }
    if (held_out_kv) {
      const auto& kv = captured_record.at("all_attention_kv_records");
      REQUIRE(kv.size() == 64);
      for (const std::string phase : {"p128", "d1"})
        for (int layer = 3; layer < 64; layer += 4)
          for (const std::string stage : {"before", "after"}) {
            const auto& entry = kv.at(phase + "_l" + std::to_string(layer) + "_" + stage);
            const int length = stage == "before" ? (phase == "p128" ? 0 : 128)
                                                 : (phase == "p128" ? 128 : 129);
            CHECK(entry.at("phase") == phase); CHECK(entry.at("layer") == layer);
            CHECK(entry.at("stage") == stage);
            CHECK(entry.at("written_logical_length") == length);
            CHECK(entry.at("copied") == (length > 0));
            CHECK(entry.at("logical_addresses").size() == size_t(length));
            for (const std::string scale : {"_k_scale", "_v_scale"}) {
              const auto& values = entry.at("scales").at(scale).at("values");
              if (values.is_array()) { REQUIRE(values.size() == 1); CHECK(values.at(0) == 1.0f); }
              else CHECK(values == 1.0f);
            }
          }
    }
  }
  std::vector<vllm::SafetensorsFile> oracles;
  for (int repeat = 0; repeat < repeats; ++repeat)
    oracles.push_back(vllm::SafetensorsFile::Open(
        (capture_dir / ("repeat-" + std::to_string(repeat) + ".safetensors")).string()));
  const auto config = vllm::LoadHfConfig((model_dir / "config.json").string());
  REQUIRE(config.num_hidden_layers == 64);
  REQUIRE(config.vocab_size == 248320);
  std::vector<vllm::SafetensorsFile> shards;
  for (const char* name : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
    shards.push_back(vllm::SafetensorsFile::Open((model_dir / name).string()));
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto weights = vllm::LoadQwen3_5Dense(shards, config, &gpu.q);
  REQUIRE(weights.exl3_checkpoint);
  REQUIRE(weights.precision.activation == DType::kF16);
  REQUIRE(weights.layers.size() == 64);
  REQUIRE(weights.lm_head_exl3.Bits() == 6);
  REQUIRE(weights.lm_head_exl3.OutFeatures() == 248320);
  REQUIRE(weights.lm_head.Empty());
  std::cout << "REAL_TARGET_LOAD layers=64 head_bits=6 vocab=248320\n" << std::flush;
  constexpr int slots = 5, active = 4, page = 1600;
  std::vector<std::unique_ptr<xpu_test::Buffer>> owners;
  std::vector<vllm::GdnStateCache> states;
  std::vector<vllm::PagedKvCache> caches;
  for (const auto& layer : weights.layers) {
    if (layer.is_linear_attention) {
      auto conv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF16,
          std::initializer_list<int64_t>{slots, 10240, 3});
      auto ssm = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF32,
          std::initializer_list<int64_t>{slots, 48, 128, 128});
      conv->put(std::vector<float>(slots * 10240 * 3, 0.125f));
      ssm->put(std::vector<float>(slots * 48 * 128 * 128, 0.125f));
      vllm::GdnStateCache state; state.conv_state = conv->tensor; state.ssm_state = ssm->tensor;
      states.push_back(state);
      owners.push_back(std::move(conv)); owners.push_back(std::move(ssm));
    } else {
      auto kv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kI8,
          std::initializer_list<int64_t>{2 * page * 4 * 256});
      std::vector<uint8_t> poison(2 * page * 4 * 256, 0x7f);
      kv->upload(poison.data());
      vllm::PagedKvCache cache;
      cache.data = kv->tensor.data; cache.dtype = DType::kI8;
      cache.num_blocks = 1; cache.block_size = page; cache.num_kv_heads = 4; cache.head_size = 256;
      cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
      caches.push_back(cache); owners.push_back(std::move(kv));
    }
  }
  REQUIRE(states.size() == 48); REQUIRE(caches.size() == 16);
  for (int step = 0; step <= (diagnostic ? diagnostic_stop : decode_steps); ++step) {
    const bool prefill = step == 0;
    const int rows = prefill ? 128 : 1;
    const std::string phase = prefill ? "p128" : "d" + std::to_string(step);
    std::vector<int32_t> ids(rows), positions(rows);
    for (int i = 0; i < rows; ++i) {
      ids[i] = prefill ? (held_out_states ? prompt_ids.at(i) : 1000 + (i * 37) % 4096)
                       : captured_ids[step - 1];
      positions[i] = prefill ? i : 127 + step;
    }
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1; am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = {0, rows};
    am.seq_lens = am.seq_lens_cpu = {128 + step};
    am.max_query_len = rows; am.max_seq_len = am.seq_lens[0];
    am.block_table_num_cols = 1; am.block_table_tensor = {0};
    am.slot_mapping.assign(positions.begin(), positions.end());
    am.causal = true;
    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = rows;
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{active};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, rows};
    if (prefill) {
      gm.num_prefills = 1; gm.num_prefill_tokens = rows;
      gm.has_initial_state = std::vector<uint8_t>{0};
      gm.prefill_query_start_loc = std::vector<int32_t>{0, rows};
      gm.prefill_state_indices = std::vector<int32_t>{active};
      gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto chunks = vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = chunks.batch_ptr; gm.token_chunk_offset_ptr = chunks.token_chunk_offset_ptr;
    } else { gm.num_decodes = gm.num_decode_tokens = 1; }
    std::cout << "REAL_TARGET_FORWARD_START " << phase << '\n' << std::flush;
    if (held_out_states && !prefill)
      CheckHeldOutGdnStates(phase, "before", gpu.q, weights, states, oracles[0]);
    if (held_out_kv && !prefill)
      CheckHeldOutAttentionKv(phase, "before", gpu.q, weights, caches, 128, oracles[0]);
    const bool capture_hidden = diagnostic &&
        (step == 0 || step == 1 || step == 11 || step == 27 || step == 29);
    const bool capture_gdn_state = diagnostic && !attention3_detail &&
        (step == 29 || (gdn_history && step <= 1)) &&
        std::getenv("VT_B70_EXL3_STATE_OUTPUT") != nullptr;
    const std::string state_phase = step == 29 ? "D29" : phase;
    if (attention3_detail && step == 29) {
      REQUIRE(owners[6]->tensor.data == caches[0].data);
      ExportAttention3Kv("own-D29-attention3-before", gpu.q, caches[0], 156);
    }
    if (capture_gdn_state) {
      REQUIRE(gdn_detail_layer >= 0);
      REQUIRE(gdn_detail_layer < int(weights.layers.size()));
      REQUIRE(weights.layers[gdn_detail_layer].is_linear_attention);
      size_t gdn_index = 0, owner_index = 0;
      for (int i = 0; i < gdn_detail_layer; ++i) {
        gdn_index += weights.layers[i].is_linear_attention;
        owner_index += weights.layers[i].is_linear_attention ? 2 : 1;
      }
      REQUIRE(owners[owner_index]->tensor.data == states[gdn_index].conv_state.data);
      REQUIRE(owners[owner_index + 1]->tensor.data == states[gdn_index].ssm_state.data);
      // Cold prefill ignores its seed; compare only the produced state.
      if (!prefill)
        ExportGdnState("own-" + state_phase + "-layer" +
                       std::to_string(gdn_detail_layer) + "-before",
                       *owners[owner_index], *owners[owner_index + 1]);
    }
    vllm::Qwen3_5MTPHiddenStates actual_hidden;
    const auto out = vllm::Qwen3_5DenseModel::ForwardDeviceTap(
        ids, positions, am, gm, caches, states, weights, config, gpu.q,
        capture_hidden ? &actual_hidden : nullptr, {rows - 1});
    REQUIRE(out.rows == 1); REQUIRE(out.device_tensor.dtype == DType::kF32);
    std::vector<float> logits(248320);
    auto& backend = vt::GetBackend(gpu.q.device.type);
    backend.Copy(gpu.q, logits.data(), out.device_tensor.data, logits.size() * sizeof(float));
    backend.Synchronize(gpu.q);
    if (held_out_states)
      CheckHeldOutGdnStates(phase, "after", gpu.q, weights, states, oracles[0]);
    if (held_out_kv)
      CheckHeldOutAttentionKv(phase, "after", gpu.q, weights, caches, prefill ? 128 : 129, oracles[0]);
    if (capture_gdn_state) {
      size_t owner_index = 0;
      for (int i = 0; i < gdn_detail_layer; ++i)
        owner_index += weights.layers[i].is_linear_attention ? 2 : 1;
      ExportGdnState("own-" + state_phase + "-layer" +
                     std::to_string(gdn_detail_layer) + "-after",
                     *owners[owner_index], *owners[owner_index + 1]);
    }
    if (attention3_detail && step == 29)
      ExportAttention3Kv("own-D29-attention3-after", gpu.q, caches[0], 157);
    REQUIRE(std::all_of(logits.begin(), logits.end(), [](float x) { return std::isfinite(x); }));
    const char* output_prefix = std::getenv("VT_B70_EXL3_TARGET_OUTPUT_PREFIX");
    const std::string prefix = output_prefix ? output_prefix :
        (decode_steps == 1 ? "target-native" : "target-native-d64");
    const auto file = output_dir / (prefix + "-" + phase + "-logits.f32");
    REQUIRE(!std::filesystem::exists(file));
    std::ofstream raw(file, std::ios::binary);
    raw.write(reinterpret_cast<const char*>(logits.data()), logits.size() * sizeof(float));
    raw.close(); REQUIRE(raw.good());
    if (capture_hidden) {
      REQUIRE(actual_hidden.storage);
      REQUIRE(actual_hidden.tensor.dtype == DType::kF16);
      REQUIRE(actual_hidden.tensor.IsContiguous());
      REQUIRE(actual_hidden.tensor.rank == 2);
      REQUIRE(actual_hidden.tensor.shape[0] == rows);
      REQUIRE(actual_hidden.tensor.shape[1] == 5120);
      std::vector<unsigned char> hidden(5120 * 2);
      const auto* selected = static_cast<const unsigned char*>(actual_hidden.tensor.data) +
                             size_t(rows - 1) * hidden.size();
      backend.Copy(gpu.q, hidden.data(), selected, hidden.size()); backend.Synchronize(gpu.q);
      const auto path = output_dir / (prefix + "-" + phase + "-hidden.f16");
      REQUIRE_FALSE(std::filesystem::exists(path));
      std::ofstream bytes(path, std::ios::binary);
      bytes.write(reinterpret_cast<const char*>(hidden.data()), hidden.size()); bytes.close();
      REQUIRE(bytes.good());
      const auto layout = output_dir / (prefix + "-" + phase + "-hidden.json");
      REQUIRE_FALSE(std::filesystem::exists(layout));
      std::ofstream metadata(layout);
      metadata << nlohmann::json({{"phase", phase}, {"dtype", "F16"}, {"shape", {1, 5120}},
          {"selected_row", rows - 1}, {"input_ids", ids}, {"positions", positions},
          {"active_state_slot", active}, {"native_owned_evolving_state", true},
          {"sampling_mode", "teacher_forced_captured_token_prefix"},
          {"scope", "diagnostic final-normalized hidden only; no oracle state injection"}}).dump(2);
      metadata.close(); REQUIRE(metadata.good());
    }
    const auto probabilities = [](const std::vector<float>& x) {
      const double maximum = *std::max_element(x.begin(), x.end());
      std::vector<double> p(x.size()); double sum = 0;
      for (size_t i = 0; i < x.size(); ++i) sum += p[i] = std::exp(double(x[i]) - maximum);
      for (auto& v : p) v /= sum;
      return p;
    };
    const auto top10 = [](const std::vector<float>& x) {
      std::vector<size_t> indices(x.size()); std::iota(indices.begin(), indices.end(), 0);
      std::partial_sort(indices.begin(), indices.begin() + 10, indices.end(),
                        [&](size_t a, size_t b) { return x[a] > x[b]; });
      indices.resize(10); return indices;
    };
    const auto actual_p = probabilities(logits);
    const auto actual_top = top10(logits);
    for (int repeat = 0; repeat < repeats; ++repeat) {
      const auto expected = CapturedFloats(oracles[repeat].Get(phase + "_logits"));
      REQUIRE(expected.size() == logits.size());
      const auto expected_p = probabilities(expected);
      const auto expected_top = top10(expected);
      double tv = 0, kl = 0;
      for (size_t i = 0; i < logits.size(); ++i) {
        tv += std::abs(actual_p[i] - expected_p[i]) * 0.5;
        if (expected_p[i] > 0) kl += expected_p[i] * std::log(expected_p[i] / actual_p[i]);
      }
      int overlap = 0;
      for (const auto id : actual_top)
        overlap += std::find(expected_top.begin(), expected_top.end(), id) != expected_top.end();
      std::cout << "REAL_TARGET_COMPARE " << phase << " repeat=" << repeat << " TV=" << tv
                << " KL=" << kl << " top10=" << overlap << " greedy=" << actual_top[0]
                << " reference_greedy=" << captured_ids[step] << '\n';
      CHECK(tv <= 0.02); CHECK(kl <= 0.002); CHECK(overlap >= 9);
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real target: eager P128 D1 full vocabulary comparison") {
  RunRealEagerTarget(1);
}

TEST_CASE("XPU EXL3 real target: eager P128 D64 full vocabulary continuation") {
  RunRealEagerTarget(64);
}

TEST_CASE("XPU EXL3 real target diagnostic: held-out P128 D1 all GDN states") {
  // Separate controlled held-out fixture: native state evolves from its own
  // poisoned caches. Only token prefixes are teacher-forced, never states.
  RunRealEagerTarget(1, 1, false, 1, false, true);
}

TEST_CASE("XPU EXL3 real target diagnostic: held-out P128 D1 all hybrid states") {
  // Independent native GDN and FP8 caches; only previously written logical KV
  // rows are compared. Cold poison/capacity is never an original fixture.
  RunRealEagerTarget(1, 1, false, 1, false, true, true);
}

TEST_CASE("XPU EXL3 real target diagnostic: integrated C1 Q4 original trace replay") {
  // Full target replay of inputs/previous accepted lengths observed in an
  // original MTP run. Native Conv/SSM/KV evolve independently from cold caches;
  // original states are comparison operands only, never inference seeds.
  // This does not test autonomous native draft/rejection equivalence or C4.
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* capture = std::getenv("VT_B70_EXL3_MTP_CAPTURE");
  const char* output = std::getenv("VT_B70_EXL3_DIAGNOSTIC_OUTPUT");
  if (!model || !capture || !output) std::exit(77);
  const char* attribution = std::getenv("VT_B70_EXL3_MTP_LAYER0_ATTRIBUTION");
  REQUIRE((attribution == nullptr || std::string(attribution) == "1"));
  const bool layer0_attribution = attribution != nullptr;
  const vllm::actdump::StepSelectionScope only_first_q4(layer0_attribution ? 1 : -1);
  const vllm::actdump::StageLayerSelectionScope only_layer0(layer0_attribution ? 0 : -1);
  const std::filesystem::path model_dir(model), capture_dir(capture), output_dir(output);
  REQUIRE_FALSE(std::filesystem::exists(output_dir));
  std::ifstream file(capture_dir / "capture.json"); REQUIRE(file.good());
  const auto reference = nlohmann::json::parse(file);
  REQUIRE(reference.at("schema") == "b70-integrated-mtp-C1-Q4-v1");
  const std::string label = reference.at("reference_label");
  REQUIRE((label == "default" || label == "controlled_deterministic_BA"));
  const auto& steps = reference.at("steps"); REQUIRE(steps.size() == 3);
  std::vector<vllm::SafetensorsFile> oracles;
  for (int step = 0; step < 3; ++step) {
    const auto& record = steps.at(step);
    const int rows = step == 0 ? 128 : 4;
    REQUIRE(record.at("step") == step);
    REQUIRE(record.at("embedding_token_ids_exact") == true);
    REQUIRE(record.at("request_ids").size() == 1);
    REQUIRE(record.at("query_start_loc") == std::vector<int32_t>{0, rows});
    REQUIRE(record.at("token_ids").size() == size_t(rows));
    REQUIRE(record.at("seq_lens").size() == 1);
    const int seq = record.at("seq_lens").at(0);
    REQUIRE(seq >= 128); REQUIRE(seq <= 160);
    for (int i = 0; i < rows; ++i) {
      const auto& token = record.at("token_ids").at(i);
      REQUIRE(token.is_number_integer()); REQUIRE_FALSE(token.is_boolean());
      REQUIRE(token.get<int64_t>() >= 0); REQUIRE(token.get<int64_t>() < 248320);
      for (int axis = 0; axis < 3; ++axis)
        REQUIRE(record.at("positions").at(axis).at(i) == seq - rows + i);
    }
    REQUIRE(record.at("gdn").size() == 48); REQUIRE(record.at("attention").size() == 16);
    const std::string name = "step-" + std::to_string(step) + ".safetensors";
    REQUIRE(record.at("file") == name);
    oracles.push_back(vllm::SafetensorsFile::Open((capture_dir / name).string()));
    REQUIRE(oracles.back().Names().size() == size_t(step == 0 ? 129 : 401));
  }
  const auto config = vllm::LoadHfConfig((model_dir / "config.json").string());
  REQUIRE(config.num_hidden_layers == 64); REQUIRE(config.vocab_size == 248320);
  std::vector<vllm::SafetensorsFile> shards;
  for (const char* name : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
    shards.push_back(vllm::SafetensorsFile::Open((model_dir / name).string()));
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto weights = vllm::LoadQwen3_5Dense(shards, config, &gpu.q);
  REQUIRE(weights.exl3_checkpoint); REQUIRE(weights.precision.activation == DType::kF16);
  auto& backend = vt::GetBackend(gpu.q.device);
  constexpr int slots = 4, active = 3, channels = 10240, history = 6, page = 1600;
  constexpr size_t ssm_bytes = 48 * 128 * 128 * sizeof(float);
  std::vector<std::unique_ptr<xpu_test::Buffer>> owners;
  std::vector<vllm::GdnStateCache> states;
  std::vector<vllm::PagedKvCache> caches;
  for (const auto& layer : weights.layers) {
    if (layer.is_linear_attention) {
      auto conv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF16,
          std::initializer_list<int64_t>{slots, channels, history});
      auto ssm = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF32,
          std::initializer_list<int64_t>{slots, 48, 128, 128});
      conv->put(std::vector<float>(slots * channels * history, 0.125f));
      ssm->put(std::vector<float>(slots * 48 * 128 * 128, 0.125f));
      vllm::GdnStateCache state; state.conv_state = conv->tensor; state.ssm_state = ssm->tensor;
      states.push_back(state);
      owners.push_back(std::move(conv)); owners.push_back(std::move(ssm));
    } else {
      auto kv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kI8,
          std::initializer_list<int64_t>{2 * page * 4 * 256});
      kv->upload(std::vector<uint8_t>(kv->bytes, 0x7f).data());
      vllm::PagedKvCache cache;
      cache.data = kv->tensor.data; cache.dtype = DType::kI8;
      cache.num_blocks = 1; cache.block_size = page; cache.num_kv_heads = 4; cache.head_size = 256;
      cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
      caches.push_back(cache); owners.push_back(std::move(kv));
    }
  }
  REQUIRE(states.size() == 48); REQUIRE(caches.size() == 16);
  std::filesystem::create_directories(output_dir);
  nlohmann::json result = {{"reference_label", label},
      {"ordinary_original_ids_exact", reference.at("ordinary_ids_exact")},
      {"native_states_injected", false}, {"native_snapshot_slots", {3, 2, 1, 0}},
      {"raw_export_scope", layer0_attribution ? "layer0_and_target_hidden" : "all"},
      {"comparisons", nlohmann::json::array()},
      {"scope", "Cold native full-target replay of original MTP inputs/accepted lengths; eager C1, not autonomous native draft/rejection or ordinary admission"}};
  auto save = [&] {
    std::ofstream stream(output_dir / "comparison.json");
    stream << result.dump(2) << '\n'; stream.close(); REQUIRE(stream.good());
  };
  auto compare = [&](int step, const std::string& key, const std::vector<uint8_t>& bytes,
                     const std::string& dtype, const std::vector<int64_t>& shape) {
    CAPTURE(step);
    CAPTURE(key);
    const auto& expected = oracles.at(step).Get(key);
    REQUIRE(expected.dtype == dtype); REQUIRE(expected.shape == shape);
    REQUIRE(expected.nbytes == bytes.size());
    const auto path = output_dir / ("step-" + std::to_string(step) + "-" + key + ".bin");
    const bool export_raw = !layer0_attribution || key.starts_with("l0_") || key == "target_hidden";
    if (export_raw) {
      REQUIRE_FALSE(std::filesystem::exists(path));
      std::ofstream raw(path, std::ios::binary);
      raw.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      raw.close(); REQUIRE(raw.good());
    }
    const size_t different = std::inner_product(bytes.begin(), bytes.end(), expected.data,
        size_t{0}, std::plus<size_t>(), [](auto a, auto b) { return size_t(a != b); });
    bool finite = true;
    double max_error = 0;
    if (dtype != "U8") {
      const size_t width = dtype == "F32" ? 4 : 2;
      for (size_t i = 0; i < bytes.size(); i += width) {
        float a, e;
        if (width == 4) {
          std::memcpy(&a, bytes.data() + i, 4); std::memcpy(&e, expected.data + i, 4);
        } else {
          uint16_t av, ev;
          std::memcpy(&av, bytes.data() + i, 2); std::memcpy(&ev, expected.data + i, 2);
          a = vt::F16ToF32(av); e = vt::F16ToF32(ev);
        }
        finite &= std::isfinite(a) && std::isfinite(e);
        max_error = std::max(max_error, std::abs(double(a) - e));
      }
    }
    result["comparisons"].push_back({{"step", step}, {"key", key}, {"dtype", dtype},
        {"shape", shape}, {"bytes", bytes.size()}, {"different_bytes", different},
        {"finite", finite}, {"max_abs_error", max_error},
        {"file", export_raw ? nlohmann::json(path.filename().string()) : nlohmann::json(nullptr)}});
    std::cout << "INTEGRATED_NATIVE_COMPARE step=" << step << " key=" << key
              << " different_bytes=" << different << " max_error=" << max_error << '\n';
    CHECK(finite); CHECK(different == 0);
  };
  for (int step = 0; step < 3; ++step) {
    const auto& record = steps.at(step);
    const auto ids = record.at("token_ids").get<std::vector<int32_t>>();
    const auto positions = record.at("positions").at(0).get<std::vector<int32_t>>();
    const int rows = ids.size(), seq = record.at("seq_lens").at(0);
    const int accepted = step == 0 ? 1 : record.at("gdn").at("l0").at("previous_accepted_tokens").at(0).get<int>();
    REQUIRE(accepted >= 1); REQUIRE(accepted <= 4);
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1; am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = {0, rows};
    am.seq_lens = am.seq_lens_cpu = {seq}; am.max_query_len = rows; am.max_seq_len = seq;
    am.block_table_num_cols = 1; am.block_table_tensor = {0};
    am.slot_mapping.assign(positions.begin(), positions.end()); am.causal = true;
    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = rows;
    if (step == 0) {
      gm.num_prefills = 1; gm.num_prefill_tokens = rows;
      gm.non_spec_state_indices_tensor = gm.prefill_state_indices = std::vector<int32_t>{active};
      gm.non_spec_query_start_loc = gm.prefill_query_start_loc = std::vector<int32_t>{0, rows};
      gm.has_initial_state = gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto chunks = vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = chunks.batch_ptr; gm.token_chunk_offset_ptr = chunks.token_chunk_offset_ptr;
    } else {
      gm.num_spec_decodes = 1; gm.num_spec_decode_tokens = rows;
      gm.spec_state_indices_num_cols = 4;
      gm.spec_state_indices_tensor = std::vector<int32_t>{3, 2, 1, 0};
      gm.spec_query_start_loc = std::vector<int32_t>{0, rows};
      gm.spec_sequence_masks = std::vector<uint8_t>{1};
      gm.spec_token_indx = std::vector<int32_t>{0, 1, 2, 3};
      gm.num_accepted_tokens = std::vector<int32_t>{accepted};
    }
    auto boundary = [&](const std::string& stage) {
      size_t gi = 0, ai = 0;
      for (size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const std::string key = "l" + std::to_string(layer);
        if (weights.layers[layer].is_linear_attention) {
          const auto& state = states.at(gi++);
          const auto& plan = record.at("gdn").at(key).at("plan");
          if (step == 0) REQUIRE(record.at("gdn").at(key).at("cold_unconsumed_seed_excluded") == true);
          else REQUIRE(record.at("gdn").at(key).at("previous_accepted_tokens") == std::vector<int32_t>{accepted});
          const int begin = stage == "before" ? accepted - 1 : 0;
          const int width = stage == "before" || step == 0 ? 3 : 6;
          REQUIRE(plan.at("conv_" + stage) == std::vector<int32_t>{begin, begin + width});
          auto conv = state.conv_state;
          conv.data = static_cast<uint16_t*>(conv.data) + active * channels * history + begin;
          conv.shape[0] = 1; conv.shape[2] = width;
          xpu_test::Buffer packed(gpu.q, DType::kF16, {1, channels, width});
          vt::Copy(gpu.q, packed.tensor, conv);
          const auto values = packed.download();
          std::vector<uint8_t> canonical(values.size());
          for (int c = 0; c < channels; ++c) for (int h = 0; h < width; ++h)
            std::memcpy(canonical.data() + (h * channels + c) * 2,
                        values.data() + (c * width + h) * 2, 2);
          compare(step, key + "_conv_" + stage, canonical, "F16", {width, channels});
          const int snapshots = stage == "before" || step == 0 ? 1 : 4;
          for (int token = 0; token < snapshots; ++token) {
            const int slot = stage == "before" ? 4 - accepted : 3 - token;
            std::vector<uint8_t> ssm(ssm_bytes);
            backend.Copy(gpu.q, ssm.data(), static_cast<const uint8_t*>(state.ssm_state.data) + slot * ssm_bytes, ssm_bytes);
            backend.Synchronize(gpu.q);
            compare(step, key + "_ssm_" + stage + (stage == "after" ? "_t" + std::to_string(token) : ""),
                    ssm, "F32", {48, 128, 128});
          }
        } else {
          const auto& cache = caches.at(ai++);
          const int length = stage == "before" ? seq - rows : seq;
          REQUIRE(record.at("attention").at(key).at(stage).at("written_logical_length") == length);
          for (int which = 0; which < 2; ++which) {
            auto view = vllm::dense_attn::KvSlice(cache, gpu.q.device, which);
            view.shape[1] = length;
            xpu_test::Buffer packed(gpu.q, DType::kI8, {1, length, 4, 256});
            vt::Copy(gpu.q, packed.tensor, view);
            compare(step, key + (which ? "_value_" : "_key_") + stage,
                    packed.download(), "U8", {length, 4, 256});
          }
        }
      }
    };
    if (step > 0) boundary("before");
    vllm::Qwen3_5MTPHiddenStates hidden;
    std::cout << "INTEGRATED_NATIVE_FORWARD " << step << '\n' << std::flush;
    auto logits = vllm::Qwen3_5DenseModel::ForwardDeviceTap(ids, positions, am, gm, caches,
        states, weights, config, gpu.q, &hidden, step == 0 ? std::vector<int32_t>{127} : std::vector<int32_t>{});
    REQUIRE(logits.rows == (step == 0 ? 1 : 4)); REQUIRE(hidden.tensor.dtype == DType::kF16);
    boundary("after");
    xpu_test::Buffer hidden_copy(gpu.q, DType::kF16, {rows, 5120});
    vt::Copy(gpu.q, hidden_copy.tensor, hidden.tensor);
    compare(step, "target_hidden", hidden_copy.download(), "F16", {rows, 5120});
    save();
  }
  CHECK(result.at("comparisons").size() == 931);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real target diagnostic: integrated C4 C2 C1 original trace digest replay") {
  // Full target only: actual original tokens/positions/accepted lengths replayed
  // from cold native caches. No original state injection or autonomous native
  // draft/rejection/cancellation-scheduler equivalence. SHA covers every byte,
  // not selected rows or a relaxed numeric tolerance; raw arrays are transient.
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* capture = std::getenv("VT_B70_EXL3_MTP_CAPTURE");
  const char* output = std::getenv("VT_B70_EXL3_DIAGNOSTIC_OUTPUT");
  if (!model || !capture || !output) std::exit(77);
  const std::filesystem::path model_dir(model), capture_dir(capture), output_dir(output);
  REQUIRE_FALSE(std::filesystem::exists(output_dir));
  std::ifstream source(capture_dir / "capture.json"); REQUIRE(source.good());
  const auto reference = nlohmann::json::parse(source);
  REQUIRE(reference.at("schema") == "b70-integrated-mtp-C4-C2-C1-digests-v1");
  REQUIRE(reference.at("reference_label") == "controlled_deterministic_BA");
  REQUIRE(reference.at("ordinary_observed_outputs_exact") == true);
  REQUIRE(reference.at("actual_target_counts") == std::vector<int>{4, 4, 2, 1});
  const auto& steps = reference.at("steps"); REQUIRE(steps.size() == 4);
  const auto initial_ids = steps.at(0).at("request_ids").get<std::vector<std::string>>();
  REQUIRE(initial_ids.size() == 4);
  std::map<std::string, int> request_owners;
  for (int i = 0; i < 4; ++i) REQUIRE(request_owners.emplace(initial_ids[i], i).second);
  constexpr int counts[] = {4, 4, 2, 1}, expected_arrays[] = {513, 1601, 801, 401};
  for (int step = 0; step < 4; ++step) {
    const auto& record = steps.at(step);
    const int count = counts[step], width = step == 0 ? 128 : 4;
    REQUIRE(record.at("step") == step); REQUIRE(record.at("embedding_token_ids_exact") == true);
    REQUIRE(record.at("request_ids").size() == size_t(count));
    REQUIRE(record.at("seq_lens").size() == size_t(count)); REQUIRE(record.at("rows").size() == size_t(count));
    REQUIRE(record.at("token_ids").size() == size_t(count * width));
    std::vector<int32_t> qsl(count + 1); std::iota(qsl.begin(), qsl.end(), 0);
    for (auto& v : qsl) v *= width;
    REQUIRE(record.at("query_start_loc") == qsl);
    REQUIRE(record.at("gdn").size() == 48); REQUIRE(record.at("attention").size() == 16);
    REQUIRE(record.at("tensors").size() == size_t(expected_arrays[step]));
    REQUIRE(record.at("positions").size() == 3);
    std::set<std::string> actual;
    for (int row = 0; row < count; ++row) {
      const std::string id = record.at("request_ids").at(row);
      REQUIRE(request_owners.contains(id)); REQUIRE(actual.insert(id).second);
      const int owner = request_owners.at(id);
      REQUIRE((step < 2 || (step == 2 && owner >= 2) || (step == 3 && owner == 3)));
      const auto& r = record.at("rows").at(row);
      REQUIRE(r.at("request_id") == id); REQUIRE(r.at("row") == row);
      REQUIRE(r.at("begin") == row * width); REQUIRE(r.at("end") == (row + 1) * width);
      REQUIRE(r.at("cold_prefill") == (step == 0));
      const int seq = record.at("seq_lens").at(row);
      REQUIRE(seq >= 128 + (step == 0 ? 0 : width)); REQUIRE(seq <= 160);
      if (step == 0) REQUIRE(seq == 128);
      for (int token = row * width; token < (row + 1) * width; ++token) {
        const auto& id_value = record.at("token_ids").at(token);
        REQUIRE(id_value.is_number_integer()); REQUIRE_FALSE(id_value.is_boolean());
        REQUIRE(id_value.get<int64_t>() >= 0); REQUIRE(id_value.get<int64_t>() < 248320);
        for (int axis = 0; axis < 3; ++axis)
          REQUIRE(record.at("positions").at(axis).at(token) == seq - width + token - row * width);
      }
    }
  }
  const auto config = vllm::LoadHfConfig((model_dir / "config.json").string());
  REQUIRE(config.num_hidden_layers == 64); REQUIRE(config.vocab_size == 248320);
  std::vector<vllm::SafetensorsFile> shards;
  for (const char* name : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
    shards.push_back(vllm::SafetensorsFile::Open((model_dir / name).string()));
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto weights = vllm::LoadQwen3_5Dense(shards, config, &gpu.q);
  REQUIRE(weights.exl3_checkpoint); REQUIRE(weights.precision.activation == DType::kF16);
  auto& backend = vt::GetBackend(gpu.q.device);
  constexpr int slots = 16, channels = 10240, history = 6, page = 1600;
  constexpr size_t ssm_bytes = 48 * 128 * 128 * sizeof(float);
  std::vector<std::unique_ptr<xpu_test::Buffer>> owners;
  std::vector<vllm::GdnStateCache> states;
  std::vector<vllm::PagedKvCache> caches;
  for (const auto& layer : weights.layers) {
    if (layer.is_linear_attention) {
      auto conv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF16,
          std::initializer_list<int64_t>{slots, channels, history});
      auto ssm = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF32,
          std::initializer_list<int64_t>{slots, 48, 128, 128});
      conv->put(std::vector<float>(slots * channels * history, 0.125f));
      ssm->put(std::vector<float>(slots * 48 * 128 * 128, 0.125f));
      vllm::GdnStateCache state; state.conv_state = conv->tensor; state.ssm_state = ssm->tensor;
      states.push_back(state); owners.push_back(std::move(conv)); owners.push_back(std::move(ssm));
    } else {
      auto kv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kI8,
          std::initializer_list<int64_t>{2 * 4 * page * 4 * 256});
      kv->upload(std::vector<uint8_t>(kv->bytes, 0x7f).data());
      vllm::PagedKvCache cache;
      cache.data = kv->tensor.data; cache.dtype = DType::kI8;
      cache.num_blocks = 4; cache.block_size = page; cache.num_kv_heads = 4; cache.head_size = 256;
      cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
      caches.push_back(cache); owners.push_back(std::move(kv));
    }
  }
  REQUIRE(states.size() == 48); REQUIRE(caches.size() == 16);
  std::filesystem::create_directories(output_dir);
  nlohmann::json result = {{"reference_label", reference.at("reference_label")},
      {"native_states_injected", false}, {"raw_arrays_persisted", false},
      {"native_ownership", request_owners}, {"comparisons", nlohmann::json::array()},
      {"scope", "Full-array digests for cold native target replay of actual original C4/C2/C1 MTP inputs/accepted lengths; not autonomous draft/rejection or native cancellation scheduler equivalence"}};
  std::set<std::pair<int, std::string>> compared;
  auto compare = [&](int step, const std::string& key, const std::vector<uint8_t>& raw,
                     const std::string& dtype, const std::vector<int64_t>& shape) {
    CAPTURE(step);
    CAPTURE(key);
    REQUIRE(compared.emplace(step, key).second);
    const auto& expected = steps.at(step).at("tensors").at(key);
    REQUIRE(expected.at("dtype") == dtype); REQUIRE(expected.at("shape") == shape);
    REQUIRE(expected.at("bytes") == raw.size()); REQUIRE(expected.at("finite") == true);
    const std::string expected_sha = expected.at("sha256"); REQUIRE(expected_sha.size() == 64);
    REQUIRE(std::all_of(expected_sha.begin(), expected_sha.end(), [](char c) {
      return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    }));
    const auto digest = vllm::v1::sha256_bytes(std::string(reinterpret_cast<const char*>(raw.data()), raw.size()));
    constexpr char hex[] = "0123456789abcdef";
    REQUIRE(digest.size() == 32);
    std::string sha; sha.reserve(64);
    for (unsigned char c : digest) { sha += hex[c >> 4]; sha += hex[c & 15]; }
    bool finite = true;
    if (dtype != "U8") {
      const size_t width = dtype == "F32" ? 4 : 2;
      for (size_t i = 0; i < raw.size(); i += width) {
        float value;
        if (width == 4) std::memcpy(&value, raw.data() + i, 4);
        else { uint16_t v; std::memcpy(&v, raw.data() + i, 2); value = vt::F16ToF32(v); }
        finite &= std::isfinite(value);
      }
    }
    const bool exact = sha == expected_sha;
    result["comparisons"].push_back({{"step", step}, {"key", key}, {"dtype", dtype}, {"shape", shape},
        {"bytes", raw.size()}, {"finite", finite}, {"native_sha256", sha},
        {"original_sha256", expected_sha}, {"bit_exact", exact}});
    std::cout << "MTP_TRANSITION_NATIVE_COMPARE step=" << step << " key=" << key << " exact=" << exact << '\n';
    CHECK(finite); CHECK(exact);
  };
  for (int step = 0; step < 4; ++step) {
    const auto& record = steps.at(step);
    const int count = counts[step], width = step == 0 ? 128 : 4, rows = count * width;
    const auto ids = record.at("token_ids").get<std::vector<int32_t>>();
    const auto positions = record.at("positions").at(0).get<std::vector<int32_t>>();
    const auto qsl = record.at("query_start_loc").get<std::vector<int32_t>>();
    const auto seq = record.at("seq_lens").get<std::vector<int32_t>>();
    std::vector<int32_t> bases, blocks, accepted, spec_slots;
    for (int row = 0; row < count; ++row) {
      const int owner = request_owners.at(record.at("request_ids").at(row).get<std::string>());
      blocks.push_back(owner); bases.push_back(owner * 4 + 3);
      for (int token = 0; token < 4; ++token) spec_slots.push_back(owner * 4 + 3 - token);
      if (step) {
        const auto& n = record.at("gdn").at("l0").at("previous_accepted_tokens").at(row);
        REQUIRE(n.is_number_integer()); REQUIRE_FALSE(n.is_boolean());
        REQUIRE(n.get<int>() >= 1); REQUIRE(n.get<int>() <= 4); accepted.push_back(n.get<int>());
      }
    }
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = count; am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = qsl;
    am.seq_lens = am.seq_lens_cpu = seq; am.max_query_len = width;
    am.max_seq_len = *std::max_element(seq.begin(), seq.end());
    am.block_table_num_cols = 1; am.block_table_tensor = blocks; am.causal = true;
    for (int row = 0; row < count; ++row) for (int t = qsl[row]; t < qsl[row + 1]; ++t)
      am.slot_mapping.push_back(blocks[row] * page + positions[t]);
    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = rows;
    if (step == 0) {
      gm.num_prefills = count; gm.num_prefill_tokens = rows;
      gm.non_spec_state_indices_tensor = gm.prefill_state_indices = bases;
      gm.non_spec_query_start_loc = gm.prefill_query_start_loc = qsl;
      gm.has_initial_state = gm.prefill_has_initial_state = std::vector<uint8_t>(count, 0);
      const auto chunks = vllm::v1::ComputeCausalConv1dMetadata(qsl);
      gm.batch_ptr = chunks.batch_ptr; gm.token_chunk_offset_ptr = chunks.token_chunk_offset_ptr;
    } else {
      gm.num_spec_decodes = count; gm.num_spec_decode_tokens = rows;
      gm.spec_state_indices_num_cols = 4; gm.spec_state_indices_tensor = spec_slots;
      gm.spec_query_start_loc = qsl; gm.spec_sequence_masks = std::vector<uint8_t>(count, 1);
      std::vector<int32_t> token_indices(rows); std::iota(token_indices.begin(), token_indices.end(), 0);
      gm.spec_token_indx = token_indices; gm.num_accepted_tokens = accepted;
    }
    auto boundary = [&](const std::string& stage) {
      size_t gi = 0, ai = 0;
      for (size_t layer = 0; layer < weights.layers.size(); ++layer) {
        const std::string key = "l" + std::to_string(layer);
        const bool linear = weights.layers[layer].is_linear_attention;
        const auto index = linear ? gi++ : ai++;
        for (int row = 0; row < count; ++row) {
          const std::string prefix = "r" + std::to_string(row) + "_" + key;
          if (linear) {
            const auto& state = states.at(index);
            const auto& entry = record.at("gdn").at(key);
            REQUIRE(entry.at("cold_unconsumed_seed_excluded") == (step == 0));
            if (step) REQUIRE(entry.at("previous_accepted_tokens") == accepted);
            const auto& plan = entry.at("plans").at(row); REQUIRE(plan.at("row") == row);
            const int begin = stage == "before" ? accepted[row] - 1 : 0;
            const int length = stage == "before" || step == 0 ? 3 : 6;
            REQUIRE(plan.at("conv_" + stage) == std::vector<int32_t>{begin, begin + length});
            auto conv = state.conv_state;
            conv.data = static_cast<uint16_t*>(conv.data) + bases[row] * channels * history + begin;
            conv.shape[0] = 1; conv.shape[2] = length;
            xpu_test::Buffer packed(gpu.q, DType::kF16, {1, channels, length});
            vt::Copy(gpu.q, packed.tensor, conv);
            const auto values = packed.download();
            std::vector<uint8_t> canonical(values.size());
            for (int c = 0; c < channels; ++c) for (int h = 0; h < length; ++h)
              std::memcpy(canonical.data() + (h * channels + c) * 2,
                          values.data() + (c * length + h) * 2, 2);
            compare(step, prefix + "_conv_" + stage, canonical, "F16", {length, channels});
            const int snapshots = stage == "before" || step == 0 ? 1 : 4;
            if (stage == "before")
              REQUIRE(plan.at("ssm_before") == entry.at("spec_state_indices").at(row).at(accepted[row] - 1));
            else REQUIRE(plan.at("ssm_after").size() == size_t(snapshots));
            for (int token = 0; token < snapshots; ++token) {
              const int slot = bases[row] - (stage == "before" ? accepted[row] - 1 : token);
              std::vector<uint8_t> ssm(ssm_bytes);
              backend.Copy(gpu.q, ssm.data(), static_cast<const uint8_t*>(state.ssm_state.data) + slot * ssm_bytes, ssm_bytes);
              backend.Synchronize(gpu.q);
              compare(step, prefix + "_ssm_" + stage + (stage == "after" ? "_t" + std::to_string(token) : ""),
                      ssm, "F32", {48, 128, 128});
            }
          } else {
            const auto& cache = caches.at(index);
            const int length = stage == "before" ? seq[row] - width : seq[row];
            const auto& scope = record.at("attention").at(key).at(stage).at(row);
            REQUIRE(scope.at("row") == row); REQUIRE(scope.at("written_logical_length") == length);
            const auto& addresses = scope.at("addresses"); REQUIRE(addresses.size() == size_t(length));
            for (int t = 0; t < length; ++t) {
              REQUIRE(addresses.at(t).at(0) == addresses.at(0).at(0)); REQUIRE(addresses.at(t).at(1) == t);
            }
            for (int which = 0; which < 2; ++which) {
              auto view = vllm::dense_attn::KvSlice(cache, gpu.q.device, which);
              view.data = static_cast<uint8_t*>(view.data) + blocks[row] * view.stride[0];
              view.shape[0] = 1; view.shape[1] = length;
              xpu_test::Buffer packed(gpu.q, DType::kI8, {1, length, 4, 256});
              vt::Copy(gpu.q, packed.tensor, view);
              compare(step, prefix + (which ? "_value_" : "_key_") + stage,
                      packed.download(), "U8", {length, 4, 256});
            }
          }
        }
      }
    };
    if (step) boundary("before");
    vllm::Qwen3_5MTPHiddenStates hidden;
    std::vector<int32_t> gather;
    if (step == 0) for (int row = 0; row < count; ++row) gather.push_back(qsl[row + 1] - 1);
    std::cout << "MTP_TRANSITION_NATIVE_FORWARD " << step << '\n' << std::flush;
    auto logits = vllm::Qwen3_5DenseModel::ForwardDeviceTap(ids, positions, am, gm, caches,
        states, weights, config, gpu.q, &hidden, gather);
    REQUIRE(logits.rows == (step == 0 ? count : rows)); REQUIRE(hidden.tensor.dtype == DType::kF16);
    boundary("after");
    xpu_test::Buffer hidden_copy(gpu.q, DType::kF16, {rows, 5120});
    vt::Copy(gpu.q, hidden_copy.tensor, hidden.tensor);
    compare(step, "target_hidden", hidden_copy.download(), "F16", {rows, 5120});
    std::ofstream out(output_dir / "comparison.json"); out << result.dump(2) << '\n'; out.close(); REQUIRE(out.good());
  }
  CHECK(result.at("comparisons").size() == 3316);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 real target diagnostic: own-state through D29 selected final hidden") {
  // A bounded collector retains the failing distribution checks. It does not
  // replace or shorten the existing full D64 qualification test above.
  RunRealEagerTarget(64, 29);
}

TEST_CASE("XPU EXL3 real target diagnostic: P128 D1 layer0 substage boundaries") {
  const vllm::actdump::StageLayerSelectionScope only_first_layer(0);
  RunRealEagerTarget(1, 1);
}

TEST_CASE("XPU EXL3 real target diagnostic: D29 attention3 boundaries") {
  REQUIRE(std::getenv("VT_DUMP_ACT_SUB") != nullptr);
  REQUIRE(std::getenv("VT_B70_EXL3_STATE_OUTPUT") != nullptr);
  const vllm::actdump::StageLayerSelectionScope only_attention3(3);
  RunRealEagerTarget(64, 29, true);
}

TEST_CASE("XPU EXL3 real target diagnostic: P128 D1 GDN21 history") {
  REQUIRE(std::getenv("VT_DUMP_ACT_SUB") != nullptr);
  REQUIRE(std::getenv("VT_B70_EXL3_STATE_OUTPUT") != nullptr);
  const vllm::actdump::StageLayerSelectionScope only_gdn21(21);
  // The D64 prefix is retained; this early collector is not its qualification.
  RunRealEagerTarget(64, 1, false, 21, true);
}

TEST_CASE("XPU EXL3 real target diagnostic: P128 D1 earlier block boundaries") {
  REQUIRE(std::getenv("VT_DUMP_ACT_SUB") != nullptr);
  // All native outer stages are observed for two forwards only. The original
  // comparison stops at GDN21; no full D64 qualification is shortened.
  RunRealEagerTarget(64, 1);
}

TEST_CASE("XPU EXL3 real target diagnostic: D29 GDN21 boundaries") {
  REQUIRE(std::getenv("VT_DUMP_ACT_SUB") != nullptr);
  REQUIRE(std::getenv("VT_B70_EXL3_STATE_OUTPUT") != nullptr);
  const vllm::actdump::StageLayerSelectionScope only_gdn21(21);
  RunRealEagerTarget(64, 29, false, 21);
}

TEST_CASE("XPU EXL3 real target head: identical original P128 D1 D29 operands") {
  // Isolate the full head from the failing own-state D29 trajectory. Captured
  // hidden operands are used only in this operator test, never in inference.
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* fixtures = std::getenv("VT_B70_EXL3_S1_FIXTURES");
  const char* output = std::getenv("VT_B70_EXL3_HEAD_OUTPUT");
  if (!model || !fixtures || !output) std::exit(77);
  const auto checkpoint = vllm::SafetensorsFile::Open(
      (std::filesystem::path(model) / "model-00002-of-00002.safetensors").string());
  const auto oracle = vllm::SafetensorsFile::Open(
      (std::filesystem::path(fixtures) / "target-d64/repeat-0.safetensors").string());
  const auto& tr = checkpoint.Get("lm_head.trellis");
  const auto& suh = checkpoint.Get("lm_head.suh");
  const auto& svh = checkpoint.Get("lm_head.svh");
  REQUIRE(tr.dtype == "I16"); REQUIRE(suh.dtype == "F16"); REQUIRE(svh.dtype == "F16");
  REQUIRE((tr.shape == std::vector<int64_t>{320, 15520, 96}));
  REQUIRE((suh.shape == std::vector<int64_t>{5120}));
  REQUIRE((svh.shape == std::vector<int64_t>{248320}));
  // Marker presence is the checkpoint's mul1 codebook contract.
  REQUIRE(checkpoint.Get("lm_head.mul1").dtype == "I32");
  vllm::Exl3Weight head;
  head.name = "lm_head"; head.codebook = 2;
  head.trellis = Own(DType::kI8, {320, 15520, 192}, tr.data, tr.nbytes);
  head.suh = Own(DType::kF16, {5120}, suh.data, suh.nbytes);
  head.svh = Own(DType::kF16, {248320}, svh.data, svh.nbytes);
  REQUIRE(head.Bits() == 6); REQUIRE(head.OutFeatures() == 248320);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto& backend = vt::GetBackend(gpu.q.device.type);
  vllm::dense_attn::Dev d{backend, gpu.q, DType::kF16};
  std::filesystem::create_directories(output);
  for (const std::string phase : {"p128", "d1", "d29"}) {
    CAPTURE(phase);
    const auto& hidden = oracle.Get(phase + "_logits_hidden");
    const auto& expected = oracle.Get(phase + "_logits");
    REQUIRE(hidden.dtype == "F16"); REQUIRE(expected.dtype == "F16");
    REQUIRE((hidden.shape == std::vector<int64_t>{1, 5120}));
    REQUIRE((expected.shape == std::vector<int64_t>{1, 248320}));
    xpu_test::Buffer input(gpu.q, DType::kF16, {1, 5120}); input.upload(hidden.data);
    auto projected = vllm::dense_attn::Exl3MatmulD(d, input.tensor, head, DType::kF16);
    REQUIRE(projected.t().dtype == DType::kF16);
    std::vector<unsigned char> raw(expected.nbytes);
    backend.Copy(gpu.q, raw.data(), projected.t().data, raw.size()); backend.Synchronize(gpu.q);
    size_t differences = 0;
    for (size_t i = 0; i < raw.size(); i += 2)
      differences += vt::LoadUnaligned<uint16_t>(raw.data() + i) !=
                     vt::LoadUnaligned<uint16_t>(expected.data + i);
    std::cout << "SAME_INPUT_TARGET_HEAD phase=" << phase
              << " vocab=248320 bits=6 half_differences=" << differences << '\n';
    const auto file = std::filesystem::path(output) / (phase + "-same-input-head.f16");
    REQUIRE_FALSE(std::filesystem::exists(file));
    std::ofstream bytes(file, std::ios::binary);
    bytes.write(reinterpret_cast<const char*>(raw.data()), raw.size()); bytes.close();
    REQUIRE(bytes.good());
    CHECK(differences == 0);
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU EXL3 autonomous target: real prompt native feedback cold reset") {
  const char* model = std::getenv("VT_B70_EXL3_MODEL");
  const char* input = std::getenv("VT_B70_EXL3_SMOKE_PROMPT");
  const char* output = std::getenv("VT_B70_EXL3_SMOKE_OUTPUT");
  if (!model || !input || !output) std::exit(77);
  const std::filesystem::path model_dir(model), output_dir(output);
  REQUIRE(!std::filesystem::exists(output_dir));
  std::ifstream prompt_file(input);
  const auto prompt_record = nlohmann::json::parse(prompt_file);
  const auto prompt = prompt_record.at("prompt_token_ids").get<std::vector<int32_t>>();
  const int limit = prompt_record.at("output_limit").get<int>();
  // R05: one real eager prefill, without hidden-state padding or rechunking.
  REQUIRE(!prompt.empty()); REQUIRE(prompt.size() <= 4096);
  REQUIRE((limit == 16 || limit == 64 || limit == 256));
  const auto tokenizer = vllm::tok::Tokenizer::FromHfJson((model_dir / "tokenizer.json").string());
  REQUIRE(tokenizer.Encode(prompt_record.at("rendered_prompt").get<std::string>()) == prompt);
  const auto config = vllm::LoadHfConfig((model_dir / "config.json").string());
  REQUIRE(config.num_hidden_layers == 64); REQUIRE(config.vocab_size == 248320);
  REQUIRE(!config.generation_config_eos_ids.empty());
  std::vector<vllm::SafetensorsFile> shards;
  for (const char* name : {"model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"})
    shards.push_back(vllm::SafetensorsFile::Open((model_dir / name).string()));
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto weights = vllm::LoadQwen3_5Dense(shards, config, &gpu.q);
  REQUIRE(weights.exl3_checkpoint); REQUIRE(weights.precision.activation == DType::kF16);
  REQUIRE(weights.lm_head.Empty()); REQUIRE(weights.lm_head_exl3.Bits() == 6);
  REQUIRE(weights.lm_head_exl3.OutFeatures() == config.vocab_size);
  constexpr int page = 1600;  // Actual physical FP8 page contract, not a context cap.
  const int budget = static_cast<int>(prompt.size()) + limit - 1;
  const int blocks = (budget + page - 1) / page;
  std::vector<std::unique_ptr<xpu_test::Buffer>> conv_owners, ssm_owners, kv_owners;
  std::vector<vllm::GdnStateCache> states;
  std::vector<vllm::PagedKvCache> caches;
  for (const auto& layer : weights.layers) {
    if (layer.is_linear_attention) {
      auto conv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF16,
          std::initializer_list<int64_t>{1, 10240, 3});
      auto ssm = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kF32,
          std::initializer_list<int64_t>{1, 48, 128, 128});
      vllm::GdnStateCache state; state.conv_state = conv->tensor; state.ssm_state = ssm->tensor;
      states.push_back(state); conv_owners.push_back(std::move(conv)); ssm_owners.push_back(std::move(ssm));
    } else {
      auto kv = std::make_unique<xpu_test::Buffer>(gpu.q, DType::kI8,
          std::initializer_list<int64_t>{2LL * blocks * page * 4 * 256});
      vllm::PagedKvCache cache;
      cache.data = kv->tensor.data; cache.dtype = DType::kI8;
      cache.num_blocks = blocks; cache.block_size = page; cache.num_kv_heads = 4; cache.head_size = 256;
      cache.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
      caches.push_back(cache); kv_owners.push_back(std::move(kv));
    }
  }
  REQUIRE(states.size() == 48); REQUIRE(caches.size() == 16);
  REQUIRE(std::filesystem::create_directory(output_dir));
  auto& backend = vt::GetBackend(gpu.q.device.type);
  nlohmann::json runs = nlohmann::json::array();
  const auto dump_layer0 = [&](const std::string& phase) {
    // Selected exports only; no full-model activation/state download campaign.
    for (const auto& entry : {std::make_tuple("conv.f16", states[0].conv_state.data, size_t(10240 * 3 * 2)),
                              std::make_tuple("ssm.f32", states[0].ssm_state.data, size_t(48 * 128 * 128 * 4))}) {
      std::vector<uint8_t> bytes(std::get<2>(entry));
      backend.Copy(gpu.q, bytes.data(), std::get<1>(entry), bytes.size());
      backend.Synchronize(gpu.q);
      std::ofstream file(output_dir / (phase + "-layer0-" + std::get<0>(entry)), std::ios::binary);
      file.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
      file.close(); REQUIRE(file.good());
    }
  };
  for (int request = 0; request < 2; ++request) {
    nlohmann::json witnesses = nlohmann::json::array();
    double prefill_seconds = 0, decode_seconds = 0;
    int decode_calls = 0;
    const auto generated = exl3_test::RunAutonomousC1(prompt, limit, config.generation_config_eos_ids,
        static_cast<int>(config.vocab_size), [&] {
          for (auto& conv : conv_owners) conv->put(std::vector<float>(10240 * 3, 0));
          for (auto& ssm : ssm_owners) ssm->put(std::vector<float>(48 * 128 * 128, 0));
          const std::vector<uint8_t> poison(2LL * blocks * page * 4 * 256, 0x7f);
          for (auto& kv : kv_owners) kv->upload(poison.data());
          backend.Synchronize(gpu.q);
          dump_layer0("request" + std::to_string(request) + "-initial");
        }, [&](const auto& ids, const auto& positions, bool prefill) {
          const int rows = static_cast<int>(ids.size());
          const int length = positions.back() + 1;
          vllm::v1::CommonAttentionMetadata am;
          am.num_reqs = 1; am.num_actual_tokens = rows;
          am.query_start_loc = am.query_start_loc_cpu = {0, rows};
          am.seq_lens = am.seq_lens_cpu = {length}; am.max_query_len = rows; am.max_seq_len = length;
          am.block_table_num_cols = blocks; am.block_table_tensor.resize(blocks);
          std::iota(am.block_table_tensor.begin(), am.block_table_tensor.end(), 0);
          am.slot_mapping.assign(positions.begin(), positions.end()); am.causal = true;
          vllm::v1::GDNAttentionMetadata gm;
          gm.num_actual_tokens = rows;
          gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
          gm.non_spec_query_start_loc = std::vector<int32_t>{0, rows};
          if (prefill) {
            gm.num_prefills = 1; gm.num_prefill_tokens = rows;
            gm.has_initial_state = std::vector<uint8_t>{0};
            gm.prefill_query_start_loc = std::vector<int32_t>{0, rows};
            gm.prefill_state_indices = std::vector<int32_t>{0};
            gm.prefill_has_initial_state = std::vector<uint8_t>{0};
            const auto chunks = vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
            gm.batch_ptr = chunks.batch_ptr; gm.token_chunk_offset_ptr = chunks.token_chunk_offset_ptr;
          } else { gm.num_decodes = gm.num_decode_tokens = 1; }
          const auto start = std::chrono::steady_clock::now();
          const auto out = vllm::Qwen3_5DenseModel::ForwardDeviceTap(
              ids, positions, am, gm, caches, states, weights, config, gpu.q, nullptr, {rows - 1});
          REQUIRE(out.rows == 1); REQUIRE(out.device_tensor.dtype == DType::kF32);
          std::vector<float> logits(config.vocab_size);
          backend.Copy(gpu.q, logits.data(), out.device_tensor.data, logits.size() * sizeof(float));
          backend.Synchronize(gpu.q);
          const auto chosen = exl3_test::GreedyToken(logits);
          const double seconds = std::chrono::duration<double>(
              std::chrono::steady_clock::now() - start).count();
          if (prefill) prefill_seconds += seconds;
          else { decode_seconds += seconds; ++decode_calls; }
          witnesses.push_back({{"input_ids", ids}, {"positions", positions},
                               {"prefill", prefill}, {"selected_global_id", chosen},
                               {"model_and_selection_seconds", seconds}});
          std::cout << "AUTONOMOUS_TARGET request=" << request << " emitted=" << witnesses.size()
                    << " selected_global_id=" << chosen << " seconds=" << seconds
                    << '\n' << std::flush;
          if (prefill) dump_layer0("request" + std::to_string(request) + "-prefill");
          return chosen;
        });
    dump_layer0("request" + std::to_string(request) + "-final");
    REQUIRE(!generated.output_ids.empty());
    CHECK(generated.output_ids.size() <= static_cast<size_t>(limit));
    runs.push_back({{"output_ids", generated.output_ids}, {"output_text", tokenizer.Decode(generated.output_ids)},
                    {"stop_reason", generated.stop_reason}, {"forward_calls", generated.forward_calls},
                    {"input_witnesses", witnesses}, {"prefill_seconds", prefill_seconds},
                    {"prefill_tokens_per_second", prompt.size() / prefill_seconds},
                    {"decode_forward_calls", decode_calls}, {"decode_seconds", decode_seconds},
                    {"decode_tokens_per_second", decode_calls ? decode_calls / decode_seconds : 0.0}});
  }
  const bool repeat_equal = runs[0]["output_ids"] == runs[1]["output_ids"];
  const bool native_only = vt::GetReferenceTierHits() == 0;
  CHECK(repeat_equal); CHECK(native_only);
  const nlohmann::json result = {
      {"schema", 1}, {"task", "R05"}, {"prompt", prompt_record}, {"runs", runs},
      {"timing_scope", "eager model forward plus full-row download and greedy selection; excludes reset, state exports and witness logging; request0 includes lazy weight uploads/compilation"},
      {"kv_page_size", page}, {"kv_blocks_per_layer", blocks}, {"token_budget", budget},
      {"sampling", "native_cpp_greedy_full_vocabulary"}, {"mtp", false}, {"graphs", false},
      {"prefix_reuse", false}, {"executed", true}, {"operator_pass", nullptr},
      {"target_parity_pass", false}, {"autonomous_smoke_pass", repeat_equal && native_only},
      {"serving_qualified", false}, {"blocking_issue_ids", {"S1_D64", "S1_strict_state"}},
      {"qualification", "functional development smoke; frozen numerical failures remain open"},
      {"state_exports", {{"layer", 0}, {"slot", 0}, {"conv_shape", {1, 10240, 3}},
                         {"ssm_shape", {1, 48, 128, 128}}, {"initial_state", "zero"}}}};
  std::ofstream report(output_dir / "result.json"); report << result.dump(2) << '\n';
  report.close(); REQUIRE(report.good());
}

TEST_CASE("XPU dense EXL3 FP16: QKVZ/QKV groups preserve independent source projections") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto c = Config();
  auto w = Weights(c, false);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  for (bool qkv : {false, true}) {
    CAPTURE(qkv);
    auto& g = w.layers[0].gdn;
    auto& a = w.layers[1].attn;
    const std::vector<const vllm::Exl3Weight*> sources = qkv
        ? std::vector<const vllm::Exl3Weight*>{&a.q_proj_exl3, &a.k_proj_exl3, &a.v_proj_exl3}
        : std::vector<const vllm::Exl3Weight*>{&g.in_proj_qkv_exl3, &g.in_proj_z_exl3};
    auto& cache = qkv ? a.qkv_proj_exl3 : g.in_proj_qkvz_exl3;
    const void* resident = nullptr;
    for (int64_t m : {1, 4, 129}) {
      CAPTURE(m);
      const auto values = xpu_test::Values(m * 128, 13, 0.007f);
      std::vector<uint16_t> input_bits(values.size());
      for (size_t i = 0; i < values.size(); ++i) input_bits[i] = vt::F32ToF16(values[i]);
      vllm::dense_attn::DBuf input(d, DType::kF16, {m, 128}, input_bits.data());
      auto merged = vllm::dense_exl3::GroupedLinear(d, input.t(), sources, cache);
      REQUIRE(merged.t().shape[1] == 512);
      REQUIRE(cache.suh.shape[0] == static_cast<int64_t>(sources.size()));
      CHECK(cache.trellis.host_released);
      CHECK(cache.source_map.host_released);
      if (resident) CHECK(cache.trellis.d_dev.get() == resident);
      else for (auto* source : sources) CHECK(source->trellis.d_dev == nullptr);
      resident = cache.trellis.d_dev.get();
      // Materialize logical views with the same native Copy used by the
      // unfused attention consumer, and compare every row/column independently.
      int64_t offset = 0;
      for (auto* source : sources) {
        const int64_t width = source->OutFeatures();
        auto view = merged.t().Slice(1, offset, offset + width);
        REQUIRE(view.stride[0] == 512);
        vllm::dense_attn::DBuf copy(d, DType::kF16, {m, width});
        vt::Copy(d.q, copy.t(), view);
        vllm::OwnedTensor none;
        auto separate = vllm::dense_exl3::Linear(d, input.t(), none, *source, DType::kF16);
        std::vector<uint16_t> got(m * width), expected(m * width);
        copy.Download(d, got.data());
        separate.Download(d, expected.data());
        CHECK(got == expected);
        CHECK(std::any_of(got.begin(), got.end(), [](uint16_t v) { return vt::F16ToF32(v) != 0; }));
        offset += width;
      }
    }
  }
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU dense EXL3 MM embed: ordered visual rows and authoritative IDs") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto c=Config(); c.raw["quantization_config"]={{"quant_method","exl3"}};
  const auto w=Weights(c,false);
  const auto read=[&](const vt::Tensor& tensor) {
    std::vector<unsigned char> bytes(tensor.Bytes());
    auto& b=vt::GetBackend(gpu.q.device.type);
    b.Copy(gpu.q,bytes.data(),tensor.data,bytes.size()); b.Synchronize(gpu.q);
    return bytes;
  };
  const std::vector<int32_t> ids{3,7,2,5,9},stale(5,0),positions{0,1,2,3,4,0,1,2,3,4,0,1,2,3,4};
  xpu_test::Buffer device_ids(gpu.q,DType::kI32,{5}), first(gpu.q,DType::kF16,{4,128}),
      second(gpu.q,DType::kF16,{3,128});
  device_ids.upload(ids.data());
  first.put(xpu_test::Values(4*128,13,0.5)); second.put(xpu_test::Values(3*128,29,0.25));
  const auto first_bytes=first.download(),second_bytes=second.download(),id_bytes=device_ids.download();
  std::vector<unsigned char> retained_bytes;
  vllm::MmForwardBuffers retained;
  for (const auto& mask : {std::vector<char>{0,0,0,0,0},std::vector<char>{0,1,1,0,1},
                           std::vector<char>{1,1,1,1,1}}) {
    std::vector<vt::Tensor> slices;
    if (mask[1] && !mask[0]) {
      // The unused neighboring source rows are deliberately different.
      slices={first.tensor.Slice(0,1,3),second.tensor.Slice(0,2,3)};
    } else if (mask[0]) slices={first.tensor.Slice(0,0,3),second.tensor.Slice(0,1,3)};
    std::vector<unsigned char> expected(5*128*2);
    for (size_t row=0;row<ids.size();++row)
      std::memcpy(expected.data()+256*row,w.embed_tokens.bytes.data()+256*ids[row],256);
    size_t destination=0;
    for (const auto& slice : slices) {
      const auto source=read(slice);
      for (int64_t row=0;row<slice.shape[0];++row) {
        while (!mask[destination]) ++destination;
        std::memcpy(expected.data()+256*destination,source.data()+256*row,256);
        ++destination;
      }
    }
    for (bool mirrored : {false,true}) {
      vllm::MmEmbedInputs input;
      input.token_ids=mirrored ? &stale : &ids;
      input.mm_embeds=&slices; input.is_mm_embed=&mask; input.mrope_positions=&positions;
      input.device_token_ids=mirrored ? static_cast<int32_t*>(device_ids.tensor.data) : nullptr;
      input.host_token_ids_stale=mirrored;
      const auto result=vllm::Qwen3_5DenseEmbedMultimodal(w,c,gpu.q,input);
      REQUIRE(result.storage.size()==2);
      REQUIRE(result.mm.inputs_embeds.dtype==DType::kF16);
      xpu_test::SameBytes(read(result.mm.inputs_embeds),expected);
      const auto pos=read(result.mm.positions3);
      REQUIRE(pos.size()==positions.size()*sizeof(int32_t));
      CHECK(std::memcmp(pos.data(),positions.data(),pos.size())==0);
      if (!retained.storage.empty()) xpu_test::SameBytes(read(retained.mm.inputs_embeds),retained_bytes);
      else { retained=result; retained_bytes=expected; }
      if (mirrored) {
        auto bad=input; bad.device_token_ids=nullptr;
        CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseEmbedMultimodal(w,c,gpu.q,bad),
                            doctest::Contains("stale host IDs"),std::runtime_error);
      }
    }
  }
  xpu_test::SameBytes(first.download(),first_bytes);
  xpu_test::SameBytes(second.download(),second_bytes);
  xpu_test::SameBytes(device_ids.download(),id_bytes);
  std::vector<char> invalid_mask{1,0,0,0,0}; std::vector<vt::Tensor> slices;
  vllm::MmEmbedInputs invalid;
  invalid.token_ids=&ids; invalid.mm_embeds=&slices;
  invalid.is_mm_embed=&invalid_mask; invalid.mrope_positions=&positions;
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseEmbedMultimodal(w,c,gpu.q,invalid),
                      doctest::Contains("balance the placeholder mask"),std::runtime_error);
  auto wrong=first.tensor; wrong.dtype=DType::kBF16; slices={wrong};
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseEmbedMultimodal(w,c,gpu.q,invalid),
                      doctest::Contains("FP16 visual rows"),std::runtime_error);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU dense EXL3 MM: device embeddings, positions and owned hidden outputs") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto c = Config();
  c.raw["quantization_config"] = {{"quant_method","exl3"}};
  c.rope_parameters.mrope_section = {11,11,10};
  c.rope_parameters.mrope_interleaved = true;
  c.max_position_embeddings = 4096;
  const auto w = Weights(c,false);
  auto registered = vllm::BorrowQwen3_5DenseLoadedModel(w);
  xpu_test::Buffer kv(gpu.q,DType::kF16,{2*2*8*128}),
      conv(gpu.q,DType::kF16,{2,384,3}), ssm(gpu.q,DType::kF32,{2,1,128,128}),
      embeds(gpu.q,DType::kF16,{4,128}), axes(gpu.q,DType::kI32,{3,4}),
      actual_ids(gpu.q,DType::kI32,{4});
  auto reset = [&] {
    kv.put(std::vector<float>(2*2*8*128,0));
    conv.put(std::vector<float>(2*384*3,0));
    ssm.put(std::vector<float>(2*128*128,0));
  };
  const auto read = [&](const vt::Tensor& tensor) {
    std::vector<unsigned char> bytes(tensor.Bytes());
    auto& b=vt::GetBackend(gpu.q.device.type);
    b.Copy(gpu.q,bytes.data(),tensor.data,bytes.size()); b.Synchronize(gpu.q);
    return bytes;
  };
  vllm::PagedKvCache cache;
  cache.data=kv.tensor.data; cache.dtype=DType::kF16; cache.num_blocks=2;
  cache.block_size=8; cache.num_kv_heads=1; cache.head_size=128;
  std::vector<vllm::PagedKvCache> caches{cache};
  std::vector<vllm::GdnStateCache> states(1);
  states[0].conv_state=conv.tensor; states[0].ssm_state=ssm.tensor;
  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs=1; am.num_actual_tokens=4;
  am.query_start_loc=am.query_start_loc_cpu={0,4};
  am.seq_lens=am.seq_lens_cpu={4}; am.max_query_len=am.max_seq_len=4;
  am.block_table_num_cols=1; am.block_table_tensor={1};
  am.slot_mapping={8,9,10,11}; am.causal=true;
  vllm::v1::GDNAttentionMetadata gm;
  gm.num_actual_tokens=gm.num_prefill_tokens=4; gm.num_prefills=1;
  gm.non_spec_state_indices_tensor=std::vector<int32_t>{1};
  gm.non_spec_query_start_loc=gm.prefill_query_start_loc=std::vector<int32_t>{0,4};
  gm.has_initial_state=gm.prefill_has_initial_state=std::vector<uint8_t>{0};
  gm.prefill_state_indices=std::vector<int32_t>{1};
  const auto chunks=vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
  gm.batch_ptr=chunks.batch_ptr; gm.token_chunk_offset_ptr=chunks.token_chunk_offset_ptr;
  const std::vector<int32_t> logical{0,1,2,3}, placeholders{0,0,0,0}, gather{3};
  const int32_t same_axes[]={0,1,2,3,0,1,2,3,0,1,2,3};
  const std::vector<int32_t> axis_source(std::begin(same_axes),std::end(same_axes));
  axes.upload(same_axes);
  std::vector<unsigned char> first_logits;
  for (const auto& ids : {std::vector<int32_t>{3,7,2,5},std::vector<int32_t>{3,9,2,5}}) {
    std::vector<uint16_t> merged(4*128);
    for (size_t row=0;row<ids.size();++row)
      std::memcpy(merged.data()+128*row,w.embed_tokens.bytes.data()+256*ids[row],256);
    embeds.upload(merged.data());
    const auto input_bytes=embeds.download(), position_bytes=axes.download();
    reset();
    vllm::Qwen3_5MTPHiddenStates plain_tap;
    const auto plain=vllm::Qwen3_5DenseModel::ForwardDeviceTap(
        ids,logical,am,gm,caches,states,w,c,gpu.q,&plain_tap,gather);
    const auto expected_logits=read(plain.device_tensor), expected_hidden=read(plain_tap.tensor);
    const auto expected_kv=kv.download(), expected_conv=conv.download(), expected_ssm=ssm.download();
    if (first_logits.empty()) first_logits=expected_logits;
    else CHECK(expected_logits != first_logits);
    reset();
    vllm::Qwen3_5MTPHiddenStates tap;
    vllm::ModelForwardInput input{placeholders,logical,am,gm,caches,states,c,gpu.q,gather};
    input.num_reqs=1; input.hidden_tap=&tap;
    input.mm=vllm::MultiModalForwardInput{};
    input.mm->inputs_embeds=embeds.tensor; input.mm->positions3=axes.tensor;
    auto actual=vllm::Qwen3_5DenseForwardEmbeddings(input,w);
    REQUIRE(actual.on_device()); REQUIRE(actual.rows==1); REQUIRE(actual.vocab==128);
    REQUIRE(tap.storage != nullptr); REQUIRE(tap.tensor.dtype==DType::kF16);
    xpu_test::SameBytes(read(actual.device_tensor),expected_logits);
    xpu_test::SameBytes(read(tap.tensor),expected_hidden);
    xpu_test::SameBytes(kv.download(),expected_kv);
    xpu_test::SameBytes(conv.download(),expected_conv);
    xpu_test::SameBytes(ssm.download(),expected_ssm);
    reset();
    // A graph hint must not substitute the text embedding/1-D rotation path.
    input.pure_decode=true;
    actual=vllm::ModelRegistry::Forward(*registered,input);
    CHECK_FALSE(actual.non_owning_view);
    xpu_test::SameBytes(read(actual.device_tensor),expected_logits);
    xpu_test::SameBytes(read(tap.tensor),expected_hidden);
    xpu_test::SameBytes(kv.download(),expected_kv);
    xpu_test::SameBytes(conv.download(),expected_conv);
    xpu_test::SameBytes(ssm.download(),expected_ssm);
    input.pure_decode=false;
    reset();
    input.num_speculative_tokens=1;
    actual=vllm::ModelRegistry::Forward(*registered,input);
    xpu_test::SameBytes(read(actual.device_tensor),expected_logits);
    xpu_test::SameBytes(read(tap.tensor),expected_hidden);
    xpu_test::SameBytes(kv.download(),expected_kv);
    xpu_test::SameBytes(conv.download(),expected_conv);
    xpu_test::SameBytes(ssm.download(),expected_ssm);
    input.num_speculative_tokens=0;
    reset();
    const int32_t token_rows[]={3,0,2,5}; actual_ids.upload(token_rows);
    const std::vector<char> image_mask{0,1,0,0};
    const std::vector<vt::Tensor> visual_rows{embeds.tensor.Slice(0,1,2)};
    vllm::MmEmbedInputs merge;
    merge.token_ids=&placeholders; merge.mm_embeds=&visual_rows;
    merge.is_mm_embed=&image_mask; merge.mrope_positions=&axis_source;
    merge.device_token_ids=static_cast<int32_t*>(actual_ids.tensor.data);
    merge.host_token_ids_stale=true;
    const auto prepared=vllm::Qwen3_5DenseEmbedMultimodal(w,c,gpu.q,merge);
    auto merged_input=input; merged_input.mm=prepared.mm;
    merged_input.device_token_ids=merge.device_token_ids;
    merged_input.host_token_ids_stale=true;
    actual=vllm::ModelRegistry::Forward(*registered,merged_input);
    xpu_test::SameBytes(read(actual.device_tensor),expected_logits);
    xpu_test::SameBytes(read(tap.tensor),expected_hidden);
    xpu_test::SameBytes(kv.download(),expected_kv);
    xpu_test::SameBytes(conv.download(),expected_conv);
    xpu_test::SameBytes(ssm.download(),expected_ssm);
    const auto retained=tap;
    reset();
    actual=vllm::Qwen3_5DenseForwardEmbeddings(input,w);
    xpu_test::SameBytes(read(actual.device_tensor),expected_logits);
    xpu_test::SameBytes(read(retained.tensor),expected_hidden);
    xpu_test::SameBytes(embeds.download(),input_bytes);
    xpu_test::SameBytes(axes.download(),position_bytes);
    // Distinct spatial axes must affect the real language attention, while
    // ordinary cache addressing still writes the same logical slots.
    const int32_t spatial[]={0,1,2,3,0,17,51,103,0,23,71,157};
    axes.upload(spatial); reset();
    const auto different=vllm::Qwen3_5DenseForwardEmbeddings(input,w);
    CHECK(read(different.device_tensor) != expected_logits);
    const auto new_kv=kv.download();
    CHECK(std::all_of(new_kv.begin(),new_kv.begin()+8*2*128*2,
                      [](unsigned char v){return v==0;}));
    xpu_test::SameBytes(read(retained.tensor),expected_hidden);
    axes.upload(same_axes);
    // The host-logit carrier still uses the provided embeddings.
    reset(); input.hidden_tap=nullptr; input.gather_logits=false;
    const auto host=vllm::Qwen3_5DenseForwardEmbeddings(input,w);
    REQUIRE(host.host.size()*sizeof(float)==expected_logits.size());
    CHECK(std::memcmp(host.host.data(),expected_logits.data(),expected_logits.size())==0);
  }
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU dense EXL3 MM: invalid inputs fail before persistent state writes") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto c=Config(); c.raw["quantization_config"]={{"quant_method","exl3"}};
  c.rope_parameters.mrope_section={11,11,10};
  const auto w=Weights(c,false);
  xpu_test::Buffer embeds(gpu.q,DType::kF16,{1,128}), axes(gpu.q,DType::kI32,{3,1}),
      kv(gpu.q,DType::kF16,{2*8*128}), conv(gpu.q,DType::kF16,{1,384,3}),
      ssm(gpu.q,DType::kF32,{1,1,128,128});
  kv.put(std::vector<float>(2*8*128,2)); conv.put(std::vector<float>(384*3,3));
  ssm.put(std::vector<float>(128*128,4));
  const auto kv_initial=kv.download(),conv_initial=conv.download(),ssm_initial=ssm.download();
  const std::vector<int32_t> ids{0},positions{0},gather{};
  vllm::v1::CommonAttentionMetadata am; am.num_reqs=1;
  vllm::v1::GDNAttentionMetadata gm;
  vllm::PagedKvCache cache;
  cache.data=kv.tensor.data; cache.dtype=DType::kF16; cache.num_blocks=1;
  cache.block_size=8; cache.num_kv_heads=1; cache.head_size=128;
  std::vector<vllm::PagedKvCache> caches{cache}; std::vector<vllm::GdnStateCache> states(1);
  states[0].conv_state=conv.tensor; states[0].ssm_state=ssm.tensor;
  vllm::ModelForwardInput input{ids,positions,am,gm,caches,states,c,gpu.q,gather};
  input.num_reqs=1;
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(input,w),
                      doctest::Contains("multimodal inputs required"),std::runtime_error);
  input.mm=vllm::MultiModalForwardInput{};
  input.mm->inputs_embeds=embeds.tensor; input.mm->positions3=axes.tensor;
  auto bad=input; bad.mm->inputs_embeds.dtype=DType::kBF16;
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(bad,w),
                      doctest::Contains("FP16 embeddings"),std::runtime_error);
  auto bad_axes=input; bad_axes.mm->positions3.rank=1;
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(bad_axes,w),
                      doctest::Contains("positions [3,T]"),std::runtime_error);
  auto bad_spec=input; bad_spec.num_speculative_tokens=4;
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(bad_spec,w),
                      doctest::Contains("C1-C4 target-only"),std::runtime_error);
  auto bad_batch=input; bad_batch.num_reqs=2;
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(bad_batch,w),
                      doctest::Contains("C1-C4 target-only"),std::runtime_error);
  am.num_reqs=5;
  auto overlimit=input; overlimit.num_reqs=5;
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(overlimit,w),
                      doctest::Contains("C1-C4 target-only"),std::runtime_error);
  am.num_reqs=1;
  auto bad_stack=input; bad_stack.mm->deepstack_levels=1;
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(bad_stack,w),
                      doctest::Contains("no DeepStack"),std::runtime_error);
  c.rope_parameters.mrope_section={11,11,9};
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(input,w),
                      doctest::Contains("sections must cover"),std::runtime_error);
  c.rope_parameters.mrope_section={128,0,0}; c.rotary_dim=256;
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(input,w),
                      doctest::Contains("sections must cover"),std::runtime_error);
  c.rope_parameters.mrope_section={11,11,10}; c.rotary_dim=64;
  c.rope_theta=std::numeric_limits<double>::infinity();
  CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseForwardEmbeddings(input,w),
                      doctest::Contains("finite positive"),std::runtime_error);
  c.rope_theta=10000;
  c.architectures={"Qwen3_5ForCausalLM"};
  auto text_model=vllm::MakeQwen3_5DenseLoadedModel(Weights(c,false),c);
  REQUIRE(text_model->registration().architecture=="Qwen3_5ForCausalLM");
  CHECK_THROWS_WITH_AS(vllm::ModelRegistry::Forward(*text_model,input),
                      doctest::Contains("conditional-generation registration required"),std::runtime_error);
  xpu_test::SameBytes(kv.download(),kv_initial);
  xpu_test::SameBytes(conv.download(),conv_initial);
  xpu_test::SameBytes(ssm.download(),ssm_initial);
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU dense EXL3 FP16: paged hybrid prefill and decode preserve model/head precision") {
  vt::EnableOpProviderCallStats(true);
  const auto grouped_before = vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections;
  const auto packed_before = vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kXPU).selections;
  bool split_ba = false;
  SUBCASE("merged BA owner") {}
  SUBCASE("split BA owners") { split_ba = true; }
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto c = Config();
  const auto w = Weights(c, split_ba);
  xpu_test::Buffer kv(gpu.q, DType::kF16, {2 * 2 * 8 * 128});
  xpu_test::Buffer conv(gpu.q, DType::kF16, {2, 384, 3});
  xpu_test::Buffer ssm(gpu.q, DType::kF32, {2, 1, 128, 128});
  kv.put(std::vector<float>(2 * 2 * 8 * 128, 0));
  conv.put(std::vector<float>(2 * 384 * 3, 0));
  ssm.put(std::vector<float>(2 * 128 * 128, 0));
  vllm::PagedKvCache cache;
  cache.data = kv.tensor.data;
  cache.dtype = DType::kF16;
  cache.num_blocks = 2;
  cache.block_size = 8;
  cache.num_kv_heads = 1;
  cache.head_size = 128;
  std::vector<vllm::GdnStateCache> states(1);
  states[0].conv_state = conv.tensor;
  states[0].ssm_state = ssm.tensor;
  const std::vector<vllm::PagedKvCache> caches{cache};
  for (int step : {0, 1}) {
    CAPTURE(step);
    const std::vector<int32_t> ids = step == 0 ? std::vector<int32_t>{3, 7, 2, 5}
                                             : std::vector<int32_t>{9};
    const std::vector<int32_t> positions = step == 0 ? std::vector<int32_t>{0, 1, 2, 3}
                                                   : std::vector<int32_t>{4};
    const int32_t rows = static_cast<int32_t>(ids.size());
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1;
    am.num_actual_tokens = rows;
    am.query_start_loc = am.query_start_loc_cpu = {0, rows};
    am.seq_lens = am.seq_lens_cpu = {step == 0 ? 4 : 5};
    am.max_query_len = rows;
    am.max_seq_len = am.seq_lens[0];
    am.block_table_num_cols = 1;
    am.block_table_tensor = {1};
    for (int32_t p : positions) am.slot_mapping.push_back(8 + p);
    am.causal = true;
    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = rows;
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{1};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, rows};
    if (step == 0) {
      gm.num_prefills = 1;
      gm.num_prefill_tokens = rows;
      gm.has_initial_state = std::vector<uint8_t>{0};
      gm.prefill_query_start_loc = std::vector<int32_t>{0, rows};
      gm.prefill_state_indices = std::vector<int32_t>{1};
      gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto metadata = vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = metadata.batch_ptr;
      gm.token_chunk_offset_ptr = metadata.token_chunk_offset_ptr;
    } else {
      gm.num_decodes = gm.num_decode_tokens = 1;
    }
    vllm::Qwen3_5MTPHiddenStates tap;
    const auto logits = vllm::Qwen3_5DenseModel::ForwardDeviceTap(
        ids, positions, am, gm, caches, states, w, c, gpu.q, &tap, {rows - 1});
    REQUIRE(logits.device_tensor.dtype == DType::kF32);
    REQUIRE(logits.rows == 1);
    REQUIRE(tap.tensor.dtype == DType::kF16);
    REQUIRE(tap.tensor.shape[0] == rows);
    vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
    auto head = vllm::dense_exl3::Linear(d, tap.tensor.Slice(0, rows - 1, rows),
                                       w.lm_head, w.lm_head_exl3, DType::kF16);
    std::vector<uint16_t> expected(128);
    head.Download(d, expected.data());
    std::vector<float> got(128);
    auto& backend = vt::GetBackend(gpu.q.device.type);
    backend.Copy(gpu.q, got.data(), logits.device_tensor.data, got.size() * 4);
    backend.Synchronize(gpu.q);
    bool nonzero = false;
    for (size_t i = 0; i < got.size(); ++i) {
      CHECK(std::isfinite(got[i]));
      CHECK(got[i] == vt::F16ToF32(expected[i]));
      nonzero |= got[i] != 0;
    }
    CHECK(nonzero);
    if (step == 0) {
      // Pooling replays the same fresh prefill and gathers reordered hidden
      // rows before widening. The tap is an independently owned output.
      const auto pooled = vllm::Qwen3_5DenseModel::ForwardHidden(
          ids, positions, am, gm, caches, states, w, c, gpu.q, {3, 0});
      REQUIRE(pooled.rows == 2);
      REQUIRE(pooled.host.size() == 256);
      std::vector<uint16_t> hidden(4 * 128);
      backend.Copy(gpu.q, hidden.data(), tap.tensor.data, hidden.size() * 2);
      backend.Synchronize(gpu.q);
      for (int i = 0; i < 128; ++i) {
        CHECK(pooled.host[i] == vt::F16ToF32(hidden[3 * 128 + i]));
        CHECK(pooled.host[128 + i] == vt::F16ToF32(hidden[i]));
      }
    }
    const auto recurrent = ssm.floats();
    CHECK(std::all_of(recurrent.begin(), recurrent.begin() + 128 * 128,
                      [](float v) { return v == 0; }));
    CHECK(std::any_of(recurrent.begin() + 128 * 128, recurrent.end(),
                      [](float v) { return v != 0; }));
    auto invalid = states;
    invalid[0].conv_state.dtype = DType::kBF16;
    CHECK_THROWS_WITH_AS(vllm::Qwen3_5DenseModel::ForwardDevice(
        ids, positions, am, gm, caches, invalid, w, c, gpu.q, {rows - 1}),
        doctest::Contains("GDN convolution state must be FP16"), std::runtime_error);
  }
  CHECK(vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections > grouped_before);
  CHECK(vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kXPU).selections == packed_before);
  REQUIRE(w.layers[0].gdn.in_proj_qkvz_exl3.suh.shape[0] == 2);
  REQUIRE(w.layers[1].attn.qkv_proj_exl3.suh.shape[0] == 3);
  CHECK(w.layers[0].gdn.in_proj_qkv_exl3.trellis.d_dev == nullptr);
  CHECK(w.layers[0].gdn.in_proj_z_exl3.trellis.d_dev == nullptr);
  CHECK(w.layers[1].attn.q_proj_exl3.trellis.d_dev == nullptr);
  CHECK(w.layers[1].attn.k_proj_exl3.trellis.d_dev == nullptr);
  CHECK(w.layers[1].attn.v_proj_exl3.trellis.d_dev == nullptr);
  CHECK(vt::GetReferenceTierHits() == 0);
}

TEST_CASE("XPU dense EXL3 FP16: unpaged model entries preserve hidden and head rounding") {
  vt::EnableOpProviderCallStats(true);
  const auto grouped_before = vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections;
  const auto packed_before = vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kXPU).selections;
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  const auto c = Config();
  const auto w = Weights(c, false);
  const std::vector<int32_t> ids{3, 7, 2, 5}, positions{0, 1, 2, 3};
  const auto full = vllm::Qwen3_5DenseModel::ForwardDense(ids, positions, w, c, gpu.q);
  const auto last = vllm::Qwen3_5DenseModel::ForwardDenseLastLogits(ids, positions, w, c, gpu.q);
  const auto hidden = vllm::Qwen3_5DenseModel::ForwardDenseHidden(ids, positions, w, c, gpu.q);
  REQUIRE(full.size() == 4 * 128);
  REQUIRE(last.size() == 128);
  REQUIRE(hidden.size() == 4 * 128);
  // Producer M1 GEMV has half tile-local accumulation; the M4 DPAS head
  // accumulates in F32. They need numerical agreement, not bit identity.
  double difference = 0, magnitude = 0;
  for (size_t i = 0; i < last.size(); ++i) {
    const double delta = double(last[i]) - full[3 * 128 + i];
    difference += delta * delta;
    magnitude += double(full[3 * 128 + i]) * full[3 * 128 + i];
  }
  REQUIRE(magnitude > 0);
  const double relative = std::sqrt(difference / magnitude);
  std::cout << "MODEL_SMALLM_HEAD M1_vs_M4_relative=" << relative << '\n';
  CHECK(std::isfinite(relative));
  CHECK(relative < 2e-3);
  for (const auto* values : {&full, &hidden}) {
    bool nonzero = false;
    for (float value : *values) {
      CHECK(std::isfinite(value));
      CHECK(value == vt::F16ToF32(vt::F32ToF16(value)));
      nonzero |= value != 0;
    }
    CHECK(nonzero);
  }
  // The hidden API stops before the head. Independently invoke the F16 head
  // on its selected row to check that the logits API does not omit rounding.
  std::vector<uint16_t> row(128);
  for (size_t i = 0; i < row.size(); ++i) row[i] = vt::F32ToF16(hidden[3 * 128 + i]);
  vllm::dense_attn::Dev d{vt::GetBackend(gpu.q.device.type), gpu.q, DType::kF16};
  vllm::dense_attn::DBuf input(d, DType::kF16, {1, 128}, row.data());
  auto head = vllm::dense_exl3::Linear(d, input.t(), w.lm_head, w.lm_head_exl3, DType::kF16);
  std::vector<uint16_t> expected(128);
  head.Download(d, expected.data());
  for (size_t i = 0; i < row.size(); ++i) CHECK(last[i] == vt::F16ToF32(expected[i]));
  // Independently check every full-logit row against the producer M4 head,
  // so changing the selected-row comparison cannot hide an indexing error.
  std::vector<uint16_t> all_hidden(hidden.size()), all_expected(full.size());
  for (size_t i = 0; i < hidden.size(); ++i) all_hidden[i] = vt::F32ToF16(hidden[i]);
  vllm::dense_attn::DBuf all_input(d, DType::kF16, {4, 128}, all_hidden.data());
  auto all_head = vllm::dense_exl3::Linear(d, all_input.t(), w.lm_head,
                                         w.lm_head_exl3, DType::kF16);
  all_head.Download(d, all_expected.data());
  for (size_t i = 0; i < full.size(); ++i) CHECK(full[i] == vt::F16ToF32(all_expected[i]));
  CHECK(vt::GetOpProviderStats(vt::OpId::kExl3GroupedLinear, vt::DeviceType::kXPU).selections > grouped_before);
  CHECK(vt::GetOpProviderStats(vt::OpId::kExl3Gemm, vt::DeviceType::kXPU).selections == packed_before);
  CHECK(w.layers[0].gdn.in_proj_qkvz_exl3.trellis.host_released);
  CHECK(w.layers[1].attn.qkv_proj_exl3.trellis.host_released);
  CHECK(vt::GetReferenceTierHits() == 0);
}


TEST_CASE("XPU dense EXL3 MM: MTP3 verification preserves native forced acceptance histories") {
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto c=Config(); c.raw["quantization_config"]={{"quant_method","exl3"}};
  c.rope_parameters.mrope_section={11,11,10}; c.rope_parameters.mrope_interleaved=true;
  c.max_position_embeddings=64;
  const auto w=Weights(c,false);
  auto registered=vllm::BorrowQwen3_5DenseLoadedModel(w);
  // A permuted state row and a nonidentity physical KV page prevent an identity
  // selector/slot mapping from passing. Spec conv history has (K-1)+3 taps.
  xpu_test::Buffer plain_kv(gpu.q,DType::kF16,{2*2*16*128}),
      image_kv(gpu.q,DType::kF16,{2*2*16*128}),
      plain_conv(gpu.q,DType::kF16,{4,384,6}), image_conv(gpu.q,DType::kF16,{4,384,6}),
      plain_ssm(gpu.q,DType::kF32,{4,1,128,128}), image_ssm(gpu.q,DType::kF32,{4,1,128,128}),
      embeds(gpu.q,DType::kF16,{4,128}), axes(gpu.q,DType::kI32,{3,4});
  auto cache=[&](xpu_test::Buffer& kv) {
    vllm::PagedKvCache k; k.data=kv.tensor.data; k.dtype=DType::kF16;
    k.num_blocks=2; k.block_size=16; k.num_kv_heads=1; k.head_size=128;
    return std::vector<vllm::PagedKvCache>{k};
  };
  auto plain_caches=cache(plain_kv),image_caches=cache(image_kv);
  std::vector<vllm::GdnStateCache> plain_states(1),image_states(1);
  plain_states[0].conv_state=plain_conv.tensor; plain_states[0].ssm_state=plain_ssm.tensor;
  image_states[0].conv_state=image_conv.tensor; image_states[0].ssm_state=image_ssm.tensor;
  auto read=[&](const vt::Tensor& tensor) {
    std::vector<unsigned char> bytes(tensor.Bytes()); auto& backend=vt::GetBackend(gpu.q.device.type);
    backend.Copy(gpu.q,bytes.data(),tensor.data,bytes.size()); backend.Synchronize(gpu.q); return bytes;
  };
  std::vector<unsigned char> previous_accepted_logits;
  for (int accepted : {1,2,3,4}) {
    CAPTURE(accepted); // reject at draft 0/1/2, or all three drafts accepted.
    plain_kv.put(std::vector<float>(2*2*16*128,0)); image_kv.put(std::vector<float>(2*2*16*128,0));
    plain_conv.put(std::vector<float>(4*384*6,0)); image_conv.put(std::vector<float>(4*384*6,0));
    plain_ssm.put(std::vector<float>(4*128*128,0)); image_ssm.put(std::vector<float>(4*128*128,0));
    for (int step=0;step<3;++step) {
      const std::vector<int32_t> ids=step==0 ? std::vector<int32_t>{3,7,2,5} :
          step==1 ? std::vector<int32_t>{9,13,22,17} : std::vector<int32_t>{11,19,23,29};
      const int start=step==0 ? 0 : step==1 ? 4 : 4+accepted;
      const std::vector<int32_t> positions{start,start+1,start+2,start+3},gather{},placeholders(4,0);
      vllm::v1::CommonAttentionMetadata am;
      am.num_reqs=1; am.num_actual_tokens=4; am.max_query_len=4; am.max_seq_len=start+4;
      am.query_start_loc=am.query_start_loc_cpu={0,4}; am.seq_lens=am.seq_lens_cpu={start+4};
      am.num_computed_tokens_cpu={start}; am.block_table_num_cols=1; am.block_table_tensor={1};
      am.slot_mapping={16+start,17+start,18+start,19+start}; am.causal=true;
      vllm::v1::GDNAttentionMetadata gm; gm.num_actual_tokens=4;
      if (step==0) {
        gm.num_prefill_tokens=4; gm.num_prefills=1;
        gm.non_spec_state_indices_tensor=gm.prefill_state_indices=std::vector<int32_t>{3};
        gm.non_spec_query_start_loc=gm.prefill_query_start_loc=std::vector<int32_t>{0,4};
        gm.has_initial_state=gm.prefill_has_initial_state=std::vector<uint8_t>{0};
        const auto chunks=vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
        gm.batch_ptr=chunks.batch_ptr; gm.token_chunk_offset_ptr=chunks.token_chunk_offset_ptr;
      } else {
        gm.num_spec_decodes=1; gm.num_spec_decode_tokens=4; gm.spec_state_indices_num_cols=4;
        gm.spec_state_indices_tensor=std::vector<int32_t>{3,2,1,0};
        gm.spec_query_start_loc=std::vector<int32_t>{0,4}; gm.spec_sequence_masks=std::vector<uint8_t>{1};
        gm.spec_token_indx=std::vector<int32_t>{0,1,2,3}; gm.num_accepted_tokens=std::vector<int32_t>{step==1 ? 1 : accepted};
      }
      std::vector<uint16_t> merged(4*128); std::vector<int32_t> rotary;
      for (size_t row=0;row<ids.size();++row)
        std::memcpy(merged.data()+128*row,w.embed_tokens.bytes.data()+256*ids[row],256);
      for (int axis=0;axis<3;++axis) rotary.insert(rotary.end(),positions.begin(),positions.end());
      embeds.upload(merged.data()); axes.upload(rotary.data());
      vllm::Qwen3_5MTPHiddenStates plain_tap,image_tap;
      const auto expected=vllm::Qwen3_5DenseModel::ForwardDeviceTap(
          ids,positions,am,gm,plain_caches,plain_states,w,c,gpu.q,&plain_tap,gather);
      vllm::ModelForwardInput input{placeholders,positions,am,gm,image_caches,image_states,c,gpu.q,gather};
      input.num_reqs=1; input.num_speculative_tokens=3; input.hidden_tap=&image_tap;
      input.mm=vllm::MultiModalForwardInput{}; input.mm->inputs_embeds=embeds.tensor; input.mm->positions3=axes.tensor;
      const auto actual=vllm::ModelRegistry::Forward(*registered,input);
      xpu_test::SameBytes(read(actual.device_tensor),read(expected.device_tensor));
      xpu_test::SameBytes(read(image_tap.tensor),read(plain_tap.tensor));
      xpu_test::SameBytes(image_kv.download(),plain_kv.download());
      xpu_test::SameBytes(image_conv.download(),plain_conv.download());
      xpu_test::SameBytes(image_ssm.download(),plain_ssm.download());
      if (step==2) {
        const auto logits=read(actual.device_tensor);
        if (!previous_accepted_logits.empty()) CHECK(logits!=previous_accepted_logits);
        previous_accepted_logits=logits;
      }
    }
  }
  CHECK(vt::GetReferenceTierHits()==0);
}

TEST_CASE("XPU dense EXL3 MM: MTP3 graph owns embeddings and axes across replay and fresh prefixes") {
  // A global fast-verifier request must still allow this unsupported small
  // F16 geometry to replay through generic attention at growing contexts.
  struct RestoreVerify {
    std::string old; bool had;
    ~RestoreVerify() {
      if (had) setenv("VT_XPU_XE2_VERIFY", old.c_str(), 1);
      else unsetenv("VT_XPU_XE2_VERIFY");
    }
  } restore{std::getenv("VT_XPU_XE2_VERIFY") ? std::getenv("VT_XPU_XE2_VERIFY") : "",
            std::getenv("VT_XPU_XE2_VERIFY") != nullptr};
  REQUIRE(setenv("VT_XPU_XE2_VERIFY", "1", 1) == 0);
  xpu_test::Queue gpu(vt::DeviceType::kXPU);
  auto c=Config(); c.raw["quantization_config"]={{"quant_method","exl3"}};
  c.rope_parameters.mrope_section={11,11,10}; c.rope_parameters.mrope_interleaved=true;
  c.max_position_embeddings=128; c.architectures={"Qwen3_5ForConditionalGeneration"};
  const auto w=Weights(c,false);
  const int max_concurrency=4;
  const int max_tokens=16;
  vllm::Qwen3_5DenseDecodeGraph graph(w,c,gpu.q,4);
  auto registered=vllm::BorrowQwen3_5DenseLoadedModel(w);
  // A permuted state row and a nonidentity physical KV page prevent an identity
  // selector/slot mapping from passing. Spec conv history has (K-1)+3 taps.
  xpu_test::Buffer plain_kv(gpu.q,DType::kF16,{2*2*max_concurrency*64*128}),
      image_kv(gpu.q,DType::kF16,{2*2*max_concurrency*64*128}),
      plain_conv(gpu.q,DType::kF16,{4*max_concurrency,384,6}), image_conv(gpu.q,DType::kF16,{4*max_concurrency,384,6}),
      plain_ssm(gpu.q,DType::kF32,{4*max_concurrency,1,128,128}), image_ssm(gpu.q,DType::kF32,{4*max_concurrency,1,128,128}),
      embeds(gpu.q,DType::kF16,{max_tokens,128}), axes(gpu.q,DType::kI32,{3,max_tokens});
  auto cache=[&](xpu_test::Buffer& kv) {
    vllm::PagedKvCache k; k.data=kv.tensor.data; k.dtype=DType::kF16;
    k.num_blocks=2*max_concurrency; k.block_size=64; k.num_kv_heads=1; k.head_size=128;
    return std::vector<vllm::PagedKvCache>{k};
  };
  auto plain_caches=cache(plain_kv),image_caches=cache(image_kv);
  std::vector<vllm::GdnStateCache> plain_states(1),image_states(1);
  plain_states[0].conv_state=plain_conv.tensor; plain_states[0].ssm_state=plain_ssm.tensor;
  image_states[0].conv_state=image_conv.tensor; image_states[0].ssm_state=image_ssm.tensor;
  auto read=[&](const vt::Tensor& tensor) {
    std::vector<unsigned char> bytes(tensor.Bytes()); auto& backend=vt::GetBackend(gpu.q.device.type);
    backend.Copy(gpu.q,bytes.data(),tensor.data,bytes.size()); backend.Synchronize(gpu.q); return bytes;
  };
  std::optional<vllm::ForwardLogits> across_shape_logits;
  std::optional<vllm::Qwen3_5MTPHiddenStates> across_shape_tap;
  std::vector<unsigned char> across_shape_logit_bytes, across_shape_tap_bytes;
  const std::vector<std::pair<int,bool>> shapes{{1,true},{2,true},{4,true},
      {3,true},{1,false},{4,false},{2,true},{1,true}};
  for (const auto& [concurrency,use_mm] : shapes) {
    CAPTURE(concurrency);
    CAPTURE(use_mm);
    const auto prior_replays = graph.replay_count();
    const int tokens=4*concurrency;
    embeds.tensor.shape[0]=tokens;
    axes.tensor.shape[1]=tokens; axes.tensor.stride[0]=tokens;
    std::vector<unsigned char> previous_accepted_logits;
    for (int accepted : {1,2,3,4}) {
      CAPTURE(accepted); // reject at draft 0/1/2, or all three drafts accepted.
      plain_kv.put(std::vector<float>(2*2*max_concurrency*64*128,0)); image_kv.put(std::vector<float>(2*2*max_concurrency*64*128,0));
      plain_conv.put(std::vector<float>(4*max_concurrency*384*6,0)); image_conv.put(std::vector<float>(4*max_concurrency*384*6,0));
      plain_ssm.put(std::vector<float>(4*max_concurrency*128*128,0)); image_ssm.put(std::vector<float>(4*max_concurrency*128*128,0));
      std::optional<vllm::ForwardLogits> retained_logits;
      std::optional<vllm::Qwen3_5MTPHiddenStates> retained_tap;
      std::vector<unsigned char> retained_logit_bytes, retained_tap_bytes;
      for (int step=0;step<7;++step) {
        std::vector<int32_t> ids,positions,gather{},placeholders(tokens,0),accepted_by_request;
        const std::vector<int32_t> base_ids=step==0 ? std::vector<int32_t>{3,7,2,5} :
            step==1 ? std::vector<int32_t>{9,13,22,17} : std::vector<int32_t>{11,19,23,29};
        vllm::v1::CommonAttentionMetadata am;
        am.num_reqs=concurrency; am.num_actual_tokens=tokens; am.max_query_len=4;
        am.block_table_num_cols=1; am.causal=true; am.query_start_loc={0};
        std::vector<int32_t> prefill_slots,spec_slots,spec_indices;
        for (int request=0;request<concurrency;++request) {
          const int accepted_here=1+(accepted-1+request)%4;
          const int start=step==0 ? 0 : step==1 ? 4 : 4+accepted_here+4*(step-2);
          am.max_seq_len=std::max(am.max_seq_len,start+4);
          am.query_start_loc.push_back(4*(request+1));
          am.seq_lens.push_back(start+4); am.num_computed_tokens_cpu.push_back(start);
          const int page=2*request+1;
          am.block_table_tensor.push_back(page);
          prefill_slots.push_back(4*request+3);
          accepted_by_request.push_back(step==1 ? 1 : step==2 ? accepted_here : 4);
          for (int row=0;row<4;++row) {
            ids.push_back(base_ids[row]+request*3);
            positions.push_back(start+row);
            am.slot_mapping.push_back(page*64+start+row);
            spec_slots.push_back(4*request+3-row);
            spec_indices.push_back(4*request+row);
          }
        }
        am.query_start_loc_cpu=am.query_start_loc; am.seq_lens_cpu=am.seq_lens;
        vllm::v1::GDNAttentionMetadata gm; gm.num_actual_tokens=tokens;
        if (step==0) {
          gm.num_prefill_tokens=tokens; gm.num_prefills=concurrency;
          gm.non_spec_state_indices_tensor=gm.prefill_state_indices=prefill_slots;
          gm.non_spec_query_start_loc=gm.prefill_query_start_loc=am.query_start_loc;
          gm.has_initial_state=gm.prefill_has_initial_state=std::vector<uint8_t>(concurrency,0);
          const auto chunks=vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
          gm.batch_ptr=chunks.batch_ptr; gm.token_chunk_offset_ptr=chunks.token_chunk_offset_ptr;
        } else {
          gm.num_spec_decodes=concurrency; gm.num_spec_decode_tokens=tokens; gm.spec_state_indices_num_cols=4;
          gm.spec_state_indices_tensor=spec_slots; gm.spec_query_start_loc=am.query_start_loc;
          gm.spec_sequence_masks=std::vector<uint8_t>(concurrency,1);
          gm.spec_token_indx=spec_indices; gm.num_accepted_tokens=accepted_by_request;
        }
        std::vector<uint16_t> merged(max_tokens*128); std::vector<int32_t> rotary;
        for (size_t row=0;row<ids.size();++row)
          std::memcpy(merged.data()+128*row,w.embed_tokens.bytes.data()+256*ids[row],256);
        for (int axis=0;axis<3;++axis) for (size_t row=0;row<positions.size();++row)
          rotary.push_back(use_mm ? positions[row] + 17*axis + accepted +
              5*static_cast<int>(row/4) : positions[row]);
        rotary.resize(3*max_tokens);
        embeds.upload(merged.data()); axes.upload(rotary.data());
        vllm::Qwen3_5MTPHiddenStates plain_tap,image_tap;
        // Eager MM is the qualified reference. Change all axes and the visual
        // prefix between histories; physical KV positions remain independent.
        vllm::ModelForwardInput reference{placeholders,positions,am,gm,plain_caches,plain_states,c,gpu.q,gather};
        reference.num_reqs=concurrency; reference.num_speculative_tokens=3; reference.hidden_tap=&plain_tap;
        reference.mm=vllm::MultiModalForwardInput{};
        reference.mm->inputs_embeds=embeds.tensor; reference.mm->positions3=axes.tensor;
        const auto expected=use_mm ? vllm::Qwen3_5DenseForwardEmbeddings(reference,w) :
          vllm::Qwen3_5DenseModel::ForwardDeviceTap(ids,positions,am,gm,plain_caches,
              plain_states,w,c,gpu.q,&plain_tap,gather);
        const auto& graph_ids=use_mm ? placeholders : ids;
        vllm::ModelForwardInput input{graph_ids,positions,am,gm,image_caches,image_states,c,gpu.q,gather};
        input.num_reqs=concurrency; input.num_speculative_tokens=3; input.hidden_tap=&image_tap;
        input.mm=vllm::MultiModalForwardInput{}; input.mm->inputs_embeds=embeds.tensor; input.mm->positions3=axes.tensor;
        if (step==1) {
          const auto kv_before=image_kv.download(),conv_before=image_conv.download(),ssm_before=image_ssm.download();
          auto invalid=input; invalid.mm->positions3.shape[1]--;
          CHECK_THROWS_WITH_AS(graph.Step(graph_ids,positions,am,gm,image_caches,image_states,nullptr,&image_tap,&invalid),
                               doctest::Contains("positions [3,T]"),std::runtime_error);
          auto missing=input; missing.mm.reset();
          CHECK_THROWS_WITH_AS(graph.Step(graph_ids,positions,am,gm,image_caches,image_states,nullptr,&image_tap,&missing),
                               doctest::Contains("multimodal inputs required"),std::runtime_error);
          xpu_test::SameBytes(image_kv.download(),kv_before);
          xpu_test::SameBytes(image_conv.download(),conv_before);
          xpu_test::SameBytes(image_ssm.download(),ssm_before);
        }
        const auto actual=step==0 ?
            (use_mm ? vllm::ModelRegistry::Forward(*registered,input) :
             vllm::Qwen3_5DenseModel::ForwardDeviceTap(ids,positions,am,gm,image_caches,
                 image_states,w,c,gpu.q,&image_tap,gather)) :
            graph.Step(graph_ids,positions,am,gm,image_caches,image_states,nullptr,
                       &image_tap,use_mm ? &input : nullptr);
        if (step>0) {
          // Poison caller-owned inputs after submission. A later replay must use
          // newly staged graph inputs, not a captured pointer into these bytes.
          embeds.put(std::vector<float>(max_tokens*128,9));
          axes.upload(std::vector<int32_t>(3*max_tokens,777).data());
        }
        xpu_test::SameBytes(read(actual.device_tensor),read(expected.device_tensor));
        xpu_test::SameBytes(read(image_tap.tensor),read(plain_tap.tensor));
        xpu_test::SameBytes(image_kv.download(),plain_kv.download());
        xpu_test::SameBytes(image_conv.download(),plain_conv.download());
        xpu_test::SameBytes(image_ssm.download(),plain_ssm.download());
        CHECK(vt::xpu::GetMemoryInfo(gpu.q.device.index).graph_count<=6);
        if (across_shape_logits) {
          xpu_test::SameBytes(read(across_shape_logits->device_tensor),across_shape_logit_bytes);
          xpu_test::SameBytes(read(across_shape_tap->tensor),across_shape_tap_bytes);
        }
        if (!across_shape_logits && step==3) {
          across_shape_logits=actual; across_shape_tap=image_tap;
          across_shape_logit_bytes=read(actual.device_tensor);
          across_shape_tap_bytes=read(image_tap.tensor);
        }
        if (step==3) {
          retained_logits=actual; retained_tap=image_tap;
          retained_logit_bytes=read(actual.device_tensor); retained_tap_bytes=read(image_tap.tensor);
        } else if (step>3) {
          // Holding both paired outputs forces retirement when their slot would
          // be reused; neither old carrier may be overwritten by the new graph.
          xpu_test::SameBytes(read(retained_logits->device_tensor),retained_logit_bytes);
          xpu_test::SameBytes(read(retained_tap->tensor),retained_tap_bytes);
        }
        if (step==2) {
          const auto logits=read(actual.device_tensor);
          if (!previous_accepted_logits.empty()) CHECK(logits!=previous_accepted_logits);
          previous_accepted_logits=logits;
        }
      }
    }
    CHECK(graph.captured());
    CHECK(graph.replay_count()>prior_replays+2);
    CHECK(vt::GetReferenceTierHits()==0);
  }
}
