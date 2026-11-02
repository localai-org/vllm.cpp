// Deterministic public-loader fixture for BACKEND-ROCM-QUANT-GATHER (#3093).
// The GGUF metadata follows the pinned llama.cpp qwen35 loader. This fixture
// is a synthetic test vehicle, not a qualification of a published checkpoint.
#pragma once

#include <cmath>
#include <cstdint>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/gguf_builder.h"
#include "vt/dtype.h"
#include "vt/quant.h"

namespace rocm_gather_test {

inline constexpr uint32_t kSeed = 0x524f434d;
inline constexpr int64_t kHidden = 256;
inline constexpr int64_t kVocab = 128;

struct Format {
  vt::DType dtype;
  uint32_t ggml;
  const char* name;
};

inline constexpr Format kFormats[] = {
    {vt::DType::kQ4_0, 2, "Q4_0"},
    {vt::DType::kQ5_0, 6, "Q5_0"},
    {vt::DType::kQ8_0, 8, "Q8_0"},
    {vt::DType::kQ2_K, 10, "Q2_K"},
    {vt::DType::kQ3_K, 11, "Q3_K"},
    {vt::DType::kQ4_K, 12, "Q4_K"},
    {vt::DType::kQ5_K, 13, "Q5_K"},
    {vt::DType::kQ6_K, 14, "Q6_K"},
    {vt::DType::kQ8_K, 15, "Q8_K"},
    {vt::DType::kIQ2_XXS, 16, "IQ2_XXS"},
    {vt::DType::kIQ2_XS, 17, "IQ2_XS"},
    {vt::DType::kIQ3_XXS, 18, "IQ3_XXS"},
    {vt::DType::kIQ1_S, 19, "IQ1_S"},
    {vt::DType::kIQ4_NL, 20, "IQ4_NL"},
    {vt::DType::kIQ3_S, 21, "IQ3_S"},
    {vt::DType::kIQ2_S, 22, "IQ2_S"},
    {vt::DType::kIQ4_XS, 23, "IQ4_XS"},
    {vt::DType::kMXFP4, 39, "MXFP4"},
    {vt::DType::kIQ1_XXXS, 66, "IQ1_XXXS"},
};

inline void PutHalf(std::string& b, size_t offset, uint16_t bits) {
  b[offset] = static_cast<char>(bits & 255);
  b[offset + 1] = static_cast<char>(bits >> 8);
}

// Every block consumes exactly BlockBytes outputs of mt19937. Its raw payload
// uses each output's low byte. Scale fields are then set to finite powers of
// two, independently of the decoded values. Projection RNG is separate, so
// changing the embedding format cannot change a dense projection.
inline std::string PackedTable(const Format& format, int64_t rows = kVocab,
                               int64_t width = kHidden) {
  const auto dtype = format.dtype;
  const int64_t block_bytes = vt::BlockBytes(dtype);
  const int64_t blocks = rows * width / vt::BlockElems(dtype);
  std::string packed(static_cast<size_t>(blocks * block_bytes), '\0');
  std::mt19937 rng(kSeed);
  for (int64_t block = 0; block < blocks; ++block) {
    const size_t base = static_cast<size_t>(block * block_bytes);
    for (int64_t byte = 0; byte < block_bytes; ++byte) {
      packed[base + static_cast<size_t>(byte)] = static_cast<char>(rng() & 255);
    }
    switch (dtype) {
      case vt::DType::kQ2_K:
        PutHalf(packed, base + 80, 0x1800);  // d = 1/512
        PutHalf(packed, base + 82, 0x1800);
        break;
      case vt::DType::kQ3_K:
        PutHalf(packed, base + 108, 0x1800);
        break;
      case vt::DType::kQ4_K:
      case vt::DType::kQ5_K:
        PutHalf(packed, base, 0x1400);  // d = 1/1024
        PutHalf(packed, base + 2, 0x1400);
        break;
      case vt::DType::kQ6_K:
        PutHalf(packed, base + 208, 0x1000);  // d = 1/2048
        break;
      case vt::DType::kQ8_K: {
        const std::string scale = gguf_test::U32Le(0x3b000000);  // 1/512
        packed.replace(base, 4, scale);
        break;
      }
      case vt::DType::kMXFP4:
        packed[base] = static_cast<char>(119);  // e8m0 scale = 1/256
        break;
      default:
        PutHalf(packed, base, 0x2000);  // d = 1/128
        break;
    }
  }
  return packed;
}

inline std::string DenseEmbedding(const Format& format,
                                  const std::string& packed) {
  std::vector<float> decoded(static_cast<size_t>(kVocab * kHidden));
  const auto decode = vt::cpu::BlockToFloat(format.dtype);
  if (decode == nullptr) throw std::runtime_error("fixture decoder absent");
  decode(packed.data(), decoded.data(), kVocab * kHidden);
  std::string dense;
  dense.reserve(decoded.size() * 2);
  for (float value : decoded) {
    if (!std::isfinite(value)) throw std::runtime_error("nonfinite fixture block");
    const uint16_t word = vt::F32ToBF16(value);
    dense.push_back(static_cast<char>(word & 255));
    dense.push_back(static_cast<char>(word >> 8));
  }
  return dense;
}

inline std::string BuildModel(const Format& format, bool dense_embedding,
                              const Format* gate_up_format = nullptr,
                              const Format* down_format = nullptr,
                              uint32_t head_dim = 64) {
  if (head_dim == 0 || 256 % head_dim != 0)
    throw std::invalid_argument("fixture head dimension must divide 256");
  using namespace gguf_test;
  GgufModelBuilder builder;
  builder.AddKv(StrKv("general.architecture", "qwen35"));
  builder.AddKv(StrKv("general.name", "rocm-quant-gather-synthetic"));
  builder.AddKv(U32Kv("general.quantization_version", 2));
  builder.AddKv(U32Kv("qwen35.embedding_length", 256));
  builder.AddKv(U32Kv("qwen35.block_count", 1));
  builder.AddKv(U32Kv("qwen35.attention.head_count", 256 / head_dim));
  builder.AddKv(U32Kv("qwen35.attention.head_count_kv", 1));
  builder.AddKv(U32Kv("qwen35.attention.key_length", head_dim));
  builder.AddKv(U32Kv("qwen35.attention.value_length", head_dim));
  builder.AddKv(U32Kv("qwen35.feed_forward_length", 256));
  builder.AddKv(U32Kv("qwen35.full_attention_interval", 1));
  builder.AddKv(U32Kv("qwen35.context_length", 64));
  builder.AddKv(U32Kv("qwen35.vocab_size", 128));
  builder.AddKv(F32Kv("qwen35.attention.layer_norm_rms_epsilon", 1e-6F));
  builder.AddKv(F32Kv("qwen35.rope.freq_base", 1000000.0F));
  builder.AddKv(U32Kv("qwen35.rope.dimension_count", head_dim));
  // Stock qwen35.cpp::load_arch_hparams requires these even when every layer
  // is full attention. The SSM geometry stays inactive: no SSM tensor exists.
  builder.AddKv(I32ArrayKv("qwen35.rope.dimension_sections",
      {static_cast<int32_t>(head_dim / 4), static_cast<int32_t>(head_dim / 8),
       static_cast<int32_t>(head_dim / 8), 0}));
  builder.AddKv(U32Kv("qwen35.ssm.conv_kernel", 4));
  builder.AddKv(U32Kv("qwen35.ssm.inner_size", 64));
  builder.AddKv(U32Kv("qwen35.ssm.state_size", 64));
  builder.AddKv(U32Kv("qwen35.ssm.time_step_rank", 1));
  builder.AddKv(U32Kv("qwen35.ssm.group_count", 1));
  builder.AddKv(StrKv("tokenizer.ggml.model", "gpt2"));
  builder.AddKv(StrKv("tokenizer.ggml.pre", "qwen2"));
  builder.AddKv(U32Kv("tokenizer.ggml.unknown_token_id", 0));
  builder.AddKv(U32Kv("tokenizer.ggml.bos_token_id", 1));
  builder.AddKv(U32Kv("tokenizer.ggml.eos_token_id", 2));
  builder.AddKv(BoolKv("tokenizer.ggml.add_bos_token", false));
  std::vector<std::string> vocabulary = {"<unk>", "<s>", "</s>"};
  for (int id = 3; id < kVocab; ++id) vocabulary.push_back("token" + std::to_string(id));
  builder.AddKv(StrArrayKv("tokenizer.ggml.tokens", vocabulary));
  builder.AddKv(StrArrayKv("tokenizer.ggml.merges", {}));
  std::vector<int32_t> types(128, 1);
  types[0] = 2;
  types[1] = types[2] = 3;
  builder.AddKv(I32ArrayKv("tokenizer.ggml.token_type", types));

  const std::string packed = PackedTable(format);
  builder.AddTensor("token_embd.weight", {256, 128},
                    dense_embedding ? 30 : format.ggml,
                    dense_embedding ? DenseEmbedding(format, packed) : packed);
  std::mt19937 rng(kSeed);
  const auto projection = [&](const char* name, uint64_t input, uint64_t output,
                              const Format* quant = nullptr) {
    if (quant != nullptr) {
      builder.AddTensor(name, {input, output}, quant->ggml,
                        PackedTable(*quant, static_cast<int64_t>(output),
                                    static_cast<int64_t>(input)));
      return;
    }
    std::string bytes;
    bytes.reserve(static_cast<size_t>(input * output * 2));
    for (uint64_t i = 0; i < input * output; ++i) {
      const int32_t integer = static_cast<int32_t>(rng() & 65535) - 32768;
      const float value = static_cast<float>(integer) / 2097152.0F;
      const uint16_t word = vt::F32ToBF16(value);
      bytes.push_back(static_cast<char>(word & 255));
      bytes.push_back(static_cast<char>(word >> 8));
    }
    builder.AddTensor(name, {input, output}, 30, bytes);
  };
  const auto norm = [&](const char* name, uint64_t width) {
    std::string bytes;
    for (uint64_t i = 0; i < width; ++i) bytes += U32Le(0x3f800000);
    builder.AddTensor(name, {width}, 0, bytes);
  };
  projection("output.weight", 256, 128);
  projection("blk.0.attn_q.weight", 256, 512);
  projection("blk.0.attn_k.weight", 256, head_dim);
  projection("blk.0.attn_v.weight", 256, head_dim);
  projection("blk.0.attn_output.weight", 256, 256);
  projection("blk.0.ffn_gate.weight", 256, 256, gate_up_format);
  projection("blk.0.ffn_up.weight", 256, 256, gate_up_format);
  projection("blk.0.ffn_down.weight", 256, 256, down_format);
  norm("output_norm.weight", 256);
  norm("blk.0.attn_norm.weight", 256);
  norm("blk.0.post_attention_norm.weight", 256);
  norm("blk.0.attn_q_norm.weight", head_dim);
  norm("blk.0.attn_k_norm.weight", head_dim);
  return builder.Build();
}

}  // namespace rocm_gather_test
