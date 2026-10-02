// A small synthetic Qwen3.5 dense model (GDN hybrid) and a byte-level
// tokenizer carrying the Qwen chat tokens, shared by the candidate-letter
// decision tests (test_nimble, test_tev1_systemone). Extracted from
// test_nimble.cpp unchanged, with the architecture made a parameter.
#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <string_view>
#include <vector>

#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/device.h"
#include "vt/dtype.h"

namespace qwen3_5_decision_fixture {

using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseWeights;
using vllm::tok::Tokenizer;
using vt::DType;

// ── A byte-level tokenizer with the four Qwen chat tokens ──────────────────
// 256 byte tokens (id = byte value) and no merges, so every character is one
// token and "A" is id 65. The chat tokens are added tokens, as in Qwen3.5.

inline std::string JsonEscape(std::string_view s) {
  std::string out;
  for (unsigned char c : s) {
    if (c == '\\') {
      out += "\\\\";
    } else if (c == '"') {
      out += "\\\"";
    } else if (c < 0x20) {
      char buf[8];
      std::snprintf(buf, sizeof(buf), "\\u%04x", c);
      out += buf;
    } else {
      out += static_cast<char>(c);
    }
  }
  return out;
}

inline std::string CodepointToUtf8(uint32_t cp) {
  std::string out;
  if (cp < 0x80) {
    out += static_cast<char>(cp);
  } else if (cp < 0x800) {
    out += static_cast<char>(0xC0 | (cp >> 6));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  } else {
    out += static_cast<char>(0xE0 | (cp >> 12));
    out += static_cast<char>(0x80 | ((cp >> 6) & 0x3F));
    out += static_cast<char>(0x80 | (cp & 0x3F));
  }
  return out;
}

// The tokenizer.json text of ByteTokenizer(), for a test that writes a model
// directory the loader reads (test_clm).
inline const std::string& ByteTokenizerJson() {
  static const std::string json_text = [] {
    std::string vocab;
    for (int b = 0; b < 256; ++b) {
      if (b > 0) vocab += ',';
      vocab += "\"" +
               JsonEscape(CodepointToUtf8(
                   vllm::tok::ByteToUnicode(static_cast<uint8_t>(b)))) +
               "\":" + std::to_string(b);
    }
    const char* added[4][2] = {{"<|im_start|>", "true"},
                               {"<|im_end|>", "true"},
                               {"<think>", "false"},
                               {"</think>", "false"}};
    std::string added_json;
    for (int i = 0; i < 4; ++i) {
      vocab += ",\"" + std::string(added[i][0]) + "\":" + std::to_string(256 + i);
      if (i > 0) added_json += ',';
      added_json += "{\"id\":" + std::to_string(256 + i) + ",\"content\":\"" +
                    added[i][0] +
                    "\",\"single_word\":false,\"lstrip\":false,\"rstrip\":false,"
                    "\"normalized\":false,\"special\":" +
                    added[i][1] + "}";
    }
    std::string json = R"({"version":"1.0","truncation":null,"padding":null,)";
    json += "\"added_tokens\":[" + added_json + "],";
    json += R"("normalizer":null,)";
    // The Qwen split regex the loader requires; with no merges it does not
    // change the ids, every byte is still one token.
    const std::string regex =
        R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|)"
        R"([^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}|)"
        R"( ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|)"
        R"(\s*[\r\n]+|\s+(?!\S)|\s+)";
    json += R"("pre_tokenizer":{"type":"Sequence","pretokenizers":[)";
    json += "{\"type\":\"Split\",\"pattern\":{\"Regex\":\"" + JsonEscape(regex) +
            "\"},\"behavior\":\"Isolated\",\"invert\":false},";
    json += R"({"type":"ByteLevel","add_prefix_space":false,)"
            R"("trim_offsets":false,"use_regex":false}]},)";
    json += R"("post_processor":{"type":"ByteLevel","add_prefix_space":false,)"
            R"("trim_offsets":false,"use_regex":false},)";
    json += R"("decoder":{"type":"ByteLevel","add_prefix_space":false,)"
            R"("trim_offsets":false,"use_regex":false},)";
    json += R"("model":{"type":"BPE","dropout":null,"unk_token":null,)"
            R"("continuing_subword_prefix":null,"end_of_word_suffix":null,)"
            R"("fuse_unk":false,"byte_fallback":false,"ignore_merges":false,)";
    json += "\"vocab\":{" + vocab + "},\"merges\":[]}}";
    return json;
  }();
  return json_text;
}

inline const Tokenizer& ByteTokenizer() {
  static const Tokenizer tok =
      Tokenizer::FromHfJsonBytes(ByteTokenizerJson(), "qwen3_5_decision_fixture");
  return tok;
}

// ── A small synthetic Qwen3.5 dense model (GDN hybrid) ─────────────────────
// Same generator and shapes as test_qwen3_5_dense_vision.cpp, with the vocab
// widened to the byte tokenizer's 260 ids.

inline uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}

inline float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

inline OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

inline HfConfig MakeConfig(const std::string& architecture) {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {architecture};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
  c.vocab_size = 260;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 4096;
  c.rope_parameters.mrope_interleaved = true;
  c.rope_parameters.mrope_section = {1, 1, 0};
  return c;
}

inline Qwen3_5DenseWeights MakeWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size, I = c.intermediate_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.mlp.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 500);
    lw.mlp.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 600);
    lw.mlp.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 700);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

inline vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

}  // namespace qwen3_5_decision_fixture
