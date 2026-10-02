// The tiny Nemotron Nano Omni checkpoint the NemotronH_Nano_VL_V2 gates load:
// the NemotronH hybrid language tower under `language_model.` (the same tiny
// schedule test_nemotron_h_paged_forward.cpp gates), a real-geometry
// `vit_small_patch16_224` RADIO tower, `mlp1`, and the tensors the loader must
// DEFER by name (the input conditioner, the video embedder, an audio tensor).
//
// Shared by test_nano_nemotron_vl_registry.cpp (the loader and runner gates)
// and test_serve_nano_nemotron_vl_mm.cpp (the same bytes written as a model
// directory and entered through `vllm serve` and `vllm_engine_load`).
#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include <unistd.h>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/tokenizer/bpe.h"
#include "vt/dtype.h"

namespace nnvl_tiny {

using vllm::NemotronHBlock;

// The tiny language tower (test_nemotron_h_paged_forward.cpp's).
constexpr int kHidden = 24;
constexpr int kVocab = 32;
constexpr int kAttnHeads = 4;
constexpr int kKvHeads = 2;
constexpr int kHeadDim = 6;
constexpr int kMambaHeads = 4;
constexpr int kMambaHeadDim = 6;
constexpr int kNGroups = 2;
constexpr int kStateSize = 8;
constexpr int kConvKernel = 4;
constexpr int kChunkSize = 8;
constexpr int kRoutedExperts = 8;
constexpr int kExpertsPerTok = 3;
constexpr int kMoeInter = 10;
constexpr int kSharedInter = 12;
constexpr int kMambaInter = kMambaHeads * kMambaHeadDim;
constexpr int kConvDim = kMambaInter + 2 * kNGroups * kStateSize;
constexpr int kInProjOut = kMambaInter + kConvDim + kMambaHeads;
constexpr int kQDim = kAttnHeads * kHeadDim;
constexpr int kKvDim = kKvHeads * kHeadDim;
constexpr int kBlockSize = 16;
constexpr int kNumBlocks = 16;
constexpr int kMaxModelLen = 128;

// The vision side: vit_small_patch16_224 (configs/radio.py:13), CPE table
// 4x4 (cpe_max_size 64), 2 teachers -> 2 CLS + 2 registers.
constexpr int kVitH = 384, kVitL = 12, kVitI = 1536, kPatch = 16;
constexpr int kPosGrid = 4, kSkip = 4;
constexpr int kProjHidden = 32;

// The special tokens, inside the tiny vocabulary.
constexpr int32_t kImgContext = 10, kImgStart = 11, kImgEnd = 12;

struct Fx {
  std::string name;
  std::string dtype;
  std::vector<int64_t> shape;
  std::string bytes;
};

inline std::string U64Le(uint64_t v) {
  std::string s(8, '\0');
  for (int i = 0; i < 8; ++i) s[static_cast<size_t>(i)] = static_cast<char>((v >> (8 * i)) & 0xff);
  return s;
}

inline int64_t NumEl(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

inline float Synth(uint32_t& r, float scale) {
  r = r * 1664525u + 1013904223u;
  const float u = static_cast<float>(r >> 8) / static_cast<float>(1u << 24);
  return (u - 0.5f) * scale;
}

inline std::string Bf16Bytes(size_t n, int seed, float scale, float offset = 0.0f) {
  std::string s(n * 2, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2654435761u + 1u;
  for (size_t i = 0; i < n; ++i) {
    const uint16_t bf = vt::F32ToBF16(offset + Synth(r, scale));
    s[i * 2] = static_cast<char>(bf & 0xff);
    s[i * 2 + 1] = static_cast<char>((bf >> 8) & 0xff);
  }
  return s;
}

inline std::string F32Bytes(size_t n, int seed, float scale) {
  std::string s(n * 4, '\0');
  uint32_t r = static_cast<uint32_t>(seed) * 2246822519u + 1u;
  for (size_t i = 0; i < n; ++i) {
    const float f = Synth(r, scale);
    std::memcpy(&s[i * 4], &f, 4);
  }
  return s;
}

inline Fx Bf16(const std::string& n, std::vector<int64_t> sh, int seed, float scale = 0.5f,
        float offset = 0.0f) {
  return {n, "BF16", sh, Bf16Bytes(static_cast<size_t>(NumEl(sh)), seed, scale, offset)};
}
inline Fx F32(const std::string& n, std::vector<int64_t> sh, int seed, float scale = 0.5f) {
  return {n, "F32", sh, F32Bytes(static_cast<size_t>(NumEl(sh)), seed, scale)};
}

inline std::string BuildSt(const std::vector<Fx>& ts) {
  nlohmann::json hdr = nlohmann::json::object();
  std::string data;
  for (const Fx& t : ts) {
    const size_t start = data.size();
    data += t.bytes;
    hdr[t.name] = {{"dtype", t.dtype}, {"shape", t.shape}, {"data_offsets", {start, data.size()}}};
  }
  const std::string header = hdr.dump();
  return U64Le(header.size()) + header + data;
}

class TempFile {
 public:
  explicit TempFile(const std::string& bytes, const char* ext = ".safetensors") {
    static int c = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("nano_nemotron_vl_" + std::to_string(::getpid()) + "_" + std::to_string(c++) + ext))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  }
  ~TempFile() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

inline const std::vector<NemotronHBlock>& Schedule() {
  static const std::vector<NemotronHBlock> s{NemotronHBlock::kMamba, NemotronHBlock::kMoe,
                                             NemotronHBlock::kAttention, NemotronHBlock::kMamba,
                                             NemotronHBlock::kMoe};
  return s;
}

inline std::vector<Fx> BuildTensors(bool with_stray = false) {
  std::vector<Fx> v;
  int s = 1;
  const std::string lm = "language_model.";
  v.push_back(Bf16(lm + "backbone.embeddings.weight", {kVocab, kHidden}, s++));
  v.push_back(Bf16(lm + "backbone.norm_f.weight", {kHidden}, s++, 0.8f));
  v.push_back(Bf16(lm + "lm_head.weight", {kVocab, kHidden}, s++, 0.25f));
  for (size_t l = 0; l < Schedule().size(); ++l) {
    const std::string p = lm + "backbone.layers." + std::to_string(l) + ".";
    const std::string m = p + "mixer.";
    v.push_back(Bf16(p + "norm.weight", {kHidden}, s++, 0.9f));
    switch (Schedule()[l]) {
      case NemotronHBlock::kMamba:
        v.push_back(Bf16(m + "in_proj.weight", {kInProjOut, kHidden}, s++, 0.3f));
        v.push_back(Bf16(m + "out_proj.weight", {kHidden, kMambaInter}, s++, 0.3f));
        v.push_back(Bf16(m + "conv1d.weight", {kConvDim, 1, kConvKernel}, s++, 0.4f));
        v.push_back(Bf16(m + "conv1d.bias", {kConvDim}, s++, 0.2f));
        v.push_back(F32(m + "A_log", {kMambaHeads}, s++, 0.6f));
        v.push_back(F32(m + "D", {kMambaHeads}, s++, 0.6f));
        v.push_back(F32(m + "dt_bias", {kMambaHeads}, s++, 0.3f));
        v.push_back(Bf16(m + "norm.weight", {kMambaInter}, s++, 0.7f));
        break;
      case NemotronHBlock::kAttention:
        v.push_back(Bf16(m + "q_proj.weight", {kQDim, kHidden}, s++, 1.5f));
        v.push_back(Bf16(m + "k_proj.weight", {kKvDim, kHidden}, s++, 1.5f));
        v.push_back(Bf16(m + "v_proj.weight", {kKvDim, kHidden}, s++, 1.5f));
        v.push_back(Bf16(m + "o_proj.weight", {kHidden, kQDim}, s++, 0.3f));
        break;
      case NemotronHBlock::kMoe:
        v.push_back(F32(m + "gate.weight", {kRoutedExperts, kHidden}, s++, 0.35f));
        v.push_back(F32(m + "gate.e_score_correction_bias", {kRoutedExperts}, s++, 0.4f));
        for (int e = 0; e < kRoutedExperts; ++e) {
          const std::string ex = m + "experts." + std::to_string(e) + ".";
          v.push_back(Bf16(ex + "up_proj.weight", {kMoeInter, kHidden}, s++, 0.3f));
          v.push_back(Bf16(ex + "down_proj.weight", {kHidden, kMoeInter}, s++, 0.3f));
        }
        v.push_back(Bf16(m + "shared_experts.up_proj.weight", {kSharedInter, kHidden}, s++, 0.3f));
        v.push_back(Bf16(m + "shared_experts.down_proj.weight", {kHidden, kSharedInter}, s++, 0.3f));
        break;
      case NemotronHBlock::kMlp:
        break;
    }
  }
  const std::string pg = "vision_model.radio_model.model.patch_generator.";
  v.push_back(Bf16(pg + "embedder.weight", {kVitH, 3 * kPatch * kPatch}, s++, 0.08f));
  v.push_back(Bf16(pg + "pos_embed", {1, kPosGrid * kPosGrid, kVitH}, s++, 0.5f));
  v.push_back(Bf16(pg + "cls_token.token", {kSkip, kVitH}, s++, 0.5f));
  v.push_back(Bf16(pg + "video_embedder.weight", {kVitH, 2 * 3 * kPatch * kPatch}, s++, 0.1f));
  v.push_back(F32("vision_model.radio_model.input_conditioner.norm_mean", {3, 1, 1}, s++));
  v.push_back(F32("vision_model.radio_model.input_conditioner.norm_std", {3, 1, 1}, s++));
  for (int l = 0; l < kVitL; ++l) {
    const std::string b = "vision_model.radio_model.model.blocks." + std::to_string(l) + ".";
    v.push_back(Bf16(b + "norm1.weight", {kVitH}, s++, 0.2f, 1.0f));
    v.push_back(Bf16(b + "norm1.bias", {kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "attn.qkv.weight", {3 * kVitH, kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "attn.qkv.bias", {3 * kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "attn.proj.weight", {kVitH, kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "attn.proj.bias", {kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "norm2.weight", {kVitH}, s++, 0.2f, 1.0f));
    v.push_back(Bf16(b + "norm2.bias", {kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "mlp.fc1.weight", {kVitI, kVitH}, s++, 0.1f));
    v.push_back(Bf16(b + "mlp.fc1.bias", {kVitI}, s++, 0.1f));
    v.push_back(Bf16(b + "mlp.fc2.weight", {kVitH, kVitI}, s++, 0.05f));
    v.push_back(Bf16(b + "mlp.fc2.bias", {kVitH}, s++, 0.1f));
  }
  v.push_back(Bf16("mlp1.0.weight", {4 * kVitH}, s++, 0.2f, 1.0f));
  v.push_back(Bf16("mlp1.1.weight", {kProjHidden, 4 * kVitH}, s++, 0.1f));
  // Large enough that the image rows move the residual stream decisively.
  v.push_back(Bf16("mlp1.3.weight", {kHidden, kProjHidden}, s++, 4.0f));
  v.push_back(Bf16("sound_projection.linear.weight", {kHidden, 8}, s++));
  if (with_stray) v.push_back(Bf16("mystery_head.weight", {4}, s++));
  return v;
}

inline nlohmann::json TinyConfig() {
  nlohmann::json llm;
  llm["architectures"] = nlohmann::json::array({"NemotronHForCausalLM"});
  llm["model_type"] = "nemotron_h";
  llm["dtype"] = "bfloat16";
  nlohmann::json blocks = nlohmann::json::array();
  for (NemotronHBlock b : Schedule()) {
    blocks.push_back(b == NemotronHBlock::kMamba ? "mamba"
                     : b == NemotronHBlock::kMoe ? "moe"
                                                 : "attention");
  }
  llm["layers_block_type"] = blocks;
  llm["num_hidden_layers"] = static_cast<int>(Schedule().size());
  llm["hidden_size"] = kHidden;
  llm["vocab_size"] = kVocab;
  llm["max_position_embeddings"] = kMaxModelLen;
  llm["layer_norm_epsilon"] = 1e-5;
  llm["tie_word_embeddings"] = false;
  llm["num_attention_heads"] = kAttnHeads;
  llm["num_key_value_heads"] = kKvHeads;
  llm["head_dim"] = kHeadDim;
  llm["attention_bias"] = false;
  llm["mamba_num_heads"] = kMambaHeads;
  llm["mamba_head_dim"] = kMambaHeadDim;
  llm["n_groups"] = kNGroups;
  llm["ssm_state_size"] = kStateSize;
  llm["conv_kernel"] = kConvKernel;
  llm["chunk_size"] = kChunkSize;
  llm["expand"] = 2;
  llm["mamba_hidden_act"] = "silu";
  llm["mamba_ssm_cache_dtype"] = "float32";
  llm["use_conv_bias"] = true;
  llm["mamba_proj_bias"] = false;
  llm["n_routed_experts"] = kRoutedExperts;
  llm["num_experts_per_tok"] = kExpertsPerTok;
  llm["moe_intermediate_size"] = kMoeInter;
  llm["n_shared_experts"] = 1;
  llm["moe_shared_expert_intermediate_size"] = kSharedInter;
  llm["n_group"] = 1;
  llm["topk_group"] = 1;
  llm["routed_scaling_factor"] = 2.5;
  llm["norm_topk_prob"] = true;
  llm["mlp_hidden_act"] = "relu2";
  llm["mlp_bias"] = false;

  nlohmann::json vision;
  vision["patch_size"] = kPatch;
  vision["preferred_resolution"] = {32, 32};
  vision["video_temporal_patch_size"] = 2;
  vision["args"] = {{"model", "vit_small_patch16_224"},
                    {"min_num_patches", 4},
                    {"max_num_patches", 16},
                    {"register_multiple", 4},
                    {"cpe_max_size", 64},
                    {"cls_token_per_teacher", true},
                    {"teachers", nlohmann::json::array({{{"name", "a"}}, {{"name", "b"}}})}};

  nlohmann::json j;
  j["architectures"] = nlohmann::json::array({"NemotronH_Nano_Omni_Reasoning_V3"});
  j["model_type"] = "NemotronH_Nano_Omni_Reasoning_V3";
  j["torch_dtype"] = "bfloat16";
  j["downsample_ratio"] = 0.5;
  j["ps_version"] = "v2";
  j["patch_size"] = kPatch;
  j["vit_hidden_size"] = kVitH;
  j["projector_hidden_size"] = kProjHidden;
  j["norm_mean"] = {0.48145466, 0.4578275, 0.40821073};
  j["norm_std"] = {0.26862954, 0.26130258, 0.27577711};
  j["llm_config"] = llm;
  j["vision_config"] = vision;
  j["sound_config"] = nlohmann::json::object();
  return j;
}

// A byte-level BPE tokenizer.json whose every id lies inside the tiny
// vocabulary, so a prompt it encodes fits the embedding table and every token
// the tiny model samples decodes. The three image tokens sit at the ids the
// processor gates use; the other 29 ids are one byte each, enough for the
// prompts the serving gates send.
inline std::string TinyTokenizerJson() {
  nlohmann::ordered_json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = nlohmann::ordered_json::array({
      {{"id", kImgContext}, {"content", "<image>"}, {"special", true}},
      {{"id", kImgStart}, {"content", "<img>"}, {"special", true}},
      {{"id", kImgEnd}, {"content", "</img>"}, {"special", true}},
  });
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {{"type", "ByteLevel"},
                          {"add_prefix_space", false},
                          {"trim_offsets", false},
                          {"use_regex", true}};
  std::string bytes = "\n .";
  for (char c = 'a'; c <= 'z'; ++c) bytes.push_back(c);
  nlohmann::ordered_json vocab = nlohmann::ordered_json::object();
  int32_t id = 0;
  for (const char b : bytes) {
    if (id == kImgContext) id = kImgEnd + 1;
    vocab[vllm::tok::MapBytesToUnicode(std::string(1, b))] = id++;
  }
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", nlohmann::ordered_json::array()}};
  return doc.dump();
}

// The tiny checkpoint as a model DIRECTORY, the shape `vllm serve --model` and
// `vllm_engine_load` take: config.json, one safetensors shard, the tokenizer,
// and a tokenizer_config.json whose chat template emits each message's content
// verbatim (the seam hands the template the released image prefix already
// rendered, so this is enough to carry `<image>` into the prompt).
inline void WriteModelDir(const std::filesystem::path& dir) {
  std::filesystem::create_directories(dir);
  auto write = [&](const char* name, const std::string& bytes) {
    std::ofstream out(dir / name, std::ios::binary);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
  };
  write("config.json", TinyConfig().dump(2));
  write("model.safetensors", BuildSt(BuildTensors()));
  write("tokenizer.json", TinyTokenizerJson());
  nlohmann::json tc;
  tc["chat_template"] = "{% for message in messages %}{{ message['content'] }}{% endfor %}";
  write("tokenizer_config.json", tc.dump(2));
}

}  // namespace nnvl_tiny
