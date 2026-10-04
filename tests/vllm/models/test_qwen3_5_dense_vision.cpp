// Dense-arm vision tower (MODEL-QWEN35-DENSE-VL-EXL3,
// ISSUE-LOCAL-01M3AHX9DQX8HNE32G80C9VGMJ) — the CPU gates for the two things
// that were missing on `Qwen3_5ForConditionalGeneration`: the checkpoint's
// `model.visual.*` tensors are no longer silently dropped at load, and the
// dense VL greedy driver actually CONSUMES the tower rows.
//
// What these gates establish, and what they deliberately do NOT:
//
//   * LOADER — the EXL3 27B checkpoint (`Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw`,
//     pin 19441ac87) ships 333 bf16 `model.visual.*` tensors that
//     `LoadQwen3_5Dense` never read: it loaded a text-only model and answered
//     nothing about the tower. `LoadQwen3_5DenseVision` reads them through the
//     SHARED `LoadQwen3VLVisionWeights` reader (`qwen3_vl.h`) — the SAME tower
//     the MoE arm reuses (`LoadQwen3_5MoeVision`, #891) and the 27B dense arm
//     is already token-gated on at image 32/32 + video 32/32 — and REFUSES BY
//     NAME when a checkpoint carries none. Gated here on synthetic shards (a
//     real 27-block tower is ~1.6 GB f32).
//
//   * LOADER, NON-SILENT — the gate the row exists for: loading a
//     vision-inclusive checkpoint through `LoadQwen3_5Dense` must no longer
//     succeed quietly as a text-only model. The loader now reaches for the
//     tower, so an INCOMPLETE tower is a loud, named failure instead of a
//     silent drop. (The real 333-tensor load is gated on hardware against the
//     artifact; see the row spec's phase 3.)
//
//   * FORWARD INVOCATION — a green suite cannot see "the tower loads but is
//     never invoked" unless something asserts the merger rows CHANGE the
//     output. Two gates do, mirroring test_qwen3_5_moe_vision.cpp:
//       (a) the merger row for a substituted token must reproduce a plain TEXT
//           greedy run over the substituted prompt, token for token (a tower
//           that is never invoked leaves the placeholder's own embedding and
//           reproduces the placeholder run instead);
//       (b) on a NON-degenerate grid the VL run must DIFFER from the 1-D run,
//           which is what sees the MRoPE cache.
//     Both run the real `Qwen3_5VLGenerateGreedy` over a small synthetic dense
//     GDN-hybrid model on CPU — the same driver the 27B runs.
//
//   * NOT ESTABLISHED HERE: token-EXACTNESS against the pinned oracle, and the
//     real EXL3 27B tower load. Both are device legs the operator gates after
//     review (row spec phases 3-5).
#include <doctest/doctest.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/backend.h"
#include "vt/dtype.h"

using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::Qwen3_5DenseWeights;
using vllm::SafetensorsFile;
using vt::DType;

namespace {

// splitmix64-based small deterministic values in [-0.08, 0.08) — the same
// generator test_qwen3_5_moe_vision.cpp uses, so the synthetic model is the
// one the MoE vision gate already exercises.
uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u = static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
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

// A small GDN-hybrid DENSE config with the 27B's MRoPE shape: 3 mrope
// sections, interleaved, summing to rotary_dim/2 (the 27B is [11,11,10] over
// rotary_dim 64; this is [1,1,0] over rotary_dim 4). Same shapes the MoE
// vision gate's synthetic model uses; the MoE block becomes a dense SwiGLU MLP.
HfConfig MakeConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
  c.vocab_size = 40;
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
  c.max_position_embeddings = 64;
  c.rope_parameters.mrope_interleaved = true;
  c.rope_parameters.mrope_section = {1, 1, 0};
  return c;
}

// The ROPE-SENSITIVE variant: 3 of the 4 layers full_attention and the whole
// head dim rotating — the MRoPE gate's property under test IS the rope angle.
HfConfig MakeConfigAllAttn() {
  HfConfig c = MakeConfig();
  c.layer_types = {"linear_attention", "full_attention", "full_attention",
                   "full_attention"};
  c.rotary_dim = 8;                          // == head_dim
  c.rope_parameters.mrope_section = {2, 1, 1};  // sums to rotary_dim/2
  return c;
}

Qwen3_5DenseWeights MakeWeights(const HfConfig& c) {
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
      // The 27B gates its attention output (`attn_output_gate: true`), so the
      // query projection is [H, 2 * Hq * Dh] — the same shape the MoE
      // synthetic model uses for the same reason.
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

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// Tower merger output stand-in: [N, H] host f32, the shape/dtype
// `Qwen3VLVisionForward` returns. Row `r` is the model's OWN embedding row for
// a chosen token id — the exact reduction the MoE vision gate leans on (see
// that file for the full argument): on a 1x1x1 LLM grid the masked scatter
// reproduces `embed(prompt with the visual token replaced by k)` and the MRoPE
// positions degenerate to plain 1-D, so the VL run must equal the text run
// over the substituted prompt, token for token.
std::vector<float> EmbeddingRowF32(const Qwen3_5DenseWeights& w, int64_t token_id,
                                   int64_t h) {
  const auto* p = reinterpret_cast<const uint16_t*>(w.embed_tokens.bytes.data());
  std::vector<float> row(static_cast<size_t>(h));
  for (int64_t j = 0; j < h; ++j)
    row[static_cast<size_t>(j)] =
        vt::BF16ToF32(p[static_cast<size_t>(token_id * h + j)]);
  return row;
}

// Plain TEXT greedy reference through `Qwen3_5DenseModel::Forward` over the
// SAME cache shapes/dtypes the VL core allocates (bf16 paged KV, f32 GDN ssm
// state, bf16 GDN conv state, one block, one sequence), 1-D sequential
// positions. This is the "no multimodal input at all" path.
std::vector<int32_t> TextGreedy(const std::vector<int32_t>& prompt_ids,
                                const Qwen3_5DenseWeights& w, const HfConfig& c,
                                vt::Queue& q, int max_new_tokens) {
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());
  const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads;
  const int64_t Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim;
  const int64_t Kw = c.linear_conv_kernel_dim;
  const int64_t conv_dim = 2 * Hk * Dk + Hv * Dv, conv_len = Kw - 1;
  const int64_t block_size = T0 + max_new_tokens + 8;
  const vt::Device dev{vt::DeviceType::kCPU, 0};

  std::vector<std::vector<uint16_t>> kv_buf;
  std::vector<std::vector<float>> ssm_buf;
  std::vector<std::vector<uint16_t>> conv_buf;
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    if (c.layer_types[static_cast<size_t>(l)] == "linear_attention") {
      ssm_buf.emplace_back(static_cast<size_t>(Hv * Dv * Dk), 0.0F);
      conv_buf.emplace_back(static_cast<size_t>(conv_dim * conv_len), 0);
    } else {
      kv_buf.emplace_back(static_cast<size_t>(2 * block_size * Hkv * Dh), 0);
    }
  }
  std::vector<vllm::PagedKvCache> attn_kv;
  std::vector<vllm::GdnStateCache> gdn_state;
  for (auto& b : kv_buf) {
    vllm::PagedKvCache kv;
    kv.data = b.data();
    kv.dtype = DType::kBF16;
    kv.num_blocks = 1;
    kv.block_size = block_size;
    kv.num_kv_heads = Hkv;
    kv.head_size = Dh;
    attn_kv.push_back(kv);
  }
  for (size_t g = 0; g < ssm_buf.size(); ++g) {
    vllm::GdnStateCache gs;
    gs.ssm_state = vt::Tensor::Contiguous(ssm_buf[g].data(), DType::kF32, dev,
                                          {1, Hv, Dv, Dk});
    gs.conv_state = vt::Tensor::Contiguous(conv_buf[g].data(), DType::kBF16, dev,
                                           {1, conv_dim, conv_len});
    gdn_state.push_back(gs);
  }

  auto attn_meta = [&](int64_t qlen, int64_t context) {
    vllm::v1::CommonAttentionMetadata m;
    m.num_reqs = 1;
    m.num_actual_tokens = static_cast<int>(qlen);
    m.query_start_loc = {0, static_cast<int32_t>(qlen)};
    m.query_start_loc_cpu = m.query_start_loc;
    m.seq_lens = {static_cast<int32_t>(context + qlen)};
    m.seq_lens_cpu = m.seq_lens;
    m.max_query_len = static_cast<int>(qlen);
    m.max_seq_len = static_cast<int>(context + qlen);
    m.block_table_num_cols = 1;
    m.block_table_tensor = {0};
    for (int64_t t = 0; t < qlen; ++t) m.slot_mapping.push_back(context + t);
    m.causal = true;
    return m;
  };

  std::vector<int32_t> generated;
  const int64_t vocab = c.vocab_size;
  auto argmax_last = [&](const std::vector<float>& logits) {
    const size_t rows = logits.size() / static_cast<size_t>(vocab);
    const float* row = logits.data() + (rows - 1) * static_cast<size_t>(vocab);
    int32_t best = 0;
    for (int64_t v = 1; v < vocab; ++v)
      if (row[v] > row[best]) best = static_cast<int32_t>(v);
    return best;
  };

  {
    std::vector<int32_t> pos(static_cast<size_t>(T0));
    for (int64_t t = 0; t < T0; ++t) pos[static_cast<size_t>(t)] =
        static_cast<int32_t>(t);
    vllm::v1::GDNAttentionMetadata g;
    g.num_prefills = 1;
    g.num_prefill_tokens = static_cast<int>(T0);
    g.num_decodes = 0;
    g.num_decode_tokens = 0;
    g.num_actual_tokens = static_cast<int>(T0);
    g.has_initial_state = std::vector<uint8_t>{0};
    g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    g.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T0)};
    g.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T0)};
    g.prefill_state_indices = std::vector<int32_t>{0};
    g.prefill_has_initial_state = std::vector<uint8_t>{0};
    const vllm::v1::CausalConv1dMetadata conv =
        vllm::v1::ComputeCausalConv1dMetadata(*g.non_spec_query_start_loc);
    g.batch_ptr = conv.batch_ptr;
    g.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    const std::vector<float> logits = vllm::Qwen3_5DenseModel::Forward(
        prompt_ids, pos, attn_meta(T0, 0), g, attn_kv, gdn_state, w, c, q,
        {static_cast<int32_t>(T0 - 1)});
    generated.push_back(argmax_last(logits));
  }
  for (int step = 1; step < max_new_tokens; ++step) {
    const int64_t abs_idx = T0 + (step - 1);
    vllm::v1::GDNAttentionMetadata g;
    g.num_prefills = 0;
    g.num_prefill_tokens = 0;
    g.num_decodes = 1;
    g.num_decode_tokens = 1;
    g.num_actual_tokens = 1;
    g.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    g.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
    const std::vector<int32_t> one = {generated.back()};
    const std::vector<int32_t> pos = {static_cast<int32_t>(abs_idx)};
    const std::vector<float> logits = vllm::Qwen3_5DenseModel::Forward(
        one, pos, attn_meta(1, abs_idx), g, attn_kv, gdn_state, w, c, q, {});
    generated.push_back(argmax_last(logits));
  }
  return generated;
}

// ── Synthetic safetensors shard (loader gates) ───────────────────────────────
// Minimal well-formed file: 8-byte little-endian header length + JSON header +
// the data blob. One f32 scalar per named tensor is enough — the refusal gate
// fires on tensor NAMES, before a single weight byte is read.
std::string WriteShard(const std::string& path,
                       const std::vector<std::string>& names) {
  std::string json = "{";
  for (size_t i = 0; i < names.size(); ++i) {
    if (i != 0) json += ",";
    json += "\"" + names[i] + "\":{\"dtype\":\"F32\",\"shape\":[1],"
            "\"data_offsets\":[" + std::to_string(i * 4) + "," +
            std::to_string((i + 1) * 4) + "]}";
  }
  json += "}";
  while (json.size() % 8 != 0) json += " ";
  std::string out(8, '\0');
  const uint64_t n = json.size();
  for (int b = 0; b < 8; ++b) out[static_cast<size_t>(b)] =
      static_cast<char>((n >> (8 * b)) & 0xFF);
  out += json;
  out.append(names.size() * 4, '\0');
  FILE* f = std::fopen(path.c_str(), "wb");
  REQUIRE(f != nullptr);
  std::fwrite(out.data(), 1, out.size(), f);
  std::fclose(f);
  return path;
}

std::string TmpDir() {
  const char* t = std::getenv("TMPDIR");
  return std::string(t != nullptr ? t : "/tmp");
}

}  // namespace

// ── LOADER: the tower is no longer dropped, and its ABSENCE is a refusal ─────

TEST_CASE("qwen3_5_dense_vision_config_mirrors_the_checkpoint_vision_config") {
  HfConfig c = MakeConfig();
  c.hidden_size = 5120;  // the EXL3 27B dense text hidden
  const vllm::multimodal::Qwen3VLVisionConfig v = vllm::Qwen3_5DenseVisionConfig(c);
  // config.json::vision_config of the artifact — the SAME tower the MoE arm
  // composes; only `out_hidden_size` differs, and it is `config.hidden_size`.
  CHECK(v.depth == 27);
  CHECK(v.hidden_size == 1152);
  CHECK(v.num_heads == 16);
  CHECK(v.intermediate_size == 4304);
  CHECK(v.patch_size == 16);
  CHECK(v.temporal_patch_size == 2);
  CHECK(v.spatial_merge_size == 2);
  CHECK(v.num_position_embeddings == 2304);
  CHECK(v.in_channels == 3);
  // deepstack_visual_indexes: [] — the deepstack path is compiled out upstream
  // for this family (qwen3_vl.py:1709-1716).
  CHECK(v.deepstack_visual_indexes.empty());
  // The merger writes into the text residual stream, so the tower's output
  // width IS the text hidden size (5120 on the dense 27B).
  CHECK(v.out_hidden_size == 5120);
}

// ISSUE-LOCAL-01M3RT4GVEY4QBE5AYBT8RDM89: the geometry is the CHECKPOINT's,
// not the 27B's. Upstream builds Qwen3_VisionTransformer from
// config.vision_config (qwen3_vl.py:536-628 @ 5559679229), so a smaller tower
// must load as itself. Before the fix every field below read the 27B constant
// and Tev1-0.8B aborted at model.visual.blocks.12.
TEST_CASE("qwen3_5_dense_vision_config_reads_a_smaller_checkpoint_tower") {
  HfConfig c = MakeConfig();
  c.hidden_size = 1024;
  // config.json::vision_config of togethercomputer/Tev1-0.8B-experimental
  // @ 6bb2dff1 (== Qwen/Qwen3.5-0.8B).
  c.raw["vision_config"] = {
      {"depth", 12},          {"hidden_size", 768},
      {"num_heads", 12},      {"intermediate_size", 3072},
      {"out_hidden_size", 1024}, {"patch_size", 16},
      {"temporal_patch_size", 2}, {"spatial_merge_size", 2},
      {"num_position_embeddings", 2304}, {"in_channels", 3},
      {"deepstack_visual_indexes", nlohmann::json::array()}};
  const vllm::multimodal::Qwen3VLVisionConfig v = vllm::Qwen3_5DenseVisionConfig(c);
  CHECK(v.depth == 12);
  CHECK(v.hidden_size == 768);
  CHECK(v.num_heads == 12);
  CHECK(v.head_dim() == 64);
  CHECK(v.intermediate_size == 3072);
  CHECK(v.out_hidden_size == 1024);
  CHECK(v.deepstack_visual_indexes.empty());

  // A key the checkpoint omits keeps the family value, not a zero.
  HfConfig partial = MakeConfig();
  partial.hidden_size = 2560;
  partial.raw["vision_config"] = {{"depth", 24}, {"hidden_size", 1024},
                                  {"intermediate_size", 4096}};
  const vllm::multimodal::Qwen3VLVisionConfig p =
      vllm::Qwen3_5DenseVisionConfig(partial);
  CHECK(p.depth == 24);
  CHECK(p.hidden_size == 1024);
  CHECK(p.intermediate_size == 4096);
  CHECK(p.num_heads == 16);
  CHECK(p.num_position_embeddings == 2304);
  CHECK(p.out_hidden_size == 2560);
}

TEST_CASE("qwen3_5_dense_vision_absent_visual_tensors_are_refused_by_name") {
  const std::string p = TmpDir() + "/vllmcpp_dense_novisual.safetensors";
  WriteShard(p, {"model.embed_tokens.weight", "model.norm.weight",
                 "lm_head.weight"});
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(p));

  CHECK_FALSE(vllm::HasQwen3_5DenseVisionTower(shards));

  // The whole point: a text-only checkpoint must NOT quietly yield an empty
  // tower. It must name what is missing.
  std::string msg;
  bool threw = false;
  try {
    (void)vllm::LoadQwen3_5DenseVision(shards, MakeConfig());
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  REQUIRE(threw);
  CHECK(msg.find("model.visual.") != std::string::npos);
  CHECK(msg.find("patch_embed") != std::string::npos);
  CHECK(msg.find("merger") != std::string::npos);
  std::remove(p.c_str());
}

TEST_CASE("qwen3_5_dense_vision_present_visual_tensors_are_seen_not_dropped") {
  const std::string p = TmpDir() + "/vllmcpp_dense_withvisual.safetensors";
  // One real `model.visual.*` name from the artifact's index.
  WriteShard(p, {"model.embed_tokens.weight",
                 "model.visual.blocks.0.attn.proj.bias"});
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(p));

  CHECK(vllm::HasQwen3_5DenseVisionTower(shards));

  // The tower is now REACHED: this checkpoint no longer trips the "no tower"
  // refusal, so the load proceeds into the shared reader and fails on the FIRST
  // tower tensor it actually needs — a different, named error.
  std::string msg;
  try {
    (void)vllm::LoadQwen3_5DenseVision(shards, MakeConfig());
  } catch (const std::exception& e) {
    msg = e.what();
  }
  REQUIRE(!msg.empty());
  CHECK(msg.find("carries NO `model.visual.*` tensors") == std::string::npos);
  CHECK(msg.find("patch_embed.proj.weight") != std::string::npos);
  std::remove(p.c_str());
}

// ── THE ROW'S RED: a vision-inclusive dense load is no longer silent ─────────

TEST_CASE("qwen3_5_dense_loader_stops_silently_dropping_model_visual") {
  // BEFORE this row, `LoadQwen3_5Dense` on a checkpoint that carries
  // `model.visual.*` tensors returned a text-only model without a WORD about
  // the 333 tensors it never read — the silent drop this row removes. The
  // loader now reaches for the tower, so an INCOMPLETE tower is a loud, named
  // failure naming the first tensor it needs. (A complete synthetic 27-block
  // tower is ~1.6 GB f32; the real complete load is the operator's device
  // gate against the artifact.)
  const std::string p = TmpDir() + "/vllmcpp_dense_loader_visual.safetensors";
  WriteShard(p, {"model.embed_tokens.weight", "model.norm.weight",
                 "model.visual.blocks.0.attn.proj.bias"});
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(p));

  REQUIRE(vllm::HasQwen3_5DenseVisionTower(shards));

  const HfConfig c = MakeConfig();
  std::string msg;
  bool threw = false;
  try {
    (void)vllm::LoadQwen3_5Dense(shards, c, /*load_queue=*/nullptr);
  } catch (const std::exception& e) {
    threw = true;
    msg = e.what();
  }
  REQUIRE(threw);
  CHECK(msg.find("model.visual.") != std::string::npos);
  std::remove(p.c_str());
}

TEST_CASE("qwen3_5_dense_loader_text_only_checkpoint_stays_text_only") {
  // Gate 2 of the spec: a text-only checkpoint's dense load is unchanged. No
  // tower, no refusal, no visual member populated.
  const std::string p = TmpDir() + "/vllmcpp_dense_loader_textonly.safetensors";
  WriteShard(p, {"model.embed_tokens.weight", "model.norm.weight",
                 "lm_head.weight"});
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(p));

  CHECK_FALSE(vllm::HasQwen3_5DenseVisionTower(shards));

  const HfConfig c = MakeConfig();
  // A text-only checkpoint must NOT trip the tower refusal: the load proceeds
  // and fails (or succeeds) on TEXT tensors only. One layer type of the
  // synthetic config wants tensors this minimal shard lacks, so the load may
  // still throw — but the message must be about the BACKBONE, never the tower.
  std::string msg;
  try {
    (void)vllm::LoadQwen3_5Dense(shards, c, nullptr);
  } catch (const std::exception& e) {
    msg = e.what();
  }
  CHECK(msg.find("model.visual.") == std::string::npos);
  std::remove(p.c_str());
}

TEST_CASE("qwen3_5_dense_loader_leaves_the_tower_unread_at_zero_limits") {
  // #607 L3: the engine's modality limits reach the dense loader. The same
  // incomplete-tower shard the row's red test uses fails BY NAME on the tower
  // with no limits, and must fail on the BACKBONE (never `model.visual.`) when
  // every modality the tower serves is at limit 0 — which is what proves the
  // tensors were not read.
  const std::string p = TmpDir() + "/vllmcpp_dense_loader_skiptower.safetensors";
  WriteShard(p, {"model.embed_tokens.weight", "model.norm.weight",
                 "model.visual.blocks.0.attn.proj.bias"});
  std::vector<SafetensorsFile> shards;
  shards.push_back(SafetensorsFile::Open(p));
  REQUIRE(vllm::HasQwen3_5DenseVisionTower(shards));

  const HfConfig c = MakeConfig();
  auto load_with = [&](const vllm::MultiModalConfig* mm) {
    std::string msg;
    try {
      (void)vllm::LoadQwen3_5Dense(shards, c, nullptr, mm);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    return msg;
  };

  // Baseline: no limits, so the tower is reached and refused by name.
  CHECK(load_with(nullptr).find("model.visual.") != std::string::npos);

  // `--language-model-only`, as the serve flag sets it.
  vllm::MultiModalConfig lmo;
  lmo.language_model_only = true;
  CHECK(load_with(&lmo).find("model.visual.") == std::string::npos);

  // The other route to zero, which upstream treats identically.
  vllm::MultiModalConfig zero;
  zero.limit_per_prompt = {{"image", 0}, {"video", 0}};
  CHECK(load_with(&zero).find("model.visual.") == std::string::npos);

  std::remove(p.c_str());
}

// ── FORWARD: the tower output is actually CONSUMED, at the right rows ────────

TEST_CASE("qwen3_5_dense_vl_image_forward_is_the_text_forward_over_the_tower_row") {
  const HfConfig c = MakeConfig();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t H = c.hidden_size;
  const int32_t kImg = 30;      // the image placeholder id
  constexpr int kSteps = 6;

  // ONE image token, grid (1,2,2) -> a 1x1x1 LLM grid -> MRoPE degenerates to
  // the plain 1-D positions.
  const std::array<int64_t, 3> grid = {1, 2, 2};
  auto prompt_with = [&](int32_t mid) {
    return std::vector<int32_t>{1, 2, 3, mid, 5, 6};
  };
  const std::vector<int32_t> prompt_img = prompt_with(kImg);
  const std::vector<int32_t> text_placeholder =
      TextGreedy(prompt_img, w, c, q, kSteps);

  // Pick the two substituted tokens BY SEARCH rather than by hardcoding, so a
  // coarse synthetic argmax cannot make the equality vacuous silently.
  int32_t kSub = -1, kSub2 = -1;
  std::vector<int32_t> text_sub, text_sub2;
  for (int32_t k = 0; k < static_cast<int32_t>(c.vocab_size) && kSub2 < 0; ++k) {
    if (k == kImg) continue;
    const std::vector<int32_t> t = TextGreedy(prompt_with(k), w, c, q, kSteps);
    if (t == text_placeholder) continue;
    if (kSub < 0) {
      kSub = k;
      text_sub = t;
    } else if (t != text_sub) {
      kSub2 = k;
      text_sub2 = t;
    }
  }
  REQUIRE(kSub >= 0);
  REQUIRE(kSub2 >= 0);
  MESSAGE("substituted ids: kSub=" << kSub << " kSub2=" << kSub2);
  REQUIRE(text_sub != text_sub2);
  REQUIRE(text_sub != text_placeholder);

  const std::vector<int32_t> vl_sub = vllm::Qwen3_5VLGenerateGreedy(
      prompt_img, EmbeddingRowF32(w, kSub, H), grid, kImg, /*eos_token_id=*/-1,
      w, c, q, kSteps);
  const std::vector<int32_t> vl_sub2 = vllm::Qwen3_5VLGenerateGreedy(
      prompt_img, EmbeddingRowF32(w, kSub2, H), grid, kImg, -1, w, c, q, kSteps);

  // THE gate. Feeding the tower row for `kSub` must reproduce the text run
  // over the prompt with `kSub` in that position, token for token. A tower
  // that is never invoked would leave the placeholder's own embedding in that
  // row and so reproduce `text_placeholder` while still emitting plausible
  // tokens.
  CHECK(vl_sub == text_sub);
  CHECK(vl_sub2 == text_sub2);
  CHECK(vl_sub != text_placeholder);
}

TEST_CASE("qwen3_5_dense_vl_image_forward_uses_MRoPE_positions_not_plain_1d") {
  const HfConfig c = MakeConfigAllAttn();
  const Qwen3_5DenseWeights w = MakeWeights(c);
  vt::Queue q = Q();
  const int64_t H = c.hidden_size;
  const int32_t kImg = 30, kSub = 16;
  constexpr int kSteps = 6;

  // The COMPLEMENT of the degenerate case. Grid (1,16,16) -> an 8x8 LLM grid,
  // so the 64 image tokens carry MRoPE (t,h,w) positions in 0..7 per axis
  // where plain 1-D positions would run 0..63. Feeding the SAME substituted
  // embedding in every visual row isolates the positions: the only thing that
  // can still differ from the text run is the rope. This is the gate that
  // fails if the injected MRoPE cos|sin cache is ignored.
  const std::array<int64_t, 3> grid = {1, 16, 16};
  const int64_t n_vis = (grid[1] / 2) * (grid[2] / 2);  // 64
  std::vector<int32_t> prompt_img = {1, 2, 3};
  std::vector<int32_t> prompt_sub = {1, 2, 3};
  for (int64_t i = 0; i < n_vis; ++i) {
    prompt_img.push_back(kImg);
    prompt_sub.push_back(kSub);
  }
  prompt_img.push_back(5);
  prompt_sub.push_back(5);

  const std::vector<float> row = EmbeddingRowF32(w, kSub, H);
  std::vector<float> mm_rows(static_cast<size_t>(n_vis * H));
  for (int64_t r = 0; r < n_vis; ++r)
    for (int64_t j = 0; j < H; ++j)
      mm_rows[static_cast<size_t>(r * H + j)] =
          row[static_cast<size_t>(j)];

  const std::vector<int32_t> vl =
      vllm::Qwen3_5VLGenerateGreedy(prompt_img, mm_rows, grid, kImg, -1, w, c,
                                    q, kSteps);
  const std::vector<int32_t> text_1d = TextGreedy(prompt_sub, w, c, q, kSteps);

  // Not "the output moved": the VL run must be a DIFFERENT token stream from
  // the 1-D run, which is only possible if the MRoPE cache reached the rope.
  REQUIRE(vl.size() == text_1d.size());
  CHECK(vl != text_1d);
}
