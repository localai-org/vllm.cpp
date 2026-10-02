// The `laguna` GGUF loader (`LoadLagunaFromGgufShards`), which had NO test.
//
// It is the third consumer of `GgufTensorRole::kEmbeddingTable`, and #1989's
// W6a made that role keep-quant ELIGIBLE. Laguna reads `token_embd` as a flat
// host f32 array (`LagunaEmbed`, laguna.cpp:1594) and, when the file is tied,
// feeds the SAME f32 image to the final projection (`MatmulNK(src,
// ReadF32(weights.embed), ...)`, laguna.cpp:1281), so it cannot read a table
// that keeps its ggml blocks — its loader threw on every published quantized
// checkpoint and nothing in the tree noticed, because nothing loaded a laguna
// GGUF at all.
//
// This file builds a TINY SYNTHETIC `laguna` GGUF with the real tensor naming
// and a BLOCK-QUANTIZED `token_embd.weight`, and asserts the load survives it.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/laguna.h"
#include "vllm/model_executor/models/laguna_ops.h"
#include "vt/dtype.h"

using gguf_test::F32Kv;
using gguf_test::GgufModelBuilder;
using gguf_test::I32ArrayKv;
using gguf_test::StrKv;
using gguf_test::TempFile;
using gguf_test::U32Kv;

namespace {

// Tiny geometry. Every K is a multiple of 32 so Q8_0 encodes every weight,
// including the gather table.
struct Dims {
  int64_t H = 64, vocab = 16, head_dim = 32, kv_heads = 1;
  int64_t inter = 64, moe_inter = 32, shared_inter = 32;
  int64_t E = 2, topk = 1, n_layer = 2, leading_dense = 1;
  std::vector<int32_t> heads = {2, 4};  // per-layer Q heads: min == global
};

int64_t Prod(const std::vector<int64_t>& s) {
  int64_t n = 1;
  for (int64_t d : s) n *= d;
  return n;
}

std::vector<uint64_t> GgmlDims(const std::vector<int64_t>& torch_shape) {
  std::vector<uint64_t> d;
  for (auto it = torch_shape.rbegin(); it != torch_shape.rend(); ++it)
    d.push_back(static_cast<uint64_t>(*it));
  return d;
}

// Smooth, nonzero-amax fill so Q8_0 carries a real scale.
float WFill(int64_t i) { return 0.05F * static_cast<float>((i % 13) - 6); }

std::string F32Data(int64_t n) {
  std::string s;
  s.reserve(static_cast<size_t>(n) * 4);
  for (int64_t i = 0; i < n; ++i) {
    const float v = WFill(i);
    uint32_t bits = 0;
    std::memcpy(&bits, &v, 4);
    for (int k = 0; k < 4; ++k) s.push_back(static_cast<char>((bits >> (8 * k)) & 0xff));
  }
  return s;
}

// torch [.., out, in] (in % 32 == 0) as block_q8_0 = { f16 d; int8 qs[32] }.
std::string Q8Data(int64_t out, int64_t in) {
  std::string s;
  for (int64_t o = 0; o < out; ++o) {
    for (int64_t b = 0; b < in / 32; ++b) {
      float x[32];
      float amax = 0.0F;
      for (int j = 0; j < 32; ++j) {
        x[j] = WFill(o * in + b * 32 + j);
        amax = std::max(amax, std::fabs(x[j]));
      }
      const float d = amax / 127.0F;
      const uint16_t dh = vt::F32ToF16(d);
      s.push_back(static_cast<char>(dh & 0xff));
      s.push_back(static_cast<char>((dh >> 8) & 0xff));
      for (int j = 0; j < 32; ++j) {
        int q = d > 0.0F ? static_cast<int>(std::lround(x[j] / d)) : 0;
        q = std::max(-127, std::min(127, q));
        s.push_back(static_cast<char>(static_cast<int8_t>(q)));
      }
    }
  }
  return s;
}

std::string Blk(int64_t l, const std::string& s) {
  return "blk." + std::to_string(l) + "." + s;
}

// `q8_embed` stores `token_embd.weight` BLOCK-QUANTIZED, which every published
// laguna checkpoint does; `tied` omits `output.weight`.
std::string BuildGguf(const Dims& d, bool q8_embed, bool tied = false,
                      float yarn_attn_factor = 0.0F) {
  GgufModelBuilder b;
  const std::string p = "laguna.";
  b.AddKv(StrKv("general.architecture", "laguna"));
  b.AddKv(U32Kv(p + "embedding_length", d.H));
  b.AddKv(U32Kv(p + "block_count", d.n_layer));
  b.AddKv(U32Kv(p + "attention.head_count_kv", d.kv_heads));
  b.AddKv(U32Kv(p + "attention.key_length", d.head_dim));
  b.AddKv(U32Kv(p + "feed_forward_length", d.inter));
  b.AddKv(I32ArrayKv(p + "attention.head_count", d.heads));
  b.AddKv(U32Kv(p + "expert_count", d.E));
  b.AddKv(U32Kv(p + "expert_used_count", d.topk));
  b.AddKv(U32Kv(p + "expert_feed_forward_length", d.moe_inter));
  b.AddKv(U32Kv(p + "expert_shared_feed_forward_length", d.shared_inter));
  b.AddKv(U32Kv(p + "leading_dense_block_count", d.leading_dense));
  b.AddKv(U32Kv(p + "rope.dimension_count", d.head_dim));
  b.AddKv(F32Kv(p + "attention.layer_norm_rms_epsilon", 1e-6F));
  if (yarn_attn_factor > 0.0F) {
    b.AddKv(F32Kv(p + "rope.scaling.factor", 32.0F));
    b.AddKv(F32Kv(p + "rope.scaling.yarn_attn_factor", yarn_attn_factor));
  }

  auto add = [&](const std::string& name, bool q8,
                 const std::vector<int64_t>& shape) {
    if (q8) {
      const int64_t out = shape.size() == 3 ? shape[0] * shape[1] : shape[0];
      b.AddTensor(name, GgmlDims(shape), /*Q8_0=*/8, Q8Data(out, shape.back()));
    } else {
      b.AddTensor(name, GgmlDims(shape), /*F32=*/0, F32Data(Prod(shape)));
    }
  };

  add("token_embd.weight", q8_embed, {d.vocab, d.H});
  add("output_norm.weight", false, {d.H});
  if (!tied) add("output.weight", true, {d.vocab, d.H});

  for (int64_t l = 0; l < d.n_layer; ++l) {
    const int64_t nh = d.heads[static_cast<size_t>(l)];
    add(Blk(l, "attn_norm.weight"), false, {d.H});
    add(Blk(l, "ffn_norm.weight"), false, {d.H});
    add(Blk(l, "attn_q.weight"), true, {nh * d.head_dim, d.H});
    add(Blk(l, "attn_k.weight"), true, {d.kv_heads * d.head_dim, d.H});
    add(Blk(l, "attn_v.weight"), true, {d.kv_heads * d.head_dim, d.H});
    add(Blk(l, "attn_output.weight"), true, {d.H, nh * d.head_dim});
    add(Blk(l, "attn_gate.weight"), true, {nh, d.H});
    add(Blk(l, "attn_q_norm.weight"), false, {d.head_dim});
    add(Blk(l, "attn_k_norm.weight"), false, {d.head_dim});
    if (l < d.leading_dense) {
      add(Blk(l, "ffn_gate.weight"), true, {d.inter, d.H});
      add(Blk(l, "ffn_up.weight"), true, {d.inter, d.H});
      add(Blk(l, "ffn_down.weight"), true, {d.H, d.inter});
    } else {
      add(Blk(l, "ffn_gate_inp.weight"), false, {d.E, d.H});
      add(Blk(l, "exp_probs_b.bias"), false, {d.E});
      add(Blk(l, "ffn_gate_exps.weight"), true, {d.E, d.moe_inter, d.H});
      add(Blk(l, "ffn_up_exps.weight"), true, {d.E, d.moe_inter, d.H});
      add(Blk(l, "ffn_down_exps.weight"), true, {d.E, d.H, d.moe_inter});
      add(Blk(l, "ffn_gate_shexp.weight"), true, {d.shared_inter, d.H});
      add(Blk(l, "ffn_up_shexp.weight"), true, {d.shared_inter, d.H});
      add(Blk(l, "ffn_down_shexp.weight"), true, {d.H, d.shared_inter});
    }
  }
  return b.Build();
}

vllm::GgufLoadPolicy KeepPolicy() {
  vllm::GgufLoadPolicy pol;
  pol.keep_quant = true;
  return pol;
}
const vllm::GgufLoadPolicy kExpandAll;  // all defaults -> dequant everything

}  // namespace

TEST_CASE("LoadLagunaFromGguf: an F32 token_embd loads and the tower keeps quant") {
  Dims d;
  TempFile f(BuildGguf(d, /*q8_embed=*/false));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::LagunaWeights w =
      vllm::LoadLagunaFromGgufShards({&g}, &keep);

  CHECK(w.has_gguf_weights);
  CHECK(w.params.hidden_size == d.H);
  CHECK(w.params.vocab_size == d.vocab);
  CHECK(w.params.num_hidden_layers == d.n_layer);
  CHECK(w.accounted_tensors == static_cast<int64_t>(g.Tensors().size()));
  // The GEMM tower kept its blocks; the gather table and the norms did not.
  CHECK(w.embed.dtype == vt::DType::kF32);
  CHECK(w.norm.dtype == vt::DType::kF32);
  CHECK(w.lm_head.dtype == vt::DType::kQ8_0);
  REQUIRE(w.layers.size() == 2);
  CHECK(w.layers[0].mlp.down_proj.dtype == vt::DType::kQ8_0);
  CHECK(w.layers[1].moe.experts_down.dtype == vt::DType::kQ8_0);
}

// ── #1989 review F1 ──────────────────────────────────────────────────────────
// The case the shared-policy change broke. Before the narrowing this threw
// "laguna gguf: a embedding_table tensor must not keep quant blocks:
// token_embd.weight" for every encoding except F32 — i.e. for every real file.
TEST_CASE("LoadLagunaFromGguf: a BLOCK-QUANTIZED token_embd still loads") {
  Dims d;
  TempFile f(BuildGguf(d, /*q8_embed=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  // Non-vacuity, both halves: the fixture really is quantized, AND the shared
  // policy really does elect to keep a table of that shape.
  REQUIRE(g.Get("token_embd.weight").ggml_type == 8u);  // ggml Q8_0
  REQUIRE(vllm::PeekRoute(KeepPolicy(), g.Get("token_embd.weight"),
                          vllm::GgufTensorRole::kEmbeddingTable) ==
          vllm::GgufResidency::kKeepQuant);

  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::LagunaWeights wk = vllm::LoadLagunaFromGgufShards({&g}, &keep);
  const vllm::LagunaWeights we = vllm::LoadLagunaFromGgufShards({&g}, &kExpandAll);

  // Laguna reads this table as a flat f32 array, so its residency must not
  // depend on the policy: the two loads produce the SAME bytes.
  CHECK(wk.embed.dtype == vt::DType::kF32);
  REQUIRE(wk.embed.rank == 2);
  CHECK(wk.embed.shape[0] == d.vocab);
  CHECK(wk.embed.shape[1] == d.H);
  REQUIRE(wk.embed.bytes.size() == we.embed.bytes.size());
  CHECK(std::memcmp(wk.embed.bytes.data(), we.embed.bytes.data(),
                    wk.embed.bytes.size()) == 0);
  // ...and it is not silently zero, which would satisfy that equality empty.
  const auto* vals = reinterpret_cast<const float*>(wk.embed.bytes.data());
  const size_t n = wk.embed.bytes.size() / sizeof(float);
  CHECK(std::any_of(vals, vals + n, [](float v) { return v != 0.0F; }));
  // The narrowing is role-scoped, not "keep-quant off": the GEMM tower still
  // keeps its blocks in the same load.
  CHECK(wk.lm_head.dtype == vt::DType::kQ8_0);
  CHECK(wk.layers[1].moe.experts_down.dtype == vt::DType::kQ8_0);
}

// A TIED laguna file has no `output.weight`, and the final projection then reads
// the SAME f32 `w.embed` image (laguna.cpp:1281). The table therefore has to
// expand on this arm too.
TEST_CASE("LoadLagunaFromGguf: a TIED block-quantized token_embd still loads") {
  Dims d;
  TempFile f(BuildGguf(d, /*q8_embed=*/true, /*tied=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::GgufLoadPolicy keep = KeepPolicy();
  const vllm::LagunaWeights w = vllm::LoadLagunaFromGgufShards({&g}, &keep);

  CHECK(w.embed.dtype == vt::DType::kF32);
  CHECK(w.lm_head.Empty());  // tied: the head IS `embed`
  CHECK(w.params.vocab_size == d.vocab);
}

// #2841: the config FromModelDir builds for a `laguna` GGUF. The engine sizes
// its KV cache and scheduler from this HfConfig while the weight loader
// resolves the file itself, so the two must agree on every field the forward
// reads. The case pins that `ParseLagunaParams` on the built config recovers
// the loader's own `LagunaParamsFromGguf`, and that the registry resolves it.
TEST_CASE("LagunaHfConfigFromGguf: the engine config agrees with the loader's resolve") {
  Dims d;
  TempFile f(BuildGguf(d, /*q8_embed=*/true));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::LagunaParams want = vllm::LagunaParamsFromGguf(g);
  const vllm::HfConfig c = vllm::LagunaHfConfigFromGguf(g);

  REQUIRE(c.architectures.size() == 1);
  CHECK(c.architectures[0] == "LagunaForCausalLM");
  CHECK(c.hidden_size == d.H);
  CHECK(c.num_hidden_layers == d.n_layer);
  CHECK(c.num_key_value_heads == d.kv_heads);
  CHECK(c.head_dim == d.head_dim);
  // No `laguna.vocab_size` key in this file: the token table's rows answer.
  CHECK(c.vocab_size == d.vocab);
  CHECK(c.layer_types == want.layer_types);

  const vllm::LagunaParams got = vllm::ParseLagunaParams(c);
  CHECK(got.num_attention_heads == want.num_attention_heads);
  CHECK(got.num_attention_heads_per_layer == want.num_attention_heads_per_layer);
  CHECK(got.layer_types == want.layer_types);
  CHECK(got.intermediate_size == want.intermediate_size);
  CHECK(got.num_experts == want.num_experts);
  CHECK(got.num_experts_per_tok == want.num_experts_per_tok);
  CHECK(got.moe_intermediate_size == want.moe_intermediate_size);
  CHECK(got.shared_expert_intermediate_size == want.shared_expert_intermediate_size);
  CHECK(got.mlp_only_layers == want.mlp_only_layers);
  CHECK(got.norm_topk_prob == want.norm_topk_prob);
  CHECK(got.rotary_dim_full == want.rotary_dim_full);
  CHECK(got.rotary_dim_sliding == want.rotary_dim_sliding);
  CHECK(got.rope_theta_full == doctest::Approx(want.rope_theta_full));
  CHECK(got.rope_theta_sliding == doctest::Approx(want.rope_theta_sliding));
  CHECK(got.yarn_factor == doctest::Approx(want.yarn_factor));
  CHECK(got.yarn_attn_factor == doctest::Approx(want.yarn_attn_factor));
  CHECK(got.tie_word_embeddings == false);

  // A tied file (no `output.weight`) is reported as tied.
  TempFile tf(BuildGguf(d, /*q8_embed=*/true, /*tied=*/true));
  const vllm::GgufFile tg = vllm::GgufFile::Open(tf.path());
  CHECK(vllm::ParseLagunaParams(vllm::LagunaHfConfigFromGguf(tg)).tie_word_embeddings);
}

// llama.cpp `b10451` reads `rope.scaling.yarn_attn_factor` for grok only
// (src/models/grok.cpp:25) and derives the YaRN attention factor for every other
// model from `rope.scaling.factor` (src/llama-context.cpp:176-213). The
// Laguna-XS-2.1 APEX file stores the already-multiplied 1.34657 in that key;
// reading it applied the mscale twice (1.813) on every full-attention layer.
TEST_CASE("LagunaParamsFromGguf: the stored yarn_attn_factor is ignored, as llama.cpp does") {
  Dims d;
  TempFile f(BuildGguf(d, /*q8_embed=*/false, /*tied=*/false, /*yarn_attn_factor=*/1.3465736F));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::LagunaParams p = vllm::LagunaParamsFromGguf(g);
  CHECK(p.yarn_factor == doctest::Approx(32.0));
  CHECK(p.yarn_attn_factor == doctest::Approx(1.0));
  // The mscale the full-attention RoPE applies: once, as llama.cpp's kernel does.
  CHECK(vllm::LagunaYarnMscale(p.yarn_factor, p.yarn_attn_factor) ==
        doctest::Approx(1.0 + 0.1 * std::log(32.0)).epsilon(1e-6));
}
