// GGUF metadata and inverse Q/K row transform from llama.cpp 62318cd57:
// conversion/llama.py:175-181,255-264. vLLM a7c23ac96 llama.py owns the HF
// QKV/SwiGLU/RMSNorm/RoPE semantics reached by the existing Llama registry.
#include "vllm/model_executor/models/llama_gguf_weights.h"
#include <cstring>
#include <variant>
#include <type_traits>
#include "vllm/model_executor/models/qwen3_gguf_weights.h"
#include "vt/quant.h"
namespace vllm {
namespace {
int64_t Int(const GgufFile& g, const std::string& key, int64_t fallback = -1) {
  const auto* v = g.FindKv(key);
  if (!v) { VT_CHECK(fallback >= 0, "llama gguf: missing " + key); return fallback; }
  return std::visit([&](const auto& x) -> int64_t {
    using T = std::decay_t<decltype(x)>;
    if constexpr (std::is_integral_v<T>) return static_cast<int64_t>(x);
    else { throw std::runtime_error("llama gguf: expected integer " + key); }
  }, v->v);
}
double Float(const GgufFile& g, const std::string& key, double fallback) {
  const auto* v = g.FindKv(key); if (!v) return fallback;
  return std::visit([&](const auto& x) -> double {
    using T = std::decay_t<decltype(x)>;
    if constexpr (std::is_arithmetic_v<T>) return static_cast<double>(x);
    else { throw std::runtime_error("llama gguf: expected number " + key); }
  }, v->v);
}
bool Has(const GgufFile& g, const std::string& name) {
  for (const auto& t : g.Tensors()) if (t.name == name) return true;
  return false;
}
// HF [head, 2, D/2, K] became GGUF [head, D/2, 2, K]. Move whole stored
// rows back, preserving every quant block and scale without requantization.
void UndoQkRows(OwnedTensor& w, int64_t heads, int64_t dim) {
  VT_CHECK(w.rank == 2 && w.nk && w.shape[0] == heads * dim && dim % 2 == 0,
           "llama gguf: invalid Q/K row geometry");
  VT_CHECK(!w.repacked && !w.q8_0_aligned && !w.elem_kn_repacked,
           "llama gguf: Q/K row permutation requires canonical packed rows");
  const int64_t block = vt::IsBlockQuant(w.dtype) ? vt::BlockElems(w.dtype) : 1;
  VT_CHECK(w.shape[1] % block == 0, "llama gguf: Q/K row is not whole blocks");
  const size_t row_bytes = static_cast<size_t>(w.shape[1] / block) *
      (vt::IsBlockQuant(w.dtype) ? vt::BlockBytes(w.dtype) : vt::SizeOf(w.dtype));
  OwnedBytes reordered; reordered.resize(w.bytes.size());
  const OwnedTensor& source = w;
  for (int64_t h = 0; h < heads; ++h) for (int64_t i = 0; i < dim; ++i) {
    const int64_t from = h * dim + 2 * (i % (dim / 2)) + i / (dim / 2);
    std::memcpy(reordered.data() + (h * dim + i) * row_bytes,
                source.bytes.data() + from * row_bytes, row_bytes);
  }
  w.bytes = std::move(reordered);
}
}
HfConfig LlamaHfConfigFromGguf(const GgufFile& g) {
  const auto* arch = g.FindKv("general.architecture");
  VT_CHECK(arch && arch->TypeId() == kGgufString && std::get<std::string>(arch->v) == "llama",
           "llama gguf: expected architecture llama");
  const auto* scaling = g.FindKv("llama.rope.scaling.type");
  if (scaling) VT_CHECK(scaling->TypeId() == kGgufString &&
      std::get<std::string>(scaling->v) == "none",
      "llama gguf: scaled RoPE metadata is not implemented on this arm");
  VT_CHECK(!Has(g, "rope_freqs.weight") && !Has(g,"rope_factors_long.weight") &&
           !Has(g,"rope_factors_short.weight"), "llama gguf: RoPE frequency sidecar is not implemented on this arm");
  for (const auto& t : g.Tensors()) {
    VT_CHECK(t.name.find("attn_q_norm") == std::string::npos && t.name.find("attn_k_norm") == std::string::npos,
             "llama gguf: Q/K normalization variant is not implemented on this arm");
    VT_CHECK(!t.name.ends_with(".bias"), "llama gguf: bias tensors are not implemented on this arm");
  }
  HfConfig c; c.model_type = "llama"; c.architectures = {"LlamaForCausalLM"};
  c.hidden_size = Int(g,"llama.embedding_length");
  c.num_hidden_layers = Int(g,"llama.block_count");
  c.num_attention_heads = Int(g,"llama.attention.head_count");
  VT_CHECK(c.hidden_size > 0 && c.num_hidden_layers > 0 && c.num_attention_heads > 0 &&
           c.hidden_size % c.num_attention_heads == 0, "llama gguf: invalid model geometry");
  c.num_key_value_heads = Int(g,"llama.attention.head_count_kv",c.num_attention_heads);
  c.head_dim = Int(g,"llama.attention.key_length",c.hidden_size/c.num_attention_heads);
  c.rotary_dim = Int(g,"llama.rope.dimension_count",c.head_dim);
  VT_CHECK(Int(g,"llama.attention.value_length",c.head_dim) == c.head_dim,
           "llama gguf: asymmetric value heads are not implemented on this arm");
  VT_CHECK(c.num_key_value_heads > 0 && c.num_attention_heads % c.num_key_value_heads == 0 &&
           c.head_dim > 0 && c.head_dim % 2 == 0 && c.rotary_dim == c.head_dim,
           "llama gguf: invalid GQA or partial RoPE is not implemented on this arm");
  c.intermediate_size = Int(g,"llama.feed_forward_length");
  VT_CHECK(c.intermediate_size > 0, "llama gguf: feed-forward width must be positive");
  c.vocab_size = Int(g,"llama.vocab_size",g.Get("token_embd.weight").shape[0]);
  c.max_position_embeddings = Int(g,"llama.context_length",0);
  c.rope_theta = Float(g,"llama.rope.freq_base",10000.0);
  c.rope_parameters.rope_theta = c.rope_theta;
  c.rms_norm_eps = Float(g,"llama.attention.layer_norm_rms_epsilon",1e-5);
  VT_CHECK(c.vocab_size > 0 && c.rope_theta > 0 && c.rms_norm_eps > 0,
           "llama gguf: invalid vocabulary, RoPE base, or norm epsilon");
  c.torch_dtype = "bfloat16";
  c.raw["tie_word_embeddings"] = !Has(g,"output.weight");
  c.raw["attention_bias"] = false;
  return c;
}
LlamaWeights LoadLlamaFromGguf(const GgufFile& g, const HfConfig& c,
                              const GgufLoadPolicy* policy) {
  auto w = LoadDenseGgufWeights(g,c,policy,true);
  for (auto& layer : w.layers) {
    UndoQkRows(layer.attn.qkv_proj_parts[0],c.num_attention_heads,c.head_dim);
    UndoQkRows(layer.attn.qkv_proj_parts[1],c.num_key_value_heads,c.head_dim);
  }
  return w;
}
}
