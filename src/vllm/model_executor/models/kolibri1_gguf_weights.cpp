// vllm.cpp ORIGINAL — the `kolibri1` GGUF weights loader. See the header and
// .agents/specs/kolibri1-gguf-arch.md.
//
// The load expands every k-quant to bf16 and fills the SAME Kolibri1Weights
// the safetensors loader (kolibri1_weights.cpp) produces, so the production
// forward (kolibri1_forward.cpp LinearBT / LinearBTRaw) is unchanged and the
// bf16 arm of Kolibri1Projection is the arm every projection lands in. GGUF
// ne order is reversed by the reader into torch row-major [out, in]
// (gguf_reader.cpp:515-519), which is exactly the MatmulBT orientation
// LinearBTRaw consumes — every 2-D weight is a verbatim dequant + copy, with
// no transpose anywhere in this file.
#include "vllm/model_executor/models/kolibri1_gguf_weights.h"

#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

int64_t ShapeNumel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

bool HasTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "kolibri1 gguf: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

// Dequant a whole tensor to bf16, then own the bytes under `shape`.
OwnedTensor OwnBf16(const GgufFile& g, const std::string& name,
                    const std::vector<int64_t>& shape) {
  const GgufTensorInfo& t = g.Get(name);
  std::vector<uint16_t> dq =
      DequantGgufRowToBf16(t.ggml_type, t.data, ShapeNumel(t.shape), 1.0F);
  VT_CHECK(ShapeNumel(shape) == static_cast<int64_t>(dq.size()),
           "kolibri1 gguf: element-count mismatch for " + name);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  std::memcpy(o.bytes.data(), dq.data(), dq.size() * sizeof(uint16_t));
  return o;
}

// `rows` rows of `cols` elements starting at row `row_offset` of a stacked
// 3-D tensor — how one expert is cut out of ffn_{gate,up,down}_exps.
OwnedTensor OwnBf16RowSlice(const GgufTensorInfo& t,
                            int64_t rows, int64_t cols, int64_t row_offset) {
  const int64_t total_rows = t.shape[0] * t.shape[1];  // [E, out] flattened
  VT_CHECK(t.shape.size() == 3 && t.shape[2] == cols,
           "kolibri1 gguf: " + t.name + " is not the stacked expert tower");
  VT_CHECK(row_offset >= 0 && row_offset + rows <= total_rows,
           "kolibri1 gguf: row slice out of range for " + t.name);
  const int64_t numel = t.shape[0] * t.shape[1] * t.shape[2];
  std::vector<uint16_t> dq =
      DequantGgufRowToBf16(t.ggml_type, t.data, numel, 1.0F);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {rows, cols});
  std::memcpy(o.bytes.data(),
              dq.data() + static_cast<size_t>(row_offset) *
                              static_cast<size_t>(cols),
              static_cast<size_t>(rows) * static_cast<size_t>(cols) *
                  sizeof(uint16_t));
  return o;
}

// An RMS-norm gamma: F32 on disk in this container, converted to the bf16 the
// forward's ResidentWeight scratch consumes (the safetensors arm ships the
// gammas as BF16 directly).
OwnedTensor OwnNormBf16(const GgufFile& g, const std::string& name,
                        int64_t dim) {
  const GgufTensorInfo& t = g.Get(name);
  VT_CHECK(t.ggml_type == 0,  // GGML_TYPE_F32
           "kolibri1 gguf: " + name + " is not F32");
  VT_CHECK(t.shape.size() == 1 && t.shape[0] == dim,
           "kolibri1 gguf: " + name + " is not [" + std::to_string(dim) + "]");
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {dim});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  const auto* src = reinterpret_cast<const float*>(t.data);
  for (int64_t i = 0; i < dim; ++i) dst[i] = vt::F32ToBF16(src[i]);
  return o;
}

// One linear projection into the bf16 arm. `out` x `in` must equal the file's
// row-major shape, so a wrong geometry is a loud failure at the load.
Kolibri1Projection OwnProjectionBf16(const GgufFile& g, const std::string& name,
                                     int64_t out, int64_t in) {
  const GgufTensorInfo& t = g.Get(name);
  VT_CHECK(t.shape.size() == 2 && t.shape[0] == out && t.shape[1] == in,
           "kolibri1 gguf: " + name + " is [" +
               std::to_string(t.shape.empty() ? -1 : t.shape[0]) + ", " +
               std::to_string(t.shape.size() < 2 ? -1 : t.shape[1]) +
               "], expected [" + std::to_string(out) + ", " +
               std::to_string(in) + "]");
  Kolibri1Projection p;
  p.bf16 = OwnBf16(g, name, t.shape);
  return p;
}

Kolibri1ExpertWeights OwnExpert(const GgufTensorInfo& gate,
                                const GgufTensorInfo& up,
                                const GgufTensorInfo& down, int64_t e,
                                int64_t inter, int64_t hidden) {
  Kolibri1ExpertWeights x;
  x.gate_proj.bf16 = OwnBf16RowSlice(gate, inter, hidden, e * inter);
  x.up_proj.bf16 = OwnBf16RowSlice(up, inter, hidden, e * inter);
  x.down_proj.bf16 = OwnBf16RowSlice(down, hidden, inter, e * hidden);
  return x;
}

std::string Blk(int64_t il, const char* suffix) {
  return "blk." + std::to_string(il) + "." + suffix;
}

// Small KV readers, the qwen4_exp_gguf_weights.cpp shape.
int64_t KvInt(const GgufValue& v, const std::string& key) {
  switch (v.TypeId()) {
    case kGgufU8: return std::get<uint8_t>(v.v);
    case kGgufI8: return std::get<int8_t>(v.v);
    case kGgufU16: return std::get<uint16_t>(v.v);
    case kGgufI16: return std::get<int16_t>(v.v);
    case kGgufU32: return std::get<uint32_t>(v.v);
    case kGgufI32: return std::get<int32_t>(v.v);
    case kGgufU64: return static_cast<int64_t>(std::get<uint64_t>(v.v));
    case kGgufI64: return std::get<int64_t>(v.v);
    case kGgufBool: return std::get<bool>(v.v) ? 1 : 0;
    default:
      throw std::runtime_error("kolibri1 gguf: key " + key +
                               " is not an integer");
  }
}

double KvFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(KvInt(v, key));
}

int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "kolibri1 gguf: missing metadata key " + key);
  return KvInt(*v, key);
}

double ReqFloat(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "kolibri1 gguf: missing metadata key " + key);
  return KvFloat(*v, key);
}

}  // namespace

bool IsKolibri1Gguf(const GgufFile& gguf) {
  const GgufValue* v = gguf.FindKv("general.architecture");
  return v != nullptr && v->TypeId() == kGgufString &&
         std::get<std::string>(v->v) == kKolibri1GgufArch;
}

std::vector<Kolibri1Tensor> EnumerateKolibri1GgufTensors(
    const Kolibri1Params& p) {
  std::vector<Kolibri1Tensor> out;
  out.reserve(static_cast<size_t>(3 + p.num_hidden_layers * 11));
  out.push_back({"token_embd.weight", "embedding"});
  out.push_back({"output.weight", "lm_head"});
  out.push_back({"output_norm.weight", "final_norm"});
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    out.push_back({Blk(l, "attn_norm.weight"), "layer"});
    out.push_back({Blk(l, "post_attention_norm.weight"), "layer"});
    out.push_back({Blk(l, "ffn_norm.weight"), "layer"});
    out.push_back({Blk(l, "post_ffw_norm.weight"), "layer"});
    out.push_back({Blk(l, "attn_q_norm.weight"), "layer"});
    out.push_back({Blk(l, "attn_k_norm.weight"), "layer"});
    out.push_back({Blk(l, "attn_q.weight"), "layer"});
    out.push_back({Blk(l, "attn_k.weight"), "layer"});
    out.push_back({Blk(l, "attn_v.weight"), "layer"});
    out.push_back({Blk(l, "attn_output.weight"), "layer"});
    out.push_back({Blk(l, "ffn_gate_inp.weight"), "layer"});
    out.push_back({Blk(l, "exp_probs_b.bias"), "layer"});
    out.push_back({Blk(l, "ffn_gate_exps.weight"), "layer"});
    out.push_back({Blk(l, "ffn_up_exps.weight"), "layer"});
    out.push_back({Blk(l, "ffn_down_exps.weight"), "layer"});
    out.push_back({Blk(l, "ffn_gate_shexp.weight"), "layer"});
    out.push_back({Blk(l, "ffn_up_shexp.weight"), "layer"});
    out.push_back({Blk(l, "ffn_down_shexp.weight"), "layer"});
  }
  return out;
}

HfConfig Kolibri1HfConfigFromGguf(const GgufFile& gguf) {
  VT_CHECK(IsKolibri1Gguf(gguf),
           "kolibri1 gguf: general.architecture must be '" +
               std::string(kKolibri1GgufArch) + "'");
  const std::string p = "kolibri1.";

  HfConfig c;
  c.model_type = "kolibri1";
  c.architectures = {"Kolibri1ForCausalLM"};
  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_hidden_layers = ReqInt(gguf, p + "block_count");
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads =
      ReqInt(gguf, p + "attention.head_count_kv");
  c.vocab_size = gguf.Get("token_embd.weight").shape[0];

  // --- architecture-specific numbers, under HF's OWN key names -------------
  // ParseKolibri1Params reads `raw`; the names are the checkpoint's
  // config.json spellings (read 2026-10-03), NOT names invented here.
  nlohmann::json raw = nlohmann::json::object();
  raw["model_type"] = c.model_type;
  raw["architectures"] = c.architectures;
  raw["head_dim"] = ReqInt(gguf, p + "attention.key_length");
  raw["sliding_window"] = ReqInt(gguf, p + "attention.sliding_window");
  raw["use_sliding_window"] = true;
  raw["rope_theta"] = ReqFloat(gguf, p + "rope.freq_base");
  raw["rms_norm_eps"] =
      ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  raw["num_experts"] = ReqInt(gguf, p + "expert_count");
  raw["num_experts_per_tok"] = ReqInt(gguf, p + "expert_used_count");
  raw["moe_intermediate_size"] =
      ReqInt(gguf, p + "expert_feed_forward_length");
  raw["shared_expert_intermediate_size"] =
      ReqInt(gguf, p + "expert_shared_feed_forward_length");
  // `expert_weights_norm` false is the no-renormalisation contract the parser
  // already requires (norm_topk_prob=false). The layer schedule comes off the
  // file's own sliding_window_pattern array; `true` is a sliding layer,
  // corroborated field-for-field against the safetensors config.json's
  // layer_types on the real checkpoint (2026-10-23).
  raw["norm_topk_prob"] = ReqInt(gguf, p + "expert_weights_norm") != 0;
  raw["hidden_act"] = "silu";
  raw["tie_word_embeddings"] = !HasTensor(gguf, "output.weight");
  // ROUTE A (commit 39dadc637): `expert_gating_func` (= 5 on the published
  // file, outside every upstream enum) is deliberately NOT read. The
  // kolibri1 routing is fixed by the model — sigmoid, raw unscaled top-k
  // weights, per kolibri1.py:126-142; mistral.rs's build_kolibri1 never
  // reads the key either. The token gate adjudicates the interpretation.

  const GgufValue* pat = gguf.FindKv(p + "attention.sliding_window_pattern");
  VT_CHECK(pat != nullptr && pat->TypeId() == kGgufArray,
           "kolibri1 gguf: missing attention.sliding_window_pattern — the "
           "hybrid RNoPE layer schedule cannot be derived");
  const GgufArray& arr = std::get<GgufArray>(pat->v);
  VT_CHECK(static_cast<int64_t>(arr.elems.size()) == c.num_hidden_layers,
           "kolibri1 gguf: sliding_window_pattern length must equal "
           "block_count");
  nlohmann::json layer_types = nlohmann::json::array();
  for (const GgufValue& e : arr.elems) {
    layer_types.push_back(KvInt(e, "sliding_window_pattern") != 0
                              ? "sliding_attention"
                              : "full_attention");
  }
  raw["layer_types"] = std::move(layer_types);

  c.raw = std::move(raw);
  return c;
}

Kolibri1Weights LoadKolibri1FromGguf(const GgufFile& gguf,
                                     const HfConfig& config) {
  VT_CHECK(IsKolibri1Gguf(gguf),
           "kolibri1 gguf: general.architecture must be '" +
               std::string(kKolibri1GgufArch) + "'");
  const Kolibri1Params p = ParseKolibri1Params(config);

  // STRUCTURAL accounting, one direction loud and one loud here: a tensor the
  // loaders ask for and the file lacks refuses at `GgufFile::Get` by name; a
  // tensor the file carries and nothing reads refuses below.
  const std::vector<Kolibri1Tensor> expected = EnumerateKolibri1GgufTensors(p);
  std::set<std::string> present;
  for (const GgufTensorInfo& info : gguf.Tensors()) present.insert(info.name);
  std::vector<std::string> unaccounted;
  for (const std::string& n : present) {
    bool found = false;
    for (const Kolibri1Tensor& t : expected) found = found || t.name == n;
    if (!found) unaccounted.push_back(n);
  }
  VT_CHECK(unaccounted.empty(),
           "kolibri1 gguf: unaccounted tensor(s), first: " +
               (unaccounted.empty() ? std::string("(none)")
                                    : unaccounted.front()));

  const int64_t h = p.hidden_size;
  const int64_t v = p.vocab_size;
  const int64_t kv = p.num_key_value_heads * p.head_dim;
  const int64_t q = p.num_attention_heads * p.head_dim;
  const int64_t inter = p.moe_intermediate_size;

  Kolibri1Weights w;
  w.params = p;

  w.embed_tokens = OwnBf16(gguf, "token_embd.weight", {v, h});
  // Tie is read off the file (llama.cpp's writer omits output.weight exactly
  // when the head is tied); the parser refuses a tied head on this
  // architecture, so an omitting file fails there with its own message.
  if (HasTensor(gguf, "output.weight")) {
    w.lm_head = OwnBf16(gguf, "output.weight", {v, h});
  }
  w.final_norm = OwnNormBf16(gguf, "output_norm.weight", h);

  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    Kolibri1LayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.is_sliding = p.IsSlidingLayer(l);
    lw.input_layernorm = OwnNormBf16(gguf, Blk(l, "attn_norm.weight"), h);
    // The sandwich's four gammas under the converter's own names: attn_norm ->
    // input_layernorm, post_attention_norm -> post_attn_norm, ffn_norm ->
    // post_attention_layernorm (the residual-carrying norm before the MoE),
    // post_ffw_norm -> post_ffn_norm. Read off the file's own blk.0 tensor
    // table (2026-10-23), not from the issue's summary.
    lw.post_attn_norm =
        OwnNormBf16(gguf, Blk(l, "post_attention_norm.weight"), h);
    lw.post_attention_layernorm = OwnNormBf16(gguf, Blk(l, "ffn_norm.weight"), h);
    lw.post_ffn_norm = OwnNormBf16(gguf, Blk(l, "post_ffw_norm.weight"), h);

    lw.attn.q_proj = OwnProjectionBf16(gguf, Blk(l, "attn_q.weight"), q, h);
    lw.attn.k_proj = OwnProjectionBf16(gguf, Blk(l, "attn_k.weight"), kv, h);
    lw.attn.v_proj = OwnProjectionBf16(gguf, Blk(l, "attn_v.weight"), kv, h);
    lw.attn.o_proj =
        OwnProjectionBf16(gguf, Blk(l, "attn_output.weight"), h, q);
    lw.attn.q_norm = OwnNormBf16(gguf, Blk(l, "attn_q_norm.weight"),
                                 p.head_dim);
    lw.attn.k_norm = OwnNormBf16(gguf, Blk(l, "attn_k_norm.weight"),
                                 p.head_dim);

    // Router gate [num_experts, hidden], verbatim (the reader's row-major
    // [out, in] IS LinearBTRaw's orientation).
    lw.moe.router_gate =
        OwnBf16(gguf, Blk(l, "ffn_gate_inp.weight"), {p.num_experts, h});
    // e_score_correction_bias stays F32 — the same widening the safetensors
    // loader does (kolibri1_weights.cpp), already at its resident dtype here.
    {
      const GgufTensorInfo& t = gguf.Get(Blk(l, "exp_probs_b.bias"));
      VT_CHECK(t.ggml_type == 0 && t.shape.size() == 1 &&
                   t.shape[0] == p.num_experts,
               "kolibri1 gguf: exp_probs_b.bias is not F32 [num_experts]");
      lw.moe.e_score_correction_bias =
          MakeOwned(vt::DType::kF32, {p.num_experts});
      std::memcpy(lw.moe.e_score_correction_bias.bytes.data(), t.data,
                  static_cast<size_t>(p.num_experts) * sizeof(float));
    }

    const GgufTensorInfo& gate = gguf.Get(Blk(l, "ffn_gate_exps.weight"));
    const GgufTensorInfo& up = gguf.Get(Blk(l, "ffn_up_exps.weight"));
    const GgufTensorInfo& down = gguf.Get(Blk(l, "ffn_down_exps.weight"));
    lw.moe.experts.resize(static_cast<size_t>(p.num_experts));
    for (int64_t e = 0; e < p.num_experts; ++e) {
      lw.moe.experts[static_cast<size_t>(e)] =
          OwnExpert(gate, up, down, e, inter, h);
    }
    lw.moe.shared_experts.gate_proj =
        OwnProjectionBf16(gguf, Blk(l, "ffn_gate_shexp.weight"), inter, h);
    lw.moe.shared_experts.up_proj =
        OwnProjectionBf16(gguf, Blk(l, "ffn_up_shexp.weight"), inter, h);
    lw.moe.shared_experts.down_proj =
        OwnProjectionBf16(gguf, Blk(l, "ffn_down_shexp.weight"), h, inter);
  }
  return w;
}

}  // namespace vllm
