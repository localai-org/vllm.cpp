// vllm.cpp ORIGINAL — the `flint` GGUF arm. See the header and
// .agents/specs/flint-gguf-arm.md (MODEL-TEXT-FLINT wave 2, issue
// ISSUE-LOCAL-01M5PRSC2E4VR54HBSWFPMN37E).
//
// The published RecursiveSI flint containers ARE llama.cpp-conformant
// (expert_gating_func = 2 = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID,
// src/llama-hparams.h:19 of the pinned llama-cpp-flint oracle), so unlike the
// kolibri1 container llama.cpp here IS a runnable floor and supplies the
// golden greedy chains. The load expands every k-quant to bf16 and fills the
// SAME FlintWeights the safetensors loader (flint_weights.cpp) produces, so
// the production forward (flint.cpp FlintModel::Forward) is unchanged.
//
// ORIENTATION. Every flint projection is consumed as Matmul-B [K=in, N=out]
// (flint.h FlintLinear). GGUF rows come out of the reader torch row-major
// [out, in] (gguf_reader.cpp), the TRANSPOSE of what flint consumes — the
// kolibri1 loader needed no transpose because its forward consumed [out, in];
// this one transposes every 2-D projection at the load.
#include "vllm/model_executor/models/flint_gguf_weights.h"

#include <algorithm>
#include <cstring>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
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
  VT_CHECK(o.rank <= vt::kMaxRank, "flint gguf: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

// Dequant a whole tensor to bf16 (row-major [out, in] as the reader lays it
// out), then own the bytes under `shape`.
OwnedTensor OwnBf16(const GgufFile& g, const std::string& name,
                    const std::vector<int64_t>& shape) {
  const GgufTensorInfo& t = g.Get(name);
  std::vector<uint16_t> dq =
      DequantGgufRowToBf16(t.ggml_type, t.data, ShapeNumel(t.shape), 1.0F);
  VT_CHECK(ShapeNumel(shape) == static_cast<int64_t>(dq.size()),
           "flint gguf: element-count mismatch for " + name);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  std::memcpy(o.bytes.data(), dq.data(), dq.size() * sizeof(uint16_t));
  return o;
}

// Transpose helper: `dq` is row-major [rows, cols]; the owned tensor is
// [cols, rows].
OwnedTensor OwnTransposedBf16(const std::vector<uint16_t>& dq, int64_t rows,
                              int64_t cols) {
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {cols, rows});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (int64_t r = 0; r < rows; ++r) {
    for (int64_t c = 0; c < cols; ++c) {
      dst[c * rows + r] =
          dq[static_cast<size_t>(r) * static_cast<size_t>(cols) +
             static_cast<size_t>(c)];
    }
  }
  return o;
}

// One 2-D projection: dequant, verify the geometry, transpose [out, in] ->
// [in, out]. A wrong geometry is a loud failure at the load.
FlintLinear OwnProjectionBf16(const GgufFile& g, const std::string& name,
                              int64_t in, int64_t out) {
  const GgufTensorInfo& t = g.Get(name);
  VT_CHECK(t.shape.size() == 2 && t.shape[0] == out && t.shape[1] == in,
           "flint gguf: " + name + " is [" +
               std::to_string(t.shape.empty() ? -1 : t.shape[0]) + ", " +
               std::to_string(t.shape.size() < 2 ? -1 : t.shape[1]) +
               "], expected [" + std::to_string(out) + ", " +
               std::to_string(in) + "]");
  std::vector<uint16_t> dq =
      DequantGgufRowToBf16(t.ggml_type, t.data, out * in, 1.0F);
  FlintLinear l;
  l.w = OwnTransposedBf16(dq, out, in);
  return l;
}

// One routed expert cut out of an ALREADY DEQUANTIZED stacked tower
// [E, out, in], then transposed to [in, out]. The tower is dequantized ONCE
// per layer (below); slicing per expert out of a fresh dequant would re-run
// the whole-tower decode 256 times per layer — the shape of defect that turns
// a two-minute load into a two-hour one (the kolibri1 loader's OwnBf16RowSlice
// note).
FlintLinear OwnTowerSliceBf16(const std::vector<uint16_t>& dq,
                              const GgufTensorInfo& t, int64_t e, int64_t in,
                              int64_t out) {
  VT_CHECK(t.shape.size() == 3,
           "flint gguf: " + t.name + " is not the stacked expert tower");
  const int64_t total_rows = t.shape[0] * t.shape[1];  // [E, out] flattened
  VT_CHECK(t.shape[2] == in && e * out + out <= total_rows,
           "flint gguf: expert slice out of range for " + t.name);
  std::vector<uint16_t> slice(
      dq.begin() + static_cast<size_t>(e * out) * static_cast<size_t>(in),
      dq.begin() + static_cast<size_t>((e + 1) * out) * static_cast<size_t>(in));
  FlintLinear l;
  l.w = OwnTransposedBf16(slice, out, in);
  return l;
}

// One 2-D projection for the KEEP-QUANT arm: a block-carrying tensor keeps
// its VERBATIM GGUF block bytes in [N=out, K=in] nk order (the disk order,
// which IS the MatmulBTQuant orientation — no transpose); a float-carrying
// tensor (F32/F16/BF16 — there is no block to keep) takes the bf16 arm
// exactly as before. A block encoding the Tenstorrent keep-quant kernel set
// cannot decode refuses BY NAME (encoding + tensor) rather than silently
// expanding — the pre-existing expand path would resurrect the 70.5 GiB
// streaming tier this arm exists to remove, with the model resident
// (gguf_keep_quant.cpp DeviceKeepQuantSupported's own rationale).
FlintLinear OwnProjectionKeepQuant(const GgufFile& g, const std::string& name,
                                   int64_t in, int64_t out) {
  const GgufTensorInfo& t = g.Get(name);
  VT_CHECK(t.shape.size() == 2 && t.shape[0] == out && t.shape[1] == in,
           "flint gguf: " + name + " is [" +
               std::to_string(t.shape.empty() ? -1 : t.shape[0]) + ", " +
               std::to_string(t.shape.size() < 2 ? -1 : t.shape[1]) +
               "], expected [" + std::to_string(out) + ", " +
               std::to_string(in) + "]");
  FlintLinear l;
  vt::DType dt = vt::DType::kF32;
  if (!vt::BlockDTypeFromGgmlTypeId(t.ggml_type, &dt)) {
    l = OwnProjectionBf16(g, name, in, out);
    return l;
  }
  VT_CHECK(KeepQuantDType(t.ggml_type, &dt) &&
               DeviceKeepQuantSupported(dt, vt::DeviceType::kTENSTORRENT),
           "flint gguf keep-quant: " + name + " carries " +
               std::string(vt::Name(dt)) +
               ", which the Tenstorrent keep-quant kernel set does not decode "
               "(gguf_keep_quant.cpp DeviceKeepQuantSupported, kTENSTORRENT); "
               "the arm refuses by name rather than silently expanding to the "
               "70.5 GiB bf16 streaming tier");
  VT_CHECK(t.shape[1] % vt::BlockElems(dt) == 0,
           "flint gguf keep-quant: " + name +
               " K is not a whole number of " + std::string(vt::Name(dt)) +
               " blocks");
  const size_t row_bytes = vt::RowSizeBytes(dt, in);
  VT_CHECK(static_cast<int64_t>(row_bytes) * out <=
               static_cast<int64_t>(t.nbytes),
           "flint gguf keep-quant: " + name + " block span exceeds the tensor");
  l.w.dtype = dt;
  l.w.rank = 2;
  l.w.shape[0] = out;
  l.w.shape[1] = in;
  l.w.nk = true;
  l.w.bytes.resize(static_cast<size_t>(row_bytes) *
                   static_cast<size_t>(out));
  std::memcpy(l.w.bytes.data(), t.data, l.w.bytes.size());
  return l;
}

// One routed expert cut out of the KEEP-QUANT arm's stacked tower. Whole
// rows of the same K per expert, so the cut is block-aligned by
// construction (experts tile the OUT dimension). A float-carrying tower
// takes the dequant-once + slice bf16 path unchanged.
FlintLinear OwnTowerKeepQuant(const GgufFile& g, const std::string& name,
                              int64_t e, int64_t in, int64_t out) {
  const GgufTensorInfo& t = g.Get(name);
  vt::DType dt = vt::DType::kF32;
  if (!vt::BlockDTypeFromGgmlTypeId(t.ggml_type, &dt))
    return OwnTowerSliceBf16(
        DequantGgufRowToBf16(t.ggml_type, t.data, ShapeNumel(t.shape), 1.0F),
        t, e, in, out);
  // Admit ONCE per tower (the same verdict every expert slice inherits);
  // the per-slice geometry check keeps every cut loud.
  VT_CHECK(KeepQuantDType(t.ggml_type, &dt) &&
               DeviceKeepQuantSupported(dt, vt::DeviceType::kTENSTORRENT),
           "flint gguf keep-quant: " + name + " carries " +
               std::string(vt::Name(dt)) +
               ", which the Tenstorrent keep-quant kernel set does not decode "
               "(gguf_keep_quant.cpp DeviceKeepQuantSupported, kTENSTORRENT); "
               "the arm refuses by name rather than silently expanding to the "
               "70.5 GiB bf16 streaming tier");
  VT_CHECK(t.shape.size() == 3 && t.shape[2] == in,
           "flint gguf keep-quant: " + name + " is not the [E, out, in] tower (shape=[" +
               std::to_string(t.shape.size() > 0 ? t.shape[0] : -1) + "," +
               std::to_string(t.shape.size() > 1 ? t.shape[1] : -1) + "," +
               std::to_string(t.shape.size() > 2 ? t.shape[2] : -1) + "], in=" +
               std::to_string(in) + ", out=" + std::to_string(out) + ")");
  VT_CHECK((e + 1) * out <= t.shape[0] * t.shape[1],
           "flint gguf keep-quant: expert slice out of range for " + name);
  VT_CHECK(in % vt::BlockElems(dt) == 0,
           "flint gguf keep-quant: " + name +
               " K is not a whole number of " + std::string(vt::Name(dt)) +
               " blocks");
  const size_t row_bytes = vt::RowSizeBytes(dt, in);
  FlintLinear l;
  l.w.dtype = dt;
  l.w.rank = 2;
  l.w.shape[0] = out;
  l.w.shape[1] = in;
  l.w.nk = true;
  l.w.bytes.resize(static_cast<size_t>(row_bytes) *
                   static_cast<size_t>(out));
  std::memcpy(l.w.bytes.data(),
              t.data + static_cast<size_t>(e * out) * row_bytes,
              l.w.bytes.size());
  return l;
}

// An RMS-norm gamma: F32 on disk in these containers, converted to the bf16
// the forward consumes (the safetensors arm ships the gammas as BF16
// directly).
OwnedTensor OwnNormBf16(const GgufFile& g, const std::string& name,
                        int64_t dim) {
  const GgufTensorInfo& t = g.Get(name);
  VT_CHECK(t.ggml_type == 0,  // GGML_TYPE_F32
           "flint gguf: " + name + " is not F32");
  VT_CHECK(t.shape.size() == 1 && t.shape[0] == dim,
           "flint gguf: " + name + " is not [" + std::to_string(dim) + "]");
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {dim});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  const auto* src = reinterpret_cast<const float*>(t.data);
  for (int64_t i = 0; i < dim; ++i) dst[i] = vt::F32ToBF16(src[i]);
  return o;
}

// The per-head attention sinks (`blk.N.attn_sinks.weight`): F32
// [num_heads] on disk, the GGUF spelling of the checkpoint's
// `self_attn.softmax_offset` (the converter appends `.weight`).
OwnedTensor OwnSinksBf16(const GgufFile& g, const std::string& name,
                         int64_t heads) {
  const GgufTensorInfo& t = g.Get(name);
  VT_CHECK(t.ggml_type == 0 && t.shape.size() == 1 && t.shape[0] == heads,
           "flint gguf: " + name + " is not F32 [" + std::to_string(heads) +
               "]");
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {heads});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  const auto* src = reinterpret_cast<const float*>(t.data);
  for (int64_t i = 0; i < heads; ++i) dst[i] = vt::F32ToBF16(src[i]);
  return o;
}

std::string Blk(int64_t il, const char* suffix) {
  return "blk." + std::to_string(il) + "." + suffix;
}

// Small KV readers, the kolibri1_gguf_weights.cpp shape.
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
      throw std::runtime_error("flint gguf: key " + key +
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
  VT_CHECK(v != nullptr, "flint gguf: missing metadata key " + key);
  return KvInt(*v, key);
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t fallback) {
  const GgufValue* v = g.FindKv(key);
  return v == nullptr ? fallback : KvInt(*v, key);
}

double ReqFloat(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "flint gguf: missing metadata key " + key);
  return KvFloat(*v, key);
}

// A bool-array KV (`flint.attention.{sliding_window,rope}_pattern`), as one
// 0/1 byte per layer.
std::vector<uint8_t> ReqBoolArray(const GgufFile& g, const std::string& key,
                                  int64_t layers) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr && v->TypeId() == kGgufArray,
           "flint gguf: missing metadata key " + key +
               " — the hybrid layer schedule cannot be derived");
  const GgufArray& arr = std::get<GgufArray>(v->v);
  VT_CHECK(static_cast<int64_t>(arr.elems.size()) == layers,
           "flint gguf: " + key + " length must equal the block count");
  std::vector<uint8_t> out;
  out.reserve(arr.elems.size());
  for (const GgufValue& e : arr.elems)
    out.push_back(KvInt(e, key) != 0 ? 1 : 0);
  return out;
}

}  // namespace

bool IsFlintGguf(const GgufFile& gguf) {
  const GgufValue* v = gguf.FindKv("general.architecture");
  return v != nullptr && v->TypeId() == kGgufString &&
         std::get<std::string>(v->v) == kFlintGgufArch;
}

std::vector<FlintGgufTensor> EnumerateFlintGgufTensors(const FlintParams& p) {
  std::vector<FlintGgufTensor> out;
  out.reserve(static_cast<size_t>(3 + p.num_hidden_layers * 10));
  out.push_back({"token_embd.weight", "embedding"});
  out.push_back({"output.weight", "lm_head"});
  out.push_back({"output_norm.weight", "final_norm"});
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    out.push_back({Blk(l, "attn_norm.weight"), "layer"});
    out.push_back({Blk(l, "ffn_norm.weight"), "layer"});
    out.push_back({Blk(l, "attn_q.weight"), "layer"});
    out.push_back({Blk(l, "attn_k.weight"), "layer"});
    out.push_back({Blk(l, "attn_v.weight"), "layer"});
    out.push_back({Blk(l, "attn_output.weight"), "layer"});
    out.push_back({Blk(l, "attn_q_norm.weight"), "layer"});
    out.push_back({Blk(l, "attn_k_norm.weight"), "layer"});
    out.push_back({Blk(l, "attn_sinks.weight"), "layer"});
    if (p.IsDenseLayer(l)) {
      out.push_back({Blk(l, "ffn_gate.weight"), "layer"});
      out.push_back({Blk(l, "ffn_up.weight"), "layer"});
      out.push_back({Blk(l, "ffn_down.weight"), "layer"});
    } else {
      out.push_back({Blk(l, "ffn_gate_inp.weight"), "layer"});
      out.push_back({Blk(l, "ffn_gate_exps.weight"), "layer"});
      out.push_back({Blk(l, "ffn_up_exps.weight"), "layer"});
      out.push_back({Blk(l, "ffn_down_exps.weight"), "layer"});
      out.push_back({Blk(l, "ffn_gate_shexp.weight"), "layer"});
      out.push_back({Blk(l, "ffn_up_shexp.weight"), "layer"});
      out.push_back({Blk(l, "ffn_down_shexp.weight"), "layer"});
    }
  }
  return out;
}

HfConfig FlintHfConfigFromGguf(const GgufFile& gguf) {
  VT_CHECK(IsFlintGguf(gguf),
           "flint gguf: general.architecture must be '" +
               std::string(kFlintGgufArch) + "'");
  const std::string p = "flint.";

  HfConfig c;
  c.model_type = "flint";
  c.architectures = {"FlintForCausalLM"};
  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_hidden_layers = ReqInt(gguf, p + "block_count");
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads = ReqInt(gguf, p + "attention.head_count_kv");
  c.vocab_size = gguf.Get("token_embd.weight").shape[0];
  // The context length drives max_model_len and with it every runner-side
  // buffer the engine sizes from it (InputBatch's per-request token rows);
  // leaving it unset meant max_model_len 0 and a segfault on the first
  // add_request of any FromModelDir serve (ISSUE-LOCAL-01M5YRQ1SWC93MDG52QZGDFBY4).
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 0);

  // --- architecture-specific numbers, under HF's OWN key names -------------
  // ParseFlintParams reads `raw`; the names are the checkpoint's config.json
  // spellings (the same ones W1's parser validates), NOT names invented here.
  nlohmann::json raw = nlohmann::json::object();
  raw["model_type"] = c.model_type;
  raw["architectures"] = c.architectures;
  raw["head_dim"] = ReqInt(gguf, p + "attention.key_length");
  raw["sliding_window"] = ReqInt(gguf, p + "attention.sliding_window");
  raw["rope_theta"] = ReqFloat(gguf, p + "rope.freq_base");
  raw["rms_norm_eps"] =
      ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  raw["hidden_act"] = "silu";
  raw["tie_word_embeddings"] = !HasTensor(gguf, "output.weight");
  raw["num_experts"] = ReqInt(gguf, p + "expert_count");
  raw["num_experts_per_tok"] = ReqInt(gguf, p + "expert_used_count");
  raw["moe_intermediate_size"] =
      ReqInt(gguf, p + "expert_feed_forward_length");
  raw["shared_expert_intermediate_size"] =
      ReqInt(gguf, p + "expert_shared_feed_forward_length");

  // Read and REQUIRED — the container must state the routing it was written
  // with. 2 = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID, the value the
  // architecture defines (the forward's sigmoid top-k renormalised router);
  // anything else is a different routing and is refused by name rather than
  // silently run through sigmoid. expert_weights_norm=true is the
  // renormalisation the forward always applies, and the scale must be the
  // published 1.0 — both would need a forward path that does not exist.
  VT_CHECK(ReqInt(gguf, p + "expert_gating_func") == 2,
           "flint gguf: expert_gating_func must be 2 (sigmoid) — the "
           "container states a different routing than the architecture "
           "defines");
  VT_CHECK(ReqInt(gguf, p + "expert_weights_norm") != 0,
           "flint gguf: expert_weights_norm must be true — the forward "
           "renormalises the sigmoid top-k weights unconditionally");
  VT_CHECK(ReqFloat(gguf, p + "expert_weights_scale") == 1.0,
           "flint gguf: expert_weights_scale must be 1.0");

  // The layer schedule comes off the file's own arrays: sliding true = a
  // sliding layer, and the rope array must AGREE with it (the converter
  // writes the same schedule in both; a disagreement means the file means
  // something this mapping does not understand, which is a loud failure, not
  // a guess). rope_pattern false = full-attention = a no-RoPE layer — exactly
  // the invariant ParseFlintParams already validates.
  const std::vector<uint8_t> sliding =
      ReqBoolArray(gguf, p + "attention.sliding_window_pattern",
                   c.num_hidden_layers);
  const std::vector<uint8_t> rope =
      ReqBoolArray(gguf, p + "attention.rope_pattern", c.num_hidden_layers);
  VT_CHECK(sliding == rope,
           "flint gguf: attention.rope_pattern and "
           "attention.sliding_window_pattern disagree — the layer schedule "
           "cannot be derived");
  nlohmann::json layer_types = nlohmann::json::array();
  nlohmann::json no_rope = nlohmann::json::array();
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    layer_types.push_back(sliding[static_cast<size_t>(l)] != 0
                              ? "sliding_attention"
                              : "full_attention");
    if (rope[static_cast<size_t>(l)] == 0)
      no_rope.push_back(l);
  }
  raw["layer_types"] = std::move(layer_types);
  raw["no_rope_layer_indices"] = std::move(no_rope);

  // Layers 0..n-1 dense, from the file's own leading_dense_block_count — the
  // converter refuses a non-leading set and so does the parser.
  const int64_t dense = ReqInt(gguf, p + "leading_dense_block_count");
  nlohmann::json mlp_only = nlohmann::json::array();
  for (int64_t l = 0; l < dense; ++l) mlp_only.push_back(l);
  raw["mlp_only_layers"] = std::move(mlp_only);

  c.raw = std::move(raw);
  return c;
}

FlintWeights LoadFlintFromGgufImpl(const GgufFile& gguf,
                                   const HfConfig& config, bool keep_quant) {
  VT_CHECK(IsFlintGguf(gguf),
           "flint gguf: general.architecture must be '" +
               std::string(kFlintGgufArch) + "'");
  const FlintParams p = ParseFlintParams(config);

  // STRUCTURAL accounting, one direction loud and one loud here: a tensor the
  // loaders ask for and the file lacks refuses at `GgufFile::Get` by name; a
  // tensor the file carries and nothing reads refuses below.
  const std::vector<FlintGgufTensor> expected = EnumerateFlintGgufTensors(p);
  std::set<std::string> present;
  for (const GgufTensorInfo& info : gguf.Tensors()) present.insert(info.name);
  std::vector<std::string> unaccounted;
  for (const std::string& n : present) {
    bool found = false;
    for (const FlintGgufTensor& t : expected) found = found || t.name == n;
    if (!found) unaccounted.push_back(n);
  }
  VT_CHECK(unaccounted.empty(),
           "flint gguf: unaccounted tensor(s), first: " +
               (unaccounted.empty() ? std::string("(none)")
                                    : unaccounted.front()));

  const int64_t h = p.hidden_size;
  const int64_t v = p.vocab_size;
  const int64_t kv = p.num_key_value_heads * p.head_dim;
  const int64_t q = p.num_attention_heads * p.head_dim;
  const int64_t inter = p.moe_intermediate_size;
  const int64_t shared_inter = p.shared_expert_intermediate_size;
  // The DENSE intermediate (layers 0-1): the file's own feed_forward_length
  // (6144 on every published container). The safetensors arm took the shapes
  // off the shards; the GGUF arm reads the file's KV and lets the geometry
  // checks below fail loudly on a mismatch.
  const int64_t dense_inter = ReqInt(gguf, "flint.feed_forward_length");

  FlintWeights w;
  w.params = p;
  w.embed_tokens = OwnBf16(gguf, "token_embd.weight", {v, h});
  if (keep_quant) {
    // The untied head in the keep-quant arm: the CONSUMER dispatches per
    // dtype (flint_tt.cpp MatmulWeight), so a block-carrying head keeps its
    // [V, H] disk order and a float-carrying head takes the bf16 transpose.
    w.lm_head = OwnProjectionKeepQuant(gguf, "output.weight", h, v).w;
  } else {
    // The untied head, CONSUMED [H, V] Matmul-B (flint.h: lm_head
    // "[H, vocab] Matmul-B, untied" — the safetensors arm's
    // LoadBf16Transposed orientation). Stored [V, H] it would be read
    // transposed by the forward and corrupt the heap.
    const GgufTensorInfo& r = gguf.Get("output.weight");
    VT_CHECK(r.shape.size() == 2 && r.shape[0] == v && r.shape[1] == h,
             "flint gguf: output.weight is not [vocab, hidden]");
    std::vector<uint16_t> dq =
        DequantGgufRowToBf16(r.ggml_type, r.data, ShapeNumel(r.shape), 1.0F);
    w.lm_head = OwnTransposedBf16(dq, v, h);
  }
  w.final_norm = OwnNormBf16(gguf, "output_norm.weight", h);

  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    FlintLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.input_layernorm = OwnNormBf16(gguf, Blk(l, "attn_norm.weight"), h);
    lw.post_attention_layernorm =
        OwnNormBf16(gguf, Blk(l, "ffn_norm.weight"), h);
    if (keep_quant) {
      lw.attn.q_proj =
          OwnProjectionKeepQuant(gguf, Blk(l, "attn_q.weight"), h, q);
      lw.attn.k_proj =
          OwnProjectionKeepQuant(gguf, Blk(l, "attn_k.weight"), h, kv);
      lw.attn.v_proj =
          OwnProjectionKeepQuant(gguf, Blk(l, "attn_v.weight"), h, kv);
      lw.attn.o_proj =
          OwnProjectionKeepQuant(gguf, Blk(l, "attn_output.weight"), q, h);
    } else {
      lw.attn.q_proj = OwnProjectionBf16(gguf, Blk(l, "attn_q.weight"), h, q);
      lw.attn.k_proj = OwnProjectionBf16(gguf, Blk(l, "attn_k.weight"), h, kv);
      lw.attn.v_proj = OwnProjectionBf16(gguf, Blk(l, "attn_v.weight"), h, kv);
      lw.attn.o_proj =
          OwnProjectionBf16(gguf, Blk(l, "attn_output.weight"), q, h);
    }
    lw.attn.q_norm =
        OwnNormBf16(gguf, Blk(l, "attn_q_norm.weight"), p.head_dim);
    lw.attn.k_norm =
        OwnNormBf16(gguf, Blk(l, "attn_k_norm.weight"), p.head_dim);
    lw.attn.sinks = OwnSinksBf16(gguf, Blk(l, "attn_sinks.weight"),
                                 p.num_attention_heads);

    if (p.IsDenseLayer(l)) {
      if (keep_quant) {
        lw.dense_mlp.gate_proj = OwnProjectionKeepQuant(
            gguf, Blk(l, "ffn_gate.weight"), h, dense_inter);
        lw.dense_mlp.up_proj =
            OwnProjectionKeepQuant(gguf, Blk(l, "ffn_up.weight"), h, dense_inter);
        lw.dense_mlp.down_proj = OwnProjectionKeepQuant(
            gguf, Blk(l, "ffn_down.weight"), dense_inter, h);
      } else {
        lw.dense_mlp.gate_proj =
            OwnProjectionBf16(gguf, Blk(l, "ffn_gate.weight"), h, dense_inter);
        lw.dense_mlp.up_proj =
            OwnProjectionBf16(gguf, Blk(l, "ffn_up.weight"), h, dense_inter);
        lw.dense_mlp.down_proj =
            OwnProjectionBf16(gguf, Blk(l, "ffn_down.weight"), dense_inter, h);
      }
    } else {
      // Router [E, H] on disk, consumed [H, E] (the safetensors arm's
      // LoadBf16Transposed orientation).
      if (keep_quant) {
        lw.moe.router_gate =
            OwnProjectionKeepQuant(gguf, Blk(l, "ffn_gate_inp.weight"), h,
                                   p.num_experts)
                .w;  // the router is f32/bf16 on disk: the KeepQuant helper
                     // falls back to the bf16 arm for it (non-block dtype)
      } else {
        const GgufTensorInfo& r = gguf.Get(Blk(l, "ffn_gate_inp.weight"));
        VT_CHECK(r.shape.size() == 2 && r.shape[0] == p.num_experts &&
                     r.shape[1] == h,
                 "flint gguf: ffn_gate_inp.weight is not [num_experts, "
                 "hidden]");
        std::vector<uint16_t> rdq = DequantGgufRowToBf16(
            r.ggml_type, r.data, p.num_experts * h, 1.0F);
        lw.moe.router_gate = OwnTransposedBf16(rdq, p.num_experts, h);
      }

      if (keep_quant) {
        lw.moe.experts.resize(static_cast<size_t>(p.num_experts));
        for (int64_t e = 0; e < p.num_experts; ++e) {
          FlintExpertWeights& x = lw.moe.experts[static_cast<size_t>(e)];
          x.gate_proj = OwnTowerKeepQuant(gguf, Blk(l, "ffn_gate_exps.weight"),
                                          e, h, inter);
          x.up_proj =
              OwnTowerKeepQuant(gguf, Blk(l, "ffn_up_exps.weight"), e, h, inter);
          x.down_proj = OwnTowerKeepQuant(gguf, Blk(l, "ffn_down_exps.weight"),
                                          e, inter, h);
        }
        lw.moe.shared_expert.gate_proj = OwnProjectionKeepQuant(
            gguf, Blk(l, "ffn_gate_shexp.weight"), h, shared_inter);
        lw.moe.shared_expert.up_proj = OwnProjectionKeepQuant(
            gguf, Blk(l, "ffn_up_shexp.weight"), h, shared_inter);
        lw.moe.shared_expert.down_proj = OwnProjectionKeepQuant(
            gguf, Blk(l, "ffn_down_shexp.weight"), shared_inter, h);
      } else {
      // Dequantize each stacked tower ONCE for the layer, then cut + transpose
      // the 256 experts out of the resident decode (see OwnTowerSliceBf16).
      const GgufTensorInfo& gate = gguf.Get(Blk(l, "ffn_gate_exps.weight"));
      const GgufTensorInfo& up = gguf.Get(Blk(l, "ffn_up_exps.weight"));
      const GgufTensorInfo& down = gguf.Get(Blk(l, "ffn_down_exps.weight"));
      std::vector<uint16_t> gate_dq = DequantGgufRowToBf16(
          gate.ggml_type, gate.data, ShapeNumel(gate.shape), 1.0F);
      std::vector<uint16_t> up_dq = DequantGgufRowToBf16(
          up.ggml_type, up.data, ShapeNumel(up.shape), 1.0F);
      std::vector<uint16_t> down_dq = DequantGgufRowToBf16(
          down.ggml_type, down.data, ShapeNumel(down.shape), 1.0F);
      lw.moe.experts.resize(static_cast<size_t>(p.num_experts));
      for (int64_t e = 0; e < p.num_experts; ++e) {
        FlintExpertWeights& x = lw.moe.experts[static_cast<size_t>(e)];
        x.gate_proj = OwnTowerSliceBf16(gate_dq, gate, e, h, inter);
        x.up_proj = OwnTowerSliceBf16(up_dq, up, e, h, inter);
        x.down_proj = OwnTowerSliceBf16(down_dq, down, e, inter, h);
      }
      lw.moe.shared_expert.gate_proj =
          OwnProjectionBf16(gguf, Blk(l, "ffn_gate_shexp.weight"), h, shared_inter);
      lw.moe.shared_expert.up_proj =
          OwnProjectionBf16(gguf, Blk(l, "ffn_up_shexp.weight"), h, shared_inter);
      lw.moe.shared_expert.down_proj =
          OwnProjectionBf16(gguf, Blk(l, "ffn_down_shexp.weight"), shared_inter, h);
      }
    }
  }
  return w;
}

FlintWeights LoadFlintFromGguf(const GgufFile& gguf, const HfConfig& config) {
  return LoadFlintFromGgufImpl(gguf, config, /*keep_quant=*/false);
}

FlintWeights LoadFlintKeepQuantFromGguf(const GgufFile& gguf,
                                        const HfConfig& config) {
  // THE ARM IS OPT-IN: VT_FLINT_KEEP_QUANT=1 admits the keep-quant arm;
  // unset (or 0) restores the bf16 arm byte-for-byte — the production
  // entry self-gates so no caller can take the new arm by accident.
  const char* e = std::getenv("VT_FLINT_KEEP_QUANT");
  if (e == nullptr || e[0] != '1')
    return LoadFlintFromGgufImpl(gguf, config, /*keep_quant=*/false);
  return LoadFlintFromGgufImpl(gguf, config, /*keep_quant=*/true);
}

}  // namespace vllm
