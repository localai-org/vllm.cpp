// vllm.cpp ORIGINAL; see qwen3_gguf_weights.h. GGUF tensor names + metadata
// keys mirror llama.cpp @ b10451 (qwen3: src/llama-arch.cpp). The convert-time
// value transforms this file INVERTS are NONE — Qwen3 does NOT shift norm
// weights (unlike Qwen3.5/3Next which add w+1), confirmed from
// conversion/qwen.py: Qwen3Model inherits Qwen2Model -> TextModel, whose base
// modify_tensors does no norm modification.
//
// The infrastructure helpers below are duplicated from qwen3_5_gguf_weights.cpp
// because they are file-local statics there (OwnBf16, OwnMatmulWeight, etc.).
// The public API they call (OwnGgufQuantBlocks, OwnGgufF16, DequantGgufRowToBf16,
// GgufLoadPolicy::Route, etc.) is shared from the headers.
#include "vllm/model_executor/models/qwen3_gguf_weights.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

#include "vllm/config/weight_residency.h"
#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"  // OwnGgufQuantBlocks, OwnGgufF16
#include "vt/dtype.h"

namespace vllm {

namespace {

// --- small helpers (duplicated from qwen3_5_gguf_weights.cpp) ------------

// Independent opt-out for the read-once page release. Default ON with mmap
// residency. Kept separate so the page-drop can be A/B'd against the mmap
// borrow alone.
bool EnvReleaseExpandedPages() {
  const char* v = std::getenv("VT_GGUF_RELEASE_PAGES");
  if (v == nullptr) return true;
  return !(std::strcmp(v, "") == 0 || std::strcmp(v, "0") == 0 ||
           std::strcmp(v, "false") == 0 || std::strcmp(v, "off") == 0);
}

// Scopes the file's read-once-page release to ONE load, so a GgufFile that
// outlives the loader never carries a stale residency policy.
class GgufPageReleaseScope {
 public:
  GgufPageReleaseScope(const GgufFile& g, bool on)
      : g_(g), prev_(g.releases_expanded_pages()) {
    g_.ReleaseExpandedPages(on);
  }
  ~GgufPageReleaseScope() { g_.ReleaseExpandedPages(prev_); }
  GgufPageReleaseScope(const GgufPageReleaseScope&) = delete;
  GgufPageReleaseScope& operator=(const GgufPageReleaseScope&) = delete;

 private:
  const GgufFile& g_;
  bool prev_;
};

// The file to borrow kept blocks from, or null to copy them.
const GgufFile* MmapSrc(const GgufFile& g, const GgufLoadPolicy& pol) {
  return pol.mmap_residency ? &g : nullptr;
}

// Keep a verbatim 2-D weight slice resident — quant blocks or native f16 — in
// [N, K] nk=true order. Precondition: the policy already routed this (tensor,
// role) to a KEEP residency.
OwnedTensor OwnGgufKeptSlice(const GgufFile& g, const GgufLoadPolicy& pol,
                              const GgufTensorInfo& t, GgufResidency r, int64_t n,
                              int64_t k, int64_t row_offset) {
  if (r == GgufResidency::kKeepQuant) {
    return OwnGgufQuantBlocks(t, n, k, row_offset, MmapSrc(g, pol),
                              pol.quant_repack, /*cuda_align=*/false,
                              pol.prefault);
  }
  VT_CHECK(r == GgufResidency::kKeepF16,
           "qwen3 gguf: OwnGgufKeptSlice called for a non-keep residency on " +
               t.name);
  return OwnGgufF16(t, n, k, row_offset, MmapSrc(g, pol), /*nk=*/true,
                     pol.elem_kn_repack, pol.prefault, pol.weight_value_dtype);
}

bool HasTensor(const GgufFile& g, const std::string& name) {
  for (const GgufTensorInfo& t : g.Tensors()) {
    if (t.name == name) return true;
  }
  return false;
}

int64_t ShapeNumel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

OwnedTensor MakeOwned(vt::DType dt, const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = dt;
  o.rank = static_cast<int>(shape.size());
  VT_CHECK(o.rank <= vt::kMaxRank, "qwen3 gguf: rank exceeds kMaxRank");
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  o.bytes.resize(static_cast<size_t>(n) * vt::SizeOf(dt));
  return o;
}

// src bf16 [rows, cols] -> dst bf16 [cols, rows].
void TransposeBf16(const uint16_t* src, int64_t rows, int64_t cols,
                    uint16_t* dst) {
  for (int64_t r = 0; r < rows; ++r) {
    const uint16_t* src_row = src + r * cols;
    for (int64_t c = 0; c < cols; ++c) dst[c * rows + r] = src_row[c];
  }
}

// The per-tensor scalars an encoding keeps OUTSIDE its blocks (NVFP4 type 40).
// A 2-D weight has ONE scalar; the result is {1.0f} for a self-contained
// encoding.
std::vector<float> GgufNvfp4SidecarScalars(const GgufFile& g,
                                            const std::string& name,
                                            const GgufTensorInfo& t,
                                            const std::string& suffix) {
  const std::string kSuffix = ".weight";
  VT_CHECK(name.size() > kSuffix.size() &&
               name.compare(name.size() - kSuffix.size(), kSuffix.size(),
                            kSuffix) == 0,
           "qwen3 gguf: NVFP4 tensor without a .weight name: " + name);
  const std::string sidecar =
      name.substr(0, name.size() - kSuffix.size()) + suffix;
  VT_CHECK(HasTensor(g, sidecar),
           "qwen3 gguf: " + name + " is NVFP4 but the file has no " +
               sidecar + " sidecar; its blocks alone do not determine the "
                         "weight");
  const GgufTensorInfo& s = g.Get(sidecar);
  VT_CHECK(s.ggml_type == 0, "qwen3 gguf: " + sidecar + " must be F32");
  const int64_t n = ShapeNumel(s.shape);
  VT_CHECK(n == 1 || (t.shape.size() == 3 && n == t.shape[0]),
           "qwen3 gguf: " + sidecar +
               " must hold one scalar, or one per expert");
  std::vector<float> out(static_cast<size_t>(n));
  std::memcpy(out.data(), s.data, static_cast<size_t>(n) * sizeof(float));
  for (float v : out) {
    VT_CHECK(std::isfinite(v) && v > 0.0F,
             "qwen3 gguf: non-positive or non-finite scale in " + sidecar);
  }
  return out;
}

std::vector<float> GgufGlobalScales(const GgufFile& g, const std::string& name,
                                     const GgufTensorInfo& t) {
  if (!GgmlTypeNeedsGlobalScale(t.ggml_type)) return {1.0F};
  return GgufNvfp4SidecarScalars(g, name, t, ".scale");
}

// Dequant a named GGUF tensor (natural [out, in] row-major order, ne0 = in
// fastest). Returns the flat buffer; `info` is filled in.
template <typename T, typename Fn>
std::vector<T> DqSlabs(const GgufFile& g, const std::string& name,
                        const GgufTensorInfo** info, Fn dequant) {
  const GgufTensorInfo& t = g.Get(name);
  *info = &t;
  const int64_t numel = ShapeNumel(t.shape);
  const std::vector<float> scales = GgufGlobalScales(g, name, t);

  std::vector<T> out;
  if (scales.size() == 1) {
    out = dequant(t.ggml_type, t.data, numel, scales[0]);
  } else {
    const int64_t slabs = static_cast<int64_t>(scales.size());
    VT_CHECK(numel % slabs == 0,
             "qwen3 gguf: expert count does not divide " + name);
    const int64_t per = numel / slabs;
    const GgmlTypeTraits& traits = GgmlTraits(t.ggml_type);
    VT_CHECK(per % traits.block_elems == 0,
             "qwen3 gguf: expert slab of " + name + " is not whole blocks");
    const int64_t slab_bytes = per / traits.block_elems * traits.block_bytes;
    out.resize(static_cast<size_t>(numel));
    for (int64_t e = 0; e < slabs; ++e) {
      const std::vector<T> part =
          dequant(t.ggml_type, t.data + e * slab_bytes, per,
                  scales[static_cast<size_t>(e)]);
      std::memcpy(out.data() + e * per, part.data(),
                  static_cast<size_t>(per) * sizeof(T));
    }
  }
  g.DropSpanResidency(t.data, t.nbytes);
  return out;
}

std::vector<uint16_t> DqBf16(const GgufFile& g, const std::string& name,
                              const GgufTensorInfo** info) {
  return DqSlabs<uint16_t>(g, name, info,
                           [](uint32_t ty, const uint8_t* d, int64_t n,
                              float s) {
                             return DequantGgufRowToBf16(ty, d, n, s);
                           });
}

// bf16 tensor copied verbatim with `shape` (dequant, then own the bytes).
OwnedTensor OwnBf16(const GgufFile& g, const std::string& name,
                     const std::vector<int64_t>& shape) {
  const GgufTensorInfo* t = nullptr;
  std::vector<uint16_t> dq = DqBf16(g, name, &t);
  VT_CHECK(ShapeNumel(shape) == static_cast<int64_t>(dq.size()),
           "qwen3 gguf: element-count mismatch for " + name);
  OwnedTensor o = MakeOwned(vt::DType::kBF16, shape);
  std::memcpy(o.bytes.data(), dq.data(), dq.size() * sizeof(uint16_t));
  return o;
}

// bf16 [out, in] -> owned bf16 [in, out] (Matmul-B layout).
OwnedTensor OwnBf16T(const GgufFile& g, const std::string& name) {
  const GgufTensorInfo* t = nullptr;
  std::vector<uint16_t> dq = DqBf16(g, name, &t);
  VT_CHECK(t->shape.size() == 2, "qwen3 gguf: expected 2-D weight " + name);
  const int64_t out_dim = t->shape[0];
  const int64_t in_dim = t->shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {in_dim, out_dim});
  TransposeBf16(dq.data(), out_dim, in_dim,
                reinterpret_cast<uint16_t*>(o.bytes.data()));
  return o;
}

// --- residency routing (L3) ----------------------------------------------
// Every tensor this loader touches passes through GgufLoadPolicy::Route with
// an explicit role, so the policy's audit hook observes the complete tensor
// list and no tensor can reach a residency by omission.

// A 2-D GEMM weight taken verbatim from the file. Three outcomes, in order:
//  1. keep-quant -> raw blocks in the file's [N, K] order (nk = true);
//  2. expand, nk -> bf16 in the file's [N, K] order (nk = true);
//  3. expand, Matmul-B -> dequant to bf16 and transpose to [K, N].
OwnedTensor OwnMatmulWeight(const GgufFile& g, const std::string& name,
                             const GgufLoadPolicy& pol) {
  const GgufTensorInfo& t = g.Get(name);
  const GgufResidency r = pol.Route(t, GgufTensorRole::kMatmulWeight);
  VT_CHECK(r != GgufResidency::kNvfp4Fp4,
           "qwen3 gguf: " + name +
               " routed to the fp4 residency but this call site builds a bf16 "
               "OwnedTensor; use NoNvfp4()");
  if (GgufResidencyKeepsBlockWeights(r)) {
    VT_CHECK(t.shape.size() == 2, "qwen3 gguf: expected 2-D weight " + name);
    return OwnGgufKeptSlice(g, pol, t, r, t.shape[0], t.shape[1], 0);
  }
  if (pol.expand_nk) {
    VT_CHECK(t.shape.size() == 2, "qwen3 gguf: expected 2-D weight " + name);
    OwnedTensor o = OwnBf16(g, name, t.shape);
    o.nk = true;
    return o;
  }
  return OwnBf16T(g, name);
}

// A policy copy with the fp4 residency DISABLED, for call sites whose
// consumer is an OwnedTensor and has no Nvfp4Weight field.
GgufLoadPolicy NoNvfp4(const GgufLoadPolicy& pol) {
  GgufLoadPolicy p = pol;
  p.nvfp4_fp4 = false;
  p.nvfp4_w4a4 = false;
  return p;
}

// Route a tensor that can NEVER keep its blocks and assert the policy agrees.
void RequireExpand(const GgufLoadPolicy& pol, const GgufFile& g,
                    const std::string& name, GgufTensorRole role) {
  VT_CHECK(pol.Route(g.Get(name), role) == GgufResidency::kExpandBf16,
           std::string("qwen3 gguf: a ") + Name(role) +
               " tensor must not keep quant blocks: " + name);
}

// --- embedding + lm_head, with tied-head sharing (L5 bf16 / L6 f16) --------

void LoadEmbedAndHead(const GgufFile& g, const GgufLoadPolicy& pol,
                      OwnedTensor* embed, OwnedTensor* head) {
  const std::string kEmbed = "token_embd.weight";
  const GgufTensorInfo& et = g.Get(kEmbed);
  const GgufResidency embed_r =
      pol.Route(et, GgufTensorRole::kEmbeddingTable);

  if (embed_r == GgufResidency::kKeepQuant) {
    VT_CHECK(et.shape.size() == 2, "qwen3 gguf: token_embd must be 2-D");
    *embed = OwnGgufQuantBlocks(et, et.shape[0], et.shape[1], /*row_offset=*/0,
                                MmapSrc(g, pol), /*repack=*/false,
                                /*cuda_align=*/false, pol.prefault,
                                GgufTensorRole::kEmbeddingTable);
    embed->nk = false;
  } else if (embed_r == GgufResidency::kKeepF16) {
    VT_CHECK(et.shape.size() == 2, "qwen3 gguf: token_embd must be 2-D");
    *embed = OwnGgufF16(et, et.shape[0], et.shape[1], 0, MmapSrc(g, pol),
                        /*nk=*/false, /*elem_kn_repack=*/false, pol.prefault,
                        pol.weight_value_dtype);
  } else {
    VT_CHECK(embed_r == GgufResidency::kExpandBf16,
             "qwen3 gguf: unexpected embedding-table residency " +
                 std::string(Name(embed_r)));
    *embed = OwnBf16(g, kEmbed, et.shape);
  }

  const bool tied = !HasTensor(g, "output.weight");
  const std::string head_name = tied ? kEmbed : "output.weight";
  const GgufTensorInfo& ht = g.Get(head_name);
  const GgufResidency head_r =
      RouteGgufTensor(pol.keep_quant, pol.keep_f16, /*nvfp4_fp4=*/false,
                       pol.cpu_ref, GgufTensorRole::kMatmulWeight, ht.ggml_type,
                       ht.shape, pol.device, pol.weight_value_dtype);

  const bool f16_share = embed_r == GgufResidency::kKeepF16 &&
                          head_r == GgufResidency::kKeepF16;
  const bool bf16_share = embed_r == GgufResidency::kExpandBf16 &&
                          head_r == GgufResidency::kExpandBf16 && pol.expand_nk;
  if (tied && pol.share_tied_head && (f16_share || bf16_share)) {
    (void)NoNvfp4(pol).Route(ht,
                             GgufTensorRole::kMatmulWeight);  // head's 1 audit
    std::shared_ptr<const void> owner = embed->bytes.KeepAlive();
    OwnedTensor h;
    h.dtype = embed->dtype;
    h.weight_value_dtype = embed->weight_value_dtype;
    h.rank = embed->rank;
    for (int i = 0; i < embed->rank; ++i) h.shape[i] = embed->shape[i];
    h.nk = true;
    h.bytes = OwnedBytes::Borrow(embed->bytes.data(), embed->bytes.size(), owner);
    *head = std::move(h);
    return;
  }
  *head = OwnMatmulWeight(g, head_name, NoNvfp4(pol));
}

// --- config helpers -------------------------------------------------------

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
      throw std::runtime_error("qwen3 gguf: key " + key +
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
  VT_CHECK(v != nullptr, "qwen3 gguf: missing metadata key " + key);
  return KvInt(*v, key);
}

double ReqFloat(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "qwen3 gguf: missing metadata key " + key);
  return KvFloat(*v, key);
}

int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v ? KvInt(*v, key) : dflt;
}

std::string Blk(int64_t il, const std::string& suffix) {
  return "blk." + std::to_string(il) + "." + suffix;
}

// Concatenate two 2-D bf16 OwnedTensors along dim 0 (rows). Both must be bf16,
// 2-D, and share the same column count (dim 1). The result is bf16 [rows_a +
// rows_b, cols] with nk inherited from `a` (both are expected to be nk=true
// for merged matmul weights).
OwnedTensor ConcatBf16Rows(const OwnedTensor& a, const OwnedTensor& b) {
  VT_CHECK(a.dtype == vt::DType::kBF16 && b.dtype == vt::DType::kBF16,
           "qwen3 gguf: ConcatBf16Rows requires bf16 tensors");
  VT_CHECK(a.rank == 2 && b.rank == 2,
           "qwen3 gguf: ConcatBf16Rows requires 2-D tensors");
  VT_CHECK(a.shape[1] == b.shape[1],
           "qwen3 gguf: ConcatBf16Rows column mismatch");
  const int64_t rows = a.shape[0] + b.shape[0];
  const int64_t cols = a.shape[1];
  OwnedTensor o = MakeOwned(vt::DType::kBF16, {rows, cols});
  const size_t a_bytes = static_cast<size_t>(a.shape[0]) * cols * sizeof(uint16_t);
  const size_t b_bytes = static_cast<size_t>(b.shape[0]) * cols * sizeof(uint16_t);
  std::memcpy(o.bytes.data(), a.bytes.data(), a_bytes);
  std::memcpy(o.bytes.data() + a_bytes, b.bytes.data(), b_bytes);
  o.nk = a.nk;
  return o;
}

}  // namespace

// --- config ---------------------------------------------------------------

HfConfig Qwen3HfConfigFromGguf(const GgufFile& gguf) {
  const GgufValue* arch_v = gguf.FindKv("general.architecture");
  VT_CHECK(arch_v != nullptr && arch_v->TypeId() == kGgufString,
           "qwen3 gguf: general.architecture must be a string");
  const std::string arch = std::get<std::string>(arch_v->v);
  VT_CHECK(arch == "qwen3",
           "qwen3 gguf: unexpected architecture '" + arch + "'");
  const std::string p = arch + ".";

  HfConfig c;
  c.model_type = arch;
  c.architectures = {"Qwen3ForCausalLM"};

  c.hidden_size = ReqInt(gguf, p + "embedding_length");
  c.num_hidden_layers = ReqInt(gguf, p + "block_count");
  c.num_attention_heads = ReqInt(gguf, p + "attention.head_count");
  c.num_key_value_heads =
      OptInt(gguf, p + "attention.head_count_kv", c.num_attention_heads);
  c.head_dim = OptInt(gguf, p + "attention.key_length",
                      c.num_attention_heads > 0
                          ? c.hidden_size / c.num_attention_heads
                          : 0);
  c.intermediate_size = OptInt(gguf, p + "feed_forward_length", 0);

  // vocab_size: prefer the kv, else read token_embd's leading (out) dim.
  const GgufValue* vocab_kv = gguf.FindKv(p + "vocab_size");
  c.vocab_size = vocab_kv ? KvInt(*vocab_kv, p + "vocab_size")
                          : gguf.Get("token_embd.weight").shape[0];

  // RoPE / norm / context.
  const GgufValue* freq = gguf.FindKv(p + "rope.freq_base");
  c.rope_theta = freq ? KvFloat(*freq, p + "rope.freq_base") : 10000.0;
  // llama.cpp omits rope.dimension_count when it equals head_dim (full rotary).
  // Qwen3 has no partial_rotary_factor, so default to head_dim.
  c.rotary_dim = OptInt(gguf, p + "rope.dimension_count", c.head_dim);
  c.rms_norm_eps = ReqFloat(gguf, p + "attention.layer_norm_rms_epsilon");
  c.max_position_embeddings = OptInt(gguf, p + "context_length", 0);
  c.torch_dtype = "bfloat16";

  // tie_word_embeddings: detected by the ABSENCE of output.weight (llama.cpp
  // TENSOR_DUPLICATED). Published in raw so ParseQwen3ForCausalLMConfig and the
  // loader agree.
  c.raw["tie_word_embeddings"] = !HasTensor(gguf, "output.weight");
  c.raw["attention_bias"] = false;

  return c;
}

bool IsQwen3Gguf(const GgufFile& gguf) {
  const GgufValue* arch_v = gguf.FindKv("general.architecture");
  if (arch_v == nullptr || arch_v->TypeId() != kGgufString) return false;
  return std::get<std::string>(arch_v->v) == "qwen3";
}

// --- weights --------------------------------------------------------------

Qwen3DenseWeights LoadDenseGgufWeights(const GgufFile& gguf,
                                       const HfConfig& config,
                                       const GgufLoadPolicy* policy,
                                       bool keep_projection_parts) {
  const GgufLoadPolicy env_policy = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
  const GgufLoadPolicy& pol = policy != nullptr ? *policy : env_policy;
  // Merged matmul weights (q/k/v -> qkv_proj, gate/up -> gate_up_proj) must
  // expand to bf16 for concatenation. NoKeepQuant narrows the policy so
  // RequireExpand sees kExpandBf16 for Q4_K matmul weights, which would
  // otherwise route to kKeepQuant on CPU. o_proj and down_proj still use the
  // original pol (they may keep quant). Same pattern as laguna_weights.cpp:726
  // and deepseek_v4_weights.cpp:1644.
  const GgufLoadPolicy expand_pol = NoKeepQuant(pol);
  const GgufPageReleaseScope page_release(
      gguf, pol.mmap_residency && EnvReleaseExpandedPages());
  VT_CHECK(config.num_hidden_layers > 0,
           "qwen3 gguf: num_hidden_layers must be > 0");

  Qwen3DenseWeights w;
  w.tie_word_embeddings =
      config.raw.value("tie_word_embeddings", false);
  w.attention_bias = false;

  // Embedding + lm_head (tied-head sharing when applicable).
  LoadEmbedAndHead(gguf, pol, &w.embed_tokens, &w.lm_head);

  // Final norm: OwnBf16 (NO w+1 shift — Qwen3 does not shift norms).
  RequireExpand(expand_pol, gguf, "output_norm.weight",
                GgufTensorRole::kTransformedWeight);
  w.final_norm = OwnBf16(gguf, "output_norm.weight", {config.hidden_size});

  w.layers.reserve(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t il = 0; il < config.num_hidden_layers; ++il) {
    Qwen3DenseLayerWeights layer;

    // Norms: OwnBf16 (NO w+1 shift).
    RequireExpand(expand_pol, gguf, Blk(il, "attn_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.input_layernorm =
        OwnBf16(gguf, Blk(il, "attn_norm.weight"), {config.hidden_size});
    RequireExpand(expand_pol, gguf, Blk(il, "ffn_norm.weight"),
                  GgufTensorRole::kTransformedWeight);
    layer.post_attention_layernorm =
        OwnBf16(gguf, Blk(il, "ffn_norm.weight"),
                {config.hidden_size});

    if (keep_projection_parts) {
      GgufLoadPolicy parts_pol = NoNvfp4(pol);
      parts_pol.expand_nk = true;
      parts_pol.elem_kn_repack = false;
      parts_pol.quant_repack = false;
      for (const auto* part : {"attn_q.weight", "attn_k.weight", "attn_v.weight"})
        layer.attn.qkv_proj_parts.push_back(OwnMatmulWeight(gguf, Blk(il, part), parts_pol));
    } else {
    // Attention: load q/k/v separately, then merge into one qkv_proj.
    // The merged weight is ALWAYS bf16 expand (no keep-quant for merged
    // weights in the initial implementation — the spec defers that).
    // Each source is forced to expand via RequireExpand, then OwnBf16
    // produces bf16 [out, in] in the file's own [N, K] order (nk=true).
    RequireExpand(expand_pol, gguf, Blk(il, "attn_q.weight"),
                  GgufTensorRole::kMatmulWeight);
    OwnedTensor q = OwnBf16(gguf, Blk(il, "attn_q.weight"),
                             gguf.Get(Blk(il, "attn_q.weight")).shape);
    q.nk = true;
    RequireExpand(expand_pol, gguf, Blk(il, "attn_k.weight"),
                  GgufTensorRole::kMatmulWeight);
    OwnedTensor k = OwnBf16(gguf, Blk(il, "attn_k.weight"),
                             gguf.Get(Blk(il, "attn_k.weight")).shape);
    k.nk = true;
    RequireExpand(expand_pol, gguf, Blk(il, "attn_v.weight"),
                  GgufTensorRole::kMatmulWeight);
    OwnedTensor v = OwnBf16(gguf, Blk(il, "attn_v.weight"),
                             gguf.Get(Blk(il, "attn_v.weight")).shape);
    v.nk = true;
    layer.attn.qkv_proj = ConcatBf16Rows(ConcatBf16Rows(q, k), v);

    }

    // o_proj: standard matmul weight (may keep-quant or expand).
    layer.attn.o_proj =
        OwnMatmulWeight(gguf, Blk(il, "attn_output.weight"), pol);

    // Per-head q/k norms: loaded when present, left empty when absent.
    // Qwen3-4B has them; Qwen3-0.6B does not. OwnBf16 (NO w+1 shift).
    if (HasTensor(gguf, Blk(il, "attn_q_norm.weight"))) {
      RequireExpand(expand_pol, gguf, Blk(il, "attn_q_norm.weight"),
                    GgufTensorRole::kTransformedWeight);
      layer.attn.q_norm =
          OwnBf16(gguf, Blk(il, "attn_q_norm.weight"), {config.head_dim});
    }
    if (HasTensor(gguf, Blk(il, "attn_k_norm.weight"))) {
      RequireExpand(expand_pol, gguf, Blk(il, "attn_k_norm.weight"),
                    GgufTensorRole::kTransformedWeight);
      layer.attn.k_norm =
          OwnBf16(gguf, Blk(il, "attn_k_norm.weight"), {config.head_dim});
    }

    if (keep_projection_parts) {
      GgufLoadPolicy parts_pol = NoNvfp4(pol);
      parts_pol.expand_nk = true;
      parts_pol.elem_kn_repack = false;
      parts_pol.quant_repack = false;
      for (const auto* part : {"ffn_gate.weight", "ffn_up.weight"})
        layer.mlp.gate_up_proj_parts.push_back(OwnMatmulWeight(gguf, Blk(il, part), parts_pol));
    } else {
    // MLP: load gate/up separately, then merge into one gate_up_proj.
    RequireExpand(expand_pol, gguf, Blk(il, "ffn_gate.weight"),
                  GgufTensorRole::kMatmulWeight);
    OwnedTensor gate = OwnBf16(gguf, Blk(il, "ffn_gate.weight"),
                                gguf.Get(Blk(il, "ffn_gate.weight")).shape);
    gate.nk = true;
    RequireExpand(expand_pol, gguf, Blk(il, "ffn_up.weight"),
                  GgufTensorRole::kMatmulWeight);
    OwnedTensor up = OwnBf16(gguf, Blk(il, "ffn_up.weight"),
                              gguf.Get(Blk(il, "ffn_up.weight")).shape);
    up.nk = true;
    layer.mlp.gate_up_proj = ConcatBf16Rows(gate, up);

    }

    // down_proj: standard matmul weight (may keep-quant or expand).
    layer.mlp.down_proj =
        OwnMatmulWeight(gguf, Blk(il, "ffn_down.weight"), pol);

    w.layers.push_back(std::move(layer));
  }
  return w;
}

Qwen3DenseWeights LoadQwen3FromGguf(const GgufFile& gguf,
                                  const HfConfig& config,
                                  const GgufLoadPolicy* policy) {
  return LoadDenseGgufWeights(gguf, config, policy, false);
}

}  // namespace vllm
