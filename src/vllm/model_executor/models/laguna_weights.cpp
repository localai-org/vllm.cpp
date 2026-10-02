// Laguna-S-2.1 config parse + name-map + (scaffolded) weight loaders. W1/W2.
//
// This TU owns three testable, CPU-buildable pieces:
//   1. ParseLagunaParams — resolves + validates every consumed field from the
//      HfConfig (typed + `raw`), including the NESTED per-layer-type
//      `rope_parameters` (the OLMo-3 gate hazard; laguna handles it explicitly)
//      and the per-layer VARIABLE Q-head array.
//   2. The Laguna GGUF `blk.N.*` name-map + the UD-Q4_K_XL per-tensor quant-mix
//      enumeration (pure string / metadata helpers; the byte-exact per-tensor
//      types are CONFIRMED against the real GGUF header at W4 when a checkpoint is
//      fetched — this increment records the EXPECTED unsloth "XL" mix).
//   3. LoadLaguna{ForCausalLMWeights,FromGguf} — parse params + build the seam,
//      then VT_CHECK(false, W3) on the actual device materialization (the towers
//      compose LANDED reuse; nothing here invents a kernel). Mirrors the ds4
//      loader philosophy: RESOLVE + account, DEFER materialize.
//
// Ground truth: poolside/Laguna-S-2.1/config.json (VERIFIED 2026-07-30, scope
// spec §1) + vllm/model_executor/models/laguna.py (MIRROR-vLLM) + the llama.cpp
// Poolside-fork `laguna` branch (GGUF name-map authority, scope spec §3).
#include "vllm/model_executor/models/laguna.h"

#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <variant>
#include <vector>

#include <memory>

#include "vllm/model_executor/model_loader/gguf_dequant.h"
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/model_loader/safetensors_reader.h"  // SafetensorsFile, StTensor
#include "vllm/model_executor/models/dense_weight_loaders.h"      // dense_loaders::LoadBf16Direct/MakeOwned
#include "vllm/model_executor/layers/quantization/compressed_tensors/nvfp4_emulation.h"  // DequantCtNvfp4WeightToF32
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"  // OwnGgufQuantBlocks
#include "vt/dtype.h"
#include "vt/unaligned.h"

namespace vllm {
namespace {

// --- raw (nlohmann::json) readers, mirroring the deepseek_v4_weights.cpp helpers
// `RawInt`/`RawDouble`/`RawBool`/`RawString` (cited by SYMBOL: the line range this
// comment used to name went stale the moment ds4 grew a loader arm above it) ------
int64_t RawInt(const nlohmann::json& doc, const char* key, int64_t fallback) {
  auto it = doc.find(key);
  return (it != doc.end() && it->is_number()) ? it->get<int64_t>() : fallback;
}
double RawDouble(const nlohmann::json& doc, const char* key, double fallback) {
  auto it = doc.find(key);
  return (it != doc.end() && it->is_number()) ? it->get<double>() : fallback;
}
bool RawBool(const nlohmann::json& doc, const char* key, bool fallback) {
  auto it = doc.find(key);
  return (it != doc.end() && it->is_boolean()) ? it->get<bool>() : fallback;
}
std::string RawString(const nlohmann::json& doc, const char* key,
                      const std::string& fallback) {
  auto it = doc.find(key);
  return (it != doc.end() && it->is_string()) ? it->get<std::string>() : fallback;
}
std::vector<int64_t> RawIntArray(const nlohmann::json& doc, const char* key) {
  std::vector<int64_t> out;
  auto it = doc.find(key);
  if (it != doc.end() && it->is_array())
    for (const auto& v : *it)
      if (v.is_number()) out.push_back(v.get<int64_t>());
  return out;
}

// Nested per-layer-type rope block reader. laguna.py walks
// `rope_parameters[layer_type]` (falling back to "full_attention"); we read the
// two known blocks explicitly. This is the field that KeyError-aborted OLMo-3 on
// an older transformers — laguna handles it, and our pin ships transformers 5.14.1.
const nlohmann::json* RopeBlock(const nlohmann::json& doc, const char* layer_type) {
  auto rp = doc.find("rope_parameters");
  if (rp == doc.end() || !rp->is_object()) return nullptr;
  auto lt = rp->find(layer_type);
  if (lt != rp->end() && lt->is_object()) return &*lt;
  return nullptr;
}

}  // namespace

LagunaParams ParseLagunaParams(const HfConfig& config) {
  const nlohmann::json& raw = config.raw;
  LagunaParams p;

  // --- shared geometry ---
  p.hidden_size = config.hidden_size > 0 ? config.hidden_size
                                         : RawInt(raw, "hidden_size", 0);
  p.num_hidden_layers = config.num_hidden_layers > 0
                            ? config.num_hidden_layers
                            : RawInt(raw, "num_hidden_layers", 0);
  p.vocab_size =
      config.vocab_size > 0 ? config.vocab_size : RawInt(raw, "vocab_size", 0);
  p.num_attention_heads = config.num_attention_heads > 0
                              ? config.num_attention_heads
                              : RawInt(raw, "num_attention_heads", 0);
  p.num_key_value_heads = config.num_key_value_heads > 0
                              ? config.num_key_value_heads
                              : RawInt(raw, "num_key_value_heads", 0);
  p.head_dim = config.head_dim > 0 ? config.head_dim : RawInt(raw, "head_dim", 0);
  p.intermediate_size = config.intermediate_size > 0
                            ? config.intermediate_size
                            : RawInt(raw, "intermediate_size", 0);
  p.rms_norm_eps = static_cast<float>(RawDouble(raw, "rms_norm_eps", 1e-6));
  p.tie_word_embeddings = RawBool(raw, "tie_word_embeddings", false);
  p.max_position_embeddings = config.max_position_embeddings > 0
                                  ? config.max_position_embeddings
                                  : RawInt(raw, "max_position_embeddings", 0);

  // --- interleaved attention ---
  p.sliding_window = config.sliding_window.value_or(RawInt(raw, "sliding_window", 0));
  p.layer_types = config.layer_types;  // HfConfig already parses the string array
  if (p.layer_types.empty()) {
    // Synthesize the 1:3 global:sliding pattern (layer % 4 == 0 -> global).
    for (int64_t l = 0; l < p.num_hidden_layers; ++l)
      p.layer_types.emplace_back(l % 4 == 0 ? "full_attention" : "sliding_attention");
  }
  p.num_attention_heads_per_layer =
      RawIntArray(raw, "num_attention_heads_per_layer");
  if (p.num_attention_heads_per_layer.empty()) {
    // Fall back to the 1:3 head pattern (global=base, sliding=1.5x base).
    for (int64_t l = 0; l < p.num_hidden_layers; ++l)
      p.num_attention_heads_per_layer.push_back(
          p.IsGlobalLayer(l) ? p.num_attention_heads
                             : p.num_attention_heads * 3 / 2);
  }

  // --- per-head softplus output gate ---
  p.per_head_output_gate = RawString(raw, "gating", "per-head") == "per-head";

  // --- MoE ---
  p.num_experts =
      config.num_experts > 0 ? config.num_experts : RawInt(raw, "num_experts", 0);
  p.num_experts_per_tok = config.num_experts_per_tok > 0
                              ? config.num_experts_per_tok
                              : RawInt(raw, "num_experts_per_tok", 0);
  p.moe_intermediate_size = config.moe_intermediate_size > 0
                                ? config.moe_intermediate_size
                                : RawInt(raw, "moe_intermediate_size", 0);
  p.shared_expert_intermediate_size =
      RawInt(raw, "shared_expert_intermediate_size", 0);
  p.norm_topk_prob = RawBool(raw, "norm_topk_prob", true);
  p.moe_routed_scaling_factor =
      static_cast<float>(RawDouble(raw, "moe_routed_scaling_factor", 1.0));
  // Router: sigmoid noaux_tc + e_score_correction_bias, UNGROUPED. laguna.py sets
  // use_grouped_topk=False + scoring_func="sigmoid" (the DeepSeek-V3 aux-loss-free
  // router MINUS the group step). The marketing "softplus router" is a misnomer;
  // softplus lives only in the attention out-gate.
  p.use_grouped_topk = false;
  p.has_e_score_correction_bias = true;
  p.mlp_only_layers = RawIntArray(raw, "mlp_only_layers");
  if (p.mlp_only_layers.empty()) p.mlp_only_layers = {0};  // layer 0 dense default

  // --- dual per-layer RoPE (nested rope_parameters) ---
  if (const nlohmann::json* full = RopeBlock(raw, "full_attention")) {
    p.rope_theta_full = RawDouble(*full, "rope_theta", 500000.0);
    // config.json (HF safetensors) uses factor 128 (1M ctx); the UD-Q4_K GGUF
    // uses factor 32 (262144 ctx). The GGUF loader OVERRIDES these from the GGUF
    // KV (laguna.rope.scaling.*) so the same-quant gate matches llama.cpp; this
    // typed path is the safetensors/NVFP4-oracle fallback.
    p.yarn_factor = RawDouble(*full, "factor", 32.0);
    p.yarn_orig_max_pos =
        RawInt(*full, "original_max_position_embeddings", 8192);
    p.yarn_beta_fast = RawDouble(*full, "beta_fast", 32.0);
    p.yarn_beta_slow = RawDouble(*full, "beta_slow", 1.0);
    // HF ships a precomputed `attention_factor` (== the llama.cpp mscale formula
    // output). Back out the raw yarn_attn_factor so LagunaYarnMscale reproduces it
    // for BOTH paths; if absent, default 1.0 (the GGUF value).
    if (const double af = RawDouble(*full, "attention_factor", -1.0); af > 0.0) {
      const double base = p.yarn_factor > 1.0 ? 1.0 + 0.1 * std::log(p.yarn_factor) : 1.0;
      p.yarn_attn_factor = base > 0.0 ? af / base : 1.0;
    } else {
      p.yarn_attn_factor = 1.0;
    }
    p.partial_rotary_factor_full = RawDouble(*full, "partial_rotary_factor", 0.5);
  }
  p.rotary_dim_full = static_cast<int64_t>(p.head_dim * p.partial_rotary_factor_full);
  if (const nlohmann::json* slide = RopeBlock(raw, "sliding_attention")) {
    p.rope_theta_sliding = RawDouble(*slide, "rope_theta", 10000.0);
    const double pr = RawDouble(*slide, "partial_rotary_factor", 1.0);
    p.rotary_dim_sliding = static_cast<int64_t>(p.head_dim * pr);
  } else {
    p.rotary_dim_sliding = p.head_dim;
  }

  // --- invariants ---
  VT_CHECK(p.hidden_size > 0 && p.num_hidden_layers > 0 && p.vocab_size > 0,
           "laguna: missing core geometry (hidden/layers/vocab)");
  VT_CHECK(p.num_key_value_heads > 0 && p.head_dim > 0,
           "laguna: missing GQA geometry (kv_heads/head_dim)");
  VT_CHECK(p.num_experts > 0 && p.num_experts_per_tok > 0,
           "laguna: missing MoE geometry (num_experts/top_k)");
  VT_CHECK(static_cast<int64_t>(p.layer_types.size()) == p.num_hidden_layers,
           "laguna: layer_types length must equal num_hidden_layers");
  VT_CHECK(static_cast<int64_t>(p.num_attention_heads_per_layer.size()) ==
               p.num_hidden_layers,
           "laguna: num_attention_heads_per_layer length must equal num_hidden_layers");
  VT_CHECK(p.rotary_dim_full > 0 && p.rotary_dim_full <= p.head_dim,
           "laguna: partial rotary_dim (full-attn YaRN) out of range");
  return p;
}

void ParseLagunaConfig(const HfConfig& config) {
  // The resolve itself IS the validation (throws on every unsupported field).
  (void)ParseLagunaParams(config);
}

// ─── GGUF name-map + UD-Q4_K_XL quant-mix (llama.cpp Poolside-fork `laguna`) ──
//
// The Laguna `blk.N.*` tensor names, mirroring the ds4 blk.N.* map + the standard
// llama.cpp MoE naming. Exposed as pure helpers so a name-map coverage checker
// (the check-dsv4-gguf-namemap.py pattern) can be extended to laguna WITHOUT a
// download. The per-tensor quant TYPE is read from the GGUF header at W4; the
// EXPECTED unsloth UD-Q4_K_XL "XL" mix is recorded in the table below.
std::string LagunaGgufAttnName(int64_t layer, const char* proj) {
  // proj in {attn_norm, attn_q, attn_k, attn_v, attn_output, attn_gate,
  //          attn_q_norm, attn_k_norm}. VERIFIED W4 from the real GGUF headers.
  return "blk." + std::to_string(layer) + "." + proj + ".weight";
}
std::string LagunaGgufMoeName(int64_t layer, const char* which) {
  // which in {ffn_norm, ffn_gate_inp, ffn_gate_exps, ffn_up_exps, ffn_down_exps,
  //   ffn_gate_shexp, ffn_up_shexp, ffn_down_shexp} (+ exp_probs_b.bias, no
  //   ".weight" suffix). Dense layer 0: ffn_gate/ffn_up/ffn_down. VERIFIED W4.
  return "blk." + std::to_string(layer) + "." + which + ".weight";
}

// The VERIFIED UD-Q4_K_XL per-role quant mix (unsloth dynamic "XL"), read W4
// 2026-07-31 from the real GGUF tensor-info headers (814 tensors, split.count=3).
// Every type is ALREADY decoded in-tree (Q4_K/Q5_K/Q6_K/Q8_0 — cuda_quant_dot.cu +
// cpu_quant_dot.cpp) => ZERO new decode kernel for ANY tensor in the mix.
//   token_embd / output(lm_head)      -> Q6_K/Q8_0 (accuracy-critical)
//   attn_q/k/v/output/gate            -> Q8_0 (VERIFIED — all attn linears Q8_0)
//   attn_q_norm/k_norm/attn_norm      -> F32 (per-head QK-RMSNorm + pre-attn norm)
//   ffn_gate_inp (router)             -> F32 [H,256]; exp_probs_b.bias -> F32 [256]
//   ffn_{gate,up}_exps (256 experts)  -> Q4_K (the bulk)
//   ffn_down_exps                     -> Q5_K (UD "richer down-proj" rule)
//   ffn_{gate,up,down}_shexp (shared) -> Q8_0
//   ffn_norm                          -> F32
// GGUF ne-order is [in, out] (reverse of torch [out, in]); OwnGgufQuantBlocks +
// vt::MatmulBT consume the on-disk [N,K] block layout with no transpose.

namespace {

// F32 scalar read (weight/input global scales are F32 scalars), from the shared
// seam. The local copy this replaces checked the dtype but bounded the size with
// `nbytes >= 4`, a FLOOR, so a scale ARRAY was read as element 0 (#1181).
using dense_loaders::ReadF32Scalar;

// F32 tensor materialized (the e_score_correction_bias is F32 [E] on most shards, but
// some poolside NVFP4 shards store it BF16 — upconvert those so the router/topk always
// sees F32). Output dtype is always F32 (the "F32Direct" contract).
OwnedTensor LnLoadF32Direct(const TensorResolver& get, const std::string& name) {
  const StTensor& t = get(name);
  OwnedTensor o = dense_loaders::MakeOwned(vt::DType::kF32, t.shape);
  if (t.dtype == "F32") {
    VT_CHECK(t.nbytes == o.bytes.size(), "laguna nvfp4: F32 size mismatch " + name);
    std::memcpy(o.bytes.data(), t.data, t.nbytes);
  } else if (t.dtype == "BF16") {
    const size_t n = t.nbytes / 2;  // bf16 = 2 bytes/elem
    VT_CHECK(o.bytes.size() == n * 4, "laguna nvfp4: BF16->F32 size mismatch " + name);
    // Unaligned: `t.data` is an arbitrary byte offset into the mmap (#627).
    auto* dst = reinterpret_cast<float*>(o.bytes.data());
    for (size_t i = 0; i < n; ++i) {
      const uint32_t bits = static_cast<uint32_t>(
                                vt::LoadUnaligned<uint16_t>(t.data + i * 2))
                            << 16;  // bf16 -> high 16 bits of f32
      std::memcpy(&dst[i], &bits, 4);
    }
  } else {
    VT_CHECK(false, "laguna nvfp4: F32 or BF16 expected for " + name + " (got " + t.dtype + ")");
  }
  return o;
}

// W4A4 NVFP4 routed-expert loader. Byte-identical to qwen3_5_dense_weights.cpp
// LoadCtNvfp4Raw (weight_packed U8 [N,K/2] + weight_scale F8_E4M3 [N,K/16] +
// weight_global_scale/input_global_scale F32 scalars); duplicated here to avoid
// touching the SACRED-gated qwen3_5 dense TU — TODO(S4): promote LoadCtNvfp4Raw
// to dense_weight_loaders.h and share (see laguna-nvfp4-arm spec §N1b).
Nvfp4Weight LnLoadCtNvfp4Raw(const TensorResolver& get, const std::string& proj) {
  const StTensor& packed = get(proj + ".weight_packed");
  VT_CHECK(packed.dtype == "U8" && packed.shape.size() == 2,
           "laguna nvfp4: U8 2-D weight_packed for " + proj);
  const int64_t out_dim = packed.shape[0];
  const int64_t in_dim = packed.shape[1] * 2;
  VT_CHECK(in_dim % 16 == 0, "laguna nvfp4: in_dim must be %16 for " + proj);
  const StTensor& ws = get(proj + ".weight_scale");
  VT_CHECK(ws.dtype == "F8_E4M3", "laguna nvfp4: F8_E4M3 weight_scale for " + proj);
  const float wgs = ReadF32Scalar(get, proj + ".weight_global_scale");
  VT_CHECK(wgs != 0.0F, "laguna nvfp4: zero weight_global_scale for " + proj);
  Nvfp4Weight r;
  r.n = out_dim;
  r.k = in_dim;
  r.weight_global_scale_inv = wgs;
  r.scale2 = 1.0F / wgs;
  const float igs = ReadF32Scalar(get, proj + ".input_global_scale");
  VT_CHECK(igs != 0.0F, "laguna nvfp4: zero input_global_scale for " + proj);
  r.input_global_scale_inv = igs;
  r.alpha = r.scale2 * (1.0F / igs);
  r.packed = dense_loaders::MakeOwned(vt::DType::kI8, {out_dim, in_dim / 2});
  VT_CHECK(packed.nbytes == r.packed.bytes.size(), "laguna nvfp4: packed size " + proj);
  std::memcpy(r.packed.bytes.data(), packed.data, packed.nbytes);
  r.scale = dense_loaders::MakeOwned(vt::DType::kI8, {out_dim, in_dim / 16});
  VT_CHECK(ws.nbytes == r.scale.bytes.size(), "laguna nvfp4: scale size " + proj);
  std::memcpy(r.scale.bytes.data(), ws.data, ws.nbytes);
  return r;
}

// Shared-expert weight loader. Laguna-S-2.1 keeps the shared expert BF16 (`.weight`);
// Laguna-XS-2.1 quantizes it to NVFP4 (`.weight_packed` + scales). Dequant the NVFP4
// case to BF16 at load so the (BF16) shared-expert forward GEMMs are unchanged (the
// shared expert is small — the bf16-vs-fp4 dtype barely moves decode tok/s).
OwnedTensor LnLoadSharedExpertBf16(const TensorResolver& get,
                                   const std::function<bool(const std::string&)>& has,
                                   const std::string& proj) {
  if (!has(proj + ".weight_packed"))
    return dense_loaders::LoadBf16Direct(get, proj + ".weight");  // S-2.1 bf16 path
  const Nvfp4Weight r = LnLoadCtNvfp4Raw(get, proj);              // XS NVFP4 path
  const float wgs = ReadF32Scalar(get, proj + ".weight_global_scale");
  std::vector<float> f32(static_cast<size_t>(r.n) * static_cast<size_t>(r.k));
  DequantCtNvfp4WeightToF32(reinterpret_cast<const uint8_t*>(r.packed.bytes.data()),
                            reinterpret_cast<const uint8_t*>(r.scale.bytes.data()), wgs, r.n, r.k,
                            f32.data());
  OwnedTensor o = dense_loaders::MakeOwned(vt::DType::kBF16, {r.n, r.k});
  auto* dst = reinterpret_cast<uint16_t*>(o.bytes.data());
  for (size_t i = 0; i < f32.size(); ++i) {  // f32 -> bf16 round-to-nearest-even
    uint32_t x;
    std::memcpy(&x, &f32[i], 4);
    x += 0x7FFFu + ((x >> 16) & 1u);
    dst[i] = static_cast<uint16_t>(x >> 16);
  }
  return o;
}

// This helper is HOST-ONLY bf16 row concat with no device dependency of its own; it is
// build-gated purely because its only CONSUMER (the resident/graph decode) is CUDA-only,
// and building the stacked copies unconditionally would cost a CPU-only build memory it
// can never use. REPAIR OWED: ask at RUNTIME instead (LagunaDeviceKernelsAvailable() /
// non-kCPU queue) so the shared layer stops branching at build time. Deferred — it moves
// weight loading, so it needs the GB10 Laguna re-gate.
// DSR-ALLOW(S1): host-only concat, build-gated on its CUDA-only consumer; repair owed.
#ifdef VT_MARLIN_NVFP4
// Stack several bf16 [N_i, K] row-major weight tensors into ONE bf16 [sum N_i, K]
// tensor (plain row-block memcpy concat; the listed order is preserved). Used to
// pre-fuse the resident/graph-decode projections that all read the SAME activation
// (q/k/v/g off the input norm; router/shared_gate/shared_up off the post-attn
// norm) so the decode issues ONE wider M=1 GEMV instead of 3-4 narrow ones — the
// MEASURED #1 steady-decode lever (raising the GEMV's N = grid blocks fixes the
// small-N under-fill). Byte-exact in principle: each output row is an independent
// fp32-accumulated dot over the same activation, so stacking rows changes no row's
// math (only the launch is fused). Guarded to the NVFP4 build (the only build whose
// resident/graph decode consumes it); the split originals are KEPT for the host/
// prefill forwards, so this concat is purely ADDITIVE.
OwnedTensor LnStackBf16RowsNK(const std::vector<const OwnedTensor*>& parts) {
  VT_CHECK(!parts.empty(), "laguna nvfp4 fuse: empty stack");
  int64_t k = -1;
  int64_t n_total = 0;
  for (const OwnedTensor* t : parts) {
    VT_CHECK(t != nullptr && !t->Empty(), "laguna nvfp4 fuse: missing stack part");
    VT_CHECK(t->dtype == vt::DType::kBF16 && t->rank == 2,
             "laguna nvfp4 fuse: expected 2-D bf16 stack part");
    if (k < 0) k = t->shape[1];
    VT_CHECK(t->shape[1] == k, "laguna nvfp4 fuse: stack parts must share K");
    n_total += t->shape[0];
  }
  OwnedTensor o = dense_loaders::MakeOwned(vt::DType::kBF16, {n_total, k});
  size_t off = 0;
  for (const OwnedTensor* t : parts) {
    std::memcpy(o.bytes.data() + off, t->bytes.data(), t->bytes.size());
    off += t->bytes.size();
  }
  VT_CHECK(off == o.bytes.size(), "laguna nvfp4 fuse: byte accounting mismatch");
  o.nk = true;  // raw [N,K] torch orientation (vt::MatmulBT), matches the parts
  return o;
}
#endif  // VT_MARLIN_NVFP4

}  // namespace

// N1b (task #230, spec laguna-nvfp4-arm-2026-07-31.md): the real safetensors
// NVFP4 loader for poolside/Laguna-S-2.1-NVFP4. Recipe VERIFIED against the
// checkpoint index: bf16 attn/dense/norms/embed/lm_head/router/shared-expert,
// F32 e_score_correction_bias, W4A4-NVFP4 routed experts. ~67 GiB, fits one
// GB10. The dual-RoPE caches are built lazily in the forward (as the GGUF path
// does), so this fills params + weights only.
LagunaWeights LoadLagunaForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  using dense_loaders::LoadBf16Direct;
  LagunaWeights w;
  w.params = ParseLagunaParams(config);
  const LagunaParams& p = w.params;

  // name -> shard resolver (mirrors qwen3_5_weights.cpp:423).
  auto where =
      std::make_shared<std::unordered_map<std::string, const SafetensorsFile*>>();
  for (const SafetensorsFile& s : shards)
    for (const std::string& n : s.Names()) (*where)[n] = &s;
  const TensorResolver get =
      [where](const std::string& name) -> const StTensor& {
    auto it = where->find(name);
    VT_CHECK(it != where->end(), "laguna nvfp4: tensor not found: " + name);
    return it->second->Get(name);
  };
  const std::function<bool(const std::string&)> has =
      [where](const std::string& name) { return where->find(name) != where->end(); };

  // model level (all BF16; untied lm_head).
  w.embed = LoadBf16Direct(get, "model.embed_tokens.weight");
  w.norm = LoadBf16Direct(get, "model.norm.weight");
  w.lm_head = LoadBf16Direct(get, "lm_head.weight");

  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    LagunaLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.is_dense = p.IsDenseLayer(l);
    const std::string b = "model.layers." + std::to_string(l) + ".";

    lw.input_norm = LoadBf16Direct(get, b + "input_layernorm.weight");
    lw.post_attn_norm = LoadBf16Direct(get, b + "post_attention_layernorm.weight");
    lw.attn.q_proj = LoadBf16Direct(get, b + "self_attn.q_proj.weight");
    lw.attn.k_proj = LoadBf16Direct(get, b + "self_attn.k_proj.weight");
    lw.attn.v_proj = LoadBf16Direct(get, b + "self_attn.v_proj.weight");
    lw.attn.o_proj = LoadBf16Direct(get, b + "self_attn.o_proj.weight");
    lw.attn.g_proj = LoadBf16Direct(get, b + "self_attn.g_proj.weight");
    lw.attn.q_norm = LoadBf16Direct(get, b + "self_attn.q_norm.weight");
    lw.attn.k_norm = LoadBf16Direct(get, b + "self_attn.k_norm.weight");

    if (lw.is_dense) {  // layer 0 (mlp_only_layers = {0})
      lw.mlp.gate_proj = LoadBf16Direct(get, b + "mlp.gate_proj.weight");
      lw.mlp.up_proj = LoadBf16Direct(get, b + "mlp.up_proj.weight");
      lw.mlp.down_proj = LoadBf16Direct(get, b + "mlp.down_proj.weight");
    } else {  // MoE layers 1..47
      lw.moe.router = LoadBf16Direct(get, b + "mlp.gate.weight");  // BF16 [E,H]
      lw.moe.e_score_correction_bias =
          LnLoadF32Direct(get, b + "mlp.experts.e_score_correction_bias");  // F32 [E]
      const std::string ex = b + "mlp.experts.";
      lw.moe.experts_gate_fp4.reserve(static_cast<size_t>(p.num_experts));
      lw.moe.experts_up_fp4.reserve(static_cast<size_t>(p.num_experts));
      lw.moe.experts_down_fp4.reserve(static_cast<size_t>(p.num_experts));
      for (int64_t e = 0; e < p.num_experts; ++e) {
        const std::string ep = ex + std::to_string(e) + ".";
        lw.moe.experts_gate_fp4.push_back(LnLoadCtNvfp4Raw(get, ep + "gate_proj"));
        lw.moe.experts_up_fp4.push_back(LnLoadCtNvfp4Raw(get, ep + "up_proj"));
        lw.moe.experts_down_fp4.push_back(LnLoadCtNvfp4Raw(get, ep + "down_proj"));
      }
      lw.moe.shared_gate = LnLoadSharedExpertBf16(get, has, b + "mlp.shared_expert.gate_proj");
      lw.moe.shared_up = LnLoadSharedExpertBf16(get, has, b + "mlp.shared_expert.up_proj");
      lw.moe.shared_down = LnLoadSharedExpertBf16(get, has, b + "mlp.shared_expert.down_proj");
    }

    // DSR-ALLOW(S1): call site of the host-only concat above — same reason, same repair.
#ifdef VT_MARLIN_NVFP4
    // Fused decode projections (the steady-decode lever): stack q|k|v|g (all read
    // the input-norm activation) and router|shared_gate|shared_up (all read the
    // post-attn activation) so the resident/graph decode fires ONE wider GEMV each.
    // ADDITIVE — the split originals above are KEPT (the host/prefill forwards read
    // them). Order MUST match the forward's slice offsets (q,k,v,g / router,sg,su).
    lw.attn.qkvg_proj = LnStackBf16RowsNK(
        {&lw.attn.q_proj, &lw.attn.k_proj, &lw.attn.v_proj, &lw.attn.g_proj});
    if (!lw.is_dense)
      lw.moe.router_shared_gu = LnStackBf16RowsNK(
          {&lw.moe.router, &lw.moe.shared_gate, &lw.moe.shared_up});
#endif  // VT_MARLIN_NVFP4
  }
  w.accounted_tensors = static_cast<int64_t>(where->size());
  w.has_gguf_weights = false;  // safetensors NVFP4 arm, not the GGUF keep-quant tower
  w.has_nvfp4_weights = true;  // N2 forward selects the W4A4 arm off this flag / non-empty fp4 experts
  return w;
}

// ════════════════════════════════════════════════════════════════════════════
// W5 — the REAL multi-shard GGUF keep-quant tower materialization.
//
// Mirrors the ds4 keep-quant loader — `struct V4GgufCtx` through
// `LoadDeepseekV4FromGguf` in `src/vllm/model_executor/models/deepseek_v4_weights.cpp`,
// cited by SYMBOL because a line range there does not survive a new loader arm:
// the big GEMM weights (attn q/k/v/o/gate Q8_0, dense-L0 ffn Q8_0, routed experts
// Q4_K/Q5_K, shared experts Q8_0, lm_head Q8_0) stay block-COMPRESSED via
// OwnGgufQuantBlocks (the memory enabler — a bf16 expansion of the 256 experts is
// ~226 GiB and OOM-reboots the 119 GiB unified pool); the small norms / router /
// bias / embed dequant to f32. The Laguna UD-Q4_K_XL ships as 3 shards (shard-1 =
// header only), so a name→shard ROUTING map (below) routes each tensor to the
// GgufFile that holds it — loader-local, no reader rewrite, no merge.
namespace {

// --- GGUF KV readers (mirror deepseek_v4_weights.cpp GgKvInt/GgKvFloat) --------
int64_t GgInt(const GgufValue& v, const std::string& key) {
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
      throw std::runtime_error("laguna gguf: key " + key + " is not an integer");
  }
}
double GgFloat(const GgufValue& v, const std::string& key) {
  if (v.TypeId() == kGgufF32) return std::get<float>(v.v);
  if (v.TypeId() == kGgufF64) return std::get<double>(v.v);
  return static_cast<double>(GgInt(v, key));
}
int64_t ReqInt(const GgufFile& g, const std::string& key) {
  const GgufValue* v = g.FindKv(key);
  VT_CHECK(v != nullptr, "laguna gguf: missing required key " + key);
  return GgInt(*v, key);
}
int64_t OptInt(const GgufFile& g, const std::string& key, int64_t dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? GgInt(*v, key) : dflt;
}
double OptFloat(const GgufFile& g, const std::string& key, double dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? GgFloat(*v, key) : dflt;
}
bool OptBool(const GgufFile& g, const std::string& key, bool dflt) {
  const GgufValue* v = g.FindKv(key);
  return v != nullptr ? (GgInt(*v, key) != 0) : dflt;
}

}  // namespace

LagunaParams LagunaParamsFromGguf(const GgufFile& g) {
  const GgufValue* arch = g.FindKv("general.architecture");
  VT_CHECK(arch != nullptr && arch->TypeId() == kGgufString &&
               std::get<std::string>(arch->v) == "laguna",
           "laguna gguf: expected general.architecture 'laguna'");
  const std::string p = "laguna.";
  LagunaParams d;

  // --- shared geometry ---
  d.hidden_size = ReqInt(g, p + "embedding_length");
  d.num_hidden_layers = ReqInt(g, p + "block_count");
  d.num_key_value_heads = ReqInt(g, p + "attention.head_count_kv");
  d.head_dim = ReqInt(g, p + "attention.key_length");
  d.intermediate_size = ReqInt(g, p + "feed_forward_length");
  d.rms_norm_eps =
      static_cast<float>(OptFloat(g, p + "attention.layer_norm_rms_epsilon", 1e-6));
  d.max_position_embeddings = OptInt(g, p + "context_length", 0);
  d.tie_word_embeddings = false;  // Laguna ships an untied `output.weight`.

  // --- per-layer VARIABLE Q-head count (the `laguna.attention.head_count` ARRAY)
  //     + the derived layer_types (global = the smaller head count, sliding = the
  //     larger; the 1:3 pattern gives 12 global / 36 sliding). ---
  const GgufValue* hc = g.FindKv(p + "attention.head_count");
  VT_CHECK(hc != nullptr && hc->TypeId() == kGgufArray,
           "laguna gguf: attention.head_count must be a per-layer array");
  const GgufArray& hca = std::get<GgufArray>(hc->v);
  VT_CHECK(static_cast<int64_t>(hca.elems.size()) == d.num_hidden_layers,
           "laguna gguf: head_count array length must equal block_count");
  int64_t min_heads = -1;
  for (const GgufValue& e : hca.elems) {
    const int64_t h = GgInt(e, "attention.head_count");
    d.num_attention_heads_per_layer.push_back(h);
    if (min_heads < 0 || h < min_heads) min_heads = h;
  }
  d.num_attention_heads = min_heads;  // base (global) head count
  for (int64_t l = 0; l < d.num_hidden_layers; ++l)
    d.layer_types.emplace_back(
        d.num_attention_heads_per_layer[static_cast<size_t>(l)] == min_heads
            ? "full_attention"
            : "sliding_attention");
  d.sliding_window = OptInt(g, p + "attention.sliding_window", 0);

  // --- per-head softplus attn output gate + QK-RMSNorm (present in the tensor map)
  d.per_head_output_gate = true;
  d.has_qk_norm = true;

  // --- MoE ---
  d.num_experts = ReqInt(g, p + "expert_count");
  d.num_experts_per_tok = ReqInt(g, p + "expert_used_count");
  d.moe_intermediate_size = ReqInt(g, p + "expert_feed_forward_length");
  d.shared_expert_intermediate_size =
      OptInt(g, p + "expert_shared_feed_forward_length", 0);
  d.norm_topk_prob = OptBool(g, p + "expert_weights_norm", true);
  d.moe_routed_scaling_factor =
      static_cast<float>(OptFloat(g, p + "expert_weights_scale", 1.0));
  d.use_grouped_topk = false;  // laguna.py sets use_grouped_topk=False
  // expert_gating_func: 1=softmax, 2=sigmoid (llama.cpp enum). Laguna uses 2.
  const int64_t gating = OptInt(g, p + "expert_gating_func", 2);
  VT_CHECK(gating == 2, "laguna gguf: only sigmoid gating (func=2) is scoped");
  d.has_e_score_correction_bias = true;
  const int64_t leading_dense = OptInt(g, p + "leading_dense_block_count", 1);
  for (int64_t l = 0; l < leading_dense; ++l) d.mlp_only_layers.push_back(l);

  // --- dual per-layer RoPE (global YaRN-partial / sliding plain) ---
  d.rope_theta_full = OptFloat(g, p + "rope.freq_base", 500000.0);
  d.rope_theta_sliding = OptFloat(g, p + "rope.freq_base_swa", 10000.0);
  d.rotary_dim_full = OptInt(g, p + "rope.dimension_count", 64);
  d.rotary_dim_sliding = OptInt(g, p + "rope.dimension_count_swa", d.head_dim);
  d.partial_rotary_factor_full =
      d.head_dim > 0 ? static_cast<double>(d.rotary_dim_full) / d.head_dim : 0.5;
  d.yarn_factor = OptFloat(g, p + "rope.scaling.factor", 32.0);
  d.yarn_orig_max_pos =
      OptInt(g, p + "rope.scaling.original_context_length", 8192);
  d.yarn_beta_fast = OptFloat(g, p + "rope.scaling.yarn_beta_fast", 32.0);
  d.yarn_beta_slow = OptFloat(g, p + "rope.scaling.yarn_beta_slow", 1.0);
  // `rope.scaling.yarn_attn_factor` is NOT read. llama.cpp `b10451` reads that
  // key for grok only (src/models/grok.cpp:25); for every other YaRN model it
  // derives the factor from `rope.scaling.factor` (src/llama-context.cpp:176-213)
  // and ggml's rope multiplies in (1 + 0.1 ln factor) itself. The UD-Q4_K S-2.1
  // file stores 1.0, so reading it was harmless there; the XS-2.1 APEX file stores
  // the already-multiplied 1.34657, and reading it applied the mscale twice on
  // every full-attention layer (1.813 instead of 1.347).
  d.yarn_attn_factor = 1.0;

  // vocab: prefer the KV, else the token_embd leading (out) dim.
  const GgufValue* vk = g.FindKv(p + "vocab_size");
  d.vocab_size = vk != nullptr ? GgInt(*vk, p + "vocab_size") : 0;

  VT_CHECK(d.hidden_size > 0 && d.num_hidden_layers > 0 && d.head_dim > 0,
           "laguna gguf: degenerate geometry");
  VT_CHECK(d.num_experts > 0 && d.num_experts_per_tok > 0,
           "laguna gguf: missing MoE geometry");
  VT_CHECK(d.rotary_dim_full > 0 && d.rotary_dim_full <= d.head_dim,
           "laguna gguf: partial rotary_dim (full-attn) out of range");
  return d;
}

HfConfig LagunaHfConfigFromGguf(const GgufFile& g) {
  const LagunaParams d = LagunaParamsFromGguf(g);
  int64_t vocab = d.vocab_size;
  if (vocab <= 0) vocab = g.Get("token_embd.weight").shape[0];
  // A tied file omits `output.weight` (llama.cpp TENSOR_DUPLICATED); the weight
  // loader handles both, so the config reports which one this file is.
  bool has_output = false;
  for (const GgufTensorInfo& t : g.Tensors())
    if (t.name == "output.weight") has_output = true;
  const double yarn_base =
      d.yarn_factor > 1.0 ? 1.0 + 0.1 * std::log(d.yarn_factor) : 1.0;
  nlohmann::json doc = {
      {"architectures", {"LagunaForCausalLM"}},
      {"model_type", "laguna"},
      {"torch_dtype", "bfloat16"},
      {"hidden_size", d.hidden_size},
      {"num_hidden_layers", d.num_hidden_layers},
      {"vocab_size", vocab},
      {"num_attention_heads", d.num_attention_heads},
      {"num_key_value_heads", d.num_key_value_heads},
      {"head_dim", d.head_dim},
      {"intermediate_size", d.intermediate_size},
      {"rms_norm_eps", d.rms_norm_eps},
      {"tie_word_embeddings", !has_output},
      {"max_position_embeddings", d.max_position_embeddings},
      {"sliding_window", d.sliding_window},
      {"layer_types", d.layer_types},
      {"num_attention_heads_per_layer", d.num_attention_heads_per_layer},
      {"gating", d.per_head_output_gate ? "per-head" : "none"},
      {"num_experts", d.num_experts},
      {"num_experts_per_tok", d.num_experts_per_tok},
      {"moe_intermediate_size", d.moe_intermediate_size},
      {"shared_expert_intermediate_size", d.shared_expert_intermediate_size},
      {"norm_topk_prob", d.norm_topk_prob},
      {"moe_routed_scaling_factor", d.moe_routed_scaling_factor},
      {"mlp_only_layers", d.mlp_only_layers},
      {"rope_parameters",
       {{"full_attention",
         {{"rope_type", "yarn"},
          {"rope_theta", d.rope_theta_full},
          {"factor", d.yarn_factor},
          {"original_max_position_embeddings", d.yarn_orig_max_pos},
          {"beta_fast", d.yarn_beta_fast},
          {"beta_slow", d.yarn_beta_slow},
          // ParseLagunaParams backs yarn_attn_factor out of this product.
          {"attention_factor", d.yarn_attn_factor * yarn_base},
          {"partial_rotary_factor", d.partial_rotary_factor_full}}},
        {"sliding_attention",
         {{"rope_type", "default"},
          {"rope_theta", d.rope_theta_sliding},
          {"partial_rotary_factor",
           static_cast<double>(d.rotary_dim_sliding) / d.head_dim}}}}},
  };
  return ParseHfConfig(doc, "laguna gguf");
}

namespace {

// --- tower materializers (mirror deepseek_v4_weights.cpp MakeBf16Owned/MakeF32Owned)
OwnedTensor MakeF32Owned(const std::vector<float>& dq,
                         const std::vector<int64_t>& shape) {
  OwnedTensor o;
  o.dtype = vt::DType::kF32;
  o.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  VT_CHECK(static_cast<int64_t>(dq.size()) == n, "laguna gguf: f32 size mismatch");
  o.bytes.resize(static_cast<size_t>(n) * sizeof(float));
  std::memcpy(o.bytes.data(), dq.data(), o.bytes.size());
  return o;
}
OwnedTensor MakeBf16Owned(const std::vector<uint16_t>& dq,
                          const std::vector<int64_t>& shape, bool nk) {
  OwnedTensor o;
  o.dtype = vt::DType::kBF16;
  o.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < o.rank; ++i) {
    o.shape[i] = shape[i];
    n *= shape[i];
  }
  VT_CHECK(static_cast<int64_t>(dq.size()) == n, "laguna gguf: bf16 size mismatch");
  o.bytes.resize(static_cast<size_t>(n) * sizeof(uint16_t));
  std::memcpy(o.bytes.data(), dq.data(), o.bytes.size());
  o.nk = nk;
  return o;
}

// A multi-shard routing context. Scans every shard's tensor table into a
// name→{owning GgufFile, &info} map; Take() resolves the shard that holds a
// tensor so a keep-quant borrow refcounts the RIGHT shard's mmap. Mw/Sew/Vec
// mirror the ds4 V4GgufCtx roles.
struct LagunaGgufCtx {
  const GgufLoadPolicy& pol;
  std::unordered_map<std::string, std::pair<const GgufFile*, const GgufTensorInfo*>>
      index;
  std::unordered_set<std::string> consumed;

  explicit LagunaGgufCtx(const std::vector<const GgufFile*>& shards,
                         const GgufLoadPolicy& p)
      : pol(p) {
    for (const GgufFile* s : shards) {
      VT_CHECK(s != nullptr, "laguna gguf: null shard");
      for (const GgufTensorInfo& t : s->Tensors())
        index[t.name] = {s, &t};
    }
  }

  std::pair<const GgufFile*, const GgufTensorInfo*> Take(const std::string& name) {
    auto it = index.find(name);
    VT_CHECK(it != index.end(),
             "laguna gguf loader: expected tensor missing across shards: " + name);
    consumed.insert(name);
    return it->second;
  }
  // 2-D [out,in] matmul weight: keep its blocks (keep-quant) else expand bf16,
  // both in the file's own [N,K] order (nk=true) — GGUF disk order IS MatmulBT.
  OwnedTensor Mw(const std::string& name) {
    auto [g, t] = Take(name);
    VT_CHECK(t->shape.size() == 2, "laguna gguf: expected 2-D MW " + name);
    const GgufResidency r = pol.Route(*t, GgufTensorRole::kMatmulWeight);
    if (GgufResidencyKeepsBlockWeights(r))
      return OwnGgufQuantBlocks(*t, t->shape[0], t->shape[1], /*row_offset=*/0,
                                pol.mmap_residency ? g : nullptr, pol.quant_repack);
    return MakeBf16Owned(
        DequantGgufRowToBf16(t->ggml_type, t->data, t->shape[0] * t->shape[1]),
        {t->shape[0], t->shape[1]}, /*nk=*/true);
  }
  // Stacked [E,out,in] expert weight: KEEP the whole slab COMPRESSED [E*out,in]
  // (each expert = out whole rows = whole blocks), else expand bf16.
  OwnedTensor Sew(const std::string& name, int64_t experts) {
    auto [g, t] = Take(name);
    VT_CHECK(t->shape.size() == 3 && t->shape[0] == experts,
             "laguna gguf: expected [E,out,in] expert tensor " + name);
    const int64_t rows = t->shape[0] * t->shape[1];  // E*out
    const int64_t k = t->shape[2];                   // in
    const GgufResidency r = pol.Route(*t, GgufTensorRole::kStackedExpertWeight);
    if (GgufResidencyKeepsBlockWeights(r))
      return OwnGgufQuantBlocks(*t, rows, k, /*row_offset=*/0,
                                pol.mmap_residency ? g : nullptr, pol.quant_repack);
    return MakeBf16Owned(DequantGgufRowToBf16(t->ggml_type, t->data, rows * k),
                         {rows, k}, /*nk=*/true);
  }
  // A value tensor (norm/bias/router): NEVER keep-quant, dequant f32.
  OwnedTensor Vec(const std::string& name, GgufTensorRole role) {
    return VecWith(pol, name, role);
  }
  // `token_embd.weight`. Laguna gathers it as a flat host f32 array
  // (`LagunaEmbed`, laguna.cpp:1594) and, on a TIED file, feeds the same f32
  // image to the final projection (`MatmulNK(src, ReadF32(weights.embed), ...)`,
  // laguna.cpp:1281), so it cannot read a table that keeps its ggml blocks —
  // which is what the SHARED policy elects for every published quantized laguna
  // checkpoint since MODEL-MM-QWEN4-EXP W6a (#1989). The narrowing is stated by
  // name here; see `NoKeepQuant`. A keep-quant gather for this model is owed to
  // #1978.
  OwnedTensor EmbedF32(const std::string& name) {
    return VecWith(NoKeepQuant(pol), name, GgufTensorRole::kEmbeddingTable);
  }
  OwnedTensor VecWith(const GgufLoadPolicy& p, const std::string& name,
                      GgufTensorRole role) {
    auto [g, t] = Take(name);
    (void)g;
    VT_CHECK(p.Route(*t, role) == GgufResidency::kExpandBf16,
             std::string("laguna gguf: a ") + Name(role) +
                 " tensor must not keep quant blocks: " + name);
    int64_t numel = 1;
    for (int64_t dim : t->shape) numel *= dim;
    return MakeF32Owned(DequantGgufRowToF32(t->ggml_type, t->data, numel), t->shape);
  }
  bool Has(const std::string& name) const {
    return index.find(name) != index.end();
  }
};

std::string Blk(int64_t l, const std::string& suffix) {
  return "blk." + std::to_string(l) + "." + suffix;
}

}  // namespace

LagunaWeights LoadLagunaFromGgufShards(const std::vector<const GgufFile*>& shards,
                                       const GgufLoadPolicy* policy) {
  VT_CHECK(!shards.empty() && shards[0] != nullptr,
           "laguna gguf: shards[0] (the metadata shard) is required");
  // A null `policy` means NO ENGINE RESOLVED A DEVICE: every production GGUF
  // entry point now builds one from `ModelSource::device` in its registry hook.
  // `kCPU` is STATED rather than probed — see `GgufLoadPolicy::FromEnv`.
  const GgufLoadPolicy env = GgufLoadPolicy::FromEnv(vt::DeviceType::kCPU);
  const GgufLoadPolicy& pol = policy != nullptr ? *policy : env;

  LagunaWeights w;
  w.params = LagunaParamsFromGguf(*shards[0]);  // KV lives only in shard-1
  const LagunaParams& p = w.params;

  LagunaGgufCtx ctx(shards, pol);
  const int64_t E = p.num_experts;
  const bool tied = !ctx.Has("output.weight");

  // vocab fallback from token_embd's [V,H] leading dim.
  if (w.params.vocab_size <= 0) {
    auto [g, t] = ctx.Take("token_embd.weight");
    (void)g;
    w.params.vocab_size = t->shape[0];
    ctx.consumed.erase("token_embd.weight");  // re-routed below with its role
  }

  // ── model level ──────────────────────────────────────────────────────────
  w.embed = ctx.EmbedF32("token_embd.weight");  // f32 gather (and tied head)
  w.norm = ctx.Vec("output_norm.weight", GgufTensorRole::kVector);
  w.lm_head = tied ? OwnedTensor{} : ctx.Mw("output.weight");  // keep-quant Q8_0

  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));
  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    LagunaLayerWeights& lw = w.layers[static_cast<size_t>(l)];
    lw.is_dense = p.IsDenseLayer(l);

    // norms (f32)
    lw.input_norm = ctx.Vec(Blk(l, "attn_norm.weight"), GgufTensorRole::kVector);
    lw.post_attn_norm = ctx.Vec(Blk(l, "ffn_norm.weight"), GgufTensorRole::kVector);

    // attention linears (keep-quant Q8_0) + gate + per-head QK-norms (f32)
    lw.attn.q_proj = ctx.Mw(Blk(l, "attn_q.weight"));
    lw.attn.k_proj = ctx.Mw(Blk(l, "attn_k.weight"));
    lw.attn.v_proj = ctx.Mw(Blk(l, "attn_v.weight"));
    lw.attn.o_proj = ctx.Mw(Blk(l, "attn_output.weight"));
    lw.attn.g_proj = ctx.Mw(Blk(l, "attn_gate.weight"));
    lw.attn.q_norm = ctx.Vec(Blk(l, "attn_q_norm.weight"), GgufTensorRole::kVector);
    lw.attn.k_norm = ctx.Vec(Blk(l, "attn_k_norm.weight"), GgufTensorRole::kVector);

    if (lw.is_dense) {
      lw.mlp.gate_proj = ctx.Mw(Blk(l, "ffn_gate.weight"));
      lw.mlp.up_proj = ctx.Mw(Blk(l, "ffn_up.weight"));
      lw.mlp.down_proj = ctx.Mw(Blk(l, "ffn_down.weight"));
    } else {
      lw.moe.router = ctx.Vec(Blk(l, "ffn_gate_inp.weight"), GgufTensorRole::kVector);  // f32 [E,H]
      lw.moe.e_score_correction_bias =
          ctx.Vec(Blk(l, "exp_probs_b.bias"), GgufTensorRole::kVector);  // f32 [E]
      lw.moe.experts_gate = ctx.Sew(Blk(l, "ffn_gate_exps.weight"), E);  // Q4_K [E*moeI,H]
      lw.moe.experts_up = ctx.Sew(Blk(l, "ffn_up_exps.weight"), E);      // Q4_K [E*moeI,H]
      lw.moe.experts_down = ctx.Sew(Blk(l, "ffn_down_exps.weight"), E);  // Q5_K [E*H,moeI]
      lw.moe.shared_gate = ctx.Mw(Blk(l, "ffn_gate_shexp.weight"));      // Q8_0
      lw.moe.shared_up = ctx.Mw(Blk(l, "ffn_up_shexp.weight"));
      lw.moe.shared_down = ctx.Mw(Blk(l, "ffn_down_shexp.weight"));
    }
  }

  w.accounted_tensors = static_cast<int64_t>(ctx.consumed.size());
  w.has_gguf_weights = true;
  return w;
}

LagunaWeights LoadLagunaFromGguf(const GgufFile& gguf, const HfConfig& config,
                                 const GgufLoadPolicy* policy) {
  (void)config;  // geometry resolved from the GGUF KV (self-describing vehicle)
  return LoadLagunaFromGgufShards({&gguf}, policy);
}

}  // namespace vllm
