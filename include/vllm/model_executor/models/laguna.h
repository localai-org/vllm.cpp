// Laguna-S-2.1 (`LagunaForCausalLM` / `model_type=laguna`) — the ADDITIVE model
// TU skeleton for the Poolside Laguna bring-up (W1/W2). This header defines the
// config parse + the (scaffolded) weight layout + the forward / KV-cache seams;
// the genuinely-NEW primitives (per-head softplus attention output gate, the
// UNGROUPED sigmoid-noaux router, the per-layer VARIABLE Q-head runner wiring)
// are STUBBED with precise `TODO(W3/W4)` port markers and a `VT_CHECK(false, ...)`
// forward so the TU BUILDS but a forward LOUDLY reports the pending brick — never
// a silent wrong answer. The full forward + the strict dual-oracle gate are NAMED
// residuals (W3/W4). See `.agents/specs/laguna-s21-scope-2026-07-30.md` +
// `.agents/specs/laguna-s21-w1w2-2026-07-30.md`.
//
// ─── ~85-90% REUSE, ZERO new compute kernel (the extensibility payoff) ────────
// Every heavy component Laguna needs already exists in-tree from a prior model.
// The forward COMPOSES the reuse; only three small HOST ops are genuinely new.
//
//   COMPONENT                         REUSE SOURCE (file:line ported FROM)
//   ─────────────────────────────────────────────────────────────────────────
//   Q4_K/Q5_K/Q6_K/Q8_0 keep-quant    src/vt/cuda/cuda_quant_dot.cu DotQ4K /
//     decode  (feared "big kernel")     DotSuperblock<kQ4_K>; CPU cpu_quant_dot.cpp
//                                       VecDotQ4_K — ALREADY LANDED for ds4. ZERO new.
//   GGUF keep-quant load + name-map    src/vllm/model_executor/model_loader/
//                                       gguf_keep_quant.cpp + deepseek_v4_weights.cpp
//                                       LoadDeepseekV4FromGguf (the blk.N.* mirror);
//                                       qwen3_5_gguf_weights.cpp (dequant small tensors)
//   MoE: sigmoid noaux_tc +            src/vllm/model_executor/models/
//     e_score_correction_bias +         deepseek_v2.cpp:340-365 (RunMoeBlock, noaux
//     shared expert + routed_scaling    sigmoid + bias) + deepseek_v2_weights.cpp:186-320
//                                       (has_e_score_correction_bias, routed_scaling).
//                                       NEW = the UNGROUPED variant (drop ds2's group step).
//   Interleaved sliding-window(512)    src/vllm/model_executor/models/gemma3.cpp +
//     attn (1:3 global:sliding)          gemma3_registry.cpp:103-121 (layer_types /
//                                       is_sliding, window masked at the kernel).
//   Dual per-layer RoPE (YaRN full-    src/vllm/model_executor/models/
//     attn / plain sliding)              olmo2_weights.cpp:198-217 BuildOlmo3YarnCache
//                                       (get_rope yarn) — build TWO caches, select by
//                                       layer_types.
//   Partial rotary (full-attn dim 64)  src/vllm/model_executor/models/phi_weights.cpp
//                                       (rotary_dim = head_dim * partial_rotary_factor).
//   Per-head SOFTPLUS attn out-gate    NEW small op. Softplusf exists
//                                       (gemma4_audio.cpp:20). g_proj: hidden->num_heads,
//                                       softplus fp32, broadcast over head_dim.
//   Variable per-layer Q-head count    NEW runner wiring, extends the Gemma-4
//     (48 global / 72 sliding)           heterogeneous-per-layer-KV path (G1b, task #148).
//   RMSNorm / SwiGLU dense MLP (layer  shared vt::RmsNorm + MlpGateUpMethodBase +
//     0) / GQA / embed / untied head    FusedChain(kFusedAddRmsNorm). Config-drive only.
//
// ─── ORACLE (W1 decision) ────────────────────────────────────────────────────
// vLLM has a NATIVE `vllm/model_executor/models/laguna.py` (landed before our pin
// 555967922 / 0.26.0.dev0), so MIRROR-vLLM applies and the pinned oracle is
// EXPECTED to construct + run the config (config-constructs check per
// oracle-gateability-model-runs-not-config-constructs). Dual-oracle plan (ds4
// pattern): (a) vLLM on poolside/Laguna-S-2.1-NVFP4/-FP8 (fits GB10 119 GiB; BF16
// 235 GiB does NOT) for a behavior/coherence golden; (b) llama.cpp Poolside-fork
// `laguna` branch on the SAME UD-Q4_K GGUF for the token-exact greedy gate. No
// 73 GB download this increment — that is the W4 follow-on.
#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"          // PagedKvCache, ForwardLogits
#include "vllm/model_executor/models/qwen3_5_weights.h"  // OwnedTensor
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"  // CommonAttentionMetadata
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"

namespace vllm {

class SafetensorsFile;
class GgufFile;
struct GgufLoadPolicy;

// Every Laguna config field the loader/forward consume, resolved ONCE from the
// HfConfig. Laguna keys (`num_experts`, `moe_routed_scaling_factor`,
// `num_attention_heads_per_layer`, the nested `rope_parameters`, ...) are mostly
// NOT on the typed HfConfig struct, so most are read from `config.raw`. Values in
// the comments are the shipped `poolside/Laguna-S-2.1/config.json` (VERIFIED
// 2026-07-30, see the scope spec §1).
struct LagunaParams {
  // --- shared geometry ---
  int64_t hidden_size = 0;          // 3072
  int64_t num_hidden_layers = 0;    // 48
  int64_t vocab_size = 0;           // 100352
  int64_t num_attention_heads = 0;  // 48 (BASE / global layers)
  int64_t num_key_value_heads = 0;  // 8  (GQA)
  int64_t head_dim = 0;             // 128
  int64_t intermediate_size = 0;    // 12288 (dense MLP, layer 0 only)
  float rms_norm_eps = 1e-6f;       // 1e-6
  bool tie_word_embeddings = false; // false (untied lm_head)
  int64_t max_position_embeddings = 0;  // 1048576 (1M)

  // --- interleaved attention (1:3 global:sliding) ---
  int64_t sliding_window = 0;  // 512
  // Per-layer attention regime: "full_attention" (global) / "sliding_attention".
  // 48 entries, 1:3 pattern -> 12 global + 36 sliding.
  std::vector<std::string> layer_types;
  // Per-layer Q-head COUNT (global=48, sliding=72). KV heads = 8 both, head_dim
  // 128 both -> GQA group 6 (global) / 9 (sliding). This is the riskiest bit:
  // the runner does per-layer KV head_dim (Gemma-4) but not per-layer Q-head COUNT.
  std::vector<int64_t> num_attention_heads_per_layer;
  bool IsGlobalLayer(int64_t l) const {
    return l < static_cast<int64_t>(layer_types.size()) &&
           layer_types[static_cast<size_t>(l)] == "full_attention";
  }
  // Per-layer VARIABLE Q-head wiring (the riskiest bit). KV heads + head_dim are
  // uniform; only the Q-head COUNT (and thus the GQA group + q_proj width) vary.
  int64_t QHeadsForLayer(int64_t l) const {
    if (l >= 0 && l < static_cast<int64_t>(num_attention_heads_per_layer.size()))
      return num_attention_heads_per_layer[static_cast<size_t>(l)];
    return num_attention_heads;  // fallback = base (global) count
  }
  int64_t QDimForLayer(int64_t l) const { return QHeadsForLayer(l) * head_dim; }
  int64_t GqaGroupForLayer(int64_t l) const {
    return num_key_value_heads > 0 ? QHeadsForLayer(l) / num_key_value_heads : 0;
  }
  // The dual per-layer RoPE regime + finite window are keyed off layer type.
  int64_t RotaryDimForLayer(int64_t l) const {
    return IsGlobalLayer(l) ? rotary_dim_full : rotary_dim_sliding;
  }
  int64_t WindowForLayer(int64_t l) const {
    return IsGlobalLayer(l) ? 0 : sliding_window;  // 0 => full causal
  }

  // --- per-head softplus attention OUTPUT gate (all layers) ---
  // gate = softplus(g_proj(x).float()).type_as(attn); broadcast over head_dim.
  // g_proj: [num_heads_of_layer, hidden]. NEW small op (Softplusf exists).
  bool per_head_output_gate = true;  // config `gating == "per-head"`

  // --- per-head QK-RMSNorm (VERIFIED W4 from the real GGUF) ---------------------
  // The GGUF ships `blk.N.attn_q_norm.weight` / `attn_k_norm.weight`, both F32
  // [head_dim] — Laguna applies an RMSNorm to EACH head's q and k vector before
  // RoPE (the Qwen3/OLMo-2 QK-norm pattern). The W1-W3 scope MISSED this (there is
  // no `qk_layernorm` flag in config.json; it surfaced only in the GGUF tensor
  // map), so the W3 forward omitted it. VERIFIED 2026-07-31 by dumping the GGUF
  // tensor-info headers (blk.*.attn_q_norm/attn_k_norm F32 [128]).
  bool has_qk_norm = true;

  // --- MoE (layers 1..47; layer 0 is dense, mlp_only_layers=[0]) ---
  int64_t num_experts = 0;             // 256
  int64_t num_experts_per_tok = 0;     // 10 (top-10)
  int64_t moe_intermediate_size = 0;   // 1024
  int64_t shared_expert_intermediate_size = 0;  // 1024 (1 shared expert)
  bool norm_topk_prob = true;          // renormalize the top-k weights
  float moe_routed_scaling_factor = 2.5f;  // applied to the routed OUTPUT
  // Router = sigmoid noaux_tc + e_score_correction_bias (DeepSeek-V3 style, NOT
  // softplus — softplus lives ONLY in the attention out-gate above). The NEW
  // wiring vs ds2/glm4 is the UNGROUPED variant (use_grouped_topk == False).
  bool use_grouped_topk = false;
  bool has_e_score_correction_bias = true;
  // mlp_only_layers[]: dense-MLP layer indices (everything else is MoE). = {0}.
  std::vector<int64_t> mlp_only_layers;
  bool IsDenseLayer(int64_t l) const {
    for (int64_t d : mlp_only_layers)
      if (d == l) return true;
    return false;
  }

  // --- dual per-layer RoPE (by layer-type) ---
  // full_attention: YaRN — theta 500000, factor 128, orig_max_pos 8192,
  //   beta_fast 32 / beta_slow 1, attention_factor(mscale) 1.4852030263919618,
  //   partial_rotary_factor 0.5 -> rotary_dim 64 of 128.
  double rope_theta_full = 500000.0;
  // YaRN `factor` = ctx_len / orig_ctx. The unsloth UD-Q4_K GGUF is built for a
  // 262144 context (factor 32 = 262144/8192), NOT the HF config.json 1M context
  // (factor 128). For the same-quant token-exact gate vs llama.cpp on THIS GGUF,
  // the GGUF value (32) is authoritative — VERIFIED W4 from laguna.rope.scaling.*.
  double yarn_factor = 32.0;
  int64_t yarn_orig_max_pos = 8192;
  double yarn_beta_fast = 32.0;
  double yarn_beta_slow = 1.0;
  // llama.cpp `yarn_attn_factor` (default 1.0). The FULL mscale llama.cpp applies is
  //   mscale = yarn_attn_factor * (1 + 0.1*ln(factor))     [rope_yarn, ext_factor!=0]
  // which reproduces HF's precomputed `attention_factor` too (factor 128 -> 1.48520,
  // factor 32 -> 1.34657). See LagunaYarnMscale. The GGUF resolve keeps it 1.0 and
  // does NOT read `laguna.rope.scaling.yarn_attn_factor`, as llama.cpp does not:
  // the XS-2.1 APEX file stores the already-multiplied 1.34657 there.
  double yarn_attn_factor = 1.0;
  double partial_rotary_factor_full = 0.5;  // rotary_dim_full = 64
  int64_t rotary_dim_full = 64;
  // sliding_attention: plain RoPE — theta 10000, full 128-dim rotary.
  double rope_theta_sliding = 10000.0;
  int64_t rotary_dim_sliding = 128;
};

// Resolve + validate LagunaParams from an HfConfig (throws on unsupported).
LagunaParams ParseLagunaParams(const HfConfig& config);

// Per-family config hook (registry `parse_config`).
void ParseLagunaConfig(const HfConfig& config);

// ─── Weight layout (SCAFFOLD) ────────────────────────────────────────────────
// One Laguna self-attention block. q_proj is per-layer VARIABLE width
// (num_heads_of_layer * head_dim); k/v are fixed (num_key_value_heads * head_dim).
// g_proj is the per-head softplus output gate (num_heads_of_layer, hidden).
struct LagunaAttnWeights {
  OwnedTensor q_proj;   // raw-NK [num_heads_l*Dh, H]   (GGUF blk.N.attn_q)
  OwnedTensor k_proj;   // raw-NK [Hkv*Dh, H]           (GGUF blk.N.attn_k)
  OwnedTensor v_proj;   // raw-NK [Hkv*Dh, H]           (GGUF blk.N.attn_v)
  OwnedTensor o_proj;   // raw-NK [H, num_heads_l*Dh]   (GGUF blk.N.attn_output)
  OwnedTensor g_proj;   // raw-NK [num_heads_l, H]  (per-head softplus out-gate; GGUF attn_gate)
  OwnedTensor q_norm;   // f32 [head_dim]  per-head QK-RMSNorm (GGUF blk.N.attn_q_norm) — VERIFIED W4
  OwnedTensor k_norm;   // f32 [head_dim]  per-head QK-RMSNorm (GGUF blk.N.attn_k_norm) — VERIFIED W4

  // ─── Fused decode projection (NVFP4 arm; steady-decode lever) ───────────────
  // The NVFP4 checkpoint keeps q/k/v/g in BF16, and all four read the SAME
  // input-norm activation, so they stack row-wise into ONE bf16 [qdim + 2*kvdim +
  // Hq, H] tensor consumed by a SINGLE wider M=1 GEMV in the resident/graph decode
  // (row order q, k, v, g). Fusing raises the GEMV's N (grid = N blocks) so the
  // small-N projections stop under-filling the GPU — MEASURED #1 steady-decode
  // cost (52.7% of decode GPU time was separate bf16 M=1 GEMVs). ADDITIVE: the
  // split q/k/v/g above are KEPT (the host/prefill forwards read them); this is a
  // load-time row-block concat, populated ONLY by the NVFP4 loader (Empty() on the
  // GGUF keep-quant path, which never reaches the resident decode). Byte-exact in
  // principle (each output row is an independent fp32-accumulated dot); the wider N
  // may latch a different cuBLASLt algo -> near-tie (the accepted device regime).
  OwnedTensor qkvg_proj;  // bf16 [qdim + 2*kvdim + Hq, H]  (q | k | v | g stacked)
};

// Laguna dense SwiGLU MLP (layer 0 only). silu(gate)*up -> down (separate tensors).
struct LagunaMlpWeights {
  // GGUF ships SEPARATE dense FFN tensors (blk.0.ffn_gate/up/down.weight); the
  // W1-W3 scaffold assumed a merged gate_up — corrected to the real name-map (W4).
  OwnedTensor gate_proj;  // raw-NK [I, H]  (ffn_gate)
  OwnedTensor up_proj;    // raw-NK [I, H]  (ffn_up)
  OwnedTensor down_proj;  // raw-NK [H, I]  (ffn_down)
};

// Laguna MoE block (layers 1..47): sigmoid noaux_tc router (+ e_score_correction_
// bias) UNGROUPED top-10 of 256 + 1 shared expert, routed_scaling 2.5 on output.
struct LagunaMoeWeights {
  OwnedTensor router;                 // raw-NK [num_experts, H]  (GGUF ffn_gate_inp, F32)
  OwnedTensor e_score_correction_bias;  // f32 [num_experts]      (GGUF exp_probs_b.bias)
  // 256 routed experts + 1 shared expert. The GGUF ships SEPARATE gate/up tensors
  // (VERIFIED W4: ffn_gate_exps Q4_K [E,moe_I,H] + ffn_up_exps Q4_K + ffn_down_exps
  // Q5_K); the W1-W3 scaffold assumed a MERGED gate_up — corrected here. The
  // keep-quant expert slabs stay COMPRESSED for the GGUF path (ds4 kStackedExpert),
  // consumed per-selected-expert via a row-slice keep-quant GEMM.
  OwnedTensor experts_gate;   // grouped [E, moe_I, H]  (ffn_gate_exps, Q4_K)
  OwnedTensor experts_up;     // grouped [E, moe_I, H]  (ffn_up_exps,   Q4_K)
  OwnedTensor experts_down;   // grouped [E, H, moe_I]  (ffn_down_exps, Q5_K)
  OwnedTensor shared_gate;    // [moe_I, H]  (ffn_gate_shexp, Q8_0)
  OwnedTensor shared_up;      // [moe_I, H]  (ffn_up_shexp,   Q8_0)
  OwnedTensor shared_down;    // [H, moe_I]  (ffn_down_shexp, Q8_0)

  // ─── Fused router+shared-gate+shared-up (NVFP4 arm; steady-decode lever) ─────
  // The NVFP4 checkpoint keeps the router + shared-expert gate/up in BF16, all
  // reading the SAME post-attn activation (K=H), so they stack row-wise into ONE
  // bf16 [E + 2*moe_I, H] tensor consumed by a SINGLE M=1 GEMV in the resident/
  // graph decode (row order router, shared_gate, shared_up). Slice: [0,E)=router
  // logits -> sigmoid_topk; [E, E+moe_I)=shared_gate; [E+moe_I, E+2*moe_I)=
  // shared_up (pre-SiLU). ADDITIVE (the split tensors above are KEPT for the host/
  // prefill forwards); populated ONLY by the NVFP4 loader (Empty() otherwise).
  OwnedTensor router_shared_gu;  // bf16 [E + 2*moe_I, H]  (router | s_gate | s_up)

  // NVFP4 arm (task #230, spec laguna-nvfp4-arm-2026-07-31.md). The
  // poolside/Laguna-S-2.1-NVFP4 checkpoint keeps attention + dense-MLP + shared
  // norms/router/embed/lm_head in BF16 (loaded into the OwnedTensor fields
  // above / in LagunaAttnWeights, LagunaMlpWeights, LagunaWeights) and quantizes
  // ONLY the routed + shared EXPERTS to W4A4 NVFP4 (weight_packed U8 [N,K/2] +
  // weight_scale F8 [N,K/16] + weight_global_scale f32 + input_global_scale f32).
  // These per-expert Nvfp4Weight fields mirror qwen3_5 MoeBlockWeights.expert_*_fp4
  // and are consumed by kMoeGroupedGemmNvfp4 / vt::MatmulNvfp4Fp4 in the N2
  // forward-branch. ADDITIVE + dead until the N1 loader (LoadLagunaNvfp4)
  // populates them; the GGUF keep-quant path never touches them (Empty()).
  std::vector<Nvfp4Weight> experts_gate_fp4;  // E * [N=moe_I, K=H]
  std::vector<Nvfp4Weight> experts_up_fp4;    // E * [N=moe_I, K=H]
  std::vector<Nvfp4Weight> experts_down_fp4;  // E * [N=H, K=moe_I]
  Nvfp4Weight shared_gate_fp4;   // [N=moe_I, K=H]
  Nvfp4Weight shared_up_fp4;     // [N=moe_I, K=H]
  Nvfp4Weight shared_down_fp4;   // [N=H, K=moe_I]

  // Resident Marlin MoE constants (issue #237; see ResidentSlot in
  // qwen3_5_weights.h). Same defect and same fix as the Qwen3.5 blocks: this
  // used to be a process-lifetime map keyed on the address of these weights.
  ResidentSlot resident_marlin;
};

// One Laguna decoder layer.
struct LagunaLayerWeights {
  OwnedTensor input_norm;       // [H]  pre-attn RMSNorm
  OwnedTensor post_attn_norm;   // [H]  pre-MLP/MoE RMSNorm
  LagunaAttnWeights attn;
  bool is_dense = false;        // layer 0
  LagunaMlpWeights mlp;         // valid iff is_dense
  LagunaMoeWeights moe;         // valid iff !is_dense
};

// The full Laguna model weights. W5: the GGUF keep-quant tower is REAL — the big
// GEMM weights (attn q/k/v/o/gate Q8_0, dense-L0 ffn Q8_0, shared-expert Q8_0,
// routed experts Q4_K/Q5_K, lm_head Q8_0) stay COMPRESSED as block-typed
// OwnedTensors (consumed keep-quant via vt::MatmulBT); the small norms / router /
// bias / embed dequant to f32. Mirrors DeepseekV4Weights (gguf keep-quant tower).
struct LagunaWeights {
  LagunaParams params;
  OwnedTensor embed;    // [V, H]  f32 gather table (dequant token_embd)
  OwnedTensor norm;     // [H]  final RMSNorm (f32)
  OwnedTensor lm_head;  // [V, H]  untied, keep-quant Q8_0 (or f32 tiny path)
  // Dual per-layer RoPE caches (OLMo-3 BuildOlmo3YarnCache pattern): one YaRN
  // (full-attn, 64-dim) + one plain (sliding, 128-dim). Built at load.
  OwnedTensor rope_cos_sin_yarn_full;  // bf16 [rows, 64]
  OwnedTensor rope_cos_sin_plain_slide;  // bf16 [rows, 128]
  std::vector<LagunaLayerWeights> layers;
  int64_t accounted_tensors = 0;  // W2 accounting-pass count
  bool has_gguf_weights = false;  // W5: the keep-quant tower is materialized
  bool has_nvfp4_weights = false;  // N1b/N2: the safetensors NVFP4 W4A4 arm is materialized
};

// Safetensors loader (BF16 checkpoint; MEMORY-INFEASIBLE on one GB10 at 235 GiB,
// present for structural completeness + the NVFP4 behavior path). Encodes the
// Laguna name-map + the W2 accounting pass.
LagunaWeights LoadLagunaForCausalLMWeights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config);

// GGUF keep-quant loader (the single-GB10 vehicle: unsloth/Laguna-S-2.1-GGUF
// UD-Q4_K_XL, ~69 GiB, 3 shards). Reuses the ds4 gguf_keep_quant path + a Laguna
// blk.N.* name-map; the MoE expert blocks stay COMPRESSED (Q4_K/Q5_K/Q6_K/Q8_0
// all already decoded — ZERO new kernel). Single-shard wrapper of the multi-shard
// entrypoint below (the real model ships 3 shards; a single-file GGUF works too).
LagunaWeights LoadLagunaFromGguf(const GgufFile& gguf, const HfConfig& config,
                                 const GgufLoadPolicy* policy = nullptr);

// W5 multi-shard GGUF keep-quant loader. `shards[0]` is the METADATA shard (the
// laguna.* KV geometry + tokenizer live only in shard-1); the tensors are routed
// to whichever shard holds them by scanning every shard's tensor table (the
// UD-Q4_K_XL 3-shard split: shard-1 = header only, shards 2/3 = the 814 tensors).
// Loader-local — NO reader rewrite, NO merge. Keep every `GgufFile` alive for the
// weights' lifetime (mmap-borrowed keep-quant blocks refcount their shard's map).
LagunaWeights LoadLagunaFromGgufShards(const std::vector<const GgufFile*>& shards,
                                       const GgufLoadPolicy* policy = nullptr);

// Resolve LagunaParams from the GGUF `laguna.*` KV of the metadata shard (mirrors
// DeepseekV4ParamsFromGguf): per-layer head_count array, dual-rope keys, MoE keys.
LagunaParams LagunaParamsFromGguf(const GgufFile& meta);

// The HF-shaped config a `laguna` GGUF presents to the engine, so
// `LoadedEngine::FromModelDir` can resolve the registry row and size the engine
// from the file. Built FROM `LagunaParamsFromGguf`, so `ParseLagunaParams` on the
// result recovers the same geometry the GGUF weight loader resolves itself.
HfConfig LagunaHfConfigFromGguf(const GgufFile& meta);

// W5 REAL keep-quant forward. The `LagunaModel::Forward` composition with the ~9
// GEMM sites routed through vt::MatmulBT (keep-quant on the block-typed weight,
// dispatches to CPU/CUDA kMatmulBTQuant) instead of the f32 MatmulNK reference.
// Stateless whole-sequence recompute (mirrors DeepseekV4ForwardGguf); the greedy
// driver (examples/laguna_gen) loops it. `q` = CPU queue (default) or a CUDA queue
// (--gpu: the block-quant GEMMs run on the GB10 off the unified-memory blocks).
std::vector<float> LagunaForwardGguf(const LagunaWeights& w, vt::Queue& q,
                                     const std::vector<int32_t>& token_ids,
                                     const std::vector<int32_t>& positions,
                                     const std::vector<int32_t>& logits_indices);

// VT_LAGUNA_SHARED_FP4 (default OFF): keep the per-layer SHARED expert fp4-resident
// (XS-NVFP4 tower) and route it through the Marlin W4A16 grouped GEMM instead of
// dequantizing to bf16 — 4x less shared-tower DRAM traffic on the M=1 decode GEMV,
// matching vLLM's fp4 shared expert. Read once per process; `=1` opts in.
bool LagunaSharedFp4Enabled();

// True iff the loaded weights carry the fp4 shared expert (the flag is a no-op on a
// bf16-shared tower like S-2.1-NVFP4).
bool LagunaHasFp4SharedExpert(const LagunaWeights& w);

// Populate the fp4 shared-expert fields (`moe.shared_{gate,up,down}_proj_fp4`) from
// the checkpoint shards when VT_LAGUNA_SHARED_FP4=1 and the tower quantizes the
// shared expert. Call AFTER LoadLagunaForCausalLMWeights and BEFORE the shards are
// released (the fp4 bytes are copied out). When it loads, the now-dead bf16 shared
// copies + the fused router|shared projection are freed. No-op otherwise (bf16
// tower, GGUF path, or flag off). ADDITIVE — does NOT touch laguna_weights.cpp.
void LagunaLoadSharedExpertFp4(const std::vector<SafetensorsFile>& shards, LagunaWeights& w);

// N5 campaign-B: pre-build all routed-expert MARLIN W4A16 residents at model-LOAD
// time (mirrors vLLM's process_weights_after_loading), so the repack cost is not a
// first-token TTFT spike. No-op unless the Marlin path is enabled (VT_LAGUNA_MARLIN_MOE)
// and on GPU (and unless built with VT_MARLIN_NVFP4). Call once after loading the
// NVFP4 tower; the forward's lazy build then finds every resident ready.
void LagunaBuildMarlinResidents(vt::Queue& q, const LagunaWeights& w);

// ─── W6: per-layer K/V cache for INCREMENTAL decode ──────────────────────────
// Laguna is MIXED attention: 12 GLOBAL layers (full causal — cache grows unbounded)
// + 36 SLIDING-WINDOW-512 layers (cache CAPPED at the 512 window — the oldest rows
// are evicted, mirror of gemma2/3 is_sliding). The cached K/V are POST-QK-RMSNorm,
// POST-RoPE keys and RAW values; RoPE/QK-norm depend only on a token's OWN absolute
// position, and Laguna attention is causal, so a token's cached K/V equal the values
// the full-recompute forward would produce at any later step (the KV-cache identity).
// Stored at f32 to stay BIT-EXACT to LagunaForwardGguf. Mirror of DeepseekV4KvCache
// (deepseek_v4.h), extended from MLA's single `deck` latent to GQA multi-head K/V.
struct LagunaKvCache {
  std::vector<std::vector<float>> k;  // [layer] -> flat [rows_l * Hkv * Dh]
  std::vector<std::vector<float>> v;  // [layer] -> flat [rows_l * Hkv * Dh]
  // Global position of the FIRST cached row of each layer. Global layers keep 0
  // (all history); sliding layers advance it as they evict (rows capped at window).
  std::vector<int64_t> first_pos;     // [layer]
  int64_t len = 0;                    // tokens processed = global position of next
  int64_t head_dim = 0;
  int64_t kv_heads = 0;
  // --- Resident-decode device KV (Brick A1): fixed-capacity per-layer buffers
  // (unified → device-accessible). The prefill KV (host `k`/`v`) is migrated in
  // ONCE on the first resident step; thereafter each token's K/V is appended
  // ON-STREAM at index dev_rows (no host insert, no per-layer DrainQueue). No
  // eviction — the decode_attn window mask handles sliding layers (full-deck),
  // so dev_first_pos stays the prefill value (graph-ready: a per-layer constant).
  std::vector<std::vector<float>> k_dev, v_dev;  // [layer] max_cap*kvdim (f32 path)
  // LEVER A (VT_LAGUNA_KV_BF16): the bf16 alternative to k_dev/v_dev — HALF the KV DRAM
  // (2 B/elem), moved every layer every step, matching vLLM's bf16 KV. Only ONE of the two
  // (f32 k_dev OR bf16 k_dev16) is allocated per run (env-gated), so the proven f32 path is
  // byte-identical when bf16 is off. Stored as raw bf16 bits (uint16_t); passed to the CUDA
  // kernels via the float* param as an address pun (the kernels reinterpret to bf16).
  std::vector<std::vector<uint16_t>> k_dev16, v_dev16;  // [layer] max_cap*kvdim (bf16 path)
  std::vector<int64_t> dev_first_pos;            // [layer] frozen at migration
  std::vector<int64_t> dev_rows;                 // [layer] cached-row count (grows 1/token)
  int64_t max_cap = 0;
  bool resident_ready = false;
  // Brick A2: opaque persistent decode-CUDA-graph state (the fixed-capacity KV +
  // persistent scratch + captured graph), lazily built on the first graphed decode
  // step and reused across steps. void so this public header stays decoupled from the
  // CUDA-only LagunaGraph type; the deleter (set in laguna.cpp) frees it. Mirror of
  // `DeepseekV4KvCache::decode_graph` (deepseek_v4.h; cited by SYMBOL, not by line).
  std::shared_ptr<void> decode_graph;
  // On-device greedy sample result (VT_LAGUNA_ONDEV_SAMPLE): the decode graph argmaxes
  // its logits ON-DEVICE and leaves the winning token id here (>=0) so the driver never
  // downloads the full vocab for a host argmax. -1 when off / not a graphed decode step
  // (prefill + host-argmax fallback), so the caller host-argmaxes the returned logits.
  int32_t last_sampled = -1;
  void Reset(int64_t num_layers, int64_t hd, int64_t hkv) {
    k.assign(static_cast<size_t>(num_layers), {});
    v.assign(static_cast<size_t>(num_layers), {});
    first_pos.assign(static_cast<size_t>(num_layers), 0);
    len = 0;
    head_dim = hd;
    kv_heads = hkv;
    k_dev.clear();
    v_dev.clear();
    k_dev16.clear();
    v_dev16.clear();
    dev_first_pos.clear();
    dev_rows.clear();
    max_cap = 0;
    resident_ready = false;
    decode_graph.reset();
    last_sampled = -1;
  }
};

// W6 INCREMENTAL-decode variant of LagunaForwardGguf. On the FIRST call (prefill)
// pass all prompt tokens with cache.len==0 (positions 0..P-1); each later call
// passes ONE new token with positions={cache.len} and logits_indices={0}. The
// forward appends each token's per-layer K/V to the cache (sliding layers evict
// beyond the 512 window) and attends the new query over the cached K/V — TOKEN-
// IDENTICAL to LagunaForwardGguf run over the growing context (a pure equivalence:
// same tokens, ~ctx× fewer FLOPs), which kills the O(n²) full-recompute. Mirror of
// DeepseekV4ForwardGgufCached. `logits_indices` are LOCAL indices into THIS call's
// tokens.
std::vector<float> LagunaForwardGgufCached(const LagunaWeights& w, vt::Queue& q,
                                           LagunaKvCache& cache,
                                           const std::vector<int32_t>& token_ids,
                                           const std::vector<int32_t>& positions,
                                           const std::vector<int32_t>& logits_indices);

// The Laguna forward. STUB (W3/W4): composes the reuse (variable-Q-head GQA +
// interleaved sliding-window mask + dual per-layer RoPE + per-head softplus
// out-gate; dense MLP at layer 0; ungrouped sigmoid-noaux MoE at layers 1..47;
// untied lm_head). Both entrypoints VT_CHECK(false, ...) so a forward LOUDLY
// reports the pending brick.
class LagunaModel {
 public:
  static std::vector<float> Forward(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});

  static ForwardLogits ForwardDevice(
      const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
      const v1::CommonAttentionMetadata& attn_meta,
      const std::vector<PagedKvCache>& attn_kv, const LagunaWeights& weights,
      const HfConfig& config, vt::Queue& queue,
      const std::vector<int32_t>& logits_indices = {});
};

// KV-cache spec builder. One FULL-ATTENTION KV group over all layers; the
// interleaved sliding-window layers are masked at the attention kernel (window
// 512), NOT by a smaller SlidingWindowSpec cache — the gemma3 topology the
// shape-agnostic runner already handles (gemma3_registry.cpp:103-121).
v1::KVCacheConfig MakeLagunaKVCache(const HfConfig& config, int block_size,
                                    int num_blocks);

}  // namespace vllm
