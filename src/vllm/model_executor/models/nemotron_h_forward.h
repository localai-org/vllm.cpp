// Nemotron-H (`NemotronHForCausalLM`) — W4, the FORWARD
// ([spec](../../../../.agents/specs/nemotron-h-model.md) §4 W4, issue #517).
//
// W3 made the architecture KNOWN, PARSED, ENUMERATED and KV-SHAPED
// (nemotron_h.h). This header is where it first COMPUTES: the hybrid 52-layer
// decoder loop over the three block kinds `layers_block_type` names — 23 Mamba2,
// 6 GQA attention at indices {5,12,19,26,33,42}, 23 non-gated relu² MoE.
//
// ─── WHAT THIS IS A PORT OF (file:line on BOTH sides, @ pin 555967922) ───────
//   NemotronHModel::Forward   <-  nemotron_h.py:604-660 — the single-branch
//                                 pre-norm residual stream. NOTE: unlike Llama,
//                                 a NemotronH decoder layer has exactly ONE norm
//                                 and ONE mixer (:301-313, :344-355, :390-401,
//                                 :525-528), not an attention branch plus an MLP
//                                 branch. 52 layers == 52 (norm, mixer) pairs.
//   Mamba2Mixer               <-  layers/mamba/mamba_mixer2.py:548-586 (in_proj →
//                                 conv+SSM → gated group norm → out_proj), the
//                                 split points at :532-543 and :692-696, the
//                                 prefill scan call at :866-891.
//   Attention                 <-  nemotron_h.py:473-486. Ported deliberately
//                                 WITHOUT any positional embedding — see
//                                 `kNemotronHAttentionHasNoRope`.
//   MoE                       <-  nemotron_h.py:126-256 + the W2 arm recorded in
//                                 spec §6a (up → relu² → down → scaled combine).
//
// ─── SCOPE, exactly ─────────────────────────────────────────────────────────
// This is the HOST (CPU) reference forward, the same cadence Kimi-Linear's W2-W6
// took (kimi_linear_forward.cpp:1-26) and DeepSeek-V4's `DeepseekV4ForwardHost`
// before it: a real forward composed out of the ALREADY-LANDED `vt::` ops, run
// on a CPU queue, so the mechanism is gateable before any GPU is involved. What
// it is NOT:
//   * NOT a weight loader. Nothing here reads a checkpoint. The safetensors
//     materialization of the 18487 tensors `EnumerateNemotronHTensors` names —
//     including the NVFP4 W4A16 g16 experts and the FP8 W8A8 mamba projections —
//     is still owed, and `NemotronHForward` REFUSES BY NAME when handed
//     unmaterialized weights rather than returning zeros.
//   * NOT the MTP head (W5), NOT the GGUF arm (W7), NOT a speed claim.
//   * NOT the paged/device runner path (W6). The recurrent state here is FRESH
//     per call and single-sequence; `MakeNemotronHKVCache` already declares the
//     het-KV topology the runner will drive, and wiring it is W6's.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/models/nemotron_h.h"
// OwnedTensor — the tree's SHARED weight-residency type (qwen3_5_weights.h:47),
// included here exactly as gemma3.h / glm4.h / qwen3.h / deepseek_v2.h include
// it. A2-R (#810, .agents/specs/nemotron-h-abi-e2e.md) moves the weights the
// DEVICE path consumes onto it so they upload ONCE through
// `dense_attn::ResidentWeight`. See the residency note on NemotronHOwned below
// for why the quantized weights deliberately stay behind.
#include "vllm/model_executor/models/qwen3_5_weights.h"
// A2-P (#810): `ForwardLogits`, `PagedKvCache` and `GdnStateCache` — the three
// runner-owned types the paged forward consumes. This is the SHARED header the
// runner itself allocates them through (runner.cpp:906-916, :970-978), never a
// NemotronH-local restatement of their layout.
#include "vllm/model_executor/models/qwen3_5.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

namespace vllm {

// A2-P: the paged forward takes `ModelForwardInput` WHOLE, mirroring
// `KimiLinearModel::ForwardPaged(input, weights)` (kimi_linear_registry.cpp:101)
// — the only in-tree instance of exactly this fold. A forward declaration is
// enough for a by-reference parameter and keeps `model_registry.h` (which
// includes the MTP and multimodal surfaces) out of every consumer of this
// header.
struct ModelForwardInput;

// NemotronH's attention carries NO positional embedding of any kind. This is not
// an omission to be repaired later: `models/nemotron_h.py` @ 555967922 contains
// ZERO occurrences of `rope`, `rotary` or `Rotary` — `NemotronHAttention.__init__`
// (:415-471) builds `qkv_proj`, `o_proj` and `Attention` and nothing else, and
// `.forward` (:473-486) is qkv → split → attn → o_proj with no rotation step.
// Position information reaches the residual stream through the 23 Mamba2 layers'
// recurrence instead, which is what makes a 1M context cheap here.
//
// The released config.json DOES ship `rope_theta: 10000` and
// `partial_rotary_factor: 1.0`, and `NemotronHParams` parses both (nemotron_h.h)
// — they are INERT for this architecture. Applying them would be numerically
// plausible, would not change a single tensor SHAPE, and on a short prompt might
// not move a token; the forward gate mutates exactly this to prove it is caught.
inline constexpr bool kNemotronHAttentionHasNoRope = true;

// ─── host weights ───────────────────────────────────────────────────────────
//
// One owned, contiguous host tensor in a DECLARED dtype. Deliberately a lean
// byte buffer rather than the safetensors-backed `OwnedTensor`: nothing in W4
// loads a checkpoint, so pulling the loader's includes into this header would
// buy nothing, and the byte-buffer form is what a later loader writes into
// directly.
//
// The DTYPE IS THE POINT, not an implementation detail. vLLM resolves ONE model
// dtype and every layer inherits it (AGENTS.md); this struct carries it
// explicitly so a gate can assert the memory format instead of inferring it from
// matching tokens — a dtype that is too WIDE is numerically correct and
// therefore invisible to a token comparison.
// The MEMORY FORMAT a materialized weight is held in. This is the whole point
// of the enum existing: the released checkpoint is `quant_algo:
// MIXED_PRECISION` and is NOT uniform — routed/shared experts and `lm_head` are
// NVFP4 W4A16 group-16, the 46 mamba `in_proj`/`out_proj` are FP8 W8A8 static,
// and attention, conv1d, gates, norms and embeddings are plain bf16. Holding
// any of those in a WIDER form than it ships is numerically correct, invisible
// to a token gate, and moves the wrong bytes (AGENTS.md).
//
// It is also what makes the checkpoint fit at all. The 5888 routed-expert
// projections alone are 29.4e9 parameters; dequantized to bf16 at load they are
// 58.7 GB, against 16.5 GB packed (15.4 GiB — nibbles plus group scales, and
// both figures here are DECIMAL GB). So `kNvfp4W4A16G16` is not an optimization
// — a load that widens it does not run on any box this project owns.
enum class NemotronHWeightForm : uint8_t {
  // `bytes` holds Numel() elements of `dtype`, contiguous. Every W4-era weight.
  kDense,
  // ModelOpt `W4A16_NVFP4`, `group_size=16` (spec §1, config_groups group_1):
  //   bytes  U8      [rows, cols/2]   two E2M1 nibbles per byte, LOW nibble
  //                                   first (the torchao/ModelOpt convention,
  //                                   `Nvfp4NibbleOrder::kLowFirst`)
  //   scale  F8_E4M3 [rows, cols/16]  one linear e4m3 scale per 16 inputs
  //   global_scale    F32 scalar      `weight_scale_2`, MULTIPLIED (not
  //                                   reciprocated)
  // `shape`/`dtype` stay LOGICAL: [rows, cols] at the model dtype, which is what
  // a dequantized view yields and what every shape check compares against.
  kNvfp4W4A16G16,
  // ModelOpt FP8 W8A8 static (config_groups group_0, the 46 mamba projections):
  //   bytes         F8_E4M3 [rows, cols]  one IEEE e4m3 byte per element
  //   global_scale  F32 scalar            `weight_scale`, MULTIPLIED
  //   input_scale   F32 scalar            the STATIC activation scale
  // The host reference forward is weight-only (W4A16-shaped): it consumes
  // `global_scale` and carries `input_scale` without applying it, because
  // nothing here quantizes the activation. That is recorded rather than
  // silently dropped — W6's device path is where the activation scale is live.
  kFp8W8A8Static,
};

// ─── RESIDENCY: why some weights are OwnedTensor and this one is not (A2-R) ───
//
// A2-R (#810) puts embeddings, the 53 RMSNorm weights and the 6 GQA attention
// layers' q/k/v/o on the DEVICE. Those are the weights that ship plain bf16, so
// they move to the tree's shared `OwnedTensor` and upload once through
// `dense_attn::ResidentWeight` (dense_attn_block.h:178). Nothing is hand-rolled:
// AGENTS.md forbids a parallel path, and adding a `d_dev` to NemotronHOwned
// would have been exactly one.
//
// NemotronHOwned SURVIVES for the weights whose device arm is not ported yet —
// the 5935 NVFP4 W4A16 g16 projections (routed + shared experts, lm_head) and
// the 46 FP8 W8A8 static mamba projections. It is the right holder for them
// BECAUSE it is not OwnedTensor: OwnedTensor carries no `form`, no group
// `scale`, no `global_scale` and no `input_scale` (qwen3_5_weights.h:47-77), so
// converting a quantized weight to it would DISCARD the memory format the
// checkpoint ships and silently commit this model to a widened one. When those
// arms land they move to the shared `Nvfp4Weight` / `Fp8Weight`
// (qwen3_5_weights.h:198, :273), never to OwnedTensor.
//
// The arithmetic behind that boundary, which is also why A2-R stops at the 6
// attention layers: the 5935 NVFP4 projections are 30.19e9 parameters, 15.8 GiB
// packed and 56.2 GiB dequantized to bf16. There is no device NVFP4->bf16
// dequant kernel in vt at all, so a "bf16 everywhere" device forward would mean
// a host dequant plus a 56.2 GiB upload — on two unified-memory boxes where
// that is a reboot, not an OOM. See nemotron_h_loader.h:36-46, which rejected
// the same design for the load.
struct NemotronHOwned {
  // The payload. Its meaning is `form`'s (see NemotronHWeightForm): dense
  // elements of `dtype` for kDense, packed nibbles for NVFP4, e4m3 bytes for
  // FP8.
  std::vector<uint8_t> bytes;
  // The LOGICAL dtype — what a dense view of this weight yields. Not the
  // storage dtype when `form != kDense`.
  vt::DType dtype = vt::DType::kF32;
  // The LOGICAL shape, likewise: [out, in] for every projection, whatever the
  // packing.
  std::vector<int64_t> shape;

  NemotronHWeightForm form = NemotronHWeightForm::kDense;
  // NVFP4 only: the per-16-element e4m3 group scales, [rows, cols/16] bytes.
  std::vector<uint8_t> scale;
  // NVFP4 `weight_scale_2` / FP8 `weight_scale`. Multiplied, never reciprocated.
  float global_scale = 1.0F;
  // FP8 W8A8 `input_scale`, carried for the device path (W6). `has_input_scale`
  // distinguishes "the checkpoint shipped 1.0" from "no scale shipped", because
  // a defaulted 1.0 that silently stands in for a missing tensor is exactly the
  // class of load defect a token gate absorbs.
  float input_scale = 1.0F;
  bool has_input_scale = false;

  bool Empty() const { return bytes.empty(); }
  bool IsDense() const { return form == NemotronHWeightForm::kDense; }
  int64_t Numel() const;
  // Bytes actually resident for this weight, payload + scales. The number a
  // load report adds up; not derivable from `shape` once `form != kDense`.
  int64_t HostBytes() const {
    return static_cast<int64_t>(bytes.size() + scale.size());
  }
  // A non-owning contiguous view over the current buffer. Rebuilt on each call
  // so it survives moves/reallocations of the owning struct. DENSE ONLY: a view
  // over packed nibbles typed as bf16 reads plausible garbage, so this refuses
  // rather than handing one out.
  vt::Tensor View(vt::Device device) const;

  // Materialize a DENSE bf16 copy of a quantized weight, through the SHARED
  // ModelOpt dequant seam (`model_loader/nvfp4_dequant.h`), which is the one
  // this tree's other ModelOpt consumers use. bf16 is the target because that is
  // what the shared seam produces AND the released checkpoint's model dtype; a
  // caller running the f32 reference arm widens the result losslessly.
  //
  // This exists because the HOST reference forward has NO NVFP4 and NO FP8 GEMM
  // — those are the CUDA `kMoeGroupedGemmNvfp4Marlin` / fp8-linear arms W6
  // selects. It is a DECLARED host dequant, counted and reported by the loader,
  // never a silent fallback: the weight keeps its quantized form in memory, and
  // only the GEMM operand is widened, transiently, at the call site.
  std::vector<uint8_t> DenseBf16() const;

  // Pack canonical f32 values into `dtype`. This is the seam a real loader
  // replaces: it hands over checkpoint bytes that are ALREADY in the model
  // dtype, and no packing happens per forward.
  static NemotronHOwned FromF32(const std::vector<float>& values, vt::DType dtype,
                                std::vector<int64_t> shape);
};

// Mamba2 mixer weights, by the names the checkpoint ships (nemotron_h.h's
// enumeration): `mixer.{in_proj,conv1d,A_log,D,dt_bias,norm,out_proj}`.
struct NemotronHMambaWeights {
  // torch Linear orientation [out, in] — vt::MatmulBT's `b [N,K]`.
  NemotronHOwned in_proj;   // [in_proj_out_features, hidden_size]
  NemotronHOwned out_proj;  // [hidden_size, mamba_intermediate_size]
  // The disk tensor is [conv_dim, 1, conv_kernel]; vt::CausalConv1dFwd takes the
  // squeezed [conv_dim, conv_kernel] view, exactly as upstream's
  // `self.conv_weights = self.conv1d.weight.view(conv_dim, K)`
  // (mamba_mixer2.py, `conv_weights`).
  NemotronHOwned conv1d_weight;  // [conv_dim, conv_kernel]
  NemotronHOwned conv1d_bias;    // [conv_dim] — use_conv_bias=true here
  // f32 BY CONTRACT, not by choice: vt::Mamba2ChunkScan validates A/D/dt_bias as
  // f32 (ops.cpp CheckMamba2*), mirroring upstream, which keeps `A` in f32
  // (`self.A = -torch.exp(self.A_log.float())`) regardless of the model dtype.
  NemotronHOwned A_log;    // [mamba_num_heads] f32
  NemotronHOwned D;        // [mamba_num_heads] f32
  NemotronHOwned dt_bias;  // [mamba_num_heads] f32
  // Mixer2RMSNormGated weight over the SSM intermediate width.
  NemotronHOwned norm_weight;  // [mamba_intermediate_size]
};

// GQA attention weights. q/k/v ship SEPARATE on disk (upstream fuses them into
// `qkv_proj` at load through its stacked-params mapping), so they are separate
// here too — that is the shape `EnumerateNemotronHTensors` already claims.
struct NemotronHAttentionWeights {
  // A2-R: OwnedTensor, because these four are what the DEVICE attention block
  // (NemotronHAttnBlock) consumes and they ship plain bf16 on the released
  // checkpoint — 140.4e6 parameters over the 6 layers, 280.8 MB. `nk` stays
  // false-by-name here: the shape below is the raw torch-Linear [out, in] the
  // checkpoint ships, consumed via vt::MatmulBT exactly as the host arm does.
  OwnedTensor q_proj;  // [num_attention_heads*head_dim, hidden_size]
  OwnedTensor k_proj;  // [num_key_value_heads*head_dim, hidden_size]
  OwnedTensor v_proj;  // [num_key_value_heads*head_dim, hidden_size]
  OwnedTensor o_proj;  // [hidden_size, num_attention_heads*head_dim]
  // The fp8 KV-cache scales the checkpoint ships as `k_proj.k_scale` /
  // `v_proj.v_scale` (`quantization_config.kv_cache_scheme`, num_bits 8, type
  // float). MATERIALIZED but UNUSED on this path: the host reference forward
  // holds K and V in the model dtype for the whole prompt and pages nothing, so
  // there is no fp8 KV store to scale. W6's paged device path is where they
  // become live. They are loaded rather than skipped so the tensor accounting
  // is honest — 12 tensors dropped on the floor is 12 tensors nobody notices.
  float k_scale = 1.0F;
  float v_scale = 1.0F;
  bool has_kv_scales = false;
};

// One NON-GATED expert: `ckpt_names=("up_proj","down_proj","")`
// (nemotron_h.py:220). There is no gate_proj tensor anywhere in the checkpoint.
struct NemotronHExpertWeights {
  NemotronHOwned up_proj;    // [moe_intermediate_size, hidden_size]
  NemotronHOwned down_proj;  // [hidden_size, moe_intermediate_size]
};

struct NemotronHMoeWeights {
  // The router. Its OUTPUT DTYPE IS F32 AND THAT IS MIRRORED, NOT INHERITED:
  // `GateLinear(..., out_dtype=torch.float32, force_fp32_compute=True)`
  // (nemotron_h.py:150-156). This is the annotated f32 escape AGENTS.md allows,
  // and it is upstream's own polarity rather than a local precision choice.
  NemotronHOwned gate;  // [n_routed_experts, hidden_size]
  // `self.gate.e_score_correction_bias` (nemotron_h.py:158-160), f32 upstream.
  NemotronHOwned e_score_correction_bias;  // [n_routed_experts] f32
  std::vector<NemotronHExpertWeights> experts;
  // The shared expert is the SAME non-gated shape at
  // moe_shared_expert_intermediate_size * n_shared_experts (nemotron_h.py:176-190).
  NemotronHExpertWeights shared;
  bool has_shared = false;

  // A2-Q2a (#810): this layer's device-resident Marlin arena, built ONCE on
  // first device-MoE use and owned BY THE WEIGHTS (issue #237's `ResidentSlot`,
  // qwen3_5_weights.h:183). Opaque here on purpose — the arena type is a CUDA
  // implementation detail of nemotron_h_device.cpp, exactly as `MoeBlockWeights`
  // keeps qwen3_5.cpp's out of its own header.
  //
  // KEYED ON THE SLOT, NEVER ON AN ADDRESS. `dense_nvfp4_gemm.h:379` caches its
  // Marlin repack in a `static unordered_map<const Nvfp4Weight*, ...>`, and an
  // address is only a valid identity while the object lives: across two engine
  // builds in one process a hit returns the PREVIOUS engine's repacked buffer —
  // plausible, wrong values (issue #984, whose fix is not this row's). A2-Q2a
  // cannot inherit that defect, because it never calls EITHER function named
  // `MarlinDenseResidentFor` — not even for the shared expert, which runs as an
  // E=1 slice of this same arena (the documented dense route,
  // dense_nvfp4_gemm.h:38-43).
  //
  // PROVE IT BY ABSENCE. Grep for the accessor's name immediately followed by an
  // open parenthesis — the CALL form — restricted to `src include`; it hits
  // `dense_nvfp4_gemm.h` and `qwen3_5.cpp` ONLY, and no `nemotron_h*` file.
  //
  // Two ways that grep lies if you take a shortcut, both of which bit this
  // comment before it settled. The BARE NAME matches prose like this paragraph,
  // so a reviewer ends up eyeballing which hits are comments. And spelling the
  // call form out literally HERE would make this very line a hit — which is why
  // the command is described rather than quoted. `scripts/check-fp4-resident-
  // consistency.py` is the checker that owns this class of question if it ever
  // needs to be mechanical rather than reviewed.
  //
  // Nor does any nemotron_h TU `#include` that header: the mentions here and in
  // nemotron_h_device.cpp are citations, not directives (verified with an
  // anchored `#include` regex, which returns 0).
  ResidentSlot moe_marlin;
};

// The dense `-` block. No released in-scope NemotronH checkpoint ships one, so
// this arm exists because the schedule can name it and a silent zero would be
// worse than a computed answer; it is exercised only by the unit gate.
struct NemotronHMlpWeights {
  NemotronHOwned up_proj;    // [intermediate_size, hidden_size]
  NemotronHOwned down_proj;  // [hidden_size, intermediate_size]
};

struct NemotronHLayerWeights {
  NemotronHBlock block = NemotronHBlock::kMamba;
  // The layer's SINGLE norm (`self.norm`, one per decoder layer). A2-R:
  // OwnedTensor — every one of the 52 is consumed by the device residual
  // stream, whatever kind of mixer the layer carries.
  OwnedTensor norm;  // [hidden_size]
  NemotronHMambaWeights mamba;
  NemotronHAttentionWeights attn;
  NemotronHMoeWeights moe;
  NemotronHMlpWeights mlp;
};

struct NemotronHHostWeights {
  // The ONE model dtype every activation and every weight above inherits
  // (AGENTS.md: "vLLM resolves ONE model dtype and every layer inherits it").
  // bf16 is the released checkpoint's; the gate also sweeps f32, because a bf16
  // store absorbs reduction-order defects.
  vt::DType act_dtype = vt::DType::kBF16;
  // A2-R: OwnedTensor. The embedding table is the single largest DENSE tensor in
  // the checkpoint (131072 x 2688 bf16 = 704.6 MB) and the device stream reads
  // it on every step, so it is the one that most needs uploading once.
  OwnedTensor embeddings;  // [vocab_size, hidden_size]
  std::vector<NemotronHLayerWeights> layers;
  OwnedTensor norm_f;  // [hidden_size]
  // STAYS NemotronHOwned: NVFP4 W4A16 g16 on the released checkpoint, so the
  // final projection runs on the HOST in A2-R and the device stream hands its
  // gathered rows back before it. This is the reason A2-R's token gate is
  // meaningful — both arms end in the identical host projection, so any token
  // difference is attributable to the 6 device attention blocks.
  NemotronHOwned lm_head;  // [vocab_size, hidden_size]
  // ── A2-Q2b (#810): the DEVICE lm_head's Marlin resident ────────────────────
  //
  // The repacked NVFP4 operands for `lm_head`, built once on first device
  // forward and reused on every step. It lives HERE, on the weights, for the
  // reason issue #237 added `ResidentSlot` at all: state keyed on the ADDRESS
  // of a weight outlives the engine that built it, and a second engine in the
  // same process can inherit device pointers the first one freed. That is
  // issue #984, which the shared `dense_nvfp4_gemm.h` cache still has; A2-Q2a
  // declined to inherit it for the MoE arena (`moe_marlin` above) and this
  // slot is the same decision for `lm_head`, taken explicitly per that spec's
  // §4.3 rather than by default.
  //
  // Opaque `shared_ptr<void>`: the resident type is a CUDA-path detail of
  // nemotron_h_device.cpp, and holding it here would drag Marlin's headers
  // into every consumer of this one.
  ResidentSlot lm_head_marlin;
  // False until a loader materializes the enumerated tensors. The forward
  // refuses by name on false rather than computing on zeros.
  bool materialized = false;
};

// ─── per-block entry points (each independently gateable) ───────────────────
//
// Every one takes the ALREADY-NORMED hidden `[T, hidden_size]` in `act_dtype`
// and returns the mixer output `[T, hidden_size]` in the same dtype, which is
// exactly the contract of an upstream `layer.mixer(hidden_states)` call. They
// are public so the gate can compare a BLOCK's activations against an
// independent reference — a mechanism can be missing while the argmax is
// unchanged, so a tokens-only comparison cannot see a dropped one
// (porting-a-model.md §3).

// The Mamba2 mixer's two recurrent states, carried ACROSS calls so a sequence
// can be fed in more than one leg (upstream's chunked prefill: leg 2 reads leg
// 1's `final_states` as its `initial_states`, ssd_combined.py:79,:194).
//
// The two dtypes are resolved SEPARATELY and that is the whole point:
//   * `conv` is f32 because `vt::CausalConv1dFwd` validates the conv state as
//     f32 (ops.cpp CheckConvCommon; bf16 only where the backend advertises
//     SupportsCompressedConvState). The persistent page `MakeNemotronHKVCache`
//     declares is the cache dtype and W6 owns reconciling the two.
//   * `ssm` carries `NemotronHSsmCacheDType(params, act_dtype)` — "float32" on
//     this checkpoint whatever the model dtype. A shared helper keyed on
//     Qwen3.5's `mamba_ssm_dtype` spelling returns the CONVOLUTION dtype here
//     and silently halves the recurrent state; nemotron_h.h records why.
//
// With FRESH state and a single leg the ssm dtype is UNOBSERVABLE in the output
// — the scan computes in f32 and only STORES `final_states` at this dtype. It
// becomes observable exactly when a state is carried in, which is what the
// two-leg gate exercises and what the W6 paged decode does on every step.
struct NemotronHMambaState {
  std::vector<float> conv;  // [conv_dim, conv_kernel-1] f32
  NemotronHOwned ssm;       // [mamba_num_heads, mamba_head_dim, ssm_state_size]
  // False on the first leg (both states read as zero); set by the mixer on
  // return so the next leg carries them in.
  bool has_initial = false;
};

// Mamba2 mixer over ONE sequence of T tokens. `state` nullptr means FRESH state
// discarded on return (the single-leg case); non-null carries it in and out.
std::vector<float> NemotronHMamba2Mixer(const NemotronHMambaWeights& w,
                                        const NemotronHParams& params,
                                        const std::vector<float>& hidden_normed,
                                        int64_t num_tokens, vt::DType act_dtype,
                                        vt::Queue& queue,
                                        NemotronHMambaState* state = nullptr);

std::vector<float> NemotronHAttentionMixer(const NemotronHAttentionWeights& w,
                                           const NemotronHParams& params,
                                           const std::vector<float>& hidden_normed,
                                           int64_t num_tokens, vt::DType act_dtype,
                                           vt::Queue& queue);

std::vector<float> NemotronHMoeMixer(const NemotronHMoeWeights& w,
                                     const NemotronHParams& params,
                                     const std::vector<float>& hidden_normed,
                                     int64_t num_tokens, vt::DType act_dtype,
                                     vt::Queue& queue);

std::vector<float> NemotronHMlpMixer(const NemotronHMlpWeights& w,
                                     const NemotronHParams& params,
                                     const std::vector<float>& hidden_normed,
                                     int64_t num_tokens, vt::DType act_dtype,
                                     vt::Queue& queue);

// ─── the whole decoder ──────────────────────────────────────────────────────

// Optional per-layer capture. `hidden` holds the residual stream AFTER each
// layer's mixer add, `normed` the norm input each mixer saw; both [T,H] and in
// f32 for comparison. Populated only when `capture` is true. This is the
// instrument porting-a-model.md §3 asks for: per-layer activations, not tokens.
struct NemotronHTrace {
  bool capture = false;
  std::vector<std::vector<float>> normed;  // [L][T*H]
  std::vector<std::vector<float>> mixer;   // [L][T*H]
  std::vector<std::vector<float>> hidden;  // [L][T*H] (residual after the layer)
  std::vector<float> final_normed;         // [T*H]
};

// Logits `[num_requested, vocab_size]` in f32 for the requested positions (all
// positions when `logits_indices` is empty).
//
// `inputs_embeds` (optional, [T*H] f32 holding model-dtype values) replaces
// the embedding lookup: a multimodal wrapper's merged rows
// (nano_nemotron_vl.py:1462-1481). `token_ids` still sizes the step.
std::vector<float> NemotronHForward(const NemotronHHostWeights& host,
                                    const NemotronHParams& params,
                                    const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& logits_indices,
                                    vt::Queue& queue, NemotronHTrace* trace = nullptr,
                                    const std::vector<float>* inputs_embeds = nullptr);

// Greedy argmax decode over a fresh forward per step. Single sequence; the
// incremental paged/recurrent decode is W6's.
std::vector<int32_t> NemotronHGreedyDecode(const NemotronHHostWeights& host,
                                           const NemotronHParams& params,
                                           const std::vector<int32_t>& prompt,
                                           int num_new, vt::Queue& queue);

// ─── A2-R: the DEVICE arm (#810, .agents/specs/nemotron-h-abi-e2e.md) ────────
//
// WHAT RUNS WHERE, and this split is the unit's whole scope:
//   DEVICE  embeddings, all 52 layer norms + norm_f, and the 6 GQA attention
//           blocks (NemotronHAttnBlock). The residual stream is device-resident
//           for the whole forward.
//   HOST    the 23 Mamba2 blocks (their in_proj/out_proj are FP8 W8A8 and the
//           shared FP8 linear seam is not extracted yet — issue #940), the 23
//           MoE blocks and lm_head (NVFP4 W4A16 g16, 30.19e9 parameters).
//
// So 46 of 52 layers still compute on the host, and each of them costs one
// download of the normed hidden and one upload of the mixer output. That bounce
// is SCAFFOLD, not architecture: every later unit deletes one pair of it. It is
// also why this arm makes NO speed claim of any kind.
//
// NON-PAGED, SINGLE REQUEST. Nothing here consumes `attn_kv`, `gdn_state`,
// `gdn_meta`, `gdn_state_slots` or `num_reqs`, so the G-SAFE interlock in
// `ForwardNemotronHForCausalLM` (nemotron_h_registry.cpp:161-170) keeps ALL
// THREE of its clauses. A2-R does not create the capability that interlock
// guards, so it does not narrow it.

// One NemotronH GQA attention block, computed on `dev_queue`, with host-side
// input and output so a gate can drive ONE block in isolation.
//
// This is the per-block equivalence seam: feed it the host reference's
// `trace.normed[l]` and compare against that same trace's `mixer[l]`. A token
// comparison cannot see a wrongly-applied rotation on a short prompt, and it
// cannot see a too-wide dtype at all, so the gate is NUMERIC.
std::vector<float> NemotronHAttnBlockHostIO(const NemotronHAttentionWeights& w,
                                            const NemotronHParams& params,
                                            const std::vector<float>& hidden_normed,
                                            int64_t num_tokens, vt::DType act_dtype,
                                            vt::Queue& dev_queue);

// A2-Q2a (#810): ONE NemotronH MoE block on the device, with host-side input and
// output so a gate can drive a single block in isolation — the same per-block
// equivalence seam `NemotronHAttnBlockHostIO` is, and for the same reason.
//
// WHY THE GATE IS NUMERIC AND NOT TOKENS. A token comparison cannot see a
// flipped NVFP4 nibble order, an ignored `weight_scale_2`, an expert stride off
// by one in the arena, or a `routed_scaling_factor` folded into the router
// logits instead of the output — every one of those is finite, correctly shaped
// and plausible. It is worse than that here: `vt::MoeGroupedGemmNvfp4Marlin`
// validates almost nothing at the op boundary (ops.cpp:874-895 checks a/c rank
// and dtype, `size_k % 16`, and that `b_q_weight` is rank-3 — it checks NO
// extent of `b_q_weight` and NOTHING AT ALL about `b_scales`), so a transposed
// K/N or a mis-strided expert reaches the kernel silently. The per-block numeric
// comparison against `NemotronHMoeMixer` on the SAME weights is the only
// instrument that sees them.
//
// Requires `act_dtype == kBF16`: Marlin's a/c operands are bf16 by contract
// (ops.cpp:879), which is also the released checkpoint's model dtype. An f32
// caller is refused BY NAME rather than silently widened or silently rounded.
std::vector<float> NemotronHMoeBlockDeviceHostIO(const NemotronHMoeWeights& w,
                                                 const NemotronHParams& params,
                                                 const std::vector<float>& hidden_normed,
                                                 int64_t num_tokens, vt::DType act_dtype,
                                                 vt::Queue& dev_queue);

// The final output projection, on the HOST, over `num_rows` already-gathered
// and already-final-normed rows `[num_rows, hidden_size]` (f32 in, f32 logits
// `[num_rows, vocab_size]` out).
//
// It exists so that there is exactly ONE lm_head implementation and BOTH arms
// call it. `lm_head` is NVFP4 W4A16 g16 on the released checkpoint, so it stays
// on the host in A2-R — and because the host reference and the device arm end in
// the identical projection, a token difference between them is attributable to
// the device attention blocks and the device residual stream alone. A second
// copy of this projection would quietly destroy that property.
// A2-Q2b: the DEVICE twins of `NemotronHHostLmHead` below, argument for
// argument. `...Eligible` is the same predicate the production paged forward
// branches on, exposed so a gate can assert it DISCRIMINATES rather than
// assuming it; a predicate that is constantly false disables the whole arm and
// still passes every "the host arm was used" assertion.
bool NemotronHDeviceLmHeadEligible(const NemotronHHostWeights& host, vt::DType adt,
                                   vt::Queue& dev_queue);
std::vector<float> NemotronHDeviceLmHead(const NemotronHHostWeights& host,
                                         const NemotronHParams& params,
                                         const std::vector<float>& gathered_normed,
                                         int64_t num_rows, vt::Queue& dev_queue);

std::vector<float> NemotronHHostLmHead(const NemotronHHostWeights& host,
                                       const NemotronHParams& params,
                                       const std::vector<float>& gathered_normed,
                                       int64_t num_rows, vt::Queue& host_queue);

// The hybrid forward. `dev_queue` must be a non-CPU queue; `host_queue` must be
// a CPU queue and is what the 46 host-resident mixers and lm_head run on.
// Returns logits `[num_requested, vocab_size]` in f32, the same contract as
// `NemotronHForward`, so the two are directly comparable.
std::vector<float> NemotronHDeviceForward(const NemotronHHostWeights& host,
                                          const NemotronHParams& params,
                                          const std::vector<int32_t>& token_ids,
                                          const std::vector<int32_t>& logits_indices,
                                          vt::Queue& dev_queue, vt::Queue& host_queue,
                                          NemotronHTrace* trace = nullptr);

// ─── A2-P: the PAGED forward (#810, .agents/specs/nemotron-h-a2p-paged-forward.md)
//
// THE DIFFERENCE FROM EVERY FORWARD ABOVE, in one sentence: this one reads and
// writes the RUNNER'S caches instead of rebuilding them. `NemotronHForward` and
// `NemotronHDeviceForward` recompute Q/K/V over the whole sequence on every call
// (nemotron_h.cpp:657-659) and start each call from FRESH recurrent state, so a
// server past decode step 1 would produce fluent WRONG tokens. This forward
// writes each step's K/V into `input.attn_kv` at `input.attn_meta.slot_mapping`
// and reads attention back out of those pages, and it gathers the conv/SSM rows
// out of `input.gdn_state` at the step's state indices and scatters the updated
// rows back. That is what makes a multi-step decode correct, and it is what
// narrows the G-SAFE interlock at `nemotron_h_registry.cpp:161`.
//
// SINGLE REQUEST. `input.num_reqs <= 1` stays refused by that interlock until
// A2-B: nothing here reorders a batch or splits decodes from prefills across
// requests. The per-request INDEXING machinery is nonetheless real — the state
// slot comes from the metadata's state-index vector and the block table, never
// from a hardcoded 0 (spec §4.1) — because a forward that hardcodes slot 0
// passes every gate A2-P owns and then fails silently under A2-B.
//
// WHAT STILL RUNS ON THE HOST, and why it is not this unit's to move:
//   * the 23 Mamba2 blocks. Their `in_proj` is FP8 W8A8 static and the block is
//     not splittable, so the compute stays on `NemotronHMamba2Mixer` (A2-Q1 owns
//     the device arm, issue #940). A2-P carries the STATE — gather from the
//     device page, run the host mixer over it, scatter back — which is exactly
//     what the spec's §1.1 means by "the paged wiring can land against them".
//   * `lm_head`, NVFP4 W4A16 g16, refused on a non-CPU queue at
//     nemotron_h.cpp:1031-1034. A2-Q2b owns it, so this forward still returns
//     HOST logits and `scripts/runner-routing-allowlist.txt` is NARROWED rather
//     than removed (spec §3.5).
//   * a MoE block whose experts are not NVFP4, or a build with no Marlin arm.
//     A2-Q2a's device arm is taken whenever it is available.
//
// Runs on WHATEVER queue the runner hands it. On CUDA that is the device path;
// on a CPU queue every op below is registered too, which is what lets the
// multi-step gate run without a GPU. `positions` is deliberately unread:
// NemotronH has NO positional embedding of any kind
// (`kNemotronHAttentionHasNoRope`).
ForwardLogits NemotronHPagedForward(const NemotronHHostWeights& host,
                                    const NemotronHParams& params,
                                    const ModelForwardInput& input,
                                    NemotronHTrace* trace = nullptr);

}  // namespace vllm
