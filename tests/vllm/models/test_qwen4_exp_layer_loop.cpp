// Qwen4-Exp W5f — `Qwen4ExpTextModel::Forward` against the ORACLE, end to end.
//
// Row MODEL-MM-QWEN4-EXP, issues #2031 and #2336, spec
// `.agents/specs/qwen4-exp-flash-next.md`.
//
// ─── WHAT THIS SUITE CAN SEE THAT NO EARLIER ONE COULD ───────────────────────
// Eleven waves of this row gated ONE thing each: `RunGdnBlockPaged`,
// `RunQwen4ExpQsaBlockPaged`, `RunQwen4ExpMoeBlock`, `RunQwen4ExpPleBlock`, the
// two hyper-connection ops. Every one of them is green and NONE of them can see:
//
//   * the LAYER ORDER — which index runs Gated DeltaNet and which runs Qwen
//     Sparse Attention;
//   * the hyper-connection ARITHMETIC BETWEEN the blocks — that `hyper_input` is
//     the RAW stream, that the write-back is rank-1 onto it, and that there are
//     TWO sites per layer and not one;
//   * where the PLE layer SITS — first in its own decoder layer, on the hc-wide
//     stream, and added to it;
//   * the `repeat(1, 1, hc_count)` WIDEN at the top and the `use_combine=False`
//     mixer at the bottom, with NO final RMSNorm between the mixer and the head.
//
// A composition of correct parts in the wrong order is exactly the defect class
// a per-block gate is blind to, and it produces fluent wrong text rather than a
// crash. The golden below is an observation of `Qwen4ExpTextModel.forward`
// itself, so it sees all four.
//
// ─── ORACLE ──────────────────────────────────────────────────────────────────
// transformers **5.16.0**, this row's accepted lane pin
// (`.agents/oracles/transformers.md`), sha256
// `77fec77d87f2a0eb23b95fa04276fb5779698a7c7f523cf5061e49c118bcc459`, generated
// by `scripts/gen-qwen4-exp-forward-goldens.py`, which ASSERTS that sha256
// against the file it imported before it observes anything. vLLM registers no
// `qwen4_exp` at the pinned revision, so there is no primary oracle to mirror.
//
// ─── THE TOLERANCE, AND WHY IT IS NOT 1e-5 ───────────────────────────────────
// The oracle runs the whole tower in f32; this tree runs the model path at the
// dtype the model resolves, which is bf16, exactly as AGENTS.md "Inherit vLLM
// defaults" requires. Four layers of bf16 rounding is the residual this gate
// measures, and it is NOT a tolerance chosen to make the case pass: the WEIGHTS
// are bf16-exact by construction (the generator asserts a bf16 round trip per
// tensor), so the only rounding is in the activations. The bound is stated as a
// constant below, the MEASURED residual is printed, and every mutation in the
// spec's record separates by more than an order of magnitude above it — which is
// the property that makes the bound a gate rather than a mute switch.
#include <doctest/doctest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen4_exp.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen4_exp_forward.h"
#include "vllm/model_executor/models/qwen4_exp_gguf_weights.h"
#include "vllm/model_executor/models/qwen4_exp_weights.h"
#include "vllm/platforms/interface.h"  // CurrentPlatform — the instrument's own precondition
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/attention/backends/gdn_attn.h"
#include "vt/dtype.h"
#include "vt/ops.h"
#include "vt/tensor.h"

#include "support/qwen4_exp_gguf_fixture.h"

#include "../../support/max_abs_diff.h"

namespace {

#include "qwen4_exp_forward_goldens.inc"

using vllm::OwnedTensor;
using vllm::Qwen4ExpLayerKind;
using vllm::Qwen4ExpParams;
using vllm::Qwen4ExpWeights;
using vt::DType;

constexpr int64_t kStream = kFwdHcCount * kFwdHiddenSize;
constexpr int64_t kNgramHeads = (kFwdNgramSize - 1) * kFwdHeadsPerNgram;

// THE BOUND. See the header note: bf16 activations against an f32 oracle over
// four layers. Reported as a measured value by the case itself.
constexpr float kTol = 3.0e-2F;

vt::Queue CpuQ() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

int64_t Numel(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t s : shape) n *= s;
  return n;
}

OwnedTensor Owned(DType dt, const std::vector<int64_t>& shape, bool nk = false) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  for (size_t i = 0; i < shape.size(); ++i) t.shape[i] = shape[i];
  t.nk = nk;
  t.bytes.assign(static_cast<size_t>(Numel(shape)) * vt::SizeOf(dt), 0);
  return t;
}

// Every weight below arrives from the golden in the ORACLE's own orientation and
// in the RAW HuggingFace gamma parameterization, which is precisely what the
// loader leaves in `Qwen4ExpWeights` (#2218). Nothing is transposed or folded
// here: a fixture that pre-chewed either would be gating itself.
//
// `nk` DEFAULTS TO TRUE BECAUSE `LoadMatmul` PRODUCES TRUE, unconditionally
// (`qwen4_exp_weights.cpp`: every arm — keep-quant blocks, f16 and the bf16
// expansion — passes `nk = true`). The flag is not decoration: consumers that
// branch on it read the SAME BYTES in two orientations. `BorrowWhole` in the MoE
// adapter preserves the source's flag and hands it to `MatmulF32D`, so a fixture
// that built the shared expert's projections `nk = false` fed the seam a [K, N]
// weight where a [N, K] one was meant — the same element count, no shape error,
// and a wrong answer. That was the first defect this gate found, and it was in
// the fixture rather than in the loop, which is why the default is stated here
// once instead of at thirty call sites.
OwnedTensor Bf16(const float* src, const std::vector<int64_t>& shape,
                 bool nk = true) {
  OwnedTensor t = Owned(DType::kBF16, shape, nk);
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  const int64_t n = Numel(shape);
  for (int64_t i = 0; i < n; ++i) p[i] = vt::F32ToBF16(src[i]);
  return t;
}

OwnedTensor F32(const float* src, const std::vector<int64_t>& shape) {
  OwnedTensor t = Owned(DType::kF32, shape);
  std::memcpy(t.bytes.data(), src, t.bytes.size());
  return t;
}

Qwen4ExpParams MakeParams() {
  Qwen4ExpParams p;
  p.hidden_size = kFwdHiddenSize;
  p.num_hidden_layers = kFwdLayers;
  p.vocab_size = kFwdVocabSize;
  p.rms_norm_eps = kFwdRmsNormEps;
  p.hc_count = kFwdHcCount;
  p.hc_lowrank = kFwdHcLowrank;
  p.num_experts = kFwdNumExperts;
  p.num_experts_per_tok = kFwdNumExpertsPerTok;
  p.moe_intermediate_size = kFwdMoeIntermediateSize;
  p.shared_expert_intermediate_size = kFwdSharedExpertIntermediateSize;
  p.linear_num_key_heads = kFwdLinearNumKeyHeads;
  p.linear_num_value_heads = kFwdLinearNumValueHeads;
  p.linear_key_head_dim = kFwdLinearKeyHeadDim;
  p.linear_value_head_dim = kFwdLinearValueHeadDim;
  p.linear_conv_kernel_dim = kFwdLinearConvKernelDim;
  p.eos_token_id = kFwdEosTokenId;
  p.num_attention_heads = kFwdNumAttentionHeads;
  p.num_key_value_heads = kFwdNumKeyValueHeads;
  p.head_dim = kFwdHeadDim;
  p.rotary_dim = kFwdRotaryDim;
  p.qsa.n_heads = kFwdIndexerNHeads;
  p.qsa.kv_heads = kFwdIndexerKvHeads;
  p.qsa.head_dim = kFwdIndexerHeadDim;
  p.qsa.budget = kFwdIndexerBudget;
  p.qsa.compress_ratio = kFwdIndexerCompressRatio;
  p.ple.embed_dim = kFwdPleEmbedDim;
  p.ple.conv_kernel_size = kFwdPleConvKernelSize;
  p.ple.ngram_size = kFwdNgramSize;
  p.ple.heads_per_ngram = kFwdHeadsPerNgram;
  p.ple.ngram_vocab_size_base = kFwdNgramVocabBase;
  p.ple.make_ngram_vocab_size_divisible_by = kFwdNgramVocabDivisor;
  p.ple.seed = kFwdSeed;
  p.ple.layer_ids_zero_based = {kFwdPleLayerZeroBased};
  // THE LAYER SCHEDULE, spelled out rather than synthesized from an interval,
  // because it is the thing under test. `full_attention_interval` 4 over 4
  // layers gives 3 linear then 1 sparse, which is the released pattern at its
  // shortest, and the golden's own config declares exactly this list.
  p.layer_types = {Qwen4ExpLayerKind::kLinearAttention,
                   Qwen4ExpLayerKind::kLinearAttention,
                   Qwen4ExpLayerKind::kLinearAttention,
                   Qwen4ExpLayerKind::kQwenSparseAttention};
  return p;
}

vllm::HfConfig MakeHfConfig(const Qwen4ExpParams& p) {
  vllm::HfConfig c;
  c.model_type = "qwen4_exp_text";
  c.hidden_size = p.hidden_size;
  c.num_hidden_layers = p.num_hidden_layers;
  c.rms_norm_eps = p.rms_norm_eps;
  c.num_experts = p.num_experts;
  c.linear_num_key_heads = p.linear_num_key_heads;
  c.linear_num_value_heads = p.linear_num_value_heads;
  c.linear_key_head_dim = p.linear_key_head_dim;
  c.linear_value_head_dim = p.linear_value_head_dim;
  c.linear_conv_kernel_dim = p.linear_conv_kernel_dim;
  c.rotary_dim = p.rotary_dim;
  c.rope_theta = kFwdRopeTheta;
  c.rope_parameters.rope_theta = kFwdRopeTheta;
  // THE VALUE THE CHECKPOINT ASKED FOR, and the reason `Qwen4ExpGdnHfConfig`
  // carries it rather than deriving it: `Qwen4ExpParams` has no field for it.
  c.output_gate_type = kFwdOutputGateType;
  c.mamba_ssm_dtype = "float32";
  return c;
}

struct HcArrays {
  const float* norm;
  const float* down;
  const float* up;
  const float* inject;  // nullptr on the terminal mixer
};

void FillHc(vllm::Qwen4ExpGatedResidualWeights& g, const HcArrays& a,
            const Qwen4ExpParams& p) {
  g.hc_norm = Bf16(a.norm, {kStream});
  g.down = Bf16(a.down, {p.hc_lowrank, kStream});
  g.up = Bf16(a.up, {kStream, p.hc_lowrank});
  g.has_inject = a.inject != nullptr;
  if (g.has_inject) g.inject = Bf16(a.inject, {p.hc_count, kStream});
}

struct GdnArrays {
  const float *qkv, *z, *b, *a, *conv, *alog, *dt, *norm, *out;
};

void FillGdn(vllm::Qwen4ExpGdnWeights& g, const GdnArrays& s,
             const Qwen4ExpParams& p) {
  const int64_t H = p.hidden_size;
  const int64_t num_v = p.linear_num_value_heads;
  const int64_t key_dim = p.linear_num_key_heads * p.linear_key_head_dim;
  const int64_t value_dim = num_v * p.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  // `nk = true` — the `[N, K]` orientation both GGUF loaders produce
  // (`MakeGdnProj` under `GgufLoadPolicy::gdn_expand_nk`). Passing these as
  // `nk = false` would be read transposed with no shape error.
  g.in_proj_qkv = Bf16(s.qkv, {conv_dim, H}, /*nk=*/true);
  g.in_proj_z = Bf16(s.z, {value_dim, H}, /*nk=*/true);
  g.in_proj_b = Bf16(s.b, {num_v, H}, /*nk=*/true);
  g.in_proj_a = Bf16(s.a, {num_v, H}, /*nk=*/true);
  g.conv1d = Bf16(s.conv, {conv_dim, p.linear_conv_kernel_dim});
  g.a_log = F32(s.alog, {num_v});
  g.dt_bias = F32(s.dt, {num_v});
  // THE ONE GAMMA WITH NO FOLD ANYWHERE: `Qwen4ExpTextRMSNormGated.weight` is
  // ones-init and its forward multiplies by `weight` directly, which is why
  // ggml-org/llama.cpp#27742 excludes exactly this tensor and the loader does
  // not unshift it.
  g.norm_weight = Bf16(s.norm, {p.linear_value_head_dim});
  g.out_proj = Bf16(s.out, {H, value_dim}, /*nk=*/true);
}

// ─── THE DEFERRED V-HEAD PERMUTATION, END TO END (F4 of the fresh review) ───
//
// `FillGdn` above builds every GDN projection in HuggingFace GROUPED V-head
// order and leaves `v_head_perm_key_heads` at 0, which is what every loader but
// the `qwen4exp` GGUF one produces. With the flag at 0 `vt::VHeadPermute` is
// never called, so this suite ran the whole layer loop with the deferred
// permutation INERT: inverting the op's two index maps at every production site
// left it green.
//
// `TileGdnVHeads` puts the fixture into the shape the `qwen4exp` loader
// actually hands the block — the SAME VALUES re-indexed into the converter's
// TILED V-head order on exactly the five projections the loader now leaves
// verbatim, with the flag set to the key-head count. `conv1d`, `a_log` and
// `dt_bias` stay GROUPED, because that is the loader's rule and it is what
// confines the run-time permutation to the projection boundary.
//
// The arm is then gated against THE SAME transformers 5.16.0 golden, at the
// same bound. That is what makes it convict: a wrong map pairs value head g
// with the wrong key head, and the fixture's ratio is K = 2 against R = 3, so
// the map is neither the identity (K == 1) nor its own inverse (K == R) and
// both wrong answers are reachable.

// Destination head `t` takes source head `(t % K) * R + (t / K)` — GROUPED to
// TILED, the direction that turns a HuggingFace-order tensor into the file's
// own layout. Composed with `vt::VHeadPermute`'s forward map it is the identity.
void PermuteBf16HeadBlocks(OwnedTensor& t, int64_t outer, int64_t seg,
                           int64_t prefix, int64_t heads, int64_t width,
                           int64_t kk, int64_t rr) {
  REQUIRE(t.dtype == DType::kBF16);
  REQUIRE(prefix + heads * width == seg);
  const size_t n = t.bytes.size() / 2;
  REQUIRE(static_cast<int64_t>(n) == outer * seg);
  auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
  const std::vector<uint16_t> src(p, p + n);
  for (int64_t o = 0; o < outer; ++o) {
    const uint16_t* s = src.data() + static_cast<size_t>(o * seg);
    uint16_t* dst = p + static_cast<size_t>(o * seg);
    for (int64_t h = 0; h < heads; ++h) {
      const int64_t g = (h % kk) * rr + (h / kk);
      std::memcpy(dst + prefix + h * width, s + prefix + g * width,
                  static_cast<size_t>(width) * sizeof(uint16_t));
    }
  }
}

void TileGdnVHeads(Qwen4ExpWeights& w, const Qwen4ExpParams& p) {
  const int64_t H = p.hidden_size;
  const int64_t kk = p.linear_num_key_heads;
  const int64_t num_v = p.linear_num_value_heads;
  REQUIRE(num_v % kk == 0);
  REQUIRE(kk != 1);          // the identity map: it can convict nothing
  REQUIRE(num_v / kk != kk); // its own inverse: it cannot see a flipped map
  const int64_t rr = num_v / kk;
  const int64_t Dv = p.linear_value_head_dim;
  const int64_t key_dim = kk * p.linear_key_head_dim;
  const int64_t value_dim = num_v * Dv;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  for (auto& lw : w.layers) {
    if (!lw.is_linear_attention) continue;
    vllm::Qwen4ExpGdnWeights& g = lw.gdn;
    // `[conv_dim, H]` `nk = true`: the V-head axis is FIRST, one head is Dv
    // whole rows of H, and the leading `2 * key_dim` q/k rows are the PREFIX
    // the converter does not tile.
    PermuteBf16HeadBlocks(g.in_proj_qkv, 1, conv_dim * H, 2 * key_dim * H,
                          num_v, Dv * H, kk, rr);
    PermuteBf16HeadBlocks(g.in_proj_z, 1, value_dim * H, 0, num_v, Dv * H, kk,
                          rr);
    // `[num_v, H]`: one row per V head.
    PermuteBf16HeadBlocks(g.in_proj_b, 1, num_v * H, 0, num_v, H, kk, rr);
    PermuteBf16HeadBlocks(g.in_proj_a, 1, num_v * H, 0, num_v, H, kk, rr);
    // `out_proj` is `[H, value_dim]` `nk = true` — THE RELEASED ORIENTATION,
    // head axis LAST, which `test_gdn_v_head_permute.cpp` does not run (it
    // builds `[value_dim, H]` with `nk` unset). The permutation is on the
    // COLUMNS, so the block applies its INVERSE to the out-projection input.
    PermuteBf16HeadBlocks(g.out_proj, H, value_dim, 0, num_v, Dv, kk, rr);
    // `conv1d`, `a_log`, `dt_bias` and `norm_weight` are NOT touched: the
    // loader keeps permuting those at load, and the block relies on it.
    g.v_head_perm_key_heads = kk;
  }
}

struct MoeArrays {
  const float *router, *gate_exps, *up_exps, *down_exps, *shared_gate,
      *shared_gate_proj, *shared_up_proj, *shared_down_proj;
};

void FillMoe(vllm::Qwen4ExpMoeWeights& m, const MoeArrays& s,
             const Qwen4ExpParams& p) {
  const int64_t H = p.hidden_size;
  const int64_t E = p.num_experts;
  const int64_t I = p.moe_intermediate_size;
  const int64_t S = p.shared_expert_intermediate_size;
  // f32 BY THE LOADER'S OWN CHOICE (`qwen4_exp_weights.cpp:437-447`), and
  // `Qwen4ExpMoeBlockWeights` refuses either of them at any other dtype.
  m.router = F32(s.router, {E, H});
  m.shared_gate = F32(s.shared_gate, {H});
  m.gate_exps = Bf16(s.gate_exps, {E, I, H});
  m.up_exps = Bf16(s.up_exps, {E, I, H});
  m.down_exps = Bf16(s.down_exps, {E, H, I});
  m.shared_gate_proj = Bf16(s.shared_gate_proj, {S, H});
  m.shared_up_proj = Bf16(s.shared_up_proj, {S, H});
  m.shared_down_proj = Bf16(s.shared_down_proj, {H, S});
}

Qwen4ExpWeights MakeWeights(const Qwen4ExpParams& p) {
  Qwen4ExpWeights w;
  w.params = p;
  w.embed_tokens = Bf16(kFwdEmbedTokens, {p.vocab_size, p.hidden_size});
  w.tied_word_embeddings = true;
  w.layers.resize(static_cast<size_t>(p.num_hidden_layers));

  const HcArrays hc[kFwdLayers][2] = {
      {{kFwdL0AttnHcNorm, kFwdL0AttnHcDown, kFwdL0AttnHcUp, kFwdL0AttnHcInject},
       {kFwdL0MlpHcNorm, kFwdL0MlpHcDown, kFwdL0MlpHcUp, kFwdL0MlpHcInject}},
      {{kFwdL1AttnHcNorm, kFwdL1AttnHcDown, kFwdL1AttnHcUp, kFwdL1AttnHcInject},
       {kFwdL1MlpHcNorm, kFwdL1MlpHcDown, kFwdL1MlpHcUp, kFwdL1MlpHcInject}},
      {{kFwdL2AttnHcNorm, kFwdL2AttnHcDown, kFwdL2AttnHcUp, kFwdL2AttnHcInject},
       {kFwdL2MlpHcNorm, kFwdL2MlpHcDown, kFwdL2MlpHcUp, kFwdL2MlpHcInject}},
      {{kFwdL3AttnHcNorm, kFwdL3AttnHcDown, kFwdL3AttnHcUp, kFwdL3AttnHcInject},
       {kFwdL3MlpHcNorm, kFwdL3MlpHcDown, kFwdL3MlpHcUp, kFwdL3MlpHcInject}},
  };
  const GdnArrays gdn[3] = {
      {kFwdL0GdnInProjQkv, kFwdL0GdnInProjZ, kFwdL0GdnInProjB, kFwdL0GdnInProjA,
       kFwdL0GdnConv1d, kFwdL0GdnALog, kFwdL0GdnDtBias, kFwdL0GdnNorm,
       kFwdL0GdnOutProj},
      {kFwdL1GdnInProjQkv, kFwdL1GdnInProjZ, kFwdL1GdnInProjB, kFwdL1GdnInProjA,
       kFwdL1GdnConv1d, kFwdL1GdnALog, kFwdL1GdnDtBias, kFwdL1GdnNorm,
       kFwdL1GdnOutProj},
      {kFwdL2GdnInProjQkv, kFwdL2GdnInProjZ, kFwdL2GdnInProjB, kFwdL2GdnInProjA,
       kFwdL2GdnConv1d, kFwdL2GdnALog, kFwdL2GdnDtBias, kFwdL2GdnNorm,
       kFwdL2GdnOutProj},
  };
  const MoeArrays moe[kFwdLayers] = {
      {kFwdL0MoeRouter, kFwdL0MoeGateExps, kFwdL0MoeUpExps, kFwdL0MoeDownExps,
       kFwdL0MoeSharedGate, kFwdL0MoeSharedGateProj, kFwdL0MoeSharedUpProj,
       kFwdL0MoeSharedDownProj},
      {kFwdL1MoeRouter, kFwdL1MoeGateExps, kFwdL1MoeUpExps, kFwdL1MoeDownExps,
       kFwdL1MoeSharedGate, kFwdL1MoeSharedGateProj, kFwdL1MoeSharedUpProj,
       kFwdL1MoeSharedDownProj},
      {kFwdL2MoeRouter, kFwdL2MoeGateExps, kFwdL2MoeUpExps, kFwdL2MoeDownExps,
       kFwdL2MoeSharedGate, kFwdL2MoeSharedGateProj, kFwdL2MoeSharedUpProj,
       kFwdL2MoeSharedDownProj},
      {kFwdL3MoeRouter, kFwdL3MoeGateExps, kFwdL3MoeUpExps, kFwdL3MoeDownExps,
       kFwdL3MoeSharedGate, kFwdL3MoeSharedGateProj, kFwdL3MoeSharedUpProj,
       kFwdL3MoeSharedDownProj},
  };

  int64_t gi = 0;
  for (int64_t il = 0; il < p.num_hidden_layers; ++il) {
    vllm::Qwen4ExpLayerWeights& lw = w.layers[static_cast<size_t>(il)];
    lw.is_linear_attention =
        p.layer_types[static_cast<size_t>(il)] == Qwen4ExpLayerKind::kLinearAttention;
    FillHc(lw.attn_hc, hc[il][0], p);
    FillHc(lw.mlp_hc, hc[il][1], p);
    FillMoe(lw.moe, moe[il], p);
    if (lw.is_linear_attention) {
      FillGdn(lw.gdn, gdn[gi++], p);
    } else {
      const int64_t H = p.hidden_size;
      lw.qsa.q_proj = Bf16(kFwdL3QsaQProj, {p.num_attention_heads * p.head_dim * 2, H});
      lw.qsa.k_proj = Bf16(kFwdL3QsaKProj, {p.num_key_value_heads * p.head_dim, H});
      lw.qsa.v_proj = Bf16(kFwdL3QsaVProj, {p.num_key_value_heads * p.head_dim, H});
      lw.qsa.o_proj = Bf16(kFwdL3QsaOProj, {H, p.num_attention_heads * p.head_dim});
      lw.qsa.q_norm = Bf16(kFwdL3QsaQNorm, {p.head_dim});
      lw.qsa.k_norm = Bf16(kFwdL3QsaKNorm, {p.head_dim});
      lw.qsa.idx_q_proj = Bf16(kFwdL3QsaIdxQProj, {p.qsa.n_heads * p.qsa.head_dim, H});
      lw.qsa.idx_k_proj = Bf16(kFwdL3QsaIdxKProj, {p.qsa.kv_heads * p.qsa.head_dim, H});
      lw.qsa.idx_q_norm = Bf16(kFwdL3QsaIdxQNorm, {p.qsa.head_dim});
      lw.qsa.idx_k_norm = Bf16(kFwdL3QsaIdxKNorm, {p.qsa.head_dim});
    }
    if (il == kFwdPleLayerZeroBased) {
      lw.has_ple = true;
      lw.ple.key_proj = Bf16(kFwdL1PleKeyProj, {kStream, p.ple.embed_dim});
      lw.ple.value_proj = Bf16(kFwdL1PleValueProj, {p.hidden_size, p.ple.embed_dim});
      lw.ple.norm_key = Bf16(kFwdL1PleNormKey, {kStream});
      lw.ple.norm_query = Bf16(kFwdL1PleNormQuery, {kStream});
      lw.ple.norm_conv = Bf16(kFwdL1PleNormConv, {kStream});
      lw.ple.conv1d = Bf16(kFwdL1PleConv1d, {kStream, p.ple.conv_kernel_size});
      w.ngram_table =
          Bf16(kFwdL1PleNgramTable, {kFwdNgramPaddedVocab, kFwdNgramHeadDim});
    }
  }
  FillHc(w.mixer, {kFwdMixerNorm, kFwdMixerDown, kFwdMixerUp, nullptr}, p);
  return w;
}

// The address of a weight's bytes as a NUMBER, so a red prints the two
// addresses instead of doctest's bool-stringified `1 == 1` for a `uint8_t*`.
uintptr_t Addr(const OwnedTensor& t) {
  return reinterpret_cast<uintptr_t>(t.bytes.data());
}

std::vector<float> Download(vllm::dense_attn::Dev d, const vt::Tensor& t) {
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
  std::vector<float> out(static_cast<size_t>(n));
  if (t.dtype == DType::kBF16) {
    std::vector<uint16_t> raw(static_cast<size_t>(n));
    d.b.Copy(d.q, raw.data(), t.data, raw.size() * sizeof(uint16_t));
    for (int64_t i = 0; i < n; ++i)
      out[static_cast<size_t>(i)] = vt::BF16ToF32(raw[static_cast<size_t>(i)]);
  } else {
    REQUIRE(t.dtype == DType::kF32);
    d.b.Copy(d.q, out.data(), t.data, out.size() * sizeof(float));
  }
  return out;
}


}  // namespace

TEST_CASE(
    "qwen4_exp layer loop: Qwen4ExpTextModel::Forward matches the transformers "
    "5.16.0 oracle end to end") {
  // ─── THE FIXTURE'S OWN CONDITIONING, ASSERTED BEFORE ANYTHING RUNS ───────
  // A top-k router is a DISCRETE SELECTION and its error is BIMODAL: either the
  // two sides pick the same experts and the residual is bf16-sized, or they pick
  // different ones and it is O(1). No tolerance can straddle that. This suite is
  // gateable only because the fixture's worst router margin is far above the
  // hidden-state residual the bf16 model path introduces, and the generator
  // measures that margin on the same run that produced the golden.
  //
  // THE FLOOR IS NOT DECORATION. The first draft of this fixture had a margin of
  // 0.0164 against a residual of 0.0152, layer 0 flipped an expert, and the
  // end-to-end residual read 0.466 — a fixture artefact that reads exactly like a
  // broken layer loop. Regenerating the golden onto another draw can put it back
  // there, and this REQUIRE is what makes that loud instead of silent.
  REQUIRE_MESSAGE(kFwdMinRouterMargin > 0.25,
                  "the fixture's worst router margin is "
                      << kFwdMinRouterMargin
                      << ", which is too close to the bf16 hidden-state residual "
                         "for a top-k selection to be stable; regenerate with "
                         "scripts/gen-qwen4-exp-forward-goldens.py and pick a "
                         "better-conditioned SALT / ROUTER_SCALE rather than "
                         "widening the tolerance");

  const Qwen4ExpParams p = MakeParams();
  const vllm::HfConfig config = MakeHfConfig(p);
  Qwen4ExpWeights w = MakeWeights(p);

  // ─── TWO ARMS, ONE GOLDEN ────────────────────────────────────────────────
  // The second is the `qwen4exp` GGUF loader's own shape: the five GDN
  // projections in the converter's TILED V-head order with
  // `v_head_perm_key_heads` set, which is the ONLY configuration in which
  // `vt::VHeadPermute` runs at all. Without it this whole suite is blind to
  // V-head order — inverting the op's index maps at every production site left
  // it green. See `TileGdnVHeads`.
  int64_t want_perm_key_heads = 0;
  SUBCASE("GROUPED projections, v_head_perm_key_heads == 0 (every other loader)") {
    want_perm_key_heads = 0;
  }
  SUBCASE("TILED projections with the deferred V-head permutation ACTIVE") {
    TileGdnVHeads(w, p);
    want_perm_key_heads = p.linear_num_key_heads;
  }
  for (auto& lw : w.layers) {
    if (!lw.is_linear_attention) continue;
    REQUIRE(lw.gdn.v_head_perm_key_heads == want_perm_key_heads);
    // And the flag reaches the block through the SAME carrier production uses.
    REQUIRE(vllm::Qwen4ExpGdnBlockWeights(lw.gdn, p).v_head_perm_key_heads ==
            want_perm_key_heads);
  }

  // THE GDN OUTPUT GATE IS SIGMOID HERE AND THE SHARED READER DEFAULTS TO SILU.
  // Asserted rather than assumed, because it is the one field `Qwen4ExpParams`
  // cannot carry and the whole reason `Qwen4ExpGdnHfConfig` takes a source
  // config: the two activations differ on 36 of 48 layers of the released model
  // and no shape check can tell them apart (#489).
  REQUIRE(std::string(kFwdOutputGateType) == "sigmoid");
  REQUIRE(vllm::Qwen4ExpGdnHfConfig(p, config).output_gate_type == "sigmoid");

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  const int64_t T = kFwdSeqLen;
  std::vector<int32_t> token_ids(static_cast<size_t>(T));
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    token_ids[static_cast<size_t>(t)] = static_cast<int32_t>(kFwdInputIds[t]);
    positions[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  // ── the step metadata, exactly as the runner builds it for a prefill ──
  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  // ── the caches ──
  const int64_t key_dim = p.linear_num_key_heads * p.linear_key_head_dim;
  const int64_t value_dim = p.linear_num_value_heads * p.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = p.linear_conv_kernel_dim - 1;
  const int64_t ssm_row =
      p.linear_num_value_heads * p.linear_value_head_dim * p.linear_key_head_dim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  vllm::Qwen4ExpForwardCaches caches;
  caches.gdn.resize(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(d, DType::kF32,
                       std::vector<int64_t>{1, p.linear_num_value_heads,
                                            p.linear_value_head_dim,
                                            p.linear_key_head_dim},
                       ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    caches.gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    caches.gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(T * p.num_key_value_heads *
                                               p.head_dim * 2),
                           0);
  std::vector<uint16_t> index_key(
      static_cast<size_t>(T * p.qsa.head_dim), 0);
  std::vector<int32_t> bt{0};
  std::vector<int64_t> slots(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) slots[static_cast<size_t>(t)] = t;
  // RANK 4, NOT 5 (#2435). The paged K/V page is logically
  // `[2, num_blocks, block_size, num_kv_heads, head_size]`, and `vt::Tensor`
  // holds four dimensions (`vt::kMaxRank`). Folding the leading K/V axis into
  // the block axis keeps the same bytes, the same element count and the same
  // `data` pointer -- which is all this fixture takes from the buffer, because
  // `PagedKvCache` carries the five extents as separate scalars. The rank-5
  // spelling wrote past `Tensor::shape` and `Tensor::stride` and set
  // `Tensor::repacked` as a side effect; `dense_attn::MakeTensor` now refuses it.
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, T, p.num_key_value_heads, p.head_dim},
                              kv.data());
  // W5i: the indexer side cache is PAGED, `[num_pages, block_size, D]`. One page
  // of `T` rows here, matching the single-page K/V beside it.
  vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {1, T, p.qsa.head_dim},
                               index_key.data());
  vllm::dense_attn::DBuf bt_b(d, DType::kI32, {1, 1}, bt.data());
  vllm::dense_attn::DBuf slot_b(d, DType::kI64, {T}, slots.data());
  caches.qsa.resize(1);
  caches.qsa[0].kv.data = kv_b.t().data;
  caches.qsa[0].kv.dtype = DType::kBF16;
  caches.qsa[0].kv.num_blocks = 1;
  caches.qsa[0].kv.block_size = T;
  caches.qsa[0].kv.num_kv_heads = p.num_key_value_heads;
  caches.qsa[0].kv.head_size = p.head_dim;
  caches.qsa[0].block_table = bt_b.t();
  caches.qsa[0].slot_mapping = slot_b.t();
  caches.qsa[0].index_key = idx_b.t();
  caches.qsa[0].index_block_table = bt_b.t();

  const int64_t state_len = p.ple.short_conv_state_len();
  std::vector<float> ple_conv(static_cast<size_t>(kStream * state_len), 0.0F);
  std::vector<int64_t> ple_tok(static_cast<size_t>(p.ple.ngram_size - 1), 0);
  // THE RING CARRIES THE STREAM DTYPE (W5k, #2031). It was f32 against a bf16
  // loop, which upstream cannot produce: each cache slot is allocated with the
  // dtype of the tensor that first reaches it (`cache_utils.py:1019-1023`), and
  // the tensor reaching this one is `hidden_states`
  // (`modeling_qwen4_exp.py:1157-1159`). THE GOLDEN DOES NOT MOVE, and that is a
  // property rather than luck: on a `past_len == 0` call the block zeroes the ring
  // on entry and the conv only WRITES it at the end, so nothing reads a ring value
  // and its dtype cannot reach a single output. It would have reached the answer at
  // the first continuing step, which is what W5k makes possible.
  vllm::dense_attn::DBuf ple_conv_b(d, vllm::kQwen4ExpStreamDType,
                                    {1, kStream, state_len}, ple_conv.data());
  caches.ple.resize(1);
  caches.ple[0].conv_state = ple_conv_b.t();
  caches.ple[0].tokens = vllm::dense_attn::MakeTensor(
      ple_tok.data(), DType::kI64, vt::Device{vt::DeviceType::kCPU, 0},
      {1, p.ple.ngram_size - 1});
  caches.ple[0].state_row = 0;

  // ── the n-gram LAYOUT, checked against the oracle's own buffers ──
  // The C++ side DERIVES the prime chain and the offsets; the golden carries
  // what upstream derived. Comparing them is what stops this fixture from
  // agreeing with itself about a 320-million-row table's addressing.
  {
    const vllm::qwen4_exp::NGramTableLayout layout = vllm::Qwen4ExpPleLayout(p, 0);
    REQUIRE(static_cast<int64_t>(layout.head_vocab_sizes.size()) == kNgramHeads);
    for (int64_t i = 0; i < kNgramHeads; ++i) {
      CHECK(layout.head_vocab_sizes[static_cast<size_t>(i)] ==
            kFwdNgramHeadVocabSizes[i]);
      CHECK(layout.head_offsets[static_cast<size_t>(i)] == kFwdNgramHeadOffsets[i]);
    }
    CHECK(layout.padded_vocab_size == kFwdNgramPaddedVocab);
    REQUIRE(layout.layer_multipliers.size() == 3);
    for (int i = 0; i < 3; ++i)
      CHECK(layout.layer_multipliers[static_cast<size_t>(i)] ==
            kFwdNgramLayerMultipliers[i]);
  }

  const vllm::Qwen4ExpTextModelOutput out = vllm::Qwen4ExpTextModelForward(
      d, w, config, token_ids, positions, am, gm, caches, /*past_len=*/0);

  REQUIRE(out.storage != nullptr);
  REQUIRE(out.tensor.rank == 2);
  REQUIRE(out.tensor.shape[0] == T);
  REQUIRE(out.tensor.shape[1] == p.hidden_size);

  const std::vector<float> got = Download(d, out.tensor);
  REQUIRE(got.size() == static_cast<size_t>(T * p.hidden_size));

  // FINITENESS BEFORE ANY TOLERANCE (#2272, #449). `vllm_test::MaxAbsDiff`
  // returns +infinity on a non-finite operand rather than the NaN-blind zero the
  // two obvious `std::max` folds return, and it raises on the offending index —
  // but an explicit scan first says WHICH failure a red is.
  for (size_t i = 0; i < got.size(); ++i) {
    REQUIRE_MESSAGE(std::isfinite(got[i]),
                    "the layer loop emitted a non-finite value at index " << i);
  }

  const double worst = vllm_test::MaxAbsDiff(got, kFwdExpectedHidden,
                                             got.size());
  MESSAGE("layer loop vs transformers 5.16.0: max|diff| = " << worst
          << " against a bound of " << kTol
          << " (v_head_perm_key_heads = " << want_perm_key_heads << ")");
  CHECK(worst < kTol);
}

// ─────────────────────────────────────────────────────────────────────────────
// REACHABILITY (AGENTS.md "Nothing lands dead", `.agents/reachability.md`).
//
// The case above drives `Qwen4ExpTextModelForward` DIRECTLY, which proves the
// arithmetic and proves nothing about whether anything reaches it. This one
// enters through a PRODUCTION entry point — `ModelRegistry::Forward` — on a
// model produced by `ModelRegistry::Load` from a real `qwen4exp` GGUF, and it is
// the first time this architecture has run a forward from one.
//
// WHAT IT CAN AND CANNOT BE. It cannot be a token gate: the synthetic fixture's
// weights are a deterministic ramp, not a checkpoint, so there is no reference
// token stream. It cannot run through `GPUModelRunner` either — that path sets
// `multi_kv`, which `ModelRegistry::Forward` refuses by name for every model, a
// refusal #2353 established must NOT be lifted yet. What it IS: proof that the
// registered hook is entered, opens its handle, assembles the caches, runs all
// four block seams over 4 layers, returns finite logits of the right shape, and
// that `vt::GreedyArgmax` turns them into a token id.
// The mutation that reds it is deleting the `Qwen4ExpTextModelForward` call
// site, which is exactly the mutation `.agents/reachability.md` step 5 asks for
// and which the three preceding waves could only record as VACUOUS.
TEST_CASE(
    "qwen4_exp layer loop: ModelRegistry::Forward reaches it on a loaded "
    "qwen4exp GGUF") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    // Every id in range and the LAST one EOS, so the n-gram hash sees a segment
    // boundary rather than a uniform ramp.
    ids[static_cast<size_t>(t)] = static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  // The two POSITIONAL channels, sized from the fixture's own geometry: three
  // Gated DeltaNet layers and one Qwen Sparse Attention layer.
  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(
      static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, T, kKvHeads, kHeadDim}, kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids,     pos,      am, gm, attn_kv,
                             gdn,     config,   q,  logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  // ─── WHAT THIS CASE ASSERTS: A COMPLETED FORWARD AND A SAMPLED TOKEN ──────
  //
  // W5f could only get to layer 1's PLE block, because W5e-2's
  // `Qwen4ExpPleLayout` derived the n-gram head vocabulary from a DEFAULTED
  // `ngram_vocab_size_base` and refused when the file's stated sizes disagreed.
  // A `qwen4exp` GGUF never states that base — llama.cpp #27742's converter
  // writes the RESOLVED arrays instead — so the comparison was against a
  // default, and it held for exactly one file in the world: the released
  // checkpoint, whose base really is 20,000,000. W5g makes the stated set the
  // authority for the layout, which is what `NgramTableRows` and
  // `Qwen4ExpPleParams`'s own field comment already said it was, and narrows
  // the cross-check to sources that state the base. See
  // `.agents/specs/qwen4-exp-flash-next.md` `## The PLE layout's two sources`.
  //
  // SO THE FORWARD NOW COMPLETES. `ModelRegistry::Forward` enters the hook, the
  // hook opens its handle, assembles the caches, runs all four block seams over
  // 4 layers, gathers the requested row and multiplies it by the lm_head, and
  // returns `[1, vocab]` f32 logits. `vt::GreedyArgmax` — the on-device sampler
  // AGENTS.md routes decode through — then turns them into a token id. That is
  // the FIRST token this architecture has ever produced in this tree.
  //
  // WHAT IT IS NOT. It is not a token GATE. The fixture's weights are a
  // deterministic ramp, not a checkpoint, so no reference token stream exists
  // and the id below is not compared against one. The ARITHMETIC of the whole
  // tower is gated by the golden case above against transformers 5.16.0.
  //
  // FINITENESS IS ASSERTED BEFORE ANYTHING ELSE, and it is not decoration.
  // `MaxAbsDiff`-style folds over `std::max` return the non-NaN operand, so an
  // all-NaN logit row reads as a perfect match to any tolerance and an argmax
  // over it still returns an index in range. Every element is checked.
  vllm::ForwardLogits fl;
  REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  CHECK(fl.rows == 1);
  CHECK(fl.vocab == kVocab);
  REQUIRE(fl.on_device());
  REQUIRE(fl.device_tensor.data != nullptr);
  REQUIRE(fl.device_tensor.dtype == DType::kF32);

  std::vector<float> host(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
  d.b.Copy(q, host.data(), fl.device_tensor.data, host.size() * sizeof(float));
  d.b.Synchronize(q);

  int finite = 0;
  for (float v : host) finite += std::isfinite(v) ? 1 : 0;
  REQUIRE(finite == static_cast<int>(host.size()));
  // A CONSTANT row would satisfy every check above and mean the tower
  // contributed nothing — the lm_head times a zero hidden is finite, in range,
  // and argmaxes to 0. So the row must actually VARY.
  float lo = host[0];
  float hi = host[0];
  for (float v : host) {
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  CHECK(hi > lo);

  // ─── THE TOKEN ────────────────────────────────────────────────────────────
  // `vt::GreedyArgmax` is the production greedy sampler (`vt/ops.h`), the same
  // primitive `GPUModelRunner` samples with, and it is driven here on the
  // device tensor the forward returned rather than on a host copy.
  std::vector<int64_t> tok(1, -1);
  {
    vllm::dense_attn::DBuf tok_b(d, DType::kI64, {1});
    vt::Tensor tt = tok_b.t();
    vt::GreedyArgmax(q, tt, fl.device_tensor);
    tok_b.Download(d, tok.data());
  }
  MESSAGE("qwen4_exp sampled token id: " << tok[0] << " of " << fl.vocab
                                         << " (logit range [" << lo << ", "
                                         << hi << "])");
  CHECK(tok[0] >= 0);
  CHECK(tok[0] < fl.vocab);
  // The argmax must be the row's own maximum, so the sampler is reading THESE
  // logits and not an uninitialised buffer.
  CHECK(host[static_cast<size_t>(tok[0])] == hi);
  // And the old refusals must be gone: neither the pre-W5f unconditional one
  // nor W5f's PLE layout stop can still be reachable on this input.
  CHECK(finite > 0);

  // ─── AND THE FORWARD DEPENDS ON ITS INPUT ─────────────────────────────────
  //
  // EVERY ASSERTION ABOVE IS SATISFIED BY A HOOK THAT NEVER READS `token_ids`.
  // MUT-REACH measured exactly that: with the whole four-layer tower deleted,
  // 90 of this case's assertions still held and only `CHECK(hi > lo)` went red.
  // But `hi > lo` says the row is not CONSTANT, and `lm_head` times any
  // non-constant hidden clears it — including a hidden derived from nothing but
  // the shapes. A forward that ignored the prompt entirely would still pass
  // every check above, so "a token came out" is not yet "a token came out of
  // THIS prompt".
  //
  // A SECOND PROMPT IS THE MISSING HALF, and it is the sibling row's own shape:
  // `test_glm5_next_forward.cpp` runs a second `Step` and asserts the logits
  // moved. THE CACHES ARE REBUILT RATHER THAN REUSED. The first forward WRITES
  // the paged KV and both recurrent states, so a second call on the same
  // buffers could differ because the cache was dirty rather than because the
  // tokens changed — which a `token_ids`-ignoring hook would also produce. Fresh
  // zeroed buffers leave the prompt as the only input that moved.
  std::vector<int32_t> ids2(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    // A DIFFERENT prompt, still every id in range and still EOS-terminated, so
    // the n-gram hash sees the same segment structure and a different history.
    ids2[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : (t + 1) * 4 + 1);
    REQUIRE(ids2[static_cast<size_t>(t)] < static_cast<int32_t>(kVocab));
  }
  REQUIRE(ids2 != ids);

  std::vector<std::vector<float>> ssm2(3), conv2(3);
  std::vector<vllm::dense_attn::DBuf> ssm2_b, conv2_b;
  std::vector<vllm::GdnStateCache> gdn2(3);
  ssm2_b.reserve(3);
  conv2_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm2[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv2[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm2_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm2[i].data());
    conv2_b.emplace_back(d, DType::kF32,
                         std::vector<int64_t>{1, conv_dim, conv_len},
                         conv2[i].data());
    gdn2[static_cast<size_t>(i)].ssm_state = ssm2_b.back().t();
    gdn2[static_cast<size_t>(i)].conv_state = conv2_b.back().t();
  }
  std::vector<uint16_t> kv2(
      static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv2_b(d, DType::kBF16,
                               {2, T, kKvHeads, kHeadDim}, kv2.data());
  std::vector<vllm::PagedKvCache> attn_kv2(1);
  attn_kv2[0] = attn_kv[0];
  attn_kv2[0].data = kv2_b.t().data;

  vllm::ModelForwardInput in_p2{ids2, pos,    am, gm, attn_kv2,
                                gdn2, config, q,  logits_indices};
  in_p2.num_reqs = 1;
  in_p2.gdn_state_slots = 1;

  vllm::ForwardLogits fl2;
  REQUIRE_NOTHROW(fl2 = vllm::ModelRegistry::Forward(*model, in_p2));
  REQUIRE(fl2.on_device());
  REQUIRE(fl2.device_tensor.data != nullptr);
  REQUIRE(fl2.rows == fl.rows);
  REQUIRE(fl2.vocab == fl.vocab);
  std::vector<float> host2(static_cast<size_t>(fl2.rows * fl2.vocab), 0.0F);
  d.b.Copy(q, host2.data(), fl2.device_tensor.data, host2.size() * sizeof(float));
  d.b.Synchronize(q);
  // `MaxAbsDiff` RAISES on a non-finite operand rather than folding it away, so
  // this is also the second row's finiteness check.
  const double moved = vllm_test::MaxAbsDiff(host, host2);
  MESSAGE("qwen4_exp second-prompt logit movement (row T-1 only, DIAGNOSTIC): " << moved);

  // ─── #2851: THE ROW ABOVE CANNOT CARRY THIS CLAIM, AND NEVER COULD ────────
  //
  // `logits_indices` is `{T - 1}`, so `moved` compares exactly ONE row — and
  // both prompts are deliberately EOS-terminated, so it is the row for the SAME
  // final token. Only 3 tokens of context can move it, and in this synthetic
  // fixture that is below the arithmetic noise floor. `CHECK(moved > 0.0)` was
  // therefore satisfied by rounding, not by prompt dependence, and it is a mute
  // switch in both directions. Measured on `origin/main`'s own arithmetic,
  // varying only `ids2`'s formula:
  //
  //   (t + 1) * 4 + 1  (committed)  0.0546875     kVocab - 2 - t        0
  //   t + 2                         0.0546875     (t + 1) * 3           0.0546875
  //   (t * 5 + 2) % (kVocab - 1)    0.0546875     kVocab - 4 + t        0
  //
  // Two of six valid prompts make `origin/main` itself read exactly 0, so the
  // committed prompt is a winning ticket rather than a demonstration; and the
  // SAME 0.0546875 (= 7/128) for four different prompts is a fixed rounding
  // artifact, because a quantity that does not move when the prompt moves is
  // not measuring the prompt. `max|logit|` is 95090.7, so it is ~5.7e-7 of the
  // compared magnitude, against sibling movement gates in this same file that
  // read 31.8438 (:1349) and 129102 (:2158) on rows whose token differs.
  //
  // So the claim is re-asserted where it is unambiguous: over ALL T rows, of
  // which rows 0..T-2 carry a DIFFERENT TOKEN in the two prompts. That is the
  // "a token came out of THIS prompt" the comment above asks for, and it does
  // not depend on which arithmetic the GDN prefill takes.
  auto run_fresh_all_rows = [&](const std::vector<int32_t>& prompt) {
    std::vector<std::vector<float>> fs(3), fc(3);
    std::vector<vllm::dense_attn::DBuf> fs_b, fc_b;
    std::vector<vllm::GdnStateCache> fg(3);
    fs_b.reserve(3);
    fc_b.reserve(3);
    for (int i = 0; i < 3; ++i) {
      fs[i].assign(static_cast<size_t>(ssm_row), 0.0F);
      fc[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
      fs_b.emplace_back(
          d, DType::kF32,
          std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
          fs[i].data());
      fc_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len}, fc[i].data());
      fg[static_cast<size_t>(i)].ssm_state = fs_b.back().t();
      fg[static_cast<size_t>(i)].conv_state = fc_b.back().t();
    }
    std::vector<uint16_t> fkv(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
    vllm::dense_attn::DBuf fkv_b(d, DType::kBF16, {2, T, kKvHeads, kHeadDim},
                                 fkv.data());
    std::vector<vllm::PagedKvCache> fattn(1);
    fattn[0] = attn_kv[0];
    fattn[0].data = fkv_b.t().data;
    std::vector<int32_t> all_rows(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) all_rows[static_cast<size_t>(t)] = static_cast<int32_t>(t);
    vllm::ModelForwardInput fin{prompt, pos,    am, gm, fattn,
                                fg,     config, q,  all_rows};
    fin.num_reqs = 1;
    fin.gdn_state_slots = 1;
    vllm::ForwardLogits ffl;
    REQUIRE_NOTHROW(ffl = vllm::ModelRegistry::Forward(*model, fin));
    REQUIRE(ffl.on_device());
    REQUIRE(ffl.rows == T);
    std::vector<float> h(static_cast<size_t>(ffl.rows * ffl.vocab), 0.0F);
    d.b.Copy(q, h.data(), ffl.device_tensor.data, h.size() * sizeof(float));
    d.b.Synchronize(q);
    return h;
  };
  const std::vector<float> all1 = run_fresh_all_rows(ids);
  const std::vector<float> all2 = run_fresh_all_rows(ids2);
  REQUIRE(all1.size() == all2.size());
  const double moved_all = vllm_test::MaxAbsDiff(all1, all2);
  // Rerunning the SAME prompt must be bit-identical, or `moved_all` is measuring
  // run-to-run noise rather than the prompt. This is the control that stops the
  // floor below from being a second mute switch.
  const std::vector<float> all1_again = run_fresh_all_rows(ids);
  const double self = vllm_test::MaxAbsDiff(all1, all1_again);
  // MAX-ABS IS SATURATED IN THIS FIXTURE AND DOES NOT TRACK THE PROMPT.
  // `moved_all` reads the identical 31.8438 for every one of the six `ids2`
  // formulas swept in #2851 — which is the same shape of defect this case was
  // repaired for, one order up: a value that does not move when the prompt
  // moves is not measuring the prompt. Whole-vector statistics DO move
  // (L2 107.857 / 108.49 / 108.572 / 76.7717 and 16-or-32 of 64 elements
  // differing across those six), so the gate below asserts on one of those as
  // well and the MESSAGE reports all three. The residue is the fixture's
  // prompt-invariant common mode (`max|logit|` 95090.7), which is #2851's
  // remaining half and belongs to the qwen4_exp row.
  double l2_diff = 0.0;
  std::size_t n_diff = 0;
  for (std::size_t i = 0; i < all1.size(); ++i) {
    const double d = static_cast<double>(all1[i]) - all2[i];
    if (all1[i] != all2[i]) ++n_diff;
    l2_diff += d * d;
  }
  l2_diff = std::sqrt(l2_diff);
  MESSAGE("qwen4_exp second-prompt logit movement over ALL " << T
          << " rows: max|d|=" << moved_all << " (SATURATED — see comment) L2="
          << l2_diff << " elements differing=" << n_diff << "/" << all1.size()
          << " (same-prompt rerun control: " << self << ")");
  // THE DETERMINISM CONTROL IS WHAT MAKES `> 0.0` A GATE RATHER THAN A MUTE
  // SWITCH, and it is the half the one-row form never had. With the same prompt
  // reproducing bit-for-bit, a nonzero difference between two prompts can only
  // have come from the prompt. Without it, `> 0.0` is satisfied by any noise —
  // which is exactly how the row-T-1 form passed on a 7/128 rounding artifact.
  CHECK(self == 0.0);
  CHECK(moved_all > 0.0);
  // The statistic that actually tracks the prompt, asserted alongside the one
  // that does not, so a future regression to a saturated constant is visible
  // here rather than only in an issue.
  CHECK(l2_diff > 0.0);
  CHECK(n_diff > 0);
  // NO MAGNITUDE FLOOR IS ASSERTED, and that is deliberate rather than lazy.
  // This fixture's logits are dominated by a prompt-INVARIANT common mode
  // (`max|logit|` 95090.7 against a prompt-driven movement of 0.32 on the
  // sequential arm and 31.84 on the chunked one), so max-abs-diff has no
  // meaningful scale here and any constant would be picked to fit rather than
  // derived. Whether this fixture SHOULD have that common mode is #2851's
  // remaining half and belongs to the qwen4_exp row, not to a GDN kernel row.
  // What is gated here is the claim the comment above actually makes.

  // ─── THE TWO REFUSALS THIS HOOK ADVERTISES, GATED BY THEIR MESSAGE ────────
  //
  // A BARE `CHECK_THROWS` HERE WOULD BE A MUTE SWITCH, and it was one until this
  // repair. Every input below reaches the loop if its refusal is deleted, and
  // the loop then throws anyway at the SAME PLE layout cross-check the golden
  // reach above lands on. An unrelated exception satisfied the assertion, so
  // deleting either `VT_CHECK` in `qwen4_exp_registry.cpp` left this suite at
  // 2 cases / 76 assertions SUCCESS and the scaffold suite at 12 / 294.
  //
  // So each refusal is asserted TWO-SIDED on its own MESSAGE: the bytes that
  // identify it are present, AND the PLE message that means the input got past
  // it is absent. Deleting either `VT_CHECK` flips both halves of its pair.
  auto refusal_message = [&](const vllm::ModelForwardInput& bad,
                             const char* what) {
    std::string msg;
    try {
      (void)vllm::ModelRegistry::Forward(*model, bad);
      FAIL(what);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    return msg;
  };

  // `past_len != 0` means the QSA indexer side cache and the PLE conv ring and
  // n-gram history would have had to persist across a step, and no channel
  // carries them. The refusal must fire BEFORE the loop runs.
  vllm::v1::CommonAttentionMetadata am2 = am;
  am2.seq_lens.assign(1, static_cast<int32_t>(T + 1));
  vllm::ModelForwardInput in2{ids,     pos,      am2, gm, attn_kv,
                              gdn,     config,   q,   logits_indices};
  in2.num_reqs = 1;
  in2.gdn_state_slots = 1;
  const std::string past_len_msg = refusal_message(
      in2,
      "ModelRegistry::Forward returned on a step at past_len 1 with NO by-name "
      "cache index; the scratch-arm refusal is gone");
  INFO("the past_len refusal said: ", past_len_msg);
  CHECK(past_len_msg.find("Qwen4ExpForConditionalGeneration") !=
        std::string::npos);
  // W5k: THE REFUSAL IS NOW ABOUT THIS ARM, NOT ABOUT THE MODEL. It used to say
  // the hook served a "SINGLE-SHOT PREFILL" because the PLE states could not
  // persist at all; they persist now, in the engine's own recurrent group, and a
  // continuing step on the BY-NAME arm returns a token (see the by-name case).
  // What is refused here is the POSITIONAL arm specifically: nothing published
  // the PLE states, so the hook allocates them per call, and a per-call buffer is
  // zeroed on entry — every step would re-seed the history with EOS and decode as
  // though each token were the first.
  CHECK(past_len_msg.find("NO by-name cache index") != std::string::npos);
  // The word that named the OLD limit must be GONE, not merely joined by a new
  // one: a message that still claims a single-shot-prefill hook would send the
  // next reader to rebuild what this wave landed.
  CHECK(past_len_msg.find("SINGLE-SHOT") == std::string::npos);
  // The VALUE, not only the word: the hook reports the past_len it was handed.
  CHECK(past_len_msg.find("at past_len 1") != std::string::npos);
  // It stopped at the boundary and never entered the loop, so the PLE layout
  // cross-check the golden reach above lands on cannot be what threw.
  CHECK(past_len_msg.find("qwen4_exp ple layout") == std::string::npos);

  // `num_reqs != 1` is the OTHER refusal, and nothing drove it at all before
  // this repair. `RunQwen4ExpQsaBlockPaged` takes a block_table of i32
  // [1, max_pages], so a ragged multi-request batch has no plumbing here.
  vllm::ModelForwardInput in3{ids,     pos,      am, gm, attn_kv,
                              gdn,     config,   q,  logits_indices};
  in3.num_reqs = 2;
  in3.gdn_state_slots = 1;
  const std::string num_reqs_msg = refusal_message(
      in3,
      "ModelRegistry::Forward returned on a step carrying num_reqs 2; the "
      "one-sequence-per-call refusal is gone");
  INFO("the num_reqs refusal said: ", num_reqs_msg);
  CHECK(num_reqs_msg.find("Qwen4ExpForConditionalGeneration") !=
        std::string::npos);
  CHECK(num_reqs_msg.find("ONE sequence per call") != std::string::npos);
  // The VALUE again: the count the step actually carried.
  CHECK(num_reqs_msg.find("the step carries 2") != std::string::npos);
  CHECK(num_reqs_msg.find("qwen4_exp ple layout") == std::string::npos);
}

// ─────────────────────────────────────────────────────────────────────────────
// W5j — THE BY-NAME CACHE CHANNEL, THROUGH THE PRODUCTION ENTRY POINT.
//
// The case above enters `ModelRegistry::Forward` with `multi_kv == nullptr`,
// which is the shape no engine step has: `qwen4_exp` publishes THREE KV cache
// groups, so `GPUModelRunner` sets `multi_cache_topology_` and hands the forward
// a `MultiKvCacheIndex`. Three things stood between that step and a token, and
// this case drives all three.
//
//   1. `ModelRegistry::Forward` refused ANY non-null `multi_kv`. It now refuses
//      only a topology reaching a forward whose `ModelFactory::consumes_multi_kv`
//      is false, and this architecture sets it. A green run below IS the proof
//      that the guard let it past; every refusal asserted further down also
//      asserts the ENGINE's message is ABSENT, so a case that stopped at the
//      guard cannot read as this hook refusing.
//
//   2. `attn_kv` ARRIVES AT 2 x n_qsa, not n_qsa. The runner allocates one paged
//      buffer per (attention group x layer) and this model publishes TWO
//      attention groups over the same QSA layers — group 0's K/V and group 2's
//      indexer side cache. The hook's old `attn_kv.size() == n_qsa` assertion
//      would have turned the lifted guard into a DIFFERENT refusal rather than a
//      token, which is why the channel is resolved by NAME here.
//
//   3. W5i's production path used a per-call scratch behind an IDENTITY block
//      table, so the permutation it gates was never exercised by an allocator.
//      Group 2's own buffer and its own gathered table reach the block now.
//
// ─── HOW THE PAGING IS GATED, AND WHY NOT BY VALUE ───────────────────────────
// W5i measured the trap: 9 of 23 rows landed in the wrong physical page and BOTH
// the paged-vs-contiguous value diff (0 of 1472 words) and the oracle bound
// stayed green, because the store and the read share ONE translation and a
// consistently wrong one returns the right answer. So the gate here is
// STRUCTURAL and comparative:
//
//   * the group-2 buffer is POISONED with a sentinel before the step, and after
//     it exactly the rows the group's OWN table names for logical [0, T) differ
//     from the sentinel — every other row is byte-for-byte the sentinel still;
//   * the SAME step is run twice with two DIFFERENT group-2 tables. The logits
//     must be BIT-IDENTICAL (paging is transparent to the arithmetic) and the
//     written row sets must DIFFER (paging is not transparent to the buffer).
//     A hook that ignored the table would pass the first half and fail this one;
//   * the written set is asserted to be neither the IDENTITY set nor group 0's
//     translation, which are the two wrong tables physically present in the step.
//
// The tables are NON-IDENTITY and FIXED-POINT-FREE over the pages actually used,
// and the final page is PARTIAL (T = 6 over a page size of 4), so a translation
// that dropped the row-within-page term or clamped the last page is separated.
TEST_CASE(
    "qwen4_exp: ModelRegistry::Forward serves the BY-NAME multi-cache channel") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  // The fixture's topology: layers 0,1,2 linear_attention and layer 3
  // qwen_sparse_attention, so ONE QSA layer and THREE recurrent ones.
  constexpr int64_t kQsaLayer = 3;
  constexpr int64_t kGdnLayers = 3;
  constexpr int64_t T = 6;
  // `RunQwen4ExpQsaBlockPaged` requires the K/V page size to be a multiple of
  // `indexer_compress_ratio`, so 4 is the smallest legal page here — and it
  // leaves T = 6 spanning two pages with a PARTIAL final one.
  constexpr int64_t kPage = 4;
  static_assert(kPage % kCompressRatio == 0, "the QSA page must hold whole compress blocks");
  constexpr int64_t kCols = 2;   // pages named per sequence
  constexpr int64_t kBlocks0 = 3;  // group 0's pool
  constexpr int64_t kBlocks2 = 4;  // group 2's pool — a DIFFERENT size on purpose

  // THE THREE PERMUTATIONS, all distinct over the two pages this step uses and
  // none of them the identity or a fixed point there.
  const std::vector<int32_t> bt0{2, 0};        // group 0's map
  const std::vector<int32_t> bt2_a{1, 3};      // group 2's map, run A
  const std::vector<int32_t> bt2_b{2, 0};      // group 2's map, run B — same
                                               // pages as group 0's, in group
                                               // 2's OWN pool
  const std::vector<int32_t> bt_recurrent{0};  // the recurrent group's row

  // The physical row a logical position lands on under a given table.
  const auto rows_of = [](const std::vector<int32_t>& table, int64_t n) {
    std::vector<int64_t> r(static_cast<size_t>(n));
    for (int64_t t = 0; t < n; ++t)
      r[static_cast<size_t>(t)] =
          static_cast<int64_t>(table[static_cast<size_t>(t / kPage)]) * kPage + t % kPage;
    return r;
  };

  std::vector<int32_t> ids(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    ids[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
  for (int32_t v : ids) REQUIRE(v < static_cast<int32_t>(kVocab));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  // The bf16 sentinel the group-2 buffer is poisoned with: 0x3F80 is 1.0f.
  // A row the block wrote is a raw indexer key and is not plausibly sixteen
  // consecutive exact 1.0s, and a row it did not write is EXACTLY this.
  constexpr uint16_t kPoison = 0x3F80;
  const int64_t kIdxRows = kBlocks2 * kPage;

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  // ─── ONE BY-NAME STEP, ASSEMBLED THE WAY THE RUNNER ASSEMBLES ONE ─────────
  // Every buffer is FRESH per call: the first step writes the paged K/V, both
  // recurrent states and the indexer pages, so reusing them would let a second
  // run differ because the caches were dirty rather than because the input was.
  struct StepResult {
    std::vector<float> logits;
    std::vector<uint16_t> index_cache;
    std::vector<uint16_t> kv_cache;
    int64_t vocab = 0;
    int64_t token = -1;
  };
  // `corrupt` names ONE way the channel can be malformed; 0 is a well-formed
  // step. When `err` is non-null the call is expected to refuse and the message
  // is returned through it instead of the step's results.
  enum Corrupt { kNone = 0, kWrongName = 1, kWrongKind = 2, kNoGroupTable = 3 };
  const auto run_step = [&](const std::vector<int32_t>& idx_table,
                            const std::vector<int32_t>& prompt,
                            int corrupt = kNone, std::string* err = nullptr) {
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1;
    am.num_actual_tokens = static_cast<int>(T);
    am.block_table_num_cols = static_cast<int>(kCols);
    am.block_table_tensor = bt0;
    am.seq_lens.assign(1, static_cast<int32_t>(T));
    am.query_start_loc = {0, static_cast<int32_t>(T)};
    am.slot_mapping.resize(static_cast<size_t>(T));
    {
      const std::vector<int64_t> r0 = rows_of(bt0, T);
      for (int64_t t = 0; t < T; ++t)
        am.slot_mapping[static_cast<size_t>(t)] =
            static_cast<int32_t>(r0[static_cast<size_t>(t)]);
    }

    vllm::v1::GDNAttentionMetadata gm;
    gm.num_prefills = 1;
    gm.num_prefill_tokens = static_cast<int>(T);
    gm.num_actual_tokens = static_cast<int>(T);
    gm.has_initial_state = std::vector<uint8_t>{0};
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    gm.prefill_state_indices = std::vector<int32_t>{0};
    gm.prefill_has_initial_state = std::vector<uint8_t>{0};
    {
      const auto conv =
          vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = conv.batch_ptr;
      gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    }

    const int64_t key_dim = kNumKHeads * kLinHeadDim;
    const int64_t value_dim = kNumVHeads * kLinHeadDim;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    const int64_t conv_len = kConvKernel - 1;
    const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

    std::vector<std::vector<float>> ssm(kGdnLayers), conv(kGdnLayers);
    std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
    std::vector<vllm::GdnStateCache> gdn(kGdnLayers);
    ssm_b.reserve(kGdnLayers);
    conv_b.reserve(kGdnLayers);
    for (int i = 0; i < kGdnLayers; ++i) {
      ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
      conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
      ssm_b.emplace_back(d, DType::kF32,
                         std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
                         ssm[i].data());
      conv_b.emplace_back(d, DType::kF32,
                          std::vector<int64_t>{1, conv_dim, conv_len}, conv[i].data());
      gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
      gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
    }

    // GROUP 0: [2, num_blocks, block_size, num_kv_heads, head_size].
    std::vector<uint16_t> kv(
        static_cast<size_t>(2 * kBlocks0 * kPage * kKvHeads * kHeadDim), 0);
    vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                                {2 * kBlocks0, kPage, kKvHeads, kHeadDim}, kv.data());
    // GROUP 2: the fused MLA page, [num_blocks, block_size, indexer_head_dim],
    // POISONED so an unwritten row is distinguishable from a written zero.
    std::vector<uint16_t> idx(static_cast<size_t>(kIdxRows * kIdxHeadDim), kPoison);
    vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {kBlocks2, kPage, kIdxHeadDim},
                                 idx.data());

    std::vector<vllm::PagedKvCache> attn_kv(2);
    attn_kv[0].data = kv_b.t().data;
    attn_kv[0].dtype = DType::kBF16;
    attn_kv[0].num_blocks = kBlocks0;
    attn_kv[0].block_size = kPage;
    attn_kv[0].num_kv_heads = kKvHeads;
    attn_kv[0].head_size = kHeadDim;
    attn_kv[1].data = idx_b.t().data;
    attn_kv[1].dtype = DType::kBF16;
    attn_kv[1].num_blocks = kBlocks2;
    attn_kv[1].block_size = kPage;
    // `indexer_kv_heads` is 1 — upstream requires it — so the MLA page is one
    // vector of `indexer_head_dim` per token slot.
    attn_kv[1].num_kv_heads = kIdxKvHeads;
    attn_kv[1].head_size = kIdxHeadDim;

    // ─── THE CHANNEL, IN THE RUNNER'S OWN PUBLICATION ORDER ────────────────
    // One pass over the groups in the order `MakeQwen4ExpKVCache` published
    // them, emitting each group's names in the group's own order — which is
    // exactly `runner.cpp`'s by-name index build. The recurrent group sits
    // BETWEEN the two attention groups, so a flat index is NOT a payload slot.
    std::vector<std::string> names;
    std::vector<int32_t> group_ids, layer_indices, payload_slots;
    std::vector<uint8_t> payload_kinds;
    names.push_back("model.layers." + std::to_string(kQsaLayer) + ".self_attn.attn");
    group_ids.push_back(0);
    layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
    payload_slots.push_back(0);
    for (int i = 0; i < kGdnLayers; ++i) {
      names.push_back("model.layers." + std::to_string(i) + ".linear_attn");
      group_ids.push_back(1);
      layer_indices.push_back(i);
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kRecurrent));
      payload_slots.push_back(i);
    }
    names.push_back("model.layers." + std::to_string(kQsaLayer) +
                    ".self_attn.indexer.k_cache");
    group_ids.push_back(2);
    layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
    payload_slots.push_back(1);

    std::vector<std::vector<int32_t>> group_tables{bt0, bt_recurrent, idx_table};
    std::vector<int32_t> group_cols{static_cast<int32_t>(kCols), 1,
                                    static_cast<int32_t>(kCols)};

    // ─── THE THREE MALFORMED CHANNELS ─────────────────────────────────────
    // Each is a shape the ENGINE cannot judge — it does not know which names
    // this model expects — and each returns a wrong buffer with no shape error
    // if the hook trusts it.
    if (corrupt == kWrongName) {
      // The publisher and the consumer disagree about where the side cache is.
      names.back() = "model.layers." + std::to_string(kQsaLayer) +
                     ".self_attn.indexer.key_cache";
    } else if (corrupt == kWrongKind) {
      // A recurrent group classified as paged: `gdn_state[0]` would be read out
      // of `attn_kv` instead.
      payload_kinds[1] = static_cast<uint8_t>(vllm::KvCachePayload::kPaged);
    } else if (corrupt == kNoGroupTable) {
      // Group 2 allocated and its page map never gathered — the exact state
      // W5c-2 exists to remove, and the one that leaves a cache unreadable.
      group_tables[2].clear();
    }

    vllm::MultiKvCacheIndex mk;
    mk.layer_names = &names;
    mk.group_ids = &group_ids;
    mk.layer_indices = &layer_indices;
    mk.payload_kinds = &payload_kinds;
    mk.payload_slots = &payload_slots;
    mk.group_block_tables = &group_tables;
    mk.group_block_table_cols = &group_cols;
    if (corrupt == kNone) {
      REQUIRE(mk.size() == names.size());
      REQUIRE(mk.num_paged() == 2);
      REQUIRE(mk.num_recurrent() == kGdnLayers);
      REQUIRE(mk.num_published_groups() == 3);
      REQUIRE(mk.num_group_block_tables() == 3);
    }

    const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
    vllm::ModelForwardInput in{prompt,  pos,    am, gm, attn_kv,
                               gdn,     config, q,  logits_indices};
    in.num_reqs = 1;
    in.gdn_state_slots = 1;
    in.multi_kv = &mk;

    StepResult out;
    vllm::ForwardLogits fl;
    if (err != nullptr) {
      try {
        (void)vllm::ModelRegistry::Forward(*model, in);
        *err = "<the step RETURNED; the refusal is gone>";
      } catch (const std::exception& e) {
        *err = e.what();
      }
      return out;
    }
    // The guard is what this must get past: `ModelRegistry::Forward` refused
    // every non-null `multi_kv` before W5j.
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
    REQUIRE(fl.on_device());
    REQUIRE(fl.device_tensor.data != nullptr);
    REQUIRE(fl.device_tensor.dtype == DType::kF32);
    REQUIRE(fl.rows == 1);
    out.vocab = fl.vocab;
    out.logits.assign(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
    d.b.Copy(q, out.logits.data(), fl.device_tensor.data,
             out.logits.size() * sizeof(float));
    d.b.Synchronize(q);
    out.index_cache.assign(idx.size(), 0);
    idx_b.Download(d, out.index_cache.data());
    out.kv_cache.assign(kv.size(), 0);
    kv_b.Download(d, out.kv_cache.data());
    // THE TOKEN, sampled on the device the forward returned, by the same
    // `vt::GreedyArgmax` `GPUModelRunner` samples with. AGENTS.md routes decode
    // through on-device sampling and this is that primitive, not a host argmax.
    {
      std::vector<int64_t> t1(1, -1);
      vllm::dense_attn::DBuf tok_b(d, DType::kI64, {1});
      vt::Tensor tt = tok_b.t();
      vt::GreedyArgmax(q, tt, fl.device_tensor);
      tok_b.Download(d, t1.data());
      out.token = t1[0];
    }
    return out;
  };

  // ─── RUN A ────────────────────────────────────────────────────────────────
  StepResult a;
  REQUIRE_NOTHROW(a = run_step(bt2_a, ids));
  CHECK(a.vocab == kVocab);

  int finite = 0;
  for (float v : a.logits) finite += std::isfinite(v) ? 1 : 0;
  REQUIRE(finite == static_cast<int>(a.logits.size()));
  float lo = a.logits[0];
  float hi = a.logits[0];
  for (float v : a.logits) {
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  // Not a constant row: the lm_head times a zero hidden would be finite, in
  // range and argmax to 0.
  CHECK(hi > lo);

  MESSAGE("qwen4_exp by-name step: sampled token id "
          << a.token << " of " << a.vocab << " (logit range [" << lo << ", "
          << hi << "])");
  CHECK(a.token >= 0);
  CHECK(a.token < a.vocab);
  // The argmax must be THIS row's maximum, so the sampler read these logits and
  // not an uninitialised buffer.
  CHECK(a.logits[static_cast<size_t>(a.token)] == hi);

  // ─── THE STRUCTURAL ROW-SET ASSERTION ─────────────────────────────────────
  // Exactly the rows group 2's OWN table names for logical [0, T) were written,
  // and nothing else was touched. A value gate cannot see this; W5i measured a
  // 9-of-23 mis-page that left every value check green.
  const auto written_rows = [&](const std::vector<uint16_t>& cache) {
    std::vector<int64_t> w;
    for (int64_t r = 0; r < kIdxRows; ++r) {
      bool touched = false;
      for (int64_t c = 0; c < kIdxHeadDim; ++c)
        if (cache[static_cast<size_t>(r * kIdxHeadDim + c)] != kPoison) touched = true;
      if (touched) w.push_back(r);
    }
    return w;
  };
  std::vector<int64_t> expect_a = rows_of(bt2_a, T);
  std::sort(expect_a.begin(), expect_a.end());
  const std::vector<int64_t> got_a = written_rows(a.index_cache);
  {
    std::string s;
    for (int64_t r : got_a) s += std::to_string(r) + " ";
    INFO("group-2 rows written under table {1,3}: ", s);
    CHECK(got_a == expect_a);
  }
  // And it is NEITHER of the two wrong maps physically present in the step.
  std::vector<int64_t> identity(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) identity[static_cast<size_t>(t)] = t;
  CHECK(got_a != identity);
  std::vector<int64_t> via_group0 = rows_of(bt0, T);
  std::sort(via_group0.begin(), via_group0.end());
  CHECK(got_a != via_group0);

  // ─── RUN B: THE SAME STEP THROUGH A DIFFERENT GROUP-2 MAP ─────────────────
  // Paging is transparent to the ARITHMETIC and not to the BUFFER. A hook that
  // ignored the group's table would pass the first half of this pair and fail
  // the second.
  StepResult b;
  REQUIRE_NOTHROW(b = run_step(bt2_b, ids));
  REQUIRE(b.logits.size() == a.logits.size());
  const double drift = vllm_test::MaxAbsDiff(a.logits, b.logits);
  MESSAGE("qwen4_exp by-name: logit drift across two group-2 page maps = " << drift);
  CHECK(drift == 0.0);
  std::vector<int64_t> expect_b = rows_of(bt2_b, T);
  std::sort(expect_b.begin(), expect_b.end());
  const std::vector<int64_t> got_b = written_rows(b.index_cache);
  CHECK(got_b == expect_b);
  CHECK(got_b != got_a);

  // ─── AND THE STEP STILL DEPENDS ON ITS PROMPT ─────────────────────────────
  // Every assertion above is satisfied by a hook that never reads `token_ids`:
  // MUT-REACH on this row measured 90 of 91 assertions surviving the deletion of
  // the whole tower.
  std::vector<int32_t> ids2(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    ids2[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : (t + 1) * 2 + 1);
  for (int32_t v : ids2) REQUIRE(v < static_cast<int32_t>(kVocab));
  REQUIRE(ids2 != ids);
  StepResult c;
  REQUIRE_NOTHROW(c = run_step(bt2_a, ids2));
  const double moved = vllm_test::MaxAbsDiff(a.logits, c.logits);
  MESSAGE("qwen4_exp by-name second-prompt logit movement: " << moved);
  CHECK(moved > 0.0);
  // The indexer keys are the prompt's own, so the CONTENT of the written rows
  // moves too while the row SET does not.
  CHECK(written_rows(c.index_cache) == expect_a);
  // ─── WHAT THIS FIXTURE CANNOT GATE, MEASURED RATHER THAN ASSUMED ─────────
  // The CONTENT of neither cache moves with the prompt here, and the count
  // below is printed so the next reader does not write the assertion that
  // reddened on this exact line. It is a DYNAMIC-RANGE property of the fixture,
  // not a defect: the four-layer ramp puts the layer-3 activations near 2^18,
  // where one bf16 ULP is ~1024, and the two prompts move the logits by 31.84
  // out of 95090 — 0.03%, an order of magnitude under one ULP at that exponent.
  // The paged K/V, whose store is `vt::ReshapeAndCache` and is gated by every
  // other model in this tree, is prompt-invariant here for the same reason and
  // by the same count, which is what separates "the fixture saturates" from
  // "the indexer writes one row T times".
  //
  // So the ROW SET is what gates the paging on this fixture, exactly as
  // `.agents/reachability.md` and W5i's measurement require, and the CONTENT is
  // gated at the block instead (`test_qwen4_exp_qsa_block.cpp`, the W5i case,
  // whose values are chosen to separate). Recorded in the spec's `## Owed` as a
  // fixture the row may want to rescale.
  {
    int idx_moved = 0;
    for (size_t i = 0; i < a.index_cache.size(); ++i)
      idx_moved += a.index_cache[i] != c.index_cache[i] ? 1 : 0;
    int kv_moved = 0;
    for (size_t i = 0; i < a.kv_cache.size(); ++i)
      kv_moved += a.kv_cache[i] != c.kv_cache[i] ? 1 : 0;
    MESSAGE("qwen4_exp by-name: across two prompts, "
            << idx_moved << " of " << a.index_cache.size()
            << " indexer-cache words moved and " << kv_moved << " of "
            << a.kv_cache.size() << " paged K/V words moved");
  }
}

// ─────────────────────────────────────────────────────────────────────────────
// W5j — WHAT THE LIFTED GUARD STILL REFUSES.
//
// `ModelRegistry::Forward` no longer stops this architecture's multi-cache
// topology, so the model is now the only thing that can judge the CONTENT of the
// channel: the engine does not know which layer names `qwen4_exp` expects, which
// payload kind each is, or which group's block table has to have been gathered.
// Three ways the channel can lie, each refused by name at the hook's boundary.
//
// A BARE `CHECK_THROWS` HERE WOULD BE A MUTE SWITCH — that exact defect was
// found on this row, where the only assertion on a refusal was satisfied by a
// DIFFERENT exception thrown earlier. So each is asserted TWO-SIDED on its
// message: the bytes that identify THIS refusal are present, and the bytes of
// two OTHER refusals are absent —
//
//   * `qwen4_exp ple layout`, which is where an input that got past the boundary
//     and into the loop lands;
//   * `does not consume a cache set keyed by layer name`, the ENGINE's guard. A
//     case that stopped there proves nothing about the hook, and its absence is
//     what makes the green run above a lift and not a coincidence.
TEST_CASE(
    "qwen4_exp: the by-name channel is refused by name when it cannot be "
    "consumed") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  constexpr int64_t kQsaLayer = 3;
  constexpr int64_t kGdnLayers = 3;
  constexpr int64_t T = 6;
  constexpr int64_t kPage = 4;
  constexpr int64_t kCols = 2;
  constexpr int64_t kBlocks0 = 3;
  constexpr int64_t kBlocks2 = 4;
  const std::vector<int32_t> bt0{2, 0};
  const std::vector<int32_t> bt2{1, 3};
  const std::vector<int32_t> bt_recurrent{0};

  std::vector<int32_t> ids(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    ids[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  enum Corrupt { kWrongName = 1, kWrongKind = 2, kNoGroupTable = 3 };
  const auto refusal = [&](int corrupt) {
    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1;
    am.num_actual_tokens = static_cast<int>(T);
    am.block_table_num_cols = static_cast<int>(kCols);
    am.block_table_tensor = bt0;
    am.seq_lens.assign(1, static_cast<int32_t>(T));
    am.query_start_loc = {0, static_cast<int32_t>(T)};
    am.slot_mapping.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t)
      am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(
          static_cast<int64_t>(bt0[static_cast<size_t>(t / kPage)]) * kPage + t % kPage);

    vllm::v1::GDNAttentionMetadata gm;
    gm.num_prefills = 1;
    gm.num_prefill_tokens = static_cast<int>(T);
    gm.num_actual_tokens = static_cast<int>(T);
    gm.has_initial_state = std::vector<uint8_t>{0};
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    gm.prefill_state_indices = std::vector<int32_t>{0};
    gm.prefill_has_initial_state = std::vector<uint8_t>{0};
    {
      const auto conv =
          vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = conv.batch_ptr;
      gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    }

    const int64_t key_dim = kNumKHeads * kLinHeadDim;
    const int64_t value_dim = kNumVHeads * kLinHeadDim;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    const int64_t conv_len = kConvKernel - 1;
    const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;
    std::vector<std::vector<float>> ssm(kGdnLayers), conv(kGdnLayers);
    std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
    std::vector<vllm::GdnStateCache> gdn(kGdnLayers);
    ssm_b.reserve(kGdnLayers);
    conv_b.reserve(kGdnLayers);
    for (int i = 0; i < kGdnLayers; ++i) {
      ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
      conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
      ssm_b.emplace_back(d, DType::kF32,
                         std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
                         ssm[i].data());
      conv_b.emplace_back(d, DType::kF32,
                          std::vector<int64_t>{1, conv_dim, conv_len}, conv[i].data());
      gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
      gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
    }

    std::vector<uint16_t> kv(
        static_cast<size_t>(2 * kBlocks0 * kPage * kKvHeads * kHeadDim), 0);
    vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                                {2 * kBlocks0, kPage, kKvHeads, kHeadDim}, kv.data());
    std::vector<uint16_t> idx(
        static_cast<size_t>(kBlocks2 * kPage * kIdxHeadDim), 0);
    vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {kBlocks2, kPage, kIdxHeadDim},
                                 idx.data());
    std::vector<vllm::PagedKvCache> attn_kv(2);
    attn_kv[0].data = kv_b.t().data;
    attn_kv[0].dtype = DType::kBF16;
    attn_kv[0].num_blocks = kBlocks0;
    attn_kv[0].block_size = kPage;
    attn_kv[0].num_kv_heads = kKvHeads;
    attn_kv[0].head_size = kHeadDim;
    attn_kv[1].data = idx_b.t().data;
    attn_kv[1].dtype = DType::kBF16;
    attn_kv[1].num_blocks = kBlocks2;
    attn_kv[1].block_size = kPage;
    attn_kv[1].num_kv_heads = kIdxKvHeads;
    attn_kv[1].head_size = kIdxHeadDim;

    std::vector<std::string> names;
    std::vector<int32_t> group_ids, layer_indices, payload_slots;
    std::vector<uint8_t> payload_kinds;
    names.push_back("model.layers." + std::to_string(kQsaLayer) + ".self_attn.attn");
    group_ids.push_back(0);
    layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
    payload_slots.push_back(0);
    for (int i = 0; i < kGdnLayers; ++i) {
      names.push_back("model.layers." + std::to_string(i) + ".linear_attn");
      group_ids.push_back(1);
      layer_indices.push_back(i);
      payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kRecurrent));
      payload_slots.push_back(i);
    }
    names.push_back("model.layers." + std::to_string(kQsaLayer) +
                    ".self_attn.indexer.k_cache");
    group_ids.push_back(2);
    layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
    payload_slots.push_back(1);
    std::vector<std::vector<int32_t>> group_tables{bt0, bt_recurrent, bt2};
    std::vector<int32_t> group_cols{static_cast<int32_t>(kCols), 1,
                                    static_cast<int32_t>(kCols)};

    if (corrupt == kWrongName) {
      names.back() = "model.layers." + std::to_string(kQsaLayer) +
                     ".self_attn.indexer.key_cache";
    } else if (corrupt == kWrongKind) {
      payload_kinds[1] = static_cast<uint8_t>(vllm::KvCachePayload::kPaged);
    } else if (corrupt == kNoGroupTable) {
      group_tables[2].clear();
    }

    vllm::MultiKvCacheIndex mk;
    mk.layer_names = &names;
    mk.group_ids = &group_ids;
    mk.layer_indices = &layer_indices;
    mk.payload_kinds = &payload_kinds;
    mk.payload_slots = &payload_slots;
    mk.group_block_tables = &group_tables;
    mk.group_block_table_cols = &group_cols;

    const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
    vllm::ModelForwardInput in{ids,  pos,    am, gm, attn_kv,
                               gdn,  config, q,  logits_indices};
    in.num_reqs = 1;
    in.gdn_state_slots = 1;
    in.multi_kv = &mk;

    std::string msg = "<the step RETURNED; this refusal is gone>";
    try {
      (void)vllm::ModelRegistry::Forward(*model, in);
    } catch (const std::exception& e) {
      msg = e.what();
    }
    return msg;
  };

  // The two strings that must be ABSENT from every one of them.
  const char* kPle = "qwen4_exp ple layout";
  const char* kEngine = "does not consume a cache set keyed by layer name";

  SUBCASE("a published name this model does not expect") {
    const std::string m = refusal(kWrongName);
    INFO("the wrong-name refusal said: ", m);
    CHECK(m.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(m.find("published no KV cache under") != std::string::npos);
    // The NAME, not only the shape of the complaint: the hook reports the string
    // it asked for, which is what routes the reader to the disagreement.
    CHECK(m.find("self_attn.indexer.k_cache") != std::string::npos);
    CHECK(m.find(kPle) == std::string::npos);
    CHECK(m.find(kEngine) == std::string::npos);
  }

  SUBCASE("a recurrent group published as a paged one") {
    const std::string m = refusal(kWrongKind);
    INFO("the wrong-kind refusal said: ", m);
    CHECK(m.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(m.find("model.layers.0.linear_attn") != std::string::npos);
    CHECK(m.find("RECURRENT") != std::string::npos);
    CHECK(m.find("PAGED") != std::string::npos);
    CHECK(m.find(kPle) == std::string::npos);
    CHECK(m.find(kEngine) == std::string::npos);
  }

  SUBCASE("the indexer group's block table was never gathered") {
    const std::string m = refusal(kNoGroupTable);
    INFO("the ungathered-table refusal said: ", m);
    CHECK(m.find("Qwen4ExpForConditionalGeneration") != std::string::npos);
    CHECK(m.find("gathered no block table for published group 2") !=
          std::string::npos);
    // The DENOMINATOR, so a partial gather reads as partial: two of three.
    CHECK(m.find("2 of 3 published group(s) carry one") != std::string::npos);
    CHECK(m.find(kPle) == std::string::npos);
    CHECK(m.find(kEngine) == std::string::npos);
  }
}

// ─── A SECOND STEP: THE DECODE (W5k, #2031) ──────────────────────────────────
//
// Every case above serves ONE step. This one runs TWO through
// `ModelRegistry::Forward` over caches that PERSIST between them, which is the
// thing the row has been unable to do since W5c published the topology:
// `past_len > 0` was refused because the recurrent group publishes the PLE conv
// ring at the model dtype and the n-gram history on the device, while
// `RunQwen4ExpPleBlock` required an f32 ring and a host history.
//
// W5k settled which side was wrong by INSTALLING the lane oracle rather than
// reasoning from this tree's convention — transformers 5.16.0,
// `models/qwen4_exp/modeling_qwen4_exp.py` sha256 77fec77d…c459, the pin
// `scripts/gen-qwen4-exp-forward-goldens.py` asserts — and reading the running
// model. It types each cache slot from the tensor that first reaches it
// (`cache_utils.py:1019-1023`), so the ring carries the MODEL dtype (observed:
// `conv_states[1] dtype=torch.bfloat16` on a bf16 run) and the history carries
// `input_ids.long()` on `input_ids.device` (`:1070`, `:1089-1091`). The
// PUBLISHER was right twice and both requirements moved to the block.
//
// ─── WHAT THIS GATES, AND WHY NOT BY VALUE ───────────────────────────────────
// W5j MEASURED that this fixture cannot gate cache CONTENT: across two prompts,
// 0 of 128 indexer words and 0 of 192 paged K/V words moved while the logits
// moved 31.84, because layer-3 activations sit near 2^18 where one bf16 ULP is
// ~1024 and the store saturates. A rescaled fixture is owed. So nothing here
// asserts a cache VALUE.
//
// What it asserts instead is the one cross-step observable that CANNOT saturate:
// the n-gram history is int64 TOKEN IDS. Upstream keeps the last
// `ngram_size - 1` raw ids and rolls them every step — measured on the pinned
// oracle over four steps of the prompt [5,9,13,3,7,2]:
//
//     after prefill      ngram_history = [7, 2]      (the prompt's last two)
//     after decode 11    ngram_history = [2, 11]
//     after decode 4     ngram_history = [11, 4]
//     after decode 3     ngram_history = [4, 3]
//
// That is a FIFO whose contents are exact integers at any dtype, so the
// assertions below read the engine's own recurrent state and compare it to ids,
// not to a rounded activation. And the step-2 read's WRITER IS A PRIOR CALL,
// which is the property W5j's single-step cases could not have.
TEST_CASE("qwen4_exp: a SECOND step decodes on the engine's own persistent caches") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  constexpr int64_t kQsaLayer = 3;
  constexpr int64_t kGdnLayers = 3;
  constexpr int64_t kPleLayer = 1;   // ple_layer_ids = [2], one-based
  constexpr int64_t T1 = 6;          // the prefill
  constexpr int64_t kPage = 4;
  constexpr int64_t kCols = 3;       // T1 + 1 token spans 2 pages; 3 is slack
  constexpr int64_t kBlocks0 = 4;
  constexpr int64_t kBlocks2 = 5;
  const int64_t kIdxRows = kBlocks2 * kPage;

  // The recurrent state geometry, exactly as `MakeQwen4ExpKVCache` publishes it.
  const int64_t kConvDim = 2 * kKeyDim + kValueDim;
  const int64_t kGdnConvLen = kConvKernel - 1;
  const int64_t kSsmRow = kNumVHeads * kLinHeadDim * kLinHeadDim;
  const int64_t kPleStateLen = (kConvKernel - 1) * kNgramSize;  // DILATED: 9
  const int64_t kCtx = kNgramSize - 1;                          // 2

  const std::vector<int32_t> bt0{2, 0, 3};
  const std::vector<int32_t> bt2{1, 3, 4};
  const std::vector<int32_t> bt_recurrent{0};

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  // ─── THE CACHES, ALLOCATED ONCE AND SHARED BY BOTH STEPS ──────────────────
  // This is the whole point: a per-call scratch is zeroed on entry, so a second
  // step over one would re-seed the n-gram history with EOS and re-zero the ring
  // and the model would decode as though each token were the first.
  std::vector<std::vector<unsigned char>> st_gdn_conv(kGdnLayers),
      st_ssm(kGdnLayers), st_ple_conv(kGdnLayers);
  std::vector<std::vector<int64_t>> st_ngram(kGdnLayers);
  std::vector<vllm::dense_attn::DBuf> b_gdn_conv, b_ssm, b_ple_conv, b_ngram;
  std::vector<vllm::GdnStateCache> gdn(kGdnLayers);
  b_gdn_conv.reserve(kGdnLayers); b_ssm.reserve(kGdnLayers);
  b_ple_conv.reserve(kGdnLayers); b_ngram.reserve(kGdnLayers);
  for (int i = 0; i < kGdnLayers; ++i) {
    st_gdn_conv[i].assign(static_cast<size_t>(kConvDim * kGdnConvLen) * sizeof(float), 0);
    st_ssm[i].assign(static_cast<size_t>(kSsmRow) * sizeof(float), 0);
    st_ple_conv[i].assign(
        static_cast<size_t>(qwen4_exp_fixture::kStream * kPleStateLen) *
            vt::SizeOf(vllm::kQwen4ExpStreamDType), 0);
    st_ngram[i].assign(static_cast<size_t>(kCtx), 0);
    b_gdn_conv.emplace_back(d, DType::kF32,
                            std::vector<int64_t>{1, kConvDim, kGdnConvLen},
                            st_gdn_conv[i].data());
    b_ssm.emplace_back(d, DType::kF32,
                       std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
                       st_ssm[i].data());
    // THE RING AT THE MODEL DTYPE and the HISTORY ON THIS STEP'S DEVICE — the
    // two the publisher declares and the two the block refused before W5k.
    b_ple_conv.emplace_back(d, vllm::kQwen4ExpStreamDType,
                            std::vector<int64_t>{1, qwen4_exp_fixture::kStream,
                                                 kPleStateLen},
                            st_ple_conv[i].data());
    b_ngram.emplace_back(d, DType::kI64, std::vector<int64_t>{1, kCtx},
                         st_ngram[i].data());
    gdn[static_cast<size_t>(i)].conv_state = b_gdn_conv.back().t();
    gdn[static_cast<size_t>(i)].ssm_state = b_ssm.back().t();
    // THE PUBLISHED ORDER: [gdn_conv, temporal, ple_conv, ngram]. Filled by
    // `GPUModelRunner::initialize_kv_cache` in production; filled here because
    // this case stands in for it.
    gdn[static_cast<size_t>(i)].states = {b_gdn_conv.back().t(), b_ssm.back().t(),
                                          b_ple_conv.back().t(), b_ngram.back().t()};
  }
  REQUIRE(gdn[kPleLayer].states.size() == 4);

  std::vector<uint16_t> kv(
      static_cast<size_t>(2 * kBlocks0 * kPage * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2 * kBlocks0, kPage, kKvHeads, kHeadDim}, kv.data());
  constexpr uint16_t kPoison = 0x3F80;
  std::vector<uint16_t> idx(static_cast<size_t>(kIdxRows * kIdxHeadDim), kPoison);
  vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {kBlocks2, kPage, kIdxHeadDim},
                               idx.data());

  std::vector<vllm::PagedKvCache> attn_kv(2);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = kBlocks0;
  attn_kv[0].block_size = kPage;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;
  attn_kv[1].data = idx_b.t().data;
  attn_kv[1].dtype = DType::kBF16;
  attn_kv[1].num_blocks = kBlocks2;
  attn_kv[1].block_size = kPage;
  attn_kv[1].num_kv_heads = kIdxKvHeads;
  attn_kv[1].head_size = kIdxHeadDim;

  std::vector<std::string> names;
  std::vector<int32_t> group_ids, layer_indices, payload_slots;
  std::vector<uint8_t> payload_kinds;
  names.push_back("model.layers." + std::to_string(kQsaLayer) + ".self_attn.attn");
  group_ids.push_back(0);
  layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
  payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
  payload_slots.push_back(0);
  for (int i = 0; i < kGdnLayers; ++i) {
    names.push_back("model.layers." + std::to_string(i) + ".linear_attn");
    group_ids.push_back(1);
    layer_indices.push_back(i);
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kRecurrent));
    payload_slots.push_back(i);
  }
  names.push_back("model.layers." + std::to_string(kQsaLayer) +
                  ".self_attn.indexer.k_cache");
  group_ids.push_back(2);
  layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
  payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
  payload_slots.push_back(1);

  std::vector<std::vector<int32_t>> group_tables{bt0, bt_recurrent, bt2};
  std::vector<int32_t> group_cols{static_cast<int32_t>(kCols), 1,
                                  static_cast<int32_t>(kCols)};
  vllm::MultiKvCacheIndex mk;
  mk.layer_names = &names;
  mk.group_ids = &group_ids;
  mk.layer_indices = &layer_indices;
  mk.payload_kinds = &payload_kinds;
  mk.payload_slots = &payload_slots;
  mk.group_block_tables = &group_tables;
  mk.group_block_table_cols = &group_cols;

  // The physical slot a logical position lands on under group 0's table.
  const auto slot_of = [&](int64_t t) {
    return static_cast<int32_t>(bt0[static_cast<size_t>(t / kPage)] * kPage + t % kPage);
  };

  // ─── ONE STEP, OVER THE CACHES ABOVE ──────────────────────────────────────
  struct Step { std::vector<float> logits; int64_t vocab = 0; int64_t token = -1; };
  const auto run = [&](const std::vector<int32_t>& tok_ids, int64_t past_len) {
    const auto T = static_cast<int64_t>(tok_ids.size());
    std::vector<int32_t> pos(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t)
      pos[static_cast<size_t>(t)] = static_cast<int32_t>(past_len + t);

    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1;
    am.num_actual_tokens = static_cast<int>(T);
    am.max_query_len = static_cast<int>(T);
    am.query_start_loc.assign({0, static_cast<int32_t>(T)});
    am.seq_lens.assign(1, static_cast<int32_t>(past_len + T));
    am.block_table_tensor = bt0;
    am.block_table_num_cols = static_cast<int>(kCols);
    am.max_seq_len = static_cast<int>(past_len + T);
    am.slot_mapping.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t)
      am.slot_mapping[static_cast<size_t>(t)] = slot_of(past_len + t);

    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = static_cast<int>(T);
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    if (past_len == 0) {
      gm.num_prefills = 1;
      gm.num_prefill_tokens = static_cast<int>(T);
      gm.has_initial_state = std::vector<uint8_t>{0};
      gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
      gm.prefill_state_indices = std::vector<int32_t>{0};
      // `prefill_has_initial_state == 0` IS upstream's
      // `has_previous_state(layer_idx, state_idx=2) == False`, which is the
      // branch that seeds the n-gram history with `eos_token_id` rather than
      // trusting the zero-filled cache (`modeling_qwen4_exp.py:1073-1076`).
      // ZERO IS A VALID TOKEN ID, so that seed is not decoration.
      gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto conv =
          vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = conv.batch_ptr;
      gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    } else {
      // A PURE DECODE. Upstream populates `has_initial_state` only when there is
      // a prefill (`gdn_attn.py:389-405`), and the single-token update kernel is
      // selected instead of the conv program map.
      gm.num_decodes = 1;
      gm.num_decode_tokens = static_cast<int>(T);
    }

    const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
    vllm::ModelForwardInput in{tok_ids, pos,    am, gm, attn_kv,
                               gdn,     config, q,  logits_indices};
    in.num_reqs = 1;
    in.gdn_state_slots = 1;
    in.multi_kv = &mk;

    Step out;
    vllm::ForwardLogits fl;
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
    REQUIRE(fl.on_device());
    REQUIRE(fl.rows == 1);
    out.vocab = fl.vocab;
    out.logits.assign(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
    d.b.Copy(q, out.logits.data(), fl.device_tensor.data,
             out.logits.size() * sizeof(float));
    d.b.Synchronize(q);
    {
      std::vector<int64_t> t1(1, -1);
      vllm::dense_attn::DBuf tok_b(d, DType::kI64, {1});
      vt::Tensor tt = tok_b.t();
      vt::GreedyArgmax(q, tt, fl.device_tensor);
      tok_b.Download(d, t1.data());
      out.token = t1[0];
    }
    return out;
  };

  // The prompt carries an EOS in the INTERIOR, which is a segment boundary in
  // the hashed n-gram construction, so the ids after it are not a straight ramp.
  std::vector<int32_t> prompt{5, 9, 13, static_cast<int32_t>(kEosTokenId), 7, 2};
  REQUIRE(static_cast<int64_t>(prompt.size()) == T1);
  for (int32_t v : prompt) REQUIRE(v < static_cast<int32_t>(kVocab));

  // ─── STEP 1: THE PREFILL ──────────────────────────────────────────────────
  Step s1;
  REQUIRE_NOTHROW(s1 = run(prompt, /*past_len=*/0));
  CHECK(s1.vocab == kVocab);
  for (float v : s1.logits) REQUIRE(std::isfinite(v));
  CHECK(s1.token >= 0);
  CHECK(s1.token < s1.vocab);
  MESSAGE("qwen4_exp STEP 1 (prefill, T=" << T1 << ") sampled token " << s1.token);

  // THE HISTORY THE PREFILL LEFT BEHIND is the prompt's last `ngram_size - 1`
  // ids — upstream's `[7, 2]` for this prompt. Read out of the ENGINE's own
  // recurrent state, and compared to the INPUT IDS rather than to a value read
  // back from the cache before the run, so a cache that was never written fails.
  std::vector<int64_t> hist(static_cast<size_t>(kCtx), -1);
  b_ngram[kPleLayer].Download(d, hist.data());
  for (int64_t i = 0; i < kCtx; ++i) {
    INFO("history slot ", i);
    CHECK(hist[static_cast<size_t>(i)] ==
          static_cast<int64_t>(prompt[static_cast<size_t>(T1 - kCtx + i)]));
  }

  // NOTHING WAS WRITTEN TO THE OTHER LAYERS' PLE SLOTS. Only layer 1 carries a
  // PLE block, and the uniform recurrent group gives all three linear layers the
  // same state set; a hook that indexed the PLE cache by DECODER layer index
  // instead of by rank among linear layers would write layer 2's or layer 3's.
  for (int i = 0; i < kGdnLayers; ++i) {
    if (i == kPleLayer) continue;
    std::vector<int64_t> other(static_cast<size_t>(kCtx), -1);
    b_ngram[static_cast<size_t>(i)].Download(d, other.data());
    INFO("linear layer ", i, " must own no PLE history");
    CHECK(other == std::vector<int64_t>(static_cast<size_t>(kCtx), 0));
  }

  // ─── STEP 2: THE DECODE — past_len > 0, ON THE SAME BUFFERS ───────────────
  // This is the call every wave since W5c has been refused at.
  const std::vector<int32_t> next{static_cast<int32_t>(s1.token)};
  Step s2;
  REQUIRE_NOTHROW(s2 = run(next, /*past_len=*/T1));
  CHECK(s2.vocab == kVocab);
  for (float v : s2.logits) REQUIRE(std::isfinite(v));
  CHECK(s2.token >= 0);
  CHECK(s2.token < s2.vocab);
  MESSAGE("qwen4_exp STEP 2 (decode, past_len=" << T1 << ", token " << s1.token
          << ") sampled token " << s2.token);

  // THE FIFO ROLLED, AND THAT IS THE CROSS-STEP GATE. Upstream's history after a
  // decode is `[previous_last, this_token]` — measured on the pinned oracle as
  // `[7,2] -> [2,11]` after decoding 11. The first slot must now hold what the
  // SECOND slot held before the step, which no single-step run can produce and
  // which a re-seeded scratch would render as `[eos, token]` instead.
  std::vector<int64_t> hist2(static_cast<size_t>(kCtx), -1);
  b_ngram[kPleLayer].Download(d, hist2.data());
  CHECK(hist2[static_cast<size_t>(kCtx - 1)] == s1.token);
  for (int64_t i = 0; i + 1 < kCtx; ++i) {
    INFO("rolled slot ", i);
    CHECK(hist2[static_cast<size_t>(i)] == hist[static_cast<size_t>(i + 1)]);
  }
  // And it is NOT the seed a per-call scratch would have produced.
  const std::vector<int64_t> seeded{static_cast<int64_t>(kEosTokenId), s1.token};
  CHECK(hist2 != seeded);

  // THE LOGITS MOVED. Step 2 sees a one-token query at position T1 against six
  // cached keys; step 1 saw six queries. Identical rows would mean the second
  // call recomputed the first.
  REQUIRE(s1.logits.size() == s2.logits.size());
  CHECK(std::memcmp(s1.logits.data(), s2.logits.data(),
                    s1.logits.size() * sizeof(float)) != 0);
}

// ─────────────────────────────────────────────────────────────────────────────
// W5p (#2031) — A Q8_0 MIX WEIGHT REACHES THE PRODUCTION ENTRY POINT.
//
// The released `unsloth/Qwen3.8-Flash-Next-GGUF` stores all 194 hyper-connection
// mix weights as Q8_0 and loaded cleanly into this tree; the first prefill then
// died inside `vt::Qwen4ExpGatedResidual`'s own validation, because every arm of
// the fixture and every op-level case wrote those tensors F32. This case is the
// arm that was missing: the SAME `ModelRegistry::Forward` the case above drives,
// over a file whose `hc_*_down` and `hc_*_inject` are block-typed.
//
// IT IS A REACHABILITY GATE, NOT A NUMERIC ONE. The fixture's weights are a
// deterministic pattern rather than a checkpoint, so there is no reference token
// stream; the arithmetic of the quantized route is gated at op level by
// `test_qwen4_exp_hc_device.cpp` against an independent double reference, and
// the loader half by `test_qwen4_exp_gguf_weights.cpp`. What is asserted here is
// that a block-typed mix weight survives the whole path from the file to a
// sampled token through the entry point AGENTS.md names.
//
// `hc_*_up` stays F32 in the file because ggml forbids a quantized tensor whose
// fastest dim is not a whole 32-element block and this miniature's low-rank is
// 8. The released config's low-rank is 320.
TEST_CASE("qwen4_exp: ModelRegistry::Forward runs a Q8_0 hyper-connection mix weight") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  FixtureOpts opts;
  opts.hc_mix_q8_0 = true;
  const gguf_test::TempFile f(BuildFixture(opts));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  // THE FILE, ASSERTED BEFORE THE MODEL. An option that quietly wrote F32 would
  // turn this whole case into a second run of the case above.
  REQUIRE(g.Get("blk.0.hc_attn_down.weight").ggml_type == 8);
  REQUIRE(g.Get("blk.0.hc_attn_inject.weight").ggml_type == 8);
  REQUIRE(g.Get("blk.0.hc_ffn_down.weight").ggml_type == 8);
  REQUIRE(g.Get("output_hc_down.weight").ggml_type == 8);

  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids[static_cast<size_t>(t)] = static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, T, kKvHeads, kHeadDim}, kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids, pos, am, gm, attn_kv, gdn, config, q,
                             logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  // THE RED THIS CASE WAS WRITTEN AGAINST, in its own words: before this wave
  // `vt::Qwen4ExpGatedResidual` threw
  //   `qwen4_exp_gated_residual: input_mix_weight_down must be float
  //    (f32/bf16 for outputs)`
  // here, on the FIRST hyper-connection of layer 0.
  vllm::ForwardLogits fl;
  REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  CHECK(fl.rows == 1);
  CHECK(fl.vocab == kVocab);
  REQUIRE(fl.on_device());
  REQUIRE(fl.device_tensor.data != nullptr);
  REQUIRE(fl.device_tensor.dtype == DType::kF32);

  std::vector<float> host(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
  d.b.Copy(q, host.data(), fl.device_tensor.data, host.size() * sizeof(float));
  d.b.Synchronize(q);

  // FINITENESS BEFORE ANY OTHER READING (#449): a max-fold over `std::max`
  // returns the non-NaN operand, so an all-NaN row reads as a match to any
  // tolerance and still argmaxes to an index in range.
  int finite = 0;
  for (float v : host) finite += std::isfinite(v) ? 1 : 0;
  REQUIRE(finite == static_cast<int>(host.size()));
  float lo = host[0];
  float hi = host[0];
  for (float v : host) {
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  // A CONSTANT row is what a tower contributing nothing produces.
  CHECK(hi > lo);

  std::vector<int64_t> tok(1, -1);
  {
    vllm::dense_attn::DBuf tok_b(d, DType::kI64, {1});
    vt::Tensor tt = tok_b.t();
    vt::GreedyArgmax(q, tt, fl.device_tensor);
    tok_b.Download(d, tok.data());
  }
  MESSAGE("qwen4_exp Q8_0-mix sampled token id: " << tok[0] << " of " << fl.vocab
                                                  << " (logit range [" << lo
                                                  << ", " << hi << "])");
  CHECK(tok[0] >= 0);
  CHECK(tok[0] < fl.vocab);
  CHECK(host[static_cast<size_t>(tok[0])] == hi);

  // AND IT DEPENDS ON THE PROMPT. Every assertion above is satisfied by a hook
  // that never reads `token_ids` — MUT-REACH measured exactly that on the
  // sibling case. Fresh zeroed caches leave the prompt as the only input that
  // moved.
  std::vector<int32_t> ids2(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids2[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : (t + 1) * 4 + 1);
    REQUIRE(ids2[static_cast<size_t>(t)] < static_cast<int32_t>(kVocab));
  }
  REQUIRE(ids2 != ids);

  std::vector<std::vector<float>> ssm2(3), conv2(3);
  std::vector<vllm::dense_attn::DBuf> ssm2_b, conv2_b;
  std::vector<vllm::GdnStateCache> gdn2(3);
  ssm2_b.reserve(3);
  conv2_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm2[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv2[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm2_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm2[i].data());
    conv2_b.emplace_back(d, DType::kF32,
                         std::vector<int64_t>{1, conv_dim, conv_len},
                         conv2[i].data());
    gdn2[static_cast<size_t>(i)].ssm_state = ssm2_b.back().t();
    gdn2[static_cast<size_t>(i)].conv_state = conv2_b.back().t();
  }
  std::vector<uint16_t> kv2(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv2_b(d, DType::kBF16,
                               {2, T, kKvHeads, kHeadDim}, kv2.data());
  std::vector<vllm::PagedKvCache> attn_kv2(1);
  attn_kv2[0] = attn_kv[0];
  attn_kv2[0].data = kv2_b.t().data;

  vllm::ModelForwardInput in2{ids2, pos, am, gm, attn_kv2, gdn2, config, q,
                              logits_indices};
  in2.num_reqs = 1;
  in2.gdn_state_slots = 1;

  vllm::ForwardLogits fl2;
  REQUIRE_NOTHROW(fl2 = vllm::ModelRegistry::Forward(*model, in2));
  REQUIRE(fl2.on_device());
  std::vector<float> host2(static_cast<size_t>(fl2.rows * fl2.vocab), 0.0F);
  d.b.Copy(q, host2.data(), fl2.device_tensor.data, host2.size() * sizeof(float));
  d.b.Synchronize(q);
  const double moved = vllm_test::MaxAbsDiff(host, host2);
  MESSAGE("qwen4_exp Q8_0-mix second-prompt logit movement: " << moved);
  CHECK(moved > 0.0);
}

// ── THE PRODUCTION ENTRY POINT, ON A CUDA QUEUE ─────────────────────────────
//
// `ModelRegistry::Forward` is the entry point AGENTS.md names, and until #2396
// and #2391 landed it could not be driven on a device at all: the loader refused
// every non-CPU device before a tensor was read, and four of this model's `vt::`
// ops had no `kCUDA` arm. Both are gone, so this case asks the question that
// could not previously be asked — HOW FAR does a CUDA forward of this
// architecture get, and what stops it?
//
// IT IS A MEASUREMENT, NOT AN ASSERTION THAT THE MODEL RUNS. The case prints
// where the forward stopped and asserts only what this row owns: that the reason
// is not a RESIDENCY. Before #2421 the answer was
// `the two rope layouts are cross-checked on the host, so both must be
// CPU-resident`, thrown from `CheckRopeLayoutsAgree` at decoder layer 3 — the
// first `qwen_sparse_attention` layer, after PLE and three MoE blocks. A reader
// who wants to see that red should check out the parent of #2421's commit and
// run this case; that is the red this case was written against.
//
// THE INSTRUMENT'S OWN PRECONDITION IS ASSERTED FIRST. `LoadThroughRegistry`
// takes its device from `platforms::CurrentPlatform()`, not from an argument, so
// on a build where CUDA did not register this case would quietly load on the CPU
// and measure nothing while reporting a pass. That is checked rather than
// assumed.
namespace {

bool LayerLoopHasCuda() {
  try {
    vt::GetBackend(vt::DeviceType::kCUDA);
    return true;
  } catch (const std::exception&) {
    return false;
  }
}

// The three refusals #2421 removed, by the words they refused with.
bool LayerLoopIsResidencyRefusal(const std::string& what) {
  return what.find("CPU-resident") != std::string::npos ||
         what.find("read on the host") != std::string::npos ||
         what.find("cross-checked on the host") != std::string::npos;
}

}  // namespace

TEST_CASE("qwen4_exp: ModelRegistry::Forward on a CUDA queue gets past the QSA block") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  if (!LayerLoopHasCuda()) {
    MESSAGE("no CUDA backend in this build: the device forward is UNMEASURED by this run");
    return;
  }
  // THE PRECONDITION. The loader reads the device off the platform registry, so
  // this is what separates "loaded on CUDA" from "loaded on the CPU and told you
  // nothing".
  REQUIRE(vllm::platforms::CurrentPlatform().device_type() == vt::DeviceType::kCUDA);

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);

  // THE LOAD IS PART OF THE MEASUREMENT. #2396 opened this gate; if it closes
  // again the case must say so rather than fail somewhere later.
  std::unique_ptr<vllm::LoadedModel> model;
  std::string load_stopped_with;
  try {
    model = LoadThroughRegistry(g);
  } catch (const std::exception& e) {
    load_stopped_with = e.what();
  }
  INFO("CUDA load stopped with: ", load_stopped_with);
  REQUIRE(load_stopped_with.empty());
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids[static_cast<size_t>(t)] = static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  // THE CUDA QUEUE. Everything the hook allocates is a `dense_attn::DBuf` over
  // `d.q.device`, so this one line is what makes every operand the QSA block
  // sees device-resident — which is exactly the condition the three refusals
  // this row removed used to reject.
  vt::Backend& gpu = vt::GetBackend(vt::DeviceType::kCUDA);
  vt::Queue q = gpu.CreateQueue();
  vllm::dense_attn::Dev d{gpu, q};

  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, T, kKvHeads, kHeadDim}, kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids, pos, am, gm, attn_kv, gdn, config, q,
                             logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  vllm::ForwardLogits fl;
  std::string stopped_with;
  try {
    fl = vllm::ModelRegistry::Forward(*model, in);
  } catch (const std::exception& e) {
    stopped_with = e.what();
  }

  // ─── WHAT THIS ROW OWNS ───────────────────────────────────────────────────
  MESSAGE("CUDA ModelRegistry::Forward stopped with: ",
          stopped_with.empty() ? std::string("(it did not stop)") : stopped_with);
  CHECK_FALSE(LayerLoopIsResidencyRefusal(stopped_with));

  if (!stopped_with.empty()) {
    // NOT A PASS DRESSED AS ONE. The forward did not complete, the case says so
    // in its own output, and the assertion above is the only claim it makes.
    MESSAGE("the CUDA forward did not complete; a residency is no longer the reason");
    gpu.DestroyQueue(q);
    return;
  }

  // ─── AND WHAT IT DOES NOT ─────────────────────────────────────────────────
  // FINITENESS FIRST: a fold over `std::max` returns the non-NaN operand, so an
  // all-NaN row reads as a perfect match to any tolerance and an argmax over it
  // still returns an index in range.
  REQUIRE(fl.on_device());
  REQUIRE(fl.device_tensor.data != nullptr);
  REQUIRE(fl.device_tensor.dtype == DType::kF32);
  std::vector<float> host(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
  gpu.Copy(q, host.data(), fl.device_tensor.data, host.size() * sizeof(float));
  gpu.Synchronize(q);
  int finite = 0;
  for (float v : host) finite += std::isfinite(v) ? 1 : 0;
  CHECK(finite == static_cast<int>(host.size()));
  float lo = host[0], hi = host[0];
  for (float v : host) {
    lo = v < lo ? v : lo;
    hi = v > hi ? v : hi;
  }
  // A CONSTANT row is finite, in range, and argmaxes to 0 while meaning the
  // tower contributed nothing.
  CHECK(hi > lo);

  std::vector<int64_t> tok(1, -1);
  {
    vllm::dense_attn::DBuf tok_b(d, DType::kI64, {1});
    vt::Tensor tt = tok_b.t();
    vt::GreedyArgmax(q, tt, fl.device_tensor);
    tok_b.Download(d, tok.data());
  }
  MESSAGE("qwen4_exp CUDA forward sampled token id: ", tok[0], " of ", kVocab,
          " (logit range [", lo, ", ", hi, "])");
  CHECK(tok[0] >= 0);
  CHECK(tok[0] < kVocab);
  gpu.DestroyQueue(q);
}

// ─────────────────────────────────────────────────────────────────────────────
// OPERAND LIFETIME (issue #2476, spec `.agents/specs/gdn-qkvz-operand-lifetime.md`)
//
// WHAT FAILED, AND WHY NO EXISTING CASE COULD SEE IT. `compute-sanitizer` on
// `thor:gpu0` reported `Warp illegal address` inside
// `nvjet_sm110_tst_512x8_64x3_2x1_v_bz_TNT`, reached through `cublasLtMatmul` <-
// `MatmulBTKernelCuda` <- `MatmulBf16D` <- `ProjectGdnQkvz`. Every extent that
// GEMM was given — `M`, `N`, `K`, and the declared byte length of each operand —
// agreed with the allocation AT LAUNCH TIME. What did not survive was the
// allocation: `Qwen4ExpGdnBlockWeights` DEEP-COPIED the loader's owned weight
// bytes into a per-step temporary, `ResidentWeight`'s host-alias arm handed
// cuBLASLt a pointer INTO that temporary, and the temporary was destroyed at the
// end of the layer's scope while the GEMM was still only queued.
//
// A CPU forward cannot fault on that, so this case does not try to. It asserts
// the PROPERTY whose violation is the defect: the bytes a block hands to a
// kernel belong to the model, not to a temporary. That is checkable with no
// device, and it is what `qwen4_exp_forward.h` already CLAIMS above the
// declaration ("nothing is ... reallocated here — the returned `OwnedTensor`s
// are COPIES OF THE HANDLES and share the loader's bytes"). The claim was true
// of the comment and false of the code.
TEST_CASE(
    "qwen4_exp: the GDN block adapter shares the loader's bytes rather than "
    "copying them into a per-step temporary") {
  const Qwen4ExpParams p = MakeParams();
  Qwen4ExpWeights w = MakeWeights(p);

  // The layer the fixture builds as a Gated DeltaNet one.
  vllm::Qwen4ExpLayerWeights& lw = w.layers[0];
  REQUIRE(lw.is_linear_attention);
  REQUIRE(!lw.gdn.in_proj_qkv.bytes.empty());

  const vllm::GdnLayerWeights gw = vllm::Qwen4ExpGdnBlockWeights(lw.gdn, p);

  // POINTER IDENTITY, PER TENSOR. A pair that compares EQUAL is a view; a pair
  // that differs is a second allocation the kernel will be pointed at and the
  // model will not keep alive. Listed one by one rather than folded, so a red
  // names the tensor.
  // `Addr` and not the raw `const uint8_t*`: doctest stringifies a `char`-like
  // pointer as a bool, so a raw comparison reds with the useless `1 == 1` and a
  // reader cannot tell a second allocation from an offset one.
  CHECK(Addr(gw.in_proj_qkv) == Addr(lw.gdn.in_proj_qkv));
  CHECK(Addr(gw.in_proj_z) == Addr(lw.gdn.in_proj_z));
  CHECK(Addr(gw.in_proj_b) == Addr(lw.gdn.in_proj_b));
  CHECK(Addr(gw.in_proj_a) == Addr(lw.gdn.in_proj_a));
  CHECK(Addr(gw.conv1d_weight) == Addr(lw.gdn.conv1d));
  CHECK(Addr(gw.a_log) == Addr(lw.gdn.a_log));
  CHECK(Addr(gw.dt_bias) == Addr(lw.gdn.dt_bias));
  CHECK(Addr(gw.norm_weight) == Addr(lw.gdn.norm_weight));
  CHECK(Addr(gw.out_proj) == Addr(lw.gdn.out_proj));

  // ...AND THE VIEW CARRIES ITS OWN KEEP-ALIVE, which is the half that makes
  // the identity SAFE rather than merely equal: a raw equal pointer with no
  // ownership is the dangling operand this case exists to stop.
  CHECK(gw.in_proj_qkv.bytes.borrowed());
  CHECK(gw.out_proj.bytes.borrowed());

  // The shape/orientation metadata a wrong-answer defect would ride on is
  // unchanged by the sharing.
  CHECK(gw.in_proj_qkv.nk == lw.gdn.in_proj_qkv.nk);
  CHECK(gw.in_proj_qkv.dtype == lw.gdn.in_proj_qkv.dtype);
  CHECK(gw.in_proj_qkv.bytes.size() == lw.gdn.in_proj_qkv.bytes.size());
  CHECK(gw.out_proj.bytes.size() == lw.gdn.out_proj.bytes.size());

  // AND THE BYTES OUTLIVE THE ADAPTER. Destroying the block's view must leave
  // the model's own buffer readable and unchanged — the exact ordering the CUDA
  // arm violated, stated where a host run can check it.
  const uint8_t* base = lw.gdn.in_proj_qkv.bytes.data();
  const size_t nb = lw.gdn.in_proj_qkv.bytes.size();
  std::vector<uint8_t> before(base, base + nb);
  {
    const vllm::GdnLayerWeights scoped =
        vllm::Qwen4ExpGdnBlockWeights(lw.gdn, p);
    CHECK(Addr(scoped.in_proj_qkv) == reinterpret_cast<uintptr_t>(base));
  }
  REQUIRE(lw.gdn.in_proj_qkv.bytes.data() == base);
  REQUIRE(lw.gdn.in_proj_qkv.bytes.size() == nb);
  CHECK(std::memcmp(lw.gdn.in_proj_qkv.bytes.data(), before.data(), nb) == 0);
}

// The same property, entered through a PRODUCTION entry point rather than the
// adapter, because a unit assertion on `Qwen4ExpGdnBlockWeights` proves the
// function shares and proves nothing about what the forward calls.
// `ModelRegistry::Forward` runs the layer loop over a real `qwen4exp` GGUF; the
// observable afterwards is that the loaded model's OWN Gated DeltaNet buffers
// have been shared with the step's adapter (`OwnedBytes::borrowed()`), which is
// only reachable through the sharing construction the loop performs. Before the
// fix the loop copied instead, the model's buffers stayed owned, and every step
// allocated and freed a duplicate underneath a queued kernel.
TEST_CASE(
    "qwen4_exp: a forward through ModelRegistry::Forward leaves the model's own "
    "GDN buffers shared, not duplicated") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  auto* loaded = dynamic_cast<vllm::Qwen4ExpLoadedModel*>(model.get());
  REQUIRE(loaded != nullptr);

  // The loader leaves them OWNED. Asserted, so the case cannot pass because the
  // load already shared them for some unrelated reason.
  {
    const vllm::Qwen4ExpWeights& before = loaded->weights();
    REQUIRE(before.layers[0].is_linear_attention);
    REQUIRE(!before.layers[0].gdn.in_proj_qkv.bytes.borrowed());
  }
  const uint8_t* base = loaded->weights().layers[0].gdn.in_proj_qkv.bytes.data();

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16, {2, T, kKvHeads, kHeadDim},
                              kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids, pos,     am, gm, attn_kv,
                             gdn, config,  q,  logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  vllm::ForwardLogits fl;
  REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  REQUIRE(fl.rows == 1);

  // THE OBSERVABLE. The forward's own adapter took a view of these bytes, which
  // `OwnedBytes::KeepAlive()` records ON THE SOURCE by turning it into a shared
  // read-only buffer. A forward that copied instead leaves this false.
  const vllm::Qwen4ExpWeights& after = loaded->weights();
  CHECK(after.layers[0].gdn.in_proj_qkv.bytes.borrowed());
  CHECK(after.layers[0].gdn.out_proj.bytes.borrowed());
  // ...and sharing did not MOVE the bytes: the address the first step handed a
  // kernel is the address every later step hands it.
  CHECK(Addr(after.layers[0].gdn.in_proj_qkv) ==
        reinterpret_cast<uintptr_t>(base));
}
// ─────────────────────────────────────────────────────────────────────────────
// MoE ADAPTER LIFETIME (`ISSUE-LOCAL-01M2AA9C31GCSDV8NRW26GMEVS`, spec
// `.agents/specs/qwen4-exp-flash-next.md` §"W6 scope: hoist the MoE adapter onto
// the model").
//
// WHAT FAILED, AND WHY TWELVE GREEN WAVES NEVER SAW IT. The layer loop composed
// `Qwen4ExpMoeBlockWeights` INSIDE the loop, so every layer of every step got a
// fresh set of handles. `BorrowWholeOwnedTensor` carries dtype, rank, shape,
// `nk` and every layout marker and does NOT carry `d_dev`, which is
// `ResidentWeight`'s residency memo, so every step's handles presented a null
// memo and every expert tower took the `Alloc` + `Copy` staging arm again. On
// `dgx:gpu0` that is 378 `cudaMalloc` calls per decode step — 48 layers x (3
// expert towers + 3 shared-expert weights + router + shared gate) = 384 — and
// 2.25 s of the step in the allocator against 0.101 s in all GPU kernels.
//
// THE TOKENS WERE ALWAYS CORRECT. Restaging a weight produces the same bytes on
// the device that the last staging produced, so the golden above, every
// per-block gate and every value comparison in this file pass in both arms. A
// value gate cannot see a residency lifetime, which is the same sentence #2476
// wrote next door about the Gated DeltaNet twin.
//
// WHY THIS CASE DOES NOT READ `load_stats::device_upload_bytes`. That counter is
// the direct expression of the defect and it is the one the operator's
// re-measurement reads — but `ResidentWeight` only reaches
// `load_stats::AddDeviceUpload` on its DEVICE arm, and on a CPU queue it takes
// the host-alias arm and counts nothing. On a CPU-only host the counter is
// therefore 0 in BOTH arms, and an assertion on it would pass whether or not the
// adapter is hoisted: an instrument that succeeds at the wrong question. So this
// case asserts the PRECONDITION of the counter instead, which is exactly what
// `d_dev` needs and what a CPU run can check: the handles the second step hands
// `ResidentWeight` are THE SAME OBJECTS the first step handed it. A memo written
// onto those handles survives; one written onto a temporary cannot.
//
// THE SENTINEL IS NOT DECORATION. Comparing addresses alone is not enough: a
// rebuilt adapter frees its predecessor's buffers first, and the allocator
// hands the same address straight back for a same-sized request, so two equal
// addresses are consistent with a rebuild. Writing a byte into the adapter
// between the two forwards and finding it afterwards is not — only an adapter
// that was never recomposed still carries it.
TEST_CASE(
    "qwen4_exp: a second forward through ModelRegistry::Forward REUSES the MoE "
    "adapter the first one built, so nothing is staged again") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  auto* loaded = dynamic_cast<vllm::Qwen4ExpLoadedModel*>(model.get());
  REQUIRE(loaded != nullptr);

  // The LOAD does not build it. Asserted, so a green cannot come from the
  // loader having populated the adapter for some unrelated reason: what this
  // case measures is what the FORWARD does with it.
  REQUIRE(!loaded->weights().layers[0].moe_block.has_value());

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids[static_cast<size_t>(t)] =
        static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16, {2, T, kKvHeads, kHeadDim},
                              kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids, pos,     am, gm, attn_kv,
                             gdn, config,  q,  logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  // ── STEP ONE ───────────────────────────────────────────────────────────────
  vllm::ForwardLogits fl;
  REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  REQUIRE(fl.rows == 1);

  // EVERY layer, because the MoE is on every layer of this architecture and a
  // hoist that reaches only the first one is the same defect on 47 of them.
  for (size_t il = 0; il < loaded->weights().layers.size(); ++il) {
    CAPTURE(il);
    REQUIRE(loaded->weights().layers[il].moe_block.has_value());
  }

  vllm::Qwen4ExpLayerWeights& lw = loaded->weights().layers[0];
  const vllm::MoeBlockWeights* adapter = &lw.moe_block.value();

  // THE BORROW, which is constraint (2) of the spec's three and the half that
  // makes holding the adapter SAFE rather than merely stable. The shared
  // expert's projections must be VIEWS of the model's own bytes: spelling them
  // as assignment deep-copies `OwnedTensor`'s owned buffer, which is what
  // #2476 was.
  CHECK(Addr(adapter->shared_gate_proj) == Addr(lw.moe.shared_gate_proj));
  CHECK(Addr(adapter->shared_up_proj) == Addr(lw.moe.shared_up_proj));
  CHECK(Addr(adapter->shared_down_proj) == Addr(lw.moe.shared_down_proj));
  CHECK(adapter->shared_gate_proj.bytes.borrowed());
  CHECK(adapter->shared_down_proj.bytes.borrowed());

  // The addresses the first step handed `ResidentWeight`, one per tensor the
  // memo would be written onto. Listed rather than folded, so a red names the
  // tensor that was rebuilt.
  const uintptr_t a_router = Addr(adapter->router_gate);
  const uintptr_t a_sgate = Addr(adapter->shared_gate);
  const uintptr_t a_sgp = Addr(adapter->shared_gate_proj);
  const uintptr_t a_sup = Addr(adapter->shared_up_proj);
  const uintptr_t a_sdn = Addr(adapter->shared_down_proj);

  // THE SENTINEL. `router_gate` is the adapter's OWN bf16 re-rounding of the
  // model's f32 router, so writing here touches nothing the model owns and
  // nothing a later load reads. A recomposed adapter overwrites this byte with
  // the conversion's output; the adapter that survives the step still has it.
  REQUIRE(!adapter->router_gate.bytes.empty());
  auto* sentinel_at =
      const_cast<uint8_t*>(adapter->router_gate.bytes.data());  // NOLINT
  const uint8_t sentinel_was = sentinel_at[0];
  const uint8_t kSentinel = static_cast<uint8_t>(sentinel_was ^ 0x5AU);
  sentinel_at[0] = kSentinel;

  // ── STEP TWO, the one the defect lives in ──────────────────────────────────
  REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  REQUIRE(fl.rows == 1);

  // (1) THE SAME OBJECT. Not a new `MoeBlockWeights` at the same address: the
  // optional still holds the one the first step emplaced.
  REQUIRE(loaded->weights().layers[0].moe_block.has_value());
  CHECK(&loaded->weights().layers[0].moe_block.value() == adapter);

  // (2) THE SAME BYTES, so the `d_dev` the first step's staging wrote is on the
  // handle the second step reads.
  CHECK(Addr(adapter->router_gate) == a_router);
  CHECK(Addr(adapter->shared_gate) == a_sgate);
  CHECK(Addr(adapter->shared_gate_proj) == a_sgp);
  CHECK(Addr(adapter->shared_up_proj) == a_sup);
  CHECK(Addr(adapter->shared_down_proj) == a_sdn);

  // (3) AND IT WAS NEVER RECOMPOSED, which is the assertion an equal address
  // cannot make on its own.
  CHECK(static_cast<unsigned>(adapter->router_gate.bytes.data()[0]) ==
        static_cast<unsigned>(kSentinel));

  // Restored, because a poisoned router is not a state any later assertion in
  // this binary should inherit.
  sentinel_at[0] = sentinel_was;
}

// ─────────────────────────────────────────────────────────────────────────────
// DECODEDIV (#2496) — WHERE THE CUDA ARM LEAVES THE CPU ARM, BY STEP AND BY
// TENSOR.
//
// The defect this case was written against is a whole-output symptom: on
// `thor:gpu0`, over the released `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S
// artifact, a CUDA server emits `11751 271 271 271 271 271 0 0` where the CPU
// arm on the same tree and the same file emits `11751 13 15767 411 2029 11 1092
// 369`. Token 0 agrees, so PREFILL is right; every decode token is wrong, and
// bit-stably so across builds, trees and `CUDA_LAUNCH_BLOCKING`. That excludes
// the whole async/ordering class and points at STATE the second step carries.
//
// A whole-output symptom cannot name a tensor. This case converts it into one:
// it drives the SAME two steps — one prefill, one decode — through
// `ModelRegistry::Forward` on a CPU queue and on a CUDA queue over the SAME
// fixture and the SAME token ids, and compares, in order:
//
//   1. the prefill logits,
//   2. every PERSISTENT buffer the prefill wrote and the decode reads — the GDN
//      conv ring, the GDN temporal state, the PLE conv ring, the PLE n-gram
//      history, the paged K/V and the indexer side cache,
//   3. the decode logits.
//
// The first row of that list that disagrees IS the answer, and the ordering is
// the point: a state buffer that differs after a prefill whose logits agree is
// exactly the shape "prefill right, decode wrong" describes. Every difference is
// PRINTED whether or not it trips an assertion, because a diagnostic that only
// says pass/fail cannot name a tensor either.
//
// WHY THE INPUTS ARE PINNED RATHER THAN SAMPLED. The second step's token is a
// CONSTANT here, not the first step's argmax. Sampling it per arm would feed the
// two arms different ids the moment the prefill logits disagree at all, and the
// decode comparison would then be measuring two different questions.
//
// WHAT IT CANNOT SEE. `qwen4_exp_gguf_fixture.h` is a miniature whose layer-3
// activations sit near 2^18, where one bf16 ULP is ~1024 and the K/V store
// saturates — W5j measured 0 of 192 paged K/V words moving across two different
// prompts. A CPU/CUDA difference small enough to be absorbed by that store is
// invisible here. So a GREEN result on this case is NOT a claim that the device
// arm decodes correctly at released width; it is the statement that the
// difference is not one this fixture can hold.
namespace {

struct DecodeDivArm {
  std::vector<float> logits1, logits2;
  // After the prefill, in the published order: [gdn_conv, temporal, ple_conv,
  // ngram] per linear layer, then the two paged pools.
  std::vector<std::vector<unsigned char>> gdn_conv, ssm, ple_conv;
  std::vector<std::vector<int64_t>> ngram;
  std::vector<uint16_t> kv, idx;
  // The same six after the decode.
  std::vector<std::vector<unsigned char>> gdn_conv2, ssm2, ple_conv2;
  std::vector<std::vector<int64_t>> ngram2;
  std::vector<uint16_t> kv2, idx2;
};

// The largest absolute difference between two f32 vectors, and the largest
// magnitude on the CPU side, so the caller can report a RELATIVE figure without
// dividing by a zero row.
void DecodeDivMaxAbs(const std::vector<float>& a, const std::vector<float>& b,
                     double* max_diff, double* max_mag) {
  *max_diff = 0.0;
  *max_mag = 0.0;
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    const double x = static_cast<double>(a[i]);
    const double y = static_cast<double>(b[i]);
    const double m = std::fabs(x) > std::fabs(y) ? std::fabs(x) : std::fabs(y);
    if (m > *max_mag) *max_mag = m;
    const double dd = std::fabs(x - y);
    if (dd > *max_diff) *max_diff = dd;
  }
}

// How many bytes of two raw buffers differ, and the index of the FIRST one. Raw
// rather than typed because these snapshots carry three different element
// widths and the question here is "did the two arms write the same bytes", not
// "how far apart are the values".
size_t DecodeDivBytesDiffer(const std::vector<unsigned char>& a,
                            const std::vector<unsigned char>& b, size_t* first) {
  *first = a.size();
  size_t n = 0;
  REQUIRE(a.size() == b.size());
  for (size_t i = 0; i < a.size(); ++i) {
    if (a[i] != b[i]) {
      if (n == 0) *first = i;
      ++n;
    }
  }
  return n;
}

// `device_next_id`, when non-null, is what the runner's asynchronous combine
// spliced into the DEVICE identifier buffer for the decode row, while `next_id`
// stays the STALE host value that combine deliberately never wrote back. Null on
// the plain two-arm comparison, where both arms read the host vector.
DecodeDivArm RunQwen4ExpDecodeDivArm(const vllm::GgufFile& g,
                                     const vllm::HfConfig& config,
                                     vt::DeviceType dev_type,
                                     const std::vector<int32_t>& prompt,
                                     int32_t next_id,
                                     const int32_t* device_next_id = nullptr) {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  std::unique_ptr<vllm::LoadedModel> model = LoadThroughRegistry(g, dev_type);
  REQUIRE(model != nullptr);

  constexpr int64_t kQsaLayer = 3;
  constexpr int64_t kGdnLayers = 3;
  constexpr int64_t kPage = 4;
  constexpr int64_t kCols = 3;
  constexpr int64_t kBlocks0 = 4;
  constexpr int64_t kBlocks2 = 5;
  const int64_t kIdxRows = kBlocks2 * kPage;
  const auto T1 = static_cast<int64_t>(prompt.size());

  const int64_t kConvDimL = 2 * kKeyDim + kValueDim;
  const int64_t kGdnConvLen = kConvKernel - 1;
  const int64_t kSsmRow = kNumVHeads * kLinHeadDim * kLinHeadDim;
  const int64_t kPleStateLen = (kConvKernel - 1) * kNgramSize;
  const int64_t kCtx = kNgramSize - 1;

  const std::vector<int32_t> bt0{2, 0, 3};
  const std::vector<int32_t> bt2{1, 3, 4};

  vt::Backend& backend = vt::GetBackend(dev_type);
  vt::Queue q = dev_type == vt::DeviceType::kCPU
                    ? vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}
                    : backend.CreateQueue();
  vllm::dense_attn::Dev d{backend, q};

  const size_t gdn_conv_bytes =
      static_cast<size_t>(kConvDimL * kGdnConvLen) * sizeof(float);
  const size_t ssm_bytes = static_cast<size_t>(kSsmRow) * sizeof(float);
  // QUALIFIED, and that qualification is the whole of a heap overflow this file
  // already paid for once. This translation unit has a SECOND `kStream` in scope
  // — the golden fixture's, 16 — and an unqualified use here resolved to it while
  // the `DBuf` below was built from `qwen4_exp_fixture::kStream`, 128. The
  // snapshot then downloaded 2304 bytes into a 288-byte destination: 2016 bytes
  // of heap corruption, which surfaced as `malloc(): unsorted double linked list
  // corrupted` inside an unrelated `operator new` three statements later.
  const int64_t kStreamWidth = qwen4_exp_fixture::kStream;
  const size_t ple_conv_bytes = static_cast<size_t>(kStreamWidth * kPleStateLen) *
                                vt::SizeOf(vllm::kQwen4ExpStreamDType);

  std::vector<std::vector<unsigned char>> host_gdn_conv(kGdnLayers),
      host_ssm(kGdnLayers), host_ple_conv(kGdnLayers);
  std::vector<std::vector<int64_t>> host_ngram(kGdnLayers);
  std::vector<vllm::dense_attn::DBuf> b_gdn_conv, b_ssm, b_ple_conv, b_ngram;
  std::vector<vllm::GdnStateCache> gdn(kGdnLayers);
  b_gdn_conv.reserve(kGdnLayers);
  b_ssm.reserve(kGdnLayers);
  b_ple_conv.reserve(kGdnLayers);
  b_ngram.reserve(kGdnLayers);
  for (int i = 0; i < kGdnLayers; ++i) {
    host_gdn_conv[i].assign(gdn_conv_bytes, 0);
    host_ssm[i].assign(ssm_bytes, 0);
    host_ple_conv[i].assign(ple_conv_bytes, 0);
    host_ngram[i].assign(static_cast<size_t>(kCtx), 0);
    b_gdn_conv.emplace_back(d, DType::kF32,
                            std::vector<int64_t>{1, kConvDimL, kGdnConvLen},
                            host_gdn_conv[i].data());
    b_ssm.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        host_ssm[i].data());
    b_ple_conv.emplace_back(
        d, vllm::kQwen4ExpStreamDType,
        std::vector<int64_t>{1, qwen4_exp_fixture::kStream, kPleStateLen},
        host_ple_conv[i].data());
    b_ngram.emplace_back(d, DType::kI64, std::vector<int64_t>{1, kCtx},
                         host_ngram[i].data());
    gdn[static_cast<size_t>(i)].conv_state = b_gdn_conv.back().t();
    gdn[static_cast<size_t>(i)].ssm_state = b_ssm.back().t();
    gdn[static_cast<size_t>(i)].states = {b_gdn_conv.back().t(), b_ssm.back().t(),
                                          b_ple_conv.back().t(), b_ngram.back().t()};
  }

  std::vector<uint16_t> kv_host(
      static_cast<size_t>(2 * kBlocks0 * kPage * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2 * kBlocks0, kPage, kKvHeads, kHeadDim},
                              kv_host.data());
  constexpr uint16_t kPoison = 0x3F80;
  std::vector<uint16_t> idx_host(static_cast<size_t>(kIdxRows * kIdxHeadDim),
                                 kPoison);
  vllm::dense_attn::DBuf idx_b(d, DType::kBF16, {kBlocks2, kPage, kIdxHeadDim},
                               idx_host.data());

  std::vector<vllm::PagedKvCache> attn_kv(2);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = kBlocks0;
  attn_kv[0].block_size = kPage;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;
  attn_kv[1].data = idx_b.t().data;
  attn_kv[1].dtype = DType::kBF16;
  attn_kv[1].num_blocks = kBlocks2;
  attn_kv[1].block_size = kPage;
  attn_kv[1].num_kv_heads = kIdxKvHeads;
  attn_kv[1].head_size = kIdxHeadDim;

  std::vector<std::string> names;
  std::vector<int32_t> group_ids, layer_indices, payload_slots;
  std::vector<uint8_t> payload_kinds;
  names.push_back("model.layers." + std::to_string(kQsaLayer) + ".self_attn.attn");
  group_ids.push_back(0);
  layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
  payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
  payload_slots.push_back(0);
  for (int i = 0; i < kGdnLayers; ++i) {
    names.push_back("model.layers." + std::to_string(i) + ".linear_attn");
    group_ids.push_back(1);
    layer_indices.push_back(i);
    payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kRecurrent));
    payload_slots.push_back(i);
  }
  names.push_back("model.layers." + std::to_string(kQsaLayer) +
                  ".self_attn.indexer.k_cache");
  group_ids.push_back(2);
  layer_indices.push_back(static_cast<int32_t>(kQsaLayer));
  payload_kinds.push_back(static_cast<uint8_t>(vllm::KvCachePayload::kPaged));
  payload_slots.push_back(1);

  const std::vector<int32_t> bt_recurrent{0};
  std::vector<std::vector<int32_t>> group_tables{bt0, bt_recurrent, bt2};
  std::vector<int32_t> group_cols{static_cast<int32_t>(kCols), 1,
                                  static_cast<int32_t>(kCols)};
  vllm::MultiKvCacheIndex mk;
  mk.layer_names = &names;
  mk.group_ids = &group_ids;
  mk.layer_indices = &layer_indices;
  mk.payload_kinds = &payload_kinds;
  mk.payload_slots = &payload_slots;
  mk.group_block_tables = &group_tables;
  mk.group_block_table_cols = &group_cols;

  const auto slot_of = [&](int64_t t) {
    return static_cast<int32_t>(bt0[static_cast<size_t>(t / kPage)] * kPage +
                                t % kPage);
  };

  // The device identifier buffer this step publishes, or null. Declared out here
  // so it outlives the `run` call that hands its address to the forward.
  vllm::dense_attn::DBuf dev_ids_buf;
  const int32_t* dev_ids_ptr = nullptr;

  const auto run = [&](const std::vector<int32_t>& tok_ids, int64_t past_len) {
    const auto T = static_cast<int64_t>(tok_ids.size());
    std::vector<int32_t> pos(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t)
      pos[static_cast<size_t>(t)] = static_cast<int32_t>(past_len + t);

    vllm::v1::CommonAttentionMetadata am;
    am.num_reqs = 1;
    am.num_actual_tokens = static_cast<int>(T);
    am.max_query_len = static_cast<int>(T);
    am.query_start_loc.assign({0, static_cast<int32_t>(T)});
    am.seq_lens.assign(1, static_cast<int32_t>(past_len + T));
    am.block_table_tensor = bt0;
    am.block_table_num_cols = static_cast<int>(kCols);
    am.max_seq_len = static_cast<int>(past_len + T);
    am.slot_mapping.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t)
      am.slot_mapping[static_cast<size_t>(t)] = slot_of(past_len + t);

    vllm::v1::GDNAttentionMetadata gm;
    gm.num_actual_tokens = static_cast<int>(T);
    gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
    gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
    if (past_len == 0) {
      gm.num_prefills = 1;
      gm.num_prefill_tokens = static_cast<int>(T);
      gm.has_initial_state = std::vector<uint8_t>{0};
      gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
      gm.prefill_state_indices = std::vector<int32_t>{0};
      gm.prefill_has_initial_state = std::vector<uint8_t>{0};
      const auto conv =
          vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
      gm.batch_ptr = conv.batch_ptr;
      gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
    } else {
      gm.num_decodes = 1;
      gm.num_decode_tokens = static_cast<int>(T);
    }

    const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
    vllm::ModelForwardInput in{tok_ids, pos,    am, gm, attn_kv,
                               gdn,     config, q,  logits_indices};
    in.num_reqs = 1;
    in.gdn_state_slots = 1;
    in.multi_kv = &mk;
    in.device_token_ids = dev_ids_ptr;

    vllm::ForwardLogits fl;
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
    REQUIRE(fl.on_device());
    REQUIRE(fl.rows == 1);
    std::vector<float> out(static_cast<size_t>(fl.rows * fl.vocab), 0.0F);
    d.b.Copy(q, out.data(), fl.device_tensor.data, out.size() * sizeof(float));
    d.b.Synchronize(q);
    return out;
  };

  DecodeDivArm r;
  r.gdn_conv.resize(kGdnLayers);
  r.ssm.resize(kGdnLayers);
  r.ple_conv.resize(kGdnLayers);
  r.ngram.resize(kGdnLayers);
  r.gdn_conv2.resize(kGdnLayers);
  r.ssm2.resize(kGdnLayers);
  r.ple_conv2.resize(kGdnLayers);
  r.ngram2.resize(kGdnLayers);

  const auto snapshot = [&](std::vector<std::vector<unsigned char>>& gc,
                            std::vector<std::vector<unsigned char>>& ss,
                            std::vector<std::vector<unsigned char>>& pc,
                            std::vector<std::vector<int64_t>>& ng,
                            std::vector<uint16_t>& kvo, std::vector<uint16_t>& idxo) {
    for (int i = 0; i < kGdnLayers; ++i) {
      gc[static_cast<size_t>(i)].assign(gdn_conv_bytes, 0);
      ss[static_cast<size_t>(i)].assign(ssm_bytes, 0);
      pc[static_cast<size_t>(i)].assign(ple_conv_bytes, 0);
      ng[static_cast<size_t>(i)].assign(static_cast<size_t>(kCtx), 0);
      b_gdn_conv[static_cast<size_t>(i)].Download(d, gc[static_cast<size_t>(i)].data());
      b_ssm[static_cast<size_t>(i)].Download(d, ss[static_cast<size_t>(i)].data());
      b_ple_conv[static_cast<size_t>(i)].Download(d, pc[static_cast<size_t>(i)].data());
      b_ngram[static_cast<size_t>(i)].Download(d, ng[static_cast<size_t>(i)].data());
    }
    kvo.assign(kv_host.size(), 0);
    idxo.assign(idx_host.size(), 0);
    kv_b.Download(d, kvo.data());
    idx_b.Download(d, idxo.data());
  };

  r.logits1 = run(prompt, /*past_len=*/0);
  snapshot(r.gdn_conv, r.ssm, r.ple_conv, r.ngram, r.kv, r.idx);
  // PUBLISHED ONLY FOR THE DECODE ROW, which is the only row the runner's
  // combine ever patches. The prefill above ran with a null pointer, so it is
  // byte-identical to the plain arm.
  std::vector<int32_t> dev_ids{device_next_id != nullptr ? *device_next_id : next_id};
  if (device_next_id != nullptr) {
    dev_ids_buf = vllm::dense_attn::DBuf(d, DType::kI32, {1}, dev_ids.data());
    dev_ids_ptr = static_cast<const int32_t*>(dev_ids_buf.ptr());
  }
  r.logits2 = run(std::vector<int32_t>{next_id}, /*past_len=*/T1);
  snapshot(r.gdn_conv2, r.ssm2, r.ple_conv2, r.ngram2, r.kv2, r.idx2);
  return r;
}

}  // namespace

TEST_CASE("qwen4_exp #2496: the CUDA decode step agrees with the CPU decode step") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  if (!LayerLoopHasCuda()) {
    MESSAGE("no CUDA backend in this build: #2496's CPU-vs-CUDA decode "
            "comparison is UNMEASURED by this run");
    return;
  }

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);

  // The prompt carries an EOS in the INTERIOR, which is a segment boundary in
  // the hashed n-gram construction, so the ids after it are not a straight ramp.
  const std::vector<int32_t> prompt{5, 9, 13, static_cast<int32_t>(kEosTokenId), 7, 2};
  const int32_t next_id = 11;  // PINNED; see the header for why it is not sampled.
  for (int32_t v : prompt) REQUIRE(v < static_cast<int32_t>(kVocab));
  REQUIRE(next_id < static_cast<int32_t>(kVocab));

  DecodeDivArm cpu = RunQwen4ExpDecodeDivArm(g, config, vt::DeviceType::kCPU,
                                             prompt, next_id);
  DecodeDivArm gpu = RunQwen4ExpDecodeDivArm(g, config, vt::DeviceType::kCUDA,
                                             prompt, next_id);

  for (float v : cpu.logits1) REQUIRE(std::isfinite(v));
  for (float v : cpu.logits2) REQUIRE(std::isfinite(v));
  for (float v : gpu.logits1) REQUIRE(std::isfinite(v));
  for (float v : gpu.logits2) REQUIRE(std::isfinite(v));

  double d1 = 0.0, m1 = 0.0, d2 = 0.0, m2 = 0.0;
  DecodeDivMaxAbs(cpu.logits1, gpu.logits1, &d1, &m1);
  DecodeDivMaxAbs(cpu.logits2, gpu.logits2, &d2, &m2);
  const double rel1 = m1 > 0.0 ? d1 / m1 : d1;
  const double rel2 = m2 > 0.0 ? d2 / m2 : d2;
  MESSAGE("#2496 STEP 1 (prefill) logits: max|cpu-cuda| = " << d1 << " over a max "
          "magnitude of " << m1 << " (relative " << rel1 << ")");
  MESSAGE("#2496 STEP 2 (decode)  logits: max|cpu-cuda| = " << d2 << " over a max "
          "magnitude of " << m2 << " (relative " << rel2 << ")");

  // EVERY PERSISTENT BUFFER, PRINTED WHETHER OR NOT IT TRIPS. This is the list
  // whose first disagreeing row names the defect; a diagnostic that only reports
  // the logits repeats the symptom instead of locating it.
  const char* kNames[4] = {"gdn_conv", "gdn_temporal", "ple_conv_ring", "ngram_history"};
  for (int i = 0; i < 3; ++i) {
    const std::vector<std::vector<unsigned char>>* pairs[3][2] = {
        {&cpu.gdn_conv, &gpu.gdn_conv},
        {&cpu.ssm, &gpu.ssm},
        {&cpu.ple_conv, &gpu.ple_conv}};
    for (int k = 0; k < 3; ++k) {
      size_t first = 0;
      const size_t n = DecodeDivBytesDiffer((*pairs[k][0])[static_cast<size_t>(i)],
                                            (*pairs[k][1])[static_cast<size_t>(i)],
                                            &first);
      MESSAGE("#2496 after PREFILL, linear layer " << i << " " << kNames[k]
              << ": " << n << " differing bytes, first at " << first);
    }
    MESSAGE("#2496 after PREFILL, linear layer " << i << " " << kNames[3]
            << ": cpu/cuda equal = "
            << (cpu.ngram[static_cast<size_t>(i)] == gpu.ngram[static_cast<size_t>(i)]));
    // THE HISTORY IS INT64 TOKEN IDS, so it is the one cross-step state whose
    // disagreement cannot be a rounding difference.
    CHECK(cpu.ngram[static_cast<size_t>(i)] == gpu.ngram[static_cast<size_t>(i)]);
    CHECK(cpu.ngram2[static_cast<size_t>(i)] == gpu.ngram2[static_cast<size_t>(i)]);
  }
  {
    std::vector<unsigned char> a(reinterpret_cast<const unsigned char*>(cpu.kv.data()),
                                 reinterpret_cast<const unsigned char*>(cpu.kv.data()) +
                                     cpu.kv.size() * sizeof(uint16_t));
    std::vector<unsigned char> b(reinterpret_cast<const unsigned char*>(gpu.kv.data()),
                                 reinterpret_cast<const unsigned char*>(gpu.kv.data()) +
                                     gpu.kv.size() * sizeof(uint16_t));
    size_t first = 0;
    MESSAGE("#2496 after PREFILL, paged K/V: " << DecodeDivBytesDiffer(a, b, &first)
            << " differing bytes, first at " << first);
  }
  {
    std::vector<unsigned char> a(reinterpret_cast<const unsigned char*>(cpu.idx.data()),
                                 reinterpret_cast<const unsigned char*>(cpu.idx.data()) +
                                     cpu.idx.size() * sizeof(uint16_t));
    std::vector<unsigned char> b(reinterpret_cast<const unsigned char*>(gpu.idx.data()),
                                 reinterpret_cast<const unsigned char*>(gpu.idx.data()) +
                                     gpu.idx.size() * sizeof(uint16_t));
    size_t first = 0;
    MESSAGE("#2496 after PREFILL, indexer side cache: "
            << DecodeDivBytesDiffer(a, b, &first) << " differing bytes, first at "
            << first);
  }

  // THE HEADLINE, AND IT IS A COMPARISON OF THE TWO STEPS RATHER THAN AN
  // ABSOLUTE BAND. #2496's whole shape is that the prefill agrees and the decode
  // does not, so the assertion that convicts it is that the decode's arm-to-arm
  // distance is not an order of magnitude worse than the prefill's. An absolute
  // tolerance would have to be picked against this fixture's own noise floor and
  // would say nothing about the defect.
  // THE ASSERTION IS THE RATIO, AND THE ABSOLUTE BAND THIS REPLACES WAS A
  // DIFFERENT CLAIM FROM THE ONE THE HEADER ARGUES FOR. It read
  // `rel1 <= 1e-3 && rel2 <= 1e-3`, which asserts that the two arms AGREE
  // NUMERICALLY -- and they do not, for reasons this case does not own:
  // [#2547](https://github.com/mudler/vllm.cpp/issues/2547) measures the released
  // artifact's PREFILL hidden state differing by about 0.3% between the arms,
  // before any decode state is read, and names
  // `vt::Qwen4ExpGatedResidual`'s CUDA arm — which routes its three projections
  // through the shared `vt::MatmulBT` and so re-associates the K reduction, as
  // `src/vt/cuda/cuda_qwen4_exp.cu` states in its own words — as the largest
  // non-bit-identical surface on the path. An absolute band here would red on
  // that, and reporting #2547 through this case would be reporting it in the
  // wrong place.
  //
  // What #2496 IS, and what this case therefore asserts, is that the DECODE step
  // is not an order of magnitude further apart than the PREFILL already is. The
  // stale-identifier defect made the decode arm-to-arm distance O(1) against a
  // prefill distance of ~1e-3, which this ratio convicts and a widened absolute
  // band would not. The floor keeps the ratio meaningful when `rel1` is at or
  // near zero.
  const double floor = 1.0e-3;
  const double allowed = 32.0 * (rel1 > floor ? rel1 : floor);
  MESSAGE("#2496 decode-vs-prefill arm distance ratio: " << (rel1 > 0.0 ? rel2 / rel1 : rel2)
          << " (decode " << rel2 << " against an allowance of " << allowed << ")");
  CHECK(rel2 <= allowed);
}

// ─────────────────────────────────────────────────────────────────────────────
// #2496 — THE FORWARD RUNS ON THE DEVICE IDENTIFIERS, NOT THE STALE HOST VECTOR.
//
// `ModelForwardInput::device_token_ids` states its own contract: "the input ids
// for this step are ALREADY on the device; the host vector is stale for decode
// rows". The asynchronous runner's combine splices each decode row's sampled
// token into that DEVICE buffer on the main queue and never writes it back,
// because materialising it on the host is the synchronise that path exists to
// remove. Nine registries consume the field under [#1305]; this architecture did
// not, so on `--device cuda` — where the mirror is the default — every decode
// row embedded, and hashed into the PLE n-gram context, whatever the host array
// happened to hold. For a row the host never wrote that is ZERO.
//
// MEASURED, before the fix, on `thor:gpu0` over the released
// `unsloth/Qwen3.8-Flash-Next-GGUF` UD-IQ1_S artifact with `VT_Q4EXP_STATE_FP=1`,
// the n-gram history after each step:
//
//     --device cpu    [9338, 369] -> [369, 11751] -> [11751, 13] -> [13, 15767] …
//     --device cuda   [9338, 369] -> [369,     0] -> [    0,  0] -> [ 0,     0] …
//
// The prefill agrees; from the first decode the model is fed token id 0 for ever.
//
// ─── WHY THIS CASE RUNS ON A CPU QUEUE AND STILL CONVICTS ───────────────────
// The defect is not arithmetic and not a device kernel: it is WHICH ARRAY the
// hook reads. So the case publishes a device identifier buffer that DISAGREES
// with the host vector and asks which one the answer came from — a question a
// CPU queue answers exactly, and one no GPU is needed to ask.
//
// THE PRECONDITION IS ASSERTED FIRST, and it is what stops this being a
// tautology: the two identifiers must actually produce different logits on this
// fixture. If `kFresh` and `kStale` decoded to the same row, the gate below
// would pass on a hook that read either array.
// THE NAME CARRIES NO COMMA, AND THAT IS LOAD-BEARING. `doctest`'s `-tc`
// filter SPLITS ITS ARGUMENT ON COMMAS, so a case whose name contains one can
// never be selected by name: the run reports `assertions: 0 | 0 passed | 0
// failed` at rc 0, which reads as a pass. This case was first written as
// "... DEVICE identifiers, not the stale host vector" and its red leg AND its
// green leg both came back as that empty success on `thor:gpu0` -- a mutation
// that measured nothing while reporting rc 0.
TEST_CASE("qwen4_exp #2496: the decode step reads the DEVICE identifiers rather than the stale host vector") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);

  const std::vector<int32_t> prompt{5, 9, 13, static_cast<int32_t>(kEosTokenId), 7, 2};
  // What the runner sampled and spliced into the device buffer.
  const int32_t kFresh = 11;
  // What a host row the combine never wrote holds. ZERO IS THE REAL VALUE, not a
  // stand-in: `token_ids_cpu` is zero-initialised and the mirror arm never writes
  // a decode row, which is why the released artifact hashed id 0 every step.
  const int32_t kStale = 0;
  REQUIRE(kFresh < static_cast<int32_t>(kVocab));

  // THE PLE LAYER'S RANK AMONG THE LINEAR LAYERS, which is the index the arm
  // helper snapshots under — never the decoder layer index.
  constexpr size_t kPleRank = 1;

  const DecodeDivArm fresh =
      RunQwen4ExpDecodeDivArm(g, config, vt::DeviceType::kCPU, prompt, kFresh);
  const DecodeDivArm stale =
      RunQwen4ExpDecodeDivArm(g, config, vt::DeviceType::kCPU, prompt, kStale);
  const DecodeDivArm spliced = RunQwen4ExpDecodeDivArm(
      g, config, vt::DeviceType::kCPU, prompt, kStale, &kFresh);

  REQUIRE(fresh.logits2.size() == stale.logits2.size());
  REQUIRE(fresh.logits2.size() == spliced.logits2.size());
  const size_t bytes = fresh.logits2.size() * sizeof(float);
  for (float v : fresh.logits2) REQUIRE(std::isfinite(v));
  for (float v : spliced.logits2) REQUIRE(std::isfinite(v));

  // THE PRECONDITION. Two different identifiers must decode to two different
  // rows, or nothing below is being measured.
  REQUIRE(std::memcmp(fresh.logits2.data(), stale.logits2.data(), bytes) != 0);

  // THE GATE. The published DEVICE identifier decides the answer, BIT FOR BIT:
  // one queue, one order, one arithmetic, so the only thing that can differ is
  // which array was read.
  CHECK(std::memcmp(spliced.logits2.data(), fresh.logits2.data(), bytes) == 0);
  CHECK(std::memcmp(spliced.logits2.data(), stale.logits2.data(), bytes) != 0);

  // AND THE N-GRAM HISTORY CARRIES IT TOO, which the logits alone do not prove:
  // the history is what the NEXT step reads, and it is int64 token ids, so its
  // disagreement can never be a rounding difference. This is the observable the
  // released-artifact measurement above was taken on.
  REQUIRE(spliced.ngram2.size() > kPleRank);
  REQUIRE(!spliced.ngram2[kPleRank].empty());
  CHECK(spliced.ngram2[kPleRank].back() == static_cast<int64_t>(kFresh));
  CHECK(stale.ngram2[kPleRank].back() == static_cast<int64_t>(kStale));
}

// ─── THE QSA QUERY BUFFER'S WIDTH, AT THE PRODUCTION ENTRY POINT (#2488) ─────
// vLLM keeps the query at `model_config.dtype` from the fused q/gate GEMM to
// `q_norm`: `torch.chunk` takes two views and does not widen
// (`vllm/model_executor/models/qwen3_next.py:430`), and the norm's `.float()`
// lives inside its own kernel (`vllm/models/qwen4_exp/nvidia/ple_layer.py:70`
// then `:80` `.to(input_dtype)`). Both citations are FORWARD REFERENCES to vLLM
// `origin/main` `cdefd9d499`, 1566 commits past this row's pin, which has no
// `qwen4_exp` at all.
//
// WHY THIS CANNOT BE A VALUE TEST. `AGENTS.md`: "A token gate cannot detect a
// dtype that is too wide. The tokens still match, and the goldens still pass,
// although the path moves twice the bytes." Every other case in this file reads
// logits, and not one of them could see an f32 `q`. So this case reads the
// DTYPE and the BYTE COUNT of the buffer the block actually allocates.
//
// HOW IT SEES A BUFFER INSIDE `QsaBlockCore`. A pass-through provider is
// installed for `kAttnGateSplit` on kCPU above the `vt-native` priority. It
// records `q_out` and forwards to the native kernel, captured BEFORE the
// registration so the forward cannot recurse into the spy. Nothing else changes:
// the same kernel runs on the same tensors, so this instrument cannot alter the
// result it is measuring.
//
// WHY IT IS A REACHABILITY GATE AND NOT A CLASS TEST. It enters through
// `vllm::ModelRegistry::Forward`, and the count it asserts is a COUNTED PROPERTY:
// deleting the `RunQwen4ExpQsaBlockPaged` call in `qwen4_exp_forward.cpp` takes
// `calls` from 1 to 0 and reds this case. A test that constructed the block by
// hand would stay green through that deletion.
namespace qsa_q_width {

struct Record {
  int calls = 0;
  bool every_q_is_bf16 = true;
  int64_t last_bytes = -1;
  int64_t last_numel = -1;
};

Record& Rec() {
  static Record r;
  return r;
}

vt::AttnGateSplitFn& Native() {
  static vt::AttnGateSplitFn f = nullptr;
  return f;
}

void SpySplit(vt::Queue& q, vt::Tensor& q_out, vt::Tensor& gate_out, const vt::Tensor& qgate) {
  Record& r = Rec();
  ++r.calls;
  r.every_q_is_bf16 = r.every_q_is_bf16 && q_out.dtype == DType::kBF16;
  r.last_numel = q_out.Numel();
  r.last_bytes = q_out.Numel() * static_cast<int64_t>(vt::SizeOf(q_out.dtype));
  Native()(q, q_out, gate_out, qgate);
}

// Idempotent: doctest may enter this file's cases in any order, and a second
// registration under the same name is dropped by `RegisterOpProvider` anyway.
void InstallOnce() {
  static bool installed = false;
  if (installed) return;
  Native() = reinterpret_cast<vt::AttnGateSplitFn>(
      vt::GetOp(vt::OpId::kAttnGateSplit, vt::DeviceType::kCPU));
  vt::OpProvider p;
  p.name = "qsa-q-width-spy";
  p.priority = 1;  // above kNativeProviderName's 0, so Choose() takes it
  p.supports = nullptr;
  p.fn = reinterpret_cast<void*>(static_cast<vt::AttnGateSplitFn>(&SpySplit));
  vt::RegisterOpProvider(vt::OpId::kAttnGateSplit, vt::DeviceType::kCPU, p);
  installed = true;
}

}  // namespace qsa_q_width

TEST_CASE("qwen4_exp: ModelRegistry::Forward splits the QSA query at the MODEL dtype") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  // THE INSTRUMENT'S OWN PRECONDITION, before anything it measures. A null
  // native pointer would make the spy forward into nothing.
  qsa_q_width::InstallOnce();
  REQUIRE(qsa_q_width::Native() != nullptr);
  qsa_q_width::Rec() = qsa_q_width::Record{};
  REQUIRE(qsa_q_width::Rec().calls == 0);

  const gguf_test::TempFile f(BuildFixture(FixtureOpts{}));
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids[static_cast<size_t>(t)] = static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv = vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(d, DType::kF32,
                       std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
                       ssm[i].data());
    conv_b.emplace_back(d, DType::kF32, std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16, {2, T, kKvHeads, kHeadDim}, kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids, pos, am, gm, attn_kv, gdn, config, q, logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  vllm::ForwardLogits fl;
  REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  REQUIRE(fl.rows == 1);

  // 1. THE INSTRUMENT RAN. Without this the three assertions below are vacuous
  //    on a forward that never reached the QSA layer, and a `[SKIP]` would wear
  //    a PASS. This is also the count the reachability mutation drives to zero.
  const qsa_q_width::Record r = qsa_q_width::Rec();
  INFO("kAttnGateSplit calls seen through ModelRegistry::Forward: ", r.calls);
  REQUIRE(r.calls > 0);

  // 2. THE SHAPE the block asked for is the model's query, not something else.
  CHECK(r.last_numel == T * kQHeads * kHeadDim);

  // 3. THE DTYPE and 4. THE BYTES. `T*kQHeads*kHeadDim*2` = 128 here. Before
  //    #2488 this buffer was `DType::kF32` and this number was 256 — the same
  //    tokens, twice the traffic, which is the whole reason the assertion is on
  //    the width and not on the output.
  CHECK(r.every_q_is_bf16);
  CHECK(r.last_bytes == T * kQHeads * kHeadDim * 2);
}

// ─────────────────────────────────────────────────────────────────────────────
// THE LAYER FINGERPRINT (issue #2547, wave PREFILLDIV)
//
// #2547 measured that a CUDA prefill of this architecture disagrees with a CPU
// prefill at step 0, by about 0.3% on `sum(|x|)` over the model's own output,
// and could not say WHERE: `VT_Q4EXP_STATE_FP` prints that output once per step
// and has no layer axis. `VT_Q4EXP_LAYER_FP` adds one, and this case is what
// says the axis exists and is reached from the production entry point.
//
// WHAT IT ASSERTS IS A COUNT, not the presence of a string in a stream. The tap
// total for this fixture is derivable and exact: 2 taps outside the loop
// (`emb`, `wide`), 9 per decoder layer, 2 more on the one PLE layer, and 1 for
// the model output. `kLayers` is 4 and `kPleLayer` is 1, so 2 + 4*9 + 2 + 1 =
// 41. A count is the assertion that survives a redirect that did not happen and
// a grep that matched nothing, both of which read as a pass.
//
// AND IT IS A REACHABILITY CASE. `ModelRegistry::Forward` is the production
// entry point; nothing here constructs `Qwen4ExpTextModelForward` by hand. The
// paired mutation is to delete any one `LayerFp(` call site in the forward: the
// total drops below 41 and this case reds.
TEST_CASE("qwen4_exp: VT_Q4EXP_LAYER_FP taps every layer of ModelRegistry::Forward") {
  using namespace qwen4_exp_fixture;  // NOLINT(build/namespaces)

  const gguf_test::TempFile f(BuildFixture());
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  const vllm::HfConfig config = vllm::Qwen4ExpHfConfigFromGguf(g);
  std::unique_ptr<vllm::LoadedModel> model;
  REQUIRE_NOTHROW(model = LoadThroughRegistry(g));
  REQUIRE(model != nullptr);

  const int64_t T = 4;
  std::vector<int32_t> ids(static_cast<size_t>(T));
  std::vector<int32_t> pos(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t) {
    ids[static_cast<size_t>(t)] = static_cast<int32_t>(t == T - 1 ? kEosTokenId : t + 1);
    pos[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  vllm::v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.block_table_num_cols = 1;
  am.block_table_tensor.assign(1, 0);
  am.seq_lens.assign(1, static_cast<int32_t>(T));
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t t = 0; t < T; ++t)
    am.slot_mapping[static_cast<size_t>(t)] = static_cast<int32_t>(t);

  vllm::v1::GDNAttentionMetadata gm;
  gm.num_prefills = 1;
  gm.num_prefill_tokens = static_cast<int>(T);
  gm.num_actual_tokens = static_cast<int>(T);
  gm.has_initial_state = std::vector<uint8_t>{0};
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_query_start_loc = std::vector<int32_t>{0, static_cast<int32_t>(T)};
  gm.prefill_state_indices = std::vector<int32_t>{0};
  gm.prefill_has_initial_state = std::vector<uint8_t>{0};
  {
    const auto conv =
        vllm::v1::ComputeCausalConv1dMetadata(*gm.non_spec_query_start_loc);
    gm.batch_ptr = conv.batch_ptr;
    gm.token_chunk_offset_ptr = conv.token_chunk_offset_ptr;
  }

  vt::Queue q = CpuQ();
  vllm::dense_attn::Dev d{vt::GetBackend(q.device.type), q};

  const int64_t key_dim = kNumKHeads * kLinHeadDim;
  const int64_t value_dim = kNumVHeads * kLinHeadDim;
  const int64_t conv_dim = 2 * key_dim + value_dim;
  const int64_t conv_len = kConvKernel - 1;
  const int64_t ssm_row = kNumVHeads * kLinHeadDim * kLinHeadDim;

  std::vector<std::vector<float>> ssm(3), conv(3);
  std::vector<vllm::dense_attn::DBuf> ssm_b, conv_b;
  std::vector<vllm::GdnStateCache> gdn(3);
  ssm_b.reserve(3);
  conv_b.reserve(3);
  for (int i = 0; i < 3; ++i) {
    ssm[i].assign(static_cast<size_t>(ssm_row), 0.0F);
    conv[i].assign(static_cast<size_t>(conv_dim * conv_len), 0.0F);
    ssm_b.emplace_back(
        d, DType::kF32,
        std::vector<int64_t>{1, kNumVHeads, kLinHeadDim, kLinHeadDim},
        ssm[i].data());
    conv_b.emplace_back(d, DType::kF32,
                        std::vector<int64_t>{1, conv_dim, conv_len},
                        conv[i].data());
    gdn[static_cast<size_t>(i)].ssm_state = ssm_b.back().t();
    gdn[static_cast<size_t>(i)].conv_state = conv_b.back().t();
  }

  std::vector<uint16_t> kv(static_cast<size_t>(2 * T * kKvHeads * kHeadDim), 0);
  vllm::dense_attn::DBuf kv_b(d, DType::kBF16,
                              {2, T, kKvHeads, kHeadDim}, kv.data());
  std::vector<vllm::PagedKvCache> attn_kv(1);
  attn_kv[0].data = kv_b.t().data;
  attn_kv[0].dtype = DType::kBF16;
  attn_kv[0].num_blocks = 1;
  attn_kv[0].block_size = T;
  attn_kv[0].num_kv_heads = kKvHeads;
  attn_kv[0].head_size = kHeadDim;

  const std::vector<int32_t> logits_indices{static_cast<int32_t>(T - 1)};
  vllm::ModelForwardInput in{ids, pos, am, gm, attn_kv, gdn, config, q,
                             logits_indices};
  in.num_reqs = 1;
  in.gdn_state_slots = 1;

  // 2 outside the loop + 9 per layer + 2 on the one PLE layer + 1 output.
  const int64_t kExpectedTaps = 2 + kLayers * 9 + 2 + 1;

  // ─── OFF IS THE RED. The switch unset must print NOTHING, because an
  // instrument that runs whether or not it was asked for is a permanent cost on
  // every forward and its `taps=` line stops being evidence that this run set
  // the variable.
  ::unsetenv("VT_Q4EXP_LAYER_FP");
  vllm::detail::Qwen4ExpLayerFpResetForTest();
  {
    vllm::ForwardLogits fl;
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  }
  CHECK(vllm::detail::Qwen4ExpLayerFpTapsForTest() == 0);

  // ─── ON, budget 1. One forward's worth of taps and not two.
  ::setenv("VT_Q4EXP_LAYER_FP", "1", 1);
  vllm::detail::Qwen4ExpLayerFpResetForTest();
  {
    vllm::ForwardLogits fl;
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  }
  const int64_t after_one = vllm::detail::Qwen4ExpLayerFpTapsForTest();
  INFO("taps after one forward: ", after_one, " expected ", kExpectedTaps);
  CHECK(after_one == kExpectedTaps);

  // THE BUDGET IS A BUDGET. A second forward under a budget of 1 adds nothing,
  // which is what keeps a decode loop from printing 41 lines a step forever.
  {
    vllm::ForwardLogits fl;
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  }
  CHECK(vllm::detail::Qwen4ExpLayerFpTapsForTest() == after_one);

  // Leave the process as this case found it: another case in this binary that
  // runs a forward must not inherit an armed instrument.
  ::unsetenv("VT_Q4EXP_LAYER_FP");
  vllm::detail::Qwen4ExpLayerFpResetForTest();
  CHECK(vllm::detail::Qwen4ExpLayerFpTapsForTest() == 0);

  // ─── VT_Q4EXP_LOGITS: the same three properties for the logits tap, plus one
  // CONTENT property that the layer tap cannot offer. `LayerFp` stops at `out`,
  // so without this tap the `lm_head` decode - int5, group 64 - has no
  // observation at all, and the acceptance gate this row designs cannot see it.
  //
  // OFF IS THE RED, again: the variable unset must print nothing and count
  // nothing, or the line stops being evidence that a run armed it.
  ::unsetenv("VT_Q4EXP_LOGITS");
  ::unsetenv("VT_Q4EXP_LOGITS_STEPS");
  vllm::detail::Qwen4ExpLogitsTapResetForTest();
  {
    vllm::ForwardLogits fl;
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  }
  CHECK(vllm::detail::Qwen4ExpLogitsTapTapsForTest() == 0);
  CHECK(vllm::detail::Qwen4ExpLogitsTapLastTop1ForTest() == -1);

  // ON, one step, top 4. One forward taps once - the tap is per step, not per
  // layer, so a correct count is 1 and not kLayers.
  ::setenv("VT_Q4EXP_LOGITS", "4", 1);
  ::setenv("VT_Q4EXP_LOGITS_STEPS", "1", 1);
  vllm::detail::Qwen4ExpLogitsTapResetForTest();
  vllm::ForwardLogits tapped;
  REQUIRE_NOTHROW(tapped = vllm::ModelRegistry::Forward(*model, in));
  CHECK(vllm::detail::Qwen4ExpLogitsTapTapsForTest() == 1);

  // CONTENT: the tap's reported top-1 must be the argmax of the very tensor the
  // forward returned, read independently here. This is the property that makes
  // the line usable as a gate: a tap that printed a plausible id from the wrong
  // row, or a stale one from the previous step, fails it.
  {
    REQUIRE(tapped.device_tensor.dtype == DType::kF32);
    REQUIRE(tapped.device_tensor.rank == 2);
    const int64_t vocab = tapped.device_tensor.shape[1];
    std::vector<float> row(static_cast<size_t>(vocab));
    std::memcpy(row.data(), tapped.device_tensor.data, row.size() * sizeof(float));
    int64_t argmax = 0;
    for (int64_t i = 1; i < vocab; ++i) {
      if (row[static_cast<size_t>(i)] > row[static_cast<size_t>(argmax)]) argmax = i;
    }
    INFO("taps ", vllm::detail::Qwen4ExpLogitsTapTapsForTest(), " top1 ",
         vllm::detail::Qwen4ExpLogitsTapLastTop1ForTest(), " argmax ", argmax);
    CHECK(vllm::detail::Qwen4ExpLogitsTapLastTop1ForTest() == argmax);
  }

  // THE BUDGET IS A BUDGET: a second forward under a budget of 1 adds nothing.
  {
    vllm::ForwardLogits fl;
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  }
  CHECK(vllm::detail::Qwen4ExpLogitsTapTapsForTest() == 1);

  // ...and the step counter only advances on a tap that PRINTED, so re-arming
  // with a fresh budget resumes instead of staying stuck.
  ::setenv("VT_Q4EXP_LOGITS_STEPS", "2", 1);
  vllm::detail::Qwen4ExpLogitsTapResetForTest();
  for (int step = 0; step < 3; ++step) {
    vllm::ForwardLogits fl;
    REQUIRE_NOTHROW(fl = vllm::ModelRegistry::Forward(*model, in));
  }
  CHECK(vllm::detail::Qwen4ExpLogitsTapTapsForTest() == 2);

  // ─── THE INSTRUMENT MUST BE AN OBSERVER. `Qwen4ExpLogitsTap` copies the whole
  // logits buffer to the host and synchronises the queue MID-FORWARD, which is
  // exactly the shape of instrument that can perturb the thing it measures: a
  // copy through the wrong queue, a missing sync, or a write-back bug would move
  // the numbers the gate then compares. A gate whose instrument changes its own
  // measurement is worth nothing, so the same fixed-seed forward must produce
  // BIT-IDENTICAL logits armed and unarmed.
  ::unsetenv("VT_Q4EXP_LOGITS");
  ::unsetenv("VT_Q4EXP_LOGITS_STEPS");
  vllm::detail::Qwen4ExpLogitsTapResetForTest();
  vllm::ForwardLogits unarmed;
  REQUIRE_NOTHROW(unarmed = vllm::ModelRegistry::Forward(*model, in));
  CHECK(vllm::detail::Qwen4ExpLogitsTapTapsForTest() == 0);

  ::setenv("VT_Q4EXP_LOGITS", "4", 1);
  ::setenv("VT_Q4EXP_LOGITS_STEPS", "1", 1);
  vllm::detail::Qwen4ExpLogitsTapResetForTest();
  vllm::ForwardLogits armed;
  REQUIRE_NOTHROW(armed = vllm::ModelRegistry::Forward(*model, in));
  // The tap actually ran, so the comparison below is not vacuous: an unarmed
  // "armed" arm would pass this check trivially and prove nothing.
  CHECK(vllm::detail::Qwen4ExpLogitsTapTapsForTest() == 1);

  {
    REQUIRE(unarmed.device_tensor.dtype == armed.device_tensor.dtype);
    REQUIRE(unarmed.device_tensor.rank == armed.device_tensor.rank);
    size_t elems = 1;
    for (int r = 0; r < unarmed.device_tensor.rank; ++r) {
      REQUIRE(unarmed.device_tensor.shape[r] == armed.device_tensor.shape[r]);
      elems *= static_cast<size_t>(unarmed.device_tensor.shape[r]);
    }
    REQUIRE(elems > 0);
    const auto* a = static_cast<const unsigned char*>(unarmed.device_tensor.data);
    const auto* b = static_cast<const unsigned char*>(armed.device_tensor.data);
    const size_t bytes = elems * vt::SizeOf(unarmed.device_tensor.dtype);
    // Byte equality, not an epsilon: the two arms are the SAME binary running the
    // SAME kernels on the SAME inputs, so the only permitted difference is none.
    CHECK(std::memcmp(a, b, bytes) == 0);
  }

  ::unsetenv("VT_Q4EXP_LOGITS");
  ::unsetenv("VT_Q4EXP_LOGITS_STEPS");
  vllm::detail::Qwen4ExpLogitsTapResetForTest();
  CHECK(vllm::detail::Qwen4ExpLogitsTapTapsForTest() == 0);
}
