// Qwen4-Exp W5f — `Qwen4ExpTextModel::Forward`. See qwen4_exp_forward.h for the
// oracle anchors, the four ordering facts a block gate cannot see, and the scope.
#include "vllm/model_executor/models/qwen4_exp_forward.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"     // ResidentWeight
#include "vllm/model_executor/models/qwen3_5_gdn_block.h"    // RunGdnBlockPaged
#include "vllm/model_executor/models/qwen3_5_mrope.h"        // BuildMropeCosSinHost
#include "vllm/model_executor/models/qwen4_exp_moe.h"        // RunQwen4ExpMoeBlock
#include "vllm/model_executor/moe_placement_seam.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {

using dense_attn::DBuf;
using dense_attn::Dev;
using vt::DType;
using vt::Tensor;

namespace {

// The whole tensor under a different shape, same bytes. Same helper the PLE and
// QSA blocks carry, and for the same reason: the element count must agree, which
// is the check that stops a reshape from quietly renaming a stride.
Tensor Reshape(const Tensor& t, const std::vector<int64_t>& shape) {
  int64_t have = 1;
  for (int i = 0; i < t.rank; ++i) have *= t.shape[i];
  int64_t want = 1;
  for (int64_t s : shape) want *= s;
  VT_CHECK(t.IsContiguous(),
           "qwen4_exp forward: Reshape needs a contiguous tensor");
  VT_CHECK(have == want,
           "qwen4_exp forward: Reshape would change the element count");
  return dense_attn::MakeTensor(t.data, t.dtype, t.device, shape);
}

void CheckOwned(const OwnedTensor& t, const char* name,
                const std::vector<int64_t>& shape) {
  VT_CHECK(t.rank == static_cast<int>(shape.size()),
           std::string("qwen4_exp forward: ") + name + " has rank " +
               std::to_string(t.rank) + ", expected " +
               std::to_string(shape.size()));
  for (size_t i = 0; i < shape.size(); ++i) {
    VT_CHECK(t.shape[i] == shape[i],
             std::string("qwen4_exp forward: ") + name + " dim " +
                 std::to_string(i) + " is " + std::to_string(t.shape[i]) +
                 ", expected " + std::to_string(shape[i]));
  }
}

// ─── VT_Q4EXP_LAYER_FP — the per-layer, per-tap arm fingerprint (#2547) ──────
//
// `VT_Q4EXP_STATE_FP` (#2496) prints the layer loop's OUTPUT once per step. It
// has no LAYER axis, so it can say that a CUDA prefill disagrees with a CPU one
// — measured at 0.3% on `sumabs` at step 0 — and it cannot say WHERE. This
// prints one line per tap per layer for the first `VT_Q4EXP_LAYER_FP` forward
// calls, which is the axis that converts that aggregate into one op.
//
// THE DTYPE IS PART OF THE LINE, and that is not decoration. A token gate cannot
// see a dtype and a value gate cannot see a lifetime; this row has paid for both
// (#2493, #2476). A tap that reported values alone would pass a buffer that is
// f32 on one arm and bf16 on the other while moving twice the bytes.
//
// IT COUNTS ITS OWN TAPS, and prints the count at the end of every fingerprinted
// step. A grep that matches nothing is not evidence of absence, and an
// instrument that never ran reads as an arm whose taps agreed. `taps=` is the
// counted property that separates the two.
//
// IT DRAINS THE QUEUE AT EVERY TAP, so it CANNOT see a race. That is admissible
// here only because #2547 measured this divergence as bit-stable across builds,
// across trees and across `CUDA_LAUNCH_BLOCKING`, which is a deterministic
// arithmetic difference. A symptom that appears only WITHOUT this instrument is
// a race and belongs to a different method.
// The instrument's whole mutable state, in one object so a test can put it back
// (`detail::Qwen4ExpLayerFpResetForTest`). A `static const bool` read once at
// first use would be cheaper by one branch and untestable: every case in a
// doctest binary shares the process, so the budget would be spent by whichever
// forward ran first and the case that sets the variable would measure nothing
// and pass. The tree already carries this shape for the same reason --
// `detail::ExpertStreamSetForceFallback` in qwen3_5.cpp.
struct LayerFpState {
  int64_t budget = 0;  // forward calls to fingerprint; 0 == the instrument is off
  int64_t step = 0;    // the forward call this process is on
  int64_t taps = 0;    // lines printed, the counted property
};

int64_t LayerFpBudgetFromEnv() {
  const char* e = std::getenv("VT_Q4EXP_LAYER_FP");
  if (e == nullptr || e[0] == '\0') return 0;
  const long long parsed = std::atoll(e);
  return parsed > 0 ? static_cast<int64_t>(parsed) : 0;
}

LayerFpState& LayerFpS() {
  static LayerFpState s = [] {
    LayerFpState t;
    t.budget = LayerFpBudgetFromEnv();
    return t;
  }();
  return s;
}

// ─── Fixed-seed random projections (#2877) ──────────────────────────────
//
// `sumabs` is sign-INSENSITIVE: it sums |x_i|, so a zero-mean perturbation
// (every reassociation, every rounding difference) cancels at O(sqrt(n)),
// and `rel_sumabs` under-reports the true divergence by a median 75-140x
// (measured over 400 seeds in test_q4exp_layerfp_diff.py::MetricSpread).
//
// A fixed-seed random projection `S w_i x_i` is a LINEAR FUNCTIONAL of the
// tensor: its difference between two arms is `S w_i (a_i - b_i)`, which
// cannot cancel because the signs of `w_i` are fixed and independent of the
// perturbation. Eight projections give 8 independent estimates of `||a-b||`,
// enough to see a divergence that any single projection might miss by
// landing near a zero of `w . (a-b)`.
//
// The PRNG is a 64-bit LCG with fixed constants. It is seeded from the tap's
// element count `n`, so two arms tapping the same tensor at the same `n` draw
// the same `w_i` and the projection difference is meaningful. No env var, no
// per-process state: the same tap on any run draws the same weights.
struct ProjRng {
  uint64_t s;
  explicit ProjRng(uint64_t seed) : s(seed * 6364136223846793005ULL + 1442695040888963407ULL) {}
  float next() {
    s = s * 6364136223846793005ULL + 1442695040888963407ULL;
    // Map to [-1, 1) via xorshift high bits. Using the high 24 bits gives a
    // uniform float in [-1, 1); the sign is the bit that makes the projection
    // sign-sensitive.
    uint32_t u = static_cast<uint32_t>(s >> 40);
    return (static_cast<float>(u) / static_cast<float>(1u << 24) - 0.5f) * 2.0f;
  }
};

// One tap. `il` is the decoder layer, or -1 for a tap outside the loop.
//
// THE COPY GOES THROUGH `Backend::Copy`, which is `cudaMemcpyDefault` on a CUDA
// queue and a `memcpy` on a CPU one, so ONE spelling reads both arms and neither
// arm gets a private readback path the other does not have. `Synchronize` after
// it is what makes the bytes the ones this tap names.
//
// THE EIGHT RANDOM PROJECTIONS (#2877). `sumabs` sums |x_i|, which is
// sign-INSENSITIVE: a zero-mean perturbation cancels at O(sqrt(n)), and
// `rel_sumabs` under-reports the true divergence by a median 75-140x (measured
// over 400 seeds in test_q4exp_layerfp_diff.py::MetricSpread). Each `proj_k` is
// `S w_i x_i` over the whole tensor, with fixed-seed `w_i` (ProjRng seeded
// from `n`). The difference `proj_k(a) - proj_k(b) = S w_i (a_i - b_i)` is a
// LINEAR FUNCTIONAL of the difference and CANNOT cancel. Eight independent
// projections give 8 estimates of `||a-b||`, enough to see a divergence that
// any single one might miss by landing near a zero of `w . (a-b)`.
void LayerFp(Dev d, int64_t il, const char* tag, const Tensor& t) {
  LayerFpState& s = LayerFpS();
  if (s.step >= s.budget) return;
  VT_CHECK(t.IsContiguous(),
           "qwen4_exp forward: VT_Q4EXP_LAYER_FP taps a contiguous tensor");
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) n *= t.shape[i];
  if (n <= 0) return;
  const size_t bytes = static_cast<size_t>(n) * vt::SizeOf(t.dtype);
  std::vector<uint8_t> host(bytes);
  d.b.Copy(d.q, host.data(), t.data, bytes);
  d.b.Synchronize(d.q);

  double sumabs = 0.0;
  double maxabs = 0.0;
  int64_t nonfinite = 0;
  float head[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  // Eight fixed-seed random projections (#2877). The seed is `n`, so two arms
  // tapping the same shape draw the same `w_i` and the projection difference
  // is meaningful. ProjRng is deterministic and stateless beyond its seed.
  constexpr int kNumProj = 8;
  ProjRng rng(static_cast<uint64_t>(n));
  float w[kNumProj];
  for (int k = 0; k < kNumProj; ++k) w[k] = rng.next();
  double proj[kNumProj] = {0.0};

  for (int64_t i = 0; i < n; ++i) {
    float v = 0.0f;
    switch (t.dtype) {
      case DType::kF32: v = reinterpret_cast<const float*>(host.data())[i]; break;
      case DType::kF16: v = vt::F16ToF32(reinterpret_cast<const uint16_t*>(host.data())[i]); break;
      case DType::kBF16: v = vt::BF16ToF32(reinterpret_cast<const uint16_t*>(host.data())[i]); break;
      default:
        VT_CHECK(false,
                 std::string("qwen4_exp forward: VT_Q4EXP_LAYER_FP cannot read dtype ") +
                     vt::Name(t.dtype));
    }
    if (i < 4) head[i] = v;
    if (!std::isfinite(v)) {
      ++nonfinite;
      continue;
    }
    const double a = std::fabs(static_cast<double>(v));
    sumabs += a;
    if (a > maxabs) maxabs = a;
    // Random projection accumulator. `w[k]` is a fresh weight per element per
    // projection, so `proj[k] = S w_k_i * x_i` over the whole tensor.
    for (int k = 0; k < kNumProj; ++k) {
      proj[k] += static_cast<double>(w[k]) * static_cast<double>(v);
    }
    for (int k = 0; k < kNumProj; ++k) w[k] = rng.next();
  }
  ++s.taps;
  std::fprintf(stderr,
               "q4fp step=%lld L%+03lld tag=%-10s dtype=%-4s dev=%d n=%lld "
               "nonfinite=%lld maxabs=%.9g sumabs=%.9g "
               "v=%.9g,%.9g,%.9g,%.9g "
               "proj=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g\n",
               static_cast<long long>(s.step), static_cast<long long>(il), tag,
               vt::Name(t.dtype), static_cast<int>(t.device.type),
               static_cast<long long>(n), static_cast<long long>(nonfinite), maxabs, sumabs,
               static_cast<double>(head[0]), static_cast<double>(head[1]),
               static_cast<double>(head[2]), static_cast<double>(head[3]),
               proj[0], proj[1], proj[2], proj[3],
               proj[4], proj[5], proj[6], proj[7]);
}

// Closes a fingerprinted step. `taps=` is the counted property that says the
// instrument RAN on the run being reported.
void LayerFpEndStep() {
  LayerFpState& s = LayerFpS();
  if (s.step < s.budget) {
    std::fprintf(stderr, "q4fp step=%lld taps=%lld END\n",
                 static_cast<long long>(s.step), static_cast<long long>(s.taps));
  }
  ++s.step;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
HfConfig Qwen4ExpGdnHfConfig(const Qwen4ExpParams& p, const HfConfig& source) {
  // The GDN block reads a small, named set of `HfConfig` fields, and every one
  // of them is copied from `p` EXCEPT the output gate, which `p` cannot answer.
  // See the header: `ParseQwen4ExpParams` validates `output_gate_type` and
  // stores no field, so the checkpoint's answer survives only on `HfConfig`.
  VT_CHECK(source.linear_num_key_heads == p.linear_num_key_heads &&
               source.linear_num_value_heads == p.linear_num_value_heads &&
               source.linear_key_head_dim == p.linear_key_head_dim &&
               source.linear_value_head_dim == p.linear_value_head_dim &&
               source.linear_conv_kernel_dim == p.linear_conv_kernel_dim &&
               source.hidden_size == p.hidden_size,
           "qwen4_exp forward: the HfConfig's Gated DeltaNet geometry disagrees "
           "with Qwen4ExpParams; the two describe ONE checkpoint, so a "
           "disagreement means the config was composed from two models");
  VT_CHECK(source.output_gate_type == "silu" ||
               source.output_gate_type == "sigmoid",
           "qwen4_exp forward: output_gate_type is \"" +
               source.output_gate_type +
               "\"; upstream accepts only sigmoid or silu "
               "(configuration_qwen4_exp.py:193-195) and the GDN output gate is "
               "its only consumer, so a third value is a silent wrong "
               "activation rather than a shape error");
  HfConfig c;
  c.model_type = "qwen4_exp_text";
  c.hidden_size = p.hidden_size;
  c.num_hidden_layers = p.num_hidden_layers;
  c.rms_norm_eps = p.rms_norm_eps;
  c.linear_num_key_heads = p.linear_num_key_heads;
  c.linear_num_value_heads = p.linear_num_value_heads;
  c.linear_key_head_dim = p.linear_key_head_dim;
  c.linear_value_head_dim = p.linear_value_head_dim;
  c.linear_conv_kernel_dim = p.linear_conv_kernel_dim;
  // THE ONE FIELD THAT IS CARRIED AND NOT DERIVED.
  c.output_gate_type = source.output_gate_type;
  c.mamba_ssm_dtype = source.mamba_ssm_dtype;
  // CARRIED, AND `GdnBlockPaged` DOES NOT READ IT — checked rather than assumed,
  // because an earlier draft of this comment claimed it did. The field used to
  // drive a dense-vs-MoE width choice through a `bool dense_model` parameter
  // that GDN-MOE-BF16-OUT (#1168) DELETED, so nothing on this path branches on
  // model shape any more. It is set because this architecture IS a MoE one and a
  // projection that described it as dense would be false about the model, not
  // because any consumer here is known to look.
  c.num_experts = p.num_experts;
  return c;
}

// ─────────────────────────────────────────────────────────────────────────────
GdnLayerWeights Qwen4ExpGdnBlockWeights(Qwen4ExpGdnWeights& g,
                                        const Qwen4ExpParams& p) {
  const int64_t H = p.hidden_size;
  const int64_t num_v = p.linear_num_value_heads;
  const int64_t key_dim = p.linear_num_key_heads * p.linear_key_head_dim;
  const int64_t value_dim = num_v * p.linear_value_head_dim;
  const int64_t conv_dim = 2 * key_dim + value_dim;

  // The `qwen4exp` loader writes `[N, K]` with `nk = true` (`MakeGdnProj` with
  // `GgufLoadPolicy::gdn_expand_nk`), which is the orientation the qwen3_5 GGUF
  // path produces for the same tensors. Both are checked, because a tower that
  // arrived in the other orientation has the same element count and would be
  // read transposed — a wrong answer, not a crash.
  CheckOwned(g.in_proj_qkv, "gdn.in_proj_qkv", {conv_dim, H});
  CheckOwned(g.in_proj_z, "gdn.in_proj_z", {value_dim, H});
  CheckOwned(g.in_proj_b, "gdn.in_proj_b", {num_v, H});
  CheckOwned(g.in_proj_a, "gdn.in_proj_a", {num_v, H});
  CheckOwned(g.conv1d, "gdn.conv1d", {conv_dim, p.linear_conv_kernel_dim});
  CheckOwned(g.a_log, "gdn.a_log", {num_v});
  CheckOwned(g.dt_bias, "gdn.dt_bias", {num_v});
  CheckOwned(g.norm_weight, "gdn.norm_weight", {p.linear_value_head_dim});
  CheckOwned(g.out_proj, "gdn.out_proj", {H, value_dim});

  // ZERO-COPY VIEWS, NOT ASSIGNMENT (issue #2476,
  // `.agents/specs/gdn-qkvz-operand-lifetime.md`). This block used to spell the
  // pass-through as `w.in_proj_qkv = g.in_proj_qkv;`, and the comment above the
  // declaration in the header already said what that was supposed to mean:
  // "nothing is ... reallocated here". `OwnedTensor`'s implicit copy DEEP-COPIES
  // an owned buffer, so the assignment reallocated ~115 MiB of Gated DeltaNet
  // weights per linear layer per step — and `ResidentWeight`'s host-alias arm
  // then handed cuBLASLt a pointer INTO that per-step copy, which this
  // function's caller destroys at the end of the layer while the GEMM is still
  // queued. `compute-sanitizer` on `thor:gpu0` reported that as `Warp illegal
  // address` in `nvjet_sm110_tst_512x8_64x3_2x1_v_bz_TNT` with every declared
  // extent correct. The view is what makes the operand belong to the model.
  GdnLayerWeights w;
  w.in_proj_qkv = BorrowWholeOwnedTensor(g.in_proj_qkv);
  w.in_proj_z = BorrowWholeOwnedTensor(g.in_proj_z);
  w.in_proj_b = BorrowWholeOwnedTensor(g.in_proj_b);
  w.in_proj_a = BorrowWholeOwnedTensor(g.in_proj_a);
  w.conv1d_weight = BorrowWholeOwnedTensor(g.conv1d);
  w.a_log = BorrowWholeOwnedTensor(g.a_log);
  w.dt_bias = BorrowWholeOwnedTensor(g.dt_bias);
  w.norm_weight = BorrowWholeOwnedTensor(g.norm_weight);
  w.out_proj = BorrowWholeOwnedTensor(g.out_proj);
  // The loader kept the five projections in the converter's TILED V-head order
  // so they could stay quantized; the GDN block undoes that on the projection
  // vectors. Carried, never re-derived: the predicate that decided the layout
  // and the predicate that undoes it must be the same one.
  w.v_head_perm_key_heads = g.v_head_perm_key_heads;
  // `in_proj_ba` and `in_proj_qkvz` are left EMPTY — see the header. That is
  // what keeps the split fields above live and it is exactly what qwen3_5's own
  // GGUF loader does.
  return w;
}

// ─────────────────────────────────────────────────────────────────────────────
namespace {

// The two RoPE layouts one set of angles has to be handed in, over the FULL
// position range `[0, kv_len)` rather than this step's `T`.
//
// TWO LAYOUTS, ONE TABLE, and the reason is in `qwen4_exp_qsa_block.h`:
// `vt::RopeFromCache` reads a PACKED `[P, rot]` cache whose columns are
// `[cos(rot/2) | sin(rot/2)]`, while `vt::Qwen4ExpQsaCompress` reads the two
// FULL `[P, rot]` tables separately and in f32. Upstream's `emb = cat(freqs,
// freqs)` (:135) makes the second half of each row a copy of the first, so both
// layouts describe the same angles BY CONSTRUCTION — which is the property the
// QSA block cross-checks rather than trusts.
//
// FULL RANGE, not this step's rows: the compressor reads row `compress_ratio*b`,
// the FIRST token of each pooled block, so it addresses positions this step did
// not produce.
struct RopeTables {
  std::vector<uint16_t> packed;  // [P, rot] bf16, cos|sin
  std::vector<float> cos;        // [P, rot] f32, second half duplicates first
  std::vector<float> sin;        // [P, rot] f32
};

RopeTables BuildQsaRope(const HfConfig& config, int64_t P) {
  const int64_t rot = config.rotary_dim;
  VT_CHECK(rot > 0 && rot % 2 == 0,
           "qwen4_exp forward: rotary_dim must be positive and even, got " +
               std::to_string(rot));
  const int64_t half = rot / 2;

  // THE SHARED BUILDER, NOT A PRIVATE ROPE LOOP. `BuildMropeCosSinHost` is
  // W5d-2's seam (#2249 item 5) and it is what the Qwen3.5/3.6 VL drivers use,
  // so this architecture builds the SAME angles rather than a second copy.
  //
  // ALL THREE AXES CARRY THE SAME POSITIONS, and that is what makes the call
  // legitimate here. On the TEXT tower upstream expands one position row to
  // three (`position_ids.view(1,1,-1).expand(4, B, -1)`, :1370) and
  // `apply_interleaved_mrope` then overwrites slots of `freqs[0]` with slots of
  // `freqs[1]` / `freqs[2]` — which is the IDENTITY when the three are equal.
  // The section split is therefore unobservable on this path, and a config that
  // carries no `mrope_section` gets the smallest valid one rather than a private
  // rope loop, which would be the parallel path AGENTS.md "Shared seams" forbids.
  HfConfig rc = config;
  if (rc.rope_parameters.mrope_section.size() != 3 ||
      rc.rope_parameters.mrope_section[0] + rc.rope_parameters.mrope_section[1] +
              rc.rope_parameters.mrope_section[2] !=
          half) {
    rc.rope_parameters.mrope_section = {half, 0, 0};
  }
  std::vector<int32_t> pos3(static_cast<size_t>(3 * P));
  for (int64_t axis = 0; axis < 3; ++axis) {
    for (int64_t i = 0; i < P; ++i)
      pos3[static_cast<size_t>(axis * P + i)] = static_cast<int32_t>(i);
  }
  const std::vector<float> cs = BuildMropeCosSinHost(pos3, P, rc);

  RopeTables r;
  r.packed.resize(static_cast<size_t>(P * rot));
  r.cos.resize(static_cast<size_t>(P * rot));
  r.sin.resize(static_cast<size_t>(P * rot));
  for (int64_t pp = 0; pp < P; ++pp) {
    for (int64_t j = 0; j < half; ++j) {
      const float c = cs[static_cast<size_t>(pp * rot + j)];
      const float s = cs[static_cast<size_t>(pp * rot + half + j)];
      r.packed[static_cast<size_t>(pp * rot + j)] = vt::F32ToBF16(c);
      r.packed[static_cast<size_t>(pp * rot + half + j)] = vt::F32ToBF16(s);
      r.cos[static_cast<size_t>(pp * rot + j)] = c;
      r.cos[static_cast<size_t>(pp * rot + half + j)] = c;
      r.sin[static_cast<size_t>(pp * rot + j)] = s;
      r.sin[static_cast<size_t>(pp * rot + half + j)] = s;
    }
  }
  return r;
}

}  // namespace

// ─────────────────────────────────────────────────────────────────────────────
Qwen4ExpTextModelOutput Qwen4ExpTextModelForward(
    Dev d, Qwen4ExpWeights& w, const HfConfig& config,
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const v1::GDNAttentionMetadata& gdn_meta,
    const Qwen4ExpForwardCaches& caches, int64_t past_len) {
  const Qwen4ExpParams& p = w.params;
  const auto T = static_cast<int64_t>(token_ids.size());
  const int64_t H = p.hidden_size;
  const int64_t hc = p.hc_count;
  const int64_t W = p.stream_width();
  const int64_t L = p.num_hidden_layers;
  const auto eps = static_cast<float>(p.rms_norm_eps);

  VT_CHECK(T > 0, "qwen4_exp forward: token_ids must not be empty");
  VT_CHECK(positions.size() == token_ids.size(),
           "qwen4_exp forward: positions and token_ids must have one entry per "
           "token; got " +
               std::to_string(positions.size()) + " and " +
               std::to_string(token_ids.size()));
  VT_CHECK(past_len >= 0, "qwen4_exp forward: past_len must not be negative");
  VT_CHECK(static_cast<int64_t>(w.layers.size()) == L,
           "qwen4_exp forward: the weights carry " +
               std::to_string(w.layers.size()) + " layers and the config names " +
               std::to_string(L));
  VT_CHECK(static_cast<int64_t>(p.layer_types.size()) == L,
           "qwen4_exp forward: layer_types has " +
               std::to_string(p.layer_types.size()) + " entries for " +
               std::to_string(L) + " layers");

  // THE CACHE VECTORS ARE COUNTED AGAINST THE CONFIG BEFORE ANY INDEXING, not
  // checked per layer as the loop reaches them. A short vector discovered at
  // layer 40 has already mutated 39 layers' worth of state.
  int64_t n_gdn = 0, n_qsa = 0;
  for (int64_t i = 0; i < L; ++i) {
    if (p.layer_types[static_cast<size_t>(i)] ==
        Qwen4ExpLayerKind::kLinearAttention) {
      ++n_gdn;
    } else {
      ++n_qsa;
    }
  }
  VT_CHECK(static_cast<int64_t>(caches.gdn.size()) == n_gdn,
           "qwen4_exp forward: " + std::to_string(caches.gdn.size()) +
               " Gated DeltaNet state caches for " + std::to_string(n_gdn) +
               " linear_attention layers");
  VT_CHECK(static_cast<int64_t>(caches.qsa.size()) == n_qsa,
           "qwen4_exp forward: " + std::to_string(caches.qsa.size()) +
               " Qwen Sparse Attention caches for " + std::to_string(n_qsa) +
               " qwen_sparse_attention layers");
  VT_CHECK(caches.ple.size() == p.ple.layer_ids_zero_based.size(),
           "qwen4_exp forward: " + std::to_string(caches.ple.size()) +
               " PLE caches for " +
               std::to_string(p.ple.layer_ids_zero_based.size()) +
               " PLE layers");

  const DType dt = kQwen4ExpStreamDType;

  // ─── :1415  inputs_embeds = self.embed_tokens(input_ids) ──────────────────
  CheckOwned(w.embed_tokens, "embed_tokens", {p.vocab_size, H});
  DBuf embed(d, dt, {T, H});
  {
    std::vector<int64_t> ids(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) {
      const int64_t id = token_ids[static_cast<size_t>(t)];
      VT_CHECK(id >= 0 && id < p.vocab_size,
               "qwen4_exp forward: token id " + std::to_string(id) +
                   " at position " + std::to_string(t) +
                   " is outside [0, vocab_size)");
      ids[static_cast<size_t>(t)] = id;
    }
    DBuf d_ids(d, DType::kI64, {T}, ids.data());
    Tensor table = dense_attn::ResidentWeight(d, w.embed_tokens, {p.vocab_size, H});
    Tensor out = embed.t();
    vt::Embedding(d.q, out, table, d_ids.t());
  }
  LayerFp(d, -1, "emb", embed.t());

  // ─── :1417  hidden_states = hidden_states.repeat(1, 1, hc_count) ──────────
  // THE WIDEN, and it is `vt::IndexSelect` over a repeat index rather than a new
  // op (#2336 §4). `repeat` on the last axis of `[T, H]` gives
  // `out[t, j*H + h] = in[t, h]`; read `out` as `[T*hc, H]` and that is exactly
  // a row gather with `idx[i] = i / hc`. A TILE — `out[t, j*H+h] = in[t, ...]`
  // with the branches interleaved the other way — would be `idx[i] = i % T` and
  // is a different tensor; the two agree only at hc == 1.
  DBuf stream(d, dt, {T, W});
  {
    std::vector<int32_t> idx(static_cast<size_t>(T * hc));
    for (int64_t i = 0; i < T * hc; ++i)
      idx[static_cast<size_t>(i)] = static_cast<int32_t>(i / hc);
    DBuf d_idx(d, DType::kI32, {T * hc}, idx.data());
    Tensor out = Reshape(stream.t(), {T * hc, H});
    vt::IndexSelect(d.q, out, embed.t(), d_idx.t());
  }
  LayerFp(d, -1, "wide", stream.t());

  // ─── :1416  position_embeddings, built ONCE for the whole step ────────────
  // Upstream builds them at the model level and hands the SAME pair to every
  // decoder layer; only the QSA layers read them. The tables cover
  // `[0, past_len + T)` because the compressor addresses pooled blocks, not
  // this step's rows.
  const int64_t kv_len = past_len + T;
  RopeTables rope;
  DBuf rope_packed, rope_cos, rope_sin;
  const bool need_rope = n_qsa > 0;
  if (need_rope) {
    rope = BuildQsaRope(config, kv_len);
    const int64_t rot = config.rotary_dim;
    rope_packed = DBuf(d, DType::kBF16, {kv_len, rot}, rope.packed.data());
    rope_cos = DBuf(d, DType::kF32, {kv_len, rot}, rope.cos.data());
    rope_sin = DBuf(d, DType::kF32, {kv_len, rot}, rope.sin.data());
  }
  DBuf d_positions(d, DType::kI32, {T}, positions.data());

  // ─── the GDN step upload, ONCE per step ───────────────────────────────────
  // `qwen3_5_gdn_block.h` is explicit that rebuilding this inside each block
  // call reinstates the per-layer upload it was factored out to remove — 36
  // times per step on this architecture.
  const HfConfig gdn_config = Qwen4ExpGdnHfConfig(p, config);
  GdnStepInputs gdn_step;
  if (n_gdn > 0) {
    int64_t slots = 0;
    for (const GdnStateCache& s : caches.gdn) {
      VT_CHECK(s.ssm_state.rank >= 1,
               "qwen4_exp forward: a Gated DeltaNet state cache has no ssm_state");
      if (slots == 0) slots = s.ssm_state.shape[0];
      VT_CHECK(s.ssm_state.shape[0] == slots,
               "qwen4_exp forward: the Gated DeltaNet state caches disagree on "
               "their slot count, and ONE step upload validates the state "
               "indices against ONE of them");
    }
    gdn_step = BuildGdnStepInputs(d.q, positions, attn_meta, gdn_meta, slots);
  }

  // ─── :1419  the layer loop ────────────────────────────────────────────────
  int64_t gdn_i = 0, qsa_i = 0;
  for (int64_t il = 0; il < L; ++il) {
    Qwen4ExpLayerWeights& lw = w.layers[static_cast<size_t>(il)];
    LayerFp(d, il, "in", stream.t());
    const bool linear = p.layer_types[static_cast<size_t>(il)] ==
                        Qwen4ExpLayerKind::kLinearAttention;
    VT_CHECK(lw.is_linear_attention == linear,
             "qwen4_exp forward: layer " + std::to_string(il) +
                 " was LOADED as a " +
                 (lw.is_linear_attention ? "linear_attention" : "sparse-attention") +
                 " layer and the config calls it the other kind; the two read "
                 "different tensor names, so this is a checkpoint/config "
                 "mismatch and not a dispatch choice");

    // ─── :1218  hidden_states = hidden_states + self.ple(...) ───────────────
    // FIRST IN THE LAYER, on the hc-WIDE stream, and ADDED to it. Not last, not
    // at the model level, and not on the collapsed hidden state — see the
    // header's ordering fact 1.
    if (lw.has_ple) {
      // Which PLE cache this is, by upstream's own indexing:
      // `config.ple_layer_ids.index(layer_idx + 1)` (:1202). NOT the decoder
      // layer index, and the layer multipliers are derived from it, so getting
      // it wrong gathers somebody else's rows from a 320-million-row table with
      // no shape error.
      int64_t ple_i = -1;
      for (size_t k = 0; k < p.ple.layer_ids_zero_based.size(); ++k) {
        if (p.ple.layer_ids_zero_based[k] == il) {
          ple_i = static_cast<int64_t>(k);
          break;
        }
      }
      VT_CHECK(ple_i >= 0,
               "qwen4_exp forward: layer " + std::to_string(il) +
                   " was loaded WITH PLE weights and `ple_layer_ids` does not "
                   "name it");
      std::vector<int64_t> ple_ids(static_cast<size_t>(T));
      for (int64_t t = 0; t < T; ++t)
        ple_ids[static_cast<size_t>(t)] = token_ids[static_cast<size_t>(t)];
      const qwen4_exp::NGramTableLayout layout = Qwen4ExpPleLayout(p, ple_i);
      const Qwen4ExpPleBlockOutput ple = RunQwen4ExpPleBlock(
          d, lw.ple, w.ngram_table, p, layout, stream.t(), ple_ids.data(),
          /*conv_mask=*/nullptr, caches.ple[static_cast<size_t>(ple_i)], past_len);
      LayerFp(d, il, "ple", ple.tensor);
      Tensor s = stream.t();
      vt::Add(d.q, s, s, ple.tensor);
      LayerFp(d, il, "s.ple", s);
    }

    // ─── :1222  the ATTENTION hyper-connection ──────────────────────────────
    // `mixed` is the block's [T, H] input; `injection` is the [T, hc] rank-1
    // weight the write-back uses; `stream` is untouched and stays the RAW
    // stream, which is what :1237 adds to (ordering fact 3).
    DBuf mixed(d, dt, {T, H});
    DBuf injection(d, dt, {T, hc});
    {
      vt::Qwen4ExpGatedResidualArgs args;
      args.hc_count = hc;
      args.hidden_size = H;
      args.lowrank = p.hc_lowrank;
      args.eps = eps;
      VT_CHECK(lw.attn_hc.has_inject,
               "qwen4_exp forward: layer " + std::to_string(il) +
                   "'s attention hyper-connection has no block_inject_weight; "
                   "`use_combine=False` is the MODEL-level mixer alone");
      Tensor m = mixed.t();
      Tensor inj = injection.t();
      // W5r/#2449: THROUGH THE SAME STAGING ITS SIBLINGS USE. `View()` builds a
      // tensor over the OwnedTensor's HOST bytes and tags it with the queue's
      // device, so on a CUDA queue `vt::Qwen4ExpGatedResidual` refused this
      // operand -- `block_inject_weight device mismatch` -- at decoder layer 0,
      // before PLE and before the MoE, which is every CUDA forward for this
      // architecture. `hc_norm`, `down` and `up` in this same call already go
      // through `dense_attn::ResidentWeight`; the inject weight is a matmul
      // operand of the same op and had no reason to be the exception.
      Tensor bi = dense_attn::ResidentWeight(d, lw.attn_hc.inject, {hc, W});
      vt::Qwen4ExpGatedResidual(
          d.q, m, &inj, stream.t(),
          dense_attn::ResidentWeight(d, lw.attn_hc.hc_norm, {W}),
          dense_attn::ResidentWeight(d, lw.attn_hc.down, {p.hc_lowrank, W}),
          dense_attn::ResidentWeight(d, lw.attn_hc.up, {W, p.hc_lowrank}), &bi,
          args);
    }
    LayerFp(d, il, "ahc.mix", mixed.t());
    LayerFp(d, il, "ahc.inj", injection.t());

    // ─── :1224 / :1228  the mixer block, one arm per layer kind ─────────────
    Tensor block_out;
    std::shared_ptr<void> block_store;
    if (linear) {
      // BUILT ONCE AND HELD BY THE MODEL (issue #2476). A `GdnLayerWeights`
      // built here per step is a set of handles this scope OWNS, and
      // `ResidentWeight` writes its residency memo — the device copy, and the
      // host-alias decision that hands a kernel `bytes.data()` — onto the
      // handle it is given. So a per-step adapter both re-established residency
      // every step and destroyed the operand at the closing brace below, with
      // the block's GEMMs still queued. `lw.gdn_block` outlives every step,
      // holds no bytes of its own, and is the memo.
      if (!lw.gdn_block.has_value())
        lw.gdn_block.emplace(Qwen4ExpGdnBlockWeights(lw.gdn, p));
      const GdnBlockOutput o = RunGdnBlockPaged(
          d.q, *lw.gdn_block, gdn_config, mixed.t(), gdn_step, gdn_meta,
          caches.gdn[static_cast<size_t>(gdn_i)], T);
      block_out = o.tensor;
      block_store = o.storage;
      ++gdn_i;
    } else {
      VT_CHECK(need_rope, "qwen4_exp forward: a sparse-attention layer needs "
                          "the rope tables and none were built");
      const Qwen4ExpQsaBlockOutput o = RunQwen4ExpQsaBlockPaged(
          d, lw.qsa, p, mixed.t(), d_positions.t(), rope_packed.t(),
          rope_cos.t(), rope_sin.t(),
          caches.qsa[static_cast<size_t>(qsa_i)], past_len);
      block_out = o.tensor;
      block_store = o.storage;
      ++qsa_i;
    }
    LayerFp(d, il, "blk", block_out);

    // ─── :1236-1237  the rank-1 write-back, IN PLACE on the raw stream ──────
    {
      vt::Qwen4ExpGatedResidualArgs args;
      args.hc_count = hc;
      args.hidden_size = H;
      args.lowrank = p.hc_lowrank;
      args.eps = eps;
      Tensor s = stream.t();
      vt::Qwen4ExpGatedResidualWriteBack(d.q, s, block_out, injection.t(), args);
    }
    LayerFp(d, il, "s.attn", stream.t());
    block_store.reset();

    // ─── :1239  the MLP hyper-connection ────────────────────────────────────
    DBuf mlp_in(d, dt, {T, H});
    DBuf mlp_injection(d, dt, {T, hc});
    {
      vt::Qwen4ExpGatedResidualArgs args;
      args.hc_count = hc;
      args.hidden_size = H;
      args.lowrank = p.hc_lowrank;
      args.eps = eps;
      VT_CHECK(lw.mlp_hc.has_inject,
               "qwen4_exp forward: layer " + std::to_string(il) +
                   "'s MLP hyper-connection has no block_inject_weight");
      Tensor m = mlp_in.t();
      Tensor inj = mlp_injection.t();
      // #2449, the MLP twin of the attention site above: same op, same operand
      // role, same staging.
      Tensor bi = dense_attn::ResidentWeight(d, lw.mlp_hc.inject, {hc, W});
      vt::Qwen4ExpGatedResidual(
          d.q, m, &inj, stream.t(),
          dense_attn::ResidentWeight(d, lw.mlp_hc.hc_norm, {W}),
          dense_attn::ResidentWeight(d, lw.mlp_hc.down, {p.hc_lowrank, W}),
          dense_attn::ResidentWeight(d, lw.mlp_hc.up, {W, p.hc_lowrank}), &bi,
          args);
    }
    LayerFp(d, il, "mhc.mix", mlp_in.t());
    LayerFp(d, il, "mhc.inj", mlp_injection.t());

    // ─── :1240  hidden_states = self.mlp(hidden_states) ─────────────────────
    // Through the SHARED sparse-MoE seam, which is what W5d-4's adapter exists
    // for; this loop composes `Qwen4ExpMoeBlockWeights` ONCE PER LAYER and
    // holds it on that layer, rather than writing a second MoE path.
    {
      // BUILT ONCE AND HELD BY THE MODEL, exactly as `lw.gdn_block` above is
      // and for the same reason (W6,
      // `ISSUE-LOCAL-01M2AA9C31GCSDV8NRW26GMEVS`). A `MoeBlockWeights` composed
      // HERE is a set of handles this scope owns, and `BorrowWholeOwnedTensor`
      // carries every field EXCEPT `d_dev` — so each step presented
      // `ResidentWeight` a null residency memo and each of the three expert
      // towers, the three shared-expert projections, the router and the shared
      // gate took the staging arm again. Measured on `dgx:gpu0` (2026-09-12,
      // `51c248190`, GB10, released UD-IQ1_S): 378 `cudaMalloc` calls per decode
      // step against the 48 x 8 = 384 this composition asks for, 2.25 s of the
      // step inside the allocator against 0.101 s in every GPU kernel together.
      // `lw.moe_block` outlives every step, so the memo written onto its handles
      // is still there on the next one.
      //
      // NON-CONST by W5d-4's own contract: the per-expert views BORROW the
      // tower's bytes and `OwnedBytes::KeepAlive()` converts the owned buffer
      // into a shared read-only one IN PLACE. That is also why the adapter is
      // not a pure function of the layer and is built OUTSIDE the placed body
      // rather than inside the lambda: it must not be re-evaluated per
      // placement arm.
      //
      // THE PASS-THROUGH STAYS A BORROW, and the hoist does not make that
      // optional. `Qwen4ExpMoeBlockWeights` passes the towers through
      // `BorrowWhole`, so the buffer a queued kernel reads belongs to the
      // model. The GDN adapter next door spelled the same pass-through as
      // assignment, which deep-copies ~115 MiB per linear layer per step and
      // left a freed operand underneath a queued GEMM (issue #2476). Holding
      // the adapter makes the handles outlive the kernels; it does not make a
      // copy safe.
      if (!lw.moe_block.has_value())
        lw.moe_block.emplace(Qwen4ExpMoeBlockWeights(lw.moe, p));
      const MoeBlockWeights& mw = *lw.moe_block;

      // ENG-HYBRID-PLACEMENT (#2424): through the SHARED placement seam, inert by
      // construction when this layer is not placed. The block already has the
      // seam's shape — a `[T,H]` bf16 block in, an owning `MoeBlockOutput` pair
      // out — so this is the `qwen3_moe.cpp` adapter verbatim and not a refactor.
      //
      // NO `placeable` REFUSAL IS OWED HERE. `Qwen4ExpMoeBlockWeights` fills only
      // the keep-quant stacked arm or the per-expert bf16 borrowed-view arm and
      // hard-refuses a third dtype; it never touches `expert_*_fp4`, so this
      // architecture has no eagerly-device-resident expert arm for the fp4
      // refusal to protect against. Passing `expert_gate_fp4.empty()` would be
      // correct but permanently dead, which reads as a guard that is doing
      // something.
      //
      // The lambda's device parameter is `pd`, NOT `p`: `p` is the
      // `Qwen4ExpParams` this body captures, and shadowing it would silently
      // compile against the wrong object.
      const MoePlacedOutput placed = vllm::RunMoePlacedPair(
          d, il, mlp_in.t(), T, H,
          [&](dense_attn::Dev pd, const Tensor& h) {
            MoeBlockOutput b = RunQwen4ExpMoeBlock(pd.q, mw, p, h, T);
            return MoePlacedOutput{b.tensor, std::move(b.storage)};
          });
      const MoeBlockOutput o{placed.tensor, placed.storage};
      LayerFp(d, il, "moe", o.tensor);

      vt::Qwen4ExpGatedResidualArgs args;
      args.hc_count = hc;
      args.hidden_size = H;
      args.lowrank = p.hc_lowrank;
      args.eps = eps;
      Tensor s = stream.t();
      vt::Qwen4ExpGatedResidualWriteBack(d.q, s, o.tensor, mlp_injection.t(),
                                         args);
    }
    LayerFp(d, il, "s.mlp", stream.t());
  }

  VT_CHECK(gdn_i == n_gdn && qsa_i == n_qsa,
           "qwen4_exp forward: the loop consumed " + std::to_string(gdn_i) +
               " of " + std::to_string(n_gdn) + " Gated DeltaNet caches and " +
               std::to_string(qsa_i) + " of " + std::to_string(n_qsa) +
               " sparse-attention caches");

  // ─── :1430  hidden_states = self.hyper_connection_mixer(hidden_states) ────
  // The terminal `use_combine=False` collapse: a NULL `block_inject` and a NULL
  // `injection` out-parameter, which is the arm that returns `mixed_input`
  // alone. There is NO final RMSNorm after it (ordering fact 4) — the mixer's
  // own `hc_norm` is the last normalization in the model.
  VT_CHECK(!w.mixer.has_inject,
           "qwen4_exp forward: the model-level mixer carries a "
           "block_inject_weight; `use_combine=False` is what makes it the "
           "terminal collapse and an inject tensor means it was loaded as a "
           "per-layer site");
  DBuf out(d, dt, {T, H});
  {
    vt::Qwen4ExpGatedResidualArgs args;
    args.hc_count = hc;
    args.hidden_size = H;
    args.lowrank = p.hc_lowrank;
    args.eps = eps;
    Tensor o = out.t();
    vt::Qwen4ExpGatedResidual(
        d.q, o, /*injection=*/nullptr, stream.t(),
        dense_attn::ResidentWeight(d, w.mixer.hc_norm, {W}),
        dense_attn::ResidentWeight(d, w.mixer.down, {p.hc_lowrank, W}),
        dense_attn::ResidentWeight(d, w.mixer.up, {W, p.hc_lowrank}),
        /*block_inject=*/nullptr, args);
  }
  LayerFp(d, -1, "out", out.t());
  LayerFpEndStep();

  Qwen4ExpTextModelOutput r;
  r.tensor = out.t();
  r.storage = out.ReleaseShared();
  return r;
}

namespace detail {

// ─── VT_Q4EXP_LOGITS: the step's logits ─────────────────────────────────────
//
// The layer tap stops at `out`; the logits are produced in the registry, after
// the `lm_head` matmul, so the head's own decode had no observation. This tap
// reads the finished `[rows, vocab]` f32 buffer through the same `Backend::Copy`
// + `Synchronize` pair the layer tap uses, so one spelling reads a CPU queue and
// a CUDA one.
struct LogitsTapState {
  int64_t topk = 0;    // pairs to print; 0 == the instrument is off
  int64_t budget = 0;  // forward calls to tap
  int64_t step = 0;    // the forward this process is on
  int64_t taps = 0;    // lines printed, the counted property
  int64_t last_top1 = -1;
};

int64_t LogitsTapTopkFromEnv() {
  const char* e = std::getenv("VT_Q4EXP_LOGITS");
  if (e == nullptr || e[0] == '\0') return 0;
  const long long k = std::atoll(e);
  if (k <= 0) return 0;
  return k > 64 ? 64 : static_cast<int64_t>(k);
}

int64_t LogitsTapBudgetFromEnv() {
  const char* e = std::getenv("VT_Q4EXP_LOGITS_STEPS");
  if (e == nullptr || e[0] == '\0') return 8;
  const long long v = std::atoll(e);
  return v > 0 ? static_cast<int64_t>(v) : 0;
}

LogitsTapState& LogitsTapS() {
  static LogitsTapState s = [] {
    LogitsTapState t;
    t.topk = LogitsTapTopkFromEnv();
    t.budget = t.topk > 0 ? LogitsTapBudgetFromEnv() : 0;
    return t;
  }();
  return s;
}

void Qwen4ExpLogitsTap(Dev d, const Tensor& logits) {
  LogitsTapState& s = LogitsTapS();
  if (s.topk <= 0 || s.step >= s.budget) return;
  VT_CHECK(logits.IsContiguous(),
           "qwen4_exp logits tap: VT_Q4EXP_LOGITS taps a contiguous [rows, vocab] tensor");
  VT_CHECK(logits.rank == 2 && logits.dtype == DType::kF32,
           "qwen4_exp logits tap: VT_Q4EXP_LOGITS taps f32 [rows, vocab]");
  const int64_t rows = logits.shape[0];
  const int64_t vocab = logits.shape[1];
  if (rows <= 0 || vocab <= 0) return;
  std::vector<float> host(static_cast<size_t>(rows) * static_cast<size_t>(vocab));
  d.b.Copy(d.q, host.data(), logits.data, host.size() * sizeof(float));
  d.b.Synchronize(d.q);

  // The top-k and the extremes come from the FIRST row, which is the row the
  // sampler asked for first on a prefill; the projections cover the whole buffer
  // so a multi-row step is still one comparison per arm.
  const size_t n = static_cast<size_t>(vocab);
  std::vector<int32_t> idx(n);
  for (size_t i = 0; i < n; ++i) idx[i] = static_cast<int32_t>(i);
  const auto k = static_cast<size_t>(std::min<int64_t>(s.topk, vocab));
  std::partial_sort(idx.begin(), idx.begin() + static_cast<std::ptrdiff_t>(k), idx.end(),
                    [&host](int32_t a, int32_t b) { return host[static_cast<size_t>(a)] > host[static_cast<size_t>(b)]; });
  float vmax = host[0];
  float vmin = host[0];
  for (size_t i = 1; i < n; ++i) {
    if (host[i] > vmax) vmax = host[i];
    if (host[i] < vmin) vmin = host[i];
  }

  constexpr int kNumProj = 8;
  ProjRng rng(static_cast<uint64_t>(host.size()));
  float w[kNumProj];
  for (int i = 0; i < kNumProj; ++i) w[i] = rng.next();
  double proj[kNumProj] = {0.0};
  double sumabs = 0.0;
  for (size_t i = 0; i < host.size(); ++i) {
    const float v = host[i];
    if (std::isfinite(v)) {
      sumabs += std::fabs(static_cast<double>(v));
      for (int p = 0; p < kNumProj; ++p) proj[p] += static_cast<double>(w[p]) * static_cast<double>(v);
    }
    for (int p = 0; p < kNumProj; ++p) w[p] = rng.next();
  }

  s.last_top1 = k > 0 ? static_cast<int64_t>(idx[0]) : -1;
  ++s.taps;
  std::fprintf(stderr,
               "q4logit step=%lld rows=%lld n=%lld top=%lld top1=%lld vmax=%.9g vmin=%.9g "
               "sumabs=%.9g proj=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g pairs=",
               static_cast<long long>(s.step), static_cast<long long>(rows),
               static_cast<long long>(vocab), static_cast<long long>(k),
               static_cast<long long>(s.last_top1), static_cast<double>(vmax),
               static_cast<double>(vmin), sumabs, proj[0], proj[1], proj[2], proj[3], proj[4],
               proj[5], proj[6], proj[7]);
  for (size_t j = 0; j < k; ++j) {
    std::fprintf(stderr, "%s%lld:%.9g", j == 0 ? "" : ",",
                 static_cast<long long>(idx[j]), static_cast<double>(host[static_cast<size_t>(idx[j])]));
  }
  std::fputc('\n', stderr);
  ++s.step;
}

void Qwen4ExpLogitsTapResetForTest() {
  LogitsTapState& s = LogitsTapS();
  s.topk = LogitsTapTopkFromEnv();
  s.budget = s.topk > 0 ? LogitsTapBudgetFromEnv() : 0;
  s.step = 0;
  s.taps = 0;
  s.last_top1 = -1;
}

int64_t Qwen4ExpLogitsTapTapsForTest() { return LogitsTapS().taps; }

int64_t Qwen4ExpLogitsTapLastTop1ForTest() { return LogitsTapS().last_top1; }

void Qwen4ExpLayerFpResetForTest() {
  LayerFpState& s = LayerFpS();
  s.budget = LayerFpBudgetFromEnv();
  s.step = 0;
  s.taps = 0;
}

int64_t Qwen4ExpLayerFpTapsForTest() { return LayerFpS().taps; }

}  // namespace detail

}  // namespace vllm
