// LinearMethod — the per-(scheme, device) linear-apply seam.
//
// UPSTREAM (ported FROM, ground-every-impl rule):
//   vllm/model_executor/layers/linear.py:141-181  class LinearMethodBase
//                                                  (create_weights + apply)
//   vllm/model_executor/layers/linear.py:184-230  class UnquantizedLinearMethod
//                                                  (apply == F.linear(x, weight))
//
// A LinearMethod owns HOW one logical linear's weights are laid out and
// multiplied on the running device. The model forward calls `method.Apply(x)`
// and never asks which scheme (bf16 / nvfp4 / …) or which device — exactly as a
// vLLM model calls `self.qkv_proj(x)` and the bound LinearMethod does the rest.
// The concrete method is chosen ONCE (MakeQkvMethod/MakeMlpDownMethod/… in the
// scheme headers), from the checkpoint's populated weights, not per forward call
// by a tensor-name probe.
//
// The UNQUANTIZED (bf16) method here is scheme-neutral and device-neutral: it
// runs the same vt::MatmulBT the inline path did, on whatever device the queue
// names. Quantized methods (compressed_tensors NVFP4 W4A16, …) live in the
// scheme headers under quantization/, and ask the vt::OpProvider table
// (`vt::OpRegistered`) which kernel is available here rather than testing
// `device == kCUDA` — see base_config.h for the policy/implementation split.
#pragma once
#include "vllm/model_executor/layers/quantization/base_config.h"
#include "vllm/model_executor/models/dense_attn_block.h"  // Dev, DBuf, ResidentWeight
#include "vllm/model_executor/models/qwen3_5_weights.h"   // OwnedTensor
#include "vt/compiled_gemma.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace layers {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::ResidentWeight;

// LinearMethodBase (linear.py:141). y[M,N] = x[M,K] @ W[N,K]^T for the bound
// weight; out_dtype selects the f32/bf16 result buffer. The N (output width) is
// taken from the weight, so a merged qkv_proj (N = q+2kv) or gate_up (N = 2I)
// flows through the SAME contract as a plain o_proj/down_proj — the caller
// splits/activates afterwards exactly as before.
class LinearMethodBase : public QuantizeMethodBase {
 public:
  virtual DBuf Apply(Dev d, const vt::Tensor& x, vt::DType out_dtype) const = 0;
  // The bound scheme's name (mirrors QuantizationConfig.get_name) — a diagnostic
  // + test hook proving which method the factory selected.
  virtual const char* Name() const = 0;
};

// UnquantizedLinearMethod (linear.py:184). apply == F.linear: one vt::MatmulBT
// over the resident raw-NK bf16 weight — byte-for-byte the inline
// `ResidentWeight + MatmulBT` the dense forward carried at every projection.
// Device-neutral: MatmulBT runs on whatever device the op table realizes it
// for, so nothing here says `kCUDA`.
class UnquantizedLinearMethod : public LinearMethodBase {
 public:
  explicit UnquantizedLinearMethod(const OwnedTensor* weight) : w_(weight) {}

  DBuf Apply(Dev d, const vt::Tensor& x, vt::DType out_dtype) const override {
    const int64_t M = x.shape[0];
    const int64_t N = w_->shape[0];  // raw-NK [N=out, K=in]
    vt::Tensor rw = ResidentWeight(d, *w_);
    DBuf out(d, out_dtype, {M, N});
    vt::MatmulBT(d.q, out.t(), x, rw);
    return out;
  }

  const char* Name() const override { return "bf16-unquantized"; }

 private:
  const OwnedTensor* w_;
};

// The SwiGLU merged gate_up + SiluAndMul as one method: returns silu(gate)·up as
// [M,I]. This is a distinct method from a plain linear because a quantized
// scheme may FUSE the gate_up GEMM with the activation (one Marlin GEMM over the
// N-concatenated pair, dense_nvfp4::GateUpFusedMarlinD), which a plain
// linear+SiluAndMul cannot express. Mirrors vLLM's MergedColumnParallelLinear
// (gate_up_proj) apply followed by SiluAndMul, kept together so the fused-kernel
// scheme choice has a home.
class MlpGateUpMethodBase : public QuantizeMethodBase {
 public:
  virtual DBuf Apply(Dev d, const vt::Tensor& x) const = 0;
  virtual const char* Name() const = 0;
};

// Unquantized (bf16) gate_up: one MatmulBT over the merged [2I,H] weight then
// vt::SiluAndMul — byte-for-byte the inline bf16 MLP path.
class UnquantizedMlpGateUpMethod : public MlpGateUpMethodBase {
 public:
  // `act_dtype` is the dtype of the gate_up product and of the activation. It
  // defaults to bf16, which every caller before Cohere2Moe used, so their op
  // sequence is unchanged. Cohere2Moe's f32 ARITHMETIC gate arm passes f32
  // (specs/cohere2-moe.md); its served path stays bf16.
  UnquantizedMlpGateUpMethod(const OwnedTensor* gate_up, int64_t intermediate,
                             vt::DType act_dtype = vt::DType::kBF16)
      : gate_up_(gate_up), I_(intermediate), act_dtype_(act_dtype) {}

  DBuf Apply(Dev d, const vt::Tensor& x) const override {
    const int64_t M = x.shape[0];
    vt::Tensor wgu = ResidentWeight(d, *gate_up_);  // [2I, H] raw-NK
    DBuf gate_up(d, act_dtype_, {M, 2 * I_});
    vt::MatmulBT(d.q, gate_up.t(), x, wgu);
    DBuf act(d, act_dtype_, {M, I_});
    vt::SiluAndMul(d.q, act.t(), gate_up.t());  // silu(gate)*up
    return act;
  }

  const char* Name() const override { return "bf16-gate-up"; }

 private:
  const OwnedTensor* gate_up_;
  int64_t I_;
  vt::DType act_dtype_;
};

// Unquantized (bf16) GeGLU gate_up: one MatmulBT over the merged [2I,H] weight then
// vt::GeluAndMul(tanh) — byte-for-byte the inline bf16 GeGLU MLP path the Gemma
// family (Gemma-1/2/3/4) hand-rolled. The GeGLU sibling of
// SPLIT gate/up, for a checkpoint that ships the two projections as two tensors
// and cannot merge them (MODEL-TEXT-GLM-MOE-DSA W9, #2214).
//
// WHY THE MERGED METHOD ABOVE CANNOT SERVE IT. Its operand is one `[2I,H]`
// weight. A llama.cpp k-quant conversion writes `ffn_gate.weight` and
// `ffn_up.weight` as two tensors, each with its OWN block encoding chosen by the
// dynamic-quant recipe, so there is no `[2I,H]` buffer to point at: two
// different encodings cannot be concatenated at all, and even two of the same
// encoding would have to be COPIED out of the mmap into a new host buffer, which
// on GLM-5.3 is ~1.3 GiB of resident bytes bought for nothing.
//
// SO THE SPLIT ARM IS A METHOD RATHER THAN A HAND-ROLLED PAIR OF GEMMS IN A
// MODEL TU. That is the whole point of `MlpGateUpMethodBase`: the fused-kernel
// scheme choice has ONE home, and a model that cannot merge its operands still
// arrives through it instead of writing the epilogue itself. `vt::MoeSiluMul` is
// the shared split-operand form of `vt::SiluAndMul` — the same
// `silu(gate) * up`, reading two `[M,I]` tensors instead of one `[M,2I]` — so
// the arithmetic is identical to the merged method's and only the launch count
// differs, which is the same trade the MLA A-projections already record.
class UnquantizedMlpGateUpSplitMethod : public MlpGateUpMethodBase {
 public:
  UnquantizedMlpGateUpSplitMethod(const OwnedTensor* gate, const OwnedTensor* up,
                                  int64_t intermediate)
      : gate_(gate), up_(up), I_(intermediate) {}

  DBuf Apply(Dev d, const vt::Tensor& x) const override {
    const int64_t M = x.shape[0];
    vt::Tensor wg = ResidentWeight(d, *gate_);  // [I, H] raw-NK
    vt::Tensor wu = ResidentWeight(d, *up_);    // [I, H] raw-NK
    DBuf g(d, vt::DType::kBF16, {M, I_});
    DBuf u(d, vt::DType::kBF16, {M, I_});
    vt::MatmulBT(d.q, g.t(), x, wg);
    vt::MatmulBT(d.q, u.t(), x, wu);
    DBuf act(d, vt::DType::kBF16, {M, I_});
    vt::MoeSiluMul(d.q, act.t(), g.t(), u.t());  // silu(gate)*up
    return act;
  }

  const char* Name() const override { return "bf16-gate-up-split"; }

 private:
  const OwnedTensor* gate_;
  const OwnedTensor* up_;
  int64_t I_;
};

// Unquantized (bf16) SwiGLU gate_up WITH A BIAS on the merged projection:
// `silu(gate) * up` where `gate|up = x @ W^T + b`. Same merged [2I,H] operand
// and same single MatmulBT as `UnquantizedMlpGateUpMethod`, plus one
// row-broadcast `vt::Add` of the [2I] bias before the activation.
//
// WHY THE SEAM GREW AN ARM RATHER THAN A CALLER GROWING TWO GEMMS. AGENTS.md's
// "Shared seams" says to extend a shared seam when it cannot represent the
// upstream behaviour, and otherwise to record one exact tracked exception —
// never to write a parallel path. Every member above returns `silu(gate) * up`
// from WEIGHTS ALONE, so an MLP whose projection carries a bias could not be
// expressed at all. Three upstream MLPs in this tree's reach do carry one:
//
//   * dots3-note's `dots` SPEECH encoder (`nvidia/audio_encoder.py:334-335` @
//     `9035151d6`): `nn.Linear(1280, 2*5120)` and `nn.Linear(5120, 1280)`, both
//     with torch's DEFAULT `bias=True`, and the released
//     `dots-studio/dots3-note-prev` ships `fc1.bias [10240]` and
//     `fc2.bias [1280]` for all 32 layers. That is the caller this arm has, and
//     it is why the arm lands REACHED (AGENTS.md, "Nothing lands dead") rather
//     than as a capability written for a fixture.
//   * dots3-note's VISION tower under `use_bias = true` (`vision.py:129-133`,
//     `:159` @ `9035151d6`), which is still refused BY NAME for reasons this arm
//     does not lift — see issue #2616 and
//     `.agents/specs/dots3-note.md` §4.14.4.
//   * OPT (`opt.py:149-163`), whose fc1/fc2 carry one under `config.enable_bias`
//     — but OPT is a plain ReLU MLP, not a gated one, so it does not route here.
//
// EVERY EXISTING CALLER IS BYTE-IDENTICAL BY CONSTRUCTION. This is a FOURTH
// derived class; the three above are untouched, so "no bias" is not a runtime
// branch through new code, it is the same code it always was. That is the same
// shape `UnquantizedMlpGateUpSplitMethod` and `UnquantizedMlpGateUpGeluMethod`
// took when they were added, and it is why the vision suites are re-run at
// their existing counts as the proof rather than at a tolerance.
//
// THE BIAS IS ADDED BEFORE THE ACTIVATION, on BOTH halves, which is what
// `F.linear(x, W, b)` followed by `x.chunk(2, -1)` means. Adding it after the
// SiLU, or to only the `up` half, would produce correctly-shaped wrong numbers;
// the dots3-note audio gate's reference computes the upstream order and the
// tower is compared against it.
class UnquantizedMlpGateUpBiasMethod : public MlpGateUpMethodBase {
 public:
  UnquantizedMlpGateUpBiasMethod(const OwnedTensor* gate_up,
                                 const OwnedTensor* bias, int64_t intermediate)
      : gate_up_(gate_up), bias_(bias), I_(intermediate) {}

  DBuf Apply(Dev d, const vt::Tensor& x) const override {
    const int64_t M = x.shape[0];
    vt::Tensor wgu = ResidentWeight(d, *gate_up_);  // [2I, H] raw-NK
    DBuf gate_up(d, vt::DType::kBF16, {M, 2 * I_});
    vt::MatmulBT(d.q, gate_up.t(), x, wgu);
    // ROW-BROADCAST: `b` is rank-1 [2I] matching the last dim, applied to every
    // row (`vt::Add`'s second shape, ops.h:3486-3495). `out` aliases `a`, which
    // that op documents as supported.
    vt::Tensor bias = ResidentWeight(d, *bias_);
    vt::Tensor gu = gate_up.t();
    vt::Add(d.q, gu, gu, bias);
    DBuf act(d, vt::DType::kBF16, {M, I_});
    vt::SiluAndMul(d.q, act.t(), gate_up.t());  // silu(gate + bg) * (up + bu)
    return act;
  }

  const char* Name() const override { return "bf16-gate-up-bias"; }

 private:
  const OwnedTensor* gate_up_;
  const OwnedTensor* bias_;
  int64_t I_;
};

// UnquantizedMlpGateUpMethod: same merged [2I,H] operand and same single MatmulBT;
// the ONLY difference is the activation epilogue (GeluAndMul(approximate="tanh")
// instead of SiluAndMul), mirroring vLLM's GemmaMLP (gemma.py::GemmaMLP.act_fn =
// GeluAndMul("tanh")) vs LlamaMLP's SiluAndMul. Shares MlpGateUpMethodBase so the
// merged-GEMM descriptor / fused-kernel scheme choice has one home across both
// activation families; a future nvfp4 checkpoint gets a GeGLU quant arm the same
// way the SwiGLU one gets GateUpFusedMarlinD.
class UnquantizedMlpGateUpGeluMethod : public MlpGateUpMethodBase {
 public:
  UnquantizedMlpGateUpGeluMethod(const OwnedTensor* gate_up, int64_t intermediate,
                                 bool compiled_erf = false)
      : gate_up_(gate_up), I_(intermediate), compiled_erf_(compiled_erf) {}

  DBuf Apply(Dev d, const vt::Tensor& x) const override {
    const int64_t M = x.shape[0];
    vt::Tensor wgu = ResidentWeight(d, *gate_up_);  // [2I, H] raw-NK
    DBuf gate_up(d, vt::DType::kBF16, {M, 2 * I_});
    vt::MatmulBT(d.q, gate_up.t(), x, wgu);
    DBuf act(d, vt::DType::kBF16, {M, I_});
    if (compiled_erf_)
      vt::CompiledGeluErfMul(d.q, act.t(), gate_up.t());
    else
      vt::GeluAndMul(d.q, act.t(), gate_up.t());
    return act;
  }

  const char* Name() const override { return "bf16-gate-up-gelu"; }

 private:
  const OwnedTensor* gate_up_;
  int64_t I_;
  bool compiled_erf_;
};

}  // namespace layers
}  // namespace vllm
