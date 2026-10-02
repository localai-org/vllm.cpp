// Nemotron-H W4: the hybrid forward. See nemotron_h_forward.h for the port
// anchors, the scope boundary and the no-RoPE finding.
//
// EVERY numeric step below is a landed `vt::` op — nothing here re-implements a
// kernel. The three Mamba2 primitives are #496's (`vt::Mamba2ChunkScan`,
// `vt::RmsNormGatedGroup`), the causal conv is the one the GDN hybrids already
// use (`vt::CausalConv1dFwd`), the non-gated expert is #517 W2's
// (`vt::MoeRelu2` + `vt::MoeCombine`'s `routed_scale`), and the router, the
// attention and the norms are the shared ops every other model calls. This file
// is the WIRING, which is exactly what W4 owns.
//
// HOST-QUEUE ONLY, deliberately. The forward asserts a CPU queue: it materializes
// fresh per-call conv/SSM state, splits the fused projections with host copies,
// and manages no paged cache. The device/paged runner path is W6 and is a
// separate entry point, exactly as Kimi-Linear kept `Forward` (host reference)
// and `ForwardDevice`/`ForwardPaged` apart.
#include "vllm/model_executor/models/nemotron_h_forward.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>

// The SHARED ModelOpt dequant seam (DequantNvfp4ToBf16 / DequantFp8ToBf16 /
// kNvfp4GroupSize). The host reference forward has no NVFP4 and no FP8 GEMM, so
// a quantized operand is widened here rather than in a hand-rolled sibling of
// the utility every other ModelOpt consumer in this tree already uses.
#include "vllm/model_executor/model_loader/nvfp4_dequant.h"
#include "vt/ops.h"
#include "vt/recipes.h"

namespace vllm {
namespace {

// VT_FUSED_CHAIN_ADOPT (default ON, consistent with the framework): route the
// residual add + RMSNorm preamble through the declared fusion recipe
// (`kFusedAddRmsNormStd`) via `vt::FusedChain` rather than hand-calling the
// residual overload of `vt::RmsNorm` — AGENTS.md, "Route model fusion through
// `vt::FusedChain`", and the seam `scripts/check-fusion-consistency.py` enforces.
//
// Behaviour-preserving BY CONSTRUCTION: the recipe encodes exactly this op order
// (`res += x; out = std-RMSNorm(res)`) and the default Tier-0 composite
// dispatches to the SAME `vt::RmsNorm(..., &residual)` primitive, so the fused
// path is bit-identical to the hand-call (tests/vt/test_ops_fused_chain.cpp).
// `VT_FUSED_CHAIN_ADOPT=0` restores the exact hand-call as a same-binary A/B,
// and the forward gate runs both arms.
//
// Defined file-locally rather than pulled from `dense_attn_block.h`: that header
// is the DEVICE seam (device pool, resident weights, paged KV), and this is the
// host reference forward. qwen3_5.cpp:1698 keeps its own reader for the same
// reason; this is a five-line env read, not a second numeric path.
bool FusedChainAdoptEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

using vt::DType;
using vt::Queue;
using vt::Tensor;

// ─── dtype-generic host element access ──────────────────────────────────────
// The same LoadF32/StoreF32 idiom the CPU kernels use file-locally
// (src/vt/cpu/cpu_ops.cpp:31). Only f32 and bf16 are admitted: those are the two
// `IsOutFloat` dtypes every op below can store into, and admitting f16 here
// would create an arm no `vt::MoeRelu2` / `vt::MoeCombine` output accepts.
float LoadAt(const void* base, DType dt, int64_t i) {
  if (dt == DType::kF32) return static_cast<const float*>(base)[i];
  return vt::BF16ToF32(static_cast<const uint16_t*>(base)[i]);
}

void StoreAt(void* base, DType dt, int64_t i, float v) {
  if (dt == DType::kF32) {
    static_cast<float*>(base)[i] = v;
    return;
  }
  static_cast<uint16_t*>(base)[i] = vt::F32ToBF16(v);
}

void CheckActDType(DType dt) {
  VT_CHECK(dt == DType::kF32 || dt == DType::kBF16,
           "NemotronHForCausalLM forward: the model dtype must be bf16 (the "
           "released checkpoint's) or f32 (the reference sweep arm); no other "
           "activation dtype is reachable from the ops this forward composes");
}

// An owned host activation buffer in a declared dtype, plus its view.
struct Buf {
  std::vector<uint8_t> bytes;
  DType dtype = DType::kF32;
  std::vector<int64_t> shape;

  Buf() = default;
  Buf(DType dt, std::vector<int64_t> s) : dtype(dt), shape(std::move(s)) {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    VT_CHECK(n >= 0, "NemotronH forward: negative buffer extent");
    bytes.assign(static_cast<size_t>(n) * vt::SizeOf(dtype), 0);
  }
  int64_t Numel() const {
    int64_t n = 1;
    for (int64_t d : shape) n *= d;
    return n;
  }
  // A contiguous view over the current bytes, at `as` (defaults to the buffer's
  // own shape). A reshape is free because every Buf is contiguous.
  Tensor t(vt::Device dev, const std::vector<int64_t>& as) const {
    Tensor r;
    r.data = const_cast<uint8_t*>(bytes.data());
    r.dtype = dtype;
    r.device = dev;
    r.rank = static_cast<int>(as.size());
    VT_CHECK(r.rank >= 1 && r.rank <= vt::kMaxRank,
             "NemotronH forward: buffer rank out of range");
    int64_t stride = 1;
    int64_t n = 1;
    for (int i = r.rank - 1; i >= 0; --i) {
      r.shape[i] = as[static_cast<size_t>(i)];
      r.stride[i] = stride;
      stride *= as[static_cast<size_t>(i)];
      n *= as[static_cast<size_t>(i)];
    }
    VT_CHECK(n == Numel(), "NemotronH forward: reshape changes the element count");
    return r;
  }
  Tensor t(vt::Device dev) const { return t(dev, shape); }
  float Get(int64_t i) const { return LoadAt(bytes.data(), dtype, i); }
  void Set(int64_t i, float v) { StoreAt(bytes.data(), dtype, i, v); }
};

Buf PackF32(const std::vector<float>& v, DType dt, std::vector<int64_t> shape) {
  Buf b(dt, std::move(shape));
  VT_CHECK(b.Numel() == static_cast<int64_t>(v.size()),
           "NemotronH forward: packed buffer element count mismatch");
  for (size_t i = 0; i < v.size(); ++i) b.Set(static_cast<int64_t>(i), v[i]);
  return b;
}

std::vector<float> UnpackF32(const Buf& b) {
  std::vector<float> out(static_cast<size_t>(b.Numel()));
  for (size_t i = 0; i < out.size(); ++i) out[i] = b.Get(static_cast<int64_t>(i));
  return out;
}

Tensor I32(std::vector<int32_t>& v, vt::Device dev, std::vector<int64_t> shape) {
  Tensor t;
  t.data = v.data();
  t.dtype = DType::kI32;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

Tensor F32V(std::vector<float>& v, vt::Device dev, std::vector<int64_t> shape) {
  Tensor t;
  t.data = v.data();
  t.dtype = DType::kF32;
  t.device = dev;
  t.rank = static_cast<int>(shape.size());
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

// A2-R (#810): the same three checks for the weights that moved to the shared
// `OwnedTensor` residency type (embeddings, the 53 norms, attention q/k/v/o).
// An overload rather than a template so the refusal messages stay byte-identical
// to the NemotronHOwned arm below — diffing the two should show one difference,
// the shape accessor.
void RequireWeight(const OwnedTensor& w, const char* what, DType want,
                   std::vector<int64_t> shape) {
  VT_CHECK(!w.Empty(), std::string("NemotronHForCausalLM forward: weight '") + what +
                           "' is not materialized (the safetensors/quantized "
                           "weight load is owed; see "
                           ".agents/specs/nemotron-h-model.md §5b)");
  VT_CHECK(w.dtype == want,
           std::string("NemotronHForCausalLM forward: weight '") + what +
               "' has the wrong dtype for this arm");
  VT_CHECK(w.rank == static_cast<int>(shape.size()) &&
               std::equal(shape.begin(), shape.end(), w.shape),
           std::string("NemotronHForCausalLM forward: weight '") + what +
               "' has the wrong shape");
}

void RequireWeight(const NemotronHOwned& w, const char* what, DType want,
                   std::vector<int64_t> shape) {
  VT_CHECK(!w.Empty(), std::string("NemotronHForCausalLM forward: weight '") + what +
                           "' is not materialized (the safetensors/quantized "
                           "weight load is owed; see "
                           ".agents/specs/nemotron-h-model.md §5b)");
  VT_CHECK(w.dtype == want,
           std::string("NemotronHForCausalLM forward: weight '") + what +
               "' has the wrong dtype for this arm");
  VT_CHECK(w.shape == shape, std::string("NemotronHForCausalLM forward: weight '") +
                                 what + "' has the wrong shape");
}

// A DENSE operand for one GEMM, in `want`.
//
// Dense weights are viewed in place — no copy, byte-identical to the W4 path.
// A quantized weight (NVFP4 W4A16 g16 experts / lm_head, FP8 W8A8 static mamba
// projections) is dequantized through the SHARED ModelOpt seam into a TRANSIENT
// buffer that lives exactly as long as this GEMM.
//
// This is the declared host arm, not a silent fallback. The host reference
// forward composes `vt::MatmulBT`, which has no NVFP4 and no FP8 entry point;
// the quantized GEMMs are the CUDA `kMoeGroupedGemmNvfp4Marlin` and fp8-linear
// registrations W6 selects. Two properties keep it honest: the weight KEEPS its
// quantized memory format in host memory (so RSS and the load report describe
// the checkpoint, not a widened copy of it), and the arm is NAMED by the loader
// header and reported by the load report rather than being discoverable only by
// reading this function.
//
// What the report counts is the QUANTIZED WEIGHTS — 5935 NVFP4 W4A16 g16
// projections and 46 FP8 W8A8 static ones, the population this widening can be
// applied to. It does NOT count dequant EVENTS: a dequant here is transient and
// per GEMM call, so its count is a property of the workload, not of the load.
struct DenseOperand {
  std::vector<uint8_t> owned;  // non-empty only when a dequant happened
  Tensor view;
};

DenseOperand DenseFor(const NemotronHOwned& w, DType want, vt::Device dev) {
  DenseOperand d;
  if (w.IsDense()) {
    d.view = w.View(dev);
    return d;
  }
  // The shared seam produces bf16; widen losslessly for the f32 reference arm.
  d.owned = w.DenseBf16();
  const int64_t n = w.Numel();
  if (want == DType::kF32) {
    std::vector<uint8_t> wide(static_cast<size_t>(n) * sizeof(float));
    const auto* src = reinterpret_cast<const uint16_t*>(d.owned.data());
    auto* dst = reinterpret_cast<float*>(wide.data());
    for (int64_t i = 0; i < n; ++i) dst[i] = vt::BF16ToF32(src[i]);
    d.owned = std::move(wide);
  }
  d.view.data = d.owned.data();
  d.view.dtype = want;
  d.view.device = dev;
  d.view.rank = static_cast<int>(w.shape.size());
  VT_CHECK(d.view.rank >= 1 && d.view.rank <= vt::kMaxRank,
           "NemotronH forward: dequantized operand rank out of range");
  int64_t stride = 1;
  for (int i = d.view.rank - 1; i >= 0; --i) {
    d.view.shape[i] = w.shape[static_cast<size_t>(i)];
    d.view.stride[i] = stride;
    stride *= w.shape[static_cast<size_t>(i)];
  }
  return d;
}

// An OWNED dense `NemotronHOwned` at `want`, so one dequant can serve several
// GEMMs (the expert-major MoE loop). Same seam and same declared-arm reasoning
// as `DenseFor`; a dense input is returned unchanged.
NemotronHOwned DenseCopy(const NemotronHOwned& w, DType want) {
  if (w.IsDense()) return w;
  NemotronHOwned out;
  out.dtype = want;
  out.shape = w.shape;
  out.bytes = w.DenseBf16();
  if (want == DType::kF32) {
    const int64_t n = w.Numel();
    std::vector<uint8_t> wide(static_cast<size_t>(n) * sizeof(float));
    const auto* src = reinterpret_cast<const uint16_t*>(out.bytes.data());
    auto* dst = reinterpret_cast<float*>(wide.data());
    for (int64_t i = 0; i < n; ++i) dst[i] = vt::BF16ToF32(src[i]);
    out.bytes = std::move(wide);
  }
  return out;
}

// out[M,N] = a[M,K] @ b^T, b [N,K] — the torch-Linear orientation every weight
// above is stored in.
Buf Linear(Queue& q, const Buf& a, const NemotronHOwned& w, int64_t M, int64_t K,
           int64_t N, const char* what) {
  RequireWeight(w, what, a.dtype, {N, K});
  Buf out(a.dtype, {M, N});
  Tensor at = a.t(q.device, {M, K});
  const DenseOperand wd = DenseFor(w, a.dtype, q.device);
  Tensor ot = out.t(q.device);
  vt::MatmulBT(q, ot, at, wd.view);
  return out;
}

// A2-R: the OwnedTensor arm. There is NO DenseFor here and that is the point —
// an OwnedTensor weight is dense by construction, so this arm has no dequant
// branch to take and cannot silently widen anything. `View()` yields a host/CPU
// tensor, which is what this HOST reference forward's queue is.
Buf Linear(Queue& q, const Buf& a, const OwnedTensor& w, int64_t M, int64_t K,
           int64_t N, const char* what) {
  RequireWeight(w, what, a.dtype, {N, K});
  // `nk` IS CONSUMED HERE, not merely recorded. `vt::MatmulBT` reads `b` as
  // [N=out, K=in] — the raw torch-Linear orientation `nk = true` names
  // (qwen3_5_weights.h:57-61). A weight the loader recorded as [K, N] is a
  // TRANSPOSED GEMM operand: same byte count, same shape, a plausible wrong
  // answer. Refused by name, as qwen3_5.cpp:3353 / :3611 / :7638 refuse the
  // same thing for their own MatmulBT operands. Before this check `nk` had no
  // reader on this path at all, so flipping it in the loader changed nothing
  // any gate could see.
  VT_CHECK(w.nk, std::string("NemotronHForCausalLM forward: weight '") + what +
                     "' is not in the [out, in] torch-Linear orientation "
                     "vt::MatmulBT consumes");
  Buf out(a.dtype, {M, N});
  Tensor at = a.t(q.device, {M, K});
  Tensor wt = w.View();
  Tensor ot = out.t(q.device);
  vt::MatmulBT(q, ot, at, wt);
  return out;
}

// Copy a column range [c0, c0+width) out of `src [rows, src_cols]` into a fresh
// CONTIGUOUS buffer. `vt::Mamba2ChunkScan` validates every operand contiguous
// (ops.cpp CheckMamba2Operand), so the fused zxbcdt / xBC splits cannot be
// handed to it as the row-strided views `Tensor::Slice` produces — only
// `vt::CausalConv1dFwd` documents a padded-row `x`.
Buf SliceCols(const Buf& src, int64_t rows, int64_t src_cols, int64_t c0, int64_t width,
              std::vector<int64_t> out_shape) {
  Buf out(src.dtype, std::move(out_shape));
  VT_CHECK(out.Numel() == rows * width, "NemotronH forward: slice extent mismatch");
  const size_t esz = vt::SizeOf(src.dtype);
  for (int64_t r = 0; r < rows; ++r) {
    std::memcpy(out.bytes.data() + static_cast<size_t>(r * width) * esz,
                src.bytes.data() + static_cast<size_t>(r * src_cols + c0) * esz,
                static_cast<size_t>(width) * esz);
  }
  return out;
}

// The non-gated expert, on the seam spec §6a fixed: ONE projection, relu², one
// projection. `rows` tokens of `in [rows, H]` against expert weights.
Buf NonGatedExpert(Queue& q, const Buf& in, const NemotronHExpertWeights& w, int64_t rows,
                   int64_t H, int64_t I, const char* what) {
  Buf h = Linear(q, in, w.up_proj, rows, H, I, what);
  Buf a(in.dtype, {rows, I});
  Tensor ht = h.t(q.device);
  Tensor at = a.t(q.device);
  vt::MoeRelu2(q, at, ht);
  return Linear(q, a, w.down_proj, rows, I, H, what);
}

}  // namespace

// ─── NemotronHOwned ─────────────────────────────────────────────────────────

int64_t NemotronHOwned::Numel() const {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

vt::Tensor NemotronHOwned::View(vt::Device device) const {
  // A view over packed NVFP4 nibbles or e4m3 bytes, TYPED as the logical dtype,
  // reads finite plausible garbage of the right shape — no kernel and no shape
  // check can catch it, and a token gate cannot see it. So the dense view
  // refuses a quantized weight by name and `DenseFor` is the only way to a GEMM
  // operand.
  VT_CHECK(form == NemotronHWeightForm::kDense,
           "NemotronHOwned::View: this weight is held in its SHIPPED quantized "
           "form (NVFP4 W4A16 g16 or FP8 W8A8 static); a dense view of it would "
           "reinterpret packed bytes as the model dtype. Materialize it with "
           "DenseBf16() at the call site instead.");
  Tensor t;
  t.data = const_cast<uint8_t*>(bytes.data());
  t.dtype = dtype;
  t.device = device;
  t.rank = static_cast<int>(shape.size());
  VT_CHECK(t.rank >= 1 && t.rank <= vt::kMaxRank,
           "NemotronHOwned::View: rank out of range");
  int64_t stride = 1;
  for (int i = t.rank - 1; i >= 0; --i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    t.stride[i] = stride;
    stride *= shape[static_cast<size_t>(i)];
  }
  return t;
}

std::vector<uint8_t> NemotronHOwned::DenseBf16() const {
  VT_CHECK(shape.size() == 2,
           "NemotronHOwned::DenseBf16: only a 2-D [out, in] projection is "
           "quantized in this architecture");
  const int64_t rows = shape[0];
  const int64_t cols = shape[1];
  std::vector<uint8_t> out(static_cast<size_t>(rows * cols) * sizeof(uint16_t));
  auto* dst = reinterpret_cast<uint16_t*>(out.data());
  switch (form) {
    case NemotronHWeightForm::kNvfp4W4A16G16:
      VT_CHECK(bytes.size() == static_cast<size_t>(rows * cols / 2),
               "NemotronHOwned::DenseBf16: NVFP4 payload is not [rows, cols/2]");
      VT_CHECK(scale.size() == static_cast<size_t>(rows * cols / kNvfp4GroupSize),
               "NemotronHOwned::DenseBf16: NVFP4 group scales are not "
               "[rows, cols/16]");
      // The SHARED ModelOpt seam, at the DEFAULTED nibble order — ModelOpt and
      // compressed-tensors checkpoints are low-first
      // (.agents/specs/nvfp4-nibble-order.md), and this is one.
      DequantNvfp4ToBf16(bytes.data(), scale.data(), global_scale, rows, cols,
                         dst);
      return out;
    case NemotronHWeightForm::kFp8W8A8Static:
      VT_CHECK(bytes.size() == static_cast<size_t>(rows * cols),
               "NemotronHOwned::DenseBf16: FP8 payload is not [rows, cols]");
      // Weight-only: `input_scale` is carried, not applied. Nothing on the host
      // path quantizes the activation, so applying it here would scale the
      // product by a factor upstream applies to the OTHER operand.
      DequantFp8ToBf16(bytes.data(), global_scale, rows * cols, dst);
      return out;
    case NemotronHWeightForm::kDense:
      break;
  }
  VT_CHECK(false,
           "NemotronHOwned::DenseBf16: called on a weight that is already dense");
  return out;
}

NemotronHOwned NemotronHOwned::FromF32(const std::vector<float>& values, vt::DType dtype,
                                       std::vector<int64_t> shape) {
  CheckActDType(dtype);
  NemotronHOwned w;
  w.dtype = dtype;
  w.shape = std::move(shape);
  VT_CHECK(w.Numel() == static_cast<int64_t>(values.size()),
           "NemotronHOwned::FromF32: element count does not match the shape");
  w.bytes.assign(values.size() * vt::SizeOf(dtype), 0);
  for (size_t i = 0; i < values.size(); ++i) {
    StoreAt(w.bytes.data(), dtype, static_cast<int64_t>(i), values[i]);
  }
  return w;
}

// ─── (1) the Mamba2 mixer ───────────────────────────────────────────────────
//
// mamba_mixer2.py:548-586 `MambaMixer2.forward`, with the conv+SSM body from
// `conv_ssm_forward` (:687-696 the split, :830-891 the prefill arm).
std::vector<float> NemotronHMamba2Mixer(const NemotronHMambaWeights& w,
                                        const NemotronHParams& params,
                                        const std::vector<float>& hidden_normed,
                                        int64_t num_tokens, vt::DType act_dtype,
                                        vt::Queue& queue,
                                        NemotronHMambaState* state) {
  CheckActDType(act_dtype);
  const int64_t T = num_tokens;
  const int64_t H = params.hidden_size;
  const int64_t I = params.mamba_intermediate_size();      // 4096
  const int64_t Cd = params.conv_dim();                    // 6144
  const int64_t P = params.mamba_head_dim;                 // 64
  const int64_t Hh = params.mamba_num_heads;               // 64
  const int64_t G = params.n_groups;                       // 8
  const int64_t N = params.ssm_state_size;                 // 128
  const int64_t K = params.conv_kernel;                    // 4
  const int64_t proj = params.in_proj_out_features();      // 10304
  VT_CHECK(T > 0, "NemotronH mamba mixer: empty token sequence");
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NemotronH mamba mixer: hidden size mismatch");
  VT_CHECK(Hh * P == I, "NemotronH mamba mixer: num_heads*head_dim != intermediate");
  VT_CHECK(I + 2 * G * N == Cd, "NemotronH mamba mixer: conv_dim mismatch");

  const vt::Device dev = queue.device;
  const Buf x = PackF32(hidden_normed, act_dtype, {T, H});

  // 1. the fused zxbcdt projection (mamba_mixer2.py:550).
  const Buf zxbcdt = Linear(queue, x, w.in_proj, T, H, proj, "mixer.in_proj");

  // 2. split: z | xBC | dt (mamba_mixer2.py:692-696 reads xBC/dt off the tail,
  //    :583 reads the gate off the head).
  Buf z = SliceCols(zxbcdt, T, proj, 0, I, {T, I});
  Buf xbc = SliceCols(zxbcdt, T, proj, I, Cd, {T, Cd});
  Buf dt = SliceCols(zxbcdt, T, proj, I + Cd, Hh, {T, Hh});

  // 3. the causal depthwise conv with the silu activation
  //    (`activation=config.mamba_hidden_act` = "silu", mamba_mixer2.py:832-846).
  VT_CHECK(params.mamba_hidden_act == "silu",
           "NemotronHForCausalLM forward: only mamba_hidden_act=silu is ported "
           "(the checkpoint ships silu); an other activation is refused rather "
           "than silently substituted");
  RequireWeight(w.conv1d_weight, "mixer.conv1d.weight", act_dtype, {Cd, K});
  // conv_state is f32 BY OP CONTRACT (ops.cpp CheckConvCommon admits bf16 only
  // where Backend::SupportsCompressedConvState()), and it is a TRANSIENT
  // per-call buffer here, never the persistent cache — `MakeNemotronHKVCache`
  // already declares the persistent conv page at the cache dtype and W6 owns
  // driving it.
  std::vector<float> conv_state(static_cast<size_t>(Cd * (K - 1)), 0.0f);
  const bool carry_in = state != nullptr && state->has_initial;
  if (carry_in) {
    VT_CHECK(state->conv.size() == conv_state.size(),
             "NemotronH mamba mixer: carried conv state has the wrong extent");
    conv_state = state->conv;
  }
  std::vector<int32_t> qsl = {0, static_cast<int32_t>(T)};
  std::vector<int32_t> has_init = {carry_in ? 1 : 0};
  Buf xbc_out(act_dtype, {T, Cd});
  {
    Tensor xt = xbc.t(dev);
    Tensor wt = w.conv1d_weight.View(dev);
    Tensor st = F32V(conv_state, dev, {1, Cd, K - 1});
    Tensor ot = xbc_out.t(dev);
    Tensor qt = I32(qsl, dev, {2});
    Tensor it = I32(has_init, dev, {1});
    vt::CausalConv1dArgs cargs;
    cargs.silu_activation = true;
    if (params.use_conv_bias) {
      // The conv bias is a MODEL-DTYPE tensor (ColumnParallelLinear's bias), not
      // one of the three f32 SSM parameters below.
      RequireWeight(w.conv1d_bias, "mixer.conv1d.bias", act_dtype, {Cd});
      Tensor bt = w.conv1d_bias.View(dev);
      vt::CausalConv1dFwd(queue, ot, xt, wt, &bt, st, qt, it, cargs);
    } else {
      vt::CausalConv1dFwd(queue, ot, xt, wt, nullptr, st, qt, it, cargs);
    }
  }

  // 4. split the conv output into x | B | C (mamba_mixer2.py:535-543).
  Buf ssm_x = SliceCols(xbc_out, T, Cd, 0, I, {T, Hh, P});
  Buf ssm_b = SliceCols(xbc_out, T, Cd, I, G * N, {T, G, N});
  Buf ssm_c = SliceCols(xbc_out, T, Cd, I + G * N, G * N, {T, G, N});

  // 5. the SSD scan. `A = -exp(A_log)` in f32 — upstream keeps it f32 whatever
  //    the model dtype (`self.A = -torch.exp(self.A_log.float())`) and
  //    vt::Mamba2ChunkScan validates A/D/dt_bias as f32.
  RequireWeight(w.A_log, "mixer.A_log", DType::kF32, {Hh});
  RequireWeight(w.D, "mixer.D", DType::kF32, {Hh});
  RequireWeight(w.dt_bias, "mixer.dt_bias", DType::kF32, {Hh});
  std::vector<float> A(static_cast<size_t>(Hh));
  for (int64_t h = 0; h < Hh; ++h) {
    A[static_cast<size_t>(h)] =
        -std::exp(LoadAt(w.A_log.bytes.data(), DType::kF32, h));
  }
  // The SSM state dtype is resolved INDEPENDENTLY of every activation dtype
  // above — `mamba_ssm_cache_dtype` is "float32" on this checkpoint while the
  // tower is bf16 (nemotron_h.h NemotronHSsmCacheDType records why the shared
  // qwen3_5 resolver is the wrong reader). Deriving it from act_dtype halves the
  // recurrent state and is invisible to a token gate.
  const DType ssm_dtype = NemotronHSsmCacheDType(params, act_dtype);
  Buf final_states(ssm_dtype, {1, Hh, P, N});
  Buf initial_states(ssm_dtype, {1, Hh, P, N});
  if (carry_in) {
    VT_CHECK(state->ssm.dtype == ssm_dtype && state->ssm.Numel() == Hh * P * N,
             "NemotronH mamba mixer: carried SSM state has the wrong dtype or extent");
    std::memcpy(initial_states.bytes.data(), state->ssm.bytes.data(),
                state->ssm.bytes.size());
  }

  // One sequence, chunked on the GLOBAL token position — the single-sequence
  // case of `compute_varlen_chunk_metadata` (v1/attention/backends/mamba2_attn.py
  // :22-88); starting at position 0, every logical chunk is a full physical one
  // but the last.
  const int64_t chunk = params.chunk_size;
  std::vector<int32_t> cu_seqlens = {0, static_cast<int32_t>(T)};
  std::vector<int32_t> cu_chunk = {0};
  std::vector<int32_t> seq_idx;
  for (int64_t pos = 0; pos < T; pos += chunk) {
    cu_chunk.push_back(static_cast<int32_t>(std::min(pos + chunk, T)));
    seq_idx.push_back(0);
  }
  std::vector<int32_t> last_chunk = {static_cast<int32_t>(seq_idx.size()) - 1};

  Buf y(act_dtype, {T, Hh, P});
  {
    Tensor outt = y.t(dev);
    Tensor fst = final_states.t(dev);
    Tensor xt = ssm_x.t(dev);
    Tensor dtt = dt.t(dev);
    Tensor At = F32V(A, dev, {Hh});
    Tensor Bt = ssm_b.t(dev);
    Tensor Ct = ssm_c.t(dev);
    Tensor Dt = w.D.View(dev);
    Tensor dbt = w.dt_bias.View(dev);
    Tensor cut = I32(cu_seqlens, dev, {2});
    Tensor cct = I32(cu_chunk, dev, {static_cast<int64_t>(cu_chunk.size())});
    Tensor lct = I32(last_chunk, dev, {1});
    Tensor sit = I32(seq_idx, dev, {static_cast<int64_t>(seq_idx.size())});
    vt::Mamba2Args args;
    args.chunk_size = chunk;
    // mamba_mixer2.py:888-889: dt_softplus=True, dt_limit=(0.0, +inf). `z` is
    // NOT passed to the scan — upstream gates in the norm below (:583-585), and
    // passing it here would apply silu(z) twice.
    args.dt_softplus = true;
    args.dt_min = 0.0f;
    args.dt_max = std::numeric_limits<float>::infinity();
    Tensor ist = initial_states.t(dev);
    vt::Mamba2ChunkScan(queue, outt, fst, xt, dtt, At, Bt, Ct, &Dt, /*z=*/nullptr, &dbt,
                        carry_in ? &ist : nullptr, cut, cct, lct, sit, args);
  }
  if (state != nullptr) {
    state->conv = conv_state;
    state->ssm.dtype = ssm_dtype;
    state->ssm.shape = {Hh, P, N};
    state->ssm.bytes = final_states.bytes;
    state->has_initial = true;
  }

  // 6. the silu-gated GROUP RMS norm (Mixer2RMSNormGated, mamba_mixer2.py:478-480,
  //    :583-585). n_groups is the mixer's, NOT 1.
  RequireWeight(w.norm_weight, "mixer.norm.weight", act_dtype, {I});
  Buf normed(act_dtype, {T, I});
  {
    Tensor ot = normed.t(dev);
    Tensor xt = y.t(dev, {T, I});
    Tensor gt = z.t(dev);
    Tensor wt = w.norm_weight.View(dev);
    vt::RmsNormGatedGroupArgs args;
    args.eps = static_cast<float>(params.layer_norm_epsilon);
    args.n_groups = G;
    vt::RmsNormGatedGroup(queue, ot, xt, gt, &wt, args);
  }

  // 7. out_proj (mamba_mixer2.py:586).
  const Buf out = Linear(queue, normed, w.out_proj, T, I, H, "mixer.out_proj");
  return UnpackF32(out);
}

// ─── (2) the GQA attention mixer ────────────────────────────────────────────
//
// nemotron_h.py:473-486. NO RoPE — see kNemotronHAttentionHasNoRope.
std::vector<float> NemotronHAttentionMixer(const NemotronHAttentionWeights& w,
                                           const NemotronHParams& params,
                                           const std::vector<float>& hidden_normed,
                                           int64_t num_tokens, vt::DType act_dtype,
                                           vt::Queue& queue) {
  CheckActDType(act_dtype);
  const int64_t T = num_tokens;
  const int64_t H = params.hidden_size;
  const int64_t Hq = params.num_attention_heads;
  const int64_t Hkv = params.num_key_value_heads;
  const int64_t Dh = params.head_dim;
  VT_CHECK(T > 0, "NemotronH attention: empty token sequence");
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NemotronH attention: hidden size mismatch");
  VT_CHECK(!params.attention_bias,
           "NemotronHForCausalLM forward: attention_bias is not ported (the "
           "checkpoint has attention_bias=false and ships no q/k/v/o bias)");
  VT_CHECK(!params.sliding_window.has_value(),
           "NemotronHForCausalLM forward: per-layer sliding_window is not ported "
           "(this checkpoint ships sliding_window=null)");

  const vt::Device dev = queue.device;
  const Buf x = PackF32(hidden_normed, act_dtype, {T, H});
  const int64_t qdim = params.q_proj_out_features();
  const int64_t kvdim = params.kv_proj_out_features();

  const Buf qb = Linear(queue, x, w.q_proj, T, H, qdim, "mixer.q_proj");
  const Buf kb = Linear(queue, x, w.k_proj, T, H, kvdim, "mixer.k_proj");
  const Buf vb = Linear(queue, x, w.v_proj, T, H, kvdim, "mixer.v_proj");

  Buf attn(act_dtype, {T, Hq, Dh});
  {
    Tensor ot = attn.t(dev);
    Tensor qt = qb.t(dev, {T, Hq, Dh});
    Tensor kt = kb.t(dev, {T, Hkv, Dh});
    Tensor vt_ = vb.t(dev, {T, Hkv, Dh});
    vt::AttentionArgs args;
    // `self.scaling = self.head_dim**-0.5` (nemotron_h.py:440).
    args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(Dh)));
    args.causal = true;
    // VT-ATTN-NAIVE: the HOST reference arm. `NemotronHAttnBlock` in
    // nemotron_h_device.cpp is the device arm, and the equivalence gate between
    // them holds only while both run the same kernel — the fast rungs are NOT
    // bit-identical to this one. Changing one side reroutes the comparison
    // rather than the model (#1544).
    vt::Attention(queue, ot, qt, kt, vt_, args);
  }

  const Buf out = Linear(queue, attn, w.o_proj, T, qdim, H, "mixer.o_proj");
  return UnpackF32(out);
}

// ─── (3) the non-gated relu² MoE mixer ──────────────────────────────────────
//
// nemotron_h.py:126-256. Three things this block is the whole reason to gate:
//  * the router runs in f32 (`GateLinear(out_dtype=torch.float32,
//    force_fp32_compute=True)`, :150-156) — MIRRORED, not inherited;
//  * `routed_scaling_factor` is applied to the OUTPUT
//    (`apply_routed_scale_to_output=True`, :234) via MoeCombine's `routed_scale`,
//    NOT folded into the router weights or the logits — the ROUTER's own factor
//    is forced to 1.0 in exactly this case (layer.py:291-300);
//  * the shared expert is added UNSCALED after the routed sum is scaled
//    (moe_runner.py:402-406 then :722-725).
std::vector<float> NemotronHMoeMixer(const NemotronHMoeWeights& w,
                                     const NemotronHParams& params,
                                     const std::vector<float>& hidden_normed,
                                     int64_t num_tokens, vt::DType act_dtype,
                                     vt::Queue& queue) {
  CheckActDType(act_dtype);
  const int64_t T = num_tokens;
  const int64_t H = params.hidden_size;
  const int64_t E = params.n_routed_experts;
  const int64_t Kk = params.num_experts_per_tok;
  const int64_t I = params.moe_intermediate_size;
  VT_CHECK(T > 0, "NemotronH moe: empty token sequence");
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NemotronH moe: hidden size mismatch");
  VT_CHECK(!params.moe_latent_size.has_value(),
           "NemotronHForCausalLM forward: moe_latent_size is out of scope "
           "(fc1_latent_proj/fc2_latent_proj, spec §0); it is null in the "
           "released checkpoint");
  VT_CHECK(static_cast<int64_t>(w.experts.size()) == E,
           "NemotronH moe: expert count does not match n_routed_experts");

  const vt::Device dev = queue.device;

  // --- router. f32 END TO END: the logits buffer, the GEMM operands and the
  // bias. This is upstream's own polarity (`force_fp32_compute=True`), which is
  // why an f32 buffer appears on a bf16 model path here and nowhere else in this
  // file except the SSM state and the transient conv state.
  RequireWeight(w.gate, "mixer.gate.weight", DType::kF32, {E, H});
  RequireWeight(w.e_score_correction_bias, "mixer.gate.e_score_correction_bias",
                DType::kF32, {E});
  std::vector<float> logits(static_cast<size_t>(T * E), 0.0f);
  {
    std::vector<float> hf32 = hidden_normed;
    // The activation reaching the router is the model-dtype one, widened — not a
    // separately-computed f32 activation. Round-tripping it through act_dtype
    // first keeps that true when act_dtype is bf16.
    if (act_dtype != DType::kF32) {
      const Buf rounded = PackF32(hidden_normed, act_dtype, {T, H});
      hf32 = UnpackF32(rounded);
    }
    Tensor at = F32V(hf32, dev, {T, H});
    Tensor wt = w.gate.View(dev);
    Tensor ot = F32V(logits, dev, {T, E});
    vt::MatmulBT(queue, ot, at, wt);
  }

  std::vector<float> topk_w(static_cast<size_t>(T * Kk), 0.0f);
  std::vector<int32_t> topk_id(static_cast<size_t>(T * Kk), -1);
  {
    vt::MoeRouterTopKArgs args;
    args.top_k = static_cast<int>(Kk);
    args.renormalize = params.norm_topk_prob;
    args.scoring_func = vt::MoeScoringFunc::kSigmoid;  // nemotron_h.py:225
    args.num_expert_group = static_cast<int>(params.n_group);
    args.topk_group = static_cast<int>(params.topk_group);
    // NOT params.routed_scaling_factor. layer.py:291-300 forces the router's
    // factor to 1.0 whenever apply_routed_scale_to_output is set, which it is.
    args.routed_scaling_factor = 1.0f;
    Tensor lt = F32V(logits, dev, {T, E});
    Tensor wt = F32V(topk_w, dev, {T, Kk});
    Tensor it = I32(topk_id, dev, {T, Kk});
    Tensor bt = w.e_score_correction_bias.View(dev);
    vt::MoeRouterTopK(queue, wt, it, lt, args, &bt);
  }

  // --- routed experts, one (token, slot) pair at a time. The grouped-GEMM arms
  // (kMoeGroupedGemmBf16 / kMoeGroupedGemmNvfp4Marlin) are CUDA-only
  // registrations, so the host reference takes the per-expert MatmulBT loop the
  // CPU MoE path already uses for every other arch (deepseek_v2.cpp's CPU
  // fallback); the composition is identical and W6 selects the grouped arm.
  const Buf x = PackF32(hidden_normed, act_dtype, {T, H});
  Buf expert_out(act_dtype, {T, Kk, H});
  const size_t esz = vt::SizeOf(act_dtype);
  // EXPERT-MAJOR iteration over the (token, slot) pairs. Each pair's own
  // `NonGatedExpert` call is unchanged — one row against one expert, so every
  // output row's arithmetic is byte-identical to the token-major order this
  // replaces — but visiting the pairs grouped by expert lets ONE dequant of a
  // quantized expert serve all of that expert's rows. On the released
  // checkpoint an expert is an NVFP4 W4A16 g16 pair (~10e6 elements); the
  // token-major order re-materializes it per row.
  //
  // Ordering cannot change the RESULT: each pair writes its own disjoint
  // `expert_out` slot and reads nothing another pair wrote.
  std::vector<std::vector<int64_t>> pairs(static_cast<size_t>(E));
  for (int64_t t = 0; t < T; ++t) {
    for (int64_t j = 0; j < Kk; ++j) {
      const int32_t e = topk_id[static_cast<size_t>(t * Kk + j)];
      VT_CHECK(e >= 0 && e < E, "NemotronH moe: router emitted an invalid expert id");
      pairs[static_cast<size_t>(e)].push_back(t * Kk + j);
    }
  }
  for (int64_t e = 0; e < E; ++e) {
    const std::vector<int64_t>& slots = pairs[static_cast<size_t>(e)];
    if (slots.empty()) continue;
    const NemotronHExpertWeights& src = w.experts[static_cast<size_t>(e)];
    // The dense stand-in exists only while this expert is being served, so peak
    // host RSS carries ONE dequantized expert, not 128 of them.
    NemotronHExpertWeights dense;
    const NemotronHExpertWeights* use = &src;
    if (!src.up_proj.IsDense() || !src.down_proj.IsDense()) {
      dense.up_proj = DenseCopy(src.up_proj, act_dtype);
      dense.down_proj = DenseCopy(src.down_proj, act_dtype);
      use = &dense;
    }
    for (int64_t slot : slots) {
      const int64_t t = slot / Kk;
      Buf row(act_dtype, {1, H});
      std::memcpy(row.bytes.data(), x.bytes.data() + static_cast<size_t>(t * H) * esz,
                  static_cast<size_t>(H) * esz);
      const Buf y = NonGatedExpert(queue, row, *use, 1, H, I, "mixer.experts");
      std::memcpy(expert_out.bytes.data() + static_cast<size_t>(slot * H) * esz,
                  y.bytes.data(), static_cast<size_t>(H) * esz);
    }
  }

  // --- shared expert (nemotron_h.py:176-190): the same non-gated shape, at
  // moe_shared_expert_intermediate_size * n_shared_experts.
  Buf shared;
  if (w.has_shared) {
    VT_CHECK(params.n_shared_experts > 0,
             "NemotronH moe: shared expert weights present but n_shared_experts is 0");
    const int64_t Is = params.moe_shared_expert_intermediate_size * params.n_shared_experts;
    shared = NonGatedExpert(queue, x, w.shared, T, H, Is, "mixer.shared_experts");
  }

  Buf out(act_dtype, {T, H});
  {
    Tensor ot = out.t(dev);
    Tensor eo = expert_out.t(dev);
    Tensor wt = F32V(topk_w, dev, {T, Kk});
    Tensor st = shared.bytes.empty() ? Tensor{} : shared.t(dev);
    vt::MoeCombine(queue, ot, eo, wt, shared.bytes.empty() ? nullptr : &st,
                   static_cast<float>(params.routed_scaling_factor));
  }
  return UnpackF32(out);
}

// ─── (4) the dense `-` MLP mixer ────────────────────────────────────────────
//
// nemotron_h.py:86-123 `NemotronHMLP`: up_proj -> act_fn -> down_proj, the same
// non-gated relu² shape as an expert.
std::vector<float> NemotronHMlpMixer(const NemotronHMlpWeights& w,
                                     const NemotronHParams& params,
                                     const std::vector<float>& hidden_normed,
                                     int64_t num_tokens, vt::DType act_dtype,
                                     vt::Queue& queue) {
  CheckActDType(act_dtype);
  const int64_t T = num_tokens;
  const int64_t H = params.hidden_size;
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NemotronH mlp: hidden size mismatch");
  VT_CHECK(params.mlp_hidden_act == "relu2",
           "NemotronHForCausalLM forward: only mlp_hidden_act=relu2 is ported "
           "(the checkpoint's); another activation is refused by name rather "
           "than silently substituted");
  VT_CHECK(!params.mlp_bias,
           "NemotronHForCausalLM forward: mlp_bias is not ported (false in the "
           "released checkpoint)");
  const Buf x = PackF32(hidden_normed, act_dtype, {T, H});
  NemotronHExpertWeights e;
  e.up_proj = w.up_proj;
  e.down_proj = w.down_proj;
  const Buf out = NonGatedExpert(queue, x, e, T, H, params.intermediate_size, "mixer");
  return UnpackF32(out);
}

// ─── (5) the whole decoder ──────────────────────────────────────────────────

std::vector<float> NemotronHForward(const NemotronHHostWeights& host,
                                    const NemotronHParams& params,
                                    const std::vector<int32_t>& token_ids,
                                    const std::vector<int32_t>& logits_indices,
                                    vt::Queue& queue, NemotronHTrace* trace,
                                    const std::vector<float>* inputs_embeds) {
  VT_CHECK(host.materialized,
           "NemotronHForCausalLM forward: host weights are not materialized. W4 "
           "ports the forward MECHANISM; the safetensors/NVFP4/FP8 weight load "
           "that fills NemotronHHostWeights is still owed (see "
           ".agents/specs/nemotron-h-model.md §5b), and the GGUF arm is W7. "
           "Refusing by name rather than computing on zeros.");
  VT_CHECK(queue.device.type == vt::DeviceType::kCPU,
           "NemotronHForCausalLM forward: this is the HOST reference forward and "
           "requires a CPU queue; the device/paged runner path is W6 of "
           ".agents/specs/nemotron-h-model.md");
  CheckActDType(host.act_dtype);

  const DType adt = host.act_dtype;
  const vt::Device dev = queue.device;
  const int64_t H = params.hidden_size;
  const int64_t V = params.vocab_size;
  const int64_t L = params.num_hidden_layers();
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "NemotronHForCausalLM forward: empty token sequence");
  VT_CHECK(static_cast<int64_t>(host.layers.size()) == L,
           "NemotronHForCausalLM forward: host layer count != layers_block_type length");
  RequireWeight(host.embeddings, "backbone.embeddings.weight", adt, {V, H});
  RequireWeight(host.norm_f, "backbone.norm_f.weight", adt, {H});
  VT_CHECK(!params.tie_word_embeddings,
           "NemotronHForCausalLM forward: tie_word_embeddings is false in the "
           "released checkpoint and the tied arm is not ported");
  RequireWeight(host.lm_head, "lm_head.weight", adt, {V, H});

  // The residual stream carries the MODEL dtype, mirroring vLLM's
  // fused_add_rms_norm residual. Widening it is numerically correct, invisible
  // to a token gate, and doubles the bytes — the trap AGENTS.md names.
  Buf residual(adt, {T, H});
  if (inputs_embeds != nullptr) {
    VT_CHECK(static_cast<int64_t>(inputs_embeds->size()) == T * H,
             "NemotronHForCausalLM forward: inputs_embeds must be [T, hidden]");
    Tensor ot = residual.t(dev);
    if (adt == DType::kBF16) {
      std::vector<uint16_t> bits(inputs_embeds->size());
      for (size_t i = 0; i < bits.size(); ++i) bits[i] = vt::F32ToBF16((*inputs_embeds)[i]);
      std::memcpy(ot.data, bits.data(), bits.size() * sizeof(uint16_t));
    } else {
      std::memcpy(ot.data, inputs_embeds->data(), inputs_embeds->size() * sizeof(float));
    }
  } else {
    std::vector<int32_t> ids = token_ids;
    for (int32_t id : ids) {
      VT_CHECK(id >= 0 && id < V, "NemotronHForCausalLM forward: token id out of range");
    }
    Tensor ot = residual.t(dev);
    Tensor tab = host.embeddings.View();
    Tensor it = I32(ids, dev, {T});
    vt::Embedding(queue, ot, tab, it);
  }

  vt::RmsNormArgs nargs;
  nargs.eps = static_cast<float>(params.layer_norm_epsilon);
  nargs.gemma = false;

  if (trace != nullptr && trace->capture) {
    trace->normed.assign(static_cast<size_t>(L), {});
    trace->mixer.assign(static_cast<size_t>(L), {});
    trace->hidden.assign(static_cast<size_t>(L), {});
  }

  // The single-branch pre-norm stream (nemotron_h.py:625-640 + each layer's
  // :301-313). Layer 0 sees `residual is None`, so the embedding IS the residual
  // and the norm is un-fused; every later layer folds the previous mixer output
  // into the residual inside the norm.
  Buf carry(adt, {T, H});  // the previous layer's mixer output
  for (int64_t l = 0; l < L; ++l) {
    const NemotronHLayerWeights& lw = host.layers[static_cast<size_t>(l)];
    VT_CHECK(lw.block == params.layers_block_type[static_cast<size_t>(l)],
             "NemotronHForCausalLM forward: host layer block kind disagrees with "
             "layers_block_type");
    RequireWeight(lw.norm, "layer norm", adt, {H});
    Buf normed(adt, {T, H});
    {
      Tensor ot = normed.t(dev);
      Tensor wt = lw.norm.View();
      if (l == 0) {
        // `residual is None` (nemotron_h.py:627-631): the embedding IS the
        // residual and the norm is UN-fused, so there is no add to fuse here.
        Tensor xt = residual.t(dev);
        vt::RmsNorm(queue, ot, xt, wt, nargs, nullptr);
      } else {
        Tensor xt = carry.t(dev);
        Tensor rt = residual.t(dev);
        if (FusedChainAdoptEnabled()) {
          vt::FusedChain(queue, ot, xt, wt, &rt, vt::kFusedAddRmsNormStd,
                         static_cast<float>(params.layer_norm_epsilon));
        } else {
          vt::RmsNorm(queue, ot, xt, wt, nargs, &rt);
        }
      }
    }

    const std::vector<float> nvec = UnpackF32(normed);
    std::vector<float> mvec;
    switch (lw.block) {
      case NemotronHBlock::kMamba:
        mvec = NemotronHMamba2Mixer(lw.mamba, params, nvec, T, adt, queue);
        break;
      case NemotronHBlock::kAttention:
        mvec = NemotronHAttentionMixer(lw.attn, params, nvec, T, adt, queue);
        break;
      case NemotronHBlock::kMoe:
        mvec = NemotronHMoeMixer(lw.moe, params, nvec, T, adt, queue);
        break;
      case NemotronHBlock::kMlp:
        mvec = NemotronHMlpMixer(lw.mlp, params, nvec, T, adt, queue);
        break;
    }
    carry = PackF32(mvec, adt, {T, H});

    if (trace != nullptr && trace->capture) {
      trace->normed[static_cast<size_t>(l)] = nvec;
      trace->mixer[static_cast<size_t>(l)] = mvec;
      // The residual AFTER this layer is what the next norm will fold `carry`
      // into; report the sum so a per-layer comparison sees the stream itself.
      std::vector<float> h = UnpackF32(residual);
      for (size_t i = 0; i < h.size(); ++i) h[i] += mvec[i];
      trace->hidden[static_cast<size_t>(l)] = std::move(h);
    }
  }

  // `hidden_states, _ = self.norm_f(hidden_states, residual)` (nemotron_h.py:641).
  Buf final_normed(adt, {T, H});
  {
    Tensor ot = final_normed.t(dev);
    Tensor xt = carry.t(dev);
    Tensor rt = residual.t(dev);
    Tensor wt = host.norm_f.View();
    if (FusedChainAdoptEnabled()) {
      vt::FusedChain(queue, ot, xt, wt, &rt, vt::kFusedAddRmsNormStd,
                     static_cast<float>(params.layer_norm_epsilon));
    } else {
      vt::RmsNorm(queue, ot, xt, wt, nargs, &rt);
    }
  }
  if (trace != nullptr && trace->capture) trace->final_normed = UnpackF32(final_normed);

  std::vector<int64_t> want;
  if (logits_indices.empty()) {
    want.resize(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) want[static_cast<size_t>(i)] = i;
  } else {
    for (int32_t idx : logits_indices) {
      VT_CHECK(idx >= 0 && idx < T,
               "NemotronHForCausalLM forward: logits index out of range");
      want.push_back(idx);
    }
  }
  const int64_t R = static_cast<int64_t>(want.size());
  Buf gathered(adt, {R, H});
  const size_t esz = vt::SizeOf(adt);
  for (int64_t r = 0; r < R; ++r) {
    std::memcpy(gathered.bytes.data() + static_cast<size_t>(r * H) * esz,
                final_normed.bytes.data() +
                    static_cast<size_t>(want[static_cast<size_t>(r)] * H) * esz,
                static_cast<size_t>(H) * esz);
  }
  // A2-R: through the SHARED projection, so the host reference and the device
  // arm cannot drift apart in their output layer. `UnpackF32` here and
  // `PackF32` inside the helper are exact inverses for both admitted dtypes
  // (bf16->f32->bf16 is lossless; f32 is the identity), so this is
  // bit-identical to the direct `Linear` call it replaces.
  return NemotronHHostLmHead(host, params, UnpackF32(gathered), R, queue);
}

std::vector<float> NemotronHHostLmHead(const NemotronHHostWeights& host,
                                       const NemotronHParams& params,
                                       const std::vector<float>& gathered_normed,
                                       int64_t num_rows, vt::Queue& queue) {
  CheckActDType(host.act_dtype);
  const DType adt = host.act_dtype;
  const int64_t H = params.hidden_size;
  const int64_t V = params.vocab_size;
  VT_CHECK(num_rows > 0, "NemotronH lm_head: no rows requested");
  VT_CHECK(static_cast<int64_t>(gathered_normed.size()) == num_rows * H,
           "NemotronH lm_head: gathered row count does not match hidden_size");
  VT_CHECK(queue.device.type == vt::DeviceType::kCPU,
           "NemotronH lm_head: this projection is the HOST arm and requires a "
           "CPU queue — `lm_head` is NVFP4 W4A16 g16 on the released checkpoint "
           "and the device NVFP4 arm is not ported (#810 A2-Q)");
  RequireWeight(host.lm_head, "lm_head.weight", adt, {V, H});
  const Buf gathered = PackF32(gathered_normed, adt, {num_rows, H});
  const Buf logits =
      Linear(queue, gathered, host.lm_head, num_rows, H, V, "lm_head.weight");
  return UnpackF32(logits);
}

std::vector<int32_t> NemotronHGreedyDecode(const NemotronHHostWeights& host,
                                           const NemotronHParams& params,
                                           const std::vector<int32_t>& prompt,
                                           int num_new, vt::Queue& queue) {
  std::vector<int32_t> seq = prompt;
  std::vector<int32_t> out;
  const int64_t V = params.vocab_size;
  for (int step = 0; step < num_new; ++step) {
    const std::vector<float> logits = NemotronHForward(
        host, params, seq, {static_cast<int32_t>(seq.size() - 1)}, queue, nullptr);
    int32_t best = 0;
    float best_v = logits[0];
    for (int64_t o = 1; o < V; ++o) {
      if (logits[static_cast<size_t>(o)] > best_v) {
        best_v = logits[static_cast<size_t>(o)];
        best = static_cast<int32_t>(o);
      }
    }
    out.push_back(best);
    seq.push_back(best);
  }
  return out;
}

}  // namespace vllm
