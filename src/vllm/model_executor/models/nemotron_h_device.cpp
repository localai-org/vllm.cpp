// Nemotron-H (`NemotronHForCausalLM`) — A2-R, the DEVICE arm
// ([spec](../../../../.agents/specs/nemotron-h-abi-e2e.md), issue #810; parent
// row #517).
//
// ─── WHAT RUNS WHERE, WHICH IS THE WHOLE SCOPE OF THIS FILE ─────────────────
//
//   DEVICE  the embedding lookup, all 52 layer norms + norm_f, and the 6 GQA
//           attention blocks. The residual stream is device-resident for the
//           entire forward.
//   HOST    the 23 Mamba2 blocks, the 23 MoE blocks, and lm_head.
//
// The line is drawn by the MEMORY FORMAT THE CHECKPOINT SHIPS, not by taste.
// The released checkpoint is `quant_algo: MIXED_PRECISION` and only 216 of its
// tensors are plain bf16 (gated: test_nemotron_h_loader.cpp:254). Everything
// this file puts on the device is from that bf16 population. Everything it
// leaves on the host is not:
//
//   * the 23 Mamba2 blocks are entered through `mixer.in_proj`, which is FP8
//     W8A8 static. The block is NOT splittable — `in_proj` produces the fused
//     zxbcdt that the conv and the scan both consume — so it moves as a unit,
//     after the shared FP8 W8A8 linear seam is extracted out of qwen3_5.cpp
//     (`ResidentFp8` :1456, `MatmulFp8CutlassD` :1495) into a real shared
//     header. That extraction is issue #940 and is deliberately NOT done here:
//     its own gate is Qwen3.5 byte-identity, which has no business landing
//     inside a NemotronH row.
//   * the 23 MoE blocks' 5888 routed + 23 shared expert projections and
//     `lm_head` are NVFP4 W4A16 g16 — 30.19e9 parameters, 15.8 GiB packed and
//     56.2 GiB dequantized to bf16. There is NO device NVFP4->bf16 dequant
//     kernel in vt (the only standalone device dequant is
//     `vt::DequantFp8ChannelBf16`, fused_ops.h:31, ROCm-only and per-channel
//     FP8), so putting them on the device in bf16 would mean a host dequant
//     plus a 56.2 GiB upload. Both gate hosts are unified-memory, so that is a
//     REBOOT rather than an OOM. nemotron_h_loader.h:36-46 rejected the same
//     design for the load; this file does not re-open it.
//
// So 46 of 52 layers still compute on the host, and each costs one download of
// the normed hidden and one upload of the mixer output. THAT BOUNCE IS
// SCAFFOLD, NOT ARCHITECTURE: every later unit deletes one pair of it. It is
// also why this file makes NO SPEED CLAIM OF ANY KIND — it is slower than the
// host reference, and that is expected and irrelevant to what it gates.
//
// ─── WHY NOT `dense_attn::AttnBlock` (issue #941) ──────────────────────────
//
// The spec's §2 names `dense_attn::AttnBlock` as this model's seam. That is
// wrong, and routing through it would reintroduce the exact defect #810 just
// removed from the runner — a shared function reading HF-config fields this
// architecture does not ship. Three measured reasons:
//
//   1. It takes `Qwen3DenseAttnWeights` (dense_attn_block.h:335), not this
//      model's separate q/k/v/o.
//   2. It reads `cfg.rms_norm_eps`. NemotronH's config ships
//      `layer_norm_epsilon` and `norm_eps` and NO `rms_norm_eps`, which
//      `hf_config.cpp:551` defaults to **0.0** — a silent eps=0 normalization.
//   3. Its default path calls `vt::RopeNeox` unconditionally
//      (dense_attn_block.h:497 — this read `:496`, which is the COMMENT line
//      above the call; corrected here per the A2-P spec §2.3 and issue #941).
//      NemotronH has no positional embedding at all
//      (`kNemotronHAttentionHasNoRope`; nemotron_h.py:473-486 @ 555967922).
//
// The tree's own idiom for exactly this case is a MODEL-LOCAL block —
// `granite.cpp:84 GraniteAttnBlock`, `gemma.cpp:42`, `gemma2.cpp:123`,
// `gemma3.cpp:108`, `glm4.cpp:80`, `gemma4.cpp:206`, none of them allowlisted.
// `NemotronHAttnBlock` below follows it and documents its deltas.
//
// ─── G-SAFE: A2-R DID NOT TOUCH IT, A2-P NARROWS IT ─────────────────────────
//
// `NemotronHDeviceForward` (A2-R, below) consumes NONE of `attn_kv`,
// `gdn_state`, `gdn_meta`, `gdn_state_slots` or `num_reqs`. It is NON-PAGED: it
// recomputes Q/K/V over the whole sequence every call, exactly as the host
// reference does, so it does not create the capability the interlock guards.
//
// `NemotronHPagedForward` (A2-P, at the bottom of this file) DOES create it. It
// writes this step's K/V into the runner's pages and reads attention back out
// of them, and it gathers and scatters the recurrent rows the runner allocated.
// So the interlock at `nemotron_h_registry.cpp` loses its `attn_kv` and
// `gdn_state` clauses in the same change, and keeps `num_reqs <= 1` — which
// A2-B removes, not A2-P.
#include "vllm/model_executor/models/nemotron_h_forward.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <string>
#include <utility>
#include <vector>

// A2-P (#810): `ModelForwardInput`. The paged forward takes the runner's step
// input WHOLE, as `ForwardKimiLinearForCausalLM` does
// (kimi_linear_registry.cpp:101), rather than decomposing it into a
// twelve-argument signature that the two non-paged seams above could not
// express anyway — neither of them can see `gdn_meta`, `gdn_state_slots`,
// `num_reqs` or `pure_decode` at all.
#include "vllm/model_executor/models/model_registry.h"
// `HostLogits` — the ONE carrier both non-paged seams already return through
// (nemotron_h_registry.cpp:185). A2-P still returns host logits because
// `lm_head` is NVFP4 and its device arm is A2-Q2b's; see the spec's §3.5.
#include "vllm/model_executor/models/qwen3_5_common.h"

// The SHARED device glue: Dev, DBuf, MakeTensor, Reshape and — the reason this
// header rather than dense_device_glue.h — `ResidentWeight`, the lazy
// upload-once seam this row converts NemotronH's dense weights onto.
#include "vllm/model_executor/models/dense_attn_block.h"
#include "vllm/model_executor/moe_placement_seam.h"
#include "vt/backend.h"
#include "vt/ops.h"
#include "vt/recipes.h"  // kFusedAddRmsNormStd

// DSR-ALLOW(A2-Q2a): TYPES, not behaviour -- vt/cuda/marlin_repack.h is BUILT ONLY under VT_MARLIN_NVFP4 (its own header note), so this include cannot be resolved by a runtime query. This is the platform leg for the CUDA Marlin repack.
#ifdef VT_MARLIN_NVFP4
// A2-Q2a: the load-time NVFP4 -> Marlin repack primitives, which the MoE arena
// drives directly with no cache attached.
#include "vt/cuda/marlin_repack.h"
#endif

// A2-Q2b: the SHARED NVFP4 W4A16 dense route, for `lm_head`.
//
// ★ THIS INCLUDE'S OWN HISTORY IS THE REVIEW NOTE. A2-Q2a deliberately did NOT
// include this header, and said so here, because `MarlinDenseResidentFor`
// (dense_nvfp4_gemm.h:379) keys its repack cache on the WEIGHT'S ADDRESS —
// issue #984 — and including it put that unsafe accessor one unqualified call
// away from a reviewer's eye. That reason was correct and it has been ANSWERED
// rather than overruled: the seam now accepts a CALLER-OWNED resident, and the
// only entry point this file calls (`dense_nvfp4::MatmulNvfp4W4A16D`) is called
// with one, out of the `lm_head_marlin` slot the weights own. Nothing in this
// translation unit calls `MarlinDenseResidentFor`, and a reviewer should check
// that claim rather than take it — it is one grep, and it is the whole
// difference between inheriting #984 and declining it.
#include "vllm/model_executor/models/dense_nvfp4_gemm.h"

namespace vllm {
namespace {

using vt::DType;
using vt::Queue;
using vt::Tensor;

// Same reuse the other model-local blocks take (granite.cpp:52): Dev / DBuf /
// ResidentWeight / MakeTensor / Reshape verbatim, and NOT AttnBlock.
using namespace dense_attn;

int64_t NumelOf(const std::vector<int64_t>& shape) {
  int64_t n = 1;
  for (int64_t d : shape) n *= d;
  return n;
}

// VT_FUSED_CHAIN_ADOPT, read exactly as the host arm reads it
// (nemotron_h.cpp:55). This is NOT an incidental duplicate: the whole point of
// A2-R's gate is that the two arms compose the IDENTICAL vt:: op sequence and
// differ only in which backend runs it. A device arm that hand-called
// `vt::RmsNorm(..., &residual)` while the host arm routed the same chain
// through `vt::FusedChain` would be comparing two different compositions and
// calling the result an equivalence — and `scripts/check-fusion-consistency.py`
// refuses it outright (AGENTS.md, "Route model fusion through `vt::FusedChain`").
//
// PRECISELY WHAT IS AND IS NOT GUARANTEED. These are TWO file-local
// function-local statics with byte-identical predicates — this one and
// `nemotron_h.cpp:56` — each latching on its own first call. They read the same
// variable with the same default, so within one process they resolve the same
// way and the equivalence gate below compares like with like. What is NOT true,
// and an earlier draft of this comment claimed, is that they "can never
// straddle": a caller that changed `VT_FUSED_CHAIN_ADOPT` between the two first
// calls would latch two different answers. Nothing in this tree does that, and
// the duplication is the tree's idiom for this env read (qwen3_5.cpp:1699), but
// the property is a convention rather than an impossibility. Hoisting the
// predicate into one shared reader would make it an impossibility, and that is
// a tree-wide change, not this row's.
bool FusedChainAdoptEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_FUSED_CHAIN_ADOPT");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

// The residual-add + RMSNorm preamble, through the declared fusion recipe.
// `residual` is updated in place (res += x) and `out` receives the norm, which
// is the contract `kFusedAddRmsNormStd` encodes and the Tier-0 composite
// dispatches to the same `vt::RmsNorm(..., &residual)` primitive, so the two
// branches are bit-identical (tests/vt/test_ops_fused_chain.cpp).
void AddRmsNorm(Dev d, Tensor& out, Tensor& x, const Tensor& w, Tensor& residual,
                const vt::RmsNormArgs& nargs, double eps) {
  if (FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, out, x, w, &residual, vt::kFusedAddRmsNormStd,
                   static_cast<float>(eps));
  } else {
    vt::RmsNorm(d.q, out, x, w, nargs, &residual);
  }
}

// Pack a host f32 vector into `dt` and upload it. The staging buffer is kept
// alive across an explicit Synchronize: `DBuf`'s constructor issues an ASYNC
// `cudaMemcpyAsync` (dense_device_glue.h:126, cuda_backend.cu:90) and does not
// wait, so a staging buffer that died at the end of this function would be a
// use-after-free that pageable-memory semantics happen to hide most of the
// time. This arm records no throughput number, so an explicit sync per upload
// costs nothing it is measuring.
DBuf UploadAs(Dev d, const std::vector<float>& v, DType dt,
              const std::vector<int64_t>& shape) {
  const int64_t n = NumelOf(shape);
  VT_CHECK(static_cast<int64_t>(v.size()) == n,
           "NemotronH device: upload element count does not match the shape");
  std::vector<uint8_t> staging(static_cast<size_t>(n) * vt::SizeOf(dt));
  if (dt == DType::kF32) {
    std::memcpy(staging.data(), v.data(), staging.size());
  } else {
    auto* dst = reinterpret_cast<uint16_t*>(staging.data());
    for (int64_t i = 0; i < n; ++i) dst[i] = vt::F32ToBF16(v[static_cast<size_t>(i)]);
  }
  DBuf b(d, dt, shape, staging.data());
  d.b.Synchronize(d.q);
  return b;
}

// Download a device buffer and widen it to f32, the comparison currency every
// host-side entry point in nemotron_h_forward.h already speaks.
std::vector<float> DownloadF32(Dev d, DBuf& b, DType dt, int64_t n) {
  std::vector<uint8_t> staging(static_cast<size_t>(n) * vt::SizeOf(dt));
  b.Download(d, staging.data());  // Copy + Synchronize (dense_device_glue.h:158)
  std::vector<float> out(static_cast<size_t>(n));
  if (dt == DType::kF32) {
    std::memcpy(out.data(), staging.data(), staging.size());
  } else {
    const auto* src = reinterpret_cast<const uint16_t*>(staging.data());
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  }
  return out;
}

// Refuse by name when a weight the device arm is about to upload is absent or
// mis-shaped. Mirrors the host arm's `RequireWeight` (nemotron_h.cpp:180) —
// same three properties, so a defect that would refuse on the host refuses here
// too rather than reaching a kernel with a null pointer.
void RequireDeviceWeight(const OwnedTensor& w, const char* what, DType want,
                         const std::vector<int64_t>& shape) {
  VT_CHECK(!w.Empty(),
           std::string("NemotronH device forward: weight '") + what +
               "' is not materialized");
  VT_CHECK(w.dtype == want, std::string("NemotronH device forward: weight '") + what +
                                "' has the wrong dtype for this arm");
  VT_CHECK(w.rank == static_cast<int>(shape.size()) &&
               std::equal(shape.begin(), shape.end(), w.shape),
           std::string("NemotronH device forward: weight '") + what +
               "' has the wrong shape");
}

// ─── the model-local attention block ────────────────────────────────────────
//
// One NemotronH GQA self-attention block (nemotron_h.py:415-486 @ 555967922).
// `normed` is the already-normed hidden [T,H] in `adt` on the device; returns
// the o_proj output [T,H] in `adt` on the device.
//
// DELTAS vs `dense_attn::AttnBlock`, each one measured, not assumed:
//   (a) NO POSITIONAL EMBEDDING. `nemotron_h.py` contains zero occurrences of
//       `rope`/`rotary`/`Rotary`; `.forward` (:473-486) is qkv -> split -> attn
//       -> o_proj with no rotation step. `kNemotronHAttentionHasNoRope` states
//       it and the mutation gate proves the property is armed NUMERICALLY —
//       applying a rotation here changes no tensor shape and need not move a
//       token on a short prompt.
//   (b) SEPARATE q/k/v, not a merged qkv owner. The checkpoint ships
//       `q_proj`/`k_proj`/`v_proj` as three tensors (upstream fuses them at
//       load through its stacked-params mapping); `EnumerateNemotronHTensors`
//       already claims them separately, so there is nothing to merge here.
//   (c) NO per-head q/k RMSNorm. `NemotronHAttention.__init__` (:415-471)
//       builds `qkv_proj`, `o_proj` and `Attention` and nothing else.
//   (d) NO biases — `attention_bias=false` on this checkpoint and the loader
//       ships no q/k/v/o bias tensor.
//   (e) NON-PAGED. No `PagedKvCache`, no `ReshapeAndCache`, no slot mapping: a
//       dense causal `vt::Attention` over the whole [T,·], which is exactly
//       what the host reference does (nemotron_h.cpp:615-626). That is what
//       keeps this arm inside G-SAFE, and it is A2-P that makes it paged.
//   (f) Every geometry and epsilon comes from `NemotronHParams`, never from
//       `HfConfig` — the defect class of #810 and #941.
DBuf NemotronHAttnBlock(Dev d, const NemotronHAttentionWeights& w,
                        const NemotronHParams& params, const Tensor& normed,
                        int64_t T, DType adt) {
  const int64_t H = params.hidden_size;
  const int64_t Hq = params.num_attention_heads;
  const int64_t Hkv = params.num_key_value_heads;
  const int64_t Dh = params.head_dim;
  const int64_t qdim = params.q_proj_out_features();
  const int64_t kvdim = params.kv_proj_out_features();

  // The same two refusals the host arm raises, in the same order, so an
  // unsupported checkpoint fails identically on both arms.
  VT_CHECK(!params.attention_bias,
           "NemotronH device forward: attention_bias is not ported (the "
           "checkpoint has attention_bias=false and ships no q/k/v/o bias)");
  VT_CHECK(!params.sliding_window.has_value(),
           "NemotronH device forward: per-layer sliding_window is not ported "
           "(this checkpoint ships sliding_window=null)");

  RequireDeviceWeight(w.q_proj, "mixer.q_proj", adt, {qdim, H});
  RequireDeviceWeight(w.k_proj, "mixer.k_proj", adt, {kvdim, H});
  RequireDeviceWeight(w.v_proj, "mixer.v_proj", adt, {kvdim, H});
  RequireDeviceWeight(w.o_proj, "mixer.o_proj", adt, {H, qdim});

  // `nk` IS CONSUMED HERE, exactly as the host arm consumes it
  // (nemotron_h.cpp:306). All four projections below go through `vt::MatmulBT`,
  // which reads `b` as [N=out, K=in] — the raw torch-Linear orientation
  // `nk = true` names. A weight recorded as [K, N] is a transposed GEMM operand
  // with the same shape and the same byte count, so nothing else here would
  // notice it.
  VT_CHECK(w.q_proj.nk && w.k_proj.nk && w.v_proj.nk && w.o_proj.nk,
           "NemotronH device forward: an attention projection is not in the "
           "[out, in] torch-Linear orientation vt::MatmulBT consumes");

  // THE RESIDENCY SEAM. Each of the four uploads ONCE, on the first step, and
  // every later step reuses the same device allocation. This is the shared
  // `dense_attn::ResidentWeight` (dense_attn_block.h:178) and not a local
  // equivalent: a `d_dev` bolted onto NemotronHOwned would have been the
  // parallel path AGENTS.md forbids, even though it is the smaller diff.
  Tensor wq = ResidentWeight(d, w.q_proj);
  Tensor wk = ResidentWeight(d, w.k_proj);
  Tensor wv = ResidentWeight(d, w.v_proj);
  Tensor wo = ResidentWeight(d, w.o_proj);

  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kvdim});
  DBuf v(d, adt, {T, kvdim});
  vt::MatmulBT(d.q, q.t(), normed, wq);
  vt::MatmulBT(d.q, k.t(), normed, wk);
  vt::MatmulBT(d.q, v.t(), normed, wv);

  // (a) NO RoPE HERE. This gap is the port, not an omission.

  DBuf attn(d, adt, {T, Hq, Dh});
  {
    Tensor qt = Reshape(q.t(), {T, Hq, Dh});
    Tensor kt = Reshape(k.t(), {T, Hkv, Dh});
    Tensor vt_ = Reshape(v.t(), {T, Hkv, Dh});
    vt::AttentionArgs args;
    // `self.scaling = self.head_dim**-0.5` (nemotron_h.py:440) — the SAME
    // expression the host arm evaluates (nemotron_h.cpp:622), in f64 before the
    // narrowing, so the two arms feed `vt::Attention` bit-identical scales.
    args.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(Dh)));
    args.causal = true;
    // VT-ATTN-NAIVE: the DEVICE half of the pair above. `NemotronHAttentionMixer`
    // in nemotron_h.cpp is the host reference, and this arm exists to be
    // bit-comparable to it — same scale, same rounding, same kernel. A fast rung
    // here would make the two arms measure different things instead of the same
    // one (#1544).
    vt::Attention(d.q, attn.t(), qt, kt, vt_, args);
  }

  DBuf out(d, adt, {T, H});
  {
    Tensor at = Reshape(attn.t(), {T, qdim});
    vt::MatmulBT(d.q, out.t(), at, wo);
  }
  return out;
}

// Refuse by name unless every expert projection is the NVFP4 W4A16 g16 form the
// arena is built from. The synthetic `BuildTiny` fixture is all `kDense`, so
// this is what keeps it on the host reference arm instead of reaching a kernel
// with garbage.
bool MoeIsNvfp4(const NemotronHMoeWeights& w) {
  auto q = [](const NemotronHOwned& t) {
    return t.form == NemotronHWeightForm::kNvfp4W4A16G16 && !t.bytes.empty() &&
           !t.scale.empty();
  };
  if (w.experts.empty()) return false;
  for (const NemotronHExpertWeights& e : w.experts) {
    if (!q(e.up_proj) || !q(e.down_proj)) return false;
  }
  if (w.has_shared && (!q(w.shared.up_proj) || !q(w.shared.down_proj))) return false;
  return true;
}

// DSR-ALLOW(A2-Q2a): TYPES, not behaviour -- the whole arena region names vt::cuda::Marlin* functions declared only in the guarded header above, so it cannot compile on a build without them. vt::OpRegistered answers availability, never declaration. Mirrors laguna.cpp:456, the same CUDA-leg arena.
#ifdef VT_MARLIN_NVFP4

// Fetch (building on first use) the resident state a weight owns. A VERBATIM
// copy of qwen3_5.cpp:680, which is file-local `static` there and so cannot be
// called from here. The duplication is deliberate and is the same call this file
// already makes for `FusedChainAdoptEnabled` (:125): hoisting a helper out of
// qwen3_5.cpp into a shared header is a tree-wide change that would put an
// unrelated diff on this row's critical path. What matters is the PROPERTY, and
// it is identical — the state is keyed on the slot the weights own, never on
// their address (issue #237).
template <typename R>
R& ResidentIn(const ResidentSlot& slot) {
  static std::mutex mu;
  std::lock_guard<std::mutex> lk(mu);
  if (!slot.state) slot.state = std::make_shared<R>();
  return *static_cast<R*>(slot.state.get());
}

// ─── A2-Q2a: one MoE layer's device-resident Marlin arena ───────────────────
//
// THE SHAPE IS AN ARENA, NOT A POINTER ARRAY, and that distinction is the whole
// size of this work. `vt::MoeGroupedGemmBf16` (ops.h:1642) takes an `[E]` i64
// device array of per-expert `[K,N]` pointers and needs no repack.
// `vt::MoeGroupedGemmNvfp4Marlin` (ops.h:1685) takes a rank-3 STRIDED arena
// validated at ops.cpp:884, so every expert must be repacked into Marlin's
// interleaved layout at load. The two are not interchangeable.
//
// Sizes at this checkpoint's geometry (H=2688, I=1856, E=128), per MoE layer:
//   w_up   [E, H/16=168, I*2=3712] i32  319.0 MB     s_up   [E,168,1856] i8  39.9 MB
//   w_down [E, I/16=116, H*2=5376] i32  319.3 MB     s_down [E,116,2688] i8  39.9 MB
// = 718 MB per layer x 23 layers = 16.5 GB device-resident, plus 11.2 MB per
// layer for the shared expert's E=1 slice.
//
// PEAK IS THE ARENA PLUS 2.8 MB, NOT PLUS THE RAW TOWER. qwen3_5.cpp:5751 and
// laguna.cpp:638 upload each expert's packed fp4 through `ResidentNvfp4`, which
// CACHES the device copy on the weight, so both accumulate the whole raw tower
// and free it in a tail sweep after the loop (qwen3_5.cpp:5820-5857). The A2-Q2
// spec's §3 read that as "freed as the repack proceeds"; it is not, and a
// whole-model loop written from that reading would peak at 16.5 + 15.8 = 32 GB
// on a box that reboots rather than OOM-kills. This build instead streams each
// expert through ONE REUSED 2.8 MB staging pair, so there is no tower to free
// and no accumulation to get wrong.
struct NemotronHMoeMarlinResident {
  void* w_up = nullptr;    // i32 [E, H/16, I*2]
  void* s_up = nullptr;    // fp8 [E, H/16, I]
  void* g_up = nullptr;    // f32 [E]
  void* w_down = nullptr;  // i32 [E, I/16, H*2]
  void* s_down = nullptr;  // fp8 [E, I/16, H]
  void* g_down = nullptr;  // f32 [E]
  // The shared expert, as an E=1 slice of the same machinery — the documented
  // dense route (dense_nvfp4_gemm.h:38-43: "the SINGLE-EXPERT grouped GEMM is
  // how a dense [M,K]x[N,K]^T W4A16 linear runs on the MoE Marlin entry point",
  // which is also how vLLM reaches the same csrc kernel). Using it here means
  // NO second mechanism, no `Nvfp4Weight` copy of the 23 shared pairs, and
  // therefore no change to the pinned `rep.host_bytes`.
  void* sw_up = nullptr;   // i32 [1, H/16, Is*2]
  void* ss_up = nullptr;   // fp8 [1, H/16, Is]
  void* sg_up = nullptr;   // f32 [1]
  void* sw_down = nullptr;  // i32 [1, Is/16, H*2]
  void* ss_down = nullptr;  // fp8 [1, Is/16, H]
  void* sg_down = nullptr;  // f32 [1]
  void* workspace = nullptr;  // i32 [sms*4] reduction locks
  int sms = 0;
  bool ready = false;
};


// Upload one DENSE `NemotronHOwned` to the device, refusing by name on the same
// three properties `RequireWeight` checks on the host arm (nemotron_h.cpp:180),
// so an absent or mis-shaped router weight fails identically on both arms rather
// than reaching a kernel with a null pointer. Uploaded per call, not resident:
// the router gate is 1.4 MB and A2-Q2a makes no speed claim (:39).
DBuf UploadOwned(Dev d, const NemotronHOwned& w, const char* what, DType want,
                 const std::vector<int64_t>& shape) {
  VT_CHECK(!w.Empty(),
           std::string("NemotronH device moe: weight '") + what + "' is not materialized");
  VT_CHECK(w.IsDense(),
           std::string("NemotronH device moe: weight '") + what +
               "' is not dense; the router is bf16/f32 on this checkpoint, never quantized");
  VT_CHECK(w.dtype == want, std::string("NemotronH device moe: weight '") + what +
                                "' has the wrong dtype for this arm");
  VT_CHECK(w.shape == shape,
           std::string("NemotronH device moe: weight '") + what + "' has the wrong shape");
  DBuf b(d, want, shape, w.bytes.data());
  d.b.Synchronize(d.q);  // `w.bytes` outlives this, but the copy is async; see UploadAs
  return b;
}

// Repack ONE projection into `dst_w`/`dst_s`, streaming its packed codes and
// group scales through the caller's reused staging buffers. `K`/`N` are the
// LOGICAL in/out features: the on-disk weight is [N, K/2] packed and
// [N, K/16] scales, which is exactly what `MarlinRepackExpertWeight` and
// `MarlinProcessExpertScales` read (marlin_repack.h:15,:21).
void RepackOne(Dev d, const NemotronHOwned& src, void* stage_w, void* stage_s,
               uint32_t* dst_w, uint8_t* dst_s, int K, int N, float sf) {
  const size_t packed_b = static_cast<size_t>(N) * (static_cast<size_t>(K) / 2);
  const size_t scale_b = static_cast<size_t>(N) * (static_cast<size_t>(K) / 16);
  VT_CHECK(src.bytes.size() == packed_b,
           "NemotronH MoE repack: packed byte count does not match [N, K/2]");
  VT_CHECK(src.scale.size() == scale_b,
           "NemotronH MoE repack: group-scale byte count does not match [N, K/16]");
  // Copy then repack on the SAME stream, so this expert's repack reads its
  // staging bytes before the next expert's copy overwrites them — the ordering
  // qwen3_5.cpp:5765 relies on for its fused-w13 staging.
  d.b.Copy(d.q, stage_w, src.bytes.data(), packed_b);
  d.b.Copy(d.q, stage_s, src.scale.data(), scale_b);
  vt::cuda::MarlinRepackExpertWeight(d.q.handle, d.q.device.index, dst_w,
                                     static_cast<const uint8_t*>(stage_w), K, N);
  vt::cuda::MarlinProcessExpertScales(d.q.handle, static_cast<const uint8_t*>(stage_s), dst_s,
                                      K, N, sf);
}

// `combined_scale_factor` over every expert sharing ONE Marlin GEMM — up
// together, down together, mirroring qwen3_5.cpp:5716's gu/dn split. It reads
// HOST scale buffers (marlin_repack.h:38), which is what `NemotronHOwned::scale`
// already is, so nothing round-trips.
float CombinedSfOver(const NemotronHMoeWeights& w, bool up) {
  std::vector<const uint8_t*> bufs;
  std::vector<size_t> lens;
  for (const NemotronHExpertWeights& e : w.experts) {
    const NemotronHOwned& t = up ? e.up_proj : e.down_proj;
    bufs.push_back(t.scale.data());
    lens.push_back(t.scale.size());
  }
  return vt::cuda::MarlinNvfp4CombinedScaleFactor(bufs, lens);
}

void BuildNemotronHMoeMarlinResident(Dev d, const NemotronHMoeWeights& w,
                                     const NemotronHParams& params,
                                     NemotronHMoeMarlinResident& mr) {
  if (mr.ready) return;
  const int E = static_cast<int>(params.n_routed_experts);
  const int H = static_cast<int>(params.hidden_size);
  const int I = static_cast<int>(params.moe_intermediate_size);
  const int Is = static_cast<int>(params.moe_shared_expert_intermediate_size *
                                  params.n_shared_experts);

  mr.sms = vt::cuda::MarlinDeviceSms(d.q.device.index);

  const size_t wu_i32 = static_cast<size_t>(H / 16) * (static_cast<size_t>(I) * 2);
  const size_t wd_i32 = static_cast<size_t>(I / 16) * (static_cast<size_t>(H) * 2);
  const size_t su_b = static_cast<size_t>(H / 16) * static_cast<size_t>(I);
  const size_t sd_b = static_cast<size_t>(I / 16) * static_cast<size_t>(H);

  mr.w_up = d.b.Alloc(static_cast<size_t>(E) * wu_i32 * 4);
  mr.s_up = d.b.Alloc(static_cast<size_t>(E) * su_b);
  mr.g_up = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.w_down = d.b.Alloc(static_cast<size_t>(E) * wd_i32 * 4);
  mr.s_down = d.b.Alloc(static_cast<size_t>(E) * sd_b);
  mr.g_down = d.b.Alloc(static_cast<size_t>(E) * sizeof(float));
  mr.workspace = d.b.Alloc(static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));

  // ONE staging pair, sized for the LARGEST projection this layer repacks (the
  // shared expert's, at Is > I), reused by every expert.
  const size_t stage_w_b = static_cast<size_t>(Is > I ? Is : I) * static_cast<size_t>(H) / 2;
  const size_t stage_s_b = static_cast<size_t>(Is > I ? Is : I) * static_cast<size_t>(H) / 16;
  void* stage_w = d.b.Alloc(stage_w_b > (static_cast<size_t>(H) * static_cast<size_t>(Is) / 2)
                                ? stage_w_b
                                : static_cast<size_t>(H) * static_cast<size_t>(Is) / 2);
  void* stage_s = d.b.Alloc(stage_s_b > (static_cast<size_t>(H) * static_cast<size_t>(Is) / 16)
                                ? stage_s_b
                                : static_cast<size_t>(H) * static_cast<size_t>(Is) / 16);

  const float sf_up = CombinedSfOver(w, /*up=*/true);
  const float sf_down = CombinedSfOver(w, /*up=*/false);

  std::vector<float> gu(static_cast<size_t>(E)), gd(static_cast<size_t>(E));
  for (int e = 0; e < E; ++e) {
    const size_t se = static_cast<size_t>(e);
    const NemotronHExpertWeights& x = w.experts[se];
    RepackOne(d, x.up_proj, stage_w, stage_s, static_cast<uint32_t*>(mr.w_up) + se * wu_i32,
              static_cast<uint8_t*>(mr.s_up) + se * su_b, H, I, sf_up);
    RepackOne(d, x.down_proj, stage_w, stage_s, static_cast<uint32_t*>(mr.w_down) + se * wd_i32,
              static_cast<uint8_t*>(mr.s_down) + se * sd_b, I, H, sf_down);
    gu[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(x.up_proj.global_scale, sf_up);
    gd[se] = vt::cuda::MarlinNvfp4ProcessGlobalScale(x.down_proj.global_scale, sf_down);
  }
  d.b.Copy(d.q, mr.g_up, gu.data(), gu.size() * sizeof(float));
  d.b.Copy(d.q, mr.g_down, gd.data(), gd.size() * sizeof(float));

  if (w.has_shared) {
    const size_t swu_i32 = static_cast<size_t>(H / 16) * (static_cast<size_t>(Is) * 2);
    const size_t swd_i32 = static_cast<size_t>(Is / 16) * (static_cast<size_t>(H) * 2);
    const size_t ssu_b = static_cast<size_t>(H / 16) * static_cast<size_t>(Is);
    const size_t ssd_b = static_cast<size_t>(Is / 16) * static_cast<size_t>(H);
    mr.sw_up = d.b.Alloc(swu_i32 * 4);
    mr.ss_up = d.b.Alloc(ssu_b);
    mr.sg_up = d.b.Alloc(sizeof(float));
    mr.sw_down = d.b.Alloc(swd_i32 * 4);
    mr.ss_down = d.b.Alloc(ssd_b);
    mr.sg_down = d.b.Alloc(sizeof(float));
    // Its own combined scale factor: it is its own one-expert GEMM, so it shares
    // the factor with nobody (the E=1 case of qwen3_5.cpp:5716).
    std::vector<const uint8_t*> ub{w.shared.up_proj.scale.data()};
    std::vector<size_t> ul{w.shared.up_proj.scale.size()};
    std::vector<const uint8_t*> db{w.shared.down_proj.scale.data()};
    std::vector<size_t> dl{w.shared.down_proj.scale.size()};
    const float sf_su = vt::cuda::MarlinNvfp4CombinedScaleFactor(ub, ul);
    const float sf_sd = vt::cuda::MarlinNvfp4CombinedScaleFactor(db, dl);
    RepackOne(d, w.shared.up_proj, stage_w, stage_s, static_cast<uint32_t*>(mr.sw_up),
              static_cast<uint8_t*>(mr.ss_up), H, Is, sf_su);
    RepackOne(d, w.shared.down_proj, stage_w, stage_s, static_cast<uint32_t*>(mr.sw_down),
              static_cast<uint8_t*>(mr.ss_down), Is, H, sf_sd);
    const float g_su = vt::cuda::MarlinNvfp4ProcessGlobalScale(w.shared.up_proj.global_scale, sf_su);
    const float g_sd =
        vt::cuda::MarlinNvfp4ProcessGlobalScale(w.shared.down_proj.global_scale, sf_sd);
    d.b.Copy(d.q, mr.sg_up, &g_su, sizeof(float));
    d.b.Copy(d.q, mr.sg_down, &g_sd, sizeof(float));
  }

  d.b.Memset(d.q, mr.workspace, 0, static_cast<size_t>(mr.sms) * 4 * sizeof(int32_t));
  d.b.Synchronize(d.q);  // every repack has landed -> the staging pair is dead
  d.b.Free(stage_w);
  d.b.Free(stage_s);
  mr.ready = true;
}

// The E=1 grouped GEMM every dense NVFP4 projection here runs on: all `M` rows
// route to expert 0. Buffers are pooled `DBuf`s rather than a process-lifetime
// cache — A2-Q2a makes NO speed claim of any kind (the same posture A2-R took,
// :39), and a per-token-count static cache is state whose lifetime A2-P would
// have to revisit the moment this arm is reached from production.
DBuf DenseMarlinE1(Dev d, const Tensor& x, void* w, void* s, void* g, void* ws, int sms,
                   int64_t M, int64_t K, int64_t N) {
  const int Mi = static_cast<int>(M);
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(Mi, 1, 1);
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(Mi, 1, 1, block, &max_tok, &max_blk);
  DBuf ids(d, DType::kI32, {M});
  d.b.Memset(d.q, ids.t().data, 0, static_cast<size_t>(M) * sizeof(int32_t));  // all -> expert 0
  DBuf sorted(d, DType::kI32, {max_tok});
  DBuf experts(d, DType::kI32, {max_blk});
  DBuf npad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(d.q.handle, static_cast<const int32_t*>(ids.t().data), Mi, 1, 1,
                                    block, static_cast<int32_t*>(sorted.t().data),
                                    static_cast<int32_t*>(experts.t().data),
                                    static_cast<int32_t*>(npad.t().data));
  // `mul_topk_weights` is false, so these are read by nothing; the op still
  // requires an f32 tensor of the right extent.
  const std::vector<float> ones(static_cast<size_t>(M), 1.0F);
  DBuf tw(d, DType::kF32, {M}, ones.data());
  DBuf out(d, DType::kBF16, {M, N});
  Tensor wq = MakeTensor(w, DType::kI32, d.q.device, {1, K / 16, N * 2});
  Tensor sc = MakeTensor(s, DType::kI8, d.q.device, {1, K / 16, N});
  Tensor gg = MakeTensor(g, DType::kF32, d.q.device, {1});
  Tensor wst = MakeTensor(ws, DType::kI32, d.q.device, {sms * 4});
  vt::MoeGroupedGemmNvfp4Marlin(
      d.q, out.t(), x, wq, sc, gg, wst, sorted.t(), experts.t(), npad.t(), tw.t(),
      vt::MoeMarlinArgs{block, 1, Mi, static_cast<int>(N), static_cast<int>(K), false});
  return out;
}

// ONE NemotronH MoE block on the device. Statement for statement the host arm's
// composition (nemotron_h.cpp:689-824), with the per-(token,slot) MatmulBT loop
// replaced by the grouped Marlin GEMM and NOTHING ELSE changed:
//
//   f32 router GEMM -> MoeRouterTopK(sigmoid, grouped, bias, factor 1.0)
//     -> moe_align -> grouped GEMM (up) -> MoeRelu2 -> grouped GEMM (down)
//     -> shared expert (E=1) -> MoeCombine(routed_scale)
//
// The three properties the whole block exists to gate, unchanged from the host
// arm and each carried by an argument rather than reimplemented:
//   * the router runs in f32 — MIRRORED from `force_fp32_compute=True`
//     (nemotron_h.py:150-156), not a local precision choice;
//   * `routed_scaling_factor` reaches `MoeCombine`'s `routed_scale`, so it
//     multiplies the ROUTED sum on the OUTPUT; the ROUTER's own factor is forced
//     to 1.0 in exactly this case (layer.py:291-300);
//   * the shared expert is added UNSCALED, after the routed sum is scaled
//     (moe_runner.py:402-406 then :722-725) — which is what `MoeCombine`'s
//     `shared` argument does (ops.h:2438-2446).
DBuf NemotronHMoeBlockDevice(Dev d, const NemotronHMoeWeights& w,
                             const NemotronHParams& params, const Tensor& dh, int64_t T) {
  const int64_t H = params.hidden_size;
  const int64_t E = params.n_routed_experts;
  const int64_t Kk = params.num_experts_per_tok;
  const int64_t I = params.moe_intermediate_size;
  const int64_t P = T * Kk;

  VT_CHECK(!params.moe_latent_size.has_value(),
           "NemotronH device moe: moe_latent_size is out of scope "
           "(fc1_latent_proj/fc2_latent_proj); it is null in the released checkpoint");
  VT_CHECK(static_cast<int64_t>(w.experts.size()) == E,
           "NemotronH device moe: expert count does not match n_routed_experts");
  VT_CHECK(dh.dtype == DType::kBF16,
           "NemotronH device moe: the Marlin arm requires a bf16 activation "
           "(ops.cpp:879), which is the released checkpoint's model dtype");

  NemotronHMoeMarlinResident& mr = ResidentIn<NemotronHMoeMarlinResident>(w.moe_marlin);
  // ── LAZY, AND EXPLICITLY TRANSITIONAL (A2-P owns moving it) ────────────────
  // The A2-Q2 spec's §4.2 puts this repack in `PrepareNemotronHForCausalLM`,
  // because a 16.5 GB allocation inside a forward would land inside a CUDA-graph
  // capture. That reason is FORWARD-LOOKING and is false today: nothing captures
  // `NemotronHDeviceForward`, which has no production caller at all
  // (`ForwardNemotronHForCausalLM` still routes to the host reference,
  // nemotron_h_registry.cpp:185-187). Putting it in `Prepare` NOW would instead
  // make every production NemotronH engine load pay 16.5 GB of device memory for
  // a path nothing reaches — `ModelRegistry::Prepare` is called unconditionally
  // from both `GPUModelRunner` constructors (runner.cpp:414, :455) — on a
  // unified-memory box that REBOOTS rather than OOM-kills.
  //
  // "Nothing lands dead" covers an unreached FORWARD, which costs nothing. It
  // does not cover an unreached ALLOCATION inside a REACHED hook. So A2-Q2a
  // builds on first use and `PrepareNemotronHForCausalLM` stays a no-op.
  //
  // THIS IS NOT THE INTENDED END STATE. A2-P moves it to `Prepare` at exactly
  // the moment §4.2's justification stops being false — when a production caller
  // and a capture both exist. Do not read a lazy build here as a decision that
  // the forward is the right home for it.
  if (!mr.ready) BuildNemotronHMoeMarlinResident(d, w, params, mr);

  // --- router. f32 END TO END, exactly as the host arm does it
  // (nemotron_h.cpp:712-733): the activation reaching the router is the
  // MODEL-DTYPE one widened, never a separately-computed f32 activation, so the
  // bf16 rounding the host arm applies first is applied here too by construction
  // (`dh` is already bf16).
  DBuf gate = UploadOwned(d, w.gate, "mixer.gate.weight", DType::kF32, {E, H});
  DBuf bias = UploadOwned(d, w.e_score_correction_bias,
                          "mixer.gate.e_score_correction_bias", DType::kF32, {E});
  DBuf hf32(d, DType::kF32, {T, H});
  vt::CastF32(d.q, hf32.t(), dh);
  DBuf logits(d, DType::kF32, {T, E});
  vt::MatmulBT(d.q, logits.t(), hf32.t(), gate.t());

  DBuf topk_w(d, DType::kF32, {T, Kk});
  DBuf topk_id(d, DType::kI32, {T, Kk});
  {
    vt::MoeRouterTopKArgs args;
    args.top_k = static_cast<int>(Kk);
    args.renormalize = params.norm_topk_prob;
    args.scoring_func = vt::MoeScoringFunc::kSigmoid;  // nemotron_h.py:225
    args.num_expert_group = static_cast<int>(params.n_group);
    args.topk_group = static_cast<int>(params.topk_group);
    // NOT params.routed_scaling_factor — see the block comment above.
    args.routed_scaling_factor = 1.0f;
    Tensor bt = bias.t();
    vt::MoeRouterTopK(d.q, topk_w.t(), topk_id.t(), logits.t(), args, &bt);
  }

  // --- moe_align over the router's top-k ids.
  const int Ti = static_cast<int>(T), Hi = static_cast<int>(H), Ii = static_cast<int>(I);
  const int tki = static_cast<int>(Kk), Ei = static_cast<int>(E), Pi = static_cast<int>(P);
  const int block = vt::cuda::MarlinMoeAlignBlockSizeSelect(Ti, tki, Ei);
  int max_tok = 0, max_blk = 0;
  vt::cuda::MarlinMoeAlignSizes(Ti, tki, Ei, block, &max_tok, &max_blk);
  DBuf sorted(d, DType::kI32, {max_tok});
  DBuf expert_ids(d, DType::kI32, {max_blk});
  DBuf npad(d, DType::kI32, {1});
  vt::cuda::MarlinMoeAlignBlockSize(d.q.handle, static_cast<const int32_t*>(topk_id.t().data), Ti,
                                    tki, Ei, block, static_cast<int32_t*>(sorted.t().data),
                                    static_cast<int32_t*>(expert_ids.t().data),
                                    static_cast<int32_t*>(npad.t().data));

  Tensor wst = MakeTensor(mr.workspace, DType::kI32, d.q.device, {mr.sms * 4});

  // --- up, relu², down. NON-GATED: `ckpt_names=("up_proj","down_proj","")`
  // (nemotron_h.py:220), so there is no gate half, the fused
  // `kMoeGroupedGemmBf16GateUpSilu` does not apply, and the activation is
  // `vt::MoeRelu2` — which exists FOR this architecture (ops.h:1728-1738).
  DBuf dup(d, DType::kBF16, {P, I});
  {
    Tensor wq = MakeTensor(mr.w_up, DType::kI32, d.q.device, {E, H / 16, I * 2});
    Tensor sc = MakeTensor(mr.s_up, DType::kI8, d.q.device, {E, H / 16, I});
    Tensor gg = MakeTensor(mr.g_up, DType::kF32, d.q.device, {E});
    vt::MoeGroupedGemmNvfp4Marlin(d.q, dup.t(), dh, wq, sc, gg, wst, sorted.t(), expert_ids.t(),
                                  npad.t(), topk_w.t(),
                                  vt::MoeMarlinArgs{block, tki, Ti, Ii, Hi, false});
  }
  DBuf dact(d, DType::kBF16, {P, I});
  vt::MoeRelu2(d.q, dact.t(), dup.t());
  DBuf ddown(d, DType::kBF16, {P, H});
  {
    Tensor wq = MakeTensor(mr.w_down, DType::kI32, d.q.device, {E, I / 16, H * 2});
    Tensor sc = MakeTensor(mr.s_down, DType::kI8, d.q.device, {E, I / 16, H});
    Tensor gg = MakeTensor(mr.g_down, DType::kF32, d.q.device, {E});
    vt::MoeGroupedGemmNvfp4Marlin(d.q, ddown.t(), dact.t(), wq, sc, gg, wst, sorted.t(),
                                  expert_ids.t(), npad.t(), topk_w.t(),
                                  vt::MoeMarlinArgs{block, 1, Pi, Hi, Ii, false});
  }

  // --- shared expert (nemotron_h.py:176-190): the SAME non-gated shape, at
  // moe_shared_expert_intermediate_size * n_shared_experts.
  DBuf shared_out(d, DType::kBF16, {T, H});
  bool have_shared = false;
  if (w.has_shared) {
    VT_CHECK(params.n_shared_experts > 0,
             "NemotronH device moe: shared expert weights present but n_shared_experts is 0");
    const int64_t Is = params.moe_shared_expert_intermediate_size * params.n_shared_experts;
    DBuf su = DenseMarlinE1(d, dh, mr.sw_up, mr.ss_up, mr.sg_up, mr.workspace, mr.sms, T, H, Is);
    DBuf sa(d, DType::kBF16, {T, Is});
    vt::MoeRelu2(d.q, sa.t(), su.t());
    shared_out = DenseMarlinE1(d, sa.t(), mr.sw_down, mr.ss_down, mr.sg_down, mr.workspace, mr.sms,
                               T, Is, H);
    have_shared = true;
  }

  DBuf out(d, DType::kBF16, {T, H});
  {
    Tensor eo = Reshape(ddown.t(), {T, Kk, H});
    Tensor st = shared_out.t();
    vt::MoeCombine(d.q, out.t(), eo, topk_w.t(), have_shared ? &st : nullptr,
                   static_cast<float>(params.routed_scaling_factor));
  }
  return out;
}

#endif  // VT_MARLIN_NVFP4

}  // namespace

// ─── the per-block equivalence seam ─────────────────────────────────────────

std::vector<float> NemotronHAttnBlockHostIO(const NemotronHAttentionWeights& w,
                                            const NemotronHParams& params,
                                            const std::vector<float>& hidden_normed,
                                            int64_t num_tokens, DType act_dtype,
                                            Queue& dev_queue) {
  const int64_t T = num_tokens;
  const int64_t H = params.hidden_size;
  VT_CHECK(T > 0, "NemotronH device attention: empty token sequence");
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NemotronH device attention: hidden size mismatch");
  VT_CHECK(act_dtype == DType::kBF16 || act_dtype == DType::kF32,
           "NemotronH device attention: the model dtype must be bf16 or f32");
  VT_CHECK(dev_queue.device.type != vt::DeviceType::kCPU,
           "NemotronH device attention: this is the DEVICE arm and requires a "
           "non-CPU queue; the host reference is NemotronHAttentionMixer");

  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};
  // Round the input through `act_dtype` on the way in, exactly as the host arm
  // does with `PackF32` (nemotron_h.cpp:610). Feeding the device f32 values the
  // host arm would have rounded first is the kind of "more precise" deviation
  // that makes an equivalence gate quietly meaningless.
  DBuf x = UploadAs(d, hidden_normed, act_dtype, {T, H});
  DBuf out = NemotronHAttnBlock(d, w, params, x.t(), T, act_dtype);
  return DownloadF32(d, out, act_dtype, T * H);
}

std::vector<float> NemotronHMoeBlockDeviceHostIO(const NemotronHMoeWeights& w,
                                                 const NemotronHParams& params,
                                                 const std::vector<float>& hidden_normed,
                                                 int64_t num_tokens, DType act_dtype,
                                                 Queue& dev_queue) {
  const int64_t T = num_tokens;
  const int64_t H = params.hidden_size;
  VT_CHECK(T > 0, "NemotronH device moe: empty token sequence");
  VT_CHECK(static_cast<int64_t>(hidden_normed.size()) == T * H,
           "NemotronH device moe: hidden size mismatch");
  VT_CHECK(dev_queue.device.type != vt::DeviceType::kCPU,
           "NemotronH device moe: this is the DEVICE arm and requires a non-CPU "
           "queue; the host reference is NemotronHMoeMixer");
// DSR-ALLOW(A2-Q2a): TYPES, not behaviour -- this arm calls NemotronHMoeBlockDevice, which does not EXIST without the guarded region. It carries an #else that refuses by name, so a build without Marlin reports the missing arm rather than silently computing on the host.
#ifdef VT_MARLIN_NVFP4
  VT_CHECK(act_dtype == DType::kBF16,
           "NemotronH device moe: the NVFP4 Marlin arm requires act_dtype=bf16 "
           "(ops.cpp:879), which is the released checkpoint's model dtype");
  VT_CHECK(MoeIsNvfp4(w),
           "NemotronH device moe: this layer's experts are not NVFP4 W4A16 g16, "
           "which is the only form A2-Q2a's arena is built from");
  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};
  // Round the input through act_dtype on the way in, exactly as the host arm
  // does with `PackF32` (nemotron_h.cpp:759). Feeding the device f32 values the
  // host arm would have rounded first is the deviation that makes an
  // equivalence gate quietly meaningless.
  DBuf x = UploadAs(d, hidden_normed, act_dtype, {T, H});
  DBuf out = NemotronHMoeBlockDevice(d, w, params, x.t(), T);
  return DownloadF32(d, out, act_dtype, T * H);
#else
  (void)w;
  (void)params;
  (void)act_dtype;
  VT_CHECK(false,
           "NemotronH device moe: this build has no Marlin NVFP4 grouped GEMM "
           "(VT_MARLIN_NVFP4 is off), so the device MoE arm is not compiled in");
  return {};
#endif
}


// ═══ A2-Q2b: `lm_head` on the DEVICE, through the shared NVFP4 dense route ═══
//
// .agents/specs/nemotron-h-a2q2b-realckpt-lmhead.md, issue #810.
//
// ── WHY THIS IS THE LAST HOST RE-EXPANSION, MEASURED ────────────────────────
//
// `lm_head` is [131072, 2688] NVFP4 W4A16 g16. On the HOST arm every decode
// step widens it whole through `NemotronHOwned::DenseBf16` — 352 321 536
// elements, a 704.6 MB transient bf16 buffer, for ONE GEMM against ONE row.
//
// Counted at the single dequant seam on the real checkpoint, one decode step
// (T=1, top_k=6, 23 MoE layers), the host arm re-expands 3 078 512 640
// elements. `lm_head` is 11.44% of that, and it is 28.35% of the 1 242 587 136
// that remain once A2-Q2a moved the MoE arm to the device. It is 352 321 536
// of 352 321 536 — everything left — once the mamba arm follows. It is also
// the largest SINGLE re-expansion in the model by 12.7x: 352.3e6 elements in
// one call, against 27.7e6 for the next largest (mamba `in_proj`). On a
// unified-memory box that transient is the allocation that matters.
//
// ── THE RESIDENCY DECISION (spec `## 5. Owed`), TAKEN EXPLICITLY ────────────
//
// The spec's `## 5. Owed` required this row to CHOOSE rather than default —
// "A2-Q2a routed around it by never calling either `MarlinDenseResidentFor`;
// `lm_head` must do the same or say why not" — because that accessor keys its
// repack cache on the weight's ADDRESS (issue #984) and NemotronH is exactly
// the second-consumer condition an address key cannot survive.
//
// Chosen: route through the shared seam `dense_nvfp4::MatmulNvfp4W4A16D`, and
// hand it a resident this model OWNS, in the `lm_head_marlin` `ResidentSlot`
// on the weights. That is the property `qwen3_5.cpp` gets by hand-rolling its
// own Marlin path, reached WITHOUT hand-rolling one, because
// the seam was extended to accept a caller-owned resident rather than forked.
// A2-Q2a made the same call for the MoE arena; this keeps the two arms
// consistent, and it leaves #984 exactly as it was for every other caller
// (which is where its own two-engine red-before belongs).

// A transient `Nvfp4Weight` VIEW over the host `lm_head` payload.
//
// UNGUARDED, unlike the arena helpers above, because it names nothing that the
// Marlin build adds: `Nvfp4Weight` comes from qwen3_5_weights.h and
// `OwnedBytes` from the loader, both unconditional, and no `vt::cuda::` symbol
// appears below. A `#ifdef` here would have been a device-specific reference
// paying for nothing — the one thing `scripts/check-device-leakage.py` counts.
// It has external linkage at `namespace vllm` scope, so a build without
// VT_MARLIN_NVFP4, where its single call site below is compiled out, emits no
// unused-function diagnostic.
//
// It BORROWS: `Nvfp4Weight` wants `OwnedTensor`s and `NemotronHOwned` already
// holds the packed codes and the group scales in exactly the layout
// `BuildMarlinDenseResident` reads ([N, K/2] and [N, K/16], torch [out, in]).
// Copying would add 373 MB of host residency to a load whose report is pinned
// to the byte (`rep.host_bytes == 18888922112`,
// test_nemotron_h_loader.cpp:310) and whose whole design is that a quantized
// weight KEEPS its quantized memory format.
//
// The keep-alive is a no-op deleter, and that is correct rather than lazy:
// `OwnedBytes::Borrow` rejects a null owner because a borrowed view must not
// outlive its bytes, and here the bytes are the caller's `NemotronHHostWeights`,
// which outlive this stack temporary by construction — the value is consumed by
// the repack inside this same call and never stored.
Nvfp4Weight LmHeadNvfp4View(const NemotronHOwned& w, int64_t V, int64_t H) {
  VT_CHECK(w.form == NemotronHWeightForm::kNvfp4W4A16G16,
           "NemotronH device lm_head: this arm is NVFP4 W4A16 g16 only");
  VT_CHECK(w.shape.size() == 2 && w.shape[0] == V && w.shape[1] == H,
           "NemotronH device lm_head: weight shape is not [vocab_size, hidden_size]");
  VT_CHECK(w.bytes.size() == static_cast<size_t>(V) * static_cast<size_t>(H) / 2,
           "NemotronH device lm_head: packed payload is not [vocab_size, hidden_size/2]");
  VT_CHECK(w.scale.size() == static_cast<size_t>(V) * static_cast<size_t>(H) / 16,
           "NemotronH device lm_head: group scales are not [vocab_size, hidden_size/16]");
  // A keep-alive that owns nothing: see the note above.
  static const std::shared_ptr<const void> kNoopOwner(reinterpret_cast<const void*>(1),
                                                      [](const void*) {});
  Nvfp4Weight nw;
  nw.packed.bytes = OwnedBytes::Borrow(w.bytes.data(), w.bytes.size(), kNoopOwner);
  nw.packed.dtype = DType::kI8;
  nw.packed.rank = 2;
  nw.packed.shape[0] = V;
  nw.packed.shape[1] = H / 2;
  nw.scale.bytes = OwnedBytes::Borrow(w.scale.data(), w.scale.size(), kNoopOwner);
  nw.scale.dtype = DType::kI8;
  nw.scale.rank = 2;
  nw.scale.shape[0] = V;
  nw.scale.shape[1] = H / 16;
  nw.scale2 = w.global_scale;
  nw.n = V;
  nw.k = H;
  nw.group_size = 16;
  nw.is_mxfp4 = false;
  // W4A16, never true-W4A4: the released checkpoint quantizes no activation, so
  // `alpha` stays 0 and `IsTrueW4A4()` is false. The seam asserts this too; the
  // fields are left at their defaults deliberately rather than set to 0 twice.
  return nw;
}

// True when the DEVICE `lm_head` arm can serve this weight on this queue.
//
// TWO clauses, and the split between them is the point:
//   * `dense_nvfp4::MarlinW4A16Selects` is the shared dispatcher's OWN gate,
//     called rather than restated. It carries VT_MARLIN_NVFP4, the op table's
//     realization for THIS device (an availability question, not a `== kCUDA`
//     one), `MarlinW4A16Enabled()` (the VT_NVFP4_MARLIN A/B escape hatch), and
//     Marlin's bf16 a/c contract.
//   * the weight is actually NVFP4 — the one clause that is THIS model's and
//     not the dispatcher's (a dense `lm_head` is a different arm, and a
//     synthetic fixture ships one).
//
// ★ IT WAS A RESTATEMENT AND THE RESTATEMENT WAS WRONG. The first submission
// listed the dispatcher's clauses by hand and dropped `MarlinW4A16Enabled()`,
// so under an explicit `VT_NVFP4_MARLIN=0` this predicate said eligible, the
// seam fell to the naive redundant-dequant arm, and `LmHeadNvfp4View`'s
// transient weight made `ResidentNvfp4`'s cache miss every time — re-uploading
// the whole [131072, 2688] operand on EVERY decode step. The value is
// unchanged, so a token gate sees nothing. Calling the seam's own predicate is
// what makes the two unable to disagree.
//
// Anything else falls through to `NemotronHHostLmHead`, which refuses BY NAME
// on a non-CPU queue rather than computing on the wrong operand.
bool DeviceLmHeadEligible(Dev d, const NemotronHHostWeights& host, DType adt) {
  return host.lm_head.form == NemotronHWeightForm::kNvfp4W4A16G16 &&
         dense_nvfp4::MarlinW4A16Selects(d, adt);
}

// A device-side row gather, the same helper every other model's logits path
// carries verbatim (opt.cpp:229, qwen3.cpp, gemma3.cpp:283). It exists here
// because the host arm's gather was a `std::memcpy` out of a DOWNLOADED f32
// copy of `final_normed`; the device arm must never make that download, which
// is the whole point of moving the projection.
void GatherRowsD(Dev d, void* dst, const Tensor& src, const std::vector<int64_t>& idx,
                 int64_t row_elems) {
  const size_t rb = static_cast<size_t>(row_elems) * vt::SizeOf(src.dtype);
  auto* dp = static_cast<char*>(dst);
  const auto* sp = static_cast<const char*>(src.data);
  for (size_t s = 0; s < idx.size(); ++s)
    d.b.Copy(d.q, dp + s * rb, sp + static_cast<size_t>(idx[s]) * rb, rb);
}

// y[R, vocab] f32 = gathered[R, hidden] bf16 @ dequant(lm_head).T, on the device.
//
// `out_dtype` is f32 and that is the CARRIER, not a widened compute: Marlin's
// output is bf16 (`c_type=kBFloat16`) and the seam upcasts it, so this is the
// same value the bf16 GEMM produced. f32 is what `ForwardLogits` and the
// on-device sampler take, exactly as every other model's device logits do
// (opt.cpp:308, minicpm3.cpp:303).
DBuf DeviceLmHeadD(Dev d, const NemotronHHostWeights& host,
                   const NemotronHParams& params, const Tensor& gathered) {
// DSR-ALLOW(A2-Q2b): TYPES, not behaviour -- this body names two symbols that do not EXIST without the guarded arena region above: the `ResidentIn` template, which is defined INSIDE that region, and `dense_nvfp4::MarlinDenseResident`, which dense_nvfp4_gemm.h DECLARES unconditionally but DEFINES only under VT_MARLIN_NVFP4, so the reference `ResidentIn` returns cannot bind to an incomplete type. The SELECTION is already a runtime op-table query: `DeviceLmHeadEligible` calls `dense_nvfp4::MarlinW4A16Selects`, which carries NO guard, and only this call site needs one. It has an #else that refuses BY NAME, so a build without Marlin reports the missing arm rather than silently computing on the host. Same shape and same reason as the `NemotronHMoeBlockDeviceD` body and the `moe_on_device` dispatch branches.
#ifdef VT_MARLIN_NVFP4
  const int64_t V = params.vocab_size;
  const int64_t H = params.hidden_size;
  VT_CHECK(gathered.rank == 2 && gathered.shape[1] == H,
           "NemotronH device lm_head: gathered rows are not [R, hidden_size]");
  VT_CHECK(gathered.shape[0] > 0, "NemotronH device lm_head: no rows requested");
  const Nvfp4Weight nw = LmHeadNvfp4View(host.lm_head, V, H);
  // SLOT-KEYED, never the header's address-keyed static (#984, and the `## 5.
  // Owed` bullet in the spec that asks this arm to route around it or say why
  // not).
  dense_nvfp4::MarlinDenseResident& mr =
      ResidentIn<dense_nvfp4::MarlinDenseResident>(host.lm_head_marlin);
  // ★ A FALLBACK HERE IS A DEFECT, NOT A SLOWER ANSWER, so it refuses by name.
  //
  // `DeviceLmHeadEligible` asked `MarlinW4A16Selects` and was told yes. Asking
  // the SAME predicate again on the operand actually about to be handed over is
  // what catches an eligibility answer taken against a different dtype or a
  // different queue. If it were ever false here the dispatcher would take its
  // naive redundant-dequant arm, and the consequence is invisible to every
  // value-based gate: that arm computes the SAME logits, while re-uploading
  // 198.18 MB — the whole [131072, 2688] packed operand plus its group scales —
  // on EVERY call, because `LmHeadNvfp4View` hands out a stack temporary and
  // `ResidentNvfp4` caches on `w.d_packed`, a member of the weight it was given.
  //
  // The predicate and not the seam's `fallback_gemms` counter, deliberately.
  // `MutableW4A16Stats()` is a plain non-atomic static shared by every consumer
  // in the process, so a counter window would refuse a correct run whenever
  // anything else took a fallback GEMM concurrently — a false refusal is worse
  // than the silence it replaces. The counter is the right instrument in a test,
  // where single-threadedness is a property of the harness, and that is where
  // `tests/vllm/models/test_nemotron_h_moe_device.cpp` asserts it.
  VT_CHECK(dense_nvfp4::MarlinW4A16Selects(d, gathered.dtype),
           "NemotronH device lm_head: the shared NVFP4 W4A16 dispatcher would take "
           "its redundant-dequant fallback on this queue and dtype although the "
           "device arm was selected — the fallback re-uploads the whole lm_head on "
           "every call, so this refuses rather than computing the right answer the "
           "wrong way");
  return dense_nvfp4::MatmulNvfp4W4A16D(d, gathered, nw, DType::kF32, &mr);
#else
  (void)d;
  (void)host;
  (void)params;
  (void)gathered;
  VT_CHECK(false,
           "NemotronH device lm_head: this build has no Marlin NVFP4 GEMM "
           "(VT_MARLIN_NVFP4 is off), so the device lm_head arm is not compiled in");
  return DBuf(d, DType::kF32, {1, 1});
#endif
}

// ─── the hybrid forward ─────────────────────────────────────────────────────

std::vector<float> NemotronHDeviceForward(const NemotronHHostWeights& host,
                                          const NemotronHParams& params,
                                          const std::vector<int32_t>& token_ids,
                                          const std::vector<int32_t>& logits_indices,
                                          Queue& dev_queue, Queue& host_queue,
                                          NemotronHTrace* trace) {
  VT_CHECK(host.materialized,
           "NemotronH device forward: host weights are not materialized");
  VT_CHECK(dev_queue.device.type != vt::DeviceType::kCPU,
           "NemotronH device forward: `dev_queue` must be a non-CPU queue");
  VT_CHECK(host_queue.device.type == vt::DeviceType::kCPU,
           "NemotronH device forward: `host_queue` must be a CPU queue — it is "
           "what the 23 Mamba2 blocks, the 23 MoE blocks and lm_head run on");

  const DType adt = host.act_dtype;
  VT_CHECK(adt == DType::kBF16 || adt == DType::kF32,
           "NemotronH device forward: the model dtype must be bf16 or f32");
  const int64_t H = params.hidden_size;
  const int64_t V = params.vocab_size;
  const int64_t L = params.num_hidden_layers();
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(T > 0, "NemotronH device forward: empty token sequence");
  VT_CHECK(static_cast<int64_t>(host.layers.size()) == L,
           "NemotronH device forward: host layer count != layers_block_type length");

  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};

  // --- the embedding lookup, on the device, off the resident table.
  RequireDeviceWeight(host.embeddings, "backbone.embeddings.weight", adt, {V, H});
  RequireDeviceWeight(host.norm_f, "backbone.norm_f.weight", adt, {H});
  DBuf residual(d, adt, {T, H});
  {
    std::vector<int32_t> ids = token_ids;
    for (int32_t id : ids) {
      VT_CHECK(id >= 0 && id < V, "NemotronH device forward: token id out of range");
    }
    DBuf it(d, DType::kI32, {T}, ids.data());
    d.b.Synchronize(d.q);  // `ids` is a local; see UploadAs for why this waits.
    Tensor tab = ResidentWeight(d, host.embeddings);
    vt::Embedding(d.q, residual.t(), tab, it.t());
  }

  vt::RmsNormArgs nargs;
  nargs.eps = static_cast<float>(params.layer_norm_epsilon);
  nargs.gemma = false;

  if (trace != nullptr && trace->capture) {
    trace->normed.assign(static_cast<size_t>(L), {});
    trace->mixer.assign(static_cast<size_t>(L), {});
    trace->hidden.assign(static_cast<size_t>(L), {});
  }

  // The single-branch pre-norm stream (nemotron_h.py:625-640). Layer 0 sees
  // `residual is None`, so the embedding IS the residual and the norm is
  // un-fused; every later layer folds the previous mixer output into the
  // residual inside the norm. Identical control flow to the host arm
  // (nemotron_h.cpp:872-...), which is what makes the two comparable.
  DBuf carry(d, adt, {T, H});
  for (int64_t l = 0; l < L; ++l) {
    const NemotronHLayerWeights& lw = host.layers[static_cast<size_t>(l)];
    VT_CHECK(lw.block == params.layers_block_type[static_cast<size_t>(l)],
             "NemotronH device forward: host layer block kind disagrees with "
             "layers_block_type");
    RequireDeviceWeight(lw.norm, "layer norm", adt, {H});

    DBuf normed(d, adt, {T, H});
    {
      Tensor wt = ResidentWeight(d, lw.norm);
      if (l == 0) {
        // `residual is None` (nemotron_h.py:627-631): the embedding IS the
        // residual, so there is no add to fuse and this is not a fusion site.
        vt::RmsNorm(d.q, normed.t(), residual.t(), wt, nargs, nullptr);
      } else {
        Tensor rt = residual.t();
        Tensor xt = carry.t();
        Tensor ot = normed.t();
        AddRmsNorm(d, ot, xt, wt, rt, nargs, params.layer_norm_epsilon);
      }
    }

    // The 6 attention layers stay on the device end to end. The other 46
    // bounce: download the normed hidden, run the HOST mixer on `host_queue`,
    // upload the result. One helper, one place, so the scaffold is visible and
    // deletable rather than scattered through the loop.
    // A2-Q2a: a MoE layer whose experts are NVFP4 runs on the DEVICE now, so it
    // needs no host bounce. A `kDense` MoE layer (the synthetic `BuildTiny`
    // fixture, and any future unquantized NemotronH) still bounces, because the
    // arena is built from the NVFP4 form alone — the fallback is stated here
    // rather than discovered as a silent slow path.
    // NO `#ifdef` HERE. This site only SELECTS a path, and every term it reads
    // is available in every build: `MoeIsNvfp4` names only NemotronHWeightForm,
    // and `vt::OpRegistered` IS the op/provider table's own answer to "is the
    // Marlin arm realized for this device". Asking the table rather than the
    // preprocessor is what `check-device-leakage.py` asks for, and it is
    // correct by construction here -- a build without VT_MARLIN_NVFP4 does not
    // register kMoeGroupedGemmNvfp4Marlin, so this resolves false on exactly the
    // builds the guard used to exclude.
    const bool moe_on_device =
        lw.block == NemotronHBlock::kMoe && adt == DType::kBF16 && MoeIsNvfp4(lw.moe) &&
        vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type);
    std::vector<float> nvec;
    const bool needs_host =
        lw.block != NemotronHBlock::kAttention && !moe_on_device;
    if (needs_host || (trace != nullptr && trace->capture)) {
      nvec = DownloadF32(d, normed, adt, T * H);
    }

    // Assigned STRAIGHT INTO `carry`, which is the only thing that reads the
    // mixer output. A `DBuf mixer_out(d, adt, {T, H})` declared here and
    // move-assigned over on both branches was one dead device allocation per
    // layer per step; the previous `carry` block still returns to the pool at
    // exactly the same statement it did before, after every enqueue for this
    // layer, so the lifetimes are unchanged. No speed claim is made or implied.
    std::vector<float> mvec;
    if (lw.block == NemotronHBlock::kAttention) {
      carry = NemotronHAttnBlock(d, lw.attn, params, normed.t(), T, adt);
      if (trace != nullptr && trace->capture) {
        mvec = DownloadF32(d, carry, adt, T * H);
      }
// DSR-ALLOW(A2-Q2a): TYPES, not behaviour -- same call, same reason: the symbol is absent without the guarded region. The SELECTION is already a runtime op-table query (moe_on_device above); only the call site needs the build guard. Mirrors laguna.cpp:1187, a dispatch branch with an else fallback.
#ifdef VT_MARLIN_NVFP4
    } else if (moe_on_device) {
      // ENG-HYBRID-PLACEMENT W3d: through the shared seam, inert by construction
      // when this layer is not placed. This file uses the SHARED `dense_attn`
      // glue, so it takes the plain spelling rather than the templated one.
      carry = vllm::RunMoePlaced(
          d, l, normed.t(), T, params.hidden_size,
          [&](Dev p, const Tensor& h) {
            return NemotronHMoeBlockDevice(p, lw.moe, params, h, T);
          });
      if (trace != nullptr && trace->capture) {
        mvec = DownloadF32(d, carry, adt, T * H);
      }
#endif
    } else {
      switch (lw.block) {
        case NemotronHBlock::kMamba:
          mvec = NemotronHMamba2Mixer(lw.mamba, params, nvec, T, adt, host_queue);
          break;
        case NemotronHBlock::kMoe:
          mvec = NemotronHMoeMixer(lw.moe, params, nvec, T, adt, host_queue);
          break;
        case NemotronHBlock::kMlp:
          mvec = NemotronHMlpMixer(lw.mlp, params, nvec, T, adt, host_queue);
          break;
        case NemotronHBlock::kAttention:
          break;  // handled above
      }
      carry = UploadAs(d, mvec, adt, {T, H});
    }

    if (trace != nullptr && trace->capture) {
      trace->normed[static_cast<size_t>(l)] = std::move(nvec);
      // The residual AFTER this layer is what the next norm folds `carry` into.
      std::vector<float> h = DownloadF32(d, residual, adt, T * H);
      for (size_t i = 0; i < h.size(); ++i) h[i] += mvec[i];
      trace->mixer[static_cast<size_t>(l)] = std::move(mvec);
      trace->hidden[static_cast<size_t>(l)] = std::move(h);
    }
  }

  // `hidden_states, _ = self.norm_f(hidden_states, residual)` (nemotron_h.py:641).
  DBuf final_normed(d, adt, {T, H});
  {
    Tensor wt = ResidentWeight(d, host.norm_f);
    Tensor rt = residual.t();
    Tensor xt = carry.t();
    Tensor ot = final_normed.t();
    AddRmsNorm(d, ot, xt, wt, rt, nargs, params.layer_norm_epsilon);
  }
  const std::vector<float> fvec = DownloadF32(d, final_normed, adt, T * H);
  if (trace != nullptr && trace->capture) trace->final_normed = fvec;

  // --- lm_head, on the HOST, and A2-Q2b LEAVES IT THERE DELIBERATELY.
  //
  // A2-Q2b moves `lm_head` to the device in `NemotronHPagedForward` — the
  // production path — and not here. Three reasons, and the first is the one
  // that matters:
  //
  //   * THIS SEAM IS THE GATE'S OPERAND. A2-Q2b's numeric gate compares the
  //     device projection against the host projection on the SAME gathered
  //     rows. Moving both arms to the device deletes the reference the
  //     comparison is against, which is the same mistake A2-Q2a's §4.1 refused
  //     when it supplemented the weights instead of replacing them.
  //   * A2-R's token gate is attributable BECAUSE both arms end in the
  //     identical host projection: a token difference can only have come from
  //     the 6 device attention blocks and the device residual stream. That
  //     property is worth keeping on the seam that still has it. The spec
  //     required this row to DISCLOSE that the property ends where `lm_head`
  //     moves — it ends in the paged forward, and it survives here.
  //   * `NemotronHDeviceForward` has no production caller at all (:651), so
  //     leaving it costs no user-visible behaviour.
  std::vector<int64_t> want;
  if (logits_indices.empty()) {
    want.resize(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) want[static_cast<size_t>(i)] = i;
  } else {
    for (int32_t idx : logits_indices) {
      VT_CHECK(idx >= 0 && idx < T,
               "NemotronH device forward: logits index out of range");
      want.push_back(idx);
    }
  }
  std::vector<float> gathered(want.size() * static_cast<size_t>(H));
  for (size_t r = 0; r < want.size(); ++r) {
    std::memcpy(gathered.data() + r * static_cast<size_t>(H),
                fvec.data() + static_cast<size_t>(want[r]) * static_cast<size_t>(H),
                static_cast<size_t>(H) * sizeof(float));
  }
  return NemotronHHostLmHead(host, params, gathered,
                             static_cast<int64_t>(want.size()), host_queue);
}

// ═══ A2-P: the PAGED forward ════════════════════════════════════════════════
//
// .agents/specs/nemotron-h-a2p-paged-forward.md, issue #810. What is new here,
// and nowhere above, is that the caches OUTLIVE THE CALL. Everything above
// rebuilds K/V and the recurrent state from scratch on every invocation, which
// is why the G-SAFE interlock had to refuse a runner step outright.

namespace {

// The NemotronH twin of `detail::ValidateGdnStateCacheLayout`
// (qwen3_5.cpp:428-441), and it refuses BY NAME rather than sharing that one:
// the Qwen3.5 helper is keyed on a `[slots,Hv,Dv,Dk]` GDN recurrent state and
// has no conv-state opinion at all, while this architecture's pages are the
// Mamba2 pair `MakeNemotronHKVCache` declares (nemotron_h_registry.cpp:263-270)
// and must be checked against `NemotronHParams`, never against `HfConfig` —
// re-deriving a per-layer signal from the HF config IS issue #810.
//
// This runs before a single byte is read. A mis-shaped state cache reaches
// `vt::GdnStateGather` with a plausible pointer and returns finite garbage; the
// token gate cannot see it and neither can the numeric one, because both arms
// would read the same wrong rows.
int64_t ValidateNemotronHStateCacheLayout(const std::vector<GdnStateCache>& caches,
                                          const NemotronHParams& params,
                                          DType want_ssm_dtype) {
  const int64_t Cd = params.conv_dim();
  const int64_t Kw = params.conv_kernel;
  const int64_t Hh = params.mamba_num_heads;
  const int64_t P = params.mamba_head_dim;
  const int64_t N = params.ssm_state_size;
  int64_t slots = -1;
  for (const GdnStateCache& c : caches) {
    VT_CHECK(c.conv_state.rank == 3 && c.ssm_state.rank == 4,
             "NemotronH paged forward: the recurrent pages must be conv rank-3 "
             "[slots, conv_dim, conv_kernel-1] and SSM rank-4 [slots, heads, "
             "head_dim, state_size] -- the shapes MakeNemotronHKVCache declares");
    VT_CHECK(c.conv_state.shape[1] == Cd && c.conv_state.shape[2] >= Kw - 1,
             "NemotronH paged forward: conv page geometry does not match "
             "conv_dim x (conv_kernel-1)");
    VT_CHECK(c.ssm_state.shape[1] == Hh && c.ssm_state.shape[2] == P &&
                 c.ssm_state.shape[3] == N,
             "NemotronH paged forward: SSM page geometry does not match "
             "mamba_num_heads x mamba_head_dim x ssm_state_size");
    // ★ §4.4: the persistent CONV page is the CACHE dtype (bf16 on this
    // checkpoint), never widened to f32 to satisfy a kernel precondition. The
    // f32 the conv kernel wants is the TRANSIENT working row the gather
    // produces, which is what `ops.cpp:1641-1642` names as the alternative to a
    // compressed-state backend arm. Widening the page is the too-wide dtype
    // AGENTS.md names and every gate this row owns is blind to it.
    VT_CHECK(c.conv_state.dtype == DType::kBF16 || c.conv_state.dtype == DType::kF16 ||
                 c.conv_state.dtype == DType::kF32,
             "NemotronH paged forward: the conv page must be a float cache dtype");
    VT_CHECK(c.ssm_state.dtype == want_ssm_dtype,
             "NemotronH paged forward: the SSM page dtype does not match "
             "`mamba_ssm_cache_dtype` -- it is resolved INDEPENDENTLY of the "
             "model dtype (mamba_utils.py:96-107) and collapsing it to the "
             "activation dtype is a silent precision loss a token gate absorbs");
    VT_CHECK(c.conv_state.shape[0] == c.ssm_state.shape[0],
             "NemotronH paged forward: conv/SSM slot counts disagree");
    if (slots < 0) {
      slots = c.conv_state.shape[0];
    } else {
      VT_CHECK(c.conv_state.shape[0] == slots,
               "NemotronH paged forward: all recurrent layers must share one "
               "state slot count");
    }
  }
  return slots;
}

// Read a `NemotronHOwned` back out as f32, whatever dtype it holds. The inverse
// of `NemotronHOwned::FromF32`, needed because the host mixer hands its
// `final_states` back in the SSM cache dtype and `vt::GdnStateScatter` takes an
// f32 working buffer.
std::vector<float> OwnedToF32(const NemotronHOwned& w) {
  const int64_t n = w.Numel();
  std::vector<float> out(static_cast<size_t>(n));
  if (w.dtype == DType::kF32) {
    std::memcpy(out.data(), w.bytes.data(), out.size() * sizeof(float));
  } else {
    const auto* src = reinterpret_cast<const uint16_t*>(w.bytes.data());
    for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(src[i]);
  }
  return out;
}

// ─── #1157 DIAGNOSTIC (VT_NEMOTRON_H_DIAG, documented in ENVIRONMENT.md) ────
//
// Off unless `VT_NEMOTRON_H_DIAG` is set to something other than "0", and every
// download it does is inside that guard, so a production step pays nothing.
//
// It exists to answer the question no CPU gate on this model can: the runner
// hands a decode step a device-resident input id and a recurrent page, and when
// the tokens come out wrong, only the per-layer numbers say WHICH of the two the
// step actually read. On #1157 they said the carry was exact — the state
// gathered at step k+1 equalled the state written at step k, on host and on
// GB10 alike — and that layer 0's embedding row was constant across two decode
// steps that consumed different tokens. It stays for the next reader of this
// model, because the next divergence here will be diagnosed the same way.
bool NemotronHDiagEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_NEMOTRON_H_DIAG");
    return e != nullptr && e[0] != '0';
  }();
  return on;
}

// #1157 BISECT SWITCH. The device MoE arm is 23 of this model's 52 layers and
// has never run at T=1 anywhere: its own gate (test_nemotron_h_moe_device.cpp)
// exercises T=4 and T=2. Setting `VT_NEMOTRON_H_DEVICE_MOE=0` routes those
// layers back through the host reference the CPU arm already proves token-exact
// on this checkpoint, so one run says whether the device MoE is the difference.
bool NemotronHDeviceMoeEnabled() {
  static const bool on = [] {
    const char* e = std::getenv("VT_NEMOTRON_H_DEVICE_MOE");
    return e == nullptr || e[0] != '0';
  }();
  return on;
}

double DiagL2(const std::vector<float>& v, int64_t off, int64_t n) {
  double acc = 0.0;
  for (int64_t i = 0; i < n; ++i) {
    const double x = v[static_cast<size_t>(off + i)];
    acc += x * x;
  }
  return std::sqrt(acc);
}

// ─── the per-step device inputs ─────────────────────────────────────────────
//
// Uploaded ONCE per step and shared by all 6 attention layers and all 23
// recurrent layers, mirroring `dense_attn::BuildStepInputs`
// (dense_attn_block.h:266) — minus its `cos_sin` / `rope_row_idx` members,
// which this architecture has no use for.
struct NemotronHPagedStep {
  DBuf slot_mapping;      // i64 [T]         attn_meta.slot_mapping
  DBuf block_table;       // i32 [R, cols]   attn_meta.block_table_tensor
  DBuf seq_lens;          // i32 [R]
  DBuf query_start_loc;   // i32 [R+1]
  DBuf state_idx;         // i32 [R]         the recurrent slot per request
  DBuf state_has_initial; // i32 [R]         the fresh-vs-continuing mask
};

NemotronHPagedStep BuildNemotronHPagedStep(Dev d, const ModelForwardInput& input,
                                           int64_t T, int64_t state_slots) {
  const v1::CommonAttentionMetadata& am = input.attn_meta;
  const v1::GDNAttentionMetadata& gm = input.gdn_meta;
  const int64_t R = input.num_reqs;
  VT_CHECK(R >= 1, "NemotronH paged forward: num_reqs must be >= 1");
  VT_CHECK(static_cast<int64_t>(am.slot_mapping.size()) == T,
           "NemotronH paged forward: attn_meta.slot_mapping must carry one slot "
           "per token");
  VT_CHECK(static_cast<int64_t>(am.seq_lens.size()) == R,
           "NemotronH paged forward: attn_meta.seq_lens must carry one entry per "
           "request");
  VT_CHECK(static_cast<int64_t>(am.query_start_loc.size()) == R + 1,
           "NemotronH paged forward: attn_meta.query_start_loc must be [num_reqs+1]");
  const int64_t cols = am.block_table_num_cols;
  VT_CHECK(cols >= 1 && static_cast<int64_t>(am.block_table_tensor.size()) >= R * cols,
           "NemotronH paged forward: attn_meta.block_table_tensor is smaller than "
           "num_reqs x block_table_num_cols");

  // ★ THE DECODE/PREFILL CLASSIFICATION, and it is decode-FIRST.
  // `mamba_mixer2.py:758-767` splits every tensor as
  // `[num_decode_tokens, num_prefill_tokens]`, and `mamba_attn.py:523-532`
  // splits the state indices the same way. A2-P implements the ordering even
  // though at `num_reqs == 1` exactly one side is non-empty, because
  // retrofitting an ordering convention under A2-B is how the two halves come
  // to disagree (spec §4.2).
  const int64_t nd = gm.num_decodes;
  const int64_t np = gm.num_prefills;
  VT_CHECK(nd + np == R,
           "NemotronH paged forward: the GDN metadata's decode+prefill request "
           "counts do not sum to num_reqs");
  VT_CHECK(gm.num_decode_tokens + gm.num_prefill_tokens == T,
           "NemotronH paged forward: the GDN metadata's token counts do not sum "
           "to the step's token count");
  VT_CHECK(gm.non_spec_state_indices_tensor.has_value() &&
               static_cast<int64_t>(gm.non_spec_state_indices_tensor->size()) == R,
           "NemotronH paged forward: the GDN metadata carries no per-request "
           "recurrent state index (block table column 0, mamba_attn.py:513-518). "
           "Speculative decoding is not ported for this architecture (#810 W5)");
  VT_CHECK(gm.non_spec_query_start_loc.has_value() &&
               static_cast<int64_t>(gm.non_spec_query_start_loc->size()) == R + 1,
           "NemotronH paged forward: the GDN metadata carries no non-spec query "
           "offsets, so the recurrent half cannot find each request's tokens");
  VT_CHECK(gm.num_spec_decodes == 0,
           "NemotronH paged forward: speculative rows are not ported (the MTP "
           "head is #517 W5); refusing rather than decoding the drafts as "
           "ordinary tokens");

  // ★ §4.1: INDEX THROUGH THE VECTORS EVEN AT ONE REQUEST. A forward that
  // hardcodes slot 0 passes every gate A2-P owns and then fails silently under
  // A2-B, and the mutation that would catch it cannot fire because there is
  // nothing to mutate.
  std::vector<int32_t> idx(static_cast<size_t>(R));
  std::vector<int32_t> init(static_cast<size_t>(R));
  for (int64_t r = 0; r < R; ++r) {
    const int32_t s = (*gm.non_spec_state_indices_tensor)[static_cast<size_t>(r)];
    VT_CHECK(s >= 0 && s < state_slots,
             "NemotronH paged forward: recurrent state slot out of range for the "
             "allocated state cache");
    idx[static_cast<size_t>(r)] = s;
    if (r < nd) {
      // A DECODE always continues an existing sequence, which is exactly why
      // upstream leaves `has_initial_state` None on a decode-only step
      // (gdn_attn.py:405, mirrored at gdn_attn.cpp:314).
      init[static_cast<size_t>(r)] = 1;
    } else {
      VT_CHECK(gm.prefill_has_initial_state.has_value() &&
                   static_cast<int64_t>(gm.prefill_has_initial_state->size()) == np,
               "NemotronH paged forward: a prefill request carries no "
               "has_initial_state mask (mamba_attn.py:554-556)");
      init[static_cast<size_t>(r)] =
          (*gm.prefill_has_initial_state)[static_cast<size_t>(r - nd)] != 0 ? 1 : 0;
    }
  }

  NemotronHPagedStep sdi{
      DBuf(d, DType::kI64, {T}, am.slot_mapping.data()),
      DBuf(d, DType::kI32, {R, cols}, am.block_table_tensor.data()),
      DBuf(d, DType::kI32, {R}, am.seq_lens.data()),
      DBuf(d, DType::kI32, {R + 1}, am.query_start_loc.data()),
      DBuf(d, DType::kI32, {R}, idx.data()),
      DBuf(d, DType::kI32, {R}, init.data()),
  };
  // `idx` / `init` are locals and every DBuf copy above is ASYNC (see UploadAs
  // for the same hazard and the same remedy). Waiting here costs nothing this
  // unit measures: A2-P records no throughput number on any axis (spec §5).
  d.b.Synchronize(d.q);
  if (NemotronHDiagEnabled()) {
    std::fprintf(stderr, "[NH-DIAG] step T=%lld R=%lld nd=%lld np=%lld idx=[",
                 static_cast<long long>(T), static_cast<long long>(R),
                 static_cast<long long>(nd), static_cast<long long>(np));
    for (int64_t r = 0; r < R; ++r)
      std::fprintf(stderr, "%d%s", idx[static_cast<size_t>(r)], r + 1 < R ? "," : "");
    std::fprintf(stderr, "] init=[");
    for (int64_t r = 0; r < R; ++r)
      std::fprintf(stderr, "%d%s", init[static_cast<size_t>(r)], r + 1 < R ? "," : "");
    std::fprintf(stderr, "] qsl_attn=[");
    for (size_t i = 0; i < am.query_start_loc.size(); ++i)
      std::fprintf(stderr, "%d%s", am.query_start_loc[i],
                   i + 1 < am.query_start_loc.size() ? "," : "");
    std::fprintf(stderr, "] seq_lens=[");
    for (size_t i = 0; i < am.seq_lens.size(); ++i)
      std::fprintf(stderr, "%d%s", am.seq_lens[i], i + 1 < am.seq_lens.size() ? "," : "");
    std::fprintf(stderr, "]\n");
  }
  return sdi;
}

// ─── the paged attention block ──────────────────────────────────────────────
//
// `NemotronHAttnBlock` with exactly one thing replaced: the dense causal
// `vt::Attention` over the whole `[T,·]` becomes a WRITE into the runner's
// pages followed by a READ back out of them. Everything else — the three
// separate projections, the absent RoPE, the absent q/k norm, the `Dh^-0.5`
// scale, the residency seam — is byte-for-byte the same block, which is what
// makes the two directly comparable in the per-block numeric gate.
//
// Upstream does the same two steps as two separate ops in the same order:
// `unified_kv_cache_update(key, value, ...)` then
// `unified_attention_with_output(...)` (layers/attention/attention.py:544-561),
// over `reshape_and_cache_flash(...)` (v1/attention/backends/flash_attn.py:1122-1131).
DBuf NemotronHAttnBlockPaged(Dev d, const NemotronHAttentionWeights& w,
                             const NemotronHParams& params, const Tensor& normed,
                             int64_t T, DType adt, const PagedKvCache& kv,
                             const v1::CommonAttentionMetadata& meta,
                             NemotronHPagedStep& sdi) {
  const int64_t H = params.hidden_size;
  const int64_t Hq = params.num_attention_heads;
  const int64_t Hkv = params.num_key_value_heads;
  const int64_t Dh = params.head_dim;
  const int64_t qdim = params.q_proj_out_features();
  const int64_t kvdim = params.kv_proj_out_features();

  VT_CHECK(!params.attention_bias,
           "NemotronH paged forward: attention_bias is not ported (the "
           "checkpoint has attention_bias=false and ships no q/k/v/o bias)");
  VT_CHECK(!params.sliding_window.has_value(),
           "NemotronH paged forward: per-layer sliding_window is not ported "
           "(this checkpoint ships sliding_window=null)");
  VT_CHECK(kv.num_kv_heads == Hkv && kv.head_size == Dh,
           "NemotronH paged forward: the paged KV page geometry does not match "
           "num_key_value_heads x head_dim");

  RequireDeviceWeight(w.q_proj, "mixer.q_proj", adt, {qdim, H});
  RequireDeviceWeight(w.k_proj, "mixer.k_proj", adt, {kvdim, H});
  RequireDeviceWeight(w.v_proj, "mixer.v_proj", adt, {kvdim, H});
  RequireDeviceWeight(w.o_proj, "mixer.o_proj", adt, {H, qdim});
  VT_CHECK(w.q_proj.nk && w.k_proj.nk && w.v_proj.nk && w.o_proj.nk,
           "NemotronH paged forward: an attention projection is not in the "
           "[out, in] torch-Linear orientation vt::MatmulBT consumes");

  Tensor wq = ResidentWeight(d, w.q_proj);
  Tensor wk = ResidentWeight(d, w.k_proj);
  Tensor wv = ResidentWeight(d, w.v_proj);
  Tensor wo = ResidentWeight(d, w.o_proj);

  DBuf q(d, adt, {T, qdim});
  DBuf k(d, adt, {T, kvdim});
  DBuf v(d, adt, {T, kvdim});
  vt::MatmulBT(d.q, q.t(), normed, wq);
  vt::MatmulBT(d.q, k.t(), normed, wk);
  vt::MatmulBT(d.q, v.t(), normed, wv);

  // NO RoPE, and this gap is the port. `NemotronHAttentionDecoderLayer.forward`
  // accepts `positions` (nemotron_h.py:516) and never uses it; `input.positions`
  // is read by nothing in this file for the same reason.

  Tensor q3 = Reshape(q.t(), {T, Hq, Dh});
  Tensor k3 = Reshape(k.t(), {T, Hkv, Dh});
  Tensor v3 = Reshape(v.t(), {T, Hkv, Dh});

  // The "auto" ReshapeAndCache copy requires `cache dtype == k/v dtype`
  // (ops.cpp, and qwen3_5.h:47-49 records the same constraint), so down-cast
  // this step's K/V to the page's dtype and nothing else. The QUERY is not
  // cast: the attention kernel converts the cache reads up and accumulates in
  // f32 either way, so casting the query would only lose precision the page
  // never asked for.
  DBuf kcast(d, kv.dtype, {T, Hkv, Dh});
  DBuf vcast(d, kv.dtype, {T, Hkv, Dh});
  Tensor kw = k3;
  Tensor vw = v3;
  if (kv.dtype != adt) {
    if (kv.dtype == DType::kBF16) {
      vt::CastBf16(d.q, kcast.t(), k3);
      vt::CastBf16(d.q, vcast.t(), v3);
    } else if (kv.dtype == DType::kF32) {
      vt::CastF32(d.q, kcast.t(), k3);
      vt::CastF32(d.q, vcast.t(), v3);
    } else {
      VT_CHECK(false,
               "NemotronH paged forward: the paged KV page dtype is neither the "
               "model dtype nor a dtype this arm can cast to (bf16/f32). The "
               "fp8 KV scheme the checkpoint ships k_scale/v_scale for is a "
               "SEPARATE decision with its own gate and is not selected here");
    }
    kw = kcast.t();
    vw = vcast.t();
  }

  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  vt::ReshapeAndCache(d.q, kw, vw, k_cache, v_cache, sdi.slot_mapping.t());

  DBuf attn(d, adt, {T, Hq, Dh});
  {
    vt::PagedAttentionArgs pa;
    // `self.scaling = self.head_dim**-0.5` (nemotron_h.py:440), evaluated in
    // f64 before the narrowing so this arm and the host reference feed the
    // kernel a bit-identical scale.
    pa.scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(Dh)));
    pa.causal = meta.causal;
    // Host-resident grid bounds, so the prefill launchers size their query-tile
    // grid without a per-layer D2H copy plus stream synchronize. `meta` outlives
    // this call — it is the runner's own step metadata.
    pa.query_start_loc_host = meta.query_start_loc.data();
    pa.max_seq_len = meta.max_seq_len;
    vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache, sdi.block_table.t(),
                       sdi.seq_lens.t(), sdi.query_start_loc.t(), pa);
  }

  DBuf out(d, adt, {T, H});
  {
    Tensor at = Reshape(attn.t(), {T, qdim});
    vt::MatmulBT(d.q, out.t(), at, wo);
  }
  return out;
}

// ─── the recurrent half ─────────────────────────────────────────────────────
//
// One Mamba2 layer's state I/O around the HOST mixer. The COMPUTE stays on
// `NemotronHMamba2Mixer` because the block is entered through an FP8 W8A8
// `in_proj` and is not splittable (this file's header note; A2-Q1 owns that
// arm, issue #940), and the A2-P spec's §1.1 puts "any change to the FP8 mamba
// projections" explicitly out of scope. What A2-P owns is the CARRY, and it is
// the whole difference between a decode that continues a sequence and one that
// silently restarts it.
//
// ★ THE ZEROING OBLIGATION (gdn_attn.h:126-139) IS DISCHARGED BY THE GATHER,
// AND THAT IS DELIBERATE. The recurrence kernels read the state buffer
// UNCONDITIONALLY, so a request whose mask is 0 must be handed ZEROS, not the
// previous tenant's rows. `vt::GdnStateGather` fuses indexing, the
// cache-dtype -> f32 widening and that zeroing into one launch, which is
// upstream's own `torch.where(has_initial_states_p[...], ssm_state[...], 0)`
// (mamba_mixer2.py:854-866).
//
// The mixer is then told `has_initial = true` in EVERY case, including a fresh
// request. That is not a shortcut, and getting it wrong in the other direction
// is what would make the trap invisible:
//
//   * it is EXACT. With a zeroed state row, `has_initial_state = 1` and
//     `has_initial_state = 0` compute the identical answer — the conv kernel's
//     out-of-window read is `v = 0.0f` when the flag is clear and
//     `v = old_row[...]` when it is set (cpu_ops.cpp CausalConv1dFwdKernel), and
//     `old_row` is zeros. The SSD scan is the same: upstream always passes the
//     gathered-and-zeroed `initial_states` rather than passing None.
//   * coupling the mixer's flag to the mask instead would make the ZEROING
//     UNOBSERVABLE — the mixer would ignore a stale row it was handed — and
//     mutation P-M4 (drop the zeroing) would survive green. The gate would then
//     be blind to the loudest silent-wrong-answer path in this unit (spec §3.3,
//     R4).
struct NemotronHRecurrentIo {
  // The compact f32 working rows, [R, ...]. f32 by op contract on both sides:
  // `vt::GdnStateGather` requires an f32 working buffer, and the conv kernel
  // reads its state as f32 unless the backend advertises a compressed-state
  // arm. The PAGE stays at its cache dtype throughout (§4.4).
  DBuf conv;  // f32 [R, conv_dim, conv_kernel-1]
  DBuf ssm;   // f32 [R, heads, head_dim, state_size]
};

NemotronHRecurrentIo GatherNemotronHState(Dev d, const GdnStateCache& cache,
                                          const NemotronHParams& params, int64_t R,
                                          NemotronHPagedStep& sdi) {
  const int64_t Cd = params.conv_dim();
  const int64_t Kw = params.conv_kernel;
  const int64_t Hh = params.mamba_num_heads;
  const int64_t P = params.mamba_head_dim;
  const int64_t N = params.ssm_state_size;
  NemotronHRecurrentIo io{DBuf(d, DType::kF32, {R, Cd, Kw - 1}),
                          DBuf(d, DType::kF32, {R, Hh, P, N})};
  Tensor hinit = sdi.state_has_initial.t();
  vt::GdnStateGather(d.q, io.conv.t(), cache.conv_state, sdi.state_idx.t(), &hinit);
  vt::GdnStateGather(d.q, io.ssm.t(), cache.ssm_state, sdi.state_idx.t(), &hinit);
  return io;
}

void ScatterNemotronHState(Dev d, const GdnStateCache& cache, NemotronHRecurrentIo& io,
                           NemotronHPagedStep& sdi) {
  // `cache` is the runner's page and is updated IN PLACE. The mutable copies are
  // views over the same storage; `GdnStateScatter` writes only the rows named by
  // `state_idx` and leaves every other slot byte-identical, which is the
  // property that keeps two concurrent sequences from overwriting each other
  // once A2-B lifts the request count.
  Tensor conv_page = cache.conv_state;
  Tensor ssm_page = cache.ssm_state;
  vt::GdnStateScatter(d.q, conv_page, io.conv.t(), sdi.state_idx.t());
  vt::GdnStateScatter(d.q, ssm_page, io.ssm.t(), sdi.state_idx.t());
}

}  // namespace

ForwardLogits NemotronHPagedForward(const NemotronHHostWeights& host,
                                    const NemotronHParams& params,
                                    const ModelForwardInput& input,
                                    NemotronHTrace* trace) {
  VT_CHECK(host.materialized,
           "NemotronH paged forward: host weights are not materialized");
  const DType adt = host.act_dtype;
  VT_CHECK(adt == DType::kBF16 || adt == DType::kF32,
           "NemotronH paged forward: the model dtype must be bf16 or f32");
  VT_CHECK(!params.tie_word_embeddings,
           "NemotronH paged forward: tie_word_embeddings is false in the "
           "released checkpoint and the tied arm is not ported");

  const int64_t H = params.hidden_size;
  const int64_t V = params.vocab_size;
  const int64_t L = params.num_hidden_layers();
  const int64_t T = static_cast<int64_t>(input.token_ids.size());
  const int64_t R = input.num_reqs;
  VT_CHECK(T > 0, "NemotronH paged forward: empty token sequence");
  VT_CHECK(static_cast<int64_t>(host.layers.size()) == L,
           "NemotronH paged forward: host layer count != layers_block_type length");

  // The caches, matched to the two groups `MakeNemotronHKVCache` publishes in
  // exactly the order it publishes them: the full-attention group over the GQA
  // layers first, the Mamba2 recurrent group second
  // (nemotron_h_registry.cpp:235-270).
  const std::vector<int64_t> attn_layers = params.LayerIndices(NemotronHBlock::kAttention);
  const std::vector<int64_t> mamba_layers = params.LayerIndices(NemotronHBlock::kMamba);
  VT_CHECK(input.attn_kv.size() == attn_layers.size(),
           "NemotronH paged forward: the runner supplied a different number of "
           "paged KV layers than this model has full-attention layers");
  VT_CHECK(input.gdn_state.size() == mamba_layers.size(),
           "NemotronH paged forward: the runner supplied a different number of "
           "recurrent state layers than this model has Mamba2 layers");

  const DType ssm_dtype = NemotronHSsmCacheDType(params, adt);
  const int64_t state_slots =
      ValidateNemotronHStateCacheLayout(input.gdn_state, params, ssm_dtype);
  VT_CHECK(input.gdn_state_slots == 0 || input.gdn_state_slots == state_slots,
           "NemotronH paged forward: the runner's declared recurrent slot count "
           "disagrees with the allocated pages");

  vt::Queue& queue = input.queue;
  Dev d{vt::GetBackend(queue.device.type), queue};

  // The host mixers and `lm_head` need a CPU queue. When the runner is already
  // on the host that IS `input.queue`, and the paged path is then end-to-end on
  // one queue; on a device queue this is the same bounce A2-R documents at the
  // top of this file, and every later unit deletes one pair of it.
  vt::Queue host_queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr};
  vt::Queue& hq = queue.device.type == vt::DeviceType::kCPU ? queue : host_queue;

  NemotronHPagedStep sdi = BuildNemotronHPagedStep(d, input, T, state_slots);

  RequireDeviceWeight(host.embeddings, "backbone.embeddings.weight", adt, {V, H});
  RequireDeviceWeight(host.norm_f, "backbone.norm_f.weight", adt, {H});
  DBuf residual(d, adt, {T, H});
  {
    Tensor tab = ResidentWeight(d, host.embeddings);
    Tensor rt = residual.t();
    if (input.mm.has_value() && input.mm->inputs_embeds.data != nullptr) {
      // A multimodal step (`NemotronH_Nano_VL_V2`, nano_nemotron_vl.py:1462-
      // 1481): the wrapper's `embed_mm` already embedded the ids and spliced
      // the projected image rows, so the residual STARTS from those rows and
      // the ids are not looked up again (upstream passes `inputs_embeds` and
      // the language model skips `embed_input_ids`). It is a COPY: the merged
      // rows are already in the model dtype.
      const Tensor& e = input.mm->inputs_embeds;
      VT_CHECK(e.rank == 2 && e.shape[0] == T && e.shape[1] == H && e.dtype == adt,
               "NemotronH paged forward: the multimodal inputs_embeds must be "
               "[num_tokens, hidden] in the model dtype");
      d.b.Copy(d.q, rt.data, e.data, static_cast<size_t>(T * H) * vt::SizeOf(adt));
    } else if (input.device_token_ids != nullptr) {
      // ★ ENG-ASYNC-SCHED W4 (#1157). `ModelForwardInput::device_token_ids` is
      // non-null exactly when the async runner's device combine has already
      // spliced each DECODE row's sampled token into ITS device buffer and left
      // the host `token_ids` STALE on purpose — materializing it on the host is
      // the synchronize W4 exists to remove (model_registry.h:314-324,
      // runner.cpp:1175-1194). A forward that embeds the host vector therefore
      // embeds the same placeholder id on every decode step.
      //
      // That is not a hypothesis. With the host vector, this model's A3 token
      // gate read 4/24 on GB10 while the SAME binary read 24/24 in
      // fresh-prefill mode (no decode step is ever taken) and 96/96 on CPU
      // (where this pointer is always null), and the per-layer trace showed the
      // layer-0 embedding row identical across two consecutive decode steps
      // that consumed different tokens.
      //
      // Kimi-Linear was cut from this same divergence
      // (kimi_linear_device.cpp:2270-2280) and every other registered forward
      // already honours the field. This one did not, and nothing could see it:
      // the runner sets the pointer only under VLLM_CPP_CUDA with a live device
      // mirror, so no CPU gate can reach the branch at all.
      //
      // The host-side range check below is deliberately NOT repeated here. The
      // ids live on the device and validating them would need the D2H
      // synchronize this path exists to delete; `LaunchCombineSampledAndDraft
      // Tokens` produces them from the sampler's own output, and vt::Embedding
      // bounds-checks the gather.
      Tensor ids = MakeTensor(const_cast<int32_t*>(input.device_token_ids),
                              DType::kI32, d.q.device, {T});
      vt::Embedding(d.q, rt, tab, ids);
    } else {
      std::vector<int32_t> ids = input.token_ids;
      for (int32_t id : ids) {
        VT_CHECK(id >= 0 && id < V, "NemotronH paged forward: token id out of range");
      }
      DBuf it(d, DType::kI32, {T}, ids.data());
      d.b.Synchronize(d.q);  // `ids` is a local; see UploadAs.
      vt::Embedding(d.q, rt, tab, it.t());
    }
  }

  vt::RmsNormArgs nargs;
  nargs.eps = static_cast<float>(params.layer_norm_epsilon);
  nargs.gemma = false;

  if (trace != nullptr && trace->capture) {
    trace->normed.assign(static_cast<size_t>(L), {});
    trace->mixer.assign(static_cast<size_t>(L), {});
    trace->hidden.assign(static_cast<size_t>(L), {});
  }

  size_t attn_i = 0;
  size_t mamba_i = 0;
  DBuf carry(d, adt, {T, H});
  for (int64_t l = 0; l < L; ++l) {
    const NemotronHLayerWeights& lw = host.layers[static_cast<size_t>(l)];
    VT_CHECK(lw.block == params.layers_block_type[static_cast<size_t>(l)],
             "NemotronH paged forward: host layer block kind disagrees with "
             "layers_block_type");
    RequireDeviceWeight(lw.norm, "layer norm", adt, {H});

    DBuf normed(d, adt, {T, H});
    {
      Tensor wt = ResidentWeight(d, lw.norm);
      if (l == 0) {
        vt::RmsNorm(d.q, normed.t(), residual.t(), wt, nargs, nullptr);
      } else {
        Tensor rt = residual.t();
        Tensor xt = carry.t();
        Tensor ot = normed.t();
        AddRmsNorm(d, ot, xt, wt, rt, nargs, params.layer_norm_epsilon);
      }
    }

    const bool moe_on_device =
        lw.block == NemotronHBlock::kMoe && adt == DType::kBF16 && MoeIsNvfp4(lw.moe) &&
        NemotronHDeviceMoeEnabled() &&
        vt::OpRegistered(vt::OpId::kMoeGroupedGemmNvfp4Marlin, d.q.device.type);
    const bool needs_host = lw.block != NemotronHBlock::kAttention && !moe_on_device;
    std::vector<float> nvec;
    if (needs_host || (trace != nullptr && trace->capture)) {
      nvec = DownloadF32(d, normed, adt, T * H);
    }

    std::vector<float> mvec;
    if (lw.block == NemotronHBlock::kAttention) {
      carry = NemotronHAttnBlockPaged(d, lw.attn, params, normed.t(), T, adt,
                                      input.attn_kv[attn_i], input.attn_meta, sdi);
      ++attn_i;
      if (trace != nullptr && trace->capture) mvec = DownloadF32(d, carry, adt, T * H);
// DSR-ALLOW(A2-P): TYPES, not behaviour -- the same guarded call A2-Q2a introduced at :936. The SELECTION above is already a runtime op-table query; only the call site needs the build guard, because the symbol does not EXIST without the guarded region.
#ifdef VT_MARLIN_NVFP4
    } else if (moe_on_device) {
      // ENG-HYBRID-PLACEMENT W3d: through the shared seam, inert by construction
      // when this layer is not placed. This file uses the SHARED `dense_attn`
      // glue, so it takes the plain spelling rather than the templated one.
      carry = vllm::RunMoePlaced(
          d, l, normed.t(), T, params.hidden_size,
          [&](Dev p, const Tensor& h) {
            return NemotronHMoeBlockDevice(p, lw.moe, params, h, T);
          });
      if (trace != nullptr && trace->capture) mvec = DownloadF32(d, carry, adt, T * H);
#endif
    } else if (lw.block == NemotronHBlock::kMamba) {
      // ── the CARRY. This is the unit. ──
      const GdnStateCache& cache = input.gdn_state[mamba_i];
      NemotronHRecurrentIo io = GatherNemotronHState(d, cache, params, R, sdi);
      const int64_t conv_row = params.conv_dim() * (params.conv_kernel - 1);
      const int64_t ssm_row =
          params.mamba_num_heads * params.mamba_head_dim * params.ssm_state_size;
      std::vector<float> conv_all = DownloadF32(d, io.conv, DType::kF32, R * conv_row);
      std::vector<float> ssm_all = DownloadF32(d, io.ssm, DType::kF32, R * ssm_row);
      if (NemotronHDiagEnabled()) {
        std::fprintf(stderr,
                     "[NH-DIAG]   L%lld mamba GATHERED |conv|=%.6g |ssm|=%.6g\n",
                     static_cast<long long>(l), DiagL2(conv_all, 0, conv_row),
                     DiagL2(ssm_all, 0, ssm_row));
      }

      // At `num_reqs == 1` this loop runs once, and it is written as a loop for
      // the reason §4.1 gives: the indexing machinery lands here, only the
      // count is one.
      mvec.assign(static_cast<size_t>(T * H), 0.0F);
      for (int64_t r = 0; r < R; ++r) {
        NemotronHMambaState state;
        state.conv.assign(
            conv_all.begin() + static_cast<std::ptrdiff_t>(r * conv_row),
            conv_all.begin() + static_cast<std::ptrdiff_t>((r + 1) * conv_row));
        state.ssm = NemotronHOwned::FromF32(
            std::vector<float>(
                ssm_all.begin() + static_cast<std::ptrdiff_t>(r * ssm_row),
                ssm_all.begin() + static_cast<std::ptrdiff_t>((r + 1) * ssm_row)),
            ssm_dtype,
            {params.mamba_num_heads, params.mamba_head_dim, params.ssm_state_size});
        // See the block comment above `NemotronHRecurrentIo`: ALWAYS true, over
        // a row the gather has already zeroed when the mask said fresh.
        state.has_initial = true;
        // The RECURRENT half's own query offsets. In the non-spec path this is
        // `m.query_start_loc` verbatim (gdn_attn.cpp, the non-spec branch), so
        // at `num_reqs == 1` it is the same vector the attention half uses — but
        // reading the recurrent metadata for the recurrent split is what stays
        // correct when A2-B introduces a mixed batch.
        const std::vector<int32_t>& qsl = *input.gdn_meta.non_spec_query_start_loc;
        const int64_t t0 = qsl[static_cast<size_t>(r)];
        const int64_t t1 = qsl[static_cast<size_t>(r + 1)];
        VT_CHECK(t1 > t0 && t1 <= T,
                 "NemotronH paged forward: a request's query range is empty or "
                 "runs past the step's tokens");
        const std::vector<float> rows(
            nvec.begin() + static_cast<std::ptrdiff_t>(t0 * H),
            nvec.begin() + static_cast<std::ptrdiff_t>(t1 * H));
        const std::vector<float> got = NemotronHMamba2Mixer(
            lw.mamba, params, rows, t1 - t0, adt, hq, &state);
        std::copy(got.begin(), got.end(),
                  mvec.begin() + static_cast<std::ptrdiff_t>(t0 * H));
        std::copy(state.conv.begin(), state.conv.end(),
                  conv_all.begin() + static_cast<std::ptrdiff_t>(r * conv_row));
        const std::vector<float> ssm_out = OwnedToF32(state.ssm);
        VT_CHECK(static_cast<int64_t>(ssm_out.size()) == ssm_row,
                 "NemotronH paged forward: the mixer returned an SSM state of "
                 "the wrong extent");
        std::copy(ssm_out.begin(), ssm_out.end(),
                  ssm_all.begin() + static_cast<std::ptrdiff_t>(r * ssm_row));
      }

      if (NemotronHDiagEnabled()) {
        std::fprintf(stderr,
                     "[NH-DIAG]   L%lld mamba WROTE    |conv|=%.6g |ssm|=%.6g "
                     "|out|=%.6g\n",
                     static_cast<long long>(l), DiagL2(conv_all, 0, conv_row),
                     DiagL2(ssm_all, 0, ssm_row),
                     DiagL2(mvec, (T - 1) * H, H));
      }
      io.conv = UploadAs(d, conv_all, DType::kF32, {R, params.conv_dim(),
                                                    params.conv_kernel - 1});
      io.ssm = UploadAs(d, ssm_all, DType::kF32,
                        {R, params.mamba_num_heads, params.mamba_head_dim,
                         params.ssm_state_size});
      ScatterNemotronHState(d, cache, io, sdi);
      ++mamba_i;
      carry = UploadAs(d, mvec, adt, {T, H});
    } else {
      switch (lw.block) {
        case NemotronHBlock::kMoe:
          mvec = NemotronHMoeMixer(lw.moe, params, nvec, T, adt, hq);
          break;
        case NemotronHBlock::kMlp:
          mvec = NemotronHMlpMixer(lw.mlp, params, nvec, T, adt, hq);
          break;
        case NemotronHBlock::kMamba:
        case NemotronHBlock::kAttention:
          break;  // handled above
      }
      carry = UploadAs(d, mvec, adt, {T, H});
    }

    if (NemotronHDiagEnabled()) {
      const std::vector<float> cv = DownloadF32(d, carry, adt, T * H);
      const std::vector<float> rs = DownloadF32(d, residual, adt, T * H);
      const char* kind = lw.block == NemotronHBlock::kAttention ? "attn"
                         : lw.block == NemotronHBlock::kMamba   ? "mamba"
                         : lw.block == NemotronHBlock::kMoe     ? "moe"
                                                                : "mlp";
      std::fprintf(stderr,
                   "[NH-DIAG]   L%lld %-5s |mixer_last|=%.6g |resid_last|=%.6g\n",
                   static_cast<long long>(l), kind, DiagL2(cv, (T - 1) * H, H),
                   DiagL2(rs, (T - 1) * H, H));
    }
    if (trace != nullptr && trace->capture) {
      trace->normed[static_cast<size_t>(l)] = std::move(nvec);
      std::vector<float> h = DownloadF32(d, residual, adt, T * H);
      for (size_t i = 0; i < h.size(); ++i) h[i] += mvec[i];
      trace->mixer[static_cast<size_t>(l)] = std::move(mvec);
      trace->hidden[static_cast<size_t>(l)] = std::move(h);
    }
  }
  VT_CHECK(attn_i == attn_layers.size() && mamba_i == mamba_layers.size(),
           "NemotronH paged forward: the layer loop did not consume every paged "
           "KV layer and every recurrent state layer exactly once");

  DBuf final_normed(d, adt, {T, H});
  {
    Tensor wt = ResidentWeight(d, host.norm_f);
    Tensor rt = residual.t();
    Tensor xt = carry.t();
    Tensor ot = final_normed.t();
    AddRmsNorm(d, ot, xt, wt, rt, nargs, params.layer_norm_epsilon);
  }
  // ── ONE download of `final_normed`, whichever arm and whichever trace ──────
  //
  // Two consumers want these bytes and NEITHER always runs: the trace (the
  // numeric gate's operand, and nothing else reads it) and the HOST projection
  // below. Before A2-Q2b the download was unconditional because the host
  // projection always needed it. Making it conditional is the point of this row
  // — the device arm must not pay a T*H device-to-host copy per step — but the
  // first submission wrote the condition as an EXTRA download rather than as a
  // shared one, so a traced host step copied the same buffer twice.
  //
  // `have_fvec` rather than `fvec.empty()`: emptiness is a property of T*H and
  // would silently become "download again" if a caller ever asked for zero rows.
  std::vector<float> fvec;
  bool have_fvec = false;
  if (trace != nullptr && trace->capture) {
    fvec = DownloadF32(d, final_normed, adt, T * H);
    trace->final_normed = fvec;
    have_fvec = true;
  }

  // The gather-before-lm_head rows. An EMPTY `logits_indices` is the runner's
  // VT_LOGITS_GATHER=0 path and means "every row", which is also what the two
  // non-paged seams mean by it — so this branch serves both gather settings and
  // no runner step can escape the paged path on that flag.
  std::vector<int64_t> want;
  if (input.logits_indices.empty()) {
    want.resize(static_cast<size_t>(T));
    for (int64_t i = 0; i < T; ++i) want[static_cast<size_t>(i)] = i;
  } else {
    for (int32_t idx : input.logits_indices) {
      VT_CHECK(idx >= 0 && idx < T,
               "NemotronH paged forward: logits index out of range");
      want.push_back(idx);
    }
  }
  // `n_out`, not `R`: `R` is already this function's REQUEST count (:1673) and
  // these are the gathered LOGIT rows, which are not the same number.
  const int64_t n_out = static_cast<int64_t>(want.size());

  // ── A2-Q2b: THE DEVICE PROJECTION, AND THE PRODUCTION CALL SITE ───────────
  //
  // This is the line `scripts/runner-routing-allowlist.txt` was NARROWED for —
  // narrowed, not removed; the entry is still in that file and its text says
  // why.
  //
  // ★ NOTHING AUTOMATED HOLDS THIS BRANCH IN PLACE, AND THAT IS MEASURED.
  // Delete this whole `if` block and `scripts/check-runner-routing-consistency.py`
  // still exits 0 with byte-identical output ("3 host-logits off-framework
  // (3 allowlisted)"), because the allowlist entry that would have to be
  // removed first is exactly what keeps the model off the checker's red list —
  // and the entry cannot be removed while #1410 makes the checker misclassify
  // a cross-TU free-function device forward as HOST. So the checker is not a
  // guard here in either direction. An earlier draft of this comment claimed
  // the opposite; it was wrong, and it was wrong in the direction that stops a
  // reader looking for the real evidence.
  //
  // What DOES hold it is the reachability deletion mutation
  // (`.agents/reachability.md`): delete this block, run the device `lm_head`
  // gate, and it must go red. That mutation needs CUDA — every case that can
  // reach this branch is `TryCudaQueue`-gated, because `MarlinW4A16Selects` is
  // false on a CPU queue — so it is recorded PENDING a `dgx:gpu0` window in the
  // spec's `## 6. Now`, not claimed. #1410 owns making the checker able to see
  // this at all.
  if (DeviceLmHeadEligible(d, host, adt)) {
    DBuf grows(d, adt, {n_out, H});
    GatherRowsD(d, grows.ptr(), final_normed.t(), want, H);
    DBuf dlogits = DeviceLmHeadD(d, host, params, grows.t());
    ForwardLogits fl;
    fl.rows = n_out;
    fl.vocab = params.vocab_size;
    fl.device_tensor = dlogits.t();
    // The pool block's lifetime moves into a shared_ptr whose deleter returns
    // it to the DevicePool — the shared `WrapDeviceLogits` carrier every
    // device-logits model uses (opt.cpp:315, minicpm3.cpp:303).
    fl.device_storage = dlogits.ReleaseShared();
    return fl;
  }

  // The HOST projection, kept below the fold exactly as the host reference
  // forward is: it is the operand A2-Q2b's numeric gate compares against, and
  // it is what serves a build with no Marlin NVFP4 GEMM and a checkpoint whose
  // `lm_head` is dense. It refuses BY NAME on a non-CPU queue.
  if (!have_fvec) fvec = DownloadF32(d, final_normed, adt, T * H);
  std::vector<float> gathered(want.size() * static_cast<size_t>(H));
  for (size_t r = 0; r < want.size(); ++r) {
    std::memcpy(gathered.data() + r * static_cast<size_t>(H),
                fvec.data() + static_cast<size_t>(want[r]) * static_cast<size_t>(H),
                static_cast<size_t>(H) * sizeof(float));
  }
  return HostLogits(NemotronHHostLmHead(host, params, gathered, n_out, hq),
                    params.vocab_size);
}

// ─── A2-Q2b: the PUBLIC per-block entry points for `lm_head` ────────────────
//
// The twins of `NemotronHHostLmHead`, argument for argument, so the numeric
// gate compares like with like instead of against a second implementation of
// the same projection written inside the test. That symmetry is the whole
// point: this file's header says the per-block entry points "are public so the
// gate can compare a BLOCK's activations against an independent reference".
//
// THEY ARE NOT THE REACHABILITY PROOF, and nothing here should be read as one.
// Production reaches the device projection through
// `ModelRegistry::Forward` -> `ForwardNemotronHForCausalLM` ->
// `NemotronHPagedForward`, at the `DeviceLmHeadEligible` branch. These two
// localize a failure to the projection; the real-checkpoint gate is what
// enters through the production path (reachability.md, "Does a test enter
// through it?").
bool NemotronHDeviceLmHeadEligible(const NemotronHHostWeights& host, vt::DType adt,
                                   vt::Queue& dev_queue) {
  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};
  return DeviceLmHeadEligible(d, host, adt);
}

std::vector<float> NemotronHDeviceLmHead(const NemotronHHostWeights& host,
                                         const NemotronHParams& params,
                                         const std::vector<float>& gathered_normed,
                                         int64_t num_rows, vt::Queue& dev_queue) {
  const DType adt = host.act_dtype;
  const int64_t H = params.hidden_size;
  const int64_t V = params.vocab_size;
  VT_CHECK(num_rows > 0, "NemotronH device lm_head: no rows requested");
  VT_CHECK(static_cast<int64_t>(gathered_normed.size()) == num_rows * H,
           "NemotronH device lm_head: gathered row count does not match hidden_size");
  Dev d{vt::GetBackend(dev_queue.device.type), dev_queue};
  VT_CHECK(DeviceLmHeadEligible(d, host, adt),
           "NemotronH device lm_head: this arm needs an NVFP4 W4A16 g16 `lm_head`, a "
           "bf16 activation dtype, and a device whose op table realizes the Marlin "
           "NVFP4 grouped GEMM — the host arm is NemotronHHostLmHead");
  // Upload the gathered rows at the MODEL dtype, not f32: the operand Marlin
  // takes is bf16, and widening here would move twice the bytes for a value the
  // kernel narrows again on the way in.
  DBuf rows(d, adt, {num_rows, H});
  {
    // PackF32 lives in nemotron_h.cpp; the conversion is one loop and doing it
    // here avoids widening this file's dependency surface for two call sites.
    std::vector<uint16_t> bf(static_cast<size_t>(num_rows * H));
    for (size_t i = 0; i < bf.size(); ++i) bf[i] = vt::F32ToBF16(gathered_normed[i]);
    d.b.Copy(d.q, rows.ptr(), bf.data(), bf.size() * sizeof(uint16_t));
    d.b.Synchronize(d.q);
  }
  DBuf dlogits = DeviceLmHeadD(d, host, params, rows.t());
  std::vector<float> out(static_cast<size_t>(num_rows * V));
  dlogits.Download(d, out.data());
  return out;
}

}  // namespace vllm
