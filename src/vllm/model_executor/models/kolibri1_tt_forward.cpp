// Kolibri-1 — Tenstorrent B2b-i dense-resident device forward
// (MODEL-TEXT-kolibri-1-tenstorrent, spec .agents/specs/kolibri-tt.md
// ### B2 scope — B2b addendum, slice i; issue
// ISSUE-LOCAL-01M4E22DM790W0E69M07D5XA9D).
//
// The slice-i completion condition is ONE GREEDY DECODE of a golden prompt
// on the P150 with the resident non-expert set running entirely on device
// and the routed-expert tier absent. This TU is that forward: it mirrors
// the CPU row (kolibri1_forward.cpp) op-for-op — the same vt ops, the same
// order, the same shapes — over the resident slice, with the router output
// used ONLY for the shared expert plus the named unimplemented-routed-
// expert refusal (B2b-ii owns the routed path).
//
// COMPUTE DISPOSITION (the b2i precedent, restated): the device GEMMs are
// plain bf16 vt::MatmulBT over the bf16 dequant of the fp8-block weights —
// the CPU row's documented R1 disposition — dequanted ONCE per weight into
// the device-resident context (the same DequantRowsBf16 bytes the CPU row
// dequants per call; the b2i bring-up verified the device GEMM consumes
// exactly these dequants of the byte-verified staged fp8). Tile-layout
// consumption of the staged FP8_E4M3 operands is the B2b compute wave's,
// owed per the addendum's FP8/trace constraints, not assumed here.
//
// Backend-agnostic TU (vt ops + the shared residency seam only); a
// non-Tenstorrent queue is refused by name at the boundary.
#include "vllm/model_executor/models/kolibri1_tt_forward.h"

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <utility>

#include "vllm/model_executor/models/dense_attn_block.h"  // KvSlice
#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf, ResidentWeight
#include "vllm/model_executor/models/kolibri1_fp8_dequant.h"
#include "vllm/model_executor/models/kolibri1_shared.h"  // SigmoidLogitAddRouting
#include "vllm/model_executor/models/kv_cache_route.h"   // WriteKvCache
#include "vllm/model_executor/models/host_parallel.h"    // the ONE pool (#1664)
#include "vt/ops.h"

namespace vllm {

using dense_attn::DBuf;
using dense_attn::Dev;
using dense_attn::KvSlice;
using dense_attn::MakeTensor;
using dense_attn::Reshape;
using dense_attn::ResidentWeight;
using vt::DType;
using vt::Tensor;

namespace {

// The non-Tenstorrent refusal, mirroring the CPU row's kGpuRefusal polarity
// (the CPU row refuses non-CPU queues; this slice refuses non-TT queues —
// the registry dispatches CPU -> the CPU row, TT -> this slice, and refuses
// every other device by name at the registry boundary).
constexpr const char* kNonTTRefusal =
    "Kolibri1ForCausalLM: the Tenstorrent B2b-i dense-resident forward runs "
    "on a Tenstorrent queue only. Row MODEL-TEXT-kolibri-1-tenstorrent, "
    "spec .agents/specs/kolibri-tt.md ### B2 scope — B2b addendum, slice i — "
    "the CPU arm is the landed CPU row (MODEL-TEXT-kolibri-1, "
    ".agents/specs/kolibri-1-cpu.md).";

int64_t CDiv(int64_t a, int64_t b) { return (a + b - 1) / b; }

double NowSec() {
  return std::chrono::duration<double>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

bool ProgressOn() {
  static const bool on = std::getenv("VT_KOLIBRI1_TT_B2BI_PROGRESS") != nullptr;
  return on;
}

// ---- The routed-expert refusal (fires by name, counted, never thrown) ------

std::mutex& RefusalMutex() {
  static std::mutex* m = new std::mutex();  // never destroyed (#1486)
  return *m;
}
int64_t& RefusalCountRef() {
  static int64_t* n = new int64_t(0);  // never destroyed (#1486)
  return *n;
}
bool& RefusalAnnouncedRef() {
  static bool* a = new bool(false);  // never destroyed (#1486)
  return *a;
}

// Fires the named refusal for one layer's router request: increments the
// process-wide count and prints the full message ONCE per process (the
// per-layer message would flood the log 50x per step). NOT a throw: the
// slice's completion condition is the decode COMPLETING with the refusal
// active.
void NoteRoutedExpertRequest(int64_t layer,
                             const std::vector<int32_t>& requested_ids) {
  std::lock_guard<std::mutex> g(RefusalMutex());
  ++RefusalCountRef();
  if (RefusalAnnouncedRef()) return;
  RefusalAnnouncedRef() = true;
  std::fprintf(stderr, "%s\n",
               Kolibri1TTRoutedExpertRefusalMessage(layer, requested_ids)
                   .c_str());
}

// ---- The device-resident compute context ------------------------------------

// One fp8-block projection's memoized bf16 dequant: dequanted host-side
// into a backend allocation (the R1 disposition), held for the model's
// lifetime. The TT backend's Alloc returns host-authoritative registered
// memory, so the host dequant writes the bytes directly and the first
// device use stages them — the same flow the CPU row's DequantFp8Block
// uses, memoized instead of per-call. The returned pair owns the
// allocation (freed through the backend) beside its tensor view.
struct OwnedDeviceTensor {
  std::shared_ptr<void> owner;
  vt::Tensor view;
};

OwnedDeviceTensor DequantProjectionOwned(vt::Backend& be, vt::Queue& q,
                                        const Fp8BlockWeight& w,
                                        int64_t* uploaded_bytes) {
  VT_CHECK(w.n > 0 && w.k > 0 && w.block_n > 0 && w.block_k > 0,
           "kolibri1-tt forward: degenerate fp8 block weight");
  const int64_t scale_cols = CDiv(w.k, w.block_k);
  const int64_t scale_rows = CDiv(w.n, w.block_n);
  VT_CHECK(static_cast<int64_t>(w.scale.bytes.size()) ==
               scale_rows * scale_cols * 4,
           "kolibri1-tt forward: fp8 scale grid is not f32 "
           "[cdiv(n,bn), cdiv(k,bk)]");
  VT_CHECK(static_cast<int64_t>(w.packed.bytes.size()) == w.n * w.k,
           "kolibri1-tt forward: fp8 packed bytes are not [n, k]");
  const size_t nb = static_cast<size_t>(w.n * w.k) * 2;
  void* p = be.Alloc(nb);
  *uploaded_bytes += static_cast<int64_t>(nb);
  // The row is the unit of work, never a K-chunk, so the threaded dequant
  // is bit-identical to the serial loop BY CONSTRUCTION (the pool
  // determinism contract) — the same bytes the CPU row's per-call dequant
  // produces over the same packed bytes and scale grid.
  const auto* src = w.packed.bytes.data();
  const auto* sc = reinterpret_cast<const float*>(w.scale.bytes.data());
  auto* dst = static_cast<uint16_t*>(p);
  host_parallel::ForOutputRows(w.n, w.k, [&](int64_t n0, int64_t n1) {
    kolibri1_fp8::DequantRowsBf16(src, sc, scale_cols, n0, n1, w.k, w.block_n,
                                  w.block_k, dst);
  });
  vt::Backend* bk = &be;
  OwnedDeviceTensor out;
  out.owner = std::shared_ptr<void>(p, [bk](void* ptr) { bk->Free(ptr); });
  out.view = MakeTensor(p, DType::kBF16, q.device, {w.n, w.k});
  return out;
}

// ---- The linear helpers (mirror the CPU row's LinearBT arms) ----------------

// out[T, N] = x[T, K] @ W[N, K]^T over a device-resident bf16 weight.
DBuf LinearBTDevice(Dev d, const Tensor& x, const Tensor& wt, int64_t t,
                    DType out_dtype = DType::kBF16) {
  DBuf out(d, out_dtype, {t, wt.shape[0]});
  vt::MatmulBT(d.q, out.t(), x, wt);
  return out;
}

// The bf16-module arm (the router gate): the shared residency seam.
DBuf LinearBTRaw(Dev d, const Tensor& x, const OwnedTensor& wt_raw, int64_t t,
                 DType out_dtype = DType::kBF16) {
  Tensor wt = ResidentWeight(d, wt_raw);
  DBuf out(d, out_dtype, {t, wt.shape[0]});
  vt::MatmulBT(d.q, out.t(), x, wt);
  return out;
}

// SwiGLU expert MLP over device-resident bf16 weights (the routed experts
// and the shared expert share the shape; slice i computes the shared one).
DBuf ExpertMlp(Dev d, const Tensor& x, int64_t t, int64_t inter,
               const Tensor& gate_w, const Tensor& up_w,
               const Tensor& down_w) {
  DBuf g = LinearBTDevice(d, x, gate_w, t);  // [t, I]
  DBuf u = LinearBTDevice(d, x, up_w, t);    // [t, I]
  DBuf a(d, DType::kBF16, {t, inter});
  vt::MoeSiluMul(d.q, a.t(), g.t(), u.t());
  return LinearBTDevice(d, a.t(), down_w, t);  // [t, H]
}

// ---- Attention block (mirror the CPU row's AttentionBlock op-for-op) --------
DBuf AttentionBlock(Dev d, const Kolibri1AttnWeights& w, const Kolibri1Params& p,
                    bool is_sliding, const Tensor& dhn, const Tensor& positions,
                    const dense_attn::StepInputs& si, const PagedKvCache& kv,
                    int64_t t, const Tensor& q_w, const Tensor& k_w,
                    const Tensor& v_w, const Tensor& o_w) {
  const int64_t hq = p.num_attention_heads;
  const int64_t hkv = p.num_key_value_heads;
  const int64_t dh = p.head_dim;
  const float scale = static_cast<float>(1.0 / std::sqrt(static_cast<double>(dh)));

  // q/k/v projections (separate linears upstream, kolibri1.py:64-79) over
  // the device-resident bf16 dequants.
  DBuf q = LinearBTDevice(d, dhn, q_w, t);  // [T, Hq*Dh]
  DBuf k = LinearBTDevice(d, dhn, k_w, t);  // [T, Hkv*Dh]
  DBuf v = LinearBTDevice(d, dhn, v_w, t);  // [T, Hkv*Dh]

  // Per-head qk-norm BEFORE RoPE (kolibri1.py:107-118): RMSNorm(head_dim)
  // per head — reshape [T, H*Dh] -> [T*H, Dh] and run the shared RmsNorm
  // with the per-head gamma (every head shares one gamma vector).
  Tensor q3 = Reshape(q.t(), {t, hq, dh});
  Tensor k3 = Reshape(k.t(), {t, hkv, dh});
  Tensor v3 = Reshape(v.t(), {t, hkv, dh});
  {
    Tensor qh = Reshape(q.t(), {t * hq, dh});
    Tensor kh = Reshape(k.t(), {t * hkv, dh});
    Tensor qw = ResidentWeight(d, w.q_norm);
    Tensor kw = ResidentWeight(d, w.k_norm);
    DBuf qn(d, DType::kBF16, {t * hq, dh});
    DBuf kn(d, DType::kBF16, {t * hkv, dh});
    vt::RmsNorm(d.q, qn.t(), qh, qw,
                vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), false});
    vt::RmsNorm(d.q, kn.t(), kh, kw,
                vt::RmsNormArgs{static_cast<float>(p.rms_norm_eps), false});
    q3 = Reshape(qn.t(), {t, hq, dh});
    k3 = Reshape(kn.t(), {t, hkv, dh});
  }

  // RNoPE (kolibri1.py:81-95): RoPE on the SLIDING layers only;
  // full-attention layers carry NO positional encoding. Full rotary:
  // rotary_dim == head_dim.
  if (is_sliding) {
    vt::RopeArgs ra{};
    ra.base = static_cast<float>(p.rope_theta);
    ra.rotary_dim = static_cast<int>(dh);
    ra.is_neox_style = true;
    vt::RopeNeox(d.q, q3, k3, positions, ra);
  }

  // KV cache write + paged attention. Sliding layers cap the window at 513
  // (kolibri1.py:85-106): AttentionWindow{left = W-1, right = 0} is the
  // causal decoder window of W tokens (the mimo-v2 house convention).
  Tensor k_cache = KvSlice(kv, d.q.device, 0);
  Tensor v_cache = KvSlice(kv, d.q.device, 1);
  DBuf attn(d, DType::kBF16, {t, hq, dh});
  {
    vt::PagedAttentionArgs pa{};
    pa.scale = scale;
    pa.causal = true;
    if (is_sliding) {
      pa.window_size = vt::AttentionWindow{
          static_cast<int32_t>(p.sliding_window - 1), 0};
    } else {
      pa.window_size = std::nullopt;  // full attention, RNoPE or not
    }
    dense_attn::ApplyKvCacheQuant(pa, kv);
    dense_attn::WriteKvCache(d.q, kv, k3, v3, k_cache, v_cache,
                             si.slot_mapping.t());
    vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache,
                       si.block_table.t(), si.seq_lens.t(),
                       si.query_start_loc.t(), pa);
  }

  Tensor o_in = Reshape(attn.t(), {t, hq * dh});
  return LinearBTDevice(d, o_in, o_w, t);  // [T, H]
}

// ---- MoE block (slice i: router + refusal + shared expert ONLY) --------------
//
// The CPU row's MoeBlock computes shared + the weighted routed combine
// (vt::MoeCombine). Slice i has NO routed tier: the router runs on device
// (bf16 gate, f32 logits — the CPU row's contract), the sigmoid-logit-add
// top-6-of-384 runs host-side over the readback (the inherited f32 compute
// path), the routed request fires the NAMED REFUSAL, and the block returns
// the shared expert's output — the routed contribution is absent, exactly
// as the refusal records.
DBuf MoeBlock(Dev d, const Kolibri1MoeWeights& w, const Kolibri1Params& p,
              const Tensor& dhn, int64_t t, int64_t layer,
              const Tensor& sh_gate_w, const Tensor& sh_up_w,
              const Tensor& sh_down_w) {
  const int64_t e = p.num_experts;
  const int64_t top_k = p.num_experts_per_tok;

  // Router logits, f32 (kolibri1.py:171-177) — the bf16 gate over the
  // shared residency seam, f32 out like the CPU row's LinearBTRaw.
  DBuf dlog = LinearBTRaw(d, dhn, w.router_gate, t, DType::kF32);
  std::vector<float> logits(static_cast<size_t>(t * e));
  dlog.Download(d, logits.data());
  std::vector<float> bias(static_cast<size_t>(e));
  {
    Tensor bt = ResidentWeight(d, w.e_score_correction_bias, {e});
    vt::Backend& be = vt::GetBackend(d.q.device.type);
    be.Copy(d.q, bias.data(), bt.data, static_cast<size_t>(e) * sizeof(float));
    be.Synchronize(d.q);
  }

  // The inherited sigmoid-logit-add routing (kolibri1_shared.h, verbatim
  // from the CPU row): selection on logits + bias, weights = sigmoid of
  // the UNBIASED logits, no renormalisation.
  Kolibri1HostRouting route =
      SigmoidLogitAddRouting(logits, bias, t, e, top_k);

  // SLICE i: the router requested routed experts; the routed path is not
  // implemented in this slice. Fire the refusal BY NAME (counted) and
  // proceed with the shared expert only.
  NoteRoutedExpertRequest(layer, route.ids);

  // Shared expert: UNGATED, always added (kolibri1.py:146-188) — and in
  // this slice it is the WHOLE MoE output (the routed term is absent).
  return ExpertMlp(d, dhn, t, p.shared_expert_intermediate_size, sh_gate_w,
                   sh_up_w, sh_down_w);
}

}  // namespace

// ---- The routed-expert refusal (public contract) -----------------------------

std::string Kolibri1TTRoutedExpertRefusalMessage(
    int64_t layer, const std::vector<int32_t>& requested_ids) {
  std::string ids;
  for (size_t i = 0; i < requested_ids.size() && i < 8; ++i) {
    if (i != 0) ids += ",";
    ids += std::to_string(requested_ids[i]);
  }
  if (requested_ids.size() > 8) ids += ",...";
  return "Kolibri1ForCausalLM Tenstorrent B2b-i: layer " +
         std::to_string(layer) + " router selected routed experts [" + ids +
         "] but the routed-expert path is NOT IMPLEMENTED in this slice — "
         "the resident set excludes the routed tier by design. The routed "
         "experts arrive in slice B2b-ii (streaming MoE, spec "
         ".agents/specs/kolibri-tt.md ### B2 scope — B2b addendum), row "
         "MODEL-TEXT-kolibri-1-tenstorrent, issue "
         "ISSUE-LOCAL-01M4E22DM790W0E69M07D5XA9D. Proceeding with the shared "
         "expert only (the routed contribution is absent).";
}

int64_t Kolibri1TTRoutedExpertRefusalCount() {
  std::lock_guard<std::mutex> g(RefusalMutex());
  return RefusalCountRef();
}

void Kolibri1TTResetRoutedExpertRefusalCount() {
  std::lock_guard<std::mutex> g(RefusalMutex());
  RefusalCountRef() = 0;
  RefusalAnnouncedRef() = false;
}

// ---- The device-resident compute context (public contract) -------------------

std::unique_ptr<Kolibri1TTResidentDeviceContext>
BuildKolibri1TTResidentDeviceContext(vt::Backend& backend, vt::Queue& queue,
                                     const Kolibri1Weights& weights) {
  const double t0 = NowSec();
  auto ctx = std::make_unique<Kolibri1TTResidentDeviceContext>();
  ctx->layers.resize(weights.layers.size());
  ctx->keepalive.reserve(weights.layers.size() * 7);
  int64_t uploaded = 0;
  for (size_t l = 0; l < weights.layers.size(); ++l) {
    const Kolibri1LayerWeights& lw = weights.layers[l];
    Kolibri1TTResidentDeviceContext::Layer& cl = ctx->layers[l];
    auto fill = [&](const Kolibri1Projection& proj, vt::Tensor& dst) {
      if (!proj.IsFp8Block()) {
        throw std::runtime_error(
            "kolibri1-tt forward: the B2b-i resident slice's attention and "
            "shared-expert projections must be fp8-block weights (the wave-A "
            "dtype decision); a non-fp8 projection here is refused rather "
            "than silently re-armed");
      }
      OwnedDeviceTensor od =
          DequantProjectionOwned(backend, queue, proj.fp8_block, &uploaded);
      ctx->keepalive.push_back(std::move(od.owner));
      dst = od.view;
      ++ctx->projections;
    };
    fill(lw.attn.q_proj, cl.q);
    fill(lw.attn.k_proj, cl.k);
    fill(lw.attn.v_proj, cl.v);
    fill(lw.attn.o_proj, cl.o);
    fill(lw.moe.shared_experts.gate_proj, cl.sh_gate);
    fill(lw.moe.shared_experts.up_proj, cl.sh_up);
    fill(lw.moe.shared_experts.down_proj, cl.sh_down);
  }
  ctx->uploaded_bytes = uploaded;
  ctx->build_seconds = NowSec() - t0;
  ctx->built = true;
  return ctx;
}

// ---- The B2b-i forward --------------------------------------------------------

ForwardLogits ForwardKolibri1TTResidentForward(
    const std::vector<int32_t>& token_ids, const std::vector<int32_t>& positions,
    const v1::CommonAttentionMetadata& attn_meta,
    const std::vector<PagedKvCache>& attn_kv, const Kolibri1Weights& weights,
    const MultiKvCacheIndex* multi_kv, vt::Queue& queue,
    const std::vector<int32_t>& logits_indices,
    Kolibri1TTResidentDeviceContext& ctx) {
  VT_CHECK(queue.device.type == vt::DeviceType::kTENSTORRENT, kNonTTRefusal);
  VT_CHECK(ctx.built, "kolibri1-tt forward: the device context is not built");
  const Kolibri1Params& p = weights.params;
  Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t t = static_cast<int64_t>(token_ids.size());
  const int64_t h = p.hidden_size;
  const int64_t vocab = p.vocab_size;

  VT_CHECK(t > 0, "kolibri1-tt: empty token batch");
  VT_CHECK(static_cast<int64_t>(positions.size()) == t,
           "kolibri1-tt: positions size mismatch");
  VT_CHECK(static_cast<int64_t>(attn_meta.slot_mapping.size()) == t,
           "kolibri1-tt: slot_mapping size mismatch");
  VT_CHECK(static_cast<int64_t>(ctx.layers.size()) == p.num_hidden_layers,
           "kolibri1-tt: the device context does not cover every layer");
  if (ProgressOn()) {
    std::fprintf(stderr,
                 "[kolibri1-tt-b2bi] forward step: T=%lld layers=%lld "
                 "experts=%lld topk=%lld\n",
                 static_cast<long long>(t),
                 static_cast<long long>(p.num_hidden_layers),
                 static_cast<long long>(p.num_experts),
                 static_cast<long long>(p.num_experts_per_tok));
  }

  // Embedding (bf16 table, [vocab, H] raw orientation) — the shared
  // residency seam, exactly like the CPU row.
  DBuf hidden_buf(d, DType::kBF16, {t, h});
  {
    DBuf ids(d, DType::kI32, {t}, const_cast<int32_t*>(token_ids.data()));
    Tensor tab = ResidentWeight(d, weights.embed_tokens, {vocab, h});
    vt::Embedding(d.q, hidden_buf.t(), tab, ids.t());
  }
  Tensor hidden = hidden_buf.t();
  DBuf res(d, DType::kBF16, {t, h});
  res.Zero(d);

  const float eps = static_cast<float>(p.rms_norm_eps);
  std::shared_ptr<void> hidden_hold;

  // The per-step device inputs are layer-invariant: upload ONCE per step
  // (the CPU row rebuilds them per layer; the same bytes either way).
  dense_attn::StepInputs si = Kolibri1BuildStepInputs(d, attn_meta, positions);

  for (int64_t l = 0; l < p.num_hidden_layers; ++l) {
    const Kolibri1LayerWeights& lw = weights.layers[static_cast<size_t>(l)];
    const Kolibri1TTResidentDeviceContext::Layer& cl = ctx.layers[static_cast<size_t>(l)];

    const PagedKvCache* kv_ptr = nullptr;
    if (multi_kv != nullptr) {
      const std::string name =
          "model.layers." + std::to_string(l) + ".self_attn";
      const int64_t idx = multi_kv->Find(name);
      VT_CHECK(idx >= 0 && idx < static_cast<int64_t>(attn_kv.size()),
               "kolibri1-tt: KV cache not found for layer " + std::to_string(l));
      kv_ptr = &attn_kv[static_cast<size_t>(idx)];
    } else {
      VT_CHECK(l < static_cast<int64_t>(attn_kv.size()),
               "kolibri1-tt: KV cache missing for layer " + std::to_string(l));
      kv_ptr = &attn_kv[static_cast<size_t>(l)];
    }

    // input_layernorm + residual (the vLLM fused add-norm contract) — the
    // CPU row's exact conditional, so the op sequence matches its default.
    DBuf dhn(d, DType::kBF16, {t, h});
    Tensor w_in = ResidentWeight(d, lw.input_layernorm, {h});
    Tensor dhn_t = dhn.t();
    Tensor res_t = res.t();
    if (dense_attn::FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, dhn_t, hidden, w_in, &res_t,
                     vt::kFusedAddRmsNormStd, eps);
    } else {
      vt::RmsNorm(d.q, dhn_t, hidden, w_in,
                  vt::RmsNormArgs{eps, false}, &res_t);
    }

    // Attention -> post_attn_norm (no residual).
    DBuf attn = AttentionBlock(d, lw.attn, p, lw.is_sliding, dhn.t(),
                               si.positions.t(), si, *kv_ptr, t, cl.q, cl.k,
                               cl.v, cl.o);
    DBuf attn_n(d, DType::kBF16, {t, h});
    Tensor w_pa = ResidentWeight(d, lw.post_attn_norm, {h});
    vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa,
                vt::RmsNormArgs{eps, false});

    // post_attention_layernorm carries the residual: residual += the
    // POST-NORMED attention output (kolibri1.py:250).
    DBuf dh2(d, DType::kBF16, {t, h});
    Tensor w_pal = ResidentWeight(d, lw.post_attention_layernorm, {h});
    Tensor dh2_t = dh2.t();
    res_t = res.t();
    if (dense_attn::FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, dh2_t, attn_n.t(), w_pal, &res_t,
                     vt::kFusedAddRmsNormStd, eps);
    } else {
      vt::RmsNorm(d.q, dh2_t, attn_n.t(), w_pal,
                  vt::RmsNormArgs{eps, false}, &res_t);
    }

    // MoE on EVERY layer -> post_ffn_norm (no residual). Slice i: router
    // + the named routed-expert refusal + the shared expert only.
    DBuf moe = MoeBlock(d, lw.moe, p, dh2.t(), t, l, cl.sh_gate, cl.sh_up,
                        cl.sh_down);
    DBuf moe_n(d, DType::kBF16, {t, h});
    Tensor w_pf = ResidentWeight(d, lw.post_ffn_norm, {h});
    vt::RmsNorm(d.q, moe_n.t(), moe.t(), w_pf, vt::RmsNormArgs{eps, false});

    auto* held = new DBuf(std::move(moe_n));
    hidden = held->t();
    hidden_hold = std::shared_ptr<void>(held, [](void* q) {
      delete static_cast<DBuf*>(q);
    });
    if (ProgressOn()) {
      std::fprintf(stderr, "[kolibri1-tt-b2bi]   layer %lld/%lld done\n",
                   static_cast<long long>(l + 1),
                   static_cast<long long>(p.num_hidden_layers));
    }
  }

  // Final norm carries the residual (Qwen3MoeModel.norm(hidden, residual)).
  DBuf dnorm(d, DType::kBF16, {t, h});
  Tensor w_fn = ResidentWeight(d, weights.final_norm, {h});
  Tensor dnorm_t = dnorm.t();
  Tensor res_t = res.t();
  if (dense_attn::FusedChainAdoptEnabled()) {
    vt::FusedChain(d.q, dnorm_t, hidden, w_fn, &res_t,
                   vt::kFusedAddRmsNormStd, eps);
  } else {
    vt::RmsNorm(d.q, dnorm_t, hidden, w_fn, vt::RmsNormArgs{eps, false},
                &res_t);
  }

  // logits_indices gather, then the UNTIED lm_head.
  const bool do_gather = !logits_indices.empty() &&
                         static_cast<int64_t>(logits_indices.size()) < t;
  const int64_t n_idx = static_cast<int64_t>(logits_indices.size());
  DBuf dgather(d, DType::kBF16, {do_gather ? n_idx : int64_t{0}, h});
  Tensor src = dnorm.t();
  if (do_gather) {
    const size_t rb = static_cast<size_t>(h) * vt::SizeOf(DType::kBF16);
    auto* dp = static_cast<char*>(dgather.ptr());
    const auto* sp = static_cast<const char*>(dnorm.t().data);
    for (size_t s = 0; s < logits_indices.size(); ++s)
      d.b.Copy(d.q, dp + s * rb,
               sp + static_cast<size_t>(logits_indices[s]) * rb, rb);
    src = dgather.t();
  }
  const int64_t n_out = do_gather ? n_idx : t;

  Tensor lm = ResidentWeight(d, weights.lm_head, {vocab, h});
  DBuf logits(d, DType::kF32, {n_out, vocab});
  vt::MatmulBT(d.q, logits.t(), src, lm);

  ForwardLogits fl;
  fl.rows = n_out;
  fl.vocab = vocab;
  fl.device_tensor = logits.t();
  fl.device_storage = logits.ReleaseShared();
  return fl;
}

}  // namespace vllm
