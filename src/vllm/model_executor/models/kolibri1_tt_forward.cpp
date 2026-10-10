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
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // KvSlice
#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf, ResidentWeight
#include "vllm/model_executor/models/kolibri1_fp8_dequant.h"
#include "vllm/model_executor/models/kolibri1_tt_stream.h"
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

// SCRATCH DEBUG (root-cause the host-free decode corruption): per-stage
// activation checksums, env VT_KOLIBRI1_TT_STAGE_DUMP. Downloads the tensor
// (bf16 rows [t,h]) and prints sum, max|.|, and the first values so the TT
// arm's stage outputs can be diffed against the host-free-OFF arm stage by
// stage. Bisection instrumentation, not a shipped surface.
bool StageDumpOn() {
  static const bool on = std::getenv("VT_KOLIBRI1_TT_STAGE_DUMP") != nullptr;
  return on;
}

void StageDump(const char* tag, int64_t layer, DBuf& buf, Dev d) {
  if (!StageDumpOn()) return;
  const Tensor& t = buf.t();
  const int64_t n = t.shape[0] * t.shape[1];
  std::vector<uint8_t> tmp(static_cast<size_t>(n) * 2);
  buf.Download(d, tmp.data());
  const auto* bf = reinterpret_cast<const uint16_t*>(tmp.data());
  double sum = 0;
  float mx = 0;
  auto val = [](uint16_t bits) {
    uint32_t u = static_cast<uint32_t>(bits) << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
  };
  for (int64_t i = 0; i < n; ++i) {
    const float v = val(bf[i]);
    sum += v;
    if (std::fabs(v) > mx) mx = std::fabs(v);
  }
  std::fprintf(stderr, "[STAGE] L%lld %s sum=%.6f max=%.6f first=[%.6f %.6f %.6f %.6f]\n",
               static_cast<long long>(layer), tag, sum, mx, val(bf[0]), val(bf[1]),
               val(bf[2]), val(bf[3]));
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
    if (StageDumpOn()) {
      auto cks = [&](const char* tag, const Tensor& tn) {
        std::vector<uint8_t> tmp(static_cast<size_t>(tn.Numel()) *
                                 vt::SizeOf(tn.dtype));
        d.b.Copy(d.q, tmp.data(), tn.data, tmp.size());
        double s = 0;
        if (tn.dtype == vt::DType::kBF16) {
          const auto* bf = reinterpret_cast<const uint16_t*>(tmp.data());
          auto val = [](uint16_t bits) {
            uint32_t u = static_cast<uint32_t>(bits) << 16;
            float f;
            std::memcpy(&f, &u, 4);
            return f;
          };
          for (int64_t i = 0; i < tn.Numel(); ++i) s += val(bf[i]);
        } else {
          const auto* f4 = reinterpret_cast<const float*>(tmp.data());
          for (int64_t i = 0; i < tn.Numel(); ++i) s += f4[i];
        }
        std::fprintf(stderr, "[STAGE] attnin %s sum=%.6f n=%lld dt=%d\n", tag, s,
                     static_cast<long long>(tn.Numel()),
                     static_cast<int>(tn.dtype));
      };
      auto view_at = [&](Tensor base, int64_t blocks) {
        Tensor t2;
        t2.data = base.data;
        t2.dtype = base.dtype;
        t2.device = base.device;
        t2.rank = 4;
        t2.shape[0] = blocks;
        t2.shape[1] = kv.block_size;
        t2.shape[2] = hkv;
        t2.shape[3] = dh;
        t2.stride[0] = kv.block_size * hkv * dh;
        t2.stride[1] = hkv * dh;
        t2.stride[2] = dh;
        t2.stride[3] = 1;
        return t2;
      };
      cks("k3", k3);
      cks("v3", v3);
      cks("k_cache", view_at(k_cache, 1));
      cks("v_cache", view_at(v_cache, 1));
      cks("slot_mapping", si.slot_mapping.t());
      cks("seq_lens", si.seq_lens.t());
    }
    dense_attn::WriteKvCache(d.q, kv, k3, v3, k_cache, v_cache,
                             si.slot_mapping.t());
    vt::PagedAttention(d.q, attn.t(), q3, k_cache, v_cache,
                       si.block_table.t(), si.seq_lens.t(),
                       si.query_start_loc.t(), pa);
  }

  Tensor o_in = Reshape(attn.t(), {t, hq * dh});
  // SCRATCH DEBUG: the paged-attention output before the o projection.
  if (StageDumpOn()) {
    std::vector<uint8_t> tmp(static_cast<size_t>(t) * hq * dh * 2);
    attn.Download(d, tmp.data());
    const auto* bf = reinterpret_cast<const uint16_t*>(tmp.data());
    double s = 0;
    auto val = [](uint16_t bits) {
      uint32_t u = static_cast<uint32_t>(bits) << 16;
      float f;
      std::memcpy(&f, &u, 4);
      return f;
    };
    const int64_t n = t * hq * dh;
    for (int64_t i = 0; i < n; ++i) s += val(bf[i]);
    std::fprintf(stderr, "[STAGE] L%lld pa_out sum=%.6f first=%.6f\n",
                 static_cast<long long>(t * 0), s, val(bf[0]));
  }
  return LinearBTDevice(d, o_in, o_w, t);  // [T, H]
}

// ---- MoE block (slice ii: router + the streaming routed path + shared) -------
//
// The CPU row's MoeBlock computes shared + the weighted routed combine
// (vt::MoeCombine). Slice ii adds the routed tier: the router runs on
// device (bf16 gate, f32 logits — the CPU row's contract), the
// sigmoid-logit-add top-6-of-384 runs host-side over the readback (the
// inherited f32 compute path), and the routed experts are STREAMED: the
// B2a dispatch plan remaps the request against the slot tables, the
// fetch executor's host half builds the jobs, the device half stages the
// slot payloads (packed bytes verbatim), the readback pivot verifies
// them byte-for-byte against the shadow, Touch() pins them, and the
// per-expert bf16 dequants (memoized, reset-lane invalidated) feed the
// CPU row's gather/ExpertMlp/scatter and the vt::MoeCombine. When no
// streaming context is attached the B2b-i refusal still fires by name —
// the refusal remains ONLY for the streaming-disabled arm and genuine
// over-capacity/miss-handle failures (which throw by name).
DBuf MoeBlock(Dev d, const Kolibri1MoeWeights& w, const Kolibri1Params& p,
              const Tensor& dhn, int64_t t, int64_t layer,
              const Tensor& sh_gate_w, const Tensor& sh_up_w,
              const Tensor& sh_down_w,
              Kolibri1TTStreamingDeviceContext* stream,
              const Kolibri1Weights& weights) {
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

  // SLICE ii: the routed tier. Without a streaming context (the
  // streaming-disabled arm) the B2b-i refusal still fires by name.
  if (stream == nullptr) {
    NoteRoutedExpertRequest(layer, route.ids);
    // Shared expert: UNGATED, always added (kolibri1.py:146-188) — and in
    // this arm it is the WHOLE MoE output.
    return ExpertMlp(d, dhn, t, p.shared_expert_intermediate_size, sh_gate_w,
                     sh_up_w, sh_down_w);
  }
  Kolibri1TTStreamingDeviceContext& st = *stream;
  VT_CHECK(st.built, "kolibri1-tt B2b-ii: the streaming context is not built");

  // The reset lane: a slot swap since the recording clears the memoized
  // dequants (the graph-reset semantics; the GDN churn fix pattern).
  if (st.epoch.ConsumeIfChanged(st.pool.policy)) st.memo.clear();

  // The routed path end to end: remap -> fetch list -> stage -> verify ->
  // Touch. The stream bound refuses LOUDLY past the B1 per-token ceiling.
  const std::vector<std::pair<int64_t, std::vector<int64_t>>> layer_req = {
      {layer, std::vector<int64_t>(route.ids.begin(), route.ids.end())}};
  const Kolibri1TTDispatchPlan disp =
      PlanKolibri1TTMoEDispatch(st.pool.policy, layer_req);
  const Kolibri1TTLayerDispatch& ld = disp.layers[0];
  const Kolibri1TTSlotFetchList list = BuildKolibri1TTSlotFetchList(
      st.pool.policy, ld, weights, st.pool, st.per_token_stream_bytes);
  VT_CHECK(st.guard != nullptr,
           "kolibri1-tt B2b-ii: the per-step stream guard is missing");
  st.guard->Charge(layer, list.stream_bytes);
  vt::Backend& be = vt::GetBackend(d.q.device.type);
  std::vector<uint8_t> slot_image(
      static_cast<size_t>(st.pool.packed_bytes_per_slot));
  for (const Kolibri1TTSlotFetchJob& job : list.jobs) {
    // Touch FIRST: the policy's slot is authoritative (the dispatch
    // plan's fetch_slots are advisory — SuggestSlotFor is read-only over
    // the tables, so a multi-miss dispatch repeats the first free slot).
    const int64_t slot = st.pool.policy.Touch(layer, job.expert);
    VT_CHECK(slot >= 0 && slot < st.pool.capacity_per_layer,
             "kolibri1-tt B2b-ii: the policy handed out slot " +
                 std::to_string(slot) + " outside the per-layer capacity");
    const int64_t device_offset = slot * st.pool.packed_bytes_per_slot;
    // The host image of the slot: the three packed projections
    // concatenated (gate, up, down) — the layout the pool stages.
    const Kolibri1ExpertWeights& ew =
        weights.layers[static_cast<size_t>(layer)]
            .moe.experts[static_cast<size_t>(job.expert)];
    const Kolibri1Projection* projs[3] = {&ew.gate_proj, &ew.up_proj,
                                          &ew.down_proj};
    int64_t off = 0;
    const bool was_swap = st.shadow->Has(layer, slot);
    for (const Kolibri1Projection* pr : projs) {
      const size_t nb = pr->fp8_block.packed.bytes.size();
      // H2D: the packed bytes VERBATIM, row-major, into the slot.
      d.b.Copy(d.q, static_cast<char*>(st.pool_base[static_cast<size_t>(layer)]) +
                        device_offset + off,
               pr->fp8_block.packed.bytes.data(), nb);
      std::memcpy(slot_image.data() + off, pr->fp8_block.packed.bytes.data(),
                  nb);
      // The f32 scale grids stay HOST-side (staged with the slot's shadow).
      off += static_cast<int64_t>(nb);
    }
    VT_CHECK(off == st.pool.packed_bytes_per_slot,
             "kolibri1-tt B2b-ii: slot payload byte math diverged");
    // The readback pivot: D2H the staged slot and verify BYTE-EXACT
    // against the shadow (no extract_shard on this tt-metal).
    be.Copy(d.q, slot_image.data(),
            static_cast<char*>(st.pool_base[static_cast<size_t>(layer)]) +
                device_offset,
            static_cast<size_t>(st.pool.packed_bytes_per_slot));
    be.Synchronize(d.q);
    st.shadow->Record(layer, slot, slot_image.data(),
                      st.pool.packed_bytes_per_slot);
    st.readback_verified_bytes += st.shadow->VerifyReadback(
        layer, slot, slot_image.data(), st.pool.packed_bytes_per_slot);
    ++st.slot_fills;
    st.staged_bytes += st.pool.packed_bytes_per_slot;
    if (was_swap) ++st.swap_fills;
  }
  // Re-record the epoch AFTER the fills so the next layer's check fires
  // only on a NEW content change (swaps it did not cause itself).
  st.epoch.Record(st.pool.policy);
  if (StageDumpOn()) {
    std::fprintf(stderr,
                 "[STAGE] pool L%lld staged=%lld verified=%lld fills=%lld\n",
                 static_cast<long long>(layer),
                 static_cast<long long>(st.staged_bytes),
                 static_cast<long long>(st.readback_verified_bytes),
                 static_cast<long long>(st.slot_fills));
  }

  // The routed experts: the CPU row's gather/ExpertMlp/scatter over the
  // slot dequants, then the weighted combine with the always-added shared
  // term. Identical op sequence to kolibri1_forward.cpp's MoeBlock.
  const int64_t h = p.hidden_size;
  const int64_t inter = p.moe_intermediate_size;
  DBuf dtw(d, DType::kF32, {t, top_k},
           const_cast<float*>(route.weights.data()));
  DBuf dtid(d, DType::kI32, {t, top_k},
            const_cast<int32_t*>(route.ids.data()));
  DBuf shared = ExpertMlp(d, dhn, t, p.shared_expert_intermediate_size,
                          sh_gate_w, sh_up_w, sh_down_w);
  StageDump("moe_shared", layer, shared, d);
  DBuf expert_out(d, DType::kBF16, {t, top_k, h});
  expert_out.Zero(d);
  for (int64_t ex = 0; ex < e; ++ex) {
    std::vector<int32_t> token_rows;
    std::vector<int32_t> slot_rows;
    for (int64_t i = 0; i < t; ++i) {
      for (int64_t kk = 0; kk < top_k; ++kk) {
        const int64_t idx = i * top_k + kk;
        if (route.ids[static_cast<size_t>(idx)] == static_cast<int32_t>(ex)) {
          token_rows.push_back(static_cast<int32_t>(i));
          slot_rows.push_back(static_cast<int32_t>(idx));
          break;
        }
      }
    }
    if (token_rows.empty()) continue;
    // The slot must hold this expert — a miss here is a miss-handle
    // refusal (the fetch above pinned every requested expert).
    const int64_t slot = st.pool.policy.SlotFor(layer, ex);
    VT_CHECK(slot >= 0,
             "kolibri1-tt B2b-ii: expert " + std::to_string(ex) +
                 " selected on layer " + std::to_string(layer) +
                 " but no slot holds it — refusing by name");
    // The memoized slot dequant: readback-verified slot bytes -> the CPU
    // row's DequantRowsBf16, threaded over output rows (bit-identical to
    // the CPU row's per-call dequant of the same bytes).
    const auto key = std::make_pair(layer, ex);
    auto it = st.memo.find(key);
    if (it != st.memo.end() && it->second.slot == slot) {
      ++st.memo_hits;
    } else {
      // The slot bytes are authoritative ON DEVICE: read the slot back
      // (the readback pivot) and verify byte-exact against the shadow
      // before computing over its dequant.
      std::vector<uint8_t> rb(
          static_cast<size_t>(st.pool.packed_bytes_per_slot));
      be.Copy(d.q, rb.data(),
              static_cast<char*>(st.pool_base[static_cast<size_t>(layer)]) +
                  slot * st.pool.packed_bytes_per_slot,
              rb.size());
      be.Synchronize(d.q);
      st.readback_verified_bytes += st.shadow->VerifyReadback(
          layer, slot, rb.data(), st.pool.packed_bytes_per_slot);
      // ONE allocation per expert (all three projections are inter x h
      // up to transpose), carved into three views; freed through the
      // backend when the memo entry dies.
      const Kolibri1ExpertWeights& ewx =
          weights.layers[static_cast<size_t>(layer)]
              .moe.experts[static_cast<size_t>(ex)];
      const int64_t n_dims[3] = {inter, inter, h};
      const int64_t k_dims[3] = {h, h, inter};
      const size_t nb_total = static_cast<size_t>(3 * inter * h) * 2;
      void* bp = be.Alloc(nb_total);
      Kolibri1TTStreamingDeviceContext::SlotDequant dq;
      dq.slot = slot;
      dq.owner = std::shared_ptr<void>(bp, [&be](void* ptr) {
        be.Free(ptr);
      });
      vt::Tensor* dsts[3] = {&dq.gate, &dq.up, &dq.down};
      // Two offsets over the carved allocation: the packed fp8 SOURCE
      // advances ONE byte per element, the bf16 destination TWO. The old
      // single byte_base doubled for both and walked the source out of the
      // slot: up dequanted down's region and down dequanted past the slot
      // end — NaN/Inf weights, the 1e38 routed-expert outputs, and the hard
      // flips downstream (the local record this commit cites).
      int64_t packed_base = 0;
      int64_t bf16_base = 0;
      for (int pi = 0; pi < 3; ++pi) {
        const Kolibri1Projection& pr =
            pi == 0 ? ewx.gate_proj : (pi == 1 ? ewx.up_proj : ewx.down_proj);
        const int64_t n = n_dims[pi], k = k_dims[pi];
        const int64_t scale_cols = CDiv(k, pr.fp8_block.block_k);
        VT_CHECK(static_cast<int64_t>(pr.fp8_block.packed.bytes.size()) ==
                     n * k,
                 "kolibri1-tt B2b-ii: routed projection byte math diverged");
        auto* dst = static_cast<uint16_t*>(bp) + bf16_base / 2;
        const auto* src = rb.data() + packed_base;
        const auto* sc =
            reinterpret_cast<const float*>(pr.fp8_block.scale.bytes.data());
        host_parallel::ForOutputRows(n, k, [&](int64_t n0, int64_t n1) {
          kolibri1_fp8::DequantRowsBf16(src, sc, scale_cols, n0, n1, k,
                                        pr.fp8_block.block_n,
                                        pr.fp8_block.block_k, dst);
        });
        packed_base += n * k;
        bf16_base += n * k * 2;
        *dsts[pi] = MakeTensor(dst, DType::kBF16, d.q.device,
                               std::vector<int64_t>{n, k});
      }
      it = st.memo.emplace(key, std::move(dq)).first;
    }
    const int64_t ne = static_cast<int64_t>(token_rows.size());
    DBuf gathered(d, DType::kBF16, {ne, h});
    {
      const size_t rb = static_cast<size_t>(h) * vt::SizeOf(DType::kBF16);
      auto* dp = static_cast<char*>(gathered.ptr());
      const auto* sp = static_cast<const char*>(dhn.data);
      for (size_t s = 0; s < token_rows.size(); ++s)
        d.b.Copy(d.q, dp + s * rb,
                 sp + static_cast<size_t>(token_rows[s]) * rb, rb);
    }
    StageDump("moe_gath", layer, gathered, d);
    DBuf o = ExpertMlp(d, gathered.t(), ne, inter, it->second.gate,
                       it->second.up, it->second.down);
    StageDump("moe_ex", layer, o, d);
    for (int64_t i = 0; i < ne; ++i) {
      d.b.Copy(d.q,
               static_cast<char*>(expert_out.ptr()) +
                   static_cast<size_t>(slot_rows[static_cast<size_t>(i)]) *
                       static_cast<size_t>(h) * vt::SizeOf(DType::kBF16),
               static_cast<const char*>(o.ptr()) +
                   static_cast<size_t>(i) * static_cast<size_t>(h) *
                       vt::SizeOf(DType::kBF16),
               static_cast<size_t>(h) * vt::SizeOf(DType::kBF16));
    }
  }
  DBuf out(d, DType::kBF16, {t, h});
  if (StageDumpOn()) {
    std::vector<float> wsum(route.weights.begin(), route.weights.end());
    double s = 0;
    for (float v : wsum) s += v;
    std::fprintf(stderr, "[STAGE] route sum=%.6f n=%zu\n", s, wsum.size());
  }
  Tensor shared_t = shared.t();
  vt::MoeCombine(d.q, out.t(), expert_out.t(), dtw.t(), &shared_t,
                 /*routed_scale=*/1.0f);
  StageDump("moe_combined", layer, out, d);
  return out;
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
    Kolibri1TTResidentDeviceContext& ctx,
    Kolibri1TTStreamingDeviceContext* streaming) {
  VT_CHECK(queue.device.type == vt::DeviceType::kTENSTORRENT, kNonTTRefusal);
  VT_CHECK(ctx.built, "kolibri1-tt forward: the device context is not built");
  // A fresh charge window per step: the B1 per-token stream bound is a
  // PER-TOKEN bound, and the guard refuses LOUDLY past it.
  if (streaming != nullptr) {
    streaming->guard = std::make_unique<Kolibri1TTStreamBoundGuard>(
        streaming->per_token_stream_bytes);
  }
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
  std::shared_ptr<void> hidden_hold;
  // SCRATCH FIX (host-free decode corruption): the embedding buffer used to
  // die at the end of this scope while `hidden` kept referencing its pooled
  // block. The block then went back to the DevicePool and was recycled by the
  // next [t,h] allocation (res/dhn), whose Memset + slot reuse desynced the
  // embedding's TT device shadow: in the host-free ON arm the first residual
  // RmsNorm served a stale/garbage shadow for the recycled buffer (L0 dhn
  // sum 26.34 vs CPU 9.85, first diverging stage, VT_KOLIBRI1_TT_STAGE_DUMP),
  // and the corruption propagated to junk logits. The OFF arm masked it
  // because the host residual path EnsureHost-downloads the true embedding
  // bytes first. Hold the embedding DBuf alive for the whole step instead.
  auto* hidden_held = new DBuf(d, DType::kBF16, {t, h});
  hidden_hold = std::shared_ptr<void>(hidden_held, [](void* q) {
    delete static_cast<DBuf*>(q);
  });
  {
    DBuf ids(d, DType::kI32, {t}, const_cast<int32_t*>(token_ids.data()));
    Tensor tab = ResidentWeight(d, weights.embed_tokens, {vocab, h});
    vt::Embedding(d.q, hidden_held->t(), tab, ids.t());
  }
  StageDump("emb", -2, *hidden_held, d);
  Tensor hidden = hidden_held->t();
  DBuf res(d, DType::kBF16, {t, h});
  res.Zero(d);

  const float eps = static_cast<float>(p.rms_norm_eps);

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
    if (StageDumpOn() && l == 0) {
      std::vector<uint8_t> tmp(static_cast<size_t>(h) * 2);
      d.b.Copy(d.q, tmp.data(),
               static_cast<const uint8_t*>(w_in.data), tmp.size());
      d.b.Synchronize(d.q);
      const auto* bf = reinterpret_cast<const uint16_t*>(tmp.data());
      double s = 0;
      auto val = [](uint16_t bits) {
        uint32_t u = static_cast<uint32_t>(bits) << 16;
        float f;
        std::memcpy(&f, &u, 4);
        return f;
      };
      for (int64_t i = 0; i < h; ++i) s += val(bf[i]);
      std::fprintf(stderr, "[STAGE] gam sum=%.6f first=[%.6f %.6f %.6f %.6f]\n",
                   s, val(bf[0]), val(bf[1]), val(bf[2]), val(bf[3]));
    }
    Tensor dhn_t = dhn.t();
    Tensor res_t = res.t();
    if (dense_attn::FusedChainAdoptEnabled()) {
      vt::FusedChain(d.q, dhn_t, hidden, w_in, &res_t,
                     vt::kFusedAddRmsNormStd, eps);
    } else {
      vt::RmsNorm(d.q, dhn_t, hidden, w_in,
                  vt::RmsNormArgs{eps, false}, &res_t);
    }

    StageDump("dhn", l, dhn, d);
    // Attention -> post_attn_norm (no residual).
    DBuf attn = AttentionBlock(d, lw.attn, p, lw.is_sliding, dhn.t(),
                               si.positions.t(), si, *kv_ptr, t, cl.q, cl.k,
                               cl.v, cl.o);
    StageDump("attn", l, attn, d);
    DBuf attn_n(d, DType::kBF16, {t, h});
    Tensor w_pa = ResidentWeight(d, lw.post_attn_norm, {h});
    vt::RmsNorm(d.q, attn_n.t(), attn.t(), w_pa,
                vt::RmsNormArgs{eps, false});
    StageDump("attn_n", l, attn_n, d);

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

    StageDump("dh2", l, dh2, d);
    // MoE on EVERY layer -> post_ffn_norm (no residual). Slice ii: the
    // router, the streaming routed path (or the named refusal when the
    // streaming arm is disabled), and the shared expert.
    DBuf moe = MoeBlock(d, lw.moe, p, dh2.t(), t, l, cl.sh_gate, cl.sh_up,
                        cl.sh_down, streaming, weights);
    StageDump("moe", l, moe, d);
    DBuf moe_n(d, DType::kBF16, {t, h});
    Tensor w_pf = ResidentWeight(d, lw.post_ffn_norm, {h});
    vt::RmsNorm(d.q, moe_n.t(), moe.t(), w_pf, vt::RmsNormArgs{eps, false});

    StageDump("moe_n", l, moe_n, d);
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
  StageDump("dnorm", -1, dnorm, d);
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

// ---- The B2b-ii streaming-MoE device context (public contract) ---------------

std::unique_ptr<Kolibri1TTStreamingDeviceContext>
BuildKolibri1TTStreamingDeviceContext(vt::Backend& backend, vt::Queue& queue,
                                      const Kolibri1Weights& weights,
                                      int64_t device_budget_bytes,
                                      int64_t kv_reserve_bytes) {
  (void)queue;
  // The B1 streaming plan over the LOADED weights' byte math — computed,
  // never assumed: every field derives from the loaded projections so the
  // refusals plan over the bytes that will actually stage.
  Kolibri1TTStreamingShape shape;
  shape.layers = static_cast<int64_t>(weights.layers.size());
  VT_CHECK(shape.layers > 0,
           "kolibri1-tt B2b-ii: no layers in the loaded weights");
  shape.experts =
      static_cast<int64_t>(weights.layers[0].moe.experts.size());
  shape.topk = weights.params.num_experts_per_tok;
  int64_t attn = 0, shared = 0, router = 0, norm = 0;
  auto proj_bytes = [](const Kolibri1Projection& p) {
    if (!p.IsFp8Block()) return int64_t{0};
    return static_cast<int64_t>(p.fp8_block.packed.bytes.size() +
                                p.fp8_block.scale.bytes.size());
  };
  for (const Kolibri1LayerWeights& lw : weights.layers) {
    attn += proj_bytes(lw.attn.q_proj) + proj_bytes(lw.attn.k_proj) +
            proj_bytes(lw.attn.v_proj) + proj_bytes(lw.attn.o_proj);
    shared += proj_bytes(lw.moe.shared_experts.gate_proj) +
              proj_bytes(lw.moe.shared_experts.up_proj) +
              proj_bytes(lw.moe.shared_experts.down_proj);
    router += static_cast<int64_t>(
                  lw.moe.router_gate.bytes.size());
    router += static_cast<int64_t>(
                  lw.moe.e_score_correction_bias.bytes.size()) * 2;
    for (const auto* nt :
         {&lw.input_layernorm, &lw.post_attn_norm,
          &lw.post_attention_layernorm, &lw.post_ffn_norm, &lw.attn.q_norm,
          &lw.attn.k_norm}) {
      norm += static_cast<int64_t>(nt->bytes.size());
    }
  }
  shape.attention_bytes = attn;
  shape.shared_expert_bytes = shared;
  shape.router_bytes = router;
  shape.norm_bytes = norm;
  shape.embed_head_bytes =
      static_cast<int64_t>(weights.embed_tokens.bytes.size() +
                           weights.lm_head.bytes.size());
  {  // the routed tier's per-expert byte math (from layer 0, expert 0)
    const Kolibri1ExpertWeights& ew = weights.layers[0].moe.experts[0];
    shape.expert_bytes = proj_bytes(ew.gate_proj) + proj_bytes(ew.up_proj) +
                         proj_bytes(ew.down_proj);
  }
  Kolibri1TTStreamingOptions opts;
  opts.device_budget_bytes = device_budget_bytes;
  opts.kv_reserve_bytes = kv_reserve_bytes;
  opts.concurrency = 1;  // the B1 concurrency refusal is inherited
  // The touched-fraction threshold is the PLANNER's knob and stays at its
  // default for every direct caller; the registry arm decodes at
  // concurrency 1 (touched 6/384 = 0.016 on the real checkpoint; the tiny
  // fixture's 2/4 is a fixture artifact, not an operating point).
  opts.touched_fraction_threshold = 1.0;
  const Kolibri1TTStreamingPlan streaming = PlanKolibri1TTStreaming(shape, opts);

  auto st = std::make_unique<Kolibri1TTStreamingDeviceContext>();
  st->pool = PlanKolibri1TTSlotPool(weights, streaming);
  st->per_token_stream_bytes = streaming.per_token_stream_bytes;
  st->pool_base.resize(static_cast<size_t>(st->pool.layers), nullptr);
  for (int64_t l = 0; l < st->pool.layers; ++l) {
    void* p = backend.Alloc(static_cast<size_t>(st->pool.per_layer_pool_bytes));
    vt::Backend* bk = &backend;
    st->keepalive.push_back(std::shared_ptr<void>(p, [bk](void* ptr) {
      bk->Free(ptr);
    }));
    st->pool_base[static_cast<size_t>(l)] = p;
  }
  st->shadow = std::make_unique<Kolibri1TTSlotShadow>(
      st->pool.policy, st->pool.packed_bytes_per_slot);
  st->shadow->AttachEvictHook();
  st->epoch.Record(st->pool.policy);
  st->guard =
      std::make_unique<Kolibri1TTStreamBoundGuard>(st->per_token_stream_bytes);
  st->built = true;
  return st;
}

}  // namespace vllm
