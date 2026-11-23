// vllm.cpp ORIGINAL — the `olala` paged engine forward (NEW-CANDIDATE-PORTS
// wiring unit, ISSUE-LOCAL-01M835QTXAG5P2SD26QZYT0Z5W).
//
// WHAT THIS IS. The production decode path: `ModelRegistry::Forward` →
// `ForwardOlalaForCausalLM` (olala_registry.cpp) → THIS, whenever the runner
// supplies its paged KV + recurrent-state caches — i.e. every engine step
// (`LoadedEngine::FromModelDir` → generate, the vllm.h ABI, the server, the
// examples). The numeric pipeline is the W2d/W2e host op set (olala_ops.h),
// composed per token against the runner-owned caches, exactly the fold
// NemotronH's A2-P paged forward took for its hybrid
// Mamba+attention model (nemotron_h_device.cpp, `NemotronHPagedForward`).
//
// STATE OWNERSHIP — the one design decision this file makes. OlalaGenerateHost
// keeps all recurrence in locals: the 29 M layers' Mamba3-MIMO state and each
// V layer's `raw_k_last/raw_v_last` (the RAW pre-shift k/v of the PREVIOUS
// token, modeling_olala.py:619 — the token shift k = a·k_prev + (1-a)·k_raw
// blends against them). Under the paged runner the state must live in the
// RUNNER-OWNED caches, so:
//   * M layers → the recurrent group's four GdnStateCache::states rows
//     (slot-indexed), exactly MakeOlalaForCausalLMKVCache publishes them;
//     f32 rows, matching the host EXACT gate's f32 state bit for bit.
//   * V layers → the sliding-window page carries TWO head planes per k/v
//     (the spec publishes 2*p.kv_heads heads): heads [0, KV) hold the
//     post-shift post-norm k/v the attention reads; heads [KV, 2*KV) hold
//     that token's RAW pre-shift k/v, so the next token's shift reads its
//     `k_prev` from the page at pos-1. Caching the raw rows per token keeps
//     the recurrence ENTIRELY in the paged cache — no forward-side per-request
//     state, so preemption/re-ordering semantics stay the runner's, and the
//     raw chain stays window-local exactly like the shifted rows it feeds.
//     The page dtype is f32 for the same reason the M state is: the W2d bf16
//     token gate is EXACT against llama-cpp-olala, and bf16-rounding the
//     cached k/v (which llama.cpp's f32 cache does not round) would buy a
//     plausible near-tie drift into the production path. This is the
//     annotated f32 exception AGENTS.md asks for on a model-path buffer.
//
// SCOPE. Single request per step (the registry refuses a batched step by
// name, NemotronH's G-SAFE predicate). Host-addressable caches only: a CUDA
// queue refuses BY NAME — the device arm is owed. Sequence lengths past the
// sliding window refuse BY NAME (the window-slide block recycle is upstream
// SWA-manager work; the pinned artifact's window dwarfs the gate batteries).
#include "vllm/model_executor/models/olala_paged.h"

#include <algorithm>
#include <cstdio>
#include <cmath>
#include <cstring>
#include <vector>

#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/olala_ops.h"
#include "vllm/model_executor/models/qwen3_5.h"  // ForwardLogits, caches, HostLogits
#include "vllm/model_executor/models/qwen3_5_common.h"  // HostLogits
#include "vllm/v1/attention/backend.h"           // CommonAttentionMetadata
#include "vllm/v1/attention/backends/gdn_attn.h" // GDNAttentionMetadata
#include "vt/dtype.h"

namespace vllm {
namespace {

// A f32 view of one k/v plane of a paged FlashAttention cache
// (num_blocks, 2, block_size, heads, head_size): which=0 → K, 1 → V, the same
// unbind(1) layout qwen3_5.cpp KvSlice documents (block stride 2·bs·H·D).
float* KvPlane(PagedKvCache& kv, int which) {
  return reinterpret_cast<float*>(kv.data) +
         static_cast<size_t>(which) * static_cast<size_t>(kv.block_size) *
             static_cast<size_t>(kv.num_kv_heads) *
             static_cast<size_t>(kv.head_size);
}

// The f32 pointer of one head of one token slot inside one plane.
float* SlotHead(PagedKvCache& kv, int which, int64_t slot, int64_t head) {
  const int64_t bs = kv.block_size, h = kv.num_kv_heads, dd = kv.head_size;
  const int64_t block = slot / bs, offset = slot % bs;
  return KvPlane(kv, which) +
         static_cast<size_t>(block * 2 * bs * h * dd + offset * h * dd +
                             head * dd);
}

// One f32 state row of a runner-allocated GdnStateCache::states entry
// ([slots, ...] slot-major view, runner.cpp slot_major_view).
float* StateRow(const vt::Tensor& t, int64_t slot) {
  int64_t row = 1;
  for (int i = 1; i < t.rank; ++i) row *= t.shape[i];
  return static_cast<float*>(t.data) + static_cast<size_t>(slot * row);
}

}  // namespace

ForwardLogits OlalaPagedForward(const OlalaWeights& w, const OlalaParams& p,
                                const ModelForwardInput& input) {
  const int64_t L = p.layers, KV = p.kv_heads, D = p.head_dim;
  const int64_t T = static_cast<int64_t>(input.token_ids.size());
  VT_CHECK(T > 0, "olala paged forward: empty token sequence");
  VT_CHECK(input.num_reqs <= 1,
           "olala paged forward: BATCHED decode is not ported — a multi-request "
           "step would be decoded as ONE concatenated causal sequence (see "
           "NemotronH G-SAFE). Refusing by name.");
  VT_CHECK(static_cast<int64_t>(input.positions.size()) == T,
           "olala paged forward: incomplete attention metadata (no positions)");

  // ── the caches, matched to the two groups MakeOlalaForCausalLMKVCache
  // publishes: the sliding-window group over the 7 V layers first, the
  // recurrent group over the 29 M layers second.
  std::vector<int64_t> m_layers, v_layers;
  for (int64_t l = 0; l < L; ++l) {
    (p.layer_types[static_cast<size_t>(l)] == 'M' ? m_layers : v_layers)
        .push_back(l);
  }
  VT_CHECK(static_cast<int64_t>(input.attn_kv.size()) ==
               static_cast<int64_t>(v_layers.size()),
           "olala paged forward: the runner supplied a different number of "
           "paged KV layers than this model has V (attention) layers");
  VT_CHECK(static_cast<int64_t>(input.gdn_state.size()) ==
               static_cast<int64_t>(m_layers.size()),
           "olala paged forward: the runner supplied a different number of "
           "recurrent state layers than this model has M (Mamba3) layers");

  const PagedKvCache& kv0 = input.attn_kv[0];
  VT_CHECK(kv0.data != nullptr && kv0.num_blocks > 0,
           "olala paged forward: an empty paged KV cache");
  VT_CHECK(kv0.num_kv_heads == 2 * KV,
           "olala paged forward: the sliding-window page must carry TWO head "
           "planes (shifted + raw); rebuild the KV spec");
  VT_CHECK(kv0.head_size == D,
           "olala paged forward: sliding-window head_size != head_dim");
  VT_CHECK(kv0.dtype == vt::DType::kF32,
           "olala paged forward: the sliding-window page must be f32 (see the "
           "KV-spec builder's parity note)");

  // ── the page maps. This model publishes TWO groups, so the runner addresses
  // it through the BY-NAME channel (ModelFactory::consumes_multi_kv) and its
  // per-group metadata is NOT the uniform `attn_meta`: the SWA group's block
  // table, the recurrent group's block table (whose column 0 is the request's
  // recurrent-state slot, gdn_attn.py:219) both arrive through
  // `MultiKvCacheIndex::BlockTableForGroup` (model_registry.h). The group ids
  // are resolved BY NAME against the SAME helper the builder published with
  // (OlalaLayerKvName), so the publisher and this consumer cannot drift.
  const MultiKvCacheIndex* mk = input.multi_kv;
  VT_CHECK(mk != nullptr,
           "olala paged forward: the engine carried no by-name cache index "
           "(multi_kv). This model publishes two KV-cache groups; a step "
           "without the channel cannot address its pages. Direct paged-cache "
           "callers (scaffolds) are not a supported path.");
  int64_t v_group = -1, m_group = -1;
  if (mk->group_ids != nullptr && mk->layer_names != nullptr) {
    for (size_t i = 0; i < mk->layer_names->size(); ++i) {
      const std::string name((*mk->layer_names)[i]);
      if (v_group < 0 && name == OlalaLayerKvName(p, v_layers[0])) {
        v_group = (*mk->group_ids)[i];
      }
      if (m_group < 0 && name == OlalaLayerKvName(p, m_layers[0])) {
        m_group = (*mk->group_ids)[i];
      }
    }
  }
  VT_CHECK(v_group >= 0 && m_group >= 0 && v_group != m_group,
           "olala paged forward: could not resolve the two published groups "
           "by name in the engine's cache channel");
  int v_cols = 0, m_cols = 0;
  const std::vector<int32_t>* v_bt = mk->BlockTableForGroup(v_group, &v_cols);
  const std::vector<int32_t>* m_bt = mk->BlockTableForGroup(m_group, &m_cols);
  VT_CHECK(v_bt != nullptr && v_cols > 0 && m_bt != nullptr && m_cols > 0,
           "olala paged forward: the engine gathered no block table for a "
           "published group — a cache with no page map is allocated and "
           "unreadable");
  VT_CHECK(static_cast<int64_t>(v_bt->size()) >= v_cols &&
               static_cast<int64_t>(m_bt->size()) >= m_cols,
           "olala paged forward: a group block table is shorter than its "
           "declared width");

  // The recurrent-state slot: the runner REMAPS the recurrent group's block
  // table column 0 to a compact per-sequence slot (runner.cpp
  // remap_gdn_state_slots) and delivers it through the GDN metadata
  // (gdn_attn.py:219). The by-name channel's table is the PHYSICAL page map
  // and must not be read as a state slot.
  int64_t slot = -1;
  if (input.gdn_meta.non_spec_state_indices_tensor.has_value() &&
      !input.gdn_meta.non_spec_state_indices_tensor->empty()) {
    slot = (*input.gdn_meta.non_spec_state_indices_tensor)[0];
  } else {
    VT_CHECK(false,
             "olala paged forward: the GDN metadata carries no state index — "
             "the runner did not wire the recurrent group this step");
  }
  for (const GdnStateCache& g : input.gdn_state) {
    VT_CHECK(g.states.size() == 4,
             "olala paged forward: the runner's recurrent cache does not carry "
             "the four published states (hand-built scaffold? the paged path "
             "needs a runner-allocated engine)");
    for (const vt::Tensor& t : g.states) {
      VT_CHECK(t.data != nullptr && t.dtype == vt::DType::kF32 &&
                   t.device.type == vt::DeviceType::kCPU,
           "olala paged forward: the recurrent state must be f32 on a "
           "host-addressable device — the CUDA arm is owed");
    }
    VT_CHECK(slot >= 0 && slot < g.states[0].shape[0],
             "olala paged forward: recurrent state slot out of range");
  }

  // A fresh request (a token at position 0 in this step) must not inherit a
  // recycled block's stale state: zero all four rows, the same
  // gather-and-zero obligation the GDN-STATE-ZEROING note pins for NemotronH.
  bool fresh = false;
  for (const int32_t pos : input.positions) {
    if (pos == 0) fresh = true;
  }
  if (fresh) {
    for (const GdnStateCache& g : input.gdn_state) {
      for (const vt::Tensor& t : g.states) {
        int64_t row = 1;
        for (int i = 1; i < t.rank; ++i) row *= t.shape[i];
        std::memset(StateRow(t, slot), 0,
                    static_cast<size_t>(row) * sizeof(float));
      }
    }
  }

  // Sequence length past the sliding window: the SWA manager recycles blocks
  // outside the window and the block table no longer indexes absolute
  // positions — refuse by name rather than attend across recycled rows.
  int64_t max_pos = 0;
  for (const int32_t pos : input.positions) {
    max_pos = std::max(max_pos, static_cast<int64_t>(pos));
  }
  const int64_t window = p.window;
  VT_CHECK(max_pos < window,
           "olala paged forward: sequence length past the sliding window is "
           "not ported (the SWA window-slide block recycle is owed)");

  // Per-token state scratch.
  OlalaMambaState mst;
  std::vector<float> scratch_k, scratch_v, raw_k(KV * D), raw_v(KV * D);

  // The hidden rows lm_head gathers (logits_indices, request order).
  std::vector<int32_t> gather = input.logits_indices;
  if (gather.empty()) {
    gather.resize(static_cast<size_t>(T));
    for (int64_t t = 0; t < T; ++t) gather[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  std::vector<float> x(static_cast<size_t>(p.hidden));
  std::vector<float> hiddens(static_cast<size_t>(gather.size()) *
                             static_cast<size_t>(p.hidden));
  int64_t midx = 0, vidx = 0;  // M/V layer ordinals
  for (int64_t t = 0; t < T; ++t) {
    const int64_t pos = static_cast<int64_t>(input.positions[static_cast<size_t>(t)]);
    OlalaEmbedRowHost(w.embed, input.token_ids[static_cast<size_t>(t)],
                      x.data(), p.hidden);
    // normalize_embeddings_ngpt=True (modeling_olala.py:2020-2021)
    double s = 0.0;
    for (int64_t i = 0; i < p.hidden; ++i) s += double(x[static_cast<size_t>(i)]) * x[static_cast<size_t>(i)];
    const double inv = 1.0 / std::sqrt(std::max(s, 1e-12));
    for (int64_t i = 0; i < p.hidden; ++i)
      x[static_cast<size_t>(i)] = float(double(x[static_cast<size_t>(i)]) * inv);

    midx = 0;
    vidx = 0;
    for (int64_t l = 0; l < L; ++l) {
      const OlalaLayerWeights& lw = w.layers[static_cast<size_t>(l)];
      const bool is_m = p.layer_types[static_cast<size_t>(l)] == 'M';
      const int64_t mixer_out = is_m ? 2 * p.hidden : (p.heads - KV) * D;
      std::vector<float> ymix(static_cast<size_t>(mixer_out), 0.0f);
      if (is_m) {
        // gather → step → scatter, one row per published state.
        const GdnStateCache& g = input.gdn_state[static_cast<size_t>(midx)];
        const int64_t nheads = 2 * p.hidden / p.headdim;
        mst.ssm.assign(StateRow(g.states[0], slot),
                       StateRow(g.states[0], slot) + nheads * p.headdim * p.d_state);
        mst.k_state.assign(
            StateRow(g.states[1], slot),
            StateRow(g.states[1], slot) + p.mimo_dim * nheads * p.d_state);
        mst.v_state.assign(StateRow(g.states[2], slot),
                           StateRow(g.states[2], slot) + nheads * p.headdim);
        mst.angle.assign(StateRow(g.states[3], slot),
                         StateRow(g.states[3], slot) + nheads * p.rope_angles);
        OlalaMambaStepHost(ymix.data(), mst, x.data(), lw, p);
        std::memcpy(StateRow(g.states[0], slot), mst.ssm.data(),
                    mst.ssm.size() * sizeof(float));
        std::memcpy(StateRow(g.states[1], slot), mst.k_state.data(),
                    mst.k_state.size() * sizeof(float));
        std::memcpy(StateRow(g.states[2], slot), mst.v_state.data(),
                    mst.v_state.size() * sizeof(float));
        std::memcpy(StateRow(g.states[3], slot), mst.angle.data(),
                    mst.angle.size() * sizeof(float));
        ++midx;
      } else {
        // Gather this V layer's page rows 0..pos into the contiguous
        // [pos+1, KV*D] view OlalaVAttentionStepHost reads, step, and write
        // the row-pos k/v AND this token's raw k/v back into the page.
        PagedKvCache& kv = input.attn_kv[static_cast<size_t>(vidx)];
        const int64_t bs = kv.block_size;
        auto page_slot = [&](int64_t row) {
          VT_CHECK(row / bs < v_cols,
                   "olala paged forward: position past the gathered block "
                   "table width");
          return static_cast<int64_t>((*v_bt)[static_cast<size_t>(row / bs)]) *
                     bs +
                 row % bs;
        };
        scratch_k.assign(static_cast<size_t>((pos + 1) * KV * D), 0.0f);
        scratch_v.assign(static_cast<size_t>((pos + 1) * KV * D), 0.0f);
        for (int64_t r = 0; r < pos; ++r) {
          const int64_t sl = page_slot(r);
          std::memcpy(scratch_k.data() + static_cast<size_t>(r * KV * D),
                      SlotHead(kv, 0, sl, 0),
                      static_cast<size_t>(KV * D) * sizeof(float));
          std::memcpy(scratch_v.data() + static_cast<size_t>(r * KV * D),
                      SlotHead(kv, 1, sl, 0),
                      static_cast<size_t>(KV * D) * sizeof(float));
        }
        if (pos > 0) {
          // the raw planes of row pos-1 are the shift's k_prev/v_prev
          const int64_t sl = page_slot(pos - 1);
          std::memcpy(raw_k.data(), SlotHead(kv, 0, sl, KV),
                      static_cast<size_t>(KV * D) * sizeof(float));
          std::memcpy(raw_v.data(), SlotHead(kv, 1, sl, KV),
                      static_cast<size_t>(KV * D) * sizeof(float));
        } else {
          std::memset(raw_k.data(), 0, static_cast<size_t>(KV * D) * sizeof(float));
          std::memset(raw_v.data(), 0, static_cast<size_t>(KV * D) * sizeof(float));
        }
        OlalaVAttentionStepHost(scratch_k.data(), scratch_v.data(),
                                raw_k.data(), raw_v.data(), x.data(), pos, lw,
                                p, ymix.data());
        // commit row pos: shifted k/v into heads [0, KV), raw into [KV, 2KV).
        // No slot_mapping on this topology: the by-name channel's per-group
        // table IS the position→page map, so row pos's slot is the same
        // page_slot arithmetic the reads used.
        const int64_t sl = page_slot(pos);
        std::memcpy(SlotHead(kv, 0, sl, 0),
                    scratch_k.data() + static_cast<size_t>(pos * KV * D),
                    static_cast<size_t>(KV * D) * sizeof(float));
        std::memcpy(SlotHead(kv, 1, sl, 0),
                    scratch_v.data() + static_cast<size_t>(pos * KV * D),
                    static_cast<size_t>(KV * D) * sizeof(float));
        std::memcpy(SlotHead(kv, 0, sl, KV), raw_k.data(),
                    static_cast<size_t>(KV * D) * sizeof(float));
        std::memcpy(SlotHead(kv, 1, sl, KV), raw_v.data(),
                    static_cast<size_t>(KV * D) * sizeof(float));
        ++vidx;
      }
      OlalaLayerTailHost(lw, p, l, ymix.data(), mixer_out, x.data());
    }
    for (size_t gi = 0; gi < gather.size(); ++gi) {
      if (gather[gi] == static_cast<int32_t>(t)) {
        std::memcpy(hiddens.data() + gi * static_cast<size_t>(p.hidden),
                    x.data(), static_cast<size_t>(p.hidden) * sizeof(float));
      }
    }
  }

  // lm_head over the gathered rows (final_norm=False), host logits — the same
  // carrier NemotronH's paged forward returns.
  const int64_t V = p.vocab;
  std::vector<float> logits(static_cast<size_t>(gather.size() * V), 0.0f);
  for (size_t gi = 0; gi < gather.size(); ++gi) {
    OlalaLmHeadHost(w, p, hiddens.data() + gi * static_cast<size_t>(p.hidden),
                    logits.data() + static_cast<size_t>(gi) * static_cast<size_t>(V));
  }
  return HostLogits(std::move(logits), V);
}

}  // namespace vllm
