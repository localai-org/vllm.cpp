// THE W4 G2 REACHABILITY GATE for `Qwen3_5DecodeGraph`, the GDN-hybrid MoE
// decode driver — the RICHEST of the nine and the one that HAS the persistent
// device input path (`StepDevInputs`, 41 lines of `qwen3_5.cpp`).
//
// Row ENG-CUDAGRAPH-BREAK W4, spec `.agents/specs/eng-cudagraph-break.md`,
// issue #1307, parent #1163.
//
// WHY THIS FILE EXISTS AND A TOKEN GATE DOES NOT SUFFICE. A driver that kept its
// hand-rolled `BeginCapture`/`EndCaptureGraph` pair produces IDENTICAL logits, an
// IDENTICAL backend call log and an identical `replay_count()`.
// `vt::GraphBreakStats::segments_captured` moves only when a
// `vt::GraphCaptureScope` closes a segment and `replays` only inside
// `vt::BreakableGraph::Replay`, so those two are the only observables in the
// tree that separate "captured a graph" from "captured a graph THROUGH THE
// SEAM". W2 (#1261) and W3 (#1291) established this shape; this file applies it
// to the two drivers W4 owns.
//
// kFULL, ASSERTED ON THE MODE ITSELF. vLLM's v1 default `FULL_AND_PIECEWISE`
// (`vllm/config/compilation.py:63` @ pin `5559679229`) is documented at
// `:630-632` as a FULL graph for DECODE batches and a piecewise one for prefill
// and mixed batches, and `decode_mode()` (`:65-66`) returns the full half. W3
// measured what happens when the mode is inferred from a side effect instead of
// counted: flipping one token from `kFull` to `kPiecewise` compiled clean and
// left a whole driver gate GREEN at 226/226, because `breaks_registered == 0` is
// true in BOTH modes for a model that registers no break point. `full_scopes`
// and `piecewise_scopes` are what actually move.
//
// The harness and what it cannot see are stated once, in
// `decode_graph_seam_harness.h`. In one line: a CPU "replay" recomputes nothing,
// so this file gates the ROUTING and the capture step's numerics and NOT that a
// replayed segment reproduces the eager forward. That is G1, it needs a real
// device, and the spec's `## Gates` G1 records where it ran.
#include <doctest/doctest.h>

#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#include "decode_graph_seam_harness.h"
#include "vllm/model_executor/models/device_pool.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_mtp.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"  // W6 (#1374) dispatch counters
#include "vt/backend.h"
#include "vt/breakable_graph.h"
#include "vt/dtype.h"
#include "vt/persistent_step_input.h"
#include "vt/tensor.h"
#ifdef VLLM_CPP_XPU
#include "vt/xpu.h"
#endif
#include <algorithm>
#include <iostream>

namespace {

using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3_5MoeWeights;
using vllm::v1::CommonAttentionMetadata;
using vllm::v1::GDNAttentionMetadata;
using vllm_test::StaticGraphCpu;
using vt::DType;

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

// The tiny model is the one `tests/vllm/models/test_qwen35_paged_forward.cpp`
// already uses, so this driver runs the SAME arithmetic that file gates.
uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u = static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
  return static_cast<float>(u * 0.16 - 0.08);
}

OwnedTensor MakeOwned(DType dt, std::vector<int64_t> shape, uint64_t seed) {
  OwnedTensor t;
  t.dtype = dt;
  t.rank = static_cast<int>(shape.size());
  int64_t n = 1;
  for (int i = 0; i < t.rank; ++i) {
    t.shape[i] = shape[static_cast<size_t>(i)];
    n *= shape[static_cast<size_t>(i)];
  }
  if (dt == DType::kBF16) {
    t.bytes.resize(static_cast<size_t>(n) * 2);
    auto* p = reinterpret_cast<uint16_t*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i)
      p[i] = vt::F32ToBF16(RandV(seed + static_cast<uint64_t>(i)));
  } else {
    t.bytes.resize(static_cast<size_t>(n) * 4);
    auto* p = reinterpret_cast<float*>(t.bytes.data());
    for (int64_t i = 0; i < n; ++i) p[i] = RandV(seed + static_cast<uint64_t>(i));
  }
  return t;
}

HfConfig TinyConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_moe_text";
  c.architectures = {"Qwen3_5MoeForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;  // [LA, LA, LA, FA]
  c.vocab_size = 40;
  c.num_attention_heads = 4;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.num_experts = 4;
  c.num_experts_per_tok = 2;
  c.moe_intermediate_size = 16;
  c.shared_expert_intermediate_size = 16;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 4;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  return c;
}

vllm::MoeBlockWeights MakeMoe(const HfConfig& c, uint64_t s) {
  vllm::MoeBlockWeights m;
  const int64_t H = c.hidden_size, E = c.num_experts, I = c.moe_intermediate_size,
                Is = c.shared_expert_intermediate_size;
  m.router_gate = MakeOwned(DType::kBF16, {H, E}, s + 1);
  m.shared_gate = MakeOwned(DType::kBF16, {H, 1}, s + 2);
  for (int64_t e = 0; e < E; ++e) {
    m.expert_gate.push_back(MakeOwned(DType::kBF16, {H, I}, s + 100 + e * 7));
    m.expert_up.push_back(MakeOwned(DType::kBF16, {H, I}, s + 200 + e * 7));
    m.expert_down.push_back(MakeOwned(DType::kBF16, {I, H}, s + 300 + e * 7));
  }
  m.shared_gate_proj = MakeOwned(DType::kBF16, {H, Is}, s + 3);
  m.shared_up_proj = MakeOwned(DType::kBF16, {H, Is}, s + 4);
  m.shared_down_proj = MakeOwned(DType::kBF16, {Is, H}, s + 5);
  return m;
}

Qwen3_5MoeWeights MakeWeights(const HfConfig& c) {
  Qwen3_5MoeWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5MoeLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.moe = MakeMoe(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// KV plus recurrent-state caches for the GDN hybrid. Each arm of a comparison
// gets its OWN pool, so neither can read the other's writes and call the
// agreement a result.
struct CachePool {
  const HfConfig& c;
  int64_t num_blocks;
  int64_t block_size;
  std::vector<std::vector<float>> full_attn_buf;
  std::vector<std::vector<float>> gdn_ssm_buf;
  std::vector<std::vector<float>> gdn_conv_buf;
  std::vector<PagedKvCache> attn_kv;
  std::vector<GdnStateCache> gdn_state;
  int64_t spec_max_query_len = 1;

  // `spec_mql` is the longest speculative query length this pool must serve.
  // `causal_conv1d_spec_update` derives it from the conv state WIDTH --
  // `state_len = (K - 1) + (mql - 1)` (`src/vt/ops.cpp:1903`) -- so a pool built
  // at the plain decode width refuses every q above 1. Default 1 keeps every
  // existing case byte-identical.
  CachePool(const HfConfig& cfg, int64_t nb, int64_t bs, int64_t spec_mql = 1)
      : c(cfg), num_blocks(nb), block_size(bs), spec_max_query_len(spec_mql) {
    const int64_t Hkv = c.num_key_value_heads, Dh = c.head_dim;
    const int64_t Hv = c.linear_num_value_heads, Dv = c.linear_value_head_dim,
                  Dk = c.linear_key_head_dim, Kw = c.linear_conv_kernel_dim;
    const int64_t key_dim = c.linear_num_key_heads * Dk, value_dim = Hv * Dv;
    const int64_t conv_dim = 2 * key_dim + value_dim;
    for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
      if (c.layer_types[static_cast<size_t>(l)] == "linear_attention") {
        gdn_ssm_buf.emplace_back(static_cast<size_t>(nb * Hv * Dv * Dk), 0.0f);
        gdn_conv_buf.emplace_back(
            static_cast<size_t>(nb * conv_dim * (Kw - 1 + spec_max_query_len - 1)),
            0.0f);
      } else {
        full_attn_buf.emplace_back(static_cast<size_t>(nb * 2 * bs * Hkv * Dh), 0.0f);
      }
    }
    const vt::Device dev{vt::DeviceType::kCPU, 0};
    for (auto& b : full_attn_buf) {
      PagedKvCache kv;
      kv.data = b.data();
      kv.dtype = DType::kF32;
      kv.num_blocks = nb;
      kv.block_size = bs;
      kv.num_kv_heads = Hkv;
      kv.head_size = Dh;
      attn_kv.push_back(kv);
    }
    for (size_t g = 0; g < gdn_ssm_buf.size(); ++g) {
      GdnStateCache gs;
      gs.ssm_state = vt::Tensor::Contiguous(gdn_ssm_buf[g].data(), DType::kF32, dev,
                                            {nb, Hv, Dv, Dk});
      gs.conv_state = vt::Tensor::Contiguous(
          gdn_conv_buf[g].data(), DType::kF32, dev,
          {nb, conv_dim, Kw - 1 + spec_max_query_len - 1});
      gdn_state.push_back(gs);
    }
  }
};

// One single-request PURE-DECODE step at `pos`.
CommonAttentionMetadata DecodeAttnMeta(int32_t pos) {
  CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = 1;
  am.query_start_loc = {0, 1};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {pos + 1};
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = 1;
  am.max_seq_len = pos + 1;
  am.block_table_num_cols = 1;
  am.block_table_tensor = {0};
  am.slot_mapping = {pos};
  am.causal = true;
  return am;
}

GDNAttentionMetadata DecodeGdnMeta() {
  GDNAttentionMetadata gm;
  gm.num_prefills = 0;
  gm.num_prefill_tokens = 0;
  gm.num_decodes = 1;
  gm.num_decode_tokens = 1;
  gm.num_actual_tokens = 1;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>{0};
  gm.non_spec_query_start_loc = std::vector<int32_t>{0, 1};
  return gm;
}

}  // namespace

// THE GATE. Four decode steps at one shape drive the driver's whole state
// machine: cold (eager pre-warm), warm (CAPTURE), captured (REPLAY), and a
// second REPLAY.
TEST_CASE("G2: Qwen3_5DecodeGraph captures and replays THROUGH the vt seam") {
  const HfConfig c = TinyConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16);

  vt::ResetGraphBreakStats();
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  // Step 1, COLD: an eager pre-warm step. Nothing is captured yet.
  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 0);
    CHECK(s.replays == 0);
    CHECK_FALSE(graph.captured());
  }

  // Step 2, WARM: the driver captures. This is the assertion that fails while
  // the driver keeps its own BeginCapture/EndCaptureGraph pair.
  graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 1);
    CHECK(graph.captured());
    CHECK(s.full_scopes == 1);
    CHECK(s.piecewise_scopes == 0);
    CHECK(s.breaks_registered == 0);
  }

  // Step 3, CAPTURED: the driver replays. `replays` moves ONLY inside
  // vt::BreakableGraph::Replay.
  graph.Step({13}, {2}, DecodeAttnMeta(2), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 1);
    CHECK(graph.replay_count() >= 1);
  }

  // Step 4, a SECOND replay: the REPLAY branch is re-entrant. A branch that
  // re-captured, reset the container or left `warm` set would still look correct
  // after one step and diverge only on the next.
  graph.Step({14}, {3}, DecodeAttnMeta(3), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 2);
    CHECK(s.segments_captured == 1);  // a replay must never re-enter capture
  }

  // The backend saw exactly one capture pair, and the seam owns the release:
  // `BreakableGraph`'s destructor routes every handle through
  // Backend::DestroyGraph, which is what lets #1162 interpose later (spec D4).
  CHECK(harness.backend().Count("Begin") == 1);
  CHECK(harness.backend().Count("EndCaptureGraph") == 1);
  CHECK(harness.backend().Count("ReplayGraph") >= 2);
}

// G4 for this driver: with the seam in place, the CAPTURE step's logits are
// bit-identical to the same step run through the plain eager forward. This is
// what makes the migration reversible — and it is NOT G1, which needs a real
// device because a CPU "replay" recomputes nothing.
TEST_CASE("G4: the seam changed no numerics on Qwen3_5DecodeGraph's capture step") {
  const HfConfig c = TinyConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();

  // EAGER reference, on the stock CPU platform where the driver does not admit
  // itself, with its own caches.
  CachePool ref(c, 4, 16);
  vllm::Qwen3_5Model::Forward({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(),
                              ref.attn_kv, ref.gdn_state, w, c, q);
  const std::vector<float> eager = vllm::Qwen3_5Model::Forward(
      {12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), ref.attn_kv, ref.gdn_state, w,
      c, q);
  REQUIRE(eager.size() == static_cast<size_t>(c.vocab_size));

  // The DRIVER's cold step then its capture step, on its own caches.
  StaticGraphCpu harness;
  CachePool pool(c, 4, 16);
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);
  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  const vllm::ForwardLogits captured =
      graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
                 pool.gdn_state);
  REQUIRE(graph.captured());

  const auto* cp = static_cast<const float*>(captured.device_tensor.data);
  REQUIRE(cp != nullptr);
  size_t differing = 0;
  for (size_t i = 0; i < eager.size(); ++i)
    if (cp[i] != eager[i]) ++differing;
  MESSAGE("driver capture step vs eager, bit for bit: " << eager.size() << " values, "
                                                        << differing << " differing");
  CHECK(differing == 0);
  for (size_t i = 0; i < eager.size(); ++i) REQUIRE(std::isfinite(cp[i]));
}

// A capture that FAILS must reach the caller, and the step must never return the
// buffer as logits. Under stream capture NOTHING between `BeginCapture` and a
// throwing `EndCaptureGraph` executed — every kernel was RECORDED — so that
// buffer holds whatever the `DevicePool` last left there. Returning it is
// silently wrong tokens with no fault, and a token gate cannot see it. This was
// a live HIGH on W2's first head; the pre-W4 driver propagated because its
// `s.graph = b.EndCaptureGraph(...)` was unguarded, and this case is what stops
// the migration from losing that.
TEST_CASE("A capture that FAILS reaches the caller; Qwen3_5DecodeGraph never returns it") {
  const HfConfig c = TinyConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();

  StaticGraphCpu harness;
  CachePool pool(c, 4, 16);
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // cold pre-warm
  harness.backend().FailNextEndCapture();
  CHECK_THROWS_AS(graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(),
                             pool.attn_kv, pool.gdn_state),
                  std::exception);
  CHECK_FALSE(graph.captured());
}

// ───────────────────────────────────────────────────────────────────────────
// THE DENSE SIBLING, `Qwen3_5DenseDecodeGraph`.
// ───────────────────────────────────────────────────────────────────────────
//
// IT OWES ITS OWN GATE and shares this file's harness, which is the split W3
// arrived at the hard way. Its own gate, because nothing else can see the
// difference: a driver that kept its raw `BeginCapture`/`EndCaptureGraph` pair
// produces identical logits, an identical backend log and an identical
// `replay_count()`, and the two drivers migrate through separate code even
// though they migrate identically. This file rather than a fifth one, because
// copying the harness per gate would reproduce inside `tests/` the exact
// duplication this row removes from `src/`, and two copies of a harness diverge
// invisibly — both stay green while measuring different things.
namespace dense {

using vllm::DenseMlpWeights;
using vllm::Qwen3_5DenseLayerWeights;
using vllm::Qwen3_5DenseWeights;

// The tiny model `tests/vllm/models/test_qwen27_paged_forward.cpp` already uses:
// 27B-shaped, layer_types [LA, LA, LA, FA], no experts, GQA ratio 3.
HfConfig TinyConfig() {
  HfConfig c;
  c.model_type = "qwen3_5_text";
  c.architectures = {"Qwen3_5ForConditionalGeneration"};
  c.hidden_size = 32;
  c.num_hidden_layers = 4;
  c.vocab_size = 40;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {"linear_attention", "linear_attention", "linear_attention",
                   "full_attention"};
  c.intermediate_size = 16;
  c.num_experts = 0;
  c.linear_num_key_heads = 2;
  c.linear_num_value_heads = 6;
  c.linear_key_head_dim = 8;
  c.linear_value_head_dim = 8;
  c.linear_conv_kernel_dim = 4;
  c.rope_theta = 10000.0;
  c.rotary_dim = 4;
  c.rms_norm_eps = 1e-6;
  c.max_position_embeddings = 64;
  return c;
}

DenseMlpWeights MakeMlp(const HfConfig& c, uint64_t s) {
  DenseMlpWeights m;
  const int64_t H = c.hidden_size, I = c.intermediate_size;
  m.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 1);
  m.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 2);
  m.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 3);
  return m;
}

Qwen3_5DenseWeights MakeWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim;
  const int64_t Hk = c.linear_num_key_heads, Hv = c.linear_num_value_heads,
                Dk = c.linear_key_head_dim, Dv = c.linear_value_head_dim,
                Kw = c.linear_conv_kernel_dim;
  const int64_t key_dim = Hk * Dk, value_dim = Hv * Dv,
                conv_dim = 2 * key_dim + value_dim;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention =
        (c.layer_types[static_cast<size_t>(l)] == "linear_attention");
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    if (lw.is_linear_attention) {
      lw.gdn.in_proj_qkv = MakeOwned(DType::kBF16, {H, conv_dim}, s + 10);
      lw.gdn.in_proj_z = MakeOwned(DType::kBF16, {H, value_dim}, s + 20);
      lw.gdn.in_proj_b = MakeOwned(DType::kBF16, {H, Hv}, s + 30);
      lw.gdn.in_proj_a = MakeOwned(DType::kBF16, {H, Hv}, s + 40);
      lw.gdn.conv1d_weight = MakeOwned(DType::kBF16, {conv_dim, Kw}, s + 50);
      lw.gdn.a_log = MakeOwned(DType::kF32, {Hv}, s + 60);
      lw.gdn.dt_bias = MakeOwned(DType::kF32, {Hv}, s + 70);
      lw.gdn.norm_weight = MakeOwned(DType::kBF16, {Dv}, s + 80);
      lw.gdn.out_proj = MakeOwned(DType::kBF16, {value_dim, H}, s + 90);
    } else {
      lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
      lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
      lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
      lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
      lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
      lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    }
    lw.mlp = MakeMlp(c, s + 500);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

}  // namespace dense

TEST_CASE("G2: Qwen3_5DenseDecodeGraph captures and replays THROUGH the vt seam") {
  const HfConfig c = dense::TinyConfig();
  const dense::Qwen3_5DenseWeights w = dense::MakeWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16);

  vt::ResetGraphBreakStats();
  vllm::Qwen3_5DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // COLD
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 0);
    CHECK(s.replays == 0);
    CHECK_FALSE(graph.captured());
  }

  graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // WARM: capture
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.segments_captured == 1);
    CHECK(graph.captured());
    CHECK(s.full_scopes == 1);
    CHECK(s.piecewise_scopes == 0);
    CHECK(s.breaks_registered == 0);
  }

  graph.Step({13}, {2}, DecodeAttnMeta(2), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // REPLAY
  CHECK(vt::GetGraphBreakStats().replays >= 1);
  CHECK(graph.replay_count() >= 1);

  graph.Step({14}, {3}, DecodeAttnMeta(3), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // REPLAY again
  {
    const vt::GraphBreakStats s = vt::GetGraphBreakStats();
    CHECK(s.replays >= 2);
    CHECK(s.segments_captured == 1);  // a replay must never re-enter capture
  }

  CHECK(harness.backend().Count("Begin") == 1);
  CHECK(harness.backend().Count("EndCaptureGraph") == 1);
  CHECK(harness.backend().Count("ReplayGraph") >= 2);
}

TEST_CASE("R10: dense graph owns paired MTP outputs across replay and retirement") {
  const HfConfig c = dense::TinyConfig();
  const auto w = dense::MakeWeights(c);
  vt::Queue q = Q();
  vt::Queue consumer = Q();
  CachePool ref(c, 4, 16);
  vllm::Qwen3_5MTPHiddenStates expected_hidden;
  auto initial = vllm::Qwen3_5DenseModel::ForwardDeviceTap(
      {11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), ref.attn_kv,
      ref.gdn_state, w, c, q, &expected_hidden);
  initial = {};
  auto expected = vllm::Qwen3_5DenseModel::ForwardDeviceTap(
      {12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), ref.attn_kv,
      ref.gdn_state, w, c, q, &expected_hidden);
  const auto bytes = [](const vt::Tensor& t) {
    size_t count = vt::SizeOf(t.dtype);
    for (int dim = 0; dim < t.rank; ++dim) count *= size_t(t.shape[dim]);
    std::vector<unsigned char> result(count);
    std::memcpy(result.data(), t.data, count);
    return result;
  };
  const auto expected_logits = bytes(expected.device_tensor);
  const auto expected_tap = bytes(expected_hidden.tensor);

  StaticGraphCpu harness;  // CPU routing/ownership, not numerical GPU replay.
  CachePool pool(c, 4, 16);
  auto graph = std::make_unique<vllm::Qwen3_5DenseDecodeGraph>(w, c, q, 4);
  vllm::Qwen3_5MTPHiddenStates tap;
  {
    auto cold = graph->Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(),
                            pool.attn_kv, pool.gdn_state, nullptr, &tap);
    CHECK_FALSE(cold.non_owning_view);
    CHECK(tap.storage.get() == tap.tensor.data);
  }
  tap = {};  // release cold consumers before the capture step
  auto captured = graph->Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(),
                              pool.attn_kv, pool.gdn_state, nullptr, &tap);
  REQUIRE(graph->captured());
  CHECK(bytes(captured.device_tensor) == expected_logits);
  CHECK(bytes(tap.tensor) == expected_tap);
  CHECK(captured.device_storage.get() == captured.device_tensor.data);
  CHECK_FALSE(captured.non_owning_view);
  REQUIRE(tap.producer_ready_event != nullptr);
  tap.WaitReady(consumer);
  const auto held_tap = tap;
  const auto held_logits = captured;
  tap = {}; captured = {};
  void* held_address = held_tap.tensor.data;

  {
    auto replacement = graph->Step({13}, {2}, DecodeAttnMeta(2), DecodeGdnMeta(),
                                    pool.attn_kv, pool.gdn_state, nullptr, &tap);
    CHECK_FALSE(graph->captured());  // held generation forces safe retirement
    CHECK(tap.tensor.data != held_address);
    CHECK(bytes(held_tap.tensor) == expected_tap);
    CHECK(bytes(held_logits.device_tensor) == expected_logits);
  }
  tap = {};
  {
    auto recaptured = graph->Step({14}, {3}, DecodeAttnMeta(3), DecodeGdnMeta(),
                                   pool.attn_kv, pool.gdn_state, nullptr, &tap);
    CHECK(graph->captured());
  }
  tap = {};
  auto replay = graph->Step({15}, {4}, DecodeAttnMeta(4), DecodeGdnMeta(),
                            pool.attn_kv, pool.gdn_state, nullptr, &tap);
  CHECK(graph->replay_count() >= 1);
  const auto final_tap = bytes(tap.tensor);
  const auto final_logits = bytes(replay.device_tensor);
  graph.reset();
  tap.WaitReady(consumer);
  CHECK(bytes(tap.tensor) == final_tap);
  CHECK(bytes(replay.device_tensor) == final_logits);
  CHECK(bytes(held_tap.tensor) == expected_tap);
  CHECK(bytes(held_logits.device_tensor) == expected_logits);
}

TEST_CASE("G4: the seam changed no numerics on Qwen3_5DenseDecodeGraph's capture step") {
  const HfConfig c = dense::TinyConfig();
  const dense::Qwen3_5DenseWeights w = dense::MakeWeights(c);
  vt::Queue q = Q();

  CachePool ref(c, 4, 16);
  vllm::Qwen3_5DenseModel::Forward({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(),
                                   ref.attn_kv, ref.gdn_state, w, c, q);
  const std::vector<float> eager = vllm::Qwen3_5DenseModel::Forward(
      {12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), ref.attn_kv, ref.gdn_state, w,
      c, q);
  REQUIRE(eager.size() == static_cast<size_t>(c.vocab_size));

  StaticGraphCpu harness;
  CachePool pool(c, 4, 16);
  vllm::Qwen3_5DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/4);
  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  const vllm::ForwardLogits captured =
      graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
                 pool.gdn_state);
  REQUIRE(graph.captured());

  const auto* cp = static_cast<const float*>(captured.device_tensor.data);
  REQUIRE(cp != nullptr);
  size_t differing = 0;
  for (size_t i = 0; i < eager.size(); ++i)
    if (cp[i] != eager[i]) ++differing;
  MESSAGE("dense driver capture step vs eager, bit for bit: "
          << eager.size() << " values, " << differing << " differing");
  CHECK(differing == 0);
  for (size_t i = 0; i < eager.size(); ++i) REQUIRE(std::isfinite(cp[i]));
}

TEST_CASE("A capture that FAILS reaches the caller; Qwen3_5DenseDecodeGraph never returns it") {
  const HfConfig c = dense::TinyConfig();
  const dense::Qwen3_5DenseWeights w = dense::MakeWeights(c);
  vt::Queue q = Q();

  StaticGraphCpu harness;
  CachePool pool(c, 4, 16);
  vllm::Qwen3_5DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  harness.backend().FailNextEndCapture();
  CHECK_THROWS_AS(graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(),
                             pool.attn_kv, pool.gdn_state),
                  std::exception);
  CHECK_FALSE(graph.captured());
}

// ───────────────────────────────────────────────────────────────────────────
// THE CAPABILITY IS REACHED FROM A PRODUCTION STEP.
// ───────────────────────────────────────────────────────────────────────────
//
// `.agents/reachability.md` and the AGENTS.md "Nothing lands dead" rule: a seam
// capability that lands unreachable by the drivers that need it is the exact
// failure they exist for, and a unit test that constructs the type by hand
// proves the class works and never that anything reaches it. So this case enters
// through `Qwen3_5DecodeGraph::Step` — the same entry the registered forward
// calls — and asserts the PROCESS-WIDE counters, which move only inside
// `vt::PersistentStepInput`.
//
// IT SETS `VT_ASYNC_EXECUTOR=1`, and that is a limit stated rather than hidden.
// The persistent device input path is behind that lever (default OFF) plus the
// speculative-decode arm, so on a default text-decode step the driver holds no
// `StepDevInputs` at all and stages nothing. What this case therefore proves is
// that the production `Step` REACHES the capability on the configuration that
// has persistent device inputs — not that every step does. The lever is read
// once into `Impl::dbuf` at construction, so it is set before the driver is
// built and restored immediately after.
namespace {
struct ScopedEnv {
  const char* name;
  std::string prev;
  bool had;
  ScopedEnv(const char* n, const char* v) : name(n) {
    const char* p = std::getenv(n);
    had = p != nullptr;
    if (had) prev = p;
    ::setenv(n, v, 1);
  }
  ~ScopedEnv() {
    if (had)
      ::setenv(name, prev.c_str(), 1);
    else
      ::unsetenv(name);
  }
};
}  // namespace

TEST_CASE("Qwen3_5DecodeGraph::Step REACHES vt::PersistentStepInput") {
  const HfConfig c = TinyConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  vt::Queue q = Q();

  StaticGraphCpu harness;
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16);
  vt::ResetStepInputStats();
  REQUIRE(vt::GetStepInputStats().binds == 0);

  const ScopedEnv async_on("VT_ASYNC_EXECUTOR", "1");
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  // THE LEVER ALSO TURNS ON THE 2-SLOT PARITY RING, so each padded size holds
  // two slots and consecutive steps alternate between them. Slot 0 is therefore
  // cold on step 1 and WARM on step 3, which is where it captures — a three-step
  // walk here rather than the two-step one the single-slot cases use. That ring
  // is the reason the lever exists: at depth 2 the engine enqueues sample(i-1)
  // AFTER forward(i), so a single persistent logits buffer would be overwritten
  // before it is read.
  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 0, COLD
  graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 1, COLD
  graph.Step({13}, {2}, DecodeAttnMeta(2), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 0, WARM: binds the persistent inputs, captures
  REQUIRE(graph.captured());
  const vt::StepInputStats after_capture = vt::GetStepInputStats();
  MESSAGE("after capture: binds=" << after_capture.binds << " host_refreshes="
                                  << after_capture.host_refreshes);
  // Five inputs are bound unconditionally (positions, slot_mapping, block_table,
  // seq_lens, query_start_loc) and the GDN state index only when the step
  // carries one, so the floor is five rather than an exact count.
  CHECK(after_capture.binds >= 5);

  graph.Step({14}, {3}, DecodeAttnMeta(3), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 1, WARM
  graph.Step({15}, {4}, DecodeAttnMeta(4), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);  // slot 0, REPLAY: stages through the seam
  const vt::StepInputStats after_replay = vt::GetStepInputStats();
  CHECK(after_replay.host_refreshes >= 5);

  // The DEVICE arm is NOT exercised here, and saying so is the point of this
  // line. No decode driver refreshes an input from a device source yet: the one
  // input an asynchronous mirror patches is the token ids, and no driver has a
  // device destination for them (see PinnedStepInputs in `qwen3_5.cpp`). The
  // capability is landed and reached; adopting its device arm is what the
  // `qwen3.cpp` decline needs, and the spec's `## Owed` names it.
  CHECK(after_replay.device_refreshes == 0);
}

// ───────────────────────────────────────────────────────────────────────────
// ENG-CUDAGRAPH-BREAK W6 (#1374): THE SLOT KEY, closing the half of
// [#1020](https://github.com/mudler/vllm.cpp/issues/1020) that is not the
// predicate.
//
// WHY THIS IS NOT A UNIT TEST OF A COMPARATOR. `DecodeGraphSlotKey`'s ordering
// is four lines and a test that constructed one by hand would prove the
// comparator sorts. The claim here is different: that two SPEC steps whose
// padded token counts are EQUAL and whose uniform query lengths DIFFER get two
// graphs rather than one. Only `Qwen3_5DecodeGraph::Step` can answer that, and
// only through the driver's own capture counters -- a token gate cannot see
// which graph a replay came from, which is precisely how the collision stayed
// invisible.
//
// THE COLLISION IS REACHABLE TODAY AND NOT ONLY AFTER THE WIDENING.
// `S = spec_step ? B : PadToCaptureSize(B)`, so 2 requests x 2 tokens and
// 1 request x 4 tokens are both S == 4. Before W6 both indexed `slots[4]`, and
// the second replayed a graph captured against the first's metadata --
// different `spec_query_start_loc`, different `num_accepted_tokens`, different
// row count. `SizeSlot::Refresh` copies IN PLACE only while the sizes match and
// REASSIGNS the vector when they do not, which also moves the host addresses a
// capture baked (spec `## Risks/decisions` D2). Silently wrong logits.
namespace {

// One PURE-SPEC batch: `reqs` requests, each verifying `q` tokens.
// Mirrors the gdn_attn.cpp spec contract the driver's
// `ValidateGdnDecodeGraphState` enforces (`qwen3_5.cpp:367-421`): no non-spec
// rows, so `num_decodes == num_prefills == 0` and the non-spec segmentation is
// nullopt by construction.
GDNAttentionMetadata SpecGdnMeta(int32_t reqs, int32_t q) {
  GDNAttentionMetadata gm;
  gm.num_prefills = 0;
  gm.num_prefill_tokens = 0;
  gm.num_decodes = 0;
  gm.num_decode_tokens = 0;
  gm.num_spec_decodes = reqs;
  gm.num_spec_decode_tokens = reqs * q;
  gm.num_actual_tokens = reqs * q;
  gm.spec_state_indices_num_cols = q;
  std::vector<int32_t> ssi(static_cast<size_t>(reqs) * static_cast<size_t>(q));
  for (int32_t r = 0; r < reqs; ++r)
    for (int32_t j = 0; j < q; ++j)
      ssi[static_cast<size_t>(r * q + j)] = r;
  gm.spec_state_indices_tensor = std::move(ssi);
  std::vector<int32_t> sqsl(static_cast<size_t>(reqs) + 1);
  for (int32_t r = 0; r <= reqs; ++r) sqsl[static_cast<size_t>(r)] = r * q;
  gm.spec_query_start_loc = std::move(sqsl);
  gm.spec_sequence_masks = std::vector<uint8_t>(static_cast<size_t>(reqs), 1);
  std::vector<int32_t> stx(static_cast<size_t>(reqs) * static_cast<size_t>(q));
  for (size_t i = 0; i < stx.size(); ++i) stx[i] = static_cast<int32_t>(i);
  gm.spec_token_indx = std::move(stx);
  // Accept ONE token per request, the shape a verify step arrives in before the
  // rejection sampler has run. Column `acc - 1 == 0` is the live initial slot.
  gm.num_accepted_tokens = std::vector<int32_t>(static_cast<size_t>(reqs), 1);
  return gm;
}

CommonAttentionMetadata SpecAttnMeta(int32_t reqs, int32_t q, int32_t pos) {
  CommonAttentionMetadata am;
  am.num_reqs = reqs;
  am.num_actual_tokens = reqs * q;
  am.query_start_loc.resize(static_cast<size_t>(reqs) + 1);
  for (int32_t r = 0; r <= reqs; ++r)
    am.query_start_loc[static_cast<size_t>(r)] = r * q;
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens.assign(static_cast<size_t>(reqs), pos + q);
  am.seq_lens_cpu = am.seq_lens;
  am.max_query_len = q;
  am.max_seq_len = pos + q;
  // THE BLOCK TABLE HAS TO REACH THE LAST POSITION IT DECLARES, and a fixed one
  // column does not. `CachePool` uses `block_size = 16`, and the CPU paged
  // attention reads `btab[r * bt_row + (j / block_size) * bt_col]` for every
  // j <= p (`src/vt/cpu/cpu_paged_attn.cpp`). Shape C below sits at pos 20, so
  // j reaches 23, so the kernel reads column 1 of a ONE-column table -- an
  // out-of-bounds read of the pooled block that holds it. It was silent until a
  // pool change moved what follows that block, and then it was a SIGSEGV
  // (#1394). Size the table for the sequence length this metadata declares, and
  // map logical block i to physical block i so it agrees with
  // `slot_mapping = pos + i`.
  const int32_t kBlockSize = 16;  // CachePool's, and the only one these cases use
  const int32_t cols = (pos + q + kBlockSize - 1) / kBlockSize;
  am.block_table_num_cols = cols;
  am.block_table_tensor.resize(static_cast<size_t>(reqs) * static_cast<size_t>(cols));
  for (int32_t r = 0; r < reqs; ++r)
    for (int32_t i = 0; i < cols; ++i)
      am.block_table_tensor[static_cast<size_t>(r * cols + i)] = i;
  am.slot_mapping.resize(static_cast<size_t>(reqs) * static_cast<size_t>(q));
  for (size_t i = 0; i < am.slot_mapping.size(); ++i)
    am.slot_mapping[i] = pos + static_cast<int32_t>(i);
  am.causal = true;
  return am;
}

std::vector<int32_t> Iota(int32_t n, int32_t base) {
  std::vector<int32_t> v(static_cast<size_t>(n));
  for (int32_t i = 0; i < n; ++i) v[static_cast<size_t>(i)] = base + i;
  return v;
}

}  // namespace

TEST_CASE("W6: two spec shapes of EQUAL S and different q get two graphs") {
  // The conv window bounds the verify length this model can run:
  // `causal_conv1d_spec_update` refuses a query length above
  // `linear_conv_kernel_dim - 1` (`src/vt/ops.cpp:1913`). The shared
  // `TinyConfig` uses 4, which caps q at 3 and leaves no room for the third
  // distinct length the capture bound needs. Widen the window in a LOCAL copy
  // and build this case's weights and caches from that same copy, so nothing
  // else in the file observes it.
  HfConfig c = TinyConfig();
  c.linear_conv_kernel_dim = 6;
  const Qwen3_5MoeWeights w = MakeWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16, /*spec_mql=*/4);

  vt::ResetGraphBreakStats();
  vllm::v1::ResetGraphDispatchStats();
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  // SHAPE A: 3 requests x 2 tokens == 6 tokens. THREE steps, because a spec step
  // always takes the two-slot parity ring (`dbuf = impl_->dbuf || spec_step`):
  // slot 0 runs cold, slot 1 runs cold, and slot 0 is warm and captures on the
  // THIRD. Two steps would leave nothing captured and the collision below
  // unexercised, which is the "assertions: 0 wearing a pass" shape.
  for (int step = 0; step < 3; ++step) {
    graph.Step(Iota(6, 10), Iota(6, 0), SpecAttnMeta(3, 2, 0), SpecGdnMeta(3, 2),
               pool.attn_kv, pool.gdn_state);
  }
  CHECK(vllm::v1::GetGraphDispatchStats().capture_shapes == 1);
  CHECK(vt::GetGraphBreakStats().segments_captured == 1);

  // SHAPE B: 2 requests x 3 tokens == 6 tokens. The SAME S, a DIFFERENT q.
  // Before W6 this indexed the ring shape A already captured, found
  // `graph.captured()` true on the FIRST step, and replayed shape A's graph
  // against shape B's metadata. The key separates them, so this opens a ring of
  // its own and takes the cold path again.
  for (int step = 0; step < 3; ++step) {
    graph.Step(Iota(6, 20), Iota(6, 0), SpecAttnMeta(2, 3, 8), SpecGdnMeta(2, 3),
               pool.attn_kv, pool.gdn_state);
  }
  const vllm::v1::GraphDispatchStats st = vllm::v1::GetGraphDispatchStats();
  CAPTURE(st.capture_shapes);
  CAPTURE(st.qlen_cap_declines);
  // THE ASSERTION THE MUTATION MOVES. Two rings, not one.
  CHECK(st.capture_shapes == 2);
  // And two captures, which is what says the second ring was actually USED
  // rather than merely created. `segments_captured` moves only when a
  // `vt::GraphCaptureScope` closes a segment.
  CHECK(vt::GetGraphBreakStats().segments_captured == 2);
  CHECK(st.qlen_cap_declines == 0);

  // SHAPE C: 1 request x 4 tokens, a THIRD distinct speculative query length.
  // `VT_SPEC_GRAPH_MAX_QLENS` defaults to 2, so this one is refused and runs
  // eager -- which is what every clamped shape did before W6. The difference is
  // that it now moves a counter instead of nothing, which is the half of #1020
  // its title is about.
  for (int step = 0; step < 3; ++step) {
    graph.Step(Iota(4, 30), Iota(4, 0), SpecAttnMeta(1, 4, 20), SpecGdnMeta(1, 4),
               pool.attn_kv, pool.gdn_state);
  }
  const vllm::v1::GraphDispatchStats st2 = vllm::v1::GetGraphDispatchStats();
  CAPTURE(st2.capture_shapes);
  CAPTURE(st2.qlen_cap_declines);
  CHECK(st2.qlen_cap_declines == 3);
  CHECK(st2.capture_shapes == 2);  // no third ring was opened
  CHECK(vt::GetGraphBreakStats().segments_captured == 2);
}

// ─── #2029: THE CAPTURE MUST ALLOCATE NOTHING, ON THE LANE THE DEFAULT SERVER TAKES ───
//
// #2029: with speculation OFF the engine dies at concurrency 8 with
// `vt cuda: cudaMalloc: operation not permitted when stream is capturing`, while
// the SAME binary with `--speculative-config` serves. The asymmetry is not in the
// speculative code. It is in `dbuf`:
//
//     const bool dbuf = impl_->dbuf || spec_step;              // qwen3_5.cpp
//
// with `impl_->dbuf` false unless `VT_ASYNC_EXECUTOR=1` and `spec_step` false on
// every non-speculative step -- and the #1380 capture pre-grow living INSIDE
// `if (dbuf)`. So the default server captures over a pool nobody prepared, and
// #1380's own failure comes back by the one route its fix does not cover.
//
// WHAT THESE CASES ASSERT, and why it is not the call. `CHECK(pre-grow was
// called)` is a transcription: it stays green when the demand profile is wrong,
// which is the half #1393's own body recorded as ungated ("the fix rests on the
// captured forward demanding no more blocks of any size class than the eager
// forward at that shape did ... no test asserts it"). The observable here is the
// GUARANTEE -- zero `Backend::Alloc` calls between `BeginCapture` and
// `EndCaptureGraph` -- which on CUDA is exactly the `cudaMalloc` that aborts the
// capture, and which `CaptureCapableCpuBackend` counts without a device.
//
// WHY THE POOL IS DRAINED between the cold step and the capture step. In
// production the free list is SHORT rather than empty: every captured `SizeSlot`
// retains its `[S, vocab]` logits and `[S, H]` hidden forever, `DevicePool` is
// keyed by SIZE CLASS, and a ramping server captures more shapes -- which is why
// #2029 fires at c=8 and not at c=2. Reproducing that by arithmetic would make
// the case depend on whether two of this tiny model's tensors happen to collide
// in a class, i.e. on a coincidence rather than on the rule. `Drain` is the same
// condition taken to its limit, it is a production API
// (`DevicePool::Drain`, called at phase changes), and it is what also settles the
// premise above: an empty free list can only be served by the pre-grow, so a
// green here says the cold step's profile COVERS the capture.
//
// The non-vacuity guard is `allocs() > 0`. The pre-grow itself is a driver
// allocation, made OUTSIDE the region, so the fixed driver must allocate in this
// step and must allocate none of it under capture. A case that measured zero
// because nothing ran at all would fail that line.

namespace {

// THE LANE THESE TWO CASES DO NOT APPLY TO, and why they say so with a SKIP
// rather than with a pass.
//
// `VT_POOL_BYPASS=1` turns every `DevicePool::Get` into a raw `Backend::Alloc`
// and every `Put` into a real `Free` (`device_pool.h:113-127`, `:246-252`).
// In that lane there is no free list, `Drain` therefore reports 0,
// and `PreGrowForCapture` returns before it grows anything (`:429`). The
// guarantee these cases assert -- that the captured region performs no driver
// allocation -- is FALSE BY DESIGN there, and false identically for the fixed
// and the unfixed driver. It is not a defect the case has found; it is the
// lane deliberately reinstating the per-op alloc/free storm the pool exists to
// remove, so that ASan can tell the pool's retained cache apart from a leak and
// can see a use-after-free of a released block.
//
// `.github/workflows/ci.yml:1598` sets it for the `sanitize-cpu` job, for BOTH
// the `address,undefined` and the `thread` lane. So this is not a TSan
// interaction and not an allocator-footprint effect: the identical
// `REQUIRE( 0 > 0 )` reproduces on an ordinary non-sanitized Release build with
// `VT_POOL_BYPASS=1` in the environment and nothing else changed.
//
// WHAT THE PRECONDITION GUARD ACTUALLY BOUGHT, stated accurately because the
// first reading of the CI failure got it backwards. Without
// `REQUIRE(freed > 0)` these cases do NOT sail through asserting nothing: they
// reach the capture assertion and FAIL it, at 107 and 195 driver allocations
// inside the capture, because under bypass every `Get` is a driver call. The
// guard converts an inevitable failure that names the SYMPTOM into one that
// names the PRECONDITION. That is worth having, and it is a smaller claim than
// "it prevented a vacuous merge".
//
// SO THE CAPTURE-ALLOCATION GUARANTEE IS NOT EXERCISED BY `sanitize-cpu`, on
// either lane, and nothing in this file can change that: `Bypass()` is read
// once into a process-wide function-local static (`device_pool.h:480-486`), so
// no scope, no locally constructed `DevicePool` and no `ActivePoolScope` can
// opt one case back into pooled behaviour. Unsetting the variable for these
// cases would be worse than the gap -- the pool would retain blocks that the
// job's own `ASAN_OPTIONS=detect_leaks=1` then reports as leaks, which is the
// exact confusion the bypass was introduced to prevent.
//
// The predicate MIRRORS `DevicePool::Bypass()` byte for byte rather than
// approximating it. If the two ever disagree, a case would run in a lane whose
// pool is bypassed, or skip in one whose pool is not.
bool PoolBypassLane() {
  const char* e = std::getenv("VT_POOL_BYPASS");
  return e != nullptr && e[0] == '1';
}

}  // namespace


namespace {

CommonAttentionMetadata FullAttnDecodeMeta(int32_t batch, int32_t position) {
  auto am = DecodeAttnMeta(position);
  am.num_reqs = am.num_actual_tokens = batch;
  am.query_start_loc.clear();
  am.seq_lens.assign(static_cast<size_t>(batch), position + 1);
  am.block_table_tensor.clear();
  am.slot_mapping.clear();
  for (int32_t r = 0; r < batch; ++r) {
    am.query_start_loc.push_back(r);
    am.block_table_tensor.push_back(r);
    am.slot_mapping.push_back(static_cast<int64_t>(r) * 16 + position);
  }
  am.query_start_loc.push_back(batch);
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens_cpu = am.seq_lens;
  return am;
}

GDNAttentionMetadata UnusedGdnMeta(int32_t batch, int mode) {
  GDNAttentionMetadata gm;
  if (mode == 0) return gm;
  gm.num_decodes = gm.num_decode_tokens = gm.num_actual_tokens = batch;
  gm.non_spec_state_indices_tensor = std::vector<int32_t>(static_cast<size_t>(batch));
  for (int32_t r = 0; r < batch; ++r)
    (*gm.non_spec_state_indices_tensor)[static_cast<size_t>(r)] = r;
  if (mode == 2) {
    // No GDN consumer can read any of this metadata. In particular, graph
    // padding must not copy this vector into an S-entry destination.
    gm.non_spec_state_indices_tensor->assign(4096, 777);
    gm.num_actual_tokens = -7;
    gm.num_spec_decodes = 999;
  }
  return gm;
}

template <class Graph, class Weights>
void FullAttnGraphRouting(const HfConfig& c, const Weights& w) {
  for (bool capture : {false, true}) {
    CAPTURE(capture);
    for (bool async : {false, true}) {
      CAPTURE(async);
      for (int mode : {0, 1, 2}) {
        CAPTURE(mode);
        for (int32_t batch : {1, 3}) {
          CAPTURE(batch);
          StaticGraphCpu harness(capture);
          const ScopedEnv async_mode("VT_ASYNC_EXECUTOR", async ? "1" : "0");
          // Padding from three requests to four needs no GDN state I/O.
          const ScopedEnv no_indexed_state("VT_GDN_INDEXED_STATE_IO", "0");
          CachePool pool(c, 4, 16);
          REQUIRE(pool.gdn_state.empty());
          vt::Queue q = Q();
          vt::ResetGraphBreakStats();
          vt::ResetStepInputStats();
          Graph graph(w, c, q, 4);
          for (int32_t step = 0; step < 6; ++step) {
            const std::vector<int32_t> ids(static_cast<size_t>(batch), 11 + step);
            const std::vector<int32_t> positions(static_cast<size_t>(batch), step);
            const auto logits = graph.Step(ids, positions, FullAttnDecodeMeta(batch, step),
                                            UnusedGdnMeta(batch, mode), pool.attn_kv,
                                            pool.gdn_state);
            REQUIRE(logits.rows == batch);
            REQUIRE(logits.vocab == c.vocab_size);
            REQUIRE(logits.on_device());
            if (step == 0) {
              CHECK_FALSE(graph.captured());
              CHECK(vt::GetGraphBreakStats().segments_captured == 0);
            }
          }
          const auto graphs = vt::GetGraphBreakStats();
          const auto inputs = vt::GetStepInputStats();
          if (capture) {
            CHECK(graph.captured());
            CHECK(graphs.segments_captured == (async ? 2 : 1));
            CHECK(graphs.replays == (async ? 4 : 5));
            CHECK(graph.replay_count() == (async ? 4 : 5));
            // Exactly five generic inputs per slot. A sixth input means the
            // no-consumer path still binds recurrent state indices.
            CHECK(inputs.binds == (async ? 10 : 0));
            CHECK(inputs.host_refreshes == (async ? 10 : 0));
          } else {
            CHECK_FALSE(graph.captured());
            CHECK(graphs.segments_captured == 0);
            CHECK(graphs.replays == 0);
            CHECK(inputs.binds == 0);
            CHECK(inputs.host_refreshes == 0);
          }
          CHECK(inputs.device_refreshes == 0);
        }
      }
    }
  }
  MESSAGE("CPU fake graph: cold, capture, staging, replay, and fallback routing only; "
          "no GPU replay numerics");
}

template <class Graph, class Weights>
void FullAttnGraphShapeGuards(const HfConfig& c, const Weights& w) {
  StaticGraphCpu harness;
  CachePool pool(c, 4, 16);
  vt::Queue q = Q();
  Graph graph(w, c, q, 4);
  auto am = DecodeAttnMeta(0);
  const char* expected = nullptr;
  SUBCASE("oversized slot mapping") {
    am.slot_mapping.push_back(0);
    expected = "slot_mapping";
  }
  SUBCASE("oversized sequence lengths") {
    am.seq_lens.push_back(1);
    expected = "full-attn metadata shapes";
  }
  SUBCASE("oversized query offsets") {
    am.query_start_loc.push_back(1);
    expected = "full-attn metadata shapes";
  }
  SUBCASE("oversized block table") {
    am.block_table_tensor.push_back(0);
    expected = "block table";
  }
  SUBCASE("query offsets do not cover the batch") {
    am.query_start_loc.back() = 2;
    expected = "query offsets";
  }
  SUBCASE("query offsets start after zero") {
    am.query_start_loc = {1, 1};
    am.query_start_loc_cpu = am.query_start_loc;
    expected = "query offsets";
  }
  SUBCASE("query offsets descend inside the batch") {
    am = FullAttnDecodeMeta(3, 0);
    am.query_start_loc = {0, 2, 1, 3};
    am.query_start_loc_cpu = am.query_start_loc;
    expected = "query offsets";
  }
  REQUIRE(expected != nullptr);
  const std::vector<int32_t> ids(static_cast<size_t>(am.num_actual_tokens), 11);
  const std::vector<int32_t> positions(static_cast<size_t>(am.num_actual_tokens), 0);
  CHECK_THROWS_WITH_AS(graph.Step(ids, positions, am, {}, pool.attn_kv, pool.gdn_state),
                       doctest::Contains(expected), std::runtime_error);
}

// A scoped routing witness, not a new backend. The real CPU backend and all
// native CPU arithmetic remain in use. Only the MoE registry's existing FP4
// capability predicate receives a test answer. The fake graph replays no math.
class ScopedFp4GraphCapability final : public vllm::platforms::Platform {
 public:
  ScopedFp4GraphCapability()
      : previous_(vllm::platforms::GetPlatform(vt::DeviceType::kCPU)) {
    vllm::platforms::RegisterPlatform(vt::DeviceType::kCPU, this);
  }
  ~ScopedFp4GraphCapability() override {
    vllm::platforms::RegisterPlatform(vt::DeviceType::kCPU, &previous_);
  }
  vt::DeviceType device_type() const override { return previous_.device_type(); }
  vt::Backend& backend() const override { return previous_.backend(); }
  vllm::platforms::DeviceCapability get_device_capability() const override {
    return previous_.get_device_capability();
  }
  std::vector<vt::DType> supported_dtypes() const override {
    return previous_.supported_dtypes();
  }
  vllm::platforms::ResidencyPolicy residency_policy() const override {
    return previous_.residency_policy();
  }
  bool support_static_graph_mode() const override { return true; }
  bool cutlass_fp4_supported() const override { return true; }

 private:
  vllm::platforms::Platform& previous_;
};

// The existing MakeNvfp4W4A16 test pattern, with finite nonzero E2M1 values and
// E4M3 scales. K and N are the CPU model's real expert projection dimensions.
vllm::Nvfp4Weight RoutingFp4(int64_t n, int64_t k) {
  vllm::Nvfp4Weight w;
  w.n = n;
  w.k = k;
  w.scale2 = 1.0F;
  w.packed.dtype = w.scale.dtype = DType::kI8;
  w.packed.rank = w.scale.rank = 2;
  w.packed.shape[0] = w.scale.shape[0] = n;
  w.packed.shape[1] = k / 2;
  w.scale.shape[1] = k / 16;
  w.packed.bytes.assign(static_cast<size_t>(n * k / 2), 0x21);
  w.scale.bytes.assign(static_cast<size_t>(n * k / 16), 0x20);
  return w;
}

void AddRoutingFp4(Qwen3_5MoeWeights& w, const HfConfig& c) {
  for (auto& layer : w.layers) {
    auto& moe = layer.moe;
    moe.expert_gate.clear();
    moe.expert_up.clear();
    moe.expert_down.clear();
    for (int64_t expert = 0; expert < c.num_experts; ++expert) {
      moe.expert_gate_fp4.push_back(RoutingFp4(c.moe_intermediate_size, c.hidden_size));
      moe.expert_up_fp4.push_back(RoutingFp4(c.moe_intermediate_size, c.hidden_size));
      moe.expert_down_fp4.push_back(RoutingFp4(c.hidden_size, c.moe_intermediate_size));
    }
    moe.shared_gate_proj = OwnedTensor{};
    moe.shared_up_proj = OwnedTensor{};
    moe.shared_down_proj = OwnedTensor{};
    moe.shared_gate_proj_fp4 = RoutingFp4(c.shared_expert_intermediate_size, c.hidden_size);
    moe.shared_up_proj_fp4 = RoutingFp4(c.shared_expert_intermediate_size, c.hidden_size);
    moe.shared_down_proj_fp4 = RoutingFp4(c.hidden_size, c.shared_expert_intermediate_size);
  }
}

void RegistryFullAttnGraph(vllm::LoadedModel& model, const HfConfig& c, bool expect_graph) {
  CachePool pool(c, 4, 16);
  vt::Queue q = Q();
  const std::vector<int32_t> indices;
  CHECK(model.registration().architecture == c.architectures.front());
  CHECK(model.registration().factory == vllm::RegistrationFor(c.architectures.front()).factory);
  vt::ResetGraphBreakStats();
  vt::ResetStepInputStats();
  for (int32_t step = 0; step < 6; ++step) {
    const std::vector<int32_t> ids = {11 + step}, positions = {step};
    const auto am = DecodeAttnMeta(step);
    const auto gm = UnusedGdnMeta(1, 2);
    vllm::ModelForwardInput input{ids, positions, am, gm, pool.attn_kv,
                                  pool.gdn_state, c, q, indices};
    input.num_reqs = 1;
    input.pure_decode = true;
    input.uniform_query_len = 1;
    // The legacy planner group carries capacity even without a state consumer.
    input.gdn_state_slots = 4;
    const auto logits = vllm::ModelRegistry::Forward(model, input);
    REQUIRE(logits.rows == 1);
    REQUIRE(logits.vocab == c.vocab_size);
  }
  const auto graphs = vt::GetGraphBreakStats();
  CHECK(graphs.segments_captured == (expect_graph ? 2 : 0));
  CHECK(graphs.replays == (expect_graph ? 4 : 0));
  CHECK(vt::GetStepInputStats().binds == (expect_graph ? 10 : 0));
  CHECK(vt::GetStepInputStats().host_refreshes == (expect_graph ? 10 : 0));
}

}  // namespace

TEST_CASE("full-attention dense graph ignores unused GDN state") {
  HfConfig c = dense::TinyConfig();
  c.num_hidden_layers = 1;
  c.layer_types = {"full_attention"};
  const auto w = dense::MakeWeights(c);
  FullAttnGraphRouting<vllm::Qwen3_5DenseDecodeGraph>(c, w);
}

TEST_CASE("full-attention MoE graph ignores unused GDN state") {
  HfConfig c = TinyConfig();
  c.num_hidden_layers = 1;
  c.layer_types = {"full_attention"};
  const auto w = MakeWeights(c);
  FullAttnGraphRouting<vllm::Qwen3_5DecodeGraph>(c, w);
}

TEST_CASE("full-attention dense graph keeps attention shape guards") {
  HfConfig c = dense::TinyConfig();
  c.num_hidden_layers = 1;
  c.layer_types = {"full_attention"};
  const auto w = dense::MakeWeights(c);
  FullAttnGraphShapeGuards<vllm::Qwen3_5DenseDecodeGraph>(c, w);
}

TEST_CASE("full-attention MoE graph keeps attention shape guards") {
  HfConfig c = TinyConfig();
  c.num_hidden_layers = 1;
  c.layer_types = {"full_attention"};
  const auto w = MakeWeights(c);
  FullAttnGraphShapeGuards<vllm::Qwen3_5DecodeGraph>(c, w);
}

TEST_CASE("full-attention registry reaches the dense graph Step") {
  StaticGraphCpu harness;
  const ScopedEnv async("VT_ASYNC_EXECUTOR", "1");
  HfConfig c = dense::TinyConfig();
  c.num_hidden_layers = 1;
  c.layer_types = {"full_attention"};
  const auto w = dense::MakeWeights(c);
  auto model = vllm::BorrowQwen3_5DenseLoadedModel(w);
  RegistryFullAttnGraph(*model, c, true);
}

TEST_CASE("full-attention registry reaches the MoE graph Step through its FP4 predicate") {
  StaticGraphCpu harness;
  const ScopedEnv async("VT_ASYNC_EXECUTOR", "1");
  HfConfig c = TinyConfig();
  c.num_hidden_layers = 1;
  c.layer_types = {"full_attention"};
  auto w = MakeWeights(c);
  AddRoutingFp4(w, c);
  // Control: valid FP4 CPU arithmetic alone does not satisfy the registry's
  // platform predicate, so the registered forward stays eager.
  {
    auto model = vllm::BorrowQwen3_5MoeLoadedModel(w);
    RegistryFullAttnGraph(*model, c, false);
  }
  {
    ScopedFp4GraphCapability fake_capability;
    auto model = vllm::BorrowQwen3_5MoeLoadedModel(w);
    RegistryFullAttnGraph(*model, c, true);
  }
  MESSAGE("MoE registry witness uses a fake CPU FP4 capability and fake graph replay; "
          "no GPU replay numerics");
}

TEST_CASE("#2029: a NON-speculative Qwen3_5DenseDecodeGraph capture allocates nothing"
          " [pooled lane only -- SKIPPED under VT_POOL_BYPASS, where the pool is"
          " disabled and the guarantee is false by design]"
          * doctest::skip(PoolBypassLane())) {
  const HfConfig c = dense::TinyConfig();
  const dense::Qwen3_5DenseWeights w = dense::MakeWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");
  REQUIRE_MESSAGE(std::getenv("VT_ASYNC_EXECUTOR") == nullptr,
                  "this gate is about the DEFAULT lane; VT_ASYNC_EXECUTOR is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16);
  vllm::Qwen3_5DenseDecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  // COLD: the one eager run at this shape, and the step whose per-class peak the
  // driver records as `s.demand`. `num_spec_decodes` is 0, so `spec_step` is
  // false and this is the lane the default server is on.
  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  REQUIRE_FALSE(graph.captured());

  const size_t freed = vllm::Pool(harness.backend()).Drain(harness.backend());
  REQUIRE_MESSAGE(freed > 0,
                  "the cold step returned no block to the pool, so this case would "
                  "assert nothing about a short free list");
  harness.backend().ResetCounters();

  // WARM: the capture.
  graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  REQUIRE(graph.captured());
  CHECK(harness.backend().Count("Begin") == 1);
  CHECK(harness.backend().Count("EndCaptureGraph") == 1);
  MESSAGE("dense capture step: " << harness.backend().allocs_during_capture()
                                 << " driver allocations INSIDE the capture, "
                                 << harness.backend().allocs() << " in the step");
  CHECK(harness.backend().allocs() > 0);
  CHECK(harness.backend().allocs_during_capture() == 0);
}

TEST_CASE("#2029: a NON-speculative Qwen3_5DecodeGraph capture allocates nothing"
          " [pooled lane only -- SKIPPED under VT_POOL_BYPASS, where the pool is"
          " disabled and the guarantee is false by design]"
          * doctest::skip(PoolBypassLane())) {
  const HfConfig c = TinyConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  REQUIRE_MESSAGE(vt::GraphCaptureEnabled(),
                  "this gate needs the CAPTURING lane; VLLM_CPP_CUDAGRAPH=0 is set");
  REQUIRE_MESSAGE(std::getenv("VT_ASYNC_EXECUTOR") == nullptr,
                  "this gate is about the DEFAULT lane; VT_ASYNC_EXECUTOR is set");

  StaticGraphCpu harness;
  vt::Queue q = Q();
  CachePool pool(c, /*num_blocks=*/4, /*block_size=*/16);
  vllm::Qwen3_5DecodeGraph graph(w, c, q, /*max_num_reqs=*/4);

  graph.Step({11}, {0}, DecodeAttnMeta(0), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  REQUIRE_FALSE(graph.captured());

  const size_t freed = vllm::Pool(harness.backend()).Drain(harness.backend());
  REQUIRE_MESSAGE(freed > 0,
                  "the cold step returned no block to the pool, so this case would "
                  "assert nothing about a short free list");
  harness.backend().ResetCounters();

  graph.Step({12}, {1}, DecodeAttnMeta(1), DecodeGdnMeta(), pool.attn_kv,
             pool.gdn_state);
  REQUIRE(graph.captured());
  CHECK(harness.backend().Count("Begin") == 1);
  CHECK(harness.backend().Count("EndCaptureGraph") == 1);
  MESSAGE("MoE capture step: " << harness.backend().allocs_during_capture()
                               << " driver allocations INSIDE the capture, "
                               << harness.backend().allocs() << " in the step");
  CHECK(harness.backend().allocs() > 0);
  CHECK(harness.backend().allocs_during_capture() == 0);
}


namespace {
struct XpuPolicyEnv {
  static constexpr const char* names[] = {"VT_XPU_ATTENTION", "VT_XPU_XE2_VERIFY",
      "VT_XPU_ATTN_SPLIT_EXTENDED", "VT_XPU_ATTN_SPLIT_SPAN", "VT_XPU_ATTN_SPLIT_MAX_PARTS",
      "VT_XPU_ATTN_SPLIT_ACTIVE_PAGE_CAP", "VT_XPU_ATTN_SPLIT_REDUCE",
      "VT_XPU_ATTN_PREFILL_TILE", "VT_XPU_ATTN_PROBABILITY"};
  std::array<std::string, 9> values;
  std::array<bool, 9> present{};
  XpuPolicyEnv() {
    for (size_t i = 0; i < 9; ++i) {
      const char* v = std::getenv(names[i]); present[i] = v != nullptr;
      if (v) values[i] = v;
      unsetenv(names[i]);
    }
  }
  ~XpuPolicyEnv() {
    for (size_t i = 0; i < 9; ++i)
      if (present[i]) setenv(names[i], values[i].c_str(), 1); else unsetenv(names[i]);
  }
};
PagedKvCache XpuPolicyKv(void* data) {
  PagedKvCache kv;
  kv.data = data; kv.dtype = DType::kI8; kv.num_blocks = 32; kv.block_size = 1600;
  kv.num_kv_heads = 4; kv.head_size = 256; kv.fp8_kind = vt::Fp8KVCacheDataType::kFp8E4M3;
  return kv;
}
}

TEST_CASE("P1: XPU attention graph policy keys routes partitions and every cache binding") {
  XpuPolicyEnv env;
  char a = 0, b = 0, c = 0;
  std::vector<PagedKvCache> caches{XpuPolicyKv(&a), XpuPolicyKv(&b)};
  const auto key = [&](int length = 4100) {
    return vllm::detail::BuildXpuAttentionGraphPolicy(4, 4, length, true, caches);
  };
  const auto original = key();
  CHECK(key(4799) == original); CHECK(key(4800) == original); CHECK_FALSE(key(4801) == original);
  for (size_t i = 0; i < 9; ++i) {
    const char* changed[] = {"verify", "1", "0", "64", "64", "0", "scalar", "q16", "residual"};
    REQUIRE(setenv(XpuPolicyEnv::names[i], changed[i], 1) == 0);
    CHECK_FALSE(key() == original); unsetenv(XpuPolicyEnv::names[i]); CHECK(key() == original);
  }
  const auto saved = caches[1];
  for (int field = 0; field < 11; ++field) {
    caches[1] = saved;
    switch (field) {
      case 0: caches[1].data = &c; break;
      case 1: caches[1].dtype = DType::kF16; break;
      case 2: ++caches[1].num_blocks; break;
      case 3: caches[1].block_size = 1664; break;
      case 4: caches[1].num_kv_heads = 2; break;
      case 5: caches[1].head_size = 128; break;
      case 6: caches[1].head_size_v = 128; break;
      case 7: caches[1].fp8_kind = vt::Fp8KVCacheDataType::kAuto; break;
      case 8: caches[1].k_scale = 0.5f; break;
      case 9: caches[1].v_scale = 0.5f; break;
      case 10: caches[1].page_size_bytes = 42; break;
    }
    CHECK_FALSE(key() == original);
  }
  caches[1] = saved; CHECK(key() == original);
  CHECK_FALSE(vllm::detail::BuildXpuAttentionGraphPolicy(4, 4, 4100, false, caches) == original);
  CHECK_FALSE(vllm::detail::BuildXpuAttentionGraphPolicy(1, 1, 960, true, caches) ==
              vllm::detail::BuildXpuAttentionGraphPolicy(1, 1, 961, true, caches));
  CHECK_FALSE(key(4095) == key(4096));
  REQUIRE(setenv("VT_XPU_ATTN_SPLIT_ACTIVE_PAGE_CAP", "0", 1) == 0);
  CHECK(key(4800) == key(4801));
  REQUIRE(setenv("VT_XPU_XE2_VERIFY", "1", 1) == 0);
  CHECK_FALSE(key(4800) == key(4801));  // Packed bound still retires when generic cap is off.
}

TEST_CASE("P1: XPU attention graph policy bounds only qualified batched packed verification") {
  XpuPolicyEnv env; char a = 0; auto kv = XpuPolicyKv(&a);
  CommonAttentionMetadata meta;
  meta.num_reqs = 4; meta.num_actual_tokens = 16; meta.max_query_len = 4;
  meta.query_start_loc = {0, 4, 8, 12, 16}; meta.max_seq_len = 4100; meta.causal = true;
  const auto bound = [&] { return vllm::detail::XpuBatchedVerifyContextBound(kv, true, 24, 16, meta); };
  CHECK(bound() == 4100); REQUIRE(setenv("VT_XPU_XE2_VERIFY", "1", 1) == 0);
  CHECK(bound() == 4800); kv.head_size_v = 256; CHECK(bound() == 4800);
  kv.head_size_v = 128; CHECK(bound() == 4100); kv.head_size_v = 0;
  meta.max_seq_len = 4800; CHECK(bound() == 4800);
  meta.max_seq_len = 4801; CHECK(bound() == 6400);
  meta.max_seq_len = 4100; kv.k_scale = 0.5f; CHECK(bound() == 4100); kv.k_scale = 1;
  CHECK(vllm::detail::XpuBatchedVerifyContextBound(kv, false, 24, 16, meta) == 4100);
  CHECK(vllm::detail::XpuBatchedVerifyContextBound(kv, true, 8, 16, meta) == 4100);
  meta.query_start_loc = {0, 2, 6, 10, 14}; CHECK(bound() == 4100);
  meta.query_start_loc = {0, 4, 8, 12, 16}; REQUIRE(setenv("VT_XPU_ATTENTION", "split", 1) == 0);
  CHECK(bound() == 4100); REQUIRE(setenv("VT_XPU_ATTENTION", "verify", 1) == 0);
  REQUIRE(setenv("VT_XPU_XE2_VERIFY", "0", 1) == 0); CHECK(bound() == 4800);
  kv.block_size = 1664; CHECK(bound() == 4992);
  meta.num_reqs = 1; CHECK(bound() == 4100);
  kv.block_size = 1600;
  for (int requests : {2, 3}) {
    meta.num_reqs = requests; meta.num_actual_tokens = requests * 4;
    meta.query_start_loc.clear();
    for (int r = 0; r <= requests; ++r) meta.query_start_loc.push_back(r * 4);
    const auto partial = [&] { return vllm::detail::XpuBatchedVerifyContextBound(kv, true, 24, requests * 4, meta); };
    CHECK(partial() == 4800);
    meta.max_seq_len = 4800; CHECK(partial() == 4800);
    meta.max_seq_len = 4801; CHECK(partial() == 6400);
    meta.max_seq_len = 32768; CHECK(partial() == 33600);
    meta.query_start_loc[1] = 3; CHECK(partial() == 32768); meta.query_start_loc[1] = 4;
    CHECK(vllm::detail::XpuBatchedVerifyContextBound(kv, true, 24, requests * 4 - 1, meta) == 32768);
    REQUIRE(setenv("VT_XPU_ATTENTION", "split", 1) == 0); CHECK(partial() == 32768);
    REQUIRE(setenv("VT_XPU_ATTENTION", "verify", 1) == 0); meta.max_seq_len = 4100;
  }
}


#ifdef VLLM_CPP_XPU
namespace {
// Real XPU replay through the model driver, with reduced materialized weights.
// This qualifies ownership and exact eager/replay state, not checkpoint EXL3
// arithmetic, a prefilling trajectory, or serving speed. Attention alone has
// the production 24Q/4KV/D256/page1600 geometry; GDN/MLP/hidden stay small.
struct P1XpuQueue {
  vt::Queue q = vt::CreateQueue({vt::DeviceType::kXPU, 0});
  ~P1XpuQueue() { vt::DestroyQueue(q); }
};
struct P1XpuBuffer {
  vt::Queue& q;
  size_t bytes;
  void* data;
  P1XpuBuffer(vt::Queue& queue, size_t count)
      : q(queue), bytes(count), data(vt::Alloc(q.device, count)) {
    vt::GetBackend(q.device).Memset(q, data, 0, bytes);
  }
  ~P1XpuBuffer() { vt::Free(q.device, data); }
  P1XpuBuffer(const P1XpuBuffer&) = delete;
  std::vector<unsigned char> Read() const {
    std::vector<unsigned char> result(bytes);
    auto& b = vt::GetBackend(q.device);
    b.Copy(q, result.data(), data, bytes); b.Synchronize(q);
    return result;
  }
  void Write(const std::vector<unsigned char>& contents) {
    REQUIRE(contents.size() == bytes);
    auto& b = vt::GetBackend(q.device);
    b.Copy(q, data, contents.data(), bytes); b.Synchronize(q);
  }
};
void P1ToF16(OwnedTensor& t) {
  if (t.dtype != DType::kBF16) return;
  for (size_t i = 0; i < t.bytes.size(); i += 2) {
    uint16_t bits; std::memcpy(&bits, t.bytes.data() + i, 2);
    bits = vt::F32ToF16(vt::BF16ToF32(bits));
    std::memcpy(t.bytes.data() + i, &bits, 2);
  }
  t.dtype = DType::kF16;
}
void P1ToNK(OwnedTensor& t) {
  REQUIRE(t.rank == 2); REQUIRE(t.dtype == DType::kF16);
  const int64_t k = t.shape[0], n = t.shape[1];
  auto bytes = t.bytes;
  for (int64_t i = 0; i < k; ++i)
    for (int64_t j = 0; j < n; ++j)
      std::memcpy(t.bytes.data() + (j * k + i) * 2, bytes.data() + (i * n + j) * 2, 2);
  t.shape[0] = n; t.shape[1] = k; t.nk = true;
}
dense::Qwen3_5DenseWeights P1XpuWeights(const HfConfig& c) {
  auto w = dense::MakeWeights(c);
  // Explicit synthetic materialized fixture selecting the same FP16 Dev as
  // EXL3. No quantized shard is provided or substituted by captured values.
  w.exl3_checkpoint = true;
  w.precision.activation = w.precision.dense_weight = DType::kF16;
  w.precision.gdn_conv_state = w.precision.gdn_recurrent_state = DType::kF32;
  P1ToF16(w.embed_tokens); P1ToF16(w.final_norm); P1ToF16(w.lm_head);
  for (auto& l : w.layers) {
    P1ToF16(l.input_layernorm); P1ToF16(l.post_attention_layernorm);
    for (auto* t : {&l.mlp.gate_proj, &l.mlp.up_proj, &l.mlp.down_proj,
                   &l.gdn.in_proj_qkv, &l.gdn.in_proj_z, &l.gdn.in_proj_b,
                   &l.gdn.in_proj_a, &l.gdn.conv1d_weight, &l.gdn.norm_weight,
                   &l.gdn.out_proj, &l.attn.q_proj, &l.attn.k_proj,
                   &l.attn.v_proj, &l.attn.o_proj, &l.attn.q_norm,
                   &l.attn.k_norm}) P1ToF16(*t);
    // Raw-NK projections emit the scoped model dtype, as the checkpoint
    // projection does. Legacy synthetic KN projections deliberately emit F32
    // and are incompatible with this FP16 Q/K + FP8 KV fixture.
    if (!l.is_linear_attention)
      for (auto* t : {&l.attn.q_proj, &l.attn.k_proj, &l.attn.v_proj}) P1ToNK(*t);
  }
  return w;
}
struct P1XpuCaches {
  vt::Queue& q;
  std::vector<std::unique_ptr<P1XpuBuffer>> kv_buffers, state_buffers;
  std::vector<PagedKvCache> kv;
  std::vector<GdnStateCache> gdn;
  P1XpuCaches(vt::Queue& queue, const HfConfig& c) : q(queue) {
    for (const auto& type : c.layer_types) {
      if (type == "full_attention") {
        auto cache = XpuPolicyKv(nullptr); cache.num_blocks = 16;
        cache.head_size_v = 256;
        kv_buffers.push_back(std::make_unique<P1XpuBuffer>(q,
            size_t(cache.num_blocks * 2 * cache.block_size * 4 * 256)));
        cache.data = kv_buffers.back()->data; kv.push_back(cache);
      } else {
        GdnStateCache state;
        const int64_t conv_dim = 2 * c.linear_num_key_heads * c.linear_key_head_dim +
            c.linear_num_value_heads * c.linear_value_head_dim;
        state.ssm_state = vt::Tensor::Contiguous(nullptr, DType::kF32, q.device,
            {16, c.linear_num_value_heads, c.linear_value_head_dim, c.linear_key_head_dim});
        state.conv_state = vt::Tensor::Contiguous(nullptr, DType::kF32, q.device,
            {16, conv_dim, 6});
        for (auto* t : {&state.ssm_state, &state.conv_state}) {
          state_buffers.push_back(std::make_unique<P1XpuBuffer>(q, t->Bytes()));
          t->data = state_buffers.back()->data;
        }
        gdn.push_back(state);
      }
    }
    vt::GetBackend(q.device).Synchronize(q);
  }
};
int32_t P1PhysicalPage(int request, int page) { return request * 4 + 3 - page; }
CommonAttentionMetadata P1XpuMeta(int32_t pos, int32_t page, int requests = 4) {
  CommonAttentionMetadata m;
  m.num_reqs = requests; m.num_actual_tokens = requests * 4; m.max_query_len = 4;
  for (int r = 0; r <= requests; ++r) m.query_start_loc.push_back(r * 4);
  m.query_start_loc_cpu = m.query_start_loc;
  // Unequal contexts share the same host partition/bound, but have distinct
  // disjoint request pages and fresh lengths/positions/slots on every replay.
  for (int r = 0; r < requests; ++r) m.seq_lens.push_back(pos + 4 - r * 7);
  m.seq_lens_cpu = m.seq_lens; m.max_seq_len = pos + 4;
  m.block_table_num_cols = 164;
  m.block_table_tensor.assign(requests * 164, -1);
  for (int r = 0; r < requests; ++r) {
    const int active = (m.seq_lens[r] + page - 1) / page;
    for (int p = 0; p < active; ++p)
      m.block_table_tensor[r * 164 + p] = P1PhysicalPage(r, p);
    for (int j = 0; j < 4; ++j) {
      const int position = pos + j - r * 7;
      m.slot_mapping.push_back(P1PhysicalPage(r, position / page) * page + position % page);
    }
  }
  m.causal = true; return m;
}
std::vector<unsigned char> P1XpuLogits(const vllm::ForwardLogits& logits, vt::Queue& q, int rows = 16) {
  REQUIRE(logits.rows == rows); REQUIRE(logits.vocab == 40);
  REQUIRE(logits.device_tensor.dtype == DType::kF32);
  std::vector<unsigned char> result(rows * 40 * sizeof(float));
  auto& b = vt::GetBackend(q.device);
  b.Copy(q, result.data(), logits.device_tensor.data, result.size()); b.Synchronize(q);
  bool nonzero = false;
  for (size_t i = 0; i < result.size(); i += sizeof(float)) {
    float value; std::memcpy(&value, result.data() + i, sizeof(float));
    REQUIRE(std::isfinite(value)); nonzero |= value != 0;
  }
  REQUIRE(nonzero); return result;
}
// Repack logical positions to a different physical page width, preserving all
// existing KV values. Old allocations stay alive as stale-write sentinels.
std::vector<unsigned char> P1Repage(const std::vector<unsigned char>& old,
                                  int old_page, int page) {
  const size_t row = 4 * 256;
  std::vector<unsigned char> result(size_t(16 * 2 * page) * row, 0);
  for (int r = 0; r < 4; ++r)
    for (int p = 0; p < 4 * old_page; ++p)
      for (int plane = 0; plane < 2; ++plane) {
        const size_t from = (size_t(P1PhysicalPage(r, p / old_page)) * 2 * old_page +
                            plane * old_page + p % old_page) * row;
        const size_t to = (size_t(P1PhysicalPage(r, p / page)) * 2 * page +
                          plane * page + p % page) * row;
        std::memcpy(result.data() + to, old.data() + from, row);
      }
  return result;
}
}  // namespace

TEST_CASE("P1: XPU dense model retires route and cache graphs with exact own-state replay") {
  if (std::getenv("VT_B70_EXL3_MODEL_POLICY_TEST") == nullptr) {
    MESSAGE("Set VT_B70_EXL3_MODEL_POLICY_TEST=1 for the bounded real XPU model-owner test");
    return;
  }
  REQUIRE(vt::xpu::DeviceCount() > 0);
  XpuPolicyEnv env;
  REQUIRE(setenv("VT_XPU_ATTENTION", "auto", 1) == 0);
  REQUIRE(setenv("VT_XPU_XE2_VERIFY", "1", 1) == 0);
  P1XpuQueue queue; auto& q = queue.q;
  REQUIRE(vt::GetBackend(q.device).SupportsGraphCapture());
  REQUIRE(vt::GraphCaptureEnabled());
  auto config = dense::TinyConfig();
  config.layer_types = {"linear_attention", "full_attention", "linear_attention", "full_attention"};
  config.num_attention_heads = 24; config.num_key_value_heads = 4; config.head_dim = 256;
  config.max_position_embeddings = 6656;
  auto weights = P1XpuWeights(config);
  P1XpuCaches eager(q, config), captured(q, config);
  // The graph is declared after every baked input owner. Retired allocations
  // and their byte snapshots remain valid until all graph work has completed.
  std::vector<std::unique_ptr<P1XpuBuffer>> retired;
  std::vector<std::vector<unsigned char>> retired_bytes;
  vt::ResetGraphBreakStats();
  auto graph = std::make_unique<vllm::Qwen3_5DenseDecodeGraph>(weights, config, q, 4);
  int step = 0, captures = 0, launches = 0;
  const auto phase = [&](const char* name, int pos, int count, int expected_captures,
                         int requests = 4, int expected_live_graphs = 2) {
    auto gm = SpecGdnMeta(requests, 4);
    for (int i = 0; i < requests * 4; ++i) (*gm.spec_state_indices_tensor)[i] = i;
    const auto before = vt::GetGraphBreakStats();
    vt::xpu::DrainProfileEvents();
    const int64_t replay_before = graph->replay_count();
    for (int i = 0; i < count; ++i, ++step) {
      CAPTURE(name);
      CAPTURE(i);
      CAPTURE(step);
      const int page = int(captured.kv.front().block_size);
      auto meta = P1XpuMeta(pos + 4 * i, page, requests);
      std::vector<int32_t> tokens, positions;
      for (int r = 0; r < requests; ++r)
        for (int j = 0; j < 4; ++j) {
          tokens.push_back((step * 3 + r * 11 + j * 5) % 40);
          positions.push_back(pos + 4 * i + j - r * 7);
        }
      std::vector<unsigned char> expected;
      {
        auto logits = vllm::Qwen3_5DenseModel::ForwardDevice(tokens, positions, meta, gm,
            eager.kv, eager.gdn, weights, config, q, {});
        expected = P1XpuLogits(logits, q, requests * 4);
      }
      {
        auto logits = graph->Step(tokens, positions, meta, gm, captured.kv, captured.gdn);
        REQUIRE(P1XpuLogits(logits, q, requests * 4) == expected);
      }
      for (size_t k = 0; k < captured.kv_buffers.size(); ++k)
        REQUIRE(captured.kv_buffers[k]->Read() == eager.kv_buffers[k]->Read());
      for (size_t k = 0; k < captured.state_buffers.size(); ++k)
        REQUIRE(captured.state_buffers[k]->Read() == eager.state_buffers[k]->Read());
    }
    const auto after = vt::GetGraphBreakStats();
    const int got = int(after.segments_captured - before.segments_captured);
    CHECK(got == expected_captures);
    // Each policy transition needs two cold steps (one per ring slot); the
    // capture step executes its graph once and counts as a replay launch.
    CHECK(graph->replay_count() - replay_before == count - expected_captures);
    CHECK(graph->captured()); CHECK(vt::xpu::GetMemoryInfo().graph_count == expected_live_graphs);
    captures += got; launches += int(graph->replay_count() - replay_before);
    const auto events = vt::xpu::DrainProfileEvents();
    const auto packs = std::count_if(events.begin(), events.end(), [](const auto& event) {
      return event.stage == "attention_verify_pack";
    });
    if (const char* profile = std::getenv("VT_XPU_PROFILE"); profile && std::string(profile) == "1") {
      if (std::string(name) == "auto-generic-verifier-off") CHECK(packs == 0);
      else CHECK(packs > 0);
    }
    std::cout << "P1_MODEL_OWNER_PHASE name=" << name << " steps=" << count
              << " captures=" << got << " launches=" << graph->replay_count() - replay_before
              << " live_graphs=" << vt::xpu::GetMemoryInfo().graph_count << " observed_eager_pack_events=" << packs << std::endl;
    for (size_t k = 0; k < retired.size(); ++k) CHECK(retired[k]->Read() == retired_bytes[k]);
  };
  phase("auto-packed-within-page", 4096, 8, 2);
  REQUIRE(setenv("VT_XPU_XE2_VERIFY", "0", 1) == 0);
  phase("auto-generic-verifier-off", 4128, 8, 2);
  REQUIRE(setenv("VT_XPU_ATTENTION", "verify", 1) == 0);
  phase("explicit-packed-verifier-flag-off", 4160, 8, 2);
  // Change the SECOND full-attention binding only. Checking the first layer's
  // pointer (or only cache shapes) would miss this and write the old sentinel.
  for (auto* arm : {&eager, &captured}) {
    auto bytes = arm->kv_buffers[1]->Read();
    auto next = std::make_unique<P1XpuBuffer>(q, bytes.size()); next->Write(bytes);
    retired_bytes.push_back(std::move(bytes)); retired.push_back(std::move(arm->kv_buffers[1]));
    arm->kv_buffers[1] = std::move(next); arm->kv[1].data = arm->kv_buffers[1]->data;
  }
  phase("second-full-attention-cache-base", 4192, 8, 2);
  // Deliberately jump across a zero-filled synthetic prefix, then cross4800
  // while both ring slots are live. This is an owner test, not real prefill.
  phase("active-page-cross-4800", 4780, 12, 2);
  for (auto* arm : {&eager, &captured})
    for (size_t k = 0; k < arm->kv.size(); ++k) {
      auto bytes = arm->kv_buffers[k]->Read();
      auto repacked = P1Repage(bytes, 1600, 1664);
      auto next = std::make_unique<P1XpuBuffer>(q, repacked.size()); next->Write(repacked);
      retired_bytes.push_back(std::move(bytes)); retired.push_back(std::move(arm->kv_buffers[k]));
      arm->kv_buffers[k] = std::move(next); arm->kv[k].data = arm->kv_buffers[k]->data;
      arm->kv[k].block_size = 1664;
    }
  phase("physical-page-layout-1664", 4828, 8, 2);
  // Each logical row shape owns its existing two-slot ring. Establish both C2
  // captures before the boundary so the next phase proves their retirement.
  phase("partial-C3-stable-bound", 4860, 8, 2, 3, 4);
  phase("partial-C2-stable-bound", 4892, 8, 2, 2, 6);
  phase("partial-C2-cross-page", 4980, 12, 2, 2, 6);
  phase("partial-C4-return", 5030, 8, 2, 4, 6);
  CHECK(step == 88); CHECK(captures == 20); CHECK(launches == 68);
  const auto peak = vt::xpu::GetMemoryInfo().peak_allocated_bytes;
  graph.reset();  // destroys every model graph before its baked buffers die
  CHECK(vt::xpu::GetMemoryInfo().graph_count == 0);
  CHECK(vt::xpu::GetMemoryInfo().graph_device_bytes == 0);
  for (size_t k = 0; k < retired.size(); ++k) CHECK(retired[k]->Read() == retired_bytes[k]);
  std::cout << "P1_MODEL_OWNER_DONE steps=" << step << " captures=" << captures
            << " launches=" << launches << " graph_bytes_after=" << vt::xpu::GetMemoryInfo().graph_device_bytes
            << " peak_backend_bytes=" << peak << std::endl;
}
#endif
