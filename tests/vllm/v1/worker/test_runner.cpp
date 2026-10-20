// M1.8 Task 4 — the batched PAGED model runner (GPUModelRunner).
//
// Ported from vllm/v1/worker/gpu/model_runner.py @ e24d1b24 (the execute_model /
// sample_tokens split, KV-cache allocation from KVCacheConfig, the decode-first
// reorder). Drives the runner directly (no scheduler) over a small SYNTHETIC MoE
// model (CPU; the real 35B greedy through the runner on dgx is the milestone
// gate, dgx-pending). Cases:
//   1. KV allocation shape from a fake KVCacheConfig (full-attn buffer + GDN
//      ssm/conv state buffer dims correct, one per layer).
//   2. THE ORDERING IDENTITY GATE (mandatory de-risk): a batch of {1 decode,
//      1 prefill} admitted prefill-first — after the decode-first reorder,
//      logits_indices / the SamplingMetadata row (via the per-req seed) / the
//      attention seq_lens+block_table row / the GDN state index / the write-back
//      slot ALL resolve to the same request.
//   3. Single-request greedy decode over N steps: token sampled + appended, the
//      KV cache grows, the next step reads it (the decode continues).
//   4. A 2-request greedy batch step: each request samples from its OWN logits
//      row (matches the standalone dense argmax).
#include "vllm/v1/worker/gpu/runner.h"

#include <doctest/doctest.h>
#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/config/speculative.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_internal.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/sampling_params.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/kv_cache_dtype.h"
#include "vllm/v1/attention/registry.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/recurrent_prefix_snapshot.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/tensor.h"

using vllm::GdnStateCache;
using vllm::HfConfig;
using vllm::OwnedTensor;
using vllm::PagedKvCache;
using vllm::Qwen3_5DenseWeights;
using vllm::Qwen3_5Model;
using vllm::Qwen3_5MoeWeights;
using vllm::SamplingParams;
using vllm::v1::CachedRequestData;
using vllm::v1::FullAttentionSpec;
using vllm::v1::GPUModelRunner;
using vllm::v1::KVCacheConfig;
using vllm::v1::MambaSpec;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::NewRequestData;
using vllm::v1::SchedulerOutput;
using vt::DType;

namespace {

uint64_t Mix(uint64_t x) {
  x += 0x9E3779B97F4A7C15ULL;
  x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
  x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
  return x ^ (x >> 31);
}
float RandV(uint64_t seed) {
  const double u =
      static_cast<double>(Mix(seed) >> 40) / static_cast<double>(1 << 24);
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

HfConfig MakeConfig() {
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
    // #810: an EMPTY `layer_types` is what a hybrid that does not speak
    // Qwen3.5's config dialect ships (NemotronH), and indexing it would be out
    // of bounds. Absent => not linear attention, which is the same polarity the
    // runner's own fallback predicate uses.
    lw.is_linear_attention =
        !c.layer_types.empty() &&
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

constexpr int kBlockSize = 16;
constexpr int kMaxModelLen = 32;
constexpr int kNumBlocks = 8;

// A fake KVCacheConfig with the gate group structure: one full-attn group + one
// mamba (GDN) group, sharing kNumBlocks blocks.
//
// #810: `conv_shape`/`ssm_shape` override the config-derived recurrent shapes.
// EMPTY (the default) reproduces the historical helper byte for byte. They
// exist for the same reason the dtype parameters already did: with both sides
// derived from one `HfConfig`, `spec->shapes == expected_*_shape` is two
// derivations of ONE config agreeing with each other, which cannot fail. A
// caller that wants to gate "the SPEC is the allocation source of truth" has to
// pass shapes the config cannot produce.
KVCacheConfig MakeKvConfig(const HfConfig& c,
                           DType conv_dtype = DType::kF32,
                           DType ssm_dtype = DType::kF32,
                           std::vector<int64_t> conv_shape = {},
                           std::vector<int64_t> ssm_shape = {}) {
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);
  const int Hv = static_cast<int>(c.linear_num_value_heads);
  const int Dv = static_cast<int>(c.linear_value_head_dim);
  const int Dk = static_cast<int>(c.linear_key_head_dim);
  const int Kw = static_cast<int>(c.linear_conv_kernel_dim);
  const int key_dim = static_cast<int>(c.linear_num_key_heads) * Dk;
  const int value_dim = Hv * Dv;
  const int conv_dim = 2 * key_dim + value_dim;

  if (conv_shape.empty()) conv_shape = {conv_dim, Kw - 1};
  if (ssm_shape.empty()) ssm_shape = {Hv, Dv, Dk};

  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa3"},
      std::make_shared<FullAttentionSpec>(kBlockSize, Hkv, Dh,
                                          vllm::v1::ResolveKvCacheDType()));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn0", "gdn1", "gdn2"},
      std::make_shared<MambaSpec>(
          kMaxModelLen,
          std::vector<std::vector<int64_t>>{std::move(conv_shape),
                                            std::move(ssm_shape)},
          std::vector<DType>{conv_dtype, ssm_dtype}));
  return kv;
}

vt::Queue Q() { return vt::Queue{vt::Device{vt::DeviceType::kCPU, 0}, nullptr}; }

SamplingParams Greedy() {
  SamplingParams sp;
  sp.temperature = 0.0;  // greedy (argmax).
  sp.PostInit();
  return sp;
}

// A NewRequestData for the gate group structure (block_ids = {full-attn, gdn}).
NewRequestData MakeNewReq(const std::string& id, std::vector<int32_t> prompt,
                          std::vector<int32_t> output, int num_computed,
                          std::vector<int> fa_blocks, int gdn_block,
                          const SamplingParams& sp) {
  NewRequestData nr;
  nr.req_id = id;
  std::vector<int32_t> all = prompt;
  all.insert(all.end(), output.begin(), output.end());
  nr.prompt_token_ids = std::move(prompt);
  nr.sampling_params = sp;
  nr.block_ids = {std::move(fa_blocks), std::vector<int>{gdn_block}};
  nr.num_computed_tokens = num_computed;
  nr.prefill_token_ids = std::move(all);
  return nr;
}

SchedulerOutput NewStep(std::vector<NewRequestData> new_reqs,
                        std::map<std::string, int> scheduled) {
  SchedulerOutput so;
  so.scheduled_cached_reqs = CachedRequestData::make_empty();
  so.scheduled_new_reqs = std::move(new_reqs);
  int total = 0;
  for (const auto& [id, n] : scheduled) total += n;
  so.num_scheduled_tokens = std::move(scheduled);
  so.total_num_scheduled_tokens = total;
  return so;
}

// A decode step for a single already-admitted request.
SchedulerOutput DecodeStep(const std::string& id, int num_computed,
                           int num_output) {
  SchedulerOutput so;
  CachedRequestData cached;
  cached.req_ids = {id};
  cached.num_computed_tokens = {num_computed};
  cached.num_output_tokens = {num_output};
  cached.new_block_ids.emplace_back(std::nullopt);  // no new blocks this step
  so.scheduled_cached_reqs = std::move(cached);
  so.num_scheduled_tokens = {{id, 1}};
  so.total_num_scheduled_tokens = 1;
  return so;
}

int GreedyArgmax(const std::vector<float>& logits, int64_t row, int64_t vocab) {
  int best = 0;
  float bv = logits[static_cast<size_t>(row * vocab)];
  for (int64_t v = 1; v < vocab; ++v) {
    const float x = logits[static_cast<size_t>(row * vocab + v)];
    if (x > bv) {
      bv = x;
      best = static_cast<int>(v);
    }
  }
  return best;
}

// ─── W1 runner-generalization fixtures: a FULL-ATTENTION-ONLY (non-hybrid)
// model. layer_types is EMPTY (pure dense, e.g. Qwen3ForCausalLM) and the KV
// config carries exactly ONE full-attention group with NO MambaSpec/GDN group.
// Pre-generalization, runner.cpp indexed config_.layer_types[l] (out of bounds
// on empty) and unconditionally gather_block_table(gdn_group_id_ == -1) →
// input_batch_.block_table[-1] (out of bounds). Both must now be skipped.

HfConfig MakeDenseOnlyConfig() {
  HfConfig c;
  c.model_type = "qwen3";
  c.architectures = {"Qwen3ForCausalLM"};
  c.hidden_size = 32;
  c.num_hidden_layers = 2;  // both FULL-ATTENTION
  c.vocab_size = 40;
  c.num_attention_heads = 6;
  c.num_key_value_heads = 2;
  c.head_dim = 8;
  c.layer_types = {};  // EMPTY → pure dense (all full-attention, no GDN)
  c.intermediate_size = 16;
  c.num_experts = 0;
  // GDN-shape fields are unused by a full-attention-only model, but stay valid
  // so the runner's (now guarded) conv_dim arithmetic is well-formed.
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

Qwen3_5DenseWeights MakeDenseOnlyWeights(const HfConfig& c) {
  Qwen3_5DenseWeights w;
  const int64_t H = c.hidden_size, V = c.vocab_size;
  const int64_t Hq = c.num_attention_heads, Hkv = c.num_key_value_heads,
                Dh = c.head_dim, I = c.intermediate_size;
  w.embed_tokens = MakeOwned(DType::kBF16, {V, H}, 11);
  w.final_norm = MakeOwned(DType::kBF16, {H}, 12);
  w.lm_head = MakeOwned(DType::kBF16, {H, V}, 13);
  for (int64_t l = 0; l < c.num_hidden_layers; ++l) {
    const uint64_t s = 1000 + static_cast<uint64_t>(l) * 5000;
    vllm::Qwen3_5DenseLayerWeights lw;
    lw.is_linear_attention = false;  // every layer full-attention
    lw.input_layernorm = MakeOwned(DType::kBF16, {H}, s + 1);
    lw.post_attention_layernorm = MakeOwned(DType::kBF16, {H}, s + 2);
    lw.attn.q_proj = MakeOwned(DType::kBF16, {H, 2 * Hq * Dh}, s + 10);
    lw.attn.k_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 20);
    lw.attn.v_proj = MakeOwned(DType::kBF16, {H, Hkv * Dh}, s + 30);
    lw.attn.o_proj = MakeOwned(DType::kBF16, {Hq * Dh, H}, s + 40);
    lw.attn.q_norm = MakeOwned(DType::kBF16, {Dh}, s + 50);
    lw.attn.k_norm = MakeOwned(DType::kBF16, {Dh}, s + 60);
    lw.mlp.gate_proj = MakeOwned(DType::kBF16, {H, I}, s + 100);
    lw.mlp.up_proj = MakeOwned(DType::kBF16, {H, I}, s + 200);
    lw.mlp.down_proj = MakeOwned(DType::kBF16, {I, H}, s + 300);
    w.layers.push_back(std::move(lw));
  }
  return w;
}

// A full-attention-ONLY KVCacheConfig: one FA group, NO mamba group (mirrors
// MakeQwen3ForCausalLMKVCache).
KVCacheConfig MakeFaOnlyKvConfig(const HfConfig& c) {
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);
  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa"},
      std::make_shared<FullAttentionSpec>(kBlockSize, Hkv, Dh,
                                          vllm::v1::ResolveKvCacheDType()));
  return kv;
}

// A NewRequestData with a SINGLE (full-attention) block-table group.
NewRequestData MakeFaNewReq(const std::string& id, std::vector<int32_t> prompt,
                            int num_computed, std::vector<int> fa_blocks,
                            const SamplingParams& sp) {
  NewRequestData nr;
  nr.req_id = id;
  nr.prompt_token_ids = prompt;
  nr.sampling_params = sp;
  nr.block_ids = {std::move(fa_blocks)};  // ONE group only (no GDN group)
  nr.num_computed_tokens = num_computed;
  nr.prefill_token_ids = std::move(prompt);
  return nr;
}

// ─── KV-DSV4-MULTICACHE W3 (#2068) — a DeepSeek-V4-SHAPED multi-cache topology ─
//
// The miniature is DeepSeek-V4-Flash's geometry at four layers instead of 43,
// with `compress_ratios` [0, 0, 4, 128] -- i.e. one SWA-only layer pair, one C4A
// layer and one C128A layer. Every spec below is constructed with the arguments
// `MakeDeepseekV4KVCache` passes at the same site, and every page-size literal
// asserted against it is the one W1's `### W1 design` table derives from
// upstream (`vllm/models/deepseek_v4/attention.py:631-645`, `:669-684`,
// `vllm/v1/attention/backends/mla/sparse_swa.py:86-101`,
// `vllm/models/deepseek_v4/compressor.py:188-200` at the pin
// `5559679229bc961848b121ccdeaa8fa5d79bec98`).
//
// TEN caches over FOUR layers: layer 0 and 1 carry the SWA cache alone (upstream
// returns `None` from `get_kv_cache_spec` when `compress_ratio <= 1`), layer 2
// carries five (latent + SWA + indexer key + two compressor states) and layer 3
// three. A runner that allocates one buffer per hidden layer produces FOUR.
constexpr int kV4BlockSize = 256;

std::shared_ptr<vllm::v1::MLAAttentionSpec> V4Latent(int ratio) {
  return std::make_shared<vllm::v1::MLAAttentionSpec>(
      kV4BlockSize, /*head_size=*/512, DType::kI8, /*num_kv_heads=*/1,
      vllm::v1::KVQuantMode::kFp8PerTensor, /*page_size_padded=*/std::nullopt,
      /*indexes_kv_by_block_stride=*/false,
      std::optional<std::string>("fp8_ds_mla"), /*alignment=*/576, ratio,
      std::optional<std::string>("deepseek_v4"));
}

std::shared_ptr<vllm::v1::SlidingWindowMLASpec> V4Sliding(int block_size,
                                                          int head_size,
                                                          DType dtype,
                                                          int window,
                                                          bool ds_mla_layout) {
  return std::make_shared<vllm::v1::SlidingWindowMLASpec>(
      block_size, /*num_kv_heads=*/1, head_size, dtype, window,
      ds_mla_layout ? std::optional<std::string>("fp8_ds_mla") : std::nullopt,
      /*alignment=*/576, /*compress_ratio=*/1,
      ds_mla_layout ? std::optional<std::string>("deepseek_v4") : std::nullopt,
      ds_mla_layout ? vllm::v1::KVQuantMode::kFp8PerTensor
                    : vllm::v1::KVQuantMode::kNone);
}

// The seven groups, in `MakeDeepseekV4KVCache`'s own publication order.
KVCacheConfig MakeMultiCacheKvConfig() {
  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.2.attn"}, V4Latent(4));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.3.attn"}, V4Latent(128));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.2.attn.indexer.k_cache"},
      std::make_shared<vllm::v1::MLAAttentionSpec>(
          kV4BlockSize, /*head_size=*/132, DType::kI8, /*num_kv_heads=*/1,
          vllm::v1::KVQuantMode::kNone, /*page_size_padded=*/std::nullopt,
          /*indexes_kv_by_block_stride=*/false,
          /*cache_dtype_str=*/std::nullopt, /*alignment=*/576,
          /*compress_ratio=*/4, /*model_version=*/std::nullopt));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.0.attn.swa_cache",
                               "model.layers.1.attn.swa_cache",
                               "model.layers.2.attn.swa_cache",
                               "model.layers.3.attn.swa_cache"},
      V4Sliding(/*block_size=*/64, /*head_size=*/512, DType::kI8,
                /*window=*/128, /*ds_mla_layout=*/true));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.2.attn.compressor.state_cache"},
      V4Sliding(/*block_size=*/4, /*head_size=*/2048, DType::kF32,
                /*window=*/8, /*ds_mla_layout=*/false));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{
          "model.layers.2.attn.indexer.compressor.state_cache"},
      V4Sliding(/*block_size=*/4, /*head_size=*/512, DType::kF32,
                /*window=*/8, /*ds_mla_layout=*/false));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.3.attn.compressor.state_cache"},
      V4Sliding(/*block_size=*/8, /*head_size=*/1024, DType::kF32,
                /*window=*/128, /*ds_mla_layout=*/false));
  return kv;
}

}  // namespace

// ─── 0. The attention cache is sized from the KV SPEC, not the HF config ─────
//
// MLA campaign W1. Upstream sizes every KV buffer from
// `spec.page_size_bytes()` (vllm/v1/kv_cache_interface.py:380-398) and shapes
// it from the backend, which is why `vllm/v1/worker/gpu_model_runner.py` needs
// no `use_mla` branch at all. We used to compute
// `num_blocks * 2 * block * config.num_key_value_heads * config.head_dim`.
// These two cases are the POSITIVE SIGNAL that the spec now drives it:
//   (a) the default spec reproduces the old bytes EXACTLY (byte-identity), and
//   (b) a `page_size_padded` spec — a value the old HF-config arithmetic could
//       not produce under ANY config — is honoured, which is only possible if
//       the allocator actually asked the spec.
TEST_CASE("runner: attention cache page size comes from the KV spec") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const int64_t Hkv = c.num_key_value_heads;
  const int64_t Dh = c.head_dim;
  const DType kv_dtype = vllm::v1::ResolveKvCacheDType();

  SUBCASE("default spec == the pre-refactor hardcoded arithmetic") {
    GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                          kMaxModelLen, /*max_num_batched_tokens=*/64);
    // The exact expression the runner used to hardcode (factor 2 = K+V).
    const int64_t legacy_page_bytes =
        2 * kBlockSize * Hkv * Dh * static_cast<int64_t>(vt::SizeOf(kv_dtype));
    CHECK(runner.fa_page_size_bytes() == legacy_page_bytes);
    CHECK(runner.attn_kv()[0].dtype == kv_dtype);
  }

  SUBCASE("page_size_padded from the spec is honoured (proves the spec ran)") {
    KVCacheConfig kv = MakeFaOnlyKvConfig(c);
    const int64_t real =
        2 * kBlockSize * Hkv * Dh * static_cast<int64_t>(vt::SizeOf(kv_dtype));
    const int64_t padded = real + 512;  // unreachable by any HF-config formula
    kv.kv_cache_groups[0].kv_cache_spec = std::make_shared<FullAttentionSpec>(
        kBlockSize, static_cast<int>(Hkv), static_cast<int>(Dh), kv_dtype,
        /*head_size_v=*/std::nullopt, vllm::v1::KVQuantMode::kNone, padded);
    GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                          /*max_num_batched_tokens=*/64);
    CHECK(runner.fa_page_size_bytes() == padded);
    CHECK(runner.fa_page_size_bytes() != real);
  }
}

// ─── 1. KV allocation shape from a fake KVCacheConfig ────────────────────────
TEST_CASE("runner: KV allocation from KVCacheConfig (full-attn + GDN state)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  CHECK(runner.full_attn_group_id() == 0);
  CHECK(runner.gdn_group_id() == 1);
  CHECK(runner.num_blocks() == kNumBlocks);
  CHECK_FALSE(runner.kv_cache_backend_resident());

  // M3: the runner resolves the ENGINE-level attention backend at init, per
  // attention group. On CPU the dense priority walk (cpu.cpp) is
  // [CPU_ATTN, FLASH_ATTN] and, since #1371 registered it, lands on CPU_ATTN —
  // upstream's own CPU answer (cpu.py:75-87). This is the proof that selection
  // is part of the runtime path, not just the registry test: the name below is
  // resolved inside GPUModelRunner::initialize_kv_cache. One name per attention
  // layer.
  REQUIRE(runner.attn_backend_names().size() == 1);
  CHECK(runner.attn_backend_names()[0] == "CPU_ATTN");

  // One PagedKvCache per full-attn layer (config has exactly 1).
  REQUIRE(runner.attn_kv().size() == 1);
  const PagedKvCache& kv = runner.attn_kv()[0];
  CHECK(kv.num_blocks == kNumBlocks);
  CHECK(kv.block_size == kBlockSize);
  CHECK(kv.num_kv_heads == c.num_key_value_heads);
  CHECK(kv.head_size == c.head_dim);
  CHECK(kv.data != nullptr);

  // One GdnStateCache per GDN layer (config has exactly 3).
  REQUIRE(runner.gdn_state().size() == 3);
  const GdnStateCache& gs = runner.gdn_state()[0];
  CHECK(gs.ssm_state.dtype == DType::kF32);
  CHECK(gs.conv_state.dtype == DType::kF32);
  // ssm_state [num_blocks, Hv, Dv, Dk].
  CHECK(gs.ssm_state.shape[0] == kNumBlocks);
  CHECK(gs.ssm_state.shape[1] == c.linear_num_value_heads);
  CHECK(gs.ssm_state.shape[2] == c.linear_value_head_dim);
  CHECK(gs.ssm_state.shape[3] == c.linear_key_head_dim);
  // conv_state [num_blocks, conv_dim, K-1].
  const int64_t key_dim = c.linear_num_key_heads * c.linear_key_head_dim;
  const int64_t conv_dim =
      2 * key_dim + c.linear_num_value_heads * c.linear_value_head_dim;
  CHECK(gs.conv_state.shape[0] == kNumBlocks);
  CHECK(gs.conv_state.shape[1] == conv_dim);
  CHECK(gs.conv_state.shape[2] == c.linear_conv_kernel_dim - 1);
}

TEST_CASE("runner: recurrent prefix restores exact SSM and convolution bytes after slot poisoning") {
  const auto config = MakeConfig();
  const auto weights = MakeWeights(config);
  auto kv = MakeKvConfig(config);
  kv.recurrent_prefix_snapshots =
      std::make_shared<vllm::v1::RecurrentPrefixSnapshotIndex>(4, kBlockSize);
  const auto index = kv.recurrent_prefix_snapshots;
  GPUModelRunner runner(config, weights, kv, Q(), 8, kMaxModelLen, 64);
  std::vector<int32_t> prompt(kBlockSize + 1);
  for (size_t i = 0; i < prompt.size(); ++i) prompt[i] = int(i % 31) + 1;
  auto cold = MakeNewReq("cold", prompt, {}, 0, {0, 1}, 0, Greedy());
  cold.prompt_block_hashes = {"exact-prefix"};
  CHECK_FALSE(runner.execute_model(NewStep({cold}, {{"cold", kBlockSize}})).has_value());
  CHECK(runner.sample_tokens(std::nullopt).sampled_token_ids[0].empty());
  REQUIRE(index->Contains("exact-prefix"));
  REQUIRE(index->Pin("probe", "exact-prefix"));
  const int snapshot_slot = int(runner.gdn_state_slots()) - index->capacity() +
                            index->Pinned("probe")->slot;
  index->Release("probe");
  const auto state_bytes = [&](int slot) {
    std::vector<std::vector<unsigned char>> result;
    for (const auto& layer : runner.gdn_state()) {
      for (const auto& tensor : layer.states) {
        const size_t bytes = size_t(tensor.stride[0]) * vt::SizeOf(tensor.dtype);
        const auto* data = static_cast<const unsigned char*>(tensor.data) + slot * bytes;
        result.emplace_back(data, data + bytes);
      }
    }
    return result;
  };
  const auto prefix_state = state_bytes(0);
  REQUIRE(prefix_state.size() == 6);  // SSM + convolution, all three GDN layers.
  CHECK(state_bytes(snapshot_slot) == prefix_state);
  const auto* page = static_cast<const unsigned char*>(runner.attn_kv()[0].data);
  const std::vector<unsigned char> prefix_kv(page, page + runner.fa_page_size_bytes());
  CHECK_FALSE(runner.execute_model(DecodeStep("cold", kBlockSize, 0)).has_value());
  const auto cold_output = runner.sample_tokens(std::nullopt).sampled_token_ids;
  const auto final_state = state_bytes(0);
  CHECK(final_state != prefix_state);
  CHECK(state_bytes(snapshot_slot) == prefix_state);

  // A new owner must restore from the immutable row, not reuse old live state.
  for (const auto& layer : runner.gdn_state())
    for (const auto& tensor : layer.states)
      std::memset(tensor.data, 0x5a, size_t(tensor.stride[0]) * vt::SizeOf(tensor.dtype));
  REQUIRE(index->Pin("warm", "exact-prefix"));
  auto warm = MakeNewReq("warm", prompt, {}, kBlockSize, {0, 1}, 0, Greedy());
  warm.prompt_block_hashes = {"exact-prefix"};
  auto step = NewStep({warm}, {{"warm", 1}});
  step.finished_req_ids.insert("cold");
  CHECK_FALSE(runner.execute_model(step).has_value());
  CHECK_FALSE(index->Pinned("warm").has_value());
  CHECK(runner.sample_tokens(std::nullopt).sampled_token_ids == cold_output);
  CHECK(state_bytes(0) == final_state);
  CHECK(state_bytes(snapshot_slot) == prefix_state);
  CHECK(std::vector<unsigned char>(page, page + runner.fa_page_size_bytes()) == prefix_kv);
}

// SPEC-MTP I5d-pre LATENT-BUG FIX: with a THIRD `fa_draft` full-attention group
// (as MakeQwen3_5KVCacheSpec appends when num_spec>0), the runner must still
// select the TARGET full-attn group (index 0), NOT the last full-attn group.
// RED-first: the pre-fix loop kept the last kFullAttention group, so this would
// report 2 (the draft) instead of 0.
TEST_CASE("runner: full-attn group selection ignores a third fa_draft group") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);

  // fa(0), gdn(1), fa_draft(2) — the draft KV layer is a second full-attn group
  // appended after the target's, exactly as the num_spec>0 spec does.
  SUBCASE("draft group unmarked (first-wins selection)") {
    KVCacheConfig kv = MakeKvConfig(c);
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"fa_draft"},
        std::make_shared<FullAttentionSpec>(kBlockSize, Hkv, Dh,
                                            vllm::v1::ResolveKvCacheDType()));
    REQUIRE(kv.kv_cache_groups.size() == 3);
    GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                          /*max_num_batched_tokens=*/64);
    CHECK(runner.full_attn_group_id() == 0);  // target, not the fa_draft at 2.
    CHECK(runner.gdn_group_id() == 1);
  }

  // Even if a future change marks the draft group as an eagle group, the
  // by-role skip keeps the target selected.
  SUBCASE("draft group marked eagle (by-role skip)") {
    KVCacheConfig kv = MakeKvConfig(c);
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"fa_draft"},
        std::make_shared<FullAttentionSpec>(kBlockSize, Hkv, Dh,
                                            vllm::v1::ResolveKvCacheDType()),
        /*is_eagle_group=*/true);
    GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                          /*max_num_batched_tokens=*/64);
    CHECK(runner.full_attn_group_id() == 0);
    CHECK(runner.gdn_group_id() == 1);
  }
}

// #810: THE ARMED HALVES. Both subcases run the SAME assertions against the
// same function; they differ only in whether the MambaSpec they feed it is
// derivable from the HfConfig.
//
// The dtype half of this case has always been armed — `MakeKvConfig(c, kBF16,
// kF16)` passes dtypes the config cannot produce and the runner honours them.
// The SHAPE half was INERT for exactly the reason the shape-override
// parameters were added: `MakeConfig()` and `MakeKvConfig(c)` derived both
// sides from one config, so `shapes == expected_*_shape` compared two
// derivations of one number
// ([[gate-comparing-shared-helper-proves-consistency-not-correctness]]).
// The second subcase feeds shapes the config CANNOT produce — the real
// NemotronH `{6144, 3}` / `{64, 64, 128}` — which is what makes the shape
// assertions falsifiable. Before #810 it refused at
// `runner.cpp` "Qwen3.5 MambaSpec shapes disagree with model config".
TEST_CASE("runner: MambaSpec is the allocation source of truth") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);

  KVCacheConfig kv;
  int max_num_reqs = 8;
  SUBCASE("shapes the config can also derive (dtype half armed)") {
    kv = MakeKvConfig(c, DType::kBF16, DType::kF16);
  }
  SUBCASE("shapes the config CANNOT produce (shape half armed)") {
    // Real NemotronH geometry (nemotron_h_registry.cpp:204-215), which
    // MakeConfig()'s linear_* fields cannot yield under any arithmetic.
    kv = MakeKvConfig(c, DType::kBF16, DType::kF16,
                      /*conv_shape=*/{6144, 3}, /*ssm_shape=*/{64, 64, 128});
    max_num_reqs = 2;  // keep the f32 SSM allocation modest in CI
  }

  const auto* spec = dynamic_cast<const MambaSpec*>(
      kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(spec != nullptr);

  GPUModelRunner runner(c, w, kv, Q(), max_num_reqs, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  REQUIRE(runner.gdn_state().size() == 3);
  const GdnStateCache& state = runner.gdn_state()[0];

  REQUIRE(spec->shapes.size() == 2);
  REQUIRE(spec->dtypes.size() == 2);
  CHECK(state.conv_state.dtype == spec->dtypes[0]);
  CHECK(state.ssm_state.dtype == spec->dtypes[1]);
  CHECK(std::vector<int64_t>{state.conv_state.shape[1],
                             state.conv_state.shape[2]} == spec->shapes[0]);
  CHECK(std::vector<int64_t>{state.ssm_state.shape[1], state.ssm_state.shape[2],
                             state.ssm_state.shape[3]} == spec->shapes[1]);

  const int64_t runtime_row_bytes =
      state.conv_state.shape[1] * state.conv_state.shape[2] *
          static_cast<int64_t>(vt::SizeOf(state.conv_state.dtype)) +
      state.ssm_state.shape[1] * state.ssm_state.shape[2] *
          state.ssm_state.shape[3] *
          static_cast<int64_t>(vt::SizeOf(state.ssm_state.dtype));
  CHECK(runtime_row_bytes == spec->page_size_bytes());
  // The slot dim is the recurrent pool, never the attention num_blocks.
  CHECK(state.conv_state.shape[0] == runner.gdn_state_slots());
  CHECK(state.ssm_state.shape[0] == runner.gdn_state_slots());
}

// ─── #810: THE BYTE-NEUTRALITY ARM ───────────────────────────────────────────
//
// `initialize_kv_cache` serves EVERY architecture — the engine builds exactly
// one `GPUModelRunner` (`src/vllm/entrypoints/model_loader.cpp::runner_`, built
// in the `LoadedEngine` member-init list) — so a refactor of it owes
// a proof that it changes nothing for the models that already work. This
// mirrors the BYTE-NEUTRALITY CONTRACT stated for the `per_layer_attn_specs`
// seam at include/vllm/v1/kv_cache_interface.h:354-374: byte-identical
// allocation, view, indexing and group selection to before the recurrent half
// read its geometry off the spec.
//
// The expected values are LITERALS, derived at the pre-refactor base SHA. They
// are deliberately not computed from `c` inside the test: a helper sharing the
// inputs of the code under test reproduces exactly the self-consistency defect
// the case above exists to remove. Only the KV element SIZE follows
// `ResolveKvCacheDType()`, because the VT_KV_CACHE_F32 A/B lane legitimately
// changes it and the geometry — K+V, block 16, 2 kv heads, head_dim 8 — is the
// part being pinned.
//
// It reports its own N ([[the-state-was-not-the-one-you-believed]]): every
// layer's class as a full literal vector, every state shape and dtype per
// layer, and the total allocated bytes.
TEST_CASE("runner: the Qwen3.5 allocation is BYTE-IDENTICAL after #810") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  REQUIRE(c.num_hidden_layers == 4);  // [LA, LA, LA, FA]

  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  using LayerKvClass = GPUModelRunner::LayerKvClass;
  // 1. Counts, and the model's own N.
  CHECK(runner.layer_kv_class().size() == 4);
  CHECK(runner.attn_kv().size() == 1);
  CHECK(runner.gdn_state().size() == 3);
  CHECK(runner.gdn_state_slots() == 8);  // max_num_reqs, spec off
  CHECK(runner.num_blocks() == kNumBlocks);

  // 2. The layer -> group assignment for EVERY layer index, as a literal
  //    vector. Spot-checking layer 0 cannot see a routing inversion.
  CHECK(runner.layer_kv_class() ==
        std::vector<LayerKvClass>{
            LayerKvClass::kRecurrent, LayerKvClass::kRecurrent,
            LayerKvClass::kRecurrent, LayerKvClass::kFullAttention});

  // 3. Group selection — the `fa_draft` case above must stay green too.
  CHECK(runner.full_attn_group_id() == 0);
  CHECK(runner.gdn_group_id() == 1);

  // 4. Every attention view, per layer.
  const int64_t kv_es =
      static_cast<int64_t>(vt::SizeOf(vllm::v1::ResolveKvCacheDType()));
  const int64_t kFaPageBytes = 2 * 16 * 2 * 8 * kv_es;  // K+V, block, Hkv, Dh
  CHECK(runner.fa_page_size_bytes() == kFaPageBytes);
  for (const PagedKvCache& kv : runner.attn_kv()) {
    CHECK(kv.num_blocks == 8);
    CHECK(kv.block_size == 16);
    CHECK(kv.num_kv_heads == 2);
    CHECK(kv.head_size == 8);
    CHECK(kv.dtype == vllm::v1::ResolveKvCacheDType());
    CHECK(kv.data != nullptr);
  }

  // 5. Every recurrent state shape and dtype, per layer. conv_dim is
  //    2*(2*8) + 4*8 == 64 and the conv row is conv_kernel-1 == 3.
  for (const GdnStateCache& gs : runner.gdn_state()) {
    CHECK(gs.conv_state.dtype == DType::kF32);
    CHECK(gs.ssm_state.dtype == DType::kF32);
    CHECK(std::vector<int64_t>{gs.conv_state.shape[0], gs.conv_state.shape[1],
                               gs.conv_state.shape[2]} ==
          std::vector<int64_t>{8, 64, 3});
    CHECK(std::vector<int64_t>{gs.ssm_state.shape[0], gs.ssm_state.shape[1],
                               gs.ssm_state.shape[2], gs.ssm_state.shape[3]} ==
          std::vector<int64_t>{8, 4, 8, 8});
    CHECK(gs.conv_state.data != nullptr);
    CHECK(gs.ssm_state.data != nullptr);
  }

  // 6. The byte-neutrality claim itself: the whole allocation, as one number.
  //      attention 1 layer  x 8 blocks x kFaPageBytes
  //      recurrent 3 layers x (8 slots*64*3*4  +  8 slots*4*8*8*4)
  int64_t total_bytes = 0;
  for (const PagedKvCache& kv : runner.attn_kv())
    total_bytes += kv.num_blocks * runner.fa_page_size_bytes();
  for (const GdnStateCache& gs : runner.gdn_state())
    total_bytes += static_cast<int64_t>(gs.conv_state.Bytes()) +
                   static_cast<int64_t>(gs.ssm_state.Bytes());
  CHECK(total_bytes == 1 * 8 * kFaPageBytes + 3 * (6144 + 8192));
}

// ─── ENG-RECURRENT-MULTISTATE (#2131): N RECURRENT STATES, NOT TWO ───────────
//
// `initialize_kv_cache` refused any `MambaSpec` that did not carry EXACTLY two
// shapes and two dtypes, and `GdnStateCache` carried exactly two named tensors.
// Upstream has no such assumption anywhere: `MambaBase.kv_cache` is
// `tuple[torch.Tensor, ...]` (`vllm/model_executor/layers/mamba/abstract.py:26`)
// and `bind_kv_cache` (`:29-43`) unpacks ONE page into as many states as
// `zip(get_state_shape(), get_state_dtype())` yields, each with its own shape
// and its own dtype. Three values of N ship at the pin `5559679229`: 1
// (`short_conv.py:87`), 2, and 5 (`mamba_mixer2.py:517-520`, whose appended ring
// states are rank 3 / rank 2 / rank 3 with a `torch.float32` between two
// activation dtypes, `mamba_utils.py:84-93` and `:202-221`).
//
// THE FIXTURE IS CHOSEN SO THE THIRD STATE CHANGES THE ANSWER. It is a
// different RANK (1-D against 2-D and 3-D), a different ELEMENT COUNT, and a
// different DTYPE (kI64 — a token-id history is integers, not activations) from
// either of the first two. A third state that merely repeated the conv shape
// would be counted correctly by an implementation that multiplied by 2, and its
// dtype would be counted correctly by one that reused `dtypes[0]`.
namespace {
// The two-state gate geometry plus a third state, over the SAME group. Sizes:
//   conv {64, 3}      f32  ->  768 B/slot
//   ssm  {4, 8, 8}    f32  -> 1024 B/slot
//   hist {7}          i64  ->   56 B/slot
// Three distinct byte counts, so a wrong per-state size cannot cancel.
constexpr int64_t kMsConvElems = 64 * 3;
constexpr int64_t kMsSsmElems = 4 * 8 * 8;
constexpr int64_t kMsHistElems = 7;

KVCacheConfig MakeMultiStateKvConfig(
    const HfConfig& c, std::vector<std::vector<int64_t>> shapes,
    std::vector<DType> dtypes) {
  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"fa3"},
      std::make_shared<FullAttentionSpec>(
          kBlockSize, static_cast<int>(c.num_key_value_heads),
          static_cast<int>(c.head_dim), vllm::v1::ResolveKvCacheDType()));
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"gdn0", "gdn1", "gdn2"},
      std::make_shared<MambaSpec>(kMaxModelLen, std::move(shapes),
                                  std::move(dtypes)));
  return kv;
}

KVCacheConfig MakeThreeStateKvConfig(const HfConfig& c) {
  return MakeMultiStateKvConfig(
      c, {{64, 3}, {4, 8, 8}, {kMsHistElems}},
      {DType::kF32, DType::kF32, DType::kI64});
}
}  // namespace

TEST_CASE("runner: a recurrent group carries N states, not two") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const KVCacheConfig kv = MakeThreeStateKvConfig(c);
  const auto* spec =
      dynamic_cast<const MambaSpec*>(kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(spec != nullptr);
  REQUIRE(spec->shapes.size() == 3);

  GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  const int64_t slots = runner.gdn_state_slots();
  REQUIRE(slots == 8);
  REQUIRE(runner.gdn_state().size() == 3);  // three GDN layers

  // 1. Every layer carries the group's OWN state count, in SPEC ORDER, and the
  //    ordered list is the mirror of `MambaBase.kv_cache`.
  for (const GdnStateCache& gs : runner.gdn_state()) {
    REQUIRE(gs.states.size() == 3);
    // The legacy names are the first two entries, unchanged, which is what
    // every model consumer in this tree reads.
    CHECK(gs.states[0].data == gs.conv_state.data);
    CHECK(gs.states[1].data == gs.ssm_state.data);
    // 2. Each state carries its OWN rank, shape and dtype off the spec, with
    //    the slot dim prepended (`bind_kv_cache`'s `state.view(-1, *shape)`).
    CHECK(gs.states[0].dtype == DType::kF32);
    CHECK(gs.states[1].dtype == DType::kF32);
    CHECK(gs.states[2].dtype == DType::kI64);
    CHECK(gs.states[0].rank == 3);
    CHECK(gs.states[1].rank == 4);
    CHECK(gs.states[2].rank == 2);
    CHECK(std::vector<int64_t>{gs.states[2].shape[0], gs.states[2].shape[1]} ==
          std::vector<int64_t>{slots, kMsHistElems});
    // 3. The third state is a DISTINCT allocation, not an alias of either
    //    other one and not a re-view of the same bytes.
    CHECK(gs.states[2].data != nullptr);
    CHECK(gs.states[2].data != gs.states[0].data);
    CHECK(gs.states[2].data != gs.states[1].data);
    // 4. Its bytes are its OWN element count times its OWN element size — the
    //    number a "multiply the conv row by 2" implementation cannot produce.
    CHECK(static_cast<int64_t>(gs.states[2].Bytes()) ==
          slots * kMsHistElems * 8);
  }

  // 5. The page-size identity holds over ALL THREE states, mirroring
  //    `MambaSpec.page_size_bytes` (`kv_cache_interface.py:698-707`).
  CHECK(spec->page_size_bytes() ==
        kMsConvElems * 4 + kMsSsmElems * 4 + kMsHistElems * 8);

  // 6. The runner's own byte report counts the third state. This is the
  //    accounting surface a short allocation would hide in
  //    (FIX-KV-GROUP-LAYER-COUNT, #1963).
  int64_t recurrent_bytes = 0;
  for (const GdnStateCache& gs : runner.gdn_state())
    for (const vt::Tensor& s : gs.states)
      recurrent_bytes += static_cast<int64_t>(s.Bytes());
  CHECK(recurrent_bytes == 3 * slots * spec->page_size_bytes());
  CHECK(runner.kv_cache_allocated_bytes() ==
        runner.kv_cache_allocated_paged_bytes() + recurrent_bytes);

  // 7. And the ENGINE-level budget the loader charges for this group agrees
  //    with what the runner took, over three states rather than two.
  CHECK(vllm::v1::recurrent_state_bytes(kv, /*max_num_seqs=*/8) ==
        recurrent_bytes);
}

// Each subcase asserts the refusal MESSAGE, and that is the whole point of the
// case. MEASURED: with a bare `CHECK_THROWS` this case is GREEN under the M1
// mutation that restores the old `shapes.size() == 2` refusal — every one of
// the three inputs still throws there, at the OLD message, for a reason that
// has nothing to do with what the subcase is named after. A case that cannot
// tell the widened refusal from the one it replaced gates nothing; the message
// is the only thing that separates them. See the mutation record in
// `.agents/specs/recurrent-multistate.md`.
TEST_CASE("runner: a malformed recurrent MambaSpec is REFUSED by name") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  // Returns the refusal text, or the empty string when nothing was thrown, so
  // a silent acceptance fails the substring check rather than escaping it.
  const auto refusal = [&](KVCacheConfig kv) {
    try {
      GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                            /*max_num_batched_tokens=*/64);
    } catch (const std::exception& e) {
      return std::string(e.what());
    }
    return std::string();
  };

  SUBCASE("shapes and dtypes of different length") {
    // Refused at the base tree too, by the CONJUNCTIVE `== 2` that stood here
    // and by `MambaSpec::page_size_bytes` — this row did not close an
    // out-of-bounds read, and it does not claim to. What it must not do is
    // stop refusing while it widens the count.
    const std::string msg = refusal(MakeMultiStateKvConfig(
        c, {{64, 3}, {4, 8, 8}, {kMsHistElems}},
        {DType::kF32, DType::kF32}));
    INFO("refusal: " << msg);
    CHECK(msg.find("with one dtype per shape") != std::string::npos);
  }
  SUBCASE("a single state (upstream ShortConv) is refused, not truncated") {
    const std::string msg =
        refusal(MakeMultiStateKvConfig(c, {{64, 3}}, {DType::kF32}));
    INFO("refusal: " << msg);
    CHECK(msg.find("must carry at least a conv and a temporal state") !=
          std::string::npos);
  }
  SUBCASE("a block-quantized state dtype is refused") {
    // `vt::SizeOf` has no per-element answer for a block encoding, so a page
    // sized from one would be arithmetic on a number that does not exist. This
    // is a THREE-state spec, so under the old `== 2` refusal it threw for the
    // count and never reached the dtype predicate at all.
    const std::string msg = refusal(MakeMultiStateKvConfig(
        c, {{64, 3}, {4, 8, 8}, {kMsHistElems}},
        {DType::kF32, DType::kF32, DType::kQ8_0}));
    INFO("refusal: " << msg);
    CHECK(msg.find("has no per-element size") != std::string::npos);
  }
}

// ─── #810: THE NEMOTRON-H ARM ────────────────────────────────────────────────
//
// The defect this row exists for, driven from a synthetic 52-layer
// NemotronH-shaped KVCacheConfig (no checkpoint, so it runs in CI). The
// topology is the real one — `layers_block_type` in
// tests/vllm/models/fixtures/nemotron_h_35_lightning/config.json: 6 attention
// blocks at {5, 12, 19, 26, 33, 42}, 23 Mamba2 blocks, 23 MoE blocks — and the
// shapes/dtypes are what `MakeNemotronHKVCache` publishes
// (nemotron_h_registry.cpp:204-215), asserted independently at
// tests/vllm/models/test_nemotron_h_scaffold.cpp:620-665.
//
// The config half is deliberately hostile and matches the real one: NO
// `layer_types`, and every `linear_*` field zero. That is precisely the state
// in which the pre-#810 runner classified all 52 layers as full attention.
namespace {
// The real fixture's `layers_block_type`, as index sets.
const std::vector<int64_t> kNemotronHAttnLayers{5, 12, 19, 26, 33, 42};
const std::vector<int64_t> kNemotronHMambaLayers{0,  2,  4,  7,  9,  11, 14, 16,
                                                 18, 21, 23, 25, 28, 30, 32, 35,
                                                 37, 39, 41, 44, 46, 48, 50};

HfConfig MakeNemotronHShapedConfig() {
  HfConfig c = MakeConfig();
  c.model_type = "nemotron_h";
  c.architectures = {"NemotronHForCausalLM"};
  c.num_hidden_layers = 52;
  c.layer_types.clear();          // NemotronH ships `layers_block_type`
  c.linear_num_key_heads = 0;     // ...and none of Qwen3.5's linear_* fields
  c.linear_num_value_heads = 0;
  c.linear_key_head_dim = 0;
  c.linear_value_head_dim = 0;
  c.linear_conv_kernel_dim = 0;
  return c;
}

std::string NemotronHLayerName(int64_t i) {
  return "backbone.layers." + std::to_string(i) + ".mixer";
}

KVCacheConfig MakeNemotronHShapedKvConfig() {
  std::vector<std::string> attn_names;
  for (int64_t i : kNemotronHAttnLayers) attn_names.push_back(NemotronHLayerName(i));
  std::vector<std::string> mamba_names;
  for (int64_t i : kNemotronHMambaLayers)
    mamba_names.push_back(NemotronHLayerName(i));

  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  kv.kv_cache_groups.emplace_back(
      std::move(attn_names),
      std::make_shared<FullAttentionSpec>(kBlockSize, /*num_kv_heads=*/2,
                                          /*head_size=*/128,
                                          vllm::v1::ResolveKvCacheDType()));
  kv.kv_cache_groups.emplace_back(
      std::move(mamba_names),
      std::make_shared<MambaSpec>(
          kBlockSize,
          std::vector<std::vector<int64_t>>{{6144, 3}, {64, 64, 128}},
          std::vector<DType>{DType::kBF16, DType::kF32}));
  return kv;
}
}  // namespace

TEST_CASE("runner: a NemotronH-shaped KV config allocates from the SPEC") {
  const HfConfig c = MakeNemotronHShapedConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const KVCacheConfig kv = MakeNemotronHShapedKvConfig();

  // max_num_reqs 1 keeps the 23 f32 SSM states at ~49 MiB of host memory.
  GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/1, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  using LayerKvClass = GPUModelRunner::LayerKvClass;
  // 1. The full index vector, not counts alone. Before #810 this was 52
  //    kFullAttention entries and zero recurrent buffers.
  std::vector<LayerKvClass> expected(52, LayerKvClass::kNone);
  for (int64_t i : kNemotronHAttnLayers)
    expected[static_cast<size_t>(i)] = LayerKvClass::kFullAttention;
  for (int64_t i : kNemotronHMambaLayers)
    expected[static_cast<size_t>(i)] = LayerKvClass::kRecurrent;
  REQUIRE(runner.layer_kv_class().size() == 52);
  CHECK(runner.layer_kv_class() == expected);

  // 2. SIX attention layers, not 52. The pre-#810 runner allocated 8.7x the
  //    pages this model needs, because every non-linear_attention layer was an
  //    attention layer and 23 of these blocks cache nothing at all.
  CHECK(runner.attn_kv().size() == 6);
  CHECK(runner.gdn_state().size() == 23);
  CHECK(runner.full_attn_group_id() == 0);
  CHECK(runner.gdn_group_id() == 1);

  // 3. Attention geometry off the FullAttentionSpec (head_size 128, which
  //    MakeNemotronHShapedConfig's head_dim of 8 cannot produce).
  const int64_t kv_es =
      static_cast<int64_t>(vt::SizeOf(vllm::v1::ResolveKvCacheDType()));
  CHECK(runner.fa_page_size_bytes() == 2 * kBlockSize * 2 * 128 * kv_es);
  for (const PagedKvCache& akv : runner.attn_kv()) {
    CHECK(akv.num_kv_heads == 2);
    CHECK(akv.head_size == 128);
    CHECK(akv.block_size == kBlockSize);
    CHECK(akv.data != nullptr);
  }

  // 4. Recurrent shapes and dtypes off the MambaSpec — the values no arithmetic
  //    over this config could yield, since every linear_* field is zero.
  for (const GdnStateCache& gs : runner.gdn_state()) {
    CHECK(gs.conv_state.dtype == DType::kBF16);
    CHECK(gs.ssm_state.dtype == DType::kF32);
    CHECK(std::vector<int64_t>{gs.conv_state.shape[0], gs.conv_state.shape[1],
                               gs.conv_state.shape[2]} ==
          std::vector<int64_t>{1, 6144, 3});
    CHECK(std::vector<int64_t>{gs.ssm_state.shape[0], gs.ssm_state.shape[1],
                               gs.ssm_state.shape[2], gs.ssm_state.shape[3]} ==
          std::vector<int64_t>{1, 64, 64, 128});
    CHECK(gs.conv_state.data != nullptr);
    CHECK(gs.ssm_state.data != nullptr);
  }

  // 5. Total bytes, as one number: 6 attention layers x 8 blocks x page, plus
  //    23 recurrent layers x (1 slot x 6144x3 bf16 + 1 slot x 64x64x128 f32).
  int64_t total_bytes = 0;
  for (const PagedKvCache& akv : runner.attn_kv())
    total_bytes += akv.num_blocks * runner.fa_page_size_bytes();
  for (const GdnStateCache& gs : runner.gdn_state())
    total_bytes += static_cast<int64_t>(gs.conv_state.Bytes()) +
                   static_cast<int64_t>(gs.ssm_state.Bytes());
  CHECK(total_bytes ==
        6 * 8 * (2 * kBlockSize * 2 * 128 * kv_es) + 23 * (36864 + 2097152));
}

// ─── 2. THE ORDERING IDENTITY GATE (mandatory de-risk) ───────────────────────
// A batch of {1 decode "D", 1 prefill "P"} admitted PREFILL-FIRST. After the
// decode-first reorder the four seams must agree on ONE order (slot 0 == D,
// slot 1 == P): logits_indices, the SamplingMetadata row (via P's seed), the
// attention seq_lens+block_table row, the GDN state index, and the write-back.
TEST_CASE("runner: four-way ordering identity (mixed decode+prefill)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);

  // P: fresh prefill, 5 tokens (seq_len 5). Random with a distinctive seed +
  // top_k so its SamplingMetadata row is identifiable.
  SamplingParams p_params;
  p_params.temperature = 0.7;
  p_params.top_k = 2;
  p_params.seed = 12345;
  p_params.PostInit();
  NewRequestData p = MakeNewReq("P", {1, 2, 3, 4, 5}, {}, /*num_computed=*/0,
                                /*fa_blocks=*/{2, 3}, /*gdn_block=*/1, p_params);

  // D: decode, prompt 3 + 1 already-produced output token, num_computed 3
  // (seq_len 4). Greedy (no seed).
  NewRequestData d = MakeNewReq("D", {6, 7, 8}, {9}, /*num_computed=*/3,
                                /*fa_blocks=*/{0, 1}, /*gdn_block=*/0, Greedy());

  // Admit PREFILL-FIRST so the reorder must move the decode to the front.
  SchedulerOutput so = NewStep({p, d}, {{"P", 5}, {"D", 1}});

  auto out_opt = runner.execute_model(so);
  CHECK_FALSE(out_opt.has_value());  // MRV2 split: forward done, no output yet.

  // (a) The reorder placed the decode "D" at slot 0, prefill "P" at slot 1.
  const auto& ib = runner.input_batch();
  REQUIRE(ib.num_reqs() == 2);
  REQUIRE(ib.req_ids[0].has_value());
  REQUIRE(ib.req_ids[1].has_value());
  CHECK(*ib.req_ids[0] == "D");
  CHECK(*ib.req_ids[1] == "P");

  // (b) Attention seq_lens + block_table rows: slot 0 == D (seq_len 4, fa block
  // 0), slot 1 == P (seq_len 5, fa block 2).
  const auto& am = runner.last_attn_meta();
  REQUIRE(am.seq_lens.size() == 2);
  CHECK(am.seq_lens[0] == 4);  // D: computed 3 + scheduled 1
  CHECK(am.seq_lens[1] == 5);  // P: computed 0 + scheduled 5
  const int cols = am.block_table_num_cols;
  CHECK(am.block_table_tensor[0] == 0);                        // D fa block 0
  CHECK(am.block_table_tensor[static_cast<size_t>(cols)] == 2);  // P fa block 2

  // (c) logits_indices: D's single token at flat index 0; P's last of 5 tokens
  // at flat index 5 (query_start_loc [0,1,6]).
  const auto& step = runner.last_step();
  REQUIRE(step.logits_indices.size() == 2);
  CHECK(step.logits_indices[0] == 0);  // D last token
  CHECK(step.logits_indices[1] == 5);  // P last token

  // (d) GDN metadata: 1 decode + 1 prefill, decode-first. State indices in the
  // reordered order = [D's gdn block 0, P's gdn block 1]; the prefill sub-batch
  // is P (state 1, fresh -> has_initial_state 0).
  const auto& gm = runner.last_gdn_meta();
  CHECK(gm.num_decodes == 1);
  CHECK(gm.num_prefills == 1);
  REQUIRE(gm.non_spec_state_indices_tensor.has_value());
  CHECK(*gm.non_spec_state_indices_tensor == std::vector<int32_t>{0, 1});
  REQUIRE(gm.prefill_state_indices.has_value());
  CHECK(*gm.prefill_state_indices == std::vector<int32_t>{1});
  REQUIRE(gm.has_initial_state.has_value());
  CHECK(*gm.has_initial_state == std::vector<uint8_t>{1, 0});  // D continues, P fresh

  // (e) SamplingMetadata row alignment: P (seeded) is at slot 1, so its seed
  // surfaces at generators[1]; D (unseeded, greedy) is absent.
  const auto sm = ib.make_sampling_metadata();
  CHECK_FALSE(sm.all_greedy);  // P is random
  REQUIRE(sm.generators.count(1) == 1);
  CHECK(sm.generators.at(1) == 12345u);
  CHECK(sm.generators.count(0) == 0);

  // (f) Write-back slot: sample -> the sampled token lands in the SAME slot the
  // request occupies. D's row grows at slot 0, P's at slot 1.
  const int d_tokens_before = ib.num_tokens_no_spec[0];  // D: prompt3+output1 = 4
  const int p_tokens_before = ib.num_tokens_no_spec[1];  // P: prompt5 = 5
  CHECK(d_tokens_before == 4);
  CHECK(p_tokens_before == 5);

  ModelRunnerOutput mro = runner.sample_tokens(std::nullopt);

  // The output order + index map match the dense (reordered) order.
  REQUIRE(mro.req_ids.size() == 2);
  CHECK(mro.req_ids[0] == "D");
  CHECK(mro.req_ids[1] == "P");
  CHECK(mro.req_id_to_index.at("D") == 0);
  CHECK(mro.req_id_to_index.at("P") == 1);
  REQUIRE(mro.sampled_token_ids.size() == 2);
  REQUIRE(mro.sampled_token_ids[0].size() == 1);
  REQUIRE(mro.sampled_token_ids[1].size() == 1);

  // Write-back appended one token to each request's OWN row.
  const auto& ib2 = runner.input_batch();
  CHECK(ib2.num_tokens_no_spec[0] == d_tokens_before + 1);  // D grew
  CHECK(ib2.num_tokens_no_spec[1] == p_tokens_before + 1);  // P grew
  // The sampled token was written at the request's next free column.
  CHECK(ib2.token_id(0, d_tokens_before) == mro.sampled_token_ids[0][0]);
  CHECK(ib2.token_id(1, p_tokens_before) == mro.sampled_token_ids[1][0]);
}

// ─── discard_request_mask (chunked prefill returns EMPTY tokens) ─────────────
// A partial prefill chunk (num_scheduled < num_tokens => optimistic seq_len <
// num_tokens) must NOT sample: gpu_model_runner.py:2048 discard_request_mask +
// outputs.py:303 valid_sampled_token_ids[i].clear(). The scheduler REQUIRES the
// runner to return empty token ids for a still-prefilling request
// (scheduler.py:1888-1890) — otherwise the spurious token is appended as output,
// and under async scheduling it underflows num_output_placeholders (the c8 +
// short-output crash, ENG-ASYNC-SCHED). This test is RED without the discard
// mask (the chunk samples a garbage token and its row grows) and GREEN with it.
TEST_CASE("runner: chunked prefill returns empty sampled tokens (discard_request_mask)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);

  // C: a 10-token prompt scheduled in a FIRST chunk of only 5 tokens. seq_len ==
  // 5 < num_tokens == 10, so this step is still consuming prefill tokens.
  NewRequestData ch =
      MakeNewReq("C", {1, 2, 3, 4, 5, 6, 7, 8, 9, 10}, {}, /*num_computed=*/0,
                 /*fa_blocks=*/{2, 3}, /*gdn_block=*/1, Greedy());
  SchedulerOutput so = NewStep({ch}, {{"C", 5}});

  auto out_opt = runner.execute_model(so);
  CHECK_FALSE(out_opt.has_value());

  const auto& ib = runner.input_batch();
  REQUIRE(ib.num_reqs() == 1);
  CHECK(*ib.req_ids[0] == "C");
  const int c_tokens_before = ib.num_tokens_no_spec[0];  // prompt 10, no output
  CHECK(c_tokens_before == 10);

  ModelRunnerOutput mro = runner.sample_tokens(std::nullopt);

  // The request is still present in the output (order/index preserved) but its
  // sampled token list is EMPTY — the scheduler appends no output token.
  REQUIRE(mro.req_ids.size() == 1);
  CHECK(mro.req_ids[0] == "C");
  REQUIRE(mro.sampled_token_ids.size() == 1);
  CHECK(mro.sampled_token_ids[0].empty());

  // No write-back: the prefill chunk generated no token, so its row must not grow.
  const auto& ib2 = runner.input_batch();
  CHECK(ib2.num_tokens_no_spec[0] == c_tokens_before);
}

// A 3-request mixed batch admitted [P0 prefill, P1 prefill, D decode]. The
// decode-first reorder must pull D to the front, moving ≥2 requests. Rather than
// hard-code the exact post-partition permutation (upstream does a MINIMUM-SWAP
// partition, not a stable sort — [P0,P1,D] -> swap(0,2) -> [D,P1,P0]), this asserts
// the order-INDEPENDENT invariant: whatever slot each request lands in, EVERY
// per-slot field (seq_len, fa block, GDN state index, seed, token count) still
// resolves to that SAME request. A field left behind during a swap_states chain
// would desync exactly one of these against the req_id at its slot.
TEST_CASE("runner: 3-request reorder keeps every per-slot field self-consistent") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);

  SamplingParams p0_params;
  p0_params.temperature = 0.7;
  p0_params.top_k = 2;
  p0_params.seed = 111;
  p0_params.PostInit();
  SamplingParams p1_params;
  p1_params.temperature = 0.7;
  p1_params.top_k = 2;
  p1_params.seed = 222;
  p1_params.PostInit();

  // Per-request expected fields, keyed by req_id (order-independent oracle).
  struct Expect {
    int seq_len;
    int fa_block;
    int gdn_state;
    unsigned seed;  // 0 = greedy / no generator
    int tokens;
  };
  const std::map<std::string, Expect> want = {
      {"D", {4, 0, 0, 0, 4}},     // decode: prompt3+out1, fa block 0, gdn 0, greedy
      {"P0", {5, 2, 1, 111, 5}},  // prefill 5, fa block 2, gdn 1, seed 111
      {"P1", {4, 4, 2, 222, 4}},  // prefill 4, fa block 4, gdn 2, seed 222
  };

  NewRequestData p0 = MakeNewReq("P0", {1, 2, 3, 4, 5}, {}, /*num_computed=*/0,
                                 /*fa_blocks=*/{2, 3}, /*gdn_block=*/1, p0_params);
  NewRequestData p1 = MakeNewReq("P1", {10, 11, 12, 13}, {}, /*num_computed=*/0,
                                 /*fa_blocks=*/{4, 5}, /*gdn_block=*/2, p1_params);
  NewRequestData d = MakeNewReq("D", {6, 7, 8}, {9}, /*num_computed=*/3,
                                /*fa_blocks=*/{0, 1}, /*gdn_block=*/0, Greedy());

  SchedulerOutput so = NewStep({p0, p1, d}, {{"P0", 5}, {"P1", 4}, {"D", 1}});
  auto out_opt = runner.execute_model(so);
  CHECK_FALSE(out_opt.has_value());

  const auto& ib = runner.input_batch();
  REQUIRE(ib.num_reqs() == 3);
  // Decode must lead after the reorder.
  CHECK(*ib.req_ids[0] == "D");

  const auto& am = runner.last_attn_meta();
  const auto& gm = runner.last_gdn_meta();
  const auto sm = ib.make_sampling_metadata();
  const int cols = am.block_table_num_cols;
  REQUIRE(gm.non_spec_state_indices_tensor.has_value());

  // For each occupied slot, cross-check ALL five per-slot fields against the
  // oracle for whichever request landed there.
  for (int i = 0; i < ib.num_reqs(); ++i) {
    REQUIRE(ib.req_ids[static_cast<size_t>(i)].has_value());
    const std::string rid = *ib.req_ids[static_cast<size_t>(i)];
    const Expect& e = want.at(rid);
    CHECK(am.seq_lens[static_cast<size_t>(i)] == e.seq_len);
    CHECK(am.block_table_tensor[static_cast<size_t>(i * cols)] == e.fa_block);
    // GDN state index is now the COMPACT per-sequence state slot
    // (remap_gdn_state_slots): the raw mamba pool block-id (col 0, scattered
    // over the shared attention pool) is remapped to a slot in
    // [0, gdn_state_slots_) assigned in first-appearance order, so the GDN state
    // cache is sized by max_num_reqs (one recurrent state per sequence) rather
    // than num_blocks. For a single step of all-new requests that slot is
    // exactly the batch row i, whichever request landed there.
    (void)e.gdn_state;
    CHECK((*gm.non_spec_state_indices_tensor)[static_cast<size_t>(i)] == i);
    CHECK(ib.num_tokens_no_spec[static_cast<size_t>(i)] == e.tokens);
    if (e.seed == 0) {
      CHECK(sm.generators.count(i) == 0);
    } else {
      REQUIRE(sm.generators.count(i) == 1);
      CHECK(sm.generators.at(i) == e.seed);
    }
  }
}

// ─── 3. Single-request greedy decode over N steps ────────────────────────────
TEST_CASE("runner: single-request greedy decode over N steps (KV grows, feedback)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  // Production Qwen3.5 planning publishes BF16 conv + FP32 temporal state.
  // Exercise that exact MambaSpec on the CPU reference runner as well: its
  // model boundary must gather/downcast compressed cache rows without relying
  // on a CUDA-only cast kernel.
  GPUModelRunner runner(c, w, MakeKvConfig(c, DType::kBF16, DType::kF32),
                        Q(), 8, kMaxModelLen, 64);

  const std::vector<int32_t> prompt = {5, 9, 2, 31, 17};
  const int P = static_cast<int>(prompt.size());

  // Step 1: prefill.
  SchedulerOutput s1 =
      NewStep({MakeNewReq("A", prompt, {}, 0, {0, 1}, 0, Greedy())}, {{"A", P}});
  CHECK_FALSE(runner.execute_model(s1).has_value());
  ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  REQUIRE(m1.sampled_token_ids.size() == 1);
  REQUIRE(m1.sampled_token_ids[0].size() == 1);
  const int32_t tok1 = m1.sampled_token_ids[0][0];

  // The token was written back at column P (== the decode input next step).
  CHECK(runner.input_batch().num_tokens_no_spec[0] == P + 1);
  CHECK(runner.input_batch().token_id(0, P) == tok1);

  // The full-attn KV cache grew: the prefill wrote non-zero K/V into block 0.
  const PagedKvCache& kv = runner.attn_kv()[0];
  const auto* kvp = static_cast<const float*>(kv.data);
  bool kv_nonzero = false;
  for (int64_t i = 0; i < 2 * kBlockSize * c.num_key_value_heads * c.head_dim;
       ++i)
    if (kvp[i] != 0.0f) kv_nonzero = true;
  CHECK(kv_nonzero);

  // Steps 2..N: decode. Each reads the previous sampled token + the grown cache.
  int computed = P;
  int outputs = 1;
  int32_t prev = tok1;
  for (int stepn = 0; stepn < 4; ++stepn) {
    SchedulerOutput sd = DecodeStep("A", computed, outputs);
    CHECK_FALSE(runner.execute_model(sd).has_value());
    // prepare_inputs must have read the previously sampled token as the input.
    CHECK(runner.last_step().input_token_ids == std::vector<int32_t>{prev});
    CHECK(runner.last_step().positions == std::vector<int64_t>{computed});
    ModelRunnerOutput md = runner.sample_tokens(std::nullopt);
    REQUIRE(md.sampled_token_ids[0].size() == 1);
    prev = md.sampled_token_ids[0][0];
    computed += 1;
    outputs += 1;
    // The new token appended at the next column.
    CHECK(runner.input_batch().num_tokens_no_spec[0] == computed + 1);
    CHECK(runner.input_batch().token_id(0, computed) == prev);
  }
  CHECK(outputs == 5);  // 1 prefill sample + 4 decodes
}

// ─── ENG-ASYNC-SCHED W3: async device-input path (combine) ───────────────────
TEST_CASE("runner: async_input_combine decode is token-identical to the sync path") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const std::vector<int32_t> prompt = {5, 9, 2, 31, 17};
  const int P = static_cast<int>(prompt.size());

  // Run a prefill + N greedy decodes through a fresh runner with the async
  // device-input path either off (host token_ids_cpu read) or on (combine splices
  // the id from last_sampled_tokens). Greedy tokens must be bit-identical (G1).
  auto run = [&](bool async_combine) {
    GPUModelRunner runner(c, w, MakeKvConfig(c, DType::kBF16, DType::kF32), Q(), 8,
                          kMaxModelLen, 64);
    runner.set_async_input_combine(async_combine);
    CHECK(runner.async_input_combine() == async_combine);
    std::vector<int32_t> tokens;
    SchedulerOutput s1 =
        NewStep({MakeNewReq("A", prompt, {}, 0, {0, 1}, 0, Greedy())}, {{"A", P}});
    CHECK_FALSE(runner.execute_model(s1).has_value());
    ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
    tokens.push_back(m1.sampled_token_ids[0][0]);
    // sample_tokens records the last sampled id per req_state (post_update).
    CHECK(runner.input_batch().last_sampled_tokens[0] == tokens.back());

    int computed = P, outputs = 1;
    for (int k = 0; k < 5; ++k) {
      SchedulerOutput sd = DecodeStep("A", computed, outputs);
      CHECK_FALSE(runner.execute_model(sd).has_value());
      // Either path feeds the previous sampled token as this step's input.
      CHECK(runner.last_step().input_token_ids ==
            std::vector<int32_t>{tokens.back()});
      ModelRunnerOutput md = runner.sample_tokens(std::nullopt);
      tokens.push_back(md.sampled_token_ids[0][0]);
      CHECK(runner.input_batch().last_sampled_tokens[0] == tokens.back());
      computed += 1;
      outputs += 1;
    }
    return tokens;
  };

  const std::vector<int32_t> sync = run(false);
  const std::vector<int32_t> async_combine = run(true);
  CHECK(async_combine == sync);  // token-for-token identical in both modes
}

TEST_CASE("runner: async device-input reads last_sampled over a stale host token") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c, DType::kBF16, DType::kF32), Q(), 8,
                        kMaxModelLen, 64);
  runner.set_async_input_combine(true);

  const std::vector<int32_t> prompt = {5, 9, 2, 31, 17};
  const int P = static_cast<int>(prompt.size());
  SchedulerOutput s1 =
      NewStep({MakeNewReq("A", prompt, {}, 0, {0, 1}, 0, Greedy())}, {{"A", P}});
  CHECK_FALSE(runner.execute_model(s1).has_value());
  ModelRunnerOutput m1 = runner.sample_tokens(std::nullopt);
  const int32_t tok1 = m1.sampled_token_ids[0][0];
  REQUIRE(runner.input_batch().last_sampled_tokens[0] == tok1);

  // Corrupt the HOST token buffer at the next decode column (== the async
  // D2H-skip: token_ids_cpu is stale because the sampled id never crossed back).
  // combine must build the input id from the GPU-resident-analog last_sampled,
  // ignoring the corrupted host value.
  const int32_t kCorrupt = 12345;
  runner.input_batch().token_ids_cpu[static_cast<size_t>(P)] = kCorrupt;

  SchedulerOutput sd = DecodeStep("A", P, 1);
  CHECK_FALSE(runner.execute_model(sd).has_value());
  CHECK(runner.last_step().input_token_ids == std::vector<int32_t>{tok1});
  CHECK(tok1 != kCorrupt);
}

// ─── ENG-ASYNC-SCHED depth-2 LIFETIME GUARD (serving heap-corruption regression) ─
// Root cause of the 35B serving abort (`malloc(): unaligned tcache chunk`) under
// async scheduling + ignore_eos past ~4-8 decode tokens: sample_tokens_async
// leaves the forward / sample / scatter on the MAIN queue UNSYNCED (the depth-2
// overlap defers the sampled-id D2H to the consuming step's get_output(), one
// step_with_batch_queue call later). Those kernels still reference exec_state_
// (device logits + StepInputs host arrays) and write input_batch_.last_sampled_
// tokens. The NEXT execute_model() reset exec_state_ and mutated input_batch_
// (update_states condense/swap) WHILE they ran -> use-after-free on GB10 unified
// memory. It only reproduces under REAL GPU overlap: the CPU eager backend and
// compute-sanitizer both serialize the queue, so the CPU gates stayed green while
// the served model aborted. This test locks the INVARIANT the fix enforces:
// sample_tokens_async marks async work outstanding, and the next execute_model
// drains it before touching any shared state. RED before the fix (execute_model
// never drained -> the flag would stay set / the field did not exist); GREEN after.
TEST_CASE("runner: async sample_tokens_async work is drained before the next execute_model") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c, DType::kBF16, DType::kF32), Q(), 8,
                        kMaxModelLen, 64);
  runner.set_async_input_combine(true);

  const std::vector<int32_t> prompt = {5, 9, 2, 31, 17};
  const int P = static_cast<int>(prompt.size());

  // Fresh runner: no async work is outstanding yet.
  CHECK_FALSE(runner.async_forward_in_flight());

  // Prefill via the ASYNC sample path (the serving path — step_with_batch_queue).
  SchedulerOutput s1 =
      NewStep({MakeNewReq("A", prompt, {}, 0, {0, 1}, 0, Greedy())}, {{"A", P}});
  CHECK_FALSE(runner.execute_model(s1).has_value());
  std::unique_ptr<vllm::v1::AsyncModelRunnerOutput> a1 =
      runner.sample_tokens_async(std::nullopt);
  // The async sample left main-queue work referencing exec_state_ / input_batch_.
  CHECK(runner.async_forward_in_flight());
  ModelRunnerOutput m1 = a1->get_output();
  REQUIRE(m1.sampled_token_ids[0].size() == 1);
  int32_t last = m1.sampled_token_ids[0][0];

  // Continue past the trigger (ignore_eos regime): every async decode step must
  // (a) find the guard already set from the prior step, then (b) DRAIN it at the
  // top of execute_model before it resets exec_state_ / condenses input_batch_.
  int computed = P, outputs = 1;
  for (int k = 0; k < 10; ++k) {
    SchedulerOutput sd = DecodeStep("A", computed, outputs);
    CHECK(runner.async_forward_in_flight());          // set by the previous step
    CHECK_FALSE(runner.execute_model(sd).has_value());
    CHECK_FALSE(runner.async_forward_in_flight());     // DRAINED before mutation
    // The decode input id is the previous step's sampled token (state intact).
    CHECK(runner.last_step().input_token_ids == std::vector<int32_t>{last});
    std::unique_ptr<vllm::v1::AsyncModelRunnerOutput> ad =
        runner.sample_tokens_async(std::nullopt);
    CHECK(runner.async_forward_in_flight());
    ModelRunnerOutput md = ad->get_output();
    REQUIRE(md.sampled_token_ids[0].size() == 1);
    last = md.sampled_token_ids[0][0];
    computed += 1;
    outputs += 1;
  }

  // The SYNC sample path leaves nothing outstanding (it synchronizes to read the
  // ids on host), so it must NOT arm the guard — the sync engine is unaffected.
  GPUModelRunner sync_runner(c, w, MakeKvConfig(c, DType::kBF16, DType::kF32),
                             Q(), 8, kMaxModelLen, 64);
  SchedulerOutput s2 =
      NewStep({MakeNewReq("A", prompt, {}, 0, {0, 1}, 0, Greedy())}, {{"A", P}});
  CHECK_FALSE(sync_runner.execute_model(s2).has_value());
  (void)sync_runner.sample_tokens(std::nullopt);
  CHECK_FALSE(sync_runner.async_forward_in_flight());
}

// ─── 4. Two-request greedy batch step (per-request logits rows) ───────────────
TEST_CASE("runner: 2-request greedy batch samples each from its own logits row") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);

  const std::vector<int32_t> a_prompt = {5, 9, 2, 31};
  const std::vector<int32_t> b_prompt = {7, 1, 22};

  // Both fresh prefills, greedy; A at fa blocks {0,1}/gdn 0, B at {2,3}/gdn 1.
  SchedulerOutput so = NewStep(
      {MakeNewReq("A", a_prompt, {}, 0, {0, 1}, 0, Greedy()),
       MakeNewReq("B", b_prompt, {}, 0, {2, 3}, 1, Greedy())},
      {{"A", static_cast<int>(a_prompt.size())},
       {"B", static_cast<int>(b_prompt.size())}});

  CHECK_FALSE(runner.execute_model(so).has_value());
  ModelRunnerOutput mro = runner.sample_tokens(std::nullopt);

  REQUIRE(mro.req_ids.size() == 2);
  // Both prefills -> no reorder; dense order == admission order [A, B].
  CHECK(mro.req_ids[0] == "A");
  CHECK(mro.req_ids[1] == "B");
  REQUIRE(mro.sampled_token_ids[0].size() == 1);
  REQUIRE(mro.sampled_token_ids[1].size() == 1);

  // Each request's greedy token == the argmax of its OWN last-token logits from
  // a standalone dense forward (the paged batch row is per-request independent).
  vt::Queue q = Q();
  std::vector<int32_t> a_pos(a_prompt.size());
  for (size_t i = 0; i < a_pos.size(); ++i) a_pos[i] = static_cast<int32_t>(i);
  std::vector<int32_t> b_pos(b_prompt.size());
  for (size_t i = 0; i < b_pos.size(); ++i) b_pos[i] = static_cast<int32_t>(i);
  const std::vector<float> a_dense =
      Qwen3_5Model::ForwardDense(a_prompt, a_pos, w, c, q);
  const std::vector<float> b_dense =
      Qwen3_5Model::ForwardDense(b_prompt, b_pos, w, c, q);
  const int a_expect = GreedyArgmax(
      a_dense, static_cast<int64_t>(a_prompt.size()) - 1, c.vocab_size);
  const int b_expect = GreedyArgmax(
      b_dense, static_cast<int64_t>(b_prompt.size()) - 1, c.vocab_size);

  CHECK(mro.sampled_token_ids[0][0] == a_expect);
  CHECK(mro.sampled_token_ids[1][0] == b_expect);
}

// ─── 5. GDN state-slot uniqueness under multi-block sequences (c16 regression) ─
// Captured engine-fatal reproduction: "vt: qwen3_5: duplicate live GDN state
// index" (ValidateGdnStateIndices, qwen3_5.cpp:73), deterministic 3/3 on the
// c16 96-request burst.
//
// Root cause: the 27B GDN/mamba KV group is configured with a sub-sequence
// block_size (MakeQwen3_5KVCache passes the attention block_size while the
// MambaSpec default cache mode is "none"). Once a sequence exceeds one block it
// accumulates cdiv(seq_len, block_size) mamba blocks and
// MambaManager::remove_skipped_blocks nulls every block but the last, so
// block-table column 0 collapses to the shared null block-id 0. The runner's
// compact GDN state pool used to key on that block-id, so two live "long"
// sequences both presenting col-0 == 0 were mapped onto ONE state slot — a
// duplicate live state index (and, before the W1D2 validator, silent
// cross-request recurrent-state corruption). vLLM instead gathers the CURRENT
// state block (mamba_get_block_table_tensor) and, semantically, owns one
// recurrent state per SEQUENCE. The fix keys the compact slot on the request
// identity, so each live sequence owns exactly one slot regardless of the
// physical block layout.
//
// These requests present col-0 == 0 exactly as the cache manager produces it
// after skipping the front blocks of a multi-block sequence.
TEST_CASE("runner: GDN state slots stay unique when col-0 collapses to null block") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  // Two decode requests, each already past its first mamba block: the cache
  // manager has nulled column 0 to the shared null block-id 0 for both.
  NewRequestData a = MakeNewReq("A", {5, 9, 2}, {7}, /*num_computed=*/3,
                                /*fa_blocks=*/{0, 1}, /*gdn_block=*/0, Greedy());
  NewRequestData b = MakeNewReq("B", {1, 4, 8}, {6}, /*num_computed=*/3,
                                /*fa_blocks=*/{2, 3}, /*gdn_block=*/0, Greedy());
  SchedulerOutput so = NewStep({a, b}, {{"A", 1}, {"B", 1}});

  // BEFORE the fix this remaps both sequences onto ONE slot -> the GDN metadata
  // carries a duplicate live state index and the validator fatals.
  CHECK_NOTHROW(runner.execute_model(so));

  const auto& gm = runner.last_gdn_meta();
  REQUIRE(gm.non_spec_state_indices_tensor.has_value());
  const std::vector<int32_t>& idx = *gm.non_spec_state_indices_tensor;
  REQUIRE(idx.size() == 2);
  // The two live sequences must occupy DIFFERENT state slots.
  CHECK(idx[0] != idx[1]);
  CHECK(idx[0] >= 0);
  CHECK(idx[1] >= 0);
  // The validator (the exact check that fatally fired on dgx) must accept it.
  CHECK_NOTHROW(vllm::detail::ValidateGdnStateIndices(
      idx, /*required=*/2, runner.gdn_state_slots()));
}

// ─── 6. GDN state slots stay unique across completion→admission churn ─────────
// Drives the ordering the c16 burst exercised: long sequences decode while
// others complete and new ones are admitted (recycling pool block-ids). Every
// step's live sequences must hold pairwise-distinct GDN state slots, a finished
// sequence's slot must be released and reusable, and a continuing sequence must
// keep its slot (so its recurrent state persists).
TEST_CASE("runner: GDN state slots unique across completion/admission churn") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  auto slots_unique = [&](int expect_n) {
    const auto& gm = runner.last_gdn_meta();
    REQUIRE(gm.non_spec_state_indices_tensor.has_value());
    const std::vector<int32_t>& idx = *gm.non_spec_state_indices_tensor;
    REQUIRE(static_cast<int>(idx.size()) == expect_n);
    for (size_t i = 0; i < idx.size(); ++i) {
      CHECK(idx[i] >= 0);
      for (size_t j = i + 1; j < idx.size(); ++j) CHECK(idx[i] != idx[j]);
    }
    CHECK_NOTHROW(vllm::detail::ValidateGdnStateIndices(
        idx, expect_n, runner.gdn_state_slots()));
    return idx;
  };
  auto slot_of = [&](const std::string& rid) -> int32_t {
    const auto& ib = runner.input_batch();
    const auto& gm = runner.last_gdn_meta();
    const int r = ib.req_id_to_index.at(rid);
    return (*gm.non_spec_state_indices_tensor)[static_cast<size_t>(r)];
  };

  // Step 1: admit A, B — both multi-block sequences (col-0 == 0).
  NewRequestData a = MakeNewReq("A", {5, 9, 2}, {7}, 3, {0, 1}, /*gdn=*/0, Greedy());
  NewRequestData b = MakeNewReq("B", {1, 4, 8}, {6}, 3, {2, 3}, /*gdn=*/0, Greedy());
  CHECK_NOTHROW(runner.execute_model(NewStep({a, b}, {{"A", 1}, {"B", 1}})));
  slots_unique(2);
  const int32_t b_slot = slot_of("B");

  // Step 2: A completes; new sequence C admitted (also col-0 == 0, the block-id
  // A freed is recycled); B continues its decode. The pool must release A's slot
  // and hand C a slot distinct from B's, while B keeps its own slot.
  NewRequestData cc = MakeNewReq("C", {3, 3, 3}, {9}, 3, {0, 1}, /*gdn=*/0, Greedy());
  SchedulerOutput s2;
  s2.finished_req_ids = {"A"};
  CachedRequestData cached;
  cached.req_ids = {"B"};
  cached.num_computed_tokens = {4};
  cached.num_output_tokens = {1};
  cached.new_block_ids.emplace_back(std::nullopt);
  s2.scheduled_cached_reqs = std::move(cached);
  s2.scheduled_new_reqs = {cc};
  s2.num_scheduled_tokens = {{"B", 1}, {"C", 1}};
  s2.total_num_scheduled_tokens = 2;
  CHECK_NOTHROW(runner.execute_model(s2));
  slots_unique(2);
  CHECK(slot_of("B") == b_slot);  // B's recurrent state slot is stable.
  CHECK(slot_of("C") != slot_of("B"));
}

// ─── ENG-ASYNC-SCHED W3: async device-OUTPUT path (sample_tokens_async) ───────
// The overlap output half: sample_tokens_async produces the sampled ids
// device-resident and returns an AsyncModelRunnerOutput whose get_output()
// materializes them off a copy queue + event. Greedy tokens must be bit-identical
// to the synchronous sample_tokens (G1), and last_sampled_tokens must be recorded
// at sample time (before get_output) so the next step's combine reads it.
TEST_CASE("runner: sample_tokens_async decode is token-identical to sync") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const std::vector<int32_t> prompt = {5, 9, 2, 31, 17};
  const int P = static_cast<int>(prompt.size());

  auto run = [&](bool async_output) {
    GPUModelRunner runner(c, w, MakeKvConfig(c, DType::kBF16, DType::kF32), Q(), 8,
                          kMaxModelLen, 64);
    runner.set_async_input_combine(async_output);
    // SPEC-DFLASH2 W7 (#1824): runner_supports_async() is the env/backend
    // capability predicate and no longer tracks the input-combine lever (async
    // SCHEDULING must survive a spec engine whose combine is vetoed — I5e).
    // The lever this case toggles is the combine itself:
    CHECK(runner.async_input_combine() == async_output);
    std::vector<int32_t> tokens;

    auto sample = [&]() -> int32_t {
      if (async_output) {
        std::unique_ptr<vllm::v1::AsyncModelRunnerOutput> a =
            runner.sample_tokens_async(std::nullopt);
        // last_sampled is recorded at sample time (on-GPU post_update), BEFORE
        // the host materialization — the next step's combine depends on it.
        const int32_t recorded = runner.input_batch().last_sampled_tokens[0];
        ModelRunnerOutput m = a->get_output();
        CHECK(m.sampled_token_ids[0][0] == recorded);
        return m.sampled_token_ids[0][0];
      }
      ModelRunnerOutput m = runner.sample_tokens(std::nullopt);
      return m.sampled_token_ids[0][0];
    };

    SchedulerOutput s1 =
        NewStep({MakeNewReq("A", prompt, {}, 0, {0, 1}, 0, Greedy())}, {{"A", P}});
    CHECK_FALSE(runner.execute_model(s1).has_value());
    tokens.push_back(sample());

    int computed = P, outputs = 1;
    for (int k = 0; k < 5; ++k) {
      SchedulerOutput sd = DecodeStep("A", computed, outputs);
      CHECK_FALSE(runner.execute_model(sd).has_value());
      // Under async both input-combine and output-D2H are on: the previous
      // sampled token feeds this step's input via last_sampled_tokens.
      CHECK(runner.last_step().input_token_ids ==
            std::vector<int32_t>{tokens.back()});
      tokens.push_back(sample());
      computed += 1;
      outputs += 1;
    }
    return tokens;
  };

  const std::vector<int32_t> sync = run(false);
  const std::vector<int32_t> async_out = run(true);
  CHECK(async_out == sync);  // token-for-token identical
}

// ─── SPEC-DFLASH2 A2-4 (#2920): sample_tokens_async's DECODE arm must propose ─
//
// A2-2 gave `sample_tokens_async` a verify arm and put `propose_after_verify`
// inside it, so a step that DRAFTS proposed correctly and a step that did not
// proposed nothing at all. The arm that runs when `StepRoutesToVerify` is false
// is a speculative engine's FIRST step — there is no previous step to have
// proposed — so an engine on that path would sample, propose nothing, and every
// step after it would find nothing to verify: `pending_drafts_` stays empty and
// the async scheduler's `-1` placeholders have nothing to fill.
//
// WHAT THESE CASES SUPPLY BY HAND, and what they do not. The runner is the
// production type and `execute_model` is the production entry point. The one
// thing supplied is `set_async_input_combine(true)`, which is exactly what the
// veto at both `GPUModelRunner` constructors withholds from every speculative
// engine. Lifting that veto is A2-5, it is not this change, and until it lands
// this arm is UNREACHED from any production configuration — recorded in the
// spec's `## Owed` rather than implied by these cases.
//
// THE SPECULATOR IS THE N-GRAM PROPOSER because it is the only arm that needs
// no draft model, which keeps the case a runner case instead of an engine one.
//
// WHAT THE DRAFT ASSERTION PROVES. `take_draft_token_ids()` moves out
// `pending_drafts_`, and the only writer of that member is `set_draft_tokens`,
// which is also the only caller of `ProposedDraftLedger::Record`. So a value
// here is a propose that ran AND the ledger entry the async placeholder fill
// reads for freshness; the two cannot come apart at this seam.
TEST_CASE("runner: sample_tokens_async's decode arm proposes on the first step") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  // The LoadedModel constructor is the one that carries a SpeculativeConfig;
  // the concrete-weight overloads do not.
  std::unique_ptr<vllm::LoadedModel> lm = vllm::BorrowQwen3_5MoeLoadedModel(w);
  const vllm::SpeculativeConfig spec = vllm::SpeculativeConfig::ResolveNgram(
      /*k=*/3, /*user_min=*/2, /*user_max=*/2);
  GPUModelRunner runner(c, *lm, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64, spec);
  runner.set_async_input_combine(true);
  REQUIRE(runner.async_input_combine());

  const std::vector<int32_t> prompt = {5, 9, 2, 31, 17};
  const int P = static_cast<int>(prompt.size());
  SchedulerOutput s1 =
      NewStep({MakeNewReq("A", prompt, {}, 0, {0, 1}, 0, Greedy())}, {{"A", P}});
  REQUIRE_FALSE(runner.execute_model(s1).has_value());

  std::unique_ptr<vllm::v1::AsyncModelRunnerOutput> a =
      runner.sample_tokens_async(std::nullopt);
  const ModelRunnerOutput m = a->get_output();
  REQUIRE(m.sampled_token_ids.size() == 1);
  REQUIRE(m.sampled_token_ids[0].size() == 1);
  const int32_t sampled = m.sampled_token_ids[0][0];

  // THE DEFECT #2920 NAMES: before the fix this is nullopt, because the decode
  // arm never reached a propose.
  const std::optional<vllm::v1::DraftTokenIds> drafts =
      runner.take_draft_token_ids();
  REQUIRE(drafts.has_value());
  CHECK(drafts->req_ids == std::vector<std::string>{"A"});
  CHECK(drafts->draft_token_ids.size() == 1u);

  // The n-gram matcher reads `token_ids_cpu[:num_tokens_no_spec]`, and the
  // async arm advances that counter without writing the column. The propose
  // branch restores the value at the column the counter moved past; without it
  // the matcher drafts off a token nobody committed, which costs acceptance and
  // raises nothing.
  const auto& ib = runner.input_batch();
  const int row = ib.req_id_to_index.at("A");
  REQUIRE(ib.num_tokens_no_spec[static_cast<size_t>(row)] == P + 1);
  CHECK(ib.last_sampled_tokens[static_cast<size_t>(row)] == sampled);
  CHECK(ib.token_ids_cpu[static_cast<size_t>(row) *
                             static_cast<size_t>(ib.max_model_len) +
                         static_cast<size_t>(P)] == sampled);
}

// The same arm on a MIXED step with num_reqs > 1, which is the shape this row
// has already shipped a wrong answer in: a per-request reading of a per-step
// rule answers differently when some rows commit a token and others do not, and
// `tests/vllm/v1/worker/test_combine_row_predicate.cpp` (#2710) records a
// version that survived 27 mutations because every case used num_reqs == 1.
//
// "P" is fully prefilled in its step and commits a token. "C" is a chunked
// prefill continuation: it is scheduled, it samples a garbage token at its
// prefill position, and the discard mask says it committed nothing. The step
// carries no drafts either way, so it is one DECODE-arm step with two rows that
// the propose must treat differently.
TEST_CASE("runner: the async decode arm's propose covers a mixed num_reqs > 1 step") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  std::unique_ptr<vllm::LoadedModel> lm = vllm::BorrowQwen3_5MoeLoadedModel(w);
  const vllm::SpeculativeConfig spec = vllm::SpeculativeConfig::ResolveNgram(
      /*k=*/3, /*user_min=*/2, /*user_max=*/2);
  GPUModelRunner runner(c, *lm, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64, spec);
  runner.set_async_input_combine(true);

  const std::vector<int32_t> p_prompt = {5, 9, 2, 31, 17, 4};
  std::vector<int32_t> c_prompt(20);
  for (int i = 0; i < 20; ++i) c_prompt[static_cast<size_t>(i)] = 1 + i;
  SchedulerOutput s1 = NewStep(
      {MakeNewReq("P", p_prompt, {}, /*num_computed=*/0, {0}, 0, Greedy()),
       MakeNewReq("C", c_prompt, {}, /*num_computed=*/0, {1, 2}, 1, Greedy())},
      {{"P", 6}, {"C", 10}});
  REQUIRE_FALSE(runner.execute_model(s1).has_value());
  REQUIRE(runner.last_step().num_draft_tokens == 0);  // the DECODE arm

  std::unique_ptr<vllm::v1::AsyncModelRunnerOutput> a =
      runner.sample_tokens_async(std::nullopt);
  const ModelRunnerOutput m = a->get_output();
  REQUIRE(m.req_id_to_index.count("P") == 1u);
  REQUIRE(m.req_id_to_index.count("C") == 1u);
  const int32_t p_sampled =
      m.sampled_token_ids[static_cast<size_t>(m.req_id_to_index.at("P"))][0];
  // The discarded row emits nothing (outputs.py:303).
  CHECK(m.sampled_token_ids[static_cast<size_t>(m.req_id_to_index.at("C"))]
            .empty());

  const std::optional<vllm::v1::DraftTokenIds> drafts =
      runner.take_draft_token_ids();
  REQUIRE(drafts.has_value());
  REQUIRE(drafts->req_ids.size() == 2u);
  REQUIRE(drafts->draft_token_ids.size() == 2u);
  std::map<std::string, std::vector<int32_t>> by_id;
  for (size_t i = 0; i < drafts->req_ids.size(); ++i) {
    by_id[drafts->req_ids[i]] = drafts->draft_token_ids[i];
  }
  REQUIRE(by_id.count("P") == 1u);
  REQUIRE(by_id.count("C") == 1u);
  // A row that committed nothing this step drafts nothing
  // (ngram_proposer.py:157-172 valid_ngram_requests).
  CHECK(by_id["C"].empty());

  const auto& ib = runner.input_batch();
  const int p_row = ib.req_id_to_index.at("P");
  const int c_row = ib.req_id_to_index.at("C");
  // The committed row's column advanced and carries its token; the discarded
  // row's counter did not move (add_request seeds it from request.num_tokens(),
  // so it still reads the 20-token prompt length and no column past it was
  // written).
  REQUIRE(ib.num_tokens_no_spec[static_cast<size_t>(p_row)] == 7);
  CHECK(ib.token_ids_cpu[static_cast<size_t>(p_row) *
                             static_cast<size_t>(ib.max_model_len) +
                         6u] == p_sampled);
  CHECK(ib.num_tokens_no_spec[static_cast<size_t>(c_row)] == 20);
}

// ─── W1: runner generalization to a FULL-ATTENTION-ONLY model ────────────────
// The first additive-model bring-up (Qwen3ForCausalLM) forces the runner to
// stop assuming the Qwen3.6 hybrid KV topology. These two cases pin the fix:
// pre-generalization both crash (empty layer_types[] index; block_table[-1]).

TEST_CASE("runner: full-attention-only KV config allocates without the GDN path") {
  const HfConfig c = MakeDenseOnlyConfig();
  const Qwen3_5DenseWeights w = MakeDenseOnlyWeights(c);
  // Pre-fix: initialize_kv_cache indexed config_.layer_types[l] with an EMPTY
  // layer_types (out of bounds). Post-fix: no mamba group ⇒ every layer is
  // full-attention and one PagedKvCache is allocated per layer.
  GPUModelRunner runner(c, w, MakeFaOnlyKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  CHECK(runner.full_attn_group_id() == 0);
  CHECK(runner.gdn_group_id() == -1);          // NO GDN group
  CHECK(runner.num_blocks() == kNumBlocks);

  // One PagedKvCache per (full-attention) layer, and ZERO GDN state caches.
  REQUIRE(runner.attn_kv().size() ==
          static_cast<size_t>(c.num_hidden_layers));
  CHECK(runner.gdn_state().empty());
  const PagedKvCache& kv = runner.attn_kv()[0];
  CHECK(kv.num_blocks == kNumBlocks);
  CHECK(kv.block_size == kBlockSize);
  CHECK(kv.num_kv_heads == c.num_key_value_heads);
  CHECK(kv.head_size == c.head_dim);
  CHECK(kv.data != nullptr);
}

TEST_CASE("runner: full-attention-only step skips GDN metadata build (no OOB)") {
  const HfConfig c = MakeDenseOnlyConfig();
  const Qwen3_5DenseWeights w = MakeDenseOnlyWeights(c);
  GPUModelRunner runner(c, w, MakeFaOnlyKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  const std::vector<int32_t> prompt = {5, 9, 2, 31, 17};
  const int P = static_cast<int>(prompt.size());
  SchedulerOutput s1 =
      NewStep({MakeFaNewReq("A", prompt, 0, {0, 1}, Greedy())}, {{"A", P}});

  REQUIRE(runner.attn_kv().size() == static_cast<size_t>(c.num_hidden_layers));
  std::vector<std::vector<uint8_t>> kv_before;
  for (const PagedKvCache& kv : runner.attn_kv()) {
    const size_t block_bytes = static_cast<size_t>(
        2 * kv.block_size * kv.num_kv_heads * kv.head_size) * vt::SizeOf(kv.dtype);
    const auto* bytes = static_cast<const uint8_t*>(kv.data);
    kv_before.emplace_back(bytes, bytes + block_bytes);
  }

  // #3098 removes the borrowed dense forward's former GDN requirement. The
  // unchanged full-attention fixture must now finish with real attention state
  // and sampled-token feedback, while the runner leaves GDN metadata empty.
  REQUIRE_FALSE(runner.execute_model(s1).has_value());
  CHECK(runner.gdn_group_id() == -1);
  CHECK(runner.gdn_state().empty());
  CHECK(runner.last_gdn_meta().num_actual_tokens == 0);
  CHECK(runner.last_step().input_token_ids == prompt);
  CHECK(runner.last_forward_rows() == 1);
  CHECK(runner.last_forward_num_actual_tokens() == P);
  CHECK(runner.last_forward_num_reqs() == 1);
  for (size_t layer = 0; layer < runner.attn_kv().size(); ++layer) {
    CAPTURE(layer);
    const auto* bytes = static_cast<const uint8_t*>(runner.attn_kv()[layer].data);
    const std::vector<uint8_t> kv_after(bytes, bytes + kv_before[layer].size());
    CHECK(kv_after != kv_before[layer]);
  }

  const ModelRunnerOutput output = runner.sample_tokens(std::nullopt);
  REQUIRE(output.req_ids == std::vector<std::string>{"A"});
  REQUIRE(output.sampled_token_ids.size() == 1);
  REQUIRE(output.sampled_token_ids[0].size() == 1);
  const int32_t token = output.sampled_token_ids[0][0];
  CHECK(token >= 0);
  CHECK(token < c.vocab_size);
  const auto& batch = runner.input_batch();
  const int row = batch.req_id_to_index.at("A");
  CHECK(batch.num_tokens_no_spec[static_cast<size_t>(row)] == P + 1);
  CHECK(batch.token_id(row, P) == token);
}

// ─── M3: THE BLOCK-SIZE CONTRACT AT ITS PRODUCTION CALL SITE ─────────────────
//
// The runner must refuse a non-multiple-of-16 block size at its production
// call site — the same failure the server's --block-size validation and the
// bench rounding exist to prevent at the entry points.
//
// WHICH guard fires, and why this case says `runtime_error` (#1608). The
// refusal happens during backend SELECTION, before any backend is constructed:
// no candidate declares support for block_size 8, so
// `SelectAttentionBackendName` exhausts the platform's priority list and throws
// `std::runtime_error` naming every rejected candidate with its reason
// ("block_size not supported"). `get_kv_cache_shape`'s own
// `std::invalid_argument` sits BEHIND that and is not reached from here.
//
// This case previously asserted that `invalid_argument`, which made it
// host-dependent and red on every CPU build. The cause was not the assertion
// but `RocmAttentionBackend`: it omitted upstream's
// `get_supported_kernel_block_sizes() == MultipleOf(16)`
// (rocm_attn.py:181-190), inheriting the base MultipleOf(1), so on ROCm alone
// the registry ACCEPTED block_size 8 and the failure surfaced later out of
// `get_kv_cache_shape`. Declaring it made every host agree.
//
// OWED, and deliberately not papered over: with the registry consistent, the
// `CheckKvCacheShape` install inside `initialize_kv_cache` is no longer
// reachable from here, so deleting that install again leaves this case green —
// the #1065 Owed item this case was written for. Restoring it needs inputs that
// pass selection and then fail the shape comparison, i.e. an MLA spec with
// `num_kv_heads != 1` (TritonMLABackend refuses that in `get_kv_cache_shape`).
// A CPU build cannot express it: `CpuPlatform::get_attn_backend_priority`
// returns {CPU_ATTN, FLASH_ATTN} and registers no MLA backend, so this needs a
// CUDA-capable host or a test-only backend registration.
TEST_CASE("runner: initialize_kv_cache refuses a non-multiple-of-16 block size") {
  const HfConfig c = MakeDenseOnlyConfig();
  const Qwen3_5DenseWeights w = MakeDenseOnlyWeights(c);

  KVCacheConfig kv = MakeFaOnlyKvConfig(c);
  kv.kv_cache_groups[0].kv_cache_spec = std::make_shared<FullAttentionSpec>(
      /*block_size=*/8, static_cast<int>(c.num_key_value_heads),
      static_cast<int>(c.head_dim), vllm::v1::ResolveKvCacheDType());

  auto make_runner = [&]() {
    GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                          /*max_num_batched_tokens=*/64);
  };
  CHECK_THROWS_WITH_AS(make_runner(),
                       doctest::Contains("block_size not supported"),
                       std::runtime_error);
}

// ─── KV-DSV4-MULTICACHE W2 (#1973) — the runner refuses what it cannot carry ──
//
// The selection loop above this comment's subject (`runner.cpp`, the
// full_attn/gdn resolution) has exactly two arms and no `else`, and the
// allocation loop keys on `!is_gdn` when the model has no recurrent group. So
// before this row a published group of any other kind produced NO buffer and NO
// message. MEASURED on the pre-fix binary with a throwaway probe over
// `MakeFaOnlyKvConfig` plus a `kSlidingWindowMla` group and a second
// `kMlaAttention` group: the runner CONSTRUCTED, reported
// `full_attn_group_id = 0`, `gdn_group_id = -1`, `attn_kv().size() = 4` (one
// buffer per HIDDEN LAYER, all sized from group 0's `fa_page_size_bytes = 1024`)
// and allocated 0 bytes for the group whose own `page_size_bytes()` is 37440.
// A silently short KV allocation is a wrong-tokens failure, not a crash.
//
// These cases are the gate for the refusal that replaces that silence, and the
// tolerated-shape case beside them is its byte-neutrality contract: every model
// shipping today publishes exactly the groups this runner consumes.
//
// KV-DSV4-MULTICACHE W3 (#2068) NARROWS this refusal, and the two subcases W2
// wrote for a `kSlidingWindowMla` group and a second `kMlaAttention` group have
// MOVED rather than been deleted: those two shapes are now ALLOCATED (see
// "a multi-cache topology allocates EVERY published cache" below), and this case
// keeps the three shapes W3 still cannot represent — an unresolvable group name,
// a SECOND recurrent group, and a spec that is neither an AttentionSpec nor a
// MambaSpec. The refusal is kept rather than deleted because each of those is
// reachable from any future registry and a short KV allocation is wrong tokens
// rather than a crash.
TEST_CASE("runner: a published KV group it cannot allocate is REFUSED by name") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  // One argument, so the doctest macros below do not split on the ctor commas.
  const auto construct = [&](const KVCacheConfig& kvc) {
    GPUModelRunner runner(c, w, kvc, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                          /*max_num_batched_tokens=*/64);
    (void)runner;
  };
  const auto refusal_message = [&](const KVCacheConfig& kvc) {
    try {
      construct(kvc);
    } catch (const std::runtime_error& e) {
      return std::string(e.what());
    }
    return std::string("<did not throw>");
  };

  // W3: the exact spec class DeepSeek-V4 publishes for 105 of its 167 entries is
  // now CARRIED, not refused — and a group whose names carry no layer identity
  // still is, because a partially-addressable topology is the silent-wrong-answer
  // shape this whole row exists to remove.
  SUBCASE("a kSlidingWindowMla group with UNRESOLVABLE names is refused") {
    KVCacheConfig kv = MakeFaOnlyKvConfig(c);
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"swa_cache"},  // no `.layers.<N>.` segment
        std::make_shared<vllm::v1::SlidingWindowMLASpec>(
            /*block_size=*/64, /*num_kv_heads=*/1, /*head_size=*/512,
            DType::kI8, /*sliding_window=*/128,
            /*cache_dtype_str=*/std::string("fp8_ds_mla"), /*alignment=*/576,
            /*compress_ratio=*/1,
            /*model_version=*/std::string("deepseek_v4")));
    CHECK_THROWS_AS(construct(kv), std::runtime_error);
    const std::string msg = refusal_message(kv);
    // It names HOW MANY, WHICH KIND, WHICH LAYER, WHAT IT WOULD HAVE COST and
    // now WHY.
    // TWO, not one: on the multi-cache path EVERY cache is addressed by name,
    // so the TARGET group has to be named too — and `MakeFaOnlyKvConfig`
    // publishes the placeholder `"fa"`. That is the correct answer and it is
    // asserted rather than worked around.
    CHECK(msg.find("2 published KV cache group(s)") != std::string::npos);
    CHECK(msg.find("group 0 kind=kFullAttention") != std::string::npos);
    CHECK(msg.find("group 1") != std::string::npos);
    CHECK(msg.find("kSlidingWindowMla") != std::string::npos);
    CHECK(msg.find("swa_cache") != std::string::npos);
    CHECK(msg.find("page_size_bytes=37440") != std::string::npos);
    CHECK(msg.find("do not all resolve") != std::string::npos);
  }

  // A SECOND recurrent group. The multi-cache path carries any number of
  // attention groups and exactly one MambaSpec group, because the recurrent
  // state is indexed per SEQUENCE SLOT rather than per block and a second one
  // would need its own slot pool.
  SUBCASE("a SECOND recurrent group is named and refused") {
    KVCacheConfig kv = MakeKvConfig(c);
    kv.kv_cache_groups[1] = vllm::v1::KVCacheGroupSpec(
        std::vector<std::string>{"model.layers.0.mixer"},
        kv.kv_cache_groups[1].kv_cache_spec);
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"model.layers.1.mixer"},
        kv.kv_cache_groups[1].kv_cache_spec);
    CHECK_THROWS_AS(construct(kv), std::runtime_error);
    CHECK(refusal_message(kv).find("a SECOND recurrent group") !=
          std::string::npos);
  }

  // A plain SlidingWindowSpec and a ChunkedLocalAttentionSpec matched no arm
  // either. No registry builds one today, which is exactly why nobody noticed.
  SUBCASE("a kSlidingWindow group is named and refused") {
    KVCacheConfig kv = MakeFaOnlyKvConfig(c);
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"swa0"},
        std::make_shared<vllm::v1::SlidingWindowSpec>(
            kBlockSize, /*num_kv_heads=*/2, /*head_size=*/8, DType::kBF16,
            /*sliding_window=*/4));
    CHECK_THROWS_AS(construct(kv), std::runtime_error);
    CHECK(refusal_message(kv).find("kSlidingWindow ") != std::string::npos);
  }
  SUBCASE("a kChunkedLocalAttention group is named and refused") {
    KVCacheConfig kv = MakeFaOnlyKvConfig(c);
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"chunk0"},
        std::make_shared<vllm::v1::ChunkedLocalAttentionSpec>(
            kBlockSize, /*num_kv_heads=*/2, /*head_size=*/8, DType::kBF16,
            /*attention_chunk_size=*/4));
    CHECK_THROWS_AS(construct(kv), std::runtime_error);
    CHECK(refusal_message(kv).find("kChunkedLocalAttention") !=
          std::string::npos);
  }

  // EVERY unallocated group is named, not just the first — a refusal that
  // stopped at the first one would understate a seven-group topology as one.
  SUBCASE("all unallocated groups are named together") {
    KVCacheConfig kv = MakeFaOnlyKvConfig(c);
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"swa_cache"},
        std::make_shared<vllm::v1::SlidingWindowMLASpec>(
            /*block_size=*/64, /*num_kv_heads=*/1, /*head_size=*/512,
            DType::kI8, /*sliding_window=*/128,
            /*cache_dtype_str=*/std::string("fp8_ds_mla"), /*alignment=*/576,
            /*compress_ratio=*/1,
            /*model_version=*/std::string("deepseek_v4")));
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"mla2"},
        std::make_shared<vllm::v1::MLAAttentionSpec>(kBlockSize, /*head_size=*/512,
                                                     DType::kI8));
    const std::string msg = refusal_message(kv);
    CHECK(msg.find("3 published KV cache group(s)") != std::string::npos);
    CHECK(msg.find("kFullAttention") != std::string::npos);
    CHECK(msg.find("kSlidingWindowMla") != std::string::npos);
    CHECK(msg.find("kMlaAttention") != std::string::npos);
  }

  // AN EAGLE GROUP, and the fourth shape. `attn_group_ids_` excludes an eagle
  // group by construction (the draft KV is allocated by its own block, not by
  // the generalized loop), so on a multi-cache topology such a group would pass
  // every other arm of this refusal and then receive NO buffer -- the runner
  // would allocate NINE of the ten published caches and say nothing, which is
  // exactly the "SUBSET of the published topology in silence" this refusal
  // exists to remove (#2084). Nothing sets `is_eagle_group` outside this file
  // today, which is why it survived W2 and W3's first pass.
  SUBCASE("an eagle group on a multi-cache topology is named and refused") {
    KVCacheConfig kv = MakeMultiCacheKvConfig();
    // The indexer-key group, an `MLAAttentionSpec` the multi-cache path
    // otherwise allocates, re-published as a draft group.
    kv.kv_cache_groups[2] = vllm::v1::KVCacheGroupSpec(
        kv.kv_cache_groups[2].layer_names, kv.kv_cache_groups[2].kv_cache_spec,
        /*is_eagle_group=*/true);
    CHECK_THROWS_AS(construct(kv), std::runtime_error);
    const std::string msg = refusal_message(kv);
    CHECK(msg.find("1 published KV cache group(s)") != std::string::npos);
    CHECK(msg.find("group 2 kind=kMlaAttention") != std::string::npos);
    CHECK(msg.find("model.layers.2.attn.indexer.k_cache") != std::string::npos);
    CHECK(msg.find("an EAGLE draft group") != std::string::npos);
    // The other nine are untouched: this refuses the topology, it does not
    // reclassify the groups the path does carry.
    CHECK(msg.find("group 0 ") == std::string::npos);
    CHECK(msg.find("group 3 ") == std::string::npos);
  }
}

// BYTE-NEUTRALITY. The four group shapes every model in the tree publishes
// today still construct, so the refusal above cannot fire for any of them. The
// draft slot is tolerated on its KIND rather than on `spec_on()`, mirroring the
// draft-KV allocation loop's own predicate; that is deliberate and is listed
// under `## Owed` against W3 in `.agents/specs/kv-dsv4-multicache.md`.
TEST_CASE("runner: the group shapes shipped today still construct") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const int Hkv = static_cast<int>(c.num_key_value_heads);
  const int Dh = static_cast<int>(c.head_dim);

  SUBCASE("one full-attention group (dense Qwen3)") {
    GPUModelRunner runner(c, w, MakeFaOnlyKvConfig(c), Q(), 8, kMaxModelLen, 64);
    CHECK(runner.full_attn_group_id() == 0);
    CHECK(runner.gdn_group_id() == -1);
  }
  SUBCASE("one MLA group (every MLA model in the tree)") {
    KVCacheConfig kv;
    kv.num_blocks = kNumBlocks;
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"mla"},
        std::make_shared<vllm::v1::MLAAttentionSpec>(
            kBlockSize, /*head_size=*/576, vllm::v1::ResolveKvCacheDType()));
    GPUModelRunner runner(c, w, kv, Q(), 8, kMaxModelLen, 64);
    CHECK(runner.full_attn_group_id() == 0);
  }
  SUBCASE("full-attention + recurrent (the hybrid gate models)") {
    GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);
    CHECK(runner.full_attn_group_id() == 0);
    CHECK(runner.gdn_group_id() == 1);
  }
  SUBCASE("full-attention + recurrent + fa_draft (num_spec>0)") {
    KVCacheConfig kv = MakeKvConfig(c);
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"fa_draft"},
        std::make_shared<FullAttentionSpec>(kBlockSize, Hkv, Dh,
                                            vllm::v1::ResolveKvCacheDType()));
    GPUModelRunner runner(c, w, kv, Q(), 8, kMaxModelLen, 64);
    CHECK(runner.full_attn_group_id() == 0);
    CHECK(runner.gdn_group_id() == 1);
  }
}

// ─── KV-DSV4-MULTICACHE W3 (#2068) — every published cache gets a buffer ─────
//
// THE PRODUCTION SEAM. `GPUModelRunner`'s constructor is the one `LoadedEngine`
// calls, and `initialize_kv_cache` is private, so entering here is entering the
// same function an engine enters. The end-to-end case at the bottom of this file
// adds the other half: the KV config comes from `reg.factory->make_kv_cache`,
// the pointer `MakeKVCacheResolved` dereferences.
TEST_CASE("runner: a multi-cache topology allocates EVERY published cache") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const KVCacheConfig kv = MakeMultiCacheKvConfig();
  GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  // TEN caches, not four. Four is what one buffer per HIDDEN LAYER produces, and
  // it is what this runner produced before W3 while reporting nothing.
  REQUIRE(runner.attn_kv().size() == 10);
  REQUIRE(c.num_hidden_layers == 4);

  // Each entry carries ITS OWN group's geometry and page, in publication order.
  // The page-size literals are W1's table, not this file's arithmetic.
  struct Expect {
    const char* name;
    int64_t block_size;
    int64_t head_size;
    DType dtype;
    int64_t page;
    int32_t group;
    int32_t layer;
  };
  const std::vector<Expect> want = {
      {"model.layers.2.attn", 256, 512, DType::kI8, 37440, 0, 2},
      {"model.layers.3.attn", 256, 512, DType::kI8, 1728, 1, 3},
      {"model.layers.2.attn.indexer.k_cache", 256, 132, DType::kI8, 8640, 2, 2},
      {"model.layers.0.attn.swa_cache", 64, 512, DType::kI8, 37440, 3, 0},
      {"model.layers.1.attn.swa_cache", 64, 512, DType::kI8, 37440, 3, 1},
      {"model.layers.2.attn.swa_cache", 64, 512, DType::kI8, 37440, 3, 2},
      {"model.layers.3.attn.swa_cache", 64, 512, DType::kI8, 37440, 3, 3},
      {"model.layers.2.attn.compressor.state_cache", 4, 2048, DType::kF32,
       32832, 4, 2},
      {"model.layers.2.attn.indexer.compressor.state_cache", 4, 512,
       DType::kF32, 8640, 5, 2},
      {"model.layers.3.attn.compressor.state_cache", 8, 1024, DType::kF32,
       32832, 6, 3},
  };
  REQUIRE(runner.attn_kv_layer_names().size() == want.size());
  int64_t total_pages = 0;
  for (size_t i = 0; i < want.size(); ++i) {
    CAPTURE(i);
    CHECK(runner.attn_kv_layer_names()[i] == want[i].name);
    CHECK(runner.attn_kv()[i].block_size == want[i].block_size);
    CHECK(runner.attn_kv()[i].num_kv_heads == 1);
    CHECK(runner.attn_kv()[i].head_size == want[i].head_size);
    CHECK(runner.attn_kv()[i].dtype == want[i].dtype);
    CHECK(runner.attn_kv()[i].num_blocks == kNumBlocks);
    CHECK(runner.attn_kv()[i].data != nullptr);
    // KV-DSV4-MULTICACHE W8 slice 4 (#2455, and the half of #2085 that is
    // expressible today): the entry carries ITS OWN allocated page, per entry
    // rather than only in the sum below.
    //
    // THE SUM CANNOT SEE A SWAP. `total_pages` stays 271872 if two entries
    // exchange their pages, and it stays correct if every entry is handed the
    // group's first page, so a carrier wired to the wrong spec passes it. This
    // is also the assertion that separates the page from the VIEW: entry 0 is
    // `block_size` 256 x `head_size` 512 = 131072 bytes of rank-3 geometry over
    // a 37440-byte page, and a store that believed the view would run 3.5x past
    // the block it was given.
    CHECK(runner.attn_kv()[i].page_size_bytes == want[i].page);
    CHECK(runner.multi_kv_index().Find(want[i].name) ==
          static_cast<int64_t>(i));
    total_pages += want[i].page;
  }
  // A cache that is not published is not found.
  CHECK(runner.multi_kv_index().Find("model.layers.0.attn") == -1);

  // The BYTES, summed over every buffer the runner created. 8 blocks x the ten
  // pages above.
  CHECK(runner.kv_cache_allocated_paged_bytes() == kNumBlocks * total_pages);
  CHECK(runner.kv_cache_allocated_bytes() == kNumBlocks * total_pages);
  CHECK(total_pages == 271872);

  // Per-layer routing. A count cannot see a routing inversion, so this asserts
  // WHICH caches each layer got, not how many.
  using LKC = GPUModelRunner::LayerKvClass;
  for (int64_t l = 0; l < 4; ++l) {
    CAPTURE(l);
    CHECK(runner.layer_kv_class()[static_cast<size_t>(l)] == LKC::kMultiCache);
  }
  REQUIRE(runner.layer_attn_kv_indices().size() == 4);
  CHECK(runner.layer_attn_kv_indices()[0] == std::vector<int32_t>{3});
  CHECK(runner.layer_attn_kv_indices()[1] == std::vector<int32_t>{4});
  CHECK(runner.layer_attn_kv_indices()[2] ==
        std::vector<int32_t>{0, 2, 5, 7, 8});
  CHECK(runner.layer_attn_kv_indices()[3] == std::vector<int32_t>{1, 6, 9});

  // The generalized group ids. `full_attn_group_id()` keeps its old meaning —
  // the FIRST non-eagle full-attention/MLA group — and is still readable.
  CHECK(runner.attn_group_ids() == std::vector<int>{0, 1, 2, 3, 4, 5, 6});
  CHECK(runner.recurrent_group_ids().empty());
  CHECK(runner.full_attn_group_id() == 0);
  CHECK(runner.gdn_group_id() == -1);
  CHECK(runner.fa_page_size_bytes() == 37440);
}

// A multi-cache topology that also carries a recurrent group. Nothing in the
// tree publishes this shape; it is gated because the generalization would
// otherwise be a hole the next hybrid falls into, and because "supported" is a
// claim that needs a test rather than a comment.
TEST_CASE("runner: a multi-cache topology keeps its recurrent group") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  KVCacheConfig kv = MakeMultiCacheKvConfig();
  // Layers 0 and 1 become recurrent, so their SWA caches must go away with them.
  kv.kv_cache_groups[3] = vllm::v1::KVCacheGroupSpec(
      std::vector<std::string>{"model.layers.2.attn.swa_cache",
                               "model.layers.3.attn.swa_cache"},
      V4Sliding(64, 512, DType::kI8, 128, true));
  const int Hv = static_cast<int>(c.linear_num_value_heads);
  const int Dv = static_cast<int>(c.linear_value_head_dim);
  const int Dk = static_cast<int>(c.linear_key_head_dim);
  const int Kw = static_cast<int>(c.linear_conv_kernel_dim);
  const int conv_dim =
      2 * static_cast<int>(c.linear_num_key_heads) * Dk + Hv * Dv;
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.0.mixer", "model.layers.1.mixer"},
      std::make_shared<MambaSpec>(
          kMaxModelLen,
          std::vector<std::vector<int64_t>>{{conv_dim, Kw - 1}, {Hv, Dv, Dk}},
          std::vector<DType>{DType::kF32, DType::kF32}));
  GPUModelRunner runner(c, w, kv, Q(), 8, kMaxModelLen, 64);

  using LKC = GPUModelRunner::LayerKvClass;
  CHECK(runner.layer_kv_class()[0] == LKC::kRecurrent);
  CHECK(runner.layer_kv_class()[1] == LKC::kRecurrent);
  CHECK(runner.layer_kv_class()[2] == LKC::kMultiCache);
  CHECK(runner.layer_kv_class()[3] == LKC::kMultiCache);
  CHECK(runner.attn_kv().size() == 8);  // 10 minus the two dropped SWA entries
  CHECK(runner.gdn_group_id() == 7);
  CHECK(runner.recurrent_group_ids() == std::vector<int>{7});
  CHECK(runner.layer_attn_kv_indices()[0].empty());
  CHECK(runner.layer_attn_kv_indices()[2].size() == 5);
}

// ─── A qwen4_exp-SHAPED topology: THREE groups, and FOUR recurrent states ────
//
// This closes two `## Owed` items in `.agents/specs/recurrent-multistate.md`
// at once, and it is the first fixture in the tree to do either:
//
//   * "The multi-cache recurrent allocation site is UNEXERCISED." The
//     `alloc_recurrent_layer_states` call inside `if (multi_cache_topology)`,
//     in its `membership_by_name && has_mamba_group` recurrent loop, could be
//     DELETED with all four recurrent suites green. The case above
//     ("keeps its recurrent group") asserts CLASSIFICATION — `layer_kv_class_`,
//     `gdn_group_id_`, the per-layer index lists — and every one of those is
//     computed before the allocation, so it cannot see the allocation go away.
//     This case asserts what was ALLOCATED.
//   * "Nothing publishes N >= 3." Every recurrent registry in the tree
//     published two states, so the N-general arm landed EXPRESSIBLE and
//     UNREACHED. `MakeQwen4ExpKVCache` (W5c, #2031) publishes FOUR, and this
//     is that shape at four layers instead of forty-eight.
//
// The miniature is `Qwen4ExpForConditionalGeneration`'s geometry: a 3:1
// linear:sparse schedule, one paged K+V group over the sparse layers, ONE
// uniform recurrent group over every linear layer carrying
// [gdn_conv, temporal, ple_conv, ngram], and the QSA indexer side cache as an
// `MLAAttentionSpec` at compress_ratio 4 — which is what makes the topology
// multi-cache at all. A `FullAttentionSpec` there would be absorbed as the
// single `fa_draft` draft-KV slot instead and get no buffer.
//
// The four state sizes are pairwise DISTINCT in element count, and the set
// covers three ranks and two dtypes, so no implementation that reuses
// `shapes[0]`, `shapes[1]`, `dtypes[0]` or a factor of 2 can produce them:
//   gdn_conv  {64, 3}     bf16 ->  384 B/slot   rank 2
//   temporal  {4, 8, 8}   f32  -> 1024 B/slot   rank 3
//   ple_conv  {128, 9}    bf16 -> 2304 B/slot   rank 2
//   ngram     {2}         i64  ->   16 B/slot   rank 1
namespace {

constexpr int64_t kQ4ConvElems = 64 * 3;
constexpr int64_t kQ4SsmElems = 4 * 8 * 8;
constexpr int64_t kQ4PleElems = 128 * 9;
constexpr int64_t kQ4NGramElems = 2;
constexpr int64_t kQ4MambaPage =
    kQ4ConvElems * 2 + kQ4SsmElems * 4 + kQ4PleElems * 2 + kQ4NGramElems * 8;

KVCacheConfig MakeQwen4ExpShapedKvConfig() {
  KVCacheConfig kv;
  kv.num_blocks = kNumBlocks;
  // Group 0 — the sparse layer's paged K+V. Layer 3 is the `full_attention`
  // entry in MakeConfig()'s [LA, LA, LA, FA] schedule, which upstream's
  // `__post_init__` rewrites to `qwen_sparse_attention`.
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.3.self_attn.attn"},
      std::make_shared<FullAttentionSpec>(kBlockSize, /*num_kv_heads=*/2,
                                          /*head_size=*/8,
                                          vllm::v1::ResolveKvCacheDType()));
  // Group 1 — ONE uniform recurrent group over EVERY linear layer, carrying the
  // PLE states that only one of them uses. That is upstream's polarity, not a
  // shortcut: `get_mamba_state_shape_from_config` is a classmethod with no
  // `layer_idx` (`interfaces.py:809-812`) and `get_mamba_groups`
  // (`mamba_utils.py:441`) asserts every `MambaSpec` in the model equal.
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.0.linear_attn",
                               "model.layers.1.linear_attn",
                               "model.layers.2.linear_attn"},
      std::make_shared<MambaSpec>(
          kBlockSize,
          std::vector<std::vector<int64_t>>{{64, 3},
                                            {4, 8, 8},
                                            {128, 9},
                                            {kQ4NGramElems}},
          std::vector<DType>{DType::kBF16, DType::kF32, DType::kBF16,
                             DType::kI64}));
  // Group 2 — the QSA indexer side cache. One key vector per FOUR tokens, no V.
  kv.kv_cache_groups.emplace_back(
      std::vector<std::string>{"model.layers.3.self_attn.indexer.k_cache"},
      std::make_shared<vllm::v1::MLAAttentionSpec>(
          kBlockSize, /*head_size=*/8, vllm::v1::ResolveKvCacheDType(),
          /*num_kv_heads=*/1, vllm::v1::KVQuantMode::kNone,
          /*page_size_padded=*/std::nullopt,
          /*indexes_kv_by_block_stride=*/false,
          /*cache_dtype_str=*/std::nullopt, /*alignment=*/std::nullopt,
          /*compress_ratio=*/4, /*model_version=*/std::nullopt));
  return kv;
}

}  // namespace

TEST_CASE("runner: a multi-cache topology ALLOCATES its N-state recurrent group") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const KVCacheConfig kv = MakeQwen4ExpShapedKvConfig();
  REQUIRE(vllm::v1::ResolveKvCacheDType() == DType::kBF16);

  const auto* mamba =
      dynamic_cast<const MambaSpec*>(kv.kv_cache_groups[1].kv_cache_spec.get());
  REQUIRE(mamba != nullptr);
  REQUIRE(mamba->shapes.size() == 4);
  REQUIRE(mamba->page_size_bytes() == kQ4MambaPage);
  REQUIRE(kQ4MambaPage == 3728);

  GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);
  const int64_t slots = runner.gdn_state_slots();
  REQUIRE(slots == 8);

  // 1. THE TOPOLOGY IS THE MULTI-CACHE ONE. Asserted rather than assumed,
  //    because if it were not, the recurrent buffers below would come from the
  //    LEGACY `is_gdn` call site and this case would gate the site that already
  //    had four suites on it.
  CHECK(runner.attn_group_ids() == std::vector<int>{0, 2});
  CHECK(runner.recurrent_group_ids() == std::vector<int>{1});
  CHECK(runner.gdn_group_id() == 1);
  CHECK(runner.full_attn_group_id() == 0);
  using LKC = GPUModelRunner::LayerKvClass;
  CHECK(runner.layer_kv_class()[0] == LKC::kRecurrent);
  CHECK(runner.layer_kv_class()[1] == LKC::kRecurrent);
  CHECK(runner.layer_kv_class()[2] == LKC::kRecurrent);
  CHECK(runner.layer_kv_class()[3] == LKC::kMultiCache);
  // Layer 3 owns TWO caches: its paged K+V and its indexer side cache.
  REQUIRE(runner.layer_attn_kv_indices().size() == 4);
  CHECK(runner.layer_attn_kv_indices()[3] == std::vector<int32_t>{0, 1});
  CHECK(runner.attn_kv_layer_names() ==
        std::vector<std::string>{"model.layers.3.self_attn.attn",
                                 "model.layers.3.self_attn.indexer.k_cache"});

  // 2. THE RECURRENT ALLOCATION HAPPENED, on the multi-cache path. This is the
  //    assertion the deletion mutation reds: with that call site removed,
  //    `recurrent_state_buf_` stays empty, so `gdn_state_` is empty too.
  REQUIRE(runner.gdn_state().size() == 3);

  for (const GdnStateCache& gs : runner.gdn_state()) {
    REQUIRE(gs.states.size() == 4);
    // The two legacy NAMES are still slots 0 and 1, which is why the temporal
    // state sits between the conv states rather than after them.
    CHECK(gs.states[0].data == gs.conv_state.data);
    CHECK(gs.states[1].data == gs.ssm_state.data);
    // Each state carries its OWN rank, shape and dtype, slot dim prepended.
    CHECK(gs.states[0].rank == 3);
    CHECK(gs.states[1].rank == 4);
    CHECK(gs.states[2].rank == 3);
    CHECK(gs.states[3].rank == 2);
    CHECK(gs.states[0].dtype == DType::kBF16);
    CHECK(gs.states[1].dtype == DType::kF32);
    CHECK(gs.states[2].dtype == DType::kBF16);
    CHECK(gs.states[3].dtype == DType::kI64);
    CHECK(std::vector<int64_t>{gs.states[2].shape[0], gs.states[2].shape[1],
                               gs.states[2].shape[2]} ==
          std::vector<int64_t>{slots, 128, 9});
    CHECK(std::vector<int64_t>{gs.states[3].shape[0], gs.states[3].shape[1]} ==
          std::vector<int64_t>{slots, kQ4NGramElems});
    // Four DISTINCT allocations, none an alias or a re-view of another.
    for (size_t i = 0; i < 4; ++i) {
      CAPTURE(i);
      CHECK(gs.states[i].data != nullptr);
      for (size_t j = i + 1; j < 4; ++j) CHECK(gs.states[i].data != gs.states[j].data);
    }
    // 3. STATES 2 AND 3 ARE LOAD-BEARING. Their byte counts are their OWN
    //    element count times their OWN element size, and both differ from
    //    everything slots 0 and 1 could supply: reusing `shapes[1]` for the PLE
    //    conv gives 8192 rather than 18432, and reusing `dtypes[0]` for the
    //    n-gram history gives 32 rather than 128.
    CHECK(static_cast<int64_t>(gs.states[2].Bytes()) == slots * kQ4PleElems * 2);
    CHECK(static_cast<int64_t>(gs.states[2].Bytes()) == 18432);
    CHECK(static_cast<int64_t>(gs.states[3].Bytes()) ==
          slots * kQ4NGramElems * 8);
    CHECK(static_cast<int64_t>(gs.states[3].Bytes()) == 128);
  }

  // 4. BYTE IDENTITY between what the runner took and what it reports, and
  //    between that and what the ENGINE's own budget charges for this group.
  //    A state the allocator skipped or the reporter missed shows up here
  //    rather than as a short cache nothing mentions.
  int64_t recurrent_bytes = 0;
  for (const GdnStateCache& gs : runner.gdn_state())
    for (const vt::Tensor& s : gs.states)
      recurrent_bytes += static_cast<int64_t>(s.Bytes());
  CHECK(recurrent_bytes == 3 * slots * kQ4MambaPage);
  CHECK(recurrent_bytes == 89472);
  CHECK(runner.kv_cache_allocated_bytes() -
            runner.kv_cache_allocated_paged_bytes() == recurrent_bytes);
  CHECK(vllm::v1::recurrent_state_bytes(kv, /*max_num_seqs=*/8) ==
        recurrent_bytes);

  // 5. And the paged half, from each group's OWN page: 16*2*(8+8)*2 = 1024 for
  //    the K+V group, and (16/4)*1*8*2 = 64 for the compress-ratio-4 side
  //    cache, over 8 blocks.
  CHECK(runner.attn_kv().size() == 2);
  CHECK(runner.kv_cache_allocated_paged_bytes() == kNumBlocks * (1024 + 64));
  CHECK(runner.kv_cache_allocated_bytes() == 8704 + 89472);
}

// ─── ENG-MULTIKV-BYNAME — the by-name channel addresses RECURRENT caches too ──
//
// THE GAP, in one number. `MakeQwen4ExpShapedKvConfig` publishes THREE groups
// over four layers: one full-attention group (layer 3), one recurrent group
// (layers 0, 1, 2) and one MLA indexer side cache (layer 3). Five caches. The
// channel reported TWO, because `attn_kv_layer_names_` is built from
// `attn_group_ids_`, which collects only `AttentionSpec` groups
// (`runner.cpp`, the group-classification loop), and a recurrent state reaches
// the forward through `gdn_state` POSITIONALLY with no name attached. Issue
// #2343 measured the same shape at full size: `22 KV cache(s) from 2 published
// group(s)` beside `block tables gathered for 3 of 3` — 34 recurrent states
// invisible while their group's block table was not.
//
// UPSTREAM HAS NO SUCH SPLIT. `_reshape_kv_cache_tensors` puts the Mamba page
// into the SAME `kv_caches: dict[str, torch.Tensor]` as every attention cache
// (`vllm/v1/worker/gpu_model_runner.py:7354` declares it, `:7418-7427` fills
// the attention arm and `:7429-7441` the `MambaSpec` arm, whose comment says
// "Keeping one tensor per layer lets the KV connector register it without
// special-casing Mamba"), `:7318-7326` asserts that dict's keys are EVERY layer
// name of EVERY published group, and `bind_kv_cache`
// (`vllm/v1/worker/utils.py:450-465`) binds out of that one dict. Read at pin
// 5559679229bc961848b121ccdeaa8fa5d79bec98.
//
// THE ORDER IS UPSTREAM'S INSERTION ORDER: published GROUP order, then the
// group's own layer-name order (`:7365-7372`, `for group in ...: for layer_name
// in group.layer_names`). That is what puts the three recurrent entries BETWEEN
// the two attention ones here, which is the case's non-identity property — the
// flat index of the indexer side cache is 4 while its slot in `attn_kv` is 1.
TEST_CASE("runner: the by-name KV channel addresses RECURRENT caches (#2343)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const KVCacheConfig kv = MakeQwen4ExpShapedKvConfig();
  GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  // The topology this case believes it is driving, asserted rather than assumed.
  REQUIRE(kv.kv_cache_groups.size() == 3);
  REQUIRE(runner.attn_group_ids() == std::vector<int>{0, 2});
  REQUIRE(runner.recurrent_group_ids() == std::vector<int>{1});
  REQUIRE(runner.attn_kv().size() == 2);
  REQUIRE(runner.gdn_state().size() == 3);

  const vllm::MultiKvCacheIndex& mk = runner.multi_kv_index();

  // 1. EVERY published cache is in the channel: 2 paged + 3 recurrent.
  CHECK(mk.size() == 5);
  // 2. And they came from THREE groups, which is the count the block-table
  //    denominator has always reported. Before this, `num_groups()` answered 2
  //    beside `num_published_groups()` 3, so a diagnostic reading both reported
  //    a full gather as partial.
  CHECK(mk.num_groups() == 3);
  CHECK(mk.num_published_groups() == 3);

  // 3. A recurrent layer RESOLVES BY NAME. Each of the three, not one: a
  //    single-name check cannot see an off-by-one in the slot mapping.
  CHECK(mk.Find("model.layers.0.linear_attn") >= 0);
  CHECK(mk.Find("model.layers.1.linear_attn") >= 0);
  CHECK(mk.Find("model.layers.2.linear_attn") >= 0);
  // 4. NON-IDENTITY, and the reason this fixture was chosen over the
  //    all-attention ones. In published-group order the recurrent group sits
  //    BETWEEN the two attention groups, so the indexer side cache's flat index
  //    is 4 while its slot in `attn_kv` is 1. A channel that merely appended
  //    the recurrent names would answer 2 here and pass every other line above.
  CHECK(mk.Find("model.layers.3.self_attn.attn") == 0);
  CHECK(mk.Find("model.layers.0.linear_attn") == 1);
  CHECK(mk.Find("model.layers.1.linear_attn") == 2);
  CHECK(mk.Find("model.layers.2.linear_attn") == 3);
  CHECK(mk.Find("model.layers.3.self_attn.indexer.k_cache") == 4);
  // 5. A name nobody published is still not found.
  CHECK(mk.Find("model.layers.3.linear_attn") == -1);

  // 6. THE LOCATOR. `Find` gives a place in the published list; `PayloadAt`
  //    gives the cache. Every one of the five, with its container, its slot,
  //    its group and its model layer — a count cannot see a routing inversion,
  //    and neither can a single spot check.
  using KP = vllm::KvCachePayload;
  struct Want {
    const char* name;
    KP kind;
    int32_t slot;
    int32_t group;
    int32_t layer;
  };
  const std::vector<Want> want = {
      {"model.layers.3.self_attn.attn", KP::kPaged, 0, 0, 3},
      {"model.layers.0.linear_attn", KP::kRecurrent, 0, 1, 0},
      {"model.layers.1.linear_attn", KP::kRecurrent, 1, 1, 1},
      {"model.layers.2.linear_attn", KP::kRecurrent, 2, 1, 2},
      {"model.layers.3.self_attn.indexer.k_cache", KP::kPaged, 1, 2, 3},
  };
  REQUIRE(mk.size() == want.size());
  for (size_t i = 0; i < want.size(); ++i) {
    CAPTURE(i);
    KP kind = KP::kPaged;
    int32_t slot = -7;
    REQUIRE(mk.Resolve(want[i].name, &kind, &slot));
    CHECK(kind == want[i].kind);
    CHECK(slot == want[i].slot);
    CHECK((*mk.group_ids)[i] == want[i].group);
    CHECK((*mk.layer_indices)[i] == want[i].layer);
    // The same answer through the two-step form, on the index `Find` returned.
    KP kind2 = KP::kPaged;
    int32_t slot2 = -7;
    REQUIRE(mk.PayloadAt(mk.Find(want[i].name), &kind2, &slot2));
    CHECK(kind2 == want[i].kind);
    CHECK(slot2 == want[i].slot);
  }
  CHECK(mk.num_paged() == 2);
  CHECK(mk.num_recurrent() == 3);
  CHECK(mk.num_paged() + mk.num_recurrent() == static_cast<int>(mk.size()));

  // 7. THE SLOT REACHES A REAL CACHE, which is the whole point of the row: a
  //    name that resolves to a slot nothing holds is the same silent hole in a
  //    different place. Each recurrent slot indexes a DISTINCT `gdn_state`
  //    entry with allocated states.
  std::vector<const void*> seen;
  for (const Want& wt : want) {
    CAPTURE(std::string(wt.name));
    KP kind = KP::kPaged;
    int32_t slot = -1;
    REQUIRE(mk.Resolve(wt.name, &kind, &slot));
    if (kind == KP::kRecurrent) {
      REQUIRE(slot >= 0);
      REQUIRE(static_cast<size_t>(slot) < runner.gdn_state().size());
      const GdnStateCache& gs = runner.gdn_state()[static_cast<size_t>(slot)];
      REQUIRE(gs.states.size() == 4);
      CHECK(gs.conv_state.data != nullptr);
      // No two names may resolve onto one state.
      for (const void* p : seen) CHECK(p != gs.conv_state.data);
      seen.push_back(gs.conv_state.data);
    } else {
      REQUIRE(slot >= 0);
      REQUIRE(static_cast<size_t>(slot) < runner.attn_kv().size());
      CHECK(runner.attn_kv()[static_cast<size_t>(slot)].data != nullptr);
    }
  }
  CHECK(seen.size() == 3);

  // 8. A name that was never published resolves to NOTHING, and writes the
  //    out-of-band values rather than leaving the caller's variables alone.
  //    An out-of-range index does the same.
  KP kind = KP::kRecurrent;
  int32_t slot = 99;
  CHECK_FALSE(mk.Resolve("model.layers.3.linear_attn", &kind, &slot));
  CHECK(kind == KP::kPaged);
  CHECK(slot == -1);
  kind = KP::kRecurrent;
  slot = 99;
  CHECK_FALSE(mk.PayloadAt(5, &kind, &slot));
  CHECK(slot == -1);
  CHECK_FALSE(mk.PayloadAt(-1, &kind, &slot));
  CHECK(slot == -1);
}

// ─── ENG-MULTIKV-BYNAME reaches the PRODUCTION forward seam ──────────────────
//
// `GPUModelRunner::execute_model` is what an engine calls and
// `ModelRegistry::Forward` is the shared decode seam AGENTS.md routes every
// forward through. The seam READS the channel to build its refusal, so the
// recurrent half is observable at a production entry point rather than only
// through a test accessor. That is what makes this row's reachability mutation
// non-vacuous: deleting the runner's publication of the recurrent entries
// changes bytes this case reads out of `ModelRegistry::Forward`.
//
// THE ASSERTION IS TWO-SIDED. A bare "it threw" is a mute switch here, because
// the runner has its OWN refusal for an unallocated group and it would satisfy
// any test that only checked that something was thrown. So the case asserts the
// forward refusal's identifying bytes PRESENT and the runner refusal's bytes
// ABSENT.
TEST_CASE("runner: the forward seam REPORTS the recurrent caches (#2343)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const KVCacheConfig kv = MakeQwen4ExpShapedKvConfig();
  GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  SchedulerOutput so;
  SamplingParams sp;
  sp.temperature = 0.0F;
  NewRequestData nr;
  nr.req_id = "r0";
  nr.prompt_token_ids = {1, 2, 3};
  nr.sampling_params = sp;
  // One block-table group per PUBLISHED group, non-identity and with no fixed
  // point, so a gather keyed on the wrong id cannot pass by coincidence.
  nr.block_ids = {{6, 2}, {4, 7}, {5, 3}};
  nr.num_computed_tokens = 0;
  nr.prefill_token_ids = {1, 2, 3};
  so.scheduled_new_reqs.push_back(std::move(nr));
  so.num_scheduled_tokens["r0"] = 3;
  so.total_num_scheduled_tokens = 3;

  std::string msg = "<did not throw>";
  try {
    (void)runner.execute_model(so);
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  CAPTURE(msg);
  // PRESENT: the forward seam's refusal, carrying the recurrent half.
  CHECK(msg.find("5 KV cache(s) (2 paged, 3 recurrent)") != std::string::npos);
  CHECK(msg.find("from 3 published group(s)") != std::string::npos);
  CHECK(msg.find("first 'model.layers.3.self_attn.attn'") != std::string::npos);
  // COMPOSITION NOTE (row/MODEL-MM-QWEN4-EXP-E2E). This read
  // `KV-DSV4-MULTICACHE W5 owns the consuming forward`, which is the wording
  // #2353 (`row/ENG-MULTIKV-FORWARD-1925`) deliberately REMOVED: three
  // architectures reach this guard and that row's W5 owns one of them. The two
  // branches merged textually clean and semantically contradictory, because
  // `test_runner.cpp:2662` asserts that exact substring ABSENT. The intent of
  // the assertion is preserved, not weakened: the message must still pin the
  // row that owns THE GUARD, and it must now name the ARRIVING architecture,
  // read from the same registry a different code path builds it from.
  CHECK(msg.find("KV-DSV4-MULTICACHE W3") != std::string::npos);
  CHECK(msg.find("KV-DSV4-MULTICACHE W5") == std::string::npos);
  const std::string arriving(
      vllm::RegistrationFor("Qwen3_5MoeForConditionalGeneration").architecture);
  REQUIRE_FALSE(arriving.empty());
  CHECK(msg.find(arriving) != std::string::npos);
  // ABSENT: the OTHER refusal this topology could plausibly have hit — the
  // runner's own "a published group got no cache". If that one fired instead,
  // every `find` above would still have been reachable by a test that only
  // asserted a throw.
  CHECK(msg.find("get NO cache from this runner") == std::string::npos);
  CHECK(msg.find("<did not throw>") == std::string::npos);
}

// ─── The third forward channel ──────────────────────────────────────────────
//
// `ModelRegistry::Forward` is the shared decode seam AGENTS.md routes every
// forward through, and `GPUModelRunner::execute_model` reaches it. It REFUSES a
// multi-cache index, because no registered forward consumes a cache set keyed by
// layer name yet. THREE architectures reach it and no ONE row owns all three
// consuming forwards (#2353), so the refusal names the ARRIVING architecture and
// leaves ownership to that architecture's row. The refusal reads the channel's
// PAYLOAD — how many caches, from how many groups, and the first name — so a
// channel that arrived empty produces a different message.
TEST_CASE("runner: a multi-cache forward is REFUSED, naming the channel") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeMultiCacheKvConfig(), Q(), 8, kMaxModelLen,
                        64);

  SchedulerOutput so;
  SamplingParams sp;
  sp.temperature = 0.0F;
  NewRequestData nr;
  nr.req_id = "r0";
  nr.prompt_token_ids = {1, 2, 3};
  nr.sampling_params = sp;
  // One block-table group per PUBLISHED group, which is what the block table
  // this runner built expects. One block each: a 3-token prompt claims one
  // block in every group, and group 0's row (block size 256 at max_model_len
  // 32) holds exactly one. The two ids this used to feed overran that row,
  // which `BlockTable::append_row` now refuses by name
  // (FIX-BLOCK-TABLE-ROW-WIDTH, gpu/block_table.py:125-131 @ e126687a9a).
  nr.block_ids.assign(7, std::vector<int>{1});
  nr.num_computed_tokens = 0;
  nr.prefill_token_ids = {1, 2, 3};
  so.scheduled_new_reqs.push_back(std::move(nr));
  so.num_scheduled_tokens["r0"] = 3;
  so.total_num_scheduled_tokens = 3;

  std::string msg = "<did not throw>";
  try {
    (void)runner.execute_model(so);
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  MESSAGE("EMITTED REFUSAL BYTES: [" << msg << "]");
  CHECK(msg.find("10 KV cache(s)") != std::string::npos);
  CHECK(msg.find("7 published group(s)") != std::string::npos);
  CHECK(msg.find("model.layers.2.attn") != std::string::npos);

  // ─── #2353 ────────────────────────────────────────────────────────────────
  //
  // THE ARRIVING ARCHITECTURE IS NAMED. Three architectures publish a
  // multi-cache topology today -- `DeepseekV4ForCausalLM` (7 groups),
  // `Qwen4ExpForConditionalGeneration` (3) and `Glm5NextForConditionalGeneration`
  // (3) -- and the message named none of them, so a reader who met it could not
  // tell which row owed the consuming forward. #2343 is the measured cost: the
  // guard fired on `dgx:gpu0` and four separate records had to reconstruct in
  // prose what the string should have said itself.
  //
  // THE EXPECTATION IS NOT A BARE LITERAL. `Qwen3_5MoeForConditionalGeneration`
  // is the architecture `BorrowQwen3_5MoeLoadedModel` registers this runner's
  // model under (`qwen3_5_moe.cpp:241-245`), and it is asserted through
  // `RegistrationFor`, which is the same registry the message reads and a
  // different code path from the one that builds it. The NEGATIVE assertion is
  // what stops a hard-coded architecture from passing: a message that named
  // `DeepseekV4ForCausalLM` for every arrival would satisfy the positive check
  // on a DeepSeek-V4 run and fail here.
  const std::string arch(
      vllm::RegistrationFor("Qwen3_5MoeForConditionalGeneration").architecture);
  REQUIRE_FALSE(arch.empty());
  CHECK(msg.find(arch) != std::string::npos);
  CHECK(msg.find("DeepseekV4ForCausalLM") == std::string::npos);

  // THE OWNERSHIP PIN MOVED, IT WAS NOT DELETED. It read
  // "KV-DSV4-MULTICACHE W5", which asserts that row's W5 owns the consuming
  // forward for whatever architecture arrives. That row's W5 is scoped in
  // `.agents/specs/kv-dsv4-multicache.md` as the DeepSeek-V4 DSA-sparse path
  // that removes `deepseek_v4.cpp`'s `(void)attn_kv`, so it is true for one of
  // the three arrivals and false for the other two. The string now names the
  // row that owns THE GUARD -- W3 landed it -- and leaves the consuming forward
  // to the arriving architecture's own row.
  CHECK(msg.find("KV-DSV4-MULTICACHE W3") != std::string::npos);
  CHECK(msg.find("KV-DSV4-MULTICACHE W5") == std::string::npos);
}

// ─── MODEL-MM-QWEN4-EXP W5c-2 (#2249 item 3) — EVERY published group gathered ─
//
// THE GAP THIS CLOSES, in the issue's own words: "`gather_block_table` reaches
// only `full_attn_group_id_` and `gdn_group_id_`, so the QSA indexer side cache
// is allocated and unread." A cache the runner allocates and whose block table
// never leaves the runner cannot be addressed by anything: the buffer is real,
// the pages are real, and the map from a sequence's LOGICAL position to the
// PHYSICAL page holding it stays inside `input_batch_`.
//
// UPSTREAM ANCHOR. `vllm/v1/worker/gpu_model_runner.py:2551-2567` @ pin
// 5559679229 loops over EVERY published group and gives each its own table:
//
//     for kv_cache_gid, kv_cache_group in enumerate(kv_cache_groups):
//         cm = copy(cm_base)
//         ...
//         if kv_cache_gid > 0:
//             cm.block_table_tensor = _get_block_table(kv_cache_gid)
//             cm.slot_mapping = slot_mappings[kv_cache_gid]
//
// with `_get_block_table` (`:2318-2334`) being exactly this tree's
// `gather_block_table`: `self.input_batch.block_table[gid].get_device_tensor()`.
// Upstream has no "the two named groups" shape at all — the count of groups it
// serves is `len(kv_cache_groups)`.
//
// THE TABLE IS DELIBERATELY PERMUTED AND HAS NO FIXED POINT. An identity map
// proves nothing: a body that ignores the table, returns the logical indices, or
// gathers the wrong group's table still agrees with `{0, 1}`. Each group gets a
// DISTINCT two-block list, none of which maps any logical index to itself, so
// gathering group 2 with `gdn_group_id_` reads `{4, 7}` where `{5, 3}` is owed
// and the case reds. The 20-token prompt spans two 16-token blocks with a
// PARTIAL final block (4 tokens), so a body that assumed a whole final page is
// visible here too.
//
// THIS DRIVES THE PRODUCTION ENTRY POINT. `GPUModelRunner::execute_model` is
// what an engine calls; the gather runs inside it, and the forward it reaches
// then REFUSES the multi-cache channel. The consuming forward is owed by the row
// that ports the ARRIVING architecture, which is what the refusal now names and
// what #2353 corrected: this comment said "KV-DSV4-MULTICACHE W5 owns the
// consuming forward", and that row's W5 is the DeepSeek-V4 DSA-sparse path
// alone. The refusal is caught and READ, because the count it reports is the
// production reader of the gathered tables — a value nothing reads is a value
// nothing can be wrong about.
TEST_CASE("runner: EVERY published KV group's block table is gathered (#2249)") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const KVCacheConfig kv = MakeQwen4ExpShapedKvConfig();
  GPUModelRunner runner(c, w, kv, Q(), /*max_num_reqs=*/8, kMaxModelLen,
                        /*max_num_batched_tokens=*/64);

  // The topology this case believes it is driving, asserted rather than assumed.
  REQUIRE(runner.full_attn_group_id() == 0);
  REQUIRE(runner.gdn_group_id() == 1);
  REQUIRE(runner.attn_group_ids() == std::vector<int>{0, 2});
  REQUIRE(kv.kv_cache_groups.size() == 3);

  // One distinct, non-identity, fixed-point-free list per PUBLISHED group.
  const std::vector<std::vector<int>> want = {
      {6, 2},  // group 0 — the sparse layers' paged K+V
      {4, 7},  // group 1 — the uniform recurrent group
      {5, 3},  // group 2 — the QSA INDEXER SIDE CACHE, the unread one
  };

  SchedulerOutput so;
  SamplingParams sp;
  sp.temperature = 0.0F;
  NewRequestData nr;
  nr.req_id = "r0";
  // 20 tokens over a 16-token block: two blocks, the second one PARTIAL.
  const std::vector<int32_t> prompt(20, 7);
  nr.prompt_token_ids = prompt;
  nr.sampling_params = sp;
  nr.block_ids = want;
  nr.num_computed_tokens = 0;
  nr.prefill_token_ids = prompt;
  so.scheduled_new_reqs.push_back(std::move(nr));
  so.num_scheduled_tokens["r0"] = 20;
  so.total_num_scheduled_tokens = 20;

  std::string msg = "<did not throw>";
  try {
    (void)runner.execute_model(so);
  } catch (const std::runtime_error& e) {
    msg = e.what();
  }
  // The forward is still refused, and the refusal now REPORTS the reach.
  //
  // ENG-MULTIKV-BYNAME repaired this count. It read `2 KV cache(s)` beside
  // `3 of 3` published groups, which is what #2343 measured as a contradiction:
  // the by-name channel omitted the recurrent group's three layers while the
  // block-table denominator counted its group. Five caches, three of them
  // recurrent, from three groups.
  CHECK(msg.find("5 KV cache(s) (2 paged, 3 recurrent)") != std::string::npos);
  CHECK(msg.find("from 3 published group(s)") != std::string::npos);
  CHECK(msg.find("block tables gathered for 3 of 3 published group(s)") !=
        std::string::npos);

  // The VALUES, per group. The expectation is the literal list fed to
  // `add_row` above, not anything read back out of the runner.
  const vllm::MultiKvCacheIndex& mk = runner.multi_kv_index();
  for (int g = 0; g < 3; ++g) {
    CAPTURE(g);
    int cols = 0;
    const std::vector<int32_t>* bt = mk.BlockTableForGroup(g, &cols);
    REQUIRE(bt != nullptr);
    // cdiv(32, 16) == 2. An attention row is rounded up to a multiple of
    // 128/16 == 8; the recurrent group 1 is a kNone row and keeps its spec's
    // width, cdiv(32, 16) + 0 speculative blocks == 2
    // (FIX-BLOCK-TABLE-ROW-WIDTH, block_table.py:322-331 @ e126687a9a).
    CHECK(cols == (g == 1 ? 2 : 8));
    REQUIRE(bt->size() == static_cast<size_t>(cols));
    CHECK((*bt)[0] == want[static_cast<size_t>(g)][0]);
    CHECK((*bt)[1] == want[static_cast<size_t>(g)][1]);
    // The columns past the sequence's two blocks were never written.
    for (int i = 2; i < cols; ++i) CHECK((*bt)[static_cast<size_t>(i)] == 0);
  }

  // Group 2's table is NOT group 1's and NOT group 0's, stated as an inequality
  // so that a gather keyed on the wrong id cannot pass by coincidence.
  int c0 = 0, c1 = 0, c2 = 0;
  CHECK(*mk.BlockTableForGroup(2, &c2) != *mk.BlockTableForGroup(1, &c1));
  CHECK(*mk.BlockTableForGroup(2, &c2) != *mk.BlockTableForGroup(0, &c0));

  // Out of range in both directions is a null answer, never a silent read.
  int cx = 7;
  CHECK(mk.BlockTableForGroup(3, &cx) == nullptr);
  CHECK(cx == 0);
  CHECK(mk.BlockTableForGroup(-1, &cx) == nullptr);
}

// A UNIFORM topology publishes no per-group tables at all: the channel stays
// null, exactly as `layer_names` / `group_ids` / `layer_indices` do, so no model
// shipping today can observe this wave.
TEST_CASE("runner: the per-group block tables stay EMPTY on a uniform topology") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);
  REQUIRE(runner.multi_kv_index().size() == 0);
  int cols = 5;
  CHECK(runner.multi_kv_index().BlockTableForGroup(0, &cols) == nullptr);
  CHECK(cols == 0);
  CHECK(runner.multi_kv_index().num_group_block_tables() == 0);
}

// ─── BYTE-NEUTRALITY: the uniform path does not enter the new code ──────────
//
// The obligation of this wave, stated as literals rather than as an absence of
// failures. Every shape shipping today keeps its buffer count, its page size,
// its per-layer classes and its total allocated bytes, and none of them
// populates the third channel or the per-layer index list.
TEST_CASE("runner: W3 is BYTE-NEUTRAL for every topology shipped today") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  const DType kv_dtype = vllm::v1::ResolveKvCacheDType();
  // The literals below are the bf16 default. Named rather than assumed, so a
  // changed default is a red REQUIRE and not a silently different number.
  REQUIRE(kv_dtype == DType::kBF16);
  // FullAttentionSpec(block 16, num_kv_heads 2, head_size 8, bf16):
  // 16 * 2 * (8 + 8) * 2 = 1024.
  constexpr int64_t kFaPage = 1024;
  using LKC = GPUModelRunner::LayerKvClass;

  const auto uniform = [&](const GPUModelRunner& r) {
    CHECK(r.layer_attn_kv_indices().empty());
    CHECK(r.attn_kv_layer_names().empty());
    CHECK(r.multi_kv_index().size() == 0);
    CHECK(r.multi_kv_index().Find("model.layers.0.attn") == -1);
    // ENG-MULTIKV-BYNAME: the two NEW pointers stay null on every uniform
    // topology, exactly like the three W3 added. The recurrent half of the
    // channel is what this row builds, and the hybrid gate models below carry a
    // recurrent group — so "the recurrent group is now named" must NOT mean
    // "the channel switched on for a hybrid".
    CHECK(r.multi_kv_index().payload_kinds == nullptr);
    CHECK(r.multi_kv_index().payload_slots == nullptr);
    CHECK(r.multi_kv_index().num_paged() == 0);
    CHECK(r.multi_kv_index().num_recurrent() == 0);
    vllm::KvCachePayload kind = vllm::KvCachePayload::kRecurrent;
    int32_t slot = 42;
    CHECK_FALSE(r.multi_kv_index().Resolve("model.layers.0.attn", &kind, &slot));
    CHECK(kind == vllm::KvCachePayload::kPaged);
    CHECK(slot == -1);
    CHECK_FALSE(r.multi_kv_index().PayloadAt(0, &kind, &slot));
  };

  SUBCASE("one full-attention group (dense Qwen3)") {
    GPUModelRunner runner(c, w, MakeFaOnlyKvConfig(c), Q(), 8, kMaxModelLen, 64);
    CHECK(runner.full_attn_group_id() == 0);
    CHECK(runner.gdn_group_id() == -1);
    CHECK(runner.attn_group_ids() == std::vector<int>{0});
    CHECK(runner.attn_kv().size() == 4);  // one per HIDDEN LAYER, as before
    CHECK(runner.fa_page_size_bytes() == kFaPage);
    CHECK(runner.kv_cache_allocated_paged_bytes() == 4 * kNumBlocks * kFaPage);
    for (int i = 0; i < 4; ++i)
      CHECK(runner.layer_kv_class()[static_cast<size_t>(i)] ==
            LKC::kFullAttention);
    uniform(runner);
  }
  SUBCASE("one MLA group (every MLA model in the tree)") {
    KVCacheConfig kv;
    kv.num_blocks = kNumBlocks;
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"mla"},
        std::make_shared<vllm::v1::MLAAttentionSpec>(kBlockSize,
                                                     /*head_size=*/576,
                                                     kv_dtype));
    GPUModelRunner runner(c, w, kv, Q(), 8, kMaxModelLen, 64);
    CHECK(runner.full_attn_group_id() == 0);
    CHECK(runner.attn_kv().size() == 4);
    // MLA drops the K+V factor 2: 16 * 1 * 576 * 2 = 18432.
    CHECK(runner.fa_page_size_bytes() == 18432);
    CHECK(runner.kv_cache_allocated_paged_bytes() == 4 * kNumBlocks * 18432);
    uniform(runner);
  }
  SUBCASE("full-attention + recurrent (the hybrid gate models)") {
    GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);
    CHECK(runner.full_attn_group_id() == 0);
    CHECK(runner.gdn_group_id() == 1);
    CHECK(runner.recurrent_group_ids() == std::vector<int>{1});
    // MakeKvConfig publishes PLACEHOLDER names, so membership falls back to
    // `layer_types` = 3 linear_attention + 1 full_attention. Unchanged by W3.
    CHECK(runner.attn_kv().size() == 1);
    CHECK(runner.fa_page_size_bytes() == kFaPage);
    CHECK(runner.layer_kv_class()[0] == LKC::kRecurrent);
    CHECK(runner.layer_kv_class()[3] == LKC::kFullAttention);
    uniform(runner);
  }
  SUBCASE("full-attention + recurrent + fa_draft (num_spec>0)") {
    KVCacheConfig kv = MakeKvConfig(c);
    kv.kv_cache_groups.emplace_back(
        std::vector<std::string>{"fa_draft"},
        std::make_shared<FullAttentionSpec>(
            kBlockSize, static_cast<int>(c.num_key_value_heads),
            static_cast<int>(c.head_dim), kv_dtype));
    GPUModelRunner runner(c, w, kv, Q(), 8, kMaxModelLen, 64);
    CHECK(runner.full_attn_group_id() == 0);
    CHECK(runner.gdn_group_id() == 1);
    CHECK(runner.attn_kv().size() == 1);
    CHECK(runner.fa_page_size_bytes() == kFaPage);
    uniform(runner);
  }
}

// ─── The real 167-entry topology, from the factory the loader dereferences ───
//
// The other production seam, and the one W2's review made blocking: reading
// `.make_kv_cache = &MakeDeepseekV4KVCache` proves what the source says, not
// what the gate measures. This case takes the pointer
// `MakeKVCacheResolved` -> `MakeKVCacheMaybeSpec` -> `ModelRegistry::MakeKVCache`
// dereferences, feeds its output to `GPUModelRunner`'s own constructor, and
// asserts the whole 167-buffer allocation.
TEST_CASE("runner: DeepSeek-V4's real 167-entry topology allocates end to end") {
  HfConfig v4;
  v4.architectures = {"DeepseekV4ForCausalLM"};
  v4.hidden_size = 4096;
  v4.num_hidden_layers = 43;
  v4.vocab_size = 129280;
  v4.num_attention_heads = 64;
  v4.num_key_value_heads = 1;
  v4.head_dim = 512;
  v4.rms_norm_eps = 1e-6;
  v4.max_position_embeddings = 1048576;
  nlohmann::json cr = nlohmann::json::array();
  for (int i = 0; i < 44; ++i)
    cr.push_back((i == 0 || i == 1 || i == 43) ? 0 : ((i % 2 == 0) ? 4 : 128));
  v4.raw = {
      {"hidden_size", 4096},        {"num_hidden_layers", 43},
      {"vocab_size", 129280},       {"num_attention_heads", 64},
      {"num_key_value_heads", 1},   {"head_dim", 512},
      {"qk_rope_head_dim", 64},     {"q_lora_rank", 1024},
      {"o_lora_rank", 1024},        {"o_groups", 8},
      {"sliding_window", 128},      {"rms_norm_eps", 1e-6},
      {"max_position_embeddings", 1048576},
      {"num_nextn_predict_layers", 1},
      {"n_routed_experts", 256},    {"num_experts_per_tok", 6},
      {"moe_intermediate_size", 2048}, {"n_shared_experts", 1},
      {"norm_topk_prob", true},     {"routed_scaling_factor", 1.5},
      {"swiglu_limit", 10.0},       {"scoring_func", "sqrtsoftplus"},
      {"topk_method", "noaux_tc"},  {"num_hash_layers", 3},
      {"expert_dtype", "fp4"},      {"hc_mult", 4},
      {"hc_sinkhorn_iters", 20},    {"hc_eps", 1e-6},
      {"index_head_dim", 128},      {"index_n_heads", 64},
      {"index_topk", 512},          {"compress_rope_theta", 160000},
      {"rope_theta", 10000},        {"tie_word_embeddings", false},
      {"compress_ratios", cr},
  };

  const vllm::ModelRegistration& reg = vllm::ModelRegistry::Resolve(v4);
  REQUIRE(reg.factory != nullptr);
  REQUIRE(reg.factory->make_kv_cache != nullptr);
  const KVCacheConfig kv =
      reg.factory->make_kv_cache(v4, /*block_size=*/256, /*num_blocks=*/2);
  REQUIRE(kv.kv_cache_groups.size() == 7);

  // The model the runner holds is irrelevant to KV allocation (it drives only
  // the forward, which this case never runs); `config_` is what the allocation
  // reads, and it is DeepSeek-V4's.
  const HfConfig qc = MakeConfig();
  const Qwen3_5MoeWeights qw = MakeWeights(qc);
  GPUModelRunner runner(v4, qw, kv, Q(), /*max_num_reqs=*/2, kMaxModelLen,
                        /*max_num_batched_tokens=*/16);

  CHECK(runner.attn_kv().size() == 167);
  CHECK(runner.attn_kv_layer_names().size() == 167);
  CHECK(runner.attn_group_ids() == std::vector<int>{0, 1, 2, 3, 4, 5, 6});
  // 21 C4A + 20 C128A + 21 indexer + 43 SWA + 21 + 21 + 20 compressor states,
  // each at 2 blocks of the page W1's table derives from upstream.
  const int64_t want_bytes =
      2 * (21 * 37440 + 20 * 1728 + 21 * 8640 + 43 * 37440 + 21 * 32832 +
           21 * 8640 + 20 * 32832);
  CHECK(runner.kv_cache_allocated_paged_bytes() == want_bytes);
  using LKC = GPUModelRunner::LayerKvClass;
  // Every one of the 43 layers has at least the SWA cache, so none is kNone.
  for (int l = 0; l < 43; ++l) {
    CAPTURE(l);
    CHECK(runner.layer_kv_class()[static_cast<size_t>(l)] == LKC::kMultiCache);
  }
  // Layer 2 is C4A: latent + SWA + indexer key + two compressor states.
  CHECK(runner.layer_attn_kv_indices()[2].size() == 5);
  // Layer 3 is C128A: latent + SWA + one compressor state.
  CHECK(runner.layer_attn_kv_indices()[3].size() == 3);
  // Layers 0 and 1 have no MLA cache at all upstream — only the SWA cache.
  CHECK(runner.layer_attn_kv_indices()[0].size() == 1);
  CHECK(runner.layer_attn_kv_indices()[1].size() == 1);
  // `compress_ratios[l]` is 4 on the EVEN layers from 2 up, 128 on the odd
  // ones, so the indexer key cache exists on 6 and not on 7.
  CHECK(runner.multi_kv_index().Find("model.layers.6.attn.indexer.k_cache") >= 0);
  CHECK(runner.multi_kv_index().Find("model.layers.7.attn.indexer.k_cache") == -1);

  // ENG-MULTIKV-BYNAME: DeepSeek-V4 publishes NO recurrent group, so the
  // channel is all-paged and every flat index IS its slot in `attn_kv`. Stated
  // as an assertion because it is the neutrality obligation for the seven-group
  // shape: this row moved the index off "parallel to attn_kv", and on a topology
  // with no recurrent member the two orders must still coincide entry for entry.
  const vllm::MultiKvCacheIndex& mk = runner.multi_kv_index();
  CHECK(mk.size() == 167);
  CHECK(mk.num_paged() == 167);
  CHECK(mk.num_recurrent() == 0);
  REQUIRE(runner.attn_kv_layer_names().size() == 167);
  for (size_t i = 0; i < 167; ++i) {
    CAPTURE(i);
    REQUIRE((*mk.layer_names)[i] == runner.attn_kv_layer_names()[i]);
    vllm::KvCachePayload kind = vllm::KvCachePayload::kRecurrent;
    int32_t slot = -1;
    REQUIRE(mk.PayloadAt(static_cast<int64_t>(i), &kind, &slot));
    CHECK(kind == vllm::KvCachePayload::kPaged);
    CHECK(slot == static_cast<int32_t>(i));
  }
}

// ─── The reorder's DECODE THRESHOLD reaches the reorder (#2129) ──────────────
//
// Upstream resolves one `reorder_batch_threshold` and passes it into
// `reorder_batch_to_split_decodes_and_prefills`
// (gpu_model_runner.py:1126-1130 @ pin 5559679229); a backend that supports
// spec-as-decode raises it to `1 + (2 if parallel_drafting else 1) * k`
// (backend.py:657-687, requested by gdn_attn.py:112 for every speculative
// configuration). W10 mirrored the arithmetic
// (`SpecAsDecodeReorderThreshold`, include/vllm/v1/attention/backend.h) and
// left the runner calling the reorder with no argument, so it took the
// declaration default of 1.
//
// WHY THE ORDER IS THE OBSERVABLE, NOT THE TOKENS. The reorder is a
// permutation and every consumer is built after it, so a wrong order changes
// which lane a row is dispatched onto without changing what is emitted — the
// acceptance-only failure class #1366 already fired. The gate is therefore
// structural.
//
// THE DISCRIMINATING POPULATION. `ResolveMtp` leaves `parallel_drafting` false
// (only `dflash`/`dspark` set it, speculative.py:963-964), so at k=8 the
// threshold is exactly `1 + 1*8 == 9`. Four rows, all with context:
//
//   row  scheduled  still prefilling   region at 1   at 9 (mirrored)   at 17
//   D1     1        no                 decode  0     decode  0         decode 0
//   V9     9        no                 long_e  2     decode  0         decode 0
//   X10   10        no                 long_e  2     long_e  2         decode 0
//   C     15        yes                long_e  2     long_e  2         short  1
//
// admitted [C, X10, V9, D1]. `V9 before X10` is the one predicate that holds
// ONLY at the mirrored value: at 1 the verify row is a long extend and sorts
// behind X10; at 17 X10 is promoted to a decode and the min-swap partition puts
// it ahead of V9. So neither dropping the argument nor widening the threshold
// to the other arm of the formula can satisfy this case — both were run as
// mutations and both red here.
//
// A hardcoded permutation would be the brittle way to say that, and the
// 3-request case above explains why this file avoids one: upstream partitions
// with minimum swaps, not a stable sort. The invariant asserted instead is the
// reorder's own contract — the final order is NON-DECREASING in the region the
// MIRRORED threshold assigns — plus the per-slot fields following whichever
// request landed where.
TEST_CASE("runner: the spec-as-decode reorder threshold reaches the reorder") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  // The public LoadedModel constructor is the one that carries a
  // SpeculativeConfig; the concrete-weight overloads do not, and widening one
  // would touch a region another row owns. Nothing is constructed by hand
  // here: the runner is the production type and `execute_model` below is the
  // production entry point the reorder sits in.
  std::unique_ptr<vllm::LoadedModel> lm = vllm::BorrowQwen3_5MoeLoadedModel(w);
  const vllm::SpeculativeConfig spec =
      vllm::SpeculativeConfig::ResolveMtp(/*mtp_num_hidden_layers=*/1,
                                          /*num_speculative_tokens=*/8);
  REQUIRE_FALSE(spec.parallel_drafting);  // mtp is serial: threshold is 1 + k.
  REQUIRE(spec.ResolvedNumSpeculativeTokens() == 8);
  GPUModelRunner runner(c, *lm, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64, spec);

  // C: chunked prefill continuation — 20-token prompt, 5 computed, 15 more
  // this step. Has context, still prefilling, above the threshold.
  std::vector<int32_t> c_prompt(20);
  for (int i = 0; i < 20; ++i) c_prompt[static_cast<size_t>(i)] = 1 + i;
  NewRequestData rc = MakeNewReq("C", c_prompt, {}, /*num_computed=*/5,
                                 /*fa_blocks=*/{2, 3}, /*gdn_block=*/2,
                                 Greedy());
  // X10: done prefilling, 10 scheduled — one token ABOVE the threshold.
  NewRequestData rx =
      MakeNewReq("X10", {21, 22, 23, 24}, {25, 26, 27, 28, 29, 30, 31, 32, 33, 34},
                 /*num_computed=*/4, /*fa_blocks=*/{1}, /*gdn_block=*/1,
                 Greedy());
  // V9: a verify row — done prefilling, 1+k == 9 scheduled, ON the threshold.
  NewRequestData rv =
      MakeNewReq("V9", {35, 36, 37, 38}, {39, 1, 2, 3, 4, 5, 6, 7, 8},
                 /*num_computed=*/4, /*fa_blocks=*/{0}, /*gdn_block=*/0,
                 Greedy());
  // D1: a plain single-token decode — a decode at every threshold.
  NewRequestData rd = MakeNewReq("D1", {9, 10, 11}, {12}, /*num_computed=*/3,
                                 /*fa_blocks=*/{4}, /*gdn_block=*/3, Greedy());

  SchedulerOutput so = NewStep(
      {rc, rx, rv, rd}, {{"C", 15}, {"X10", 10}, {"V9", 9}, {"D1", 1}});
  auto out_opt = runner.execute_model(so);
  CHECK_FALSE(out_opt.has_value());

  const auto& ib = runner.input_batch();
  REQUIRE(ib.num_reqs() == 4);
  std::vector<std::string> order;
  for (int i = 0; i < 4; ++i) {
    REQUIRE(ib.req_ids[static_cast<size_t>(i)].has_value());
    order.push_back(*ib.req_ids[static_cast<size_t>(i)]);
  }
  INFO("order: ", order[0], " ", order[1], " ", order[2], " ", order[3]);
  const auto pos = [&order](const std::string& id) {
    for (size_t i = 0; i < order.size(); ++i)
      if (order[i] == id) return static_cast<int>(i);
    return -1;
  };

  // THE DISCRIMINATOR. Only the mirrored threshold makes the verify row a
  // decode while leaving X10 a long extend.
  CHECK(pos("V9") < pos("X10"));
  // And the plain decode is a decode too, so the decode region holds both.
  CHECK(pos("D1") < pos("X10"));
  CHECK(pos("D1") < pos("C"));
  CHECK(pos("V9") < pos("C"));

  // The reorder's contract: the batch is sorted by region
  // decode(0) -> short_extend(1) -> long_extend(2) -> prefill(3), with the
  // regions taken from the MIRRORED threshold of 9.
  const std::map<std::string, int> want_region = {
      {"D1", 0}, {"V9", 0}, {"X10", 2}, {"C", 2}};
  for (int i = 1; i < 4; ++i) {
    CAPTURE(i);
    CHECK(want_region.at(order[static_cast<size_t>(i - 1)]) <=
          want_region.at(order[static_cast<size_t>(i)]));
  }

  // The ordering-dependent per-slot fields follow the SAME order, so none of
  // this can be satisfied by permuting req_ids alone.
  const std::map<std::string, int> want_seq_len = {
      {"D1", 4}, {"V9", 13}, {"X10", 14}, {"C", 20}};
  const std::map<std::string, int> want_fa_block = {
      {"D1", 4}, {"V9", 0}, {"X10", 1}, {"C", 2}};
  const std::map<std::string, int> want_scheduled = {
      {"D1", 1}, {"V9", 9}, {"X10", 10}, {"C", 15}};
  const auto& am = runner.last_attn_meta();
  const auto& step = runner.last_step();
  const int cols = am.block_table_num_cols;
  REQUIRE(am.seq_lens.size() == 4);
  REQUIRE(step.query_start_loc.size() == 5);
  CHECK(step.query_start_loc[0] == 0);
  for (int i = 0; i < 4; ++i) {
    const std::string& rid = order[static_cast<size_t>(i)];
    CAPTURE(rid);
    CHECK(am.seq_lens[static_cast<size_t>(i)] == want_seq_len.at(rid));
    CHECK(am.block_table_tensor[static_cast<size_t>(i * cols)] ==
          want_fa_block.at(rid));
    CHECK(step.query_start_loc[static_cast<size_t>(i + 1)] -
              step.query_start_loc[static_cast<size_t>(i)] ==
          want_scheduled.at(rid));
  }
}

// The non-speculative arm is byte-identical: `SpecAsDecodeReorderThreshold`
// returns 1 for k <= 0, and `num_spec()` is 0 without a SpeculativeConfig. The
// SAME four rows on a runner with no speculator keep the pre-#2129 partition,
// in which only D1 is a decode and the 9-token row sorts with the long
// extends. Deleting the threshold argument leaves this case green, which is
// exactly what makes it the byte-identity pin rather than a second copy of the
// gate above.
TEST_CASE("runner: no speculator keeps the reorder threshold at 1") {
  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), /*max_num_reqs=*/8,
                        kMaxModelLen, /*max_num_batched_tokens=*/64);

  std::vector<int32_t> c_prompt(20);
  for (int i = 0; i < 20; ++i) c_prompt[static_cast<size_t>(i)] = 1 + i;
  NewRequestData rc = MakeNewReq("C", c_prompt, {}, /*num_computed=*/5,
                                 /*fa_blocks=*/{2, 3}, /*gdn_block=*/2,
                                 Greedy());
  NewRequestData rx =
      MakeNewReq("X10", {21, 22, 23, 24}, {25, 26, 27, 28, 29, 30, 31, 32, 33, 34},
                 /*num_computed=*/4, /*fa_blocks=*/{1}, /*gdn_block=*/1,
                 Greedy());
  NewRequestData rv =
      MakeNewReq("V9", {35, 36, 37, 38}, {39, 1, 2, 3, 4, 5, 6, 7, 8},
                 /*num_computed=*/4, /*fa_blocks=*/{0}, /*gdn_block=*/0,
                 Greedy());
  NewRequestData rd = MakeNewReq("D1", {9, 10, 11}, {12}, /*num_computed=*/3,
                                 /*fa_blocks=*/{4}, /*gdn_block=*/3, Greedy());

  SchedulerOutput so = NewStep(
      {rc, rx, rv, rd}, {{"C", 15}, {"X10", 10}, {"V9", 9}, {"D1", 1}});
  auto out_opt = runner.execute_model(so);
  CHECK_FALSE(out_opt.has_value());

  const auto& ib = runner.input_batch();
  REQUIRE(ib.num_reqs() == 4);
  std::vector<std::string> order;
  for (int i = 0; i < 4; ++i) {
    REQUIRE(ib.req_ids[static_cast<size_t>(i)].has_value());
    order.push_back(*ib.req_ids[static_cast<size_t>(i)]);
  }
  INFO("order: ", order[0], " ", order[1], " ", order[2], " ", order[3]);
  // Only D1 is below a threshold of 1, so it is the only decode and it leads.
  CHECK(order[0] == "D1");
  // And the verify-shaped row is NOT promoted: it stays with the long extends.
  const std::map<std::string, int> want_region = {
      {"D1", 0}, {"V9", 2}, {"X10", 2}, {"C", 2}};
  for (int i = 1; i < 4; ++i) {
    CAPTURE(i);
    CHECK(want_region.at(order[static_cast<size_t>(i - 1)]) <=
          want_region.at(order[static_cast<size_t>(i)]));
  }
}

// --- #2534: the final-logit dump instrument ---------------------------------
//
// `VT_DUMP_LOGITS=<dir>` is the instrument the arm's remaining question needs:
// our per-step logit vector, so our delta against llama.cpp b10451 can be placed
// against the oracle's own measured 0.2020-to-1.3657 self-perturbation band. See
// .agents/specs/qwen38-27b-q4km-logit-dump.md.
//
// Two things have to hold or the measurement is worthless, and they are the two
// the spec pre-registered before the code existed:
//
//   ALIGNMENT — the argmax of the dumped row must be the token the runner
//   actually sampled at that step. If it is not, the two sides are not
//   describing the same context and every delta computed from them is noise.
//
//   NON-PERTURBATION — the dump must not change what it measures. It reads a
//   buffer that already exists and writes nothing back, and the ids must be
//   identical to a run with the instrument off.
//
// `VT_DUMP_LOGITS` is read ONCE at static init, so one process observes one
// value and this case cannot test both arms. The default registration runs with
// it unset and asserts the dump is INERT (no files, ids still produced); the
// second registration in tests/CMakeLists.txt sets it and asserts the dump
// exists, is the right size, and agrees with the sampled ids.
TEST_CASE("runner: the logit dump agrees with the sampled ids and never perturbs them") {
  const char* dump_dir = std::getenv("VT_DUMP_LOGITS");

  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);

  // One greedy decode row: prompt 3 tokens + 1 already-produced, so this step
  // samples a real token rather than a discarded prefill position.
  NewRequestData d = MakeNewReq("D", {6, 7, 8}, {9}, /*num_computed=*/3,
                                /*fa_blocks=*/{0, 1}, /*gdn_block=*/0, Greedy());
  SchedulerOutput so = NewStep({d}, {{"D", 1}});

  auto out_opt = runner.execute_model(so);
  CHECK_FALSE(out_opt.has_value());  // MRV2 split: forward done, sampling next.
  ModelRunnerOutput out = runner.sample_tokens(std::nullopt);

  REQUIRE(out.req_ids.size() == 1);
  CHECK(out.req_ids[0] == "D");
  REQUIRE(out.sampled_token_ids.size() == 1);
  REQUIRE(out.sampled_token_ids[0].size() == 1);
  const int32_t sampled = out.sampled_token_ids[0][0];
  CHECK(sampled >= 0);
  CHECK(sampled < static_cast<int32_t>(c.vocab_size));

  if (dump_dir == nullptr) {
    // INERT arm. The instrument is off, the step still samples, and nothing is
    // written. A dump that wrote unconditionally would red here.
    const std::string stray = std::string("ours_D.f32");
    std::ifstream f(stray, std::ios::binary);
    CHECK_FALSE(f.good());
    return;
  }

  // ACTIVE arm.
  const std::string base = std::string(dump_dir) + "/ours_D";

  // One step of a [1, vocab] f32 row, and NOTHING else: a dump that also wrote
  // the discarded prefill rows, or wrote the whole [num_tokens, vocab] forward
  // output instead of the gathered sample rows, reds on this size.
  std::ifstream bin(base + ".f32", std::ios::binary | std::ios::ate);
  REQUIRE(bin.good());
  const std::streamsize bytes = bin.tellg();
  CHECK(bytes == static_cast<std::streamsize>(sizeof(float) *
                                              static_cast<size_t>(c.vocab_size)));

  // ALIGNMENT: recompute the argmax from the dumped bytes and require the
  // runner's own sampled id. This is the control the spec named, executed on
  // the same data the comparison will use.
  bin.seekg(0);
  std::vector<float> row(static_cast<size_t>(c.vocab_size));
  bin.read(reinterpret_cast<char*>(row.data()),
           static_cast<std::streamsize>(row.size() * sizeof(float)));
  size_t best = 0;
  for (size_t v = 1; v < row.size(); ++v) {
    if (row[v] > row[best]) best = v;
  }
  CHECK(static_cast<int32_t>(best) == sampled);

  // The sidecar carries the same claim in text, keyed by step, and step 1 is
  // this row's 0-based decode index (prompt 3 + 1 produced = 4 tokens, minus 3
  // prompt tokens). A sidecar keyed by the wrong counter reds here.
  std::ifstream ids(base + ".ids.txt");
  REQUIRE(ids.good());
  int step = -1;
  long long tok = -1;
  ids >> step >> tok;
  CHECK(step == 1);
  CHECK(tok == static_cast<long long>(sampled));
}

// #2534, and the reason this case exists at all. The first version of the dump
// was gated only through `sample_tokens`, and a case that called that method
// DIRECTLY passed 13 of 13 while the instrument wrote nothing in the real
// engine. `async_input_combine_` is assigned directly in the runner ctor from
// `AsyncRunnerEnvDefault()`, and `AsyncRunnerFlagIsOn` answers TRUE when
// VT_ASYNC_RUNNER is unset, so the PRODUCTION default samples in the
// device-resident branch of `sample_tokens_async`. A gate that only drives
// `sample_tokens` measures a class and not the capability.
//
// This drives the async branch explicitly and requires the dump to appear there
// too. Deleting either `dump_step_logits` call site reds exactly one of the two
// cases, which is what makes them independent rather than duplicates.
TEST_CASE("runner: the logit dump fires on the ASYNC branch the default takes") {
  const char* dump_dir = std::getenv("VT_DUMP_LOGITS");
  if (dump_dir == nullptr) return;  // inert arm: covered by the case above.

  const HfConfig c = MakeConfig();
  const Qwen3_5MoeWeights w = MakeWeights(c);
  GPUModelRunner runner(c, w, MakeKvConfig(c), Q(), 8, kMaxModelLen, 64);
  runner.set_async_input_combine(true);  // the production default
  REQUIRE(runner.async_input_combine());

  NewRequestData d = MakeNewReq("A", {6, 7, 8}, {9}, /*num_computed=*/3,
                                /*fa_blocks=*/{0, 1}, /*gdn_block=*/0, Greedy());
  SchedulerOutput so = NewStep({d}, {{"A", 1}});
  auto out_opt = runner.execute_model(so);
  CHECK_FALSE(out_opt.has_value());

  std::unique_ptr<vllm::v1::AsyncModelRunnerOutput> async_out =
      runner.sample_tokens_async(std::nullopt);
  REQUIRE(async_out != nullptr);
  ModelRunnerOutput out = async_out->get_output();
  REQUIRE(out.sampled_token_ids.size() == 1);
  REQUIRE(out.sampled_token_ids[0].size() == 1);
  const int32_t sampled = out.sampled_token_ids[0][0];

  // The dump must exist for THIS request, one step of [1, vocab] f32.
  std::ifstream bin(std::string(dump_dir) + "/ours_A.f32",
                    std::ios::binary | std::ios::ate);
  REQUIRE(bin.good());
  CHECK(bin.tellg() == static_cast<std::streamsize>(
                           sizeof(float) * static_cast<size_t>(c.vocab_size)));
  bin.seekg(0);
  std::vector<float> row(static_cast<size_t>(c.vocab_size));
  bin.read(reinterpret_cast<char*>(row.data()),
           static_cast<std::streamsize>(row.size() * sizeof(float)));
  size_t best = 0;
  for (size_t v = 1; v < row.size(); ++v) if (row[v] > row[best]) best = v;
  CHECK(static_cast<int32_t>(best) == sampled);
}

// ─── SPEC-DFLASH2 A2-2 (#2802): THE VERIFY DOWNLOAD'S PAGE-LOCKED STAGING ────
//
// A fresh review found the copy-queue verify route copying into a plain
// `std::vector<int32_t>`. A device-to-host `cudaMemcpyAsync` into PAGEABLE host
// memory is host-synchronous — the driver stages it through its own pinned
// buffer and the call returns only when the bytes are across — so the fork and
// ready events around that copy were decorative and no later wave could have
// obtained overlap from them. The destination is now `PinnedGrowStaging`, which
// allocates through `vt::Backend::AllocPinned`, as this tree's async sampled-id
// route already did (`src/vllm/v1/worker/gpu/async_output.cpp`) and as upstream
// gets for free by copying into torch CPU memory (async_utils.py:124-125).
//
// THE PART THIS TIER CAN GATE is the growth rule, and it is the part that could
// silently reintroduce the defect the same review found in the accept walk's
// argmax scratch. A grow must RETAIN the block it replaces: a copy already
// issued into the old block is still writing it, and freeing it is a
// use-after-free that is invisible while the wait is in step and wrong the
// moment A2-4 moves the wait. On CPU `AllocPinned` forwards to `Alloc`, so page
// locking itself is not observable here; the retain rule is.
//
// RED-first, by mutation: making `Get` free the previous block before allocating
// the new one (the shape the CUDA argmax global had) reds the `live_blocks()`
// assertions below while every token case in this binary stays green — the
// tokens never move on this backend either, which is exactly why the rule needs
// its own assertion.
TEST_CASE("A2-2: the verify download staging is page-locked and a grow RETAINS the old block") {
  vt::Backend& backend = vt::GetBackend(vt::DeviceType::kCPU);
  vllm::v1::PinnedGrowStaging staging;
  CHECK(staging.live_blocks() == 0);

  int32_t* small = staging.Get(backend, 4);
  REQUIRE(small != nullptr);
  CHECK(staging.live_blocks() == 1);
  CHECK(staging.elems() == 4);
  for (int i = 0; i < 4; ++i) small[i] = 100 + i;

  // A request that FITS returns the same block: the steady state allocates
  // nothing, which is the reason this is a member and not a local.
  CHECK(staging.Get(backend, 4) == small);
  CHECK(staging.Get(backend, 1) == small);
  CHECK(staging.live_blocks() == 1);

  // A GROW hands back a new block and keeps the old one allocated.
  int32_t* big = staging.Get(backend, 64);
  REQUIRE(big != nullptr);
  CHECK(big != small);
  CHECK(staging.live_blocks() == 2);
  CHECK(staging.elems() == 64);
  // The retained block is still ours to read: nothing freed it, so a copy that
  // was already writing it is writing memory this object still owns.
  CHECK(small[0] == 100);
  CHECK(small[3] == 103);

  // A second grow retains again; the destructor releases every block.
  staging.Get(backend, 256);
  CHECK(staging.live_blocks() == 3);
}
