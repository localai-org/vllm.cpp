// Gemma-4 (`Gemma4ForConditionalGeneration`) — CLAIM-GEMMA4-MM-E2E: the registered
// engine mm-forward DRIVER for image->text. Mirrors the Qwen3-VL fold
// (qwen3_vl.cpp Qwen3VLGenerateGreedyViaRegistry): a single-sequence greedy driver
// that runs EVERY decoder step through ModelRegistry::Forward with
// ModelForwardInput.mm carrying the already-merged inputs_embeds + the Gemma-4
// PLE-masked ids, so the ENGINE registered forward
// (ForwardGemma4ForConditionalGeneration's mm branch) drives the decode — NOT a
// bespoke in-TU forward. This is the additive-TU seam (new TU + the driver; ZERO
// edit to the shared runner / scheduler).
//
// Gemma-4's mm decode differs from Qwen3-VL: standard 1-D positions (NO 3-D MRoPE),
// NO DeepStack, and the PLE (Per-Layer Embeddings) require the mm-masked token ids
// (image rows -> 0) for the embed_tokens_per_layer lookup — gemma4_mm.py
// embed_input_ids (:1962-1973) + gemma4.py get_per_layer_inputs (:848-870) /
// project_per_layer_inputs (:872-898) / forward inputs_embeds branch (:900-928).
// The vision soft tokens are the embed_vision projector output, masked-scattered
// into the <image> placeholder rows (SupportsMultiModal.embed_input_ids) AFTER the
// text embeds are sqrt(hidden)-scaled (the soft tokens are NOT re-scaled).
//
// Ported 1:1 from vllm/model_executor/models/gemma4_mm.py @ 555967922 (the mm
// wrapper forward) + gemma4.py (the language_model stack). The per-step forward
// itself is the shared Gemma4Model::ForwardMm (gemma4.cpp).
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/gemma4.h"
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3_5.h"        // GdnStateCache, PagedKvCache
#include "vllm/model_executor/models/qwen3_vl_text.h"  // Qwen3VLMergeMultimodal (modality-agnostic)
#include "vllm/v1/attention/backends/gdn_attn.h"        // v1::GDNAttentionMetadata
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::Backend;
using vt::DType;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using namespace dense_attn;  // Dev, DBuf, ResidentWeight

std::vector<float> Bf16BitsToF32(const uint16_t* p, int64_t n) {
  std::vector<float> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::BF16ToF32(p[i]);
  return out;
}
std::vector<uint16_t> F32ToBf16Bits(const float* p, int64_t n) {
  std::vector<uint16_t> out(static_cast<size_t>(n));
  for (int64_t i = 0; i < n; ++i) out[static_cast<size_t>(i)] = vt::F32ToBF16(p[i]);
  return out;
}
void RoundToBf16(std::vector<float>& v) {
  for (float& x : v) x = vt::BF16ToF32(vt::F32ToBF16(x));
}
int64_t ArgMax(const std::vector<float>& v) {
  int64_t best = 0;
  float bv = v.empty() ? 0.0f : v[0];
  for (int64_t i = 1; i < static_cast<int64_t>(v.size()); ++i)
    if (v[static_cast<size_t>(i)] > bv) {
      bv = v[static_cast<size_t>(i)];
      best = i;
    }
  return best;
}

// The greedy top1-top2 logit margin at a step. Small margin == a bf16 near-tie
// (per the ratified near-tie methodology); large margin == a confident pick.
float Top2Margin(const std::vector<float>& logits) {
  int64_t a = ArgMax(logits);
  float av = logits[static_cast<size_t>(a)];
  float bv = -1e30f;
  for (int64_t i = 0; i < static_cast<int64_t>(logits.size()); ++i)
    if (i != a && logits[static_cast<size_t>(i)] > bv) bv = logits[static_cast<size_t>(i)];
  return av - bv;
}

// Diagnostic (env VLLM_GEMMA4_MM_DEBUG): print the top-2 logits + their margin for
// one step, to characterize near-tie divergences. No behavioral effect when unset.
void DebugTop2(int step, const std::vector<float>& logits) {
  if (std::getenv("VLLM_GEMMA4_MM_DEBUG") == nullptr) return;
  int64_t a = ArgMax(logits);
  float av = logits[static_cast<size_t>(a)];
  int64_t b = -1;
  float bv = -1e30f;
  for (int64_t i = 0; i < static_cast<int64_t>(logits.size()); ++i)
    if (i != a && logits[static_cast<size_t>(i)] > bv) {
      bv = logits[static_cast<size_t>(i)];
      b = i;
    }
  std::fprintf(stderr, "[g4mm step %d] top1 id=%ld logit=%.5f | top2 id=%ld logit=%.5f | margin=%.6f\n",
               step, static_cast<long>(a), av, static_cast<long>(b), bv, av - bv);
}

// Build a single-sequence CommonAttentionMetadata for a step (mirror qwen3_vl.cpp
// StepMeta). first_slot is the sequence index of the first token in the step.
CommonAttentionMetadata StepMeta(int64_t T, int64_t seq_len, int64_t first_slot) {
  CommonAttentionMetadata m;
  m.num_reqs = 1;
  m.num_actual_tokens = static_cast<int>(T);
  m.query_start_loc = {0, static_cast<int32_t>(T)};
  m.query_start_loc_cpu = m.query_start_loc;
  m.seq_lens = {static_cast<int32_t>(seq_len)};
  m.seq_lens_cpu = m.seq_lens;
  m.max_query_len = static_cast<int>(T);
  m.max_seq_len = static_cast<int>(seq_len);
  m.block_table_num_cols = 1;
  m.block_table_tensor = {0};
  for (int64_t t = 0; t < T; ++t)
    m.slot_mapping.push_back(static_cast<int64_t>(first_slot + t));
  m.causal = true;
  return m;
}

// The vocab_size_per_layer_input range for the PLE mask (gemma4.py:830-831 default
// = full vocab). Read from the text_config view of the full config.json.
int64_t VocabPerLayer(const HfConfig& config) {
  const nlohmann::json& raw =
      (config.raw.contains("text_config") && config.raw.at("text_config").is_object())
          ? config.raw.at("text_config")
          : config.raw;
  const auto it = raw.find("vocab_size_per_layer_input");
  if (it != raw.end() && it->is_number_integer()) return it->get<int64_t>();
  return config.vocab_size;
}

// Gemma-4 PLE token ids for one token (gemma4_mm.py embed_input_ids :1962-1969 +
// gemma4.py get_per_layer_inputs :857-863): multimodal rows -> 0, and the range
// mask id < vocab_size_per_layer_input else 0.
int32_t PleId(int32_t id, bool is_mm, int64_t vocab_per_layer) {
  if (is_mm) return 0;
  return (id >= 0 && id < vocab_per_layer) ? id : 0;
}

// Embed one row of token ids through embed_tokens * sqrt(hidden) (device), download
// as host bf16 bits. Used for both the prefill batch and the single decode token.
std::vector<uint16_t> EmbedScaledBf16(Dev d, const Gemma4Weights& weights,
                                      const HfConfig& config,
                                      const std::vector<int32_t>& ids) {
  const int64_t H = config.hidden_size;
  const int64_t T = static_cast<int64_t>(ids.size());
  DBuf emb(d, DType::kBF16, {T, H});
  EmbedGather(d, emb, ids, weights.embed_tokens, config.vocab_size, H,
              "gemma4 mm embed");
  const float nsqrt = std::sqrt(static_cast<float>(H));
  const double normalizer = static_cast<double>(vt::BF16ToF32(vt::F32ToBF16(nsqrt)));
  vt::MulScalar(d.q, emb.t(), emb.t(), normalizer);
  std::vector<uint16_t> bits(static_cast<size_t>(T * H));
  emb.Download(d, bits.data());
  return bits;
}

}  // namespace

std::vector<int32_t> Gemma4GenerateGreedyViaRegistry(
    LoadedModel& model, const std::vector<int32_t>& prompt_ids,
    const std::vector<float>& mm_projected, int32_t image_token_id,
    int32_t eos_token_id, const Gemma4Weights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens, std::vector<float>* out_margins) {
  Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const int64_t H = config.hidden_size;
  const int64_t Hkv = config.num_key_value_heads;
  const int64_t L = config.num_hidden_layers;
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());
  const int64_t vocab_per_layer = VocabPerLayer(config);

  // Image placeholder mask + count.
  std::vector<bool> mask(static_cast<size_t>(T0), false);
  int64_t n_img = 0;
  for (int64_t t = 0; t < T0; ++t)
    if (prompt_ids[static_cast<size_t>(t)] == image_token_id) {
      mask[static_cast<size_t>(t)] = true;
      ++n_img;
    }
  VT_CHECK(n_img > 0, "gemma4 mm: no image token in prompt");
  const int64_t N = static_cast<int64_t>(mm_projected.size()) / (H > 0 ? H : 1);
  VT_CHECK(N == n_img, "gemma4 mm: mm_projected rows != image-token count");

  // Per-layer paged KV, each sized by its own head_dim (256 sliding / 512 full).
  // YOCO shared layers still allocate their own (unused) buffer; the forward reads
  // attn_kv[kv_target] for them, and the target layers are non-shared and correctly
  // sized here (the memory-only dedup is the named G1c residual).
  const int64_t block_size = T0 + max_new_tokens + 8;
  std::vector<std::shared_ptr<void>> kv_storage;
  std::vector<PagedKvCache> attn_kv;
  for (int64_t l = 0; l < L; ++l) {
    const int64_t Dh = weights.layers[static_cast<size_t>(l)].head_dim;
    const size_t kv_bytes =
        static_cast<size_t>(1 * 2 * block_size * Hkv * Dh) * vt::SizeOf(DType::kBF16);
    void* p = backend.Alloc(kv_bytes);
    backend.Memset(d.q, p, 0, kv_bytes);
    kv_storage.emplace_back(p, [&backend](void* q) { backend.Free(q); });
    PagedKvCache kv;
    kv.data = p;
    kv.dtype = DType::kBF16;
    kv.num_blocks = 1;
    kv.block_size = block_size;
    kv.num_kv_heads = Hkv;
    kv.head_size = Dh;
    attn_kv.push_back(kv);
  }

  // The registered mm STEP: pack the merged embeds + PLE ids into
  // ModelForwardInput.mm and drive ModelRegistry::Forward (Gemma-4's mm branch).
  auto step = [&](const std::vector<uint16_t>& embeds,
                  const std::vector<int32_t>& ple_ids,
                  const std::vector<int32_t>& pos1d, int64_t T,
                  const CommonAttentionMetadata& meta) -> std::vector<float> {
    const std::vector<int32_t> no_tokens;
    std::vector<GdnStateCache> no_gdn_state;
    v1::GDNAttentionMetadata gdn_meta{};
    const std::vector<int32_t> gather_li = {static_cast<int32_t>(T - 1)};
    // ENG-MM-INPUT-PIPELINE P1: the seam takes DEVICE handles, so this driver —
    // which builds its embeds and PLE ids on the host — uploads them HERE. The two
    // DBufs outlive the ModelRegistry::Forward call below. The runner slice (#2300)
    // will hand the seam a persistent device buffer and skip the upload.
    DBuf dembeds(d, DType::kBF16, {T, H}, embeds.data());
    DBuf dple(d, DType::kI32, {T}, ple_ids.data());
    MultiModalForwardInput mm{};
    mm.inputs_embeds = dembeds.t();
    mm.ple_token_ids = dple.t();
    ModelForwardInput in{
        .token_ids = no_tokens,
        .positions = pos1d,
        .attn_meta = meta,
        .gdn_meta = gdn_meta,
        .attn_kv = attn_kv,
        .gdn_state = no_gdn_state,
        .config = config,
        .queue = queue,
        .logits_indices = gather_li,
        .num_reqs = meta.num_reqs,
        .pure_decode = false,
        .gather_logits = false,
        .mm = mm,
    };
    ForwardLogits fl = ModelRegistry::Forward(model, in);
    VT_CHECK(!fl.host.empty(), "gemma4 registered mm forward returned no host logits");
    return std::move(fl.host);
  };

  // ---- PREFILL: embed(prompt)*sqrt(H) + masked-scatter vision soft tokens ----
  std::vector<uint16_t> emb_bits = EmbedScaledBf16(d, weights, config, prompt_ids);
  std::vector<float> embeds = Bf16BitsToF32(emb_bits.data(), T0 * H);
  std::vector<float> proj_bf = mm_projected;
  RoundToBf16(proj_bf);  // vLLM casts the projector output to bf16 before merge.
  multimodal::Qwen3VLMergeMultimodal(embeds, T0, H, proj_bf, mask);
  std::vector<uint16_t> merged_bits = F32ToBf16Bits(embeds.data(), T0 * H);

  std::vector<int32_t> ple_prefill(static_cast<size_t>(T0));
  std::vector<int32_t> pos_prefill(static_cast<size_t>(T0));
  for (int64_t t = 0; t < T0; ++t) {
    ple_prefill[static_cast<size_t>(t)] = PleId(prompt_ids[static_cast<size_t>(t)],
                                                mask[static_cast<size_t>(t)],
                                                vocab_per_layer);
    pos_prefill[static_cast<size_t>(t)] = static_cast<int32_t>(t);
  }

  const CommonAttentionMetadata pm = StepMeta(T0, T0, 0);
  std::vector<float> logits = step(merged_bits, ple_prefill, pos_prefill, T0, pm);

  std::vector<int32_t> generated;
  DebugTop2(0, logits);
  if (out_margins != nullptr) out_margins->push_back(Top2Margin(logits));
  int32_t next = static_cast<int32_t>(ArgMax(logits));
  generated.push_back(next);

  // ---- DECODE: one token per step, 1-D position, no image (text PLE id) ----
  for (int dstep = 1; dstep < max_new_tokens; ++dstep) {
    if (next == eos_token_id) break;
    const int64_t abs_idx = T0 + (dstep - 1);  // seq index of the token being fed
    const std::vector<int32_t> one = {next};
    const std::vector<uint16_t> tok_emb = EmbedScaledBf16(d, weights, config, one);
    const std::vector<int32_t> ple1 = {PleId(next, false, vocab_per_layer)};
    const std::vector<int32_t> pos1 = {static_cast<int32_t>(abs_idx)};
    const int64_t seq_len = abs_idx + 1;
    const CommonAttentionMetadata dm = StepMeta(1, seq_len, abs_idx);
    logits = step(tok_emb, ple1, pos1, 1, dm);
    DebugTop2(dstep, logits);
    if (out_margins != nullptr) out_margins->push_back(Top2Margin(logits));
    next = static_cast<int32_t>(ArgMax(logits));
    generated.push_back(next);
  }
  return generated;
}

}  // namespace vllm
