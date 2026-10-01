// Muse Glimmer (`MuseGlimmerForConditionalGeneration`) W4 WIRING — the seam that
// makes the perception encoder REACHABLE from the model forward. Additive TU: the
// W3 tower (muse_glimmer_vision.cpp) and the W1 text tower (muse_glimmer.cpp) are
// untouched; this file is the plumbing between them plus the registered mm driver.
//
// ─── OFF-PIN HONESTY (up front) ──────────────────────────────────────────────
// Every `muse_glimmer.py:NNNN` below is vllm#51655 head `075d645af`, an OPEN,
// CI-red upstream PR — NOT the parity pin `555967922`, which has no muse_glimmer
// at all (porting-inventory §9 deviation 16, specs/muse-glimmer.md §0). The pinned
// oracle cannot load this model, so there is NO golden decode to compare against
// and NO throughput denominator: **no speed axis is claimable here, and neither is
// image or video end-to-end correctness.** What this file establishes is that the
// tower runs, its output is projected into the text hidden space, and it lands on
// the placeholder rows — the plumbing, not the answer.
//
// ─── WHAT THIS IS A PORT OF ──────────────────────────────────────────────────
//   OURS                                <-  UPSTREAM (muse_glimmer.py)
//   MuseGlimmerVisionConfigOf           <-  :1457-1468 (tower construction dims)
//   MuseGlimmerMultimodalMask           <-  :1492-1498 (configure_mm_token_handling)
//   MuseGlimmerEncodePixelGroups        <-  :1548-1569 (_encode_pixel_groups)
//                                           + :1571-1590 (_process_{image,video}_input)
//   MuseGlimmerMergeMultimodalEmbeds    <-  :1301-1302 (embed_input_ids) + the
//                                           SupportsMultiModal masked scatter
//   MuseGlimmerGenerateGreedyViaRegistry <- the vLLM engine's mm decode loop,
//                                           folded exactly as gemma4_mm.cpp does
//
// ─── THE THREE PLACES THIS STAYS PLAUSIBLE WHEN WRONG ────────────────────────
//   1. THE PROJECTOR ORDER. It is encoder -> adapter -> vision_projection
//      (:1560-1562), and the adapter is gelu(c_proj(gelu(c_fc(x)))) with the
//      SECOND gelu on its output (:1041-1044). Every intermediate is [N, d] with
//      compatible-looking dims once adapter_dim appears twice, so a swapped pair
//      still runs.
//   2. THE NORM ASYMMETRY. Text rows get `embed_norm` (:1302); the soft tokens do
//      NOT — they carry `perception_emb_norm`, which is Identity unless
//      `normalize_tok_embeddings` (:1469-1473). Normalizing the soft tokens too
//      changes nothing structural and everything numerically.
//   3. THE SCATTER MASK. Image (200092) AND video (200091) are both placeholders
//      feeding ONE soft-token stream (:1592-1602). Masking only the image token
//      leaves video rows holding a text embedding of a token that has no text
//      meaning — coherent-looking output, wrong grounding.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // Dev/DBuf/ResidentWeight
#include "vllm/model_executor/models/host_embedding.h"  // VT_HOST_EMBEDDING gather
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/muse_glimmer.h"
#include "vllm/model_executor/models/qwen3_5.h"        // PagedKvCache, GdnStateCache
#include "vllm/model_executor/models/qwen3_vl_text.h"  // Qwen3VLMergeMultimodal (shared scatter)
#include "vllm/v1/attention/backends/gdn_attn.h"       // v1::GDNAttentionMetadata
#include "vt/backend.h"
#include "vt/dtype.h"
#include "vt/ops.h"

namespace vllm {
namespace {

using vt::DType;
using vt::Tensor;
using v1::CommonAttentionMetadata;
using namespace dense_attn;  // Dev, DBuf, ResidentWeight

constexpr const char* kNoTower =
    "MuseGlimmer mm forward: this checkpoint has no perception encoder "
    "(MuseGlimmerWeights::vision.loaded == false). A text-only Muse Glimmer "
    "checkpoint ships no `vision_config` and no vision tensors; an image or video "
    "prompt cannot be served from it. See .agents/specs/muse-glimmer.md §1.2.";

// #607 L3. The analogue of `StageMissingLayer.__call__` raising
// `f"{self} should not be called"` (vllm/model_executor/models/utils.py:700-701):
// the perception encoder WAS constructed, its geometry is known, and its weights
// were deliberately not read because every modality it serves was at limit 0
// (interfaces.py:288-293). Reaching this is a bug, not a user error — the L1
// refusal answers a zero-limit multimodal request at the entrypoint long before
// the tower — so the message names the configuration that produced the state
// rather than pretending the checkpoint is at fault.
constexpr const char* kSkippedTower =
    "MuseGlimmer mm forward: the perception encoder was SKIPPED at load because "
    "every modality it serves is at limit 0 (--language-model-only, or "
    "--limit-mm-per-prompt with image and video both 0). The checkpoint does "
    "carry an encoder; drop the zero limit and reload to serve image or video. "
    "Reaching this point at all is a defect: a zero-limit multimodal request is "
    "refused at the entrypoint (#607 L1). See .agents/specs/multimodal-track.md "
    "§1.5 L3.";

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

// One-sequence CommonAttentionMetadata for a step (mirror gemma4_mm.cpp StepMeta).
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

// `vision_projection` (:1464-1468): a bias-free Linear(adapter_dim, text_hidden)
// over the adapter output. Run on the device at the tower's compute dtype, the
// same regime MuseGlimmerVisionAdapterForward uses, so the projector is not a
// silently-higher-precision island in the middle of a bf16 tower.
std::vector<float> ProjectToTextHidden(const std::vector<float>& features,
                                       int64_t num_tokens, int64_t adapter_dim,
                                       int64_t text_hidden,
                                       const std::vector<float>& projection,
                                       DType dt, vt::Queue& queue) {
  VT_CHECK(static_cast<int64_t>(features.size()) == num_tokens * adapter_dim,
           "muse_glimmer vision_projection: adapter output has the wrong shape");
  VT_CHECK(static_cast<int64_t>(projection.size()) == text_hidden * adapter_dim,
           "muse_glimmer vision_projection: weight has the wrong shape");
  Dev d{vt::GetBackend(queue.device.type), queue};

  auto upload = [&](const std::vector<int64_t>& shape,
                    const std::vector<float>& host) -> DBuf {
    if (dt == DType::kBF16) {
      std::vector<uint16_t> bits(host.size());
      for (size_t i = 0; i < host.size(); ++i) bits[i] = vt::F32ToBF16(host[i]);
      return DBuf(d, dt, shape, bits.data());
    }
    return DBuf(d, dt, shape, host.data());
  };

  DBuf x = upload({num_tokens, adapter_dim}, features);
  DBuf w = upload({text_hidden, adapter_dim}, projection);
  DBuf out(d, dt, {num_tokens, text_hidden});
  vt::MatmulBT(d.q, out.t(), x.t(), w.t());

  std::vector<float> host(static_cast<size_t>(num_tokens * text_hidden));
  if (dt == DType::kBF16) {
    std::vector<uint16_t> bits(host.size());
    out.Download(d, bits.data());
    for (size_t i = 0; i < host.size(); ++i) host[i] = vt::BF16ToF32(bits[i]);
  } else {
    out.Download(d, host.data());
  }
  return host;
}

// `perception_emb_norm` (:1469-1473): a WEIGHTLESS RMSNorm over the soft tokens,
// applied ONLY when the text config sets `normalize_tok_embeddings` — `nn.Identity`
// otherwise, which is the released 30B's case. Computed in f32, matching
// MuseGlimmerRMSNorm's `hidden_states.float()` (:540-552).
void PerceptionEmbNorm(std::vector<float>& rows, int64_t num_tokens, int64_t H,
                       float eps) {
  for (int64_t r = 0; r < num_tokens; ++r) {
    float* row = rows.data() + static_cast<size_t>(r * H);
    double acc = 0.0;
    for (int64_t i = 0; i < H; ++i) acc += static_cast<double>(row[i]) * row[i];
    const float inv = static_cast<float>(1.0 / std::sqrt(acc / static_cast<double>(H) +
                                                         static_cast<double>(eps)));
    for (int64_t i = 0; i < H; ++i) row[i] *= inv;
  }
}

}  // namespace

multimodal::MuseGlimmerVisionConfig MuseGlimmerVisionConfigOf(
    const MuseGlimmerParams& params) {
  const MuseGlimmerVisionParams& v = params.vision;
  multimodal::MuseGlimmerVisionConfig cfg;
  cfg.hidden_size = v.hidden_size;
  cfg.num_attention_heads = v.num_attention_heads;
  cfg.num_hidden_layers = v.num_hidden_layers;
  cfg.intermediate_size = v.intermediate_size;
  cfg.patch_size = v.patch_size;
  cfg.patch_temporal = v.patch_temporal;
  cfg.merge_kernel_size = v.merge_kernel_size;
  cfg.pos_emb_height = v.pos_emb_height;
  cfg.pos_emb_width = v.pos_emb_width;
  cfg.output_dim = v.output_dim;
  cfg.adapter_dim = v.adapter_dim;
  cfg.layer_norm_eps = v.layer_norm_eps;
  cfg.layer_types = v.layer_types;
  return cfg;
}

std::vector<bool> MuseGlimmerMultimodalMask(const std::vector<int32_t>& token_ids,
                                            const MuseGlimmerParams& params) {
  const int32_t image = static_cast<int32_t>(params.image_token_id);
  const int32_t video = static_cast<int32_t>(params.video_token_id);
  std::vector<bool> mask(token_ids.size(), false);
  for (size_t i = 0; i < token_ids.size(); ++i)
    mask[i] = token_ids[i] == image || token_ids[i] == video;
  return mask;
}

std::vector<float> MuseGlimmerEncodePixelGroups(
    const std::vector<multimodal::MuseGlimmerVisionImage>& images,
    const MuseGlimmerWeights& weights, vt::Queue& queue) {
  // #607 L3: the two ways this tower can be absent have DIFFERENT fixes, so they
  // get different messages. `vision_skipped` is upstream's StageMissingLayer
  // being called (utils.py:700-701) — the checkpoint HAS an encoder and the
  // operator asked for zero limits — and the fix is the configuration. `!loaded`
  // without a skip is a text-only checkpoint, and no configuration repairs that.
  // Checked first because a skipped tower is also an unloaded one.
  VT_CHECK(!weights.vision_skipped, kSkippedTower);
  VT_CHECK(weights.vision.loaded, kNoTower);
  VT_CHECK(!images.empty(), "muse_glimmer mm forward: no pixel groups to encode");
  const MuseGlimmerVisionTower& tower = weights.vision;
  const int64_t A = tower.cfg.adapter_dim;
  const int64_t H = weights.params.text.hidden_size;
  vt::Backend& backend = vt::GetBackend(queue.device.type);

  // encoder -> adapter -> projection, in that order (:1560-1562).
  std::vector<float> features =
      multimodal::MuseGlimmerVisionForward(images, tower.encoder, tower.cfg, backend);
  const int64_t n_tokens =
      static_cast<int64_t>(features.size()) / (tower.cfg.output_dim > 0
                                                   ? tower.cfg.output_dim
                                                   : 1);
  VT_CHECK(n_tokens > 0 && static_cast<int64_t>(features.size()) ==
                               n_tokens * tower.cfg.output_dim,
           "muse_glimmer: perception encoder output is not a whole number of tokens");
  features = multimodal::MuseGlimmerVisionAdapterForward(features, n_tokens,
                                                         tower.adapter, tower.cfg,
                                                         backend);
  std::vector<float> soft = ProjectToTextHidden(features, n_tokens, A, H,
                                                tower.projection,
                                                tower.cfg.compute_dtype, queue);
  if (weights.params.text.normalize_tok_embeddings)
    PerceptionEmbNorm(soft, n_tokens, H, weights.params.text.rms_norm_eps);
  return soft;
}

std::vector<uint16_t> MuseGlimmerMergeMultimodalEmbeds(
    const std::vector<int32_t>& token_ids, const std::vector<float>& mm_embeds,
    const MuseGlimmerWeights& weights, vt::Queue& queue) {
  const MuseGlimmerTextParams& t = weights.params.text;
  const int64_t H = t.hidden_size;
  const int64_t T = static_cast<int64_t>(token_ids.size());
  VT_CHECK(weights.text_loaded,
           "muse_glimmer mm forward: the text tower is not materialized");
  VT_CHECK(T > 0, "muse_glimmer mm forward: empty prompt");

  // `embed_input_ids` (:1301-1302): embed_tokens THEN the WEIGHTLESS embed_norm.
  // Byte-for-byte the same two ops the text ForwardBody runs, so a prompt with no
  // placeholder rows produces exactly the text path's hidden stream.
  Dev d{vt::GetBackend(queue.device.type), queue};
  std::vector<uint16_t> bits(static_cast<size_t>(T * H));
  {
    std::vector<uint16_t> ones_host(static_cast<size_t>(H), vt::F32ToBF16(1.0f));
    DBuf ones(d, DType::kBF16, {H}, ones_host.data());
    DBuf emb(d, DType::kBF16, {T, H});
    DBuf normed(d, DType::kBF16, {T, H});
    // This function downloads the normed rows anyway, so the host gather also
    // skips building the device table on this arm.
    EmbedGather(d, emb, token_ids, weights.embed_tokens, t.vocab_size, H,
                "muse_glimmer mm embed");
    vt::RmsNorm(d.q, normed.t(), emb.t(), ones.t(),
                vt::RmsNormArgs{t.rms_norm_eps, /*gemma=*/false});
    normed.Download(d, bits.data());
  }
  if (mm_embeds.empty()) return bits;

  // The masked scatter, through the SHARED multimodal seam every VLM here uses.
  // bf16 -> f32 -> bf16 round-trips every untouched row exactly, so the text rows
  // survive bit-for-bit and only the placeholder rows change.
  const std::vector<bool> mask = MuseGlimmerMultimodalMask(token_ids, weights.params);
  int64_t n_slots = 0;
  for (bool m : mask) n_slots += m ? 1 : 0;
  VT_CHECK(n_slots > 0,
           "muse_glimmer mm forward: multimodal embeddings were supplied but the "
           "prompt carries no image (200092) or video (200091) placeholder token");
  const int64_t n_rows = static_cast<int64_t>(mm_embeds.size()) / H;
  VT_CHECK(n_rows * H == static_cast<int64_t>(mm_embeds.size()) && n_rows == n_slots,
           "muse_glimmer mm forward: produced " + std::to_string(n_rows) +
               " vision features for " + std::to_string(n_slots) +
               " placeholder tokens");

  std::vector<float> embeds(static_cast<size_t>(T * H));
  for (size_t i = 0; i < embeds.size(); ++i) embeds[i] = vt::BF16ToF32(bits[i]);
  std::vector<float> soft = mm_embeds;
  // vLLM casts the tower output to the model dtype before the merge.
  for (float& x : soft) x = vt::BF16ToF32(vt::F32ToBF16(x));
  multimodal::Qwen3VLMergeMultimodal(embeds, T, H, soft, mask);
  for (size_t i = 0; i < embeds.size(); ++i) bits[i] = vt::F32ToBF16(embeds[i]);
  return bits;
}

std::vector<int32_t> MuseGlimmerGenerateGreedyViaRegistry(
    LoadedModel& model, const std::vector<int32_t>& prompt_ids,
    const std::vector<multimodal::MuseGlimmerVisionImage>& images,
    int32_t eos_token_id, const MuseGlimmerWeights& weights, const HfConfig& config,
    vt::Queue& queue, int max_new_tokens) {
  const MuseGlimmerTextParams& t = weights.params.text;
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  Dev d{backend, queue};
  const int64_t T0 = static_cast<int64_t>(prompt_ids.size());
  const int64_t L = t.num_hidden_layers;
  const int64_t Hkv = t.num_key_value_heads;
  const int64_t Dh = t.head_dim;

  // Uniform KV geometry across RoPE/NoPE layers — only the WINDOW differs, and it
  // is applied at the attention kernel (see MakeMuseGlimmerKVCache).
  const int64_t block_size = T0 + max_new_tokens + 8;
  std::vector<std::shared_ptr<void>> kv_storage;
  std::vector<PagedKvCache> attn_kv;
  for (int64_t l = 0; l < L; ++l) {
    const size_t kv_bytes =
        static_cast<size_t>(2 * block_size * Hkv * Dh) * vt::SizeOf(DType::kBF16);
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

  // Every step goes through ModelRegistry::Forward with `mm` set, so the ENGINE's
  // registered mm branch drives decode (the Gemma-4 / Qwen3-VL fold).
  auto step = [&](const std::vector<uint16_t>& embeds,
                  const std::vector<int32_t>& pos1d, int64_t T,
                  const CommonAttentionMetadata& meta) -> std::vector<float> {
    const std::vector<int32_t> no_tokens;
    std::vector<GdnStateCache> no_gdn_state;
    v1::GDNAttentionMetadata gdn_meta{};
    const std::vector<int32_t> gather_li = {static_cast<int32_t>(T - 1)};
    // ENG-MM-INPUT-PIPELINE P1: the seam takes a DEVICE handle, so this driver —
    // whose merge runs on the host — uploads HERE. `dembeds` outlives the
    // ModelRegistry::Forward call below. The runner slice (#2300) will hand the
    // seam a persistent device buffer and skip the upload.
    DBuf dembeds(d, DType::kBF16, {T, t.hidden_size}, embeds.data());
    MultiModalForwardInput mm{};
    mm.inputs_embeds = dembeds.t();
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
    VT_CHECK(!fl.host.empty(),
             "muse_glimmer registered mm forward returned no host logits");
    return std::move(fl.host);
  };

  // PREFILL: encode the pixel groups, then embed_input_ids + masked scatter.
  const std::vector<float> soft =
      images.empty() ? std::vector<float>()
                     : MuseGlimmerEncodePixelGroups(images, weights, queue);
  const std::vector<uint16_t> merged =
      MuseGlimmerMergeMultimodalEmbeds(prompt_ids, soft, weights, queue);
  std::vector<int32_t> pos_prefill(static_cast<size_t>(T0));
  for (int64_t i = 0; i < T0; ++i) pos_prefill[static_cast<size_t>(i)] = static_cast<int32_t>(i);
  std::vector<float> logits = step(merged, pos_prefill, T0, StepMeta(T0, T0, 0));

  std::vector<int32_t> generated;
  int32_t next = static_cast<int32_t>(ArgMax(logits));
  generated.push_back(next);

  // DECODE: one token per step, 1-D positions, no image rows (so the merge is
  // exactly embed_input_ids for the sampled token).
  for (int s = 1; s < max_new_tokens; ++s) {
    if (next == eos_token_id) break;
    const int64_t abs_idx = T0 + (s - 1);
    const std::vector<int32_t> one = {next};
    const std::vector<uint16_t> tok_emb =
        MuseGlimmerMergeMultimodalEmbeds(one, {}, weights, queue);
    const std::vector<int32_t> pos1 = {static_cast<int32_t>(abs_idx)};
    logits = step(tok_emb, pos1, 1, StepMeta(1, abs_idx + 1, abs_idx));
    next = static_cast<int32_t>(ArgMax(logits));
    generated.push_back(next);
  }
  return generated;
}

}  // namespace vllm
