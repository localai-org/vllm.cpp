// `NemotronH_Nano_VL_V2` / `NemotronH_Nano_Omni_Reasoning_V3` registry TU
// (row MODEL-MM-nano-nemotron-vl-nemotron-h-nano-vl-v2,
// .agents/specs/nano-nemotron-vl-radio.md). One translation unit with the
// REGISTER_VLLM_MODEL lines and no edit to any shared array, exactly as the
// NemotronH and dots3-note registries are.
//
// Ported from vLLM `e126687a9a`, vllm/model_executor/models/nano_nemotron_vl.py:
//   the class (:899-1010)       -> NanoNemotronVLLoadedModel + the loader
//   embed_multimodal (:1428-1460), image arm only
//                               -> EncodeMmNanoNemotronVL
//   embed_input_ids / the merge -> EmbedMmNanoNemotronVL
//   forward (:1462-1481)        -> ForwardNanoNemotronVL, which delegates to the
//                                  NemotronH paged forward with the merged rows
//   load_weights (:1499-1566)   -> LoadNanoNemotronVL
//
// THE LANGUAGE TOWER IS NemotronHForCausalLM's, UNCHANGED. It is the same
// loader (scoped to the `language_model.` prefix) and the same paged forward,
// which reads `ModelForwardInput::mm->inputs_embeds` on a multimodal step. The
// G-SAFE batched-decode refusal is inherited with it.
//
// REFUSED BY NAME: audio and video items at encode time (the checkpoint's
// `sound_*` tensors and the RADIO video embedder are accounted and deferred at
// load), the GGUF arm, and the static InternVL tiling arm (config parse).
#include "vllm/model_executor/models/model_registry.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/dense_attn_block.h"  // MakeTensor, DBuf, ResidentWeight
#include "vllm/model_executor/models/interfaces.h"        // SkipTowerForModalities
#include "vllm/model_executor/models/nano_nemotron_vl.h"
#include "vllm/model_executor/models/nemotron_h.h"
#include "vllm/model_executor/models/nemotron_h_forward.h"
#include "vllm/model_executor/models/nemotron_h_loader.h"
#include "vllm/model_executor/models/qwen3_5.h"           // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"    // HostLogits
#include "vllm/model_executor/models/qwen3_5_internal.h"  // detail::ApplyDeviceTokenIds
#include "vllm/model_executor/models/radio.h"
#include "vllm/multimodal/inputs.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// registry.py `_ModelInfo` for NemotronH_Nano_VL_V2 (nano_nemotron_vl.py:899-
// 901: HasInnerState, IsHybrid, SupportsMultiModal).
inline constexpr ModelInfo kNanoNemotronVLInfo{
    .is_text_generation_model = true,
    .is_pooling_model = false,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = true,
    .score_type = "bi-encoder",
};

constexpr const char* kLanguagePrefix = "language_model.";

class NanoNemotronVLLoadedModel final : public LoadedModel {
 public:
  NanoNemotronVLLoadedModel(const ModelRegistration& registration,
                            NanoNemotronVLParams params, NemotronHParams text)
      : LoadedModel(registration), params_(std::move(params)), text_(std::move(text)) {}

  const NanoNemotronVLParams& params() const { return params_; }
  const NemotronHParams& text() const { return text_; }
  NemotronHHostWeights& weights() { return weights_; }
  const NemotronHHostWeights& weights() const { return weights_; }
  NemotronHLoadReport& report() { return report_; }
  NanoNemotronVLVisionLoad& vision_load() { return vision_; }
  const NanoNemotronVLVisionLoad& vision_load() const { return vision_; }
  // `load_multimodal_weights` false (nano_nemotron_vl.py:1500-1504): every
  // modality has limit 0, so no tower was read.
  bool text_only() const { return text_only_; }
  void set_text_only(bool v) { text_only_ = v; }

  // The tower and the projector are built on the ENGINE's backend, which is
  // known only at `prepare` (the load hook has no queue).
  void BuildVision(vt::Backend& backend) {
    tower_ = std::make_unique<multimodal::RadioVisionTower>(vision_.radio, params_.radio,
                                                            backend);
    projector_ = std::make_unique<multimodal::NanoNemotronVLProjector>(
        vision_.projector, params_.projector, backend);
    // The host copies are dead once uploaded.
    vision_.radio = {};
    vision_.projector = {};
  }
  const multimodal::RadioVisionTower* tower() const { return tower_.get(); }
  const multimodal::NanoNemotronVLProjector* projector() const { return projector_.get(); }

 private:
  NanoNemotronVLParams params_;
  NemotronHParams text_;
  NemotronHHostWeights weights_;
  NemotronHLoadReport report_;
  NanoNemotronVLVisionLoad vision_;
  std::unique_ptr<multimodal::RadioVisionTower> tower_;
  std::unique_ptr<multimodal::NanoNemotronVLProjector> projector_;
  bool text_only_ = false;
};

void ParseNanoNemotronVLConfig(const HfConfig& config) {
  (void)ParseNanoNemotronVLParams(config);
  (void)ParseNemotronHParams(NanoNemotronVLTextConfig(config));
}

std::unique_ptr<LoadedModel> LoadNanoNemotronVL(const ModelRegistration& registration,
                                                const HfConfig& config,
                                                const ModelSource& source) {
  const std::string arch(registration.architecture);
  if (source.kind != ModelSource::Kind::kSafetensors || source.safetensors == nullptr) {
    throw std::runtime_error(
        "Model architecture " + arch +
        " loads safetensors only: the GGUF arm of its NemotronH language tower "
        "is owed (.agents/specs/nemotron-h-model.md §5b W7) and no GGUF vision "
        "projector arm exists");
  }
  NanoNemotronVLParams params = ParseNanoNemotronVLParams(config);
  const HfConfig text_cfg = NanoNemotronVLTextConfig(config);
  NemotronHParams text = ParseNemotronHParams(text_cfg);
  auto model = std::make_unique<NanoNemotronVLLoadedModel>(registration, std::move(params),
                                                           std::move(text));
  const std::vector<SafetensorsFile>& shards = *source.safetensors;

  // The language tower: NemotronH's own loader, scoped to its prefix
  // (hf_to_vllm_mapper, nano_nemotron_vl.py:904-908). It accounts for every
  // `language_model.*` tensor in both directions.
  model->weights() =
      LoadNemotronHHostWeights(shards, model->text(), ResolveNemotronHModelDType(text_cfg),
                               &model->report(), kLanguagePrefix);
  // The tower and the projector, unless every modality has limit 0
  // (`load_multimodal_weights`, nano_nemotron_vl.py:1500-1504): upstream then
  // skips `mlp1`, `vision_model.*` and `sound*` entirely.
  const bool text_only = SkipTowerForModalities(source.multimodal, {"image", "video", "audio"});
  model->set_text_only(text_only);
  if (!text_only) {
    model->vision_load() = LoadNanoNemotronVLVisionWeights(shards, model->params());
  }

  // Everything else the checkpoint ships must be NAMED: the audio encoder and
  // its projection are deferred (audio is not ported), and nothing may be left
  // unaccounted.
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      if (name.rfind(kLanguagePrefix, 0) == 0 || name.rfind("vision_model.", 0) == 0 ||
          name.rfind("mlp1.", 0) == 0) {
        continue;
      }
      if (name.rfind("sound", 0) == 0) {
        // `is_sound_weights` is the bare `sound` prefix (:1516-1517), and a
        // sound tensor on a checkpoint whose config builds no sound encoder is
        // an error there (`assert self.sound_encoder is not None`, :1546-1548).
        if (!text_only && !model->params().has_sound) {
          throw std::runtime_error("Model architecture " + arch + ": the checkpoint ships '" +
                                   name + "' but config.json carries no `sound_config`, "
                                   "so no sound encoder exists to receive it");
        }
        model->vision_load().deferred.push_back(name + " (audio, not ported)");
        continue;
      }
      throw std::runtime_error("Model architecture " + arch + ": the checkpoint ships '" +
                               name +
                               "', which neither the language tower, the vision "
                               "tower, mlp1 nor the deferred audio arm names");
    }
  }
  return model;
}

void PrepareNanoNemotronVL(LoadedModel& model, const HfConfig& config, vt::Queue& queue) {
  (void)config;
  auto& m = ModelAs<NanoNemotronVLLoadedModel>(model, "NemotronH_Nano_VL_V2");
  if (!m.text_only() && m.tower() == nullptr) m.BuildVision(vt::GetBackend(queue.device.type));
}

ForwardLogits ForwardNanoNemotronVL(LoadedModel& model, const ModelForwardInput& input) {
  // Inherited from NemotronHForCausalLM (G-SAFE, #810): the paged forward
  // carries one request's KV pages and recurrent state per step.
  VT_CHECK(input.num_reqs <= 1,
           "Model architecture NemotronH_Nano_VL_V2: BATCHED decode is not ported "
           "(inherited from NemotronHForCausalLM, issue #810, "
           ".agents/specs/nemotron-h-a2p-paged-forward.md A2-B). Refusing by name "
           "rather than decoding a multi-request step as one sequence.");
  auto& m = ModelAs<NanoNemotronVLLoadedModel>(model, "NemotronH_Nano_VL_V2");
  if (!input.attn_kv.empty() && !input.gdn_state.empty() && m.weights().materialized) {
    return NemotronHPagedForward(m.weights(), m.text(), input);
  }
  // The host reference, for a direct caller with no paged caches.
  std::vector<float> embeds;
  const std::vector<float>* embeds_ptr = nullptr;
  if (input.mm.has_value() && input.mm->inputs_embeds.data != nullptr) {
    const vt::Tensor& e = input.mm->inputs_embeds;
    const size_t n = static_cast<size_t>(e.shape[0] * e.shape[1]);
    embeds.resize(n);
    vt::Backend& b = vt::GetBackend(e.device.type);
    if (e.dtype == vt::DType::kBF16) {
      std::vector<uint16_t> bits(n);
      b.Copy(input.queue, bits.data(), e.data, n * sizeof(uint16_t));
      b.Synchronize(input.queue);
      for (size_t i = 0; i < n; ++i) embeds[i] = vt::BF16ToF32(bits[i]);
    } else {
      b.Copy(input.queue, embeds.data(), e.data, n * sizeof(float));
      b.Synchronize(input.queue);
    }
    embeds_ptr = &embeds;
  }
  return HostLogits(NemotronHForward(m.weights(), m.text(), input.token_ids,
                                     input.logits_indices, input.queue, nullptr, embeds_ptr),
                    m.text().vocab_size);
}

// embed_multimodal, image arm (nano_nemotron_vl.py:1428-1460 ->
// _process_image_input_dynamic :1130-1142 -> extract_feature_dynamic
// :1050-1058): tower -> bf16 -> pixel shuffle -> mlp1, one item at a time.
MmEncoderOutput EncodeMmNanoNemotronVL(LoadedModel& model, const HfConfig& config,
                                       vt::Queue& queue,
                                       const multimodal::MultiModalFeatureSpec& item) {
  (void)config;
  auto& m = ModelAs<NanoNemotronVLLoadedModel>(model, "NemotronH_Nano_VL_V2");
  VT_CHECK(item.modality == "image",
           "Model architecture NemotronH_Nano_VL_V2: modality '" + item.modality +
               "' is not ported. IMAGE is; AUDIO (the Parakeet sound encoder, "
               "nano_nemotron_vl.py:1254-1283) and VIDEO (the temporal RADIO "
               "embedder and EVS, :1163-1252) are refused by name. See "
               ".agents/specs/nano-nemotron-vl-radio.md.");
  VT_CHECK(!m.text_only(),
           "Model architecture NemotronH_Nano_VL_V2: this engine was loaded "
           "text-only (every modality's limit is 0), so no vision tower was read");
  VT_CHECK(item.data != nullptr && item.data->num_patches > 0,
           "Model architecture NemotronH_Nano_VL_V2: the image item carries no "
           "processed pixels (MultiModalFeatureSpec::data)");
  if (m.tower() == nullptr) m.BuildVision(vt::GetBackend(queue.device.type));
  const NanoNemotronVLParams& p = m.params();
  const multimodal::ImageKwargs& kw = *item.data;
  const int64_t gh = kw.image_grid_thw[1];
  const int64_t gw = kw.image_grid_thw[2];
  VT_CHECK(kw.patch_feature_dim == p.radio.patch_dim() && gh * gw == kw.num_patches &&
               static_cast<int64_t>(kw.pixel_values_f32.size()) ==
                   kw.num_patches * kw.patch_feature_dim,
           "Model architecture NemotronH_Nano_VL_V2: the image item's patch grid "
           "does not match its pixel rows");

  multimodal::RadioImage im;
  im.patches = kw.pixel_values_f32;
  im.grid_h = gh;
  im.grid_w = gw;
  std::vector<float> feats = m.tower()->Forward({im});
  // `vit_embeds.to(dtype=torch.bfloat16)` (:1055) before the shuffle.
  for (float& v : feats) v = vt::BF16ToF32(vt::F32ToBF16(v));
  const std::vector<float> shuffled = multimodal::NanoNemotronVLPixelShuffle(
      feats, gh, gw, p.radio.hidden_size, p.reduction);
  const int64_t rows = (gh / p.reduction) * (gw / p.reduction);
  VT_CHECK(rows == static_cast<int64_t>(item.length),
           "Model architecture NemotronH_Nano_VL_V2: the projector produces " +
               std::to_string(rows) + " rows for a placeholder span of " +
               std::to_string(item.length) +
               " tokens; the processor's expansion and the tower disagree");
  const std::vector<float> out = m.projector()->Forward(shuffled, rows);

  const vt::DType adt = m.weights().act_dtype;
  vt::Backend& backend = vt::GetBackend(queue.device.type);
  const int64_t H = p.projector.out_dim;
  const size_t bytes = static_cast<size_t>(rows * H) * vt::SizeOf(adt);
  void* ptr = backend.Alloc(bytes);
  std::shared_ptr<void> storage(ptr, [&backend](void* q) { backend.Free(q); });
  if (adt == vt::DType::kBF16) {
    std::vector<uint16_t> bits(out.size());
    for (size_t i = 0; i < out.size(); ++i) bits[i] = vt::F32ToBF16(out[i]);
    backend.Copy(queue, ptr, bits.data(), bytes);
  } else {
    backend.Copy(queue, ptr, out.data(), bytes);
  }
  backend.Synchronize(queue);
  MmEncoderOutput enc;
  enc.storage = std::move(storage);
  enc.embeds = dense_attn::MakeTensor(ptr, adt, queue.device, {rows, H});
  return enc;
}

// embed_input_ids + the masked scatter (interfaces.py _merge_multimodal_
// embeddings): embed every id, then overwrite the `<image>` rows with the
// gathered encoder rows in mask order. A pure copy in the model dtype.
MmForwardBuffers EmbedMmNanoNemotronVL(LoadedModel& model, const HfConfig& config,
                                       vt::Queue& queue, const MmEmbedInputs& inputs) {
  (void)config;
  auto& m = ModelAs<NanoNemotronVLLoadedModel>(model, "NemotronH_Nano_VL_V2");
  const NemotronHHostWeights& w = m.weights();
  VT_CHECK(w.materialized, "NemotronH_Nano_VL_V2 embed: the language tower is not loaded");
  VT_CHECK(inputs.token_ids != nullptr && inputs.is_mm_embed != nullptr &&
               inputs.mm_embeds != nullptr,
           "NemotronH_Nano_VL_V2 embed: the runner passed a null MmEmbedInputs channel");
  const std::vector<int32_t>& ids_host = *inputs.token_ids;
  const int64_t T = static_cast<int64_t>(ids_host.size());
  const int64_t H = m.text().hidden_size;
  const int64_t V = m.text().vocab_size;
  const vt::DType adt = w.act_dtype;
  VT_CHECK(T > 0, "NemotronH_Nano_VL_V2 embed: empty step");
  VT_CHECK(static_cast<int64_t>(inputs.is_mm_embed->size()) == T,
           "NemotronH_Nano_VL_V2 embed: is_mm_embed does not cover the step");

  vt::Backend& backend = vt::GetBackend(queue.device.type);
  dense_attn::Dev d{backend, queue};
  const size_t esz = vt::SizeOf(adt);
  std::vector<uint8_t> merged(static_cast<size_t>(T * H) * esz);
  {
    for (int32_t id : ids_host) {
      VT_CHECK(id >= 0 && id < V, "NemotronH_Nano_VL_V2 embed: token id out of range");
    }
    dense_attn::DBuf ids(d, vt::DType::kI32, {T}, ids_host.data());
    // ENG-MM-EMBED-DEVICE-IDS (#2730): splice the runner's device identifiers
    // over the (stale on decode rows) host upload before the gather.
    detail::ApplyDeviceTokenIds(d.b, d.q, ids.ptr(), T,
                                detail::DeviceTokenIds{inputs.device_token_ids, T},
                                "nano-nemotron-vl mm embed");
    dense_attn::DBuf emb(d, adt, {T, H});
    vt::Tensor table = dense_attn::ResidentWeight(d, w.embeddings, {V, H});
    vt::Embedding(d.q, emb.t(), table, ids.t());
    emb.Download(d, merged.data());
  }

  int64_t n_rows = 0;
  for (const vt::Tensor& slice : *inputs.mm_embeds) {
    VT_CHECK(slice.rank == 2 && slice.shape[1] == H && slice.dtype == adt,
             "NemotronH_Nano_VL_V2 embed: a gathered encoder slice is not "
             "[rows, hidden] in the model dtype");
    n_rows += slice.shape[0];
  }
  int64_t n_masked = 0;
  for (char c : *inputs.is_mm_embed) n_masked += c != 0 ? 1 : 0;
  VT_CHECK(n_rows == n_masked,
           "NemotronH_Nano_VL_V2 embed: " + std::to_string(n_rows) +
               " encoder rows for " + std::to_string(n_masked) +
               " masked placeholder positions");
  if (n_rows > 0) {
    std::vector<uint8_t> gathered(static_cast<size_t>(n_rows * H) * esz);
    size_t off = 0;
    for (const vt::Tensor& slice : *inputs.mm_embeds) {
      const size_t n = static_cast<size_t>(slice.shape[0] * H) * esz;
      backend.Copy(queue, gathered.data() + off, slice.data, n);
      off += n;
    }
    backend.Synchronize(queue);
    int64_t r = 0;
    const size_t row = static_cast<size_t>(H) * esz;
    for (int64_t t = 0; t < T; ++t) {
      if ((*inputs.is_mm_embed)[static_cast<size_t>(t)] == 0) continue;
      std::memcpy(merged.data() + static_cast<size_t>(t) * row,
                  gathered.data() + static_cast<size_t>(r) * row, row);
      ++r;
    }
  }

  MmForwardBuffers out;
  void* p = backend.Alloc(merged.size());
  out.storage.emplace_back(p, [&backend](void* q) { backend.Free(q); });
  backend.Copy(queue, p, merged.data(), merged.size());
  out.mm.inputs_embeds = dense_attn::MakeTensor(p, adt, queue.device, {T, H});
  // No positions3: NemotronH has no positional embedding at all, and upstream
  // does not declare SupportsMRoPE for this class.
  backend.Synchronize(queue);
  return out;
}

v1::KVCacheConfig MakeNanoNemotronVLKVCache(const HfConfig& config, int block_size,
                                            int num_blocks) {
  return MakeNemotronHKVCache(NanoNemotronVLTextConfig(config), block_size, num_blocks);
}

const ModelFactory kNanoNemotronVLFactory{
    .parse_config = &ParseNanoNemotronVLConfig,
    .load_weights = &LoadNanoNemotronVL,
    .prepare = &PrepareNanoNemotronVL,
    .forward = &ForwardNanoNemotronVL,
    .make_kv_cache = &MakeNanoNemotronVLKVCache,
    .encode_mm = &EncodeMmNanoNemotronVL,
    .embed_mm = &EmbedMmNanoNemotronVL,
    .is_dense_model = false,
    // The language forward is NemotronH's, which honours device identifiers
    // on a text step; the embed hook splices them on a multimodal step.
    .consumes_device_token_ids = true,
    .embed_mm_consumes_device_token_ids = true,
};

}  // namespace

const NanoNemotronVLVisionLoad& NanoNemotronVLVisionLoadOf(const LoadedModel& model) {
  const auto* m = dynamic_cast<const NanoNemotronVLLoadedModel*>(&model);
  if (m == nullptr) {
    throw std::runtime_error(
        "NanoNemotronVLVisionLoadOf: this LoadedModel is not a NemotronH_Nano_VL_V2 model");
  }
  return m->vision_load();
}

// Upstream maps four names to this class (registry.py:512-515 @ e126687a9a).
// `NemotronH_Super_Omni_Reasoning_V3` and `NemotronH_Omni_Reasoning_V3` have no
// public checkpoint at the pin and are not registered; they are owed in
// .agents/specs/nano-nemotron-vl-radio.md.
REGISTER_VLLM_MODEL(nano_nemotron_vl_v2, "NemotronH_Nano_VL_V2", kNanoNemotronVLFactory,
                    kNanoNemotronVLInfo)
REGISTER_VLLM_MODEL(nano_nemotron_omni_v3, "NemotronH_Nano_Omni_Reasoning_V3",
                    kNanoNemotronVLFactory, kNanoNemotronVLInfo)

}  // namespace vllm
