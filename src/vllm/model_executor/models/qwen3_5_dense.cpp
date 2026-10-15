// Qwen3.6 DENSE (27B) architecture registry TU. Self-registers
// "Qwen3_5ForConditionalGeneration" via REGISTER_VLLM_MODEL and owns the dense
// arch-specific entry points (LoadedModel subclass + load/prepare/forward
// wrappers + factory + synthetic Make/Borrow adapters). The heavy dense forward
// machinery (Qwen3_5DenseModel::/Qwen3_5DenseDecodeGraph::) lives in qwen3_5.cpp
// over the shared DevicePool/matmul/GDN helpers; this TU only wires it into the
// registry. Extracted verbatim (behavior-preserving) from the former
// model_registry.cpp monolith.
#include "vllm/v1/worker/gpu/cudagraph_dispatch.h"
#include "vllm/model_executor/models/model_registry.h"

#include <cstdio>
#include <algorithm>
#include <atomic>
#include <cstdlib>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/models/interfaces.h"  // #607 L3 kVisionTowerStageName

#include "vllm/model_executor/models/qwen3_5.h"         // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"  // kQwen3_5Info, helpers
#include "vllm/model_executor/models/qwen3_5_dense.h"
#include "vllm/model_executor/models/qwen3_5_dense_mm.h"
#include "vllm/model_executor/models/dense_device_glue.h"
#include "vllm/model_executor/models/qwen3_vl_text.h"
#include "vllm/model_executor/layers/quantization/exl3_checkpoint.h"
#include "vllm/model_executor/models/qwen3_5_gguf_weights.h"
#include "qwen3_5_internal.h"  // W4 DeviceTokenIdsScope
#include "vllm/model_executor/models/qwen3_5_mtp.h"  // SPEC-MTP I5d-pre draft
#include "vllm/platforms/interface.h"  // GetPlatform(device.type) memory-model seam
#include "vllm/model_executor/model_loader/gguf_keep_quant.h"

namespace vllm {
namespace {

bool DenseDecodeGraphEnabled() {
  const char* value = std::getenv("VLLM_CPP_DENSE_DECODE_GRAPH");
  return value == nullptr || value[0] != '0';
}

bool Gptq4RouteTraceEnabled() {
  const char* value = std::getenv("VLLM_CPP_GPTQ4_TRACE_ROUTE");
  return value != nullptr && value[0] == '1' && value[1] == '\0';
}

void TraceGptq4Route(const ModelForwardInput& input, const char* selected,
                     const char* reason, bool dense_graph, bool uniform_decode,
                     bool gptq_opt_in, bool platform_graph, bool platform_opt_in,
                     int max_graph_batch) {
  std::fprintf(stderr,
               "{\"event\":\"gptq4_route\",\"selected\":\"%s\","
               "\"reason\":\"%s\",\"tokens\":%zu,\"requests\":%d,"
               "\"pure_decode\":%d,\"uniform_query_len\":%lld,"
               "\"dense_graph_enabled\":%d,\"uniform_decode\":%d,"
               "\"gptq_graph_opt_in\":%d,\"platform_graph\":%d,"
               "\"platform_requires_opt_in\":%d,\"max_graph_batch\":%d}\n",
               selected, reason, input.token_ids.size(), input.num_reqs,
               input.pure_decode ? 1 : 0,
               static_cast<long long>(input.uniform_query_len),
               dense_graph ? 1 : 0, uniform_decode ? 1 : 0,
               gptq_opt_in ? 1 : 0, platform_graph ? 1 : 0,
               platform_opt_in ? 1 : 0, max_graph_batch);
}

// The output event owner dies before the input pixel allocation. Its destructor
// drains encode/consumer events, so input staging cannot be recycled early.
class NativeVisionOutput final : public MmEncoderLifetime {
 public:
  NativeVisionOutput(std::shared_ptr<void> pixels,
                    multimodal::Qwen3VLVisionDeviceOutput output)
      : pixels_(std::move(pixels)),output_(std::move(output)) {}
  const vt::Tensor& tensor() const override { return output_.tensor(); }
  void WaitOn(vt::Queue& queue) const override { output_.WaitOn(queue); }
  void RecordUse(vt::Queue& queue) const override { output_.RecordUse(queue); }
 private:
  std::shared_ptr<void> pixels_;
  multimodal::Qwen3VLVisionDeviceOutput output_;
};

class Qwen3_5DenseLoadedModel final : public LoadedModel {
 public:
  Qwen3_5DenseLoadedModel(const ModelRegistration& registration,
                          Qwen3_5DenseWeights weights)
      : LoadedModel(registration),
        owned_weights_(std::move(weights)),
        weights_(&*owned_weights_) {}
  Qwen3_5DenseLoadedModel(const ModelRegistration& registration,
                          const Qwen3_5DenseWeights& weights,
                          BorrowedWeightsTag)
      : LoadedModel(registration), weights_(&weights) {}

  const Qwen3_5DenseWeights& weights() const { return *weights_; }
  void ObserveVisionEncode(const std::string& hash, int64_t rows) {
    const auto count = vision_encode_submissions_.fetch_add(1, std::memory_order_relaxed) + 1;
    if (const char* trace = std::getenv("VT_NATIVE_VISION_TRACE");
        trace && (trace[0] == '1' || trace[0] == '2')) {
      std::fprintf(stderr, "NATIVE_VISION_ENCODE count=%llu hash=%s rows=%lld\n",
                   static_cast<unsigned long long>(count), hash.c_str(),
                   static_cast<long long>(rows));
    }
  }
  std::shared_ptr<const multimodal::Qwen3VLVisionDeviceWeights> VisionWeights(vt::Queue& queue) {
    if (!vision_weights_) {
      vision_weights_=multimodal::PrepareVisionDeviceWeights(
          weights_->visual,weights_->visual_cfg,vt::GetBackend(queue.device),vt::DType::kF16);
      vision_device_=queue.device;
    }
    VT_CHECK(vision_device_==queue.device,"Qwen3.5 vision: resident weights belong to another device");
    return vision_weights_;
  }
  multimodal::Qwen3VLVisionWorkspace& VisionWorkspace(
      const std::array<int64_t,3>& grid,vt::Queue& queue) {
    if (!vision_workspace_ || grid!=vision_grid_ || queue.id!=vision_queue_id_) {
      auto next=multimodal::PrepareVisionWorkspace(
          grid,weights_->visual_cfg,vt::GetBackend(queue.device),queue);
      vision_workspace_=std::move(next); vision_grid_=grid; vision_queue_id_=queue.id;
    }
    return *vision_workspace_;
  }
  // #607 L3: non-empty only when the load deliberately left `model.visual.*`
  // unread because every modality the tower serves was at limit 0.
  std::vector<std::string> skipped_towers() const override {
    if (!weights_->vision_skipped) return {};
    return {std::string(kVisionTowerStageName)};
  }
  bool uses_nvfp4_w4a4() const override {
    return !weights_->layers.empty() &&
           weights_->layers.front().mlp.gate_proj_fp4.IsTrueW4A4();
  }
  std::unique_ptr<Qwen3_5DenseDecodeGraph>& decode_graph() {
    return decode_graph_;
  }

  // SPEC-MTP I5d-pre: retain the loaded `mtp.*` draft weights + build the draft
  // sharing this target's embed_tokens/lm_head. Inert unless FromModelDir
  // attached weights (i.e. unless a SpeculativeConfig is configured).
  bool supports_mtp_draft() const override { return true; }
  // SPEC-DFLASH / SPEC-DSPARK: this forward routes to ForwardDeviceMultiTap.
  bool supports_aux_multi_tap() const override { return true; }
  // #1946: LEND the embedding table to a block drafter that shares it, so the
  // DFlash/DFlash2 draft runs the target's own OwnedTensor and `ResidentWeight`
  // uploads it once instead of once per tensor. Null before a loader has filled
  // the table, so a half-built model lends nothing.
  const OwnedTensor* shared_embed_tokens() const override {
    return weights_->embed_tokens.Empty() ? nullptr : &weights_->embed_tokens;
  }
  void AttachMtpDraftWeights(Qwen3_5MTPWeights weights) override {
    mtp_draft_weights_ = std::move(weights);
  }
  std::unique_ptr<Qwen3_5MTPModel> BuildMtpDraft(
      const HfConfig& config) const override {
    if (!mtp_draft_weights_.has_value()) return nullptr;
    return std::make_unique<Qwen3_5MTPModel>(*mtp_draft_weights_, *weights_,
                                             config);
  }

 private:
  std::optional<Qwen3_5DenseWeights> owned_weights_;
  const Qwen3_5DenseWeights* weights_ = nullptr;
  std::unique_ptr<Qwen3_5DenseDecodeGraph> decode_graph_;
  // Retained draft weights (SPEC-MTP I5d-pre); empty on the production default.
  std::optional<Qwen3_5MTPWeights> mtp_draft_weights_;
  std::shared_ptr<multimodal::Qwen3VLVisionDeviceWeights> vision_weights_;
  vt::Device vision_device_;
  std::array<int64_t,3> vision_grid_{};
  uint64_t vision_queue_id_=0;
  std::shared_ptr<multimodal::Qwen3VLVisionWorkspace> vision_workspace_;
  std::atomic<uint64_t> vision_encode_submissions_{0};
};

void CheckNativeVision(const Qwen3_5DenseWeights& weights,const HfConfig& config) {
  VT_CHECK(weights.exl3_checkpoint && weights.precision.activation==vt::DType::kF16 &&
               IsExl3Checkpoint(config),"Qwen3.5 vision: native EXL3 FP16 model required");
  VT_CHECK(weights.has_visual && !weights.vision_skipped,
           "Qwen3.5 vision: this load carries no vision tower");
  VT_CHECK(weights.visual_cfg.out_hidden_size==config.hidden_size &&
               weights.visual_cfg.deepstack_visual_indexes.empty(),
           "Qwen3.5 vision: tower width must match text hidden size, without DeepStack");
}

int64_t NativeImageRows(const multimodal::ImageKwargs& image,
                        const multimodal::Qwen3VLVisionConfig& cfg) {
  const auto& grid=image.image_grid_thw;
  VT_CHECK(grid[0]==1 && grid[1]>0 && grid[2]>0 && grid[1]<=32768 && grid[2]<=32768 &&
               grid[1]*grid[2]<=16384 && cfg.spatial_merge_size>0 &&
               cfg.spatial_merge_size<=128 && grid[1]%cfg.spatial_merge_size==0 &&
               grid[2]%cfg.spatial_merge_size==0,
           "Qwen3.5 vision: invalid or oversized image grid");
  return grid[1]*grid[2]/cfg.merge_unit();
}

MmEncoderOutput EncodeMmQwen3_5Dense(LoadedModel& model,const HfConfig& config,
                                    vt::Queue& queue,const multimodal::MultiModalFeatureSpec& item) {
  auto& qwen=ModelAs<Qwen3_5DenseLoadedModel>(model,"Qwen3_5ForConditionalGeneration");
  const auto& weights=qwen.weights(); CheckNativeVision(weights,config);
  VT_CHECK(queue.device.type==vt::DeviceType::kXPU,"Qwen3.5 vision: native XPU encoder required");
  VT_CHECK(item.modality=="image" && item.data,"Qwen3.5 vision: processed image input required");
  const auto& image=*item.data; const auto& cfg=weights.visual_cfg;
  const int64_t rows=NativeImageRows(image,cfg),length=image.image_grid_thw[1]*image.image_grid_thw[2];
  VT_CHECK(item.length==rows,"Qwen3.5 vision: encoder rows must match the placeholder length");
  VT_CHECK(cfg.patch_size>0 && cfg.patch_size<=64 && cfg.temporal_patch_size>0 &&
               cfg.temporal_patch_size<=8 && cfg.in_channels>0 && cfg.in_channels<=4,
           "Qwen3.5 vision: invalid patch geometry");
  const int64_t patch_dim=cfg.in_channels*cfg.temporal_patch_size*cfg.patch_size*cfg.patch_size;
  VT_CHECK(image.pixel_dtype==multimodal::ImagePixelDType::kF16 && image.num_patches==length &&
               image.patch_feature_dim==patch_dim &&
               static_cast<int64_t>(image.pixel_values_f16.size())==length*patch_dim,
           "Qwen3.5 vision: direct FP16 processed patches required");
  auto& workspace=qwen.VisionWorkspace(image.image_grid_thw,queue);
  auto prepared=qwen.VisionWeights(queue);
  auto& backend=vt::GetBackend(queue.device);
  dense_attn::Dev d{backend,queue,vt::DType::kF16};
  dense_attn::DBuf pixels(d,vt::DType::kF16,{length,patch_dim},image.pixel_values_f16.data());
  const auto pixel_view=pixels.t(); auto pixel_owner=pixels.ReleaseShared();
  auto output=[&] {
    try {
      return multimodal::Qwen3VLVisionForwardDevice(pixel_view,prepared,workspace,backend,queue);
    } catch (...) {
      // A failing submission may have read input staging. Drain only on this
      // exceptional path before its owner can return the block to the pool.
      backend.Synchronize(queue); throw;
    }
  }();
  auto owner=std::make_shared<NativeVisionOutput>(std::move(pixel_owner),std::move(output));
  VT_CHECK(owner->tensor().shape[0]==rows && owner->tensor().shape[1]==config.hidden_size,
           "Qwen3.5 vision: native encoder produced incompatible rows");
  qwen.ObserveVisionEncode(item.mm_hash, rows);
  MmEncoderOutput result;
  result.storage=owner; result.embeds=owner->tensor(); result.lifetime=std::move(owner);
  return result;
}

MmForwardBuffers EmbedMmQwen3_5Dense(LoadedModel& model,const HfConfig& config,
                                    vt::Queue& queue,const MmEmbedInputs& inputs) {
  auto& qwen=ModelAs<Qwen3_5DenseLoadedModel>(model,"Qwen3_5ForConditionalGeneration");
  CheckNativeVision(qwen.weights(),config);
  VT_CHECK(inputs.mm_embeds,"Qwen3.5 vision embed: source channel required");
  if (!inputs.mm_embeds->empty()) {
    VT_CHECK(inputs.mm_lifetimes && inputs.mm_lifetimes->size()==inputs.mm_embeds->size(),
             "Qwen3.5 vision embed: live source owners required");
    for (const auto& owner : *inputs.mm_lifetimes)
      VT_CHECK(owner,"Qwen3.5 vision embed: live source owners required");
  }
  auto result=Qwen3_5DenseEmbedMultimodal(qwen.weights(),config,queue,inputs);
  if (const char* trace=std::getenv("VT_NATIVE_VISION_TRACE"); trace && trace[0]=='2') {
    // Diagnostic host metadata already carried by the runner. No image,
    // embedding or position tensor is downloaded for this observation.
    auto slices=nlohmann::json::array();
    for (size_t i=0;i<inputs.mm_embeds->size();++i) {
      const auto& parent=(*inputs.mm_lifetimes)[i]->tensor();
      const auto& slice=(*inputs.mm_embeds)[i];
      const auto bytes_per_row=static_cast<uintptr_t>(parent.shape[1])*vt::SizeOf(parent.dtype);
      const auto first=(reinterpret_cast<uintptr_t>(slice.data)-
                        reinterpret_cast<uintptr_t>(parent.data))/bytes_per_row;
      slices.push_back({first,slice.shape[0]});
    }
    const auto metadata=nlohmann::json{
        {"tokens",inputs.token_ids->size()}, {"image_mask",*inputs.is_mm_embed},
        {"mrope",*inputs.mrope_positions}, {"source_slices",std::move(slices)}}.dump();
    std::fprintf(stderr,"%s %s\n",
                 inputs.draft_prefill ? "NATIVE_VISION_DRAFT_EMBED" : "NATIVE_VISION_EMBED",
                 metadata.c_str());
  }
  return result;
}

MropePromptPositions MropeQwen3_5Dense(LoadedModel& model,const HfConfig& config,
    const std::vector<int32_t>& tokens,const std::vector<multimodal::MultiModalFeatureSpec>& features) {
  auto& qwen=ModelAs<Qwen3_5DenseLoadedModel>(model,"Qwen3_5ForConditionalGeneration");
  CheckNativeVision(qwen.weights(),config);
  VT_CHECK(!tokens.empty(),"Qwen3.5 vision positions: nonempty prompt required");
  std::vector<multimodal::MmImageSpan> images;
  for (const auto& item : features) {
    VT_CHECK(item.modality=="image" && item.data,"Qwen3.5 vision positions: processed image required");
    const auto rows=NativeImageRows(*item.data,qwen.weights().visual_cfg);
    VT_CHECK(item.offset>=0 && item.length==rows &&
                 static_cast<size_t>(item.offset)<=tokens.size() &&
                 static_cast<size_t>(rows)<=tokens.size()-item.offset,
             "Qwen3.5 vision positions: placeholder span must match its image grid");
    images.push_back({item.offset,item.data->image_grid_thw});
  }
  MropePromptPositions result;
  result.positions=multimodal::Qwen3VLGetRopeIndex(
      tokens,images,qwen.weights().visual_cfg.spatial_merge_size,&result.delta);
  return result;
}

std::unique_ptr<LoadedModel> LoadQwen3_5DenseModel(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind == ModelSource::Kind::kGguf) {
    // Dense-arch (`qwen35`, e.g. Qwen3.5-2B) GGUF: same GDN / full-attention
    // block loaders as the MoE path with a dense SwiGLU MLP per layer.
    if (source.gguf == nullptr) {
      throw std::runtime_error("GGUF model source is empty");
    }
    // ENG-GGUF-RESIDENCY-RESOLVED-DEVICE: the residency policy is built from
    // the device the ENGINE resolved for this load, never from
    // `platforms::CurrentPlatform()`. The two disagree on `--device cpu` on a
    // CUDA-capable process, and this hook is where the disagreement reached the
    // loader.
    const GgufLoadPolicy gguf_policy =
        GgufLoadPolicy::FromEnv(source.device, detail::ActDType(source.device));
    VT_CHECK(!gguf_policy.weight_value_dtype || config.torch_dtype.empty() ||
                 config.torch_dtype == "bfloat16" || config.torch_dtype == "bf16",
             "Qwen3.5 retained F16 GGUF: this forward resolves BF16 model values; "
             "an unsupported model dtype cannot silently select BF16 semantics");
    return std::make_unique<Qwen3_5DenseLoadedModel>(
        registration,
        LoadQwen3_5DenseFromGguf(*source.gguf, config, &gguf_policy));
  }
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture Qwen3_5ForConditionalGeneration does not support "
        "this weight source");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      registration,
      LoadQwen3_5Dense(*source.safetensors, config, source.load_queue,
                       source.multimodal));
}

void PrepareQwen3_5Dense(LoadedModel& model, const HfConfig& config,
                         vt::Queue& queue) {
  auto& qwen = ModelAs<Qwen3_5DenseLoadedModel>(
      model, "Qwen3_5ForConditionalGeneration");
  if (qwen.weights().gptq4_checkpoint) {
    VT_CHECK(queue.device.type == vt::DeviceType::kXPU,
             "gptq4: packed resident requires XPU");
    for (const Qwen3_5DenseLayerWeights& layer : qwen.weights().layers)
      layer.gptq4.PrepareResident(queue);
    std::fprintf(stderr,
                 "[gptq4] target text path: XPU oneDNN W4A16, FP16 activations "
                 "and BA, FP32 GDN recurrence, eager forward; %zu layers, "
                 "%zu packed resident bytes\n",
                 qwen.weights().layers.size(),
                 qwen.weights().Gptq4ResidentBytes());
    return;
  }
  // MODEL-FP8-BLOCK-LINEAR (#1189 M4). FIRST, before any resident is built: the
  // dense forward READS `Fp8BlockWeight`s now, so what is refused here is a
  // device with no block-scaled GEMM rather than the weight itself. The
  // question is asked at `ModelRegistry::Prepare` — called by every runner
  // before the first forward and before graph capture — so a CUDA user is told
  // here rather than inside the first GEMM or, worse, inside a capture. Inert
  // on CPU and on every other checkpoint. M5 (`489a9a4c0`) narrowed rather than
  // removed it: the CUTLASS kernel covers `VT_CUTLASS_FP8_ARCHS` (12.0a, 12.1a)
  // only, and a CUDA arch outside that cell is still refused here by name.
  RefuseUnrunnableQwen3_5DenseFp8Block(
      ModelAs<Qwen3_5DenseLoadedModel>(model,
                                        "Qwen3_5ForConditionalGeneration")
          .weights(),
      queue.device.type);
  // PERF-27B-LMHEAD-FP4 (issue #213): build the packed lm_head's resident HERE —
  // on CUDA before the runner captures a decode graph, elsewhere before the first
  // forward pays the dequant. Inert on every BF16/FP8/GGUF/tied checkpoint.
  Qwen3_5DenseModel::PrepareLmHeadResident(qwen.weights(), queue);
  // PERF-27B-GDN-FP8-QKVZ: build the merged FP8 GDN [qkv;z] operand here, at
  // model prepare — before the first forward, so it can never allocate or copy
  // inside a CUDA-graph capture. No-op on CPU, on a non-FP8 owner, and when the
  // merge is rolled back (VT_GDN_MERGED_QKVZ_FP8=0).
  Qwen3_5DenseModel::PrepareGdnFp8Resident(qwen.weights(), config, queue);
}

ForwardLogits ForwardQwen3_5Dense(LoadedModel& model,
                                  const ModelForwardInput& input) {
  auto& qwen = ModelAs<Qwen3_5DenseLoadedModel>(model, "Qwen3_5ForConditionalGeneration");
  const Qwen3_5DenseWeights& weights = qwen.weights();

  static const bool spec_graph = [] {
    const char* v = std::getenv("VT_SPEC_DECODE_GRAPH");
    return v == nullptr || (v[0] != '\0' && v[0] != '0');
  }();

  if (input.mm.has_value()) {
    VT_CHECK(model.registration().architecture == "Qwen3_5ForConditionalGeneration",
             "Qwen3.5 dense MM: conditional-generation registration required");
    if (const char* trace=std::getenv("VT_NATIVE_VISION_TRACE"); trace && trace[0]=='2') {
      const auto record=nlohmann::json{{"requests",input.num_reqs},
          {"tokens",input.token_ids.size()},
          {"prefill_tokens",input.gdn_meta.num_prefill_tokens},
          {"spec_requests",input.gdn_meta.num_spec_decodes}}.dump();
      std::fprintf(stderr,"NATIVE_VISION_FORWARD %s\n",record.c_str());
    }
    // Encoding/prefill stay eager. Supported uniform decode shapes stage MM
    // embeddings/axes into their own graph inputs and preserve physical KV slots.
    const auto& platform = platforms::GetPlatform(input.queue.device.type);
    const bool pure_mm_decode = input.pure_decode &&
        input.attn_meta.num_actual_tokens == input.num_reqs && input.attn_meta.max_query_len == 1;
    const bool mm_verify = spec_graph && input.uniform_query_len > 1 &&
        input.gdn_meta.num_spec_decodes > 0 &&
        (input.logits_indices.empty() ||
         static_cast<int64_t>(input.logits_indices.size()) == input.attn_meta.num_actual_tokens);
    const bool graph_eligible = input.gdn_meta.num_prefill_tokens == 0 &&
        (pure_mm_decode || mm_verify) && DenseDecodeGraphEnabled() &&
        platform.support_static_graph_mode() &&
        !platform.static_graph_requires_opt_in(input.config.architectures) &&
        input.num_reqs <= platform.max_static_graph_batch_size();
    if (graph_eligible) {
      if (!qwen.decode_graph())
        qwen.decode_graph() = std::make_unique<Qwen3_5DenseDecodeGraph>(
            weights, input.config, input.queue, input.gdn_state_slots);
      auto result = qwen.decode_graph()->Step(
          input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
          input.attn_kv, input.gdn_state, input.aux_tap, input.hidden_tap, &input);
      if (const char* trace=std::getenv("VT_NATIVE_VISION_TRACE"); trace && trace[0]=='2') {
        const auto record=nlohmann::json{{"tokens",input.token_ids.size()},
            {"requests",input.num_reqs},
            {"captured",qwen.decode_graph()->captured()},
            {"replay_count",qwen.decode_graph()->replay_count()}}.dump();
        std::fprintf(stderr,"NATIVE_VISION_GRAPH %s\n",record.c_str());
      }
      return result;
    }
    return Qwen3_5DenseForwardEmbeddings(input, weights);
  }

  // ENG-ASYNC-SCHED W4: publish the async runner's device-resident input ids for
  // the duration of THIS forward, so the embed at the top of every route below
  // (eager, gathered, tap, multi-tap, decode-graph replay) reads them instead of
  // uploading the host vector, which is stale for decode rows on that path.
  // Null on every other path, and RAII-scoped so it cannot outlive the call.
  const detail::DeviceTokenIdsScope device_ids_scope(
      input.device_token_ids, static_cast<int64_t>(input.token_ids.size()));

  // SPEC-DFLASH D1 (DF-AUX-TAPS): non-null routes to ForwardDeviceMultiTap
  // (byte-identical logits + the [T,H×taps] aux capture); null is byte-identical to
  // the path below. Mutually exclusive with hidden_tap.

  // vLLM's CUDA-graph selection is independent of weight quantization. The
  // driver below already captures the shared dense forward, so restricting it
  // to the 27B FP4 checkpoint left ordinary BF16 Qwen3.5 decode eager.
  // #2812/#1625 gate: the ARCH-SCOPED overload, so the evidence families
  // (Qwen3.5-GDN included) capture ambient while every other family keeps
  // the explicit opt-in — the same shape as the qwen3 driver's gate.
  // GPTQ oneDNN command-graph support is qualified separately from the eager
  // path. Keep eager as the default until the full target/state gates pass.
  const char* gptq_graph = std::getenv("VT_GPTQ4_GRAPH");
  const bool gptq_opt_in = gptq_graph != nullptr &&
                           gptq_graph[0] == '1' && gptq_graph[1] == '\0';
  const auto& platform = platforms::GetPlatform(input.queue.device.type);
  const bool graph_cuda =
      (!weights.gptq4_checkpoint || gptq_opt_in) &&
      platform.support_static_graph_mode() &&
      !platform.static_graph_requires_opt_in(input.config.architectures);
  const int kMaxDecodeGraphBatch = platform.max_static_graph_batch_size();

  // SPEC-DSPARK W8 (#442): mirror vLLM's UNIFORM-decode predicate instead of
  // "query_len == 1". Upstream's captured decode length is
  // `1 + num_speculative_tokens` (cudagraph_dispatcher.py:37), so its T=1+k
  // speculative VERIFY is graph-captured; ours fell to the eager path EVERY step,
  // which the paired 35B measurement charged at ~4.8 ms/step (0.870x where
  // acceptance is high). With num_speculative_tokens == 0 this is exactly
  // `pure_decode`, so the non-spec path is unchanged.
  // DEFAULT ON. The staging defect that forced this off is fixed: the captured
  // graph reads gdn_spec_* / num_accepted, which StageSpecStepInputs now refills
  // per step, and the per-request arrays are sized by the REQUEST count rather
  // than the token count. Gated green with capture ON and OFF across the MTP,
  // DFlash, 35B and concurrent e2e suites, and measured +8.5% / +4.7% on the 35B
  // cells (0.870x -> 0.986x of the pinned graphed oracle on the high-acceptance
  // one). VT_SPEC_DECODE_GRAPH=0 restores the eager verify for an A/B.
  //
  // ENG-CUDAGRAPH-BREAK W6 (#1374): the predicate itself is the RUNNER's now.
  // This file used to re-derive uniformity, the GDN-prefill conjunct and the
  // configured-width comparison for itself, and `qwen3_5_dense.cpp` carried a
  // byte-identical copy of the same twenty lines -- one predicate written twice,
  // which is the shape this row exists to remove. `input.uniform_query_len` is
  // the answer, computed once in `GPUModelRunner::execute_model` through
  // `v1::ActualUniformDecodeQueryLen`.
  //
  // WHAT WIDENED, AND IT IS [#1020]. The old test demanded the batch's uniform
  // length equal `1 + num_speculative_tokens` EXACTLY, the width configured for
  // the engine's lifetime. The scheduler clamps drafts to the step's token
  // budget, so a step every request entered with the same SHORTER draft prefix
  // is uniform at a shorter length, is exactly the shape a graph can serve, and
  // got none. `uniform_query_len > 1` admits it at its actual depth. The driver
  // keys its slot ring on `(S, q, spec)` so the wider predicate cannot reach a
  // graph captured for a different shape.
  //
  // `> 1` rather than `>= 1` because the pure-decode arm is `input.pure_decode`
  // on the left: at q == 1 the two are the same population and `pure_decode` is
  // the one every other driver reads. A value above 1 also PROVES speculation is
  // configured, because the runner bounds the length by `1 + num_spec()`.
  const bool uniform_decode =
      input.pure_decode ||
      (spec_graph && input.uniform_query_len > 1 &&
       // The captured region emits EVERY row, so only a no-op gather may take it.
       // A spec verify needs all 1+k rows anyway, so this holds there.
       (input.logits_indices.empty() ||
        static_cast<int64_t>(input.logits_indices.size()) ==
            input.attn_meta.num_actual_tokens));
  if (std::getenv("VT_TT_ALLOC_TRACE") != nullptr)
    std::fprintf(stderr,
                 "[TT-FWD] route T=%zu num_reqs=%d pure_decode=%d "
                 "uniform_qlen=%lld gather=%d n_idx=%zu -> %s\n",
                 input.token_ids.size(), input.attn_meta.num_reqs,
                 input.pure_decode ? 1 : 0,
                 static_cast<long long>(input.uniform_query_len),
                 input.gather_logits ? 1 : 0, input.logits_indices.size(),
                 uniform_decode ? "graph" : "eager");
  const bool dense_graph = DenseDecodeGraphEnabled();
  const bool use_graph = dense_graph && uniform_decode && graph_cuda &&
                         input.num_reqs <= kMaxDecodeGraphBatch;
  if (weights.gptq4_checkpoint && Gptq4RouteTraceEnabled()) {
    const bool platform_graph = platform.support_static_graph_mode();
    const bool platform_opt_in =
        platform.static_graph_requires_opt_in(input.config.architectures);
    const char* reason = use_graph ? "eligible"
        : !dense_graph ? "dense_graph_disabled"
        : !uniform_decode ? "not_uniform_decode"
        : !gptq_opt_in ? "gptq_graph_opt_in_missing"
        : !platform_graph ? "platform_graph_disabled"
        : platform_opt_in ? "platform_requires_opt_in"
        : "batch_exceeds_graph_limit";
    TraceGptq4Route(input, use_graph ? "graph" : "eager", reason,
                    dense_graph, uniform_decode, gptq_opt_in, platform_graph,
                    platform_opt_in, kMaxDecodeGraphBatch);
  }
  if (use_graph) {
    if (!qwen.decode_graph()) {
      qwen.decode_graph() = std::make_unique<Qwen3_5DenseDecodeGraph>(
          weights, input.config, input.queue, input.gdn_state_slots);
    }
    return qwen.decode_graph()->Step(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, input.aux_tap, input.hidden_tap);
  }

  // Prefill, mixed batches and unsupported graph shapes retain the eager
  // owning MTP tap. Supported decode shapes publish leased graph outputs above.
  if (input.hidden_tap != nullptr) {
    if (weights.gptq4_checkpoint && Gptq4RouteTraceEnabled())
      std::fprintf(stderr,
                   "{\"event\":\"gptq4_route\",\"selected\":\"eager\","
                   "\"reason\":\"hidden_tap\",\"tokens\":%zu,"
                   "\"requests\":%d}\n",
                   input.token_ids.size(), input.num_reqs);
    return Qwen3_5DenseModel::ForwardDeviceTap(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.hidden_tap, input.logits_indices);
  }

  // SPEC-DSPARK W8 (#442): the aux multi-tap forward is the DFlash/DSpark
  // VERIFY path, and it used to return BEFORE the decode-graph gate, which is
  // why the T=1+k verify never captured. The graph branch above now serves it
  // (writing the taps into the slot's persistent buffer); this remains the
  // eager fallback for every batch the graph declines.
  if (input.aux_tap != nullptr) {
    return Qwen3_5DenseModel::ForwardDeviceMultiTap(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.aux_tap, input.logits_indices);
  }
  if (input.gather_logits) {
    return Qwen3_5DenseModel::ForwardDevice(
        input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
        input.attn_kv, input.gdn_state, weights, input.config, input.queue,
        input.logits_indices);
  }
  return HostLogits(
      Qwen3_5DenseModel::Forward(
          input.token_ids, input.positions, input.attn_meta, input.gdn_meta,
          input.attn_kv, input.gdn_state, weights, input.config, input.queue,
          input.logits_indices),
      input.config.vocab_size);
}

const ModelFactory kQwen3_5DenseFactory{
    .parse_config = &ParseQwen3_5Config,
    .load_weights = &LoadQwen3_5DenseModel,
    .prepare = &PrepareQwen3_5Dense,
    .forward = &ForwardQwen3_5Dense,
    .make_kv_cache = &MakeQwen3_5KVCache,
    .is_dense_model = true,
    .consumes_device_token_ids = true,
};

const ModelFactory kQwen3_5DenseVisionFactory=[] {
  auto factory=kQwen3_5DenseFactory;
  factory.encode_mm=&EncodeMmQwen3_5Dense;
  factory.embed_mm=&EmbedMmQwen3_5Dense;
  factory.mrope_prompt_positions=&MropeQwen3_5Dense;
  factory.embed_mm_consumes_device_token_ids=true;
  return factory;
}();

}  // namespace

std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights) {
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      RegistrationFor("Qwen3_5ForConditionalGeneration"), std::move(weights));
}

std::unique_ptr<LoadedModel> MakeQwen3_5DenseLoadedModel(
    Qwen3_5DenseWeights weights, const HfConfig& config) {
  const ModelRegistration* registration =
      &RegistrationFor("Qwen3_5ForConditionalGeneration");
  try {
    const ModelRegistration& resolved = ModelRegistry::Resolve(config);
    if (resolved.factory != nullptr &&
        resolved.factory->load_weights == kQwen3_5DenseFactory.load_weights) {
      registration = &resolved;
    }
  } catch (const std::exception&) {
    // An unregistered architecture keeps the default, as before.
  }
  return std::make_unique<Qwen3_5DenseLoadedModel>(*registration,
                                                   std::move(weights));
}

std::unique_ptr<LoadedModel> BorrowQwen3_5DenseLoadedModel(
    const Qwen3_5DenseWeights& weights) {
  return std::make_unique<Qwen3_5DenseLoadedModel>(
      RegistrationFor("Qwen3_5ForConditionalGeneration"), weights,
      BorrowedWeightsTag{});
}

REGISTER_VLLM_MODEL(qwen3_5_dense, "Qwen3_5ForConditionalGeneration",
                    kQwen3_5DenseVisionFactory, kQwen3_5Info)

// TEXT-ONLY arm of the SAME backbone. Upstream's `Qwen3_5ForCausalLM` IS
// `Qwen3_5ForCausalLMBase` unchanged (`class Qwen3_5ForCausalLM(...): pass`,
// qwen3_5.py:439-440 @ `ad5d29db7`) and is registered against the same `qwen3_5`
// module (registry.py:202 @ `ad5d29db7`, PR #50210), so this is the SAME
// base factory: no forward, KV-cache spec or loader fork. Only the conditional
// registration adds native vision hooks; the text-only arm keeps them null.
//
// AHEAD OF THE PIN, DELIBERATELY. `555967922` (.agents/upstream-sync.md) carries
// only the ForConditionalGeneration entries; the text-only arms landed upstream
// after it. This is a forward port of ONE upstream PR and does not advance the
// pin. See .agents/specs/qwen38-text-only.md §Gates for the owed run gate.
REGISTER_VLLM_MODEL(qwen3_5_dense_text, "Qwen3_5ForCausalLM",
                    kQwen3_5DenseFactory, kQwen3_5TextInfo)

}  // namespace vllm
