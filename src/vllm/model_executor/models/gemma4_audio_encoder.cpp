// Pinned Gemma4AudioModel.forward and gemma4_mm.py::_process_audio_input.
// CPU reference adapters; production device execution remains owed.
#include "vllm/model_executor/models/gemma4_loaded.h"
#include <cmath>
#include <cstring>

namespace vllm {
namespace {
MmEncoderOutput EncodeMmGemma4(LoadedModel& model, const HfConfig& config,
    vt::Queue& queue, const multimodal::MultiModalFeatureSpec& item) {
  if (item.modality == "audio") return EncodeMmGemma4Audio(model, config, queue, item);
  VT_CHECK(queue.device.type == vt::DeviceType::kCPU, "gemma4 vision: CPU reference only");
  VT_CHECK(item.modality == "image" && item.data && !item.audio_data,
           "gemma4 encoder: image or audio required");
  const auto& w = ModelAs<Gemma4LoadedModel>(model, "Gemma4ForConditionalGeneration").weights();
  VT_CHECK(w.vision_config, "gemma4 vision: tower absent");
  const auto& cfg = *w.vision_config;
  const auto& a = *item.data;
  const auto& grid = a.image_grid_thw;
  VT_CHECK(cfg.patch_size == 16 && cfg.pooling_kernel_size == 3 &&
               cfg.text_hidden_size == config.hidden_size, "gemma4 vision: processor config");
  VT_CHECK(a.num_patches > 0 && a.num_patches <= 10080 &&
               a.patch_feature_dim == 768 && a.pixel_values_bf16.empty() &&
               grid[0] == 1 && grid[1] > 0 && grid[2] > 0 &&
               grid[1] <= a.num_patches / grid[2] && grid[1] % 3 == 0 && grid[2] % 3 == 0,
           "gemma4 vision: image metadata");
  const int64_t valid = grid[1] * grid[2], rows = valid / 9;
  VT_CHECK(item.length == rows, "gemma4 vision: placeholder length");
  std::vector<int64_t> positions(static_cast<size_t>(a.num_patches) * 2, -1);
  for (int64_t row = 0; row < valid; ++row) {
    positions[row * 2] = row % grid[2]; positions[row * 2 + 1] = row / grid[2];
  }
  const auto& p = w.vision.embedding_projection;
  VT_CHECK(p.dtype == vt::DType::kBF16 && p.rank == 2 &&
               p.shape[0] == cfg.text_hidden_size && p.shape[1] == cfg.hidden_size &&
               cfg.hidden_size > 0 && cfg.text_hidden_size > 0 &&
               cfg.text_hidden_size <= INT64_MAX / 2 / cfg.hidden_size &&
               p.bytes.size() == static_cast<size_t>(cfg.text_hidden_size * cfg.hidden_size) * 2,
           "gemma4 vision: BF16 connector shape");
  auto vision = w.vision_patch_weights;
  vision.embed_projection.resize(p.bytes.size() / 2);
  std::memcpy(vision.embed_projection.data(), p.bytes.data(), p.bytes.size());
  auto values = std::make_shared<std::vector<float>>(multimodal::Gemma4VisionForward(
      a.pixel_values_f32, positions, a.num_patches, vision, cfg,
      vt::GetBackend(vt::DeviceType::kCPU)));
  VT_CHECK(values->size() == static_cast<size_t>(rows * cfg.text_hidden_size),
           "gemma4 vision: encoder output shape");
  return {values, vt::Tensor::Contiguous(values->data(), vt::DType::kF32,
                                        queue.device, {rows, cfg.text_hidden_size})};
}
}  // namespace
constinit const ModelFactory kGemma4MmHooks{.encode_mm = &EncodeMmGemma4,
    .embed_mm = &EmbedMmGemma4, .mm_inputs_ready = true};
MmForwardBuffers EmbedMmGemma4(LoadedModel& model, const HfConfig& config,
    vt::Queue& queue, const MmEmbedInputs& in) {
  VT_CHECK(queue.device.type == vt::DeviceType::kCPU && !in.host_token_ids_stale,
           "gemma4 merge: CPU current host ids required");
  VT_CHECK(in.token_ids && in.mm_embeds && in.is_mm_embed &&
           (!in.mrope_positions || in.mrope_positions->empty()), "gemma4 merge: inputs");
  const auto& ids=*in.token_ids; const auto& mask=*in.is_mm_embed;
  const int64_t H=config.hidden_size, V=config.vocab_size;
  VT_CHECK(H>0 && V>0 && !ids.empty() && ids.size()==mask.size() &&
           ids.size()<=SIZE_MAX/sizeof(uint16_t)/H &&
           ids.size()<=static_cast<uint64_t>(INT64_MAX/H), "gemma4 merge: shape");
  const auto& w=ModelAs<Gemma4LoadedModel>(model,"Gemma4ForConditionalGeneration").weights();
  const auto& tab=w.embed_tokens;
  VT_CHECK(V<=INT64_MAX/H && static_cast<uint64_t>(V)<=SIZE_MAX/2/H &&
           tab.dtype==vt::DType::kBF16 && tab.rank==2 && tab.shape[0]==V &&
           tab.shape[1]==H && tab.bytes.size()==static_cast<size_t>(V*H)*2,
           "gemma4 merge: token table");
  size_t wanted=0;
  for(size_t t=0;t<ids.size();++t) {
    VT_CHECK(ids[t]>=0 && ids[t]<V, "gemma4 merge: token range");
    VT_CHECK(mask[t]==0 || mask[t]==1, "gemma4 merge: boolean mask");
    wanted+=mask[t];
  }
  size_t rows=0;
  for(const auto& x:*in.mm_embeds) {
    VT_CHECK(x.device.type==vt::DeviceType::kCPU && x.dtype==vt::DType::kF32 &&
             x.rank==2 && x.data && x.shape[0]>0 && x.shape[1]==H &&
             x.stride[0]==H && x.stride[1]==1 &&
             static_cast<uint64_t>(x.shape[0])<=wanted-rows,
             "gemma4 merge: slice shape/dtype/cardinality");
    rows+=static_cast<size_t>(x.shape[0]);
  }
  VT_CHECK(rows==wanted,"gemma4 merge: placeholder cardinality");
  const auto& raw=config.raw.contains("text_config") ? config.raw.at("text_config") : config.raw;
  const int64_t ple_vocab=raw.value("vocab_size_per_layer_input",V);
  VT_CHECK(ple_vocab>0 && ple_vocab<=V,"gemma4 merge: PLE vocabulary");
  auto values=std::make_shared<std::vector<uint16_t>>(ids.size()*H);
  auto ple=std::make_shared<std::vector<int32_t>>(ids.size());
  const float scale=vt::BF16ToF32(vt::F32ToBF16(std::sqrt(static_cast<float>(H))));
  size_t slice=0; int64_t row=0;
  for(size_t t=0;t<ids.size();++t) {
    (*ple)[t]=!mask[t] && ids[t]<ple_vocab ? ids[t] : 0;
    for(int64_t h=0;h<H;++h) {
      uint16_t bits; std::memcpy(&bits,tab.bytes.data()+2*(ids[t]*H+h),2);
      float value=vt::BF16ToF32(bits)*scale;
      if(mask[t]) value=(*in.mm_embeds)[slice].Ptr<float>()[row*H+h];
      VT_CHECK(std::isfinite(value),"gemma4 merge: nonfinite embedding");
      (*values)[t*H+h]=vt::F32ToBF16(value);
    }
    if(mask[t] && ++row==(*in.mm_embeds)[slice].shape[0]) {++slice;row=0;}
  }
  MmForwardBuffers out;
  out.storage={values,ple};
  out.mm.inputs_embeds=vt::Tensor::Contiguous(values->data(),vt::DType::kBF16,
                                            queue.device,{static_cast<int64_t>(ids.size()),H});
  out.mm.ple_token_ids=vt::Tensor::Contiguous(ple->data(),vt::DType::kI32,
                                            queue.device,{static_cast<int64_t>(ids.size())});
  return out;
}
MmEncoderOutput EncodeMmGemma4Audio(LoadedModel& model, const HfConfig& config,
    vt::Queue& queue, const multimodal::MultiModalFeatureSpec& item) {
  VT_CHECK(queue.device.type == vt::DeviceType::kCPU, "gemma4 audio: CPU reference only");
  VT_CHECK(item.modality == "audio" && item.audio_data, "gemma4 encoder: audio only");
  const auto& w = ModelAs<Gemma4LoadedModel>(model, "Gemma4ForConditionalGeneration").weights();
  VT_CHECK(w.audio_config && w.audio_frontend_config, "gemma4 audio: tower absent");
  auto cfg = *w.audio_config;
  const auto& a = *item.audio_data;
  VT_CHECK(a.num_chunks == 1 && a.chunk_num_samples.empty() && a.chunk_num_tokens.empty(),
           "gemma4 audio: one unchunked clip required");
  VT_CHECK(a.n_mels > 0 && a.n_frames > 0 && a.n_frames <= INT64_MAX / a.n_mels &&
           static_cast<uint64_t>(a.n_frames) <= SIZE_MAX / sizeof(float) / a.n_mels &&
           a.input_features.size() == static_cast<size_t>(a.n_frames * a.n_mels),
           "gemma4 audio: feature shape");
  VT_CHECK(cfg.hidden_size % 2 == 0 && cfg.hidden_size % cfg.num_heads == 0 &&
           cfg.hidden_act == "silu" && cfg.context_right == 0 && cfg.context_left > 1,
           "gemma4 audio: unsupported forward config");
  VT_CHECK(a.n_mels <= INT64_MAX-3 && (a.n_mels+3)/4 == cfg.sub_ch0/4,
           "gemma4 audio: mel projection width");
  VT_CHECK(config.hidden_size == cfg.text_hidden_size, "gemma4 audio: text width");
  VT_CHECK(a.feature_mask.empty() || a.feature_mask.size() == static_cast<size_t>(a.n_frames),
           "gemma4 audio: mask length");
  int64_t valid = 0;
  bool padding = false;
  for (int64_t t=0; t<a.n_frames; ++t) {
    const int bit = a.feature_mask.empty() ? 1 : a.feature_mask[t];
    VT_CHECK(bit == 0 || bit == 1, "gemma4 audio: mask must be boolean");
    VT_CHECK(!padding || bit == 0, "gemma4 audio: mask must be valid prefix");
    if (bit) ++valid; else padding = true;
  }
  VT_CHECK(valid > 0, "gemma4 audio: no valid frames");
  const int64_t rows = (valid-1)/4+1;
  VT_CHECK(item.length == rows && (a.num_tokens == 0 || a.num_tokens == rows),
           "gemma4 audio: placeholder length");
  VT_CHECK(rows <= INT64_MAX / cfg.text_hidden_size &&
           static_cast<uint64_t>(rows) <= SIZE_MAX / sizeof(float) / cfg.text_hidden_size,
           "gemma4 audio: output size overflow");
  // Causal attention and masked convolutions: trim the invalid suffix before
  // encoding. This emits exactly mask[::2][::2]'s true rows, never padded rows.
  std::vector<float> features(static_cast<size_t>(valid)*a.n_mels);
  for (int64_t t=0; t<valid; ++t)
    for (int64_t m=0; m<a.n_mels; ++m) {
      const float value=a.input_features[m*a.n_frames+t];
      VT_CHECK(std::isfinite(value), "gemma4 audio: nonfinite features");
      features[t*a.n_mels+m]=value;
    }
  cfg.feature_size=a.n_mels;
  const auto& p=w.audio.embedding_projection;
  VT_CHECK(p.dtype == vt::DType::kBF16 && p.rank == 2 &&
           p.shape[0] == cfg.text_hidden_size && p.shape[1] == cfg.output_proj_dims &&
           p.bytes.size() == static_cast<size_t>(p.shape[0]*p.shape[1])*2,
           "gemma4 audio: BF16 connector shape");
  std::vector<float> connector(p.bytes.size()/2);
  for (size_t i=0; i<connector.size(); ++i) {
    uint16_t bits; std::memcpy(&bits,p.bytes.data()+2*i,2);
    connector[i]=vt::BF16ToF32(bits);
  }
  auto values=std::make_shared<std::vector<float>>(multimodal::Gemma4AudioForward(
      features, valid, {}, w.audio_frontend_weights, cfg, nullptr, &connector));
  VT_CHECK(values->size() == static_cast<size_t>(rows*cfg.text_hidden_size),
           "gemma4 audio: encoder output shape");
  return {values, vt::Tensor::Contiguous(values->data(), vt::DType::kF32,
                                        queue.device, {rows,cfg.text_hidden_size})};
}
}  // namespace vllm
