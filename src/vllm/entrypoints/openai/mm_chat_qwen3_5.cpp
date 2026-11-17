// Native EXL3 Qwen3.5 uses the shared Qwen image processor, with the executed
// reference's torchvision uint8 resize and direct FP16 patch contract.
#include <filesystem>
#include <algorithm>
#include <cmath>
#include <memory>
#include <stdexcept>

#include "vllm/entrypoints/openai/mm_chat_registry.h"
#include "vllm/model_executor/layers/quantization/exl3_checkpoint.h"
#include "vllm/model_executor/models/qwen3_5_weights.h"
#include "vllm/multimodal/processing/context.h"
#include "vllm/multimodal/qwen3vl_processor.h"

namespace vllm::entrypoints::openai {
namespace {

MultiModalChatSeam MakeQwen3_5ChatSeam(const MultiModalChatContext& ctx) {
  if (!ctx.config || !IsExl3Checkpoint(*ctx.config) || !ctx.device ||
      ctx.device->type != vt::DeviceType::kXPU) {
    throw std::runtime_error("Qwen3.5 native vision chat requires a resolved EXL3 XPU model");
  }
  if (!ctx.tokenizer || !ctx.mm_config || !ctx.prompt_fn || !ctx.codec) {
    throw std::runtime_error("Qwen3.5 native vision chat: incomplete processor context");
  }
  const auto directory = std::filesystem::path(std::u8string(
      reinterpret_cast<const char8_t*>(ctx.model_dir.data()), ctx.model_dir.size()));
  const auto processor_path = directory / "preprocessor_config.json";
  const auto utf8 = processor_path.u8string();
  auto config = multimodal::LoadQwen3VLProcessorConfig(
      std::string(utf8.begin(), utf8.end()), ctx.config_path, ctx.served_model_name);
  const auto tower = Qwen3_5DenseVisionConfig(*ctx.config);
  if (tower.hidden_size != 1152 || tower.num_heads != 16 || tower.head_dim() != 72 ||
      tower.depth <= 0 || tower.depth > 27 || tower.intermediate_size <= 0 ||
      tower.intermediate_size > 16384 || tower.out_hidden_size != ctx.config->hidden_size ||
      tower.out_hidden_size <= 0 || tower.out_hidden_size > 16384 ||
      tower.num_position_embeddings <= 0 || tower.num_position_embeddings > 1048576 ||
      tower.in_channels != 3 || !std::isfinite(tower.norm_eps) || tower.norm_eps <= 0 ||
      !tower.deepstack_visual_indexes.empty() || config.patch_size != tower.patch_size ||
      config.temporal_patch_size != tower.temporal_patch_size ||
      config.merge_size != tower.spatial_merge_size) {
    throw std::runtime_error(
        "Qwen3.5 native vision chat: unsupported tower geometry or mismatched processor");
  }
  // The initial envelope has the pinned reference's 4.2 MP upper bound,
  // additionally limited by the workspace's 16384 patches. Announce the
  // effective budget below; the checkpoint's 16 MP default is not qualified.
  if (config.patch_size <= 0 || config.patch_size > 64 ||
      config.merge_size <= 0 || config.merge_size > 128 ||
      config.temporal_patch_size <= 0 || config.temporal_patch_size > 8 ||
      config.min_pixels <= 0 || config.max_pixels < config.min_pixels) {
    throw std::runtime_error(
        "Qwen3.5 native vision chat: processor budget exceeds the bounded native tower; "
        "configure the reference and native processor with the same supported pixel budget");
  }
  config.max_pixels = std::min({config.max_pixels, multimodal::kNativeQwen3_5MaxImagePixels,
                               multimodal::kNativeQwen3_5MaxImagePatches *
                                   config.patch_size * config.patch_size});
  if (config.min_pixels > config.max_pixels)
    throw std::runtime_error("Qwen3.5 native vision chat: minimum pixel budget exceeds native envelope");
  config.torchvision_bicubic_resize = true;
  config.pixel_dtype = multimodal::ImagePixelDType::kF16;
  config.retain_pixel_values_f32 = false;
  // Both encoder and target-prefix keys consume this identity. Keep legacy
  // Qwen3-VL hashing unchanged, and key native images by their EFFECTIVE
  // processor settings. Checkpoint location/config namespace this engine's
  // resident weights; caches do not survive a model reload or cross engines.
  // This is not a content hash of the weight files or Python hash-byte parity.
  config.model_id = nlohmann::json{
      {"scheme", "qwen3.5-native-f16-torchvision-v1"},
      {"model", config.model_id},
      {"checkpoint_location", ctx.model_dir},
      {"checkpoint_config", ctx.config->raw},
      {"patch", {config.patch_size, config.temporal_patch_size, config.merge_size}},
      {"pixels", {config.min_pixels, config.max_pixels}},
      {"mean", config.channel_mean.value_or(std::array<double,3>{
          config.image_mean, config.image_mean, config.image_mean})},
      {"std", config.channel_std.value_or(std::array<double,3>{
          config.image_std, config.image_std, config.image_std})},
      {"rescale_factor", config.rescale_factor},
      {"markers", {config.image_token_id, config.vision_start_token_id,
                    config.vision_end_token_id}}}.dump();
  auto processor = std::make_shared<multimodal::Qwen3VLImageProcessor>(std::move(config));
  auto info = std::make_shared<multimodal::BaseProcessingInfo>(
      *ctx.mm_config, std::map<std::string, std::optional<int>>{{"image", 2}, {"video", 0}});
  auto body = MakeQwen3VLImageRequestChatFn(
      *processor, *ctx.tokenizer, ctx.prompt_fn, ctx.codec, *info);
  MultiModalChatSeam result;
  result.allowed_limits = info->AllowedMmLimits();
  result.detail = "native EXL3 Qwen3.5 FP16 image processor (two-image envelope, max_pixels=" +
                  std::to_string(processor->config().max_pixels) + ")";
  result.request_chat_fn = [processor, info, body = std::move(body)](
      const std::vector<ChatMessage>& messages,
      const std::vector<ChatCompletionToolsParam>& tools,
      const nlohmann::ordered_json& kwargs) {
    return body(messages, tools, kwargs);
  };
  return result;
}

}  // namespace
REGISTER_VLLM_MM_CHAT(qwen3_5_dense, "Qwen3_5ForConditionalGeneration", &MakeQwen3_5ChatSeam)
}  // namespace vllm::entrypoints::openai
