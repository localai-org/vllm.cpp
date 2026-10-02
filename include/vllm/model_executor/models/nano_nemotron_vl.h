// `NemotronH_Nano_VL_V2` / `NemotronH_Nano_Omni_Reasoning_V3`: the config
// parse and the vision-side weight load. The language tower is
// `NemotronHForCausalLM` under the `language_model.` prefix; the tower and
// projector are radio.h.
//
// Ported from vLLM `e126687a9a`:
//   ParseNanoNemotronVLParams   <- nano_nemotron_vl.py:921-1010 (__init__) and
//                                  :1568-1598 (get_vit_model_from_radio_config),
//                                  configs/radio.py:12-110, radio.py:523-566,
//                                  radio.py:60-106 (ClsToken register count),
//                                  processors/nano_nemotron_vl.py:573-627
//   LoadNanoNemotronVLVisionWeights <- nano_nemotron_vl.py:1499-1566
//                                  (load_weights), radio.py:696-743
//
// Refused BY NAME, each with the piece that is missing:
//   * the static InternVL tiling arm (`min_num_patches` absent from
//     `vision_config.args`, which is `NemotronH_Nano_VL_V2`'s 12B release):
//     `dynamic_preprocess` + thumbnail (processors/nano_nemotron_vl.py:94-143)
//     is not ported;
//   * `ps_version` other than "v2";
//   * a RADIO `model` outside VIT_TIMM_DIM_BY_NAME, qk normalization, a norm
//     type other than layer_norm, an activation other than gelu, and a
//     non-unit `initializer_factor` (the layer scale).
// Audio (`sound_config`) and video are NOT refused here: the checkpoint that
// carries them still serves images. Their tensors are accounted and deferred by
// name at load, and an audio or video request is refused at encode time.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/model_executor/models/radio.h"
#include "vllm/multimodal/nano_nemotron_vl_processor.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

class LoadedModel;

struct NanoNemotronVLParams {
  std::string architecture;
  multimodal::RadioVisionConfig radio;
  multimodal::NanoNemotronVLProjectorConfig projector;
  // max_model_len is left 0 here: it is the ENGINE's value and the serving
  // path fills it (models/nano_nemotron_vl.py:209).
  multimodal::NanoNemotronVLProcessorConfig processor;
  int64_t reduction = 2;  // 1 / downsample_ratio
  // The token strings the processor encodes (processors/...:37-39). The ids
  // are resolved from the tokenizer by string, never from config numbers.
  std::string img_start_token = "<img>";
  std::string img_end_token = "</img>";
  std::string img_context_token = "<image>";
  bool has_sound = false;
  bool has_video_embedder = false;
};

// Parse the top-level config.json of either architecture. Throws, naming the
// missing piece, on anything the image path cannot represent.
NanoNemotronVLParams ParseNanoNemotronVLParams(const HfConfig& config);

// The language tower's config: `llm_config` (the released spelling) or
// `text_config`, as an HfConfig the NemotronH parser consumes. Its
// `quantization_config`, when the top-level config carries one, is folded in:
// a ModelOpt checkpoint quantizes the language tower and names its modules
// under `language_model.`.
HfConfig NanoNemotronVLTextConfig(const HfConfig& config);

struct NanoNemotronVLVisionLoad {
  multimodal::RadioVisionWeights radio;
  multimodal::NanoNemotronVLProjectorWeights projector;
  // Every `vision_model.*` / `mlp1.*` tensor the checkpoint ships, and how
  // many of them were materialized or deferred by name.
  int64_t shipped = 0;
  int64_t materialized = 0;
  std::vector<std::string> deferred;  // names, e.g. the video embedder
};

// Read the tower and the projector out of the checkpoint shards (bf16 only;
// anything else is refused by name). Accounts for EVERY `vision_model.` and
// `mlp1.` tensor: materialized, or deferred by name (the input conditioner,
// which the processor applies from `norm_mean`/`norm_std`; the video
// embedder, whose modality is not ported).
NanoNemotronVLVisionLoad LoadNanoNemotronVLVisionWeights(
    const std::vector<SafetensorsFile>& shards, const NanoNemotronVLParams& params);

// The vision-side accounting of the load that produced `model`: zero shipped
// and zero materialized when the engine loaded text-only (every modality's
// limit is 0, nano_nemotron_vl.py:1500-1504), because no vision or `mlp1`
// tensor was read. The load happens inside the type-erased
// `ModelRegistry::Load` factory, so a gate has no other way to reach it, as
// with `NemotronHLoadReportOf`. Throws if `model` is not this architecture.
const NanoNemotronVLVisionLoad& NanoNemotronVLVisionLoadOf(const LoadedModel& model);

}  // namespace vllm
