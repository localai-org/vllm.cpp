// vllm.cpp original container reader. Sliding-window normalization mirrors
// vllm/config/model.py:542-559,654-660,723-726,1232-1234; typed RoPE
// normalization mirrors vllm/transformers_utils/config.py:439-509 and
// model_executor/layers/rotary_embedding/__init__.py:33-112,200-230,243-283,
// 315-335
// @ e24d1b24fe96.
#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <nlohmann/json.hpp>

namespace vllm {

// Typed effective view of the scalar scaled-RoPE portion of HuggingFace's
// rope_parameters (or legacy rope_scaling) dictionary. Defaults are the ones
// consumed by pinned get_rope(). Formula families that follow W5 add their
// fields here without making model code parse JSON again.
struct RopeParameters {
  std::string rope_type = "default";
  double rope_theta = 10000.0;
  std::optional<int64_t> rope_dim = std::nullopt;
  double partial_rotary_factor = 1.0;

  // Common scaling dispatch fields. YaRN and Llama 3 require both; they remain
  // optional so default RoPE has one unambiguous representation.
  std::optional<double> factor = std::nullopt;
  std::optional<int64_t> original_max_position_embeddings = std::nullopt;
  // Llama 3 frequency-band boundaries. Required only for rope_type=llama3.
  std::optional<double> low_freq_factor = std::nullopt;
  std::optional<double> high_freq_factor = std::nullopt;
  // Phi-3 LongRoPE frequency divisors and optional per-cache magnitudes.
  std::vector<double> short_factor;
  std::vector<double> long_factor;
  std::optional<double> short_mscale = std::nullopt;
  std::optional<double> long_mscale = std::nullopt;
  // Dynamic-NTK alternatives. Presence of alpha takes precedence over factor;
  // factor mode optionally names its trained context length.
  std::optional<double> alpha = std::nullopt;
  std::optional<int64_t> max_trained_positions = std::nullopt;
  // Remaining YaRN-only controls.
  double extrapolation_factor = 1.0;
  double attn_factor = 1.0;
  int64_t beta_fast = 32;
  int64_t beta_slow = 1;
  bool apply_yarn_scaling = true;
  bool truncate = true;

  // MRoPE T/H/W frequency sections. Empty means one-dimensional RoPE.
  std::vector<int64_t> mrope_section;
  bool mrope_interleaved = false;

  friend bool operator==(const RopeParameters&, const RopeParameters&) = default;
};

// Typed view over a HuggingFace config.json. Key names and defaults follow
// the pinned upstream config classes (vllm/transformers_utils/configs/
// qwen3_next.py and friends). Missing optional fields default to 0/empty;
// missing required fields (model_type, hidden_size, num_hidden_layers) throw.
//
// Composite multimodal wrapper configs (e.g.
// Qwen3_5MoeForConditionalGeneration) nest the text-model fields under a
// `text_config` sub-dict; LoadHfConfig resolves that nested object as the source
// of the text fields, mirroring upstream PretrainedConfig.get_text_config().
// `model_type` and `architectures` are always read from the top-level wrapper.
// The sampling keys of ONE generation_config.json, as
// ModelConfig.try_get_generation_config (vllm/config/model.py) surfaces them.
//
// Upstream returns `GenerationConfig.to_diff_dict()`, i.e. only the keys that
// differ from a bare `GenerationConfig()`. The pinned floor is
// `transformers >= 5.5.3` (vLLM requirements/common.txt:10), and at 5.x every
// sampling field of a bare GenerationConfig is None, so every key the JSON
// declares survives the diff and reading the file literally is exact. Measured
// against transformers 5.3.0 on a Qwen-shaped file: repetition_penalty 1.05,
// temperature 1.0, top_k 20 and top_p 0.95 all survive. If the pin's floor ever
// moves back to transformers 4.x -- whose defaults were temperature 1.0,
// top_k 50, top_p 1.0 -- this parse becomes wrong and has to grow that table.
struct GenerationConfigSampling {
  std::optional<double> repetition_penalty;
  std::optional<double> temperature;
  std::optional<int> top_k;
  std::optional<double> top_p;
  std::optional<double> min_p;
  // Upstream renames this to max_tokens when it narrows; kept under its HF name
  // here because this struct is the FILE, not the narrowed result.
  std::optional<int> max_new_tokens;
};

struct HfConfig {
  std::string model_type;
  std::vector<std::string> architectures;
  int64_t hidden_size = 0;
  int64_t num_hidden_layers = 0;
  int64_t vocab_size = 0;
  int64_t num_attention_heads = 0;
  int64_t num_key_value_heads = 0;  // absent -> num_attention_heads
  int64_t head_dim = 0;             // absent -> hidden_size / num_attention_heads
  // Model-level local-attention width. Absent/null/0 => full attention, matching
  // ModelConfig's pinned normalization before backend construction.
  std::optional<int64_t> sliding_window = std::nullopt;
  // Hybrid models: "linear_attention"/"full_attention" per layer; empty = all
  // full attention.
  std::vector<std::string> layer_types;
  int64_t intermediate_size = 0;
  // MoE (0 when absent):
  int64_t num_experts = 0;
  int64_t num_experts_per_tok = 0;
  int64_t moe_intermediate_size = 0;
  int64_t shared_expert_intermediate_size = 0;
  // Gated DeltaNet (0 when absent):
  int64_t linear_num_key_heads = 0;
  int64_t linear_num_value_heads = 0;
  int64_t linear_key_head_dim = 0;
  int64_t linear_value_head_dim = 0;
  int64_t linear_conv_kernel_dim = 0;
  // GDN output-gate activation, CANONICALIZED at parse time: only "silu" or
  // "sigmoid" ever reach a consumer. Mirrors
  // vllm/model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py:452-456
  // (`getattr(config, "output_gate_type", "silu")`, "swish" collapsed to
  // "silu", membership in {silu, swish, sigmoid} asserted); LoadHfConfig
  // REFUSES anything else instead of silently defaulting.
  std::string output_gate_type = "silu";
  // Qwen3.5/3.6 checkpoint contract for the recurrent (SSM) cache. Upstream's
  // verify hook copies this into mamba_ssm_cache_dtype when the CLI setting is
  // "auto"; an absent value leaves the recurrent cache at the convolution-cache
  // dtype. Known checkpoint values are float16/bfloat16/float32.
  std::string mamba_ssm_dtype;
  // RoPE:
  double rope_theta = 10000.0;
  int64_t rotary_dim = 0;  // partial_rotary_factor * head_dim, rounded
  RopeParameters rope_parameters;
  // Distinguishes a checkpoint dictionary from effective defaults synthesized
  // from top-level rope_theta/partial_rotary_factor.
  bool has_rope_parameters = false;
  double rms_norm_eps = 0.0;
  int64_t max_position_embeddings = 0;
  // Resolved model declaration from `dtype` or legacy `torch_dtype`, with the
  // effective text config preferred over its wrapper. Empty means undeclared.
  std::string torch_dtype;
  std::string dtype_source;
  nlohmann::json raw;  // full doc for fields we don't type yet
  // eos_token_id from the sibling generation_config.json, sorted and unique.
  // Upstream ModelConfig.try_get_generation_config (config/model.py) loads that
  // file whenever --generation-config is "auto" (the default) or "vllm", and
  // SamplingParams.update_from_generation_config (sampling_params.py:645-655)
  // merges its ids into stop_token_ids. It is a SEPARATE field rather than a
  // rewrite of raw["eos_token_id"] because the two have different roles: the
  // checkpoint's own eos_token_id supplies the PRIMARY eos id, while these are
  // secondary stop ids gated on ignore_eos. Empty when the file is absent,
  // unparseable, or carries no eos_token_id.
  std::vector<int32_t> generation_config_eos_ids;
  // What upstream's try_get_generation_config falls back to when the sibling
  // generation_config.json does NOT exist: GenerationConfig.from_model_config,
  // i.e. the TEXT config's own eos_token_id (transformers_utils/config.py:
  // 1066-1090 @ 5559679229), sorted and unique. Empty whenever the file exists.
  // A nested wrapper (text_config) that names no top-level eos otherwise stops
  // on nothing (MODEL-TEV1 Phase 7).
  std::vector<int32_t> model_config_eos_ids;
  // The `eos_token` of the sibling tokenizer_config.json (a string, or an
  // object's "content"), empty when absent. Upstream's PRIMARY eos is the
  // tokenizer's eos_token_id (renderers/base.py:310-317); InputProcessor uses
  // this only when config.json and the tokenizer.json post_processor name no
  // eos (MODEL-TEV1 Phase 7, .agents/specs/tev1-eos-fallback.md).
  std::string tokenizer_eos_token;
  // The six SAMPLING keys of the same sibling generation_config.json, which
  // upstream reads through the same one file read and then narrows in
  // ModelConfig.get_diff_sampling_param (config/model.py). Every field is unset
  // when the file is absent, unparseable, or does not declare that key, which
  // is what makes "the checkpoint said nothing" distinguishable from "the
  // checkpoint said the neutral value" -- the distinction the whole defaulting
  // rule turns on. See include/vllm/config/generation.h for the narrowing and
  // for the --generation-config selector that can point somewhere else.
  GenerationConfigSampling generation_config_sampling;
};

// Loads and parses `path`. Throws std::runtime_error (message includes the
// path) on missing file, malformed JSON, or missing required fields.
HfConfig LoadHfConfig(const std::string& path);

// Reads the sampling keys of the generation_config.json at `path`. Mirrors
// ModelConfig.try_get_generation_config's failure polarity exactly: a missing
// file, unparseable JSON, a non-object document, a null value and a
// wrong-typed value each leave the corresponding field unset rather than
// throwing, because upstream's loader returns {} and never raises.
//
// Exposed because --generation-config can name a DIRECTORY other than the
// checkpoint's, which is the one case the sibling read on HfConfig cannot
// serve.
GenerationConfigSampling ReadGenerationConfigSamplingFile(const std::string& path);

// The same parse, from a config object already in hand. `source` appears in
// every error message exactly where the path would, so a refusal still names
// where the config came from.
//
// Exists because a config does not always arrive as a sibling `config.json`.
// LTX-2.5's text encoder ships as ONE safetensors file whose HuggingFace config
// rides in the file's own `__metadata__["gemma_config"]`
// (gemma_assets.py:34, :110-114) — and the shipped `vonkaiser` NVFP4 build
// carries no `__metadata__` at all, so its caller has to source the config out
// of band and hand it over as an object. Writing it to a temporary file just to
// read it back would make the temp directory part of a model path.
//
// Reads no sibling `generation_config.json`: without a path there is no sibling
// to read, so `generation_config_eos_ids` stays empty rather than picking up
// whatever happens to sit in the working directory.
HfConfig ParseHfConfig(const nlohmann::json& doc, const std::string& source);

// Cheap, non-throwing peek at config.json's `architectures` array — empty on
// any parse/read problem. Exists for TASK dispatch BEFORE the full text-model
// HfConfig parse: a SupportsTranscription-only checkpoint (Parakeet) nests its
// fields under `encoder_config`, so LoadHfConfig's required-field errors would
// fire before the registry's refuse-by-task message could (ARCH-ONE-SURFACE
// ROW 1). Callers that need the parsed config still go through LoadHfConfig;
// this never replaces it.
std::vector<std::string> PeekHfArchitectures(const std::string& path);

}  // namespace vllm
