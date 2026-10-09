// kev System 1 decision model registry (MODEL-KEV).
//
// Self-registers "KevModel" via REGISTER_VLLM_MODEL. kev is a pooling model:
// it forwards the Qwen3.5 dense backbone to post-final-norm hidden states
// (ForwardDenseHidden) and applies a PointerHead readout at inference time
// (Phase 5c). The LoRA adapter is merged into the base weights at CONVERT
// time by scripts/convert-kev.py, so this loader sees a single set of
// safetensors shards: model.safetensors (merged Qwen3.5 BF16) + head.safetensors
// (PointerHead F32). No runtime LoRA merge, no two-directory loading.
//
// Ported from jaredpalmer/kev kev/model.py @ 19dcae9b6e3e1a48200c5825aad9fc200d31e20a.
#include "vllm/model_executor/models/model_registry.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "vllm/model_executor/model_loader/safetensors_reader.h"  // SafetensorsFile, StTensor
#include "vllm/model_executor/models/kev.h"
#include "vllm/model_executor/models/kev_inference.h"
#include "vllm/model_executor/models/qwen3_5.h"          // ForwardLogits
#include "vllm/model_executor/models/qwen3_5_common.h"   // ParseQwen3_5Config, MakeQwen3_5KVCache, HostLogits
#include "vllm/model_executor/models/qwen3_5_dense.h"     // LoadQwen3_5Dense, ForwardDenseHidden
#include "vllm/tokenizer/tokenizer.h"

namespace vllm {
namespace {

// kev is a POOLING model (System 1 decision): no text-generation path.
// is_hybrid mirrors the Qwen3.5 backbone (GDN + full-attention layers).
inline constexpr ModelInfo kKevInfo{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = true,
    .has_inner_state = false,
    .supports_multimodal = false,
    .supports_transcription = false,
    .supports_transcription_only = false,
    .score_type = "bi-encoder",
};

// Owned model: merged Qwen3.5 dense weights + PointerHead weights + head params.
// The PointerHead readout (KevReadout) runs in the inference pipeline (Phase 5c),
// not in this forward — the factory forward returns hidden states for the
// pooling runner, exactly as the Llama embedding path does.
class KevLoadedModel final : public LoadedModel {
 public:
  KevLoadedModel(const ModelRegistration& registration,
                 Qwen3_5DenseWeights weights,
                 kev::HeadWeights head_weights,
                 kev::HeadParams head_params,
                 float temperature,
                 HfConfig config)
      : LoadedModel(registration),
        weights_(std::move(weights)),
        head_weights_(std::move(head_weights)),
        head_params_(head_params),
        temperature_(temperature),
        config_(std::move(config)) {}

  const Qwen3_5DenseWeights& weights() const { return weights_; }
  const kev::HeadWeights& head_weights() const { return head_weights_; }
  const kev::HeadParams& head_params() const { return head_params_; }
  float temperature() const { return temperature_; }
  const HfConfig& config() const { return config_; }

  // Stored by PrepareKev (the ModelFactory prepare callback). KevInference
  // retrieves it from here, avoiding any shared-header accessor.
  void set_queue(vt::Queue q) { queue_ = q; }
  vt::Queue queue() const { return queue_; }

 private:
  Qwen3_5DenseWeights weights_;
  kev::HeadWeights head_weights_;
  kev::HeadParams head_params_;
  float temperature_;
  HfConfig config_;
  vt::Queue queue_;
};

// Load PointerHead weights (q.weight, q.bias, k.weight, k.bias) from
// head.safetensors, which co-exists with model.safetensors in the shard list.
// All four tensors are F32, row-major (PyTorch nn.Linear convention).
kev::HeadWeights LoadHeadWeights(const std::vector<SafetensorsFile>& shards) {
  kev::HeadWeights hw;
  for (const SafetensorsFile& shard : shards) {
    for (const std::string& name : shard.Names()) {
      // Only load the four PointerHead tensors; skip base model tensors
      // (model.safetensors carries BF16 Qwen3.5 weights).
      if (name != "q.weight" && name != "q.bias" &&
          name != "k.weight" && name != "k.bias") {
        continue;
      }
      const StTensor& t = shard.Get(name);
      if (t.dtype != "F32") {
        throw std::runtime_error(
            "kev: head tensor " + name + " must be F32, got " + t.dtype);
      }
      size_t count = t.nbytes / sizeof(float);
      std::vector<float> dst(count);
      std::memcpy(dst.data(), t.data, t.nbytes);
      if (name == "q.weight") {
        hw.q_weight = std::move(dst);
      } else if (name == "q.bias") {
        hw.q_bias = std::move(dst);
      } else if (name == "k.weight") {
        hw.k_weight = std::move(dst);
      } else if (name == "k.bias") {
        hw.k_bias = std::move(dst);
      }
    }
  }
  if (hw.q_weight.empty() || hw.q_bias.empty() ||
      hw.k_weight.empty() || hw.k_bias.empty()) {
    throw std::runtime_error(
        "kev: head.safetensors not found or incomplete "
        "(q.weight/q.bias/k.weight/k.bias required). "
        "Run scripts/convert-kev.py first.");
  }
  return hw;
}

std::unique_ptr<LoadedModel> LoadKev(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture KevModel does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }

  // Load merged Qwen3.5 dense weights. The LoRA adapter was merged at
  // convert time (convert-kev.py), so the safetensors shards carry the
  // final merged weights with the original Qwen3.5 tensor-name layout.
  Qwen3_5DenseWeights weights =
      LoadQwen3_5Dense(*source.safetensors, config, source.load_queue,
                       source.multimodal);

  // Load PointerHead weights from head.safetensors in the same shard list.
  kev::HeadWeights head_weights = LoadHeadWeights(*source.safetensors);

  // Read head params from config (written by convert-kev.py).
  kev::HeadParams head_params;
  head_params.hidden_size = config.hidden_size;
  if (config.raw.contains("kev_head_dim")) {
    head_params.head_dim =
        config.raw["kev_head_dim"].get<int64_t>();
  }
  float temperature = 1.0f;
  if (config.raw.contains("kev_temperature")) {
    temperature = config.raw["kev_temperature"].get<float>();
  }
  return std::make_unique<KevLoadedModel>(
      registration, std::move(weights), std::move(head_weights),
      head_params, temperature, config);
}

void PrepareKev(LoadedModel& model, const HfConfig& config,
                vt::Queue& queue) {
  (void)config;
  ModelAs<KevLoadedModel>(model, "KevModel").set_queue(queue);
}

// The factory forward runs the Qwen3.5 dense backbone to post-final-norm
// hidden states (ForwardDenseHidden) and wraps them as ForwardLogits with
// vocab=hidden_size. The pooling runner consumes these; the PointerHead
// readout (KevReadout) runs in the inference pipeline (Phase 5c), not here.
ForwardLogits ForwardKev(LoadedModel& model, const ModelForwardInput& input) {
  auto& kev = ModelAs<KevLoadedModel>(model, "KevModel");
  std::vector<float> hidden = Qwen3_5DenseModel::ForwardDenseHidden(
      input.token_ids, input.positions, kev.weights(), input.config,
      input.queue);
  return HostLogits(std::move(hidden), input.config.hidden_size);
}

const ModelFactory kKevFactory{
    .parse_config = &ParseQwen3_5Config,
    .load_weights = &LoadKev,
    .prepare = &PrepareKev,
    .forward = &ForwardKev,
    .make_kv_cache = &MakeQwen3_5KVCache,
    .is_dense_model = true,
};

// ── Special token resolution ────────────────────────────────────────────
// kev reuses five Qwen special tokens as delimiters (model.py:18). They are
// already in the tokenizer's added-tokens table, so no embedding rows are
// added. The ID for <|fim_suffix|> is the decide token (pointer query), and
// <|box_end|> is the option key token.
kev::KevSpecialTokens ResolveSpecialTokens(const tok::Tokenizer& tokenizer) {
  kev::KevSpecialTokens st;
  for (const auto& at : tokenizer.AddedTokens()) {
    if (at.text == "<|fim_prefix|>") st.fim_prefix_id = at.id;
    else if (at.text == "<|fim_middle|>") st.fim_middle_id = at.id;
    else if (at.text == "<|box_start|>") st.box_start_id = at.id;
    else if (at.text == "<|box_end|>") st.box_end_id = at.id;
    else if (at.text == "<|fim_suffix|>") st.fim_suffix_id = at.id;
  }
  return st;
}

}  // namespace

// ── Phase 5c: Decision inference ───────────────────────────────────────
// Full decision pipeline: resolve special tokens, tokenize state/instructions/
// options, encode as a causal row, forward the Qwen3.5 dense backbone to
// post-final-norm hidden states, extract the <decide> and per-option <box_end>
// hidden states, apply PointerHead + temperature, and return per-option logits
// (pre-softmax). The server applies softmax and confidence in
// BuildSystemOneAnswerDecision, mirroring the Laya contract.
KevDecisionResult KevInference(
    const LoadedModel& model,
    const tok::Tokenizer& tokenizer,
    const std::string& state,
    const std::string& qtype,
    const std::string& instructions,
    const std::vector<std::string>& options) {
  KevDecisionResult result;
  if (options.empty()) return result;

  // qtype determines the confidence formula applied server-side in
  // BuildSystemOneAnswerDecision; the raw logits are qtype-independent.
  (void)qtype;

  const auto& m = ModelAs<KevLoadedModel>(model, "KevModel");
  vt::Queue queue = m.queue();

  // 1. Resolve special tokens from the tokenizer.
  kev::KevSpecialTokens special = ResolveSpecialTokens(tokenizer);

  // 2. Tokenize state, instructions, and options.
  std::vector<int32_t> state_tokens = tokenizer.Encode(state);
  std::vector<int32_t> instr_tokens = tokenizer.Encode(instructions);
  std::vector<std::vector<int32_t>> option_tokens;
  option_tokens.reserve(options.size());
  for (const std::string& opt : options) {
    option_tokens.push_back(tokenizer.Encode(opt));
  }

  // 3. Encode the question as a causal row.
  kev::KevEncoded enc =
      kev::KevEncodeQuestion(special, state_tokens, instr_tokens, option_tokens);
  if (enc.token_ids.empty()) return result;

  // 4. Forward the Qwen3.5 dense backbone to post-final-norm hidden states.
  std::vector<float> hidden = Qwen3_5DenseModel::ForwardDenseHidden(
      enc.token_ids, enc.positions, m.weights(), m.config(), queue);

  const int64_t H = m.config().hidden_size;
  const int64_t K = static_cast<int64_t>(enc.opt_idx.size());

  // 5. Extract hidden states at the decide and option positions.
  //    h_decide = hidden[decide_idx], h_opts[k] = hidden[opt_idx[k]].
  std::vector<float> h_decide(static_cast<size_t>(H));
  std::copy_n(hidden.data() +
                  static_cast<size_t>(enc.decide_idx) * static_cast<size_t>(H),
              static_cast<size_t>(H), h_decide.data());

  std::vector<float> h_opts(static_cast<size_t>(K) * static_cast<size_t>(H));
  for (int64_t k = 0; k < K; ++k) {
    std::copy_n(hidden.data() +
                    static_cast<size_t>(enc.opt_idx[static_cast<size_t>(k)]) *
                        static_cast<size_t>(H),
                static_cast<size_t>(H),
                h_opts.data() + static_cast<size_t>(k) * static_cast<size_t>(H));
  }

  // 6. PointerHead forward: logits[k] = (k(h_opts[k]) . q(h_decide)) * scale.
  std::vector<float> logits = kev::PointerHeadForward(
      m.head_params(), m.head_weights(), h_decide, h_opts, K);

  // 7. Temperature scaling (matches PointerHead.forward eval mode).
  const float temperature = m.temperature();
  if (temperature != 1.0f) {
    for (float& z : logits) z /= temperature;
  }

  result.scores = std::move(logits);
  result.prompt_tokens = static_cast<int64_t>(enc.token_ids.size());
  return result;
}

REGISTER_VLLM_MODEL(kev_model, "KevModel", kKevFactory, kKevInfo)

}  // namespace vllm
