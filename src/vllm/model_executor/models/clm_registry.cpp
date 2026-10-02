// CLM (Contrastive-LM) System 1 decision model — registry (MODEL-CLM).
//
// Bi-encoder: Qwen3-8B frozen backbone (last-token pooling) + two MLP
// projection heads (state head + action head), L2-normalized, scored via
// scaled cosine similarity. The heads, text construction and answers live in
// clm_head.cpp; this TU owns loading, the backbone forward and ClmDecide.
//
// Ported from Contrastive-LM/CLM @ bb42c6c (spec .agents/specs/clm.md).

#include "vllm/model_executor/models/clm.h"
#include "vllm/model_executor/models/clm_inference.h"

#include "vllm/model_executor/model_loader/safetensors_reader.h"  // SafetensorsFile, StTensor
#include "vllm/model_executor/models/dense_device_glue.h"  // Dev, DBuf
#include "vllm/model_executor/models/model_registry.h"
#include "vllm/model_executor/models/qwen3.h"
#include "vllm/tokenizer/tokenizer.h"  // tok::Tokenizer
#include "vllm/transformers_utils/hf_config.h"
#include "vllm/v1/attention/backend.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vt/device.h"
#include "vt/tensor.h"

#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace vllm {
namespace {

// ── ModelInfo ────────────────────────────────────────────────────────────
inline constexpr ModelInfo kClmInfo{
    .is_text_generation_model = false,
    .is_pooling_model = true,
    .is_hybrid = false,
    .has_inner_state = false,
    .supports_multimodal = false,
    .score_type = "bi-encoder",
};

// ── Loaded model ──────────────────────────────────────────────────────────
class ClmLoadedModel final : public LoadedModel {
 public:
  ClmLoadedModel(const ModelRegistration& registration,
                 Qwen3DenseWeights weights,
                 clm::HeadWeights head_weights,
                 clm::HeadParams head_params,
                 HfConfig config)
      : LoadedModel(registration),
        weights_(std::move(weights)),
        head_weights_(std::move(head_weights)),
        head_params_(head_params),
        config_(std::move(config)) {}

  const Qwen3DenseWeights& weights() const { return weights_; }
  const clm::HeadWeights& head_weights() const { return head_weights_; }
  const clm::HeadParams& head_params() const { return head_params_; }
  const HfConfig& config() const { return config_; }

  void set_queue(vt::Queue q) { queue_ = q; }
  vt::Queue queue() const { return queue_; }

 private:
  Qwen3DenseWeights weights_;
  clm::HeadWeights head_weights_;
  clm::HeadParams head_params_;
  HfConfig config_;
  vt::Queue queue_;
};

// ── Load ──────────────────────────────────────────────────────────────────
std::unique_ptr<LoadedModel> LoadClm(
    const ModelRegistration& registration, const HfConfig& config,
    const ModelSource& source) {
  if (source.kind != ModelSource::Kind::kSafetensors) {
    throw std::runtime_error(
        "Model architecture ClmModel does not support GGUF weights");
  }
  if (source.safetensors == nullptr) {
    throw std::runtime_error("safetensors model source is empty");
  }

  // Load merged Qwen3 dense weights (Qwen3-8B backbone).
  Qwen3DenseWeights weights =
      LoadQwen3ForCausalLMWeights(*source.safetensors, config);

  // The head config and raw logit_scale are clm_* keys of config.json, and
  // the heads are head.safetensors, both written by scripts/convert-clm.py.
  // A missing key or tensor is refused by name: a defaulted scale or a
  // skipped tensor loads and answers wrong.
  clm::HeadParams head_params =
      clm::ParseHeadParams(config.raw, config.hidden_size);
  clm::HeadWeights head_weights =
      clm::LoadHeadWeights(*source.safetensors, head_params);

  return std::make_unique<ClmLoadedModel>(
      registration, std::move(weights), std::move(head_weights),
      head_params, config);
}

// ── Prepare ──────────────────────────────────────────────────────────────
void PrepareClm(LoadedModel& model, const HfConfig& config,
                vt::Queue& queue) {
  (void)config;
  ModelAs<ClmLoadedModel>(model, "ClmModel").set_queue(queue);
}

// ── Forward (factory) ─────────────────────────────────────────────────────
// The factory forward delegates to Qwen3DenseModel::ForwardHidden, using the
// runner-provided attn_meta / attn_kv (same pattern as llama_embedding_registry).
ForwardLogits ForwardClm(LoadedModel& model, const ModelForwardInput& input) {
  auto& clm = ModelAs<ClmLoadedModel>(model, "ClmModel");
  return Qwen3DenseModel::ForwardHidden(
      input.token_ids, input.positions, input.attn_meta, input.attn_kv,
      clm.weights(), input.config, input.queue, input.logits_indices);
}

const ModelFactory kClmFactory{
    .parse_config = &ParseQwen3ForCausalLMConfig,
    .load_weights = &LoadClm,
    .prepare = &PrepareClm,
    .forward = &ForwardClm,
    .make_kv_cache = &MakeQwen3ForCausalLMKVCache,
    .is_dense_model = true,
};

// ── Single-seq hidden forward for inference ──────────────────────────────
// Constructs minimal CommonAttentionMetadata + PagedKvCache for a single
// prefill, then calls Qwen3DenseModel::ForwardHidden. This is the CLM
// analogue of kev's direct ForwardDenseHidden call — needed because Qwen3
// dense (unlike Qwen3.5) has no single-seq ForwardDenseHidden helper.
std::vector<float> ForwardClmHidden(
    const std::vector<int32_t>& token_ids,
    const Qwen3DenseWeights& weights,
    const HfConfig& config, vt::Queue& queue) {
  const int64_t T = static_cast<int64_t>(token_ids.size());
  if (T == 0) return {};

  // Positions [0, T)
  std::vector<int32_t> positions(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    positions[static_cast<size_t>(i)] = static_cast<int32_t>(i);

  // Construct CommonAttentionMetadata for a single-seq prefill.
  const int block_size = 32;
  const int num_blocks =
      static_cast<int>((T + block_size - 1) / block_size);
  const int num_kv_heads = static_cast<int>(config.num_key_value_heads);
  const int head_dim = static_cast<int>(config.head_dim);

  v1::CommonAttentionMetadata am;
  am.num_reqs = 1;
  am.num_actual_tokens = static_cast<int>(T);
  am.max_query_len = static_cast<int>(T);
  am.max_seq_len = static_cast<int>(T);
  am.query_start_loc = {0, static_cast<int32_t>(T)};
  am.query_start_loc_cpu = am.query_start_loc;
  am.seq_lens = {static_cast<int32_t>(T)};
  am.seq_lens_cpu = am.seq_lens;
  am.causal = true;
  am.block_table_num_cols = num_blocks;
  am.block_table_tensor.resize(static_cast<size_t>(num_blocks));
  for (int b = 0; b < num_blocks; ++b)
    am.block_table_tensor[static_cast<size_t>(b)] = b;
  am.slot_mapping.resize(static_cast<size_t>(T));
  for (int64_t i = 0; i < T; ++i)
    am.slot_mapping[static_cast<size_t>(i)] = i;

  // Allocate PagedKvCache (one per layer). Each cache holds K and V for
  // [num_blocks, block_size, num_kv_heads, head_dim].
  dense_attn::Dev d{vt::GetBackend(queue.device.type), queue};
  const int64_t kv_elems_per_layer =
      static_cast<int64_t>(num_blocks) * block_size *
      num_kv_heads * head_dim * 2;  // *2 for K and V
  std::vector<PagedKvCache> attn_kv(
      static_cast<size_t>(config.num_hidden_layers));
  std::vector<dense_attn::DBuf> kv_bufs(static_cast<size_t>(config.num_hidden_layers));
  for (int64_t l = 0; l < config.num_hidden_layers; ++l) {
    kv_bufs[static_cast<size_t>(l)] =
        dense_attn::DBuf(d, dense_attn::DType::kBF16, {kv_elems_per_layer});
    kv_bufs[static_cast<size_t>(l)].Zero(d);
    auto& kv = attn_kv[static_cast<size_t>(l)];
    kv.data = kv_bufs[static_cast<size_t>(l)].ptr();
    kv.dtype = vt::DType::kBF16;
    kv.num_blocks = num_blocks;
    kv.block_size = block_size;
    kv.num_kv_heads = num_kv_heads;
    kv.head_size = head_dim;
  }

  // Forward and extract the last-token hidden [H] from [T, H].
  ForwardLogits fl = Qwen3DenseModel::ForwardHidden(
      token_ids, positions, am, attn_kv, weights, config, queue);
  return fl.host;  // [T, H] f32
}

}  // namespace

std::vector<float> ClmLastHidden(const LoadedModel& model,
                                 const std::vector<int32_t>& token_ids) {
  const auto& m = ModelAs<ClmLoadedModel>(model, "ClmModel");
  if (token_ids.empty()) {
    throw std::runtime_error("clm: cannot encode an empty token sequence");
  }
  vt::Queue queue = m.queue();
  const std::vector<float> all =
      ForwardClmHidden(token_ids, m.weights(), m.config(), queue);
  const size_t H = static_cast<size_t>(m.config().hidden_size);
  const size_t last = (token_ids.size() - 1) * H;
  return std::vector<float>(all.begin() + static_cast<std::ptrdiff_t>(last),
                            all.begin() + static_cast<std::ptrdiff_t>(last + H));
}

nlohmann::ordered_json ClmDecide(const LoadedModel& model,
                                 const tok::Tokenizer& tokenizer,
                                 const nlohmann::ordered_json& body) {
  const auto& m = ModelAs<ClmLoadedModel>(model, "ClmModel");
  auto embed = [&](const std::string& text, int64_t* tokens) {
    // vLLM /v1/embeddings tokenizes with add_special_tokens=True, then
    // truncate_prompt_tokens keeps the first kClmMaxTokens.
    std::vector<int32_t> ids = tokenizer.EncodeWithSpecialTokens(text);
    if (static_cast<int64_t>(ids.size()) > kClmMaxTokens) {
      ids.resize(static_cast<size_t>(kClmMaxTokens));
    }
    if (ids.empty()) {
      // An empty text has no last token. The reference sends it to vLLM,
      // which refuses an empty prompt; refuse it here by name instead.
      throw clm::RequestError("clm: a state or candidate text is empty");
    }
    *tokens += static_cast<int64_t>(ids.size());
    return ClmLastHidden(model, ids);
  };
  return clm::Answer(m.head_weights(), m.head_params(), body, embed);
}

std::unique_ptr<LoadedModel> MakeClmLoadedModel(Qwen3DenseWeights weights,
                                                clm::HeadWeights heads,
                                                clm::HeadParams params,
                                                HfConfig config, vt::Queue queue) {
  auto model = std::make_unique<ClmLoadedModel>(
      RegistrationFor("ClmModel"), std::move(weights), std::move(heads), params,
      std::move(config));
  model->set_queue(queue);
  return model;
}

REGISTER_VLLM_MODEL(clm_model, "ClmModel", kClmFactory, kClmInfo)

}  // namespace vllm
