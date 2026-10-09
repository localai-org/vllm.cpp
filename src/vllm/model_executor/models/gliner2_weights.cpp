// Production weight loader for GLiNER2.5-multi-v1 (MODEL-GLINER25).
//
// Reads all F32 tensors from the safetensors checkpoint, infers the DeBERTa
// encoder config from weight shapes (mdeberta-v3-base defaults for fields the
// checkpoint does not carry), reads the boundary head config from
// config.raw["boundary_head"], and calls the existing host reference loaders
// deberta_v2::Load + gliner2::LoadBoundaryHead.
//
// The checkpoint is 334 F32 tensors (198 encoder + 136 head), all dtype "F32".
// There is no BF16 arm — the dense_weight_loaders.h BF16 helpers do not apply,
// so this loader copies raw F32 bytes straight from the safetensors mmap into
// deberta_v2::CheckpointTensors.
#include "vllm/model_executor/models/gliner2.h"

#include <cstddef>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/safetensors_reader.h"
#include "vllm/transformers_utils/hf_config.h"
#include "vt/dtype.h"

namespace vllm {
namespace {

// Copy one F32 StTensor into the CheckpointTensors map. Throws if the tensor
// is not F32 — the checkpoint carries only F32, so any other dtype is a
// format mismatch the loader should surface.
void LoadF32Tensor(const StTensor& t, const std::string& name,
                   deberta_v2::CheckpointTensors& out) {
  VT_CHECK(t.dtype == "F32",
           "gliner2: tensor '" + name + "' has dtype " + t.dtype +
               ", expected F32");
  VT_CHECK(t.data != nullptr,
           "gliner2: tensor '" + name + "' has null data");
  VT_CHECK(t.nbytes % sizeof(float) == 0,
           "gliner2: tensor '" + name + "' nbytes not a multiple of 4");
  size_t numel = t.nbytes / sizeof(float);
  const float* ptr = reinterpret_cast<const float*>(t.data);
  std::vector<float> data(ptr, ptr + numel);
  std::vector<int64_t> shape(t.shape.begin(), t.shape.end());
  out.Set(name, std::move(shape), std::move(data));
}

// Infer DeBERTa encoder params from weight shapes, with mdeberta-v3-base
// defaults for fields the checkpoint does not carry (num_attention_heads,
// max_position_embeddings, layer_norm_eps, attention flags).
deberta_v2::Params InferEncoderParams(
    const deberta_v2::CheckpointTensors& tensors, const HfConfig& config) {
  deberta_v2::Params p;

  // hidden_size and vocab_size from word_embeddings [vocab, H].
  const auto& we_shape = tensors.Shape("encoder.embeddings.word_embeddings.weight");
  p.vocab_size = we_shape[0];
  p.hidden_size = we_shape[1];

  // num_hidden_layers from counting encoder.encoder.layer.N keys.
  int64_t num_layers = 0;
  while (tensors.Has("encoder.encoder.layer." + std::to_string(num_layers) +
                     ".attention.self.query_proj.weight")) {
    ++num_layers;
  }
  p.num_hidden_layers = num_layers;

  // intermediate_size from layer 0 intermediate.dense.weight [I, H].
  const auto& inter_shape =
      tensors.Shape("encoder.encoder.layer.0.intermediate.dense.weight");
  p.intermediate_size = inter_shape[0];

  // position_buckets from rel_embeddings [pos_ebd_size, H] → pos_ebd_size / 2.
  const auto& rel_shape = tensors.Shape("encoder.encoder.rel_embeddings.weight");
  p.position_buckets = rel_shape[0] / 2;

  // num_attention_heads: the config's own value is authoritative when the
  // checkpoint carries one (the e2e fixture declares 4 heads at hidden_size
  // 24, which the /64 rule below would round to 0 and deberta_v2::Load would
  // refuse). The published GLiNER2.5 configs carry no encoder fields, so the
  // fallback keeps the DeBERTa-v3 head_dim-64 derivation (4b252aff4): DeBERTa-v3
  // always uses head_dim 64, so hidden_size / 64 covers mdeberta-v3-base
  // (768/12) and deberta-v3-large (1024/16) — and equals the config's own
  // value there, so real checkpoints are unchanged either way.
  p.num_attention_heads = config.num_attention_heads > 0
                              ? config.num_attention_heads
                              : p.hidden_size / 64;
  p.max_position_embeddings = 512;
  p.layer_norm_eps = 1e-7;
  p.position_biased_input = false;
  p.type_vocab_size = 0;
  p.norm_rel_ebd = true;   // "layer_norm"
  p.share_att_key = true;
  p.use_c2p = true;
  p.use_p2c = true;

  return p;
}

// Parse boundary head params from config.raw["boundary_head"], with defaults
// matching the published gliner2.5-multi-v1 checkpoint.
gliner2::BoundaryParams ParseBoundaryParams(const HfConfig& config) {
  gliner2::BoundaryParams p;
  const auto& raw = config.raw;
  const auto it = raw.find("boundary_head");
  if (it == raw.end() || !it->is_object()) return p;
  const auto& bh = *it;

  if (bh.contains("boundary_dim") && bh["boundary_dim"].is_number())
    p.boundary_dim = bh["boundary_dim"].get<int64_t>();
  if (bh.contains("boundary_attention_heads") && bh["boundary_attention_heads"].is_number())
    p.boundary_attention_heads = bh["boundary_attention_heads"].get<int64_t>();
  if (bh.contains("boundary_attention_layers") && bh["boundary_attention_layers"].is_number())
    p.boundary_attention_layers = bh["boundary_attention_layers"].get<int64_t>();
  if (bh.contains("boundary_attention_window") && bh["boundary_attention_window"].is_number())
    p.boundary_attention_window = bh["boundary_attention_window"].get<int64_t>();
  if (bh.contains("boundary_refinement_layers") && bh["boundary_refinement_layers"].is_number())
    p.boundary_refinement_layers = bh["boundary_refinement_layers"].get<int64_t>();
  if (bh.contains("boundary_ffn_multiplier") && bh["boundary_ffn_multiplier"].is_number())
    p.boundary_ffn_multiplier = bh["boundary_ffn_multiplier"].get<double>();

  // hidden_size comes from the encoder, set later by the caller.
  return p;
}

}  // namespace

Gliner2ModelWeights LoadGliner2Weights(
    const std::vector<SafetensorsFile>& shards, const HfConfig& config) {
  // Build the name → shard index so each Get(name) finds the right file.
  std::unordered_map<std::string, const SafetensorsFile*> where;
  for (const SafetensorsFile& shard : shards)
    for (const std::string& name : shard.Names()) where[name] = &shard;

  // Read all F32 tensors into CheckpointTensors. The checkpoint is 334 tensors,
  // all F32. Skip the "__metadata__" key (not in Names()).
  deberta_v2::CheckpointTensors tensors;
  for (const auto& [name, shard_ptr] : where) {
    const StTensor& t = shard_ptr->Get(name);
    LoadF32Tensor(t, name, tensors);
  }

  // Infer encoder params from weight shapes.
  deberta_v2::Params enc_params = InferEncoderParams(tensors, config);

  // Parse boundary params from config, then set hidden_size from the encoder.
  gliner2::BoundaryParams bnd_params = ParseBoundaryParams(config);
  bnd_params.hidden_size = enc_params.hidden_size;

  // Load encoder weights via the host reference loader.
  deberta_v2::Weights enc_weights = deberta_v2::Load(enc_params, tensors);

  // Load boundary head weights only if the checkpoint contains
  // boundary_head tensors. The GLiNER2.5-Decide checkpoint does not
  // include a boundary head (it uses the classification head instead).
  gliner2::BoundaryHeadWeights bnd_weights;
  if (tensors.Has("boundary_head.boundary_encoder.left_projection.weight")) {
    bnd_weights = gliner2::LoadBoundaryHead(bnd_params, tensors);
  }

  return Gliner2ModelWeights{
      std::move(enc_params), std::move(enc_weights),
      std::move(bnd_params), std::move(bnd_weights)};
}

}  // namespace vllm
