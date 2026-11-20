// vllm.cpp ORIGINAL — the `olala` GGUF weights arm (name-map + census), row
// NEW-CANDIDATE-PORTS W1. See .agents/specs/new-candidate-ports-olala.md and
// docs/bench-evidence/new-candidate-ports-olala-20261120/olala-namemap.md
// (the committed census of the pinned artifact this map is written against).
//
// W1 scope: classify and validate the tensor table — every tensor maps to a
// role, counts and shapes agree with the parsed HfConfig. The FORWARD still
// refuses by name (OlalaForwardRefusal); loading weights into a model struct
// is the W2+ work. No upstream (vLLM or llama.cpp) carries an `olala` arm, so
// the file's own tensor names are the record.
#pragma once

#include <cstdint>
#include <map>
#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// The role the loader assigns one GGUF tensor. Order is stable for reports.
enum class OlalaTensorRole {
  kEmbed,          // token_embd.weight        [vocab, hidden]
  kLmHead,         // output.weight            [vocab, hidden] (untied)
  kAttnOutput,     // blk.N.attn_output.weight [hidden, inner] (M: 3072, V: 4608)
  kRouterBias,     // blk.N.exp_probs_b        [num_experts] F32
  kRouterWeight,   // blk.N.ffn_gate_inp.weight [num_experts, hidden] F32
  kFfnUpExps,      // blk.N.ffn_up_exps.weight   [experts, latent, ff]
  kFfnDownExps,    // blk.N.ffn_down_exps.weight [experts, ff, latent]
  kFfnUpShexp,     // blk.N.ffn_up_shexp.weight  [hidden, shared_ff]
  kFfnDownShexp,   // blk.N.ffn_down_shexp.weight [shared_ff, hidden]
  kFfnLatentUp,    // blk.N.ffn_latent_up.weight [hidden, latent]
  kFfnLatentDown,  // blk.N.ffn_latent_down.weight [latent, hidden]
  kGeoMixerBias, kGeoMixerScale, kGeoMlpBias, kGeoMlpScale,  // blk.N.geo_* [1]
  // M-layer (SSM) only:
  kSsmIn,          // blk.N.ssm_in.weight       [6288, hidden]
  kSsmInDyn,       // blk.N.ssm_in_dyn.weight   [1056, hidden]
  kSsmBBias, kSsmBNorm, kSsmCBias, kSsmCNorm, kSsmD, kSsmDtBias,
  kSsmM3Bias, kSsmM3Misc, kSsmM3Mxz, kSsmM3Norms,
  kSsmMimoX, kSsmMimoZ, kSsmMimoO,
  // V-layer (full attention) only:
  kAttnQ,          // blk.N.attn_q.weight       [heads*key_length, hidden]
  kAttnGate, kAttnQNorm, kAttnKNorm, kAttnLambda, kAttnShiftK, kAttnShiftV,
  kAttnSoftmaxScaler, kAttnWaK, kAttnWaV, kAttnWbK, kAttnWbV,
  kUnmapped,       // anything the map does not know — listed by name
};

const char* OlalaRoleName(OlalaTensorRole role);

struct OlalaCensusEntry {
  std::string name;
  OlalaTensorRole role = OlalaTensorRole::kUnmapped;
  uint32_t ggml_type = 0;
  std::vector<int64_t> shape;
  int64_t layer = -1;  // -1 for globals
};

struct OlalaGgufCensus {
  std::vector<OlalaCensusEntry> entries;
  std::map<OlalaTensorRole, int64_t> role_counts;
  std::vector<std::string> unmapped;  // names with role kUnmapped
  int64_t num_layers = 0;
  std::vector<int64_t> m_layers;  // SSM layers (no attn_q)
  std::vector<int64_t> v_layers;  // full-attention layers (attn_q present)
};

// Classify every tensor in the file. Never throws on unknown names — they
// land in `unmapped` so the report is loud, not silent.
OlalaGgufCensus CensusOlalaGguf(const GgufFile& gguf);

// Throw std::runtime_error (naming the offenders) unless the census is clean
// AND every count/shape agrees with `cfg` (as built by OlalaHfConfigFromGguf).
void ValidateOlalaGguf(const GgufFile& gguf, const HfConfig& cfg);

}  // namespace vllm
