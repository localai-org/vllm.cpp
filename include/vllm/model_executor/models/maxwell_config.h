// vllm.cpp ORIGINAL — the `maxwell_moe` config.json builder (PARSE-ONLY row
// NEW-CANDIDATE-PARSERS; the forward is owed to NEW-CANDIDATE-PORTS).
//
// The pinned artifact is `/mnt/models/maxwell-v1-flash/snapshot/` — 35
// safetensors shards plus a config.json whose `auto_map` names REMOTE CODE
// (configuration_maxwell_moe.MaxwellMoeConfig /
// modeling_maxwell_moe.MaxwellMoeForCausalLM). The REMOTE CODE IS NEVER
// EXECUTED here: the config.json's explicit fields are the only source, and
// every field the remote config class would default is recorded UNKNOWN under
// `raw["maxwell_unknown"]` rather than guessed.
//
// The committed field map and shard-header walk are
// docs/bench-evidence/new-candidates-parsers-20261119/maxwell-fields.txt.
#pragma once

#include <string>

#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// True when the parsed config names the Maxwell family (model_type or the
// auto_map's architectures entry).
bool IsMaxwellMoeConfig(const HfConfig& config);

// Full parse over the model directory: validates the typed fields the generic
// LoadHfConfig already read, opens EVERY shard's safetensors header (header
// reads only — no weight bytes move) and checks the geometry the declared
// counts imply, records the remote-code-defaulted fields as UNKNOWN, and
// carries the unmapped config.json fields verbatim in `raw`. Throws on a
// missing shard, a missing per-layer tensor, or a shape disagreement.
HfConfig MaxwellConfigFromJson(const HfConfig& config,
                               const std::string& model_dir);

// The refuse-at-the-forward message for a model this row PARSES.
std::string MaxwellForwardRefusal();

}  // namespace vllm
