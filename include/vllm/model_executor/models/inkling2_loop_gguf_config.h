// vllm.cpp ORIGINAL — the `inkling2_loop` GGUF config builder (PARSE-ONLY W0 of
// row NEW-CANDIDATE-PORTS; the looped forward is owed to a later wave).
//
// The pinned artifact is
// `/mnt/models/inkling2-mini-apex-gguf/Inkling-2-Mini-APEX-Balanced.gguf`
// (GGUF v3, `general.architecture = "inkling2_loop"`, 704 tensors, 59 metadata
// KVs; full walk at
// docs/bench-evidence/new-candidate-inkling2-20261121/apex-balanced-walk.txt).
//
// Naming/validation mirror the primary oracle vLLM PR #69028
// (vllm/models/inkling2/configs.py): model_type `inkling2_loop`, architecture
// `Inkling2LoopForConditionalGeneration`, and a `loop_config` whose fields use
// upstream's spelling (core_start/core_end/num_passes/gate_rank/is_varloop).
// The nested HF `text_config` shape is port-row scope.
#pragma once

#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// The GGUF `general.architecture` value this arm dispatches.
inline constexpr const char* kInkling2LoopGgufArch = "inkling2_loop";

// True when `general.architecture` names this family. The model that owns the
// family answers the question so the loader's dispatch holds no second list.
bool IsInkling2LoopGguf(const GgufFile& gguf);

// Build the HfConfig from the file's metadata. Standard fields by name, upstream
// loop_config fields under raw["loop_config"], every unmapped KV carried
// verbatim under raw["gguf_unmapped"]. Throws on a missing required key or a
// loop configuration that disagrees with the block count / upstream rules.
HfConfig Inkling2LoopHfConfigFromGguf(const GgufFile& gguf);

// The refuse-at-the-forward message for a file this row PARSES. Returned rather
// than thrown so the loader and the tests read the same text.
std::string Inkling2LoopForwardRefusal();

}  // namespace vllm
