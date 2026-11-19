// vllm.cpp ORIGINAL — the `hy_v4_flash` GGUF config builder (PARSE-ONLY row
// NEW-CANDIDATE-PARSERS; the forward is owed to NEW-CANDIDATE-PORTS).
//
// The pinned artifact is `/mnt/models/hy4-flash-gguf/Hy4-flash-Q4_K_M-*.gguf`
// (GGUF v3 split over two shards, `general.architecture = "hy_v4_flash"`, 1415
// tensors, 76 metadata KVs; the committed header walk is
// `docs/bench-evidence/new-candidates-parsers-20261119/hy4-kvs.txt`). The
// tencent Hy4-flash family (Engram n-gram tables, compressed indexer, DFLY
// MTP, vision tower) has no llama.cpp or vLLM release arm, so the file's own
// KVs are the key-name record.
//
// Parsers only: the indexer / Engram / PLE semantics are the port row's. This
// builder maps the STANDARD fields by name, carries every unmapped KV
// verbatim under `raw["gguf_unmapped"]`, and validates per-layer array lengths
// against `block_count`.
#pragma once

#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// The GGUF `general.architecture` value this arm dispatches.
inline constexpr const char* kHyV4FlashGgufArch = "hy_v4_flash";

// True when `general.architecture` names this family.
bool IsHyV4FlashGguf(const GgufFile& gguf);

// Build the HfConfig from the file's metadata. Standard fields by name; every
// unmapped KV carried verbatim under `raw["gguf_unmapped"]`. Throws on a
// missing required key or an array-vs-block_count disagreement.
HfConfig HyV4FlashHfConfigFromGguf(const GgufFile& gguf);

// The refuse-at-the-forward message for a model this row PARSES (see
// OlalaForwardRefusal for the boundary this records).
std::string HyV4FlashForwardRefusal();

}  // namespace vllm
