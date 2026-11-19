// vllm.cpp ORIGINAL — the `olala` GGUF config builder (PARSE-ONLY row
// NEW-CANDIDATE-PARSERS; the forward is owed to NEW-CANDIDATE-PORTS).
//
// The pinned artifact is `/mnt/models/olala-v1-mini/Olala-v1.0-mini-7A1B-*.gguf`
// (GGUF v3, `general.architecture = "olala"`, 989 tensors, 52 metadata KVs;
// the full header walk is committed at
// `docs/bench-evidence/new-candidates-parsers-20261119/olala-kvs.txt`). The
// enumeration is THIS row's source of truth for key names — no upstream
// llama.cpp or vLLM release carries an `olala` arm, so there is no container
// oracle to mirror and the file's own KVs are the record.
//
// The builder maps the STANDARD fields by name and carries every KV it did not
// map verbatim under `raw["gguf_unmapped"]` with its GGUF key spelling. It does
// NOT invent semantics: `olala.layer_types` is a compact 'M'/'V' schedule whose
// expansion (mamba vs full attention) belongs to the port row, so the string is
// carried and `layer_types` stays EMPTY.
#pragma once

#include <string>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// The GGUF `general.architecture` value this arm dispatches.
inline constexpr const char* kOlalaGgufArch = "olala";

// True when `general.architecture` names this family. Mirrors IsKolibri1Gguf:
// the model that owns the family answers the question, so the dispatch in
// model_loader.cpp does not carry a second copy of the arch list.
bool IsOlalaGguf(const GgufFile& gguf);

// Build the HfConfig from the file's metadata. Standard fields by name; every
// unmapped KV carried verbatim under `raw["gguf_unmapped"]`. Throws on a
// missing required key or a geometry that disagrees with the tensor table.
HfConfig OlalaHfConfigFromGguf(const GgufFile& gguf);

// The refuse-at-the-forward message for a model this row PARSES. The parser
// succeeded; the FORWARD is owed to row NEW-CANDIDATE-PORTS. Returned rather
// than thrown so the loader and the tests read the same text.
std::string OlalaForwardRefusal();

}  // namespace vllm
