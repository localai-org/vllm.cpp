// vllm.cpp ORIGINAL — the `kolibri1` GGUF arm. See
// .agents/specs/kolibri1-gguf-arch.md and
// ISSUE-LOCAL-01M5HJRRHBYMRX4G4JXXKCBFJS.
//
// The published Hob-forge Kolibri-1 containers are NOT llama.cpp-conformant
// (expert_gating_func = 5 aborts llama.cpp's build_moe_ffn), so llama.cpp is
// not the floor; the ROUTE A decision (commit 39dadc637) gates the file AS
// PUBLISHED and never reads that key. mistral.rs is the working reference
// implementation (mistralrs-core/src/gguf/normal_config.rs build_kolibri1).
// Tensor names follow llama.cpp PR #30928; the tokenizer pre name is
// mistralrs gguf_tokenizer.rs:364, byte-identical to this tree's
// kClassicQwen2N1Regex.
#pragma once

#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/kolibri1_weights.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// The GGUF `general.architecture` value this arm dispatches.
inline constexpr const char* kKolibri1GgufArch = "kolibri1";

// True when `general.architecture` names this family. Mirrors IsQwen4ExpGguf:
// the model that owns the family answers the question, so the dispatch in
// model_loader.cpp does not carry a second copy of the arch list.
bool IsKolibri1Gguf(const GgufFile& gguf);

// Build the HfConfig from the file's metadata. The routing keys
// (expert_gating_func) are deliberately NOT read — see the spec. Throws on a
// missing required key.
HfConfig Kolibri1HfConfigFromGguf(const GgufFile& gguf);

// The EXACT set of tensor names the loader reads for `p`.
std::vector<Kolibri1Tensor> EnumerateKolibri1GgufTensors(const Kolibri1Params& p);

// Load the whole model from a GGUF file into the same bf16-owned
// Kolibri1Weights the safetensors loader produces, so the production forward
// (LinearBT / LinearBTRaw) is unchanged. Every projection lands in the bf16
// arm; the k-quants (Q5_0, Q6_K) expand through DequantGgufRowToBf16. Throws
// on a missing or unaccounted tensor and on an unsupported ggml type.
Kolibri1Weights LoadKolibri1FromGguf(const GgufFile& gguf, const HfConfig& config);

}  // namespace vllm
