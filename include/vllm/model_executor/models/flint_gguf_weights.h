// vllm.cpp ORIGINAL — the `flint` GGUF arm. See
// .agents/specs/flint-gguf-arm.md and ISSUE-LOCAL-01M5PRSC2E4VR54HBSWFPMN37E.
//
// The published RecursiveSI flint containers ARE llama.cpp-conformant
// (expert_gating_func = 2 = LLAMA_EXPERT_GATING_FUNC_TYPE_SIGMOID at the
// pinned llama-cpp-flint oracle), so llama.cpp here IS a runnable floor and
// supplies the golden greedy chains. Tensor names and KV names follow llama.cpp
// PR #30928 (conversion/flint.py + src/models/flint.cpp); the `flint`
// pre-tokenizer maps through SplitPattern::kFlint (the PR's own regexes,
// ported verbatim).
#pragma once

#include <string>
#include <vector>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/model_executor/models/flint.h"
#include "vllm/transformers_utils/hf_config.h"

namespace vllm {

// The GGUF `general.architecture` value this arm dispatches.
inline constexpr const char* kFlintGgufArch = "flint";

// One expected GGUF tensor name and the role the loader assigns it.
struct FlintGgufTensor {
  std::string name;
  const char* role;
};

// True when `general.architecture` names this family. Mirrors IsKolibri1Gguf:
// the model that owns the family answers the question, so the dispatch in
// model_loader.cpp does not carry a second copy of the arch list.
bool IsFlintGguf(const GgufFile& gguf);

// Build the HfConfig from the file's metadata. The routing keys
// (expert_gating_func, expert_weights_norm, expert_weights_scale) are read and
// REQUIRED — the container must state the routing it was written with. Throws
// on a missing required key.
HfConfig FlintHfConfigFromGguf(const GgufFile& gguf);

// The EXACT set of tensor names the loader reads for `p`.
std::vector<FlintGgufTensor> EnumerateFlintGgufTensors(const FlintParams& p);

// Load the whole model from a GGUF file into the same bf16-owned FlintWeights
// the safetensors loader produces, so the production forward
// (flint.cpp FlintModel::Forward) is unchanged. Every projection lands in the
// bf16 arm in the [K=in, N=out] Matmul-B orientation; the k-quants expand
// through DequantGgufRowToBf16 and are transposed at the load. Throws on a
// missing or unaccounted tensor and on an unsupported ggml type.
FlintWeights LoadFlintFromGguf(const GgufFile& gguf, const HfConfig& config);

// The keep-quant arm (.agents/specs/flint-keepquant.md, issue
// ISSUE-LOCAL-01M5WCPZC4TWR9GZVJHH73E1XE): the SAME FlintWeights, but every
// weight tensor that carries an ADMITTED block encoding (KeepQuantDType AND
// DeviceKeepQuantSupported(dt, kTENSTORRENT)) keeps its VERBATIM GGUF block
// bytes in [N, K] nk order — the MatmulBTQuant orientation, no transpose.
// A weight tensor carrying an inadmissible block encoding refuses BY NAME
// (encoding + tensor); float-carrying tensors (embedding table, norms,
// sinks, a float router) take the bf16 arm exactly as before. The
// `flint_tt.cpp` device arm dispatches per-tensor dtype, and the CPU
// reference forward refuses a keep-quant weight set by name (it consumes
// bf16).
FlintWeights LoadFlintKeepQuantFromGguf(const GgufFile& gguf,
                                        const HfConfig& config);

// The checked accessor over the registry's loaded flint model (the
// kolibri1_registry.cpp seam): reads the weights back out of the LoadedModel
// the production ModelRegistry::Load produced, so a gate can assert on the
// TREE IDENTITY (which arm actually loaded) rather than only on its outputs.
// Throws when `model` is not the flint family.
class LoadedModel;
const FlintWeights& FlintLoadedModelWeights(LoadedModel& model);

}  // namespace vllm
