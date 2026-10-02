// vllm.cpp original: where a model directory keeps its tokenizer files.
//
// Most HuggingFace checkpoints keep `tokenizer.json` and
// `tokenizer_config.json` at the root of the model directory. Some keep them in
// a `tokenizer/` subdirectory instead, for example `convaiinnovations/laya`
// (MODEL-LAYA), whose snapshot has `tokenizer/tokenizer.json` and
// `tokenizer/tokenizer_config.json` and no root copy. vLLM reaches such a
// checkpoint only when the user passes `--tokenizer <dir>/tokenizer`; this
// loader has no such flag, so it looks in the subdirectory itself.
//
// ISSUE-LOCAL-01M3SDXGYYTS9FDXKZDAE24N0R.
#pragma once

#include <filesystem>
#include <string>

namespace vllm {

// The path of tokenizer file `name` (for example "tokenizer.json") for the
// model directory `model_dir`.
//
// The root file wins whenever it exists, so every layout that loaded before
// loads the same file. `<model_dir>/tokenizer/<name>` is used only when the root
// file is absent and the subdirectory file exists. When neither exists the
// result is the root path, so a caller's existing "absent" handling and error
// messages do not change.
std::filesystem::path ResolveTokenizerFile(const std::filesystem::path& model_dir,
                                           const std::string& name);

}  // namespace vllm
