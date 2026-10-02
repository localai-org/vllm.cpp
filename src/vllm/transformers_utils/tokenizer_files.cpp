// vllm.cpp original: where a model directory keeps its tokenizer files.
// See include/vllm/transformers_utils/tokenizer_files.h.
#include "vllm/transformers_utils/tokenizer_files.h"

#include <system_error>

namespace vllm {

std::filesystem::path ResolveTokenizerFile(const std::filesystem::path& model_dir,
                                           const std::string& name) {
  namespace fs = std::filesystem;
  const fs::path root = model_dir / name;
  std::error_code ec;
  // fs::exists, not is_regular_file: the root check keeps the exact predicate
  // the callers used before this fallback existed.
  if (fs::exists(root, ec)) return root;
  const fs::path sub = model_dir / "tokenizer" / name;
  if (fs::is_regular_file(sub, ec)) return sub;
  return root;
}

}  // namespace vllm
