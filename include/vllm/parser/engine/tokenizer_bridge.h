// Local production tokenizer adaptation, not an upstream tokenizer policy.
#pragma once
#include "vllm/parser/engine/token_id_scanner.h"
#include "vllm/tokenizer/tokenizer.h"
namespace vllm::parser::engine {
// Caller must keep tokenizer immutable, unmoved and alive longer than this
// bridge, and keep the bridge alive longer than every borrowing parser.
// Rejects distinct assigned IDs with identical stored text (inverse ambiguity).
class TokenizerBridge final : public EngineTokenizer {
 public:
  explicit TokenizerBridge(const tok::Tokenizer& tokenizer);
  TokenizerBridge(tok::Tokenizer&&) = delete;
  std::string decode(int id) const override;
  const std::map<std::string, int>& get_vocab() const override;
  bool all_special(std::vector<std::string>& tokens,
                   std::vector<int>& ids) const override;
 private:
  const tok::Tokenizer& tokenizer_;
  std::map<std::string, int> vocab_;
};
}  // namespace vllm::parser::engine
