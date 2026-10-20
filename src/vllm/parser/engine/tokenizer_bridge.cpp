// Local adapter: preserve the real tokenizer's stored and decoded domains.
#include "vllm/parser/engine/tokenizer_bridge.h"

#include <stdexcept>

namespace vllm::parser::engine {
TokenizerBridge::TokenizerBridge(const tok::Tokenizer& tokenizer)
    : tokenizer_(tokenizer) {
  for (int id = 0; id < tokenizer_.VocabSize(); ++id) {
    if (!tokenizer_.HasToken(id)) continue;
    if (!vocab_.emplace(tokenizer_.TokenText(id), id).second) {
      throw std::invalid_argument("ambiguous tokenizer inverse vocabulary");
    }
  }
}

std::string TokenizerBridge::decode(int id) const {
  return tokenizer_.Decode({id});
}

const std::map<std::string, int>& TokenizerBridge::get_vocab() const {
  return vocab_;
}

bool TokenizerBridge::all_special(std::vector<std::string>& tokens,
                                  std::vector<int>& ids) const {
  tokens.clear();
  ids.clear();
  for (int id = 0; id < tokenizer_.VocabSize(); ++id) {
    if (tokenizer_.HasToken(id) && tokenizer_.IsSpecial(id)) {
      tokens.push_back(tokenizer_.TokenText(id));
      ids.push_back(id);
    }
  }
  return true;
}
}  // namespace vllm::parser::engine
