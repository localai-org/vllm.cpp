// Ported from: vllm/v1/structured_output/__init__.py @ e24d1b24
// See include/vllm/v1/structured_output/manager.h for scope + T0 simplifications.
#include "vllm/v1/structured_output/manager.h"

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <stdexcept>
#include <utility>

#include "vllm/v1/request.h"
#include "vllm/v1/structured_output/request.h"

namespace vllm::v1 {

StructuredOutputManager::StructuredOutputManager(int max_num_seqs,
                                                 BackendFactory backend_factory)
    : backend_factory_(std::move(backend_factory)),
      max_num_seqs_(max_num_seqs) {}

void StructuredOutputManager::grammar_init(Request& request) {
  // __init__.py:116-117: no-op for a non-structured request.
  if (!request.structured_output_request.has_value()) {
    return;
  }

  // __init__.py:130-165: initialize the backend the first time it is needed.
  // Upstream selects the backend class off the request's `_backend` field; here
  // the injected factory IS that decision (see the header). We only support a
  // single backend for the whole engine (upstream NOTE at :127-129).
  if (backend_ == nullptr) {
    assert(backend_factory_ &&
           "grammar_init: a structured request reached a manager with no "
           "backend factory");
    backend_ = backend_factory_();
  }

  // __init__.py:167-171: async ThreadPoolExecutor submit is deferred (plan Global
  // Constraints) — compile SYNCHRONOUSLY and store the owned grammar handle.
  request.structured_output_request->grammar = create_grammar(request);
}

std::unique_ptr<StructuredOutputGrammar> StructuredOutputManager::create_grammar(
    Request& request) {
  // __init__.py:173-184.
  const StructuredOutputKey key =
      request.structured_output_request->structured_output_key();
  assert(backend_ != nullptr);
  return backend_->compile_grammar(key.first, key.second);
}

void StructuredOutputManager::fill_bitmask_row(StructuredOutputGrammar& grammar,
                                               int index, bool apply_bitmask) {
  // __init__.py:186-197 (_fill_bitmasks, single-row).
  assert(grammar_bitmask_.has_value());
  if (apply_bitmask && !grammar.is_terminated()) {
    grammar.fill_bitmask(*grammar_bitmask_, index);
  } else {
    // Terminated / not-to-be-filled: the whole row is the full mask (-1 => all
    // tokens allowed). (Thinking-support partial reset deferred, per upstream.)
    const std::size_t base =
        static_cast<std::size_t>(index) * grammar_bitmask_->num_words;
    for (int w = 0; w < grammar_bitmask_->num_words; ++w) {
      grammar_bitmask_->data[base + static_cast<std::size_t>(w)] = kFullMask;
    }
  }
}

std::optional<TokenBitmask> StructuredOutputManager::grammar_bitmask(
    const std::map<std::string, std::unique_ptr<Request>>& requests,
    const std::vector<std::string>& structured_output_request_ids,
    const std::map<std::string, std::vector<int32_t>>&
        scheduled_spec_decode_tokens) {
  // __init__.py:210-212: nothing to do without structured reqs.
  if (structured_output_request_ids.empty()) {
    return std::nullopt;
  }

  // A structured request contributes one row per draft plus its bonus row.
  // Grow on demand; max_num_seqs alone is insufficient for MTP verification.
  int required_rows = static_cast<int>(structured_output_request_ids.size());
  for (const auto& req_id : structured_output_request_ids) {
    const auto it = scheduled_spec_decode_tokens.find(req_id);
    if (it != scheduled_spec_decode_tokens.end()) {
      required_rows += static_cast<int>(it->second.size());
    }
  }
  if (!grammar_bitmask_.has_value() ||
      grammar_bitmask_->num_seqs < required_rows) {
    assert(backend_ != nullptr);
    grammar_bitmask_ = backend_->allocate_token_bitmask(
        std::max(max_num_seqs_, required_rows));
  }

  // __init__.py:232, 263-294: serial fill (the >128-req parallel fast path is
  // deferred — see the header). One structured req per row, in the given order.
  int cumulative_index = 0;
  for (const std::string& req_id : structured_output_request_ids) {
    Request& request = *requests.at(req_id);
    assert(request.structured_output_request.has_value());
    assert(request.structured_output_request->grammar != nullptr);
    StructuredOutputGrammar& grammar =
        *request.structured_output_request->grammar;

    bool apply_bitmask = should_fill_bitmask(request);

    // __init__.py:276-282: spec-decode tokens followed by the bonus row.
    std::vector<int32_t> token_iter;
    auto spec_it = scheduled_spec_decode_tokens.find(req_id);
    if (spec_it != scheduled_spec_decode_tokens.end()) {
      token_iter = spec_it->second;
    }
    token_iter.push_back(-1);  // the bonus / non-speculative placeholder.

    int state_advancements = 0;
    try {
      for (const int32_t token : token_iter) {
        fill_bitmask_row(grammar, cumulative_index, apply_bitmask);
        if (token == -1) {
          // __init__.py:285-287: stop advancing once we hit the padding token.
          apply_bitmask = false;
        }
        if (apply_bitmask && !grammar.is_terminated()) {
          const bool accepted = grammar.accept_tokens(req_id, {token});
          if (!accepted) {
            throw std::runtime_error(
                "grammar rejected a scheduled spec-decode token for " + req_id);
          }
          ++state_advancements;
        }
        ++cumulative_index;
      }
    } catch (...) {
      grammar.rollback(state_advancements);
      throw;
    }
    if (state_advancements > 0) {
      // __init__.py:293-294: undo the FSM advances made purely to fill the
      // per-position spec bitmasks (the real advance happens in update_from_output).
      grammar.rollback(state_advancements);
    }
  }

  // __init__.py:296-303: return the first cumulative_index rows (a fresh copy,
  // mirroring the `.numpy()` copy upstream makes for serialization).
  TokenBitmask out;
  out.num_words = grammar_bitmask_->num_words;
  out.num_seqs = cumulative_index;
  const std::size_t count =
      static_cast<std::size_t>(cumulative_index) * grammar_bitmask_->num_words;
  out.data.assign(grammar_bitmask_->data.begin(),
                  grammar_bitmask_->data.begin() +
                      static_cast<std::ptrdiff_t>(count));
  return out;
}

bool StructuredOutputManager::should_advance(const Request& request) const {
  // __init__.py:325-373. T0 STUB: `if not use_structured_output: return False`,
  // then the reasoner is null (reasoner_cls unset at T0) so the `reasoner is
  // None -> return True` branch applies. The thinking-mode / reasoning-ended
  // gating (:340-373) is deferred (plan Global Constraints).
  return request.use_structured_output();
}

bool StructuredOutputManager::should_fill_bitmask(const Request& request) const {
  // __init__.py:305-323. T0 STUB: the reasoner is null (reasoner_cls unset at
  // T0), so upstream's `return True` fall-through applies. The
  // enable_in_reasoning / reasoning_ended gating is deferred.
  (void)request;
  return true;
}

void StructuredOutputManager::clear_backend() {
  // __init__.py:375-377.
  if (backend_ != nullptr) {
    backend_->destroy();
  }
}

}  // namespace vllm::v1
