// Ported from: vllm/v1/structured_output/__init__.py @ e24d1b24
//
// Scope (M3.4 Task 2): the engine-level StructuredOutputManager — the
// backend-agnostic driver that (1) compiles a per-request grammar the first
// time a structured request is added (grammar_init), and (2) fills the batched
// per-step token bitmask the scheduler hands to the sampler (grammar_bitmask).
// Name-for-name with upstream: this is the parity-critical seam. The manager
// holds ONE backend (upstream: a single backend for the whole engine), built
// LAZILY the first time a grammar is needed. Task 4 provides the concrete native
// backend; Task 2 is testable with a MOCK backend injected via a factory.
//
// T0 SIMPLIFICATIONS (marked, cite upstream), slotted in later without
// reshaping:
//   - Backend selection: upstream __init__ picks xgrammar/guidance/outlines/
//     lm-format-enforcer off request.sampling_params.structured_outputs._backend
//     (__init__.py:130-165). Here the manager takes a BackendFactory std::function
//     so Task 2 is backend-agnostic (mock in tests, native in Task 4); the
//     factory IS the "which backend" decision, invoked once at first grammar.
//   - SYNCHRONOUS compile: upstream submits _create_grammar to a ThreadPoolExecutor
//     when _use_async_grammar_compilation (__init__.py:167-171); at T0 we compile
//     synchronously (plan Global Constraints) — the async Future path is deferred.
//   - Parallel bitmask fill: upstream has an executor_for_fillmask fast path for
//     >128 structured reqs (__init__.py:236-262); T0 uses only the serial
//     fallback (__init__.py:263-294) — the ThreadPool is deferred.
//   - Diffusion multi-row bitmask is deferred. Spec-decode fills each draft
//     position and the bonus row, temporarily advancing and restoring the FSM.
//   - Reasoning/thinking gating: should_advance / should_fill_bitmask return the
//     no-reasoner branch (reasoner_cls is null at T0) — see their bodies.
#ifndef VLLM_V1_STRUCTURED_OUTPUT_MANAGER_H_
#define VLLM_V1_STRUCTURED_OUTPUT_MANAGER_H_

#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/v1/structured_output/backend_types.h"

namespace vllm::v1 {

struct Request;  // vllm/v1/request.h (manager.cpp includes it).

// StructuredOutputManager (__init__.py:36-378). Engine-level manager for
// structured-output requests: compiles grammars on demand and fills the batched
// per-step token bitmask.
class StructuredOutputManager {
 public:
  // The backend factory: constructs the ONE backend the manager owns, lazily on
  // the first grammar (upstream's `if self.backend is None:` block). Task 4
  // supplies a factory building the native backend; tests supply a mock.
  using BackendFactory =
      std::function<std::unique_ptr<StructuredOutputBackend>()>;

  // Default: no backend factory (structured output effectively off — grammar_init
  // over a structured req would assert; only used where no structured request can
  // reach the manager).
  StructuredOutputManager() = default;

  // max_num_seqs sizes the initial bitmask (upstream
  // vllm_config.scheduler_config.max_num_seqs). backend_factory builds the
  // backend lazily (see BackendFactory).
  StructuredOutputManager(int max_num_seqs, BackendFactory backend_factory);

  // grammar_init (__init__.py:115-171): if the request carries a structured-output
  // constraint, ensure the backend exists (build it via the factory on first use)
  // and SYNCHRONOUSLY compile the request's grammar, storing it on the request's
  // StructuredOutputRequest. No-op for a non-structured request. (Async Future
  // path deferred — see the file header.)
  void grammar_init(Request& request);

  // grammar_bitmask (__init__.py:204-303): fill a batched
  // [sum(1 + drafts per structured req), ceil(vocab/32)] int32 bitmask,
  // in structured_output_request_ids order. A live grammar fills its row
  // via fill_bitmask; a terminated / should-not-fill grammar's row is set to -1
  // (all tokens allowed). Returns nullopt when there are no structured reqs.
  // (Parallel-fill fast path deferred — see the header.)
  std::optional<TokenBitmask> grammar_bitmask(
      const std::map<std::string, std::unique_ptr<Request>>& requests,
      const std::vector<std::string>& structured_output_request_ids,
      const std::map<std::string, std::vector<int32_t>>&
          scheduled_spec_decode_tokens);

  // should_advance (__init__.py:325-373): whether the FSM should advance on this
  // step's sampled tokens. T0 STUB: returns request.use_structured_output()
  // (the no-reasoner branch — reasoner_cls is null at T0; the thinking-mode
  // gating is deferred).
  bool should_advance(const Request& request) const;

  // should_fill_bitmask (__init__.py:305-323): whether to fill (constrain) this
  // request's bitmask row this step. T0 STUB: returns true (the no-reasoner
  // branch — reasoning-mode gating deferred).
  bool should_fill_bitmask(const Request& request) const;

  // clear_backend (__init__.py:375-377): backend-specific cleanup.
  void clear_backend();

  // Test/inspection accessor: the lazily-built backend (null until first grammar).
  StructuredOutputBackend* backend() const { return backend_.get(); }

 private:
  // _create_grammar (__init__.py:173-184): compile the request's (type, spec) key
  // into a grammar via the backend.
  std::unique_ptr<StructuredOutputGrammar> create_grammar(Request& request);

  // _fill_bitmasks single-row helper (__init__.py:186-197): live+non-terminated
  // grammar fills its row; otherwise the row is set to the full (-1, all-allowed)
  // mask.
  void fill_bitmask_row(StructuredOutputGrammar& grammar, int index,
                        bool apply_bitmask);

  // The single owned backend, built lazily on first grammar (__init__.py:40).
  std::unique_ptr<StructuredOutputBackend> backend_;
  BackendFactory backend_factory_;
  // Initial row capacity (upstream scheduler_config.max_num_seqs).
  int max_num_seqs_ = 0;
  // The reused bitmask grows when expanded speculative rows need more capacity.
  std::optional<TokenBitmask> grammar_bitmask_;
  // _full_mask = -1: every bit set => every token allowed (__init__.py:59).
  static constexpr int32_t kFullMask = -1;
};

}  // namespace vllm::v1

#endif  // VLLM_V1_STRUCTURED_OUTPUT_MANAGER_H_
