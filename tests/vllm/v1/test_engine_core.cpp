// Tests for the V1 EngineCore step loop (vllm/v1/engine/core.py @ e24d1b24)
// and the Executor pass-through seam (vllm/v1/executor/abstract.py +
// uniproc_executor.py @ e24d1b24).
//
// M1.8 Task 1 is a pure control-flow port (no goldens): drive EngineCore.step()
// over the REAL M1.4 Scheduler + a runner STUB (a ModelRunnerBase test double
// that records the SchedulerOutput it received and returns a canned
// ModelRunnerOutput — one sampled token per scheduled request). Cases:
//   (a) empty scheduler        -> step() returns ({}, false); runner not called.
//   (b) one request scheduled  -> execute_model + sample_tokens called (in that
//                                 order) with the scheduled SchedulerOutput;
//                                 update_from_output fed the ModelRunnerOutput
//                                 (the sampled token appended, EngineCoreOutputs
//                                 returned); model_executed == true.
//   (c) abort_requests         -> the request is finished (removed from the
//                                 scheduler; next step early-returns).
//   (d) call ORDER              -> schedule -> execute -> sample -> update.
#include <doctest/doctest.h>

#include <map>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/core.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/executor/executor.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/manager.h"
#include "vllm/v1/worker/gpu/model_runner_base.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::v1::EngineCore;
using vllm::v1::EngineCoreOutputs;
using vllm::v1::Executor;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerBase;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::Request;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vllm::v1::StructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputManager;
using vllm::v1::StructuredOutputOptions;
using vllm::v1::TokenBitmask;
using vt::DType;

namespace {

// The canned token the stub "samples" for every scheduled request.
constexpr int32_t kCannedToken = 42;

// A ModelRunnerBase test double mirroring the MRV2 execute_model / sample_tokens
// split: execute_model STASHES the SchedulerOutput and returns nullopt ("forward
// done, no output yet"); sample_tokens builds a ModelRunnerOutput (one canned
// token per scheduled request) from the stashed step, exactly as the real runner
// samples from the hidden states it stashed. Records call counts + order so the
// loop's schedule -> execute -> sample -> update sequence is checkable.
class RunnerStub : public ModelRunnerBase {
 public:
  std::optional<ModelRunnerOutput> execute_model(
      const SchedulerOutput& scheduler_output) override {
    ++execute_calls;
    execute_order = ++global_seq_;
    // Record what the forward saw (proves the Executor forwarded the scheduled
    // output unchanged).
    last_total_scheduled = scheduler_output.total_num_scheduled_tokens;
    last_scheduled_ids.clear();
    for (const auto& [req_id, n] : scheduler_output.num_scheduled_tokens) {
      last_scheduled_ids.push_back(req_id);
    }
    stashed_output_ = scheduler_output;  // stash for sample_tokens.
    return std::nullopt;                 // MRV2: forward done, no output yet.
  }

  ModelRunnerOutput sample_tokens(
      const std::optional<vllm::v1::GrammarOutput>& grammar_output) override {
    ++sample_calls;
    sample_order = ++global_seq_;
    // Record whether the EngineCore threaded a grammar_output to sampling and,
    // if so, its structured-req ids (M3.4 Task 2 seam).
    last_grammar_present = grammar_output.has_value();
    last_grammar_req_ids.clear();
    if (grammar_output.has_value()) {
      last_grammar_req_ids = grammar_output->structured_output_request_ids;
    }
    // Build the ModelRunnerOutput from the stashed step: one canned token per
    // scheduled request (the test keeps prefills single-step, so every
    // scheduled request completes and samples this step).
    ModelRunnerOutput mro;
    int idx = 0;
    for (const auto& [req_id, n] : stashed_output_.num_scheduled_tokens) {
      mro.req_ids.push_back(req_id);
      mro.req_id_to_index[req_id] = idx++;
      mro.sampled_token_ids.push_back(suppress_tokens ? std::vector<int32_t>{}
                                                   : std::vector<int32_t>{kCannedToken});
    }
    return mro;
  }

  int execute_calls = 0;
  int sample_calls = 0;
  int execute_order = 0;
  int sample_order = 0;
  int last_total_scheduled = 0;
  std::vector<std::string> last_scheduled_ids;
  bool last_grammar_present = false;
  bool suppress_tokens = false;
  std::vector<std::string> last_grammar_req_ids;

 private:
  SchedulerOutput stashed_output_;
  int global_seq_ = 0;
};

// Mirror of the scheduler tests' create_scheduler (T0 subset): caching ON
// (M1.3 only supports the prefix-caching coordinator), large budget so short
// prompts prefill in a single step.
std::unique_ptr<Scheduler> CreateScheduler(int max_num_seqs = 16,
                                           int max_num_batched_tokens = 8192,
                                           int num_blocks = 10000,
                                           int block_size = 16,
                                           int max_model_len = 8192) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = max_num_seqs;
  cfg.max_num_batched_tokens = max_num_batched_tokens;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = max_model_len;
  cfg.watermark = 0.0;

  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = num_blocks;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(block_size, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));

  return std::make_unique<Scheduler>(cfg, kv_cfg, block_size,
                                     /*enable_caching=*/true);
}

// A single request with a short prompt (prefills in one step under the budget).
std::unique_ptr<Request> MakeRequest(const std::string& id, int num_tokens = 4) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  const int block_size = 16;
  auto block_hasher = get_request_block_hasher(block_size, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  std::vector<int32_t> prompt(num_tokens, /*value=*/std::stoi(id) + 1);
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   block_hasher);
}

// --- M3.4 Task 2 (e): structured-output seam mocks -------------------------
class MockGrammar : public StructuredOutputGrammar {
 public:
  bool accept_tokens(const std::string&,
                     const std::vector<int32_t>&) override {
    return true;
  }
  std::vector<int32_t> validate_tokens(
      const std::vector<int32_t>& tokens) override {
    return tokens;
  }
  void rollback(int) override {}
  void fill_bitmask(TokenBitmask& bitmask, int batch_index) override {
    const int base = batch_index * bitmask.num_words;
    for (int w = 0; w < bitmask.num_words; ++w) {
      bitmask.data[base + w] = 0x2A;
    }
  }
  bool is_terminated() override { return false; }
  void reset() override {}
};

class MockBackend : public StructuredOutputBackend {
 public:
  std::unique_ptr<StructuredOutputGrammar> compile_grammar(
      StructuredOutputOptions, const std::string&) override {
    return std::make_unique<MockGrammar>();
  }
  TokenBitmask allocate_token_bitmask(int max_num_seqs) override {
    TokenBitmask m;
    m.num_seqs = max_num_seqs;
    m.num_words = 4;  // vocab 100 -> ceil(100/32)
    m.data.assign(static_cast<std::size_t>(max_num_seqs) * m.num_words, 0);
    return m;
  }
  void destroy() override {}
};

std::unique_ptr<Request> MakeStructuredRequest(const std::string& id,
                                               int num_tokens = 4) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  auto block_hasher = get_request_block_hasher(16, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  vllm::StructuredOutputsParams so;
  so.grammar = R"(root ::= "a")";
  params.structured_outputs = so;
  std::vector<int32_t> prompt(num_tokens, /*value=*/std::stoi(id) + 1);
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   block_hasher);
}

std::unique_ptr<Scheduler> CreateSchedulerWithManager(
    StructuredOutputManager* mgr) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = 16;
  cfg.max_num_batched_tokens = 8192;
  cfg.enable_chunked_prefill = true;
  cfg.max_model_len = 8192;
  cfg.watermark = 0.0;
  KVCacheConfig kv_cfg;
  kv_cfg.num_blocks = 10000;
  kv_cfg.kv_cache_groups.emplace_back(
      std::vector<std::string>{"layer"},
      std::make_shared<FullAttentionSpec>(16, /*num_kv_heads=*/1,
                                          /*head_size=*/1, DType::kF32));
  return std::make_unique<Scheduler>(cfg, kv_cfg, /*block_size=*/16,
                                     /*enable_caching=*/true, mgr);
}

}  // namespace

// ---------------------------------------------------------------------------
// (a) Empty scheduler: step() early-returns ({}, false) and never touches the
// runner (core.py:488 `if not self.scheduler.has_requests(): return {}, False`).
// ---------------------------------------------------------------------------
TEST_CASE("EngineCore.step: empty scheduler returns ({}, false), no runner call") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor);

  auto [outputs, model_executed] = engine.step();

  CHECK(outputs.empty());
  CHECK_FALSE(model_executed);
  CHECK(runner.execute_calls == 0);
  CHECK(runner.sample_calls == 0);
}

TEST_CASE("EngineCore.step: prefix lookup statistics survive a tokenless prefill chunk") {
  bool queued = false;
  SUBCASE("synchronous step") {}
  SUBCASE("batch queue at depth one") { queued = true; }
  auto scheduler = CreateScheduler(/*max_num_seqs=*/1, /*max_num_batched_tokens=*/16);
  RunnerStub runner;
  runner.suppress_tokens = true;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor);
  engine.add_request(MakeRequest("0", /*num_tokens=*/80));
  const auto step = [&]() {
    return queued ? engine.step_with_batch_queue() : engine.step();
  };

  const auto [first, executed] = step();
  CHECK(executed);
  REQUIRE(first.size() == 1);
  const auto& observed = first.at(0);
  CHECK(observed.outputs.empty());
  CHECK(observed.scheduler_stats.prefix_cache_stats.requests == 1);
  CHECK(observed.scheduler_stats.prefix_cache_stats.queries == 80);
  CHECK(observed.scheduler_stats.prefix_cache_stats.hits == 0);
  CHECK(scheduler->requests.at("0")->output_token_ids.empty());

  // Subsequent tokenless chunks must not duplicate the admission observation.
  const auto [second, executed_again] = step();
  CHECK(executed_again);
  CHECK(second.empty());
  engine.abort_requests({"0"});
  CHECK(step().first.empty());
}

// ---------------------------------------------------------------------------
// (b) One request: schedule -> execute_model -> sample_tokens -> update_from_
// output. The Executor forwards the scheduled SchedulerOutput unchanged, the
// runner's ModelRunnerOutput is fed to update_from_output (token appended), and
// the EngineCoreOutputs carry the sampled token; model_executed == true.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCore.step: one request drives the full schedule/exec/sample/update loop") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor);

  engine.add_request(MakeRequest("0"));
  Request* req = scheduler->requests.at("0").get();

  auto [outputs, model_executed] = engine.step();

  // Runner driven exactly once through each seam method.
  CHECK(runner.execute_calls == 1);
  CHECK(runner.sample_calls == 1);
  // The Executor forwarded the scheduled step to the forward: the prompt's 4
  // tokens were scheduled for request "0".
  CHECK(runner.last_total_scheduled == 4);
  REQUIRE(runner.last_scheduled_ids.size() == 1);
  CHECK(runner.last_scheduled_ids[0] == "0");

  // Model executed (total_num_scheduled_tokens > 0).
  CHECK(model_executed);

  // update_from_output was fed the runner's ModelRunnerOutput: the canned token
  // was appended to the request...
  REQUIRE(req->output_token_ids.size() == 1);
  CHECK(req->output_token_ids[0] == kCannedToken);
  // ...and surfaced in the per-client EngineCoreOutputs map (single client 0).
  REQUIRE(outputs.size() == 1);
  const EngineCoreOutputs& eco = outputs.at(0);
  REQUIRE(eco.outputs.size() == 1);
  CHECK(eco.outputs[0].request_id == "0");
  REQUIRE(eco.outputs[0].new_token_ids.size() == 1);
  CHECK(eco.outputs[0].new_token_ids[0] == kCannedToken);
  CHECK_FALSE(eco.outputs[0].Finished());  // max_tokens 16, not finished yet.
}

// ---------------------------------------------------------------------------
// (d) Call ORDER: execute_model precedes sample_tokens within step(); and a
// second step continues the request (decode) after the first sampled a token.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCore.step: execute precedes sample; a second step decodes") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor);

  engine.add_request(MakeRequest("0"));
  Request* req = scheduler->requests.at("0").get();

  engine.step();  // prefill + first sample.
  CHECK(runner.execute_order < runner.sample_order);  // execute before sample.
  REQUIRE(req->output_token_ids.size() == 1);

  // Second step: the request is now running; it decodes one token.
  auto [outputs, model_executed] = engine.step();
  CHECK(model_executed);
  CHECK(runner.execute_calls == 2);
  CHECK(runner.sample_calls == 2);
  CHECK(runner.last_total_scheduled == 1);  // one decode token this step.
  REQUIRE(req->output_token_ids.size() == 2);
  CHECK(req->output_token_ids[1] == kCannedToken);
}

// ---------------------------------------------------------------------------
// (c) Abort: abort_requests(FINISHED_ABORTED) finishes the request; it is
// removed from the scheduler and the next step early-returns.
// ---------------------------------------------------------------------------
TEST_CASE("EngineCore.abort_requests: finishes the request; next step is empty") {
  auto scheduler = CreateScheduler();
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor);

  engine.add_request(MakeRequest("0"));
  REQUIRE(scheduler->get_num_unfinished_requests() == 1);

  engine.abort_requests({"0"});

  // The request was finished + removed (finish_requests erases it from requests_
  // and records it in finished_req_ids).
  CHECK(scheduler->get_num_unfinished_requests() == 0);
  CHECK(scheduler->requests.find("0") == scheduler->requests.end());
  // finished_req_ids keeps has_requests() true for exactly one more step (so the
  // runner can clear the finished req).
  CHECK(scheduler->has_finished_requests());

  // Flush step: nothing to schedule (total 0), so model_executed is false and no
  // token outputs are produced (empty map). schedule() flushes finished_req_ids.
  auto [flush_outputs, flush_executed] = engine.step();
  CHECK_FALSE(flush_executed);
  CHECK(flush_outputs.empty());
  CHECK_FALSE(scheduler->has_finished_requests());

  // Engine is now quiesced: the next step early-returns without touching the
  // runner again (only the flush step drove the runner's empty batch).
  auto [outputs, model_executed] = engine.step();
  CHECK(outputs.empty());
  CHECK_FALSE(model_executed);
  CHECK(runner.execute_calls == 1);
  CHECK(runner.sample_calls == 1);
}

// ---------------------------------------------------------------------------
// (e) M3.4 Task 2: EngineCore.step threads the GrammarOutput to sample_tokens.
// A structured request flows grammar_init (add_request) -> get_grammar_bitmask
// (step) -> sample_tokens(grammar_output); the runner records receiving it.
// A plain request threads nullopt (no-op, no regression).
// ---------------------------------------------------------------------------
TEST_CASE("EngineCore.step: a structured request threads grammar_output to sample_tokens") {
  StructuredOutputManager manager(
      /*max_num_seqs=*/16, []() { return std::make_unique<MockBackend>(); });
  auto scheduler = CreateSchedulerWithManager(&manager);
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor, &manager);

  engine.add_request(MakeStructuredRequest("0"));
  // grammar_init ran in add_request: the grammar was compiled + stored.
  REQUIRE(scheduler->requests.at("0")->structured_output_request->grammar !=
          nullptr);

  auto [outputs, model_executed] = engine.step();
  (void)outputs;
  CHECK(model_executed);
  // The EngineCore produced a GrammarOutput and threaded it to sample_tokens.
  CHECK(runner.last_grammar_present);
  REQUIRE(runner.last_grammar_req_ids.size() == 1);
  CHECK(runner.last_grammar_req_ids[0] == "0");
}

TEST_CASE("EngineCore.step: a plain request threads nullopt grammar_output") {
  StructuredOutputManager manager(
      /*max_num_seqs=*/16, []() { return std::make_unique<MockBackend>(); });
  auto scheduler = CreateSchedulerWithManager(&manager);
  RunnerStub runner;
  Executor executor(runner);
  EngineCore engine(*scheduler, executor, &manager);

  engine.add_request(MakeRequest("0"));
  engine.step();

  CHECK_FALSE(runner.last_grammar_present);
  CHECK(runner.last_grammar_req_ids.empty());
}
