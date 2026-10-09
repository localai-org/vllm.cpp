// Tests for the M3.4 Task 1 structured-output type/contract skeleton
// (vllm/v1/structured_output/{backend_types,request}.py @ e24d1b24).
#include <doctest/doctest.h>

#include <map>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "vllm/config/scheduler.h"
#include "vllm/sampling_params.h"
#include "vllm/v1/core/kv_cache_utils.h"
#include "vllm/v1/core/sched/output.h"
#include "vllm/v1/core/sched/scheduler.h"
#include "vllm/v1/engine/types.h"
#include "vllm/v1/kv_cache_interface.h"
#include "vllm/v1/request.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/structured_output/manager.h"
#include "vllm/v1/structured_output/request.h"
#include "vt/dtype.h"

using vllm::SamplingParams;
using vllm::SchedulerConfig;
using vllm::StructuredOutputsParams;
using vllm::v1::BitmaskWordsForVocab;
using vllm::v1::FullAttentionSpec;
using vllm::v1::get_request_block_hasher;
using vllm::v1::get_structured_output_key;
using vllm::v1::GrammarOutput;
using vllm::v1::init_none_hash;
using vllm::v1::KVCacheConfig;
using vllm::v1::ModelRunnerOutput;
using vllm::v1::Request;
using vllm::v1::Scheduler;
using vllm::v1::SchedulerOutput;
using vllm::v1::sha256_cbor;
using vllm::v1::StructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputKey;
using vllm::v1::StructuredOutputManager;
using vllm::v1::StructuredOutputOptions;
using vllm::v1::StructuredOutputRequest;
using vllm::v1::TokenBitmask;
using vt::DType;

namespace {

// A trivial concrete grammar proving the ABC is usable as an interface. It
// accepts everything (recording each accept_tokens call), reports a settable
// terminated flag, and fills each allowed row with a settable sentinel value so
// tests can distinguish a grammar-filled row from the manager's -1 (all-allowed)
// fill.
class MockGrammar : public StructuredOutputGrammar {
 public:
  bool accept_tokens(const std::string& request_id,
                     const std::vector<int32_t>& tokens) override {
    accepted_request_ids.push_back(request_id);
    accepted_tokens.push_back(tokens);
    return accept_result_;
  }
  std::vector<int32_t> validate_tokens(
      const std::vector<int32_t>& tokens) override {
    return tokens;
  }
  void rollback(int num_tokens) override { rolled_back_ += num_tokens; }
  void fill_bitmask(TokenBitmask& bitmask, int batch_index) override {
    const int base = batch_index * bitmask.num_words;
    for (int w = 0; w < bitmask.num_words; ++w) {
      bitmask.data[base + w] = fill_value_;
    }
  }
  bool is_terminated() override { return terminated_; }
  void reset() override { terminated_ = false; }

  int rolled_back_ = 0;
  bool terminated_ = false;
  bool accept_result_ = true;
  int32_t fill_value_ = ~0;  // default: all tokens allowed (existing tests)
  std::vector<std::string> accepted_request_ids;
  std::vector<std::vector<int32_t>> accepted_tokens;
};

// A trivial concrete backend proving the engine-level ABC is usable. Records how
// many grammars it compiled (to verify the manager builds ONE backend lazily).
class MockBackend : public StructuredOutputBackend {
 public:
  explicit MockBackend(int vocab_size) : vocab_size_(vocab_size) {}
  std::unique_ptr<StructuredOutputGrammar> compile_grammar(
      StructuredOutputOptions, const std::string&) override {
    ++compiled;
    return std::make_unique<MockGrammar>();
  }
  TokenBitmask allocate_token_bitmask(int max_num_seqs) override {
    TokenBitmask m;
    m.num_seqs = max_num_seqs;
    m.num_words = BitmaskWordsForVocab(vocab_size_);
    m.data.assign(static_cast<std::size_t>(max_num_seqs) * m.num_words, 0);
    return m;
  }
  void destroy() override { ++destroyed; }

  int compiled = 0;
  int destroyed = 0;

 private:
  int vocab_size_;
};

}  // namespace

namespace {
// A stateful sequence grammar: a rejected draft leaves the accepted prefix
// intact, and validation must leave the real state untouched.
class SequenceGrammar : public StructuredOutputGrammar {
 public:
  bool accept_tokens(const std::string&,
                     const std::vector<int32_t>& tokens) override {
    for (int32_t token : tokens) {
      if (state >= sequence.size() || token != sequence[state]) return false;
      ++state;
    }
    return true;
  }
  std::vector<int32_t> validate_tokens(
      const std::vector<int32_t>& tokens) override {
    std::vector<int32_t> prefix;
    for (int32_t token : tokens) {
      const size_t pos = state + prefix.size();
      if (pos >= sequence.size() || token != sequence[pos]) break;
      prefix.push_back(token);
    }
    return prefix;
  }
  void rollback(int n) override { state -= static_cast<size_t>(n); }
  void fill_bitmask(TokenBitmask& mask, int row) override {
    for (int w = 0; w < mask.num_words; ++w)
      mask.data[row * mask.num_words + w] = 0;
    if (state < sequence.size())
      mask.data[row * mask.num_words] = 1 << sequence[state];
  }
  bool is_terminated() override { return state == sequence.size(); }
  void reset() override { state = 0; }
  const std::vector<int32_t> sequence{7, 8, 9, 10, 11};
  size_t state = 0;
};
}  // namespace

TEST_CASE("get_structured_output_key maps each option to its (type, spec) key") {
  SUBCASE("json") {
    StructuredOutputsParams p;
    p.json = R"({"type":"object"})";
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kJson);
    CHECK(key.second == R"({"type":"object"})");
  }
  SUBCASE("json_object") {
    StructuredOutputsParams p;
    p.json_object = true;
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kJsonObject);
    CHECK(key.second.empty());
  }
  SUBCASE("regex") {
    StructuredOutputsParams p;
    p.regex = "[0-9]+";
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kRegex);
    CHECK(key.second == "[0-9]+");
  }
  SUBCASE("choice serialized as a JSON array") {
    StructuredOutputsParams p;
    p.choice = std::vector<std::string>{"yes", "no"};
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kChoice);
    CHECK(key.second == R"(["yes","no"])");
  }
  SUBCASE("grammar") {
    StructuredOutputsParams p;
    p.grammar = R"(root ::= "a")";
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kGrammar);
    CHECK(key.second == R"(root ::= "a")");
  }
  SUBCASE("structural_tag (deferred backend, key still maps)") {
    StructuredOutputsParams p;
    p.structural_tag = "<tag/>";
    const StructuredOutputKey key = get_structured_output_key(p);
    CHECK(key.first == StructuredOutputOptions::kStructuralTag);
    CHECK(key.second == "<tag/>");
  }
  SUBCASE("no constraint set throws") {
    StructuredOutputsParams p;
    CHECK_THROWS_AS(get_structured_output_key(p), std::runtime_error);
  }
}

TEST_CASE("json_object=false is not-set (upstream `if params.json_object`)") {
  // json_object is optional<bool>; only a truthy value selects JSON_OBJECT.
  StructuredOutputsParams p;
  p.json_object = false;
  // has_value() => counts for mutual exclusion, but the key falls through.
  CHECK_THROWS_AS(get_structured_output_key(p), std::runtime_error);
}

TEST_CASE("StructuredOutputsParams carries the spec through SamplingParams") {
  SamplingParams sp;
  StructuredOutputsParams so;
  so.grammar = R"(root ::= "hi")";
  sp.structured_outputs = so;
  sp.PostInit();  // must not throw with exactly one constraint set

  auto req = StructuredOutputRequest::from_sampling_params(&sp);
  REQUIRE(req.has_value());
  REQUIRE(req->params.grammar.has_value());
  CHECK(*req->params.grammar == R"(root ::= "hi")");

  const StructuredOutputKey key = req->structured_output_key();
  CHECK(key.first == StructuredOutputOptions::kGrammar);
  CHECK(key.second == R"(root ::= "hi")");
}

TEST_CASE("from_sampling_params returns nullopt without a constraint") {
  CHECK_FALSE(StructuredOutputRequest::from_sampling_params(nullptr).has_value());

  SamplingParams sp;  // no structured_outputs
  CHECK_FALSE(StructuredOutputRequest::from_sampling_params(&sp).has_value());
}

TEST_CASE("mutually-exclusive constraint validation") {
  SUBCASE("more than one set throws") {
    StructuredOutputsParams p;
    p.json = "{}";
    p.regex = "x";
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("none set throws") {
    StructuredOutputsParams p;
    CHECK_THROWS_AS(p.Verify(), std::runtime_error);
  }
  SUBCASE("exactly one set is valid") {
    StructuredOutputsParams p;
    p.choice = std::vector<std::string>{"a", "b"};
    CHECK_NOTHROW(p.Verify());
  }
  SUBCASE("PostInit propagates the constraint validation") {
    SamplingParams sp;
    StructuredOutputsParams so;
    so.json = "{}";
    so.grammar = "root ::= \"x\"";  // two constraints
    sp.structured_outputs = so;
    CHECK_THROWS_AS(sp.PostInit(), std::runtime_error);
  }
}

TEST_CASE("all_constraints_none / all_non_structural_tag_constraints_none") {
  StructuredOutputsParams p;
  CHECK(p.all_constraints_none());
  CHECK(p.all_non_structural_tag_constraints_none());

  p.structural_tag = "<t/>";
  CHECK_FALSE(p.all_constraints_none());
  CHECK(p.all_non_structural_tag_constraints_none());  // structural_tag excluded
}

TEST_CASE("BitmaskWordsForVocab is ceil(vocab/32)") {
  CHECK(BitmaskWordsForVocab(32) == 1);
  CHECK(BitmaskWordsForVocab(33) == 2);
  CHECK(BitmaskWordsForVocab(1) == 1);
  CHECK(BitmaskWordsForVocab(64) == 2);
  CHECK(BitmaskWordsForVocab(65) == 3);
}

TEST_CASE("the ABCs are usable as interfaces (mock backend + grammar)") {
  const int kVocab = 100;   // -> ceil(100/32) = 4 words
  const int kMaxSeqs = 3;
  MockBackend backend(kVocab);

  TokenBitmask mask = backend.allocate_token_bitmask(kMaxSeqs);
  CHECK(mask.num_seqs == kMaxSeqs);
  CHECK(mask.num_words == 4);
  CHECK(mask.data.size() == static_cast<std::size_t>(kMaxSeqs) * 4);

  auto grammar = backend.compile_grammar(StructuredOutputOptions::kGrammar, "g");
  REQUIRE(grammar != nullptr);
  CHECK(grammar->accept_tokens("req-0", {1, 2, 3}));
  CHECK(grammar->validate_tokens({7, 8}) == std::vector<int32_t>{7, 8});
  CHECK_FALSE(grammar->is_terminated());

  grammar->fill_bitmask(mask, /*batch_index=*/1);
  // Row 1 all-allowed; rows 0 and 2 untouched (all-forbidden).
  for (int w = 0; w < mask.num_words; ++w) {
    CHECK(mask.data[0 * mask.num_words + w] == 0);
    CHECK(mask.data[1 * mask.num_words + w] == ~0);
    CHECK(mask.data[2 * mask.num_words + w] == 0);
  }

  grammar->rollback(2);
  grammar->reset();
  backend.destroy();
}

// ===========================================================================
// M3.4 Task 2: StructuredOutputManager + scheduler plumbing
// ===========================================================================
namespace {

constexpr int kVocab = 100;    // -> ceil(100/32) = 4 words per row
constexpr int kMaxSeqs = 16;

// A manager wired to a MockBackend factory; keeps a raw pointer to the built
// backend so the test can inspect it.
struct ManagerFixture {
  MockBackend* backend = nullptr;
  StructuredOutputManager manager;
  ManagerFixture()
      : manager(kMaxSeqs, [this]() {
          auto b = std::make_unique<MockBackend>(kVocab);
          backend = b.get();
          return b;
        }) {}
};

// A Request carrying a GBNF grammar constraint (so use_structured_output()).
std::unique_ptr<Request> MakeStructuredRequest(const std::string& id,
                                               int num_tokens = 4) {
  static bool none_hash_initialized = false;
  if (!none_hash_initialized) {
    init_none_hash(sha256_cbor);
    none_hash_initialized = true;
  }
  const int block_size = 16;
  auto block_hasher = get_request_block_hasher(block_size, sha256_cbor);
  SamplingParams params;
  params.max_tokens = 16;
  StructuredOutputsParams so;
  so.grammar = R"(root ::= "a")";
  params.structured_outputs = so;
  std::vector<int32_t> prompt(num_tokens, /*value=*/std::stoi(id) + 1);
  return std::make_unique<Request>(id, prompt, params, /*arrival_time=*/0.0,
                                   block_hasher);
}

// A plain (non-structured) request.
std::unique_ptr<Request> MakePlainRequest(const std::string& id,
                                          int num_tokens = 4) {
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

MockGrammar* GrammarOf(Request& req) {
  return static_cast<MockGrammar*>(req.structured_output_request->grammar.get());
}

std::unique_ptr<Scheduler> CreateScheduler(StructuredOutputManager* mgr) {
  SchedulerConfig cfg;
  cfg.max_num_seqs = kMaxSeqs;
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

// (a) grammar_init: a structured request gets its grammar compiled + stored; the
// backend is built LAZILY exactly once, shared across requests.
TEST_CASE("StructuredOutputManager.grammar_init compiles + stores the grammar") {
  ManagerFixture fx;
  CHECK(fx.backend == nullptr);  // not built until first grammar

  auto req = MakeStructuredRequest("0");
  REQUIRE(req->use_structured_output());
  REQUIRE(req->structured_output_request->grammar == nullptr);

  fx.manager.grammar_init(*req);

  // Backend built lazily; grammar compiled + stored on the request.
  REQUIRE(fx.backend != nullptr);
  CHECK(fx.backend->compiled == 1);
  REQUIRE(req->structured_output_request->grammar != nullptr);

  // A second structured request reuses the SAME backend (single-backend engine).
  auto req2 = MakeStructuredRequest("1");
  fx.manager.grammar_init(*req2);
  CHECK(fx.backend->compiled == 2);
  CHECK(fx.manager.backend() == fx.backend);

  // A non-structured request is a no-op.
  auto plain = MakePlainRequest("2");
  fx.manager.grammar_init(*plain);
  CHECK(fx.backend->compiled == 2);
  CHECK_FALSE(plain->use_structured_output());
}

// (b) grammar_bitmask: one row per structured req id, in order. A live grammar
// fills its row (sentinel); a terminated grammar's row is -1 (all-allowed).
TEST_CASE("StructuredOutputManager.grammar_bitmask fills live rows + -1 for terminated") {
  ManagerFixture fx;

  auto live = MakeStructuredRequest("0");
  auto term = MakeStructuredRequest("1");
  fx.manager.grammar_init(*live);
  fx.manager.grammar_init(*term);
  // Make the live grammar write a distinguishable sentinel; terminate the other.
  GrammarOf(*live)->fill_value_ = 0x2A;
  GrammarOf(*term)->terminated_ = true;

  std::map<std::string, std::unique_ptr<Request>> requests;
  requests["0"] = std::move(live);
  requests["1"] = std::move(term);

  const std::vector<std::string> ids = {"0", "1"};
  auto bitmask = fx.manager.grammar_bitmask(requests, ids, /*spec=*/{});
  REQUIRE(bitmask.has_value());
  CHECK(bitmask->num_seqs == 2);
  CHECK(bitmask->num_words == BitmaskWordsForVocab(kVocab));

  // Row 0 (live) filled by the grammar; row 1 (terminated) all -1 (all-allowed).
  for (int w = 0; w < bitmask->num_words; ++w) {
    CHECK(bitmask->data[0 * bitmask->num_words + w] == 0x2A);
    CHECK(bitmask->data[1 * bitmask->num_words + w] == -1);
  }

  // No structured reqs -> nullopt.
  CHECK_FALSE(fx.manager.grammar_bitmask(requests, {}, {}).has_value());
}

// (c) Scheduler::get_grammar_bitmask: nullopt for a non-structured batch; a
// GrammarOutput (ordered rows) for a structured one.
TEST_CASE("Scheduler.get_grammar_bitmask: nullopt without structured, else GrammarOutput") {
  SUBCASE("no structured requests -> nullopt") {
    ManagerFixture fx;
    auto scheduler = CreateScheduler(&fx.manager);
    scheduler->add_request(MakePlainRequest("0"));
    SchedulerOutput out = scheduler->schedule();
    CHECK_FALSE(out.has_structured_output_requests);
    CHECK_FALSE(scheduler->get_grammar_bitmask(out).has_value());
  }
  SUBCASE("a structured request -> GrammarOutput with its row") {
    ManagerFixture fx;
    auto scheduler = CreateScheduler(&fx.manager);
    auto req = MakeStructuredRequest("0");
    fx.manager.grammar_init(*req);  // compile before scheduling (EngineCore does)
    GrammarOf(*req)->fill_value_ = 0x2A;
    scheduler->add_request(std::move(req));

    SchedulerOutput out = scheduler->schedule();
    // Prefill completes in one step (4 == NumTokens) so it is not a prefill chunk.
    CHECK(out.has_structured_output_requests);

    auto grammar_output = scheduler->get_grammar_bitmask(out);
    REQUIRE(grammar_output.has_value());
    REQUIRE(grammar_output->structured_output_request_ids.size() == 1);
    CHECK(grammar_output->structured_output_request_ids[0] == "0");
    REQUIRE(grammar_output->grammar_bitmask.num_seqs == 1);
    for (int w = 0; w < grammar_output->grammar_bitmask.num_words; ++w) {
      CHECK(grammar_output->grammar_bitmask.data[w] == 0x2A);
    }
  }
}

// (d) update_from_output advances the FSM: accept_tokens is called with the
// sampled tokens on the structured request's grammar.
TEST_CASE("Scheduler.update_from_output calls grammar.accept_tokens on sampled tokens") {
  ManagerFixture fx;
  auto scheduler = CreateScheduler(&fx.manager);
  auto req = MakeStructuredRequest("0");
  fx.manager.grammar_init(*req);
  MockGrammar* grammar = GrammarOf(*req);
  scheduler->add_request(std::move(req));

  SchedulerOutput out = scheduler->schedule();

  // Feed a sampled token back through update_from_output.
  ModelRunnerOutput mro;
  mro.req_ids.push_back("0");
  mro.req_id_to_index["0"] = 0;
  mro.sampled_token_ids.push_back({7});
  scheduler->update_from_output(out, mro);

  // The grammar's FSM was advanced with the sampled token(s).
  REQUIRE(grammar->accepted_tokens.size() == 1);
  CHECK(grammar->accepted_request_ids[0] == "0");
  REQUIRE(grammar->accepted_tokens[0].size() == 1);
  CHECK(grammar->accepted_tokens[0][0] == 7);
}

TEST_CASE("Structured MTP: scheduler validates the draft prefix without advancing") {
  ManagerFixture fx;
  auto scheduler = CreateScheduler(&fx.manager);
  auto req = MakeStructuredRequest("0");
  fx.manager.grammar_init(*req);
  auto grammar = std::make_unique<SequenceGrammar>();
  auto* state = grammar.get();
  auto* request = req.get();
  req->structured_output_request->grammar = std::move(grammar);
  auto plain = MakePlainRequest("1");
  auto* plain_request = plain.get();
  scheduler->add_request(std::move(req));
  scheduler->add_request(std::move(plain));
  vllm::v1::DraftTokenIds drafts;
  drafts.req_ids = {"0", "1"};
  drafts.draft_token_ids = {{7, 8, 14}, {7, 8, 14}};
  scheduler->update_draft_token_ids(drafts);
  CHECK(request->spec_token_ids == std::vector<int32_t>{7, 8});
  CHECK(plain_request->spec_token_ids == std::vector<int32_t>{7, 8, 14});
  CHECK(state->state == 0);

  SchedulerOutput out;
  out.scheduled_spec_decode_tokens = {{"0", {-1, -1, -1}},
                                    {"1", {-1, -1, -1}}};
  scheduler->update_draft_token_ids_in_output(drafts, out);
  CHECK(out.scheduled_spec_decode_tokens.at("0") ==
        std::vector<int32_t>{7, 8, -1});
  CHECK(out.scheduled_spec_decode_tokens.at("1") ==
        std::vector<int32_t>{7, 8, 14});
  CHECK(out.num_invalid_spec_tokens.at("0") == 1);
  CHECK(out.num_invalid_spec_tokens.count("1") == 0);
  CHECK(state->state == 0);
}

TEST_CASE("Structured MTP: expanded masks grow and restore each grammar state") {
  StructuredOutputManager manager(1, [] {
    return std::make_unique<MockBackend>(kVocab);
  });
  std::map<std::string, std::unique_ptr<Request>> requests;
  std::vector<SequenceGrammar*> grammars;
  for (const auto& id : {"0", "1"}) {
    auto req = MakeStructuredRequest(id);
    manager.grammar_init(*req);
    auto grammar = std::make_unique<SequenceGrammar>();
    grammars.push_back(grammar.get());
    req->structured_output_request->grammar = std::move(grammar);
    requests[id] = std::move(req);
  }
  // First allocate a single row, then grow beyond both that capacity and C2.
  REQUIRE(manager.grammar_bitmask(requests, {"0"}, {})->num_seqs == 1);
  const auto mask = manager.grammar_bitmask(
      requests, {"1", "0"}, {{"1", {7, 8, 9}}, {"0", {7, 8}}});
  REQUIRE(mask.has_value());
  REQUIRE(mask->num_seqs == 7);
  const std::vector<int> allowed{7, 8, 9, 10, 7, 8, 9};
  for (size_t row = 0; row < allowed.size(); ++row) {
    CHECK(mask->data[row * mask->num_words] == (1 << allowed[row]));
    for (int w = 1; w < mask->num_words; ++w)
      CHECK(mask->data[row * mask->num_words + w] == 0);
  }
  for (auto* grammar : grammars) CHECK(grammar->state == 0);
  const auto padded = manager.grammar_bitmask(
      requests, {"0"}, {{"0", {7, -1, -1}}});
  REQUIRE(padded->num_seqs == 4);
  CHECK(padded->data[0] == (1 << 7));
  CHECK(padded->data[padded->num_words] == (1 << 8));
  for (int row = 2; row < 4; ++row)
    for (int w = 0; w < padded->num_words; ++w)
      CHECK(padded->data[row * padded->num_words + w] == -1);
  CHECK(grammars[0]->state == 0);
  CHECK_THROWS_AS(manager.grammar_bitmask(requests, {"0"},
                                        {{"0", {7, 14}}}), std::runtime_error);
  CHECK(grammars[0]->state == 0);
}
