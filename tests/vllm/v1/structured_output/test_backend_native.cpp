// Tests for the M3.4 Task 4 NATIVE grammar engine (backend_native.{h,cpp}),
// an ORIGINAL vllm.cpp component (§9) behind the 1:1-ported StructuredOutput
// seam. Exercises: GBNF (yes/no), CHOICE (cat/dog/bird), char-class ([0-9]+),
// byte-alignment across a token boundary + a multi-byte UTF-8 codepoint, the
// EOS/stop token only at an accepting state, the sub-O(vocab) fill (a trie-node
// visit counter), rollback/reset, validate_tokens, and regex lowering.
//
// The tokenizer is a REAL byte-level BPE fixture (built via Tokenizer::FromHfJson
// like tests/vllm/test_detokenizer.cpp) so token->raw-bytes decoding is real.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"
#include "vllm/v1/structured_output/backend_native.h"
#include "vllm/v1/structured_output/backend_types.h"
#include "vllm/v1/engine/input_processor.h"

using nlohmann::json;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::Tokenizer;
using vllm::v1::BitmaskWordsForVocab;
using vllm::v1::NativeGrammar;
using vllm::v1::NativeStructuredOutputBackend;
using vllm::v1::StructuredOutputGrammar;
using vllm::v1::StructuredOutputOptions;
using vllm::v1::TokenBitmask;

namespace {

using Ids = std::vector<int32_t>;

class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_native_grammar_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// The EOS/stop token id used across tests (an added, special token).
constexpr int32_t kEos = 40;
// Number of junk tokens inflating the vocab (for the perf property).
constexpr int kJunk = 300;

// A real byte-level BPE tokenizer whose vocab is designed for the grammar
// tests. All hand-picked tokens are ASCII (so the byte-map is identity) except
// the é fragments; junk tokens all start with 'q' so a "yes"/"[0-9]"-anchored
// grammar never descends into them (the perf property).
Tokenizer BuildFixture() {
  json vocab = {
      {"y", 0},    {"e", 1},   {"s", 2},    {"n", 3},   {"o", 4},
      {"ye", 5},   {"es", 6},  {"yes", 7},  {"no", 8},  {"c", 9},
      {"a", 10},   {"t", 11},  {"d", 12},   {"g", 13},  {"b", 14},
      {"i", 15},   {"r", 16},  {"cat", 17}, {"dog", 18}, {"bird", 19},
      {"x", 20},   {"z", 21},  {"0", 22},   {"1", 23},   {"5", 24},
      {"9", 25},   {"12", 26}, {"ab", 27},  {"!", 31}};
  // é = C3 A9; its two byte-fragments; used for multi-byte grammar matching.
  vocab[MapBytesToUnicode("\xC3\xA9")] = 28;  // full é
  vocab[MapBytesToUnicode("\xC3")] = 29;      // lead byte
  vocab[MapBytesToUnicode("\xA9")] = 30;      // trailing byte
  // Junk tokens q0..q(kJunk-1): never a prefix of the tested grammars.
  for (int i = 0; i < kJunk; ++i) {
    vocab["q" + std::to_string(i)] = 100 + i;
  }

  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", kEos}, {"content", "<eos>"}, {"special", true}},
       {{"id", 41}, {"content", "<tool>"}, {"special", false}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array()}};
  TempJson f(doc.dump());
  return Tokenizer::FromHfJson(f.path());
}

const Tokenizer& Fixture() {
  static const Tokenizer tok = BuildFixture();
  return tok;
}

int VocabSize() { return static_cast<int>(Fixture().VocabSize()); }

std::unique_ptr<NativeStructuredOutputBackend> MakeBackend() {
  return std::make_unique<NativeStructuredOutputBackend>(
      Fixture(), VocabSize(), std::vector<int32_t>{kEos});
}

// Compiles a grammar and returns it as the concrete type (so tests can read the
// perf counter).
std::unique_ptr<NativeGrammar> Compile(NativeStructuredOutputBackend& backend,
                                       StructuredOutputOptions type,
                                       const std::string& spec) {
  std::unique_ptr<StructuredOutputGrammar> g =
      backend.compile_grammar(type, spec);
  return std::unique_ptr<NativeGrammar>(
      static_cast<NativeGrammar*>(g.release()));
}

// Whether token id `t` is allowed by a freshly filled single-row bitmask.
bool Allowed(NativeGrammar& g, int32_t t) {
  TokenBitmask bm;
  bm.num_seqs = 1;
  bm.num_words = BitmaskWordsForVocab(VocabSize());
  bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
  g.fill_bitmask(bm, 0);
  const int32_t word = bm.data[static_cast<std::size_t>(t >> 5)];
  return (word & (1 << (t & 31))) != 0;
}

}  // namespace

// (a) GBNF: root ::= "yes" | "no".
TEST_CASE("native GBNF: yes/no allows only valid prefixes, terminates on match") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes" | "no")");

  // At the start only tokens whose bytes are a valid prefix of "yes"/"no".
  CHECK(Allowed(*g, 0));   // "y"
  CHECK(Allowed(*g, 5));   // "ye"
  CHECK(Allowed(*g, 7));   // "yes"
  CHECK(Allowed(*g, 3));   // "n"
  CHECK(Allowed(*g, 8));   // "no"
  CHECK_FALSE(Allowed(*g, 2));   // "s" — not a valid first byte
  CHECK_FALSE(Allowed(*g, 20));  // "x" — forbidden
  CHECK_FALSE(Allowed(*g, 6));   // "es"
  CHECK_FALSE(Allowed(*g, kEos));  // EOS not allowed before a complete match

  CHECK_FALSE(g->is_terminated());
  REQUIRE(g->accept_tokens("r", {7}));  // "yes"
  CHECK(g->is_terminated());
  // After a full match EOS is the only continuation (fill even when terminated).
  CHECK(Allowed(*g, kEos));
}

TEST_CASE("native grammar: resolved model primary and chat EOS remain valid stops") {
  vllm::HfConfig config;
  config.max_position_embeddings = 32;
  config.raw = {{"eos_token_id", kEos}};
  config.generation_config_eos_ids = {kEos, 41};
  vllm::v1::InputProcessor input(Fixture(), config);
  REQUIRE(input.eos_token_ids() == std::vector<int32_t>{kEos, 41});
  auto factory = vllm::v1::MakeNativeBackendFactory(
      Fixture(), VocabSize(), input.eos_token_ids());
  auto backend = factory();
  for (int32_t eos : input.eos_token_ids()) {
    auto grammar = backend->compile_grammar(StructuredOutputOptions::kGrammar,
                                           R"(root ::= "yes")");
    auto& native = static_cast<NativeGrammar&>(*grammar);
    CHECK_FALSE(Allowed(native, eos));
    CHECK_FALSE(native.accept_tokens("r", {eos}));
    REQUIRE(native.accept_tokens("r", {7}));
    CHECK(Allowed(native, eos));
    CHECK(native.accept_tokens("r", {eos}));
    CHECK(native.is_terminated());
    CHECK_FALSE(native.accept_tokens("r", {7}));
  }
}

// (d) byte-alignment across a token boundary: "ye" then "s".
TEST_CASE("native GBNF: FSM advances correctly across a token boundary") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes" | "no")");
  REQUIRE(g->accept_tokens("r", {5}));  // "ye"
  CHECK_FALSE(g->is_terminated());
  // Now only "s" continues; "yes" (starts with 'y') no longer valid.
  CHECK(Allowed(*g, 2));         // "s"
  CHECK_FALSE(Allowed(*g, 7));   // "yes"
  CHECK_FALSE(Allowed(*g, 0));   // "y"
  REQUIRE(g->accept_tokens("r", {2}));  // "s"
  CHECK(g->is_terminated());
}

// A token forbidden by the grammar is rejected by accept_tokens.
TEST_CASE("native GBNF: accept_tokens rejects an invalid token") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes")");
  CHECK_FALSE(g->accept_tokens("r", {8}));  // "no"
}

// (b) CHOICE: ["cat","dog","bird"].
TEST_CASE("native CHOICE: only the listed completions") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kChoice,
                   R"(["cat","dog","bird"])");
  CHECK(Allowed(*g, 9));    // "c"
  CHECK(Allowed(*g, 17));   // "cat"
  CHECK(Allowed(*g, 12));   // "d"
  CHECK(Allowed(*g, 18));   // "dog"
  CHECK(Allowed(*g, 14));   // "b"
  CHECK(Allowed(*g, 19));   // "bird"
  CHECK_FALSE(Allowed(*g, 0));   // "y"
  CHECK_FALSE(Allowed(*g, 11));  // "t" — not a valid first byte

  REQUIRE(g->accept_tokens("r", {18}));  // "dog"
  CHECK(g->is_terminated());
}

// (c) char class: root ::= [0-9]+.
TEST_CASE("native char class: [0-9]+ allows digits, forbids letters") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= [0-9]+)");
  // Digit tokens (incl. the multi-digit "12") allowed; letters forbidden.
  CHECK(Allowed(*g, 22));   // "0"
  CHECK(Allowed(*g, 24));   // "5"
  CHECK(Allowed(*g, 26));   // "12"
  CHECK_FALSE(Allowed(*g, 20));  // "x"
  CHECK_FALSE(Allowed(*g, 27));  // "ab"
  CHECK_FALSE(Allowed(*g, kEos));  // + requires >= 1 digit before EOS

  REQUIRE(g->accept_tokens("r", {24}));  // "5"
  CHECK_FALSE(g->is_terminated());       // "+" can still repeat
  CHECK(Allowed(*g, 22));                // another digit
  CHECK(Allowed(*g, kEos));              // now EOS is allowed (accepting)
}

// Multi-byte UTF-8 codepoint literal in the grammar, matched via byte-level
// tokens (both the whole-codepoint token and its byte fragments).
TEST_CASE("native GBNF: multi-byte UTF-8 codepoint literal") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "é" "!")");  // é then !
  CHECK(Allowed(*g, 28));        // full "é" (C3 A9)
  CHECK(Allowed(*g, 29));        // just the C3 lead byte (valid prefix)
  CHECK_FALSE(Allowed(*g, 30));  // A9 alone is not valid first
  CHECK_FALSE(Allowed(*g, 31));  // "!" not valid before é

  SUBCASE("whole codepoint in one token") {
    REQUIRE(g->accept_tokens("r", {28}));  // é
    CHECK(Allowed(*g, 31));                // now "!"
    REQUIRE(g->accept_tokens("r", {31}));
    CHECK(g->is_terminated());
  }
  SUBCASE("codepoint split across two byte tokens") {
    REQUIRE(g->accept_tokens("r", {29}));  // C3
    CHECK(Allowed(*g, 30));                // A9 completes é
    CHECK_FALSE(Allowed(*g, 31));          // "!" not yet
    REQUIRE(g->accept_tokens("r", {30}));  // A9
    CHECK(Allowed(*g, 31));
    REQUIRE(g->accept_tokens("r", {31}));
    CHECK(g->is_terminated());
  }
}

// (e) EOS/stop token allowed only at an accepting state; consuming it
// transitions to terminated and nothing may follow.
TEST_CASE("native: EOS allowed only at an accepting state") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "ab")");  // tokens: "a"=10, "b"=14
  CHECK_FALSE(Allowed(*g, kEos));              // start: not accepting
  CHECK_FALSE(g->accept_tokens("r", {kEos}));  // EOS rejected mid-grammar
  REQUIRE(g->accept_tokens("r", {10}));        // "a"
  CHECK_FALSE(Allowed(*g, kEos));              // still not complete
  REQUIRE(g->accept_tokens("r", {14}));        // "b" -> complete
  CHECK(g->is_terminated());
  CHECK(Allowed(*g, kEos));
  REQUIRE(g->accept_tokens("r", {kEos}));      // consume EOS
  CHECK(g->is_terminated());
  CHECK_FALSE(g->accept_tokens("r", {14}));    // nothing follows EOS
}

// ADDED tokens are matchable by their LITERAL content bytes (a structural marker
// like `<tool>` is a grammar literal, mirroring the real Qwen3.6 `<tool_call>`);
// only STOP/EOS tokens stay non-matchable mid-derivation. (This test previously
// asserted the OLD exclude-all-added behavior, which broke lazy tool-calling —
// see the "added-marker" section below for the regression it caused.)
TEST_CASE("native: added tokens match by content; stop tokens stay excluded") {
  auto backend = MakeBackend();
  // A permissive grammar that accepts the literal bytes of "<tool>".
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= [<a-z>]+)");
  CHECK(Allowed(*g, 41));       // "<tool>" (non-special added) IS matchable now
  CHECK_FALSE(Allowed(*g, kEos));  // the stop/EOS token is NOT matchable mid-grammar
}

// (f) perf: fill_bitmask visits far fewer trie nodes than the vocab size.
TEST_CASE("native: fill_bitmask is sub-O(vocab) (trie-node visit counter)") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes")");
  TokenBitmask bm;
  bm.num_seqs = 1;
  bm.num_words = BitmaskWordsForVocab(VocabSize());
  bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
  g->fill_bitmask(bm, 0);
  // Only the trie nodes along "y","ye","yes" (+ root) are visited; nothing near
  // the ~330-token vocab (hundreds of junk 'q...' tokens are pruned).
  CHECK(g->last_fill_visited_nodes() < 10);
  CHECK(g->last_fill_visited_nodes() < VocabSize() / 4);
}

// (g) rollback / reset.
TEST_CASE("native: rollback and reset restore prior FSM state") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes")");
  REQUIRE(g->accept_tokens("r", {5}));  // "ye"
  CHECK(Allowed(*g, 2));                // "s" continues
  CHECK_FALSE(Allowed(*g, 0));          // "y" no longer

  g->rollback(1);  // back to start
  CHECK(Allowed(*g, 0));   // "y" allowed again
  CHECK(Allowed(*g, 7));   // "yes" allowed again
  CHECK_FALSE(Allowed(*g, 2));  // "s" not valid at start

  REQUIRE(g->accept_tokens("r", {7}));  // "yes"
  CHECK(g->is_terminated());
  g->reset();
  CHECK_FALSE(g->is_terminated());
  CHECK(Allowed(*g, 0));  // start state again
}

// (h) validate_tokens returns the accepted prefix without advancing.
TEST_CASE("native: validate_tokens returns the accepted prefix") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kGrammar,
                   R"(root ::= "yes")");
  // "ye","s" accepted; then "x" rejected -> prefix stops there.
  const Ids result = g->validate_tokens({5, 2, 20, 0});
  CHECK(result == Ids{5, 2});
  // validate_tokens must NOT have advanced the real FSM.
  CHECK(Allowed(*g, 0));  // still at the start
}

// REGEX lowering: a common pattern.
TEST_CASE("native REGEX: [0-9]+ via regex lowering") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kRegex, R"([0-9]+)");
  CHECK(Allowed(*g, 22));   // "0"
  CHECK(Allowed(*g, 26));   // "12"
  CHECK_FALSE(Allowed(*g, 20));  // "x"
  REQUIRE(g->accept_tokens("r", {26}));  // "12"
  CHECK(Allowed(*g, kEos));
}

TEST_CASE("native REGEX: \\d escape + quantifier") {
  auto backend = MakeBackend();
  auto g = Compile(*backend, StructuredOutputOptions::kRegex, R"(\d{1,3})");
  CHECK(Allowed(*g, 24));   // "5"
  CHECK_FALSE(Allowed(*g, 20));
}

// The factory helper the StructuredOutputManager consumes builds a working
// native backend from (tokenizer, vocab_size).
TEST_CASE("native: MakeNativeBackendFactory builds a working backend") {
  auto factory = vllm::v1::MakeNativeBackendFactory(
      Fixture(), VocabSize(), std::vector<int32_t>{kEos});
  std::unique_ptr<vllm::v1::StructuredOutputBackend> backend = factory();
  REQUIRE(backend != nullptr);

  TokenBitmask mask = backend->allocate_token_bitmask(4);
  CHECK(mask.num_seqs == 4);
  CHECK(mask.num_words == BitmaskWordsForVocab(VocabSize()));

  auto g = backend->compile_grammar(StructuredOutputOptions::kChoice,
                                    R"(["yes","no"])");
  REQUIRE(g != nullptr);
  CHECK(g->accept_tokens("r", {7}));  // "yes"
  CHECK(g->is_terminated());
  backend->destroy();
}

// JSON / json_object now compile (M3.4 Task 5: JSON-schema -> GBNF). The
// schema-acceptance behavior is covered in test_json_schema_to_gbnf.cpp (this
// fixture's vocab has no JSON-structural single-char tokens); here we only
// assert the compile path no longer throws and yields a grammar. A malformed
// schema still throws.
TEST_CASE("native: JSON/json_object compile (M3.4 Task 5)") {
  auto backend = MakeBackend();
  CHECK_NOTHROW(backend->compile_grammar(StructuredOutputOptions::kJson,
                                         R"({"type":"object"})"));
  CHECK_NOTHROW(
      backend->compile_grammar(StructuredOutputOptions::kJsonObject, ""));
  CHECK_THROWS(backend->compile_grammar(StructuredOutputOptions::kJson,
                                        "{not valid json"));
}

// ─── THE INVARIANT: fill_bitmask(t) == accept_tokens([t]) over the WHOLE vocab ──
// This is the load-bearing correctness guard for constrained decoding: the set of
// tokens fill_bitmask marks ALLOWED must be EXACTLY the set accept_tokens would
// accept from the same state. If fill ever allowed a token accept rejects, the
// sampler could pick a grammar-INVALID token. Differentially check every vocab id
// across several grammars and prefix states (recompiling a fresh grammar per token
// so accept_tokens' state mutation doesn't leak). Recommended by the Task-4 review.
TEST_CASE("native: fill_bitmask agrees with accept_tokens for every vocab token") {
  auto backend = MakeBackend();
  const int vocab = VocabSize();

  struct Case {
    StructuredOutputOptions type;
    std::string spec;
    std::vector<int32_t> prefix;  // tokens to accept before checking
  };
  const std::vector<Case> cases = {
      {StructuredOutputOptions::kGrammar, "root ::= \"yes\" | \"no\"", {}},
      {StructuredOutputOptions::kGrammar, "root ::= \"yes\" | \"no\"", {5}},  // "ye"
      {StructuredOutputOptions::kGrammar, "root ::= [0-9]+", {}},
      {StructuredOutputOptions::kGrammar, "root ::= [0-9] [0-9]", {}},
      {StructuredOutputOptions::kGrammar, "root ::= [0-9] [0-9]", {22}},  // "0" (1 digit)
      {StructuredOutputOptions::kGrammar, "root ::= \"yes\"", {}},
      {StructuredOutputOptions::kChoice, R"(["yes","no","bird"])", {}},
      {StructuredOutputOptions::kRegex, "[0-9]+", {}},
  };

  for (const Case& c : cases) {
    // Fill once on a grammar advanced through the prefix.
    auto filled = Compile(*backend, c.type, c.spec);
    REQUIRE(filled != nullptr);
    if (!c.prefix.empty()) REQUIRE(filled->accept_tokens("p", c.prefix));

    TokenBitmask bm;
    bm.num_seqs = 1;
    bm.num_words = BitmaskWordsForVocab(vocab);
    bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
    filled->fill_bitmask(bm, 0);

    for (int32_t t = 0; t < vocab; ++t) {
      const bool fill_allows =
          (bm.data[static_cast<std::size_t>(t >> 5)] &
           static_cast<int32_t>(1u << (static_cast<uint32_t>(t) & 31u))) != 0;
      // Fresh grammar at the same prefix state, then probe accept([t]).
      auto probe = Compile(*backend, c.type, c.spec);
      if (!c.prefix.empty()) REQUIRE(probe->accept_tokens("p", c.prefix));
      const bool accept_allows = probe->accept_tokens("t", {t});
      INFO("grammar='" << c.spec << "' prefix_len=" << c.prefix.size()
                       << " token=" << t << " fill=" << fill_allows
                       << " accept=" << accept_allows);
      CHECK(fill_allows == accept_allows);
    }
  }
  backend->destroy();
}

// ─────────────────────────── M3.3b: LAZY grammar triggers ───────────────────
// A LAZY grammar (llama.cpp awaiting_trigger mechanism, src/llama-grammar.cpp
// :1339-1427) is INERT (every token allowed) until a trigger WORD (a literal
// substring of the decoded output, e.g. `<tool_call>`) or a trigger TOKEN id
// appears; then it activates + REPLAYS the buffered bytes so the following JSON
// is grammar-constrained. This is exactly vLLM's xgrammar StructuralTag /
// TriggeredTagsFormat, implemented natively. Below: a lazy tool-call grammar
// `<tool_call>{<digit>}</tool_call>` driven token-by-token over a real BPE
// fixture whose vocab spells the wrapper + a tiny JSON body.
namespace {

// EOS for the lazy fixture.
constexpr int32_t kLazyEos = 40;

// A real byte-level BPE fixture. All tokens are printable-ASCII (identity
// byte-map). Single-char tokens spell `<tool_call>{<digit>}</tool_call>`; the
// multi-char tokens 16-19 let a trigger fire from ONE token and MID-token (the
// replay path): 17=`<tool_call>{` (trigger + first JSON byte), 19=`P<tool_call>{`
// (free-text `P` straddling the trigger start).
Tokenizer BuildLazyFixture() {
  json vocab = {
      {"<", 0},  {">", 1},  {"/", 2},  {"t", 3},  {"o", 4},  {"l", 5},
      {"_", 6},  {"c", 7},  {"a", 8},  {"{", 9},  {"}", 10}, {"5", 11},
      {"0", 12}, {"9", 13}, {"Z", 14}, {"P", 15}, {"<tool_call>", 16},
      {"<tool_call>{", 17}, {"</tool_call>", 18}, {"P<tool_call>{", 19}};

  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", kLazyEos}, {"content", "<eos>"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array()}};
  TempJson f(doc.dump());
  return Tokenizer::FromHfJson(f.path());
}

const Tokenizer& LazyFixture() {
  static const Tokenizer tok = BuildLazyFixture();
  return tok;
}

int LazyVocabSize() { return static_cast<int>(LazyFixture().VocabSize()); }

std::unique_ptr<NativeStructuredOutputBackend> MakeLazyBackend() {
  return std::make_unique<NativeStructuredOutputBackend>(
      LazyFixture(), LazyVocabSize(), std::vector<int32_t>{kLazyEos});
}

std::unique_ptr<NativeGrammar> CompileLazy(
    NativeStructuredOutputBackend& backend, const std::string& gbnf,
    std::vector<std::string> words, std::vector<int32_t> tokens = {}) {
  std::unique_ptr<StructuredOutputGrammar> g =
      backend.compile_lazy_grammar(gbnf, std::move(words), std::move(tokens));
  return std::unique_ptr<NativeGrammar>(
      static_cast<NativeGrammar*>(g.release()));
}

bool LazyAllowed(NativeGrammar& g, int32_t t) {
  TokenBitmask bm;
  bm.num_seqs = 1;
  bm.num_words = BitmaskWordsForVocab(LazyVocabSize());
  bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
  g.fill_bitmask(bm, 0);
  const int32_t word = bm.data[static_cast<std::size_t>(t >> 5)];
  return (word & (1 << (t & 31))) != 0;
}

// The wrapped tool-call grammar: `<tool_call>` + a tiny JSON body `{<digit>}` +
// `</tool_call>`. After the trigger, only `{`, then a digit, `}`, the closing
// tag are valid — a schema-constrained body.
const char* const kToolGrammar =
    "root ::= \"<tool_call>\" \"{\" [0-9] \"}\" \"</tool_call>\"";

}  // namespace

// (a) Before the trigger the grammar is INERT: every token is allowed (free
// text) and the grammar may terminate (a plain reply can end).
TEST_CASE("native lazy: before the trigger every token is allowed (inert)") {
  auto backend = MakeLazyBackend();
  auto g = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});

  // fill_bitmask allows ALL tokens (a letter, `{`, `<`, and even EOS).
  CHECK(LazyAllowed(*g, 14));       // "Z" (a letter)
  CHECK(LazyAllowed(*g, 9));        // "{"
  CHECK(LazyAllowed(*g, 0));        // "<"
  CHECK(LazyAllowed(*g, 3));        // "t"
  CHECK(LazyAllowed(*g, kLazyEos)); // EOS (a plain reply may end)
  CHECK(g->is_terminated());        // awaiting == accepting (plain reply can end)

  // accept_tokens succeeds on arbitrary free text.
  CHECK(g->accept_tokens("r", {14}));  // "Z"
  CHECK(g->accept_tokens("r", {15}));  // "P"
  CHECK(g->accept_tokens("r", {9}));   // "{" as free text (no tool call yet)
  // Still awaiting (no trigger) -> still all allowed + terminable.
  CHECK(LazyAllowed(*g, 14));
  CHECK(LazyAllowed(*g, kLazyEos));
  CHECK(g->is_terminated());
}

// (b) Feed free text then the trigger `<tool_call>` -> awaiting flips; AFTER the
// trigger the grammar CONSTRAINS the JSON body; terminates only when complete.
TEST_CASE("native lazy: after the trigger the grammar constrains the JSON") {
  auto backend = MakeLazyBackend();
  auto g = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});

  REQUIRE(g->accept_tokens("r", {14}));  // "Z" (free text)
  CHECK(g->is_terminated());             // still just a plain reply

  REQUIRE(g->accept_tokens("r", {16}));  // "<tool_call>" (the trigger)
  CHECK_FALSE(g->is_terminated());       // triggered: mid tool-call
  CHECK(LazyAllowed(*g, 9));             // only "{" opens the body
  CHECK_FALSE(LazyAllowed(*g, 14));      // a letter is forbidden now
  CHECK_FALSE(LazyAllowed(*g, 0));       // "<" forbidden
  CHECK_FALSE(LazyAllowed(*g, kLazyEos)); // not accepting mid tool-call

  REQUIRE(g->accept_tokens("r", {9}));   // "{"
  CHECK(LazyAllowed(*g, 11));            // a digit "5"
  CHECK_FALSE(LazyAllowed(*g, 9));       // another "{" forbidden
  REQUIRE(g->accept_tokens("r", {11}));  // "5"
  REQUIRE(g->accept_tokens("r", {10}));  // "}"
  CHECK_FALSE(g->is_terminated());       // still needs "</tool_call>"
  REQUIRE(g->accept_tokens("r", {18}));  // "</tool_call>"
  CHECK(g->is_terminated());             // full <tool_call>{5}</tool_call>
}

// (c) REPLAY: the trigger fires MID-token; the byte AFTER the trigger in the same
// token must advance the FSM (the replay covers [start, end) of the buffer).
TEST_CASE("native lazy: trigger fires mid-token and the next byte advances") {
  auto backend = MakeLazyBackend();

  SUBCASE("whole token = trigger + first JSON byte") {
    auto g = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});
    REQUIRE(g->accept_tokens("r", {17}));  // "<tool_call>{" in ONE token
    CHECK_FALSE(g->is_terminated());
    CHECK(LazyAllowed(*g, 11));            // next required byte is a DIGIT (post {)
    CHECK_FALSE(LazyAllowed(*g, 9));       // NOT another "{" (already consumed)
  }
  SUBCASE("token straddles the trigger start (leading free text discarded)") {
    auto g = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});
    REQUIRE(g->accept_tokens("r", {19}));  // "P<tool_call>{": P is free text
    CHECK(LazyAllowed(*g, 11));            // digit next (replay started at "<")
    CHECK_FALSE(LazyAllowed(*g, 9));       // "{" already consumed
  }
}

// (d) A plain reply that never emits the trigger stays all-allowed throughout and
// can terminate — no tool call is ever forced.
TEST_CASE("native lazy: a reply that never triggers stays free and can end") {
  auto backend = MakeLazyBackend();
  auto g = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});
  REQUIRE(g->accept_tokens("r", {15, 14, 15}));  // "P","Z","P"
  CHECK(LazyAllowed(*g, 14));       // still all allowed
  CHECK(LazyAllowed(*g, 9));
  CHECK(g->is_terminated());        // may end (no tool call forced)
  REQUIRE(g->accept_tokens("r", {kLazyEos}));  // end the plain reply
  CHECK(g->is_terminated());
  CHECK_FALSE(g->accept_tokens("r", {14}));    // nothing follows EOS
}

// (e) Rollback across the trigger boundary restores awaiting + the buffer; reset
// restores awaiting = lazy.
TEST_CASE("native lazy: rollback across the trigger boundary restores awaiting") {
  auto backend = MakeLazyBackend();
  auto g = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});
  REQUIRE(g->accept_tokens("r", {16}));  // trigger fires
  CHECK_FALSE(g->is_terminated());
  CHECK(LazyAllowed(*g, 9));             // constrained: "{"
  CHECK_FALSE(LazyAllowed(*g, 14));      // "Z" forbidden

  g->rollback(1);                        // back across the trigger boundary
  CHECK(g->is_terminated());             // awaiting again -> terminable
  CHECK(LazyAllowed(*g, 14));            // all allowed again ("Z")
  CHECK(LazyAllowed(*g, 9));             // and "{"
  CHECK(LazyAllowed(*g, kLazyEos));      // EOS allowed again

  REQUIRE(g->accept_tokens("r", {16}));  // re-fire the trigger
  CHECK(LazyAllowed(*g, 9));
  CHECK_FALSE(LazyAllowed(*g, 14));

  g->reset();                            // reset -> awaiting = lazy
  CHECK(g->is_terminated());
  CHECK(LazyAllowed(*g, 14));
}

// A trigger TOKEN id (not a word) activates the grammar; the token's bytes then
// advance the FSM through the "<tool_call>" literal.
TEST_CASE("native lazy: a trigger TOKEN id activates the grammar") {
  auto backend = MakeLazyBackend();
  auto g = CompileLazy(*backend, kToolGrammar, /*words=*/{}, /*tokens=*/{16});
  CHECK(g->is_terminated());             // awaiting
  CHECK(LazyAllowed(*g, 14));            // all allowed
  REQUIRE(g->accept_tokens("r", {16}));  // trigger TOKEN
  CHECK_FALSE(g->is_terminated());
  CHECK(LazyAllowed(*g, 9));             // constrained: "{"
  CHECK_FALSE(LazyAllowed(*g, 14));
}

// A lazy grammar with no triggers at all is rejected (it would never activate).
TEST_CASE("native lazy: compile_lazy_grammar rejects an empty trigger set") {
  auto backend = MakeLazyBackend();
  CHECK_THROWS(backend->compile_lazy_grammar(kToolGrammar, {}, {}));
}

// (f) THE INVARIANT holds POST-trigger: once triggered the grammar behaves like a
// normal grammar, so fill_bitmask == accept_tokens over the WHOLE vocab at every
// state along the JSON body.
TEST_CASE("native lazy: fill==accept invariant holds POST-trigger") {
  auto backend = MakeLazyBackend();
  const int vocab = LazyVocabSize();
  const std::vector<std::vector<int32_t>> prefixes = {
      {16},             // after "<tool_call>"  -> expects "{"
      {16, 9},          // after "{"            -> expects a digit
      {16, 9, 11},      // after a digit        -> expects "}"
      {16, 9, 11, 10},  // after "}"            -> expects "</tool_call>"
  };
  for (const auto& prefix : prefixes) {
    auto filled = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});
    REQUIRE(filled->accept_tokens("p", prefix));
    REQUIRE_FALSE(filled->is_terminated());  // triggered, mid tool-call

    TokenBitmask bm;
    bm.num_seqs = 1;
    bm.num_words = BitmaskWordsForVocab(vocab);
    bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
    filled->fill_bitmask(bm, 0);

    for (int32_t t = 0; t < vocab; ++t) {
      const bool fill_allows =
          (bm.data[static_cast<std::size_t>(t >> 5)] &
           static_cast<int32_t>(1u << (static_cast<uint32_t>(t) & 31u))) != 0;
      auto probe = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});
      REQUIRE(probe->accept_tokens("p", prefix));
      const bool accept_allows = probe->accept_tokens("t", {t});
      INFO("prefix_len=" << prefix.size() << " token=" << t
                         << " fill=" << fill_allows
                         << " accept=" << accept_allows);
      CHECK(fill_allows == accept_allows);
    }
  }
  backend->destroy();
}

// ───────── M3.3b: ADDED-token structural markers (real-Qwen3.6 shape) ─────────
// The REAL Qwen3.6 tokenizer emits `<tool_call>`/`</tool_call>` each as a SINGLE
// ADDED token (special:false, ids 248058/248059), NOT as regular BPE sub-text.
// An added token carries LITERAL content bytes; the grammar's `<tool_call>`
// literal must match that content, so added markers MUST live in the trie +
// token_bytes (keyed by their raw content). This fixture mirrors that shape:
// `<tool_call>`/`</tool_call>` are ADDED tokens (single ids), the JSON body is
// regular single-char tokens. (The earlier LazyFixture spells the wrapper with
// REGULAR vocab tokens, so it does NOT exercise this path.)
namespace {

constexpr int32_t kAddedToolOpen = 10;   // "<tool_call>"  (added, special:false)
constexpr int32_t kAddedToolClose = 11;  // "</tool_call>" (added, special:false)
constexpr int32_t kAddedEos = 12;        // "<eos>"        (added, special:true)

Tokenizer BuildAddedFixture() {
  json vocab = {{"{", 0}, {"}", 1}, {"5", 2}, {"Z", 3}, {"P", 4}};

  json doc;
  doc["version"] = "1.0";
  doc["added_tokens"] = json::array(
      {{{"id", kAddedToolOpen}, {"content", "<tool_call>"}, {"special", false}},
       {{"id", kAddedToolClose},
        {"content", "</tool_call>"},
        {"special", false}},
       {{"id", kAddedEos}, {"content", "<eos>"}, {"special", true}}});
  doc["normalizer"] = nullptr;
  doc["pre_tokenizer"] = {
      {"type", "Sequence"},
      {"pretokenizers",
       json::array(
           {{{"type", "Split"},
             {"pattern",
              {{"Regex",
                R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)"}}},
             {"behavior", "Isolated"},
             {"invert", false}},
            {{"type", "ByteLevel"},
             {"add_prefix_space", false},
             {"trim_offsets", false},
             {"use_regex", false}}})}};
  doc["model"] = {{"type", "BPE"},
                  {"ignore_merges", false},
                  {"vocab", vocab},
                  {"merges", json::array()}};
  TempJson f(doc.dump());
  return Tokenizer::FromHfJson(f.path());
}

const Tokenizer& AddedFixture() {
  static const Tokenizer tok = BuildAddedFixture();
  return tok;
}

int AddedVocabSize() { return static_cast<int>(AddedFixture().VocabSize()); }

std::unique_ptr<NativeStructuredOutputBackend> MakeAddedBackend() {
  return std::make_unique<NativeStructuredOutputBackend>(
      AddedFixture(), AddedVocabSize(), std::vector<int32_t>{kAddedEos});
}

bool AddedAllowed(NativeGrammar& g, int32_t t) {
  TokenBitmask bm;
  bm.num_seqs = 1;
  bm.num_words = BitmaskWordsForVocab(AddedVocabSize());
  bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
  g.fill_bitmask(bm, 0);
  const int32_t word = bm.data[static_cast<std::size_t>(t >> 5)];
  return (word & (1 << (t & 31))) != 0;
}

}  // namespace

// (a) The WORD trigger `<tool_call>` FIRES when the model emits the `<tool_call>`
// ADDED token (a single id) — proving the added token is buffered as its literal
// content. BEFORE the fix the added token was excluded from token_bytes, so it
// was never buffered, the WORD trigger never fired, and auto tool-calling was a
// silent no-op for the real model.
TEST_CASE("native added-marker: WORD trigger fires on the <tool_call> ADDED token") {
  auto backend = MakeAddedBackend();
  auto g = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});

  REQUIRE(g->accept_tokens("r", {3}));  // "Z" free text
  CHECK(g->is_terminated());            // still just a plain reply (awaiting)

  REQUIRE(g->accept_tokens("r", {kAddedToolOpen}));  // the ADDED trigger token
  CHECK_FALSE(g->is_terminated());       // triggered: now mid tool-call
  CHECK(AddedAllowed(*g, 0));            // only "{" opens the body
  CHECK_FALSE(AddedAllowed(*g, 3));      // a letter is forbidden now
  CHECK_FALSE(AddedAllowed(*g, kAddedEos));  // not accepting mid tool-call
}

// (b) The FULL tool call completes when `</tool_call>` is the closing ADDED
// token: <tool_call>(added) + "{" + digit + "}" + </tool_call>(added) -> the FSM
// consumes the closing marker's literal content and terminates. BEFORE the fix
// `</tool_call>` was excluded from token_bytes, so the active FSM could never
// consume it and the tool call could never complete.
TEST_CASE("native added-marker: full tool call completes via </tool_call> ADDED token") {
  auto backend = MakeAddedBackend();
  auto g = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});

  REQUIRE(g->accept_tokens("r", {kAddedToolOpen}));  // "<tool_call>" (added)
  REQUIRE(g->accept_tokens("r", {0}));               // "{"
  REQUIRE(g->accept_tokens("r", {2}));               // "5" (a digit)
  REQUIRE(g->accept_tokens("r", {1}));               // "}"
  CHECK_FALSE(g->is_terminated());                   // still needs "</tool_call>"
  REQUIRE(g->accept_tokens("r", {kAddedToolClose}));  // "</tool_call>" (added)
  CHECK(g->is_terminated());  // full <tool_call>{5}</tool_call>
}

// (c) THE fill==accept differential invariant still holds with the added-token
// vocab, at every state along the tool-call body (fill and accept share the same
// trie/token_bytes source, so this is preserved by construction — verify it).
TEST_CASE("native added-marker: fill==accept invariant holds with added markers") {
  auto backend = MakeAddedBackend();
  const int vocab = AddedVocabSize();
  const std::vector<std::vector<int32_t>> prefixes = {
      {},                                       // awaiting (inert)
      {kAddedToolOpen},                         // after "<tool_call>" -> "{"
      {kAddedToolOpen, 0},                      // after "{"           -> digit
      {kAddedToolOpen, 0, 2},                   // after digit         -> "}"
      {kAddedToolOpen, 0, 2, 1},                // after "}"           -> "</tool_call>"
  };
  for (const auto& prefix : prefixes) {
    auto filled = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});
    if (!prefix.empty()) REQUIRE(filled->accept_tokens("p", prefix));

    TokenBitmask bm;
    bm.num_seqs = 1;
    bm.num_words = BitmaskWordsForVocab(vocab);
    bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
    filled->fill_bitmask(bm, 0);

    for (int32_t t = 0; t < vocab; ++t) {
      const bool fill_allows =
          (bm.data[static_cast<std::size_t>(t >> 5)] &
           static_cast<int32_t>(1u << (static_cast<uint32_t>(t) & 31u))) != 0;
      auto probe = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});
      if (!prefix.empty()) REQUIRE(probe->accept_tokens("p", prefix));
      const bool accept_allows = probe->accept_tokens("t", {t});
      INFO("prefix_len=" << prefix.size() << " token=" << t
                         << " fill=" << fill_allows
                         << " accept=" << accept_allows);
      CHECK(fill_allows == accept_allows);
    }
  }
  backend->destroy();
}

// (d) A plain reply that never emits `<tool_call>` stays free throughout and can
// terminate — the added markers being matchable does NOT force a tool call.
TEST_CASE("native added-marker: a plain reply never triggers and stays free") {
  auto backend = MakeAddedBackend();
  auto g = CompileLazy(*backend, kToolGrammar, {"<tool_call>"});
  REQUIRE(g->accept_tokens("r", {4, 3, 4}));  // "P","Z","P"
  CHECK(AddedAllowed(*g, 3));                  // still all allowed
  CHECK(AddedAllowed(*g, 0));
  CHECK(g->is_terminated());                   // may end (no tool call forced)
  REQUIRE(g->accept_tokens("r", {kAddedEos}));  // end the plain reply
  CHECK(g->is_terminated());
}

// ───────────────────── M3.3b Task 2: STRUCTURAL_TAG native compile ───────────
// The native structural-tag SPEC (our own JSON — the SEAM is 1:1 with vLLM, the
// content is backend-private, see backend_native.cpp):
//   {"lazy": bool, "triggers": [str], "stop_after_first": bool,
//    "tags": [{"begin": str, "content_schema": <schema|true>, "end": str}]}
// compile_grammar(kStructuralTag, spec) parses it and builds a grammar: each tag
// -> begin-literal + JSON body (schema-lowered, or any-JSON when content_schema
// is true) + end-literal; MULTIPLE tags -> an alternation; stop_after_first ->
// exactly one tag, else one-or-more; lazy -> the Task-1 lazy matcher triggered by
// `triggers`, else forced from token 0. Driven token-by-token over the same real
// BPE fixture whose vocab spells the `<tool_call>`/`{`/digit/`}`/`</tool_call>`
// wrapper (LazyFixture, above).
namespace {

std::unique_ptr<NativeGrammar> CompileStructural(
    NativeStructuredOutputBackend& backend, const std::string& spec) {
  std::unique_ptr<StructuredOutputGrammar> g =
      backend.compile_grammar(StructuredOutputOptions::kStructuralTag, spec);
  return std::unique_ptr<NativeGrammar>(
      static_cast<NativeGrammar*>(g.release()));
}

// A LAZY structural tag: triggered by `<tool_call>`, a single tag whose body is
// an integer schema wrapped as `<tool_call>` <int> `</tool_call>`.
const char* const kLazyIntStructuralTag = R"JSON({
  "lazy": true,
  "triggers": ["<tool_call>"],
  "stop_after_first": true,
  "tags": [
    {"begin": "<tool_call>", "content_schema": {"type": "integer"},
     "end": "</tool_call>"}
  ]
})JSON";

}  // namespace

// (b) A LAZY structural-tag spec is inert until the trigger, then constrains the
// JSON body to the tag's schema, and terminates when the tag completes.
TEST_CASE("native structural-tag: lazy spec is inert then constrains after the "
          "trigger") {
  auto backend = MakeLazyBackend();
  auto g = CompileStructural(*backend, kLazyIntStructuralTag);

  // Before the trigger: inert — every token allowed (free text) + terminable.
  CHECK(LazyAllowed(*g, 14));        // "Z" (a letter)
  CHECK(LazyAllowed(*g, 9));         // "{"
  CHECK(LazyAllowed(*g, kLazyEos));  // a plain reply may end
  CHECK(g->is_terminated());

  // The trigger fires: after `<tool_call>` the body is constrained to an integer.
  REQUIRE(g->accept_tokens("r", {16}));  // "<tool_call>"
  CHECK_FALSE(g->is_terminated());       // mid tool-call
  CHECK(LazyAllowed(*g, 11));            // a digit "5"
  CHECK(LazyAllowed(*g, 12));            // "0"
  CHECK_FALSE(LazyAllowed(*g, 14));      // a letter is forbidden
  CHECK_FALSE(LazyAllowed(*g, 0));       // "<" forbidden
  CHECK_FALSE(LazyAllowed(*g, 9));       // "{" forbidden (integer, not object)
  CHECK_FALSE(LazyAllowed(*g, kLazyEos));

  REQUIRE(g->accept_tokens("r", {11}));  // "5" (the integer)
  CHECK(LazyAllowed(*g, 18));            // the end literal "</tool_call>"
  CHECK(LazyAllowed(*g, 12));            // integer may continue with a digit
  REQUIRE(g->accept_tokens("r", {18}));  // "</tool_call>"
  CHECK(g->is_terminated());             // the tag completed
}

// A plain reply that never triggers stays free and can end (no tool call forced).
TEST_CASE("native structural-tag: lazy spec, a reply that never triggers stays "
          "free") {
  auto backend = MakeLazyBackend();
  auto g = CompileStructural(*backend, kLazyIntStructuralTag);
  REQUIRE(g->accept_tokens("r", {15, 14, 15}));  // "P","Z","P" free text
  CHECK(LazyAllowed(*g, 14));
  CHECK(g->is_terminated());
  REQUIRE(g->accept_tokens("r", {kLazyEos}));
  CHECK(g->is_terminated());
}

// (c) A NON-LAZY (required) spec is FORCED from token 0 and matches one-or-more
// tags; the alternation of two tags is exercised (begin `<tool_call>` OR `{`).
TEST_CASE("native structural-tag: non-lazy required spec is forced from token 0, "
          "one-or-more, alternation") {
  auto backend = MakeLazyBackend();
  auto g = CompileStructural(*backend, R"JSON({
    "lazy": false,
    "stop_after_first": false,
    "tags": [
      {"begin": "<tool_call>", "content_schema": {"type": "integer"},
       "end": "</tool_call>"},
      {"begin": "{", "content_schema": {"type": "integer"}, "end": "}"}
    ]
  })JSON");

  // FORCED from token 0: NOT inert — only a tag's begin-literal opens.
  CHECK_FALSE(g->is_terminated());
  CHECK(LazyAllowed(*g, 0));         // "<"  (tag A begin)
  CHECK(LazyAllowed(*g, 16));        // "<tool_call>" (tag A begin)
  CHECK(LazyAllowed(*g, 9));         // "{"  (tag B begin — alternation)
  CHECK_FALSE(LazyAllowed(*g, 14));  // "Z" forbidden (forced, not free)
  CHECK_FALSE(LazyAllowed(*g, 11));  // "5" forbidden (must open a tag first)
  CHECK_FALSE(LazyAllowed(*g, kLazyEos));  // ≥1 tag required

  // Drive tag A: <tool_call> 5 </tool_call>.
  REQUIRE(g->accept_tokens("r", {16}));  // "<tool_call>"
  CHECK(LazyAllowed(*g, 11));            // integer digit
  CHECK_FALSE(LazyAllowed(*g, 9));       // not "{" (tag A body is an integer)
  REQUIRE(g->accept_tokens("r", {11}));  // "5"
  REQUIRE(g->accept_tokens("r", {18}));  // "</tool_call>"
  // one-or-more: after one tag it may STOP (accepting) or start ANOTHER.
  CHECK(LazyAllowed(*g, kLazyEos));      // accepting -> EOS allowed
  CHECK(LazyAllowed(*g, 16));            // may start another tag A
  CHECK(LazyAllowed(*g, 9));             // ...or tag B ("{")
}

// (d) A NAMED (stop_after_first) spec is forced and matches EXACTLY ONE tag.
TEST_CASE("native structural-tag: named spec (stop_after_first) is exactly one "
          "tag") {
  auto backend = MakeLazyBackend();
  auto g = CompileStructural(*backend, R"JSON({
    "lazy": false,
    "stop_after_first": true,
    "tags": [
      {"begin": "<tool_call>", "content_schema": {"type": "integer"},
       "end": "</tool_call>"}
    ]
  })JSON");

  CHECK_FALSE(g->is_terminated());
  CHECK(LazyAllowed(*g, 16));  // forced: the begin literal
  CHECK_FALSE(LazyAllowed(*g, 14));

  REQUIRE(g->accept_tokens("r", {16}));  // "<tool_call>"
  REQUIRE(g->accept_tokens("r", {11}));  // "5"
  REQUIRE(g->accept_tokens("r", {18}));  // "</tool_call>"
  CHECK(g->is_terminated());             // exactly one tag -> complete
  CHECK_FALSE(LazyAllowed(*g, 16));      // cannot start another tag
  CHECK_FALSE(LazyAllowed(*g, 9));
  CHECK(LazyAllowed(*g, kLazyEos));      // only EOS follows
}

// (e) content_schema: true -> any well-formed JSON value inside the tag.
TEST_CASE("native structural-tag: content_schema true allows any JSON inside") {
  auto backend = MakeLazyBackend();
  auto g = CompileStructural(*backend, R"JSON({
    "lazy": true,
    "triggers": ["<tool_call>"],
    "stop_after_first": true,
    "tags": [
      {"begin": "<tool_call>", "content_schema": true, "end": "</tool_call>"}
    ]
  })JSON");

  REQUIRE(g->accept_tokens("r", {16}));  // "<tool_call>" (trigger)
  CHECK_FALSE(g->is_terminated());
  CHECK(LazyAllowed(*g, 11));            // a number ("5") is a JSON value
  CHECK(LazyAllowed(*g, 9));             // an object ("{") is a JSON value
  CHECK(LazyAllowed(*g, 3));             // "t" opens the literal "true"
  CHECK_FALSE(LazyAllowed(*g, 14));      // "Z" is not a JSON value start
  REQUIRE(g->accept_tokens("r", {11}));  // "5" (a number value)
  REQUIRE(g->accept_tokens("r", {18}));  // "</tool_call>"
  CHECK(g->is_terminated());
}

// A malformed / empty structural-tag spec throws loudly.
TEST_CASE("native structural-tag: malformed spec throws") {
  auto backend = MakeLazyBackend();
  CHECK_THROWS(CompileStructural(*backend, "{not valid json"));
  CHECK_THROWS(CompileStructural(*backend, R"({"tags": []})"));  // no tags
  CHECK_THROWS(CompileStructural(
      *backend, R"({"lazy": true, "tags": [{"begin":"<tool_call>","end":"</tool_call>"}]})"));  // lazy but no triggers
}

// (f) THE INVARIANT holds POST-trigger for a structural-tag lazy grammar: once
// triggered it behaves like a normal grammar, so fill_bitmask == accept_tokens
// over the WHOLE vocab at every state along the tag body.
TEST_CASE("native structural-tag: fill==accept invariant holds POST-trigger") {
  auto backend = MakeLazyBackend();
  const int vocab = LazyVocabSize();
  const std::vector<std::vector<int32_t>> prefixes = {
      {16},          // after "<tool_call>" -> expects an integer digit
      {16, 11},      // after "5"           -> more digits or "</tool_call>"
  };
  for (const auto& prefix : prefixes) {
    auto filled = CompileStructural(*backend, kLazyIntStructuralTag);
    REQUIRE(filled->accept_tokens("p", prefix));
    REQUIRE_FALSE(filled->is_terminated());

    TokenBitmask bm;
    bm.num_seqs = 1;
    bm.num_words = BitmaskWordsForVocab(vocab);
    bm.data.assign(static_cast<std::size_t>(bm.num_words), 0);
    filled->fill_bitmask(bm, 0);

    for (int32_t t = 0; t < vocab; ++t) {
      const bool fill_allows =
          (bm.data[static_cast<std::size_t>(t >> 5)] &
           static_cast<int32_t>(1u << (static_cast<uint32_t>(t) & 31u))) != 0;
      auto probe = CompileStructural(*backend, kLazyIntStructuralTag);
      REQUIRE(probe->accept_tokens("p", prefix));
      const bool accept_allows = probe->accept_tokens("t", {t});
      INFO("prefix_len=" << prefix.size() << " token=" << t
                         << " fill=" << fill_allows
                         << " accept=" << accept_allows);
      CHECK(fill_allows == accept_allows);
    }
  }
  backend->destroy();
}
