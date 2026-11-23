// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE. The handcrafted fixture below (kTinyJson) was verified against the
// real HF tokenizers library (NOT hand-guessed):
//   scp tiny_tokenizer.json oracle_tiny.py dgx.casa:/tmp/ &&
//   ssh dgx.casa '~/venvs/vllm-oracle/bin/python /tmp/oracle_tiny.py'
// (tokenizers 0.22.2). Every expected-id vector in the encode/decode tests
// below is copied verbatim from that oracle run, including the
// ignore_merges=true variant.
#include <doctest/doctest.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>

#include "gguf_builder.h"
#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/tokenizer.h"

using vllm::tok::BpeSplit;
using vllm::tok::ByteToUnicode;
using vllm::tok::MapBytesToUnicode;
using vllm::tok::MergeRanks;
using vllm::tok::SplitPattern;
using vllm::tok::Tokenizer;
using vllm::tok::UnicodeToByte;
using vllm::tok::UnmapUnicodeToBytes;

namespace {

using Ids = std::vector<int32_t>;

// Writes `body` to a unique file under the system temp dir; removed in the
// destructor so test runs don't accumulate files.
class TempJson {
 public:
  explicit TempJson(const std::string& body) {
    static int counter = 0;
    path_ = (std::filesystem::temp_directory_path() /
             ("vllm_bpe_test_" + std::to_string(counter++) + ".json"))
                .string();
    std::ofstream out(path_, std::ios::binary);
    out << body;
  }
  ~TempJson() { std::remove(path_.c_str()); }
  const std::string& path() const { return path_; }

 private:
  std::string path_;
};

// Replaces exactly one occurrence of `from` (which must be present).
std::string ReplaceOnce(std::string s, const std::string& from,
                        const std::string& to) {
  const size_t pos = s.find(from);
  REQUIRE(pos != std::string::npos);
  return s.replace(pos, from.size(), to);
}

// Checks that `fn` throws std::runtime_error whose message contains `needle`.
template <typename Fn>
void CheckThrowsContains(Fn&& fn, const std::string& needle) {
  try {
    fn();
    FAIL_CHECK("expected std::runtime_error containing \"" << needle << "\"");
  } catch (const std::runtime_error& e) {
    CHECK_MESSAGE(std::string(e.what()).find(needle) != std::string::npos,
                  "message was: " << e.what());
  }
}

// Handcrafted 19-token vocab + 3 added tokens; oracle-verified (see header).
// Merge order deliberately interleaves ("hello" needs ll before "he llo";
// "Ġworld" needs or+ld -> orld before "Ġw orld").
constexpr const char* kTinyJson = R"json({
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [
    {"id": 19, "content": "<|end|>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true},
    {"id": 20, "content": "<tool>", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": false},
    {"id": 21, "content": "<|end|>of", "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true}
  ],
  "normalizer": null,
  "pre_tokenizer": {
    "type": "Sequence",
    "pretokenizers": [
      {"type": "Split", "pattern": {"Regex": "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+"}, "behavior": "Isolated", "invert": false},
      {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
    ]
  },
  "post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false},
  "model": {
    "type": "BPE",
    "dropout": null,
    "unk_token": null,
    "continuing_subword_prefix": null,
    "end_of_word_suffix": null,
    "fuse_unk": false,
    "byte_fallback": false,
    "ignore_merges": false,
    "vocab": {"h": 0, "e": 1, "l": 2, "o": 3, "w": 4, "r": 5, "d": 6, "Ġ": 7, "1": 8, "2": 9, "ll": 10, "he": 11, "llo": 12, "hello": 13, "Ġw": 14, "or": 15, "orld": 16, "Ġworld": 17, "ld": 18},
    "merges": [["l", "l"], ["h", "e"], ["ll", "o"], ["he", "llo"], ["Ġ", "w"], ["o", "r"], ["l", "d"], ["or", "ld"], ["Ġw", "orld"]]
  }
})json";

// Verbatim regex variants (as they appear escaped inside tokenizer.json).
constexpr const char* kQwen36RegexJson =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?[\\p{L}\\p{M}]+|\\p{N}| ?[^\\s\\p{L}\\p{M}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";
constexpr const char* kClassicQwen2RegexJson =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";
constexpr const char* kLlama3RegexJson =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}{1,3}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+)";

Tokenizer LoadTiny(const std::string& body = kTinyJson) {
  TempJson f(body);
  return Tokenizer::FromHfJson(f.path());
}

std::string WithRegex(const char* regex_json) {
  return ReplaceOnce(kTinyJson, kQwen36RegexJson, regex_json);
}

// GGUF kv blocks carrying the same oracle-blessed fixture as kTinyJson:
// tokens indexed 0..21 (index = id), specials expressed via token_type
// (3=control -> special added token, 4=user-defined -> non-special).
// Kv order (tests mutate by index): 0=model 1=pre 2=tokens 3=token_type
// 4=merges 5=eos_token_id.
std::vector<std::string> TinyGgufKvs() {
  const std::vector<std::string> tokens = {
      "h",  "e",  "l",   "o",     "w",  "r",  "d",    "Ġ",      "1",
      "2",  "ll", "he",  "llo",   "hello",    "Ġw",   "or",     "orld",
      "Ġworld",   "ld",  "<|end|>",     "<tool>",     "<|end|>of"};
  std::vector<int32_t> types(19, 1);  // ids 0..18: normal
  types.insert(types.end(), {3, 4, 3});
  const std::vector<std::string> merges = {"l l",  "h e", "ll o",
                                           "he llo",     "Ġ w", "o r",
                                           "l d",  "or ld",     "Ġw orld"};
  return {
      gguf_test::StrKv("tokenizer.ggml.model", "gpt2"),
      gguf_test::StrKv("tokenizer.ggml.pre", "qwen35"),
      gguf_test::StrArrayKv("tokenizer.ggml.tokens", tokens),
      gguf_test::I32ArrayKv("tokenizer.ggml.token_type", types),
      gguf_test::StrArrayKv("tokenizer.ggml.merges", merges),
      gguf_test::U32Kv("tokenizer.ggml.eos_token_id", 19),
  };
}

// Assembles a zero-tensor GGUF from the kv blocks and loads it via FromGguf.
Tokenizer LoadGguf(const std::vector<std::string>& kvs) {
  std::string bytes = gguf_test::Header(3, /*tensor_count=*/0, kvs.size());
  for (const auto& kv : kvs) bytes += kv;
  gguf_test::TempFile f(bytes);
  const vllm::GgufFile g = vllm::GgufFile::Open(f.path());
  return Tokenizer::FromGguf(g);
}

}  // namespace

TEST_CASE("bytes_to_unicode bijection: 256 distinct cps, exact GPT-2 table") {
  std::set<uint32_t> seen;
  for (int b = 0; b < 256; ++b) {
    const uint32_t cp = ByteToUnicode(static_cast<uint8_t>(b));
    CHECK(seen.insert(cp).second);  // injective
    CHECK(UnicodeToByte(cp) == b);  // round-trip
    const bool printable = (b >= 0x21 && b <= 0x7E) ||
                           (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF);
    if (printable) CHECK(cp == static_cast<uint32_t>(b));
  }
  // Spot values of the 0x100+n branch (n counts non-printables in byte order).
  CHECK(ByteToUnicode(0x00) == 0x100);
  CHECK(ByteToUnicode(0x0A) == 0x10A);  // '\n' -> 'Ċ'
  CHECK(ByteToUnicode(0x20) == 0x120);  // ' '  -> 'Ġ'
  CHECK(ByteToUnicode(0x7F) == 0x121);
  CHECK(ByteToUnicode(0xA0) == 0x142);
  CHECK(ByteToUnicode(0xAD) == 0x143);
  // Codepoints outside the image.
  CHECK(UnicodeToByte(0x20) == -1);
  CHECK(UnicodeToByte(0x144) == -1);
  CHECK(UnicodeToByte(0x20AC) == -1);
}

TEST_CASE("MapBytesToUnicode/UnmapUnicodeToBytes round-trip all bytes") {
  std::string all;
  for (int b = 0; b < 256; ++b) all.push_back(static_cast<char>(b));
  const std::string mapped = MapBytesToUnicode(all);
  CHECK(UnmapUnicodeToBytes(mapped) == all);
  CHECK(MapBytesToUnicode(" hi\n") == "Ġhi\xC4\x8A");  // Ġ + hi + Ċ
  CHECK(MapBytesToUnicode("") == "");
  // Codepoints outside the bijection image throw.
  CHECK_THROWS_AS((void)UnmapUnicodeToBytes(" "), std::runtime_error);
  CHECK_THROWS_AS((void)UnmapUnicodeToBytes("€"), std::runtime_error);
  // Invalid UTF-8 decodes to U+FFFD, which is outside the image.
  CHECK_THROWS_AS((void)UnmapUnicodeToBytes("\xFF"), std::runtime_error);
}

TEST_CASE("BpeSplit merges lowest rank first, leftmost on ties") {
  MergeRanks ranks;
  CHECK(ranks.Insert("l", "l", 0));
  CHECK(ranks.Insert("h", "e", 1));
  CHECK(ranks.Insert("ll", "o", 2));
  CHECK(ranks.Insert("he", "llo", 3));
  // A repeated pair is refused rather than silently re-ranked; the loader
  // turns that into a named failure.
  CHECK_FALSE(ranks.Insert("l", "l", 99));
  using V = std::vector<std::string>;
  // h e l l o -> h e ll o -> he ll o -> he llo -> hello
  CHECK(BpeSplit("hello", ranks) == V{"hello"});
  // Rank-0 "l l" fires at the LEFT of the lll run (oracle: "helllo" ->
  // [11, 10, 2, 3] = he ll l o), then nothing else applies.
  CHECK(BpeSplit("helllo", ranks) == V{"he", "ll", "l", "o"});
  CHECK(BpeSplit("lll", ranks) == V{"ll", "l"});  // oracle: [10, 2]
  CHECK(BpeSplit("eh", ranks) == V{"e", "h"});  // no pair applies
  CHECK(BpeSplit("", ranks).empty());
  CHECK(BpeSplit("h", ranks) == V{"h"});
  // Multi-byte mapped symbols merge as whole codepoints.
  MergeRanks r2;
  CHECK(r2.Insert("Ġ", "Ġ", 0));
  CHECK(BpeSplit("ĠĠĠ", r2) == V{"ĠĠ", "Ġ"});  // leftmost tie
}

TEST_CASE("the leftmost tie holds on a long symbol list") {
  // The comparator is where leftmost-wins lives. `Ord for Merge`
  // (tokenizers/src/models/bpe/word.rs:28-35) inverts both comparisons so the
  // max-heap yields the lowest rank first and the lowest POSITION on a tie. A
  // three-symbol case cannot separate leftmost from rightmost past the first
  // merge; these can.
  //
  // ORACLE: HF tokenizers 0.22.2 on a tokenizer.json with vocab {"a":0,"aa":1}
  // and merges [["a","a"]] returns, for "aa".."aaaaaa":
  //   [1] [1,0] [1,1] [1,1,0] [1,1,1] = aa / aa a / aa aa / aa aa a / aa aa aa
  // A rightmost-wins order gives "a aa" for "aaa" and "a aa aa" for "aaaaa".
  MergeRanks ranks;
  CHECK(ranks.Insert("a", "a", 0));
  using V = std::vector<std::string>;
  CHECK(BpeSplit("aa", ranks) == V{"aa"});
  CHECK(BpeSplit("aaa", ranks) == V{"aa", "a"});
  CHECK(BpeSplit("aaaa", ranks) == V{"aa", "aa"});
  CHECK(BpeSplit("aaaaa", ranks) == V{"aa", "aa", "a"});
  CHECK(BpeSplit("aaaaaa", ranks) == V{"aa", "aa", "aa"});
}

TEST_CASE("a queued candidate invalidated by an earlier merge is SKIPPED") {
  // The stale-entry case, built so that applying the stale entry produces
  // DIFFERENT identifiers rather than the same ones by luck.
  //
  //   rank 0  b + c  -> bc
  //   rank 1  a + b  -> ab
  //   rank 2  a + bc -> abc
  //
  // Seeded from "a b c" the queue holds (pos 0, rank 1, new_id ab) and
  // (pos 1, rank 0, new_id bc). Rank 0 pops first and merges b+c, which leaves
  // the rank-1 entry at position 0 STALE: the pair there is now (a, bc).
  //
  // (a, bc) IS in the table, so "does the pair exist" ACCEPTS the stale entry
  // and applies it with new_id = ab, yielding {"ab"} and losing "c" entirely.
  // Upstream compares `new_id` instead (word.rs:197-205): abc != ab, so the
  // entry is skipped, the rank-2 entry runs, and the answer is {"abc"}.
  //
  // ORACLE: HF tokenizers 0.22.2 on a tokenizer.json with that vocab and those
  // merges returns abc -> ["abc"], ababc -> ["ab", "abc"], aab -> ["a", "ab"],
  // abcb -> ["abc", "b"].
  MergeRanks ranks;
  CHECK(ranks.Insert("b", "c", 0));
  CHECK(ranks.Insert("a", "b", 1));
  CHECK(ranks.Insert("a", "bc", 2));
  using V = std::vector<std::string>;
  CHECK(BpeSplit("abc", ranks) == V{"abc"});
  CHECK(BpeSplit("ababc", ranks) == V{"ab", "abc"});
  CHECK(BpeSplit("aab", ranks) == V{"a", "ab"});
  CHECK(BpeSplit("abcb", ranks) == V{"abc", "b"});
  // The two pairs that reach position 0 do NOT share a new_id, which is what
  // makes the comparison observable at all.
  const MergeRanks::Entry* ab = ranks.Find(ranks.Find("a"), ranks.Find("b"));
  const MergeRanks::Entry* abc = ranks.Find(ranks.Find("a"), ranks.Find("bc"));
  REQUIRE(ab != nullptr);
  REQUIRE(abc != nullptr);
  CHECK(ab->new_id != abc->new_id);
}

TEST_CASE("a queued candidate whose left symbol lost its right neighbour") {
  // The second validation (word.rs:191). Same three tokens as the case above,
  // different merge ORDER, chosen so a queued entry survives the merge that
  // takes its position's right neighbour away:
  //
  //   rank 0  b + c  -> bc      rank 1  a + bc -> abc      rank 2  a + b -> ab
  //
  // Seeded from "a b c" the queue holds (pos 0, rank 2) and (pos 1, rank 0).
  // Rank 0 merges b+c and pushes (pos 0, rank 1). Rank 1 then merges a+bc, and
  // position 0's `next` becomes -1 because bc was the last symbol. The rank-2
  // entry is still queued at position 0, and it pops with NO RIGHT NEIGHBOUR.
  //
  // Deleting that guard does not change these ids: `next` is -1, the cast to an
  // index reads one Symbol BEFORE the vector, and the garbage identifier it
  // finds does not name a merge. It is a memory-safety guard, so its proof is
  // the sanitizer, not a value comparison — see the spec's `## Outcome`. This
  // case exists so the branch is REACHED at all, which is the half a value
  // assertion can still carry.
  //
  // ORACLE: HF tokenizers 0.22.2 on that vocab and that merge order returns
  // abc -> ["abc"].
  MergeRanks ranks;
  CHECK(ranks.Insert("b", "c", 0));
  CHECK(ranks.Insert("a", "bc", 1));
  CHECK(ranks.Insert("a", "b", 2));
  using V = std::vector<std::string>;
  CHECK(BpeSplit("abc", ranks) == V{"abc"});
  CHECK(BpeSplit("abcabc", ranks) == V{"abc", "abc"});
}

TEST_CASE("MergeRanks interns symbols and keys merges on identifier pairs") {
  // Mirrors HF `tokenizers` `MergeMap` (models/bpe/model.rs:19): the key is a
  // pair of identifiers and the value carries `(rank, new_id)`.
  MergeRanks ranks;
  REQUIRE(ranks.empty());
  CHECK(ranks.Insert("a", "b", 0));
  CHECK(ranks.Insert("ab", "c", 1));
  CHECK(ranks.size() == 2);
  // "a", "b", "ab", "c", "abc" — the concatenation is interned too, which is
  // what makes `new_id` a real identifier the next probe can use.
  CHECK(ranks.SymbolCount() == 5);

  const auto a = ranks.Find("a");
  const auto b = ranks.Find("b");
  const auto ab = ranks.Find("ab");
  REQUIRE(a != MergeRanks::kNoSymbol);
  REQUIRE(b != MergeRanks::kNoSymbol);
  REQUIRE(ab != MergeRanks::kNoSymbol);
  CHECK(ranks.Text(a) == "a");
  CHECK(ranks.Text(ab) == "ab");

  const MergeRanks::Entry* e = ranks.Find(a, b);
  REQUIRE(e != nullptr);
  CHECK(e->rank == 0);
  CHECK(e->new_id == ab);          // new_id IS the identifier of "ab"
  CHECK(ranks.Text(e->new_id) == "ab");
  CHECK(ranks.Find(b, a) == nullptr);  // the pair is ordered

  // A symbol no merge names has no identifier, and no merge can apply to it.
  CHECK(ranks.Find("z") == MergeRanks::kNoSymbol);
  CHECK(ranks.Find(MergeRanks::kNoSymbol, a) == nullptr);
  CHECK(ranks.Find(a, MergeRanks::kNoSymbol) == nullptr);
}

TEST_CASE("MergeRanks: two pairs with the same concatenation share one new_id") {
  // This is what upstream's staleness test rests on. `word.rs:197-205` compares
  // `new_id`, NOT the pair, so a table where two distinct pairs produce the
  // same string treats a queued entry for either as live. A pair comparison
  // would reject it. The property is only expressible on an identifier-keyed
  // table, which is why W2 comes before W3.
  MergeRanks ranks;
  CHECK(ranks.Insert("ab", "c", 0));   // -> "abc"
  CHECK(ranks.Insert("a", "bc", 1));   // -> "abc", a DIFFERENT pair
  const MergeRanks::Entry* left_first = ranks.Find(ranks.Find("ab"), ranks.Find("c"));
  const MergeRanks::Entry* right_first = ranks.Find(ranks.Find("a"), ranks.Find("bc"));
  REQUIRE(left_first != nullptr);
  REQUIRE(right_first != nullptr);
  CHECK(left_first->rank != right_first->rank);
  CHECK(left_first->new_id == right_first->new_id);
  CHECK(ranks.Text(left_first->new_id) == "abc");
}

TEST_CASE("FromHfJson: oracle-verified encode/decode on tiny fixture") {
  const Tokenizer tok = LoadTiny();
  CHECK(tok.VocabSize() == 22);
  CHECK(tok.Pattern() == SplitPattern::kQwen2);
  CHECK(tok.EosId() == -1);  // ByteLevel post_processor carries no eos/bos
  CHECK(tok.BosId() == -1);
  CHECK(tok.AddedTokens().size() == 3);
  CHECK(tok.AddedTokens()[0].special);
  CHECK_FALSE(tok.AddedTokens()[1].special);

  // Expected ids verbatim from the HF tokenizers oracle run (see header).
  CHECK(tok.Encode("hello") == Ids{13});
  CHECK(tok.Encode("hello world") == Ids{13, 17});
  CHECK(tok.Encode(" world") == Ids{17});
  CHECK(tok.Encode("world") == Ids{4, 16});  // "w orld" is not a merge
  CHECK(tok.Encode("hello12") == Ids{13, 8, 9});  // qwen splits digits
  CHECK(tok.Encode("") == Ids{});
  // Merge ties resolve leftmost (oracle: "lll" -> [10, 2], "llll" -> two ll).
  CHECK(tok.Encode("lll") == Ids{10, 2});
  CHECK(tok.Encode("llll") == Ids{10, 10});

  // Decode inverts, including the byte-level space (Ġ) unmapping.
  CHECK(tok.Decode({13, 17}) == "hello world");
  CHECK(tok.Decode({4, 16}) == "world");
  CHECK(tok.Decode({}) == "");

  // TokenText returns the raw byte-level-alphabet string.
  CHECK(tok.TokenText(17) == "Ġworld");
  CHECK(tok.TokenText(7) == "Ġ");
}

TEST_CASE("added tokens: leftmost-longest match before pretokenization") {
  const Tokenizer tok = LoadTiny();
  // Oracle: special and non-special added tokens split identically.
  CHECK(tok.Encode("hello<|end|> world") == Ids{13, 19, 17});
  CHECK(tok.Encode("hello<tool>world") == Ids{13, 20, 4, 16});
  // Longest wins at the same start: "<|end|>of" (21) beats "<|end|>" (19).
  CHECK(tok.Encode("hello<|end|>of world") == Ids{13, 21, 17});
  // Added tokens decode to their literal content.
  CHECK(tok.Decode({13, 19, 17}) == "hello<|end|> world");
  CHECK(tok.Decode({13, 21, 17}) == "hello<|end|>of world");
  CHECK(tok.TokenText(19) == "<|end|>");
}

TEST_CASE("string-form merges parse identically to array form") {
  const std::string body = ReplaceOnce(
      kTinyJson,
      R"("merges": [["l", "l"], ["h", "e"], ["ll", "o"], ["he", "llo"], ["Ġ", "w"], ["o", "r"], ["l", "d"], ["or", "ld"], ["Ġw", "orld"]])",
      R"("merges": ["l l", "h e", "ll o", "he llo", "Ġ w", "o r", "l d", "or ld", "Ġw orld"])");
  const Tokenizer tok = LoadTiny(body);
  CHECK(tok.Encode("hello world") == Ids{13, 17});
}

TEST_CASE("ignore_merges: whole-pretoken vocab hit wins, else normal BPE") {
  // Oracle-verified variant: ignore_merges=true, vocab += {"wo": 22}.
  std::string body =
      ReplaceOnce(kTinyJson, R"("ignore_merges": false)",
                  R"("ignore_merges": true)");
  body = ReplaceOnce(body, R"("h": 0,)", R"("h": 0, "wo": 22,)");
  const Tokenizer tok = LoadTiny(body);
  CHECK(tok.Encode("hello") == Ids{13});
  CHECK(tok.Encode("wo") == Ids{22});          // direct vocab hit, no merges
  CHECK(tok.Encode("world") == Ids{4, 16});    // not in vocab -> BPE ("wo" is
  CHECK(tok.Encode("hello world") == Ids{13, 17});  // NOT used mid-word)
}

TEST_CASE("NFC normalizer entry is accepted (Qwen3.6 has it)") {
  const std::string body = ReplaceOnce(kTinyJson, R"("normalizer": null)",
                                       R"("normalizer": {"type": "NFC"})");
  CHECK(LoadTiny(body).Encode("hello") == Ids{13});
}

TEST_CASE("post_processor TemplateProcessing: trivially extractable bos/eos") {
  const std::string body = ReplaceOnce(
      kTinyJson,
      R"("post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})",
      R"("post_processor": {"type": "TemplateProcessing",
        "single": [{"SpecialToken": {"id": "<|end|>", "type_id": 0}},
                   {"Sequence": {"id": "A", "type_id": 0}},
                   {"SpecialToken": {"id": "<|end|>of", "type_id": 0}}],
        "pair": [],
        "special_tokens": {
          "<|end|>": {"id": "<|end|>", "ids": [19], "tokens": ["<|end|>"]},
          "<|end|>of": {"id": "<|end|>of", "ids": [21], "tokens": ["<|end|>of"]}}})");
  const Tokenizer tok = LoadTiny(body);
  CHECK(tok.BosId() == 19);
  CHECK(tok.EosId() == 21);
}

TEST_CASE("llama3-style \\p{N}{1,3} split regex maps to kLlama3") {
  const Tokenizer tok = LoadTiny(WithRegex(kLlama3RegexJson));
  CHECK(tok.Pattern() == SplitPattern::kLlama3);
  // Digit grouping differs from qwen but the merged ids here coincide
  // (no digit merges in the tiny vocab).
  CHECK(tok.Encode("hello12") == Ids{13, 8, 9});
}

// HF AddedToken lstrip/rstrip. Every id vector below is copied VERBATIM from the
// real HF tokenizers 0.22.2 oracle run on the same fixture (the fast/Rust path
// vLLM itself runs):
//   ssh dgx.casa '~/venvs/vllm-oracle/bin/python /tmp/oracle_tiny_strip.py'
// Semantics (transformers 4.54.1 tokenization_utils.py:670-677): rstrip makes
// the matched token eat the WHOLE whitespace run to its right, lstrip the whole
// run to its left. Ignoring the flags silently mis-tokenizes any checkpoint that
// sets them (microsoft/Phi-4-mini-instruct sets rstrip=true on 10 of its 12
// added tokens), so this is a correctness contract, not a nicety.
TEST_CASE("added token lstrip/rstrip eat the adjacent whitespace") {
  const std::string kLstrip =
      ReplaceOnce(kTinyJson, R"("single_word": false, "lstrip": false)",
                  R"("single_word": false, "lstrip": true)");
  const std::string kRstrip = ReplaceOnce(
      kTinyJson, R"("single_word": false, "lstrip": false, "rstrip": false)",
      R"("single_word": false, "lstrip": false, "rstrip": true)");
  const std::string kBoth = ReplaceOnce(
      kTinyJson, R"("single_word": false, "lstrip": false, "rstrip": false)",
      R"("single_word": false, "lstrip": true, "rstrip": true)");

  SUBCASE("flags load instead of throwing, and are recorded") {
    const Tokenizer l = LoadTiny(kLstrip);
    CHECK(l.AddedTokens()[0].lstrip);
    CHECK_FALSE(l.AddedTokens()[0].rstrip);
    CHECK_FALSE(l.AddedTokens()[1].lstrip);  // untouched neighbours stay off
    const Tokenizer r = LoadTiny(kRstrip);
    CHECK_FALSE(r.AddedTokens()[0].lstrip);
    CHECK(r.AddedTokens()[0].rstrip);
    // Default (both false) is the historical behaviour, bit-for-bit.
    const Tokenizer b = LoadTiny();
    CHECK_FALSE(b.AddedTokens()[0].lstrip);
    CHECK_FALSE(b.AddedTokens()[0].rstrip);
  }
  SUBCASE("baseline: no flags, whitespace tokenizes normally") {
    const Tokenizer tok = LoadTiny();
    CHECK(tok.Encode("hello <|end|> world") == Ids{13, 7, 19, 17});
    CHECK(tok.Encode("hello  <|end|>  world") == Ids{13, 7, 7, 19, 7, 17});
    CHECK(tok.Encode("hello<|end|> world") == Ids{13, 19, 17});
    CHECK(tok.Encode("hello <|end|>world") == Ids{13, 7, 19, 4, 16});
    // Multi-space run on both sides (each ' ' -> Ġ=7): all six preserved.
    CHECK(tok.Encode("hello   <|end|>   world") ==
          Ids{13, 7, 7, 7, 19, 7, 7, 17});
    CHECK(tok.Encode("<|end|> world") == Ids{19, 17});
    CHECK(tok.Encode("hello <|end|>") == Ids{13, 7, 19});
  }
  SUBCASE("lstrip eats the whole run to the LEFT") {
    const Tokenizer tok = LoadTiny(kLstrip);
    CHECK(tok.Encode("hello <|end|> world") == Ids{13, 19, 17});
    CHECK(tok.Encode("hello  <|end|>  world") == Ids{13, 19, 7, 17});
    CHECK(tok.Encode("hello<|end|> world") == Ids{13, 19, 17});
    CHECK(tok.Encode("hello <|end|>world") == Ids{13, 19, 4, 16});
    // Whole LEFT run (three spaces) eaten; the RIGHT run stays (7,7).
    CHECK(tok.Encode("hello   <|end|>   world") == Ids{13, 19, 7, 7, 17});
    CHECK(tok.Encode("<|end|> world") == Ids{19, 17});  // nothing to the left
    CHECK(tok.Encode("hello <|end|>") == Ids{13, 19});
    // A non-flagged added token in the SAME tokenizer is unaffected.
    CHECK(tok.Encode("hello <tool> world") == Ids{13, 7, 20, 17});
  }
  SUBCASE("rstrip eats the whole run to the RIGHT") {
    const Tokenizer tok = LoadTiny(kRstrip);
    CHECK(tok.Encode("hello <|end|> world") == Ids{13, 7, 19, 4, 16});
    CHECK(tok.Encode("hello  <|end|>  world") == Ids{13, 7, 7, 19, 4, 16});
    CHECK(tok.Encode("hello<|end|> world") == Ids{13, 19, 4, 16});
    CHECK(tok.Encode("hello <|end|>world") == Ids{13, 7, 19, 4, 16});
    // LEFT run (7,7,7) stays; whole RIGHT run eaten.
    CHECK(tok.Encode("hello   <|end|>   world") == Ids{13, 7, 7, 7, 19, 4, 16});
    CHECK(tok.Encode("<|end|> world") == Ids{19, 4, 16});
    CHECK(tok.Encode("hello <|end|>") == Ids{13, 7, 19});  // nothing right
    CHECK(tok.Encode("hello <tool> world") == Ids{13, 7, 20, 17});
  }
  SUBCASE("both flags eat both runs") {
    const Tokenizer tok = LoadTiny(kBoth);
    CHECK(tok.Encode("hello <|end|> world") == Ids{13, 19, 4, 16});
    CHECK(tok.Encode("hello  <|end|>  world") == Ids{13, 19, 4, 16});
    CHECK(tok.Encode("hello<|end|> world") == Ids{13, 19, 4, 16});
    CHECK(tok.Encode("hello <|end|>world") == Ids{13, 19, 4, 16});
    CHECK(tok.Encode("hello   <|end|>   world") == Ids{13, 19, 4, 16});
    CHECK(tok.Encode("<|end|> world") == Ids{19, 4, 16});
    CHECK(tok.Encode("hello <|end|>") == Ids{13, 19});
    CHECK(tok.Encode("hello <tool> world") == Ids{13, 7, 20, 17});
  }
}

TEST_CASE("loader failure modes throw with actionable messages") {
  SUBCASE("unknown model type") {
    CheckThrowsContains(
        [&] { LoadTiny(ReplaceOnce(kTinyJson, R"("type": "BPE")",
                                   R"("type": "Unigram")")); },
        "Unigram");
  }
  SUBCASE("classic qwen2 regex resolves to kQwen2Classic") {
    // The first additive model (Qwen/Qwen3-0.6B, MODEL-TEXT-qwen3) forced classic
    // qwen2 pre-tokenizer support in: it is now ACCEPTED as SplitPattern::
    // kQwen2Classic (single-\p{N}, no \p{M} awareness), not rejected.
    const Tokenizer tok = LoadTiny(WithRegex(kClassicQwen2RegexJson));
    CHECK(tok.Pattern() == SplitPattern::kQwen2Classic);
  }
  SUBCASE("unrecognized split regex names the regex") {
    CheckThrowsContains(
        [&] { LoadTiny(WithRegex(R"([a-z]+|\\s+)")); },
        R"([a-z]+|\s+)");  // message carries the decoded regex
  }
  SUBCASE("unknown normalizer") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(kTinyJson, R"("normalizer": null)",
                               R"("normalizer": {"type": "Lowercase"})"));
        },
        "Lowercase");
  }
  SUBCASE("ByteLevel add_prefix_space=true unsupported") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(
              kTinyJson,
              R"("type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false}
    ])",
              R"("type": "ByteLevel", "add_prefix_space": true, "trim_offsets": false, "use_regex": false}
    ])"));
        },
        "add_prefix_space");
  }
  SUBCASE("added token id colliding with a vocab id") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(kTinyJson, R"({"id": 19, "content": "<|end|>")",
                               R"({"id": 13, "content": "<|end|>")"));
        },
        "duplicate token id");
  }
  SUBCASE("added token id far beyond the vocab") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(kTinyJson, R"({"id": 19, "content": "<|end|>")",
                               R"({"id": 90000000, "content": "<|end|>")"));
        },
        "out of range");
  }
  SUBCASE("malformed merge entry") {
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(kTinyJson, R"(["l", "l"])", R"(["l"])"));
        },
        "merge");
  }
  SUBCASE("duplicate merge pair names the pair") {
    // HF keeps the LAST rank for duplicates; we keep neither and fail loud.
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(
              kTinyJson,
              R"("merges": [["l", "l"], ["h", "e"], ["ll", "o"], ["he", "llo"], ["Ġ", "w"], ["o", "r"], ["l", "d"], ["or", "ld"], ["Ġw", "orld"]])",
              R"("merges": ["a b", "b c", "a b"])"));
        },
        "a b");
  }
  SUBCASE("added token single_word semantics unsupported") {
    // single_word changes word-boundary matching (the token is NOT split off
    // mid-word; transformers tokenization_utils.py:678-681). We do not emulate
    // it, so it stays a LOUD refusal rather than a silent mis-tokenization.
    // lstrip/rstrip are a different story: they ARE implemented (see the
    // "added token lstrip/rstrip eat the adjacent whitespace" case), so they
    // must NOT throw.
    CheckThrowsContains(
        [&] {
          LoadTiny(ReplaceOnce(kTinyJson, R"("single_word": false)",
                               R"("single_word": true)"));
        },
        "single_word");
  }
  SUBCASE("missing file") {
    CHECK_THROWS_AS((void)Tokenizer::FromHfJson("/nonexistent/tok.json"),
                    std::runtime_error);
  }
}

TEST_CASE("encode/decode runtime failure modes") {
  const Tokenizer tok = LoadTiny();
  // 'z' maps to symbol "z" which is not in the tiny vocab.
  CHECK_THROWS_AS((void)tok.Encode("z"), std::runtime_error);
  // Unknown / out-of-range ids.
  CHECK_THROWS_AS((void)tok.Decode({22}), std::runtime_error);
  CHECK_THROWS_AS((void)tok.Decode({-1}), std::runtime_error);
  CHECK_THROWS_AS((void)tok.Decode({1000000}), std::runtime_error);
  CHECK_THROWS_AS((void)tok.TokenText(1000000), std::runtime_error);
}

TEST_CASE("FromGguf: matches the FromHfJson-loaded tiny fixture") {
  const Tokenizer tok = LoadGguf(TinyGgufKvs());
  const Tokenizer hf = LoadTiny();
  CHECK(tok.VocabSize() == 22);
  CHECK(tok.Pattern() == SplitPattern::kQwen2);
  CHECK(tok.EosId() == 19);  // tokenizer.ggml.eos_token_id kv
  CHECK(tok.BosId() == -1);  // bos kv absent -> -1
  REQUIRE(tok.AddedTokens().size() == 3);
  CHECK(tok.AddedTokens()[0].special);        // 19: control
  CHECK_FALSE(tok.AddedTokens()[1].special);  // 20: user-defined
  CHECK(tok.AddedTokens()[2].special);        // 21: control

  // Encode/Decode must agree with the tokenizer.json-loaded equivalent.
  const std::vector<std::string> corpus = {
      "hello",      "hello world",        " world",
      "world",      "hello12",            "",
      "lll",        "llll",               "hello<|end|> world",
      "hello<tool>world",                 "hello<|end|>of world"};
  for (const auto& text : corpus) {
    CAPTURE(text);
    const Ids ids = tok.Encode(text);
    CHECK(ids == hf.Encode(text));
    CHECK(tok.Decode(ids) == text);
  }
  // Oracle ids directly, so both loaders cannot drift together unnoticed.
  CHECK(tok.Encode("hello world") == Ids{13, 17});
  CHECK(tok.Encode("hello12") == Ids{13, 8, 9});
  CHECK(tok.Encode("hello<|end|>of world") == Ids{13, 21, 17});
  CHECK(tok.TokenText(17) == "Ġworld");
  CHECK(tok.TokenText(19) == "<|end|>");
}

TEST_CASE("merge naming a token outside the vocabulary is refused AT LOAD") {
  // Mirrors HF `tokenizers` `MergeTokenOutOfVocabulary`
  // (tokenizers/src/models/bpe/model.rs:174-192). Before this rule the failure
  // arrived PER REQUEST, out of Tokenizer::EncodePlain, as `symbol "..." not in
  // vocab` — a message naming a string the checkpoint's author never wrote.
  //
  // Both load surfaces reach the same table through the same InsertMerge, and
  // both are gated here: `tokenizer.ggml.merges` is written by a CONVERTER
  // rather than copied from the original checkpoint, and HF never reads GGUF,
  // so that arm has no oracle and needs its own case.
  SUBCASE("FromHfJson: left token absent") {
    // "q" is in no vocab entry of kTinyJson.
    const std::string body =
        ReplaceOnce(kTinyJson, R"(["Ġw", "orld"]])", R"(["Ġw", "orld"], ["q", "e"]])");
    CheckThrowsContains([&] { LoadTiny(body); }, "\"q\"");
  }
  SUBCASE("FromHfJson: right token absent") {
    const std::string body =
        ReplaceOnce(kTinyJson, R"(["Ġw", "orld"]])", R"(["Ġw", "orld"], ["h", "q"]])");
    CheckThrowsContains([&] { LoadTiny(body); }, "\"q\"");
  }
  SUBCASE("FromHfJson: the CONCATENATION is absent") {
    // Both halves are in the vocab; "hw" is not. Upstream refuses this too,
    // because `new_id` has to exist for the merge to name a token.
    const std::string body =
        ReplaceOnce(kTinyJson, R"(["Ġw", "orld"]])", R"(["Ġw", "orld"], ["h", "w"]])");
    CheckThrowsContains([&] { LoadTiny(body); }, "\"hw\"");
  }
  SUBCASE("FromHfJson: the message names the merge as well as the token") {
    const std::string body =
        ReplaceOnce(kTinyJson, R"(["Ġw", "orld"]])", R"(["Ġw", "orld"], ["h", "q"]])");
    CheckThrowsContains([&] { LoadTiny(body); }, "h q");
  }
  SUBCASE("FromGguf: a merge naming an absent token is refused, with the name") {
    auto kvs = TinyGgufKvs();
    const std::vector<std::string> merges = {"l l",   "h e",   "ll o",
                                             "he llo", "Ġ w",  "o r",
                                             "l d",   "or ld", "Ġw orld",
                                             "h q"};
    kvs[4] = gguf_test::StrArrayKv("tokenizer.ggml.merges", merges);
    CheckThrowsContains([&] { LoadGguf(kvs); }, "\"q\"");
  }
  SUBCASE("FromGguf: the well-formed fixture still loads") {
    // The regression the refusal can cause, asserted rather than assumed.
    const Tokenizer tok = LoadGguf(TinyGgufKvs());
    CHECK(tok.Encode("hello world") == Ids{13, 17});
  }
  SUBCASE("every committed HF golden still loads") {
    // The stop condition of the spec: if the mirror costs us a checkpoint that
    // works today, that is a developer decision, not a quiet widening.
    for (const char* rel : {"/tokenizer_qwen36/tokenizer.json",
                            "/tokenizer_mistral/tokenizer.json",
                            "/tokenizer_deepseek_v2/tokenizer.json",
                            "/tokenizer_muse_glimmer/tokenizer.json"}) {
      CAPTURE(rel);
      const std::string path = std::string(PARITY_GOLDENS_DIR) + rel;
      const Tokenizer tok = Tokenizer::FromHfJson(path);
      CHECK(tok.VocabSize() > 0);
    }
  }
}

TEST_CASE("FromGguf: pre \"llama-bpe\" maps to kLlama3, bos kv honored") {
  auto kvs = TinyGgufKvs();
  kvs[1] = gguf_test::StrKv("tokenizer.ggml.pre", "llama-bpe");
  kvs.push_back(gguf_test::U32Kv("tokenizer.ggml.bos_token_id", 20));
  const Tokenizer tok = LoadGguf(kvs);
  CHECK(tok.Pattern() == SplitPattern::kLlama3);
  CHECK(tok.BosId() == 20);
  CHECK(tok.EosId() == 19);
}

// Issue #1924. Three GGUF pre names used to resolve to kLlama3 as a documented
// "close APPROXIMATION" while this tree already shipped, or was about to ship,
// the exact pre-tokenizer llama.cpp resolves each of them onto. An
// approximation here is not a smaller gate — it is silently wrong ids on
// ordinary text, the #347 failure again — so each name is pinned to the family
// llama.cpp maps it to (src/llama-vocab.cpp:308-325, :2162, :2170, :2351,
// :2355 @ b10451 / 10bf611e5, the pinned oracle). `laguna` stays kLlama3 and is
// pinned here too: llama.cpp has no such pre name, so nothing exact exists to
// resolve it onto, and that distinction is the point of this case.
TEST_CASE("FromGguf: DeepSeek pre names resolve to their EXACT families") {
  const auto resolve = [](const char* pre) {
    auto kvs = TinyGgufKvs();
    kvs[1] = gguf_test::StrKv("tokenizer.ggml.pre", pre);
    return LoadGguf(kvs).Pattern();
  };
  // LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM — the seven-stage DeepSeek-V2 pipeline.
  CHECK(resolve("deepseek-llm") == SplitPattern::kDeepSeek);
  // LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM — the four-stage DeepSeek-V3 pipeline.
  // "joyai-llm" is what the DeepSeek-V4-Flash GGUFs (antirez/ds4 q2-imatrix)
  // tag, and it shares llama.cpp's case with "deepseek-v3" and "hunyuan-dense".
  CHECK(resolve("deepseek-v3") == SplitPattern::kDeepSeekV3);
  CHECK(resolve("joyai-llm") == SplitPattern::kDeepSeekV3);
  CHECK(resolve("hunyuan-dense") == SplitPattern::kDeepSeekV3);
  // The two families are NOT interchangeable, which is the whole reason the
  // three names above could not keep sharing one arm.
  CHECK(resolve("deepseek-llm") != resolve("deepseek-v3"));
  // No llama.cpp pre-type exists for this one, so the approximation stands.
  CHECK(resolve("laguna") == SplitPattern::kLlama3);
}

namespace {

// TOK-GLM4-PRE (issue #2277). `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL states
// `tokenizer.ggml.pre = "glm4"` beside `tokenizer.ggml.model = "gpt2"`, read out
// of shard 1's own kv block, and `FromGguf` refused that name -- which is where
// `LoadedEngine::FromModelDir` stopped on the staged artifact.
//
// A GGUF fixture with a DISCRIMINATING vocabulary, not the shared tiny one. The
// only thing that separates kLlama3 from the two Qwen patterns on ordinary text
// is digit grouping (`\p{N}{1,3}` versus `\p{N}`), so the vocabulary carries the
// merges "1 2" -> "12" and "12 3" -> "123". Those merges can only fire when the
// three digits arrive in ONE pretoken, so the split rule becomes visible in the
// IDS rather than only in `Pattern()`: a pre name that is accepted and splits
// wrongly emits a fluent, wrong tokenization that a "it loaded" check cannot
// see, which is the #347 and #1924 failure both times.
//
// Ids 22 and 23 are `[gMASK]` and `<sop>`, the published checkpoint's own texts
// at the ids its `tokenizer.ggml.bos_token_id` and chat template point at.
std::vector<std::string> Glm4GgufKvs(const char* pre) {
  const std::vector<std::string> tokens = {
      "h",   "e",     "l",    "o",   "w",   "r",  "d",       "\xc4\xa0",
      "1",   "2",     "3",    "ll",  "he",  "llo", "hello",  "\xc4\xa0w",
      "or",  "orld",  "\xc4\xa0world",     "ld",  "12",      "123",
      "[gMASK]",      "<sop>"};
  std::vector<int32_t> types(22, 1);   // ids 0..21: normal
  types.insert(types.end(), {3, 3});   // 22, 23: control -> special added
  const std::vector<std::string> merges = {
      "l l", "h e", "ll o", "he llo", "\xc4\xa0 w",
      "o r", "l d", "or ld", "\xc4\xa0w orld", "1 2", "12 3"};
  return {
      gguf_test::StrKv("tokenizer.ggml.model", "gpt2"),
      gguf_test::StrKv("tokenizer.ggml.pre", pre),
      gguf_test::StrArrayKv("tokenizer.ggml.tokens", tokens),
      gguf_test::I32ArrayKv("tokenizer.ggml.token_type", types),
      gguf_test::StrArrayKv("tokenizer.ggml.merges", merges),
      // The staged artifact's own two ids, at this fixture's scale: 154820
      // `<|endoftext|>` and 154822 `[gMASK]`.
      gguf_test::U32Kv("tokenizer.ggml.eos_token_id", 22),
      gguf_test::U32Kv("tokenizer.ggml.bos_token_id", 22),
  };
}

// The pinned llama.cpp's two regex arms, transcribed from
// `10bf611e533d81f739128304991c5e133c6aebd8` (b10451) with the C escapes
// resolved. This CHECK records the comparison as an executable statement rather
// than as a sentence; the AUTHORITATIVE comparison is over the pinned object's
// own bytes and its sha256 is in `.agents/specs/glm5-next-flash.md` O21, because
// a transcription cannot gate what it transcribes.
constexpr const char* kUpstreamLlama3Regex =
    R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
constexpr const char* kUpstreamChatGlm4Regex =
    R"((?:'[sS]|'[tT]|'[rR][eE]|'[vV][eE]|'[mM]|'[lL][lL]|'[dD])|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

}  // namespace

TEST_CASE("FromGguf: pre \"glm4\"/\"chatglm-bpe\" resolve to the CHATGLM4 arm") {
  // LLAMA_VOCAB_PRE_TYPE_CHATGLM4's regex list (llama-vocab.cpp:396-399 @
  // b10451) against LLAMA_VOCAB_PRE_TYPE_LLAMA3's (:283-290). One expression
  // each, and the same bytes -- which is why kLlama3 is EXACT here and not the
  // approximation `laguna` above still is.
  CHECK(std::string(kUpstreamChatGlm4Regex) ==
        std::string(kUpstreamLlama3Regex));

  const auto resolve = [](const char* pre) {
    return LoadGguf(Glm4GgufKvs(pre)).Pattern();
  };
  // The two pre names llama.cpp maps to CHATGLM4 (llama-vocab.cpp:2256-2258).
  CHECK(resolve("glm4") == SplitPattern::kLlama3);
  CHECK(resolve("chatglm-bpe") == SplitPattern::kLlama3);
  // Same arm as `llama-bpe`, which is the point of the byte comparison above.
  CHECK(resolve("glm4") == resolve("llama-bpe"));
}

TEST_CASE("FromGguf: pre \"glm4\" splits digits in THREES, in the IDS") {
  const std::string text = "hello world 123";
  // kLlama3 pretokens: "hello" | " world" | " " | "123". The last one is one
  // pretoken only because `\p{N}{1,3}` groups it, so "1 2" and "12 3" fire.
  const Ids kGrouped{14, 18, 7, 21};
  // kQwen2/kQwen2Classic pretokens end "... | " " | "1" | "2" | "3"": three
  // separate pretokens, so neither digit merge can apply.
  const Ids kSingles{14, 18, 7, 8, 9, 10};

  const Tokenizer glm4 = LoadGguf(Glm4GgufKvs("glm4"));
  CHECK(glm4.Encode(text) == kGrouped);
  CHECK(LoadGguf(Glm4GgufKvs("chatglm-bpe")).Encode(text) == kGrouped);
  // Byte-identical arm, so byte-identical ids on the same input.
  CHECK(LoadGguf(Glm4GgufKvs("llama-bpe")).Encode(text) == kGrouped);
  // And it is NOT the single-digit families. Without this the case would pass
  // on any pre name that merely loads.
  CHECK(LoadGguf(Glm4GgufKvs("qwen35")).Encode(text) == kSingles);
  CHECK(LoadGguf(Glm4GgufKvs("qwen2")).Encode(text) == kSingles);
  CHECK(kGrouped != kSingles);
  // Round-trips, so the split is a SPLIT and not a dropped byte.
  CHECK(glm4.Decode(kGrouped) == text);
}

// The BOS trap, and it is not the one #2277 stated. llama.cpp's `glm4` arm sets
// `special_bos_id = LLAMA_TOKEN_NULL` (llama-vocab.cpp:2259), but that is a
// DEFAULT: the loop at :2559-2578 reads `tokenizer.ggml.bos_token_id` out of the
// file and assigns it OVER the null whenever the key is present and in range.
// So llama.cpp does not discard the id -- the staged artifact's 154822 survives
// that load. What it declines to do is PREPEND it: the prepend at :3382-3384
// tests `add_bos`, which defaults false (:1815) and which the `glm4` arm does
// NOT set, unlike the `llama-bpe` arm three cases up (:2159). The staged file
// carries no `tokenizer.ggml.add_bos_token` key at all -- 72 kvs, measured --
// so `add_bos` stays false there.
//
// On this checkpoint that is load-bearing rather than academic: id 154822 is
// `[gMASK]`, and the file's own `tokenizer.chat_template` opens with the LITERAL
// text `[gMASK]<sop>`. A tokenizer that also prepended the id would emit it
// twice on every request -- one extra token per prompt, which a shape check, a
// load check and a "does it generate" check all pass.
TEST_CASE("FromGguf: pre \"glm4\" reads the BOS id and does NOT prepend it") {
  const Tokenizer tok = LoadGguf(Glm4GgufKvs("glm4"));
  // The file's own id, honored: llama-vocab.cpp:2577 over :2259.
  CHECK(tok.BosId() == 22);
  CHECK(tok.TokenText(22) == "[gMASK]");
  // And nothing prepends it. `EncodeWithSpecialTokens` is the prompt path's
  // add_special_tokens=True; `template_bos_` is left -1 by FromGguf, which is
  // what mirrors `add_bos == false`.
  const std::string text = "hello world 123";
  const Ids plain = tok.Encode(text);
  CHECK(tok.EncodeWithSpecialTokens(text) == plain);
  CHECK(tok.EncodeWithSpecialTokens(text).front() != tok.BosId());
  CHECK(tok.EncodeWithSpecialTokens(text).back() != tok.EosId());
}

// The table did not become permissive. Every name here is one a reader could
// plausibly expect to work now, and each is still refused BY NAME.
TEST_CASE("FromGguf: an unknown pre name is still refused, and named") {
  for (const char* name : {"glm5", "glm4v", "chatglm", "chatglm-bpe4", "glm",
                           "glm5next", "GLM4", "glm4 "}) {
    const std::string pre(name);
    CAPTURE(pre);
    CheckThrowsContains([&] { LoadGguf(Glm4GgufKvs(name)); },
                        "unsupported tokenizer.ggml.pre");
    CheckThrowsContains([&] { LoadGguf(Glm4GgufKvs(name)); }, pre);
  }
}

TEST_CASE(
    "FromGguf: token_type unknown(2), unused(5), byte(6) stay normal vocab") {
  auto kvs = TinyGgufKvs();
  std::vector<int32_t> types(19, 1);
  types[0] = 6;  // "h" tagged byte
  types[1] = 2;  // "e" tagged unknown
  types[4] = 5;  // "w" tagged unused (PAD-row padding in real GGUFs)
  types.insert(types.end(), {3, 4, 3});
  kvs[3] = gguf_test::I32ArrayKv("tokenizer.ggml.token_type", types);
  const Tokenizer tok = LoadGguf(kvs);
  CHECK(tok.Encode("hello") == Ids{13});  // h/e still merge via BPE
  CHECK(tok.AddedTokens().size() == 3);   // none became added tokens
  CHECK(tok.TokenText(4) == "w");         // unused entry loads as plain vocab
}

TEST_CASE("FromGguf failure modes throw with actionable messages") {
  SUBCASE("missing tokenizer.ggml.model") {
    auto kvs = TinyGgufKvs();
    kvs.erase(kvs.begin());
    CheckThrowsContains([&] { LoadGguf(kvs); }, "tokenizer.ggml.model");
  }
  SUBCASE("non-gpt2 model names the value") {
    auto kvs = TinyGgufKvs();
    kvs[0] = gguf_test::StrKv("tokenizer.ggml.model", "llama");
    CheckThrowsContains([&] { LoadGguf(kvs); }, "llama");
  }
  SUBCASE("missing tokenizer.ggml.pre") {
    auto kvs = TinyGgufKvs();
    kvs.erase(kvs.begin() + 1);
    CheckThrowsContains([&] { LoadGguf(kvs); }, "tokenizer.ggml.pre");
  }
  SUBCASE("unknown pre names the value") {
    auto kvs = TinyGgufKvs();
    kvs[1] = gguf_test::StrKv("tokenizer.ggml.pre", "gpt-2");
    CheckThrowsContains([&] { LoadGguf(kvs); }, "gpt-2");
  }
  SUBCASE("classic \"qwen2\" pre resolves to kQwen2Classic") {
    auto kvs = TinyGgufKvs();
    kvs[1] = gguf_test::StrKv("tokenizer.ggml.pre", "qwen2");
    const Tokenizer tok = LoadGguf(kvs);
    CHECK(tok.Pattern() == SplitPattern::kQwen2Classic);
  }
  SUBCASE("missing tokens array") {
    auto kvs = TinyGgufKvs();
    kvs.erase(kvs.begin() + 2);
    CheckThrowsContains([&] { LoadGguf(kvs); }, "tokenizer.ggml.tokens");
  }
  SUBCASE("tokens kv of the wrong type") {
    auto kvs = TinyGgufKvs();
    kvs[2] = gguf_test::StrKv("tokenizer.ggml.tokens", "x");
    CheckThrowsContains([&] { LoadGguf(kvs); }, "array");
  }
  SUBCASE("token_type length mismatch") {
    auto kvs = TinyGgufKvs();
    std::vector<int32_t> types(21, 1);  // 21 != 22 tokens
    kvs[3] = gguf_test::I32ArrayKv("tokenizer.ggml.token_type", types);
    CheckThrowsContains([&] { LoadGguf(kvs); }, "token_type");
  }
  SUBCASE("unsupported token_type value names it") {
    auto kvs = TinyGgufKvs();
    std::vector<int32_t> types(19, 1);
    types[0] = 7;  // not one of 1/2/3/4/5/6
    types.insert(types.end(), {3, 4, 3});
    kvs[3] = gguf_test::I32ArrayKv("tokenizer.ggml.token_type", types);
    CheckThrowsContains([&] { LoadGguf(kvs); }, "7");
  }
  SUBCASE("merge entry without exactly one space") {
    auto kvs = TinyGgufKvs();
    kvs[4] = gguf_test::StrArrayKv("tokenizer.ggml.merges", {"l l", "he"});
    CheckThrowsContains([&] { LoadGguf(kvs); }, "merge");
    kvs[4] = gguf_test::StrArrayKv("tokenizer.ggml.merges", {"l l o"});
    CheckThrowsContains([&] { LoadGguf(kvs); }, "merge");
  }
  SUBCASE("duplicate merge pair") {
    auto kvs = TinyGgufKvs();
    kvs[4] = gguf_test::StrArrayKv("tokenizer.ggml.merges",
                                   {"l l", "h e", "l l"});
    CheckThrowsContains([&] { LoadGguf(kvs); }, "duplicate merge");
  }
  SUBCASE("duplicate token text") {
    auto kvs = TinyGgufKvs();
    auto tokens = std::vector<std::string>{"h", "e", "h"};
    kvs[2] = gguf_test::StrArrayKv("tokenizer.ggml.tokens", tokens);
    kvs[3] = gguf_test::I32ArrayKv("tokenizer.ggml.token_type", {1, 1, 1});
    kvs[5] = gguf_test::U32Kv("tokenizer.ggml.eos_token_id", 0);
    CheckThrowsContains([&] { LoadGguf(kvs); }, "duplicate token text");
  }
  SUBCASE("eos id out of range") {
    auto kvs = TinyGgufKvs();
    kvs[5] = gguf_test::U32Kv("tokenizer.ggml.eos_token_id", 22);
    CheckThrowsContains([&] { LoadGguf(kvs); }, "out of range");
  }
}

// ─── tokenizer_config.json is NOT a special-token source (MLA campaign W8) ───
//
// This test exists to STOP a plausible-looking "fix" that measurement refuted.
//
// DeepSeek-V2-Lite's `tokenizer_config.json` says `tokenizer_class:
// LlamaTokenizerFast` and **`add_bos_token: true`** with `bos_token`
// `<|begin of sentence|>` (id 100000), while its tokenizer.json post_processor
// is a plain ByteLevel declaring no special tokens. Reading only those files it
// looks certain that a prompt must be BOS-prefixed, and that our loader — which
// takes special tokens ONLY from the post_processor — is one token short: the
// exact silent, correctness-fatal failure OPT hit from the other direction
// (post_processor BOS parsed but never applied, 0/6 prompts on fluent English).
//
// It is not. MEASURED on dgx against the pinned vLLM 0.25.0 oracle's own
// tokenizer (2026-07-22):
//   AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V2-Lite",
//                                 trust_remote_code=True)
//     class:                     TokenizersBackend
//     add_bos_token (resolved):  False        <-- NOT the config.json value
//     encode("The capital of France is", add_special_tokens=True)
//                             -> [549, 6077, 280, 7239, 317]   (NO BOS)
// The class HF actually instantiates is driven by tokenizer.json and does not
// honour tokenizer_config.json's `add_bos_token`, so vLLM's `prompt_token_ids`
// carry no BOS — as the committed `p{i}_prompt.i32` goldens of the DeepSeek-V2
// gate independently show. Our loader already matches, bit for bit, and a
// "helpful" tokenizer_config.json reader would have BROKEN a passing gate.
//
// So: the post_processor stays the ONLY special-token source, and this test
// pins that, so a future reader of tokenizer_config.json has to come here and
// re-measure first ([[ground-premises-before-dispatching]]).
namespace {

// A temp DIRECTORY holding tokenizer.json plus (optionally) its sibling
// tokenizer_config.json. A directory rather than the shared temp dir on
// purpose: a stray tokenizer_config.json next to every other fixture would
// silently change what those tests load.
class TempTokenizerDir {
 public:
  TempTokenizerDir(const std::string& tokenizer_json,
                   const std::string& tokenizer_config_json) {
    static int counter = 0;
    dir_ = std::filesystem::temp_directory_path() /
           ("vllm_tokcfg_test_" + std::to_string(counter++));
    std::filesystem::create_directories(dir_);
    std::ofstream(tokenizer_path(), std::ios::binary) << tokenizer_json;
    if (!tokenizer_config_json.empty()) {
      std::ofstream(dir_ / "tokenizer_config.json", std::ios::binary)
          << tokenizer_config_json;
    }
  }
  ~TempTokenizerDir() {
    std::error_code ec;
    std::filesystem::remove_all(dir_, ec);
  }
  std::string tokenizer_path() const {
    return (dir_ / "tokenizer.json").string();
  }
  const std::filesystem::path& dir() const { return dir_; }


 private:
  std::filesystem::path dir_;
};

// Holds the process working directory inside `new_cwd` for the enclosing
// scope: FromHfJson("tokenizer.json") resolves a BARE filename against the
// CWD, and its sibling lookup must follow the same resolution (PR #3363).
// Declared AFTER the TempTokenizerDir it scopes so destruction order restores
// the CWD before the directory is removed.
class ScopedCwd {
 public:
  explicit ScopedCwd(const std::filesystem::path& new_cwd)
      : saved_(std::filesystem::current_path()) {
    std::filesystem::current_path(new_cwd);
  }
  ~ScopedCwd() { std::filesystem::current_path(saved_); }

 private:
  std::filesystem::path saved_;
};

// NOTE: there is deliberately no `kTinySpecialId` constant here. It named
// kTinyJson's added token id 19 ("<|end|>") as a stand-in BOS/EOS, but no case
// below ever referenced it — the tests that use that id spell it literally
// (see the Encode/Decode cases above). Clang's -Wunused-const-variable flagged
// it while building on macOS (BACKEND-METAL-MLX W0); dead code, so it was
// deleted rather than suppressed.

}  // namespace

TEST_CASE("tokenizer_config.json add_bos_token is NOT applied, but the "
          "bos/eos NAMES resolve (DeepSeek-V2 shape)") {
  SUBCASE("the exact DeepSeek-V2 shape adds no BOS but the name resolves") {
    // Byte-for-byte the DeepSeek-V2-Lite situation: an AddedToken OBJECT for
    // bos_token, `add_bos_token: true`, `tokenizer_class: LlamaTokenizerFast`,
    // and a tokenizer.json whose post_processor is a plain ByteLevel. HF's
    // resolved tokenizer adds NOTHING to an encode here (measured — see the
    // block comment above), so neither may we.
    //
    // What CHANGED (ISSUE-LOCAL-01M3RY20H90NMK37V74EH34Z77): `add_bos_token`
    // remains ignored, but the NAME resolves. HF's resolved
    // `tokenizer.bos_token_id` is the config-named token's id even when
    // `add_bos_token` resolves False — measured 2026-09-30 with transformers
    // 5.17.0 on a checkpoint whose tokenizer_config.json declares bos_token as
    // an AddedToken object (bos_token_id=248046 while add_bos_token=False and
    // no BOS is prepended). BosId() naming that id is HF-faithful AND what
    // lets InputProcessor stop on it; encoding is still the post_processor's
    // alone.
    const TempTokenizerDir d(kTinyJson, R"json({
      "tokenizer_class": "LlamaTokenizerFast",
      "add_bos_token": true,
      "add_eos_token": false,
      "bos_token": {"__type": "AddedToken", "content": "<|end|>", "normalized": true}
    })json");
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.BosId() == 19);  // the config-NAMED id; HF resolves it too
    CHECK(t.EncodeWithSpecialTokens("hello world") == t.Encode("hello world"));
  }

  SUBCASE("add_eos_token=true likewise appends nothing") {
    const TempTokenizerDir d(kTinyJson, R"json({
      "add_bos_token": false, "add_eos_token": true, "eos_token": "<|end|>"
    })json");
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.EosId() == 19);  // name resolves even though nothing is appended
    CHECK(t.EncodeWithSpecialTokens("hello") == t.Encode("hello"));
  }

  SUBCASE("no tokenizer_config.json at all is likewise a no-op") {
    const TempTokenizerDir d(kTinyJson, "");
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.EncodeWithSpecialTokens("hello world") == t.Encode("hello world"));
  }

  SUBCASE("the post_processor REMAINS the sole special-token source (OPT)") {
    // OPT's shape: a TemplateProcessing post_processor declaring the BOS, plus
    // an add_bos_token in the config naming a DIFFERENT token. The
    // post_processor is what applies — this is what keeps OPT's committed
    // 6/6 STRICT gate where it is.
    const std::string tpl_json = ReplaceOnce(
        kTinyJson,
        R"("post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})",
        R"("post_processor": {"type": "TemplateProcessing",
            "single": [{"SpecialToken": {"id": "<tool>", "type_id": 0}},
                       {"Sequence": {"id": "A", "type_id": 0}}],
            "pair": [], "special_tokens": {"<tool>": {"id": "<tool>", "ids": [20], "tokens": ["<tool>"]}}})");
    const TempTokenizerDir d(tpl_json, R"json({
      "add_bos_token": true, "bos_token": "<|end|>"
    })json");
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.BosId() == 20);  // the post_processor's token; 19 is never used
    CHECK(t.EncodeWithSpecialTokens("hello world")[0] == 20);
  }
}

TEST_CASE("tokenizer_config.json eos_token/bos_token NAMES resolve when the "
          "post_processor declares none (ISSUE-LOCAL-01M3RY20H90NMK37V74EH34Z77)") {
  // The Qwen3.5-9B-EXL3 shape: a ByteLevel post_processor (no TemplateProcessing
  // ids) plus tokenizer_config.json naming eos_token. HF resolves
  // eos_token_id from that name -- AutoTokenizer on the real checkpoint
  // returns 248046 for "<|im_end|>" (measured 2026-09-30, transformers
  // 5.17.0) -- so InputProcessor's tokenizer fallback must see it too.
  // Without this the request carries NO eos id and <|im_end|> is generated
  // then ignored.
  SUBCASE("string-form eos_token resolves to the added-token id") {
    const TempTokenizerDir d(kTinyJson, R"json({
      "eos_token": "<|end|>",
      "bos_token": "<tool>"
    })json");
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.EosId() == 19);
    CHECK(t.BosId() == 20);
    // Encoding is untouched: the names set the reported ids only, and the
    // post_processor still owns what gets added to a prompt.
    CHECK(t.EncodeWithSpecialTokens("hello world") ==
          t.Encode("hello world"));
  }

  SUBCASE("AddedToken object form resolves identically") {
    const TempTokenizerDir d(kTinyJson, R"json({
      "eos_token": {"__type": "AddedToken", "content": "<|end|>",
                    "lstrip": false, "normalized": false, "rstrip": false,
                    "single_word": false, "special": true}
    })json");
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.EosId() == 19);
  }

  SUBCASE("special_tokens_map.json names resolve too") {
    const TempTokenizerDir d(kTinyJson, R"json({})json");
    // TempTokenizerDir only writes tokenizer_config.json; drop the map file
    // beside it manually for the older-file arm.
    std::filesystem::path dir =
        std::filesystem::path(d.tokenizer_path()).parent_path();
    std::ofstream(dir / "special_tokens_map.json", std::ios::binary)
        << R"json({"eos_token": "<|end|>"})json";
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.EosId() == 19);
  }

  SUBCASE("tokenizer_config.json overrides special_tokens_map.json") {
    // HF from_pretrained applies the config file's kwargs AFTER the map file
    // has been consumed, so the config name must win when both declare eos.
    // Map says <|missing|> (unresolvable); config says <|end|> (19).
    const TempTokenizerDir d(kTinyJson, R"json({"eos_token": "<|end|>"})json");
    std::filesystem::path dir =
        std::filesystem::path(d.tokenizer_path()).parent_path();
    std::ofstream(dir / "special_tokens_map.json", std::ios::binary)
        << R"json({"eos_token": "<|missing|>"})json";
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.EosId() == 19);
  }

  SUBCASE("a post_processor id keeps precedence over the config name") {
    // Same OPT shape as the pin above, mirrored on eos: TemplateProcessing
    // says <tool> (20), the config says <|end|> (19); the post_processor wins.
    const std::string tpl_json = ReplaceOnce(
        kTinyJson,
        R"("post_processor": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false})",
        R"("post_processor": {"type": "TemplateProcessing",
            "single": [{"Sequence": {"id": "A", "type_id": 0}},
                       {"SpecialToken": {"id": "<tool>", "type_id": 0}}],
            "pair": [], "special_tokens": {"<tool>": {"id": "<tool>", "ids": [20], "tokens": ["<tool>"]}}})");
    const TempTokenizerDir d(tpl_json, R"json({"eos_token": "<|end|>"})json");
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.EosId() == 20);
    CHECK(t.BosId() == -1);  // template's front is Sequence A -> -1, and the
                             // config names no bos_token, so it stays unset
  }

  SUBCASE("an unresolvable name leaves the id unset") {
    const TempTokenizerDir d(kTinyJson, R"json({"eos_token": "<|missing|>"})json");
    const Tokenizer t = Tokenizer::FromHfJson(d.tokenizer_path());
    CHECK(t.EosId() == -1);
  }

  SUBCASE("a bare filename resolves siblings from the CWD too (PR #3363)") {
    // parent_path() of "tokenizer.json" is empty; the sibling join must still
    // find tokenizer_config.json in the CWD. The unfixed `!dir.empty()` guard
    // skipped the lookup on this spelling and left EosId() at -1 while
    // "./tokenizer.json" resolved 19.
    const TempTokenizerDir d(kTinyJson, R"json({"eos_token": "<|end|>"})json");
    const ScopedCwd cwd(d.dir());
    const Tokenizer bare = Tokenizer::FromHfJson("tokenizer.json");
    CHECK(bare.EosId() == 19);
    const Tokenizer dotted = Tokenizer::FromHfJson("./tokenizer.json");
    CHECK(dotted.EosId() == 19);
  }
}
