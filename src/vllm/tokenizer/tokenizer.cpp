// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE. FromHfJson loads an HF tokenizer.json; FromGguf loads the same vocab
// family from GGUF tokenizer.ggml.* kvs. Anything that is not the byte-level
// BPE family we implement throws loudly (no silent wrong tokenization).
#include "vllm/tokenizer/tokenizer.h"

#include <cstddef>
#include <cstdio>
#include <fstream>
#include <stdexcept>
#include <utility>

#include <nlohmann/json.hpp>

#include "vllm/model_executor/model_loader/gguf_reader.h"
#include "vllm/tokenizer/unicode_data.h"

namespace vllm::tok {
namespace {

using nlohmann::json;

// Split regexes recorded verbatim (decoded form) — the escaped originals and
// their provenance live in src/vllm/tokenizer/pretokenizer.cpp.
constexpr const char* kQwen36Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
constexpr const char* kClassicQwen2Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// Aleph-Alpha/Kolibri-1 (MODEL-TEXT-kolibri-1 R7, tokenizer.json read
// 2026-10-08). Byte-identical to kClassicQwen2Regex except ONE insert: the
// number alternative is written \p{N}{1} where the constant writes \p{N}.
// {1} is the identity quantifier, so the two patterns match the SAME
// language; the only reason this is a separate constant is that recognition
// here is by whole-pattern byte equality, never by a substring a sibling
// shares. It maps onto SplitPattern::kQwen2Classic below, whose scanner
// implements single-codepoint \p{N} grouping — exactly what \p{N}{1} means.
constexpr const char* kClassicQwen2N1Regex =
    R"((?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// Mistral Tekken (mistralai/Mistral-Nemo-Instruct-2407 and the other Tekken
// checkpoints). Byte-equal to tiktoken's o200k_base pat_str except that the
// optional (?i:'s|'t|...) group is absent from both letter alternatives and
// numbers are \p{N} rather than \p{N}{1,3}.
constexpr const char* kTekkenRegex =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";
// GPT-4o / o200k (the "llama4" GGUF pre name). o200k_base pat_str VERBATIM --
// i.e. kTekkenRegex with the two edits above put back: the contraction group
// is present as a SUFFIX of both letter alternatives, and numbers are
// \p{N}{1,3}. That last one is why this MUST be matched exactly and BEFORE the
// `\p{N}{1,3}` heuristic below -- see the comment at that heuristic.
constexpr const char* kGpt4oRegex =
    R"([^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n/]*|\s*[\r\n]+|\s+(?!\S)|\s+)";

[[noreturn]] void Fail(const std::string& msg) {
  throw std::runtime_error("tokenizer: " + msg);
}

// The SentencePiece unk sentinel. `Tokenizer::EncodePlainSp` pushes it as a
// symbol for a character with no vocabulary token and no byte fallback, so it
// travels through `BpeMerge` and must never be merged. Upstream keeps the same
// split -- `merge_word` builds the Word, including its unk handling, before
// `merge_all` runs (`tokenizers/src/models/bpe/model.rs:382-460`).
//
// `Tokenizer::FinalizeTables` asserts at LOAD that no merge names it, which is
// the reserved-identifier guarantee `.agents/specs/bpe-quadratic-merge.md`
// §Risks asks for: `MergeRanks::Find` then answers `kNoSymbol` for it on every
// checkpoint we accept, and no merge can apply to a `kNoSymbol` symbol.
const std::string& UnkSentinel() {
  static const std::string kUnk("\x01\x01unk\x01\x01");
  return kUnk;
}

// Splits a legacy-form merge entry "left right" (exactly one space, both
// halves nonempty); used by both the HF and GGUF loaders.
void SplitMergeEntry(const std::string& s, std::string& left,
                     std::string& right) {
  const size_t sp = s.find(' ');
  if (sp == std::string::npos || sp != s.rfind(' ') || sp == 0 ||
      sp + 1 == s.size()) {
    Fail("malformed merge entry \"" + s + "\"");
  }
  left = s.substr(0, sp);
  right = s.substr(sp + 1);
}

// Inserts one merge pair at `rank`; duplicates fail loud (HF silently keeps
// the LAST rank for duplicates; we keep neither).
//
// The VOCABULARY RULE mirrors HF `tokenizers`
// (`tokenizers/src/models/bpe/model.rs:174-192`): a merge whose left token,
// right token, or concatenation is absent from the vocabulary is
// `MergeTokenOutOfVocabulary` AT LOAD. We had no such rule, so a merge could
// produce a symbol that `Tokenizer::EncodePlain` then failed on PER REQUEST,
// with a message about a symbol nobody wrote. Moving the failure to load is
// where upstream puts it and where a user can act on it.
//
// This is a behaviour change on malformed inputs, and it is deliberate: a
// checkpoint that loads today and fails on some prompts is refused at load
// instead. All seven `tokenizer.json` files committed under `tests/` carry zero
// offending merges, so it costs nothing that this tree can load. The GGUF arm
// is the one with no oracle -- HF never reads GGUF, and
// `tokenizer.ggml.merges` is written by a CONVERTER rather than copied -- which
// is why the message names the missing token.
//
// `merged` is `left + right` whole. Upstream trims
// `continuing_subword_prefix` off the right half (`model.rs:169-173,186`), and
// `FromHfJson` below already refuses a non-empty `continuing_subword_prefix`,
// so that term is 0 on everything we accept.
void InsertMerge(MergeRanks& ranks,
                 const std::unordered_map<std::string, int32_t>& vocab,
                 const std::string& left, const std::string& right,
                 int32_t rank) {
  const std::string merged = left + right;
  for (const std::string* token : {&left, &right, &merged}) {
    if (vocab.find(*token) == vocab.end()) {
      Fail("merge token \"" + *token + "\" at merge rank " +
           std::to_string(rank) + " (\"" + left + " " + right +
           "\") is not in the vocabulary");
    }
  }
  if (!ranks.Insert(left, right, rank)) {
    Fail("duplicate merge pair \"" + left + " " + right + "\" at rank " +
         std::to_string(rank));
  }
}

// Walks a pre_tokenizer node collecting Split regexes and checking that
// every component is one we can emulate.
// The DeepSeek family's pre_tokenizer is a `Sequence` PIPELINE, not a single
// alternation regex (see src/vllm/tokenizer/pretokenizer.cpp for the full
// semantics). Its five Split stages carry EXPLICIT enumerated codepoint classes
// rather than `\p{...}` properties, so the classes we compiled into
// SplitPattern::kDeepSeek are only valid for a checkpoint that ships exactly
// these patterns. They are therefore compared VERBATIM — a DeepSeek variant with
// different ranges must fail loudly, not tokenize subtly wrong.
//
// Transcribed from deepseek-ai/DeepSeek-V2-Lite tokenizer.json (snapshot
// 604d5664dddd88a0433dbae533b7fe9472482de0, read 2026-07-22).
// Written with explicit \u / \U escapes rather than literal UTF-8: several of
// these codepoints have visually IDENTICAL lookalikes in other Unicode blocks
// (a first transcription silently picked up U+03CE where the checkpoint has
// U+1F7D — same glyph, different character, and the pattern comparison below
// then rejected the real DeepSeek tokenizer). Escapes make the bytes auditable.
constexpr const char* kDsNewlineRegex = "[\r\n]";
constexpr const char* kDsLettersRegex =
    "\\s?[A-Za-z\u00B5\u00C0-\u00D6\u00D8-\u00F6\u00F8-\u01BA\u01BC-"
    "\u01BF\u01C4-\u0293\u0295-\u02AF\u0370-\u0373\u0376\u0377\u037B-"
    "\u037D\u037F\u0386\u0388-\u038A\u038C\u038E-\u03A1\u03A3-\u03F5"
    "\u03F7-\u0481\u048A-\u052F\u0531-\u0556\u10A0-\u10C5\u13A0-\u13F5"
    "\u13F8-\u13FD\u1C90-\u1CBA\u1CBD-\u1CBF\u1D00-\u1D2B\u1D6B-\u1D77"
    "\u1D79-\u1D9A\u1E00-\u1F15\u1F18-\u1F1D\u1F20-\u1F45\u1F48-\u1F4D"
    "\u1F50-\u1F57\u1F59\u1F5B\u1F5D\u1F5F-\u1F7D\u1F80-\u1FB4\u1FB6-"
    "\u1FBC\u1FBE\u1FC2-\u1FC4\u1FC6-\u1FCC\u1FD0-\u1FD3\u1FD6-\u1FDB"
    "\u1FE0-\u1FEC\u1FF2-\u1FF4\u1FF6-\u1FFC\u2102\u2107\u210A-\u2113"
    "\u2115\u2119-\u211D\u2124\u2126\u2128\u212A-\u212D\u212F-\u2134"
    "\u2139\u213C-\u213F\u2145-\u2149\u214E\u2183\u2184\u2C00-\u2C7B"
    "\u2C7E-\u2CE4\u2CEB-\u2CEE\u2CF2\u2CF3\uA640-\uA66D\uA680-\uA69B"
    "\uA722-\uA76F\uA771-\uA787\uA78B-\uA78E\uAB70-\uABBF\uFB00-\uFB06"
    "\uFB13-\uFB17\uFF21-\uFF3A\uFF41-\uFF5A\U00010400-\U0001044F"
    "\U000104B0-\U000104D3\U000104D8-\U000104FB\U00010C80-\U00010CB2"
    "\U00010CC0-\U00010CF2\U000118A0-\U000118DF\U0001E900-\U0001E943]+";
constexpr const char* kDsPunctRegex =
    "\\s?[!-/:-~\uFF01-\uFF0F\uFF1A-\uFF5E\u2018-\u201F\u3000-\u3002]+";
constexpr const char* kDsTrailWsRegex = "\\s+$";
constexpr const char* kDsCjkRegex = "[\u4E00-\u9FA5\u0800-\u4E00\uAC00-\uD7FF]+";

// The DeepSeek-V3 family (DeepSeek-V3, DeepSeek-R1, DeepSeek-V4-Flash) ships a
// DIFFERENT `Sequence`: FOUR stages, three of them `Split`, and it shares not
// one pattern with the five above. llama.cpp keeps the two as separate cases of
// one switch -- LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM and
// LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM, src/llama-vocab.cpp:308-325 @ b10451
// (10bf611e5, the pinned llama.cpp oracle) -- and so does this file.
//
// Transcribed VERBATIM from the DeepSeek-V4-Flash checkpoint's tokenizer.json
// (sha256 8f9f37ca…33cf; the exact bytes are committed at
// tests/parity/goldens/tokenizer_deepseek_v3/tokenizer.json), and compared
// byte-for-byte below, for the same reason the V2 patterns are: a DeepSeek
// variant that ships different classes must fail loudly rather than tokenize
// subtly wrong. The transcription is SELF-CHECKING -- if any byte here is
// wrong the recognizer does not fire on the real file and
// test_tokenizer_parity_deepseek_v3 reds on the very first case.
//
// The CJK class is written with \u escapes for the same lookalike reason the V2
// classes are: U+3040 is UNASSIGNED and U+30A0 is a hyphen, so a literal-UTF-8
// spelling of that class would be three invisible or misleading glyphs in a
// row. Note that the WORD regex carries LITERAL CR/LF BYTES inside its
// `[\r\n]` classes where the Qwen checkpoints write the two-character escapes;
// `\r`/`\n` in a C++ literal produce exactly those bytes, so it compares equal
// without the canonicalization DetectPattern applies on the single-Split path.
constexpr const char* kDsV3DigitsRegex = "\\p{N}{1,3}";
constexpr const char* kDsV3CjkRegex =
    "[\u4E00-\u9FA5\u3040-\u309F\u30A0-\u30FF]+";
constexpr const char* kDsV3WordRegex =
    "[!\"#$%&'()*+,\\-./:;<=>?@\\[\\\\\\]^_`{|}~][A-Za-z]+"
    "|[^\r\n\\p{L}\\p{P}\\p{S}]?[\\p{L}\\p{M}]+"
    "| ?[\\p{P}\\p{S}]+[\r\n]*"
    "|\\s*[\r\n]+"
    "|\\s+(?!\\S)"
    "|\\s+";

// Recognizes the DeepSeek-V2 `Sequence` pre_tokenizer EXACTLY: five Splits with
// the verbatim kDsNewline/kDsLetters/kDsPunct/kDsTrailWs/kDsCjk patterns above
// (all Isolated, non-inverted), then Digits(individual_digits=true), then
// ByteLevel(add_prefix_space=false, use_regex=false). Returns false (not
// "fail") for anything else, so the ordinary single-Split path still gets its
// own diagnostics. The V3 sibling is IsDeepSeekV3PreTokenizer below.
bool IsDeepSeekPreTokenizer(const json& node) {
  if (!node.is_object() || node.value("type", "") != "Sequence") return false;
  const auto it = node.find("pretokenizers");
  if (it == node.end() || !it->is_array() || it->size() != 7) return false;
  const char* want[5] = {kDsNewlineRegex, kDsLettersRegex, kDsPunctRegex,
                         kDsTrailWsRegex, kDsCjkRegex};
  for (int i = 0; i < 5; ++i) {
    const json& s = (*it)[static_cast<size_t>(i)];
    if (!s.is_object() || s.value("type", "") != "Split") return false;
    if (s.value("behavior", "") != "Isolated" || s.value("invert", false)) {
      return false;
    }
    const auto pat = s.find("pattern");
    if (pat == s.end() || !pat->is_object() || !pat->contains("Regex") ||
        !(*pat)["Regex"].is_string()) {
      return false;
    }
    if ((*pat)["Regex"].get<std::string>() != want[i]) return false;
  }
  const json& digits = (*it)[5];
  if (!digits.is_object() || digits.value("type", "") != "Digits" ||
      !digits.value("individual_digits", false)) {
    return false;
  }
  const json& bl = (*it)[6];
  return bl.is_object() && bl.value("type", "") == "ByteLevel" &&
         !bl.value("add_prefix_space", true) && !bl.value("use_regex", false);
}

// Recognizes the DeepSeek-V3 `Sequence` pre_tokenizer EXACTLY: three Splits
// with the verbatim patterns above (all Isolated, non-inverted), then
// ByteLevel(add_prefix_space=false, use_regex=false). Returns false (not
// "fail") for anything else, exactly like its V2 sibling, so the ordinary
// single-Split path still gets its own diagnostics.
//
// WHY IT CANNOT BE THE GENERIC WALK. The walk collects Split regexes into a
// flat list and then insists on exactly ONE, because for a single full-cover
// alternation the pattern IS the whole splitting rule. Here there are three,
// and they COMPOSE: each stage re-splits the pieces the previous one produced,
// so the list order is semantics rather than an accident of the file. A walk
// that concatenated or alternated them would be a different function
// (`\p{N}{1,3}` alternated with the third pattern never groups digits before
// the word rule sees them). See src/vllm/tokenizer/pretokenizer.cpp.
bool IsDeepSeekV3PreTokenizer(const json& node) {
  if (!node.is_object() || node.value("type", "") != "Sequence") return false;
  const auto it = node.find("pretokenizers");
  if (it == node.end() || !it->is_array() || it->size() != 4) return false;
  const char* want[3] = {kDsV3DigitsRegex, kDsV3CjkRegex, kDsV3WordRegex};
  for (int i = 0; i < 3; ++i) {
    const json& s = (*it)[static_cast<size_t>(i)];
    if (!s.is_object() || s.value("type", "") != "Split") return false;
    if (s.value("behavior", "") != "Isolated" || s.value("invert", false)) {
      return false;
    }
    const auto pat = s.find("pattern");
    if (pat == s.end() || !pat->is_object() || !pat->contains("Regex") ||
        !(*pat)["Regex"].is_string()) {
      return false;
    }
    if ((*pat)["Regex"].get<std::string>() != want[i]) return false;
  }
  const json& bl = (*it)[3];
  return bl.is_object() && bl.value("type", "") == "ByteLevel" &&
         !bl.value("add_prefix_space", true) && !bl.value("use_regex", false);
}

void WalkPreTokenizer(const json& node, std::vector<std::string>& regexes,
                      bool& saw_byte_level, bool& byte_level_use_regex) {
  if (!node.is_object()) Fail("pre_tokenizer entry is not an object");
  const std::string type = node.value("type", "");
  if (type == "Sequence") {
    const auto it = node.find("pretokenizers");
    if (it == node.end() || !it->is_array()) {
      Fail("pre_tokenizer Sequence without \"pretokenizers\" array");
    }
    for (const auto& sub : *it)
      WalkPreTokenizer(sub, regexes, saw_byte_level, byte_level_use_regex);
    return;
  }
  if (type == "Split") {
    // Two equivalent encodings of "keep each regex match as its own pre-token"
    // over a FULLY-COVERING pattern (the GPT-4/cl100k regex every byte-level BPE
    // family here uses):
    //   - behavior=Isolated, invert=false  — Qwen/Llama-3/GLM/DeepSeek express it
    //     this way (the pattern is the delimiter, Isolated keeps it as a piece);
    //   - behavior=Removed,  invert=true   — OLMo-2 (GPT-NeoX/tiktoken-derived)
    //     expresses it this way (invert makes the pattern's matches the pieces,
    //     Removed drops the — inverted, i.e. non-matched — gaps).
    // For a full-cover pattern the two produce the IDENTICAL piece sequence, so
    // the inverted form is accepted and pushes the SAME regex (mapped to kLlama3
    // by DetectPattern via the shared `\p{N}{1,3}` marker). Verified by round-trip
    // vs the HF tokenizer on the OLMo-2 prompt battery. Additive: no existing
    // checkpoint uses the inverted form, so their tokenization is unchanged.
    const std::string behavior = node.value("behavior", "");
    const bool invert = node.value("invert", false);
    const bool isolated = behavior == "Isolated" && !invert;
    const bool removed_inverted = behavior == "Removed" && invert;
    if (!isolated && !removed_inverted) {
      Fail("Split pre-tokenizer must be behavior=Isolated,invert=false or "
           "behavior=Removed,invert=true");
    }
    const auto pat = node.find("pattern");
    if (pat == node.end() || !pat->is_object() || !pat->contains("Regex") ||
        !(*pat)["Regex"].is_string()) {
      Fail("Split pre-tokenizer without a Regex pattern");
    }
    regexes.push_back((*pat)["Regex"].get<std::string>());
    return;
  }
  if (type == "ByteLevel") {
    // add_prefix_space defaults to true in HF; we only support false.
    if (node.value("add_prefix_space", true)) {
      Fail("ByteLevel pre-tokenizer with add_prefix_space=true unsupported");
    }
    // use_regex=true applies the ORIGINAL GPT-2 split regex INSIDE ByteLevel
    // instead of carrying it in an explicit Split component. The Qwen/Llama-3
    // checkpoints use the Split form; the pre-Llama byte-level BPE family (OPT,
    // GPT-2, ...) uses this one, so it selects SplitPattern::kGpt2 below.
    if (node.value("use_regex", false)) byte_level_use_regex = true;
    saw_byte_level = true;
    return;
  }
  Fail("unsupported pre_tokenizer component \"" + type + "\"");
}

// A SentencePiece tokenizer.json (Mistral, Gemma, ...) carries a `Metaspace`
// pre_tokenizer instead of the byte-level Split/ByteLevel components: whitespace
// is replaced with a replacement char (▁ = U+2581) and BPE runs over the
// resulting string (NOT the byte-mapped alphabet), with byte-fallback for
// characters absent from the vocab. Recognized here so FromHfJson can dispatch
// to the SentencePiece family; anything else is left to DetectPattern (byte
// level) or fails loudly there. Mirrors HF tokenizers `pre_tokenizers::Metaspace`
// (tokenizers 0.22, pre_tokenizers/metaspace.rs). Returns false if the
// pre_tokenizer is not a bare Metaspace node.
bool DetectMetaspace(const json& doc, std::string& replacement,
                     std::string& prepend_scheme, bool& split) {
  const auto it = doc.find("pre_tokenizer");
  if (it == doc.end() || it->is_null() || !it->is_object()) return false;
  const json* pt = &*it;
  // GLiNER2.5 (mdeberta-v3-base) wraps a single Metaspace in a Sequence.
  // Unwrap it so the same validation path applies.
  if (pt->value("type", "") == "Sequence") {
    const auto subs = pt->find("pretokenizers");
    if (subs == pt->end() || !subs->is_array() || subs->size() != 1) return false;
    pt = &(*subs)[0];
  }
  if (pt->value("type", "") != "Metaspace") return false;
  // HF Metaspace defaults: replacement "▁", prepend_scheme "always", split true.
  replacement = pt->value("replacement", std::string("\xE2\x96\x81"));
  prepend_scheme = pt->value("prepend_scheme", std::string("always"));
  split = pt->value("split", true);
  return true;
}

// The modern SentencePiece tokenizer.json (Gemma-4, and some Llama-3-SP
// variants) expresses metaspace NOT as a single `Metaspace` pre_tokenizer but
// as a `Replace(" " -> "▁")` NORMALIZER plus a `Split(" ", MergedWithPrevious)`
// pre_tokenizer. HF tokenizers treats this pair as EQUIVALENT to
// Metaspace(replacement="▁", split=false, prepend_scheme="never"): the
// normalizer maps every literal space to ▁ and the merged-with-previous split
// keeps the ▁ attached to the following word — exactly the metaspace layout. We
// fold it onto the SAME validated metaspace machinery (space->▁ encode, ▁->space
// decode). Returns true iff the normalizer + pre_tokenizer are exactly this
// Gemma pair. Additive: no previously-loadable tokenizer used this shape (the
// bare `Replace` normalizer used to fail loudly in CheckNormalizer).
bool DetectGemmaMetaspaceNormalizer(const json& doc) {
  const auto nz = doc.find("normalizer");
  if (nz == doc.end() || !nz->is_object()) return false;
  if (nz->value("type", "") != "Replace") return false;
  const auto pat = nz->find("pattern");
  if (pat == nz->end() || !pat->is_object()) return false;
  if (pat->value("String", "") != " ") return false;
  if (nz->value("content", "") != "\xE2\x96\x81") return false;  // ▁ = U+2581
  const auto pt = doc.find("pre_tokenizer");
  if (pt == doc.end() || !pt->is_object()) return false;
  if (pt->value("type", "") != "Split") return false;
  if (pt->value("behavior", "") != "MergedWithPrevious") return false;
  const auto ppat = pt->find("pattern");
  if (ppat == pt->end() || !ppat->is_object()) return false;
  if (ppat->value("String", "") != " ") return false;
  return true;
}

// Parses an HF byte-fallback token "<0xNN>" (case-insensitive hex) to its byte
// value, or -1 if `token` is not exactly a 6-char "<0xNN>" form. Mirrors HF
// tokenizers `ByteFallback` decode (decoders/byte_fallback.rs).
int ParseByteToken(const std::string& token) {
  if (token.size() != 6 || token[0] != '<' || token[1] != '0' ||
      token[2] != 'x' || token[5] != '>') {
    return -1;
  }
  const auto hex = [](char c) -> int {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'a' && c <= 'f') return c - 'a' + 10;
    if (c >= 'A' && c <= 'F') return c - 'A' + 10;
    return -1;
  };
  const int hi = hex(token[3]);
  const int lo = hex(token[4]);
  if (hi < 0 || lo < 0) return -1;
  return (hi << 4) | lo;
}

// Strict UTF-8 validation matching Rust's std::str::from_utf8 (used by HF's
// ByteFallback decoder): rejects overlong encodings, surrogates
// (U+D800..U+DFFF) and codepoints above U+10FFFF.
bool IsValidUtf8(const uint8_t* p, size_t n) {
  size_t i = 0;
  while (i < n) {
    const uint8_t b0 = p[i];
    if (b0 < 0x80) {
      ++i;
      continue;
    }
    size_t need;
    uint8_t lo = 0x80, hi = 0xBF;  // first continuation byte's valid range
    if (b0 >= 0xC2 && b0 <= 0xDF) {
      need = 1;
    } else if (b0 == 0xE0) {
      need = 2;
      lo = 0xA0;
    } else if (b0 >= 0xE1 && b0 <= 0xEC) {
      need = 2;
    } else if (b0 == 0xED) {
      need = 2;
      hi = 0x9F;
    } else if (b0 >= 0xEE && b0 <= 0xEF) {
      need = 2;
    } else if (b0 == 0xF0) {
      need = 3;
      lo = 0x90;
    } else if (b0 >= 0xF1 && b0 <= 0xF3) {
      need = 3;
    } else if (b0 == 0xF4) {
      need = 3;
      hi = 0x8F;
    } else {
      return false;  // 0x80..0xC1, 0xF5..0xFF: never a valid lead
    }
    if (i + need >= n) return false;
    for (size_t k = 1; k <= need; ++k) {
      const uint8_t b = p[i + k];
      const uint8_t klo = (k == 1) ? lo : uint8_t{0x80};
      const uint8_t khi = (k == 1) ? hi : uint8_t{0xBF};
      if (b < klo || b > khi) return false;
    }
    i += need + 1;
  }
  return true;
}

SplitPattern DetectPattern(const json& doc) {
  const auto it = doc.find("pre_tokenizer");
  if (it == doc.end() || it->is_null()) Fail("missing pre_tokenizer");
  // Checked BEFORE the single-Split walk: both DeepSeek families are
  // Sequence PIPELINES whose components the walk cannot express -- V2's
  // Digits stage and five Splits, V3's three COMPOSING Splits. The walk
  // accepts exactly one Split, so without these two lines a DeepSeek
  // checkpoint is refused rather than mis-tokenized, which is what
  // `vllm-server` did to every DeepSeek-V4-Flash safetensors artifact
  // (#1924): `expected exactly one Split pre-tokenizer, found 3`.
  if (IsDeepSeekPreTokenizer(*it)) return SplitPattern::kDeepSeek;
  if (IsDeepSeekV3PreTokenizer(*it)) return SplitPattern::kDeepSeekV3;
  std::vector<std::string> regexes;
  bool saw_byte_level = false;
  bool byte_level_use_regex = false;
  WalkPreTokenizer(*it, regexes, saw_byte_level, byte_level_use_regex);
  if (!saw_byte_level) Fail("pre_tokenizer has no ByteLevel component");
  // ByteLevel(use_regex=true) with NO explicit Split component == the original
  // GPT-2 byte-level BPE split (facebook/opt-125m, gpt2, ...).
  if (byte_level_use_regex && regexes.empty()) return SplitPattern::kGpt2;
  if (regexes.size() != 1) {
    Fail("expected exactly one Split pre-tokenizer, found " +
         std::to_string(regexes.size()));
  }
  // Some checkpoints (stabilityai/stablelm-2-*, Arcade100k) store the cl100k/Qwen2
  // split regex with LITERAL CR/LF control characters inside its `[\r\n]` character
  // classes, where the Qwen checkpoints write the `\r`/`\n` escape sequences. A
  // regex engine treats a literal CR and the escape `\r` identically, so the two
  // forms are the SAME pattern — canonicalize the control chars to their escape
  // form so the byte-equality checks below match. This is a no-op for every
  // checkpoint that already uses the escape form (none ship a literal CR/LF byte in
  // the split regex), so no existing detection changes.
  std::string re = regexes[0];
  {
    std::string canon;
    canon.reserve(re.size());
    for (const char c : re) {
      if (c == '\r') {
        canon += "\\r";
      } else if (c == '\n') {
        canon += "\\n";
      } else {
        canon += c;
      }
    }
    re.swap(canon);
  }
  if (re == kQwen36Regex) return SplitPattern::kQwen2;
  if (re == kClassicQwen2Regex) return SplitPattern::kQwen2Classic;
  // The Kolibri-1 spelling of the classic Qwen2 pattern: \p{N}{1} instead of
  // \p{N}. Same language (the {1} quantifier is the identity on its atom),
  // so the SAME scanner is exact — see the constant's comment above. Matched
  // exactly, before the `\p{N}{1,3}` heuristic: this pattern contains
  // `\p{N}{1}|`, not the heuristic's `\p{N}{1,3}` substring, but recognition
  // stays whole-pattern per the rule at the heuristic below.
  if (re == kClassicQwen2N1Regex) return SplitPattern::kQwen2Classic;
  if (re == kTekkenRegex) return SplitPattern::kTekken;
  // BOTH exact matches BEFORE the `\p{N}{1,3}` heuristic. kGpt4oRegex groups
  // digits in threes as well, so the heuristic below claims it and hands back
  // kLlama3 -- a split that tokenizes plausibly and WRONGLY (it never breaks
  // CamelCase and it detaches every contraction). Muse Glimmer's HF
  // tokenizer.json is exactly that shape, so this line is what stops the
  // safetensors path from silently mistokenizing it. Issue #347.
  // kTekkenRegex uses single-codepoint `\p{N}` and so was never at risk from
  // the heuristic, but it is matched exactly for the same reason: recognition
  // here is by whole pattern, never by a substring that a sibling shares.
  if (re == kGpt4oRegex) return SplitPattern::kGpt4o;
  if (re.find(R"(\p{N}{1,3})") != std::string::npos) {
    return SplitPattern::kLlama3;
  }
  Fail("unrecognized pre-tokenizer split regex: " + re);
}

void CheckNormalizer(const json& doc) {
  const auto it = doc.find("normalizer");
  if (it == doc.end() || it->is_null()) return;
  const std::string type =
      it->is_object() ? it->value("type", "") : std::string();
  // DEVIATION (recorded): Qwen3.6 declares an NFC normalizer. We accept it
  // but do NOT apply NFC — inputs are assumed already NFC-normalized (true
  // for the parity corpus; divergence is only possible on decomposed input).
  if (type == "NFC") return;
  // DeepSeek-V2 declares `{"type": "Sequence", "normalizers": []}` — an EMPTY
  // pipeline, i.e. a genuine no-op. Accept exactly that; a Sequence with any
  // component still fails, because we would then be skipping real work.
  //
  // GLiNER2.5 (mdeberta-v3-base tokenizer) declares a Sequence with Replace
  // (whitespace collapse), NFC, and Strip. Accept this combination without
  // applying it — the NER pipeline's offset mapping handles raw text directly,
  // and for ASCII English input the normalizer is effectively a no-op.
  if (type == "Sequence") {
    const auto sub = it->find("normalizers");
    if (sub != it->end() && sub->is_array()) {
      if (sub->empty()) return;  // empty Sequence = no-op (DeepSeek-V2)
      bool all_skippable = true;
      for (const auto& n : *sub) {
        const std::string t = n.value("type", "");
        if (t != "NFC" && t != "Strip" && t != "Replace") {
          all_skippable = false;
          break;
        }
      }
      if (all_skippable) return;  // GLiNER2.5: Replace+NFC+Strip = skippable
    }
    Fail("unsupported non-empty normalizer Sequence");
  }
  Fail("unsupported normalizer \"" + type + "\"");
}

// eos/bos are only extracted when trivially available: a TemplateProcessing
// post_processor whose "single" template starts/ends with a SpecialToken.
// Otherwise they stay -1 and callers must use the model config's ids (e.g.
// Qwen's ByteLevel post_processor carries neither).
void ExtractBosEos(const json& doc, int32_t& bos, int32_t& eos) {
  const auto pp = doc.find("post_processor");
  if (pp == doc.end() || !pp->is_object()) return;
  // The TemplateProcessing node may be the post_processor itself (OPT) OR one
  // member of a `Sequence` pipeline (Llama-3 wraps a ByteLevel + a
  // TemplateProcessing in a Sequence; the ByteLevel carries no template tokens,
  // the TemplateProcessing prepends BOS <|begin_of_text|> = 128000). Resolve the
  // TemplateProcessing node in either shape.
  const json* tp = nullptr;
  const std::string pp_type = pp->value("type", "");
  if (pp_type == "TemplateProcessing") {
    tp = &*pp;
  } else if (pp_type == "Sequence") {
    const auto procs = pp->find("processors");
    if (procs != pp->end() && procs->is_array()) {
      for (const auto& p : *procs) {
        if (p.is_object() && p.value("type", "") == "TemplateProcessing") {
          tp = &p;
          break;
        }
      }
    }
  }
  if (tp == nullptr) return;
  const auto single = tp->find("single");
  const auto specials = tp->find("special_tokens");
  if (single == tp->end() || !single->is_array() || single->empty() ||
      specials == tp->end() || !specials->is_object()) {
    return;
  }
  const auto id_of = [&](const json& step) -> int32_t {
    if (!step.is_object() || !step.contains("SpecialToken")) return -1;
    const auto& st = step["SpecialToken"];
    if (!st.is_object() || !st.contains("id") || !st["id"].is_string()) {
      return -1;
    }
    const auto entry = specials->find(st["id"].get<std::string>());
    if (entry == specials->end() || !entry->is_object()) return -1;
    const auto ids = entry->find("ids");
    if (ids == entry->end() || !ids->is_array() || ids->size() != 1 ||
        !(*ids)[0].is_number_integer()) {
      return -1;
    }
    return (*ids)[0].get<int32_t>();
  };
  bos = id_of(single->front());
  eos = id_of(single->back());
}

// ---- GGUF kv access (FromGguf) ----

const GgufValue& RequireKv(const GgufFile& f, const char* key) {
  const GgufValue* v = f.FindKv(key);
  if (v == nullptr) Fail(std::string("GGUF missing kv \"") + key + "\"");
  return *v;
}

const std::string& KvString(const GgufFile& f, const char* key) {
  const GgufValue& v = RequireKv(f, key);
  if (v.TypeId() != kGgufString) {
    Fail(std::string("GGUF kv \"") + key + "\" is not a string");
  }
  return std::get<std::string>(v.v);
}

// Array kv whose elements are of GGUF value type `elem_type`. The reader
// guarantees every element matches the array's declared elem_type, so
// checking it once here makes the std::get in the callers safe.
const GgufArray& KvArray(const GgufFile& f, const char* key,
                         uint32_t elem_type, const char* elem_name) {
  const GgufValue& v = RequireKv(f, key);
  if (v.TypeId() != kGgufArray) {
    Fail(std::string("GGUF kv \"") + key + "\" is not an array");
  }
  const GgufArray& arr = std::get<GgufArray>(v.v);
  if (arr.elem_type != elem_type) {
    Fail(std::string("GGUF kv \"") + key + "\" is not a " + elem_name +
         " array");
  }
  return arr;
}

}  // namespace

void Tokenizer::FinalizeTables() {
  // Ids must be sane before sizing the table: bound them by the entry counts
  // (BPE vocabs are dense) so a hostile file cannot force a huge allocation.
  const size_t bound = vocab_.size() + added_tokens_.size() + 4096;
  size_t max_id = 0;
  const auto check_id = [&](int32_t id) {
    if (id < 0 || static_cast<size_t>(id) >= bound) {
      Fail("token id " + std::to_string(id) + " out of range (" +
           std::to_string(vocab_.size()) + " vocab + " +
           std::to_string(added_tokens_.size()) + " added tokens)");
    }
    if (static_cast<size_t>(id) > max_id) max_id = static_cast<size_t>(id);
  };
  for (const auto& kv : vocab_) check_id(kv.second);
  for (const auto& t : added_tokens_) check_id(t.id);
  if (vocab_.empty() && added_tokens_.empty()) Fail("empty vocab");

  // The unk sentinel must stay outside the merge table's identifier space, or
  // `EncodePlainSp` could merge a symbol that is not a token. `InsertMerge`
  // above already refuses any merge naming a token absent from the vocabulary,
  // so this can only fire on a checkpoint that puts the sentinel's bytes in its
  // vocabulary AND names them in a merge. Checked once here rather than per
  // request. (Unigram has no merges, so the check is skipped.)
  if (!is_unigram_ &&
      merge_ranks_.Find(UnkSentinel()) != MergeRanks::kNoSymbol) {
    Fail("the unk sentinel is named by a merge; it must not be a real symbol");
  }

  token_text_.assign(max_id + 1, std::string());
  is_added_.assign(max_id + 1, 0);
  for (const auto& kv : vocab_) {
    auto& slot = token_text_[static_cast<size_t>(kv.second)];
    if (!slot.empty()) Fail("duplicate token id " + std::to_string(kv.second));
    slot = kv.first;
  }
  for (const auto& t : added_tokens_) {
    auto& slot = token_text_[static_cast<size_t>(t.id)];
    // Llama-3-style files list specials in BOTH model.vocab and
    // added_tokens with identical content; accept that, reject conflicts.
    if (!slot.empty() && slot != t.text) {
      Fail("duplicate token id " + std::to_string(t.id) +
           " with conflicting text");
    }
    slot = t.text;
    is_added_[static_cast<size_t>(t.id)] = t.special ? 2 : 1;
  }

  // The longest STORED token text, computed once here rather than per request.
  // Callers use it as an upper bound on the bytes ONE token can account for in
  // the input, so it has to be an over-estimate and never an under-estimate.
  // The stored form gives that for free on both families: a byte-level token
  // stores one mapped CODEPOINT per input byte (1-2 UTF-8 bytes each), and a
  // SentencePiece token stores its literal text with the metaspace mark (3
  // bytes) standing in for one space and "<0xNN>" (6 bytes) for one fallback
  // byte. Measured over the four committed goldens: 256 (Qwen3.6, DeepSeek-V2),
  // 192 (muse_glimmer), 48 (Mistral).
  max_token_bytes_ = 0;
  for (const std::string& text : token_text_) {
    if (text.size() > max_token_bytes_) max_token_bytes_ = text.size();
  }
}

Tokenizer Tokenizer::FromHfJson(const std::string& tokenizer_json_path) {
  std::ifstream in(tokenizer_json_path, std::ios::binary);
  if (!in) Fail("cannot open " + tokenizer_json_path);
  std::string bytes((std::istreambuf_iterator<char>(in)),
                    std::istreambuf_iterator<char>());
  return FromHfJsonBytes(
      std::string_view(bytes.data(), bytes.size()), tokenizer_json_path);
}

Tokenizer Tokenizer::FromHfJsonBytes(std::string_view tokenizer_json,
                                     const std::string& source) {
  json doc;
  try {
    doc = json::parse(tokenizer_json);
  } catch (const json::exception& e) {
    Fail("JSON parse error in " + source + ": " + e.what());
  }
  if (!doc.is_object()) Fail("top-level JSON is not an object");

  Tokenizer tok;
  // Gemma-4 metaspace-via-normalizer+split (see DetectGemmaMetaspaceNormalizer):
  // recognized BEFORE CheckNormalizer, whose whitelist would otherwise reject the
  // `Replace(" "->"▁")` normalizer. Folded onto the SentencePiece family with the
  // metaspace layout (replacement "▁", split=false, prepend "never" — the Replace
  // form maps existing spaces only, no implicit prefix).
  const bool gemma_metaspace = DetectGemmaMetaspaceNormalizer(doc);
  if (!gemma_metaspace) CheckNormalizer(doc);
  // Dispatch on the pre_tokenizer FAMILY: a bare `Metaspace` node selects the
  // SentencePiece family (Mistral/Gemma); everything else is byte-level BPE and
  // goes through DetectPattern (which fails loudly on Metaspace, so the two
  // families never overlap).
  std::string ms_repl;
  std::string ms_scheme;
  bool ms_split = true;
  std::string unk_token;  // model.unk_token, resolved to an id below
  if (gemma_metaspace) {
    tok.family_ = Family::kSentencePiece;
    tok.metaspace_replacement_ = "\xE2\x96\x81";  // ▁
    tok.metaspace_split_ = false;
    tok.prepend_scheme_ = PrependScheme::kNever;
  } else if (DetectMetaspace(doc, ms_repl, ms_scheme, ms_split)) {
    tok.family_ = Family::kSentencePiece;
    if (ms_repl.empty()) Fail("Metaspace has empty replacement");
    tok.metaspace_replacement_ = ms_repl;
    tok.metaspace_split_ = ms_split;
    if (ms_scheme == "never") {
      tok.prepend_scheme_ = PrependScheme::kNever;
    } else if (ms_scheme == "first") {
      tok.prepend_scheme_ = PrependScheme::kFirst;
    } else if (ms_scheme == "always") {
      tok.prepend_scheme_ = PrependScheme::kAlways;
    } else {
      Fail("unsupported Metaspace prepend_scheme \"" + ms_scheme + "\"");
    }
    // split=true pre-splits the metaspace string into per-▁ pretokens
    // (SplitDelimiterBehavior::MergedWithNext) before BPE, so merges cannot
    // cross a ▁ boundary — see EncodePlainSp. This used to Fail ("no golden in
    // scope"); the golden arrived with the Parakeet checkpoints (every one
    // ships `split: true`), gated by tests/vllm/test_tokenizer_metaspace_split
    // .cpp against HF tokenizers 0.22 pre_tokenizers/metaspace.rs semantics
    // and the pre-refactor examples/parakeet_transcribe DecodeIds reference.
  } else {
    tok.pattern_ = DetectPattern(doc);
  }

  // DECODER selection for the SentencePiece family. Mistral/Gemma ship a
  // Sequence decoder (Replace -> ByteFallback -> Fuse -> Strip), which is the
  // long-standing default SpDecodeTokens applies. Parakeet ships a bare
  // `Metaspace` DECODER node whose decode_chain rule differs observably (the
  // first token DROPS every replacement instead of Strip's one leading space),
  // so recognize it and record its own replacement + prepend_scheme (HF's
  // decode_chain reads the DECODER node's fields, tokenizers 0.22
  // decoders/mod.rs + pre_tokenizers/metaspace.rs).
  if (tok.family_ == Family::kSentencePiece) {
    const auto dec = doc.find("decoder");
    if (dec != doc.end() && dec->is_object() &&
        dec->value("type", "") == "Metaspace") {
      tok.sp_decoder_metaspace_ = true;
      tok.sp_decoder_replacement_ =
          dec->value("replacement", std::string("\xE2\x96\x81"));
      if (tok.sp_decoder_replacement_.empty()) {
        Fail("Metaspace decoder has empty replacement");
      }
      tok.sp_decoder_prepend_never_ =
          dec->value("prepend_scheme", std::string("always")) == "never";
    }
  }

  const auto model_it = doc.find("model");
  if (model_it == doc.end() || !model_it->is_object()) Fail("missing model");
  const json& model = *model_it;
  const std::string model_type = model.value("type", "");
  const bool is_unigram = (model_type == "Unigram");
  if (model_type != "BPE" && !is_unigram) {
    Fail("unsupported model type \"" + model_type +
         "\" (only BPE and Unigram)");
  }
  tok.is_unigram_ = is_unigram;
  // These options change BPE segmentation semantics; neither family uses them.
  // (Unigram has no merges, so the options are irrelevant, but we still check.)
  for (const char* key : {"continuing_subword_prefix", "end_of_word_suffix"}) {
    const auto it = model.find(key);
    if (it != model.end() && !it->is_null() &&
        !(it->is_string() && it->get<std::string>().empty())) {
      Fail(std::string("unsupported BPE option \"") + key + "\"");
    }
  }
  tok.ignore_merges_ = model.value("ignore_merges", false);
  // SentencePiece BPE model options (byte-fallback + unk fusion). byte_fallback
  // decomposes an out-of-vocab character into its UTF-8 bytes as "<0xNN>"
  // tokens; fuse_unk collapses consecutive unknowns into one unk id.
  if (tok.family_ == Family::kSentencePiece) {
    tok.byte_fallback_ = model.value("byte_fallback", false);
    tok.fuse_unk_ = model.value("fuse_unk", false);
    const auto unk_it = model.find("unk_token");
    if (unk_it != model.end() && unk_it->is_string()) {
      unk_token = unk_it->get<std::string>();
    }
    // Unigram models carry unk_id (an int) instead of unk_token (a string).
    if (is_unigram) {
      const auto uid_it = model.find("unk_id");
      if (uid_it != model.end() && uid_it->is_number_integer()) {
        tok.unk_id_ = uid_it->get<int32_t>();
      }
    }
  }

  // Vocab. BPE stores an object {token: id}; Unigram stores an array of
  // [token, score] pairs where index == id and score is a log-probability.
  // Note: nlohmann keeps the LAST value for duplicate JSON keys; a well-formed
  // tokenizer.json has none.
  const auto vocab_it = model.find("vocab");
  if (vocab_it == model.end()) Fail("missing model.vocab");
  if (is_unigram) {
    if (!vocab_it->is_array()) Fail("Unigram model.vocab must be an array");
    int32_t id = 0;
    float min_score = 0.0f;
    for (const auto& entry : *vocab_it) {
      if (!entry.is_array() || entry.size() != 2 ||
          !entry[0].is_string() || !entry[1].is_number()) {
        Fail("malformed Unigram vocab entry at index " + std::to_string(id));
      }
      const std::string text = entry[0].get<std::string>();
      if (text.empty()) Fail("empty string in Unigram vocab");
      const float score = entry[1].get<float>();
      tok.vocab_[text] = id;
      tok.unigram_scores_[text] = score;
      if (score < min_score) min_score = score;
      ++id;
    }
    tok.unigram_min_score_ = min_score;
  } else {
    if (!vocab_it->is_object()) Fail("missing model.vocab object");
    for (const auto& [text, id] : vocab_it->items()) {
      if (text.empty()) Fail("empty string in vocab");
      if (!id.is_number_integer()) Fail("non-integer id for vocab entry");
      tok.vocab_.emplace(text, id.get<int32_t>());
    }
  }

  // Merges: legacy "left right" strings or newer [left, right] arrays.
  // Unigram models have no merges (Viterbi replaces the BPE merge loop).
  if (!is_unigram) {
    const auto merges_it = model.find("merges");
    if (merges_it == model.end() || !merges_it->is_array()) {
      Fail("missing model.merges array");
    }
    int32_t rank = 0;
    for (const auto& entry : *merges_it) {
      std::string left;
      std::string right;
      if (entry.is_string()) {
        SplitMergeEntry(entry.get<std::string>(), left, right);
      } else if (entry.is_array() && entry.size() == 2 && entry[0].is_string() &&
                 entry[1].is_string()) {
        left = entry[0].get<std::string>();
        right = entry[1].get<std::string>();
      } else {
        Fail("malformed merge entry at rank " + std::to_string(rank));
      }
      if (left.empty() || right.empty() ||
          left.find(' ') != std::string::npos ||
          right.find(' ') != std::string::npos) {
        Fail("malformed merge entry at rank " + std::to_string(rank));
      }
      InsertMerge(tok.merge_ranks_, tok.vocab_, left, right, rank);
      ++rank;
    }
  }

  // Added tokens. The special flag does not change encoding; it drives
  // detokenizer skip logic. lstrip/rstrip ARE implemented (see SpecialToken and
  // Encode(): the matched token eats the adjacent whitespace, mirroring
  // transformers tokenization_utils.py:670-677). single_word alters word-boundary
  // matching in a way we do not emulate and stays a loud failure; normalized is a
  // no-op without an applied normalizer.
  const auto added_it = doc.find("added_tokens");
  if (added_it != doc.end() && !added_it->is_null()) {
    if (!added_it->is_array()) Fail("added_tokens is not an array");
    for (const auto& entry : *added_it) {
      if (!entry.is_object() || !entry.contains("id") ||
          !entry["id"].is_number_integer() || !entry.contains("content") ||
          !entry["content"].is_string()) {
        Fail("malformed added_tokens entry");
      }
      SpecialToken t;
      t.id = entry["id"].get<int32_t>();
      t.text = entry["content"].get<std::string>();
      t.special = entry.value("special", false);
      if (t.text.empty()) Fail("added token with empty content");
      // REAL semantics, not accept-and-ignore: a token flagged rstrip eats the
      // whitespace after its match, lstrip the whitespace before it. Silently
      // dropping the flag mis-tokenizes every checkpoint that sets it (e.g.
      // Phi-4-mini's <|assistant|>/<|end|>/<|user|>/<|system|>/<|tool|> family,
      // all rstrip=true), so we carry it into Encode().
      t.lstrip = entry.value("lstrip", false);
      t.rstrip = entry.value("rstrip", false);
      // single_word genuinely changes word-boundary matching (the token is NOT
      // split off mid-word, transformers tokenization_utils.py:678-681); we do
      // not emulate it, so refuse loudly rather than mis-tokenize.
      if (entry.value("single_word", false)) {
        Fail("added token \"" + t.text +
             "\" uses unsupported option \"single_word\"");
      }
      tok.added_tokens_.push_back(std::move(t));
    }
  }

  // Resolve the SentencePiece unk token to its id (used only when a character
  // has no vocab token AND byte-fallback cannot cover it — for Mistral every
  // byte 0x00..0xFF has a "<0xNN>" token so this path is never taken).
  if (tok.family_ == Family::kSentencePiece && !unk_token.empty()) {
    const auto it = tok.vocab_.find(unk_token);
    if (it != tok.vocab_.end()) tok.unk_id_ = it->second;
  }

  ExtractBosEos(doc, tok.bos_id_, tok.eos_id_);
  // The HF post_processor template is the ONLY thing that licenses adding
  // special tokens at encode time (see EncodeWithSpecialTokens). Capture it
  // separately from the model's bos/eos ids so the GGUF path — where those ids
  // are declared but nothing is auto-added — is unaffected.
  tok.template_bos_ = tok.bos_id_;
  tok.template_eos_ = tok.eos_id_;
  tok.FinalizeTables();
  return tok;
}

Tokenizer Tokenizer::FromGguf(const GgufFile& f) {
  Tokenizer tok;

  // "gpt2" is llama.cpp's name for byte-level BPE (the only family we do).
  const std::string& model = KvString(f, "tokenizer.ggml.model");
  if (model != "gpt2") {
    Fail("unsupported tokenizer.ggml.model \"" + model +
         "\" (only \"gpt2\" byte-level BPE)");
  }

  // Pre-tokenizer, by llama.cpp pre name. llama.cpp DOES distinguish the
  // \p{M}-aware regex variant: pre "qwen35" selects
  // LLAMA_VOCAB_PRE_TYPE_QWEN35 whose regex equals our kQwen2 pattern (the
  // APEX GGUFs carry pre "qwen35"; see llama-vocab.cpp:382,2200 in the
  // phase93 fork). Pre "qwen2" is the CLASSIC qwen2 regex, which differs on
  // combining marks and which we do not implement — accepting it would
  // silently mistokenize real Qwen2.5/Qwen3 GGUFs, so it fails loudly like
  // the HfJson path.
  const std::string& pre = KvString(f, "tokenizer.ggml.pre");
  if (pre == "qwen35") {
    tok.pattern_ = SplitPattern::kQwen2;
  } else if (pre == "qwen2") {
    tok.pattern_ = SplitPattern::kQwen2Classic;
  } else if (pre == "llama-bpe") {
    tok.pattern_ = SplitPattern::kLlama3;
  } else if (pre == "gpt-4o" || pre == "llama4" || pre == "kanana2" ||
             pre == "talkie") {
    // The GPT-4o / o200k family. These are exactly the four pre names
    // llama.cpp maps to LLAMA_VOCAB_PRE_TYPE_GPT4O (llama.cpp
    // src/llama-vocab.cpp:2294-2299 @ 153d324bcf), and that pre-type carries
    // its own regex, ported verbatim as SplitPattern::kGpt4o (see
    // src/vllm/tokenizer/pretokenizer.cpp for the pattern and its provenance).
    // It is NOT kLlama3 with different digit grouping: the word alternatives
    // split CamelCase and swallow the contraction as a SUFFIX, so aliasing it
    // would mistokenize ordinary English silently — the same reason "qwen2"
    // above is refused rather than folded into "qwen35".
    //
    // llama.cpp also sets clean_spaces = false for this pre-type. That flag
    // drives llama.cpp's own detokenizer space fixups, which our byte-level
    // Decode() never performs, so "off" is already our behaviour and there is
    // nothing to carry over.
    tok.pattern_ = SplitPattern::kGpt4o;
  } else if (pre == "deepseek-llm") {
    // The DeepSeek-V2 family. llama.cpp maps this pre name to
    // LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM (src/llama-vocab.cpp:2162 @ b10451,
    // regexes at :308-316) — the same splitting rule this file ports as the
    // seven-stage SplitPattern::kDeepSeek. It used to resolve to kLlama3 as a
    // "close approximation" — added by the DeepSeek-V4 W8 spike before
    // kDeepSeek existed, and never revisited once it did. It is not close:
    // kDeepSeek isolates every newline, groups digits ONE at a time rather
    // than in threes, and carries enumerated letter/punct/CJK classes that
    // kLlama3's `\p{L}`/`\p{N}` alternation does not have.
    tok.pattern_ = SplitPattern::kDeepSeek;
  } else if (pre == "deepseek-v3" || pre == "joyai-llm" ||
             pre == "hunyuan-dense") {
    // The DeepSeek-V3 family (DeepSeek-V3, DeepSeek-R1, DeepSeek-V4-Flash).
    // These are exactly the three pre names llama.cpp maps to
    // LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM (src/llama-vocab.cpp:2170, :2351 and
    // :2355 @ b10451), and that pre-type carries its own three regexes at
    // :318-325 — BYTE-IDENTICAL to this checkpoint's three, and ported verbatim
    // as SplitPattern::kDeepSeekV3 (#1924).
    //
    // These also used to resolve to kLlama3 as a "close approximation", with a
    // comment naming the DeepSeek-V4-Flash GGUF (antirez/ds4 q2-imatrix,
    // pre="joyai-llm") as the artifact that forced it. The approximation is
    // real and it is wrong on ordinary text: kLlama3 gives "(" + "x" where
    // this family binds "(x" into one piece, and it splits contractions the
    // other way round. Same failure shape as #347, where a `\p{N}{1,3}`
    // lookalike silently claimed the GPT-4o family.
    //
    // llama.cpp also sets clean_spaces = false for this pre-type. That flag
    // drives llama.cpp's own detokenizer space fixups, which our byte-level
    // Decode() never performs, so "off" is already our behaviour.
    tok.pattern_ = SplitPattern::kDeepSeekV3;
  } else if (pre == "laguna") {
    // Laguna-S-2.1 GGUFs tag pre="laguna"; the vocab is gpt2 byte-level BPE and
    // the pretokenizer is the GPT-2/Llama-3 byte-level family. Mapped to kLlama3
    // as a close APPROXIMATION for Encode() (feed the reference engine's own ids
    // for a token-EXACT cross-check); Decode() is vocab-based and exact. Left
    // alone by #1924: llama.cpp has no `laguna` pre name at all, so unlike the
    // DeepSeek names above there is no exact pre-type to resolve it onto.
    tok.pattern_ = SplitPattern::kLlama3;
  } else if (pre == "glm4" || pre == "chatglm-bpe") {
    // The GLM-4 / GLM-5 family, which is what every GLM GGUF in the ecosystem
    // carries -- `unsloth/GLM-5.3-Flash-GGUF` UD-Q2_K_XL states
    // `tokenizer.ggml.pre = "glm4"` beside `tokenizer.ggml.model = "gpt2"`
    // (#2277). These are exactly the two pre names llama.cpp maps to
    // LLAMA_VOCAB_PRE_TYPE_CHATGLM4 (src/llama-vocab.cpp:2256-2258 @ b10451),
    // and that pre-type's regex list at :396-399 is a single expression that is
    // BYTE-IDENTICAL to LLAMA_VOCAB_PRE_TYPE_LLAMA3's at :283-290 -- proven by
    // extracting both lines from the pinned object and comparing them, not by
    // reading them (`.agents/specs/glm5-next-flash.md` O21 records the sha256).
    // So kLlama3 is the EXACT splitting rule here, NOT the "close
    // approximation" that `laguna` above still is and that #347 and #1924 each
    // had to undo.
    //
    // WHAT DELIBERATELY DOES NOT COME WITH IT. llama.cpp's `llama-bpe` arm
    // (:2157-2159) sets `ignore_merges = true` and `add_bos = true` beside its
    // pre-type; the `glm4` arm sets NEITHER. Aliasing the two pre names onto one
    // SplitPattern must therefore carry the split rule and none of those flags,
    // and it does: `ignore_merges_` stays false on every GGUF path (see the note
    // below), and no GGUF path prepends a BOS -- `template_bos_` is left -1 by
    // this function, which is what mirrors llama.cpp's `add_bos = false`
    // default (:1815) surviving for this arm, and `add_bos` is what :3382-3384
    // actually tests before pushing `special_bos_id`.
    //
    // THE BOS ID IS STILL READ, AND THAT IS ALSO THE MIRROR. :2259 sets
    // `special_bos_id = LLAMA_TOKEN_NULL` on this arm, but that is a DEFAULT:
    // the loop at :2559-2578 reads `tokenizer.ggml.bos_token_id` out of the file
    // and assigns it over the null whenever the key is present and in range. On
    // the staged artifact that key is 154822, so llama.cpp finishes this load
    // with `special_bos_id = 154822` and `add_bos = false` -- it does NOT
    // discard the id, it declines to PREPEND it. `BosId()` therefore reports
    // 154822 here too, and nothing puts it in a prompt. That matters on this
    // checkpoint: id 154822 is `[gMASK]`, and the file's own chat template
    // opens with the literal text `[gMASK]<sop>`, so a prepended BOS would
    // DOUBLE it on every request.
    tok.pattern_ = SplitPattern::kLlama3;
  } else {
    Fail("unsupported tokenizer.ggml.pre \"" + pre + "\"");
  }
  // GGUF carries no ignore_merges flag and llama.cpp's BPE has no such
  // option, so it stays false. (HF Llama-3 sets ignore_merges=true; a
  // llama-bpe GGUF therefore matches llama.cpp, not HF, on the rare
  // pretokens where that flag matters. Irrelevant for the qwen2 family.)
  // Muse Glimmer's HF tokenizer.json ("llama4" pre) also sets the flag.
  // MEASURED rather than assumed on that checkpoint: encoding 29704 distinct
  // vocab-derived strings plus a 320-string mixed corpus with ignore_merges on
  // and off yields byte-identical ids in every case, so leaving it false here
  // costs that vocab nothing.

  // Vocab: tokens[i] is the string for id i, already in the byte-mapped
  // alphabet (same convention as HF tokenizer.json). token_type says which
  // entries are added tokens instead of plain vocab.
  const GgufArray& tokens =
      KvArray(f, "tokenizer.ggml.tokens", kGgufString, "string");
  const GgufArray& types =
      KvArray(f, "tokenizer.ggml.token_type", kGgufI32, "i32");
  if (tokens.elems.size() != types.elems.size()) {
    Fail("tokenizer.ggml.tokens has " + std::to_string(tokens.elems.size()) +
         " entries but tokenizer.ggml.token_type has " +
         std::to_string(types.elems.size()));
  }

  for (size_t i = 0; i < tokens.elems.size(); ++i) {
    const std::string& text = std::get<std::string>(tokens.elems[i].v);
    const int32_t type = std::get<int32_t>(types.elems[i].v);
    const int32_t id = static_cast<int32_t>(i);
    if (text.empty()) Fail("empty token string at id " + std::to_string(id));
    switch (type) {
      case 1:    // normal
      case 2:    // unknown — still a plain vocab slot for byte-level BPE
      case 5:    // unused — inert PAD-row padding (243x in the APEX GGUF);
                 // plain vocab entry, unreachable via merges
      case 6: {  // byte
        const auto [it, inserted] = tok.vocab_.emplace(text, id);
        if (!inserted) {
          Fail("duplicate token text \"" + text + "\" at ids " +
               std::to_string(it->second) + " and " + std::to_string(id));
        }
        break;
      }
      case 3:  // control -> added token, special (detokenizer may skip)
        // NOTE: real Qwen GGUFs tag FIM/tool tokens control(3) while the HF
        // json has special=false — skip_special_tokens=true detokenization
        // diverges between loaders (faithful to each file).
        tok.added_tokens_.push_back({text, id, /*special=*/true});
        break;
      case 4:  // user-defined -> added token, kept on decode
        tok.added_tokens_.push_back({text, id, /*special=*/false});
        break;
      default:
        Fail("unsupported tokenizer.ggml.token_type " + std::to_string(type) +
             " for token id " + std::to_string(id));
    }
  }

  // Merges are legacy-form "left right" strings.
  const GgufArray& merges =
      KvArray(f, "tokenizer.ggml.merges", kGgufString, "string");
  int32_t rank = 0;
  for (const auto& entry : merges.elems) {
    std::string left;
    std::string right;
    SplitMergeEntry(std::get<std::string>(entry.v), left, right);
    InsertMerge(tok.merge_ranks_, tok.vocab_, left, right, rank);
    ++rank;
  }

  // bos/eos ids are optional u32 kvs; absent -> -1 (callers fall back to the
  // model config, as with FromHfJson).
  const auto token_id_kv = [&](const char* key) -> int32_t {
    const GgufValue* v = f.FindKv(key);
    if (v == nullptr) return -1;
    if (v->TypeId() != kGgufU32) {
      Fail(std::string("GGUF kv \"") + key + "\" is not u32");
    }
    const uint32_t id = std::get<uint32_t>(v->v);
    if (id >= tokens.elems.size()) {
      Fail(std::string(key) + " = " + std::to_string(id) +
           " out of range (vocab size " + std::to_string(tokens.elems.size()) +
           ")");
    }
    return static_cast<int32_t>(id);
  };
  tok.bos_id_ = token_id_kv("tokenizer.ggml.bos_token_id");
  tok.eos_id_ = token_id_kv("tokenizer.ggml.eos_token_id");

  tok.FinalizeTables();
  return tok;
}

void Tokenizer::EncodePlain(std::string_view text,
                            std::vector<int32_t>& out) const {
  for (const auto& [begin, end] : Pretokenize(text, pattern_)) {
    const std::string mapped =
        MapBytesToUnicode(text.substr(begin, end - begin));
    if (ignore_merges_) {
      const auto it = vocab_.find(mapped);
      if (it != vocab_.end()) {
        out.push_back(it->second);
        continue;
      }
    }
    for (const std::string& sym : BpeSplit(mapped, merge_ranks_)) {
      const auto it = vocab_.find(sym);
      if (it == vocab_.end()) {
        // Cannot happen for byte-level vocabs with a complete byte alphabet.
        Fail("symbol \"" + sym +
             "\" not in vocab (incomplete byte-level alphabet?)");
      }
      out.push_back(it->second);
    }
  }
}

void Tokenizer::EncodePlainSp(std::string_view text, bool at_input_start,
                             std::vector<int32_t>& out) const {
  // 1) Metaspace normalization: replace every ASCII space (0x20) with the
  //    replacement (▁), then prepend one replacement per prepend_scheme when
  //    the result does not already start with it (HF Metaspace::pre_tokenize,
  //    tokenizers 0.22 pre_tokenizers/metaspace.rs). Only 0x20 is replaced;
  //    \t/\n are left for byte-fallback below (matches HF).
  std::string s;
  s.reserve(text.size() + metaspace_replacement_.size());
  for (const char c : text) {
    if (c == ' ') {
      s += metaspace_replacement_;
    } else {
      s.push_back(c);
    }
  }
  const bool prepend =
      prepend_scheme_ == PrependScheme::kAlways ||
      (prepend_scheme_ == PrependScheme::kFirst && at_input_start);
  if (prepend && !s.empty() &&
      s.compare(0, metaspace_replacement_.size(), metaspace_replacement_) != 0) {
    s.insert(0, metaspace_replacement_);
  }
  if (s.empty()) return;

  // 2+3 per PRETOKEN) Build the initial BPE symbols, merge, map to ids. HF
  //    constructs the Word BEFORE merging: each character maps to itself when
  //    present in the vocab, else (with byte_fallback) decomposes into its
  //    UTF-8 bytes as "<0xNN>" tokens, else becomes unk. Merges then run over
  //    these symbols; fuse_unk collapses consecutive unks WITHIN the pretoken.
  const std::string& kUnk = UnkSentinel();  // never a real symbol
  const auto encode_piece = [&](std::string_view piece) {
    if (is_unigram_) {
      EncodePlainUnigram(piece, out);
      return;
    }
    std::vector<std::string> symbols;
    size_t pos = 0;
    while (pos < piece.size()) {
      const size_t begin = pos;
      (void)DecodeUtf8(piece, pos);
      std::string ch(piece.substr(begin, pos - begin));
      if (vocab_.find(ch) != vocab_.end()) {
        symbols.push_back(std::move(ch));
        continue;
      }
      if (byte_fallback_) {
        std::vector<std::string> bytes;
        bool all = true;
        for (const unsigned char b : ch) {
          char buf[7];
          std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned>(b));
          std::string bt(buf);
          if (vocab_.find(bt) == vocab_.end()) {
            all = false;
            break;
          }
          bytes.push_back(std::move(bt));
        }
        if (all) {
          for (auto& b : bytes) symbols.push_back(std::move(b));
          continue;
        }
      }
      if (unk_id_ < 0) {
        Fail("SentencePiece: character \"" + std::string(ch) +
             "\" has no vocab token, byte-fallback unavailable, and no unk_token");
      }
      symbols.push_back(kUnk);
    }

    BpeMerge(symbols, merge_ranks_);

    int32_t prev = -1;
    for (const std::string& sym : symbols) {
      int32_t id;
      if (sym == kUnk) {
        id = unk_id_;
      } else {
        const auto it = vocab_.find(sym);
        if (it != vocab_.end()) {
          id = it->second;
        } else if (unk_id_ >= 0) {
          id = unk_id_;
        } else {
          Fail("SentencePiece: merged symbol \"" + sym + "\" not in vocab");
        }
      }
      if (fuse_unk_ && id == unk_id_ && prev == unk_id_) continue;
      out.push_back(id);
      prev = id;
    }
  };

  if (!metaspace_split_) {
    encode_piece(s);
    return;
  }
  // Metaspace `split: true` (HF tokenizers 0.22 pre_tokenizers/metaspace.rs
  // `pre_tokenize`, SplitDelimiterBehavior::MergedWithNext): every occurrence
  // of the replacement STARTS a new pretoken with the replacement attached to
  // what follows, and BPE runs per pretoken, so merges never cross a ▁
  // boundary. Text before the first occurrence (prepend_scheme "never", or a
  // non-space-initial segment) is its own leading pretoken.
  size_t piece_start = 0;
  size_t next = s.find(metaspace_replacement_,
                       piece_start + (s.compare(0, metaspace_replacement_.size(),
                                                metaspace_replacement_) == 0
                                          ? metaspace_replacement_.size()
                                          : 0));
  while (next != std::string::npos) {
    encode_piece(std::string_view(s).substr(piece_start, next - piece_start));
    piece_start = next;
    next = s.find(metaspace_replacement_, next + metaspace_replacement_.size());
  }
  encode_piece(std::string_view(s).substr(piece_start));
}

void Tokenizer::EncodePlainUnigram(std::string_view piece,
                                   std::vector<int32_t>& out) const {
  const size_t n = piece.size();
  if (n == 0) return;

  const size_t max_len = max_token_bytes_;
  const float unk_penalty = unigram_min_score_ - 10.0f;
  constexpr float kNegInf = -1e30f;

  // Viterbi DP: best[i] = best score for piece[0..i), back[i] = start of the
  // segment ending at i. Matches HF tokenizers UnigramModel::tokenize
  // (tokenizers/src/models/unigram/model.rs): at each position, try all vocab
  // entries that match, plus unk for one UTF-8 character.
  std::vector<float> best(n + 1, kNegInf);
  std::vector<int32_t> back(n + 1, -1);
  best[0] = 0.0f;

  for (size_t i = 0; i < n; ++i) {
    if (best[i] == kNegInf) continue;

    // Try all vocab entries starting at byte position i.
    const size_t limit = max_len < n - i ? max_len : n - i;
    for (size_t l = 1; l <= limit; ++l) {
      const std::string sub(piece.substr(i, l));
      const auto vit = vocab_.find(sub);
      if (vit != vocab_.end()) {
        const auto sit = unigram_scores_.find(sub);
        const float score =
            best[i] + (sit != unigram_scores_.end() ? sit->second : 0.0f);
        if (score > best[i + l]) {
          best[i + l] = score;
          back[i + l] = static_cast<int32_t>(i);
        }
      }
    }

    // Try unk for one UTF-8 character at position i.
    if (unk_id_ >= 0) {
      size_t j = i;
      (void)DecodeUtf8(piece, j);
      if (j > i) {
        const float score = best[i] + unk_penalty;
        if (score > best[j]) {
          best[j] = score;
          back[j] = static_cast<int32_t>(i);
        }
      }
    }
  }

  // Backtrack: collect segment start positions in reverse order.
  std::vector<size_t> starts;
  size_t pos = n;
  while (pos > 0) {
    const int32_t start = back[pos];
    if (start < 0) {
      Fail("Unigram: unreachable position " + std::to_string(pos) +
           " in Viterbi backtracking (no tokenization found)");
    }
    starts.push_back(static_cast<size_t>(start));
    pos = static_cast<size_t>(start);
  }

  // Emit ids in forward order with fuse_unk and byte_fallback.
  int32_t prev = -1;
  for (auto it = starts.rbegin(); it != starts.rend(); ++it) {
    const size_t start = *it;
    const size_t end = (it + 1 != starts.rend()) ? *(it + 1) : n;
    const std::string sub(piece.substr(start, end - start));

    const auto vit = vocab_.find(sub);
    if (vit != vocab_.end()) {
      if (fuse_unk_ && vit->second == unk_id_ && prev == unk_id_) continue;
      out.push_back(vit->second);
      prev = vit->second;
    } else if (byte_fallback_) {
      for (const unsigned char b : sub) {
        char buf[7];
        std::snprintf(buf, sizeof(buf), "<0x%02X>", static_cast<unsigned>(b));
        const std::string bt(buf);
        const auto bit = vocab_.find(bt);
        const int32_t id = (bit != vocab_.end()) ? bit->second : unk_id_;
        if (fuse_unk_ && id == unk_id_ && prev == unk_id_) continue;
        out.push_back(id);
        prev = id;
      }
    } else if (unk_id_ >= 0) {
      if (fuse_unk_ && unk_id_ == prev) continue;
      out.push_back(unk_id_);
      prev = unk_id_;
    } else {
      Fail("Unigram: segment \"" + sub +
           "\" has no vocab token, byte-fallback unavailable, and no unk_id");
    }
  }
}

namespace {

// AddedToken rstrip: the index just past the whitespace run that starts at
// `pos`. Mirrors python `right.lstrip()` on the segment FOLLOWING the matched
// token (transformers tokenization_utils.py:670-673); IsWhitespace() is exactly
// python str.isspace(), the predicate str.lstrip() uses.
size_t SkipWhitespaceForward(std::string_view text, size_t pos) {
  while (pos < text.size()) {
    size_t next = pos;
    const uint32_t cp = DecodeUtf8(text, next);
    if (!IsWhitespace(cp)) break;
    pos = next;
  }
  return pos;
}

// AddedToken lstrip: the index of the start of the trailing whitespace run of
// text[begin, end). Mirrors python `left.rstrip()` on the segment PRECEDING the
// matched token (transformers tokenization_utils.py:675-676).
size_t TrimWhitespaceBackward(std::string_view text, size_t begin, size_t end) {
  while (end > begin) {
    // Walk back to the lead byte of the last codepoint (continuation bytes are
    // 10xxxxxx); a malformed tail decodes as one byte, matching DecodeUtf8.
    size_t start = end - 1;
    while (start > begin &&
           (static_cast<uint8_t>(text[start]) & 0xC0) == 0x80 &&
           end - start < 4) {
      --start;
    }
    size_t next = start;
    const uint32_t cp = DecodeUtf8(text, next);
    if (next != end || !IsWhitespace(cp)) break;
    end = start;
  }
  return end;
}

}  // namespace

std::vector<int32_t> Tokenizer::Encode(std::string_view text) const {
  std::vector<int32_t> out;
  size_t pos = 0;
  while (pos < text.size()) {
    // Leftmost-longest added-token match over the raw text (HF added_tokens
    // semantics; special and non-special split identically).
    size_t best_pos = std::string_view::npos;
    const SpecialToken* best = nullptr;
    for (const auto& t : added_tokens_) {
      const size_t found = text.find(t.text, pos);
      if (found == std::string_view::npos) continue;
      if (best == nullptr || found < best_pos ||
          (found == best_pos && t.text.size() > best->text.size())) {
        best_pos = found;
        best = &t;
      }
    }
    // Metaspace prepend_scheme="first" prepends ▁ ONLY to the segment that
    // begins at byte 0 of the whole input (a segment following a special token
    // is not "first"); at_input_start captures exactly that. Inert for the
    // byte-level family.
    const bool at_input_start = pos == 0;
    if (best == nullptr) {
      if (family_ == Family::kSentencePiece) {
        EncodePlainSp(text.substr(pos), at_input_start, out);
      } else {
        EncodePlain(text.substr(pos), out);
      }
      break;
    }
    // lstrip: the matched token eats the whitespace run immediately before it,
    // so the preceding segment is shortened (python `left.rstrip()`).
    const size_t seg_end =
        best->lstrip ? TrimWhitespaceBackward(text, pos, best_pos) : best_pos;
    if (family_ == Family::kSentencePiece) {
      EncodePlainSp(text.substr(pos, seg_end - pos), at_input_start, out);
    } else {
      EncodePlain(text.substr(pos, seg_end - pos), out);
    }
    out.push_back(best->id);
    pos = best_pos + best->text.size();
    // rstrip: the matched token also eats the whitespace run right after it, so
    // the next segment starts past it (python `right.lstrip()`).
    if (best->rstrip) pos = SkipWhitespaceForward(text, pos);
  }
  return out;
}

std::vector<int32_t> Tokenizer::EncodeWithSpecialTokens(
    std::string_view text) const {
  // Realize the TemplateProcessing "single" template ExtractBosEos parsed: a
  // leading SpecialToken is prepended, a trailing one appended; -1 means the
  // template has no such step. Both are -1 for every Qwen tokenizer (their
  // post_processor is ByteLevel, not TemplateProcessing) AND for every GGUF
  // tokenizer (no HF post_processor exists there at all), so this reduces to
  // Encode() for every model that predates the OPT bring-up.
  std::vector<int32_t> out;
  if (template_bos_ >= 0) out.push_back(template_bos_);
  const std::vector<int32_t> body = Encode(text);
  out.insert(out.end(), body.begin(), body.end());
  if (template_eos_ >= 0) out.push_back(template_eos_);
  return out;
}

std::string Tokenizer::SpDecodeTokens(const std::vector<std::string>& tokens,
                                     size_t begin, size_t end) const {
  if (sp_decoder_metaspace_) {
    // The bare `Metaspace` DECODER (HF tokenizers 0.22 pre_tokenizers/
    // metaspace.rs `decode_chain`, reached via decoders/mod.rs): per token,
    // each replacement occurrence is DROPPED in the first token (when
    // prepend_scheme != "never") and becomes ONE space in every later token;
    // all other bytes pass through; tokens concatenate. Deliberately NO
    // ByteFallback / Fuse / Strip — that is the Sequence chain below, and the
    // two differ observably (a first token "▁▁x" decodes to "x" here and to
    // " x" there). A window decodes as its own sequence (t == begin is "first"),
    // which is exactly how the incremental HF decode of a token slice behaves.
    std::string out;
    for (size_t t = begin; t < end; ++t) {
      const std::string& tk = tokens[t];
      size_t p = 0;
      while (p < tk.size()) {
        if (tk.compare(p, sp_decoder_replacement_.size(),
                       sp_decoder_replacement_) == 0) {
          if (t != begin || sp_decoder_prepend_never_) out.push_back(' ');
          p += sp_decoder_replacement_.size();
        } else {
          out.push_back(tk[p]);
          ++p;
        }
      }
    }
    return out;
  }
  // HF tokenizers Sequence decoder for the SentencePiece family:
  //   Replace(▁->" ") -> ByteFallback -> Fuse -> Strip(1 leading space).
  // Replace only affects ▁ (never inside a "<0xNN>" byte token), so it is
  // applied inline when a non-byte token is emitted. ByteFallback accumulates a
  // run of consecutive "<0xNN>" tokens and, on the next non-byte token or at
  // the end, decodes the run: valid UTF-8 -> the decoded text, else one U+FFFD
  // per byte (decoders/byte_fallback.rs). Fuse is the implicit concatenation.
  std::string out;
  std::vector<uint8_t> byte_run;
  const auto flush = [&]() {
    if (byte_run.empty()) return;
    if (IsValidUtf8(byte_run.data(), byte_run.size())) {
      out.append(reinterpret_cast<const char*>(byte_run.data()),
                 byte_run.size());
    } else {
      for (size_t k = 0; k < byte_run.size(); ++k) out += "\xEF\xBF\xBD";
    }
    byte_run.clear();
  };
  for (size_t t = begin; t < end; ++t) {
    const std::string& tk = tokens[t];
    const int byte = ParseByteToken(tk);
    if (byte >= 0) {
      byte_run.push_back(static_cast<uint8_t>(byte));
      continue;
    }
    flush();
    // Replace every ▁ with a space while appending.
    size_t p = 0;
    while (p < tk.size()) {
      const size_t hit = tk.find(metaspace_replacement_, p);
      if (hit == std::string::npos) {
        out.append(tk, p, tk.size() - p);
        break;
      }
      out.append(tk, p, hit - p);
      out.push_back(' ');
      p = hit + metaspace_replacement_.size();
    }
  }
  flush();
  // Strip(content=" ", start=1): remove at most one leading space.
  if (!out.empty() && out.front() == ' ') out.erase(out.begin());
  return out;
}

std::string Tokenizer::Decode(const std::vector<int32_t>& ids) const {
  if (family_ == Family::kSentencePiece) {
    // ids -> stored token strings (vocab: raw "▁Hello"/"<0xNN>"; added: literal
    // content), then the SentencePiece decoder chain. Includes special tokens
    // literally (full decode), matching the byte-level branch below.
    std::vector<std::string> tokens;
    tokens.reserve(ids.size());
    for (const int32_t id : ids) tokens.push_back(TokenText(id));
    return SpDecodeTokens(tokens, 0, tokens.size());
  }
  std::string out;
  for (const int32_t id : ids) {
    const std::string& text = TokenText(id);
    if (is_added_[static_cast<size_t>(id)]) {
      out += text;  // literal content
    } else {
      out += UnmapUnicodeToBytes(text);
    }
  }
  return out;
}

const std::string& Tokenizer::TokenText(int32_t id) const {
  if (!HasToken(id)) Fail("unknown token id " + std::to_string(id));
  return token_text_[static_cast<size_t>(id)];
}

bool Tokenizer::HasToken(int32_t id) const {
  return id >= 0 && static_cast<size_t>(id) < token_text_.size() &&
         !token_text_[static_cast<size_t>(id)].empty();
}

bool Tokenizer::IsSpecial(int32_t id) const {
  return id >= 0 && static_cast<size_t>(id) < is_added_.size() &&
         is_added_[static_cast<size_t>(id)] == 2;
}

}  // namespace vllm::tok
