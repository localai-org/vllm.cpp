// vllm.cpp original (tokenizer); semantics mirror HF tokenizers' Split
// pre-tokenizer (behavior=Isolated) with the byte-level BPE split regexes.
//
// Authoritative patterns (verbatim from tokenizer.json "pre_tokenizer"):
//
// kQwen2 — unsloth/Qwen3.6-27B-NVFP4 tokenizer.json (fetched 2026-07-03):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+|\p{N}| ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
// kQwen2Classic — classic Qwen2/Qwen3 (e.g. Qwen/Qwen3-0.6B, Qwen3-Coder-30B):
//   the same pattern WITHOUT \p{M} in the letter-run and punct classes:
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//   We implement BOTH: kQwen2 (\p{M}-aware) and kQwen2Classic differ only on
//   combining marks adjacent to letter/punctuation runs (kQwen2Classic treats
//   a mark like ordinary punct-run text, exactly as Llama-3 does, but keeps the
//   single-codepoint \p{N} number grouping). The first ADDITIVE model
//   (Qwen3-0.6B, MODEL-TEXT-qwen3) forced kQwen2Classic in.
//
// kLlama3 — PROVISIONAL: no Llama-3 checkpoint exists in the DGX HF cache to
// read verbatim (checked 2026-07-03); pattern from public knowledge of
// meta-llama/Meta-Llama-3-8B tokenizer.json (cl100k-style):
//   (?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
//
// The scanner mirrors the regex alternation: at each position the rules are
// tried in order and the first that matches wins (leftmost-alternation
// semantics); the match is emitted as a byte span and scanning resumes at its
// end. Every position matches some rule (regex-space -> rule 5/6/7; letter ->
// rule 2; number -> rule 3; mark -> rule 2 run for Qwen, rule 4 class for
// Llama-3; everything else, incl. non-space controls and U+FFFD from invalid
// UTF-8, is in rule 4's negated class), so Split's "unmatched text between
// matches" case never occurs and the spans tile the input exactly.
//
// Engine notes (oracle-verified against HF tokenizers 'onig' regex engine,
// tools/gen_pretok_goldens.py):
//  * `\s` is Unicode White_Space, NOT python str.isspace(): U+001C..U+001F
//    are isspace-only and behave as rule-4 punctuation for the regex
//    (oracle: ".\x1c." stays a single punct run).
//  * `(?i:...)` uses Unicode simple case folding: besides ASCII letters,
//    U+017F (LATIN SMALL LETTER LONG S) folds to 's' and matches the 's
//    contraction (oracle: "a'ſx" -> ["a", "'ſ", "x"]). No other
//    non-ASCII simple fold hits these letters; multi-char full folds
//    (ss -> ß etc.) do not apply.

#include "vllm/tokenizer/pretokenizer.h"

#include <cstddef>
#include <cstdint>
#include <string_view>
#include <vector>

#include "vllm/tokenizer/unicode_data.h"

namespace vllm::tok {
namespace {

struct Cp {
  uint32_t cp;
  size_t end;  // byte offset just past this codepoint
};

Cp DecodeAt(std::string_view s, size_t pos) {
  size_t p = pos;
  const uint32_t cp = DecodeUtf8(s, p);
  return {cp, p};
}

// Regex `\s` (onig): Unicode White_Space. IsWhitespace is python
// str.isspace(); the sets differ exactly on U+001C..U+001F (isspace-only).
bool IsRegexSpace(uint32_t cp) {
  return IsWhitespace(cp) && !(cp >= 0x1C && cp <= 0x1F);
}

bool IsNewlineByte(char c) { return c == '\r' || c == '\n'; }

char AsciiLower(char c) {
  return (c >= 'A' && c <= 'Z') ? static_cast<char>(c - 'A' + 'a') : c;
}

// All Match* helpers return the byte offset just past the match, or 0 if the
// rule does not match at `pos` (0 is never a valid match end since matches
// are non-empty). Precondition: pos < t.size().

// Rule 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
size_t MatchContraction(std::string_view t, size_t pos) {
  if (t[pos] != '\'' || pos + 1 >= t.size()) return 0;
  // U+017F (UTF-8 C5 BF) simple-case-folds to 's'.
  if (pos + 3 <= t.size() && static_cast<unsigned char>(t[pos + 1]) == 0xC5 &&
      static_cast<unsigned char>(t[pos + 2]) == 0xBF) {
    return pos + 3;
  }
  const char c1 = AsciiLower(t[pos + 1]);
  if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return pos + 2;
  if (pos + 2 >= t.size()) return 0;
  const char c2 = AsciiLower(t[pos + 2]);
  if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
      (c1 == 'l' && c2 == 'l')) {
    return pos + 3;
  }
  return 0;
}

// Rule 2: [^\r\n\p{L}\p{N}]?[\p{L}\p{M}]+   (Qwen3.6; marks_in_run=true)
//         [^\r\n\p{L}\p{N}]?\p{L}+          (Llama-3; marks_in_run=false)
// One optional prefix codepoint (anything but \r, \n, letters, numbers),
// then a maximal run of letters (and marks, for Qwen). When the first
// codepoint is itself a run character, taking it as prefix vs. run start
// yields the same match end, so we always treat it as run start.
size_t MatchLetterRun(std::string_view t, size_t pos, bool marks_in_run) {
  const auto is_run = [marks_in_run](uint32_t cp) {
    const UCat c = Category(cp);
    return c == UCat::kLetter || (marks_in_run && c == UCat::kMark);
  };
  const Cp c0 = DecodeAt(t, pos);
  size_t run = pos;
  if (!is_run(c0.cp)) {
    const UCat cat = Category(c0.cp);
    if (c0.cp == U'\r' || c0.cp == U'\n' || cat == UCat::kLetter ||
        cat == UCat::kNumber || c0.end >= t.size()) {
      return 0;
    }
    if (!is_run(DecodeAt(t, c0.end).cp)) return 0;
    run = c0.end;  // prefix consumed
  }
  size_t p = run;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (!is_run(c.cp)) break;
    p = c.end;
  }
  return p;
}

// The CASE-AWARE letter classes:
//   U = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]      L = [\p{Ll}\p{Lm}\p{Lo}\p{M}]
// SHARED by kTekken and kGpt4o. Both regexes build their two word
// alternatives out of exactly this pair -- `U* L+` and `U+ L*`, each behind an
// optional [^\r\n\p{L}\p{N}]? prefix -- and differ only AROUND them (kGpt4o
// keeps o200k's (?i:'s|'t|...) suffix and \p{N}{1,3}; kTekken drops both).
// They are one definition on purpose: two copies of a character class are two
// things that can disagree, and a disagreement here mistokenizes silently.
// The two classes OVERLAP on {Lm, Lo, M}, so `U* L+` genuinely backtracks and
// cannot be folded into MatchLetterRun's single-predicate scan.
bool InUpperLetterClass(uint32_t cp) {
  if (Category(cp) == UCat::kMark) return true;  // \p{M} is in BOTH classes
  const LetterCase lc = LetterCaseOf(cp);
  return lc == LetterCase::kUpper || lc == LetterCase::kCaseless;
}

bool InLowerLetterClass(uint32_t cp) {
  if (Category(cp) == UCat::kMark) return true;  // \p{M} is in BOTH classes
  const LetterCase lc = LetterCaseOf(cp);
  return lc == LetterCase::kLower || lc == LetterCase::kCaseless;
}

// Scans a maximal run of `pred` from `from`, recording every codepoint
// boundary so the caller can give characters back (regex backtracking).
void ScanRun(std::string_view t, size_t from, bool (*pred)(uint32_t),
             std::vector<size_t>& stops) {
  stops.clear();
  stops.push_back(from);
  size_t p = from;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (!pred(c.cp)) break;
    p = c.end;
    stops.push_back(p);
  }
}

// `upper_first_required`: alternative 2 needs [U]+ then [L]*; alternative 1
// needs [U]* then [L]+. Returns the match end, or 0 for no match.
size_t MatchTekkenAlt(std::string_view t, size_t from, bool upper_first_required,
                      std::vector<size_t>& scratch) {
  ScanRun(t, from, InUpperLetterClass, scratch);
  // [U]* / [U]+ are greedy, so walk the give-back positions longest-first;
  // the FIRST that lets the second half match is what the regex engine picks.
  for (size_t i = scratch.size(); i-- > 0;) {
    const size_t split = scratch[i];
    const size_t upper_len = i;  // codepoints consumed by the upper run
    if (upper_first_required && upper_len == 0) break;  // [U]+ needs >= 1
    size_t p = split;
    size_t lower_len = 0;
    while (p < t.size()) {
      const Cp c = DecodeAt(t, p);
      if (!InLowerLetterClass(c.cp)) break;
      p = c.end;
      ++lower_len;
    }
    if (!upper_first_required && lower_len == 0) continue;  // [L]+ needs >= 1
    if (p > from) return p;
  }
  return 0;
}

size_t MatchTekkenLetterRun(std::string_view t, size_t pos,
                            std::vector<size_t>& scratch) {
  // Alternation is ORDERED and `X?` is greedy, so the engine tries, in order:
  // alt1-with-prefix, alt1-without, alt2-with-prefix, alt2-without.
  for (int alt = 0; alt < 2; ++alt) {
    const bool upper_first_required = alt == 1;
    for (int use_prefix = 1; use_prefix >= 0; --use_prefix) {
      size_t start = pos;
      if (use_prefix != 0) {
        const Cp c0 = DecodeAt(t, pos);
        const UCat cat = Category(c0.cp);
        if (c0.cp == U'\r' || c0.cp == U'\n' || cat == UCat::kLetter ||
            cat == UCat::kNumber || c0.end >= t.size()) {
          continue;  // this codepoint cannot serve as the optional prefix
        }
        start = c0.end;
      }
      const size_t end = MatchTekkenAlt(t, start, upper_first_required, scratch);
      if (end != 0) return end;
    }
  }
  return 0;
}

// Rule 3: \p{N} (Qwen, max_digits=1) | \p{N}{1,3} (Llama-3, max_digits=3;
// greedy, so long digit runs split into groups of three from the left).
size_t MatchNumbers(std::string_view t, size_t pos, int max_digits) {
  size_t p = pos;
  int count = 0;
  while (p < t.size() && count < max_digits) {
    const Cp c = DecodeAt(t, p);
    if (Category(c.cp) != UCat::kNumber) break;
    p = c.end;
    ++count;
  }
  return count > 0 ? p : 0;
}

// Rule 4: ` ?[^\s\p{L}\p{M}\p{N}]+[\r\n]*`  (Qwen3.6; marks_excluded=true)
//         ` ?[^\s\p{L}\p{N}]+[\r\n]*`       (Llama-3; marks_excluded=false)
//         ` ?[^\s\p{L}\p{N}]+[\r\n/]*`      (Tekken and GPT-4o, which share
//                                            this rule byte-for-byte;
//                                            slash_in_tail=true)
// Optional single literal ASCII space, then >=1 codepoints that are not
// regex-space/letters/numbers (nor marks, for Qwen), then any trailing \r/\n
// bytes -- and '/' too for Tekken and GPT-4o. If only the space matched, the
// rule fails as a whole.
// as a whole.
size_t MatchPunctRun(std::string_view t, size_t pos, bool marks_excluded,
                     bool slash_in_tail) {
  size_t p = pos;
  if (t[p] == ' ') ++p;
  const size_t run_begin = p;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (IsRegexSpace(c.cp)) break;
    const UCat cat = Category(c.cp);
    if (cat == UCat::kLetter || cat == UCat::kNumber ||
        (marks_excluded && cat == UCat::kMark)) {
      break;
    }
    p = c.end;
  }
  if (p == run_begin) return 0;
  while (p < t.size() &&
         (IsNewlineByte(t[p]) || (slash_in_tail && t[p] == '/'))) {
    ++p;
  }
  return p;
}

// Rule 5: \s*[\r\n]+
// Backtracking semantics: greedy \s* over the whitespace run, then [\r\n]+
// backtracks to the LAST \r/\n in the run, consuming exactly it. Net effect:
// match [pos, last_newline_in_run + 1) iff the run contains a newline;
// trailing non-newline whitespace is left for later rules.
size_t MatchWsNewlines(std::string_view t, size_t pos) {
  size_t p = pos;
  size_t after_last_nl = 0;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (!IsRegexSpace(c.cp)) break;
    if (c.cp == U'\r' || c.cp == U'\n') after_last_nl = c.end;
    p = c.end;
  }
  return after_last_nl;  // 0 when the run has no \r/\n (or no run at all)
}

// Rule 6: \s+(?!\S)
// Whitespace run not followed by a non-space: the full run matches at end of
// input; otherwise the last whitespace codepoint is left attached to the
// following token (it becomes e.g. the ` ?` prefix of rules 2/4), so the
// match is the run minus its last codepoint — failing entirely when the run
// is a single codepoint.
size_t MatchWsNotBeforeNonSpace(std::string_view t, size_t pos) {
  size_t p = pos;
  size_t last_begin = pos;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (!IsRegexSpace(c.cp)) break;
    last_begin = p;
    p = c.end;
  }
  if (p == pos) return 0;      // no whitespace at all
  if (p == t.size()) return p; // (?!\S) holds at end of input
  return last_begin == pos ? 0 : last_begin;
}

// Rule 7: \s+  (maximal whitespace run; the catch-all for e.g. a single
// space before a digit, which no earlier rule absorbs).
size_t MatchWs(std::string_view t, size_t pos) {
  size_t p = pos;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (!IsRegexSpace(c.cp)) break;
    p = c.end;
  }
  return p > pos ? p : 0;
}

// ---------------------------------------------------------------------------
// GPT-4o / o200k rules (llama.cpp's LLAMA_VOCAB_PRE_TYPE_GPT4O; GGUF pre names
// "gpt-4o", "llama4", "kanana2", "talkie" — llama.cpp/src/llama-vocab.cpp:
// 2294-2299 @ 153d324bcf).
//
// The pattern is transcribed VERBATIM from the checkpoint that forced it in,
// $CHECKPOINT_ROOT/muse-glimmer-30b/tokenizer.json (read 2026-08-11),
// pre_tokenizer.pretokenizers[0].pattern.Regex, behavior=Isolated. It is the
// same string llama.cpp records as "original regex from tokenizer.json" above
// its own GPT4O entry (llama.cpp/src/llama-vocab.cpp:432 @ 153d324bcf):
//
//   [^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?
//  |\p{N}{1,3}
//  | ?[^\s\p{L}\p{N}]+[\r\n/]*
//  |\s*[\r\n]+
//  |\s+(?!\S)
//  |\s+
//
// WHY THE ORIGINAL AND NOT llama.cpp's TRANSCRIPTION (llama-vocab.cpp:433):
// llama.cpp cannot spell the minor letter categories in its regex engine, so
// it approximates [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}] as `((?=[\p{L}])([^a-z]))`
// and [\p{Ll}\p{Lm}\p{Lo}\p{M}] as `((?=[\p{L}])([^A-Z]))` — "a letter that is
// not an ASCII lowercase/uppercase char". Those classes are NOT the same sets:
// they drop \p{M} entirely and they put every non-ASCII cased letter in BOTH.
// Verified divergences against the HF tokenizers oracle on this checkpoint's
// own regex (tools/gen_pretok_goldens.py engine):
//     "éÉ"  (U+00E9 U+00C9)  original ["é","É"]  llama.cpp ["éÉ"]
//     "éX" (e U+0301 X)      original ["é","X"]    llama.cpp ["e","́X"]
// The BPE merges were trained under the ORIGINAL regex, so the original is
// what reproduces the checkpoint's token ids; llama.cpp's approximation is a
// regex-engine limitation, not the reference behavior. We therefore port the
// pattern llama.cpp itself cites as authoritative and record the difference
// here rather than reproducing its approximation error.
//
// Structurally this is NOT kLlama3 with different digits. Three differences,
// each of which changes token ids on ordinary text:
//   1. ONE letter alternative becomes TWO, over the minor categories:
//        A = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]   B = [\p{Ll}\p{Lm}\p{Lo}\p{M}]
//      alternative 1 is `A* B+`, alternative 2 is `A+ B*`. Together they split
//      a CamelCase word at each lower->upper boundary ("HTMLParser" stays
//      whole, but "abcDEF" -> ["abc","DEF"]), which \p{L}+ never does.
//   2. The contraction is a SUFFIX of the word alternatives, not a leading
//      alternative of its own. "don't" is ONE piece here; kLlama3 gives
//      ["don","'t"].
//   3. The punctuation run absorbs a trailing `/` as well as \r/\n.
//
// The scanner below is a direct transcription of onig's leftmost-first
// backtracking for those two alternatives (greedy `?`, `*` and `+`, first
// success wins), not a re-derivation: every expectation in
// tests/vllm/test_pretokenizer.cpp and every row of the golden table was
// produced by running this exact regex through HF tokenizers.

// The `A` and `B` named below are `InUpperLetterClass` and
// `InLowerLetterClass` from the top of this file:
//   A = [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]   B = [\p{Ll}\p{Lm}\p{Lo}\p{M}]
// kTekken's regex spells those two classes identically, so this scanner reuses
// its predicates rather than carrying a second copy that could drift. Lm, Lo
// and M are in BOTH classes; only Lu/Lt and Ll separate them, which is why the
// scan below has to backtrack.

// `[^\r\n\p{L}\p{N}]?` — one optional codepoint that is neither a newline nor
// a letter nor a number. Returns the offset past it, or 0 if it cannot match.
// NOTE a combining mark passes this class AND is in A and B, which is exactly
// why the alternatives below have to be backtracked rather than scanned once.
size_t MatchGpt4oWordPrefix(std::string_view t, size_t pos) {
  const Cp c = DecodeAt(t, pos);
  if (c.cp == U'\r' || c.cp == U'\n') return 0;
  const UCat cat = Category(c.cp);
  if (cat == UCat::kLetter || cat == UCat::kNumber) return 0;
  return c.end;
}

// The shared `(?i:'s|'t|'re|'ve|'m|'ll|'d)?` tail. Greedy and last in the
// alternative, so it never backtracks.
size_t ApplyGpt4oContractionSuffix(std::string_view t, size_t end) {
  if (end >= t.size()) return end;
  const size_t c = MatchContraction(t, end);
  return c == 0 ? end : c;
}

// Alternative 1: `[^\r\n\p{L}\p{N}]? A* B+ (?i:...)?`
//
// onig tries the optional prefix present-first, then greedy `A*`, then `B+`
// which needs at least one codepoint. When `B+` fails at the end of the A run,
// `A*` gives codepoints back one at a time — so the B run starts at the LAST
// position in [run start, run end] whose codepoint is in B. If there is none,
// the whole alternative retries without the prefix, then fails.
size_t MatchGpt4oWordLower(std::string_view t, size_t pos) {
  for (int with_prefix = 1; with_prefix >= 0; --with_prefix) {
    size_t q = pos;
    if (with_prefix != 0) {
      const size_t pe = MatchGpt4oWordPrefix(t, pos);
      if (pe == 0 || pe >= t.size()) continue;
      q = pe;
    }
    // Greedy `A*`, remembering the last A codepoint that is also in B (that is
    // the backtrack target, since A \ B codepoints can never start `B+`).
    size_t p = q;
    size_t last_a_in_b = std::string_view::npos;
    while (p < t.size()) {
      const Cp c = DecodeAt(t, p);
      if (!InUpperLetterClass(c.cp)) break;
      if (InLowerLetterClass(c.cp)) last_a_in_b = p;
      p = c.end;
    }
    size_t b_begin;
    if (p < t.size() && InLowerLetterClass(DecodeAt(t, p).cp)) {
      b_begin = p;  // `A*` kept the whole run; `B+` starts right after it
    } else if (last_a_in_b != std::string_view::npos) {
      b_begin = last_a_in_b;  // `A*` backtracked to the last B-capable cp
    } else {
      continue;  // `B+` cannot match under this prefix choice
    }
    size_t e = b_begin;
    while (e < t.size()) {
      const Cp c = DecodeAt(t, e);
      if (!InLowerLetterClass(c.cp)) break;
      e = c.end;
    }
    return ApplyGpt4oContractionSuffix(t, e);
  }
  return 0;
}

// Alternative 2: `[^\r\n\p{L}\p{N}]? A+ B* (?i:...)?`
// `B*` and the contraction can both match empty, so greedy `A+` never has to
// give anything back; only the optional prefix is retried.
size_t MatchGpt4oWordUpper(std::string_view t, size_t pos) {
  for (int with_prefix = 1; with_prefix >= 0; --with_prefix) {
    size_t q = pos;
    if (with_prefix != 0) {
      const size_t pe = MatchGpt4oWordPrefix(t, pos);
      if (pe == 0 || pe >= t.size()) continue;
      q = pe;
    }
    size_t p = q;
    while (p < t.size()) {
      const Cp c = DecodeAt(t, p);
      if (!InUpperLetterClass(c.cp)) break;
      p = c.end;
    }
    if (p == q) continue;  // `A+` needs at least one codepoint
    size_t e = p;
    while (e < t.size()) {
      const Cp c = DecodeAt(t, e);
      if (!InLowerLetterClass(c.cp)) break;
      e = c.end;
    }
    return ApplyGpt4oContractionSuffix(t, e);
  }
  return 0;
}

std::vector<std::pair<size_t, size_t>> PretokenizeGpt4o(std::string_view text) {
  std::vector<std::pair<size_t, size_t>> spans;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t end = MatchGpt4oWordLower(text, pos);
    if (end == 0) end = MatchGpt4oWordUpper(text, pos);
    if (end == 0) end = MatchNumbers(text, pos, /*max_digits=*/3);
    if (end == 0) {
      end = MatchPunctRun(text, pos, /*marks_excluded=*/false,
                          /*slash_in_tail=*/true);
    }
    if (end == 0) end = MatchWsNewlines(text, pos);
    if (end == 0) end = MatchWsNotBeforeNonSpace(text, pos);
    if (end == 0) end = MatchWs(text, pos);
    // Unreachable (the rules cover every codepoint class), but guarantee
    // forward progress rather than looping if the tables ever change.
    if (end == 0) end = DecodeAt(text, pos).end;
    spans.emplace_back(pos, end);
    pos = end;
  }
  return spans;
}

// ---------------------------------------------------------------------------
// GPT-2 rules. The ORIGINAL byte-level BPE split, applied implicitly by HF's
// ByteLevel pre-tokenizer when `use_regex: true` and no explicit Split
// component is present (facebook/opt-125m, gpt2, and friends):
//
//   's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
//
// Four of the six alternatives differ from the Qwen/Llama-3 family above, so
// they get their own matchers rather than extra flags on the existing ones.

// GPT-2 rule 1: 's|'t|'re|'ve|'m|'ll|'d — CASE-SENSITIVE (no `(?i:)` wrapper),
// so ASCII lowercase only and no U+017F long-s fold.
size_t MatchGpt2Contraction(std::string_view t, size_t pos) {
  if (t[pos] != '\'' || pos + 1 >= t.size()) return 0;
  const char c1 = t[pos + 1];
  if (c1 == 's' || c1 == 't' || c1 == 'm' || c1 == 'd') return pos + 2;
  if (pos + 2 >= t.size()) return 0;
  const char c2 = t[pos + 2];
  if ((c1 == 'r' && c2 == 'e') || (c1 == 'v' && c2 == 'e') ||
      (c1 == 'l' && c2 == 'l')) {
    return pos + 3;
  }
  return 0;
}

// GPT-2 rules 2 and 3: ` ?\p{L}+` and ` ?\p{N}+`.
// An optional single literal ASCII space, then a MAXIMAL run of the wanted
// category. Note both differences vs the Qwen form: the prefix is a plain space
// (not "any non-letter/number codepoint"), and digit runs are unbounded rather
// than grouped.
size_t MatchGpt2CategoryRun(std::string_view t, size_t pos, UCat want) {
  size_t p = pos;
  if (t[p] == ' ') ++p;
  const size_t run_begin = p;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (Category(c.cp) != want) break;
    p = c.end;
  }
  return p == run_begin ? 0 : p;
}

// GPT-2 rule 4: ` ?[^\s\p{L}\p{N}]+` — as MatchPunctRun but with NO trailing
// `[\r\n]*` (GPT-2 has no newline-absorbing tail and no `\s*[\r\n]+` rule, so
// newlines fall through to the two whitespace rules).
size_t MatchGpt2PunctRun(std::string_view t, size_t pos) {
  size_t p = pos;
  if (t[p] == ' ') ++p;
  const size_t run_begin = p;
  while (p < t.size()) {
    const Cp c = DecodeAt(t, p);
    if (IsRegexSpace(c.cp)) break;
    const UCat cat = Category(c.cp);
    if (cat == UCat::kLetter || cat == UCat::kNumber) break;
    p = c.end;
  }
  return p == run_begin ? 0 : p;
}

std::vector<std::pair<size_t, size_t>> PretokenizeGpt2(std::string_view text) {
  std::vector<std::pair<size_t, size_t>> spans;
  size_t pos = 0;
  while (pos < text.size()) {
    size_t end = MatchGpt2Contraction(text, pos);
    if (end == 0) end = MatchGpt2CategoryRun(text, pos, UCat::kLetter);
    if (end == 0) end = MatchGpt2CategoryRun(text, pos, UCat::kNumber);
    if (end == 0) end = MatchGpt2PunctRun(text, pos);
    if (end == 0) end = MatchWsNotBeforeNonSpace(text, pos);
    if (end == 0) end = MatchWs(text, pos);
    if (end == 0) end = DecodeAt(text, pos).end;  // forward-progress guarantee
    spans.emplace_back(pos, end);
    pos = end;
  }
  return spans;
}

// ---------------------------------------------------------------------------
// DeepSeek-V2 rules (DeepSeek-V2, V2-Lite; GGUF pre `deepseek-llm`).
// MLA campaign W8. This header used to name V3 as well and it was wrong:
// V3 is a different pipeline, PretokenizeDeepSeekV3 below (#1924).
//
// This family is NOT another alternation regex. Its tokenizer.json declares a HF
// `Sequence` PIPELINE of seven pre-tokenizers, and HF applies them in order,
// each one further splitting the pieces the previous one produced:
//
//   0. Split(Regex("[\r\n]"),        Isolated)
//   1. Split(Regex("\s?[<cased letters>]+"), Isolated)
//   2. Split(Regex("\s?[<ascii + fullwidth + CJK punctuation>]+"), Isolated)
//   3. Split(Regex("\s+$"),          Isolated)
//   4. Split(Regex("[<CJK ideographs + Hangul>]+"), Isolated)
//   5. Digits(individual_digits=true)
//   6. ByteLevel(add_prefix_space=false, trim_offsets=true, use_regex=false)
//
// Two consequences worth stating, because they are what make a single-pass
// alternation scanner the WRONG shape here:
//  * stage 2's class `[!-/:-~…]+` spans 0x3A-0x7E, which CONTAINS A-Z and a-z.
//    It is only correct because stage 1 already isolated the letter runs. Order
//    is load-bearing, so the implementation is a genuine pipeline.
//  * stage 3's `\s+$` anchors to the end of the PIECE. `$` in onig is
//    end-of-LINE, not end-of-string — but stage 0 has already isolated every
//    \r and \n into its own piece, so no piece reaching stage 3 contains a
//    newline and the two readings coincide. (Recorded rather than relied on
//    silently.)
//
// The classes are ENUMERATED codepoint ranges in the checkpoint, not `\p{...}`
// properties, so they are transcribed here verbatim from
// deepseek-ai/DeepSeek-V2-Lite tokenizer.json (snapshot 604d5664, read
// 2026-07-22) and the tokenizer loader compares the checkpoint's regex strings
// against the same verbatim patterns before selecting this family — a DeepSeek
// variant that ships different ranges is REFUSED loudly instead of being
// mis-tokenized.

// Stage 1: "\s?[A-Za-zµÀ-Ö…]+" — 87 ranges, ascending and disjoint as shipped.
constexpr uint32_t kDsLetterRanges[][2] = {
    {0x0041, 0x005A}, {0x0061, 0x007A}, {0x00B5, 0x00B5}, {0x00C0, 0x00D6},
    {0x00D8, 0x00F6}, {0x00F8, 0x01BA}, {0x01BC, 0x01BF}, {0x01C4, 0x0293},
    {0x0295, 0x02AF}, {0x0370, 0x0373}, {0x0376, 0x0376}, {0x0377, 0x0377},
    {0x037B, 0x037D}, {0x037F, 0x037F}, {0x0386, 0x0386}, {0x0388, 0x038A},
    {0x038C, 0x038C}, {0x038E, 0x03A1}, {0x03A3, 0x03F5}, {0x03F7, 0x0481},
    {0x048A, 0x052F}, {0x0531, 0x0556}, {0x10A0, 0x10C5}, {0x13A0, 0x13F5},
    {0x13F8, 0x13FD}, {0x1C90, 0x1CBA}, {0x1CBD, 0x1CBF}, {0x1D00, 0x1D2B},
    {0x1D6B, 0x1D77}, {0x1D79, 0x1D9A}, {0x1E00, 0x1F15}, {0x1F18, 0x1F1D},
    {0x1F20, 0x1F45}, {0x1F48, 0x1F4D}, {0x1F50, 0x1F57}, {0x1F59, 0x1F59},
    {0x1F5B, 0x1F5B}, {0x1F5D, 0x1F5D}, {0x1F5F, 0x1F7D}, {0x1F80, 0x1FB4},
    {0x1FB6, 0x1FBC}, {0x1FBE, 0x1FBE}, {0x1FC2, 0x1FC4}, {0x1FC6, 0x1FCC},
    {0x1FD0, 0x1FD3}, {0x1FD6, 0x1FDB}, {0x1FE0, 0x1FEC}, {0x1FF2, 0x1FF4},
    {0x1FF6, 0x1FFC}, {0x2102, 0x2102}, {0x2107, 0x2107}, {0x210A, 0x2113},
    {0x2115, 0x2115}, {0x2119, 0x211D}, {0x2124, 0x2124}, {0x2126, 0x2126},
    {0x2128, 0x2128}, {0x212A, 0x212D}, {0x212F, 0x2134}, {0x2139, 0x2139},
    {0x213C, 0x213F}, {0x2145, 0x2149}, {0x214E, 0x214E}, {0x2183, 0x2183},
    {0x2184, 0x2184}, {0x2C00, 0x2C7B}, {0x2C7E, 0x2CE4}, {0x2CEB, 0x2CEE},
    {0x2CF2, 0x2CF2}, {0x2CF3, 0x2CF3}, {0xA640, 0xA66D}, {0xA680, 0xA69B},
    {0xA722, 0xA76F}, {0xA771, 0xA787}, {0xA78B, 0xA78E}, {0xAB70, 0xABBF},
    {0xFB00, 0xFB06}, {0xFB13, 0xFB17}, {0xFF21, 0xFF3A}, {0xFF41, 0xFF5A},
    {0x10400, 0x1044F}, {0x104B0, 0x104D3}, {0x104D8, 0x104FB},
    {0x10C80, 0x10CB2}, {0x10CC0, 0x10CF2}, {0x118A0, 0x118DF},
    {0x1E900, 0x1E943},
};

// Stage 2: "\s?[!-/:-~！-／：-～‘-‟　-。]+", sorted here (the checkpoint lists the
// two General-Punctuation/CJK ranges after the fullwidth ones).
constexpr uint32_t kDsPunctRanges[][2] = {
    {0x0021, 0x002F}, {0x003A, 0x007E}, {0x2018, 0x201F},
    {0x3000, 0x3002}, {0xFF01, 0xFF0F}, {0xFF1A, 0xFF5E},
};

// Stage 4: "[一-龥ࠀ-一가-퟿]+" = {0x4E00-0x9FA5, 0x0800-0x4E00, 0xAC00-0xD7FF}.
// The first two OVERLAP at 0x4E00 and are contiguous, so they are merged here
// into 0x0800-0x9FA5; the raw shipped form is recorded above.
constexpr uint32_t kDsCjkRanges[][2] = {
    {0x0800, 0x9FA5},
    {0xAC00, 0xD7FF},
};

template <size_t N>
bool InRanges(uint32_t cp, const uint32_t (&ranges)[N][2]) {
  size_t lo = 0, hi = N;
  while (lo < hi) {
    const size_t mid = lo + (hi - lo) / 2;
    if (cp < ranges[mid][0]) {
      hi = mid;
    } else if (cp > ranges[mid][1]) {
      lo = mid + 1;
    } else {
      return true;
    }
  }
  return false;
}

using Span = std::pair<size_t, size_t>;

// One HF `Split(behavior=Isolated)` stage. For every current piece, scan left to
// right; at each position ask `match` for the end of a match STARTING THERE
// (0 = no match, and matches are never empty). A match is emitted as its own
// piece and the unmatched text before it is emitted as a piece too — that is
// exactly what "Isolated" means. Pieces with no match at all pass through whole.
template <typename Match>
void ApplySplitIsolated(std::string_view text, std::vector<Span>& pieces,
                        Match match) {
  std::vector<Span> out;
  out.reserve(pieces.size());
  for (const Span& piece : pieces) {
    size_t gap_begin = piece.first;
    size_t pos = piece.first;
    while (pos < piece.second) {
      const size_t end = match(text, pos, piece.second);
      if (end == 0) {
        pos = DecodeAt(text, pos).end;
        continue;
      }
      if (pos > gap_begin) out.emplace_back(gap_begin, pos);
      out.emplace_back(pos, end);
      pos = end;
      gap_begin = pos;
    }
    if (piece.second > gap_begin) out.emplace_back(gap_begin, piece.second);
  }
  pieces.swap(out);
}

// `Digits(individual_digits=true)`: HF splits on the digit predicate with
// Isolated behaviour, so EACH digit codepoint becomes its own piece and the
// non-digit stretches between them stay whole. Rust's `char::is_numeric()` is
// the Unicode N category — our UCat::kNumber.
void ApplyDigitsIndividual(std::string_view text, std::vector<Span>& pieces) {
  std::vector<Span> out;
  out.reserve(pieces.size());
  for (const Span& piece : pieces) {
    size_t gap_begin = piece.first;
    size_t pos = piece.first;
    while (pos < piece.second) {
      const Cp c = DecodeAt(text, pos);
      if (Category(c.cp) == UCat::kNumber) {
        if (pos > gap_begin) out.emplace_back(gap_begin, pos);
        out.emplace_back(pos, c.end);
        gap_begin = c.end;
      }
      pos = c.end;
    }
    if (piece.second > gap_begin) out.emplace_back(gap_begin, piece.second);
  }
  pieces.swap(out);
}

// Stage 0: `[\r\n]` — a single newline codepoint.
size_t MatchDsNewline(std::string_view t, size_t pos, size_t /*piece_end*/) {
  return IsNewlineByte(t[pos]) ? pos + 1 : 0;
}

// Stages 1/2: `\s?[CLASS]+` — one OPTIONAL whitespace codepoint (any Unicode
// White_Space, not just ASCII space), then a maximal run of class members.
template <size_t N>
size_t MatchDsWsClassRun(std::string_view t, size_t pos, size_t piece_end,
                         const uint32_t (&ranges)[N][2]) {
  size_t p = pos;
  const Cp c0 = DecodeAt(t, p);
  if (c0.end <= piece_end && IsRegexSpace(c0.cp) && !InRanges(c0.cp, ranges)) {
    p = c0.end;
  }
  const size_t run_begin = p;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || !InRanges(c.cp, ranges)) break;
    p = c.end;
  }
  return p == run_begin ? 0 : p;
}

// Stage 3: `\s+$` — a whitespace run that reaches the END of the piece.
size_t MatchDsTrailingWs(std::string_view t, size_t pos, size_t piece_end) {
  size_t p = pos;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || !IsRegexSpace(c.cp)) break;
    p = c.end;
  }
  return (p > pos && p == piece_end) ? p : 0;
}

// Stage 4: `[CLASS]+` — a maximal class run with NO whitespace prefix.
template <size_t N>
size_t MatchDsClassRun(std::string_view t, size_t pos, size_t piece_end,
                       const uint32_t (&ranges)[N][2]) {
  size_t p = pos;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || !InRanges(c.cp, ranges)) break;
    p = c.end;
  }
  return p == pos ? 0 : p;
}

std::vector<Span> PretokenizeDeepSeek(std::string_view text) {
  std::vector<Span> pieces;
  if (text.empty()) return pieces;
  pieces.emplace_back(0, text.size());
  ApplySplitIsolated(text, pieces, MatchDsNewline);
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t e) {
    return MatchDsWsClassRun(t, p, e, kDsLetterRanges);
  });
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t e) {
    return MatchDsWsClassRun(t, p, e, kDsPunctRanges);
  });
  ApplySplitIsolated(text, pieces, MatchDsTrailingWs);
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t e) {
    return MatchDsClassRun(t, p, e, kDsCjkRanges);
  });
  ApplyDigitsIndividual(text, pieces);
  // Stage 6 is ByteLevel(use_regex=false): it maps bytes, it does not split.
  return pieces;
}

// ---------------------------------------------------------------------------
// DeepSeek-V3 rules (DeepSeek-V3, DeepSeek-R1, DeepSeek-V4-Flash). Issue #1924.
//
// A SEPARATE FAMILY from the DeepSeek-V2 rules above, not a variant of them.
// llama.cpp keeps them as two cases of one switch --
// LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM and LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM,
// src/llama-vocab.cpp:308-325 @ b10451 (10bf611e5, the pinned llama.cpp
// oracle) -- with disjoint regex lists, and so do we. Nothing about the V2
// pipeline carries over: there is no `Digits` stage, no `\s+$` stage, no Hangul
// in the CJK class, and the word and punctuation rules are `\p{...}` PROPERTIES
// where V2 enumerates codepoints.
//
// It is a HF `Sequence` PIPELINE of four pre-tokenizers, applied in order, each
// further splitting the pieces the previous one produced:
//
//   0. Split(Regex("\p{N}{1,3}"),               Isolated)
//   1. Split(Regex("[<kana + CJK ideographs>]+"), Isolated)
//   2. Split(Regex(<the six-alternative regex below>), Isolated)
//   3. ByteLevel(add_prefix_space=false, trim_offsets=true, use_regex=false)
//
// Transcribed VERBATIM from the DeepSeek-V4-Flash checkpoint's own
// tokenizer.json (`0xSero/deepseek-v4-flash-0731-spark`, EXL3 snapshot on the
// NAS, sha256 8f9f37ca…33cf, read 2026-08-25); the exact bytes are committed at
// tests/parity/goldens/tokenizer_deepseek_v3/tokenizer.json and the loader
// compares the checkpoint's regex strings against the same three patterns
// before selecting this family, so a DeepSeek variant that ships anything else
// is REFUSED loudly instead of mis-tokenized.
//
// STAGE ORDER IS LOAD-BEARING, and differently from V2's:
//  * stage 2's alternation matches NO digit -- `\p{N}` appears in none of its
//    six alternatives -- so a digit only ever becomes its own piece because
//    stage 0 already carved it out. Run stage 2 first and a digit run comes
//    back as one undifferentiated gap.
//  * stage 2's rule-2 prefix class `[^\r\n\p{L}\p{P}\p{S}]?` MATCHES a digit,
//    so without stage 0 it would absorb the trailing digit of a run into the
//    following word: "abc123def" -> "abc" + "123def" instead of the correct
//    "abc" + "123" + "def".
//  * stage 1 does NOT have the last word. Stage 2 re-splits its output, which
//    is why "぀ゟ゠ヿ" (one stage-1 run, U+3040-U+30FF) comes back in three
//    pieces: U+3040 is unassigned, U+30A0 is \p{Pd}, and the other two are
//    \p{Lo}.
// Each of those three is a case in
// tests/vllm/test_tokenizer_parity_deepseek_v3.cpp.
//
// The `(?!\S)` in alternative 5 and the piece bounds: HF applies each `Split`
// stage to every current piece as its OWN string, so `(?!\S)` succeeds at the
// end of a PIECE, not only at the end of the input. Every Match* below is
// therefore bounded by `piece_end` rather than by `t.size()`, which is the same
// contract ApplySplitIsolated already gives the V2 stages.

// Stage 1's class, as shipped: `[一-龥぀-ゟ゠-ヿ]` = U+4E00-U+9FA5,
// U+3040-U+309F, U+30A0-U+30FF. The two kana blocks are CONTIGUOUS and are
// merged here into U+3040-U+30FF; the ranges are ascending and disjoint for
// InRanges' binary search. HANGUL IS ABSENT, which is the sharpest single
// difference from kDsCjkRanges above (V2 carries U+AC00-U+D7FF).
constexpr uint32_t kDsV3CjkRanges[][2] = {
    {0x3040, 0x30FF},
    {0x4E00, 0x9FA5},
};

// Stage 2 alternative 1's leading class, as shipped:
// `[!"#$%&'()*+,\-./:;<=>?@\[\\\]^_`{|}~]`. Those 32 characters are exactly
// ASCII 0x21-0x2F, 0x3A-0x40, 0x5B-0x60 and 0x7B-0x7E -- i.e. every ASCII
// character that is neither alphanumeric, nor space, nor a control code.
// Written as ranges because the enumeration and the ranges were checked equal
// as SETS, and ranges cannot pick up a transcription typo in a 32-character
// list.
bool IsDsV3AsciiPunct(uint32_t cp) {
  return (cp >= 0x21 && cp <= 0x2F) || (cp >= 0x3A && cp <= 0x40) ||
         (cp >= 0x5B && cp <= 0x60) || (cp >= 0x7B && cp <= 0x7E);
}

// Stage 0: `\p{N}{1,3}` -- greedy, at most three \p{N} codepoints. Greedy and
// leftmost, so a long run splits into groups of three FROM THE LEFT
// ("1234567890" -> 123|456|789|0), which is the same grouping kLlama3 and
// kGpt4o get from their own `\p{N}{1,3}` alternative.
size_t MatchDsV3DigitGroup(std::string_view t, size_t pos, size_t piece_end) {
  size_t p = pos;
  for (int taken = 0; taken < 3 && p < piece_end; ++taken) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || Category(c.cp) != UCat::kNumber) break;
    p = c.end;
  }
  return p == pos ? 0 : p;
}

// Stage 2 alternative 1: `[<ascii punct>][A-Za-z]+`. One punctuation character
// bound to the ASCII letters that follow it, so "(x" and "$var" are single
// pieces. No other family in this file has this rule, and alternative 3 would
// otherwise give "(" + "x".
size_t MatchDsV3PunctWord(std::string_view t, size_t pos, size_t piece_end) {
  const Cp c0 = DecodeAt(t, pos);
  if (c0.end > piece_end || !IsDsV3AsciiPunct(c0.cp)) return 0;
  size_t p = c0.end;
  const size_t run_begin = p;
  while (p < piece_end) {
    const char c = t[p];
    if (!((c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z'))) break;
    ++p;
  }
  return p == run_begin ? 0 : p;
}

// `[\p{L}\p{M}]` -- alternative 2's run class.
bool InDsV3LetterOrMark(uint32_t cp) {
  const UCat cat = Category(cp);
  return cat == UCat::kLetter || cat == UCat::kMark;
}

// `[^\r\n\p{L}\p{P}\p{S}]` -- alternative 2's optional prefix class. Note that
// it contains \p{M}, \p{N}, \p{Z} and the control codes, which is why the
// alternative genuinely BACKTRACKS: a combining mark passes this class AND the
// run class above.
bool InDsV3WordPrefixClass(uint32_t cp) {
  if (cp == '\r' || cp == '\n') return false;
  const UCat cat = Category(cp);
  return cat != UCat::kLetter && cat != UCat::kPunct && cat != UCat::kSymbol;
}

// Stage 2 alternative 2: `[^\r\n\p{L}\p{P}\p{S}]?[\p{L}\p{M}]+`.
// onig tries the greedy `?` PRESENT first; if no letter/mark run follows the
// prefix the alternative retries with the prefix absent, and only then fails.
// That retry is not decoration: for "<mark><space>" the present-first attempt
// fails and the retry matches the mark alone.
size_t MatchDsV3Word(std::string_view t, size_t pos, size_t piece_end) {
  const auto run_from = [&](size_t from) {
    size_t p = from;
    while (p < piece_end) {
      const Cp c = DecodeAt(t, p);
      if (c.end > piece_end || !InDsV3LetterOrMark(c.cp)) break;
      p = c.end;
    }
    return p;
  };
  const Cp c0 = DecodeAt(t, pos);
  if (c0.end <= piece_end && InDsV3WordPrefixClass(c0.cp)) {
    const size_t end = run_from(c0.end);
    if (end > c0.end) return end;
  }
  const size_t end = run_from(pos);
  return end == pos ? 0 : end;
}

// Stage 2 alternative 3: ` ?[\p{P}\p{S}]+[\r\n]*`. The optional prefix is the
// LITERAL ASCII space, not `\s` -- a tab or a NBSP before punctuation is left
// to the whitespace alternatives. \p{P} and \p{S} are BOTH in the run class, so
// "€5"'s currency sign and "+"'s math sign land in the same rule as ",".
size_t MatchDsV3PunctRun(std::string_view t, size_t pos, size_t piece_end) {
  size_t p = pos;
  if (t[p] == ' ') ++p;
  const size_t run_begin = p;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end) break;
    const UCat cat = Category(c.cp);
    if (cat != UCat::kPunct && cat != UCat::kSymbol) break;
    p = c.end;
  }
  if (p == run_begin) return 0;
  while (p < piece_end && IsNewlineByte(t[p])) ++p;
  return p;
}

// Stage 2 alternative 4: `\s*[\r\n]+`. Same backtracking as MatchWsNewlines
// above -- greedy `\s*` over the whitespace run, then `[\r\n]+` gives back to
// the LAST newline in it -- but bounded by the piece rather than by the input.
size_t MatchDsV3WsNewlines(std::string_view t, size_t pos, size_t piece_end) {
  size_t p = pos;
  size_t last_newline_end = 0;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || !IsRegexSpace(c.cp)) break;
    if (c.cp == '\r' || c.cp == '\n') last_newline_end = c.end;
    p = c.end;
  }
  return last_newline_end;
}

// Stage 2 alternative 5: `\s+(?!\S)`. The whole whitespace run matches at the
// END OF THE PIECE (see the header note: HF hands each stage the piece as its
// own string, so `(?!\S)` is satisfied there); otherwise the last whitespace
// codepoint is given back for the next token's ` ?` prefix, and the
// alternative fails outright on a single-codepoint run.
size_t MatchDsV3WsNotBeforeNonSpace(std::string_view t, size_t pos,
                                    size_t piece_end) {
  size_t p = pos;
  size_t last_begin = pos;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || !IsRegexSpace(c.cp)) break;
    last_begin = p;
    p = c.end;
  }
  if (p == pos) return 0;
  if (p == piece_end) return p;
  return last_begin == pos ? 0 : last_begin;
}

// Stage 2 alternative 6: `\s+` -- the maximal whitespace run, the catch-all
// for e.g. the single space before a digit that no earlier alternative takes.
size_t MatchDsV3Ws(std::string_view t, size_t pos, size_t piece_end) {
  size_t p = pos;
  while (p < piece_end) {
    const Cp c = DecodeAt(t, p);
    if (c.end > piece_end || !IsRegexSpace(c.cp)) break;
    p = c.end;
  }
  return p == pos ? 0 : p;
}

// Stage 2's regex, as one ORDERED alternation: onig is leftmost-first, so the
// alternatives are tried in the order they are written and the first success
// wins. Returning 0 here is a RESULT, not a failure: the alternation covers no
// digit and no unassigned codepoint, and ApplySplitIsolated turns an unmatched
// stretch into its own piece, which is exactly what HF `Split(Isolated)` does.
size_t MatchDsV3Alternation(std::string_view t, size_t pos, size_t piece_end) {
  size_t end = MatchDsV3PunctWord(t, pos, piece_end);
  if (end == 0) end = MatchDsV3Word(t, pos, piece_end);
  if (end == 0) end = MatchDsV3PunctRun(t, pos, piece_end);
  if (end == 0) end = MatchDsV3WsNewlines(t, pos, piece_end);
  if (end == 0) end = MatchDsV3WsNotBeforeNonSpace(t, pos, piece_end);
  if (end == 0) end = MatchDsV3Ws(t, pos, piece_end);
  return end;
}

std::vector<Span> PretokenizeDeepSeekV3(std::string_view text) {
  std::vector<Span> pieces;
  if (text.empty()) return pieces;
  pieces.emplace_back(0, text.size());
  ApplySplitIsolated(text, pieces, MatchDsV3DigitGroup);
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t e) {
    return MatchDsClassRun(t, p, e, kDsV3CjkRanges);
  });
  ApplySplitIsolated(text, pieces, MatchDsV3Alternation);
  // Stage 3 is ByteLevel(use_regex=false): it maps bytes, it does not split.
  return pieces;
}

// ─── RecursiveSI Flint ───────────────────────────────────────────────────────
// llama.cpp's LLAMA_VOCAB_PRE_TYPE_FLINT (the pinned llama-cpp-flint oracle,
// PR #30928 head 1b44e30775dcb2f93952116b05f89e5bd797fb00,
// src/llama-vocab.cpp:284-289) — the pre name every published flint container
// carries (`tokenizer.ggml.pre = "flint"`). A three-regex Sequence pipeline
// exactly like kDeepSeekV3's: stage 1 `\p{N}{1,3}`, stage 2
// `[一-龥぀-ゟ゠-ヿ]+` (BYTE-IDENTICAL to kDsV3CjkRanges — U+4E00-U+9FA5 plus
// the two contiguous kana blocks U+3040-U+30FF; see its comment above), stage
// 3 one six-alternative word/punct/ws regex. Stage 3 is NOT kGpt4o: its word
// alternatives are the same two case-aware camel-split rules, but its
// punctuation class is the EXPLICIT `[\p{P}\p{S}]` (marks EXCLUDED — a mark
// after a punct starts its own piece, where kGpt4o's `[^\s\p{L}\p{N}]` run
// absorbs it) and its punct tail is `[\r\n]*` with NO `/` (kGpt4o keeps
// `[\r\n/]*`). Both differences move real pretoken boundaries, so kFlint runs
// its own stage 3 over kGpt4o's word matchers with different flags.

// Stage 3's ordered alternation (onig leftmost-first, first success wins).
size_t MatchFlintAlternation(std::string_view t, size_t pos) {
  size_t end = MatchGpt4oWordLower(t, pos);
  if (end == 0) end = MatchGpt4oWordUpper(t, pos);
  if (end == 0)
    end = MatchPunctRun(t, pos, /*marks_excluded=*/true,
                        /*slash_in_tail=*/false);
  if (end == 0) end = MatchWsNewlines(t, pos);
  if (end == 0) end = MatchWsNotBeforeNonSpace(t, pos);
  if (end == 0) end = MatchWs(t, pos);
  return end;
}

std::vector<Span> PretokenizeFlint(std::string_view text) {
  std::vector<Span> pieces;
  if (text.empty()) return pieces;
  pieces.emplace_back(0, text.size());
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t) {
    return MatchNumbers(t, p, /*max_digits=*/3);
  });
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t e) {
    return MatchDsClassRun(t, p, e, kDsV3CjkRanges);
  });
  // Stage 3 over each surviving piece. The alternation matchers are unbounded
  // matchers over a whole string, so hand each one the piece AS the string;
  // `(?!\S)` and the `$`-style behaviours then evaluate against the piece
  // boundary, which is exactly what HF Split(Isolated) means.
  ApplySplitIsolated(text, pieces, [](std::string_view t, size_t p, size_t e) {
    if (p >= e) return static_cast<size_t>(0);
    const size_t end = MatchFlintAlternation(t.substr(p, e - p), 0);
    return end == 0 ? static_cast<size_t>(0) : p + end;
  });
  return pieces;
}

}  // namespace

std::vector<std::pair<size_t, size_t>> Pretokenize(std::string_view text,
                                                   SplitPattern pattern) {
  // The three Sequence-PIPELINE families are not alternations (see above).
  // They are separate switch arms because they share no stage: V2 has seven,
  // V3 has four, Flint three, and no stage's classes appear in another.
  if (pattern == SplitPattern::kDeepSeek) return PretokenizeDeepSeek(text);
  if (pattern == SplitPattern::kDeepSeekV3) return PretokenizeDeepSeekV3(text);
  if (pattern == SplitPattern::kFlint) return PretokenizeFlint(text);
  // GPT-2's alternation differs in four of six rules, so it runs its own
  // scanner rather than threading more flags through the Qwen/Llama-3 one.
  if (pattern == SplitPattern::kGpt2) return PretokenizeGpt2(text);
  // GPT-4o splits its letter rule into two minor-category alternatives with
  // real backtracking between them (see above), so it too gets its own pass.
  if (pattern == SplitPattern::kGpt4o) return PretokenizeGpt4o(text);
  // \p{M} awareness (letter runs absorb marks; punct runs exclude them) is
  // UNIQUE to the Qwen3.6 regex. Classic Qwen2/Qwen3 and Llama-3 both treat
  // marks like ordinary punct-run codepoints. Number grouping is single-digit
  // for BOTH Qwen variants; only Llama-3 groups \p{N}{1,3}.
  // kTekken needs these two APART: marks live inside its letter classes (like
  // kQwen2) while its punct negation is [^\s\p{L}\p{N}] with no \p{M} (like
  // kLlama3). No other pattern mixes them, which is why one flag sufficed
  // until now.
  const bool tekken = pattern == SplitPattern::kTekken;
  const bool marks_in_run = pattern == SplitPattern::kQwen2 || tekken;
  const bool marks_excluded = pattern == SplitPattern::kQwen2;
  const int max_digits = pattern == SplitPattern::kLlama3 ? 3 : 1;
  std::vector<std::pair<size_t, size_t>> spans;
  std::vector<size_t> scratch;  // reused give-back stops for the Tekken scan
  size_t pos = 0;
  while (pos < text.size()) {
    // Tekken has NO (?i:'s|'t|...) alternative at all.
    size_t end = tekken ? 0 : MatchContraction(text, pos);
    if (end == 0) {
      end = tekken ? MatchTekkenLetterRun(text, pos, scratch)
                   : MatchLetterRun(text, pos, marks_in_run);
    }
    if (end == 0) end = MatchNumbers(text, pos, max_digits);
    if (end == 0) {
      end = MatchPunctRun(text, pos, marks_excluded, /*slash_in_tail=*/tekken);
    }
    if (end == 0) end = MatchWsNewlines(text, pos);
    if (end == 0) end = MatchWsNotBeforeNonSpace(text, pos);
    if (end == 0) end = MatchWs(text, pos);
    // Unreachable (the rules cover every codepoint class), but guarantee
    // forward progress rather than looping if the tables ever change.
    if (end == 0) end = DecodeAt(text, pos).end;
    spans.emplace_back(pos, end);
    pos = end;
  }
  return spans;
}

}  // namespace vllm::tok
