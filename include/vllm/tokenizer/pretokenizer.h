// vllm.cpp original (tokenizer); semantics mirror HF tokenizers' Split
// pre-tokenizer (behavior=Isolated) for the byte-level BPE split regexes.
// See src/vllm/tokenizer/pretokenizer.cpp for the verbatim patterns.
#pragma once

#include <cstddef>
#include <string_view>
#include <utility>
#include <vector>

namespace vllm::tok {

// Which pre-tokenizer Split regex to emulate.
enum class SplitPattern {
  kQwen2,   // Qwen3.6 family: single-codepoint \p{N}, \p{M}-aware letter/punct
            // runs (regex from Qwen3.6-27B, marks fold into letter runs).
  kQwen2Classic,  // CLASSIC Qwen2/Qwen3 family (e.g. Qwen/Qwen3-0.6B,
                  // Qwen3-Coder): single-codepoint \p{N} like kQwen2 but WITHOUT
                  // \p{M} in the letter run / punct-negation (marks fall into the
                  // punct run, exactly like Llama-3's classes but with 1-digit
                  // number grouping). Also the Kolibri-1 spelling of this pattern,
                  // whose number alternative is written \p{N}{1} — the identity
                  // quantifier, so the same scanner is exact.
  kLlama3,  // Llama-3 family: \p{N}{1,3} digit groups, no \p{M} awareness.
  kGpt2,    // ORIGINAL GPT-2 byte-level BPE (OPT, GPT-2, and every other
            // pre-Llama HF checkpoint whose tokenizer.json carries NO explicit
            // Split component and instead sets ByteLevel `use_regex: true`).
            // Materially different from all three above, not a variant of them:
            // case-SENSITIVE contractions, a plain ` ?` space prefix instead of
            // the `[^\r\n\p{L}\p{N}]?` prefix, UNBOUNDED `\p{N}+` digit runs,
            // no `[\r\n]*` punct tail and no `\s*[\r\n]+` rule at all.
  kTekken,  // Mistral Tekken family (Mistral-Nemo, and the Tekken-v3/v7
            // checkpoints that share its tokenizer.json shape). Case-aware:
            // two alternatives,
            // [\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+ then
            // the same pair with the quantifiers swapped, so an uppercase run
            // ends a piece when a lowercase run follows ("HelloWorld" ->
            // "Hello" + "World"). It is tiktoken's o200k_base pat_str with
            // exactly two edits: NO (?i:'s|'t|...) contraction group, and
            // single-codepoint \p{N} instead of \p{N}{1,3}. Marks sit INSIDE
            // both letter classes (like kQwen2) while the punct negation is
            // [^\s\p{L}\p{N}] with no \p{M} (like kLlama3) -- a combination no
            // other pattern has. Its punct run also ends [\r\n/]*, absorbing a
            // '/' that follows a newline.
  kGpt4o,   // GPT-4o / o200k family (llama.cpp's LLAMA_VOCAB_PRE_TYPE_GPT4O:
            // pre names "gpt-4o", "llama4", "kanana2", "talkie"). NOT a
            // variant of kLlama3 despite sharing its \p{N}{1,3} digit
            // grouping: the single letter-run alternative is replaced by the
            // SAME two case-aware alternatives kTekken uses, the contraction
            // is a SUFFIX of the word rather than its own leading alternative,
            // and the punctuation run absorbs a trailing `/` as well as \r/\n.
            // The difference from kTekken is exactly the two edits named in
            // its comment above, run in reverse: kGpt4o KEEPS the o200k
            // contraction group and KEEPS \p{N}{1,3}. See
            // src/vllm/tokenizer/pretokenizer.cpp for the verbatim pattern.
  kDeepSeek,  // DeepSeek-V2 family (DeepSeek-V2, V2-Lite, and every GGUF
              // tagged `tokenizer.ggml.pre = "deepseek-llm"`; llama.cpp's
              // LLAMA_VOCAB_PRE_TYPE_DEEPSEEK_LLM). STRUCTURALLY UNLIKE every
              // pattern above: not ONE alternation regex but a HF `Sequence`
              // PIPELINE of seven pre-tokenizers, each further splitting the
              // pieces the previous one produced — five
              // `Split(behavior=Isolated)` stages over EXPLICIT codepoint
              // classes (newlines; cased letters; ASCII+fullwidth+CJK
              // punctuation; trailing whitespace; CJK/Hangul), then
              // `Digits(individual_digits=true)`, then a
              // `ByteLevel(use_regex=false)` that splits nothing. The classes
              // are enumerated codepoint ranges in the checkpoint rather than
              // `\p{...}` properties, so they are transcribed verbatim in
              // src/vllm/tokenizer/pretokenizer.cpp with their provenance.
              //
              // NOT DeepSeek-V3. This comment used to claim V3 as well and it
              // was wrong: V3 ships a different pipeline, kDeepSeekV3 below.
  kDeepSeekV3,  // DeepSeek-V3 family (DeepSeek-V3, DeepSeek-R1,
                // DeepSeek-V4-Flash, and every GGUF tagged
                // `tokenizer.ggml.pre = "deepseek-v3"` or `"joyai-llm"`;
                // llama.cpp's LLAMA_VOCAB_PRE_TYPE_DEEPSEEK3_LLM). Also a HF
                // `Sequence` PIPELINE, and a genuinely SEPARATE family from
                // kDeepSeek rather than a variant of it: FOUR stages, not
                // seven — `Split(\p{N}{1,3})`, `Split([<kana+CJK>]+)`, a
                // `Split` over a six-alternative regex, then
                // `ByteLevel(use_regex=false)`. It has no `Digits` stage, no
                // trailing-whitespace stage, no Hangul in its CJK class, and
                // its word/punctuation rules are `\p{...}` properties rather
                // than enumerated ranges. Aliasing either onto the other, or
                // onto kLlama3 because both group digits `\p{N}{1,3}`,
                // mis-tokenizes silently (#1924). Pipeline and provenance in
                // src/vllm/tokenizer/pretokenizer.cpp.
};

// Splits `text` into pretoken byte spans [first, second), exactly as HF
// tokenizers' Split(Regex(pattern), behavior="isolated") does. Spans are
// contiguous, non-overlapping, and cover the input: concatenating
// text.substr(f, s - f) over all spans reconstructs `text` byte-for-byte.
// Invalid UTF-8 decodes as U+FFFD (category So) for classification purposes,
// but the spans always index the original bytes.
std::vector<std::pair<size_t, size_t>> Pretokenize(std::string_view text,
                                                   SplitPattern pattern);

}  // namespace vllm::tok
