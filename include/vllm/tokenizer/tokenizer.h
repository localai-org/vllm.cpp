// vllm.cpp original (tokenizer); semantics mirror HF tokenizers byte-level
// BPE. Loads HF tokenizer.json (FromHfJson) and GGUF vocab (FromGguf) into
// one engine: added-token longest-match pre-pass -> pretokenize ->
// merge-ranked BPE -> vocab ids. Anything unsupported throws loudly.
#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

#include "vllm/tokenizer/bpe.h"
#include "vllm/tokenizer/pretokenizer.h"

namespace vllm {
class GgufFile;  // model_executor/model_loader/gguf_reader.h
}

namespace vllm::tok {

// One entry of tokenizer.json "added_tokens" (or a GGUF special/user-defined
// token). `special` mirrors HF's flag: it does NOT change encoding (all added
// tokens split the text the same way); it drives detokenizer skip logic.
struct SpecialToken {
  std::string text;
  int32_t id = -1;
  bool special = false;
  // HF AddedToken lstrip/rstrip. They DO change encoding: when this token is
  // matched, the whitespace immediately to its left (lstrip) / right (rstrip)
  // is EATEN by the token instead of being tokenized. Mirrors transformers
  // `tokenization_utils.py` PreTrainedTokenizer.tokenize()
  //   if tok_extended.rstrip and right:  tokens[i + 1] = right.lstrip()
  //   if tok_extended.lstrip  and left:  tokens[i - 1] = left.rstrip()
  // (transformers 4.54.1 src/transformers/tokenization_utils.py:670-677), which
  // is the semantic the fast Rust `tokenizers` AddedVocabulary implements by
  // extending the match span over the adjacent whitespace. Real checkpoints set
  // them: microsoft/Phi-4-mini-instruct marks 10 of its 12 added tokens
  // rstrip=true (<|assistant|>, <|end|>, <|user|>, <|system|>, <|tool|>, ...).
  bool lstrip = false;
  bool rstrip = false;
};

class Tokenizer {
 public:
  // Loads an HF tokenizer.json. Throws std::runtime_error on anything that is
  // not byte-level BPE with a recognized Split pre-tokenizer regex (no silent
  // wrong tokenization).
  static Tokenizer FromHfJson(const std::string& tokenizer_json_path);
  // The same parse, from bytes already in hand. `source` appears in error
  // messages exactly where the path would.
  //
  // Exists because a tokenizer does not always arrive as a FILE. LTX-2.5's text
  // encoder ships its tokenizer AS A TENSOR — `tokenizer_json` U8 [32169626]
  // inside the one safetensors file, alongside `hf_asset__*` sidecars
  // (gemma_assets.py:34-36) — so a loader that assumes a sibling
  // `tokenizer.json` cannot read that checkpoint at all. Spilling 32 MB to a
  // temporary file just to read it back would put the temp directory on a model
  // path and leave a 32 MB file behind on every failure.
  static Tokenizer FromHfJsonBytes(std::string_view tokenizer_json,
                                   const std::string& source);
  // Loads a GGUF byte-level BPE vocab (tokenizer.ggml.* kvs). Throws
  // std::runtime_error unless tokenizer.ggml.model == "gpt2" and
  // tokenizer.ggml.pre is a recognized pre-tokenizer name.
  static Tokenizer FromGguf(const GgufFile& f);

  // An empty tokenizer for models that do not use a standard HF/GGUF tokenizer
  // (e.g. cua-s1-forms, which uses byte-level ByteCollator). Encode/Decode are
  // unsafe on the result; the caller must never call them.
  static Tokenizer Empty() { return Tokenizer{}; }

  // Encodes UTF-8 text. No BOS/EOS added (caller policy). Added tokens are
  // matched leftmost-longest against the raw text before pretokenization.
  std::vector<int32_t> Encode(std::string_view text) const;

  // Encodes UTF-8 text and APPLIES the post_processor's special tokens —
  // HF tokenizers' `encode(text, add_special_tokens=True)`, which is what vLLM
  // tokenizes prompts with. Concretely it realizes the TemplateProcessing
  // "single" template that ExtractBosEos already parses: prepend BosId() when
  // >= 0, append EosId() when >= 0.
  //
  // ADDED by the OPT (`OPTForCausalLM`) bring-up. Every Qwen tokenizer in the
  // tree has a ByteLevel post_processor carrying NEITHER id, so both are -1 and
  // this is byte-identical to Encode() for them — but OPT's post_processor DOES
  // declare `[SpecialToken </s>, Sequence A]`, i.e. a prepended BOS (id 2). We
  // parsed that field but never applied it, so OPT prompts came out one token
  // short and every continuation diverged (spec "seams that leaked" L4).
  std::vector<int32_t> EncodeWithSpecialTokens(std::string_view text) const;

  // Full (non-incremental) decode: ids -> raw UTF-8 bytes. Added tokens
  // contribute their literal content. Throws on unknown ids.
  std::string Decode(const std::vector<int32_t>& ids) const;

  // Raw token string as stored: byte-level (mapped) alphabet for vocab
  // tokens, literal content for added tokens. Throws on unknown ids.
  const std::string& TokenText(int32_t id) const;

  // True when `id` has an assigned token (in range and not a vocab hole).
  bool HasToken(int32_t id) const;
  // True when `id` is an added token carrying the special flag. Drives the
  // detokenizer's skip_special_tokens; encoding is unaffected.
  bool IsSpecial(int32_t id) const;

  // The longest STORED token text in bytes, over every assigned id. An UPPER
  // BOUND on how many input bytes one token can account for, because the token
  // texts of an encode concatenate back to the input and the stored form is
  // never shorter than the decoded one (see FinalizeTables). A request boundary
  // uses it to reject, WITHOUT tokenizing, a prompt that cannot possibly fit in
  // max_model_len tokens: a prompt of B bytes costs at least
  // B / MaxTokenBytes() tokens.
  size_t MaxTokenBytes() const { return max_token_bytes_; }

  // SERVE-REQUEST-LENGTH-GUARD (#1541): the largest prompt, in BYTES, that can
  // still fit in `max_model_len` tokens, `max_model_len * MaxTokenBytes()`.
  // Anything longer is unservable, so a request boundary may refuse it without
  // tokenizing. 0 means unbounded: `max_model_len <= 0` or no token length is
  // known. Clamped rather than wrapped on overflow. This is the ONE copy of the
  // derivation; the HTTP guard (ApiServer::set_tokenizer) and the rendered chat
  // prompt check (InputProcessor::max_prompt_bytes) both call it. vLLM's
  // counterpart is `max_input_tokens * tokenizer.max_chars_per_token`
  // (vllm/renderers/params.py:342-370 @ e126687a9a).
  size_t MaxPromptBytes(int64_t max_model_len) const {
    if (max_model_len <= 0 || max_token_bytes_ == 0) return 0;
    const size_t len = static_cast<size_t>(max_model_len);
    return len > std::numeric_limits<size_t>::max() / max_token_bytes_
               ? std::numeric_limits<size_t>::max()
               : len * max_token_bytes_;
  }

  // Number of id slots (max id + 1); ids in [0, VocabSize) may still be
  // unassigned for vocabs with holes.
  int32_t VocabSize() const { return static_cast<int32_t>(token_text_.size()); }
  // From the post_processor when trivially extractable (TemplateProcessing),
  // else -1. NOTE: callers should prefer the model config's eos_token_id;
  // e.g. Qwen's ByteLevel post_processor carries no eos/bos.
  int32_t EosId() const { return eos_id_; }
  int32_t BosId() const { return bos_id_; }
  SplitPattern Pattern() const { return pattern_; }
  const std::vector<SpecialToken>& AddedTokens() const { return added_tokens_; }

  // Tokenizer family. kByteLevel is the GPT-2 bytes_to_unicode BPE family
  // (Qwen/OPT/DeepSeek/Llama); kSentencePiece is the Metaspace + byte-fallback
  // BPE family (Mistral, Gemma, and SentencePiece-derived tokenizer.json).
  enum class Family { kByteLevel, kSentencePiece };
  Family GetFamily() const { return family_; }
  bool IsSentencePiece() const { return family_ == Family::kSentencePiece; }

  // Applies this SentencePiece tokenizer's DECODER chain to a window of already
  // resolved token strings [begin, end). Two chains exist, selected from the
  // tokenizer.json `decoder` node at load:
  //   - default (Mistral/Gemma, no `Metaspace` decoder node): HF tokenizers'
  //     Sequence decoder Replace(▁->space) -> ByteFallback (runs of "<0xNN>" ->
  //     raw bytes, invalid UTF-8 run -> one U+FFFD per byte) -> Fuse ->
  //     Strip(1 leading space);
  //   - a bare `Metaspace` decoder node (every published Parakeet checkpoint):
  //     HF tokenizers 0.22 decoders `Metaspace::decode_chain` — inside the
  //     FIRST token of the window each replacement is DROPPED (unless
  //     prepend_scheme == "never", where it maps to a space like everywhere
  //     else), inside every later token it becomes ONE space; no ByteFallback,
  //     no Fuse, no Strip.
  // Shared by Decode() and the incremental detokenizer so both mirror HF
  // exactly. Precondition: IsSentencePiece().
  std::string SpDecodeTokens(const std::vector<std::string>& tokens,
                             size_t begin, size_t end) const;

 private:
  Tokenizer() = default;

  // Builds token_text_/is_added_ from vocab_ + added_tokens_ (shared by the
  // HF and GGUF loaders). Throws on id collisions or out-of-range ids.
  void FinalizeTables();
  // Pretokenize + BPE for a byte-level text segment with no added tokens inside.
  void EncodePlain(std::string_view text, std::vector<int32_t>& out) const;
  // Metaspace transform + byte-fallback BPE for a SentencePiece text segment
  // with no added tokens inside. `at_input_start` is true iff this segment
  // begins at byte 0 of the whole input (drives prepend_scheme="first").
  void EncodePlainSp(std::string_view text, bool at_input_start,
                     std::vector<int32_t>& out) const;
  // Viterbi DP tokenization for a SentencePiece Unigram piece (model.type ==
  // "Unigram"). Replaces BPE merges with a lattice search: at each byte
  // position, try every vocab entry that matches, plus unk for one UTF-8
  // character. Matches HF tokenizers UnigramModel::tokenize
  // (tokenizers/src/models/unigram/model.rs).
  void EncodePlainUnigram(std::string_view piece,
                         std::vector<int32_t>& out) const;

  std::unordered_map<std::string, int32_t> vocab_;  // mapped symbol -> id
  MergeRanks merge_ranks_;
  std::vector<SpecialToken> added_tokens_;
  std::vector<std::string> token_text_;  // id -> stored text ("" = unassigned)
  size_t max_token_bytes_ = 0;           // longest token_text_ entry, in bytes
  // id -> 0 plain vocab, 1 added, 2 added+special. Nonzero decodes literally.
  std::vector<uint8_t> is_added_;
  Family family_ = Family::kByteLevel;
  // --- SentencePiece (Metaspace + byte-fallback) parameters (family_ ==
  // kSentencePiece). Mirror the HF tokenizer.json `pre_tokenizer` (Metaspace)
  // and `model` (BPE byte_fallback/fuse_unk/unk_token). ---
  enum class PrependScheme { kNever, kFirst, kAlways };
  std::string metaspace_replacement_;  // e.g. "▁" (U+2581), UTF-8
  PrependScheme prepend_scheme_ = PrependScheme::kFirst;
  bool metaspace_split_ = false;  // Metaspace `split` flag
  // --- DECODER selection (tokenizer.json `decoder` node). False => the
  // Sequence chain (Replace -> ByteFallback -> Fuse -> Strip). True => the bare
  // Metaspace decode_chain (Parakeet), with the DECODER node's own replacement
  // and prepend_scheme (HF reads them off the decoder, not the pre_tokenizer).
  bool sp_decoder_metaspace_ = false;
  std::string sp_decoder_replacement_;
  bool sp_decoder_prepend_never_ = false;
  bool byte_fallback_ = false;    // model.byte_fallback
  bool fuse_unk_ = false;         // model.fuse_unk (fuse consecutive unk ids)
  int32_t unk_id_ = -1;           // model.unk_token resolved to an id, else -1
  SplitPattern pattern_ = SplitPattern::kQwen2;
  int32_t eos_id_ = -1;
  int32_t bos_id_ = -1;
  // The tokens the POST-PROCESSOR itself prepends/appends, which is a strictly
  // narrower thing than "this model's bos/eos ids". Set ONLY from an HF
  // TemplateProcessing "single" template (FromHfJson), or explicit/default
  // add_bos_token/add_eos_token flags on Llama GGUF. Other GGUF families keep
  // these -1: their BOS/EOS IDs alone never license prompt insertion.
  int32_t template_bos_ = -1;
  int32_t template_eos_ = -1;
  bool ignore_merges_ = false;  // BPE option: whole-pretoken vocab hit wins
  // --- Unigram (SentencePiece Unigram model, model.type == "Unigram"). Uses
  // Viterbi DP instead of BPE merges. The vocab is an array of [token, score]
  // pairs where score is a log-probability; index == id. The unk penalty is
  // unigram_min_score_ - 10.0 (HF tokenizers default when no unk_score is
  // declared). ---
  bool gguf_spm_ = false;
  bool smollm_digits_ = false;
  std::unordered_map<std::string, float> spm_scores_;
  bool is_unigram_ = false;
  std::unordered_map<std::string, float> unigram_scores_;
  float unigram_min_score_ = 0.0f;
};

}  // namespace vllm::tok
