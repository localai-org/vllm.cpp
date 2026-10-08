ID: ISSUE-LOCAL-01M4D7DFJRQSCSRC03CHZW3EQN
Title: R7: tokenizer engine refuses the Kolibri-1 tokenizer.json (\p{N}{1} variant of the classic Qwen2 split regex)
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: 2026-10-08

## Problem

Tokenizer::FromHfJson refuses /mnt/models/Aleph-Alpha/Kolibri-1/tokenizer.json with "unrecognized pre-tokenizer split regex" (src/vllm/tokenizer/tokenizer.cpp:527, DetectPattern). The Kolibri-1 pre-tokenizer Split regex is byte-identical to the recognized classic Qwen2 pattern (kClassicQwen2Regex) except that the number alternative is written \p{N}{1} instead of \p{N} — a semantically null quantifier the whole-pattern byte-equality recognition does not cover. The (?i:'s|'t|...) contraction group form IS already implemented (MatchContraction, Unicode simple case folding); the spec R7 text blaming the (?i:...) group form is corrected by measurement. Consequence: the engine cannot encode with the real Kolibri-1 tokenizer, so the W3 token gate feeds HF ids directly and the no-BOS encode contract (add_bos_token: false, eos 127906, pad 127901) is ungated. Fix: recognize the \p{N}{1} variant by whole-pattern match and map it to the kQwen2Classic scanner (\p{N}{1} == \p{N}: single-codepoint number grouping, which is exactly what that scanner implements).

## Resolution

Landed 2026-10-08 on row/kolibri-r7 (spec commit 4097ca5e1, fix commit
fa981d145, records commit cda132013; pushed to mine, no PR per the task).

- Diagnosis (measured, correcting the issue text): the Kolibri-1 Split
  regex is byte-identical to kClassicQwen2Regex except the number
  alternative is written \p{N}{1} instead of \p{N} — a semantically null
  quantifier the whole-pattern byte-equality recognition did not cover. The
  (?i:...) contraction group was never the problem; MatchContraction already
  implements it case-insensitively (the R7 spec text's original diagnosis
  was wrong and is corrected in .agents/specs/kolibri-1-cpu.md
  "R7 resolution").
- Fix (recognition-only): one verbatim constant kClassicQwen2N1Regex
  (src/vllm/tokenizer/tokenizer.cpp:29-38) mapped onto
  SplitPattern::kQwen2Classic in DetectPattern (:506-512) — \p{N}{1} matches
  exactly \p{N}, and the kQwen2Classic scanner's single-codepoint number
  grouping is what both spellings mean. No scanner code changed; no other
  model's recognition can change.
- Red-first: the reworked W1 test (tests/vllm/models/test_kolibri1.cpp:754-849)
  failed before the fix with the named refusal "unrecognized pre-tokenizer
  split regex" (capture /tmp/r7-red.log).
- Green: 52/52 assertions. The HF-id match: our Tokenizer::Encode reproduces
  the HF reference input_ids from tests/vllm/models/kolibri1_goldens.json
  (produced by the real HF tokenizers library,
  scripts/gen-kolibri1-goldens.py:268-273, add_special_tokens=False) on all
  8 golden prompts — the ids the W3 gate had been feeding directly — plus a
  Decode round-trip, an equivalence probe against the same file respelled
  \p{N} (recognized before the change) over an 18-string
  contraction/digit/whitespace/unicode corpus, and the no-BOS special-token
  contract (add_bos_token false, eos <|im_end|> 127906, pad <|endoftext|>
  127901, Encode == EncodeWithSpecialTokens).
- Non-regression: the full tokenizer surface stays green — test_pretokenizer
  151557/151557, test_unicode_data 878/878, test_bpe 1009/1009,
  test_bpe_equivalence 334/334, test_detokenizer 221/221,
  test_tokenizer_metaspace_split 28/28, test_tokenizer_parity 2243/2243,
  test_tokenizer_parity_deepseek 2461/2461, test_tokenizer_parity_deepseek_v3
  1616/1616, test_tokenizer_parity_gpt4o 1000/1000,
  test_tokenizer_parity_mistral 421/421.
- Gates: test_kolibri1 27/27 test cases, 232/232 assertions (main baseline
  27/186 — the reworked tokenizer test carries 52 assertions where the
  refusal test carried 6). W3 900/900 assertions, 2/2 test cases
  (VLLM_CPP_CPU_THREADS=8, quiet window 2026-10-08 11:57:51, zero
  test_kolibri1_* procs, avail 210 GB; ARGMAX CHAIN 141/145 with 4 near-tie
  flips and 0 hard flips, worst topk 2.18646 — identical to the recorded
  main result). decode_bench rc=0, anchor last token 109726 reproduced.
  check-agent-record rc=0 (ANCHOR-ROT=0 after repairing the 17-line shift's
  stale citations). Commit style + trailer checks rc=0.
- The 17-line engine insertion shifted later lines; every stale
  tokenizer.cpp citation in the records was repaired in cda132013 and
  verified to point at byte-identical content.
