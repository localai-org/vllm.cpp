ID: ISSUE-LOCAL-01M4D7DFJRQSCSRC03CHZW3EQN
Title: R7: tokenizer engine refuses the Kolibri-1 tokenizer.json (\p{N}{1} variant of the classic Qwen2 split regex)
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: -

## Problem

Tokenizer::FromHfJson refuses /mnt/models/Aleph-Alpha/Kolibri-1/tokenizer.json with "unrecognized pre-tokenizer split regex" (src/vllm/tokenizer/tokenizer.cpp:510, DetectPattern). The Kolibri-1 pre-tokenizer Split regex is byte-identical to the recognized classic Qwen2 pattern (kClassicQwen2Regex) except that the number alternative is written \p{N}{1} instead of \p{N} — a semantically null quantifier the whole-pattern byte-equality recognition does not cover. The (?i:'s|'t|...) contraction group form IS already implemented (MatchContraction, Unicode simple case folding); the spec R7 text blaming the (?i:...) group form is corrected by measurement. Consequence: the engine cannot encode with the real Kolibri-1 tokenizer, so the W3 token gate feeds HF ids directly and the no-BOS encode contract (add_bos_token: false, eos 127906, pad 127901) is ungated. Fix: recognize the \p{N}{1} variant by whole-pattern match and map it to the kQwen2Classic scanner (\p{N}{1} == \p{N}: single-codepoint number grouping, which is exactly what that scanner implements).

## Resolution

-
