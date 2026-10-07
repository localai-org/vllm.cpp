ID: ISSUE-LOCAL-01M3RY20H90NMK37V74EH34Z77
Title: No eos_token_id resolution from tokenizer_config.json: requests carry no eos and generation never stops
Row: MODEL-QWEN35-EXL3
State: OPEN
Kind: bug
GitHub: 3364
Mirror: SYNCED
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

Reproduced live on the vllm-qwen35-9b-exl3 container (Qwen3.5-9B-EXL3-4.00bpw, ROCm gfx1101, 2026-09-30). The checkpoint's config.json carries NO eos_token_id and ships no generation_config.json, and its tokenizer.json post_processor is a bare ByteLevel with no TemplateProcessing template, so Tokenizer::FromHfJson leaves eos_id_ = -1 (ExtractBosEos only reads post_processor). InputProcessor then finds no eos anywhere (input_processor.cpp:42-67) and every request goes out with eos_token_id unset: the model emits <|im_end|> (id 248046) at token 10 of a no-think reply and the engine ignores it, generating fake 'user\n\nassistant\n<think>' turns to max_tokens (finish_reason=length). Upstream resolves the same id from the tokenizer itself: HF eos_token_id comes from tokenizer_config.json's eos_token ('<|im_end|>'), which FromHfJson never reads. User-visible symptom: thinking looks inconsistent — every answer ends by hallucinating more think blocks because <|im_end|> never terminates the turn.

## Resolution

FIXED 2026-09-30 on branch row/MODEL-QWEN35-EXL3-eos (commit c09275757). FromHfJson now reads sibling tokenizer_config.json/special_tokens_map.json and resolves eos_token/bos_token NAMES to ids where the post_processor left them unset. Verified LIVE on the vllm-qwen35-9b-exl3 container (rebuilt image): enable_thinking=false stops at <|im_end|> (finish_reason=stop, 11 tokens) where it previously ran to length hallucinating turns; thinking-on request stops at finish_reason=stop with content after the </think> split; think_auto parser (compose adds --reasoning-parser auto) emits reasoning_content on the thinking arm. test_bpe 30/30 green incl. new naming-resolution cases.
