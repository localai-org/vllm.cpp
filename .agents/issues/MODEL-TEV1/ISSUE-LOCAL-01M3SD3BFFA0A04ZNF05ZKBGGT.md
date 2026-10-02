ID: ISSUE-LOCAL-01M3SD3BFFA0A04ZNF05ZKBGGT
Title: Serve Tev1 through /v1/systemone and vllm_decide
Row: MODEL-TEV1
State: CLOSED
Kind: feature
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: 2026-09-30

## Problem

Ollama 0.35 serves togethercomputer/Tev1-4B-experimental (tev1) and Tev1-0.8B-experimental (tev1:0.8b) on /v1/systemone, the same endpoint as nimble: it compiles typed choice/noul/score questions, scores the allowed answer letters from next-token logits, and returns probabilities with entropy confidence. vllm.cpp serves Tev1 only through /v1/chat/completions, which returns one sampled letter and no distribution, and vllm_decide refuses the architecture by name. LocalAI reaches vllm.cpp only through the C ABI, so it cannot get a Tev1 decision with probabilities. The request-level machinery that Nimble uses (question compilation, candidate token check, openjev answer) exists but is private to Nimble.

## Resolution

2026-09-30: a Tev1Model engine serves /v1/systemone (vllm-server, beside the chat routes) and vllm_decide through one production entry, Tev1Decide, over the request-level scorer now shared with Nimble (decision_scorer). The prompt is the model's decide.py turn; the logits come from the engine through max_tokens=1 + logprob_token_ids (vLLM generative scoring). test_tev1_systemone 9/9 (prompts byte-equal to decide.py through both chat templates, answers equal to Ollama decision.Answer run from Go, refusals, engine and vllm_decide reachability vs an independent dense forward); real-tokenizer ids equal for both checkpoints. CPU real weights vs transformers 5.3.0 BF16 on 7 questions: 4B argmax 7/7 max dp 0.0004, 0.8B 6/7 (near tie) max dp 0.031. The vLLM generative-scoring gate, CUDA and GGUF stay owed in .agents/specs/tev1.md.
