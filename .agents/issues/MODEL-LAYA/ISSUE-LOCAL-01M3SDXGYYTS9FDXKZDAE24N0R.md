ID: ISSUE-LOCAL-01M3SDXGYYTS9FDXKZDAE24N0R
Title: Loader reads only <model_dir>/tokenizer.json; convaiinnovations/laya keeps it under tokenizer/
Row: MODEL-LAYA
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: 2026-09-30

## Problem

The HF snapshot of convaiinnovations/laya stores its tokenizer at <model_dir>/tokenizer/tokenizer.json and <model_dir>/tokenizer/tokenizer_config.json. The engine only looks for <model_dir>/tokenizer.json, so serving the snapshot as downloaded fails with 'tokenizer: symbol "c" not in vocab' unless the file is symlinked to the top level. The loader needs a fallback to the tokenizer/ subdirectory when the top-level file is absent, without changing any layout that works today.

## Resolution

2026-09-30: vllm::ResolveTokenizerFile reads <model_dir>/tokenizer.json, else <model_dir>/tokenizer/tokenizer.json (same for tokenizer_config.json); the root file always wins. Used by LoadedEngine::FromModelDir, vllm-server and the C ABI chat-template path. tests/vllm/entrypoints/test_tokenizer_subdir.cpp is red without the loader change ('no *.safetensors shards found' instead of reading tokenizer/tokenizer.json) and green with it. End to end on CPU: the convaiinnovations/laya snapshot @ 55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851, served as downloaded, returned 500 'tokenizer: symbol "c" not in vocab' before and a choice answer (billing 0.9419) after. fastino/GLiNER2.5-Decide @ 5a7adf72a23b4d311abae6ce050d7f0012bb3416 keeps tokenizer.json at the root and serves unchanged.
