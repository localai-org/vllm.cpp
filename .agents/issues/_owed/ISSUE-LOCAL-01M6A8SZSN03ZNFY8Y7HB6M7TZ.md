ID: ISSUE-LOCAL-01M6A8SZSN03ZNFY8Y7HB6M7TZ
Title: The served Lumen-3.8-Flare binary cannot be reproduced from main
Row: -
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-01
Updated: 2026-11-01
Closed: -

## Problem

The production server on port 11435 was built from local, uncommitted changes in a scratch tree, so no committed revision serves the deployed artifact. Two of them are required to load models/Nemotron-3.5-Bolt-E4B-GSQ-RCO-IQ3_M.gguf: the file declares 1,301,418 GGUF KV array elements while src/vllm/model_executor/model_loader/gguf_reader.cpp caps the file-wide budget at kMaxCount = 1000000, and its gemma4 architecture is not in the build's supported GGUF architecture list (deepseek4, muse-glimmer, qwen35, qwen35moe, qwen3next, qwen3, llama, qwen4exp, glm5next, glm-dsa, laguna, kolibri1, flint). A clean build of the current branch therefore rejects the model at startup with 'array element budget exceeded' and then with 'GGUF architecture gemma4 is not supported by this build'. The scratch tree raises the budget to 4000000 and carries the gemma4 path; both need an owning row, a review of the untrusted-count bound, and a reproduction recipe, because today the deployment cannot be rebuilt or bisected from anything a reader can check out.

## Resolution

-
