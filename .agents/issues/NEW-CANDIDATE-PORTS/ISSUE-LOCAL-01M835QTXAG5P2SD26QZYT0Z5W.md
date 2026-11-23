ID: ISSUE-LOCAL-01M835QTXAG5P2SD26QZYT0Z5W
Title: Wire olala greedy decode through a production seam (registry forward or thin example)
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-23
Updated: 2026-11-23
Closed: -

## Problem

Reachability check for benchmark axis p150-olala-bf16 (spec .agents/specs/p150-benchmark-axes.md @ row/QWEN38-DFLASH2-STAGE-C-bench-plan, worktree /tmp/vllm-bench-plan) fails at the W2e head 048c6b22d. OlalaGenerateHost (src/vllm/model_executor/models/olala_ops.cpp:811) is reached only from tests/vllm/models/test_olala_ops.cpp:767,784,948. The registered production forward, ForwardOlalaForCausalLM in src/vllm/model_executor/models/olala_registry.cpp:76-85, is still the W2b refuse-by-name stub (VT_CHECK(false, OlalaForwardRefusalOps())), and no olala example, CLI, or server path exists. Per AGENTS.md section Nothing lands dead and the spec's identical-workload rules, a benchmark number measured through the test harness cannot be published, so the axis stays PENDING regardless of the token gates being EXACT (bf16, q6_k, q8_0-safe2 per docs/bench-evidence/new-candidate-ports-olala-w2e-20261122/). Wiring unit owed: port the ForwardOlalaForCausalLM body onto OlalaGenerateHost (or a block-forward equivalent, reusing the W2e arms and the KV spec already registered in MakeOlalaForCausalLMKVCache), or add a thin olala example like examples/kimi_linear_gen as the sanctioned seam, so decode is reachable from ModelRegistry::Forward, the loader, or a registered CLI path on default config. Only then can axis A1/A2/A8 numbers for p150-olala-bf16 be produced and published with the llama-cpp-olala @ bd20ca9b denominator.

## Resolution

-
