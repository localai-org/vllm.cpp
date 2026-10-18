ID: ISSUE-LOCAL-01M4AK4PWVEK8GWAR4AK5CTKGX
Title: test_model_loader_gguf's expected supported-architectures list omits Kolibri1ForCausalLM (main build-test-cpu red)
Row: MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-07
Updated: 2026-10-07
Closed: 2026-10-07

## Problem

The kolibri-1 CPU row landed the Kolibri1ForCausalLM registration on main (#3392/#3398) without updating the hardcoded supported-architectures dict_keys string in tests/vllm/test_model_loader_gguf.cpp (TEST_CASE "FromModelDir rejects an unknown dense architecture before loading"). The registry's refusal message now emits 'Kolibri1ForCausalLM' (alphabetically between 'KimiLinearForCausalLM' and 'LagunaForCausalLM'), so the CHECK_THROWS_WITH_AS exact-match fails and main's build-test-cpu CI is red on test_model_loader_gguf. The sibling list in tests/vllm/models/test_model_registry.cpp was updated by the row (4167149c9); the GGUF loader test's list was not. Red at base 452617154: the thrown message contains 'Kolibri1ForCausalLM' but the expected string does not.

## Resolution

- 2026-10-07 RED at base 452617154 (branch fix/kolibri1-loader-test-list, build /tmp/build-loader-fix): ctest -R test_model_loader_gguf exits 8 — 1 of 14 doctest cases fails ("FromModelDir rejects an unknown dense architecture before loading", tests/vllm/test_model_loader_gguf.cpp:239): the thrown refusal contains 'Kolibri1ForCausalLM' between 'KimiLinearForCausalLM' and 'LagunaForCausalLM'; the expected string does not. Identical failure is main's build-test-cpu CI at this SHA (run 37548431732, job 112557940349, test #30 Failed).
- 2026-10-07 FIX (one line): tests/vllm/test_model_loader_gguf.cpp:248 gains 'Kolibri1ForCausalLM', at the registry's emitted position (src/vllm/model_executor/models/model_registry.cpp sorts registrations by architecture name). The only other hardcoded copy of this list, tests/vllm/models/test_model_registry.cpp:1017, already carries it; no other occurrence contradicts the registry order.
- 2026-10-07 GREEN: test_model_loader_gguf 14/14 doctest cases, 61/61 assertions, doctest Status: SUCCESS. Adjacent model-loader/registry/gguf binaries 15/15 pass: test_model_registry, test_hf_config, test_gguf, test_gguf_dequant, test_gguf_nvfp4, test_gguf_qwen36_loader, test_gguf_keep_quant, test_gguf_device_fit, test_gguf_device_fit_reach, test_gguf_mmproj_reach, test_gguf_accounting_reach, test_gguf_expert_span, test_modelopt_mixed_precision, test_modelopt_mixed_precision_checkpoint.
