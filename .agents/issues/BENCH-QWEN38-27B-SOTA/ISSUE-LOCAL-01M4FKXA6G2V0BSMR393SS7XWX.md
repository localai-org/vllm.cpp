ID: ISSUE-LOCAL-01M4FKXA6G2V0BSMR393SS7XWX
Title: test_bench_eos_chat_template red in every CPU lane: vllm-bench lost the EOS/chat-template flags in the #293 reapply
Row: BENCH-QWEN38-27B-SOTA
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-09
Updated: 2026-10-09
Closed: 2026-10-09

## Problem

Mechanism (measured 2026-10-09, repro binaries from the sanitize-cpu lane): tests/examples/test_bench_eos_chat_template.cpp execs the built vllm-bench binary; all 9 cases fail with 'vllm-bench: unknown argument' for --no-ignore-eos / --ignore-eos / --no-skip-chat-template / --chat-template / --no-enable-thinking (exit 2) — none of the flags exist in examples/bench/main.cpp at origin/main. 961645d0d (#2759) added the four settings (ignore_eos, skip_chat_template, chat_template source, enable_thinking), the MakeSampling pass-through (sp.ignore_eos = cfg.ignore_eos), the chat-template render through MakeChatTemplatePromptFn before admission, the two refusals (--chat-template without --no-skip-chat-template; --no-skip-chat-template with no template source), and the 'Ignore EOS (requested)/(resolved sampling):' + 'Chat template:' + 'Chat template kwargs:' report lines read back out of MakeSampling/DefaultChatTemplateKwargs. The later PR #293 reapply (cd4b4c6a1) rewrote examples/bench from a pre-#2759 tree, silently dropping all of it while keeping the test; MakeSampling again hardcodes sp.ignore_eos = true. Fails identically in build-test-cpu and both sanitize-cpu lanes (.agents/specs/ci-main-ctest-residue.md) — a product regression in the benchmark harness, not a sanitizer finding. Fix: re-port 961645d0d onto the current main.cpp/bench_core.h (BenchConfig fields, ParseArgs arms, Usage text, MakeSampling pass-through, ResolveBenchChatTemplate + render, refusals, BenchResult read-back fields, the three report lines, the stderr header fields).

## Resolution

FIXED 2026-10-09. Re-ported 961645d0d (#2759) onto the current examples/bench: the --ignore-eos/--no-ignore-eos, --skip-chat-template/--no-skip-chat-template, --chat-template and --enable-thinking/--no-enable-thinking ParseArgs arms, BenchConfig fields, the MakeSampling pass-through (sp.ignore_eos = cfg.ignore_eos), the ResolveBenchChatTemplate source resolution (file or single-line literal, else LoadChatTemplateForModel), the render through MakeChatTemplatePromptFn before admission, the two refusals (--chat-template without --no-skip-chat-template; --no-skip-chat-template with no template source), the BenchResult read-backs (resolved_ignore_eos from MakeSampling, resolved_chat_template, resolved_chat_template_kwargs from DefaultChatTemplateKwargs), the three report lines, the Usage text and the stderr header fields. The flags had been dropped when the PR #293 reapply (cd4b4c6a1) rewrote examples/bench from a pre-#2759 tree. Evidence: BEFORE, ctest in the VLLM_CPP_SANITIZE=address,undefined tree with the CI env: all 9 test_bench_eos_chat_template cases failed with "vllm-bench: unknown argument" for --no-ignore-eos/--ignore-eos/--no-skip-chat-template/--chat-template/--no-enable-thinking, exit 2 (log /tmp/sanitize-runs/test_bench_eos_chat_template.log, rc=1). AFTER the fix, same tree rebuilt: the 9-test residue ctest selection -> 100% tests passed out of 9 (test_bench_eos_chat_template Passed 14.83 sec, 9/9 cases incl. the MakeSampling read-back both ways, the render changing the admitted token count, both refusals, and the file-vs-literal prompt equality). Normal Release build (/tmp/build-c): same selection -> 100% passed out of 9 (Passed 19.53 sec).
