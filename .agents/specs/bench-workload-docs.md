# Benchmark workload controls in the public guide

## Scope

Document the existing `vllm-bench` workload controls restored by `3f3cab2c8`
and `fb5b5ca4b`. This is a documentation correction within
`BENCH-QWEN38-27B-SOTA`, tracked by
`ISSUE-LOCAL-01M4MEJ5Z64DFXCMDFX39KZA90`. It does not advance that benchmark
row or publish a performance result.

Base: `6c4323226`. Open documentation PRs cover ABI and Kolibri updates.
Exclude those subjects from this change.

## Source and design

| Surface | Binding source | Documentation change |
|---|---|---|
| Accepted options | `examples/bench/main.cpp`, `ParseArgs` and `Usage` | List workload options and paired boolean spellings |
| Defaults | `examples/bench/bench_core.h`, `BenchConfig` | State raw prompts, ignored EOS, unset thinking, and `auto` KV dtype |
| Sampling | `examples/bench/bench_core.h`, `MakeSampling` | Explain fixed output length versus natural EOS stopping |
| Template source and rendering | `ResolveBenchChatTemplate` and `RunBench` in the same header | Explain file or literal override, required template activation, one user message, and missing-template refusal |
| Runtime report | `PrintReport` in the same header | Name the fields that record resolved settings |
| Existing tests | `tests/examples/test_bench_eos_chat_template.cpp`, `tests/examples/test_bench_kv_cache_dtype.cpp` | Use their assertions as corroborating evidence; no new product behavior or test port |

The public guide replaces the dense benchmark bullet with a dedicated section.
Include a synthetic CPU smoke command, a real-model command with explicit
workload settings, and a compact table. Keep the polling and pretokenization
notes. Add one concise README news entry linking to the section. Do not change
benchmark numbers or claim equivalence from flags alone.

The source comments cite vLLM benchmark option and sampling semantics. This
change documents the local parser and runtime as shipped. No oracle execution,
GPU, model download, or dependency kernel inspection is needed for prose-only
corrections.

## Work breakdown

1. Commit this scope and local issue before changing public prose.
2. A fresh implementer edits only `docs/USAGE.md` and the README news section.
3. A fresh reviewer checks the immutable change against the parser, defaults,
   rendering, sampling, report, and existing tests.
4. The coordinator reruns the focused checks and opens a fork PR.

## Verification and evidence

Run on CPU:

```sh
python3 scripts/check-readme-structure.py
python3 scripts/check-quickstart-recipes.py
python3 scripts/check-agent-record.py
python3 tests/scripts/test_check_readme_structure.py
git diff --check
```

Run `scripts/agent-preflight.sh` before edits and before commit. Record existing
failures separately. The baseline environment lacks a compiler and Python on
PATH. A temporary Python interpreter under `/tmp` runs the document gates.
No C++ or inference behavior changes, so compilation and performance comparison
are not acceptance gates for this correction. Review every added flag and
example against source; test mutations are not applicable without changed tests.

## Risks and stop conditions

Do not imply that synthetic timings measure model performance. Do not label
unset thinking as false. Do not imply a template override enables rendering.
Do not claim that matching the new settings alone establishes a fair benchmark.
Stop for contradictory source or an unrelated required edit. Keep any baseline
record failures visible instead of weakening checkers.

## Now

Documentation implemented and independently reviewed. Awaiting upstream PR
review. The owning benchmark lifecycle and pending hardware gates are unchanged.

## Outcome

The documentation implementation is `db60c0813156aca4d92243b563e73341104ab3aa`.
A fresh reviewer returned PASS on that immutable head. The review checked the
parser, defaults, template rendering, sampling, KV resolution, and existing
benchmark tests. A scratch em-dash mutation failed the README checker as
expected; the reviewer restored the original bytes.

The coordinator reran the five focused commands above successfully, including
all 19 README tests. A source check found all 19 documented options in the
parser and resolved the new links. Commit style and trailer checks passed.
`check-tree-compiles.py --base upstream/main` returned zero: no C++ source,
header, or build file changed. No inference example ran.

Full preflight is not green in this environment. The baseline and staged runs
encounter missing PyYAML and CMake, release metadata tooling failures, and
subprocess library-path failures from the temporary interpreter. These failures
are outside the changed prose. Logs for this session are
`/tmp/vllm-docs-baseline.log` and `/tmp/vllm-docs-spec-preflight.log`.

No performance measurement or benchmark lifecycle changed. The guide preserves
the shipped defaults instead of selecting new ones. The local issue remains
open until the documentation lands upstream.
