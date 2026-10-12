# Document current Kolibri-1 usage

Work ID: `DOCS-KOLIBRI1-CURRENT-USAGE`.
Base: `c6a971e53`.

## Now

Documentation patch implemented for review. No model, backend, or benchmark
lifecycle changes. README news links the CPU recipe and validation limits.
The feature row now reflects compressed loading and the Tenstorrent forward.
The usage section preserves incomplete checkpoint provenance explicitly.

## Scope and source inventory

| Surface | Source | Correction |
|---|---|---|
| README news | `src/vllm/model_executor/models/kolibri1_registry.cpp` | Announce the reachable model with explicit validation limits |
| Features model row | `src/vllm/model_executor/models/kolibri1_weights.cpp`, `kolibri1_forward.cpp`, `kolibri1_registry.cpp` in the same directory | Replace historical prose with the current loader, CPU and TT availability, and refused arms |
| Usage recipe | `examples/cli/main.cpp`, `src/vllm/entrypoints/openai/server_main.cpp`, `include/vllm.h`, Kolibri registry | Add source-checked local-checkpoint instructions, defaults, and memory limits |
| Existing correctness and performance evidence | `.agents/specs/kolibri1*.md`, `docs/gap-kolibri1-tt-captured-decode.md` | Distinguish measured comparisons from pending primary-oracle parity |

Use a focused `docs/models/kolibri1.md` guide if the recipe needs more than a
short usage section. Link the guide from `docs/USAGE.md` and the website model
navigation. Preserve existing checkpoint provenance without inventing revisions
or hashes. A missing artifact pin must remain explicit.

## Design and exclusions

Keep one compact feature row. Put commands and necessary limits in the usage
surface. Put detailed historical evidence in its existing specs and link to it.
Use plain sentences and avoid internal attempt names in the public overview.

Do not edit runtime code, tests, checker semantics, measurements, capability
states, unrelated models, ABI documentation, or benchmark index dispositions.
Do not infer a speed result or full token parity from a registration, a fixture,
a microbenchmark, or accepted near-ties. Do not run a GPU or download models.

## Upstream and dependency chain

This change ports no runtime behavior. The local public entry points define the
commands. The retained Kolibri specs identify the Aleph Alpha plugin reference
and its gateability. No new vLLM execution is applicable to this prose correction.
The existing source and evidence define every public claim.

## Tests and gates

Run `scripts/check-readme-structure.py`, `scripts/check-supported-models.py`,
`scripts/check-surface-coverage.py`, `scripts/check-benchmark-index.py`, and
`scripts/check-quickstart-recipes.py`. Run their existing mutation suites.
Check new relative links and every documented flag against source.
Run `git diff --check` and the full `scripts/agent-preflight.sh`.
Report baseline failures and missing tools separately from scoped results.
No new test is necessary for this reversible prose change.

A fresh implementer edits from this committed spec in a separate worktree.
A fresh reviewer checks the immutable head against source and retained evidence.
Mutation of product behavior is not applicable because no behavior changes.
The reviewer reruns the existing documentation mutation suites. The operator
reruns the scoped gate before pushing the exact commit to the fork.

## Work breakdown

1. Commit the canonical issue and this spec before public edits.
2. Delegate the bounded documentation patch and source audit.
3. Review the immutable patch and repeat its CPU-only checks.
4. Open one fork pull request against upstream `main`.

## Risks and stop conditions

Missing source evidence requires narrower wording. Unrelated preflight failures
must match the base before they can be reported as pre-existing. Never weaken a
gate. No merge, remote compute, service change, or model download is authorized.

## Owed

- [ISSUE-LOCAL-01M4Q12F0EMYDYQV1P1PNJKEB3](../issues/_owed/ISSUE-LOCAL-01M4Q12F0EMYDYQV1P1PNJKEB3.md): correct the Kolibri-1 public overview and usage.

## Source audit and verification

Checked against implementation base `23a85fa47` on 2026-10-12:

| Claim | Source or retained evidence |
|---|---|
| Compressed FP8 and BF16 loading | `src/vllm/model_executor/models/kolibri1_weights.cpp:97-119` |
| CPU dequantization and disabled-by-default cache | `src/vllm/model_executor/models/kolibri1_forward.cpp:175-205`, `kolibri1_dequant_cache.h:320-326` in the same directory |
| Streamed TT experts, CPU and TT dispatch, refused arms | `src/vllm/model_executor/models/kolibri1_registry.cpp:256-275`, `:289-294`, `:319-348` |
| CLI flags and executable | `examples/cli/main.cpp:109-151`, `examples/CMakeLists.txt:55-57` |
| Server flags and executable | `src/vllm/entrypoints/openai/server_main.cpp:498-541`, `:677-681`, `examples/CMakeLists.txt:98-115` |
| Thinking switch | `src/vllm/entrypoints/openai/reasoning_parsers/kolibri1.cpp:17-32` |
| Parser aliases | `src/vllm/entrypoints/openai/reasoning_parsers/abstract.cpp:69-75`, `src/vllm/entrypoints/openai/tool_parsers/abstract.cpp:99-106` |
| Checkpoint size and mixed storage | `.agents/specs/kolibri-1-cpu.md:17-30` |
| Partial CPU token agreement | `.agents/specs/kolibri-1-cpu.md:226-241` |
| Serving evidence and unmeasured plugin gate | `docs/bench-evidence/kolibri1-serve-20261008.md`, `.agents/oracles/aleph-alpha-inference.md` |

The short checkpoint revision is preserved from the pre-change feature row.
No full revision or shard hashes were found in that provenance. The usage
section marks them missing. The old feature row's dequant-at-load, owed TT,
and Transformers-generation descriptions are falsified by the anchors above.

With `PATH=/tmp/vllm-doc-tools/bin:$PATH`, all commands below exited 0:

```sh
python3 scripts/check-readme-structure.py
python3 scripts/check-supported-models.py
python3 scripts/check-surface-coverage.py
python3 scripts/check-benchmark-index.py
python3 scripts/check-quickstart-recipes.py
python3 tests/scripts/test_check_readme_structure.py
python3 tests/scripts/test_check_supported_models.py
python3 tests/scripts/test_check_surface_coverage.py
python3 tests/scripts/test_check_benchmark_index.py
python3 tests/scripts/test_check_quickstart_recipes.py
git diff --check
```

The existing mutation suites ran 19, 14, 46, 6, and 32 tests respectively.
A local path-and-heading check passed for every new link. Every new command
flag was checked against the source above. No product test or product mutation
was added because the patch changes prose only. No model download, model
execution, GPU use, or new performance measurement was performed.

The three new shell blocks pass `bash -n`. The curl body parses as JSON with
`enable_thinking` set to the boolean `false`.

Full preflight was invoked before edits. The required staged run was invoked
before commit and remains pending at this checkpoint. The earlier duplicate
run was stopped during mutation tests and is incomplete. Both runs report
missing PyYAML for `check-release-workflow` and missing CMake for
`check-test-registration`, matching the operator's base observations.
These are environment failures, not successful gates. The final operator
report must give the completed staged result before remote handoff.
Logs: `/tmp/vllm-docs-preflight-impl.log` and
`/tmp/vllm-docs-preflight-impl-staged.log`.
