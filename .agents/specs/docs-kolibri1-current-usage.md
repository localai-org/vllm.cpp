# Document current Kolibri-1 usage

Work ID: `DOCS-KOLIBRI1-CURRENT-USAGE`.
Base: `c6a971e53`.

## Now

Documentation audit only. No model, backend, or benchmark lifecycle changes.
The feature table describes an older loader and calls the Tenstorrent arm owed.
The README news and usage guide omit Kolibri-1.

## Scope and source inventory

| Surface | Source | Correction |
|---|---|---|
| README news | `src/vllm/model_executor/models/kolibri1_registry.cpp` | Announce the reachable model with explicit validation limits |
| Features model row | `src/vllm/model_executor/models/kolibri1_weights.cpp`, `kolibri1_forward.cpp`, `kolibri1_registry.cpp` in the same directory | Replace historical prose with the current loader, CPU and TT availability, and refused arms |
| Usage recipe | `examples/simple/main.cpp`, `src/vllm/entrypoints/openai/server_main.cpp`, `include/vllm.h`, Kolibri registry | Add source-checked local-checkpoint instructions, defaults, and memory limits |
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
