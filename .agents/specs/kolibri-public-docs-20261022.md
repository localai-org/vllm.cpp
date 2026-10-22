# Kolibri-1 public guidance

## Scope

Correct documentation for `MODEL-TEXT-kolibri-1-kolibri1-for-causal-lm`.
Tracking issue: `ISSUE-LOCAL-01M5GRYH6WTN3CNQE1D025H5EM`.
Base: `845b3c59d`. No model lifecycle or performance disposition changes.

| Surface | Gap | Required change |
|---|---|---|
| README news | Missing default CPU cache announcement | Add a compact item linking the existing measurement and configuration |
| USAGE Kolibri section | Obsolete 1 GiB floor | Link the authoritative environment policy and explain the off switch |
| FEATURES model row | Implementation details overwhelm the table | Keep capability, validation limits, and links |
| Kolibri model guide | Stale download and Tenstorrent limitations | Match production entry points and retain unverified-oracle caveats |
| ENVIRONMENT cache section | Repeated history and emphatic prose | Explain the current policy once, preserving source and measurement links |

## Source and design

`src/vllm/model_executor/models/kolibri1_dequant_cache.h:465` defines
the divisor, 16 GiB floor, 32 GiB cap, and 8 GiB reserve.
`DefaultBudgetBytesFor` disables the cache below 24 GiB of reported
available memory. Unknown memory uses 16 GiB. The plan uses the greater
of the process-start and plan-time readings. An explicit MiB value bypasses
the derived policy, and zero disables caching.

Read the CLI model-resolution path and Kolibri Tenstorrent dispatch before
changing their descriptions. Use existing benchmark evidence at
`docs/benchmarks/kolibri1-cpu-dequant-cache-default.md` for links only.
Keep upstream oracle limitations from `.agents/oracles/aleph-alpha-inference.md`.
No upstream behavior or tests are ported by this documentation correction.

## Gates and evidence

Before editing, capture the stale claims and their contradictory source.
After editing, inspect each changed claim against source and check local links.
Run `git diff --check`, `scripts/check-readme-structure.py`,
`scripts/check-env-doc.py`, `scripts/check-supported-models.py`,
`scripts/check-benchmark-index.py`, and `scripts/check-agent-record.py`.
Run the corresponding README, environment, and benchmark checker suites
when present. Run preflight and report unchanged baseline failures separately.
The reviewer independently checks the immutable commit and runs the focused gate.
No new implementation-mirroring test is needed for a prose-only correction.

## Dependencies, risks, and stop conditions

Local Python and Git suffice. No GPU, model download, inference, or new
benchmark is permitted. Benchmark execution is not applicable because runtime
behavior does not change. Preserve existing measurements and provenance.
Stop for missing source evidence or an unrelated change needed to pass a gate.

## Work breakdown

1. Commit the issue and this spec.
2. A fresh implementer edits only the five public documents listed above.
3. A fresh reviewer checks source accuracy, readability, and local gates.
4. The operator repeats the focused checks and opens a fork PR.

## Now

The five public documents now describe the shipped cache policy and streamed
Tenstorrent experts. The oracle record confirms CPU execution with converted
BF16 weights, so the guides distinguish that reference from the released FP8
checkpoint. Full model parity remains unverified.

## Outcome

The README announces the default without adding a speed claim. The environment
guide owns the memory policy. The usage and model guides link to it, and the
feature table retains the capability and validation limits.

Source review preserved the CLI's local-path requirement. Only the server
resolves remote model IDs. No runtime default, model lifecycle, or benchmark
disposition changed. No GPU or model execution was needed.

Focused gates passed: agent record, README structure, environment documentation,
supported models, benchmark index, and `git diff --check`. The related suites
passed 19 README, 30 environment, and 6 benchmark tests. Full preflight encounters
missing PyYAML and build tools on the unchanged baseline. Independent review
and the fork pull request remain the integration steps.
