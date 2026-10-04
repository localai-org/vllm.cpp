# Refresh the public server guide

## Scope

Audit base: `fce36733b`, fetched from upstream `main` on 1 October 2026.
This documentation repair describes shipped behavior. It changes no runtime,
model lifecycle, oracle pin, or benchmark result. No GPU is required.

| Surface | Implementation anchors | Documentation repair |
|---|---|---|
| Decision routes | `src/vllm/entrypoints/openai/api_server.cpp` route registration; `src/vllm/entrypoints/openai/server_main.cpp` callback wiring; `docs/models/tev1.md` | Add the conditional decision routes to the server reference and Tev1 to README news |
| Prompt log probabilities | `serving_completion.cpp` choice builder; `serving_chat.cpp` response builder; `protocol.cpp` validation and serialization, all under `src/vllm/entrypoints/openai/` | Replace the stale HTTP-unavailable claim in the server reference and usage guide |
| EOS fallback | `src/vllm/v1/engine/input_processor.cpp`; `src/vllm/transformers_utils/hf_config.cpp`; `.agents/specs/tev1-eos-fallback.md` | Explain tokenizer fallback and config precedence in the server reference |

## Design

Keep reference prose short. Describe request conditions, response locations,
and registration conditions with links to existing examples and model recipes.
Give prompt log probabilities one authoritative explanation in the server
reference and link it from the usage guide. Correct the usage introduction
that calls every decision model non-generative. Tev1 samples one answer token
per question through the same engine as chat.

README news describes the newly shipped Tev1 decision path without claiming
GPU correctness, full vLLM parity, or speed. Preserve existing benchmark values.
Avoid unrelated ABI-version edits covered by open pull request 3343.

## Upstream chain and tests to port

No implementation is ported. The checked-in request validators, serializers,
route registration, engine input processing, and their existing tests define
this repair. Existing model specs retain their pinned reference evidence.
No new oracle execution, performance measurement, or runtime test is applicable.

## Gates

- Trace every changed behavior claim to code and existing tests.
- Run `scripts/check-readme-structure.py`, `scripts/check-site.py`, and
  `scripts/check-benchmark-index.py` with Python.
- Check changed local links and heading anchors. Parse any added JSON example.
- Run `git diff --check`, commit-style, and commit-trailer checks against the
  pinned base. No source, test, or build file may change.
- Run the CPU-only preflight. Compare failures with the untouched baseline.
  Do not claim full success if unrelated checks fail or prerequisites are absent.
- A fresh reviewer checks the immutable commit. Scratch mutations of links,
  anchors, and any added JSON must fail the scoped checks. Runtime reachability
  mutations do not apply to a documentation-only change.
- The operator independently reruns the focused gates before the fork push.

## Work breakdown

1. Commit this scope and its canonical local issue.
2. Delegate the public prose to a fresh implementer in a separate worktree.
3. Review the immutable implementation independently and repair any findings.
4. Record results here and open a reviewed pull request from the user's fork.

## Constraints and stop conditions

Only `README.md`, `docs/reference/server.md`, and the relevant server sections
of `docs/USAGE.md` are implementation scope. The spec and its issue carry the
work record. No lifecycle transition requires a shared matrix edit.
Use local CPU tools only. Do not download weights, use an accelerator, change
services, or merge upstream. Stop on an unresolved source contradiction.
Unspecified external environment values remain unavailable. This task needs
none. The user authorizes autonomous fork work and an upstream pull request.

## Owed

- ISSUE-LOCAL-01M3TPN0QZ8856GY1MHN2TKNHX: repair the public server descriptions.


## Coordination

The implementer uses the `SERVE-OAI-BASIC` helper role for this serving-document
repair. The canonical issue remains rowless and belongs to this spec. No
implementation or lifecycle transition of that matrix row is claimed.
The operator owns the record and the fork pull request. The implementer owns
only the three public files listed in scope.


## Outcome

The public edits are committed in `88e00bf3a` and the independently requested
EOS repair in `6bf6e37a8`. The operator integrated those commits as `299cb5302`
and `ed8d708a4`. The three public files are byte-identical to the reviewed
repair head.

The server reference now documents conditional decision routes, prompt log
probability response fields, request limits, and EOS precedence. The usage
page links to those explanations. README news links Tev1's setup and measured
limits. No runtime, test, build file, benchmark value, or lifecycle changes.
Benchmark disposition: NOT APPLICABLE to this documentation-only repair.
The existing Tev1 entry in `docs/FEATURES.md` already describes the shipped
decision lane and remains unchanged.

Source verification uses these anchors at the audited base:

| Claim | Code and existing tests |
|---|---|
| Conditional decision routes and Tev1 chat coexistence | `src/vllm/entrypoints/openai/api_server.cpp:1797`; `src/vllm/entrypoints/openai/server_main.cpp:1414,1610,2305`; `src/vllm/model_executor/models/tev1_inference.cpp:120` |
| Prompt log probability validation and response fields | `src/vllm/entrypoints/openai/protocol.cpp:63,845`; `serving_completion.cpp:435` and `serving_chat.cpp:1114` in that directory; `tests/vllm/entrypoints/openai/test_api_server.cpp:5308` |
| EOS precedence and missing-generation-config fallback | `src/vllm/v1/engine/input_processor.cpp:39`; `src/vllm/transformers_utils/hf_config.cpp:399,679`; `tests/vllm/v1/test_input_processor.cpp:598,611` |

The independent reviewer found one remaining contradiction in the usage page's
old EOS explanation. A fresh implementer replaced the duplicate with a link.
The reviewer checked the repair independently and reported no remaining static
findings. Scratch mutations of a local link and the EOS heading each fail the
link check. Each restoration is byte-exact and the restored checks pass.
No JSON example was added. Runtime reachability mutations do not apply.

Operator checks on the repaired public files, using local Python 3.14.7:

- `python3 scripts/check-readme-structure.py`: PASS.
- `python3 scripts/check-site.py`: PASS, 12 published documents.
- `python3 -m unittest tests.scripts.test_check_readme_structure`: 19 PASS.
- Changed local links and heading anchors: 7 PASS against `fce36733b`.
- `python3 scripts/check-agent-record.py`: PASS.
- `python3 scripts/check-tree-compiles.py --base fce36733b`: no C++ source,
  header, or build file in scope.
- `git diff --check`, commit style, and commit trailers: PASS.
- `python3 scripts/check-benchmark-index.py`: FAIL, with the same 16 orphan
  detail files as the base. This repair changes none of those files or the index.

No model was loaded and no accelerator was used. Source inspection and the
existing tests establish what the public descriptions mean. This session
makes no new runtime correctness or performance claim.

The operator's full `scripts/agent-preflight.sh --quiet` exits 1 with 51
failed checks and 14 skips. The run starts before public edits. Its later
README and site checks read the repaired files. Logs remain at
`/tmp/vllm-docs-baseline.txt` in this session's environment.

Failures include existing benchmark-index, release-state, environment-document,
test-registration, and oracle-pin drift. The Alpine environment lacks PyYAML,
NumPy, CMake, Ninja, and binary-inspection tools. Process-management fixture
suites also fail here. The wrapper fixes its comparison base to the stale
`origin/main` at `1de097c46`, which includes unrelated upstream commits in its
style and trailer checks. Checks of this contribution instead use the fetched
base `fce36733b` and pass. The diff-scoped compilation check finds no compilation
work for this change. No full-preflight or integration-readiness success is
claimed.

The implementation's full preflight also exits 1 with 51 failures and 14
skips. The operator independently compares both sets with the initial run:
no failure or skip is added or removed. Its log is
`/tmp/vllm-docs-implement-preflight.txt`. The final EOS repair changes only a
usage paragraph to a checked local link. Focused checks were rerun on that
repair by its implementer, the reviewer, and the operator.

The independent review's full preflight on `88e00bf3a` exits 1 with the same
51 failures and 14 skips. The operator compares those sets independently and
finds no additions or removals. The reviewer then checks the final repair at
`6bf6e37a8`, including the seventh link and fresh scratch mutations, and returns
PASS for the documentation. Its log is `/tmp/vllm-docs-review-preflight.txt`.

## Now

Documentation review is complete. The local issue stays open until the fork
pull request lands. No upstream merge is authorized by this task. Resume with
`git log --oneline -- .agents/specs/docs-server-surface-refresh.md` and the
fork branch `docs/source-alignment-20261001`.
