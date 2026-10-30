# GATE-PR-BODY-TRAILERS: read the bytes the squash will land, in the operator's own shell

**Issue:** [#1263](https://github.com/mudler/vllm.cpp/issues/1263).
**Kind:** one new operator command plus the procedure that names it, and (since
2026-09-11) one rule the command applies itself. `scripts/check-commit-trailers.py`
is still not touched by this row; the added rule lives in `scripts/agent-pr-body.py`,
which this row owns, and it reads the canonical local issue records rather than any
commit trailer.
**Row:** `GATE-PR-BODY-TRAILERS`.

## Now

The repository sets `squash_merge_commit_message = PR_BODY`, so a pull request
body **is** the landed commit message. One guard reads it, at
[`.github/workflows/ci.yml:626-635`](../../.github/workflows/ci.yml) in
`commit-protocol-tag`: it writes `$PR_BODY` to a file and runs
`scripts/check-commit-trailers.py --message-file <file> --filled` (#848).

That guard is correct, and it is not a precondition of merging. On #1257 it was
still `pending` at merge time because the runner pool was saturated. It did not
fail; it never rendered a verdict. The body carried
`Assisted-by: AGENT:claude-opus-5 CLI`, and that string is the message of
`281b4bc76c0e` on `main`, where it cannot be repaired because `main` is never
force-pushed. #1262 closed the lane with an enumerated exception; this row
closes the hole that let the body through.

**The gap, measured at `origin/main` `489a9a4c0`.** The only caller of
`--message-file` anywhere in the tree:

```console
$ grep -rn -- --message-file .github scripts .agents AGENTS.md
.github/workflows/ci.yml:635:              --message-file "$body_file" --filled
```

Everything else the grep returns is the checker's own argument parser, its own
suite, or prose in a spec. No local command reads a body. An operator merging at
their own keyboard has nothing to run.

This is the fourth issue in the class: #406, #581, #1058, #1262.

## Scope

In scope:

- `scripts/agent-pr-body.py`: fetch one pull request body and hold it to the
  commit contract, by delegating to the checker that already implements it.
- `tests/scripts/test_agent_pr_body.py`, registered in `scripts/agent-preflight.sh`
  and in `.github/workflows/ci.yml`.
- The operator step, in `AGENTS.md` under `## Landing work` and `## Commands`,
  and in [`.agents/workflow.md`](../workflow.md) under `## Landing`.
- The [#1298](https://github.com/mudler/vllm.cpp/issues/1298) row in
  [`issue-index.md`](../completed/issue-index.md), for a defect this row measured and does
  not fix. See `## Owed`.

Out of scope, deliberately:

- `scripts/check-commit-trailers.py`. The rule exists, has one implementation,
  and is already gated by `tests/scripts/test_check_commit_trailers.py`. This
  row adds a second **caller**, never a second copy.
- The CI guard. It stays exactly as it is. A local check is a belt to its
  braces, not a replacement: CI reads the body on the forge's own event payload
  and catches an edit made after the operator looked.
- The `## Owed` row in `issue-index.md` for #1263. That row is already appended
  and the file is append-only, so it is not edited and no second row is added.
- Repairing `scripts/agent-integration.py`. See `## Owed`.

## Design

### The split the constraint forces

No checker in this repository may make a network call. Reading a live pull
request body needs one. So the two halves are separated, and only one of them is
a gate:

| Half | Where | Network | Gated by |
|---|---|---|---|
| Validate a message | `check-commit-trailers.py --message-file - --filled` | never | `test_check_commit_trailers.py`, `agent-preflight.sh`, CI |
| Fetch a body | `scripts/agent-pr-body.py --pr N` | one `gh` call | nothing; it is a command an operator runs |

The fetch asks for `--json body` and parses the JSON in process, rather than
asking `gh` to apply `--jq .body`. A body that is absent, `null`, or not a string
is then distinguishable from a body that is the empty string, and the answer
does not depend on how one `gh` build renders a null field.

The new tool is the seam between them, and it carries an **offline door**:
`--body-file PATH` validates a message that is already on disk and never invokes
`gh`. Every case in the suite drives that door or a stubbed `gh` on `PATH`, so
the suite makes no network call and can run on a CI runner with no forge
credentials. Nothing in `agent-preflight.sh` or in a CI job invokes the fetching
half.

### Why a script, and not the one-liner the issue proposed

#1263 proposed:

```sh
gh pr view <N> --json body --jq .body | check-commit-trailers.py --message-file - --filled
```

Measured at `489a9a4c0`, against a pull request number that does not exist:

```console
$ gh pr view 999999 --json body --jq .body | python3 scripts/check-commit-trailers.py --message-file - --filled
commit trailer check FAILED: the message is empty. Under PR_BODY an empty body
lands a commit with no trailers at all
rc=1
```

The forge said `GraphQL: Could not resolve to a PullRequest with the number of
999999`. The pipeline reported a verdict **about the body**. Two defects in one
line, and both are recorded failure modes here:

1. **The exit status dies in the pipe.** `$?` is the checker's, never `gh`'s.
2. **A broken instrument fails toward a verdict about the code.** "Cannot reach
   the forge" and "the body is empty" render identically, and an operator who
   fixed a body that was never read would be repairing the wrong thing.

The protocol's answer to the first is `REMOTE_UNVERIFIED`, and a skip is not an
answer. That needs `gh`'s own status, which needs it not to be piped. So the
tool runs `gh` with a captured exit status, and feeds the body to the checker on
**stdin** from its own process, never through a shell pipeline and never on a
command line, because a body is attacker-controlled text on a fork.

### Three outcomes, three exit statuses

| Status | Meaning | Line |
|---:|---|---|
| 0 | the body satisfies the contract it will land under | the checker's `OK` |
| 1 | the body was read and fails the contract | the checker's own error list |
| 2 | not ours | `argparse` exits 2 on a usage error, and it is not overridden |
| 3 | the body was not read | `REMOTE_UNVERIFIED: <what gh said>` |

`scripts/agent-ready.py` returns 1 for its own `REMOTE_UNVERIFIED`, and this
tool deliberately differs. This row exists because "not read" was allowed to
pass for "read and fine"; collapsing "not read" onto "read and bad" would leave
the same two states indistinguishable to any future caller. Both are refusals,
so a caller testing `rc != 0` is correct either way.

### Which repository `--pr` means

A pull request number is unique inside one repository and means nothing
outside it, and this tool derived its repository from `origin` alone. On a
maintainer checkout that is the repository the pull request lands in. On a
**fork checkout** it is the fork, which carries none of the upstream numbers,
so every invocation exited 3: `REMOTE_UNVERIFIED` for a pull request that
exists. Measured on the `phantomic12/vllm.cpp` checkout at `c6c0d8674`,
2026-10-30:

```console
$ python3 scripts/agent-pr-body.py --pr 3474
REMOTE_UNVERIFIED: GraphQL: Could not resolve to a PullRequest with the number of 3474.
rc=3
$ python3 scripts/agent-pr-body.py --pr 3474 --repo localai-org/vllm.cpp
OK: commit trailer contract
rc=0
```

`gh pr view 3474 --repo phantomic12/vllm.cpp` exits 1 and the fork has no pull
requests at all, so the defect is repository **selection**, not reachability.

So `--pr` now asks the remotes in a fixed order, `origin` first and `upstream`
second, and the first repository that carries the number answers:

- **`origin` first** keeps every invocation that works today meaning exactly
  what it meant. A maintainer whose `origin` is canonical never spends a call
  on somebody else's fork, and a number that exists in both repositories is
  still read from the one this clone was made from.
- **`upstream` second** is the fork case: the first read is unknown, the second
  answers, and the run is `rc=0` rather than `rc=3`.
- **A fallback is named, never silent.** When the answer does not come from the
  first candidate, a `NOTE:` line on stderr says which repository answered, so
  a fork-local pull request with a colliding number cannot be mistaken for the
  upstream one without the operator being told.
- **No candidate answering is `REMOTE_UNVERIFIED`, not "absent".** The message
  names every repository tried and carries each failure, because only a
  `Could not resolve` reply proves absence and the rest may be a forge that did
  not answer.

`--repo` still overrides the whole list and is asked exactly once, which stays
the documented way to read a pull request the configured remotes cannot
resolve.

### Naming

`agent-pr-body.py`, in the family of `agent-ready.py`, `agent-integration.py`,
`agent-role.py` and `agent-start.py`: operator and helper tools, which are the
scripts in this tree that already reach the forge. `scripts/check-*.py` means an
offline tree checker that `agent-preflight.sh` runs with no arguments, and
`scripts/check-pr-size.py:170` classifies exactly that spelling as a
`governance_checker` carrying the checker-evidence contract. This script
implements no rule and makes a network call, so `check-` would be wrong twice:
it would tell a reader it is safe to wire into preflight, and it would enrol a
delegating tool in an evidence contract meant for rules.

### Where the operator meets it

`AGENTS.md` `## Landing work` is the section an operator reads before a merge,
and it already states that the body is the landed commit message. The step is
added there, in `## Commands`, and in the `## Landing` section of the workflow
guide. The suite asserts all three literally, so deleting the operator-facing
line is red. For a command whose production entry point is a human typing it,
that document line **is** the call site, and this is its mutation.

### Two homes that were considered and rejected, with the measurement

**`scripts/agent-ready.py`** already fetches the pull request and already has an
offline fixture door (`--pr-json`), so it is the cheapest possible home. It is
the wrong one: it runs before the **handoff**, and #1257's body was stale at
**merge**, several reviews later. A pre-handoff read cannot see a body edited
afterwards, which is the case this row exists for. Adding the check there too is
not refused and is not needed here.

**`scripts/agent-integration.py`** is the documented pre-merge command in
`AGENTS.md` `## Commands` and would be the natural home. **It cannot run.**
Measured at `489a9a4c0`:

```python
>>> integration.cutover_oid()
ValueError: missing policy cutover: [Errno 2] No such file or directory: '<worktree>/.agents/policy-cutover'
```

`main()` calls `cutover_oid()` before it can report anything, and
`.agents/policy-cutover` was deleted by `0f3e44eee` on 2026-08-09. Every
invocation exits `INTEGRATION FAILED` whatever the tree holds. Wiring this check
there would land it behind a permanent refusal, which is the "nothing lands
dead" failure in its purest form. Filed as
[#1298](https://github.com/mudler/vllm.cpp/issues/1298) and owed below.

## Risks

| Risk | Mitigation |
|---|---|
| The tool grows a copy of the rule | It has no grammar, no regex over trailers and no vocabulary of its own. `test_the_rule_is_not_reimplemented` asserts the source contains no `Assisted-by` matching of any kind and that it invokes the checker with `--message-file` and `--filled`. |
| `--filled` is dropped and the placeholder passes | `test_the_template_placeholder_is_refused` drives the literal `AGENT:MODEL [TOOL]`, which is legal without the flag and illegal with it, so the flag is the only thing that can produce the verdict. |
| An operator reads a failed fetch as a bad body | Exit 3 and the `REMOTE_UNVERIFIED:` prefix, both pinned. `argparse` already owns 2, so the refusal codes do not collide with a usage error. |
| A successful fetch of a genuinely empty body is misread as a failed fetch | `test_an_empty_body_read_successfully_is_a_contract_failure_not_remote_unverified` pins exit 1 for that case, against exit 3 for the unreachable one. |
| The command is documented and never run | It is one line in the section an operator already reads at merge time, and CI keeps reading the body independently. This row narrows a window; it does not claim to close every path. |
| The suite needs a forge | It never calls one. `gh` is stubbed on `PATH` by the cases that exercise the fetching half. |
| A fork's own pull request shadows the upstream one of the same number | `origin` is asked first, so no answer that exists today changes meaning, and the `NOTE:` line names the repository that answered whenever it is not the first candidate, so the shadowing is stated rather than silent. |

## Tests

All in `tests/scripts/test_agent_pr_body.py`, before the `__main__` guard,
registered in `scripts/agent-preflight.sh` `SUITES` and in the CI step that
already runs `tests/scripts/test_agent_gates.py`. Twenty-nine cases, none of
which reaches a network.

| Case | Proves |
|---|---|
| `test_the_exact_landed_malformed_value_is_refused` | The bytes that landed on `281b4bc76c0e` are refused offline. |
| `test_a_filled_body_is_accepted` | A correct body passes, so the refusal above is a verdict and not a constant. |
| `test_the_template_placeholder_is_refused` | `--filled` is passed. That value is legal without the flag. |
| `test_a_body_with_no_trailer_paragraph_is_refused` | The whole contract applies, not one trailer of it. |
| `test_an_unreadable_body_file_is_unverified_not_a_verdict` | A message that could not be read says nothing about a message. Exit 3. |
| `test_exactly_one_of_pr_and_body_file` | The argument contract. |
| `test_the_pr_number_is_rejected_unless_it_is_a_positive_integer` | No shell metacharacter reaches `gh`, proven by a stub that stays untouched. |
| `test_a_fetched_body_is_held_to_the_same_contract` | The fetching half feeds the validating half. Stubbed `gh`. |
| `test_a_fetched_filled_body_passes` | The fetched arm has both verdicts, not only the refusal. |
| `test_an_unreachable_forge_is_remote_unverified_and_never_a_verdict` | Exit 3, `REMOTE_UNVERIFIED`, `gh`'s own message, and never the checker's empty-message line. |
| `test_an_absent_gh_is_remote_unverified` | A missing tool is unknown, not a bad body. |
| `test_malformed_forge_json_is_remote_unverified` | Four shapes: not JSON, a list, no `body` key, a null `body`. |
| `test_an_empty_body_read_successfully_is_a_contract_failure` | Exit 1, not 3. Read and empty is not the same state as never read. |
| `test_gh_is_not_invoked_by_the_offline_door` | `--body-file` makes no network call, which is what makes it gateable. |
| `test_the_rule_is_not_reimplemented` | One implementation, two callers. |
| `test_the_checker_it_delegates_to_exists` | The delegate is a path, and a path can be moved. |
| `test_the_landing_procedure_names_the_command` | `AGENTS.md` and `.agents/workflow.md` name it, so the entry point cannot be deleted silently. |
| `test_the_suite_is_registered_where_gates_run` | A suite nothing runs is not a gate. |
| `test_the_spec_table_names_exactly_these_cases` | This table is compared with the loaded suite, not sampled, so it cannot go stale inside the change that writes it. |
| `test_closing_an_issue_with_no_local_record_warns_but_is_accepted` | A missing record is migration debt rather than a contradiction: an authority that does not exist cannot disagree with the body, and this arm fired on 22 of 24 refusals across the open queue. |
| `test_closing_an_issue_whose_record_reads_open_is_refused` | The arm that earns a refusal: the record exists and reads OPEN, so closing the mirror leaves the authoritative half saying OPEN. |
| `test_a_bare_reference_without_a_closing_keyword_is_accepted` | The gate is narrow deliberately. Open pull requests carry six to ten bare `#N` citations each, most without a local record, and failing those would fire the gate on ordinary work rather than on the defect. |
| `test_a_fork_checkout_reads_the_pull_request_from_upstream` | The defect in one case: `origin` is the fork and cannot answer, the run continues to `upstream`, and the exit is 0 rather than 3. Driven in process on stubbed `subprocess.run`, because the repository choice depends on this checkout's remotes and a `gh` stub on `PATH` cannot vary them. |
| `test_the_fallback_names_the_repository_it_read` | A fallback is never silent: the `NOTE:` line names the repository whose bytes were judged, so a fork-local number cannot shadow the upstream one unannounced. |
| `test_origin_answering_is_not_second_guessed` | A maintainer checkout keeps today's meaning and spends one call, because `upstream` may be somebody else's fork. |
| `test_one_repository_listed_twice_is_asked_once` | Two remotes naming one repository are one candidate, not two calls. |
| `test_no_repository_carrying_it_is_unverified_and_names_each_one` | Exit 3 rather than 1, and the refusal names every repository asked with its own failure, so "absent" is never inferred from a forge that did not answer. |
| `test_an_explicit_repo_is_asked_exactly_once` | `--repo` replaces the candidate list rather than leading it. |
| `test_a_checkout_without_origin_is_still_unverified` | The fallback does not invent a repository when `origin` is absent. |

## Gates

| Gate | Expected |
|---|---|
| `python3 tests/scripts/test_agent_pr_body.py` | all cases green |
| `python3 tests/scripts/test_agent_gates.py` | unchanged green |
| `python3 -m unittest tests.scripts.test_check_commit_trailers` | unchanged green |
| `python3 scripts/agent-pr-body.py --body-file <malformed fixture>` | rc=1 naming the malformed value |
| `scripts/agent-preflight.sh --fail-on-skip` | every gate green, no skip |

## Stop conditions

- Stop and return `NEEDS_DECISION` if closing this needs a change to
  `check-commit-trailers.py`. It does not; a second implementation of the rule
  is the outcome this row exists to avoid.
- Stop if the check has to become a CI gate that makes a network call. The
  constraint is not negotiable and the forge already runs the guard.
- Stop if `gh` has to write anything. This command reads.

## Owed

ISSUE-GH-1298: `agent-integration.py`
raises before it reads anything, because the `.agents/policy-cutover` anchor it
loads was deleted on 2026-08-09 and never restored. Measured here and not fixed
here: the repair is a decision about whether `--cutover` belongs in that command
at all, which is a gate semantic change owing its own spec, red-before evidence
and reviewer. `tests/scripts/test_agent_gates.py:205` exercises `cutover_oid`
against a synthetic repository it writes the anchor into, so the function is
tested while the command is not, and no suite is red today.

## Outcome

Filled when the row reaches `DONE`.
