# The issue index is derived, attribution is enforced once, and intake gets an exit rule

Issues: [#2290](https://github.com/mudler/vllm.cpp/issues/2290),
[#883](https://github.com/mudler/vllm.cpp/issues/883),
[#2157](https://github.com/mudler/vllm.cpp/issues/2157),
[#467](https://github.com/mudler/vllm.cpp/issues/467),
[#1808](https://github.com/mudler/vllm.cpp/issues/1808),
[#2298](https://github.com/mudler/vllm.cpp/issues/2298).
Row: `ENG-RECORD-CONFLICT-SURFACES`. Follow-up to
[#364](https://github.com/mudler/vllm.cpp/issues/364) /
[retire-shared-record-surfaces.md](retire-shared-record-surfaces.md), and the
same move [now-derived.md](now-derived.md) made for `.agents/NOW.md`.

`.agents/issue-index.md` is the last shared surface every pull request must
write, and it is the one the invariant #364 added to `AGENTS.md` cannot protect —
because that invariant admits a shape GitHub does not implement.

## Scope

**In scope.** `.agents/issue-index.md` and its `merge=union` attribute; the four
consumers that read it; `scripts/check-issue-index-append-only.py`; the
deleted-checker case in `scripts/check-pr-size.py` (added to scope 2026-08-29,
see below); the
`UNOWNED_HIGH_WATER` ratchet; the post-hoc commit-trailer walk over `main`;
`scripts/agent-preflight.sh`'s checker coverage; and four `AGENTS.md` sections —
§"Every change starts from an issue", §"Records", §"Landing work", and the
§"How work gets done" force-push sentence.

**Out of scope.** Every other record surface, every public document, the
`agent-pr-body.py` contract itself (it is the thing kept, not changed), and all
product source. No kernel, model, backend or gate semantic outside the four
named checkers moves.

## Upstream chain

None. vLLM has no counterpart to this protocol machinery, so the mirror rule does
not apply and there is no upstream `file:line` to port from. Governed instead by
`AGENTS.md` §"Changing the rules or a checker", which requires a spec, a
red-before test or mutation, and green-after evidence, and forbids making a red
gate green by deleting an assertion.

## Our baseline — what was measured

Measured 2026-08-29 at `origin/main` `e541be98`.

**The file is written by more than half of all commits.** 115 of the last 200
commits touch `.agents/issue-index.md`. It is 854 rows.

**It is the dominant conflict source, and the situation regressed.** 16 of 21
open pull requests report `CONFLICTING` (76%). #364 measured 16 of 29 (55%) on
2026-08-11 and set out to fix it; the ratio is worse now, not better.

**The index alone causes it.** For #2267, #2248, #1726 and #1703, `git
merge-tree` against `origin/main` reports a clean merge locally and exactly one
conflict — `.agents/issue-index.md` — once the union driver is disabled with
`-c merge.union.driver=false`. Those four PRs share no other conflicting path.
#2248 touches 21 files, 20 of them Vulkan shaders and backend source, and the
index is the only one that collides.

**A PR born conflicted never gets a verdict.** #2248 was created 2026-08-29,
conflicted, and carries **0 check-runs**. Not red — never scheduled. #2267 and
#2281, opened the same day and merging clean, carry 29 each.

**The rule permits a shape the forge does not implement.** `AGENTS.md` §Records
lists three admissible record shapes. The second — *"a genuinely append-only file
that can union-merge"* — is not deliverable on GitHub: the forge does not run
`.gitattributes` merge drivers, so the driver hides the collision locally and the
pull request conflicts anyway. #883 recorded this on 2026-08-15 and the shape
stayed in the rule. `.agents/issue-index.md` is a *correct implementation* of a
defective rule, which is why fixing the file alone would not hold.

**Half of the checker is union-driver defense.** `check_issue_index`
(`check-agent-record.py:2018`) spends its preamble-drift check, its duplicate-row
check and the whole of `check-issue-index-append-only.py` on failure modes that
exist only because two branches append to one file. Their own messages say so:
*"A union merge DUPLICATES an edited preamble line"*, *"Under `merge=union` a
duplicate is what two branches appending the same issue look like"*. #1619,
#1644, #1002 and #2266 are four further open issues about that driver misbehaving.

**Attribution is enforced twice, and the two copies disagree.** `ci.yml:872`
skips merge commits (`>1 parent`); `scripts/check-commit-trailers.py` has no such
rule and is called by that same job (#2157). Twelve open issues — #406, #438,
#467, #581, #653, #841, #858, #870, #1058, #1260, #1262, #2157 — are all the same
cause: attribution re-parsed out of commit text *after* GitHub has rewritten it.
`scripts/agent-pr-body.py` already reads the bytes that will land, before they
land.

**"All gates green" covers 12% of the gates.** 42 checkers exist in `scripts/`;
`agent-preflight.sh` names 5 (`check-agent-record.py`, `check-commit-style.py`,
`check-commit-trailers.py`, `check-issue-index-append-only.py`,
`check-now-current.py`). The remaining 37 reach the working tree only where their
test suite happens to call the checker against the real `ROOT`, which varies per
suite and is a property of the test, not of the gate.

**Intake has no exit.** 701 open issues, every one filed in August 2026 — ~29 new
open issues per day. A mechanical sweep of the 62 protocol issues found six the
tree already falsifies (#420, #1543, #1213, #647, #587, #1533). Details in #2298.

## Port map

| Item | From | To |
|---|---|---|
| Index storage | `.agents/issue-index.md`, tracked, `merge=union` | `gh issue list`, refreshed into untracked `.agents/issue-index.generated.md` |
| Row association | index `Row` column | a `Row:` line in the issue body |
| Historical rows | 854 rows in the tracked file | `.agents/issue-index.md`, moved not deleted |
| Append-only gate | `check-issue-index-append-only.py` | deleted; a generated file cannot be appended to wrongly |
| Ownership gate | `UNOWNED_HIGH_WATER = 33`, global ratchet | diff-scoped: a change owns the issues it references |
| Attribution | PR-body gate **and** a walk over `main` | PR-body gate only |
| Preflight coverage | 5 named checkers | every checker in `scripts/check-*.py` |

## Design

**1. `scripts/agent-issue-index.py`.** Mirrors `scripts/now.py:127` exactly,
which is the in-tree precedent for a derived record: one network call, a timeout,
and degradation to `REMOTE_UNVERIFIED` rather than an exception or a silent
empty result. `--refresh` calls `gh issue list --state open --json
number,title,body,labels`, parses `Row:` out of each body, and writes
`.agents/issue-index.generated.md` in the table shape the consumers already
parse, so `ISSUE_ROW` (`check-agent-record.py:1960`) needs no change. A failed
call writes nothing — a partial snapshot is worse than none, because a consumer
cannot tell it apart from a complete one.

**2. The snapshot is untracked.** `.gitignore` carries it. No pull request writes
it, so it cannot conflict; and because nothing tracks it, it cannot be the lock
in a new place. `.agents/issue-index.md` and its `merge=union` line in
`.gitattributes` both go.

**3. Consumers SKIP loudly on absence.** `check-agent-record.py`,
`check-pr-size.py` and `check-symbol-anchors.py` read the snapshot when present.
When it is absent or older than 24h they report `SKIP` with the reason and the
refresh command — never a silent pass, and non-zero under
`agent-preflight.sh --fail-on-skip`. This is the shape `now.py` already uses and
the shape #467 says preflight got wrong.

**4. The ownership ratchet dies; ownership becomes diff-scoped.** A global count
over a *remote* surface cannot be a per-commit obligation: it moves when anyone
files an issue anywhere, so a ratchet on it would red `main` for reasons no
commit caused — a new lock in a new place. Instead the gate resolves the issues
*this change references* (its branch commits and its PR body) and requires each
to name a `Row:` or appear under a spec's `## Owed`. `owed_issues()`
(`check-agent-record.py:2000`) already globs specs network-free and is unchanged.

**5. Attribution is enforced once.** `scripts/agent-pr-body.py` stays and is the
authority, because it reads the exact bytes `squash_merge_commit_message =
PR_BODY` will land.

`--range` had THREE deployments, and the first draft of this spec collapsed
them: `agent-preflight.sh` pre-push, the CI pull-request lane, and the CI PUSH
lane on `main`. The first two are diff-scoped to commits their author wrote and
can still amend or force-push, and they stay. The third re-read LANDED history,
where the only repair is rewriting `main`, which AGENTS.md forbids outright — so
a violation there could never clear itself.

It was also redundant. Anything that reaches `main` through a pull request has
already had its landing bytes checked by the `--message-file` gate before they
froze. The push lane's only unique coverage was a direct push to `main`, which
§"Landing work" already prohibits.

What it produced instead was forgiveness work. Two mechanisms existed solely to
excuse landed commits: `scripts/ci-enforcement-floor.txt`, whose current value
forgives 42 commits dated 2026-08-13 to 2026-08-24, and
`LANDED_MESSAGE_EXCEPTIONS`, which carried one. That is 43 commits on `main`
that violate the contract and cannot be repaired, each forgiven by a deliberate
reviewed act. A gate whose three-week output is 43 forgiveness decisions rather
than 43 prevented defects is measuring the wrong thing.

So both trailer steps of `commit-protocol-tag` become `pull_request`-only, and
`LANDED_MESSAGE_EXCEPTIONS` goes with the reason it existed for — a dead
exception list invites a live one. The floor FILE stays: it still governs
`documentation-checkpoint` and `agent-record`'s role-discipline step, which do
walk the push lane, and its value does not move. Only its scope narrows, and its
own comment and `ci-walk-base.py`'s docstring are corrected to stop naming the
trailer steps.

**6. Preflight runs the checkers.** `agent-preflight.sh` discovers
`scripts/check-*.py` by glob and runs each, rather than naming five. A checker
that needs an argument it cannot supply is `SKIP` with its reason, not silence.

**7. AGENTS.md.** §Records drops the union-merge shape and says why, leaving two
admissible shapes. §"Every change starts from an issue" points at `gh` and the
`Row:` line, and gains the two exit rules from #2298: a PR that lands a fix
closes its issue, and an issue the tree falsifies is closed with that evidence
rather than re-specced. §"Landing work" line 512's unscoped *"Never force-push"*
is scoped to `main`, matching line 194 (#1808).

### Added to scope during implementation

**`check-pr-size.py` cannot express a deleted checker.** Its evidence contract
asks a checker change to prove a guarantee moved: the paired suite must fail
against the BASE checker and pass against the HEAD one. A DELETION has no head
checker to run, and when the suite goes in the same change there is no module to
import, so the contract collapsed into `ModuleNotFoundError` reported as "HEAD
checker/test pair failed" — an absent checker reading as a broken one.

`pr-size` is a REQUIRED check, so this row could not land at all: retiring
`check-issue-index-append-only.py` is exactly the shape the contract cannot
describe. That makes it a blocker rather than an adjacent tidy, which is why it
is taken here instead of filed. It IS checker semantics, so it gets the full
treatment AGENTS.md §"Changing the rules or a checker" demands — this scope
entry, three red-first cases, and green-after evidence — rather than the in-flow
rule, which explicitly does not cover a checker-semantic change.

Keyed on the CHECKER's absence via `git diff --diff-filter=D`, never on the
suite's and never on a diff that merely looks deletion-shaped, so a live checker
cannot borrow the exemption. The second case pins that half: without it, marking
every checker change "deleted" would silence the whole contract.

**`docs/bench-evidence/*.csv` has no path class**
([#2316](https://github.com/mudler/vllm.cpp/issues/2316), from #2289). Filed
first as not-blocking, on the reasoning that the CHECKER was green and only the
suite's whole-tree sweep was red. **That was wrong, and the correction is the
interesting part.** Editing `check-pr-size.py` makes
`tests/scripts/test_check_pr_size.py` its mutation evidence, and the evidence
contract runs the WHOLE module — so one unrelated red anywhere in that suite
fails the HEAD pair and blocks the change. A checker's suite is load-bearing for
every future edit to that checker, which means a red in it is never merely
someone else's problem once you touch the checker.

Fixed here: `csv` joins `BENCH_EVIDENCE`, because `ncu --csv` writes one and a
profiler export is evidence exactly as a `.log` is. Two cases, one for the
classification and one pinning that a `csv` elsewhere under `docs/` still fails
closed, so the widening cannot be read as "any csv anywhere". The second half of
#2316 — that the whole-tree sweep executes in no lane at all, which is the #467
shape — stays open and is NOT taken here.

## Tests to port

None from upstream; there is no upstream. Written red-first here:

| Suite | Red before, green after |
|---|---|
| `tests/scripts/test_agent_issue_index.py` | a `gh` failure returns `REMOTE_UNVERIFIED` and writes no file; a partial payload writes no file; a well-formed payload writes rows the existing `ISSUE_ROW` regex matches; a body with no `Row:` renders the dash |
| `tests/scripts/test_agent_record.py` | absent snapshot is SKIP-with-reason, not pass and not error; a referenced issue naming no `Row:` and absent from `## Owed` FAILS; the same issue listed under `## Owed` passes |
| `tests/scripts/test_check_commit_trailers.py` | the PR-body arm still fails a body missing the block and still passes both squash shapes; the removed walk's cases are deleted, not weakened |
| `tests/scripts/test_agent_preflight_skip_report.py` | a checker added to `scripts/` is run without editing preflight; a checker that cannot run reports SKIP and reddens `--fail-on-skip` |
| merge-shape regression | two branches that each file an issue merge with no conflict, with the union driver disabled — RED at `origin/main`, GREEN after |

The merge-shape regression is the one that proves the campaign's claim, so it
runs `git -c merge.union.driver=false merge-tree`, which is what GitHub does.
Testing it *with* the driver would reproduce the local false green this row
exists to remove.

## Gates

```sh
python3 scripts/check-agent-record.py
python3 scripts/agent-issue-index.py --refresh && python3 scripts/check-agent-record.py
scripts/agent-preflight.sh --staged --fail-on-skip
python3 -m pytest tests/scripts/test_agent_issue_index.py tests/scripts/test_agent_record.py \
  tests/scripts/test_check_commit_trailers.py tests/scripts/test_agent_preflight_skip_report.py
python3 scripts/agent-pr-body.py --pr <N>
```

Every one must be run on an idle tree and reported with its exit status, per
`.agents/verification.md`. `$?` is read via `out=$(cmd); rc=$?`, never after a
pipe.

## Evidence

Measured at `origin/main` `e541be98` on 2026-08-29; the numbers are in
§"Our baseline". Re-derivation for each: `git log --format=%H -200 | ...` for the
115/200 write rate, `gh pr list --json mergeable` for 16/21, `git -c
merge.union.driver=false merge-tree --write-tree origin/main <sha>` per PR for
the single-file attribution, `gh pr view 2248 --json statusCheckRollup` for the
zero check-runs, and `ls scripts/check-*.py | wc -l` against the names in
`agent-preflight.sh` for 5 of 42.

## Dependencies

None on product rows. It touches `.agents/issue-index.md`, which nearly every
in-flight branch also writes, so it lands as one change and the affected branches
rebase onto it once — the last conflict of this class rather than a new one.

**Measured, not assumed.** Re-running `git -c merge.union.driver=false
merge-tree` for all 16 conflicting pull requests against this branch shows the
conflict does NOT disappear for them: git follows the rename, so a branch that
appends an index row now conflicts on `.agents/completed/issue-index.md`
instead. That is the one-time migration cost, and it is the whole of it. The
resolution is mechanical and always the same — **drop the branch's index hunk**,
because the archive is frozen and the row's content now lives in the issue body
that `--backfill` wrote. Nothing is lost by discarding it.

There is a second, smaller cost with the same cause, and it recurs until the
window closes. A row an in-flight branch appended carries links written relative
to `.agents/`, and the archive now sits one directory deeper, so git merges the
row cleanly and `check-agent-record` then reports a dangling link. It happened on
both merges of `origin/main` taken during this row's implementation. Repoint the
appended row's links, or drop the hunk; the same one-line loop that moved the
original 523 links resolves them by target.

What the change does remove is the *recurrence*. Two fresh branches that each
file an issue no longer share an append target at all, which
`tests/scripts/test_agent_issue_index.py::FilingAnIssueNoLongerCollides` pins
with its own control: the old shape collides, the new shape does not, both with
the union driver disabled because that is what GitHub does.

## Work breakdown

One pull request, developer-selected 2026-08-29. The spec commit precedes the
implementation commits, which is what proves the order.

| ID | Work | Issues |
|---|---|---|
| W6 | Derive the index; retire the file, the driver and the append-only gate; rewrite §Records; scope the force-push sentence | #2290 #883 #364 #1745 #1002 #1619 #1644 #2266 #1808 |
| W7 | The trailer walk gets its caller's merge-commit rule | #2157 |
| W9 | Attribution enforced ONCE: the push-lane walk goes, and the exception registry with it | #858 #406 #581 #1058 #1260 #1262 |
| W8 | Preflight runs every checker; add the two exit rules to §intake | #467 #2298 |

## Risks / decisions

- **The snapshot can be stale.** Accepted, and made visible: absence and age both
  report SKIP with the reason and the refresh command. `now.py` has run on this
  contract since #374 without a staleness incident. A stale snapshot is strictly
  better than a file whose freshness is purchased by making every PR write it.
- **Dropping the `main` trailer walk removes a gate.** Accepted. The walk never
  prevented a bad landing — it reported one afterwards, on bytes GitHub composed,
  and #1260 and #1262 are both landed commits it could not repair. The PR-body
  gate acts before the bytes are frozen, which is the only point where the
  outcome can still change.
- **Diff-scoped ownership checks fewer issues than the ratchet counted.**
  Accepted, and it checks the right ones. The ratchet's global count could only
  ever be paid down by work no single commit owned, which is why it sat at 33.
- **This change edits the surfaces it is fixing.** Unavoidable and acknowledged,
  as #364 acknowledged before it.
- **854 rows of index prose.** Moved to `.agents/completed/`, not deleted, per
  §Records. Some rows carry detail that is not in the GitHub issue body.

## Stop conditions

Return `NEEDS_DECISION` rather than widening scope if: the `Row:` line cannot be
parsed reliably out of existing issue bodies, because that would mean the
migration needs a backfill pass this spec does not budget; or if removing the
`main` trailer walk turns out to drop an obligation the PR-body gate does not
carry, which would mean the two were not duplicates and §5's premise is wrong; or
if running all 42 checkers in preflight exceeds a workable runtime, which would
make the coverage fix a different design rather than a list change.

## Now

`ACTIVE`. Spec committed 2026-08-29; W6-W8 implement in this branch. The row's
earlier waves (W1-W5 in
[retire-shared-record-surfaces.md](retire-shared-record-surfaces.md)) are
overtaken: `STATUS_RATCHET` and `check-public-doc-tables.py` were both removed by
#1714, and `NOW.md` was derived by `ENG-NOW-DERIVED` (#374).

## Owed

Nothing yet.

## Outcome

Pending; written when the row reaches `DONE`.
