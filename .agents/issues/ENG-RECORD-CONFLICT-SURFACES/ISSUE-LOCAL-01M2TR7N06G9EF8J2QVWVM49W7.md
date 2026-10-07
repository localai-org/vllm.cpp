ID: ISSUE-LOCAL-01M2TR7N06G9EF8J2QVWVM49W7
Title: The local issue migration stops at #2448: 501 reachable forge issues (261 open) have no canonical record, and the spec forbids the gap
Row: ENG-RECORD-CONFLICT-SURFACES
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-18
Updated: 2026-09-18
Closed: -

## Problem

The local issue migration (`a9f6186c2`, `feat(ENG-RECORD-CONFLICT-SURFACES): cut over local issue authority`, 2026-09-07) captured 1053 `ISSUE-GH-` numbers. It reaches #3188, so this is not a watermark. It is a contiguous hole that starts at #2449 and has never been closed.

## The measured distribution

Captured numbers, on this worktree at `origin/main` `a05f6098f`:

```sh
find .agents/issues -name 'ISSUE-GH-*.md' \
  | sed -E 's/.*ISSUE-GH-([0-9]+)\.md/\1/' | sort -n > /tmp/gh_nums.txt
wc -l < /tmp/gh_nums.txt                                    # 1053
awk '$1<2500'  /tmp/gh_nums.txt | wc -l                     # 1023
awk '$1>=2500 && $1<3000' /tmp/gh_nums.txt                  # 2773 2987 2988
awk '$1>=3000' /tmp/gh_nums.txt | wc -l                     # 27
tail -1 /tmp/gh_nums.txt                                    # 3188
```

There are no duplicates, and 85 `ISSUE-LOCAL-` records bring the tracked total to 1138 files.

Per century the captured count runs 32-60 from #400 to #2499, then collapses:

| century | captured | forge issues | forge OPEN uncovered |
|---|---|---|---|
| 2300-2399 | 47 | 37 | 4 |
| 2400-2499 | 32 | 65 | 8 |
| 2500-2599 | 0 | 65 | 22 |
| 2600-2699 | 0 | 81 | 61 |
| 2700-2799 | 1 | 75 | 35 |
| 2800-2899 | 0 | 61 | 22 |
| 2900-2999 | 2 | 63 | 34 |
| 3000-3099 | 9 | 77 | 49 |
| 3100-3199 | 18 | 43 | 15 |

The captured counts above #2499 are new records that agents created after the cutover, not migrated ones. Below #2449 the migration covers every reachable forge issue except four.

## What is on the forge

Enumerated with an explicit page size, not a default:

```sh
gh issue list --state all --limit 5000 --json number,state,title,createdAt,closedAt
```

That returns 698 issues, which `gh api graphql` confirms is the complete set: `issues.totalCount` 698, OPEN 353, CLOSED 345, `pullRequests.totalCount` 460. The number space reaches #3216, so roughly 2058 numbers are deleted, transferred or otherwise unreadable; that is a separate known condition and not this finding.

A second instrument agrees exactly. `gh api --paginate '/repos/mudler/vllm.cpp/issues?state=all&per_page=100'` with the pull requests filtered out returns the same 698 issues, the same 345 in #2500-#2999, and the same 177 open among them, so the GraphQL and REST reads are not one measurement counted twice.

In #2500-#2999 the forge holds **345 issues: 177 OPEN, 168 CLOSED**. The migration captured three of them.

Over the whole hole #2449-#3216 the forge holds **501 issues: 261 OPEN, 240 CLOSED**.

## How much of the hole is invisible

A number cited in-tree is visible even without a canonical file. The citation set is every `ISSUE-GH-<n>`, `#<n>` and `issues/<n>` occurrence under `.agents/`, `docs/`, `AGENTS.md` and `README.md`:

```sh
grep -rhoE '(ISSUE-GH-|#|issues/)[0-9]{1,6}' .agents docs AGENTS.md README.md \
  | grep -oE '[0-9]+$' | sort -u -n > /tmp/cited.txt
```

Positive control: `#2390` and `#2298`, both cited in `AGENTS.md` and in specs, are present. Negative control: `999999` is absent. Two sampled hits were opened and confirmed to be real spec references (`.agents/specs/gdn-chunked-mirror.md:1061` cites #2999; `.agents/specs/upstream-pin-advance-e126687.md:473` cites #2971), so this is not a substring artifact.

| range | OPEN with no canonical record | of those, cited in-tree | **cited NOWHERE** |
|---|---|---|---|
| #2500-#2999 | 174 | 147 | **27** |
| #2449-#3216 | 247 | 169 | **78** |

The 27 in #2500-#2999 are #2716, #2775, #2776, #2803, #2806, #2821, #2824, #2826, #2871, #2872, #2885, #2886, #2887, #2888, #2918, #2924, #2934, #2936, #2937, #2943, #2961, #2976, #2978, #2983, #2986, #2992, #2995. They include an `agent-ready.py` deadlock report, a doctest SKIP-counted-as-PASS report, an unregistered op gate, and an unbounded `resize` on a user-supplied `--mmproj`. #2987 and #2988 were in this set until `df1a4552f` re-anchored them.

`#<n>` citations also match pull request numbers, so 147 and 169 are upper bounds on visibility and 27 and 78 are lower bounds on the debt.

## The hole is NOT explained by the migration tooling

`.agents/specs/roadmap-consistency.md:311-315` states the migration input set exactly:

> Migration operates on the union of two sets:
> 1. Every GitHub number in the frozen `.agents/completed/issue-index.md`.
> 2. Every currently reachable GitHub issue, open or closed, returned by a complete paginated enumeration.

There is no range, cutoff, date filter or exclusion list in that spec, and `grep -inE 'cutoff|--since|--range|--max-number|exclude'` over `scripts/issue_records.py` and `scripts/agent-issue.py` finds only an unrelated path-exclusion argument. Every issue in the hole was and is reachable, so set 2 required all 501 of them. The hole is an accident, not a decision.

The mechanism is staleness. The frozen archive `.agents/completed/issue-index.md` stops at #2353, so everything from #2354 upward could only enter through set 2. The highest captured reachable number is #2448, created 2026-08-31T23:39:19Z. #2449, created 2026-08-31T23:55:42Z, is reachable and uncaptured. The enumeration therefore ran inside that sixteen-minute window on 2026-08-31, and `a9f6186c2` landed 2026-09-07T13:35:30Z. Six and a half days of issue filing fell between the snapshot and the merge, and nothing re-ran the enumeration before the cutover.

A second, smaller breach of the same rule: #2350, #2354, #2356 and #2357, all created 2026-08-30 and all OPEN today, are reachable and uncaptured although they predate the snapshot. The enumeration was not complete even within its own window.

## Why nothing noticed

The restructured checker's canonical-reference pass: when a branch reference fails to resolve to a canonical record, the pass appends an error only if the reference is a stable-ID form. A bare `#<n>` hits `continue` and is silently accepted. (The pass is
`check_canonical_issue_references` in that restructured file; the pre-restructure
`scripts/check-agent-record.py` that `main` carries has no canonical-reference pass at
all.) The 169 in-tree `#<n>` citations into the hole therefore cost nothing today. The hazard is latent rather than live: every `ISSUE-GH-<n>` form citation into #2449-#3216 currently in the tree (#2987, #3094, #3103, #3116, #3153, #3155) does resolve, so no gate is red, but the first spec that writes `ISSUE-GH-` for a hole number reds the record gate on its own diff.

## No other gap of this size

Between #400 and #2448 no century has a captured count below 32, and every reachable forge issue below #2449 has a canonical record except the four named above. The thin coverage from #3000 upward is not a second hole; it is this one refilling organically as new work files canonical records under the current rules.

## What closing it would cost

- **261 open issues** in #2449-#3216 need one `import-github` each. That is 261 GitHub reads. It is read-shaped, but the 369-write suspension precedent means any batch must be throttled and must not mirror.
- **Ownership is the expensive part.** `roadmap-consistency.md:317-322` resolves ownership by the archived row first, and the archive stops at #2353, so no issue in the hole has one. Every record falls to the spec `## Owed` rule, which needs exactly one owning spec. 169 have at least one in-tree citation to seed that; **78 have none** and need human triage or an `_owed` placement that some spec must then anchor.
- **240 closed issues** cannot be batch-imported at all. `roadmap-consistency.md:332` requires non-placeholder resolution evidence from the body, a binding comment, the tree or an owning spec, and says migration stops rather than inventing it. Each closed record is a judgement.
- **`_intake` is not an escape.** `roadmap-consistency.md:303` fails any diff-scoped citation that resolves to `_intake`, so parking the hole there converts 169 currently-free `#<n>` citations into gate failures.

Whether ~500 numbers become canonical records is a decision for the developer and the owner of this migration, not a side effect of measuring it. Nothing in this issue was imported, labelled or edited on the forge.

## Resolution

-
