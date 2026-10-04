# The diff-scoped gates get a floor, so an unrepairable commit cannot freeze them

Issue: [#1809](https://github.com/mudler/vllm.cpp/issues/1809)
Row: `GATE-CI-ENFORCEMENT-FLOOR`

The main-branch diff-scoped gates walk from the head of the last SUCCESSFUL push
run. That base is what makes a cancelled run lossless (#822, #863). It is also
what turns one unrepairable commit into a permanent red: no green run means the
base never advances, so every later push re-walks the same violations and adds
its own commit to the range.

This row keeps the self-healing base and clamps it from below with a recorded
**enforcement floor** — one commit the walk never goes behind. It also moves the
base selection out of four copies of inline workflow shell and into one script
that has a test suite.

## Why

### Measured, 2026-08-23, at `bacb71109c8d63b5f862c9b121dd86e04e1a07ee`

`gh api repos/mudler/vllm.cpp/actions/workflows/ci.yml/runs?branch=main&event=push&status=success&per_page=1`
returns `fafa16f0f32acc8255e113a2cbc35f8b99cf2072`, whose commit date is
2026-08-13T01:53:21+02:00. That is `LAST_GREEN`, and it is ten days stale.

| Quantity | Value |
|---|---|
| First-parent commits in `fafa16f0f..bacb71109` | 499 |
| Of those, merge commits | 0 |
| `commit-protocol-tag` grep step: commits with no `FOLLOWING_AGENTS_PROTOCOL` | 20 |
| `commit-protocol-tag` strict step: commits failing `check-commit-trailers.py --range` | 35 |
| `documentation-checkpoint`: commits failing `check-role-discipline.py` | 6 |
| `documentation-checkpoint`: `check-now-current.py` | passes |

The range is linear — `rev-list` and `rev-list --first-parent` both return the
same 499 commits — so the two trailer walks cover the same population and the
grep step's 20 are a strict subset of the strict step's 35. The difference is the
contract, not the walk: the grep asks whether the marker appears at all, and the
other 15 commits carry it (once to nine times, measured) in a form
`check-commit-trailers.py` rejects. Presence is not parseability. The 6
role-discipline commits are disjoint from the 35. **41 distinct commits.**

That count is the measurement at `bacb71109` and is left as measured. A 42nd
commit, `c00b99c7c`, landed on 2026-08-24 while this row was in flight and moved
the split to 35 trailers / 7 role discipline. A 43rd, `e1b5df1a6`, landed on
2026-08-25 and moved it to 35 / 8. A 44th, `96c5e4719`, landed on
2026-09-09 and moved it to 35 / 9. `## The 44 forgiven commits`,
`### The first advance`, `### The second advance` and
`### The third advance` carry them.

### Why no remedy exists

Every one of the 41 is on `main`. A trailer or a task-branch arrival can only be
added to a commit by rewriting it, and `AGENTS.md` forbids rewriting `main`
without exception. So the gate asks for a repair that cannot be performed, and
the loop closes:

```
main is red -> no successful push run -> LAST_GREEN frozen
            -> next push walks a range one commit wider
            -> re-hits the same 41 -> still red
```

The practical effect is #1722's effect: a job that is always red trains every
reader to skip it, and a skipped job protects nothing. Violations kept arriving
during the freeze and nobody read the gate that named them —
`1757330006f6` landed without the trailer on 2026-08-23, and `6e73bdee3` landed
without a task branch the same day.

### The property that must survive

`ci.yml` records the reason for the `LAST_GREEN` base at its definition:
`github.event.before` is the previous push's sha whether or not that push was
gated, so a cancelled run's commits are skipped and **nothing re-covers them**.
That is what allows the push lane to be latest-only (#822), and reverting the
base to `before` reintroduces exactly the gap #863 measured. Cancelled runs are
common here (#1285). Losslessness is therefore a requirement on the fix, not a
nice-to-have.

## Design

### The floor

One commit sha, recorded in `scripts/ci-enforcement-floor.txt`. The walk never
starts behind it.

```
base = last_green or before            # unchanged
if floor is newer than base:           # newer == base is a proper ancestor of floor
    base = floor
```

"Newer" is decided by ancestry, not by date: `git merge-base --is-ancestor`.
A commit date is author-controlled and can go backwards across a rebase, so a
date comparison can choose the wrong commit; ancestry on a linear first-parent
`main` cannot.

The floor is set to `96c5e4719dcb1f859cb9e59573309f16c026b523`, which is the
last of the 44 commits enumerated below. The floor is the forgiven commit
itself rather than its child, because the walk is `FLOOR..HEAD` and excludes
`FLOOR`; every commit after it stays enforced.

It was first set to `bacb71109c8d63b5f862c9b121dd86e04e1a07ee`, which was past
the 41 commits then known. `c00b99c7c` landed while this row was in flight and
the value was advanced once, before merge, under `### Advancing the floor`.
`e1b5df1a6` landed on 2026-08-25 and the value was advanced a second time, on
2026-09-03, under the same section. See `### The first advance` and
`### The second advance` below.

### The base selection moves into a script

`scripts/ci-walk-base.py` resolves the base for every diff-scoped gate.
`.github/workflows/ci.yml` had four byte-similar copies of the selection —
`agent-record`'s role-discipline step, `documentation-checkpoint`,
and both steps of `commit-protocol-tag`. Four copies of a rule is four places to
get the floor wrong, and the rule was very nearly untested. Measured rather than
assumed: `tests/scripts/test_main_baseline.py::AgentRecordDiffRangeTests` replays
exactly ONE of the four bodies — `agent-record`'s — and it does so under a shim
that stubs every `python3` call, so what it pins is which checker gets invoked
with which range string. It cannot see the base rule itself, and the other three
bodies were executed by no test at all. Nothing in the tree could have caught the
ratchet.

The extraction gives the rule a test surface, and that is what makes requirement
2 — a cancelled run stays lossless — an executable assertion instead of a claim.
The `AgentRecordDiffRangeTests` shim now executes the resolver for real rather
than stubbing it, so those cases test the real composition of the resolver and
the step shell.

Contract:

```
scripts/ci-walk-base.py --event <name> --head <sha>
    [--pr-base <sha>] [--push-base <sha>] [--last-green <sha>]
    [--floor-file <path>] [--floor <sha>]
```

It prints the resolved base on stdout, diagnostics on stderr, and exits non-zero
only on an unusable floor record or an invalid argument.

Resolution order, in one place:

1. `pull_request` lane: return `pull_request.base.sha` unchanged. The floor does
   not apply — see "Why the floor is push-lane only".
2. Otherwise `base = LAST_GREEN`, falling back to `github.event.before`.
3. If `base` is empty or unknown to git — the all-zero sha of a new branch, or a
   force-push whose `before` is gone — return it unchanged, so the existing
   downstream guard still degrades to the tip commit alone. The floor raises a
   USABLE base; it never substitutes for an unusable one.
4. If the floor is unknown to git, or is not an ancestor of `head`, warn on
   stderr and return `base` unchanged. A floor that the current history does not
   contain cannot bound that history, and `floor..head` for an unrelated floor
   is not a range anybody asked for.
5. If the floor is an ancestor of `base`, `base` is already at or past the floor:
   return `base`.
6. Otherwise return the floor.

### Why the floor is push-lane only

The pull-request lane bases on `pull_request.base.sha` and has been green
throughout the freeze — verified on #1786 on 2026-08-23. Applying the floor
there would only ever raise a base, which is a narrowing of what that lane
enforces, and no defect asks for it. The narrowest change that fixes the bug
leaves the PR lane byte-identical.

### Cancelled runs stay lossless

The floor is a lower clamp on a base that is otherwise chosen exactly as it is
today. While the floor is behind `LAST_GREEN`, step 5 returns `LAST_GREEN` and
the resolved base is byte-identical to today's. A cancelled run does not advance
`LAST_GREEN`, the next run walks the wider range, and the cancelled run's
commits are covered.

**That is not the regime this repository is in, and the spec should not pretend
otherwise.** `LAST_GREEN` advances only on a green push run, and measured on
2026-08-23 it is still `fafa16f0f`, the run of 2026-08-12T23:53:24Z — the same
instant `### Measured` records above as `2026-08-13T01:53:21+02:00`, in the
commit's own zone. Of the 100 most recent `push` runs of `ci.yml` on `main`,
every one of them after that date, 93 are `cancelled`, 6 are `failure` and one
is still running; none is a success. `.agents/verification.md` records the same
shape from the other side: of 40 consecutive runs measured for #274, 26 were
`cancelled` and exactly one completed. So the floor sits **ahead** of
`LAST_GREEN`, not behind it, the resolved base is the floor on every push, and
the walk grows by one commit per merge — the same unbounded growth this file
levels at the rejected exemption list in `### The alternative that was
rejected`. It stays that way until a `push` run on `main` finishes green, which
nothing in this row brings closer.

Correctness is unaffected: `floor..HEAD` still covers every commit that landed
after the floor, which is every commit any contributor can still do anything
about. What is affected is the claim, so the claim is corrected rather than
repeated.

The one window where losslessness is suspended is the interval
`LAST_GREEN..floor`, which opens at a floor advance and closes at the next green
push run — and on the evidence above that can be a long time, not a moment. What
the window skips does **not** grow while it is open: it is fixed at the advance
by the two recorded shas, and every commit after the floor is still walked. That
window is exactly the forgiveness being asked for, it is bounded by a recorded
sha, and what it forgives is enumerated below. It is not silent.

`tests/scripts/test_ci_walk_base.py::CancelledRunLosslessTests` builds a real
throwaway repository and replays the sequence: C1 gated green, C2 pushed and its
run cancelled, C3 pushed. It asserts the resolved base is C1 and that
`rev-list base..C3` CONTAINS C2 — and, as the positive control that proves the
assertion discriminates, that the naive `github.event.before` base for the same
push resolves to C2 and its range does NOT contain C2.

### What is narrowed, and the argument for it

This change narrows enforcement: 43 commits that the gate currently reports are
no longer walked. The argument is that enforcing on an immutable already-landed
commit is not enforcement. There is no action any contributor can take that
turns those 43 reds green, because the only action that would is a `main`
rewrite the protocol forbids. A gate with no available remedy is a permanent
red, and a permanent red is read by nobody — which is a strictly worse outcome
than a smaller gate that is read.

The gate's purpose is to stop a NEW violation, and that is untouched: a commit
landing after the floor with no trailer, or with no task branch, still reds the
job. Proved by mutation, not by reading the diff — see `## Gates`.

No assertion is deleted. `check-commit-trailers.py`, `check-role-discipline.py`
and `check-now-current.py` are not modified by this row, and the grep step's
condition is unchanged. Only the base of the walk moves.

**The floor forgives by RANGE, not by violation, and that is a real cost.** The
43 commits are what the three checkers report *today*. A checker written
tomorrow that finds a new class of defect in the pre-floor range will be
forgiven for that range too, silently, without anybody deciding to forgive it
and without a line appearing anywhere. **This is no longer hypothetical.**
#2745, found while measuring the second advance, is a cutover constant in
`check-role-discipline.py` that names an unlanded sha, so the integration-path
rule it was meant to switch on has never run. Repairing it would make the
checker report a class of defect across the whole pre-floor range, and this
floor would forgive every instance of it without a line appearing anywhere.
The cost is real, it is paid in advance, and #2745 is where it is recorded. An exemption list would not have that
property: it names shas and one error each, so a new checker's finding on an
old commit would still red. This is the strongest argument against the shape
chosen here, and `### The alternative that was rejected` is not an honest
comparison without it. It is accepted because the alternative's three costs are
judged worse and because the range is bounded by a sha a reviewer can read,
not because this cost is small.

**The floor also absorbs the one in-checker exception, and nobody chose that
either.** `check-commit-trailers.py` carries a single annotated
`LANDED_MESSAGE_EXCEPTIONS` entry for `281b4bc76c0e` (#1262), and every run that
applies it prints `1 landed-message exception(s) applied. This is DEBT, not
success`. That commit is dated 2026-08-18 and is an ancestor of the floor, so
after this lands the main push lane never walks it. Measured:
`check-commit-trailers.py --range fafa16f0f..origin/main` prints that banner
once, and `--range bacb71109..origin/main` prints it zero times. The commit is
therefore forgiven twice, and its DEBT line — written precisely so a reader of
a green lane can see what the lane is carrying — stops reaching that reader.
Recording it here was the replacement, and it was a weaker one than the banner.

**SUPERSEDED 2026-08-29 by [derived-issue-index.md](derived-issue-index.md) W9
(#2322).** Both trailer steps of `commit-protocol-tag` now run on the
PULL-REQUEST lane only, so the push lane walks no commit that is already on
`main` and cannot need forgiving. `LANDED_MESSAGE_EXCEPTIONS` and its test class
are removed with the reason they existed for. The double-forgiveness described
above is moot: the entry is gone, the banner is gone, and the paragraph is kept
because deleting it would erase the record that `281b4bc76c0e` violated the
contract at all. THIS FILE remains the only place that says so.

The floor itself is untouched and still governs `documentation-checkpoint` and
`agent-record`'s role-discipline step, which do walk the push lane. That change
narrowed the floor's SCOPE and did not move its VALUE, which stayed at
`c00b99c7c` and forgave the same 42 commits for those two gates.
`### The second advance` later moved the value to `e1b5df1a6` and left the
narrowed scope exactly as #2322 set it; the count for those two gates is now 43.

### Advancing the floor

Editing one line of `scripts/ci-enforcement-floor.txt`, in a reviewed pull
request whose body says which commits the advance forgives and why each is
unrepairable. Git is the history of the floor: `git log -p` on that file lists
every advance with its reason. There is no registry and no accumulating list.

This does not make an unrepairable commit free. It makes it cost a reviewed
commit that has to name it, which is what `AGENTS.md` means by visible debt.

### Where the floor lives, and the record-lock rule

`AGENTS.md` `## Records` forbids a surface that every pull request must write.
The floor is not one: an ordinary pull request never touches it, and only a
deliberate advance does. A one-value data file beside the script that reads it
matches `scripts/*-allowlist.txt`, which are the tree's existing shape for a
script's data.

Rejected homes:

- **A top-level `env:` in `ci.yml`.** No new file, but `ci.yml` has no top-level
  `env:` block today and seven checkers and test suites parse that file. A
  structural addition risks a red that has nothing to do with this row.
- **A git ref or tag.** Movable without review and invisible in a diff, which
  removes the whole reason for choosing a floor over a time window.
- **A new `.agents/` document.** `check-pr-size.py::classify_path` fails closed
  on an unclassified path, so the file would require an edit to
  `check-pr-size.py`, which is itself a governance-checker change requiring its
  own mutation evidence. A cascade in exchange for nothing.

### The alternative that was rejected

**A per-commit exemption list**: keep walking `LAST_GREEN..HEAD` forever and
name the 43 shas in a file the checkers consult. It records more precisely than a
floor does, and `check-commit-trailers.py` already carries one landed-message
exception, so the mechanism is not foreign.

Rejected on three grounds:

1. `AGENTS.md` `## Changing the rules or a checker` states the project has no
   waiver registry, because an exception registry is a state log and this
   protocol has no state log. One in-checker exception carrying its reason is
   not a registry; a file of 43 growing to N is precisely one.
2. It needs the mechanism built three more times. The grep step and
   `check-role-discipline.py` have no exemption concept, so the change would add
   an exemption surface to code that currently has none — more new enforcement
   machinery than the fix it delivers.
3. It never shrinks the walk. The range stays 499 commits and grows by one per
   merge forever, so the cost and the log noise of every run grow without bound,
   and the next unrepairable commit appends to the list rather than being
   confronted. The floor bounds the walk and makes forgiveness cost a review.

Also considered and rejected: reverting the base to `github.event.before`
(reintroduces #863 outright); making the two jobs report-only (`AGENTS.md`
`## Gates`: a permanent report-only state is not a result); and a rolling
time-window base such as "the newer of `LAST_GREEN` and 7 days ago", which
forgives continuously and silently and records nothing.

## Scope

In scope:

- `scripts/ci-walk-base.py`, new.
- `scripts/ci-enforcement-floor.txt`, new: the recorded floor.
- `tests/scripts/test_ci_walk_base.py`, new.
- `.github/workflows/ci.yml`: the four base-selection blocks call the script.
- `scripts/agent-preflight.sh`: the new suite joins `SUITES`.

Out of scope, deliberately:

- Repairing the 43 commits. It cannot be done without rewriting `main`.
- The three checkers themselves. Not one line changes.
- `agent-record`'s missing-`hugo` red (#1722, fix in flight as #1726) and the two
  `windows-msvc` reds (#584). Both are inherited and neither is this row's.
- The PR lane's base selection, which is unchanged.

## The 44 forgiven commits

Real protocol violations that landed unread between 2026-08-13 and 2026-09-09,
recorded here because after the floor advances no gate will name them again.

35 fail the strict trailer contract and 9 fail role discipline. The two sets are
disjoint. 41 of them were enumerated when the floor was first recorded at
`bacb71109`; the 42nd, `c00b99c7c`, arrived afterwards and moved the split from
35/6 to 35/7 — see `### The first advance`. The 43rd, `e1b5df1a6`, arrived on
2026-08-25 and moved it to 35/8 — see `### The second advance`. The 44th,
`96c5e4719`, arrived on 2026-09-09 and moved it to 35/9 — see
`### The third advance`.

### Fail `check-commit-trailers.py --range` (35)

| Commit | Date | Subject |
|---|---|---|
| `7572b0f4e2fb` | 2026-08-13 | guard the parity pin header against declaration-order breaks (#558) |
| `7ba9a675f491` | 2026-08-13 | feat(rocm): implement the vt::Backend graph-capture seam on hipGraph (W1, #332) (#473) |
| `7965f12bf4bc` | 2026-08-13 | fix(GATE-PR-SIZE-BINARY): retire the fail-closed binary guard (#615) (#619) |
| `a3aa02e197ec` | 2026-08-14 | spec(MODEL-MUSIC-MUSIC3): scope MiniMax-Music3 (#672) (#679) |
| `34dc578760d0` | 2026-08-14 | oracle(MODEL-MUSIC-MUSIC3): the diffusers oracle GENERATES AUDIO (#672) (#708) |
| `8d0c2779b91a` | 2026-08-14 | feat(MODEL-MUSIC-MUSIC3): W1 — the modular checkpoint loader (#672) (#714) |
| `373aa125142a` | 2026-08-14 | spec(BACKEND-ROCM): the ROCm head_dim=128 decode arm (#564) |
| `f8cbc2310bca` | 2026-08-14 | fix(#664): the video registry's existence probes stop reaching Windows with POSIX stat |
| `0011bedf0c75` | 2026-08-14 | fix(#720): M_PI is not defined by MSVC |
| `fc903b8dd73f` | 2026-08-14 | fix(#674): the LTX-2.5 VAE loader read the safetensors mmap through a uint16_t* |
| `9f2b9bb9a30b` | 2026-08-14 | feat(tenstorrent): allowlist MistralForCausalLM + device-aware gate (#431) |
| `d8efb1fa0ccf` | 2026-08-14 | build(nix): add rocwmma to the ROCm dev shell (#444) (#638) |
| `3921160e569d` | 2026-08-14 | fix(#757): six C4456 shadowed locals block the Windows test compile |
| `c629b5d0ff78` | 2026-08-14 | feat(ltx-2.5): image conditioning at crf=0 (#644) |
| `5da1d7f2fa89` | 2026-08-14 | fix(GATE-FORK-ANCESTRY): diff a PR from its merge base (#773) (#782) |
| `ddff090936bb` | 2026-08-15 | policy(POLICY-SINGLE-PR-AND-STYLE): one PR carries the spec and its code (#827) |
| `be4a3edf17b2` | 2026-08-15 | fix(GATE-WINDOWS-WARNING-POLICY): /WX- is not /WX (#774) (#795) |
| `6680aab68912` | 2026-08-15 | fix(GATE-AUDIT-BRANCH-EVIDENCE): reach the IN-FLIGHT verdict in CI (#726) (#802) |
| `ca01719e6b29` | 2026-08-15 | fix(#772): four loaders cast mmap'd safetensors to uint16_t* (#815) |
| `3ce5a1dc1b0f` | 2026-08-15 | feat(MUSIC3-W7): a gated GGUF Q4_K arm (#672) (#832) |
| `51e0cb5b15fe` | 2026-08-15 | policy(POLICY-ISSUE-INTAKE): the issue index moves out of the roadmap (#846) |
| `b5a5f3b182d7` | 2026-08-15 | feat(MODEL-MUSIC-MUSIC3): W2's remainder (#672) (#831) |
| `6e6bba63d7c1` | 2026-08-15 | fix(GATE-OP-PARITY-MANIFEST): refuse a throwing golden by name (#776) (#853) |
| `bc570da0d387` | 2026-08-15 | MODEL-NEMOTRON-H: the WEIGHT LOADER (#752) |
| `34962d96bea0` | 2026-08-15 | fix(#775): the NemotronH forward's downcast was a promise, not a check (#868) |
| `b3d0f3ed5dc8` | 2026-08-15 | fix(capi): hoist SpeechRegistry() out of extern "C" (#805) (#814) |
| `1e2408526419` | 2026-08-15 | docs(dspark): the user-facing docs asserted a ratio measurement has refuted (#442) (#894) |
| `04be1390b227` | 2026-08-15 | fix(FIX-REGISTRY-DOWNCAST-SWEEP): open every registry handle with a check (#901) |
| `b5f27c9a4c7d` | 2026-08-15 | record(intake): place #904, the third sanitize-cpu red (#906) |
| `2688e6586675` | 2026-08-15 | fix(MODEL-MUSIC-MUSIC3): the e2e gate asked for 60 s of music (#852, #925) (#942) |
| `e34d71379e70` | 2026-08-16 | fix(qwen3.5): drop redundant AppleClang capture (#1054) |
| `aba8d5ffb77c` | 2026-08-18 | instrument(MUSIC3): a per-stage split (#672) (#1231) |
| `055ff1143704` | 2026-08-21 | docs: align README with current surfaces (#1302) |
| `2d2a66715ef4` | 2026-08-22 | fix(#817): CAMPPlus trusted a default over the weight (#1739) |
| `1757330006f6` | 2026-08-23 | fix(BACKEND-TENSTORRENT-GDN): W2 review repairs (#1715) |

The 20 that also fail the grep step are the subset of the above whose message
carries no `FOLLOWING_AGENTS_PROTOCOL` string at all: `7572b0f4e2fb`,
`7ba9a675f491`, `7965f12bf4bc`, `373aa125142a`, `9f2b9bb9a30b`, `be4a3edf17b2`,
`6680aab68912`, `ca01719e6b29`, `b5a5f3b182d7`, `6e6bba63d7c1`, `bc570da0d387`,
`34962d96bea0`, `b3d0f3ed5dc8`, `04be1390b227`, `b5f27c9a4c7d`, `2688e6586675`,
`e34d71379e70`, `aba8d5ffb77c`, `2d2a66715ef4`, `1757330006f6`. The remaining 15
carry the marker in a form the strict contract rejects.

### Fail `check-role-discipline.py` (9)

Repository changes that reached `main` without arriving on a task branch.

| Commit | Date | Subject |
|---|---|---|
| `dd8a3b0e184c` | 2026-08-17 | windows: fix native MSVC/Vulkan build portability |
| `8daf58e7752f` | 2026-08-18 | fix(ENG-RELEASE-WINDOWS): the api-server gate can report its own failure again |
| `38ec0da4aae8` | 2026-08-18 | feat(BACKEND-ROCM): register a ROCm attention backend for kROCM |
| `5073df62228e` | 2026-08-18 | feat(BACKEND-ROCM): select the attention backend in the runner |
| `65d6cdaed3e2` | 2026-08-18 | build: make the tree compile on gcc 16, and add a CI lane so it stays that way |
| `6e73bdee3ea1` | 2026-08-23 | fix(LTX25-POSITION-CONTRACT): gate the tower positions as integers |
| `c00b99c7c8b6` | 2026-08-24 | fix(LTX25-DIT-ATTN-ARM-PARSE): match every DiT attention arm exactly and refuse a fourth value |
| `e1b5df1a6b5b` | 2026-08-25 | fix(LTX25-DIT-ATTN-FA2-HD128): pin the distilled NVFP4 DiT to the artefact every measurement loaded, and hash it |
| `96c5e4719dcb` | 2026-09-09 | fix(BACKEND-TENSTORRENT-KEEPQUANT): pin the int8-dot sweep for every registered encoding |

`c00b99c7c8b6` is the 42nd. It landed
`src/vllm/model_executor/models/ltx2_device.cpp` and
`tests/vllm/models/test_ltx2_device.cpp` — product code and its test — straight
onto `main` with no `row/<ID>` branch in its history, which is what
`check-role-discipline.py` names. The checker writes one unwrapped line, and
this is that line verbatim, copied from
`python3 scripts/check-role-discipline.py --base b207f34d3 --head c00b99c7c`,
which exits 1:

```
ERROR: c00b99c7c: repository change (src/vllm/model_executor/models/ltx2_device.cpp, tests/vllm/models/test_ltx2_device.cpp) reached main without arriving on a task branch. Work happens in its own worktree on a `row/<ID>` branch and lands through a reviewed PR or an authorized local merge naming that branch; never directly on the shared checkout
```

Its message is clean: it carries `FOLLOWING_AGENTS_PROTOCOL` and passes
`check-commit-trailers.py`. This is a **role-discipline** violation and not a
trailer one, so it is the seventh row of this table and not the thirty-sixth of
the one above, and the split across the 42 was 35 trailers / 7 role discipline
rather than 35 / 6.

`e1b5df1a6b5b` is the 43rd and the newest, authored by
`mudler's LocalAI [bot]` on 2026-08-25. It changed four paths —
`.agents/specs/ltx25-dit-attn-fa2-hd128.md`, `docs/USAGE.md`,
`docs/models/ltx-2-5.md` and `tests/vllm/multimodal/ltx2_video_fixture.h` —
and reached `main` with no `row/<ID>` branch and no `(#N)` in its message, so
`arrives_via_row_pr` finds nothing to accept. The checker writes one unwrapped
line, and this is that line verbatim, copied from
`python3 scripts/check-role-discipline.py --base c7ad92f87 --head e1b5df1a6`,
which exits 1:

```
ERROR: e1b5df1a6: repository change (tests/vllm/multimodal/ltx2_video_fixture.h) reached main without arriving on a task branch. Work happens in its own worktree on a `row/<ID>` branch and lands through a reviewed PR or an authorized local merge naming that branch; never directly on the shared checkout
```

**Only one of its four paths is named, and that is worth reading.** The other
three are integration paths, which `policy_commit_violations` still exempts
because `worktree_enforced` is `False` — `WORKTREE_DISCIPLINE_SINCE` names
`f236ca960`, which is not an ancestor of `main`. That is #2745, filed from this
advance and not repaired by it. The violation stands on the feature path alone,
so the finding does not depend on #2745 either way.

Its message is clean too: it carries `FOLLOWING_AGENTS_PROTOCOL` and passes
`check-commit-trailers.py`, so it is the eighth row of this table and not the
thirty-sixth of the one above, and the split across the 43 is 35 trailers /
8 role discipline.

`96c5e4719dcb` is the 44th and the newest, authored by Luca Barbato on
2026-09-09. It changed three paths — `.agents/backend-matrix.md`,
`src/vt/tenstorrent/tenstorrent_ops.cpp` and
`tests/vt/test_tenstorrent_backend.cpp` — and reached `main` with no
`row/<ID>` branch and no `(#N)` in its subject, so `arrives_via_row_pr` finds
nothing to accept. The checker names only the two feature paths, because
`.agents/backend-matrix.md` is a record path and is exempt. The checker
writes one unwrapped line, and this is that line verbatim, copied from
`python3 scripts/check-role-discipline.py --base e1b5df1a6 --head 96c5e4719`,
which exits 1:

```
ERROR: 96c5e4719: repository change (src/vt/tenstorrent/tenstorrent_ops.cpp, tests/vt/test_tenstorrent_backend.cpp) reached main without arriving on a task branch. Work happens in its own worktree on a `row/<ID>` branch and lands through a reviewed PR or an authorized local merge naming that branch; never directly on the shared checkout
```

Its message is clean too: it carries `FOLLOWING_AGENTS_PROTOCOL` and passes
`check-commit-trailers.py`, so it is the ninth row of this table and not the
thirty-sixth of the one above, and the split across the 44 is 35 trailers /
9 role discipline.

`check-now-current.py` passes over the whole range and forgives nothing.

**Every one of the 44 shas above resolves.** Verified with
`git rev-parse --verify -q '<sha>^{commit}'` over all 42, first at `origin/main`
`d60692c89` and again at `3574065e7` after the merge, and re-run over all 43 at
`origin/main` `8853af6bf` for the second advance, and re-run over all 44 at
`origin/main` `ec23d7b89e71` for the third advance, every time against the shas
parsed back out of this committed table rather than a hand-kept copy: 44
resolved, 0 missing, and all 44 are ancestors of `origin/main` by
`git merge-base --is-ancestor`. This is a full sweep and not a
spot-check, because after this lands the enumeration is the only witness that
these violations happened, and an earlier round of this list carried two shas
that resolved to nothing.

### The first advance

**This is the first exercise of `### Advancing the floor`, and it happened
before the pull request that introduces the mechanism had merged.** The row's
own risk 1 — "a violating commit lands between the recorded floor and the merge
of this row" — arrived on 2026-08-24, one day after the floor was recorded, and
was resolved by the procedure the row defines rather than by an exception to it.

Two things are worth reading off that, and they point in opposite directions.
The mechanism works: the deadlock the row exists to break re-formed at a scale
of one commit instead of 41, and a one-line reviewed edit cleared it. And the
violations are still arriving: `c00b99c7c` is the second role-discipline
violation in two days, after `6e73bdee3ea1` on 2026-08-23. The floor is a way
to stop an unrepairable commit freezing a gate. It is not a fix for whatever is
putting product code on `main` without a task branch.

**The advance is minimal by construction.** Measured at `origin/main`
`d60692c89`, ten first-parent commits sat above `bacb71109`, and exactly one of
them violated anything. The floor moved to that commit and no further. Setting
it to `origin/main` instead would have been one character of extra typing and
would have forgiven `e6f4f566f` and `d60692c89` unexamined, which is the abuse
risk 2 says nothing in this mechanism can detect. Both stay enforced.

### The second advance

**Advanced from `c00b99c7c` to `e1b5df1a6` on 2026-09-03, for #2743.** This is
the second exercise of `### Advancing the floor`, and unlike the first it
happened long after the mechanism merged, on the failure the mechanism was built
for: `agent-record`'s role-discipline step was red on `main` with no action any
contributor could take to clear it.

**Why it surfaced when it did.** It is not a regression from any recent merge.
`agent-record` runs under `set -e` and two earlier defects were failing ahead of
the role-discipline step, so it was never reached: a hardcoded benchmark-index
count in `tests/scripts/test_check_site.py` (#2643, fixed in `d9f394459`) and
the #2590 tier suites running before numpy was installed (#2705, fixed in
`975ade3de`). `main`'s previous run at `4d10c8acc` failed `agent-record` at
`AssertionError: 11 != 10`. This is the third defect in one stack, each hidden
behind the last, and reaching it is progress rather than breakage.

**The advance is minimal by construction, and this was measured rather than
assumed.** `c00b99c7c..e1b5df1a6` is **30 commits, 0 of them merges**. Each was
also checked on its own with `--commit`, which for a non-merge is exactly what
the range walk does, because `merged_pr_content` is empty when the range holds
no merge and so nothing can be exempted as reviewed content. **29 are clean and
exactly one violates**: `e1b5df1a6`. The floor moved to that commit and no
further. Setting it to `origin/main` `8853af6bf` instead would have been the
same edit and would have forgiven every commit above it unexamined, which is
the abuse risk 2 says nothing in this mechanism can detect.

That the whole walk names exactly one commit is measured too, and it is the
same number from both ends. Measured at `origin/main` `8853af6bf`, each rc
captured as `rc=$?` on the command itself and never after a pipe, because a
pipe reports its last stage and has misread a red as a green twice in this
row's history:

| Floor | `check-role-discipline.py --head origin/main` | `ERROR:` lines | `REPORT:` lines |
|---|---|---|---|
| `c00b99c7c` (old) | **rc 1** | 1, `e1b5df1a6` | 0 |
| `e1b5df1a6` (new) | rc 0, `OK: every change on main arrived on a task branch.` | 0 | 0 |

`git merge-base --is-ancestor e1b5df1a6b5b30555639e0a0459f79a467544579
origin/main` is rc 0, so the new floor is on this history and the resolver's
ancestry guard accepts it. `tests/scripts/test_ci_walk_base.py` re-run over the
new value: 31 tests, 0 failures, 0 errors, with
`RecordedFloorTests::test_recorded_floor_is_an_ancestor_of_head` reading the
committed file rather than a literal.

**One test pinned the old value and had to be re-pinned.**
`tests/scripts/test_check_commit_trailers.py`'s
`test_the_enforcement_floor_no_longer_claims_the_trailer_steps` asserts the file
ends with an exact sha, so that #2322 could prove it narrowed the floor's scope
without moving its value. It is 64/64 green at the base commit, reds with
exactly that one failure the moment the floor line changes, and is green again
at 64/64 once re-pinned — a red-before/green-after proof that the pin is live
and not decorative. The assertion keeps its exact-sha form and its strength;
only the sha and its message move. Deriving the expected value from this spec's
table instead would remove the re-pin, but that is a checker-semantics change
and it is deliberately not made here: an advance having to edit the pin is a
tripwire that puts the value in front of a reviewer, which is what
`### Advancing the floor` asks for.

**The arrival rate DID fall, and that is worth recording because the first
advance predicted the opposite.** Three role-discipline violations landed on
three consecutive days — `6e73bdee3ea1` on 2026-08-23, `c00b99c7c` on
2026-08-24, `e1b5df1a6` on 2026-08-25 — and `### The first advance` read that
run as "the violations are still arriving". Measured over
`e1b5df1a6..origin/main` at `8853af6bf`: **1365 commits, 297 of them merges,
spanning 2026-08-25 to 2026-09-03, and zero role-discipline violations.** Nine
days and 1365 commits is a far better denominator than the three-day run the
first advance generalised from, so the honest reading is that the three
consecutive days were a cluster and not a rate.

What the advance still does not do is fix anything. `### Advancing the floor`
costs one reviewed commit per unrepairable violation and buys nothing else; the
floor stops an unrepairable commit freezing a gate, and if a fourth violation
lands it will cost a third advance.

### The third advance

**Advanced from `e1b5df1a6` to `96c5e4719` on 2026-09-11, for #3135.** This is
the third exercise of `### Advancing the floor`, on the same failure the
mechanism was built for: `agent-record`'s role-discipline step was red on
`main` with no action any contributor could take to clear it.

**Why it surfaced when it did.** Commit `96c5e4719`
(`fix(BACKEND-TENSTORRENT-KEEPQUANT): pin the int8-dot sweep for every
registered encoding`), authored by Luca Barbato on 2026-09-09, reached `main`
as a direct push — single parent, no `row/<ID>` branch in its history, no
`(#N)` in its subject — changing `src/vt/tenstorrent/tenstorrent_ops.cpp` and
`tests/vt/test_tenstorrent_backend.cpp`, both feature paths. The checker writes
one unwrapped line, and this is that line verbatim, copied from
`python3 scripts/check-role-discipline.py --base e1b5df1a6 --head origin/main`,
which exits 1:

```
ERROR: 96c5e4719: repository change (src/vt/tenstorrent/tenstorrent_ops.cpp, tests/vt/test_tenstorrent_backend.cpp) reached main without arriving on a task branch. Work happens in its own worktree on a `row/<ID>` branch and lands through a reviewed PR or an authorized local merge naming that branch; never directly on the shared checkout
```

**The advance is minimal by construction, and this was measured rather than
assumed.** `e1b5df1a6..96c5e4719` is **2164 commits, 555 of them merges**.
Unlike the first two advances, whose ranges held no merges, this one does, so
the per-commit check is not equivalent to the range walk: a merge can exempt
content as reviewed through `merged_pr_content`. **Exactly one commit
violates**: `96c5e4719`. The floor moved to that commit and no further.
Setting it to `origin/main` instead would have been the same edit and would
have forgiven every commit above it unexamined, which is the abuse risk 2 says
nothing in this mechanism can detect.

That the whole walk names exactly one commit is measured too, and it is the
same number from both ends. Measured at `origin/main` `ec23d7b89e71`, each rc
captured as `rc=$?` on the command itself and never after a pipe, because a
pipe reports its last stage and has misread a red as a green twice in this
row's history:

| Floor | `check-role-discipline.py --base <floor> --head origin/main` | `ERROR:` lines | `REPORT:` lines |
|---|---|---|---|
| `e1b5df1a6` (old) | **rc 1** | 1, `96c5e4719` | 0 |
| `96c5e4719` (new) | rc 0, `OK: every change on main arrived on a task branch.` | 0 | 0 |

`git merge-base --is-ancestor 96c5e4719dcb1f859cb9e59573309f16c026b523
origin/main` is rc 0, so the new floor is on this history and the resolver's
ancestry guard accepts it. `tests/scripts/test_ci_walk_base.py` re-run over the
new value: 31 tests, 0 failures, 0 errors, with
`RecordedFloorTests::test_recorded_floor_is_an_ancestor_of_head` reading the
committed file rather than a literal.

**One test pinned the old value and had to be re-pinned.**
`tests/scripts/test_check_commit_trailers.py`'s
`test_the_enforcement_floor_no_longer_claims_the_trailer_steps` asserts the file
ends with an exact sha, so that #2322 could prove it narrowed the floor's scope
without moving its value. It is 65/65 green at the base commit, reds with
exactly that one failure the moment the floor line changes, and is green again
at 65/65 once re-pinned — a red-before/green-after proof that the pin is live
and not decorative. The assertion keeps its exact-sha form and its strength;
only the sha and its message move. Deriving the expected value from this spec's
table instead would remove the re-pin, but that is a checker-semantics change
and it is deliberately not made here: an advance having to edit the pin is a
tripwire that puts the value in front of a reviewer, which is what
`### Advancing the floor` asks for.

**The arrival rate held.** Measured over `e1b5df1a6..origin/main` at
`ec23d7b89e71`: **2208 commits, 558 of them merges, spanning 2026-08-25 to
2026-09-11, and one role-discipline violation.** Seventeen days and 2208
commits with one violation is consistent with the rate the second advance
measured (1365 commits, zero violations, over nine days), read as a cluster
and not a rate.

What the advance still does not do is fix anything. `### Advancing the floor`
costs one reviewed commit per unrepairable violation and buys nothing else; the
floor stops an unrepairable commit freezing a gate, and if a fifth violation
lands it will cost a fourth advance.

## Gates

| Gate | Command | Result |
|---|---|---|
| G1 base selection | `python3 tests/scripts/test_ci_walk_base.py` | PASS, 31 tests |
| G2 cancelled-run losslessness | `python3 -m unittest tests.scripts.test_ci_walk_base.CancelledRunLosslessTests`, plus the live replay in `## Outcome` | PASS |
| G3 a new violation still reds | a scratch commit replayed through the real `ci.yml` step bodies | PASS, red in both directions |
| G4 the deadlock is broken | the same bodies replayed on unmutated `HEAD` | PASS, three gates green |
| G5 the floor is load-bearing | the floor moved back to the frozen base, and the four call sites deleted | PASS, both mutations red |
| G6 preflight | `scripts/agent-preflight.sh` | PASS |

## Risks

1. **A violating commit lands between the recorded floor and the merge of this
   row.** The gate reds on that one commit, correctly, and the remedy now exists:
   advance the floor in a reviewed commit that names it. This is the designed
   behaviour and not a regression, but it means the floor value has to be
   re-checked immediately before merge. **This risk has fired twice.**
   `c00b99c7c` landed on 2026-08-24 and the floor was advanced to it by exactly
   that remedy (`### The first advance`); `e1b5df1a6` landed on 2026-08-25 and
   the floor was advanced again on 2026-09-03 (`### The second advance`). The
   obligation it names does not expire with a discharge — `main` moves roughly
   every twenty minutes, so whoever merges re-runs the four commands over
   `<floor>..origin/main` again. Twice is enough to call this the steady state
   rather than an anomaly.
2. **The floor is set too far forward by mistake.** It would skip commits nobody
   examined, and **nothing in this change detects it.** The two ancestry guards
   cover a different mistake: `resolve_base` warns and leaves the base alone
   when the floor is not an ancestor of `HEAD`, and `RecordedFloorTests` fails
   when the recorded floor is not a real ancestor of `HEAD`. Both are about a
   floor that is not on this history — typed ahead of `HEAD`, or from another
   branch — and a floor advanced too far to a commit that really is on `main`
   is an ancestor of `HEAD`, so both accept it. Measured on this branch at
   `f7ef4fe19`, with the floor set to `HEAD` itself, the maximally
   over-forgiving value: `read_floor`'s sha pattern accepts it, `known` and
   `is_ancestor(floor, HEAD)` are both true — which is every assertion
   `RecordedFloorTests` makes — the resolver prints `base fafa16f0f… is behind
   the enforcement floor; walking from f7ef4fe19… instead` and returns `HEAD`,
   and the walk `HEAD..HEAD` is empty, so the grep step iterates zero commits
   and `check-commit-trailers.py`, `check-role-discipline.py` and
   `check-now-current.py` each return **rc 0 vacuously**. This is not the
   fail-closed case in `## Outcome` G5; an unreadable record is an error, and an
   over-forward but readable one is a silent pass. The only mitigation is the
   one in `### Advancing the floor`: the value moves only in a reviewed pull
   request whose body names every commit the advance forgives. The review is
   the control, and there is no second one.
3. **The script fails and takes four gates with it.** It runs under `set -eu` in
   a command substitution, so a crash reds the job. That is fail-closed and the
   right direction, but it makes the script's own suite load-bearing; it is
   registered in `agent-record` and in `scripts/agent-preflight.sh`.
4. **Someone re-inlines the base selection into the YAML.** The suite asserts
   that `ci.yml` carries no residual `base="${LAST_GREEN:-}"` fallback and that
   the script is invoked once per diff-scoped step, so a re-inlining reds.

## Owed

The 44 commits are recorded above rather than owed: no future change can repair
them.

- [#2745](https://github.com/mudler/vllm.cpp/issues/2745) —
  `check-role-discipline.py`'s `WORKTREE_DISCIPLINE_SINCE` names `f236ca960`,
  which is not an ancestor of `main`; its landed twin is `bd6b3936b`. The
  second cutover has therefore never switched on, so integration paths keep the
  exemption the module docstring says it removed. Found while measuring
  `### The second advance`. Not repaired there: repointing the constant changes
  checker semantics and needs its own red-before measurement of how much landed
  history it newly names, which `AGENTS.md` routes to a row and a fresh review
  rather than an in-flow fix.

- `ISSUE-GH-3155` — `sanitize-cpu (address,undefined)` fails on every PR with
  a UBSan alignment error inside `LoadUnaligned<T>` calls in the AVX2 and AVX512
  element matmul kernels. The compiler optimises the `memcpy` in `LoadUnaligned`
  into a direct load, which UBSan flags. The fix adds
  `__attribute__((no_sanitize("alignment")))` to the function.

- `ISSUE-LOCAL-01M41PGNENY6EDHWVE75EEECJA` — the `pr-size` evidence lanes
  and the PowerShell AST stage of `check-windows-portability.py`.

- `ISSUE-LOCAL-01M433BAC470XAV8VYFF8KJ1WD` — the vulkan platform-gate log
  capture and the `EVIDENCE_REQUIRED_TOOLS` pwsh declaration that the pr-size
  install alone could not reach.


## Outcome

Measured on `row/1809` at `6de046d36`, against a clone of that commit with `main`
pointed at it, so a push to `main` could be replayed without touching `main`.

### The instrument

`scripts/ci-walk-base.py` is exercised by its own suite. The three gates are
exercised by READING their `run:` bodies out of `.github/workflows/ci.yml` with
a YAML parser and executing those exact bytes, with each step's declared `env:`
resolved from a supplied event payload. Nothing about the gates is transcribed,
so a change to the workflow changes the evidence.

One trap was hit and is recorded because it invalidates this class of result: the
first no-trailer mutation carried the words "no `FOLLOWING_AGENTS_PROTOCOL`
paragraph" in its own body, which SATISFIED the presence grep and read as a
passing gate. The mutation was re-authored to name no marker at all
(`marker_count=0`, printed before each run) and the gate then went red.

### G4 — the deadlock is broken

Replaying a push of `6de046d36` with `LAST_GREEN = fafa16f0f` (the real frozen
value) and `before = bacb71109`:

| Step | rc |
|---|---|
| `commit-protocol-tag` / presence grep | 0 |
| `commit-protocol-tag` / strict trailer walk | 0 |
| `documentation-checkpoint` | 0 |

The resolver printed `base fafa16f0f… is behind the enforcement floor; walking
from bacb71109… instead` on each. All three are red on `main` today.

### G5 — the floor is what makes them green

Two mutations, each restored and each verified restored by `sha256sum -c` with a
clean `git status`.

| Mutation | Diff | Result |
|---|---|---|
| the floor moved back to `fafa16f0f`, the frozen base | `scripts/ci-enforcement-floor.txt \| 2 +-` | rc 1, 1, 1 — the same 20 grep violations, the same 35 strict ones, the same 6 role-discipline ones |
| the four resolver call sites replaced by the old inline selection | `.github/workflows/ci.yml \| 40 ++++----` | rc 1, 1, 1, and `test_ci_walk_base.py` red at 2 of 31 (`WorkflowWiringTests`) |
| the floor record emptied to a comment, testing FAIL-CLOSED | `scripts/ci-enforcement-floor.txt \| 26 +-` | rc **2**, 2, 2, each step aborting under `set -eu` with `must hold exactly one commit sha outside its comments, found 0` before any checker ran |

The third is the one that had been asserted rather than executed. A floor record
that cannot be read is an ERROR and never "no floor": reading a broken record as
absent would restore the ratchet silently, which is the failure this file exists
to end. `set -eu` makes the command substitution's non-zero status abort the
step, so the direction is fail-closed and now measured.

The second is the reachability mutation: a resolver nothing calls resolves
nothing, and both the gates and the focused suite notice the deletion.

### G3 — a new violation on a new commit still reds

Each mutant is a real commit authored on top of `6de046d36`, replayed as a push
whose `before` is `6de046d36`. A positive control shares the mutants' path and
trailers so a red is attributable to the violation and not to the harness.

| Mutant | grep | strict | doc-checkpoint |
|---|---|---|---|
| `059dfb59d`, no marker and no trailers | **1** | **1** | 0 |
| `d47fd7e8b`, product path, full trailers, no PR reference | 0 | 0 | **1** |
| `003b71af0`, the control: same product path, full trailers, `(#1809)` | 0 | 0 | 0 |

The strict walk named `059dfb59d` on all three of its contract clauses, and the
role-discipline step named `d47fd7e8b` with the path it touched.

### G2 — a cancelled run is still lossless

The unit case is `CancelledRunLosslessTests`, which replays C1-green,
C2-cancelled, C3-pushed against a throwaway repository and carries the naive
`github.event.before` base as its positive control.

Replayed live through the real step bodies as well, with the floor in place. C2
is a VIOLATING commit whose run is cancelled, so `LAST_GREEN` stays at
`6de046d36`, and C3 is clean:

| Run | base | grep | strict |
|---|---|---|---|
| C3 pushed, `before = C2`, `LAST_GREEN = 6de046d36` | `6de046d36` | **1**, naming `7f28451310f5` | **1** |

The cancelled run's violation is caught by the next run. Under the naive base,
`rev-list C2..C3` returns `['54d9021a…']` alone and `7f28451310f5` is covered by
nothing, which is #863 exactly. The floor did not interfere in this replay
because it was placed behind `LAST_GREEN`. That is the arrangement the
losslessness argument needs, and, as
`### Cancelled runs stay lossless` now records, it is **not** the arrangement
`main` is in: `LAST_GREEN` has been frozen since 2026-08-12 and the floor is
ahead of it, so the resolved base on `main` is the floor. The replay proves the
clamp does not break losslessness when it is behind; it does not claim `main`
is there.

### What was rejected while implementing

Three existing assertions matched the old inline shell as a STRING and had to
move rather than be deleted, because the rule they were about now lives in the
resolver:

- `test_main_baseline.py`'s shim stubbed every `python3` call, which made the
  resolver return an empty base and skipped the very checker calls two cases
  exist to require. The shim now EXECUTES the resolver — it is not a checker, it
  is the thing that decides what the checkers get — so those cases test the real
  composition rather than a transcription of the rule.
- `test_the_base_falls_back_when_no_successful_run_is_found` asserted the literal
  `base="$PUSH_BASE"`. It now asserts that both event values reach the resolver
  AND executes the resolver to prove the degradation, which is stronger than the
  literal it replaces.
- `test_agent_gates.py`'s `test_ci_role_suite_uses_exact_event_range_not_detached_head`
  asserted `base="$PR_BASE"`. It now asserts the event values reach the resolver,
  refuses three checkout-derived base forms, and executes the resolver on both
  lanes.

Widening any of the three to make it pass was available and was not taken. A
string match that no longer sees the rule is not a weaker gate, it is no gate.

### Residue

`main` advanced to `849a7dd73` while this row was in flight, and the floor was
re-checked against it rather than assumed. Both new commits, `0a0a53e5a` and
its child `849a7dd73`, are CLEAN on all three gates —
`check-commit-trailers.py --range` returns `OK: commit trailer contract` and
`check-role-discipline.py` returns `OK: every change on main arrived on a task
branch` over `bacb71109..849a7dd73`. That verdict is true of that range and
**superseded** by the re-measurement at `e6f4f566f` further down: `main` moved
again and the floor no longer stands.

The merge commit on this branch names `0a0a53e5a` as the tip, which is wrong:
`git log --oneline` prints newest first and the pair was read in that order.
`849a7dd73` is the tip and `0a0a53e5a` is its parent. The range measured was
`bacb71109..origin/main`, which covered both either way, so the verdict is
unaffected and only the name was.

This re-check is not a formality, it is risk 1 arriving. If a violating commit
lands before this merges, the gate reds on that one commit, which is the
designed behaviour, and the remedy is a one-line reviewed floor advance that
names it. Whoever merges this should repeat those commands over
`bacb71109..origin/main`, and one of them now reds — see below.

`scripts/agent-preflight.sh` does **not** report "All gates green" on this
branch, and the earlier claim that it did at `173b7f32d` was wrong. Rerun at
`f7ef4fe19` against `origin/main` `e6f4f566f`: **rc 0, zero failures, and two
SKIPS** — `commit-trailers` and `commit-style`, both with the reason
`origin/main … is not an ancestor of HEAD, so this branch is behind it and the
trailer gates did NOT run`. The script says so itself in the same breath:
`NOT a green preflight: a skipped gate reported nothing about this tree`. A rc 0
that carries a SKIP is exactly the third state `scripts/agent-preflight.sh`
documents at its top and the exit status cannot express, so reading the rc alone
is how the wrong claim was made.

Both skipped gates were therefore run by hand over the branch's own range,
`849a7dd73..HEAD` — the merge base with `origin/main` to the head, which covers
every commit this branch adds. Measured at `f7ef4fe19` and measured again at the
head of this repair, with the same verdict both times:

| Gate | Command | rc |
|---|---|---|
| `commit-trailers` | `python3 scripts/check-commit-trailers.py --range 849a7dd73..HEAD` | **0**, `OK: commit trailer contract` |
| `commit-style` | `python3 scripts/check-commit-style.py --range 849a7dd73..HEAD` | **0**, `OK: commit writing style` |

The SKIP was not a defect in these commits: it was the branch being behind
`origin/main`, and a trial merge conflicted in `scripts/agent-preflight.sh`'s
`SUITES` array against `af320abb2`, which also meant GitHub could produce no
merge ref for the pull request lane to check out.

**Both are resolved.** `origin/main` `3574065e7` is merged into the branch and
the conflict is taken as a union of the two additions: `af320abb2`'s
`test_ltx2_dit_attn_knob_arms`, `test_ltx25_ab_memwatch` and
`test_tower_skip_rss_report`, plus this branch's `test_ci_walk_base`. Rerun
after the merge, `scripts/agent-preflight.sh` reports `ok commit-trailers` and
`ok commit-style` inside the script, against `origin/main` `3574065e7`, with no
SKIP. The merge is also what makes the advanced floor recordable:
`RecordedFloorTests::test_recorded_floor_is_an_ancestor_of_head` asserts the
recorded value is an ancestor of `HEAD`, `c00b99c7c` landed on `main` after this
branch left it, and the suite reds at 1 of 31 without the merge and is 31 of 31
with it. `.agents/issue-index.md`'s #1809 row appears exactly once afterwards
and is no longer the tail, which is correct for an append-only union file; its
prose still says 41 because an index row is never rewritten.

One site of the corrected `LAST_GREEN` claim is deliberately left alone:
`scripts/ci-walk-base.py`'s module docstring still calls a floor behind
`LAST_GREEN` "the steady state". The resolver, its data file, `ci.yml`, its
suite and the three checkers were all excluded from this review repair so that
the reviewed mechanism stays byte-identical, and a docstring edit inside that
boundary is not worth reopening it for. It is named here rather than left to be
found: the sentence is wrong for the same reason `### Cancelled runs stay
lossless` was, and it should go in whichever change next touches that file.

**Risk 1 arrived, and the floor has been advanced once. RESOLVED.** It was
first measured 2026-08-23T22:30Z at `origin/main` `e6f4f566f`, over
`bacb71109..origin/main`, nine first-parent commits: the presence grep rc 0,
`check-commit-trailers.py` rc 0, `check-now-current.py` rc 0, and
**`check-role-discipline.py` rc 1** on `c00b99c7c`
(`fix(LTX25-DIT-ATTN-ARM-PARSE)`), which reached `main` with
`src/vllm/model_executor/models/ltx2_device.cpp` and its test without arriving
on a task branch. That review repair deliberately left the floor alone, because
advancing it silently inside a review repair is what `### Advancing the floor`
forbids.

The advance is this commit, and it is a separate reviewed act with the argument
in its body. Re-measured independently at `origin/main` `d60692c89`, ten
first-parent commits, so the finding was reproduced rather than inherited:

| Floor | grep step | `check-commit-trailers.py` | `check-role-discipline.py` | `check-now-current.py` |
|---|---|---|---|---|
| `bacb71109` (old) | rc 0 | rc 0 | **rc 1**, `c00b99c7c` | rc 0 |
| `c00b99c7c` (new) | rc 0 | rc 0 | rc 0 | rc 0 |

Each rc was captured as `rc=$?` on the command itself and never after a pipe,
which reports the last stage of the pipeline and has misread a red as a green
twice in this row's history. `git merge-base --is-ancestor c00b99c7c
origin/main` is rc 0, so the new floor is on this history and the resolver's
ancestry guard accepts it. `tests/scripts/test_ci_walk_base.py` re-run over the
new value: 31 tests, 0 failures, 0 errors.

The gate still bites over the new floor. A scratch commit on top of
`origin/main` whose message contains no `FOLLOWING_AGENTS_PROTOCOL` string
anywhere — marker count printed as **0** before the run, because a mutation
whose own message mentions the marker satisfies the presence grep and reads as
a pass — reds both trailer steps by name over `c00b99c7c..<scratch>`. The tree
was restored and the restore proved by sha256 on both changed files, not by a
`git status` that a mutation can leave clean.

Nothing about the value is permanent. `main` keeps moving, so whoever merges
repeats the four commands over `<floor>..origin/main` and, if one reds again,
performs another advance the same way. **One did.** `check-role-discipline.py`
reddened on `e1b5df1a6` and the floor was advanced a second time on 2026-09-03
for #2743; the measurement is in `### The second advance` rather than here,
because this section records the FIRST advance and is left as measured.

`test_cpu_x86_llamacpp_floor` was red in two earlier runs and is #618, not this
row. It was discriminated rather than asserted, twice over. Pristine
`origin/main` at `0a0a53e5a`, run serially in its own clone with no change from
this branch in it, fails
`test_a_contended_leg_is_discarded_and_never_summarised` and
`test_the_published_figures_are_computed_not_transcribed` with
`NO_QUIET_WINDOW` at loadavg 63-68. And the green run above is the same tree as
the red one plus a prose edit, taken after the box quietened, which is what a
load-dependent harness does. Never asserted from the green run alone: a gate
that passes once is not a gate that cannot fail.

`agent-record`'s missing-`hugo` red (#1722, #1726) and the two `windows-msvc`
reds (#584) are inherited and unaffected by this row.

## Now

`DONE` pending review. Spec committed ahead of the implementation on `row/1809`;
the evidence above was measured on the implementation commit.
