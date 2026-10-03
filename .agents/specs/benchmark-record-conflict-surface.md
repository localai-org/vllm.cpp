# `.agents/benchmark-record.md` is a lock, and `merge=union` cannot unlock it

Issue: [#1373](https://github.com/mudler/vllm.cpp/issues/1373).
Row: `ENG-RECORD-CONFLICT-SURFACES`.
Measured against `origin/main` `9fb40279d` on 2026-08-31.

[#1373](https://github.com/mudler/vllm.cpp/issues/1373) asks for one line of
`.gitattributes`: give `.agents/benchmark-record.md` the `merge=union` attribute
that `.agents/issue-index.md` has, so that two pull requests appending an entry
stop conflicting.

**The answer is no, on three independent grounds, and this spec records why so
that nobody re-derives it.** The attribute the issue asks us to copy no longer
exists. The shape it belongs to was retired from `AGENTS.md` §Records two days
before this spec. And it would be unsound here even if both of those were false,
because **23.9% of the writes to this file are not appends**, and union against a
non-append edit silently produces a file neither side wrote.

## Scope

**In scope.** The soundness question for `merge=union` on
`.agents/benchmark-record.md`; the write-pattern measurement that answers it; a
comparison of the four available shapes against `AGENTS.md` §Records; a
recommendation; two documentation repairs that this investigation found and that
nothing else owns; and the correction of the one prior statement this
investigation falsified, `.agents/specs/retire-shared-record-surfaces.md:106`,
annotated in place as **O4** because it is the probable source of #1373.

**Out of scope, deliberately.** `.agents/parity-ledger.md`, which
`scripts/check-pr-size.py:78-83` classifies in the same `APPEND_ONLY_FILES` set
and which was not measured here. Any rewrite, reflow, compaction or re-ordering
of the 396 existing entries: the file is a forensic archive and its content is
evidence. Any change to `docs/BENCHMARKS.md`, which is a keyed table and a
different surface with a different defect (#460).

**Not a correctness change.** No product source, kernel, gate semantic, or
measured number moves.

## The issue's premise is falsified by the tree

`#1373` cites `.gitattributes:7`. At `9fb40279d` the whole file is three lines:

```text
# Vendored Triton AOT artifacts are generated code (embedded cubins): collapse
# in diffs/review and exclude from language stats. Regen: scripts/regen-triton-aot.sh
src/vt/cuda/triton_aot_vendored/** linguist-generated=true
```

There is no line 7 and no `merge=union` anywhere in the tree. Commit
`7dc2ef1ea` (2026-08-29, `feat(ENG-RECORD-CONFLICT-SURFACES): W6`) deleted both
the attribute and its subject:

```text
-# The issue index is append-only, so two branches appending a row merge
-# cleanly instead of conflicting. See .agents/specs/issue-intake.md
-.agents/issue-index.md merge=union
```

`.agents/issue-index.md` now lives in `.agents/completed/`. The same commit
removed the shape from `AGENTS.md` §Records, which today reads: *"An append-only
file with `merge=union` is NOT one of them."* GitHub does not run
`.gitattributes` merge drivers, so the driver resolves the collision on the
author's machine and the forge conflicts anyway (#883).

So the requested remedy is barred by policy and its factual premise is gone.
Per §Records — *"An issue the tree falsifies closes with that evidence"* — the
remedy half of #1373 is closable on this spec. **The complaint half is not.** The
file is still a shared surface that concurrent pull requests write, and that part
of the issue is answered below on its own evidence rather than dismissed.

## The soundness question, answered on the writes themselves

`merge=union` is sound only while every write is genuinely an append. It is not
a merge strategy that understands the file; it concatenates both sides' version
of a conflicting region. Against a non-append edit it emits both.

### Method

Every non-merge commit touching `.agents/benchmark-record.md` was classified
from `git show --unified=0` against the file's line count at the commit's parent.
A commit is `APPEND` iff every hunk's old-side start is at or past the parent's
last line **and** the diff deletes zero lines. Everything else is `NON_APPEND`.
222 commits carry a diff for the file.

**The predicate and its implementation disagreed on exactly one commit, and this
spec now carries the predicate's answer.** The first classifier run compared each
hunk start against `parent_lines - 1` rather than `parent_lines`, which is not
what the sentence above says. The two readings differ on `68bc6ef49` alone: its
sole hunk is `@@ -15494,0 +15495,69 @@` against a 15,495-line parent whose final
line is blank, so it inserts *before* that blank line. The stated predicate calls
that `NON_APPEND`; the looser implementation called it `APPEND`. **The spec states
53 of 222, 23.9%**, which is the stated predicate's count and the larger of the
two. The looser reading gives 52 of 222, 23.4%. Nothing in this spec's verdict
turns on the 0.5pp: both readings put roughly a quarter of the writes outside the
append shape, which is the finding.

### Result

| Class | commits | share |
|---|---|---|
| pure tail append | 168 | 75.7% |
| **non-append, with deletions** | **18** | **8.1%** |
| **non-append, mid-file insert** | **18** | **8.1%** |
| **non-append, head prepend at line 21** | **17** | **7.7%** |
| initial create | 1 | 0.5% |
| **all non-append** | **53** | **23.9%** |

**The rate is rising, not decaying:**

| window | non-append | share |
|---|---|---|
| last 25 commits | 12 / 25 | **48.0%** |
| last 50 commits | 18 / 50 | 36.0% |
| last 100 commits | 23 / 100 | 23.0% |
| all 222 commits | 53 / 222 | 23.9% |

The three windowed rows are unchanged by the predicate reconciliation above:
`68bc6ef49` is from 2026-08-07 and falls outside all of them.

The newest write to the file, `b426de5ac` (2026-08-30), is itself a non-append:
two hunks, 165 insertions and **6 deletions**, one hunk at `-28726` against a
29,129-line parent. This is not a historical artefact that a convention could
retire. It is what the file's writers do now, and they do it more than they used
to.

This reproduces the observation that prompted this investigation. A cumulative
merge of `origin/main` showing a header hunk at `@@ -19` plus two entry hunks
deep in the file plus an append is exactly the shape of several commits in the
`NON_APPEND` classes above — `6756f9131` (4 hunks at 21, 289, 474, 26503),
`1db7e59cf` (6 hunks, 11 deletions), `438305e15` (2 hunks, 7 deletions at 24325
and 24336). No single commit carries that exact hunk triple, because the observed
diff was cumulative over a range rather than one commit; the class it belongs to
is confirmed 53 times over.

### Two demonstrations that union corrupts this file

Run in throwaway repositories under the scratchpad, with
`merge.union.driver` configured exactly as `.gitattributes` would invoke it. No
build, no product code.

**Demo 1 — two concurrent head-prepends, the file's own second convention.**
Both branches insert a new entry directly under the `# Benchmarks` heading,
which is what 17 commits in the table above do, at line 21 every time.

```text
# Benchmarks

## Entry NEW-2 (2026-08-06)
## Entry NEW-1 (2026-08-05)
Placement. Newest-first. This sits above Entry A.

## Entry A (2026-08-01)
FA_USABLE=0. Unresolved.
```

`git merge` exited **0** with no conflict marker. `## Entry NEW-2` is now a
**heading with no body**: its evidence was silently dropped, because the two
entries' body lines were identical and the union collapsed them. A forensic
record acquired an entry that claims a measurement and contains none.

**Demo 2 — a retraction merged against a concurrent edit to the same entry.**
This is the `NON_APPEND` deletions class, 18 commits, and retraction is precisely
what those commits do.

```text
## Entry A
SPEED: 1.50x vs vLLM.
Gate: PASS (reconfirmed 08-30).
SPEED: RETRACTED, the arm was misbuilt.
Gate: FAILING.
```

Again exit **0**, no conflict. The retracted claim and its retraction now stand
in the same entry, as do `PASS` and `FAILING` for the same gate. A reader cannot
tell which is live, and neither can a checker.

**Verdict: `merge=union` is UNSOUND for `.agents/benchmark-record.md`.** It
converts a visible conflict, which a human resolves, into a silent corruption of
the one record whose entire purpose is to be trustworthy about superseded
numbers. Adding it would be strictly worse than the problem it solves, and it
would not even fix that problem, because the forge ignores the driver.

## Who writes this file, and how

**No generator exists.** `.agents/benchmark-record.md:10-11` tells the reader:

> `scripts/roll-benchmark-record.py` moves any narrative section that
> accumulates in the scoreboard down into this file.

That script was added by `8a0744ae0` and **deleted by `1db7e59cf`** (2026-08-22,
`docs: retire shared status and split benchmark details (#1714)`). It is not in
the tree. Every write to this file is hand-authored by an agent, and nothing
mechanical constrains a write to be an append. This is repair **R1** below.

**No gate requires or budgets a write.** `scripts/check-pr-size.py:78-83` places
the path in `APPEND_ONLY_FILES`, but that is a classification feeding
`classify()` at `:448`, and the per-class line budgets were retired on
2026-08-10, so the classification now carries no obligation and no limit. The
only instruction to write the file is prose:
`.agents/benchmarking.md:221` — *"Accepted and pending results go in
`benchmark-record.md`"*. The append-only property is a convention, gated by
nothing, which is what the dflash2 spec already recorded at
`.agents/specs/dflash2-spec-decode.md:3033-3036` when it declined to annotate an
entry in place.

**The file contradicts itself about placement, three ways.** Line 3 calls it an
"Append-only forensic record". 168 commits append at EOF. 17 commits prepend at
line 21, and the entry now sitting at line 22 opens with *"**Placement.**
Newest-first. This sits above `QUANT-QWEN38-27B-GGUF-ARM W3`"*. So the header,
the entry prose, and the actual writes each assert a different convention. This
is repair **R2**, and it matters for more than tidiness — see the anchors below.

## Structure: is the header separable from the entries?

Yes, and cleanly. Lines 1-18 are the header: purpose, the pointer to
`docs/BENCHMARKS.md`, and the provenance note for the 2026-08-04 migration. Line
18 is a `---` rule, line 20 is `# Benchmarks`, and the 396 entries are `## `
headings from line 22 to EOF.

An append-only rule could therefore be enforced for the entry region while the
header stayed deliberately editable. **That does not rescue `merge=union`,**
because the attribute binds per file and cannot be scoped to a line range, and
because 36 of the 53 non-append commits edit the *entry* region rather than the
header. It is recorded because it is the one structural fact that would matter
if a future row ever wants an append-only gate on the entries, which this spec
does not propose.

## The options, compared

### Option 1 — add `merge=union`

**REJECTED, three times over.** Unsound (23.9% non-append, and the two demos
above). Ineffective (GitHub does not run merge drivers; #883). Barred (§Records
names the shape as not admissible, since `7dc2ef1ea`). Nothing further is owed to
this option.

### Option 2 — split per row, read with a glob

This is the shape §Records prefers, and it removes the lock rather than
mitigating it. **REJECTED as specified, and deferred in its workable variant.**

*As literally specified — one file per row — it is not mechanically possible.*
The 396 entry headings yield **184 distinct leading tokens**, of which **128 occur
exactly once**:

```sh
git show 9fb40279d:.agents/benchmark-record.md \
  | grep '^## ' | sed 's/^## //' | awk '{print $1}' | sort -u | wc -l   # 184
```

Stripping backticks and trailing punctuation gives 182; splitting on punctuation
as well gives 173. The figure is stable in the low 180s under every tokenization
tried, and the 313 this spec first asserted does not reproduce under any of them.
It was wrong and is corrected here; the count of *distinct whole headings* is 395,
so 313 was not that either.

The conclusion is unaffected, because it never rested on the size of the number.
Many leading tokens are not row IDs at all: `Metal` (21), `DeepSeek-V4-Flash`
(21), `2026-08-09` (14), `MiniMax-H3` (13), `CPU` (10). Only `SPEC-DSPARK` (36,
as 24 `SPEC-DSPARK:` plus 12 `SPEC-DSPARK`) and a handful of others cluster, and
128 tokens name exactly one entry each. The owning row simply cannot be derived
from the heading for most entries, so the migration would require a human to
classify 396 forensic entries, and a misclassification silently files evidence
where no one will look for it.

*The workable variant is one file per entry* —
`.agents/benchmark-records/<date>-<slug>.md`, chronology derivable from the
filename, index derived at read time. That genuinely removes the lock: two
concurrent entries never touch the same path. Its costs are concrete and were
measured:

- **210 citing lines across 101 files** point at the file. All become wrong.
  Measured with the strict pattern, at the pinned base:

  ```sh
  git grep -I -c 'benchmark-record\.md' 9fb40279d \
    -- . ':(exclude).agents/benchmark-record.md'      # 210 lines over 101 files
  ```

  This spec first reported **252 lines over 117 files**, which is the same command
  with the pattern `benchmark-record` and no `\.md`. That looser pattern also
  matches `roll-benchmark-record.py` (6 hits) and the bare string
  `benchmark-record` (37 hits, which includes the `benchmark-records/` directory
  this option proposes). Those are not citations of the file, so 210/101 is the
  honest count and the one that would actually break.
- **35 of those name a line number**, including two in product source:
  `src/vt/cuda/cuda_mamba2_ssd.cuh:54` (`benchmark-record.md:532`) and
  `src/vllm/model_executor/model_loader/gguf_keep_quant.cpp:290`
  (`benchmark-record.md:10722`).
- **All 201 relative links *inside* the file break.** **163** are `.agents/`-relative
  (`](specs/multimodal-speed.md)` 12 times, `](parity-ledger.md)` 5,
  `](completed/state-events/)` 4, and so on) and the other **38** are `../`-prefixed
  (`](../README.md)`, `](../include/vt/dtype.h)`), which resolve through `.agents/`
  to the repository root. This spec first described all 201 as `.agents/`-relative,
  which is imprecise. The load-bearing claim is unchanged: moving the content one
  directory deeper re-bases both forms, so all 201 break either way, and the
  migration must therefore either rewrite 201
  links — which violates the byte-for-byte preservation this archive exists to
  provide — or extend `link_bases()` in `scripts/check-agent-record.py:1078-1107`,
  which today already carries a special second base for this exact file and
  documents at length (#460) why that is weaker than followability.

The cost is real and the benefit is proportional to the conflict rate, which is
measured below and is low. **Filed as owed debt with a trigger, not done now.**

### Option 3 — derive it at read time

**REJECTED, and not close.** There is no source to derive from. The entries are
hand-authored narrative: refuted hypotheses, profiler tables, reasoning about why
a lever was abandoned, and the reasons a default has its value. The only
generator this file ever had was deleted (above). Derivation requires an
authority that already holds the facts in structured form; for a prose archive of
human judgement, none exists and none can be built.

### Option 4 — leave the file, record why

**RECOMMENDED**, with repairs R1 and R2, and with option 2's workable variant
filed as owed.

The conflict rate does not justify migrating a 24,000-line forensic archive
today, and the comparison is against a surface this repository has already
migrated for exactly this reason:

| surface | commits writing it | open PRs conflicting |
|---|---|---|
| `.agents/issue-index.md` (retired `7dc2ef1ea`) | **115 of last 200 (57.5%)** | 16 of 21, 4 of them on that file alone |
| `.agents/benchmark-record.md` (this file) | **91 of last 1000 (9.1%)**, 2 of last 200 | 2 of 29, per the measurement below |

The 1000-commit window spans 2026-08-12 to 2026-08-31, 19 days; the 200-commit
window spans only the last two days, which is why the two figures differ. Either
way the file is written at roughly one-sixth the rate that justified retiring the
index.

That is also not a new finding. `.agents/specs/retire-shared-record-surfaces.md`
measured this file on 2026-08-11 at **2 of 29 conflicting open pull requests**,
the same tier as `docs/FEATURES.md`, and placed it under *"Out of scope,
deliberately… measured below as not implicated; removing them would be scope this
evidence does not support."* Nothing in the present measurement overturns that.
The write rate is still an order of magnitude below the surface that did get
retired.

**That spec is cited here for its conflict-rate measurement only.** Its
`## What the evidence exonerates` list also calls this file *"append-only and
already union-mergeable"*, and this spec falsifies both halves of that clause.
It is the likely source of #1373. It is corrected in place, and tracked as **O4**
below.

**The honest counter-argument, recorded rather than buried.** A conflicted pull
request carries zero check-runs, because GitHub never schedules CI for one, so it
reads as unverified rather than red (#2248). Each collision therefore costs more
than its frequency suggests.

**Both rates are decelerating, which strengthens this recommendation rather than
qualifying it.** This spec first said the file grows by *"roughly 500 lines a
day"*. That is a life-of-file average and it is not the current rate. Over the
26 days from `8a0744ae0` (2026-08-04, 11,424 lines) to `b426de5ac` (2026-08-30,
29,288 lines) the file grew **687 lines a day**. Over the last week it grew from
28,675 lines on 2026-08-23 to 29,288 on 2026-08-30 — **613 lines in 7 days, 88 a
day**, with no single day above +218. The write rate falls the same way: 9.1%
over the last 1000 commits against **1.0% over the last 200**. So the file is
being written less often *and* growing more slowly than at any point in its life,
and the case for deferring option 2 is stronger today than the raw averages
suggest. It is filed as owed with a trigger rather than refused because
deceleration is not a guarantee, and the trigger is what converts that into a
decision with an expiry.

## Recommendation

1. **Do not add `merge=union`.** Close the remedy half of #1373 citing this spec.
2. **Keep the file as one file.** Record the exception in the landing commit
   message, per §"Changing the rules or a checker" — the project has no waiver
   registry, so the argument lives with the diff.
3. **R1 — correct the header.** Lines 10-11 name `scripts/roll-benchmark-record.py`
   as if it maintained the file. It was deleted at `1db7e59cf`. A reader currently
   believes a generator owns this file and that their hand-edit is unusual; the
   opposite is true.
4. **R2 — standardise on tail-append, and say so in the header.** Not for
   tidiness: **head-prepends are what break the line-number citations.** An
   insert at line 21 shifts every one of the 35 recorded anchors, and it has
   happened 17 times. Tail-append leaves every existing line number stable, which
   is the only reason those anchors can survive at all. This also halves nothing
   and costs nothing — 76% of writes already do it.
5. **File as owed** (`## Owed` below): the per-entry split with its trigger, and
   the conversion of line-number citations to heading anchors.

## Owed

- **O1 — the per-entry split.** `.agents/benchmark-records/<date>-<slug>.md`,
  read with a glob, content preserved byte-for-byte, `link_bases()` extended
  rather than links rewritten. Below the trigger the migration costs more than it
  returns.

  **Trigger, arm 1: the file is written by more than 15% of the last 200
  non-merge commits.** Compute it with G5. This spec first set that arm at 25%,
  which was chosen rather than derived and could not fire. The recalibration is
  derived from this file's own history, at the pinned base:

  | measure | value |
  |---|---|
  | current, newest 200-commit window | **2 / 200 = 1.0%** |
  | mean over the last 1000 commits | **91 / 1000 = 9.1%** |
  | peak 200-commit window within the last 1000 | **35 / 200 = 17.5%** (2026-08-18 .. 08-23) |
  | peak 200-commit window all-time | **85 / 200 = 42.5%** (2026-08-04 .. 08-07) |

  Sweeping every one of the 3070 possible 200-commit windows separates the two
  thresholds cleanly, and by date rather than by argument:

  | threshold | windows exceeding it | most recent such window |
  |---|---|---|
  | old, >25% | 211 of 3070 | ends **2026-08-09** |
  | new, >15% | 372 of 3070 | ends **2026-08-23** |

  Every window that clears 25% falls between 2026-08-01 and 2026-08-09 — the
  migration burst that created this archive on 2026-08-04, and nothing a future
  writer would repeat. So the old arm has been unreachable for the file's entire
  steady-state life, three weeks by the pin date. 15% sits below the 17.5%
  steady-state peak and was last cleared on 2026-08-23, eight days before the
  pin, so it is a live threshold rather than a decorative one. It also sits 1.6x
  above the 9.1% mean over the last 1000 commits, so it does not fire on this
  file's ordinary write rate, per §"How we write" — a gate that fires on ordinary
  work is the defect. A trigger above everything the file has done since it was
  established is the opposite defect: a decision not to act whose review trigger
  cannot fire is a decision with no expiry.

  **The recalibration does not change the recommendation.** The current rate is
  1.0%, an order of magnitude below the new 15% arm, so option 4 still holds and
  option 2 stays deferred.

  **Trigger, arm 2: more than 4 open pull requests report `CONFLICTING` with the
  file as a path.** Unchanged, and well placed against the measured 2 of 29. It
  cannot be computed offline: it needs tracker access, which this campaign does
  not have (see the note at the end of this section), so only arm 1 is gated by
  G5.
- **O2 — the 35 line-number citations.** They are already unreliable. Checked at
  `9fb40279d`: `benchmark-record.md:532`, cited from
  `src/vt/cuda/cuda_mamba2_ssd.cuh:54` as recording *"a pre-rounded `v²`
  differing by <= 1 ulp… and flipping a near-tie"*, resolves to *"A container
  cannot drop the host page cache, so this run took an `rc hold` on"*. Four more
  spot-checked anchors (`:4654`, `:10722`, `:17209`, `:19021`, `:24510`) likewise
  land mid-sentence on unrelated prose. They should cite heading text, which
  survives every insert. Needs its own issue.
- **O3 — `scripts/check-pr-size.py:93`** still lists `.agents/issue-index.md` in
  `PROJECT_RECORD_FILES`, a path deleted by `7dc2ef1ea`. Harmless today, because
  classification carries no budget, but it is a stale reference to a retired
  surface. Needs its own issue.
- **O4 — `.agents/specs/retire-shared-record-surfaces.md:106`** asserts that this
  file is *"append-only and already union-mergeable."* Both halves are false, and
  this spec measures both: 53 of 222 writes (23.9%) are not appends, and the file
  has never carried `merge=union` at any commit — the attribute entered the tree
  on 2026-08-15 in `51e0cb5b1`, four days after that line was written in
  `87308dea3`, and it named `.agents/issue-index.md`. This is the most probable
  origin of #1373's false premise, and this spec cites that file twice for its
  conflict-rate measurement, so leaving the clause unmarked is what would let the
  question be derived a third time. **Annotated in place in the same change as
  this spec**, rather than deleted, because a reader who reaches that line needs
  the correction attached to it; the retirement decision it supports is unaffected
  and is deliberately not rewritten. Needs its own issue only if the annotation is
  judged insufficient.

**Why none of O1-O4 carries an issue number.** `AGENTS.md` §"Every change starts
from an issue" requires an issue for each, and §Records requires that an issue a
change references name an owning row or be listed under a spec's `## Owed`. This
section is that listing, and `ENG-RECORD-CONFLICT-SURFACES` is that row. The
issues are not filed because **this campaign has no tracker write access**: the
GitHub API answers as a different credential and returns *"Could not resolve to an
issue"* for every issue in this campaign, including #1373 itself. Those issues are
hidden from this credential, not deleted — a mass-write incident on this
repository previously turned exactly that signal into a false claim that 817
issues had been deleted, and two automation accounts were suspended in one day.
So no `gh` call was made here, by instruction. The absence of issue numbers is a
recorded external blocker, not an oversight, and filing them is owed to whoever
next holds tracker access.

## Risks

- **R-1: the recommendation is a decision not to act, so it decays.** If the
  write rate climbs, option 4 silently becomes wrong and nothing fires. Mitigated
  by O1's arm 1, which **G5 computes directly** and reports against its threshold,
  so the check is arithmetic rather than judgement. This spec first claimed the
  trigger was *"checkable with the two commands in §Gates"*, and it was not: G1
  counts commits carrying a diff and G2 uses a 1000-commit window, while arm 1 is
  specified over 200. Neither computed either arm, so the mitigation named for
  this risk did not exist. G5 is that mitigation. Arm 2 stays unmitigated offline
  and is recorded as such in O1.
- **R-2: R2 standardises a convention that is not gated,** so it can drift back.
  Accepted deliberately. An append-only gate on the entry region is possible
  (§Structure) but would fail every legitimate retraction, which is 8.1% of
  writes, and this spec's whole finding is that retractions are legitimate here.
  A gate that fires on ordinary work is the defect.
- **R-3: the conflict-rate denominators are windows over `main`,** and `main`
  moves. Both numbers are pinned to `9fb40279d` in §Evidence so a re-measurement
  is comparable rather than merely different.
- **R-4: closing the remedy half of #1373 while its complaint half survives**
  risks the complaint being lost. Mitigated by O1, which carries the complaint
  forward with a trigger and an owner.

## Tests

This spec changes no executable behaviour, so it ports no test and adds no gate.
The two claims it makes that could be wrong are both executable, and both were
run:

1. **The union-corruption claim** is demonstrated by the two throwaway
   repositories in §"Two demonstrations". Each is reproducible in under ten
   seconds with no build: `git init`, set `merge.union.driver`, write
   `rec.md merge=union` to `.gitattributes`, branch, edit both sides, merge.
   Both exit 0 and both produce the corrupt file shown.
2. **The write-classification claim** is reproducible from `git` alone with the
   commands in §Gates. It reads only committed history, so it is stable against
   anything in the working tree.

The mutation a reviewer should perform: drop the deletion term from the `APPEND`
predicate, so that a commit is judged by hunk position alone. **The non-append
total falls from 53 to 51, not from 53 to 35.** That result was measured, and it
corrects the number this section first asserted from arithmetic.

The two-commit delta is the finding, not a weakness in the predicate. It says
that **16 of the 18 deletion-bearing commits are already non-tail by position**:
when a writer edits this file destructively, they almost always do it in the
middle of the archive, because that is where the entry being corrected lives.
Only `7c84d3710` and `c3fb0d247` delete lines while writing at the tail, one
line each. So the
23.9% figure does not depend on the deletion term at all — position alone
recovers 51 of the 53 — and a reviewer who suspects the deletion count is doing
the work can delete it and still reach the same verdict.

3. **O1 arm 1 is reachable**, which the 25% threshold this spec first carried was
   not. G5 was run at two bases to separate the two outcomes rather than assert
   one. At the pinned base `9fb40279d` it prints `2/200 = 1.0% -> not fired` and
   exits 0. At `11ccdcf76`, the base of the densest 200-commit window in the last
   1000, it prints `35/200 = 17.5% -> FIRES` and exits 1. The same window under
   the old 25% threshold does **not** fire. A trigger that cannot be shown firing
   on any real history is not a trigger, so this is recorded as executed evidence
   rather than as a claim about the arithmetic.

## Gates

Both run offline, in seconds, with no build and no GPU.

Every command names the pinned base `9fb40279d` explicitly. Reading `HEAD`
instead makes them drift: G2 prints 91 at the base and **90** at `5fe1bba6a`,
one commit later, because the 1000-commit window slides by one. A gate whose
value moves when nothing it measures has changed cannot be re-measured against.

```sh
# G1 — the write-pattern classification, at the pinned base.
git log --no-merges --format=%H 9fb40279d -- .agents/benchmark-record.md \
  | wc -l                                                                # 222

# G2 — the conflict-rate denominator, 1000-commit window, PINNED to the base.
git log --no-merges -1000 --name-only --format=%H 9fb40279d \
  | grep -c '^\.agents/benchmark-record\.md$'                            # 91

# G3 — the premise check: no merge driver is configured anywhere.
# NOTE: written as an absence test. A bare `grep -c` prints 0 and EXITS 1 here,
# and rc 1 on a passing gate is how a reader mistakes this for a failure.
! grep -q 'merge=union' .gitattributes                                   # rc 0

# G4 — the generator named in the header does not exist.
test ! -e scripts/roll-benchmark-record.py                               # rc 0

# G5 — O1 arm 1: is the deferral still justified?
# Writes to the file as a share of the last 200 non-merge commits, against the
# 15% threshold derived in O1. Prints the share and exits 1 when the trigger
# fires, so it can be read as a gate rather than as a number to interpret.
git log --no-merges -200 --name-only --format=%H 9fb40279d \
  | grep -c '^\.agents/benchmark-record\.md$' \
  | awk '{ pct = 100 * $1 / 200;
           printf "O1 arm 1: %d/200 = %.1f%% (trigger >15%%) -> %s\n",
                  $1, pct, (pct > 15 ? "FIRES: option 2 is now the work"
                                     : "not fired: option 4 still holds");
           exit (pct > 15) }'
# at 9fb40279d: "O1 arm 1: 2/200 = 1.0% (trigger >15%) -> not fired", rc 0
```

Drop `9fb40279d` from G5 to evaluate the trigger at any later base. That is the
intended re-measurement, and it is the one command §"Stop conditions" refers to.

`scripts/agent-preflight.sh` is the full gate for a records-only change and was
run on the head that carries this spec; its result is in §Evidence.

## Evidence

Measured on `origin/main` `9fb40279d79f327cfa5e64358b6aa2e822b782b5`,
2026-08-31, authoring host, in the worktree
`.claude/worktrees/agent-ab6244bdf23ad2bda`. No GPU was used, no lease was
taken, and nothing was built — the host had **7.0G free on `/` (99% used)** and
a load average of 53.17, which is why every check in §Gates reads committed
history instead of compiling anything.

**Review repairs, 2026-08-31, worktree `.claude/worktrees/agent-a4f275c689fef66e4`.**
The figures in §Method, §Option 2, §Option 4, §Owed, §Tests and the table below
were re-derived from scratch against the same pinned base `9fb40279d` and
corrected where they disagreed. No GPU, no lease, no build: the host had **110G
free on `/` (75% used)** and a load average of 34.74. Every measurement in this
spec now names the command that produces it, and G2 and G5 name the base
explicitly so that a re-measurement is comparable rather than merely different.

| Claim | Source |
|---|---|
| `merge=union` is absent from the tree | `.gitattributes`, 3 lines, at `9fb40279d` |
| the attribute and its subject were deleted together | `git show 7dc2ef1ea -- .gitattributes` |
| the shape is no longer admissible | `AGENTS.md` §Records, since `7dc2ef1ea` |
| the forge ignores merge drivers | #883, quoted in `7dc2ef1ea`'s body |
| 53 of 222 writes are non-append | classifier over `git show --unified=0`, §Method |
| the stated predicate and its first implementation differ on one commit | `68bc6ef49`, `@@ -15494,0` against a 15,495-line parent ending in a blank line |
| the newest write is non-append | `b426de5ac`, 2 hunks, +165 −6, parent 29,129 lines |
| union drops an entry body | scratch repo demo 1, `git merge` rc 0 |
| union keeps a retracted claim and its retraction | scratch repo demo 2, `git merge` rc 0 |
| the generator was deleted | `git log --diff-filter=AD -- scripts/roll-benchmark-record.py` → A `8a0744ae0`, D `1db7e59cf` |
| no gate requires or budgets a write | `scripts/check-pr-size.py:78-83,448`; budgets retired 2026-08-10 |
| the only instruction is prose | `.agents/benchmarking.md:221` |
| append-only is a convention, not a gate | `.agents/specs/dflash2-spec-decode.md:3033-3036` |
| 396 entries, 184 distinct heading prefixes, 128 of them singletons | `grep '^## ' \| awk '{print $1}' \| sort -u`, §Option 2 |
| 210 citing lines across 101 files | `git grep -I -c 'benchmark-record\.md' 9fb40279d`, excluding the file itself |
| the 252/117 first reported over-counts | same command with pattern `benchmark-record`: also matches `roll-benchmark-record.py` (6) and bare `benchmark-record` (37) |
| 35 citations name a line number | `grep -rnoE 'benchmark-record\.md:[0-9]+'` |
| `:532` no longer resolves to what cites it | `sed -n 532p`, vs `cuda_mamba2_ssd.cuh:50-58` |
| 201 internal links break on a move: 163 `.agents/`-relative, 38 `../`-prefixed | `grep -oE '\]\([^)h][^)]*\)'` over the file, then `grep -c '^\](\.\./'` |
| O1 arm 1 recalibrated to 15%, and it is reachable | G5 at `9fb40279d` (1.0%, rc 0) and at `11ccdcf76` (17.5%, rc 1) |
| all-time peak 200-commit window is 42.5% | sliding window over 3269 non-merge commits, 2026-08-04 .. 08-07 |
| >25% was last reachable 2026-08-09, >15% on 2026-08-23 | sweep of all 3070 200-commit windows: 211 clear 25%, 372 clear 15% |
| growth is decelerating: 687 lines/day life-of-file, 88/day since 2026-08-23 | line counts at `8a0744ae0`, `b426de5ac` and each day's last write |
| `retire-shared-record-surfaces.md:106` is false in both halves | this spec §Method (23.9%) and `git log -- .gitattributes` (no such attribute, ever) |
| this file was already measured not-implicated | `.agents/specs/retire-shared-record-surfaces.md`, §Scope and §baseline |
| the retired index wrote 115 of 200 commits | `7dc2ef1ea` body, measured at `e541be98` |

## Stop conditions

- **Stop and return `NEEDS_DECISION`** if the developer or operator wants the
  per-entry split (O1) executed now rather than filed. That is a defensible
  reading of §Records and this spec does not foreclose it; it is deferred on
  cost, not on principle, and the trigger in O1 is a number somebody chose.
- **Stop** before rewriting, reflowing, re-ordering or compacting any existing
  entry. The archive's content is evidence. R1 and R2 touch only the header.
- **Stop** before adding an append-only checker over the entry region. It would
  fail the 18 legitimate retractions this spec identifies.
- **Stop** if a re-measurement at a later base puts the write rate above O1's
  trigger. At that point option 4 has expired and O1 is the work.
