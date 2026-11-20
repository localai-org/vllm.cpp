# Spec — the issue index is a table nothing measured

Issue: [#1033](https://github.com/mudler/vllm.cpp/issues/1033)
Row: `GATE-ISSUE-INDEX-TABLE-SHAPE` (unplaced record/gate defect; the index is a
record surface, not a matrix row)
State: `ACTIVE`

## Scope

`scripts/check-agent-record.py::check_table_shapes` counts the
unescaped pipes on every table line of every path it is handed and reports any
line whose count differs from the first line of that table:

```python
pipes = len(re.findall(r"(?<!\\)\|", line))
```

That is exactly the measurement a malformed index row needs. Its call site
(`:1527-1530`) hands it `roadmap_v1.md`, `coordination.md`, `*MATRIX_PATHS` and
the spec paths of every `READY`/`ACTIVE` row. **`.agents/issue-index.md` is not
in that list**, and no other checker in the tree counts its cells.

So the index — the one record surface every change must write, the one file
whose rows are prose long enough to hide a stray pipe in a code span — has been
the only markdown table in the record set with no shape gate at all.

In scope:

1. `.agents/issue-index.md` joins the paths `check_table_shapes` is called with.
2. The one row that change reds is repaired.
3. Red-before cases in `tests/scripts/test_agent_record.py`.

Out of scope: any change to `check_table_shapes` itself. The function is
correct; only its argument list was short. Changing it would widen a gate to
make a red go green, which is the one move
[`AGENTS.md`](../../CLAUDE.md) § *Changing the rules or a checker* forbids.

Also out of scope, and for a different reason: **making the checker report all
findings instead of stopping at the first**. #1033 attributes the two-day
concealment partly to that shape. It measured false. `main()` threads one
`errors` list through every check and prints all of it after the last one; the
`if not errors:` guard above the block covers only the missing-canonical-record
case, where continuing would raise on a file that is not there. Measured in
`## Evidence` rather than read off the source, because reading is how the wrong
premise got into the issue. There is nothing here to contain, so nothing is
changed.

## Upstream anchors

None. This is a repository checker, not a ported behavior. vLLM has no
equivalent surface.

## Design

The call site gains one element:

```python
        check_table_shapes(
            [
                AGENTS / "roadmap_v1.md",
                AGENTS / "coordination.md",
                ISSUE_INDEX,
                *MATRIX_PATHS,
                *spec_paths,
            ],
            errors,
        )
```

`ISSUE_INDEX` is already a module constant (`:1322`), defined for
`check_issue_index`, so the path has exactly one spelling in the file and the
two gates over the same file cannot drift onto different paths.

### What the gate then measures

The index is one table: a header (`| Issue | Row | Title | Kind |`, five
pipes), a separator, and one row per issue. `check_table_shapes` takes the
header's count as expected and reports every row that disagrees. A row that lost
its trailing pipe reads four; a row carrying an unescaped pipe inside a code
span reads six or more. Both render wrong on GitHub, and both were invisible to
every gate before this.

### The one row this reds

On `origin/main` at `100026481` a pipe histogram over the index's 289 table
lines reads `{5: 288, 9: 1}`. The single outlier is line 279, the `#1003`
`ORACLE-LLAMACPP-REPIN-STOCK` row, which arrived with `283c7e492` (#1051) and
carries **four** unescaped pipes inside code spans, at columns 2705, 3106, 3115
and 3338 — one in a regex alternation over the word `match`, two delimiting a
quoted table cell, and one in a character-class alternation. The repair escapes
each of the four and changes nothing else.

They are **not** the spans the dispatch named, which placed all four inside a
single `git diff … | grep …` code span. This row carries `git diff` spans and
`git grep` spans, and none of them is piped into anything. Same row, same count,
different spans. The four were located by re-running the checker's own regex
over the line rather than by searching for the quoted text, which is the only
reason the discrepancy is visible at all.

GFM replaces an escaped pipe with a literal pipe inside a table cell before
inline parsing, so each code span then renders exactly as its author wrote it.
Today they do not: the cell splits at each of the four instead.

## Risks

**This edits a row of an append-only file.** `check-issue-index-append-only.py`
forbids that and is RED on this branch. The exception is argued in the commit
body and the pull request body rather than in a registry, per
[`AGENTS.md`](../../CLAUDE.md) § *Changing the rules or a checker*, and it is
the same argument `ff264cb82` (#1025) made for the duplicate `#995` repair:

- The append-only contract **cannot** repair a malformed row. Appending a
  corrected copy leaves the broken one in place and adds a duplicate key, which
  makes `check-agent-record` angrier. The file only becomes well-formed by
  editing the row where it sits. The two gates are in genuine contradiction
  here, and that contradiction is the defect this row exists to remove.
- The rule is preserved in substance and verified mechanically, not by eye: the
  repaired row is byte-identical to the original once the four added backslashes
  are removed again, every other row is byte-identical and in the same order,
  the row count is unchanged, and no key is added or lost.
- That gate is preflight-only. `.github/workflows/ci.yml:121` runs the record
  gate and no job runs the append-only gate, so the exception costs no CI red,
  and the violation is confined to this branch: once merged, a later branch
  diffs a `main` whose row is already repaired and sees no removal in its range.

**The union merge driver.** `.gitattributes` gives the index `merge=union`, and
GitHub does not run that driver, so the file goes CONFLICTING on the pull
request the moment `main` appends a row. The remedy is mechanical and is stated
in the pull request body: merge `origin/main` locally, discard the auto-merged
file, re-apply the four escapes and re-append this row, then assert the prefix
property by hand.

**A future long row is still easy to get wrong.** This gate reports the defect;
it does not prevent it. That is the right division of labour: the author now
sees a named line number in their own preflight instead of a mis-rendered table
two days later.

## Tests

`tests/scripts/test_agent_record.py` gains three cases, all against the shipped
file, so none of them can pass on a fixture the tree does not have:

- `test_check_table_shapes_covers_the_issue_index` — captures the paths
  `main()` actually hands `check_table_shapes` and asserts `ISSUE_INDEX` is
  among them. **This is the red-before case:** it fails on `origin/main`,
  where the path is absent. It captures the real call rather than reading the
  source, so a rename or a re-spelling of the path cannot make it vacuous.
- `test_the_shipped_issue_index_is_a_well_formed_table` — runs
  `check_table_shapes` on the real `.agents/issue-index.md` and requires zero
  errors. Red on `origin/main` for the `#1003` row, green after the repair. This
  is the case that would have fired in #1051's own preflight.
- `test_a_malformed_index_row_is_caught` — the mutation. Copies the shipped
  index into a temporary file, strips the trailing pipe from its last row, and
  requires an error naming that line and its pipe count. Without it the two
  cases above prove only that a list contains a path and that a file is
  currently clean; this one proves the instrument fires.

The third case is what answers "a mutation that never applied reads as a passing
test": it asserts on error text that cannot exist unless the mutated line was
both written and read back.

## Gates

`python3 scripts/check-agent-record.py`, run bare with its exit code echoed.

- RED-before: path added, `#1003` unrepaired — exit 1, naming
  `.agents/issue-index.md:279: table has 9 pipes; expected 5`.
- GREEN-after: same command, exit 0.
- `python3 tests/scripts/test_agent_record.py` red before the call-site change,
  green after.
- `scripts/agent-preflight.sh` — green except for the failures accounted for
  below.

## Evidence

Captured verbatim with exit codes in the pull request body. No gate command
whose status is needed is piped: `cmd | head` reports `head`'s status, which is
how this tree has produced false green verdicts before.

### The checker does not stop at the first finding

Measured, not read. A scratch index carrying three defects at once — a duplicate
key, a row that lost its trailing pipe, and the unrepaired `#1003` row — reports
all three in one run of `python3 scripts/check-agent-record.py`:

```
ERROR: .agents/issue-index.md: issue #168 listed twice. Under `merge=union` a duplicate is what two branches appending the same issue look like
ERROR: .agents/issue-index.md:279: table has 9 pipes; expected 5
ERROR: .agents/issue-index.md:307: table has 3 pipes; expected 5
RC=1
```

The two synthetic rows were appended to a working copy and reverted with
`git checkout --` immediately after, and `check_issue_index` reported the
duplicate on the row it was given rather than the number of the real issue.

So the concealment #1033 describes was real and its cause was not ordering: it
was solely that `check_table_shapes` never saw this path. One reporting run,
three findings, exit 1 once.

Accounted-for failures:

- `issue-index append-only` in preflight — argued above.
- `windows-msvc-cpu` and `windows-msvc-vulkan` in CI — red on every pull
  request ([#584](https://github.com/mudler/vllm.cpp/issues/584),
  [#968](https://github.com/mudler/vllm.cpp/issues/968)) with no `main`
  baseline. Not this change: the diff carries no C++.

## Stop conditions

- Stop if repairing the `#1003` row changes any character other than the four
  added backslashes. The row is a record, and its meaning is not this row's to
  edit.
- Stop and report `NEEDS_DECISION` if adding the path reds a row other than
  `#1003`. That would mean a second, unmeasured defect landed while this was in
  flight, and it needs its own issue rather than a silent repair here.
- Stop if the only way to reach green is to change `check_table_shapes`.

## Now

`ACTIVE`. The call site, the row repair and the three test cases are one pull
request; the spec commits first.
