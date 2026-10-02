# Spec — the frozen archive dropped 27 rows its own evidence records still cite

Row: `GATE-ISSUE-ARCHIVE-RESTORE` (unplaced record/gate defect; the completed
archive is a record surface, not a matrix row)
State: `ACTIVE`

## Scope

`.agents/completed/issue-index.md` is the frozen archive every
`### Frozen archive evidence` block quotes. It is not frozen in the way the
name claims:

1. Commit `2e84a073b` deleted the file wholesale — 913 lines — alongside its
   (legitimate) spec addition.
2. Commit `e3539d994` restored a hand-picked copy: "The last pre-retirement
   revision is restored at `.agents/completed/issue-index.md`, with its
   internal links re-pointed." The restore measured 886 content lines: 27
   archived rows came back missing, and one surviving line (406) was
   deliberately re-pointed.

The 27 dropped rows are not idle history. Exactly 43 records under
`.agents/issues/` cite them: 24 cite a line past the restored file's end, 19
cite a line that now holds a different row. No gate sees any of that today:
the frozen-evidence comparison runs only in the `_intake` branch of
`validate_issue_record`, whose 9 records all cite surviving lines, so all 43
stale quotes are row-owned records that validate untouched. But every one of
those 43 quotes is byte-true to the pre-deletion archive, so no edit to any
record can repair them against a truncated archive. The defect is in the
archive, not in the quoting.

Measured over all 831 records that carry a frozen-evidence block, against
the committed LF blob:

- before the restore: 350 byte-equal, 458 failing under this base's
  byte-only comparison (481 under the #3350 link-rebase comparison);
- after re-inserting the 27 lines at the positions `difflib` reports
  between `2e84a073b~1` and the restored file: the file is byte-identical
  to the pre-deletion archive except line 406, which keeps `e3539d994`'s own
  deliberate re-point; 373 byte-equal, and under #3350's comparison all 831
  resolve, 0 failing.

`scripts/check-agent-record.py` stays green with unchanged row counts
(ENGINE=179 MODEL=384 QUANT=87 KERNEL=60 BACKEND=90); none of the re-pointed
links in the restored lines dangle from `.agents/completed/`, so nothing
needed re-pointing beyond what `e3539d994` already did.

In scope:

1. Splice the 27 dropped rows back at their original positions.
2. Re-point any restored link that does not resolve from
   `.agents/completed/` (measured: none).
3. The red-before / green-after census in the commit message.

Out of scope, each for its own reason:

1. **Editing any record.** The 43 quotes are correct; "repairing" them would
   re-anchor records onto a truncated archive and make the falsification
   load-bearing.
2. **Changing any checker.** Widening or narrowing a gate to make a red go
   green is what AGENTS.md forbids; the archive, not the gate, is wrong.
3. **Enforcing the frozen-evidence comparison outside `_intake`.** That is
   the checker half of this pair and lands as its own stacked PR (#3350 is
   the comparison it needs; this restore is the data it needs).
4. **The 4 row-cell anomalies in the ROW bucket** — 3 records whose archived
   cell is the em-dash placeholder (`ISSUE-GH-83`, `ISSUE-GH-606`,
   `ISSUE-GH-408`) and 1 genuine cross-row citation (`ISSUE-GH-298` cites a
   `PERF-27B-LMHEAD-DSR` line while living under `PERF-27B-LMHEAD-FP4`).
   They are pre-existing record-content questions, orthogonal to line
   existence, and every one of them still resolves after the restore.

## Enforcement (the stacked checker PR)

With the rows restored and #3350's relative-link comparison landed, the
frozen-evidence contract is enforced wherever the block appears, in every
owner directory: a record that QUOTES an archived row must quote the line it
declares, modulo exactly the relative-link rebase the record's directory
forces, and the quoted line must carry the record's own GitHub number.
Absence of the block stays legal everywhere except `_intake`; presence is
not. Measured over the corpus on the restored archive: 831 records carry a
block, 831 resolve, 831 identify their own issue, 0 violations -- the
ratchet adds no new red. Working-copy EOL no longer changes the answer: the
comparison strips a trailing CR from the archived line, because the
committed blob is LF and a Windows checkout is not.
