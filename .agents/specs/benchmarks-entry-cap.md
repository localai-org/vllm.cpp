# The BENCHMARKS cap is a file cap, and a file cap is a lock

Issue: [#460](https://github.com/mudler/vllm.cpp/issues/460).
Row: `ENG-RECORD-CONFLICT-SURFACES`.
Measured against `origin/main` `918c568a` on 2026-08-12.

`docs/BENCHMARKS.md` is gated by a 45,000-character budget on the whole file.
The page measures 44,795 characters, so 205 characters are free. Adding a
measurement row therefore means deleting somebody else's row, and the documented
way to delete one, moving it byte-for-byte into `.agents/benchmark-record.md`,
is broken for exactly the rows that carry evidence links.

This is the same defect this row already retired twice on 2026-08-11: the
`MAX_CHARS` budget in `scripts/check-now-current.py` and the `chars` key of
`STATUS_RATCHET` in `scripts/check-public-doc-tables.py`, both removed by
`87308dea` under #364 with the reasoning recorded in place. The scoreboard's own
`max_chars` was left standing in that pass. It is now the binding constraint on
every remaining roadmap measurement.

AGENTS.md, Records, states the rule this spec applies: **cap the entry, never
the file**, and **a gate is what usually creates the lock: if a checker requires
every change to touch a shared file, that is the defect**.

## Scope

**In scope.**

1. `max_chars` on `PageRules` in `scripts/check-public-doc-tables.py`, which
   applies to `docs/BENCHMARKS.md` (45,000) and `docs/FEATURES.md` (30,000).
2. The rules that take over its obligation: a per-row character cap, a regrowth
   guard on per-attempt headings at any depth, and, added for review finding F1,
   a `_prose_paragraphs` that counts list items and blockquote lines so the
   append-only wall the byte cap was also catching stays caught. The first two
   are entry-scoped; the third is a count, and Risks says so plainly.
3. `check_links` in `scripts/check-agent-record.py`, so the archive path the
   scoreboard points at works for a row that carries a relative link.
4. The mutation suites `tests/scripts/test_check_public_doc_tables.py` and
   `tests/scripts/test_agent_record.py`.
5. The acceptance demonstration: the owed 35B canonical regrid row named by
   #481 lands with nothing evicted. **Implemented as a test that adds the row to
   the real page and drops it again, not as an edit to the page.** #481 is open
   and rewrites the 35B row in place, so writing a second copy of the same fact
   would duplicate a keyed row the moment both merge. The surface's ability to
   accept the row is what this row owes; the row's content is #481's.

**Out of scope, deliberately.** The `STATUS_RATCHET` keys kept by #364; the
required-section, canonical-section, prose-paragraph, paragraph-length and
cell-length rules, all of which are kept and none of which are widened; the
content of any existing row; `scripts/roll-benchmark-record.py`'s move logic;
and rebuilding the public scoreboard as a derived or globbed surface, which is
argued against in Design and deferred in Work breakdown.

**Not a correctness change.** No product source, kernel, ABI or model path
moves. Nothing measured changes value.

## Upstream chain

**No vLLM counterpart. This is project infrastructure.** vLLM has no equivalent
of `docs/BENCHMARKS.md`, of `.agents/benchmark-record.md`, or of the checker
suite that gates them: they exist to serve this project's protocol, which vLLM
does not run. The authority for this change is AGENTS.md, Records, and the
precedent set by `87308dea` (#364) on the two sibling budgets.

## Our baseline

**The page has had no usable headroom for 25 commits.** Free characters against
the 45,000 cap at each of the last 25 commits that touched `docs/BENCHMARKS.md`,
with the page's table-row count:

| commit | rows | chars | free | subject |
|---|---:|---:|---:|---|
| `918c568a` | 162 | 44,795 | 205 | measure(SPEC-DSPARK) fibonacci gap |
| `523b8a6f` | 162 | 44,826 | 174 | measure(SPEC-DSPARK) 5-rep interleaved |
| `887e04ff` | 163 | 44,579 | 421 | **docs(release): compact benchmark projection (#475)** |
| `1c9dbe08` | 163 | 44,931 | 69 | perf(SPEC-DSPARK) sync-free Markov chain |
| `bbc482a2` | 163 | 44,942 | 58 | merge: Whisper encoder FA-2 |
| `c5615cfe` | 164 | 44,692 | 308 | perf(SPEC-DSPARK) speculative verify |
| `4112ac8c` | 165 | 44,968 | 32 | fix(mm-speed) review findings |
| `425abf7c` | 163 | 44,936 | 64 | bench(cpu) x86_64 floor |
| `93613baa` | 165 | 44,964 | 36 | **docs(benchmarks): trim the Voxtral encoder row back inside** |
| `04b2b9fa` | 165 | **45,007** | **-7** | **merge: origin/main into row/MM-SPEED-ENC-FA2** |

Three facts follow, and each of them is the thing AGENTS.md names.

**The success mode is unsafe.** `04b2b9fa` is a clean automatic merge that
landed the page at 45,007 characters, 7 over the cap. Two PRs each paid for
their row by evicting a different one; the three-way merge applied both
additions and neither eviction cancelled the other. That is verbatim the
corollary in Records: "merging two such edits cleanly is worse than
conflicting". It has already happened here, in the tree, not in theory.

**The eviction is real and it is winning.** Row count fell from 165 to 162 over
these 25 commits while the project gained measurements. Two commits,
`93613baa` and `887e04ff`, exist for no purpose but to pay rent: their subjects
are "trim the Voxtral encoder row back inside" and "compact benchmark
projection".

**The payment mechanism does not work for the rows that carry evidence.**
`check_links` (`scripts/check-agent-record.py:599-610`) runs `LINK_RE.findall`
over the raw file with no fenced-span stripping and resolves every hit from
`source.parent`. Three of the 162 rows carry a `docs/`-relative link
(`bench-evidence/qwen35-4b-sm120-main-20260807.md`,
`bench-evidence/rpi5-a76-q8-dot-20260806.md`,
`bench-evidence/rpi5-a76-llamacpp-20260806.md`), and none of them can be moved
into `.agents/benchmark-record.md` byte-for-byte: the target resolves from
`.agents/` and dangles. #433 hit this and had to archive a shorter link-free row
instead. The subset of payable rows shrinks every time one is spent.

**What the cap is actually still catching.** `_h2_headers` matches `## ` only,
so `### ` subsections are ungoverned by the canonical-section allowlist. The
live page carries six of them. An appended per-attempt `### ` section is
rejected today by nothing except the character budget.

**Corrected 2026-08-12, review finding F1: that is not the only obligation.**
`_prose_paragraphs` also excludes every line starting with `-`, `*`, `>`, `|` or
`#`, so a bulleted or blockquoted wall of forensics is outside the paragraph
count, `MAX_PARAGRAPH_CHARS`, `MAX_CELL_CHARS`, `MAX_ROW_CHARS` and the heading
guard at once. The character budget is the only thing that ever caught it, which
means the replacement has to pick up **two** obligations, not one: the appended
subsection, and the append-only wall of list and quote lines. Design §3a is the
second. The mutation table in Risks is the measurement.

## Port map

**No upstream file to port from.** The local anchors this change edits:

| Anchor | What changes |
|---|---|
| `scripts/check-public-doc-tables.py` | `max_chars` removed from `PageRules`; `MAX_ROW_CHARS` and `DATED_HEADING_RE` added; `page_errors` gains the two entry-scoped checks; `_prose_paragraphs` folds list items and blockquote lines in (`LIST_ITEM_RE`), so `max_prose_paragraphs` re-baselines to 21 on FEATURES and `STATUS_RATCHET["long_paragraphs"]` tightens 82 to 75 |
| `scripts/check-agent-record.py` | `check_links` gains `extract_links`, a pure fenced/inline-code-aware link scanner; `FENCE_RE` and `strip_code_spans` implement CommonMark's closing-fence rule; `link_base` becomes `link_bases` |
| `tests/scripts/test_check_public_doc_tables.py` | three `max_chars` tests replaced by entry-cap, regrowth and no-eviction tests |
| `tests/scripts/test_agent_record.py` | new `LinkExtraction` cases |
| `docs/BENCHMARKS.md` | **unchanged.** The owed row is added and dropped inside `test_the_shipped_page_can_accept_the_next_measurement_row`, so the surface is proven without writing a fact #481 already owns |

## Design

**Remove the file cap. Relocate its obligation to two entry-scoped rules.**

*1. `max_chars` is deleted from `PageRules`, with the reason recorded in place.*
A byte budget on a shared page is a lock by construction: every addition is a
read-modify-write of one global, the conflict is the lucky outcome, and the
clean merge is the unsafe one. `87308dea` removed the two sibling budgets on
exactly this argument; this is the third, left behind in that pass.

*2. `MAX_ROW_CHARS = 600` caps one table row.* This is the literal "cap the
entry": a measurement's cost is bounded locally, by the author of that
measurement, and never by deleting a row somebody else owns. It is the same
shape as `MAX_ENTRY_CHARS` in `check-now-current.py`. 600 is set from the live
pages: the longest row on `docs/BENCHMARKS.md` is 520 characters and on
`docs/FEATURES.md` 580. It is a genuine constraint, and a tighter one than what
it joins: `MAX_CELL_CHARS = 220` alone permits a five-column row of 1,100
characters.

*3. `DATED_HEADING_RE` rejects a per-attempt heading at any depth.* This is the
regrowth guard, the same shape as the `ROW_TABLE_LINE` guard `87308dea` added to
`check-now-current.py`, and it closes the `### ` hole the character cap was
covering. The shape is measured, not invented: of the **310 sections already
rolled into `.agents/benchmark-record.md`, 287 name a date in their heading**
(`2026-08-07`, `2026-07-31`, ...), which is what a per-attempt entry looks like
here. **Zero of the 36 live headings across the two public pages name a date**
(18 each, at every depth, `_headings`). So the guard fires on the first appended
DATED checkpoint section, at `##` or `###`, and never on a subject section.

**What this guard is, and is not, tighter than.** It is strictly tighter than
the canonical-section allowlist it joins, which runs over `_h2_headers` and so
sees `## ` only: a DATED `### ` subsection was previously rejected by nothing
but the character budget. It is **not** tighter than the retired byte cap in
general, and the first revision of this spec, of the checker comment and of the
PR body all said that it was. That claim is false and was proven false by
mutation. An **undated** appended subsection passes this guard by construction,
because the guard fires on a *date*. See §3a, which is the rule that closes
that.

*3a. `_prose_paragraphs` folds list items and blockquote lines in.* Added
2026-08-12 in response to review finding F1, which is the finding that showed
§3 alone does not discharge the stop condition. `_prose_paragraphs` excluded
every line starting with `-`, `*`, `>`, `|` or `#`, so a bulleted or quoted wall
was counted by **nothing**: not the paragraph count, not `MAX_PARAGRAPH_CHARS`,
not `MAX_CELL_CHARS`, not `MAX_ROW_CHARS`, and not the heading guard. Two
mutations measured it against the real checker with the entry cap in place:

| mutant | page size | verdict before §3a | verdict after |
|---|---:|---|---|
| 3,000 appended bullet lines (the exact mutant `test_oversized_page_fails` used, which BASE rejects as `113833 chars, over the 45000-char scoreboard budget`) | 113,833 | **SURVIVED**, `[]` | CAUGHT, `prose paragraph of 194014 chars exceeds 700` |
| 500 appended UNDATED `### Attempt N` sections with bulleted forensics | 117,222 | **SURVIVED**, exit 0, 68/68 green | CAUGHT, `535 prose paragraphs, over the 35 budget` |
| 3,000 appended blockquote lines | 127,718 | **SURVIVED**, `[]` | CAUGHT, `prose paragraph of 83014 chars exceeds 700` |

A contiguous run of list items folds into ONE paragraph rather than N, so a
legitimate short list costs one paragraph and a wall trips
`MAX_PARAGRAPH_CHARS`, while a run per appended section trips the paragraph
COUNT. `docs/BENCHMARKS.md` carries no list item and is unmoved at 35;
`docs/FEATURES.md` carries one and moves 20 to 21, so `max_prose_paragraphs` is
**re-baselined to 21**, which is a re-measurement under a larger counted
population and not slack. `docs/STATUS.md` shares the function and moves the
other way, 82 long paragraphs to 75, because its 29 list items now join
neighbouring paragraphs instead of splitting them; `STATUS_RATCHET` follows the
measurement **down** to 75 in the same change, since banking the 7 units would
be exactly the slack the ratchet exists to refuse.

*4. `check_links` stops validating text that is not a link, and the archive
resolves what it archived.* Two changes, both narrow:

- Fenced code blocks and inline code spans are stripped before `LINK_RE`. A
  target inside a fence is not a link: CommonMark renders it as literal text, no
  reader can follow it, and there is nothing for the checker's rule ("every link
  resolves") to be about. Today the checker forbids any document in the tree
  from *showing* a link in sample output, which is why #460's own reproduction
  is a fence.
- **The pairing rule is CommonMark's**, added 2026-08-12 for review finding F2.
  A closing fence must use the opener's character, be at least as long, and
  carry nothing but whitespace after the marker; a line with an info string
  opens a block and never closes one. The first revision closed on any line
  matching `^\s*(```+|~~~+)`, which does not fail safe: it **inverts fence phase**
  for the rest of the file. With the one unbalanced fence this tree already has,
  the bare ` ``` ` at
  `.agents/completed/state-events/0000-00/STATE-LEGACY-000001.md:17697` was
  "closed" by the ` ```sh ` at `:17948`, and from there ordinary prose was
  blanked, including a live reader-followable spec link at `:18297`. Under the
  corrected rule that link is validated again and three lines of genuinely
  fenced sample are not.
- `.agents/benchmark-record.md` resolves a target from `docs/` as well as from
  `.agents/`. It is the declared archive of `docs/BENCHMARKS.md`, so content
  moved into it verbatim was written against `docs/`. `link_base` already
  carries one such special case for migrated legacy payloads; this generalises
  it to `link_bases`, a tuple, and a target still has to exist under one of
  them. Fence-stripping alone does not cover this case, because
  `roll-benchmark-record.py` moves sections as live markdown, not fenced.

**Why not (a) alone, the `check_links` fix.** It unblocks payment and leaves the
ratchet. Every measurement would still evict a row somebody else owns, and the
clean merge of two such payments would still land the page over budget, as
`04b2b9fa` did. It treats the symptom the issue was filed from and not the
defect the issue describes.

**Why not (b), per-row files or a derived page.** The three admissible shapes in
Records govern *record* surfaces. `docs/BENCHMARKS.md` is not one: AGENTS.md,
Public documents, defines it as a projection whose purpose is to be one readable
page a user reaches from the README badge. "Derived at read time, so nobody
writes it" removes the lock only when the rendered artifact is not committed,
and GitHub renders committed markdown with no build step, so the generated page
would still be committed and still be a file every measurement PR writes. The
lock would move from the author to the generator, not die. It is also not what
is blocking: the binding pain today is the *cap*, not conflicts. Deferred as W4
with the condition that would justify it.

**Why not simply raise the cap.** Prohibited by the task, and the checker's own
comment already records why it does not work: the previous occupant of this line
"answered it by adding slack to the constant, which only postponed it to the
next cadence of parallel work".

## Tests to port

**No upstream tests exist.** These are written against this project's checkers.

`tests/scripts/test_check_public_doc_tables.py`:

- `test_the_shipped_page_can_accept_the_next_measurement_row`: **the acceptance
  test.** It adds the owed 35B canonical regrid row to the REAL page,
  immediately below the last row of the Open gaps table so markdown renders it
  inside that table, and asserts every pre-existing row survives. The row is
  added and dropped inside the test, so the page is not edited and #481 is not
  collided with. It does NOT assert the grown page exceeds the retired budget:
  review finding F3 showed that assertion stored a measurement of
  `docs/BENCHMARKS.md` inside a test file, with 173 characters of margin, so any
  PR compacting the page by more than that turned it red in a file it does not
  own. `887e04ff` shrank the page by 280 and `93613baa` by 43. The
  above-the-budget point belongs to the synthetic test below and is made there.
- `test_a_new_row_costs_no_eviction`: the same property on a synthetic page
  sized past the retired 45,000 budget. RED on BASE, which reports
  `48836 chars, over the 45000-char scoreboard budget`.
- `test_oversized_row_fails`: a single row past `MAX_ROW_CHARS` is rejected.
  RED on BASE, which has no row cap.
- `test_the_row_cap_is_not_subsumed_by_the_cell_cap`: a row of legal cells whose
  sum is illegal is rejected, so the entry cap adds a rule `MAX_CELL_CHARS` does
  not already carry.
- `test_a_row_at_the_cap_is_allowed`: the boundary is inclusive.
- `test_a_dated_h2_is_rejected`, `test_a_dated_h3_is_rejected`, and
  `test_a_dated_h3_is_rejected_on_the_feature_matrix_too`: an appended
  per-attempt section fails at the first one, at either depth, on either page.
  RED on BASE for `###`, which no rule covered.
- `test_a_new_subject_subsection_is_allowed` and `test_a_dated_row_is_still_allowed`:
  the guard is not a section freeze and does not reach into rows.
- `test_a_date_inside_a_fence_is_not_a_heading`: sample output is not a section.
- `test_the_shipped_pages_carry_no_dated_heading`: the two live pages satisfy
  the new guard as shipped.
- `test_no_page_carries_a_whole_file_size_budget`: the invariant behind this
  row, held as a rule rather than as a habit.
- `test_a_wall_of_BULLETS_fails`, `test_a_wall_of_BLOCKQUOTES_fails` and
  `test_appended_UNDATED_subsections_with_bullets_fail`: mutants M8, M8b and M7
  from Risks, each RED on the first revision of this row and on nothing before
  it except the byte cap.
- `test_a_short_bullet_list_is_still_allowed` and
  `test_a_bullet_run_is_ONE_paragraph_not_many`: the fold is a budget, not a ban,
  and a run costs one paragraph rather than N, which is what keeps the count
  budget off ordinary documents.
- `test_an_EMPHASIS_lead_wall_is_a_KNOWN_residue`: characterisation of M11. It
  goes RED the day W8 lands, which is the point.
- `test_the_shipped_pages_sit_at_their_folded_paragraph_budget`: the two
  paragraph budgets are pinned to what the pages measure under the fold, so
  re-baselining FEATURES from 20 to 21 is a re-measurement and cannot become
  slack.
- Replaced: `test_oversized_page_fails`,
  `test_release_projection_fits_after_current_main_merge`'s `max_chars`
  assertion, `test_the_two_pages_have_distinct_budgets`'s `max_chars`
  comparison. Each is replaced by an assertion on the rule that took the
  obligation over, never deleted outright.

`tests/scripts/test_agent_record.py`, new `LinkExtraction`:

- `test_fenced_link_is_not_extracted`, `test_tilde_fenced_link_is_not_extracted`
  and `test_inline_code_link_is_not_extracted`: RED on BASE, which extracts all
  three.
- `test_live_link_is_still_extracted`, `test_a_backticked_label_is_still_a_link`,
  `test_link_after_a_closed_fence_is_still_extracted` and
  `test_link_beside_an_inline_span_is_still_extracted`: the narrowing does not
  swallow real links, including the `` [`name`](derived-issue-index.md) `` form this tree uses
  everywhere.
- `test_stripping_preserves_line_and_column_positions`: spans are blanked, not
  deleted, so every line and column offset survives. Review finding F6: this is
  NOT evidence of anything today, because `check_links` reports no line numbers
  at all. It is held so that a caller that does report positions cannot be
  broken by this function, and the docstring now says so.
- `test_a_fence_with_an_INFO_STRING_does_not_close_a_block`,
  `test_a_closing_fence_must_match_the_opener` and
  `test_a_LONGER_closing_fence_does_close`: CommonMark's closing-fence rule, both
  directions (F2).
- `test_prose_two_lines_below_a_closed_fence_is_still_scanned`: the live case, on
  the real file that mis-paired. RED on the first revision of this row.
- `test_a_link_straddled_by_two_INLINE_SPANS_is_not_extracted`: two stray
  backticks hide a target, matching the renderer, and the ordinary
  `` [`name`](derived-issue-index.md) `` form is still a link (F5).
- `test_an_archived_row_with_a_docs_relative_link_is_accepted`: the #460
  reproduction, moved into the record as live markdown, resolves. RED on BASE
  with `dangling link bench-evidence/rpi5-a76-q8-dot-20260806.md`.
- `test_an_archived_row_with_a_MISSING_link_still_dangles`: the second base is a
  base, not an amnesty.
- `test_the_benchmark_record_also_resolves_from_docs` and the two `link_bases`
  cases in `MigratedLegacyLinks`: every other file keeps single-base resolution.
- `test_the_tree_has_no_dangling_link`: the whole-tree run stays green.

## Gates

1. `python3 -m pytest tests/scripts/ -q`, unbounded, full run.
2. `python3 scripts/check-public-doc-tables.py`.
3. `python3 scripts/check-agent-record.py`.
4. `scripts/agent-preflight.sh --staged`.
5. `python3 scripts/check-pr-size.py` red-before/green-after harness on both
   changed checkers: it reruns each HEAD test module against the BASE checker in
   an isolated worktree and refuses the PR unless BASE goes red.
6. Acceptance: the live `docs/BENCHMARKS.md` plus the owed row is valid and
   carries one more row and no fewer, with the row placed directly under the
   last Open gaps row so it renders in that table. Measured on the merged tree:
   **162 rows to 163, errors `[]`**, with every pre-existing row asserted still
   present. The test no longer compares the page against the retired budget: see
   Tests to port, and review finding F3.
7. The mutation set, run against three trees. Table in Risks.

**No GPU. Nothing here measures.**

## Dependencies

- #364 / `87308dea`, which set the precedent and removed the two sibling
  budgets, is on `main`.
- #481 is open and rewrites the 35B row on `docs/BENCHMARKS.md` in place. This
  row does not edit the page at all, so the two cannot conflict, and #481 keeps
  ownership of the "regrid owed" fact. That is why the acceptance demonstration
  is a test rather than an edit.
- `origin/main` moved from `918c568a` to `e1087a88` mid-row (12 commits,
  GATE-PIN-UNPINNED-SNAPSHOTS #471 and four SPEC-DSPARK measurements #442).
  Merged, and every gate rerun on the merged tree. The measurements in Our
  baseline are as taken at `918c568a` and are not restated.
- #481 also blocks W8 / #507, for the same reason it shapes the acceptance
  demonstration: closing the emphasis-lead residue owes an edit to four
  paragraphs on a page #481 holds open.
- Nothing else blocks.

## Work breakdown

| W | Work | State |
|---|---|---|
| W1 | This spec, committed alone | this commit |
| W2 | `check-public-doc-tables.py`: retire `max_chars`, add `MAX_ROW_CHARS` and `DATED_HEADING_RE`, with tests | in this PR |
| W3 | `check-agent-record.py`: fenced/inline-code-aware `extract_links`, `link_bases` for the archive, with tests | in this PR |
| W4 | Rebuild `docs/BENCHMARKS.md` as a derived index over per-row files | **DEFERRED.** Justified only if the page becomes a *conflict* hotspot after the cap is gone. Trigger: `git merge-tree` shows it conflicting in 3 or more concurrently open PRs, measured, as #364 measured its three surfaces. Not justified by the cap, which W2 removes. |
| W5 | Teach `roll-benchmark-record.py` to record the archived section's origin explicitly, rather than relying on W3's two-base resolution | **DEFERRED.** W3 makes the move work; W5 would make it self-describing. No blocker depends on it. |
| W6 | Make `_h2_headers` fence-aware, so the gate and the rollup agree on what a section is ([#495](https://github.com/mudler/vllm.cpp/issues/495)) | **DEFERRED, filed not fixed.** Found doing W2 and reproduced: `_h2_headers` is a bare `startswith("## ")` scan while `split_sections` tracks fences, so a heading-shaped line inside a fence is a section to the gate and not to the script the gate tells you to run. It changes what an existing gate counts, so it takes its own spec and red-before rather than riding along here. Neither page has a fenced heading today. W2's `_headings` is already fence-aware and is the natural basis for the repair. |
| W7 | Retire `MAX_README_CHARS` the same way ([#498](https://github.com/mudler/vllm.cpp/issues/498)) | **DEFERRED, filed not fixed.** `README.md` measures 29,965 of 30,000: **35 characters free**, tighter than any of the three budgets already retired. 13 of the last 20 commits touching it sat under 60 free, `031410e8` landed it 52 OVER, and `row/DOCS-README-BUDGET` (#161) is a whole merged row whose purpose was paying rent. Same defect, third checker (`check-readme-structure.py`), so it needs its own spec and its own mutation in `tests/scripts/test_check_readme_structure.py` rather than riding along here. |
| W8 | Close the EMPHASIS-lead paragraph residue ([#507](https://github.com/mudler/vllm.cpp/issues/507)) | **DEFERRED, filed not fixed.** §3a folds list items and blockquote lines into `_prose_paragraphs`, but a line opening with emphasis still starts with `*` and is still excluded, so a `**bold**`-led prose wall is unbounded (mutant M11). Replacing the bare `*` prefix with `LIST_ITEM_RE` closes it and immediately turns FOUR paragraphs already shipped on `docs/BENCHMARKS.md` red against `MAX_PARAGRAPH_CHARS`, at 717, 719, 748 and 1,084 characters, so it owes an edit to a page #481 holds open and it changes what an existing gate counts. Own spec, own red-before. `test_an_EMPHASIS_lead_wall_is_a_KNOWN_residue` pins the residue and goes red the day it lands. |

## Risks/decisions

**Risk: removing a size gate lets the page bloat.** Answered by measurement, not
assertion, and the first answer was wrong. What bloated the page to 11,405 lines
was per-attempt sections, and after W2 a DATED one fails at either heading
depth, where before only `##` was covered. But a bulleted or blockquoted wall,
with or without undated `###` headings above it, was caught by nothing at all
until §3a folded those lines into `_prose_paragraphs`. The full mutation set,
run against BASE `origin/main`, against the first revision of this row, and
against the revision that lands (`scratchpad/mutate.py`, reproduced in the PR
body):

| mutant | BASE | first revision | landed |
|---|---|---|---|
| M1 dated `##` appended | CAUGHT | CAUGHT | CAUGHT |
| M1b dated `###` appended | CAUGHT | CAUGHT | CAUGHT |
| M2 one row over 600 | CAUGHT (by size) | CAUGHT | CAUGHT |
| M3 legal cells, illegal row | CAUGHT (by size) | CAUGHT | CAUGHT |
| M4 required section dropped | CAUGHT | CAUGHT | CAUGHT |
| M5 em-dash | CAUGHT | CAUGHT | CAUGHT |
| M6 200 legal rows, 599 chars each | CAUGHT (by size) | SURVIVED | **SURVIVED, intended** |
| M7 500 undated `###` + bullets | CAUGHT (by size) | **SURVIVED** | CAUGHT |
| M8 3,000 bullet lines | CAUGHT (by size) | **SURVIVED** | CAUGHT |
| M8b 3,000 blockquote lines | CAUGHT (by size) | **SURVIVED** | CAUGHT |
| M9 live post-fence link redirected to a missing target | CAUGHT | **SURVIVED** | CAUGHT |
| M10 archived row, missing `docs/`-relative link | CAUGHT | CAUGHT | CAUGHT |
| M11 500 `**bold**`-lead paragraphs | CAUGHT (by size) | SURVIVED | **SURVIVED, known residue** |

Two mutants the byte cap caught are not caught after this row, and both are
stated rather than glossed. **M6 is the point of the row**: rows are the growth
mode of a keyed table, each is capped at 600, and no author pays for one by
deleting another's. **M11 is a defect**, filed as
[#507](https://github.com/mudler/vllm.cpp/issues/507) and deferred as W8: a line
opening with emphasis starts with `*` and is still excluded, so a `**bold**`-led
wall is unbounded. It is deferred, not dismissed, because closing it turns four
paragraphs already shipped on `docs/BENCHMARKS.md` red against
`MAX_PARAGRAPH_CHARS` (717, 719, 748 and 1,084 characters) and so owes an edit
to a page #481 holds open. `test_an_EMPHASIS_lead_wall_is_a_KNOWN_residue` pins
it and goes red the day it is closed.

**Risk: `DATED_HEADING_RE` fires on a legitimate heading.** A pinned-date
subject would trip it, for example "vLLM 0.26.0 as of 2026-08-12". Measured
against both live pages: zero of 36 headings match, 18 per page at every depth.
If one is ever wanted, the date belongs in the row or the prose, which are
unaffected. The error message says so.

**Risk: the paragraph budget is itself a whole-page count.** It is, and both
pages now sit exactly on it: `docs/BENCHMARKS.md` at 35 of 35 and
`docs/FEATURES.md` at 21 of 21. So this row removes the file lock on ROWS and
leaves one on PROSE. That is deliberate and is the same line #364 drew when it
deleted `chars` from `STATUS_RATCHET` and kept `long_paragraphs`: rows are how a
keyed table grows and prose is how it decays, so the count is a quality gate on
the decay mode rather than rent on the growth mode. It is still a real cost to
an author adding a paragraph, and it is now recorded in the `PageRules`
docstring so the next one meets it there instead of in CI.

**Risk: `MAX_ROW_CHARS = 600` is tuned to current content.** It is, and that is
sound for an entry cap in a way it is not for a file cap: an author who needs a
longer row shortens their own row, and never anyone else's. The FEATURES margin
is thin (580 of 600). Accepted; the alternative, no row cap, leaves
`MAX_CELL_CHARS` permitting a 1,100-character row.

**Risk: skipping fenced links narrows a checker.** It narrows it to what the
rule was always about. A fenced target is not a link under CommonMark, so no
reader can follow it and no rendering can dangle. Real links, including one on
the same line after a closed inline span, stay checked, and
`test_the_tree_has_no_dangling_link` holds the whole tree.

Measured at merge `fdbc8ae6`, over the markdown files the checker scans:
**4,170 targets before the strip, 4,163 after**, so 7 stop being validated, and
every one of the 7 is a code sample. Three are inside the unclosed block at
`.agents/completed/state-events/0000-00/STATE-LEGACY-000001.md:17697`
(`porting.md`, `../tests/vt/test_ops_matmul_elem.cpp`,
`specs/accelerator-seam-audit.md`); the other four are the `path` placeholder in
this spec's own samples. **The loose fence rule the first revision shipped keeps
4,162 on this same tree, one FEWER than the CommonMark rule, and the one it
loses is not a sample**: it is the live prose link at `:18297`, lost to the
phase inversion F2 describes. The corrected pairing restores it.

These totals move every time the tree gains a markdown file, so they are dated
to a SHA rather than treated as constants. Reproduce with `extract_links` and
`LINK_RE.findall` over `markdown_files()`; the RATIO that matters, and the thing
that does not move, is that the corrected rule validates strictly more than the
loose one and loses only samples.

The spec's earlier figures, 4,109 before and 4,105 after with "all 4 are quoted
samples", were both stale and wrong about the count: the review re-measured 5
lost, and the fifth was the live link.

**Risk: two stray backticks hide a target that does not exist.** They do. This
form, which this spec itself carried until F5 found it:

```text
`[`name`](derived-issue-index.md)`
```

is four backticks, so CommonMark reads it as a code span holding an open
bracket, the literal text name, and a second code span holding the rest, and
there is no link for the checker to validate. The checker agrees with the
renderer, which is the rule, but the consequence is that wrapping a dangling
link in backticks hides it. Pinned by
`test_a_link_straddled_by_two_INLINE_SPANS_is_not_extracted`, which asserts the
same answer for the double-backtick form an author should use instead, and
asserts that the ordinary `` [`name`](derived-issue-index.md) `` form is still a link.

**Risk: an unbalanced fence blanks the rest of a file.** It does, and one file
in the tree has one: `.agents/specs/laguna-s21-scope-2026-07-30.md` ends on a
stray closing fence at its last line. It costs nothing today, because there is
no content after it, and the same behaviour already exists in
`_prose_paragraphs`, `_table_rows` and `split_sections`. Making an unbalanced
fence an error is a separate rule with a separate red-before, not part of this
change.

**Risk: two-base resolution in the archive hides a genuine dangling link.** The
second base applies to one file, `.agents/benchmark-record.md`, the declared
archive of `docs/BENCHMARKS.md`, and the target must still exist under one of
the two bases. Every other file keeps single-base resolution.

**Decision: `docs/FEATURES.md` loses its `max_chars` too.** The constant lives on
the shared `PageRules`, the argument is identical, FEATURES is at 29,740 of
30,000 with 260 characters free, and leaving it would leave a known lock armed
on the page that grows every time a feature ships. It gains the same two
entry-scoped rules.

## Evidence

- Char and row history: reproduce with `git show <rev>:docs/BENCHMARKS.md | wc -c`
  over `git log --format=%h -25 -- docs/BENCHMARKS.md`. Table in Our baseline.
- `04b2b9fa` over the cap: `git show 04b2b9fa:docs/BENCHMARKS.md | wc -c` gives
  45,007.
- Heading-shape survey, re-measured on the merged tree after review finding F8
  found the first numbers unreproducible: at merge `fdbc8ae6`, **287 of 310**
  archived section titles carry a date, and **0 of 36** live headings do, 18 per
  page. The archive total climbs as rows land, so it is dated. Reproduce with
  `_headings` and `DATED_HEADING_RE` over the two pages, and with
  `grep -c '^## '` over `.agents/benchmark-record.md`. The spec previously said
  278 of 301 and 0 of 32; 32 was `_h2_headers`-shaped, not `_headings`-shaped.
- #460's reproduction, red before and green after W3.
- The full mutation set, three trees: BASE `origin/main`, the first revision of
  this row, and the revision that lands. Table in Risks; harness reproduced in
  the PR body.
- Link-extraction census at merge `fdbc8ae6`: 4,170 raw targets, 4,163 after
  the strip, the 7 losses enumerated in Risks, against 4,162 for the loose fence
  rule, whose extra loss is the live link.
- `check-pr-size.py`'s own red-before/green-after harness, for both checkers.
- **CI on this PR: a red `windows-msvc-*` is NOT this row's.** Both lanes are
  `if: github.event_name == 'pull_request'` (`ci.yml:640`), so the lane
  `scripts/main-baseline.py` reads never runs them, and it reports main GREEN
  while they fail on every PR that reaches them. `main` does not compile under
  MSVC: `tests/vt/test_cpu_isa_x86.cpp` lacks `<ostream>`, and MSVC's
  `<string_view>` does not supply it transitively. Reproduced byte-for-byte on
  `row/ENG-RELEASE-WINDOWS` @`673c2f3d` and here @`104d3f36`: same file, same
  `__msvc_string_view.hpp(550,23) error C2027`, same target, same failing step.
  This PR touches no `src/`, `include/`, `tests/`, `cmake/` or `.ps1` path, so
  it cannot be the cause. Filed as
  [#503](https://github.com/mudler/vllm.cpp/issues/503).

## Stop conditions

- **Stop if** removing `max_chars` cannot be shown to leave the append-log class
  caught. The regrowth guard is the condition of the removal, not a nicety.
  **This condition failed on the first revision and was met on the second.**
  M7 and M8, both append-log-shaped, survived the dated-heading guard because it
  fires on a *date*; §3a is what discharges the condition, and the three-tree
  mutation table in Risks is the evidence. The residual survivor M11 is the same
  class and is NOT claimed as caught: it is filed as #507, deferred as W8, and
  pinned by a test.
- **Stop if** the entry cap or the heading guard fires on either live page as
  shipped. That would mean the replacement is a different rule, not a relocated
  one, and the page would owe an edit this row is not authorised to make.
- **Stop and return `NEEDS_DECISION`** if closing #460 turns out to require
  rewriting an archived link, which would break the byte-for-byte guarantee the
  archive exists to give.
- **Never** raise a cap to pass, and never delete an assertion to turn a gate
  green.

## Outcome

Pending. Filled on `DONE` with the measured before/after, what was rejected, and
why each constant is set where it is.
