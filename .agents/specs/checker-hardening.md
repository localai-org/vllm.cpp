# SPEC — `CHECKER-HARDENING` — the four detection gaps the campaign's reviews exposed

- Owning row: `CHECKER-HARDENING`
- Kind: policy/checker batch (four small, independent checker changes in one row — same class, same review economy)
- Issue: created in the same flow (row-scoped local issue)

## The four gaps (each found by a fresh review, none owned elsewhere)

1. **Evidence content vs pin sha256**: nothing compares a bench-evidence
   report's recorded artifact sha256 against the oracle pin file's claim
   (FLINT-CPU-PARITY review M3). Checker: when an oracle pin names an
   evidence dir, and that dir's README/report records an artifact sha256,
   the two must agree (or the report must not name a sha). Red-first:
   mutate the parity report's sha → checker fails naming both files.
2. **Registry multiplicity**: `check-oracle-pins.py` compares registry ids
   as SETS (`:559-585`), so a duplicated AGENTS.md registry row (or a
   duplicated pin block) is invisible (FLINT-CPU-PARITY review M4).
   Checker change: compare as multisets — a duplicated id fails. Red-first:
   duplicate the llama-cpp-flint row in AGENTS.md → fails.
3. **CI wiring coverage**: deleting a wired step from the agent-record job
   reds nothing (`check-test-registration` pins only its own step and three
   named suites) — the FIX-EVIDENCE-LOG-GATE review's M5. Checker change:
   `check-test-registration` (or a sibling) records the full expected
   checker-step list of the agent-record job and fails when a registered
   checker's step is missing. Red-first: delete the check-evidence-refs
   step → fails.
4. **Stderr-count gate for the kolibri1 NUMA fallback**: the fallback
   warning is enforced by count evidence only; a silent fallback corrupts
   nothing but loses the audit trail (FIX-TT-NUMA-DEGRADE review M1).
   Checker: a unit test that captures stderr while running a guard
   construction with a failing `set_mempolicy` (patched at the syscall
   wrapper in-process) and asserts exactly one warning per construction.
   Red-first: revert the warning → test fails.

## Scope discipline

Each item is an independent commit with its own red-first capture. The
checker-semantics contract applies per item: no gate is weakened; each
red-first is a real observed failure of the pre-change world. If any item
turns out to need a semantics change beyond the stated shape, split it into
its own row rather than growing this one.

## Known risk

Item 1's sha comparison must not false-positive on reports that record
shas of files other than the pinned artifact — the rule binds only when the
pin's evidence dir records exactly one artifact sha and the pin names an
artifact. Keep the trigger narrow; document it in the checker's docstring.

## Tests

Per item: unit tests in the house pattern + the red-first mutation captured
into docs/bench-evidence/checker-hardening-20261030/README.md (four
sections, one per item, each with the red output).

## Gates

check-agent-record, style, trailers, env-doc (if any env surfaces), preflight
(named host dotnet failure), and CI green — these checkers are themselves
wired, so check-test-registration must stay green after their registration.

## Git integration

One pull request, `row/CHECKER-HARDENING`, spec committed first, four
implementation commits.
