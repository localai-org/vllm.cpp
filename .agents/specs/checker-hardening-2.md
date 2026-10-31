# SPEC — `CHECKER-HARDENING-2` — the second batch of review-exposed checker gaps

- Owning row: `CHECKER-HARDENING-2`
- Kind: policy/checker batch (four small, independent checker changes; the follow-up to CHECKER-HARDENING's F1/F2 and two newer review findings)
- Issue: created in the same flow (row-scoped local issue)

## The four gaps (each found by a fresh review, none owned elsewhere)

1. **Documentation sha256 pins vs disk**: no checker validates a sha256
   recorded in a documentation artifact (USAGE.md artifact pins,
   bench-evidence reports) against the file on disk (FLINT-KQ-CPU-PARITY
   review M3; kolibri1-GGUF review M2 confirmed it for USAGE.md). Checker:
   scan docs/USAGE.md + docs/bench-evidence/**/ for recorded 64-hex
   sha256s colisted with an existing file path; verify each against the
   file's content (buffered streaming hash). Fail naming both. Narrow
   trigger documented (only sha+path adjacencies, not bare hex).
2. **Secondary oracle pin values are review-bound**: reverting the
   llama-cpp pin block to b10451 is undetected — the pin checker binds
   only the primary vllm authority's parity spans
   (LLAMA-CPP-PIN-ADVANCE review M1; `PIN_SURFACES` at
   `check-oracle-pins.py:~744`). Checker change: extend
   `check-oracle-pins.py` so a pin block whose `gateable = yes` carries
   an `evidence` path that records the pinned sha (same narrow
   evidence-sha trigger as item 1) must match; secondary pins whose
   evidence does NOT record a sha stay evidence-path-checked (no false
   positives). Red-first: revert llama-cpp to b10451 → if the advanced
   evidence records 9df0b73c, the checker must fail.
3. **Registry multiplicity — pin-side inverse**: CHECKER-HARDENING item 2
   covered duplicated registry rows and duplicated pin ids; the
   LLAMA-CPP-PIN-ADVANCE review's M4 found the registry multiplicity
   path already covered but the ORACLE-FILE side (two oracle files
   claiming the same pin sha) un-checked. Checker change: two oracle
   pin blocks carrying the same `pin` sha → fail. Red-first: copy the
   flint pin sha into a second oracle file → fails.
4. **Wiring-coverage F1 — inline `run:` steps escape the
   AGENT_RECORD_CHECKER_STEPS pin** (CHECKER-HARDENING review F1;
   `check-test-registration.py:1035` reads only literal-block `run: |`
   steps). Checker change: collect inline single-line `run:` steps too.
   Red-first: add an inline-step checker to the agent-record job without
   recording it → fails.

## Scope discipline

One commit per item, each with its own red-first capture into
docs/bench-evidence/checker-hardening-2-20261031/README.md. Any item
needing semantics beyond the stated shape splits into its own row.
House rules apply: a checker's message defines what it enforces; no gate
weakened; self-test corpora extended where the checker has one.

## Known risk

Item 1's disk-hashing cost: USAGE.md + evidence dirs are bounded; hash
only files the adjacency trigger names. Item 2's trigger must not
false-positive on pins whose evidence legitimately records a DIFFERENT
sha (e.g. an evidence dir documenting multiple artifacts) — the one-sha
narrow trigger from check-evidence-pin-sha.py is the precedent to reuse
(literally: reuse its extraction helper if importable).

## Gates

check-agent-record, style, trailers, preflight (named host dotnet
failure), check-oracle-pins + self-test, check-test-registration, and the
new suites. Each new/changed checker wired per the registration contract
(exact-equality list — record additions in the pin list).

## Git integration

One pull request, `row/CHECKER-HARDENING-2`, spec committed first, four
implementation commits.
