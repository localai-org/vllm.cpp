# CHECKER-HARDENING-2 — the red-first captures of the checker changes

Row: `CHECKER-HARDENING-2`
Spec: `.agents/specs/checker-hardening-2.md`
Date: 2026-10-31. Host: the shared dev checkout; every capture is offline,
standard library only, no GPU, no network.

Each section records the observed failure of the PRE-change world (or, for a
new checker, the failure the checker produces on the mutated tree), verbatim,
before the fix that the same commit carries.

## 1. Documented artifact sha256 vs disk (`check-doc-sha-pins`)

The checker was green on the staged probe before the mutation: a scratch
artifact `docs/model-00001-of-00032.safetensors` (content
`kolibri1 bf16 shard probe artifact`) plus a probe line in the Kolibri-1
section of `docs/USAGE.md` recording its true sha256 — rc 0, both restored
byte-for-byte afterwards. Mutation: the probe line's sha256 replaced with 64
zeroes, checker run against the mutated tree:

```text
$ python3 scripts/check-doc-sha-pins.py; echo rc=$?
check-doc-sha-pins: documented sha256 pins disagree with disk:
  docs/USAGE.md:917: the recorded sha256 does not match model-00001-of-00032.safetensors on disk: the document records 0000000000000000000000000000000000000000000000000000000000000000, the file hashes to bb88d80e3c3c06c2d1c13e29e7ba6d64a30dc3bf41a6d1698ae363e8fca1ee74
Re-measure the artifact and record the SAME sha256 in the document, or re-publish whichever side recorded the stale hash.
rc=1
```

Green after restore: `check-doc-sha-pins: ok` at rc=0, self-test
`7 doc-sha-pin fixtures` ok, unit suite `tests/scripts/test_check_doc_sha_pins.py`
OK (11 tests).

## 2. Secondary pin values bound (`check-oracle-pins`) — BLOCKED, the tree falsifies the trigger

Not implemented. The stated shape — a `gateable = yes` pin block whose
evidence file records exactly ONE distinct hex sha must match the block's
`pin` — fails on the UNTOUCHED tree, before any mutation, on
`llama-cpp-kolibri1.md`: its evidence
(`docs/bench-evidence/kolibri1-llamacpp-crosscheck-20261020.md`) records
exactly one full 40-hex revision, `62318cd57dc63900ecb153272c0b10f78550855d`,
which is the PRIOR llama.cpp pin named on purpose by the crosscheck report,
while the block's `pin` is `9df0b73c707900b7277a46b5ae52ff7e7e567d13`
(`LLAMA-CPP-PIN-ADVANCE`, 2026-10-31). Measured over every `gateable = yes`
record whose evidence resolves to a file (distinct 40-hex tokens, full-sha
boundaries so 64-hex artifact shas never match):

```text
llama-cpp-kolibri1.md pin= 9df0b73c7079 ev= docs/bench-evidence/kolibri1-llamacpp-crosscheck-20261020.md s40= 1 <<< WOULD FIRE
   ev40: {'62318cd57dc63900ecb153272c0b10f78550855d'}
```

This is exactly the false positive the spec's known-risk paragraph forbids:
the one-distinct-sha trigger cannot tell an evidence report that binds the pin
from one that documents the pin it replaced. The 64-hex reading of the trigger
does not help either — llama.cpp pins are 40-hex commit revisions, and the
64-hex value the llama-cpp advance evidence records (`3edfaf09...`) is an
artifact hash, not the pin. Resolving this needs a decision the spec does not
carry (advance the kolibri1 evidence, or scope the rule to evidence written by
the pin's own advance report), so per the scope-discipline section it is
reported, not grown. Red-first for the intended defect (revert the llama-cpp
pin to `b10451`) was NOT captured: the checker cannot be landed green, so
there is no honest red-before/green-after pair.

## 3. Oracle-file pin-sha multiplicity (`check-oracle-pins`) — BLOCKED, the tree falsifies the premise

Not implemented. The stated shape — two oracle pin blocks carrying the same
`pin` sha must fail — fires on the UNTOUCHED tree, before any mutation:

```text
dup pins: {'9df0b73c707900b7277a46b5ae52ff7e7e567d13': 2}
  .agents/oracles/llama-cpp.md:167 (pin block)
  .agents/oracles/llama-cpp-kolibri1.md:126 (pin block)
```

The duplication is deliberate and documented in
`.agents/oracles/llama-cpp-kolibri1.md`: under `LLAMA-CPP-PIN-ADVANCE`
(2026-10-31) the stock `llama-cpp` pin advanced onto the SAME release object
(`b11765`) that the `llama-cpp-kolibri1` scope already pinned, because both
scopes gate against the identical llama.cpp build. Two records at one object
is therefore ordinary work in this tree, and a gate that fails on ordinary
work is the defect, not the discipline. The M4 finding's premise (same-sha
oracle records indicate a forgotten advance) is falsified by the tree; any
narrower rule (same sha AND same upstream, for instance) also loses its
red-first, because `llama-cpp-flint` shares that upstream too. Reported, not
grown; the intended red-first (copy the flint pin sha into a second oracle
file) was NOT captured for the same reason as item 2.

## 4. Inline `run:` steps escape the AGENT_RECORD_CHECKER_STEPS pin (`check-test-registration`)

Mutation: an unpinned INLINE checker step added to the `agent-record` job of
`.github/workflows/ci.yml` (single-line
`run: python3 scripts/check-doc-sha-pins-probe.py`), checker run against the
mutated tree:

```text
$ python3 scripts/check-test-registration.py; echo rc=$?
ERROR: check-doc-sha-pins-probe.py runs in the agent-record job but is not in AGENT_RECORD_CHECKER_STEPS -- record it in the pin or move the step
rc=1
```

Green after restore (the step removed; item 1's real `check-doc-sha-pins.py`
is the only addition this row records, and it is recorded in the pin and the
CI step): rc=0. The literal-block per-line shell-continuation scanning is
untouched; `tests/scripts/test_check_test_registration.py` is green before
and after, four fixtures richer (unrecorded inline fails, pinned inline
passes, the collector sees an inline step, a conditional inline step is not a
registration).
