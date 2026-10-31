# SPEC — `TT-SMALL-FIXES` — two named small defects from the anchoring and evidence campaigns

- Owning row: `TT-SMALL-FIXES`
- Kind: bug (two independent test/tooling fixes; one branch because both are tiny and already fully diagnosed)
- Issue: created in the same flow (row-scoped local issue)

## Fix 1 — the B2BI/B2II env mismatch (silent skip wearing a pass)

`tests/vllm/models/test_kolibri1_tt_b2bi.cpp:1021` reads
`VT_KOLIBRI1_TT_B2BI_MODEL` while every doc, spec, and operator instruction
sets `VT_KOLIBRI1_TT_B2II_MODEL` — the device leg silently SKIPS while the
suite reads "10/10 SUCCESS" (TT-ANCHORING evidence; this is how the 0.29 s
phantom-green ctest runs happened, and how a reviewer's own serving pass
nearly repeated the class).

Fix: the test reads BOTH spellings — `B2II` first (the documented one),
falling back to `B2BI` (backward compatibility for any operator script) —
plus a one-line comment citing TT-ANCHORING. Red-first: with only `B2II`
set and a real checkpoint path, HEAD silently skips; after the fix the leg
RUNS (or refuses by name if the path is bogus — never silently skips).

## Fix 2 — `agent-preflight.sh --staged` crashes with a Python SyntaxError

Reported by the kolibri1-GGUF implementer (2026-10-30): the `--staged`
branch crashes with a Python SyntaxError around `scripts/agent-preflight.sh:86`
(inline python in the bash script; the non-`--staged` run is clean).
Diagnose, fix, and add a regression case to the preflight's own test suite
(`tests/scripts/test_agent_preflight*`) that runs `--staged` in a scratch
repo and asserts exit != SyntaxError. Red-first: reproduce the crash at
HEAD, capture it.

## Gates

check-agent-record, style, trailers, preflight (named host dotnet failure).
Fix 1's device leg needs the TT env + the kolibri1 artifact dir — the gate
binary initializes the backend; run under the standing recipe (mutex, reset,
envs) or verify host-side per the test's own skip semantics.

## Git integration

One pull request, `row/TT-SMALL-FIXES`, spec committed first, two commits
(one per fix).
