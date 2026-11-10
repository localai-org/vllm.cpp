ID: ISSUE-LOCAL-01M738WQD57K9D3R2AQ2KCDHND
Title: flint.cpp reads a borrowed bf16 embedding through a misaligned uint16_t* (UBSan aborts test_flint_paged)
Row: MODEL-TEXT-FLINT
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-10
Updated: 2026-11-10
Closed: -

## Problem

FlintForward's embedding lookup (flint.cpp:540) reads emb[tok*h+d] where emb = Bf16Data(w.embed_tokens) is a const uint16_t* over a borrowed block payload whose base need not be 2-byte aligned. UBSan with -fno-sanitize-recover=all aborts test_flint_paged on sanitize-cpu: 'load of misaligned address ... which requires 2 byte alignment' (CI run 42817025725). Fix with the existing vt::LoadUnaligned seam at every uint16_t dereference site in flint.cpp (432, 449, 540, 595, 641, 752), preserving the CONVERT-not-memcpy bf16->f32 semantics byte-identically. RED evidence: docs/bench-evidence/flint-ubsan-embed-align-2026-11-10/red-ubsan.log.

## Resolution

-

## Resolution

2026-11-10, `row/FLINT-UBSAN-EMBED-ALIGN`. Fixed in
`src/vllm/model_executor/models/flint.cpp` by routing every bf16 scalar load
through the existing `vt::LoadUnaligned<uint16_t>` seam (sites at the
embedding lookup, `MatmulBf16`, `RmsNormBf16`, q/k head norm, sinks, MoE
router) — the same arm the FIX-UNALIGNED-LOADERS-772 fix used. The
CONVERT-not-memcpy semantics are untouched: `LoadUnaligned` returns the same
uint16_t bits and `Bf16()` converts them identically. RED evidence:
`docs/bench-evidence/flint-ubsan-embed-align-2026-11-10/red-ubsan.log`
(exactly one finding, flint.cpp:540:49). GREEN: `green-ubsan.log`
(test_flint_paged 208/208, zero runtime errors), `green-ubsan-forward.log`
(110/110). Attribution: the numeric value of every load is bit-identical by
construction; `test_flint_paged_engine` could not run locally because
`tests/vllm/models/flint_golden_q4_k_m.json` is not present on this host
(missing large fixture, recorded in `green-ubsan-engine-missing-golden.log`).

2026-11-10, `row/FLINT-UBSAN-EMBED-ALIGN`. Fresh-reviewer follow-up: the
bf16 lm_head loop in `FlintModel::Forward` (the `lrow[n]` scalar load, now
flint.cpp:853) was the one remaining Bf16Data site still dereferencing
`uint16_t` directly — the same borrowed-block-payload hazard class, masked
because the pre-fix UBSan run aborted at :540 before reaching it. Repaired
in the same flow through `vt::LoadUnaligned<uint16_t>`, mirroring the six
earlier sites; `Bf16()` conversion untouched. RED:
`red-lmhead.log` (new doctest probe in `test_flint_forward.cpp` builds a
borrowed lm_head at an ODD byte offset — the tied-expansion/mmap shape —
making it the first and only misaligned load; finding at
flint.cpp:852:66). GREEN under the same sanitize config
(`address,undefined`, `-fno-sanitize-recover=all`, `VT_POOL_BYPASS=1`):
`green-lmhead.log` (test_flint_forward 9 cases / 127 assertions, includes
the new probe) and `green-lmhead-paged.log` (test_flint_paged 208/208),
zero runtime errors. Grep over flint.cpp confirms zero remaining scalar
`uint16_t` subscripts over any Bf16Data pointer.
