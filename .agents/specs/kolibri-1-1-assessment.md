# Kolibri-1.1 — assessment declaration (`MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm`)

Row: `MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm`

This file declares the row and carries the assessment unit's record. The full
assessment (enumeration, delta vs kolibri1, oracle pins, effort estimate,
refusal verification) is
[docs/bench-evidence/kolibri11-assessment-20261123/ASSESSMENT.md](../../docs/bench-evidence/kolibri11-assessment-20261123/ASSESSMENT.md);
its GGUF walk is
[docs/bench-evidence/kolibri11-assessment-20261123/walk-q4km.txt](../../docs/bench-evidence/kolibri11-assessment-20261123/walk-q4km.txt).

## Scope of this unit

ASSESSMENT ONLY, landed 2026-11-23, release day:

- One GGUF walk of `Aleph-Alpha/Kolibri-1.1-GGUF` Q4_K_M (45.5 GB) and the KV
  diff against `Hob-forge-Kolibri-1-GGUF` Q4_K_M.
- Delta analysis vs our kolibri1 rows, from the llama.cpp PR #32891 source
  (open, head `5f4d222898dd`) and the HF config (`e9515166e22131ef`).
- Oracle landscape recorded in `.agents/oracles/aleph-alpha-inference.md`
  (v2.0.0 note, no pin advance — CUDA-only) and
  `.agents/oracles/llama-cpp-kolibri1.md` (PR #32891 unmerged note, no pin
  advance). Transformers carries NO kolibri1_1 at check time (v5.24.0 / main).
- Refusal verification: HF path refuses `Kolibri1_1ForCausalLM` by name
  (registry exact match); GGUF path is a TRAP — the 1.1 GGUF keeps
  `general.architecture = kolibri1`, so our GGUF arm would claim it. The
  refusal guard (reject a kolibri1 GGUF carrying the indexer KVs) is owed by
  the row's first wave.
- No product code, no checker, no forward implementation.

## Owed

- The kolibri1_1 port itself: waves U1–U6 as sized in ASSESSMENT.md.
- llama.cpp pin advance for kolibri1_1 once PR #32891 merges (re-states both
  `llama-cpp` and `llama-cpp-kolibri1` records).
- aleph-alpha-inference v2.0.0 pin advance: CUDA-lease gateability measurement.
- Transformers kolibri1_1: re-check before W3; no oracle exists at 2026-11-23.

## Stop conditions

Not applicable to this unit; it lands complete with the walk and the records.
