ID: ISSUE-LOCAL-01M66B4KT6QFENVDGN6G99M8CM
Title: Kolibri-1 GGUF (mudler APEX tiers) refused: no kolibri1 arch row, and CheckQuantConfig refuses k-quants
Row: KOLIBRI1-GGUF-ARM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-30
Updated: 2026-10-30
Closed: -

## Problem

mudler/Kolibri-1-APEX-GGUF @ 55a4781439 (nine tiers, 22-56 GB, standard ggml types: Q2_K/Q4_K/Q5_K/Q3_K/IQ1_M/IQ2_XS, header verified by range-request parse) is refused by our tree at two verified points: the GGUF loader dispatch (model_loader.cpp:1221) has no kolibri1 arch row, and kolibri1_weights.cpp CheckQuantConfig (:64-80) refuses any non-fp8 checkpoint. The developer asked for the artifact to be downloaded and made to work. Spec: .agents/specs/kolibri1-gguf-arm.md. Deliverable: arch row + kolibri1_gguf_weights.cpp TU mapping the 903 tensors into the existing CPU kolibri1 forward (dequant at load; all types known to gguf_dequant.cpp), serving in-band vs the fp8 arm (primary oracle), throughput recorded; llama-cpp-kolibri1 floor pin (b10451 predates PR #30673/#30865) named as follow-up. I-Micro downloading to /mnt/models/kolibri1-apex-gguf/.

## Resolution

-

## Findings (2026-10-30, implementer)

- RECONCILED: both named blockers are stale against this branch — the arch
  row landed at `src/vllm/entrypoints/model_loader.cpp:1277` and the GGUF
  TU bypasses the fp8 checks by design (commits `71ee347e9` + `0350841b7`,
  2026-10-23, spec `kolibri1-gguf-arch.md`). The remaining deliverable was
  gating the mudler APEX artifact, which needed NO product-code change.
- The APEX I-Micro file (sha256 `3edfaf09…c90e0e` verified) LOADS; its 903
  tensors map onto the existing structs with zero unmapped/leftover
  (evidence `docs/bench-evidence/kolibri1-gguf-arm-20261030/`).
- STOP HIT (spec §Stop conditions): the fp8-vs-GGUF token cross-check is
  BEYOND the W3 band — 37/44 golden greedy positions match, worst topk logit
  diff 3.245 > 2.5, 2 hard flips (Hob-forge Q2_K adjudicated earlier: 5/13,
  3.97, 3 hard). Fluency probe rules out a load defect (prompt-appropriate
  looping, the bf16 goldens' own shape). Numerics adjudication — whether the
  band is re-measured for quantized arms tier-by-tier or the tier is
  accepted at the recorded envelope — is the developer's call, exactly as
  the kolibri1-gguf-arch row left it. Throughput and public-doc weight pins
  are NOT landed.
