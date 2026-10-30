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
