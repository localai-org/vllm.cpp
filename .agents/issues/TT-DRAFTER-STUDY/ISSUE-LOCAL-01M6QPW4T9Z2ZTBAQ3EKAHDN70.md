ID: ISSUE-LOCAL-01M6QPW4T9Z2ZTBAQ3EKAHDN70
Title: Study the DFlash2 on-card speculative-decode runtime as a design reference for our models
Row: TT-DRAFTER-STUDY
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-06
Updated: 2026-11-06
Closed: -

## Problem

The developer's reframe: the dflash2 bundle's advantage is ~500x and most of it is the DRAFTER and other shared code we will need for other models they do not cover as well. The bundle is Apache-2.0 end to end (verified: the bundle, the kernel SPDX headers, the base model Qwen/Qwen3.8-27B, the drafter incoai/Qwen3.8-27B-DFlash2 @ dedf8df68). Deliverable: a written study note (docs/design/) answering five questions: the DFlash2 mechanism (draft/verify/acceptance on-card, host touches each step once), the on-card decode loop structure (96-97% under host load), the 1x4 mesh sharding, portability to OUR models (kolibri1/flint), and the adoption sketch. File:line citations into the bundle's sources (fetched to scratch, NOT committed); no product code; the adoption row files after the study. Spec: .agents/specs/tt-drafter-study.md

## Resolution

-
