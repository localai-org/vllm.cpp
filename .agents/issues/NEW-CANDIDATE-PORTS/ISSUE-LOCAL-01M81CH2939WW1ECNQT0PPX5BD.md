ID: ISSUE-LOCAL-01M81CH2939WW1ECNQT0PPX5BD
Title: Inkling-2: record llama.cpp NVFP4 loop-scale hazard and the minimum oracle pin
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: -

## Problem

llama.cpp inkling2_loop had a loop-slot re-share bug: slots were copied from physical layers before the scale pass, so passes 2-3 through core layers 6-17 ran NVFP4 experts without per-expert scales (PPL ~130-168 vs ~7.65). Fixed in ggml-org/llama.cpp PR #32704, merged 2026-11-21 at 690b87d88151dbd775bb6eff5581663688785928; inkling2_loop reads began at b12600 (29b52d398). Our APEX Balanced artifact is a ggml quant (Q4_1/Q5_1) and is in the unaffected class. The W0 spec and docs/USAGE currently name no minimum llama.cpp pin for inkling2-based comparisons, so a future parity run could use an affected revision. Record the hazard, the >= 690b87d floor, and the unaffected artifact class. No inkling2 pin predating 690b87d exists in the W0 spec (verified: only unversioned ggml geometry references), so no pin advance is owed.

## Resolution

-
