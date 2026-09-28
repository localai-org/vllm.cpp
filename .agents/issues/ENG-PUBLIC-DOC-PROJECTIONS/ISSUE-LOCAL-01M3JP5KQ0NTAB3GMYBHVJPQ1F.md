ID: ISSUE-LOCAL-01M3JP5KQ0NTAB3GMYBHVJPQ1F
Title: nine registered architectures have no docs/FEATURES.md row: four rows were dropped in a merge and five were never written
Row: ENG-PUBLIC-DOC-PROJECTIONS
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-27
Updated: 2026-09-27
Closed: -

## Problem

check-supported-models is red on main with one error, and it is the checker's stated purpose rather than a checker defect: nine architectures the C++ registry resolves have no row in the docs/FEATURES.md supported-architecture table.

BoundaryExtractor (gliner2_registry.cpp), ClmModel (clm_registry.cpp), CuaS1Forms (cua_s1_registry.cpp), KevModel (kev_registry.cpp), LayaModel (laya_registry.cpp), MiMoV2ForCausalLM (mimo_v2_registry.cpp), SpanExtractor (gliner25_decide_registry.cpp), Tev1Model (tev1_registry.cpp), XorModel (xor_registry.cpp). Every one is a live REGISTER_VLLM_MODEL call, so ModelRegistry::SupportedArchs() returns it, so the engine can serve it, so the public "supported models" list understates the tree by nine.

Four of the nine (BoundaryExtractor, CuaS1Forms, KevModel, LayaModel) HAD rows and lost them: docs/FEATURES.md at 4f11dfc10 carries all four with their checkpoint and gate text, and the table at 1de097c46 ends at CohereForCausalLM. The rows were dropped somewhere between those two revisions in the tmp-merge series (c12b376b2, a42660ce0, 92cf8cf29 all touch this file), which also lost the ParakeetForRNNT and ParakeetForTDT keys and moved DeepseekV41ForCausalLM. A merge dropped a table tail and the gate was the only thing that noticed.

The other five never had one, and each needs a row written from its own record rather than from a neighbour's shape, because they differ in kind: four are POOLING decision models on /v1/systemone or /v1/score (Clm, Kev, Laya, CuaS1Forms) plus two GLiNER2.5 archs, one is an autoregressive text model served by the ordinary chat path (Tev1Model), and MiMoV2ForCausalLM is REGISTERED AND LOADABLE BUT ITS FORWARD IS A STUB that VT_CHECK(false, ...) -- so its row must say that, not imply a run.

Two honesty constraints on the rows, both from the tree rather than from taste:

1. Five of the nine specs have a ## Now section that contradicts the code in the same tree. specs/tev1.md says "Phases 1-5 pending implementation" and specs/mimov2.md says "No mimo_v2 code exists in the tree. grep -ri mimo src/ include/ returns nothing" -- while tev1_registry.cpp, mimo_v2_registry.cpp, mimo_v2_weights.cpp and tests/vllm/models/test_mimov2_w{1,2}.cpp are all present. specs/clm.md says "not yet implemented" against clm_registry.cpp. Writing rows that cite those specs as the authority would restate a false claim on a public page, so each row cites the code and the gate that actually ran.

2. Tev1Model's 12 cases assert a prompt builder that lives in the TEST FILE (BuildUserContent at tests/vllm/models/test_tev1.cpp:27), not in shipped code -- grep for it under src/ and include/ returns nothing. The row records that, because "12 tests pass" read without it says the prompt construction is gated and it is not.

This change adds the nine rows and restores the four that were dropped. It does not touch the checker, whose contract ("fix the list or register the arch") is correct.

## Resolution

-
