ID: ISSUE-LOCAL-01M3T6ZZ2WBM4FSMTQZAWM8974
Title: CLM engine does not match the Contrastive-LM/CLM reference, and scripts/convert-clm.py does not exist
Row: MODEL-CLM
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-09-30
Updated: 2026-09-30
Closed: -

## Problem

docs/models/clm.md and the ClmModel loader error name scripts/convert-clm.py, which is not in the tree or its history, so no published checkpoint can be loaded. Checked against Contrastive-LM/CLM @ bb42c6c and CLM_v0.1-8B.pt: (1) the loader reads head tensors .0/.2/.4/.6, the checkpoint names them inp/hidden.0/norms.0/out; (2) clm_head.cpp runs Linear, GELU, LayerNorm, Linear, GELU, Linear, the reference make_head runs Linear, GELU, Linear, LayerNorm, GELU, Linear; (3) the scale is exp(min(ls, 100)) = 100.81 for ls = 4.6132, the reference is min(exp(ls), 100) = 100.0, and a missing clm_logit_scale key silently falls back to 4.6; (4) ClmInference ignores the question instructions, the reference state text is state + blank line + instructions; (5) the head input is the raw last-token hidden state, the reference head sees it L2-normalized (vLLM pooling normalize plus Embedder.l2); (6) candidate texts use the shared kev OptionText ('key: description', 'no'/'yes'), the reference uses the description alone, 'true: Yes. This is true: <instructions>' for noul, and its own to_text rendering for object states; (7) the HTTP /v1/systemone path answers CLM through the kev builder (rounded, kev confidence, rl_agent field). The 13 recorded tests use synthetic weights built in the engine's own layout and recompute the confidence formula inline, so none of them could see any of this.

## Resolution

-
