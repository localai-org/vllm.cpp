ID: ISSUE-LOCAL-01M8026Y8N21TP6H4H47RQHPRH
Title: NEW-CANDIDATE-PORTS W2c-1 (olala): host ops for the geodesic norm and the latent MoE, unit-gated vs the author's reference formulas
Row: NEW-CANDIDATE-PORTS
State: OPEN
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-21
Updated: 2026-11-21
Closed: -

## Problem

W2a/W2b landed (oracles pinned, registry + bf16 GGUF load + hybrid KV spec, forward refuses by name). W2c-1 ports the two simplest ops as host ops with unit tests whose expected values are transcribed from the author's modeling_olala.py @ f781f124 (cited file:line; the full transformers run stays owed — olala-transformers oracle is gateable=no): (1) OlalaGeodesicNorm (modeling_olala.py:1596-1617): sphere projection out = x*cos(theta) + unit_tangent*R*sin(theta) with theta = clamp(clamp(||grad||/||x||, pi/4)*scale + bias)/(layer_idx+1), pi/4); geo_* are [1] F32 scalars. (2) OlalaMoE (modeling_olala.py:1472-1552): router logits = x@f_gate^T (F32) + exp_probs_b; sigmoid; top-k(6); probs normalized * expert_weights_scale(2.45); latent down ffn_latent_down [384,1536]; per-expert h = relu(x@up_e)^2 (act at :1487), y = h@down_e, weighted sum in f32 (TorchMoEMLP :1448-1469); latent up ffn_latent_up [1536,384]; shared MLP ffn_up_shexp/down_shexp [1536,1536] (OlalaMLP :1406-1414); out = shared + routed. Weights in the W2b OlalaWeights. Then W2c-2 = V differential attention, W2c-3 = Mamba3-MIMO. No torch on this host, so unit expected values come from numpy transcription of those formulas (tiny shapes), recorded as interim; the real oracle run is the owed gateable measurement.

## Resolution

-
