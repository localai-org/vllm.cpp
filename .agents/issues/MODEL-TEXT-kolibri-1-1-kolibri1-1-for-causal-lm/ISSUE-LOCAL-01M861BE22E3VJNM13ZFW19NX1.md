ID: ISSUE-LOCAL-01M861BE22E3VJNM13ZFW19NX1
Title: Kolibri-1.1 U2+U4: dense prefix and shared-KV layer plumbing
Row: MODEL-TEXT-kolibri-1-1-kolibri1-1-for-causal-lm
State: CLOSED
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-24
Updated: 2026-11-24
Closed: 2026-11-24

## Problem

Wave W2 of the accepted Kolibri-1.1 assessment combines U2 (sliding layers onto the shared-KV cache spec) and U4 (shared-KV plumbing). Kolibri-1.1 adds mlp_only_layers=[0,1] with mlp_only_intermediate_size=3584 (dense FFN prefix), index_layers=[4,14,19,29], kv_source_layers with per-layer KV reuse, and layer_types containing indexed_attention — which our Kolibri1Params currently refuses as neither sliding_attention nor full_attention, while IsSlidingLayer would silently classify it as full. Implement the dense-prefix and layer-kind/KV-source plumbing with hermetic tests over the four LFS-verified GGUF arms from U1; keep the six marker-KV guard set aligned with burn-vllm PR #8. No token parity claim: no gateable CPU oracle exists for kolibri1_1.

## Resolution

Landed in wave W2 (U2+U4) on branch row/MODEL-TEXT-kolibri-1-1-U2 (spec 4baefceb6, implementation commit pending). Kolibri1Params now carries index_layers, kv_source_layers, mlp_only_layers and mlp_only_intermediate_size plus IsIndexedLayer/IsDenseLayer/KvSourceLayer; the parser accepts indexed_attention and validates all lengths/ranges by name; tensor enumeration accepts the dense FFN prefix on layers 0-1 and the missing K/V tensors on the six shared-KV layers; MakeKolibri1KVCache builds one group per KV source with a byte-neutral 1.0 fast path; the forward refuses the unimplemented sparse path by name. Focused gate 6 cases / 88 assertions green; related ctest 3/3. Three mutations are red with byte-for-byte restoration (indexed kind, dense prefix, KV source). The unrelated 1.0 Q2_K token gate failure was proven byte-identical on unmodified base 92ea7d3a8, so it is pre-existing. Evidence in docs/bench-evidence/kolibri11-u2-20261124/. No token parity claimed: no gateable CPU oracle exists.
