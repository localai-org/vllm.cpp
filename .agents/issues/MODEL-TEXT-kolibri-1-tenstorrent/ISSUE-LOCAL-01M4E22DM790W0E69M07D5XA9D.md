ID: ISSUE-LOCAL-01M4E22DM790W0E69M07D5XA9D
Title: Kolibri-1 TT B2b-i completion: dense-resident device forward — one greedy decode on the P150
Row: MODEL-TEXT-kolibri-1-tenstorrent
State: OPEN
Kind: enhancement
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-08
Updated: 2026-10-08
Closed: -

## Problem

The B2b-i bring-up slice landed 2026-10-08 (merged as cf6258c76 + b52c0baeb): the TT build links against the pin source via the fresh /tmp/pin-build lib64, the resident non-expert slice (3,311,163,520 B = 3.084 GiB, 753 tensors) stages on the P150 with byte-exact readback of all 1103 operands, and one device op (embedding bit-exact + layer-0 q_proj GEMM within its stated envelope) is verified against the CPU row. What remains is the B2b-i completion condition named in .agents/specs/kolibri-tt.md ### B2 scope — B2b addendum: the dense-resident device forward running entirely on device with the routed-expert tier absent, completing one greedy decode of a golden prompt on the card. Concretely (addendum lines 360-382): (1) attention — the hybrid geometry as staged, 40 sliding-window layers at window 513 and 10 full-attention layers with full RNoPE (no rope tables on the full group), two-group KV per the wave-A design, per-head q/k RMS norms inside the attention path; (2) the four sandwich norms (input_layernorm, post_attn_norm, post_attention_layernorm, post_ffn_norm) per the landed CPU row (.agents/specs/kolibri-1-cpu.md); (3) the router — bf16 [384,2560] gate, f32 e_score_correction_bias, sigmoid-logit-add scoring, top-6-of-384, used in this slice only for the shared expert plus a nameable unimplemented-routed-expert refusal; (4) embed + untied lm_head staged bf16; (5) on-device sampling through the landed decode seam (ModelRegistry::Forward, dense_attn::AttnBlock) where the TT backend provides it; no streaming, no slot pool. Gates: device-free host-side tests first (red-first, manifest-driven style in tests/vllm/models/), then the token gate vs the CPU golden chains (W3 methodology: 141/145 argmax positions, the 4 known flips adjudicated inside the 2.5-nat band, 0 hard flips allowed — any flip outside the band is a gate failure), then the production bench anchor (only after the token gate passes). No device measurement is B2b evidence until the greedy decode completes. Stop conditions per the addendum lines 444-456: pin tree cannot compile the B2b device TU -> stop, record, row stays ACTIVE; stale _ttnncpp.so link blocker (fresh /tmp/pin-build lib64 is the verified workaround, already in place); no card window -> row stays ACTIVE with implementation owed.

## Resolution

-
