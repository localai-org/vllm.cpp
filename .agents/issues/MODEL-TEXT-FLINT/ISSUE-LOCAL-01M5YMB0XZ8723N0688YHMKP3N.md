ID: ISSUE-LOCAL-01M5YMB0XZ8723N0688YHMKP3N
Title: cpu_paged_attn seeds the attention sink at exp(0) instead of exp(sink - m)
Row: MODEL-TEXT-FLINT
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-27
Updated: 2026-10-27
Closed: 2026-10-27

## Problem

PagedAttentionKernel (src/vt/cpu/cpu_paged_attn.cpp) seeds the softmax running max at the per-head sink and the denominator at exp(0)=1, in a TWO-PASS form that never rescales the seed when pass 1 raises m past the sink. Whenever any score exceeds the sink, the sink term silently becomes exp(m-m)=1 instead of exp(sink-m) — a larger term than the sink deserves — which shrinks every value weight. MiMoV2's large learned sinks always stay the max, so every existing gate passed; Flint's small per-head sinks expose it: attention output off ~6%, model logits off ~2 on the flint-Q4_K_M paged gate. The online-softmax arms (cpu_mla_attn.cpp, cuda_mla_attn.cu) rescale their seed as m grows and never had the defect. Fix: seed the denominator at exp(sink - m_final) in both the NEON and scalar lanes; bit-identical when the sink is the max (exp(0)==1.0f).

## Resolution

Fixed 2026-10-27 in the same flow (worktree /tmp/flint-paged, branch row/FLINT-PAGED): both lanes seed std::exp(sink_p[h] - m) after pass 1. The flint paged synthetic gate reproduced the defect red-first (mixed-batch max logit gap 1.21; L0 attention output -0.128 vs hand-computed -1.139 over identical bf16 streams) and runs green with the fix (L0 exact vs the reference control).
