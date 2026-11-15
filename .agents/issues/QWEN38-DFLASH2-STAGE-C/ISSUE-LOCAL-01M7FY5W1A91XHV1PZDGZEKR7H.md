ID: ISSUE-LOCAL-01M7FY5W1A91XHV1PZDGZEKR7H
Title: The drafter-ON verify forward degenerates upstream of the GDN spec kernels: the conv-spec x stream collapses at GDN layer 8 with NaNs — the emitted stream is [220, 0, 0, ...] (the W4 token-identity STOP)
Row: QWEN38-DFLASH2-STAGE-C
State: OPEN
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-15
Updated: 2026-11-15
Closed: -

## Problem

With kGdnSpecDecode (id 12) and kGreedyRejectionSample (id 107) both ported, the w4-on-probe leg (Q4_K_M KV 64, drafter-ON, direct arm, bfp8, eager) runs the loop to completion for the first time — exit 0, 24 tokens, acceptance telemetry live (0.75) — and the emitted token stream is [220, 0, 0, ...] vs the recorded greedy stream [220, 16, 23066, ...] (d49c53d2...): the W4 greedy token-identity gate FAILS (the spec's STOP condition). Probe evidence (VT_TT_GDN_SPEC_DEBUG, w4-diag-spec.stderr.log): (1) the conv-spec kernel's x stream — the spec tokens' qkvz projection — has NaN at the FIRST call (step 0, GDN layer 0: x_asum=nan) and collapses to a frozen ~43.3553 sum at GDN layer 8+ in EVERY step (steps 0-3), while the state-window reads stay real (T_asum ~6-15k) — so the degeneracy is UPSTREAM of both spec kernels (the projection GEMM or the residual it reads); (2) the GDN spec kernel's gathered state is real (s_asum ~0.6-9.5k) and its received activations are EXACTLY zero (q_asum=0, v_asum=0) for the collapsed layers — the zero outs are a downstream consequence, not the cause; (3) the acceptance telemetry (24/32 proposed accepted, num_sampled live through finalize) plus the all-zero ids fit BOTH argmaxes degenerating to token 0 — the classic NaN-argmax signature (NaN comparisons false -> best stays 0) — i.e. the verify's block logits AND the draft's block logits are NaN-poisoned. The GDN spec kernel and the sampler are exonerated: their focused gates are bit-exact vs the CPU oracle, the metadata (nat0/init0/slots) is sane every call, and the first NaN precedes any GdnSpecDecode call. Owed: bisect the NaN injection point in the verify forward's spec rows (the embed of the draft's proposals -> the qkvz projection GEMM -> the conv-spec x), with the bfp8 weight-staging lever as the prime suspect (the draft trunk stages BFP8_B; the no-bfp8 A/B is capacity-blocked at KV 32 on this base — the draft's bf16 trunk OOMs at the same 178257920 B ffn_down allocation).

## Resolution

-
