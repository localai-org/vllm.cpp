ID: ISSUE-LOCAL-01M8061APCKHGTM8JHPRHRCMJ4
Title: NEW-CANDIDATE-PORTS W2c-2 (olala): V-layer differential TPA attention host op, unit-gated vs reference
Row: NEW-CANDIDATE-PORTS
State: CLOSED
Kind: task
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-11-22
Updated: 2026-11-22
Closed: 2026-11-22

## Problem

W2c-1 landed (geodesic + MoE). W2c-2 ports the V-layer mixer core OlalaVAttentionHost (modeling_olala.py:498-745): TPA projections (q = attn_q; k/v = (attn_wa @ attn_wb)/tpa_rank per kv head), token shift with sigmoid alpha zeroed at doc starts (:595-640), plain RMSNorm qk-norm (zero-centered weights are zero), NoPE on this artifact (rope_theta=0), scalable softmax q *= scaler*log(min(pos+1, wsize)) (:661-668), scale 1/head_dim (:727), softcap tanh (:469), causal + left window 1024, GQA repeat_interleave 48/12, differential merge sig - sigmoid(lambda)*noi with lambda = attn_lambda (:734-740). Unit-gated vs an independent in-test f64 transcription: 3/3 cases 67/67 green. Vestigial checkpoint tensors the reference does NOT read (recorded, not ported): attn_rel*, attn_*_conv*, attn_q_clip, attn_res_query, attn_sink_proj, shortconv — token_conv1d_attn is 'no longer implemented' (:195).

## Resolution

Landed on row/NEW-CANDIDATE-PORTS (W2c-2 commit): OlalaVAttentionHost ports the V mixer core; test_olala_ops 3/3 cases 67/67 assertions green in a CPU-only build, including the doc-start alpha-zeroing slip the reference test caught. Vestigial tensors recorded as unused-by-reference.
