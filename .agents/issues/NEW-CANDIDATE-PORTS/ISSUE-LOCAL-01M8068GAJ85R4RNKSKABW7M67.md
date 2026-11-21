ID: ISSUE-LOCAL-01M8068GAJ85R4RNKSKABW7M67
Title: NEW-CANDIDATE-PORTS W2c-3 (olala): Mamba3-MIMO host step op, unit-gated vs reference
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

W2c-2 landed (V attention). W2c-3 ports the M-layer step OlalaMambaStepHost (modeling_olala.py:1265-1370 + :861-889 + :764-770): in_proj split z/x/dt/A/trap per head, in_proj_dyn B/C/angles, softplus dt + clamped -softplus A, sigmoid trap, B/C RMSNorm over d_state, data-dependent rotary angle cumsum(tanh(angle)*dt*pi) on pairs (i, i+N/2), MIMO v=x*MIMO_V, state update alpha/beta/gamma, o = state@C + xv*D gated by silu(z*MIMO_Z), y = sum_r o*MIMO_O. State: angle[H,A], ssm[H,P,N], k[R,H,N], v[H,P]. Unit-gated vs in-test f64 transcription over 3 tokens incl. state: 4/4 cases 243/243 green. ssm_m3_* tensors are vestigial (unreferenced in the reference, like the conv/rel ones). Prefill = stepping per token (chunked kernel matches to bf16 noise, :1148).

## Resolution

Landed on row/NEW-CANDIDATE-PORTS (W2c-3 commit 7b6c159f3): OlalaMambaStepHost, 4/4 cases 243/243 assertions green vs the in-test f64 reference over 3 tokens incl. state. ssm_m3_* recorded as vestigial.
