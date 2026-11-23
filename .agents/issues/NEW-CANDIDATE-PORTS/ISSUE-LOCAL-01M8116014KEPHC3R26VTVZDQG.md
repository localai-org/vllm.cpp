ID: ISSUE-LOCAL-01M8116014KEPHC3R26VTVZDQG
Title: olala bf16 token gate: first generated token diverges from llama-cpp-olala
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

Greedy decode on the pinned bf16 artifact diverges from the llama-cpp-olala oracle at the first generated token (705 vs 151644) and degrades into repeated 705. Evidence and both id sequences recorded in docs/bench-evidence/new-candidate-ports-olala-20261122/token-gate-bf16-20261122.txt. The fork golden harness is in the same directory. Config-level hypotheses already ruled out: final_norm, embedding normalization, prompt tokenization (now pinned to the real fork-tokenized ids).

## Resolution

- 2026-11-22 FIXED on row/NEW-CANDIDATE-PORTS-parity. Four root causes, all
  porting errors against the reference/fork, found by per-layer hidden-state
  bisection (fork OLALA_DUMP_DIR hook + new OlalaGenerateHost dump hook):
  1. V-mixer elementwise gate missing: `y *= silu(attn_gate @ x + olala.gate_bias)`
     (modeling_olala.py:1777-1786; fork src/models/olala.cpp:3795-3799). The
     tensor was loaded and commented but never applied. Added OlalaParams::gate_bias.
  2. Mamba3 halved rotary clobbered the rotated high half: the pass-through
     else-branch wrote Q/K[n] for ALL n >= A, overwriting the n+N/2 rotary
     results of groups 2/3 (fork olala.cpp apply_rotary groups 0..3).
  3. MIMO projections read with a rank-2 stride on rank-3 [H,R,P] tensors:
     WAt(w, h*R+r, p) strides by shape[1]=R instead of shape[2]=P.
     Now W3At (element (h,r,p) at ((h*R)+r)*P + p). Same fix in the test's
     RefMambaStep transcription (it mirrored the bug, red-before by mutation).
  4. Norm weights applied as (1 + w) although this GGUF stores the EFFECTIVE
     weight (1 + zero-centered gamma): attn_q_norm/attn_k_norm store ~7-225
     (not raw gamma); the fork multiplies directly (olala.cpp:3754-3755).
     Fixed RmsNormHead and ssm_b_norm/ssm_c_norm usage. Same in test refs.
  5. (reviewer finding) MoE routing misread: topk SELECTS by
     sigmoid(logits) + expert_bias (:1512-1513) but weights normalize the
     gathered RAW sigmoid scores (:1514-1517); impl and test RefMoe
     normalized biased scores. Fixed both.
  6. V token shift at document start: the decode-step host did not mask the
     shift alpha to 0 at pos==0 (reference :636-637; fork v_shift_fused reads
     inp_pos); the batch host already did. Fixed.
  Token gate re-measured 2026-11-22 (same prompt/artifact/harness):
  ours [151644, 77091, 151646, 34484, 151654, 1654] == oracle, all 6 greedy
  ids exact. Gates: test_olala_ops 5/5 cases 254/254 assertions in BOTH the
  Release and Debug builds.
  moe_shared_expert_gate checked: no shared_gate tensor in this GGUF and no
  config key, so the reference branch (:1543-1549) is disabled on this
  artifact; nothing to port.
- 2026-11-23 records correction: the body of commit ccf038670 claimed that
  commit also recorded the EXACT gate, the six porting errors, the real prompt
  ids and the Finding 3 disposition in
  `.agents/specs/new-candidate-ports-olala-w2.md`. It did not — that commit
  touched only `tests/vllm/models/test_olala_ops.cpp`; the spec stayed at
  5/5 cases 253/253 with no `## Owed`. The fresh re-review caught the
  overclaim. The spec now carries those facts, the Finding 3 disposition and
  an `## Owed` section, supplied by a follow-up records commit on this branch.
  No history was rewritten.
