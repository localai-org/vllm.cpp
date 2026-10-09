ID: ISSUE-LOCAL-01M43XCSPS01QDG7D65Q8NKHQ7
Title: Served greedy decode corrupt on continuation prompts at rocm-gfx11-exl3-perf head (oracle-clean comparison)
Row: BACKEND-ROCM
State: CLOSED
Kind: bug
GitHub: -
Mirror: PENDING
Availability: FULL
Created: 2026-10-04
Updated: 2026-10-05
Closed: 2026-10-05

## Problem

Reproduced 2026-10-04 on gfx1100, head ae0c9c874, default knobs, model Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw: the 128-token greedy continuation of the campaign's standard history-of-computing prompt (BASE x3, ~45 tokens) degenerates into verbatim prompt copying in a loop. The pinned secondary oracle exllamav3 (container exl3-pr:rocm, greedy, same prompt bytes) produces a clean diverse continuation for the same prompt, so this is a vllm.cpp defect, not checkpoint behavior. The cont2048 (BASE x48) loop is checkpoint-inherent: the oracle loops with the same token cycle. Knob A/B at head: loop byte-identical with VT_ATTN_DECODE_GQA4=0, VT_EXL3_GEMV_FOLD_OUT=0, VT_EXL3_M1K3_WNT=2 (those arms exonerated). Isolation ladder (VT_GDN_SCAN_ZSPLIT=0, VT_GDN_SCAN_ZSPLIT_PREFILL=0, VT_EXL3_GEMV=0, VLLM_CPP_ROCM_STATIC_GRAPH=0, VT_ROCM_GDN_POSTCONV_ROW=0, VT_GDN_NORMGATED_COOP=0) in flight. Evidence: ~/agent-artifacts/exl3-perf-gfx1100/review/loopaudit/. Note: every unit gate stays green; the campaign's cross-build md5 checks compared builds to each other and could not see an output that was wrong in all of them.

## Resolution

CORRECTED 2026-10-05 (reconciles the parallel session's logits falsification with the review session's evidence; the earlier 'oracle-clean conviction' below was INVALID — its exllamav3 Job ran with the default sampler (the checkpoint's generation_config carries temperature 1 / top_k 20 / top_p 0.95), so the 'clean diverse continuation' was a SAMPLE, not greedy).

The echo is the checkpoint's own greedy attractor, not a vllm.cpp defect: exllamav3 logits at the first decode position of the same BASE x3 prompt put ' The' (561) top-1 at 10.92 with '1' (16) at 10.88 — a 0.046-logit near-tie; vllm.cpp's greedy argmax breaks it toward ' The' and the echo takes over; the oracle breaks it the same way when greedy. The review session's measurements corroborate and are explained by the tie landscape: (a) the N=40..70 intermittency map (loops at 41,42,45,47,48,54-60,62-64,66-69; clean at 40,43,44,46,49-53,61,70) is the per-prompt near-tie pattern, not a KV-offset signature; (b) main looping at ~33 vs head clean at 43 = accumulated numeric drift between builds moving which side of ties each prompt falls — and confirms the campaign did not introduce the behavior; (c) EXL3 kernels byte-exact at served shapes vs the f64 chain at (3,2)/(4,2)/(6,2) x k=1024/5120/17408/248320 x m=1/64 (scratch_probe_v5.log; an earlier ~100% out_diff reading was a probe bug decoding f32 words as f16); (d) fifteen knob arms byte-identical (md5 9cb38029d3): the echo is upstream of every switched arm, consistent with an attractor; (e) all 13 GPU unit gates green throughout, as expected for non-defect behavior. The cont2048 loop was already known checkpoint-inherent. OWED (moved to the served-gate issue): a committed greedy-sampler oracle comparison per prompt family so future echo reports are checked against logits, not sampled output; the served-greedy gate of ISSUE-LOCAL-01M43XDG3FEXNKWBQPRKNSV3CF.
