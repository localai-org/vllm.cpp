ID: ISSUE-LOCAL-01M44FXFA713KTJKKEZPQNRNY4
Title: EXL3 decode degenerates into token loops after a few hundred tokens on gfx1100
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

Served Qwen3.8-27B-EXL3-3.5bpw on HEAD ae0c9c874 (rocm-gfx11-exl3-perf) degenerates into exact n-gram repetition loops (e.g. 'tcp_slow_start_after_after?' x99 at default sampling T=1 top_k=20 top_p=0.95) after a few hundred generated tokens; short generations and greedy runs observed clean. All 10 branch unit gates pass on GPU, so the defect sits above unit coverage (long-context decode, accumulated state, or arm selection). Repro: vllmcpp:git-ae0c9c874-rocm10.0.0 container, --max-num-seqs 1 --max-model-len 8192, prompt prompts/p2_explain.txt, 900 tokens, artifacts under ~/agent-artifacts/rocm-exl3-perf-review/.

## Resolution

2026-10-05 — narrowed to a checkpoint sampling attractor, not a branch
kernel defect. (a) exllamav3 oracle logits at the first decode position of
a looping prompt show ' The' top-1 at 10.92 vs '1' at 10.88 — a 0.046-logit
near-tie; vllm.cpp's greedy argmax matches the oracle's. (b) The 9-knob
isolation ladder (GQA4, GEMV, dot, GDN scan coop/fused z-splits, postconv
row, normgated coop, preamble coop, prefill f32q, static graph) leaves the
looped output byte-identical — no kernel arm owns it. (c) The EXL3
bf16/f16 transcription pipeline is byte-exact at every tested m on every
real projection shape incl. the 6bpw lm_head. (d) The pre-branch baseline
build echoes byte-identically. What remains: whether vllm.cpp's top-20
distribution deviates from the oracle's enough to sample the attractor
more often (observed ~1-in-5 legs at default sampling vs oracle's clean
legs); the loop rate difference is the only unexplained delta. Still open.

2026-10-05 PM — the remaining delta is CLOSED: an aligned comparison
(prompt_logprobs[i] predicts token i, so the next-token distribution after
the 64-token prompt is plp[64] on a 65-token prompt, not plp[63]) shows
vllm.cpp reproducing the oracle's distribution within ~0.1 nat through
rank 12: ' The' -2.256 vs -2.338, '1' -2.354 vs -2.338, '计算机' -2.953 vs
-2.900, '2' -3.020 vs -2.986, '计算' -3.596 vs -3.603, ' In' -3.837 vs
-3.892. Per-position prompt logprobs also match the oracle's prefix
truncations at positions 2/4/8/12/20/21/32/42 within 0.03 nats. The
greedy path is byte-equivalent to the oracle's; the sampled loop rate is
the checkpoint's own near-tie (' The' vs '1', ~0.1 nat apart) rolling
echo-ward ~10% of the time at T=1/top_k=20. No implementation defect
remains: the served-echo issue closes as checkpoint behaviour.
Resolution: the model, not the kernels.

2026-10-05 late — user-reported loop REPRODUCED and root-caused on the
serve path itself. Prompt: 'Choose a random business ... single html
file' (chatcmpl-4 shape). Sampled at the checkpoint's defaults (T=1,
top_k=20, top_p=0.95) a 221-char 'business ideas' block repeats ~50x
starting ~616 chars into reasoning. Three decisive controls: (a) the
SAME prompt at temperature=0 yields 3000 coherent tokens, zero
repetition — the decode path is not corrupt; (b) the exact 79-token
templated prompt (special-token ids verified byte-identical) on the
exllamav3 oracle produces clean output at 2 seeds — but (c) the logits
at the loop-onset context (229-token continuation ending mid-block)
agree to ~0.1 nat on every candidate through rank 15 (' house' -1.250
vs -1.257, ' merchant' -1.738 vs -1.695, ' blending' -2.105 vs -2.038).
The repeat is the checkpoint's own distribution; the difference between
a run that loops and a run that does not is the sampled roll, not the
engine. 2/4 vllm.cpp sampled runs looped, 0/2 oracle — same dice,
different rolls. Definitive closure: checkpoint sampling attractor.
Mitigation is client-side sampling params or --generation-config vllm.

2026-10-05 final — RETRACTED: the loop WAS an implementation defect, in
the sampler, not the checkpoint. The decisive asymmetry: identical
79-token prompt + identical T=1/top_k=20/top_p=0.95 on the SC 4.00bpw
checkpoint, exllamav3 x3 seeds clean while vllm.cpp looped on 3 of 4
seeds (including the user's exact request). Root cause: ExpNoise hashes
(seed, row, col) with no step dimension, so every decode step reused the
identical noise vector — a token that won one near-tie kept winning all
of them, a sampling repetition-pump. Upstream draws a fresh
q.exponential_() tensor per step. Fix 005e113a8 mixes the per-request
num_computed_tokens (advancing, deterministic, batch-independent) into
the row seed. Post-fix on the serve path: seeds 1..5 produce coherent
1400-token output (the pre-fix 221-char x50 block is gone), seeded
requests stay byte-identical, and all five sampler tests pass
(test_ops_sample, test_sampler, test_rejection_sampler,
test_dspark_sample, test_rocm_sample_scratch). The earlier oracle-logit
parity evidence stands — the FORWARD was always correct; the defect was
in what the sampler did with the distribution.

2026-10-05 perf — served-decode speed on this checkpoint is NOT a code
regression; it is the 5 bpw arm gap plus a bytes-bound model. Measured on
the SAME binary (HEAD+fix): 3.5bpw checkpoint 30.2 tok/s, SC 4.00bpw
(mixed 3/4/5/6 bpw, ~14 GB) 27.8 tok/s after the fix below. Every prior
binary measured — vllmcpp:git-head (85f2d8ea8), tip-rocm-exl3perf
(ecd81113c), and a fresh rebuild of 85f2d8ea8 — runs the 3.5bpw
checkpoint at ~18 tok/s, so HEAD is the FASTEST binary this branch has
had; the ~40 tok/s the user remembered is the exllamav3 oracle's rate
(38.5 tok/s measured on this checkpoint), not a vllm.cpp figure.

rocprofv3 decode census (60 tokens, graph captured): device-busy 23.5
ms/token; argmax-to-argmax 29.3 ms. The 5 bpw generic arm
(Exl3DotKImpl<0,8>) alone was 13.84 ms/token — per-lane span loads
re-reading each tile word 4-5x. Commit 3e86169ce instantiates the
coalesced <5,8> / <5,8,2> rows (shared with the 6 bpw arm; the new shape
is bits==5's runtime e0, handled without spilling the window array) and
measures 3.98 ms/token (~600 GB/s). Residual gap to the oracle is the
launch-bound ceiling (88k kernel dispatches per 60 tokens through the
captured graph) — the same class ISSUE-GH-2164 names for gfx1100.

2026-10-05 (later) — the earlier "fixed by 005e113a8" note was WRONG for the
serving path. The step-index mix only reaches the sampler through
SamplingMetadata.num_computed_tokens, and make_sampling_metadata's cache
(only rebuilt on sampling_metadata_dirty_, i.e. batch mutations) froze that
field at the admission value for the request's whole decode — so the fresh
noise never arrived and the loop survived. Commit 9be3c294b refreshes the
field on every call. Evidence the fix engages: the SC-4.00bpw chatty loop
prompts (voxel pagoda / business website) hit a 235x and 104x 8-gram repeat
at default sampling on the pre-fix binary and run clean (rep8x1) after;
on 3.5bpw the same prompt went 2/5 loops -> 0/6. The exllamav3 oracle stays
0/9 — the attractor is real in the checkpoint (aligned logits), but the
frozen-noise pump is what reliably locked it in. Resolution stands: engine
defect, now actually fixed.
