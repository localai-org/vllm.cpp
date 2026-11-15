#!/usr/bin/env python3
# GLM/DSA G1 (GLM-4.7-Flash, `Glm4MoeLiteForCausalLM`) — the TEACHER-FORCED
# DIVERGENCE DIAGNOSTIC. Mirror of scripts/deepseek-v2-neartie-gap.py.
#
# The gate is STRICT TOKEN-EXACT wherever vLLM is deterministic. This is NOT a way
# to soften it: it is the ratified diagnostic run WHEN our engine diverges from a
# deterministic vLLM — teacher-force vLLM on OUR exact generated sequence and ask,
# per position, how many nats vLLM's OWN argmax beats OUR token by, GIVEN OUR
# PREFIX. gap 0 = our token IS vLLM's argmax (downstream of an earlier bf16 tie);
# gap <= 0.5 nats = a bf16 near-tie vLLM cannot separate; gap large / our token
# outside vLLM's top-K = a REAL forward divergence (fix the model, do not widen a
# band).
#
# Emits into --golden-dir:
#   our_ids.npy           [N,T] i32 — OUR engine's exact greedy tokens (anchor).
#   neartie_gap_mnats.npy [N,T] i32 — vLLM's teacher-forced gap in MILLI-nats.
#   teacher_forced_gap_nats.npy [N,T] f64 — the same gap in NATS, unrounded (K1).
#   teacher_forced_argmax.npy   [N,T] i32 — vLLM's teacher-forced argmax id there.
#
# WHY TWO GAPS (#2839 K1). The millinat array rounds `gap * 1000`, so every gap
# below half a millinat is committed as 0 and "the oracle prefers its own token
# by 0.0004 nats" is indistinguishable from "the oracle agrees with ours". At a
# first divergence the prefix IS the oracle's, so its teacher-forced argmax must
# be its own greedy token and the gap must be positive -- which makes that single
# number the whole question. `teacher_forced_argmax.npy` answers the neighbouring
# one directly: is the argmax our token, or the oracle's?
# Input: our_ids.i32 in --golden-dir ([N,T] i32 raw dump of our engine's tokens).
#
# MEMORY: same UNIFIED-memory constraints and moe_backend=triton substitution as
# the capture (both arms MUST see one oracle configuration). Run ALONE on dgx,
# oracle venv, PATH including ~/venvs/vllm-oracle/bin.
import argparse
import os

import numpy as np

# MUST match test_glm4_moe_lite_paged_engine.cpp::Prompts() and the capture.
PROMPTS = [
    "The capital of France is",
    "def fibonacci(n):",
    "In a shocking finding, scientists discovered a herd of unicorns living in",
    "Q: What is 17 * 23?\nA:",
    "The three laws of robotics are",
    "Once upon a time, in a land far away,",
    "The chemical symbol for gold is",
    "To be or not to be, that is",
]
OUTSIDE_TOPK_MNATS = 99_999_000
GAP_NATS_NAME = "teacher_forced_gap_nats.npy"
ARGMAX_NAME = "teacher_forced_argmax.npy"
# A gap this small is a tie the bf16 forward cannot separate; it is
# also the entire range the millinat array cannot represent.
SUB_MILLINAT_NATS = 0.0005

def gap_rows(prompt_ids, our_ids, prompt_logprobs, topk):
    """vLLM's teacher-forced gap for one prompt, at three resolutions.

    `prompt_logprobs` is vLLM's list for the concatenated prompt + our tokens,
    one entry per position (None where the API reports nothing). The generated
    tokens start at `len(prompt_ids)`.

    Returns `(mnats, nats, argmax)`, each a plain list of `len(our_ids)`:

    - `mnats` -- int MILLINATS, floored at zero. This is the committed
      `neartie_gap_mnats.npy` and keeps its exact current meaning.
    - `nats` -- the same gap as an unrounded float, in nats. `0.0004` survives
      here and is `0` above, which is the reason this array exists.
    - `argmax` -- vLLM's teacher-forced argmax id at that position, or -1 when
      the API reported no distribution. When this equals our token, the oracle
      reproduced OUR token given OUR prefix.

    A token outside vLLM's top-`topk` is a real divergence, not a near-tie, and
    carries `OUTSIDE_TOPK_MNATS` in both numeric arrays.
    """
    prompt_len = len(prompt_ids)
    mnats, nats, argmax = [], [], []
    for j in range(len(our_ids)):
        pos = prompt_len + j
        d = (prompt_logprobs[pos] if pos < len(prompt_logprobs) else None) or {}
        our_tid = int(our_ids[j])
        arg_tid = max(d, key=lambda k: d[k].logprob) if d else -1
        argmax.append(int(arg_tid))
        if our_tid in d:
            gap = max(0.0, d[arg_tid].logprob - d[our_tid].logprob)
            mnats.append(int(round(gap * 1000.0)))
            nats.append(float(gap))
        else:
            mnats.append(OUTSIDE_TOPK_MNATS)
            nats.append(float(OUTSIDE_TOPK_MNATS))
    return mnats, nats, argmax


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.72)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--max-num-seqs", type=int, default=4)
    ap.add_argument("--moe-backend", default="triton")
    args = ap.parse_args()
    from vllm import LLM, SamplingParams

    N, T = len(PROMPTS), args.max_tokens
    our = np.fromfile(os.path.join(args.golden_dir, "our_ids.i32"),
                      dtype="<i4").reshape(N, T)
    greedy = np.load(os.path.join(args.golden_dir, "greedy_ids.npy"))

    llm_kwargs = dict(model=args.model, dtype="bfloat16", enforce_eager=True,
                      trust_remote_code=True,
                      gpu_memory_utilization=args.gpu_mem_util,
                      max_model_len=args.max_model_len,
                      max_num_batched_tokens=args.max_num_batched_tokens,
                      max_num_seqs=args.max_num_seqs)
    if args.moe_backend:
        llm_kwargs["moe_backend"] = args.moe_backend
    print(f"moe_backend={args.moe_backend or 'auto'}")
    llm = LLM(**llm_kwargs)

    gap_mnats = np.zeros((N, T), dtype="<i4")
    gap_nats = np.zeros((N, T), dtype="<f8")
    argmax_ids = np.zeros((N, T), dtype="<i4")
    print(f"=== teacher-forced gap: {args.model} (vLLM's logits on OUR prefix) ===")
    max_gap = 0.0
    worst = None
    n_div = 0
    for i in range(N):
        prompt_ids = np.fromfile(
            os.path.join(args.golden_dir, f"p{i}_prompt.i32"), dtype="<i4").tolist()
        our_ids = [int(x) for x in our[i]]
        full = list(prompt_ids) + our_ids
        sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=args.topk)
        out = llm.generate({"prompt_token_ids": full}, sp)[0]
        plp = out.prompt_logprobs
        P = len(prompt_ids)
        row_mnats, row_nats, row_argmax = gap_rows(prompt_ids, our_ids, plp,
                                                  args.topk)
        gap_mnats[i, :] = row_mnats
        gap_nats[i, :] = row_nats
        argmax_ids[i, :] = row_argmax
        for j in range(T):
            if 0.0 < row_nats[j] < float(OUTSIDE_TOPK_MNATS)                     and row_nats[j] > max_gap:
                max_gap, worst = row_nats[j], (i, j, row_nats[j])
            if row_mnats[j] == OUTSIDE_TOPK_MNATS:
                print(f"  p{i:2d} tok{j:2d}: OUR TOKEN {our_ids[j]} OUTSIDE vLLM "
                      f"top-{args.topk} (REAL divergence)")
            if our_ids[j] != int(greedy[i, j]):
                n_div += 1
                print(f"  p{i:2d} tok{j:2d}: our={our_ids[j]} "
                      f"vLLM_greedy={int(greedy[i,j])} "
                      f"vLLM_argmax={row_argmax[j]} "
                      f"gap={row_nats[j]:.6f} nats")

    np.save(os.path.join(args.golden_dir, "our_ids.npy"), our)
    np.save(os.path.join(args.golden_dir, "neartie_gap_mnats.npy"), gap_mnats)
    np.save(os.path.join(args.golden_dir, GAP_NATS_NAME), gap_nats)
    np.save(os.path.join(args.golden_dir, ARGMAX_NAME), argmax_ids)
    flat_ours = our.reshape(-1)
    sub_millinat = int(np.count_nonzero(
        (gap_nats > 0.0) & (gap_nats < SUB_MILLINAT_NATS)))
    oracle_took_ours = int(np.count_nonzero(
        (argmax_ids >= 0) & (argmax_ids == flat_ours)))
    print(f"=== K1 resolution: {sub_millinat} position(s) hold a POSITIVE gap the "
          f"millinat array commits as 0 (< {SUB_MILLINAT_NATS} nats); the oracle's "
          f"teacher-forced argmax IS our token at {oracle_took_ours} of "
          f"{N * T} positions ===")
    print(f"=== {n_div} token-divergent positions vs vLLM greedy; "
          f"max teacher-forced gap {max_gap:.4f} nats (worst {worst}) ===")
    print(f"wrote {args.golden_dir}/our_ids.npy + neartie_gap_mnats.npy + K1 "
          f"{gap_mnats.shape}")


if __name__ == "__main__":
    main()
