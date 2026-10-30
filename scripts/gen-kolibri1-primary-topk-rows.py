#!/usr/bin/env python3
"""Task 1: per-position top-100 logprob rows from the PRIMARY oracle.

Maki needs the primary's own per-position top1-top2 and top6-top7 margins, so the two
implementations' margin distributions can be compared on the same instrument. vLLM does not
surface raw logits, but it does surface per-position logprob distributions via
`prompt_logprobs`, so the generated chain is teacher-forced: generate greedily, then re-feed
prompt+chain as a prompt and read the distribution that predicted each generated token.

Output: JSON, per prompt, per generated position, the top-100 (token_id, logprob) pairs.
"""
import json
import os
import sys

MODEL = "/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/primary_topk_rows.json"

PROMPTS = [
    "alpha beta gamma delta epsilon zeta",
    "der Mond scheint hell über den Bergen",
    "The capital of Australia is",
    "Wissen ist Macht, aber Zeit ist Geld",
    "x1 = 3, x2 = 7, x3 = 11, x4 =",
    "Translation to German: the weather is beautiful today",
    "Ein gutes Buch liest man am besten",
    "counting: one two three four five six seven",
]
MAX_NEW = 32
TOPK = 100

os.environ.setdefault("VLLM_PLUGINS", "aleph_alpha_inference")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402
from vllm.inputs import TokensPrompt  # noqa: E402


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL, tokenizer=MODEL, dtype="bfloat16",
        max_model_len=1024, max_num_seqs=1, gpu_memory_utilization=0.74,
        tensor_parallel_size=4, distributed_executor_backend="mp",
        enforce_eager=True, disable_log_stats=True, trust_remote_code=True,
        max_logprobs=128,
    )

    greedy = SamplingParams(temperature=0.0, max_tokens=MAX_NEW)
    # prompt_logprobs returns the distribution at every prompt position; index i of that
    # list is the predictive distribution OVER token i, so the distribution predicting the
    # first generated token sits at index len(prompt).
    score = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOPK)

    out = {"model": MODEL, "topk": TOPK, "max_new_tokens": MAX_NEW, "prompts": []}
    for i, prompt in enumerate(PROMPTS):
        p_ids = tok.encode(prompt, add_special_tokens=False)
        gen = list(llm.generate([prompt], greedy, use_tqdm=False)[0].outputs[0].token_ids)

        full = p_ids + gen
        res = llm.generate([TokensPrompt(prompt_token_ids=full)], score, use_tqdm=False)[0]

        rows = []
        for k, gid in enumerate(gen):
            # prompt_logprobs[i] is the predictive distribution over full[i], so the
            # distribution that PREDICTED gen[k] sits at index len(p_ids) + k. (The
            # original len(p_ids) - 1 + k stored each position one step behind; see
            # ISSUE-LOCAL-01M5CQ4GMFF9ZAYYMFYDV6WZ3Q.)
            pos = len(p_ids) + k
            pl = res.prompt_logprobs[pos]
            if pl is None:
                raise RuntimeError(
                    f"prompt {i!r}: no logprob distribution at pos {pos} for gen_index {k}")
            # pl is dict[int, Logprob] for that position (flat_logprobs defaults False).
            items = [(tid, lp.logprob) for tid, lp in pl.items() if lp is not None]
            items.sort(key=lambda t: -t[1])
            # Under greedy decode gen[k] must be the argmax of the distribution that
            # predicted it. The old code silently skipped positions where it was not
            # in the (wrong) window; make that a loud failure instead.
            if items[0][0] != gid:
                raise RuntimeError(
                    f"prompt {i!r} gen_index {k}: token {gid} is not the top-1 "
                    f"(top-1 is {items[0][0]}); greedy chain and scored chain diverged")
            top = items[:TOPK]
            lp_of = dict(items)
            vals = [lp for _, lp in top]
            rows.append({
                "gen_index": k,
                "token": gid,
                "top": [[int(t), round(float(v), 6)] for t, v in top],
                "logprob_of_token": round(float(lp_of[gid]), 6),
                "top1_top2_margin": round(float(vals[0] - vals[1]), 6) if len(vals) > 1 else None,
                "top6_top7_margin": round(float(vals[5] - vals[6]), 6) if len(vals) > 6 else None,
            })

        out["prompts"].append({
            "prompt": prompt,
            "input_ids": p_ids,
            "generated_ids": gen,
            "rows": rows,
        })
        m12 = [r["top1_top2_margin"] for r in rows if r["top1_top2_margin"] is not None]
        m67 = [r["top6_top7_margin"] for r in rows if r["top6_top7_margin"] is not None]
        print(f"[{i+1}/8] {prompt[:34]!r}: {len(rows)} positions, "
              f"median top1-2 {min(m12 or [0]):.4f}..{max(m12 or [0]):.4f}", flush=True)

    with open(OUT, "w") as f:
        json.dump(out, f)
        f.write("\n")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
