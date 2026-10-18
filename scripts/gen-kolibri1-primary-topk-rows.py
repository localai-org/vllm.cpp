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
    # prompt_logprobs returns the distribution at every prompt position; index i of that list
    # is the distribution that PREDICTED token i, so position len(prompt)-1 is the first
    # generated token.
    score = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=TOPK)

    out = {"model": MODEL, "topk": TOPK, "max_new_tokens": MAX_NEW, "prompts": []}
    for i, prompt in enumerate(PROMPTS):
        p_ids = tok.encode(prompt, add_special_tokens=False)
        gen = list(llm.generate([prompt], greedy, use_tqdm=False)[0].outputs[0].token_ids)

        full = p_ids + gen
        res = llm.generate([TokensPrompt(prompt_token_ids=full)], score, use_tqdm=False)[0]

        rows = []
        for k, gid in enumerate(gen):
            pos = len(p_ids) - 1 + k          # position whose distribution predicted gen[k]
            pl = res.prompt_logprobs[pos]
            if pl is None:
                continue
            # pl is dict[int, Logprob] for that position (flat_logprobs defaults False).
            items = [(tid, lp.logprob) for tid, lp in pl.items() if lp is not None]
            if not any(tid == gid for tid, _ in items):
                continue
            items.sort(key=lambda t: -t[1])
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
