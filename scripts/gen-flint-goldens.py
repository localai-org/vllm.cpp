#!/usr/bin/env python3
# vllm.cpp ORIGINAL — the flint GGUF arm's golden generator.
#
# MODEL-TEXT-FLINT-GGUF (.agents/specs/flint-gguf-arm.md): the greedy golden
# chains come from the PINNED llama-cpp-flint oracle (PR #30928 head
# 1b44e30775dcb2f93952116b05f89e5bd797fb00), run as `llama serve` on the
# PUBLISHED container. Every chain is greedy (temperature 0, top_k 1) over the
# RAW prompt ids the oracle's own /tokenize returns — no chat template, no BOS
# (tokens_evaluated == len(ids) proves the server added none). The generated
# token ids come from `n_probs: 1` on the same request (post-sampling, greedy ⇒
# the chosen token is the top-1 and carries its id).
#
# Usage: python3 scripts/gen-flint-goldens.py \
#          --model /mnt/models/flint-GGUF/flint-Q4_K_M.gguf \
#          --port 8901 --out tests/vllm/models/flint_goldens_q4_k_m.json
import argparse
import json
import urllib.request


def post(port, path, body):
    req = urllib.request.Request(
        f"http://127.0.0.1:{port}{path}",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=1800) as r:
        return json.loads(r.read())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--n-predict", type=int, default=24)
    args = ap.parse_args()

    prompts = [
        "The capital of France is",
        "RecursiveSI published the Flint model on",
        "The first three prime numbers are",
        "Water boils at a temperature of",
        "In machine learning, a mixture of experts layer routes",
        "Der Name der deutschen Hauptstadt ist",
        "def fibonacci(n):",
        "The largest planet in the solar system is",
    ]
    out = {
        "model": args.model,
        "oracle": "llama-cpp-flint PR #30928 head 1b44e30775dcb2f93952116b05f89e5bd797fb00",
        "sampling": "greedy (temperature 0, top_k 1), raw prompt ids, no BOS",
        "prompts": [],
    }
    for text in prompts:
        ids = post(args.port, "/tokenize", {"content": text})["tokens"]
        assert len(ids) > 0, text
        r = post(
            args.port,
            "/completion",
            {
                "prompt": ids,
                "n_predict": args.n_predict,
                "temperature": 0.0,
                "top_k": 1,
                "cache_prompt": False,
                "n_probs": 1,
                "post_sampling_probs": True,
            },
        )
        gen = [step["id"] for step in r["completion_probabilities"]]
        assert len(gen) == args.n_predict, (text, len(gen))
        out["prompts"].append(
            {
                "prompt": text,
                "input_ids": ids,
                "generated_ids": gen,
            }
        )
        print(f"{text!r}: {ids} -> {gen}")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
