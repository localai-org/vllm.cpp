#!/usr/bin/env python3
"""Generate Kolibri-1 goldens from the PRIMARY oracle.

The existing goldens come from a pure-torch transcription of the plugin's math. This runs the
plugin itself, through vLLM, at the pinned commit - so the fingerprints are the model author's
own implementation rather than a proxy for it.

Same shape as tests/vllm/models/kolibri1_goldens.json: greedy, temperature 0, 32 tokens, the
8-prompt set, so the two files can be diffed directly.

Slow by design: 0.6 tok/s. It runs once, offline. Speed is irrelevant to a fingerprint.
"""
import json
import os
import sys

MODEL = "/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean"
OUT = sys.argv[1] if len(sys.argv) > 1 else "/tmp/opencode/kolibri1_goldens_primary.json"

MAX_NEW_TOKENS = 32

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

os.environ.setdefault("VLLM_PLUGINS", "aleph_alpha_inference")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from transformers import AutoTokenizer  # noqa: E402
from vllm import LLM, SamplingParams  # noqa: E402


def main() -> int:
    tok = AutoTokenizer.from_pretrained(MODEL)
    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,
        dtype="bfloat16",
        max_model_len=512,
        max_num_seqs=1,
        gpu_memory_utilization=0.74,
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        enforce_eager=True,
        disable_log_stats=True,
        trust_remote_code=True,
    )
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, logprobs=8)

    entries = []
    for i, prompt in enumerate(PROMPTS):
        ids = tok.encode(prompt, add_special_tokens=False)
        out = llm.generate([prompt], params, use_tqdm=False)[0]
        gen = list(out.outputs[0].token_ids)
        # Final-position top-8, as logprobs: vLLM does not surface raw logits.
        final = {}
        if out.outputs[0].logprobs:
            last = out.outputs[0].logprobs[-1]
            final = {
                "top8_token_ids": [t for t, _ in sorted(last.items(), key=lambda kv: -kv[1].logprob)],
                "top8_logprobs": [
                    round(float(v.logprob), 4)
                    for _, v in sorted(last.items(), key=lambda kv: -kv[1].logprob)
                ],
            }
        entries.append({
            "prompt": prompt,
            "input_ids": ids,
            "generated_ids": gen,
            "final_topk": final,
        })
        print(f"[{i+1}/{len(PROMPTS)}] {prompt!r}: argmax-first {gen[0] if gen else None}, "
              f"{len(gen)} tokens", flush=True)

    old = json.load(open(
        "/home/lu_zero/Sources/vllm.cpp/tests/vllm/models/kolibri1_goldens.json"))
    agree = sum(
        1 for a, b in zip(entries, old["prompts"]) if a["generated_ids"] == b["generated_ids"]
    )
    first = sum(1 for a, b in zip(entries, old["prompts"]) if a["generated_ids"][:1] == b["generated_ids"][:1])

    out = {
        "model": MODEL,
        "max_new_tokens": MAX_NEW_TOKENS,
        "decode": "greedy, temperature 0",
        "provenance": "PRIMARY oracle: aleph-alpha-inference 1.0.0 (049a6a7) via vLLM "
                      f"{__import__('vllm').__version__} CPU, tensor_parallel_size=4",
        "vs_transcription_goldens": {
            "identical_full_sequences": f"{agree}/{len(PROMPTS)}",
            "identical_first_token": f"{first}/{len(PROMPTS)}",
        },
        "prompts": entries,
    }
    with open(OUT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"\nidentical full sequences vs the transcription goldens: "
          f"{agree}/{len(PROMPTS)}")
    print(f"identical first token: {first}/{len(PROMPTS)}")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
