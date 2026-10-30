#!/usr/bin/env python3
"""Smoke test: can vLLM's CPU backend serve Kolibri-1 via the author's own plugin?

The question is whether the primary oracle is reachable without a GPU lease. It registers
and imports; this checks that it executes. One short prompt, one token, eager mode, minimal
context - the smallest thing that would still exercise the MoE forward.
"""
import os
import sys
import time

MODEL = "/mnt/models/Aleph-Alpha/Kolibri-1-bf16-clean"

os.environ.setdefault("VLLM_LOGGING_LEVEL", "INFO")
os.environ.setdefault("VLLM_PLUGINS", "aleph_alpha_inference")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

from vllm import LLM, SamplingParams  # noqa: E402


def main():
    t0 = time.perf_counter()
    llm = LLM(
        model=MODEL,
        tokenizer=MODEL,


        max_model_len=64,
        max_num_seqs=1,
        gpu_memory_utilization=0.74,
        tensor_parallel_size=4,
        distributed_executor_backend="mp",
        enforce_eager=True,
        disable_log_stats=True,
        trust_remote_code=True,
    )
    print(f"[smoke] model loaded in {time.perf_counter()-t0:.1f}s", flush=True)

    prompts = ["The capital of Australia is"]
    t1 = time.perf_counter()
    out = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=8))
    dt = time.perf_counter() - t1
    for o in out:
        print(f"[smoke] prompt {o.prompt!r} -> {o.outputs[0].text!r}")
        print(f"[smoke] token ids: {list(o.outputs[0].token_ids)}")
    print(f"[smoke] 8 tokens in {dt:.1f}s ({8/dt:.3f} tok/s)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
