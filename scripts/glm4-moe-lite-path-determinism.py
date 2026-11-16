#!/usr/bin/env python3
# GLM/DSA G1 (GLM-4.7-Flash, `Glm4MoeLiteForCausalLM`) -- K2, THE ORACLE'S OWN
# PATH DETERMINISM.
#
# The row's STRICT bar is licensed by `greedy_dist.npy`: five greedy runs of the
# pinned vLLM agree at every cell, so its greedy decode reads as deterministic.
# Those five are the SAME call five times -- one prefill shape, one batch shape,
# one decode kernel -- so that experiment cannot see the oracle disagreeing with
# ITSELF ACROSS PATHS, which is what `neartie_gap_mnats.npy` points at:
# teacher-forced on our prefix, the oracle's argmax is OUR token, within half a
# millinat of its own greedy answer.
#
# This probe asks that question directly, on the committed artifacts:
#
#   1. GREEDY DECODE  -- `SamplingParams(temperature=0, max_tokens=T)` on the
#      prompt alone. This must reproduce `greedy_ids.npy`; if it does not, the
#      goldens are not reproducible and nothing downstream is a measurement.
#   2. TEACHER-FORCED -- `prompt_logprobs=topk` on prompt + OUR tokens, exactly
#      as scripts/glm4-moe-lite-neartie-gap.py runs it. Its argmax per generated
#      position is what the committed gap was measured against.
#   3. BOTH AGAIN at each `--max-num-seqs` value, because batch shape selects a
#      different decode kernel and is the cheapest second path available.
#
# Writes `path_determinism.json` beside the goldens and prints ONE verdict:
# `ORACLE PATH-DEPENDENT` when two paths disagree anywhere both were measured,
# else `ORACLE PATH-STABLE`. It changes no golden and no assertion. An oracle
# whose own argmax moves with the path cannot license a STRICT bar against a
# single capture of it, and that is the evidence a ratification or a forward
# repair would rest on. See .agents/specs/glm4-moe-lite-gate-2839.md, K2.
#
# RUN: on dgx, oracle venv at the parity pin, PATH including
# ~/venvs/vllm-oracle/bin, the 31.2B `zai-org/GLM-4.7-Flash` snapshot, ALONE on
# the card. Input: `our_ids.i32` in --golden-dir (our engine's raw token dump).
import argparse
import json
import os

import numpy as np

# MUST match test_glm4_moe_lite_paged_engine.cpp::Prompts() and both captures.
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
NO_DISTRIBUTION = -1
REPORT_NAME = "path_determinism.json"


def compare_paths(greedy_ids, teacher_argmax, our_ids):
    """Compare the oracle's two numeric paths, position by position.

    `greedy_ids`     -- what the oracle's greedy DECODE produced (the goldens).
    `teacher_argmax` -- what its PREFILL plus `prompt_logprobs` named, with
                        NO_DISTRIBUTION wherever the API reported nothing.
    `our_ids`        -- our engine's tokens, used only to locate the first
                        divergence of each prompt.

    Returns counts plus the positions that matter. `path_dependent` is the one
    fact K2 needs: the two paths disagreed where both were measured.
    """
    g = np.asarray(greedy_ids)
    a = np.asarray(teacher_argmax)
    o = np.asarray(our_ids)
    assert g.shape == a.shape == o.shape, (g.shape, a.shape, o.shape)
    assert g.ndim == 2, g.shape
    n, t = g.shape

    compared = 0
    disagree = 0
    reproduced = 0
    flip = 0
    first = []
    for i in range(n):
        fd = next((j for j in range(t) if int(o[i, j]) != int(g[i, j])), None)
        if fd is not None:
            first.append((int(i), int(fd)))
        for j in range(t):
            named = int(a[i, j])
            if named == NO_DISTRIBUTION:
                continue
            compared += 1
            if named != int(g[i, j]):
                disagree += 1
            if named == int(o[i, j]) and int(o[i, j]) != int(g[i, j]):
                reproduced += 1
                if fd is not None and j == fd:
                    flip += 1

    return {
        "positions": int(n * t),
        "compared": compared,
        "disagree": disagree,
        "first_divergences": first,
        "flip_at_first_divergence": flip,
        "oracle_reproduced_ours": reproduced,
        "path_dependent": disagree > 0,
    }


def greedy_decode(llm, prompt_ids, max_tokens):
    """The path `greedy_ids.npy` was captured through."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_tokens)
    return [int(x) for x in llm.generate(
        {"prompt_token_ids": prompt_ids}, sp)[0].outputs[0].token_ids]


def teacher_forced_argmax(llm, prompt_ids, our_ids, topk):
    """The path `neartie_gap_mnats.npy` was measured through."""
    from vllm import SamplingParams

    full = list(prompt_ids) + [int(x) for x in our_ids]
    sp = SamplingParams(temperature=0.0, max_tokens=1, prompt_logprobs=topk)
    plp = llm.generate({"prompt_token_ids": full}, sp)[0].prompt_logprobs
    prompt_len = len(prompt_ids)
    named = []
    for j in range(len(our_ids)):
        pos = prompt_len + j
        d = (plp[pos] if pos < len(plp) else None) or {}
        if not d:
            named.append(NO_DISTRIBUTION)
            continue
        named.append(int(max(d, key=lambda k: d[k].logprob)))
    return named


def build_llm(args, max_num_seqs):
    from vllm import LLM

    kwargs = dict(model=args.model, dtype="bfloat16",
                  enforce_eager=not args.cudagraph,
                  trust_remote_code=True,
                  gpu_memory_utilization=args.gpu_mem_util,
                  max_model_len=args.max_model_len,
                  max_num_batched_tokens=args.max_num_batched_tokens,
                  max_num_seqs=max_num_seqs)
    if args.moe_backend:
        kwargs["moe_backend"] = args.moe_backend
    return LLM(**kwargs)


def _rows_as_int(rows):
    return [[int(v) for v in row] for row in rows]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--golden-dir", required=True)
    ap.add_argument("--max-num-seqs", default="1,4",
                    help="comma-separated batch shapes; each is a path")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--gpu-mem-util", type=float, default=0.72)
    ap.add_argument("--max-model-len", type=int, default=2048)
    ap.add_argument("--max-num-batched-tokens", type=int, default=2048)
    ap.add_argument("--moe-backend", default="triton")
    ap.add_argument("--cudagraph", action="store_true",
                    help="drop enforce_eager; the committed capture was eager")
    args = ap.parse_args()

    greedy_golden = np.load(os.path.join(args.golden_dir, "greedy_ids.npy"))
    our = np.fromfile(os.path.join(args.golden_dir, "our_ids.i32"),
                      dtype="<i4").reshape(greedy_golden.shape)
    prompts = [np.fromfile(
        os.path.join(args.golden_dir, f"p{i}_prompt.i32"), dtype="<i4").tolist()
        for i in range(greedy_golden.shape[0])]
    t = greedy_golden.shape[1]
    golden_rows = _rows_as_int(greedy_golden)

    per_shape = {}
    for raw in args.max_num_seqs.split(","):
        shape = int(raw)
        print(f"=== max_num_seqs={shape} "
              f"(enforce_eager={not args.cudagraph}) ===")
        llm = build_llm(args, shape)
        greedy = [greedy_decode(llm, p, t) for p in prompts]
        teacher = [teacher_forced_argmax(llm, p, our[i], args.topk)
                   for i, p in enumerate(prompts)]
        per_shape[shape] = {"greedy": greedy, "teacher_argmax": teacher}
        del llm

    report = {"model": args.model, "comparisons": {}}
    for shape, row in sorted(per_shape.items()):
        rep = compare_paths(golden_rows, row["teacher_argmax"], our)
        rep["reproduced_goldens"] = _rows_as_int(row["greedy"]) == golden_rows
        report["comparisons"][f"decode_vs_prefill@seqs={shape}"] = rep

    shapes = sorted(per_shape)
    for a_i, b_i in zip(shapes, shapes[1:]):
        cross = compare_paths(per_shape[a_i]["greedy"],
                              per_shape[b_i]["greedy"], our)
        cross["reproduced_goldens"] = (
            _rows_as_int(per_shape[b_i]["greedy"]) == golden_rows)
        report["comparisons"][f"decode@{a_i}_vs_decode@{b_i}"] = cross

    dependent = any(r["path_dependent"] for r in report["comparisons"].values())
    reproducible = all(r.get("reproduced_goldens", True)
                       for r in report["comparisons"].values())
    report["verdict"] = ("ORACLE PATH-DEPENDENT" if dependent
                         else "ORACLE PATH-STABLE")
    report["goldens_reproduced"] = reproducible

    for name, rep in sorted(report["comparisons"].items()):
        print(f"  {name}: compared={rep['compared']} "
              f"disagree={rep['disagree']} "
              f"flip_at_first_divergence={rep['flip_at_first_divergence']} "
              f"oracle_reproduced_ours={rep['oracle_reproduced_ours']}")
    print(f"=== {report['verdict']}; greedy goldens reproduced: "
          f"{reproducible} ===")

    out = os.path.join(args.golden_dir, REPORT_NAME)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, sort_keys=True)
        fh.write("\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
