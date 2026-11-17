#!/usr/bin/env python3
"""Compare VT_B70_QUALITY runs of the same checkpoint at identical prefixes.

Usage: b70_compare_probabilities.py REFERENCE_DIRECTORY CANDIDATE_DIRECTORY
Each directory comes from VT_DUMP_LOGITS + test_xpu_qwen_checkpoint. No third
party dependencies. Thresholds are fixed before qualification, not fitted to
observed activation/logit differences. This small regression corpus does not
establish general benchmark quality or parity with a differently quantized model.
"""
import argparse
import array
import json
import math
from pathlib import Path
import re
import sys

VOCAB = 248320
MAX_KL = 0.01  # nats, reference || candidate
MAX_TV = 0.02


def read_dump(directory, case):
    path = directory / case["logits"]
    expected = case.get("teacher_forced", {}).get(
        "planned_steps", case["response"]["usage"]["completion_tokens"])
    assert case["response"]["usage"]["completion_tokens"] == expected, path
    assert 0 < expected <= case["request"]["max_tokens"], path
    values = array.array("f")
    with path.open("rb") as stream:
        values.frombytes(stream.read())
    if sys.byteorder != "little":
        values.byteswap()
    assert len(values) == expected * VOCAB, (path, len(values) // VOCAB, expected)
    assert all(math.isfinite(x) for x in values), path
    ids = [tuple(map(int, line.split())) for line in path.with_suffix(".ids.txt").read_text().splitlines()]
    assert len(ids) == expected, (path, len(ids), expected)
    rows = []
    for step, (recorded_step, best) in enumerate(ids):
        row = values[step * VOCAB:(step + 1) * VOCAB]
        assert recorded_step == step and max(range(VOCAB), key=row.__getitem__) == best, path
        rows.append((row, best))
    return rows


def check_forced_metadata(reference, candidate):
    required = ("planned_steps", "forced_token_ids", "prefix_hashes", "prefix_hash_algorithm", "position_ids",
                "chunk_schedule", "state_reset", "checkpoint_revision")
    left, right = reference["teacher_forced"], candidate["teacher_forced"]
    assert all(key in left and key in right and left[key] == right[key] for key in required)
    steps = left["planned_steps"]
    assert isinstance(steps, int) and steps > 0
    assert len(left["forced_token_ids"]) == len(left["prefix_hashes"]) == len(left["position_ids"]) == steps
    assert all(isinstance(token, int) and 0 <= token < VOCAB for token in left["forced_token_ids"])
    assert all(isinstance(position, int) and position >= 0 for position in left["position_ids"])
    assert all(re.fullmatch(r"[0-9a-f]{64}", value) for value in left["prefix_hashes"])
    assert left["prefix_hash_algorithm"] == "blake3-token-ids-le32-v1"
    assert isinstance(left["state_reset"], bool) and left["state_reset"]
    assert left["checkpoint_revision"]
    assert left["chunk_schedule"] and all(isinstance(chunk, int) and chunk > 0
                                          for chunk in left["chunk_schedule"])
    return steps


def distances(reference, candidate):
    def normalized(row):
        peak = max(row)
        weights = [math.exp(x - peak) for x in row]
        total = math.fsum(weights)
        return [w / total for w in weights], peak + math.log(total)

    p, log_zp = normalized(reference)
    q, log_zq = normalized(candidate)
    kl = math.fsum(prob * ((x - log_zp) - (y - log_zq))
                   for prob, x, y in zip(p, reference, candidate))
    tv = 0.5 * math.fsum(abs(a - b) for a, b in zip(p, q))
    return max(0.0, kl), tv


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--long-context", action="store_true", help="compare the separate one-case long retrieval corpus")
    parser.add_argument("--teacher-forced", action="store_true", help="compare all 32 fixed steps of the 24-case corpus")
    args = parser.parse_args()
    assert not (args.long_context and args.teacher_forced)
    reference = json.loads((args.reference / "results.json").read_text())
    candidate = json.loads((args.candidate / "results.json").read_text())
    expected_cases = 1 if args.long_context else 24 if args.teacher_forced else 8
    assert len(reference) == len(candidate) == expected_cases
    if args.long_context:
        assert reference[0]["name"] == candidate[0]["name"] == "long_retrieval"
        assert reference[0]["response"]["usage"]["prompt_tokens"] >= 4096
    failed_distributions = 0
    failed_answers = 0
    compared = 0
    for ref, got in zip(reference, candidate):
        assert ref["name"] == got["name"] and ref["request"] == got["request"]
        assert ref["response"]["usage"]["prompt_tokens"] == got["response"]["usage"]["prompt_tokens"]
        if args.teacher_forced:
            assert check_forced_metadata(ref, got) == 32
        left, right = read_dump(args.reference, ref), read_dump(args.candidate, got)
        count, worst_kl, worst_tv = 0, 0.0, 0.0
        for (p, p_best), (q, q_best) in zip(left, right):
            kl, tv = distances(p, q)
            worst_kl, worst_tv = max(worst_kl, kl), max(worst_tv, tv)
            failed_distributions += int(kl > MAX_KL or tv > MAX_TV)
            count += 1
            # The current distribution has the same conditioning prefix. Once
            # greedy choices diverge, subsequent rows no longer do; don't claim
            # that comparing them measures numerical drift at identical input.
            if not args.teacher_forced and p_best != q_best:
                break
        assert count > 0
        same_answer = ref["answer"] == got["answer"]
        if not args.teacher_forced:
            failed_answers += int(not same_answer)
        compared += count
        print(json.dumps({"case": ref["name"], "comparable_steps": count,
                          "reference_steps": len(left), "candidate_steps": len(right),
                          "max_kl": worst_kl, "max_tv": worst_tv,
                          "same_answer": same_answer}), flush=True)
    print(json.dumps({"cases": len(reference), "compared_distributions": compared,
                      "failed_distributions": failed_distributions, "failed_answers": failed_answers,
                      "max_allowed_kl": MAX_KL,
                      "max_allowed_tv": MAX_TV}), flush=True)
    return int(failed_distributions != 0 or failed_answers != 0)


if __name__ == "__main__":
    sys.exit(main())
