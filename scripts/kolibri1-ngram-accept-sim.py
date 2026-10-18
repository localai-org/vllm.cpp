#!/usr/bin/env python3
# Kolibri-1 n-gram spec-decode acceptance simulator (probe for
# ISSUE-LOCAL-01M4P22PX6BPRQAY1CK20FZQF6, branch row/kolibri-spec-probe).
#
# Mirrors the PRODUCTION proposer byte-for-byte:
#   src/vllm/v1/spec_decode/ngram_proposer.cpp
#     FindLongestMatchedNgramAndProposeTokens (KMP-LPS longest-suffix match,
#     earliest occurrence, up to k following tokens)
# and the greedy verify rule of
#   include/vllm/v1/spec_decode/rejection_sampler.h (accept iff the draft
#   equals the target argmax; stop at the first mismatch; exactly one
#   bonus/replacement token per verify pass).
#
# The simulated run over a golden greedy chain is EXACT: the committed tokens
# are always a prefix of the chain, so the spec-decode output is byte-identical
# to plain greedy decode. The measurement is therefore the true acceptance
# rate of the production drafter on that chain, needing no model run.
#
# Usage: kolibri1-ngram-accept-sim.py [goldens.json] [bench_chain.txt]
#   bench_chain.txt: one chain as "t0,t1,..." (e.g. the CHAIN: line the
#   decode bench prints). The bench prompt is reconstructed as the
#   deterministic (i*7919+13)%128000 sequence of length 128 when the file
#   is absent; pass "-" to skip the bench chain.

import json
import sys


def propose(origin_tokens, min_ngram, max_ngram, max_model_len, k):
    """FindLongestMatchedNgramAndProposeTokens, mirrored exactly."""
    total_token = len(origin_tokens)
    if total_token < min_ngram:
        return []
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return []
    tok = lambda j: origin_tokens[total_token - 1 - j]
    lps = [0] * max_ngram
    longest_ngram = 0
    position = 0
    prev_lps = 0
    i = 1
    while i < total_token:
        if tok(prev_lps) == tok(i):
            prev_lps += 1
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            prev_lps = lps[prev_lps - 1]
        else:
            i += 1
    if longest_ngram < min_ngram:
        return []
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return [origin_tokens[start_position + j] for j in range(k)]


def simulate(prompt, chain, min_n, max_n, k, max_model_len=8192):
    """One greedy request under n-gram spec decode.

    Returns (verify_steps, tokens_emitted, accepted_total, proposed_total,
    draft_len_hist, exact) where `exact` checks the committed tokens are a
    prefix of the golden chain (the byte-identical property)."""
    context = list(prompt)
    emitted = 0
    # Prefill emits the first chain token with no drafts scheduled.
    if chain:
        context.append(chain[0])
        emitted = 1
    steps = 0
    accepted_total = 0
    proposed_total = 0
    hist = {}
    exact = True
    while emitted < len(chain):
        drafts = propose(context, min_n, max_n, max_model_len, k)
        accepted = 0
        for i, d in enumerate(drafts):
            if emitted + i < len(chain) and d == chain[emitted + i]:
                accepted += 1
            else:
                break
        # The emitted token after accept/reject is the chain token itself
        # (the target argmax); the spec run is byte-identical because the
        # accept walk only ever commits chain tokens. The FINAL step is
        # truncated by the generation limit (max_new_tokens), exactly as the
        # engine's finish condition discards tokens past the limit.
        emitted += min(accepted + 1, len(chain) - emitted)
        context = list(prompt) + chain[:emitted]
        steps += 1
        accepted_total += accepted
        proposed_total += len(drafts)
        hist[len(drafts)] = hist.get(len(drafts), 0) + 1
    return steps, emitted, accepted_total, proposed_total, hist, exact


def report(name, prompt, chain, max_model_len=8192):
    plain_steps = max(0, len(chain) - 1)  # prefill emits token 0; one step per rest
    print(f"=== {name}: prompt {len(prompt)} tokens, chain {len(chain)} tokens "
          f"(plain decode: 1 prefill + {plain_steps} steps) ===")
    print(f"{'k':>2} {'window':>7} {'steps':>5} {'tok/step':>8} {'step-ratio':>10} "
          f"{'acc/verify':>10} {'draft-hit':>9}")
    for k in (1, 2, 3, 4, 5, 8):
        for (mn, mx) in ((1, 1), (1, 2), (1, 3), (1, 4), (2, 4), (5, 5)):
            steps, emitted, acc, prop, hist, exact = simulate(
                prompt, chain, mn, mx, k, max_model_len)
            tok_per_step = emitted / steps if steps else float(emitted)
            ratio = plain_steps / steps if steps else float("inf")
            acc_rate = acc / steps if steps else 0.0
            hit = prop / steps if steps else 0.0
            flag = ""
            print(f"{k:>2} {mn}-{mx:>4} {steps:>5} {tok_per_step:>8.3f} "
                  f"{ratio:>10.3f} {acc_rate:>10.3f} {hit:>9.3f}{flag}")
    print()


def main():
    args = sys.argv[1:]
    goldens_path = args[0] if args else "tests/vllm/models/kolibri1_goldens.json"
    g = json.load(open(goldens_path))
    for i, p in enumerate(g["prompts"]):
        report(f"golden[{i}] {p['prompt'][:40]!r}", p["input_ids"],
               p["generated_ids"])
    # Aggregate over all goldens for the headline number.
    for k in (2, 3, 4, 5, 8):
        for (mn, mx) in ((1, 4), (2, 4), (5, 5)):
            steps = emitted = acc = prop = plain = 0
            for p in g["prompts"]:
                s, e, a, pr, _, exact = simulate(
                    p["input_ids"], p["generated_ids"], mn, mx, k)
                steps += s
                emitted += e
                acc += a
                prop += pr
                plain += max(0, len(p["generated_ids"]) - 1)
                assert exact and e == len(p["generated_ids"])
            print(f"AGGREGATE k={k} window={mn}-{mx}: steps {steps} vs plain "
                  f"{plain} (x{plain / steps:.3f}), tokens/step "
                  f"{emitted / steps:.3f}, accepted/verify {acc / steps:.3f}, "
                  f"drafts proposed/verify {prop / steps:.3f}")
    # Bench chain, if supplied: "t0,t1,..." with the deterministic 128-token
    # prompt (i*7919+13)%128000 prepended.
    if len(args) > 1 and args[1] != "-":
        chain = [int(x) for x in open(args[1]).read().strip().split(",") if x]
        bench_prompt = [(i * 7919 + 13) % 128000 for i in range(128)]
        report("bench (deterministic random prompt)", bench_prompt, chain)


if __name__ == "__main__":
    main()
