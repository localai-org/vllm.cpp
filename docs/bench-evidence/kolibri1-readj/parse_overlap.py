#!/usr/bin/env python3
"""Derive the CPU-vs-TT top-6-of-384 expert-set overlap table (doc Q2).

Input: the tf-p{1,5,4}.log(.gz) teacher-forced legs committed in this
directory (the `[STAGE] ... route ... ids=[...]` lines). Labels: rows span
the WHOLE one-token-per-forward walk (prompt tokens first, then the
teacher-forced chain), so a row index is a walk position, not a decode
position — the aggregate table is not step-labelled and needs no offset.
The single-step example quoted in the doc is the walk's first forward
(prompt token 1).

Usage: python3 parse_overlap.py [repo/root]
"""
import gzip
import json
import os
import re
import statistics as st
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..")
r = re.compile(r"\[STAGE\]\[(CPU|TT)\] L(\d+) route tok=(\d+) logit_sum=(-?[0-9.eE+-]+) ids=\[([0-9,]+)\]")
LEGS = (("p1", 1), ("p5", 5), ("p4", 4))


def open_log(which):
    p = os.path.join(HERE, f"tf-{which}.log.gz")
    if os.path.exists(p):
        return gzip.open(p, "rt", errors="replace")
    return open(os.path.join(HERE, f"tf-{which}.log"), errors="replace")


def main():
    ps = json.load(open(os.path.join(
        ROOT, "tests", "vllm", "models", "kolibri1_goldens.json")))["prompts"]
    for which, pi in LEGS:
        rows = []
        for l in open_log(which):
            m = r.search(l)
            if m:
                rows.append((m.group(1), int(m.group(2)), float(m.group(4)),
                             set(map(int, m.group(5).split(",")))))
        cpu = [x for x in rows if x[0] == "CPU"]
        tt = [x for x in rows if x[0] == "TT"]
        ovl, noise, per = [], [], {}
        for c, t in zip(cpu, tt):
            o = len(c[3] & t[3])
            ovl.append(o)
            per.setdefault(c[1], []).append(o)
            noise.append(abs(c[2] - t[2]) / abs(c[2]) if c[2] else 0)
        print(f"prompt {pi} ({ps[pi]['prompt'][:35]}): pairs={len(ovl)} "
              f"mean_overlap={st.mean(ovl):.2f}/6 "
              f"full_match={sum(1 for o in ovl if o == 6) / len(ovl) * 100:.1f}% "
              f"median_logitsum_reldiff={st.median(noise) * 100:.2f}%")
        print("  per-layer mean overlap:",
              {L: round(st.mean(v), 1) for L, v in sorted(per.items())})


if __name__ == "__main__":
    main()
