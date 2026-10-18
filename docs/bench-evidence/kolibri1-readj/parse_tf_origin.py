#!/usr/bin/env python3
"""Derive the L0-drift-origin table from the SCRATCH dbg TF prompt logs.

Input: the tf-p{1,5,4}.log(.gz) teacher-forced legs of
`test_kolibri1_tt_b2ii --test-case='*SCRATCH dbg TF prompt*'`
(VT_KOLIBRI1_TT_STAGE_DUMP=1), CPU and TT arms in one process, as committed
in this directory. Output: tf_origin.json, one row per teacher-forced
forward on the three instrumented prompts.

STEP-LABEL MAPPING (the walk in test_kolibri1_tt_b2ii.cpp, SCRATCH dbg TF
prompt): walk iteration i feeds input_ids[i] for i < input_len and
generated_ids[i - input_len] afterwards, and its logits sit at context
position i, so they predict generated_ids[i - input_len + 1]. The log's
raw `TFstep` counter prints d = i - input_len BEFORE that +1, therefore
    raw `TFstep N`  ==  DECODE position N + 1,
and the last prompt token (raw TFstep -1) produces decode-position-0
logits. This script labels every row with the decode position, matching
the gate's decode-step numbering (the `HARD flip ... step d` lines, where
d is already the decode position).

Usage: python3 parse_tf_origin.py [path/to/repo/root]
Writes tf_origin.json next to this file.
"""
import gzip
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "..", "..")
h = re.compile(r"\[STAGE\]\[(CPU|TT)\] L(\d+) (\S+) sum=(\S+) .*?hash=(\w+)")
LEG_STEPS = (("p1", 1, 25), ("p5", 5, 23), ("p4", 4, 10))


def open_log(which):
    for cand in (f"tf-{which}.log", f"tf-{which}.log.gz"):
        p = os.path.join(HERE, cand)
        if os.path.exists(p):
            return gzip.open(p, "rt", errors="replace") if cand.endswith(".gz") \
                else open(p, errors="replace")
    raise SystemExit(f"no tf-{which}.log(.gz) under {HERE}")


def main():
    ps = json.load(open(os.path.join(
        ROOT, "tests", "vllm", "models", "kolibri1_goldens.json")))["prompts"]
    res = {}
    for which, pi, steps in LEG_STEPS:
        plen = len(ps[pi]["input_ids"])
        segs, cur = [], None
        for l in open_log(which):
            m = h.search(l)
            if not m:
                continue
            arm, L, tag, su, hs = m.groups()
            if tag == "dh2" and L == "0":
                cur = {"arm": arm, "stages": {}}
                segs.append(cur)
            if cur is None:
                continue
            cur["stages"].setdefault((int(L), tag), (hs, float(su)))
        pairs = [(segs[i], segs[i + 1]) for i in range(0, len(segs) - 1, 2)]
        rows = []
        # pair index fi == walk iteration i. Decode position 0 is the last
        # prompt-token forward (fi = plen - 1, raw TFstep -1); every
        # teacher-forced forward fi >= plen is decode position fi-plen+1.
        for fi in range(plen - 1, min(plen + steps, len(pairs))):
            c, t = pairs[fi]
            if (0, "dh2") not in c["stages"] or (0, "dh2") not in t["stages"]:
                continue
            dh2c = c["stages"][(0, "dh2")]
            dh2t = t["stages"][(0, "dh2")]
            rel = abs(dh2c[1] - dh2t[1]) / abs(dh2c[1]) if dh2c[1] else 0
            idflips = 0
            lid = []
            for k in c["stages"]:
                if k[1].startswith("moe_ex"):
                    if c["stages"][k][0] != t["stages"].get(k, ("x",))[0]:
                        idflips += 1
                        lid.append(k[0])
            rows.append({
                "decode_position": fi - plen + 1,
                "raw_tfstep_label": fi - plen,  # the pre-fix log label
                "l0_dh2_rel_drift_pct": round(rel * 100, 2),
                "l0_dh2_hash_cpu": dh2c[0],
                "l0_dh2_hash_tt": dh2t[0],
                "expert_set_flips": idflips,
                "first_flip_layers": lid[:6],
            })
        res[f"p{pi}"] = rows
        diverged = all(r["l0_dh2_hash_cpu"] != r["l0_dh2_hash_tt"] for r in rows)
        print(f"prompt {pi} ({ps[pi]['prompt'][:35]}): "
              f"{len(rows)} decode positions, L0 dh2 diverges at every "
              f"position: {diverged}")
    out = os.path.join(HERE, "tf_origin.json")
    json.dump(res, open(out, "w"), indent=1)
    print("wrote", out)


if __name__ == "__main__":
    main()
