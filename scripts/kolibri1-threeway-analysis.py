#!/usr/bin/env python3
"""Kolibri-1 three-way numerics analysis (row/kolibri-three-way).

Compares OUR per-position logit rows (teacher-forced on the PRIMARY oracle's
chains, dumped by tests/vllm/models/test_kolibri1_threeway_capture.cpp behind
VT_THREEWAY_DUMP_DIR) against the primary oracle's per-position top-100
logprob rows (tests/vllm/models/kolibri1_primary_topk_rows.json).

Both sides score position k under the IDENTICAL prefix prompt+gen[0..k-1], so
every distance below is a numerics distance, not a chain-divergence artifact.

Per aligned position we compute, after log-softmax of our raw logit row:
  rank        the primary's emitted token's rank in OUR full-vocab logprob
  gap_nats    our logprob(emitted) - primary logprob(emitted)
  cos         cosine similarity over the primary's top-100 support
  kl_nats     KL(primary || ours), both renormalized on the top-100 support
              (the primary publishes no tail; this is a truncated support)

Usage: kolibri1-threeway-analysis.py DUMP_DIR [DUMP_DIR ...]
Each DUMP_DIR must hold manifest.json + promptN.bin and carry a distinguishing
name (the mode label). Prints the per-mode table and the depth-growth table.
"""
import json
import math
import struct
import sys

VOCAB = 128000


def log_softmax(x):
    m = max(x)
    s = sum(math.exp(v - m) for v in x)
    return m + math.log(s)


def load_dump(dump_dir):
    with open(f"{dump_dir}/manifest.json") as f:
        man = json.load(f)
    prompts = []
    for pj in man["prompts"]:
        rows = []
        with open(f"{dump_dir}/{pj['bin']}", "rb") as f:
            raw = f.read()
        n = len(raw) // (VOCAB * 4)
        for i in range(n):
            rows.append(struct.unpack_from(f"<{VOCAB}f", raw, i * VOCAB * 4))
        prompts.append({"gen_index": pj["gen_index"], "rows": rows})
    return prompts


def analyze(dump_dir, fixture_path):
    with open(fixture_path) as f:
        fixture = json.load(f)
    prompts = load_dump(dump_dir)
    per_pos = []
    for pi, (pj, mine) in enumerate(zip(fixture["prompts"], prompts)):
        frows = {r["gen_index"]: r for r in pj["rows"]}
        for k, g in enumerate(mine["gen_index"]):
            ref = frows[g]
            row = mine["rows"][k]
            lse = log_softmax(row)
            top = ref["top"]
            tids = [t for t, _ in top]
            tvals = [v for _, v in top]
            # rank of the emitted token in OUR full-vocab logprob
            etok = ref["token"]
            our_lp_e = row[etok] - lse
            rank = 1 + sum(1 for v in row if v > row[etok])
            # cosine over the primary's top-100 support
            ours = [row[t] - lse for t in tids]
            dot = sum(a * b for a, b in zip(ours, tvals))
            na = math.sqrt(sum(a * a for a in ours))
            nb = math.sqrt(sum(b * b for b in tvals))
            cos = dot / (na * nb) if na > 0 and nb > 0 else float("nan")
            # truncated-support KL(primary || ours)
            m_p = max(tvals)
            z_p = sum(math.exp(v - m_p) for v in tvals)
            m_q = max(ours)
            z_q = sum(math.exp(v - m_q) for v in ours)
            kl = 0.0
            for p_val, q_val in zip(tvals, ours):
                p = math.exp(p_val - m_p) / z_p
                q = math.exp(q_val - m_q) / z_q
                kl += p * (math.log(p) - math.log(q))
            per_pos.append({
                "prompt": pi, "gen_index": g, "rank": rank,
                "gap_nats": our_lp_e - ref["logprob_of_token"],
                "cos": cos, "kl_nats": kl,
                "our_top1": max(row), "token": etok,
            })
    return per_pos


def agg(rows, key):
    vals = [r[key] for r in rows]
    vals.sort()
    n = len(vals)
    mean = sum(vals) / n
    med = vals[n // 2]
    return mean, med


def main():
    fixture_path = "tests/vllm/models/kolibri1_primary_topk_rows.json"
    modes = {}
    for d in sys.argv[1:]:
        modes[d.rstrip("/").split("/")[-1]] = analyze(d, fixture_path)

    print("== PER-MODE AGGREGATE (202 aligned positions) ==")
    print(f"{'mode':<10} {'top1%':>6} {'top5%':>6} {'top100%':>8} "
          f"{'gap mean':>9} {'gap med':>8} {'cos mean':>9} {'kl mean':>8} {'kl med':>8}")
    for mode, rows in modes.items():
        n = len(rows)
        top1 = sum(1 for r in rows if r["rank"] == 1) / n * 100
        top5 = sum(1 for r in rows if r["rank"] <= 5) / n * 100
        top100 = sum(1 for r in rows if r["rank"] <= 100) / n * 100
        gm, gmed = agg(rows, "gap_nats")
        cm, _ = agg(rows, "cos")
        km, kmed = agg(rows, "kl_nats")
        print(f"{mode:<10} {top1:6.1f} {top5:6.1f} {top100:8.1f} "
              f"{gm:9.3f} {gmed:8.3f} {cm:9.4f} {km:8.3f} {kmed:8.3f}")

    print("\n== DEPTH GROWTH (mean, per bin) ==")
    bins = [("0-5", 0, 5), ("6-15", 6, 15), ("16-31", 16, 31)]
    print(f"{'mode':<10} {'bin':>6} {'n':>4} {'top1%':>6} {'gap mean':>9} "
          f"{'cos mean':>9} {'kl mean':>8}")
    for mode, rows in modes.items():
        for name, lo, hi in bins:
            sel = [r for r in rows if lo <= r["gen_index"] <= hi]
            if not sel:
                continue
            top1 = sum(1 for r in sel if r["rank"] == 1) / len(sel) * 100
            gm, _ = agg(sel, "gap_nats")
            cm, _ = agg(sel, "cos")
            km, _ = agg(sel, "kl_nats")
            print(f"{mode:<10} {name:>6} {len(sel):>4} {top1:6.1f} "
                  f"{gm:9.3f} {cm:9.4f} {km:8.3f}")

    print("\n== PER-PROMPT (mean gap / mean KL) ==")
    for mode, rows in modes.items():
        by_p = {}
        for r in rows:
            by_p.setdefault(r["prompt"], []).append(r)
        s = " ".join(
            f"p{p}:{agg(v, 'gap_nats')[0]:.2f}/{agg(v, 'kl_nats')[0]:.2f}"
            for p, v in sorted(by_p.items()))
        print(f"{mode:<10} {s}")

    # full detail for the record
    out = {mode: rows for mode, rows in modes.items()}
    with open("threeway_detail.json", "w") as f:
        json.dump(out, f)
    print("\nwrote threeway_detail.json")


if __name__ == "__main__":
    main()
