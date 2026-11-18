#!/usr/bin/env python3
"""Compare two `VT_Q4EXP_LOGITS` captures by ID UNION, and settle the KL question.

WHY A SECOND DIFFER. `q4exp-layerfp-diff.py` compares the LAYER fingerprints.
`VT_Q4EXP_LOGITS` (PR #3569) is a different line -- `q4logit step= rows= n= top=
top1= vmax= vmin= sumabs= proj=<8> pairs=id:val,...` -- with a different failure
mode, because its payload is a RANKED top-k rather than one row per layer.

A POSITIONAL COMPARISON MEASURES THE SORT, NOT THE MODEL. `pairs=` is ordered by
DESCENDING value (`std::partial_sort` in `Qwen4ExpLogitsTap`), so comparing the
value at rank r of one arm with the value at rank r of the other compares two
different tokens whenever anything crosses the cut, or whenever two logits tie
and the sort settles them differently. One token moving across the cut reds every
rank below it: k apparent differences where the tensors have one. This tool keys
on the ID instead:

    union   = ids in EITHER arm's top-k
    shared  = ids in BOTH
    only_a  = ids base's cut kept and other's left out
    only_b  = the converse

and every difference statistic is computed over `shared`, i.e. over ids that are
the same token in both arms. The dropped/appeared ids are reported separately,
with their values, because that is what a cut boundary looks like: a token whose
value in one arm sits just above the k-th value of the other. A positional
differ cannot express `shared` at all, so it cannot tell whether two arms
disagree about a VALUE or about WHICH TOKENS MADE THE CUT.

IT REFUSES ARMS THAT ARE NOT COMPARABLE. `n` (vocabulary), `top` (k) and `rows`
must agree between the arms for every step. `n` because an id is only the same
token in a model with the same vocabulary; `top` because a union over different
k's is not the same cut and the `only_*` sets would be artifacts of the budget
rather than of the model; `rows` because `proj` is summed over the WHOLE buffer
while the top-k comes from row 0 - two different row counts mean the projection
comparison is between different quantities.

IT REFUSES A LINE THAT DISAGREES WITH ITSELF. The tap prints `top=` and then
exactly `top` pairs; `top1=` is `pairs[0]`'s id and the pairs descend. A line
with fewer pairs than it declares is a TRUNCATED capture -- the writer died or
stderr was cut -- and a truncated capture compares short, which reads like
agreement. Same reason as the layer differ's `taps=N END` check: an empty or
short comparison is not a passing one.

═══════════════════════════════════════════════════════════════════════════════
THE KL QUESTION, SETTLED: TOP-K CANNOT SUPPORT AN UNBIASED KL. NOT "NEEDS CARE",
CANNOT.
═══════════════════════════════════════════════════════════════════════════════

The tempting move is to renormalise both arms over the union and call the result
KL. It is not a truthful KL, and no rescaling makes it one. Write p, q for the
full distributions over the n ids, U for the union of the two top-k cuts,
c_p = sum_{i in U} p_i and c_q = sum_{i in U} q_i. Then

    KL(p_U || q_U) = (1/c_p) * sum_{i in U} p_i log(p_i/q_i) + log(c_q/c_p)

is the KL between DIFFERENT measures on a different support. Three consequences,
in increasing order of how fatal they are:

1. It is a different functional, whose value depends on k, on n, and on which
   ids both cuts happened to keep. Two runs of the metric on the same arms with
   different `VT_Q4EXP_LOGITS` values are not comparable, so it cannot gate a
   change: the gate's own budget moves the number.

2. The unobserved tail is not a small correction with a sign. Expanding the true
   quantity,

       KL(p||q) = sum_{i in U} p_i log(p_i/q_i) + sum_{i not in U} p_i log(p_i/q_i)

   and each omitted term is NEGATIVE where q_i > p_i. The partial sum is
   therefore neither an upper nor a lower bound on KL(p||q): truncation can push
   the number either way, by an amount set by mass this line never prints.

3. It is DEGENERATE, exactly and unrepairably. At k = 1 with both arms keeping
   the same argmax, U is one id, p_U = q_U = delta on it, c_p = c_q, and
   KL(p_U||q_U) = 0 EXACTLY -- whatever the two tails do. A metric that reads
   exactly 0 on a divergence it cannot see is worse than a noisy one: it is a
   green light wired to nothing. Scaling by c_p, or by k, or by n, multiplies a
   zero.

   Supplementing the line with a full-row `logsumexp` per arm does NOT rescue it.
   That fixes only the normalisers c_p, c_q; the tail sum in (2) needs the
   individual tail probabilities of BOTH arms against each other, which a top-k
   line does not carry and no aggregate over the tail reconstructs (log p_i/q_i
   is not linear in anything a projection can sum).

SO THIS TOOL PRINTS NO KL, AND REFUSES TO BE ASKED FOR ONE. `--kl` is a REFUSAL,
not an option: it exits 2 and prints the argument above, naming what a KL would
actually require -- the FULL row for both arms (an `n`-float line, ~600 KB at
n = 151936, which is an offline gate's budget and not a server tap's), so that
the exact softmax is computable on both sides. That is a change to the tap and
needs its own spec, test and review; it is OWED, not done here.

WHAT THIS TOOL PRINTS INSTEAD, all of it exact for the ids it can see:

  top_dmax    max |v_a - v_b| over SHARED ids. A genuine difference of VALUES,
              cannot cancel, and non-zero PROVES the arms' logits differ --
              but it samples at most k of n ids, so it is a witness and a floor,
              never a magnitude. Zero proves nothing.
  top_dmed    the median of the same, so one outlier rank does not set the story.
  top1_flip   whether the two arms rank a different token first, with both ids.
  max_rank_disp  the largest |rank_a - rank_b| over shared ids -- how far the
              ordering moved for tokens both arms kept.
  shared/jaccard/dropped/appeared  the cut structure: agreement here proves the
              two arms rank the SAME SET, which `top_dmax` alone does not.
  rel_proj    the sign-SENSITIVE whole-buffer projection difference, exactly as
              in `q4exp-layerfp-diff.py`, and for the same reason `sumabs` is
              useless: summed |x| cancels a zero-mean perturbation at
              O(sqrt(n)) and under-reports a real divergence by a median 75-140x
              at this tap's sizes. `rel_sumabs` is printed for continuity with
              the evidence rows that quote it -- read ITS docstring there before
              reading a ratio of it, because at a fixed true divergence two
              readings of it differ by a median 2.1x and by 24x at p95.
"""

from __future__ import annotations

import argparse
import math
import re
import sys

# `q4logit step=%lld rows=%lld n=%lld top=%lld top1=%lld vmax=%.9g vmin=%.9g
#  sumabs=%.9g proj=%.9g,... (8) pairs=<id>:<val>,<id>:<val>,...`
_PAIR = re.compile(r"^(\d+):([-+0-9.eE]+)$")
_REQUIRED = ("step", "rows", "n", "top", "top1", "vmax", "vmin", "sumabs", "proj",
             "pairs")
_NUM_PROJ = 8


class ParseError(RuntimeError):
    pass


def _one_line(path, lineno, ln):
    toks = ln.split()
    if not toks or toks[0] != "q4logit":
        return None
    f = {}
    for t in toks[1:]:
        if "=" in t:
            k, v = t.split("=", 1)
            f[k] = v
    missing = [k for k in _REQUIRED if k not in f]
    if missing:
        raise ParseError(
            "%s:%d: q4logit line missing %s. The tap prints all of %s; a missing "
            "field means the format changed or the capture was cut mid-line."
            % (path, lineno, ", ".join(missing), ", ".join(_REQUIRED)))

    def as_int(key):
        try:
            return int(f[key])
        except ValueError:
            raise ParseError("%s:%d: %s=%r is not an integer." % (path, lineno, key, f[key]))

    step, rows, n, top, top1 = (as_int(k) for k in ("step", "rows", "n", "top", "top1"))
    if n <= 0 or top <= 0 or rows <= 0:
        raise ParseError("%s:%d: n=%d top=%d rows=%d; all three must be positive."
                         % (path, lineno, n, top, rows))
    if top > n:
        raise ParseError("%s:%d: top=%d exceeds n=%d; k is clamped to the vocabulary "
                         "in the tap, so this line is not one the instrument can print."
                         % (path, lineno, top, n))
    try:
        proj = [float(x) for x in f["proj"].split(",")]
    except ValueError:
        raise ParseError("%s:%d: proj=%r is not a float list." % (path, lineno, f["proj"]))
    if len(proj) != _NUM_PROJ:
        raise ParseError("%s:%d: proj has %d values, the tap emits %d. A short proj "
                         "vector is a truncated line, and a truncated comparison reads "
                         "like agreement." % (path, lineno, len(proj), _NUM_PROJ))

    raw = f["pairs"]
    pairs = []
    if raw:
        for item in raw.split(","):
            m = _PAIR.match(item)
            if not m:
                raise ParseError("%s:%d: pair %r is not `id:value`." % (path, lineno, item))
            pairs.append((int(m.group(1)), float(m.group(2))))
    if len(pairs) != top:
        raise ParseError(
            "%s:%d: top=%d but %d pairs. The tap prints exactly `top` pairs (it clamps "
            "with std::min(topk, vocab)), so this line is TRUNCATED: stderr was cut or "
            "the writer died mid-line. A short k compares short, and a short comparison "
            "reads like two agreeing arms." % (path, lineno, top, len(pairs)))
    ids = [i for i, _ in pairs]
    if len(set(ids)) != len(ids):
        dup = sorted({i for i in ids if ids.count(i) > 1})
        raise ParseError("%s:%d: duplicate ids in pairs: %s. partial_sort yields each id "
                         "once; a repeat means the parse or the format is wrong."
                         % (path, lineno, dup))
    if top1 != pairs[0][0]:
        raise ParseError("%s:%d: top1=%d but the first pair is id %d. The tap sets "
                         "`last_top1` from the same partial_sort that fills `pairs`, so "
                         "these two fields cannot disagree in a line the instrument wrote."
                         % (path, lineno, top1, pairs[0][0]))
    vals = [v for _, v in pairs]
    for a, b in zip(vals, vals[1:]):
        if b > a:
            raise ParseError("%s:%d: pairs are not in descending-value order (%r before "
                             "%r). `pairs=` is a sorted top-k; an unsorted line means the "
                             "payload was reassembled, not captured." % (path, lineno, a, b))
    return {"step": step, "rows": rows, "n": n, "top": top, "top1": top1,
            "pairs": pairs, "proj": proj,
            "vmax": float(f["vmax"]), "vmin": float(f["vmin"]),
            "sumabs": float(f["sumabs"])}


def parse(path):
    """Return ({step: row}, [step, ...]). Refuses a duplicate step or an empty file."""
    rows, order = {}, []
    with open(path, encoding="utf-8", errors="replace") as fh:
        for lineno, ln in enumerate(fh, 1):
            r = _one_line(path, lineno, ln)
            if r is None:
                continue
            if r["step"] in rows:
                raise ParseError(
                    "%s:%d: step=%d appears twice. The tap advances `step` on every line "
                    "it prints (`++s.step` per forward), so two lines for one step means "
                    "two captures were concatenated or the budget was reset mid-file. "
                    "Deduplicating one would hide whichever the reader is not seeing."
                    % (path, lineno, r["step"]))
            rows[r["step"]] = r
            order.append(r["step"])
    if not rows:
        raise ParseError(
            "%s: parsed ZERO q4logit rows. An empty comparison prints an empty table, "
            "and an empty table reads like two agreeing arms -- it is not one. Check "
            "that the run set VT_Q4EXP_LOGITS (and VT_Q4EXP_LOGITS_STEPS > 0)." % path)
    return rows, order


def refusable_pairing(a, b):
    """The comparability contract: same vocabulary, same cut, same buffer shape."""
    for key, why in (("n", "an id is only the same token when the vocabulary matches"),
                     ("top", "a union over different k's is a different cut, and "
                             "`only_*` would be an artifact of the budget"),
                     ("rows", "`proj` sums the WHOLE buffer while the top-k is row 0, "
                              "so different row counts compare different quantities")):
        if a[key] != b[key]:
            return ("%s differs between the arms (%d vs %d): %s. Refusing rather than "
                    "reporting a comparison between different things."
                    % (key, a[key], b[key], why))
    return None


def union_stats(a, b):
    """Compare by ID UNION. Position never enters: `shared` is the same token twice."""
    va = dict(a["pairs"])
    vb = dict(b["pairs"])
    ra = {i: r for r, (i, _) in enumerate(a["pairs"])}
    rb = {i: r for r, (i, _) in enumerate(b["pairs"])}
    shared = [i for i, _ in a["pairs"] if i in vb]
    only_a = [(i, va[i]) for i, _ in a["pairs"] if i not in vb]
    only_b = [(i, vb[i]) for i, _ in b["pairs"] if i not in va]
    deltas = [abs(va[i] - vb[i]) for i in shared]
    return {
        "shared": shared,
        "only_a": only_a,
        "only_b": only_b,
        "dmax": max(deltas) if deltas else 0.0,
        "dmed": _median(deltas),
        "max_rank_disp": max((abs(ra[i] - rb[i]) for i in shared), default=0),
        "jaccard": (len(shared) / float(len(shared) + len(only_a) + len(only_b)))
        if (shared or only_a or only_b) else 1.0,
    }


def _median(xs):
    if not xs:
        return 0.0
    s = sorted(xs)
    m = len(s) // 2
    return s[m] if len(s) % 2 else 0.5 * (s[m - 1] + s[m])


def rel_sumabs(a, b):
    """DIFFERENCE OF NORMS. Read `q4exp-layerfp-diff.py` before reading a ratio."""
    x, y = float(a), float(b)
    m = max(abs(x), abs(y))
    return 0.0 if m == 0.0 else abs(x - y) / m


def rel_proj(a, b):
    """DIFFERENCE of fixed-seed random projections: sign-SENSITIVE, cannot cancel."""
    worst = 0.0
    for x, y in zip(a["proj"], b["proj"]):
        m = max(abs(x), abs(y))
        if m > 0.0:
            worst = max(worst, abs(x - y) / m)
    return worst


KL_REFUSAL = (
    "REFUSED: --kl. Top-k logits cannot support an unbiased KL, and this tool will "
    "not print a number wearing that name.\n"
    "\n"
    "Renormalising both arms over the union gives KL(p_U||q_U), the KL of DIFFERENT "
    "measures on a different support:\n"
    "\n"
    "    KL(p_U||q_U) = (1/c_p) * sum_{i in U} p_i log(p_i/q_i) + log(c_q/c_p)\n"
    "\n"
    "(1) Its value depends on k and on which ids both cuts kept, so the gate's own "
    "VT_Q4EXP_LOGITS budget moves the number a gate would compare.\n"
    "(2) The omitted tail terms p_i log(p_i/q_i) are NEGATIVE where q_i > p_i, so the "
    "partial sum bounds the true KL on neither side.\n"
    "(3) At k=1 with a shared argmax, U is one id, p_U = q_U = delta and the value is "
    "EXACTLY 0 whatever the tails do -- a green light wired to nothing.\n"
    "\n"
    "A full-row logsumexp per arm does not fix it either: that supplies only the "
    "normalisers c_p, c_q, while (2) needs both arms' individual tail probabilities, "
    "and log(p_i/q_i) is not linear in anything a projection can sum.\n"
    "\n"
    "What an exact KL needs is the FULL row for both arms (n floats, ~600 KB per line "
    "at n=151936) so both softmaxes are computable: a change to the tap, with its own "
    "spec, red-first test and review. It is OWED, not done here. Until then use "
    "top_dmax (a difference of VALUES on shared ids), top1_flip, max_rank_disp and "
    "rel_proj (sign-SENSITIVE over the whole buffer)."
)


def fixed_width(x, fmt="%.3e"):
    return fmt % x


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("base")
    ap.add_argument("other")
    ap.add_argument("--label", default="BASE vs OTHER")
    ap.add_argument("--step", type=int, default=None, help="print only this step")
    ap.add_argument("--top", type=int, default=20, help="rows to print (0 = all)")
    ap.add_argument("--kl", action="store_true",
                    help="REFUSED, always: top-k cannot support an unbiased KL. "
                         "Asking explains why and exits 2.")
    args = ap.parse_args(argv)

    if args.kl:
        print(KL_REFUSAL, file=sys.stderr)
        return 2

    try:
        A, order = parse(args.base)
        B, _ = parse(args.other)
    except ParseError as exc:
        print("REFUSED: %s" % exc, file=sys.stderr)
        return 2

    print("=== %s ===" % args.label)
    print("steps: base=%d other=%d" % (len(A), len(B)))

    common = [s for s in order if s in B]
    if not common:
        print("REFUSED: the two captures share no step (base %s, other %s). Nothing was "
              "compared; that is not agreement."
              % (sorted(A), sorted(B)), file=sys.stderr)
        return 2
    missing = [s for s in order if s not in B]
    for s in common:
        why = refusable_pairing(A[s], B[s])
        if why:
            print("REFUSED at step=%d: %s" % (s, why), file=sys.stderr)
            return 2

    print()
    print("NO KL IS PRINTED, AND `--kl` IS A REFUSAL. Top-k renormalised over the "
          "union is KL between")
    print("different measures on a different support: it moves with the tap's own k, "
          "its omitted tail")
    print("bounds the true KL on neither side, and at k=1 with a shared argmax it is "
          "EXACTLY 0.")
    print("Differences of VALUES over shared ids (top_dmax) and the sign-sensitive "
          "rel_proj are what")
    print("this capture can support. Run `--kl` for the full argument.")
    print()

    if args.step is not None:
        common = [s for s in common if s == args.step]
        if not common:
            print("REFUSED: step=%d is not in both captures." % args.step, file=sys.stderr)
            return 2

    # The column names are the names the docstring and docs/USAGE.md use, spelled
    # in full: `flip`/`rank` were abbreviations of `top1_flip`/`max_rank_disp`, and a
    # reader matching the table against the text had to guess the mapping.
    print("%-6s %-5s %-7s %-8s %-8s %-9s %-6s %-6s %9s %9s %-13s %8s %8s %10s %10s"
          % ("step", "rows", "n", "top1_a", "top1_b", "top1_flip", "shared", "jacc",
             "top_dmax", "top_dmed", "max_rank_disp", "dropped", "appeared",
             "rel_sumabs", "rel_proj"))
    flips = 0
    dz = 0
    rpz = 0
    worst = (0.0, None)
    worst_rp = (0.0, None)
    for s in (common if args.top == 0 else common[:args.top]):
        a, b = A[s], B[s]
        u = union_stats(a, b)
        # The WHOLE-BUFFER pair the docstring, the trailer and KL_REFUSAL all tell a
        # reader to use. Both functions existed and neither had a caller, so the tool
        # told a reader to trust a column it never printed.
        rs = rel_sumabs(a["sumabs"], b["sumabs"])
        rp = rel_proj(a, b)
        flip = a["top1"] != b["top1"]
        flips += int(flip)
        if u["dmax"] > 0.0:
            dz += 1
        if rp > 0.0:
            rpz += 1
        if u["dmax"] > worst[0]:
            worst = (u["dmax"], s)
        if rp > worst_rp[0]:
            worst_rp = (rp, s)
        print("%-6d %-5d %-7d %-8d %-8d %-9s %-6d %-6.2f %9.3e %9.3e %-13d %8d %8d "
              "%10.3e %10.3e"
              % (s, a["rows"], a["n"], a["top1"], b["top1"], "YES" if flip else "-",
                 len(u["shared"]), u["jaccard"], u["dmax"], u["dmed"],
                 u["max_rank_disp"], len(u["only_a"]), len(u["only_b"]), rs, rp))
        if flip:
            print("       top1 flip: base id %d (%.9g) vs other id %d (%.9g); the arm "
                  "that lost it holds %.9g where the other holds %.9g"
                  % (a["top1"], dict(a["pairs"])[a["top1"]],
                     b["top1"], dict(b["pairs"])[b["top1"]],
                     dict(a["pairs"]).get(b["top1"], float("nan")),
                     dict(b["pairs"]).get(a["top1"], float("nan"))))
        for tag, only, other in (("dropped", u["only_a"], b), ("appeared", u["only_b"], a)):
            if not only:
                continue
            kth = other["pairs"][-1][1]
            for i, v in only[:3]:
                print("       %-8s id %d = %.9g; the other arm's k-th value is %.9g "
                      "(delta %.3e) -> a cut-boundary move, not a value disagreement"
                      % (tag, i, v, kth, v - kth))

    print()
    print("--- %s SUMMARY ---" % args.label)
    print("STEPS COMPARED                : %d  (base has %d, other %d; %d unnamed in other)"
          % (len(common), len(A), len(B), len(missing)))
    print("STEPS WITH A TOP-1 FLIP       : %d" % flips)
    print("STEPS WITH top_dmax != 0      : %d   <- a DIFFERENCE of values; these PROVE the arms differ"
          % dz)
    if worst[1] is not None:
        print("LARGEST top_dmax              : %.6e at step=%d" % worst)
    print("STEPS WITH rel_proj != 0      : %d   <- sign-SENSITIVE over the whole buffer; these PROVE a difference"
          % rpz)
    if worst_rp[1] is not None:
        print("LARGEST rel_proj              : %.6e at step=%d" % worst_rp)
    print("KL                            : NOT PRINTED. Top-k cannot support an unbiased KL;")
    print("                                `--kl` explains why and exits 2 (see docstring).")
    print()
    print("rel_sumabs is a DIFFERENCE OF NORMS: zero means EQUAL L1 NORM, not equal values,")
    print("and at a fixed true divergence two readings differ by a median 2.1x (24x at p95)")
    print("over the 400 seeds of MetricSpread in tests/scripts/test_q4exp_layerfp_diff.py.")
    print("top_dmax is exact but samples at most k of n ids: non-zero PROVES a difference,")
    print("zero proves nothing. rel_proj is sign-SENSITIVE over the whole buffer.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
