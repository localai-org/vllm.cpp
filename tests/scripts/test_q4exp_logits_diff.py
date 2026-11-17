#!/usr/bin/env python3
"""The logits differ keys on IDS, refuses incomparable arms, and prints no KL.

`scripts/q4logit-diff.py` compares two `VT_Q4EXP_LOGITS` captures. Its payload is
a RANKED top-k, which gives it two ways to be wrong that the layer differ does
not have, and this file is the control for both.

`PositionVersusUnion` is the RED, and it is the contrast the tool exists for: on
two captures whose `pairs=` differ by ONE token crossing the cut, a positional
comparison reports a difference at every rank below the move, while the union
comparison reports `shared` (the tokens both arms kept) with `top_dmax == 0`. On
two captures that differ only in how the sort settled a TIE, the positional
comparison reports differences and the union one reports none. Both are built
from the instrument's exact `printf`, so the red is about the format, not about a
convenient stub.

`Refusals` pins the comparability contract: two arms that disagree on `n`, `top`
or `rows` are not a comparison and the tool exits 2 rather than reporting one.
`Truncation` pins the per-line counted property -- `top=` says how many pairs must
follow, `top1=` must be the first id, pairs must descend, ids must be unique, one
line per `step=` -- because a capture cut mid-line compares SHORT, and a short
comparison reads exactly like agreement.

`KLIsRefused` settles the question the row was asked to settle. At k=1 with a
shared argmax, the union-renormalised KL is EXACTLY 0.0 no matter what the two
tails do; this file computes that quantity in-test and shows the arms still differ
by `top_dmax != 0`, then asserts the tool prints no KL and that `--kl` exits 2
with the argument. `Publisher` asserts the settlement is in the RUNTIME output of
every run, not only in the docstring, so a reader of a capture meets it too.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import pathlib
import subprocess
import sys
import tempfile
import unittest

ROOT = pathlib.Path(__file__).resolve().parents[2]
TOOL = ROOT / "scripts" / "q4logit-diff.py"
sys.path.insert(0, str(ROOT / "scripts"))

_spec = importlib.util.spec_from_file_location("q4logit_diff", TOOL)
diff = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(diff)


def logit_line(step, pairs, n=151936, top=None, rows=1, top1=None, sumabs=1.0,
               proj=None, vmax=None, vmin=None):
    """The instrument's EXACT format, src/vllm/model_executor/models/qwen4_exp_forward.cpp."""
    if proj is None:
        proj = (0.0,) * 8
    if top is None:
        top = len(pairs)
    if top1 is None:
        top1 = pairs[0][0]
    if vmax is None:
        vmax = pairs[0][1]
    if vmin is None:
        vmin = pairs[-1][1]
    body = ",".join("%d:%.9g" % (i, v) for i, v in pairs)
    return ("q4logit step=%d rows=%d n=%d top=%d top1=%d vmax=%.9g vmin=%.9g "
            "sumabs=%.9g proj=%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g,%.9g pairs=%s\n"
            % (step, rows, n, top, top1, vmax, vmin, sumabs,
               proj[0], proj[1], proj[2], proj[3], proj[4], proj[5], proj[6],
               proj[7], body))


def capture(path, lines):
    pathlib.Path(path).write_text("".join(lines), encoding="utf-8")
    return str(path)


def run(*args):
    p = subprocess.run([sys.executable, str(TOOL)] + [str(a) for a in args],
                       capture_output=True, text=True)
    return p.returncode, p.stdout, p.stderr


def positional_diffs(a, b):
    """What a rank-by-rank differ reports. The thing this tool replaces."""
    n = 0
    for (ia, va), (ib, vb) in zip(a, b):
        if ia != ib or va != vb:
            n += 1
    return n + abs(len(a) - len(b))


def union_renormalized_kl(a, b):
    """KL(p_U||q_U), the quantity the tool REFUSES to call a KL.

    Values are treated as logits: p_i = exp(v_i) / sum_U exp(v). This is what the
    tempting renormalisation computes, and the point of `KLIsRefused` is that it
    depends only on the ids both cuts kept.
    """
    va, vb = dict(a), dict(b)
    ids = sorted(set(va) | set(vb))
    za = sum(math.exp(va[i]) for i in ids if i in va)
    zb = sum(math.exp(vb[i]) for i in ids if i in vb)
    total = 0.0
    for i in ids:
        if i not in va or i not in vb:
            continue  # renormalised over the union, so the id is simply absent
        p = math.exp(va[i]) / za
        q = math.exp(vb[i]) / zb
        total += p * math.log(p / q)
    return total


class PositionVersusUnion(unittest.TestCase):
    def test_a_token_crossing_the_cut_reds_every_rank_positionally(self):
        a = [(5, 9.0), (7, 8.0), (9, 7.0)]
        b = [(5, 9.0), (9, 7.0), (11, 6.0)]
        self.assertEqual(positional_diffs(a, b), 2,
                         "the positional differ must see a difference at each rank the "
                         "move displaced, which is the defect this tool removes")
        d = tmpdir_capture("cut")
        rc, out, err = run(d["a"], d["b"])
        self.assertEqual(rc, 0, err)
        self.assertIn("top_dmax", out)
        self.assertIn("0.000e+00", out, "shared ids 5 and 9 hold identical values; the "
                                        "union differ must report no value difference")
        u = diff.union_stats(diff.parse(d["a"])[0][0], diff.parse(d["b"])[0][0])
        self.assertEqual([i for i, _ in u["only_a"]], [7])
        self.assertEqual([i for i, _ in u["only_b"]], [11])
        self.assertEqual(u["shared"], [5, 9])
        self.assertAlmostEqual(u["jaccard"], 0.5)

    def test_a_reordered_TIE_is_not_a_difference(self):
        a = [(5, 9.0), (7, 8.0), (9, 8.0)]
        b = [(5, 9.0), (9, 8.0), (7, 8.0)]
        self.assertGreater(positional_diffs(a, b), 0)
        u = diff.union_stats(diff.parse(_write("tie", logit_line(0, a)))[0][0],
                             diff.parse(_write("tie2", logit_line(0, b)))[0][0])
        self.assertEqual(u["dmax"], 0.0)
        self.assertEqual(u["shared"], [5, 7, 9])
        self.assertEqual(u["only_a"], [])
        self.assertEqual(u["only_b"], [])


class Refusals(unittest.TestCase):
    def _refused(self, line_a, line_b, needle):
        rc, out, err = run(_write("ra", line_a), _write("rb", line_b))
        self.assertEqual(rc, 2, out)
        self.assertIn("REFUSED", err)
        self.assertIn(needle, err)

    def test_vocabulary_must_match(self):
        self._refused(logit_line(0, [(1, 9.0)], n=151936),
                      logit_line(0, [(1, 9.0)], n=152064), "n differs")

    def test_the_cut_must_match(self):
        self._refused(logit_line(0, [(1, 9.0), (2, 8.0)]),
                      logit_line(0, [(1, 9.0), (2, 8.0), (3, 7.0)]), "top differs")

    def test_rows_must_match_because_proj_covers_the_whole_buffer(self):
        self._refused(logit_line(0, [(1, 9.0)], rows=1),
                      logit_line(0, [(1, 9.0)], rows=4), "rows differs")

    def test_no_shared_step_is_a_refusal_not_an_empty_pass(self):
        rc, out, err = run(_write("sa", logit_line(0, [(1, 9.0)])),
                           _write("sb", logit_line(7, [(1, 9.0)])))
        self.assertEqual(rc, 2)
        self.assertIn("share no step", err)
        self.assertIn("not agreement", err)


class Truncation(unittest.TestCase):
    def _refused(self, body, needle):
        rc, out, err = run(_write("ta", body), _write("tb", logit_line(0, [(1, 9.0)])))
        self.assertEqual(rc, 2, out)
        self.assertIn(needle, err)

    def test_fewer_pairs_than_declared_top(self):
        self._refused(logit_line(0, [(1, 9.0)], top=4), "TRUNCATED")

    def test_duplicate_id(self):
        self._refused("q4logit step=0 rows=1 n=8 top=2 top1=1 vmax=9 vmin=8 "
                      "sumabs=1 proj=0,0,0,0,0,0,0,0 pairs=1:9,1:8\n", "duplicate ids")

    def test_pairs_must_descend(self):
        self._refused("q4logit step=0 rows=1 n=8 top=2 top1=2 vmax=8 vmin=9 "
                      "sumabs=1 proj=0,0,0,0,0,0,0,0 pairs=2:8,1:9\n", "descending")

    def test_top1_must_be_the_first_pair(self):
        self._refused(logit_line(0, [(1, 9.0), (2, 8.0)], top1=2), "top1=2")

    def test_proj_must_have_eight_values(self):
        self._refused(
            "q4logit step=0 rows=1 n=8 top=1 top1=1 vmax=9 vmin=9 sumabs=1 "
            "proj=0,0,0,0,0,0,0 pairs=1:9\n", "proj has 7")

    def test_one_line_per_step(self):
        self._refused(logit_line(0, [(1, 9.0)]) + logit_line(0, [(1, 9.0)]),
                      "appears twice")

    def test_empty_capture(self):
        rc, out, err = run(_write("ea", ""), _write("eb", logit_line(0, [(1, 9.0)])))
        self.assertEqual(rc, 2)
        self.assertIn("ZERO q4logit rows", err)


class KLIsRefused(unittest.TestCase):
    def test_at_k1_a_shared_argmax_makes_the_renormalised_KL_exactly_zero(self):
        # Two arms that disagree on the top value and on the whole-buffer
        # projections, but keep the same single id: this is the case that makes the
        # renormalised quantity degenerate rather than merely biased.
        a = [(42, 9.0)]
        b = [(42, 8.0)]
        self.assertEqual(union_renormalized_kl(a, b), 0.0,
                         "KL over the union of two single-id cuts is 0 by construction: "
                         "this is why a top-k 'KL' cannot gate anything")
        u = diff.union_stats(diff.parse(_write("ka", logit_line(0, a, proj=(1.0,) + (0.0,) * 7)))[0][0],
                             diff.parse(_write("kb", logit_line(0, b, proj=(2.0,) + (0.0,) * 7)))[0][0])
        self.assertAlmostEqual(u["dmax"], 1.0)
        self.assertGreater(diff.rel_proj(diff.parse(_write("kc", logit_line(0, a, proj=(1.0,) + (0.0,) * 7)))[0][0],
                                        diff.parse(_write("kd", logit_line(0, b, proj=(2.0,) + (0.0,) * 7)))[0][0]),
                           0.0, "the arms DO differ; only the KL-shaped statistic reads 0")

    def test_the_tool_prints_no_kl_number(self):
        rc, out, err = run(_write("n1", logit_line(0, [(1, 9.0)])),
                           _write("n2", logit_line(0, [(1, 9.0)])))
        self.assertEqual(rc, 0, err)
        for line in out.splitlines():
            self.assertFalse(line.strip().startswith("KL ") and "NOT" not in line,
                             "no KL value may be printed: %r" % line)
        self.assertIn("KL                            : NOT PRINTED", out)

    def test_dash_dash_kl_is_a_refusal_with_the_argument(self):
        rc, out, err = run(_write("k1", logit_line(0, [(1, 9.0)])),
                           _write("k2", logit_line(0, [(1, 9.0)])), "--kl")
        self.assertEqual(rc, 2)
        self.assertIn("cannot support an unbiased KL", err)
        self.assertIn("EXACTLY 0", err)
        self.assertIn("FULL row", err)


class Publisher(unittest.TestCase):
    def test_the_settlement_is_in_the_runtime_output_of_a_normal_run(self):
        rc, out, err = run(_write("p1", logit_line(0, [(1, 9.0), (2, 8.0)])),
                           _write("p2", logit_line(0, [(1, 9.0), (2, 8.0)])))
        self.assertEqual(rc, 0, err)
        self.assertIn("NO KL IS PRINTED", out)
        self.assertIn("EXACTLY 0", out)
        self.assertIn("different support", out)

    def test_the_docstring_and_the_behaviour_agree(self):
        doc = diff.__doc__
        self.assertIn("CANNOT SUPPORT AN UNBIASED KL", doc)
        self.assertIn("DEGENERATE", doc)
        self.assertIn("not linear in anything a projection can sum", doc)


_TMP = tempfile.mkdtemp(prefix="q4logit-diff-")
_N = itertools.count()


def _write(name, body):
    """Write one capture and return its path. Never reuses a name."""
    p = pathlib.Path(_TMP) / ("%s_%d.log" % (name, next(_N)))
    p.write_text(body, encoding="utf-8")
    return str(p)


def tmpdir_capture(name):
    a = _write(name + "_a", logit_line(0, [(5, 9.0), (7, 8.0), (9, 7.0)]))
    b = _write(name + "_b", logit_line(0, [(5, 9.0), (9, 7.0), (11, 6.0)]))
    return {"a": a, "b": b}


if __name__ == "__main__":
    unittest.main()
