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

`WholeBufferColumns` pins the two whole-buffer comparisons the table owed and did
not print: `rel_sumabs()` and `rel_proj()` had no caller while the docstring, the
run preamble, the trailer and `--kl` all told a reader to read `rel_proj`
(ISSUE-LOCAL-01M7P9S7RJC7XCHHSYD9BSJ2S6). It asserts the printed value equals the
function applied to the parsed fields, and that a pair with the same `sumabs`
reads 0 there while `rel_proj` sees it -- the claim the documents make.

`PublishedUsageSection` pins `docs/USAGE.md`, which nothing read. The layer
comparator has a publisher for exactly this reason (see
`test_q4exp_layerfp_diff.py::MetricSpread`), and a section deleted by a rebase or
edited to promise output the tool does not produce reds nowhere. It matches each
claim against the thing it describes -- `union_stats` for the union rule, the tap
source for `std::partial_sort` and the two knobs, real runs for the exit code, the
refusals and the printed columns, and `KL_REFUSAL` for the full-row cost -- so the
section cannot drift from the tool, and cannot be deleted.
"""

from __future__ import annotations

import importlib.util
import itertools
import math
import pathlib
import re
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


def _rows(out):
    """The per-step table's data rows, found by its header rather than by position."""
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "top_dmax" in line and "step" in line:
            rows = []
            for nxt in lines[i + 1:]:
                if not nxt.strip():
                    break
                rows.append(nxt.split())
            return rows
    raise AssertionError("the output carries no per-step table:\n%s" % out)


def _table(out):
    """The header row and the first data row under it."""
    lines = out.splitlines()
    for i, line in enumerate(lines):
        if "top_dmax" in line and "step" in line:
            return line.split(), lines[i + 1].split()
    raise AssertionError("the output carries no per-step table:\n%s" % out)


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


USAGE = ROOT / "docs" / "USAGE.md"
TAP = ROOT / "src" / "vllm" / "model_executor" / "models" / "qwen4_exp_forward.cpp"


class WholeBufferColumns(unittest.TestCase):
    """The tool promised `rel_proj` four times and printed it zero times.

    `rel_sumabs()` and `rel_proj()` sat in `scripts/q4logit-diff.py` with no
    caller: the per-step table ended at `appeared`, while the docstring's "WHAT
    THIS TOOL PRINTS INSTEAD" list, the preamble of every run, the trailer and
    `KL_REFUSAL` all told a reader to read `rel_proj`, and
    `.agents/specs/quant-mach2-additive.md` specifies the `top_dmax`/`rel_proj`
    columns. So this was implementation lag against a committed spec, not a
    design change (ISSUE-LOCAL-01M7P9S7RJC7XCHHSYD9BSJ2S6).

    PRINTING TWO NUMBERS IS NOT THE PROPERTY. They have to MEAN what the
    documents say, and the second case is the one that says so: `rel_sumabs` is a
    difference of NORMS, so two arms with the same `sumabs` read exactly 0 there
    however far apart their buffers are, and `rel_proj` must not. That is the
    whole reason the section tells a reader to trust one and not the other.
    """

    P = (1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)

    def test_the_step_row_carries_the_two_whole_buffer_comparisons(self):
        a = logit_line(0, [(5, 9.0), (7, 8.0), (9, 7.0)], sumabs=100.0, proj=self.P)
        b = logit_line(0, [(5, 9.0), (9, 7.0), (11, 6.0)], sumabs=101.0,
                       proj=(1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
        rc, out, err = run(_write("wb", a), _write("wb", b))
        self.assertEqual(rc, 0, err)
        header, row = _table(out)
        self.assertEqual(header[-2:], ["rel_sumabs", "rel_proj"],
                         "the row must END with the two whole-buffer comparisons the "
                         "docstring, the trailer and --kl all name; header was %s" % header)
        want_sumabs = abs(100.0 - 101.0) / 101.0
        want_proj = abs(1.0 - 1.5) / 1.5
        got_sumabs, got_proj = float(row[-2]), float(row[-1])
        self.assertLessEqual(abs(got_sumabs - want_sumabs), want_sumabs * 1e-3,
                             "rel_sumabs is not |sumabs_a - sumabs_b| / max: %r vs %r"
                             % (got_sumabs, want_sumabs))
        self.assertLessEqual(abs(got_proj - want_proj), want_proj * 1e-3,
                             "rel_proj is not the max projection difference: %r vs %r"
                             % (got_proj, want_proj))

    def test_rel_sumabs_cancels_where_rel_proj_cannot(self):
        """The documents' central claim, made executable.

        Identical `sumabs`, different buffer: the difference of norms reads
        EXACTLY 0 and the projection comparison reads 2/3. A later edit that made
        `rel_proj` insensitive in the same way reds here, which is the only thing
        that keeps the section's advice from becoming false.
        """
        a = logit_line(0, [(5, 9.0)], sumabs=100.0, proj=self.P)
        b = logit_line(0, [(5, 9.0)], sumabs=100.0,
                       proj=(3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0))
        rc, out, err = run(_write("wb0", a), _write("wb0", b))
        self.assertEqual(rc, 0, err)
        header, row = _table(out)
        self.assertEqual(float(row[header.index("rel_sumabs")]), 0.0,
                         "rel_sumabs must CANCEL here: both arms have the same L1 norm, "
                         "which is exactly why the section says not to read it")
        got = float(row[header.index("rel_proj")])
        self.assertLessEqual(abs(got - 2.0 / 3.0), (2.0 / 3.0) * 1e-3,
                             "rel_proj must see a whole-buffer difference that rel_sumabs "
                             "cancels; got %r" % got)

    def test_the_summary_counts_the_steps_where_rel_proj_is_nonzero(self):
        same = logit_line(1, [(5, 9.0)], sumabs=100.0, proj=self.P)
        a = logit_line(0, [(5, 9.0)], sumabs=100.0, proj=self.P) + same
        b = logit_line(0, [(5, 9.0)], sumabs=100.0,
                       proj=(3.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0)) + same
        rc, out, err = run(_write("wbs", a), _write("wbs", b))
        self.assertEqual(rc, 0, err)
        flat = re.sub(r"\s+", " ", out)
        self.assertIn("STEPS WITH rel_proj != 0 : 1", flat,
                      "the summary must count the steps rel_proj proves differ, since "
                      "top_dmax != 0 is the other count and they are not the same set")
        self.assertIn("LARGEST rel_proj", flat)


class PublishedUsageSection(unittest.TestCase):
    """`docs/USAGE.md` is public, and nothing read the section this tool owes.

    The layer comparator's suite publishes its figures out of the same file
    (`test_q4exp_layerfp_diff.py::MetricSpread`, #2879): a figure copied into a
    second file and never re-read drifts, and a section deleted by a rebase reds
    nowhere. `Publisher` above keeps the KL settlement in the RUNTIME output;
    this class keeps the USAGE section itself.

    IT PINS CLAIMS AND READS THEM BACK OFF THE IMPLEMENTATION. Each anchor is a
    claim, and every case that can check its claim does: the three-set union rule
    against `union_stats`, `std::partial_sort` and the two knobs against the tap
    source, `exit 2` and the refusal list against runs that are actually
    refused, the columns against the header a run prints, and `n` plus the KB
    cost against `KL_REFUSAL`. A section edited to promise something the tool
    does not do cannot pass, and `_section` fails by NAME when the heading is
    gone, so deleting the surface is a failure and not a silent skip.

    WRAPPING IS NOT A CLAIM. Anchors are matched on whitespace-flattened text, so
    rewrapping the paragraph stays green and editing what it says does not.
    """

    HEADING = "## Comparing two logits captures (Qwen4-EXP Q4Logit)"
    PREDECESSOR = "## Diagnosing a CPU/CUDA divergence (Qwen4-EXP LayerFp)"

    ANCHORS = (
        ("the command and its path", "python3 scripts/q4logit-diff.py"),
        ("the tap it reads", "`VT_Q4EXP_LOGITS=<k>`"),
        ("the tap's budget", "`VT_Q4EXP_LOGITS_STEPS`"),
        ("the id union", "keys on the id"),
        ("the union's definition", "the union is the ids in either arm's top-k"),
        ("shared, defined", "`shared` the ids in both"),
        ("the two exclusive sets", "`only_a`/`only_b` the rest"),
        ("where the value statistics come from",
         "computes every value difference over `shared`"),
        ("why a positional comparison is wrong", "`std::partial_sort`"),
        ("the one-token consequence", "a single token moving across the cut"),
        ("what a positional differ cannot express", "cannot express `shared` at all"),
        ("top_dmax's definition",
         "`top_dmax` is the largest value difference over shared ids"),
        ("top_dmax's floor", "zero proves nothing"),
        ("the sign-sensitive column", "`rel_proj` is the sign-sensitive"),
        ("the sign-insensitive one", "`rel_sumabs` is a difference of norms"),
        ("the refusal exit code", "exit 2, with the reason"),
        ("--kl is a refusal", "**`--kl` is a refusal, not an option.**"),
        ("--kl's exit code", "It exits 2 and prints the argument."),
        ("the degenerate k=1 case", "at k=1 with a shared argmax it is EXACTLY 0"),
        ("the owed tap change", "OWED, not done"),
        ("the suite that pins it", "`tests/scripts/test_q4exp_logits_diff.py`"),
    )

    REFUSAL_ANCHORS = (
        ("a vocabulary mismatch", "arms that disagree on `n`"),
        ("a cut mismatch", "on `top`"),
        ("a row-count mismatch", "or on `rows`"),
        ("a truncated capture", "prints fewer pairs"),
        ("no shared step", "share no step"),
    )

    @staticmethod
    def _flat(text):
        return re.sub(r"\s+", " ", text.replace("\r\n", "\n")).strip()

    def _text(self):
        return USAGE.read_text(encoding="utf-8")

    def _section(self, text=None):
        """The section alone, flattened. Fails BY NAME when it is not there."""
        flat = self._flat(self._text() if text is None else text)
        self.assertGreaterEqual(
            flat.find(self.HEADING), 0,
            "docs/USAGE.md no longer carries %r; the surface this tool owes was "
            "deleted or retitled, and nothing else in the tree would say so"
            % self.HEADING)
        rest = flat[flat.index(self.HEADING) + len(self.HEADING):]
        nxt = rest.find("## ")
        return rest if nxt < 0 else rest[:nxt]

    @staticmethod
    def _constant(src, function, pattern):
        """One integer the tap's own function reads. Fails loudly if the shape moved."""
        i = src.index("int64_t %s()" % function)
        body = src[i:src.index("\n}\n", i)]
        m = re.search(pattern, body)
        if m is None:
            raise AssertionError("the tap's %s no longer has the shape this pin reads "
                                 "(%r); read it and update the pin, or the section is "
                                 "now describing a constant nobody reads"
                                 % (function, pattern))
        return int(m.group(1))

    def test_the_section_is_present_and_beside_the_layer_comparator(self):
        flat = self._flat(self._text())
        self.assertEqual(flat.count(self.HEADING), 1,
                         "docs/USAGE.md must carry %r exactly once" % self.HEADING)
        self.assertIn(self.PREDECESSOR, flat)
        between = flat[flat.index(self.PREDECESSOR) + len(self.PREDECESSOR):
                        flat.index(self.HEADING)]
        self.assertNotIn("## ", between,
                         "another section was inserted between the layer comparator and "
                         "the logits one; the logits section belongs beside the differ it "
                         "is read after")

    def test_the_section_carries_every_anchor_it_owes(self):
        section = self._section()
        for what, anchor in self.ANCHORS + self.REFUSAL_ANCHORS:
            with self.subTest(what):
                self.assertIn(self._flat(anchor), section,
                              "docs/USAGE.md no longer states: %s" % what)

    def test_the_UNION_RULE_is_the_one_union_stats_implements(self):
        """The section's three sets, checked against the function -- not a second copy."""
        a = [(5, 9.0), (7, 8.0), (9, 7.0)]
        b = [(5, 9.0), (9, 7.0), (11, 6.0)]
        A, _ = diff.parse(_write("usage_u", logit_line(0, a)))
        B, _ = diff.parse(_write("usage_u", logit_line(0, b)))
        u = diff.union_stats(A[0], B[0])
        ids_a = {i for i, _ in a}
        ids_b = {i for i, _ in b}
        shared = set(u["shared"])
        only_a = {i for i, _ in u["only_a"]}
        only_b = {i for i, _ in u["only_b"]}
        self.assertEqual(shared | only_a | only_b, ids_a | ids_b,
                         "the union is not `the ids in either arm's top-k`")
        self.assertEqual(shared, ids_a & ids_b, "`shared` is not `the ids in both`")
        self.assertEqual(only_a, ids_a - ids_b, "`only_a` is not `the rest` of base's cut")
        self.assertEqual(only_b, ids_b - ids_a, "`only_b` is not `the rest` of other's cut")
        self.assertEqual(dict(u["only_a"]), {i: v for i, v in a if i in only_a},
                         "an only-a id does not carry BASE's value, so the printed "
                         "cut-boundary delta is measured against the wrong arm")
        self.assertEqual(u["dmax"], max(abs(dict(a)[i] - dict(b)[i]) for i in shared),
                         "top_dmax is not `the largest value difference over shared ids`")

    def test_the_SORT_it_blames_is_the_sort_the_tap_runs(self):
        src = TAP.read_text(encoding="utf-8")
        self.assertIn("std::partial_sort", src,
                      "the tap no longer sorts with std::partial_sort, so the section's "
                      "explanation of why a positional comparison measures the sort now "
                      "describes code that is gone")
        self.assertRegex(src, r"host\[static_cast<size_t>\(a\)\] > host\[static_cast<size_t>\(b\)\]",
                         "the cut is no longer the k LARGEST logits, so `pairs=` is not "
                         "descending by value and the section's premise is wrong")
        self.assertIn("pairs=", src, "the tap no longer prints `pairs=`, which both the "
                                      "section and the tool parse")

    def test_the_COLUMNS_it_names_are_columns_a_run_prints(self):
        section = self._section()
        rc, out, err = run(
            _write("usage_cols", logit_line(0, [(5, 9.0), (7, 8.0), (9, 7.0)])),
            _write("usage_cols", logit_line(0, [(5, 9.0), (9, 7.0), (11, 6.0)])))
        self.assertEqual(rc, 0, err)
        header, _ = _table(out)
        para = section[section.index("Columns:"):]
        para = para[:para.index(" It refuses rather than")]
        named = set(re.findall(r"`([a-z_0-9]+)`", para))
        self.assertTrue(named, "the Columns paragraph names no column at all")
        self.assertEqual(named - set(header), set(),
                         "the section names column(s) the tool never prints: %s"
                         % sorted(named - set(header)))
        self.assertEqual(
            {"rel_sumabs", "rel_proj", "top_dmax", "top_dmed", "top1_flip",
             "max_rank_disp"} - named, set(),
            "the section stopped naming a column a reader needs: %s"
            % sorted({"rel_sumabs", "rel_proj", "top_dmax", "top_dmed", "top1_flip",
                      "max_rank_disp"} - named))

    def test_the_EXIT_CODE_it_states_is_the_code_the_tool_returns(self):
        section = self._section()
        stated = {int(c) for c in re.findall(r"exits? (\d+)", section)}
        self.assertTrue(stated, "the section states no exit code, so a reader cannot tell "
                                "a refusal from a crash")
        a = _write("usage_rc", logit_line(0, [(5, 9.0), (7, 8.0)]))
        b = _write("usage_rc", logit_line(0, [(5, 9.0), (7, 8.0)]))
        rc, out, err = run(a, b, "--kl")
        self.assertEqual(rc, 2)
        self.assertIn("REFUSED: --kl", err,
                      "`--kl` must be REFUSED by the tool and not by argparse; argparse "
                      "also exits 2, so the code alone would pass for the wrong reason")
        observed = {
            "an incomparable vocabulary": run(
                _write("usage_rc", logit_line(0, [(5, 9.0)], n=151936)),
                _write("usage_rc", logit_line(0, [(5, 9.0)], n=1000)))[0],
            "a truncated capture": run(
                _write("usage_rc", logit_line(0, [(5, 9.0)], top=4)), b)[0],
            "no shared step": run(
                _write("usage_rc", logit_line(0, [(5, 9.0)])),
                _write("usage_rc", logit_line(1, [(5, 9.0)])))[0],
        }
        self.assertEqual(set(observed.values()), stated,
                         "the section states exit %s; the tool returned %s"
                         % (sorted(stated), observed))

    def test_the_REFUSALS_it_lists_are_refusals_the_tool_makes(self):
        a = logit_line(0, [(5, 9.0), (7, 8.0)])
        b = logit_line(0, [(5, 9.0), (7, 8.0)])
        cases = (
            ("a vocabulary mismatch", "n differs between the arms",
             logit_line(0, [(5, 9.0)], n=151936), logit_line(0, [(5, 9.0)], n=1000)),
            ("a cut mismatch", "top differs between the arms",
             a, logit_line(0, [(5, 9.0)], top=1)),
            ("a row-count mismatch", "rows differs between the arms",
             logit_line(0, [(5, 9.0)], rows=1), logit_line(0, [(5, 9.0)], rows=2)),
            ("a truncated capture", "TRUNCATED",
             logit_line(0, [(5, 9.0)], top=4), b),
            ("no shared step", "share no step",
             logit_line(0, [(5, 9.0)]), logit_line(1, [(5, 9.0)])),
        )
        for what, needle, la, lb in cases:
            with self.subTest(what):
                rc, out, err = run(_write("usage_ref", la), _write("usage_ref", lb))
                self.assertEqual(rc, 2, "%s: expected the refusal exit code, got %d (%s)"
                                 % (what, rc, err.strip()))
                self.assertIn("REFUSED", err)
                self.assertIn(needle, err, "%s: the refusal no longer names why" % what)

    def test_the_TAP_KNOBS_it_documents_are_the_ones_the_tap_reads(self):
        section = self._section()
        src = TAP.read_text(encoding="utf-8")
        cap = self._constant(src, "LogitsTapTopkFromEnv", r"\? (\d+) :")
        default = self._constant(src, "LogitsTapBudgetFromEnv", r"\) return (\d+);")
        disabled = self._constant(src, "LogitsTapBudgetFromEnv", r": (\d+);")
        self.assertEqual({int(c) for c in re.findall(r"capped at (\d+)", section)}, {cap},
                         "the section's cap is not the cap LogitsTapTopkFromEnv applies")
        self.assertEqual({int(c) for c in re.findall(r"default (\d+)", section)}, {default},
                         "the section's default budget is not the one "
                         "LogitsTapBudgetFromEnv applies")
        self.assertEqual(disabled, 0, "the section says `0` disables, but the tap's "
                                      "non-positive branch returns %d" % disabled)

    def test_the_FULL_ROW_COST_it_quotes_is_the_one_the_refusal_names(self):
        section = self._section()
        doc = re.search(r"about (\d+) KB per line at n=(\d+)", section)
        refusal = re.search(r"~(\d+) KB per line at n=(\d+)", diff.KL_REFUSAL)
        self.assertIsNotNone(doc, "the section no longer quotes the full-row cost")
        self.assertIsNotNone(refusal, "KL_REFUSAL no longer quotes the full-row cost")
        self.assertEqual(doc.groups(), refusal.groups(),
                         "docs/USAGE.md quotes %s KB at n=%s and the tool's own `--kl` text "
                         "quotes %s KB at n=%s; one of them moved"
                         % (doc.groups() + refusal.groups()))

    def test_the_TWO_FLAGS_it_documents_do_what_it_says(self):
        section = self._section()
        self.assertIn("`--step N` prints one step and `--top 0` prints all of them.",
                      section)
        body = (logit_line(0, [(5, 9.0), (7, 8.0)]) + logit_line(1, [(5, 9.0)])
                + logit_line(2, [(5, 9.0)]))
        a = _write("usage_flags", body)
        b = _write("usage_flags", body)
        rc, out, err = run(a, b, "--top", "0")
        self.assertEqual(rc, 0, err)
        self.assertEqual(len(_rows(out)), 3, "`--top 0` must print every shared step")
        rc, out, err = run(a, b, "--step", "1")
        self.assertEqual(rc, 0, err)
        rows = _rows(out)
        self.assertEqual(len(rows), 1, "`--step N` must print exactly that step")
        self.assertEqual(rows[0][0], "1")

    def test_the_PIN_is_NOT_VACUOUS_on_a_moved_code_or_a_lost_anchor(self):
        """The positive control for every case above.

        A matcher that silently matched nothing reads exactly like one that
        matched everything. Both failure modes are exercised on a COPY of the real
        section: move the documented exit code, delete an anchor, delete the
        heading.
        """
        section = self._section()
        self.assertTrue(section, "the section is empty")
        moved = section.replace("It exits 2 and prints the argument.",
                                "It exits 3 and prints the argument.")
        self.assertNotEqual(moved, section, "the exit-code anchor is no longer in the "
                                            "section, so this control is stale")
        self.assertNotEqual({int(c) for c in re.findall(r"exits? (\d+)", moved)},
                            {int(c) for c in re.findall(r"exits? (\d+)", section)},
                            "a moved exit code did not reach the extraction")
        lost = section.replace("`std::partial_sort`", "")
        self.assertNotEqual(lost, section)
        self.assertIn(
            ("why a positional comparison is wrong", "`std::partial_sort`"),
            [site for site in self.ANCHORS if self._flat(site[1]) not in lost],
            "a deleted anchor did not reach the anchor list")
        with self.assertRaises(AssertionError) as caught:
            self._section(self._text().replace(self.HEADING, "## Something else"))
        self.assertIn(self.HEADING, str(caught.exception),
                      "deleting the heading must fail by NAME")


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
