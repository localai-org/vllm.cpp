#!/usr/bin/env python3
"""`scripts/agent-preflight.sh` never reports green over a block that did not run.

Row `GATE-PREFLIGHT-SKIP-REPORT`, issue #998, spec
`.agents/specs/gate-preflight-skip-report.md`.

The defect this suite pins: the trailer block was guarded by
`git merge-base --is-ancestor origin/main HEAD`, and when that guard was false
the block emitted NOTHING and the run still printed `All gates green.` and
exited 0. Two gates left the report without a word. It fired three times in one
session, twice because `origin/main` is a remote-tracking ref that every
worktree of one checkout shares, so another worktree's fetch moved it BETWEEN
the top of the run and the guard. On the third the `ok` count fell from 76 to 74
and the banner did not change.

That last number is the whole problem in one line. Nothing in the output
distinguished a run that checked the trailers from a run that did not, so the
only reader who could catch it was one who read the script.

## Why this suite executes the script instead of grepping it

Every assertion here could be spelled as a text match against the script, and
every one of them would then be satisfied by a REWRITE while missing an
OVERRIDE on a later line. `tests/scripts/test_agent_onboard.py` already learned
that lesson about `REQUIRE_ROLE` and switched to executing the script. The same
reasoning applies harder here, because what is under test is control flow across
four `git` calls, not the value of a variable.

## Why it can execute the script without recursing

`scripts/agent-preflight.sh` runs this very suite from its `SUITES` array, so a
nested full run would recurse without bound. `--role-only` exists for that
problem and is too narrow here, because the blocks under test are the last two
in the file.

So the run is pointed at a SCRATCH repository instead: the script is copied into
a temporary git repo and a stub `python3` that exits 0 is put on `PATH`. Every
`run` then reports `ok` without executing a checker, nothing re-enters this
suite, and the git-shaped control flow under test executes for real against refs
the test owns. `ROOT` in the script is derived from `BASH_SOURCE`, so the copy
makes the scratch repo the script's own tree.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PREFLIGHT = ROOT / "scripts/agent-preflight.sh"
READY = ROOT / "scripts/agent-ready.py"
CI = ROOT / ".github/workflows/ci.yml"
SUITE_NAME = Path(__file__).stem

# Resolved BEFORE any test prepends its scratch `bin` to PATH, so the shim in
# `NON_NUMERIC_GIT` can forward to the real program rather than to itself.
GIT = shutil.which("git")

ANSI = re.compile(r"\x1b\[[0-9;]*m")

# Exits 0 for every checker and every suite. The script's `run()` reports `ok`.
#
# It also RECORDS its argv, one invocation per line, when the test asks for it.
# Without that, this suite pins the ancestry decision against the pinned SHA and
# says nothing about the SHA the five range checkers are actually handed: putting
# `--base origin/main` back on all three range gates, and `--range
# origin/main..HEAD` back on both trailer gates, parses and leaves every other
# case here green. The script's own comment above `BASE_REF` states the guarantee
# ("Every range block below compares against this SHA and never against the
# ref"), and nothing detected its loss. `scripts/check-test-registration.py`
# traces preflight the same way for the same reason.
INERT_PYTHON3 = """#!/bin/sh
if [ -n "${VLLM_TEST_ARGV_LOG:-}" ]; then
  printf '%s\\n' "$*" >> "$VLLM_TEST_ARGV_LOG"
fi
exit 0
"""

# The gates that take the pinned base as an argument, by the script name each one
# is invoked with. Every one of them must be handed `BASE_SHA`.
#
# `check-issue-index-append-only.py` was a fifth until #2290 deleted it. It
# guarded a tracked append-only file against a wrong union merge, and the index
# is derived now -- nothing appends to a generated file, so there is no ordering
# to violate. The entry is removed with the gate, not to quiet a red.
BASE_ARGUMENT_GATES = (
    "scripts/check-now-current.py",
    "scripts/check-commit-trailers.py",
    "scripts/check-commit-style.py",
)

# Advances `refs/remotes/origin/main` on its FIRST call and then goes inert.
# This is what another worktree of the same checkout does when it fetches while
# a preflight is running, and it is occurrences 2 and 3 of #998 reproduced
# deterministically. `cd "$ROOT"` at the top of the script means the stub's cwd
# is the scratch repo, so the ref it moves is the one the script reads.
MOVING_PYTHON3 = """#!/bin/sh
if [ ! -e "$VLLM_TEST_MOVE_MARKER" ]; then
  : > "$VLLM_TEST_MOVE_MARKER"
  git update-ref refs/remotes/origin/main "$VLLM_TEST_MOVE_TO"
fi
exit 0
"""

# Answers `rev-list` with a line that is not a number and exits 0, and forwards
# every other subcommand to the real program. No git known to this suite behaves
# this way, and that is the point: the script must hold an arm for a count it
# cannot read, rather than inferring from exit 0 that stdout is a decimal
# integer. That inference is what produced the defect this class pins.
NON_NUMERIC_GIT = """#!/bin/sh
if [ "$1" = "rev-list" ]; then
  echo "not-a-commit-count"
  exit 0
fi
exec {git} "$@"
"""

# A path that does not exist, named as an alternate object database. Git prints
# `error: unable to normalize alternate object path: ...` to stderr on nearly
# every object-reading command, writes its ordinary answer to stdout, and exits
# 0. It is the cheapest reproduction of "stderr without a failure" and it needs
# no shim at all.
MISSING_ALTERNATE = "/nonexistent/vllm-cpp-preflight-alternate/objects"


class Report:
    """One preflight run, parsed into the three states it can report."""

    def __init__(self, returncode: int, text: str) -> None:
        self.returncode = returncode
        self.text = ANSI.sub("", text)
        lines = self.text.splitlines()
        self.ok = [line[7:] for line in lines if line.startswith("  ok   ")]
        self.fail = [line[7:] for line in lines if line.startswith("  FAIL ")]
        self.skip = [line[7:] for line in lines if line.startswith("  SKIP ")]
        self.headings = [line for line in lines if re.fullmatch(r"[A-Z][^\n]*:", line)]

    @property
    def green(self) -> bool:
        return "All gates green." in self.text

    def __str__(self) -> str:  # shown verbatim on any failure below
        return self.text


class PreflightHarness(unittest.TestCase):
    """A scratch repo with a known ref topology and an inert `python3`."""

    def setUp(self) -> None:
        self.tmp = Path(tempfile.mkdtemp(prefix="preflight-skip-"))
        self.addCleanup(shutil.rmtree, self.tmp, True)
        (self.tmp / "scripts").mkdir()
        (self.tmp / "bin").mkdir()
        shutil.copy2(PREFLIGHT, self.tmp / "scripts" / PREFLIGHT.name)
        self.script = self.tmp / "scripts" / PREFLIGHT.name
        self.argv_log = self.tmp / "argv.log"
        self.set_python3(INERT_PYTHON3)

        self.git("init", "--quiet", ".")
        self.git("config", "user.email", "preflight@test.invalid")
        self.git("config", "user.name", "preflight test")
        self.git("config", "commit.gpgsign", "false")
        (self.tmp / "a").write_text("a\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "base")
        self.base = self.rev("HEAD")

        # A divergent commit that HEAD does NOT contain. This is what
        # `origin/main` looks like from a branch that is behind it.
        self.git("checkout", "--quiet", "-b", "divergent")
        (self.tmp / "b").write_text("b\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "divergent")
        self.divergent = self.rev("HEAD")

        self.git("checkout", "--quiet", "-B", "work", self.base)
        (self.tmp / "c").write_text("c\n", encoding="utf-8")
        self.git("add", "-A")
        self.git("commit", "--quiet", "-m", "work")
        self.head = self.rev("HEAD")

        self.set_origin_main(self.base)

    # -- scratch repo helpers ------------------------------------------------

    def git(self, *args: str) -> str:
        result = subprocess.run(
            ["git", *args], cwd=self.tmp, capture_output=True, text=True, check=True
        )
        return result.stdout.strip()

    def rev(self, revision: str) -> str:
        return self.git("rev-parse", revision)

    def set_origin_main(self, oid: str) -> None:
        self.git("update-ref", "refs/remotes/origin/main", oid)

    def unset_origin_main(self) -> None:
        self.git("update-ref", "-d", "refs/remotes/origin/main")

    def set_python3(self, body: str) -> None:
        stub = self.tmp / "bin" / "python3"
        stub.write_text(body, encoding="utf-8")
        stub.chmod(0o755)

    def environment(self, **env: str) -> dict[str, str]:
        environment = dict(os.environ)
        environment["PATH"] = f"{self.tmp / 'bin'}{os.pathsep}{environment['PATH']}"
        # Truncated per run, so a log always describes exactly one invocation of
        # the script even in the cases that run it twice.
        self.argv_log.write_text("", encoding="utf-8")
        environment["VLLM_TEST_ARGV_LOG"] = str(self.argv_log)
        environment.update(env)
        return environment

    def preflight(self, *args: str, **env: str) -> Report:
        result = subprocess.run(
            ["bash", str(self.script), "--quiet", *args],
            cwd=self.tmp,
            capture_output=True,
            text=True,
            check=False,
            env=self.environment(**env),
        )
        return Report(result.returncode, result.stdout + result.stderr)

    def base_arguments(self, gate: str) -> list[str]:
        """Every recorded `python3` invocation of `gate` that carried a base."""

        recorded = self.argv_log.read_text(encoding="utf-8").splitlines()
        return [
            line
            for line in recorded
            if line.startswith(gate) and ("--base " in line or "--range " in line)
        ]

    def skip_reason(self, report: Report, label: str) -> str:
        """The indented reason block printed under one SKIP label.

        `report.skip` carries the labels only, so an assertion made against
        `report.text` cannot tell a reason printed under the gate it explains
        from the same words printed anywhere else in the run. This reads the
        block `skip()` indents by nine columns, which is the reason and nothing
        else.
        """

        lines = report.text.splitlines()
        for index, line in enumerate(lines):
            if not (line.startswith("  SKIP ") and line[7:].strip() == label):
                continue
            reason = []
            for following in lines[index + 1 :]:
                if not following.startswith("         "):
                    break
                reason.append(following[9:])
            return "\n".join(reason)
        return ""

    # -- preconditions -------------------------------------------------------

    def assert_ran_something(self, report: Report) -> None:
        """A run that reported nothing at all asserts nothing at all.

        The stub `python3` makes every gate cheap, and a broken harness makes
        every gate ABSENT. Both produce a short transcript with no `FAIL`, and
        only this check tells them apart. `assertEqual([], report.fail)` on its
        own is satisfied by a run that never started.
        """

        self.assertGreater(
            len(report.ok),
            40,
            f"harness precondition failed: only {len(report.ok)} gates reported "
            f"`ok`, so this run asserts nothing.\n{report}",
        )
        self.assertEqual([], report.fail, f"scratch run had failures:\n{report}")


class SkipIsReportedTests(PreflightHarness):
    def test_a_branch_behind_the_base_runs_both_trailer_gates(self) -> None:
        """RED before: both trailer gates reported SKIP and the ok count fell.

        The first occurrence of #998 was a branch behind `main` whose trailer
        block said nothing. The SKIP arm that replaced the silent `&&` is gone
        now (#2366): a base that is not an ancestor of HEAD no longer deletes
        the two gates from the report. `rev-list BASE..HEAD` selects the
        branch's own commits under any ancestry, and check-commit-style.py
        walks from the merge base, so a branch behind the base is gated, not
        skipped.
        """

        self.set_origin_main(self.divergent)
        report = self.preflight()
        self.assert_ran_something(report)

        for label in ("commit-trailers", "commit-style"):
            with self.subTest(gate=label):
                self.assertNotIn(
                    f"SKIP {label}",
                    report.text,
                    f"{label} skipped on a branch behind the base:\n{report}",
                )
                self.assertIn(
                    label,
                    report.ok,
                    f"{label} did not report ok:\n{report}",
                )
        self.assertEqual(
            [],
            report.skip,
            f"a behind-base branch reported a skip:\n{report}",
        )
        self.assertTrue(
            report.green,
            f"both trailer gates ran and passed, but no green banner:\n{report}",
        )

    def test_the_ok_count_falls_only_with_a_reported_skip(self) -> None:
        """A gate that stops running has to say so; a behind base stops none.

        This was occurrence 3 of #998, back when a non-ancestor base deleted
        the two trailer gates and the ok count fell without a word. Since
        #2366 the behind run executes the same gates as the control, so the
        count cannot fall and no skip is owed for the ancestry alone.
        """

        self.set_origin_main(self.base)
        ancestor = self.preflight()
        self.assert_ran_something(ancestor)
        self.assertEqual([], ancestor.skip, f"the control run skipped:\n{ancestor}")

        self.set_origin_main(self.divergent)
        behind = self.preflight()
        self.assert_ran_something(behind)

        self.assertEqual(
            len(ancestor.ok),
            len(behind.ok),
            f"{len(ancestor.ok) - len(behind.ok)} gate(s) disappeared from the "
            "report and no SKIP accounts for them. Every gate that stops "
            f"running has to say so.\n{behind}",
        )
        self.assertEqual(
            [],
            behind.skip,
            f"a behind-base branch reported a skip:\n{behind}",
        )

    def test_a_base_that_moves_mid_run_does_not_change_the_verdict(self) -> None:
        """RED before: the trailer block vanishes and the run still says green.

        Occurrences 2 and 3. `origin/main` is shared by every worktree of one
        checkout, so the guard was true when the run started and false when it
        was evaluated. Pinning the base at the start is what fixes it, and the
        run has to name the SHA it pinned.
        """

        marker = self.tmp / "moved"
        self.set_python3(MOVING_PYTHON3)
        self.set_origin_main(self.base)
        report = self.preflight(
            VLLM_TEST_MOVE_MARKER=str(marker),
            VLLM_TEST_MOVE_TO=self.divergent,
        )
        self.assert_ran_something(report)

        # PRECONDITION, asserted and never skipped: if the stub never moved the
        # ref then this test reproduces nothing and must go red.
        self.assertTrue(
            marker.exists(),
            "precondition failed: the stub python3 never ran, so origin/main "
            f"never moved and this test asserts nothing.\n{report}",
        )
        self.assertEqual(
            self.divergent,
            self.rev("refs/remotes/origin/main"),
            f"precondition failed: origin/main did not move.\n{report}",
        )

        self.assertIn(
            self.base,
            report.text,
            "the run does not name the base SHA it gated against, so a reader "
            f"cannot tell which revision it compared to.\n{report}",
        )
        self.assertEqual(
            [],
            report.skip,
            "the base was pinned to an ancestor at the start of the run, so no "
            f"block should have been skipped when the ref moved:\n{report}",
        )
        self.assertTrue(
            any("commit-trailers" in line for line in report.ok),
            f"the trailer gate did not run against the pinned base:\n{report}",
        )
        self.assertTrue(report.green, f"the run earned the banner and withheld it:\n{report}")

    def test_an_unresolvable_base_reports_skip_for_both_range_blocks(self) -> None:
        """RED before: five gates vanish and the run says green.

        The same shape in the `Committed range` block. A checkout whose remote
        is not named `origin`, or one that has never fetched, resolves no
        `origin/main`. `AGENTS.md`: "Unknown is not absence or success."
        """

        self.unset_origin_main()
        report = self.preflight()
        self.assert_ran_something(report)

        for label in (
            "now-current range",
            "commit-trailers",
            "commit-style",
        ):
            with self.subTest(gate=label):
                self.assertTrue(
                    any(label in line for line in report.skip),
                    f"{label} vanished without reporting a SKIP:\n{report}",
                )
        self.assertFalse(report.green, f"five skipped gates still printed green:\n{report}")


class TheBannerStaysReachableTests(PreflightHarness):
    def test_an_ancestor_base_still_earns_the_banner(self) -> None:
        """The control. A fix that suppresses the banner always cannot pass."""

        self.set_origin_main(self.base)
        report = self.preflight()
        self.assert_ran_something(report)
        self.assertEqual([], report.skip, f"an ordinary run skipped a gate:\n{report}")
        self.assertTrue(report.green, f"an ordinary run lost the banner:\n{report}")
        self.assertEqual(0, report.returncode, f"an ordinary run exited non-zero:\n{report}")
        for label in ("commit-trailers", "commit-style"):
            with self.subTest(gate=label):
                self.assertTrue(
                    any(label in line for line in report.ok),
                    f"{label} did not run on an ancestor base:\n{report}",
                )

    def test_an_empty_range_is_not_a_skip(self) -> None:
        """An empty input set is not an unexamined one.

        `HEAD` level with the base means the range gates have zero commits to
        read, and a verdict over zero commits is not withheld information. If
        this reported `SKIP`, the ordinary session-start run of a freshly cut
        branch would print a false alarm and lose the banner, and
        `AGENTS.md` is explicit that a gate firing on ordinary work is the
        defect. Without this case the obvious over-correction passes every
        other test in this file.
        """

        self.set_origin_main(self.head)
        report = self.preflight()
        self.assert_ran_something(report)
        self.assertEqual([], report.skip, f"an empty range reported a SKIP:\n{report}")
        self.assertTrue(report.green, f"an empty range lost the banner:\n{report}")
        self.assertEqual(0, report.returncode, f"an empty range exited non-zero:\n{report}")

    def test_a_skipped_run_exits_zero_and_a_failing_run_does_not(self) -> None:
        """SKIP and FAIL are different facts and keep different exit codes.

        A base that does not resolve is ordinary work, so a skip does not fail
        the run. Widening exit 1 to mean "a gate did not run" would merge the
        two facts into one signal, which is the conflation this row removes.
        """

        self.unset_origin_main()
        skipped = self.preflight()
        self.assert_ran_something(skipped)
        self.assertNotEqual([], skipped.skip, f"nothing was skipped:\n{skipped}")
        self.assertEqual(
            0,
            skipped.returncode,
            f"a skipped block failed the run:\n{skipped}",
        )

        # And a real failure still exits 1, so exit 0 above is a decision rather
        # than a script that cannot fail.
        self.set_python3("#!/bin/sh\nexit 1\n")
        failing = self.preflight()
        self.assertNotEqual(
            0,
            failing.returncode,
            f"every gate failed and the run exited 0:\n{failing}",
        )
        self.assertFalse(failing.green, f"a failing run printed the banner:\n{failing}")


class ThePinnedBaseReachesTheCheckersTests(PreflightHarness):
    def test_every_range_gate_is_handed_the_pinned_sha_and_never_the_ref(self) -> None:
        """The pin is worth nothing if the checkers are still passed the ref.

        Every other case in this file reads the script's REPORT, so all of them
        stay green when `--base "$BASE_SHA"` reverts to `--base origin/main` on
        the three range gates or `--range "${BASE_SHA}..HEAD"` reverts to
        `--range origin/main..HEAD` on the two trailer gates. The ancestry
        decision would still be made against the pinned SHA, the heading would
        still name it, and the five checkers would judge whatever the ref points
        at when each one starts. With `origin/main` moving mid-run, that is a
        report whose heading and whose verdict are about different revisions.
        """

        self.set_origin_main(self.base)
        report = self.preflight()
        self.assert_ran_something(report)
        self.assertEqual([], report.skip, f"an ordinary run skipped a gate:\n{report}")

        recorded = {gate: self.base_arguments(gate) for gate in BASE_ARGUMENT_GATES}

        # PRECONDITION: a log that recorded nothing satisfies every `assertNotIn`
        # below. Count first, and count ALL of them, so a gate that stops being
        # invoked at all cannot pass as a gate invoked correctly.
        # DERIVED from the tuple, not written as a literal. It was `4` and went
        # stale the moment a gate was deleted (#2290), which turned a
        # precondition into the only red in the suite. One invocation per gate is
        # the property; the number is an implementation detail of the list.
        self.assertEqual(
            len(BASE_ARGUMENT_GATES),
            sum(len(lines) for lines in recorded.values()),
            f"precondition failed: the stub recorded {recorded}, which is not "
            f"one base-carrying invocation per gate.\n{report}",
        )

        for gate, lines in recorded.items():
            with self.subTest(gate=gate):
                self.assertEqual(
                    1,
                    len(lines),
                    f"{gate} was invoked with a base {len(lines)} time(s): {lines}",
                )
                self.assertIn(
                    self.base,
                    lines[0],
                    f"{gate} was not handed the pinned SHA {self.base}: {lines[0]}",
                )
                self.assertNotIn(
                    "origin/main",
                    lines[0],
                    f"{gate} was handed the moving ref instead of the pinned "
                    f"SHA: {lines[0]}",
                )


class AnUnknownIsNotAnEmptyRangeTests(PreflightHarness):
    def test_an_unborn_head_reports_skip_rather_than_an_empty_range(self) -> None:
        """RED before: three gates take the empty-range exemption and say nothing.

        `git rev-list --count` was spelled `|| echo 0`, which maps a FAILED
        count onto the deliberate exemption for a range with no commits in it.
        The two are opposite facts: an empty range withholds nothing, and a
        count that could not be taken withholds everything. `git checkout
        --orphan` reaches it with `origin/main` perfectly resolvable.

        The same run also pins the ancestry reason. `--is-ancestor` answers 1
        for "no" and 128 for "that question cannot be asked here", and both used
        to take the arm that says the branch is behind `origin/main`. That names
        a cause which is not the cause, and it sends the reader to `git merge`
        for a tree that has no commit to merge into.
        """

        self.set_origin_main(self.base)
        self.git("checkout", "--orphan", "unborn")

        # PRECONDITION: this reproduces nothing unless HEAD really is unborn
        # while the base still resolves.
        self.assertNotEqual(
            0,
            subprocess.run(
                ["git", "rev-parse", "--verify", "-q", "HEAD"],
                cwd=self.tmp, capture_output=True, text=True, check=False,
            ).returncode,
            "precondition failed: HEAD still resolves, so the git queries under "
            "test do not fail and this test asserts nothing.",
        )

        report = self.preflight()
        self.assert_ran_something(report)

        for label in (
            "now-current range",
            "commit-trailers",
            "commit-style",
        ):
            with self.subTest(gate=label):
                self.assertTrue(
                    any(label in line for line in report.skip),
                    f"{label} did not report a SKIP:\n{report}",
                )
        self.assertFalse(report.green, f"five unknown gates printed green:\n{report}")
        self.assertNotIn(
            "empty, HEAD adds no commits",
            report.text,
            f"a failed count was reported as an empty range:\n{report}",
        )
        self.assertNotIn(
            "is not an ancestor of HEAD",
            report.text,
            "a failed ancestry query was reported as a branch behind the base, "
            f"which names the wrong cause:\n{report}",
        )
        self.assertIn(
            "Unknown is not an empty range.",
            report.text,
            f"the range skip does not say what is unknown:\n{report}",
        )
        self.assertIn(
            "Unknown is not a verdict on ancestry.",
            report.text,
            f"the ancestry skip does not say what is unknown:\n{report}",
        )

    def test_a_range_skip_carries_the_message_git_printed(self) -> None:
        """RED before: the reason named a status and showed an empty value.

        Discarding stderr is what keeps it out of the VALUE, and `2>/dev/null`
        also threw the message away. The reader was told that the count
        `exited 128 and printed [] on stdout` and never told why, although git
        had already said why in one line. A skip that names no cause is honest
        and not actionable, and this row owes both halves.

        The fix is the discipline the same block already uses two lines above
        for `ANCESTRY_ERROR`: the message goes into its OWN variable, never into
        the value that an arm is selected from. `2>&1 >/dev/null` is the
        ordering that yields the message alone, because stderr is duplicated
        onto the capture before stdout leaves for `/dev/null`.
        """

        self.set_origin_main(self.base)
        self.git("checkout", "--orphan", "unborn")

        # PRECONDITION: git has to actually write a diagnostic to stderr for the
        # exact command the script runs. If it prints nothing there is no
        # message to carry and this case asserts nothing.
        probe = subprocess.run(
            ["git", "rev-list", "--count", f"{self.base}..HEAD"],
            cwd=self.tmp, capture_output=True, text=True, check=False,
        )
        self.assertNotEqual(
            0,
            probe.returncode,
            "precondition failed: the count succeeded, so no skip is reported "
            "and this case asserts nothing.",
        )
        printed = probe.stderr.strip().splitlines()
        self.assertNotEqual(
            [],
            printed,
            "precondition failed: git wrote nothing to stderr, so there is no "
            "message for the reason to carry.",
        )
        message = printed[0]
        self.assertIn(
            "fatal:",
            message,
            f"precondition failed: git's first stderr line is not a diagnostic: {message}",
        )

        report = self.preflight()
        self.assert_ran_something(report)

        # The trailer gates take the ANCESTRY arm here (`--is-ancestor` exits
        # 128 on an unborn HEAD), so `RANGE_UNKNOWN` reaches exactly these
        # three. Asserting it on the other two would pass on the wrong message.
        for label in (
            "now-current range",
        ):
            with self.subTest(gate=label):
                reason = self.skip_reason(report, label)
                self.assertNotEqual(
                    "",
                    reason,
                    f"{label} reported no SKIP with a reason:\n{report}",
                )
                self.assertIn(
                    message,
                    reason,
                    f"{label} skipped without the message git printed, so the "
                    f"reader is told a status and not a cause:\n{report}",
                )


class StderrIsNotTheValueTests(PreflightHarness):
    """A git that writes to stderr and exits 0 must not delete a range block.

    The `|| echo 0` repair above was first written as
    `RANGE_COUNT="$(git rev-list --count ... 2>&1)"`, which reintroduced the
    silence this row exists to remove, through a narrower door and in the
    DISHONEST direction. Folding stderr into the value leaves `RANGE_STATUS`
    catching only the case where git FAILS. When git exits 0 and also writes to
    stderr, the status is 0 and the value is the error text with the number
    after it, so `[ "$RANGE_COUNT" -gt 0 ]` does not evaluate false, it ERRORS
    with status 2. A `[` that errors reads as false, both range blocks take
    their empty-range arm, and the run prints `All gates green.` over five gates
    that never executed.

    Neither case here is exotic. `test_an_unborn_head_...` above covers a git
    that fails. Nothing covered a git that succeeds noisily, and the pre-repair
    script handled it correctly because it discarded stderr.
    """

    def break_alternates(self) -> None:
        alternates = self.tmp / ".git" / "objects" / "info" / "alternates"
        alternates.parent.mkdir(parents=True, exist_ok=True)
        alternates.write_text(f"{MISSING_ALTERNATE}\n", encoding="utf-8")

    def raw_git(self, *args: str) -> subprocess.CompletedProcess[str]:
        """A git run the way the script runs it, through the scratch PATH."""

        return subprocess.run(
            ["git", *args],
            cwd=self.tmp,
            capture_output=True,
            text=True,
            check=False,
            env=self.environment(),
        )

    def test_a_git_that_warns_and_exits_zero_still_runs_every_range_gate(self) -> None:
        """RED before: five gates vanish, `empty` is printed, the banner prints.

        The count is perfectly readable on stdout here, so the correct report is
        not a SKIP. It is the ordinary run, with all five gates executed. A
        repair that answered this case with a SKIP would cost five gates to a
        stderr line that changed no answer.
        """

        self.set_origin_main(self.base)
        control = self.preflight()
        self.assert_ran_something(control)
        self.assertEqual([], control.skip, f"the control run skipped:\n{control}")

        self.break_alternates()

        # PRECONDITION, asserted and never assumed: this reproduces nothing
        # unless git really does exit 0, write to stderr, and print the count.
        probe = self.raw_git("rev-list", "--count", f"{self.base}..HEAD")
        self.assertEqual(
            0,
            probe.returncode,
            "precondition failed: git did not exit 0, so this case is the "
            f"already-covered failing-git case instead: {probe.stderr}",
        )
        self.assertNotEqual(
            "",
            probe.stderr.strip(),
            "precondition failed: git wrote nothing to stderr, so there is no "
            "stderr for the value to absorb and this case asserts nothing.",
        )
        self.assertEqual(
            "1",
            probe.stdout.strip(),
            "precondition failed: stdout is not the count, so the repair under "
            "test is not the one being exercised.",
        )

        report = self.preflight()
        self.assert_ran_something(report)

        for label in (
            "now-current range",
            "commit-trailers",
            "commit-style",
        ):
            with self.subTest(gate=label):
                self.assertTrue(
                    any(label in line for line in report.ok),
                    f"{label} did not run although the count was readable on "
                    f"stdout:\n{report}",
                )
        self.assertEqual([], report.skip, f"a readable count reported a SKIP:\n{report}")
        self.assertNotIn(
            "empty, HEAD adds no commits",
            report.text,
            f"HEAD adds a commit and the run reported an empty range:\n{report}",
        )
        # The same invariant as the ok-count case, against the same control: a
        # gate may leave the report only by saying that it did.
        self.assertEqual(
            len(control.ok),
            len(report.ok),
            f"{len(control.ok) - len(report.ok)} gate(s) disappeared and "
            f"{len(report.skip)} were reported as skipped:\n{report}",
        )
        self.assertTrue(report.green, f"the run earned the banner and withheld it:\n{report}")

    def test_a_count_that_is_not_a_number_reports_skip_rather_than_empty(self) -> None:
        """RED before, and red against a repair that only discards stderr.

        Keeping stderr out of the value fixes the case above and leaves this arm
        missing: any value that is not a decimal integer still reaches `-gt`,
        where the status-2 error is indistinguishable from "zero commits". The
        exit status is not evidence about stdout, and reading it as evidence is
        the assumption that produced this defect twice.
        """

        self.assertIsNotNone(GIT, "precondition failed: no git on PATH to forward to")
        shim = self.tmp / "bin" / "git"
        shim.write_text(NON_NUMERIC_GIT.format(git=GIT), encoding="utf-8")
        shim.chmod(0o755)
        self.set_origin_main(self.base)

        # PRECONDITION: the shim has to be the git the script finds, it has to
        # exit 0, and it has to answer with something that is not a count.
        probe = self.raw_git("rev-list", "--count", f"{self.base}..HEAD")
        self.assertEqual(0, probe.returncode, f"precondition failed: {probe.stderr}")
        self.assertEqual("not-a-commit-count", probe.stdout.strip())
        # And it must still forward everything else, or the run under test fails
        # for a reason that has nothing to do with the count.
        self.assertEqual(self.base, self.raw_git("rev-parse", "refs/remotes/origin/main").stdout.strip())

        report = self.preflight()
        self.assert_ran_something(report)

        for label in (
            "now-current range",
            "commit-trailers",
            "commit-style",
        ):
            with self.subTest(gate=label):
                self.assertTrue(
                    any(label in line for line in report.skip),
                    f"{label} did not report a SKIP for a count that is not a "
                    f"count:\n{report}",
                )
        self.assertFalse(report.green, f"five unknown gates printed green:\n{report}")
        self.assertNotIn(
            "empty, HEAD adds no commits",
            report.text,
            f"an unreadable count was reported as an empty range:\n{report}",
        )
        self.assertIn(
            "Unknown is not an empty range.",
            report.text,
            f"the range skip does not say what is unknown:\n{report}",
        )


class FailOnSkipTests(PreflightHarness):
    """The third state has to reach a caller that can only read the exit status."""

    def test_the_flag_makes_a_skip_exit_1_and_the_default_still_exits_0(self) -> None:
        """Both facts in one case, because each is the other's justification.

        Exit 0 by default is the row's own §3.4 decision: a checkout whose
        base does not resolve is ordinary work. Exit 1 under the flag is what a program
        needs, because the exit status carries two of the three states and a
        program cannot read the report that carries the third.
        """

        self.unset_origin_main()

        plain = self.preflight()
        self.assert_ran_something(plain)
        self.assertNotEqual([], plain.skip, f"nothing was skipped:\n{plain}")
        self.assertEqual(0, plain.returncode, f"the default failed a skip:\n{plain}")

        strict = self.preflight("--fail-on-skip")
        self.assert_ran_something(strict)
        self.assertEqual(
            plain.skip,
            strict.skip,
            "the flag changed WHAT was reported, and it must change only the "
            f"exit status:\n{strict}",
        )
        self.assertEqual(
            1,
            strict.returncode,
            f"--fail-on-skip exited 0 over {len(strict.skip)} skipped "
            f"gate(s):\n{strict}",
        )
        self.assertFalse(strict.green, f"a skipped run printed the banner:\n{strict}")
        self.assertIn(
            "--fail-on-skip",
            strict.text,
            f"the refusal does not name the flag that caused it:\n{strict}",
        )

    def test_the_flag_does_not_fire_on_a_run_that_skipped_nothing(self) -> None:
        """A flag that reds an ordinary run is the defect, not the discipline."""

        self.set_origin_main(self.base)
        report = self.preflight("--fail-on-skip")
        self.assert_ran_something(report)
        self.assertEqual([], report.skip, f"an ordinary run skipped:\n{report}")
        self.assertTrue(report.green, f"an ordinary run lost the banner:\n{report}")
        self.assertEqual(0, report.returncode, f"an ordinary run exited 1:\n{report}")


class AgentReadyRefusesASkipTests(PreflightHarness):
    """`scripts/agent-ready.py` is the one consumer that reads the exit status.

    It is `AGENTS.md`'s documented gate before a remote handoff, and it read
    preflight by return code alone. So a branch behind `origin/main`, or any
    checkout where `origin/main` does not resolve, reached
    `READY: local and live PR/CI evidence are green` with two or five gates
    never run. The word "green" was in the output over a trailer check that had
    not executed, which is this row's own thesis failing one layer up.

    These cases run the real `agent-ready.py` against the scratch repo, so what
    is proved is that the refusal REACHES it, not that a flag exists.
    """

    def setUp(self) -> None:
        super().setUp()
        shutil.copy2(READY, self.tmp / "scripts" / READY.name)
        self.ready_script = self.tmp / "scripts" / READY.name

    def ready(self) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, str(self.ready_script)],
            cwd=self.tmp,
            capture_output=True,
            text=True,
            check=False,
            env=self.environment(),
        )

    def test_a_skipped_preflight_stops_the_handoff_gate(self) -> None:
        """RED before: preflight exits 0 over skipped gates and this passes."""

        self.unset_origin_main()

        # PRECONDITION: the same tree must actually make preflight skip, or the
        # case below proves nothing about a skip.
        report = self.preflight()
        self.assert_ran_something(report)
        self.assertNotEqual([], report.skip, f"nothing was skipped:\n{report}")

        result = self.ready()
        output = result.stdout + result.stderr
        self.assertNotEqual(0, result.returncode, f"agent-ready passed a skip:\n{output}")
        self.assertIn(
            "READY FAILED: local preflight",
            result.stderr,
            f"agent-ready did not stop at the local preflight:\n{output}",
        )
        self.assertIn(
            "SKIPPED",
            result.stdout,
            f"agent-ready did not relay the skip report to its caller:\n{output}",
        )
        # It stopped at the preflight and never reached the remote question. If
        # it had, the refusal below would be the one that appeared instead, and
        # a skip would be indistinguishable from having no remote.
        self.assertNotIn("REMOTE_UNVERIFIED", output)
        self.assertNotIn("READY:", output)

    def test_an_unskipped_preflight_lets_the_handoff_gate_continue(self) -> None:
        """The control. A flag that refuses every run gates nothing.

        The scratch repo has no `origin` remote, so a run that gets past the
        local preflight fails at the remote question instead. That refusal is
        the proof that the local one did not fire.
        """

        self.set_origin_main(self.base)
        report = self.preflight()
        self.assert_ran_something(report)
        self.assertEqual([], report.skip, f"the control run skipped:\n{report}")

        result = self.ready()
        output = result.stdout + result.stderr
        self.assertIn(
            "REMOTE_UNVERIFIED",
            result.stderr,
            f"agent-ready did not get past the local preflight:\n{output}",
        )
        self.assertNotIn("READY FAILED: local preflight", output)


class RegistrationTests(unittest.TestCase):
    def test_the_suite_runs_in_preflight_and_in_ci(self) -> None:
        """A suite wired into neither surface runs on no machine.

        `test_main_baseline.py` shipped in neither and its 24 tests ran nowhere,
        so both surfaces are asserted here and removing either is red.
        """

        preflight = PREFLIGHT.read_text(encoding="utf-8")
        self.assertIn(SUITE_NAME, preflight, "not in the preflight SUITES array")
        self.assertIn(
            f"python3 tests/scripts/{SUITE_NAME}.py",
            CI.read_text(encoding="utf-8"),
            "CI does not run this suite",
        )


class BenchmarkIndexRegistrationTests(PreflightHarness):
    def test_preflight_executes_the_benchmark_index_checker(self) -> None:
        """Deleting the production call must leave a red reachability test."""

        report = self.preflight()
        self.assert_ran_something(report)
        invocations = self.argv_log.read_text(encoding="utf-8").splitlines()
        self.assertIn(
            "scripts/check-benchmark-index.py",
            invocations,
            "preflight did not execute the mechanical benchmark-index checker",
        )


class StagedInvocationTests(PreflightHarness):
    """`--staged` runs the bash script, and the run reports a parse-clean pass.

    The kolibri1-GGUF implementer report (2026-10-30) read
    "SyntaxError around scripts/agent-preflight.sh:86" as a defect in inline
    python inside the --staged branch. The diagnosis the tree gives is
    different: the script carries NO inline python, and the crash is Python
    parsing the BASH file after a `python3 scripts/agent-preflight.sh --staged`
    invocation -- the parser stops at the first bash-only token, the
    `--staged)` case label on line 86 ("SyntaxError: unmatched ')'"), which is
    why the report names that line. The non-staged run those instructions
    compared against was the documented `bash` invocation, hence "clean".

    A bash file cannot defend itself against a python3 parse, so the repairable
    surface is the documented invocation itself: this case executes it the way
    AGENTS.md spells it and pins that the --staged branch runs a real report
    with no interpreter traceback in either stream.
    """

    def test_staged_run_in_a_scratch_repo_reports_no_syntax_error(self) -> None:
        report = self.preflight("--staged")

        self.assertNotIn("SyntaxError", report.text)
        self.assertNotIn("Traceback", report.text)
        # The run reached the staged branch and answered it, rather than dying
        # on the way there: the heading, the gate result line under it, and an
        # exit status from the documented set (0 green, 1 failing gates).
        self.assertIn("Staged change:", report.text)
        self.assertIn("now-current --staged", report.text)
        self.assertIn(report.returncode, (0, 1), str(report))


if __name__ == "__main__":
    unittest.main()
