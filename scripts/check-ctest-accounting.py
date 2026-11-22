#!/usr/bin/env python3
"""Account for every CTest entry a run reported.

The arm64-full lane used to parse the summary's ``out of N`` as its executed
count. That number is the enumeration total, so the guard could never see a
short run, and a skipped entry counted as a pass. This tool reads the per-test
terminal status lines instead, reports each kind, and exits non-zero when an
enumerated entry produced no terminal status at all.

A skip is not a pass: it is reported separately and never added to the count
of tests the lane executed. ctest's own exit status still decides failures.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass

# One terminal line per executed test. ctest marks Failed/Skipped/Not Run/Timeout
# with "***" and leaves Passed unmarked; both shapes carry a seconds field.
TERMINAL = re.compile(
    r"(?m)^\s*(?:\d+/\d+\s+)?Test\s+#\d+:\s+.*?"
    r"(\*\*\*)?(Passed|Failed|Skipped|Not Run|Timeout)\s+\d+\.\d+\s+sec"
)


@dataclass(frozen=True)
class Accounting:
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    not_run: int = 0
    timeout: int = 0

    @property
    def accounted(self) -> int:
        return self.passed + self.failed + self.skipped + self.not_run + self.timeout

    @property
    def executed(self) -> int:
        return self.passed + self.failed + self.timeout


def parse(text: str) -> Accounting:
    """Count the terminal status lines in one captured ctest run."""
    counts = {"Passed": 0, "Failed": 0, "Skipped": 0, "Not Run": 0, "Timeout": 0}
    for match in TERMINAL.finditer(text):
        counts[match.group(2)] += 1
    return Accounting(**{k.lower().replace(" ", "_"): v for k, v in counts.items()})


def evaluate(accounting: Accounting, enumerated: int) -> list[str]:
    """Return the refusal reasons; an empty list means the run is accounted.

    The assertion is about coverage, not about pass/fail: every enumerated
    entry must have produced one terminal status. A skipped entry is accounted
    for but is never added to the executed count, so a skip is not a pass.
    ctest's own exit status still decides whether the failures are acceptable.
    """
    reasons: list[str] = []
    if enumerated <= 0:
        reasons.append(
            f"REFUSED: ctest enumerated {enumerated} entries; a lane that "
            "selects nothing certifies nothing"
        )
    if accounting.accounted != enumerated:
        reasons.append(
            f"REFUSED: ctest reported {accounting.accounted} terminal statuses "
            f"for {enumerated} enumerated entries; the run is not the suite"
        )
    return reasons


def report(accounting: Accounting, enumerated: int, reasons: list[str]) -> None:
    print(
        "ctest ACCOUNTED '{a.accounted}' entries; ctest -N enumerated "
        "'{n}': passed={a.passed} failed={a.failed} skipped={a.skipped} "
        "not-run={a.not_run} timeout={a.timeout}".format(
            a=accounting, n=enumerated
        )
    )
    print(
        f"ctest EXECUTED '{accounting.executed}' entries; "
        f"{accounting.skipped} skipped entries are not passes"
    )
    for reason in reasons:
        print(reason, file=sys.stderr)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, help="captured ctest output file")
    parser.add_argument("--count", required=True, type=int, help="ctest -N total")
    args = parser.parse_args(argv)
    try:
        text = open(args.run, encoding="utf-8", errors="replace").read()
    except OSError as exc:
        print(f"REFUSED: cannot read ctest run output: {exc}", file=sys.stderr)
        return 1
    accounting = parse(text)
    reasons = evaluate(accounting, args.count)
    report(accounting, args.count, reasons)
    return 1 if reasons else 0


if __name__ == "__main__":
    raise SystemExit(main())
