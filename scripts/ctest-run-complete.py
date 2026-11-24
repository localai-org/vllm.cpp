#!/usr/bin/env python3
"""Exit 0 when a ctest log has a terminal run summary, 1 when it does not.

A ctest process that actually finished prints either a pass/fail percentage or
the total test time, even when tests failed. A runner death before that point
leaves neither, so an incomplete log is the discriminator CI uses to retry a
killed run without ever retrying a real finding into green.

Usage: ctest-run-complete.py <log-file>
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

SUMMARY = re.compile(r"(?m)^\s*(?:\d+% tests passed|Total Test time\s*=)")


def complete(text: str) -> bool:
    """True when ctest reached a terminal summary in ``text``."""
    return SUMMARY.search(text) is not None


def main() -> int:
    if len(sys.argv) != 2:
        print("usage: ctest-run-complete.py <log-file>", file=sys.stderr)
        return 2
    try:
        text = Path(sys.argv[1]).read_text(encoding="utf-8", errors="replace")
    except OSError as error:
        print(f"cannot read ctest log: {error}", file=sys.stderr)
        return 2
    return 0 if complete(text) else 1


if __name__ == "__main__":
    sys.exit(main())
