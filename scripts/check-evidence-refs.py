#!/usr/bin/env python3
"""Fail when an evidence report names a raw-log file that is not in the tree.

Motivated by FIX-EVIDENCE-LOG-GATE. `.gitignore` carried a global `*.log`
rule, so `git add` on an evidence directory silently skipped every raw log it
contained; the merged worktree removal then took the untracked files. Three
landed rows' evidence dirs (tt-kolibri1-readback-2026-10-28,
fix-paged-regression-20261028, fix-tt-numa-degrade-2026-10-29) committed only
their README: assertion bytes with no backing measurement records. The
whitelist `!docs/bench-evidence/**/*.log` stops the silent drop; this checker
makes the residue red.

Scope: `docs/bench-evidence/*/README.md` and `report.md`. The scan is a plain
filename-token scan over evidence suffixes (`.log`, `.log.txt`) — it does not
parse prose semantics. It is a floor, not a ceiling: it must catch the
observed failure (a report lists logs, the logs are absent), not intent. Every
referenced basename must exist on disk AND be tracked; the basename is
resolved anywhere in the tracked tree, because evidence dirs legitimately
reference shared helpers. A file that exists on disk but is not tracked fails
exactly as loudly as a missing one — that is the original incident.

Unreferenced extra files PASS. The checker polices the reference direction
only; it is not a completeness gate.

A report may record a loss instead of a reference: any line containing the
marker `EVIDENCE-LOST` exempts the filenames on that line. This is how the
2026-10-30 loss notes for the unrecoverable dirs stay honest — the loss is
recorded in the same file that used to promise the bytes, and the checker
reads it as a recorded loss rather than a live promise. An `EVIDENCE-LOST:`
line naming nothing is noise, not a gate failure.

Standard library only, no network.
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

EVIDENCE_GLOB_md = "*/README.md"
EVIDENCE_SUFFIXES = (".log", ".log.txt")

# A filename token: letters, digits, dot, underscore, dash, slash, ending in an
# evidence suffix. `*.log` (a glob mention) does not match because `*` is not
# in the class; an absolute `/tmp/x.log` does match and is skipped as
# path-scoped, not a tree reference.
FILENAME_RE = re.compile(r"[A-Za-z0-9._/-]+\.(?:log\.txt|log)\b")

# Lines carrying this marker record a loss instead of asserting a live
# reference; filenames on the marker line and on the rest of that note block
# (up to the next blank line) are exempt.
LOSS_MARKER = "EVIDENCE-LOST"


def evidence_reports(root: Path) -> list[Path]:
    base = root / "docs/bench-evidence"
    if not base.is_dir():
        return []
    return sorted(base.glob(EVIDENCE_GLOB_md)) + sorted(base.glob("*/report.md"))


def basename_index(tracked: list[str]) -> dict[str, list[str]]:
    index: dict[str, list[str]] = {}
    for path in tracked:
        index.setdefault(Path(path).name, []).append(path)
    return index


def scan_report(
    report: Path,
    tracked_names: dict[str, list[str]],
    root: Path,
) -> list[tuple[Path, int, str]]:
    """Return (report, line, filename) triples whose file is missing or untracked."""
    failures: list[tuple[Path, int, str]] = []
    lines = report.read_text(encoding="utf-8", errors="replace").splitlines()
    recorded_losses: set[str] = set()
    in_loss_block = False
    for line in lines:
        if not line.strip().lstrip(">"):
            in_loss_block = False
        elif LOSS_MARKER in line:
            in_loss_block = True
        if in_loss_block:
            recorded_losses.update(match.rstrip(".") for match in FILENAME_RE.findall(line))
    in_fence = False
    for lineno, line in enumerate(lines, 1):
        if line.lstrip().startswith("```"):
            # A fenced block quotes captured output (a red-before excerpt, a
            # mutation transcript); its filenames are evidence of a check, not
            # a live claim on the tree.
            in_fence = not in_fence
            continue
        if in_fence or LOSS_MARKER in line:
            continue
        for token in FILENAME_RE.findall(line):
            name = token.split("/")[-1].rstrip(".")
            if name in recorded_losses:
                continue
            if token.startswith("/"):
                # An absolute path describes a scratch location, not a tree
                # artifact; the loss gate cannot reach outside the tree.
                continue
            hits = tracked_names.get(name, [])
            on_disk = any((root / path).is_file() for path in hits)
            if not hits or not on_disk:
                failures.append((report, lineno, name))
    return failures


def run(root: Path = ROOT) -> list[tuple[Path, int, str]]:
    tracked = subprocess.run(
        ["git", "ls-files"], cwd=root, capture_output=True, text=True, check=True
    ).stdout.splitlines()
    tracked_names = basename_index(tracked)
    failures: list[tuple[Path, int, str]] = []
    for report in evidence_reports(root):
        failures.extend(scan_report(report, tracked_names, root))
    return failures


def self_test() -> int:
    """Build the four fixture cases in a temp tree and assert the verdicts."""
    import tempfile

    verdicts: list[tuple[str, int, str]] = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        ev = base / "docs/bench-evidence/fixture"
        ev.mkdir(parents=True)
        (ev / "present.log").write_text("ok\n")
        (ev / "on-disk-untracked.log").write_text("ok\n")
        (ev / "extra.log").write_text("unreferenced\n")
        (ev / "README.md").write_text(
            "logs: present.log, missing.log, on-disk-untracked.log\n"
            "EVIDENCE-LOST: gone.log — dropped by the old ignore rule\n"
        )
        tracked = ["docs/bench-evidence/fixture/present.log"]
        failures = scan_report(
            ev / "README.md", basename_index(tracked), base
        )
        named = sorted(name for _, _, name in failures)
        if named != ["missing.log", "on-disk-untracked.log"]:
            verdicts.append(("fixture triage", 1, f"expected missing+untracked, got {named}"))
        else:
            verdicts.append(("fixture triage", 0, "missing and untracked both caught; present and unreferenced pass"))
    for label, code, detail in verdicts:
        print(f"self-test {label}: {'ok' if code == 0 else 'FAIL'} — {detail}")
    return max(code for _, code, _ in verdicts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the built-in fixture cases and exit",
    )
    args = parser.parse_args()
    if args.self_test:
        return self_test()

    failures = run()
    if not failures:
        print("check-evidence-refs: ok — every referenced evidence log is tracked and on disk")
        return 0
    print("check-evidence-refs: evidence references point at files that are missing or untracked:")
    for report, lineno, name in failures:
        print(f"  {report.relative_to(ROOT)}:{lineno}: {name}")
    print(
        "Either restore the file under docs/bench-evidence/, or record the loss "
        f"with an `{LOSS_MARKER}` note naming it."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
