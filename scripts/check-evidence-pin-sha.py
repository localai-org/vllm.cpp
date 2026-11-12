#!/usr/bin/env python3
"""Fail when an evidence report's artifact sha256 contradicts its oracle pin.

Motivated by the FLINT-CPU-PARITY review (M3): the pin record
`.agents/oracles/llama-cpp-flint.md` and its evidence report
`docs/bench-evidence/flint-cpu-parity-20261029/report.md` each record the
measured artifact's sha256 in prose, and nothing compared them. A sha edited
or re-quantized on one side only would leave both files individually
consistent while the record disagreed with itself.

TRIGGER (deliberately narrow): for every `oracle-pin` block whose `evidence`
field resolves to a file under `docs/bench-evidence/`, the rule fires only
when the evidence file records EXACTLY ONE distinct 64-hex sha256 AND the pin
file records EXACTLY ONE distinct 64-hex sha256 — and then the two must be
equal. Everything else stays unchecked, on purpose:

* an evidence file that names several artifacts (a ladder report, a manifest
  table) has no single sha to bind, and guessing which one the pin means
  would invent semantics the records do not carry;
* a pin file that names several shas (a multi-artifact pin, a prior-pin
  mention) likewise has no single claim;
* an `evidence = #NNN` issue reference, or a path outside
  `docs/bench-evidence/`, names no measured artifact report at all.

This is a floor, not a ceiling: it must catch the observed failure (one file
re-quantized, the other not), not intent. Reports that record no sha pass.

Standard library only, no network.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ORACLES = ROOT / ".agents/oracles"
EVIDENCE_PREFIX = "docs/bench-evidence/"

BLOCK = re.compile(r"^```oracle-pin\n(.*?)^```", re.M | re.S)
FIELD = re.compile(r"^([a-z_]+)\s*=\s*(.*)$", re.M)
SHA256 = re.compile(r"\b[0-9a-fA-F]{64}\b")


def distinct_shas(text: str) -> set[str]:
    return set(SHA256.findall(text))


def pin_evidence_pairs(
    oracles_dir: Path,
    root: Path = ROOT,
) -> list[tuple[str, Path]]:
    """(oracle id, evidence path) for pins naming a docs/bench-evidence file."""
    pairs: list[tuple[str, Path]] = []
    for path in sorted(oracles_dir.glob("*.md")):
        if path.name == "README.md":
            continue
        text = path.read_text(encoding="utf-8", errors="replace")
        for block in BLOCK.findall(text):
            fields = dict(FIELD.findall(block))
            evidence = fields.get("evidence", "").strip()
            oracle_id = fields.get("id", path.stem).strip()
            if not evidence.startswith(EVIDENCE_PREFIX):
                continue
            evidence_path = root / evidence
            if evidence_path.is_file():
                pairs.append((oracle_id, evidence_path))
    return pairs


def check_pair(pin_text: str, evidence_text: str) -> str | None:
    """The disagreement message, or None when the narrow trigger does not fire."""
    pin_shas = distinct_shas(pin_text)
    evidence_shas = distinct_shas(evidence_text)
    if len(evidence_shas) != 1 or len(pin_shas) != 1:
        return None
    evidence_sha = next(iter(evidence_shas))
    pin_sha = next(iter(pin_shas))
    if evidence_sha == pin_sha:
        return None
    return (
        f"evidence report and oracle pin disagree on the artifact sha256: "
        f"the report records {evidence_sha}, the pin file records {pin_sha}"
    )


def run(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for oracle_id, evidence_path in pin_evidence_pairs(root / ".agents/oracles"):
        pin_path = root / ".agents/oracles" / f"{oracle_id}.md"
        if not pin_path.is_file():
            continue
        message = check_pair(
            pin_path.read_text(encoding="utf-8", errors="replace"),
            evidence_path.read_text(encoding="utf-8", errors="replace"),
        )
        if message is not None:
            pin_rel = pin_path.relative_to(root)
            ev_rel = evidence_path.relative_to(root)
            errors.append(f"{pin_rel} vs {ev_rel}: {message}")
    return errors


def self_test() -> int:
    """Build the fixture cases in a temp tree and assert the verdicts."""
    import tempfile

    sha_a = "a" * 64
    sha_b = "b" * 64
    pin = (
        "```oracle-pin\n"
        "id = fixture\n"
        "role = secondary\n"
        f"evidence = {EVIDENCE_PREFIX}fixture/report.md\n"
        "```\n"
    )
    agree = pin + f"sha256 `{sha_a}` over the artifact\n"
    disagree = pin + f"sha256 `{sha_b}` over the artifact\n"
    cases: list[tuple[str, str, str, bool]] = [
        ("agree", agree, f"sha256 `{sha_a}` recorded\n", False),
        ("disagree", disagree, f"sha256 `{sha_a}` recorded\n", True),
        # The narrow trigger: several shas on either side stay unchecked.
        (
            "report names two artifacts",
            pin + f"sha256 `{sha_a}`\n",
            f"`{sha_a}` and `{sha_b}`\n",
            False,
        ),
        (
            "pin names two shas",
            pin + f"`{sha_a}` and the prior `{sha_b}`\n",
            f"sha256 `{sha_a}`\n",
            False,
        ),
        ("report names no sha", agree, "no hash here\n", False),
        ("issue-reference evidence stays unchecked", agree, "anything\n", False),
    ]
    failures: list[str] = []
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / ".agents/oracles").mkdir(parents=True)
        (base / (EVIDENCE_PREFIX + "fixture")).mkdir(parents=True)
        for name, pin_text, evidence_text, bad in cases:
            if name == "issue-reference evidence stays unchecked":
                issue_pin = pin.replace(
                    f"evidence = {EVIDENCE_PREFIX}fixture/report.md",
                    "evidence = #647",
                )
                record = check_pair(issue_pin, evidence_text)
                reported = record is not None
            else:
                record = check_pair(pin_text, evidence_text)
                reported = record is not None
            if reported != bad:
                verb = "was reported" if reported else "was not reported"
                failures.append(f"fixture {name!r} {verb}: {record}")
    for failure in failures:
        print(f"self-test: {failure}", file=sys.stderr)
    if failures:
        print(f"self-test FAILED ({len(failures)} of {len(cases)})", file=sys.stderr)
        return 1
    print(f"self-test ok ({len(cases)} evidence-pin-sha fixtures)")
    return 0


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

    errors = run()
    if not errors:
        print(
            "check-evidence-pin-sha: ok — every single-sha evidence report "
            "agrees with its oracle pin"
        )
        return 0
    print("check-evidence-pin-sha: evidence reports disagree with their pins:")
    for error in errors:
        print(f"  {error}")
    print(
        "Re-measure the artifact and record the SAME sha256 in both the "
        "evidence report and the oracle pin, or re-quantize whichever file "
        "recorded the stale hash."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
