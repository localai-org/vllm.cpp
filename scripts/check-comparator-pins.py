#!/usr/bin/env python3
"""Validate pinned implementation/performance comparators.

Comparators on this surface are deliberately not correctness oracles.  Their
records pin implementations and deployment recipes used for mechanism study or
performance measurement without changing the secondary-oracle policy.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COMPARATORS = ROOT / ".agents" / "comparators"
BLOCK = re.compile(r"^```comparator-pin\n(.*?)^```", re.M | re.S)
FIELD = re.compile(r"^([a-z_]+)\s*=\s*(.*)$")
FULL_SHA = re.compile(r"^[0-9a-f]{40}$")
ISO_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
REQUIRED_KEYS = (
    "id",
    "role",
    "upstream",
    "pin",
    "recipe_upstream",
    "recipe_pin",
    "scope",
    "correctness",
    "pinned_on",
)


def check_registry() -> list[str]:
    errors: list[str] = []
    records = sorted(COMPARATORS.glob("*.md")) if COMPARATORS.exists() else []
    records = [path for path in records if path.name != "README.md"]
    if not records:
        return [f"{COMPARATORS}: comparator registry has no records"]

    for path in records:
        text = path.read_text(encoding="utf-8")
        blocks = BLOCK.findall(text)
        if len(blocks) != 1:
            errors.append(f"{path}: expected exactly one comparator-pin block")
            continue
        fields: dict[str, str] = {}
        for line in blocks[0].splitlines():
            match = FIELD.fullmatch(line)
            if not match:
                errors.append(f"{path}: malformed comparator field: {line!r}")
                continue
            key, value = match.groups()
            if key in fields:
                errors.append(f"{path}: duplicate key {key}")
            fields[key] = value.strip()
        missing = [key for key in REQUIRED_KEYS if not fields.get(key)]
        unknown = sorted(set(fields) - set(REQUIRED_KEYS))
        if missing:
            errors.append(f"{path}: missing keys: {', '.join(missing)}")
        if unknown:
            errors.append(f"{path}: unknown keys: {', '.join(unknown)}")
        if fields.get("id") != path.stem:
            errors.append(f"{path}: id must match filename stem")
        if fields.get("role") != "implementation-performance":
            errors.append(f"{path}: role must be implementation-performance")
        if fields.get("correctness") != "not-an-oracle":
            errors.append(f"{path}: correctness must be not-an-oracle")
        for key in ("pin", "recipe_pin"):
            value = fields.get(key, "")
            if value and not FULL_SHA.fullmatch(value):
                errors.append(f"{path}: {key} must be a lowercase 40-hex commit")
        if fields.get("pinned_on") and not ISO_DATE.fullmatch(fields["pinned_on"]):
            errors.append(f"{path}: pinned_on must be an ISO date")
        for key in ("upstream", "recipe_upstream"):
            value = fields.get(key, "")
            if value and not value.startswith("https://github.com/"):
                errors.append(f"{path}: {key} must be a canonical GitHub URL")
    return errors


def main() -> int:
    errors = check_registry()
    if errors:
        for error in errors:
            print(f"ERROR: {error}", file=sys.stderr)
        return 1
    print("comparator pins: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
