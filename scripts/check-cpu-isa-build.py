#!/usr/bin/env python3
"""Audit W3 x86 CPU tier isolation in CMake compile commands."""

from __future__ import annotations

import argparse
import json
import shlex
import sys
from pathlib import Path


BASELINE_SOURCES = (
    "src/vt/cpu/cpu_isa_x86.cpp",
    "src/vt/cpu/cpu_matmul_elem.cpp",
)
TIER_FLAGS = {
    "src/vt/cpu/cpu_matmul_elem_avx2.cpp": frozenset(("-mavx2", "-mf16c")),
    "src/vt/cpu/cpu_quant_dot_avx2.cpp": frozenset(("-mavx2", "-mf16c")),
    "src/vt/cpu/cpu_matmul_elem_avx512.cpp": frozenset(
        ("-mavx512f", "-mavx512bw", "-mavx512vl", "-mf16c")
    ),
}


def relative_source(value: str) -> str | None:
    normalized = value.replace("\\", "/")
    marker = "/src/"
    if marker not in normalized:
        return None
    return "src/" + normalized.split(marker, 1)[1]


def command_arguments(entry: dict[str, object]) -> list[str]:
    arguments = entry.get("arguments")
    if isinstance(arguments, list):
        return [str(value) for value in arguments]
    return shlex.split(str(entry.get("command", "")))


def explicit_isa_flags(arguments: list[str]) -> frozenset[str]:
    return frozenset(
        argument
        for argument in arguments
        if argument.startswith("-mavx") or argument == "-mf16c"
    )


def validate_compile_commands(path: Path) -> list[str]:
    entries = json.loads(path.read_text(encoding="utf-8"))
    commands: dict[str, list[str]] = {}
    errors: list[str] = []
    for entry in entries:
        arguments = command_arguments(entry)
        source = relative_source(str(entry.get("file", "")))
        label = source or str(entry.get("file", "<unknown>"))
        if "-march=native" in arguments:
            errors.append(f"{label}: forbidden -march=native")
        if source is not None:
            commands[source] = arguments

    for source in BASELINE_SOURCES:
        if source not in commands:
            errors.append(f"missing CPU compile command for {source}")
            continue
        leaked = sorted(explicit_isa_flags(commands[source]))
        if leaked:
            errors.append(f"{source}: baseline dispatcher leaks ISA flags {leaked}")

    for source, required in TIER_FLAGS.items():
        if source not in commands:
            errors.append(f"missing CPU compile command for {source}")
            continue
        actual = explicit_isa_flags(commands[source])
        missing = sorted(required.difference(actual))
        extra = sorted(actual.difference(required))
        if missing:
            errors.append(f"{source}: missing required flags {missing}")
        if extra:
            errors.append(f"{source}: undeclared ISA flags {extra}")
    return errors


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--compile-commands", type=Path, required=True)
    args = parser.parse_args()

    errors = validate_compile_commands(args.compile_commands)
    if errors:
        print("CPU ISA build audit FAILED:", file=sys.stderr)
        for error in errors:
            print(f"  - {error}", file=sys.stderr)
        return 1
    print("CPU ISA build audit: portable baseline and exact x86 tiers OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
