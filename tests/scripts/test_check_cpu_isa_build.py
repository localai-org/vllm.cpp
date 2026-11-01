#!/usr/bin/env python3
"""Mutation tests for the W3 x86 ISA compile-command audit."""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-cpu-isa-build.py"


def entry(source: str, flags: str = "") -> dict[str, str]:
    return {
        "file": str(ROOT / source),
        "command": f"c++ {flags} -c {source}",
    }


def valid_commands() -> list[dict[str, str]]:
    return [
        entry("src/vt/cpu/cpu_isa_x86.cpp"),
        entry("src/vt/cpu/cpu_matmul_elem.cpp"),
        entry("src/vt/cpu/cpu_matmul_elem_avx2.cpp", "-mavx2 -mf16c"),
        entry("src/vt/cpu/cpu_quant_dot_avx2.cpp", "-mavx2 -mf16c"),
        entry(
            "src/vt/cpu/cpu_matmul_elem_avx512.cpp",
            "-mavx512f -mavx512bw -mavx512vl -mf16c",
        ),
    ]


class CpuIsaBuildContract(unittest.TestCase):
    def run_checker(
        self, entries: list[dict[str, str]]
    ) -> subprocess.CompletedProcess[str]:
        with tempfile.TemporaryDirectory() as temporary:
            commands = Path(temporary) / "compile_commands.json"
            commands.write_text(json.dumps(entries), encoding="utf-8")
            return subprocess.run(
                [
                    sys.executable,
                    str(CHECKER),
                    "--compile-commands",
                    str(commands),
                ],
                text=True,
                capture_output=True,
                check=False,
            )

    def test_exact_per_source_flags_pass(self) -> None:
        result = self.run_checker(valid_commands())
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_march_native_anywhere_fails(self) -> None:
        entries = valid_commands()
        entries.append(entry("src/vt/cpu/cpu_ops.cpp", "-march=native"))
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("-march=native", result.stdout + result.stderr)

    def test_baseline_dispatch_source_rejects_avx_leak(self) -> None:
        entries = valid_commands()
        entries[1] = entry("src/vt/cpu/cpu_matmul_elem.cpp", "-mavx2")
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("cpu_matmul_elem.cpp", result.stdout + result.stderr)

    def test_avx2_source_requires_f16c(self) -> None:
        entries = valid_commands()
        entries[2] = entry("src/vt/cpu/cpu_matmul_elem_avx2.cpp", "-mavx2")
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("-mf16c", result.stdout + result.stderr)

    def test_quant_dot_avx2_source_requires_f16c(self) -> None:
        entries = valid_commands()
        entries[3] = entry("src/vt/cpu/cpu_quant_dot_avx2.cpp", "-mavx2")
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("-mf16c", result.stdout + result.stderr)

    def test_avx512_source_requires_exact_feature_set(self) -> None:
        entries = valid_commands()
        entries[4] = entry(
            "src/vt/cpu/cpu_matmul_elem_avx512.cpp", "-mavx512f -mf16c"
        )
        result = self.run_checker(entries)
        self.assertNotEqual(result.returncode, 0)
        output = result.stdout + result.stderr
        self.assertIn("-mavx512bw", output)
        self.assertIn("-mavx512vl", output)

    def test_required_tier_source_must_be_compiled(self) -> None:
        result = self.run_checker(valid_commands()[:-1])
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("missing", result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
