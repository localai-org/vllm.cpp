#!/usr/bin/env python3
"""The default-off contract of VT_FLINT_PHASE_TIMING, pinned on the binary.

`VT_FLINT_PHASE_TIMING` (docs/ENVIRONMENT.md, row TT-NONROUTED-ATTRIBUTION)
gates the `[FLINT-PHASE]` / `[FLINT-STEP]` timers around the production
`FlintForwardPaged` call sites (`src/vllm/model_executor/models/flint.cpp`).
Default OFF means an unset run must produce ZERO such lines on the smallest
binary that passes through the instrumented forward — `test_flint_paged` —
and `=1` must produce them. This is the runtime half of that contract: the
static gate (`check-env-doc.py`) can prove the knob is documented, never that
an unset run stays silent.

The run is skipped BY NAME when the box has no built `test_flint_paged`, so a
missing build directory cannot read as a pass. Point `VLLM_CPP_FLINT_TEST_BIN`
at the binary, or build one (`cmake --build <dir> --target test_flint_paged`).
"""

from __future__ import annotations

import os
import subprocess
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

PHASE_MARKERS = ("[FLINT-PHASE]", "FLINT-STEP")


def binary() -> Path | None:
    override = os.environ.get("VLLM_CPP_FLINT_TEST_BIN")
    if override:
        candidate = Path(override)
        return candidate if candidate.is_file() else None
    for build_dir in sorted(ROOT.glob("build*")):
        candidate = build_dir / "tests" / "test_flint_paged"
        if candidate.is_file():
            return candidate
    return None


HAVE_BIN = binary() is not None


def run_test_binary(bin_path: Path, timing: str | None) -> str:
    env = dict(os.environ)
    if timing is None:
        env.pop("VT_FLINT_PHASE_TIMING", None)
    else:
        env["VT_FLINT_PHASE_TIMING"] = timing
    proc = subprocess.run(
        [str(bin_path)],
        capture_output=True,
        text=True,
        env=env,
        timeout=600,
        check=False,
    )
    assert proc.returncode == 0, f"test_flint_paged exited {proc.returncode}"
    return proc.stderr


class PhaseTimingDefaultOffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.bin_path = binary()

    @unittest.skipUnless(HAVE_BIN, "no built test_flint_paged on this box "
                         "(build one, or set VLLM_CPP_FLINT_TEST_BIN)")
    def test_unset_env_is_silent(self) -> None:
        stderr = run_test_binary(self.bin_path, timing=None)
        for marker in PHASE_MARKERS:
            self.assertNotIn(marker, stderr)

    @unittest.skipUnless(HAVE_BIN, "no built test_flint_paged on this box "
                         "(build one, or set VLLM_CPP_FLINT_TEST_BIN)")
    def test_equal_one_emits_phase_lines(self) -> None:
        stderr = run_test_binary(self.bin_path, timing="1")
        self.assertIn("[FLINT-PHASE]", stderr)


if __name__ == "__main__":
    unittest.main()
