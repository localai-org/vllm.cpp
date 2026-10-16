"""Host-only exit-code contract for external C++ test admission."""
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

PROBE = sys.argv.pop(1)


class ExternalArtifacts(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="exl3 inputs ")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.payload = self.root / "tiny payload"
        self.payload.write_bytes(b"not validated by admission")

    def run_probe(self, kind, name, mode=None, value=None):
        env = os.environ.copy()
        env.pop("EXL3_REQUIRE_ARTIFACTS", None)
        env.pop("EXL3_TEST_INPUT", None)
        if mode is not None:
            env["EXL3_REQUIRE_ARTIFACTS"] = mode
        if value is not None:
            env["EXL3_TEST_INPUT"] = value
        return subprocess.run([PROBE, kind, str(name)], env=env,
                              text=True, capture_output=True, timeout=10)

    def test_valid_paths_with_spaces_and_explicit_environment(self):
        for mode in (None, "0", "1"):
            for kind, path in (("file", self.payload), ("dir", self.root)):
                self.assertEqual(self.run_probe(kind, path, mode).returncode, 0)
            self.assertEqual(self.run_probe("env", "EXL3_TEST_INPUT", mode,
                                            str(self.root)).returncode, 0)

    def test_missing_or_empty_environment_optional_skips_required_fails(self):
        for value in (None, ""):
            for mode, expected in ((None, 77), ("0", 77), ("1", 1)):
                result = self.run_probe("env", "EXL3_TEST_INPUT", mode, value)
                self.assertEqual(result.returncode, expected)
                self.assertIn("missing EXL3 test artifact", result.stderr)

    def test_missing_file_or_directory_optional_skips_required_fails(self):
        for kind in ("file", "dir"):
            for mode, expected in ((None, 77), ("0", 77), ("1", 1)):
                self.assertEqual(self.run_probe(kind, self.root / "missing",
                                                mode).returncode, expected)

    def test_wrong_type_fails_even_in_optional_mode(self):
        for mode in (None, "0", "1"):
            for kind, path in (("file", self.root), ("dir", self.payload)):
                result = self.run_probe(kind, path, mode)
                self.assertEqual(result.returncode, 1)
                self.assertIn("wrong artifact type", result.stderr)

    def test_invalid_admission_mode_never_skips(self):
        for mode in ("", "2", "true"):
            for path in (self.payload, self.root / "missing"):
                result = self.run_probe("file", path, mode)
                self.assertEqual(result.returncode, 1)
                self.assertIn("must be0 or1", result.stderr)


if __name__ == "__main__":
    unittest.main()
