"""Tiny generated projection artifacts: relocation, CLI selection and failures."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools/exl3_reference"))
from artifacts import ArtifactRoot
from extract_projection import digest, headers, write_safetensors
from compare_projection import metrics


class ArtifactTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.base = Path(self.temp.name)
        self.first = self.base / "first"
        self.second = self.base / "unrelated root with spaces"
        self.first.mkdir()
        self.second.mkdir()
        self.raw = b"\0\x3c\0\xbc"
        self.fixture = self.first / "tiny.safetensors"
        write_safetensors(self.fixture, {"activation": ("F16", [1, 2], self.raw)}, {})
        self.receipt = {"schema": 1, "fixture_sha256": digest(self.fixture.read_bytes()),
                        "tensor_hashes": {"activation": {"dtype": "F16", "shape": [1, 2],
                                                         "sha256": digest(self.raw)}}}
        self.write_receipt()
        for name in ("tiny.safetensors", "tiny.json"):
            shutil.copyfile(self.first / name, self.second / name)

    def write_receipt(self):
        (self.first / "tiny.json").write_text(json.dumps(self.receipt))

    def cli(self, root=None, *flags, env_root=None, fixture="tiny.safetensors", cwd=None):
        env = os.environ.copy()
        env.pop("EXL3_ARTIFACT_ROOT", None)
        if env_root is not None:
            env["EXL3_ARTIFACT_ROOT"] = str(env_root)
        command = [sys.executable, str(ROOT / "tools/exl3_reference/capture_projection.py"),
                   "--fixture", str(fixture), "--check-only"]
        if root is not None:
            command += ["--artifact-root", str(root)]
        return subprocess.run(command + list(flags), env=env, cwd=cwd,
                              text=True, capture_output=True, timeout=10)

    def test_relocation_keeps_digest_tensor_identity_and_comparison(self):
        outputs, comparisons = [], []
        for root in (self.first, self.second):
            result = self.cli(root)
            self.assertEqual(result.returncode, 0, result.stderr)
            outputs.append(json.loads(result.stdout))
            path = ArtifactRoot(root).resolve("tiny.safetensors")
            report = headers.read_shard_header(path)
            with path.open("rb") as stream:
                stream.seek(8 + report["header_bytes"])
                comparisons.append(metrics(stream.read(), self.raw, "F16"))
        self.assertEqual(outputs[0], outputs[1])
        self.assertEqual(outputs[0]["fixture_sha256"], self.receipt["fixture_sha256"])
        self.assertEqual(comparisons[0], comparisons[1])
        self.assertTrue(comparisons[0]["bit_exact"])

    def test_cli_root_overrides_environment_and_environment_is_supported(self):
        result = self.cli(self.second, env_root=self.base / "missing root")
        self.assertEqual(result.returncode, 0, result.stderr)
        result = self.cli(env_root=self.first)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout)["fixture_sha256"], self.receipt["fixture_sha256"])

    def test_existing_absolute_filename_without_root_remains_supported(self):
        result = self.cli(fixture=self.fixture)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_corrupt_payload_fails_even_when_external_check_is_optional(self):
        with self.fixture.open("r+b") as stream:
            stream.seek(-1, 2)
            stream.write(b"\xff")
        result = self.cli(self.first, "--optional-artifacts")
        self.assertEqual(result.returncode, 1)
        self.assertIn("fixture SHA-256 mismatch", result.stderr)

    def test_missing_payload_or_manifest_skips_only_optional_external_check(self):
        for name in ("tiny.safetensors", "tiny.json"):
            with self.subTest(name=name):
                path = self.first / name
                original = path.read_bytes()
                path.unlink()
                required = self.cli(self.first)
                optional = self.cli(self.first, "--optional-artifacts")
                self.assertEqual(required.returncode, 1)
                self.assertEqual(optional.returncode, 77)
                self.assertIn("missing artifact", required.stderr)
                self.assertEqual(required.stdout, "")
                path.write_bytes(original)

    def test_incompatible_or_malformed_manifest_is_never_skipped(self):
        for field, value in (("schema", 2), ("schema", True), ("tensor_hashes", None),
                             ("fixture_sha256", "wrong")):
            with self.subTest(field=field, value=value):
                original = self.receipt[field]
                self.receipt[field] = value
                self.write_receipt()
                self.assertEqual(self.cli(self.first, "--optional-artifacts").returncode, 1)
                self.receipt[field] = original
        self.write_receipt()
        spec = self.receipt["tensor_hashes"]["activation"]
        for field, value in (("shape", [True, 2]), ("dtype", "invalid"), ("sha256", "wrong")):
            with self.subTest(field=field, value=value):
                original = spec[field]
                spec[field] = value
                self.write_receipt()
                self.assertEqual(self.cli(self.first, "--optional-artifacts").returncode, 1)
                spec[field] = original

    def test_traversal_and_payload_symlink_escape_fail(self):
        result = self.cli(self.first, "--optional-artifacts", fixture="../tiny.safetensors")
        self.assertEqual(result.returncode, 1)
        self.assertIn("traversal", result.stderr)
        (self.first / "escape.safetensors").symlink_to(self.second / "tiny.safetensors")
        result = self.cli(self.first, fixture="escape.safetensors")
        self.assertEqual(result.returncode, 1)
        self.assertIn("escapes", result.stderr)

    def test_sidecar_symlink_escape_fails_for_explicit_and_default_relative_roots(self):
        (self.first / "tiny.json").unlink()
        (self.first / "tiny.json").symlink_to(self.second / "tiny.json")
        for root in (self.first, None):
            with self.subTest(root=root):
                result = self.cli(root, cwd=self.first)
                self.assertEqual(result.returncode, 1)
                self.assertIn("escapes", result.stderr)


if __name__ == "__main__":
    unittest.main()
