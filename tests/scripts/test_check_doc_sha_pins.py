#!/usr/bin/env python3
"""Unit checks for scripts/check-doc-sha-pins.py."""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-doc-sha-pins.py"
SPEC = importlib.util.spec_from_file_location("check_doc_sha_pins", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)

SHA_A = "a" * 64
SHA_B = "b" * 64


class SelfTestTests(unittest.TestCase):
    def test_self_test_is_green(self) -> None:
        self.assertEqual(mod.self_test(), 0)


class CheckSurfaceTests(unittest.TestCase):
    def surface(self, tmp: str, line: str) -> Path:
        base = Path(tmp)
        (base / "docs/bench-evidence/fixture").mkdir(parents=True, exist_ok=True)
        surface = base / "docs/bench-evidence/fixture/report.md"
        surface.write_text(line, encoding="utf-8")
        return surface

    def artifact(self, tmp: str, name: str = "model-q4.gguf") -> tuple[Path, str]:
        base = Path(tmp)
        path = base / "docs/bench-evidence/fixture" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"fixture weights\n")
        return path, mod.file_sha256(path)

    def test_a_corrupted_sha_names_the_surface_line_path_and_both_hashes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path, actual = self.artifact(tmp)
            surface = self.surface(
                tmp, f"Artifact: `{path.name}` sha256 `{SHA_B}`\n"
            )
            errors = mod.check_surface(surface, Path(tmp))
            self.assertEqual(len(errors), 1)
            self.assertIn("report.md:1", errors[0])
            self.assertIn(path.name, errors[0])
            self.assertIn(SHA_B, errors[0])
            self.assertIn(actual, errors[0])

    def test_the_document_relative_path_resolves(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            _, actual = self.artifact(tmp)
            surface = self.surface(
                tmp, f"Artifact: `model-q4.gguf` sha256 `{actual}`\n"
            )
            self.assertEqual(mod.check_surface(surface, Path(tmp)), [])

    def test_a_repo_relative_path_resolves_from_the_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "weights").mkdir()
            artifact = base / "weights/model-q4.gguf"
            artifact.write_bytes(b"fixture weights\n")
            actual = mod.file_sha256(artifact)
            surface = self.surface(
                tmp, f"Artifact: `weights/model-q4.gguf` sha256 `{actual}`\n"
            )
            self.assertEqual(mod.check_surface(surface, base), [])

    def test_an_artifact_path_absent_from_the_tree_stays_unchecked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            surface = self.surface(
                tmp, f"Artifact: `absent.gguf` sha256 `{SHA_A}`\n"
            )
            self.assertEqual(mod.check_surface(surface, Path(tmp)), [])

    def test_a_sha_beside_a_document_path_stays_unchecked(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            surface = self.surface(
                tmp,
                f"See `.agents/oracles/fixture.md` sha256 `{SHA_A}`\n",
            )
            self.assertEqual(mod.check_surface(surface, Path(tmp)), [])

    def test_an_index_json_dotted_name_is_not_an_artifact_token(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            surface = self.surface(
                tmp, f"`model.safetensors.index.json` {SHA_A}\n"
            )
            self.assertEqual(mod.check_surface(surface, Path(tmp)), [])

    def test_a_64_hex_sha_inside_a_longer_token_is_not_matched(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            surface = self.surface(
                tmp, f"sha256 0{SHA_A}9 over `model-q4.gguf`\n"
            )
            self.assertEqual(mod.check_surface(surface, Path(tmp)), [])

    def test_only_readme_and_report_md_evidence_files_are_surfaces(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            (base / "docs/bench-evidence/fixture").mkdir(parents=True)
            other = base / "docs/bench-evidence/fixture/notes.md"
            other.write_text("not a surface\n", encoding="utf-8")
            self.assertNotIn(other, mod.surface_files(base))

    def test_a_missing_usage_document_is_tolerated(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(mod.run(Path(tmp)), [])


class FileSha256Tests(unittest.TestCase):
    def test_matches_hashlib(self) -> None:
        import hashlib

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "blob"
            path.write_bytes(b"x" * (mod.CHUNK * 2 + 7))
            self.assertEqual(
                mod.file_sha256(path), hashlib.sha256(b"x" * (mod.CHUNK * 2 + 7)).hexdigest()
            )


if __name__ == "__main__":
    sys.exit(unittest.main())
