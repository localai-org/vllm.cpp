#!/usr/bin/env python3
"""Fail when a documented artifact sha256 disagrees with the file on disk.

Motivated by the FLINT-KQ-CPU-PARITY review (M3) and the kolibri1-GGUF review
(M2, CHECKER-HARDENING-2): documentation artifacts record a 64-hex sha256
beside an artifact path, and nothing compared the recorded hash with the
file's content. A re-quantized or re-uploaded artifact recorded on one side
only left the document individually consistent while the pin pointed at bytes
nobody can reproduce.

SURFACE: `docs/USAGE.md` and every `README.md` / `report.md` under
`docs/bench-evidence/` -- the two projections that name measured artifacts to
a reader who wants to reproduce them.

TRIGGER (deliberately narrow, like `check-evidence-pin-sha.py`): a line of a
surface is in scope only when it contains BOTH a 64-hex sha256 AND a relative
path token whose extension is a measured-artifact extension (`.gguf`,
`.safetensors`, `.pt`, `.bin`, `.ckpt`). Only tokens whose extension is an
artifact extension are read, so the ordinary prose shapes stay free:

* a sha recorded beside a document path (`.agents/oracles/<id>.md`,
  `report.md`) names a citation, not a hashed artifact, and stays unchecked --
  the red-first captures of the prior row quote checker output that records a
  GGUF sha beside two `.md` paths, and a rule over those lines would red on
  evidence of the checker itself;
* a sha recorded with NO path (an upstream LFS object id, a repo revision)
  names no file this tree holds, and stays unchecked;
* a checkpoint sha beside a path that does not exist in this tree names an
  upstream artifact nobody can hash offline, and stays unchecked -- this
  checker is network-free on purpose;
* the HuggingFace revision ids (`@ <40-hex>`) are 40 characters, not 64, and
  never match.

When the trigger fires and the path resolves (relative to the surface's
directory first, then to the repository root), the file is hashed with a
buffered streaming read and the two values must be equal. A mismatch fails
naming the surface, the line, the path, and both hashes.

This is a floor, not a ceiling: it must catch the observed failure (one side
re-hashed, the other not), not intent. Today the tree carries no sha+artifact
pair whose file resolves, so the checker is green on the untouched tree and
exercises its rule through the fixtures below and its unit suite.

Standard library only, no network.
"""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
USAGE = ROOT / "docs/USAGE.md"
EVIDENCE = ROOT / "docs/bench-evidence"

SHA256 = re.compile(r"\b[0-9a-fA-F]{64}\b")
# A relative path token with a measured-artifact extension. Word boundaries on
# both sides so a longer dotted token (a URL path, `model.safetensors.index`)
# never yields a phantom artifact name.
ARTIFACT_PATH = re.compile(
    r"(?<![\w./-])[\w][\w.-]*(?:/[\w][\w.-]*)*\.(?:gguf|safetensors|pt|bin|ckpt)"
    r"(?![\w.-])"
)
CHUNK = 1 << 20


def surface_files(root: Path = ROOT) -> list[Path]:
    """The documentation surfaces the rule reads."""
    files = [root / "docs/USAGE.md"]
    if (root / "docs/bench-evidence").is_dir():
        files.extend(
            path
            for path in sorted((root / "docs/bench-evidence").rglob("*.md"))
            if path.name in ("README.md", "report.md")
        )
    return files


def file_sha256(path: Path) -> str:
    """The buffered streaming sha256 of *path*."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_artifact(token: str, surface: Path, root: Path) -> Path | None:
    """The on-disk file *token* names, from the surface's directory or the root."""
    for base in (surface.parent, root):
        candidate = base / token
        if candidate.is_file():
            return candidate
    return None


def check_surface(surface: Path, root: Path = ROOT) -> list[str]:
    """Every mismatch in one surface, naming the line, the path, and both hashes."""
    errors: list[str] = []
    try:
        text = surface.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return errors
    for line_no, line in enumerate(text.splitlines(), 1):
        shas = set(SHA256.findall(line))
        if not shas:
            continue
        for token in ARTIFACT_PATH.findall(line):
            artifact = resolve_artifact(token, surface, root)
            if artifact is None:
                continue  # Not in this tree: nothing to hash, no claim to test.
            actual = file_sha256(artifact)
            if actual in shas:
                continue
            recorded = ", ".join(sorted(shas))
            errors.append(
                f"{surface.relative_to(root)}:{line_no}: the recorded sha256 does "
                f"not match {token} on disk: the document records {recorded}, "
                f"the file hashes to {actual}"
            )
    return errors


def run(root: Path = ROOT) -> list[str]:
    errors: list[str] = []
    for surface in surface_files(root):
        errors.extend(check_surface(surface, root))
    return errors


def self_test() -> int:
    """Build the fixture tree in a temp dir and assert the verdicts."""
    import tempfile

    sha_a = "a" * 64
    sha_b = "b" * 64
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        (base / "docs/bench-evidence/fixture").mkdir(parents=True)
        artifact = base / "docs/bench-evidence/fixture/model-q4.gguf"
        artifact.write_bytes(b"fixture weights\n")
        actual = file_sha256(artifact)
        surface = base / "docs/bench-evidence/fixture/README.md"
        agree = f"Artifact: `model-q4.gguf` sha256 `{actual}`\n"
        disagree = f"Artifact: `model-q4.gguf` sha256 `{sha_b}`\n"
        cases: list[tuple[str, str, bool]] = [
            ("agreeing sha", agree, False),
            ("corrupted sha", disagree, True),
            # The narrow triggers: these lines name no hashable artifact.
            (
                "sha beside a document path",
                f"Compared against `.agents/oracles/fixture.md` sha256 `{sha_a}`\n",
                False,
            ),
            ("sha with no path at all", f"upstream LFS id {sha_a}\n", False),
            ("artifact path that is not in the tree", disagree.replace("model-q4.gguf", "absent.gguf"), False),
            ("no sha on the line", "Artifact: `model-q4.gguf`\n", False),
            ("a 40-character revision is not a sha256", "@ `e52eb462aabbccddeeff00112233445566778899`\n", False),
        ]
        failures: list[str] = []
        for name, line, bad in cases:
            surface.write_text(line, encoding="utf-8")
            errors = check_surface(surface, base)
            reported = bool(errors)
            if reported != bad:
                verb = "was reported" if reported else "was not reported"
                failures.append(f"fixture {name!r} {verb}: {errors}")
    for failure in failures:
        print(f"self-test: {failure}", file=sys.stderr)
    if failures:
        print(f"self-test FAILED ({len(failures)} of {len(cases)})", file=sys.stderr)
        return 1
    print(f"self-test ok ({len(cases)} doc-sha-pin fixtures)")
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
            "check-doc-sha-pins: ok — every documented artifact sha256 that "
            "resolves to a file agrees with the file's content hash"
        )
        return 0
    print("check-doc-sha-pins: documented sha256 pins disagree with disk:")
    for error in errors:
        print(f"  {error}")
    print(
        "Re-measure the artifact and record the SAME sha256 in the document, "
        "or re-publish whichever side recorded the stale hash."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
