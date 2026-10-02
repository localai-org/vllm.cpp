#!/usr/bin/env python3
"""Bind the public supported-architecture list to the C++ model registry.

The "supported models" claim on docs/FEATURES.md must never drift from what the
engine actually registers. The authority on "supported" is the C++ registry:
each architecture self-registers from its own translation unit via
`REGISTER_VLLM_MODEL(symbol, "ArchString", factory, info)`
(src/vllm/model_executor/models/*.cpp). `ModelRegistry::SupportedArchs()`
enumerates exactly those strings, and tests/vllm/test_model_loader_gguf.cpp pins
the same set as the canonical "Supported architectures:" error string, so the
C++ side is already self-consistent. This checker closes the loop to the public
doc: the registered set and the FEATURES list must be equal.

It asserts, over docs/FEATURES.md's marked supported-architecture table:
  1. every registered architecture has a row (no silently-supported arch missing
     from the public list), and
  2. no row claims an architecture the code does not register (no aspirational or
     stale entry on the public list).

The table is delimited by
  <!-- supported-arch-table:begin --> ... <!-- supported-arch-table:end -->
so non-registered lanes (standalone audio/diffusion forwards, speculator draft
heads, and inventoried-but-blocked archs) can be described elsewhere on the page
without tripping the registry-equality check. Only the FIRST column (the
Architecture key) of the rows inside that block is scanned, so an arch name
mentioned in a prose cell never counts as a claim.

The validation logic is the pure function `supported_models_errors(registered,
features_text)` so it is unit- and mutation-testable (see
tests/scripts/test_check_supported_models.py), mirroring
check-benchmark-index.py and check-model-checklist.py. Per AGENTS.md, never
weaken this checker to make the public list pass: fix the list or register the
arch.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "src/vllm/model_executor/models"
FEATURES = ROOT / "docs/FEATURES.md"

BEGIN_MARKER = "<!-- supported-arch-table:begin -->"
END_MARKER = "<!-- supported-arch-table:end -->"

# The macro CALL form, e.g. REGISTER_VLLM_MODEL(olmo3, "Olmo3ForCausalLM", ...).
# The comments that merely mention the macro name ("via REGISTER_VLLM_MODEL")
# lack the "(symbol, \"Arch\"" shape and do not match.
REGISTER_RE = re.compile(
    r"""REGISTER_VLLM_MODEL\(\s*[A-Za-z0-9_]+\s*,\s*"([^"]+)"""
)

# Every registrable architecture string ends in one of these HF class suffixes
# (or the GLiNER2 "Extractor" suffix); the arch key in the FEATURES table is
# written verbatim in backticks. If a future registered arch stops matching this,
# the self-check below fails loudly rather than silently dropping it.
# `_V<n>` covers upstream's versioned wrapper names, which carry no task suffix
# (`NemotronH_Nano_VL_V2`, `NemotronH_Nano_Omni_Reasoning_V3`,
# registry.py:512-513 @ e126687a9a).
ARCH_TOKEN_RE = re.compile(
    r"`([A-Za-z0-9_]+(?:For(?:CausalLM|ConditionalGeneration|CTC|RNNT|TDT)"
    r"|Model|Extractor|Forms|_V[0-9]+))`"
)


def parse_registered_archs(model_dir: Path) -> set[str]:
    """Return the architecture strings registered in the C++ model TUs."""
    archs: set[str] = set()
    for path in sorted(model_dir.glob("*.cpp")):
        for match in REGISTER_RE.finditer(path.read_text(encoding="utf-8")):
            archs.add(match.group(1))
    return archs


def _supported_block(text: str) -> str | None:
    """Return the text between the supported-arch-table markers, or None."""
    start = text.find(BEGIN_MARKER)
    end = text.find(END_MARKER)
    if start == -1 or end == -1 or end < start:
        return None
    return text[start + len(BEGIN_MARKER) : end]


def _is_separator_row(cells: list[str]) -> bool:
    return bool(cells) and all(set(cell) <= set("-: ") for cell in cells)


def parse_features_archs(features_text: str) -> set[str]:
    """Return arch keys from column 1 of the marked FEATURES table.

    Returns the empty set if the marked block is absent; the caller turns that
    into an explicit error so a deleted table can never read as "no drift".
    """
    block = _supported_block(features_text)
    if block is None:
        return set()
    archs: set[str] = set()
    for raw in block.splitlines():
        stripped = raw.strip()
        if not (stripped.startswith("|") and stripped.endswith("|")):
            continue
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        if _is_separator_row(cells) or not cells:
            continue
        # Column 1 is the Architecture key; ignore the header row (which has no
        # backticked arch token) and any arch mentioned in a later cell.
        for token in ARCH_TOKEN_RE.findall(cells[0]):
            archs.add(token)
    return archs


def supported_models_errors(registered: set[str], features_text: str) -> list[str]:
    """Return drift between the registry and the public supported list."""
    errors: list[str] = []

    if not registered:
        errors.append(
            "no REGISTER_VLLM_MODEL registrations found under "
            "src/vllm/model_executor/models/; the registry parser is broken or "
            "the source moved, so the public list cannot be verified"
        )
        return errors

    # Guard the parser itself: every registered arch must be expressible as a
    # FEATURES arch token, or the equality below would silently exclude it.
    unrepresentable = sorted(a for a in registered if not ARCH_TOKEN_RE.match(f"`{a}`"))
    if unrepresentable:
        errors.append(
            "registered architecture(s) do not match the FEATURES arch-token "
            f"pattern ({', '.join(unrepresentable)}); broaden ARCH_TOKEN_RE in "
            "scripts/check-supported-models.py so they are compared, never dropped"
        )

    if _supported_block(features_text) is None:
        errors.append(
            "docs/FEATURES.md is missing the supported-arch-table markers "
            f"({BEGIN_MARKER} ... {END_MARKER}); they delimit the registry-bound "
            "list, and without them the public list is unverifiable"
        )
        return errors

    listed = parse_features_archs(features_text)

    missing = sorted(registered - listed)
    if missing:
        errors.append(
            "registered architectures with no docs/FEATURES.md supported-arch "
            f"row: {', '.join(missing)}; the engine registers them, so the "
            "public list must name each with its tested checkpoint and gate"
        )

    unregistered = sorted(listed - registered)
    if unregistered:
        errors.append(
            "docs/FEATURES.md supported-arch rows claim architectures the code "
            f"does not register: {', '.join(unregistered)}; move them to the "
            "standalone-lanes or inventoried-but-blocked table (outside the "
            "markers), or register the arch"
        )

    return errors


def main() -> int:
    if not FEATURES.exists():
        print(f"ERROR: {FEATURES.relative_to(ROOT)} is missing", file=sys.stderr)
        return 1
    registered = parse_registered_archs(MODEL_DIR)
    errors = supported_models_errors(registered, FEATURES.read_text(encoding="utf-8"))
    if errors:
        print(
            "ERROR: the public supported-model list has drifted from the "
            "C++ registry:",
            file=sys.stderr,
        )
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(
        f"OK: docs/FEATURES.md lists exactly the {len(registered)} architectures "
        "registered via REGISTER_VLLM_MODEL."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
