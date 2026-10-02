#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-supported-models.py.

The checker binds docs/FEATURES.md's supported-architecture table to the C++
model registry (REGISTER_VLLM_MODEL). These tests prove both directions of the
equality are enforced, that only the marked block's first column is scanned, and
that the shipped page and source agree today.
"""

from __future__ import annotations

import importlib.util
import sys
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load(name: str, relative: str):
    path = ROOT / relative
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


chk = _load("check_supported_models", "scripts/check-supported-models.py")


def _features(col1_rows: list[str], *, markers: bool = True) -> str:
    """Build a minimal FEATURES page with a supported-arch table."""
    body = [
        "| Architecture | Checkpoint | Gate | Speed |",
        "|---|---|---|---|",
        *[f"| {c} | ckpt | strict | pending |" for c in col1_rows],
    ]
    table = "\n".join(body)
    if markers:
        table = f"{chk.BEGIN_MARKER}\n{table}\n{chk.END_MARKER}"
    return "\n".join(["# Features", "", "### Registered architectures", "", table, ""])


TWO = ["`FooForCausalLM`", "`BarForConditionalGeneration`"]
REGISTERED = {"FooForCausalLM", "BarForConditionalGeneration"}


class ShippedTreeTests(unittest.TestCase):
    def test_shipped_features_matches_the_shipped_registry(self) -> None:
        registered = chk.parse_registered_archs(chk.MODEL_DIR)
        text = chk.FEATURES.read_text(encoding="utf-8")
        self.assertEqual(chk.supported_models_errors(registered, text), [])

    def test_registry_parse_finds_the_known_archs(self) -> None:
        registered = chk.parse_registered_archs(chk.MODEL_DIR)
        # Representative members: a dense arch, a same-file second registration
        # (Olmo3 shares olmo2_registry.cpp), and a conditional-generation arch.
        for arch in (
            "LlamaForCausalLM",
            "InternLM3ForCausalLM",
            "Olmo3ForCausalLM",
            "Qwen3_5MoeForConditionalGeneration",
            "CohereForCausalLM",
        ):
            with self.subTest(arch=arch):
                self.assertIn(arch, registered)

    def test_registry_parse_ignores_macro_mentions_in_comments(self) -> None:
        # model_registry.cpp names the macro in prose ("via REGISTER_VLLM_MODEL")
        # without the call shape; nothing spurious must enter the set.
        registered = chk.parse_registered_archs(chk.MODEL_DIR)
        self.assertTrue(all(a and '"' not in a for a in registered), registered)
        self.assertTrue(
            all(chk.ARCH_TOKEN_RE.match(f"`{a}`") for a in registered), registered
        )


class DriftTests(unittest.TestCase):
    def test_minimal_matching_page_passes(self) -> None:
        self.assertEqual(chk.supported_models_errors(REGISTERED, _features(TWO)), [])

    def test_registered_but_missing_row_fails(self) -> None:
        text = _features(["`FooForCausalLM`"])  # Bar registered but not listed
        errors = chk.supported_models_errors(REGISTERED, text)
        self.assertTrue(any("no docs/FEATURES.md supported-arch row" in e for e in errors), errors)
        self.assertTrue(any("BarForConditionalGeneration" in e for e in errors), errors)

    def test_listed_but_unregistered_row_fails(self) -> None:
        text = _features(TWO + ["`GhostForCausalLM`"])
        errors = chk.supported_models_errors(REGISTERED, text)
        self.assertTrue(any("does not register" in e for e in errors), errors)
        self.assertTrue(any("GhostForCausalLM" in e for e in errors), errors)

    def test_missing_markers_fails(self) -> None:
        text = _features(TWO, markers=False)
        errors = chk.supported_models_errors(REGISTERED, text)
        self.assertTrue(any("missing the supported-arch-table markers" in e for e in errors), errors)

    def test_empty_registry_fails(self) -> None:
        errors = chk.supported_models_errors(set(), _features(TWO))
        self.assertTrue(any("no REGISTER_VLLM_MODEL registrations" in e for e in errors), errors)

    def test_extractor_suffix_registered_arch_passes_self_check(self) -> None:
        # The GLiNER2 "Extractor" suffix was added to ARCH_TOKEN_RE alongside
        # the BoundaryExtractor registration. A registered *Extractor arch must
        # pass the self-check; remove the suffix and it fails as unrepresentable.
        registered = REGISTERED | {"Gliner2Extractor"}
        text = _features(TWO + ["`Gliner2Extractor`"])
        self.assertEqual(chk.supported_models_errors(registered, text), [])

    def test_forms_suffix_registered_arch_passes_self_check(self) -> None:
        # The cua-s1-forms registration (#3265) is a bare *Forms arch. It landed
        # without widening ARCH_TOKEN_RE, so the shipped self-check dropped it
        # from the comparison and then flagged it missing from FEATURES; the
        # suffix joins Extractor so the arch is compared, never dropped.
        registered = REGISTERED | {"CuaS1Forms"}
        text = _features(TWO + ["`CuaS1Forms`"])
        self.assertEqual(chk.supported_models_errors(registered, text), [])

    def test_versioned_suffix_registered_arch_passes_self_check(self) -> None:
        # Upstream registers the Nemotron Nano VL wrappers under names that end
        # in a version, not a task suffix (`NemotronH_Nano_VL_V2`,
        # `NemotronH_Nano_Omni_Reasoning_V3`, registry.py:512-513). They must
        # be compared, never dropped as unrepresentable.
        registered = REGISTERED | {"NemotronH_Nano_VL_V2", "NemotronH_Nano_Omni_Reasoning_V3"}
        text = _features(TWO + ["`NemotronH_Nano_VL_V2`", "`NemotronH_Nano_Omni_Reasoning_V3`"])
        self.assertEqual(chk.supported_models_errors(registered, text), [])

    def test_unrepresentable_registered_arch_fails_the_self_check(self) -> None:
        # A registered string the FEATURES arch-token pattern cannot express must
        # surface as an error, never be silently excluded from the comparison.
        # (ARCH-ONE-SURFACE ROW 6 widened the pattern to bare `*Model` archs —
        # the upstream _EMBEDDING_MODELS naming, e.g. `LlamaModel` — so the
        # unrepresentable example is now a suffix the pattern still cannot
        # express, not a `*Model` name.)
        registered = REGISTERED | {"WeirdArchitecture"}
        errors = chk.supported_models_errors(registered, _features(TWO))
        self.assertTrue(any("do not match the FEATURES arch-token pattern" in e for e in errors), errors)
        self.assertTrue(any("WeirdArchitecture" in e for e in errors), errors)


class ScopingTests(unittest.TestCase):
    def test_only_column_one_is_scanned(self) -> None:
        # An unregistered arch mentioned in a non-key cell is not a claim.
        text = "\n".join(
            [
                "# Features",
                chk.BEGIN_MARKER,
                "| Architecture | Checkpoint | Gate | Speed |",
                "|---|---|---|---|",
                "| `FooForCausalLM` | ckpt | strict | faster than `GhostForCausalLM` |",
                "| `BarForConditionalGeneration` | ckpt | strict | pending |",
                chk.END_MARKER,
            ]
        )
        self.assertEqual(chk.parse_features_archs(text), REGISTERED)
        self.assertEqual(chk.supported_models_errors(REGISTERED, text), [])

    def test_rows_outside_the_markers_are_ignored(self) -> None:
        # Standalone lanes (Voxtral, MiniMax-H3) and blocked archs live outside
        # the block and must not be read as registry claims.
        text = _features(TWO) + "\n".join(
            [
                "",
                "### Standalone and non-registered lanes",
                "",
                "| Lane | Gate |",
                "|---|---|",
                "| Voxtral (`VoxtralForConditionalGeneration`) | near-tie |",
                "| `DeepseekV3ForCausalLM` | blocked |",
            ]
        )
        self.assertEqual(chk.parse_features_archs(text), REGISTERED)
        self.assertEqual(chk.supported_models_errors(REGISTERED, text), [])


if __name__ == "__main__":
    unittest.main()
