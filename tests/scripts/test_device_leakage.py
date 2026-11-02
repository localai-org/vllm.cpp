#!/usr/bin/env python3
"""Mutation suite for scripts/check-device-leakage.py (the DSR ratchet, work row
`S1` of .agents/specs/accelerator-seam-audit.md).

An unpoliced checker is worse than none: a metric that silently stops catching
leaks converts a real guarantee into a green badge. So every claim the ratchet
makes is asserted here against a MUTANT — a synthetic shared-layer tree with one
deliberate defect planted in it — in the style of cmake/CudaArchFeaturesTest.cmake
(hard expectations, standalone, no CUDA toolkit and no GPU) and
tests/scripts/test_agent_record.py (mutate, then require the specific error).

The four things that must be true, and are each proven by a mutant below:

  1. A device reference ADDED to the shared layer FAILS the check (M1-M6, M18).
  2. A platform-leg site does NOT fail it — adding a backend must never trip the
     ratchet (M7) — but the per-file allowance is an EXACT budget, so it cannot
     be used as a blanket exemption for that file (M8, M9).
  3. A REDUCTION also fails until the baseline is lowered in the same commit,
     which is what makes it a ratchet rather than a threshold (M10-M12).
  4. The escape hatch is real, bounded and LOUD (M13-M15).
  5. The `dev_cast` bucket is derived from the PROPERTY — an integer becoming a
     `vt::DeviceType` outside the platform seam — and not from the one spelling
     that happened to be in the tree when it was written (M20-M46, #660) — and
     that what it actually enforces is a SET OF SPELLINGS rather than that
     property, which is why M46 pins the DECLARED BLIND SPOTS as still blind:
     the checker's message is the authority on what it enforces, so the message
     and the pattern have to be able to disagree loudly. #828 tracks the
     AST-level check that would enforce the property itself.

    python3 tests/scripts/test_device_leakage.py

WHY (5) IS ASSERTED SO HARD. `kcuda` is the token grep `\bkCUDA\b`.
`minimax_h3_video.cpp` wrote `static_cast<vt::DeviceType>(device)` against
`kCUDA = 1` and scored ZERO in that bucket while the TEST that asserted the same
mapping spelled the token honestly and was counted — the gate read the confession
and missed the act (#660). A bucket that closed only `static_cast<vt::DeviceType>`
would be that same defect one spelling later: a regression test wearing a gate's
clothes. So every mutant below is a spelling that is NOT in the tree, and each is
asserted on its own rather than in a batch, because a batch passes when one
alternative in the pattern works.
"""

from __future__ import annotations

import contextlib
import importlib.util
import io
import shutil
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-device-leakage.py"
SPEC = importlib.util.spec_from_file_location("device_leakage", CHECKER)
assert SPEC is not None and SPEC.loader is not None
dl = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = dl
SPEC.loader.exec_module(dl)


# --- a minimal synthetic shared layer ----------------------------------------
# Small enough to reason about exactly, shaped like the real tree: one model file
# (pure shared layer), one platform leg and one registrar (both allowlisted), and
# one device-leg file that must not be scanned at all.
BASE_TREE: dict[str, str] = {
    "src/vllm/model_executor/models/toy.cpp": (
        "#include \"vt/ops.h\"\n"
        "namespace vllm {\n"
        "bool FastPath(Dev d) {\n"
        "  return d.q.device.type == vt::DeviceType::kCUDA;\n"
        "}\n"
        "}  // namespace vllm\n"
    ),
    "src/vllm/platforms/cuda.cpp": (
        "#include <cuda_runtime.h>\n"
        "DeviceType device_type() const override { return DeviceType::kCUDA; }\n"
        "Backend& backend() const { return vt::GetBackend(DeviceType::kCUDA); }\n"
        "void reg() { RegisterPlatform(DeviceType::kCUDA, &platform); }\n"
    ),
    "src/vllm/platforms/platform.cpp": (
        "static const DeviceType kOrder[] = {DeviceType::kCUDA, DeviceType::kCPU};\n"
        # The registry-walk inverse: an array INDEXED by DeviceType turned back
        # into its type. Allowlisted at exactly one `dev_cast`, so the fixture
        # has to carry it or the two-way allowlist ratchet reads the synthetic
        # tree as a platform leg that lost its cast.
        "Platform* Find(std::string_view n) {\n"
        "  const DeviceType type = static_cast<DeviceType>(i);\n"
        "  return nullptr;\n"
        "}\n"
    ),
    "include/vllm/platforms/interface.h": (
        "bool is_cuda() const { return device_type() == DeviceType::kCUDA; }\n"
    ),
    "src/vllm/v1/attention/backend.cpp": (
        "const AttentionBackendRegistrar kA{vt::DeviceType::kCUDA, \"TRITON_MLA\", f};\n"
        "const AttentionBackendRegistrar kB{vt::DeviceType::kCUDA, \"FLASH_ATTN\", g};\n"
    ),
    "src/vllm/v1/attention/backends/gdn_attn.cpp": (
        "const AttentionBackendRegistrar kG{vt::DeviceType::kCUDA, \"GDN_ATTN\", h};\n"
    ),
    "src/vllm/v1/attention/cuda_legacy.cpp": (
        'const AttentionBackendRegistrar kLegacy{vt::DeviceType::kCUDA, "CUDA_LEGACY", f};\n'
    ),
    # A device leg. NOT part of the scanned shared layer: this is exactly where
    # device-specific code is SUPPOSED to live, and counting it would invert the
    # metric's meaning.
    "src/vt/cuda/marlin.cu": (
        "void Marlin() { Use(vt::DeviceType::kCUDA); }\n"
        "#ifdef VT_MARLIN_NVFP4\n#endif\n"
    ),
}

# The synthetic tree's own DSR: one `kCUDA` in toy.cpp; everything else is either
# allowlisted platform leg or outside the scanned roots.
BASE_BUCKETS = {"kcuda": 1, "is_cuda": 0, "dev_cast": 0, "cuda_inc": 0, "vt_ifdef": 0}


class Tree:
    """A synthetic tree plus a baseline file, both disposable."""

    def __init__(self, files: dict[str, str], buckets: dict[str, int] | None = None):
        self.dir = Path(tempfile.mkdtemp(prefix="dsr-mutant-"))
        for rel, text in files.items():
            path = self.dir / rel
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(text, encoding="utf-8")
        self.baseline = self.dir / "baseline.json"
        self._prev_baseline = dl.BASELINE_PATH
        dl.BASELINE_PATH = self.baseline
        if buckets is None:
            buckets = BASE_BUCKETS
        self.set_baseline(buckets)

    def set_baseline(self, buckets: dict[str, int]) -> None:
        import json

        self.baseline.write_text(
            json.dumps({"total": sum(buckets.values()), "buckets": buckets}),
            encoding="utf-8",
        )

    def run(self, *extra: str) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = dl.main(["--root", str(self.dir), *extra])
        return rc, out.getvalue(), err.getvalue()

    def scan(self):
        return dl.scan(self.dir)

    def write(self, rel: str, text: str) -> None:
        path = self.dir / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")

    def append(self, rel: str, text: str) -> None:
        path = self.dir / rel
        path.write_text(path.read_text(encoding="utf-8") + text, encoding="utf-8")

    def close(self) -> None:
        dl.BASELINE_PATH = self._prev_baseline
        shutil.rmtree(self.dir, ignore_errors=True)


class DsrRatchetMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.tree = Tree(dict(BASE_TREE))
        self.addCleanup(self.tree.close)

    def require(self, text: str, needle: str) -> None:
        if needle not in text:
            raise AssertionError(f"missing expected message {needle!r} in:\n{text}")

    # --- sanity: the unmutated tree passes -----------------------------------

    def test_M0_unmutated_tree_passes(self) -> None:
        rc, out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.require(out, "ratchet holds")

    # --- 1. an ADDED device reference must FAIL ------------------------------

    def test_M1_planted_kcuda_in_shared_layer_fails(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool Extra(Dev d) { return d.q.device.type == vt::DeviceType::kCUDA; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'kcuda': 2 > baseline 1")

    def test_M2_planted_is_cuda_call_site_fails(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool E(Dev d) { return GetPlatform(d.q.device.type).is_cuda(); }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'is_cuda': 1 > baseline 0")

    def test_M3_unconditional_cuda_include_fails(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp", '#include "vt/cuda/marlin.h"\n'
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'cuda_inc': 1 > baseline 0")

    def test_M4_guarded_cuda_include_does_not_count(self) -> None:
        # The distinction the metric exists to make: an include under a CUDA/VT_*
        # guard still compiles on a non-CUDA platform, so it is not leakage.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            '#ifdef VLLM_CPP_CUDA\n#include "vt/cuda/marlin.h"\n#endif\n',
        )
        self.assertEqual(self.tree.scan().counts["cuda_inc"], 0)

    def test_M5_cuda_include_in_the_portable_else_arm_counts(self) -> None:
        # The negative arm of a CUDA guard is the PORTABLE build; a CUDA include
        # there breaks it exactly as an unguarded one would.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            '#ifdef VLLM_CPP_CUDA\n#else\n#include "vt/cuda/marlin.h"\n#endif\n',
        )
        self.assertEqual(self.tree.scan().counts["cuda_inc"], 1)

    def test_M6_planted_vt_build_gate_fails(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "#ifdef VT_CUTLASS_NVFP4\nint x;\n#endif\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'vt_ifdef': 1 > baseline 0")

    def test_M18_leak_hidden_behind_a_helper_still_counts(self) -> None:
        # Risk 2 of the audit: the metric must not be gameable by moving the test
        # into a helper — the helper lives in the shared layer and is counted.
        self.tree.write(
            "include/vllm/model_executor/models/toy_helper.h",
            "inline bool OnCuda(Dev d) { return d.q.device.type == vt::DeviceType::kCUDA; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'kcuda': 2 > baseline 1")

    # --- 2. the allowlist exempts the platform leg, EXACTLY ------------------

    def test_M7_adding_a_backend_leg_does_not_trip_the_ratchet(self) -> None:
        # `platforms/cuda.cpp` is allowlisted "*": it IS the CUDA platform. Adding
        # a method to it — or a whole new platform file — must stay green, or the
        # metric would punish the very additivity it exists to protect.
        self.tree.append(
            "src/vllm/platforms/cuda.cpp",
            "bool more() { return t == DeviceType::kCUDA; }\n",
        )
        self.tree.write(
            "src/vllm/platforms/metal.cpp",
            "DeviceType device_type() const { return DeviceType::kMETAL; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 0, err)

    def test_M8_extra_reference_in_a_budgeted_allowlist_file_fails(self) -> None:
        # backend.cpp is allowed EXACTLY its 2 registrar keys. A third `kCUDA` is
        # not a registrar key, and a per-file allowlist that swallowed it would be
        # a blanket exemption — which is how ratchets rot.
        self.tree.append(
            "src/vllm/v1/attention/backend.cpp",
            "bool Hack(Dev d) { return d.type == vt::DeviceType::kCUDA; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "ALLOWLIST STALE: src/vllm/v1/attention/backend.cpp")
        self.require(err, "has 3 reference(s) but the allowlist expects exactly 2")

    def test_M9_removing_an_allowlisted_registrar_fails_until_declared(self) -> None:
        # The allowlist is a ratchet in both directions: it may not stay wider
        # than the platform leg it describes.
        self.tree.write(
            "src/vllm/v1/attention/backend.cpp",
            "const AttentionBackendRegistrar kA{vt::DeviceType::kCUDA, \"TRITON_MLA\", f};\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "ALLOWLIST STALE")
        self.require(err, "has 1 reference(s) but the allowlist expects exactly 2")

    def test_legacy_registrar_extra_device_branch_fails(self) -> None:
        self.tree.append(
            "src/vllm/v1/attention/cuda_legacy.cpp",
            "bool Branch(Dev d) { return d.type == vt::DeviceType::kCUDA; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "ALLOWLIST STALE: src/vllm/v1/attention/cuda_legacy.cpp")
        self.require(err, "has 2 reference(s) but the allowlist expects exactly 1")

    def test_legacy_registrar_missing_key_fails(self) -> None:
        self.tree.write("src/vllm/v1/attention/cuda_legacy.cpp", "// removed registrar\n")
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "ALLOWLIST STALE: src/vllm/v1/attention/cuda_legacy.cpp")
        self.require(err, "has 0 reference(s) but the allowlist expects exactly 1")

    # --- 3. monotonic decrease ------------------------------------------------

    def test_M10_a_reduction_fails_until_the_baseline_is_lowered(self) -> None:
        self.tree.write(
            "src/vllm/model_executor/models/toy.cpp",
            "bool FastPath(Dev d) { return vt::OpRegistered(kOp, d.q.device); }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR baseline STALE in bucket 'kcuda': 0 < baseline 1")
        self.require(err, "must lower the baseline in the SAME")

    def test_M11_lowering_the_baseline_in_the_same_change_passes(self) -> None:
        self.tree.write(
            "src/vllm/model_executor/models/toy.cpp",
            "bool FastPath(Dev d) { return vt::OpRegistered(kOp, d.q.device); }\n",
        )
        rc, _out, err = self.tree.run("--write-baseline")
        self.assertEqual(rc, 0, err)
        rc, out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.require(out, "DSR 0 == baseline 0")

    def test_M12_write_baseline_refuses_to_ratchet_upward(self) -> None:
        # The one move the ratchet must never make, even when asked directly.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool Extra(Dev d) { return d.q.device.type == vt::DeviceType::kCUDA; }\n",
        )
        rc, _out, err = self.tree.run("--write-baseline")
        self.assertEqual(rc, 1)
        self.require(err, "REFUSING to write a HIGHER baseline")

    # --- 4. the escape hatch: real, bounded, and LOUD ------------------------

    def test_M13_dsr_allow_on_the_same_line_exempts_and_is_printed(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool E(Dev d) { return d.t == vt::DeviceType::kCUDA; }"
            "  // DSR-ALLOW(S6): pending the OpRegistered migration\n",
        )
        rc, out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.require(out, "DSR-ALLOW exemptions in force: 1")
        self.require(out, "pending the OpRegistered migration")

    def test_M14_dsr_allow_on_the_preceding_line_exempts(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "// DSR-ALLOW(S3): capability field not yet on Platform\n"
            "bool E(Dev d) { return d.t == vt::DeviceType::kCUDA; }\n",
        )
        rc, out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.require(out, "DSR-ALLOW exemptions in force: 1")

    def test_M15_malformed_dsr_allow_does_not_exempt(self) -> None:
        # No row id and no reason means no traceable owner, so it must not buy an
        # exemption — otherwise the hatch degrades into a magic comment.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "bool E(Dev d) { return d.t == vt::DeviceType::kCUDA; }  // DSR-ALLOW\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'kcuda'")

    # --- composition: prose is not leakage, code is --------------------------

    def test_M16_device_name_in_a_line_comment_is_not_counted(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "// historical note: this used to branch on kCUDA before row M3a\n",
        )
        self.assertEqual(self.tree.scan().counts["kcuda"], 1)

    def test_M17_device_name_in_a_block_comment_or_string_is_not_counted(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "/* kCUDA and is_cuda() described in prose */\n"
            'void C() { VT_CHECK(ok, "this kernel needs kCUDA"); }\n',
        )
        counts = self.tree.scan().counts
        self.assertEqual(counts["kcuda"], 1)
        self.assertEqual(counts["is_cuda"], 0)

    def test_M19_device_legs_under_src_vt_are_not_scanned(self) -> None:
        # src/vt/cuda|metal|vulkan are device legs by definition. If they were
        # scanned, the metric would count the kernels themselves and be useless.
        self.tree.append(
            "src/vt/cuda/marlin.cu",
            "void More() { Use(vt::DeviceType::kCUDA); }\n#ifdef VT_CUTLASS_FP8\n#endif\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 0, err)

    # --- 5. `dev_cast` is a PROPERTY, not a spelling (#660) ------------------
    #
    # The spelling already in the tree when this bucket was written was
    # `static_cast<vt::DeviceType>(device)`. NONE of the mutants below use it.
    # Each is asserted individually: a single test that planted all of them at
    # once would pass while four of the five alternatives were dead.

    def plant(self, body: str) -> tuple[int, str, str]:
        self.tree.append("src/vllm/model_executor/models/toy.cpp", body)
        return self.tree.run()

    def test_M20_c_style_cast_to_devicetype_fails(self) -> None:
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(int32_t d) { return (vt::DeviceType)d; }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M21_functional_cast_to_devicetype_fails(self) -> None:
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(int32_t d) { return vt::DeviceType(d); }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M22_cast_of_an_intermediate_int_variable_fails(self) -> None:
        # The operand is neither a literal nor the parameter, so a pattern
        # anchored on WHAT IS CAST rather than on the TARGET TYPE would miss it.
        # The qualifier is dropped too: `DeviceType`, not `vt::DeviceType`.
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(const Params& p) {\n"
            "  int raw = p.device;\n"
            "  raw = raw ? 1 : 0;\n"
            "  return static_cast<DeviceType>(raw);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M23_brace_initialized_cast_to_devicetype_fails(self) -> None:
        # C++17 permits list-initialising a scoped enum with a fixed underlying
        # type from an integer, so this is a real, compiling laundering route.
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(int32_t d) { return vt::DeviceType{d}; }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M24_cast_split_across_lines_fails(self) -> None:
        # clang-format wraps long casts. A line-oriented matcher reads the two
        # halves as two innocent lines; the bucket matches over the whole
        # comment-stripped text and maps the offset back to a line.
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(int32_t device_selector_from_the_public_abi) {\n"
            "  return static_cast<\n"
            "      vt::DeviceType>(device_selector_from_the_public_abi);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    # --- and the negatives, which are what keep the bucket usable ------------

    def test_M25_casting_a_devicetype_TO_an_integer_does_not_count(self) -> None:
        # The safe direction. `std::to_string(static_cast<int>(type))` is in the
        # platform seam itself; a bucket that counted it would be unlivable.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "int Index(vt::DeviceType t) { return static_cast<int>(t); }\n"
            "size_t Slot(vt::DeviceType t) { return static_cast<size_t>(t); }\n"
            "const char* N(vt::DeviceType t) { return vt::DeviceTypeName(t); }\n"
            "std::vector<vt::DeviceType> All() { return kOrder; }\n",
        )
        self.assertEqual(self.tree.scan().counts["dev_cast"], 0)

    def test_M26_a_devicetype_cast_in_prose_or_a_string_does_not_count(self) -> None:
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "// history: this was (vt::DeviceType)device before the seam landed\n"
            "/* static_cast<vt::DeviceType>(1) is what #660 was about */\n"
            'void D() { VT_CHECK(ok, "never write vt::DeviceType(raw) here"); }\n',
        )
        self.assertEqual(self.tree.scan().counts["dev_cast"], 0)

    def test_M27_the_platform_registry_walk_is_allowlisted_exactly(self) -> None:
        # platform.cpp turns a REGISTRY INDEX back into a DeviceType. That is the
        # seam's own inverse, not a model file hardcoding a device — but it is
        # budgeted at exactly one, so a second cast in that file still fails.
        self.tree.write(
            "src/vllm/platforms/platform.cpp",
            "static const DeviceType kOrder[] = {DeviceType::kCUDA, DeviceType::kCPU};\n"
            "Platform* Find(std::string_view n) {\n"
            "  const DeviceType type = static_cast<DeviceType>(i);\n"
            "  return nullptr;\n"
            "}\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.tree.append(
            "src/vllm/platforms/platform.cpp",
            "DeviceType Sneak(int d) { return (DeviceType)d; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "ALLOWLIST STALE: src/vllm/platforms/platform.cpp")
        self.require(err, "bucket 'dev_cast' has 2 reference(s)")

    def test_M29_a_parameter_declaration_is_not_a_c_style_cast(self) -> None:
        # The false positive this bucket actually produced on the real tree, kept
        # as a mutant so it cannot come back. `(vt::DeviceType /*device*/)` strips
        # to `(vt::DeviceType )` and is then textually a cast applied to `const`.
        # Both discriminators are exercised: the `(` glued to the function name,
        # and the declarator suffix after the `)`.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "struct Sink {\n"
            "  virtual bool supports_on(vt::DeviceType /*device*/) const { return true; }\n"
            "  virtual void note (vt::DeviceType) const noexcept;\n"
            "};\n"
            "using Fn = void(vt::DeviceType);\n",
        )
        self.assertEqual(self.tree.scan().counts["dev_cast"], 0)
        # …and the discriminator must not have cost the real thing. The operand
        # asserted FIRST is a LITERAL, because that is the form the discriminator
        # actually did cost: `(vt::DeviceType)1` — the device named by its raw
        # enum value — is the PUREST case this bucket exists to police, and an
        # identifier-only trailing class let it through while still catching
        # `(vt::DeviceType)d`. A discriminator exercised only on the case it was
        # tuned for is a guard that certifies itself, which is the disease this
        # row is fixing; so both operand kinds are pinned here.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "vt::DeviceType RLit(int d) { return (vt::DeviceType)1; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")
        # A SIGNED literal is the same conversion with one more character.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "vt::DeviceType RNeg(int d) { return (vt::DeviceType)-1; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 2 > baseline 0")
        # And the identifier operand, in the SAME file as the false-positive
        # fixture, so the two discriminators are shown not to have cost it.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "vt::DeviceType RId(int d) { return (vt::DeviceType)d; }\n",
        )
        rc, _out, err = self.tree.run()
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 3 > baseline 0")

    def test_M30_a_global_scope_qualified_target_fails(self) -> None:
        # `::vt::DeviceType` and `vt::DeviceType` name ONE type. A pattern that
        # recognised only the second would be the token grep again, one
        # qualification later.
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(int32_t d) {\n"
            "  return static_cast<::vt::DeviceType>(d);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M31_an_elaborated_type_specifier_target_fails(self) -> None:
        # `enum vt::DeviceType` is the same type spelled the C way. It compiles.
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(int32_t d) {\n"
            "  return static_cast<enum vt::DeviceType>(d);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M32_list_initialising_a_NAMED_declaration_fails(self) -> None:
        # M23 catches the UNNAMED temporary `vt::DeviceType{d}`. Giving the same
        # conversion a name is the spelling a person would actually write, and it
        # is the one M23 alone let through.
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(int32_t d) {\n"
            "  vt::DeviceType dt{d};\n"
            "  return dt;\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M33_pointer_punning_onto_a_devicetype_fails(self) -> None:
        # The ONE form `reinterpret_cast` can legally take: casting to a scoped
        # enum directly is ill-formed, so without the pointer target the three
        # non-`static_cast` keywords in the pattern are decoration — they could
        # only ever match code that does not compile.
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(uint8_t r) {\n"
            "  return *reinterpret_cast<vt::DeviceType*>(&r);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M34_a_function_RETURNING_DeviceType_is_not_a_conversion(self) -> None:
        # The false positive M32's declaration form produces if it accepts `(` as
        # well as `{`. These three shapes are all real sites in this repository
        # (`MiniMaxH3VideoDeviceType`, `ResolveExplicitDeviceType`), and reading a
        # function definition as a cast would red the tree for declaring the very
        # seam this bucket wants people to use.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "vt::DeviceType MiniMaxH3VideoDeviceType(int32_t device);\n"
            "vt::DeviceType ResolveExplicitDeviceType(const Params& p) { return kCpu; }\n"
            "enum vt::DeviceType Named(int d);\n"
            "::vt::DeviceType AlsoNamed(int d) { return kCpu; }\n",
        )
        self.assertEqual(self.tree.scan().counts["dev_cast"], 0)

    # --- M35-M40: the row's own thesis, turned on the instrument a THIRD time --
    #
    # M33 made the pointer target real, and the docstring then claimed the bucket
    # is anchored on "the TARGET TYPE, not on the operand and not on one cast
    # keyword". Four spellings that DO write the target type at the conversion
    # site still scored zero, so the message asserted coverage the pattern did not
    # have — which is #660 exactly, one sigil later. Every mutant below is
    # compile-verified legal C++ (g++ -std=c++20 -Wall -Wextra, exit 0), because a
    # "miss" that does not compile is not a miss.

    def test_M35_a_REFERENCE_target_fails(self) -> None:
        # `reinterpret_cast<vt::DeviceType&>(raw)` is the same pun as M33's
        # `*reinterpret_cast<vt::DeviceType*>(&raw)` — the standard defines the
        # reference form in terms of the pointer form — and it is one character
        # shorter. A `\*?` that admits a star but not an ampersand is a spelling
        # grep wearing the target type's clothes.
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(uint8_t& raw) {\n"
            "  return reinterpret_cast<vt::DeviceType&>(raw);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M36_an_EAST_const_target_fails(self) -> None:
        # `vt::DeviceType const` and `const vt::DeviceType` are ONE type. The
        # pattern took the west spelling only, so the east one laundered the same
        # conversion.
        #
        # This comment used to say "all three places a cv-qualifier can sit" and
        # then list three. There are FOUR: the named cast, the named cast with a
        # reference (the form that compiles WITHOUT -Wignored-qualifiers, so it
        # survives a -Werror build), the C-style cast — and the DECLARATION form,
        # `vt::DeviceType const d{raw}`, which had no cv-group in the pattern at
        # all. An enumeration that certifies its own completeness is the exact
        # defect this row exists to fix (it is M29's disease, one round later), so
        # the count is now the measured one and the fourth case is asserted below.
        # Each is asserted on its own, so one working alternative cannot certify
        # the others.
        rc, _out, err = self.plant(
            "vt::DeviceType Resolve(int32_t d) { return static_cast<vt::DeviceType const>(d); }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")
        rc, _out, err = self.plant(
            "const vt::DeviceType& Ref(const vt::DeviceType& t) {\n"
            "  return static_cast<vt::DeviceType const&>(t);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 2 > baseline 0")
        rc, _out, err = self.plant(
            "vt::DeviceType CStyle(int32_t d) { return (vt::DeviceType const)d; }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 3 > baseline 0")
        # The FOURTH place, and the one the "all three" comment above hid: the
        # DECLARATION form. `vt::DeviceType const d{raw}` and
        # `vt::DeviceType volatile d{raw}` both compile (g++ -std=c++20 -Wall
        # -Wextra, exit 0) and both scored 0, because alternative (3) carried no
        # cv-group. The member-declaration spelling — a named device CONSTANT
        # initialised from the literal `1` — is the one a person actually writes.
        rc, _out, err = self.plant(
            "vt::DeviceType DeclConst(uint8_t raw) {\n"
            "  vt::DeviceType const d{raw};\n"
            "  return d;\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 4 > baseline 0")
        rc, _out, err = self.plant(
            "uint8_t DeclVol(uint8_t raw) {\n"
            "  vt::DeviceType volatile d{raw};\n"
            "  return static_cast<uint8_t>(d);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 5 > baseline 0")
        rc, _out, err = self.plant(
            "struct Cfg { vt::DeviceType const kD{1}; };\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 6 > baseline 0")

    def test_M37_a_UNARY_operand_after_a_c_style_cast_fails(self) -> None:
        # `(vt::DeviceType)*cursor` is the WIRE-FORMAT DECODE the checker's own
        # docstring names as the expected `DSR-ALLOW` case — so the one site the
        # bucket predicted it would meet was the one its trailing class could not
        # see. The class admitted a leading SIGN but not a dereference. `~` and
        # `!` are the same hole. Each asserted individually.
        rc, _out, err = self.plant(
            "vt::DeviceType Decode(const uint8_t* cursor) { return (vt::DeviceType)*cursor; }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")
        rc, _out, err = self.plant(
            "vt::DeviceType Inv(uint8_t m) { return (vt::DeviceType)~m; }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 2 > baseline 0")
        rc, _out, err = self.plant(
            "vt::DeviceType Not(bool f) { return (vt::DeviceType)!f; }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 3 > baseline 0")

    def test_M38_a_POINTER_TO_POINTER_target_fails(self) -> None:
        # `\*?` is ONE star. Two stars is the same pun through one more level of
        # indirection, and it compiles.
        rc, _out, err = self.plant(
            "vt::DeviceType** Table(uint8_t** p) { return reinterpret_cast<vt::DeviceType**>(p); }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M39_bit_cast_to_devicetype_fails(self) -> None:
        # The docstring used to list `std::bit_cast` as unreachable "because no
        # spelling of the target type appears at the conversion site". It does
        # appear — `std::bit_cast<vt::DeviceType>(raw)` writes it in full — so the
        # entry gave a REASON that was false for the case it listed. `bit_cast`
        # is now a cast keyword; `memcpy` and union punning keep the entry, and
        # keep the reason, because for them it is true.
        rc, _out, err = self.plant(
            "vt::DeviceType Pun(uint8_t raw) { return std::bit_cast<vt::DeviceType>(raw); }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M40_the_widening_did_not_swallow_declarators(self) -> None:
        # What M35-M38 could have cost, pinned as a negative. A run of `*`/`&` in
        # a named cast's target and a `*` in the C-style trailing class are both
        # reachable only after a cast token, but a POINTER- or REFERENCE-returning
        # declaration and a ref-qualified member function are the shapes that look
        # closest to them. The last two carry a SPACE before the `(`, which is
        # what defeats the glued-identifier discriminator and leaves only the
        # declarator-suffix guard — the reason `&` is deliberately NOT in the
        # C-style trailing class, and the reason the C-style POINTER pun is
        # declared blind in the checker's docstring instead of being closed here.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "vt::DeviceType* Slot(int i);\n"
            "vt::DeviceType& SlotRef(int i);\n"
            "const vt::DeviceType& Peek();\n"
            "vt::DeviceType** Table();\n"
            "struct Sink {\n"
            "  void note (vt::DeviceType*) &;\n"
            "  void note2 (vt::DeviceType*) &&;\n"
            "  void note3 (vt::DeviceType) volatile;\n"
            "};\n"
            "std::vector<vt::DeviceType*> AllPtrs();\n",
        )
        self.assertEqual(self.tree.scan().counts["dev_cast"], 0)

    # --- M41-M46: round 4. The blind-spot ENTRY bound two members under one
    #
    # reason, and the reason was true of only one of them. The old note covered
    # "a C-STYLE cast whose target is a pointer OR REFERENCE" with the single
    # reason "both are `)` followed by `&`" — true of `*(vt::DeviceType*)&x`,
    # false of `(vt::DeviceType&)raw`, whose next character is an identifier. So
    # the reference form was reachable by widening only INSIDE the parens, at
    # zero cost, while the entry declared it blind. That is the same defect as
    # rounds 1-3, moved from the pattern into the per-entry REASON, which is why
    # every closure below is paired with an assertion that its blind-spot twin is
    # STILL blind (M46) rather than with a wider claim.
    #
    # Every mutant below is compile-verified legal C++ (g++ -std=c++20 -Wall
    # -Wextra -fsyntax-only, exit 0) and measured individually at 0 before / 1
    # after, over the same 760-file scan roots that keep `dev_cast` at its single
    # allowlisted hit.

    def test_M41_a_C_STYLE_pointer_or_reference_target_fails(self) -> None:
        # Widening INSIDE the parens, where no declarator can be confused with a
        # target. `(vt::DeviceType&)raw` is the C-style spelling of M35's pun and
        # `(vt::DeviceType*)vp` is a plain pointer cast off a `void*`; both
        # compile, both scored 0. Asserted separately, because one alternative
        # working is not the other one working.
        rc, _out, err = self.plant(
            "vt::DeviceType Ref(uint8_t& raw) { return (vt::DeviceType&)raw; }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")
        rc, _out, err = self.plant(
            "vt::DeviceType* Ptr(void* vp) { return (vt::DeviceType*)vp; }\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 2 > baseline 0")

    def test_M42_bit_cast_may_name_its_SOURCE_type_too(self) -> None:
        # M39 added `bit_cast` on the argument that it spells the target type at
        # the conversion site. It is a FUNCTION template with TWO parameters, not
        # a cast operator with one, so `From` may be written — and terminating the
        # target at `>` assumed the one-argument shape the four real casts have.
        # `std::bit_cast<vt::DeviceType, std::uint8_t>(raw)` compiles and scored
        # 0. Over-matching is not a risk: the scanned roots contain ZERO
        # `bit_cast` and ZERO `__builtin_bit_cast` occurrences — measured over the
        # comment-stripped text of all 760 files, with `static_cast` = 9950 as the
        # positive control in the same pass, so the zero is an absence and not a
        # broken grep.
        rc, _out, err = self.plant(
            "vt::DeviceType Pun(uint8_t raw) {\n"
            "  return std::bit_cast<vt::DeviceType, std::uint8_t>(raw);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M43_a_cv_qualifier_INSIDE_the_pointer_run_fails(self) -> None:
        # M35 made the target suffix a run of `[*&]`, and the docstring then
        # claimed a REFERENCE target. `vt::DeviceType* const&` is exactly that —
        # a reference to a const pointer — and the run stopped at the first
        # cv-qualifier. One position to the right of round 2's own finding.
        rc, _out, err = self.plant(
            "vt::DeviceType* const& Slot(vt::DeviceType** p) {\n"
            "  return reinterpret_cast<vt::DeviceType* const&>(*p);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M44_the_BUILTIN_spelling_of_bit_cast_fails(self) -> None:
        # `__builtin_bit_cast(vt::DeviceType, raw)` compiles under g++ 13.3 and
        # names the target type at the conversion site just as fully as
        # `std::bit_cast` does — but it is not a template-id, so alternative (1)'s
        # `<…>` anchor cannot reach it and alternative (2)'s glued-identifier
        # discriminator rejects it. Listing `bit_cast` as caught while this
        # spelling scored 0 would be the same overclaim in a smaller font.
        rc, _out, err = self.plant(
            "vt::DeviceType Pun(uint8_t raw) {\n"
            "  return __builtin_bit_cast(vt::DeviceType, raw);\n"
            "}\n"
        )
        self.assertEqual(rc, 1)
        self.require(err, "DSR REGRESSION in bucket 'dev_cast': 1 > baseline 0")

    def test_M45_sizeof_and_alignof_convert_nothing(self) -> None:
        # A false positive the widening WOULD have grown. `sizeof (vt::DeviceType)
        # + 1` already scored 1 before this round — `sizeof` is not glued to its
        # paren, so the identifier discriminator passes, and `+` is in the
        # trailing class — and admitting `*` inside the parens would have extended
        # it to the pointer spelling. `sizeof` and `alignof` convert nothing, so
        # both are excluded outright rather than left as a gap made wider. The
        # first two of these are RED against the SHIPPED pattern.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "size_t A() { return sizeof (vt::DeviceType) + 1; }\n"
            "size_t B() { return alignof (vt::DeviceType) + 1; }\n"
            "size_t C() { return sizeof (vt::DeviceType*) + 1; }\n"
            "size_t D() { return sizeof(vt::DeviceType) + 1; }\n",
        )
        self.assertEqual(self.tree.scan().counts["dev_cast"], 0)

    def test_M46_the_declared_blind_spots_are_STILL_blind(self) -> None:
        # The other half of a per-entry reason: the docstring says these three
        # score 0, so this pins that they do. If a later widening closes one, this
        # test goes RED and the message has to be corrected in the same change —
        # which is the only mechanism that keeps a checker's message the authority
        # on what it enforces. It is not an argument that they SHOULD stay blind.
        #
        #   * `*(vt::DeviceType*)&x` — closing it needs `&` in the TRAILING class,
        #     which is what makes `void note (vt::DeviceType*) &;` a false
        #     positive (M40 pins that negative);
        #   * `(vt::DeviceType)'\x01'` — character literals are blanked to
        #     whitespace by strip_comments_and_strings before the pattern runs, so
        #     the operand is gone by match time;
        #   * `std::memcpy(&dt, &raw, sizeof(vt::DeviceType))` — there is no cast
        #     EXPRESSION to anchor on. Note it DOES name the type at the site, so
        #     the old shared reason ("the type is named at the declaration, never
        #     at the conversion") was false for it.
        #
        # All three compile (g++ -std=c++20 -Wall -Wextra -fsyntax-only, exit 0).
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "vt::DeviceType Pun(uint8_t x) { return *(vt::DeviceType*)&x; }\n"
            "vt::DeviceType Chr() { return (vt::DeviceType)'\\x01'; }\n"
            "vt::DeviceType Mem(uint8_t raw) {\n"
            "  vt::DeviceType dt{};\n"
            "  std::memcpy(&dt, &raw, sizeof(vt::DeviceType));\n"
            "  return dt;\n"
            "}\n",
        )
        self.assertEqual(self.tree.scan().counts["dev_cast"], 0)
        # The positive control for that zero, in the same test: the pattern is
        # alive, and one character more of operand IS caught. A null result from a
        # dead regex would otherwise prove nothing.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "vt::DeviceType Buf(const uint8_t* b) { return (vt::DeviceType)b[0]; }\n",
        )
        self.assertEqual(self.tree.scan().counts["dev_cast"], 1)

    def test_M47_the_declared_OVER_MATCH_still_fires(self) -> None:
        # M46's shape, applied to a FALSE POSITIVE instead of a blind spot.
        #
        # Alternative (3) matches `vt::DeviceType d{x}` on the target type alone,
        # so a plain COPY-INITIALISATION whose operand is already a DeviceType
        # fires while converting nothing. The docstring's "WHAT `dev_cast`
        # OVER-MATCHES" section states that, because a reader who takes a RED as
        # proof of leakage is misled by a message that omits it.
        #
        # It is not narrowed, and the reason is that `vt::DeviceType d{raw}` —
        # the declaration spelling of the real conversion, which M32 and M36 pin
        # — is textually identical. Narrowing to remove the false positive
        # deletes the true positive with it. So this test pins the CURRENT
        # behaviour: if a later change makes any of these stop firing, it goes
        # RED and the docstring entry must be corrected in the same change.
        #
        # All five compile (g++ 13.3 -std=c++20 -Wall -Wextra -fsyntax-only,
        # exit 0), and each is asserted on its OWN file rather than appended to a
        # shared one, so a spelling that stopped firing cannot be hidden by the
        # next one still firing.
        probe = "src/vllm/model_executor/models/probe.cpp"
        for source in (
            "void L(vt::DeviceType other) { vt::DeviceType d{other}; (void)d; }\n",
            "struct S { vt::DeviceType d{vt::DeviceType::kCPU}; };\n",
            "void F(const Plat& p) { vt::DeviceType d{p.device_type()}; (void)d; }\n",
            "struct T { vt::DeviceType const d{vt::DeviceType::kCPU}; };\n",
        ):
            with self.subTest(source=source.strip()):
                self.tree.write(probe, source)
                self.assertEqual(self.tree.scan().counts["dev_cast"], 1)
        # The NEGATIVE control in the same test, and the boundary of the entry: a
        # value-initialisation converts nothing AND does not fire, because the
        # empty-braces lookahead already rejects it. Without this line the four
        # assertions above would be consistent with "alternative (3) matches every
        # `vt::DeviceType` declaration", which is a different and larger claim.
        self.tree.write(probe, "void V() { vt::DeviceType d{}; (void)d; }\n")
        self.assertEqual(self.tree.scan().counts["dev_cast"], 0)

    def test_M28_dsr_allow_exempts_a_dev_cast_and_says_so_loudly(self) -> None:
        # The legitimate case the risk register names: a deserialization boundary
        # that reads a device off the wire. It buys an exemption only with a row
        # id and a reason, and it is printed on every run.
        self.tree.append(
            "src/vllm/model_executor/models/toy.cpp",
            "// DSR-ALLOW(S9): wire-format decode, validated against kNumDeviceTypes\n"
            "vt::DeviceType Decode(uint8_t b) { return (vt::DeviceType)b; }\n",
        )
        rc, out, err = self.tree.run()
        self.assertEqual(rc, 0, err)
        self.require(out, "DSR-ALLOW exemptions in force: 1")
        self.require(out, "[dev_cast]")


class RealTreeTests(unittest.TestCase):
    """Hard expectations about THIS repository, not a synthetic mutant."""

    def test_laguna_marlin_gate_helper_is_feature_guarded(self) -> None:
        path = ROOT / "src/vllm/model_executor/models/laguna.cpp"
        lines = path.read_text(encoding="utf-8").splitlines()
        definitions = [
            index
            for index, line in enumerate(lines)
            if "inline bool LagunaMarlinMoeEnabled()" in line
        ]
        self.assertEqual(len(definitions), 1)
        self.assertTrue(
            dl.cuda_guard_depth(lines)[definitions[0]],
            "LagunaMarlinMoeEnabled must not compile outside VT_MARLIN_NVFP4",
        )

    def test_baseline_matches_the_tree_exactly(self) -> None:
        # This is the CI gate itself, asserted here too so a local run catches it
        # before the push.
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            rc = dl.main([])
        self.assertEqual(rc, 0, out.getvalue() + err.getvalue())

    def test_server_main_dsr_allows_cover_only_the_profiler_guard(self) -> None:
        # #189 moved the server body into the shared layer, bringing 5
        # `#ifdef VT_BENCH_PROFILE_CONTROL` sites with it and pushing vt_ifdef
        # 32 -> 37. They are exempted per-site with DSR-ALLOW rather than by a
        # per-file budget, because a budget can be spent on something else: swap
        # one guard for a real device fork and the count still reads 5. This pins
        # that every exempted guard in that TU is the profiler one.
        path = ROOT / "src/vllm/entrypoints/openai/server_main.cpp"
        lines = path.read_text(encoding="utf-8").splitlines()
        guarded = [
            index
            for index, line in enumerate(lines)
            if line.lstrip().startswith("#ifdef VT_")
            or line.lstrip().startswith("#if defined(VT_")
        ]
        self.assertTrue(guarded)
        for index in guarded:
            self.assertIn("VT_BENCH_PROFILE_CONTROL", lines[index])
            self.assertTrue(
                dl.RE_DSR_ALLOW.search(lines[index])
                or (index and dl.RE_DSR_ALLOW.search(lines[index - 1])),
                f"line {index + 1} has no DSR-ALLOW on it or directly above it",
            )

    def test_the_diffusion_lanes_resolve_device_through_the_seam(self) -> None:
        # #660's actual site, pinned as a fact about THIS tree rather than as a
        # number in a baseline. The positive control is in the same assertion:
        # the pattern must still fire on the defect's own text, or a null result
        # below would prove only that the regex is broken.
        defect = "return static_cast<vt::DeviceType>(device);"
        self.assertTrue(
            dl.RE_DEVTYPE_CAST.search(defect),
            "positive control failed: the dev_cast pattern no longer matches the "
            "very line #660 was filed about, so its absence below proves nothing",
        )
        for rel in (
            "src/vllm/multimodal/minimax_h3_video.cpp",
            "src/vllm/multimodal/ltx2_video.cpp",
        ):
            code = "\n".join(
                dl.strip_comments_and_strings((ROOT / rel).read_text(encoding="utf-8"))
            )
            found = [m.group(0) for m in dl.RE_DEVTYPE_CAST.finditer(code)]
            self.assertEqual(found, [], f"{rel} casts an integer into a DeviceType")

    def test_every_allowlisted_path_exists(self) -> None:
        # A stale allowlist entry is a silent exemption for a file that may later
        # be recreated with different contents.
        for rel in dl.ALLOWLIST:
            self.assertTrue((ROOT / rel).is_file(), f"allowlisted path missing: {rel}")

    def test_allowlist_entries_all_state_a_reason(self) -> None:
        for rel, buckets in dl.ALLOWLIST.items():
            for bucket, (_count, reason) in buckets.items():
                self.assertIn(bucket, dl.BUCKETS, f"{rel}: unknown bucket {bucket}")
                self.assertGreater(
                    len(reason), 40, f"{rel}[{bucket}] allowlist reason is not a reason"
                )

    def test_deepseek_v4_device_allowlist_pins_mhc_rocm_fallback(self) -> None:
        # O34 (#3198) added one OpRegistered(kDeepseekV4Mhc, kCUDA) probe to
        # MhcDevice() so it can try kCUDA and fall back to kROCm. The allowlist
        # for this file must read exactly 9 — 8 hides the ROCm probe and the
        # checker reports the real tree as non-compliant; 10 hides a future leak.
        entry = dl.ALLOWLIST.get(
            "src/vllm/model_executor/models/deepseek_v4_device.cpp", {}
        )
        self.assertIn("kcuda", entry, "deepseek_v4_device.cpp must be allowlisted")
        count, reason = entry["kcuda"]
        self.assertEqual(count, 9, "deepseek_v4_device.cpp kcuda budget must be 9")
        self.assertIn("O34", reason, "the O34 ROCm fallback must be the stated reason")

    def test_leakage_is_concentrated_in_one_model_file(self) -> None:
        # The audit's headline finding, asserted as an executable fact rather than
        # a claim in a document: the shared layer's device branching is not spread
        # thin, it is piled into the model TU that has no `layers/` library to put
        # it in. If this ever stops being true, the S4/S7 plan is mis-aimed.
        res = dl.scan(ROOT)
        hot = "src/vllm/model_executor/models/qwen3_5.cpp"
        self.assertIn(hot, res.per_file)
        self.assertGreater(sum(res.per_file[hot].values()) * 2, res.total)


if __name__ == "__main__":
    unittest.main()
