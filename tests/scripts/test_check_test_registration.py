#!/usr/bin/env python3
"""Mutation checks for the required CMake/CTest registration guard."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import sys
import tempfile
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-test-registration.py"
SPEC = importlib.util.spec_from_file_location("check_test_registration", CHECKER)
assert SPEC is not None and SPEC.loader is not None
mod = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = mod
SPEC.loader.exec_module(mod)


PASSING_CMAKE = """
cmake_minimum_required(VERSION 3.20)
project(registration_guard LANGUAGES CXX)
enable_testing()
add_library(vllm_core INTERFACE)
add_library(vllm::vllm ALIAS vllm_core)
add_library(vllm_test_main INTERFACE)

function(vllm_cpp_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE vllm::vllm vllm_test_main)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

vllm_cpp_add_test(test_device_selection
  vllm/entrypoints/test_device_selection.cpp)
"""

PASSING_CI = """jobs:
  checks:
    steps:
      - name: Critical regression tests remain executable and CTest-registered
        run: |
          python3 scripts/check-test-registration.py
          python3 tests/scripts/test_check_test_registration.py
      - name: The SGLang lease identity gate detects a tree that is not the pin
        run: |
          python3 tests/scripts/test_sglang_lease_identity.py
      - name: The committed LTX-2 oracle goldens still hash to what records them
        run: |
          python3 tests/scripts/test_ltx2_oracle_goldens.py
      - name: The layer-fingerprint comparator parses every tap the instrument prints
        run: |
          python3 tests/scripts/test_q4exp_layerfp_diff.py
"""

PASSING_PREFLIGHT = """CHECKERS=(
  check-test-registration
)
SUITES=(
  test_check_test_registration
  test_sglang_lease_identity
  test_ltx2_oracle_goldens
  test_q4exp_layerfp_diff
)
for checker in "${CHECKERS[@]}"; do
  python3 "scripts/$checker.py"
done
for suite in "${SUITES[@]}"; do
  python3 "tests/scripts/$suite.py"
done
"""


# The labelled-gate fixture.  Three registered tests, TWO of them labelled -- the
# production selection since #1839 -- so a mutation that widens the selection has
# somewhere to widen into, and a mutation that drops ONE of the two has somewhere
# to drop from.  The second case only exists at n > 1 and is the one a floor
# cannot see: a gate quietly leaving a lane whose whole purpose is that a human
# runs it deliberately inside a lease.
PASSING_LABEL_CMAKE = """
cmake_minimum_required(VERSION 3.20)
project(label_guard LANGUAGES CXX)
enable_testing()
add_library(vllm_core INTERFACE)
add_library(vllm::vllm ALIAS vllm_core)
add_library(vllm_test_main INTERFACE)

function(vllm_cpp_add_test name)
  add_executable(${name} ${ARGN})
  target_link_libraries(${name} PRIVATE vllm::vllm vllm_test_main)
  add_test(NAME ${name} COMMAND ${name})
endfunction()

vllm_cpp_add_test(test_device_selection
  vllm/entrypoints/test_device_selection.cpp)
vllm_cpp_add_test(test_minimax_music3_device_arm_real
  parity/test_minimax_music3_device_arm_real.cpp)
set_tests_properties(test_minimax_music3_device_arm_real PROPERTIES
  LABELS "gpu;checkpoint;music3")
vllm_cpp_add_test(test_minimax_music3_depth_arm_real
  parity/test_minimax_music3_depth_arm_real.cpp)
set_tests_properties(test_minimax_music3_depth_arm_real PROPERTIES
  LABELS "gpu;checkpoint;music3")
"""


def _suite_integrity_errors(source: str) -> list[str]:
    """Exercise the production-owned, canonical mutation-suite contract."""

    return mod.mutation_suite_integrity_errors(
        source, manifest_path=mod.MUTATION_MANIFEST
    )


class RegistrationMutationTests(unittest.TestCase):
    def assert_error(self, text: str, needle: str) -> None:
        errors = mod.registration_errors(text)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_minimal_complete_registration_passes(self) -> None:
        self.assertEqual(mod.registration_errors(PASSING_CMAKE), [])

    def test_M1_deleting_test_invocation_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)\n",
            "",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M2_changing_test_source_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm/entrypoints/test_device_selection.cpp", "vllm/entrypoints/other.cpp"
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "must compile vllm/entrypoints/test_device_selection.cpp")

    def test_M3_deleting_add_executable_from_helper_fails(self) -> None:
        mutated = PASSING_CMAKE.replace("  add_executable(${name} ${ARGN})\n", "")
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "does not create an executable")

    def test_M4_deleting_add_test_from_helper_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})\n", ""
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "does not register that executable with CTest")

    def test_M5_duplicate_target_fails(self) -> None:
        mutated = PASSING_CMAKE + (
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)\n"
        )
        self.assert_error(mutated, "registered 2 times")

    def test_M6_commented_out_invocation_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)",
            "# vllm_cpp_add_test(test_device_selection "
            "vllm/entrypoints/test_device_selection.cpp)",
        )
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M7_commented_out_add_executable_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_executable(${name} ${ARGN})", "  # add_executable(${name} ${ARGN})"
        )
        self.assert_error(mutated, "does not create an executable")

    def test_M8_commented_out_add_test_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  # add_test(NAME ${name} COMMAND ${name})",
        )
        self.assert_error(mutated, "does not register that executable with CTest")

    def test_M13_bracket_commented_target_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)",
            "#[[\nvllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)\n]]",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M14_target_in_false_conditional_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)",
            "if(FALSE)\n"
            "  vllm_cpp_add_test(test_device_selection\n"
            "    vllm/entrypoints/test_device_selection.cpp)\n"
            "endif()",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M15_quoted_target_text_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "vllm_cpp_add_test(test_device_selection\n"
            "  vllm/entrypoints/test_device_selection.cpp)",
            'set(dead "vllm_cpp_add_test(test_device_selection '
            'vllm/entrypoints/test_device_selection.cpp)")',
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M16_helper_registration_in_false_conditional_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  if(FALSE)\n"
            "    add_test(NAME ${name} COMMAND ${name})\n"
            "  endif()",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "is not registered with CTest")

    def test_M17_helper_executable_in_false_conditional_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_executable(${name} ${ARGN})",
            "  if(FALSE)\n"
            "    add_executable(${name} ${ARGN})\n"
            "  endif()",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "missing required test target test_device_selection")

    def test_M20_ctest_noop_command_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  add_test(NAME ${name} COMMAND ${CMAKE_COMMAND} -E true)",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(
            mutated, "must execute configured target test_device_selection exactly"
        )

    def test_M21_ctest_wrong_binary_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  add_test(NAME ${name} COMMAND other)",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(
            mutated, "must execute configured target test_device_selection exactly"
        )

    def test_M22_ctest_extra_argument_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  add_test(NAME ${name} COMMAND ${name} --list-test-cases)",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(
            mutated, "must execute configured target test_device_selection exactly"
        )

    def test_M28_disabled_ctest_fails(self) -> None:
        mutated = PASSING_CMAKE.replace(
            "  add_test(NAME ${name} COMMAND ${name})",
            "  add_test(NAME ${name} COMMAND ${name})\n"
            "  set_tests_properties(${name} PROPERTIES DISABLED TRUE)",
        )
        self.assertNotEqual(mutated, PASSING_CMAKE)
        self.assert_error(mutated, "must not be DISABLED")

    def test_ninja_multi_config_resolves_release_ctest_command(self) -> None:
        old_generator = os.environ.get("CMAKE_GENERATOR")
        os.environ["CMAKE_GENERATOR"] = "Ninja Multi-Config"
        try:
            self.assertEqual(mod.registration_errors(PASSING_CMAKE), [])
        finally:
            if old_generator is None:
                os.environ.pop("CMAKE_GENERATOR", None)
            else:
                os.environ["CMAKE_GENERATOR"] = old_generator


class WiringMutationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.preflight = PASSING_PREFLIGHT
        self.ci = PASSING_CI

    def assert_wiring_error(self, preflight: str, ci: str, needle: str) -> None:
        errors = mod.wiring_errors(preflight, ci)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_complete_preflight_and_ci_wiring_passes(self) -> None:
        self.assertEqual(mod.wiring_errors(self.preflight, self.ci), [])

    def test_M9_deleting_preflight_checker_fails(self) -> None:
        mutated = self.preflight.replace("  check-test-registration\n", "")
        self.assert_wiring_error(mutated, self.ci, "preflight CHECKERS")

    def test_M10_deleting_preflight_suite_fails(self) -> None:
        mutated = self.preflight.replace("  test_check_test_registration\n", "")
        self.assert_wiring_error(mutated, self.ci, "preflight SUITES")

    def test_M11_deleting_ci_checker_fails(self) -> None:
        mutated = self.ci.replace(
            "          python3 scripts/check-test-registration.py\n", ""
        )
        self.assert_wiring_error(self.preflight, mutated, "CI checker")

    def test_M12_deleting_ci_suite_fails(self) -> None:
        mutated = self.ci.replace(
            "          python3 tests/scripts/test_check_test_registration.py\n", ""
        )
        self.assert_wiring_error(self.preflight, mutated, "CI mutation suite")

    # The pinned-suite lane (#1833).  Before it, deleting either registration of
    # `test_sglang_lease_identity` left `check-test-registration.py` at rc=0 --
    # and so did deleting an unrelated entry from the same array, which is what
    # makes the finding a property of `SUITES` rather than of that one suite.
    def test_M56_deleting_the_pinned_preflight_suite_fails(self) -> None:
        mutated = self.preflight.replace("  test_sglang_lease_identity\n", "")
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(
            mutated, self.ci, "test_sglang_lease_identity is missing from preflight SUITES"
        )

    def test_M57_deleting_the_pinned_ci_suite_fails(self) -> None:
        mutated = self.ci.replace(
            "          python3 tests/scripts/test_sglang_lease_identity.py\n", ""
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(
            self.preflight,
            mutated,
            "test_sglang_lease_identity is missing from the CI suite lane",
        )

    # The SECOND pinned suite (#2553).  These two cases are what makes widening
    # `REQUIRED_SUITE_REGISTRATIONS` a semantic change rather than a fixture
    # edit.  Adding the suite to `PASSING_CI` and `PASSING_PREFLIGHT` alone is
    # symmetric and additive, so the base checker -- which pins only
    # `test_sglang_lease_identity` -- stays green on it and the widening carries
    # no evidence at all; `check-pr-size.py` rejected exactly that, because a
    # test whose only job is to KEEP an existing case green cannot fail for the
    # intended reason.  Deleting one lane's registration must name THIS suite.
    #
    # They assert the checker's output on a mutated input, never
    # `"test_ltx2_oracle_goldens" in mod.REQUIRED_SUITE_REGISTRATIONS`, which
    # would be a tautology over the constant it is supposed to hold.  And they
    # are deliberately not named `test_M*`: `mutation_suite_integrity_errors`
    # requires the `test_M*` set to equal the digest-pinned manifest exactly, so
    # an M-number here would drag `check_test_registration_mutations.txt` and
    # `MUTATION_MANIFEST_SHA256` into a change that is not about the manifest.
    def test_pinned_ltx2_goldens_suite_absent_from_ci_is_named(self) -> None:
        mutated = self.ci.replace(
            "          python3 tests/scripts/test_ltx2_oracle_goldens.py\n", ""
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(
            self.preflight,
            mutated,
            "test_ltx2_oracle_goldens is missing from the CI suite lane",
        )

    def test_pinned_ltx2_goldens_suite_absent_from_preflight_is_named(self) -> None:
        mutated = self.preflight.replace("  test_ltx2_oracle_goldens\n", "")
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(
            mutated,
            self.ci,
            "test_ltx2_oracle_goldens is missing from preflight SUITES",
        )

    # The THIRD pinned suite (#2877), on the pattern the two cases above set.
    # Same reasoning: adding it to `PASSING_CI` and `PASSING_PREFLIGHT` alone is
    # symmetric and additive, so the base checker stays green either way and the
    # widening would carry no evidence. Deleting one lane's registration must
    # name THIS suite. Not `test_M*`, for the manifest-digest reason above.
    def test_pinned_layerfp_diff_suite_absent_from_ci_is_named(self) -> None:
        mutated = self.ci.replace(
            "          python3 tests/scripts/test_q4exp_layerfp_diff.py\n", ""
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(
            self.preflight,
            mutated,
            "test_q4exp_layerfp_diff is missing from the CI suite lane",
        )

    def test_pinned_layerfp_diff_suite_absent_from_preflight_is_named(self) -> None:
        mutated = self.preflight.replace("  test_q4exp_layerfp_diff\n", "")
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(
            mutated,
            self.ci,
            "test_q4exp_layerfp_diff is missing from preflight SUITES",
        )

    def test_M18_ci_commands_behind_false_shell_branch_fail(self) -> None:
        mutated = self.ci.replace(
            "          python3 scripts/check-test-registration.py\n"
            "          python3 tests/scripts/test_check_test_registration.py\n",
            "          if false; then\n"
            "            python3 scripts/check-test-registration.py\n"
            "            python3 tests/scripts/test_check_test_registration.py\n"
            "          fi\n",
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M19_ci_commands_as_quoted_text_fail(self) -> None:
        mutated = self.ci.replace(
            "          python3 scripts/check-test-registration.py\n",
            '          echo "python3 scripts/check-test-registration.py"\n',
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M23_ci_step_with_false_condition_fails(self) -> None:
        mutated = self.ci.replace(
            "      - name: Critical regression tests remain executable and CTest-registered\n"
            "        run: |\n",
            "      - name: Critical regression tests remain executable and CTest-registered\n"
            "        if: ${{ false }}\n"
            "        run: |\n",
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M24_ci_job_with_false_condition_fails(self) -> None:
        mutated = self.ci.replace(
            "  checks:\n",
            "  checks:\n"
            "    if: ${{ false }}\n",
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M25_preflight_names_in_inert_arrays_fail(self) -> None:
        mutated = self.preflight.replace(
            "  check-test-registration\n", ""
        ).replace(
            "  test_check_test_registration\n", ""
        ) + (
            "INERT=(\n"
            "  check-test-registration\n"
            "  test_check_test_registration\n"
            ")\n"
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "preflight CHECKERS")
        self.assert_wiring_error(mutated, self.ci, "preflight SUITES")

    def test_M26_preflight_checker_loop_rebound_fails(self) -> None:
        mutated = self.preflight.replace(
            'for checker in "${CHECKERS[@]}"; do',
            'for checker in "${INERT[@]}"; do',
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "execute CHECKERS")

    def test_M27_preflight_suite_loop_rebound_fails(self) -> None:
        mutated = self.preflight.replace(
            'for suite in "${SUITES[@]}"; do',
            'for suite in "${INERT[@]}"; do',
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "execute SUITES")

    def test_M29_ci_continue_on_error_fails(self) -> None:
        mutated = self.ci.replace(
            "        run: |\n", "        continue-on-error: true\n        run: |\n"
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M30_ci_quoted_false_condition_fails(self) -> None:
        mutated = self.ci.replace(
            "        run: |\n", "        'if': ${{ false }}\n        run: |\n"
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M31_ci_spaced_false_condition_fails(self) -> None:
        mutated = self.ci.replace(
            "        run: |\n", "        if : ${{ false }}\n        run: |\n"
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M32_ci_inert_shell_fails(self) -> None:
        mutated = self.ci.replace(
            "        run: |\n", "        shell: /bin/true {0}\n        run: |\n"
        )
        self.assertNotEqual(mutated, self.ci)
        self.assert_wiring_error(self.preflight, mutated, "direct active commands")

    def test_M33_preflight_checker_loop_noop_body_fails(self) -> None:
        mutated = self.preflight.replace(
            '  python3 "scripts/$checker.py"', "  true"
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "execute CHECKERS")

    def test_M34_preflight_suite_loop_noop_body_fails(self) -> None:
        mutated = self.preflight.replace(
            '  python3 "tests/scripts/$suite.py"', "  true"
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "execute SUITES")

    def test_M35_preflight_checker_loop_immediate_continue_fails(self) -> None:
        mutated = self.preflight.replace(
            'for checker in "${CHECKERS[@]}"; do\n',
            'for checker in "${CHECKERS[@]}"; do\n  continue\n',
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "execute CHECKERS")

    def test_M36_preflight_loops_behind_false_outer_branch_fail(self) -> None:
        loop_block = (
            'for checker in "${CHECKERS[@]}"; do\n'
            '  python3 "scripts/$checker.py"\n'
            "done\n"
            'for suite in "${SUITES[@]}"; do\n'
            '  python3 "tests/scripts/$suite.py"\n'
            "done\n"
        )
        mutated = self.preflight.replace(loop_block, "if false; then\n" + loop_block + "fi\n")
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "execute CHECKERS")
        self.assert_wiring_error(mutated, self.ci, "execute SUITES")

    def test_M37_preflight_unsets_checkers_before_loop_fails(self) -> None:
        mutated = self.preflight.replace(
            'for checker in "${CHECKERS[@]}"; do',
            'unset CHECKERS\nfor checker in "${CHECKERS[@]}"; do',
        )
        self.assertNotEqual(mutated, self.preflight)
        self.assert_wiring_error(mutated, self.ci, "execute CHECKERS")


class HostPortabilityTests(unittest.TestCase):
    """Cases a host, not the tree, used to decide.

    Every one of these was a RED on a Windows host while the same tree was
    green on a POSIX one, and every one of them was a message naming the tree
    rather than the host that could not answer.  None of them is named `test_M*`:
    that inventory is the MUTATION set, pinned by digest in the checker, and a
    host-portability regression is not a mutation of the contract.
    """

    # --- _target_artifact: the MSVC debug-symbol side artifact -------------

    @staticmethod
    def detail(*artifacts: str, kind: str = "EXECUTABLE") -> dict:
        return {"type": kind, "artifacts": [{"path": path} for path in artifacts]}

    def test_debug_symbol_artifact_does_not_hide_the_executable(self) -> None:
        """Multi-config generators list the .pdb BESIDE the .exe.

        Measured on MSVC 19.44 / CMake 4.4.2 against this tree: the codemodel
        describes `test_device_selection` as an EXECUTABLE with two artifacts,
        `tests/Release/test_device_selection.exe` and
        `.../test_device_selection.pdb`, and a rule that wanted the artifact
        LIST to hold exactly one entry answered 'has no single configured
        executable artifact' about a target that has one.
        """

        build_dir = Path("C:/build")
        resolved = mod._target_artifact(
            build_dir,
            self.detail(
                "tests/Release/test_device_selection.exe",
                "tests/Release/test_device_selection.pdb",
            ),
        )
        self.assertEqual(
            resolved,
            (build_dir / "tests/Release/test_device_selection.exe").resolve(),
        )

    def test_dsym_bundle_does_not_hide_the_executable(self) -> None:
        """The same shape from Xcode, and the case the suffix list lowercases."""

        build_dir = Path("/build")
        self.assertEqual(
            mod._target_artifact(
                build_dir,
                self.detail("Release/test", "Release/test.dSYM"),
            ),
            (build_dir / "Release/test").resolve(),
        )

    def test_two_executable_artifacts_are_still_refused(self) -> None:
        """The control: filtering symbols must not admit a second binary.

        If this ever passes with a non-None result, the filter above has become
        a hole and the gate stopped being able to see an ambiguous target.
        """

        self.assertIsNone(
            mod._target_artifact(
                Path("C:/build"),
                self.detail("tests/Debug/test.exe", "tests/Release/test.exe"),
            )
        )

    def test_single_artifact_target_is_unchanged(self) -> None:
        """The ordinary single-config shape still resolves, unchanged."""

        self.assertEqual(
            mod._target_artifact(Path("C:/build"), self.detail("tests/test")),
            (Path("C:/build") / "tests/test").resolve(),
        )

    def test_non_executable_target_has_no_artifact(self) -> None:
        """A STATIC_LIBRARY with a lone .a is still not an executable."""

        self.assertIsNone(
            mod._target_artifact(Path("C:/build"), self.detail("libcore.a", kind="STATIC_LIBRARY"))
        )

    # --- _include_spelling: the separator an #include never uses ------------

    def test_include_spelling_normalizes_the_windows_separator(self) -> None:
        """`str(Path)` renders a relative path with backslashes on Windows.

        None of the prefixes below ever matched one, so on a Windows host
        `_gated_declaring_headers` resolved 1 of the 6 gated translation units
        in this tree and 0 in the suite's own fixtures -- a guard that cannot
        name the header it forbids cannot fire.  The key also has to equal the
        spelling inside an `#include "..."` directive, which is forward slashes
        on every host.
        """

        for relative, expected in (
            ("include\\vllm\\engine.h", "vllm/engine.h"),
            ("src\\vllm\\entrypoints\\openai\\api_server.h", "vllm/entrypoints/openai/api_server.h"),
            ("tests\\parity\\test_plain.cpp", "parity/test_plain.cpp"),
            ("include/vllm/engine.h", "vllm/engine.h"),
            ("vllm/engine.h", "vllm/engine.h"),
        ):
            with self.subTest(relative=relative):
                self.assertEqual(mod._include_spelling(relative), expected)

    def test_every_gated_translation_unit_resolves_its_declaring_header(self) -> None:
        """Derived from the tree, so it extends itself.

        The expected set is the guarded `target_sources` sources the top-level
        CMakeLists actually declares -- read, never listed -- so a seventh gated
        unit is covered the day it lands.  On Windows this read 1 of 6.
        """

        top = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
        gated_sources = [
            argument
            for _, name, arguments, guarded in mod._guarded_commands(top)
            if name == "target_sources" and guarded
            for argument in arguments
            if argument.endswith(mod._SOURCE_SUFFIXES)
        ]
        self.assertTrue(gated_sources, "the tree declares no guarded target_sources")
        headers = mod._gated_declaring_headers(ROOT)
        self.assertEqual(len(headers), len(gated_sources), sorted(headers))

    # --- the CPU-only configure must not reach the network ----------------

    def test_every_fetch_content_site_is_pinned_off(self) -> None:
        """The count is read from the tree, so a fourth site cannot join quietly.

        `NETWORK_FETCH_OPTIONS` is what the configure turns off; the tree is
        what actually fetches.  While those two agree, the CPU-only configure
        this gate performs is offline, and a failed git checkout can no longer
        be reported as a missing test target.
        """

        sites = sum(
            path.read_text(encoding="utf-8", errors="replace").count("FetchContent_Declare")
            for pattern in ("CMakeLists.txt", "*.cmake")
            for path in ROOT.rglob(pattern)
            if ".git" not in path.parts
        )
        self.assertEqual(sites, len(mod.NETWORK_FETCH_OPTIONS))

    def test_each_network_fetch_option_is_forced_off(self) -> None:
        for option in mod.NETWORK_FETCH_OPTIONS:
            with self.subTest(option=option):
                self.assertIn(f"-D{option}=OFF", mod.CPU_ONLY_CONFIGURE_ARGS)

    def test_the_parakeet_pin_is_load_bearing(self) -> None:
        """The control: the option still defaults ON, so the pin is not a no-op.

        If this stops matching, someone turned the option off at its
        declaration and the argument above became decoration.  The gate would
        still be correct -- which is exactly why it has to be checked rather
        than assumed.
        """

        top = (ROOT / "CMakeLists.txt").read_text(encoding="utf-8", errors="replace")
        self.assertRegex(
            top, r"option\(\s*VLLM_CPP_WITH_DIARIZATION\b[^\n]*\bON\s*\)"
        )


class ShippedTreeTests(unittest.TestCase):
    def test_shipped_tree_is_registered_and_wired(self) -> None:
        self.assertEqual(mod.check_tree(ROOT), [])


class SuiteIntegrityTests(unittest.TestCase):
    def test_deleting_wrapper_assertions_is_caught(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        mutated = source.replace(
            "        self.assertTrue(any(needle in error for error in errors), errors)\n",
            "",
        )
        self.assertNotEqual(mutated, source)
        errors = _suite_integrity_errors(mutated)
        self.assertTrue(any("assert_error has no direct semantic assertion" in e for e in errors))
        self.assertTrue(
            any("assert_wiring_error has no direct semantic assertion" in e for e in errors)
        )

    def test_deleting_a_manifested_mutation_is_caught(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "test_M19_ci_commands_as_quoted_text_fail"
        )
        lines = source.splitlines(keepends=True)
        mutated = "".join(lines[: method.lineno - 1] + lines[method.end_lineno :])
        errors = _suite_integrity_errors(mutated)
        self.assertTrue(any("fixed manifest" in error for error in errors), errors)

    def test_mutation_cases_keep_their_outcome_assertions(self) -> None:
        """Make deleting a mutation's only meaningful assertion turn RED."""

        source = Path(__file__).read_text(encoding="utf-8")
        self.assertEqual(_suite_integrity_errors(source), [])

    def test_assertion_helpers_keep_semantic_outcome_assertions(self) -> None:
        """The mutation wrappers must themselves retain a real assertion."""

        source = Path(__file__).read_text(encoding="utf-8")
        errors = _suite_integrity_errors(source)
        self.assertFalse(any("assert_error" in error for error in errors), errors)
        self.assertFalse(any("assert_wiring_error" in error for error in errors), errors)

        tree = ast.parse(source)
        methods = {
            node.name: node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        }
        for name in {
            "test_minimal_complete_registration_passes",
            "test_complete_preflight_and_ci_wiring_passes",
            "test_shipped_tree_is_registered_and_wired",
        }:
            method = methods[name]
            calls = {
                call.func.attr
                for call in ast.walk(method)
                if isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
            }
            self.assertIn("assertEqual", calls, name)

    def test_suite_integrity_contract_is_pinned(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        self.assertEqual(_suite_integrity_errors(source), [])

    def test_M38_deleting_M18_and_manifest_entry_fails(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "test_M18_ci_commands_behind_false_shell_branch_fail"
        )
        lines = source.splitlines(keepends=True)
        mutated_source = "".join(lines[: method.lineno - 1] + lines[method.end_lineno :])
        manifest = mod.MUTATION_MANIFEST.read_text(encoding="utf-8")
        mutated_manifest = manifest.replace(
            "test_M18_ci_commands_behind_false_shell_branch_fail\n", ""
        )
        errors = mod.mutation_suite_integrity_errors(
            mutated_source, manifest_text=mutated_manifest
        )
        self.assertTrue(any("pinned digest" in error for error in errors), errors)

    def test_M39_renaming_M18_and_manifest_entry_fails(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        old = "test_M18_ci_commands_behind_false_shell_branch_fail"
        new = "test_M18_ci_commands_hidden_by_false_shell_branch_fail"
        mutated_source = source.replace(old, new)
        manifest = mod.MUTATION_MANIFEST.read_text(encoding="utf-8")
        mutated_manifest = manifest.replace(old, new)
        errors = mod.mutation_suite_integrity_errors(
            mutated_source, manifest_text=mutated_manifest
        )
        self.assertTrue(any("pinned digest" in error for error in errors), errors)

    def test_M40_redirecting_suite_to_alternate_manifest_fails(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        mutated = source.replace(
            "source, manifest_path=mod.MUTATION_MANIFEST",
            'source, manifest_path=ROOT / "tests/scripts/alternate_mutations.txt"',
            1,
        )
        self.assertNotEqual(mutated, source)
        errors = mod.mutation_suite_integrity_errors(
            mutated, manifest_text=mod.MUTATION_MANIFEST.read_text(encoding="utf-8")
        )
        self.assertTrue(any("canonical manifest" in error for error in errors), errors)

    def test_M41_deleting_manifest_integrity_test_fails(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        tree = ast.parse(source)
        method = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name == "test_suite_integrity_contract_is_pinned"
        )
        lines = source.splitlines(keepends=True)
        mutated = "".join(lines[: method.lineno - 1] + lines[method.end_lineno :])
        errors = mod.mutation_suite_integrity_errors(
            mutated, manifest_text=mod.MUTATION_MANIFEST.read_text(encoding="utf-8")
        )
        self.assertTrue(any("integrity method is missing" in error for error in errors), errors)

    def test_M42_byte_identical_alternate_manifest_path_fails(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        manifest = mod.MUTATION_MANIFEST.read_text(encoding="utf-8")
        with tempfile.TemporaryDirectory(
            prefix="vllm-registration-manifest-"
        ) as temporary:
            alternate = Path(temporary) / mod.MUTATION_MANIFEST.name
            alternate.write_text(manifest, encoding="utf-8")
            self.assertTrue(
                alternate.read_bytes() == mod.MUTATION_MANIFEST.read_bytes()
            )
            errors = mod.mutation_suite_integrity_errors(
                source, manifest_path=alternate
            )
        self.assertEqual(
            errors, ["mutation suite must use the canonical manifest path"]
        )

    def test_M43_M42_keeps_exact_path_outcome_assertion(self) -> None:
        source = Path(__file__).read_text(encoding="utf-8")
        assertion = (
            "        self.assertEqual(\n"
            '            errors, ["mutation suite must use the canonical manifest path"]\n'
            "        )\n"
        )
        for label, replacement in {
            "deleted": "",
            "wrong diagnostic": "        self.assertEqual(errors, [])\n",
        }.items():
            with self.subTest(mutation=label):
                mutated = source.replace(assertion, replacement, 1)
                self.assertNotEqual(mutated, source)
                errors = _suite_integrity_errors(mutated)
                self.assertTrue(
                    "test_M42 must assert the exact canonical-manifest path diagnostic"
                    in errors,
                    errors,
                )



class LabelSelectionMutationTests(unittest.TestCase):
    """The pinned CTest label selection, and the four ways it goes wrong.

    `ctest -L gpu` prints `No tests were found!!!` and returns **0** when the
    label selects nothing (measured, CMake 3.28.3), so the documented
    device-gate recipe fails OPEN. Only a pinned selection separates a lane
    that ran the gate from a lane that ran nothing.
    """

    def test_label_selection_baseline_passes(self) -> None:
        self.assertEqual(mod.label_errors(PASSING_LABEL_CMAKE), [])

    def test_M44_renaming_the_gpu_label_fails(self) -> None:
        mutated = PASSING_LABEL_CMAKE.replace(
            'LABELS "gpu;checkpoint;music3"', 'LABELS "device;checkpoint;music3"'
        )
        # Both labelled entries carry the same property text, so this renames the
        # lane out of existence rather than half of it -- the empty selection is
        # the dangerous case, because `ctest -L` returns 0 over it.
        self.assertNotEqual(mutated, PASSING_LABEL_CMAKE)
        errors = mod.label_errors(mutated)
        self.assertTrue(
            any("ctest -L gpu selects 0 test(s) [<none>]" in error for error in errors),
            errors,
        )

    def test_M45_deleting_the_labels_property_fails(self) -> None:
        mutated = PASSING_LABEL_CMAKE.replace(
            "set_tests_properties(test_minimax_music3_device_arm_real PROPERTIES\n"
            '  LABELS "gpu;checkpoint;music3")\n',
            "",
        ).replace(
            "set_tests_properties(test_minimax_music3_depth_arm_real PROPERTIES\n"
            '  LABELS "gpu;checkpoint;music3")\n',
            "",
        )
        self.assertNotEqual(mutated, PASSING_LABEL_CMAKE)
        errors = mod.label_errors(mutated)
        self.assertTrue(
            any("ctest -L gpu selects 0 test(s) [<none>]" in error for error in errors),
            errors,
        )

    def test_M46_a_second_test_taking_the_gpu_label_fails(self) -> None:
        mutated = PASSING_LABEL_CMAKE + (
            "set_tests_properties(test_device_selection PROPERTIES\n"
            '  LABELS "gpu")\n'
        )
        self.assertNotEqual(mutated, PASSING_LABEL_CMAKE)
        errors = mod.label_errors(mutated)
        self.assertTrue(
            any("selects 3 test(s)" in error for error in errors), errors
        )

    def test_M48_renaming_one_of_two_gpu_labels_fails(self) -> None:
        """One gate leaves the lane and the other stays: a NON-empty miss.

        Only reachable at n > 1, which the production pin has been since #1839.
        A floor ("at least one test carries the label") passes this mutation and
        the lane silently runs half of what it documents.
        """

        mutated = PASSING_LABEL_CMAKE.replace(
            "set_tests_properties(test_minimax_music3_depth_arm_real PROPERTIES\n"
            '  LABELS "gpu;checkpoint;music3")\n',
            "set_tests_properties(test_minimax_music3_depth_arm_real PROPERTIES\n"
            '  LABELS "checkpoint;music3")\n',
        )
        self.assertNotEqual(mutated, PASSING_LABEL_CMAKE)
        errors = mod.label_errors(mutated)
        self.assertTrue(
            any(
                "selects 1 test(s) [test_minimax_music3_device_arm_real]" in error
                for error in errors
            ),
            errors,
        )

    def test_M47_deleting_the_labelled_test_registration_fails(self) -> None:
        mutated = PASSING_LABEL_CMAKE.replace(
            "vllm_cpp_add_test(test_minimax_music3_device_arm_real\n"
            "  parity/test_minimax_music3_device_arm_real.cpp)\n",
            "",
        ).replace(
            "set_tests_properties(test_minimax_music3_device_arm_real PROPERTIES\n"
            '  LABELS "gpu;checkpoint;music3")\n',
            "",
        ).replace(
            "vllm_cpp_add_test(test_minimax_music3_depth_arm_real\n"
            "  parity/test_minimax_music3_depth_arm_real.cpp)\n",
            "",
        ).replace(
            "set_tests_properties(test_minimax_music3_depth_arm_real PROPERTIES\n"
            '  LABELS "gpu;checkpoint;music3")\n',
            "",
        )
        self.assertNotEqual(mutated, PASSING_LABEL_CMAKE)
        errors = mod.label_errors(mutated)
        self.assertTrue(
            any("ctest -L gpu selects 0 test(s) [<none>]" in error for error in errors),
            errors,
        )



class ServerGuardMutationTests(unittest.TestCase):
    """The registration-must-agree-with-the-link guard (#1883).

    A test target registered outside `if(VLLM_CPP_SERVER)` that links a
    translation unit compiled only inside it configures cleanly and then fails
    at `ld`. No configure-only gate can see that, so this one is static.
    """

    #
    # These build a small TREE rather than a CMake string, because the check
    # reads three things that only exist on disk together: the gated
    # `target_sources` in the top-level file, the declaring header beside the
    # gated source, and the `#include` inside the test source.  A string
    # fixture could not express the third and would gate a parser instead of
    # the contract.
    # ------------------------------------------------------------------

    GUARD_TOP = """
cmake_minimum_required(VERSION 3.20)
project(server_guard LANGUAGES CXX)
add_library(vllm)
if(VLLM_CPP_SERVER)
  target_sources(vllm PRIVATE src/vllm/entrypoints/openai/api_server.cpp)
endif()
target_sources(vllm PRIVATE src/vllm/engine.cpp)
"""

    GUARD_TESTS = """
vllm_cpp_add_test(test_plain parity/test_plain.cpp)
vllm_cpp_add_test(test_links_server parity/test_links_server.cpp)
if(VLLM_CPP_SERVER)
  vllm_cpp_add_test(test_inside parity/test_inside.cpp)
endif()
"""

    GUARD_FILES = {
        "include/vllm/entrypoints/openai/api_server.h": "#pragma once\nstruct ApiServer;\n",
        "include/vllm/engine.h": "#pragma once\nstruct Engine;\n",
        "src/vllm/entrypoints/openai/api_server.cpp": '#include "vllm/entrypoints/openai/api_server.h"\n',
        "src/vllm/engine.cpp": '#include "vllm/engine.h"\n',
        "tests/parity/test_plain.cpp": '#include "vllm/engine.h"\n',
        "tests/parity/test_links_server.cpp": '#include "vllm/entrypoints/openai/api_server.h"\n',
        "tests/parity/test_inside.cpp": '#include "vllm/entrypoints/openai/api_server.h"\n',
    }

    def guard_tree(
        self,
        stack: contextlib.ExitStack,
        *,
        top: str | None = None,
        tests: str | None = None,
        files: dict[str, str] | None = None,
    ) -> Path:
        root = Path(stack.enter_context(tempfile.TemporaryDirectory(prefix="vllm-server-guard-")))
        (root / "CMakeLists.txt").write_text(self.GUARD_TOP if top is None else top, encoding="utf-8")
        (root / "tests").mkdir()
        (root / "tests/CMakeLists.txt").write_text(
            self.GUARD_TESTS if tests is None else tests, encoding="utf-8"
        )
        for relative, body in {**self.GUARD_FILES, **(files or {})}.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(body, encoding="utf-8")
        return root

    def assert_server_guard_error(self, root: Path, needle: str) -> None:
        errors = mod.server_guard_errors(root)
        self.assertTrue(any(needle in error for error in errors), errors)

    def test_server_guard_passes_when_every_linking_target_is_inside(self) -> None:
        """The positive control, and it is not vacuous.

        `test_inside` reaches the SAME gated header as the mutation cases below
        and is clean only because of where it is registered.  Without it the
        suite could not tell a check that reads the guard from one that never
        finds the header at all.
        """

        with contextlib.ExitStack() as stack:
            tests = self.GUARD_TESTS.replace(
                "vllm_cpp_add_test(test_links_server parity/test_links_server.cpp)\n", ""
            )
            self.assertNotEqual(tests, self.GUARD_TESTS)
            self.assertEqual(mod.server_guard_errors(self.guard_tree(stack, tests=tests)), [])

    def test_server_guard_ignores_a_comment_only_mention(self) -> None:
        """A source that only NAMES the header in prose links nothing."""

        with contextlib.ExitStack() as stack:
            tests = self.GUARD_TESTS.replace(
                "vllm_cpp_add_test(test_links_server parity/test_links_server.cpp)\n", ""
            )
            files = {
                "tests/parity/test_plain.cpp": (
                    '// see "vllm/entrypoints/openai/api_server.h" for the route table\n'
                    '/* #include "vllm/entrypoints/openai/api_server.h" */\n'
                    '#include "vllm/engine.h"\n'
                ),
            }
            self.assertEqual(
                mod.server_guard_errors(self.guard_tree(stack, tests=tests, files=files)), []
            )

    def test_server_guard_accepts_the_real_tree(self) -> None:
        """The production tree itself, which is what CI runs."""

        self.assertEqual(mod.server_guard_errors(mod.ROOT), [])

    def test_M49_registering_a_server_linking_target_outside_the_guard_fails(self) -> None:
        with contextlib.ExitStack() as stack:
            self.assert_server_guard_error(
                self.guard_tree(stack), "test_links_server is registered outside"
            )

    def test_M50_moving_a_gated_target_out_of_the_guard_fails(self) -> None:
        """The `test_inside` target, un-guarded. The exact #1883 shape."""

        with contextlib.ExitStack() as stack:
            tests = (
                "vllm_cpp_add_test(test_plain parity/test_plain.cpp)\n"
                "vllm_cpp_add_test(test_inside parity/test_inside.cpp)\n"
            )
            self.assert_server_guard_error(
                self.guard_tree(stack, tests=tests), "test_inside is registered outside"
            )

    def test_M51_a_transitive_gated_include_outside_the_guard_fails(self) -> None:
        """`downloader.h` includes `hf_hub.h`: one hop is not the only shape."""

        with contextlib.ExitStack() as stack:
            files = {
                "include/vllm/relay.h": (
                    '#pragma once\n#include "vllm/entrypoints/openai/api_server.h"\n'
                ),
                "tests/parity/test_links_server.cpp": '#include "vllm/relay.h"\n',
            }
            self.assert_server_guard_error(
                self.guard_tree(stack, files=files), "test_links_server is registered outside"
            )

    def test_M52_the_else_branch_of_the_server_guard_is_not_guarded(self) -> None:
        """`else()` is the branch that runs when the flag is OFF."""

        with contextlib.ExitStack() as stack:
            tests = (
                "if(VLLM_CPP_SERVER)\n"
                "  vllm_cpp_add_test(test_plain parity/test_plain.cpp)\n"
                "else()\n"
                "  vllm_cpp_add_test(test_links_server parity/test_links_server.cpp)\n"
                "endif()\n"
            )
            self.assert_server_guard_error(
                self.guard_tree(stack, tests=tests), "test_links_server is registered outside"
            )

    def test_M53_deleting_the_gated_target_sources_fails(self) -> None:
        """An empty gated set must be reported, never passed as clean."""

        with contextlib.ExitStack() as stack:
            top = self.GUARD_TOP.replace(
                "  target_sources(vllm PRIVATE src/vllm/entrypoints/openai/api_server.cpp)\n",
                "",
            )
            self.assertNotEqual(top, self.GUARD_TOP)
            self.assert_server_guard_error(
                self.guard_tree(stack, top=top), "would pass vacuously"
            )

    def test_M54_commenting_out_the_gated_target_sources_fails(self) -> None:
        with contextlib.ExitStack() as stack:
            top = self.GUARD_TOP.replace(
                "  target_sources(vllm PRIVATE src/vllm/entrypoints/openai/api_server.cpp)",
                "  # target_sources(vllm PRIVATE src/vllm/entrypoints/openai/api_server.cpp)",
            )
            self.assertNotEqual(top, self.GUARD_TOP)
            self.assert_server_guard_error(
                self.guard_tree(stack, top=top), "would pass vacuously"
            )

    def test_M55_the_guard_helper_must_call_the_production_check(self) -> None:
        """The integrity layer pins `assert_server_guard_error` to its callee."""

        source = mod.MUTATION_SUITE.read_text(encoding="utf-8").replace(
            "errors = mod.server_guard_errors(root)",
            "errors = []  # mod.server_guard_errors(root)",
        )
        self.assertNotEqual(source, mod.MUTATION_SUITE.read_text(encoding="utf-8"))
        errors = _suite_integrity_errors(source)
        self.assertTrue(
            "assert_server_guard_error does not call server_guard_errors" in errors,
            errors,
        )



if __name__ == "__main__":
    unittest.main()
