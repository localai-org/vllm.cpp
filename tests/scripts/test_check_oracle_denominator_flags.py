#!/usr/bin/env python3
"""Unit and mutation checks for scripts/check-oracle-denominator-flags.py (#414, #607).

The case this suite exists for is `test_real_tree_discovers_the_known_launches`.
A detector that stopped matching would report a clean tree, and a clean tree is
what a broken detector and a correct one both look like from the exit status
alone. So the discovered SET is pinned by file, not only the violation count,
and the checker itself exits 2 rather than 0 when it finds nothing at all.

`test_flag_removed_from_the_canonical_driver_goes_red` is the mutation: it is
the state the tree was actually in before #607 wave L4, and it must be
detectable, not merely described.
"""

from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CHECKER = ROOT / "scripts/check-oracle-denominator-flags.py"
DRIVER = ROOT / "scripts/dgx-online-serving.sh"


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, str(CHECKER), *args],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )


def tree(**files: str) -> tempfile.TemporaryDirectory:
    """A scratch root carrying the given harness files."""
    tmp = tempfile.TemporaryDirectory()
    root = Path(tmp.name)
    for relative, body in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(body, encoding="utf-8")
    return tmp


# The real shape, reduced: an oracle server launched from a shell array.
SERVE_NO_FLAG = """\
start_server() {
  server_cmd=(
    env "PATH=$(dirname "${client}"):${PATH}"
    "${client}" serve "${snapshot}"
    --served-model-name gate
    --port "${port}"
  )
}
"""

SERVE_WITH_FLAG = SERVE_NO_FLAG.replace(
    "    --served-model-name gate\n",
    "    --served-model-name gate\n    --language-model-only\n",
)

SERVE_EXEMPT = SERVE_NO_FLAG.replace(
    "  server_cmd=(\n",
    "  # ORACLE-DENOMINATOR-EXEMPT: dense text arch, no multimodal_config\n"
    "  server_cmd=(\n",
)

SERVE_EXEMPT_NO_REASON = SERVE_NO_FLAG.replace(
    "  server_cmd=(\n",
    "  # ORACLE-DENOMINATOR-EXEMPT:\n  server_cmd=(\n",
)


class OracleDenominatorFlagsTest(unittest.TestCase):
    def test_a_serve_launch_without_the_flag_is_refused(self) -> None:
        with tree(**{"scripts/h.sh": SERVE_NO_FLAG}) as root:
            got = run("--root", root)
        self.assertEqual(got.returncode, 1, got.stdout)
        self.assertIn("scripts/h.sh:4", got.stdout)
        self.assertIn("--language-model-only", got.stdout)
        self.assertIn("#414", got.stdout)

    def test_a_serve_launch_with_the_flag_passes(self) -> None:
        with tree(**{"scripts/h.sh": SERVE_WITH_FLAG}) as root:
            got = run("--root", root)
        self.assertEqual(got.returncode, 0, got.stdout)
        self.assertIn("1 oracle server launch(es)", got.stdout)

    def test_an_exemption_with_a_reason_passes_and_prints_the_reason(self) -> None:
        with tree(**{"scripts/h.sh": SERVE_EXEMPT}) as root:
            got = run("--root", root)
        self.assertEqual(got.returncode, 0, got.stdout)
        self.assertIn("dense text arch, no multimodal_config", got.stdout)

    def test_an_exemption_without_a_reason_is_not_an_exemption(self) -> None:
        with tree(**{"scripts/h.sh": SERVE_EXEMPT_NO_REASON}) as root:
            got = run("--root", root)
        self.assertEqual(got.returncode, 1, got.stdout)

    def test_bench_serve_is_the_timed_client_and_is_not_a_launch(self) -> None:
        # `vllm bench serve` issues the requests. Matching it would demand a
        # server flag on a client, which vLLM would reject.
        body = 'cmd=(\n  "${client}" bench serve --backend openai\n)\n'
        with tree(**{"scripts/h.sh": body}) as root:
            got = run("--root", root)
        self.assertEqual(got.returncode, 2, got.stdout)
        self.assertIn("broken detector", got.stdout)

    def test_any_variable_spelling_of_the_client_is_still_a_launch(self) -> None:
        # The evasion surface. The detector used to name the two variables this
        # tree happens to use, `${client}` and `${VLLM_ORACLE}`, so a harness
        # that called its binary anything else launched the oracle unscanned and
        # the checker reported a clean tree. What identifies a launch is the
        # `serve` subcommand sitting directly after the binary, not the name
        # somebody gave the variable holding it.
        for spelling in ('"$ORACLE"', '"${server_bin}"', "$oracle_cli", "${VENV}"):
            with self.subTest(spelling=spelling):
                body = f'cmd=(\n  {spelling} serve "${{snapshot}}" --port 8000\n)\n'
                with tree(**{"scripts/h.sh": body}) as root:
                    got = run("--root", root)
                self.assertEqual(got.returncode, 1, got.stdout)
                self.assertIn("scripts/h.sh:2", got.stdout)

    def test_bench_serve_stays_a_client_under_any_variable_spelling(self) -> None:
        # The guard on the line above. Widening to any `$`-expansion must not
        # start matching `bench serve`, which is the timed CLIENT: the token
        # before `serve` there is `bench`, which is no expansion at all.
        for spelling in ('"$ORACLE"', '"${server_bin}"', '"${client}"'):
            with self.subTest(spelling=spelling):
                body = f'cmd=(\n  {spelling} bench serve --backend openai\n)\n'
                with tree(**{"scripts/h.sh": body}) as root:
                    got = run("--root", root)
                self.assertEqual(got.returncode, 2, got.stdout)
                self.assertIn("broken detector", got.stdout)

    def test_prose_naming_the_flag_free_oracle_is_not_a_launch(self) -> None:
        # Case matters: prose writes "vLLM", commands write "vllm". A comment
        # line is never a command either.
        body = (
            '# the vLLM serve arm runs without it\n'
            'echo "see vllm/benchmarks/serve.py for the client"\n'
            'true_cmd=(\n  "${client}" serve "${snap}" --language-model-only\n)\n'
        )
        with tree(**{"scripts/h.sh": body}) as root:
            got = run("--root", root, "--json")
        self.assertEqual(got.returncode, 0, got.stdout)
        payload = json.loads(got.stdout[got.stdout.index("[") : got.stdout.rindex("]") + 1])
        self.assertEqual([r["line"] for r in payload], [4], payload)

    def test_a_python_harness_launch_is_matched_too(self) -> None:
        body = (
            '"vllm_server": [\n'
            '    "<VLLM_ORACLE>/bin/vllm", "serve", "<MODEL_SNAPSHOT>",\n'
            '    "--served-model-name", "gate",\n'
            "],\n"
        )
        with tree(**{"tools/bench/h.py": body}) as root:
            got = run("--root", root)
        self.assertEqual(got.returncode, 1, got.stdout)
        self.assertIn("tools/bench/h.py:2", got.stdout)

    def test_an_empty_tree_is_a_broken_detector_not_a_pass(self) -> None:
        with tree(**{"scripts/h.sh": "echo nothing here\n"}) as root:
            got = run("--root", root)
        self.assertEqual(got.returncode, 2, got.stdout)

    def test_real_tree_is_clean(self) -> None:
        got = run()
        self.assertEqual(got.returncode, 0, got.stdout)

    def test_real_tree_discovers_the_known_launches(self) -> None:
        got = run("--json")
        self.assertEqual(got.returncode, 0, got.stdout)
        payload = json.loads(got.stdout[got.stdout.index("[") : got.stdout.rindex("]") + 1])
        files = sorted({r["file"] for r in payload})
        self.assertEqual(
            files,
            [
                "scripts/dgx-online-serving.sh",
                "scripts/gen-clm-goldens.py",
                "tools/bench/run_serve_low.py",
            ],
            payload,
        )
        # Both arms of the canonical driver, plus the low-batch harness. A drop
        # below three means the detector stopped seeing a launch it used to see.
        self.assertGreaterEqual(len(payload), 3, payload)
        self.assertTrue(
            any(r["exempt_reason"] for r in payload),
            "the q3mxfp4 dense arm should be the exempt one",
        )
        self.assertTrue(
            any(
                r["file"] == "tools/bench/run_serve_low.py" and r["has_flag"]
                for r in payload
            ),
            payload,
        )

    def test_flag_removed_from_the_canonical_driver_goes_red(self) -> None:
        # The pre-L4 state of the tree, reconstructed. This is the mutation that
        # proves the gate detects the defect rather than describing it.
        original = DRIVER.read_text(encoding="utf-8")
        self.assertIn("      --language-model-only\n", original)
        mutated = original.replace("      --language-model-only\n", "", 1)
        self.assertNotEqual(mutated, original, "mutation did not apply")
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            (root / "scripts").mkdir()
            (root / "scripts" / DRIVER.name).write_text(mutated, encoding="utf-8")
            got = run("--root", str(root))
        self.assertEqual(got.returncode, 1, got.stdout)
        self.assertIn("dgx-online-serving.sh", got.stdout)
        # The exempt q3mxfp4 arm must stay exempt: the mutation removes one
        # launch's flag, not the exemption of the other.
        self.assertEqual(got.stdout.count("launches the pinned vLLM oracle"), 1, got.stdout)


if __name__ == "__main__":
    unittest.main(verbosity=2)
