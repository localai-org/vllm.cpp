#!/usr/bin/env python3
# GLM/DSA G1 (GLM-4.7-Flash, `Glm4MoeLiteForCausalLM`) -- the K1 + K2 LEASE RUNNER.
#
# Two probes decide this row's next step, and both are oracle runs:
#
#   K1  scripts/glm4-moe-lite-neartie-gap.py       (#3565) -- the teacher-forced
#       gap at full resolution: is the oracle's argmax ours, and by how much?
#   K2  scripts/glm4-moe-lite-path-determinism.py  (#3566) -- does that argmax
#       depend on the numeric path the oracle ran through?
#
# Neither is safe to point at the tracked directory. K1 rewrites `our_ids.npy`
# and `neartie_gap_mnats.npy` into whatever `--golden-dir` names, and that
# directory is `tests/parity/goldens/glm4_moe_lite_greedy`, whose files are
# committed: run it in place and the next `git add -A` sweeps a regenerated
# artifact into an unrelated commit. So this runner copies the goldens into a
# scratch directory first, runs both tools there, and then diffs every artifact
# the run produced against the committed copy.
#
# The diff is the reason to have a runner at all. "The oracle still reproduces
# its own capture" is a claim about files, and it should be computed rather than
# eyeballed: `summary.json` records each artifact's differing cell count, the K1
# sub-millinat count, and the K2 verdict.
#
# RUN INSIDE A LEASE, NEVER OVER `ssh` (AGENTS.md, "Work on a GPU happens inside
# a lease"). The client is `rc`; confirm its flags with `rc run --help` on the
# host that has it, because the form below is the one this repository records:
#
#   rc run -d dgx:gpu0 --max-runtime 180m --cwd /workspace/vllm.cpp -- \
#       python3 scripts/glm4-moe-lite-k1k2-run.py \
#           --python ~/venvs/vllm-oracle/bin/python \
#           --out-dir /tmp/glm4-k1k2
#
# The oracle python must be the parity-pin venv (`a7c23ac96d`), the snapshot must
# be `zai-org/GLM-4.7-Flash`, and nothing else may be on the card.
#
# USEFUL ARMS
#   --dry-run            print the plan and the tool resolution, run nothing.
#   --keep               leave the scratch directory in place for inspection.
#   --fail-on-diff       exit 1 when the oracle does not reproduce the goldens;
#                        default is to report and exit 0, because a diff is a
#                        finding and not a broken run.
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

TOOL_K1 = "glm4-moe-lite-neartie-gap.py"
TOOL_K2 = "glm4-moe-lite-path-determinism.py"
TOOL_K1_PROVIDER = "#3565"
TOOL_K2_PROVIDER = "#3566"

# What a successful run writes into the scratch directory, in the order the
# summary reads them. The last two only exist once K1 has landed.
ARTIFACTS = [
    "our_ids.npy",
    "neartie_gap_mnats.npy",
    "teacher_forced_gap_nats.npy",
    "teacher_forced_argmax.npy",
]
SUMMARY_NAME = "summary.json"
K1_NATS_NAME = "teacher_forced_gap_nats.npy"
K2_REPORT_NAME = "path_determinism.json"
SUB_MILLINAT_NATS = 0.0005


class ToolMissing(RuntimeError):
    """A probe the runner needs is not in the tree yet."""


def required_tools(tool_dir):
    """Resolve both probe scripts, or say exactly what is missing and where from.

    A runner that silently skips a missing probe would report a clean run having
    measured half of what it claims, so absence is an error that names the file
    AND the pull request that supplies it.
    """
    resolved = {}
    missing = []
    for name, provider in ((TOOL_K1, TOOL_K1_PROVIDER),
                           (TOOL_K2, TOOL_K2_PROVIDER)):
        path = Path(tool_dir) / name
        if path.is_file():
            resolved[name] = path
        else:
            missing.append(f"{path} (provided by {provider})")
    if missing:
        raise ToolMissing(
            "the K1/K2 runner cannot run: " + "; ".join(missing) +
            ". Land those probes first, or pass --tool-dir at a tree that has "
            "them.")
    return resolved


def plan_commands(python, tools, golden_dir, topk, max_num_seqs, model,
                  max_tokens):
    """The exact argv for each probe, K1 first because K2 reads its context.

    Both probes see the SAME `--golden-dir`, which is the scratch copy, so one
    invocation cannot leave the other reading a half-written artifact set.
    """
    k1 = [
        python, str(tools[TOOL_K1]),
        "--golden-dir", str(golden_dir),
        "--model", model,
        "--topk", str(topk),
        "--max-tokens", str(max_tokens),
    ]
    k2 = [
        python, str(tools[TOOL_K2]),
        "--golden-dir", str(golden_dir),
        "--model", model,
        "--topk", str(topk),
        "--max-num-seqs", max_num_seqs,
    ]
    return [
        {"name": "k1", "description": "teacher-forced gap at full resolution",
         "argv": k1},
        {"name": "k2", "description": "oracle path determinism", "argv": k2},
    ]


def diff_artifacts(committed_dir, regenerated_dir, names):
    """Compare each artifact the run produced against the committed one.

    `present` is False when the run never wrote the file -- which is the honest
    state before K1 has landed, and is reported rather than counted as equal. A
    present file is compared cell by cell and the first differing index is
    recorded so a reader can open the .npy and look at it.
    """
    diffs = []
    for name in names:
        fresh = Path(regenerated_dir) / name
        old = Path(committed_dir) / name
        if not fresh.is_file():
            diffs.append({"name": name, "present": False, "differing": 0,
                          "first_differing": None})
            continue
        new_arr = np.load(fresh)
        if not old.is_file():
            diffs.append({"name": name, "present": True, "differing": 0,
                          "first_differing": None, "committed_absent": True})
            continue
        old_arr = np.load(old)
        if old_arr.shape != new_arr.shape:
            diffs.append({"name": name, "present": True,
                          "differing": int(new_arr.size),
                          "first_differing": [0, 0], "shape_changed": True})
            continue
        differing = int(np.count_nonzero(old_arr != new_arr))
        first = None
        if differing:
            flat = int(np.flatnonzero(old_arr != new_arr)[0])
            first = list(np.unravel_index(flat, old_arr.shape))
            first = [int(first[0]), int(first[1])]
        diffs.append({"name": name, "present": True, "differing": differing,
                      "first_differing": first})
    return diffs


def summarise(diffs, k1_sub_millinat, k2_verdict):
    """Fold the diffs and the probes' own findings into one record."""
    reproduced = all(d["present"] and d["differing"] == 0 and
                     not d.get("shape_changed") for d in diffs)
    return {
        "reproduced": bool(reproduced),
        "k1_sub_millinat": k1_sub_millinat,
        "k2_verdict": k2_verdict,
        "artifacts": diffs,
        "absent_artifacts": [d["name"] for d in diffs if not d["present"]],
    }


def k1_sub_millinat(out_dir):
    """How many positive gaps the committed millinat array would store as 0."""
    path = Path(out_dir) / K1_NATS_NAME
    if not path.is_file():
        return None
    nats = np.load(path)
    return int(np.count_nonzero((nats > 0.0) & (nats < SUB_MILLINAT_NATS)))


def k2_verdict(out_dir):
    path = Path(out_dir) / K2_REPORT_NAME
    if not path.is_file():
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("verdict")


def stage_goldens(source, destination):
    """Copy the committed artifact set into a scratch directory to run against."""
    source = Path(source).resolve()
    destination = Path(destination).resolve()
    if destination == source or source in destination.parents:
        raise SystemExit(
            f"refusing to run in {destination}: it is the tracked golden "
            f"directory {source}, and both probes rewrite artifacts there")
    destination.mkdir(parents=True, exist_ok=True)
    for entry in sorted(source.iterdir()):
        if entry.is_file():
            shutil.copy2(entry, destination / entry.name)
    return destination


def main():
    ap = argparse.ArgumentParser(
        description="Run the GLM4-MoE-Lite K1 and K2 probes on a leased GPU "
                    "and diff the result against the committed goldens.")
    ap.add_argument("--golden-dir", default="tests/parity/goldens/glm4_moe_lite_greedy")
    ap.add_argument("--out-dir", default=None,
                    help="scratch directory; defaults to a fresh temp dir")
    ap.add_argument("--tool-dir", default="scripts")
    ap.add_argument("--python", default=sys.executable,
                    help="interpreter that has the parity-pin vLLM importable")
    ap.add_argument("--model", default="zai-org/GLM-4.7-Flash")
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=16)
    ap.add_argument("--max-num-seqs", default="1,4")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep", action="store_true")
    ap.add_argument("--fail-on-diff", action="store_true")
    args = ap.parse_args()

    tools = required_tools(args.tool_dir)
    scratch = args.out_dir or tempfile.mkdtemp(prefix="glm4-k1k2-")
    plan = plan_commands(args.python, tools, scratch, args.topk,
                         args.max_num_seqs, args.model, args.max_tokens)

    if args.dry_run:
        print(f"goldens: {args.golden_dir}")
        print(f"scratch: {scratch}")
        for step in plan:
            print(f"  {step['name']}: {' '.join(step['argv'])}")
        print(f"then diff {', '.join(ARTIFACTS)} against {args.golden_dir}")
        return 0

    stage_goldens(args.golden_dir, scratch)
    for step in plan:
        print(f"=== {step['name']}: {step['description']} ===", flush=True)
        result = subprocess.run(step["argv"])
        if result.returncode != 0:
            print(f"the {step['name']} probe exited {result.returncode}; "
                  f"nothing is compared against a half-finished run",
                  file=sys.stderr)
            return result.returncode

    summary = summarise(
        diff_artifacts(args.golden_dir, scratch, ARTIFACTS),
        k1_sub_millinat(scratch),
        k2_verdict(scratch))
    out = Path(scratch) / SUMMARY_NAME
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, sort_keys=True)
        fh.write("\n")

    for diff in summary["artifacts"]:
        state = ("absent" if not diff["present"]
                 else f"{diff['differing']} differing"
                 + (f", first at {diff['first_differing']}"
                    if diff.get("first_differing") else ""))
        print(f"  {diff['name']}: {state}")
    print(f"=== reproduced={summary['reproduced']} "
          f"k1_sub_millinat={summary['k1_sub_millinat']} "
          f"k2_verdict={summary['k2_verdict']} ===")
    print(f"wrote {out}")
    if not args.keep and args.out_dir is None:
        print(f"(scratch kept for inspection: {scratch})")
    return 1 if (args.fail_on_diff and not summary["reproduced"]) else 0


if __name__ == "__main__":
    sys.exit(main())
