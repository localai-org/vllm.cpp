#!/usr/bin/env python3
"""Compare bounded native full-target logits to matched-prefix original repeats."""
import argparse
import json
from pathlib import Path
import struct

from capture_projection import IMAGE
from capture_target import probability_metrics, target_phases, validate_prefix_witnesses
from compare_projection import blob
from extract_projection import digest, headers


def compare_row(native, expected, dtype, columns):
    width, fmt = {"F32": (4, "<f"), "F16": (2, "<e")}[dtype]
    headers.require(len(native) == columns * 4 and len(expected) == columns * width,
                    "full-target logit byte count mismatch")
    a = [x for (x,) in struct.iter_unpack("<f", native)]
    e = [x for (x,) in struct.iter_unpack(fmt, expected)]
    return probability_metrics(a, e)


def write_comparison_report(report, result, mode):
    headers.require(mode in ("gate", "report_only"), "unknown target comparison mode")
    headers.require(bool(result["comparisons"]), "empty comparison cannot pass a gate")
    triggers = [row for row in result["comparisons"] if row["investigation_trigger"]]
    passed = not triggers
    result.update({"schema": 2, "mode": mode, "executed": True,
                   "investigation_triggers": triggers, "logit_gate_pass": passed,
                   "status": "passed_bounded_logit_comparison" if passed else "investigation_required",
                   "operator_pass": None, "target_parity_pass": None if passed else False,
                   "autonomous_smoke_pass": None, "serving_qualified": False,
                   "blocking_issue_ids": [] if passed else ["S1_D64"],
                   "qualification_scope": "Matched-prefix logits only; state, autonomous and serving qualification not evaluated.",
                   "gate_exit_code": 0 if passed else 1,
                   "process_exit_code": 1 if mode == "gate" and not passed else 0})
    with report.open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print("TARGET_COMPARISONS", len(result["comparisons"]),
          "INVESTIGATION_TRIGGERS", len(triggers), "STATUS", result["status"], "MODE", mode)
    return result["process_exit_code"]


def compare(args):
    headers.require(not args.report.exists(), "refusing to overwrite target comparison")
    metadata = headers.read_json(args.reference_dir / "comparison.json")
    headers.require(metadata["image"] == IMAGE and metadata["decode_steps"] == 64,
                    "requires pinned original D64 reference")
    native_ids = headers.read_json(args.native_trace_json)["output_ids"]
    references = []
    for i, identity in enumerate(metadata["repeats"]):
        path = args.reference_dir / f"repeat-{i}.safetensors"
        record = headers.read_json(path.with_suffix(".json"))
        headers.require("actual_input_witnesses" in record, "actual reference inputs were not observed")
        validate_prefix_witnesses(record["actual_input_witnesses"], metadata["prompt_token_ids"], native_ids[0])
        headers.require(record["output_ids"] == native_ids and
                        digest(path.read_bytes()) == identity["sha256"] == record["capture_sha256"],
                        "reference prefix or capture identity mismatch")
        references.append((path, headers.read_shard_header(path)))
    headers.require(bool(references), "reference repeats missing")
    rows, native_identities = [], []
    for phase in target_phases(64):
        path = Path(str(args.native_prefix) + "-" + phase + "-logits.f32")
        native = path.read_bytes()
        native_identities.append({"path": str(path), "sha256": digest(native)})
        for repeat, (reference, header) in enumerate(references):
            entry, expected = blob(reference, header, phase + "_logits")
            headers.require(entry["shape"] == [1,248320], "requires full target vocabulary")
            result = compare_row(native, expected, entry["dtype"], 248320)
            rows.append({"phase": phase, "reference_repeat": repeat, **result})
    result = {"schema": 1, "image": IMAGE, "native_trace_json_sha256": digest(args.native_trace_json.read_bytes()),
              "reference_comparison_sha256": digest((args.reference_dir / "comparison.json").read_bytes()),
              "native_logits": native_identities, "comparisons": rows,
              "investigation_triggers": [r for r in rows if r["investigation_trigger"]],
              "scope": "Same-prefix whole-target logits only. Does not replace failed native tests or qualify state/performance."}
    return write_comparison_report(args.report, result, args.mode)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-prefix", type=Path, required=True)
    parser.add_argument("--native-trace-json", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--gate", dest="mode", action="store_const", const="gate",
                       help="Fail if the frozen logit gates are unmet (default).")
    modes.add_argument("--report-only", dest="mode", action="store_const", const="report_only",
                       help="Write diagnostics without treating exit0 as a quality pass.")
    parser.set_defaults(mode="gate")
    return parser.parse_args(argv)


if __name__ == "__main__":
    raise SystemExit(compare(parse_args()))
