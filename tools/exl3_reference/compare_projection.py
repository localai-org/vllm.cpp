#!/usr/bin/env python3
"""Compare S0b stages, retaining diagnostic/production attribution and hashes."""
import argparse
import json
import math
from pathlib import Path
import struct
import sys

from capture_projection import IMAGE, LIB_SHA, OPS_SHA, checked_fixture
from extract_projection import digest, headers, read_span

LIMIT = 2e-3


def blob(path, report, name):
    entry = next(t for t in report["tensors"] if t["name"] == name)
    return entry, read_span(path.parent, report, entry["data_offsets"][0], entry["stored_bytes"])


def bit_stats(actual, expected, width):
    headers.require(len(actual) == len(expected) and len(actual) % width == 0, "stage byte count mismatch")
    first, mismatches = None, 0
    if actual != expected:
        for i in range(0, len(actual), width):
            if actual[i:i + width] != expected[i:i + width]:
                mismatches += 1
                if first is None:
                    first = i // width
    return {"bit_exact": mismatches == 0, "bit_mismatches": mismatches,
            "first_bit_mismatch": first, "elements": len(actual) // width}


def metrics(actual, expected, dtype):
    width, fmt = {"F16": (2, "<e"), "F32": (4, "<f")}[dtype]
    result = bit_stats(actual, expected, width)
    error, norm, maximum = 0.0, 0.0, 0.0
    for (a,), (e,) in zip(struct.iter_unpack(fmt, actual), struct.iter_unpack(fmt, expected)):
        if not math.isfinite(a) or not math.isfinite(e):
            return dict(result, finite=False, relative_norm=None, max_abs=None, gate_pass=False)
        delta = a - e
        error += delta * delta
        norm += e * e
        maximum = max(maximum, abs(delta))
    relative = math.sqrt(error / norm) if norm else (0.0 if error == 0 else None)
    return dict(result, finite=True, relative_norm=relative, max_abs=maximum,
                gate_pass=relative is not None and relative < LIMIT)


def checked_oracle(path, fixture_sha):
    receipt = headers.read_json(path.with_suffix(".json"))
    headers.require(receipt["schema"] == 2 and
                    {r["m"] for r in receipt["production_routes"]} == {1, 4} and
                    all(r["ordinary_vs_observed_output_bit_exact"] for r in receipt["production_routes"]),
                    "oracle requires actual production intermediates and ordinary-call equivalence")
    headers.require(receipt["image"] == IMAGE and receipt["library_sha256"] == LIB_SHA
                    and receipt["ops_sha256"] == OPS_SHA, "oracle runtime identity mismatch")
    headers.require(receipt["fixture_sha256"] == fixture_sha, "oracle fixture identity mismatch")
    headers.require(digest(path.read_bytes()) == receipt["capture_sha256"], "oracle capture SHA-256 mismatch")
    report = headers.read_shard_header(path)
    headers.require({t["name"] for t in report["tensors"]} == receipt["tensor_hashes"].keys(),
                    "oracle tensor set mismatch")
    for name, spec in receipt["tensor_hashes"].items():
        entry, raw = blob(path, report, name)
        headers.require((entry["dtype"], entry["shape"], digest(raw)) ==
                        (spec["dtype"], spec["shape"], spec["sha256"]), "oracle tensor identity mismatch")
    return receipt, report


def compare(fixture, oracle, native, binary):
    fixture, oracle, native, binary = map(Path, (fixture, oracle, native, binary))
    source = checked_fixture(fixture)
    receipt, expected = checked_oracle(oracle, source["fixture_sha256"])
    actual = headers.read_shard_header(native)
    fixture_header = headers.read_shard_header(fixture)
    rows = []

    def stage(name, native_name, oracle_name, attribution, exact=False):
        ae, a = blob(native, actual, native_name)
        ee, e = blob(oracle, expected, oracle_name)
        headers.require((ae["dtype"], ae["shape"]) == (ee["dtype"], ee["shape"]), "stage dtype/shape mismatch")
        if exact:
            stats = bit_stats(a, e, headers.DTYPE_BYTES[ae["dtype"]])
            stats["gate_pass"] = stats["bit_exact"]
        else:
            stats = metrics(a, e, ae["dtype"])
        row = {"stage": name, "attribution": attribution, "dtype": ae["dtype"], "shape": ae["shape"],
               "native_sha256": digest(a), "oracle_sha256": digest(e), **stats}
        rows.append(row)
        return row

    decode = stage("packed_decode", "decoded_weight_f16", "decoded_weight_f16",
                   "native GPU decoder versus registered producer reconstruction", exact=True)
    cases = []
    for m in (1, 4):
        key = f"m{m}"
        ae, a = blob(native, actual, key + "_activation_f16")
        ee, e = blob(fixture, fixture_header, "activation_" + key + "_fp16")
        headers.require((ae["dtype"], ae["shape"], a) == (ee["dtype"], ee["shape"], e),
                        "native activation differs from fixture")
        ih = stage(key + "_input_hadamard", key + "_input_hadamard_f16", key + "_production_input_hadamard_f16",
                   "same FP16 activation/scales; actual retained producer SmallM input", exact=True)
        raw = stage(key + "_production_gemm_ordered_sum", key + "_native_raw_f32", key + "_production_raw_ordered_sum_f32",
                    "native raw capture versus CPU FP32 increasing-split sum of ACTUAL retained production parts")
        diagnostic = stage(key + "_diagnostic_gemm", key + "_native_raw_f32", key + "_diagnostic_raw_split4_f32",
                    "native increasing-K accumulation versus producer FIXED-SPLIT-4 helper; not production intermediate")
        operand = stage(key + "_output_hadamard_operand", key + "_output_hadamard_operand_f16", key + "_output_hadamard_operand_f16",
                        "native probe input bytes versus recorded oracle operand", exact=True)
        oh = stage(key + "_output_hadamard_probe", key + "_output_hadamard_probe_f16", key + "_output_hadamard_probe_f16",
                   "identical saved FP16 operand and output scales; separate from production FP32-part reduction")
        out = stage(key + "_production_output", key + "_production_output_f16", key + "_production_output_f16",
                    "native auto/packed leaf versus producer normal SmallM with dynamic split-K")
        first = next((s["stage"] for s in (decode, ih, raw, out) if not s["bit_exact"]), None)
        cases.append({"m": m, "first_bit_difference_in_production_stage_replay": first,
                      "local_projection_gate_pass": all(s["gate_pass"] for s in (decode, ih, raw, diagnostic, operand, oh, out)),
                      "production_sum_attribution": "derived ordered sum of original split bytes; not a separately exported binary sum"})
    return {"schema": 1, "kind": "S0b_projection_stage_comparison", "projection": source["projection"],
            "fixture_sha256": source["fixture_sha256"], "oracle_sha256": receipt["capture_sha256"],
            "native_capture_sha256": digest(native.read_bytes()), "native_binary_sha256": digest(binary.read_bytes()),
            "compare_tool_sha256": digest(Path(__file__).read_bytes()),
            "production_routes": receipt["production_routes"],
            "relative_norm_limit_strict": LIMIT, "cases": cases, "stages": rows,
            "scope": "standalone native XPU leaves; not loader, full engine, full head, or model parity",
            "S0_complete": False,
            "remaining": ["actual KV/state layouts and dependent model/state qualification are outside this projection replay"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("fixture", "oracle", "native", "binary", "report"):
        parser.add_argument("--" + name, required=True, type=Path)
    args = parser.parse_args()
    try:
        headers.require(not args.report.exists(), "refusing to overwrite comparison report")
        result = compare(args.fixture, args.oracle, args.native, args.binary)
        with args.report.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print(json.dumps({"report": str(args.report), "cases": result["cases"]}))
        return 0 if all(case["local_projection_gate_pass"] for case in result["cases"]) else 1
    except (headers.InventoryError, OSError, KeyError, ValueError, StopIteration) as exc:
        print(f"compare_projection: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
