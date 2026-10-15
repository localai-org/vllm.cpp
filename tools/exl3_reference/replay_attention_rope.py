#!/usr/bin/env python3
"""Independent host replay of captured original FP16 RoPE boundaries."""
import argparse
import json
from pathlib import Path
import struct

from compare_projection import blob, metrics
from extract_projection import digest, headers


def half(value):
    return struct.unpack("<e", struct.pack("<e", value))[0]


def rope_half(values, coefficients, rows, heads, dim, rotary):
    headers.require(rows > 0 and heads > 0 and 0 < rotary <= dim and rotary % 2 == 0 and
                    len(values) == rows * heads * dim and len(coefficients) == rows * rotary,
                    "invalid bounded RoPE operand geometry")
    result = list(values)
    width = rotary // 2
    for row in range(rows):
        for head in range(heads):
            base = (row * heads + head) * dim
            for pair in range(width):
                first, second = values[base + pair], values[base + width + pair]
                c, s = coefficients[row * rotary + pair], coefficients[row * rotary + width + pair]
                result[base + pair] = half(half(first * c) - half(second * s))
                result[base + width + pair] = half(half(first * s) + half(second * c))
    return result


def replay(path):
    record = json.loads(path.with_suffix(".json").read_text())
    headers.require(digest(path.read_bytes()) == record["capture_sha256"], "capture hash mismatch")
    header = headers.read_shard_header(path)
    results = {}
    for phase, rows in (("p128", 128), ("d1", 1)):
        coeff_entry, raw_coeff = blob(path, header, phase + "_rope_cos_sin")
        headers.require(coeff_entry["dtype"] == "F16" and coeff_entry["shape"] == [rows, 64],
                        "requires actual original FP16 coefficients")
        coefficients = [v for v, in struct.iter_unpack("<e", raw_coeff)]
        for part, heads in (("q", 24), ("k", 4)):
            entry, raw = blob(path, header, phase + "_rope_" + part + "_input")
            headers.require(entry["dtype"] == "F16" and entry["shape"] == [rows, heads * 256],
                            "requires actual original normalized FP16 inputs")
            values = [v for v, in struct.iter_unpack("<e", raw)]
            output = rope_half(values, coefficients, rows, heads, 256, 64)
            packed = b"".join(struct.pack("<e", v) for v in output)
            _, expected = blob(path, header, phase + "_" + part + "_rope")
            results[phase + "_" + part] = metrics(packed, expected, "F16")
    return {"scope": "Original captured normalized inputs and coefficients; independent half-operation replay, not native parity.",
            "capture_sha256": record["capture_sha256"], "stages": results,
            "all_bit_exact": all(v["bit_exact"] for v in results.values())}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = replay(args.capture)
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print(json.dumps(result, indent=2))
    raise SystemExit(0 if result["all_bit_exact"] else 1)
