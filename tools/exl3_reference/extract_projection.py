#!/usr/bin/env python3
"""Extract packed EXL3 projection fixtures on the host; never run model math.

Uses the explicit S0 reference manifest, not the legacy inventory identity pins.
Column slices must preserve whole 128-column output Hadamard blocks. Synthetic
FP16 inputs at M=1/4 accompany the real weights; oracle outputs remain pending.
"""

import argparse
import hashlib
import json
from pathlib import Path
import struct
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import b70_inventory as headers


def digest(raw):
    return hashlib.sha256(raw).hexdigest()


def read_span(model_dir, shard, begin, size):
    with (model_dir / shard["name"]).open("rb", buffering=0) as stream:
        stream.seek(8 + shard["header_bytes"] + begin)
        raw = stream.read(size)
    headers.require(len(raw) == size, f"short tensor read in {shard['name']}")
    return raw


def extract_projection(model_dir, report, prefix, column_start, column_count):
    """Copy bit patterns per input tile, without expanding or reading a full head."""
    model_dir = Path(model_dir)
    tensors = {t["name"]: t for t in report["tensors"]}
    shards = {s["name"]: s for s in report["shards"]}
    parts = {p: tensors.get(prefix + "." + p) for p in ("trellis", "suh", "svh", "mul1")}
    headers.require(all(parts.values()), f"incomplete EXL3 projection: {prefix}")
    tr, suh, svh, marker = (parts[p] for p in ("trellis", "suh", "svh", "mul1"))
    shape = tr["shape"]
    headers.require(tr["dtype"] == "I16" and len(shape) == 3
                    and shape[2] in (64, 96), "expected 4/6-bpw I16 trellis")
    k, n, bits = shape[0] * 16, shape[1] * 16, shape[2] // 16
    headers.require(k > 0 and n > 0 and k % 128 == n % 128 == 0,
                    "projection dimensions must be positive multiples of 128")
    headers.require(type(column_start) is int and type(column_count) is int
                    and column_start >= 0 and column_count > 0
                    and column_start % 128 == column_count % 128 == 0
                    and column_start + column_count <= n,
                    "column slice must contain whole in-range Hadamard128 blocks")
    headers.require(suh["dtype"] == svh["dtype"] == "F16"
                    and suh["shape"] == [k] and svh["shape"] == [n],
                    "invalid EXL3 scale shape/dtype")
    headers.require(marker["dtype"] == "I32" and marker["shape"] == [],
                    "mul1 must be an I32 scalar")
    mul1 = read_span(model_dir, shards[marker["shard"]], marker["data_offsets"][0], 4)
    headers.require(struct.unpack("<I", mul1)[0] == 2212286765, "unexpected mul1 marker bits")
    # Trellis is tiled [K/16,N/16,16*bits] I16. A column block is contiguous
    # within each K tile; flattening and taking one global span is incorrect.
    tile_bytes = 32 * bits
    row_bytes = column_count // 16 * tile_bytes
    packed = bytearray()
    for tile in range(k // 16):
        begin = tr["data_offsets"][0] + (tile * (n // 16) + column_start // 16) * tile_bytes
        packed.extend(read_span(model_dir, shards[tr["shard"]], begin, row_bytes))
    out = {
        "trellis": ("I16", [k // 16, column_count // 16, 16 * bits], bytes(packed)),
        "suh": ("F16", [k], read_span(model_dir, shards[suh["shard"]], suh["data_offsets"][0], 2 * k)),
        "svh": ("F16", [column_count], read_span(model_dir, shards[svh["shard"]],
                                               svh["data_offsets"][0] + 2 * column_start, 2 * column_count)),
        "mul1": ("I32", [], mul1),
    }
    for m in (1, 4):
        # Exact dyadic values; independent of Python/NumPy RNG implementation.
        raw = b"".join(struct.pack("<e", ((i * 37 + row * 101) % 511 - 255) / 256)
                       for row in range(m) for i in range(k))
        out[f"activation_m{m}_fp16"] = ("F16", [m, k], raw)
    return out, {"prefix": prefix, "source_k": k, "source_n": n,
                 "bits": bits, "codebook": "mul1", "column_start": column_start,
                 "column_count": column_count, "source_components": parts}


def write_safetensors(path, tensors, metadata):
    """Write a single exclusive fixture, retaining source I16/FP16 bit patterns."""
    offset = 0
    header = {"__metadata__": metadata}
    for name, (dtype, shape, raw) in sorted(tensors.items()):
        expected = headers.DTYPE_BYTES[dtype]
        for dim in shape:
            expected *= dim
        headers.require(len(raw) == expected, f"fixture byte size mismatch: {name}")
        header[name] = {"dtype": dtype, "shape": shape,
                        "data_offsets": [offset, offset + len(raw)]}
        offset += len(raw)
    raw_header = json.dumps(header, separators=(",", ":")).encode()
    raw_header += b" " * (-len(raw_header) % 8)
    with Path(path).open("xb") as stream:
        stream.write(struct.pack("<Q", len(raw_header)))
        stream.write(raw_header)
        for _, _, raw in (tensors[n] for n in sorted(tensors)):
            stream.write(raw)


def export(model_dir, reference_path, prefix, column_start, column_count, output):
    model_dir, output = Path(model_dir).resolve(), Path(output).resolve()
    sidecar = output.with_suffix(".json")
    headers.require(output != sidecar, "fixture output must have a non-JSON suffix")
    headers.require(not output.is_relative_to(model_dir), "fixture must be outside checkpoint directory")
    headers.require(not output.exists() and not sidecar.exists(), "refusing to overwrite fixture or sidecar")
    reference = headers.read_json(Path(reference_path))["reference_B"]["checkpoint"]
    identity = reference["identity"]
    for entry in reference["metadata_files"]:
        path = model_dir / Path(entry["path"]).name
        headers.require(digest(path.read_bytes()) == entry["sha256"], f"reference metadata hash mismatch: {path.name}")
    report = headers.inventory(model_dir)
    headers.require(not report["errors"], "invalid checkpoint tensor classification")
    expected_headers = {s["name"]: s["header_sha256"] for s in reference["header_inventory"]["shards"]}
    headers.require({s["name"]: s["header_sha256"] for s in report["shards"]} == expected_headers,
                    "reference shard header hash mismatch")
    for shard in report["shards"]:
        provenance = model_dir / ".cache/huggingface/download" / (shard["name"] + ".metadata")
        headers.require(provenance.read_text().splitlines()[0] == identity["revision"],
                        f"checkpoint revision mismatch: {shard['name']}")
    tensors, projection = extract_projection(model_dir, report, prefix, column_start, column_count)
    source_sha = digest(Path(__file__).read_bytes())
    result = {"schema": 1, "kind": "packed_projection_with_synthetic_inputs",
              "checkpoint": {"model": identity["model"], "revision": identity["revision"]},
              "projection": projection, "exporter_sha256": source_sha,
              "input_recipe": "F16(((i*37 + row*101) % 511 - 255)/256), M=1/4",
              "tensor_hashes": {n: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                for n, t in sorted(tensors.items())},
              "reference_manifest_sha256": digest(Path(reference_path).read_bytes()),
              "oracle_outputs": {"status": "pending", "reason": "No arithmetic executed or producer output captured."},
              "full_checkpoint_payload_verified": False}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_safetensors(output, tensors, {"checkpoint_revision": identity["revision"], "prefix": prefix})
    result["fixture_sha256"] = digest(output.read_bytes())
    result["fixture_file"] = str(output)
    with sidecar.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--prefix", required=True)
    parser.add_argument("--column-start", type=int, default=0)
    parser.add_argument("--column-count", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        result = export(args.model_dir, args.reference_manifest, args.prefix,
                        args.column_start, args.column_count, args.output)
    except (headers.InventoryError, OSError, KeyError, IndexError) as exc:
        print(f"extract_projection: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({"fixture": result["fixture_file"], "sha256": result["fixture_sha256"],
                      "projection": {k: v for k, v in result["projection"].items() if k != "source_components"},
                      "oracle_outputs": result["oracle_outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
