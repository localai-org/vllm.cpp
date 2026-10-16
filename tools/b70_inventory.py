#!/usr/bin/env python3
"""PR00: inventory local Qwen EXL3 Safetensors headers without loading weights.

Usage: python3 tools/b70_inventory.py /explicit/model/directory > inventory.json

Reads the index and exactly the length prefix plus JSON header of each indexed
shard. No runtime, accelerator, network, or third-party Python package is used.
By default also validates config.json and quantization_config.json and derives
EXL3 matrix descriptors. --headers-only skips configuration checks.
--verify-checkpoint explicitly enables full shard hashes and scalar marker reads.

Like SafetensorsFile::Open, this accepts padding gaps and reports their bytes.
Unlike the runtime reader, it rejects unknown dtypes: an inventory must validate
every tensor's byte size rather than treating it as an opaque span.
"""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import struct
import subprocess
import sys


MAX_HEADER_BYTES = 64 * 1024 * 1024
MAX_DIM = (1 << 63) - 1
MAX_BYTES = (1 << 64) - 1
DTYPE_BYTES = {
    "F64": 8, "I64": 8, "U64": 8,
    "F32": 4, "I32": 4, "U32": 4,
    "F16": 2, "BF16": 2, "I16": 2, "U16": 2,
    "U8": 1, "I8": 1, "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1,
}
GROUPS = ("text", "mtp", "vision", "embedding", "head", "unclassified")
EXL3_COMPONENTS = {"suh", "svh", "mul1", "trellis"}
MODEL_ID = "Mia-AiLab/Qwen3.8-27B-EXL3-3.5bpw"
MODEL_REVISION = "19441ac874c4018295da848e250f23511361cda4"
SOURCE_REVISION = "9e63db5dd33b35e7cc57d0f0e80fe6c7d5ababa6"
VLLM_REVISION = "e126687a9a828d513c01a07cd69f025f27d63280"
SHARD_HASHES = {
    "model-00001-of-00002.safetensors": "7b77214fe58ff15fed0b4af55e3cd92f38842b8711886d68954e8071ff8270c6",
    "model-00002-of-00002.safetensors": "411c83bb1070b27f3d670fc93e38dca0f17eb66429f64b5706901b12613188b2",
}


class InventoryError(ValueError):
    """Invalid input or inconsistent checkpoint metadata."""


def require(condition, message):
    if not condition:
        raise InventoryError(message)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f"duplicate JSON key: {key}")
        result[key] = value
    return result


def reject_constant(value):
    raise InventoryError(f"invalid JSON constant: {value}")


def json_object(raw, source):
    try:
        result = json.loads(raw, object_pairs_hook=unique_object,
                            parse_constant=reject_constant)
        require(isinstance(result, dict), "JSON root must be an object")
        return result
    except (ValueError, RecursionError) as exc:
        raise InventoryError(f"{source}: {exc}") from exc


def unsigned_integer(value, maximum=MAX_BYTES):
    # bool is an int subclass, but never a valid shape, offset, or byte count.
    return type(value) is int and 0 <= value <= maximum


def classify(name):
    if name == "model.language_model.embed_tokens.weight":
        return "embedding"
    if name.startswith("lm_head."):
        return "head"
    if name.startswith("mtp."):
        return "mtp"
    if name.startswith("model.visual."):
        return "vision"
    if (name.startswith("model.language_model.layers.")
            or name == "model.language_model.norm.weight"):
        return "text"
    return "unclassified"


def read_shard_header(path):
    """Read only metadata; offsets in the result are relative to the data section."""
    path = Path(path)
    try:
        file_bytes = path.stat().st_size
        # Unbuffered reads avoid prefetching tensor bytes beyond the JSON header.
        with path.open("rb", buffering=0) as stream:
            prefix = stream.read(8)
            require(len(prefix) == 8, f"{path}: incomplete 8-byte header prefix")
            header_bytes = struct.unpack("<Q", prefix)[0]
            require(header_bytes <= file_bytes - 8,
                    f"{path}: header length exceeds file size")
            require(header_bytes <= MAX_HEADER_BYTES,
                    f"{path}: header length exceeds {MAX_HEADER_BYTES}-byte limit")
            raw = stream.read(header_bytes)
            require(len(raw) == header_bytes, f"{path}: incomplete JSON header")
    except OSError as exc:
        raise InventoryError(f"{path}: {exc}") from exc

    header = json_object(raw, path)
    data_bytes = file_bytes - 8 - header_bytes
    metadata = header.pop("__metadata__", {})
    require(isinstance(metadata, dict) and all(isinstance(v, str) for v in metadata.values()),
            f"{path}: __metadata__ must be a string-valued object")
    tensors, spans = [], []
    for name, entry in sorted(header.items()):
        label = f"{path.name}: {name}"
        require(isinstance(entry, dict), f"{label}: tensor entry must be an object")
        dtype, shape, offsets = (entry.get(k) for k in ("dtype", "shape", "data_offsets"))
        require(isinstance(dtype, str) and dtype in DTYPE_BYTES,
                f"{label}: unknown or invalid dtype {dtype!r}")
        require(isinstance(shape, list) and all(unsigned_integer(d, MAX_DIM) for d in shape),
                f"{label}: invalid shape {shape!r}")
        require(isinstance(offsets, list) and len(offsets) == 2
                and all(unsigned_integer(o) for o in offsets)
                and offsets[0] <= offsets[1] <= data_bytes,
                f"{label}: invalid data_offsets {offsets!r} for {data_bytes} data bytes")
        numel = 1
        for dim in shape:
            numel *= dim
            require(numel <= MAX_BYTES, f"{label}: shape element count overflows")
        expected_bytes = numel * DTYPE_BYTES[dtype]
        require(expected_bytes <= MAX_BYTES, f"{label}: tensor byte size overflows")
        begin, end = offsets
        require(end - begin == expected_bytes,
                f"{label}: byte size {end - begin} disagrees with {dtype} {shape} ({expected_bytes})")
        spans.append((begin, end, name))
        suffix = name.rsplit(".", 1)[-1]
        tensors.append({
            "name": name, "shard": path.name, "dtype": dtype, "shape": shape,
            "data_offsets": offsets, "stored_bytes": expected_bytes,
            "group": classify(name),
            "exl3_component": suffix if suffix in EXL3_COMPONENTS else None,
        })

    previous_end, previous_name = 0, None
    for begin, end, name in sorted(spans):
        require(begin >= previous_end,
                f"{path.name}: {name} overlaps {previous_name}")
        previous_end, previous_name = end, name
    stored_bytes = sum(t["stored_bytes"] for t in tensors)
    return {
        "name": path.name, "file_bytes": file_bytes, "header_bytes": header_bytes,
        "header_sha256": hashlib.sha256(raw).hexdigest(), "metadata": metadata,
        "data_bytes": data_bytes, "stored_bytes": stored_bytes,
        "unclaimed_data_bytes": data_bytes - stored_bytes, "tensor_count": len(tensors),
        "tensors": tensors,
    }


def inventory(model_dir):
    model_dir = Path(model_dir)
    index_path = model_dir / "model.safetensors.index.json"
    try:
        raw_index = index_path.read_bytes()
    except OSError as exc:
        raise InventoryError(f"{index_path}: {exc}") from exc
    index = json_object(raw_index, index_path)
    weight_map, metadata = index.get("weight_map"), index.get("metadata")
    require(isinstance(weight_map, dict) and weight_map,
            f"{index_path}: weight_map must be a nonempty object")
    require(isinstance(metadata, dict) and unsigned_integer(metadata.get("total_size")),
            f"{index_path}: metadata.total_size must be a nonnegative integer")
    for name, shard in weight_map.items():
        require(name != "__metadata__", f"{index_path}: reserved tensor name {name}")
        require(isinstance(shard, str) and shard.endswith(".safetensors")
                and "/" not in shard and "\\" not in shard,
                f"{index_path}: invalid shard name {shard!r} for {name}")

    shards, tensors = [], {}
    for shard in sorted(set(weight_map.values())):
        record = read_shard_header(model_dir / shard)
        for tensor in record.pop("tensors"):
            name = tensor["name"]
            require(name not in tensors, f"tensor {name} occurs in multiple shards")
            tensors[name] = tensor
        shards.append(record)
    missing = sorted(weight_map.keys() - tensors.keys())
    extra = sorted(tensors.keys() - weight_map.keys())
    require(not missing and not extra,
            f"index/header names disagree: missing headers={missing}; absent from index={extra}")
    for name, tensor in sorted(tensors.items()):
        require(tensor["shard"] == weight_map[name],
                f"wrong shard for {name}: index={weight_map[name]}, header={tensor['shard']}")
    rows = [tensors[name] for name in sorted(tensors)]
    stored_bytes = sum(t["stored_bytes"] for t in rows)
    require(stored_bytes == metadata["total_size"],
            f"index total_size={metadata['total_size']} disagrees with header bytes={stored_bytes}")

    groups = {}
    for group in GROUPS:
        members = [t for t in rows if t["group"] == group]
        groups[group] = {
            "tensor_count": len(members),
            "stored_bytes": sum(t["stored_bytes"] for t in members),
            "dtypes": dict(sorted(Counter(t["dtype"] for t in members).items())),
        }
    return {
        "schema_version": 1, "inspection": "headers_only",
        "payload_hashes_verified": False,
        "index_sha256": hashlib.sha256(raw_index).hexdigest(),
        "tensor_count": len(rows), "stored_bytes": stored_bytes,
        "groups": groups, "shards": shards, "tensors": rows,
        "errors": [f"unclassified tensor: {t['name']}" for t in rows
                   if t["group"] == "unclassified"],
    }


def read_json(path):
    try:
        return json_object(path.read_bytes(), path)
    except OSError as exc:
        raise InventoryError(f"{path}: {exc}") from exc


def checkpoint_inventory(model_dir):
    """Validate EXL3 metadata and derive dimensions from real headers, not estimates."""
    model_dir = Path(model_dir)
    report = inventory(model_dir)
    config = read_json(model_dir / "config.json")
    quant = read_json(model_dir / "quantization_config.json")
    require(quant.get("quant_method") == "exl3" and quant.get("codebook") == "mul1",
            "expected EXL3 mul1 quantization_config")
    embedded = config.get("quantization_config")
    require(isinstance(embedded, dict), "config.quantization_config is missing")
    for key, value in embedded.items():
        require(key in quant and quant[key] == value,
                f"config/quantization_config disagreement: {key}")
    storage = quant.get("tensor_storage")
    require(isinstance(storage, dict), "quantization_config.tensor_storage must be an object")
    tensors = {t["name"]: t for t in report["tensors"]}
    declared = set()
    torch_dtypes = {"torch.float16": "F16", "torch.bfloat16": "BF16",
                    "torch.float32": "F32", "torch.int16": "I16", "torch.int32": "I32"}
    for module, entry in storage.items():
        require(isinstance(entry, dict) and isinstance(entry.get("stored_tensors"), dict),
                f"invalid tensor_storage entry: {module}")
        for name, expected in entry["stored_tensors"].items():
            require(name in tensors and name not in declared,
                    f"missing or duplicate tensor_storage name: {name}")
            require(isinstance(expected, dict) and isinstance(expected.get("dtype"), str),
                    f"invalid tensor_storage descriptor: {name}")
            actual = tensors[name]
            require(expected.get("shape") == actual["shape"]
                    and expected.get("n_bytes") == actual["stored_bytes"]
                    and torch_dtypes.get(expected.get("dtype")) == actual["dtype"],
                    f"tensor_storage/header disagreement: {name}")
            declared.add(name)
    matrices = []
    prefixes = sorted({t["name"].rsplit(".", 1)[0] for t in tensors.values()
                       if t["exl3_component"] is not None})
    for prefix in prefixes:
        components = {c: tensors.get(f"{prefix}.{c}") for c in sorted(EXL3_COMPONENTS)}
        require(all(components.values()), f"incomplete EXL3 components: {prefix}")
        trellis, suh, svh, marker = (components[c] for c in ("trellis", "suh", "svh", "mul1"))
        shape = trellis["shape"]
        require(trellis["dtype"] == "I16" and len(shape) == 3
                and shape[0] > 0 and shape[1] > 0 and shape[2] in range(16, 129, 16),
                f"invalid EXL3 trellis: {prefix}")
        k, n, bits = shape[0] * 16, shape[1] * 16, shape[2] // 16
        require(k % 128 == 0 and n % 128 == 0, f"EXL3 Had128 dimensions: {prefix}")
        require(suh["dtype"] == svh["dtype"] == "F16"
                and suh["shape"] == [k] and svh["shape"] == [n],
                f"EXL3 scale shape/dtype mismatch: {prefix}")
        require(marker["dtype"] == "I32" and marker["shape"] == [],
                f"EXL3 mul1 marker must be an I32 scalar: {prefix}")
        if prefix in storage:
            require(storage[prefix].get("quant_format") == "exl3"
                    and storage[prefix].get("bits_per_weight") == bits
                    and storage[prefix].get("mul1_multiplier") == 2212286765,
                    f"EXL3 quantization metadata mismatch: {prefix}")
        matrices.append({"name": prefix, "group": trellis["group"], "k": k, "n": n,
                         "bits": bits, "stored_bytes": sum(c["stored_bytes"] for c in components.values()),
                         "components": {c: t["name"] for c, t in components.items()}})
    for name, entry in storage.items():
        if entry.get("quant_format") == "exl3":
            require(name in prefixes, f"EXL3 metadata without matrix: {name}")
    text_matrices = [m for m in matrices if m["group"] in ("text", "head")]
    unlisted = [t for t in report["tensors"] if t["name"] not in declared]
    # A loader plan is data only. MTP and vision are never staged by this tool.
    text_groups = ("text", "embedding", "head")
    report.update({
        "model_config": config,
        "quantization_config": {k: v for k, v in quant.items() if k != "tensor_storage"},
        "matrices": matrices,
        "text_matrix_bit_counts": dict(sorted(Counter(str(m["bits"]) for m in text_matrices).items())),
        "tensors_absent_from_quantization_metadata": [t["name"] for t in unlisted],
        "loader_plan": {"groups": list(text_groups), "mtp": False, "vision": False,
                        "stored_bytes": sum(report["groups"][g]["stored_bytes"] for g in text_groups)},
        "plan_comparison": {"expected_text_head_matrices": 401, "actual_text_head_matrices": len(text_matrices),
                            "expected_unlisted_tensors": 516, "actual_unlisted_tensors": len(unlisted)},
        "pins": {"model": MODEL_ID, "model_revision": MODEL_REVISION,
                 "vllm_cpp_target": SOURCE_REVISION, "vllm_reference": VLLM_REVISION},
        "remaining_baseline_work": ["Technical report [Lxx] source index not bundled with the plan.",
                                    "Donor wheel/source numerical parity is not established by header inspection."],
    })
    report["findings"] = []
    if len(text_matrices) != 401:
        report["findings"].append(f"plan expected 401 text/head matrices; observed {len(text_matrices)}")
    if len(unlisted) != 516:
        report["findings"].append(f"plan expected 516 unlisted tensors; observed {len(unlisted)}")
    return report


def verify_checkpoint(model_dir, report):
    """Explicit opt-in full-file SHA256 pass, separate from header-only inventory."""
    require({s["name"] for s in report["shards"]} == SHARD_HASHES.keys(), "pinned shard names differ")
    model_dir = Path(model_dir)
    try:
        provenance = (model_dir / "SOURCE-REVISION.txt").read_text()
        require(f"Repository: {MODEL_ID}" in provenance and f"Revision: {MODEL_REVISION}" in provenance,
                "local checkpoint provenance does not match model pin")
        for shard in report["shards"]:
            with (model_dir / shard["name"]).open("rb") as stream:
                digest = hashlib.file_digest(stream, "sha256").hexdigest()
            require(digest == SHARD_HASHES[shard["name"]], f"pinned SHA256 mismatch: {shard['name']}")
            shard["sha256"] = digest
        shards = {s["name"]: s for s in report["shards"]}
        for tensor in report["tensors"]:
            if tensor["exl3_component"] != "mul1":
                continue
            shard = shards[tensor["shard"]]
            with (model_dir / tensor["shard"]).open("rb") as stream:
                stream.seek(8 + shard["header_bytes"] + tensor["data_offsets"][0])
                marker = struct.unpack("<I", stream.read(4))[0]
            # The loader uses the unsigned bit pattern of the stored I32 scalar.
            require(marker == 2212286765, f"unexpected mul1 marker: {tensor['name']}: {marker}")
    except (OSError, struct.error) as exc:
        raise InventoryError(str(exc)) from exc
    report["payload_hashes_verified"] = True
    report["mul1_markers_verified"] = sum(t["exl3_component"] == "mul1" for t in report["tensors"])
    report["inspection"] = "headers_and_explicit_payload_verification"


def source_state(path, revision):
    if path is None:
        return {"expected_revision": revision, "status": "PENDING", "reason": "source path not supplied"}
    try:
        def git(*args):
            return subprocess.check_output(["git", "-C", str(path), *args], text=True,
                                           stderr=subprocess.DEVNULL).strip()
        actual = git("rev-parse", "HEAD")
        git("cat-file", "-e", revision + "^{commit}")
        return {"expected_revision": revision, "actual_revision": actual,
                "pinned_commit_available": True, "dirty": bool(git("status", "--porcelain")),
                "status": "MATCH" if actual == revision else "DIFFERENT_CHECKOUT"}
    except (OSError, subprocess.CalledProcessError) as exc:
        return {"expected_revision": revision, "status": "PENDING", "reason": str(exc)}


def add_source_pins(report, vllm_source=None, donor_lock=None):
    root = Path(__file__).resolve().parents[1]
    sources = {"vllm_cpp": source_state(root, SOURCE_REVISION),
               "vllm": source_state(vllm_source, VLLM_REVISION)}
    # Hash the exact local implementation too: HEAD alone cannot describe an
    # uncommitted development step. The target checkout is never reset/modified.
    sources["local_implementation_sha256"] = {
        name: hashlib.sha256((root / name).read_bytes()).hexdigest() for name in (
            "tools/b70_inventory.py", "src/vt/op_provider.cpp",
            "src/vllm/model_executor/model_loader/safetensors_reader.cpp")}
    if donor_lock:
        lock = read_json(donor_lock)
        sources["donor_lock_sha256"] = hashlib.sha256(donor_lock.read_bytes()).hexdigest()
        for key in ("vllm_xpu_kernels", "sycl_tla"):
            entry = lock.get(key, {})
            require(isinstance(entry.get("commit"), str), f"missing donor revision: {key}")
            sources[key] = source_state(entry.get("default_host_path"), entry["commit"])
            sources[key]["repository"] = entry.get("repository")
        sources["donor_runtime_image"] = lock.get("runtime_image", {})
    else:
        sources["donors"] = {"status": "PENDING", "reason": "pass --donor-lock sources.lock.json"}
    report["sources"] = sources


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dir", type=Path, help="explicit local checkpoint directory")
    parser.add_argument("--headers-only", action="store_true", help="skip EXL3 configuration checks")
    parser.add_argument("--verify-checkpoint", action="store_true", help="also hash full shards and read mul1 scalars")
    parser.add_argument("--microbench-output", type=Path, help="write separate text/head matrix configuration")
    parser.add_argument("--vllm-source", type=Path, help="local pinned vLLM checkout")
    parser.add_argument("--donor-lock", type=Path, help="local B70 donor sources.lock.json")
    args = parser.parse_args(argv)
    try:
        report = inventory(args.model_dir) if args.headers_only else checkpoint_inventory(args.model_dir)
        add_source_pins(report, args.vllm_source, args.donor_lock)
        if args.verify_checkpoint:
            verify_checkpoint(args.model_dir, report)
        if args.microbench_output:
            require(not args.headers_only, "microbench output requires EXL3 configuration checks")
            require(not args.microbench_output.resolve().is_relative_to(args.model_dir.resolve()),
                    "microbench output must be outside the read-only checkpoint directory")
            microbench = {"schema_version": 1, "m": [1, 2, 4, 8, 16, 128], "input_dtype": "F16",
                          "output_dtypes": ["F16", "F32"], "executes_kernels": False,
                          "matrices": [m for m in report["matrices"] if m["group"] in ("text", "head")]}
            args.microbench_output.write_text(json.dumps(microbench, indent=2, sort_keys=True) + "\n")
    except (InventoryError, OSError) as exc:
        print(f"b70_inventory: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    for error in report["errors"]:
        print(f"b70_inventory: {error}", file=sys.stderr)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    sys.exit(main())
