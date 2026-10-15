#!/usr/bin/env python3
"""Executed original C1 verifier family on derived, explicitly named operands.

No model execution or new worker trajectory: source Q/K/V bytes come from the
frozen page1600/Q4 worker capture. Longer KV repeats its initialized rows.
"""
import argparse
import json
import os
from pathlib import Path

from capture_runtime_layout import PROFILE, verify_inputs
from compare_projection import blob
from extract_projection import digest, headers, write_safetensors

SOURCE_SHA = "5b203e2397f7db9c87a2fba041eac31975d96d13d3287244c09186aaedab2a38"
VERIFY_SHA = "efb06f59250cbf4efb16f3eb6eb119bbd4c24c9ff086459b4afb85cc21adacb2"
PAGE, BLOCKS, BASE_ROWS = 1600, 29, 4100
BLOCK_IDS = [3, 10, 23] + [i for i in range(BLOCKS) if i not in (3, 10, 23)]
LAYOUTS = {
    "planar": {"strides": [1600 * 1024, 1024, 256, 1],
               "v_offset": BLOCKS * 1600 * 1024, "bytes": 2 * BLOCKS * 1600 * 1024},
    "interleaved": {"strides": [1600 * 2048, 2048, 512, 1],
                    "v_offset": 256, "bytes": BLOCKS * 1600 * 2048},
    "padded_interleaved": {"strides": [1664 * 2048, 2048, 512, 1],
                           "v_offset": 256, "bytes": BLOCKS * 1664 * 2048},
}


def family_cases():
    cases = [(4, length, False) for length in
             (4096, 4100, 1599, 1600, 1601, 4799, 4800, 4801, 32768)]
    cases += [(rows, 4100, False) for rows in (2, 3, 5)]
    cases += [(4, 4100, True)]
    return [{"id": f"q{q}-l{length}-{layout}" + ("-tail-values" if mutate else ""),
             "queries": q, "length": length, "layout": layout,
             "mutate_tail_values": mutate}
            for q, length, mutate in cases for layout in LAYOUTS]


def query_row_ids(rows):
    return {2: [1, 3], 3: [3, 0, 2], 4: [0, 1, 2, 3], 5: [2, 0, 3, 1, 2]}[rows]


def cache_bytes(keys, values, length, layout, poison=0, mutate=False):
    if len(keys) != BASE_ROWS * 1024 or len(values) != len(keys):
        raise ValueError("requires frozen initialized4100-row K/V")
    if length not in {1599, 1600, 1601, 4096, 4100, 4799, 4800, 4801, 32768}:
        raise ValueError("unsupported bounded active length")
    spec = LAYOUTS[layout]
    page_stride, row_stride, head_stride, _ = spec["strides"]
    storage = bytearray([poison]) * spec["bytes"]
    for row in range(length):
        block = BLOCK_IDS[row // PAGE]
        for head in range(4):
            src = ((row % BASE_ROWS) * 4 + head) * 256
            dst = block * page_stride + (row % PAGE) * row_stride + head * head_stride
            storage[dst:dst + 256] = keys[src:src + 256]
            begin = dst + spec["v_offset"]
            storage[begin:begin + 256] = (bytes([0x38]) * 256 if mutate and row >= length - 3
                                        else values[src:src + 256])
    return storage


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix(".json").exists(),
                    "refusing existing family capture")
    headers.require(digest(args.source.read_bytes()) == SOURCE_SHA, "source worker fixture changed")
    reference = verify_inputs(args.reference_manifest, args.model, args.image_identity)
    index = headers.read_shard_header(args.source)
    base_query = blob(args.source, index, "query")[1]
    keys = b"".join(blob(args.source, index, f"k_page{i}")[1] for i in range(3))
    values = b"".join(blob(args.source, index, f"v_page{i}")[1] for i in range(3))
    expected_worker = blob(args.source, index, "output")[1]
    headers.require(len(base_query) == 4 * 24 * 256 * 2, "wrong worker query")
    import yaml
    for key, value in yaml.safe_load(PROFILE.read_text())["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    import torch
    from exl3xpu import shared_kv_verify as verify
    from torch.utils._python_dispatch import TorchDispatchMode
    library = Path(os.environ["EXL3_M04_LIBRARY"])
    headers.require(digest(library.read_bytes()) == os.environ["EXL3_M04_LIBRARY_SHA256"],
                    "pinned M04 binary changed")
    headers.require(digest(Path(verify.__file__).read_bytes()) == VERIFY_SHA,
                    "pinned verifier wrapper changed")
    torch.ops.load_library(str(library))
    tensors = {"base_keys": ("U8", [BASE_ROWS, 4, 256], keys),
               "base_values": ("U8", [BASE_ROWS, 4, 256], values)}
    records, canonical = [], {}
    for number, case in enumerate(family_cases()):
        rows, length = case["queries"], case["length"]
        spec = LAYOUTS[case["layout"]]
        row_ids = query_row_ids(rows)
        query_raw = b"".join(base_query[i * 12288:(i + 1) * 12288] for i in row_ids)
        host = cache_bytes(keys, values, length, case["layout"], mutate=case["mutate_tail_values"])
        storage = torch.frombuffer(host, dtype=torch.uint8).to("xpu")
        shape = (BLOCKS, PAGE, 4, 256)
        key = storage.as_strided(shape, spec["strides"]).view(torch.float8_e4m3fn)
        value = storage.as_strided(shape, spec["strides"], spec["v_offset"]).view(torch.float8_e4m3fn)
        query = torch.frombuffer(bytearray(query_raw), dtype=torch.float16).reshape(rows, 24, 256).to("xpu")
        table = BLOCK_IDS[:(length + PAGE - 1) // PAGE]
        table += [-1] * (164 - len(table))
        descale = torch.ones(1, dtype=torch.float32, device="xpu")
        data = {"q": query, "k": key, "v": value,
                "out": torch.empty_like(query), "max_seqlen_q": rows,
                "max_seqlen_k": length, "softmax_scale": 0.0625, "causal": True,
                "cu_seqlens_q": torch.tensor([0, rows], dtype=torch.int32, device="xpu"),
                "seqused_k": torch.tensor([length], dtype=torch.int32, device="xpu"),
                "block_table": torch.tensor([table], dtype=torch.int32, device="xpu"),
                "k_descale": descale, "v_descale": descale}
        headers.require(verify.eligible(data), "actual wrapper declined derived case")
        calls = []

        class Witness(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if str(func) == "b70_exl3_attention.shared_kv_verify_out.default":
                    calls.append({"operator": str(func), "max_keys": args[-3],
                                  "splits": args[-2], "tile": args[-1]})
                return func(*args, **(kwargs or {}))

        with Witness():
            result = verify.run(data)
        torch.xpu.synchronize()
        raw = result.cpu().contiguous().numpy().tobytes()
        headers.require(len(calls) == 1, "executed original witness missing")
        headers.require(torch.isfinite(result).all().item(), "nonfinite original output")
        repeated = verify.run(data).cpu().contiguous().numpy().tobytes()
        headers.require(raw == repeated, "ordinary repeat differs from observed original")
        identity = (rows, length, case["mutate_tail_values"])
        headers.require(raw == canonical.setdefault(identity, raw), "original differs across layouts")
        if rows == 4 and length == 4100 and not case["mutate_tail_values"]:
            headers.require(raw == expected_worker, "base operator differs from captured worker")
        if case["mutate_tail_values"]:
            ordinary = canonical[(4, 4100, False)]
            headers.require(raw[:12288] == ordinary[:12288], "causal row0 sees last3 values")
            headers.require(raw[12288:] != ordinary[12288:], "tail perturbation has no causal witness")
        prefix = f"case{number}"
        tensors[prefix + "_query"] = ("F16", [rows, 24, 256], query_raw)
        tensors[prefix + "_output"] = ("F16", [rows, 24, 256], raw)
        records.append(case | {"prefix": prefix, "query_row_ids": row_ids,
                               "KV_strides": spec["strides"], "V_offset": spec["v_offset"],
                               "storage_bytes": spec["bytes"], "route": calls[0],
                               "ordinary_observed_repeat_exact": True})
        print("ORIGINAL_VERIFY_FAMILY_CASE_PASS", case["id"], flush=True)
        del data, result, query, key, value, storage, host, descale
    memory = {"allocated_bytes": torch.xpu.memory_allocated(),
              "reserved_bytes": torch.xpu.memory_reserved(),
              "peak_allocated_bytes": torch.xpu.max_memory_allocated(),
              "peak_reserved_bytes": torch.xpu.max_memory_reserved()}
    write_safetensors(args.output, tensors, {"scope": __doc__})
    report = {"schema": "b70-exl3-P1-C1-verifier-family-v1", "cases": records,
              "source_sha256": SOURCE_SHA, "image": args.image_identity,
              "checkpoint": reference["checkpoint"]["identity"],
              "wrapper_sha256": VERIFY_SHA, "library_sha256": digest(library.read_bytes()),
              "tool_sha256": digest(Path(__file__).read_bytes()), "memory": memory,
              "capture_sha256": digest(args.output.read_bytes()), "physical_blocks": BLOCKS,
              "block_ids": BLOCK_IDS, "page": PAGE, "base_rows": BASE_ROWS,
              "KV_recipe": "row modulo4100; byte-exact initialized worker rows. Q rows explicitly reordered/repeated.",
              "inactive_original_storage": "zero-filled; native zero and NaN poison checked separately",
              "scope": "Executed original isolated operators; not fresh long-context model trajectories, state or serving proof."}
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference-manifest", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
