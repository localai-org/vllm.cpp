#!/usr/bin/env python3
"""Executed original C4 verifier on distinct derived worker-byte operands.

Isolated operator proof, not a new four-request model trajectory. Each request
uses disjoint pages and explicitly rotated initialized rows from the frozen
C1 worker capture. Original inactive storage is zero; native poison is separate.
"""
import argparse
import json
import os
from pathlib import Path

from capture_runtime_layout import PROFILE, verify_inputs
from capture_verify_family import BASE_ROWS, PAGE, SOURCE_SHA, VERIFY_SHA
from compare_projection import blob
from extract_projection import digest, headers, write_safetensors

BLOCKS, COLUMNS = 19, 164
LENGTHS = [1599, 1601, 4100, 4801]
REQUEST_BLOCKS = [[7], [12, 3], [17, 0, 9], [15, 4, 11, 2]]
LAYOUTS = {
    "planar": {"strides": [PAGE * 1024, 1024, 256, 1],
               "v_offset": BLOCKS * PAGE * 1024, "bytes": 2 * BLOCKS * PAGE * 1024},
    "interleaved": {"strides": [PAGE * 2048, 2048, 512, 1],
                    "v_offset": 256, "bytes": BLOCKS * PAGE * 2048},
    "padded_interleaved": {"strides": [1664 * 2048, 2048, 512, 1],
                           "v_offset": 256, "bytes": BLOCKS * 1664 * 2048},
}


def batch_cases():
    return [{"id": f"c4-q4-{layout}-{name}", "layout": layout,
             "request_ids": order, "queries_per_request": 4,
             "lengths": [LENGTHS[r] for r in order], "mutate_request": mutate}
            for layout in LAYOUTS
            for name, order, mutate in (
                ("ordinary", [0, 1, 2, 3], None),
                ("permuted", [2, 0, 3, 1], None),
                ("tail-values-r2", [0, 1, 2, 3], 2))]


def query_bytes(base_query, request):
    if len(base_query) != 4 * 12288 or request not in range(4):
        raise ValueError("requires frozen Q4 and request0..3")
    return b"".join(base_query[((row + request) % 4) * 12288:
                              ((row + request) % 4 + 1) * 12288] for row in range(4))


def table_rows(order):
    if sorted(order) != list(range(4)):
        raise ValueError("requires a permutation of four request IDs")
    return [REQUEST_BLOCKS[r] + [-1] * (COLUMNS - len(REQUEST_BLOCKS[r])) for r in order]


def cache_bytes(keys, values, layout, poison=0, mutate_request=None):
    if len(keys) != BASE_ROWS * 1024 or len(values) != len(keys):
        raise ValueError("requires frozen initialized4100-row K/V")
    if mutate_request not in (None, 2):
        raise ValueError("only bounded request2 tail mutation is supported")
    spec = LAYOUTS[layout]
    page_stride, row_stride, head_stride, _ = spec["strides"]
    storage = bytearray([poison]) * spec["bytes"]
    for request, length in enumerate(LENGTHS):
        for row in range(length):
            block = REQUEST_BLOCKS[request][row // PAGE]
            for head in range(4):
                src = (((row + request * 337) % BASE_ROWS) * 4 + head) * 256
                dst = block * page_stride + (row % PAGE) * row_stride + head * head_stride
                storage[dst:dst + 256] = keys[src:src + 256]
                begin = dst + spec["v_offset"]
                storage[begin:begin + 256] = (
                    bytes([0x38]) * 256 if request == mutate_request and row >= length - 3
                    else values[src:src + 256])
    return storage


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix(".json").exists(),
                    "refusing existing batch capture")
    headers.require(digest(args.source.read_bytes()) == SOURCE_SHA, "source worker fixture changed")
    reference = verify_inputs(args.reference_manifest, args.model, args.image_identity)
    index = headers.read_shard_header(args.source)
    base_query = blob(args.source, index, "query")[1]
    keys = b"".join(blob(args.source, index, f"k_page{i}")[1] for i in range(3))
    values = b"".join(blob(args.source, index, f"v_page{i}")[1] for i in range(3))
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
    isolated_calls = 0
    for number, case in enumerate(batch_cases()):
        spec, order = LAYOUTS[case["layout"]], case["request_ids"]
        host = cache_bytes(keys, values, case["layout"], mutate_request=case["mutate_request"])
        storage = torch.frombuffer(host, dtype=torch.uint8).to("xpu")
        shape = (BLOCKS, PAGE, 4, 256)
        key = storage.as_strided(shape, spec["strides"]).view(torch.float8_e4m3fn)
        value = storage.as_strided(shape, spec["strides"], spec["v_offset"]).view(torch.float8_e4m3fn)
        query_raw = b"".join(query_bytes(base_query, r) for r in order)
        query = torch.frombuffer(bytearray(query_raw), dtype=torch.float16).reshape(16, 24, 256).to("xpu")
        descale = torch.ones(1, dtype=torch.float32, device="xpu")
        data = {"q": query, "k": key, "v": value, "out": torch.empty_like(query),
                "max_seqlen_q": 4, "max_seqlen_k": max(case["lengths"]),
                "softmax_scale": 0.0625, "causal": True,
                "cu_seqlens_q": torch.tensor([0, 4, 8, 12, 16], dtype=torch.int32, device="xpu"),
                "seqused_k": torch.tensor(case["lengths"], dtype=torch.int32, device="xpu"),
                "block_table": torch.tensor(table_rows(order), dtype=torch.int32, device="xpu"),
                "k_descale": descale, "v_descale": descale}
        headers.require(verify.eligible(data), "actual wrapper declined C4 uniform case")
        calls = []

        class Witness(TorchDispatchMode):
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                if str(func) == "b70_exl3_attention.shared_kv_verify_out.default":
                    calls.append({"operator": str(func), "packed_query_shape": list(args[0].shape),
                                  "physical_offsets": args[4].cpu().tolist(),
                                  "max_keys": args[-3], "splits": args[-2], "tile": args[-1]})
                return func(*args, **(kwargs or {}))

        with Witness():
            result = verify.run(data)
        torch.xpu.synchronize()
        raw = result.cpu().contiguous().numpy().tobytes()
        headers.require(len(calls) == 1 and calls[0]["physical_offsets"] == [0, 1, 2, 3, 4],
                        "actual single C4 packed invocation missing")
        headers.require(result.data_ptr() == data["out"].data_ptr(), "original lost C4 direct output")
        headers.require(torch.isfinite(result).all().item(), "nonfinite original C4 output")
        headers.require(raw == verify.run(data).cpu().contiguous().numpy().tobytes(),
                        "ordinary repeat differs from observed original")
        headers.require(storage.cpu().numpy().tobytes() == bytes(host), "original wrote cache")
        headers.require(query.cpu().numpy().tobytes() == query_raw, "original wrote query")
        pieces = [raw[r * 49152:(r + 1) * 49152] for r in range(4)]
        if case["mutate_request"] is None:
            for slot, request in enumerate(order):
                headers.require(pieces[slot] == canonical.setdefault(request, pieces[slot]),
                                "C4 request output differs across permutation/layout")
        else:
            for slot, request in enumerate(order):
                ordinary = canonical[request]
                if request != case["mutate_request"]:
                    headers.require(pieces[slot] == ordinary, "request2 mutation contaminated another request")
                else:
                    headers.require(pieces[slot][:12288] == ordinary[:12288], "causal row0 sees tail3")
                    headers.require(pieces[slot][12288:] != ordinary[12288:], "tail mutation lacks causal witness")
        if case["id"].endswith("-ordinary"):
            for slot, request in enumerate(order):
                one = data | {"q": query[slot * 4:(slot + 1) * 4].contiguous(),
                              "out": torch.empty_like(query[:4]), "max_seqlen_k": LENGTHS[request],
                              "cu_seqlens_q": torch.tensor([0, 4], dtype=torch.int32, device="xpu"),
                              "seqused_k": data["seqused_k"][slot:slot + 1],
                              "block_table": data["block_table"][slot:slot + 1]}
                headers.require(verify.eligible(one), "original declined isolated same request")
                headers.require(verify.run(one).cpu().numpy().tobytes() == pieces[slot],
                                "true C4 differs from independent original C1 invocation")
                isolated_calls += 1
                del one
        # True max-query metadata4 cannot classify mixed2/4/4/4 total14 as uniform.
        ragged = data | {"q": query[:14], "out": data["out"][:14],
                         "cu_seqlens_q": torch.tensor([0, 2, 6, 10, 14], dtype=torch.int32, device="xpu")}
        headers.require(not verify.eligible(ragged), "original incorrectly admitted ragged batch")
        prefix = f"case{number}"
        tensors[prefix + "_query"] = ("F16", [16, 24, 256], query_raw)
        tensors[prefix + "_output"] = ("F16", [16, 24, 256], raw)
        records.append(case | {"prefix": prefix, "KV_strides": spec["strides"],
                               "V_offset": spec["v_offset"], "storage_bytes": spec["bytes"],
                               "block_table": table_rows(order), "logical_offsets": [0, 4, 8, 12, 16],
                               "route": calls[0], "repeat_exact": True,
                               "direct_C4_output": True, "ragged_guard_declined": True})
        print("ORIGINAL_VERIFY_BATCH_CASE_PASS", case["id"], flush=True)
        del ragged, data, result, query, key, value, storage, host, descale
    memory = {"allocated_bytes": torch.xpu.memory_allocated(),
              "reserved_bytes": torch.xpu.memory_reserved(),
              "peak_allocated_bytes": torch.xpu.max_memory_allocated(),
              "peak_reserved_bytes": torch.xpu.max_memory_reserved()}
    write_safetensors(args.output, tensors, {"scope": __doc__})
    report = {"schema": "b70-exl3-P1-C4-verifier-batch-v1", "cases": records,
              "source_sha256": SOURCE_SHA, "image": args.image_identity,
              "checkpoint": reference["checkpoint"]["identity"],
              "wrapper_sha256": VERIFY_SHA, "library_sha256": digest(library.read_bytes()),
              "tool_sha256": digest(Path(__file__).read_bytes()), "memory": memory,
              "capture_sha256": digest(args.output.read_bytes()), "physical_blocks": BLOCKS,
              "request_blocks": REQUEST_BLOCKS, "page": PAGE, "base_rows": BASE_ROWS,
              "isolated_original_C1_comparisons": isolated_calls,
              "KV_recipe": "request r uses initialized row (row+r*337) modulo4100; disjoint request pages",
              "query_recipe": "request r rotates original four query rows by r",
              "inactive_original_storage": "zero-filled; native poison checked separately",
              "scope": __doc__}
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "reference-manifest", "model", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
