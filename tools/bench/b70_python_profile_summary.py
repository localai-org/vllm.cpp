#!/usr/bin/env python3
"""Assign pinned vLLM XPU kernels to V2 worker prefill/decode annotations.

Kernel execution is asynchronous. Assignment uses the correlated CPU enqueue
timestamp, not the GPU execution timestamp, to avoid moving late prefill work
into the next decode annotation. All durations are sums of kernel events; they
are not disjoint wall-clock times.
"""

import argparse
import bisect
import collections
import gzip
import json
from pathlib import Path


def family(name: str, parent: str) -> str:
    if parent == "_xpu_C::int4_gemm_w4a16":
        return "gptq_int4"
    if parent in ("aten::mm", "aten::addmm") and "gemm_kernel" in name:
        return "dense_fp16"
    if parent == "_xpu_C::gdn_attention" or name.startswith("gdn::"):
        if "causal_conv" in name:
            return "gdn_conv"
        if "norm" in name:
            return "gdn_norm"
        return "gdn_core_state"
    if parent == "_vllm_fa2_C::varlen_fwd":
        return "attention"
    if "reshape_and_cache" in name or "kv_" in name:
        return "kv_write"
    if ("norm" in name.lower() or "act_and_mul" in name or
            parent in ("_C::silu_and_mul", "_C::fused_add_gemma_rms_norm")):
        return "norm_activation"
    return "other_glue"


def read_trace(path: Path) -> list[dict]:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as file:
        return json.load(file)["traceEvents"]


def summarize(path: Path) -> dict:
    events = read_trace(path)
    marks = sorted(
        (event for event in events
         if event.get("cat") == "user_annotation" and
         event.get("name", "").startswith("execute_context_")),
        key=lambda event: event["ts"])
    if not marks or not any("generation_1(1)" in mark["name"] for mark in marks):
        raise ValueError("no active V2 prefill/decode worker annotations")
    starts = [mark["ts"] for mark in marks]
    parent_by_id = {
        event["args"]["External id"]: event["name"]
        for event in events if event.get("cat") == "cpu_op" and
        "External id" in event.get("args", {})
    }
    enqueue_by_id = {
        event["args"]["External id"]: event
        for event in events if event.get("cat") == "xpu_runtime" and
        event.get("name", "").startswith("urEnqueueKernelLaunch") and
        "External id" in event.get("args", {})
    }
    copy_enqueue_by_id = {
        event["args"]["External id"]: event
        for event in events if event.get("cat") == "xpu_runtime" and
        event.get("name") == "urEnqueueUSMMemcpy" and
        "External id" in event.get("args", {})
    }
    per_mark = [collections.defaultdict(lambda: [0, 0.0, 0.0]) for _ in marks]
    per_mark_copies = [collections.defaultdict(lambda: [0, 0, 0.0]) for _ in marks]
    unassigned = collections.Counter()
    for event in events:
        if event.get("cat") != "kernel":
            continue
        external_id = event.get("args", {}).get("External id")
        enqueue = enqueue_by_id.get(external_id)
        if enqueue is None:
            unassigned["missing_correlated_enqueue"] += 1
            unassigned["missing_correlated_enqueue_us"] += event["dur"]
            continue
        index = bisect.bisect_right(starts, enqueue["ts"]) - 1
        inside = (index >= 0 and enqueue["ts"] < marks[index]["ts"] +
                  marks[index]["dur"])
        parent = parent_by_id.get(external_id, "")
        if not inside:
            # compute_logits() runs after execute_model's V2 annotation.
            # Correlated aten::mm launches in that gap are the dense head.
            if (index < 0 or
                    "context_0(0)_generation_0(0)" in marks[index]["name"] or
                    parent != "aten::mm" or event["name"] != "gemm_kernel"):
                unassigned["outside_worker_annotation"] += 1
                unassigned["outside_worker_annotation_us"] += event["dur"]
                continue
            label = "post_execute_dense_head"
        else:
            label = family(event["name"], parent)
        row = per_mark[index][label]
        row[0] += 1
        row[1] += event["dur"]
        row[2] += enqueue["dur"]

    for event in events:
        if event.get("cat") != "gpu_memcpy":
            continue
        enqueue = copy_enqueue_by_id.get(event.get("args", {}).get("External id"))
        if enqueue is None:
            unassigned["copy_missing_correlated_enqueue"] += 1
            continue
        index = bisect.bisect_right(starts, enqueue["ts"]) - 1
        if (index < 0 or enqueue["ts"] >= marks[index]["ts"] +
                marks[index]["dur"]):
            unassigned["copy_outside_worker_annotation"] += 1
            unassigned["copy_outside_worker_bytes"] += event["args"].get("bytes", 0)
            continue
        row = per_mark_copies[index][event["name"]]
        row[0] += 1
        row[1] += event["args"].get("bytes", 0)
        row[2] += event["dur"]

    phases = {}
    for index, mark in enumerate(marks):
        name = mark["name"]
        if "generation_1(1)" in name:
            phase = "decode"
        elif "context_1(" in name:
            phase = "prefill"
        else:
            continue
        group = phases.setdefault(phase, {
            "worker_annotations": 0, "worker_annotation_us": 0.0,
            "families": {}, "memcpy": {}, "per_forward": []})
        group["worker_annotations"] += 1
        group["worker_annotation_us"] += mark["dur"]
        forward = {key: {"kernel_count": value[0], "kernel_us": value[1],
                         "enqueue_cpu_us": value[2]}
                   for key, value in per_mark[index].items()}
        group["per_forward"].append(forward)
        for key, value in per_mark[index].items():
            total = group["families"].setdefault(key, {
                "kernel_count": 0, "kernel_us": 0.0, "enqueue_cpu_us": 0.0})
            total["kernel_count"] += value[0]
            total["kernel_us"] += value[1]
            total["enqueue_cpu_us"] += value[2]
        for key, value in per_mark_copies[index].items():
            total = group["memcpy"].setdefault(key, {
                "count": 0, "bytes": 0, "device_us": 0.0})
            total["count"] += value[0]
            total["bytes"] += value[1]
            total["device_us"] += value[2]
    return {
        "trace": path.name,
        "assignment": "kernel External id -> urEnqueueKernelLaunch timestamp -> V2 worker annotation",
        "duration_scope": "sum of kernel durations; overlapping GPU work and nested CPU work are not wall time",
        "worker_annotations": len(marks),
        "unassigned": dict(unassigned),
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.trace)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase, group in result["phases"].items():
        print(phase, group["worker_annotations"], sorted(
            ((key, value["kernel_count"], round(value["kernel_us"] / 1000, 3))
             for key, value in group["families"].items()),
            key=lambda row: -row[2]))
    print("unassigned", result["unassigned"])


if __name__ == "__main__":
    main()
