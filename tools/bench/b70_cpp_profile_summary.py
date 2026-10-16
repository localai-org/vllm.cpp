#!/usr/bin/env python3
"""Summarize native B70 SYCL stage and host-span profile records."""

import argparse
import collections
import json
from pathlib import Path


def family(stage: str) -> str:
    if stage == "onednn_gptq4_stream":
        return "gptq_int4"
    if stage == "onednn_dense_f16_stream":
        return "dense_fp16_and_head"
    if stage.startswith("gdn_chunk_") or stage.startswith("gdn_decode_recurrence"):
        return "gdn_core_state"
    if stage.startswith("conv1d_"):
        return "gdn_conv"
    if stage == "gdn_postconv":
        return "gdn_postconv"
    if stage.startswith("gdn_gated_norm") or stage.startswith("gdn_state_"):
        return "gdn_norm_state_io"
    if stage.startswith("attention_") or stage.startswith("attn_qk_"):
        return "attention"
    if stage.startswith("kv_write"):
        return "kv_write"
    return "norm_rope_glue"


def union_ns(intervals: list[tuple[int, int]]) -> int:
    total = 0
    current_start = current_end = None
    for start, end in sorted(intervals):
        if current_start is None:
            current_start, current_end = start, end
        elif start <= current_end:
            current_end = max(current_end, end)
        else:
            total += current_end - current_start
            current_start, current_end = start, end
    if current_start is not None:
        total += current_end - current_start
    return total


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.read_text().splitlines()
            if line.startswith("{")]
    config = next(row for row in rows if row.get("event") == "gptq4_benchmark_config")
    round_row = next(row for row in rows if row.get("event") == "gptq4_benchmark_round")
    host = next((row for row in rows if row.get("event") == "gptq4_host_spans"), None)
    phases = {}
    for row in rows:
        if row.get("event") != "gptq4_stage_trace":
            continue
        totals = collections.defaultdict(lambda: [0, 0, 0])
        intervals = []
        for event in row["events"]:
            start, end = event["start_ns"], event["end_ns"]
            if end < start:
                raise ValueError("negative device interval")
            intervals.append((start, end))
            entry = totals[family(event["stage"])]
            entry[0] += 1
            entry[1] += end - start
            entry[2] += bool(event["stream_span"])
        device_span_ns = max(end for _, end in intervals) - min(start for start, _ in intervals)
        phases[row["phase"]] = {
            "event_count": len(intervals),
            "device_span_ms": device_span_ns / 1e6,
            "device_interval_union_ms": union_ns(intervals) / 1e6,
            "device_gap_within_span_ms": (device_span_ns - union_ns(intervals)) / 1e6,
            "families": {
                name: {"event_count": item[0], "device_event_sum_ms": item[1] / 1e6,
                       "stream_span_count": item[2]}
                for name, item in totals.items()},
            "host_spans": host[row["phase"]] if host is not None else None,
            "forward_wall_ms": round_row["prefill_seconds" if row["phase"] == "prefill"
                                          else "decode_seconds"] * 1000,
        }
    return {
        "profile_scope": "SYCL event profiling; device family sums and nested host spans are not disjoint wall time",
        "P": config["prompt_tokens"],
        "D": config["decode_forward_steps"],
        "O": config["output_tokens"],
        "kv_dtype": config["kv_dtype"],
        "page_tokens": config["kv_block_size"],
        "prompt_ids_fnv1a64": config["prompt_ids_fnv1a64"],
        "decode_ids_fnv1a64": config["decode_ids_fnv1a64"],
        "phases": phases,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = summarize(args.log)
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for phase, row in result["phases"].items():
        print(phase, "wall_ms", round(row["forward_wall_ms"], 3),
              "events", row["event_count"], "device_gap_ms",
              round(row["device_gap_within_span_ms"], 3))
        print(sorted(((name, group["event_count"],
                       round(group["device_event_sum_ms"], 3))
                      for name, group in row["families"].items()),
                     key=lambda item: -item[2]))


if __name__ == "__main__":
    main()
