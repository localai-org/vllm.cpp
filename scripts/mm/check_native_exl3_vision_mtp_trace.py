#!/usr/bin/env python3
"""Check actual target/MTP image HTTP receipts and committed-token traces.

Consumes captures from an isolated C1 native server with VT_NATIVE_VISION_TRACE=2.
No server management, inference fallback or performance claim.
"""
import argparse
import json
import math
from pathlib import Path


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def check(report, log, depth):
    require(report.get("status") == "PASS" and report.get("depth") == depth,
            "wrong or failed serving capture")
    require(report.get("server_removed") and report.get("production_stopped"),
            "isolated server not drained/removed")
    cases = report["cases"]
    require(len(cases) == 5, "five PNG/JPEG/SSE cases required")
    require([c["stream"] for c in cases] == [False] * 4 + [True], "missing live SSE case")
    require(all(c["semantic_pass"] and c["usage"] == {
        "prompt_tokens": 245, "completion_tokens": 64, "total_tokens": 309
    } and all(v == 0 for v in c["gauges"].values()) for c in cases), "semantic/usage/drain failure")
    require(cases[0]["text"] == cases[4]["text"], "SSE completion differs")
    sampled = {}
    targets, drafts, encodes = [], [], []
    prompt_axes = [0, 1, 2, 3] + [4] * 192 + list(range(20, 69))
    spatial_h = [0, 1, 2, 3] + [4 + i // 16 for i in range(192)] + list(range(20, 69))
    spatial_w = [0, 1, 2, 3] + [4 + i % 16 for i in range(192)] + list(range(20, 69))
    prompt_mask = [0] * 4 + [1] * 192 + [0] * 49
    request_index = -1
    for line in log.splitlines():
        if line.startswith("NATIVE_VISION_ENCODE "):
            encodes.append(line)
        elif line.startswith("NATIVE_VISION_EMBED "):
            frame = json.loads(line.split(" ", 1)[1])
            targets.append(frame)
            if frame["tokens"] == 245:
                request_index += 1
                require(frame["mrope"] == prompt_axes + spatial_h + spatial_w,
                        "target prompt axes differ")
                require(frame["image_mask"] == prompt_mask and frame["source_slices"] == [[0, 192]],
                        "target image source differs")
            else:
                committed = len(sampled.get("chatcmpl-" + str(request_index), []))
                start = 69 + committed - 1
                require(frame["mrope"] == list(range(start, start + frame["tokens"])) * 3,
                        "decode/verification coordinate or rollback differs")
                require(not any(frame["image_mask"]) and not frame["source_slices"],
                        "image rows reused during text verification")
        elif line.startswith("NATIVE_VISION_DRAFT_EMBED "):
            drafts.append(json.loads(line.split(" ", 1)[1]))
        elif line.startswith("NATIVE_VISION_SAMPLED "):
            record = json.loads(line.split(" ", 1)[1])
            require(all(isinstance(t, int) and 0 <= t < 248320 for t in record["token_ids"]),
                    "invalid committed target token")
            sampled.setdefault(record["request_id"], []).extend(record["token_ids"])
    require(request_index == 4 and len(encodes) == 4, "encoder reuse/request count differs")
    require(len(drafts) == (5 if depth else 0), "actual draft image merge missing")
    for frame in drafts:
        require(frame["tokens"] == 245 and frame["source_slices"] == [[0, 192]]
                and frame["image_mask"] == prompt_mask[1:] + [0], "draft lookahead/source differs")
    ids = [sampled.get("chatcmpl-" + str(i), [])[:64] for i in range(5)]
    require(all(len(tokens) == 64 for tokens in ids) and ids == report["sampled_ids"],
            "committed trace/report token IDs differ")
    require(ids[0] == ids[4], "SSE token IDs differ")
    if depth:
        delta = {k: v - report["metrics_before"].get(k, 0)
                 for k, v in report["metrics_after"].items() if "spec_decode" in k}
        require(all(math.isfinite(v) and v >= 0 for v in delta.values()), "invalid MTP counters")
        require(delta.get("vllm:spec_decode_num_draft_tokens_total", 0) > 0
                and delta.get("vllm:spec_decode_num_accepted_tokens_total", 0) > 0,
                "no actual proposal/verification")
    return ids



def check_graphs(log, depth):
    frames = [json.loads(line.split(" ", 1)[1]) for line in log.splitlines()
              if line.startswith("NATIVE_VISION_GRAPH ")]
    require(len(frames) > 2, "missing actual native MM graph observations")
    counts = [frame["replay_count"] for frame in frames]
    require(all(0 < frame["tokens"] <= depth + 1 for frame in frames), "unexpected graph shape")
    require(counts == sorted(counts) and counts[-1] > counts[0] + 2,
            "no progressing native MM graph executions")
    require(any(frame["captured"] and frame["replay_count"] > 2 for frame in frames),
            "eager fallback cannot qualify graph execution")
    return {"observations": len(frames), "graph_executions": counts[-1]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-report", type=Path, required=True)
    parser.add_argument("--target-log", type=Path, required=True)
    parser.add_argument("--mtp-report", type=Path, required=True)
    parser.add_argument("--mtp-log", type=Path, required=True)
    parser.add_argument("--depth", type=int, choices=[1, 3], required=True)
    parser.add_argument("--require-graphs", action="store_true")
    args = parser.parse_args()
    target = check(json.loads(args.target_report.read_text()), args.target_log.read_text(), 0)
    draft = check(json.loads(args.mtp_report.read_text()), args.mtp_log.read_text(), args.depth)
    require(target == draft, "native target/MTP greedy token sequence differs")
    graphs = check_graphs(args.mtp_log.read_text(), args.depth) if args.require_graphs else None
    print(json.dumps({"status": "PASS", "depth": args.depth, "requests_per_mode": 5,
                      "exact_committed_tokens_per_mode": 320, "graphs": graphs,
                      "scope": "bounded native image MTP; no Python numerical or performance parity"}))


if __name__ == "__main__":
    main()
