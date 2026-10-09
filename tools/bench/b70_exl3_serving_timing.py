#!/usr/bin/env python3
"""Frozen R11 workloads and emitted-token timing; no model runtime here."""
import argparse
import hashlib
import json
import math
import statistics
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def identity(path):
    data = Path(path).read_bytes()
    return {"path": str(Path(path).resolve()), "sha256": hashlib.sha256(data).hexdigest()}


def write_new(path, data):
    with Path(path).open("x") as stream:
        json.dump(data, stream, indent=2)
        stream.write("\n")


def prepare(p4k, p32k, c4_prompts):
    four = read(p4k)["prompt_token_ids"]
    long = read(p32k)["prompt_token_ids"]
    assert len(four) == 4096 and len(long) == 32768
    assert four[:1600] != long[:1600], "mixed cold requests must not share their first page"
    distinct = [p["prompt_token_ids"] for p in read(c4_prompts)["prompts"]]
    assert len(distinct) == 4 and all(len(p) == 4096 for p in distinct)
    # Prefix caching stays enabled. Different first physical pages prevent
    # cross-request hits from turning the C4 cold-prefill case into warm reuse.
    assert len({tuple(p[:1600]) for p in distinct}) == 4
    cases = []
    for name, concurrency, prompt, outputs in (
        ("c1-p4096-o1024", 1, four, 1024),
        ("c4-p4096-o256", 4, four, 256),
        ("c4-p4096-o1024", 4, four, 1024),
        ("c1-p32768-o256", 1, long, 256),
    ):
        cases.append({"id": name, "cache_state": "cold", "requests": [
            {"id": f"request-{i}", "prompt_ids": distinct[i] if concurrency == 4 else prompt,
             "output_tokens": outputs}
            for i in range(concurrency)]})
    cases.append({"id": "mixed-p128-p32768-o256", "cache_state": "cold", "requests": [
        {"id": "short", "prompt_ids": four[:128], "output_tokens": 256},
        {"id": "long", "prompt_ids": long, "output_tokens": 256,
         "after_emitted": {"request": "short", "tokens": 16}},
    ]})
    for concurrency in (1, 4):
        cases.append({"id": f"c{concurrency}-p4096-o256-m0", "cache_state": "cold",
                      "mtp_depth": 0, "max_model_len": 262144, "requests": [
            {"id": f"request-{i}", "prompt_ids": distinct[i] if concurrency == 4 else four,
             "output_tokens": 256} for i in range(concurrency)]})
    # Keep the historical P128 mixed case identifiable. The performance plan
    # asks for a separate genuine4K decode owner before the32K admission.
    cases.append({"id": "mixed-p4096-p32768-o256", "cache_state": "cold", "requests": [
        {"id": "short", "prompt_ids": four, "output_tokens": 256},
        {"id": "long", "prompt_ids": long, "output_tokens": 256,
         "after_emitted": {"request": "short", "tokens": 16}},
    ]})
    # Frozen natural-token composition: four distinct4K introductions followed
    # by the same frozen32K tail. No runtime tokenization or cross-prefix hits.
    cases.append({"id": "c4-p32768-o256", "cache_state": "cold", "requests": [
        {"id": f"request-{i}", "prompt_ids": distinct[i] + long[4096:],
         "output_tokens": 256} for i in range(4)]})
    return {"schema": "b70-exl3-r11-serving-workload-v1",
            "sources": [identity(p4k), identity(p32k), identity(c4_prompts)],
            "sampling": {"temperature": 0.0, "ignore_eos": True},
            "note": "Exact token budgets in both arms; this is throughput work, not text quality. "
                    "C4 uses four distinct frozen natural prompts with different first pages; "
                    "C4-32K composes each distinct4K introduction with the frozen32K tail; "
                    "cold means a fresh engine and each request starts at position zero.",
            "cases": cases}


def positive(value):
    assert math.isfinite(value) and value > 0
    return value


def summarize(raw, case):
    native = raw["schema"] == "b70-exl3-r11-serving-output-v1"
    assert native or raw["schema"] == "b70-exl3-r11-producer-serving-output-v1"
    assert raw["case"] == case["id"] and raw["cache_state"] == case["cache_state"] == "cold"
    assert raw["sampling"] == {"temperature": 0.0, "ignore_eos": True}
    depth = case.get("mtp_depth", 3)
    assert type(depth) is int and depth in (0, 3)
    actual_depth = raw.get("launch", {}).get("mtp_depth", 3) if native else raw.get("mtp_depth", 3)
    assert actual_depth == depth, "actual engine speculation differs from workload"
    if "max_model_len" in case:
        limit = case["max_model_len"]
        actual_limit = raw["launch"]["max_model_len"] if native else raw["requested_engine_args"]["max_model_len"]
        assert actual_limit == limit, "actual context limit differs from workload"
    requested = {spec["id"]: spec for spec in case["requests"]}
    assert set(raw["requests"]) == set(requested)
    traces = raw["requests"]
    per_request = {}
    for key, spec in requested.items():
        trace = traces[key]
        assert trace["prompt_ids"] == spec["prompt_ids"]
        if native:
            assert trace["first_scheduled_position"] == 0
        else:
            # Public producer output reports cache hits, not scheduled positions.
            assert trace["num_cached_tokens"] == 0
        count = len(trace["ids"])
        assert count == trace["output_limit"] == spec["output_tokens"]
        assert trace["finish_reason"] == "length"
        previous, previous_time = 0, trace["admitted_at_s"]
        observations = trace["observations"]
        assert observations
        for obs in observations:
            assert obs["at_s"] >= previous_time
            assert obs["new_tokens"] > 0
            assert obs["tokens"] - previous == obs["new_tokens"]
            previous, previous_time = obs["tokens"], obs["at_s"]
        assert previous == count
        first, last = observations[0]["at_s"], observations[-1]["at_s"]
        assert trace["finished_at_s"] >= last
        per_request[key] = {
            "prompt_tokens": len(spec["prompt_ids"]), "emitted_tokens": count,
            "ttft_ms": 1000 * positive(first - trace["admitted_at_s"]),
            "tpot_ms": 1000 * (last - first) / (count - 1) if count > 1 else None,
            "longest_observed_streaming_pause_ms": max(
                (1000 * (right["at_s"] - left["at_s"])
                 for left, right in zip(observations, observations[1:])), default=None),
            "end_to_end_ms": 1000 * positive(trace["finished_at_s"] - trace["admitted_at_s"]),
            "token_timestamp_granularity": "one timestamp per emitted MTP chunk" if depth else
                                           "one timestamp per emitted frontend chunk",
        }
        if "after_emitted" in spec:
            trigger = spec["after_emitted"]
            observed = [o for o in traces[trigger["request"]]["observations"]
                        if o["at_s"] <= trace["admitted_at_s"]]
            assert observed and observed[-1]["tokens"] >= trigger["tokens"], "early mixed admission"
            assert len(observed) == 1 or observed[-2]["tokens"] < trigger["tokens"], "late mixed admission"
            per_request[key]["admission_trigger"] = dict(
                trigger, observed_tokens_at_admission=observed[-1]["tokens"],
                observed_at_s=observed[-1]["at_s"])
    # Use the SAME common interval for every request; never add individual tok/s.
    overlap_start = max(t["observations"][0]["at_s"] for t in traces.values())
    overlap_end = min(t["observations"][-1]["at_s"] for t in traces.values())
    overlap_tokens = sum(obs["new_tokens"] for t in traces.values() for obs in t["observations"]
                         if overlap_start < obs["at_s"] <= overlap_end)
    overlap = None if overlap_end <= overlap_start else {
        "start_s": overlap_start, "end_s": overlap_end, "emitted_tokens": overlap_tokens,
        "aggregate_emitted_tokens_per_s": overlap_tokens / (overlap_end - overlap_start),
        "boundary_policy": "exclude start, include end; whole emitted chunks only",
    }
    duration = positive(raw["end_to_end_wall_s"])
    common = {"schema": "b70-exl3-r11-serving-report-v1", "case": case["id"],
              "mtp_depth": depth,
              "profiled": raw["profiled"], "unprofiled_serving_measurement": not raw["profiled"],
              "startup_state": raw.get("startup_state", "constructor_only_native_lazy_work_in_first_request"),
              "warmup": raw.get("warmup"), "per_request": per_request,
              "common_decode_overlap": overlap, "end_to_end_wall_s": duration,
              "end_to_end_emitted_tokens_per_s": sum(len(t["ids"]) for t in traces.values()) / duration,
              "target_qualified": False, "serving_qualified": False}
    if not native:
        assert not raw["profiled"]
        assert raw["startup_state"] == "request_warmup_o32_then_prefix_reset"
        batches = raw["frontend_batches"]
        assert batches
        assert sum(b["emitted_tokens"] for b in batches) == sum(len(t["ids"]) for t in traces.values())
        previous_end = 0.0
        supplied = []
        for b in batches:
            assert b["start_s"] >= previous_end and b["end_s"] >= b["start_s"]
            assert abs(b["frontend_wall_s"] - (b["end_s"] - b["start_s"])) < 1e-8
            previous_end = b["end_s"]
            if b["spec_stats"] is not None:
                stat = b["spec_stats"]
                assert stat["num_spec_tokens"] == depth
                assert 0 <= stat["accepted"] <= stat["proposed"]
                supplied.append(stat)
        assert duration >= previous_end
        proposed = sum(s["proposed"] for s in supplied)
        accepted = sum(s["accepted"] for s in supplied)
        if depth:
            assert proposed > 0, "producer MTP statistics missing"
        else:
            assert proposed == accepted == 0, "target-only run reported speculative tokens"
        return dict(common, cycles=None, decode_cycles=None, mean_decode_cycle_ms=None,
                    mean_emitted_tokens_per_decode_cycle=None, graph_captures=None, graph_replays=None,
                    proposed=proposed, accepted=accepted, draft_acceptance=accepted / proposed if proposed else None,
                    frontend_batches=len(batches), spec_statistics_batches=len(supplied),
                    mean_frontend_batch_ms=1000 * statistics.mean(b["frontend_wall_s"] for b in batches),
                    profile_spans={}, accounting_note="Producer frontend batches are not target forwards or GPU MTP cycles. "
                    "Proposal counters sum actual supplied scheduler stats; no inference from emitted chunks.")
    cycles = raw["cycles"]
    assert cycles
    assert sum(c["emitted_tokens"] for c in cycles) == sum(len(t["ids"]) for t in traces.values())
    previous_end = 0.0
    decode = []
    for c in cycles:
        assert c["start_s"] >= previous_end and c["end_s"] >= c["start_s"]
        assert abs(c["cycle_wall_s"] - (c["end_s"] - c["start_s"])) < 1e-8
        assert 0 <= c["accepted"] <= c["proposed"]
        previous_end = c["end_s"]
        if c["positions"] and all(position >= len(requested[key]["prompt_ids"])
                                  for key, position in c["positions"].items()):
            decode.append(c)
    assert duration >= previous_end
    proposed = sum(c["proposed"] for c in cycles)
    accepted = sum(c["accepted"] for c in cycles)
    if not depth:
        assert proposed == accepted == 0, "target-only run reported speculative tokens"
    profile = {}
    if raw["profiled"]:
        if raw["device_profile"]:
            stages = ("runner_target_forward", "runner_mtp_draft") if depth else ("runner_target_forward",)
            if not depth:
                assert not any(r["stage"] == "runner_mtp_draft" for r in raw["device_profile_records"])
            for stage in stages:
                spans = [r for r in raw["device_profile_records"] if r["stage"] == stage]
                assert len(spans) == len(cycles), (stage, len(spans), len(cycles))
                assert all(r["stream_span"] and r["end_ns"] >= r["start_ns"] for r in spans)
                milliseconds = [(r["end_ns"] - r["start_ns"]) / 1e6 for r in spans]
                profile[stage] = {"count": len(spans), "total_ms": sum(milliseconds),
                                  "mean_ms": statistics.mean(milliseconds),
                                  "decode_mean_ms": statistics.mean(milliseconds[c["cycle"]] for c in decode)
                                  if decode else None}
        profile["note"] = "Separate diagnostic run. Queue spans include host gaps; nested operator/graph spans must not be added."
    return dict(common, cycles=len(cycles), decode_cycles=len(decode),
                mean_decode_cycle_ms=1000 * statistics.mean(c["cycle_wall_s"] for c in decode) if decode else None,
                mean_emitted_tokens_per_decode_cycle=statistics.mean(c["emitted_tokens"] for c in decode) if decode else None,
                accepted=accepted, proposed=proposed, draft_acceptance=accepted / proposed if proposed else None,
                graph_captures=sum(c["graph_captures"] for c in cycles),
                graph_replays=sum(c["graph_replays"] for c in cycles), profile_spans=profile)


def self_check():
    case = {"id": "control", "cache_state": "cold", "requests": [
        {"id": "a", "prompt_ids": [1], "output_tokens": 3},
        {"id": "b", "prompt_ids": [2], "output_tokens": 3}]}
    raw = {"schema": "b70-exl3-r11-serving-output-v1", "case": "control", "cache_state": "cold",
           "sampling": {"temperature": 0.0, "ignore_eos": True}, "profiled": False,
           "requests": {}, "cycles": [], "end_to_end_wall_s": 5.0}
    for key, first, last in (("a", 1.0, 4.0), ("b", 2.0, 5.0)):
        raw["requests"][key] = {"prompt_ids": case["requests"][key == "b"]["prompt_ids"],
            "ids": [7, 8, 9], "output_limit": 3, "finish_reason": "length", "first_scheduled_position": 0,
            "admitted_at_s": 0.0, "finished_at_s": last, "observations": [
                {"at_s": first, "tokens": 1, "new_tokens": 1},
                {"at_s": last, "tokens": 3, "new_tokens": 2}]}
    for index, (end, tokens) in enumerate(((1, 1), (2, 1), (4, 2), (5, 2))):
        start = end - 0.5
        raw["cycles"].append({"cycle": index, "start_s": start, "end_s": float(end), "cycle_wall_s": 0.5,
                              "emitted_tokens": tokens, "accepted": 0, "proposed": 0,
                              "positions": {"a": 1}, "graph_captures": 0, "graph_replays": 0})
    report = summarize(raw, case)
    assert report["common_decode_overlap"]["aggregate_emitted_tokens_per_s"] == 1.0
    assert report["per_request"]["a"]["tpot_ms"] == 1500.0
    assert report["per_request"]["a"]["longest_observed_streaming_pause_ms"] == 3000.0
    assert report["end_to_end_emitted_tokens_per_s"] == 1.2
    triggered_case = json.loads(json.dumps(case))
    triggered_case["requests"][1]["after_emitted"] = {"request": "a", "tokens": 1}
    triggered = json.loads(json.dumps(raw))
    triggered["requests"]["b"]["admitted_at_s"] = 1.5
    assert summarize(triggered, triggered_case)["per_request"]["b"]["admission_trigger"][
        "observed_tokens_at_admission"] == 1
    for arrived, first, reason in ((0.5, 2.0, "early mixed admission"),
                                  (4.5, 4.75, "late mixed admission")):
        triggered["requests"]["b"]["admitted_at_s"] = arrived
        triggered["requests"]["b"]["observations"][0]["at_s"] = first
        try:
            summarize(triggered, triggered_case)
        except AssertionError as error:
            assert str(error) == reason
        else:
            raise AssertionError("early/late mixed admission cannot qualify the requested trigger")
    producer = json.loads(json.dumps(raw))
    producer.update(schema="b70-exl3-r11-producer-serving-output-v1",
                    startup_state="request_warmup_o32_then_prefix_reset")
    producer["frontend_batches"] = [dict(batch=c["cycle"], start_s=c["start_s"], end_s=c["end_s"],
        frontend_wall_s=c["cycle_wall_s"], emitted_tokens=c["emitted_tokens"],
        spec_stats={"num_spec_tokens": 3, "proposed": 3, "accepted": 1}) for c in producer.pop("cycles")]
    for trace in producer["requests"].values():
        del trace["first_scheduled_position"]
        trace["num_cached_tokens"] = 0
    producer_report = summarize(producer, case)
    assert producer_report["per_request"] == report["per_request"]
    assert producer_report["common_decode_overlap"] == report["common_decode_overlap"]
    assert producer_report["cycles"] is None and producer_report["mean_decode_cycle_ms"] is None
    assert producer_report["accepted"] == 4 and producer_report["proposed"] == 12
    # No-MTP needs an explicit runtime witness and zero proposal counters;
    # absent producer speculative stats are expected in this mode only.
    no_mtp_case = dict(case, mtp_depth=0)
    target_only = json.loads(json.dumps(raw))
    target_only["launch"] = {"mtp_depth": 0}
    target_report = summarize(target_only, no_mtp_case)
    assert target_report["mtp_depth"] == 0 and target_report["draft_acceptance"] is None
    producer_only = json.loads(json.dumps(producer))
    producer_only["mtp_depth"] = 0
    for batch in producer_only["frontend_batches"]:
        batch["spec_stats"] = None
    producer_only_report = summarize(producer_only, no_mtp_case)
    assert producer_only_report["per_request"] == target_report["per_request"]
    assert producer_only_report["common_decode_overlap"] == target_report["common_decode_overlap"]
    assert producer_only_report["accepted"] == producer_only_report["proposed"] == 0
    for invalid in (producer, dict(target_only, launch={"mtp_depth": 3})):
        try:
            summarize(invalid, no_mtp_case)
        except AssertionError:
            pass
        else:
            raise AssertionError("MTP runtime must not be scored as no-MTP")
    bounded_case = dict(no_mtp_case, max_model_len=32768)
    target_only["launch"]["max_model_len"] = 32768
    producer_only["requested_engine_args"] = {"max_model_len": 32768}
    assert summarize(target_only, bounded_case)["mtp_depth"] == 0
    assert summarize(producer_only, bounded_case)["mtp_depth"] == 0
    target_only["launch"]["max_model_len"] = 262144
    try:
        summarize(target_only, bounded_case)
    except AssertionError:
        pass
    else:
        raise AssertionError("different context limits must not be scored together")
    producer["requests"]["a"]["num_cached_tokens"] = 16
    try:
        summarize(producer, case)
    except AssertionError:
        pass
    else:
        raise AssertionError("producer cache hits cannot qualify cold timing")
    raw["requests"]["a"]["finish_reason"] = "error"
    try:
        summarize(raw, case)
    except AssertionError:
        pass
    else:
        raise AssertionError("generation errors cannot qualify timing")
    print("PASS: both frontend contracts, common-interval tokens, chunk TPOT, E2E, cold-cache/error guards; no invented producer cycles")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--prompt-4k", required=True)
    p.add_argument("--prompt-32k", required=True)
    p.add_argument("--c4-prompts", required=True)
    p.add_argument("--out", required=True)
    p = sub.add_parser("report")
    p.add_argument("--workload", required=True)
    p.add_argument("--raw", required=True)
    p.add_argument("--out", required=True)
    sub.add_parser("self-check")
    args = parser.parse_args()
    if args.command == "prepare":
        write_new(args.out, prepare(args.prompt_4k, args.prompt_32k, args.c4_prompts))
    elif args.command == "self-check":
        self_check()
    else:
        raw = read(args.raw)
        case = next(c for c in read(args.workload)["cases"] if c["id"] == raw["case"])
        report = summarize(raw, case)
        report["workload_identity"] = identity(args.workload)
        report["raw_identity"] = identity(args.raw)
        write_new(args.out, report)
        print(json.dumps({k: report[k] for k in ("case", "end_to_end_wall_s", "accepted", "proposed")}))


if __name__ == "__main__":
    main()
