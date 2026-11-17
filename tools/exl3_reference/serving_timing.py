#!/usr/bin/env python3
"""Pinned producer's public-engine R11 timing, isolated from native execution.

Run in the frozen oracle image. Frontend delivery batches are deliberately not
reported as target forwards, graph replays or GPU MTP-cycle latency.
"""
import argparse
import dataclasses
import hashlib
import json
import os
import resource
import time
from pathlib import Path

from capture_runtime_layout import PROFILE, verify_inputs
from runtime_layout import describe


def identity(path):
    path = Path(path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


class ServingTimingWorker:
    """Public worker RPC for allocations and optional separate diagnostics."""
    def serving_timing_enable_timeline(self, limit):
        """Separate diagnostic: observe actual pure-Q4 target and MTP calls."""
        import torch
        assert type(limit) is int and 1 <= limit <= 4
        assert not hasattr(self, "_serving_timeline")
        runner = self.model_runner
        assert runner.speculator is not None
        self._serving_timeline = []
        self._serving_timeline_pending = None
        self._serving_timeline_targets = 0
        execute = runner.execute_model
        compute_logits = runner.model.compute_logits
        propose = runner.speculator.propose

        def observe(stage, call, args, kwargs, shape):
            begin = torch.xpu.Event(enable_timing=True)
            end = torch.xpu.Event(enable_timing=True)
            with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                   torch.profiler.ProfilerActivity.XPU]) as prof:
                begin.record()
                result = call(*args, **kwargs)
                end.record()
                torch.xpu.synchronize()
            events = [{"name": e.name, "us": e.time_range.elapsed_us(),
                       "device_type": str(e.device_type)} for e in prof.events()
                      if str(e.device_type).endswith("XPU")]
            assert events, "no executed original XPU timeline events"
            self._serving_timeline.append(dict(stage=stage, shape=shape, device_events=events,
                                              queue_span_ms=begin.elapsed_time(end)))
            return result

        def target(*args, **kwargs):
            scheduled = args[0] if args else kwargs["scheduler_output"]
            rows = scheduled.num_scheduled_tokens
            if (self._serving_timeline_targets >= limit or not rows or
                    not scheduled.scheduled_spec_decode_tokens or
                    not all(count == 4 for count in rows.values())):
                return execute(*args, **kwargs)
            shape = {"num_requests": len(rows), "actual_rows": sum(rows.values()),
                     "scheduled_rows": dict(rows),
                     "proposed_lengths": {k: len(v) for k, v in
                                          scheduled.scheduled_spec_decode_tokens.items()}}
            result = observe("target_execute_model", execute, args, kwargs, shape)
            self._serving_timeline_targets += 1
            self._serving_timeline_pending = shape
            return result

        def draft(*args, **kwargs):
            shape = self._serving_timeline_pending
            if shape is None:
                return propose(*args, **kwargs)
            self._serving_timeline_pending = None
            return observe("mtp_propose", propose, args, kwargs, shape)

        def head(*args, **kwargs):
            shape = self._serving_timeline_pending
            if shape is None:
                return compute_logits(*args, **kwargs)
            hidden = args[0] if args else kwargs["hidden_states"]
            actual = dict(shape, head_rows=hidden.shape[0], hidden_dtype=str(hidden.dtype))
            return observe("target_compute_logits", compute_logits, args, kwargs, actual)

        runner.execute_model = target
        runner.model.compute_logits = head
        runner.speculator.propose = draft
        return {"enabled": True, "limit": limit,
                "scope": "Separate profiling only; actual calls/outputs, no tensor replacement"}

    def serving_timing_collect_timeline(self):
        assert self._serving_timeline_targets > 0
        assert self._serving_timeline_pending is None
        assert len(self._serving_timeline) == 3 * self._serving_timeline_targets
        return self._serving_timeline

    def serving_timing_snapshot(self):
        import torch
        torch.xpu.synchronize()
        runner = self.model_runner
        spec = runner.speculative_config
        return {"model_dtype": str(runner.dtype), "kv_dtype": str(runner.kv_cache_dtype),
                "cache_dtype_policy": runner.cache_config.cache_dtype,
                "speculation": None if spec is None else {"method": spec.method, "tokens": spec.num_speculative_tokens,
                    "draft_sample_method": spec.draft_sample_method,
                    "rejection_sample_method": spec.rejection_sample_method},
                "kv_cache_config": describe(runner.kv_cache_config),
                "speculative_config": describe(runner.speculative_config),
                "compilation_config": describe(runner.compilation_config),
                "memory_allocated_bytes": torch.xpu.memory_allocated(),
                "memory_reserved_bytes": torch.xpu.memory_reserved(),
                "peak_allocated_bytes": torch.xpu.max_memory_allocated(),
                "peak_reserved_bytes": torch.xpu.max_memory_reserved(),
                "host_peak_rss_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024}


def drive(engine, specs, sampling_factory, warmup=False, allow_prefix_hits=False):
    """Actual emitted feedback/arrival policy; no reference token injection."""
    traces, emitted, finished, ids = {}, {}, set(), {}
    batches, stats = [], []
    original_get_output = engine.engine_core.get_output

    def observe_output():
        output = original_get_output()
        scheduler = output.scheduler_stats
        spec = scheduler.spec_decoding_stats if scheduler is not None else None
        stats.append(None if spec is None else {
            "num_spec_tokens": spec.num_spec_tokens, "proposed": spec.num_draft_tokens,
            "accepted": spec.num_accepted_tokens, "drafts": spec.num_drafts})
        return output

    # Read the already-produced scalar scheduler stats; no model/kernel hook,
    # extra device wait or IPC request inside the measured loop.
    if not warmup:
        engine.engine_core.get_output = observe_output
    began = time.perf_counter()
    seconds = lambda: time.perf_counter() - began

    def admit():
        for request in specs:
            key = request["id"]
            if key in traces:
                continue
            trigger = request.get("after_emitted")
            if trigger and emitted.get(trigger["request"], 0) < trigger["tokens"]:
                continue
            external = ("warmup-" if warmup else "") + key
            cap = 32 if warmup else request["output_tokens"]
            arrived = seconds()
            internal = engine.add_request(external,
                {"prompt_token_ids": request["prompt_ids"]}, sampling_factory(cap))
            ids[external] = ids[internal] = key
            traces[key] = {"prompt_ids": request["prompt_ids"], "output_limit": cap,
                           "admitted_at_s": arrived, "observations": []}

    try:
        admit()
        for cycle in range(4096):
            if not engine.has_unfinished_requests():
                break
            start = seconds()
            before = len(stats)
            outputs = engine.step()
            end = seconds()
            count = 0
            for output in outputs:
                key = ids[output.request_id]
                trace = traces[key]
                assert output.prompt_token_ids == trace["prompt_ids"]
                if allow_prefix_hits:
                    assert 0 <= output.num_cached_tokens < len(trace["prompt_ids"])
                    if "num_cached_tokens" in trace:
                        assert trace["num_cached_tokens"] == output.num_cached_tokens
                else:
                    assert output.num_cached_tokens == 0, (key, output.num_cached_tokens)
                trace["num_cached_tokens"] = output.num_cached_tokens
                assert len(output.outputs) == 1
                completion = output.outputs[0]
                n = len(completion.token_ids)
                delta = n - emitted.get(key, 0)
                assert delta >= 0
                count += delta
                emitted[key] = n
                if delta:
                    trace["observations"].append({"at_s": end, "tokens": n, "new_tokens": delta})
                if output.finished:
                    trace.update(finished_at_s=end, ids=list(completion.token_ids),
                                 finish_reason=completion.finish_reason)
                    finished.add(key)
            if not warmup:
                assert len(stats) == before + 1
                batches.append({"batch": cycle, "start_s": start, "end_s": end,
                                "frontend_wall_s": end - start, "emitted_tokens": count,
                                "spec_stats": stats[-1]})
            admit()
        duration = seconds()
        assert len(finished) == len(specs) and not engine.has_unfinished_requests()
        for trace in traces.values():
            assert len(trace["ids"]) == trace["output_limit"]
            assert trace["finish_reason"] == "length"
            if not allow_prefix_hits:
                assert trace["num_cached_tokens"] == 0
        return {"requests": traces, "frontend_batches": batches, "end_to_end_wall_s": duration}
    finally:
        if not warmup:
            engine.engine_core.get_output = original_get_output


def capture(args):
    assert not args.output.exists(), "refusing to overwrite producer output"
    workload = json.loads(args.workload.read_text())
    assert workload["schema"] == "b70-exl3-r11-serving-workload-v1"
    assert workload["sampling"] == {"temperature": 0.0, "ignore_eos": True}
    task = next(c for c in workload["cases"] if c["id"] == args.case)
    assert task["cache_state"] == "cold" and 1 <= len(task["requests"]) <= 4
    depth = task.get("mtp_depth", 3)
    assert type(depth) is int and depth in (0, 3)
    context_limit = task.get("max_model_len", 262144)
    assert type(context_limit) is int and 0 < context_limit <= 262144
    assert all(len(r["prompt_ids"]) + r["output_tokens"] <= context_limit for r in task["requests"])
    assert not args.timeline_cycles or depth == 3, "Q4/MTP timeline requires MTP3"
    assert not args.timeline_eager or args.timeline_cycles, "eager diagnostic needs timeline cycles"
    prefix_pair = getattr(args, "prefix_pair", False)
    if prefix_pair:
        assert depth == 3 and len(task["requests"]) == 1 and not args.timeline_cycles
        assert len(task["requests"][0]["prompt_ids"]) == 32768
        assert task["requests"][0]["output_tokens"] == 64
    reference = verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    pinned_subset = Path(os.environ["EXL3_DRAFT_VOCAB"])
    assert identity(pinned_subset)["sha256"] == identity(args.subset)["sha256"]
    os.environ["EXL3_DRAFT_VOCAB"] = str(args.subset)
    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["EXL3_LOADER_REPORT_DIR"] = str(args.output.parent / (args.output.stem + "-loader"))
    from vllm import LLM, SamplingParams
    from vllm.config import ReasoningConfig
    from vllm.engine.arg_utils import EngineArgs
    from vllm.sampling_params import RequestOutputKind
    import torch
    assert torch.__version__ == "2.13.0+xpu"
    assert "B70" in torch.xpu.get_device_name(0)
    allowed = {f.name for f in dataclasses.fields(EngineArgs)}
    kwargs = {k: v for k, v in profile["vllm"].items() if k in allowed}
    if isinstance(kwargs.get("reasoning_config"), dict):
        kwargs["reasoning_config"] = ReasoningConfig(**kwargs["reasoning_config"])
    overrides = {"max_model_len": context_limit, "max_num_seqs": 4,
                 "max_num_batched_tokens": 1600, "num_gpu_blocks_override": 180,
                 "async_scheduling": False, "disable_log_stats": False,
                 "compilation_config": {"cudagraph_mode": "FULL_DECODE_ONLY",
                                        "cudagraph_capture_sizes": [1, 2, 4, 8, 12, 16]}}
    if args.timeline_eager:
        overrides.update(enforce_eager=True, compilation_config={"cudagraph_mode": "NONE"})
    if depth == 0:
        overrides["speculative_config"] = None
        # Keep the primary physical attention page for target-only isolation;
        # without speculation the automatic Mamba-derived page would shrink.
        overrides["block_size"] = 1600
    kwargs.update(overrides, model=str(args.model_dir),
                  worker_extension_cls="serving_timing.ServingTimingWorker")
    assert kwargs["dtype"] == "float16" and kwargs["kv_cache_dtype"] == "fp8"
    assert kwargs["speculative_config"] == ({"method": "mtp", "num_speculative_tokens": 3} if depth else None)
    config = EngineArgs(**kwargs).create_engine_config()
    result = {"schema": "b70-exl3-r11-producer-serving-output-v1", "case": args.case,
              "cache_state": "cold", "profiled": bool(args.timeline_cycles), "mtp_depth": depth,
              "startup_state": "request_warmup_o32_then_prefix_reset",
              "sampling": {"temperature": 0.0, "ignore_eos": True},
              "image": args.image_identity, "checkpoint": reference["checkpoint"]["identity"],
              "profile_identity": identity(PROFILE), "tool_identity": identity(__file__),
              "workload_identity": identity(args.workload), "subset_identity": identity(args.subset),
              "profile_overrides": overrides, "requested_engine_args": describe(kwargs),
              "resolved_config": {"model": describe(config.model_config),
                  "cache": describe(config.cache_config), "scheduler": describe(config.scheduler_config),
                  "speculative": describe(config.speculative_config),
                  "compilation": describe(config.compilation_config)},
              "scope": "Actual public frontend chunk timestamps. Frontend batches are not GPU MTP cycles. "
                       "Constructor startup and identical O32 request warmup are excluded; prefix reset precedes scoring."}
    if prefix_pair:
        result["prefix_pair_requested"] = True
    if args.config_only:
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
        print("CONFIG_ONLY_PASS", args.output, flush=True)
        return
    llm = None
    try:
        llm = LLM(**kwargs)
        engine = llm.llm_engine
        sample = lambda cap: SamplingParams(temperature=0, max_tokens=cap, ignore_eos=True,
                                            output_kind=RequestOutputKind.CUMULATIVE)
        # Constructor warmup is producer-owned; perform the same additional
        # real-request O32 warmup as native, then restore cold prefix state.
        assert engine.reset_prefix_cache()
        warm = drive(engine, task["requests"], sample, warmup=True)
        assert engine.reset_prefix_cache()
        result["warmup"] = {"enabled": True, "output_tokens_per_request": 32,
                            "requests": len(warm["requests"]), "prefix_reset_succeeded": True}
        result["initialized_worker"] = llm.collective_rpc("serving_timing_snapshot", timeout=60)
        assert len(result["initialized_worker"]) == 1
        with args.output.with_suffix(".initialized.json").open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
        worker = result["initialized_worker"][0]
        assert worker["model_dtype"] == "torch.float16"
        assert worker["cache_dtype_policy"] == "fp8" and worker["kv_dtype"] == "torch.uint8"
        if depth:
            assert worker["speculation"]["method"] == "mtp" and worker["speculation"]["tokens"] == 3
        else:
            assert worker["speculation"] is None and worker["speculative_config"] is None
        assert worker["kv_cache_config"]["fields"]["num_blocks"] == 180
        if args.timeline_cycles:
            result["timeline_install"] = llm.collective_rpc(
                "serving_timing_enable_timeline", args=(args.timeline_cycles,), timeout=60)
        result.update(drive(engine, task["requests"], sample))
        if prefix_pair:
            first = task["requests"][0]
            repeat = dict(first, id="request-repeat")
            result["prefix_repeat"] = drive(engine, [repeat], sample, allow_prefix_hits=True)
            cold = result["requests"][first["id"]]
            warm = result["prefix_repeat"]["requests"][repeat["id"]]
            result["prefix_pair_checks"] = {
                "cold_repeat_ids_exact": cold["ids"] == warm["ids"],
                "repeat_cached_tokens": warm["num_cached_tokens"],
                # The pinned MTP full-attention manager drops the final
                # matched page: draft KV depends on the next input token.
                "expected_aligned_cached_tokens": max(0, (len(first["prompt_ids"]) - 1) // 1600 - 1) * 1600}
        if args.timeline_cycles:
            result["diagnostic_timeline"] = llm.collective_rpc(
                "serving_timing_collect_timeline", timeout=60)
            result["scope"] += (" Separate actual Q4 target/propose profiler observers enabled; "
                                "profiled frontend times are not serving scores. "
                                "Device kernel durations can overlap and are not additive latency.")
        result["finished_worker"] = llm.collective_rpc("serving_timing_snapshot", timeout=60)
        result["frontend_host_peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024
        final = result["finished_worker"][0]
        assert final["peak_allocated_bytes"] <= 32 << 30
        assert final["host_peak_rss_bytes"] + result["frontend_host_peak_rss_bytes"] <= 32 << 30
        if depth:
            assert any(b["spec_stats"] and b["spec_stats"]["proposed"] > 0
                       for b in result["frontend_batches"]), "MTP proposal witness missing"
        else:
            assert all(b["spec_stats"] is None or
                       (b["spec_stats"]["num_spec_tokens"] == 0 and
                        b["spec_stats"]["proposed"] == b["spec_stats"]["accepted"] == 0)
                       for b in result["frontend_batches"]), "target-only speculative statistics"
        with args.output.open("x") as f:
            json.dump(result, f, indent=2, allow_nan=False); f.write("\n")
        if prefix_pair:
            # Retain the raw observations even when the independent prefix
            # qualification fails; the actual worker exit then remains nonzero.
            checks = result["prefix_pair_checks"]
            assert checks["cold_repeat_ids_exact"], "original cold/repeated prefix IDs differ"
            assert checks["repeat_cached_tokens"] == checks["expected_aligned_cached_tokens"]
        print("PRODUCER_SERVING_PASS", args.output, result["end_to_end_wall_s"], flush=True)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("model-dir", "reference-manifest", "subset", "workload", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--image-identity", required=True)
    p.add_argument("--case", required=True)
    p.add_argument("--config-only", action="store_true")
    p.add_argument("--prefix-pair", action="store_true",
                   help="After one coldP32768/O64 MTP3 request, repeat its exact prefix without reset")
    p.add_argument("--timeline-cycles", type=int, choices=range(5), default=0,
                   help="Separate profiler diagnostic for up to four actual pure-Q4 target/MTP calls")
    p.add_argument("--timeline-eager", action="store_true",
                   help="Disable compilation and graphs for the explicitly separate timeline diagnostic")
    capture(p.parse_args())


if __name__ == "__main__":
    main()
