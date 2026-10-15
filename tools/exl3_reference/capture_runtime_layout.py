#!/usr/bin/env python3
"""Capture real initialized cache layouts in an isolated pinned vLLM worker.

Uses the supported worker-extension RPC and a read-only prefill observer. It
does not replace kernels, read cache capacity contents, or serve an HTTP API.
"""
import argparse
import dataclasses
import functools
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import struct
import sys

from capture_projection import IMAGE, LIB_SHA, OPS_SHA
from extract_projection import digest, headers
from runtime_layout import attention_call_layout, describe, initialized_values, tensor_layout

PROFILE = Path("/opt/exl3xpu/models/qwen3.8-27b-exl3-4.00bpw/migration-target-optimized.yaml")
VLLM_ROOT = Path("/opt/venv/lib/python3.12/site-packages/vllm")


def verify_inputs(reference_path, model_dir, image_identity):
    headers.require(image_identity == IMAGE, "requires pinned production image")
    reference = headers.read_json(reference_path)["reference_B"]
    for relative, expected in (("exl3xpu/_C.so", LIB_SHA), ("exl3xpu/ops.py", OPS_SHA)):
        headers.require(digest((Path("/opt/exl3xpu") / relative).read_bytes()) == expected,
                        f"runtime hash mismatch: {relative}")
    guards = reference["active_guards"]["records"]
    headers.require(len(guards) == 13, "requires complete active guard union")
    for guard in guards:
        headers.require(digest((VLLM_ROOT / guard["guarded_path"]).read_bytes()) == guard["expected_sha256"],
                        f"installed guard mismatch: {guard['guarded_path']}")
    checkpoint = reference["checkpoint"]
    headers.require(checkpoint["packed_byte_identity"]["status"] == "verified", "complete checkpoint hashes required")
    for metadata in checkpoint["metadata_files"]:
        headers.require(digest((model_dir / Path(metadata["path"]).name).read_bytes()) == metadata["sha256"],
                        "checkpoint metadata hash mismatch")
    return reference


def cache_layers(runner):
    """Read actual layer views bound by init_kv_cache, never infer from config."""
    context = runner.compilation_config.static_forward_context
    records = []
    for group_id, group in enumerate(runner.kv_cache_config.kv_cache_groups):
        for name in group.layer_names:
            layer = context[name]
            cache = getattr(layer, "kv_cache", None)
            if cache is None:
                raise ValueError(f"cache not bound: {name}")
            scales = {}
            for attr in ("_k_scale", "_v_scale", "_q_scale", "_k_scale_float", "_v_scale_float"):
                value = getattr(layer, attr, None)
                if hasattr(value, "untyped_storage"):
                    scales[attr] = initialized_values(value, max_elements=1)
                elif value is not None:
                    scales[attr] = describe(value)
            records.append({"layer": name, "group": group_id,
                            "class": type(layer).__module__ + "." + type(layer).__qualname__,
                            "cache": describe(cache), "scales": scales})
    return records


def active_metadata(metadata):
    """Copy only initialized active views; unused block-table columns stay unread."""
    result = {"description": describe(metadata), "active_values": {}}
    if metadata is None:
        return result
    names = ("query_start_loc", "seq_lens", "slot_mapping", "non_spec_query_start_loc", "spec_query_start_loc",
             "non_spec_state_indices_tensor", "spec_state_indices_tensor", "has_initial_state",
             "prefill_query_start_loc", "prefill_state_indices", "prefill_has_initial_state",
             "num_accepted_tokens", "spec_token_indx", "non_spec_token_indx", "spec_sequence_masks")
    # These metadata fields are built as active views by the pinned builders;
    # unlike block tables, no capacity-only tail is copied here.
    for name in names:
        value = getattr(metadata, name, None)
        if hasattr(value, "untyped_storage") and value.numel() <= 4096:
            result["active_values"][name] = initialized_values(value)
    for name in ("block_table", "block_table_tensor"):
        value = getattr(metadata, name, None)
        if hasattr(value, "untyped_storage") and value.ndim == 2 and value.shape[1] > 0:
            result["active_values"][name + "_first_column"] = initialized_values(value[:, :1])
            result["block_table_value_scope"] = "first column only; unused capacity columns not read"
    return result


class RuntimeLayoutCapture:
    """Mixin injected by vLLM's supported worker_extension_cls mechanism."""
    def capture_exl3_layout(self, phase):
        import torch
        runner = self.model_runner
        torch.xpu.synchronize()
        spec = runner.speculative_config
        return {"phase": phase, "worker_pid": os.getpid(),
                "runner_class": type(runner).__module__ + "." + type(runner).__qualname__,
                "device": str(runner.device), "model_dtype": str(runner.dtype),
                "kv_dtype": str(runner.kv_cache_dtype), "kernel_block_sizes": describe(runner.kernel_block_sizes),
                "kv_cache_config": describe(runner.kv_cache_config), "layers": cache_layers(runner),
                "speculation": {"method": spec.method, "tokens": spec.num_speculative_tokens,
                                "draft_sample_method": spec.draft_sample_method,
                                "rejection_sample_method": spec.rejection_sample_method,
                                "draft_logits": describe(getattr(runner.speculator, "draft_logits", None))},
                "memory_allocated": torch.xpu.memory_allocated(), "memory_reserved": torch.xpu.memory_reserved(),
                "prefill_observations": getattr(self, "_exl3_layout_observations", []),
                "attention_call_observations": getattr(self, "_exl3_attention_observations", [])}

    def install_exl3_prefill_observer(self):
        from vllm.forward_context import get_forward_context
        from vllm.v1.attention.backends import flash_attn as fa
        runner = self.model_runner
        self._exl3_layout_observations = []
        self._exl3_attention_observations = []
        headers.require(getattr(fa, "_exl3_guarded_attention", False), "guarded attention must already be installed")
        original_attention = fa.flash_attn_varlen_func
        layer_by_pointer = {layer["cache"]["tensor_layout"]["data_ptr"]: layer["layer"]
                            for layer in cache_layers(runner)
                            if "tensor_layout" in layer["cache"]}

        @functools.wraps(original_attention)
        def observe_attention(*args, **kwargs):
            # Installed after warmup; only observe calls belonging to the first
            # real target prefill. Delegate the original arguments unchanged.
            if self._exl3_layout_observations and len(self._exl3_attention_observations) < 16:
                headers.require(not args, "expected keyword-only guarded attention call")
                record = attention_call_layout(kwargs)
                record["layer"] = layer_by_pointer.get(kwargs["k"].data_ptr())
                headers.require(record["layer"] is not None, "attention K view does not match bound layer")
                self._exl3_attention_observations.append(record)
            return original_attention(*args, **kwargs)

        self._exl3_original_attention = original_attention
        fa.flash_attn_varlen_func = observe_attention

        def observe(module, args, kwargs):
            if self._exl3_layout_observations:
                return None
            context = get_forward_context()
            metadata = context.attn_metadata
            if not metadata:
                return None
            req_states = getattr(runner, "req_states", None)
            request_map = dict(getattr(req_states, "req_id_to_index", {}))
            if not request_map:
                return None
            self._exl3_layout_observations.append({
                "phase": "first_real_target_prefill_before_forward",
                "request_id_to_index": request_map,
                "attention_metadata": {name: active_metadata(meta) for name, meta in metadata.items()},
                "layers": cache_layers(runner)})
            return None

        self._exl3_layout_hook = runner.model.register_forward_pre_hook(observe, with_kwargs=True)
        return {"observer_installed": True, "arithmetic_change": False}

    def remove_exl3_prefill_observer(self):
        from vllm.v1.attention.backends import flash_attn as fa
        self._exl3_layout_hook.remove()
        fa.flash_attn_varlen_func = self._exl3_original_attention
        return {"observer_removed": True}


def capture(args):
    headers.require(not args.output.exists(), "refusing to overwrite runtime layout capture")
    reference = verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["EXL3_LOADER_REPORT_DIR"] = str(args.output.parent / "loader")
    from vllm import LLM, SamplingParams
    from vllm.config import ReasoningConfig
    from vllm.engine.arg_utils import EngineArgs
    import torch

    allowed = {field.name for field in dataclasses.fields(EngineArgs)}
    kwargs = {key: value for key, value in profile["vllm"].items() if key in allowed}
    if isinstance(kwargs.get("reasoning_config"), dict):
        kwargs["reasoning_config"] = ReasoningConfig(**kwargs["reasoning_config"])
    excluded = sorted(set(profile["vllm"]) - allowed)
    headers.require(torch.__version__ == "2.13.0+xpu", "unexpected Torch runtime")
    headers.require("B70" in torch.xpu.get_device_name(0), "requires B70")
    kwargs.update(model=str(args.model_dir), worker_extension_cls="capture_runtime_layout.RuntimeLayoutCapture")
    print("S0_LAYOUT_START", json.dumps({"engine_args": describe(kwargs), "excluded_HTTP_only_arguments": excluded}), flush=True)
    if args.config_only:
        config = EngineArgs(**kwargs).create_engine_config()
        print("S0_LAYOUT_CONFIG_PASS", json.dumps({"dtype": str(config.model_config.dtype),
              "draft_sample_method": config.speculative_config.draft_sample_method,
              "cache_dtype": config.cache_config.cache_dtype,
              "graph_mode": str(config.compilation_config.cudagraph_mode)}), flush=True)
        return
    llm = None
    try:
        llm = LLM(**kwargs)
        initialized = llm.collective_rpc("capture_exl3_layout", timeout=60, args=("initialized",))
        llm.collective_rpc("install_exl3_prefill_observer", timeout=60)
        tokens = [1000 + (i * 37) % 4096 for i in range(128)]
        output = llm.generate({"prompt_token_ids": tokens},
                              SamplingParams(temperature=0, max_tokens=1, ignore_eos=True), use_tqdm=False)
        after = llm.collective_rpc("capture_exl3_layout", timeout=60, args=("after_P128_O1",))
        llm.collective_rpc("remove_exl3_prefill_observer", timeout=60)
        headers.require(len(after) == 1 and len(after[0]["prefill_observations"]) == 1,
                        "missing real target prefill metadata capture")
        headers.require(len(after[0]["attention_call_observations"]) == 16,
                        "missing actual target attention arguments")
        result = {"schema": 1, "kind": "pinned_worker_actual_layout", "image": args.image_identity,
                  "at": datetime.now(timezone.utc).isoformat(), "device": torch.xpu.get_device_name(0),
                  "reference_manifest_sha256": digest(args.reference_manifest.read_bytes()),
                  "capture_tool_sha256": digest(Path(__file__).read_bytes()),
                  "layout_helper_sha256": digest(Path(__file__).with_name("runtime_layout.py").read_bytes()),
                  "profile_sha256": digest(PROFILE.read_bytes()), "engine_args": describe(kwargs),
                  "excluded_HTTP_only_arguments": excluded, "checkpoint": reference["checkpoint"]["identity"],
                  "initialized": initialized, "after": after, "prompt_token_ids": tokens,
                  "prompt_ids_le_i32_sha256": digest(struct.pack("<128i", *tokens)),
                  "output_token_ids": [list(o.outputs[0].token_ids) for o in output],
                  "scope": "Actual initialized allocations plus one synthetic P128/O1 prefill metadata observation; no cache payload/state parity or benchmark."}
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x") as stream:
            json.dump(result, stream, indent=2, allow_nan=False)
            stream.write("\n")
        print("S0_LAYOUT_CAPTURED", args.output, digest(args.output.read_bytes()), flush=True)
    finally:
        if llm is not None:
            llm.llm_engine.engine_core.shutdown()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-only", action="store_true", help="validate frozen profile conversion without loading a worker")
    args = parser.parse_args()
    try:
        capture(args)
    except (headers.InventoryError, OSError, ValueError, RuntimeError, KeyError) as exc:
        print(f"capture_runtime_layout: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
