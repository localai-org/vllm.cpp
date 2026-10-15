#!/usr/bin/env python3
"""Replay the resolved pinned SwiGLU method on real layer0 Gate/Up outputs.

Oracle-only Torch. No model worker, dense weight reconstruction or serving.
The exact profile resolves CustomOp selection; this is an eager operator gate,
not qualification of compiled graphs or a complete MLP/block.
"""
import argparse
import dataclasses
import json
import os
from pathlib import Path

from capture_projection import IMAGE, LIB_SHA, OPS_SHA
from capture_runtime_layout import PROFILE, VLLM_ROOT, verify_inputs
from extract_projection import digest, headers, write_safetensors

SOURCE_SHA = "3a12014b2132f39a3a8d43a362b40f9b7a1784c0e56806f489297d5ff647fbe1"
ROWS = (1, 4, 12, 16, 128)


def checked_source(path):
    receipt = headers.read_json(path.with_suffix(".json"))
    headers.require(digest(path.read_bytes()) == receipt["capture_sha256"] == SOURCE_SHA,
                    "requires immutable layer0 full Gate/Up capture")
    headers.require((receipt["image"], receipt["library_sha256"], receipt["ops_sha256"],
                     receipt["k"], receipt["n"], receipt["groups"], receipt["bits"],
                     receipt["bounds"]) ==
                    (IMAGE, LIB_SHA, OPS_SHA, 5120, 34816, 2, 4, [0, 17408, 34816]),
                    "source runtime/geometry mismatch")
    report = headers.read_shard_header(path)
    entries = {t["name"]: t for t in report["tensors"]}
    headers.require(entries.keys() == receipt["tensor_hashes"].keys(), "source tensor set mismatch")
    with path.open("rb") as stream:
        for name, t in entries.items():
            spec = receipt["tensor_hashes"][name]
            headers.require((t["dtype"], t["shape"]) == (spec["dtype"], spec["shape"]),
                            "source tensor geometry mismatch: " + name)
            begin, end = t["data_offsets"]
            stream.seek(8 + report["header_bytes"] + begin)
            headers.require(digest(stream.read(end - begin)) == spec["sha256"],
                            "source tensor hash mismatch: " + name)
    return receipt


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix(".json").exists(),
                    "refusing to overwrite capture")
    checked_source(args.source)
    verify_inputs(args.reference_manifest, args.model_dir, args.image_identity)
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    os.environ["HF_HUB_OFFLINE"] = "1"
    import torch
    from safetensors.torch import load_file
    from vllm.config import ReasoningConfig, set_current_vllm_config
    from vllm.engine.arg_utils import EngineArgs
    from vllm.model_executor.layers.activation import SiluAndMul

    headers.require(torch.__version__ == "2.13.0+xpu" and torch.xpu.is_available() and
                    "B70" in torch.xpu.get_device_name(0), "requires pinned Torch XPU and B70")
    allowed = {f.name for f in dataclasses.fields(EngineArgs)}
    kwargs = {k: v for k, v in profile["vllm"].items() if k in allowed}
    if isinstance(kwargs.get("reasoning_config"), dict):
        kwargs["reasoning_config"] = ReasoningConfig(**kwargs["reasoning_config"])
    kwargs["model"] = str(args.model_dir)
    config = EngineArgs(**kwargs).create_engine_config()
    host = load_file(str(args.source), device="cpu")
    tensors, observations = {}, []

    def save(name, t):
        t = t.detach().cpu().contiguous()
        headers.require(t.dtype == torch.float16 and torch.isfinite(t).all().item(),
                        "nonfinite or wrong precision oracle stage: " + name)
        tensors[name] = ("F16", list(t.shape), t.numpy().tobytes())

    with set_current_vllm_config(config):
        layer = SiluAndMul(compile_native=False)
        selected = layer._forward_method.__name__
        headers.require(selected in ("forward_xpu", "forward_native"), "unexpected SwiGLU method")
        for m in ROWS:
            x = host[f"m{m}_production_output_f16"].to("xpu")
            headers.require(list(x.shape) == [m, 34816], "wrong Gate/Up operand")
            out = layer(x)
            save(f"m{m}_gate_up_f16", x)
            save(f"m{m}_production_swiglu_f16", out)
            # Diagnostics identify the rounding contract; neither replaces out.
            gate, up = x.chunk(2, dim=-1)
            rounded = torch.nn.functional.silu(gate) * up
            fused = (torch.nn.functional.silu(gate.float()) * up.float()).half()
            observations.append({"m": m, "rounded_silu_vs_producer_bit_exact":
                                 torch.equal(rounded.view(torch.int16), out.view(torch.int16)),
                                 "unrounded_silu_vs_producer_bit_exact":
                                 torch.equal(fused.view(torch.int16), out.view(torch.int16))})
            print("SWIGLU_CAPTURE", selected, observations[-1], flush=True)
    torch.xpu.synchronize()
    source_paths = [VLLM_ROOT / n for n in (
        "model_executor/models/qwen3_5.py", "model_executor/models/qwen2_moe.py",
        "model_executor/layers/activation.py", "model_executor/custom_op.py")]
    binary = VLLM_ROOT.parent / "vllm_xpu_kernels/_C.abi3.so"
    result = {"schema": 1, "kind": "pinned_eager_swiglu_capture", "image": IMAGE,
              "torch": torch.__version__, "device": torch.xpu.get_device_name(0),
              "source_capture_sha256": SOURCE_SHA, "profile_sha256": digest(PROFILE.read_bytes()),
              "custom_ops": config.compilation_config.custom_ops, "selected_method": selected,
              "source_hashes": {str(p): digest(p.read_bytes()) for p in source_paths},
              "xpu_kernel_binary_sha256": digest(binary.read_bytes()),
              "capture_tool_sha256": digest(Path(__file__).read_bytes()),
              "observations": observations,
              "tensor_hashes": {n: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                for n, t in sorted(tensors.items())},
              "scope": "Resolved eager SwiGLU method on immutable real layer0 Gate/Up projection outputs at M1/4/12/16/128. No compiled graph or full MLP/block/state parity."}
    write_safetensors(args.output, tensors, {"image": IMAGE, "source_capture_sha256": SOURCE_SHA})
    result["capture_sha256"] = digest(args.output.read_bytes())
    with args.output.with_suffix(".json").open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    print("SWIGLU_CAPTURE_DONE", args.output, result["capture_sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference-manifest", type=Path, required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
