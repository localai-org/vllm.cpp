#!/usr/bin/env python3
"""Observe installed Gemma dispatch on identical real FP16 norm operands.

This isolated operator diagnostic does not initialize a model, change an IR
provider, or replace the frozen literal-expression fixtures. The selected
installed operator, its forward_native method (which itself dispatches IR),
and the explicitly selected native IR implementation are recorded separately.
"""
import argparse
import inspect
import json
from pathlib import Path

from capture_projection import IMAGE, checked_runtime
from extract_projection import digest


def capture(args):
    if args.image_identity != IMAGE:
        raise ValueError("requires pinned original image")
    report = args.output.with_suffix(".json")
    if args.output.exists() or report.exists():
        raise ValueError("refusing to overwrite fixture or receipt")
    metadata = json.loads(args.fixture.with_suffix(".json").read_text())
    if digest(args.fixture.read_bytes()) != metadata["fixture_sha256"]:
        raise ValueError("frozen operand fixture SHA-256 mismatch")
    checked_runtime(Path("/opt/exl3xpu"))
    import torch
    from safetensors.torch import load_file, save_file
    from torch.utils._python_dispatch import TorchDispatchMode
    from vllm import ir
    from vllm.config import VllmConfig, set_current_vllm_config
    from vllm.model_executor.layers.layernorm import GemmaRMSNorm
    if torch.__version__ != "2.13.0+xpu" or "B70" not in torch.xpu.get_device_name(0):
        raise ValueError("requires pinned Torch and B70")
    source = load_file(str(args.fixture))
    tensors, cases = {}, []

    class Observe(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.calls = []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            self.calls.append(str(func))
            return func(*args, **(kwargs or {}))

    def compare(a, b):
        different = a.view(torch.int16) != b.view(torch.int16)
        indices = different.nonzero()
        return {"elements": a.numel(), "different_bits": int(different.sum()),
                "first_difference": indices[0].tolist() if len(indices) else None,
                "max_abs": float((a.float() - b.float()).abs().max())}

    with torch.inference_mode(), set_current_vllm_config(VllmConfig()):
        for entry in metadata["cases"]:
            label = entry["label"]
            x, w = (source[label + suffix].to("xpu") for suffix in ("_input", "_weight"))
            residual = source[label + "_residual"].to("xpu") if entry["residual"] else None
            layer = GemmaRMSNorm(5120).to(device="xpu", dtype=torch.float16)
            layer.weight.copy_(w)
            selected = getattr(layer, "_forward_method", None)
            selected_method = getattr(selected, "__qualname__", repr(selected))
            results, calls = {}, {}
            for name in ("forward", "forward_xpu", "forward_native", "ir_native"):
                xx = x.clone()
                rr = residual.clone() if residual is not None else None
                observer = Observe()
                with observer:
                    if name == "ir_native":
                        operation = ir.ops.fused_add_rms_norm if rr is not None else ir.ops.rms_norm
                        implementation = operation.impls["native"].impl_fn
                        operands = (xx, rr) if rr is not None else (xx,)
                        result = implementation(*operands, w.float() + 1.0, 1e-6)
                    else:
                        result = getattr(layer, name)(xx, rr)
                output, stored = result if rr is not None else (result, None)
                results[name] = (output.cpu().contiguous(),
                                 stored.cpu().contiguous() if stored is not None else None)
                calls[name] = observer.calls
            # Preserve the existing C++ row-fixture ABI for an independent
            # same-input test against the ordinary installed operation.
            for suffix in ("_input", "_weight", "_residual"):
                if label + suffix in source:
                    tensors[label + suffix] = source[label + suffix].clone()
            tensors[label + "_output"] = results["forward"][0].clone()
            if residual is not None:
                tensors[label + "_stored_residual"] = results["forward"][1].clone()
            comparisons = {}
            for name, (output, stored) in results.items():
                tensors[label + "_" + name + "_output"] = output
                comparisons[name] = {"against_literal": compare(output, source[label + "_output"]),
                                     "against_ir_native": compare(output, results["ir_native"][0])}
                if stored is not None:
                    tensors[label + "_" + name + "_stored_residual"] = stored
                    comparisons[name]["residual_against_literal"] = compare(
                        stored, source[label + "_stored_residual"])
                # Each invocation gets fresh inputs even for in-place kernels.
                # Record rather than infer the actual dispatched operator.
            cases.append({**entry, "selected_method": selected_method,
                          "calls": calls, "comparisons": comparisons})
            print(label, selected_method, comparisons["forward"], flush=True)
    save_file(tensors, str(args.output), {"scope": __doc__})
    args.output.chmod(0o644)
    inspected = {"GemmaRMSNorm": Path(inspect.getfile(GemmaRMSNorm))}
    for name in ("rms_norm", "fused_add_rms_norm"):
        inspected[name + "_native"] = Path(inspect.getfile(ir.ops.__getattribute__(name).impls["native"].impl_fn))
    receipt = {"schema": "b70-exl3-installed-gemma-materialization-v1",
               "scope": __doc__, "torch": torch.__version__, "cases": cases,
               "image_identity": args.image_identity, "epsilon": 1e-6,
               "configuration_scope": "Isolated default VllmConfig dispatch; forward_xpu is an "
                   "explicit diagnostic. The compiled serving-worker norm is not captured here.",
               "compiled_gemma_available": hasattr(torch.ops._C, "gemma_rms_norm"),
               "fixture_input": {"path": str(args.fixture), "sha256": digest(args.fixture.read_bytes())},
               "fixture_sha256": digest(args.output.read_bytes()),
               "capture_tool_sha256": digest(Path(__file__).read_bytes()),
               "installed_sources": {name: {"path": str(path), "sha256": digest(path.read_bytes())}
                                     for name, path in inspected.items()},
               "target_qualified": False, "serving_qualified": False}
    with report.open("x") as stream:
        json.dump(receipt, stream, indent=2)
        stream.write("\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
