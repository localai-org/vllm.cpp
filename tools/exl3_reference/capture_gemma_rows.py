#!/usr/bin/env python3
"""Pinned Torch Gemma expression on real layer0/layer3 operands, selected row forms.

This is an identical-operand operator reference, not a new full-worker capture.
The original P128 checkpoints are checked before deriving short/padded forms.
"""
import argparse
import hashlib
import json
from pathlib import Path


def capture(args):
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file, save_file
    report = args.output.with_suffix(".json")
    if args.output.exists() or report.exists():
        raise ValueError("refusing to overwrite Gemma row fixture or receipt")
    inputs = {layer: load_file(str(args.fixtures / name)) for layer, name in
              ((0, "real_block_oracle.safetensors"), (3, "real_attention3_oracle.safetensors"))}
    tensors, cases = {}, []
    with safe_open(str(args.model / "model-00001-of-00002.safetensors"), framework="pt", device="cpu") as model:
        for layer in (0, 3):
            source = inputs[layer]
            for residual in (False, True):
                post = layer == 0 and residual
                x = source["p128_post_norm_input" if post else "p128_hidden_in"].contiguous()
                res = source["p128_post_norm_residual_input" if post else "p128_residual_in"] if residual else None
                part = "post_attention" if post else "input"
                weight = model.get_tensor(f"model.language_model.layers.{layer}.{part}_layernorm.weight").half()
                def reference(x, res):
                    value = x.to("xpu").float()
                    if res is not None:
                        value = value + res.to("xpu").float()
                    inv = torch.rsqrt(value.pow(2).mean(-1, keepdim=True) + 1e-6)
                    out = ((value * inv) * (weight.to("xpu").float() + 1)).half().cpu()
                    return out, value.half().cpu() if res is not None else None
                # The no-residual layer3 form is derived operator work, rather
                # than an assertion that the worker used that form in this layer.
                if layer == 0 or residual:
                    baseline, _ = reference(x, res)
                    expected = source["p128_post_norm_output" if post else "p128_input_norm_output"]
                    if not torch.equal(baseline.view(torch.int16), expected.view(torch.int16)):
                        raise ValueError("literal Gemma expression no longer matches original P128 capture")
                for rows, active in ((3, 3), (4, 4), (12, 12), (16, 16), (4, 3)):
                    label = f"l{layer}_res{int(residual)}_m{rows}_active{active}"
                    xx = x[:rows].clone()
                    rr = res[:rows].clone() if res is not None else None
                    if active < rows:
                        xx[active:] = 0
                        if rr is not None: rr[active:] = 0
                    out, stored = reference(xx, rr)
                    tensors[label + "_input"] = xx
                    tensors[label + "_weight"] = weight.contiguous().clone()
                    tensors[label + "_output"] = out
                    if rr is not None:
                        tensors[label + "_residual"] = rr
                        tensors[label + "_stored_residual"] = stored
                    cases.append({"label": label, "layer": layer, "physical_rows": rows,
                                  "logical_rows": active, "residual": residual})
    save_file(tensors, str(args.output), {"scope": __doc__})
    args.output.chmod(0o644)
    identity = {"schema": 1, "scope": __doc__, "torch": torch.__version__, "eps": 1e-6,
                "cases": cases, "fixture_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
                "capture_tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "sources": [{"path": str(args.fixtures / name),
                             "sha256": hashlib.sha256((args.fixtures / name).read_bytes()).hexdigest()}
                            for name in ("real_block_oracle.safetensors", "real_attention3_oracle.safetensors")]}
    with report.open("x") as f: json.dump(identity, f, indent=2); f.write("\n")
    print("GEMMA_ROW_CASES", len(cases), "FIXTURE_SHA256", identity["fixture_sha256"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    capture(parser.parse_args())
