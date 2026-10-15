#!/usr/bin/env python3
"""Original fused W8A8 on frozen real packed groups and real worker norm rows.

Retains original INT8/scale/F16 allocations without changing the pinned binary.
M128 remains the original SmallM control. Larger forms cyclically repeat the real
P128 operands, including the actual prefill geometry M1600/M896, with explicit
zero and near-zero rows. These are bounded operator inputs, not a newly captured
long-prompt worker trajectory. No unquantized comparison.
"""
import argparse
import json
from pathlib import Path
import sys

from capture_projection import IMAGE, LIB_SHA, OPS_SHA, checked_runtime, checked_fixture
from extract_projection import digest, headers, write_safetensors


def capture(args):
    output = args.output.resolve()
    report_path = output.with_suffix(".json")
    headers.require(args.image_identity == IMAGE, "pinned production image required")
    headers.require(not output.exists() and not report_path.exists(), "refusing to overwrite W8A8 capture")
    checked_runtime(args.runtime_root)
    source = args.group or args.packed_source
    if args.group:
        group_report = headers.read_json(source.with_suffix(".json"))
        headers.require(digest(source.read_bytes()) == group_report["capture_sha256"], "group SHA mismatch")
        headers.require(group_report["library_sha256"] == LIB_SHA and group_report["ops_sha256"] == OPS_SHA,
                        "group reference identity mismatch")
    else:
        packed_report = checked_fixture(source)
        p = packed_report["projection"]
        headers.require(p["codebook"] == "mul1" and p["column_count"] % 128 == 0,
                        "requires whole packed mul1 blocks")
        group_report = {"k": p["source_k"], "n": p["column_count"], "groups": 1,
                        "bits": p["bits"], "bounds": [0, p["column_count"]],
                        "checkpoint": packed_report["checkpoint"]}
    sys.path.insert(0, str(args.runtime_root))
    import torch
    from safetensors.torch import load_file
    from exl3xpu import ops
    from torch.utils._python_dispatch import TorchDispatchMode
    headers.require(torch.__version__ == "2.13.0+xpu" and torch.xpu.is_available() and
                    "B70" in torch.xpu.get_device_name(0), "pinned Torch and B70 required")
    host = load_file(str(source), device="cpu")
    real = load_file(str(args.operands), device="cpu")[args.operand_name]
    if args.group:
        tr, su, sv, mapping = [host[name].to("xpu") for name in
                              ("merged_trellis", "stacked_suh", "merged_svh", "source_map")]
    else:
        tr = host["trellis"].to("xpu")
        su = host["suh"].reshape(1, -1).to("xpu")
        sv = host["svh"].to("xpu")
        mapping = torch.zeros(sv.numel() // 128, dtype=torch.int32, device="xpu")
    k, n, groups, bits = group_report["k"], group_report["n"], group_report["groups"], group_report["bits"]
    bounds = group_report["bounds"]
    headers.require(real.dtype == torch.float16 and list(real.shape) == [128, k], "real operand geometry mismatch")
    headers.require(bits in (4, 6) and (groups == 1 or not torch.equal(su[0], su[1])),
                    "requires actual unequal group transforms")
    tensors, routes = {}, []

    def save(name, tensor):
        t = tensor.detach().cpu().contiguous()
        headers.require(not t.is_floating_point() or torch.isfinite(t).all().item(), f"nonfinite stage {name}")
        dtype = {torch.float16: "F16", torch.float32: "F32", torch.int8: "I8",
                 torch.int16: "I16", torch.int32: "I32"}[t.dtype]
        tensors[name] = dtype, list(t.shape), t.numpy().tobytes()

    class Retain(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.allocations, self.calls = [], 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if str(func) == "exl3xpu_C.linear.default":
                self.calls += 1
                with self:
                    return func.redispatch(torch._C.DispatchKeySet(torch._C.DispatchKey.XPU), *args, **kwargs)
            value = func(*args, **kwargs)
            if str(func).startswith(("aten.empty.", "aten.zeros.")):
                self.allocations.append(value)
            return value

    for name, t in zip(("merged_trellis", "stacked_suh", "merged_svh", "source_map"), (tr, su, sv, mapping)):
        save(name, t)
    E = ops._get_esimd()
    E.exl3_set_int8(1)
    for m in args.rows:
        headers.require(m in (128, 129, 256, 896, 1600, 4096), "unsupported capture row form")
        x = real.repeat((m + 127) // 128, 1)[:m].clone()
        x[0] = 0
        x[1] = x[1] * (2.0 ** -20)
        x = x.to("xpu")
        save(f"input_m{m}", x)
        ordinary = ops.exl3_linear(x, tr, su, sv, mapping, bounds, bits, 2)
        retained = Retain()
        with retained:
            observed = ops.exl3_linear(x, tr, su, sv, mapping, bounds, bits, 2)
        headers.require(retained.calls == 1 and torch.equal(ordinary.view(torch.int16), observed.view(torch.int16)),
                        "retention changed original output or route")
        save(f"output_m{m}", ordinary)
        if m == 128:
            parts = [t for t in retained.allocations if t.dtype == torch.float32 and t.ndim == 3 and
                     list(t.shape[1:]) == [m, n]]
            headers.require(len(parts) == 1, "M128 did not retain the unchanged SmallM control")
            routes.append({"m": m, "leaf": "original_smallm", "splits": parts[0].shape[0]})
        else:
            ms = (m + 255) // 256 * 256
            def unique(dtype, shape, exclude=None):
                found = [t for t in retained.allocations if t.dtype == dtype and list(t.shape) == shape and
                         (exclude is None or t.data_ptr() != exclude.data_ptr())]
                headers.require(len(found) == 1, f"missing/ambiguous original stage {dtype} {shape}")
                return found[0]
            xq = unique(torch.int8, [groups, ms, k])
            sx = unique(torch.float32, [groups, ms])
            y = unique(torch.float16, [ms, n], observed)
            save(f"xq_m{m}", xq[:, :m])
            save(f"sx_m{m}", sx[:, :m])
            save(f"y_m{m}", y[:m])
            weights = [t for t in retained.allocations if t.dtype == torch.int8 and t.ndim == 2 and t.shape[0] == k]
            headers.require(sorted(t.shape[1] for t in weights) == sorted(b-a for a, b in zip(bounds, bounds[1:])),
                            "original W8A8 reconstruction group mismatch")
            # Capture the last block so native panel reconstruction has a direct
            # integer witness too, rather than inferring it from final output.
            save(f"last_weight_panel_m{m}", weights[-1][:, -128:])
            routes.append({"m": m, "padded_m": ms, "leaf": "original_signed_int8_onednn_f16_hadamard",
                           "intermediate_stride": n, "weight_allocations_bytes": [t.numel() for t in weights]})
        print(f"CAPTURED_W8A8 M={m} K={k} N={n}", flush=True)
        del retained, ordinary, observed
    torch.xpu.synchronize()
    write_safetensors(output, tensors, {"scope": __doc__, "image": IMAGE})
    output.chmod(0o644)
    result = {"schema": 1, "kind": "pinned_original_grouped_w8a8_operator", "scope": __doc__,
              "image": IMAGE, "library_sha256": LIB_SHA, "ops_sha256": OPS_SHA,
              "checkpoint": group_report["checkpoint"], "k": k, "n": n, "groups": groups,
              "bits": bits, "bounds": bounds, "routes": routes,
              "source_path": str(source), "group_sha256": digest(source.read_bytes()),
              "source_mode": "grouped" if args.group else "packed_selected_columns",
              "operands_sha256": digest(args.operands.read_bytes()),
              "operand_name": args.operand_name, "capture_tool_sha256": digest(Path(__file__).read_bytes()),
              "tensor_hashes": {name: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                for name, t in sorted(tensors.items())},
              "capture_sha256": digest(output.read_bytes())}
    with report_path.open("x") as f: json.dump(result, f, indent=2); f.write("\n")
    print("CAPTURE_SHA256", result["capture_sha256"], flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sources = parser.add_mutually_exclusive_group(required=True)
    sources.add_argument("--group", type=Path)
    sources.add_argument("--packed-source", type=Path)
    parser.add_argument("--operands", type=Path, required=True)
    parser.add_argument("--operand-name", default="p128_input_norm_output")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rows", type=int, nargs="+", choices=(128, 129, 256, 896, 1600, 4096),
                        default=[128, 129, 256])
    parser.add_argument("--runtime-root", type=Path, default=Path("/opt/exl3xpu"))
    parser.add_argument("--image-identity", required=True)
    capture(parser.parse_args())
