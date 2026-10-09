#!/usr/bin/env python3
"""Capture real packed groups through the unchanged pinned SmallM binary.

No dense reconstruction. Torch is confined to this independent oracle tool.
The ordinary output must equal the allocation-observed output bit-for-bit.
"""
import argparse
import json
from pathlib import Path
import struct
import sys

from capture_projection import IMAGE, LIB_SHA, OPS_SHA, checked_fixture, checked_runtime
from extract_projection import digest, headers, write_safetensors

ROWS = (1, 4, 12, 16, 128)
CHECKPOINT = {"model": "turboderp/Qwen3.8-27B-exl3",
              "revision": "113cf7ab958054860e43fb7f3063b1af19171095"}


def checked_group(fixtures, single_source=False):
    headers.require(len(fixtures) == 1 if single_source else len(fixtures) >= 2,
                    "single-source capture needs exactly one source" if single_source
                    else "group needs at least two sources")
    receipts = [checked_fixture(p) for p in fixtures]
    bounds = [0]
    k = bits = None
    for r in receipts:
        p, tensors = r["projection"], r["tensor_hashes"]
        headers.require(r["checkpoint"] == CHECKPOINT, "group checkpoint mismatch")
        if k is None:
            k, bits = p["source_k"], p["bits"]
        n = p["source_n"]
        headers.require(k > 0 and k % 128 == 0 and bits in (4, 6) and
                        p["source_k"] == k and p["bits"] == bits and p["codebook"] == "mul1" and
                        n > 0 and n % 128 == 0 and p["column_start"] == 0 and p["column_count"] == n,
                        "group requires compatible full-width mul1 sources")
        for name, dtype, shape in (("trellis", "I16", [k // 16, n // 16, 16 * bits]),
                                    ("suh", "F16", [k]), ("svh", "F16", [n]),
                                    ("mul1", "I32", [])):
            headers.require((tensors[name]["dtype"], tensors[name]["shape"]) == (dtype, shape),
                            f"group tensor geometry mismatch: {name}")
        for m in (1, 4):
            name = f"activation_m{m}_fp16"
            headers.require(tensors[name] == receipts[0]["tensor_hashes"][name],
                            "group activation mismatch")
        bounds.append(bounds[-1] + n)
    headers.require(bounds[-1] <= 2**31 - 1 and k <= 2**31 - 1, "group geometry overflow")
    return receipts, k, bits, bounds


def capture(fixtures, output, runtime_root, image_identity, single_source=False):
    headers.require(image_identity == IMAGE, "capture requires the pinned production image")
    fixtures = [Path(p).resolve() for p in fixtures]
    output = Path(output).resolve()
    sidecar = output.with_suffix(".json")
    headers.require(output != sidecar and output not in fixtures and
                    not output.exists() and not sidecar.exists(), "refusing to overwrite capture/input")
    receipts, k, bits, bounds = checked_group(fixtures, single_source)
    checked_runtime(runtime_root)
    sys.path.insert(0, str(runtime_root))
    import torch
    from safetensors.torch import load_file
    from exl3xpu import ops
    from torch.utils._python_dispatch import TorchDispatchMode
    headers.require(torch.__version__ == "2.13.0+xpu" and torch.xpu.is_available() and
                    "B70" in torch.xpu.get_device_name(0), "requires pinned Torch XPU and B70")
    host = [load_file(str(p), device="cpu") for p in fixtures]
    headers.require(all(t["mul1"].item() & 0xffffffff == 0x83DCD12D for t in host),
                    "unexpected mul1 marker bits")
    # A complete target head needs no extra host packed concatenation either.
    tr = (host[0]["trellis"] if single_source else
          torch.cat([t["trellis"] for t in host], dim=1).contiguous()).to("xpu")
    suh = torch.stack([t["suh"] for t in host]).to("xpu")
    svh = torch.cat([t["svh"] for t in host]).to("xpu")
    n, groups = bounds[-1], len(host)
    mapping = torch.cat([torch.full(((b-a)//128,), s, dtype=torch.int32)
                         for s, (a, b) in enumerate(zip(bounds, bounds[1:]))]).to("xpu")
    tensors, routes = {}, []

    def save(name, tensor):
        t = tensor.detach().cpu().contiguous()
        headers.require(not t.is_floating_point() or torch.isfinite(t).all().item(),
                        f"nonfinite oracle stage: {name}")
        dtype = {torch.float16: "F16", torch.float32: "F32",
                 torch.int16: "I16", torch.int32: "I32"}[t.dtype]
        tensors[name] = (dtype, list(t.shape), t.numpy().tobytes())

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

    save("merged_trellis", tr)
    save("stacked_suh", suh)
    save("merged_svh", svh)
    save("source_map", mapping)
    for m in ROWS:
        raw = bytearray(b"".join(struct.pack("<e", ((i*37 + row*101) % 511 - 255)/256)
                                  for row in range(m) for i in range(k)))
        x = torch.frombuffer(raw, dtype=torch.float16).reshape(m, k).to("xpu")
        save(f"activation_m{m}_fp16", x)
        ordinary = ops.exl3_linear(x, tr, suh, svh, mapping, bounds, bits, 2)
        retained = Retain()
        with retained:
            observed = ops.exl3_linear(x, tr, suh, svh, mapping, bounds, bits, 2)
        headers.require(retained.calls == 1 and
                        torch.equal(ordinary.view(torch.int16), observed.view(torch.int16)),
                        "allocation retention changed production output bits/route")
        parts = [a for a in retained.allocations if a.dtype == torch.float32 and
                 a.ndim == 3 and list(a.shape[1:]) == [m, n]]
        inputs = [a for a in retained.allocations if a.dtype == torch.float16 and
                  a.ndim == 3 and a.shape[0] == groups and a.shape[-1] == k]
        headers.require(len(parts) == len(inputs) == 1, "ambiguous/missing SmallM intermediate")
        part, blocked = parts[0], inputs[0]
        padded = blocked.shape[1]
        active = blocked.reshape(groups, k//16, padded, 16).permute(0, 2, 1, 3)
        save(f"m{m}_production_input_hadamard_f16", active.reshape(groups, padded, k)[:, :m])
        save(f"m{m}_production_split_parts_f32", part)
        save(f"m{m}_production_output_f16", ordinary)
        routes.append({"m": m, "splits": part.shape[0], "padded_m": padded,
                       "ordinary_vs_observed_output_bit_exact": True})
        print(f"captured group K={k} N={n} S={groups} M={m} P={part.shape[0]}", flush=True)
    torch.xpu.synchronize()
    result = {"schema": 1, "kind": "pinned_producer_grouped_smallm_capture",
              "image": image_identity, "library_sha256": LIB_SHA, "ops_sha256": OPS_SHA,
              "checkpoint": CHECKPOINT, "torch": torch.__version__,
              "device": torch.xpu.get_device_name(0), "k": k, "n": n, "bits": bits,
              "groups": groups, "bounds": bounds, "single_source": single_source,
              "production_routes": routes,
              "source_fixtures": [{"path": str(p), "sha256": r["fixture_sha256"],
                                   "projection": r["projection"]} for p, r in zip(fixtures, receipts)],
              "capture_tool_sha256": digest(Path(__file__).read_bytes()),
              "tensor_hashes": {name: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                                for name, t in sorted(tensors.items())},
              "scope": "Full packed sources, synthetic same-input M1/4/12/16/128; original SmallM intermediates/output. No dense reconstruction or complete model/state parity."}
    write_safetensors(output, tensors, {"image": image_identity, "checkpoint_revision": CHECKPOINT["revision"]})
    result["capture_file"], result["capture_sha256"] = str(output), digest(output.read_bytes())
    with sidecar.open("x") as stream:
        json.dump(result, stream, indent=2); stream.write("\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, default=Path("/opt/exl3xpu"))
    parser.add_argument("--image-identity", required=True)
    parser.add_argument("--single-source", action="store_true",
                        help="Capture one full-width projection without dense reconstruction")
    args = parser.parse_args()
    result = capture(args.fixture, args.output, args.runtime_root, args.image_identity, args.single_source)
    print(json.dumps({k: result[k] for k in ("capture_file", "capture_sha256", "production_routes")}))
