#!/usr/bin/env python3
"""Capture isolated projection stages in the pinned producer image.

Torch is used only by this oracle tool, never by the native product. Allocation
retention exposes the original SmallM intermediate without modifying its binary;
the instrumented output must match an ordinary call byte-for-byte. The separate
raw GEMM helper fixes split-K to four, and the FP16 output-Hadamard probe is also
diagnostic rather than the production FP32-part reduction.
"""

import argparse
import json
import os
from pathlib import Path
import sys

from extract_projection import digest, headers, write_safetensors
from artifacts import ArtifactRoot, MissingArtifact, checked_fixture

IMAGE = "sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918"
LIB_SHA = "e60e499aefa95b290924ed98545eb815aca9d0fb1f3ede39b37d1ad49f3b522d"
OPS_SHA = "9c05ea0aefca817fd37cee27f6e6216b6cee2d2a16c6480cafff52dc5ff67c13"


def checked_runtime(root):
    root = Path(root)
    for relative, expected in (("exl3xpu/_C.so", LIB_SHA), ("exl3xpu/ops.py", OPS_SHA)):
        headers.require(digest((root / relative).read_bytes()) == expected,
                        f"pinned runtime SHA-256 mismatch: {relative}")
    # Overrides change the arithmetic or route; refuse rather than hide them.
    overrides = [k for k in os.environ if k.startswith("EXL3_") and k not in
                 {"EXL3_TARGET_RUNTIME", "EXL3_FIX_SPARSE_GDN_RETIREMENT", "EXL3_SDPA_CACHE_CAPACITY",
                  "EXL3_ARTIFACT_ROOT"}]
    headers.require(not overrides, f"unexpected EXL3 overrides: {overrides}")


def capture(fixture, output, runtime_root, image_identity):
    headers.require(image_identity == IMAGE, "capture requires the pinned production image")
    fixture, output = Path(fixture).resolve(), Path(output).resolve()
    sidecar = output.with_suffix(".json")
    headers.require(output != sidecar and output != fixture
                    and not output.exists() and not sidecar.exists(), "refusing to overwrite capture/input")
    receipt = checked_fixture(fixture)
    checked_runtime(runtime_root)
    sys.path.insert(0, str(runtime_root))
    import torch
    from safetensors.torch import load_file
    from exl3xpu import ops
    from torch.utils._python_dispatch import TorchDispatchMode

    headers.require(torch.__version__ == "2.13.0+xpu" and torch.xpu.is_available(),
                    "requires pinned Torch XPU runtime and GPU")
    headers.require("B70" in torch.xpu.get_device_name(0), "requires the B70")
    E = ops._get_esimd()
    class RetainSmallMAllocations(TorchDispatchMode):
        """Redispatch the original XPU kernel, retaining its nested factories."""
        def __init__(self):
            super().__init__()
            self.allocations = []
            self.linear_calls = 0

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            kwargs = kwargs or {}
            if str(func) == "exl3xpu_C.linear.default":
                self.linear_calls += 1
                # Keep this mode active for nested at::empty/zeros calls while
                # redispatching ONLY the original top-level XPU implementation.
                with self:
                    return func.redispatch(torch._C.DispatchKeySet(torch._C.DispatchKey.XPU), *args, **kwargs)
            value = func(*args, **kwargs)
            if str(func).startswith(("aten.empty.", "aten.zeros.")):
                self.allocations.append(value)
            return value
    host = load_file(str(fixture), device="cpu")
    tr = host["trellis"].to("xpu")
    suh = host["suh"].reshape(1, -1).to("xpu")
    svh = host["svh"].to("xpu")
    k, n, bits = suh.shape[1], svh.numel(), receipt["projection"]["bits"]
    shard = torch.zeros(n // 128, dtype=torch.int32, device="xpu")
    tensors = {}
    production_routes = []

    def save(name, tensor):
        value = tensor.detach().cpu().contiguous()
        headers.require(torch.isfinite(value).all().item(), f"nonfinite oracle stage: {name}")
        dtype = {torch.float16: "F16", torch.float32: "F32"}[value.dtype]
        tensors[name] = (dtype, list(value.shape), value.numpy().tobytes())

    decoded = torch.empty((k, n), dtype=torch.float16, device="xpu")
    E.exl3_reconstruct(tr, decoded, 0, bits, 2)
    save("decoded_weight_f16", decoded)
    del decoded
    for m in (1, 4):
        x = host[f"activation_m{m}_fp16"].to("xpu")
        production = ops.exl3_linear(x, tr, suh, svh, shard, [0, n], bits, 2)
        save(f"m{m}_production_output_f16", production)
        retained = RetainSmallMAllocations()
        with retained:
            observed = ops.exl3_linear(x, tr, suh, svh, shard, [0, n], bits, 2)
        headers.require(retained.linear_calls == 1, "capture did not redispatch exactly one original linear")
        headers.require(torch.equal(production.view(torch.int16), observed.view(torch.int16)),
                        "allocation retention changed production output bits")
        parts = [a for a in retained.allocations if a.dtype == torch.float32
                 and a.ndim == 3 and list(a.shape[1:]) == [m, n]]
        inputs = [a for a in retained.allocations if a.dtype == torch.float16
                  and a.ndim == 3 and a.shape[0] == 1 and a.shape[-1] == k]
        headers.require(len(parts) == len(inputs) == 1, "missing/ambiguous production SmallM allocations")
        part, blocked_input = parts[0], inputs[0]
        save(f"m{m}_production_split_parts_f32", part)
        padded_m = blocked_input.shape[1]
        actual_input = blocked_input.reshape(1, k // 16, padded_m, 16).permute(0, 2, 1, 3)
        save(f"m{m}_production_input_hadamard_f16", actual_input.reshape(1, padded_m, k)[0, :m])
        # HadOutKernel's source sums splits in increasing order. Keep this
        # derived FP32 stage distinct from the directly retained split bytes.
        cpu_parts = part.detach().cpu()
        ordered_sum = torch.zeros((m, n), dtype=torch.float32)
        for p in range(cpu_parts.shape[0]):
            ordered_sum.add_(cpu_parts[p])
        save(f"m{m}_production_raw_ordered_sum_f32", ordered_sum)
        production_routes.append({"m": m, "split_count": part.shape[0], "padded_m": padded_m,
                                  "ordinary_vs_observed_output_bit_exact": True})
        xh = torch.empty((1, m, k), dtype=torch.float16, device="xpu")
        E.exl3_had_in_rm(x, suh, xh)
        headers.require(torch.equal(actual_input.reshape(1, padded_m, k)[0, :m].view(torch.int16),
                                    xh[0].view(torch.int16)), "row-major helper differs from actual SmallM input transform")
        save(f"m{m}_input_hadamard_f16", xh[0])
        blocked = xh.reshape(1, m, k // 16, 16).permute(0, 2, 1, 3).contiguous()
        raw = E.exl3_gemm_raw(blocked, tr, shard, n, bits, 2, 0 if m == 1 else 1)
        save(f"m{m}_diagnostic_raw_split4_f32", raw)
        operand = raw.to(torch.float16)
        save(f"m{m}_output_hadamard_operand_f16", operand)
        out_had = torch.empty_like(operand)
        E.exl3_had_out_h(operand, svh, out_had)
        save(f"m{m}_output_hadamard_probe_f16", out_had)
        print(f"captured {fixture.name} M={m} path={'vector' if m == 1 else 'DPAS'}", flush=True)
    torch.xpu.synchronize()
    result = {
        "schema": 2, "kind": "pinned_producer_projection_capture",
        "image": image_identity, "library_sha256": LIB_SHA, "ops_sha256": OPS_SHA,
        "torch": torch.__version__, "device": torch.xpu.get_device_name(0),
        "capture_tool_sha256": digest(Path(__file__).read_bytes()),
        "fixture_sha256": receipt["fixture_sha256"], "fixture": str(fixture),
        "projection": receipt["projection"],
        "production_routes": production_routes,
        "stages": {
            "decoded_weight": "registered producer exl3_reconstruct, FP16[K,N]",
            "input_hadamard": "registered producer exl3_had_in_rm, FP16[M,K]",
            "diagnostic_raw": "registered exl3_gemm_raw; fixed four splits; vector M1, DPAS M4; NOT production intermediate",
            "output_hadamard_probe": "registered exl3_had_out_h on saved FP16 operand; NOT production FP32-part reduction",
            "production_output": "pinned ops.exl3_linear, native SmallM, FP16 input/output; default dynamic split-K",
            "production_split_parts": "original binary's retained F32[P,M,N] allocation; observed/ordinary output bit-exact",
            "production_raw_ordered_sum": "CPU FP32 increasing-split sum of retained parts, as in HadOutKernel source; derived stage",
        },
        "tensor_hashes": {name: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                          for name, t in sorted(tensors.items())},
        "production_intermediate_capture": "captured: actual blocked input and split parts via nested allocation retention",
    }
    write_safetensors(output, tensors, {"image": image_identity, "fixture_sha256": receipt["fixture_sha256"]})
    result["capture_sha256"] = digest(output.read_bytes())
    result["capture_file"] = str(output)
    with sidecar.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixture", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path,
                        help="Local artifact root; overrides EXL3_ARTIFACT_ROOT")
    parser.add_argument("--check-only", action="store_true",
                        help="Validate local fixture identity on the host without inference")
    parser.add_argument("--optional-artifacts", action="store_true",
                        help="Return CTest skip code 77 only for missing external artifacts")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--runtime-root", type=Path, default=Path("/opt/exl3xpu"))
    parser.add_argument("--image-identity")
    args = parser.parse_args(argv)
    if not args.check_only and (args.output is None or args.image_identity is None):
        parser.error("capture requires --output and --image-identity")
    try:
        root = ArtifactRoot(args.artifact_root)
        fixture = root.resolve(args.fixture)
        root.resolve(fixture.with_suffix(".json"), constrain=not args.fixture.is_absolute())
        receipt = checked_fixture(fixture)
        if args.check_only:
            print(json.dumps({"fixture_sha256": receipt["fixture_sha256"],
                              "tensor_hashes": receipt["tensor_hashes"]}, sort_keys=True))
            return 0
        result = capture(fixture, args.output, args.runtime_root, args.image_identity)
    except MissingArtifact as exc:
        print(f"capture_projection: {exc}", file=sys.stderr)
        return 77 if args.optional_artifacts else 1
    except (headers.InventoryError, OSError, KeyError, RuntimeError) as exc:
        print(f"capture_projection: {exc}", file=sys.stderr)
        return 1
    print(json.dumps({k: result[k] for k in ("capture_file", "capture_sha256", "production_intermediate_capture")}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
