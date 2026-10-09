#!/usr/bin/env python3
"""Bounded pinned SmallM comparison on real representative target/draft weights.

Synthetic, explicitly named identical F16 operands; no model/state/serving proof.
Capture ordinary original intermediates without changing the original library.
Unprofiled complete operator timing and separate actual GPU profiles never mix.
"""
import argparse
import json
import os
from pathlib import Path
import struct
import sys
import time

from capture_projection import IMAGE, LIB_SHA, OPS_SHA, checked_fixture, checked_runtime
from capture_runtime_layout import PROFILE
from extract_projection import digest, headers, write_safetensors


def run(args):
    headers.require(args.image_identity == IMAGE, "requires pinned original image")
    headers.require(not args.output.exists(), "refusing existing report")
    headers.require(args.profile != bool(args.fixture_output),
                    "unprofiled capture needs fixture-output; separate profile must omit it")
    if args.fixture_output:
        headers.require(not args.fixture_output.exists(), "refusing existing capture")
    manifest = json.loads(args.manifest.read_text())
    headers.require(manifest["schema"] == "b70-exl3-p6-smallm-input-v1", "wrong manifest")
    checked_runtime(Path("/opt/exl3xpu"))
    import yaml
    profile = yaml.safe_load(PROFILE.read_text())
    for key, value in profile["env"].items():
        if value is not None:
            os.environ[key] = str(value).replace("{model_dir}", str(PROFILE.parent))
    sys.path.insert(0, "/opt/exl3xpu")
    import torch
    from safetensors.torch import load_file
    from exl3xpu import ops
    from torch.utils._python_dispatch import TorchDispatchMode
    headers.require(torch.__version__ == "2.13.0+xpu" and
                    torch.xpu.is_available() and "B70" in torch.xpu.get_device_name(0),
                    "requires pinned Torch and B70")

    class Observe(TorchDispatchMode):
        def __init__(self):
            super().__init__()
            self.allocations, self.calls, self.setters = [], [], []

        def __torch_dispatch__(self, func, types, args=(), kwargs=None):
            name = str(func)
            if name.startswith("exl3xpu_C.exl3_set_"):
                self.setters.append({"operator": name, "args": list(args)})
            if name in ("exl3xpu_C.linear.default", "exl3xpu_C.exl3_gemm_small.default"):
                self.calls.append(name)
                with self:
                    return func.redispatch(torch._C.DispatchKeySet(torch._C.DispatchKey.XPU),
                                           *args, **(kwargs or {}))
            result = func(*args, **(kwargs or {}))
            if name.startswith(("aten.empty.", "aten.zeros.")):
                self.allocations.append(result)
            return result

    settings = Observe()
    with settings:
        ops._get_esimd()
    tensors, weights, cases = {}, [], []

    def save(name, tensor):
        if not args.fixture_output:
            return
        host = tensor.detach().cpu().contiguous()
        dtype = {torch.float16: "F16", torch.float32: "F32",
                 torch.int16: "I16", torch.int32: "I32"}[host.dtype]
        headers.require(not host.is_floating_point() or torch.isfinite(host).all().item(),
                        "nonfinite original stage: " + name)
        tensors[name] = (dtype, list(host.shape), host.numpy().tobytes())

    for wi, spec in enumerate(manifest["weights"]):
        sources = []
        for source in spec["sources"]:
            path = Path(source["path"])
            headers.require(digest(path.read_bytes()) == source["sha256"], "source changed: " + str(path))
            if spec["mode"] == "merged":
                receipt = json.loads(path.with_suffix(".json").read_text())
                headers.require(receipt["library_sha256"] == LIB_SHA and
                                receipt["ops_sha256"] == OPS_SHA, "changed original group")
            else:
                checked_fixture(path)
            sources.append(load_file(str(path), device="cpu"))
        if spec["mode"] == "merged":
            headers.require(len(sources) == 1, "one frozen merged source required")
            tr, su, sv, mapping = [sources[0][key] for key in
                                  ("merged_trellis", "stacked_suh", "merged_svh", "source_map")]
        else:
            tr = torch.cat([s["trellis"] for s in sources], dim=1).contiguous()
            su = torch.stack([s["suh"] for s in sources])
            sv = torch.cat([s["svh"] for s in sources])
            mapping = torch.cat([torch.full((s["svh"].numel() // 128,), i, dtype=torch.int32)
                                 for i, s in enumerate(sources)])
        if spec["mode"] == "compact":
            subset_path = Path(spec["subset"]["path"])
            headers.require(digest(subset_path.read_bytes()) == spec["subset"]["sha256"], "changed subset")
            subset = json.loads(subset_path.read_text())
            blocks = subset["blocks"]
            headers.require(subset["block_size"] == 128 and len(blocks) == 512 and
                            len(set(blocks)) == 512 and all(0 <= b < 1940 for b in blocks),
                            "requires exact65536 whole head blocks")
            columns = torch.tensor([b * 8 + j for b in blocks for j in range(8)], dtype=torch.long)
            token_map = torch.tensor([b * 128 + j for b in blocks for j in range(128)], dtype=torch.int32)
            tr = tr.index_select(1, columns).contiguous()
            sv = sv.index_select(0, token_map.long()).contiguous()
            mapping = torch.zeros(512, dtype=torch.int32)
            save("compact_token_map", token_map)
        k, n, groups = su.shape[1], sv.numel(), su.shape[0]
        bits = tr.shape[2] // 16
        headers.require((k, n, groups, bits) == tuple(spec[key] for key in ("k", "n", "groups", "bits")),
                        "weight shape mismatch: " + spec["name"])
        tr, su, sv, mapping = [t.to("xpu") for t in (tr, su, sv, mapping)]
        prefix = f"w{wi}"
        for name, tensor in zip(("packed", "su", "sv", "map"), (tr, su, sv, mapping)):
            save(prefix + "_" + name, tensor)
        weights.append((tr, su, sv, mapping, spec["bounds"], bits))
        for m in spec["rows"]:
            raw = bytearray(b"".join(struct.pack("<e", ((i * 37 + row * 101) % 511 - 255) / 256)
                                     for row in range(m) for i in range(k)))
            x = torch.frombuffer(raw, dtype=torch.float16).reshape(m, k).to("xpu")
            cases.append({"weight": wi, "m": m, "x": x, "prefix": f"c{len(cases)}"})
        del sources

    records = []
    for case in cases:
        wi, m, x, prefix = [case[key] for key in ("weight", "m", "x", "prefix")]
        spec = manifest["weights"][wi]
        k, n, groups = [spec[key] for key in ("k", "n", "groups")]
        tr, su, sv, mapping, bounds, bits = weights[wi]
        call = lambda: ops.exl3_linear(x, tr, su, sv, mapping, bounds, bits, 2)
        ordinary = call()
        observed = Observe()
        with observed:
            retained = call()
        torch.xpu.synchronize()
        headers.require(observed.calls == ["exl3xpu_C.linear.default"] and
                        torch.equal(ordinary.view(torch.int16), retained.view(torch.int16)),
                        "wrong original route or changed observation")
        parts = [t for t in observed.allocations if t.dtype == torch.float32 and
                 list(t.shape[1:]) == [m, n] and t.ndim == 3]
        had = [t for t in observed.allocations if t.dtype == torch.float16 and t.ndim == 3 and
               t.shape[0] == groups and t.shape[-1] == k]
        headers.require(len(parts) == len(had) == 1, "missing original SmallM allocations")
        save(prefix + "_x", x)
        save(prefix + "_out", ordinary)
        save(prefix + "_parts", parts[0])
        save(prefix + "_had_blocked", had[0])
        case["expected"] = ordinary.detach().cpu().contiguous().numpy().tobytes()
        records.append({"prefix": prefix, "weight": wi, "name": spec["name"], "role": spec["role"],
                        "m": m, "k": k, "n": n, "bits": bits, "groups": groups,
                        "padded_m": had[0].shape[1], "splits": parts[0].shape[0],
                        "ordinary_observed_exact": True, "calls": observed.calls,
                        "complete_operator_wall_ms": [], "device_events": []})
        del observed, retained, ordinary, parts, had

    # Two complete different-weight sequence warmups, then three samples per
    # shape. Keep checks/copies outside operator timers; no hot single-matrix loop.
    for _ in range(2):
        for case in cases:
            tr, su, sv, mapping, bounds, bits = weights[case["weight"]]
            ops.exl3_linear(case["x"], tr, su, sv, mapping, bounds, bits, 2)
        torch.xpu.synchronize()
    for sample in range(3):
        for case, record in zip(cases, records):
            tr, su, sv, mapping, bounds, bits = weights[case["weight"]]
            if args.profile:
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                                       torch.profiler.ProfilerActivity.XPU]) as prof:
                    result = ops.exl3_linear(case["x"], tr, su, sv, mapping, bounds, bits, 2)
                    torch.xpu.synchronize()
                events = [{"name": e.name, "us": e.time_range.elapsed_us(), "device_type": str(e.device_type)}
                          for e in prof.events() if str(e.device_type).endswith("XPU")]
                headers.require(bool(events), "missing executed original GPU profile events")
                record["device_events"].append({"sample": sample, "events": events})
            else:
                start = time.perf_counter()
                result = ops.exl3_linear(case["x"], tr, su, sv, mapping, bounds, bits, 2)
                torch.xpu.synchronize()
                record["complete_operator_wall_ms"].append(1000 * (time.perf_counter() - start))
            raw = result.detach().cpu().contiguous().numpy().tobytes()
            headers.require(raw == case["expected"], "original repeat output drift")
            del result
    if args.fixture_output:
        write_safetensors(args.fixture_output, tensors, {"scope": __doc__, "image": IMAGE})
    report = {"schema": "b70-exl3-p6-smallm-original-v1", "image": IMAGE,
              "profiled": args.profile, "source_manifest_sha256": digest(args.manifest.read_bytes()),
              "profile_sha256": digest(PROFILE.read_bytes()), "library_sha256": LIB_SHA, "ops_sha256": OPS_SHA,
              "tool_sha256": digest(Path(__file__).read_bytes()), "torch": torch.__version__,
              "setup_setter_calls": settings.setters,
              "EXL3_environment": {k: v for k, v in os.environ.items() if k.startswith("EXL3_")},
              "cases": records, "weights": manifest["weights"],
              "memory": {"allocated": torch.xpu.memory_allocated(), "reserved": torch.xpu.memory_reserved(),
                         "peak_allocated": torch.xpu.max_memory_allocated(), "peak_reserved": torch.xpu.max_memory_reserved()},
              "scope": __doc__}
    if args.fixture_output:
        report["fixture"] = {"path": str(args.fixture_output), "sha256": digest(args.fixture_output.read_bytes())}
        report["tensors"] = {n: {"dtype": t[0], "shape": t[1], "sha256": digest(t[2])}
                             for n, t in sorted(tensors.items())}
    with args.output.open("x") as stream:
        json.dump(report, stream, indent=2); stream.write("\n")
    print("P6_ORIGINAL_DONE", len(records), "profiled", args.profile, flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fixture-output", type=Path)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--image-identity", required=True)
    run(parser.parse_args())
