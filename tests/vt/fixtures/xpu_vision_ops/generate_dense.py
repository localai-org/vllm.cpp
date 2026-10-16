"""Record executed pinned Torch XPU FP16 linear projections, no public weights."""
import hashlib
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

# Fixed before candidate comparisons. The absolute floor is the existing
# MatmulDenseF16 unit contract's 0.02; relative L2 additionally bounds aggregate
# error tightly, without the old unit's per-element 1% allowance.
contract = {"rel_l2": 0.0003, "max_abs": 0.02}

def raw(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

cases = []
for m, k, n, affine in ((5, 37, 19, True), (3, 1536, 16, False), (3, 4304, 16, True)):
    a = (((torch.arange(m * k) * 7 + 3) % 41 - 20).float() / 17).reshape(m, k)
    w = (((torch.arange(n * k) * 11 + 5) % 47 - 23).float() / 127).reshape(n, k)
    b = (torch.arange(n) % 13 - 6).float() / 13 if affine else None
    a = a.to(dtype=torch.float16, device="xpu")
    w = w.to(dtype=torch.float16, device="xpu")
    b = b.to(dtype=torch.float16, device="xpu") if b is not None else None
    y = F.linear(a, w, b)
    assert torch.equal(y, F.linear(a, w, b))
    files = {}
    for name, value in (("input", a), ("weight", w), ("bias", b), ("expected", y)):
        if value is None:
            continue
        payload = raw(value)
        filename = f"dense-{len(cases)}-{name}.bin"
        (out / filename).write_bytes(payload)
        files[name] = {"file": filename, "sha256": hashlib.sha256(payload).hexdigest()}
    cases.append({"m": m, "k": k, "n": n, "bias": affine, "files": files, "repeat_exact": True})
manifest = {
    "reference_image": "sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918",
    "torch": torch.__version__, "torch_git": torch.version.git_version,
    "device": torch.xpu.get_device_name(), "operator": "torch.nn.functional.linear",
    "dtype": "float16", "contract": contract, "cases": cases,
}
(out / "dense.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Executed and repeated {len(cases)} small Torch XPU linear cases")

if len(sys.argv) > 3:
    import numpy as np
    from safetensors import safe_open
    capture, model = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
    boundaries = json.loads((capture / "vision-boundaries-capture.json").read_text())
    index = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    replays = []
    for stage, input_name, output_name, key in (
            ("patch", "patch-input-0", "patch-output", "model.visual.patch_embed.proj"),
            ("merger-fc1", "merger-fc1-input-0", "merger-fc1-output-0", "model.visual.merger.linear_fc1"),
            ("merger-fc2", "merger-fc2-input-0", "merger-fc2-output-0", "model.visual.merger.linear_fc2")):
        with safe_open(model / index[key + ".weight"], framework="pt", device="cpu") as f:
            w = f.get_tensor(key + ".weight").to(dtype=torch.float16, device="xpu")
            b = f.get_tensor(key + ".bias").to(dtype=torch.float16, device="xpu")
        w = w.reshape(w.shape[0], -1)
        inputs = [e for e in boundaries["worker"]["captures"] if e["name"] == input_name]
        outputs = [e for e in boundaries["worker"]["captures"] if e["name"] == output_name]
        assert len(inputs) == len(outputs) == 2
        for i, (inp, expected) in enumerate(zip(inputs, outputs)):
            a = torch.from_numpy(np.fromfile(capture / inp["file"], dtype=np.float16).copy())
            a = a.reshape(inp["shape"]).to("xpu")
            y = F.linear(a, w, b)
            first = raw(y)
            repeats = [first == raw(F.linear(a, w, b)) for _ in range(2)]
            original = np.fromfile(capture / expected["file"], dtype=np.float16).astype(np.float64)
            values = y.cpu().float().numpy().astype(np.float64).reshape(-1)
            delta = values - original
            relative = float(np.linalg.norm(delta) / np.linalg.norm(original))
            absolute = float(np.max(np.abs(delta)))
            assert relative <= contract["rel_l2"] and absolute <= contract["max_abs"]
            filename = f"dense-real-{stage}-image{i}.float16"
            (capture / filename).write_bytes(first)
            replays.append({"stage": stage, "image": i, "key": key,
                            "input": inp, "expected": expected,
                            "standalone_file": filename, "standalone_repeat_exact": repeats,
                            "standalone_matches_worker_bytes": first == (capture / expected["file"]).read_bytes(),
                            "standalone_worker_rel_l2": relative, "standalone_worker_max_abs": absolute,
                            "weight_f16_sha256": hashlib.sha256(raw(w)).hexdigest(),
                            "bias_f16_sha256": hashlib.sha256(raw(b)).hexdigest()})
    (capture / "dense-real-projections.json").write_text(json.dumps(
        {"origin": "Standalone pinned F.linear on actual worker inputs and checkpoint FP16 parameters",
         "contract": contract, "cases": replays}, indent=2) + "\n")
    print("Executed six real projection replays with two repeats each")
