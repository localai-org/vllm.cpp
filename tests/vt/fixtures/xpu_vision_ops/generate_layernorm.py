"""Capture actual pinned Torch XPU LayerNorm; generated patterns, no weights."""
import hashlib
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
cases = []
specs = [(5, 1152, "float16", True), (3, 4608, "float16", True),
         (3, 1152, "bfloat16", True), (3, 37, "float32", True),
         (3, 72, "float16", False), (3, 1, "float16", True),
         (0, 1152, "float16", False)]
for number, (rows, width, dtype_name, affine) in enumerate(specs):
    dtype = getattr(torch, dtype_name)
    indices = torch.arange(rows * width).reshape(rows, width)
    x = ((indices * 7 + 13) % 41 - 20).float() * 0.07
    if rows:
        x[0] = 3.0  # constant row; biased variance and epsilon behavior
    if rows > 1:
        x[1] = 1.0 + (torch.arange(width) % 7 - 3).float() / 1024.0
    if rows > 2:
        x[2] += 50.0  # variance must not use E[x^2] - E[x]^2
    x = x.to(dtype=dtype, device="xpu")
    columns = torch.arange(width)
    w = (0.7 + (columns % 11).float() * 0.03).to(dtype=dtype, device="xpu") if affine else None
    b = ((columns % 13 - 6).float() * 0.01).to(dtype=dtype, device="xpu") if affine else None
    expected = F.layer_norm(x, (width,), w, b, 1e-6)
    repeated = F.layer_norm(x, (width,), w, b, 1e-6)
    assert torch.equal(expected, repeated)
    files = {}
    for name, value in [("input", x), ("weight", w), ("bias", b), ("expected", expected)]:
        if value is None:
            continue
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        filename = f"layernorm-{number}-{name}.bin"
        (out / filename).write_bytes(raw)
        files[name] = {"file": filename, "sha256": hashlib.sha256(raw).hexdigest()}
    cases.append({"shape": [rows, width], "dtype": dtype_name, "eps": 1e-6,
                  "affine": affine, "repeat_exact": True, "files": files})
(out / "layernorm.json").write_text(json.dumps({
    "reference_image": "sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918",
    "torch": torch.__version__, "torch_git": torch.version.git_version,
    "device": torch.xpu.get_device_name(), "operator": "torch.nn.functional.layer_norm",
    "cases": cases,
    "contract": {"float16": {"rel_l2": 0.0003, "max_abs": 0.002},
                 "bfloat16": {"rel_l2": 0.003, "max_abs": 0.03125},
                 "float32": {"rel_l2": 0.00005, "max_abs": 0.0001}},
}, indent=2) + "\n")
print(f"Executed and repeated {len(cases)} Torch XPU LayerNorm cases")
