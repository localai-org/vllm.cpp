"""Capture the pinned executing Torch XPU GELU variants, without model weights."""
import hashlib
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
cases = []
for approximate in ("tanh", "none"):
    for rows, width, dtype_name in ((3, 4304, "float16"),
                                   (3, 72, "bfloat16"), (3, 37, "float32")):
        dtype = getattr(torch, dtype_name)
        indices = torch.arange(rows * width).reshape(rows, width)
        x = ((indices * 7 + 13) % 1027 - 513).float() / 43.0
        x.flatten()[:12] = torch.tensor(
            [0., -0., 1e-5, -1e-5, .001, -.001, .5, -.5, 3., -3., 8., -8.])
        x = x.to(dtype=dtype, device="xpu")
        expected = F.gelu(x, approximate=approximate)
        assert torch.equal(expected, F.gelu(x, approximate=approximate))
        files = {}
        for name, value in (("input", x), ("expected", expected)):
            raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
            filename = f"gelu-{len(cases)}-{name}.bin"
            (out / filename).write_bytes(raw)
            files[name] = {"file": filename, "sha256": hashlib.sha256(raw).hexdigest()}
        cases.append({"shape": [rows, width], "dtype": dtype_name,
                      "approximate": approximate, "repeat_exact": True, "files": files})
manifest = {
    "reference_image": "sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918",
    "torch": torch.__version__, "torch_git": torch.version.git_version,
    "torch_xpu_ops_git": "bc294243807debb350dca694049ba7a806dfdceb",
    "device": torch.xpu.get_device_name(), "operator": "torch.nn.functional.gelu",
    "cases": cases,
    # Fixed before candidate comparisons, independently for storage dtypes.
    "contract": {"float16": {"rel_l2": 0.0003, "max_abs": 0.002},
                 "bfloat16": {"rel_l2": 0.003, "max_abs": 0.03125},
                 "float32": {"rel_l2": 0.00001, "max_abs": 0.00001}},
}
(out / "gelu.json").write_text(json.dumps(manifest, indent=2) + "\n")
print(f"Executed and repeated {len(cases)} Torch XPU GELU cases")

if len(sys.argv) > 2:
    import numpy as np
    capture = pathlib.Path(sys.argv[2])
    boundaries = json.loads((capture / "vision-boundaries-capture.json").read_text())
    real = []
    for entry in boundaries["worker"]["captures"]:
        if entry["name"] != "merger-fc1-output-0":
            continue
        assert entry["dtype"] == "torch.float16" and entry["shape"] == [192, 4608]
        x = torch.from_numpy(np.fromfile(capture / entry["file"], dtype=np.float16).copy())
        x = x.reshape(entry["shape"]).to("xpu")
        expected = F.gelu(x, approximate="none")
        assert torch.equal(expected, F.gelu(x, approximate="none"))
        raw = expected.cpu().contiguous().view(torch.uint8).numpy().tobytes()
        filename = f"gelu-real-merger-{len(real)}.float16"
        (capture / filename).write_bytes(raw)
        real.append({"input": entry, "expected": filename,
                     "expected_sha256": hashlib.sha256(raw).hexdigest(), "repeat_exact": True})
    assert len(real) == 2
    (capture / "gelu-real-merger.json").write_text(json.dumps(
        {"operator": "torch.nn.functional.gelu(approximate='none')",
         "origin": "Standalone pinned GPU operator on actual executed merger fc1 outputs",
         "cases": real, "contract": manifest["contract"]["float16"]}, indent=2) + "\n")
    print("Executed two standalone GPU GELU references on real merger boundary inputs")
