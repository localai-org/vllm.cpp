"""Execute production NeoX ApplyRotaryEmb with FP16 product boundaries on XPU."""
import hashlib
import json
import pathlib
import sys

import torch
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

def raw(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

def save(name, value):
    payload = raw(value)
    (out / name).write_bytes(payload)
    return {"file": name, "sha256": hashlib.sha256(payload).hexdigest()}

config = VllmConfig(compilation_config=CompilationConfig(
    mode=0, cudagraph_mode="NONE", custom_ops=["none"]))
with set_current_vllm_config(config):
    op = ApplyRotaryEmb(enforce_enable=True)
    assert op.is_neox_style and not op.enable_fp32_compute
    cases = []
    for rows, heads, width in ((3, 2, 8), (2, 1, 10), (5, 16, 72)):
        number = len(cases)
        size = rows * heads * width
        values = torch.arange(size, dtype=torch.float32, device="xpu")
        q = ((values * 7 % 101 - 50) * 0.173).reshape(rows, heads, width).half()
        k = ((values * 13 % 71 - 35) * 0.119).reshape(rows, heads, width).half()
        phase = (torch.arange(rows * width // 2, device="xpu", dtype=torch.float32)
                 * 0.065).reshape(rows, width // 2)
        cos, sin = phase.cos().half(), phase.sin().half()
        cache = torch.cat((cos, sin), dim=-1)
        qk = torch.stack((q, k)).contiguous()
        result = op(qk, cos, sin)
        assert raw(result) == raw(op(qk, cos, sin))
        files = {name: save(f"vision-rope-{number}-{name}.bin", value)
                 for name, value in (("q", q), ("k", k), ("cache", cache),
                                     ("q_out", result[0]), ("k_out", result[1]))}
        cases.append({"shape": [rows, heads, width], "files": files, "repeat_exact": True})
    manifest = {
        "reference_image": "sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918",
        "torch": torch.__version__, "torch_git": torch.version.git_version,
        "device": torch.xpu.get_device_name(),
        "class": type(op).__module__ + "." + type(op).__qualname__,
        "selected_method": op._forward_method.__qualname__,
        "is_neox_style": op.is_neox_style, "enable_fp32_compute": op.enable_fp32_compute,
        "contract": {"rel_l2": 0, "max_abs": 0, "bytes": "exact"}, "cases": cases}
    (out / "vision-rope.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"VISION_ROPE_FIXTURES_PASS {len(cases)} cases", flush=True)
