"""Execute pinned non-causal MMEncoderAttention; optionally replay real Q/K/V."""
import hashlib
import json
import pathlib
import sys
import time

import torch
from vllm.config import CompilationConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.attention import MMEncoderAttention

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
# Matching shared XPU unpaged-attention FP16 unit contract, declared before
# any native candidate: abs(delta) <= absolute + relative * abs(reference).
contract = {"relative": 0.001, "absolute": 0.001,
            "rule": "per-element absolute + relative*abs(reference)",
            "origin": "test_xpu_attention.cpp: XPU unpaged attention FP16 output"}

def raw(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

def save(directory, name, value):
    payload = raw(value)
    (directory / name).write_bytes(payload)
    return {"file": name, "shape": list(value.shape), "dtype": str(value.dtype),
            "sha256": hashlib.sha256(payload).hexdigest()}

config = VllmConfig(compilation_config=CompilationConfig(
    mode=0, cudagraph_mode="NONE", custom_ops=["none"]))
with set_current_vllm_config(config):
    torch.set_default_dtype(torch.float16)
    with torch.device("xpu"):
        operation = MMEncoderAttention(num_heads=16, head_size=72, scale=72**-0.5)
    assert not operation.fp8_enabled
    routes = []
    original = torch.ops._vllm_fa2_C.varlen_fwd
    def observed(*args, **kwargs):
        route = {"operator": "_vllm_fa2_C.varlen_fwd", "q_shape": list(args[0].shape),
                 "dtype": str(args[0].dtype), "device": str(args[0].device),
                 "causal": bool(args[19])}
        try:
            result = original(*args, **kwargs)
        except RuntimeError as error:
            routes.append({**route,"returned":False,"error":str(error)})
            raise
        routes.append({**route,"returned":True})
        return result
    torch.ops._vllm_fa2_C.varlen_fwd = observed
    import vllm_xpu_kernels.flash_attn_interface as interface
    original_fallback = interface._fallback_varlen_attn
    def observed_fallback(*args, **kwargs):
        result = original_fallback(*args, **kwargs)
        routes.append({"operator":"_fallback_varlen_attn","q_shape":list(args[0].shape),
                       "device":str(args[0].device),"returned":True})
        return result
    interface._fallback_varlen_attn = observed_fallback

    def execute(q, k, v):
        rows = q.shape[0]
        offsets = torch.tensor([0, rows], dtype=torch.int32, device="xpu")
        torch.xpu.synchronize()
        start = time.perf_counter()
        value = operation(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                          cu_seqlens=offsets, max_seqlen=torch.tensor(rows, device="cpu"))
        torch.xpu.synchronize()
        elapsed = time.perf_counter() - start
        repeated = operation(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0),
                             cu_seqlens=offsets, max_seqlen=torch.tensor(rows, device="cpu"))
        assert raw(value) == raw(repeated)
        return value.reshape(rows, 16, 72), elapsed

    cases = []
    for rows in (1, 5, 37):
        values = torch.arange(rows*16*72, device="xpu", dtype=torch.float32)
        q = ((values*7 % 101 - 50)*0.031).reshape(rows,16,72).half()
        k = ((values*13 % 71 - 35)*0.027).reshape(rows,16,72).half()
        v = ((values*19 % 61 - 30)*0.053).reshape(rows,16,72).half()
        result, elapsed = execute(q,k,v)
        number = len(cases)
        files = {name: save(out,f"attention-{number}-{name}.bin", value)
                 for name,value in (("q",q),("k",k),("v",v),("expected",result))}
        cases.append({"shape": [rows,16,72], "scale": 72**-0.5, "causal": False,
                      "files": files, "repeat_exact": True, "first_call_seconds": elapsed})
    identity = {"reference_image": "sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918",
                "torch": torch.__version__, "torch_git": torch.version.git_version,
                "device": torch.xpu.get_device_name(),
                "class": type(operation).__module__+"."+type(operation).__qualname__,
                "selected_method": operation._forward_method.__qualname__,
                "backend": str(operation.attn_backend), "fp8_enabled": operation.fp8_enabled,
                "contract": contract}
    (out/"attention.json").write_text(json.dumps({**identity,"cases":cases,"routes":list(routes)},indent=2)+"\n")
    print(f"VISION_ATTENTION_FIXTURES_PASS {len(cases)}", flush=True)

    if len(sys.argv)>2:
        import numpy as np
        capture = pathlib.Path(sys.argv[2])
        rope = json.loads((capture/"vision-rope-reference.json").read_text())
        qkv = json.loads((capture/"qkv-real-replays.json").read_text())
        replays = []
        for c in rope["cases"]:
            image = c["image"]
            inputs = {}
            files = {"q": c["q_out"], "k": c["k_out"], "v": qkv["cases"][image]["files"]["v"]}
            for name, entry in files.items():
                payload = (capture/entry["file"]).read_bytes()
                assert hashlib.sha256(payload).hexdigest()==entry["sha256"]
                inputs[name] = torch.from_numpy(np.frombuffer(payload,dtype=np.float16).copy()).reshape(768,16,72).to("xpu")
            result, elapsed = execute(inputs['q'],inputs['k'],inputs['v'])
            replays.append({"image":image,"files":{**files,"expected":save(capture,f"attention-real-image{image}.float16",result)},
                            "shape":[768,16,72],"scale":72**-0.5,"causal":False,
                            "repeat_exact":True,"first_call_seconds":elapsed})
        (capture/"attention-real-replays.json").write_text(json.dumps({**identity,"cases":replays,"routes":routes,
            "origin":"Standalone production MMEncoderAttention on the real-input rotated Q/K and V replays; not a new full-worker attention capture"},indent=2)+"\n")
        print("VISION_ATTENTION_REAL_REPLAYS_PASS",flush=True)
