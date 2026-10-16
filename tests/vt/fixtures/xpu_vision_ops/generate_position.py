"""Execute the pinned production position method on XPU; no public weights."""
import hashlib
import json
import pathlib
import sys
from types import SimpleNamespace

import torch
import vllm.model_executor.models.qwen3_vl as vision

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
contract = {"rel_l2": 0.0003, "max_abs": 0.002}
route = "triton_pos_embed_interpolate" if vision.HAS_TRITON else "pos_embed_interpolate_native"
original = getattr(vision, route)
calls = []

def observed(*args, **kwargs):
    result = original(*args, **kwargs)
    calls.append({"route": route, "grid": list(args[1:4]), "dtype": str(result.dtype),
                  "device": str(result.device)})
    return result

setattr(vision, route, observed)

def raw(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

def execute(table, grid, side, merge):
    proxy = SimpleNamespace(pos_embed=SimpleNamespace(weight=table),
                            num_grid_per_side=side, spatial_merge_size=merge,
                            dtype=table.dtype)
    # Execute the unchanged production method, including its route selection.
    return vision.Qwen3_VisionTransformer.fast_pos_embed_interpolate(proxy, [grid])

cases = []
try:
    for grid, side, hidden, merge in (([1,4,6],4,9,2), ([2,2,4],5,37,2),
                                     ([1,24,32],48,9,2), ([1,1,1],1,7,1)):
        table = ((torch.arange(side*side*hidden)*13+5) % 59 - 29).float() / 113
        table = table.reshape(side*side, hidden).to(device="xpu", dtype=torch.float16)
        value = execute(table, grid, side, merge)
        assert raw(value) == raw(execute(table, grid, side, merge))
        files = {}
        for name, tensor in (("table", table), ("expected", value)):
            payload = raw(tensor)
            filename = f"position-{len(cases)}-{name}.bin"
            (out/filename).write_bytes(payload)
            files[name] = {"file": filename, "sha256": hashlib.sha256(payload).hexdigest()}
        cases.append({"grid":grid, "side":side, "hidden":hidden, "merge":merge,
                      "files":files, "repeat_exact":True})
    manifest = {"reference_image":"sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918",
                "torch":torch.__version__, "torch_git":torch.version.git_version,
                "device":torch.xpu.get_device_name(), "dtype":"float16",
                "source_sha256":hashlib.sha256(pathlib.Path(vision.__file__).read_bytes()).hexdigest(),
                "has_triton":vision.HAS_TRITON, "route":route, "executed_calls":calls.copy(),
                "contract":contract, "cases":cases}
    (out/'position.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f"Executed and repeated {len(cases)} production position cases via {route}", flush=True)

    if len(sys.argv) > 3:
        from safetensors import safe_open
        capture, model = pathlib.Path(sys.argv[2]), pathlib.Path(sys.argv[3])
        index = json.loads((model/'model.safetensors.index.json').read_text())["weight_map"]
        key = "model.visual.pos_embed.weight"
        with safe_open(model/index[key],framework="pt",device="cpu") as f:
            table = f.get_tensor(key).to(device="xpu",dtype=torch.float16)
        table_payload = raw(table)
        table_name = "position-real-table.float16"
        (capture/table_name).write_bytes(table_payload)
        boundaries = json.loads((capture/'vision-boundaries-capture.json').read_text())
        actual = [e for e in boundaries['worker']['captures'] if e['name']=='encoder-metadata-pos_embeds']
        assert len(actual)==2
        replays=[]
        for i, expected in enumerate(actual):
            value=execute(table,[1,24,32],48,2)
            payload=raw(value)
            assert payload==raw(execute(table,[1,24,32],48,2))
            same=payload==(capture/expected['file']).read_bytes()
            assert same, "standalone method disagrees with actual worker metadata"
            name=f"position-real-image{i}.float16"
            (capture/name).write_bytes(payload)
            replays.append({"image":i,"expected":expected,"standalone_file":name,
                            "standalone_matches_worker_bytes":same})
        receipt={**{k:manifest[k] for k in ('reference_image','torch','torch_git','source_sha256','has_triton','route','contract')},
                 "origin":"Unchanged production fast_pos_embed_interpolate method on checkpoint FP16 table; compared with two saved actual worker metadata boundaries",
                 "grid":[1,24,32],"side":48,"merge":2,"hidden":1152,
                 "table":{"file":table_name,"sha256":hashlib.sha256(table_payload).hexdigest()},
                 "executed_calls":calls[len(manifest['executed_calls']):],"cases":replays}
        (capture/'position-real-replays.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print("Production position method matches both actual worker boundaries byte-for-byte",flush=True)
finally:
    setattr(vision,route,original)
