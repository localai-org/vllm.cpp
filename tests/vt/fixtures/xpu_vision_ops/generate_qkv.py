"""Execute pinned Torch XPU QKV splits; optionally replay real QKV projection."""
import hashlib
import json
import pathlib
import sys

import torch
import torch.nn.functional as F

out = pathlib.Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)

def raw(value):
    return value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()

def save(directory, filename, value):
    payload=raw(value)
    (directory/filename).write_bytes(payload)
    return {"file":filename,"sha256":hashlib.sha256(payload).hexdigest()}

cases=[]
for dtype,rows,widths in ((torch.float16,5,[19,31,13]), (torch.bfloat16,3,[3,7,7]),
                          (torch.float32,7,[5,11,3]), (torch.float16,3,[1152]*3)):
    is32=dtype==torch.float32
    patterns=[0,0x80000000,0x7f800000,0xff800000,0x7fc00123,0x7f800001,1,0x3f800000] if is32 else [0,0x8000,0x7c00,0xfc00,0x7e01,0x7c01,1,0x3c00]
    if dtype==torch.bfloat16:patterns=[0,0x8000,0x7f80,0xff80,0x7fc1,0x7f81,1,0x3f80]
    integer=torch.uint32 if is32 else torch.uint16
    bits=[patterns[i%len(patterns)] for i in range(rows*sum(widths))]
    value=torch.tensor(bits,dtype=integer).view(dtype).reshape(rows,sum(widths)).to('xpu')
    split=torch.split(value,widths,dim=-1)
    repeat=torch.split(value,widths,dim=-1)
    assert all(raw(a)==raw(b) for a,b in zip(split,repeat))
    files={"input":save(out,f"qkv-{len(cases)}-input.bin",value)}
    for name,tensor in zip(('q','k','v'),split):
        files[name]=save(out,f"qkv-{len(cases)}-{name}.bin",tensor)
    cases.append({"dtype":str(dtype).removeprefix('torch.'),"rows":rows,"widths":widths,
                  "files":files,"repeat_exact":True})
manifest={"reference_image":"sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918",
          "torch":torch.__version__,"torch_git":torch.version.git_version,
          "device":torch.xpu.get_device_name(),"operator":"torch.split(...,dim=-1), contiguous output bytes",
          "cases":cases}
(out/'qkv.json').write_text(json.dumps(manifest,indent=2)+'\n')
print(f"Executed and repeated {len(cases)} typed QKV split cases",flush=True)

if len(sys.argv)>3:
    import numpy as np
    from safetensors import safe_open
    capture,model=pathlib.Path(sys.argv[2]),pathlib.Path(sys.argv[3])
    boundaries=json.loads((capture/'vision-boundaries-capture.json').read_text())
    inputs=[e for e in boundaries['worker']['captures'] if e['name']=='block0-norm1-output']
    assert len(inputs)==2
    index=json.loads((model/'model.safetensors.index.json').read_text())["weight_map"]
    key="model.visual.blocks.0.attn.qkv"
    with safe_open(model/index[key+'.weight'],framework='pt',device='cpu') as f:
        weight=f.get_tensor(key+'.weight').to(dtype=torch.float16,device='xpu')
        bias=f.get_tensor(key+'.bias').to(dtype=torch.float16,device='xpu')
    replays=[]
    for i,entry in enumerate(inputs):
        a=torch.from_numpy(np.fromfile(capture/entry['file'],dtype=np.float16).copy())
        a=a.reshape(768,1152).to('xpu')
        value=F.linear(a,weight,bias)
        payload=raw(value)
        repeats=[payload==raw(F.linear(a,weight,bias)) for _ in range(2)]
        files={"merged":save(capture,f'qkv-real-image{i}-merged.float16',value)}
        for name,tensor in zip(('q','k','v'),torch.split(value,[1152]*3,dim=-1)):
            files[name]=save(capture,f'qkv-real-image{i}-{name}.float16',tensor)
        replays.append({"image":i,"input":entry,"files":files,"standalone_repeat_exact":repeats})
    receipt={**{k:manifest[k] for k in ('reference_image','torch','torch_git','device')},
             "origin":"Standalone F.linear and split on actual first-block norm outputs with checkpoint FP16 QKV parameters; not a new full-worker QKV capture",
             "contract":{"rel_l2":0.0003,"max_abs":0.02},"key":key,"rows":768,"hidden":1152,
             "weight_sha256":hashlib.sha256(raw(weight)).hexdigest(),
             "bias_sha256":hashlib.sha256(raw(bias)).hexdigest(),"cases":replays}
    (capture/'qkv-real-replays.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print("Executed two real-input QKV projection/split replays",flush=True)
