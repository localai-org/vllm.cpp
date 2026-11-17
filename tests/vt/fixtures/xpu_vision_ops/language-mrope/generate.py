import hashlib
import inspect
import json
import sys
from pathlib import Path
import torch
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.model_executor.layers.rotary_embedding.mrope import MRotaryEmbedding

root = Path(sys.argv[1])
root.mkdir(parents=True, exist_ok=True)
def save(name, tensor):
    data = tensor.detach().cpu().contiguous().numpy().tobytes()
    (root / name).write_bytes(data)
    return {'file': name, 'sha256': hashlib.sha256(data).hexdigest()}

metadata = {
    'reference_image': 'sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918',
    'torch': torch.__version__, 'torch_git': torch.version.git_version,
    'device': torch.xpu.get_device_name(),
    'class': 'vllm.model_executor.layers.rotary_embedding.mrope.MRotaryEmbedding',
    'selected_method': 'forward_xpu', 'cache_construction_device': 'xpu',
    'scope': 'bounded executed operator; full language-model worker cache initialization not witnessed',
    'source_sha256': hashlib.sha256(Path(inspect.getfile(MRotaryEmbedding)).read_bytes()).hexdigest(),
    'head_size': 256, 'rotary_dim': 64, 'base': 10000000.0,
    'is_neox_style': True, 'cache_dtype': 'float16',
    'contract': {'bytes': 'exact'}, 'cases': [],
}
# The gate is fixed before native evaluation: selected FP16 coefficients must
# match the executing pinned XPU Triton M-RoPE, including long-position tails.
positions = torch.tensor([[0,1,3,17,31,127,2047,8191],
                          [0,7,9,53,83,511,4095,8190],
                          [0,2,5,19,37,251,1023,8189]], dtype=torch.int64)
for number, (section, interleaved, equal_axes) in enumerate([
        ([11,11,10],True,False), ([11,11,10],False,False),
        ([11,11,10],True,True), ([1,20,11],False,False),
        ([32,0,0],True,False), ([0,16,16],False,False),
        ([11,11,10],True,False), ([11,11,10],False,False)]):
    high = number >= 6
    source_positions = torch.tensor([
        [0,1,8191,32767,65535,131071,262142,262143],
        [0,5,8189,32769,65533,131073,262141,262140],
        [0,7,8193,32765,65537,131069,262139,262138]],dtype=torch.int64) if high else positions
    with set_current_vllm_config(VllmConfig()), torch.device('xpu'):
        rope = MRotaryEmbedding(256,64,65536 if high else 4096,10000000.0,True,torch.float16,
                               section,interleaved)
    p = (source_positions[0:1].expand(3,-1) if equal_axes else source_positions).contiguous().to('xpu')
    # A single nonzero half of each Neox rotary pair directly exposes the
    # selected coefficient from the real forward_xpu, with exact 1* and 0*.
    q = torch.zeros((8,256), dtype=torch.float16, device='xpu')
    q[:,:32] = 1
    out, _ = rope.forward_xpu(p,q,q.clone())
    cache = out[:,:64].contiguous()
    c = {'section': section, 'interleaved': interleaved, 'tokens': 8,
         'equal_axes': equal_axes, 'positions_dtype': 'int64',
         'cache_rows': int(rope.cos_sin_cache.shape[0]),
         'files': {'positions': save(f'language-mrope-{number}-positions.bin',p),
                   'expected': save(f'language-mrope-{number}-cache.bin',cache)}}
    assert rope.cos_sin_cache.device.type == 'xpu'
    assert torch.equal(cache[0,:32],torch.ones_like(cache[0,:32]))
    assert torch.equal(cache[0,32:],torch.zeros_like(cache[0,32:]))
    metadata['cases'].append(c)
torch.xpu.synchronize()
(root/'language-mrope.json').write_text(json.dumps(metadata,indent=2)+'\n')
print(json.dumps({'cases':len(metadata['cases']), 'status':'PASS', 'metadata':metadata},indent=2))
