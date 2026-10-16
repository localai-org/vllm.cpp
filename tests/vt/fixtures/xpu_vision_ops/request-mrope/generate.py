import argparse
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration

parser=argparse.ArgumentParser(description='Execute pinned Qwen3.5 request M-RoPE generation')
parser.add_argument('model_config',type=Path)
parser.add_argument('output_directory',type=Path)
args=parser.parse_args()
root=args.output_directory; root.mkdir(parents=True,exist_ok=True)
model=json.loads(args.model_config.read_text())
config=SimpleNamespace(vision_config=SimpleNamespace(**model['vision_config']),
    video_token_id=model['video_token_id'],vision_start_token_id=model['vision_start_token_id'],
    vision_end_token_id=model['vision_end_token_id'])
class Proxy:
    _get_mrope_input_positions=staticmethod(Qwen3_5ForConditionalGeneration._get_mrope_input_positions)
    def __init__(self):self.config=config
record={'reference_image':'sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918',
    'class':'vllm.model_executor.models.qwen3_5.Qwen3_5ForConditionalGeneration',
    'method':'get_mrope_input_positions','scope':'executed inherited CPU request-position generator, no weights or worker forward',
    'source_sha256':hashlib.sha256(Path(inspect.getfile(Qwen3_5ForConditionalGeneration._get_mrope_input_positions)).read_bytes()).hexdigest(),
    'spatial_merge_size':config.vision_config.spatial_merge_size,'contract':{'integers':'exact','delta':'exact'},'cases':[]}
for number,(length,spans) in enumerate([
    (12,[]),(200,[(0,[1,24,32])]),(221,[(17,[1,24,32])]),
    (211,[(19,[1,24,32])]),(260,[(230,[1,8,12]),(2,[1,24,32])])]):
    tokens=[100+i for i in range(length)]
    features=[]
    for offset,grid in spans:
        n=grid[1]*grid[2]//4
        tokens[offset:offset+n]=[model['image_token_id']]*n
        features.append(SimpleNamespace(modality='image',mm_position=SimpleNamespace(offset=offset,length=n),
            data={'image_grid_thw':SimpleNamespace(data=torch.tensor(grid,dtype=torch.long))}))
    pos,delta=Qwen3_5ForConditionalGeneration.get_mrope_input_positions(Proxy(),tokens,features)
    assert tuple(pos.shape)==(3,length)
    record['cases'].append({'tokens':tokens,'spans':[{'offset':o,'grid':g,'length':g[1]*g[2]//4} for o,g in spans],
        'positions':pos.tolist(),'delta':int(delta)})
(root/'request-mrope.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps({'status':'PASS','cases':len(record['cases']),'method':record['method']}))
