#!/usr/bin/env python3
"""Capture actual reference M-RoPE for the bounded max/small image recipes.

Run only in the pinned Python reference environment, without GPU exposure.
The input is an executed native request receipt, not learned tensors. Grids
below are the qualified fixture processor shapes; this is a position capture,
not another pixel/resize or full-tower comparison.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import struct
from types import SimpleNamespace


def digest(values):
    return hashlib.sha256(struct.pack('<' + str(len(values)) + 'i', *values)).hexdigest()


def main():
    import torch
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration as Model

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--requests', required=True, type=Path)
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    config = json.loads(args.config.read_text())
    resolved = SimpleNamespace(vision_config=SimpleNamespace(**config['vision_config']),
        video_token_id=config['video_token_id'], vision_start_token_id=config['vision_start_token_id'],
        vision_end_token_id=config['vision_end_token_id'])

    class Proxy:
        _get_mrope_input_positions = staticmethod(Model._get_mrope_input_positions)

        def __init__(self):
            self.config = resolved

    receipt = json.loads(args.requests.read_text())
    if receipt['status'] != 'PASS':
        raise RuntimeError('failed request capture')
    result = {'scope': 'executed CPU reference positions for qualified max/small fixture grids; no learned inference',
              'reference_image': 'sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918',
              'source_sha256': hashlib.sha256(Path(inspect.getfile(Model._get_mrope_input_positions)).read_bytes()).hexdigest(),
              'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(), 'cases': []}
    for name, rows, grid in [('orbit-max-2048.png', 4096, [1, 128, 128]), ('comet.png', 192, [1, 24, 32])]:
        case = next(c for c in receipt['cases'] if c['file'] == name)
        tokens = case['expanded_tokens']
        offset = case['offset']
        if offset != 4 or case['visual_rows'] != rows or tokens[offset:offset + rows] != [248056] * rows:
            raise RuntimeError('wrong qualified recipe')
        feature = SimpleNamespace(modality='image', mm_position=SimpleNamespace(offset=offset, length=rows),
            data={'image_grid_thw': SimpleNamespace(data=torch.tensor(grid, dtype=torch.long))})
        positions, delta = Model.get_mrope_input_positions(Proxy(), tokens, [feature])
        if tuple(positions.shape) != (3, len(tokens)):
            raise RuntimeError('wrong reference position shape')
        result['cases'].append({'file': name, 'image_sha256': case['sha256'], 'offset': offset,
            'rows': rows, 'grid_thw': grid, 'prompt_tokens': len(tokens), 'tokens_sha256': digest(tokens),
            'positions_sha256': digest(positions.flatten().tolist()), 'delta': int(delta)})
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('CPU_MAX_SMALL_MROPE_PASS', len(result['cases']))


if __name__ == '__main__':
    main()
