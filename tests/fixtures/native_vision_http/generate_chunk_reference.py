#!/usr/bin/env python3
"""Run the installed pinned Qwen3.5 CPU request-position method.

Reference-only tool: requires Torch/vLLM in the recorded image, no GPU/weights.
Reports contain native /tokenize IDs and measured placeholder offsets. Output
stores compact digests; it does not serialize full prompts or model captures.
"""
import argparse
import hashlib
import inspect
import json
from pathlib import Path
import struct
from types import SimpleNamespace
import torch
from vllm.model_executor.models.qwen3_5 import Qwen3_5ForConditionalGeneration


def digest(values):
    return hashlib.sha256(struct.pack('<' + str(len(values)) + 'i', *values)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--requests', type=Path, nargs='+', required=True)
    parser.add_argument('--reference-image', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve earlier evidence')
    model = json.loads(args.model_config.read_text())
    config = SimpleNamespace(vision_config=SimpleNamespace(**model['vision_config']),
        video_token_id=model['video_token_id'], vision_start_token_id=model['vision_start_token_id'],
        vision_end_token_id=model['vision_end_token_id'])
    class Proxy:
        _get_mrope_input_positions = staticmethod(Qwen3_5ForConditionalGeneration._get_mrope_input_positions)
        def __init__(self):
            self.config = config
    result = {
        'reference_image': args.reference_image,
        'source_sha256': hashlib.sha256(Path(inspect.getfile(
            Qwen3_5ForConditionalGeneration._get_mrope_input_positions)).read_bytes()).hexdigest(),
        'scope': 'executed CPU request M-RoPE for six actual native HTTP boundary prompts; no weights/worker/GPU',
        'encoding': 'row-major 3xT signed int32 little-endian, SHA-256', 'cases': []}
    seen = set()
    for path in args.requests:
        request_run = json.loads(path.read_text())
        if request_run['status'] != 'PASS':
            raise RuntimeError('request run failed')
        for case in request_run['modes'][0]['cases']:
            offset = case['offset']
            if offset in seen:
                continue
            seen.add(offset)
            grid = [1, 24, 32]  # fixed 512x384 fixture, patch16/merge2
            if config.vision_config.patch_size != 16 or config.vision_config.spatial_merge_size != 2:
                raise RuntimeError('reference model does not match fixed fixture geometry')
            rows = 192
            before = case['tokenized_pre_expansion']
            if before[offset] != model['image_token_id']:
                raise RuntimeError('wrong placeholder offset')
            tokens = before[:offset] + [model['image_token_id']] * rows + before[offset + 1:]
            if len(tokens) != case['expected_prompt_tokens']:
                raise RuntimeError('wrong expanded length')
            feature = SimpleNamespace(modality='image',
                mm_position=SimpleNamespace(offset=offset, length=rows),
                data={'image_grid_thw': SimpleNamespace(data=torch.tensor(grid, dtype=torch.long))})
            positions, delta = Qwen3_5ForConditionalGeneration.get_mrope_input_positions(Proxy(), tokens, [feature])
            if tuple(positions.shape) != (3, len(tokens)):
                raise RuntimeError('reference returned wrong shape')
            result['cases'].append({'offset': offset, 'prompt_tokens': len(tokens), 'grid': grid,
                'length': rows, 'delta': int(delta), 'tokens_sha256': digest(tokens),
                'positions_sha256': digest([value for axis in positions.tolist() for value in axis])})
    if len(result['cases']) != 6:
        raise RuntimeError('six distinct boundary offsets required')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('CPU_CHUNK_MROPE_CAPTURE_PASS', len(result['cases']))


if __name__ == '__main__':
    main()
