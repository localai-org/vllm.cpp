#!/usr/bin/env python3
"""Reproduce pinned CPU M-RoPE digests from actual native history tokenize IDs.

Reference-only, no learned inference, GPU or weights. The supervisor must use
the declared immutable reference environment. Output preserves frozen history
text, token/position hashes and the executing installed method's source hash.
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


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(values):
    return hashlib.sha256(struct.pack('<' + str(len(values)) + 'i', *values)).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('model_config', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--reference-image', required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve evidence')
    require(args.reference_image == 'sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918', 'wrong declared pinned runtime')
    source_sha = hashlib.sha256(Path(inspect.getfile(Qwen3_5ForConditionalGeneration._get_mrope_input_positions)).read_bytes()).hexdigest()
    require(source_sha == '22f02a1ee1faba276a93243d23ecada1094484e39e5bde79452d8a44f32238ba', 'executing M-RoPE source differs')
    report = json.loads(args.requests.read_text())
    require(report['status'] == 'PASS' and report['mode'] == 'history' and len(report['cases']) == 2,
            'successful actual cold/warm history report required')
    model = json.loads(args.model_config.read_text())
    require(model['vision_config']['patch_size'] == 16 and model['vision_config']['spatial_merge_size'] == 2,
            'wrong frozen image geometry')
    config = SimpleNamespace(vision_config=SimpleNamespace(**model['vision_config']),
        video_token_id=model['video_token_id'], vision_start_token_id=model['vision_start_token_id'],
        vision_end_token_id=model['vision_end_token_id'])
    class Proxy:
        _get_mrope_input_positions = staticmethod(Qwen3_5ForConditionalGeneration._get_mrope_input_positions)
        def __init__(self):
            self.config = config
    before = report['tokenized_pre_expansion']
    image_token = model['image_token_id']
    require(before.count(image_token) == 1 and before.index(image_token) == 1599, 'wrong actual image boundary')
    tokens = before[:1599] + [image_token] * 192 + before[1600:]
    require(len(tokens) == report['expected_prompt_tokens'] and digest(tokens) == report['expanded_prompt_sha256'],
            'actual report token digest differs')
    grid = [1, 24, 32]
    feature = SimpleNamespace(modality='image', mm_position=SimpleNamespace(offset=1599, length=192),
        data={'image_grid_thw': SimpleNamespace(data=torch.tensor(grid, dtype=torch.long))})
    positions, delta = Qwen3_5ForConditionalGeneration.get_mrope_input_positions(Proxy(), tokens, [feature])
    require(tuple(positions.shape) == (3, len(tokens)), 'wrong executed positions shape')
    prior = report['history_assistant_text']
    result = {'reference_image': args.reference_image, 'source_sha256': source_sha,
        'scope': 'executed CPU M-RoPE for a frozen one-image conversation history; no learned-inference quality proof',
        'encoding': 'row-major 3xT signed int32 little-endian, SHA-256', 'assistant_text': prior,
        'assistant_text_sha256': hashlib.sha256(prior.encode()).hexdigest(),
        'cases': [{'offset': 1599, 'prompt_tokens': len(tokens), 'grid': grid, 'length': 192,
            'delta': int(delta), 'tokens_sha256': digest(tokens),
            'positions_sha256': digest([v for axis in positions.tolist() for v in axis])}]}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print('CPU_HISTORY_MROPE_CAPTURE_PASS', len(tokens))


if __name__ == '__main__':
    main()
