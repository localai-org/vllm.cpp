#!/usr/bin/env python3
"""Pinned exact-K oneDNN on real layer3 Q/FP8 bytes and actual hybrid strides.

Repeated real operands provide page-boundary/large operator coverage, not a new
full-worker long-context trajectory. Unused physical tokens are poisoned.
"""
import argparse
import json
from pathlib import Path

from capture_projection import IMAGE
from capture_runtime_layout import verify_inputs
from extract_projection import digest, headers, write_safetensors


def capture(args):
    headers.require(not args.output.exists() and not args.output.with_suffix('.json').exists(),
                    'refusing to overwrite exact-K fixture')
    verify_inputs(args.reference_manifest, args.model, args.image_identity)
    source_receipt = json.loads(args.source.with_suffix('.json').read_text())
    headers.require(source_receipt['image'] == IMAGE and
                    digest(args.source.read_bytes()) == source_receipt['capture_sha256'],
                    'real attention operand identity mismatch')
    import torch
    from safetensors.torch import load_file
    from exl3xpu.ops import _get_esimd
    from exl3xpu.fp8kv_prefill import prefill_attention_onednn, L_BUCKET, Q_BUCKET
    headers.require(L_BUCKET == 1 and Q_BUCKET == 256, 'exact K / query bucket256 required')
    original = _get_esimd()
    host = load_file(str(args.source))
    base_q = host['p128_q_rope'].reshape(128, 24, 256)
    base_k, base_v = host['p128_key_bytes_after'], host['p128_value_bytes_after']
    tensors, cases = {}, []

    def save(name, value):
        value = value.detach().cpu().contiguous()
        tensors[name] = ({torch.float16: 'F16', torch.int32: 'I32', torch.uint8: 'U8'}[value.dtype],
                         list(value.shape), value.numpy().tobytes())

    for index, (length, rows, scales) in enumerate(((1599, 3, (1., 1., .0625)),
            (1600, 129, (.5, 1.5, .125)), (1601, 256, (1., 1., .0625)),
            (4096, 4096, (1., 1., .0625)))):
        blocks = (length + 1599) // 1600
        owner = torch.full((blocks, 1600, 4, 512), 0x7f, dtype=torch.uint8, device='xpu')
        keys = owner[..., :256].view(torch.float8_e4m3fn)
        values = owner[..., 256:].view(torch.float8_e4m3fn)
        pages = torch.arange(blocks-1, -1, -1, dtype=torch.int32, device='xpu')
        for logical in range(blocks):
            begin, end = logical*1600, min(length, (logical+1)*1600)
            ids = torch.arange(begin, end) % 128
            physical = blocks-1-logical
            keys[physical, :end-begin].view(torch.uint8).copy_(base_k[ids].to('xpu'))
            values[physical, :end-begin].view(torch.uint8).copy_(base_v[ids].to('xpu'))
        query = base_q[torch.arange(rows) % 128].to('xpu').contiguous()
        output = torch.empty_like(query)
        prefill_attention_onednn(original, query, keys, values, pages, length, *scales, output)
        headers.require(torch.isfinite(output).all().item(), 'original exact-K output nonfinite')
        label = f'case{index}'
        for name, tensor in (('query', query), ('cache', owner), ('pages', pages), ('output', output)):
            save(label+'_'+name, tensor)
        cases.append({'label': label, 'logical_q': rows, 'padded_q': (rows+255)//256*256,
                      'exact_k': length, 'page': 1600, 'blocks': blocks,
                      'k_scale': scales[0], 'v_scale': scales[1], 'attention_scale': scales[2],
                      'cache_shape': list(keys.shape), 'cache_strides': list(keys.stride()),
                      'value_offset_bytes': 256})
        print('EXACT_K_ORIGINAL', cases[-1], flush=True)
    write_safetensors(args.output, tensors, {'scope': __doc__})
    result = {'schema': 1, 'image': IMAGE, 'cases': cases, 'scope': __doc__,
              'source_sha256': digest(args.source.read_bytes()),
              'capture_sha256': digest(args.output.read_bytes()),
              'tool_sha256': digest(Path(__file__).read_bytes()), 'torch': torch.__version__}
    with args.output.with_suffix('.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--reference-manifest', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--image-identity', required=True)
    parser.add_argument('--output', type=Path, required=True)
    capture(parser.parse_args())
