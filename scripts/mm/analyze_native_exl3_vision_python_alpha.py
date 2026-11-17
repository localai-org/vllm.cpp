#!/usr/bin/env python3
"""CPU validation of pinned Python same-input C1/M4 self-control.

No tensor tolerance, native quality waiver, or performance gate is assigned.
The captured head precedes sampling masks; forced sampled IDs are not evidence.
"""
import argparse
import json
import math
from pathlib import Path

from analyze_native_exl3_vision_adjacent_prefix import PREFIX
from analyze_native_exl3_vision_mtp_prefix import PROMPT_IDS
from analyze_native_exl3_vision_prefix import digest, floats, metrics, require
from capture_native_exl3_vision_reference import EXPECTED


STATE_NAMES = ({f'gdn{i}-{kind}' for i in range(48) for kind in ('conv', 'ssm')} |
               {f'attn{i}-{kind}' for i in range(16) for kind in ('k', 'v')})
PHASES = {'c1_pre', 'c1_post', 'c1_logits', 'c4_pre', 'c4_restored', 'c4_post', 'c4_logits'}


def validate(directory, report):
    require(report['status'] == 'CAPTURED' and report['runtime'] == EXPECTED, 'complete pinned reference required')
    require(report['environment'].get('EXL3_INT8_PREFILL') == '1' and
            report['environment'].get('EXL3_TARGET_RUNTIME') == 'vllm030', 'pinned reference profile differs')
    require(report['recipe']['alpha']['ids'] == PROMPT_IDS + PREFIX[:-1] and
            report['recipe']['alpha']['query_token'] == 2702, 'wrong actual selected prefix/query')
    require(report['prompt_lengths'] == [39, 38, 265, 265], 'sibling prompt geometry differs')
    frames = report['frames']
    require(set(frames) == PHASES, 'missing or extra reference phase')
    seen = set()
    bindings = []
    for phase, frame in frames.items():
        blobs = frame['blobs']
        is_logits = phase.endswith('logits')
        require(set(blobs) == ({'logits'} if is_logits else STATE_NAMES), 'incomplete reference state/head')
        info = frame['info']
        n = 1 if phase.startswith('c1_') else 4
        row = info['row']
        require(info['rows'] == n and type(row) is int and 0 <= row < n, 'wrong selected reference row')
        context = 39 if phase.endswith('post') else 38
        if not is_logits:
            require(len(info['positions']) == 3 and all(len(axis) == n for axis in info['positions']) and
                    info['input_token_ids'][row] == 2702 and
                    [axis[row] for axis in info['positions']] == [38, 38, 38], 'selected input differs')
            require(sorted(info['input_token_ids']) == ([2702] if n == 1 else [11, 11, 2702, 11846]),
                    'actual sibling queries differ')
            md = list(info['metadata'].values())
            require(len(md) == 64 and sum(x['kind'] == 'gdn' for x in md) == 48 and
                    sum(x['kind'] == 'attention' for x in md) == 16, 'incomplete cache ownership')
            for item in md:
                if item['kind'] == 'attention':
                    require(item['cache_dtype'] == 'torch.uint8' and item['implementation'] == 'FlashAttentionImpl' and
                            item['block_size'] == 1600 and len(item['cache_shape']) == 4 and
                            item['cache_shape'][1:] == [4, 1600, 512], 'actual packed FP8 cache contract differs')
                    require(item['scales'] == [1.0, 1.0] and item['seq_lens'][row] == 39 and
                            sorted(item['seq_lens']) == ([39] if n == 1 else [39, 40, 266, 266]),
                            'attention geometry/scales differ')
                    require(len(item['physical_pages']) == n and len(set(item['physical_pages'])) == n,
                            'aliased attention request owners')
                else:
                    require(item['num_actual_tokens'] == n and len(item['slots']) == n and
                            len(set(item['slots'])) == n and item['slots'][row] == item['slot'],
                            'aliased recurrent request owners')
        total = 0
        for name, entry in blobs.items():
            filename = entry['file']
            require(Path(filename).name == filename and filename not in seen, 'unsafe or reused payload')
            seen.add(filename)
            shape = entry['shape']
            require(shape and all(type(x) is int and x > 0 for x in shape), 'invalid reference shape')
            if name == 'logits':
                # The pinned EXL3 head returns FP16; conversion for sampling
                # occurs later. Preserve the original unmasked tensor dtype.
                dtype, count = 'torch.float16', 248320
                require(shape == [count], 'partial reference head')
            elif name.endswith('conv'):
                dtype, count = 'torch.float16', 30720
            elif name.endswith('ssm'):
                dtype, count = 'torch.float32', 786432
            else:
                dtype, count = 'torch.uint8', context * 4 * 256
                require(shape == [context, 4, 256], 'wrong logical FP8 KV history')
            width = {'torch.float32': 4, 'torch.float16': 2, 'torch.uint8': 1}[dtype]
            require(entry['dtype'] == dtype and math.prod(shape) == count and entry['bytes'] == count * width,
                    'reference dtype/extent differs')
            total += entry['bytes']
            path = directory / filename
            require(path.stat().st_size == entry['bytes'] and digest(path) == entry['sha256'], 'payload size/checksum mismatch')
            bindings.append({'phase': phase, 'name': name, **entry})
        require(0 < total <= 256 * 1024 * 1024, 'reference payload bound exceeded')
    for name in STATE_NAMES:
        a, b = (frames[p]['blobs'][name] for p in ('c1_pre', 'c4_restored'))
        require(all(a[k] == b[k] for k in ('shape', 'dtype', 'bytes', 'sha256')), 'incoming reference states differ')
    require(frames['c1_pre']['info']['selected_embedding_sha256'] ==
            frames['c4_pre']['info']['selected_embedding_sha256'], 'incoming reference embeddings differ')
    return bindings


def values(directory, entry):
    typed = dict(entry, dtype={'torch.float32': 0, 'torch.float16': 1}[entry['dtype']])
    return floats(directory, typed)


def analyze(directory, report):
    bindings = validate(directory, report)
    frames = report['frames']
    heads = [values(directory, frames[phase]['blobs']['logits']) for phase in ('c1_logits', 'c4_logits')]
    require(all(math.isfinite(x) for head in heads for x in head), 'nonfinite complete head')
    top = []
    for head in heads:
        best = sorted(range(len(head)), key=lambda i: (-head[i], i))[:2]
        top.append({'ids': best, 'logits': [head[i] for i in best], 'margin': head[best[0]] - head[best[1]]})
    delta = metrics(*heads)
    changed = sorted(name for name in STATE_NAMES if
                     frames['c1_post']['blobs'][name]['sha256'] != frames['c4_post']['blobs'][name]['sha256'])
    groups = {kind: {'different': sum(name.endswith('-' + kind) for name in changed),
                     'total': sum(name.endswith('-' + kind) for name in STATE_NAMES)}
              for kind in ('conv', 'ssm', 'k', 'v')}
    first = {name: metrics(*(values(directory, frames[phase]['blobs'][name]) for phase in ('c1_post', 'c4_post')))
             for name in ('gdn0-conv', 'gdn0-ssm')}
    return {'status': 'DIAGNOSTIC', 'scope': __doc__.strip(), 'result_sha256': digest(directory / 'result.json'),
            'incoming_states_byte_exact': 128, 'different_post_states': changed, 'post_state_groups': groups,
            'full_head': delta, 'top2': top, 'selected_argmax_differs': top[0]['ids'][0] != top[1]['ids'][0],
            'c1_margin_gt_twice_logit_delta': top[0]['margin'] > 2 * delta['max_abs'],
            'first_gdn_post': first, 'layouts': [frames[p]['info'] for p in ('c1_pre', 'c4_pre')], 'bindings': bindings,
            'limitations': ['Teacher-forced query; sampled IDs are not a greedy quality check.',
                            'Actual Python C1 states restored in M4; not Python versus native cache parity.',
                            'Different initial mixed prefills are excluded from this local arithmetic control.',
                            'No waiver of the historical strict native mixed token failure or tower gates.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), 'preserve prior analysis')
    report = {'status': 'FAIL'}
    try:
        report = analyze(args.capture, json.loads((args.capture / 'result.json').read_text()))
    except Exception as error:
        report['error'] = repr(error)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report.get(k) for k in ('status', 'error', 'full_head', 'top2', 'post_state_groups')}))
    return 0 if report['status'] == 'DIAGNOSTIC' else 1


if __name__ == '__main__':
    raise SystemExit(main())
