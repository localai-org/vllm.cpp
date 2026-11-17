#!/usr/bin/env python3
"""Check supervised boundary runs and exact target/MTP3/chunk-budget IDs.

Run-list JSON names existing request/trace/execution files and declared options.
Every run first passes the unchanged position/slice checker. Not tensor parity.
"""
import argparse
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

from check_native_exl3_vision_chunk_trace import require, run as check_metadata


def run(args, result):
    runs = json.loads(args.runs.read_text())
    require(2 <= len(runs) <= 8, 'two to eight bounded runs required')
    eos = json.loads(args.generation_config.read_text())['eos_token_id']
    eos = {eos} if isinstance(eos, int) else set(eos)
    require(eos and all(isinstance(v, int) for v in eos), 'missing EOS contract')
    variants, baselines, binaries, heads, tokens_checked = set(), {}, set(), set(), 0
    for entry in runs:
        boundary, budget, depth = (entry[k] for k in ('boundary', 'budget', 'depth'))
        require(boundary in (1600, 4096) and budget in (1600, 4096) and depth in (0, 3), 'wrong declared cohort')
        variant = (boundary, budget, depth)
        require(variant not in variants, 'duplicate cohort')
        variants.add(variant)
        def path(key):
            value = Path(entry[key])
            return value if value.is_absolute() else args.runs.parent / value
        request, trace, execution = path('requests'), path('trace'), path('execution')
        report = json.loads(request.read_text())
        require(report['boundary'] == boundary, 'wrong declared boundary')
        require(report['draft_tokens'] > 0 if depth else report['draft_tokens'] == 0,
                'missing/unexpected actual draft observation')
        metadata = {'cases': []}
        check_metadata(SimpleNamespace(requests=request, trace_log=trace, execution=execution,
                       budget=budget, spec_depth=depth, reference=args.reference), metadata)
        binaries.add(metadata['binary_sha256'])
        heads.add(metadata['execution_head'])
        require(len(binaries) == len(heads) == 1, 'pair must use the same binary/source head')
        mode = next(m for m in report['modes'] if m['budget'] == budget)
        require([c['offset'] for c in mode['cases']] == list(range(boundary - 1, boundary + 2)), 'wrong ordered boundary cases')
        ids, graphs = {}, []
        for line in trace.read_text().splitlines():
            if line.startswith('NATIVE_VISION_SAMPLED '):
                item = json.loads(line.split(' ', 1)[1])
                ids.setdefault(item['request_id'], []).extend(item['token_ids'])
            elif line.startswith('NATIVE_VISION_GRAPH '):
                graphs.append(json.loads(line.split(' ', 1)[1]))
        require(set(ids) == {'chatcmpl-' + str(i) for i in range(3)}, 'missing fresh three-request sample trace')
        require(graphs and all(g['requests'] == 1 and g['tokens'] == (4 if depth else 1) for g in graphs)
                and any(g['captured'] and g['replay_count'] > 1 for g in graphs),
                'missing progressing C1 verification graph')
        for index, case in enumerate(mode['cases']):
            response = case['response']
            require(response['id'] == 'chatcmpl-' + str(index), 'wrong response identity')
            raw = ids[response['id']]
            count = response['usage']['completion_tokens']
            require(all(isinstance(t, int) and 0 <= t < 248320 for t in raw), 'invalid sampled ID')
            require(count <= len(raw) <= count + depth, 'wrong committed sampler count')
            first_eos = next((i + 1 for i, t in enumerate(raw) if t in eos), None)
            choice = response['choices'][0]
            if choice['finish_reason'] == 'stop':
                require(first_eos == count, 'first EOS differs from reported completion boundary')
            else:
                require(choice['finish_reason'] == 'length' and count == 64 and
                        (first_eos is None or first_eos >= count), 'wrong length stop boundary')
            committed = raw[:count]
            observed = (committed, response['usage'], choice['message']['content'], choice['finish_reason'])
            offset = case['offset']
            if offset in baselines:
                require(observed == baselines[offset], 'native decision/text/usage differs at offset ' + str(offset))
            else:
                baselines[offset] = observed
            tokens_checked += count
        result['runs'].append({'boundary': boundary, 'budget': budget, 'depth': depth,
                               'draft_tokens': report['draft_tokens'], 'metadata': metadata})
    # Every boundary/budget represented must contain its paired target/MTP3 run.
    require(all((edge, budget, 3 - depth) in variants for edge, budget, depth in variants), 'missing same-budget target/MTP3 pair')
    result.update(committed_ids_checked=tokens_checked, image_offsets=sorted(baselines),
                  cross_mode_and_present_chunk_budget_ids_exact=True,
                  generation_config_sha256=hashlib.sha256(args.generation_config.read_bytes()).hexdigest())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, required=True)
    parser.add_argument('--generation-config', type=Path, required=True)
    parser.add_argument('--reference', type=Path, default=Path(__file__).resolve().parents[2] /
                        'tests/fixtures/native_vision_http/chunk-mrope-reference.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    result = {'status': 'FAIL', 'runs': [], 'scope': 'bounded same-binary C1 native boundary metadata/committed IDs; not Python logits or full tower parity'}
    code = 1
    try:
        run(args, result)
        result['status'], code = 'PASS', 0
    except Exception as error:
        result['error'] = repr(error)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'status': result['status'], 'committed_ids_checked': result.get('committed_ids_checked'), 'error': result.get('error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
