#!/usr/bin/env python3
"""Score the four direct C-ABI image pairs and compare committed sampler IDs.

Sampler traces precede scheduler EOS trimming. Require the first configured
EOS at the reported completion boundary; do not blindly truncate a mismatch.
"""
import argparse
import hashlib
import json
from pathlib import Path

from native_exl3_vision_facts import evaluate


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def run(args, result):
    bundle = json.loads(args.bundle.read_text())
    require([c['name'] for c in bundle] == ['backup-dialog', 'invoice-unaligned',
        'ordered-orbit-comet', 'ordered-comet-orbit'], 'wrong frozen request wave')
    eos = json.loads(args.generation_config.read_text())['eos_token_id']
    eos = {eos} if isinstance(eos, int) else set(eos)
    require(eos and all(isinstance(token, int) for token in eos), 'missing generation EOS contract')
    sequences = []
    for depth, requests, trace, execution in ((0, args.target, args.target_trace, args.target_execution),
                                            (3, args.mtp, args.mtp_trace, args.mtp_execution)):
        report = json.loads(requests.read_text())
        receipt = json.loads(execution.read_text())
        require(report['status'] == 'PASS' and report['spec_depth'] == depth and report['engine_freed'], 'wrong/unfinished C-API execution')
        require(receipt['exit_code'] == 0 and receipt['instance_removed'] and receipt['production_stopped'], 'missing actual cleanup receipt')
        require(receipt['request_sha256'] == hashlib.sha256(args.bundle.read_bytes()).hexdigest(), 'request bundle differs from executed one')
        require(len(report['cases']) == 4, 'incomplete C-API wave')
        ids, encodes, graphs = {}, [], []
        for line in trace.read_text().splitlines():
            if line.startswith('NATIVE_VISION_SAMPLED '):
                record = json.loads(line.split(' ', 1)[1])
                ids.setdefault(record['request_id'], []).extend(record['token_ids'])
            elif line.startswith('NATIVE_VISION_ENCODE '):
                encodes.append(line)
            elif line.startswith('NATIVE_VISION_GRAPH '):
                graphs.append(json.loads(line.split(' ', 1)[1]))
        require(set(ids) == {'chatcmpl-' + str(i) for i in range(8)}, 'fresh eight-request trace required')
        require(len(encodes) == 4, 'expected four distinct images encoded once')
        require(graphs and all(g['requests'] == 1 and g['tokens'] == (4 if depth else 1) for g in graphs), 'wrong actual verification shape')
        require(any(g['captured'] and g['replay_count'] > 1 for g in graphs), 'missing progressing graph replay')
        committed, scores, discarded = [], [], 0
        for index, (case, entry) in enumerate(zip(report['cases'], bundle)):
            require(case['name'] == entry['name'] and case['ordinary_stream_equal'] and case['terminal_callbacks'] == 1, 'wrong C-API pair')
            response = case['ordinary_response']
            require(response['id'] == 'chatcmpl-' + str(2 * index), 'wrong ordinary request identity')
            choice = response['choices'][0]
            require(choice['finish_reason'] == 'stop' and case['streamed_usage'] == response['usage'], 'wrong EOS/stream usage')
            score = evaluate(choice['message']['content'], entry)
            require(score['pass'] and case['streamed_text'] == choice['message']['content'], 'incorrect image facts or stream text')
            count = response['usage']['completion_tokens']
            pair = []
            for suffix in (0, 1):
                raw = ids['chatcmpl-' + str(2 * index + suffix)]
                require(raw and all(isinstance(token, int) and 0 <= token < 248320 for token in raw), 'invalid sampler ID')
                boundary = next((n + 1 for n, token in enumerate(raw) if token in eos), None)
                require(boundary == count and 0 <= len(raw) - count <= depth, 'first EOS differs from committed usage boundary')
                pair.append(raw[:boundary])
                discarded += len(raw) - boundary
            require(pair[0] == pair[1], 'ordinary/stream committed IDs differ')
            committed.extend(pair)
            scores.append({'name': case['name'], 'score': score, 'usage': response['usage']})
        require(report['drafts_proposed'] > 0 if depth else report['drafts_proposed'] == 0, 'missing/unexpected actual proposals')
        sequences.append(committed)
        result['modes'].append({'depth': depth, 'scores': scores, 'encoder_invocations': len(encodes),
            'drafts_proposed': report['drafts_proposed'], 'graph_observations': len(graphs),
            'committed_tokens': sum(map(len, committed)), 'sampler_tokens_discarded_after_eos': discarded,
            'binary_sha256': receipt['binary_sha256'], 'library_sha256': receipt['library_sha256']})
    require(sequences[0] == sequences[1], 'target-only/MTP3 committed IDs differ')
    result.update(ordinary_stream_and_target_mtp3_ids_equal=True,
                  exact_committed_tokens_per_mode=sum(map(len, sequences[0])),
                  generation_config_sha256=hashlib.sha256(args.generation_config.read_bytes()).hexdigest(),
                  scope='four direct C1 public C-ABI image task pairs per mode; not full tensor or Python parity')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('bundle', 'generation-config', 'target', 'mtp', 'target-trace', 'mtp-trace',
                 'target-execution', 'mtp-execution', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous evidence')
    result = {'status': 'FAIL', 'modes': []}
    code = 1
    try:
        run(args, result)
        result['status'], code = 'PASS', 0
    except Exception as error:
        result['error'] = repr(error)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({k: result.get(k) for k in ('status', 'exact_committed_tokens_per_mode', 'error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
