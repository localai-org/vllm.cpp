#!/usr/bin/env python3
"""Check the bounded resize screenshot reference and paired real-serving traces.

This certifies named facts, encoder row count, committed native IDs and cleanup.
It does not certify full-tower numeric parity or Python/native token equality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re

from capture_native_exl3_vision_reference import EXPECTED, REFERENCE_IMAGE
from native_exl3_vision_facts import evaluate, qualification_inputs


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, result):
    tasks, manifest_name, compact_name = qualification_inputs(args.fixtures, True)
    task = tasks[0]
    compact = json.loads((args.fixtures / compact_name).read_text())
    reference = json.loads(args.reference.read_text())
    execution = json.loads(args.reference_execution.read_text())
    require(execution['status'] == 'PASS' and execution['exit_code'] == 0 and
            execution['server_removed'] and execution['production_stopped'], 'reference cleanup missing')
    require(execution['actual_image'] == reference['reference_image'] == REFERENCE_IMAGE and
            reference['runtime'] == EXPECTED, 'actual reference runtime differs')
    require(execution['result_sha256'] == compact['actual_capture_sha256'] == sha(args.reference),
            'executed reference hash differs')
    require(reference['status'] == compact['status'] == 'PASS' and
            reference['manifest_sha256'] == compact['manifest_sha256'] == sha(args.fixtures / manifest_name),
            'reference manifest differs')
    require(len(reference['cases']) == len(compact['cases']) == 1, 'incomplete screenshot reference')
    case = reference['cases'][0]
    require(case['image_sha256'] == task['sha256'] and case['name'] == task['name'] and
            case['finish_reason'] == 'stop' and evaluate(case['reference_text'], task)['pass'] and
            all(case[k] == compact['cases'][0][k] for k in compact['cases'][0]), 'reference named facts differ')
    actual = reference['actual_tower_input']
    require(actual == compact['actual_tower_input'] and actual['grid_thw'] == task['grid_thw'] and
            actual['pixels_shape'] == [2160, 1536] and actual['pixels_dtype'] == 'torch.float16' and
            actual['device'].startswith('xpu') and reference['processed_size_wh'] == task['processed_size_wh'],
            'actual resize input differs')
    eos = json.loads(args.generation_config.read_text())['eos_token_id']
    eos = {eos} if isinstance(eos, int) else set(eos)
    require(eos and all(type(token) is int for token in eos), 'missing EOS contract')
    paired_ids, paired_answers, binaries, heads = [], [], [], []
    for depth, report_path, trace_path, receipt_path in (
            (0, args.target, args.target_trace, args.target_execution),
            (3, args.mtp, args.mtp_trace, args.mtp_execution)):
        report, receipt = json.loads(report_path.read_text()), json.loads(receipt_path.read_text())
        require(receipt['status'] == 'PASS' and receipt['exit_code'] == receipt['client_exit_code'] == 0 and
                receipt['server_removed'] and receipt['production_stopped'], 'native cleanup missing')
        require(receipt['result_sha256'] == sha(report_path) and receipt['trace_sha256'] == sha(trace_path),
                'executed native report/trace hash differs')
        require(receipt['sources']['tests/fixtures/native_vision_qualification/resize-screenshot.json'] ==
                sha(args.fixtures / manifest_name) and
                receipt['sources']['tests/fixtures/native_vision_qualification/resize-screenshot-reference.json'] ==
                sha(args.fixtures / compact_name), 'executed native fixtures differ')
        require(report['status'] == 'PASS' and report['declared_spec_depth'] == receipt['declared_spec_depth'] == depth and
                report['manifest_sha256'] == sha(args.fixtures / manifest_name) and
                report['reference_image'] == REFERENCE_IMAGE and len(report['cases']) == 2, 'wrong native cohort')
        sampled, encodes, graphs = {}, [], []
        with trace_path.open() as trace:
            for line in trace:
                if line.startswith('NATIVE_VISION_SAMPLED '):
                    item = json.loads(line.split(' ', 1)[1])
                    sampled.setdefault(item['request_id'], []).extend(item['token_ids'])
                elif line.startswith('NATIVE_VISION_ENCODE '):
                    encodes.append(line.strip())
                elif line.startswith('NATIVE_VISION_GRAPH '):
                    graphs.append(json.loads(line.split(' ', 1)[1]))
        require(len(encodes) == 1 and re.fullmatch(r'NATIVE_VISION_ENCODE count=1 hash=[0-9a-f]{64} rows=540', encodes[0]),
                'resize encoder rows/reuse differ')
        require(set(sampled) == {'chatcmpl-0', 'chatcmpl-1'}, 'incomplete native request trace')
        require(graphs and all(g['requests'] == 1 and g['tokens'] == (4 if depth else 1) for g in graphs) and
                any(g['captured'] and g['replay_count'] > 1 for g in graphs), 'actual decode graph differs')
        committed, answers = [], []
        for index, item in enumerate(report['cases']):
            require(item['name'] == task['name'] and item['image_sha256'] == task['sha256'] and
                    item['stream'] is bool(index) and item['request_id'] == 'chatcmpl-' + str(index) and
                    item['finish_reason'] == 'stop' and evaluate(item['text'], task)['pass'], 'native image facts differ')
            usage = item['usage']
            require(all(type(usage.get(k)) is int for k in ('prompt_tokens', 'completion_tokens', 'total_tokens')) and
                    usage['prompt_tokens'] == 597 and 0 < usage['completion_tokens'] <= 128 and
                    usage['total_tokens'] == 597 + usage['completion_tokens'], 'native screenshot usage differs')
            require(all(item['metrics'][k] == 0 for k in ('vllm:num_requests_running', 'vllm:num_requests_waiting')),
                    'native request did not drain')
            raw = sampled[item['request_id']]
            require(raw and all(type(token) is int and 0 <= token < 248320 for token in raw), 'invalid sampled ID')
            boundary = next((i + 1 for i, token in enumerate(raw) if token in eos), None)
            require(boundary == usage['completion_tokens'] and 0 <= len(raw) - boundary <= depth,
                    'sampled first EOS/committed boundary differs')
            committed.append(raw[:boundary])
            answers.append({'text': item['text'], 'usage': usage})
        require(committed[0] == committed[1] and answers[0] == answers[1], 'ordinary/SSE differs')
        require(report['draft_tokens'] > 0 if depth else report['draft_tokens'] == 0, 'actual draft activity differs')
        paired_ids.append(committed)
        paired_answers.append(answers)
        binaries.append(receipt['binary_sha256'])
        heads.append(receipt['head'])
        result['modes'].append({'depth': depth, 'usage': answers[0]['usage'], 'encoder_invocations': 1,
                                'draft_tokens': report['draft_tokens'], 'facts': evaluate(answers[0]['text'], task)})
    require(paired_ids[0] == paired_ids[1] and paired_answers[0] == paired_answers[1], 'target-only/MTP3 differs')
    require(binaries[0] == binaries[1] and heads[0] == heads[1], 'native paired build differs')
    result.update(input_size_wh=task['size_wh'], processed_size_wh=task['processed_size_wh'],
                  reference_tower_input=actual, exact_committed_tokens_per_mode=sum(map(len, paired_ids[0])),
                  binary_sha256=binaries[0], head=heads[0], held_out=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_qualification')
    for name in ('reference', 'reference-execution', 'target', 'target-trace', 'target-execution',
                 'mtp', 'mtp-trace', 'mtp-execution', 'generation-config', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    result = {'status': 'FAIL', 'scope': __doc__.split('\n\n', 1)[1].strip(), 'modes': []}
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
