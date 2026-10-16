#!/usr/bin/env python3
"""Check real-serving profile spans separately from trace-free speed results."""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import statistics

from capture_native_exl3_vision_max_eviction import require
from native_exl3_vision_facts import evaluate, load_tasks


def check(report_path, execution_path, requests_path, depth, task):
    report = json.loads(report_path.read_text())
    execution = json.loads(execution_path.read_text())
    require(execution['exit_code'] == 0 and execution['instance_removed'] and execution['production_stopped'],
            'missing actual execution/cleanup')
    require(execution['result_sha256'] == hashlib.sha256(report_path.read_bytes()).hexdigest() and
            execution['requests_sha256'] == hashlib.sha256(requests_path.read_bytes()).hexdigest(),
            'wrong supervised report/request bundle')
    require(report['status'] == 'CAPTURED' and report['engine_freed'] and report['spec_depth'] == depth and
            report['fixture'] == task['name'] and report['expected'] == task['expected'] and
            report['prefix_cache'] == 'disabled' and report['prefill_budget'] == 1600,
            'wrong capture recipe')
    bundle = json.loads(requests_path.read_text())
    body = bundle[0]['body']
    require(bundle[0]['name'] == task['name'] and body['messages'][0]['content'][-1]['text'] == task['prompt'],
            'wrong frozen request prompt')
    images = [p for p in body['messages'][0]['content'] if p['type'] == 'image_url']
    require(len(images) == 1, 'single-image profile required')
    payload = images[0]['image_url']['url'].split(',', 1)[1]
    require(hashlib.sha256(base64.b64decode(payload, validate=True)).hexdigest() == task['sha256'], 'wrong frozen request image')
    cases = report['cases']
    require(len(cases) == 4 and [c['repeat'] for c in cases] == list(range(4)), 'cold/three-repeat profile wave required')
    durations = []
    for case in cases:
        response = case['response']
        require(len(response['choices']) == 1 and response['choices'] == cases[0]['response']['choices'] and
                response['usage'] == cases[0]['response']['usage'], 'repeat response differs')
        require(evaluate(response['choices'][0]['message']['content'], task)['pass'], 'profile image facts failed')
        usage = response['usage']
        require(0 < usage['prompt_tokens'] < 1600 and 0 < usage['completion_tokens'] <= body['max_tokens'] and
                usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'wrong single-chunk usage')
        require(response['choices'][0]['finish_reason'] in ('stop', 'length'), 'profile request did not finish')
        target, draft = case['target_spans'], case['draft_spans']
        require(len(target) > 1 and (bool(draft) if depth else not draft), 'missing/unexpected executed spans')
        require(case['target_prefill'] == target[0] and case['initial_mtp_proposal'] == (draft[0] if depth else None),
                'wrong initial phase attribution')
        require(case['collected_events'] >= len(target) + len(draft), 'missing raw event collection')
        for stage, events in [('runner_target_forward', target), ('runner_mtp_draft', draft)]:
            for index, event in enumerate(events):
                require(event['stage'] == stage and isinstance(event['start_ns'], int) and isinstance(event['end_ns'], int)
                        and 0 < event['start_ns'] < event['end_ns'], 'invalid actual device interval')
                duration = (event['end_ns'] - event['start_ns']) * 1e-9
                require(math.isfinite(event['duration_s']) and duration == event['duration_s'], 'duration differs from timestamps')
                if index:
                    require(events[index - 1]['queue_id'] == event['queue_id'] and
                            events[index - 1]['end_ns'] <= event['start_ns'], 'invalid event order/clock')
        first = case['target_prefill']
        proposal = case['initial_mtp_proposal']
        if depth:
            require(proposal['queue_id'] == first['queue_id'] and first['end_ns'] <= proposal['start_ns'],
                    'overlapping target/proposal clocks')
        durations.append({'target_prefill_s': first['duration_s'],
                          'initial_mtp_proposal_s': proposal['duration_s'] if depth else None})
    require(report['drafts_proposed'] > 0 if depth else report['drafts_proposed'] == 0, 'missing/unexpected actual MTP')
    require(0 < report['backend_live_bytes'] <= report['backend_peak_bytes'], 'invalid backend memory accounting')
    return {'spec_depth': depth, 'fixture': task['name'], 'usage': cases[0]['response']['usage'],
            '_choices': cases[0]['response']['choices'],
            'cold': durations[0], 'warm_target_prefill_median_s': statistics.median(c['target_prefill_s'] for c in durations[1:]),
            'warm_initial_mtp_proposal_median_s': statistics.median(c['initial_mtp_proposal_s'] for c in durations[1:]) if depth else None,
            'drafts_proposed': report['drafts_proposed'], 'backend_peak_bytes': report['backend_peak_bytes'],
            'binary_sha256': execution['binary_sha256'], 'head': execution['head']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--target', type=Path, required=True)
    parser.add_argument('--target-execution', type=Path, required=True)
    parser.add_argument('--mtp', type=Path, required=True)
    parser.add_argument('--mtp-execution', type=Path, required=True)
    parser.add_argument('--requests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve evidence')
    result = {'status': 'FAIL', 'scope': 'profiled C1 single-chunk target-prefill and initial MTP3 proposal queue spans; not trace-free speed or summed kernel time'}
    code = 1
    try:
        tasks = load_tasks(Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_qualification')
        task = next(t for t in tasks if t['name'] == 'backup-dialog')
        target = check(args.target, args.target_execution, args.requests, 0, task)
        mtp = check(args.mtp, args.mtp_execution, args.requests, 3, task)
        require(target['binary_sha256'] == mtp['binary_sha256'] and target['head'] == mtp['head'], 'mixed native binary/source identities')
        require(target.pop('_choices') == mtp.pop('_choices') and target['usage'] == mtp['usage'],
                'profiled target/MTP response or usage differs')
        result.update(status='PASS', cohorts=[target, mtp],
            profiled_target_mtp_response_equal=True,
            proposal_scope='first proposal includes shifted draft prefill, sampling and two extra draft decode forwards; not isolated draft-prefill kernels',
            clock_scope='same in-order queue device-clock spans can include host submission gaps; nested kernel/library spans are not summed')
        code = 0
    except Exception as error:
        result['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
