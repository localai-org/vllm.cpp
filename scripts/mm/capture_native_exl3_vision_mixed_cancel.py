#!/usr/bin/env python3
"""Abort cold two-image prefill and a live C4/MTP3 decode, then reuse slots.

Use a fresh server, page/prefill budget 1600, two images, four effective slots,
MTP3, prefix caching off, trace2, prefill progress and explicit dev-mode abort.
Only test-client observation polls; no services or learned Python inference.
"""
import argparse
import base64
import concurrent.futures
import hashlib
import json
import math
from pathlib import Path
import re
import struct
import threading
import time
import urllib.error
import urllib.request
from native_vision_cache_trace import cache_record

from capture_native_exl3_vision_max_eviction import require


class Trace:
    def __init__(self, path):
        self.file = path.open('rb')
        self.encodes, self.requests, self.idle, self.graphs = [], [], [], []
        self.first_frame = None
        self.prefill_done = False

    def read(self):
        while True:
            position = self.file.tell()
            line = self.file.readline()
            if not line.endswith(b'\n'):
                self.file.seek(position)
                return
            text = line.decode()
            match = re.search(r'INFO Received request (chatcmpl-\d+) endpoint=', text)
            if match:
                self.requests.append(match.group(1))
            if text.startswith('NATIVE_VISION_ENCODE '):
                self.encodes.append(text.strip())
            elif text.startswith('NATIVE_VISION_EMBED ') and self.first_frame is None:
                row = json.loads(text.split(' ', 1)[1])
                self.first_frame = {k: row[k] for k in ('tokens', 'source_slices')}
            elif (row:=cache_record(text)) is not None:
                if row['event'] == 'idle':
                    self.idle.append(row)
            elif text.startswith('NATIVE_VISION_GRAPH '):
                self.graphs.append(json.loads(text.split(' ', 1)[1]))
            if re.search(r'INFO prefill id=chatcmpl-0 .*status=done', text):
                self.prefill_done = True


def run(args, report):
    url = args.url.rstrip('/')

    def api(path, body=None, raw=False, timeout=120):
        request = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode() if raw else json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError('HTTP ' + str(error.code) + ': ' + error.read().decode()) from error

    def metrics():
        values = {}
        wanted = ('vllm:num_requests_running', 'vllm:num_requests_waiting',
                  'vllm:spec_decode_num_draft_tokens_total')
        for line in api('/metrics', raw=True, timeout=5).splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            key = line.split('{', 1)[0].split(' ', 1)[0]
            if key in wanted:
                value = float(line.rsplit(' ', 1)[-1])
                require(math.isfinite(value) and value >= 0, 'invalid lifecycle metric')
                values[key] = values.get(key, 0) + value
        require(all(k in values for k in wanted), 'missing drain/speculation metrics')
        return values

    def gauges():
        return {k: v for k, v in metrics().items() if k != 'vllm:spec_decode_num_draft_tokens_total'}

    before = metrics()
    require(not any(gauges().values()), 'unused server required')
    fixtures = json.loads((args.fixtures / 'fixtures.json').read_text())
    images = []
    report['fixture_sha256'] = {}
    for name in ('orbit', 'comet'):
        fixture = next(f for f in fixtures if f['name'] == name)
        image = (args.fixtures / fixture['file']).read_bytes()
        digest = hashlib.sha256(image).hexdigest()
        require(digest == fixture['sha256'] and fixture['size'] == [512, 384], 'frozen image changed')
        report['fixture_sha256'][name] = digest
        images.append({'type': 'image_url', 'image_url': {'url':
            'data:image/png;base64,' + base64.b64encode(image).decode()}})
    question = "Read the large heading and number in each picture. List both in conversation order as 'First: ...' and 'Second: ...'."
    prefix = ' x' * 1595
    tokenized = api('/tokenize', {'model': args.model, 'messages': [{'role': 'user',
        'content': prefix + '<|vision_start|><|image_pad|><|vision_end|>' * 2 + question}],
        'add_special_tokens': True, 'chat_template_kwargs': {'enable_thinking': False}})['tokens']
    offsets = [i for i, t in enumerate(tokenized) if t == 248056]
    require(len(offsets) == 2 and offsets[0] == 1599, 'wrong first-image boundary')
    expanded = [out for token in tokenized for out in ([token] * 192 if token == 248056 else [token])]
    require(len(expanded) == 2022, 'wrong frozen two-image prompt length')
    report['expanded_prompt_sha256'] = hashlib.sha256(struct.pack('<' + str(len(expanded)) + 'i', *expanded)).hexdigest()

    def pair_body(padding=False, streaming=False):
        parts = ([{'type': 'text', 'text': prefix}] if padding else []) + images + [{'type': 'text', 'text': question}]
        body = {'model': args.model, 'messages': [{'role': 'user', 'content': parts}], 'temperature': 0.,
                'max_tokens': 64, 'chat_template_kwargs': {'enable_thinking': False}, 'stream': streaming}
        if streaming:
            body['stream_options'] = {'include_usage': True}
        return body

    def stream(body, state):
        request = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        lines = []
        with urllib.request.urlopen(request, timeout=120) as response:
            for raw in response:
                line = raw.decode()
                lines.append(line)
                if line.startswith('data: ') and line[6:].strip() != '[DONE]':
                    item = json.loads(line[6:])
                    require(item['model'] == args.model, 'wrong streamed model')
                    require('id' not in state or state['id'] == item['id'], 'stream request identity changed')
                    state['id'] = item['id']
                    if any(c['delta'].get('content') for c in item.get('choices', [])):
                        state['content_started'] = True
        return ''.join(lines)

    def cancelled(raw, decoded, owner):
        require(raw.count('data: [DONE]') == 1, 'missing/duplicate SSE done')
        frames = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line[6:] != '[DONE]']
        require(frames and all(f['id'] == owner and f['model'] == args.model for f in frames), 'wrong cancelled SSE identity')
        choices = [c for f in frames for c in f.get('choices', [])]
        usages = [f['usage'] for f in frames if f.get('usage')]
        require(len(usages) == 1, 'missing/duplicate abort usage')
        usage = usages[0]
        require([c['finish_reason'] for c in choices if c.get('finish_reason')] == ['abort'], 'wrong abort finish')
        text = ''.join(c['delta'].get('content', '') or '' for c in choices)
        require(bool(text) == decoded, 'unexpected abort content')
        require((0 < usage['completion_tokens'] < 64) if decoded else usage['completion_tokens'] == 0, 'wrong abort output count')
        require(usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens'], 'wrong abort usage sum')
        return {'raw_sse': raw, 'text': text, 'usage': usage, 'id': owner}

    trace = Trace(args.trace_log)
    try:
        trace.read()
        require(not trace.requests and not trace.encodes and not trace.idle, 'fresh server trace required')

        def released(previous):
            deadline = time.monotonic() + 30
            while True:
                trace.read()
                observed = gauges()
                if len(trace.idle) > previous and not any(observed.values()):
                    return observed, trace.idle[-1]
                require(time.monotonic() < deadline, 'missing drained device-owner release')
                time.sleep(.02)  # Client-only diagnostic observation, outside performance measurement.

        def pair_check(response, previous):
            require(response['model'] == args.model and len(response['choices']) == 1, 'wrong retry identity/choices')
            text = response['choices'][0]['message']['content']
            require(re.findall(r'\b(orbit|comet)\s*[:#-]?\s*(731|924)', text.lower()) ==
                    [('orbit', '731'), ('comet', '924')], 'incorrect ordered image facts')
            usage = response['usage']
            require(0 < usage['completion_tokens'] <= 64 and usage['total_tokens'] ==
                    usage['prompt_tokens'] + usage['completion_tokens'], 'wrong retry usage')
            observed, _ = released(previous)
            return {'response': response, 'text': text, 'gauges': observed}

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(stream, pair_body(True, True), {})
            deadline = time.monotonic() + 18
            while True:
                trace.read()
                require(not future.done() and not trace.prefill_done, 'cold prefill finished before abort')
                if trace.encodes and trace.first_frame:
                    require(trace.requests == ['chatcmpl-0'] and len(trace.encodes) == 1 and
                            trace.first_frame == {'tokens': 1600, 'source_slices': [[0, 1]]}, 'wrong cold partial-image submission')
                    break
                require(time.monotonic() < deadline, 'cold encoder/submission deadline')
                time.sleep(.01)
            report['cold_abort'] = api('/abort_requests', {'request_ids': ['chatcmpl-0']}, timeout=10)
            require(report['cold_abort'] == {'status': 'aborted', 'aborted': 1}, 'cold abort missed live request')
            report['cold_cancel'] = cancelled(future.result(timeout=30), False, 'chatcmpl-0')
            require(report['cold_cancel']['usage']['prompt_tokens'] == 2022, 'wrong actual cold prompt usage')
        report['cold_post_abort_gauges'], report['cold_idle_release'] = released(0)
        report['cold_retry'] = pair_check(api('/v1/chat/completions', pair_body(True)), len(trace.idle))
        require(report['cold_retry']['response']['usage']['prompt_tokens'] == 2022, 'cold retry prompt changed')

        question_one = ('Read the large heading and number exactly. Then describe the color and shape on the left, '
                        'followed by the color and shape on the right. Explain what is visible in a few complete English sentences.')
        companion = {'model': args.model, 'messages': [{'role': 'user', 'content': [images[1],
            {'type': 'text', 'text': question_one}]}], 'temperature': 0., 'max_tokens': 64,
            'chat_template_kwargs': {'enable_thinking': False}}
        texts = [{'model': args.model, 'prompt': 'Give a factual explanation of ' + op + '.', 'temperature': 0.,
                  'max_tokens': 96, 'ignore_eos': True, 'logprobs': 0} for op in ('addition', 'subtraction')]
        idle_before, graphs_before = len(trace.idle), len(trace.graphs)
        state, barrier = {}, threading.Barrier(4)

        def go_stream():
            barrier.wait(timeout=10)
            return stream(pair_body(False, True), state)

        def go_api(path, body):
            barrier.wait(timeout=10)
            return api(path, body)

        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            victim = pool.submit(go_stream)
            others = [pool.submit(go_api, '/v1/chat/completions', companion)] + [
                pool.submit(go_api, '/v1/completions', body) for body in texts]
            deadline = time.monotonic() + 18
            while True:
                trace.read()
                graphs = trace.graphs[graphs_before:]
                if state.get('content_started') and any(g['requests'] == 4 and g['captured'] for g in graphs):
                    break
                require(not victim.done(), 'decode victim ended before C4 witness')
                require(time.monotonic() < deadline, 'C4 decode witness deadline')
                time.sleep(.01)
            report['decode_pre_abort_graphs'] = graphs
            report['decode_pre_abort_gauges'] = gauges()
            require(report['decode_pre_abort_gauges'] == {'vllm:num_requests_running': 4.,
                    'vllm:num_requests_waiting': 0.}, 'four siblings no longer active')
            report['decode_abort'] = api('/abort_requests', {'request_ids': [state['id']]}, timeout=10)
            require(report['decode_abort'] == {'status': 'aborted', 'aborted': 1}, 'decode abort missed live request')
            report['decode_cancel'] = cancelled(victim.result(timeout=30), True, state['id'])
            responses = [f.result(timeout=30) for f in others]
            report['unaffected_siblings'] = responses
            require(len({r['id'] for r in responses} | {state['id']}) == 4, 'mixed request owners not distinct')
            for i, response in enumerate(responses):
                usage = response['usage']
                require(response['model'] == args.model and len(response['choices']) == 1 and
                        response['choices'][0]['finish_reason'] == 'length', 'wrong sibling identity/finish')
                require(usage['completion_tokens'] == (64 if i == 0 else 96) and usage['total_tokens'] ==
                        usage['prompt_tokens'] + usage['completion_tokens'], 'sibling stopped or lost usage')
            lower = responses[0]['choices'][0]['message']['content'].lower()
            require(all(word in lower for word in ('comet', '924')), 'incorrect unaffected image heading')
            for side, words in {'left': ['green', 'square'], 'right': ['yellow', 'circle']}.items():
                start = lower.find(side)
                require(start >= 0 and all(w in lower[start:start + 100] for w in words), 'incorrect unaffected image spatial facts')
        report['mixed_post_abort_gauges'], report['mixed_idle_release'] = released(idle_before)
        for index in range(3):
            retry = pair_check(api('/v1/chat/completions', pair_body()), len(trace.idle))
            report['followups'].append(retry)
            require(retry['response']['choices'] == report['followups'][0]['response']['choices'] and
                    retry['response']['usage'] == report['followups'][0]['response']['usage'], 'ordered retry changed')
            print(json.dumps({'retry': index + 1, 'usage': retry['response']['usage']}), flush=True)
        trace.read()
        report.update(encodes=len(trace.encodes), idle=trace.idle,
            actual_draft_tokens=metrics()['vllm:spec_decode_num_draft_tokens_total'] - before['vllm:spec_decode_num_draft_tokens_total'])
        require(report['encodes'] == 2 and len(trace.idle) >= 6, 'missing cache reuse or idle release')
        require(all(e['entries'] == 2 and e['embedding_bytes'] == 3932160 for e in trace.idle[-4:]), 'unbounded image owners')
        require(all(0 < e['backend_allocated_bytes'] <= e['backend_peak_allocated_bytes'] for e in trace.idle), 'invalid memory telemetry')
        require(trace.idle[-1]['backend_allocated_bytes'] == trace.idle[-2]['backend_allocated_bytes'], 'warm retry memory still grows')
        require(report['actual_draft_tokens'] > 0, 'actual MTP3 missing')
    finally:
        trace.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--trace-log', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_http')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve evidence')
    report = {'status': 'FAIL', 'followups': [], 'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'scope': 'two-image cold partial-prefill and actual C4/MTP3 decode dev abort; not transport/kernel abort or numerical/RNG parity'}
    code = 1
    try:
        run(args, report)
        report['status'], code = 'PASS', 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'ordered_retries': len(report['followups']), 'error': report.get('error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
