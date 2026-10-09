#!/usr/bin/env python3
"""Abort partial-image prefill, then submit three retries (six in warm-owner mode).

Use an unused C1 target-only server with trace2, prefill progress and dev mode.
This explicitly calls /abort_requests, not transport-disconnect cancellation.
No service management or learned Python fallback.
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
import time
import urllib.error
import urllib.request
from native_vision_cache_trace import cache_record

from capture_native_exl3_vision_max_eviction import require
from native_vision_warm_owners import check_warm_owners


class Trace:
    def __init__(self, path):
        self.file = path.open('rb')
        self.idle = []
        self.encodes = []
        self.first_frame = None
        self.requests = []
        self.prefill_done = False
        self.witness = []

    def read(self):
        while True:
            position = self.file.tell()
            line = self.file.readline()
            if not line.endswith(b'\n'):
                self.file.seek(position)
                break
            text = line.decode()
            match = re.search(r'INFO Received request (chatcmpl-\d+) endpoint=', text)
            if match:
                self.requests.append(match.group(1))
            if text.startswith('NATIVE_VISION_ENCODE '):
                self.encodes.append(text.strip())
                self.witness.append(text.strip())
            elif text.startswith('NATIVE_VISION_EMBED ') and self.first_frame is None:
                record = json.loads(text.split(' ', 1)[1])
                self.first_frame = {k: record[k] for k in ('tokens', 'source_slices')}
                self.witness.append('NATIVE_VISION_EMBED ' + json.dumps(self.first_frame))
            elif (record:=cache_record(text)) is not None:
                if record['event'] == 'idle':
                    self.idle.append(record)
            if re.search(r'INFO prefill id=chatcmpl-0 .*status=done', text):
                self.prefill_done = True
                self.witness.append(text.strip())


def run(args, report):
    url = args.url.rstrip('/')
    fixture = next(f for f in json.loads((args.fixtures / 'fixtures.json').read_text()) if f['name'] == 'orbit')
    image = (args.fixtures / fixture['file']).read_bytes()
    require(hashlib.sha256(image).hexdigest() == fixture['sha256'], 'frozen fixture changed')
    reference = next(c for c in json.loads((args.fixtures / 'chunk-mrope-reference.json').read_text())['cases'] if c['offset'] == 1599)
    prompt = ('Read the large heading and number exactly. Then describe the color and shape on the left, '
              'followed by the color and shape on the right. Explain what is visible in a few complete English sentences.')

    def api(path, body=None, raw=False, timeout=30):
        request = urllib.request.Request(url + path, data=None if body is None else json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                return response.read().decode() if raw else json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError('HTTP ' + str(error.code) + ': ' + error.read().decode()) from error

    def gauges():
        values = {}
        for line in api('/metrics', raw=True, timeout=5).splitlines():
            if not line.strip() or line.startswith('#'):
                continue
            key = line.split('{', 1)[0].split(' ', 1)[0]
            if key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
                value = float(line.rsplit(' ', 1)[-1])
                require(math.isfinite(value) and value >= 0, 'invalid drain gauge')
                values[key] = values.get(key, 0) + value
        require(len(values) == 2, 'missing drain gauges')
        return values

    require(not any(gauges().values()), 'unused server required')
    count = 1595
    for _ in range(8):
        prefix = ' x' * count
        tokens = api('/tokenize', {'model': args.model, 'messages': [{'role': 'user',
            'content': prefix + '<|vision_start|><|image_pad|><|vision_end|>' + prompt}],
            'add_special_tokens': True, 'chat_template_kwargs': {'enable_thinking': False}})['tokens']
        offset = tokens.index(248056)
        if offset == 1599:
            break
        count += 1599 - offset
        require(count >= 0, 'invalid padding')
    else:
        raise RuntimeError('cannot reach partial-image boundary')
    expanded = tokens[:1599] + [248056] * 192 + tokens[1600:]
    require(hashlib.sha256(struct.pack('<' + str(len(expanded)) + 'i', *expanded)).hexdigest() ==
            reference['tokens_sha256'], 'frozen partial-image prompt differs')
    body = {'model': args.model, 'messages': [{'role': 'user', 'content': [
        {'type': 'text', 'text': prefix}, {'type': 'image_url', 'image_url': {'url':
         'data:image/png;base64,' + base64.b64encode(image).decode()}},
        {'type': 'text', 'text': prompt}]}], 'temperature': 0., 'max_tokens': 64,
        'stream': True, 'stream_options': {'include_usage': True},
        'chat_template_kwargs': {'enable_thinking': False}}
    report['request_sha256'] = hashlib.sha256(json.dumps(body).encode()).hexdigest()
    trace = Trace(args.trace_log)
    try:
        trace.read()
        require(not trace.requests and not trace.encodes and not trace.idle, 'fresh server trace required')

        def wait_idle(count):
            deadline = time.monotonic() + 15
            while True:
                trace.read()
                observed = gauges()
                require(len(trace.idle) <= count, 'unexpected concurrent release')
                if len(trace.idle) == count and not any(observed.values()):
                    return observed
                require(time.monotonic() < deadline, 'missing drained device-owner release')
                time.sleep(.02)  # Test-client observation; no product polling is added.

        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            response = pool.submit(api, '/v1/chat/completions', body, True)
            deadline = time.monotonic() + 15
            while True:
                trace.read()
                require(not response.done() and not trace.prefill_done, 'prefill finished before scoped abort')
                if trace.encodes and trace.first_frame:
                    require(trace.requests == ['chatcmpl-0'] and len(trace.encodes) == 1 and
                            trace.first_frame == {'tokens': 1600, 'source_slices': [[0, 1]]},
                            'wrong live partial-image request')
                    break
                require(time.monotonic() < deadline, 'missing encode/partial-prefill witness')
                time.sleep(.01)
            report['pre_abort_log_tail'] = '\n'.join(trace.witness)
            report['abort'] = api('/abort_requests', {'request_ids': ['chatcmpl-0']}, timeout=10)
            require(report['abort'] == {'status': 'aborted', 'aborted': 1}, 'abort missed live request')
            raw = response.result(timeout=30)
            report['cancelled_sse'] = raw
            require(raw.count('data: [DONE]') == 1, 'missing/duplicate cancelled SSE completion')
            frames = [json.loads(line[6:]) for line in raw.splitlines()
                      if line.startswith('data: ') and line[6:] != '[DONE]']
            require(frames and all(f['id'] == 'chatcmpl-0' and f['model'] == args.model for f in frames), 'wrong cancelled SSE identity')
            usages = [f['usage'] for f in frames if f.get('usage')]
            require(len(usages) == 1, 'missing/duplicate cancelled usage')
            report['cancelled_usage'] = usages[0]
            require(usages[0] == {'prompt_tokens': 1840, 'completion_tokens': 0, 'total_tokens': 1840}, 'cancel happened after decode')
            choices = [c for f in frames for c in f['choices']]
            require(all(not c['delta'].get('content') for c in choices) and
                    [c['finish_reason'] for c in choices if c.get('finish_reason')] == ['abort'], 'wrong cancelled output/finish')
        report['post_abort_gauges'] = wait_idle(1)
        body['stream'] = False
        del body['stream_options']
        retry_count = 6 if args.memory_contract == 'warm-owner' else 3
        for index in range(retry_count):
            response = api('/v1/chat/completions', body, timeout=120)
            require(response['id'] == 'chatcmpl-' + str(index + 1) and response['model'] == args.model, 'wrong retry owner/model')
            require(len(response['choices']) == 1 and response['choices'][0]['finish_reason'] == 'length' and
                    response['usage'] == {'prompt_tokens': 1840, 'completion_tokens': 64, 'total_tokens': 1904}, 'wrong retry usage/finish')
            text = response['choices'][0]['message']['content']
            lower = text.lower()
            require(all(word in lower for word in fixture['expected']), 'incorrect retry image facts')
            for side, words in fixture['sides'].items():
                start = lower.find(side)
                require(start >= 0 and all(word in lower[start:start + 100] for word in words), 'incorrect retry image side')
            observed = wait_idle(index + 2)
            report['followups'].append({'response': response, 'text': text, 'gauges': observed})
            require(text == report['followups'][0]['text'], 'retry answer changed')
            print(json.dumps({'retry': index + 1, 'usage': response['usage']}), flush=True)
        trace.read()
        report.update(encoder_submissions=trace.encodes, idle_records=trace.idle)
        require(len(trace.encodes) == 1 and len(trace.idle) == retry_count + 1, 'missing cache reuse/release')
        require(all(e['entries'] == 1 and e['embedding_bytes'] == 1966080 and
                    0 < e['backend_allocated_bytes'] <= e['backend_peak_allocated_bytes'] for e in trace.idle), 'invalid cache/allocator telemetry')
        retries = trace.idle[1:]
        if args.memory_contract == 'resident-exact':
            require(all(e['backend_allocated_bytes'] == retries[-1]['backend_allocated_bytes'] for e in retries), 'retry memory grows')
        else:
            # Graph capture retains scratch. The first later prefill can grow
            # its free pool. Attribute every byte, require equal active owners,
            # and still demand an exact final total/free-pool plateau.
            for row in retries:
                require(isinstance(row['scratch_pool_retained_bytes'], int) and
                        0 <= row['scratch_pool_retained_bytes'] <= row['backend_allocated_bytes'], 'invalid retained pool accounting')
            active = [e['backend_allocated_bytes'] - e['scratch_pool_retained_bytes'] for e in retries]
            require(len(set(active)) == 1 and len({e['scratch_pool_live_blocks'] for e in retries}) == 1,
                    'active owners grow after retry')
            require(retries[0]['backend_graph_count'] > 0 and
                    len({e['backend_graph_count'] for e in retries}) == 1 and
                    len({e['backend_graph_device_bytes'] for e in retries}) == 1, 'graph residency grows after retry')
            for key in ('backend_allocated_bytes', 'scratch_pool_retained_bytes', 'scratch_pool_misses'):
                require(retries[-2][key] == retries[-1][key], 'warm retry total/pool still grows')
            report['active_retry_backend_bytes'] = active[-1]
            if args.memory_contract == 'warm-owner':
                report['warm_owner_observations'] = check_warm_owners(retries)
    finally:
        trace.file.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--trace-log', type=Path, required=True)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_http')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--memory-contract', choices=['resident-exact', 'graph-pool-owned', 'warm-owner'], default='resident-exact',
                        help='legacy total gate stays default; graph mode checks active/pool plateau; warm-owner adds three fixed-shape repetitions')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve prior evidence')
    report = {'status': 'FAIL', 'followups': [],
              'memory_contract': args.memory_contract,
              'scope': 'C1 target-only dev abort after first visual-row submission; not transport/C4/kernel-interrupt parity',
              'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    code = 1
    try:
        run(args, report)
        report['status'], code = 'PASS', 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'successful_retries': len(report['followups']), 'error': report.get('error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
