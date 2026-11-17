#!/usr/bin/env python3
"""Submit the frozen C1 MTP3 max/small eviction wave to an existing server.

No service management, generated images or Python inference. A fresh local
level-2 server trace supplies device-idle witnesses outside request timing.
Cleanup/binary identity belongs to a separate supervisor execution receipt.
"""
import argparse
import base64
import hashlib
import json
import math
from pathlib import Path
import struct
import time
import urllib.error
import urllib.request
from native_vision_warm_owners import check_warm_sequence
from native_vision_cache_trace import cache_record
from native_vision_completion import completion_length


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


class CacheTrace:
    def __init__(self, path):
        self.file = path.open('rb')
        self.records = []

    def read(self):
        while True:
            position = self.file.tell()
            line = self.file.readline()
            if not line.endswith(b'\n'):
                self.file.seek(position)
                break
            record = cache_record(line)
            if record is not None:
                self.records.append(record)
        return [r for r in self.records if r['event'] == 'idle']

    def close(self):
        self.file.close()


def run(args, report):
    url = args.url.rstrip('/')
    reference = json.loads((args.fixtures / 'max-eviction-mrope-reference.json').read_text())
    refs = {r['file']: r for r in reference['cases']}
    fixtures = {r['name']: r for r in json.loads((args.fixtures / 'fixtures.json').read_text())}
    images = {}
    for name in ('orbit-max-2048.png', 'comet.png'):
        data = (args.fixtures / name).read_bytes()
        require(hashlib.sha256(data).hexdigest() == refs[name]['image_sha256'], 'frozen image changed')
        images[name] = 'data:image/png;base64,' + base64.b64encode(data).decode()

    def api(path, body):
        request = urllib.request.Request(url + path, data=json.dumps(body).encode(),
            headers={'Content-Type': 'application/json', 'Connection': 'close'})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except urllib.error.HTTPError as error:
            raise RuntimeError('HTTP ' + str(error.code) + ': ' + error.read().decode()) from error

    def gauges():
        with urllib.request.urlopen(url + '/metrics', timeout=5) as response:
            lines = response.read().decode().splitlines()
        values = {}
        for line in lines:
            if not line.strip() or line.startswith('#'):
                continue
            key = line.split('{', 1)[0].split(' ', 1)[0]
            if key in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
                value = float(line.rsplit(' ', 1)[-1])
                require(math.isfinite(value) and value >= 0, 'invalid drain gauge')
                values[key] = values.get(key, 0) + value
        require(len(values) == 2 and not any(values.values()), 'missing/nonzero drain gauges')
        return values

    prompt = ('Read the large heading and number exactly. Then describe the color and shape on the left, '
              'followed by the color and shape on the right. Explain what is visible in a few complete English sentences.')
    gauges()
    trace = CacheTrace(args.trace_log)
    try:
        require(not trace.read() and not trace.records, 'trace must belong to a fresh unused server')
        tokens = api('/tokenize', {'model': args.model, 'messages': [{'role': 'user',
            'content': '<|vision_start|><|image_pad|><|vision_end|>' + prompt}],
            'add_special_tokens': True, 'chat_template_kwargs': {'enable_thinking': False}})['tokens']
        offset = tokens.index(248056)
        require(offset == 4, 'unexpected visual offset')
        prior = {}
        wave_count = 5 if args.warm_owner else 2
        names = (['orbit-max-2048.png'] * 2 + ['comet.png'] * 3) * wave_count
        for index, name in enumerate(names):
            ref = refs[name]
            rows = ref['rows']
            expanded = tokens[:offset] + [248056] * rows + tokens[offset + 1:]
            digest = hashlib.sha256(struct.pack('<' + str(len(expanded)) + 'i', *expanded)).hexdigest()
            require(digest == ref['tokens_sha256'], 'expanded prompt differs from frozen reference')
            body = {'model': args.model, 'messages': [{'role': 'user', 'content': [
                {'type': 'image_url', 'image_url': {'url': images[name]}},
                {'type': 'text', 'text': prompt}]}], 'temperature': 0., 'max_tokens': 64,
                'chat_template_kwargs': {'enable_thinking': False}}
            case = {'file': name, 'sha256': ref['image_sha256'], 'visual_rows': rows,
                    'offset': offset, 'expanded_tokens': expanded,
                    'request_sha256': hashlib.sha256(json.dumps(body).encode()).hexdigest()}
            report['cases'].append(case)
            begin = time.monotonic()
            response = api('/v1/chat/completions', body)
            case.update(response=response, wall_s=time.monotonic() - begin)
            require(response['id'] == 'chatcmpl-' + str(index) and response['model'] == args.model,
                    'fresh sequential response identity required')
            require(len(response['choices']) == 1 and response['choices'][0]['index'] == 0,
                    'wrong choice count/index')
            completion_length(response, ref['prompt_tokens'])
            text = response['choices'][0]['message']['content']
            lower = text.lower()
            facts = fixtures['orbit' if rows == 4096 else 'comet']
            require(all(word in lower for word in facts['expected']), 'incorrect image facts')
            for side, words in facts['sides'].items():
                start = lower.find(side)
                require(start >= 0 and all(word in lower[start:start + 100] for word in words), 'incorrect image side')
            require(name not in prior or prior[name] == text, 'answer changed across cache reuse/eviction')
            prior[name] = text
            case['gauges'] = gauges()
            deadline = time.monotonic() + 15
            while True:
                idle = trace.read()
                require(len(idle) <= index + 1, 'unexpected concurrent device-idle witness')
                if len(idle) == index + 1:
                    case['device_idle'] = idle[-1]
                    break
                require(time.monotonic() < deadline, 'missing device-idle witness')
                time.sleep(.05)  # Client diagnostic wait; outside request timing/model execution.
            print(json.dumps({'case': index, 'file': name, 'usage': response['usage']}), flush=True)
        events = [r for r in trace.records if r['event'] in ('insert', 'evict')]
        require([r['event'] for r in events] == ['insert'] + ['evict', 'insert'] * (2 * wave_count - 1),
                'missing actual current-capacity eviction')
        idle = trace.read()
        require(len(idle) == len(names) and all(r['backend_allocated_bytes'] == idle[-1]['backend_allocated_bytes']
                for r in idle[-3:]), 'missing/still-growing small-image release baseline')
        require(idle[-1]['backend_allocated_bytes'] == idle[4]['backend_allocated_bytes']
                and idle[-1]['backend_allocated_bytes'] < idle[6]['backend_allocated_bytes'],
                'large workspace did not return to prior small baseline')
        report.update(idle_cache=idle, cache_events=events)
        if args.warm_owner:
            report['warm_owner_observations'] = check_warm_sequence(idle, 5, 5)
    finally:
        trace.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--trace-log', required=True, type=Path)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] /
                        'tests/fixtures/native_vision_http')
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--warm-owner', action='store_true',
                        help='add three max,max,small,small,small waves after the historical two-wave warm-up')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous evidence')
    report = {'status': 'FAIL', 'spec_depth': 3, 'cases': [],
              'warm_owner': args.warm_owner,
              'scope': 'ten C1 max/small image requests and release witnesses; separate trace checker required',
              'producer_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    code = 1
    try:
        run(args, report)
        report['status'], code = 'PASS', 0
    except Exception as error:
        report['error'] = repr(error)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'cases': len(report['cases']), 'error': report.get('error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
