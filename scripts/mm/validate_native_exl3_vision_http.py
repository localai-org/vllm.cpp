#!/usr/bin/env python3
"""Target-only C1 native PNG/JPEG HTTP semantic/SSE gate. No inference fallback.

Use an already started native EXL3 server. This stdlib client never manages
services. Failed semantics, usage, SSE or telemetry exit nonzero.
"""
import argparse, base64, hashlib, json, math, time, urllib.request
from pathlib import Path

def run(args, report):
    fixtures = json.loads((args.fixtures / 'fixtures.json').read_text())
    prompt = 'Read the large heading and number exactly. Then describe the color and shape on the left, followed by the color and shape on the right. Explain what is visible in a few complete English sentences.'
    if not len(fixtures) == 4:
        raise RuntimeError('exactly four fixtures required')
    url = args.url.rstrip('/')

    def gauges():
        with urllib.request.urlopen(url + '/metrics', timeout=5) as f:
            raw = f.read().decode()
        result = {}
        for line in raw.splitlines():
            if line.startswith('#') or not line.strip():
                continue
            k = line.split('{', 1)[0].split(' ', 1)[0]
            if k in ('vllm:num_requests_running', 'vllm:num_requests_waiting'):
                v = float(line.rsplit(' ', 1)[-1])
                if not (math.isfinite(v) and v >= 0):
                    raise RuntimeError('math.isfinite(v) and v >= 0')
                result[k] = result.get(k, 0) + v
        if not len(result) == 2:
            raise RuntimeError(result)
        return result
    report['initial_gauges'] = gauges()
    for fixture, stream in [(f, False) for f in fixtures] + [(fixtures[0], True)]:
        pixels = (args.fixtures / fixture['file']).read_bytes()
        if not hashlib.sha256(pixels).hexdigest() == fixture['sha256']:
            raise RuntimeError('fixture changed')
        media = 'data:' + fixture['media_type'] + ';base64,' + base64.b64encode(pixels).decode()
        body = {'model': args.model, 'messages': [{'role': 'user', 'content': [{'type': 'image_url', 'image_url': {'url': media}}, {'type': 'text', 'text': prompt}]}], 'temperature': 0.0, 'max_tokens': 64, 'stream': stream, 'chat_template_kwargs': {'enable_thinking': False}}
        if stream:
            body['stream_options'] = {'include_usage': True}
        before = time.monotonic()
        case = {'fixture': fixture, 'stream': stream, 'request': body}
        report['cases'].append(case)
        req = urllib.request.Request(url + '/v1/chat/completions', data=json.dumps(body).encode(), headers={'Content-Type': 'application/json', 'Connection': 'close'})
        with urllib.request.urlopen(req, timeout=120) as f:
            if not f.status == 200:
                raise RuntimeError('f.status == 200')
            raw = f.read().decode()
            case['raw_response'] = raw
        case['wall_s'] = time.monotonic() - before
        if stream:
            parts = [json.loads(line[6:]) for line in raw.splitlines() if line.startswith('data: ') and line[6:] != '[DONE]']
            if not raw.count('data: [DONE]') == 1:
                raise RuntimeError('missing or duplicate SSE completion')
            text = ''.join((c['delta'].get('content', '') or '' for part in parts for c in part.get('choices', [])))
            usage = [p['usage'] for p in parts if p.get('usage')][-1]
            finishes = [c['finish_reason'] for p in parts for c in p.get('choices', []) if c.get('finish_reason')]
            if not len(finishes) == 1:
                raise RuntimeError('len(finishes) == 1')
            finish = finishes[0]
        else:
            response = json.loads(raw)
            text = response['choices'][0]['message'].get('content') or ''
            usage = response['usage']
            finish = response['choices'][0]['finish_reason']
        case.update(text=text, usage=usage, finish_reason=finish)
        if not (usage['prompt_tokens'] > 192 and usage['completion_tokens'] > 0):
            raise RuntimeError("usage['prompt_tokens'] > 192 and usage['completion_tokens'] > 0")
        if not usage['total_tokens'] == usage['prompt_tokens'] + usage['completion_tokens']:
            raise RuntimeError('incorrect total usage')
        if not finish in ('length', 'stop'):
            raise RuntimeError("finish in ('length', 'stop')")
        if not (usage['completion_tokens'] == 64 or finish == 'stop'):
            raise RuntimeError("usage['completion_tokens'] == 64 or finish == 'stop'")
        case['missing_semantic_words'] = [word for word in fixture['expected'] if word not in text.lower()]
        for side, expected in fixture['sides'].items():
            start = text.lower().find(side)
            words = text.lower()[start:start + 100] if start >= 0 else ''
            case['missing_semantic_words'] += [side + ':' + word for word in expected if word not in words]
        case['semantic_pass'] = not case['missing_semantic_words']
        case['gauges'] = gauges()
        if not all((v == 0 for v in case['gauges'].values())):
            raise RuntimeError("all((v == 0 for v in case['gauges'].values()))")
        print(json.dumps({'fixture': fixture['name'], 'stream': stream, 'text': text, 'semantic_pass': case['semantic_pass'], 'usage': usage}), flush=True)
    report['stream_text_equal'] = report['cases'][0]['text'] == report['cases'][-1]['text']
    if not report['stream_text_equal']:
        raise RuntimeError("report['stream_text_equal']")
    if not report['cases'][0]['text'] != report['cases'][1]['text']:
        raise RuntimeError('different images ignored')
    if not all((c['semantic_pass'] for c in report['cases'])):
        raise RuntimeError('semantic mismatch')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--url', default='http://127.0.0.1:8000')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_http')
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve previous evidence')
    report = {'status': 'FAIL', 'scope': 'C1 target-only PNG HTTP functional gate; not Python numerical, MTP, C4 or throughput parity', 'cases': []}
    result = 1
    try:
        run(args, report)
        report['status'] = 'PASS'
        result = 0
    except Exception as error:
        report['error'] = repr(error)
        print(report['error'], flush=True)
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + '\n')
    return result
if __name__ == '__main__':
    raise SystemExit(main())
