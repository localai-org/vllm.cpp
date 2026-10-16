#!/usr/bin/env python3
"""Exercise the real smoke client against a model-free HTTP fixture."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

CLIENT = Path(__file__).resolve().parents[2] / 'tools/bench/exl3_http_smoke.py'
RUNNING = 'vllm:num_requests_running'
WAITING = 'vllm:num_requests_waiting'
DRAFT = 'vllm:spec_decode_num_draft_tokens_total'


class ClientTest(unittest.TestCase):
    def run_client(self, samples, mode='target', draft=(0, 0)):
        class Handler(BaseHTTPRequestHandler):
            metric_reads = 0
            request_id = 0

            def log_message(self, *_):
                pass

            def reply(self, data, content_type='application/json'):
                encoded = data.encode() if isinstance(data, str) else json.dumps(data).encode()
                self.send_response(200)
                self.send_header('Content-Type', content_type)
                self.send_header('Content-Length', str(len(encoded)))
                self.end_headers()
                try:
                    self.wfile.write(encoded)
                except (BrokenPipeError, ConnectionResetError):
                    pass  # The cancellation probe intentionally closes its stream.

            def do_GET(self):
                if self.path == '/metrics':
                    i = type(self).metric_reads
                    type(self).metric_reads += 1
                    gauges = samples[min(i, len(samples)-1)]
                    values = dict(gauges)
                    if draft is not None:
                        values[DRAFT] = draft[min(i, len(draft)-1)]
                    self.reply('\n'.join(f'{k} {v}' for k, v in values.items())+'\n', 'text/plain')
                elif self.path == '/v1/models':
                    self.reply({'data': [{'id': 'fixture'}]})
                else:
                    self.reply({})

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                if self.path == '/tokenize':
                    self.reply({'count': 3, 'tokens': [1, 2, 3]})
                    return
                count = body['max_tokens'] if body.get('ignore_eos', True) else 1
                reason = 'length' if body.get('ignore_eos', True) else 'stop'
                usage = {'prompt_tokens': 3, 'completion_tokens': count, 'total_tokens': 3+count}
                type(self).request_id += 1
                response = {'id': str(type(self).request_id), 'model': 'fixture', 'usage': usage,
                            'choices': [{'index': 0, 'text': 'x'*count, 'finish_reason': reason,
                                         'logprobs': {'tokens': ['x']*count}}]}
                if self.path == '/v1/chat/completions':
                    response['choices'][0]['message'] = {'role': 'assistant', 'content': 'x'*count}
                if body.get('stream'):
                    self.reply('data: '+json.dumps(response)+'\n\ndata: [DONE]\n\n', 'text/event-stream')
                else:
                    self.reply(response)

        server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        thread = threading.Thread(target=server.serve_forever, kwargs={'poll_interval': .01})
        thread.start()
        try:
            with tempfile.TemporaryDirectory() as temp:
                out = Path(temp) / 'result.json'
                result = subprocess.run([sys.executable, str(CLIENT), '--url',
                    f'http://127.0.0.1:{server.server_port}', '--model', 'fixture',
                    '--mode', mode, '--out', str(out)], capture_output=True, text=True, timeout=12)
                self.assertTrue(out.exists(), result.stderr)
                return result.returncode, json.loads(out.read_text())
        finally:
            server.shutdown()
            thread.join()
            server.server_close()

    def test_present_zero_target_and_mtp(self):
        for mode, draft in [('target', None), ('mtp3', (0, 32))]:
            with self.subTest(mode=mode):
                code, report = self.run_client([{RUNNING: 0, WAITING: 0}], mode, draft)
                self.assertEqual(code, 0, report)
                self.assertEqual(report['status'], 'PASS')

    def test_missing_gauges_fail_both_modes(self):
        for mode, draft in [('target', None), ('mtp3', (0, 32))]:
            for gauges in [{}, {RUNNING: 0}, {WAITING: 0}]:
                with self.subTest(mode=mode, gauges=gauges):
                    code, report = self.run_client([gauges], mode, draft)
                    self.assertEqual(code, 1, report)
                    self.assertIn('missing scheduler gauge', report['error'])

    def test_registered_nonadvancing_draft_is_valid_target(self):
        code, report = self.run_client([{RUNNING: 0, WAITING: 0}], draft=(7, 7))
        self.assertEqual(code, 0, report)

    def test_advancing_draft_is_invalid_target(self):
        code, report = self.run_client([{RUNNING: 0, WAITING: 0}], draft=(7, 8))
        self.assertEqual(code, 1, report)

    def test_invalid_gauges(self):
        for name in [RUNNING, WAITING]:
            for value in ['NaN', '+Inf', '-Inf', -1]:
                with self.subTest(name=name, value=value):
                    gauges = {RUNNING: 0, WAITING: 0, name: value}
                    code, report = self.run_client([gauges], draft=None)
                    self.assertEqual(code, 1, report)
                    self.assertIn('invalid scheduler gauge', report['error'])

    def test_nonzero_gauge_times_out_with_observations(self):
        for name in [RUNNING, WAITING]:
            with self.subTest(name=name):
                code, report = self.run_client([{RUNNING: 0, WAITING: 0},
                    {RUNNING: 0, WAITING: 0, name: 1}], draft=None)
                self.assertEqual(code, 1, report)
                self.assertIn('scheduler drain deadline', report['error'])
                self.assertTrue(report['drain_observations'])
                self.assertTrue(all(x['gauges'][name] == 1 for x in report['drain_observations']))

    def test_delayed_drain_retains_values_and_times(self):
        code, report = self.run_client([{RUNNING: 0, WAITING: 0},
            {RUNNING: 1, WAITING: 2}, {RUNNING: 0, WAITING: 0}], draft=None)
        self.assertEqual(code, 0, report)
        observed = report['drain_observations']
        self.assertEqual(observed[0]['gauges'][WAITING], 2)
        self.assertEqual(observed[-1]['gauges'], {RUNNING: 0, WAITING: 0})
        self.assertLess(observed[0]['elapsed_s'], observed[-1]['elapsed_s'])
        self.assertTrue(all(x['unix_s'] > 0 for x in observed))


if __name__ == '__main__':
    unittest.main()
