"""Production qualification CLI contracts without GPU dependencies."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tarfile
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
MODEL = '1cfa9a7208912126459214e8b04321603b3df60c'
PINS = {'vllm.cpp': '6e3cbfb940be89e28d1d71c264fd8c3a4e44afeb',
        'vLLM': 'e126687a9a828d513c01a07cd69f025f27d63280',
        'patched SGLang': 'f63458b5beaceabbd9d749b9fc956370e1b649e6',
        'llama.cpp': '10bf611e533d81f739128304991c5e133c6aebd8'}
SETTINGS = dict(backend='rocm', model_dtype='bfloat16', kv_dtype='bfloat16',
                context_per_sequence=2048, slots=4, prefix_caching=False)
PROMPTS = ['The capital city of France is', 'The three primary colors are',
           'Water boils at a temperature of', 'The Pythagorean theorem states that',
           'In 1969, humans first walked on', 'A prime number is a natural number']


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def binding(path):
    return dict(path=str(path), sha256=digest(path))


FAKE = '''import json, os, sys
config = None
mode = os.environ.get('CASE', '')
for line in sys.stdin:
    command = json.loads(line)
    result = dict(schema=1, id=command['id'], status='ok')
    if command['command'] == 'configure':
        config = command
        ids = [[i + 10] for i in range(6)]
        result.update(prompt_ids=ids, requested=json.loads(os.environ['SETTINGS']))
        if mode == 'settings': result['requested']['prefix_caching'] = True
        if mode == 'capacity': result['requested']['context_per_sequence'] = 1024
        if mode == 'prompt': result['prompt_ids'][0] = [99]
    elif command['command'] == 'run':
        c = command['concurrency']
        rep = command['repetition']
        requests = []
        for i in range(6):
            start = 100.0 + (i // c) * (1 + rep)
            ids = [20 + i] * 128
            if mode == 'tokens': ids[127] = 99
            if mode == 'repeat' and rep: ids[127] = 99
            if mode == 'truncate': ids.pop()
            requests.append(dict(index=i, prompt_ids=[i + 10], output_ids=ids,
                dispatched=start, completed=start + 0.9 * (1 + rep), finish_reason='length', status='ok'))
        result.update(requests=requests, started=100.0, completed=max(r['completed'] for r in requests))
        if mode == 'wrong-input': result['requests'][0]['prompt_ids'] = [999]
        if mode == 'overrun':
            for r in requests: r['dispatched'] = 100.0
        if mode == 'timing': result['started'] += 1
        if mode == 'eos': requests[0]['finish_reason'] = 'stop'
        if mode == 'missing': requests.pop()
        if mode == 'model-change':
            with open(config['model'] + '/config.json', 'a') as f: f.write(' ')
    elif command['command'] == 'shutdown':
        print(json.dumps(result), flush=True)
        sys.exit(0)
    if mode == 'identity': result['id'] += 1
    if mode == 'error': result.update(status='error', error='injected')
    if mode == 'exit': sys.exit(4)
    if mode == 'log': sys.stderr.write('x' * 200000); sys.stderr.flush()
    if mode == 'timeout':
        import time; time.sleep(10)
    print(json.dumps(result), flush=True)
    if mode == 'duplicate': print(json.dumps(result), flush=True)
'''


# CI-context exemption, recorded: real qualification is bound to the measured
# Strix host (Linux x86-64 glibc 2.39, per
# tools/bench/strix_four_engine/child_lifecycle.py) and needs the Strix lease.
# Off that host the class SKIPS with this reason instead of failing; it still
# runs for real on the Strix x86-64 lane.
import platform

STRIX_RUNTIME = (sys.platform == "linux" and platform.machine() == "x86_64"
                 and platform.libc_ver()[1] == "2.39")


@unittest.skipUnless(
    STRIX_RUNTIME,
    "the qualification runtime refuses non-Linux-x86_64/glibc-2.39 hosts and "
    "needs the Strix lease; no aarch64 port exists to exercise",
)
class QualificationTests(unittest.TestCase):
    def fixture(self, root, mode='', evidence=True):
        model = root / 'model'
        model.mkdir()
        for name in ('config.json', 'tokenizer.json', 'tokenizer_config.json', 'weights.safetensors'):
            (model / name).write_text('{}')
        (model / 'model.safetensors.index.json').write_text(json.dumps({'weight_map': {'weight': 'weights.safetensors'}}))
        gguf = root / 'model.gguf'
        gguf.write_text('fixture GGUF')
        model_record = dict(revision=MODEL, directory=str(model), files={p.name: digest(p) for p in model.iterdir()})
        audit = root / 'audit.json'
        audit.write_text(json.dumps(dict(result='PASS', tensor_count=398, model=model_record,
            gguf_sha256=digest(gguf), converter=dict(revision=PINS['llama.cpp'],
            inventory_sha256='ad7a105b10602373f7936b15fe9ff76c4ba349b982c985115e2520b9172e0ad7'))))
        adapter = root / 'adapter.py'
        adapter.write_text(FAKE)
        proof = root / 'trace.log'
        proof.write_text('operator inspected fixture')
        manifest = dict(schema=1, run_id='new-run', model=model_record, gguf=binding(gguf), audit=binding(audit),
                        timeout_seconds=1, log_bytes=100000, engines={})
        workload = canonical(dict(prompts=PROMPTS, prompt_ids=[[i + 10] for i in range(6)], tokens=128,
                                  concurrency=[1, 4], settings=SETTINGS))
        for name, revision in PINS.items():
            source = root / (name + '.source.tar')
            with tarfile.open(source, 'w', format=tarfile.PAX_FORMAT, pax_headers={'comment': revision}) as archive:
                archive.add(adapter, arcname='adapter.py')
            env = dict(variables=dict(PATH='/usr/bin:/bin', SETTINGS=json.dumps(SETTINGS),
                       CASE=mode if name == 'patched SGLang' else ''), files=[binding(adapter)])
            engine = dict(source_revision=revision, source=binding(source), binary=binding(Path(sys.executable)),
                          command=[sys.executable, str(adapter)], environment=env)
            if name == 'patched SGLang':
                engine['patch'] = binding(ROOT / 'tools/bench/strix_four_engine/patches/sglang-f63458b5-gfx1151.patch')
            if evidence:
                path = root / (name + '.evidence.json')
                path.write_text(json.dumps(dict(schema=1, verified=True, engine=name, source_revision=revision,
                    binary_sha256=engine['binary']['sha256'], model_revision=MODEL, model_files=model_record['files'],
                    gguf_sha256=digest(gguf), workload_sha256=workload, environment_sha256=canonical(env),
                    resolved=SETTINGS, evidence_run_id='prior-trace-run', artifacts=[dict(binding(proof), kind='trace')])) )
                engine['runtime_evidence'] = binding(path)
            manifest['engines'][name] = engine
        return manifest

    def run_cli(self, root, manifest):
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        result = subprocess.run([sys.executable, '-m', 'tools.bench.strix_four_engine.qualify',
                                 '--manifest', str(path), '--output', str(root / 'output')],
                                cwd=ROOT, text=True, capture_output=True, timeout=35,
                                env=dict(os.environ, RC_DEVICE='strix:gpu0', RC_JOB_ID='cpu-fixture'))
        return result, json.loads((root / 'output/result.json').read_text())

    def test_all_corpora_and_true_wall_throughput_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, report = self.run_cli(root, self.fixture(root))
            self.assertEqual(result.returncode, 0, result.stderr + str(report))
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(set(report['engines']), set(PINS))
            for engine in report['engines'].values():
                self.assertEqual(len(engine['runs']), 12)
                for c, duration in (('1', 5.9), ('4', 1.9)):
                    rates = engine['throughput'][c]
                    self.assertEqual(len(rates['values']), 3)
                    for actual, multiplier in zip(rates['values'], (2, 3, 4)):
                        self.assertAlmostEqual(actual, 768 / (duration * multiplier))
                    self.assertEqual(rates['median'], rates['values'][1])
                    self.assertEqual(rates['minimum'], min(rates['values']))
                    self.assertEqual(rates['maximum'], max(rates['values']))
                    self.assertEqual(rates['ratio_to_vllm'], 1)

    def test_missing_runtime_proof_retains_diagnostics_without_measurement(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result, report = self.run_cli(root, self.fixture(root, evidence=False))
            self.assertEqual(result.returncode, 1)
            for engine in report['engines'].values():
                self.assertEqual(engine['status'], 'PENDING')
                self.assertEqual(len(engine['runs']), 4)
                self.assertNotIn('throughput', engine)

    def test_runtime_evidence_requires_a_distinct_prior_run_identity(self):
        for identity in ('new-run', '', '   ', None, True, 42, ['prior-run']):
            with self.subTest(identity=identity), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                for engine in manifest['engines'].values():
                    path = Path(engine['runtime_evidence']['path'])
                    evidence = json.loads(path.read_text())
                    evidence['evidence_run_id'] = identity
                    path.write_text(json.dumps(evidence))
                    engine['runtime_evidence'] = binding(path)
                result, report = self.run_cli(root, manifest)
                self.assertEqual(result.returncode, 1, report['status'])
                self.assertEqual(report['status'], 'NOT_ACCEPTED')
                for engine in report['engines'].values():
                    self.assertEqual(engine['status'], 'FAIL')
                    self.assertIn('unverified or mismatched runtime evidence', engine['error'])
                    self.assertNotIn('throughput', engine)

    def test_bad_adapter_outputs_and_changed_bindings_never_qualify(self):
        for mode in ('settings', 'capacity', 'prompt', 'tokens', 'repeat', 'truncate', 'wrong-input',
                     'overrun', 'timing', 'eos', 'missing', 'identity', 'error', 'exit', 'log', 'timeout',
                     'duplicate', 'model-change'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                result, report = self.run_cli(root, self.fixture(root, mode))
                self.assertEqual(result.returncode, 1, str(report))
                self.assertNotEqual(report['status'], 'PASS')
                if 'engines' in report:
                    self.assertEqual(set(report['engines']), set(PINS))
                    self.assertEqual(report['engines']['patched SGLang']['status'], 'FAIL')
                    self.assertTrue(all('throughput' not in e for e in report['engines'].values()))

    def test_invalid_runtime_evidence_and_manifest_bindings_are_refused(self):
        for mode in ('audit-hash', 'audit-model', 'model-hash', 'binary-hash', 'source-pin', 'argv',
                     'environment-hash', 'unverified', 'cpu', 'wrong-workload', 'wrong-environment',
                     'wrong-binary', 'wrong-model', 'missing-trace', 'trace-hash', 'duplicate-json', 'patch', 'log-only'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                engine = manifest['engines']['patched SGLang']
                if mode == 'audit-hash': manifest['audit']['sha256'] = '0'*64
                if mode == 'audit-model':
                    path = Path(manifest['audit']['path'])
                    value = json.loads(path.read_text());value['model']['revision'] = 'wrong'
                    path.write_text(json.dumps(value));manifest['audit'] = binding(path)
                if mode == 'model-hash': manifest['model']['files']['config.json'] = '0'*64
                if mode == 'binary-hash': engine['binary']['sha256'] = '0'*64
                if mode == 'source-pin': engine['source_revision'] = '0'*40
                if mode == 'argv': engine['command'] = 'echo bad'
                if mode == 'environment-hash': engine['environment']['files'][0]['sha256'] = '0'*64
                if mode == 'patch':
                    path = root / 'wrong.patch'; path.write_text('different patch')
                    engine['patch'] = binding(path)
                proof = Path(engine['runtime_evidence']['path'])
                value = json.loads(proof.read_text())
                if mode == 'unverified': value['verified'] = False
                if mode == 'cpu': value['resolved']['backend'] = 'cpu'
                if mode == 'wrong-workload': value['workload_sha256'] = '0'*64
                if mode == 'wrong-environment': value['environment_sha256'] = '0'*64
                if mode == 'wrong-binary': value['binary_sha256'] = '0'*64
                if mode == 'wrong-model': value['model_files']['config.json'] = '0'*64
                if mode == 'missing-trace': value['artifacts'] = []
                if mode == 'log-only': value['artifacts'][0]['kind'] = 'log'
                if mode == 'trace-hash': value['artifacts'][0]['sha256'] = '0'*64
                proof.write_text(json.dumps(value))
                if mode == 'duplicate-json': proof.write_text('{"verified":true,' + proof.read_text()[1:])
                engine['runtime_evidence'] = binding(proof)
                result, report = self.run_cli(root, manifest)
                self.assertEqual(result.returncode, 1, str(report))
                self.assertNotEqual(report['status'], 'PASS')

    def test_no_lease_never_launches_a_bound_adapter(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / 'manifest.json'
            manifest.write_text(json.dumps(self.fixture(root)))
            env = dict(os.environ)
            env.pop('RC_DEVICE', None)
            env.pop('RC_JOB_ID', None)
            result = subprocess.run([sys.executable, '-m', 'tools.bench.strix_four_engine.qualify',
                                     '--manifest', str(manifest), '--output', str(root / 'output')],
                                    cwd=ROOT, env=env, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertIn('qualification requires a Strix lease', result.stderr)
            self.assertEqual([p.name for p in (root / 'output').iterdir()], ['result.json'])

    def test_rebound_provenance_fields_are_independently_refused(self):
        cases = ('pax', 'result', 'tensor_count', 'files', 'gguf_sha256',
                 'converter_revision', 'inventory_sha256', 'insecure')
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                manifest = self.fixture(root)
                engine = manifest['engines']['patched SGLang']
                expected = 'conversion audit does not bind model'
                if case == 'pax':
                    source = Path(engine['source']['path'])
                    with tarfile.open(source, 'w', format=tarfile.PAX_FORMAT,
                                      pax_headers={'comment': '0' * 40}) as archive:
                        archive.add(root / 'adapter.py', arcname='adapter.py')
                    engine['source'] = binding(source)
                    expected = 'source archive revision mismatch'
                elif case == 'insecure':
                    engine['environment']['variables']['VLLM_ALLOW_INSECURE_SERIALIZATION'] = '1'
                    proof = Path(engine['runtime_evidence']['path'])
                    value = json.loads(proof.read_text())
                    value['environment_sha256'] = canonical(engine['environment'])
                    proof.write_text(json.dumps(value))
                    engine['runtime_evidence'] = binding(proof)
                    expected = 'insecure serialization refused'
                else:
                    path = Path(manifest['audit']['path'])
                    value = json.loads(path.read_text())
                    if case == 'result': value['result'] = 'FAIL'
                    if case == 'tensor_count': value['tensor_count'] = 397
                    if case == 'files': value['model']['files']['config.json'] = '0' * 64
                    if case == 'gguf_sha256': value['gguf_sha256'] = '0' * 64
                    if case == 'converter_revision': value['converter']['revision'] = '0' * 40
                    if case == 'inventory_sha256': value['converter']['inventory_sha256'] = '0' * 64
                    path.write_text(json.dumps(value))
                    manifest['audit'] = binding(path)
                result, report = self.run_cli(root, manifest)
                self.assertEqual(result.returncode, 1, report['status'])
                self.assertIn(expected, result.stderr)
                self.assertNotIn('engines', report)  # Refuse before an adapter starts.

    def test_copied_audit_and_relocated_model_keep_content_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = self.fixture(root)
            original_audit = Path(manifest['audit']['path'])
            copied_audit = root / 'copied-audit.json'
            shutil.copyfile(original_audit, copied_audit)
            relocated = root / 'relocated-model'
            shutil.copytree(manifest['model']['directory'], relocated)
            manifest['model']['directory'] = str(relocated)
            manifest['audit'] = binding(copied_audit)
            result, report = self.run_cli(root, manifest)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(report['status'], 'PASS')
            self.assertEqual(copied_audit.read_bytes(), original_audit.read_bytes())
            self.assertEqual(set(report['engines']), set(PINS))
            self.assertTrue(all(len(e['runs']) == 12 for e in report['engines'].values()))

    def test_invalid_manifest_never_runs_or_overwrites(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest = root / "manifest.json"
            manifest.write_text('{}')
            output = root / "output"
            result = subprocess.run([sys.executable, '-m', 'tools.bench.strix_four_engine.qualify',
                                     '--manifest', str(manifest), '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertIn('QUALIFICATION_FAIL', result.stderr)
            self.assertTrue((output / 'result.json').is_file())
            sentinel = output / 'sentinel'
            sentinel.write_text('preserve')
            result = subprocess.run([sys.executable, '-m', 'tools.bench.strix_four_engine.qualify',
                                     '--manifest', str(manifest), '--output', str(output)],
                                    cwd=ROOT, capture_output=True, text=True, timeout=10)
            self.assertEqual(result.returncode, 1)
            self.assertIn('output already exists', result.stderr)
            self.assertEqual(sentinel.read_text(), 'preserve')


if __name__ == '__main__':
    unittest.main()
