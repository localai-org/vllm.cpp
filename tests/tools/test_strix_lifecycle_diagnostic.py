"""Public lifecycle diagnostic with bound CPU fixtures and real publication."""
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import unittest

from tests.tools import test_strix_four_engine_qualify as qualification_fixture
from tests.tools.test_strix_four_engine_qualify import digest, PROMPTS, SETTINGS

ROOT = Path(__file__).resolve().parents[2]
DRIVER = ROOT / 'tools/bench/strix_four_engine/lifecycle_diagnostic.py'
REVISION = '4954b465fa20917016966bf73d001ceead7216c1'

# Keep the real bound-source validators and publisher. Only engine work is fake.
ENGINE_DOUBLE = r'''
import json
from types import SimpleNamespace
from tools.bench.strix_four_engine import child_lifecycle
def event(kind, **fields):
    with open(os.environ['DIAGNOSTIC_EVENTS'], 'a') as stream:
        stream.write(json.dumps(dict(kind=kind, **fields)) + '\n')
event('import')
if os.environ.get('DIAGNOSTIC_CASE') == 'reference-before-parse':
    with open(os.environ['DIAGNOSTIC_REFERENCE'], 'a') as stream: stream.write(' ')
original_manifest_validation = validate_manifest
binding_calls = 0
def validate_manifest(manifest):
    global binding_calls
    binding_calls += 1
    event('bindings', call=binding_calls)
    result = original_manifest_validation(manifest)
    if os.environ.get('DIAGNOSTIC_CASE') == 'observations' and binding_calls == 2:
        result.append(dict(changed=True))
    return result
original_run_validation = validate_run
def validate_run(result, prompts, concurrency, expected=None):
    event('validate_run', prompts=prompts, concurrency=concurrency)
    return original_run_validation(result, prompts, concurrency, expected)
class Adapter:
    def __init__(self, record, output, limit, timeout):
        self.name = record['environment']['variables']['DIAGNOSTIC_ENGINE']
        self.mode = os.environ.get('DIAGNOSTIC_CASE', '')
        self.output = output
        event('construct', engine=self.name, owned=child_lifecycle._controller is not None,
              record=record, limit=limit, timeout=timeout)
        if self.mode == 'constructor':
            failure = RuntimeError('constructor injected')
            failure.lifecycle_evidence = dict(restored=True, adopted=[], cleanup_errors=[])
            raise failure
        self.process = SimpleNamespace(pid=os.getpgrp() if self.mode=='group' else 2000000000, returncode=None)
        self.lifecycle_evidence = dict(restored=False, adopted=[], cleanup_errors=[])
    def exchange(self, command):
        event('exchange', engine=self.name, command=command)
        if command['command'] == 'configure':
            self.config = command
            if self.name in ('vllm.cpp', 'llama.cpp') and not isinstance(command['prompt_ids'], list):
                raise RuntimeError('canonical prompt IDs required')
            if self.mode in ('configure', 'combined', 'combined-publish'): raise RuntimeError('configure injected')
            result = dict(prompt_ids=[[i+10] for i in range(6)], requested=dict(
                backend='rocm',model_dtype='bfloat16',kv_dtype='bfloat16',
                context_per_sequence=2048,slots=4,prefix_caching=False))
            if self.mode == 'prompt-count': result['prompt_ids'].pop()
            if self.mode == 'prompt-token': result['prompt_ids'][0]=[-1]
            if self.mode == 'prompt-drift' and self.name != 'vLLM': result['prompt_ids'][0]=[99]
            if self.mode == 'settings': result['requested']['prefix_caching']=True
            return result
        if self.mode == 'warmup': raise RuntimeError('warmup injected')
        requests = [dict(index=i,prompt_ids=[i+10],output_ids=[i+20]*128,status='ok',
            finish_reason='length',dispatched=100.0+i//command['concurrency'],
            completed=100.9+i//command['concurrency']) for i in range(6)]
        if self.mode == 'structure': requests[0]['output_ids'].pop()
        return dict(requests=requests,started=100.0,completed=requests[-1]['completed'])
    def close(self):
        event('close', engine=self.name)
        self.process.returncode = 0
        self.lifecycle_evidence.update(restored=True, adopted=[dict(pid=2000000001,status=0,cleanup=False)])
        if self.mode == 'parent': self.process.returncode=4
        if self.mode == 'parent-missing': self.process.returncode=None
        if self.mode == 'parent-bool': self.process.returncode=False
        if self.mode == 'restoration': self.lifecycle_evidence['restored']=False
        if self.mode == 'restored-type': self.lifecycle_evidence['restored']=1
        if self.mode == 'missing-restored': self.lifecycle_evidence.pop('restored')
        if self.mode == 'cleanup': self.lifecycle_evidence['cleanup_errors']=['cleanup injected']
        if self.mode == 'missing-cleanup': self.lifecycle_evidence.pop('cleanup_errors')
        if self.mode == 'cleanup-type': self.lifecycle_evidence['cleanup_errors']={}
        if self.mode == 'unresolved': self.lifecycle_evidence['unresolved_children']=[dict(pid=2000000002)]
        if self.mode == 'unresolved-type': self.lifecycle_evidence['unresolved_children']=None
        if self.mode == 'adopted-exit': self.lifecycle_evidence['adopted'][0]['status']=256
        if self.mode == 'adopted-signal': self.lifecycle_evidence['adopted'][0]['status']=9
        if self.mode == 'adopted-type': self.lifecycle_evidence['adopted'][0]['status']=False
        if self.mode == 'adopted-pid-missing': self.lifecycle_evidence['adopted'][0].pop('pid')
        if self.mode == 'adopted-pid-type': self.lifecycle_evidence['adopted'][0]['pid']=True
        if self.mode == 'adopted-pid-invalid': self.lifecycle_evidence['adopted'][0]['pid']=0
        if self.mode == 'adopted-cleanup-missing': self.lifecycle_evidence['adopted'][0].pop('cleanup')
        if self.mode == 'adopted-cleanup': self.lifecycle_evidence['adopted'][0]['cleanup']=True
        if self.mode == 'missing-adopted': self.lifecycle_evidence.pop('adopted')
        if self.mode == 'adopted-container': self.lifecycle_evidence['adopted']={}
        if self.mode == 'model-change':
            with open(self.config['model']+'/config.json','a') as stream: stream.write(' ')
        if self.mode == 'manifest-change':
            with open(os.environ['DIAGNOSTIC_MANIFEST'],'a') as stream: stream.write(' ')
        if self.mode == 'source-change':
            with open(os.environ['DIAGNOSTIC_SOURCE'],'ab') as stream: stream.write(b' ')
        if self.mode == 'reference-change':
            with open(os.environ['DIAGNOSTIC_REFERENCE'],'a') as stream: stream.write(' ')
        if self.mode == 'module-change':
            with open(__file__,'a') as stream: stream.write('\n# changed source\n')
        if self.mode == 'module-path-change':
            copied = self.output/'same-module.py'
            copied.write_bytes(Path(__file__).read_bytes())
            sys.modules[__name__].__file__ = str(copied)
        if self.mode == 'engine-collision': (self.output/'result.json').write_text('{"sentinel":true}')
        if self.mode == 'final-collision': (self.output.parent/'result.json').write_text('{"sentinel":true}')
        if self.mode in ('teardown', 'combined', 'combined-publish'): raise RuntimeError('teardown injected')
'''

PUBLISH_SPY = r'''
_real_publish = publish_result
def publish_result(result, output):
    with open(os.environ['DIAGNOSTIC_EVENTS'], 'a') as stream:
        stream.write(json.dumps(dict(kind='publish',path=str(output),record=result))+'\n')
    mode=os.environ.get('DIAGNOSTIC_CASE','')
    aggregate='engines' in result
    if mode in ('engine-publish','combined-publish') and not aggregate: raise OSError('engine publication injected')
    if mode=='final-publish' and aggregate: raise OSError('final publication injected')
    published = _real_publish(result, output)
    if not aggregate and result['status']=='PASS' and mode.startswith('between-'):
        target={'between-manifest':os.environ['DIAGNOSTIC_MANIFEST'],
                'between-source':os.environ['DIAGNOSTIC_SOURCE'],
                'between-model':os.environ['DIAGNOSTIC_MODEL']}[mode]
        with open(target,'ab') as stream: stream.write(b' ')
    return published
'''

REAL_ADAPTER_OBSERVER = r'''
def event(kind, **fields):
    with open(os.environ['DIAGNOSTIC_EVENTS'], 'a') as stream:
        stream.write(json.dumps(dict(kind=kind, **fields)) + '\n')
TransportAdapter = Adapter
class Adapter(TransportAdapter):
    def __init__(self, record, output, limit, timeout):
        event('construct', engine=record['environment']['variables']['DIAGNOSTIC_ENGINE'], timeout=timeout)
        super().__init__(record, output, limit, timeout)
    def exchange(self, command):
        event('exchange', command=command)
        return super().exchange(command)
    def close(self):
        event('close')
        return super().close()
'''


# CI-context exemption, recorded: the diagnostic drives the qualification
# runtime, which is bound to the measured Strix host (Linux x86-64 glibc 2.39
# only, per tools/bench/strix_four_engine/child_lifecycle.py) and needs the
# Strix lease. Off that host the class SKIPS with this reason instead of
# failing; it still runs for real on the Strix x86-64 lane.
import platform

STRIX_RUNTIME = (sys.platform == "linux" and platform.machine() == "x86_64"
                 and platform.libc_ver()[1] == "2.39")


@unittest.skipUnless(
    STRIX_RUNTIME,
    "the qualification runtime refuses non-Linux-x86_64/glibc-2.39 hosts; no "
    "aarch64 port exists to exercise",
)
class LifecycleDiagnosticTests(unittest.TestCase):
    def test_duplicate_selection_refuses_before_launch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            result, report, events = self.run_cli(root, fixture, ('vLLM', 'vLLM'))
            self.assertEqual(result.returncode, 1)
            self.assertIn('duplicate selected engine', result.stderr)
            self.assertFalse(any(event['kind'] == 'construct' for event in events))
            self.assertIsNone(report)

    def test_same_byte_module_relocation_cannot_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root, 'module-path-change')
            result, report, events = self.run_cli(root, fixture)
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIsNotNone(report)
            self.assertEqual(report['status'], 'FAIL')
            self.assertIn('controller module changed', ' '.join(report['engines']['vLLM']['errors']))

    def test_single_engine_failure_and_publication_error_cannot_pass_aggregate(self):
        for mode in ('warmup', 'engine-publish'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.fixture(root, mode)
                result, report, events = self.run_cli(root, fixture)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIsNotNone(report)
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(list(report['engines']), ['vLLM'])
                if mode == 'engine-publish':
                    self.assertEqual(report['engines']['vLLM']['status'], 'PASS')
                    self.assertIn('publication', ' '.join(report['errors']))
                else:
                    self.assertEqual(report['engines']['vLLM']['status'], 'FAIL')

    def test_sglang_first_supplies_native_canonical_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            names = ('patched SGLang', 'vllm.cpp', 'llama.cpp')
            result, report, events = self.run_cli(root, fixture, names)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNotNone(report)
            self.assertEqual(report['status'], 'PASS')
            configurations = [event['command'] for event in events
                              if event['kind'] == 'exchange' and event['command']['command'] == 'configure']
            self.assertEqual([entry['engine'] for entry in configurations], list(names))
            self.assertIsNone(configurations[0]['prompt_ids'])
            self.assertEqual([entry['prompt_ids'] for entry in configurations[1:]],
                             [[[i + 10] for i in range(6)]] * 2)

    def test_native_first_refuses_before_launch(self):
        for name in ('vllm.cpp', 'llama.cpp'):
            with self.subTest(engine=name), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.fixture(root)
                result, report, events = self.run_cli(root, fixture, (name,))
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertIn('native-first selection requires canonical prompt IDs', result.stderr)
                self.assertFalse(any(event['kind'] == 'construct' for event in events))
                self.assertIsNone(report)
                self.assertFalse(fixture['output'].exists())

    def fixture(self, root, mode=''):
        manifest = qualification_fixture.QualificationTests().fixture(root)
        for name, engine in manifest['engines'].items():
            engine['environment']['variables']['DIAGNOSTIC_ENGINE'] = name
        path = root / 'manifest.json'
        path.write_text(json.dumps(manifest))
        source = root / 'controller.tar'
        members = {}
        for relative in ('tools/__init__.py', 'tools/bench/__init__.py',
                         'tools/bench/strix_four_engine/qualify.py',
                         'tools/bench/strix_four_engine/child_lifecycle.py',
                         'tools/bench/strix_four_engine/audit.py'):
            members[relative] = (ROOT / relative).read_text()
        members['tools/bench/strix_four_engine/qualify.py'] += ENGINE_DOUBLE
        members['tools/bench/strix_four_engine/audit.py'] += PUBLISH_SPY
        self.archive(source, members)
        return dict(source=source, manifest=path, output=root/'output', mode=mode,
                    source_sha=digest(source), manifest_sha=digest(path), members=members)

    def archive(self, path, members, revision=REVISION):
        with tarfile.open(path, 'w', format=tarfile.PAX_FORMAT,
                          pax_headers={'comment': revision}) as archive:
            for name, text in members.items():
                data = text.encode()
                member = tarfile.TarInfo(name)
                member.size = len(data)
                archive.addfile(member, io.BytesIO(data))

    def native_fixture(self, root, mode='', timeout=1):
        fixture = self.fixture(root, mode)
        manifest = json.loads(fixture['manifest'].read_text())
        manifest['timeout_seconds'] = timeout
        fixture['manifest'].write_text(json.dumps(manifest))
        fixture['manifest_sha'] = digest(fixture['manifest'])
        prompts = [[10], [11], [12], [13], [14], [15]]
        reference = dict(schema=1, status='NOT_ACCEPTED', manifest=manifest,
            prompt_ids=prompts, workload_sha256=qualification_fixture.canonical(dict(
                prompts=PROMPTS, prompt_ids=prompts, tokens=128, concurrency=[1, 4], settings=SETTINGS)),
            engines={'vLLM': dict(status='FAIL', configuration=dict(
                schema=1, id=1, status='ok', prompt_ids=prompts, requested=SETTINGS))})
        fixture['reference'] = root / 'reference.json'
        fixture['reference'].write_text(json.dumps(reference))
        fixture['extra_args'] = ['--native-first-c1', '--canonical-reference', str(fixture['reference']),
                                 '--canonical-reference-sha256', digest(fixture['reference'])]
        return fixture

    def test_native_c1_public_entry_runs_only_first_qualification(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.native_fixture(root)
            result, report, events = self.run_cli(root, fixture, ())
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIsNotNone(report)
            self.assertEqual(report['selected_engines'], ['vllm.cpp'])
            commands = [event['command'] for event in events if event['kind'] == 'exchange']
            self.assertEqual([command['command'] for command in commands], ['configure', 'run'])
            self.assertEqual(commands[0]['engine'], 'vllm.cpp')
            self.assertEqual(commands[0]['prompt_ids'], [[10], [11], [12], [13], [14], [15]])
            self.assertEqual(commands[1], dict(command='run', phase='qualification', concurrency=1, repetition=0))
            self.assertEqual(report['diagnostic'], 'native-first-c1')
            self.assertEqual(report['prompt_ids'], commands[0]['prompt_ids'])
            self.assertIn('sha256', report['canonical_reference'])
            self.assertEqual(report['canonical_reference']['sha256'], digest(fixture['reference']))
            self.assertEqual(report['canonical_reference']['path'], str(fixture['reference']))
            item = report['engines']['vllm.cpp']
            self.assertNotIn('warmup', item)
            self.assertEqual(item['qualification'], item['exchanges'][1]['reply'])
            self.assertEqual(item['configuration'], item['exchanges'][0]['reply'])
            self.assertEqual([event['engine'] for event in events if event['kind'] == 'construct'], ['vllm.cpp'])
            self.assertEqual([event['kind'] for event in events if event['kind'] in ('construct', 'exchange', 'close', 'publish')],
                             ['construct', 'exchange', 'exchange', 'close', 'publish', 'publish'])

    def test_native_c1_real_transport_retains_shutdown_and_canonical_configuration(self):
        for mode in ('', 'error', 'timeout', 'exit', 'truncate'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.native_fixture(root)
                manifest = json.loads(fixture['manifest'].read_text())
                manifest['engines']['vllm.cpp']['environment']['variables']['CASE'] = mode
                fixture['manifest'].write_text(json.dumps(manifest))
                fixture['manifest_sha'] = digest(fixture['manifest'])
                reference = json.loads(fixture['reference'].read_text())
                reference['manifest'] = manifest
                fixture['reference'].write_text(json.dumps(reference))
                fixture['extra_args'][-1] = digest(fixture['reference'])
                relative = 'tools/bench/strix_four_engine/qualify.py'
                fixture['members'][relative] = (ROOT / relative).read_text() + REAL_ADAPTER_OBSERVER
                self.archive(fixture['source'], fixture['members'])
                fixture['source_sha'] = digest(fixture['source'])
                result, report, events = self.run_cli(root, fixture, ())
                self.assertEqual(result.returncode, 1 if mode else 0, result.stderr + repr(report))
                self.assertIsNotNone(report, result.stderr)
                item = report['engines']['vllm.cpp']
                self.assertIn('exchanges', item)
                self.assertEqual([event['engine'] for event in events if event['kind'] == 'construct'], ['vllm.cpp'])
                self.assertEqual(len([event for event in events if event['kind'] == 'publish']), 2)
                self.assertEqual(item, json.loads((fixture['output']/'vllm.cpp/result.json').read_text()))
                self.assertTrue(item['group_absent'])
                self.assertTrue(item['lifecycle']['restored'])
                if mode:
                    self.assertEqual(item['status'], 'FAIL')
                    reason = {'error': 'adapter failure', 'timeout': 'adapter command timeout',
                              'exit': 'adapter exited', 'truncate': 'generation stopped or truncated'}[mode]
                    self.assertIn(reason, ' '.join(item['errors']))
                    self.assertLessEqual(len([row for row in item['exchanges'] if row['command']['command'] == 'run']), 1)
                else:
                    self.assertEqual(item['status'], 'PASS')
                    self.assertEqual([row['command']['command'] for row in item['exchanges']], ['configure', 'run', 'shutdown'])
                    self.assertEqual([row['reply']['id'] for row in item['exchanges']], [1, 2, 3])
                    self.assertEqual(item['exchanges'][0]['command']['prompt_ids'], [[10], [11], [12], [13], [14], [15]])
                    self.assertEqual(item['exchanges'][1]['command'], dict(command='run', phase='qualification', concurrency=1, repetition=0))
                    self.assertEqual([event['command']['command'] for event in events if event['kind'] == 'exchange'],
                                     ['configure', 'run', 'shutdown'])

    def test_native_c1_reference_refusals_precede_any_engine_launch(self):
        cases = ('missing-file', 'malformed', 'hash', 'schema-missing', 'schema-bool', 'schema-v2', 'manifest-schema-v2',
                 'manifest', 'model', 'tokenizer', 'oracle-pin', 'config-missing', 'config-schema-bool',
                 'config-schema-v2', 'config-id-bool', 'config-id', 'config-status',
                 'prompts-missing', 'prompts-type', 'prompts-count', 'prompts-empty', 'prompts-bool',
                 'prompts-negative', 'prompts-upper', 'config-prompts', 'config-prompt-bool',
                 'settings', 'settings-type', 'workload', 'workload-missing', 'workload-v2', 'duplicate-key')
        for mode in cases:
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.native_fixture(root)
                reference = json.loads(fixture['reference'].read_text())
                config = reference['engines']['vLLM']['configuration']
                if mode == 'schema-missing': reference.pop('schema')
                if mode == 'schema-bool': reference['schema'] = True
                if mode == 'schema-v2': reference['schema'] = 2
                if mode == 'manifest-schema-v2':
                    reference['manifest']['schema'] = 2
                    fixture['manifest'].write_text(json.dumps(reference['manifest']))
                    fixture['manifest_sha'] = digest(fixture['manifest'])
                if mode == 'manifest': reference['manifest']['run_id'] = 'different'
                if mode == 'model': reference['manifest']['model']['revision'] = 'f'*40
                if mode == 'tokenizer': reference['manifest']['model']['files']['tokenizer.json'] = 'f'*64
                if mode == 'oracle-pin':
                    reference['manifest']['engines']['vLLM']['source_revision'] = 'f'*40
                    fixture['manifest'].write_text(json.dumps(reference['manifest']))
                    fixture['manifest_sha'] = digest(fixture['manifest'])
                if mode == 'config-missing': reference['engines']['vLLM'].pop('configuration')
                if mode == 'config-schema-bool': config['schema'] = True
                if mode == 'config-schema-v2': config['schema'] = 2
                if mode == 'config-id-bool': config['id'] = True
                if mode == 'config-id': config['id'] = 2
                if mode == 'config-status': config['status'] = 'error'
                if mode == 'prompts-missing': reference.pop('prompt_ids')
                if mode == 'prompts-type': reference['prompt_ids'] = {'0': [10]}
                if mode == 'prompts-count':
                    reference['prompt_ids'].pop()
                    config['prompt_ids'].pop()
                for case, value in (('prompts-empty', []), ('prompts-bool', [True]),
                                    ('prompts-negative', [-1]), ('prompts-upper', [151936])):
                    if mode == case:
                        reference['prompt_ids'][0] = value
                        config['prompt_ids'][0] = value
                if mode == 'config-prompts': config['prompt_ids'][0] = [99]
                if mode == 'config-prompt-bool':
                    reference['prompt_ids'][0] = [1]
                    config['prompt_ids'][0] = [True]
                if mode == 'settings': config['requested']['slots'] = 32
                if mode == 'settings-type': config['requested']['prefix_caching'] = 0
                if mode == 'workload': reference['workload_sha256'] = '0'*64
                if mode == 'workload-missing': reference.pop('workload_sha256')
                if mode == 'workload-v2':
                    reference['workload_sha256'] = qualification_fixture.canonical(dict(
                        prompts=PROMPTS, prompt_ids=reference['prompt_ids'], tokens=128,
                        concurrency=[1, 4, 32], settings=SETTINGS))
                if ((mode.startswith('prompts-') and mode not in ('prompts-missing', 'prompts-type')) or
                        mode == 'config-prompt-bool'):
                    reference['workload_sha256'] = qualification_fixture.canonical(dict(
                        prompts=PROMPTS, prompt_ids=reference['prompt_ids'], tokens=128,
                        concurrency=[1, 4], settings=SETTINGS))
                fixture['reference'].write_text(json.dumps(reference))
                if mode == 'malformed': fixture['reference'].write_text('{')
                if mode == 'duplicate-key': fixture['reference'].write_text('{"schema":1,"schema":1}')
                fixture['extra_args'][-1] = digest(fixture['reference'])
                if mode == 'hash': fixture['extra_args'][-1] = '0'*64
                if mode == 'missing-file': fixture['reference'].unlink()
                result, report, events = self.run_cli(root, fixture, ())
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertFalse([event for event in events if event['kind'] == 'construct'])
                self.assertIsNone(report)
                self.assertFalse(fixture['output'].exists())

    def test_native_c1_reference_changes_during_bootstrap_refuse_before_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.native_fixture(root, 'reference-before-parse')
            result, report, events = self.run_cli(root, fixture, ())
            self.assertEqual(result.returncode, 1, result.stderr)
            self.assertIn('canonical reference hash mismatch', result.stderr)
            self.assertIsNone(report)
            self.assertFalse([event for event in events if event['kind'] == 'construct'])
            self.assertFalse(fixture['output'].exists())

    def test_native_c1_reference_options_are_mandatory_and_scoped(self):
        options = (['--native-first-c1'], ['--native-first-c1', '--canonical-reference', 'missing'],
                   ['--native-first-c1', '--canonical-reference-sha256', '0'*64],
                   ['--canonical-reference', 'missing'], ['--canonical-reference-sha256', '0'*64])
        for args in options:
            with self.subTest(args=args), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.native_fixture(root)
                fixture['extra_args'] = args
                result, report, events = self.run_cli(root, fixture, ())
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertFalse(events, 'source imported despite invalid mode arguments')
                self.assertIsNone(report)

    def test_native_c1_engine_conflicts_refuse_before_launch(self):
        for engines in (('vLLM',), ('patched SGLang',), ('llama.cpp',), ('vllm.cpp', 'vLLM'),
                        ('vLLM', 'vllm.cpp'), ('vllm.cpp', 'vllm.cpp')):
            with self.subTest(engines=engines), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.native_fixture(root)
                result, report, events = self.run_cli(root, fixture, engines)
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertFalse([event for event in events if event['kind'] == 'construct'])
                self.assertIsNone(report)

    def test_native_c1_caps_each_exchange_without_relaxing_manifest_timeout(self):
        for timeout, expected in ((7200, 180), (180, 180), (17, 17)):
            with self.subTest(timeout=timeout), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.native_fixture(root, timeout=timeout)
                result, report, events = self.run_cli(root, fixture, ('vllm.cpp',))
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual([event['timeout'] for event in events if event['kind'] == 'construct'], [expected])
                self.assertEqual(report['selected_engines'], ['vllm.cpp'])

    def test_native_c1_failures_finalize_once_without_retry(self):
        for mode, reason in (('configure', 'configure injected'), ('warmup', 'warmup injected'),
                             ('teardown', 'teardown injected'), ('parent', 'lifecycle evidence'),
                             ('cleanup', 'lifecycle evidence'), ('structure', 'generation stopped'),
                             ('reference-change', 'canonical reference hash mismatch'),
                             ('model-change', 'file binding mismatch')):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                fixture = self.native_fixture(root, mode)
                result, report, events = self.run_cli(root, fixture, ())
                self.assertEqual(result.returncode, 1, result.stderr)
                self.assertEqual(report['status'], 'FAIL')
                self.assertEqual(list(report['engines']), ['vllm.cpp'])
                item = report['engines']['vllm.cpp']
                self.assertEqual(item['status'], 'FAIL')
                self.assertIn(reason, ' '.join(item['errors']))
                self.assertEqual([event['engine'] for event in events if event['kind'] == 'construct'], ['vllm.cpp'])
                self.assertEqual(len([event for event in events if event['kind'] == 'publish']), 2)
                self.assertEqual(item, json.loads((fixture['output']/'vllm.cpp/result.json').read_text()))
                if mode in ('configure', 'warmup'):
                    self.assertIn(reason, item['exchanges'][-1]['error'])

    def run_cli(self, root, fixture, engines=('vLLM',), extra_env=None):
        environment = dict(os.environ, RC_DEVICE='strix:gpu0', RC_JOB_ID='cpu-fixture',
            DIAGNOSTIC_EVENTS=str(root/'events.jsonl'), DIAGNOSTIC_CASE=fixture['mode'],
            DIAGNOSTIC_MANIFEST=str(fixture['manifest']), DIAGNOSTIC_SOURCE=str(fixture['source']),
            DIAGNOSTIC_MODEL=str(root/'model/config.json'), DIAGNOSTIC_REFERENCE=str(root/'reference.json'),
            PYTHONDONTWRITEBYTECODE='1')
        environment.pop('PYTHONPATH', None)
        environment.update(extra_env or {})
        args = ['--source', str(fixture['source']), '--manifest', str(fixture['manifest']),
                '--output', str(fixture['output'])]
        for engine in engines:
            args += ['--engine', engine]
        command = [sys.executable, str(DRIVER)] + args + [
            '--source-sha256', fixture['source_sha'], '--source-revision', REVISION,
            '--manifest-sha256', fixture['manifest_sha']] + fixture.get('extra_args', [])
        result = subprocess.run(command, cwd=root, env=environment, text=True,
                                capture_output=True, timeout=20)
        events = [json.loads(line) for line in (root/'events.jsonl').read_text().splitlines()] if (root/'events.jsonl').exists() else []
        report = json.loads((fixture['output']/'result.json').read_text()) if (fixture['output']/'result.json').exists() else None
        return result, report, events

    def test_single_engine_publishes_only_the_terminal_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            fixture = self.fixture(root)
            result, report, events = self.run_cli(root, fixture)
            self.assertEqual(result.returncode, 0, result.stderr + repr(report))
            self.assertIsNotNone(report)
            self.assertEqual(report['status'], 'PASS')
            publications = [event for event in events if event['kind'] == 'publish']
            self.assertEqual([Path(event['path']).relative_to(fixture['output']).as_posix()
                              for event in publications], ['vLLM/result.json', 'result.json'])
            engine = json.loads((fixture['output']/'vLLM/result.json').read_text())
            self.assertEqual(report['engines']['vLLM'], engine)
            self.assertEqual(engine['status'], 'PASS')
            self.assertTrue(engine['group_absent'])
            self.assertTrue(engine['bindings_unchanged'])
            self.assertTrue(engine['lifecycle']['restored'])
            self.assertEqual(engine['parent_status'], 0)
            commands = [event['command'] for event in events if event['kind'] == 'exchange']
            self.assertEqual(commands[0]['prompts'], PROMPTS)
            self.assertEqual(commands[0]['model'], str(root/'model'))
            self.assertEqual(commands[0]['gguf'], json.loads(fixture['manifest'].read_text())['gguf']['path'])
            self.assertEqual(commands[1], dict(command='run',phase='warmup',concurrency=4,repetition=0))
            self.assertTrue(next(event for event in events if event['kind']=='construct')['owned'])
            self.assertLess(next(i for i,e in enumerate(events) if e['kind']=='close'),
                            next(i for i,e in enumerate(events) if e['kind']=='publish'))

    def test_multiple_engines_finalize_before_one_equal_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture=self.fixture(root)
            names=('vLLM','patched SGLang','vllm.cpp','llama.cpp')
            result, report, events=self.run_cli(root,fixture,names)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(report['status'],'PASS')
            self.assertEqual(report['selected_engines'],list(names))
            publications=[e for e in events if e['kind']=='publish']
            self.assertEqual([Path(e['path']).relative_to(fixture['output']).as_posix() for e in publications],
                             [name.replace(' ','-')+'/result.json' for name in names]+['result.json'])
            for name in names:
                item=json.loads((fixture['output']/name.replace(' ','-')/'result.json').read_text())
                self.assertEqual(item,report['engines'][name])
                self.assertEqual(item['status'],'PASS')
            configurations=[e['command'] for e in events if e['kind']=='exchange' and e['command']['command']=='configure']
            self.assertEqual([c['prompt_ids'] for c in configurations],[None]+[[[i+10] for i in range(6)]]*3)
            self.assertEqual(len([e for e in events if e['kind']=='bindings']),8)
            self.assertEqual(len([e for e in events if e['kind']=='validate_run']),4)

    def test_failures_retain_reasons_and_stop_before_next_engine(self):
        cases={'constructor':'constructor injected','configure':'configure injected',
               'warmup':'warmup injected','teardown':'teardown injected','combined':'configure injected',
               'structure':'generation stopped or truncated','prompt-count':'canonical prompt mismatch',
               'prompt-token':'invalid token IDs','settings':'requested settings mismatch',
               'model-change':'file binding mismatch','manifest-change':'manifest',
               'source-change':'archive','module-change':'module',
               'observations':'before/after bindings'}
        for mode,reason in cases.items():
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); fixture=self.fixture(root,mode)
                result,report,events=self.run_cli(root,fixture,('vLLM','patched SGLang'))
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertIsNotNone(report,result.stderr)
                self.assertEqual(report['status'],'FAIL')
                self.assertEqual(list(report['engines']),['vLLM'])
                item=report['engines']['vLLM']
                self.assertEqual(item['status'],'FAIL')
                self.assertIn(reason,' '.join(item['errors']))
                if mode=='combined': self.assertIn('teardown injected',' '.join(item['errors']))
                if mode=='constructor': self.assertTrue(item.get('exception_lifecycles'))
                self.assertEqual(item,json.loads((fixture['output']/'vLLM/result.json').read_text()))
                self.assertEqual(len([e for e in events if e['kind']=='construct']),1)
                self.assertEqual(len([e for e in events if e['kind']=='publish']),2)

    def test_missing_or_failed_lifecycle_observations_cannot_pass(self):
        for mode in ('parent','parent-missing','parent-bool','restoration','restored-type',
                     'missing-restored','cleanup','missing-cleanup','cleanup-type','unresolved','unresolved-type',
                     'adopted-exit','adopted-signal','adopted-type','adopted-pid-missing','adopted-pid-type',
                     'adopted-pid-invalid','adopted-cleanup-missing','adopted-cleanup',
                     'missing-adopted','adopted-container','group'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); fixture=self.fixture(root,mode)
                result,report,events=self.run_cli(root,fixture,('vLLM','patched SGLang'))
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertIsNotNone(report,result.stderr)
                self.assertEqual(report['status'],'FAIL')
                item=report['engines']['vLLM']
                self.assertEqual(item['status'],'FAIL')
                self.assertIn('group' if mode=='group' else 'lifecycle evidence', ' '.join(item['errors']))
                self.assertEqual(len([e for e in events if e['kind']=='construct']),1)

    def test_publication_errors_preserve_available_evidence(self):
        for mode in ('engine-publish','final-publish','engine-collision','final-collision'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); fixture=self.fixture(root,mode)
                result,report,events=self.run_cli(root,fixture,('vLLM','patched SGLang'))
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertEqual(len([e for e in events if e['kind']=='construct']),
                                 1 if mode.startswith('engine-') else 2)
                if mode.startswith('engine-'):
                    self.assertIsNotNone(report,result.stderr)
                    self.assertEqual(report['status'],'FAIL')
                    self.assertIn('publication',' '.join(report['errors']))
                    self.assertEqual(list(report['engines']),['vLLM'])
                else:
                    self.assertIn('publication',result.stderr)
                    self.assertTrue((fixture['output']/'vLLM/result.json').is_file())
                if mode.endswith('collision'):
                    collision=fixture['output']/('vLLM/result.json' if mode.startswith('engine') else 'result.json')
                    self.assertEqual(collision.read_text(),'{"sentinel":true}')

    def test_bootstrap_and_binding_refusals_never_launch(self):
        for mode in ('source-hash','manifest-hash','revision','unsafe-archive','lease-device','lease-job','pre-bindings','fresh-output'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); fixture=self.fixture(root)
                env={}
                if mode=='source-hash': fixture['source_sha']='0'*64
                if mode=='manifest-hash': fixture['manifest_sha']='0'*64
                if mode=='revision':
                    self.archive(fixture['source'],fixture['members'],'f'*40)
                    fixture['source_sha']=digest(fixture['source'])
                if mode=='unsafe-archive':
                    fixture['members']['../'+root.name+'/escaped']='escape'
                    self.archive(fixture['source'],fixture['members'])
                    fixture['source_sha']=digest(fixture['source'])
                if mode=='lease-device': env['RC_DEVICE']='wrong:gpu0'
                if mode=='lease-job': env['RC_JOB_ID']=''
                if mode=='pre-bindings': (root/'model/config.json').write_text('changed')
                if mode=='fresh-output':
                    fixture['output'].mkdir(); (fixture['output']/'sentinel').write_text('keep')
                result,report,events=self.run_cli(root,fixture,extra_env=env)
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertFalse([e for e in events if e['kind']=='construct'])
                self.assertFalse((root/'escaped').exists())
                if mode=='pre-bindings': self.assertIn('file binding mismatch',' '.join(report['engines']['vLLM']['errors']))
                else:
                    self.assertIsNone(report)
                    reason={'source-hash':'archive','manifest-hash':'manifest','revision':'revision',
                            'unsafe-archive':'outside','lease-device':'lease','lease-job':'lease','fresh-output':'exist'}[mode]
                    self.assertIn(reason,result.stderr.lower())
                    if mode!='fresh-output': self.assertFalse(events, 'unverified source was imported')

    def test_import_origins_and_canonical_source_are_required(self):
        for mode in ('qualify-origin','audit-origin','lifecycle-origin','prompts','settings'):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); fixture=self.fixture(root)
                target='tools/bench/strix_four_engine/'
                if mode.endswith('origin'):
                    target += {'qualify-origin':'qualify.py','audit-origin':'audit.py',
                               'lifecycle-origin':'child_lifecycle.py'}[mode]
                    external=root/'external.py'; external.write_text('outside source')
                    fixture['members'][target] += '\n__file__ = '+repr(str(external))+'\n'
                else:
                    target += 'qualify.py'
                    fixture['members'][target] += '\n'+('PROMPTS[0]="different prompt"' if mode=='prompts' else 'RESOLVED["slots"]=32')+'\n'
                self.archive(fixture['source'],fixture['members'])
                fixture['source_sha']=digest(fixture['source'])
                result,report,events=self.run_cli(root,fixture)
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertIsNone(report)
                self.assertIn('import escaped' if mode.endswith('origin') else 'canonical',result.stderr)
                self.assertFalse([e for e in events if e['kind']=='construct'])

    def test_provenance_and_default_selection_are_retained(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture=self.fixture(root)
            result,report,events=self.run_cli(root,fixture,())
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertEqual(report['controller_head'],REVISION)
            self.assertEqual(report['controller_sha256'],fixture['source_sha'])
            self.assertEqual(report['manifest_sha256'],fixture['manifest_sha'])
            self.assertEqual(report['driver_sha256'],digest(DRIVER))
            self.assertEqual(report['driver_path'],str(DRIVER.resolve()))
            self.assertEqual(report['rc_device'],'strix:gpu0')
            self.assertEqual(report['rc_job_id'],'cpu-fixture')
            self.assertEqual(report['selected_engines'],['vLLM','patched SGLang','vllm.cpp','llama.cpp'])
            self.assertEqual(len(report['modules']),5)
            for name,identity in report['modules'].items():
                relative=Path(identity['path']).relative_to(report['controller_source']).as_posix()
                import hashlib
                self.assertEqual(identity['sha256'],hashlib.sha256(fixture['members'][relative].encode()).hexdigest())

    def test_publication_error_does_not_hide_execution_or_teardown_errors(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture=self.fixture(root,'combined-publish')
            result,report,events=self.run_cli(root,fixture,('vLLM','patched SGLang'))
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertEqual(report['status'],'FAIL')
            self.assertIn('engine publication injected',' '.join(report['errors']))
            item=report['engines']['vLLM']
            self.assertIn('configure injected',' '.join(item['errors']))
            self.assertIn('teardown injected',' '.join(item['errors']))
            self.assertEqual(item['status'],'FAIL')
            self.assertEqual(len([e for e in events if e['kind']=='construct']),1)
            self.assertEqual(len([e for e in events if e['kind']=='publish']),2)

    def test_later_prompt_drift_preserves_the_first_finalized_record(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); fixture=self.fixture(root,'prompt-drift')
            result,report,events=self.run_cli(root,fixture,('vLLM','patched SGLang','llama.cpp'))
            self.assertEqual(result.returncode,1,result.stderr)
            self.assertEqual(report['engines']['vLLM']['status'],'PASS')
            self.assertEqual(report['engines']['patched SGLang']['status'],'FAIL')
            self.assertIn('canonical prompt mismatch',' '.join(report['engines']['patched SGLang']['errors']))
            self.assertNotIn('llama.cpp',report['engines'])
            for name in ('vLLM','patched SGLang'):
                self.assertEqual(report['engines'][name],json.loads((fixture['output']/name.replace(' ','-')/'result.json').read_text()))

    def test_bindings_are_rechecked_before_each_engine_launch(self):
        for mode,reason in (('between-manifest','manifest'),('between-source','archive'),('between-model','file binding mismatch')):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                root=Path(directory); fixture=self.fixture(root,mode)
                result,report,events=self.run_cli(root,fixture,('vLLM','patched SGLang','llama.cpp'))
                self.assertEqual(result.returncode,1,result.stderr)
                self.assertEqual(report['status'],'FAIL')
                self.assertEqual(report['engines']['vLLM']['status'],'PASS')
                self.assertEqual(report['engines']['patched SGLang']['status'],'FAIL')
                self.assertIn(reason,' '.join(report['engines']['patched SGLang']['errors']))
                self.assertEqual([e['engine'] for e in events if e['kind']=='construct'],['vLLM'])
                self.assertNotIn('llama.cpp',report['engines'])


if __name__ == '__main__':
    unittest.main()
