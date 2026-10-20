"""Strict CPU-only contract tests; synthetic reports are not inference evidence."""
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / 'scripts/mm/check_gemma4_e4b_manifest.py'
FIXTURES = ROOT / 'tests/fixtures/gemma4_e4b_google'
spec = importlib.util.spec_from_file_location('e4b_manifest', SCRIPT)
validator = importlib.util.module_from_spec(spec)
spec.loader.exec_module(validator)


class ManifestTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name) / 'fixtures'
        shutil.copytree(FIXTURES, self.root)
        self.path = self.root / 'manifest.json'
        self.metadata = json.loads(self.path.read_text())

    def save(self):
        self.path.write_text(json.dumps(self.metadata))

    def report(self):
        return dict(
            model_id=validator.MODEL_ID, revision=validator.REVISION,
            vllm_revision=validator.VLLM_REVISION,
            transformers_revision=validator.TRANSFORMERS_REVISION,
            cases=[dict(id=name, status='passed', executed=1, skipped=0,
                        token_mismatches=0, integer_mismatches=0,
                        max_abs_error=0.0, max_rel_error=0.0)
                   for name in validator.MANDATORY_CASES])

    def test_invalid_schema_versions(self):
        for value in [True, False, 1.0, 1.5, '1', None, [], {}, 0, 2, -1]:
            with self.subTest(value=value):
                self.metadata['schema_version'] = value
                self.save()
                with self.assertRaisesRegex(ValueError, '^unsupported schema_version$'):
                    validator.validate_metadata(self.path)

    def corrupt_same_length_fixture(self):
        path = self.root / 'processor_config.json'
        original = path.read_bytes()
        # JSON whitespace changes bytes without changing size or parsed values.
        changed = original.replace(b' ', b'\t', 1)
        self.assertNotEqual(original, changed)
        self.assertEqual(len(original), len(changed))
        self.assertEqual(json.loads(original), json.loads(changed))
        path.write_bytes(changed)
        return path, changed

    def test_same_length_fixture_corruption(self):
        self.corrupt_same_length_fixture()
        with self.assertRaisesRegex(ValueError, '^processor_config.json: hash mismatch$'):
            validator.validate_metadata(self.path)

    def test_coordinated_fixture_and_manifest_replacement(self):
        path, changed = self.corrupt_same_length_fixture()
        entry = next(e for e in self.metadata['files'] if e['file'] == path.name)
        entry['sha256'] = hashlib.sha256(changed).hexdigest()
        self.save()
        with self.assertRaisesRegex(ValueError, '^processor_config.json: fixture pin mismatch$'):
            validator.validate_metadata(self.path)

    def test_same_cardinality_duplicate_and_unknown_cases(self):
        for name in [validator.MANDATORY_CASES[1], 'unknown_case']:
            with self.subTest(name=name):
                report = self.report()
                report['cases'][0]['id'] = name
                self.assertEqual(len(report['cases']), len(validator.MANDATORY_CASES))
                with self.assertRaisesRegex(ValueError, '^unknown or duplicate case$'):
                    validator.validate_report_structure(report)

    def test_noninteger_counts(self):
        for key, values in [('executed', [True, 1.0, 1.5]),
                            ('skipped', [False, 0.0, 0.5])]:
            for value in values:
                with self.subTest(key=key, value=value):
                    report = self.report()
                    report['cases'][0][key] = value
                    with self.assertRaisesRegex(ValueError, f'^text_prefill: invalid {key}$'):
                        validator.validate_report_structure(report)

    def test_runtime_validates_structure_before_pending(self):
        report = self.report()
        report['cases'][0]['executed'] = True
        with self.assertRaisesRegex(ValueError, '^text_prefill: invalid executed$'):
            validator.validate_runtime(report)

    def run_cli(self, *args):
        return subprocess.run(
            [sys.executable, str(SCRIPT), '--metadata', str(self.path), *args],
            capture_output=True, text=True)

    def test_cli_metadata_validation_is_reached(self):
        self.metadata['schema_version'] = True
        self.save()
        report = self.root / 'report.json'
        report.write_text(json.dumps(self.report()))
        for args in [('--offline',), ('--report', str(report))]:
            with self.subTest(args=args):
                result = self.run_cli(*args)
                self.assertEqual(result.returncode, 1)
                self.assertEqual(result.stderr, 'FAIL: unsupported schema_version\n')
                self.assertEqual(result.stdout, '')

    def test_cli_report_validation_is_reached(self):
        report = self.report()
        report['cases'][0]['id'] = 'unknown_case'
        path = self.root / 'report.json'
        path.write_text(json.dumps(report))
        result = self.run_cli('--report', str(path))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr, 'FAIL: unknown or duplicate case\n')
        self.assertEqual(result.stdout, '')

    def test_cli_valid_structure_reaches_pending_barrier(self):
        path = self.root / 'report.json'
        path.write_text(json.dumps(self.report()))
        result = self.run_cli('--report', str(path))
        self.assertEqual(result.returncode, 1)
        self.assertEqual(result.stderr,
                         'FAIL: PENDING: numeric processor/tower bars and measured oracle '
                         'compatibility are unresolved; runtime acceptance is disabled\n')
        self.assertEqual(result.stdout, '')

    def test_upstream_inventory_literal_paths_and_urls(self):
        inventory = json.loads((FIXTURES / 'upstream-tests.json').read_text())
        expected = {
            'tests/models/gemma4/__init__.py',
            'tests/models/gemma4_assistant/__init__.py',
            'tests/models/gemma4_unified/__init__.py',
            'tests/models/gemma4_unified_assistant/__init__.py',
        }
        entries = inventory['files']
        self.assertTrue(expected.issubset({e['path'] for e in entries}))
        for entry in entries:
            with self.subTest(path=entry['path']):
                self.assertNotIn('//', entry['path'])
                self.assertEqual(entry['revision'], {
                    'huggingface/transformers': validator.TRANSFORMERS_REVISION,
                    'vllm-project/vllm': validator.VLLM_REVISION,
                }[entry['repository']])
                self.assertEqual(entry['url'],
                                 f"https://raw.githubusercontent.com/{entry['repository']}/"
                                 f"{entry['revision']}/{entry['path']}")
                if entry['path'] in expected:
                    self.assertEqual(entry['sha256'], hashlib.sha256(b'').hexdigest())
                    self.assertEqual(entry['tests'], [])

    def test_original_metadata(self):
        self.assertEqual(validator.validate_metadata(self.path)['model_id'], validator.MODEL_ID)

    def test_wrong_identity(self):
        for name in ['unsloth/gemma-4-E4B-it', 'google/gemma-4-E2B-it', '']:
            with self.subTest(name=name):
                self.metadata['model_id'] = name
                self.save()
                with self.assertRaisesRegex(ValueError, 'model_id'):
                    validator.validate_metadata(self.path)

    def test_wrong_revision(self):
        for key in ['revision', 'vllm_revision', 'transformers_revision']:
            with self.subTest(key=key):
                original = self.metadata[key]
                self.metadata[key] = '0' * 40
                self.save()
                with self.assertRaisesRegex(ValueError, key):
                    validator.validate_metadata(self.path)
                self.metadata[key] = original

    def test_wrong_hash_and_self_attested_replacement(self):
        self.metadata['files'][0]['sha256'] = '0' * 64
        self.save()
        with self.assertRaisesRegex(ValueError, 'pin'):
            validator.validate_metadata(self.path)

    def test_corrupt_config(self):
        (self.root / 'config.json').write_text('{}')
        with self.assertRaisesRegex(ValueError, 'hash|size'):
            validator.validate_metadata(self.path)

    def test_absent_fixture(self):
        (self.root / 'processor_config.json').unlink()
        with self.assertRaisesRegex(ValueError, 'fixture'):
            validator.validate_metadata(self.path)

    def test_missing_file_entry(self):
        self.metadata['files'].pop()
        self.save()
        with self.assertRaises(ValueError):
            validator.validate_metadata(self.path)

    def test_path_escape_and_symlink(self):
        self.metadata['files'][0]['file'] = '../config.json'
        self.save()
        with self.assertRaises(ValueError):
            validator.validate_metadata(self.path)
        self.metadata = json.loads((FIXTURES / 'manifest.json').read_text())
        self.save()
        (self.root / 'config.json').unlink()
        (self.root / 'config.json').symlink_to(FIXTURES / 'config.json')
        with self.assertRaisesRegex(ValueError, 'symlink'):
            validator.validate_metadata(self.path)

    def test_report_structure_positive_only(self):
        self.assertIsNone(validator.validate_report_structure(self.report()))

    def test_empty_or_missing_case(self):
        for cases in [[], self.report()['cases'][:-1]]:
            report = self.report()
            report['cases'] = cases
            with self.assertRaisesRegex(ValueError, 'case'):
                validator.validate_report_structure(report)

    def test_skipped_mandatory_case(self):
        for key, value in [('status', 'skipped'), ('skipped', 1), ('executed', 0)]:
            report = self.report()
            report['cases'][0][key] = value
            with self.assertRaises(ValueError):
                validator.validate_report_structure(report)

    def test_nonfinite_and_invalid_metrics(self):
        for key in ['max_abs_error', 'max_rel_error', 'token_mismatches', 'integer_mismatches']:
            for value in [float('nan'), float('inf'), -1, True, '0', None]:
                with self.subTest(key=key, value=value):
                    report = self.report()
                    report['cases'][0][key] = value
                    with self.assertRaises(ValueError):
                        validator.validate_report_structure(report)

    def test_permissive_mismatch(self):
        for key in ['token_mismatches', 'integer_mismatches']:
            report = self.report()
            report['cases'][0][key] = 1
            with self.assertRaisesRegex(ValueError, 'mismatch'):
                validator.validate_report_structure(report)
        report = self.report()
        report['allow_mismatch'] = True
        with self.assertRaises(ValueError):
            validator.validate_report_structure(report)

    def test_report_identity_and_duplicates(self):
        report = self.report()
        report['model_id'] = 'unsloth/gemma-4-E4B-it'
        with self.assertRaises(ValueError):
            validator.validate_report_structure(report)
        report = self.report()
        report['cases'].append(copy.deepcopy(report['cases'][0]))
        with self.assertRaises(ValueError):
            validator.validate_report_structure(report)

    def test_unresolved_thresholds_cannot_be_self_certified(self):
        with self.assertRaisesRegex(ValueError, 'PENDING'):
            validator.validate_runtime(self.report())
        report = self.report()
        report['thresholds'] = {'max_abs_error': 1000}
        with self.assertRaises(ValueError):
            validator.validate_runtime(report)

    def test_cli_offline_is_not_gateability(self):
        result = subprocess.run([sys.executable, str(SCRIPT), '--metadata', str(self.path), '--offline'], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        output = json.loads(result.stdout)
        self.assertEqual(output['level'], 'metadata-only')
        self.assertFalse(output['runtime_accepted'])
        self.assertEqual(output['original_model_gateability'], 'PENDING')

    def test_cli_runtime_fails_closed_and_offline_cannot_ignore_report(self):
        report = self.root / 'report.json'
        report.write_text(json.dumps(self.report()))
        for args in [[], ['--report', str(report)], ['--offline', '--report', str(report)]]:
            result = subprocess.run([sys.executable, str(SCRIPT), '--metadata', str(self.path), *args], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)

    def test_duplicate_json_keys_rejected(self):
        self.path.write_text('{"model_id":"wrong","model_id":"google/gemma-4-E4B-it"}')
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            validator.validate_metadata(self.path)


if __name__ == '__main__':
    unittest.main()
