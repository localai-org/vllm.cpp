"""Model-free proof of mixed vision test skip/fail and SHA admission semantics."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

BINARY = sys.argv.pop(1)
SYNTHETIC = 'XPU typed vision attention: model-free reference scalar contract'
EXTERNAL = 'XPU typed vision tower: real checkpoint and actual two-image worker boundaries'


class VisionArtifactAdmission(unittest.TestCase):
    def execute(self, mode=None, reference=None, model=None):
        env = os.environ.copy()
        for name in ('EXL3_REQUIRE_ARTIFACTS', 'VT_B70_VISION_REFERENCE_DIR', 'VT_B70_VISION_MODEL_DIR'):
            env.pop(name, None)
        if mode is not None:
            env['EXL3_REQUIRE_ARTIFACTS'] = mode
        if reference is not None:
            env['VT_B70_VISION_REFERENCE_DIR'] = str(reference)
        if model is not None:
            env['VT_B70_VISION_MODEL_DIR'] = str(model)
        result = subprocess.run([BINARY, '--test-case=' + SYNTHETIC + ',' + EXTERNAL,
                                 '--reporters=xml', '--no-colors'], env=env,
                                capture_output=True, text=True, timeout=30)
        tree = ET.fromstring(result.stdout)
        cases = {case.attrib['name']: case for case in tree.iter('TestCase')}
        self.assertIn(SYNTHETIC, cases)
        self.assertNotEqual(cases[SYNTHETIC].get('skipped'), 'true')
        self.assertEqual(cases[SYNTHETIC].find('OverallResultsAsserts').get('failures'), '0')
        self.assertEqual(cases[SYNTHETIC].find('OverallResultsAsserts').get('successes'), '14')
        return result, cases[EXTERNAL]

    def test_optional_missing_artifacts_are_framework_skips(self):
        for mode in (None, '0'):
            result, case = self.execute(mode)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(case.get('skipped'), 'true')
            self.assertIsNone(case.find('OverallResultsAsserts'))

    def test_required_missing_artifacts_fail_without_suppressing_synthetic_case(self):
        result, case = self.execute('1')
        self.assertEqual(result.returncode, 1)
        self.assertNotEqual(case.get('skipped'), 'true')
        self.assertIn('missing EXL3 test artifact', result.stdout)

    def test_invalid_mode_fails_without_a_false_skip(self):
        result, case = self.execute('true')
        self.assertEqual(result.returncode, 1)
        self.assertNotEqual(case.get('skipped'), 'true')
        self.assertIn('must be0 or1', result.stdout)

    def test_corrupt_fixture_hash_fails_before_model_or_gpu_admission(self):
        with tempfile.TemporaryDirectory(prefix='vision artifact ') as directory:
            root = Path(directory)
            (root / 'payload.bin').write_bytes(b'corrupt reference payload')
            (root / 'vision-boundaries-capture.json').write_text(json.dumps({'worker': {'captures': [
                {'file': 'payload.bin', 'sha256': '0' * 64}]}}))
            # No model config/index and no GPU are needed: fixture admission fails first.
            for mode in ('0', '1'):
                result, case = self.execute(mode, root, root)
                self.assertEqual(result.returncode, 1)
                self.assertNotEqual(case.get('skipped'), 'true')
                self.assertIn('vision reference SHA-256 differs', result.stdout)


if __name__ == '__main__':
    unittest.main()
