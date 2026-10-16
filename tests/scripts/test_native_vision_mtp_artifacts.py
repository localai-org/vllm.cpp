"""No-GPU admission proof for the standalone external vision-MTP executable."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import xml.etree.ElementTree as ET

BINARY = sys.argv.pop(1)
CASE = 'XPU vision MTP: supplied real image embeddings survive repeat and refusal'


class MtpArtifactAdmission(unittest.TestCase):
    def execute(self, mode=None, root=None):
        env = os.environ.copy()
        for name in ('EXL3_REQUIRE_ARTIFACTS', 'VT_B70_EXL3_MODEL', 'VT_B70_VISION_MTP_CAPTURE'):
            env.pop(name, None)
        if mode is not None:
            env['EXL3_REQUIRE_ARTIFACTS'] = mode
        if root is not None:
            env['VT_B70_EXL3_MODEL'] = env['VT_B70_VISION_MTP_CAPTURE'] = str(root)
        result = subprocess.run([BINARY, '--test-case=' + CASE, '--reporters=xml', '--no-colors'],
                                env=env, capture_output=True, text=True, timeout=30)
        cases = list(ET.fromstring(result.stdout).iter('TestCase'))
        self.assertEqual(len(cases), 1)
        self.assertEqual(cases[0].get('name'), CASE)
        return result, cases[0]

    def test_optional_missing_is_a_framework_skip(self):
        for mode in (None, '0'):
            result, case = self.execute(mode)
            self.assertEqual(result.returncode, 0)
            self.assertEqual(case.get('skipped'), 'true')
            self.assertIsNone(case.find('OverallResultsAsserts'))

    def test_required_missing_fails(self):
        result, case = self.execute('1')
        self.assertEqual(result.returncode, 1)
        self.assertNotEqual(case.get('skipped'), 'true')
        self.assertIn('missing EXL3 test artifact', result.stdout)

    def test_invalid_mode_fails(self):
        result, case = self.execute('true')
        self.assertEqual(result.returncode, 1)
        self.assertNotEqual(case.get('skipped'), 'true')
        self.assertIn('must be0 or1', result.stdout)

    def test_corrupt_capture_fails_before_model_or_gpu(self):
        with tempfile.TemporaryDirectory(prefix='vision mtp artifact ') as directory:
            root = Path(directory)
            (root / 'payload.bin').write_bytes(b'corrupt capture')
            (root / 'capture.json').write_text(json.dumps({'worker': {'captures': [
                {'file': 'payload.bin', 'sha256': '0' * 64}]}}))
            for mode in ('0', '1'):
                result, case = self.execute(mode, root)
                self.assertEqual(result.returncode, 1)
                self.assertNotEqual(case.get('skipped'), 'true')
                self.assertIn('vision reference SHA-256 differs', result.stdout)

    def test_required_boundary_without_checksum_fails_before_gpu(self):
        with tempfile.TemporaryDirectory(prefix='vision mtp unhashed ') as directory:
            root = Path(directory)
            (root / 'payload.bin').write_bytes(bytes(234 * 4))
            (root / 'capture.json').write_text(json.dumps({'status': 'PASS', 'worker': {'captures': [
                {'name': '1-draft-forward-input_ids', 'dtype': 'torch.int32', 'shape': [234],
                 'file': 'payload.bin'}]}}))
            result, case = self.execute('1', root)
            self.assertEqual(result.returncode, 1)
            self.assertNotEqual(case.get('skipped'), 'true')
            self.assertIn('contains("sha256")', ''.join(case.itertext()))


if __name__ == '__main__':
    unittest.main()
