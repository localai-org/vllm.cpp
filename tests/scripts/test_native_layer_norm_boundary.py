#!/usr/bin/env python3
"""Focused N1 frozen-operand admission negatives; no Torch or GPU import."""
import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts/mm'))
from capture_native_exl3_layer_norm_boundary import read_operands, EXPECTED, CONFIG_SHA256


class BoundaryAdmission(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = b'\0\0' * (768 * 1152)
        sha = hashlib.sha256(raw).hexdigest()
        stages = {}
        for role in ('input', 'residual1', 'norm1', 'norm2'):
            name = role + '.float16'
            (self.root / name).write_bytes(raw)
            stages[role] = {'file': name, 'dtype': 'torch.float16', 'shape': [768, 1, 1152], 'sha256': sha}
        self.doc = {'status': 'CAPTURED', 'block': 26, 'runtime': EXPECTED,
                    'config_sha256': CONFIG_SHA256, 'matches_original_block_output': True,
                    'stages': stages, 'parameter_f16_sha256': {
                        norm + '.' + role: 'a' * 64 for norm in ('norm1', 'norm2')
                        for role in ('weight', 'bias')}}

    def manifest(self):
        path = self.root / 'capture.json'
        path.write_text(json.dumps(self.doc))
        return hashlib.sha256(path.read_bytes()).hexdigest()

    def test_complete_full_shape_admitted_without_gpu(self):
        cases = read_operands(self.root, self.manifest(), 26)
        self.assertEqual([c['label'] for c in cases], ['block26-norm1', 'block26-norm2'])
        self.assertNotIn('torch', sys.modules)
        self.assertNotIn('vllm', sys.modules)

    def test_changed_manifest_rejected(self):
        sha = self.manifest()
        self.doc['block'] = 25
        self.manifest()
        with self.assertRaisesRegex(RuntimeError, 'capture identity differs'):
            read_operands(self.root, sha, 26)

    def test_same_length_corrupt_payload_rejected(self):
        sha = self.manifest()
        path = self.root / 'input.float16'
        data = bytearray(path.read_bytes()); data[-1] = 1; path.write_bytes(data)
        with self.assertRaisesRegex(RuntimeError, 'operand bytes differ'):
            read_operands(self.root, sha, 26)

    def test_row_slice_with_matching_bytes_rejected(self):
        self.doc['stages']['input']['shape'] = [1, 1, 1152]
        with self.assertRaisesRegex(RuntimeError, 'full shape/dtype required'):
            read_operands(self.root, self.manifest(), 26)

    def test_path_escape_rejected(self):
        self.doc['stages']['input']['file'] = '../input.float16'
        with self.assertRaisesRegex(RuntimeError, 'invalid operand path'):
            read_operands(self.root, self.manifest(), 26)

    def test_symlink_operand_rejected(self):
        path = self.root / 'input.float16'
        path.rename(self.root / 'real.float16')
        path.symlink_to('real.float16')
        with self.assertRaisesRegex(RuntimeError, 'invalid operand path'):
            read_operands(self.root, self.manifest(), 26)

    def test_wrong_runtime_rejected_even_with_new_manifest_hash(self):
        self.doc['runtime'] = copy.deepcopy(EXPECTED)
        self.doc['runtime']['torch_git'] = '0' * 40
        with self.assertRaisesRegex(RuntimeError, 'runtime/config pin differs'):
            read_operands(self.root, self.manifest(), 26)


if __name__ == '__main__':
    unittest.main()
