#!/usr/bin/env python3
"""Model-free portable attribution artifact admission, including Python -O."""
import ast
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
TOOLS = ROOT / 'scripts/mm'
sys.path.insert(0, str(TOOLS))
from native_vision_attribution_pins import PINS_PATH, load_attribution_pins


class AttributionPins(unittest.TestCase):
    def test_public_manifest_relocation(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pins.json'
            path.write_bytes(PINS_PATH.read_bytes())
            self.assertEqual(load_attribution_pins(path), load_attribution_pins())

    def test_missing_manifest_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileNotFoundError):
                load_attribution_pins(Path(temp) / 'pins.json')

    def test_modified_frozen_identity_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pins.json'
            data = load_attribution_pins()
            data['first_mtp_metadata'][0]['sha256'] = '0' * 64
            path.write_text(json.dumps(data))
            with self.assertRaisesRegex(RuntimeError, 'manifest changed'):
                load_attribution_pins(path)

    def test_oversized_manifest_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pins.json'
            path.write_bytes(b'x' * (16 * 1024 + 1))
            with self.assertRaisesRegex(RuntimeError, 'bound exceeded'):
                load_attribution_pins(path)

    def test_empty_manifest_fails(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'pins.json'
            path.write_bytes(b'')
            with self.assertRaisesRegex(RuntimeError, 'bound exceeded'):
                load_attribution_pins(path)

    def test_consumers_without_development_capsules(self):
        # Package just the consumer import closure and public pin fixture;
        # there is deliberately no docs directory, model or GPU runtime.
        modules = {'analyze_native_exl3_vision_adjacent_prefix',
                   'replay_native_exl3_vision_target_states',
                   'capture_native_exl3_vision_python_alpha'}
        pending = list(modules)
        while pending:
            module = pending.pop()
            for node in ast.walk(ast.parse((TOOLS / (module + '.py')).read_text())):
                if isinstance(node, ast.ImportFrom) and node.module and (TOOLS / (node.module + '.py')).is_file():
                    if node.module not in modules:
                        modules.add(node.module)
                        pending.append(node.module)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            tools = root / 'scripts/mm'
            tools.mkdir(parents=True)
            fixture = root / 'tests/fixtures/native_vision_attribution/pins.json'
            fixture.parent.mkdir(parents=True)
            shutil.copyfile(PINS_PATH, fixture)
            for module in modules:
                shutil.copyfile(TOOLS / (module + '.py'), tools / (module + '.py'))
            code = '''
import sys
from pathlib import Path
sys.path.insert(0, sys.argv[1])
from native_vision_attribution_pins import load_attribution_pins
import analyze_native_exl3_vision_adjacent_prefix
import replay_native_exl3_vision_target_states
from capture_native_exl3_vision_python_alpha import recipe
pins = load_attribution_pins()
if len(pins['first_mtp_metadata']) != 4 or len(pins['original_last_prefix_metadata']) != 3:
    raise RuntimeError('missing artifact identities')
try:
    recipe(Path('missing-external-captures'), Path('missing-images'))
except FileNotFoundError as error:
    if 'c2-alpha-mtp-state-v1' not in str(error):
        raise RuntimeError('unexpected missing input') from error
else:
    raise RuntimeError('missing captures accepted')
if any(name in sys.modules for name in ('torch', 'vllm', 'torchvision')):
    raise RuntimeError('GPU runtime initialized during host admission')
print('portable import and fail-closed external admission PASS')
'''
            result = subprocess.run([sys.executable, '-O', '-c', code, str(tools)], cwd=root,
                                    capture_output=True, text=True, timeout=15)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertIn('admission PASS', result.stdout)


if __name__ == '__main__':
    unittest.main()
