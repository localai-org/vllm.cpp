"""Portable frozen artifact identities for the bounded alpha attribution tools.

The external tensor captures and model remain caller-supplied. No historical
receipt capsule or private development directory is needed at runtime.
"""
import hashlib
import json
from pathlib import Path

PINS_SHA256 = "90c0050f30cdcc7502df5e872a56c6cac5f98c4618911caadcf6d2d27eee4fb6"
PINS_PATH = Path(__file__).resolve().parents[2] / 'tests/fixtures/native_vision_attribution/pins.json'


def load_attribution_pins(path=None):
    path = PINS_PATH if path is None else Path(path)
    with path.open('rb') as stream:
        raw = stream.read(16 * 1024 + 1)
    if not 0 < len(raw) <= 16 * 1024:
        raise RuntimeError('attribution pin manifest bound exceeded')
    if hashlib.sha256(raw).hexdigest() != PINS_SHA256:
        raise RuntimeError('frozen attribution pin manifest changed')
    return json.loads(raw)
