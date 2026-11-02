#!/usr/bin/env python3
"""Exercise actual old and new command names without loading a model."""
import os
import pathlib
import subprocess
import sys

root = pathlib.Path(sys.argv[1]).resolve()
legacy_version = (pathlib.Path(__file__).resolve().parents[2] /
                  '.agents/evidence/lumenengine-brand/legacy-server-version.txt').read_bytes()
extension = '.exe' if os.name == 'nt' else ''
for suffix in ('cli', 'server', 'bench'):
    old = root / ('vllm-' + suffix + extension)
    new = root / ('lumenengine-' + suffix + extension)
    assert old.is_file(), old
    assert new.is_file(), new
    assert old.read_bytes() == new.read_bytes(), f'{suffix}: aliases diverged'
    for flag in (('--help', '--version') if suffix == 'server' else ('--help',)):
        before = subprocess.run([str(old), flag], capture_output=True)
        after = subprocess.run([str(new), flag], capture_output=True)
        assert before.returncode == after.returncode == 0, (suffix, flag, before.stderr, after.stderr)
        for channel in ('stdout', 'stderr'):
            left = getattr(before, channel).replace(str(old).encode(), b'COMMAND')
            right = getattr(after, channel).replace(str(new).encode(), b'COMMAND')
            assert left == right, (suffix, flag, channel)
        if flag == '--version':
            assert before.stdout == after.stdout == legacy_version, 'legacy server version changed'
        if flag == '--help':
            assert b'LumenEngine' in after.stdout + after.stderr, suffix
print('PASS: cli/server/bench aliases, help, and compatible server version')
