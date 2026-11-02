#!/usr/bin/env python3
"""Build and install real compatibility consumers and run command aliases.

Usage: python3 tests/branding/run_gate.py /path/to/build
The build directory is dedicated to this external consumer.
"""
import pathlib
import shutil
import shlex
import subprocess
import sys
import tempfile

source = pathlib.Path(__file__).resolve().parents[2]
build = pathlib.Path(sys.argv[1]).resolve()


def run(*args):
    print('+', shlex.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True)


run('cmake', '-S', source / 'tests/branding', '-B', build,
    '-DENGINE_SOURCE=' + str(source), '-DVLLM_CPP_CUDA=OFF',
    '-DVLLM_CPP_WITH_DIARIZATION=OFF', '-DVLLM_CPP_BUILD_TESTS=OFF',
    '-DVLLM_CPP_BUILD_EXAMPLES=ON', '-DVLLM_CPP_HF_DOWNLOAD=OFF',
    '-DCMAKE_BUILD_TYPE=Release', '-DCMAKE_CXX_FLAGS_RELEASE=-O0 -DNDEBUG')
run('cmake', '--build', build, '--target', 'branding_old', 'branding_new',
    'branding_shared', 'branding_c', 'branding_old_c', 'branding_static', 'branding_dlopen', 'lumenengine-cli',
    'lumenengine-server', 'lumenengine-bench', '-j4')
run('ctest', '--test-dir', build, '-R', '^branding_', '--output-on-failure')
run(sys.executable, source / 'tests/branding/check_commands.py', build / 'engine/examples')
with tempfile.TemporaryDirectory(prefix='branding-install-', dir=build) as scratch:
    scratch = pathlib.Path(scratch)
    staged = scratch / 'stage'
    relocated = scratch / 'relocated'
    run('cmake', '--install', build / 'engine', '--prefix', staged)
    shutil.move(staged, relocated)
    run('cmake', '-S', source / 'tests/branding', '-B', scratch / 'consumer',
        '-DCMAKE_PREFIX_PATH=' + str(relocated))
    run('cmake', '--build', scratch / 'consumer', '-j4')
    run('ctest', '--test-dir', scratch / 'consumer', '-R', '^branding_', '--output-on-failure')
    run(sys.executable, source / 'tests/branding/check_commands.py', relocated / 'bin')
print('PASS: build-tree and relocated installed consumers and commands', flush=True)
