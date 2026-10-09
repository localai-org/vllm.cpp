#!/usr/bin/env python3
"""Verify frozen capture/checkpoint hashes, then run the native tower diagnostic.

Standard-library supervision only, no Python learned inference. A nonzero
native numerical result remains nonzero; this wrapper never rewrites its gates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def require(ok, message):
    if not ok:
        raise RuntimeError(message)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args, report):
    manifest = json.loads((args.reference / 'manifest.json').read_text())
    require(manifest['status'] == 'CAPTURED' and manifest['reference_image'] ==
            'sha256:8d0e1dbe1e6a3a31e79b5ddcc1c050589c08721360af9374b9acd01236f97918', 'wrong/incomplete oracle capture')
    require(manifest['tower'] == {'parameters': 333, 'dtype': 'torch.float16', 'device': 'xpu',
            'source_sha256': '22f02a1ee1faba276a93243d23ecada1094484e39e5bde79452d8a44f32238ba'}, 'wrong actual tower source/dtype')
    require(digest(args.model / 'config.json') == manifest['config_sha256'], 'checkpoint config differs')
    expected = [('orbit', 'native_vision_http/orbit.png'),
                ('comet-unaligned', 'native_vision_http/comet-unaligned.png'),
                ('portrait', 'native_vision_qualification/field-day-portrait.png'),
                ('maximum', 'native_vision_http/orbit-max-2048.png')]
    require([(c['name'], c['image']) for c in manifest['cases']] == expected, 'wrong frozen four-geometry wave')
    count = 0
    for case in manifest['cases']:
        require(digest(args.fixtures / case['image']) == case['image_sha256'], 'fixture changed')
        require(set(case['boundaries']) == {'pixels', 'grid', 'patch', 'position', 'cos', 'sin', 'block0', 'merger'},
                'incomplete capture boundaries')
        for boundary in case['boundaries'].values():
            filename = boundary['file']
            require(Path(filename).name == filename, 'invalid boundary path')
            require(digest(args.reference / filename) == boundary['sha256'], 'captured boundary changed')
            count += 1
    report.update(reference_manifest_sha256=digest(args.reference / 'manifest.json'),
                  binary_sha256=digest(args.binary), config_sha256=manifest['config_sha256'],
                  verified_boundaries=count, all_reference_and_fixture_hashes_verified=True)
    command = [str(args.binary.resolve()), str(args.model.resolve()), str(args.fixtures.resolve()),
               str(args.reference.resolve()), str(args.output.resolve())]
    if args.case:
        command.append(args.case)
        report['selected_case'] = args.case
    report['command'] = command
    report['native_exit_code'] = subprocess.run(command).returncode
    require(args.output.exists(), 'native comparison did not write a report')
    result = json.loads(args.output.read_text())
    require(report['native_exit_code'] in (0, 1), 'native process terminated unexpectedly')
    require((report['native_exit_code'] == 0) == (result['status'] == 'PASS'), 'native exit/status mismatch')
    report['status'] = result['status']
    return report['native_exit_code']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('binary', 'model', 'reference', 'output', 'execution'):
        parser.add_argument('--' + name, required=True, type=Path)
    parser.add_argument('--fixtures', type=Path, default=Path(__file__).resolve().parents[2] / 'tests/fixtures')
    parser.add_argument('--case', choices=('orbit', 'comet-unaligned', 'portrait', 'maximum'),
                        help='one existing geometry for attribution; default qualifies all four')
    args = parser.parse_args()
    if args.output.exists() or args.execution.exists():
        parser.error('output/receipt exists; preserve previous evidence')
    report = {'status': 'FAIL', 'scope': 'verified hash contract and native diagnostic process; external GPU/service cleanup is separate'}
    code = 1
    try:
        code = run(args, report)
    except Exception as error:
        report['error'] = repr(error)
    args.execution.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'status': report['status'], 'native_exit_code': report.get('native_exit_code'), 'error': report.get('error')}))
    return code


if __name__ == '__main__':
    raise SystemExit(main())
