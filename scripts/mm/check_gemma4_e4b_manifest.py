#!/usr/bin/env python3
"""Validate pinned Google E4B metadata, not original-checkpoint gateability.

Only the standard library is used. No downloads, imports of oracle runtimes,
weight loading, or inference occur. Runtime acceptance deliberately fails closed
until independent compatibility evidence and numeric bars are ratified.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys

MODEL_ID = 'google/gemma-4-E4B-it'
REVISION = 'ee0ef6023621cff504d758262d4e04895a5af4a2'
VLLM_REVISION = 'a7c23ac96d7806e7c7e7d862eadbce5a33529b94'
TRANSFORMERS_REVISION = 'a08ace4bbd97e721c98751deec37d87b026acadc'
IDENTITY = dict(model_id=MODEL_ID, revision=REVISION,
                vllm_revision=VLLM_REVISION,
                transformers_revision=TRANSFORMERS_REVISION)
# Independent pins: a caller cannot authorize replacement bytes in its manifest.
FILE_PINS = {
    'config.json': (5145, '33b10c02df3c2e8536cf323d29d53262aaa2f4d11dbe19bc729373fbe90295d4'),
    'processor_config.json': (1689, '32bdf45d2ad4cc29a0822ddd157a182de76644f0419a6228d151495256e9813c'),
    'generation_config.json': (208, 'd4226bbe3117d2d253ba4609720ba82c6c4ce4627a9a6ae05387c78983ac03de'),
    'chat_template.jinja': (18569, '0a2c8073c878ab1da004bee933a998606537bbb62016310352c7285c3f01c5b5'),
}
# Minimum structural matrix, not a complete T11 acceptance harness.
MANDATORY_CASES = (
    'text_prefill', 'text_decode', 'text_regression', 'tool_streaming',
    'tool_nonstreaming', 'image_prefill', 'image_decode', 'audio_prefill',
    'audio_decode', 'video_prefill', 'video_decode', 'mixed_prefill',
    'mixed_decode', 'batch_isolation',
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def exact_keys(value, keys, label):
    require(isinstance(value, dict) and set(value) == set(keys),
            f'{label}: missing or unexpected fields')


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        require(key not in result, f'duplicate JSON key: {key}')
        result[key] = value
    return result


def reject_constant(value):
    raise ValueError(f'nonfinite JSON constant: {value}')


def read_json(path):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'),
                          object_pairs_hook=unique_object,
                          parse_constant=reject_constant)
    except (OSError, UnicodeError) as exc:
        raise ValueError(f'cannot read fixture/report: {path}: {exc}') from exc


def check_identity(value):
    for key, expected in IDENTITY.items():
        require(value.get(key) == expected, f'{key}: expected {expected}')


def validate_metadata(path):
    path = Path(path)
    metadata = read_json(path)
    exact_keys(metadata, [*IDENTITY, 'schema_version', 'files'], 'metadata')
    check_identity(metadata)
    require(type(metadata['schema_version']) is int and metadata['schema_version'] == 1,
            'unsupported schema_version')
    entries = metadata['files']
    require(isinstance(entries, list) and len(entries) == len(FILE_PINS),
            'fixture list must contain every pinned file exactly once')
    seen = set()
    for entry in entries:
        exact_keys(entry, ['file', 'bytes', 'sha256', 'url'], 'fixture')
        name = entry['file']
        require(isinstance(name, str) and name in FILE_PINS and name not in seen,
                'unexpected or duplicate fixture pin')
        seen.add(name)
        size, digest = FILE_PINS[name]
        url = f'https://huggingface.co/{MODEL_ID}/resolve/{REVISION}/{name}'
        require(type(entry['bytes']) is int and entry['bytes'] == size
                and entry['sha256'] == digest and entry['url'] == url,
                f'{name}: fixture pin mismatch')
        fixture = path.parent / name
        require(not fixture.is_symlink(), f'{name}: symlink fixture is forbidden')
        require(fixture.is_file(), f'{name}: absent fixture')
        require(fixture.stat().st_size == size, f'{name}: size mismatch')
        require(hashlib.sha256(fixture.read_bytes()).hexdigest() == digest,
                f'{name}: hash mismatch')
    # Byte pins cover the complete original configuration, not selected fields.
    config = read_json(path.parent / 'config.json')
    require(config['architectures'] == ['Gemma4ForConditionalGeneration']
            and config['dtype'] == 'bfloat16'
            and config['text_config']['hidden_size'] == 2560,
            'original E4B config mismatch')
    return metadata


def validate_report_structure(report):
    """Reject invalid evidence structure; a return is NOT runtime acceptance."""
    exact_keys(report, [*IDENTITY, 'cases'], 'report')
    check_identity(report)
    cases = report['cases']
    require(isinstance(cases, list) and len(cases) == len(MANDATORY_CASES),
            'case matrix must contain all mandatory cases exactly once')
    seen = set()
    for case in cases:
        exact_keys(case, ['id', 'status', 'executed', 'skipped', 'token_mismatches',
                          'integer_mismatches', 'max_abs_error', 'max_rel_error'], 'case')
        name = case['id']
        require(isinstance(name, str) and name in MANDATORY_CASES and name not in seen,
                'unknown or duplicate case')
        seen.add(name)
        require(case['status'] == 'passed', f'{name}: case did not pass')
        for key in ['executed', 'skipped', 'token_mismatches', 'integer_mismatches']:
            require(type(case[key]) is int and case[key] >= 0,
                    f'{name}: invalid {key}')
        require(case['executed'] > 0 and case['skipped'] == 0,
                f'{name}: unexecuted or skipped mandatory case')
        require(case['token_mismatches'] == 0 and case['integer_mismatches'] == 0,
                f'{name}: exact token/integer mismatch')
        for key in ['max_abs_error', 'max_rel_error']:
            value = case[key]
            require(type(value) in (int, float), f'{name}: invalid metric {key}')
            try:
                finite = math.isfinite(value)
            except OverflowError:
                finite = False
            require(finite and value >= 0, f'{name}: nonfinite or negative metric {key}')


def validate_runtime(report):
    validate_report_structure(report)
    # No report-supplied tolerance or compatibility flag can lift this barrier.
    raise ValueError('PENDING: numeric processor/tower bars and measured oracle '
                     'compatibility are unresolved; runtime acceptance is disabled')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--offline', action='store_true', help='check metadata only')
    mode.add_argument('--report', type=Path, help='validate runtime evidence (currently PENDING)')
    args = parser.parse_args(argv)
    try:
        validate_metadata(args.metadata)
        if args.report is not None:
            validate_runtime(read_json(args.report))
        print(json.dumps(dict(level='metadata-only', model_id=MODEL_ID,
                              revision=REVISION, runtime_accepted=False,
                              original_model_gateability='PENDING',
                              comparison='fixture bytes against independent pinned SHA-256 values',
                              verified_files=len(FILE_PINS))))
        return 0
    except (ValueError, OSError) as exc:
        print(f'FAIL: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())
