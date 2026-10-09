#!/usr/bin/env python3
"""Freeze an unaligned derivative of the authored backup dialog before inference."""
import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image, __version__ as pillow_version


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path, required=True, help='new output directory or existing fixture directory')
    args = parser.parse_args()
    source = next(task for task in json.loads((args.source / 'tasks.json').read_text())['tasks']
                  if task['name'] == 'backup-dialog')
    source_path = args.source / source['file']
    if hashlib.sha256(source_path.read_bytes()).hexdigest() != source['sha256']:
        raise RuntimeError('frozen source screenshot changed')
    image_path = args.output / 'backup-dialog-resize.png'
    manifest_path = args.output / 'resize-screenshot.json'
    if image_path.exists() or manifest_path.exists():
        parser.error('preserve frozen derivative; use a new directory')
    args.output.mkdir(parents=True, exist_ok=True)
    with Image.open(source_path) as original:
        if original.size != (640, 480) or original.mode != 'RGB':
            raise RuntimeError('unexpected source geometry/mode')
        original.resize((853, 641), Image.Resampling.BICUBIC).save(image_path)
    task = dict(source, name='backup-dialog-resize', file=image_path.name, held_out=False,
                sha256=hashlib.sha256(image_path.read_bytes()).hexdigest(), size_wh=[853, 641],
                processed_size_wh=[864, 640], grid_thw=[1, 40, 54], visual_rows=540)
    manifest = {'scope': 'derived resize-required screenshot, not an additional held-out task',
                'source_sha256': source['sha256'], 'generation': {'pillow': pillow_version,
                'operation': 'RGB bicubic resize of authored 640x480 dialog to 853x641'}, 'task': task}
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'image_sha256': task['sha256'], 'size_wh': task['size_wh'],
                      'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
