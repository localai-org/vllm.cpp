#!/usr/bin/env python3
"""Build a bounded C-ABI request bundle from frozen redistributable images."""
import argparse
import base64
import hashlib
import json
from pathlib import Path

from native_exl3_vision_facts import load_tasks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', default='model', help='C-ABI served name: checkpoint directory basename')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error('output exists; preserve evidence')
    root = Path(__file__).resolve().parents[2] / 'tests/fixtures'
    tasks = load_tasks(root / 'native_vision_qualification')

    def part(path, mime, expected_hash):
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != expected_hash:
            raise RuntimeError('frozen image changed')
        return {'type': 'image_url', 'image_url': {'url': 'data:' + mime + ';base64,' + base64.b64encode(raw).decode()}}

    def body(parts, prompt, limit):
        return {'model': args.model, 'messages': [{'role': 'user', 'content':
            parts + [{'type': 'text', 'text': prompt}]}], 'temperature': 0., 'max_tokens': limit,
            'chat_template_kwargs': {'enable_thinking': False}}

    bundle = []
    for name in ('backup-dialog', 'invoice-unaligned'):
        task = next(t for t in tasks if t['name'] == name)
        image = part(root / 'native_vision_qualification' / task['file'],
                     task['media_type'], task['sha256'])
        bundle.append({'name': name, 'body': body([image], task['prompt'], 128),
                       'expected': task['expected']})
    fixtures = {f['name']: f for f in json.loads((root / 'native_vision_http/fixtures.json').read_text())}
    prompt = ('Read the heading and number in each image. Return a JSON object with exactly '
              'two string fields, "first" for the first image and "second" for the second image. '
              'Each string must contain only that image\'s heading, a space, and its number.')
    aliases = {'orbit': ['ORBIT 731'], 'comet': ['COMET 924']}
    for names in (('orbit', 'comet'), ('comet', 'orbit')):
        images = [part(root / 'native_vision_http' / fixtures[name]['file'],
                       fixtures[name]['media_type'], fixtures[name]['sha256']) for name in names]
        bundle.append({'name': 'ordered-' + '-'.join(names), 'body': body(images, prompt, 64),
                       'expected': {'first': aliases[names[0]], 'second': aliases[names[1]]}})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(bundle, indent=2) + '\n')
    print(json.dumps({'cases': len(bundle), 'names': [c['name'] for c in bundle]}))


if __name__ == '__main__':
    main()
