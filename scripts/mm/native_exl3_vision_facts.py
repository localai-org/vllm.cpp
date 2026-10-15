"""Shared exact named-fact scoring for frozen held-out image tasks.

This module does not perform inference or contain model-specific fallbacks.
Expected aliases are frozen in the task manifest before native evaluation.
"""
import hashlib
import json
import re
import struct


def load_tasks(root):
    manifest = json.loads((root / 'tasks.json').read_text())
    tasks = manifest['tasks']
    if len(tasks) != 6 or len({task['name'] for task in tasks}) != 6:
        raise RuntimeError('six distinct frozen held-out tasks required')
    for task in tasks:
        if task['held_out'] is not True or not task['expected']:
            raise RuntimeError('missing held-out/fact contract')
        if hashlib.sha256((root / task['file']).read_bytes()).hexdigest() != task['sha256']:
            raise RuntimeError('fixture changed: ' + task['name'])
        if any(not isinstance(aliases, list) or not aliases or
               not all(isinstance(value, str) for value in aliases)
               for aliases in task['expected'].values()):
            raise RuntimeError('invalid frozen aliases')
    return tasks


def qualification_inputs(root, resize_screenshot=False):
    """Keep the original six-task contract separate from a derived resize case."""
    tasks = load_tasks(root)
    if not resize_screenshot:
        return tasks, 'tasks.json', 'reference-answers.json'
    source = next(task for task in tasks if task['name'] == 'backup-dialog')
    manifest = json.loads((root / 'resize-screenshot.json').read_text())
    task = manifest['task']
    if (manifest['source_sha256'] != source['sha256'] or
            task['name'] != 'backup-dialog-resize' or task['held_out'] is not False or
            task['prompt'] != source['prompt'] or task['expected'] != source['expected'] or
            task['file'] != 'backup-dialog-resize.png' or task['media_type'] != 'image/png' or
            task['size_wh'] != [853, 641] or task['processed_size_wh'] != [864, 640] or
            task['grid_thw'] != [1, 40, 54] or task['visual_rows'] != 540):
        raise RuntimeError('derived screenshot contract changed')
    pixels = (root / task['file']).read_bytes()
    if (hashlib.sha256(pixels).hexdigest() != task['sha256'] or
            pixels[:8] != b'\x89PNG\r\n\x1a\n' or pixels[12:16] != b'IHDR' or
            list(struct.unpack('>II', pixels[16:24])) != task['size_wh']):
        raise RuntimeError('derived screenshot bytes/geometry changed')
    return [task], 'resize-screenshot.json', 'resize-screenshot-reference.json'


def normalize(text):
    return ' '.join(text.casefold().split())


def evaluate(text, task):
    result = {'pass': False, 'facts': None, 'errors': []}
    try:
        clean = text.strip()
        if clean.startswith('```') and clean.endswith('```'):
            clean = re.sub(r'^```(?:json)?\s*', '', clean)[:-3].strip()
        facts = json.loads(clean)
        result['facts'] = facts
        if not isinstance(facts, dict) or set(facts) != set(task['expected']):
            raise RuntimeError('incorrect named fact fields')
        for field, aliases in task['expected'].items():
            value = facts[field]
            if not isinstance(value, str):
                result['errors'].append(field + ': expected string fact')
            elif normalize(value) not in {normalize(alias) for alias in aliases}:
                result['errors'].append(field + ': incorrect fact ' + repr(value))
        result['pass'] = not result['errors']
    except Exception as error:
        result['errors'].append(repr(error))
    return result
