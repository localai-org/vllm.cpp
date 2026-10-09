"""Read cache observations with the observed shared-stderr INFO preamble.

The logger can write its INFO prefix before the runner's complete diagnostic
line. Accept only that known preamble, not arbitrary text containing a marker.
Missing or malformed observations must still fail their owner/count checks.
"""
import json


def cache_record(line):
    if isinstance(line, bytes):
        line = line.decode('utf-8')
    if line.startswith('INFO '):
        line = line[5:]
    prefix = 'NATIVE_VISION_CACHE '
    if not line.startswith(prefix):
        return None
    record = json.loads(line[len(prefix):])
    if not isinstance(record, dict) or record.get('event') not in ('idle', 'insert', 'evict'):
        raise RuntimeError('invalid native vision cache observation')
    return record
