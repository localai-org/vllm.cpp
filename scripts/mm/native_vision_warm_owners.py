"""Strict fixed-shape observations after three successful warm-up requests.

These backend/cache/pool counters complement weak-owner tests; they are not an
independent census of every request owner or untracked driver allocation.
"""

WARM_FIELDS = (
    'backend_allocated_bytes', 'scratch_pool_retained_bytes',
    'scratch_pool_misses', 'scratch_pool_live_blocks',
    'backend_graph_count', 'backend_graph_device_bytes',
    'backend_exl3_workspace_bytes', 'backend_w8a8_workspace_bytes',
    'backend_sampling_workspace_bytes', 'entries', 'embedding_bytes',
)


def check_warm_sequence(records, baseline_start, sequence_length):
    """Compare like phases of a baseline wave and three additional waves."""
    if baseline_start < 0 or sequence_length <= 0 or len(records) != baseline_start + 4 * sequence_length:
        raise RuntimeError('one baseline sequence and three additional warm sequences required')
    warm = records[baseline_start:]
    for row in warm:
        for key in WARM_FIELDS:
            if type(row.get(key)) is not int or row[key] < 0:
                raise RuntimeError('missing/invalid warm owner counter: ' + key)
        if row['scratch_pool_retained_bytes'] > row['backend_allocated_bytes']:
            raise RuntimeError('free pool exceeds backend allocation')
        if row['backend_graph_count'] == 0:
            raise RuntimeError('warm graph family was not exercised')
    baseline = [{key: row[key] for key in WARM_FIELDS} for row in warm[:sequence_length]]
    for index, row in enumerate(warm[sequence_length:], sequence_length):
        for key, value in baseline[index % sequence_length].items():
            if row[key] != value:
                raise RuntimeError(f'warm owner counter changed at wave {index // sequence_length}, '
                                   f'phase {index % sequence_length}: {key}')
    return {'baseline_start_request_index': baseline_start, 'sequence_length': sequence_length,
            'additional_same_sequence_repetitions': 3, 'baseline': baseline,
            'scope': 'like-phase backend/cache/pool observations; separate weak-owner and teardown proof required'}


def check_warm_owners(retries):
    if len(retries) != 6:
        raise RuntimeError('three warm-up and three additional warm requests required')
    warm = retries[2:]  # Third warm-up is the declared baseline, not a cold total.
    for row in warm:
        for key in WARM_FIELDS:
            if type(row.get(key)) is not int or row[key] < 0:
                raise RuntimeError('missing/invalid warm owner counter: ' + key)
        if row['scratch_pool_retained_bytes'] > row['backend_allocated_bytes']:
            raise RuntimeError('free pool exceeds backend allocation')
        if row['backend_graph_count'] == 0:
            raise RuntimeError('warm graph family was not exercised')
    baseline = {key: warm[0][key] for key in WARM_FIELDS}
    for index, row in enumerate(warm[1:], 1):
        for key, value in baseline.items():
            if row[key] != value:
                raise RuntimeError(f'warm owner counter changed at additional request {index}: {key}')
    return {'baseline_after_successful_request': 3,
            'additional_same_shape_requests': 3, 'baseline': baseline,
            'scope': 'backend-tracked idle counters; complemented by separate weak-owner teardown tests'}
