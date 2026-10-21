"""Detecting replay of the actual boundary defect plus independent corruption controls."""
from pathlib import Path
import copy,json
from qualify_counter_windows import qualify,qualify_worker
D=Path(__file__).resolve().parent;p=D/'native-matrix/concurrency-4k-c4-r1'
a=json.loads((p/'summary.json').read_text());s=[json.loads(x) for x in (p/'metrics.jsonl').read_text().splitlines()]
assert not a['fully_overlapped_counter_window_valid'] and 'prefill_inside_window' in a['fully_overlapped_counter_window_issues']
final_prefill=1356549.0
b,receipt=qualify(a,s,final_prefill)
assert receipt['trimmed_leading_samples']==1 and b['fully_overlapped_counter_window_valid']
assert b['fully_overlapped_draft_tokens']==3972 and b['fully_overlapped_accepted_tokens']==2134 and b['fully_overlapped_generation_tokens']==3458
assert b['fully_overlapped_token_accounting_residual']==0
# Removing a leading prefill sample must never repair arbitrary draft accounting.
for label,mutate in [('unmatched_draft',lambda rows:rows[-1].__setitem__('draft_tokens',rows[-1]['draft_tokens']+3)),('counter_reset',lambda rows:rows[-1].__setitem__('prefill_computed',0))]:
 rows=copy.deepcopy([x for x in s if a['fully_overlapped_sample_start_s']<=x['elapsed_s']<=a['fully_overlapped_sample_end_s']]);mutate(rows)
 try:qualify(a,rows,final_prefill)
 except ValueError:pass
 else:raise AssertionError('corruption was accepted: '+label)
try:qualify(a,s,final_prefill+1)
except ValueError:pass
else:raise AssertionError('incomplete committed-prefill plateau accepted')
raw=json.loads((D/'native-matrix/results.json').read_text());records=qualify_worker(raw,D/'native-matrix')
assert len(records)==70 and sum(r['trimmed_leading_samples'] for r in records)==1
assert sum(not r['original_counter_window_valid'] for r in records)==1
assert all(r['qualified_counter_window_valid'] for r in records)
assert all(r['original_throughput']==r['qualified_throughput'] for r in records if not r['trimmed_leading_samples'])
(D/'counter-window-native-qualification.json').write_text(json.dumps({'status':'PASS_NATIVE_70_WINDOWS_UNCHANGED_GATE_WITH_DISCLOSED_SINGLE_BOUNDARY_CORRECTION','detecting_controls':['actual_bad_boundary_detected','actual_3458_equals2134_plus3972_over3','unmatched_draft_rejected','counter_reset_rejected','incomplete_prefill_plateau_rejected','other69_windows_unchanged'],'records':records},indent=2)+'\n')
print(json.dumps({'status':'PASS_ACTUAL_BOUNDARY_REPLAY_AND_NEGATIVE_CONTROLS','corrected':receipt,'all_native_windows':70,'other_windows_unchanged':69},indent=2))
