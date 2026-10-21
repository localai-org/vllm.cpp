"""Offline final-prefill-plateau qualification; immutable original driver and raw data.

This is a disclosed post-capture boundary correction, not preregistered sampling.
Always apply the same rule to every original engine/block/wave. Never search for
endpoints that improve throughput or make token residuals pass.
"""
from pathlib import Path
import hashlib,importlib.util,json
D=Path(__file__).resolve().parent
SOURCE=Path('/home/sebastian/LocalLLM/intel-b70-qwen38-vllm-onednn/scripts/decode_overlap.py')
contract=json.loads((D/'contract.json').read_text())
assert hashlib.sha256(SOURCE.read_bytes()).hexdigest()==contract['source_sha256'][str(SOURCE)]
spec=importlib.util.spec_from_file_location('frozen_overlap_for_qualification',SOURCE)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

def qualify(case,samples,final_prefill,prefill_key="prefill_computed"):
    start=case['fully_overlapped_sample_start_s'];stop=case['fully_overlapped_sample_end_s']
    concurrency=case['scenario']['concurrency'];speculative=3
    window=[s for s in samples if start<=s['elapsed_s']<=stop]
    if len(window)<2 or any('error' in s for s in window):
        raise ValueError('missing/error samples; no boundary repair permitted')
    if any(b['elapsed_s']<=a['elapsed_s'] or b[prefill_key]<a[prefill_key] for a,b in zip(window,window[1:])):
        raise ValueError('nonmonotonic clock or committed-prefill counter')
    if any(s['running']!=concurrency or s['waiting']!=0 for s in window):
        raise ValueError('occupancy mismatch; no occupancy repair permitted')
    # Earliest sample on the FINAL committed-prefill plateau, never a latency/
    # throughput-dependent cutoff or a scan for a preferred accounting residual.
    if window[-1][prefill_key]!=final_prefill:
        raise ValueError('full committed-prefill end state not reached inside original window')
    cut=next(i for i,s in enumerate(window) if s[prefill_key]==final_prefill)
    chosen=window[cut:]
    original=module.summarize_overlap(samples,start,stop,concurrency,speculative)
    qualified=module.summarize_overlap(samples,chosen[0]['elapsed_s'],stop,concurrency,speculative)
    if not qualified['fully_overlapped_counter_window_valid']:
        raise ValueError('unchanged strict gate still fails: '+repr(qualified['fully_overlapped_counter_window_issues']))
    seconds=chosen[-1]['elapsed_s']-chosen[0]['elapsed_s']
    generated=chosen[-1]['generation_tokens']-chosen[0]['generation_tokens']
    qualified.update(fully_overlapped_decode_sampled_s=seconds,
                     fully_overlapped_generation_tokens=generated,
                     fully_overlapped_aggregate_decode_tokens_per_s=generated/seconds)
    record={'method':'final_committed_prefill_plateau_v1','trimmed_leading_samples':cut,
            'original_sample_count':len(window),'qualified_sample_count':len(chosen),
            'original_counter_window_valid':original['fully_overlapped_counter_window_valid'],
            'original_issues':original['fully_overlapped_counter_window_issues'],
            'original_start_s':start,'qualified_start_s':chosen[0]['elapsed_s'],'end_s':stop,
            'original_seconds':case['fully_overlapped_decode_sampled_s'],'qualified_seconds':seconds,
            'original_generation_tokens':case['fully_overlapped_generation_tokens'],
            'qualified_generation_tokens':generated,
            'original_throughput':case['fully_overlapped_aggregate_decode_tokens_per_s'],
            'qualified_throughput':generated/seconds,
            'qualified_draft_tokens':qualified['fully_overlapped_draft_tokens'],
            'qualified_accepted_tokens':qualified['fully_overlapped_accepted_tokens'],
            'qualified_token_accounting_residual':qualified['fully_overlapped_token_accounting_residual'],
            'qualified_counter_window_valid':True}
    return qualified,record

def qualify_worker(raw,root):
    records=[]
    # Python computed-token histograms publish only at request completion.
    # Its live prompt counter publishes with the first prefill output instead.
    # This field is ONLY a decode-window activity guard, never computed work.
    prefill_key='prefill_computed' if root.name.startswith('native-') else 'prompt_tokens'
    metric=json.loads((root/'command.json').read_text())['metric_binding'][prefill_key]
    for case in raw['cases']:
        path=root/f"{case['scenario']['name']}-r{case['repeat']}"/'metrics.jsonl'
        samples=[json.loads(s) for s in path.read_text().splitlines()]
        after=path.with_name('metrics-after.prom')
        counters=[float(line.split()[1]) for line in after.read_text().splitlines() if line and not line.startswith('#') and line.split()[0].split('{',1)[0]==metric]
        if not counters:raise ValueError('actual final committed-prefill metric missing')
        final_prefill=sum(counters)
        updates,record=qualify(case,samples,final_prefill,prefill_key)
        record.update(prefill_activity_sample_field=prefill_key,final_prefill_cumulative=final_prefill,final_prefill_metric=metric,final_metrics_sha256=hashlib.sha256(after.read_bytes()).hexdigest())
        record.update(worker=root.name,scenario=case['scenario']['name'],repeat=case['repeat'],
                      metrics_path=str(path),metrics_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        records.append(record);case.update(updates)
    for summary in raw['summary']:
        cases=[c for c in raw['cases'] if c['scenario']['name']==summary['name']]
        summary['fully_overlapped_aggregate_decode_tokens_per_s']=sum(c['fully_overlapped_generation_tokens'] for c in cases)/sum(c['fully_overlapped_decode_sampled_s'] for c in cases)
    return records
