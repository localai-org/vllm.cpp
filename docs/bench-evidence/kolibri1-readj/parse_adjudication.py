#!/usr/bin/env python3
"""Derive the adjudication tables (off.json / on.json) from a full-gate log.

Input: a `test_kolibri1_tt_b2ii --test-case='*device token gate*'` log with
VT_KOLIBRI1_TT_STAGE_DUMP=1 (the readjudication run's arm-on3.log /
arm-off.log class — full stage dump, both passes of the near-tie
instrument). Output: <arm>.json with the summary counters, the argmax-chain
counters, and one record per HARD teacher-forced flip carrying the
route_scores probe gaps at that forward.

STEP LABELS: the gate's `HARD flip ... step d` and the flip records use
d = walk_step - input_len, which IS the decode position (the logits at
that forward predict generated_ids[d]); no offset applies here. Only the
SCRATCH dbg TF prompt legs print the pre-increment raw `TFstep` label —
see parse_tf_origin.py for the +1 mapping.

The committed OFF.json / ON.json in this directory were produced by:

    python3 parse_adjudication.py /tmp/readj/arm-off.log OFF
    python3 parse_adjudication.py /tmp/readj/arm-on3.log ON

(the raw arm logs are 95.7 MB / 97.7 MB and are not committed).

Usage: python3 parse_adjudication.py <log> <arm> [repo/root] [outdir]
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = sys.argv[3] if len(sys.argv) > 3 else os.path.join(HERE, "..", "..", "..")


def forwards(path):
    fwd, cur = [], None
    dh2re = re.compile(r'\[STAGE\]\[TT\] L(\d+) dh2 sum=(\S+) sq=(\S+) hash=(\w+) max=(\S+)')
    rsre = re.compile(r'\[STAGE\]\[TT\] L(\d+) route_scores tok=(\d+) top8=\[([^\]]*)\]')
    with open(path, errors='replace') as f:
        for ln, line in enumerate(f):
            if '[STAGE][TT] L0 dh2 ' in line:
                cur = {'line': ln, 'dh2': {}, 'scores': {}}
                fwd.append(cur)
            if cur is None:
                continue
            m = dh2re.search(line)
            if m:
                cur['dh2'][int(m.group(1))] = (m.group(2), m.group(3), m.group(4), m.group(5))
            m = rsre.search(line)
            if m:
                cur['scores'][int(m.group(1))] = [float(x) for x in m.group(3).split(',')]
    return fwd


def load_prompts():
    d = json.load(open(os.path.join(
        ROOT, 'tests', 'vllm', 'models', 'kolibri1_goldens.json')))
    return [(p['prompt'], len(p['input_ids']), len(p['generated_ids']))
            for p in d['prompts']]


def parse_results(path):
    flips, summ, chain = [], None, None
    for line in open(path, errors='replace'):
        m = re.search(r'NEAR-TIE INSTRUMENT: (\d+) free-running divergences, '
                      r'(\d+) teacher-forced flips \((\d+) near-ties, '
                      r'(\d+) HARD\), worst teacher-forced gap (\S+) nats', line)
        if m:
            summ = tuple(int(m.group(i)) for i in (1, 2, 3, 4)) + (float(m.group(5)),)
        m = re.search(r"REAL hard flip: '(.+)' step (\d+): golden (\d+) "
                      r"tf-argmax (\d+), gap (\S+) nats, in-top5=(\w+)", line)
        if m:
            flips.append({'prompt': m.group(1), 'step': int(m.group(2)),
                          'golden': int(m.group(3)), 'tfarg': int(m.group(4)),
                          'gap': float(m.group(5)), 'intop5': m.group(6)})
        m = re.search(r'ARGMAX CHAIN: (\d+)/(\d+) compared positions match.*'
                      r'flips (\d+) \(near-ties (\d+), hard (\d+)\)', line)
        if m:
            chain = tuple(int(m.group(i)) for i in (1, 2, 3, 4, 5))
    return flips, summ, chain


def main(path, arm):
    ps = load_prompts()
    fwd = forwards(path)
    # doctest buffers MESSAGEs, so line position is useless; the empirical
    # structure: every forward dumps 50 dh2 lines in the GATE phase (prefill
    # + decode) and in the INSTRUMENT decode phases only. total =
    # (74 + compared) + 2*(74+256) - 2*74.
    flips, summ, chain = parse_results(path)
    gate_forwards = 74 + (chain[1] if chain else 0)
    if len(fwd) != gate_forwards + 8 * 64:
        print(f'WARN forward count {len(fwd)} != {gate_forwards + 8*64}; '
              f'router-gap table unreliable')
    print('SUMMARY divergences,tfflips,near-ties,hard,worst =', summ)
    print('CHAIN matched/compared, flips, near-ties, hard =', chain)
    out = []
    for fl in flips:
        pi = next(i for i, p in enumerate(ps) if p[0] == fl['prompt'])
        off = sum((q[1] + q[2]) * 2 for q in ps[:pi])
        fi = gate_forwards + pi * 64 + fl['step']
        if fi >= len(fwd):
            print('MISSING forward for', fl)
            continue
        f = fwd[fi]
        gaps12, gaps67 = [], []
        for Lay in sorted(f['scores']):
            s = f['scores'][Lay]
            if len(s) >= 7:
                gaps12.append(s[0] - s[1])
                gaps67.append(s[5] - s[6])
        rec = dict(fl)
        rec['layer_top12_min'] = min(gaps12)
        rec['layer_top12_med'] = sorted(gaps12)[len(gaps12) // 2]
        rec['layer_top67_min'] = min(gaps67)
        rec['layer_top67_med'] = sorted(gaps67)[len(gaps67) // 2]
        rec['n_layers_gap67_below_0p05'] = sum(1 for g in gaps67 if g < 0.05)
        rec['n_layers_gap12_below_0p05'] = sum(1 for g in gaps12 if g < 0.05)
        out.append(rec)
    for r in out:
        print(json.dumps(r))
    dest = sys.argv[4] if len(sys.argv) > 4 else HERE
    json.dump({'arm': arm, 'summary': summ, 'chain': chain, 'hard_flips': out},
              open(os.path.join(dest, f'{arm}.json'), 'w'), indent=1)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2])
