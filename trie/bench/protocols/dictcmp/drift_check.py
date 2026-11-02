#!/usr/bin/env python3
"""Drift screening of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md,
section 9.2). Compares the control processes of one dataset start with the pre-campaign baseline (three accepted
control processes). Metrics: the control's mixed direct point query (ns per query) and its 1 % insertion (ns per key),
each a process mean over the four measured rounds. A metric is flagged when the dataset start's process mean (the
median of its accepted processes) differs from the baseline median by more than max(5 %, 3 * relative MAD of the
baseline process means). Two-mode insertion check (user decision, October 26; progress record 16.8): the control's
insertion is bimodal per process (about 147 and about 129 ns per key; the baseline itself holds one process of each
mode), so an insertion value outside the tolerance of the baseline median also passes when it is within the same
tolerance of a baseline process mean that is itself outside it (the baseline's second mode); a value slower than the
slow mode or between the modes is still flagged. The point check is unchanged. A screening rule, not a significance
test; no result is normalized by it.
usage: drift_check.py <baseline-control-dir> <control-dir>    prints DRIFT_OK or DRIFT_FLAG lines; exit 0 / 3
"""
import os
import re
import statistics
import sys

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')


def process_means(d):
    out = {'point': [], 'insert': []}
    raw = os.path.join(d, 'raw')
    ledger = os.path.join(d, 'ledger.tsv')
    accepted = set()
    if os.path.exists(ledger):
        for line in open(ledger):
            f = line.rstrip('\n').split('\t')
            if len(f) > 7 and f[7] == 'ACCEPTED':
                accepted.add(f[0])
    for name in sorted(accepted):
        pts, ins = [], []
        for line in open(os.path.join(raw, name + '.log'), errors='replace'):
            tag = line.split(' ', 1)[0]
            kv = dict(KV.findall(line))
            if tag == 'POINT_ROUND' and kv.get('measured') == '1':
                pts.append(float(kv['ns_per_query']))
            elif tag == 'INSERT_ROUND' and kv.get('measured') == '1':
                ins.append(float(kv['ns_per_key']))
        if pts:
            out['point'].append(statistics.mean(pts))
        if ins:
            out['insert'].append(statistics.mean(ins))
    return out


def main():
    base, cur = process_means(sys.argv[1]), process_means(sys.argv[2])
    flagged = False
    for metric in ('point', 'insert'):
        b, c = base[metric], cur[metric]
        if len(b) < 3 or not c:
            print(f'DRIFT_INCOMPLETE metric={metric} baseline_processes={len(b)} current_processes={len(c)}')
            flagged = True
            continue
        med = statistics.median(b)
        mad = statistics.median([abs(x - med) for x in b])
        tol = max(0.05, 3 * mad / med)
        x = statistics.median(c)
        dev = abs(x - med) / med
        mode = 'primary'
        if dev > tol and metric == 'insert':
            second = [v for v in b if abs(v - med) / med > tol]
            if second:
                ref = min(second, key=lambda v: abs(x - v))
                if abs(x - ref) / ref <= tol:
                    dev, mode = abs(x - ref) / ref, f'second mode_ref={ref:.3f}'
        tag = 'DRIFT_FLAG' if dev > tol else 'DRIFT_OK'
        flagged |= dev > tol
        print(f'{tag} metric={metric} baseline_median={med:.3f} baseline_rel_mad={mad / med:.4f} current={x:.3f} '
              f'deviation={dev:.4f} tolerance={tol:.4f} mode={mode} current_processes={len(c)}')
    sys.exit(3 if flagged else 0)


if __name__ == '__main__':
    main()
