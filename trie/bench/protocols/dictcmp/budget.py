#!/usr/bin/env python3
"""Pilot analysis of the dictionary comparison campaign (docs/dictionary_comparison_experiment_plan_2026-10-23.md,
sections 6, 7.1 and 9.3). Reads one dataset's pilot launches (schedule.py pilot) and writes

  L.json      the frozen distinct-batch count per point mix and per range k: the smallest L that gives the fastest
              eligible arm/mode at least 200 ms per round (ceiling 1,000), the same L for every arm and mode of that
              mix or k (keys "point:<mix>:<mode>" and "range:<k>:<mode>")
  budget.csv  a machine-time estimate per scheduled cell of the measure phase and in total, from the pilot's costs:
              launch overhead (input load, split, materialization, domain check, process start/exit), builds, the
              operations of two warm-up and four measured rounds, teardown; times --procs processes; plus the plan's
              25 % allowance and a retry envelope (up to three attempts)

usage: budget.py <pilot-dir> <out-dir> --dataset <id> [--procs 3] [--impls a,b,...] [--tag t] [--qproto c2trace-v1]
                 [--ledger <measure ledger.tsv>]
--qproto (plan section 17): the point/range facts come from the PQT/PRT pilots only (the revision-11 PQ/PR pilots are
ignored) and the query L is frozen in <out-dir>/L_trace.json instead of L.json; query rows are QT/RT cells.
--ledger: measure cells already ACCEPTED there are left out, so the estimate is the REMAINING work of the visit.
--impls restricts the analysis to one tier's arms (the pilot directory holds every tier's pilot launches); the budget
goes to <out-dir>/budget_<tag>.csv. L.json is FROZEN: it is written only when <out-dir>/L.json does not exist yet (the
first tier's pilot), and later tiers reuse it unchanged (plan section 15); the decision is logged either way.
"""
import argparse
import collections
import json
import math
import os
import re

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
ROUNDS = 6            # two discarded + four measured
TARGET_NS = 200e6     # 200 ms per round for the fastest eligible mode
L_MAX = 1000
PCTS = ['0.001', '0.01', '0.1', '1', '10']


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def wall_seconds(text):
    m = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)', text)
    if not m:
        return None
    parts = [float(x) for x in m.group(1).split(':')]
    s = 0.0
    for x in parts:
        s = s * 60 + x
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('pilot')
    ap.add_argument('out')
    ap.add_argument('--dataset', required=True)
    ap.add_argument('--procs', type=int, default=3)
    ap.add_argument('--impls', default='')
    ap.add_argument('--tag', default='all')
    ap.add_argument('--qproto', default='')
    ap.add_argument('--ledger', default='')
    a = ap.parse_args()
    accepted = set()
    if a.ledger and os.path.exists(a.ledger):
        last = {}
        for line in open(a.ledger):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 8 and f[0] != 'name':
                last[f[0]] = f[7]
        accepted = {n for n, v in last.items() if v == 'ACCEPTED'}
    only = set(x for x in a.impls.split(',') if x)
    raw = os.path.join(a.pilot, 'raw')
    cost = collections.defaultdict(dict)       # impl -> facts
    point_batch = collections.defaultdict(list)   # mix -> [ns per batch of every arm/mode]
    range_round = collections.defaultdict(list)   # k -> [ns per round]
    point_ns = {}                               # (impl, mix, mode) -> ns per batch
    range_ns = {}                               # (impl, k, mode) -> ns per round
    for fn in sorted(os.listdir(raw)):
        if not fn.endswith('.log'):
            continue
        name = fn[:-4]
        kind, ds, impl = name.split('__')[:3]
        if only and impl not in only:
            continue
        if kind in (('PQ', 'PR') if a.qproto else ('PQT', 'PRT')):
            continue   # query pilots of the other protocol
        text = open(os.path.join(raw, fn), errors='replace').read()
        c = cost[impl]
        wall = wall_seconds(text)
        ops_ns = 0.0
        for line in text.splitlines():
            tag = line.split(' ', 1)[0]
            d = kv(line)
            if tag == 'LOAD':
                c['load_s'] = float(d['total_ms']) / 1e3
            elif tag == 'SPLIT':
                c['split_s'] = float(d['split_ms']) / 1e3
            elif tag == 'MATERIALIZE':
                c.setdefault('materialize_s', 0.0)
                c['materialize_s'] = max(c['materialize_s'], float(d['ms']) / 1e3)
            elif tag == 'BUILD_ROUND':
                c[f'build_{d["contract"]}_s'] = float(d['ns']) / 1e9
                c['teardown_s'] = float(d['teardown_ns']) / 1e9
                ops_ns += float(d['ns']) + float(d['teardown_ns'])
            elif tag == 'INSERT_BUILD':
                c['insert_build_s'] = float(d['build_ns']) / 1e9
                ops_ns += float(d['build_ns'])
            elif tag == 'INSERT_ROUND':
                pct = name.split('__pct')[1]
                c[f'insert_ops_{pct}_s'] = float(d['ns']) / 1e9
                c['teardown_s'] = float(d['teardown_ns']) / 1e9
                ops_ns += float(d['ns']) + float(d['teardown_ns'])
            elif tag == 'QUERY_BUILD':
                c['query_build_s'] = float(d['build_ns']) / 1e9
                ops_ns += float(d['build_ns'])
            elif tag == 'POINT_ROUND' and d.get('measured') == '1':
                ns = float(d['ns'])
                point_ns[(impl, d['mix'], d['mode'])] = ns
                point_batch[d['mix']].append(ns)
                ops_ns += ns
            elif tag == 'RANGE_ROUND' and d.get('measured') == '1':
                ns = float(d['ns'])
                range_ns[(impl, d['k'], d['mode'])] = ns
                range_round[d['k']].append(ns)
                ops_ns += ns
        if wall is not None:
            c.setdefault('overhead_s', [])
            c['overhead_s'].append(max(0.0, wall - ops_ns / 1e9))
    L = {}
    for mix, v in point_batch.items():
        fastest = min(v)
        n = max(1, min(L_MAX, math.ceil(TARGET_NS / fastest))) if fastest > 0 else 1
        for mode in ('direct', 'shared'):
            L[f'point:{mix}:{mode}'] = n
    for k, v in range_round.items():
        fastest = min(v)
        n = max(1, min(L_MAX, math.ceil(TARGET_NS / fastest))) if fastest > 0 else 1
        for mode in ('direct', 'shared'):
            L[f'range:{k}:{mode}'] = n
    os.makedirs(a.out, exist_ok=True)
    lpath = os.path.join(a.out, 'L_trace.json' if a.qproto else 'L.json')
    if os.path.exists(lpath):
        computed = L
        L = json.load(open(lpath))
        print(f'L_FROZEN_REUSED {lpath} (this pilot would give {json.dumps(computed, sort_keys=True)})')
    elif not L:
        print(f'L_NOT_WRITTEN {lpath}: this pilot has no point or range rounds (nothing is frozen from an empty pilot)')
    else:
        json.dump(L, open(lpath, 'w'), indent=1, sort_keys=True)
        print(f'L_FROZEN_WRITTEN {lpath} from impls={sorted(only) if only else "all"}')
    rows = []
    QQ, QG = ('QT', 'RT') if a.qproto else ('Q', 'R')

    def remaining(name):   # processes of the cell not yet accepted in the measure ledger
        return sum(1 for p in range(1, a.procs + 1) if f'{name}__p{p}' not in accepted)

    for impl, c in sorted(cost.items()):
        oh = max(c.get('overhead_s', [0.0]))
        if 'build_build_only_s' in c:
            for contract in ('build_only', 'build_end_to_end'):
                b = c.get(f'build_{contract}_s', c['build_build_only_s'])
                rows.append((impl, f'build:{contract}', oh + ROUNDS * (b + c.get('teardown_s', 0.0)),
                             remaining(f'B__{a.dataset}__{impl}__{contract}')))
        if 'insert_ops_1_s' in c:
            for pct in PCTS:
                ops = c.get(f'insert_ops_{pct}_s', c['insert_ops_1_s'] if pct != '10' else c.get('insert_ops_10_s', 0.0))
                rows.append((impl, f'insert:{pct}', oh + ROUNDS * (c.get('insert_build_s', 0.0) + ops + c.get('teardown_s', 0.0)),
                             remaining(f'I__{a.dataset}__{impl}__pct{pct}')))
        pts = [ns * L.get(f'point:{mix}:{mode}', 1) for (i, mix, mode), ns in point_ns.items() if i == impl]
        if pts:
            rows.append((impl, 'point', oh + c.get('query_build_s', 0.0) + ROUNDS * sum(pts) / 1e9, remaining(f'{QQ}__{a.dataset}__{impl}')))
        rgs = [ns * L.get(f'range:{k}:{mode}', 1) for (i, k, mode), ns in range_ns.items() if i == impl]
        if rgs:
            rows.append((impl, 'range', oh + c.get('query_build_s', 0.0) + ROUNDS * sum(rgs) / 1e9, remaining(f'{QG}__{a.dataset}__{impl}')))
    total = sum(r[2] * r[3] for r in rows)
    with open(os.path.join(a.out, f'budget_{a.tag}.csv'), 'w') as f:
        f.write('dataset,impl,cell,seconds_per_process,processes,seconds_total\n')
        for impl, cellname, s, procs in rows:   # processes = those not yet accepted (all of them without --ledger)
            f.write(f'{a.dataset},{impl},{cellname},{s:.1f},{procs},{s * procs:.1f}\n')
        f.write(f'{a.dataset},ALL,no_interference_total,,{a.procs},{total:.1f}\n')
        f.write(f'{a.dataset},ALL,with_25pct_allowance,,{a.procs},{total * 1.25:.1f}\n')
        f.write(f'{a.dataset},ALL,retry_envelope_3_attempts,,{a.procs},{total * 3:.1f}\n')
    print(json.dumps(L, sort_keys=True))
    print(f'{a.dataset}: remaining measure phase {total / 3600:.2f} h without interference, {total * 1.25 / 3600:.2f} h with 25% '
          f'allowance (query protocol {a.qproto or "r11"}; {len(accepted)} accepted measure cells excluded)')


if __name__ == '__main__':
    main()
