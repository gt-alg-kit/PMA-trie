#!/usr/bin/env python3
"""Point-query batch-size supplement of the dictionary comparison campaign (experiment point-batch-sweep-v1;
docs/dictionary_comparison_experiment_plan_2026-10-23.md, section 18). Query input semantics stay c2trace-v1: one
fixed owned-string trace per cell, generator round 0, replayed in every round, process, arm and mode.

Grid: the insertion panel's five integer batch sizes, bq = max(1, N // d) for d = 100000, 10000, 1000, 100, 10
(0.001 %, 0.01 %, 0.1 %, 1 %, 10 % of the full population N), queried against the immutable initial S-key index.
Stages: `mixed` (the mixed mix), then `hitmiss` (hit and miss). One measure process per (arm, stage, repetition) builds
the index once and runs every (mix, size) cell of its stage, each with its own two warm-up and four measured rounds;
the (mix, size) order is rotated by the repetition index, identically for every arm, and DFUDS/DT run direct then shared
at each size (the shared mode reuses the trace that direct materialized; one size/mix is held in memory at a time).

subcommands
  sizes     --dataset D --manifest <canonical manifest.json> --main <main-campaign results/D>
            prints the size table (label, divisor, N, S, bq, insertion b) and exits 1 unless every bq equals the
            insertion panel's batch size b recorded by the main campaign (INSERT_PLAN of its DFUDS insertion cells)
  schedule  <pilot|measure|control> --dataset D --stage mixed|hitmiss --bin B --launch L [--procs 3] [--L L.json]
            [--impls a,b] [--na impl=STATUS,...] [--control-keys K] [--tag T] [--salt s]
            one TSV line per process (run_cells.sh format: name, kind, salt, command); NA rows for --na arms; --tag names
            the visit's arm group in its control cells (a fresh control identity per appended group)
  freeze    <pilot-dir> <dataset-dir> --dataset D --stage S [--procs 3] [--ledger <measure ledger>] [--impls a,b]
            [--tag T] [--from-stage mixed]
            from the pilot (L = 1, one warm-up, one measured round): the common L per (mix, size), the smallest L that
            gives the fastest arm/mode at least 200 ms per round, at most 1,000 (capped short rounds are kept and
            reported), identical for every arm and mode; written once to L_<stage>.json (an existing file is reused
            and the recomputed values are only logged); and budget_<stage>.csv: per process load/build/overhead plus
            trace preparation and 2 + 4 rounds of every cell, times the remaining processes, the 25 % allowance and
            the three-attempt retry envelope. --impls restricts the analysis to those arms' pilots (the pilot directory
            holds every group's pilots), --tag writes budget_<stage>_<tag>.csv; --from-stage mixed budgets a hitmiss visit
            from the mixed pilot (each mixed cell time stands for both hit and miss at that size; the frozen L_hitmiss.json
            gives the batch counts), so an appended arm needs one sequential-build pilot per dataset; L is never written
            from such a pilot
Stage all (user decision October 26 12:21, query-only arms MARISA / PDT / PDT-Dyn): one process per (arm, repetition)
builds once and runs all 15 (mix, size) cells (rotated by the repetition index) with the frozen L of both L_mixed.json and
L_hitmiss.json (--L a,b); no pilot; process 1 also prints the space census after its last timed cell (--space-after-p1).
Arms (plan revision 15, section 19): phase five-arm-v1 = DFUDS, DT, ART-OLC, HOT-ROWEX, FST; phase eight-arm-v2 adds
MARISA, PDT and PDT-Dyn (direct only, 64 readers). Appended arms reuse the frozen sizes, L and traces.
Cell names: PSW__<D>__<arm>__<stage> (pilot), SW__<D>__<arm>__<stage>__p<k> (measure),
K__<D>__sw_<stage>__n<keys>__p<k> (drift control).
"""
import argparse
import collections
import json
import math
import os
import re
import sys

EXPERIMENT = 'point-batch-sweep-v1'
REQUIRED = ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-upstream']          # phase five-arm-v1
ARMS = REQUIRED + ['marisa', 'pdt-static', 'dynpdt-compact']                       # phase eight-arm-v2 (revision 15)
ARMS += ['fst-64']   # phase fst64-v3 (October 27): the patched FST, stage all, its own binary (docs/fst64_patch_2026-10-27.md)
SHARED = {'dfuds', 'dfuds-dt'}
SIZES = [('0.001', 100000), ('0.01', 10000), ('0.1', 1000), ('1', 100), ('10', 10)]
STAGES = {'mixed': ['mixed'], 'hitmiss': ['hit', 'miss'], 'all': ['mixed', 'hit', 'miss']}
ROUNDS = 6            # two discarded + four measured
TARGET_NS = 200e6     # 200 ms per round for the fastest arm/mode
L_MAX = 1000
KV = re.compile(r'(\w+)=("[^"]*"|\S+)')


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def rotate(xs, r):
    r %= len(xs)
    return xs[r:] + xs[:r]


def bq_of(n, d):
    return max(1, n // d)


def pairs(stage):
    return [(m, p) for m in STAGES[stage] for p, _ in SIZES]


def modes(impl):
    return ['direct', 'shared'] if impl in SHARED else ['direct']


def cells(impl, stage, r, L):
    """The --cells list of one process: rotated (mix, size) pairs, direct then shared at each size."""
    out = []
    for m, p in rotate(pairs(stage), r):
        for mode in modes(impl):
            out.append(f'{m}:{mode}:{L.get(f"point:{m}:pct{p}", 1) if L is not None else 1}:pct{p}')
    return ','.join(out)


def wall_seconds(text):
    m = re.search(r'Elapsed \(wall clock\) time \(h:mm:ss or m:ss\): (\S+)', text)
    if not m:
        return None
    s = 0.0
    for x in m.group(1).split(':'):
        s = s * 60 + float(x)
    return s


def cmd_sizes(a):
    man = json.load(open(a.manifest))
    n = int(man['N'])
    s = 7 * n // 10
    bad = 0
    print('label\tdivisor\tN\tS\tbq\tinsert_b\tmatch')
    for p, d in SIZES:
        b = None
        log = os.path.join(a.main, 'measure', 'raw', f'I__{a.dataset}__dfuds__pct{p}__p1.log')
        if os.path.exists(log):
            for line in open(log, errors='replace'):
                if line.startswith('INSERT_PLAN '):
                    b = int(kv(line)['b'])
        q = bq_of(n, d)
        ok = b == q
        bad += not ok
        print(f'pct{p}\t{d}\t{n}\t{s}\t{q}\t{b if b is not None else "missing"}\t{"OK" if ok else "MISMATCH"}')
    sys.exit(1 if bad else 0)


def cmd_schedule(a):
    impls = [x for x in a.impls.split(',') if x]
    for x in impls:
        if x not in ARMS:
            sys.exit(f'unknown or out-of-scope impl {x}')
    na = dict(x.split('=', 1) for x in a.na.split(',') if x)
    L = None
    for f in [x for x in a.L.split(',') if x]:   # several frozen L files are merged (stage all: mixed + hitmiss)
        L = {**(L or {}), **json.load(open(f))}
    if a.phase == 'measure' and L is None:
        sys.exit('measure needs the frozen --L')
    rows = []
    ds = a.dataset

    def cell(name, impl, args):
        if impl in na:
            rows.append((name, 'NA', a.salt, na[impl]))
        else:
            rows.append((name, 'POINT', a.salt, f'{a.launch} {a.bin} {ds} --impl {impl} --mode point --experiment {EXPERIMENT} {args}'))

    if a.phase == 'control':
        if not a.control_keys:
            sys.exit('--control-keys is required for the control phase')
        for p in range(1, a.procs + 1):
            rows.append((f'K__{ds}__sw_{a.stage}{"_" + a.tag if a.tag else ""}__n{a.control_keys}__p{p}', 'CONTROL', a.salt,
                         f'{a.launch} {a.bin} control --mode control --control-keys {a.control_keys}'))
    elif a.phase == 'pilot':
        for impl in impls:
            cell(f'PSW__{ds}__{impl}__{a.stage}', impl, f'--cells {cells(impl, a.stage, 0, None)} --warm-rounds 1 --rounds 1')
    else:
        for impl in impls:
            if impl in na:
                rows.append((f'SW__{ds}__{impl}__{a.stage}', 'NA', a.salt, na[impl]))
        for p in range(1, a.procs + 1):
            for impl in rotate(impls, p - 1):
                if impl not in na:
                    extra = ' --space-after' if a.space_after_p1 and p == 1 else ''
                    cell(f'SW__{ds}__{impl}__{a.stage}__p{p}', impl, f'--cells {cells(impl, a.stage, p - 1, L)}{extra}')
    for r in rows:
        print('\t'.join(r))


def cmd_freeze(a):
    raw = os.path.join(a.pilot, 'raw')
    accepted = set()
    if a.ledger and os.path.exists(a.ledger):
        last = {}
        for line in open(a.ledger):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 8 and f[0] != 'name':
                last[f[0]] = f[7]
        accepted = {n for n, v in last.items() if v == 'ACCEPTED'}
    batch_ns = collections.defaultdict(list)       # (mix, label) -> per-batch ns of every arm/mode (L = 1)
    per = collections.defaultdict(dict)            # impl -> {(mix, label, mode): ns per batch}
    prep = collections.defaultdict(dict)           # impl -> {(mix, label): trace preparation ms at L = 1}
    fixed = {}                                     # impl -> load/split/materialize/build/overhead seconds per process
    only = set(x for x in a.impls.split(',') if x)
    src_stage = a.from_stage or a.stage
    for fn in sorted(os.listdir(raw)):
        if not (fn.startswith('PSW__') and fn.endswith('.log')):
            continue
        impl = fn[:-4].split('__')[2]
        if only and impl not in only:
            continue
        text = open(os.path.join(raw, fn), errors='replace').read()
        if 'CELL_END status=OK' not in text:
            continue
        lab = cur = None
        for line in text.splitlines():
            tag = line.split(' ', 1)[0]
            if tag == 'QUERY_TRACE':
                cur = float(kv(line)['prep_ms'])
            elif tag == 'POINT_CELL':
                d = kv(line)
                if 'status' in d:
                    continue
                lab = d['bq_label']
                if cur is not None:   # the cell that materialized this (mix, size) trace; the shared cell reuses it
                    prep[impl][(d['mix'], lab)] = cur
                    cur = None
            elif tag == 'POINT_ROUND' and ' measured=1 ' in line:
                d = kv(line)
                ns = float(d['ns'])
                per[impl][(d['mix'], lab, d['mode'])] = ns
                batch_ns[(d['mix'], lab)].append(ns)
        rounds_ns = sum(per[impl].values()) * 2   # pilot: one warm-up and one measured round per cell
        w = wall_seconds(text)
        fixed[impl] = (w if w is not None else 0.0) - rounds_ns / 1e9 - sum(prep[impl].values()) / 1e3
    L = {}
    for (m, lab), v in sorted(batch_ns.items()):
        f = min(v)
        L[f'point:{m}:{lab}'] = max(1, min(L_MAX, math.ceil(TARGET_NS / f))) if f > 0 else 1
    lpath = os.path.join(a.out, f'L_{a.stage}.json')
    if src_stage != a.stage and not os.path.exists(lpath):
        sys.exit(f'--from-stage {src_stage}: {lpath} must already be frozen')
    if os.path.exists(lpath):
        computed = L
        L = json.load(open(lpath))
        print(f'L_FROZEN_REUSED {lpath} (this pilot would give {json.dumps(computed, sort_keys=True)})')
    elif not L:
        print(f'L_NOT_WRITTEN {lpath}: no accepted pilot rounds')
        sys.exit(1)
    else:
        json.dump(L, open(lpath, 'w'), indent=1, sort_keys=True)
        print(f'L_FROZEN_WRITTEN {lpath}')
    capped = []
    for key, n in sorted(L.items()):
        m, lab = key.split(':')[1:]
        src = (m if src_stage == a.stage else src_stage, lab)
        if src not in batch_ns:
            continue
        f = min(batch_ns[src])
        if only and f * n < TARGET_NS:   # an appended arm faster than the frozen L's target: reported, workload unchanged
            capped.append(f'{key} arms={",".join(sorted(only))} (their fastest round {f * n / 1e6:.1f} ms at the frozen L={n})')
        elif not only and n == L_MAX and f * n < TARGET_NS:
            capped.append(f'{key} (fastest round {f * n / 1e6:.1f} ms at the cap)')
    for c in capped:
        print(f'L_{"SHORT" if only else "CAPPED_SHORT"}_ROUND {c}')
    rows = []
    targets = (lambda m: STAGES[a.stage]) if src_stage != a.stage else (lambda m: [m])
    for impl in sorted(per):
        s = max(0.0, fixed[impl])
        for (m, lab, mode), ns in per[impl].items():
            for t in targets(m):
                s += ROUNDS * L.get(f'point:{t}:{lab}', 1) * ns / 1e9
        for (m, lab), ms in prep[impl].items():
            for t in targets(m):
                s += ms / 1e3 * L.get(f'point:{t}:{lab}', 1)
        left = sum(1 for p in range(1, a.procs + 1) if f'SW__{a.dataset}__{impl}__{a.stage}__p{p}' not in accepted)
        rows.append((impl, s, left))
    total = sum(s * n for _, s, n in rows)
    with open(os.path.join(a.out, f'budget_{a.stage}{"_" + a.tag if a.tag else ""}.csv'), 'w') as f:
        f.write('dataset,stage,impl,seconds_per_process,processes,seconds_total\n')
        for impl, s, n in rows:
            f.write(f'{a.dataset},{a.stage},{impl},{s:.1f},{n},{s * n:.1f}\n')
        f.write(f'{a.dataset},{a.stage},ALL,no_interference_total,,{total:.1f}\n')
        f.write(f'{a.dataset},{a.stage},ALL,with_25pct_allowance,,{total * 1.25:.1f}\n')
        f.write(f'{a.dataset},{a.stage},ALL,retry_envelope_3_attempts,,{total * 3:.1f}\n')
    print(json.dumps(L, sort_keys=True))
    print(f'BUDGET {a.dataset} stage={a.stage} group={a.tag or "-"} pilot_stage={src_stage} remaining_s={total:.0f} with_allowance_s={total * 1.25:.0f} '
          f'({total / 3600:.2f} h / {total * 1.25 / 3600:.2f} h; {len(accepted)} accepted cells excluded)')


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest='cmd', required=True)
    s = sub.add_parser('sizes')
    s.add_argument('--dataset', required=True)
    s.add_argument('--manifest', required=True)
    s.add_argument('--main', required=True)
    s = sub.add_parser('schedule')
    s.add_argument('phase', choices=['pilot', 'measure', 'control'])
    s.add_argument('--dataset', required=True)
    s.add_argument('--stage', required=True, choices=sorted(STAGES))
    s.add_argument('--bin', required=True)
    s.add_argument('--launch', required=True)
    s.add_argument('--procs', type=int, default=3)
    s.add_argument('--L', default='')
    s.add_argument('--impls', default=','.join(REQUIRED))
    s.add_argument('--na', default='')
    s.add_argument('--control-keys', type=int, default=0)
    s.add_argument('--tag', default='')
    s.add_argument('--space-after-p1', action='store_true', help='measure: process 1 prints the space census after its cells')
    s.add_argument('--salt', default=EXPERIMENT)
    s = sub.add_parser('freeze')
    s.add_argument('pilot')
    s.add_argument('out')
    s.add_argument('--dataset', required=True)
    s.add_argument('--stage', required=True, choices=sorted(STAGES))
    s.add_argument('--procs', type=int, default=3)
    s.add_argument('--ledger', default='')
    s.add_argument('--impls', default='')
    s.add_argument('--tag', default='')
    s.add_argument('--from-stage', default='', choices=['', 'mixed'])
    a = ap.parse_args()
    {'sizes': cmd_sizes, 'schedule': cmd_schedule, 'freeze': cmd_freeze}[a.cmd](a)


if __name__ == '__main__':
    main()
