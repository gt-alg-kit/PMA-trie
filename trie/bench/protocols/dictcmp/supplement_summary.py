#!/usr/bin/env python3
"""Summary of the single-worker / sorted-baseline supplement (plan section 21; supplement.py has the cells).

usage: supplement_summary.py <supplement root> <out dir> --main <main results root> --sweep <point-sweep root>

Reads the ACCEPTED cells (last ledger verdict) of every <root>/<dataset>/<step>/ and writes into <out dir>:
  insert_cells.csv    one row per accepted insertion process: dataset, step, impl, pct, process, workers, method, b,
                      mean ns/key of the measured rounds (round ns / keys added), sort share of the timed time, mean
                      untimed build seconds of the measured rounds
  insert_summary.csv  per (dataset, variant, impl, pct): processes and the median / min / max of the process means;
                      variant main64 = the main campaign's accepted I__ cells of the same arm and size (64 workers)
  query_cells.csv     one row per accepted SQ process and cell: dataset, impl, process, mix, mode, bq, mean ns/query
  query_summary.csv   per (dataset, impl, mix, bq, mode): processes and median / min / max; mode sweep_direct = the
                      point-query sweep's accepted direct cells of the same arm (separate mixed / hitmiss processes)
  checks.txt          trace identity: every insertion batch ids_hash against the main campaign's cells of the same
                      dataset and size, every SQ batch ids_hash against the sweep's cells of the same arm, mix, bq and
                      batch index; measured-round counts; schedules with unaccepted cells. Last line: problems=<n>
  report.md           the comparison tables
Statistics follow the campaign: the mean of the four measured rounds per process, the median over processes.
"""
import argparse
import collections
import csv
import glob
import os
import re
import statistics
import sys

KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
STEP_VARIANT = {'w1batch': 'batch_w1', 'w1point': 'point_w1', 'sinsert': 'sorted64', 'pdtdyn': 'native_seq'}
PCTS = ['0.001', '0.01', '0.1', '1', '10']


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def accepted(d):
    led = os.path.join(d, 'ledger.tsv')
    last = {}
    if os.path.exists(led):
        for line in open(led):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 8 and f[0] != 'name':
                last[f[0]] = f[7]
    return {n for n, v in last.items() if v == 'ACCEPTED'}, last


def scheduled(d):
    s = os.path.join(d, 'schedule.tsv')
    return [l.split('\t')[0] for l in open(s) if l.strip() and l.split('\t')[1] != 'NA'] if os.path.exists(s) else []


def parse_insert(path):
    r = {'workers': None, 'method': None, 'b': None, 'ids': {}, 'round_npk': [], 'sort': 0.0, 'timed': 0.0, 'build': []}
    for line in open(path, errors='replace'):
        if line.startswith('ENV '):
            r['workers'] = int(kv(line)['workers'])
        elif line.startswith('INSERT_PLAN '):
            x = kv(line); r['method'] = x['method']; r['b'] = int(x['b'])
        elif line.startswith('INSERT_BATCH_IDS '):
            x = kv(line); r['ids'][int(x['batch'])] = x['ids_hash']
        elif line.startswith('INSERT_BUILD ') and ' measured=1 ' in line:
            r['build'].append(float(kv(line)['build_ns']) / 1e9)
        elif line.startswith('INSERT_BATCH ') and ' measured=1 ' in line:
            x = kv(line); r['sort'] += float(x['sort_ns']); r['timed'] += float(x['ns'])
        elif line.startswith('INSERT_ROUND ') and ' measured=1 ' in line:
            x = kv(line); r['round_npk'].append(float(x['ns']) / float(x['added']))
    return r


def parse_shared(path):
    """--share-build process: {pct: {'round_npk': [...], 'ids': {batch: hash}}}, shared build seconds, launch workers"""
    per = collections.defaultdict(lambda: {'round_npk': [], 'ids': {}})
    build, cur, workers = None, None, None
    for line in open(path, errors='replace'):
        if line.startswith('ENV '):
            workers = int(kv(line)['workers'])
        elif line.startswith('INSERT_SIZE '):
            cur = kv(line)['pct']
        elif line.startswith('INSERT_BATCH_IDS ') and cur is not None:
            x = kv(line); per[cur]['ids'][int(x['batch'])] = x['ids_hash']
        elif line.startswith('SHARED_BUILD '):
            build = float(kv(line)['build_ns']) / 1e9
        elif line.startswith('INSERT_ROUND ') and ' measured=1 ' in line:
            x = kv(line); per[x['pct']]['round_npk'].append(float(x['ns']) / float(x['added']))
    return per, build, workers


def parse_point(path):
    """{(mix, mode, bq): [ns/query of each measured round]}, {(mix, bq, batch): ids_hash}"""
    rounds = collections.defaultdict(list)
    hashes = {}
    cur = None
    for line in open(path, errors='replace'):
        if line.startswith('POINT_CELL '):
            x = kv(line)
            cur = None if 'status' in x else (x['mix'], x['mode'], int(x['bq']))
        elif line.startswith('POINT_ROUND ') and ' measured=1 ' in line and cur:
            rounds[cur].append(float(kv(line)['ns_per_query']))
        elif line.startswith('POINT_BATCH '):
            x = kv(line)
            hashes[(x['mix'], int(x['bq']), int(x['batch']))] = x['ids_hash']
    return rounds, hashes


def name_fields(name):
    f = name.split('__')   # <prefix>__<ds>__<impl>[__pct<p>]__p<k>
    pct = next((x[3:] for x in f if x.startswith('pct')), None)
    return f[0], f[1], f[2], pct, int(f[-1][1:])


def med(xs):
    return (statistics.median(xs), min(xs), max(xs)) if xs else (None, None, None)


def fmt(x, nd=1):
    return '-' if x is None else f'{x:.{nd}f}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root')
    ap.add_argument('out')
    ap.add_argument('--main', required=True)
    ap.add_argument('--sweep', required=True)
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    problems = []
    checks = []
    ins_cells = []                                   # dicts
    ins_proc = collections.defaultdict(list)         # (ds, variant, impl, pct) -> process means
    q_cells = []
    q_proc = collections.defaultdict(list)           # (ds, impl, mix, bq, mode) -> process means
    datasets = sorted(d for d in os.listdir(a.root) if os.path.isdir(os.path.join(a.root, d)) and d != os.path.basename(a.out.rstrip('/')))
    main_ids = {}                                    # (ds, pct) -> {batch: ids_hash} from the main campaign's DFUDS cell
    for ds in datasets:
        # the main campaign's 64-worker insertion cells of the four batch arms (reference rows and trace identity)
        mdir = os.path.join(a.main, ds, 'measure')
        macc, _ = accepted(mdir)
        for n in sorted(macc):
            if not n.startswith('I__'):
                continue
            _, _, impl, pct, p = name_fields(n)
            if impl not in ('dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex') or pct is None:
                continue
            r = parse_insert(os.path.join(mdir, 'raw', n + '.log'))
            if len(r['round_npk']) == 4:
                ins_proc[(ds, 'main64', impl, pct)].append(statistics.mean(r['round_npk']))
            if impl == 'dfuds' and p == 1:
                main_ids[(ds, pct)] = r['ids']
        for step, variant in STEP_VARIANT.items():
            d = os.path.join(a.root, ds, step)
            if not os.path.isdir(d):
                continue
            acc, last = accepted(d)
            un = [n for n in scheduled(d) if last.get(n) != 'ACCEPTED']
            if un:
                problems.append(f'{ds} {step}: {len(un)} scheduled cells not accepted: {" ".join(sorted(un)[:6])}')
            for n in sorted(acc):
                _, _, impl, pct, p = name_fields(n)
                r = parse_insert(os.path.join(d, 'raw', n + '.log'))
                if len(r['round_npk']) != 4:
                    problems.append(f'{n}: {len(r["round_npk"])} measured rounds')
                    continue
                ref = main_ids.get((ds, pct))
                if ref is None or ref != r['ids']:
                    problems.append(f'{n}: insertion batches differ from the main campaign ({ds} pct{pct})')
                m = statistics.mean(r['round_npk'])
                ins_proc[(ds, variant, impl, pct)].append(m)
                ins_cells.append({'dataset': ds, 'step': step, 'impl': impl, 'pct': pct, 'process': p, 'workers': r['workers'],
                                  'method': r['method'], 'b': r['b'], 'ns_per_key': f'{m:.2f}',
                                  'sort_share': f'{r["sort"] / r["timed"]:.4f}' if r['timed'] else '',
                                  'build_s': f'{statistics.mean(r["build"]):.3f}' if r['build'] else ''})
        # PDT-Dyn with one shared build per process (pdtshared)
        d = os.path.join(a.root, ds, 'pdtshared')
        if os.path.isdir(d):
            acc, last = accepted(d)
            un = [n for n in scheduled(d) if last.get(n) != 'ACCEPTED']
            if un:
                problems.append(f'{ds} pdtshared: {len(un)} scheduled cells not accepted: {" ".join(sorted(un))}')
            for n in sorted(acc):
                p = int(n.rsplit('__p', 1)[1])
                per, build, workers = parse_shared(os.path.join(d, 'raw', n + '.log'))
                for pct, r in sorted(per.items()):
                    if len(r['round_npk']) != 4:
                        problems.append(f'{n} pct{pct}: {len(r["round_npk"])} measured rounds')
                        continue
                    ref = main_ids.get((ds, pct))
                    if ref is None or ref != r['ids']:
                        problems.append(f'{n}: insertion batches at pct{pct} differ from the main campaign')
                    m = statistics.mean(r['round_npk'])
                    ins_proc[(ds, 'native_seq_shared', 'dynpdt-compact', pct)].append(m)
                    ins_cells.append({'dataset': ds, 'step': 'pdtshared', 'impl': 'dynpdt-compact', 'pct': pct, 'process': p, 'workers': workers,
                                      'method': 'sequential_native_loop (shared build)', 'b': '', 'ns_per_key': f'{m:.2f}', 'sort_share': '',
                                      'build_s': f'{build:.3f}' if build is not None else ''})
        # sorted point queries, and the sweep's direct cells of the same arms
        d = os.path.join(a.root, ds, 'squery')
        if os.path.isdir(d):
            acc, last = accepted(d)
            un = [n for n in scheduled(d) if last.get(n) != 'ACCEPTED']
            if un:
                problems.append(f'{ds} squery: {len(un)} scheduled cells not accepted: {" ".join(sorted(un))}')
            sweep_hash = collections.defaultdict(dict)    # impl -> {(mix, bq, batch): ids_hash}
            for impl in ('art-olc', 'hot-rowex'):
                for stg in ('mixed', 'hitmiss'):
                    sd = os.path.join(a.sweep, ds, f'measure_{stg}')
                    sacc, _ = accepted(sd)
                    for n in sorted(x for x in sacc if x.startswith(f'SW__{ds}__{impl}__{stg}__p')):
                        rounds, hashes = parse_point(os.path.join(sd, 'raw', n + '.log'))
                        sweep_hash[impl].update(hashes)
                        for (mix, mode, bq), xs in rounds.items():
                            if mode == 'direct' and len(xs) == 4:
                                q_proc[(ds, impl, mix, bq, 'sweep_direct')].append(statistics.mean(xs))
            groups = bad = 0
            for n in sorted(acc):
                _, _, impl, _, p = name_fields(n)
                rounds, hashes = parse_point(os.path.join(d, 'raw', n + '.log'))
                if len(rounds) != 30 or any(len(xs) != 4 for xs in rounds.values()):
                    problems.append(f'{n}: {len(rounds)} cells, measured rounds {sorted(set(len(x) for x in rounds.values()))}')
                    continue   # an incomplete process stays out of every table (as the insertion branch)
                for k, h in hashes.items():
                    ref = sweep_hash[impl].get(k)
                    if ref is not None:
                        groups += 1
                        bad += ref != h
                for (mix, mode, bq), xs in sorted(rounds.items()):
                    m = statistics.mean(xs)
                    q_proc[(ds, impl, mix, bq, mode)].append(m)
                    q_cells.append({'dataset': ds, 'impl': impl, 'process': p, 'mix': mix, 'mode': mode, 'bq': bq, 'ns_per_query': f'{m:.3f}'})
            checks.append(f'{ds} squery trace_groups_vs_sweep={groups} inconsistent={bad}')
            if bad or not groups:
                problems.append(f'{ds} squery: {bad} of {groups} batch hashes differ from the sweep')
    with open(os.path.join(a.out, 'insert_cells.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, ['dataset', 'step', 'impl', 'pct', 'process', 'workers', 'method', 'b', 'ns_per_key', 'sort_share', 'build_s'],
                           lineterminator='\n')
        w.writeheader(); w.writerows(ins_cells)
    with open(os.path.join(a.out, 'insert_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'variant', 'impl', 'pct', 'processes', 'median_ns_per_key', 'min', 'max'])
        for k in sorted(ins_proc):
            md, lo, hi = med(ins_proc[k])
            w.writerow([*k, len(ins_proc[k]), f'{md:.2f}', f'{lo:.2f}', f'{hi:.2f}'])
    with open(os.path.join(a.out, 'query_cells.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, ['dataset', 'impl', 'process', 'mix', 'mode', 'bq', 'ns_per_query'], lineterminator='\n')
        w.writeheader(); w.writerows(q_cells)
    with open(os.path.join(a.out, 'query_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'impl', 'mix', 'bq', 'mode', 'processes', 'median_ns_per_query', 'min', 'max'])
        for k in sorted(q_proc):
            md, lo, hi = med(q_proc[k])
            w.writerow([*k, len(q_proc[k]), f'{md:.3f}', f'{lo:.3f}', f'{hi:.3f}'])
    # report
    M = {k: med(v)[0] for k, v in ins_proc.items()}
    Q = {k: med(v)[0] for k, v in q_proc.items()}
    rep = ['# Single-worker / sorted-baseline supplement (plan section 21)', '',
           'ns per inserted key (insertion) or per query (queries): mean of the four measured rounds per process, median over '
           'processes. One setting per table: the one-core tables hold one-worker and sequential cells only, the 64-worker '
           'tables 64-worker cells only (main64 = the main campaign\'s cells, same keys and batches); the cross-setting '
           'speedup is a separate ratio table.', '']
    for ds in datasets:
        rep += [f'## {ds}', '', '### One core: DFUDS / DT batch vs one key per call, PDT-Dyn sequential insertion', '',
                '| arm | % of N | batch | one key per call | one key / batch | PDT-Dyn (sequential) |', '|---|---|---:|---:|---:|---:|']
        for impl in ('dfuds', 'dfuds-dt'):
            for p in PCTS:
                b1, p1 = M.get((ds, 'batch_w1', impl, p)), M.get((ds, 'point_w1', impl, p))
                if b1 is None and p1 is None:
                    continue
                rep.append(f'| {impl} | {p} | {fmt(b1)} | {fmt(p1)} | {fmt(p1 / b1 if p1 and b1 else None, 2)} | '
                           f'{fmt(M.get((ds, "native_seq", "dynpdt-compact", p)))} |')
        rep += ['', '### Speedup of the 64-worker batch over the one-worker batch (ratio across settings)', '',
                '| arm | % of N | batch 1 worker / batch 64 workers |', '|---|---|---:|']
        for impl in ('dfuds', 'dfuds-dt'):
            for p in PCTS:
                b1, b64 = M.get((ds, 'batch_w1', impl, p)), M.get((ds, 'main64', impl, p))
                if b1 is not None and b64 is not None:
                    rep.append(f'| {impl} | {p} | {fmt(b1 / b64, 2)} |')
        if any(k[0] == ds and k[1] == 'native_seq_shared' for k in ins_proc):
            rep += ['', '### PDT-Dyn sequential insertion, one build per size and trial (fork per round)', '',
                    '| % of N | shared build | fresh build per round | shared ÷ fresh | PMA-DFUDS batch, 1 worker |', '|---|---:|---:|---:|---:|']
            for p in PCTS:
                sh, fr = M.get((ds, 'native_seq_shared', 'dynpdt-compact', p)), M.get((ds, 'native_seq', 'dynpdt-compact', p))
                if sh is not None:
                    rep.append(f'| {p} | {fmt(sh)} | {fmt(fr)} | {fmt(sh / fr if sh and fr else None, 2)} | {fmt(M.get((ds, "batch_w1", "dfuds", p)))} |')
            rep += ['', '### PDT-Dyn sequential ÷ PMA-DFUDS batch at 64 workers (ratio across settings)', '', '| % of N | ratio |', '|---|---:|']
            for p in PCTS:
                sh, b64 = M.get((ds, 'native_seq_shared', 'dynpdt-compact', p)), M.get((ds, 'main64', 'dfuds', p))
                if sh is not None and b64 is not None:
                    rep.append(f'| {p} | {fmt(sh / b64, 1)} |')
        rep += ['', '### 64 workers: ART-OLC / HOT-ROWEX insertion, held-out order vs sorted inside the timer', '',
                '| arm | % of N | as given (main64) | sorted | sorted / as given |', '|---|---|---:|---:|---:|']
        for impl in ('art-olc', 'hot-rowex'):
            for p in PCTS:
                u, s = M.get((ds, 'main64', impl, p)), M.get((ds, 'sorted64', impl, p))
                if s is None:
                    continue
                rep.append(f'| {impl} | {p} | {fmt(u)} | {fmt(s)} | {fmt(s / u if s and u else None, 2)} |')
        rep += ['', '### ART-OLC / HOT-ROWEX point queries: direct vs sorted (same process and index), 64 workers', '',
                '| arm | mix | bq | direct | sorted | sorted / direct | sweep direct |', '|---|---|---:|---:|---:|---:|---:|']
        for impl in ('art-olc', 'hot-rowex'):
            for (d2, i2, mix, bq, mode) in sorted(k for k in Q if k[0] == ds and k[1] == impl and k[4] == 'sorted'):
                dr, so, sw = Q.get((ds, impl, mix, bq, 'direct')), Q[(ds, impl, mix, bq, 'sorted')], Q.get((ds, impl, mix, bq, 'sweep_direct'))
                rep.append(f'| {impl} | {mix} | {bq} | {fmt(dr, 2)} | {fmt(so, 2)} | {fmt(so / dr if dr else None, 2)} | {fmt(sw, 2)} |')
        rep.append('')
    checks.append(f'problems={len(problems)}')
    with open(os.path.join(a.out, 'checks.txt'), 'w') as f:
        f.write('\n'.join(problems + checks) + '\n')
    with open(os.path.join(a.out, 'report.md'), 'w') as f:
        f.write('\n'.join(rep) + '\n')
    print('\n'.join(problems + checks))
    sys.exit(1 if problems else 0)


if __name__ == '__main__':
    main()
