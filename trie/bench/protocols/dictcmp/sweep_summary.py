#!/usr/bin/env python3
"""Summary of the point-query batch-size supplement (experiment point-batch-sweep-v1; plan section 18.5).

usage: sweep_summary.py <sweep-root> <out-dir>
Reads every dataset's measure_<stage>/ledger.tsv and the raw logs of its ACCEPTED SW cells (the last ledger row of a
cell decides; rejected and interfered attempts are listed, never aggregated) and writes:
  rounds.csv    one row per (dataset, arm, mode, mix, size, process, round): ns, queries, ns/query, measured flag
  process.csv   one row per (dataset, arm, mode, mix, size, process): the mean of the four measured rounds
  summary.csv   one row per (dataset, arm, mode, mix, size): median / min / max of the process means, processes, L,
                capped-short-round flag, duplicate share of the trace, status (OK, NA status, PENDING)
  ratios.csv    each DFUDS / DT mode's median divided separately by ART-OLC, HOT-ROWEX and FST (same dataset, mix, size)
  crossover.csv per (dataset, mix, arm): shared / direct per size and the smallest size where shared is faster
  coverage.csv  per (dataset, stage, arm): accepted processes of the scheduled repetitions or the NA status (PENDING:
                nothing scheduled yet)
  build.csv     the single build of every accepted measure process (QUERY_BUILD: build_only input, no warm-up):
                dataset, arm, stage, process, seconds, key count, resident and peak memory after the build
  build_summary.csv  per (dataset, arm): median / min / max over those processes (the build time of arms whose build
                panel was dropped, user decision October 26 12:16, and a comparable single-build value for every arm)
  space.csv     the space census printed by query-only processes after their last timed cell (--space-after)
  phases.csv    per expected-arm phase (five-arm-v1: DFUDS, DT, ART-OLC, HOT-ROWEX, FST; eight-arm-v2, plan revision
                15: plus MARISA, PDT, PDT-Dyn) and (dataset, stage): COMPLETE when every arm of the phase has three
                accepted processes or an evidenced NA status, else INCOMPLETE with the missing arms; expected_manifest.json
                records the phases
  checks.txt    reconciliation: one ids_hash per (dataset, mix, bq, batch) across arms, modes, processes and rounds;
                bq equal to sizes.tsv; one L per (dataset, mix, size) across arms and modes; hits = expected (no
                CHECK_FAIL); every OK cell with four measured rounds; QUERY_PROTOCOL c2trace-v1 and the experiment id
A size is keyed by its integer bq and label (pct<p>); rows of different sizes are never merged.
"""
import collections
import csv
import json
import os
import re
import statistics
import sys

EXPERIMENT = 'point-batch-sweep-v1'
KV = re.compile(r'(\w+)=("[^"]*"|\S+)')
STAGES = {'mixed': ['mixed'], 'hitmiss': ['hit', 'miss'], 'all': ['mixed', 'hit', 'miss']}   # all: query-only arms
PHASES = {'five-arm-v1': ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-upstream'],
          'eight-arm-v2': ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-upstream', 'marisa', 'pdt-static', 'dynpdt-compact'],
          'fst64-v3': ['fst-64']}
# a phase over a subset of the datasets: fst-64 (the patched FST, October 27, docs/fst64_patch_2026-10-27.md) runs on
# DNA and UK-2014 only (the planned WebBase calibration was dropped by the user); its arm has no row where it has no cell
PHASE_DATASETS = {'fst64-v3': ['dna_full_31mers', 'uk_2014_urls']}
OPTIONAL_ARMS = {'fst-64'}
ARMS = PHASES['eight-arm-v2'] + PHASES['fst64-v3']
DATASETS = ['enwiki_20261001_ns0', 'c2_xml_recon', 'webbase_2001', 'dna_full_31mers', 'uk_2014_urls']
SHARED = {'dfuds', 'dfuds-dt'}
BASE = ['art-olc', 'hot-rowex', 'fst-upstream', 'marisa', 'pdt-static', 'dynpdt-compact', 'fst-64']
L_MAX, TARGET_NS = 1000, 200e6


def kv(line):
    return {k: v.strip('"') for k, v in KV.findall(line)}


def last_verdicts(ledger):
    last = {}
    if os.path.exists(ledger):
        for line in open(ledger):
            f = line.rstrip('\n').split('\t')
            if len(f) >= 8 and f[0] != 'name':
                last[f[0]] = (f[1], f[7], f[8] if len(f) > 8 else '')
    return last


def main():
    root, out = sys.argv[1], sys.argv[2]
    os.makedirs(out, exist_ok=True)
    rounds, procs, builds, spaces = [], {}, [], []
    hashes = collections.defaultdict(set)          # (ds, mix, bq, batch) -> ids hashes
    Ls = collections.defaultdict(set)              # (ds, mix, label) -> L values
    dups = {}                                      # (ds, mix, bq) -> duplicate share of the trace
    problems = []
    coverage = []
    sizes = {}
    na = {}
    for ds in sorted(os.listdir(root)):
        d = os.path.join(root, ds)
        if not os.path.isdir(d) or ds == 'summary':
            continue
        sf = os.path.join(d, 'sizes.tsv')
        if os.path.exists(sf):
            for row in csv.DictReader(open(sf), delimiter='\t'):
                sizes[(ds, row['label'])] = int(row['bq'])
        in_all = {n.split('__')[2] for n in last_verdicts(os.path.join(d, 'measure_all', 'ledger.tsv'))}   # arms of stage all
        for stage in STAGES:
            last = last_verdicts(os.path.join(d, f'measure_{stage}', 'ledger.tsv'))
            if not last:
                continue
            for arm in ARMS:
                naname = f'SW__{ds}__{arm}__{stage}'
                if naname in last and last[naname][0] == 'NA':
                    na[(ds, stage, arm)] = last[naname][1]
                    coverage.append((ds, stage, arm, 0, 0, last[naname][1]))
                    continue
                names = [n for n in last if n.startswith(naname + '__p')]
                if not names and (stage == 'all' or arm in in_all or arm in OPTIONAL_ARMS):
                    continue   # stage all holds only the query-only arms, and they have no mixed / hitmiss rows
                acc = [n for n in names if last[n][1] == 'ACCEPTED']
                rej = [n for n in names if last[n][1] != 'ACCEPTED']
                coverage.append((ds, stage, arm, len(acc), len(names), 'OK' if names and not rej and len(acc) >= 3 else
                                 'INCOMPLETE' if names else 'PENDING'))
                for n in acc:
                    p = int(n.rsplit('__p', 1)[1])
                    log = os.path.join(d, f'measure_{stage}', 'raw', n + '.log')
                    text = open(log, errors='replace').read()
                    if f'EXPERIMENT id={EXPERIMENT} ' not in text or 'QUERY_PROTOCOL id=c2trace-v1 ' not in text:
                        problems.append(f'{n}: experiment or protocol record missing')
                    if '\nCHECK_FAIL' in text:
                        problems.append(f'{n}: CHECK_FAIL')
                    cell = None
                    trace = None
                    for line in text.splitlines():
                        tag = line.split(' ', 1)[0]
                        if tag == 'SPACE':
                            sp = kv(line)
                            spaces.append((ds, arm, stage, p, sp['checkpoint'], int(sp['keys']), int(sp['standalone_bytes']),
                                           int(sp['index_only_bytes']), float(sp['bytes_per_key']), sp['basis']))
                        elif tag == 'QUERY_BUILD':
                            qb = kv(line)
                            builds.append((ds, arm, stage, p, float(qb['build_ns']) / 1e9, int(qb['key_count']), int(qb['rss_kb']),
                                           int(qb['hwm_kb'])))
                        elif tag == 'QUERY_TRACE':
                            t = kv(line)
                            trace = (int(t['batch']), int(t['duplicates']) / max(1, int(t['queries'])))
                        elif tag == 'POINT_CELL':
                            c = kv(line)
                            if 'status' in c:
                                problems.append(f'{n}: {line}')
                                cell = None
                                continue
                            cell = (c['mix'], c['mode'], c['bq_label'], int(c['bq']), int(c['L']))
                            Ls[(ds, cell[0], cell[2])].add(cell[4])
                            if trace and trace[0] == cell[3]:
                                dups[(ds, cell[0], cell[3])] = trace[1]
                            exp = sizes.get((ds, cell[2]))
                            if exp is not None and exp != cell[3]:
                                problems.append(f'{n}: bq {cell[3]} != sizes.tsv {exp} ({cell[2]})')
                        elif tag == 'POINT_BATCH' and cell:
                            b = kv(line)
                            hashes[(ds, b['mix'], int(b['bq']), int(b['batch']))].add(b['ids_hash'])
                        elif tag == 'POINT_ROUND' and cell:
                            r = kv(line)
                            mix, mode, lab, bq, L = cell
                            ns = float(r['ns'])
                            row = dict(dataset=ds, stage=stage, impl=arm, mode=mode, mix=mix, label=lab, pct=lab[3:], bq=bq, L=L,
                                       process=p, round=int(r['round']), measured=int(r['measured']), ns=ns,
                                       queries=int(r['queries']), ns_per_query=float(r['ns_per_query']))
                            rounds.append(row)
                            if row['measured']:
                                procs.setdefault((ds, arm, mode, mix, lab, bq, L, p), []).append(row['ns_per_query'])
    for key, v in procs.items():
        if len(v) != 4:
            problems.append(f'{key}: {len(v)} measured rounds (expected 4)')
    inconsistent = [k for k, v in hashes.items() if len(v) != 1]
    l_bad = [k for k, v in Ls.items() if len(v) != 1]
    cols = ['dataset', 'stage', 'impl', 'mode', 'mix', 'label', 'pct', 'bq', 'L', 'process', 'round', 'measured', 'ns', 'queries',
            'ns_per_query']
    with open(os.path.join(out, 'rounds.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator='\n')
        w.writeheader()
        w.writerows(rounds)
    with open(os.path.join(out, 'process.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'impl', 'mode', 'mix', 'label', 'bq', 'L', 'process', 'ns_per_query_mean'])
        for (ds, arm, mode, mix, lab, bq, L, p), v in sorted(procs.items()):
            w.writerow([ds, arm, mode, mix, lab, bq, L, p, f'{statistics.mean(v):.4f}'])
    cells = collections.defaultdict(list)
    for (ds, arm, mode, mix, lab, bq, L, p), v in procs.items():
        cells[(ds, mix, lab, bq, arm, mode, L)].append(statistics.mean(v))
    summ = {}
    with open(os.path.join(out, 'summary.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'mix', 'label', 'pct', 'bq', 'impl', 'mode', 'L', 'processes', 'median_ns_per_query', 'min', 'max',
                    'capped_short_round', 'duplicate_share', 'status'])
        for (ds, mix, lab, bq, arm, mode, L), v in sorted(cells.items(), key=lambda x: (x[0][0], x[0][1], x[0][3], x[0][4], x[0][5])):
            med = statistics.median(v)
            summ[(ds, mix, bq, arm, mode)] = med
            capped = int(L == L_MAX and med * bq * L < TARGET_NS)
            w.writerow([ds, mix, lab, lab[3:], bq, arm, mode, L, len(v), f'{med:.4f}', f'{min(v):.4f}', f'{max(v):.4f}', capped,
                        f'{dups.get((ds, mix, bq), float("nan")):.4f}', 'OK' if len(v) >= 3 else 'INCOMPLETE'])
        for (ds, stage, arm), st in sorted(na.items()):
            for mix in STAGES[stage]:
                for (d2, lab), bq in sorted(sizes.items()):
                    if d2 == ds:
                        w.writerow([ds, mix, lab, lab[3:], bq, arm, 'direct', '', 0, '', '', '', '', '', st])
    with open(os.path.join(out, 'ratios.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'mix', 'bq', 'impl', 'mode', 'baseline', 'ratio'])
        for (ds, mix, bq, arm, mode), v in sorted(summ.items()):
            if arm not in SHARED:
                continue
            for b in BASE:
                ref = summ.get((ds, mix, bq, b, 'direct'))
                if ref:
                    w.writerow([ds, mix, bq, arm, mode, b, f'{v / ref:.4f}'])
    with open(os.path.join(out, 'crossover.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'mix', 'impl', 'bq', 'shared_over_direct', 'first_bq_shared_faster'])
        for ds, mix, arm in sorted({(k[0], k[1], k[3]) for k in summ if k[3] in SHARED}):
            bqs = sorted(k[2] for k in summ if k[:2] == (ds, mix) and k[3] == arm and k[4] == 'direct')
            first = next((b for b in bqs if summ.get((ds, mix, b, arm, 'shared'), 1e300) < summ[(ds, mix, b, arm, 'direct')]), '')
            for b in bqs:
                s = summ.get((ds, mix, b, arm, 'shared'))
                w.writerow([ds, mix, arm, b, f'{s / summ[(ds, mix, b, arm, "direct")]:.4f}' if s else '', first])
    with open(os.path.join(out, 'coverage.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'stage', 'impl', 'accepted_processes', 'scheduled_processes', 'status'])
        w.writerows(coverage)
    with open(os.path.join(out, 'build.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'impl', 'stage', 'process', 'build_s', 'key_count', 'rss_kb', 'hwm_kb'])
        for b in sorted(builds):
            w.writerow([b[0], b[1], b[2], b[3], f'{b[4]:.3f}', b[5], b[6], b[7]])
    with open(os.path.join(out, 'space.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'impl', 'stage', 'process', 'checkpoint', 'keys', 'standalone_bytes', 'index_only_bytes', 'bytes_per_key',
                    'basis'])
        for x in sorted(spaces):
            w.writerow(list(x[:8]) + [f'{x[8]:.4f}', x[9]])
    bgroups = collections.defaultdict(list)
    for b in builds:
        bgroups[(b[0], b[1])].append(b[4])
    with open(os.path.join(out, 'build_summary.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['dataset', 'impl', 'processes', 'median_build_s', 'min', 'max', 'basis'])
        for (ds, arm), v in sorted(bgroups.items()):
            w.writerow([ds, arm, len(v), f'{statistics.median(v):.3f}', f'{min(v):.3f}', f'{max(v):.3f}',
                        'one build per point-sweep process; build_only input; no warm-up'])
    cov = {(c[0], c[1], c[2]): c for c in coverage}
    phase_rows = []
    for ph, arms in PHASES.items():
        for ds in PHASE_DATASETS.get(ph, DATASETS):
            for stage in ('mixed', 'hitmiss'):   # an arm measured in stage all covers both
                done = ('OK', 'SIZE_LIMIT', 'UNSUPPORTED_KEY_DOMAIN', 'TIME_LIMIT')
                miss = [a for a in arms if cov.get((ds, stage, a), (0, 0, 0, 0, 0, 'PENDING'))[5] not in done and
                        cov.get((ds, 'all', a), (0, 0, 0, 0, 0, 'PENDING'))[5] not in done]
                phase_rows.append((ph, ds, stage, len(arms), len(arms) - len(miss), 'COMPLETE' if not miss else 'INCOMPLETE',
                                   ' '.join(miss)))
    with open(os.path.join(out, 'phases.csv'), 'w', newline='') as f:
        w = csv.writer(f, lineterminator='\n')
        w.writerow(['phase', 'dataset', 'stage', 'required_arms', 'complete_arms', 'status', 'missing_arms'])
        w.writerows(phase_rows)
    with open(os.path.join(out, 'expected_manifest.json'), 'w') as f:
        json.dump({'experiment': EXPERIMENT, 'phases': PHASES, 'phase_datasets': PHASE_DATASETS, 'datasets': DATASETS, 'stages': STAGES,
                   'sizes': ['pct0.001', 'pct0.01', 'pct0.1', 'pct1', 'pct10'], 'processes_per_cell': 3,
                   'shared_modes': sorted(SHARED)}, f, indent=1)
    with open(os.path.join(out, 'checks.txt'), 'w') as f:
        f.write(f'experiment={EXPERIMENT} query_protocol=c2trace-v1\n')
        f.write(f'trace_hash_groups={len(hashes)} inconsistent={len(inconsistent)}\n')
        f.write(f'L_groups={len(Ls)} inconsistent_L={len(l_bad)}\n')
        f.write(f'process_observations={len(procs)} problems={len(problems)}\n')
        for ph in PHASES:
            done = sum(1 for r in phase_rows if r[0] == ph and r[5] == 'COMPLETE')
            tot = sum(1 for r in phase_rows if r[0] == ph)
            f.write(f'phase {ph} complete_visits={done}/{tot}\n')
        for k in inconsistent[:20]:
            f.write(f'INCONSISTENT_HASH {k} {sorted(hashes[k])}\n')
        for k in l_bad:
            f.write(f'INCONSISTENT_L {k} {sorted(Ls[k])}\n')
        for p in problems[:50]:
            f.write(f'PROBLEM {p}\n')
    print(open(os.path.join(out, 'checks.txt')).read().strip())
    sys.exit(1 if inconsistent or l_bad or problems else 0)


if __name__ == '__main__':
    main()
