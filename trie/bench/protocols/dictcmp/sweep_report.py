#!/usr/bin/env python3
"""Report tables of the point-query batch-size supplement (experiment point-batch-sweep-v1; plan section 18.5).

usage: sweep_report.py <summary-dir> > tables.md
Markdown from sweep_summary.py's CSVs: an overview of every dataset and mix at its largest size (medians, ratios to
each baseline, shared/direct crossover), then per dataset and mix the median ns/query (min-max of the three process means)
of every structure and mode at every size, the ratios of each DFUDS/DT mode to ART-OLC, HOT-ROWEX and FST, the
shared/direct ratio with the first size where shared is faster, capped short rounds and duplicate shares, and the
coverage table. Measured values only; NA statuses are printed as such, pending cells as PENDING.
"""
import collections
import csv
import os
import sys

ORDER = [('dfuds', 'direct', 'DFUDS direct'), ('dfuds', 'shared', 'DFUDS shared'), ('dfuds-dt', 'direct', 'DT direct'),
         ('dfuds-dt', 'shared', 'DT shared'), ('art-olc', 'direct', 'ART-OLC'), ('hot-rowex', 'direct', 'HOT-ROWEX'),
         ('fst-upstream', 'direct', 'FST'), ('marisa', 'direct', 'MARISA'), ('pdt-static', 'direct', 'PDT'),
         ('dynpdt-compact', 'direct', 'PDT-Dyn'), ('fst-64', 'direct', 'FST-64')]
BASE = [('art-olc', 'ART-OLC'), ('hot-rowex', 'HOT-ROWEX'), ('fst-upstream', 'FST'), ('marisa', 'MARISA'),
        ('pdt-static', 'PDT'), ('dynpdt-compact', 'PDT-Dyn'), ('fst-64', 'FST-64')]
OPTIONAL = {'fst-64'}   # the patched FST (October 27): rows only on the datasets where it has cells or an NA status
LABELS = {'c2_xml_recon': 'XML', 'enwiki_20261001_ns0': 'Wikipedia', 'webbase_2001': 'WebBase 2001',
          'dna_full_31mers': 'DNA 31-mers', 'uk_2014_urls': 'UK 2014'}
DS_ORDER = ['enwiki_20261001_ns0', 'c2_xml_recon', 'webbase_2001', 'dna_full_31mers', 'uk_2014_urls']


def main():
    src = sys.argv[1]
    rows = collections.defaultdict(dict)
    na = collections.defaultdict(dict)
    meta = {}
    for r in csv.DictReader(open(os.path.join(src, 'summary.csv'))):
        key = (r['dataset'], r['mix'])
        bq = int(r['bq'])
        if r['median_ns_per_query']:
            rows[key][(r['impl'], r['mode'], bq)] = r
            meta[(key, bq)] = (r['pct'], r['L'], r['capped_short_round'], r['duplicate_share'])
        else:
            na[key][r['impl']] = r['status']
            meta.setdefault((key, bq), (r['pct'], '', '', ''))
    keys = sorted(rows, key=lambda k: (DS_ORDER.index(k[0]) if k[0] in DS_ORDER else 9, ['mixed', 'hit', 'miss'].index(k[1])))
    # overview: every dataset and mix at its largest size, and the shared/direct crossover
    print('### Overview at the largest size (10 % of N)\n')
    print('| Dataset | Mix | Queries | DFUDS shared | DT shared | ART-OLC | HOT-ROWEX | FST | MARISA | PDT | PDT-Dyn | DFUDS sh. ÷ ART | ÷ HOT | ÷ FST | ÷ MARISA | ÷ PDT | ÷ PDT-Dyn | DT sh. ÷ ART | ÷ HOT | ÷ FST | DFUDS crossover | DT crossover | FST-64 | DFUDS sh. ÷ FST-64 |')
    print('|---|---|---:|' + '---:|' * 17 + '---|---|---:|---:|')
    for key in keys:
        cell = rows[key]
        bqs = sorted({b for (k2, b) in meta if k2 == key})
        top = bqs[-1]
        med = {(i, m): float(cell[(i, m, top)]['median_ns_per_query']) for (i, m, b) in cell if b == top}
        def v(i, m='direct'):
            return f'{med[(i, m)]:.2f}' if (i, m) in med else na[key].get(i, 'PENDING')
        def r(i, m, b):
            return f'{med[(i, m)] / med[(b, "direct")]:.2f}x' if (i, m) in med and (b, 'direct') in med else '—'
        def cross(i):
            for b in bqs:
                sh, di = cell.get((i, 'shared', b)), cell.get((i, 'direct', b))
                if sh and di and float(sh['median_ns_per_query']) < float(di['median_ns_per_query']):
                    return f'{b:,}'
            return 'none measured'
        print(f'| {LABELS.get(key[0], key[0])} | {key[1]} | {top:,} | {v("dfuds", "shared")} | {v("dfuds-dt", "shared")} | {v("art-olc")} | '
              f'{v("hot-rowex")} | {v("fst-upstream")} | {v("marisa")} | {v("pdt-static")} | {v("dynpdt-compact")} | '
              f'{r("dfuds", "shared", "art-olc")} | {r("dfuds", "shared", "hot-rowex")} | {r("dfuds", "shared", "fst-upstream")} | '
              f'{r("dfuds", "shared", "marisa")} | {r("dfuds", "shared", "pdt-static")} | {r("dfuds", "shared", "dynpdt-compact")} | '
              f'{r("dfuds-dt", "shared", "art-olc")} | {r("dfuds-dt", "shared", "hot-rowex")} | '
              f'{r("dfuds-dt", "shared", "fst-upstream")} | {cross("dfuds")} | {cross("dfuds-dt")} | '
              f'{v("fst-64") if ("fst-64", "direct") in med or "fst-64" in na[key] else "-"} | {r("dfuds", "shared", "fst-64")} |')
    print()
    for key in keys:
        ds, mix = key
        cell = rows[key]
        bqs = sorted({b for (_, b) in meta if _ == key})
        print(f'### {LABELS.get(ds, ds)} · {mix}\n')
        print('| Structure | ' + ' | '.join(f'{meta[(key, b)][0]}% of N ({b:,} q)' for b in bqs) + ' |')
        print('|---|' + '---:|' * len(bqs))
        for impl, mode, name in ORDER:
            if impl in OPTIONAL and impl not in na[key] and not any(k[0] == impl for k in cell):
                continue
            vals = []
            for b in bqs:
                r = cell.get((impl, mode, b))
                if r:
                    part = '' if int(r['processes']) >= 3 else f' (n={r["processes"]})'
                    vals.append(f'{float(r["median_ns_per_query"]):.2f} ({float(r["min"]):.2f}-{float(r["max"]):.2f}){part}')
                else:
                    vals.append(na[key].get(impl, 'PENDING'))
            print(f'| {name} | ' + ' | '.join(vals) + ' |')
        print('| L (batches per round) | ' + ' | '.join(meta[(key, b)][1] + (' capped' if meta[(key, b)][2] == '1' else '') for b in bqs) + ' |')
        print('| duplicate share of the trace | ' + ' | '.join(
            f'{float(meta[(key, b)][3]):.3f}' if meta[(key, b)][3] not in ('', 'nan') else '' for b in bqs) + ' |')
        print('\nRatios (median / baseline median, same size; below 1 = faster than the baseline):\n')
        print('| Structure | Baseline | ' + ' | '.join(f'{b:,}' for b in bqs) + ' |')
        print('|---|---|' + '---:|' * len(bqs))
        for impl, mode, name in ORDER[:4]:
            for b_impl, b_name in BASE:
                if b_impl in OPTIONAL and b_impl not in na[key] and not any(k[0] == b_impl for k in cell):
                    continue
                vals = []
                for b in bqs:
                    r, ref = cell.get((impl, mode, b)), cell.get((b_impl, 'direct', b))
                    vals.append(f'{float(r["median_ns_per_query"]) / float(ref["median_ns_per_query"]):.2f}x' if r and ref else
                                na[key].get(b_impl, '') if not ref else '')
                print(f'| {name} | {b_name} | ' + ' | '.join(vals) + ' |')
        print('\nShared / direct (below 1 = shared faster):\n')
        print('| Structure | ' + ' | '.join(f'{b:,}' for b in bqs) + ' | first size where shared is faster |')
        print('|---|' + '---:|' * len(bqs) + '---|')
        for impl, name in (('dfuds', 'DFUDS'), ('dfuds-dt', 'DT')):
            vals, first = [], None
            for b in bqs:
                s, d = cell.get((impl, 'shared', b)), cell.get((impl, 'direct', b))
                if s and d:
                    x = float(s['median_ns_per_query']) / float(d['median_ns_per_query'])
                    vals.append(f'{x:.2f}')
                    if x < 1 and first is None:
                        first = b
                else:
                    vals.append('')
            print(f'| {name} | ' + ' | '.join(vals) + f' | {f"{first:,} queries" if first else "none measured"} |')
        print()
    print('### Coverage\n')
    print('| Dataset | Stage | Structure | Accepted / scheduled processes | Status |')
    print('|---|---|---|---:|---|')
    for r in csv.DictReader(open(os.path.join(src, 'coverage.csv'))):
        print(f'| {LABELS.get(r["dataset"], r["dataset"])} | {r["stage"]} | {r["impl"]} | {r["accepted_processes"]} / '
              f'{r["scheduled_processes"]} | {r["status"]} |')
    print('\nChecks:\n\n```\n' + open(os.path.join(src, 'checks.txt')).read().strip() + '\n```')


if __name__ == '__main__':
    main()
