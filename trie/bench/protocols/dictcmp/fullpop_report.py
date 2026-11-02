#!/usr/bin/env python3
"""Markdown report of the full-population, hit-only round from summary_full.py's CSVs (and, when present, the
section-7 breakdown.csv): one table block per dataset (build, point, range, space), the UK point sweep, the coverage
matrix and the breakdown table. Numbers are copied from the CSVs, never recomputed differently: query cells are the
median [min-max] of the three process means, builds the median [min-max] of three single builds, space the process-1
record. usage: fullpop_report.py <summary-dir> <out.md>"""
import csv
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # campaign_spec.py next to this script

DS = ['c2_xml_recon', 'enwiki_20261001_ns0', 'webbase_2001', 'dna_full_31mers', 'uk_2014_urls']
ARMS = ['dfuds', 'dfuds-dt', 'art-olc', 'hot-rowex', 'fst-64', 'marisa', 'pdt-static', 'dynpdt-compact']
LABEL = {'c2_xml_recon': 'XML', 'enwiki_20261001_ns0': 'Wikipedia', 'webbase_2001': 'WebBase', 'dna_full_31mers': 'DNA',
         'uk_2014_urls': 'UK-2014', 'dfuds': 'DFUDS', 'dfuds-dt': 'DT', 'art-olc': 'ART-OLC', 'hot-rowex': 'HOT-ROWEX',
         'fst-64': "FST'", 'marisa': 'MARISA', 'pdt-static': 'PDT', 'dynpdt-compact': 'PDT-Dyn'}
N = {'c2_xml_recon': 2149271, 'enwiki_20261001_ns0': 19315921, 'webbase_2001': 117603792, 'dna_full_31mers': 367422517, 'uk_2014_urls': 787801471}
UK_SIZES = ['pct0.001', 'pct0.01', 'pct0.1', 'pct1', 'pct10']


def rd(d, name):
    p = os.path.join(d, name)
    return list(csv.DictReader(open(p))) if os.path.exists(p) else []


def f(v, nd=1):
    try:
        return f'{float(v):,.{nd}f}'
    except (TypeError, ValueError):
        return '-'


def mmm(r, k, nd):
    if r is None or r.get(f'median_{k}', '') == '':
        return '-'
    return f"{f(r['median_' + k], nd)} [{f(r['min_' + k], nd)}-{f(r['max_' + k], nd)}]"


def main():
    d, out = sys.argv[1], sys.argv[2]
    # --spec <campaign_spec.json>: the declared k list and datasets drive the range tables (plan of October 30, section 4
    # item 4); without it the k columns are the k values of the dataset's paired rows, never an assumed five-k sweep
    spec = json.load(open(sys.argv[sys.argv.index('--spec') + 1])) if '--spec' in sys.argv else None
    build, point, rng, space, cov, bd = rd(d, 'build_summary.csv'), rd(d, 'point_summary.csv'), rd(d, 'range_summary.csv'), rd(d, 'space.csv'), rd(d, 'coverage.csv'), rd(d, 'breakdown.csv')
    L = []
    L.append('# Full-population, hit-only dictionary comparison: result tables\n')
    L.append('Source: `summary_full.py` over the accepted sessions (S = N, hit-only, 64 workers, 2 warm-up + 4 measured rounds per cell per '
             'process, three independent processes with one build each). Query cells: median [min-max] of the three process means '
             '(whiskers of the figures = that range). Builds: median [min-max] of the three single end-to-end builds from the shuffled '
             'list (sort inside the timer for sorted-input builders). Space: the process-1 record on the retained index, each arm\'s '
             'accounting basis. ns/query and ns/request are elapsed batch time over 64 workers divided by the requests (a throughput-'
             'derived cost, not single-query latency).\n')
    for ds in DS:
        b = {r['impl']: r for r in build if r['dataset'] == ds}
        if not b:
            continue
        L.append(f'## {LABEL[ds]} (`{ds}`, N = S = {N[ds]:,})\n')
        sizes = UK_SIZES if ds == 'uk_2014_urls' else ['pct10']
        psize = 'pct10'
        rk = 'Range k100 (paired sweep) ' if any(r.get('paired') == '1' for r in rng if r['dataset'] == ds) else 'Range k100 (archived full-pop-hit-v1) '
        L.append(f'| Arm | Build e2e s | construct s | sort s | peak build delta GB (max of 3) | Point 10 % direct ns/q | Point 10 % shared ns/q | {rk}direct ns/req | {rk}shared ns/req | Serving bytes | B/key | B/input byte |')
        L.append('|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|')
        for a in ARMS:
            br = b.get(a)
            st = [c for c in cov if c['dataset'] == ds and c['impl'] == a and c['cell'] == 'build']
            if br is None:
                status = st[0]['status'] if st else 'MISSING'
                L.append(f'| {LABEL[a]} | {status} | | | | | | | | | | |')
                continue
            pd = next((r for r in point if r['dataset'] == ds and r['impl'] == a and r['size'] == psize and r['mode'] == 'direct'), None)
            ps = next((r for r in point if r['dataset'] == ds and r['impl'] == a and r['size'] == psize and r['mode'] == 'shared'), None)
            rd_ = next((r for r in rng if r['dataset'] == ds and r['impl'] == a and r['mode'] == 'direct' and r['k'] == '100'), None)
            rs = next((r for r in rng if r['dataset'] == ds and r['impl'] == a and r['mode'] == 'shared' and r['k'] == '100'), None)
            sp = next((r for r in space if r['dataset'] == ds and r['impl'] == a), None)
            def short(r):
                return '' if r is None or r.get('short_round') != '1' else f" (rounds {f(r['round_ms_median'], 0)} ms)"
            e2e = f(br['e2e_median_s'], 3) + ' [' + f(br['e2e_min_s'], 3) + '-' + f(br['e2e_max_s'], 3) + ']'
            L.append(f"| {LABEL[a]} | {e2e} "
                     f"| {f(br['construct_median_s'], 3)} | {f(br['sort_median_s'], 3)} | {f(br['peak_delta_gb_max'], 2)} "
                     f"| {mmm(pd, 'ns_per_query', 2)}{short(pd)} | {mmm(ps, 'ns_per_query', 2)}{short(ps)} "
                     f"| {mmm(rd_, 'ns_per_request', 1)} | {mmm(rs, 'ns_per_request', 1)} "
                     f"| {f(sp['standalone_bytes'], 0) if sp else '-'} | {f(sp['bytes_per_key'], 2) if sp else '-'} | {f(sp['bytes_per_input_key_byte'], 3) if sp else '-'} |")
        L.append('')
        if len({r['size'] for r in point if r['dataset'] == ds}) == 5:
            L.append(f'### {LABEL[ds]} point batch-size sweep (ns/query, median [min-max] of 3 process means)\n')
            L.append('| Arm / mode | ' + ' | '.join(UK_SIZES) + ' |')
            L.append('|---|' + '---:|' * len(UK_SIZES))
            for a in ARMS:
                for mode in ('direct', 'shared'):
                    cells = [next((r for r in point if r['dataset'] == ds and r['impl'] == a and r['size'] == z and r['mode'] == mode), None) for z in UK_SIZES]
                    if all(c is None for c in cells):
                        continue
                    L.append(f'| {LABEL[a]} {mode} | ' + ' | '.join(mmm(c, 'ns_per_query', 2) for c in cells) + ' |')
            L.append('')
        if any(r.get('paired') == '1' for r in rng if r['dataset'] == ds):
            if spec:
                import campaign_spec
                ks = [str(k) for k in campaign_spec.spec_for(spec, ds)['range_ks']]   # per-dataset k lists (November 1 rerun)
            else:
                ks = [str(k) for k in sorted({int(r['k']) for r in rng if r['dataset'] == ds and r.get('paired') == '1'})]
            title = 'range-length sweep' if len(ks) > 1 else f'fixed-k range (k = {ks[0]})'
            L.append(f'### {LABEL[ds]} {title} (paired trace, B = 40,000, complete-key output): ns/request, median [min-max] of 3 process means\n')
            L.append('| Arm / mode | ' + ' | '.join(f'k = {k}' for k in ks) + ' |')
            L.append('|---|' + '---:|' * len(ks))
            for a in ARMS:
                for mode in ('direct', 'shared'):
                    cells = [next((r for r in rng if r['dataset'] == ds and r['impl'] == a and r['mode'] == mode and r['k'] == k and r.get('paired') == '1'), None) for k in ks]
                    if all(c is None for c in cells):
                        continue
                    L.append(f'| {LABEL[a]} {mode} | ' + ' | '.join(mmm(c, 'ns_per_request', 1) + (f" (L {c['L']}, {f(c['keys_per_request'], 1)} keys)" if c else '') for c in cells) + ' |')
            L.append('')
            incs = rd(d, 'range_paired_increment_summary.csv')
            if incs and '1' in ks and len(ks) > 1:   # the increment needs k = 1 (plan section 4: none for a k = 100-only dataset)
                L.append('Paired increment (T(k) - T(1)) / (K(k) - K(1)) per process, then median [min-max]: the incremental end-to-end cost per additional returned key (not an iterator or record-read time).\n')
                L.append('| Arm / mode | ' + ' | '.join(f'k = {k}' for k in ks[1:]) + ' |')
                L.append('|---|' + '---:|' * (len(ks) - 1))
                for a in ARMS:
                    for mode in ('direct', 'shared'):
                        cells = [next((r for r in incs if r['dataset'] == ds and r['impl'] == a and r['mode'] == mode and r['k'] == k), None) for k in ks[1:]]
                        if all(c is None for c in cells):
                            continue
                        L.append(f'| {LABEL[a]} {mode} | ' + ' | '.join(f"{f(c['median'], 2)} [{f(c['min'], 2)}-{f(c['max'], 2)}]" if c else '-' for c in cells) + ' |')
                L.append('')
        rr = [r for r in rng if r['dataset'] == ds]
        if rr:
            L.append('Range supplementary (per request): ' + '; '.join(f"{LABEL[r['impl']]} {r['mode']}: {f(r['keys_per_request'], 2)} keys, {f(r['bytes_per_request'], 0)} B, truncated {f(float(r['truncated_per_request']) * 100 if r['truncated_per_request'] else 0, 4)} %, {f(r['ns_per_returned_key_median'], 2)} ns/key" for r in rr) + '.\n')
    if bd:
        L.append('## Range breakdown (section 7): sampled CPU ns per request, summed over the 64 workers of the profiled rounds (process 1)\n')
        L.append('| Dataset | Arm | Mode | elapsed ns/req (profiled rounds) | sampled CPU ns/req | CPU / (64 x elapsed) | setup / admin | locate LB | get keys | unassigned | samples |')
        L.append('|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|')
        for r in bd:
            L.append(f"| {LABEL.get(r['dataset'], r['dataset'])} | {LABEL.get(r['impl'], r['impl'])} | {r['mode']} | {f(r['elapsed_ns_per_request'], 1)} | {f(r['sampled_cpu_ns_per_request'], 1)} | {f(r['cpu_over_64x_elapsed'], 3)} "
                     f"| {f(r['setup_admin_ns_per_request'], 1)} ({f(r['setup_admin_share'], 1)} %) | {f(r['locate_lb_ns_per_request'], 1)} ({f(r['locate_lb_share'], 1)} %) | {f(r['get_keys_ns_per_request'], 1)} ({f(r['get_keys_share'], 1)} %) | {f(r['unassigned_ns_per_request'], 1)} ({f(r['unassigned_share'], 1)} %) | {r['samples']} |")
        L.append('')
    if cov:
        L.append('## Coverage\n')
        from collections import Counter
        L.append('Status counts: ' + ', '.join(f'{k} {v}' for k, v in sorted(Counter(c['status'] for c in cov).items())) + '.\n')
        L.append('| Dataset | Arm | Cells not ACCEPTED (status) |')
        L.append('|---|---|---|')
        for ds in DS:
            for a in ARMS:
                bad = [c for c in cov if c['dataset'] == ds and c['impl'] == a and c['status'] != 'ACCEPTED']
                if bad:
                    L.append(f"| {LABEL[ds]} | {LABEL[a]} | " + '; '.join(f"{c['cell']} ({c['status']})" for c in bad) + ' |')
        L.append('')
    open(out, 'w').write('\n'.join(L))
    print('REPORT', out, len(L), 'lines')


if __name__ == '__main__':
    main()
