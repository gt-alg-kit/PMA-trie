#!/usr/bin/env python3
"""Cross-dataset k = 100 range comparison (docs/webbase_sweep_and_full_population_k100_plan_2026-10-30.md, sections 3,
4 item 5 and 7). Explicit source selection: ONE source campaign per dataset, named in a selection manifest, each
validated against its own campaign specification (campaign_spec.py) before any of its rows is used; no pooling across
campaigns. From each source only the direct (independent) k = 100 range cells of the declared arms are taken, with
their process identity (ledger name, attempt, cell identity), binary SHA, workload label, paired-trace hashes, L,
session context and the SHA-256 of the raw log. Each process mean is recomputed from its four measured rounds; the
aggregate is the median and min / max of the three process means. A reused observation (UK / WebBase sweeps) is the
same observation as in its sweep, not an additional sample.

Selection manifest (JSON):
  {"name": ..., "k": 100, "mode": "direct", "workers": 64, "range_batch": 40000, "processes": 3,
   "arms": [...], "unsupported_arms": [...],
   "sources": [{"dataset": ..., "root": <results root>, "spec": <campaign spec JSON>,
                "session_context": "single-k session ..." | "multi-k session ...", "label": ...}, ...]}
Outputs in <out-dir>: k100_round_observations.csv, k100_process_observations.csv, k100_aggregates.csv, k100_coverage.csv,
k100_source_manifest.json (the selection, the specs, every used log with its SHA-256), k100_report.md, and
figures/k100_overview.png (when matplotlib is available).
usage: k100_assemble.py <selection.json> <out-dir>
"""
import csv
import hashlib
import json
import os
import re
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import campaign_spec          # noqa: E402
from summary_full import parse_log, LABEL   # noqa: E402

DS_ORDER = ['c2_xml_recon', 'enwiki_20261001_ns0', 'webbase_2001', 'dna_full_31mers', 'uk_2014_urls']


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(1 << 20), b''):
            h.update(b)
    return h.hexdigest()


def load_source(src, sel):
    """Accepted processes of one source that pass its specification, plus the violations and the ledger rows."""
    spec = campaign_spec.load(src['spec'])
    ds = src['dataset']
    if ds not in spec['datasets']:
        raise SystemExit(f"source {src['root']}: dataset {ds} is not in its specification {src['spec']}")
    led = os.path.join(os.path.expanduser(src['root']), ds, 'sessions', 'ledger.tsv')
    if not os.path.exists(led):   # a source that has not run yet: every cell of the dataset is MISSING, never an error
        return spec, {}, {}
    rows = list(csv.DictReader(open(led), delimiter='\t'))
    last = {}
    for r in rows:
        last[r['name']] = r
    accepted, violations = {}, {}
    for name, row in last.items():
        m = re.fullmatch(r'S__(.+)__(.+)__p(\d+)', name)
        if not m or m.group(2) not in sel['arms'] or m.group(2) not in spec['arms']:
            continue
        log = os.path.join(os.path.expanduser(src['root']), ds, 'sessions', 'raw', name + '.log')
        if row['verdict'] != 'ACCEPTED' or not os.path.exists(log):
            continue
        r = parse_log(log)
        if r['status'] != 'OK':
            continue
        r['ledger'] = row
        key = (ds, m.group(2), int(m.group(3)))
        v = campaign_spec.check_process(spec, *key, r)
        if v:
            violations[key] = v
        else:
            accepted[key] = r
    for key, v in campaign_spec.trace_violations(spec, ds, accepted).items():
        violations.setdefault(key, []).extend(v)
        accepted.pop(key, None)
    return spec, accepted, violations


def main():
    sel = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    os.makedirs(out, exist_ok=True)
    k, mode = str(sel['k']), sel['mode']
    rounds_out, procs_out, aggs, cov, used = [], [], [], [], []
    datasets = [s['dataset'] for s in sel['sources']]
    if len(set(datasets)) != len(datasets):
        raise SystemExit('one source per dataset: the selection names a dataset twice (no pooling across campaigns)')
    all_violations = {}
    for src in sorted(sel['sources'], key=lambda s: DS_ORDER.index(s['dataset'])):
        ds = src['dataset']
        spec, accepted, violations = load_source(src, sel)
        all_violations.update(violations)
        for impl in sel['arms']:
            obs = []
            for p in range(1, sel['processes'] + 1):
                r = accepted.get((ds, impl, p))
                if r is None:
                    continue
                rr = [d for d in r['range_rounds'] if d['k'] == k and d['mode'] == mode]
                bt = [d for d in r['range_batches'] if d['k'] == k and d['mode'] == mode]
                env = r['env']
                if len(rr) != spec['measured_rounds'] or env.get('workers') != str(sel['workers']):
                    all_violations.setdefault((ds, impl, p), []).append(f'k{k}_{mode}_rounds={len(rr)}_workers={env.get("workers")}')
                    continue
                for d in rr:
                    rounds_out.append({'dataset': ds, 'impl': impl, 'process': p, 'round': d['round'], 'L': d['L'], 'br': d['br'],
                                       'requests': d['requests'], 'ns': d['ns'], 'ns_per_request': d['ns_per_request'],
                                       'keys': d['keys'], 'bytes': d['bytes'], 'source': src.get('label', src['root'])})
                req = sum(float(d['requests']) for d in rr)
                mean = statistics.mean(float(d['ns_per_request']) for d in rr)
                trunc = sum(int(d['truncated']) for d in bt)
                breq = sum(int(d['br']) for d in bt)
                logp = r['path']
                o = {'dataset': ds, 'impl': impl, 'process': p, 'ns_per_request_mean': mean,
                     'keys_per_request': sum(float(d['keys']) for d in rr) / req, 'bytes_per_request': sum(float(d['bytes']) for d in rr) / req,
                     'truncated_requests': trunc, 'truncated_fraction': trunc / breq if breq else '', 'L': rr[0]['L'], 'br': rr[0]['br'],
                     'workers': env.get('workers'), 'cpus': env.get('cpus'), 'paired': rr[0]['paired'], 'workload': rr[0]['workload'],
                     'starts_ids_hash': rr[0]['starts_ids_hash'], 'starts_bytes_hash': rr[0]['starts_bytes_hash'],
                     'binary_sha16': r['cell_run'].get('binary_sha'), 'ledger_ident': r['ledger']['ident'], 'attempt': r['ledger']['attempt'],
                     'ledger_reasons': r['ledger']['reasons'], 'session_ks': r['plan'].get('range_ks'), 'session_query_modes': r['plan'].get('query_modes', 'all'),
                     'session_context': src['session_context'], 'source_root': src['root'], 'source_spec': os.path.basename(src['spec']),
                     'log': os.path.relpath(logp, os.path.expanduser(src['root'])), 'log_sha256': sha256(logp)}
                obs.append(o)
                procs_out.append(o)
                used.append({'dataset': ds, 'impl': impl, 'process': p, 'log': o['log'], 'log_sha256': o['log_sha256'],
                             'ledger_ident': o['ledger_ident'], 'binary_sha16': o['binary_sha16']})
            vals = [o['ns_per_request_mean'] for o in obs]
            complete = len(vals) == sel['processes']
            a = {'dataset': ds, 'impl': impl, 'k': k, 'mode': mode, 'n_processes': len(vals),
                 'median_ns_per_request': statistics.median(vals) if vals else '', 'min_ns_per_request': min(vals) if vals else '',
                 'max_ns_per_request': max(vals) if vals else '',
                 'keys_per_request': statistics.mean(o['keys_per_request'] for o in obs) if obs else '',
                 'bytes_per_request': statistics.mean(o['bytes_per_request'] for o in obs) if obs else '',
                 'truncated_fraction': statistics.mean(o['truncated_fraction'] for o in obs if o['truncated_fraction'] != '') if obs else '',
                 'L': obs[0]['L'] if obs else '', 'session_context': src['session_context'], 'source': src.get('label', src['root']),
                 'binary_sha16': ','.join(sorted({o['binary_sha16'] for o in obs})), 'workload': ','.join(sorted({o['workload'] for o in obs})),
                 'status': 'ACCEPTED' if complete else ('PARTIAL' if vals else 'MISSING')}
            aggs.append(a)
            note = '; '.join(f"p{p}: {','.join(all_violations[(ds, impl, p)])}" for p in range(1, sel['processes'] + 1) if (ds, impl, p) in all_violations)
            cov.append({'dataset': ds, 'impl': impl, 'cell': f'range:k{k}:{mode}', 'status': a['status'], 'n_processes': len(vals), 'note': note})
            if impl in campaign_spec.SHARED:
                cov.append({'dataset': ds, 'impl': impl, 'cell': f'range:k{k}:shared', 'status': 'OUT_OF_SCOPE', 'n_processes': 0, 'note': 'independent queries only'})
        for impl in sel.get('unsupported_arms', []):
            cov.append({'dataset': ds, 'impl': impl, 'cell': f'range:k{k}:{mode}', 'status': 'UNSUPPORTED_API', 'n_processes': 0, 'note': 'no ordered scan'})
    def w(name, cols, rows):
        with open(os.path.join(out, name), 'w', newline='') as f:
            wr = csv.DictWriter(f, fieldnames=cols, extrasaction='ignore')
            wr.writeheader()
            for r in rows:
                wr.writerow(r)
    w('k100_round_observations.csv', list(rounds_out[0].keys()) if rounds_out else ['dataset'], rounds_out)
    w('k100_process_observations.csv', list(procs_out[0].keys()) if procs_out else ['dataset'], procs_out)
    w('k100_aggregates.csv', list(aggs[0].keys()) if aggs else ['dataset'], aggs)
    w('k100_coverage.csv', ['dataset', 'impl', 'cell', 'status', 'n_processes', 'note'], cov)
    manifest = {'selection': sel, 'specs': {s['dataset']: json.load(open(s['spec'])) for s in sel['sources']}, 'used_observations': used,
                'violations': {f'{k_[0]}/{k_[1]}/p{k_[2]}': v for k_, v in all_violations.items()}}
    json.dump(manifest, open(os.path.join(out, 'k100_source_manifest.json'), 'w'), indent=1)
    n_ok = sum(1 for a in aggs if a['status'] == 'ACCEPTED')
    complete = n_ok == len(aggs) == len(sel['sources']) * len(sel['arms'])
    print(f"K100_ASSEMBLE aggregates_accepted={n_ok}/{len(aggs)} process_observations={len(procs_out)} rounds={len(rounds_out)} "
          f"violations={len(all_violations)} complete={int(complete)} out={out}")
    report(out, sel, aggs, procs_out)
    try:
        figure(out, sel, aggs)
    except Exception as e:
        print(f'FIGURE_SKIPPED {type(e).__name__}: {e}')
    return 0 if complete else 1


def fmt(v, nd=1):
    try:
        return f'{float(v):,.{nd}f}'
    except (TypeError, ValueError):
        return '-'


CAPTION = ('Full-key range queries at k = 100 on the full populations (S = N), hit lower bounds, complete-key output, '
           'B = 40,000 independent requests per batch on 64 workers (64 physical cores), one paired lower-bound batch per '
           'dataset replayed L times per round; three independent processes with one build each, 2 warm-up + 4 measured '
           'rounds per cell; median of the three process means, whiskers min-max. Throughput-derived cost per request, not '
           'single-request latency. ')


def report(out, sel, aggs, procs):
    L = ['# Full-population k = 100 range comparison (independent queries, 64 workers)\n', CAPTION + '\n']
    L.append('| Dataset | Source / session context | ' + ' | '.join(LABEL.get(a, a) for a in sel['arms']) + ' |')
    L.append('|---|---|' + '---:|' * len(sel['arms']))
    for ds in DS_ORDER:
        rows = {a['impl']: a for a in aggs if a['dataset'] == ds}
        if not rows:
            continue
        ctx = next(iter(rows.values()))['session_context']
        L.append(f"| {LABEL.get(ds, ds)} | {ctx} | " + ' | '.join(
            f"{fmt(rows[i]['median_ns_per_request'])} [{fmt(rows[i]['min_ns_per_request'])}-{fmt(rows[i]['max_ns_per_request'])}] (n={rows[i]['n_processes']})"
            if i in rows and rows[i]['n_processes'] else 'MISSING' for i in sel['arms']) + ' |')
    L.append('\nns per range request, median [min-max] of the process means (n = accepted processes). '
             + ', '.join(LABEL.get(a, a) for a in sel.get('unsupported_arms', [])) + ': no ordered scan (UNSUPPORTED_API).\n')
    L.append('| Dataset | Arm | keys / request | bytes / request | truncated requests | L | binary | workload |')
    L.append('|---|---|---:|---:|---:|---:|---|---|')
    for a in aggs:
        L.append(f"| {LABEL.get(a['dataset'], a['dataset'])} | {LABEL.get(a['impl'], a['impl'])} | {fmt(a['keys_per_request'], 2)} | {fmt(a['bytes_per_request'], 0)} | "
                 f"{fmt(float(a['truncated_fraction']) * 100 if a['truncated_fraction'] != '' else '', 4)} % | {a['L']} | {a['binary_sha16']} | {a['workload']} |")
    open(os.path.join(out, 'k100_report.md'), 'w').write('\n'.join(L) + '\n')


def figure(out, sel, aggs):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dss = [d for d in DS_ORDER if any(a['dataset'] == d for a in aggs)]
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    width = 0.8 / len(sel['arms'])
    fig, ax = plt.subplots(figsize=(12, 5))
    for i, impl in enumerate(sel['arms']):
        xs, ys, lo, hi, hatch = [], [], [], [], []
        for j, ds in enumerate(dss):
            a = next((a for a in aggs if a['dataset'] == ds and a['impl'] == impl and a['n_processes']), None)
            if not a:
                continue
            xs.append(j + i * width); ys.append(float(a['median_ns_per_request']))
            lo.append(ys[-1] - float(a['min_ns_per_request'])); hi.append(float(a['max_ns_per_request']) - ys[-1])
            hatch.append('//' if a['session_context'].startswith('multi-k') else '')
        bars = ax.bar(xs, ys, width, yerr=[lo, hi], capsize=2, label=LABEL.get(impl, impl), color=colors[i % len(colors)])
        for b, h in zip(bars, hatch):
            b.set_hatch(h)
    ax.set_xticks([j + 0.4 - width / 2 for j in range(len(dss))])
    ax.set_xticklabels([LABEL.get(d, d) for d in dss])
    ax.set_yscale('log'); ax.set_ylabel('ns per range request (64 workers)')
    ax.set_title('Full-population k = 100 range queries, independent requests (hatched: k = 100 taken from a multi-k sweep session)', fontsize=9)
    ax.legend(fontsize=8)
    fig.text(0.01, 0.005, CAPTION, fontsize=6, wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    os.makedirs(os.path.join(out, 'figures'), exist_ok=True)
    fig.savefig(os.path.join(out, 'figures', 'k100_overview.png'), dpi=150)
    open(os.path.join(out, 'figures', 'k100_overview_caption.txt'), 'w').write(CAPTION + 'Unsupported (no ordered scan): '
                                                                             + ', '.join(LABEL.get(a, a) for a in sel.get('unsupported_arms', [])) + '.\n')
    print('FIGURE', os.path.join(out, 'figures', 'k100_overview.png'))


if __name__ == '__main__':
    sys.exit(main())
